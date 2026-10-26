// Manual CUDA hardware contract: lossless binary-F16 transport, never mask semantics.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)
#define CUDA_OK(x) REQUIRE((x) == cudaSuccess)
using stats_fn = bool (*)(ggml_backend_buffer_t, uint64_t *, size_t);
using stats = std::array<uint64_t, 7>;
enum { ELIGIBLE, PACKED, RAW_BYTES, PAYLOAD_BYTES, DISABLED, NONBINARY, UNAVAILABLE };

static void mode(const char * value) {
#ifdef _WIN32
    REQUIRE(_putenv_s("GGML_CUDA_MASK_PACK", value ? value : "") == 0);
#else
    REQUIRE(value ? setenv("GGML_CUDA_MASK_PACK", value, 1) == 0 : unsetenv("GGML_CUDA_MASK_PACK") == 0);
#endif
}
static void mark(ggml_tensor * t) { t->flags |= GGML_TENSOR_FLAG_BINARY_MASK; }
static std::vector<uint16_t> pattern(size_t n, unsigned seed) {
    static const uint32_t words[] = {0, 0xffffffffu, 0x55555555u, 0xaaaaaaaau,
                                    1, 0x80000000u, 0xdeadbeefu, 0x01020408u};
    std::vector<uint16_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        uint32_t word = words[((i / 32) + seed) % (sizeof(words) / sizeof(words[0]))];
        out[i] = (word & (uint32_t(1) << (i % 32))) ? 0xfc00 : 0;
    }
    return out;
}
static void roundtrip(ggml_tensor * t, std::vector<uint16_t> input) {
    REQUIRE(input.size() * sizeof(uint16_t) == ggml_nbytes(t));
    const auto expected = input;
    ggml_backend_tensor_set(t, input.data(), 0, ggml_nbytes(t));
    // tensor_set must retire mapped CPU staging before returning to its caller.
    std::fill(input.begin(), input.end(), uint16_t(0x7e01));
    std::vector<uint16_t> got(expected.size());
    ggml_backend_tensor_get(t, got.data(), 0, ggml_nbytes(t));
    REQUIRE(std::memcmp(got.data(), expected.data(), ggml_nbytes(t)) == 0);
}
static stats read_stats(stats_fn fn, ggml_backend_buffer_t buffer) {
    stats result {};
    REQUIRE(fn && fn(buffer, result.data(), result.size()));
    return result;
}
static void require_packed(stats_fn fn, ggml_backend_buffer_t buffer,
                           ggml_tensor * t, std::vector<uint16_t> input) {
    const auto before = read_stats(fn, buffer);
    roundtrip(t, std::move(input));
    const auto after = read_stats(fn, buffer);
    REQUIRE(after[PACKED] == before[PACKED] + 1);
    REQUIRE(after[RAW_BYTES] == before[RAW_BYTES] + ggml_nbytes(t));
    const size_t payload = ((ggml_nelements(t) + 31) / 32) * sizeof(uint32_t);
    REQUIRE(after[PAYLOAD_BYTES] == before[PAYLOAD_BYTES] + payload);
    REQUIRE(after[UNAVAILABLE] == before[UNAVAILABLE]);
}
static void guards(ggml_tensor * base, size_t n) {
    std::array<uint16_t, 7> first {}, last {};
    ggml_backend_tensor_get(base, first.data(), 0, sizeof(first));
    ggml_backend_tensor_get(base, last.data(), (n + 7) * 2, sizeof(last));
    for (auto v : first) REQUIRE(v == 0x3555);
    for (auto v : last) REQUIRE(v == 0x3555);
}

int main() {
    const char * inherited = std::getenv("GGML_CUDA_MASK_PACK");
    const bool had_mode = inherited != nullptr;
    const std::string old_mode = inherited ? inherited : "";
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    REQUIRE(backend);
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto fn = reinterpret_cast<stats_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_mask_pack_stats"));
    REQUIRE(fn); // A silent raw-copy fallback cannot qualify this test.
    ggml_init_params params {ggml_tensor_overhead() * 24, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx);
    constexpr size_t n = 513 * 128, odd_n = 4097 * 17;
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 513, 128); mark(mask);
    auto * odd = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4097, 17); mark(odd);
    auto * unmarked = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 513, 128);
    auto * f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 513, 128); mark(f32);
    auto * n1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 70000, 1); mark(n1);
    auto * small = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 17, 16); mark(small);
    auto * base_a = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n + 14);
    auto * base_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, odd_n + 14);
    auto * view_a = ggml_view_2d(ctx, base_a, 513, 128, 513 * 2, 14); mark(view_a);
    auto * view_b = ggml_view_2d(ctx, base_b, 4097, 17, 4097 * 2, 14); mark(view_b);
    auto * odd_byte_base = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n + 1);
    auto * odd_byte_view = ggml_view_2d(ctx, odd_byte_base, 513, 128, 513 * 2, 1); mark(odd_byte_view);
    auto * strided_base = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 514 * 128);
    auto * strided = ggml_view_2d(ctx, strided_base, 513, 128, 514 * 2, 0); mark(strided);
    REQUIRE(!ggml_is_contiguous(strided));
    // 128 MiB + 64 B of raw F16 requires an 8 MiB + 4 B packed sidecar.
    auto * huge = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, (1 << 22) + 2, 16); mark(huge);
    auto * rkva_large = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 102400, 256); mark(rkva_large);
    auto * maximum = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 131072, 512); mark(maximum);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer && mask->buffer == buffer && base_a->buffer == buffer && base_b->buffer == buffer);
    roundtrip(base_a, std::vector<uint16_t>(n + 14, 0x3555));
    roundtrip(base_b, std::vector<uint16_t>(odd_n + 14, 0x3555));
    roundtrip(odd_byte_base, std::vector<uint16_t>(n + 1, 0xa5a5));
    REQUIRE((reinterpret_cast<uintptr_t>(odd_byte_view->data) & 1u) == 1u);

    mode("0");
    auto off_before = read_stats(fn, buffer);
    roundtrip(mask, pattern(n, 0)); // Classic OFF baseline: no mapped staging/kernel initialization.
    auto off_after = read_stats(fn, buffer);
    REQUIRE(off_after[PACKED] == off_before[PACKED]);
    REQUIRE(off_after[DISABLED] == off_before[DISABLED] + 1);
    CUDA_OK(cudaDeviceSynchronize());
    size_t free_off = 0, total_off = 0, free_on = 0, total_on = 0;
    CUDA_OK(cudaMemGetInfo(&free_off, &total_off));
    mode("1");
    require_packed(fn, buffer, mask, pattern(n, 0));
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemGetInfo(&free_on, &total_on));
    std::printf("mask-pack device memory after OFF raw: free=%zu total=%zu; after ON same storage: free=%zu total=%zu\n",
                free_off, total_off, free_on, total_on);
    // Report driver/module bookkeeping rather than asserting byte-identical VRAM.
    require_packed(fn, buffer, mask, std::vector<uint16_t>(n, 0));
    require_packed(fn, buffer, mask, std::vector<uint16_t>(n, 0xfc00));
    require_packed(fn, buffer, mask, pattern(n, 3));
    uint32_t rng = 0x193256u;
    auto random = pattern(n, 0);
    for (auto & x : random) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; x = (rng & 1) ? 0xfc00 : 0; }
    require_packed(fn, buffer, mask, std::move(random));
    require_packed(fn, buffer, odd, pattern(odd_n, 7)); // Partial word and final 1-half store.
    require_packed(fn, buffer, view_a, pattern(n, 2)); // Destination is only half-aligned (offset14 B).
    require_packed(fn, buffer, view_b, pattern(odd_n, 5));
    guards(base_a, n); guards(base_b, odd_n);
    mode(nullptr); // Unset must select the enabled default, not a raw path.
    require_packed(fn, buffer, mask, pattern(n, 4));
    mode("1");
    require_packed(fn, buffer, rkva_large, pattern(size_t(ggml_nelements(rkva_large)), 5));
    require_packed(fn, buffer, maximum, pattern(size_t(ggml_nelements(maximum)), 7));

    for (auto * t : {unmarked, f32, n1, small, strided, odd_byte_view, huge}) {
        const auto before = read_stats(fn, buffer);
        roundtrip(t, pattern(ggml_nbytes(t) / 2, 1));
        const auto after = read_stats(fn, buffer);
        REQUIRE(after[PACKED] == before[PACKED]);
        REQUIRE(after[ELIGIBLE] == before[ELIGIBLE]);
    }
    // Odd byte destinations cannot use uint16 CUDA stores. The raw fallback
    // keeps the whole binary-F16 payload and both single-byte guards intact.
    auto odd_byte_payload = pattern(n, 1);
    std::vector<uint8_t> odd_byte_expected((n + 1) * 2, 0xa5), odd_byte_got((n + 1) * 2);
    std::memcpy(odd_byte_expected.data() + 1, odd_byte_payload.data(), n * 2);
    ggml_backend_tensor_get(odd_byte_base, odd_byte_got.data(), 0, odd_byte_got.size());
    REQUIRE(odd_byte_got == odd_byte_expected);
    REQUIRE(odd_byte_got.front() == 0xa5 && odd_byte_got.back() == 0xa5);
    // Partial copies bypass transport; preserve the first half's exact bits.
    roundtrip(mask, std::vector<uint16_t>(n, 0x3555));
    const auto partial_before = read_stats(fn, buffer);
    auto partial = pattern(n, 6); partial[0] = 0x3555;
    ggml_backend_tensor_set(mask, partial.data() + 1, 2, (n - 1) * 2);
    std::vector<uint16_t> partial_got(n);
    ggml_backend_tensor_get(mask, partial_got.data(), 0, n * 2);
    REQUIRE(partial_got == partial);
    REQUIRE(read_stats(fn, buffer)[PACKED] == partial_before[PACKED]);
    REQUIRE(read_stats(fn, buffer)[ELIGIBLE] == partial_before[ELIGIBLE]);

    // No semantic rounding: -0, finite, +inf and both NaN signs retain raw bits.
    for (uint16_t invalid : {uint16_t(0x8000), uint16_t(0x3c00), uint16_t(0x7c00),
                             uint16_t(0x7e01), uint16_t(0xfe01)}) {
        for (size_t where : {size_t(0), n / 2, n - 1}) {
            auto input = pattern(n, 3); input[where] = invalid;
            const auto before = read_stats(fn, buffer);
            roundtrip(mask, std::move(input));
            const auto after = read_stats(fn, buffer);
            REQUIRE(after[PACKED] == before[PACKED]);
            REQUIRE(after[NONBINARY] == before[NONBINARY] + 1);
        }
    }
    const auto parallel_before = read_stats(fn, buffer);
    auto worker = [&](ggml_tensor * t, unsigned seed) {
        CUDA_OK(cudaSetDevice(0));
        for (unsigned i = 0; i < 16; ++i) roundtrip(t, pattern(ggml_nelements(t), seed + i));
    };
    // Shared buffer owns one mapped staging pair and its guard, independent CPU sources.
    // Environment is fixed before either thread starts and never changed concurrently.
    std::thread a(worker, view_a, 0), b(worker, view_b, 11);
    a.join(); b.join();
    const auto final = read_stats(fn, buffer);
    REQUIRE(final[PACKED] == parallel_before[PACKED] + 32);
    REQUIRE(final[UNAVAILABLE] == parallel_before[UNAVAILABLE]);
    guards(base_a, n); guards(base_b, odd_n);
    std::printf("mask-pack CUDA PASS eligible=%llu packed=%llu raw_bytes=%llu payload_bytes=%llu disabled=%llu nonbinary=%llu unavailable=%llu\n",
        (unsigned long long) final[ELIGIBLE], (unsigned long long) final[PACKED],
        (unsigned long long) final[RAW_BYTES], (unsigned long long) final[PAYLOAD_BYTES],
        (unsigned long long) final[DISABLED], (unsigned long long) final[NONBINARY],
        (unsigned long long) final[UNAVAILABLE]);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    mode(had_mode ? old_mode.c_str() : nullptr);
}
