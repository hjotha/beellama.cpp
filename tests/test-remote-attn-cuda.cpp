// Focused GPU contract tests for the remote-attention wire codec.
#include "ggml-backend.h"
#include "ggml-remote-attn-cuda.h"
#include "ggml-remote-attn-trace.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#define RKVA_HAVE_CUDA_FP8_ORACLE 1
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#ifdef RKVA_CUDA_STANDALONE_TEST
extern "C" const ggml_remote_attn_cuda_fp8_interface_v2 * ggml_backend_cuda_remote_attn_fp8_interface_v2(void);
extern "C" const ggml_remote_attn_cuda_fp8_interface * ggml_backend_cuda_remote_attn_fp8_interface(void);
#endif

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
#define CUDA_OK(x) REQUIRE((x) == cudaSuccess)

static float from_bits(uint32_t bits) { float value; std::memcpy(&value, &bits, 4); return value; }
static uint32_t bits_of(float value) { uint32_t bits; std::memcpy(&bits, &value, 4); return bits; }
static float half_float(uint16_t bits) { half h; std::memcpy(&h, &bits, 2); return __half2float(h); }
static uint8_t oracle(float value) {
#ifdef RKVA_HAVE_CUDA_FP8_ORACLE
    return uint8_t(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E5M2));
#else
    // Older toolkits can still validate the software codec: search the exact
    // positive E5M2 lattice in double, with an even-code tie break.
    const double magnitude = std::fabs(double(value));
    if (magnitude >= 57344.0) return uint8_t(0x7b | ((bits_of(value) >> 24) & 0x80));
    uint8_t best = 0;
    double best_distance = magnitude;
    for (unsigned b = 1; b <= 0x7b; ++b) {
        const double distance = std::fabs(magnitude - double(half_float(uint16_t(b << 8))));
        if (distance < best_distance || (distance == best_distance && !(b & 1))) {
            best = uint8_t(b);
            best_distance = distance;
        }
    }
    return uint8_t(best | ((bits_of(value) >> 24) & 0x80));
#endif
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
        std::puts("SKIP: CUDA device unavailable");
        return 77;
    }
#ifdef RKVA_CUDA_STANDALONE_TEST
    const auto * api = ggml_backend_cuda_remote_attn_fp8_interface_v2();
    const auto * legacy = ggml_backend_cuda_remote_attn_fp8_interface();
#else
    ggml_backend_load_all();
    auto reg = ggml_backend_reg_by_name("CUDA");
    REQUIRE(reg);
    const auto getter = reinterpret_cast<ggml_backend_cuda_remote_attn_fp8_interface_v2_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_remote_attn_fp8_interface_v2"));
    const auto legacy_getter = reinterpret_cast<ggml_backend_cuda_remote_attn_fp8_interface_v1_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_remote_attn_fp8_interface_v1"));
    REQUIRE(getter && legacy_getter);
    const auto * api = getter();
    const auto * legacy = legacy_getter();
#endif
    REQUIRE(api && legacy);
    void * context = api->create_context(0);
    REQUIRE(context);
    std::vector<uint16_t> finite_half;
    for (uint32_t h = 0; h < 65536; ++h) if ((h & 0x7c00u) != 0x7c00u) finite_half.push_back(uint16_t(h));
    REQUIRE(finite_half.size() == 63488);
    auto * packed = static_cast<uint8_t *>(api->alloc_pinned(2u << 20));
    REQUIRE(packed);
    half * dev_half = nullptr;
    CUDA_OK(cudaMalloc(&dev_half, finite_half.size() * 2));
    CUDA_OK(cudaMemcpy(dev_half, finite_half.data(), finite_half.size() * 2, cudaMemcpyHostToDevice));
    REQUIRE(api->device_for_pointer(dev_half) == 0);
    REQUIRE(api->device_for_pointer(finite_half.data()) == -1);
    for (const void * source : {static_cast<const void *>(finite_half.data()), static_cast<const void *>(dev_half)}) {
        REQUIRE(api->pack_q(context, source, true, finite_half.size(), packed, nullptr));
        for (size_t i = 0; i < finite_half.size(); ++i) REQUIRE(packed[i] == oracle(half_float(finite_half[i])));
    }
    std::printf("all finite F16: %zu patterns x host/device passed\n", finite_half.size());

    std::vector<float> finite_float{0.0f, -0.0f, std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
    for (uint32_t b = 0; b < 0x7b; ++b) {
        const float mid = (half_float(uint16_t(b << 8)) + half_float(uint16_t((b + 1) << 8))) * 0.5f;
        for (float value : {mid, std::nextafter(mid, 0.0f), std::nextafter(mid, INFINITY)}) {
            finite_float.push_back(value); finite_float.push_back(-value);
        }
    }
    uint32_t random = 0x12345678;
    for (size_t i = 0; i < 1u << 20; ++i) {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        if ((random & 0x7f800000u) != 0x7f800000u) finite_float.push_back(from_bits(random));
    }
    float * dev_float = nullptr;
    CUDA_OK(cudaMalloc(&dev_float, finite_float.size() * 4));
    cudaStream_t producer;
    cudaEvent_t ready;
    CUDA_OK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    CUDA_OK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    // Test event ordering without a host synchronize between producer and pack.
    CUDA_OK(cudaMemcpyAsync(dev_float, finite_float.data(), finite_float.size() * 4, cudaMemcpyHostToDevice, producer));
    CUDA_OK(cudaEventRecord(ready, producer));
    REQUIRE(api->pack_q(context, dev_float, false, finite_float.size(), packed, ready));
    for (size_t i = 0; i < finite_float.size(); ++i) REQUIRE(packed[i] == oracle(finite_float[i]));
    REQUIRE(api->pack_q(context, finite_float.data(), false, finite_float.size(), packed, nullptr));
    for (size_t i = 0; i < finite_float.size(); ++i) REQUIRE(packed[i] == oracle(finite_float[i]));
    std::printf("F32 oracle: %zu patterns x host/device passed\n", finite_float.size());
    for (uint32_t bad : {0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00001u, 0x7f800001u}) {
        float value = from_bits(bad);
        REQUIRE(!api->pack_q(context, &value, false, 1, packed, nullptr));
    }
    float one = 1.0f;
    REQUIRE(api->pack_q(context, &one, false, 1, packed, nullptr) && packed[0] == 0x3c);

    std::vector<uint8_t> codes;
    for (uint32_t b = 0; b < 256; ++b) if ((b & 0x7cu) != 0x7cu) codes.push_back(uint8_t(b));
    REQUIRE(codes.size() == 248);
    std::copy(codes.begin(), codes.end(), packed);
    float lse = 3.25f;
    float * dev_out = nullptr;
    CUDA_OK(cudaMalloc(&dev_out, 1024 * 4));
    REQUIRE(api->unpack_out(context, packed, &lse, 1, codes.size(), 1, dev_out, 249 * 4));
    std::vector<float> out(249);
    CUDA_OK(cudaMemcpy(out.data(), dev_out, out.size() * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < codes.size(); ++i) REQUIRE(bits_of(out[i]) == bits_of(half_float(uint16_t(codes[i]) << 8)));
    REQUIRE(out.back() == lse);
    for (uint32_t b = 0; b < 256; ++b) if ((b & 0x7cu) == 0x7cu) {
        packed[0] = uint8_t(b);
        REQUIRE(!api->unpack_out(context, packed, nullptr, 1, 1, 0, dev_out, 4));
    }
    // Former alignment defect: 6 FP8 bytes precede a float LSE region. Both
    // destinations have padding which must remain untouched.
    for (size_t i = 0; i < 6; ++i) packed[i] = 0x3c;
    float ls[2] = {1.5f, 2.5f};
    std::vector<float> padded(16, -99.0f);
    CUDA_OK(cudaMemcpy(dev_out, padded.data(), padded.size() * 4, cudaMemcpyHostToDevice));
    REQUIRE(api->unpack_out(context, packed, ls, 2, 3, 1, dev_out, 8 * 4));
    CUDA_OK(cudaMemcpy(padded.data(), dev_out, padded.size() * 4, cudaMemcpyDeviceToHost));
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 3; ++col) REQUIRE(padded[row * 8 + col] == 1.0f);
        REQUIRE(padded[row * 8 + 3] == ls[row]);
        for (size_t col = 4; col < 8; ++col) REQUIRE(padded[row * 8 + col] == -99.0f);
    }
    std::fill(padded.begin(), padded.end(), -99.0f);
    REQUIRE(api->unpack_out(context, packed, ls, 2, 3, 1, padded.data(), 8 * 4));
    REQUIRE(padded[3] == ls[0] && padded[11] == ls[1] && padded[4] == -99.0f);
    REQUIRE(!api->unpack_out(context, packed, nullptr, 1, 3, 1, dev_out, 16));
    REQUIRE(!api->unpack_out(context, packed, &lse, 1, 3, 1, dev_out, 15));
    REQUIRE(!api->unpack_out(context, packed, &lse, 1, 3, 1, dev_out, 12));
    REQUIRE(!api->unpack_out(context, packed, &lse, SIZE_MAX, 3, 1, dev_out, 16));
    REQUIRE(!api->pack_q(context, dev_float, false, SIZE_MAX, packed, nullptr));
    for (float bad : {INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
        REQUIRE(!api->unpack_out(context, packed, &bad, 1, 3, 1, dev_out, 16));
    }
    REQUIRE(api->unpack_out(context, packed, &lse, 1, 3, 1, dev_out, 16));

    // Separate slots execute concurrently without sharing scratch/error flags.
    void * other = api->create_context(0);
    auto * other_packed = static_cast<uint8_t *>(api->alloc_pinned(1024));
    REQUIRE(other && other_packed);
    std::thread thread([&] {
        std::vector<float> values(1024, -2.0f);
        for (int i = 0; i < 20; ++i) {
            REQUIRE(api->pack_q(other, values.data(), false, values.size(), other_packed, nullptr));
            REQUIRE(std::all_of(other_packed, other_packed + 1024, [](uint8_t b) { return b == 0xc0; }));
        }
    });
    for (int i = 0; i < 20; ++i) REQUIRE(api->pack_q(context, &one, false, 1, packed, nullptr));
    thread.join();
    api->destroy_context(other);
    api->free_pinned(other_packed);
    // Legacy ABI retains its synchronous contract and optional completion event.
    REQUIRE(legacy->pack_q(&one, false, 1, packed, nullptr, ready));
    REQUIRE(legacy->event_sync(ready) && packed[0] == 0x3c);
    CUDA_OK(cudaEventDestroy(ready));
    CUDA_OK(cudaStreamDestroy(producer));
    CUDA_OK(cudaFree(dev_float));
    CUDA_OK(cudaFree(dev_half));
    CUDA_OK(cudaFree(dev_out));
    api->destroy_context(context);
    api->free_pinned(packed);
    std::puts("FP8 decode, odd layouts, stride/overflow/nonfinite checks, producer event and two slots passed");
    ggml_remote_attn_trace_flush();
    return 0;
}
