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
extern "C" const ggml_remote_attn_cuda_fp8_interface_v3 * ggml_backend_cuda_remote_attn_fp8_interface_v3(void);
extern "C" const ggml_remote_attn_cuda_retirement_interface_v1 * ggml_backend_cuda_remote_attn_retirement_interface_v1(void);
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

static void test_mapped_small(const ggml_remote_attn_cuda_fp8_interface_v3 * small,
                              const ggml_remote_attn_cuda_retirement_interface_v1 * retirement,
                              const ggml_remote_attn_cuda_fp8_interface_v2 * api, void * context,
                              half * dev_half, float * dev_float, uint8_t * packed,
                              const std::vector<uint16_t> & half_values, const std::vector<float> & float_values,
                              cudaStream_t producer, cudaEvent_t ready) {
    REQUIRE(small && small->base == api && small->supports_small && small->pack_small && small->unpack_small);
    REQUIRE(!small->supports_small(-1));
    // This fixture is a qualification gate for the mapped path, not a silent
    // success when a present device cannot execute the new implementation.
    const bool supported = small->supports_small(0);
    int can_map = 0, unified_addressing = 0;
    unsigned device_flags = 0;
    CUDA_OK(cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, 0));
    CUDA_OK(cudaDeviceGetAttribute(&unified_addressing, cudaDevAttrUnifiedAddressing, 0));
    CUDA_OK(cudaGetDeviceFlags(&device_flags));
    std::printf("mapped small v3 capabilities: device=0 supported=%d can_map=%d uva=%d flags=0x%x\n",
        int(supported), can_map, unified_addressing, device_flags);
    REQUIRE(supported);
    REQUIRE(small->pack_small(context, dev_half, true, 1, half_values.size(), packed, nullptr));
    for (size_t i = 0; i < half_values.size(); ++i) REQUIRE(packed[i] == oracle(half_float(half_values[i])));
    CUDA_OK(cudaMemcpyAsync(dev_float, float_values.data(), float_values.size() * sizeof(float), cudaMemcpyHostToDevice, producer));
    CUDA_OK(cudaEventRecord(ready, producer));
    REQUIRE(small->pack_small(context, dev_float, false, 1, float_values.size(), packed, ready));
    for (size_t i = 0; i < float_values.size(); ++i) REQUIRE(packed[i] == oracle(float_values[i]));

    float * output = nullptr;
    constexpr size_t max_rows = 9, max_width = 2049, heads = 3, padding = 7;
    constexpr size_t capacity = max_rows * (max_width + heads + padding);
    CUDA_OK(cudaMalloc(&output, capacity * sizeof(float)));
    std::vector<float> readback(capacity);
    std::vector<float> values(max_rows * max_width);
    std::vector<float> lse(max_rows * heads);
    std::vector<uint8_t> codes;
    for (uint32_t code = 0; code < 256; ++code) if ((code & 0x7cu) != 0x7cu) codes.push_back(uint8_t(code));

    // Grow/shrink each sidecar on the same context, covering 1/9 query bounds,
    // partial words, odd rows, inter-row word crossings, and more than one CTA.
    for (size_t nq : {size_t(1), size_t(9), size_t(2), size_t(9), size_t(1)}) {
        for (size_t width : {size_t(1), size_t(3), size_t(17), size_t(255), max_width}) {
            const size_t n = nq * width;
            const size_t stride = width + heads + padding;
            for (size_t i = 0; i < n; ++i) values[i] = half_float(uint16_t(codes[i % codes.size()]) << 8);
            for (size_t i = 0; i < nq * heads; ++i) lse[i] = (i & 1) ? -2097152.0f : std::numeric_limits<float>::max();
            CUDA_OK(cudaMemcpyAsync(dev_float, values.data(), n * sizeof(float), cudaMemcpyHostToDevice, producer));
            CUDA_OK(cudaEventRecord(ready, producer));
            REQUIRE(small->pack_small(context, dev_float, false, nq, width, packed, ready));
            for (size_t i = 0; i < n; ++i) REQUIRE(packed[i] == codes[i % codes.size()]);
            std::fill(readback.begin(), readback.end(), -99.0f);
            CUDA_OK(cudaMemcpy(output, readback.data(), readback.size() * sizeof(float), cudaMemcpyHostToDevice));
            // Pageable H2D may return after staging, before default-stream DMA
            // completes. The helper owns a NonBlocking stream and cannot infer
            // this fixture's independent destination initializer dependency.
            CUDA_OK(cudaStreamSynchronize(nullptr));
            REQUIRE(small->unpack_small(context, packed, lse.data(), nq, width, heads, output, stride * sizeof(float)));
            CUDA_OK(cudaMemcpy(readback.data(), output, readback.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t row = 0; row < nq; ++row) {
                for (size_t col = 0; col < width; ++col) {
                    if (bits_of(readback[row * stride + col]) != bits_of(values[row * width + col])) {
                        std::fprintf(stderr, "mapped unpack mismatch nq=%zu width=%zu row=%zu col=%zu code=%02x got=%08x expected=%08x\n",
                            nq, width, row, col, packed[row * width + col], bits_of(readback[row * stride + col]), bits_of(values[row * width + col]));
                    }
                    REQUIRE(bits_of(readback[row * stride + col]) == bits_of(values[row * width + col]));
                }
                for (size_t h = 0; h < heads; ++h) REQUIRE(bits_of(readback[row * stride + width + h]) == bits_of(lse[row * heads + h]));
                for (size_t col = width + heads; col < stride; ++col) REQUIRE(readback[row * stride + col] == -99.0f);
            }
            for (size_t i = nq * stride; i < readback.size(); ++i) REQUIRE(readback[i] == -99.0f);
        }
    }

    // All nonfinite F32/F16 classes, including an invalid last CTA. Invalid Q
    // must not publish any partial data into the caller's network buffer.
    constexpr size_t error_width = 2051;
    std::vector<float> erroneous(error_width, 1.0f);
    for (uint32_t bad : {0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00001u, 0x7f800001u}) {
        erroneous.back() = from_bits(bad);
        CUDA_OK(cudaMemcpy(dev_float, erroneous.data(), erroneous.size() * sizeof(float), cudaMemcpyHostToDevice));
        std::memset(packed, 0xa5, erroneous.size());
        REQUIRE(!small->pack_small(context, dev_float, false, 1, erroneous.size(), packed, nullptr));
        REQUIRE(std::all_of(packed, packed + erroneous.size(), [](uint8_t b) { return b == 0xa5; }));
        erroneous.back() = 1.0f;
        CUDA_OK(cudaMemcpy(dev_float, erroneous.data(), erroneous.size() * sizeof(float), cudaMemcpyHostToDevice));
        REQUIRE(small->pack_small(context, dev_float, false, 1, erroneous.size(), packed, nullptr));
        REQUIRE(std::all_of(packed, packed + erroneous.size(), [](uint8_t b) { return b == 0x3c; }));
    }
    for (uint16_t bad : {uint16_t(0x7c00), uint16_t(0xfc00), uint16_t(0x7e00), uint16_t(0x7c01)}) {
        CUDA_OK(cudaMemcpy(dev_half, &bad, sizeof(bad), cudaMemcpyHostToDevice));
        packed[0] = 0xa5;
        REQUIRE(!small->pack_small(context, dev_half, true, 1, 1, packed, nullptr) && packed[0] == 0xa5);
    }
    for (uint32_t code = 0; code < 256; ++code) if ((code & 0x7cu) == 0x7cu) {
        std::memset(packed, 0x3c, error_width);
        packed[error_width - 1] = uint8_t(code);
        REQUIRE(!small->unpack_small(context, packed, nullptr, 1, error_width, 0, output, error_width * sizeof(float)));
        packed[error_width - 1] = 0x3c;
        REQUIRE(small->unpack_small(context, packed, nullptr, 1, error_width, 0, output, error_width * sizeof(float)));
    }
    for (float bad : {INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
        REQUIRE(!small->unpack_small(context, packed, &bad, 1, 3, 1, output, 16));
    }
    float finite_lse = -2097152.0f;
    REQUIRE(small->unpack_small(context, packed, &finite_lse, 1, 3, 1, output, 16));

    // Reject unsupported shapes/pointer ownership/overflow before queueing any
    // work. A nq=10 call is v2 territory; v3 never silently changes algorithms.
    REQUIRE(!small->pack_small(nullptr, dev_float, false, 1, 1, packed, nullptr));
    REQUIRE(!small->pack_small(context, nullptr, false, 1, 1, packed, nullptr));
    REQUIRE(!small->pack_small(context, dev_float, false, 0, 1, packed, nullptr));
    REQUIRE(!small->pack_small(context, dev_float, false, 10, 1, packed, nullptr));
    REQUIRE(!small->pack_small(context, dev_float, false, 1, 0, packed, nullptr));
    REQUIRE(!small->pack_small(context, dev_float, false, 9, SIZE_MAX, packed, nullptr));
    REQUIRE(!small->pack_small(context, dev_float, false, 1, size_t(256) << 20, packed, nullptr));
    REQUIRE(!small->pack_small(context, values.data(), false, 1, 1, packed, nullptr));
    REQUIRE(!small->pack_small(context, reinterpret_cast<uint8_t *>(dev_float) + 1, false, 1, 1, packed, nullptr));
    REQUIRE(!small->pack_small(context, dev_float, false, 1, 1, reinterpret_cast<uint8_t *>(output), nullptr));
    REQUIRE(!small->unpack_small(nullptr, packed, nullptr, 1, 1, 0, output, 4));
    REQUIRE(!small->unpack_small(context, nullptr, nullptr, 1, 1, 0, output, 4));
    REQUIRE(!small->unpack_small(context, packed, nullptr, 10, 1, 0, output, 4));
    REQUIRE(!small->unpack_small(context, packed, nullptr, 0, 1, 0, output, 4));
    REQUIRE(!small->unpack_small(context, packed, nullptr, 1, 0, 0, output, 4));
    REQUIRE(!small->unpack_small(context, packed, nullptr, 1, 3, 1, output, 16));
    REQUIRE(!small->unpack_small(context, packed, &finite_lse, 1, 3, 1, output, 15));
    REQUIRE(!small->unpack_small(context, packed, &finite_lse, 1, 3, 1, output, 12));
    REQUIRE(!small->unpack_small(context, packed, &finite_lse, 9, SIZE_MAX, 1, output, 16));
    REQUIRE(!small->unpack_small(context, packed, &finite_lse, 1, 1, SIZE_MAX, output, 16));
    REQUIRE(!small->unpack_small(context, packed, &finite_lse, 9, 1, 1, output, SIZE_MAX - 3));
    REQUIRE(!small->unpack_small(context, packed, &finite_lse, 1, 3, 1, readback.data(), 16));
    REQUIRE(!small->unpack_small(context, reinterpret_cast<uint8_t *>(dev_float), nullptr, 1, 1, 0, output, 4));
    REQUIRE(!small->unpack_small(context, packed, output, 1, 1, 1, output, 8));
    REQUIRE(!small->unpack_small(context, packed, nullptr, 1, 1, 0, reinterpret_cast<uint8_t *>(output) + 1, 4));

    // Large -> mapped-small -> large -> mapped-small -> large on this very same
    // context: v2 scratch is recreated without invalidating grow-only sidecars.
    float one = 1.0f;
    for (int cycle = 0; cycle < 3; ++cycle) {
        REQUIRE(api->pack_q(context, float_values.data(), false, float_values.size(), packed, nullptr));
        for (size_t i = 0; i < float_values.size(); ++i) REQUIRE(packed[i] == oracle(float_values[i]));
        CUDA_OK(cudaMemcpy(dev_float, &one, sizeof(one), cudaMemcpyHostToDevice));
        REQUIRE(small->pack_small(context, dev_float, false, 1, 1, packed, nullptr) && packed[0] == 0x3c);
        REQUIRE(small->unpack_small(context, packed, nullptr, 1, 1, 0, output, sizeof(float)));
        CUDA_OK(cudaMemcpy(readback.data(), output, sizeof(float), cudaMemcpyDeviceToHost));
        REQUIRE(readback[0] == one);
    }
    REQUIRE(api->pack_q(context, float_values.data(), false, float_values.size(), packed, nullptr));

    // Two private mapped TX/RX pairs run on separate context streams without
    // sharing status or wire staging, including simultaneous pack/unpack calls.
    void * other = api->create_context(0);
    REQUIRE(other);
    float * other_input = nullptr;
    float * other_output = nullptr;
    CUDA_OK(cudaMalloc(&other_input, 64 * sizeof(float)));
    CUDA_OK(cudaMalloc(&other_output, 64 * sizeof(float)));
    std::vector<float> positive(64, 1.0f), negative(64, -2.0f);
    CUDA_OK(cudaMemcpy(dev_float, positive.data(), 64 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(other_input, negative.data(), 64 * sizeof(float), cudaMemcpyHostToDevice));
    std::thread concurrent([&] {
        CUDA_OK(cudaSetDevice(0));
        std::vector<uint8_t> wire(64);
        for (int i = 0; i < 10; ++i) {
            REQUIRE(small->pack_small(other, other_input, false, 1, 64, wire.data(), nullptr));
            REQUIRE(std::all_of(wire.begin(), wire.end(), [](uint8_t b) { return b == 0xc0; }));
            REQUIRE(small->unpack_small(other, wire.data(), nullptr, 1, 64, 0, other_output, 64 * sizeof(float)));
            REQUIRE(retirement->retire_context(other));
            float value = 0;
            CUDA_OK(cudaMemcpy(&value, other_output, sizeof(float), cudaMemcpyDeviceToHost));
            REQUIRE(value == -2.0f);
        }
    });
    for (int i = 0; i < 10; ++i) {
        REQUIRE(small->pack_small(context, dev_float, false, 1, 64, packed, nullptr));
        REQUIRE(std::all_of(packed, packed + 64, [](uint8_t b) { return b == 0x3c; }));
        REQUIRE(small->unpack_small(context, packed, nullptr, 1, 64, 0, output, 64 * sizeof(float)));
        REQUIRE(retirement->retire_context(context));
        CUDA_OK(cudaMemcpy(readback.data(), output, sizeof(float), cudaMemcpyDeviceToHost));
        REQUIRE(readback[0] == 1.0f);
    }
    concurrent.join();
    api->destroy_context(other);
    CUDA_OK(cudaFree(other_input));
    CUDA_OK(cudaFree(other_output));
    CUDA_OK(cudaFree(output));
    std::puts("mapped small v3: finite F16/F32 RNE oracle, nq1/9/10, tails/strides, status reset, invalid inputs and large/small transitions passed");
}

static void test_retirement(const ggml_remote_attn_cuda_retirement_interface_v1 * retirement,
                            const ggml_remote_attn_cuda_fp8_interface_v2 * api,
                            const ggml_remote_attn_cuda_fp8_interface_v3 * small) {
    REQUIRE(retirement && retirement->base == api && retirement->retire_context);
    REQUIRE(!retirement->retire_context(nullptr));
    void * fresh = api->create_context(0);
    REQUIRE(fresh && retirement->retire_context(fresh));
    api->destroy_context(fresh);
    auto * external = static_cast<uint8_t *>(api->alloc_pinned(16));
    REQUIRE(external);
    float * device = nullptr;
    CUDA_OK(cudaMalloc(&device, 4 * sizeof(float)));
    float nan = std::numeric_limits<float>::quiet_NaN();
    CUDA_OK(cudaMemcpy(device, &nan, sizeof(nan), cudaMemcpyHostToDevice));
    for (bool mapped : {false, true}) {
        void * pack = api->create_context(0);
        REQUIRE(pack);
        REQUIRE(!(mapped ? small->pack_small(pack, device, false, 1, 1, external, nullptr) :
                          api->pack_q(pack, &nan, false, 1, external, nullptr)));
        REQUIRE(retirement->retire_context(pack));
        // Retirement proves deallocation safety, never usable numerical output
        // or permission for the serving layer to recycle this failed slot.
        api->destroy_context(pack);
        void * unpack = api->create_context(0);
        REQUIRE(unpack);
        external[0] = 0x7c; // rejected FP8 infinity
        REQUIRE(!(mapped ? small->unpack_small(unpack, external, nullptr, 1, 1, 0, device, sizeof(float)) :
                          api->unpack_out(unpack, external, nullptr, 1, 1, 0, device, sizeof(float))));
        REQUIRE(retirement->retire_context(unpack));
        api->destroy_context(unpack);
    }
    CUDA_OK(cudaFree(device));
    api->free_pinned(external);
    std::puts("retirement v1: null rejected, fresh contexts and retired numerical failures passed; device-fault retirement not injected");
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
    const auto * small = ggml_backend_cuda_remote_attn_fp8_interface_v3();
    const auto * retirement = ggml_backend_cuda_remote_attn_retirement_interface_v1();
#else
    ggml_backend_load_all();
    auto reg = ggml_backend_reg_by_name("CUDA");
    REQUIRE(reg);
    const auto getter = reinterpret_cast<ggml_backend_cuda_remote_attn_fp8_interface_v2_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_remote_attn_fp8_interface_v2"));
    const auto legacy_getter = reinterpret_cast<ggml_backend_cuda_remote_attn_fp8_interface_v1_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_remote_attn_fp8_interface_v1"));
    const auto small_getter = reinterpret_cast<ggml_backend_cuda_remote_attn_fp8_interface_v3_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_remote_attn_fp8_interface_v3"));
    const auto retirement_getter = reinterpret_cast<ggml_backend_cuda_remote_attn_retirement_interface_v1_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_remote_attn_retirement_interface_v1"));
    REQUIRE(getter && legacy_getter && small_getter && retirement_getter);
    const auto * api = getter();
    const auto * legacy = legacy_getter();
    const auto * small = small_getter();
    const auto * retirement = retirement_getter();
#endif
    REQUIRE(api && legacy && small && small->base == api && retirement && retirement->base == api);
    if (std::getenv("GGML_RKVA_TEST_SCRATCH_PRESSURE_ONLY")) {
        constexpr size_t n = 1u << 20;
        void * ctx = api->create_context(0);
        REQUIRE(ctx);
        float * input = nullptr;
        CUDA_OK(cudaMalloc(&input, n * sizeof(float)));
        CUDA_OK(cudaMemset(input, 0, n * sizeof(float)));
        auto * output = static_cast<uint8_t *>(api->alloc_pinned(n));
        REQUIRE(output);
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_OK(cudaMemGetInfo(&free_bytes, &total_bytes));
        constexpr size_t keep_bytes = 3u << 20;
        REQUIRE(free_bytes > keep_bytes);
        void * pressure = nullptr;
        CUDA_OK(cudaMalloc(&pressure, free_bytes - keep_bytes));
        CUDA_OK(cudaMemGetInfo(&free_bytes, &total_bytes));
        REQUIRE(free_bytes < (4u << 20) && free_bytes > n);
        REQUIRE(api->pack_q(ctx, input, false, n, output, nullptr));
        REQUIRE(std::all_of(output, output + n, [](uint8_t v) { return v == 0; }));
        CUDA_OK(cudaFree(pressure));
        api->free_pinned(output);
        CUDA_OK(cudaFree(input));
        api->destroy_context(ctx);
        std::puts("remote-attn scratch pressure parity OK");
        return 0;
    }
    test_retirement(retirement, api, small);
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

    test_mapped_small(small, retirement, api, context, dev_half, dev_float, packed, finite_half, finite_float, producer, ready);

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
