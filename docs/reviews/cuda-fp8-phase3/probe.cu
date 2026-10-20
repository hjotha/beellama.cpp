#include "remote-attn-cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s at line %d\n", #x, cudaGetErrorString(e), __LINE__); std::exit(2); } } while (0)
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "check failed: %s at line %d\n", #x, __LINE__); std::exit(3); } } while (0)
static uint32_t bits(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }
static float from_bits(uint32_t b) { float x; std::memcpy(&x, &b, 4); return x; }
static float from_half_bits(uint16_t b) { __half h; std::memcpy(&h, &b, 2); return __half2float(h); }
static uint8_t oracle(float x) { return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2); }

static int valid(const ggml_remote_attn_cuda_fp8_interface * api) {
    std::vector<uint16_t> halves;
    for (unsigned b = 0; b != 65536; ++b) if ((b & 0x7c00) != 0x7c00) halves.push_back(uint16_t(b));
    std::vector<float> floats;
    for (unsigned b = 0; b < 123; ++b) {
        float lo = from_half_bits(uint16_t(b << 8));
        float hi = from_half_bits(uint16_t((b + 1) << 8));
        float mid = (lo + hi) / 2;
        for (float value : {lo, hi, std::nextafter(mid, lo), mid, std::nextafter(mid, hi)}) {
            floats.push_back(value); floats.push_back(-value);
        }
    }
    for (float value : {57344.0f, std::nextafter(57344.0f, 0.0f), std::nextafter(57344.0f, INFINITY), 60000.0f, 65504.0f, std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min()}) {
        floats.push_back(value); floats.push_back(-value);
    }
    uint32_t rng = 0x12345678u;
    for (unsigned i = 0; i < 1048576; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        if ((rng & 0x7f800000u) != 0x7f800000u) floats.push_back(from_bits(rng));
    }
    const size_t capacity = std::max(floats.size(), halves.size());
    auto * packed = static_cast<uint8_t *>(api->alloc_pinned(capacity));
    REQUIRE(packed);
    void * dev = nullptr; CUDA_OK(cudaMalloc(&dev, floats.size() * 4));
    void * stream = api->create_stream(); void * event = api->create_event(); REQUIRE(stream && event);
    for (bool on_device : {false, true}) {
        if (on_device) { CUDA_OK(cudaMemcpy(dev, floats.data(), floats.size() * 4, cudaMemcpyHostToDevice)); CUDA_OK(cudaDeviceSynchronize()); }
        REQUIRE(api->pack_q(on_device ? dev : floats.data(), false, floats.size(), packed, stream, event));
        REQUIRE(api->event_sync(event));
        for (size_t i = 0; i < floats.size(); ++i) {
            if (packed[i] != oracle(floats[i])) { std::fprintf(stderr, "F32 mismatch bits=%08x got=%02x want=%02x\n", bits(floats[i]), packed[i], oracle(floats[i])); return 4; }
        }
        std::printf("pack F32 %s: %zu values, 0 mismatches vs NVIDIA SATFINITE/RNE\n", on_device ? "device" : "host", floats.size());
    }
    for (bool on_device : {false, true}) {
        if (on_device) { CUDA_OK(cudaMemcpy(dev, halves.data(), halves.size() * 2, cudaMemcpyHostToDevice)); CUDA_OK(cudaDeviceSynchronize()); }
        REQUIRE(api->pack_q(on_device ? dev : halves.data(), true, halves.size(), packed, stream, event));
        for (size_t i = 0; i < halves.size(); ++i) REQUIRE(packed[i] == oracle(from_half_bits(halves[i])));
        std::printf("pack F16 %s: all %zu finite patterns, 0 mismatches\n", on_device ? "device" : "host", halves.size());
    }
    for (uint32_t b : {0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00001u, 0x7f800001u}) {
        float input = from_bits(b); REQUIRE(!api->pack_q(&input, false, 1, packed, stream, event));
    }
    std::puts("pack: 5 F32 nonfinite patterns rejected");
    std::vector<uint16_t> nonfinite_halves;
    for (unsigned b = 0; b < 65536; ++b) if ((b & 0x7c00) == 0x7c00) nonfinite_halves.push_back(uint16_t(b));
    REQUIRE(!api->pack_q(nonfinite_halves.data(), true, nonfinite_halves.size(), packed, stream, event));
    std::puts("pack: batch of all 2048 F16 nonfinite patterns rejected");

    std::vector<uint8_t> finite;
    for (unsigned b = 0; b < 256; ++b) if ((b & 0x7c) != 0x7c) finite.push_back(uint8_t(b));
    std::copy(finite.begin(), finite.end(), packed);
    auto * lse = static_cast<float *>(api->alloc_pinned(16)); REQUIRE(lse); lse[0] = -1.25f;
    constexpr size_t width = 248, stride = 256;
    std::vector<float> result(stride, -123.0f);
    for (bool on_device : {false, true}) {
        if (on_device) { CUDA_OK(cudaMemcpy(dev, result.data(), stride * 4, cudaMemcpyHostToDevice)); CUDA_OK(cudaDeviceSynchronize()); }
        REQUIRE(api->unpack_out(packed, lse, 1, width, 1, on_device ? dev : result.data(), stride * 4, stream, event));
        REQUIRE(api->event_sync(event));
        if (on_device) CUDA_OK(cudaMemcpy(result.data(), dev, stride * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < width; ++i) REQUIRE(bits(result[i]) == bits(from_half_bits(uint16_t(finite[i]) << 8)));
        REQUIRE(result[width] == lse[0]);
        for (size_t i = width + 1; i < stride; ++i) REQUIRE(result[i] == -123.0f);
        std::printf("unpack %s: all 248 finite FP8 patterns exact; LSE and row padding preserved\n", on_device ? "device" : "host");
    }
    // Protocol geometry and multiple rows, including nontrivial output stride.
    std::fill_n(packed, 512, uint8_t(0x3c)); lse[0] = 1.0f; lse[1] = 2.0f;
    result.assign(520, -123.0f); CUDA_OK(cudaMemcpy(dev, result.data(), result.size() * 4, cudaMemcpyHostToDevice)); CUDA_OK(cudaDeviceSynchronize());
    REQUIRE(api->unpack_out(packed, lse, 2, 256, 1, dev, 260 * 4, stream, event));
    CUDA_OK(cudaMemcpy(result.data(), dev, result.size() * 4, cudaMemcpyDeviceToHost));
    for (size_t row = 0; row < 2; ++row) {
        for (size_t i = 0; i < 256; ++i) REQUIRE(result[row * 260 + i] == 1.0f);
        REQUIRE(result[row * 260 + 256] == lse[row]);
        for (size_t i = 257; i < 260; ++i) REQUIRE(result[row * 260 + i] == -123.0f);
    }
    std::puts("unpack device: 2 protocol rows and padded stride passed");
    std::fill_n(packed, width, uint8_t(0));
    for (unsigned b = 0; b < 256; ++b) if ((b & 0x7c) == 0x7c) {
        packed[0] = uint8_t(b); REQUIRE(!api->unpack_out(packed, lse, 1, width, 1, dev, stride * 4, stream, event));
    }
    std::puts("unpack: all 8 nonfinite FP8 patterns rejected");
    packed[0] = 0;
    for (float value : {INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
        lse[0] = value; REQUIRE(!api->unpack_out(packed, lse, 1, width, 1, dev, stride * 4, stream, event));
    }
    std::puts("unpack: 3 nonfinite LSE patterns rejected");
    api->destroy_event(event); api->destroy_stream(stream); CUDA_OK(cudaFree(dev)); api->free_pinned(lse); api->free_pinned(packed);
    std::puts("VALID CASES PASSED");
    return 0;
}

int main(int argc, char ** argv) {
    auto * api = ggml_backend_cuda_remote_attn_fp8_interface(); REQUIRE(api);
    std::string mode = argc > 1 ? argv[1] : "valid";
    if (mode == "valid") return valid(api);
    const size_t width = mode == "odd-width" ? 3 : 256;
    auto * fp8 = static_cast<uint8_t *>(api->alloc_pinned(width)); REQUIRE(fp8);
    std::fill_n(fp8, width, uint8_t(0x3c));
    auto * lse = static_cast<float *>(api->alloc_pinned(4)); REQUIRE(lse); lse[0] = 1.0f;
    std::vector<float> result(width + 1, -123.0f);
    void * dev = nullptr; CUDA_OK(cudaMalloc(&dev, result.size() * 4));
    CUDA_OK(cudaMemcpy(dev, result.data(), result.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaDeviceSynchronize());
    bool ok = api->unpack_out(fp8, mode == "missing-lse" ? nullptr : lse, 1, width, 1, dev, result.size() * 4, nullptr, nullptr);
    if (mode != "odd-width") {
        CUDA_OK(cudaMemcpy(result.data(), dev, result.size() * 4, cudaMemcpyDeviceToHost));
        std::printf("missing-lse: return=%s, output LSE=%g (sentinel before call=-123)\n", ok ? "true" : "false", result.back());
        REQUIRE(ok && result.back() == -123.0f);
    } else std::printf("odd-width: return=%s\n", ok ? "true" : "false");
    cudaFree(dev); api->free_pinned(lse); api->free_pinned(fp8);
    return 0;
}
