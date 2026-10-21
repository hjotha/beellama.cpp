#include "remote-attn-cuda.cuh"
#include "ggml-remote-attn-trace.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include <cuda_runtime.h>
#include <cuda_fp16.h>

static __device__ __forceinline__ uint32_t rkva_cuda_round_shift_even(uint32_t value, unsigned shift) {
    const uint32_t base = value >> shift;
    const uint32_t remainder = value & ((uint32_t(1) << shift) - 1);
    const uint32_t halfway = uint32_t(1) << (shift - 1);
    return base + (remainder > halfway || (remainder == halfway && (base & 1)));
}

// Direct F32 to finite E5M2: nearest-even, finite saturation, signed zero.
static __device__ __forceinline__ bool f32_to_fp8_e5m2_rne(float val, uint8_t & out) {
    uint32_t bits = __float_as_uint(val);
    uint32_t magnitude = bits & 0x7fffffffu;
    uint8_t sign = uint8_t((bits >> 24) & 0x80u);
    if (magnitude >= 0x7f800000u) {
        return false;
    }
    if (magnitude >= 0x47600000u) {
        out = sign | 0x7bu;
        return true;
    }
    if (magnitude <= 0x37000000u) {
        out = sign;
        return true;
    }
    uint32_t exponent = magnitude >> 23;
    uint32_t mantissa = magnitude & 0x7fffffu;
    uint32_t encoded = magnitude < 0x38800000u ?
        rkva_cuda_round_shift_even(mantissa | 0x800000u, 134u - exponent) :
        ((exponent - 112u) << 2) + rkva_cuda_round_shift_even(mantissa, 21);
    out = sign | uint8_t(encoded);
    return true;
}

static __device__ __forceinline__ bool f16_to_fp8_e5m2_rne(half val, uint8_t & out) {
    return f32_to_fp8_e5m2_rne(__half2float(val), out);
}

static __device__ __forceinline__ bool fp8_e5m2_to_f32(uint8_t byte, float & out) {
    uint16_t h = uint16_t(byte) << 8;
    if ((h & 0x7c00u) == 0x7c00u) {
        return false;
    }
    out = __half2float(__ushort_as_half(h));
    return true;
}

static __global__ void rkva_cuda_fp8_pack_q_f32_kernel(
        const float * __restrict__ src,
        uint8_t     * __restrict__ dst,
        size_t n,
        int         * __restrict__ err_flag) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint8_t packed = 0;
    if (!f32_to_fp8_e5m2_rne(src[idx], packed)) {
        atomicExch(err_flag, 1);
        return;
    }
    dst[idx] = packed;
}

static __global__ void rkva_cuda_fp8_pack_q_f16_kernel(
        const half  * __restrict__ src,
        uint8_t     * __restrict__ dst,
        size_t n,
        int         * __restrict__ err_flag) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint8_t packed = 0;
    if (!f16_to_fp8_e5m2_rne(src[idx], packed)) {
        atomicExch(err_flag, 1);
        return;
    }
    dst[idx] = packed;
}

static __global__ void rkva_cuda_fp8_unpack_out_kernel(
        const uint8_t * __restrict__ src_fp8,
        const float   * __restrict__ src_lse,
        float         * __restrict__ dst_out,
        size_t nq,
        size_t out_width,
        size_t n_head,
        size_t dst_stride_floats,
        int           * __restrict__ err_flag) {
    size_t col_total = out_width + n_head;
    size_t total = nq * col_total;
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t row = idx / col_total;
    size_t col = idx % col_total;
    float * dst_row = dst_out + row * dst_stride_floats;
    if (col < out_width) {
        uint8_t byte = src_fp8[row * out_width + col];
        float val = 0.0f;
        if (!fp8_e5m2_to_f32(byte, val)) {
            atomicExch(err_flag, 1);
            return;
        }
        dst_row[col] = val;
    } else {
        size_t h = col - out_width;
        if (src_lse) {
            float lse = src_lse[row * n_head + h];
            if (!isfinite(lse)) {
                atomicExch(err_flag, 1);
                return;
            }
            dst_row[col] = lse;
        }
    }
}

static __device__ __forceinline__ float rkva_small_float(float value) { return value; }
static __device__ __forceinline__ float rkva_small_float(half value) { return __half2float(value); }

// Each lane writes one naturally aligned four-byte word to mapped host memory.
// The final word is padded locally; no source element beyond n is read. Every
// lane participates in the CTA reduction, including lanes outside the input.
template <typename T>
static __global__ void rkva_cuda_fp8_pack_small_kernel(
        const T * __restrict__ src, uint32_t * __restrict__ dst,
        size_t n, uint32_t * __restrict__ status) {
    const size_t word = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t first = word * 4;
    uint32_t packed = 0;
    bool invalid = false;
    if (first < n) {
        #pragma unroll
        for (unsigned lane = 0; lane < 4; ++lane) {
            uint8_t byte = 0;
            if (first + lane < n && !f32_to_fp8_e5m2_rne(rkva_small_float(src[first + lane]), byte)) {
                invalid = true;
            }
            packed |= uint32_t(byte) << (8 * lane);
        }
        dst[word] = packed;
    }
    const int cta_invalid = __syncthreads_or(invalid);
    if (threadIdx.x == 0) status[blockIdx.x] = uint32_t(cta_invalid != 0);
}

// Flattened FP8 words may cross an odd-width row: each decoded byte computes
// its own destination row, preserving arbitrary float-aligned row padding.
static __global__ void rkva_cuda_fp8_unpack_small_kernel(
        const uint32_t * __restrict__ src, const float * __restrict__ lse,
        float * __restrict__ dst, size_t n, size_t words, size_t width,
        size_t heads, size_t n_lse, size_t stride, uint32_t * __restrict__ status) {
    const size_t item = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    bool invalid = false;
    if (item < words) {
        const uint32_t packed = src[item];
        #pragma unroll
        for (unsigned lane = 0; lane < 4; ++lane) {
            const size_t index = item * 4 + lane;
            if (index < n) {
                float value = 0;
                if (!fp8_e5m2_to_f32(uint8_t(packed >> (8 * lane)), value)) {
                    invalid = true;
                } else {
                    dst[(index / width) * stride + index % width] = value;
                }
            }
        }
    } else if (item - words < n_lse) {
        const size_t index = item - words;
        const float value = lse[index];
        if (!isfinite(value)) {
            invalid = true;
        } else {
            dst[(index / heads) * stride + width + index % heads] = value;
        }
    }
    const int cta_invalid = __syncthreads_or(invalid);
    if (threadIdx.x == 0) status[blockIdx.x] = uint32_t(cta_invalid != 0);
}

struct rkva_cuda_mapped_sidecar {
    void * host = nullptr;
    void * device = nullptr; // cudaHostGetDevicePointer alias, never assumed equal
    size_t capacity = 0;
};

// Each serving slot owns its scratch and stream. No CUDA operation outlives a
// helper call, so slot reset/drain may safely reuse pinned network buffers.
struct rkva_cuda_remote_scratch {
    void * dev_data = nullptr;
    size_t dev_capacity = 0;
    int * dev_err = nullptr;
    int * host_err = nullptr;
    cudaStream_t stream = nullptr;
    int device = -1;
    bool poisoned = false;
    bool retirement_confirmed = true;
    bool small_capability_checked = false;
    bool small_supported = false;
    rkva_cuda_mapped_sidecar small_tx;
    rkva_cuda_mapped_sidecar small_rx;
    cudaEvent_t timing_events[7]{};
    bool timing_disabled = false;
    std::mutex mutex;
};

// Timing events are private to the same context mutex/stream as the scratch.
// They are resolved only after the helper's mandatory retirement; no profiling
// query, synchronization or event exists on the trace-disabled path.
struct rkva_cuda_timing_trace {
    rkva_cuda_remote_scratch & scratch;
    ggml_remote_attn_trace_event identity{};
    const char * operation;
    bool active = false;
    unsigned recorded = 0;

    rkva_cuda_timing_trace(rkva_cuda_remote_scratch & s, const char * name) : scratch(s), operation(name) {
        if (!ggml_remote_attn_trace_enabled() || s.timing_disabled) return;
        identity = ggml_remote_attn_trace_identity(name, nullptr);
        identity.begin_us = ggml_time_us();
        active = true;
        for (auto & event : scratch.timing_events) {
            if (event) continue;
            const cudaError_t status = cudaEventCreate(&event);
            if (status != cudaSuccess) { error(status); return; }
        }
    }

    void error(cudaError_t status) {
        active = false;
        scratch.timing_disabled = true;
        auto event = identity;
        std::snprintf(event.phase, sizeof(event.phase), "%s_profile_error", operation);
        event.end_us = ggml_time_us();
        event.flags |= 1u;
        event.value = static_cast<uint64_t>(status);
        ggml_remote_attn_trace_record(&event);
    }

    void mark(unsigned index, cudaStream_t stream) {
        if (!active) return;
        const cudaError_t status = cudaEventRecord(scratch.timing_events[index], stream);
        if (status != cudaSuccess) { error(status); return; }
        recorded = index + 1;
    }

    void interval(const char * phase, unsigned first, unsigned last, uint64_t bytes,
                  uint64_t status, bool inclusive = false) {
        if (!active || last >= recorded) return;
        float elapsed_ms = 0;
        const cudaError_t result = cudaEventElapsedTime(&elapsed_ms,
            scratch.timing_events[first], scratch.timing_events[last]);
        if (result != cudaSuccess) { error(result); return; }
        auto event = identity;
        std::strncpy(event.phase, phase, sizeof(event.phase) - 1);
        event.phase[sizeof(event.phase) - 1] = '\0';
        event.end_us = ggml_time_us();
        event.gpu_us = double(elapsed_ms) * 1000.0;
        event.bytes = bytes;
        event.value = status;
        event.flags |= 0x80000000u | (inclusive ? 0x40000000u : 0u) | (status ? 1u : 0u);
        ggml_remote_attn_trace_record(&event);
    }
};

struct rkva_device_guard {
    int previous = -1;
    bool ok = false;
    cudaError_t status = cudaSuccess;
    const char * failed_api = nullptr;
    explicit rkva_device_guard(int device) {
        status = cudaGetDevice(&previous);
        if (status != cudaSuccess) { failed_api = "cudaGetDevice"; return; }
        if (previous != device) {
            status = cudaSetDevice(device);
            if (status != cudaSuccess) { failed_api = "cudaSetDevice"; return; }
        }
        ok = true;
    }
    ~rkva_device_guard() {
        if (ok && previous >= 0) cudaSetDevice(previous);
    }
};

// Failure diagnostics only. -1 means a validation/capability/status condition,
// rather than a CUDA API result. Never query/clear CUDA error state for a log.
static bool rkva_small_failure(const char * operation, const char * stage, int device,
                               int cuda_status = -1, size_t detail0 = 0, size_t detail1 = 0,
                               int observed_device = -3) {
    std::fprintf(stderr, "remote-attn CUDA mapped-small: op=%s stage=%s device=%d cuda_status=%d detail0=%zu detail1=%zu observed_device=%d\n",
        operation, stage, device, cuda_status, detail0, detail1, observed_device);
    return false;
}

static bool rkva_size_mul(size_t a, size_t b, size_t & result) {
    if (b && a > SIZE_MAX / b) return false;
    result = a * b;
    return true;
}

static bool rkva_size_add(size_t a, size_t b, size_t & result) {
    if (a > SIZE_MAX - b) return false;
    result = a + b;
    return true;
}

static bool rkva_align_float(size_t value, size_t & result) {
    if (!rkva_size_add(value, sizeof(float) - 1, result)) return false;
    result &= ~(sizeof(float) - 1);
    return true;
}

static int rkva_pointer_device_impl(const void * ptr, const char * failure_operation) {
    if (!ptr) return -2;
    cudaPointerAttributes attr{};
    const cudaError_t status = cudaPointerGetAttributes(&attr, ptr);
    if (status == cudaErrorInvalidValue) {
        cudaGetLastError(); // Older runtimes report ordinary host allocations this way.
        return -1;
    }
    if (status != cudaSuccess) {
        if (failure_operation) rkva_small_failure(failure_operation, "cudaPointerGetAttributes", -1, status);
        return -2;
    }
    if (attr.type == cudaMemoryTypeDevice) return attr.device;
    if (attr.type == cudaMemoryTypeHost || attr.type == cudaMemoryTypeUnregistered) return -1;
    // Managed memory requires a separate residency/coherency contract.
    if (failure_operation) rkva_small_failure(failure_operation, "unsupported_pointer_type", attr.device, -1, size_t(attr.type));
    return -2;
}

static int rkva_pointer_device(const void * ptr) {
    return rkva_pointer_device_impl(ptr, nullptr);
}

static bool rkva_scratch_ensure(rkva_cuda_remote_scratch & s, size_t needed) {
    if (s.poisoned) return false;
    if (!s.stream && cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking) != cudaSuccess) return false;
    if (!s.dev_err && cudaMalloc(&s.dev_err, sizeof(int)) != cudaSuccess) return false;
    if (!s.host_err && cudaHostAlloc(&s.host_err, sizeof(int), cudaHostAllocPortable) != cudaSuccess) return false;
    if (s.dev_capacity < needed) {
        void * replacement = nullptr;
        const size_t alloc_sz = std::max<size_t>(needed, 4u << 20);
        if (cudaMalloc(&replacement, alloc_sz) != cudaSuccess) return false;
        if (s.dev_data && cudaFree(s.dev_data) != cudaSuccess) {
            cudaFree(replacement);
            s.poisoned = true;
            return false;
        }
        s.dev_data = replacement;
        s.dev_capacity = alloc_sz;
    }
    return true;
}

// Every early exit after an enqueue retires the stream while all input/output
// buffers and the context mutex remain owned. A failed retirement poisons the
// context so no later call can recycle its scratch after an uncertain failure.
struct rkva_stream_retirement {
    rkva_cuda_remote_scratch & scratch;
    cudaStream_t stream;
    bool pending = false;
    const char * trace_phase = "cuda_retire_host";
    const char * failure_operation = nullptr;
    bool finish() {
        if (!pending) return true;
        pending = false;
        scratch.retirement_confirmed = false;
        ggml_remote_attn_trace_scope timing(trace_phase);
        const cudaError_t status = cudaStreamSynchronize(stream);
        if (status == cudaSuccess) {
            scratch.retirement_confirmed = true;
            return true;
        }
        timing.event.flags |= 1u;
        scratch.poisoned = true;
        if (failure_operation) rkva_small_failure(failure_operation, "cudaStreamSynchronize", scratch.device, status);
        return false;
    }
    ~rkva_stream_retirement() { finish(); }
};

static bool rkva_grid(size_t elements, dim3 & grid) {
    const size_t blocks = elements / 256 + (elements % 256 != 0);
    // CUDA grid.x is limited to 2^31-1 on supported GPUs.
    if (!blocks || blocks > 0x7fffffffu) return false;
    grid = dim3(static_cast<unsigned>(blocks));
    return true;
}

static bool rkva_supports_small(int device) {
    if (device < 0) return rkva_small_failure("supports", "invalid_device", device);
    rkva_device_guard guard(device);
    if (!guard.ok) return rkva_small_failure("supports", guard.failed_api, device, guard.status);
    int mapped = 0, uva = 0;
    unsigned flags = 0;
    cudaError_t status = cudaDeviceGetAttribute(&mapped, cudaDevAttrCanMapHostMemory, device);
    if (status != cudaSuccess) return rkva_small_failure("supports", "cudaDeviceGetAttribute/can_map", device, status);
    if (!mapped) return rkva_small_failure("supports", "can_map_false", device);
    status = cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, device);
    if (status != cudaSuccess) return rkva_small_failure("supports", "cudaDeviceGetAttribute/uva", device, status);
    if (!uva) return rkva_small_failure("supports", "uva_false", device);
    status = cudaGetDeviceFlags(&flags);
    if (status != cudaSuccess) return rkva_small_failure("supports", "cudaGetDeviceFlags", device, status);
    if (!(flags & cudaDeviceMapHost)) return rkva_small_failure("supports", "device_map_host_flag_missing", device, -1, flags, cudaDeviceMapHost);
    return true;
}

// These are wire-sized, bounded sidecars, not a second unbounded device pool.
// TX and RX grow independently, and their status arrays are owned by the CTA
// which rewrites them on every launch; no GPU memset or status DMA is needed.
static constexpr size_t RKVA_SMALL_MAX_SIDECAR = size_t(256) << 20;

static bool rkva_small_ensure(rkva_cuda_remote_scratch & s, rkva_cuda_mapped_sidecar & sidecar,
                              size_t needed) {
    if (s.poisoned) return rkva_small_failure("ensure", "poisoned", s.device);
    if (!s.retirement_confirmed) return rkva_small_failure("ensure", "retirement_unconfirmed", s.device);
    if (!needed || needed > RKVA_SMALL_MAX_SIDECAR) return rkva_small_failure("ensure", "sidecar_bounds", s.device, -1, needed, RKVA_SMALL_MAX_SIDECAR);
    if (!s.small_capability_checked) {
        s.small_supported = rkva_supports_small(s.device);
        s.small_capability_checked = true;
    }
    if (!s.small_supported) return rkva_small_failure("ensure", "cached_capability_unsupported", s.device);
    if (!s.stream) {
        const cudaError_t status = cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking);
        if (status != cudaSuccess) return rkva_small_failure("ensure", "cudaStreamCreateWithFlags", s.device, status);
    }

    // All previous helper work retired while holding this context mutex. Drop
    // only the old device staging/error allocations, keeping the private stream
    // and both mapped buffers. A later v2 call recreates ordinary device scratch.
    if (s.dev_data || s.dev_err) {
        ggml_remote_attn_trace_scope trim("cuda_small_scratch_release");
        trim.event.bytes = s.dev_capacity; // allocation gauge, not PCIe or wire traffic
        trim.event.value = 1;
        trim.event.flags |= 1u;
        if (s.dev_data) {
            const cudaError_t status = cudaFree(s.dev_data);
            if (status != cudaSuccess) { s.poisoned = true; return rkva_small_failure("ensure", "cudaFree/old_dev_data", s.device, status, s.dev_capacity); }
            s.dev_data = nullptr;
            s.dev_capacity = 0;
        }
        if (s.dev_err) {
            const cudaError_t status = cudaFree(s.dev_err);
            if (status != cudaSuccess) { s.poisoned = true; return rkva_small_failure("ensure", "cudaFree/old_dev_err", s.device, status); }
            s.dev_err = nullptr;
        }
        trim.event.flags &= ~1u;
    }
    if (sidecar.capacity >= needed) return true;

    ggml_remote_attn_trace_scope grow("cuda_small_map_grow");
    grow.event.flags |= 1u;
    const size_t capacity = std::max(needed, std::min(RKVA_SMALL_MAX_SIDECAR, sidecar.capacity * 2));
    grow.event.bytes = capacity; // host allocation gauge; no device DMA
    void * host = nullptr;
    void * alias = nullptr;
    const unsigned requested = cudaHostAllocMapped | cudaHostAllocPortable;
    cudaError_t status = cudaHostAlloc(&host, capacity, requested);
    if (status != cudaSuccess) return rkva_small_failure("ensure", "cudaHostAlloc", s.device, status, capacity, requested);
    const auto discard_new_host = [&] {
        const cudaError_t cleanup = cudaFreeHost(host);
        if (cleanup != cudaSuccess) rkva_small_failure("ensure", "cudaFreeHost/new_cleanup", s.device, cleanup, capacity);
    };
    unsigned actual = 0;
    status = cudaHostGetFlags(&actual, host);
    if (status != cudaSuccess) {
        rkva_small_failure("ensure", "cudaHostGetFlags", s.device, status); discard_new_host(); return false;
    }
    // This sidecar belongs to one device/primary context. With UVA, runtime
    // host allocations are device-accessible independently of the Portable
    // flag. Some real allocations report Mapped alone after requesting both.
    // Validate the actual mapping below; flag-word equality is not proof that
    // a usable device pointer exists (nor is Portable needed by this owner).
    status = cudaHostGetDevicePointer(&alias, host, 0);
    if (status != cudaSuccess) {
        rkva_small_failure("ensure", "cudaHostGetDevicePointer", s.device, status); discard_new_host(); return false;
    }
    if (!alias || reinterpret_cast<uintptr_t>(host) % alignof(uint32_t) ||
            reinterpret_cast<uintptr_t>(alias) % alignof(uint32_t)) {
        rkva_small_failure("ensure", "mapped_alias_or_alignment", s.device, -1,
            reinterpret_cast<uintptr_t>(host), reinterpret_cast<uintptr_t>(alias));
        discard_new_host();
        return false;
    }
    if ((actual & requested) != requested) {
        std::fprintf(stderr,
            "remote-attn CUDA mapped-small: validated owner-device mapping device=%d reported_flags=%u requested_flags=%u\n",
            s.device, actual, requested);
    }
    if (sidecar.host) {
        status = cudaFreeHost(sidecar.host);
        if (status != cudaSuccess) {
            discard_new_host();
            s.poisoned = true;
            return rkva_small_failure("ensure", "cudaFreeHost/old_sidecar", s.device, status, sidecar.capacity);
        }
    }
    sidecar = {host, alias, capacity};
    grow.event.flags &= ~1u;
    return true;
}

static bool rkva_small_layout(size_t work_items, size_t payload_bytes, dim3 & grid,
                              size_t & status_offset, size_t & needed) {
    size_t status_bytes;
    return rkva_grid(work_items, grid) && rkva_align_float(payload_bytes, status_offset) &&
           rkva_size_mul(size_t(grid.x), sizeof(uint32_t), status_bytes) &&
           rkva_size_add(status_offset, status_bytes, needed) && needed <= RKVA_SMALL_MAX_SIDECAR;
}

static bool rkva_small_status_ok(const rkva_cuda_mapped_sidecar & sidecar, size_t offset, unsigned blocks,
                                 const char * operation, int device) {
    const auto * status = reinterpret_cast<const uint32_t *>(static_cast<const uint8_t *>(sidecar.host) + offset);
    for (unsigned block = 0; block < blocks; ++block) {
        const uint32_t value = status[block];
        if (value) return rkva_small_failure(operation, value == 1 ? "cta_nonfinite" : "cta_status_unexpected", device, -1, block, value);
    }
    return true;
}

static bool rkva_pack_small_impl(rkva_cuda_remote_scratch & s, const void * src, bool is_f16,
                                size_t nq, size_t width, uint8_t * dst, cudaEvent_t producer) {
    ggml_remote_attn_trace_scope timing("cuda_small_pack_host");
    timing.event.flags |= 0x40000001u;
    const size_t element_size = is_f16 ? sizeof(half) : sizeof(float);
    size_t n, input_bytes, padded, status_offset, needed;
    if (!src || !dst || nq < 1 || nq > 9 || !width)
        return rkva_small_failure("pack", "input_arguments", s.device, -1, nq, width);
    if (reinterpret_cast<uintptr_t>(src) % element_size)
        return rkva_small_failure("pack", "input_alignment", s.device, -1, element_size);
    const int src_device = rkva_pointer_device_impl(src, "pack_source");
    if (src_device != s.device)
        return rkva_small_failure("pack", "source_device_mismatch", s.device, -1, nq, width, src_device);
    const int dst_device = rkva_pointer_device_impl(dst, "pack_destination");
    if (dst_device != -1)
        return rkva_small_failure("pack", "destination_not_host", s.device, -1, nq, width, dst_device);
    if (!rkva_size_mul(nq, width, n) || !rkva_size_mul(n, element_size, input_bytes) ||
        !rkva_align_float(n, padded)) return rkva_small_failure("pack", "size_overflow", s.device, -1, nq, width);
    dim3 grid;
    if (!rkva_small_layout(padded / 4, padded, grid, status_offset, needed))
        return rkva_small_failure("pack", "layout_bounds", s.device, -1, padded, RKVA_SMALL_MAX_SIDECAR);
    if (!rkva_small_ensure(s, s.small_tx, needed)) return false;
    timing.event.bytes = n; // useful mapped PCIe payload, not zero-copy-free traffic
    rkva_stream_retirement retire{s, s.stream, false, "cuda_small_pack_retire_host", "pack"};
    rkva_cuda_timing_trace gpu(s, "cuda_small_pack");
    auto * output = static_cast<uint32_t *>(s.small_tx.device);
    auto * status = reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(s.small_tx.device) + status_offset);
    {
        ggml_remote_attn_trace_scope enqueue("cuda_small_pack_enqueue_host");
        enqueue.event.bytes = padded + size_t(grid.x) * sizeof(uint32_t);
        // Four v2 commands become one kernel (producer wait and retirement stay).
        enqueue.event.value = 3; // CUDA memcpy/memset commands avoided, not PCIe bytes
        retire.pending = true;
        s.retirement_confirmed = false;
        gpu.mark(0, s.stream);
        if (producer) {
            const cudaError_t wait_status = cudaStreamWaitEvent(s.stream, producer, 0);
            if (wait_status != cudaSuccess) return rkva_small_failure("pack", "cudaStreamWaitEvent", s.device, wait_status);
        } else {
            const cudaError_t producer_status = cudaDeviceSynchronize();
            if (producer_status != cudaSuccess) {
                s.poisoned = true;
                return rkva_small_failure("pack", "cudaDeviceSynchronize", s.device, producer_status);
            }
        }
        gpu.mark(1, s.stream);
        if (is_f16) {
            rkva_cuda_fp8_pack_small_kernel<<<grid, 256, 0, s.stream>>>(
                static_cast<const half *>(src), output, n, status);
        } else {
            rkva_cuda_fp8_pack_small_kernel<<<grid, 256, 0, s.stream>>>(
                static_cast<const float *>(src), output, n, status);
        }
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) return rkva_small_failure("pack", "launch/cudaGetLastError", s.device, launch_status, grid.x, n);
        gpu.mark(2, s.stream);
    }
    if (!retire.finish()) return false;
    const bool ok = rkva_small_status_ok(s.small_tx, status_offset, grid.x, "pack", s.device);
    gpu.interval("cuda_small_pack_producer_gpu", 0, 1, 0, !ok);
    gpu.interval("cuda_small_pack_kernel_gpu", 1, 2, padded + size_t(grid.x) * sizeof(uint32_t), !ok);
    if (!ok) return false; // never publish even a partial wire output on invalid Q
    {
        ggml_remote_attn_trace_scope copy("cuda_small_pack_cpu_copy");
        copy.event.bytes = n;
        std::memcpy(dst, s.small_tx.host, n);
    }
    timing.event.flags &= ~1u;
    return true;
}

static bool rkva_unpack_small_impl(rkva_cuda_remote_scratch & s, const uint8_t * src, const float * lse,
                                  size_t nq, size_t width, size_t heads, void * dst, size_t stride) {
    ggml_remote_attn_trace_scope timing("cuda_small_unpack_host");
    timing.event.flags |= 0x40000001u;
    size_t n, n_lse, lse_bytes, row_elements, row_bytes, extent, padded, payload, items, status_offset, needed;
    if (!src || !dst || nq < 1 || nq > 9 || !width || (heads && !lse))
        return rkva_small_failure("unpack", "input_arguments", s.device, -1, nq, width);
    if (reinterpret_cast<uintptr_t>(dst) % alignof(float) || stride % sizeof(float))
        return rkva_small_failure("unpack", "output_alignment", s.device, -1, stride);
    const int dst_device = rkva_pointer_device_impl(dst, "unpack_destination");
    if (dst_device != s.device)
        return rkva_small_failure("unpack", "destination_device_mismatch", s.device, -1, nq, width, dst_device);
    const int src_device = rkva_pointer_device_impl(src, "unpack_source");
    if (src_device != -1)
        return rkva_small_failure("unpack", "source_not_host", s.device, -1, nq, width, src_device);
    if (lse) {
        const int lse_device = rkva_pointer_device_impl(lse, "unpack_lse");
        if (lse_device != -1)
            return rkva_small_failure("unpack", "lse_not_host", s.device, -1, nq, heads, lse_device);
    }
    if (!rkva_size_mul(nq, width, n) ||
        !rkva_size_mul(nq, heads, n_lse) || !rkva_size_mul(n_lse, sizeof(float), lse_bytes) ||
        !rkva_size_add(width, heads, row_elements) || !rkva_size_mul(row_elements, sizeof(float), row_bytes) ||
        stride < row_bytes || !rkva_size_mul(nq - 1, stride, extent) || !rkva_size_add(extent, row_bytes, extent) ||
        !rkva_align_float(n, padded) || !rkva_size_add(padded, lse_bytes, payload) ||
        !rkva_size_add(padded / 4, n_lse, items)) return rkva_small_failure("unpack", "size_or_stride_bounds", s.device, -1, nq, stride);
    dim3 grid;
    if (!rkva_small_layout(items, payload, grid, status_offset, needed))
        return rkva_small_failure("unpack", "layout_bounds", s.device, -1, payload, RKVA_SMALL_MAX_SIDECAR);
    if (!rkva_small_ensure(s, s.small_rx, needed)) return false;
    timing.event.bytes = n + lse_bytes;
    auto * host = static_cast<uint8_t *>(s.small_rx.host);
    {
        ggml_remote_attn_trace_scope copy("cuda_small_unpack_cpu_copy");
        copy.event.bytes = n + lse_bytes;
        std::memcpy(host, src, n);
        // Only the final 0..3 bytes read by the word kernel need padding.
        if (padded != n) std::memset(host + n, 0, padded - n);
        if (lse_bytes) std::memcpy(host + padded, lse, lse_bytes);
    }
    rkva_stream_retirement retire{s, s.stream, false, "cuda_small_unpack_retire_host", "unpack"};
    rkva_cuda_timing_trace gpu(s, "cuda_small_unpack");
    auto * data = static_cast<uint8_t *>(s.small_rx.device);
    {
        ggml_remote_attn_trace_scope enqueue("cuda_small_unpack_enqueue_host");
        enqueue.event.bytes = payload + size_t(grid.x) * sizeof(uint32_t);
        enqueue.event.value = heads ? 4 : 3; // v2 memcpy/memset commands avoided
        retire.pending = true;
        s.retirement_confirmed = false;
        gpu.mark(0, s.stream);
        rkva_cuda_fp8_unpack_small_kernel<<<grid, 256, 0, s.stream>>>(
            reinterpret_cast<const uint32_t *>(data), reinterpret_cast<const float *>(data + padded),
            static_cast<float *>(dst), n, padded / 4, width, heads, n_lse, stride / sizeof(float),
            reinterpret_cast<uint32_t *>(data + status_offset));
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) return rkva_small_failure("unpack", "launch/cudaGetLastError", s.device, launch_status, grid.x, n);
        gpu.mark(1, s.stream);
    }
    if (!retire.finish()) return false;
    const bool ok = rkva_small_status_ok(s.small_rx, status_offset, grid.x, "unpack", s.device);
    gpu.interval("cuda_small_unpack_kernel_gpu", 0, 1, payload + size_t(grid.x) * sizeof(uint32_t), !ok);
    if (ok) timing.event.flags &= ~1u;
    return ok;
}

static bool rkva_pack_impl(rkva_cuda_remote_scratch & s, const void * src_q, bool is_f16,
                          size_t n_elements, uint8_t * dst_pinned_fp8,
                          cudaStream_t user_stream, cudaEvent_t producer_event,
                          cudaEvent_t event_done) {
    ggml_remote_attn_trace_scope call_timing("cuda_pack_host");
    call_timing.event.flags |= 0x40000001u;
    if (!src_q || !dst_pinned_fp8 || !n_elements) return false;
    const size_t elem_sz = is_f16 ? sizeof(half) : sizeof(float);
    if (reinterpret_cast<uintptr_t>(src_q) % elem_sz) return false;
    const int src_device = rkva_pointer_device(src_q);
    if (src_device < -1 || (src_device >= 0 && src_device != s.device) ||
        rkva_pointer_device(dst_pinned_fp8) != -1) return false;
    const bool is_dev = src_device >= 0;
    size_t input_extent, in_bytes, needed;
    if (!rkva_size_mul(n_elements, elem_sz, input_extent)) return false;
    call_timing.event.bytes = input_extent;
    in_bytes = is_dev ? 0 : input_extent;
    if (!rkva_size_add(in_bytes, n_elements, needed)) return false;
    dim3 grid;
    if (!rkva_grid(n_elements, grid) || !rkva_scratch_ensure(s, needed)) return false;
    cudaStream_t stream = user_stream ? user_stream : s.stream;
    unsigned stream_flags;
    if (cudaStreamGetFlags(stream, &stream_flags) != cudaSuccess) return false;
    rkva_stream_retirement retire{s, stream, false, "cuda_pack_retire_host"};
    rkva_cuda_timing_trace gpu_timing(s, "cuda_pack");
    {
        ggml_remote_attn_trace_scope enqueue_timing("cuda_pack_enqueue_host");
        enqueue_timing.event.bytes = n_elements;
        if (gpu_timing.active) retire.pending = true;
        gpu_timing.mark(0, stream);
        if (is_dev) {
            if (producer_event) {
                retire.pending = true;
                if (cudaStreamWaitEvent(stream, producer_event, 0) != cudaSuccess) return false;
            } else if (!user_stream && cudaDeviceSynchronize() != cudaSuccess) {
                s.poisoned = true;
                return false;
            }
            // A supplied legacy user_stream must itself be the producer stream or
            // already contain a wait for the producer. v2 defaults to the safe sync.
        }
        gpu_timing.mark(1, stream);
        const void * dev_src = is_dev ? src_q : s.dev_data;
        auto * dev_dst = static_cast<uint8_t *>(s.dev_data) + in_bytes;
        retire.pending = true;
        if (!is_dev && cudaMemcpyAsync(s.dev_data, src_q, in_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
        gpu_timing.mark(2, stream);
        if (cudaMemsetAsync(s.dev_err, 0, sizeof(int), stream) != cudaSuccess) return false;
        gpu_timing.mark(3, stream);
        if (is_f16) {
            rkva_cuda_fp8_pack_q_f16_kernel<<<grid, 256, 0, stream>>>(
                static_cast<const half *>(dev_src), dev_dst, n_elements, s.dev_err);
        } else {
            rkva_cuda_fp8_pack_q_f32_kernel<<<grid, 256, 0, stream>>>(
                static_cast<const float *>(dev_src), dev_dst, n_elements, s.dev_err);
        }
        if (cudaGetLastError() != cudaSuccess) return false;
        gpu_timing.mark(4, stream);
        if (cudaMemcpyAsync(dst_pinned_fp8, dev_dst, n_elements, cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
        gpu_timing.mark(5, stream);
        if (cudaMemcpyAsync(s.host_err, s.dev_err, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
        gpu_timing.mark(6, stream);
        if (event_done && cudaEventRecord(event_done, stream) != cudaSuccess) return false;
    }
    const bool retired = retire.finish();
    if (retired) {
        const uint64_t status = uint64_t(*s.host_err);
        gpu_timing.interval("cuda_pack_producer_gpu", 0, 1, 0, status);
        if (in_bytes) gpu_timing.interval("cuda_pack_input_h2d_gpu", 1, 2, in_bytes, status);
        gpu_timing.interval("cuda_pack_status_reset_gpu", 2, 3, sizeof(int), status);
        gpu_timing.interval("cuda_pack_kernel_gpu", 3, 4, input_extent, status);
        gpu_timing.interval("cuda_pack_d2h_gpu", 4, 5, n_elements, status);
        gpu_timing.interval("cuda_pack_status_d2h_gpu", 5, 6, sizeof(int), status);
        gpu_timing.interval("cuda_pack_total_gpu", 0, 6, in_bytes + n_elements + sizeof(int), status, true);
    }
    const bool ok = retired && *s.host_err == 0;
    if (ok) call_timing.event.flags &= ~1u;
    return ok;
}

static bool rkva_unpack_impl(rkva_cuda_remote_scratch & s, const uint8_t * src_pinned_fp8,
                            const float * src_lse, size_t nq, size_t out_width, size_t n_head,
                            void * dst_out, size_t dst_stride_bytes, cudaStream_t user_stream,
                            cudaEvent_t event_done) {
    ggml_remote_attn_trace_scope call_timing("cuda_unpack_host");
    call_timing.event.flags |= 0x40000001u;
    if (!src_pinned_fp8 || !dst_out || !nq || !out_width || (n_head && !src_lse) ||
        reinterpret_cast<uintptr_t>(dst_out) % alignof(float) ||
        dst_stride_bytes % sizeof(float)) return false;
    const int dst_device = rkva_pointer_device(dst_out);
    if (dst_device < -1 || (dst_device >= 0 && dst_device != s.device) ||
        rkva_pointer_device(src_pinned_fp8) != -1 ||
        (src_lse && rkva_pointer_device(src_lse) != -1)) return false;
    const bool is_dev = dst_device >= 0;
    size_t fp8_bytes, heads, lse_bytes, col_total, row_bytes, total, out_bytes;
    size_t dst_extent, lse_offset, out_offset, needed;
    if (!rkva_size_mul(nq, out_width, fp8_bytes) ||
        !rkva_size_mul(nq, n_head, heads) || !rkva_size_mul(heads, sizeof(float), lse_bytes) ||
        !rkva_size_add(out_width, n_head, col_total) ||
        !rkva_size_mul(col_total, sizeof(float), row_bytes) || dst_stride_bytes < row_bytes ||
        !rkva_size_mul(nq, col_total, total) || !rkva_size_mul(total, sizeof(float), out_bytes) ||
        !rkva_size_mul(nq - 1, dst_stride_bytes, dst_extent) || !rkva_size_add(dst_extent, row_bytes, dst_extent) ||
        !rkva_align_float(fp8_bytes, lse_offset) || !rkva_size_add(lse_offset, lse_bytes, out_offset) ||
        !rkva_size_add(out_offset, is_dev ? 0 : out_bytes, needed)) return false;
    dim3 grid;
    if (!rkva_grid(total, grid) || !rkva_scratch_ensure(s, needed)) return false;
    cudaStream_t stream = user_stream ? user_stream : s.stream;
    unsigned stream_flags;
    if (cudaStreamGetFlags(stream, &stream_flags) != cudaSuccess) return false;
    call_timing.event.bytes = fp8_bytes + lse_bytes;
    rkva_stream_retirement retire{s, stream, false, "cuda_unpack_retire_host"};
    rkva_cuda_timing_trace gpu_timing(s, "cuda_unpack");
    auto * dev_fp8 = static_cast<uint8_t *>(s.dev_data);
    auto * dev_lse = reinterpret_cast<float *>(dev_fp8 + lse_offset);
    auto * dev_out = is_dev ? static_cast<float *>(dst_out) : reinterpret_cast<float *>(dev_fp8 + out_offset);
    {
        ggml_remote_attn_trace_scope enqueue_timing("cuda_unpack_enqueue_host");
        enqueue_timing.event.bytes = fp8_bytes + lse_bytes;
        if (gpu_timing.active) retire.pending = true;
        gpu_timing.mark(0, stream);
        retire.pending = true;
        if (cudaMemcpyAsync(dev_fp8, src_pinned_fp8, fp8_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
        gpu_timing.mark(1, stream);
        if (lse_bytes && cudaMemcpyAsync(dev_lse, src_lse, lse_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
        gpu_timing.mark(2, stream);
        if (cudaMemsetAsync(s.dev_err, 0, sizeof(int), stream) != cudaSuccess) return false;
        gpu_timing.mark(3, stream);
        rkva_cuda_fp8_unpack_out_kernel<<<grid, 256, 0, stream>>>(
            dev_fp8, lse_bytes ? dev_lse : nullptr, dev_out, nq, out_width, n_head,
            is_dev ? dst_stride_bytes / sizeof(float) : col_total, s.dev_err);
        if (cudaGetLastError() != cudaSuccess) return false;
        gpu_timing.mark(4, stream);
        if (!is_dev && cudaMemcpy2DAsync(dst_out, dst_stride_bytes, dev_out, row_bytes,
                                    row_bytes, nq, cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
        gpu_timing.mark(5, stream);
        if (cudaMemcpyAsync(s.host_err, s.dev_err, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
        gpu_timing.mark(6, stream);
        if (event_done && cudaEventRecord(event_done, stream) != cudaSuccess) return false;
    }
    const bool retired = retire.finish();
    if (retired) {
        const uint64_t status = uint64_t(*s.host_err);
        gpu_timing.interval("cuda_unpack_fp8_h2d_gpu", 0, 1, fp8_bytes, status);
        if (lse_bytes) gpu_timing.interval("cuda_unpack_lse_h2d_gpu", 1, 2, lse_bytes, status);
        gpu_timing.interval("cuda_unpack_status_reset_gpu", 2, 3, sizeof(int), status);
        gpu_timing.interval("cuda_unpack_kernel_gpu", 3, 4, out_bytes, status);
        if (!is_dev) gpu_timing.interval("cuda_unpack_output_d2h_gpu", 4, 5, out_bytes, status);
        gpu_timing.interval("cuda_unpack_status_d2h_gpu", 5, 6, sizeof(int), status);
        gpu_timing.interval("cuda_unpack_total_gpu", 0, 6,
            fp8_bytes + lse_bytes + (is_dev ? 0 : out_bytes) + sizeof(int), status, true);
    }
    const bool ok = retired && *s.host_err == 0;
    if (ok) call_timing.event.flags &= ~1u;
    return ok;
}

static void * rkva_create_context(int device) {
    rkva_device_guard guard(device);
    if (!guard.ok) return nullptr;
    auto * s = new (std::nothrow) rkva_cuda_remote_scratch;
    if (s) s->device = device;
    return s;
}

// Explicit lifetime-only extension for external pinned buffers. In particular,
// a numerical helper failure may have retired successfully: poison and output
// validity are separate from this barrier, and are never reset by it.
static bool rkva_retire_context(void * context) {
    auto * s = static_cast<rkva_cuda_remote_scratch *>(context);
    if (!s) return false;
    std::lock_guard<std::mutex> lock(s->mutex);
    ggml_remote_attn_trace_scope timing("cuda_retirement_barrier");
    timing.event.flags |= 1u;
    rkva_device_guard guard(s->device);
    if (!guard.ok) {
        s->poisoned = true;
        return false;
    }
    if (s->stream) {
        if (cudaStreamSynchronize(s->stream) != cudaSuccess) {
            s->retirement_confirmed = false;
            s->poisoned = true;
            return false;
        }
        s->retirement_confirmed = true;
    }
    // No stream exists on a fresh context (or on a failed pre-enqueue setup).
    // Never turn an unknown state into proof without a stream synchronization.
    const bool retired = s->retirement_confirmed;
    timing.event.value = uint64_t(retired);
    if (retired) timing.event.flags &= ~1u;
    return retired;
}

static void rkva_destroy_context(void * context) {
    auto * s = static_cast<rkva_cuda_remote_scratch *>(context);
    if (!s) return;
    {
        std::lock_guard<std::mutex> lock(s->mutex);
        rkva_device_guard guard(s->device);
        // On an uncertain device failure, quarantine this bounded context rather
        // than free mapped pages or stream resources a kernel may still touch.
        // destroy_context is void in v2, so this is deliberately a fail-closed
        // leak on device failure, never a claim that a failed sync retired work.
        if (!guard.ok || (s->stream && cudaStreamSynchronize(s->stream) != cudaSuccess)) {
            std::fprintf(stderr, "remote-attn CUDA: context quarantined after failed retirement (device %d)\n", s->device);
            return;
        }
        if (guard.ok) {
            for (auto event : s->timing_events) if (event) cudaEventDestroy(event);
            if (s->dev_data) cudaFree(s->dev_data);
            if (s->dev_err) cudaFree(s->dev_err);
            if (s->host_err) cudaFreeHost(s->host_err);
            if (s->small_tx.host) cudaFreeHost(s->small_tx.host);
            if (s->small_rx.host) cudaFreeHost(s->small_rx.host);
            if (s->stream) cudaStreamDestroy(s->stream);
        }
    }
    delete s;
}

static bool rkva_pack_context(void * context, const void * src_q, bool is_f16, size_t n_elements,
                              uint8_t * dst_pinned_fp8, void * producer_event) {
    auto * s = static_cast<rkva_cuda_remote_scratch *>(context);
    if (!s) return false;
    std::lock_guard<std::mutex> lock(s->mutex);
    rkva_device_guard guard(s->device);
    return guard.ok && rkva_pack_impl(*s, src_q, is_f16, n_elements, dst_pinned_fp8,
                                      nullptr, static_cast<cudaEvent_t>(producer_event), nullptr);
}

static bool rkva_unpack_context(void * context, const uint8_t * src_pinned_fp8, const float * src_lse,
                                size_t nq, size_t out_width, size_t n_head, void * dst_out, size_t stride) {
    auto * s = static_cast<rkva_cuda_remote_scratch *>(context);
    if (!s) return false;
    std::lock_guard<std::mutex> lock(s->mutex);
    rkva_device_guard guard(s->device);
    return guard.ok && rkva_unpack_impl(*s, src_pinned_fp8, src_lse, nq, out_width, n_head,
                                        dst_out, stride, nullptr, nullptr);
}

static bool rkva_pack_small_context(void * context, const void * src, bool is_f16,
                                   size_t nq, size_t width, uint8_t * dst, void * producer_event) {
    auto * s = static_cast<rkva_cuda_remote_scratch *>(context);
    if (!s) return rkva_small_failure("pack", "null_context", -1, -1, nq, width);
    std::lock_guard<std::mutex> lock(s->mutex);
    rkva_device_guard guard(s->device);
    if (!guard.ok) return rkva_small_failure("pack", guard.failed_api, s->device, guard.status, nq, width);
    return rkva_pack_small_impl(*s, src, is_f16, nq, width, dst, static_cast<cudaEvent_t>(producer_event));
}

static bool rkva_unpack_small_context(void * context, const uint8_t * src, const float * lse,
                                     size_t nq, size_t width, size_t heads, void * dst, size_t stride) {
    auto * s = static_cast<rkva_cuda_remote_scratch *>(context);
    if (!s) return rkva_small_failure("unpack", "null_context", -1, -1, nq, width);
    std::lock_guard<std::mutex> lock(s->mutex);
    rkva_device_guard guard(s->device);
    if (!guard.ok) return rkva_small_failure("unpack", guard.failed_api, s->device, guard.status, nq, width);
    return rkva_unpack_small_impl(*s, src, lse, nq, width, heads, dst, stride);
}

// Legacy ABI: use one context per CUDA device, preserving its original
// serialized call semantics without device-switch allocation churn.
struct rkva_legacy_contexts {
    std::mutex mutex;
    std::unordered_map<int, void *> contexts;
    ~rkva_legacy_contexts() {
        for (auto & entry : contexts) rkva_destroy_context(entry.second);
    }
};
static rkva_legacy_contexts g_legacy;

static rkva_cuda_remote_scratch * rkva_legacy_context(const void * device_ptr) {
    int device = rkva_pointer_device(device_ptr);
    if (device < -1) return nullptr;
    if (device == -1 && cudaGetDevice(&device) != cudaSuccess) return nullptr;
    auto & context = g_legacy.contexts[device];
    if (!context) context = rkva_create_context(device);
    return static_cast<rkva_cuda_remote_scratch *>(context);
}

bool ggml_cuda_remote_attn_fp8_pack_q(const void * src_q, bool is_f16, size_t n_elements,
                                     uint8_t * dst_pinned_fp8, void * user_stream, void * event_done) {
    std::lock_guard<std::mutex> lock(g_legacy.mutex);
    auto * s = rkva_legacy_context(src_q);
    if (!s) return false;
    rkva_device_guard guard(s->device);
    return guard.ok && rkva_pack_impl(*s, src_q, is_f16, n_elements, dst_pinned_fp8,
                                      static_cast<cudaStream_t>(user_stream), nullptr,
                                      static_cast<cudaEvent_t>(event_done));
}

bool ggml_cuda_remote_attn_fp8_unpack_out(const uint8_t * src_pinned_fp8, const float * src_lse,
                                         size_t nq, size_t out_width, size_t n_head, void * dst_out,
                                         size_t stride, void * user_stream, void * event_done) {
    std::lock_guard<std::mutex> lock(g_legacy.mutex);
    auto * s = rkva_legacy_context(dst_out);
    if (!s) return false;
    rkva_device_guard guard(s->device);
    return guard.ok && rkva_unpack_impl(*s, src_pinned_fp8, src_lse, nq, out_width, n_head,
                                        dst_out, stride, static_cast<cudaStream_t>(user_stream),
                                        static_cast<cudaEvent_t>(event_done));
}

void * ggml_cuda_remote_attn_alloc_pinned(size_t bytes) {
    if (!bytes) return nullptr;
    void * ptr = nullptr;
    if (cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return ptr;
}

void ggml_cuda_remote_attn_free_pinned(void * ptr) {
    if (ptr) cudaFreeHost(ptr);
}

bool ggml_cuda_remote_attn_stream_sync(void * stream) {
    if (!stream) return false;
    return cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) == cudaSuccess;
}

bool ggml_cuda_remote_attn_event_sync(void * event) {
    if (!event) return false;
    return cudaEventSynchronize(static_cast<cudaEvent_t>(event)) == cudaSuccess;
}

void * ggml_cuda_remote_attn_create_stream(void) {
    cudaStream_t s = nullptr;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return static_cast<void *>(s);
}

void ggml_cuda_remote_attn_destroy_stream(void * stream) {
    if (stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

void * ggml_cuda_remote_attn_create_event(void) {
    cudaEvent_t e = nullptr;
    if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return static_cast<void *>(e);
}

void ggml_cuda_remote_attn_destroy_event(void * event) {
    if (event) cudaEventDestroy(static_cast<cudaEvent_t>(event));
}

static const ggml_remote_attn_cuda_fp8_interface s_cuda_remote_attn_fp8_interface = {
    ggml_cuda_remote_attn_fp8_pack_q,
    ggml_cuda_remote_attn_fp8_unpack_out,
    ggml_cuda_remote_attn_alloc_pinned,
    ggml_cuda_remote_attn_free_pinned,
    ggml_cuda_remote_attn_stream_sync,
    ggml_cuda_remote_attn_event_sync,
    ggml_cuda_remote_attn_create_stream,
    ggml_cuda_remote_attn_destroy_stream,
    ggml_cuda_remote_attn_create_event,
    ggml_cuda_remote_attn_destroy_event,
};

const struct ggml_remote_attn_cuda_fp8_interface * ggml_backend_cuda_remote_attn_fp8_interface(void) {
    return &s_cuda_remote_attn_fp8_interface;
}

static const ggml_remote_attn_cuda_fp8_interface_v2 s_cuda_remote_attn_fp8_interface_v2 = {
    rkva_pointer_device,
    rkva_create_context,
    rkva_destroy_context,
    rkva_pack_context,
    rkva_unpack_context,
    ggml_cuda_remote_attn_alloc_pinned,
    ggml_cuda_remote_attn_free_pinned,
};

const struct ggml_remote_attn_cuda_fp8_interface_v2 * ggml_backend_cuda_remote_attn_fp8_interface_v2(void) {
    return &s_cuda_remote_attn_fp8_interface_v2;
}

static const ggml_remote_attn_cuda_fp8_interface_v3 s_cuda_remote_attn_fp8_interface_v3 = {
    &s_cuda_remote_attn_fp8_interface_v2,
    rkva_supports_small,
    rkva_pack_small_context,
    rkva_unpack_small_context,
};

const struct ggml_remote_attn_cuda_fp8_interface_v3 * ggml_backend_cuda_remote_attn_fp8_interface_v3(void) {
    return &s_cuda_remote_attn_fp8_interface_v3;
}

static const ggml_remote_attn_cuda_retirement_interface_v1 s_cuda_remote_attn_retirement_interface_v1 = {
    &s_cuda_remote_attn_fp8_interface_v2,
    rkva_retire_context,
};

const struct ggml_remote_attn_cuda_retirement_interface_v1 * ggml_backend_cuda_remote_attn_retirement_interface_v1(void) {
    return &s_cuda_remote_attn_retirement_interface_v1;
}

#endif // !GGML_USE_HIP && !GGML_USE_MUSA
