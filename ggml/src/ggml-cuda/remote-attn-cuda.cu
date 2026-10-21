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
    explicit rkva_device_guard(int device) {
        ok = cudaGetDevice(&previous) == cudaSuccess &&
             (previous == device || cudaSetDevice(device) == cudaSuccess);
    }
    ~rkva_device_guard() {
        if (ok && previous >= 0) cudaSetDevice(previous);
    }
};

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

static int rkva_pointer_device(const void * ptr) {
    if (!ptr) return -2;
    cudaPointerAttributes attr{};
    const cudaError_t status = cudaPointerGetAttributes(&attr, ptr);
    if (status == cudaErrorInvalidValue) {
        cudaGetLastError(); // Older runtimes report ordinary host allocations this way.
        return -1;
    }
    if (status != cudaSuccess) return -2;
    if (attr.type == cudaMemoryTypeDevice) return attr.device;
    if (attr.type == cudaMemoryTypeHost || attr.type == cudaMemoryTypeUnregistered) return -1;
    return -2; // Managed memory requires a separate residency/coherency contract.
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
    bool finish() {
        if (!pending) return true;
        pending = false;
        ggml_remote_attn_trace_scope timing(trace_phase);
        if (cudaStreamSynchronize(stream) == cudaSuccess) return true;
        timing.event.flags |= 1u;
        scratch.poisoned = true;
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

static void rkva_destroy_context(void * context) {
    auto * s = static_cast<rkva_cuda_remote_scratch *>(context);
    if (!s) return;
    {
        std::lock_guard<std::mutex> lock(s->mutex);
        rkva_device_guard guard(s->device);
        if (guard.ok) {
            if (s->stream) cudaStreamSynchronize(s->stream);
            for (auto event : s->timing_events) if (event) cudaEventDestroy(event);
            if (s->dev_data) cudaFree(s->dev_data);
            if (s->dev_err) cudaFree(s->dev_err);
            if (s->host_err) cudaFreeHost(s->host_err);
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

#endif // !GGML_USE_HIP && !GGML_USE_MUSA
