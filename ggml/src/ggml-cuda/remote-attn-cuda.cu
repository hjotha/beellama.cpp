#include "common.cuh"
#include "remote-attn-cuda.cuh"

#include <algorithm>
#include <cstdint>
#include <mutex>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

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

struct rkva_cuda_remote_scratch {
    void * dev_data = nullptr;
    size_t dev_capacity = 0;
    int * dev_err = nullptr;
    int * host_err = nullptr;
    cudaStream_t stream = nullptr;
    int device = -1;
};

static rkva_cuda_remote_scratch g_scratch;
static std::mutex g_scratch_mutex;

static bool rkva_scratch_ensure(rkva_cuda_remote_scratch & s, int dev, size_t needed) {
    if (s.device != dev) {
        if (s.device >= 0) {
            cudaSetDevice(s.device);
            if (s.dev_data) cudaFree(s.dev_data);
            if (s.dev_err) cudaFree(s.dev_err);
            if (s.host_err) cudaFreeHost(s.host_err);
            if (s.stream) cudaStreamDestroy(s.stream);
            cudaSetDevice(dev);
        }
        s = rkva_cuda_remote_scratch{};
        s.device = dev;
    }
    if (!s.stream) {
        if (cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking) != cudaSuccess) return false;
    }
    if (!s.dev_err) {
        if (cudaMalloc(&s.dev_err, sizeof(int)) != cudaSuccess) return false;
    }
    if (!s.host_err) {
        if (cudaHostAlloc(&s.host_err, sizeof(int), cudaHostAllocPortable) != cudaSuccess) return false;
    }
    if (s.dev_capacity < needed) {
        if (s.dev_data) cudaFree(s.dev_data);
        size_t alloc_sz = std::max<size_t>(needed, 4u << 20); // 4 MiB minimum
        if (cudaMalloc(&s.dev_data, alloc_sz) != cudaSuccess) {
            s.dev_data = nullptr;
            s.dev_capacity = 0;
            return false;
        }
        s.dev_capacity = alloc_sz;
    }
    return true;
}

static bool rkva_is_device_pointer(const void * ptr, int device) {
    if (!ptr) return false;
#if !defined(GGML_USE_MUSA)
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, ptr) == cudaSuccess) {
#if defined(GGML_USE_HIP)
        if (attr.memoryType == hipMemoryTypeDevice && attr.device == device) return true;
#else
        if (attr.type == cudaMemoryTypeDevice && attr.device == device) return true;
#endif
    } else {
        cudaGetLastError();
    }
#endif
    return false;
}

bool ggml_cuda_remote_attn_fp8_pack_q(
        const void * src_q,
        bool is_f16,
        size_t n_elements,
        uint8_t * dst_pinned_fp8,
        void * user_stream,
        void * user_event_done) {
    if (!src_q || !dst_pinned_fp8 || !n_elements) return false;

    int device;
    if (cudaGetDevice(&device) != cudaSuccess) return false;

    std::lock_guard<std::mutex> guard(g_scratch_mutex);

    const bool is_dev = rkva_is_device_pointer(src_q, device);
    const size_t elem_sz = is_f16 ? sizeof(half) : sizeof(float);
    const size_t in_bytes = is_dev ? 0 : n_elements * elem_sz;
    const size_t out_bytes = n_elements * sizeof(uint8_t);
    const size_t needed = in_bytes + out_bytes;

    if (!rkva_scratch_ensure(g_scratch, device, needed)) return false;

    cudaStream_t stream = user_stream ? static_cast<cudaStream_t>(user_stream) : g_scratch.stream;

    const void * dev_src = nullptr;
    uint8_t * dev_dst = nullptr;

    if (is_dev) {
        dev_src = src_q;
        dev_dst = static_cast<uint8_t *>(g_scratch.dev_data);
    } else {
        dev_src = g_scratch.dev_data;
        dev_dst = static_cast<uint8_t *>(g_scratch.dev_data) + in_bytes;
        if (cudaMemcpyAsync(const_cast<void *>(dev_src), src_q, in_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return false;
        }
    }

    if (cudaMemsetAsync(g_scratch.dev_err, 0, sizeof(int), stream) != cudaSuccess) return false;

    dim3 block(256);
    dim3 grid(uint32_t((n_elements + block.x - 1) / block.x));

    if (is_f16) {
        rkva_cuda_fp8_pack_q_f16_kernel<<<grid, block, 0, stream>>>(
            static_cast<const half *>(dev_src), dev_dst, n_elements, g_scratch.dev_err);
    } else {
        rkva_cuda_fp8_pack_q_f32_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float *>(dev_src), dev_dst, n_elements, g_scratch.dev_err);
    }

    if (cudaMemcpyAsync(dst_pinned_fp8, dev_dst, out_bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpyAsync(g_scratch.host_err, g_scratch.dev_err, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        return false;
    }

    if (user_event_done) {
        if (cudaEventRecord(static_cast<cudaEvent_t>(user_event_done), stream) != cudaSuccess) return false;
    }

    if (cudaStreamSynchronize(stream) != cudaSuccess) return false;

    return (*g_scratch.host_err == 0);
}

bool ggml_cuda_remote_attn_fp8_unpack_out(
        const uint8_t * src_pinned_fp8,
        const float   * src_lse,
        size_t nq,
        size_t out_width,
        size_t n_head,
        void * dst_out,
        size_t dst_stride_bytes,
        void * user_stream,
        void * user_event_done) {
    if (!src_pinned_fp8 || !dst_out || !nq || !out_width) return false;

    int device;
    if (cudaGetDevice(&device) != cudaSuccess) return false;

    std::lock_guard<std::mutex> guard(g_scratch_mutex);

    const bool is_dev = rkva_is_device_pointer(dst_out, device);
    const size_t fp8_bytes = nq * out_width * sizeof(uint8_t);
    const size_t lse_bytes = src_lse ? (nq * n_head * sizeof(float)) : 0;
    const size_t col_total = out_width + n_head;
    const size_t f32_out_bytes = is_dev ? 0 : (nq * col_total * sizeof(float));
    const size_t needed = fp8_bytes + lse_bytes + f32_out_bytes;

    if (!rkva_scratch_ensure(g_scratch, device, needed)) return false;

    cudaStream_t stream = user_stream ? static_cast<cudaStream_t>(user_stream) : g_scratch.stream;

    uint8_t * dev_fp8 = static_cast<uint8_t *>(g_scratch.dev_data);
    float * dev_lse = reinterpret_cast<float *>(dev_fp8 + fp8_bytes);
    float * dev_f32_out = is_dev ? static_cast<float *>(dst_out)
                                 : reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(dev_lse) + lse_bytes);

    if (cudaMemcpyAsync(dev_fp8, src_pinned_fp8, fp8_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return false;
    }
    if (src_lse && lse_bytes > 0) {
        if (cudaMemcpyAsync(dev_lse, src_lse, lse_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return false;
        }
    }

    if (cudaMemsetAsync(g_scratch.dev_err, 0, sizeof(int), stream) != cudaSuccess) return false;

    const size_t dst_stride_floats = is_dev ? (dst_stride_bytes / sizeof(float)) : col_total;
    const size_t total_elements = nq * col_total;
    dim3 block(256);
    dim3 grid(uint32_t((total_elements + block.x - 1) / block.x));

    rkva_cuda_fp8_unpack_out_kernel<<<grid, block, 0, stream>>>(
        dev_fp8, src_lse ? dev_lse : nullptr, dev_f32_out,
        nq, out_width, n_head, dst_stride_floats, g_scratch.dev_err);

    if (!is_dev) {
        for (size_t row = 0; row < nq; ++row) {
            if (cudaMemcpyAsync(static_cast<uint8_t *>(dst_out) + row * dst_stride_bytes,
                                dev_f32_out + row * col_total,
                                col_total * sizeof(float),
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
                return false;
            }
        }
    }

    if (cudaMemcpyAsync(g_scratch.host_err, g_scratch.dev_err, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        return false;
    }

    if (user_event_done) {
        if (cudaEventRecord(static_cast<cudaEvent_t>(user_event_done), stream) != cudaSuccess) return false;
    }

    if (cudaStreamSynchronize(stream) != cudaSuccess) return false;

    return (*g_scratch.host_err == 0);
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

#endif // !GGML_USE_HIP && !GGML_USE_MUSA
