// This translation unit must retain gradual underflow and precise division.
// CMake overrides the backend's -use_fast_math for this source only.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#    include <cuda_runtime.h>
#    include <cuda_fp16.h>
#    include "ggml.h"
#    include "graph-guard.h"
#    include "kvarn-native-producer.cuh"

#    include <mutex>

static bool low4_scratch_grow(void ** ptr, size_t & capacity, size_t need) {
    if (capacity >= need) {
        return true;
    }
    if (*ptr != nullptr) {
        const cudaError_t free_status = cudaFree(*ptr);
        *ptr     = nullptr;
        capacity = 0;
        if (free_status != cudaSuccess) {
            return false;
        }
    }
    if (need == 0 || cudaMalloc(ptr, need) != cudaSuccess) {
        return false;
    }
    ggml_cuda_graph_guard::note_device_reservation();
    capacity = need;
    return true;
}
#    if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#        include "native-low4-reference.cuh"

static __global__ void low4_native_prepare_kernel(int           format,
                                                  uint32_t      flags,
                                                  const float * input,
                                                  uint8_t *     output,
                                                  size_t        heads,
                                                  size_t        record,
                                                  bool          key,
                                                  int *         invalid) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < heads && !rkva_native_low4::encode(format, input + i * 256, output + i * record, key, flags)) {
        atomicExch(invalid, 1);
    }
}

#    endif

bool ggml_cuda_low4_native_prepare(int           format,
                                   uint32_t      flags,
                                   const float * keys,
                                   const float * values,
                                   uint32_t      count,
                                   uint32_t      heads,
                                   void *        packed_keys,
                                   void *        packed_values) {
    if (!keys || !values || !packed_keys || !packed_values || !count || count > 512 || !heads || heads > 32) {
        return false;
    }
    // This ABI is CUDA-only until other backends have numerical qualification.
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return false;
#    else
    const size_t record = format == GGML_TYPE_Q4_1     ? 160 :
                          format == GGML_TYPE_IQ4_NL   ? 144 :
                          format == GGML_TYPE_SAW_INT4 ? 136 :
                          format == GGML_TYPE_TQNC_K   ? 132 :
                                                         0;
    if (!record || (flags & ~1u) || (flags && format != GGML_TYPE_SAW_INT4)) {
        return false;
    }

    struct scratch_t {
        void *       data   = nullptr;
        size_t       bytes  = 0;
        int          device = -1;
        cudaStream_t stream = nullptr;
    };

    static scratch_t            scratch;
    static std::mutex           mutex;
    std::lock_guard<std::mutex> lock(mutex);
    int                         device;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    if (scratch.device != device) {
        if (scratch.device >= 0) {
            if (cudaSetDevice(scratch.device) != cudaSuccess) {
                return false;
            }
            const cudaError_t free_status = scratch.data ? cudaFree(scratch.data) : cudaSuccess;
            const cudaError_t stream_status = scratch.stream ? cudaStreamDestroy(scratch.stream) : cudaSuccess;
            // The old allocation/stream are no longer reusable, even if restoring
            // the caller's device fails. A later call must start from empty state.
            scratch = scratch_t{};
            const cudaError_t restore_status = cudaSetDevice(device);
            if (free_status != cudaSuccess || stream_status != cudaSuccess || restore_status != cudaSuccess) {
                return false;
            }
        }
        scratch        = scratch_t{};
        scratch.device = device;
    }
    if (!scratch.stream && cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking) != cudaSuccess) {
        return false;
    }
    const size_t blocks       = size_t(count) * heads;
    const size_t input_bytes  = blocks * 256 * sizeof(float);
    const size_t output_bytes = blocks * record;
    if (!low4_scratch_grow(&scratch.data, scratch.bytes, 2 * (input_bytes + output_bytes) + sizeof(int))) {
        return false;
    }
    auto *     input   = static_cast<float *>(scratch.data);
    auto *     packed  = reinterpret_cast<uint8_t *>(reinterpret_cast<uint8_t *>(input) + 2 * input_bytes);
    auto *     invalid = reinterpret_cast<int *>(packed + 2 * output_bytes);
    const auto upload  = [&](void * dst, const void * src, size_t bytes) {
        cudaMemcpyKind kind = cudaMemcpyHostToDevice;
#        if !defined(GGML_USE_MUSA)
        cudaPointerAttributes attr{};
        if (cudaPointerGetAttributes(&attr, src) == cudaSuccess) {
#            if defined(GGML_USE_HIP)
            if (attr.memoryType == hipMemoryTypeDevice && attr.device == device)
                kind = hipMemcpyDeviceToDevice;
#            else
            if (attr.type == cudaMemoryTypeDevice) {
                if (attr.device != device)
                    return false;
                kind = cudaMemcpyDeviceToDevice;
            }
#            endif
        } else {
            cudaGetLastError();
        }
#        endif
        return cudaMemcpyAsync(dst, src, bytes, kind, scratch.stream) == cudaSuccess;
    };
    if (!upload(input, keys, input_bytes) || !upload(input + blocks * 256, values, input_bytes)) {
        cudaStreamSynchronize(scratch.stream);
        return false;
    }
    if (cudaMemsetAsync(invalid, 0, sizeof(int), scratch.stream) != cudaSuccess) {
        cudaStreamSynchronize(scratch.stream);
        return false;
    }
    const uint32_t grid = uint32_t((blocks + 31) / 32);
    low4_native_prepare_kernel<<<grid, 32, 0, scratch.stream>>>(format, flags, input, packed, blocks, record, true,
                                                                invalid);
    if (cudaGetLastError() != cudaSuccess) {
        cudaStreamSynchronize(scratch.stream);
        return false;
    }
    low4_native_prepare_kernel<<<grid, 32, 0, scratch.stream>>>(format, flags, input + blocks * 256,
                                                                packed + output_bytes, blocks, record, false, invalid);
    if (cudaGetLastError() != cudaSuccess) {
        cudaStreamSynchronize(scratch.stream);
        return false;
    }
    int        rejected = 1;
    const bool ok =
        cudaMemcpyAsync(&rejected, invalid, sizeof(int), cudaMemcpyDeviceToHost, scratch.stream) == cudaSuccess &&
        cudaMemcpyAsync(packed_keys, packed, output_bytes, cudaMemcpyDeviceToHost, scratch.stream) == cudaSuccess &&
        cudaMemcpyAsync(packed_values, packed + output_bytes, output_bytes, cudaMemcpyDeviceToHost, scratch.stream) ==
            cudaSuccess;
    return cudaStreamSynchronize(scratch.stream) == cudaSuccess && ok && !rejected;
#    endif
}

#endif
