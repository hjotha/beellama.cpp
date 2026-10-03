#pragma once

#include "fattn-mma-kvarn-decode-decl.cuh"

// This bit-independent reduction is instantiated once per head dimension in a
// dedicated translation unit rather than once per compiled K/V bit pair.
static constexpr int GGML_CUDA_FATTN_KVARN_DECODE_COMBINE_THREADS = 256;

template<int D>
static __global__ void ggml_cuda_fattn_kvarn_decode_combine_kernel(
        const float * partial,
        const float2 * partial_meta,
        float * dst,
        float2 * dst_meta,
        float * lse_out,
        int n_splits,
        int n_q,
        int n_q_heads,
        int n_stream,
        int64_t nb11,
        int64_t nb12,
        int64_t nb13,
        int64_t lse_nb0,
        int64_t lse_nb1,
        int64_t lse_nb2) {
    const int q_head = blockIdx.x;
    const int q_index = blockIdx.y;
    const int stream = blockIdx.z;
    const int tid = threadIdx.x;

    __shared__ float reduce_sh[GGML_CUDA_FATTN_KVARN_DECODE_COMBINE_THREADS];
    extern __shared__ float split_weights[];

    float local_max = -FLT_MAX / 2.0f;
    for (int split = tid; split < n_splits; split += blockDim.x) {
        const float2 meta = partial_meta[(((size_t) stream * n_q + q_index) * n_q_heads + q_head) * n_splits + split];
        if (meta.y > 0.0f) {
            local_max = fmaxf(local_max, meta.x);
        }
    }
    reduce_sh[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_sh[tid] = fmaxf(reduce_sh[tid], reduce_sh[tid + stride]);
        }
        __syncthreads();
    }
    const float m = reduce_sh[0];
    // Every warp must consume the shared maximum before thread 0 can reuse
    // reduce_sh[0] for its partial denominator below.
    __syncthreads();

    float local_denom = 0.0f;
    for (int split = tid; split < n_splits; split += blockDim.x) {
        const float2 meta = partial_meta[(((size_t) stream * n_q + q_index) * n_q_heads + q_head) * n_splits + split];
        float weight = 0.0f;
        if (meta.y > 0.0f) {
            weight = __expf(meta.x - m);
            local_denom += weight * meta.y;
        }
        split_weights[split] = weight;
    }
    reduce_sh[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_sh[tid] += reduce_sh[tid + stride];
        }
        __syncthreads();
    }
    const float denom = reduce_sh[0];

    // The FA output is declared ne = {D, q->ne[2], q->ne[1], q->ne[3]}, so its
    // ne[1] axis is the query head and ne[2] the query index. The (max, denom)
    // sink shares that layout, and the position-split LSE is declared from the
    // query layout as well. Both are indexed by strides so the kernel is
    // correct for a permuted (llama-graph) and a non-permuted query tensor.
    const size_t dst_off = (size_t) q_head * (size_t) nb11 + (size_t) q_index * (size_t) nb12 +
        (size_t) stream * (size_t) nb13;
    char * const dst_row_ptr = reinterpret_cast<char *>(dst) + dst_off;
    // src[8] keeps its historical meaning for this route: the packed
    // (query, head) (max, denom) sink. It is also the query-order input of the
    // tail contract, so its layout is not this kernel's business.
    const size_t meta_row = ((size_t) stream * n_q + q_index) * n_q_heads + q_head;
    if (tid == 0 && dst_meta != nullptr) {
        dst_meta[meta_row] = make_float2(m, denom);
    }
    if (tid == 0 && lse_out != nullptr) {
        // Position-split LSE: lse = m + log(denom); empty -> -inf. The tensor
        // is ne = {n_q_heads, n_q, n_stream}, so the head axis is contiguous.
        *reinterpret_cast<float *>(reinterpret_cast<char *>(lse_out) +
            (size_t) q_head * (size_t) lse_nb0 + (size_t) q_index * (size_t) lse_nb1 +
            (size_t) stream * (size_t) lse_nb2) =
                denom > 0.0f ? (m + logf(denom)) : -INFINITY;
    }

    if constexpr (D == 64) {
        // A one-thread-per-dimension reduction leaves three quarters of this
        // 256-thread block idle while reading the large split buffer. Give each
        // D64 dimension four contiguous split ranges, then reduce those four
        // numerators in shared memory. Each 64-thread group still issues
        // contiguous loads for a split, while all 256 threads remain useful.
        constexpr int DIM_GROUPS = GGML_CUDA_FATTN_KVARN_DECODE_COMBINE_THREADS / D;
        const int dim = tid % D;
        const int group = tid / D;
        const int splits_per_group = (n_splits + DIM_GROUPS - 1) / DIM_GROUPS;
        const int split_begin = group * splits_per_group;
        const int split_end = min(n_splits, split_begin + splits_per_group);
        float out = 0.0f;
        if (denom > 0.0f) {
            for (int split = split_begin; split < split_end; ++split) {
                const float weight = split_weights[split];
                if (weight == 0.0f) {
                    continue;
                }
                const size_t base = (((size_t) stream * n_q + q_index) * n_q_heads + q_head) *
                    n_splits + split;
                out += weight * partial[base * D + dim];
            }
        }
        reduce_sh[tid] = out;
        __syncthreads();
        if (group == 0) {
            for (int g = 1; g < DIM_GROUPS; ++g) {
                out += reduce_sh[g * D + dim];
            }
            reinterpret_cast<float *>(dst_row_ptr)[dim] = denom > 0.0f ? out / denom : 0.0f;
        }
    } else {
        for (int dim = tid; dim < D; dim += blockDim.x) {
            float out = 0.0f;
            if (denom > 0.0f) {
                for (int split = 0; split < n_splits; ++split) {
                    // Skipped splits did not write partials; their zero weight also
                    // avoids reading those unwritten values during the reduction.
                    const float weight = split_weights[split];
                    if (weight == 0.0f) {
                        continue;
                    }
                    const size_t base = (((size_t) stream * n_q + q_index) * n_q_heads + q_head) * n_splits + split;
                    out += weight * partial[base * D + dim];
                }
                out /= denom;
            }
            reinterpret_cast<float *>(dst_row_ptr)[dim] = out;
        }
    }
}

template<int D>
ggml_cuda_fattn_kvarn_decode_combine_kernel_t ggml_cuda_fattn_kvarn_decode_combine_get_kernel() {
    return ggml_cuda_fattn_kvarn_decode_combine_kernel<D>;
}
