#pragma once

#include "common.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"
#include "ptq1-ada-q8.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

template <int ncols_dst>
static __device__ __forceinline__ void vec_dot_ptq1_0_q8_1_ada(const void * __restrict__ vbq,
                                                                 const block_q8_1 * __restrict__ y_col0,
                                                                 const int &    kbx_x,
                                                                 const int &    kb,
                                                                 const uint32_t stride_col_y,
                                                                 float *        result) {
    const block_ptq1_0 * bq                 = (const block_ptq1_0 *) vbq + kbx_x;
    int                  sumi[ncols_dst][4] = {};

    // Lane-coalesced activation words: word ww of this K-block is at yw[ww*32]; 32 lanes of a
    // warp own 32 consecutive K-blocks, so every load below is 32 consecutive words.
    const int * __restrict__ yw = (const int *) y_col0 + (kb >> 5) * GGML_CUDA_PTQ1_Q8_GROUP_WORDS + (kb & 31);
    const uint32_t           sy = stride_col_y * (sizeof(block_q8_1) / 4);
#    define PTQ1_U(j, sub, m) yw[(j) * sy + ((sub) * 8 + (m)) * GGML_CUDA_PTQ1_Q8_GROUP_KB]
#    define PTQ1_DS(j, sub)   yw[(j) * sy + (32 + (sub)) * GGML_CUDA_PTQ1_Q8_GROUP_KB]

    // Widen four bytes to 16-bit lanes so multiply-by-three cannot carry between bytes.
#    pragma unroll
    for (int g = 0; g < 4; ++g) {
        const uint32_t packed = get_int_b4(bq->qs, g);
        uint32_t       v_lo   = __byte_perm(packed, 0, 0x4140);
        uint32_t       v_hi   = __byte_perm(packed, 0, 0x4342);

#    pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t w_lo = v_lo * 3;
            const uint32_t w_hi = v_hi * 3;
            v_lo                = w_lo & 0x00FF00FF;
            v_hi                = w_hi & 0x00FF00FF;

            // Raw digits {0,1,2}; the -1 bias is folded into the per-32-block exact q8 sum below.
            const int q = __byte_perm(w_lo, w_hi, 0x7531);
            const int e = t * 16 + 4 * g;
#    pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const int u     = PTQ1_U(j, e >> 5, (e & 31) >> 2);
                sumi[j][e >> 5] = ggml_cuda_dp4a(q, u, sumi[j][e >> 5]);
            }
        }
    }

#    pragma unroll
    for (int g = 0; g < 2; ++g) {
        const uint32_t packed = get_int_b4(bq->qs + 16, g);
        uint32_t       v_lo   = __byte_perm(packed, 0, 0x4140);
        uint32_t       v_hi   = __byte_perm(packed, 0, 0x4342);

#    pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t w_lo = v_lo * 3;
            const uint32_t w_hi = v_hi * 3;
            v_lo                = w_lo & 0x00FF00FF;
            v_hi                = w_hi & 0x00FF00FF;

            const int q = __byte_perm(w_lo, w_hi, 0x7531);
            const int e = 80 + t * 8 + 4 * g;
#    pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const int u     = PTQ1_U(j, e >> 5, (e & 31) >> 2);
                sumi[j][e >> 5] = ggml_cuda_dp4a(q, u, sumi[j][e >> 5]);
            }
        }
    }

    uint32_t v = (uint32_t) bq->qh[0] | ((uint32_t) bq->qh[1] << 16);
#    pragma unroll
    for (int t = 0; t < 4; t += 2) {
        const uint32_t w0 = v * 3;
        v                 = w0 & 0x00FF00FF;
        const uint32_t w1 = v * 3;
        v                 = w1 & 0x00FF00FF;

        const int q = __byte_perm(w0, w1, 0x7531);
#    pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const int u = PTQ1_U(j, 3, 6 + t / 2);
            sumi[j][3]  = ggml_cuda_dp4a(q, u, sumi[j][3]);
        }
    }

    const float d = (float) bq->d;
#    pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        float acc = 0.0f;
#    pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int   ds_bits = PTQ1_DS(j, k);
            const half2 ds      = *reinterpret_cast<const half2 *>(&ds_bits);
            // ds.y carries the exact int16 sum of the 32 q8 values (see quantize_ptq1_0_ada).
            const int   isum    = (int) __half_as_short(__high2half(ds));
            acc += __low2float(ds) * (float) (sumi[j][k] - isum);
        }
        result[j] = d * acc;
    }
#    undef PTQ1_U
#    undef PTQ1_DS
}

static __global__ void quantize_ptq1_0_ada(const float * x, int32_t * y, int ncols, int padded_cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= padded_cols) {
        return;
    }
    const float value = col < ncols ? x[col] : 0.0f;
    ggml_cuda_store_ptq1_ada_q8(value, y, col);
}

template <bool warp_per_row, bool has_gate>
__launch_bounds__(4 * WARP_SIZE, 1)
static __global__ void mul_mat_vec_ptq1_0_ada(
        const block_ptq1_0 * weights, const block_q8_1 * activations, ggml_cuda_mm_fusion_args_device fusion,
        float * dst, int blocks_per_row, int stride_row, int rows) {
    constexpr int warps = 4;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row = warp_per_row ? blockIdx.x * warps + warp : blockIdx.x;
    float up = 0.0f;
    float gate = 0.0f;

    if (row < rows) {
        const int first = warp_per_row ? lane : warp * WARP_SIZE + lane;
        constexpr int step = warp_per_row ? WARP_SIZE : warps * WARP_SIZE;
        for (int k = first; k < blocks_per_row; k += step) {
            float dot;
            // Match PrismML's look-ahead on the large-K split reduction.
#if defined(__CUDA_ARCH__)
            if constexpr (!warp_per_row) {
                if (k + step < blocks_per_row) {
                    const auto * next = weights + row * stride_row + k + step;
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(__cvta_generic_to_global(next)));
                }
            }
#endif
            vec_dot_ptq1_0_q8_1_ada<1>(weights, activations, row * stride_row + k, k, 0, &dot);
            up += dot;
            if constexpr (has_gate) {
                vec_dot_ptq1_0_q8_1_ada<1>(fusion.gate, activations, row * stride_row + k, k, 0, &dot);
                gate += dot;
            }
        }
    }
    up = warp_reduce_sum<WARP_SIZE>(up);
    if constexpr (has_gate) {
        gate = warp_reduce_sum<WARP_SIZE>(gate);
    }
    if constexpr (!warp_per_row) {
        __shared__ float partial_up[warps];
        __shared__ float partial_gate[has_gate ? warps : 1];
        if (lane == 0) {
            partial_up[warp] = up;
            if constexpr (has_gate) {
                partial_gate[warp] = gate;
            }
        }
        __syncthreads();
        if (warp != 0 || lane != 0) {
            return;
        }
#pragma unroll
        for (int i = 1; i < warps; ++i) {
            up += partial_up[i];
            if constexpr (has_gate) {
                gate += partial_gate[i];
            }
        }
    }
    if (lane == 0 && row < rows) {
        if (fusion.x_bias) {
            up += static_cast<const float *>(fusion.x_bias)[row];
        }
        if constexpr (has_gate) {
            if (fusion.gate_bias) {
                gate += static_cast<const float *>(fusion.gate_bias)[row];
            }
            up = ggml_cuda_op_glu_single(fusion.glu_op, gate, up, fusion.glu_limit);
        }
        dst[row] = up;
    }
}

template <bool warp_per_row>
static void launch_ptq1_0_ada(
        const block_ptq1_0 * weights, const block_q8_1 * activations, ggml_cuda_mm_fusion_args_device fusion,
        float * dst, int blocks_per_row, int stride_row, int rows, cudaStream_t stream) {
    const dim3 blocks(warp_per_row ? (rows + 3) / 4 : rows);
    const dim3 threads(WARP_SIZE, 4);
    if (fusion.gate) {
        mul_mat_vec_ptq1_0_ada<warp_per_row, true><<<blocks, threads, 0, stream>>>(
            weights, activations, fusion, dst, blocks_per_row, stride_row, rows);
    } else {
        mul_mat_vec_ptq1_0_ada<warp_per_row, false><<<blocks, threads, 0, stream>>>(
            weights, activations, fusion, dst, blocks_per_row, stride_row, rows);
    }
}
#endif
