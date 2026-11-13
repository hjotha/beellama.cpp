#include "gated_delta_net.cuh"
#include "../ggml-prefill-reference.h"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"

#include <atomic>
#include <cstdlib>
#include <limits>
#include <cstring>

static __global__ void gdn_precompute_exp(const float * g, float * g_exp, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += (int64_t) blockDim.x*gridDim.x) {
        g_exp[i] = expf(g[i]);
    }
}

// RAW: beta and g arrive pre-activation (ggml_gated_delta_net_set_raw_gates); the kernel applies
// sigmoid(beta) and raw_a[h] * softplus(g + raw_dt_bias[h]) with the unary kernels' formulas.
// G_PRECOMPUTED: g already holds exp(g) (GB10 long-prompt path); only used with RAW == false.
template <int S_v, bool KDA, bool keep_rs_t, bool RAW, bool G_PRECOMPUTED, bool SEGMENTED = false>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * raw_dt_bias,
                                     const float * raw_a,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int *   rows,
                                     int64_t       state_row_size) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // Segmented mode assigns one column to each quarter warp. The existing
    // DGX path instead uses the entire warp sequentially for four columns.
    const int lane = threadIdx.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    constexpr int cols_per_warp = SEGMENTED ? 4 : (S_v == 128 && !KDA ? 4 : 1);
#else
    constexpr int cols_per_warp = SEGMENTED ? 4 : 1;
#endif
    static_assert(!SEGMENTED || (S_v == 128 && !KDA), "segmented mode requires scalar gates and S_v=128");
    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int lanes_per_col = SEGMENTED ? warp_size / 4 : warp_size;
    constexpr int cols_per_lane = SEGMENTED ? 1 : cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    const int lane_in_col = lane % lanes_per_col;
    const int col_in_warp = SEGMENTED ? lane / lanes_per_col : 0;
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // rows mode (src[6] set): read each sequence's live state directly from cache
    // row rows[sequence] (stride state_row_size), avoiding the per-layer gather into
    // a dense scratch. gather mode: src[5] is a dense [S_v,S_v,H,n_seqs] scratch with
    // per-seq stride H*S_v*S_v. The output state always goes to the dst tail indexed
    // by `sequence` (unchanged), so only the input read offset differs.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = rows
        ? (int64_t) rows[sequence] * state_row_size + h_idx * S_v * S_v
        : sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[cols_per_lane][rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int c = 0; c < cols_per_lane; ++c) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            s_shard[c][r] = curr_state[(col + c) * S_v + i];
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        float beta_val = *beta_t;
        if constexpr (RAW) {
            beta_val = 1.0f / (1.0f + expf(-beta_val));
        }

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            static_assert(!(RAW && G_PRECOMPUTED), "exp(g) precompute is only defined for activated gates");
            float g0 = *g_t;
            if constexpr (RAW) {
                const float x = g0 + raw_dt_bias[h_idx];
                g0 = raw_a[h_idx] * ((x > 20.0f) ? x : logf(1.0f + expf(x)));
            }
            const float g_val = G_PRECOMPUTED ? g0 : expf(g0);

            // Each warp owns one or more columns and reuses the common q/k registers.
#pragma unroll
            for (int c = 0; c < cols_per_lane; ++c) {
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[c][r] * k_reg[r];
                }
                float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

                float delta_col = (v_t[col + c] - g_val * kv_col) * beta_val;

                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r]  = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                    attn_partial += s_shard[c][r] * q_reg[r];
                }

                float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

                if (lane_in_col == 0) {
                    attn_data[col + c] = attn_col * scale;
                }
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[0][r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[0][r]  = expf(g_t[i]) * s_shard[0][r] + k_reg[r] * delta_col;
                attn_partial += s_shard[0][r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < cols_per_lane; ++c) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * lanes_per_col + lane_in_col;
                        curr_state[(col + c) * S_v + i] = s_shard[c][r];
                    }
                }
            }
        }

    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < cols_per_lane; ++c) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                state[(col + c) * S_v + i] = s_shard[c][r];
            }
        }
    }
}

template <bool KDA, bool keep_rs_t, bool RAW, bool G_PRECOMPUTED>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * rb_d, const float * ra_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K,
        const int * rows, int64_t state_row_size, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // Opt-in only: leave all existing routes and default production behavior intact.
    static const int segmented = [] {
        const char * value = std::getenv("GGML_CUDA_GDN_SEGMENTED");
        if (value == nullptr) return 0;
        if (std::strcmp(value, "1") == 0) return 1;
        if (std::strcmp(value, "2") == 0) return 2;
        return 0;
    }();
    if constexpr (!KDA && !G_PRECOMPUTED) {
        if (segmented && (segmented == 1 || n_tokens >= 32) &&
                cc == GGML_CUDA_CC_ADA_LOVELACE && warp_size == 32 && S_v == 128) {
            int segmented_warps = 4;
            const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
            while (segmented_warps > 1 && H*n_seqs*(S_v / (4*segmented_warps)) < nsm) {
                segmented_warps /= 2;
            }
            const dim3 grid(H, n_seqs, (S_v + 4*segmented_warps - 1) / (4*segmented_warps));
            const dim3 block(warp_size, segmented_warps, 1);
            const ggml_cuda_kernel_launch_params params(grid, block, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, false, keep_rs_t, RAW, false, true>, params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, init_fastdiv_values(neqk1), init_fastdiv_values(rq3),
                scale, state_slot_stride, K, rows, state_row_size);
            return;
        }
    }
#endif
    const int num_warps = 4;
    const int cols_per_warp = cc == GGML_CUDA_CC_DGX_SPARK && S_v == 128 && !KDA ? 4 : 1;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps * cols_per_warp - 1) / (num_warps * cols_per_warp));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rows, state_row_size);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rows, state_row_size);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rows, state_row_size);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rows, state_row_size);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
        0, 0,
    };
}

// BEGIN ggml_cuda_gdn_prefix_slot_guard
// Address-only guard; false means use the unchanged whole-graph scalar path.
struct ggml_cuda_gdn_byte_span {
    uintptr_t begin;
    size_t size;
};

static bool ggml_cuda_gdn_prefix_slot_guard(
        uintptr_t base, size_t stride, size_t state_bytes, size_t slots,
        const ggml_cuda_gdn_byte_span * forbidden, size_t n_forbidden, uintptr_t & out) {
    const uintptr_t max = std::numeric_limits<uintptr_t>::max();
    if (base == 0 || slots < 2 || state_bytes == 0 || stride < state_bytes ||
        slots - 1 > max / stride) return false;
    const uintptr_t offset = (slots - 1) * stride;
    if (offset > max - base) return false;
    const uintptr_t slot = base + offset;
    if (state_bytes > max - slot) return false;
    const uintptr_t end = slot + state_bytes;
    for (size_t i = 0; i < n_forbidden; ++i) {
        const auto & range = forbidden[i];
        if (range.begin == 0 || range.size > max - range.begin) return false;
        const uintptr_t range_end = range.begin + range.size;
        if (range.size != 0 && slot < range_end && range.begin < end) return false;
    }
    out = slot;
    return true;
}
// END ggml_cuda_gdn_prefix_slot_guard

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // rows-indexed state read (src[6]): when set, src[5] is a 2D cache view and each
    // sequence reads its live state at row rows[seq] (stride = src_state->nb[1]).
    const ggml_tensor * src_rows = dst->src[6];
    const int *   rows_d         = src_rows ? (const int *) src_rows->data : nullptr;
    const int64_t state_row_size = src_rows ? (int64_t) (src_state->nb[1] / sizeof(float)) : 0;
    GGML_ASSERT(src_rows == nullptr || src_rows->type == GGML_TYPE_I32);

    const bool    raw  = ggml_get_op_params_i32(dst, 1) != 0;
    const float * rb_d = raw ? (const float *) dst->src[7]->data : nullptr;
    const float * ra_d = raw ? (const float *) dst->src[8]->data : nullptr;
    GGML_ASSERT(!(raw && kda)); // raw gates are defined for the scalar gate only

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // Experimental transaction: MMA writes the boundary snapshot (slot K-1),
    // then the existing scalar kernel writes the suffix and remaining slots.
    static const int prefix_mma_mode = [] {
        const char * value = std::getenv("GGML_CUDA_GDN_PREFIX_MMA");
        if (value == nullptr) return 0;
        if (std::strcmp(value, "1") == 0) return 1;
        if (std::strcmp(value, "2") == 0) return 2;
        return 0;
    }();
    const int64_t max_prefix = n_tokens - (K - 1);
    const int64_t prefix = prefix_mma_mode == 2 ? (max_prefix / 16) * 16 : max_prefix;
    if (prefix_mma_mode && !raw && !kda && src_rows == nullptr && n_seqs == 1 &&
        S_v == 128 && H > 0 && K > 1 && n_tokens >= K && prefix >= 64 &&
        ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_ADA_LOVELACE &&
        state_slot_stride > 0) {
        const size_t max = std::numeric_limits<size_t>::max();
        const size_t head_size = size_t(128)*128*sizeof(float);
        const bool dimensions_ok = size_t(H) <= max / head_size &&
            size_t(n_tokens) <= max / (size_t(128)*sizeof(float)) / size_t(H) &&
            uint64_t(state_slot_stride) <= max / sizeof(float);
        if (dimensions_ok) {
            const size_t state_bytes = size_t(H) * head_size;
            const size_t scores_bytes = size_t(n_tokens) * size_t(H) * 128 * sizeof(float);
            const size_t stride_bytes = size_t(state_slot_stride) * sizeof(float);
            const ggml_cuda_gdn_byte_span forbidden[] = {
                { uintptr_t(q_d), ggml_nbytes(src_q) },
                { uintptr_t(k_d), ggml_nbytes(src_k) },
                { uintptr_t(v_d), ggml_nbytes(src_v) },
                { uintptr_t(g_d), ggml_nbytes(src_g) },
                { uintptr_t(b_d), ggml_nbytes(src_beta) },
                { uintptr_t(s_d), ggml_nbytes(src_state) },
                { uintptr_t(dst_d), scores_bytes },
            };
            uintptr_t boundary = 0;
            if (ggml_cuda_gdn_prefix_slot_guard(uintptr_t(state_d), stride_bytes, state_bytes,
                    size_t(K), forbidden, sizeof(forbidden)/sizeof(forbidden[0]), boundary)) {
                // Keep the same scalar/MMA boundaries as the two aligned reference blocks.
                if (ggml_prefill_reference_get(dst) == GGML_PREFILL_REFERENCE_SHIFTED &&
                        n_tokens == 256 && K <= 16) {
                    const int64_t first_prefix = ((128 - (K - 1))/16)*16;
                    bool slots_safe = first_prefix >= 64;
                    for (int j = 1; j < K; ++j) {
                        uintptr_t checked = 0;
                        slots_safe = slots_safe && ggml_cuda_gdn_prefix_slot_guard(
                                uintptr_t(state_d), stride_bytes, state_bytes, size_t(j + 1),
                                forbidden, sizeof(forbidden)/sizeof(forbidden[0]), checked);
                    }
                    if (slots_safe) {
                        ggml_cuda_pool_alloc<float> reference_slots(ctx.pool(), size_t(K)*state_bytes/sizeof(float));
                        const int64_t reference_stride = int64_t(state_bytes/sizeof(float));
                        ggml_cuda_gdn_mma_args reference = ggml_cuda_gdn_mma_args_from_tensor(dst);
                        reference.n_tokens = first_prefix;
                        reference.state_out = reference_slots.get() + int64_t(K - 1)*reference_stride;
                        if (ggml_cuda_gdn_mma_launch(ctx.device, reference, stream)) {
                            const int64_t first_suffix = 128 - first_prefix;
                            launch_gated_delta_net<false, true, false, false>(
                                q_d + size_t(first_prefix)*size_t(sq2), k_d + size_t(first_prefix)*size_t(sq2),
                                v_d + size_t(first_prefix)*size_t(sv2), g_d + size_t(first_prefix)*size_t(sb2),
                                b_d + size_t(first_prefix)*size_t(sb2), nullptr, nullptr,
                                reference_slots.get() + int64_t(K - 1)*reference_stride,
                                dst_d + size_t(first_prefix)*size_t(H)*128, reference_slots.get(),
                                S_v, H, first_suffix, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                                sb1, sb2, sb3, neqk1, rq3, scale, reference_stride, K, nullptr, 0, stream);
                            reference.q = q_d + size_t(128)*size_t(sq2);
                            reference.k = k_d + size_t(128)*size_t(sq2);
                            reference.v = v_d + size_t(128)*size_t(sv2);
                            reference.g = g_d + size_t(128)*size_t(sb2);
                            reference.beta = b_d + size_t(128)*size_t(sb2);
                            reference.state = reference_slots.get();
                            reference.n_tokens = 128;
                            reference.dst = dst_d + size_t(128)*size_t(H)*128;
                            reference.state_out = state_d;
                            reference.reference_snapshots = K;
                            reference.reference_slot_stride = state_slot_stride;
                            if (ggml_cuda_gdn_mma_launch(ctx.device, reference, stream)) {
                                return;
                            }
                        }
                    }
                }
                // Reuse the last MMA tile to write the retained intermediate states.
                if (ggml_prefill_reference_get(dst) == GGML_PREFILL_REFERENCE_FIRST && n_tokens == 128 && K > 1 && K <= 16) {
                    bool slots_safe = true;
                    for (int j = 1; j < K; ++j) {
                        uintptr_t checked = 0;
                        slots_safe = slots_safe && ggml_cuda_gdn_prefix_slot_guard(
                                uintptr_t(state_d), stride_bytes, state_bytes, size_t(j + 1),
                                forbidden, sizeof(forbidden)/sizeof(forbidden[0]), checked);
                    }
                    if (slots_safe) {
                        ggml_cuda_gdn_mma_args reference = ggml_cuda_gdn_mma_args_from_tensor(dst);
                        reference.state_out = state_d;
                        reference.reference_snapshots = K;
                        reference.reference_slot_stride = state_slot_stride;
                        if (ggml_cuda_gdn_mma_launch(ctx.device, reference, stream)) return;
                    }
                }
                const int64_t suffix = n_tokens - prefix;
                ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
                args.n_tokens = prefix;
                args.state_out = reinterpret_cast<float *>(boundary);
                if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
                    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
                    if (!logged.test_and_set(std::memory_order_relaxed)) {
                        GGML_LOG_INFO("%s: opt-in GDN prefix MMA selected (prefix=%lld suffix=%lld K=%d); "
                                      "capture may record this launch for replay\n",
                                      __func__, (long long) prefix, (long long) suffix, K);
                    }
                    launch_gated_delta_net<false, true, false, false>(
                        q_d + size_t(prefix)*size_t(sq2), k_d + size_t(prefix)*size_t(sq2),
                        v_d + size_t(prefix)*size_t(sv2), g_d + size_t(prefix)*size_t(sb2),
                        b_d + size_t(prefix)*size_t(sb2), nullptr, nullptr,
                        reinterpret_cast<const float *>(boundary),
                        dst_d + size_t(prefix)*size_t(H)*128, state_d,
                        S_v, H, suffix, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                        nullptr, 0, stream);
                    return;
                }
                static std::atomic_flag declined_logged = ATOMIC_FLAG_INIT;
                if (!declined_logged.test_and_set(std::memory_order_relaxed)) {
                    GGML_LOG_WARN("%s: opt-in GDN prefix MMA declined by launcher; using scalar fallback\n", __func__);
                }
            } else {
                static std::atomic_flag logged = ATOMIC_FLAG_INIT;
                if (!logged.test_and_set(std::memory_order_relaxed)) {
                    GGML_LOG_WARN("%s: opt-in GDN prefix MMA overlap guard rejected a graph; "
                                  "using scalar fallback\n", __func__);
                }
            }
        } else {
            static std::atomic_flag dimensions_logged = ATOMIC_FLAG_INIT;
            if (!dimensions_logged.test_and_set(std::memory_order_relaxed)) {
                GGML_LOG_WARN("%s: opt-in GDN prefix MMA dimensions overflow; using scalar fallback\n", __func__);
            }
        }
    } else if (prefix_mma_mode) {
        static std::atomic_flag ineligible_logged = ATOMIC_FLAG_INIT;
        if (!ineligible_logged.test_and_set(std::memory_order_relaxed)) {
            GGML_LOG_INFO("%s: opt-in GDN prefix MMA shape ineligible (T=%lld K=%d raw=%d rows=%d); "
                          "using existing path\n", __func__, (long long) n_tokens, K, int(raw), int(src_rows != nullptr));
        }
    }
#endif

    // MMA publishes only the final state; retain scalar snapshots for rollback.
    if (!raw && src_rows == nullptr && !keep_rs) {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
    }

    // GB10 long-prompt path: exp(g) once per (token, head) instead of once per column-warp.
    // Only for activated gates; with raw gates (#165) the activation happens inside the kernel.
    ggml_cuda_pool_alloc<float> g_exp_alloc(ctx.pool());
    bool g_precomputed = false;
    if (!kda && !raw && S_v == 128 && n_tokens >= 32 &&
            ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_DGX_SPARK) {
        const int64_t n_g = ggml_nelements(src_g);
        g_exp_alloc.alloc(n_g);
        const int block = 256;
        const int grid = std::min<int64_t>((n_g + block - 1)/block, 4096);
        gdn_precompute_exp<<<grid, block, 0, stream>>>(g_d, g_exp_alloc.ptr, n_g);
        g_d = g_exp_alloc.ptr;
        g_precomputed = true;
    }

#define GDN_LAUNCH(KDA_, KEEP_, RAW_, PRE_)                                                       \
    launch_gated_delta_net<KDA_, KEEP_, RAW_, PRE_>(q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, \
        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                                    \
        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rows_d, state_row_size, stream)

    if (kda) {
        if (keep_rs) { GDN_LAUNCH(true,  true,  false, false); } else { GDN_LAUNCH(true,  false, false, false); }
    } else if (raw) {
        if (keep_rs) { GDN_LAUNCH(false, true,  true,  false); } else { GDN_LAUNCH(false, false, true,  false); }
    } else {
        if (g_precomputed) {
            if (keep_rs) { GDN_LAUNCH(false, true,  false, true);  } else { GDN_LAUNCH(false, false, false, true);  }
        } else {
            if (keep_rs) { GDN_LAUNCH(false, true,  false, false); } else { GDN_LAUNCH(false, false, false, false); }
        }
    }
#undef GDN_LAUNCH
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
