// RKVA v2 Wave64 rowlane decode attention matrix for Xbox Series X (RDNA2).
// Implements dedicated Wave64 rowlane kernels across all requested formats:
// - F16
// - Q4_1
// - Q5_0 / KVarN5
// - Q5_1
// - Q6_0 / KVarN6
// - Q6_1
// - IQ4_NL
// - KVarN2 (Q2_0S)
// - KVarN3 (Q3_0)
// - KVarN8 (Q8_0)
//
// Each group owns (kv_head, query, split) and processes all RKVA_GQA query heads.
// In Wave64 decode, 64 lanes map to 64 quarter-units (covering 256 dims).
// QK Phase: Each lane evaluates the full 256-dim dot product for its row using WaveReadLaneAt.
// Online Softmax: WaveActiveMax & WaveActiveSum across the wave.
// PV Phase: Each lane accumulates its assigned 4 dimensions across valid rows of the tile.
// Wave states merge once via groupshared memory at group boundary.

#include "ggml_float.hlsli"
#include "rkva_bdr.hlsli"
#include "rkva_decode_quants_group_a.hlsli"
#include "rkva_decode_quants_group_b.hlsli"
#include "rkva_decode_quants_group_c.hlsli"
#include "rkva_decode_quants_saw_int4.hlsli"
#include "rkva_decode_quants_oscar_int2.hlsli"

#ifndef RKVA_GQA
#define RKVA_GQA 6
#endif

#ifndef RKVA_DIM
#define RKVA_DIM 256
#endif

#ifndef RKVA_THREADS
#define RKVA_THREADS 128
#endif

#ifndef RKVA_MAX_WAVES
#if RKVA_EXPERIMENT_ROWLANE_WAVE64_LDS
#define RKVA_MAX_WAVES (RKVA_THREADS / 64)
#else
#define RKVA_MAX_WAVES 4
#endif
#endif

#ifndef RKVA_QUARTERS
#define RKVA_QUARTERS 64
#endif

// Experimental source-only specialization. Host must retain fixed-wave64 gating.
// No default build target enables this flag.
#if RKVA_EXPERIMENT_ROWLANE_WAVE64_LDS
#if RKVA_THREADS != 128 || RKVA_DIM != 256 || RKVA_GQA != 6 || RKVA_MAX_WAVES != 2
#error Experimental LDS requires the audited 128-thread, D256, GQA6, two-wave shape.
#endif
#if (ROWLANE_F16 + ROWLANE_SAW_INT4) != 1 || ROWLANE_Q4_1 || ROWLANE_Q5_0 || ROWLANE_KVARN5 || ROWLANE_Q5_1 || ROWLANE_Q6_0 || ROWLANE_KVARN6 || ROWLANE_Q6_1 || ROWLANE_IQ4_NL || ROWLANE_KVARN2 || ROWLANE_KVARN3 || ROWLANE_KVARN8 || ROWLANE_OSCAR_INT2 || ROWLANE_Q2_0S || ROWLANE_Q3_0 || ROWLANE_Q8_0 || ROWLANE_BDR_INT4
#error Experimental LDS is restricted to one of F16 or SAW_INT4.
#endif
#endif

cbuffer Params : register(b0) {
    uint n_query;
    uint active_count;
    uint n_head;
    uint n_head_kv;
    uint head_dim;
    float kq_scale;
    uint origin;
    uint split_k;
    uint split_rows;
    uint direct_output;
};

ByteAddressBuffer queries : register(t0);
ByteAddressBuffer keys : register(t1);
ByteAddressBuffer values : register(t2);
ByteAddressBuffer positions : register(t3);
RWByteAddressBuffer partials : register(u0);
RWByteAddressBuffer output : register(u1);

groupshared float gs_max[RKVA_MAX_WAVES * RKVA_GQA];
groupshared float gs_den[RKVA_MAX_WAVES * RKVA_GQA];
groupshared float gs_acc[RKVA_MAX_WAVES * RKVA_GQA * RKVA_DIM];

#if ROWLANE_F16
#define ROWLANE_ENTRY CSRangeAttnF16DecodeRowlane
#define ROWLANE_ROW_HEAD 512
#define ROWLANE_LOAD load_f16
#define ROWLANE_QUARTER_DIM quarter_dim_f16
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_Q4_1
#define ROWLANE_ENTRY CSRangeAttnQ4_1DecodeRowlane
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_Q4_1
#define ROWLANE_LOAD load_q4_1
#define ROWLANE_QUARTER_DIM quarter_dim_q4_1
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_Q5_0 || ROWLANE_KVARN5
#define ROWLANE_ENTRY CSRangeAttnQ5_0DecodeRowlane
#define ROWLANE_ROW_HEAD 176
#define ROWLANE_LOAD load_q5_0
#define ROWLANE_QUARTER_DIM quarter_dim_q5_0
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_Q5_1
#define ROWLANE_ENTRY CSRangeAttnQ5_1DecodeRowlane
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_Q5_1
#define ROWLANE_LOAD load_q5_1
#define ROWLANE_QUARTER_DIM quarter_dim_q5_1
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_Q6_0 || ROWLANE_KVARN6
#define ROWLANE_ENTRY CSRangeAttnQ6_0DecodeRowlane
#define ROWLANE_ROW_HEAD 208
#define ROWLANE_LOAD load_q6_0
#define ROWLANE_QUARTER_DIM quarter_dim_q6_0
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_Q6_1
#define ROWLANE_ENTRY CSRangeAttnQ6_1DecodeRowlane
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_Q6_1
#define ROWLANE_LOAD load_q6_1
#define ROWLANE_QUARTER_DIM quarter_dim_q6_1
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_IQ4_NL
#define ROWLANE_ENTRY CSRangeAttnIQ4_NLDecodeRowlane
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_IQ4_NL
#define ROWLANE_LOAD load_iq4_nl
#define ROWLANE_QUARTER_DIM quarter_dim_iq4_nl
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_KVARN2 || ROWLANE_Q2_0S
#define ROWLANE_ENTRY CSRangeAttnKvarn2DecodeRowlane
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_KVARN2
#define ROWLANE_LOAD load_kvarn2
#define ROWLANE_QUARTER_DIM quarter_dim_kvarn2
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_KVARN3 || ROWLANE_Q3_0
#define ROWLANE_ENTRY CSRangeAttnKvarn3DecodeRowlane
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_KVARN3
#define ROWLANE_LOAD load_kvarn3
#define ROWLANE_QUARTER_DIM quarter_dim_kvarn3
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_KVARN8 || ROWLANE_Q8_0
#define ROWLANE_ENTRY CSRangeAttnKvarn8DecodeRowlane
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_KVARN8
#define ROWLANE_LOAD load_kvarn8
#define ROWLANE_QUARTER_DIM quarter_dim_kvarn8
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_SAW_INT4 || ROWLANE_BDR_INT4
[numthreads(RKVA_THREADS, 1, 1)]
void CSRangeAttnSawInt4DecodeRowlane(uint tid : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint kv_head = group.x, qi = group.y, split = group.z;
    uint lane = WaveGetLaneIndex();
    uint wave = tid >> 6;
    uint n_waves = RKVA_THREADS / 64;
    uint u = lane;
    uint d0 = quarter_dim_saw_int4(u);

    int query_pos = asint(positions.Load(qi * 4));
    uint causal_count = query_pos < int(origin) ? 0 : uint(query_pos) - origin + 1;
    uint limit = min(active_count, causal_count);
    uint begin = min(limit, split * split_rows);
    uint end = min(limit, (split + 1) * split_rows);

    float4 q0[RKVA_GQA];
    float  q_sum[RKVA_GQA];
    float4 acc0[RKVA_GQA];
    float maximum[RKVA_GQA], denom[RKVA_GQA];
    uint head0 = qi * n_head + kv_head * RKVA_GQA;
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        uint qbase = (head0 + g) * RKVA_DIM;
        float4 q_raw = asfloat(queries.Load4((qbase + d0) * 4));
        // Online BDR-16 rotation of Q once at shader entry
        q0[g] = bdr16_wave4(q_raw, u);
        q_sum[g] = WaveActiveSum(dot(q0[g], 1.0f));
        acc0[g] = float4(0, 0, 0, 0);
        maximum[g] = -3.402823466e38f;
        denom[g] = 0.0f;
    }

    uint tile_row0 = begin + wave * 64;
    [loop] for (; tile_row0 < end; tile_row0 += n_waves * 64) {
        uint my_row = tile_row0 + lane;
        bool row_valid = (my_row < end);
        uint my_ba = (my_row * n_head_kv + kv_head) * RKVA_ROW_HEAD_SAW_INT4;

        // QK Phase: 32 fused low/high nibble loads + algebraic factoring of scale/zero
        float score[RKVA_GQA];
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) score[g] = 0.0f;

        if (row_valid) {
            [loop] for (uint k = 0; k < 32u; ++k) {
                uint word = keys.Load(my_ba + k * 4u);
                float4 k_lo = float4(word & 0x0Fu, (word >> 8) & 0x0Fu, (word >> 16) & 0x0Fu, (word >> 24) & 0x0Fu);
                float4 k_hi = float4((word >> 4) & 0x0Fu, (word >> 12) & 0x0Fu, (word >> 20) & 0x0Fu, (word >> 28) & 0x0Fu);
                [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                    float4 q_lo = WaveReadLaneAt(q0[g], k);
                    float4 q_hi = WaveReadLaneAt(q0[g], k + 32u);
                    score[g] += dot(q_lo, k_lo) + dot(q_hi, k_hi);
                }
            }
            float2 k_sz = asfloat(keys.Load2(my_ba + 128u));
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                score[g] = (score[g] - k_sz.y * q_sum[g]) * k_sz.x;
            }
        } else {
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                score[g] = -3.402823466e38f;
            }
        }

        // Online Softmax across the 64 rows of this wave
        float w[RKVA_GQA];
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            score[g] *= kq_scale;
            float tile_max = WaveActiveMax(score[g]);
            float next_max = max(maximum[g], tile_max);
            float old_weight = (denom[g] == 0.0f) ? 0.0f : exp(maximum[g] - next_max);
            float weight = row_valid ? exp(score[g] - next_max) : 0.0f;
            float tile_sum = WaveActiveSum(weight);
            denom[g] = denom[g] * old_weight + tile_sum;
            maximum[g] = next_max;
            acc0[g] *= old_weight;
            w[g] = weight;
        }

        // PV Phase: Factor v_scale into row weights and v_zero into tile_v_bias once per 64-row tile
        float2 v_sz = row_valid ? asfloat(values.Load2(my_ba + 128u)) : float2(0.0f, 0.0f);
        float w_scaled[RKVA_GQA];
        float tile_v_bias[RKVA_GQA];
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            w_scaled[g] = w[g] * v_sz.x;
            tile_v_bias[g] = WaveActiveSum(w_scaled[g] * v_sz.y);
        }

        uint rows_in_tile = min(64u, (end > tile_row0) ? (end - tile_row0) : 0u);
        uint v_k_off = (u & 31u) * 4u;
        uint v_shift = ((u >> 5) & 1u) * 4u;
        [loop] for (uint r = 0; r < rows_in_tile; ++r) {
            uint v_ba = ((tile_row0 + r) * n_head_kv + kv_head) * RKVA_ROW_HEAD_SAW_INT4;
            uint v_word = values.Load(v_ba + v_k_off) >> v_shift;
            float4 v_nibble = float4(v_word & 0x0Fu, (v_word >> 8) & 0x0Fu, (v_word >> 16) & 0x0Fu, (v_word >> 24) & 0x0Fu);
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                float rw = WaveReadLaneAt(w_scaled[g], r);
                acc0[g] += rw * v_nibble;
            }
        }

        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            acc0[g] -= tile_v_bias[g];
        }
    }

    // Online BDR-16 des-rotation of accumulated output once at shader exit, then publish to LDS
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        acc0[g] = bdr16_wave4(acc0[g], u);
        uint slot = wave * RKVA_GQA + g;
        if (lane == 0) {
            gs_max[slot] = maximum[g];
            gs_den[slot] = denom[g];
        }
        uint acc_base = slot * RKVA_DIM;
        gs_acc[acc_base + d0 + 0] = acc0[g].x;
        gs_acc[acc_base + d0 + 1] = acc0[g].y;
        gs_acc[acc_base + d0 + 2] = acc0[g].z;
        gs_acc[acc_base + d0 + 3] = acc0[g].w;
    }
    GroupMemoryBarrierWithGroupSync();

    uint total = n_query * n_head;
    for (uint e = tid; e < RKVA_GQA * RKVA_DIM; e += RKVA_THREADS) {
        uint g = e / RKVA_DIM, d = e % RKVA_DIM;
        float merged_max = -3.402823466e38f;
        for (uint w_idx = 0; w_idx < n_waves; ++w_idx) {
            if (gs_den[w_idx * RKVA_GQA + g] > 0) merged_max = max(merged_max, gs_max[w_idx * RKVA_GQA + g]);
        }
        float merged_den = 0, merged_acc = 0;
        for (uint w_idx = 0; w_idx < n_waves; ++w_idx) {
            uint slot = w_idx * RKVA_GQA + g;
            if (gs_den[slot] > 0) {
                float weight = exp(gs_max[slot] - merged_max);
                merged_den += gs_den[slot] * weight;
                merged_acc += gs_acc[slot * RKVA_DIM + d] * weight;
            }
        }
        uint qh = head0 + g;
        if (direct_output != 0) {
            float inv = merged_den > 0 ? 1.0f / merged_den : 0.0f;
            output.Store((qh * RKVA_DIM + d) * 4, asuint(merged_acc * inv));
            if (d == 0) {
                float lse = merged_den > 0 ? merged_max + log(merged_den) : asfloat(0xff800000);
                output.Store((total * RKVA_DIM + qh) * 4, asuint(lse));
            }
        } else {
            uint dst = ((split * total + qh) * (RKVA_DIM + 2)) * 4;
            partials.Store(dst + d * 4, asuint(merged_acc));
            if (d == 0) {
                partials.Store(dst + RKVA_DIM * 4, asuint(merged_max));
                partials.Store(dst + (RKVA_DIM + 1) * 4, asuint(merged_den));
            }
        }
    }
}
#endif

#if ROWLANE_OSCAR_INT2
#define ROWLANE_ENTRY CSRangeAttnOscarInt2DecodeRowlane
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_OSCAR_INT2
#define ROWLANE_LOAD load_oscar_int2
#define ROWLANE_QUARTER_DIM quarter_dim_oscar_int2
#include "rkva_decode_rowlane_template.hlsli"
#endif


