// RKVA v2 optimized Q4_0 positional overflow attention (decode, GQA-shared K/V).
// One group owns (KV head, query, split) and evaluates all RKVA_GQA Q heads that
// read that KV head, so every K/V row is loaded and dequantized once instead of
// once per Q head. Each wave owns whole rows (no group barrier in the row loop);
// waves merge their online-softmax states once at the end.
//
// Semantics match rkva_range_attn.hlsl with RKVA_RANGE_STORAGE=1: identical
// Q4_0 dequantized values (float(q - 8) * f16 scale), FP32 dot/softmax/V
// accumulation, the same causal limit (origin, positions, active_count) and the
// same partial layout [split][query*n_head + head][head_dim + 2] consumed by the
// split-K reducers. Only the floating-point summation order differs. When
// direct_output != 0 (split_k == 1) the group writes the final normalized O and
// LSE in the reducer's output layout and the reducer pass is skipped.
//
// Supported shape (the host gates every call): head_dim 256, n_head ==
// RKVA_GQA * n_head_kv, wave size 32..128, threadgroups of 128 threads.
#define RKVA_GQA 6
#define RKVA_DIM 256
#define RKVA_THREADS 128
#if RKVA_EXPERIMENT_ROWLANE_WAVE64_LDS
#define RKVA_MAX_WAVES (RKVA_THREADS / 64)
#else
#define RKVA_MAX_WAVES 4    // RKVA_THREADS / minimum supported wave size (32)
#endif
#define RKVA_ROW_HEAD 144   // 8 Q4_0 blocks x 18 bytes per (row, KV head)
#define RKVA_QUARTERS 64    // 4-element units per 256-wide row

// Experimental source-only specialization. Host must retain fixed-wave64 gating.
// No default build target enables this flag.
#if RKVA_EXPERIMENT_ROWLANE_WAVE64_LDS
#if RKVA_THREADS != 128 || RKVA_DIM != 256 || RKVA_GQA != 6 || RKVA_MAX_WAVES != 2
#error Experimental LDS requires the audited 128-thread, D256, GQA6, two-wave shape.
#endif
#if RKVA_DECODE_ROWLANE != 1
#error Experimental LDS requires a rowlane-only compilation.
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

groupshared uint gs_raw_k[RKVA_MAX_WAVES][2][36];
groupshared uint gs_raw_v[RKVA_MAX_WAVES][2][36];

// Quarter unit u covers block b = u >> 3, quant bytes 4c..4c+3 (c = (u >> 1) & 3)
// and nibble half h = u & 1: elements b*32 + h*16 + 4c + {0,1,2,3}.
uint quarter_dim(uint u) {
    return (u >> 3) * 32 + (u & 1) * 16 + ((u >> 1) & 3) * 4;
}

// Coalesced 144-byte wave load and dequantization from LDS:
// Lanes 0..35 store the 36 raw uints (144 bytes) of the row directly into LDS.
// All 64 lanes then dequantize their quarter units in parallel via fast LDS reads.
float4 unpack_q4_from_k(uint wave, uint row_slot, uint u) {
    uint pair_idx = u >> 4;
    uint is_odd = (u >> 3) & 1;
    uint c = (u >> 1) & 3;
    uint pair_base = pair_idx * 9;

    uint scale_word = is_odd ? gs_raw_k[wave][row_slot][pair_base + 4] : gs_raw_k[wave][row_slot][pair_base + 0];
    float scale = f16tof32((scale_word >> (is_odd * 16)) & 65535u);

    uint word;
    if (is_odd) {
        word = gs_raw_k[wave][row_slot][pair_base + 5 + c];
    } else {
        uint p0 = gs_raw_k[wave][row_slot][pair_base + c];
        uint p1 = gs_raw_k[wave][row_slot][pair_base + c + 1];
        word = (p0 >> 16) | (p1 << 16);
    }
    word >>= (u & 1u) * 4;
    float4 nibble = float4(word & 15u, (word >> 8) & 15u, (word >> 16) & 15u, (word >> 24) & 15u);
    return (nibble - 8.0f) * scale;
}

float4 unpack_q4_from_v(uint wave, uint row_slot, uint u) {
    uint pair_idx = u >> 4;
    uint is_odd = (u >> 3) & 1;
    uint c = (u >> 1) & 3;
    uint pair_base = pair_idx * 9;

    uint scale_word = is_odd ? gs_raw_v[wave][row_slot][pair_base + 4] : gs_raw_v[wave][row_slot][pair_base + 0];
    float scale = f16tof32((scale_word >> (is_odd * 16)) & 65535u);

    uint word;
    if (is_odd) {
        word = gs_raw_v[wave][row_slot][pair_base + 5 + c];
    } else {
        uint p0 = gs_raw_v[wave][row_slot][pair_base + c];
        uint p1 = gs_raw_v[wave][row_slot][pair_base + c + 1];
        word = (p0 >> 16) | (p1 << 16);
    }
    word >>= (u & 1u) * 4;
    float4 nibble = float4(word & 15u, (word >> 8) & 15u, (word >> 16) & 15u, (word >> 24) & 15u);
    return (nibble - 8.0f) * scale;
}

// Branchless aligned Q4_0 dequantization: 36-byte (2 blocks = 9 uints) alignment.
// Eliminates redundant loads: uses Load2 for unaligned chunk and Load for aligned chunk.
float4 load_q4(ByteAddressBuffer buffer, uint base, uint u) {
    uint pair_idx = u >> 4;
    uint addr_pair = base + pair_idx * 36;
    uint is_odd = (u >> 3) & 1;
    uint c = (u >> 1) & 3;
    uint scale_word = buffer.Load(addr_pair + is_odd * 16);
    float scale = f16tof32((scale_word >> (is_odd * 16)) & 65535u);
    uint word;
    if (is_odd) {
        word = buffer.Load(addr_pair + 20 + c * 4);
    } else {
        uint2 p = buffer.Load2(addr_pair + c * 4);
        word = (p.x >> 16) | (p.y << 16);
    }
    word >>= (u & 1) * 4;
    float4 nibble = float4(word & 15u, (word >> 8) & 15u, (word >> 16) & 15u, (word >> 24) & 15u);
    return (nibble - 8.0f) * scale;
}

#define RKVA_WAVE64 1

#if !RKVA_EXPERIMENT_ROWLANE_WAVE64_LDS
[numthreads(RKVA_THREADS, 1, 1)]
void CSRangeAttnQ4Opt(uint tid : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint kv_head = group.x, qi = group.y, split = group.z;
    uint lane = WaveGetLaneIndex();
    uint wave = tid >> 6;
    uint n_waves = RKVA_THREADS / 64;
    uint u0 = lane;
    uint d0 = quarter_dim(u0);

    int query_pos = asint(positions.Load(qi * 4));
    uint causal_count = query_pos < int(origin) ? 0 : uint(query_pos) - origin + 1;
    uint limit = min(active_count, causal_count);
    uint begin = min(limit, split * split_rows);
    uint end = min(limit, (split + 1) * split_rows);

    float4 q0[RKVA_GQA], acc0[RKVA_GQA];
    float maximum[RKVA_GQA], denom[RKVA_GQA];
    uint head0 = qi * n_head + kv_head * RKVA_GQA;
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        uint qbase = (head0 + g) * RKVA_DIM;
        q0[g] = asfloat(queries.Load4((qbase + d0) * 4));
        acc0[g] = float4(0, 0, 0, 0);
        maximum[g] = -3.402823466e38f;
        denom[g] = 0;
    }

    uint row_stride = n_head_kv * RKVA_ROW_HEAD;
    uint total_rows = end - begin;
    uint full_iters = total_rows / (2 * n_waves);
    [loop] for (uint iter = 0; iter < full_iters; ++iter) {
        uint row = begin + iter * (2 * n_waves) + wave;
        uint ba = (row * n_head_kv + kv_head) * RKVA_ROW_HEAD;
        uint bb = ba + n_waves * row_stride;

        if (lane < 36) {
            gs_raw_k[wave][0][lane] = keys.Load(ba + lane * 4);
            gs_raw_k[wave][1][lane] = keys.Load(bb + lane * 4);
            gs_raw_v[wave][0][lane] = values.Load(ba + lane * 4);
            gs_raw_v[wave][1][lane] = values.Load(bb + lane * 4);
        }
        GroupMemoryBarrierWithGroupSync();

        float4 ka0 = unpack_q4_from_k(wave, 0, u0);
        float4 kb0 = unpack_q4_from_k(wave, 1, u0);
        float4 va0 = unpack_q4_from_v(wave, 0, u0);
        float4 vb0 = unpack_q4_from_v(wave, 1, u0);

        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            float sa = WaveActiveSum(dot(q0[g], ka0)) * kq_scale;
            float sb = WaveActiveSum(dot(q0[g], kb0)) * kq_scale;
            float next_max = max(maximum[g], max(sa, sb));
            float old_weight = denom[g] == 0 ? 0 : exp(maximum[g] - next_max);
            float wa = exp(sa - next_max), wb = exp(sb - next_max);
            acc0[g] = acc0[g] * old_weight + wa * va0 + wb * vb0;
            denom[g] = denom[g] * old_weight + wa + wb;
            maximum[g] = next_max;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    uint rem_base = begin + full_iters * (2 * n_waves);
    uint rem_row = rem_base + wave;
    if (rem_row < end) {
        uint ba = (rem_row * n_head_kv + kv_head) * RKVA_ROW_HEAD;
        float4 ka0 = load_q4(keys, ba, u0);
        float4 va0 = load_q4(values, ba, u0);
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            float sa = WaveActiveSum(dot(q0[g], ka0)) * kq_scale;
            float next_max = max(maximum[g], sa);
            float old_weight = denom[g] == 0 ? 0 : exp(maximum[g] - next_max);
            float wa = exp(sa - next_max);
            acc0[g] = acc0[g] * old_weight + wa * va0;
            denom[g] = denom[g] * old_weight + wa;
            maximum[g] = next_max;
        }
    }
    rem_row += n_waves;
    if (rem_row < end) {
        uint ba = (rem_row * n_head_kv + kv_head) * RKVA_ROW_HEAD;
        float4 ka0 = load_q4(keys, ba, u0);
        float4 va0 = load_q4(values, ba, u0);
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            float sa = WaveActiveSum(dot(q0[g], ka0)) * kq_scale;
            float next_max = max(maximum[g], sa);
            float old_weight = denom[g] == 0 ? 0 : exp(maximum[g] - next_max);
            float wa = exp(sa - next_max);
            acc0[g] = acc0[g] * old_weight + wa * va0;
            denom[g] = denom[g] * old_weight + wa;
            maximum[g] = next_max;
        }
    }

    // Publish each wave's state, then merge across waves once.
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
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
        for (uint w = 0; w < n_waves; ++w) {
            if (gs_den[w * RKVA_GQA + g] > 0) merged_max = max(merged_max, gs_max[w * RKVA_GQA + g]);
        }
        float merged_den = 0, merged_acc = 0;
        for (uint w = 0; w < n_waves; ++w) {
            uint slot = w * RKVA_GQA + g;
            if (gs_den[slot] > 0) {
                float weight = exp(gs_max[slot] - merged_max);
                merged_den += gs_den[slot] * weight;
                merged_acc += gs_acc[slot * RKVA_DIM + d] * weight;
            }
        }
        uint qh = head0 + g;
        if (direct_output != 0) {
            float inv = merged_den > 0 ? 1 / merged_den : 0;
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

#endif // generic entry retained only in baseline compilation

#if RKVA_DECODE_ROWLANE
[numthreads(RKVA_THREADS, 1, 1)]
void CSRangeAttnQ4DecodeRowlane(uint tid : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint kv_head = group.x, qi = group.y, split = group.z;
    uint lane = WaveGetLaneIndex();
    uint wave = tid >> 6; // tid / 64: 2 waves of 64 lanes
    uint n_waves = RKVA_THREADS / 64; // 2
    uint u = lane; // On Wave64: 64 lanes cover all 64 quarter-units (256 dims)
    uint d0 = quarter_dim(u);

    int query_pos = asint(positions.Load(qi * 4));
    uint causal_count = query_pos < int(origin) ? 0 : uint(query_pos) - origin + 1;
    uint limit = min(active_count, causal_count);
    uint begin = min(limit, split * split_rows);
    uint end = min(limit, (split + 1) * split_rows);

    float4 q0[RKVA_GQA];
    float4 acc0[RKVA_GQA];
    float maximum[RKVA_GQA], denom[RKVA_GQA];
    uint head0 = qi * n_head + kv_head * RKVA_GQA;
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        uint qbase = (head0 + g) * RKVA_DIM;
        q0[g] = asfloat(queries.Load4((qbase + d0) * 4));
        acc0[g] = float4(0, 0, 0, 0);
        maximum[g] = -3.402823466e38f;
        denom[g] = 0.0f;
    }

    // Row-per-lane decode loop: each wave processes 64 rows per step.
    // Wave 0 covers begin + 0..63, begin + 128..191, ...
    // Wave 1 covers begin + 64..127, begin + 192..255, ...
    uint tile_row0 = begin + wave * 64;
    [loop] for (; tile_row0 < end; tile_row0 += n_waves * 64) {
        uint my_row = tile_row0 + lane;
        bool row_valid = (my_row < end);
        uint my_ba = (my_row * n_head_kv + kv_head) * RKVA_ROW_HEAD;

        // QK Phase: Each lane evaluates the full 256-dim dot product for its own row.
        float score[RKVA_GQA];
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) score[g] = 0.0f;

        if (row_valid) {
            [loop] for (uint u_idx = 0; u_idx < RKVA_QUARTERS; ++u_idx) {
                float4 k_val = load_q4(keys, my_ba, u_idx);
                [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                    float4 q_val = WaveReadLaneAt(q0[g], u_idx);
                    score[g] += dot(q_val, k_val);
                }
            }
        } else {
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                score[g] = -3.402823466e38f;
            }
        }

        // Online Softmax across the 64 rows of this wave:
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

        // PV Phase: Thread lane (which owns quarter-unit lane=u) accumulates V across the valid rows of the tile.
        uint rows_in_tile = min(64u, (end > tile_row0) ? (end - tile_row0) : 0u);
        [loop] for (uint r = 0; r < rows_in_tile; ++r) {
            uint v_row = tile_row0 + r;
            uint v_ba = (v_row * n_head_kv + kv_head) * RKVA_ROW_HEAD;
            float4 v_val = load_q4(values, v_ba, u);
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                float row_weight = WaveReadLaneAt(w[g], r);
                acc0[g] += row_weight * v_val;
            }
        }
    }

    // Publish each wave's state, then merge across waves once.
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
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
