#ifndef RKVA_DECODE_ROWLANE_TEMPLATE_HLSLI
#define RKVA_DECODE_ROWLANE_TEMPLATE_HLSLI

// RKVA v2 generic rowlane decode attention kernel template (Wave64 on RDNA2/Xbox).
// Each wave has 64 lanes corresponding to 64 quarter-units (covering 256 dims).
// QK Phase: Each thread computes the 256-dim dot product for its row using WaveReadLaneAt.
// PV Phase: Each thread accumulates its assigned 4 dimensions across rows of the tile.
// Waves merge online softmax states via groupshared memory at group boundary.

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
#define RKVA_MAX_WAVES 4
#endif

#ifndef RKVA_QUARTERS
#define RKVA_QUARTERS 64
#endif

#ifndef ROWLANE_ENTRY
#error "ROWLANE_ENTRY must be defined before including rkva_decode_rowlane_template.hlsli"
#endif

#ifndef ROWLANE_ROW_HEAD
#error "ROWLANE_ROW_HEAD must be defined before including rkva_decode_rowlane_template.hlsli"
#endif

#ifndef ROWLANE_LOAD
#error "ROWLANE_LOAD must be defined before including rkva_decode_rowlane_template.hlsli"
#endif

#ifndef ROWLANE_QUARTER_DIM
#error "ROWLANE_QUARTER_DIM must be defined before including rkva_decode_rowlane_template.hlsli"
#endif

[numthreads(RKVA_THREADS, 1, 1)]
void ROWLANE_ENTRY(uint tid : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint kv_head = group.x, qi = group.y, split = group.z;
    uint lane = WaveGetLaneIndex();
    uint wave = tid >> 6; // tid / 64: 2 waves of 64 lanes
    uint n_waves = RKVA_THREADS / 64; // 2
    uint u = lane; // On Wave64: 64 lanes cover all 64 quarter-units (256 dims)
    uint d0 = ROWLANE_QUARTER_DIM(u);

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
    uint tile_row0 = begin + wave * 64;
    [loop] for (; tile_row0 < end; tile_row0 += n_waves * 64) {
        uint my_row = tile_row0 + lane;
        bool row_valid = (my_row < end);
        uint my_ba = (my_row * n_head_kv + kv_head) * ROWLANE_ROW_HEAD;

        // QK Phase: Each lane evaluates the full 256-dim dot product for its own row.
        float score[RKVA_GQA];
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) score[g] = 0.0f;

        if (row_valid) {
            [loop] for (uint u_idx = 0; u_idx < RKVA_QUARTERS; ++u_idx) {
                float4 k_val = ROWLANE_LOAD(keys, my_ba, u_idx);
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

        // PV Phase: Thread lane accumulates V across the valid rows of the tile.
        uint rows_in_tile = min(64u, (end > tile_row0) ? (end - tile_row0) : 0u);
        [loop] for (uint r = 0; r < rows_in_tile; ++r) {
            uint v_row = tile_row0 + r;
            uint v_ba = (v_row * n_head_kv + kv_head) * ROWLANE_ROW_HEAD;
            float4 v_val = ROWLANE_LOAD(values, v_ba, u);
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

#endif // RKVA_DECODE_ROWLANE_TEMPLATE_HLSLI
