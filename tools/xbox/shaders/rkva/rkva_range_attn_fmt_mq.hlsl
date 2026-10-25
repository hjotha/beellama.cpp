// RKVA v2 optimized F16 / Q8_0 positional overflow attention, multi-query tile
// (MTP verification, nq 2..9). Same algorithm as rkva_range_attn_q4_mq.hlsl:
// one 64-thread group owns (KV head, query tile of RKVA_MQ_QT consecutive
// queries, split) and evaluates all RKVA_GQA Q heads of every query in the
// tile, so each K/V row is loaded and converted ONCE for up to
// RKVA_MQ_QT x RKVA_GQA (query, head) pairs instead of once per query.
//
// Work layout per 64-row tile (no per-row cross-lane reduction):
//   scores: thread i owns row tile+i and computes the full 256-wide Q.K dot for
//           every (query, head) of the tile from one K conversion. Q addresses
//           are group-uniform (scalar-load friendly).
//   softmax: one wave max per (query, head) per tile; the running maximum is
//           group-uniform, each thread keeps a partial denominator for its rows.
//   values: thread i owns dims 4i..4i+3 and accumulates the tile's rows from
//           one V conversion per row, reading the row weights from groupshared
//           memory.
// Two group barriers per 64-row tile, all in uniform control flow.
//
// Semantics match rkva_range_attn.hlsl with RKVA_RANGE_STORAGE=0 (F16) or 2
// (Q8_0) and the per-query optimized kernel rkva_range_attn_gqa_opt.hlsl: the
// same decoded values (f16tof32, or float(int8) * f16 scale; for Q8_0 scores
// the scale is applied once per 32-element block, which is exact per element
// because int8 x f16 fits in FP32, and only changes FP32 summation order),
// FP32 dot/softmax/V accumulation, each query's own causal limit (origin, its
// position, active_count), and the same partial layout
// [split][query*n_head + head][head_dim + 2] consumed by rkva_range_reduce_opt.
// Rows beyond a query's limit get weight 0 for that query only. When
// direct_output != 0 (split_k == 1) the group writes the final normalized O
// and LSE in the reducer's output layout.
//
// Specialized by RKVA_OPT_STORAGE (0 = F16, 2 = Q8_0) and RKVA_MQ_QT (2, 4).
// Supported shape (the host gates every call): head_dim 256, n_head ==
// RKVA_GQA * n_head_kv, wave size 32..128, split_rows a multiple of 64 is
// preferred (any value is correct). Uses the 12-constant format root signature.
#define RKVA_GQA 6
#define RKVA_DIM 256
#define RKVA_TILE 64
#define RKVA_THREADS 64
#define RKVA_MAX_WAVES 2 // RKVA_THREADS / minimum supported wave size (32)
#ifndef RKVA_MQ_QT
#define RKVA_MQ_QT 4
#endif
#define RKVA_QH (RKVA_MQ_QT * RKVA_GQA)
#define RKVA_NEG_MAX (-3.402823466e38f)

#ifndef RKVA_OPT_STORAGE
#error Compile with RKVA_OPT_STORAGE=0 (F16) or 2 (Q8_0).
#endif
#if RKVA_OPT_STORAGE == 0
#define RKVA_ROW_HEAD 512 // 256 halves per (row, KV head)
#elif RKVA_OPT_STORAGE == 2
#define RKVA_ROW_HEAD 272 // 8 Q8_0 blocks x 34 bytes per (row, KV head)
#else
#error RKVA_OPT_STORAGE must be 0 (F16) or 2 (Q8_0).
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
    uint records_bytes;         // unused (KVarN4 only)
    uint precision_tail_tokens; // unused (KVarN4 only)
};
ByteAddressBuffer queries : register(t0);
ByteAddressBuffer keys : register(t1);
ByteAddressBuffer values : register(t2);
ByteAddressBuffer positions : register(t3);
RWByteAddressBuffer partials : register(u0);
RWByteAddressBuffer output : register(u1);

groupshared float gs_p[RKVA_QH * RKVA_TILE];       // [query*6 + head][tile row] weights
groupshared float gs_red[RKVA_MAX_WAVES * RKVA_QH]; // per-wave max / denominator

// Decode unit u (dims 4u..4u+3) of a (row, KV head) record starting at byte
// `base` (4-byte aligned). Identical to rkva_range_attn_gqa_opt.hlsl load_unit;
// never reads outside the record.
float4 load_unit(ByteAddressBuffer buffer, uint base, uint u) {
#if RKVA_OPT_STORAGE == 0
    uint2 packed = buffer.Load2(base + u * 8);
    return float4(f16tof32(packed.x & 65535u), f16tof32(packed.x >> 16),
                  f16tof32(packed.y & 65535u), f16tof32(packed.y >> 16));
#else
    // Block b = u >> 3 starts at 34b: its scale sits at byte 0 or 2 of a dword and
    // its quant bytes 4c..4c+3 (c = u & 7) are dword-aligned only for odd b.
    uint block = base + (u >> 3) * 34;
    uint packed_scale = buffer.Load(block & ~3u);
    float scale = f16tof32((packed_scale >> ((block & 3u) * 8)) & 65535u);
    uint quant = block + 2 + (u & 7) * 4;
    uint aligned = quant & ~3u;
    uint shift = (quant & 3u) * 8;
    uint lo = buffer.Load(aligned);
    uint hi = buffer.Load(shift ? aligned + 4 : aligned); // never reads past the record
    uint word = shift ? (lo >> shift) | (hi << (32 - shift)) : lo;
    int4 quants = int4(asint(word << 24) >> 24, asint(word << 16) >> 24, asint(word << 8) >> 24,
                       asint(word) >> 24);
    return float4(float(quants.x) * scale, float(quants.y) * scale, float(quants.z) * scale,
                  float(quants.w) * scale);
#endif
}

#if RKVA_OPT_STORAGE == 2
// Four sign-extended int8 values of one dword, byte 0 first.
float4 int8x4(uint w) {
    return float4(asint(w << 24) >> 24, asint(w << 16) >> 24, asint(w << 8) >> 24, asint(w) >> 24);
}
#else
float4 halfx4(uint2 w) {
    return float4(f16tof32(w.x & 65535u), f16tof32(w.x >> 16), f16tof32(w.y & 65535u),
                  f16tof32(w.y >> 16));
}
#endif

// Adds scale * sum_j Q[dim0 + j] * e[j] over one 32-element block (e[k] holds
// elements 4k..4k+3) to every present (query, head) score.
void score_block(float4 e[8], float scale, uint dim0, uint q_first, uint nq_tile, uint kv_head,
                 inout float s[RKVA_QH]) {
    [unroll] for (uint qq = 0; qq < RKVA_MQ_QT; ++qq) {
        [branch] if (qq < nq_tile) {
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                uint qbase =
                    (((q_first + qq) * n_head + kv_head * RKVA_GQA + g) * RKVA_DIM + dim0) * 4;
                float block = dot(asfloat(queries.Load4(qbase + 0)), e[0]) +
                              dot(asfloat(queries.Load4(qbase + 16)), e[1]) +
                              dot(asfloat(queries.Load4(qbase + 32)), e[2]) +
                              dot(asfloat(queries.Load4(qbase + 48)), e[3]) +
                              dot(asfloat(queries.Load4(qbase + 64)), e[4]) +
                              dot(asfloat(queries.Load4(qbase + 80)), e[5]) +
                              dot(asfloat(queries.Load4(qbase + 96)), e[6]) +
                              dot(asfloat(queries.Load4(qbase + 112)), e[7]);
#if RKVA_OPT_STORAGE == 2
                s[qq * RKVA_GQA + g] += block * scale;
#else
                s[qq * RKVA_GQA + g] += block;
#endif
            }
        }
    }
}

[numthreads(RKVA_THREADS, 1, 1)]
void CSRangeAttnFmtMq(uint tid : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint kv_head = group.x, q_tile = group.y, split = group.z;
    uint wave_size = WaveGetLaneCount();
    uint wave = tid / wave_size;
    uint n_waves = (RKVA_THREADS + wave_size - 1) / wave_size;
    uint q_first = q_tile * RKVA_MQ_QT;
    uint nq_tile = min((uint)RKVA_MQ_QT, n_query - q_first);
    uint d = tid * 4;

    // Each query keeps its own causal limit; the tile walks their union.
    uint begin = split * split_rows;
    uint split_end = begin + split_rows;
    uint qend[RKVA_MQ_QT];
    uint end_max = 0;
    [unroll] for (uint qq = 0; qq < RKVA_MQ_QT; ++qq) {
        qend[qq] = 0;
        if (qq < nq_tile) {
            int query_pos = asint(positions.Load((q_first + qq) * 4));
            uint causal_count = query_pos < int(origin) ? 0 : uint(query_pos) - origin + 1;
            qend[qq] = min(min(active_count, causal_count), split_end);
        }
        end_max = max(end_max, qend[qq]);
    }

    float maximum[RKVA_QH], dpart[RKVA_QH];
    float4 acc[RKVA_QH];
    [unroll] for (uint h = 0; h < RKVA_QH; ++h) {
        maximum[h] = RKVA_NEG_MAX;
        dpart[h] = 0;
        acc[h] = float4(0, 0, 0, 0);
    }

    [loop] for (uint tile = begin; tile < end_max; tile += RKVA_TILE) {
        // Scores: one K conversion per row for every (query, head).
        uint row = tile + tid;
        uint clamped = min(row, end_max - 1); // in-bounds load; masked below
        uint base = (clamped * n_head_kv + kv_head) * RKVA_ROW_HEAD;
        float s[RKVA_QH];
        [unroll] for (uint h0 = 0; h0 < RKVA_QH; ++h0) {
            s[h0] = 0;
        }
#if RKVA_OPT_STORAGE == 0
        // 64 bytes (32 halves) per block, four 16-byte loads.
        [loop] for (uint blk = 0; blk < 8; ++blk) {
            uint addr = base + blk * 64;
            uint4 a = keys.Load4(addr);
            uint4 b = keys.Load4(addr + 16);
            uint4 c = keys.Load4(addr + 32);
            uint4 e4 = keys.Load4(addr + 48);
            float4 e[8] = {halfx4(a.xy), halfx4(a.zw), halfx4(b.xy), halfx4(b.zw),
                           halfx4(c.xy), halfx4(c.zw), halfx4(e4.xy), halfx4(e4.zw)};
            score_block(e, 1.0f, blk * 32, q_first, nq_tile, kv_head, s);
        }
#else
        // Two blocks (68 bytes, 4-byte aligned) per iteration:
        // w0 = scale0 | q0[0..1], w1..w7 = q0[2..29], w8 = q0[30..31] | scale1,
        // w9..w16 = q1[0..31].
        [loop] for (uint pair = 0; pair < 4; ++pair) {
            uint addr = base + pair * 68;
            uint4 a = keys.Load4(addr);
            uint4 b = keys.Load4(addr + 16);
            uint w8 = keys.Load(addr + 32);
            float4 e0[8] = {int8x4((a.x >> 16) | (a.y << 16)), int8x4((a.y >> 16) | (a.z << 16)),
                            int8x4((a.z >> 16) | (a.w << 16)), int8x4((a.w >> 16) | (b.x << 16)),
                            int8x4((b.x >> 16) | (b.y << 16)), int8x4((b.y >> 16) | (b.z << 16)),
                            int8x4((b.z >> 16) | (b.w << 16)), int8x4((b.w >> 16) | (w8 << 16))};
            score_block(e0, f16tof32(a.x & 65535u), pair * 64, q_first, nq_tile, kv_head, s);
            uint4 c = keys.Load4(addr + 36);
            uint4 e4 = keys.Load4(addr + 52);
            float4 e1[8] = {int8x4(c.x),  int8x4(c.y),  int8x4(c.z),  int8x4(c.w),
                            int8x4(e4.x), int8x4(e4.y), int8x4(e4.z), int8x4(e4.w)};
            score_block(e1, f16tof32(w8 >> 16), pair * 64 + 32, q_first, nq_tile, kv_head, s);
        }
#endif

        // Online softmax with a group-uniform running maximum per (query, head).
        [unroll] for (uint qq = 0; qq < RKVA_MQ_QT; ++qq) {
            [branch] if (qq < nq_tile) {
                bool valid = row < qend[qq];
                [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                    uint h = qq * RKVA_GQA + g;
                    s[h] = valid ? s[h] * kq_scale : RKVA_NEG_MAX;
                    float wave_max = WaveActiveMax(s[h]);
                    if (WaveIsFirstLane()) {
                        gs_red[wave * RKVA_QH + h] = wave_max;
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint qq = 0; qq < RKVA_MQ_QT; ++qq) {
            [branch] if (qq < nq_tile) {
                bool valid = row < qend[qq];
                [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                    uint h = qq * RKVA_GQA + g;
                    float tile_max = gs_red[h];
                    for (uint w = 1; w < n_waves; ++w) {
                        tile_max = max(tile_max, gs_red[w * RKVA_QH + h]);
                    }
                    float next_max = WaveReadLaneFirst(max(maximum[h], tile_max));
                    float alpha = exp(maximum[h] - next_max);
                    float p = valid ? exp(s[h] - next_max) : 0;
                    gs_p[h * RKVA_TILE + tid] = p;
                    dpart[h] = dpart[h] * alpha + p;
                    acc[h] *= alpha;
                    maximum[h] = next_max;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Values: one V conversion per row for every (query, head). Rows past
        // end_max have weight 0 and load the last in-range row.
        uint n_rows = min((uint)RKVA_TILE, end_max - tile);
        uint last = end_max - 1;
        [loop] for (uint j = 0; j < n_rows; j += 4) {
            float4 v0 = load_unit(values, (min(tile + j + 0, last) * n_head_kv + kv_head) * RKVA_ROW_HEAD, tid);
            float4 v1 = load_unit(values, (min(tile + j + 1, last) * n_head_kv + kv_head) * RKVA_ROW_HEAD, tid);
            float4 v2 = load_unit(values, (min(tile + j + 2, last) * n_head_kv + kv_head) * RKVA_ROW_HEAD, tid);
            float4 v3 = load_unit(values, (min(tile + j + 3, last) * n_head_kv + kv_head) * RKVA_ROW_HEAD, tid);
            [unroll] for (uint qq = 0; qq < RKVA_MQ_QT; ++qq) {
                [branch] if (qq < nq_tile) {
                    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                        uint h = qq * RKVA_GQA + g;
                        uint pi = h * RKVA_TILE + j;
                        acc[h] += gs_p[pi] * v0 + gs_p[pi + 1] * v1 + gs_p[pi + 2] * v2 +
                                  gs_p[pi + 3] * v3;
                    }
                }
            }
        }
        // The next tile's first barrier orders these gs_p reads before its writes.
    }

    // Denominators: sum the per-thread partials once. The running maximum and
    // each thread's 4 accumulated dims are already complete.
    [unroll] for (uint qq = 0; qq < RKVA_MQ_QT; ++qq) {
        [branch] if (qq < nq_tile) {
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                uint h = qq * RKVA_GQA + g;
                float wave_den = WaveActiveSum(dpart[h]);
                if (WaveIsFirstLane()) {
                    gs_red[wave * RKVA_QH + h] = wave_den;
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint total = n_query * n_head;
    [unroll] for (uint qq = 0; qq < RKVA_MQ_QT; ++qq) {
        [branch] if (qq < nq_tile) {
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                uint h = qq * RKVA_GQA + g;
                float den = 0;
                for (uint w = 0; w < n_waves; ++w) {
                    den += gs_red[w * RKVA_QH + h];
                }
                uint qh = (q_first + qq) * n_head + kv_head * RKVA_GQA + g;
                if (direct_output != 0) {
                    float inv = den > 0 ? 1 / den : 0;
                    output.Store4((qh * RKVA_DIM + d) * 4, asuint(acc[h] * inv));
                    if (tid == 0) {
                        float lse = den > 0 ? maximum[h] + log(den) : asfloat(0xff800000);
                        output.Store((total * RKVA_DIM + qh) * 4, asuint(lse));
                    }
                } else {
                    uint dst = ((split * total + qh) * (RKVA_DIM + 2)) * 4;
                    partials.Store4(dst + d * 4, asuint(acc[h]));
                    if (tid == 0) {
                        partials.Store(dst + RKVA_DIM * 4, asuint(den > 0 ? maximum[h] : RKVA_NEG_MAX));
                        partials.Store(dst + (RKVA_DIM + 1) * 4, asuint(den));
                    }
                }
            }
        }
    }
}
