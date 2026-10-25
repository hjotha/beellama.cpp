// RKVA v2 optimized Q4_0 positional overflow attention, multi-query tile
// (MTP verification, nq 2..9). One 64-thread group owns (KV head, query tile of
// RKVA_MQ_QT consecutive queries, split) and evaluates all RKVA_GQA Q heads of
// every query in the tile, so each K/V row is loaded and dequantized ONCE for
// up to RKVA_MQ_QT x RKVA_GQA (query, head) pairs instead of once per query.
//
// Work layout per 64-row tile (no per-row cross-lane reduction):
//   scores: thread i owns row tile+i and computes the full 256-wide Q.K dot for
//           every (query, head) of the tile from one K dequantization.
//           Q addresses are group-uniform (scalar-load friendly).
//   softmax: one wave max per (query, head) per tile; the running maximum is
//           group-uniform, each thread keeps a partial denominator for its rows.
//   values: thread i owns 4 output dims (quarter unit i) and accumulates the
//           tile's 64 rows from one V dequantization per row, reading the
//           row weights from groupshared memory.
// Two group barriers per 64-row tile.
//
// Semantics match rkva_range_attn.hlsl with RKVA_RANGE_STORAGE=1 and the
// per-query optimized kernel rkva_range_attn_q4_opt.hlsl: the same Q4_0 values
// (float(q - 8) * f16 scale; the scale is applied once per 32-element block,
// which is exact per element and only changes FP32 summation order), FP32
// dot/softmax/V accumulation, each query's own causal limit (origin, its
// position, active_count), and the same partial layout
// [split][query*n_head + head][head_dim + 2] consumed by rkva_range_reduce_opt.
// Rows beyond a query's limit get weight 0 for that query only. When
// direct_output != 0 (split_k == 1) the group writes the final normalized O
// and LSE in the reducer's output layout.
//
// Supported shape (the host gates every call): head_dim 256, n_head ==
// RKVA_GQA * n_head_kv, wave size 32..128, split_rows a multiple of 64 is
// preferred (any value is correct).
#define RKVA_GQA 6
#define RKVA_DIM 256
#define RKVA_TILE 64
#define RKVA_THREADS 64
#define RKVA_MAX_WAVES 2   // RKVA_THREADS / minimum supported wave size (32)
#define RKVA_ROW_HEAD 144  // 8 Q4_0 blocks x 18 bytes per (row, KV head)
#ifndef RKVA_MQ_QT
#define RKVA_MQ_QT 4
#endif
#define RKVA_QH (RKVA_MQ_QT * RKVA_GQA)
#define RKVA_NEG_MAX (-3.402823466e38f)

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

groupshared float gs_p[RKVA_QH * RKVA_TILE];        // [query*6 + head][tile row] weights
groupshared float gs_red[RKVA_MAX_WAVES * RKVA_QH];  // per-wave max / denominator

// Quarter unit u covers block b = u >> 3, quant bytes 4c..4c+3 (c = (u >> 1) & 3)
// and nibble half h = u & 1: elements b*32 + h*16 + 4c + {0,1,2,3}.
uint quarter_dim(uint u) {
    return (u >> 3) * 32 + (u & 1) * 16 + ((u >> 1) & 3) * 4;
}

// Dequantize one quarter unit of a (row, KV head) Q4_0 record (as the
// per-query kernel).
float4 load_q4(ByteAddressBuffer buffer, uint base, uint u) {
    uint block = base + (u >> 3) * 18;
    uint packed_scale = buffer.Load(block & ~3u);
    float scale = f16tof32((packed_scale >> ((block & 3u) * 8)) & 65535u);
    uint quant = block + 2 + ((u >> 1) & 3) * 4;
    uint aligned = quant & ~3u;
    uint shift = (quant & 3u) * 8;
    uint lo = buffer.Load(aligned);
    uint hi = buffer.Load(shift ? aligned + 4 : aligned); // never reads past the record
    uint word = shift ? (lo >> shift) | (hi << (32 - shift)) : lo;
    word >>= (u & 1) * 4;
    float4 nibble = float4(word & 15u, (word >> 8) & 15u, (word >> 16) & 15u, (word >> 24) & 15u);
    return (nibble - 8.0f) * scale;
}

float4 low_nibbles(uint w) {
    return float4(w & 15u, (w >> 8) & 15u, (w >> 16) & 15u, (w >> 24) & 15u) - 8.0f;
}

float4 high_nibbles(uint w) {
    return float4((w >> 4) & 15u, (w >> 12) & 15u, (w >> 20) & 15u, (w >> 28) & 15u) - 8.0f;
}

// Adds scale * sum_j Q[j] * (q_j - 8) of one 32-element block (quant words qw,
// element 4k+c in the low nibble of byte 4k+c, element 16+4k+c in its high
// nibble) to every present (query, head) score.
void score_block(uint4 qw, float scale, uint dim0, uint q_first, uint nq_tile, uint kv_head,
                 inout float s[RKVA_QH]) {
    float4 lo0 = low_nibbles(qw.x), lo1 = low_nibbles(qw.y);
    float4 lo2 = low_nibbles(qw.z), lo3 = low_nibbles(qw.w);
    float4 hi0 = high_nibbles(qw.x), hi1 = high_nibbles(qw.y);
    float4 hi2 = high_nibbles(qw.z), hi3 = high_nibbles(qw.w);
    [unroll] for (uint qq = 0; qq < RKVA_MQ_QT; ++qq) {
        [branch] if (qq < nq_tile) {
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                uint qbase = (((q_first + qq) * n_head + kv_head * RKVA_GQA + g) * RKVA_DIM + dim0) * 4;
                float block = dot(asfloat(queries.Load4(qbase + 0)), lo0) +
                              dot(asfloat(queries.Load4(qbase + 16)), lo1) +
                              dot(asfloat(queries.Load4(qbase + 32)), lo2) +
                              dot(asfloat(queries.Load4(qbase + 48)), lo3) +
                              dot(asfloat(queries.Load4(qbase + 64)), hi0) +
                              dot(asfloat(queries.Load4(qbase + 80)), hi1) +
                              dot(asfloat(queries.Load4(qbase + 96)), hi2) +
                              dot(asfloat(queries.Load4(qbase + 112)), hi3);
                s[qq * RKVA_GQA + g] += block * scale;
            }
        }
    }
}

[numthreads(RKVA_THREADS, 1, 1)]
void CSRangeAttnQ4Mq(uint tid : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint kv_head = group.x, q_tile = group.y, split = group.z;
    uint wave_size = WaveGetLaneCount();
    uint wave = tid / wave_size;
    uint n_waves = (RKVA_THREADS + wave_size - 1) / wave_size;
    uint q_first = q_tile * RKVA_MQ_QT;
    uint nq_tile = min((uint)RKVA_MQ_QT, n_query - q_first);
    uint d = quarter_dim(tid);

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
        // Scores: one K dequantization per row for every (query, head).
        uint row = tile + tid;
        uint clamped = min(row, end_max - 1); // in-bounds load; masked below
        uint base = (clamped * n_head_kv + kv_head) * RKVA_ROW_HEAD;
        float s[RKVA_QH];
        [unroll] for (uint h0 = 0; h0 < RKVA_QH; ++h0) {
            s[h0] = 0;
        }
        // Two blocks (36 bytes, 4-byte aligned) per iteration:
        // w0 = scale0 | q0[0..1], w1..w3 = q0[2..13], w4 = q0[14..15] | scale1,
        // w5..w8 = q1[0..15].
        [loop] for (uint pair = 0; pair < 4; ++pair) {
            uint addr = base + pair * 36;
            uint4 a = keys.Load4(addr);
            uint4 b = keys.Load4(addr + 16);
            uint w8 = keys.Load(addr + 32);
            uint4 qw0 = uint4((a.x >> 16) | (a.y << 16), (a.y >> 16) | (a.z << 16),
                              (a.z >> 16) | (a.w << 16), (a.w >> 16) | (b.x << 16));
            score_block(qw0, f16tof32(a.x & 65535u), pair * 64, q_first, nq_tile, kv_head, s);
            score_block(uint4(b.y, b.z, b.w, w8), f16tof32(b.x >> 16), pair * 64 + 32, q_first,
                        nq_tile, kv_head, s);
        }

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

        // Values: one V dequantization per row for every (query, head). Rows past
        // end_max have weight 0 and load the last in-range row.
        uint n_rows = min((uint)RKVA_TILE, end_max - tile);
        [loop] for (uint j = 0; j < n_rows; j += 4) {
            uint last = end_max - 1;
            float4 v0 = load_q4(values, (min(tile + j + 0, last) * n_head_kv + kv_head) * RKVA_ROW_HEAD, tid);
            float4 v1 = load_q4(values, (min(tile + j + 1, last) * n_head_kv + kv_head) * RKVA_ROW_HEAD, tid);
            float4 v2 = load_q4(values, (min(tile + j + 2, last) * n_head_kv + kv_head) * RKVA_ROW_HEAD, tid);
            float4 v3 = load_q4(values, (min(tile + j + 3, last) * n_head_kv + kv_head) * RKVA_ROW_HEAD, tid);
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
