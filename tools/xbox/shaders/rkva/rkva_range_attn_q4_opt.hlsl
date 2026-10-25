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
#define RKVA_MAX_WAVES 4    // RKVA_THREADS / minimum supported wave size (32)
#define RKVA_ROW_HEAD 144   // 8 Q4_0 blocks x 18 bytes per (row, KV head)
#define RKVA_QUARTERS 64    // 4-element units per 256-wide row

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

// Quarter unit u covers block b = u >> 3, quant bytes 4c..4c+3 (c = (u >> 1) & 3)
// and nibble half h = u & 1: elements b*32 + h*16 + 4c + {0,1,2,3}.
uint quarter_dim(uint u) {
    return (u >> 3) * 32 + (u & 1) * 16 + ((u >> 1) & 3) * 4;
}

// Dequantize one quarter unit of a (row, KV head) Q4_0 record starting at byte
// `base` (4-byte aligned: RKVA_ROW_HEAD is a multiple of 4).
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

[numthreads(RKVA_THREADS, 1, 1)]
void CSRangeAttnQ4Opt(uint tid : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint kv_head = group.x, qi = group.y, split = group.z;
    uint wave_size = WaveGetLaneCount();
    uint lane = WaveGetLaneIndex();
    uint wave = tid / wave_size;
    uint n_waves = RKVA_THREADS / wave_size;
    // Slot 1 exists only for 32-lane waves (64 quarter units over 32 lanes).
    bool has1 = wave_size < RKVA_QUARTERS;
    bool active0 = lane < RKVA_QUARTERS;
    bool active1 = has1 && lane + wave_size < RKVA_QUARTERS;
    uint u0 = min(lane, RKVA_QUARTERS - 1);
    uint u1 = min(lane + wave_size, RKVA_QUARTERS - 1);
    uint d0 = quarter_dim(u0), d1 = quarter_dim(u1);

    int query_pos = asint(positions.Load(qi * 4));
    uint causal_count = query_pos < int(origin) ? 0 : uint(query_pos) - origin + 1;
    uint limit = min(active_count, causal_count);
    uint begin = min(limit, split * split_rows);
    uint end = min(limit, (split + 1) * split_rows);

    float4 q0[RKVA_GQA], q1[RKVA_GQA], acc0[RKVA_GQA], acc1[RKVA_GQA];
    float maximum[RKVA_GQA], denom[RKVA_GQA];
    uint head0 = qi * n_head + kv_head * RKVA_GQA;
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        uint qbase = (head0 + g) * RKVA_DIM;
        q0[g] = active0 ? asfloat(queries.Load4((qbase + d0) * 4)) : float4(0, 0, 0, 0);
        q1[g] = active1 ? asfloat(queries.Load4((qbase + d1) * 4)) : float4(0, 0, 0, 0);
        acc0[g] = float4(0, 0, 0, 0);
        acc1[g] = float4(0, 0, 0, 0);
        maximum[g] = -3.402823466e38f;
        denom[g] = 0;
    }

    uint row_stride = n_head_kv * RKVA_ROW_HEAD;
    uint row = begin + wave;
    // Two independent rows per iteration so their loads are in flight together.
    [loop] for (; row + n_waves < end; row += 2 * n_waves) {
        uint ba = (row * n_head_kv + kv_head) * RKVA_ROW_HEAD;
        uint bb = ba + n_waves * row_stride;
        float4 ka0 = load_q4(keys, ba, u0), kb0 = load_q4(keys, bb, u0);
        float4 va0 = load_q4(values, ba, u0), vb0 = load_q4(values, bb, u0);
        float4 ka1 = float4(0, 0, 0, 0), kb1 = ka1, va1 = ka1, vb1 = ka1;
        [branch] if (has1) {
            ka1 = load_q4(keys, ba, u1);
            kb1 = load_q4(keys, bb, u1);
            va1 = load_q4(values, ba, u1);
            vb1 = load_q4(values, bb, u1);
        }
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            float sa = WaveActiveSum(dot(q0[g], ka0) + dot(q1[g], ka1)) * kq_scale;
            float sb = WaveActiveSum(dot(q0[g], kb0) + dot(q1[g], kb1)) * kq_scale;
            float next_max = max(maximum[g], max(sa, sb));
            float old_weight = denom[g] == 0 ? 0 : exp(maximum[g] - next_max);
            float wa = exp(sa - next_max), wb = exp(sb - next_max);
            acc0[g] = acc0[g] * old_weight + wa * va0 + wb * vb0;
            acc1[g] = acc1[g] * old_weight + wa * va1 + wb * vb1;
            denom[g] = denom[g] * old_weight + wa + wb;
            maximum[g] = next_max;
        }
    }
    if (row < end) {
        uint ba = (row * n_head_kv + kv_head) * RKVA_ROW_HEAD;
        float4 ka0 = load_q4(keys, ba, u0), va0 = load_q4(values, ba, u0);
        float4 ka1 = float4(0, 0, 0, 0), va1 = ka1;
        [branch] if (has1) {
            ka1 = load_q4(keys, ba, u1);
            va1 = load_q4(values, ba, u1);
        }
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            float sa = WaveActiveSum(dot(q0[g], ka0) + dot(q1[g], ka1)) * kq_scale;
            float next_max = max(maximum[g], sa);
            float old_weight = denom[g] == 0 ? 0 : exp(maximum[g] - next_max);
            float wa = exp(sa - next_max);
            acc0[g] = acc0[g] * old_weight + wa * va0;
            acc1[g] = acc1[g] * old_weight + wa * va1;
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
        if (active0) {
            gs_acc[acc_base + d0 + 0] = acc0[g].x;
            gs_acc[acc_base + d0 + 1] = acc0[g].y;
            gs_acc[acc_base + d0 + 2] = acc0[g].z;
            gs_acc[acc_base + d0 + 3] = acc0[g].w;
        }
        if (active1) {
            gs_acc[acc_base + d1 + 0] = acc1[g].x;
            gs_acc[acc_base + d1 + 1] = acc1[g].y;
            gs_acc[acc_base + d1 + 2] = acc1[g].z;
            gs_acc[acc_base + d1 + 3] = acc1[g].w;
        }
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
