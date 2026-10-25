// RKVA v2 optimized F16 / Q8_0 positional overflow attention (decode, GQA-shared K/V).
// Same structure as rkva_range_attn_q4_opt.hlsl: one group owns (KV head, query,
// split) and evaluates all RKVA_GQA Q heads that read that KV head, so every K/V
// row is loaded and decoded once instead of once per Q head. Each wave owns whole
// rows (no group barrier in the row loop); waves merge their online-softmax states
// once at the end.
//
// Semantics match rkva_range_attn.hlsl with RKVA_RANGE_STORAGE=0 (F16) or 2
// (Q8_0): identical decoded values (f16tof32, or float(int8) * f16 scale), FP32
// dot/softmax/V accumulation, the same causal limit (origin, positions,
// active_count) and the same partial layout [split][query*n_head + head]
// [head_dim + 2] consumed by the split-K reducers. Only the floating-point
// summation order differs. When direct_output != 0 (split_k == 1) the group
// writes the final normalized O and LSE in the reducer's output layout.
//
// Specialized by RKVA_OPT_STORAGE (0 = F16, 2 = Q8_0). Supported shape (the
// host gates every call): head_dim 256, n_head == RKVA_GQA * n_head_kv, wave
// size 32..128, threadgroups of 128 threads, one query.
#define RKVA_GQA 6
#define RKVA_DIM 256
#define RKVA_THREADS 128
#define RKVA_MAX_WAVES 4 // RKVA_THREADS / minimum supported wave size (32)
#define RKVA_UNITS 64    // 4-element units per 256-wide row; unit u is dims 4u..4u+3

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

groupshared float gs_max[RKVA_MAX_WAVES * RKVA_GQA];
groupshared float gs_den[RKVA_MAX_WAVES * RKVA_GQA];
groupshared float gs_acc[RKVA_MAX_WAVES * RKVA_GQA * RKVA_DIM];

// Decode unit u (dims 4u..4u+3) of a (row, KV head) record starting at byte
// `base` (4-byte aligned: RKVA_ROW_HEAD is a multiple of 4). Never reads outside
// the record.
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

[numthreads(RKVA_THREADS, 1, 1)]
void CSRangeAttnGqaOpt(uint tid : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint kv_head = group.x, qi = group.y, split = group.z;
    uint wave_size = WaveGetLaneCount();
    uint lane = WaveGetLaneIndex();
    uint wave = tid / wave_size;
    uint n_waves = RKVA_THREADS / wave_size;
    // Slot 1 exists only for 32-lane waves (64 units over 32 lanes).
    bool has1 = wave_size < RKVA_UNITS;
    bool active0 = lane < RKVA_UNITS;
    bool active1 = has1 && lane + wave_size < RKVA_UNITS;
    uint u0 = min(lane, RKVA_UNITS - 1);
    uint u1 = min(lane + wave_size, RKVA_UNITS - 1);
    uint d0 = u0 * 4, d1 = u1 * 4;

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
        float4 ka0 = load_unit(keys, ba, u0), kb0 = load_unit(keys, bb, u0);
        float4 va0 = load_unit(values, ba, u0), vb0 = load_unit(values, bb, u0);
        float4 ka1 = float4(0, 0, 0, 0), kb1 = ka1, va1 = ka1, vb1 = ka1;
        [branch] if (has1) {
            ka1 = load_unit(keys, ba, u1);
            kb1 = load_unit(keys, bb, u1);
            va1 = load_unit(values, ba, u1);
            vb1 = load_unit(values, bb, u1);
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
        float4 ka0 = load_unit(keys, ba, u0), va0 = load_unit(values, ba, u0);
        float4 ka1 = float4(0, 0, 0, 0), va1 = ka1;
        [branch] if (has1) {
            ka1 = load_unit(keys, ba, u1);
            va1 = load_unit(values, ba, u1);
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
