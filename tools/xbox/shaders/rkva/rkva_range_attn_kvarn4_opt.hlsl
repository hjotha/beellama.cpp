// RKVA v2 optimized native Bee KVarN4 range attention (decode, GQA-shared K/V).
// Reads the same descriptor-v1 buffers as rkva_range_attn_kvarn4.hlsl: 8960-byte
// KVarN4 records per (128-token group, physical head = kv_head*2 + slice), the
// rotated F16 stage (sink slot 0 and live slots 1..2) and the optional
// original-domain F16 precision tail. Nothing is re-quantized, renumbered or
// materialized: every K/V element is decoded from its native source with the
// reference expression
//     precise affine = float(quant) * scale + zero;  value = affine * other
// (K: scale/zero per dim, other per token; V: scale/zero per token, other per
// dim), staged rows are the stage halves and tail rows the tail halves.
//
// Semantics follow the reference kernel: Q is rotated by the same Bee WHT256
// butterfly sequence (H128 per slice, 1/sqrt(128), then the cross-slice H2 mix);
// record/stage rows score against WHT(Q) and accumulate V in the rotated
// domain, tail rows (row >= active - precision_tail_tokens) score against the
// original Q and accumulate in the original domain; the split numerator is
// WHT(body) + tail with the shared softmax maximum/denominator. Causal limit,
// partial layout [split][query*n_head + head][258] and the reducer contract
// are unchanged. direct_output (split_k == 1) writes final O/LSE.
//
// Differences from the reference are summation order only: one group serves
// the RKVA_GQA Q heads of a KV head, waves own whole rows / 8-token tiles, and
// body rows wholly inside complete record groups are processed as aligned
// 8-token tiles (one dword per K dim carries 8 tokens). Rows in the sink /
// live stage groups, the precision tail and unaligned split edges take the
// per-row path. Supported shape (host-gated): head_dim 256, n_head ==
// RKVA_GQA * n_head_kv, wave size 32..128, 128-thread groups, one query,
// origin % 128 == 0, precision_tail_tokens 0 or 128.
//
// RKVA_SERIAL_PREFIX=1 (descriptor revision 3, separate PSO): every query reads
// the image of its own causal prefix q_count = limit: live staged group, record
// bound (record_hi / fast_hi never reach a group sealed after q_count) and the
// exact-tail window follow q_count, and the ORIGINAL-F16 tail is a 256-row
// physical ring (row & 255). Without the define the legacy revision 1/2 source
// is unchanged token for token.
//
// RKVA_DECODE_OPT=1 (descriptor revision 3, nq=1 wave64 decode PSO,
// entry CSRangeAttnKvarn4Decode): opt-in specialization for Xbox Series X wave64
// (128 threads = 2 waves of 64 lanes, one 4-dim unit per lane). Halves LDS from
// 4 wave slots (24 KiB) to 2 wave slots (12 KiB) to double CU/WGP occupancy,
// removes the dead wave32 second-unit accumulators (q1/body1/tail1), ping-pongs
// WHT256 between the two 1536-float LDS halves (cutting barriers per WHT from
// 17 to 9 and skipping body WHT when a split has no body rows), hoists per-group
// K scale/zero and V other metadata across all 8-token tiles of a 128-token
// record group, reduces tile scores to wave-uniform scalars immediately via
// WaveActiveSum, and runs aligned record tiles before residual/tail rows so
// tail accumulators stay zero throughout the record-tile loop and original Q is
// loaded at most once per split when exact-tail rows are present.
#ifndef RKVA_SERIAL_PREFIX
#define RKVA_SERIAL_PREFIX 0
#endif
#ifndef RKVA_DECODE_OPT
#define RKVA_DECODE_OPT 0
#endif
#if RKVA_SERIAL_PREFIX
static uint rkva_window; // per-thread q_count, set once in the entry point
#define RKVA_WINDOW rkva_window
#define RKVA_TAIL_MASK 255u
#else
#define RKVA_WINDOW active_count
#define RKVA_TAIL_MASK 127u
#endif
#define RKVA_GQA 6
#define RKVA_DIM 256
#define RKVA_THREADS 128
#define RKVA_MAX_WAVES 4
#define RKVA_DECODE_WAVES 2
#define RKVA_UNITS 64 // unit u covers dims 4u..4u+3: slice u >> 5, record dims 4(u & 31)..+3
#define RKVA_TILE 8
#define RKVA_RECORD 8960
#define RKVA_ELEMS 12 // (GQA * DIM) / THREADS output elements per thread

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
    uint records_bytes;
    uint precision_tail_tokens;
};
ByteAddressBuffer queries : register(t0);
ByteAddressBuffer keys : register(t1);
ByteAddressBuffer values : register(t2);
ByteAddressBuffer positions : register(t3);
RWByteAddressBuffer partials : register(u0);
RWByteAddressBuffer output : register(u1);

#if RKVA_DECODE_OPT
groupshared float gs_max[RKVA_DECODE_WAVES * RKVA_GQA];
groupshared float gs_den[RKVA_DECODE_WAVES * RKVA_GQA];
// Rotated Q, then per-wave numerators (body, then tail), then the WHT scratch.
// 2 waves * 6 GQA * 256 dim = 3072 floats = 12,288 bytes (~12 KiB LDS, doubling occupancy).
groupshared float gs_vec[RKVA_DECODE_WAVES * RKVA_GQA * RKVA_DIM];
#else
groupshared float gs_max[RKVA_MAX_WAVES * RKVA_GQA];
groupshared float gs_den[RKVA_MAX_WAVES * RKVA_GQA];
// Rotated Q, then per-wave numerators (body, then tail), then the WHT scratch.
groupshared float gs_vec[RKVA_MAX_WAVES * RKVA_GQA * RKVA_DIM];
#endif

float half_at(ByteAddressBuffer buffer, uint byte_offset) {
    uint bits = buffer.Load(byte_offset & ~3u);
    return f16tof32((bits >> ((byte_offset & 2u) * 8)) & 65535u);
}

float4 halves4(uint2 packed) {
    return float4(f16tof32(packed.x & 65535u), f16tof32(packed.x >> 16),
                  f16tof32(packed.y & 65535u), f16tof32(packed.y >> 16));
}

float decode(uint quant, float scale, float zero, float other) {
    precise float affine = float(quant) * scale + zero;
    return affine * other;
}

// In-place Bee WHT256 of the RKVA_GQA vectors in gs_vec[g * 256 + d]. Thread
// tid owns elements v[2g] = (g, tid) and v[2g + 1] = (g, tid + 128), exactly
// the reference (lane, lane + 128) ownership and butterfly order. On return
// v holds the transform and gs_vec[0 .. GQA*256) holds it too.
void wht256x6(inout float v[RKVA_ELEMS], uint tid) {
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        gs_vec[g * RKVA_DIM + tid] = v[2 * g];
        gs_vec[g * RKVA_DIM + tid + 128] = v[2 * g + 1];
    }
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = 1; stride < 128; stride *= 2) {
        float own[RKVA_ELEMS], other[RKVA_ELEMS];
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            own[2 * g] = gs_vec[g * RKVA_DIM + tid];
            other[2 * g] = gs_vec[g * RKVA_DIM + (tid ^ stride)];
            own[2 * g + 1] = gs_vec[g * RKVA_DIM + tid + 128];
            other[2 * g + 1] = gs_vec[g * RKVA_DIM + (tid ^ stride) + 128];
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            gs_vec[g * RKVA_DIM + tid] =
                (tid & stride) ? other[2 * g] - own[2 * g] : own[2 * g] + other[2 * g];
            gs_vec[g * RKVA_DIM + tid + 128] = (tid & stride) ? other[2 * g + 1] - own[2 * g + 1]
                                                              : own[2 * g + 1] + other[2 * g + 1];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        float x = gs_vec[g * RKVA_DIM + tid] * 0.08838834764831845f;
        float y = gs_vec[g * RKVA_DIM + tid + 128] * 0.08838834764831845f;
        v[2 * g] = (x + y) * 0.7071067811865475f;
        v[2 * g + 1] = (x - y) * 0.7071067811865475f;
    }
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        gs_vec[g * RKVA_DIM + tid] = v[2 * g];
        gs_vec[g * RKVA_DIM + tid + 128] = v[2 * g + 1];
    }
    GroupMemoryBarrierWithGroupSync();
}

#if RKVA_DECODE_OPT
// Ping-pong Bee WHT256 using the two 1536-float halves of gs_vec[3072].
// Alternates source and destination between half 0 (0..1535) and half 1 (1536..3071),
// cutting group barriers from 17 to 9.
void wht256x6_decode(inout float v[RKVA_ELEMS], uint tid) {
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        gs_vec[g * RKVA_DIM + tid] = v[2 * g];
        gs_vec[g * RKVA_DIM + tid + 128] = v[2 * g + 1];
    }
    GroupMemoryBarrierWithGroupSync();

    uint src_offset = 0;
    uint dst_offset = 1536;

    [unroll] for (uint stride = 1; stride < 128; stride *= 2) {
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            float own0 = gs_vec[src_offset + g * RKVA_DIM + tid];
            float other0 = gs_vec[src_offset + g * RKVA_DIM + (tid ^ stride)];
            float own1 = gs_vec[src_offset + g * RKVA_DIM + tid + 128];
            float other1 = gs_vec[src_offset + g * RKVA_DIM + (tid ^ stride) + 128];

            gs_vec[dst_offset + g * RKVA_DIM + tid] =
                (tid & stride) ? other0 - own0 : own0 + other0;
            gs_vec[dst_offset + g * RKVA_DIM + tid + 128] =
                (tid & stride) ? other1 - own1 : own1 + other1;
        }
        GroupMemoryBarrierWithGroupSync();
        uint tmp = src_offset;
        src_offset = dst_offset;
        dst_offset = tmp;
    }

    // After 7 iterations (strides 1, 2, 4, 8, 16, 32, 64), src_offset is 1536.
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        float x = gs_vec[src_offset + g * RKVA_DIM + tid] * 0.08838834764831845f;
        float y = gs_vec[src_offset + g * RKVA_DIM + tid + 128] * 0.08838834764831845f;
        v[2 * g] = (x + y) * 0.7071067811865475f;
        v[2 * g + 1] = (x - y) * 0.7071067811865475f;
        gs_vec[g * RKVA_DIM + tid] = v[2 * g];
        gs_vec[g * RKVA_DIM + tid + 128] = v[2 * g + 1];
    }
    GroupMemoryBarrierWithGroupSync();
}
#endif

// ---- Per-row path (stage, tail, record rows at split edges) ----

// Original-domain tail halves for dims 4u..4u+3 of (row, kv_head).
float4 load_tail4(ByteAddressBuffer buffer, uint row, uint kv_head, uint u) {
    uint stage_bytes = 384 * n_head_kv * 256 * 2;
    uint element = ((row & RKVA_TAIL_MASK) * n_head_kv + kv_head) * 256 + u * 4;
    return halves4(buffer.Load2(records_bytes + stage_bytes + element * 2));
}

// Rotated-domain body dims 4u..4u+3 of (row, kv_head): stage halves or record.
float4 load_body4(ByteAddressBuffer buffer, bool value, uint row, uint kv_head, uint u,
                  uint live_group) {
    uint group = row / 128, pos = row & 127;
    uint global_group = origin / 128 + group;
    bool staged = global_group == 0 || (group == live_group && (RKVA_WINDOW & 127u) != 0);
    if (staged) {
        uint slot = global_group == 0 ? 0 : 1 + ((global_group - 1) & 1);
        // ((slot*128 + pos) * physical_heads + kv*2 + slice) * 128 + dim
        uint element = ((slot * 128 + pos) * n_head_kv + kv_head) * 256 + u * 4;
        return halves4(buffer.Load2(records_bytes + element * 2));
    }
    uint dim = (u & 31) * 4;
    uint base = (group * n_head_kv * 2 + kv_head * 2 + (u >> 5)) * RKVA_RECORD;
    float4 result;
    if (value) {
        // V: index = pos*128 + dim; 4 consecutive nibbles; scale/zero per token.
        uint index = pos * 128 + dim;
        uint word = buffer.Load(base + (index / 8) * 4) >> ((index & 7) * 4);
        float scale = half_at(buffer, base + 8192 + pos * 2);
        float zero = half_at(buffer, base + 8448 + pos * 2);
        float4 other = halves4(buffer.Load2(base + 8704 + dim * 2));
        result.x = decode(word & 15u, scale, zero, other.x);
        result.y = decode((word >> 4) & 15u, scale, zero, other.y);
        result.z = decode((word >> 8) & 15u, scale, zero, other.z);
        result.w = decode((word >> 12) & 15u, scale, zero, other.w);
    } else {
        // K: index = dim*128 + pos; one nibble per dim; scale/zero per dim.
        float4 scale = halves4(buffer.Load2(base + 8192 + dim * 2));
        float4 zero = halves4(buffer.Load2(base + 8448 + dim * 2));
        float other = half_at(buffer, base + 8704 + pos * 2);
        uint shift = (pos & 7) * 4;
        uint nib_base = base + (pos / 8) * 4;
        result.x = decode((buffer.Load(nib_base + (dim + 0) * 64) >> shift) & 15u, scale.x, zero.x, other);
        result.y = decode((buffer.Load(nib_base + (dim + 1) * 64) >> shift) & 15u, scale.y, zero.y, other);
        result.z = decode((buffer.Load(nib_base + (dim + 2) * 64) >> shift) & 15u, scale.z, zero.z, other);
        result.w = decode((buffer.Load(nib_base + (dim + 3) * 64) >> shift) & 15u, scale.w, zero.w, other);
    }
    return result;
}

// ---- 8-token record tile path ----

// Adds this unit's contribution to the 8 token scores of every Q head.
void tile_scores(uint base, uint pos0, uint u, float4 q[RKVA_GQA],
                 inout float score[RKVA_GQA * RKVA_TILE]) {
    uint dim = (u & 31) * 4;
    float4 scale = halves4(keys.Load2(base + 8192 + dim * 2));
    float4 zero = halves4(keys.Load2(base + 8448 + dim * 2));
    uint4 other_bits = keys.Load4(base + 8704 + pos0 * 2);
    float other[RKVA_TILE];
    [unroll] for (uint p = 0; p < 4; ++p) {
        other[2 * p] = f16tof32(other_bits[p] & 65535u);
        other[2 * p + 1] = f16tof32(other_bits[p] >> 16);
    }
    uint nib_base = base + (pos0 / 8) * 4;
    uint words[4] = {keys.Load(nib_base + (dim + 0) * 64), keys.Load(nib_base + (dim + 1) * 64),
                     keys.Load(nib_base + (dim + 2) * 64), keys.Load(nib_base + (dim + 3) * 64)};
    [unroll] for (uint t = 0; t < RKVA_TILE; ++t) {
        float4 k;
        k.x = decode((words[0] >> (4 * t)) & 15u, scale.x, zero.x, other[t]);
        k.y = decode((words[1] >> (4 * t)) & 15u, scale.y, zero.y, other[t]);
        k.z = decode((words[2] >> (4 * t)) & 15u, scale.z, zero.z, other[t]);
        k.w = decode((words[3] >> (4 * t)) & 15u, scale.w, zero.w, other[t]);
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            score[g * RKVA_TILE + t] += dot(q[g], k);
        }
    }
}

// Value dims 4u..4u+3 of token pos (record V plane).
float4 tile_value(uint base, uint pos, uint u, float scale, float zero, float4 other) {
    uint index = pos * 128 + (u & 31) * 4;
    uint word = values.Load(base + (index / 8) * 4) >> ((index & 7) * 4);
    float4 v;
    v.x = decode(word & 15u, scale, zero, other.x);
    v.y = decode((word >> 4) & 15u, scale, zero, other.y);
    v.z = decode((word >> 8) & 15u, scale, zero, other.z);
    v.w = decode((word >> 12) & 15u, scale, zero, other.w);
    return v;
}

[numthreads(RKVA_THREADS, 1, 1)]
void CSRangeAttnKvarn4Opt(uint tid : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint kv_head = group.x, qi = group.y, split = group.z;
    uint wave_size = WaveGetLaneCount();
    uint lane = WaveGetLaneIndex();
    uint wave = tid / wave_size;
    uint n_waves = RKVA_THREADS / wave_size;
    bool has1 = wave_size < RKVA_UNITS;
    bool active0 = lane < RKVA_UNITS;
    bool active1 = has1 && lane + wave_size < RKVA_UNITS;
    uint u0 = min(lane, RKVA_UNITS - 1);
    uint u1 = min(lane + wave_size, RKVA_UNITS - 1);
    uint head0 = qi * n_head + kv_head * RKVA_GQA;

    // Rotate the GQA Q heads exactly like the reference (bit-identical inputs).
    float v[RKVA_ELEMS];
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        v[2 * g] = asfloat(queries.Load(((head0 + g) * RKVA_DIM + tid) * 4));
        v[2 * g + 1] = asfloat(queries.Load(((head0 + g) * RKVA_DIM + tid + 128) * 4));
    }
    wht256x6(v, tid);
    float4 q0[RKVA_GQA], q1[RKVA_GQA];
    float4 body0[RKVA_GQA], body1[RKVA_GQA], tail0[RKVA_GQA], tail1[RKVA_GQA];
    float maximum[RKVA_GQA], denom[RKVA_GQA];
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        uint b0 = g * RKVA_DIM + u0 * 4, b1 = g * RKVA_DIM + u1 * 4;
        q0[g] = active0 ? float4(gs_vec[b0], gs_vec[b0 + 1], gs_vec[b0 + 2], gs_vec[b0 + 3])
                        : float4(0, 0, 0, 0);
        q1[g] = active1 ? float4(gs_vec[b1], gs_vec[b1 + 1], gs_vec[b1 + 2], gs_vec[b1 + 3])
                        : float4(0, 0, 0, 0);
        body0[g] = float4(0, 0, 0, 0);
        body1[g] = body0[g];
        tail0[g] = body0[g];
        tail1[g] = body0[g];
        maximum[g] = -3.402823466e38f;
        denom[g] = 0;
    }
    GroupMemoryBarrierWithGroupSync(); // gs_vec is reused for the wave states below

    int query_pos = asint(positions.Load(qi * 4));
    uint causal_count = query_pos < int(origin) ? 0 : uint(query_pos) - origin + 1;
    uint limit = min(active_count, causal_count);
#if RKVA_SERIAL_PREFIX
    rkva_window = limit;
#endif
    uint begin = min(limit, split * split_rows);
    uint end = min(limit, (split + 1) * split_rows);
    uint live_group = RKVA_WINDOW ? (RKVA_WINDOW - 1) / 128 : 0;
    uint tail_begin = RKVA_WINDOW - min(RKVA_WINDOW, precision_tail_tokens);
    // Record-backed body rows: complete groups, excluding the origin-0 sink group.
    uint record_lo = origin == 0 ? 128 : 0;
    uint record_hi = RKVA_WINDOW & ~127u;
    uint fast_lo = (max(begin, record_lo) + RKVA_TILE - 1) & ~(RKVA_TILE - 1);
    uint fast_hi = min(min(end, record_hi), tail_begin) & ~(RKVA_TILE - 1);
    if (fast_hi <= fast_lo) {
        fast_lo = end;
        fast_hi = end;
    }

    // Per-row path over [begin, fast_lo) and [fast_hi, end).
    [loop] for (uint segment = 0; segment < 2; ++segment) {
        uint seg_begin = segment == 0 ? begin : fast_hi;
        uint seg_end = segment == 0 ? fast_lo : end;
        [loop] for (uint row = seg_begin + wave; row < seg_end; row += n_waves) {
            bool exact = row >= tail_begin; // wave-uniform
            float4 k0, k1 = float4(0, 0, 0, 0), x0, x1 = float4(0, 0, 0, 0);
            float4 qa0[RKVA_GQA], qa1[RKVA_GQA];
            if (exact) {
                k0 = load_tail4(keys, row, kv_head, u0);
                x0 = load_tail4(values, row, kv_head, u0);
                [branch] if (has1) {
                    k1 = load_tail4(keys, row, kv_head, u1);
                    x1 = load_tail4(values, row, kv_head, u1);
                }
                [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                    uint qbase = (head0 + g) * RKVA_DIM;
                    qa0[g] = active0 ? asfloat(queries.Load4((qbase + u0 * 4) * 4))
                                     : float4(0, 0, 0, 0);
                    qa1[g] = active1 ? asfloat(queries.Load4((qbase + u1 * 4) * 4))
                                     : float4(0, 0, 0, 0);
                }
            } else {
                k0 = load_body4(keys, false, row, kv_head, u0, live_group);
                x0 = load_body4(values, true, row, kv_head, u0, live_group);
                [branch] if (has1) {
                    k1 = load_body4(keys, false, row, kv_head, u1, live_group);
                    x1 = load_body4(values, true, row, kv_head, u1, live_group);
                }
                [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                    qa0[g] = q0[g];
                    qa1[g] = q1[g];
                }
            }
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                float s = WaveActiveSum(dot(qa0[g], k0) + dot(qa1[g], k1)) * kq_scale;
                float next_max = max(maximum[g], s);
                float old_weight = denom[g] == 0 ? 0 : exp(maximum[g] - next_max);
                float w = exp(s - next_max);
                body0[g] *= old_weight;
                body1[g] *= old_weight;
                tail0[g] *= old_weight;
                tail1[g] *= old_weight;
                if (exact) {
                    tail0[g] += w * x0;
                    tail1[g] += w * x1;
                } else {
                    body0[g] += w * x0;
                    body1[g] += w * x1;
                }
                denom[g] = denom[g] * old_weight + w;
                maximum[g] = next_max;
            }
        }
    }

    // Record tiles: 8 aligned tokens of one complete record group per iteration.
    [loop] for (uint t0 = fast_lo + wave * RKVA_TILE; t0 < fast_hi; t0 += n_waves * RKVA_TILE) {
        uint rgroup = t0 / 128, pos0 = t0 & 127;
        uint base0 = (rgroup * n_head_kv * 2 + kv_head * 2 + (u0 >> 5)) * RKVA_RECORD;
        uint base1 = (rgroup * n_head_kv * 2 + kv_head * 2 + (u1 >> 5)) * RKVA_RECORD;
        float score[RKVA_GQA * RKVA_TILE];
        [unroll] for (uint i = 0; i < RKVA_GQA * RKVA_TILE; ++i) score[i] = 0;
        tile_scores(base0, pos0, u0, q0, score);
        [branch] if (has1) {
            tile_scores(base1, pos0, u1, q1, score);
        }
        // Scores -> weights; rescale the running state once per tile.
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            float tile_max = -3.402823466e38f;
            [unroll] for (uint t = 0; t < RKVA_TILE; ++t) {
                score[g * RKVA_TILE + t] = WaveActiveSum(score[g * RKVA_TILE + t]) * kq_scale;
                tile_max = max(tile_max, score[g * RKVA_TILE + t]);
            }
            float next_max = max(maximum[g], tile_max);
            float old_weight = denom[g] == 0 ? 0 : exp(maximum[g] - next_max);
            float sum = 0;
            [unroll] for (uint t = 0; t < RKVA_TILE; ++t) {
                score[g * RKVA_TILE + t] = exp(score[g * RKVA_TILE + t] - next_max);
                sum += score[g * RKVA_TILE + t];
            }
            body0[g] *= old_weight;
            body1[g] *= old_weight;
            tail0[g] *= old_weight;
            tail1[g] *= old_weight;
            denom[g] = denom[g] * old_weight + sum;
            maximum[g] = next_max;
        }
        uint4 scale_bits0 = values.Load4(base0 + 8192 + pos0 * 2);
        uint4 zero_bits0 = values.Load4(base0 + 8448 + pos0 * 2);
        float4 other0 = halves4(values.Load2(base0 + 8704 + (u0 & 31) * 8));
        float4 other1 = float4(0, 0, 0, 0);
        uint4 scale_bits1 = uint4(0, 0, 0, 0), zero_bits1 = scale_bits1;
        [branch] if (has1) {
            scale_bits1 = values.Load4(base1 + 8192 + pos0 * 2);
            zero_bits1 = values.Load4(base1 + 8448 + pos0 * 2);
            other1 = halves4(values.Load2(base1 + 8704 + (u1 & 31) * 8));
        }
        [unroll] for (uint t = 0; t < RKVA_TILE; ++t) {
            uint sh = (t & 1) * 16;
            float4 x0 = tile_value(base0, pos0 + t, u0,
                                   f16tof32((scale_bits0[t / 2] >> sh) & 65535u),
                                   f16tof32((zero_bits0[t / 2] >> sh) & 65535u), other0);
            float4 x1 = float4(0, 0, 0, 0);
            [branch] if (has1) {
                x1 = tile_value(base1, pos0 + t, u1, f16tof32((scale_bits1[t / 2] >> sh) & 65535u),
                                f16tof32((zero_bits1[t / 2] >> sh) & 65535u), other1);
            }
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                body0[g] += score[g * RKVA_TILE + t] * x0;
                body1[g] += score[g * RKVA_TILE + t] * x1;
            }
        }
    }

    // ---- Cross-wave merge: body numerators, then tail numerators ----
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        uint slot = wave * RKVA_GQA + g;
        // Reject the whole wave state before a zero merge weight can hide it.
        bool invalid_state = !isfinite(denom[g]) || denom[g] < 0 ||
            (denom[g] != 0 && (!isfinite(maximum[g]) || !(all(isfinite(body0[g])) && all(isfinite(body1[g])) && all(isfinite(tail0[g])) && all(isfinite(tail1[g])))));
        bool invalid_wave = WaveActiveAnyTrue(invalid_state);
        if (lane == 0) {
            gs_max[slot] = invalid_wave ? asfloat(0x7fc00000u) : maximum[g];
            gs_den[slot] = invalid_wave ? asfloat(0x7fc00000u) : denom[g];
        }
        if (active0) {
            gs_vec[slot * RKVA_DIM + u0 * 4 + 0] = body0[g].x;
            gs_vec[slot * RKVA_DIM + u0 * 4 + 1] = body0[g].y;
            gs_vec[slot * RKVA_DIM + u0 * 4 + 2] = body0[g].z;
            gs_vec[slot * RKVA_DIM + u0 * 4 + 3] = body0[g].w;
        }
        if (active1) {
            gs_vec[slot * RKVA_DIM + u1 * 4 + 0] = body1[g].x;
            gs_vec[slot * RKVA_DIM + u1 * 4 + 1] = body1[g].y;
            gs_vec[slot * RKVA_DIM + u1 * 4 + 2] = body1[g].z;
            gs_vec[slot * RKVA_DIM + u1 * 4 + 3] = body1[g].w;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Thread tid owns (g, tid) and (g, tid + 128) for every head g.
    float merged_max[RKVA_GQA], merged_den[RKVA_GQA], weight[RKVA_MAX_WAVES * RKVA_GQA];
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        merged_max[g] = -3.402823466e38f;
        for (uint w = 0; w < n_waves; ++w) {
            if (gs_den[w * RKVA_GQA + g] != 0) merged_max[g] = max(merged_max[g], gs_max[w * RKVA_GQA + g]);
        }
        merged_den[g] = 0;
        [unroll] for (uint w = 0; w < RKVA_MAX_WAVES; ++w) {
            uint slot = w * RKVA_GQA + g;
            weight[slot] = 0;
            if (w < n_waves && gs_den[slot] != 0) {
                weight[slot] = exp(gs_max[slot] - merged_max[g]);
                merged_den[g] += gs_den[slot] * weight[slot];
            }
        }
    }
    float body[RKVA_ELEMS], tail[RKVA_ELEMS];
    [unroll] for (uint k = 0; k < RKVA_ELEMS; ++k) {
        uint g = k / 2, d = tid + (k & 1) * 128;
        body[k] = 0;
        tail[k] = 0;
        [unroll] for (uint w = 0; w < RKVA_MAX_WAVES; ++w) {
            uint slot = w * RKVA_GQA + g;
            if (weight[slot] != 0) body[k] += gs_vec[slot * RKVA_DIM + d] * weight[slot];
        }
    }
    bool has_tail = end > max(begin, tail_begin); // group-uniform
    if (has_tail) {
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            uint slot = wave * RKVA_GQA + g;
            if (active0) {
                gs_vec[slot * RKVA_DIM + u0 * 4 + 0] = tail0[g].x;
                gs_vec[slot * RKVA_DIM + u0 * 4 + 1] = tail0[g].y;
                gs_vec[slot * RKVA_DIM + u0 * 4 + 2] = tail0[g].z;
                gs_vec[slot * RKVA_DIM + u0 * 4 + 3] = tail0[g].w;
            }
            if (active1) {
                gs_vec[slot * RKVA_DIM + u1 * 4 + 0] = tail1[g].x;
                gs_vec[slot * RKVA_DIM + u1 * 4 + 1] = tail1[g].y;
                gs_vec[slot * RKVA_DIM + u1 * 4 + 2] = tail1[g].z;
                gs_vec[slot * RKVA_DIM + u1 * 4 + 3] = tail1[g].w;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint k = 0; k < RKVA_ELEMS; ++k) {
            uint g = k / 2, d = tid + (k & 1) * 128;
            [unroll] for (uint w = 0; w < RKVA_MAX_WAVES; ++w) {
                uint slot = w * RKVA_GQA + g;
                if (weight[slot] != 0) tail[k] += gs_vec[slot * RKVA_DIM + d] * weight[slot];
            }
        }
    }
    if (direct_output != 0) {
        GroupMemoryBarrierWithGroupSync();
        wht256x6(body, tid); // linear: WHT(sum of wave bodies) == sum of WHT(wave bodies)
        uint total = n_query * n_head;
        [unroll] for (uint k = 0; k < RKVA_ELEMS; ++k) {
            uint g = k / 2, d = tid + (k & 1) * 128;
            uint qh = head0 + g;
            float numerator = body[k] + tail[k];
            float inv = merged_den[g] == 0 ? 0 : 1 / merged_den[g];
            output.Store((qh * RKVA_DIM + d) * 4, asuint(numerator * inv));
            if (d == 0) {
                float lse = merged_den[g] != 0 ? merged_max[g] + log(merged_den[g])
                                              : asfloat(0xff800000);
                output.Store((total * RKVA_DIM + qh) * 4, asuint(lse));
            }
        }
    } else {
        if (has_tail) {
            GroupMemoryBarrierWithGroupSync();
            wht256x6(tail, tid);
        }
        uint total = n_query * n_head;
        [unroll] for (uint k = 0; k < RKVA_ELEMS; ++k) {
            uint g = k / 2, d = tid + (k & 1) * 128;
            uint qh = head0 + g;
            float numerator = has_tail ? (body[k] + tail[k]) : body[k];
            uint dst = ((split * total + qh) * (RKVA_DIM + 2)) * 4;
            partials.Store(dst + d * 4, asuint(numerator));
            if (d == 0) {
                partials.Store(dst + RKVA_DIM * 4, asuint(merged_max[g]));
                partials.Store(dst + (RKVA_DIM + 1) * 4, asuint(merged_den[g]));
            }
        }
    }
}

#if RKVA_DECODE_OPT
[numthreads(RKVA_THREADS, 1, 1)]
void CSRangeAttnKvarn4Decode(uint tid : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint kv_head = group.x, qi = group.y, split = group.z;
    uint lane = WaveGetLaneIndex();
    uint wave = tid >> 6; // tid / 64: 2 waves of 64 threads on wave64 APU
    uint u = lane;        // wave64: 64 lanes cover all 64 4-dim units (256 dims)
    uint head0 = qi * n_head + kv_head * RKVA_GQA;

    // 1. Rotate the GQA Q heads into WHT domain
    float v[RKVA_ELEMS];
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        v[2 * g] = asfloat(queries.Load(((head0 + g) * RKVA_DIM + tid) * 4));
        v[2 * g + 1] = asfloat(queries.Load(((head0 + g) * RKVA_DIM + tid + 128) * 4));
    }
    wht256x6_decode(v, tid);

    float4 q[RKVA_GQA];
    float4 body[RKVA_GQA], tail[RKVA_GQA];
    float maximum[RKVA_GQA], denom[RKVA_GQA];
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        uint b0 = g * RKVA_DIM + u * 4;
        q[g] = float4(gs_vec[b0], gs_vec[b0 + 1], gs_vec[b0 + 2], gs_vec[b0 + 3]);
        body[g] = float4(0, 0, 0, 0);
        tail[g] = float4(0, 0, 0, 0);
        maximum[g] = -3.402823466e38f;
        denom[g] = 0;
    }
    GroupMemoryBarrierWithGroupSync(); // gs_vec is reused for the wave states below

    int query_pos = asint(positions.Load(qi * 4));
    uint causal_count = query_pos < int(origin) ? 0 : uint(query_pos) - origin + 1;
    uint limit = min(active_count, causal_count);
#if RKVA_SERIAL_PREFIX
    rkva_window = limit;
#endif
    uint begin = min(limit, split * split_rows);
    uint end = min(limit, (split + 1) * split_rows);
    uint live_group = RKVA_WINDOW ? (RKVA_WINDOW - 1) / 128 : 0;
    uint tail_begin = RKVA_WINDOW - min(RKVA_WINDOW, precision_tail_tokens);

    // Record-backed body rows: complete groups, excluding the origin-0 sink group.
    uint record_lo = origin == 0 ? 128 : 0;
    uint record_hi = RKVA_WINDOW & ~127u;
    uint fast_lo = (max(begin, record_lo) + RKVA_TILE - 1) & ~(RKVA_TILE - 1);
    uint fast_hi = min(min(end, record_hi), tail_begin) & ~(RKVA_TILE - 1);
    if (fast_hi <= fast_lo) {
        fast_lo = end;
        fast_hi = end;
    }

    // Phase 1: Record tiles (8 aligned tokens of one complete record group per iteration).
    // Tail is guaranteed 0 during this phase. Hoist group metadata across the 128-token group.
    uint cached_rgroup = 0xFFFFFFFFu;
    uint base_k = 0;
    uint base_v = 0;
    float4 k_scale = float4(0, 0, 0, 0);
    float4 k_zero = float4(0, 0, 0, 0);
    float4 v_other = float4(0, 0, 0, 0);
    uint dim = (u & 31) * 4;

    [loop] for (uint t0 = fast_lo + wave * RKVA_TILE; t0 < fast_hi; t0 += RKVA_DECODE_WAVES * RKVA_TILE) {
        uint rgroup = t0 / 128, pos0 = t0 & 127;
        if (rgroup != cached_rgroup) {
            cached_rgroup = rgroup;
            uint ph = kv_head * 2 + (u >> 5);
            base_k = (rgroup * n_head_kv * 2 + ph) * RKVA_RECORD;
            base_v = base_k;
            k_scale = halves4(keys.Load2(base_k + 8192 + dim * 2));
            k_zero = halves4(keys.Load2(base_k + 8448 + dim * 2));
            v_other = halves4(values.Load2(base_v + 8704 + (u & 31) * 8));
        }
        uint4 other_bits = keys.Load4(base_k + 8704 + pos0 * 2);
        float other[RKVA_TILE];
        [unroll] for (uint p = 0; p < 4; ++p) {
            other[2 * p] = f16tof32(other_bits[p] & 65535u);
            other[2 * p + 1] = f16tof32(other_bits[p] >> 16);
        }
        uint nib_base = base_k + (pos0 / 8) * 4;
        uint words[4] = {keys.Load(nib_base + (dim + 0) * 64), keys.Load(nib_base + (dim + 1) * 64),
                         keys.Load(nib_base + (dim + 2) * 64), keys.Load(nib_base + (dim + 3) * 64)};

        float score[RKVA_GQA * RKVA_TILE];
        [unroll] for (uint i = 0; i < RKVA_GQA * RKVA_TILE; ++i) score[i] = 0;

        [unroll] for (uint t = 0; t < RKVA_TILE; ++t) {
            float4 k_val;
            k_val.x = decode((words[0] >> (4 * t)) & 15u, k_scale.x, k_zero.x, other[t]);
            k_val.y = decode((words[1] >> (4 * t)) & 15u, k_scale.y, k_zero.y, other[t]);
            k_val.z = decode((words[2] >> (4 * t)) & 15u, k_scale.z, k_zero.z, other[t]);
            k_val.w = decode((words[3] >> (4 * t)) & 15u, k_scale.w, k_zero.w, other[t]);
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                score[g * RKVA_TILE + t] = dot(q[g], k_val);
            }
        }

        // Softmax rescale and weights
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            float tile_max = -3.402823466e38f;
            [unroll] for (uint t = 0; t < RKVA_TILE; ++t) {
                score[g * RKVA_TILE + t] = WaveActiveSum(score[g * RKVA_TILE + t]) * kq_scale;
                tile_max = max(tile_max, score[g * RKVA_TILE + t]);
            }
            float next_max = max(maximum[g], tile_max);
            float old_weight = denom[g] == 0 ? 0 : exp(maximum[g] - next_max);
            float sum = 0;
            [unroll] for (uint t = 0; t < RKVA_TILE; ++t) {
                score[g * RKVA_TILE + t] = exp(score[g * RKVA_TILE + t] - next_max);
                sum += score[g * RKVA_TILE + t];
            }
            body[g] *= old_weight;
            denom[g] = denom[g] * old_weight + sum;
            maximum[g] = next_max;
        }

        uint4 scale_bits = values.Load4(base_v + 8192 + pos0 * 2);
        uint4 zero_bits = values.Load4(base_v + 8448 + pos0 * 2);

        [unroll] for (uint t = 0; t < RKVA_TILE; ++t) {
            uint sh = (t & 1) * 16;
            float4 v_val = tile_value(base_v, pos0 + t, u,
                                      f16tof32((scale_bits[t / 2] >> sh) & 65535u),
                                      f16tof32((zero_bits[t / 2] >> sh) & 65535u), v_other);
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                body[g] += score[g * RKVA_TILE + t] * v_val;
            }
        }
    }

    // Phase 2: Residual body rows (stage rows and unaligned split edges).
    // Tail is still 0 during this phase; no branch on exactness.
    uint body_end_0 = min(fast_lo, tail_begin);
    uint body_end_1 = min(end, tail_begin);
    [loop] for (uint segment = 0; segment < 2; ++segment) {
        uint seg_begin = segment == 0 ? begin : fast_hi;
        uint seg_end = segment == 0 ? body_end_0 : body_end_1;
        if (seg_begin < seg_end) {
            [loop] for (uint row = seg_begin + wave; row < seg_end; row += RKVA_DECODE_WAVES) {
                float4 k_val = load_body4(keys, false, row, kv_head, u, live_group);
                float4 x_val = load_body4(values, true, row, kv_head, u, live_group);
                [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                    float s = WaveActiveSum(dot(q[g], k_val)) * kq_scale;
                    float next_max = max(maximum[g], s);
                    float old_weight = denom[g] == 0 ? 0 : exp(maximum[g] - next_max);
                    float w = exp(s - next_max);
                    body[g] = body[g] * old_weight + w * x_val;
                    denom[g] = denom[g] * old_weight + w;
                    maximum[g] = next_max;
                }
            }
        }
    }

    // Phase 3: Exact-tail rows [max(begin, tail_begin), end).
    // Original-domain Q is loaded ONCE per wave instead of per row.
    uint tail_seg_begin = max(begin, tail_begin);
    if (tail_seg_begin < end) {
        float4 qo[RKVA_GQA];
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            uint qbase = (head0 + g) * RKVA_DIM;
            qo[g] = asfloat(queries.Load4((qbase + u * 4) * 4));
        }
        [loop] for (uint row = tail_seg_begin + wave; row < end; row += RKVA_DECODE_WAVES) {
            float4 k_val = load_tail4(keys, row, kv_head, u);
            float4 x_val = load_tail4(values, row, kv_head, u);
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                float s = WaveActiveSum(dot(qo[g], k_val)) * kq_scale;
                float next_max = max(maximum[g], s);
                float old_weight = denom[g] == 0 ? 0 : exp(maximum[g] - next_max);
                float w = exp(s - next_max);
                body[g] *= old_weight;
                tail[g] = tail[g] * old_weight + w * x_val;
                denom[g] = denom[g] * old_weight + w;
                maximum[g] = next_max;
            }
        }
    }

    // Phase 4: Cross-wave merge (2 waves)
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        uint slot = wave * RKVA_GQA + g;
        // Reject the whole wave state before a zero merge weight can hide it.
        bool invalid_state = !isfinite(denom[g]) || denom[g] < 0 ||
            (denom[g] != 0 && (!isfinite(maximum[g]) || !(all(isfinite(body[g])) && all(isfinite(tail[g])))));
        bool invalid_wave = WaveActiveAnyTrue(invalid_state);
        if (lane == 0) {
            gs_max[slot] = invalid_wave ? asfloat(0x7fc00000u) : maximum[g];
            gs_den[slot] = invalid_wave ? asfloat(0x7fc00000u) : denom[g];
        }
        gs_vec[slot * RKVA_DIM + u * 4 + 0] = body[g].x;
        gs_vec[slot * RKVA_DIM + u * 4 + 1] = body[g].y;
        gs_vec[slot * RKVA_DIM + u * 4 + 2] = body[g].z;
        gs_vec[slot * RKVA_DIM + u * 4 + 3] = body[g].w;
    }
    GroupMemoryBarrierWithGroupSync();

    float merged_max[RKVA_GQA], merged_den[RKVA_GQA], weight[RKVA_DECODE_WAVES * RKVA_GQA];
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        merged_max[g] = -3.402823466e38f;
        for (uint w = 0; w < RKVA_DECODE_WAVES; ++w) {
            if (gs_den[w * RKVA_GQA + g] != 0) merged_max[g] = max(merged_max[g], gs_max[w * RKVA_GQA + g]);
        }
        merged_den[g] = 0;
        [unroll] for (uint w = 0; w < RKVA_DECODE_WAVES; ++w) {
            uint slot = w * RKVA_GQA + g;
            weight[slot] = 0;
            if (gs_den[slot] != 0) {
                weight[slot] = exp(gs_max[slot] - merged_max[g]);
                merged_den[g] += gs_den[slot] * weight[slot];
            }
        }
    }

    float merged_body[RKVA_ELEMS], merged_tail[RKVA_ELEMS];
    [unroll] for (uint k = 0; k < RKVA_ELEMS; ++k) {
        uint g = k / 2, d = tid + (k & 1) * 128;
        merged_body[k] = 0;
        merged_tail[k] = 0;
        [unroll] for (uint w = 0; w < RKVA_DECODE_WAVES; ++w) {
            uint slot = w * RKVA_GQA + g;
            if (weight[slot] != 0) merged_body[k] += gs_vec[slot * RKVA_DIM + d] * weight[slot];
        }
    }

    bool has_tail = end > max(begin, tail_begin);
    if (has_tail) {
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            uint slot = wave * RKVA_GQA + g;
            gs_vec[slot * RKVA_DIM + u * 4 + 0] = tail[g].x;
            gs_vec[slot * RKVA_DIM + u * 4 + 1] = tail[g].y;
            gs_vec[slot * RKVA_DIM + u * 4 + 2] = tail[g].z;
            gs_vec[slot * RKVA_DIM + u * 4 + 3] = tail[g].w;
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint k = 0; k < RKVA_ELEMS; ++k) {
            uint g = k / 2, d = tid + (k & 1) * 128;
            [unroll] for (uint w = 0; w < RKVA_DECODE_WAVES; ++w) {
                uint slot = w * RKVA_GQA + g;
                if (weight[slot] != 0) merged_tail[k] += gs_vec[slot * RKVA_DIM + d] * weight[slot];
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (direct_output != 0) {
        bool has_body = begin < min(end, tail_begin);
        if (has_body) {
            wht256x6_decode(merged_body, tid);
        }
        uint total = n_query * n_head;
        [unroll] for (uint k = 0; k < RKVA_ELEMS; ++k) {
            uint g = k / 2, d = tid + (k & 1) * 128;
            uint qh = head0 + g;
            float numerator = merged_body[k] + merged_tail[k];
            float inv = merged_den[g] == 0 ? 0 : 1 / merged_den[g];
            output.Store((qh * RKVA_DIM + d) * 4, asuint(numerator * inv));
            if (d == 0) {
                float lse = merged_den[g] != 0 ? merged_max[g] + log(merged_den[g])
                                              : asfloat(0xff800000);
                output.Store((total * RKVA_DIM + qh) * 4, asuint(lse));
            }
        }
    } else {
        if (has_tail) {
            wht256x6_decode(merged_tail, tid);
        }
        uint total = n_query * n_head;
        [unroll] for (uint k = 0; k < RKVA_ELEMS; ++k) {
            uint g = k / 2, d = tid + (k & 1) * 128;
            uint qh = head0 + g;
            float numerator = has_tail ? (merged_body[k] + merged_tail[k]) : merged_body[k];
            uint dst = ((split * total + qh) * (RKVA_DIM + 2)) * 4;
            partials.Store(dst + d * 4, asuint(numerator));
            if (d == 0) {
                partials.Store(dst + RKVA_DIM * 4, asuint(merged_max[g]));
                partials.Store(dst + (RKVA_DIM + 1) * 4, asuint(merged_den[g]));
            }
        }
    }
}
#endif
