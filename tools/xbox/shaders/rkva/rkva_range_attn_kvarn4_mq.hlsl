// RKVA native Bee KVarN4 serial-prefix multi-query tile (descriptor revision 3, 268).
// One 128-thread group serves RKVA_MQ_QT consecutive queries x RKVA_GQA Q heads of
// one KV head and one split. It reads the same revision-3 buffers as
// rkva_range_attn_kvarn4_opt.hlsl compiled with RKVA_SERIAL_PREFIX=1 (8960-byte
// KVarN4 records, the 384-row rotated F16 stage and the 256-row ORIGINAL-F16
// tail ring) and decodes every K/V element with the same expression
//     precise affine = float(quant) * scale + zero;  value = affine * other
// Nothing is re-quantized, renumbered or materialized.
//
// Serial prefix per query: query j reads the image of its own causal prefix
// W_j = q_count_j = min(active_count, pos_j - origin + 1). Row r < W_j is
//     tail    if r >= W_j - min(W_j, precision_tail_tokens)  (ORIGINAL F16, ring 256)
//     stage   if global group of r is 0 (sink) or r is in W_j's partial live group
//     record  otherwise (a group sealed by W_j)
// so the same row can be a stage row for an earlier query and a record row for a
// later one (or tail for the later one only). Sharing is restricted to what all
// active queries of the tile read identically:
// * shared record tiles [fast_lo, fast_hi): fast_hi = the minimum over active
//   queries of min(end_j, W_j & ~127, tail_begin_j), aligned down to 8 tokens;
//   these rows are sealed records for every active query, never a group sealed
//   after the smallest prefix. Each 8-token tile loads and decodes K/V once for
//   all queries and heads (two 4-token halves, one softmax rescale per half);
// * residual rows [split_begin, fast_lo) and [fast_hi, max end_j) are
//   classified per query; a row's K/V is loaded once per distinct
//   representation (the physical bytes of a representation do not depend on
//   the query) and each query scores it with its own Q: WHT(Q) for stage and
//   record rows, the original Q for tail rows.
// Every query keeps its own online-softmax state, rotated-domain body and
// original-domain tail numerators; output is WHT(body) + tail with the
// per-query maximum/denominator, in the unchanged partial layout
// [split][query * n_head + head][258] (or final O/LSE when split_k == 1).
// Partial rows are written for every real query, including splits it does not
// reach (den 0, max -FLT_MAX), exactly like the per-query kernel.
//
// Supported shape (host-gated): head_dim 256, n_head == RKVA_GQA * n_head_kv,
// origin % 128 == 0, precision_tail_tokens 0 or 128, descriptor revision 3,
// 2..9 queries, RKVA_MQ_QT in {2, 4}, wave size 64..128 (one 4-dim unit per
// lane). A wave size outside 64..128 poisons the outputs with NaN instead of
// computing a partial result. Query positions are used as given (no sorting
// assumption). For RKVA_MQ_QT == 4, rotated Q for queries 2..3 is staged in
// the idle 12 KiB gs_vec buffer during the record/stage loops and tail/merge
// runs in two 2-query pairs after rotated Q is dead, keeping LDS at 12,384 B
// and live VGPRs under the RDNA2 256-VGPR wave64 limit.
#define RKVA_GQA 6
#define RKVA_DIM 256
#define RKVA_THREADS 128
#define RKVA_MAX_WAVES 2 // wave size >= 64
#define RKVA_UNITS 64    // unit u covers dims 4u..4u+3: slice u >> 5, record dims 4(u & 31)..+3
#define RKVA_TILE 8
#define RKVA_HALF 4
#define RKVA_RECORD 8960
#define RKVA_ELEMS 12 // (GQA * DIM) / THREADS output elements per thread
#define RKVA_TAIL_MASK 255u
#ifndef RKVA_MQ_QT
#define RKVA_MQ_QT 2
#endif
#if RKVA_MQ_QT != 2 && RKVA_MQ_QT != 4
#error "native KVarN4 multi-query tile supports RKVA_MQ_QT == 2 or 4 only"
#endif
#define RKVA_QG (RKVA_MQ_QT * RKVA_GQA)
#define RKVA_PAIR_QG (2 * RKVA_GQA)
#define RKVA_KIND_NONE 0u
#define RKVA_KIND_TAIL 1u
#define RKVA_KIND_STAGE 2u
#define RKVA_KIND_RECORD 3u

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

groupshared float gs_max[RKVA_MAX_WAVES * RKVA_GQA];
groupshared float gs_den[RKVA_MAX_WAVES * RKVA_GQA];
// Rotated Q, then per-wave numerators (body, then tail), then the WHT scratch.
groupshared float gs_vec[RKVA_MAX_WAVES * RKVA_GQA * RKVA_DIM];

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

// In-place Bee WHT256 of the RKVA_GQA vectors in gs_vec[g * 256 + d]; identical
// to rkva_range_attn_kvarn4_opt.hlsl (thread tid owns (g, tid) and (g, tid + 128)).
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

// Representation of row for a query with serial window W (see the header). end
// is the query's split end (0 when the query is inactive in this split).
uint row_kind(uint row, uint end, uint window, uint tail_begin) {
    if (row >= end) {
        return RKVA_KIND_NONE;
    }
    if (row >= tail_begin) {
        return RKVA_KIND_TAIL;
    }
    uint group = row / 128;
    uint live_group = window ? (window - 1) / 128 : 0;
    bool staged = origin / 128 + group == 0 || (group == live_group && (window & 127u) != 0);
    return staged ? RKVA_KIND_STAGE : RKVA_KIND_RECORD;
}

// Original-domain tail halves for dims 4u..4u+3 of (row, kv_head), 256-row ring.
float4 load_tail4(ByteAddressBuffer buffer, uint row, uint kv_head, uint u) {
    uint stage_bytes = 384 * n_head_kv * 256 * 2;
    uint element = ((row & RKVA_TAIL_MASK) * n_head_kv + kv_head) * 256 + u * 4;
    return halves4(buffer.Load2(records_bytes + stage_bytes + element * 2));
}

// Rotated-domain stage halves (sink slot 0, ping-pong slots 1..2).
float4 load_stage4(ByteAddressBuffer buffer, uint row, uint kv_head, uint u) {
    uint global_group = origin / 128 + row / 128;
    uint slot = global_group == 0 ? 0 : 1 + ((global_group - 1) & 1);
    uint element = ((slot * 128 + (row & 127)) * n_head_kv + kv_head) * 256 + u * 4;
    return halves4(buffer.Load2(records_bytes + element * 2));
}

// Rotated-domain record dims 4u..4u+3 of (row, kv_head) of a sealed group.
float4 load_record4(ByteAddressBuffer buffer, bool value, uint row, uint kv_head, uint u) {
    uint group = row / 128, pos = row & 127;
    uint dim = (u & 31) * 4;
    uint base = (group * n_head_kv * 2 + kv_head * 2 + (u >> 5)) * RKVA_RECORD;
    float4 result;
    if (value) {
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

void store_result(uint qh, uint d, uint split, float numerator, float merged_max, float merged_den) {
    uint total = n_query * n_head;
    if (direct_output != 0) {
        float inv = merged_den == 0 ? 0 : 1 / merged_den;
        output.Store((qh * RKVA_DIM + d) * 4, asuint(numerator * inv));
        if (d == 0) {
            float lse = merged_den != 0 ? merged_max + log(merged_den) : asfloat(0xff800000);
            output.Store((total * RKVA_DIM + qh) * 4, asuint(lse));
        }
    } else {
        uint dst = ((split * total + qh) * (RKVA_DIM + 2)) * 4;
        partials.Store(dst + d * 4, asuint(numerator));
        if (d == 0) {
            partials.Store(dst + RKVA_DIM * 4, asuint(merged_max));
            partials.Store(dst + (RKVA_DIM + 1) * 4, asuint(merged_den));
        }
    }
}

float4 load_rotated_q(uint j, uint g, uint u, bool active, float4 q_lo[RKVA_PAIR_QG]) {
#if RKVA_MQ_QT == 4
    if (j >= 2) {
        uint b = ((3 - j) * RKVA_GQA + g) * RKVA_DIM + u * 4;
        return active ? float4(gs_vec[b], gs_vec[b + 1], gs_vec[b + 2], gs_vec[b + 3])
                      : float4(0, 0, 0, 0);
    }
#endif
    return q_lo[j * RKVA_GQA + g];
}

[numthreads(RKVA_THREADS, 1, 1)]
void CSRangeAttnKvarn4Mq(uint tid : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint kv_head = group.x, q_first = group.y * RKVA_MQ_QT, split = group.z;
    uint wave_size = WaveGetLaneCount();
    uint lane = WaveGetLaneIndex();

    // Unsupported wave size (host gates the PSO on wave lanes >= 64): NaN poison
    // instead of a partial result. Direct output: O and LSE NaN; partials:
    // numerator NaN with max 0 / den 1, so the reducer propagates it.
    if (wave_size < RKVA_UNITS || wave_size > RKVA_THREADS) {
        uint nan_bits = 0x7fc00000u;
        uint total = n_query * n_head;
        [unroll] for (uint j = 0; j < RKVA_MQ_QT; ++j) {
            uint qi = q_first + j;
            if (qi < n_query) {
                [unroll] for (uint k = 0; k < RKVA_ELEMS; ++k) {
                    uint qh = qi * n_head + kv_head * RKVA_GQA + k / 2;
                    uint d = tid + (k & 1) * 128;
                    if (direct_output != 0) {
                        output.Store((qh * RKVA_DIM + d) * 4, nan_bits);
                        if (d == 0) output.Store((total * RKVA_DIM + qh) * 4, nan_bits);
                    } else {
                        uint dst = ((split * total + qh) * (RKVA_DIM + 2)) * 4;
                        partials.Store(dst + d * 4, nan_bits);
                        if (d == 0) {
                            partials.Store(dst + RKVA_DIM * 4, asuint(0.0f));
                            partials.Store(dst + (RKVA_DIM + 1) * 4, asuint(1.0f));
                        }
                    }
                }
            }
        }
        return;
    }
    uint wave = tid / wave_size;
    uint n_waves = RKVA_THREADS / wave_size;
    bool active = lane < RKVA_UNITS;
    uint u = min(lane, RKVA_UNITS - 1);

    // Per-query serial windows and split ranges (begin_j == split_begin when active).
    uint split_begin = split * split_rows;
    uint split_end = split_begin + split_rows;
    uint record_lo = origin == 0 ? 128 : 0;
    uint window[RKVA_MQ_QT], end_q[RKVA_MQ_QT], body_end_q[RKVA_MQ_QT], tail_begin[RKVA_MQ_QT];
    uint head0[RKVA_MQ_QT];
    bool act[RKVA_MQ_QT];
    uint shared_hi = 0xffffffffu;
    uint body_end_max = split_begin;
    bool any_active = false;
    [unroll] for (uint j = 0; j < RKVA_MQ_QT; ++j) {
        uint qi = q_first + j;
        uint qc = min(qi, n_query - 1); // phantom queries of a partial tile stay in bounds
        int query_pos = asint(positions.Load(qc * 4));
        uint causal_count = query_pos < int(origin) ? 0 : uint(query_pos) - origin + 1;
        uint limit = qi < n_query ? min(active_count, causal_count) : 0;
        window[j] = limit;
        tail_begin[j] = limit - min(limit, precision_tail_tokens);
        uint end = min(limit, split_end);
        act[j] = end > split_begin;
        end_q[j] = act[j] ? end : 0;
        body_end_q[j] = min(end_q[j], tail_begin[j]);
        head0[j] = qc * n_head + kv_head * RKVA_GQA;
        if (act[j]) {
            any_active = true;
            body_end_max = max(body_end_max, body_end_q[j]);
            shared_hi = min(shared_hi, min(min(end, limit & ~127u), tail_begin[j]));
        }
    }
    uint fast_lo = (max(split_begin, record_lo) + RKVA_TILE - 1) & ~(RKVA_TILE - 1);
    uint fast_hi = any_active ? shared_hi & ~(RKVA_TILE - 1) : 0;
    if (fast_hi <= fast_lo) {
        fast_lo = split_begin;
        fast_hi = split_begin;
    }

    // Hoisted / pre-computed Q WHT: Q_rotated is staged at offset q_stride in the
    // queries ByteAddressBuffer, while exact original Q resides at offset 0 for tail.
    // Queries 0..1 stay in q_lo (48 VGPRs); for RKVA_MQ_QT == 4, rotated Q for
    // query 2 is staged in gs_vec[1536..3071] and query 3 stays in gs_vec[0..1535]
    // until the non-tail residual loop finishes (zero extra LDS).
    uint q_stride = n_query * n_head * RKVA_DIM * 4;
    float4 q_lo[RKVA_PAIR_QG];
    [unroll] for (uint j = 0; j < min(2u, uint(RKVA_MQ_QT)); ++j) {
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            uint q_rot_base = q_stride + ((head0[j] + g) * RKVA_DIM + u * 4) * 4;
            q_lo[j * RKVA_GQA + g] =
                active ? asfloat(queries.Load4(q_rot_base)) : float4(0, 0, 0, 0);
        }
    }
#if RKVA_MQ_QT == 4
    if (wave == 0 && active) {
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            uint q_rot_base2 = q_stride + ((head0[2] + g) * RKVA_DIM + u * 4) * 4;
            float4 q2 = asfloat(queries.Load4(q_rot_base2));
            uint b2 = (RKVA_GQA + g) * RKVA_DIM + u * 4;
            gs_vec[b2 + 0] = q2.x;
            gs_vec[b2 + 1] = q2.y;
            gs_vec[b2 + 2] = q2.z;
            gs_vec[b2 + 3] = q2.w;

            uint q_rot_base3 = q_stride + ((head0[3] + g) * RKVA_DIM + u * 4) * 4;
            float4 q3 = asfloat(queries.Load4(q_rot_base3));
            uint b3 = g * RKVA_DIM + u * 4;
            gs_vec[b3 + 0] = q3.x;
            gs_vec[b3 + 1] = q3.y;
            gs_vec[b3 + 2] = q3.z;
            gs_vec[b3 + 3] = q3.w;
        }
    }
    GroupMemoryBarrierWithGroupSync();
#endif
    float4 body[RKVA_QG];
    float maximum[RKVA_QG], denom[RKVA_QG];
    [unroll] for (uint i = 0; i < RKVA_QG; ++i) {
        body[i] = float4(0, 0, 0, 0);
        maximum[i] = -3.402823466e38f;
        denom[i] = 0;
    }

    // ---- Shared record tiles (first, so the still-zero tail needs no rescale) ----
    uint dim = (u & 31) * 4;
    uint cached_group = 0xffffffffu;
    uint base = 0;
    float4 kscale = float4(0, 0, 0, 0), kzero = kscale, vother = kscale;
    [loop] for (uint t0 = fast_lo + wave * RKVA_TILE; t0 < fast_hi; t0 += n_waves * RKVA_TILE) {
        uint rgroup = t0 / 128, pos0 = t0 & 127;
        if (rgroup != cached_group) {
            base = (rgroup * n_head_kv * 2 + kv_head * 2 + (u >> 5)) * RKVA_RECORD;
            kscale = halves4(keys.Load2(base + 8192 + dim * 2));
            kzero = halves4(keys.Load2(base + 8448 + dim * 2));
            vother = halves4(values.Load2(base + 8704 + dim * 2));
            cached_group = rgroup;
        }
        uint4 kother_bits = keys.Load4(base + 8704 + pos0 * 2);
        uint nib_base = base + (pos0 / 8) * 4;
        uint kw[4] = {keys.Load(nib_base + (dim + 0) * 64), keys.Load(nib_base + (dim + 1) * 64),
                      keys.Load(nib_base + (dim + 2) * 64), keys.Load(nib_base + (dim + 3) * 64)};
        uint4 vscale_bits = values.Load4(base + 8192 + pos0 * 2);
        uint4 vzero_bits = values.Load4(base + 8448 + pos0 * 2);
        [unroll] for (uint h = 0; h < RKVA_TILE / RKVA_HALF; ++h) {
            float4 k_half[RKVA_HALF], v_half[RKVA_HALF];
            [unroll] for (uint t = 0; t < RKVA_HALF; ++t) {
                uint tt = h * RKVA_HALF + t;
                uint sh = (tt & 1) * 16;
                float other = f16tof32((kother_bits[tt / 2] >> sh) & 65535u);
                k_half[t].x = decode((kw[0] >> (4 * tt)) & 15u, kscale.x, kzero.x, other);
                k_half[t].y = decode((kw[1] >> (4 * tt)) & 15u, kscale.y, kzero.y, other);
                k_half[t].z = decode((kw[2] >> (4 * tt)) & 15u, kscale.z, kzero.z, other);
                k_half[t].w = decode((kw[3] >> (4 * tt)) & 15u, kscale.w, kzero.w, other);
                float scale = f16tof32((vscale_bits[tt / 2] >> sh) & 65535u);
                float zero = f16tof32((vzero_bits[tt / 2] >> sh) & 65535u);
                uint index = (pos0 + tt) * 128 + dim;
                uint word = values.Load(base + (index / 8) * 4) >> ((index & 7) * 4);
                v_half[t].x = decode(word & 15u, scale, zero, vother.x);
                v_half[t].y = decode((word >> 4) & 15u, scale, zero, vother.y);
                v_half[t].z = decode((word >> 8) & 15u, scale, zero, vother.z);
                v_half[t].w = decode((word >> 12) & 15u, scale, zero, vother.w);
            }
            [unroll] for (uint j = 0; j < RKVA_MQ_QT; ++j) {
                [branch] if (act[j]) {
                    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                        uint i = j * RKVA_GQA + g;
                        float4 qa = load_rotated_q(j, g, u, active, q_lo);
                        float s[RKVA_HALF];
                        float half_max = -3.402823466e38f;
                        [unroll] for (uint t = 0; t < RKVA_HALF; ++t) {
                            s[t] = WaveActiveSum(dot(qa, k_half[t])) * kq_scale;
                            half_max = max(half_max, s[t]);
                        }
                        float next_max = max(maximum[i], half_max);
                        float old_weight = denom[i] == 0 ? 0 : exp(maximum[i] - next_max);
                        float sum = 0;
                        [unroll] for (uint t = 0; t < RKVA_HALF; ++t) {
                            s[t] = exp(s[t] - next_max);
                            sum += s[t];
                        }
                        body[i] *= old_weight;
                        [unroll] for (uint t = 0; t < RKVA_HALF; ++t) {
                            body[i] += s[t] * v_half[t];
                        }
                        denom[i] = denom[i] * old_weight + sum;
                        maximum[i] = next_max;
                    }
                }
            }
        }
    }

    // ---- Non-tail residual rows (stage and sealed record), classified per query ----
    [loop] for (uint segment = 0; segment < 2; ++segment) {
        uint seg_begin = segment == 0 ? split_begin : fast_hi;
        uint seg_end = segment == 0 ? fast_lo : body_end_max;
        [loop] for (uint row = seg_begin + wave; row < seg_end; row += n_waves) {
            float4 k = float4(0, 0, 0, 0), x = k;
            uint loaded = RKVA_KIND_NONE; // wave-uniform
            [unroll] for (uint j = 0; j < RKVA_MQ_QT; ++j) {
                uint kind = row_kind(row, body_end_q[j], window[j], tail_begin[j]);
                [branch] if (kind != RKVA_KIND_NONE) {
                    if (kind != loaded) {
                        if (kind == RKVA_KIND_STAGE) {
                            k = load_stage4(keys, row, kv_head, u);
                            x = load_stage4(values, row, kv_head, u);
                        } else {
                            k = load_record4(keys, false, row, kv_head, u);
                            x = load_record4(values, true, row, kv_head, u);
                        }
                        loaded = kind;
                    }
                    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                        uint i = j * RKVA_GQA + g;
                        float4 qa = load_rotated_q(j, g, u, active, q_lo);
                        float s = WaveActiveSum(dot(qa, k)) * kq_scale;
                        float next_max = max(maximum[i], s);
                        float old_weight = denom[i] == 0 ? 0 : exp(maximum[i] - next_max);
                        float w = exp(s - next_max);
                        body[i] = body[i] * old_weight + w * x;
                        denom[i] = denom[i] * old_weight + w;
                        maximum[i] = next_max;
                    }
                }
            }
        }
    }

#if RKVA_MQ_QT == 4
    // Rotated Q in gs_vec is no longer read after the non-tail residual loop.
    GroupMemoryBarrierWithGroupSync();
#endif

    // ---- Paired original-F16 tail residual rows + cross-wave merge, WHT(body) + tail, store ----
    [unroll] for (uint pair = 0; pair < RKVA_MQ_QT / 2; ++pair) {
        float4 tail_pair[RKVA_PAIR_QG];
        [unroll] for (uint t = 0; t < RKVA_PAIR_QG; ++t) {
            tail_pair[t] = float4(0, 0, 0, 0);
        }
        uint j0 = pair * 2;
        uint t_lo0 = max(fast_hi, tail_begin[j0]);
        uint t_lo1 = max(fast_hi, tail_begin[j0 + 1]);
        bool has_tail0 = end_q[j0] > t_lo0;
        bool has_tail1 = end_q[j0 + 1] > t_lo1;
        [branch] if (has_tail0 || has_tail1) {
            uint pair_lo = has_tail0 ? (has_tail1 ? min(t_lo0, t_lo1) : t_lo0) : t_lo1;
            uint pair_hi = max(has_tail0 ? end_q[j0] : 0, has_tail1 ? end_q[j0 + 1] : 0);
            uint row0 = fast_hi + ((pair_lo - fast_hi) & ~(n_waves - 1)) + wave;
            [loop] for (uint row = row0; row < pair_hi; row += n_waves) {
                float4 k = float4(0, 0, 0, 0), x = k;
                bool loaded = false;
                [unroll] for (uint p = 0; p < 2; ++p) {
                    uint j = j0 + p;
                    [branch] if (row >= tail_begin[j] && row < end_q[j]) {
                        if (!loaded) {
                            k = load_tail4(keys, row, kv_head, u);
                            x = load_tail4(values, row, kv_head, u);
                            loaded = true;
                        }
                        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                            uint i = j * RKVA_GQA + g;
                            uint ti = p * RKVA_GQA + g;
                            float4 qa =
                                active ? asfloat(queries.Load4(((head0[j] + g) * RKVA_DIM + u * 4) * 4))
                                       : float4(0, 0, 0, 0);
                            float s = WaveActiveSum(dot(qa, k)) * kq_scale;
                            float next_max = max(maximum[i], s);
                            float old_weight = denom[i] == 0 ? 0 : exp(maximum[i] - next_max);
                            float w = exp(s - next_max);
                            body[i] *= old_weight;
                            tail_pair[ti] = tail_pair[ti] * old_weight + w * x;
                            denom[i] = denom[i] * old_weight + w;
                            maximum[i] = next_max;
                        }
                    }
                }
            }
        }

        [unroll] for (uint p = 0; p < 2; ++p) {
            uint j = j0 + p;
            uint qi = q_first + j;
            [branch] if (qi < n_query) { // group-uniform
                [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                    uint i = j * RKVA_GQA + g;
                    uint slot = wave * RKVA_GQA + g;
                    // Reject the whole wave state before a zero merge weight can hide it.
                    bool invalid_state = !isfinite(denom[i]) || denom[i] < 0 ||
                        (denom[i] != 0 && (!isfinite(maximum[i]) || !(all(isfinite(body[i])) && all(isfinite(tail_pair[p * RKVA_GQA + g])))));
                    bool invalid_wave = WaveActiveAnyTrue(invalid_state);
                    if (lane == 0) {
                        gs_max[slot] = invalid_wave ? asfloat(0x7fc00000u) : maximum[i];
                        gs_den[slot] = invalid_wave ? asfloat(0x7fc00000u) : denom[i];
                    }
                    if (active) {
                        gs_vec[slot * RKVA_DIM + u * 4 + 0] = body[i].x;
                        gs_vec[slot * RKVA_DIM + u * 4 + 1] = body[i].y;
                        gs_vec[slot * RKVA_DIM + u * 4 + 2] = body[i].z;
                        gs_vec[slot * RKVA_DIM + u * 4 + 3] = body[i].w;
                    }
                }
                GroupMemoryBarrierWithGroupSync();
                float merged_max[RKVA_GQA], merged_den[RKVA_GQA], weight[RKVA_MAX_WAVES * RKVA_GQA];
                [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                    merged_max[g] = -3.402823466e38f;
                    for (uint w = 0; w < n_waves; ++w) {
                        if (gs_den[w * RKVA_GQA + g] != 0)
                            merged_max[g] = max(merged_max[g], gs_max[w * RKVA_GQA + g]);
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
                float merged_body[RKVA_ELEMS], merged_tail[RKVA_ELEMS];
                [unroll] for (uint k = 0; k < RKVA_ELEMS; ++k) {
                    uint g = k / 2, d = tid + (k & 1) * 128;
                    merged_body[k] = 0;
                    merged_tail[k] = 0;
                    [unroll] for (uint w = 0; w < RKVA_MAX_WAVES; ++w) {
                        uint slot = w * RKVA_GQA + g;
                        if (weight[slot] != 0)
                            merged_body[k] += gs_vec[slot * RKVA_DIM + d] * weight[slot];
                    }
                }
                bool has_tail = end_q[j] > max(split_begin, tail_begin[j]); // group-uniform
                if (has_tail) {
                    GroupMemoryBarrierWithGroupSync();
                    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                        uint ti = p * RKVA_GQA + g;
                        uint slot = wave * RKVA_GQA + g;
                        if (active) {
                            gs_vec[slot * RKVA_DIM + u * 4 + 0] = tail_pair[ti].x;
                            gs_vec[slot * RKVA_DIM + u * 4 + 1] = tail_pair[ti].y;
                            gs_vec[slot * RKVA_DIM + u * 4 + 2] = tail_pair[ti].z;
                            gs_vec[slot * RKVA_DIM + u * 4 + 3] = tail_pair[ti].w;
                        }
                    }
                    GroupMemoryBarrierWithGroupSync();
                    [unroll] for (uint k = 0; k < RKVA_ELEMS; ++k) {
                        uint g = k / 2, d = tid + (k & 1) * 128;
                        [unroll] for (uint w = 0; w < RKVA_MAX_WAVES; ++w) {
                            uint slot = w * RKVA_GQA + g;
                            if (weight[slot] != 0)
                                merged_tail[k] += gs_vec[slot * RKVA_DIM + d] * weight[slot];
                        }
                    }
                }
                if (direct_output != 0) {
                    GroupMemoryBarrierWithGroupSync();
                    wht256x6(merged_body, tid); // linear: WHT(sum of wave bodies) == sum of WHT(wave bodies)
                    [unroll] for (uint k = 0; k < RKVA_ELEMS; ++k) {
                        uint g = k / 2, d = tid + (k & 1) * 128;
                        store_result(qi * n_head + kv_head * RKVA_GQA + g, d, split,
                                     merged_body[k] + merged_tail[k], merged_max[g], merged_den[g]);
                    }
                } else {
                    if (has_tail) {
                        GroupMemoryBarrierWithGroupSync();
                        wht256x6(merged_tail, tid);
                        [unroll] for (uint k = 0; k < RKVA_ELEMS; ++k) {
                            uint g = k / 2, d = tid + (k & 1) * 128;
                            store_result(qi * n_head + kv_head * RKVA_GQA + g, d, split,
                                         merged_body[k] + merged_tail[k], merged_max[g], merged_den[g]);
                        }
                    } else {
                        // Retire LDS reads before the next query reuses gs_* storage.
                        GroupMemoryBarrierWithGroupSync();
                        [unroll] for (uint k = 0; k < RKVA_ELEMS; ++k) {
                            uint g = k / 2, d = tid + (k & 1) * 128;
                            store_result(qi * n_head + kv_head * RKVA_GQA + g, d, split,
                                         merged_body[k], merged_max[g], merged_den[g]);
                        }
                    }
                }
            }
        }
    }
}
