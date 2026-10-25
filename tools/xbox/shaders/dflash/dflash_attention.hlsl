// DFlash2 DRAFT attention only: RoPE and Q/K norms have already been applied.
// Dispatch (128, 1, 1), grid (32, query_count, 1). One group owns one
// query/head; lane d computes output dimension d. All current-batch K/V rows
// must have been packed before this dispatch. This is standard noncausal SWA:
// future positions are visible, while past positions >= 2048 back are not.
// The active_positions SRV is the authoritative list of rows for which BOTH
// K and V are valid after any trim/reinjection. The caller must regenerate it
// after each logical KV mutation; physical stale bytes are never evidence of
// validity. Entries must be absolute positions within kv_capacity.
//
// DflashGpuKv row: [8*128/32 key blocks][8*128/32 value blocks]; every block
// is 20 bytes: fp16 d in low 16 bits, padding, then 16 packed Q4_0 bytes.
// Unlike a causal kernel, no upper bound by query_position is applied.
cbuffer Params : register(b0) {
    uint query_count;
    uint active_count;
    uint kv_capacity;
    uint kv_layer_byte_offset;
};

ByteAddressBuffer queries          : register(t0); // F32 [query][32][128]
ByteAddressBuffer kv               : register(t1); // DflashGpuKv raw buffer
ByteAddressBuffer query_positions  : register(t2); // U32 [query]
ByteAddressBuffer active_positions : register(t3); // U32 [active_count]
ByteAddressBuffer sinks            : register(t4); // F32 [32] then UCRT exp table [64] F64
RWByteAddressBuffer output         : register(u0); // F32 [query][32][128]

#include "ggml_float.hlsli"

// UCRT x64 expf normal path from the host ucrtbase.dll, valid for x in [-104,0].
// The 64 double table values follow the 32 F32 sink values in t4.
float ggml_exp_neg(float x) {
    if (x < -104.0f) return 0.0f;
    precise double y = (double)x * asdouble(0x652b82feu,0x40571547u);
    int n = int((float)y);
    const double fraction = y - (double)(float)n;
    if (fraction < -0.5 || (fraction == -0.5 && (n & 1))) --n;
    const int j = n & 63;
    const int k = (n - j) >> 6;
    precise double r = (-(double)(float)n * asdouble(0xfefa39efu,0x3f862e42u) + (double)x);
    precise double p = (r * asdouble(0x55555555u,0x3fc55555u) + 0.5);
    precise double z = ((r*r) * p + r);
    const double t = asdouble(sinks.Load(128u + uint(j)*8u), sinks.Load(128u + uint(j)*8u+4u));
    precise double v = (z*t+t);
    precise double out_value = v * asdouble(0u,uint(k+1023)<<20u);
    return ggml_round_f32(out_value);
}


// GGML CPU Flash Attention quantizes each 32-element Q block to Q8_0 before
// calling ggml_vec_dot_q4_0_q8_0 for a Q4_0 KV row. Keep the integer dot and
// both FP16 scales: F32 Q times dequantized K changes the softmax scores.
groupshared float q_amax[128];
groupshared int dot_parts[128];
groupshared float q_scales[4];
groupshared float score_parts[8];

// Match the round-to-nearest-even FP16 scale stored by GGML Q8_0. DXC's
// f32tof16 can truncate on the console; this is the same conversion used by
// the proven Q4_0 KV pack shader.
uint ggml_half_bits(float value) {
    const uint bits = asuint(value);
    const uint sign = (bits >> 16) & 0x8000u;
    const uint exponent = (bits >> 23) & 255u;
    uint mantissa = bits & 0x7fffffu;
    if (exponent == 255u)
        return sign | (mantissa == 0 ? 0x7c00u : 0x7e00u);
    int half_exponent = int(exponent) - 127 + 15;
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        mantissa |= 0x800000u;
        const uint shift = uint(14 - half_exponent);
        uint rounded = mantissa >> shift;
        const uint remainder = mantissa & ((1u << shift) - 1u);
        const uint halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u)))
            ++rounded;
        return sign | rounded;
    }
    if (half_exponent >= 31) return sign | 0x7c00u;
    uint rounded = mantissa >> 13;
    const uint remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u)))
        ++rounded;
    if (rounded == 0x400u) {
        rounded = 0;
        ++half_exponent;
    }
    if (half_exponent >= 31) return sign | 0x7c00u;
    return sign | (uint(half_exponent) << 10) | rounded;
}

int nearest_even(float value) {
    const float lower = floor(value);
    const float fraction = value - lower;
    int result = int(lower);
    if (fraction > 0.5f || (fraction == 0.5f && (result & 1) != 0)) ++result;
    return result;
}

uint kv_block_offset(uint row_base, uint head, uint d, uint value_kind) {
    return row_base + value_kind * 640u + (head * 4u + d / 32u) * 20u;
}

int kv_q4(uint row_base, uint head, uint d, uint value_kind) {
    const uint byte_offset = kv_block_offset(row_base, head, d, value_kind);
    const uint within = d & 31u;
    const uint packed = kv.Load(byte_offset + 4u + ((within & 15u) >> 2u) * 4u);
    const uint byte_value = (packed >> (((within & 3u) * 8u))) & 255u;
    const uint nibble = (byte_value >> ((within >> 4u) * 4u)) & 15u;
    return int(nibble) - 8;
}

float kv_element(uint row_base, uint head, uint d, uint value_kind) {
    const uint byte_offset = kv_block_offset(row_base, head, d, value_kind);
    const float scale = f16tof32(kv.Load(byte_offset) & 0xffffu);
    return float(kv_q4(row_base, head, d, value_kind)) * scale;
}

[numthreads(128, 1, 1)]
void CSMain(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
    const uint head = group_id.x;
    const uint qi = group_id.y;
    if (head >= 32u || qi >= query_count) return;

    const uint query_position = query_positions.Load(qi * 4u);
    const float q = asfloat(queries.Load(((qi * 32u + head) * 128u + lane) * 4u));
    const uint kv_head = head / 4u; // 32 Q heads / 8 KV heads
    q_amax[lane] = abs(q);
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint step = 16u; step > 0u; step >>= 1u) {
        if ((lane & 31u) < step) q_amax[lane] = max(q_amax[lane], q_amax[lane + step]);
        GroupMemoryBarrierWithGroupSync();
    }
    const float q_max = q_amax[(lane / 32u) * 32u];
    const float q_scale = f16tof32(ggml_half_bits(ggml_round_f32((double)q_max / 127.0)));
    // GGML uses the unrounded max to quantize, not the stored FP16 scale.
    const float inv_q_scale = q_max != 0.0f ? ggml_round_f32(127.0 / (double)q_max) : 0.0f;
    precise float scaled_q = q * inv_q_scale;
    dot_parts[lane] = clamp(nearest_even(scaled_q), -127, 127);
    if ((lane & 31u) == 0u) q_scales[lane / 32u] = q_scale;
    GroupMemoryBarrierWithGroupSync();
    float max_score = asfloat(0xff800000u);
    precise float denominator = 0.0f;
    precise float numerator = 0.0f;

    [loop] for (uint j = 0; j < active_count; ++j) {
        const uint key_position = active_positions.Load(j * 4u);
        if (key_position >= kv_capacity ||
            (key_position <= query_position && query_position - key_position >= 2048u)) {
            continue;
        }

        const uint row_base = kv_layer_byte_offset + key_position * 1280u;
        if (lane < 8u) {
            float acc = 0.0f;
            [unroll] for (uint block = 0; block < 4u; ++block) {
                int dot = 0;
                [unroll] for (uint part = 0; part < 4u; ++part) {
                    const uint d = block * 32u + lane * 4u + part;
                    dot += dot_parts[d] * kv_q4(row_base, kv_head, d, 0u);
                }
                const uint key_block = kv_block_offset(row_base, kv_head, block * 32u, 0u);
                precise float scale = f16tof32(kv.Load(key_block) & 0xffffu) * q_scales[block];
                acc = ggml_fma_f32(scale, float(dot), acc);
            }
            score_parts[lane] = acc;
        }
        GroupMemoryBarrierWithGroupSync();
        precise float p0 = score_parts[0] + score_parts[4];
        precise float p1 = score_parts[1] + score_parts[5];
        precise float p2 = score_parts[2] + score_parts[6];
        precise float p3 = score_parts[3] + score_parts[7];
        precise float score = ((p0 + p2) + (p1 + p3)) * 0.08838834764831845f;
        GroupMemoryBarrierWithGroupSync();

        float old_weight = 1.0f, new_weight = 1.0f;
        if (score > max_score) {
            old_weight = ggml_exp_neg(max_score - score);
            numerator *= old_weight;
            max_score = score;
        } else {
            new_weight = ggml_exp_neg(score - max_score);
        }
        denominator = denominator * old_weight + new_weight;
        numerator = ggml_fma_f32(kv_element(row_base, kv_head, lane, 1u),
                                 new_weight, numerator);
    }

    // GGML adds the virtual zero-V sink after all physical KV rows.
    const float sink = asfloat(sinks.Load(head * 4u));
    if (sink > max_score) {
        const float weight = ggml_exp_neg(max_score - sink);
        numerator *= weight;
        denominator = denominator * weight + 1.0f;
    } else {
        denominator += ggml_exp_neg(sink - max_score);
    }
    const float inverse = denominator == 0.0f ? 0.0f : ggml_round_f32(1.0 / (double)denominator);
    precise float result = numerator * inverse;
    output.Store(((qi * 32u + head) * 128u + lane) * 4u,
                 asuint(result));
}
