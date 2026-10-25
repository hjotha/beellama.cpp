// RKVA v2 native bulk F16/FP8 Q/OUT GPU conversion shaders (Xbox Series X|S).
//
// Shares the 4-constant + 1-SRV (t0) + 1-UAV (u0) root signature layout of
// range_reduce_root_ so no additional root signature object is required:
//   b0 = { q_pairs, aux_words, reserved0, reserved1 }
// Q/OUT elements = 2*q_pairs, always a multiple of256 (D=256); all compact
// buffers and LSE/status offsets are4-byte aligned, even for odd head counts.
//   t0 = source ByteAddressBuffer
//   u0 = destination RWByteAddressBuffer
//
// CSBulkUnpackQF16 (after upload, before attention dispatch):
//   src (UPLOAD):  Q_f16 [q_pairs * 2] packed 2xF16 words, then positions [n_queries] i32
//   dst (DEFAULT): Q_raw [q_pairs * 2], Q_rot [q_pairs * 2] F32 words,
//                  then positions [n_queries] i32. Q_rot uses orthonormal R=H256/16.
//
// CSBulkPackOutF16 (after attention/reduce, before readback DMA copy):
//   src (DEFAULT): OUT_f32 [q_pairs * 2] F32 words, then LSE_f32 [q_heads] F32 words
//   dst (DEFAULT): OUT_f16 [q_pairs * 2] packed 2xF16 words, then LSE_f32 [q_heads] F32 words
// FP8 siblings use one byte per element. Pack appends one status word/group
// after LSE internally; serving strips status after validation. OUT conversion
// remains F32->F16->FP8 (two RNE steps), never direct F32->FP8.

cbuffer Params : register(b0) {
    uint q_pairs;
    uint aux_words;
    uint reserved0;
    uint reserved1;
};

ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

// GGML F32->F16 round-to-nearest-even (bit-exact with ggml_fp32_to_fp16).
uint ggml_half_bits(float value) {
    const uint bits = asuint(value);
    const uint sign = (bits >> 16) & 0x8000u;
    const uint exponent = (bits >> 23) & 255u;
    uint mantissa = bits & 0x7fffffu;
    if (exponent == 255u) {
        return sign | (mantissa == 0 ? 0x7c00u : 0x7e00u);
    }
    int half_exponent = int(exponent) - 127 + 15;
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return sign;
        }
        mantissa |= 0x800000u;
        const uint shift = uint(14 - half_exponent);
        uint rounded = mantissa >> shift;
        const uint remainder = mantissa & ((1u << shift) - 1u);
        const uint halfway = uint(1u << (shift - 1u));
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) {
            ++rounded;
        }
        return sign | rounded;
    }
    if (half_exponent >= 31) {
        return sign | 0x7c00u;
    }
    uint rounded = mantissa >> 13;
    const uint remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        ++rounded;
    }
    if (rounded == 0x400u) {
        rounded = 0;
        ++half_exponent;
    }
    if (half_exponent >= 31) {
        return sign | 0x7c00u;
    }
    return sign | (uint(half_exponent) << 10) | rounded;
}

// Two complete 256-element heads per group. All lanes participate in the
// butterflies, including the padded second head when the head count is odd.
groupshared float gs_query[512];

void unpack_query(uint lane, uint3 group, bool fp8) {
    const uint idx = group.x * 256 + lane;
    float lo = 0, hi = 0;
    if (idx < q_pairs) {
        uint h0, h1;
        if (fp8) {
            const uint packed = src.Load((idx / 2) * 4) >> ((idx & 1u) * 16);
            h0 = (packed & 0xffu) << 8;
            h1 = ((packed >> 8) & 0xffu) << 8;
        } else {
            const uint packed = src.Load(idx * 4);
            h0 = packed & 0xffffu;
            h1 = packed >> 16;
        }
        lo = f16tof32(h0);
        hi = f16tof32(h1);
        dst.Store((idx * 2) * 4, asuint(lo));
        dst.Store((idx * 2 + 1) * 4, asuint(hi));
    }
    gs_query[lane * 2] = lo;
    gs_query[lane * 2 + 1] = hi;
    GroupMemoryBarrierWithGroupSync();

    // Match butterfly_wht256/wht256x6: H128 / sqrt(128), then H2 / sqrt(2).
    [unroll] for (uint stride_wht = 1; stride_wht < 128; stride_wht *= 2) {
        uint d0 = lane * 2, d1 = d0 + 1;
        float own0 = gs_query[d0], other0 = gs_query[d0 ^ stride_wht];
        float own1 = gs_query[d1], other1 = gs_query[d1 ^ stride_wht];
        GroupMemoryBarrierWithGroupSync();
        gs_query[d0] = (d0 & stride_wht) ? other0 - own0 : own0 + other0;
        gs_query[d1] = (d1 & stride_wht) ? other1 - own1 : own1 + other1;
        GroupMemoryBarrierWithGroupSync();
    }
    uint d0 = lane * 2, d1 = d0 + 1;
    float x0 = gs_query[d0] * 0.08838834764831845f;
    float y0 = gs_query[d0 ^ 128] * 0.08838834764831845f;
    float x1 = gs_query[d1] * 0.08838834764831845f;
    float y1 = gs_query[d1 ^ 128] * 0.08838834764831845f;
    float rot0 = ((d0 & 128) ? y0 - x0 : x0 + y0) * 0.7071067811865475f;
    float rot1 = ((d1 & 128) ? y1 - x1 : x1 + y1) * 0.7071067811865475f;
    if (idx < q_pairs) {
        dst.Store((q_pairs * 2 + idx * 2) * 4, asuint(rot0));
        dst.Store((q_pairs * 2 + idx * 2 + 1) * 4, asuint(rot1));
    }
    if (idx < aux_words) {
        const uint pos_word = src.Load((fp8 ? q_pairs * 2 : q_pairs * 4) + idx * 4);
        dst.Store((q_pairs * 4 + idx) * 4, pos_word);
    }
}

[numthreads(256, 1, 1)]
void CSBulkUnpackQF16(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    unpack_query(lane, group, false);
}

[numthreads(256, 1, 1)]
void CSBulkUnpackQFP8(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    // Admission rejects every exponent-31 byte before native mutation.
    // Expand through the identical F16 bit pattern, then reuse the same WHT.
    unpack_query(lane, group, true);
}

[numthreads(256, 1, 1)]
void CSBulkPackOutF16(uint3 tid : SV_DispatchThreadID) {
    const uint idx = tid.x;
    if (idx < q_pairs) {
        const float lo = asfloat(src.Load((idx * 2) * 4));
        const float hi = asfloat(src.Load((idx * 2 + 1) * 4));
        const uint packed = (ggml_half_bits(lo) & 0xffffu) | (ggml_half_bits(hi) << 16);
        dst.Store(idx * 4, packed);
    }
    if (idx < aux_words) {
        const uint lse_word = src.Load((q_pairs * 2 + idx) * 4);
        dst.Store((q_pairs + idx) * 4, lse_word);
    }
}

// Exactly the existing F16 -> E5M2 codec, including finite saturation and RNE.
// Special values are preserved here for the isolated diagnostic oracle; serving
// rejects them through the per-group status before returning any payload.
uint fp8_from_half(uint h) {
    const uint sign = (h & 0x8000u) >> 8;
    if ((h & 0x7c00u) == 0x7c00u) {
        const uint mantissa = (h >> 8) & 3u;
        return sign | 0x7cu | ((h & 0x03ffu) ? (mantissa ? mantissa : 1u) : 0u);
    }
    const uint magnitude = min(h & 0x7fffu, 0x7b00u);
    return sign | ((magnitude + 0x7fu + ((magnitude >> 8) & 1u)) >> 8);
}

groupshared uint gs_bad;

[numthreads(256, 1, 1)]
void CSBulkPackOutFP8(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint idx = group.x * 256 + lane;
    const uint out_elements = q_pairs * 2;
    const uint quads = q_pairs / 2;
    if (lane == 0) gs_bad = 0;
    GroupMemoryBarrierWithGroupSync();
    uint bad = 0;
    if (idx < quads) {
        const uint4 source = src.Load4(idx * 16);
        uint packed = 0;
        [unroll] for (uint j = 0; j < 4; ++j) {
            // Keep the F16 rounding step. A direct F32->FP8 conversion has a
            // different result near midpoints and would accept F16 overflow.
            const uint h = ggml_half_bits(asfloat(source[j]));
            if ((h & 0x7c00u) == 0x7c00u) bad |= 1u;
            packed |= fp8_from_half(h) << (j * 8);
        }
        dst.Store(idx * 4, packed);
    }
    if (idx < aux_words) {
        const uint lse = src.Load((out_elements + idx) * 4);
        if ((lse & 0x7f800000u) == 0x7f800000u) bad |= 2u;
        dst.Store(out_elements + idx * 4, lse);
    }
    if (bad != 0) InterlockedOr(gs_bad, bad);
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0) {
        // One completely overwritten word per group; no global clear/race.
        // The command-list UAV barrier precedes the existing readback copy.
        dst.Store(out_elements + aux_words * 4 + group.x * 4, gs_bad);
    }
}
