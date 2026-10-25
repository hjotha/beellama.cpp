// Q4_0 KV cache. One GPU block is 5 uints: fp16 d in low 16 bits, then
// 16 GGML qs bytes in little-endian order. GGML's file block omits 2 padding
// bytes after d; the aligned GPU stride is 20 rather than 18 bytes.
#include "ggml_float.hlsli"

cbuffer Params : register(b0) {
    uint block_count;
};

ByteAddressBuffer source : register(t0);
RWByteAddressBuffer destination : register(u0);

// DXC's f32tof16 may truncate on a device. GGML stores d with IEEE
// round-to-nearest-even, so convert explicitly before packing the scale.
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

// Store the IEEE sign bit explicitly when GGML's integer-times-scale
// dequantization yields zero. Console GPU arithmetic can fold 0*d to +0.
uint ggml_product_bits(int quant, float d, uint scale_sign) {
    if (quant == 0 || d == 0.0f) {
        const uint sign = scale_sign ^ (quant < 0 ? 0x80000000u : 0u);
        return sign;
    }
    return asuint(float(quant) * d);
}

[numthreads(64, 1, 1)]
void Pack(uint3 tid : SV_DispatchThreadID) {
    const uint block = tid.x;
    if (block >= block_count) return;
    const uint src = block * 128;
    const uint dst = block * 20;
    float amax = 0.0f;
    float max_value = 0.0f;
    [unroll] for (uint j = 0; j < 32; ++j) {
        const float v = asfloat(source.Load(src + j * 4));
        if (amax < abs(v)) {
            amax = abs(v);
            max_value = v;
        }
    }
    const float d = max_value / -8.0f;
    // Match quantize_row_q4_0_ref: rounded reciprocal, then separate F32
    // multiply and add. An approximate reciprocal can cross a nibble boundary.
    const float inv_d = d != 0.0f ? ggml_round_f32(1.0 / (double)d) : 0.0f;
    destination.Store(dst, ggml_half_bits(d));
    [unroll] for (uint word = 0; word < 4; ++word) {
        uint packed = 0;
        [unroll] for (uint byte_index = 0; byte_index < 4; ++byte_index) {
            const uint j = word * 4 + byte_index;
            precise float lo = asfloat(source.Load(src + j * 4)) * inv_d;
            precise float hi = asfloat(source.Load(src + (j + 16) * 4)) * inv_d;
            // Round the addition before truncation. Fast fadd followed by
            // fptosi can otherwise lose a carry into the next integer.
            const uint qlo = min(15u, (uint)(int)ggml_round_f32((double)lo + 8.5));
            const uint qhi = min(15u, (uint)(int)ggml_round_f32((double)hi + 8.5));
            packed |= (qlo | (qhi << 4)) << (byte_index * 8);
        }
        destination.Store(dst + 4 + word * 4, packed);
    }
}

[numthreads(64, 1, 1)]
void Unpack(uint3 tid : SV_DispatchThreadID) {
    const uint block = tid.x;
    if (block >= block_count) return;
    const uint src = block * 20;
    const uint dst = block * 128;
    const uint scale_bits = source.Load(src) & 0xffffu;
    const uint scale_sign = (scale_bits & 0x8000u) << 16;
    const float d = f16tof32(scale_bits);
    [unroll] for (uint word = 0; word < 4; ++word) {
        const uint packed = source.Load(src + 4 + word * 4);
        [unroll] for (uint byte_index = 0; byte_index < 4; ++byte_index) {
            const uint j = word * 4 + byte_index;
            const uint byte_value = (packed >> (byte_index * 8)) & 255u;
            destination.Store(dst + j * 4, ggml_product_bits(int(byte_value & 15u) - 8, d, scale_sign));
            destination.Store(dst + (j + 16) * 4, ggml_product_bits(int(byte_value >> 4) - 8, d, scale_sign));
        }
    }
}
