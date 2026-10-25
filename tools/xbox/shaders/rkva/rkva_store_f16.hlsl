// RKVA F16 KV store — scatter F32 K/V rows into the per-layer F16 KV cache at
// their absolute positions, converting with GGML's round-to-nearest-even F16.
//
// RWByteAddressBuffer.Store writes 32 bits, so each thread packs TWO adjacent
// F16 values (dims 2*i, 2*i+1) into one 32-bit word. head_dim is even (256).
//
// Upload layout (F32):
//   src:       [n_tokens][n_head_kv][head_dim]
//   positions: [n_tokens] i32
// KV destination (F16, packed word = 2 dims), per layer (engine offsets u0):
//   dst words: [capacity][n_head_kv][head_dim/2]
//
// One thread per (token, kv_head, dim-pair). b0 = {n_tokens, n_head_kv,
// head_dim, capacity}. t0 = positions, t1 = src (F32), u0 = dst (F16 words).

cbuffer Params : register(b0) {
    uint n_tokens;
    uint n_head_kv;
    uint head_dim;     // 256 (even)
    uint capacity;
};

ByteAddressBuffer   positions : register(t0); // i32 [n_tokens]
ByteAddressBuffer   src       : register(t1); // F32 [n_tokens][n_head_kv][head_dim]
RWByteAddressBuffer dst       : register(u0); // F16 words [capacity][n_head_kv][head_dim/2]

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
        if (half_exponent < -10) return sign;
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
    if (half_exponent >= 31) return sign | 0x7c00u;
    uint rounded = mantissa >> 13;
    const uint remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        ++rounded;
    }
    if (rounded == 0x400u) {
        rounded = 0;
        ++half_exponent;
    }
    if (half_exponent >= 31) return sign | 0x7c00u;
    return sign | (uint(half_exponent) << 10) | rounded;
}

[numthreads(256, 1, 1)]
void CSStore(uint3 tid : SV_DispatchThreadID) {
    const uint half_dim = head_dim / 2;            // words per (token, kv_head)
    const uint pairs_per_token = n_head_kv * half_dim;
    const uint elem = tid.x;                        // pair index
    if (elem >= n_tokens * pairs_per_token) {
        return;
    }
    const uint token = elem / pairs_per_token;
    const uint within = elem % pairs_per_token;     // kv_head*half_dim + d_pair
    const int pos = positions.Load(token * 4);
    if (pos < 0 || uint(pos) >= capacity) {
        return;
    }
    // Source F32 indices for the two dims of this pair.
    const uint src_base = (token * n_head_kv * head_dim) + within * 2;
    const float lo = asfloat(src.Load(src_base * 4));
    const float hi = asfloat(src.Load((src_base + 1) * 4));
    const uint packed = (ggml_half_bits(lo) & 0xffffu) | (ggml_half_bits(hi) << 16);
    // Destination word index: (pos, kv_head, d_pair) in the packed F16 layout.
    const uint dst_word = uint(pos) * pairs_per_token + within;
    dst.Store(dst_word * 4, packed);
}
