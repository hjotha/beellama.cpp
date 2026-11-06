#ifndef RKVA_DECODE_QUANTS_SAW_INT4_HLSLI
#define RKVA_DECODE_QUANTS_SAW_INT4_HLSLI

// RKVA SAW-INT4 (Block-Diagonal Rotation + INT4 KV-cache) Wave64 Dequantization.
//
// Layout per 256-element head row:
//   - Bytes 0..127:   128 bytes of packed 4-bit nibbles (32 uints).
//                     Byte j holds element j in low nibble (bits 0..3)
//                     and element j+128 in high nibble (bits 4..7).
//   - Bytes 128..131: float scale (4 bytes, aligned).
//   - Bytes 132..135: float zero  (4 bytes, aligned).
//   Total: 136 bytes per head row (34 uints, perfectly 4-byte aligned).
//
// In Wave64 rowlane decode:
//   64 lanes map to 64 quarter-units u in [0, 63].
//   Lanes 0..31 cover elements 0..127 (low nibbles).
//   Lanes 32..63 cover elements 128..255 (high nibbles).
//   Each lane u loads 1 uint from offset (u & 31) * 4, and extracts 4 4-bit values.

#define RKVA_ROW_HEAD_SAW_INT4 136

uint quarter_dim_saw_int4(uint u) {
    // 128 + 4*(u - 32) == 4*u for all u in [0, 63]: direct shift without branch.
    return u << 2;
}

float4 load_saw_int4(ByteAddressBuffer buffer, uint base, uint u) {
    uint k = u & 31u;
    uint word = buffer.Load(base + k * 4u);
    uint is_hi = (u >> 5) & 1u;
    word >>= (is_hi * 4u);

    uint4 q = uint4(word & 0x0Fu,
                    (word >> 8) & 0x0Fu,
                    (word >> 16) & 0x0Fu,
                    (word >> 24) & 0x0Fu);

    float2 sz = asfloat(buffer.Load2(base + 128u));
    return (float4(q) - sz.y) * sz.x;
}

#endif // RKVA_DECODE_QUANTS_SAW_INT4_HLSLI
