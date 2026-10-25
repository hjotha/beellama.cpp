// GGML QK_K = 256. Offsets and bit order follow ggml-common.h and
// dequantize_row_q3_K/dequantize_row_q4_K/dequantize_row_q5_K in ggml-quants.c.
// W is declared by quant_k_matmul.hlsli before including this file.

uint dflash_load_u8(uint offset) {
    return (W.Load(offset & ~3u) >> ((offset & 3u) * 8u)) & 255u;
}

uint dflash_load_u16(uint even_offset) {
    return (W.Load(even_offset & ~3u) >> ((even_offset & 2u) * 8u)) & 65535u;
}

float dflash_q3k(uint base, uint element, float d) {
    // block_q3_K: hmask[32], qs[64], scales[12], fp16 d; 110 bytes.
    const uint half = element >> 7u;
    const uint quarter = (element >> 5u) & 3u;
    const uint lane32 = element & 31u;
    const uint low2 = (dflash_load_u8(base + 32u + half * 32u + lane32)
                     >> (quarter * 2u)) & 3u;
    const uint high_bit = (dflash_load_u8(base + lane32)
                          >> (half * 4u + quarter)) & 1u;
    const int quant = int(low2) - (high_bit == 0u ? 4 : 0);

    const uint scale_index = half * 8u + quarter * 2u + (lane32 >> 4u);
    const uint scale_byte = dflash_load_u8(base + 96u + (scale_index & 7u));
    const uint high_byte = dflash_load_u8(base + 104u + (scale_index & 3u));
    const uint low4 = (scale_byte >> (scale_index >= 8u ? 4u : 0u)) & 15u;
    const uint high2 = (high_byte >> (2u * (scale_index >> 2u))) & 3u;
    const int scale = int(low4 | (high2 << 4u)) - 32;
    return (d * float(scale)) * float(quant);
}

float dflash_q4k(uint base, uint element, float d, float dmin) {
    // block_q4_K: fp16 d,dmin; scales[12]; qs[128]; 144 bytes.
    const uint scale_index = element >> 5u;
    uint scale, minimum;
    if (scale_index < 4u) {
        scale = dflash_load_u8(base + 4u + scale_index) & 63u;
        minimum = dflash_load_u8(base + 8u + scale_index) & 63u;
    } else {
        scale = (dflash_load_u8(base + 8u + scale_index) & 15u)
                | ((dflash_load_u8(base + scale_index) >> 6u) << 4u);
        minimum = (dflash_load_u8(base + 8u + scale_index) >> 4u)
                  | ((dflash_load_u8(base + 4u + scale_index) >> 6u) << 4u);
    }

    const uint packed = dflash_load_u8(base + 16u + (element >> 6u) * 32u
                                       + (element & 31u));
    const uint quant = (packed >> ((element & 32u) != 0u ? 4u : 0u)) & 15u;
    return (d * float(scale)) * float(quant) - dmin * float(minimum);
}

float dflash_q5k(uint base, uint element, float d, float dmin) {
    // block_q5_K: fp16 d,dmin; scales[12]; qh[32]; qs[128]; 176 bytes.
    const uint scale_index = element >> 5u;
    uint scale, minimum;
    if (scale_index < 4u) {
        scale = dflash_load_u8(base + 4u + scale_index) & 63u;
        minimum = dflash_load_u8(base + 8u + scale_index) & 63u;
    } else {
        scale = (dflash_load_u8(base + 8u + scale_index) & 15u)
                | ((dflash_load_u8(base + scale_index) >> 6u) << 4u);
        minimum = (dflash_load_u8(base + 8u + scale_index) >> 4u)
                  | ((dflash_load_u8(base + 4u + scale_index) >> 6u) << 4u);
    }

    const uint lane32 = element & 31u;
    const uint packed = dflash_load_u8(base + 48u + (element >> 6u) * 32u + lane32);
    const uint low4 = (packed >> ((element & 32u) != 0u ? 4u : 0u)) & 15u;
    const uint high1 = (dflash_load_u8(base + 16u + lane32)
                        >> (element >> 5u)) & 1u;
    const float quant = float(low4 | (high1 << 4u));
    return (d * float(scale)) * quant - dmin * float(minimum);
}
