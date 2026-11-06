// RKVA Group A rowlane dequantization formats for Xbox Series X (Wave64).
// Formats:
//   1. f16   - 512 bytes / 256 dims (256 fp16 elements)
//   2. q5_0  - 176 bytes / 256 dims (8 blocks of 22 bytes: kvarn5)
//   3. q6_0  - 208 bytes / 256 dims (8 blocks of 26 bytes: kvarn6)
//
// In Wave64 rowlane decode, 64 lanes map to 64 quarter-units u in [0, 63].
// Each quarter-unit u loads 4 floats (float4) covering 4 dimensions given by quarter_dim(u).
// Across all 64 quarter-units, all 256 dimensions [0, 255] are covered without overlap.
//
// Memory Alignment:
//   ByteAddressBuffer.Load and Load2/Load3 require 4-byte aligned addresses.
//   - f16: Each unit u is 8 bytes (4 half-precision floats), perfectly 8-byte aligned.
//   - q5_0: 22 bytes/block. Blocks are paired into 44-byte chunks (11 uints, multiple of 4).
//           Even blocks are loaded with aligned Load2; odd blocks are loaded with aligned Load.
//   - q6_0: 26 bytes/block. Blocks are paired into 52-byte chunks (13 uints, multiple of 4).
//           Even blocks are loaded with aligned Load3 and Load2; odd blocks with aligned Load/Load2.

#ifndef RKVA_DECODE_QUANTS_GROUP_A_HLSLI
#define RKVA_DECODE_QUANTS_GROUP_A_HLSLI

// =============================================================================
// Format 1: F16 (512 bytes / 256 dimensions)
// =============================================================================

uint quarter_dim_f16(uint u) {
    return u * 4;
}

// Loads 4 half-precision floats at quarter-unit u and converts to float4.
float4 load_f16(ByteAddressBuffer buffer, uint base, uint u) {
    uint2 packed = buffer.Load2(base + u * 8);
    return float4(f16tof32(packed.x & 65535u), f16tof32(packed.x >> 16),
                  f16tof32(packed.y & 65535u), f16tof32(packed.y >> 16));
}

// =============================================================================
// Format 2: Q5_0 / KVarN5 (176 bytes / 256 dimensions: 8 blocks x 22 bytes)
// =============================================================================
// Struct block_q5_0:
//   ggml_half d      (2 bytes at offset 0)
//   uint8_t   qh[4]  (4 bytes at offset 2: 5th bit of quants)
//   uint8_t   qs[16] (16 bytes at offset 6: packed 4-bit nibbles)
//
// 44-byte (2 blocks = 11 uints) pair alignment:
//   Even block (is_odd == 0):
//     Offset 0: scale d (bits 0..15) + qh[0..1] (bits 16..31) -> Load2(addr_pair)
//     Offset 4: qh[2..3] (bits 0..15)
//     Offset 6: qs[16] starts (Load2 at addr_pair + 4 + c*4)
//   Odd block (is_odd == 1):
//     Offset 20: bits 16..31 contain scale d -> Load(addr_pair + 20)
//     Offset 24: qh[0..3] (4 bytes, aligned) -> Load(addr_pair + 24)
//     Offset 28: qs[16] (16 bytes, aligned)  -> Load(addr_pair + 28 + c*4)

uint quarter_dim_q5_0(uint u) {
    // Block b = u >> 3, nibble half h = u & 1, chunk c = (u >> 1) & 3
    return (u >> 3) * 32 + (u & 1) * 16 + ((u >> 1) & 3) * 4;
}

float4 load_q5_0(ByteAddressBuffer buffer, uint base, uint u) {
    uint pair_idx = u >> 4;
    uint addr_pair = base + pair_idx * 44;
    uint is_odd = (u >> 3) & 1;
    uint h = u & 1;
    uint c = (u >> 1) & 3;

    float scale;
    uint qh;
    uint qs_bytes;

    if (is_odd) {
        uint scale_word = buffer.Load(addr_pair + 20);
        scale = f16tof32(scale_word >> 16);
        qh = buffer.Load(addr_pair + 24);
        qs_bytes = buffer.Load(addr_pair + 28 + c * 4);
    } else {
        uint2 w01 = buffer.Load2(addr_pair);
        scale = f16tof32(w01.x & 65535u);
        qh = (w01.x >> 16) | (w01.y << 16);
        uint2 p = buffer.Load2(addr_pair + 4 + c * 4);
        qs_bytes = (p.x >> 16) | (p.y << 16);
    }

    uint word = qs_bytes >> (h * 4);
    uint4 qs_low = uint4(word & 15u, (word >> 8) & 15u, (word >> 16) & 15u, (word >> 24) & 15u);

    // 5th bit from qh: bits 0..15 for h=0, bits 16..31 for h=1; 4 consecutive bits per chunk c
    uint qh_bits = qh >> (h * 16 + c * 4);
    uint4 qh_high = uint4((qh_bits & 1u) << 4,
                          ((qh_bits >> 1) & 1u) << 4,
                          ((qh_bits >> 2) & 1u) << 4,
                          ((qh_bits >> 3) & 1u) << 4);

    uint4 q5 = qs_low | qh_high;
    return (float4(q5) - 16.0f) * scale;
}

// Aliases for kvarn5
#define quarter_dim_kvarn5 quarter_dim_q5_0
#define load_kvarn5 load_q5_0

// =============================================================================
// Format 3: Q6_0 / KVarN6 (208 bytes / 256 dimensions: 8 blocks x 26 bytes)
// =============================================================================
// Struct block_q6_0:
//   ggml_half d      (2 bytes at offset 0)
//   uint8_t   qh[8]  (8 bytes at offset 2: upper 2 bits of quants)
//   uint8_t   qs[16] (16 bytes at offset 10: lower 4 bits of quants)
//
// 52-byte (2 blocks = 13 uints) pair alignment:
//   Even block (is_odd == 0):
//     Offset 0: scale d (bits 0..15), qh[0..1] (bits 16..31)
//     Offset 4: qh[2..3] (bits 0..15), qh[4..5] (bits 16..31)
//     Offset 8: qh[6..7] (bits 0..15), qs[0..1] (bits 16..31)
//     -> Load3(addr_pair) gets scale, qh[0..3] (qh0), qh[4..7] (qh1)
//     Offset 10: qs[16] starts (Load2 at addr_pair + 8 + c*4)
//   Odd block (is_odd == 1):
//     Offset 24: bits 16..31 contain scale d -> Load(addr_pair + 24)
//     Offset 28: qh[0..7] (8 bytes, aligned) -> Load2(addr_pair + 28)
//     Offset 36: qs[16] (16 bytes, aligned)  -> Load(addr_pair + 36 + c*4)

uint quarter_dim_q6_0(uint u) {
    // Block b = u >> 3, nibble half h = u & 1, chunk c = (u >> 1) & 3
    return (u >> 3) * 32 + (u & 1) * 16 + ((u >> 1) & 3) * 4;
}

float4 load_q6_0(ByteAddressBuffer buffer, uint base, uint u) {
    uint pair_idx = u >> 4;
    uint addr_pair = base + pair_idx * 52;
    uint is_odd = (u >> 3) & 1;
    uint h_half = u & 1;
    uint c = (u >> 1) & 3;

    float scale;
    uint qh0, qh1;
    uint qs_bytes;

    if (is_odd) {
        uint scale_word = buffer.Load(addr_pair + 24);
        scale = f16tof32(scale_word >> 16);
        uint2 qh_odd = buffer.Load2(addr_pair + 28);
        qh0 = qh_odd.x;
        qh1 = qh_odd.y;
        qs_bytes = buffer.Load(addr_pair + 36 + c * 4);
    } else {
        uint3 w012 = buffer.Load3(addr_pair);
        scale = f16tof32(w012.x & 65535u);
        qh0 = (w012.x >> 16) | (w012.y << 16);
        qh1 = (w012.y >> 16) | (w012.z << 16);
        uint2 p = buffer.Load2(addr_pair + 8 + c * 4);
        qs_bytes = (p.x >> 16) | (p.y << 16);
    }

    // Select qh dword: c=0,2 uses qh0 (qh[0..3]); c=1,3 uses qh1 (qh[4..7])
    uint qh_word = (c & 1) ? qh1 : qh0;
    // Shift within each qh byte:
    // c=0,1 (j in 0..7) uses lower 4 bits of qh; c=2,3 (j in 8..15) uses upper 4 bits (+4).
    // h_half=0 takes bits 0..1 (+0); h_half=1 takes bits 2..3 (+2).
    uint qh_shift = ((c >> 1) & 1) * 4 + h_half * 2;

    uint4 qh_high = uint4(((qh_word >> qh_shift) & 3u) << 4,
                          ((qh_word >> (8 + qh_shift)) & 3u) << 4,
                          ((qh_word >> (16 + qh_shift)) & 3u) << 4,
                          ((qh_word >> (24 + qh_shift)) & 3u) << 4);

    uint word = qs_bytes >> (h_half * 4);
    uint4 qs_low = uint4(word & 15u, (word >> 8) & 15u, (word >> 16) & 15u, (word >> 24) & 15u);

    uint4 q6 = qs_low | qh_high;
    return (float4(q6) - 32.0f) * scale;
}

// Aliases for kvarn6
#define quarter_dim_kvarn6 quarter_dim_q6_0
#define load_kvarn6 load_q6_0

#endif // RKVA_DECODE_QUANTS_GROUP_A_HLSLI
