#ifndef RKVA_DECODE_QUANTS_GROUP_B_HLSLI
#define RKVA_DECODE_QUANTS_GROUP_B_HLSLI

// RKVA Group B Quantization Codecs for Xbox Series X (Wave64 rowlane decode):
// - Q4_1: 160 bytes / 256 dims (8 blocks x 20 bytes: fp16 d, m + 16 bytes qs. x = q * d + m)
// - Q5_1: 192 bytes / 256 dims (8 blocks x 24 bytes: fp16 d, m + 4 bytes qh + 16 bytes qs. x = q * d + m)
// - Q6_1: 224 bytes / 256 dims (8 blocks x 28 bytes: fp16 d, m + 8 bytes qh + 16 bytes qs. x = q * d + m)
//
// In Wave64 decode, 64 lanes cover 64 quarter-units u in [0, 63] (256 dimensions total).
// Every block is naturally 4-byte aligned (20, 24, 28 bytes per block are all multiples of 4),
// enabling branchless, aligned dword memory loads.

#define RKVA_ROW_HEAD_Q4_1   160
#define RKVA_ROW_HEAD_Q5_1   192
#define RKVA_ROW_HEAD_Q6_1   224

#define RKVA_BLOCK_SIZE_Q4_1  20
#define RKVA_BLOCK_SIZE_Q5_1  24
#define RKVA_BLOCK_SIZE_Q6_1  28

#ifndef RKVA_QUARTER_DIM_DEFINED
#define RKVA_QUARTER_DIM_DEFINED
// Quarter unit u covers block b = u >> 3, chunk c = (u >> 1) & 3, half h = u & 1.
// Base dimension = b * 32 + h * 16 + c * 4, covering elements {d, d+1, d+2, d+3}.
uint quarter_dim(uint u) {
    return (u >> 3) * 32 + (u & 1) * 16 + ((u >> 1) & 3) * 4;
}
#endif

uint quarter_dim_q4_1(uint u) {
    return quarter_dim(u);
}

uint quarter_dim_q5_1(uint u) {
    return quarter_dim(u);
}

uint quarter_dim_q6_1(uint u) {
    return quarter_dim(u);
}

// Branchless aligned Q4_1 dequantization: 20-byte block alignment (5 dwords).
// Reads 4 floats at quarter-unit u in [0, 63] from ByteAddressBuffer at offset base.
float4 load_q4_1(ByteAddressBuffer buffer, uint base, uint u) {
    uint b = u >> 3;
    uint block_addr = base + b * 20;
    uint c = (u >> 1) & 3;

    uint dm = buffer.Load(block_addr);
    float d = f16tof32(dm & 0xFFFFu);
    float m = f16tof32(dm >> 16);

    uint qs_word = buffer.Load(block_addr + 4 + c * 4);
    qs_word >>= (u & 1u) * 4;
    float4 q = float4(qs_word & 0x0Fu,
                      (qs_word >> 8) & 0x0Fu,
                      (qs_word >> 16) & 0x0Fu,
                      (qs_word >> 24) & 0x0Fu);
    return q * d + m;
}

// Branchless aligned Q5_1 dequantization: 24-byte block alignment (6 dwords).
// Reads 4 floats at quarter-unit u in [0, 63] from ByteAddressBuffer at offset base.
float4 load_q5_1(ByteAddressBuffer buffer, uint base, uint u) {
    uint b = u >> 3;
    uint block_addr = base + b * 24;
    uint c = (u >> 1) & 3;

    uint2 dm_qh = buffer.Load2(block_addr);
    float d = f16tof32(dm_qh.x & 0xFFFFu);
    float m = f16tof32(dm_qh.x >> 16);
    uint qh = dm_qh.y;

    uint qs_word = buffer.Load(block_addr + 8 + c * 4);
    qs_word >>= (u & 1u) * 4;
    uint4 qs_low = (uint4(qs_word, qs_word >> 8, qs_word >> 16, qs_word >> 24)) & 0x0Fu;

    uint k = (u & 1u) * 16 + c * 4;
    uint qh_bits = qh >> k;
    uint4 qh_high = (uint4(qh_bits, qh_bits >> 1, qh_bits >> 2, qh_bits >> 3) & 1u) << 4;

    float4 q = float4(qs_low | qh_high);
    return q * d + m;
}

// Branchless aligned Q6_1 dequantization: 28-byte block alignment (7 dwords).
// Reads 4 floats at quarter-unit u in [0, 63] from ByteAddressBuffer at offset base.
float4 load_q6_1(ByteAddressBuffer buffer, uint base, uint u) {
    uint b = u >> 3;
    uint block_addr = base + b * 28;
    uint c = (u >> 1) & 3;

    uint dm = buffer.Load(block_addr);
    float d = f16tof32(dm & 0xFFFFu);
    float m = f16tof32(dm >> 16);

    uint qh_word = buffer.Load(block_addr + 4 + (c & 1u) * 4);
    uint qs_word = buffer.Load(block_addr + 12 + c * 4);

    qs_word >>= (u & 1u) * 4;
    uint4 qs_low = (uint4(qs_word, qs_word >> 8, qs_word >> 16, qs_word >> 24)) & 0x0Fu;

    uint4 qh_bytes = (uint4(qh_word, qh_word >> 8, qh_word >> 16, qh_word >> 24)) & 0xFFu;
    uint shift = ((c & 2u) + (u & 1u)) * 2u;
    uint4 qh_high = ((qh_bytes >> shift) & 0x03u) << 4;

    float4 q = float4(qs_low | qh_high);
    return q * d + m;
}

#endif // RKVA_DECODE_QUANTS_GROUP_B_HLSLI
