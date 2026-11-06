#ifndef RKVA_DECODE_QUANTS_GROUP_C_HLSLI
#define RKVA_DECODE_QUANTS_GROUP_C_HLSLI

// RKVA Group C rowlane dequantization formats (Wave64 on RDNA2 / Xbox Series X):
// 1. iq4_nl (144 bytes / 256 dims: 8 blocks of 18 bytes. d fp16 + 16 bytes qs. x = d * kvalues_iq4nl[q])
// 2. kvarn2 (q2_0s) (80 bytes / 256 dims: 8 blocks of 10 bytes. d fp16 + 8 bytes qs with 2-bit planes. x = (q - 2) * d)
// 3. kvarn3 (q3_0) (112 bytes / 256 dims: 8 blocks of 14 bytes. d fp16 + 4 bytes qh + 8 bytes qs. x = (q - 4) * d)
// 4. kvarn8 (q8_0) (272 bytes / 256 dims: 8 blocks of 34 bytes. d fp16 + 32 bytes qs int8. x = q * d)

#define RKVA_ROW_HEAD_IQ4_NL 144
#define RKVA_ROW_HEAD_Q2_0S   80
#define RKVA_ROW_HEAD_KVARN2  80
#define RKVA_ROW_HEAD_Q3_0   112
#define RKVA_ROW_HEAD_KVARN3 112
#define RKVA_ROW_HEAD_Q8_0   272
#define RKVA_ROW_HEAD_KVARN8 272

// ============================================================================
// 1. iq4_nl: Non-linear 4-bit quant
// ============================================================================

static const float kvalues_iq4nl[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f
};

// Quarter unit u covers block b = u >> 3, quant bytes 4c..4c+3 (c = (u >> 1) & 3)
// and nibble half h = u & 1: elements b*32 + h*16 + 4c + {0,1,2,3}.
uint quarter_dim_iq4_nl(uint u) {
    return (u >> 3) * 32 + (u & 1) * 16 + ((u >> 1) & 3) * 4;
}

// Aligned 36-byte pair (2 blocks = 9 uints):
// Even block: scale at offset 0 (bits 0..15), quants at byte 2 (aligned Load2 at 0..12 shifted).
// Odd block: scale at offset 16 (bits 16..31), quants at byte 20 (aligned Load at 20..32).
float4 load_iq4_nl(ByteAddressBuffer buffer, uint base, uint u) {
    uint pair_idx = u >> 4;
    uint addr_pair = base + pair_idx * 36;
    uint is_odd = (u >> 3) & 1;
    uint c = (u >> 1) & 3;

    uint scale_word = buffer.Load(addr_pair + is_odd * 16);
    float scale = f16tof32((scale_word >> (is_odd * 16)) & 65535u);

    uint word;
    if (is_odd) {
        word = buffer.Load(addr_pair + 20 + c * 4);
    } else {
        uint2 p = buffer.Load2(addr_pair + c * 4);
        word = (p.x >> 16) | (p.y << 16);
    }

    word >>= (u & 1u) * 4;
    uint q0 = word & 15u;
    uint q1 = (word >> 8) & 15u;
    uint q2 = (word >> 16) & 15u;
    uint q3 = (word >> 24) & 15u;

    return float4(kvalues_iq4nl[q0], kvalues_iq4nl[q1], kvalues_iq4nl[q2], kvalues_iq4nl[q3]) * scale;
}

// ============================================================================
// 2. kvarn2 / q2_0s: 2-bit planes (qs[j % 8] >> (2 * (j / 8))) & 3
// ============================================================================

uint quarter_dim_q2_0s(uint u) {
    return u * 4;
}

uint quarter_dim_kvarn2(uint u) {
    return u * 4;
}

// Aligned 20-byte pair (2 blocks = 5 uints):
// Even block: scale at pair_base+0 (bits 0..15), qs at bytes 2..9 (Load2 at h*4 shifted).
// Odd block: scale at pair_base+8 (bits 16..31), qs at bytes 12..19 (Load at 12+h*4).
float4 load_q2_0s(ByteAddressBuffer buffer, uint base, uint u) {
    uint pair_idx = u >> 4;
    uint pair_base = base + pair_idx * 20;
    uint is_odd = (u >> 3) & 1;
    uint plane = (u >> 1) & 3;
    uint h = u & 1;

    float scale;
    uint word;
    if (is_odd) {
        uint scale_word = buffer.Load(pair_base + 8);
        scale = f16tof32(scale_word >> 16);
        word = buffer.Load(pair_base + 12 + h * 4);
    } else {
        uint scale_word = buffer.Load(pair_base);
        scale = f16tof32(scale_word & 65535u);
        uint2 p = buffer.Load2(pair_base + h * 4);
        word = (p.x >> 16) | (p.y << 16);
    }

    uint w = word >> (plane * 2);
    float4 q = float4(w & 3u, (w >> 8) & 3u, (w >> 16) & 3u, (w >> 24) & 3u);
    return (q - 2.0f) * scale;
}

float4 load_kvarn2(ByteAddressBuffer buffer, uint base, uint u) {
    return load_q2_0s(buffer, base, u);
}

// ============================================================================
// 3. kvarn3 / q3_0: 3-bit (2-bit plane in qs + high bit in qh)
// ============================================================================

uint quarter_dim_q3_0(uint u) {
    return u * 4;
}

uint quarter_dim_kvarn3(uint u) {
    return u * 4;
}

// Aligned 28-byte pair (2 blocks = 7 uints):
// Even block: scale at pair_base+0 (bits 0..15), qh at bytes 2..5 (Load2 at 0 shifted),
//             qs at bytes 6..13 (Load2 at 4+h*4 shifted).
// Odd block: scale at pair_base+12 (bits 16..31), qh at bytes 16..19 (Load at 16),
//            qs at bytes 20..27 (Load at 20+h*4).
float4 load_q3_0(ByteAddressBuffer buffer, uint base, uint u) {
    uint pair_idx = u >> 4;
    uint pair_base = base + pair_idx * 28;
    uint is_odd = (u >> 3) & 1;
    uint plane = (u >> 1) & 3;
    uint h = u & 1;

    float scale;
    uint qh;
    uint word;
    if (is_odd) {
        uint scale_word = buffer.Load(pair_base + 12);
        scale = f16tof32(scale_word >> 16);
        qh = buffer.Load(pair_base + 16);
        word = buffer.Load(pair_base + 20 + h * 4);
    } else {
        uint2 p0 = buffer.Load2(pair_base);
        scale = f16tof32(p0.x & 65535u);
        qh = (p0.x >> 16) | (p0.y << 16);
        uint2 p = buffer.Load2(pair_base + 4 + h * 4);
        word = (p.x >> 16) | (p.y << 16);
    }

    uint w = word >> (plane * 2);
    float4 q_low = float4(w & 3u, (w >> 8) & 3u, (w >> 16) & 3u, (w >> 24) & 3u);

    uint j0 = (u & 7) * 4;
    uint qh_bits = (qh >> j0) & 15u;
    float4 q_high = float4(qh_bits & 1u, (qh_bits >> 1) & 1u, (qh_bits >> 2) & 1u, (qh_bits >> 3) & 1u);

    float4 q = q_low + q_high * 4.0f;
    return (q - 4.0f) * scale;
}

float4 load_kvarn3(ByteAddressBuffer buffer, uint base, uint u) {
    return load_q3_0(buffer, base, u);
}

// ============================================================================
// 4. kvarn8 / q8_0: 8-bit signed int quants
// ============================================================================

uint quarter_dim_q8_0(uint u) {
    return u * 4;
}

uint quarter_dim_kvarn8(uint u) {
    return u * 4;
}

// Aligned 68-byte pair (2 blocks = 17 uints):
// Even block: scale at pair_base+0 (bits 0..15), quants at bytes 2..33 (Load2 at c*4 shifted).
// Odd block: scale at pair_base+32 (bits 16..31), quants at bytes 36..67 (Load at 36+c*4).
float4 load_q8_0(ByteAddressBuffer buffer, uint base, uint u) {
    uint pair_idx = u >> 4;
    uint pair_base = base + pair_idx * 68;
    uint is_odd = (u >> 3) & 1;
    uint c = u & 7;

    float scale;
    uint word;
    if (is_odd) {
        uint scale_word = buffer.Load(pair_base + 32);
        scale = f16tof32(scale_word >> 16);
        word = buffer.Load(pair_base + 36 + c * 4);
    } else {
        uint scale_word = buffer.Load(pair_base);
        scale = f16tof32(scale_word & 65535u);
        uint2 p = buffer.Load2(pair_base + c * 4);
        word = (p.x >> 16) | (p.y << 16);
    }

    int4 quants = int4(asint(word << 24) >> 24,
                       asint(word << 16) >> 24,
                       asint(word << 8)  >> 24,
                       asint(word)       >> 24);
    return float4(quants) * scale;
}

float4 load_kvarn8(ByteAddressBuffer buffer, uint base, uint u) {
    return load_q8_0(buffer, base, u);
}

// Rowlane template integration macros (optional convenience)
#if defined(ROWLANE_FORMAT_IQ4_NL)
#define ROWLANE_LOAD load_iq4_nl
#define ROWLANE_QUARTER_DIM quarter_dim_iq4_nl
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_IQ4_NL
#elif defined(ROWLANE_FORMAT_Q2_0S) || defined(ROWLANE_FORMAT_KVARN2)
#define ROWLANE_LOAD load_q2_0s
#define ROWLANE_QUARTER_DIM quarter_dim_q2_0s
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_Q2_0S
#elif defined(ROWLANE_FORMAT_Q3_0) || defined(ROWLANE_FORMAT_KVARN3)
#define ROWLANE_LOAD load_q3_0
#define ROWLANE_QUARTER_DIM quarter_dim_q3_0
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_Q3_0
#elif defined(ROWLANE_FORMAT_Q8_0) || defined(ROWLANE_FORMAT_KVARN8)
#define ROWLANE_LOAD load_q8_0
#define ROWLANE_QUARTER_DIM quarter_dim_q8_0
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_Q8_0
#endif

#endif // RKVA_DECODE_QUANTS_GROUP_C_HLSLI
