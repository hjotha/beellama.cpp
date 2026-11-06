#ifndef RKVA_DECODE_QUANTS_OSCAR_INT2_HLSLI
#define RKVA_DECODE_QUANTS_OSCAR_INT2_HLSLI

// RKVA OSCAR INT2 (Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache)
// Wave64 Rowlane Dequantization.
//
// Layout per 256-element head row (96 bytes = 24 uints):
//   - 8 blocks of 32 elements each (8 * 12 bytes = 96 bytes)
//   - Each block (12 bytes, 3 uints):
//       uint 0 (bytes 0..3):   d:fp16 in bits 0..15 (scale sigma),
//                              m:fp16 in bits 16..31 (offset mean)
//       uint 1 (bytes 4..7):   qs[0..3] (elements 0..15, 4 2-bit quants per byte)
//       uint 2 (bytes 8..11):  qs[4..7] (elements 16..31, 4 2-bit quants per byte)
//   Total: 96 bytes per head row (vs 136 bytes SAW-INT4, 272 bytes Q8_0, 512 bytes F16).
//
// In Wave64 rowlane decode:
//   64 lanes map to 64 quarter-units u in [0, 63], covering all 256 dimensions.
//   Block index ib = u >> 3 (0..7).
//   Quarter index within block iq = u & 7 (0..7).
//   Each lane loads exactly 2 uints via 4-byte aligned ByteAddressBuffer loads.
//   Reconstruction uses Lloyd-Max 4-level centroids for N(0, 1):
//     {-0.9816, -0.4528, 0.4528, 0.9816}.
//   Completely branchless execution.

#define RKVA_ROW_HEAD_OSCAR_INT2 96

static const float kOscarCentroids[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};

uint quarter_dim_oscar_int2(uint u) {
    // 64 quarters cover dims 0..255 linearly and disjointly
    return u * 4u;
}

float4 load_oscar_int2(ByteAddressBuffer buffer, uint base, uint u) {
    // Block offset: each block is 12 bytes (always 4-byte aligned: 0, 12, 24, ..., 84)
    uint blk_offset = base + (u >> 3) * 12u;

    // Load scale (sigma) and offset (mean)
    uint dm = buffer.Load(blk_offset);
    float d = f16tof32(dm & 0xFFFFu);
    float m = f16tof32(dm >> 16u);

    // Quarter within block: iq = u & 7
    // iq 0..3 in uint 1 (bytes 4..7); iq 4..7 in uint 2 (bytes 8..11)
    uint is_hi = (u >> 2) & 1u;
    uint qs_word = buffer.Load(blk_offset + 4u + is_hi * 4u);

    // Extract the specific packed byte for quarter (iq & 3)
    uint byte_idx = u & 3u;
    uint packed = (qs_word >> (byte_idx * 8u)) & 0xFFu;

    // Extract 4 2-bit Lloyd-Max codes
    uint c0 = packed & 3u;
    uint c1 = (packed >> 2u) & 3u;
    uint c2 = (packed >> 4u) & 3u;
    uint c3 = (packed >> 6u) & 3u;

    float4 lm = float4(kOscarCentroids[c0],
                       kOscarCentroids[c1],
                       kOscarCentroids[c2],
                       kOscarCentroids[c3]);

    return m + d * lm;
}

#endif // RKVA_DECODE_QUANTS_OSCAR_INT2_HLSLI
