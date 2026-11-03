#ifndef RKVA_BDR_HLSLI
#define RKVA_BDR_HLSLI

// RKVA Block-Diagonal Rotation (BDR-16) via Wave64 intrinsics for Xbox Series X (RDNA2).
//
// Computes normalized 16-point Sylvester Fast Walsh-Hadamard Transform (FWHT)
// scaled by 1/sqrt(16) = 0.25f across every 4 contiguous lanes.
//
// In Wave64 decode rowlane:
//   Each lane holds float4 (4 dimensions).
//   4 consecutive lanes (lane & ~3u) hold 4 x 4 = 16 elements = exactly one BDR-16 block.
//   64 lanes in the wave hold 16 independent BDR-16 blocks covering all 256 dimensions.
//
// Properties:
//   - Self-inverse: bdr16_wave4(bdr16_wave4(v, lane), lane) == v.
//   - Zero shared memory: uses only registers and WaveReadLaneAt.
//   - Zero threadgroup barriers: entirely wavefront-local.

float4 bdr16_wave4(float4 v, uint lane) {
    // 1. Intra-lane butterfly across the 4 elements of this lane (stages 1 & 2)
    // Stage 1 (stride 1)
    float u0 = v.x + v.y;
    float w0 = v.x - v.y;
    float u1 = v.z + v.w;
    float w1 = v.z - v.w;
    // Stage 2 (stride 2)
    v.x = u0 + u1;
    v.z = u0 - u1;
    v.y = w0 + w1;
    v.w = w0 - w1;

    // 2. Inter-lane butterfly across the 4 lanes of this block (stages 3 & 4)
    // Stage 3 (stride 4 in elements = stride 1 in lanes)
    uint partner1 = lane ^ 1u;
    float4 other1 = WaveReadLaneAt(v, partner1);
    v = ((lane & 1u) == 0u) ? (v + other1) : (other1 - v);

    // Stage 4 (stride 8 in elements = stride 2 in lanes)
    uint partner2 = lane ^ 2u;
    float4 other2 = WaveReadLaneAt(v, partner2);
    v = ((lane & 2u) == 0u) ? (v + other2) : (other2 - v);

    // Scale by 1/sqrt(16) = 0.25f
    return v * 0.25f;
}

#endif // RKVA_BDR_HLSLI
