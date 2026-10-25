// DFlash GGML K-quant/F32 W[N,K] times X[batch,K] -> Y[batch,N].
// GEMV: b0={n,k_dim,nb,pad0}, Dispatch(N,1,1), Y[row].
// Batch tile > 1: pad0=batch, Dispatch(N,ceil(batch/tile),1), Y[batch*N+row].
// Each of 256 lanes decodes one weight per GGML superblock (or one F32 value
// per 256-wide segment). Batch tiles reuse
// that decoded weight. Compile with DFLASH_BATCH_TILE=1 for GEMV and 2/4/8
// for small-batch GEMM. K must be divisible by 256, as required by GGML.
// A Q3_K tensor with an odd total block count needs a 4-byte-padded W SRV:
// reading its final fp16 scale with ByteAddressBuffer.Load spans that padding.

#ifndef DFLASH_BATCH_TILE
#define DFLASH_BATCH_TILE 1
#endif

#if DFLASH_BATCH_TILE < 1 || DFLASH_BATCH_TILE > 8
#error DFLASH_BATCH_TILE must be between 1 and 8
#endif

#if !defined(DFLASH_Q3_K) && !defined(DFLASH_Q4_K) && !defined(DFLASH_Q5_K) && !defined(DFLASH_F32)
#error Select DFLASH_Q3_K, DFLASH_Q4_K, DFLASH_Q5_K or DFLASH_F32
#endif

cbuffer Params : register(b0) {
    uint n;
    uint k_dim;
    uint nb;
    uint pad0;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer X : register(t1);
RWStructuredBuffer<float> Y : register(u0);

#include "quant_k_decode.hlsli"

groupshared float dflash_partial[DFLASH_BATCH_TILE * 256];

[numthreads(256, 1, 1)]
void CSMain(uint lane : SV_GroupIndex, uint3 group_id : SV_GroupID) {
    const uint row = group_id.x;
#if DFLASH_BATCH_TILE == 1
    const uint batch = 1u;
#else
    const uint batch = pad0;
#endif
    const uint batch_start = group_id.y * DFLASH_BATCH_TILE;
    // These are group-uniform: no lane skips a barrier independently.
    if (row >= n || batch_start >= batch)
        return;

    float accum[DFLASH_BATCH_TILE];
    [unroll]
    for (uint b = 0u; b < DFLASH_BATCH_TILE; ++b)
        accum[b] = 0.0f;

#if defined(DFLASH_F32)
    [loop]
    for (uint column = lane; column < k_dim; column += 256u) {
        const float weight = asfloat(W.Load((row * k_dim + column) * 4u));
        [unroll]
        for (uint b = 0u; b < DFLASH_BATCH_TILE; ++b) {
            if (batch_start + b < batch) {
                const uint x_bits = X.Load(((batch_start + b) * k_dim + column) * 4u);
                accum[b] += weight * asfloat(x_bits);
            }
        }
    }
#else
    const uint blocks_per_row = nb;
    [loop]
    for (uint block = 0u; block < blocks_per_row; ++block) {
#if defined(DFLASH_Q3_K)
        const uint base = (row * blocks_per_row + block) * 110u;
        const uint fp16 = dflash_load_u16(base + 108u);
        const float weight = dflash_q3k(base, lane, f16tof32(fp16));
#elif defined(DFLASH_Q4_K)
        const uint base = (row * blocks_per_row + block) * 144u;
        const uint scales = W.Load(base);
        const float weight = dflash_q4k(base, lane,
                                        f16tof32(scales & 65535u),
                                        f16tof32(scales >> 16u));
#else
        const uint base = (row * blocks_per_row + block) * 176u;
        const uint scales = W.Load(base);
        const float weight = dflash_q5k(base, lane,
                                        f16tof32(scales & 65535u),
                                        f16tof32(scales >> 16u));
#endif
        const uint column = (block << 8u) + lane;
        [unroll]
        for (uint b = 0u; b < DFLASH_BATCH_TILE; ++b) {
            if (batch_start + b < batch) {
                const uint x_bits = X.Load(((batch_start + b) * k_dim + column) * 4u);
                accum[b] += weight * asfloat(x_bits);
            }
        }
    }
#endif

    [unroll]
    for (uint b = 0u; b < DFLASH_BATCH_TILE; ++b)
        dflash_partial[b * 256u + lane] = accum[b];
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            [unroll]
            for (uint b = 0u; b < DFLASH_BATCH_TILE; ++b)
                dflash_partial[b * 256u + lane] += dflash_partial[b * 256u + lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) {
        [unroll]
        for (uint b = 0u; b < DFLASH_BATCH_TILE; ++b) {
            if (batch_start + b < batch)
                Y[(batch_start + b) * n + row] = dflash_partial[b * 256u];
        }
    }
}
