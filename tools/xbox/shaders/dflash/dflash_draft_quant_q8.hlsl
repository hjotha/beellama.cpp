// DFlash2 Q4_K/Q5_K W[N,K] x GGML Q8_K X[batch,K] -> F32 Y[batch,N].
// Q8_K layout: float d, int8 qs[256], int16 bsums[16] = 292 bytes.
// The Q8 pack kernel is shared with SYNC.
// Q4 matches GGML CPU_REPACK: GEMM accumulates each 64-element subblock,
// trailing GEMV rows each 256-element block. Q5 follows AVX2 eight-lane FMA.
// b0={N,K,blocks_per_row,batch}, W=t0, Q8=t1, Y=u0.
#if !defined(DFLASH_Q4_K) && !defined(DFLASH_Q5_K)
#error Select DFLASH_Q4_K or DFLASH_Q5_K
#endif

#include "ggml_float.hlsli"

cbuffer Params : register(b0) {
    uint n;
    uint k_dim;
    uint nb;
    uint batch;
};
ByteAddressBuffer W : register(t0);
ByteAddressBuffer Q8 : register(t1);
RWStructuredBuffer<float> Y : register(u0);

#if defined(DFLASH_Q4_K)
#define BLOCK_BYTES 144u
#define BLOCK_U32_COUNT 36u
#define BLOCK_U32X4_COUNT 9u
#else
#define BLOCK_BYTES 176u
#define BLOCK_U32_COUNT 44u
#define BLOCK_U32X4_COUNT 11u
#endif

// WARP uses four-lane waves; Xbox uses 64. Reserve for both.
groupshared int partial[2 * 8 * 64];
groupshared uint s_w[BLOCK_U32_COUNT];
groupshared uint s_q8[512];
groupshared float s_d8[8];
#if defined(DFLASH_Q5_K)
groupshared int s_q5_scaled[256];
groupshared float s_q5_acc[64];
groupshared float s_q5_summs[8];
#endif

uint load_shared_w_u8(uint offset) {
    return (s_w[offset >> 2u] >> ((offset & 3u) * 8u)) & 255u;
}

#if defined(DFLASH_Q5_K)
float add_q5_f32(float a, float b) {
    return ggml_round_f32((double)a + (double)b);
}

int q8_q5(uint batch_index, uint element) {
    const uint packed = s_q8[batch_index * 64u + (element >> 2u)];
    return int(((packed >> ((element & 3u) * 8u)) & 255u) << 24u) >> 24;
}

uint q5_minimum(uint subblock) {
    if (subblock < 4u) return load_shared_w_u8(8u + subblock) & 63u;
    return (load_shared_w_u8(8u + subblock) >> 4u) |
           ((load_shared_w_u8(4u + subblock) >> 6u) << 4u);
}
#endif

void decode_ints(uint element, out int scale, out int minimum,
                 out int quant) {
    const uint scale_index = element >> 5u;
    if (scale_index < 4u) {
        scale = int(load_shared_w_u8(4u + scale_index) & 63u);
        minimum = int(load_shared_w_u8(8u + scale_index) & 63u);
    } else {
        scale = int((load_shared_w_u8(8u + scale_index) & 15u)
                    | ((load_shared_w_u8(scale_index) >> 6u) << 4u));
        minimum = int((load_shared_w_u8(8u + scale_index) >> 4u)
                      | ((load_shared_w_u8(4u + scale_index) >> 6u) << 4u));
    }
#if defined(DFLASH_Q4_K)
    const uint packed = load_shared_w_u8(16u + (element >> 6u) * 32u
                                       + (element & 31u));
    quant = int((packed >> ((element & 32u) != 0u ? 4u : 0u)) & 15u);
#else
    const uint lane32 = element & 31u;
    const uint packed = load_shared_w_u8(48u + (element >> 6u) * 32u + lane32);
    const uint low4 = (packed >> ((element & 32u) != 0u ? 4u : 0u)) & 15u;
    const uint high1 = (load_shared_w_u8(16u + lane32)
                        >> (element >> 5u)) & 1u;
    quant = int(low4 | (high1 << 4u));
#endif
}

#if defined(DFLASH_Q4_K)
[numthreads(128,1,1)]
#else
[numthreads(256,1,1)]
#endif
void CSDot(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint row = group.x;
    const uint first = group.y * 8u;
    if (row >= n || first >= batch) return;
    const uint wave_size = WaveGetLaneCount();
    const uint wave_id = lane / wave_size;
#if defined(DFLASH_Q4_K)
    const uint wave_count = 128u / wave_size;
#else
    const uint wave_count = 256u / wave_size;
#endif
#if defined(DFLASH_Q4_K)
    precise float sum_dot = 0.0f;
    precise float sum_min = 0.0f;
#else
    float acc = 0.0f;
    float summs = 0.0f;
#endif
    const uint n_batch_in_group = min(8u, batch - first);
    const uint total_q8_vec4 = n_batch_in_group * 16u;
    [loop] for (uint block = 0u; block < nb; ++block) {
        const uint wb = (row * nb + block) * BLOCK_BYTES;
        if (lane < BLOCK_U32X4_COUNT) {
            const uint4 v = W.Load4(wb + lane * 16u);
            const uint base_idx = lane * 4u;
            s_w[base_idx + 0u] = v.x;
            s_w[base_idx + 1u] = v.y;
            s_w[base_idx + 2u] = v.z;
            s_w[base_idx + 3u] = v.w;
        }
        if (lane < n_batch_in_group) {
            s_d8[lane] = asfloat(Q8.Load(((first + lane) * nb + block) * 292u));
        }
        if (lane < total_q8_vec4) {
            const uint b = lane >> 4u;
            const uint q = lane & 15u;
            const uint4 v = Q8.Load4(((first + b) * nb + block) * 292u + 4u + q * 16u);
            const uint dst = b * 64u + q * 4u;
            s_q8[dst + 0u] = v.x;
            s_q8[dst + 1u] = v.y;
            s_q8[dst + 2u] = v.z;
            s_q8[dst + 3u] = v.w;
        }
        GroupMemoryBarrierWithGroupSync();

#if defined(DFLASH_Q4_K)
        int scale0, minimum0, quant0;
        int scale1, minimum1, quant1;
        decode_ints(lane, scale0, minimum0, quant0);
        decode_ints(lane + 128u, scale1, minimum1, quant1);
        const int sc_quant0 = scale0 * quant0;
        const int sc_quant1 = scale1 * quant1;
        [unroll] for (uint b = 0u; b < 8u; ++b) {
            if (first + b < batch) {
                const uint packed0 = s_q8[b * 64u + (lane >> 2u)];
                const uint packed1 = s_q8[b * 64u + 32u + (lane >> 2u)];
                const uint shift = (lane & 3u) * 8u;
                const int q80 = int(((packed0 >> shift) & 255u) << 24u) >> 24;
                const int q81 = int(((packed1 >> shift) & 255u) << 24u) >> 24;
                const int dot0 = WaveActiveSum(sc_quant0 * q80);
                const int corr0 = WaveActiveSum(minimum0 * q80);
                const int dot1 = WaveActiveSum(sc_quant1 * q81);
                const int corr1 = WaveActiveSum(minimum1 * q81);
                if (WaveIsFirstLane()) {
                    partial[(b * 64u + wave_id) * 2u + 0u] = dot0;
                    partial[(b * 64u + wave_id) * 2u + 1u] = corr0;
                    partial[(b * 64u + wave_count + wave_id) * 2u + 0u] = dot1;
                    partial[(b * 64u + wave_count + wave_id) * 2u + 1u] = corr1;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (lane < 8u && first + lane < batch) {
            const uint b = lane;
            const uint scales = s_w[0];
            const float dw = f16tof32(scales & 65535u);
            const float dminw = f16tof32(scales >> 16u);
            const bool gemm_row = first + b < (batch / 4u) * 4u;
            const uint chunks = gemm_row ? 4u : 1u;
            const uint waves_per_chunk = (2u * wave_count) / chunks;
            precise float d = s_d8[b] * dw;
            precise float dmin = s_d8[b] * dminw;
            for (uint chunk = 0u; chunk < chunks; ++chunk) {
                int isum_dot = 0, isum_corr = 0;
                for (uint w = 0u; w < waves_per_chunk; ++w) {
                    const uint wave = chunk * waves_per_chunk + w;
                    isum_dot += partial[(b * 64u + wave) * 2u];
                    isum_corr += partial[(b * 64u + wave) * 2u + 1u];
                }
                sum_dot = ggml_fma_f32(float(isum_dot), d, sum_dot);
                sum_min = ggml_fma_f32(float(isum_corr), dmin, sum_min);
            }
        }
#else
        int scale, minimum, quant;
        decode_ints(lane, scale, minimum, quant);
        s_q5_scaled[lane] = scale * quant;
        GroupMemoryBarrierWithGroupSync();
        if (lane < 64u) {
            const uint b = lane >> 3u;
            const uint avx_lane = lane & 7u;
            if (first + b < batch) {
                int sumi = 0;
                [unroll] for (uint j = 0u; j < 4u; ++j) {
                    [unroll] for (uint p = 0u; p < 4u; ++p) {
                        const uint e = j * 64u + avx_lane * 4u + p;
                        sumi += s_q5_scaled[e] * q8_q5(b, e) +
                                s_q5_scaled[e + 32u] * q8_q5(b, e + 32u);
                    }
                }
                const float d8 = s_d8[b];
                const float dw = f16tof32(s_w[0] & 65535u);
                precise float d = d8 * dw;
                acc = ggml_fma_f32(d, float(sumi), acc);
                if (avx_lane == 0u) {
                    int isum_min = 0;
                    const uint q8base = ((first + b) * nb + block) * 292u + 260u;
                    [unroll] for (uint subblock = 0u; subblock < 8u; ++subblock) {
                        const uint pair = Q8.Load(q8base + 4u * subblock);
                        const int bs0 = int((pair & 65535u) << 16u) >> 16;
                        const int bs1 = int(pair) >> 16;
                        isum_min += int(q5_minimum(subblock)) * (bs0 + bs1);
                    }
                    const float dminw = f16tof32(s_w[0] >> 16u);
                    precise float dmin = -d8 * dminw;
                    const float term = ggml_round_f32((double)dmin * (double)float(isum_min));
                    summs = add_q5_f32(summs, term);
                }
            }
        }
#endif
        GroupMemoryBarrierWithGroupSync();
    }
#if defined(DFLASH_Q5_K)
    if (lane < 64u) {
        s_q5_acc[lane] = acc;
        if ((lane & 7u) == 0u) s_q5_summs[lane >> 3u] = summs;
    }
    GroupMemoryBarrierWithGroupSync();
#endif
    if (lane < 8u && first + lane < batch) {
#if defined(DFLASH_Q4_K)
        precise float result = sum_dot - sum_min;
        Y[(first + lane) * n + row] = result;
#else
        const uint base = lane * 8u;
        const float a0 = add_q5_f32(s_q5_acc[base + 4u], s_q5_acc[base + 0u]);
        const float a1 = add_q5_f32(s_q5_acc[base + 5u], s_q5_acc[base + 1u]);
        const float a2 = add_q5_f32(s_q5_acc[base + 6u], s_q5_acc[base + 2u]);
        const float a3 = add_q5_f32(s_q5_acc[base + 7u], s_q5_acc[base + 3u]);
        const float lo = add_q5_f32(a0, a2);
        const float hi = add_q5_f32(a1, a3);
        Y[(first + lane) * n + row] = add_q5_f32(add_q5_f32(lo, hi),
                                                  s_q5_summs[lane]);
#endif
    }
}
