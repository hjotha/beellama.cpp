// GGML Q2_K W[N,K] times Q8_K X[batch,K] -> F32 Y[batch,N].
// Q2_K: 84 bytes/block. Q8_K: 292 bytes/block. K must divide 256.
// Dispatch(N,ceil(batch/8),1); one 64-thread group per output row/tile.
// Match GGML's x86 AVX2 path: eight FMA accumulators and its final hsum.
#include "ggml_float.hlsli"

cbuffer Params : register(b0) {
    uint n;
    uint k_dim;
    uint nb;
    uint batch;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer X : register(t1);
RWStructuredBuffer<float> Y : register(u0);

groupshared float lane_sum[8 * 8]; // batch item x AVX2 accumulator
groupshared uint s_scaled_w[64];   // 64 uint32s = 256 bytes (4 bytes per uint)
groupshared uint s_mins[4];        // 4 uint32s = 16 bytes
groupshared uint s_q8[512];
groupshared float s_d8[8];
groupshared float s_q2_d;
groupshared float s_q2_dmin;

[numthreads(64, 1, 1)]
void CSDot(uint lane : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row = gid.x;
    const uint batch_start = gid.y * 8u;
    if (row >= n || batch_start >= batch)
        return; // group-uniform exit before any barrier

    const uint n_batch_in_group = min(8u, batch - batch_start);
    const uint total_q8_vec4 = n_batch_in_group * 16u;
    const uint b = lane / 8u;
    const uint avx_lane = lane & 7u;
    precise float accum = 0.0f;

    [loop]
    for (uint block = 0u; block < nb; ++block) {
        const uint q2_base = (row * nb + block) * 288u;

        // All 64 threads cooperate to load the 64 DWORDs of scaled weights in parallel
        s_scaled_w[lane] = W.Load(q2_base + lane * 4u);

        if (lane < 4u) {
            s_mins[lane] = W.Load(q2_base + 256u + lane * 4u);
        } else if (lane == 4u) {
            const uint d_packed = W.Load(q2_base + 272u);
            s_q2_d = f16tof32(d_packed & 0xffffu);
            s_q2_dmin = f16tof32(d_packed >> 16u);
        }

        if (lane < n_batch_in_group) {
            s_d8[lane] = asfloat(X.Load(((batch_start + lane) * nb + block) * 292u));
        }
        for (uint load = lane; load < total_q8_vec4; load += 64u) {
            const uint b_idx = load >> 4u;
            const uint q = load & 15u;
            const uint4 v = X.Load4(((batch_start + b_idx) * nb + block) * 292u + 4u + q * 16u);
            const uint dst = b_idx * 64u + q * 4u;
            s_q8[dst + 0u] = v.x;
            s_q8[dst + 1u] = v.y;
            s_q8[dst + 2u] = v.z;
            s_q8[dst + 3u] = v.w;
        }
        GroupMemoryBarrierWithGroupSync();

        if (b < n_batch_in_group) {
            int integer_sum = 0;
            [unroll] for (uint segment = 0u; segment < 8u; ++segment) {
                const uint first_u32 = segment * 8u + avx_lane;
                const uint w_packed = s_scaled_w[first_u32];
                const uint packed = s_q8[b * 64u + first_u32];
                [unroll] for (uint j = 0u; j < 4u; ++j) {
                    const int w = int((w_packed >> (j * 8u)) & 255u);
                    const int q8 = int((packed >> (j * 8u)) << 24u) >> 24;
                    integer_sum += w * q8;
                }
            }
            const uint sums = X.Load(((batch_start + b) * nb + block) * 292u +
                                     260u + avx_lane * 4u);
            const int sum0 = int(sums << 16u) >> 16;
            const int sum1 = int(sums) >> 16;
            const uint min_word = s_mins[avx_lane >> 1u];
            const uint shift = (avx_lane & 1u) * 16u;
            const int min0 = int((min_word >> shift) & 255u);
            const int min1 = int((min_word >> (shift + 8u)) & 255u);
            const int minimum = sum0 * min0 + sum1 * min1;

            const float d8 = s_d8[b];
            precise float d = d8 * s_q2_d;
            precise float dmin = -d8 * s_q2_dmin;
            // HLSL mad splits the rounding on both Xbox and WARP. F64 keeps
            // the product until the single F32 rounding required by AVX2 FMA.
            accum = ggml_fma_f32(dmin, float(minimum), accum);
            accum = ggml_fma_f32(d, float(integer_sum), accum);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (b < n_batch_in_group) lane_sum[lane] = accum;
    GroupMemoryBarrierWithGroupSync();
    if (b < n_batch_in_group && avx_lane == 0u) {
        precise float a = lane_sum[lane] + lane_sum[lane + 4u];
        precise float c = lane_sum[lane + 1u] + lane_sum[lane + 5u];
        precise float e = lane_sum[lane + 2u] + lane_sum[lane + 6u];
        precise float g = lane_sum[lane + 3u] + lane_sum[lane + 7u];
        precise float result = (a + e) + (c + g);
        Y[(batch_start + b) * n + row] = result;
    }
}
