// GGML Q3_K W[N,K] times Q8_K X[batch,K] -> F32 Y[batch,N].
// Q3_K: 110 bytes/block. Q8_K: 292 bytes/block. K must divide 256.
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

groupshared float lane_sum[8 * 8];
groupshared uint s_w3[28];        // 28 uint32s = 112 bytes (covers full 110 bytes of Q3_K block)
groupshared uint s_q8[512];
groupshared float s_d8[8];
groupshared float s_q3_d;
groupshared int scaled_weights[256];

uint load_shared_q3_u8(uint offset, uint shift) {
    const uint byte_pos = shift + offset;
    return (s_w3[byte_pos >> 2u] >> ((byte_pos & 3u) * 8u)) & 255u;
}

uint load_shared_q3_u16(uint even_offset, uint shift) {
    const uint byte_pos = shift + even_offset;
    return (s_w3[byte_pos >> 2u] >> ((byte_pos & 2u) * 8u)) & 65535u;
}

int q3_scaled_integer(uint element, uint shift) {
    const uint half = element >> 7u;
    const uint quarter = (element >> 5u) & 3u;
    const uint lane32 = element & 31u;
    const uint low2 = (load_shared_q3_u8(32u + half * 32u + lane32, shift)
                       >> (quarter * 2u)) & 3u;
    const uint high_bit = (load_shared_q3_u8(lane32, shift)
                           >> (half * 4u + quarter)) & 1u;
    const int quant = int(low2) - (high_bit == 0u ? 4 : 0);

    const uint scale_index = element >> 4u;
    const uint scale_byte = load_shared_q3_u8(96u + (scale_index & 7u), shift);
    const uint high_byte = load_shared_q3_u8(104u + (scale_index & 3u), shift);
    const uint low4 = (scale_byte >> (scale_index >= 8u ? 4u : 0u)) & 15u;
    const uint high2 = (high_byte >> (2u * (scale_index >> 2u))) & 3u;
    const int scale = int(low4 | (high2 << 4u)) - 32;
    return scale * quant;
}

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
        const uint q3_base = (row * nb + block) * 110u;
        const uint start_u32 = q3_base & ~3u;
        const uint shift = q3_base & 3u;

        if (lane < 7u) {
            const uint4 v = W.Load4(start_u32 + lane * 16u);
            const uint base_idx = lane * 4u;
            s_w3[base_idx + 0u] = v.x;
            s_w3[base_idx + 1u] = v.y;
            s_w3[base_idx + 2u] = v.z;
            s_w3[base_idx + 3u] = v.w;
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

        if (lane == 0u) {
            s_q3_d = f16tof32(load_shared_q3_u16(108u, shift));
        }

        [unroll] for (uint part = 0u; part < 4u; ++part) {
            const uint element = lane + part * 64u;
            scaled_weights[element] = q3_scaled_integer(element, shift);
        }
        GroupMemoryBarrierWithGroupSync();

        if (b < n_batch_in_group) {
            int integer_sum = 0;
            [unroll] for (uint segment = 0u; segment < 8u; ++segment) {
                const uint first = segment * 32u + avx_lane * 4u;
                const uint packed = s_q8[b * 64u + first / 4u];
                const int q0 = int(packed << 24u) >> 24;
                const int q1 = int((packed >> 8u) << 24u) >> 24;
                const int q2 = int((packed >> 16u) << 24u) >> 24;
                const int q3 = int(packed) >> 24;
                integer_sum += scaled_weights[first + 0u] * q0 +
                               scaled_weights[first + 1u] * q1 +
                               scaled_weights[first + 2u] * q2 +
                               scaled_weights[first + 3u] * q3;
            }
            precise float d = s_d8[b] * s_q3_d;
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
