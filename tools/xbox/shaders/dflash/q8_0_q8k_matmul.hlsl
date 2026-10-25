// GGML Q8_0 W[N,K] times Q8_K X[batch,K] -> F32 Y[batch,N].
// Q8_0: 272 bytes per 256 weights (8 sub-blocks of 34 bytes).
// Q8_K: 292 bytes per 256 weights. K must divide 256.
// Dispatch(N, ceil(batch/8), 1); one 256-thread group per output row/tile.
// Integer products are summed per 32-element group, scaled by FP16 d_w and FP32 d_8.
// Requires SM6 WaveOps; CPU path remains the fallback if unavailable.

cbuffer Params : register(b0) {
    uint n;
    uint k_dim;
    uint nb;
    uint batch;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer X : register(t1);
RWStructuredBuffer<float> Y : register(u0);

groupshared int s_group_sum[8 * 8]; // 8 batch items x 8 groups of 32
groupshared uint s_w8[68];         // 68 uint32s = 272 bytes (exactly 8 * 34 bytes)
groupshared uint s_q8[512];
groupshared float s_d8[8];

uint load_shared_w8_u8(uint offset) {
    return (s_w8[offset >> 2u] >> ((offset & 3u) * 8u)) & 255u;
}

uint load_shared_w8_u16(uint even_offset) {
    return (s_w8[even_offset >> 2u] >> ((even_offset & 2u) * 8u)) & 65535u;
}

[numthreads(256, 1, 1)]
void CSDot(uint lane : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row = gid.x;
    const uint batch_start = gid.y * 8u;
    if (row >= n || batch_start >= batch)
        return;

    const uint wave_size = WaveGetLaneCount();
    const uint wave_id = lane / wave_size;
    const uint n_batch_in_group = min(8u, batch - batch_start);
    const uint total_q8_vec4 = n_batch_in_group * 16u;
    float accum = 0.0f;

    // Each lane corresponds to one element in the 256-element block:
    const uint group_idx = lane >> 5u; // 0..7 (group of 32 elements)
    const uint group_lane = lane & 31u; // 0..31
    // Byte offset of quant in s_w8: group_idx * 34 + 2 + group_lane
    const uint quant_byte_pos = group_idx * 34u + 2u + group_lane;

    [loop]
    for (uint block = 0u; block < nb; ++block) {
        const uint q8_base = (row * nb + block) * 272u; // 272 is a multiple of 16

        if (lane < 17u) {
            const uint4 v = W.Load4(q8_base + lane * 16u);
            const uint base_idx = lane * 4u;
            s_w8[base_idx + 0u] = v.x;
            s_w8[base_idx + 1u] = v.y;
            s_w8[base_idx + 2u] = v.z;
            s_w8[base_idx + 3u] = v.w;
        }
        if (lane < n_batch_in_group) {
            s_d8[lane] = asfloat(X.Load(((batch_start + lane) * nb + block) * 292u));
        }
        if (lane < total_q8_vec4) {
            const uint b = lane >> 4u;
            const uint q = lane & 15u;
            const uint4 v = X.Load4(((batch_start + b) * nb + block) * 292u + 4u + q * 16u);
            const uint dst = b * 64u + q * 4u;
            s_q8[dst + 0u] = v.x;
            s_q8[dst + 1u] = v.y;
            s_q8[dst + 2u] = v.z;
            s_q8[dst + 3u] = v.w;
        }

        // Initialize group sums for this block
        if (lane < 64u) {
            s_group_sum[lane] = 0;
        }
        GroupMemoryBarrierWithGroupSync();

        const uint byte_val = load_shared_w8_u8(quant_byte_pos);
        const int quant = int(byte_val << 24u) >> 24;

        [unroll]
        for (uint b = 0u; b < 8u; ++b) {
            if (batch_start + b < batch) {
                const uint packed = s_q8[b * 64u + (lane >> 2u)];
                const uint byte_value = (packed >> ((lane & 3u) * 8u)) & 255u;
                const int q8 = int(byte_value << 24u) >> 24;
                if (wave_size == 32u) {
                    const int sum_val = WaveActiveSum(quant * q8);
                    if (WaveIsFirstLane())
                        s_group_sum[b * 8u + wave_id] = sum_val;
                } else {
                    InterlockedAdd(s_group_sum[b * 8u + group_idx], quant * q8);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (lane < 8u && batch_start + lane < batch) {
            const uint b = lane;
            float block_sum = 0.0f;
            for (uint g = 0u; g < 8u; ++g) {
                const float dw = f16tof32(load_shared_w8_u16(g * 34u));
                block_sum += dw * float(s_group_sum[b * 8u + g]);
            }
            const float d8 = s_d8[b];
            accum += d8 * block_sum;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane < 8u && batch_start + lane < batch) {
        Y[(batch_start + lane) * n + row] = accum;
    }
}
