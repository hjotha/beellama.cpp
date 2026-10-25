// GGML Q6_K W[N,K] times Q8_K X[batch,K] -> F32 Y[batch,N].
// Q6_K: 210 bytes/block. Q8_K: 292 bytes/block. K must divide 256.
// Dispatch(N, ceil(batch/8), 1); one 256-thread group per output row/tile.
// Integer products are summed exactly per QK block, then scaled in F32.
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

groupshared int wave_sum[8 * 8];  // 8 batch items x 8 waves (wave32/64 on SM6)
groupshared uint s_w6[56];        // 56 uint32s = 224 bytes (covers full 210 bytes + up to 3 bytes shift)
groupshared uint s_q8[512];
groupshared float s_d8[8];

uint load_shared_q6_u8(uint offset, uint shift) {
    const uint byte_pos = shift + offset;
    return (s_w6[byte_pos >> 2u] >> ((byte_pos & 3u) * 8u)) & 255u;
}

uint load_shared_q6_u16(uint even_offset, uint shift) {
    const uint byte_pos = shift + even_offset;
    return (s_w6[byte_pos >> 2u] >> ((byte_pos & 2u) * 8u)) & 65535u;
}

int q6_scaled_integer(uint element, uint shift) {
    const uint h = element >> 7u;
    const uint rem = element & 127u;
    const uint quarter = rem >> 5u;
    const uint l = rem & 31u;

    const uint q4_offset = h * 64u + ((quarter & 1u) != 0u ? 32u : 0u) + l;
    const uint q4_byte = load_shared_q6_u8(q4_offset, shift);
    const uint nibble = (quarter >= 2u ? (q4_byte >> 4u) : q4_byte) & 15u;

    const uint qh_offset = 128u + h * 32u + l;
    const uint qh_byte = load_shared_q6_u8(qh_offset, shift);
    const uint qh_bits = (qh_byte >> (quarter * 2u)) & 3u;

    const int quant = int(nibble | (qh_bits << 4u)) - 32;

    const uint scale_idx = element >> 4u;
    const uint scale_byte = load_shared_q6_u8(192u + scale_idx, shift);
    const int scale = int(scale_byte << 24u) >> 24;

    return scale * quant;
}

[numthreads(256, 1, 1)]
void CSDot(uint lane : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row = gid.x;
    const uint batch_start = gid.y * 8u;
    if (row >= n || batch_start >= batch)
        return;

    const uint wave_size = WaveGetLaneCount();
    const uint wave_id = lane / wave_size;
    const uint wave_count = 256u / wave_size;
    const uint n_batch_in_group = min(8u, batch - batch_start);
    const uint total_q8_vec4 = n_batch_in_group * 16u;
    float accum = 0.0f;

    [loop]
    for (uint block = 0u; block < nb; ++block) {
        const uint q6_base = (row * nb + block) * 210u;
        const uint start_u32 = q6_base & ~3u;
        const uint shift = q6_base & 3u;

        if (lane < 14u) {
            const uint4 v = W.Load4(start_u32 + lane * 16u);
            const uint base_idx = lane * 4u;
            s_w6[base_idx + 0u] = v.x;
            s_w6[base_idx + 1u] = v.y;
            s_w6[base_idx + 2u] = v.z;
            s_w6[base_idx + 3u] = v.w;
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
        GroupMemoryBarrierWithGroupSync();

        const int weight = q6_scaled_integer(lane, shift);
        [unroll]
        for (uint b = 0u; b < 8u; ++b) {
            if (batch_start + b < batch) {
                const uint packed = s_q8[b * 64u + (lane >> 2u)];
                const uint byte_value = (packed >> ((lane & 3u) * 8u)) & 255u;
                const int q8 = int(byte_value << 24u) >> 24;
                const int sum_val = WaveActiveSum(weight * q8);
                if (WaveIsFirstLane())
                    wave_sum[b * 8u + wave_id] = sum_val;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (lane < 8u && batch_start + lane < batch) {
            const uint b = lane;
            const float d6 = f16tof32(load_shared_q6_u16(208u, shift));
            int integer_sum = 0;
            for (uint wave = 0u; wave < wave_count; ++wave)
                integer_sum += wave_sum[b * 8u + wave];
            const float d8 = s_d8[b];
            accum += (d8 * d6) * float(integer_sum);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane < 8u && batch_start + lane < batch) {
        Y[(batch_start + lane) * n + row] = accum;
    }
}
