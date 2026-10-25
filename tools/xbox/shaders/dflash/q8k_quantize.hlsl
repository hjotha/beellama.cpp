// GGML quantize_row_q8_K_ref for a resident F32 activation.
// Input X[batch,K] is t1, output Q8[batch,K/256] is a raw u0 buffer.
// block_q8_K = float d + int8 qs[256] + int16 bsums[16] = 292 bytes.
// Dispatch(K/256,batch,1). Requires SM6 WaveOps; Xbox Series X has WaveOps.

#if defined(DFLASH_Q8_EXACT_SCALE_FP64)
// D3D12 F32 division can differ from ggml's round-to-nearest-even by one ULP
// even with HLSL precise. SplitDouble prevents DXC from narrowing the divide
// back to F32; round the normal F64 quotient to F32 explicitly.
float ggml_divide_rne(float numerator, float denominator) {
    precise double quotient = (double)numerator / (double)denominator;
    uint low, high;
    asuint(quotient, low, high);
    int exponent = int((high >> 20u) & 2047u) - 896;
    if (exponent < 1 || exponent > 254)
        return (float)quotient;
    uint mantissa = ((high & 0xfffffu) << 3u) | (low >> 29u);
    const uint remainder = low & 0x1fffffffu;
    if (remainder > 0x10000000u ||
        (remainder == 0x10000000u && (mantissa & 1u) != 0u))
        ++mantissa;
    if (mantissa == 0x800000u) {
        mantissa = 0u;
        ++exponent;
    }
    return asfloat((high & 0x80000000u) | (uint(exponent) << 23u) | mantissa);
}
#endif

cbuffer Params : register(b0) {
    uint n;
    uint k_dim;
    uint nb;
    uint batch;
};

ByteAddressBuffer X : register(t1);
RWByteAddressBuffer Q8 : register(u0);

groupshared float block_max_abs;
groupshared float block_iscale;
groupshared float wave_max_abs[64];
groupshared uint wave_first_index[64];
groupshared int quantized[256];

[numthreads(256, 1, 1)]
void CSQuantize(uint lane : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint block = gid.x;
    const uint vector_index = gid.y;
    if (block >= nb || vector_index >= batch)
        return; // group-uniform exit before any barrier

    const uint column = block * 256u + lane;
    const float value = asfloat(X.Load((vector_index * k_dim + column) * 4u));
    const uint wave_size = WaveGetLaneCount();
    const uint wave_id = lane / wave_size;
    const uint wave_count = 256u / wave_size;
    const float wave_max = WaveActiveMax(abs(value));
    if (WaveIsFirstLane())
        wave_max_abs[wave_id] = wave_max;
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0u) {
        float amax = 0.0f;
        for (uint i = 0u; i < wave_count; ++i)
            amax = max(amax, wave_max_abs[i]);
        block_max_abs = amax;
    }
    GroupMemoryBarrierWithGroupSync();

    // GGML uses the first signed value among equal-absolute maxima.
    const uint candidate = abs(value) == block_max_abs ? lane : 0xffffffffu;
    const uint wave_first = WaveActiveMin(candidate);
    if (WaveIsFirstLane())
        wave_first_index[wave_id] = wave_first;
    GroupMemoryBarrierWithGroupSync();

    const uint q8_base = (vector_index * nb + block) * 292u;
    if (lane == 0u) {
        uint first = 0xffffffffu;
        for (uint i = 0u; i < wave_count; ++i)
            first = min(first, wave_first_index[i]);
        const float signed_max = asfloat(X.Load((vector_index * k_dim + block * 256u + first) * 4u));
        // GGML computes both divisions in F32. Keep DXC from using fast division:
        // one ULP in d changes the packed Q8_K scale bytes.
#if defined(DFLASH_Q8_EXACT_SCALE_FP64)
        precise float iscale = block_max_abs == 0.0f ? 0.0f :
            ggml_divide_rne(-127.0f, signed_max);
#else
        precise float iscale = block_max_abs == 0.0f ? 0.0f : -127.0f / signed_max;
#endif
        block_iscale = iscale;
#if defined(DFLASH_Q8_EXACT_SCALE_FP64)
        precise float d = block_iscale == 0.0f ? 0.0f :
            ggml_divide_rne(1.0f, block_iscale);
#else
        precise float d = block_iscale == 0.0f ? 0.0f : 1.0f / block_iscale;
#endif
        Q8.Store(q8_base, asuint(d));
    }
    GroupMemoryBarrierWithGroupSync();

    // GGML nearest_int rounds ties to even; DXIL round_ne has that rule.
    const int q = block_iscale == 0.0f ? 0 : min(127, int(round(block_iscale * value)));
    quantized[lane] = q;
    GroupMemoryBarrierWithGroupSync();

    if (lane < 64u) {
        const uint start = lane * 4u;
        const uint packed = (uint(quantized[start + 0u]) & 255u)
                          | ((uint(quantized[start + 1u]) & 255u) << 8u)
                          | ((uint(quantized[start + 2u]) & 255u) << 16u)
                          | ((uint(quantized[start + 3u]) & 255u) << 24u);
        Q8.Store(q8_base + 4u + lane * 4u, packed);
    }
    if (lane < 8u) {
        int sum0 = 0;
        int sum1 = 0;
        [unroll]
        for (uint i = 0u; i < 16u; ++i) {
            sum0 += quantized[lane * 32u + i];
            sum1 += quantized[lane * 32u + 16u + i];
        }
        Q8.Store(q8_base + 260u + lane * 4u,
                 (uint(sum0) & 65535u) | ((uint(sum1) & 65535u) << 16u));
    }
}
