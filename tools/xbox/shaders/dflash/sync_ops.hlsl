// DFlash2 SYNC-only operations. All inputs/outputs are F32, packed row-major.
// Compile entry points FcNorm, KeyNormRope, ScaleRows and F32Matmul as cs_6_0.
cbuffer Params : register(b0) {
    uint batch;
    uint width;
    uint position_unused;
    float scale;
    float epsilon;
    float rope_base;
    float rope_freq_scale;
    uint rope_cache_offset;
};

ByteAddressBuffer X : register(t0);
ByteAddressBuffer W : register(t1);
ByteAddressBuffer Positions : register(t2);
RWByteAddressBuffer Y : register(u0);
groupshared float reduction[128];
#include "ggml_float.hlsli"
groupshared double norm_reduction[128];

[numthreads(128, 1, 1)]
void FcNorm(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint row = group.x;
    if (row >= batch) return;
    precise double sum = 0.0;
    for (uint col = lane; col < width; col += 128u) {
        precise float v = asfloat(X.Load((row * width + col) * 4u)) * scale;
        precise float square = v * v;
        sum += (double)square;
    }
    norm_reduction[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (lane < stride) norm_reduction[lane] += norm_reduction[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    precise float factor = ggml_rms_factor(norm_reduction[0], width, epsilon);
    for (uint col = lane; col < width; col += 128u) {
        const float v = asfloat(X.Load((row * width + col) * 4u)) * scale;
        const float w = asfloat(W.Load(col * 4u));
        precise float result = v * factor * w;
        Y.Store((row * width + col) * 4u, asuint(result));
    }
}

// GGML M-RoPE sections [64,0,0,0] reduce to temporal NEOX pairing:
// (i, i+64), theta=position*base^(-2*i/128), 0<=i<64.
[numthreads(128, 1, 1)]
void KeyNormRope(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint head = group.x;
    const uint row = group.y;
    if (head >= 8u || row >= batch) return;
    const uint base_index = row * 1024u + head * 128u;
    precise float v = asfloat(X.Load((base_index + lane) * 4u)) * scale;
    precise float square = v * v;
    norm_reduction[lane] = (double)square;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (lane < stride) norm_reduction[lane] += norm_reduction[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane < 64u) {
        precise float factor = ggml_rms_factor(norm_reduction[0], 128u, epsilon);
        precise float a = asfloat(X.Load((base_index + lane) * 4u)) * scale * factor
                      * asfloat(W.Load(lane * 4u));
        precise float b = asfloat(X.Load((base_index + lane + 64u) * 4u)) * scale * factor
                      * asfloat(W.Load((lane + 64u) * 4u));
        const uint offset = rope_cache_offset + (row * 128u + lane * 2u) * 4u;
        const float c = asfloat(Positions.Load(offset));
        const float s = asfloat(Positions.Load(offset + 4u));
        precise float first = a * c - b * s;
        precise float second = a * s + b * c;
        Y.Store((base_index + lane) * 4u, asuint(first));
        Y.Store((base_index + lane + 64u) * 4u, asuint(second));
    }
}

[numthreads(128, 1, 1)]
void ScaleRows(uint3 tid : SV_DispatchThreadID) {
    const uint index = tid.x;
    if (index >= batch * width) return;
    Y.Store(index * 4u, asuint(asfloat(X.Load(index * 4u)) * scale));
}

// Optional F32 projection. n=width, k=position_unused; W is [n,k].
// Kept separate from quant kernels; no conversion or requantization of GGUF.
[numthreads(128, 1, 1)]
void F32Matmul(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint output_row = group.x;
    const uint token = group.y;
    if (output_row >= width || token >= batch) return;
    const uint k = position_unused;
    float sum = 0.0f;
    for (uint col = lane; col < k; col += 128u) {
        sum += asfloat(W.Load((output_row * k + col) * 4u))
             * asfloat(X.Load((token * k + col) * 4u));
    }
    reduction[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) Y.Store((token * width + output_row) * 4u, asuint(reduction[0]));
}
