// DFlash2 DRAFT scalar operators. All rows are contiguous F32 in GGML order.
// The quantized matmuls, dynamic convolution, attention, and KV pack are
// separate shaders so that no intermediate hidden row is read back to CPU.
cbuffer Params : register(b0) {
    uint rows;
    uint width;
    uint head_width;
    uint rope_cache_offset;
    float scale;
    float epsilon;
    float rope_base;
    float reserved1;
};

ByteAddressBuffer X : register(t0);
ByteAddressBuffer W : register(t1);
ByteAddressBuffer Positions : register(t2);
RWByteAddressBuffer Y : register(u0);

#include "ggml_float.hlsli"

groupshared double partial[128];

// Dispatch(rows,1,1). Supports hidden norm (5120) and per-head Q/K norm (128).
[numthreads(128,1,1)]
void RmsNorm(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint row = group.x;
    if (row >= rows || width == 0u) return;
    precise double ss = 0.0;
    for (uint col = lane; col < width; col += 128u) {
        precise float value = asfloat(X.Load((row * width + col) * 4u)) * scale;
        precise float square = value * value;
        ss += (double)square;
    }
    partial[lane] = ss;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 64u; stride != 0u; stride >>= 1u) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    precise float inv = ggml_rms_factor(partial[0], width, epsilon);
    for (uint col = lane; col < width; col += 128u) {
        const float value = asfloat(X.Load((row * width + col) * 4u)) * scale;
        precise float result = value * inv * asfloat(W.Load((col % head_width) * 4u));
        Y.Store((row * width + col) * 4u, asuint(result));
    }
}

// Dispatch(heads, tokens, 1). DFlash temporal M-RoPE sections [64,0,0,0]
// pair (d,d+64), not adjacent pairs. Width is heads*128.
[numthreads(64,1,1)]
void Rope128(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint head = group.x;
    const uint token = group.y;
    if (token >= rows || head * 128u >= width) return;
    const uint index = token * width + head * 128u + lane;
    const float a = asfloat(X.Load(index * 4u));
    const float b = asfloat(X.Load((index + 64u) * 4u));
    const uint offset = rope_cache_offset + (token * 128u + lane * 2u) * 4u;
    const float cs = asfloat(Positions.Load(offset));
    const float sn = asfloat(Positions.Load(offset + 4u));
    precise float first = a * cs - b * sn;
    precise float second = a * sn + b * cs;
    Y.Store(index * 4u, asuint(first));
    Y.Store((index + 64u) * 4u, asuint(second));
}

// Y=X*scale+W. A zero W resource supplies the no-residual variant. The
// caller must never bind X or W as Y in the same dispatch.
[numthreads(128,1,1)]
void ScaleAdd(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= rows * width) return;
    precise float result = asfloat(X.Load(i * 4u)) * scale + asfloat(W.Load(i * 4u));
    Y.Store(i * 4u, asuint(result));
}

// Gate and up use [token,17408]. Dispatch ceil(rows*width/128).
[numthreads(128,1,1)]
void SwiGlu(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= rows * width) return;
    const float gate = asfloat(X.Load(i * 4u));
    const float up = asfloat(W.Load(i * 4u));
    precise float denominator = 1.0f + ggml_exp_avx2(-gate);
    precise float silu = ggml_round_f32((double)gate / (double)denominator);
    precise float result = silu * up;
    Y.Store(i * 4u, asuint(result));
}
