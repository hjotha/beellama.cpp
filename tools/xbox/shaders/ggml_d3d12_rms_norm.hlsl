// ggml_d3d12_rms_norm.hlsl — plan 007, bounded chain element 1/2.
// ggml_compute_forward_rms_norm_f32 (llama.cpp ggml-cpu/ops.cpp) accumulates
//     ggml_float sum = 0.0;                 // double
//     for (i = 0; i < ne00; ++i) sum += (double)(x[i] * x[i]);
//     const float mean  = (float)(sum / ne00);
//     const float scale = 1.0f / sqrtf(mean + eps);
//     y[i] = x[i] * scale;
// in STRICTLY ASCENDING index order, then rounds once to float. This kernel
// keeps double lanes and a double tree, then rounds the same way, and uses a
// precise float divide/sqrt (build with -Gis) instead of rsqrt. The reduction
// order still differs from ascending, so bitwise equality is MEASURED (plan 007
// numerics reference), not assumed.
//
// Root (matmul layout): b0 constants {ne00, rows, bits(eps), pad}; u0 = dst,
// u1 = src. Dispatch (rows, 1, 1) — one group per row, <= 65535 rows gated by
// the host predicate; thread x walks its row in strides of 256.

RWStructuredBuffer<float> src : register(u1);
RWStructuredBuffer<float> dst : register(u0);

cbuffer rc : register(b0) {
    uint  ne00;
    uint  rows;
    float eps_f;
    uint  pad0;
};

groupshared double partial[256];

[numthreads(256, 1, 1)] void main(uint3 gtid : SV_GroupThreadID,
                                 uint3 gid : SV_GroupID) {
    const uint row = gid.x;           // one group per row (dispatch is 1-D)
    const uint lane = gtid.x;
    double acc = 0.0;
    const uint base = row * ne00;
    for (uint i = lane; i < ne00; i += 256u) {
        const float v = src[base + i];
        acc += (double)(v * v);
    }
    partial[lane] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (lane < s)
            partial[lane] += partial[lane + s];
        GroupMemoryBarrierWithGroupSync();
    }
    // Broadcast the float scale through the shared slot.
    if (lane == 0u) {
        const double sum = partial[0];
        const float mean = (float)(sum / (double)ne00);
        const float scale = 1.0f / sqrt(mean + eps_f);
        partial[0] = (double)scale;
    }
    GroupMemoryBarrierWithGroupSync();
    const float scale = (float)partial[0];
    for (uint i = lane; i < ne00; i += 256u)
        dst[base + i] = src[base + i] * scale;
}