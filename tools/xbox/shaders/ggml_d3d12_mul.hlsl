// ggml_d3d12_mul.hlsl — plan 007, bounded chain element 2/2.
// Elementwise float multiply, 1-D weight broadcast along ne[0]: this is the
// norm-weight form the graphs actually produce (weight ne[0] == dst ne[0],
// ne[1..3] == 1, so the weight index repeats every ne0 elements). Float
// multiply is exact per IEEE-754 element, so this kernel is bit-exact with the
// CPU elementwise product. GPU-produced scalars are NOT supported here: the
// host predicate only accepts this broadcast form with owned buffers, and the
// weight is bound as an SRV (t0) like the matmul path binds weights, never as
// a UAV.
//
// Root (matmul layout): b0 constants {count, ne0, pad, pad}; t0 = weight,
// u0 = dst, u1 = src. Dispatch (count/256, 1, 1); the host predicate keeps
// count/256 <= 65535.

StructuredBuffer<float> wt : register(t0);
RWStructuredBuffer<float> src : register(u1);
RWStructuredBuffer<float> dst : register(u0);

cbuffer rc : register(b0) {
    uint count;
    uint ne0;
    uint pad0;
    uint pad1;
};

[numthreads(256, 1, 1)] void main(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= count)
        return;
    dst[i] = src[i] * wt[i % ne0];
}