// ggml_d3d12_add.hlsl — plan 008 complete island: residual ADD.
// Same-shape f32 elementwise add (the only form the graphs produce here);
// bit-exact per IEEE-754 element, order-independent. Host predicate gates
// shape/contiguity and keeps everything else on the CPU kernel.
//
// Root (GDN layout): b0 constants {count, pad, pad, pad}; u0 = dst,
// u1 = src0, u2 = src1. Dispatch (count/256, 1, 1).

RWStructuredBuffer<float> dst : register(u0);
RWStructuredBuffer<float> s0 : register(u1);
RWStructuredBuffer<float> s1 : register(u2);

cbuffer rc : register(b0) {
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

[numthreads(256, 1, 1)] void main(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= count)
        return;
    dst[i] = s0[i] + s1[i];
}