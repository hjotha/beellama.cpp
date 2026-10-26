// GPU ceiling probe: ALU peak and dispatch-chain latency.
//
// Every entry point keeps its operands in registers and stores one value per
// group at the end, so DXC cannot fold the loop away and the measured rate is
// instruction throughput, not a dead-code-eliminated no-op.
//
// Entry points and their shader models (see compile-gpupeak-shaders.py):
//   CSFmaF32      cs_6_0                     8 independent FP32 FMA chains
//   CSFmaF16      cs_6_2 -enable-16bit-types 4 packed half2 FMA chains
//   CSDot2F16     cs_6_4 -enable-16bit-types dot2add (2x F16 MAC -> F32)
//   CSDot4I8      cs_6_4                     dot4add_i8packed (4x I8 MAC -> I32)
//   CSLatencyFma  cs_6_0                     one serial FP32 FMA chain
//   CSCuProbe     cs_6_0                     40 KiB LDS: one group per CU
//
// The dependent-FMA chain is a latency probe, not a throughput probe: the
// implied clock it yields is derived from the documented RDNA2 VALU FMA
// latency and is reported as derived, never as a measured clock.

cbuffer Params : register(b0) {
    uint iters;     // FMA iterations per accumulator chain
    uint groups_x;  // Dispatch X, for the flat group index
    uint groups_y;
    uint seed;      // keeps the inputs data-dependent but reproducible
};

RWStructuredBuffer<uint> Sink : register(u0);

float gpupk_seed(uint index) {
    // 0.5 .. 1.5, deterministic, never 0 or a denormal.
    return 0.5f + float((seed * 1103515245u + index * 12345u) >> 8 & 1023u) * (1.0f / 1024.0f);
}

uint gpupk_flat_group(uint3 gid) {
    return gid.y * groups_x + gid.x;
}

#if defined(GPUPK_ALU_F32)

// 8 independent chains: enough ILP to fill a scalar VALU pipe. Two FLOP per
// FMA (multiply + add) is the accounting used by the report.
[numthreads(64, 1, 1)]
void CSFmaF32(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    const float a = gpupk_seed(dtid.x);
    const float b = gpupk_seed(dtid.x + 64u);
    float acc0 = a, acc1 = b, acc2 = a + b, acc3 = a - b;
    float acc4 = a * b, acc5 = b - a, acc6 = a * 0.5f, acc7 = b * 0.5f;
    [loop]
    for (uint i = 0u; i < iters; ++i) {
        acc0 = mad(acc0, b, a);
        acc1 = mad(acc1, a, b);
        acc2 = mad(acc2, b, a);
        acc3 = mad(acc3, a, b);
        acc4 = mad(acc4, b, a);
        acc5 = mad(acc5, a, b);
        acc6 = mad(acc6, b, a);
        acc7 = mad(acc7, a, b);
    }
    const float folded = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7));
    Sink[gpupk_flat_group(gid) * 64u + dtid.x] = asuint(folded);
}

#elif defined(GPUPK_ALU_F16)

// Packed half2: one instruction carries two F16 FMAs. Four chains keep the
// packed pipe busy without spilling.
[numthreads(64, 1, 1)]
void CSFmaF16(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    const float16_t2 a = float16_t2(gpupk_seed(dtid.x), gpupk_seed(dtid.x + 1u));
    const float16_t2 b = float16_t2(gpupk_seed(dtid.x + 2u), gpupk_seed(dtid.x + 3u));
    float16_t2 acc0 = a, acc1 = b, acc2 = a + b, acc3 = a - b;
    [loop]
    for (uint i = 0u; i < iters; ++i) {
        acc0 = acc0 * b + a;
        acc1 = acc1 * a + b;
        acc2 = acc2 * b + a;
        acc3 = acc3 * a + b;
    }
    const float16_t2 folded = (acc0 + acc1) + (acc2 + acc3);
    Sink[gpupk_flat_group(gid) * 64u + dtid.x] =
        asuint(float(folded.x)) ^ asuint(float(folded.y));
}

#elif defined(GPUPK_ALU_DOT2)

// dot2add: 2 F16 multiplies accumulated into one F32. This is the instruction
// the KVarN kernels would use for a packed-score variant.
[numthreads(64, 1, 1)]
void CSDot2F16(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    const float16_t2 a = float16_t2(gpupk_seed(dtid.x), gpupk_seed(dtid.x + 1u));
    const float16_t2 b = float16_t2(gpupk_seed(dtid.x + 2u), gpupk_seed(dtid.x + 3u));
    float acc0 = gpupk_seed(dtid.x) * 0.01f;
    float acc1 = gpupk_seed(dtid.x + 5u) * 0.01f;
    float acc2 = gpupk_seed(dtid.x + 6u) * 0.01f;
    float acc3 = gpupk_seed(dtid.x + 7u) * 0.01f;
    [loop]
    for (uint i = 0u; i < iters; ++i) {
        acc0 = dot2add(a, b, acc0);
        acc1 = dot2add(b, a, acc1);
        acc2 = dot2add(a, a, acc2);
        acc3 = dot2add(b, b, acc3);
    }
    const float folded = (acc0 + acc1) + (acc2 + acc3);
    Sink[gpupk_flat_group(gid) * 64u + dtid.x] = asuint(folded);
}

#elif defined(GPUPK_ALU_DOT4)

// dot4add_i8packed: 4 signed byte multiplies accumulated into one I32.
[numthreads(64, 1, 1)]
void CSDot4I8(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    const uint a = 0x01020304u + seed;
    const uint b = 0x04030201u + dtid.x;
    int acc0 = int(dtid.x);
    int acc1 = acc0 + 1;
    int acc2 = acc0 + 2;
    int acc3 = acc0 + 3;
    [loop]
    for (uint i = 0u; i < iters; ++i) {
        acc0 = dot4add_i8packed(a, b, acc0);
        acc1 = dot4add_i8packed(b, a, acc1);
        acc2 = dot4add_i8packed(a, a, acc2);
        acc3 = dot4add_i8packed(b, b, acc3);
    }
    Sink[gpupk_flat_group(gid) * 64u + dtid.x] = asuint(acc0 ^ acc1 ^ acc2 ^ acc3);
}

#elif defined(GPUPK_LATENCY_FMA)

// One serial dependency chain: the wall time per iteration is the FMA latency,
// not the throughput. One group, 64 threads, so the Sink index never overlaps.
[numthreads(64, 1, 1)]
void CSLatencyFma(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    float acc = gpupk_seed(dtid.x) * 1e-3f;
    const float m = 1.0f - 1.0f / 1048576.0f;
    const float c = 1.0f / 1048576.0f;
    [loop]
    for (uint i = 0u; i < iters; ++i) {
        acc = mad(acc, m, c);
    }
    Sink[gpupk_flat_group(gid) * 64u + dtid.x] = asuint(acc);
}

#elif defined(GPUPK_CU_PROBE)

// LDS-limited occupancy probe. D3D12 caps a workgroup at 32 KiB of LDS, so the
// largest usable allocation is 32 KiB: with a 64 KiB CU LDS that admits two
// groups per CU, and the 16 KiB variant admits four. Total time versus group
// count steps at CUs x groups-per-CU for both, which gives two independent
// estimates of the same CU count (see tools/xbox/tests/rkva_ceiling_roofline.py).
#if defined(GPUPK_CU_LDS_16)
groupshared float4 gpupk_lds_pad[1024]; // 16 KiB
#else
groupshared float4 gpupk_lds_pad[2048]; // 32 KiB
#endif

[numthreads(64, 1, 1)]
void CSCuProbe(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    const uint flat = gpupk_flat_group(gid);
    gpupk_lds_pad[dtid.x] = float4(float(flat), float(dtid.x), 0.0f, 0.0f);
    GroupMemoryBarrierWithGroupSync();
    float acc = gpupk_lds_pad[dtid.x].x * 1e-6f;
    const float b = gpupk_seed(dtid.x);
    [loop]
    for (uint i = 0u; i < iters; ++i) {
        acc = mad(acc, b, 1e-6f);
    }
    Sink[flat * 64u + dtid.x] = asuint(acc + gpupk_lds_pad[dtid.x].y);
}

#endif
