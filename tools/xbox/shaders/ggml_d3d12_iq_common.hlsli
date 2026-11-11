// ggml backend d3d12 — shared by the IQ-quant matmul kernels (IQ2_S,
// IQ3_XXS, IQ3_S, IQ4_XS).
//
// The grids/codebooks live in one read-only ByteAddressBuffer bound as root SRV
// t1, uploaded once at backend init from the checked-in blob produced by
// scripts/gen-d3d12-iq-tables.py. They are deliberately NOT HLSL static const
// arrays: the IQ2_S grid alone is 8 KiB and every lookup index is lane
// divergent, which constant memory serialises per unique address.
//
// Offsets must match kD3d12Iq2sGrid/kD3d12Iq3xxsGrid/kD3d12Iq3sGrid/
// kD3d12IqSigns/kD3d12Iq4nl in shaders/generated/ggml_d3d12_iq_tables.h.
ByteAddressBuffer T : register(t1);

#define IQ2S_GRID_BASE 0u     // 1024 x uint64 (8 bytes, little-endian)
#define IQ3XXS_GRID_BASE 8192u // 256 x uint32 (4 bytes)
#define IQ3S_GRID_BASE 9216u  // 512 x uint32 (4 bytes)
#define IQ_SIGNS_BASE 11264u  // 128 x uint8
#define IQ4NL_BASE 11392u     // 16 x float

// The sign table is byte-addressed; every other table index lands on a dword.
uint iq_ld8(uint a) {
    const uint v = T.Load(a & ~3u);
    return (v >> ((a & 3u) * 8u)) & 0xffu;
}

// Byte load from the weight buffer: block headers are byte-packed.
uint iq_wld8(uint a) {
    const uint v = W.Load(a & ~3u);
    return (v >> ((a & 3u) * 8u)) & 0xffu;
}

// Four packed grid bytes, each one negated when its sign bit is set. sbase is 0
// for the first four values of a group and 4 for the second four.
float4 iq_grid4(uint w, uint signs, uint sbase) {
    return float4((float)(w & 0xffu) * (((signs >> (sbase + 0u)) & 1u) ? -1.0 : 1.0),
                  (float)((w >> 8u) & 0xffu) * (((signs >> (sbase + 1u)) & 1u) ? -1.0 : 1.0),
                  (float)((w >> 16u) & 0xffu) * (((signs >> (sbase + 2u)) & 1u) ? -1.0 : 1.0),
                  (float)((w >> 24u) & 0xffu) * (((signs >> (sbase + 3u)) & 1u) ? -1.0 : 1.0));
}
