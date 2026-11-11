// ggml backend d3d12: IQ3_XXS x f32 matmul (decode and per-column prefill).
// block_iq3_xxs = { half d; uint8 qs[96]; } = 98 B. qs[0..63] are 8-bit grid
// indices, qs[64..95] one uint32 per 32-value group: bits 28..31 the 4-bit
// scale, four 7-bit fields selecting sign bytes from ksigns_iq2xs.
// dequantize_row_iq3_xxs: each 8-value group is two 4-value grid rows
// (iq3xxs_grid holds 4 bytes per row); sign bits 0..3 apply to the first row
// and 4..7 to the second. A thread owns 16 consecutive values: two groups of
// one 32-value group, so both share the same db.

#include "ggml_d3d12_common.hlsli"
#include "ggml_d3d12_iq_common.hlsli"

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint col = gid.y;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint e0 = itid * 16u;
    const uint ib32 = itid >> 1;
    const uint lb = (itid & 1u) * 2u;

    float acc[NUM_ROWS];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        acc[r] = 0.0;

    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        const uint xe = blk * 256u;
        const float4 x0 = xload(col, xe + e0);
        const float4 x1 = xload(col, xe + e0 + 4u);
        const float4 x2 = xload(col, xe + e0 + 8u);
        const float4 x3 = xload(col, xe + e0 + 12u);

        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + blk * 98u;
            const float d = f16tof32(ld16(bb));
            const uint aux = ld32(bb + 66u + 4u * ib32);
            // Four consecutive 8-bit grid indices: two per 8-value group.
            const uint q4 = ld32(bb + 2u + 8u * ib32 + 2u * lb);
            const uint s0 = iq_ld8(IQ_SIGNS_BASE + ((aux >> (7u * lb)) & 127u));
            const uint s1 = iq_ld8(IQ_SIGNS_BASE + ((aux >> (7u * (lb + 1u))) & 127u));
            const uint g0 = T.Load(IQ3XXS_GRID_BASE + (q4 & 0xffu) * 4u);
            const uint g1 = T.Load(IQ3XXS_GRID_BASE + ((q4 >> 8u) & 0xffu) * 4u);
            const uint g2 = T.Load(IQ3XXS_GRID_BASE + ((q4 >> 16u) & 0xffu) * 4u);
            const uint g3 = T.Load(IQ3XXS_GRID_BASE + ((q4 >> 24u) & 0xffu) * 4u);

            const float db = d * (0.5f + (float)(aux >> 28u)) * 0.5f;
            const float dp = dot(iq_grid4(g0, s0, 0u), x0) +
                             dot(iq_grid4(g1, s0, 4u), x1) +
                             dot(iq_grid4(g2, s1, 0u), x2) +
                             dot(iq_grid4(g3, s1, 4u), x3);
            acc[r] += db * dp;
        }
    }
    reduce_store(acc, tid, row0, col);
}
