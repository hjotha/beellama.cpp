// ggml backend d3d12: IQ2_S x f32 matmul (decode and per-column prefill).
// block_iq2_s = { half d; uint8 qs[64]; uint8 qh[8]; uint8 scales[8]; } = 82 B.
// qs[0..31] are grid indices, qs[32..63] the sign bytes.
// dequantize_row_iq2_s: for each of the 8 32-value groups, four 8-value grid
// rows use qs[4*ib+l] and the 2-bit slot (qh[ib] >> 2*l) of the 10-bit grid
// index; scales[ib] holds two 4-bit scales (low = first 16, high = second 16).
// A thread owns 16 consecutive values: one 16-value half of one 32-value group,
// i.e. grid rows lb and lb+1 with the same scale. Lane map and reduction match
// ggml_d3d12_mmv_q4_k.hlsl (thread itid handles the 16 values at 16*itid).

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
    const uint half = itid & 1u;
    const uint lb = half * 2u;

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
            const uint bb = row * w_row_bytes + blk * 82u;
            const float d = f16tof32(ld16(bb));
            const uint sc = iq_wld8(bb + 74u + ib32);
            const uint qh = iq_wld8(bb + 66u + ib32);
            // Two grid indices / two sign bytes per thread, both 2-byte aligned.
            const uint qi = ld16(bb + 2u + 4u * ib32 + lb);
            const uint sg = ld16(bb + 34u + 4u * ib32 + lb);
            const uint idx0 = (qi & 0xffu) | ((qh << (8u - 2u * lb)) & 0x300u);
            const uint idx1 = (qi >> 8u) | ((qh << (8u - 2u * (lb + 1u))) & 0x300u);
            const uint2 g0 = T.Load2(IQ2S_GRID_BASE + idx0 * 8u);
            const uint2 g1 = T.Load2(IQ2S_GRID_BASE + idx1 * 8u);
            const uint s0 = sg & 0xffu;
            const uint s1 = sg >> 8u;

            const float dg = d * (0.5f + (float)(half ? (sc >> 4u) : (sc & 0xFu))) * 0.25f;
            const float dp = dot(iq_grid4(g0.x, s0, 0u), x0) +
                             dot(iq_grid4(g0.y, s0, 4u), x1) +
                             dot(iq_grid4(g1.x, s1, 0u), x2) +
                             dot(iq_grid4(g1.y, s1, 4u), x3);
            acc[r] += dg * dp;
        }
    }
    reduce_store(acc, tid, row0, col);
}
