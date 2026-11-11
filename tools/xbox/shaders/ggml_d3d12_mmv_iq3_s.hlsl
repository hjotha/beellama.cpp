// ggml backend d3d12: IQ3_S x f32 matmul (decode and per-column prefill).
// block_iq3_s = { half d; uint8 qs[64]; uint8 qh[8]; uint8 signs[32];
//                 uint8 scales[4]; } = 110 B.
// dequantize_row_iq3_s walks 64 values at a time: one qh byte and two 4-bit
// scales per 32-value half, four 8-value groups per half. Each group is two
// 4-value iq3s_grid rows; the 9th index bit comes from qh, shifted by
// (8 - 2*l) for the first row and (7 - 2*l) for the second. A thread owns 16
// consecutive values: the lb/lb+1 pair of one 32-value half.

#include "ggml_d3d12_common.hlsli"
#include "ggml_d3d12_iq_common.hlsli"

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint col = gid.y;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint e0 = itid * 16u;
    const uint m = itid >> 2;             // 64-value group: 4 threads each
    const uint second = (itid >> 1) & 1u; // first/second 32-value half
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
            const uint bb = row * w_row_bytes + blk * 110u;
            const float d = f16tof32(ld16(bb));
            const uint h = iq_wld8(bb + 66u + 2u * m + second);
            const uint q4 = ld32(bb + 2u + 16u * m + 8u * second + 2u * lb);
            const uint sg = ld16(bb + 74u + 8u * m + 4u * second + lb);
            const uint sc = iq_wld8(bb + 106u + m);

            const uint i0 = (q4 & 0xffu) | ((h << (8u - 2u * lb)) & 256u);
            const uint i1 = ((q4 >> 8u) & 0xffu) | ((h << (7u - 2u * lb)) & 256u);
            const uint i2 = ((q4 >> 16u) & 0xffu) | ((h << (6u - 2u * lb)) & 256u);
            const uint i3 = ((q4 >> 24u) & 0xffu) | ((h << (5u - 2u * lb)) & 256u);
            const uint g0 = T.Load(IQ3S_GRID_BASE + i0 * 4u);
            const uint g1 = T.Load(IQ3S_GRID_BASE + i1 * 4u);
            const uint g2 = T.Load(IQ3S_GRID_BASE + i2 * 4u);
            const uint g3 = T.Load(IQ3S_GRID_BASE + i3 * 4u);
            const uint s0 = sg & 0xffu;
            const uint s1 = sg >> 8u;

            const float db = d * (1.0f + 2.0f * (float)(second ? (sc >> 4u) : (sc & 0xFu)));
            const float dp = dot(iq_grid4(g0, s0, 0u), x0) +
                             dot(iq_grid4(g1, s0, 4u), x1) +
                             dot(iq_grid4(g2, s1, 0u), x2) +
                             dot(iq_grid4(g3, s1, 4u), x3);
            acc[r] += db * dp;
        }
    }
    reduce_store(acc, tid, row0, col);
}
