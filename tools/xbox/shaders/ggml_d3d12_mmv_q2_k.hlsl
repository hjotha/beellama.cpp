// ggml backend d3d12: Q2_K x f32 matmul (decode and per-column prefill).
// block_q2_K = { uint8 scales[16]; uint8 qs[64]; half d; half dmin; } = 84 B.
// dequantize_row_q2_K: for each 128-value half, four 32-value j groups use
// shift 0/2/4/6; each j covers two 16-value scale groups (scales[is++] low
// nibble = d scale, high nibble = min). Thread itid owns 16 values: eight at
// e_lo (first 32 of its 64-value il segment) and eight at e_hi = e_lo + 32.
// The q bytes for both halves are the same eight bytes; only the 2-bit shift
// changes. Lane map and reduction match ggml_d3d12_mmv_q4_k.hlsl.

#include "ggml_d3d12_common.hlsli"

float4 q2_4(uint word, uint shift) {
    return float4((float)((word >> (shift + 0u)) & 3u), (float)((word >> (shift + 8u)) & 3u),
                  (float)((word >> (shift + 16u)) & 3u), (float)((word >> (shift + 24u)) & 3u));
}

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint col = gid.y;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint il = itid >> 2;
    const uint ir = itid & 3u;
    const uint e_lo = il * 64u + ir * 8u;
    const uint e_hi = e_lo + 32u;
    const uint qbase = 16u + ((il & 2u) ? 32u : 0u) + ir * 8u;
    const uint sl = (il & 1u) ? 4u : 0u;
    const uint sh = sl + 2u;
    const uint si_lo = 4u * il + (ir >= 2u ? 1u : 0u);
    const uint si_hi = 4u * il + 2u + (ir >= 2u ? 1u : 0u);

    float acc[NUM_ROWS];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        acc[r] = 0.0;

    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        const uint xe = blk * 256u;
        const float4 xl0 = xload(col, xe + e_lo);
        const float4 xl1 = xload(col, xe + e_lo + 4u);
        const float4 xh0 = xload(col, xe + e_hi);
        const float4 xh1 = xload(col, xe + e_hi + 4u);
        const float sxl = dot(xl0 + xl1, float4(1, 1, 1, 1));
        const float sxh = dot(xh0 + xh1, float4(1, 1, 1, 1));

        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + blk * 84u;
            const float d = f16tof32(ld16(bb + 80u));
            const float dmin = f16tof32(ld16(bb + 82u));
            const uint sc_lo =
                (W.Load(bb + (si_lo & ~3u)) >> ((si_lo & 3u) * 8u)) & 0xFFu;
            const uint sc_hi =
                (W.Load(bb + (si_hi & ~3u)) >> ((si_hi & 3u) * 8u)) & 0xFFu;
            const uint2 q = W.Load2(bb + qbase);

            const float lo = dot(q2_4(q.x, sl), xl0) + dot(q2_4(q.y, sl), xl1);
            const float hi = dot(q2_4(q.x, sh), xh0) + dot(q2_4(q.y, sh), xh1);
            acc[r] += d * (float)(sc_lo & 0xFu) * lo - dmin * (float)(sc_lo >> 4u) * sxl +
                      d * (float)(sc_hi & 0xFu) * hi - dmin * (float)(sc_hi >> 4u) * sxh;
        }
    }
    reduce_store(acc, tid, row0, col);
}
