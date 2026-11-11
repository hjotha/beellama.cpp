// ggml backend d3d12: Q3_K x f32 matmul (decode and per-column prefill).
// block_q3_K = { uint8 hmask[32]; uint8 qs[64]; uint8 scales[12]; half d; }
// = 110 B, so block starts can be 2-byte aligned: qs/hmask/scales use ld32.
// dequantize_row_q3_K: each 128-value half has four 32-value j groups with
// shift 0/2/4/6 and hmask bit m = 1 << (2*il + j); each j covers two
// 16-value scale groups. The 12 scale bytes are unpacked into 16 signed
// six-bit values (value - 32). Lane map and reduction match the Q4_K kernel.

#include "ggml_d3d12_common.hlsli"

float4 q3_4(uint qword, uint shift, uint m, uint hword) {
    float4 r;
    [unroll]
    for (uint i = 0; i < 4u; ++i) {
        const uint low = (qword >> (shift + 8u * i)) & 3u;
        const uint high = (hword >> (8u * i)) & 0xFFu;
        r[i] = (float)low - ((high & m) ? 0.0 : 4.0);
    }
    return r;
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
    const uint qbase = 32u + ((il & 2u) ? 32u : 0u) + ir * 8u;
    const uint hbase = ir * 8u;
    const uint sl = (il & 1u) ? 4u : 0u;
    const uint sh = sl + 2u;
    const uint ml = 1u << (2u * il);
    const uint mh = 1u << (2u * il + 1u);
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

        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + blk * 110u;
            const uint qa = ld32(bb + qbase);
            const uint qb = ld32(bb + qbase + 4u);
            const uint ha = ld32(bb + hbase);
            const uint hb = ld32(bb + hbase + 4u);

            const float4 lo0 = q3_4(qa, sl, ml, ha);
            const float4 lo1 = q3_4(qb, sl, ml, hb);
            const float4 hi0 = q3_4(qa, sh, mh, ha);
            const float4 hi1 = q3_4(qb, sh, mh, hb);

            const uint s0 = ld32(bb + 96u);
            const uint s1 = ld32(bb + 100u);
            const uint s2 = ld32(bb + 104u);
            const uint tmp = s2;
            const uint aux0 = (s0 & 0x0f0f0f0fu) | (((tmp >> 0u) & 0x03030303u) << 4u);
            const uint aux1 = (s1 & 0x0f0f0f0fu) | (((tmp >> 2u) & 0x03030303u) << 4u);
            const uint aux2 = ((s0 >> 4u) & 0x0f0f0f0fu) | (((tmp >> 4u) & 0x03030303u) << 4u);
            const uint aux3 = ((s1 >> 4u) & 0x0f0f0f0fu) | (((tmp >> 6u) & 0x03030303u) << 4u);
            uint aux[4] = {aux0, aux1, aux2, aux3};
            const uint sb_lo = (aux[si_lo >> 2] >> ((si_lo & 3u) * 8u)) & 0xFFu;
            const uint sb_hi = (aux[si_hi >> 2] >> ((si_hi & 3u) * 8u)) & 0xFFu;
            const float scl = (float)((int)sb_lo - 32);
            const float sch = (float)((int)sb_hi - 32);
            const float d = f16tof32(ld16(bb + 108u));

            const float lo = dot(lo0, xl0) + dot(lo1, xl1);
            const float hi = dot(hi0, xh0) + dot(hi1, xh1);
            acc[r] += d * (scl * lo + sch * hi);
        }
    }
    reduce_store(acc, tid, row0, col);
}
