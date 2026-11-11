// ggml backend d3d12: IQ4_XS x f32 matmul (decode and per-column prefill).
// block_iq4_xs = { half d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128]; }
// = 136 B. dequantize_row_iq4_xs: eight 32-value groups; each has a 6-bit
// scale (4 bits from scales_l, 2 from scales_h) applied as d*(ls - 32), and
// 16 qs bytes holding 32 values: low nibbles are the first 16, high nibbles the
// second 16. A thread owns 16 consecutive values, i.e. one nibble half of one
// 32-value group, and looks the nibbles up in kvalues_iq4nl (float32 in the
// table buffer).

#include "ggml_d3d12_common.hlsli"
#include "ggml_d3d12_iq_common.hlsli"

float4 iq4nl4(uint w, uint sh) {
    return float4(asfloat(T.Load(IQ4NL_BASE + (((w >> (sh + 0u)) & 0xFu) * 4u))),
                  asfloat(T.Load(IQ4NL_BASE + (((w >> (sh + 8u)) & 0xFu) * 4u))),
                  asfloat(T.Load(IQ4NL_BASE + (((w >> (sh + 16u)) & 0xFu) * 4u))),
                  asfloat(T.Load(IQ4NL_BASE + (((w >> (sh + 24u)) & 0xFu) * 4u))));
}

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint col = gid.y;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint e0 = itid * 16u;
    const uint ib = itid >> 1;         // 32-value group: 2 threads each
    const uint half = itid & 1u;       // 0 = low nibbles, 1 = high nibbles
    const uint sh = half * 4u;

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
            const uint bb = row * w_row_bytes + blk * 136u;
            const float d = f16tof32(ld16(bb));
            const uint sl = iq_wld8(bb + 4u + (ib >> 1));
            const uint ls = ((sl >> (4u * (ib & 1u))) & 0xFu) |
                            (((ld16(bb + 2u) >> (2u * ib)) & 3u) << 4u);
            const uint4 q = W.Load4(bb + 8u + 16u * ib);

            const float dl = d * ((float)ls - 32.0f);
            const float dp = dot(iq4nl4(q.x, sh), x0) + dot(iq4nl4(q.y, sh), x1) +
                             dot(iq4nl4(q.z, sh), x2) + dot(iq4nl4(q.w, sh), x3);
            acc[r] += dl * dp;
        }
    }
    reduce_store(acc, tid, row0, col);
}
