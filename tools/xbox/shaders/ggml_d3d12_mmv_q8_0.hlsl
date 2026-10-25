// ggml backend d3d12: Q8_0 x f32 matmul, one invariant kernel per column.
// block_q8_0 = { half d; int8 qs[32]; } = 34 B, value = signed_byte * d.
// A 256-element chunk has eight blocks; each of its 16 lanes owns 16 values.

#include "ggml_d3d12_common.hlsli"

float4 signed_bytes(uint word) {
    float4 r;
    [unroll]
    for (uint i = 0; i < 4u; ++i)
        r[i] = (float)(asint(((word >> (8u * i)) & 0xffu) << 24) >> 24);
    return r;
}

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint col = gid.y;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint b = itid >> 1;
    const uint h = itid & 1u;
    const uint e = 32u * b + 16u * h;
    float acc[NUM_ROWS];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        acc[r] = 0.0;

    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        const uint xe = blk * 256u + e;
        const float4 x0 = xload(col, xe);
        const float4 x1 = xload(col, xe + 4u);
        const float4 x2 = xload(col, xe + 8u);
        const float4 x3 = xload(col, xe + 12u);
        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + (blk * 8u + b) * 34u;
            const float d = f16tof32(ld16(bb));
            const uint qs = bb + 2u + 16u * h;
            const float dotq = dot(signed_bytes(ld32(qs)), x0) +
                               dot(signed_bytes(ld32(qs + 4u)), x1) +
                               dot(signed_bytes(ld32(qs + 8u)), x2) +
                               dot(signed_bytes(ld32(qs + 12u)), x3);
            acc[r] += d * dotq;
        }
    }
    reduce_store(acc, tid, row0, col);
}
