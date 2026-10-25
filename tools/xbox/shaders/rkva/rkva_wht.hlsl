// RKVA WHT — 256-point normalized Hadamard transform (Xbox Series X, RDNA2).
//
// Port of ggml-vulkan kvarn_wht.comp (data_type=0, F32) for the RKVA engine.
// The transform is self-inverse (H*H = I): the same kernel rotates Q before the
// rotated-domain attention and rotates the attention output back to the
// original domain. qwen35 geometry: head_width=256 = 2 slices of 128; per-slice
// butterfly scale 1/sqrt(128), then cross-slice Hadamard mix scale 1/sqrt(2);
// the product is exactly 1/sqrt(256).
//
// One thread group owns one row (Q head row or attention output row).
// Dispatch: grid (n_rows, 1, 1), numthreads(128, 1, 1).
//
// b0 = {n_rows, head_width, data_type}, t0 = src F32 rows, u0 = dst F32 rows.
// Layout: [n_rows][head_width] F32, row-major, both buffers.

cbuffer Params : register(b0) {
    uint n_rows;
    uint head_width;   // 256
    uint data_type;    // 0 = F32 (only type used by RKVA)
};

ByteAddressBuffer   src : register(t0); // F32 [n_rows][head_width]
RWByteAddressBuffer dst : register(u0); // F32 [n_rows][head_width]

groupshared float sh_values[2 * 128];

[numthreads(128, 1, 1)]
void CSWht(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint row = group.x;
    if (row >= n_rows) {
        return;
    }
    const uint slices = head_width / 128;
    if (slices != 2) {
        return;  // RKVA geometry is fixed at head_width 256
    }

    const uint row_base = row * head_width;
    float values[2] = {0.0f, 0.0f};

    // Per-slice 128-point butterfly, 7 stages, through groupshared (no
    // subgroup dependence): value = (lane & stride)==0 ? value+other : other-value.
    for (uint slice = 0; slice < 2; ++slice) {
        float value = asfloat(src.Load((row_base + slice * 128 + lane) * 4));
        [unroll]
        for (uint stride = 1; stride < 128; stride <<= 1) {
            sh_values[slice * 128 + lane] = value;
            GroupMemoryBarrierWithGroupSync();
            const float other = sh_values[slice * 128 + (lane ^ stride)];
            GroupMemoryBarrierWithGroupSync();
            value = (lane & stride) == 0u ? value + other : other - value;
        }
        values[slice] = value * 0.08838834764831845f;  // 1/sqrt(128)
    }

    // Cross-slice Hadamard mix (2 slices): (a+b, a-b) * 1/sqrt(2).
    const float a = values[0];
    const float b = values[1];
    values[0] = (a + b) * 0.7071067811865475f;
    values[1] = (a - b) * 0.7071067811865475f;

    [unroll]
    for (uint slice = 0; slice < 2; ++slice) {
        dst.Store((row_base + slice * 128 + lane) * 4, asuint(values[slice]));
    }
}