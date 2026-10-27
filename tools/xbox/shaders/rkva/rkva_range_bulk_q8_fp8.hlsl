// Raw-domain Q8_0 bulk query expansion. Unlike KVarN4, Q8_0 attention
// consumes unrotated Q, and its positions immediately follow the F32 values.
// The existing FP8 output pack shader is shared unchanged.
#include "rkva_range_bulk_f16.hlsl"

[numthreads(256, 1, 1)]
void CSBulkUnpackQFP8Raw(uint3 tid : SV_DispatchThreadID) {
    const uint idx = tid.x;
    if (idx < q_pairs) {
        const uint packed = src.Load((idx / 2) * 4) >> ((idx & 1u) * 16);
        dst.Store((idx * 2) * 4, asuint(f16tof32((packed & 0xffu) << 8)));
        dst.Store((idx * 2 + 1) * 4, asuint(f16tof32(((packed >> 8) & 0xffu) << 8)));
    }
    if (idx < aux_words) {
        dst.Store((q_pairs * 2 + idx) * 4, src.Load(q_pairs * 2 + idx * 4));
    }
}
