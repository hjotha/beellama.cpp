// RKVA v2 split-K reduction for up to 256 splits (optimized Q4 path).
// Same input/output layout and arithmetic as rkva_range_reduce.hlsl: the split
// maximum, weights exp(m_s - M), denominator and numerator sums are evaluated
// in the same split order, but each split's (max, denom) and weight are loaded
// or computed once per group instead of once per output element.
#define RKVA_REDUCE_MAX_SPLITS 256
cbuffer Params : register(b0) {
    uint n_query;
    uint n_head;
    uint head_dim;
    uint split_k;
};
ByteAddressBuffer partials : register(t0);
RWByteAddressBuffer output : register(u0);
groupshared float gs_split_max[RKVA_REDUCE_MAX_SPLITS];
groupshared float gs_split_den[RKVA_REDUCE_MAX_SPLITS];
groupshared float gs_split_weight[RKVA_REDUCE_MAX_SPLITS];
groupshared float gs_inv;
groupshared uint gs_invalid;

#if RKVA_REDUCE_WHT
groupshared float gs_wht[256];
#endif

[numthreads(128, 1, 1)]
void CSRangeReduceOpt(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint qh = group.y * n_head + group.x;
    uint total = n_query * n_head;
    uint stride = head_dim + 2;
    uint splits = min(split_k, RKVA_REDUCE_MAX_SPLITS);
    if (lane == 0) gs_invalid = split_k > RKVA_REDUCE_MAX_SPLITS ? 1u : 0u;
    for (uint s = lane; s < splits; s += 128) {
        uint base = (s * total + qh) * stride;
        gs_split_max[s] = asfloat(partials.Load((base + head_dim) * 4));
        gs_split_den[s] = asfloat(partials.Load((base + head_dim + 1) * 4));
    }
    GroupMemoryBarrierWithGroupSync();
    float maximum = -3.402823466e38f;
    // Only den == 0 is empty. Invalid participating states must never be
    // dropped by a comparison with NaN or by an underflowed merge weight.
    for (uint s = 0; s < splits; ++s) {
        float den = gs_split_den[s];
        if (den == 0) continue;
        if (!isfinite(den) || den < 0 || !isfinite(gs_split_max[s])) {
            if (lane == 0) gs_invalid = 1;
        } else {
            maximum = max(maximum, gs_split_max[s]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gs_invalid != 0) {
        for (uint d = lane; d < head_dim; d += 128)
            output.Store((qh * head_dim + d) * 4, 0x7fc00000u);
        if (lane == 0) output.Store((total * head_dim + qh) * 4, 0x7fc00000u);
        return;
    }
    for (uint s = lane; s < splits; s += 128) {
        gs_split_weight[s] = gs_split_den[s] != 0 ? exp(gs_split_max[s] - maximum) : 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0) {
        float sum = 0;
        for (uint s = 0; s < splits; ++s) {
            if (gs_split_den[s] != 0) sum += gs_split_den[s] * gs_split_weight[s];
        }
        gs_inv = sum == 0 ? 0 : 1 / sum;
        if (!isfinite(sum) || sum < 0 || !isfinite(gs_inv)) gs_invalid = 1;
        float lse = sum == 0 ? asfloat(0xff800000) : maximum + log(sum);
        if (sum != 0 && !isfinite(lse)) gs_invalid = 1;
        output.Store((total * head_dim + qh) * 4, asuint(lse));
    }
    GroupMemoryBarrierWithGroupSync();
    float inv = gs_inv;
    bool invalid_numerator = false;
#if RKVA_REDUCE_WHT
    float acc0 = 0;
    float acc1 = 0;
    for (uint s = 0; s < splits; ++s) {
        if (gs_split_den[s] != 0) {
            float w = gs_split_weight[s];
            uint base = (s * total + qh) * stride;
            float v0 = asfloat(partials.Load((base + lane) * 4));
            float v1 = asfloat(partials.Load((base + lane + 128) * 4));
            invalid_numerator = invalid_numerator || !isfinite(v0) || !isfinite(v1);
            acc0 += v0 * w;
            acc1 += v1 * w;
        }
    }
    invalid_numerator = invalid_numerator || !isfinite(acc0) || !isfinite(acc1);
    gs_wht[lane] = acc0 * inv;
    gs_wht[lane + 128] = acc1 * inv;
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint stride_wht = 1; stride_wht < 128; stride_wht *= 2) {
        float own0 = gs_wht[lane];
        float other0 = gs_wht[lane ^ stride_wht];
        float own1 = gs_wht[lane + 128];
        float other1 = gs_wht[(lane ^ stride_wht) + 128];
        GroupMemoryBarrierWithGroupSync();
        gs_wht[lane] = (lane & stride_wht) ? other0 - own0 : own0 + other0;
        gs_wht[lane + 128] = (lane & stride_wht) ? other1 - own1 : own1 + other1;
        GroupMemoryBarrierWithGroupSync();
    }
    float x = gs_wht[lane] * 0.08838834764831845f;
    float y = gs_wht[lane + 128] * 0.08838834764831845f;
    float out0 = (x + y) * 0.7071067811865475f;
    float out1 = (x - y) * 0.7071067811865475f;

    invalid_numerator = invalid_numerator || !isfinite(out0) || !isfinite(out1);
    output.Store((qh * head_dim + lane) * 4, asuint(out0));
    output.Store((qh * head_dim + lane + 128) * 4, asuint(out1));
#else
    for (uint d = lane; d < head_dim; d += 128) {
        float acc = 0;
        for (uint s = 0; s < splits; ++s) {
            if (gs_split_den[s] != 0) {
                float value = asfloat(partials.Load(((s * total + qh) * stride + d) * 4));
                invalid_numerator = invalid_numerator || !isfinite(value);
                acc += value * gs_split_weight[s];
            }
        }
        invalid_numerator = invalid_numerator || !isfinite(acc * inv);
        output.Store((qh * head_dim + d) * 4, asuint(acc * inv));
    }
#endif
    if (invalid_numerator) InterlockedOr(gs_invalid, 1u);
    GroupMemoryBarrierWithGroupSync();
    if (gs_invalid != 0) {
        for (uint d = lane; d < head_dim; d += 128)
            output.Store((qh * head_dim + d) * 4, 0x7fc00000u);
        if (lane == 0) output.Store((total * head_dim + qh) * 4, 0x7fc00000u);
    }
}
