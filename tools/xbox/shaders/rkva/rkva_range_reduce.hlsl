// RKVA v2 split-K reduction. Output = contiguous O F32 followed by LSE F32.
cbuffer Params : register(b0) {
    uint n_query;
    uint n_head;
    uint head_dim;
    uint split_k;
};
ByteAddressBuffer partials : register(t0);
RWByteAddressBuffer output : register(u0);
groupshared float softmax_max;
groupshared float softmax_inv;
[numthreads(128, 1, 1)]
void CSRangeReduce(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint qh = group.y * n_head + group.x;
    uint total = n_query * n_head;
    uint stride = head_dim + 2;
    if (lane == 0) {
        float maximum = -3.402823466e38f;
        for (uint split = 0; split < split_k; ++split) {
            uint base = (split * total + qh) * stride;
            float denom = asfloat(partials.Load((base + head_dim + 1) * 4));
            if (denom > 0) maximum = max(maximum, asfloat(partials.Load((base + head_dim) * 4)));
        }
        float sum = 0;
        for (uint split = 0; split < split_k; ++split) {
            uint base = (split * total + qh) * stride;
            float denom = asfloat(partials.Load((base + head_dim + 1) * 4));
            if (denom > 0) sum += denom * exp(asfloat(partials.Load((base + head_dim) * 4)) - maximum);
        }
        softmax_max = maximum;
        softmax_inv = sum > 0 ? 1 / sum : 0;
        float lse = sum > 0 ? maximum + log(sum) : asfloat(0xff800000);
        output.Store((total * head_dim + qh) * 4, asuint(lse));
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint d = lane; d < head_dim; d += 128) {
        float acc = 0;
        for (uint split = 0; split < split_k; ++split) {
            uint base = (split * total + qh) * stride;
            float denom = asfloat(partials.Load((base + head_dim + 1) * 4));
            if (denom > 0) acc += asfloat(partials.Load((base + d) * 4)) *
                exp(asfloat(partials.Load((base + head_dim) * 4)) - softmax_max);
        }
        output.Store((qh * head_dim + d) * 4, asuint(acc * softmax_inv));
    }
}
