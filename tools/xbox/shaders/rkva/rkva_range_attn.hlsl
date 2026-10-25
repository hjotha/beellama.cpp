// RKVA v2: F16 positional overflow attention, partial numerators + softmax state.
// One group owns (Q head, query, split). Physical KV row r is origin+r.
// No rotation/quantization: the host uploads values from its original cache.
cbuffer Params : register(b0) {
    uint n_query;
    uint active_count;
    uint n_head;
    uint n_head_kv;
    uint head_dim;
    float kq_scale;
    uint origin;
    uint split_k;
    uint cache_storage; // informational; PSO is specialized by RKVA_RANGE_STORAGE
};
ByteAddressBuffer queries : register(t0);
ByteAddressBuffer keys : register(t1);
ByteAddressBuffer values : register(t2);
ByteAddressBuffer positions : register(t3);
RWByteAddressBuffer partials : register(u0);
groupshared float wave_dot[128];
groupshared float dot_score;

float load_half(ByteAddressBuffer buffer, uint element) {
    uint packed = buffer.Load((element >> 1) * 4);
    return f16tof32((element & 1) ? packed >> 16 : packed & 65535);
}

#ifndef RKVA_RANGE_STORAGE
#error Compile with RKVA_RANGE_STORAGE=0 (F16), 1 (Q4_0), or 2 (Q8_0).
#endif

// Standard GGML blocks have an F16 scale and packed quantized values.
// Q4_0 is 18 bytes / 32 values; Q8_0 is 34 bytes / 32 signed int8 values.
// Block scales alternate between byte offsets 0 and 2 in aligned DWORDs.
float load_cache(ByteAddressBuffer buffer, uint element) {
    #if RKVA_RANGE_STORAGE == 0
    return load_half(buffer, element);
#elif RKVA_RANGE_STORAGE == 2
    uint block = (element / 32) * 34;
    uint packed_scale = buffer.Load(block & ~3u);
    float scale = f16tof32((packed_scale >> ((block & 3u) * 8)) & 65535u);
    uint offset = block + 2 + (element & 31u);
    uint packed_quant = buffer.Load(offset & ~3u);
    uint byte_quant = (packed_quant >> ((offset & 3u) * 8)) & 255u;
    int quant = byte_quant >= 128u ? int(byte_quant) - 256 : int(byte_quant);
    return float(quant) * scale;
#else
    uint block = (element / 32) * 18;
    uint packed_scale = buffer.Load(block & ~3u);
    float scale = f16tof32((packed_scale >> ((block & 3u) * 8)) & 65535u);
    uint offset = block + 2 + (element & 15u);
    uint packed_quant = buffer.Load(offset & ~3u);
    uint byte_quant = (packed_quant >> ((offset & 3u) * 8)) & 255u;
    uint quant = (element & 31u) < 16u ? byte_quant & 15u : byte_quant >> 4;
    return float(int(quant) - 8) * scale;
#endif
}

[numthreads(128, 1, 1)]
void CSRangeAttn(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint head = group.x, qi = group.y, split = group.z;
    uint qh = qi * n_head + head;
    uint kv_head = head / (n_head / n_head_kv);
    int query_pos = asint(positions.Load(qi * 4));
    uint causal_count = query_pos < int(origin) ? 0 : uint(query_pos) - origin + 1;
    uint limit = min(active_count, causal_count);
    uint split_rows = (active_count + split_k - 1) / split_k;
    uint begin = min(limit, split * split_rows);
    uint end = min(limit, (split + 1) * split_rows);
    float q0 = asfloat(queries.Load((qh * head_dim + lane) * 4));
    float q1 = head_dim > 128 ? asfloat(queries.Load((qh * head_dim + lane + 128) * 4)) : 0;
    float acc0 = 0, acc1 = 0, denom = 0, maximum = -3.402823466e38f;
    uint wave_size = WaveGetLaneCount();
    [loop] for (uint row = begin; row < end; ++row) {
        uint base = (row * n_head_kv + kv_head) * head_dim;
        float dot = q0 * load_cache(keys, base + lane);
        if (head_dim > 128) dot += q1 * load_cache(keys, base + lane + 128);
        float sum = WaveActiveSum(dot);
        if (WaveIsFirstLane()) wave_dot[lane / wave_size] = sum;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            float all = 0;
            for (uint w = 0; w < 128 / wave_size; ++w) all += wave_dot[w];
            dot_score = all * kq_scale;
        }
        GroupMemoryBarrierWithGroupSync();
        float score = dot_score;
        float next_max = max(maximum, score);
        float old_weight = denom == 0 ? 0 : exp(maximum - next_max);
        float weight = exp(score - next_max);
        acc0 = acc0 * old_weight + weight * load_cache(values, base + lane);
        if (head_dim > 128)
            acc1 = acc1 * old_weight + weight * load_cache(values, base + lane + 128);
        denom = denom * old_weight + weight;
        maximum = next_max;
        // All threads must finish reading dot_score before the next row writes it.
        GroupMemoryBarrierWithGroupSync();
    }
    uint dst = ((split * n_query * n_head + qh) * (head_dim + 2)) * 4;
    partials.Store(dst + lane * 4, asuint(acc0));
    if (head_dim > 128) partials.Store(dst + (lane + 128) * 4, asuint(acc1));
    if (lane == 0) {
        partials.Store(dst + head_dim * 4, asuint(maximum));
        partials.Store(dst + (head_dim + 1) * 4, asuint(denom));
    }
}
