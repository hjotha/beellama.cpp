// Native Bee KVarN4 descriptor v1: host-packed records and rotated F16 stage.
// Exact precision-tail overlay is original-domain F16. Q and OUT stay original.
// This shader never converts KVarN records to a persistent alternative cache.
//
// RKVA_SERIAL_PREFIX=1 (descriptor revision 3, separate PSO): every query reads
// the image of its own causal prefix q_count = limit instead of the resident
// image of active_count. The live staged group, the record/stage split and the
// exact-tail window [q_count - precision_tail_tokens, q_count) follow q_count,
// and the ORIGINAL-F16 tail is a 256-row physical ring (row & 255). Without
// the define the legacy revision 1/2 source is unchanged token for token.
#ifndef RKVA_SERIAL_PREFIX
#define RKVA_SERIAL_PREFIX 0
#endif
#if RKVA_SERIAL_PREFIX
static uint rkva_window; // per-thread q_count, set once in the entry point
#define RKVA_WINDOW rkva_window
#define RKVA_TAIL_MASK 255u
#else
#define RKVA_WINDOW active_count
#define RKVA_TAIL_MASK 127u
#endif
cbuffer Params : register(b0) {
    uint n_query, active_count, n_head, n_head_kv, head_dim;
    float kq_scale;
    uint origin, split_k, cache_storage, records_bytes, precision_tail_tokens;
};
ByteAddressBuffer queries : register(t0);
ByteAddressBuffer keys : register(t1);
ByteAddressBuffer values : register(t2);
ByteAddressBuffer positions : register(t3);
RWByteAddressBuffer partials : register(u0);
groupshared float transform[256];
groupshared float wave_dot[128];
groupshared float dot_score;

// Exactly the H128-normalize then cross-slice-normalize ordering of Bee's WHT.
void wht256(inout float a, inout float b, uint lane) {
    transform[lane] = a;
    transform[lane + 128] = b;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = 1; stride < 128; stride *= 2) {
        float own0 = transform[lane], other0 = transform[lane ^ stride];
        float own1 = transform[lane + 128], other1 = transform[(lane ^ stride) + 128];
        GroupMemoryBarrierWithGroupSync();
        transform[lane] = (lane & stride) ? other0 - own0 : own0 + other0;
        transform[lane + 128] = (lane & stride) ? other1 - own1 : own1 + other1;
        GroupMemoryBarrierWithGroupSync();
    }
    float x = transform[lane] * 0.08838834764831845f;
    float y = transform[lane + 128] * 0.08838834764831845f;
    a = (x + y) * 0.7071067811865475f;
    b = (x - y) * 0.7071067811865475f;
    GroupMemoryBarrierWithGroupSync();
}
float half_at(ByteAddressBuffer buffer, uint byte_offset) {
    uint bits = buffer.Load(byte_offset & ~3u);
    return f16tof32((bits >> ((byte_offset & 2u) * 8)) & 65535u);
}
float load_native(ByteAddressBuffer buffer, bool value, uint token, uint head,
                  uint dim, uint slice, uint live_group) {
    uint group = token / 128, pos = token & 127;
    uint physical_heads = n_head_kv * 2;
    uint physical_head = head * 2 + slice;
    uint global_group = origin / 128 + group;
    bool staged = global_group == 0 ||
                  (group == live_group && (RKVA_WINDOW & 127u) != 0);
    if (staged) {
        uint slot = global_group == 0 ? 0 : 1 + ((global_group - 1) & 1);
        uint element = ((slot * 128 + pos) * physical_heads + physical_head) * 128 + dim;
        return half_at(buffer, records_bytes + element * 2);
    }
    uint base = (group * physical_heads + physical_head) * 8960;
    uint row = value ? pos : dim, col = value ? dim : pos;
    uint index = row * 128 + col;
    uint packed = buffer.Load(base + (index / 8) * 4);
    uint quant = (packed >> ((index & 7) * 4)) & 15;
    float scale = half_at(buffer, base + 8192 + row * 2);
    float zero = half_at(buffer, base + 8448 + row * 2);
    float other = half_at(buffer, base + 8704 + col * 2);
    precise float affine = float(quant) * scale + zero;
    return affine * other;
}
float load_tail(ByteAddressBuffer buffer, uint token, uint head, uint dim, uint tail_begin) {
    uint stage_bytes = 384 * n_head_kv * 256 * 2;
    uint element = ((token & RKVA_TAIL_MASK) * n_head_kv + head) * 256 + dim;
    return half_at(buffer, records_bytes + stage_bytes + element * 2);
}
[numthreads(128, 1, 1)]
void CSRangeAttn(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint head = group.x, qi = group.y, split = group.z;
    uint qh = qi * n_head + head;
    uint kv_head = head / (n_head / n_head_kv);
    int query_pos = asint(positions.Load(qi * 4));
    uint causal_count = query_pos < int(origin) ? 0 : uint(query_pos) - origin + 1;
    uint limit = min(active_count, causal_count);
#if RKVA_SERIAL_PREFIX
    rkva_window = limit;
#endif
    uint split_rows = (active_count + split_k - 1) / split_k;
    uint begin = min(limit, split * split_rows), end = min(limit, (split + 1) * split_rows);
    float original_q0 = asfloat(queries.Load((qh * 256 + lane) * 4));
    float original_q1 = asfloat(queries.Load((qh * 256 + lane + 128) * 4));
    float q0 = original_q0, q1 = original_q1;
    wht256(q0, q1, lane);
    float body0 = 0, body1 = 0, tail0 = 0, tail1 = 0;
    float denom = 0, maximum = -3.402823466e38f;
    uint wave_size = WaveGetLaneCount();
    uint live_group = RKVA_WINDOW ? (RKVA_WINDOW - 1) / 128 : 0;
    uint tail_begin = RKVA_WINDOW - min(RKVA_WINDOW, precision_tail_tokens);
    [loop] for (uint row = begin; row < end; ++row) {
        bool exact = row >= tail_begin;
        float k0 = exact ? load_tail(keys, row, kv_head, lane, tail_begin) :
                          load_native(keys, false, row, kv_head, lane, 0, live_group);
        float k1 = exact ? load_tail(keys, row, kv_head, lane + 128, tail_begin) :
                          load_native(keys, false, row, kv_head, lane, 1, live_group);
        float dot = exact ? original_q0 * k0 + original_q1 * k1 : q0 * k0 + q1 * k1;
        float sum = WaveActiveSum(dot);
        if (WaveIsFirstLane()) wave_dot[lane / wave_size] = sum;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            float all = 0;
            for (uint w = 0; w < 128 / wave_size; ++w) all += wave_dot[w];
            dot_score = all * kq_scale;
        }
        GroupMemoryBarrierWithGroupSync();
        float next_max = max(maximum, dot_score);
        float old_weight = denom == 0 ? 0 : exp(maximum - next_max);
        float weight = exp(dot_score - next_max);
        body0 *= old_weight; body1 *= old_weight;
        tail0 *= old_weight; tail1 *= old_weight;
        if (exact) {
            tail0 += weight * load_tail(values, row, kv_head, lane, tail_begin);
            tail1 += weight * load_tail(values, row, kv_head, lane + 128, tail_begin);
        } else {
            body0 += weight * load_native(values, true, row, kv_head, lane, 0, live_group);
            body1 += weight * load_native(values, true, row, kv_head, lane, 1, live_group);
        }
        denom = denom * old_weight + weight;
        maximum = next_max;
        GroupMemoryBarrierWithGroupSync();
    }
    wht256(body0, body1, lane);
    uint dst = ((split * n_query * n_head + qh) * 258) * 4;
    partials.Store(dst + lane * 4, asuint(body0 + tail0));
    partials.Store(dst + (lane + 128) * 4, asuint(body1 + tail1));
    if (lane == 0) {
        partials.Store(dst + 256 * 4, asuint(maximum));
        partials.Store(dst + 257 * 4, asuint(denom));
    }
}
