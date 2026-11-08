// Experimental shared-K/V MQ2, D256/GQA6/fixed Wave64. Not performance qualified.
// One wave per query; wave 0 dequantizes K/V once and publishes to both waves.
// Every thread participates in both row barriers, including the odd query tail.
// The root layout and partial/output records match the native rowlane shaders.
#include "rkva_bdr.hlsli"
#include "rkva_decode_quants_group_b.hlsli"
#include "rkva_decode_quants_group_c.hlsli"
#include "rkva_decode_quants_saw_int4.hlsli"

#ifndef NATIVE_MQ_FORMAT
#error NATIVE_MQ_FORMAT must be selected explicitly
#endif

#if NATIVE_MQ_FORMAT == 0
#define MQ_BYTES RKVA_ROW_HEAD_Q4_1
#define MQ_DIM quarter_dim_q4_1
#define MQ_LOAD load_q4_1
#elif NATIVE_MQ_FORMAT == 1
#define MQ_BYTES RKVA_ROW_HEAD_IQ4_NL
#define MQ_DIM quarter_dim_iq4_nl
#define MQ_LOAD load_iq4_nl
#elif NATIVE_MQ_FORMAT == 2
#define MQ_BYTES RKVA_ROW_HEAD_SAW_INT4
#define MQ_DIM quarter_dim_saw_int4
#define MQ_LOAD load_saw_int4
#elif NATIVE_MQ_FORMAT == 3
#define MQ_BYTES 132
uint mq_dim(uint lane) { return lane * 4u; }
#define MQ_DIM mq_dim
#else
#error Unsupported native MQ format
#endif

cbuffer Params : register(b0) {
    uint n_query; uint active_count; uint n_head; uint n_head_kv;
    uint head_dim; float kq_scale; uint origin; uint split_k;
    uint split_rows; uint direct_output;
};
ByteAddressBuffer queries : register(t0);
ByteAddressBuffer keys : register(t1);
ByteAddressBuffer values : register(t2);
ByteAddressBuffer positions : register(t3);
RWByteAddressBuffer partials : register(u0);
RWByteAddressBuffer output : register(u1);
// 2 KiB LDS, independent of query count and GQA. No query transform in LDS.
groupshared float4 mq_k[64];
groupshared float4 mq_v[64];
#if NATIVE_MQ_FORMAT == 3
groupshared float4 mq_query_wht[6 * 128];
#endif

#if NATIVE_MQ_FORMAT == 3
// Same adjacent-nibble codebook, original norm and correction as NC rowlane.
static const float mq_centroids[16] = {
    -1.7068263888e-01f, -1.2927794456e-01f, -1.0111761093e-01f, -7.8516095877e-02f,
    -5.8903016150e-02f, -4.1054993868e-02f, -2.4258611724e-02f, -8.0267284065e-03f,
     8.0267284065e-03f,  2.4258611724e-02f,  4.1054993868e-02f,  5.8903016150e-02f,
     7.8516095877e-02f,  1.0111761093e-01f,  1.2927794456e-01f,  1.7068263888e-01f
};
uint4 mq_indices(ByteAddressBuffer data, uint base, uint lane) {
    uint word = data.Load(base + (lane >> 1u) * 4u);
    word >>= (lane & 1u) * 16u;
    return uint4(word & 15u, (word >> 4u) & 15u, (word >> 8u) & 15u, (word >> 12u) & 15u);
}
#endif

[numthreads(128, 1, 1)]
void CSRangeAttnNativeMq2(uint tid : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint lane = WaveGetLaneIndex(), wave = tid >> 6u;
    uint qi = group.y * 2u + wave, kv_head = group.x, split = group.z;
    bool query_valid = qi < n_query;
    uint d0 = MQ_DIM(lane);
    uint head0 = qi * n_head + kv_head * 6u;
    uint limit = 0u;
    if (query_valid) {
        int pos = asint(positions.Load(qi * 4u));
        limit = pos < int(origin) ? 0u : min(active_count, uint(pos) - origin + 1u);
    }
    float4 q[6], acc[6];
    float maximum[6], denom[6];
    [unroll] for (uint g = 0; g < 6u; ++g) {
        q[g] = 0.0f;
        if (query_valid) q[g] = asfloat(queries.Load4(((head0 + g)*256u + d0)*4u));
#if NATIVE_MQ_FORMAT == 2
        q[g] = bdr16_wave4(q[g], lane);
#endif
        acc[g] = 0.0f; maximum[g] = -3.402823466e38f; denom[g] = 0.0f;
    }
#if NATIVE_MQ_FORMAT == 3
    // This console's WaveReadLaneAt lowering is unreliable with lane-varying indices.
    [unroll] for (uint g = 0; g < 6u; ++g) {
        float4 v = q[g];
        float a = v.x + v.y, b = v.x - v.y;
        float c = v.z + v.w, d = v.z - v.w;
        q[g] = float4(a + c, b + d, a - c, b - d);
    }
    [unroll] for (uint stride = 1u; stride < 64u; stride <<= 1u) {
        [unroll] for (uint g = 0; g < 6u; ++g) mq_query_wht[g * 128u + tid] = q[g];
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint g = 0; g < 6u; ++g) {
            float4 other = mq_query_wht[g * 128u + (tid ^ stride)];
            q[g] = (lane & stride) == 0u ? q[g] + other : other - q[g];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    [unroll] for (uint g = 0; g < 6u; ++g) q[g] *= 0.0625f;
#endif
    // Bounds are group-uniform, not query-dependent. Causal masking is per wave.
    uint begin = min(active_count, split * split_rows);
    uint end = min(active_count, (split + 1u) * split_rows);
    [loop] for (uint row = begin; row < end; ++row) {
        if (wave == 0u) {
            uint base = (row*n_head_kv + kv_head)*MQ_BYTES;
#if NATIVE_MQ_FORMAT == 3
            uint4 ki = mq_indices(keys, base, lane);
            float4 k = float4(mq_centroids[ki.x], mq_centroids[ki.y], mq_centroids[ki.z], mq_centroids[ki.w]);
            float norm_sq = WaveActiveSum(dot(k, k));
            float norm = f16tof32(keys.Load(base + 128u) & 65535u);
            mq_k[lane] = k * (norm * rsqrt(norm_sq + 1e-16f));
            uint4 vi = mq_indices(values, base, lane);
            uint meta = values.Load(base + 128u);
            mq_v[lane] = float4(vi) * f16tof32(meta & 65535u) + f16tof32(meta >> 16u);
#else
            mq_k[lane] = MQ_LOAD(keys, base, lane);
            mq_v[lane] = MQ_LOAD(values, base, lane);
#endif
        }
        GroupMemoryBarrierWithGroupSync();
        float4 k = mq_k[lane], v = mq_v[lane];
        bool visible = query_valid && row < limit;
        [unroll] for (uint g = 0; g < 6u; ++g) {
            float score = WaveActiveSum(dot(q[g], k)) * kq_scale;
            // Uniform across a wave; fully masked queries retain zero accumulators.
            if (visible) {
                float next = max(maximum[g], score);
                float old = denom[g] == 0.0f ? 0.0f : exp(maximum[g] - next);
                float p = exp(score - next);
                acc[g] = acc[g] * old + v * p;
                denom[g] = denom[g] * old + p;
                maximum[g] = next;
            }
        }
        // Do not overwrite LDS until both query waves have consumed this row.
        GroupMemoryBarrierWithGroupSync();
    }
    [unroll] for (uint g = 0; g < 6u; ++g) {
#if NATIVE_MQ_FORMAT == 2
        acc[g] = bdr16_wave4(acc[g], lane);
#endif
        if (query_valid) {
            uint qh = head0 + g, total = n_query*n_head;
            if (direct_output != 0u) {
                float inv = denom[g] > 0.0f ? 1.0f/denom[g] : 0.0f;
                output.Store4((qh*256u + d0)*4u, asuint(acc[g]*inv));
                if (lane == 0u) output.Store((total*256u + qh)*4u,
                    asuint(denom[g] > 0.0f ? maximum[g] + log(denom[g]) : asfloat(0xff800000u)));
            } else {
                uint dst = (split*total + qh)*258u*4u;
                partials.Store4(dst + d0*4u, asuint(acc[g]));
                if (lane == 0u) {
                    partials.Store(dst + 256u*4u, asuint(maximum[g]));
                    partials.Store(dst + 257u*4u, asuint(denom[g]));
                }
            }
        }
    }
}
