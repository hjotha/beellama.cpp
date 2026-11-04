// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Experimental RKVA D256 TurboQuant 4bit NC reference. K is padded to 132 B:
// 128 B adjacent nibble indices, original norm fp16, two padding bytes. V is
// 128 B adjacent nibble indices, scale fp16, minimum fp16. Q/OUT are original
// domain float32; only Q is WHT-rotated. This shader requires Wave64 and GQA6.

#define TQ_DIM 256
#define TQ_GQA 6
#define TQ_THREADS 128
#define TQ_WAVES 2
#define TQ_ROW_BYTES 132

cbuffer Params : register(b0) {
    uint n_query;
    uint active_count;
    uint n_head;
    uint n_head_kv;
    uint head_dim;
    float kq_scale;
    uint origin;
    uint split_k;
    uint split_rows;
    uint direct_output;
};

ByteAddressBuffer queries : register(t0);
ByteAddressBuffer keys : register(t1);
ByteAddressBuffer values : register(t2);
ByteAddressBuffer positions : register(t3);
RWByteAddressBuffer partials : register(u0);
RWByteAddressBuffer output : register(u1);

groupshared float tq_query[TQ_GQA*TQ_DIM];
groupshared float tq_max[TQ_WAVES*TQ_GQA];
groupshared float tq_den[TQ_WAVES*TQ_GQA];
groupshared float tq_acc[TQ_WAVES*TQ_GQA*TQ_DIM];

static const float tq_centroids[16] = {
    -1.7068263888e-01f, -1.2927794456e-01f, -1.0111761093e-01f, -7.8516095877e-02f,
    -5.8903016150e-02f, -4.1054993868e-02f, -2.4258611724e-02f, -8.0267284065e-03f,
     8.0267284065e-03f,  2.4258611724e-02f,  4.1054993868e-02f,  5.8903016150e-02f,
     7.8516095877e-02f,  1.0111761093e-01f,  1.2927794456e-01f,  1.7068263888e-01f
};

float4 tq_load_value(uint base, uint lane) {
    uint word = values.Load(base + (lane >> 1u)*4u);
    uint pairs = (word >> ((lane & 1u)*16u)) & 0xffffu;
    uint meta = values.Load(base + 128u);
    float scale = f16tof32(meta & 0xffffu);
    float minimum = f16tof32(meta >> 16u);
    float4 idx = float4(pairs & 15u, (pairs >> 4u) & 15u, (pairs >> 8u) & 15u, (pairs >> 12u) & 15u);
    return idx*scale + minimum;
}

[numthreads(TQ_THREADS, 1, 1)]
void CSRangeAttnTurboQuantNC(uint tid : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint kv_head = group.x, qi = group.y, split = group.z;
    const uint lane = WaveGetLaneIndex();
    const uint wave = tid >> 6u;
    const uint head0 = qi*n_head + kv_head*TQ_GQA;

    // The transform uses disjoint pairs at each stage, so every read/write
    // stays in its pair; the barrier publishes each stage to the next one.
    for (uint i = tid; i < TQ_GQA*TQ_DIM; i += TQ_THREADS) {
        uint g = i/TQ_DIM, d = i % TQ_DIM;
        tq_query[i] = asfloat(queries.Load(((head0 + g)*TQ_DIM + d)*4u));
    }
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = 1u; stride < TQ_DIM; stride <<= 1u) {
        for (uint pair = tid; pair < TQ_GQA*(TQ_DIM/2); pair += TQ_THREADS) {
            uint g = pair/(TQ_DIM/2), p = pair % (TQ_DIM/2);
            uint left = (p/stride)*(2u*stride) + p % stride;
            uint right = left + stride;
            float a = tq_query[g*TQ_DIM + left];
            float b = tq_query[g*TQ_DIM + right];
            tq_query[g*TQ_DIM + left] = a + b;
            tq_query[g*TQ_DIM + right] = a - b;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    for (uint i = tid; i < TQ_GQA*TQ_DIM; i += TQ_THREADS) tq_query[i] *= 0.0625f;
    GroupMemoryBarrierWithGroupSync();

    int query_pos = asint(positions.Load(qi*4u));
    uint causal_count = query_pos < int(origin) ? 0u : uint(query_pos) - origin + 1u;
    uint limit = min(active_count, causal_count);
    uint begin = min(limit, split*split_rows);
    uint end = min(limit, (split + 1u)*split_rows);
    float maximum[TQ_GQA], denom[TQ_GQA], score[TQ_GQA], weight[TQ_GQA];
    float4 acc[TQ_GQA];
    [unroll] for (uint g = 0; g < TQ_GQA; ++g) {
        maximum[g] = -3.402823466e38f;
        denom[g] = 0.0f;
        acc[g] = 0.0f;
    }

    for (uint row0 = begin + wave*64u; row0 < end; row0 += TQ_THREADS) {
        uint row = row0 + lane;
        bool valid = row < end;
        uint base = (row*n_head_kv + kv_head)*TQ_ROW_BYTES;
        float centroid_norm_sq = 0.0f;
        [unroll] for (uint g = 0; g < TQ_GQA; ++g) score[g] = 0.0f;
        if (valid) {
            [loop] for (uint wi = 0; wi < 32u; ++wi) {
                uint word = keys.Load(base + wi*4u);
                [unroll] for (uint b = 0; b < 4u; ++b) {
                    uint pair = (word >> (8u*b)) & 0xffu;
                    float c0 = tq_centroids[pair & 15u];
                    float c1 = tq_centroids[pair >> 4u];
                    uint dim = wi*8u + b*2u;
                    centroid_norm_sq += c0*c0 + c1*c1;
                    [unroll] for (uint g = 0; g < TQ_GQA; ++g) {
                        score[g] += tq_query[g*TQ_DIM + dim]*c0 + tq_query[g*TQ_DIM + dim + 1u]*c1;
                    }
                }
            }
            float norm = f16tof32(keys.Load(base + 128u) & 0xffffu);
            float factor = norm*rsqrt(centroid_norm_sq + 1e-16f)*kq_scale;
            [unroll] for (uint g = 0; g < TQ_GQA; ++g) score[g] *= factor;
        } else {
            [unroll] for (uint g = 0; g < TQ_GQA; ++g) score[g] = -3.402823466e38f;
        }
        [unroll] for (uint g = 0; g < TQ_GQA; ++g) {
            float tile_max = WaveActiveMax(score[g]);
            float next_max = max(maximum[g], tile_max);
            float old_weight = denom[g] == 0.0f ? 0.0f : exp(maximum[g] - next_max);
            weight[g] = valid ? exp(score[g] - next_max) : 0.0f;
            denom[g] = denom[g]*old_weight + WaveActiveSum(weight[g]);
            maximum[g] = next_max;
            acc[g] *= old_weight;
        }
        uint n_rows = min(64u, end - row0);
        [loop] for (uint r = 0; r < n_rows; ++r) {
            uint vbase = ((row0 + r)*n_head_kv + kv_head)*TQ_ROW_BYTES;
            float4 val = tq_load_value(vbase, lane);
            [unroll] for (uint g = 0; g < TQ_GQA; ++g) {
                // r is uniform across the wave; no divergent lane index is used.
                float p = WaveReadLaneAt(weight[g], r);
                acc[g] += p*val;
            }
        }
    }

    uint d0 = lane*4u;
    [unroll] for (uint g = 0; g < TQ_GQA; ++g) {
        uint slot = wave*TQ_GQA + g;
        if (lane == 0u) { tq_max[slot] = maximum[g]; tq_den[slot] = denom[g]; }
        tq_acc[slot*TQ_DIM + d0] = acc[g].x;
        tq_acc[slot*TQ_DIM + d0 + 1u] = acc[g].y;
        tq_acc[slot*TQ_DIM + d0 + 2u] = acc[g].z;
        tq_acc[slot*TQ_DIM + d0 + 3u] = acc[g].w;
    }
    GroupMemoryBarrierWithGroupSync();
    uint total = n_query*n_head;
    for (uint e = tid; e < TQ_GQA*TQ_DIM; e += TQ_THREADS) {
        uint g = e/TQ_DIM, d = e % TQ_DIM;
        float merged_max = -3.402823466e38f;
        [unroll] for (uint w = 0; w < TQ_WAVES; ++w) {
            if (tq_den[w*TQ_GQA + g] > 0.0f) merged_max = max(merged_max, tq_max[w*TQ_GQA + g]);
        }
        float merged_den = 0.0f, merged_acc = 0.0f;
        [unroll] for (uint w = 0; w < TQ_WAVES; ++w) {
            uint slot = w*TQ_GQA + g;
            if (tq_den[slot] > 0.0f) {
                float p = exp(tq_max[slot] - merged_max);
                merged_den += tq_den[slot]*p;
                merged_acc += tq_acc[slot*TQ_DIM + d]*p;
            }
        }
        uint qh = head0 + g;
        if (direct_output != 0u) {
            output.Store((qh*TQ_DIM + d)*4u, asuint(merged_den > 0.0f ? merged_acc/merged_den : 0.0f));
            if (d == 0u) {
                float lse = merged_den > 0.0f ? merged_max + log(merged_den) : asfloat(0xff800000u);
                output.Store((total*TQ_DIM + qh)*4u, asuint(lse));
            }
        } else {
            uint dst = ((split*total + qh)*(TQ_DIM + 2u))*4u;
            partials.Store(dst + d*4u, asuint(merged_acc));
            if (d == 0u) {
                partials.Store(dst + TQ_DIM*4u, asuint(merged_max));
                partials.Store(dst + (TQ_DIM + 1u)*4u, asuint(merged_den));
            }
        }
    }
}
