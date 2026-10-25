// RKVA KVarN flash attention split-K reduction shader.
// Combines partial attention results from split_k workgroups into final OUT.
//
// Each workgroup handles one (query, head) tile.
// numthreads(128, 1, 1): lane d processes dim d and dim d + 128 (head_dim = 256).

#include "ggml_float.hlsli"

cbuffer Params : register(b0) {
    uint n_tokens;   // queries in this call
    uint n_head;     // Q heads (24)
    uint head_dim;   // 256 (head_slices * 128)
    uint split_k;    // number of split partitions
};

ByteAddressBuffer   split_accum : register(t0);
RWByteAddressBuffer out_buf     : register(u0);

groupshared float sh_m_max;
groupshared float sh_inv_l;

float rkva_exp(float x) {
    if (x < -104.0f) {
        return 0.0f;
    }
    return ggml_exp_avx2(x);
}

[numthreads(128, 1, 1)]
void CSSplitKReduce(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint query = group.x;
    const uint head  = group.y;

    if (query >= n_tokens || head >= n_head) {
        return;
    }

    const uint total_q_heads = n_tokens * n_head;
    const uint q_head_idx    = query * n_head + head;

    // Layout of split_accum:
    // 1) O_part: [split_k][n_tokens][n_head][head_dim] floats
    // 2) lm_part: [split_k][n_tokens][n_head][2] floats (m_part, l_part)
    const uint o_part_stride  = total_q_heads * head_dim;
    const uint lm_part_offset = split_k * o_part_stride * 4; // bytes after all O_parts
    const uint lm_part_stride = total_q_heads * 2;

    if (lane == 0) {
        float m_max = -3.402823466e38f;
        for (uint k = 0; k < split_k; ++k) {
            const uint lm_idx = k * lm_part_stride + q_head_idx * 2;
            const float m_k = asfloat(split_accum.Load(lm_part_offset + lm_idx * 4));
            m_max = max(m_max, m_k);
        }

        float L = 0.0f;
        if (m_max > -3.0e38f) {
            for (uint k = 0; k < split_k; ++k) {
                const uint lm_idx = k * lm_part_stride + q_head_idx * 2;
                const float m_k = asfloat(split_accum.Load(lm_part_offset + lm_idx * 4));
                const float l_k = asfloat(split_accum.Load(lm_part_offset + (lm_idx + 1) * 4));
                L += rkva_exp(m_k - m_max) * l_k;
            }
        }

        sh_m_max = m_max;
        sh_inv_l = (L > 0.0f) ? (1.0f / L) : 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    const float m_max = sh_m_max;
    const float inv_l = sh_inv_l;

    if (m_max <= -3.0e38f || inv_l == 0.0f) {
        // Zero output if no attention tokens
        const uint dst_base = q_head_idx * head_dim;
        for (uint d = lane; d < head_dim; d += 128) {
            out_buf.Store((dst_base + d) * 4, 0);
        }
        return;
    }

    const uint dst_base = q_head_idx * head_dim;
    for (uint d = lane; d < head_dim; d += 128) {
        float o_acc = 0.0f;
        for (uint k = 0; k < split_k; ++k) {
            const uint lm_idx = k * lm_part_stride + q_head_idx * 2;
            const float m_k = asfloat(split_accum.Load(lm_part_offset + lm_idx * 4));
            const float weight = rkva_exp(m_k - m_max);

            const uint o_idx = k * o_part_stride + q_head_idx * head_dim + d;
            const float o_k = asfloat(split_accum.Load(o_idx * 4));
            o_acc += weight * o_k;
        }

        const float final_o = o_acc * inv_l;
        out_buf.Store((dst_base + d) * 4, asuint(final_o));
    }
}
