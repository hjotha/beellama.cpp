// Port of llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp.
// Qwen3.5 scalar activated gates, gathered F32 state; no KDA/raw/rows modes.
// State[j*128+i] is transposed. Tokens run chronologically inside each group.
// Slot 0 = final state; slot j = j tokens back, exactly as the CPU GDN op.

cbuffer Params : register(b0) {
    uint n_tokens, K, q_head_stride, q_token_stride;
    uint k_head_stride, k_token_stride, v_head_stride, v_token_stride;
    uint s_off, snapshot_size, H_v, H_k;
    float scale;
    uint pad0, pad1, pad2;
};

// D3D12_Host resources stay in UNORDERED_ACCESS, including the inputs.
RWStructuredBuffer<float> Q : register(u0);
RWStructuredBuffer<float> Keys : register(u1);
RWStructuredBuffer<float> V : register(u2);
RWStructuredBuffer<float> G : register(u3);
RWStructuredBuffer<float> Beta : register(u4);
RWStructuredBuffer<float> State : register(u5);
RWStructuredBuffer<float> Y : register(u6);

groupshared float red[64];

float sum_column(float value, uint tid) {
    const uint lane = tid & 7u;
    red[tid] = value;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 4u; stride > 0u; stride >>= 1u) {
        if (lane < stride)
            red[tid] += red[tid + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    const float result = red[tid - lane];
    GroupMemoryBarrierWithGroupSync();
    return result;
}

[numthreads(64, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint head = gid.x;
    const uint lane = tid & 7u;
    const uint col = gid.z * 8u + (tid >> 3);
    const uint state_base = head * 128u * 128u + col * 128u;
    precise float s[16];
    [unroll]
    for (uint r = 0; r < 16u; ++r)
        s[r] = State[state_base + r * 8u + lane];

    for (uint t = 0; t < n_tokens; ++t) {
        const uint q_off = (head % H_k) * q_head_stride + t * q_token_stride;
        const uint k_off = (head % H_k) * k_head_stride + t * k_token_stride;
        const float decay = exp(G[t * H_v + head]);
        const float beta = Beta[t * H_v + head];
        float q[16], k[16];
        precise float kv = 0.0;
        [unroll]
        for (uint r = 0; r < 16u; ++r) {
            const uint i = r * 8u + lane;
            q[r] = Q[q_off + i];
            k[r] = Keys[k_off + i];
            s[r] *= decay;
            kv += s[r] * k[r];
        }
        const float delta = (V[t * v_token_stride + head * v_head_stride + col] -
                             sum_column(kv, tid)) * beta;
        precise float attn = 0.0;
        [unroll]
        for (uint r = 0; r < 16u; ++r) {
            s[r] += k[r] * delta;
            attn += s[r] * q[r];
        }
        const float score = sum_column(attn, tid);
        if (lane == 0u)
            Y[(t * H_v + head) * 128u + col] = score * scale;

        const uint slot = n_tokens - 1u - t;
        if ((K > 1u && slot < K) || (K == 1u && t + 1u == n_tokens)) {
            const uint base = s_off + (K == 1u ? 0u : slot) * snapshot_size + state_base;
            [unroll]
            for (uint r = 0; r < 16u; ++r)
                Y[base + r * 8u + lane] = s[r];
        }
    }
}
