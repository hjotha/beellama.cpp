// RKVA flash attention — causal GQA over an F16 KV cache (Xbox Series X, RDNA2).
//
// Bring-up / correctness path and the materialize fallback for the KVarN
// engine. Computes standard scaled-dot-product causal attention with GQA,
// matching the host F16-cache flash-attention semantics closely enough for
// greedy token identity at short/moderate context (parity is measured, not
// assumed). The KVarN native-over-records kernel replaces this for long
// context; this kernel stays as the correctness oracle.
//
// Geometry (qwen35 target): n_head=24 Q heads, n_head_kv=4 KV heads
// (6 Q heads per KV head), head_dim=256, full attention (no SWA), no sinks.
//
// One thread group owns one (query, q_head). Lane d holds head dimension d.
// Dispatch: grid (n_head, query_count, 1), numthreads(256,1,1).
//
// KV layout (per layer, F16), all little-endian:
//   K: [capacity][n_head_kv][head_dim]
//   V: [capacity][n_head_kv][head_dim]
// The two are separate ByteAddressBuffers (t1=K, t2=V) already offset to the
// layer by the engine, so this shader indexes from position 0.
//
// Valid KV positions are contiguous [0, active_count) — guaranteed by the
// append-only + suffix-trim/rewind MVP semantics — so the kernel iterates
// positions directly and applies the causal mask, with no index buffer.
//
// b0 params, t0 queries (F32), t1 K (F16), t2 V (F16), t3 query_positions
// (i32), u0 output (F32).

#include "ggml_float.hlsli"

// exp with F32 underflow guard. The first online-softmax step evaluates
// exp(-inf - score); ggml_exp_avx2 does not clamp -inf, so guard explicitly
// (matches the DFlash ggml_exp_neg threshold: exp(x) underflows to 0 below
// about -104 in F32).
float rkva_exp(float x) {
    if (x < -104.0f) {
        return 0.0f;
    }
    return ggml_exp_avx2(x);
}

cbuffer Params : register(b0) {
    uint  query_count;
    uint  active_count;
    uint  n_head;        // 24
    uint  n_head_kv;     // 4
    uint  head_dim;      // 256
    float kq_scale;      // 1/sqrt(head_dim)
    uint  kv_capacity;
    uint  reserved;
};

ByteAddressBuffer   queries          : register(t0); // F32 [query][n_head][head_dim]
ByteAddressBuffer   keys             : register(t1); // F16 [capacity][n_head_kv][head_dim]
ByteAddressBuffer   values           : register(t2); // F16 [capacity][n_head_kv][head_dim]
ByteAddressBuffer   query_positions  : register(t3); // i32 [query]
RWByteAddressBuffer output           : register(u0); // F32 [query][n_head][head_dim]

groupshared float s_partial[256];

[numthreads(256, 1, 1)]
void CSFlashAttn(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint head = group.x;   // q head
    const uint qi   = group.y;   // query index
    if (head >= n_head || qi >= query_count || lane >= head_dim) {
        return;
    }
    const uint kv_head = head / (n_head / n_head_kv);  // 6 Q heads per KV head

    const int query_position = query_positions.Load(qi * 4);

    // Q for this (query, head, lane) — F32.
    const float q = asfloat(queries.Load(((qi * n_head + head) * head_dim + lane) * 4));

    float max_score = asfloat(0xff800000u);  // -inf
    float denom     = 0.0f;
    float acc       = 0.0f;                  // numerator for out[lane]

    const uint kv_row_bytes = head_dim * 2;  // F16 row stride per (pos, kv_head)
    const uint lane_word = (lane >> 1) * 4;  // byte offset of this lane's packed word
    const bool lane_high = (lane & 1u) != 0u;
    // Causal: a query only attends to positions <= its own. Clamp the loop
    // bound so decode (query at the frontier) still scans the whole prefix.
    const uint limit = min(active_count, uint(query_position) + 1u);
    [loop]
    for (uint pos = 0; pos < limit; ++pos) {
        const uint base = (pos * n_head_kv + kv_head) * kv_row_bytes;
        // K/V are F16 packed two dims per 32-bit word; pick this lane's half.
        const uint k_word = keys.Load(base + lane_word);
        const float k_d = f16tof32(lane_high ? (k_word >> 16) : (k_word & 0xffffu));

        // score = sum_d q[d]*k[d], reduced across the 256 lanes.
        s_partial[lane] = q * k_d;
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint step = 128u; step > 0u; step >>= 1) {
            if (lane < step) {
                s_partial[lane] += s_partial[lane + step];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        const float score = s_partial[0] * kq_scale;
        GroupMemoryBarrierWithGroupSync();  // protect s_partial before next iter

        // Online softmax (ggml-faithful rescaling).
        float old_weight = 1.0f;
        float new_weight = 1.0f;
        if (score > max_score) {
            old_weight = rkva_exp(max_score - score);
            acc   *= old_weight;
            denom  = denom * old_weight + 1.0f;
            max_score = score;
        } else {
            new_weight = rkva_exp(score - max_score);
            denom += new_weight;
        }
        const uint v_word = values.Load(base + lane_word);
        const float v_d = f16tof32(lane_high ? (v_word >> 16) : (v_word & 0xffffu));
        acc = ggml_fma_f32(v_d, new_weight, acc);
    }

    const float inv = denom == 0.0f ? 0.0f : ggml_round_f32(1.0 / (double) denom);
    const float result = acc * inv;
    output.Store(((qi * n_head + head) * head_dim + lane) * 4, asuint(result));
}
