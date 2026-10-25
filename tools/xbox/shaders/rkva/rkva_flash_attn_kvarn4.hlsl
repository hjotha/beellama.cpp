// RKVA KVarN flash attention — causal GQA over 4-bit records + F16 stage.
//
// Port of ggml-vulkan kvarn_flash_attn.comp for the RKVA engine, with:
// - single stream (stream 0), dense contiguous positions
// - stage F16 buffer acting as tail_groups (256-token lossless tail) + sink (group 0)
// - split_k parallel partitioning for saturated 52-CU Xbox execution
// - WaveActiveSum (Wave32) reduction for low-latency dot products
// - rotated domain: the engine applies the inverse WHT on the output.
//
// Grid: (n_query, n_kv_heads * gqa_groups, split_k), numthreads(128, 1, 1).
//
// b0 params:
// t0 = Q rotated F32 [n_tokens][n_head][head_dim],
// t1 = K records, t2 = K stage, t3 = V records, t4 = V stage,
// t5 = query_positions (i32 [n_tokens]), u0 = OUT F32 or split_accum.

#include "ggml_float.hlsli"

float rkva_exp(float x) {
    if (x < -104.0f) {
        return 0.0f;
    }
    return ggml_exp_avx2(x);
}

cbuffer Params : register(b0) {
    uint n_head;           // Q heads (24)
    uint n_head_kv;        // KV heads (4)
    uint n_heads_phys;     // physical heads = n_head_kv*head_slices (8)
    uint head_dim;         // 256
    uint n_query;          // queries in this call
    uint n_kv;             // stored positions (active_count)
    uint groups_per_stream;
    uint bits;             // 4
    uint head_slices;      // 2
    uint stage_groups;     // 3
    uint tail_groups;      // 2
    uint record_words;     // 2240
    float kq_scale;        // 0.0625
    uint split_k;          // 1 default, >1 for split-K decode
};

ByteAddressBuffer   q_rot    : register(t0); // F32 [n_tokens][n_head][head_dim]
ByteAddressBuffer   k_records: register(t1);
ByteAddressBuffer   k_stage  : register(t2);
ByteAddressBuffer   v_records: register(t3);
ByteAddressBuffer   v_stage  : register(t4);
ByteAddressBuffer   positions: register(t5); // i32 [n_tokens]
RWByteAddressBuffer out_buf  : register(u0);

static const uint KVAR_N_DIM = 128;
static const uint GQA_GROUP_MAX = 4;

groupshared float sh_wave_sum[4];
groupshared float sh_maximum[GQA_GROUP_MAX];
groupshared float sh_denominator[GQA_GROUP_MAX];
groupshared float sh_old_scale[GQA_GROUP_MAX];
groupshared float sh_weight[GQA_GROUP_MAX];

uint unpack_record(ByteAddressBuffer rec, uint record_base, uint index) {
    const uint bit_offset = index * bits;
    const uint word_index = bit_offset >> 5;
    const uint bit = bit_offset & 31;
    uint result = rec.Load((record_base + word_index) * 4) >> bit;
    if (bit + bits > 32) {
        result |= rec.Load((record_base + word_index + 1) * 4) << (32 - bit);
    }
    return result & ((1u << bits) - 1u);
}

float load_axis(uint record_base, uint payload_words, uint axis, uint index, bool value) {
    const uint word = value ?
        v_records.Load((record_base + payload_words + axis * 64 + (index >> 1)) * 4) :
        k_records.Load((record_base + payload_words + axis * 64 + (index >> 1)) * 4);
    const uint bits16 = (index & 1u) == 0u ? (word & 0xffffu) : (word >> 16u);
    return f16tof32(bits16);
}

float load_stage(bool value, uint stage_pos, uint head, uint dim) {
    const uint index = (stage_pos * n_heads_phys + head) * 64 + (dim >> 1);
    const uint word = value ?
        v_stage.Load(index * 4) : k_stage.Load(index * 4);
    const uint bits16 = (dim & 1u) == 0u ? (word & 0xffffu) : (word >> 16u);
    return f16tof32(bits16);
}

// Delayed (eager=0) record decision with dense positions:
//   from_stage = group 0 (sink) or the last tail_groups groups near live;
//   from_record = flushed groups below that.
float load_kvarn(bool value, uint logical_head, uint token, uint live_group,
                 uint live_pos, uint slice, uint dim) {
    const uint group = token / KVAR_N_DIM;
    const uint pos = token % KVAR_N_DIM;
    bool from_stage = false;
    bool from_record = false;
    from_stage = group == 0 ||
        (group > 0 && group <= live_group && group + (tail_groups - 1) >= live_group);
    from_record = !from_stage && group < live_group && group < groups_per_stream;

    const uint head = logical_head * head_slices + slice;
    if (from_stage) {
        const uint stage_slot = group == 0 ? 0 : 1 + ((group - 1) % tail_groups);
        const uint stage_pos = stage_slot * KVAR_N_DIM + pos;  // stream 0
        return load_stage(value, stage_pos, head, dim);
    }
    if (from_record) {
        const uint record_words_local = record_words;
        const uint record_base = (group * n_heads_phys + head) * record_words_local;  // stream 0
        const uint row = value ? pos : dim;
        const uint col = value ? dim : pos;
        const uint payload_words = bits * 512;
        const uint quant = unpack_record(value ? v_records : k_records,
                                         record_base, row * KVAR_N_DIM + col);
        const float scale = load_axis(record_base, payload_words, 0, row, value);
        const float zero = load_axis(record_base, payload_words, 1, row, value);
        const float other = load_axis(record_base, payload_words, 2, col, value);
        return (float(quant) * scale + zero) * other;
    }
    return 0.0;
}

[numthreads(128, 1, 1)]
void CSFlashAttnKvarn(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint query = group.x;
    const uint gqa = n_head / n_head_kv;  // 6
    const uint gqa_group_size = min(gqa, GQA_GROUP_MAX);
    const uint gqa_groups = (gqa + gqa_group_size - 1) / gqa_group_size;
    const uint logical_head = group.y / gqa_groups;
    const uint gqa_group = group.y % gqa_groups;
    const uint query_head_first = logical_head * gqa + gqa_group * gqa_group_size;
    const uint active_heads = min(gqa_group_size, gqa - gqa_group * gqa_group_size);
    const uint split_idx = group.z;
    const uint eff_split_k = max(split_k, 1u);

    if (query >= n_query || logical_head >= n_head_kv) {
        return;
    }

    const int q_pos = positions.Load(query * 4);
    const uint limit = min(n_kv, uint(q_pos) + 1u);
    const uint live = n_kv;  // dense [0, n_kv)
    const uint live_group = live > 0 ? (live - 1) / KVAR_N_DIM : 0;
    const uint live_pos = live > 0 ? (live - 1) % KVAR_N_DIM : 0;

    const uint token_start = (limit * split_idx) / eff_split_k;
    const uint token_end   = (limit * (split_idx + 1u)) / eff_split_k;

    float accumulator[GQA_GROUP_MAX * 4];
    for (uint i = 0; i < GQA_GROUP_MAX * 4; ++i) {
        accumulator[i] = 0.0f;
    }
    if (lane == 0) {
        for (uint gi = 0; gi < active_heads; ++gi) {
            sh_maximum[gi] = -3.402823466e38f;
            sh_denominator[gi] = 0.0f;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint token = token_start; token < token_end; ++token) {
        float k_values[2];
        float v_values[2];
        for (uint slice = 0; slice < head_slices; ++slice) {
            k_values[slice] = load_kvarn(false, logical_head, token, live_group, live_pos, slice, lane);
            v_values[slice] = load_kvarn(true, logical_head, token, live_group, live_pos, slice, lane);
        }

        for (uint gi = 0; gi < active_heads; ++gi) {
            const uint query_head = query_head_first + gi;
            float partial = 0.0f;
            for (uint slice = 0; slice < head_slices; ++slice) {
                const uint dim = slice * KVAR_N_DIM + lane;
                const uint q_idx = (query * n_head + query_head) * head_dim + dim;
                partial += asfloat(q_rot.Load(q_idx * 4)) * k_values[slice];
            }

            // WaveActiveSum across 32 lanes in wave
            float wave_sum = WaveActiveSum(partial);
            if (WaveIsFirstLane()) {
                sh_wave_sum[lane >> 5] = wave_sum;
            }
            GroupMemoryBarrierWithGroupSync();

            float total = 0.0f;
            if (lane < 4) {
                total = sh_wave_sum[lane];
            }
            total = WaveActiveSum(total);

            if (lane == 0) {
                float score = total * kq_scale;
                const float next_maximum = max(sh_maximum[gi], score);
                const float old_scale = sh_maximum[gi] < -3.0e38f ?
                    0.0f : rkva_exp(sh_maximum[gi] - next_maximum);
                const float weight = rkva_exp(score - next_maximum);
                sh_maximum[gi] = next_maximum;
                sh_denominator[gi] = sh_denominator[gi] * old_scale + weight;
                sh_old_scale[gi] = old_scale;
                sh_weight[gi] = weight;
            }
            GroupMemoryBarrierWithGroupSync();

            for (uint slice = 0; slice < head_slices; ++slice) {
                const uint acc = gi * 4 + slice;
                accumulator[acc] = accumulator[acc] * sh_old_scale[gi] +
                    v_values[slice] * sh_weight[gi];
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (eff_split_k <= 1) {
        // Direct single-pass output
        if (lane == 0) {
            for (uint gi = 0; gi < active_heads; ++gi) {
                sh_old_scale[gi] = 1.0f;
                sh_weight[gi] = sh_denominator[gi] > 0.0f ? 1.0f / sh_denominator[gi] : 0.0f;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint gi = 0; gi < active_heads; ++gi) {
            const uint query_head = query_head_first + gi;
            const uint dst_base = (query * n_head + query_head) * head_dim;
            for (uint slice = 0; slice < head_slices; ++slice) {
                const uint dim = slice * KVAR_N_DIM + lane;
                out_buf.Store((dst_base + dim) * 4, asuint(
                    accumulator[gi * 4 + slice] * sh_old_scale[gi] * sh_weight[gi]));
            }
        }
    } else {
        // Multi-pass split_k output: write unnormalized accumulator and (m_part, l_part)
        const uint total_q_heads = n_query * n_head;
        const uint o_part_stride = total_q_heads * head_dim;
        const uint lm_part_offset = eff_split_k * o_part_stride * 4;
        const uint lm_part_stride = total_q_heads * 2;

        for (uint gi = 0; gi < active_heads; ++gi) {
            const uint query_head = query_head_first + gi;
            const uint q_head_idx = query * n_head + query_head;
            const uint o_base = split_idx * o_part_stride + q_head_idx * head_dim;

            for (uint slice = 0; slice < head_slices; ++slice) {
                const uint dim = slice * KVAR_N_DIM + lane;
                out_buf.Store((o_base + dim) * 4, asuint(accumulator[gi * 4 + slice]));
            }

            if (lane == 0) {
                const uint lm_idx = split_idx * lm_part_stride + q_head_idx * 2;
                out_buf.Store(lm_part_offset + lm_idx * 4, asuint(sh_maximum[gi]));
                out_buf.Store(lm_part_offset + (lm_idx + 1) * 4, asuint(sh_denominator[gi]));
            }
        }
    }
}