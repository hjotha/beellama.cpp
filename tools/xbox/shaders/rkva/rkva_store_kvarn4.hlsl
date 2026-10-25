// RKVA KVarN store — WHT rotation + F16 stage + delayed Sinkhorn record flush.
//
// Port of ggml-vulkan kvarn_store.comp MODE_MONOLITHIC (eager_records=0) for
// the RKVA engine. The host sends K/V F32 rows (post-RoPE, pre-WHT) at absolute
// positions; this kernel performs the per-token 256-WHT, stores the rotated
// F16 into the stage, and lazily quantizes a group to 4-bit records (Sinkhorn
// + LSB-first payload + scale/zp/other axes) when a later group boundary opens.
//
// RKVA simplifications vs the Vulkan kernel:
//   - single stream (stream 0): positions are absolute within the session
//     capacity, group = pos/128, pos_in_group = pos%128
//   - positions arrive as an i32 buffer instead of a packed i64 index buffer
//   - eager_records=0 (delayed flush) only, tail_groups=2, stage_groups=3
//   - SWA path omitted (qwen35 full attention)
//
// Grid: (n_head_kv, 1, 1) = one workgroup per logical KV head (4); each
// workgroup owns head_slices (2) physical heads and loops tokens serially.
// numthreads(128, 1, 1).
//
// b0 params, t0 = positions (i32), t1 = current K/V F32 rows
// [n_tokens][n_head_kv][head_dim] (wire layout), u0 = stage (F16 u32 words),
// u1 = records (u32 words). One PSO serves K and V; the `value` constant
// (0=K,1=V) flips the tile orientation like the Vulkan push constant.

#include "ggml_float.hlsli"

// GGML F32->F16 round-to-nearest-even (bit-exact with ggml_fp32_to_fp16),
// shared with the F16 store kernel.
uint ggml_half_bits(float value) {
    const uint bits = asuint(value);
    const uint sign = (bits >> 16) & 0x8000u;
    const uint exponent = (bits >> 23) & 255u;
    uint mantissa = bits & 0x7fffffu;
    if (exponent == 255u) {
        return sign | (mantissa == 0 ? 0x7c00u : 0x7e00u);
    }
    int half_exponent = int(exponent) - 127 + 15;
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        mantissa |= 0x800000u;
        const uint shift = uint(14 - half_exponent);
        uint rounded = mantissa >> shift;
        const uint remainder = mantissa & ((1u << shift) - 1u);
        const uint halfway = uint(1u << (shift - 1u));
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) {
            ++rounded;
        }
        return sign | rounded;
    }
    if (half_exponent >= 31) return sign | 0x7c00u;
    uint rounded = mantissa >> 13;
    const uint remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        ++rounded;
    }
    if (rounded == 0x400u) {
        rounded = 0;
        ++half_exponent;
    }
    if (half_exponent >= 31) return sign | 0x7c00u;
    return sign | (uint(half_exponent) << 10) | rounded;
}

cbuffer Params : register(b0) {
    uint n_heads;          // physical heads = n_head_kv * head_slices (8)
    uint n_tokens;
    uint groups_per_stream;
    uint record_words;     // bits*512 + 192 (2240 for bits=4)
    uint bits;             // 4
    uint iterations;       // Sinkhorn iterations (16)
    uint value;            // 0 = K, 1 = V
    uint head_slices;      // 2
    uint stage_groups;     // tail_groups + 1 (3)
    uint tail_groups;      // 2
    uint head_dim;         // 256
};

ByteAddressBuffer   positions : register(t0); // i32 [n_tokens], absolute positions
ByteAddressBuffer   current   : register(t1); // F32 [n_tokens][n_head_kv][head_dim]
RWByteAddressBuffer stage     : register(u0); // F16 u32 words
RWByteAddressBuffer records   : register(u1); // packed records u32 words

static const uint KVAR_N_DIM = 128;
static const float KVAR_N_WHT_SCALE = 0.08838834764831845f;

groupshared float sh_wht[128];
groupshared float sh_log_s_col[128];
groupshared float sh_log_s_row[128];
groupshared float sh_best_col[128];
groupshared float sh_best_row[128];
groupshared float sh_col_std[128];
groupshared float sh_row_std[128];
groupshared float sh_best_imbalance;
groupshared uint  sh_better;

uint stage_word_index(uint stage_pos, uint head, uint dim_pair) {
    return (stage_pos * n_heads + head) * (KVAR_N_DIM / 2) + dim_pair;
}

float load_stage_value(uint stage_pos, uint head, uint dim) {
    const uint word = stage.Load(stage_word_index(stage_pos, head, dim >> 1) * 4);
    const uint bits16 = (dim & 1u) == 0u ? (word & 0xffffu) : (word >> 16u);
    return f16tof32(bits16);
}

void store_stage_pair(uint stage_pos, uint head, uint lane) {
    if (lane < KVAR_N_DIM / 2) {
        const uint lo = ggml_half_bits(sh_wht[2 * lane]) & 0xffffu;
        const uint hi = ggml_half_bits(sh_wht[2 * lane + 1]) & 0xffffu;
        stage.Store(stage_word_index(stage_pos, head, lane) * 4, lo | (hi << 16));
    }
}

float wht_stage_value(float value, uint lane) {
    [unroll]
    for (uint stride = 1; stride < KVAR_N_DIM; stride <<= 1) {
        sh_wht[lane] = value;
        GroupMemoryBarrierWithGroupSync();
        const float other = sh_wht[lane ^ stride];
        GroupMemoryBarrierWithGroupSync();
        value = (lane & stride) == 0u ? value + other : other - value;
    }
    return value * KVAR_N_WHT_SCALE;
}

float stage_matrix_value(uint head, uint stage_base, uint stage_slot, uint row, uint col) {
    const uint token = value != 0u ? row : col;
    const uint out_dim = value != 0u ? col : row;
    const uint stage_pos = stage_base + stage_slot * KVAR_N_DIM + token;
    return load_stage_value(stage_pos, head, out_dim);
}

float std_col(uint head, uint stage_base, uint stage_slot, uint col) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const float sc = exp(sh_log_s_col[col]);
    for (uint row = 0; row < KVAR_N_DIM; ++row) {
        const float raw = stage_matrix_value(head, stage_base, stage_slot, row, col);
        const float scaled = raw / (sc * exp(sh_log_s_row[row]));
        sum += scaled;
        sum_sq += scaled * scaled;
    }
    const float mean = sum / float(KVAR_N_DIM);
    return sqrt(max((sum_sq - float(KVAR_N_DIM) * mean * mean) / float(KVAR_N_DIM - 1), 0.0));
}

float std_row(uint head, uint stage_base, uint stage_slot, uint row) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const float sr = exp(sh_log_s_row[row]);
    for (uint col = 0; col < KVAR_N_DIM; ++col) {
        const float raw = stage_matrix_value(head, stage_base, stage_slot, row, col);
        const float scaled = raw / (exp(sh_log_s_col[col]) * sr);
        sum += scaled;
        sum_sq += scaled * scaled;
    }
    const float mean = sum / float(KVAR_N_DIM);
    return sqrt(max((sum_sq - float(KVAR_N_DIM) * mean * mean) / float(KVAR_N_DIM - 1), 0.0));
}

void update_best(uint head, uint stage_base, uint stage_slot, uint lane) {
    sh_col_std[lane] = std_col(head, stage_base, stage_slot, lane);
    sh_row_std[lane] = std_row(head, stage_base, stage_slot, lane);
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0) {
        float col_min = sh_col_std[0];
        float col_max = sh_col_std[0];
        float row_min = sh_row_std[0];
        float row_max = sh_row_std[0];
        for (uint i = 1; i < KVAR_N_DIM; ++i) {
            col_min = min(col_min, sh_col_std[i]);
            col_max = max(col_max, sh_col_std[i]);
            row_min = min(row_min, sh_row_std[i]);
            row_max = max(row_max, sh_row_std[i]);
        }
        const float imbalance = col_max / max(col_min, 1e-8) + row_max / max(row_min, 1e-8);
        sh_better = imbalance <= sh_best_imbalance ? 1u : 0u;
        if (sh_better != 0u) {
            sh_best_imbalance = imbalance;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (sh_better != 0u) {
        sh_best_col[lane] = exp(sh_log_s_col[lane]);
        sh_best_row[lane] = exp(sh_log_s_row[lane]);
    }
    GroupMemoryBarrierWithGroupSync();
}

// CUDA roundf = round half away from zero; here x-lo >= 0, so floor(x+0.5).
uint quant_value(float x, float lo, float scale, uint qmax) {
    return uint(clamp(floor((x - lo) / scale + 0.5f), 0.0, float(qmax)));
}

void write_payload_bits(uint record_base, uint row, uint qmax, float lo, float scale,
                        uint head, uint stage_base, uint stage_slot) {
    const uint row_words = bits * 4;
    const uint row_base = record_base + row * row_words;
    for (uint w = 0; w < row_words; ++w) {
        records.Store((row_base + w) * 4, 0u);
    }

    for (uint col = 0; col < KVAR_N_DIM; ++col) {
        const float raw = stage_matrix_value(head, stage_base, stage_slot, row, col);
        const float x = raw / (sh_best_col[col] * sh_best_row[row]);
        const uint q = quant_value(x, lo, scale, qmax);
        const uint bit_offset = col * bits;
        const uint word_index = bit_offset >> 5;
        const uint bit = bit_offset & 31;
        uint word = records.Load((row_base + word_index) * 4);
        word |= q << bit;
        records.Store((row_base + word_index) * 4, word);
        if (bit + bits > 32) {
            uint carry = records.Load((row_base + word_index + 1) * 4);
            carry |= q >> (32 - bit);
            records.Store((row_base + word_index + 1) * 4, carry);
        }
    }
}

void quantize_stage(uint head, uint stage_base, uint stage_slot, uint record_base, uint lane) {
    sh_log_s_col[lane] = 0.0;
    sh_log_s_row[lane] = 0.0;
    sh_best_col[lane] = 1.0;
    sh_best_row[lane] = 1.0;
    GroupMemoryBarrierWithGroupSync();

    sh_col_std[lane] = std_col(head, stage_base, stage_slot, lane);
    sh_row_std[lane] = std_row(head, stage_base, stage_slot, lane);
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0) {
        float col_min = sh_col_std[0];
        float col_max = sh_col_std[0];
        float row_min = sh_row_std[0];
        float row_max = sh_row_std[0];
        for (uint i = 1; i < KVAR_N_DIM; ++i) {
            col_min = min(col_min, sh_col_std[i]);
            col_max = max(col_max, sh_col_std[i]);
            row_min = min(row_min, sh_row_std[i]);
            row_max = max(row_max, sh_row_std[i]);
        }
        sh_best_imbalance = col_max / max(col_min, 1e-8) + row_max / max(row_min, 1e-8);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint iter = 0; iter < iterations; ++iter) {
        const float col_std = clamp(std_col(head, stage_base, stage_slot, lane), 1e-3, 1e3);
        sh_log_s_col[lane] = clamp(sh_log_s_col[lane] + log(col_std), -0.3, 10.0);
        GroupMemoryBarrierWithGroupSync();

        const float row_std = clamp(std_row(head, stage_base, stage_slot, lane), 1e-3, 1e3);
        sh_log_s_row[lane] = clamp(sh_log_s_row[lane] + log(row_std), -0.3, 10.0);
        GroupMemoryBarrierWithGroupSync();

        update_best(head, stage_base, stage_slot, lane);
    }

    float lo = 3.402823466e+38;
    float hi = -3.402823466e+38;
    for (uint col = 0; col < KVAR_N_DIM; ++col) {
        const float raw = stage_matrix_value(head, stage_base, stage_slot, lane, col);
        const float x = raw / (sh_best_col[col] * sh_best_row[lane]);
        lo = min(lo, x);
        hi = max(hi, x);
    }

    const uint qmax = (1u << bits) - 1u;
    const float scale = max((hi - lo) / float(qmax), 1e-10);
    write_payload_bits(record_base, lane, qmax, lo, scale, head, stage_base, stage_slot);

    sh_col_std[lane] = scale;
    sh_row_std[lane] = lo;
    GroupMemoryBarrierWithGroupSync();

    const uint payload_words = bits * 512;
    const uint axis_word = lane;
    if (axis_word < KVAR_N_DIM / 2) {
        const uint r0 = 2 * axis_word;
        const uint r1 = r0 + 1;
        const uint s0 = ggml_half_bits(sh_best_row[r0] * sh_col_std[r0]) & 0xffffu;
        const uint s1 = ggml_half_bits(sh_best_row[r1] * sh_col_std[r1]) & 0xffffu;
        const uint z0 = ggml_half_bits(sh_best_row[r0] * sh_row_std[r0]) & 0xffffu;
        const uint z1 = ggml_half_bits(sh_best_row[r1] * sh_row_std[r1]) & 0xffffu;
        const uint o0 = ggml_half_bits(sh_best_col[r0]) & 0xffffu;
        const uint o1 = ggml_half_bits(sh_best_col[r1]) & 0xffffu;
        records.Store((record_base + payload_words + axis_word) * 4, s0 | (s1 << 16));
        records.Store((record_base + payload_words + 64 + axis_word) * 4, z0 | (z1 << 16));
        records.Store((record_base + payload_words + 128 + axis_word) * 4, o0 | (o1 << 16));
    }
    GroupMemoryBarrierWithGroupSync();
}

[numthreads(128, 1, 1)]
void CSStore(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint head0 = group.x * head_slices;
    const uint n_head_kv = n_heads / head_slices;
    if (head0 + head_slices > n_heads) {
        return;
    }
    const uint stage_base = 0;  // single stream

    for (uint token = 0; token < n_tokens; ++token) {
        const int pos_abs = positions.Load(token * 4);
        if (pos_abs < 0) {
            continue;
        }
        const uint idx = uint(pos_abs);
        const uint group_global = idx / KVAR_N_DIM;
        const uint pos = idx % KVAR_N_DIM;
        if (group_global >= groups_per_stream) {
            continue;
        }

        // Delayed flush: a token opening group g seals group g-tail_groups.
        if (pos == 0 && group_global > tail_groups) {
            const uint flush_group = group_global - tail_groups;
            for (uint slice = 0; slice < head_slices; ++slice) {
                const uint head = head0 + slice;
                const uint record_base = (flush_group * n_heads + head) * record_words;
                const uint flush_slot = 1u + ((flush_group - 1u) % tail_groups);
                quantize_stage(head, stage_base, flush_slot, record_base, lane);
            }
        }

        // Rotate this token (per-slice 128-WHT + cross-slice mix), store F16.
        float values[2] = {0.0f, 0.0f};
        for (uint slice = 0; slice < head_slices; ++slice) {
            const uint head = head0 + slice;
            // current wire layout: [n_tokens][n_head_kv][head_dim]; with
            // n_heads = n_head_kv*head_slices the physical-head index maps to
            // (token*n_heads + head)*128 + lane == token*1024 + kv_head*256 +
            // slice*128 + lane, which is the same address as the wire layout.
            const float c = asfloat(current.Load((token * n_heads + head) * KVAR_N_DIM * 4 + lane * 4));
            values[slice] = wht_stage_value(c, lane);
        }
        if (head_slices == 2) {
            const float a = values[0];
            const float b = values[1];
            values[0] = (a + b) * 0.7071067811865475;
            values[1] = (a - b) * 0.7071067811865475;
        }

        const uint fallback_slot = group_global == 0u ? 0u : 1u + ((group_global - 1u) % tail_groups);
        const uint stage_pos = stage_base + fallback_slot * KVAR_N_DIM + pos;
        for (uint slice = 0; slice < head_slices; ++slice) {
            const uint head = head0 + slice;
            sh_wht[lane] = values[slice];
            GroupMemoryBarrierWithGroupSync();
            store_stage_pair(stage_pos, head, lane);
            GroupMemoryBarrierWithGroupSync();
        }
    }
}