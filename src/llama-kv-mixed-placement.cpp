#include "llama-kv-mixed-placement.h"

#include "llama-kvarn.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

void set_err(char * err, size_t err_len, const char * fmt, ...) {
    if (!err || err_len == 0) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(err, err_len, fmt, args);
    va_end(args);
}

// Portable checked arithmetic (MSVC has no __builtin_*_overflow).
bool add_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (a > UINT64_MAX - b) {
        return false;
    }
    out = a + b;
    return true;
}

bool mul_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (a != 0 && b > UINT64_MAX / a) {
        return false;
    }
    out = a * b;
    return true;
}

bool div_ceil_u64(uint64_t a, uint64_t b, uint64_t & out) {
    out = a / b + (a % b != 0 ? 1u : 0u);
    return true;
}

bool kvarn_bits_valid(int bits) {
    return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8;
}

bool tail_type_allowed(ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

// Validates the type and row width before touching ggml_row_size (which
// asserts on out-of-range types and on ne not being a block multiple).
bool row_size_checked(ggml_type type, uint32_t ne, uint64_t & out) {
    if (type < 0 || type >= GGML_TYPE_COUNT || ne == 0) {
        return false;
    }
    const int64_t blck = ggml_blck_size(type);
    if (blck <= 0 || ne % blck != 0) {
        return false;
    }
    const size_t row = ggml_row_size(type, ne);
    if (row == 0) {
        return false;
    }
    out = uint64_t(row);
    return true;
}

// Resolves the per-device handoff staging for a remote prefix of n_remote
// layers. Explicit per-device staging wins when provided; otherwise the host
// staging is derived from the CPU chunk-converter model (one remote layer at a
// time, handoff_concurrency layers in parallel). handoff_temp_max_bytes caps
// the total: derived staging is reduced to whole 128-token groups; explicit
// staging above the cap is a hard error.
llama_kv_mixed_status placement_temp(
        const std::vector<llama_kv_mixed_layer_cost> & costs,
        size_t n_remote,
        const llama_kv_mixed_sizing & sizing,
        const llama_kv_mixed_budget & budget,
        uint64_t & cuda_temp,
        uint64_t & vulkan_temp,
        uint64_t & ram_temp,
        uint64_t & eff_chunk,
        bool & capped) {
    cuda_temp = 0;
    vulkan_temp = 0;
    ram_temp = 0;
    eff_chunk = 0;
    capped = false;
    if (n_remote == 0) {
        return LLAMA_KV_MIXED_OK;
    }

    cuda_temp = budget.cuda_temp_bytes;
    vulkan_temp = budget.vulkan_temp_bytes;

    uint64_t per_group = 0;       // derived scratch for one 128-token group
    bool derived_requested = false; // chunk or full-prefix staging was asked for
    if (budget.ram_temp_bytes > 0) {
        ram_temp = budget.ram_temp_bytes;
    } else {
        uint64_t scratch_max = 0;
        for (size_t i = 0; i < n_remote; ++i) {
            scratch_max = std::max(scratch_max, costs[i].convert_scratch_per_token);
        }
        if (scratch_max > 0) {
            const uint64_t concurrency = budget.handoff_concurrency ? budget.handoff_concurrency : 1;
            uint64_t tokens = 0;
            if (budget.handoff_full_prefix_staging) {
                tokens = sizing.capacity_tokens;
                derived_requested = true;
            } else if (budget.handoff_chunk_tokens > 0) {
                uint64_t groups = 0;
                div_ceil_u64(budget.handoff_chunk_tokens, KVAR_N_GROUP, groups);
                if (!mul_u64(groups, KVAR_N_GROUP, tokens)) {
                    return LLAMA_KV_MIXED_ERR_OVERFLOW;
                }
                derived_requested = true;
            }
            // Steady state (no chunk, no flag): no derived staging at all.
            if (derived_requested) {
                if (tokens > sizing.capacity_tokens) {
                    tokens = sizing.capacity_tokens;
                }
                if (!mul_u64(scratch_max, KVAR_N_GROUP, per_group) ||
                        !mul_u64(per_group, concurrency, per_group) ||
                        !mul_u64(tokens / KVAR_N_GROUP, per_group, ram_temp)) {
                    return LLAMA_KV_MIXED_ERR_OVERFLOW;
                }
                eff_chunk = budget.handoff_full_prefix_staging ? 0 : tokens;
            }
        }
    }

    if (budget.handoff_temp_max_bytes > 0) {
        uint64_t total = 0;
        if (!add_u64(cuda_temp, vulkan_temp, total) ||
                !add_u64(total, ram_temp, total)) {
            return LLAMA_KV_MIXED_ERR_OVERFLOW;
        }
        if (total > budget.handoff_temp_max_bytes) {
            if (budget.ram_temp_bytes > 0 || !derived_requested) {
                // Declared staging (or a steady-state contract) cannot be
                // reduced by chunking.
                return LLAMA_KV_MIXED_ERR_TEMP;
            }
            uint64_t explicit_total = 0;
            if (!add_u64(cuda_temp, vulkan_temp, explicit_total) ||
                    explicit_total >= budget.handoff_temp_max_bytes) {
                return LLAMA_KV_MIXED_ERR_TEMP;
            }
            const uint64_t room = budget.handoff_temp_max_bytes - explicit_total;
            const uint64_t max_groups = room / per_group;
            if (max_groups == 0) {
                return LLAMA_KV_MIXED_ERR_TEMP;
            }
            if (!mul_u64(max_groups, per_group, ram_temp) ||
                    !mul_u64(max_groups, KVAR_N_GROUP, eff_chunk)) {
                return LLAMA_KV_MIXED_ERR_OVERFLOW;
            }
            capped = true;
        }
    }

    return LLAMA_KV_MIXED_OK;
}

} // namespace

bool llama_kv_mixed_qx_type_allowed(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q2_1:
        case GGML_TYPE_Q3_0:
        case GGML_TYPE_Q3_1:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_Q6_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
            return true;
        default:
            return false;
    }
}

const char * llama_kv_mixed_status_name(llama_kv_mixed_status status) {
    switch (status) {
        case LLAMA_KV_MIXED_OK:              return "ok";
        case LLAMA_KV_MIXED_ERR_BAD_ARG:     return "bad argument";
        case LLAMA_KV_MIXED_ERR_GEOMETRY:    return "unsupported KVarN geometry";
        case LLAMA_KV_MIXED_ERR_OVERFLOW:    return "arithmetic overflow";
        case LLAMA_KV_MIXED_ERR_RESERVE:     return "fixed reserve exceeds free memory";
        case LLAMA_KV_MIXED_ERR_TEMP:        return "handoff staging cannot fit";
        case LLAMA_KV_MIXED_ERR_NO_PLACEMENT: return "no placement satisfies every budget";
    }
    return "unknown";
}

llama_kv_mixed_status llama_kv_mixed_estimate_layer_cost(
        const llama_kv_mixed_layer_params & layer,
        const llama_kv_mixed_sizing & sizing,
        llama_kv_mixed_layer_cost & cost,
        char * err,
        size_t err_len) {
    llama_kv_mixed_layer_cost out = {};

    if (sizing.capacity_tokens == 0) {
        set_err(err, err_len, "capacity_tokens must be >= 1");
        return LLAMA_KV_MIXED_ERR_BAD_ARG;
    }
    if (sizing.n_seq_max == 0) {
        set_err(err, err_len, "n_seq_max must be >= 1");
        return LLAMA_KV_MIXED_ERR_BAD_ARG;
    }
    const int bits_k = layer.kvarn_bits_k ? layer.kvarn_bits_k : 4;
    const int bits_v = layer.kvarn_bits_v ? layer.kvarn_bits_v : 4;
    if (!kvarn_bits_valid(bits_k) || !kvarn_bits_valid(bits_v)) {
        set_err(err, err_len, "layer %u: kvarn bits must be one of 2/3/4/5/6/8 (got %d/%d)",
                layer.layer, bits_k, bits_v);
        return LLAMA_KV_MIXED_ERR_BAD_ARG;
    }
    if (layer.n_head_kv == 0) {
        set_err(err, err_len, "layer %u: n_head_kv must be >= 1", layer.layer);
        return LLAMA_KV_MIXED_ERR_BAD_ARG;
    }
    if (layer.n_embd_k_gqa == 0 || layer.n_embd_v_gqa == 0) {
        set_err(err, err_len, "layer %u: K/V row widths must be >= 1", layer.layer);
        return LLAMA_KV_MIXED_ERR_BAD_ARG;
    }

    // Row widths must agree with the geometry so records, stage and tail
    // sizing cannot diverge (checked multiply, portable).
    {
        uint64_t expect_k = 0;
        uint64_t expect_v = 0;
        if (!mul_u64(layer.head_dim_k, layer.n_head_kv, expect_k) ||
                !mul_u64(layer.head_dim_v, layer.n_head_kv, expect_v) ||
                expect_k != layer.n_embd_k_gqa || expect_v != layer.n_embd_v_gqa) {
            set_err(err, err_len,
                    "layer %u: n_embd_k_gqa=%u/n_embd_v_gqa=%u disagree with head_dim_k=%u, head_dim_v=%u, n_head_kv=%u",
                    layer.layer, layer.n_embd_k_gqa, layer.n_embd_v_gqa,
                    layer.head_dim_k, layer.head_dim_v, layer.n_head_kv);
            return LLAMA_KV_MIXED_ERR_BAD_ARG;
        }
    }

    if (!llama_kv_mixed_qx_type_allowed(layer.qx_type_k) ||
            !llama_kv_mixed_qx_type_allowed(layer.qx_type_v)) {
        set_err(err, err_len, "layer %u: remote K/V types %d/%d outside the Qx whitelist",
                layer.layer, (int) layer.qx_type_k, (int) layer.qx_type_v);
        return LLAMA_KV_MIXED_ERR_BAD_ARG;
    }

    const ggml_type tail_type = sizing.tail_type == GGML_TYPE_COUNT ?
        GGML_TYPE_F16 : sizing.tail_type;
    if (!tail_type_allowed(tail_type)) {
        set_err(err, err_len, "tail_type %d not in the F16/BF16 whitelist", (int) tail_type);
        return LLAMA_KV_MIXED_ERR_BAD_ARG;
    }

    llama_kvarn_geometry k_geom = {};
    llama_kvarn_geometry v_geom = {};
    if (!llama_kvarn_geometry_for(layer.head_dim_k, k_geom) ||
            !llama_kvarn_geometry_for(layer.head_dim_v, v_geom)) {
        set_err(err, err_len, "layer %u: KVarN head dims %u/%u unsupported by canonical geometry",
                layer.layer, layer.head_dim_k, layer.head_dim_v);
        return LLAMA_KV_MIXED_ERR_GEOMETRY;
    }

    uint64_t qx_k_row = 0;
    uint64_t qx_v_row = 0;
    if (!row_size_checked(layer.qx_type_k, layer.n_embd_k_gqa, qx_k_row) ||
            !row_size_checked(layer.qx_type_v, layer.n_embd_v_gqa, qx_v_row)) {
        set_err(err, err_len, "layer %u: unsupported remote K/V types for row widths %u/%u",
                layer.layer, layer.n_embd_k_gqa, layer.n_embd_v_gqa);
        return LLAMA_KV_MIXED_ERR_BAD_ARG;
    }
    uint64_t tail_k_row = 0;
    uint64_t tail_v_row = 0;
    if (!row_size_checked(tail_type, layer.n_embd_k_gqa, tail_k_row) ||
            !row_size_checked(tail_type, layer.n_embd_v_gqa, tail_v_row)) {
        set_err(err, err_len, "layer %u: unsupported tail type %d for row widths %u/%u",
                layer.layer, (int) tail_type, layer.n_embd_k_gqa, layer.n_embd_v_gqa);
        return LLAMA_KV_MIXED_ERR_BAD_ARG;
    }

    // Canonical KVarN record layout, one record group per 128 tokens.
    const size_t k_rec = llama_kvarn_make_record_layout(k_geom.record_dim, bits_k, false).record_bytes;
    const size_t v_rec = llama_kvarn_make_record_layout(v_geom.record_dim, bits_v, true).record_bytes;

    uint64_t k_rec_group = 0;
    uint64_t v_rec_group = 0;
    uint64_t rec_group = 0;
    if (!mul_u64(k_rec, layer.n_head_kv, k_rec_group) ||
            !mul_u64(k_rec_group, k_geom.head_slices, k_rec_group) ||
            !mul_u64(v_rec, layer.n_head_kv, v_rec_group) ||
            !mul_u64(v_rec_group, v_geom.head_slices, v_rec_group) ||
            !add_u64(k_rec_group, v_rec_group, rec_group)) {
        set_err(err, err_len, "layer %u: record sizing overflow", layer.layer);
        return LLAMA_KV_MIXED_ERR_OVERFLOW;
    }

    const uint64_t n_streams = sizing.kv_unified ? 1 : uint64_t(sizing.n_seq_max);
    const uint64_t seq_factor = sizing.kv_unified ? uint64_t(sizing.n_seq_max) : 1;
    uint64_t record_groups = 0;
    div_ceil_u64(sizing.capacity_tokens, KVAR_N_GROUP, record_groups);
    const uint64_t stage_tail_groups = sizing.stage_tail_groups ?
        uint64_t(sizing.stage_tail_groups) : 2 * seq_factor;
    const uint64_t stage_groups = stage_tail_groups +
        uint64_t(sizing.stage_reserve_groups ? sizing.stage_reserve_groups : 1);

    // Resolved exact tail: canonical intrinsic min(128, capacity), canonical
    // rollback 1 when the exact tail is non-zero.
    const uint64_t exact_tail_tokens = sizing.tail_exact_tokens ?
        uint64_t(sizing.tail_exact_tokens) :
        std::min<uint64_t>(KVAR_N_GROUP, sizing.capacity_tokens);
    const uint64_t rollback_tokens = sizing.tail_rollback_tokens ?
        uint64_t(sizing.tail_rollback_tokens) : (exact_tail_tokens > 0 ? 1 : 0);

    // KVarN records (quantized) + fp16 stage ring + exact tail (intrinsic or
    // resolved explicit) with rollback slots.
    uint64_t records_bytes = 0;
    uint64_t stage_bytes = 0;
    uint64_t tail_bytes = 0;
    {
        uint64_t tmp = 0;
        if (!mul_u64(rec_group, record_groups, tmp) ||
                !mul_u64(tmp, n_streams, records_bytes)) {
            set_err(err, err_len, "layer %u: record sizing overflow", layer.layer);
            return LLAMA_KV_MIXED_ERR_OVERFLOW;
        }
        uint64_t dims = 0;
        if (!add_u64(layer.n_embd_k_gqa, layer.n_embd_v_gqa, dims) ||
                !mul_u64(dims, 2, tmp) ||
                !mul_u64(tmp, KVAR_N_GROUP, tmp) ||
                !mul_u64(tmp, stage_groups, tmp) ||
                !mul_u64(tmp, n_streams, stage_bytes)) {
            set_err(err, err_len, "layer %u: stage sizing overflow", layer.layer);
            return LLAMA_KV_MIXED_ERR_OVERFLOW;
        }
        const uint64_t tail_slots = exact_tail_tokens + rollback_tokens;
        if (!add_u64(tail_k_row, tail_v_row, tmp) ||
                !mul_u64(tmp, tail_slots, tmp) ||
                !mul_u64(tmp, sizing.n_seq_max, tail_bytes)) {
            set_err(err, err_len, "layer %u: tail sizing overflow", layer.layer);
            return LLAMA_KV_MIXED_ERR_OVERFLOW;
        }
    }
    if (!add_u64(records_bytes, stage_bytes, out.local_bytes) ||
            !add_u64(out.local_bytes, tail_bytes, out.local_bytes)) {
        set_err(err, err_len, "layer %u: local sizing overflow", layer.layer);
        return LLAMA_KV_MIXED_ERR_OVERFLOW;
    }

    // Exact per-token record cost, rounded up (records are per 128-token group).
    {
        uint64_t tmp = 0;
        uint64_t group_cost = 0;
        if (!mul_u64(rec_group, n_streams, group_cost)) {
            set_err(err, err_len, "layer %u: record sizing overflow", layer.layer);
            return LLAMA_KV_MIXED_ERR_OVERFLOW;
        }
        div_ceil_u64(group_cost, KVAR_N_GROUP, tmp);
        out.local_per_token = tmp;
    }

    // Qx rows for the full allocated capacity, K and V independent. The
    // KVarN tail area is never inherited by the remote side; only an explicit
    // remote_tail_tokens contributes.
    {
        uint64_t remote_row = 0;
        if (!add_u64(qx_k_row, qx_v_row, remote_row) ||
                !mul_u64(remote_row, n_streams, out.remote_per_token) ||
                !mul_u64(out.remote_per_token, sizing.capacity_tokens, out.remote_bytes)) {
            set_err(err, err_len, "layer %u: remote sizing overflow", layer.layer);
            return LLAMA_KV_MIXED_ERR_OVERFLOW;
        }
        if (sizing.remote_tail_tokens > 0) {
            uint64_t tail_row = 0;
            uint64_t tmp = 0;
            if (!add_u64(tail_k_row, tail_v_row, tail_row) ||
                    !mul_u64(tail_row, sizing.remote_tail_tokens, tmp) ||
                    !mul_u64(tmp, sizing.n_seq_max, tmp) ||
                    !add_u64(out.remote_bytes, tmp, out.remote_bytes)) {
                set_err(err, err_len, "layer %u: remote tail sizing overflow", layer.layer);
                return LLAMA_KV_MIXED_ERR_OVERFLOW;
            }
        }
    }

    // Host converter scratch: F32 dequant (K+V) plus the Qx output rows, for
    // one layer and one stream (the chunk model never sums layers).
    {
        uint64_t dims = 0;
        uint64_t scratch = 0;
        if (!add_u64(layer.n_embd_k_gqa, layer.n_embd_v_gqa, dims) ||
                !mul_u64(dims, 4, scratch) ||
                !add_u64(qx_k_row, qx_v_row, dims) ||
                !add_u64(scratch, dims, out.convert_scratch_per_token)) {
            set_err(err, err_len, "layer %u: scratch sizing overflow", layer.layer);
            return LLAMA_KV_MIXED_ERR_OVERFLOW;
        }
    }

    cost = out;
    return LLAMA_KV_MIXED_OK;
}

llama_kv_mixed_status llama_kv_mixed_choose_placement(
        const std::vector<llama_kv_mixed_layer_cost> & costs,
        const llama_kv_mixed_sizing & sizing,
        const llama_kv_mixed_budget & budget,
        llama_kv_mixed_placement & placement) {
    llama_kv_mixed_placement out = {};
    const size_t n = costs.size();

    std::vector<uint64_t> prefix_remote(n + 1, 0);
    std::vector<uint64_t> suffix_local(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        if (!add_u64(prefix_remote[i], costs[i].remote_bytes, prefix_remote[i + 1])) {
            set_err(out.error, sizeof(out.error),
                    "placement cost accumulation overflow at layer %zu", i);
            out.status = LLAMA_KV_MIXED_ERR_OVERFLOW;
            placement = out;
            return out.status;
        }
    }
    for (size_t i = n; i > 0; --i) {
        if (!add_u64(suffix_local[i], costs[i - 1].local_bytes, suffix_local[i - 1])) {
            set_err(out.error, sizeof(out.error),
                    "placement cost accumulation overflow at layer %zu", i - 1);
            out.status = LLAMA_KV_MIXED_ERR_OVERFLOW;
            placement = out;
            return out.status;
        }
    }

    if (budget.cuda_reserve_bytes > budget.cuda_free_bytes) {
        set_err(out.error, sizeof(out.error),
                "CUDA fixed reserve %.1f MiB exceeds free %.1f MiB; no local KV fits",
                budget.cuda_reserve_bytes / 1024.0 / 1024.0,
                budget.cuda_free_bytes / 1024.0 / 1024.0);
        out.status = LLAMA_KV_MIXED_ERR_RESERVE;
        placement = out;
        return out.status;
    }
    const uint64_t cuda_available = budget.cuda_free_bytes - budget.cuda_reserve_bytes;
    const bool vulkan_reserve_exceeds = budget.vulkan_reserve_bytes > budget.vulkan_free_bytes;
    const uint64_t vulkan_available = vulkan_reserve_exceeds ?
        0 : budget.vulkan_free_bytes - budget.vulkan_reserve_bytes;

    bool decided = false;
    uint64_t last_cuda_used = 0;
    for (size_t n_remote = 0; n_remote <= n; ++n_remote) {
        if (n_remote > 0) {
            if (vulkan_reserve_exceeds) {
                set_err(out.error, sizeof(out.error),
                        "Vulkan fixed reserve %.1f MiB exceeds free %.1f MiB; remote placement impossible",
                        budget.vulkan_reserve_bytes / 1024.0 / 1024.0,
                        budget.vulkan_free_bytes / 1024.0 / 1024.0);
                out.status = LLAMA_KV_MIXED_ERR_RESERVE;
                decided = true;
                break;
            }
            if (prefix_remote[n_remote] > vulkan_available) {
                set_err(out.error, sizeof(out.error),
                        "remote prefix for N=%zu needs %.1f MiB but Vulkan budget is %.1f MiB "
                        "(fixed reserve subtracted); every larger prefix is worse",
                        n_remote, prefix_remote[n_remote] / 1024.0 / 1024.0,
                        vulkan_available / 1024.0 / 1024.0);
                out.status = LLAMA_KV_MIXED_ERR_NO_PLACEMENT;
                decided = true;
                break;
            }
        }

        uint64_t cuda_temp = 0;
        uint64_t vulkan_temp = 0;
        uint64_t ram_temp = 0;
        uint64_t eff_chunk = 0;
        bool capped = false;
        const llama_kv_mixed_status temp_status = placement_temp(
                costs, n_remote, sizing, budget, cuda_temp, vulkan_temp, ram_temp, eff_chunk, capped);
        if (temp_status != LLAMA_KV_MIXED_OK) {
            if (temp_status == LLAMA_KV_MIXED_ERR_TEMP) {
                set_err(out.error, sizeof(out.error),
                        "handoff staging cannot fit within handoff_temp_max_bytes %.1f MiB "
                        "(explicit staging is never reduced; derived staging needs at least one 128-token group)",
                        budget.handoff_temp_max_bytes / 1024.0 / 1024.0);
            } else {
                set_err(out.error, sizeof(out.error), "handoff staging sizing overflow");
            }
            out.status = temp_status;
            decided = true;
            break;
        }

        uint64_t ram_used = 0;
        if (n_remote > 0 && budget.ram_cap_bytes > 0) {
            if (!add_u64(budget.vulkan_reserve_bytes, prefix_remote[n_remote], ram_used) ||
                    !add_u64(ram_used, vulkan_temp, ram_used) ||
                    !add_u64(ram_used, ram_temp, ram_used)) {
                set_err(out.error, sizeof(out.error), "RAM accounting overflow");
                out.status = LLAMA_KV_MIXED_ERR_OVERFLOW;
                decided = true;
                break;
            }
            if (ram_used > budget.ram_cap_bytes) {
                set_err(out.error, sizeof(out.error),
                        "remote footprint for N=%zu needs %.1f MiB (reserve + KV + staging, counted once) "
                        "but the global RAM cap is %.1f MiB",
                        n_remote, ram_used / 1024.0 / 1024.0,
                        budget.ram_cap_bytes / 1024.0 / 1024.0);
                out.status = LLAMA_KV_MIXED_ERR_NO_PLACEMENT;
                decided = true;
                break;
            }
        }

        uint64_t cuda_used = 0;
        uint64_t vulkan_used = 0;
        uint64_t temp_total = 0;
        if (!add_u64(suffix_local[n_remote], cuda_temp, cuda_used) ||
                !add_u64(prefix_remote[n_remote], vulkan_temp, vulkan_used) ||
                !add_u64(cuda_temp, vulkan_temp, temp_total) ||
                !add_u64(temp_total, ram_temp, temp_total)) {
            set_err(out.error, sizeof(out.error), "placement sizing overflow");
            out.status = LLAMA_KV_MIXED_ERR_OVERFLOW;
            decided = true;
            break;
        }
        last_cuda_used = cuda_used;
        if (cuda_used <= cuda_available && vulkan_used <= vulkan_available) {
            out.n_remote = uint32_t(n_remote);
            out.n_local = uint32_t(n - n_remote);
            out.remote_kv_bytes = prefix_remote[n_remote];
            out.local_kv_bytes = suffix_local[n_remote];
            out.cuda_used_bytes = cuda_used;
            out.cuda_available_bytes = cuda_available;
            out.vulkan_used_bytes = vulkan_used;
            out.vulkan_available_bytes = vulkan_available;
            out.ram_used_bytes = ram_used;
            out.cuda_temp_bytes = cuda_temp;
            out.vulkan_temp_bytes = vulkan_temp;
            out.ram_temp_bytes = ram_temp;
            out.handoff_temp_bytes = temp_total;
            out.effective_chunk_tokens = eff_chunk;
            out.temp_capped = capped;
            out.status = LLAMA_KV_MIXED_OK;
            placement = out;
            return out.status;
        }
    }

    if (!decided) {
        set_err(out.error, sizeof(out.error),
                "no remote prefix N in [0, %zu] leaves a local suffix plus staging that fits "
                "CUDA budget %.1f MiB (last tried: suffix + staging %.1f MiB) and a remote "
                "prefix that fits its budget; increase the budgets or reduce staging",
                n,
                cuda_available / 1024.0 / 1024.0,
                last_cuda_used / 1024.0 / 1024.0);
        out.status = LLAMA_KV_MIXED_ERR_NO_PLACEMENT;
    }
    placement = out;
    return out.status;
}