// Pure CPU estimator for the planned draft-MTP KV reservation.
// See llama-kv-mixed-mtp-budget.h.

#include "llama-kv-mixed-mtp-budget.h"

#include "llama-kv-cache-kvarn.h" // llama_kvarn_non_swa_tail_groups (canonical F16 stage)

#include <cstdio>
#include <cstring>

llama_kv_mixed_status llama_kv_mixed_mtp_budget_estimate(
        const llama_kv_mixed_mtp_input & input, llama_kv_mixed_mtp_budget & out) {
    out = {};
    const auto fail = [&](llama_kv_mixed_status status, const char * msg) {
        std::snprintf(out.error, sizeof(out.error), "%s", msg);
        return status;
    };

    if (input.n_ctx == 0 || input.n_seq_max == 0) {
        return fail(LLAMA_KV_MIXED_ERR_BAD_ARG, "mtp budget: n_ctx and n_seq_max must be >= 1");
    }
    if (input.layers.empty()) {
        return fail(LLAMA_KV_MIXED_ERR_BAD_ARG, "mtp budget: no MTP KV layers");
    }

    // draft cache representation: KVarN bits come from the type; the optional
    // packed override must AGREE with the type's canonical bits (the engine
    // derives every width from the type, so a hidden override width would not
    // be real). Standard types otherwise.
    int bits_k = 0;
    int bits_v = 0;
    if (input.kvarn != LLAMA_KVARN_TYPE_DISABLED) {
        // the canonical type ids are 1..36; INVALID (and any unknown id that
        // the type helper cannot round-trip) must fail closed, never fall back
        // to guessed bits
        if (int(input.kvarn) <= int(LLAMA_KVARN_TYPE_DISABLED)) {
            return fail(LLAMA_KV_MIXED_ERR_BAD_ARG, "mtp budget: invalid kvarn type");
        }
        const llama_kvarn_params kp = llama_kvarn_params_for_type(input.kvarn);
        if (kp.type != input.kvarn) {
            return fail(LLAMA_KV_MIXED_ERR_BAD_ARG, "mtp budget: invalid kvarn type");
        }
        bits_k = kp.key_bits;
        bits_v = kp.value_bits;
        if (input.kvarn_bits != 0) {
            const int packed_k = int(input.kvarn_bits & 0xFFFFu);
            const int packed_v = int(input.kvarn_bits >> 16);
            if (packed_k != kp.key_bits || packed_v != kp.value_bits) {
                return fail(LLAMA_KV_MIXED_ERR_BAD_ARG,
                            "mtp budget: packed kvarn bits disagree with the selected type");
            }
        }
    }

    // resolved sizing mirrors the mixed auto-placement planner: canonical F16
    // stage (2 groups x streams) plus one sink reserve; MTP tail policy
    // (tail 0 / F16; KVarN keeps its intrinsic min(128, capacity) tail)
    llama_kv_mixed_sizing sizing = {};
    sizing.capacity_tokens = input.n_ctx;
    sizing.n_seq_max = input.n_seq_max;
    sizing.kv_unified = input.kv_unified;
    sizing.stage_tail_groups = llama_kvarn_non_swa_tail_groups(input.n_batch, input.n_ubatch) *
            (input.kv_unified ? input.n_seq_max : 1u);
    sizing.stage_reserve_groups = 1;
    sizing.tail_exact_tokens = input.tail_tokens;
    sizing.tail_rollback_tokens = input.tail_rollback_tokens;
    sizing.tail_type = input.tail_type == GGML_TYPE_COUNT ? GGML_TYPE_F16 : input.tail_type;
    sizing.remote_tail_tokens = 0;

    out.layer_bytes.reserve(input.layers.size());
    for (const llama_kv_mixed_mtp_layer & l : input.layers) {
        llama_kv_mixed_layer_params layer = {};
        layer.layer = l.layer;
        layer.head_dim_k = l.head_dim_k;
        layer.head_dim_v = l.head_dim_v;
        layer.n_head_kv = l.n_head_kv;
        layer.n_embd_k_gqa = l.n_embd_k_gqa;
        layer.n_embd_v_gqa = l.n_embd_v_gqa;
        if (input.kvarn != LLAMA_KVARN_TYPE_DISABLED) {
            layer.kvarn_bits_k = bits_k;
            layer.kvarn_bits_v = bits_v;
        } else {
            layer.qx_type_k = input.type_k;
            layer.qx_type_v = input.type_v;
        }
        llama_kv_mixed_layer_cost cost = {};
        char err[256] = {};
        const llama_kv_mixed_status status =
                llama_kv_mixed_estimate_layer_cost(layer, sizing, cost, err, sizeof(err));
        if (status != LLAMA_KV_MIXED_OK) {
            std::snprintf(out.error, sizeof(out.error), "mtp budget: layer %u: %s",
                          l.layer, err[0] ? err : llama_kv_mixed_status_name(status));
            return status;
        }
        const uint64_t layer_bytes = input.kvarn != LLAMA_KVARN_TYPE_DISABLED ?
                cost.local_bytes : cost.remote_bytes;
        if (layer_bytes > UINT64_MAX - out.total_bytes) {
            return fail(LLAMA_KV_MIXED_ERR_OVERFLOW, "mtp budget: total overflows u64");
        }
        out.total_bytes += layer_bytes;
        out.layer_bytes.push_back(layer_bytes);
    }
    return LLAMA_KV_MIXED_OK;
}