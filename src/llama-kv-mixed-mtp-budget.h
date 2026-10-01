#pragma once

// Pure CPU estimator for the PLANNED draft-MTP KV reservation contract (see
// llama_context_params::mtp_reserve_*). Reuses the reviewed mixed-KV cost
// model (llama_kv_mixed_estimate_layer_cost from llama-kv-mixed-placement):
//   - KVarN draft cache: records + F16 stage + intrinsic/explicit exact tail
//     with the DRAFT bits and the resolved capacity; tail 0 resolves to the
//     intrinsic min(128, capacity) tail plus the profile rollback, mirroring
//     the canonical KVarN tail policy;
//   - standard draft cache: actual ggml row bytes for the DRAFT K/V types
//     with the MTP tail policy (init forces tail 0 / F16, so no explicit
//     tail is counted).
// The estimator performs no GPU work, no DEVICE allocations and never
// guesses the target format: unsupported shapes/types fail closed. The input
// is the
// resolved contract; the caller derives it from the resolved speculative
// profile (common_base_params_to_speculative semantics).

#include "llama.h"
#include "llama-kv-mixed-placement.h"

#include <cstdint>
#include <vector>

// one MTP KV layer (full-attention geometry)
struct llama_kv_mixed_mtp_layer {
    uint32_t layer = 0;
    uint32_t head_dim_k = 0;
    uint32_t head_dim_v = 0;
    uint32_t n_head_kv = 0;
    uint32_t n_embd_k_gqa = 0;
    uint32_t n_embd_v_gqa = 0;
};

struct llama_kv_mixed_mtp_input {
    uint32_t n_ctx = 0;    // resolved target capacity (>= 1); MTP overrides n_ctx to llama_n_ctx(target)
    uint32_t n_seq_max = 1;
    bool kv_unified = false;
    uint32_t n_batch = 0;  // canonical F16 stage tail groups (2 x streams)
    uint32_t n_ubatch = 0;
    enum llama_kvarn_type kvarn = LLAMA_KVARN_TYPE_DISABLED; // DISABLED -> standard types
    uint32_t kvarn_bits = 0; // packed key_bits | (value_bits << 16); 0 = derive from kvarn type
    enum ggml_type type_k = GGML_TYPE_F16; // standard K type when kvarn disabled
    enum ggml_type type_v = GGML_TYPE_F16;
    uint32_t tail_tokens = 0; // 0 -> intrinsic min(128, capacity) for KVarN; MTP init forces 0
    enum ggml_type tail_type = GGML_TYPE_F16;
    uint32_t tail_rollback_tokens = 0; // 0 -> canonical 1 when the resolved exact tail is nonzero
    std::vector<llama_kv_mixed_mtp_layer> layers; // MTP KV layers, model order
};

struct llama_kv_mixed_mtp_budget {
    uint64_t total_bytes = 0;
    std::vector<uint64_t> layer_bytes; // per MTP layer, input order
    char error[256] = {};
};

// Estimated draft-MTP KV capacity bytes. LLAMA_KV_MIXED_OK on success;
// LLAMA_KV_MIXED_ERR_BAD_ARG / _GEOMETRY / _OVERFLOW on unsupported input
// (error text in out.error).
llama_kv_mixed_status llama_kv_mixed_mtp_budget_estimate(
        const llama_kv_mixed_mtp_input & input, llama_kv_mixed_mtp_budget & out);