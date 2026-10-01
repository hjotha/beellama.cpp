#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Pure CPU placement planner for the mixed-KV serving layout: full-attention
// layers in a remote prefix keep K/V in a standard low-bit Qx format on the
// remote device (Vulkan/Radeon), while the local suffix keeps K/V in KVarN on
// the CUDA device. This module performs no GPU work and no allocation; it
// sizes and chooses, nothing else.
//
// Sizing reuses the canonical KVarN helpers (llama_kvarn_geometry_for /
// llama_kvarn_make_record_layout) and mirrors the serving accounting:
//   - records: quantized payload over ceil(capacity/128) groups x n_streams,
//     where n_streams = kv_unified ? 1 : n_seq_max;
//   - stage: fp16 ring of 128 x (stage_tail_groups + stage_reserve_groups)
//     rows x n_streams (canonical tail groups = llama_kvarn_non_swa_tail_groups
//     = 2 x (kv_unified ? n_seq_max : 1), plus one sink reserve);
//   - exact tail: (tail_exact_tokens + tail_rollback_tokens) x n_seq_max slots
//     of (K row + V row) in tail_type. KVarN tail0 resolves to the intrinsic
//     128-token exact tail plus the profile rollback (canonical default 1),
//     mirroring llama-context.cpp tail-policy resolution. The Qx remote side
//     has no intrinsic tail and never inherits the KVarN tail area;
//     remote_tail_tokens is an explicit separate input (mixed remote currently
//     requires none, so the default is 0).
//
// The planner chooses the SMALLEST remote prefix N (layers 0..N-1) whose Qx
// cost plus remote staging fits the remote budget while the local KVarN suffix
// (layers N..) plus local staging fits the local budget, and whose remote
// footprint fits the optional global RAM cap. N = 0 is a valid, ordinary
// result and never forces the remote device. All arithmetic is overflow-checked
// with portable operations (no compiler builtins), and every failure carries a
// diagnostic message.

enum llama_kv_mixed_status {
    LLAMA_KV_MIXED_OK = 0,
    LLAMA_KV_MIXED_ERR_BAD_ARG,      // invalid parameter (capacity 0, bad bits, unsupported type, ...)
    LLAMA_KV_MIXED_ERR_GEOMETRY,     // KVarN head dim not supported by the canonical geometry
    LLAMA_KV_MIXED_ERR_OVERFLOW,     // sizing arithmetic overflow
    LLAMA_KV_MIXED_ERR_RESERVE,      // fixed workspace/RS/MTP reserve exceeds device free memory
    LLAMA_KV_MIXED_ERR_TEMP,         // handoff staging cannot fit (cap below one 128-token group)
    LLAMA_KV_MIXED_ERR_NO_PLACEMENT, // no remote prefix N satisfies every budget
};

const char * llama_kv_mixed_status_name(llama_kv_mixed_status status);

// Shared sizing parameters. Tail inputs are RESOLVED values: resolve the
// canonical policy first (e.g. llama_kvarn_tail_policy_for(raw, window)
// .effective_tokens for tail_exact_tokens) so this module never invents the
// intrinsic/explicit/rollback split. Defaults below implement the canonical
// profile: intrinsic min(128, capacity), rollback 1 when the exact tail is
// non-zero. The native_exact promotion (small windows where the exact tail
// covers the whole context, effective == window) is NOT modeled; contexts at
// or below 128 tokens must pass resolved inputs instead of the defaults.
// All values describe allocated capacity, never occupancy.
struct llama_kv_mixed_sizing {
    uint64_t capacity_tokens = 0;   // allocated capacity (n_ctx_seq), >= 1
    uint32_t n_seq_max = 1;         // sequences; exact-tail slots scale with this, >= 1
    bool     kv_unified = false;    // unified cache: n_streams = 1, stage tail groups x n_seq_max
    uint32_t stage_tail_groups = 0; // 0 -> canonical 2 * (kv_unified ? n_seq_max : 1)
    uint32_t stage_reserve_groups = 0; // 0 -> 1 (permanent sink)
    uint32_t tail_exact_tokens = 0; // 0 -> min(128, capacity_tokens)
    uint32_t tail_rollback_tokens = 0; // 0 -> 1 when the resolved exact tail is non-zero
    ggml_type tail_type = GGML_TYPE_COUNT; // COUNT -> GGML_TYPE_F16; whitelist F16/BF16
    uint32_t remote_tail_tokens = 0; // explicit remote Qx tail; 0 = none (mixed requires none)
};

// One full-attention layer. K and V are sized independently; head dims drive
// the canonical KVarN geometry while n_embd_k_gqa / n_embd_v_gqa drive row
// widths. The row widths are validated against head_dim x n_head_kv so the
// record and stage sizing cannot disagree with the geometry.
struct llama_kv_mixed_layer_params {
    uint32_t layer = 0;          // layer index, diagnostics only
    uint32_t head_dim_k = 0;     // 64/128/256/512 for the KVarN canonical geometry
    uint32_t head_dim_v = 0;
    uint32_t n_head_kv = 0;
    uint32_t n_embd_k_gqa = 0;   // K row width; must equal head_dim_k * n_head_kv
    uint32_t n_embd_v_gqa = 0;   // V row width; must equal head_dim_v * n_head_kv
    int kvarn_bits_k = 0;        // KVarN bits for the local format; 0 -> 4
    int kvarn_bits_v = 0;        // 0 -> 4
    ggml_type qx_type_k = GGML_TYPE_Q4_0; // remote K format
    ggml_type qx_type_v = GGML_TYPE_Q4_0; // remote V format
};

// Whitelisted remote (Qx) types. The experiment scope is Q4_0/Q5_0/Q6_0/Q8_0;
// the full fork standard low-bit KV set is accepted. Types outside the
// whitelist (IQ variants, QK, F32, COUNT, ...) are rejected before any ggml
// row-size call.
bool llama_kv_mixed_qx_type_allowed(ggml_type type);

// Estimated capacity bytes for one layer in both formats. remote_bytes is what
// the layer costs on the remote device; local_bytes is what it costs on the
// local device. The per-token values are exact per-128-group costs. The
// convert_scratch_per_token value feeds the derived host-staging model: one
// F32 dequant scratch plus the Qx output row for a single layer and stream.
struct llama_kv_mixed_layer_cost {
    uint64_t local_bytes = 0;          // KVarN K+V capacity bytes (records + stage + exact tail)
    uint64_t remote_bytes = 0;         // Qx K+V capacity bytes (+ explicit remote tail only)
    uint64_t local_per_token = 0;      // KVarN record bytes per token (ceil, K+V, x n_streams)
    uint64_t remote_per_token = 0;     // Qx row bytes per token (K+V, x n_streams)
    uint64_t convert_scratch_per_token = 0; // (k_dim + v_dim)*4 + Qx rows, one layer, one stream
};

// Memory budgets. cuda_free_bytes / vulkan_free_bytes are free memory AFTER
// model weights are resident; do not subtract weights again here. Fixed
// workspace/RS/MTP reserves are separate inputs because they are not yet
// allocated. The Radeon shares system RAM, so the optional ram_cap_bytes is a
// single additional global cap; when enabled it is checked against
// vulkan_reserve + remote KV + remote staging + host staging, each counted
// once (never GTT plus Vulkan as distinct memories). Remote workspace is only
// consulted when at least one layer is remote.
//
// Handoff staging is split per device. The converter is modeled as a CPU
// process over F32 host chunks; the safe default is therefore steady state
// (no staging at all) unless a chunk, the full-prefix opt-in flag, or explicit
// per-device staging is provided:
//   - cuda_temp_bytes: staging on CUDA (default 0; KVarN records are read in
//     place, no CUDA copy);
//   - vulkan_temp_bytes: staging on the remote device (default 0; chunks are
//     written directly; a staging design must declare it here);
//   - ram_temp_bytes: explicit host staging (default 0 -> derive).
// Derived host staging (when ram_temp_bytes == 0): one remote layer at a time
// (never summed over N), handoff_concurrency layers in parallel, over chunks
// of ceil(handoff_chunk_tokens/128)*128 tokens, of
// convert_scratch_per_token bytes per token, clamped to the allocated
// capacity. handoff_chunk_tokens == 0 without the full-prefix flag means
// steady state: no derived staging. The full-prefix flag is the explicit
// opt-in to stage the whole remote prefix at once.
// handoff_temp_max_bytes caps the total (explicit + derived) staging: derived
// staging is reduced to whole 128-token groups (temp_capped); explicit staging
// above the cap is a hard error. A cap below one group is a hard error.
struct llama_kv_mixed_budget {
    uint64_t cuda_free_bytes = 0;        // local VRAM free AFTER weights
    uint64_t cuda_reserve_bytes = 0;     // fixed local workspace/RS/MTP, not yet allocated
    uint64_t vulkan_free_bytes = 0;      // remote budget (shared RAM or VRAM, as reported)
    uint64_t vulkan_reserve_bytes = 0;   // fixed remote workspace/RS/MTP (only when N > 0)
    uint64_t ram_cap_bytes = 0;          // optional global RAM cap; 0 disables
    uint64_t cuda_temp_bytes = 0;        // explicit CUDA staging; 0 = none (default)
    uint64_t vulkan_temp_bytes = 0;      // explicit remote staging; 0 = none (default)
    uint64_t ram_temp_bytes = 0;         // explicit host staging; 0 = derive chunk model
    uint64_t handoff_chunk_tokens = 0;   // derived chunk; 0 = steady state unless the flag is set
    bool     handoff_full_prefix_staging = false; // explicit opt-in to full-prefix staging
    uint32_t handoff_concurrency = 1;    // remote layers converted in parallel (0 -> 1)
    uint64_t handoff_temp_max_bytes = 0; // cap on total staging; 0 = unlimited
};

struct llama_kv_mixed_placement {
    uint32_t n_remote = 0;            // remote prefix layers (0..N-1)
    uint32_t n_local = 0;             // local suffix layers (N..)
    uint64_t remote_kv_bytes = 0;     // Qx capacity bytes on the remote device
    uint64_t local_kv_bytes = 0;      // KVarN capacity bytes on the local device
    uint64_t cuda_used_bytes = 0;     // local suffix KV + CUDA staging
    uint64_t cuda_available_bytes = 0;
    uint64_t vulkan_used_bytes = 0;   // remote KV + remote staging
    uint64_t vulkan_available_bytes = 0;
    uint64_t ram_used_bytes = 0;      // reserve + remote KV + remote/host staging, counted once; 0 when the cap is disabled
    uint64_t cuda_temp_bytes = 0;     // staging actually required per device
    uint64_t vulkan_temp_bytes = 0;
    uint64_t ram_temp_bytes = 0;
    uint64_t handoff_temp_bytes = 0;  // total staging (cuda + vulkan + ram)
    uint64_t effective_chunk_tokens = 0; // chunk used to bound derived staging (0 = steady state or full prefix)
    bool temp_capped = false;         // derived staging was reduced by handoff_temp_max_bytes
    llama_kv_mixed_status status = LLAMA_KV_MIXED_ERR_BAD_ARG;
    char error[256] = {};
};

// Estimates local (KVarN) and remote (Qx) capacity bytes for one layer.
// err/err_len are optional (nullptr allowed) diagnostics.
llama_kv_mixed_status llama_kv_mixed_estimate_layer_cost(
        const llama_kv_mixed_layer_params & layer,
        const llama_kv_mixed_sizing & sizing,
        llama_kv_mixed_layer_cost & cost,
        char * err,
        size_t err_len);

// Chooses the smallest remote prefix N whose remote Qx cost plus remote
// staging fits the remote budget, whose local KVarN suffix plus CUDA staging
// fits the local budget, and whose remote footprint fits the global RAM cap
// when enabled. Costs must come from llama_kv_mixed_estimate_layer_cost and
// keep layer order: prefix = costs[0..N), suffix = costs[N..). N = 0 is valid.
llama_kv_mixed_status llama_kv_mixed_choose_placement(
        const std::vector<llama_kv_mixed_layer_cost> & costs,
        const llama_kv_mixed_sizing & sizing,
        const llama_kv_mixed_budget & budget,
        llama_kv_mixed_placement & placement);