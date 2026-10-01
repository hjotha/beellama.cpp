#pragma once

// Server glue for the one-way PURE KVarN -> MIXED (KVarN local + standard Qx
// remote) prefix reuse. File-scope helpers only; no server_context_impl
// internals. The thin hook in server-context.cpp (artifact patch) captures
// the source descriptor before the old context is destroyed and calls
// try_handoff() when the mixed candidate exists and the prompt-cache load
// missed for a pure source snapshot.
//
// The engine (src/llama-kv-mixed-handoff.h) owns the transaction; this glue
// owns the authoritative source restore (FULL then selected PARTIAL
// checkpoint, CPU hybrid memory via model.create_memory with offload=false)
// and the candidate-side mapping.

#include "llama.h"
#include "src/llama-cparams.h"
#include "src/llama-memory.h"
#include "src/llama-kv-cache-kvarn.h" // complete llama_kv_cache_kvarn_layer_layout

#include <cstdint>
#include <string>
#include <vector>

struct server_prompt_cache_state;
struct server_tokens;
struct common_speculative;

namespace server_mixed_kv_handoff {

// Captured from the PURE KVarN source context BEFORE it is destroyed. KVarN
// bits are recorded explicitly (never derived from the F16 metadata profile).
struct source_capture {
    // Value-initialized: every member is deterministic by default (no
    // indeterminate mem_other/swa_full).
    bool valid = false;
    llama_cparams cparams = {};             // full copy (value-copyable)
    llama_memory_params mem_params = {};    // mem_other == nullptr by default
    uint64_t memory_bytes = 0;             // source KV + recurrent allocations
    uint64_t recurrent_bytes = 0;          // bounds the additional RS serialization
    uint32_t n_ctx_seq = 0;
    uint32_t exact_tail_tokens = 0;
    bool has_cell_ext = false;
    std::string source_layout;              // common_prompt_cache_layout of the source ctx
    uint64_t source_model_instance = 0;     // llama_model_mtp_weights_get_info
    std::vector<llama_kv_cache_kvarn_layer_layout> kvarn_layers;
};

source_capture capture_source(llama_context * ctx);

struct result {
    bool committed = false;
    std::string marker;                     // exact diagnostic on success
    std::string reason;                     // failure/no-source reason
    std::vector<llama_token> prefix_tokens; // transferred evaluated prefix
    bool mtp_bootstrap = false;             // destination MTP needs the existing bootstrap
};

struct input {
    llama_context * ctx_tgt = nullptr;      // MIXED empty candidate
    llama_context * ctx_dft = nullptr;      // rebuilt draft context (may be null)
    common_speculative * spec = nullptr;
    llama_model * model = nullptr;
    llama_seq_id seq_id = 0;
    bool has_mtmd = false;                  // unsupported token mapping -> explicit reject
    const source_capture * source = nullptr;
    const server_prompt_cache_state * state = nullptr; // selected pure snapshot
    const server_tokens * tokens_new = nullptr;
};

// Runs the handoff. Returns committed=true only when the candidate was fully
// installed and every validation passed; on failure reason/marker describe
// the exact step and the candidate must be discarded by the caller (nothing
// was published).
result try_handoff(const input & in);

} // namespace server_mixed_kv_handoff