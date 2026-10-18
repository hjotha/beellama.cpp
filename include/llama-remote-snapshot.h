#pragma once
#include "llama.h"
#ifdef __cplusplus
extern "C" {
#endif
// Optional acceleration for canonical disk snapshots; no RPC for local/CUDA.
LLAMA_API bool llama_remote_disk_snapshot_enabled(const struct llama_context * ctx);
// Native overflow owner has no checkpoint codec yet. Callers must skip FULL
// automatic saves while active; a Q4 bookkeeping snapshot is not native state.
LLAMA_API bool llama_remote_native_kvarn_transient_active(const struct llama_context * ctx);
LLAMA_API bool llama_state_seq_remote_disk_snapshot_save(
        struct llama_context * ctx, llama_seq_id seq_id,
        const uint8_t host_native_sha256[32], size_t evaluated_tokens);
#ifdef __cplusplus
}
#endif
