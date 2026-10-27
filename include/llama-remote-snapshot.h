#pragma once
#include "llama.h"
#ifdef __cplusplus
extern "C" {
#endif
// Optional acceleration for canonical disk snapshots; no RPC for local/CUDA.
LLAMA_API bool llama_remote_disk_snapshot_enabled(const struct llama_context * ctx);
// True only when the live native owner negotiated bounded resident rollback.
LLAMA_API bool llama_remote_native_kvarn_can_rewind(const struct llama_context * ctx);
// True as soon as the native overflow backend is configured, including an empty
// owner. Speculators must check its rollback capability before drafting.
LLAMA_API bool llama_remote_native_kvarn_enabled(const struct llama_context * ctx);
// True while the native owner holds overflow rows. FULL saves require the
// separate snapshot capability below; a Q4 bookkeeping snapshot is not native
// state.
LLAMA_API bool llama_remote_native_kvarn_transient_active(const struct llama_context * ctx);
// True only for the supported revision-3 envelope (single sequence, n_rs_seq=0,
// opt-in disk snapshots, negotiated native snapshot capability). Other native
// combinations retain the fail-closed FULL save/restore guards.
LLAMA_API bool llama_remote_native_kvarn_snapshot_supported(const struct llama_context * ctx);
LLAMA_API bool llama_state_seq_remote_disk_snapshot_save(
        struct llama_context * ctx, llama_seq_id seq_id,
        const uint8_t host_native_sha256[32], size_t evaluated_tokens);
#ifdef __cplusplus
}
#endif
