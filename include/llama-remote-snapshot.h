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
// Optional transient-RAM reservation for the next native FULL restore. The
// callback is invoked before the RKVK payload is allocated and must reserve the
// complete staged-payload + owner-allocation peak. Null clears the hook.
typedef bool (*llama_native_restore_reserve_fn)(void * user, size_t bytes);
LLAMA_API void llama_state_seq_set_native_restore_reservation(
        struct llama_context * ctx, llama_native_restore_reserve_fn reserve, void * user);
// True only when the last sequence-state parse committed the paired native
// revision-3 trailer. Explicit native slot restore uses this to reject legacy
// Q4/Q8 blobs that would otherwise load through the generic API.
LLAMA_API bool llama_state_seq_restore_was_native(const struct llama_context * ctx);
// Strict loader for the explicit native streaming slot path. Rejects a
// non-native state file before the memory transaction commits.
LLAMA_API size_t llama_state_seq_load_file_native(
        struct llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id,
        llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);
// Bounded streaming variant used by the explicit adaptive slot API.
LLAMA_API size_t llama_state_seq_load_file_streaming_native(
        struct llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id,
        llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out,
        size_t state_size, uint64_t checksum);
#ifdef __cplusplus
}
#endif
