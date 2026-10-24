#pragma once

#include "llama.h"

// Result of the decode-failure rollback: the suffix removal succeeded, the
// whole sequence had to be invalidated, or both failed.
enum llama_rollback_result {
    LLAMA_ROLLBACK_SUFFIX,
    LLAMA_ROLLBACK_FULL,
    LLAMA_ROLLBACK_NONE,
};

// A terminal remote failure must never be retried as a batch-size issue, even
// when a local suffix removal happens to succeed. -4 is treated by the server
// as a non-retryable compute error (ret < -1).
static inline int llama_decode_aborted_code(bool rollback_failed, bool remote_failed) {
    return (rollback_failed || remote_failed) ? -4 : 2;
}

// After a decode failure, try to remove the suffix first. If the memory refuses
// (position-split exact reservation), invalidate the whole sequence instead of
// leaving a partially applied KV. The callback owns the seq_rm primitive and
// must run the on_seq_rm callback for the full clear.
template <typename SeqRm>
inline llama_rollback_result llama_rollback_sequence_after_decode_failure(
        SeqRm && seq_rm, llama_seq_id seq_id, llama_pos pos_min) {
    if (seq_rm(seq_id, pos_min, -1)) {
        return LLAMA_ROLLBACK_SUFFIX;
    }
    if (seq_rm(seq_id, -1, -1)) {
        return LLAMA_ROLLBACK_FULL;
    }
    return LLAMA_ROLLBACK_NONE;
}
