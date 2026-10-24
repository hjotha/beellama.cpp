// CPU fixture for the decode-failure rollback (B) and the cleanup policy (C).
// Simulates a remote/position-split seq_rm refusal without the Xbox/GPU.
#include "llama-rollback.h"
#include "common-rollback.h"

#include <cstdio>

namespace {

struct fake_memory {
    bool suffix_ok = true;
    bool full_ok   = true;
    bool callback_ok = true;
    bool sequence_valid = true;
    int suffix_calls = 0;
    int full_calls   = 0;

    bool operator()(llama_seq_id, llama_pos p0, llama_pos p1) {
        if (p0 < 0 && p1 < 0) {
            ++full_calls;
            if (full_ok) {
                sequence_valid = false;
            }
            return full_ok && callback_ok;
        }
        ++suffix_calls;
        return suffix_ok;
    }
};

int failures = 0;

void check(bool ok, const char * name) {
    std::printf("%s: %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) {
        ++failures;
    }
}

} // namespace

int main() {
    // B1: suffix removal succeeds; the full clear must not run.
    {
        fake_memory f;
        const auto r = llama_rollback_sequence_after_decode_failure(f, 0, 100);
        check(r == LLAMA_ROLLBACK_SUFFIX, "B1 result suffix");
        check(f.suffix_calls == 1 && f.full_calls == 0, "B1 only suffix called");
        check(f.sequence_valid, "B1 sequence remains valid");
    }
    // B2: suffix refused (partial refusal); full clear invalidates the sequence.
    {
        fake_memory f;
        f.suffix_ok = false;
        const auto r = llama_rollback_sequence_after_decode_failure(f, 0, 100);
        check(r == LLAMA_ROLLBACK_FULL, "B2 result full");
        check(f.suffix_calls == 1 && f.full_calls == 1, "B2 suffix then full");
        check(!f.sequence_valid, "B2 sequence invalidated");
    }
    // B3: both refused; error is reported, state untouched, no abort.
    {
        fake_memory f;
        f.suffix_ok = false;
        f.full_ok   = false;
        const auto r = llama_rollback_sequence_after_decode_failure(f, 0, 100);
        check(r == LLAMA_ROLLBACK_NONE, "B3 result none");
        check(f.suffix_calls == 1 && f.full_calls == 1, "B3 both attempted");
        check(f.sequence_valid, "B3 state unchanged");
    }
    // B4: local full clear succeeded but the remote/reset callback refused:
    // the batch must still be reported as a compute error (no retry/reuse).
    {
        fake_memory f;
        f.suffix_ok = false;
        f.callback_ok = false;
        const auto r = llama_rollback_sequence_after_decode_failure(f, 0, 100);
        check(r == LLAMA_ROLLBACK_NONE, "B4 result none");
        check(f.suffix_calls == 1 && f.full_calls == 1, "B4 suffix then full");
        check(!f.sequence_valid, "B4 local clear invalidated the sequence");
    }
    // D: terminal remote failure forces -4 even when local suffix removal worked,
    // so the server never retries it with a smaller batch (ret < -1 path).
    check(llama_decode_aborted_code(false, false) == 2,  "D local abort retryable");
    check(llama_decode_aborted_code(true,  false) == -4, "D rollback-failed terminal");
    check(llama_decode_aborted_code(false, true)  == -4, "D remote-failed terminal");
    check(llama_decode_aborted_code(true,  true)  == -4, "D both-failed terminal");
    // C: cleanup policy (no abort on empty/full-clear refusals; partial fails).
    check( common_seq_rm_cleanup_tolerated(true,  false), "C removed");
    check( common_seq_rm_cleanup_tolerated(false, true), "C empty full tolerated");
    check(!common_seq_rm_cleanup_tolerated(false, false), "C total refusal propagates");
    check(!common_seq_rm_cleanup_tolerated(false, false), "C partial refusal propagates");

    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
