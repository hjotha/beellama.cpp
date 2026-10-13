// F5 per-token decode timing: CPU-only unit assertions on server_slot_stats.
//
// These pin the properties the campaign depends on, because two of them were real
// bugs caught by review before any expensive run:
//   * the first interval starts at request/reset time, so it is TTFT and must be
//     EXCLUDED from the decode p95 input;
//   * tokens_per_iteration must come from a per-iteration base captured by
//     f5_begin_iteration(), otherwise it degenerates into 1,2,3,...
//   * the whole bounded capacity is reserved up front, so the decode loop never
//     reallocates.
#include "server-common.h"

#include <cstdio>
#include <cstdlib>

static int failures = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

int main() {
    // 1) disabled by default: every call is a no-op, nothing is stored
    {
        server_slot_stats st;
        check(!st.f5_per_token_enabled, "default is disabled");
        st.f5_arm(1000000);
        st.f5_begin_iteration();
        st.f5_record_token(1100000, false);
        st.f5_close_iteration(100.0);
        check(st.f5_token_ms.empty(), "disabled: no token samples");
        check(st.f5_tokens_recorded == 0, "disabled: no tokens recorded");
    }

    // 2) enabled: full capacity reserved up front (no growth in the hot loop)
    {
        server_slot_stats st;
        st.f5_per_token_enabled = true;
        st.f5_arm(1000000);
        check(st.f5_token_ms.capacity() == server_slot_stats::F5_PER_TOKEN_MAX,
              "token_ms reserved to full capacity");
        check(st.f5_iter_ms.capacity() == server_slot_stats::F5_PER_TOKEN_MAX,
              "iter_ms reserved to full capacity");
        check(st.f5_tokens_per_iter.capacity() == server_slot_stats::F5_PER_TOKEN_MAX,
              "tokens_per_iter reserved to full capacity");
    }

    // 3) 64 tokens, one per iteration: 1 TTFT + 63 decode intervals
    {
        server_slot_stats st;
        st.f5_per_token_enabled = true;
        st.f5_arm(1000000);                 // t = 1000 ms
        int64_t now = 1000000;
        for (int i = 0; i < 64; i++) {
            st.f5_begin_iteration();
            now += 40000;                   // 40 ms per token
            st.f5_record_token(now, false);
            st.f5_close_iteration(40.0);
        }
        check(st.f5_tokens_recorded == 64, "64 tokens recorded");
        check(st.f5_token_ms.size() == 63, "63 decode intervals (first token is TTFT)");
        check(st.f5_iter_ms.size() == 64, "64 decode iterations");
        check(st.f5_decode_iterations == 64, "decode_iterations counter");
        check(!st.f5_truncated, "no truncation at 64 tokens");
        check(!st.f5_spec_seen, "no speculation on the plain path");
        check(st.f5_ttft_ms > 0.0, "TTFT recorded separately");
        for (size_t i = 0; i < st.f5_tokens_per_iter.size(); i++) {
            if (st.f5_tokens_per_iter[i] != 1) {
                check(false, "tokens_per_iteration must be 1 per iteration");
                break;
            }
        }
        // first recorded value must be the 2nd-token interval (40 ms), not TTFT
        check(st.f5_token_ms.size() > 0 && st.f5_token_ms[0] > 39.0 &&
                  st.f5_token_ms[0] < 41.0,
              "first decode interval is the inter-token delta");
    }

    // 4) speculation is flagged so the canonical gate can refuse it
    {
        server_slot_stats st;
        st.f5_per_token_enabled = true;
        st.f5_arm(1000000);
        st.f5_begin_iteration();
        st.f5_record_token(1040000, false);
        st.f5_record_token(1040500, true);   // accepted inside a speculative step
        st.f5_close_iteration(40.5);
        check(st.f5_spec_seen, "speculation flagged");
    }

    // 5) bounded: beyond capacity the capture truncates instead of growing
    {
        server_slot_stats st;
        st.f5_per_token_enabled = true;
        st.f5_arm(1000000);
        int64_t now = 1000000;
        const int64_t n = (int64_t) server_slot_stats::F5_PER_TOKEN_MAX + 32;
        for (int64_t i = 0; i < n; i++) {
            st.f5_begin_iteration();
            now += 1000;
            st.f5_record_token(now, false);
            st.f5_close_iteration(1.0);
        }
        check(st.f5_truncated, "truncation flagged past capacity");
        // The cap is F5_PER_TOKEN_MAX samples; the first token is TTFT and never
        // stored, so the expected count is min(tokens_recorded - 1, capacity), and the
        // vector must never exceed the capacity reserved at arm time.
        const size_t expected = st.f5_tokens_recorded - 1 < (int64_t) server_slot_stats::F5_PER_TOKEN_MAX
                ? (size_t) (st.f5_tokens_recorded - 1)
                : (size_t) server_slot_stats::F5_PER_TOKEN_MAX;
        check(st.f5_token_ms.size() == expected, "decode intervals capped at capacity");
        check(st.f5_token_ms.capacity() == server_slot_stats::F5_PER_TOKEN_MAX,
              "no growth beyond the reserved capacity");
        check(st.f5_tokens_recorded == n, "token counter stays exact even when truncated");
    }

    // 6) re-arming between requests clears everything (no cross-request bleed)
    {
        server_slot_stats st;
        st.f5_per_token_enabled = true;
        st.f5_arm(1000000);
        for (int i = 0; i < 10; i++) {
            st.f5_begin_iteration();
            st.f5_record_token(1000000 + (i + 1) * 1000, false);
            st.f5_close_iteration(1.0);
        }
        st.f5_arm(2000000);
        check(st.f5_token_ms.empty(), "re-arm clears token samples");
        check(st.f5_tokens_recorded == 0, "re-arm clears counters");
        check(st.f5_ttft_ms < 0.0, "re-arm clears TTFT");
    }

    if (failures == 0) {
        printf("OK: f5 per-token timing unit assertions\n");
        return 0;
    }
    fprintf(stderr, "%d assertion(s) failed\n", failures);
    return 1;
}