// LLAMA_MTP_POSITION_SPLIT=1 phase-1 fallback (peça (d)).
//
// The MTP verify batch is the anchor token plus up to n_draft draft tokens.
// With n_draft <= n_rs_seq (automatically true for MTP: n_rs_seq == n_max) the
// whole verify ubatch is the protected recurrent window (1 + n_rs_seq), and
// llama_position_split::divide_ubatches_at_p rejects any window that crosses P
// fail-closed (FAILED_PREPARE). Phase 1 therefore never generates a draft
// whose verify would cross P: the crossing step is a plain non-speculative
// decode of the anchor and drafting resumes once the anchor itself is at/above
// P.
//
// This gate drives the prepare with anchors in [P - n_max, P - 1] and asserts
//   (a) the unfixed verify batch is hard-rejected (the blocker this fallback
//       removes), with no partial output and an explicit reason;
//   (b) the fallback batch is accepted unchanged (state consistent) and never
//       crosses P;
//   (c) the step whose anchor is P-1 is exactly the one-token non-speculative
//       step, and the next anchor (P) restores the full draft limit;
//   (d) the env contract accepts only the exact string "1".

#include "llama-memory-hybrid.h"
#include "llama-position-split.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

static int g_checks = 0;
static int g_failures = 0;

static void check(bool ok, const std::string & what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

// Builds a single-sequence-set token ubatch whose positions start at pos0.
static llama_ubatch make_ubatch(uint32_t n_tokens, llama_pos pos0) {
    llama_ubatch ub;
    ub.b_equal_seqs = 1;
    ub.n_tokens     = n_tokens;
    ub.n_seq_tokens = n_tokens;
    ub.n_seqs       = 1;
    ub.n_seqs_unq   = 1;
    ub.n_pos        = 1;

    auto data = std::make_shared<llama_ubatch::data_t>();
    data->token.resize(n_tokens);
    data->pos.resize(n_tokens);
    data->n_seq_id.assign(n_tokens, 1);
    data->seq_id.resize(n_tokens);
    data->output.assign(n_tokens, 0);
    const llama_seq_id kSeq = 7;
    data->seq_id_unq = { kSeq };
    data->seq_idx.assign(LLAMA_MAX_SEQ, -1);
    data->seq_idx[kSeq] = 0;
    data->seq_id_data.assign(n_tokens, kSeq);

    for (uint32_t i = 0; i < n_tokens; ++i) {
        data->token[i] = llama_token(1000 + i);
        data->pos[i]   = pos0 + llama_pos(i);
        data->seq_id[i] = data->seq_id_data.data() + i;
    }

    ub.data       = data;
    ub.token      = data->token.data();
    ub.embd       = nullptr;
    ub.pos        = data->pos.data();
    ub.n_seq_id   = data->n_seq_id.data();
    ub.seq_id     = data->seq_id.data();
    ub.seq_id_unq = data->seq_id_unq.data();
    ub.seq_idx    = data->seq_idx.data();
    ub.output     = data->output.data();
    return ub;
}

// Runs the phase-1 fallback decision exactly as the server does: clamp the
// requested draft to spec_draft_limit_at_p and build the verify batch.
static llama_ubatch make_verify_ubatch(int64_t anchor_pos, uint32_t n_max, uint32_t p) {
    const int64_t limit = llama_position_split::spec_draft_limit_at_p(anchor_pos, p);
    const uint32_t n_draft = uint32_t(std::min<int64_t>(n_max, std::max<int64_t>(limit, 0)));
    return make_ubatch(n_draft + 1, llama_pos(anchor_pos >= 0 ? anchor_pos : 0));
}

static bool prepare(llama_ubatch & ub, uint32_t p, uint32_t n_keep,
                    std::vector<llama_ubatch> & out, std::string & err) {
    std::vector<llama_ubatch> in;
    in.push_back(std::move(ub));
    return llama_position_split::divide_ubatches_at_p(in, p, n_keep, out, err);
}

int main() {
    const uint32_t P = 1024;           // aligned boundary used by the gate
    const uint32_t N_MAX = 2;          // spec-draft-n-max for the phase-1 smoke
    const uint32_t NKEEP = 1 + N_MAX;  // MTP: n_rs_seq == n_max

    // 0) env contract: only the exact string "1" enables the new mode.
    {
        check(llama_position_split::mtp_position_split_env_enabled("1"),
              "env: exact \"1\" enables the MTP position-split mode");
        check(!llama_position_split::mtp_position_split_env_enabled(nullptr), "env: absent stays off");
        check(!llama_position_split::mtp_position_split_env_enabled(""), "env: empty stays off");
        check(!llama_position_split::mtp_position_split_env_enabled("0"), "env: \"0\" stays off");
        check(!llama_position_split::mtp_position_split_env_enabled("true"), "env: \"true\" stays off");
        check(!llama_position_split::mtp_position_split_env_enabled("01"), "env: \"01\" stays off");
        check(!llama_position_split::mtp_position_split_env_enabled("10"), "env: \"10\" stays off");
        check(!llama_position_split::mtp_position_split_env_enabled(" 1"), "env: leading space stays off");
    }

    // 0b) LLAMA_KV_POSITION_SPLIT_P fallback must NOT resurrect P for the MTP
    //     draft (or any auxiliary) context: common_base_params_to_speculative
    //     zeroes the draft P deliberately and llama_model::create_memory
    //     rejects an MTP context with P != 0. The plain target keeps the env
    //     fallback (the E2E P-via-env path is exercised on device separately;
    //     this CPU gate pins the predicate contract).
    {
        check(llama_position_split::position_split_env_p_fallback_applies(false),
              "env-P fallback: plain target context keeps the fallback");
        check(!llama_position_split::position_split_env_p_fallback_applies(true),
              "env-P fallback: MTP draft / auxiliary context is exempt");
    }

    // Body-only local KVarN must not seal a record after an earlier query
    // inside the same verify batch. Anchor-at-seal is safe; after P the local
    // body is frozen. Check every offset, not just the observed regression.
    {
        const int64_t unlimited = std::numeric_limits<int64_t>::max();
        const auto limit = [](int64_t anchor, uint32_t p = 0, uint32_t seqs = 1) {
            return llama_position_split::spec_draft_limit_before_kvarn_seal(anchor, p, seqs);
        };
        check(limit(1789) == 1, "tail0: anchor125 permits one draft");
        check(limit(1790) == 0, "tail0: anchor126 decodes alone");
        check(limit(1791) == 127, "tail0: anchor127 may cross into next group");
        check(limit(1792) == 126, "tail0: new group permits126 drafts");
        check(limit(-1) == 0 && limit(1790, 0, 2) == 0 && limit(1790, 0, 0) == 0,
                "tail0: invalid anchor/multiple-sequence mapping fails closed");
        check(limit(int64_t(P), P) == unlimited && limit(int64_t(P) + 1, P) == unlimited,
                "tail0: frozen local body after P adds no clamp");
        check(std::min(limit(P - 1, P), llama_position_split::spec_draft_limit_at_p(P - 1, P)) == 0,
                "tail0: existing P-crossing guard still wins at anchor P-1");
        check(std::min(limit(P - 2, P), llama_position_split::spec_draft_limit_at_p(P - 2, P)) == 0,
                "tail0: do not include the local seal immediately before P");
        for (int64_t anchor = 128; anchor < 512; ++anchor) {
            const int64_t n = limit(anchor);
            check(n >= 0 && n <= 127, "tail0: bounded proposal budget");
            bool future_seal = false;
            for (int64_t pos = anchor + 1; pos <= anchor + n; ++pos) {
                future_seal |= pos % llama_position_split::kKvarnGroup == 127;
            }
            check(!future_seal, "tail0: no later query seals a group");
            check((anchor + n + 1) % llama_position_split::kKvarnGroup == 127,
                    "tail0: first excluded query is exactly the next seal");
        }
    }

    // 1) draft limit: never let the verify [anchor, anchor + n_draft] cross P.
    {
        const int64_t kUnlimited = std::numeric_limits<int64_t>::max();
        check(llama_position_split::spec_draft_limit_at_p(0, 0) == kUnlimited,
              "limit: P == 0 leaves the draft unlimited (current mode)");
        check(llama_position_split::spec_draft_limit_at_p(int64_t(P), P) == kUnlimited,
              "limit: anchor at P is unlimited");
        check(llama_position_split::spec_draft_limit_at_p(int64_t(P) + 5, P) == kUnlimited,
              "limit: anchor above P is unlimited");
        check(llama_position_split::spec_draft_limit_at_p(int64_t(P) - 1, P) == 0,
              "limit: anchor P-1 forbids drafts (non-speculative step)");
        check(llama_position_split::spec_draft_limit_at_p(int64_t(P) - 2, P) == 1,
              "limit: anchor P-2 allows exactly one draft");
        check(llama_position_split::spec_draft_limit_at_p(int64_t(P) - 3, P) == 2,
              "limit: anchor P-3 allows the full n_max=2");
        check(llama_position_split::spec_draft_limit_at_p(0, P) == int64_t(P) - 1,
              "limit: anchor 0 keeps the boundary-1 allowance");
        check(llama_position_split::spec_draft_limit_at_p(-1, P) == 0,
              "limit: negative anchor fails closed");
    }

    // 2) (a) the unfixed verify batch crossing P is hard-rejected by the
    //    prepare, before any mutation: anchors in [P - n_max, P - 1].
    {
        for (int64_t anchor = int64_t(P) - N_MAX; anchor <= int64_t(P) - 1; ++anchor) {
            const uint32_t n_draft = N_MAX; // no fallback: ask for the full draft
            llama_ubatch ub = make_ubatch(n_draft + 1, llama_pos(anchor));
            check(llama_position_split::ubatch_crosses_p(uint32_t(anchor), n_draft + 1, P),
                  "reject: verify at anchor " + std::to_string(anchor) + " crosses P");
            std::vector<llama_ubatch> out;
            std::string err;
            const bool ok = prepare(ub, P, NKEEP, out, err);
            check(!ok, "reject: crossing verify at anchor " + std::to_string(anchor) + " is rejected");
            check(out.empty(), "reject: rejection leaves no prepared ubatch");
            check(!err.empty(), "reject: rejection carries an explicit reason");
        }
    }

    // 2b) the rejection class also covers n_draft > n_rs_seq: the protected
    //     window (last 1 + n_rs_seq tokens) straddles P even when the whole
    //     ubatch is not the window. The fallback prevents this form too.
    {
        llama_ubatch ub = make_ubatch(5, llama_pos(P) - 3); // [P-3, P+2), window [P-1,P+2)
        std::vector<llama_ubatch> out;
        std::string err;
        check(!prepare(ub, P, NKEEP, out, err),
              "reject: window straddling P with n_draft > n_rs_seq is rejected");
        check(out.empty(), "reject: window rejection leaves no prepared ubatch");
    }

    // 3) (b) the fallback batch is accepted unchanged and never crosses P:
    //    the prepare must behave as a pass-through (state consistent).
    {
        for (int64_t anchor = int64_t(P) - N_MAX; anchor <= int64_t(P) - 1; ++anchor) {
            llama_ubatch in = make_verify_ubatch(anchor, N_MAX, P);
            const uint32_t expected = uint32_t(std::min<int64_t>(N_MAX, std::max<int64_t>(
                    llama_position_split::spec_draft_limit_at_p(anchor, P), 0))) + 1;
            check(in.n_tokens == expected,
                  "fallback: anchor " + std::to_string(anchor) + " builds the clamped verify size");
            check(!llama_position_split::ubatch_crosses_p(uint32_t(anchor), in.n_tokens, P),
                  "fallback: anchor " + std::to_string(anchor) + " never crosses P");

            llama_ubatch copy = make_verify_ubatch(anchor, N_MAX, P);
            std::vector<llama_ubatch> out;
            std::string err;
            check(prepare(in, P, NKEEP, out, err),
                  "fallback: clamped verify at anchor " + std::to_string(anchor) + " is accepted (" + err + ")");
            check(out.size() == 1,
                  "fallback: clamped verify at anchor " + std::to_string(anchor) + " is not divided");
            if (out.size() == 1) {
                bool same = out[0].n_tokens == copy.n_tokens &&
                            out[0].n_seq_tokens == copy.n_seq_tokens &&
                            out[0].n_seqs == copy.n_seqs &&
                            out[0].n_seqs_unq == copy.n_seqs_unq &&
                            out[0].n_pos == copy.n_pos;
                for (uint32_t i = 0; same && i < copy.n_tokens; ++i) {
                    same = out[0].token[i] == copy.token[i] &&
                           out[0].pos[i] == copy.pos[i] &&
                           out[0].output[i] == copy.output[i] &&
                           out[0].n_seq_id[i] == copy.n_seq_id[i];
                }
                check(same, "fallback: prepared ubatch is byte-identical (no state mutation)");
            }
        }
    }

    // 4) (c) the crossing step itself: anchor P-1 decodes the anchor alone
    //    (non-speculative), and the next anchor at P restores the full draft.
    {
        llama_ubatch crossing_step = make_verify_ubatch(int64_t(P) - 1, N_MAX, P);
        check(crossing_step.n_tokens == 1, "boundary: anchor P-1 yields the one-token non-spec step");
        check(crossing_step.pos[0] == llama_pos(P) - 1, "boundary: the non-spec step keeps the anchor position");

        std::vector<llama_ubatch> out;
        std::string err;
        llama_ubatch step = make_verify_ubatch(int64_t(P) - 1, N_MAX, P);
        check(prepare(step, P, NKEEP, out, err),
              "boundary: the one-token non-spec step at P-1 is accepted (" + err + ")");
        check(out.size() == 1 && out[0].n_tokens == 1, "boundary: the non-spec step is prepared as-is");

        // The anchor at P (the sampled token of the P-1 step) restores n_max.
        llama_ubatch resumed = make_verify_ubatch(int64_t(P), N_MAX, P);
        check(resumed.n_tokens == N_MAX + 1, "boundary: anchor at P restores the full verify size");
        step = make_verify_ubatch(int64_t(P), N_MAX, P);
        check(prepare(step, P, NKEEP, out, err),
              "boundary: the full verify at anchor P is accepted (" + err + ")");
        check(out.size() == 1 && out[0].n_tokens == N_MAX + 1,
              "boundary: the full verify at anchor P is not divided");
    }

    // 5) invariant: every anchor below P and every requested draft size yields
    //    a prepared verify that never crosses P.
    {
        for (int64_t anchor = 0; anchor <= int64_t(P) + 3; ++anchor) {
            llama_ubatch ub = make_verify_ubatch(anchor, N_MAX, P);
            uint32_t pos_min = uint32_t(ub.pos[0]);
            uint32_t pos_max = uint32_t(ub.pos[ub.n_tokens - 1]);
            check(!(pos_min < P && pos_max >= P),
                  "invariant: anchor " + std::to_string(anchor) + " prepared without crossing");
        }
    }

    std::printf("test-position-split-mtp-fallback: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
