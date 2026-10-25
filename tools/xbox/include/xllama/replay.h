// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Controlled single-vs-batch replay (plan 004 boundary diagnosis): teacher-force
// an observed verify block once token-by-token and once as one batch, on
// freshly built identical KV, and compare the full-vocabulary logits for the
// same position. Diagnostic only: answers whether a verify-vs-sequential
// divergence is batching arithmetic (rows differ here) or accumulated state
// (rows match here, so the divergence entered upstream). Never samples, never
// drafts, never changes production paths; rows carry their own self-checks so
// a wrong prefix/position fails loudly instead of comparing garbage.
//
// Indexing (explicit): the failing round's feed is block[0..2] at positions
// [P, P+1, P+2] where P = prefix.size(). Batch row i predicts position P+i+1
// EXCEPT the last row, which predicts P+2 (past the block end, the bonus row
// the loop samples next). The verdict compares the row predicting the
// divergent position: row 2 here (bonus), after rows 0..1 prove the formation
// reproduces the sequential steps. ctx_ids are the committed outputs forming
// the prefix tail; known_next holds the sequential argmax after block[0],
// block[1], and the final row, in order.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xllama {

// One recorded logits row: which branch produced it and its top candidates.
struct ReplayLogits {
    std::string branch; // "single" or "batch"
    int rep = 0;
    int argmax = -1;
    double margin = 0.0; // top1 - top2
    int top[5] = {-1, -1, -1, -1, -1};
    bool finite = true;      // false if any logit is inf or nan
    std::vector<float> full; // full-vocabulary logits, for maxabs/maxrel
};

// Pure comparison of two recorded rows: max |a-b|, max |a-b|/max|a|,
// top-5/margin of each (recomputed) and whether the argmax agrees.
struct ReplayCmp {
    double maxabs = 0.0;
    double maxrel = 0.0;
    bool argmax_match = false;
};

ReplayCmp compare_replay_rows(const std::vector<float>& a, const std::vector<float>& b);

// Runs the replay: prefix = tokenized prompt + ctx_ids (committed outputs),
// then the observed feed block teacher-forced three ways. Branch "seq"
// decodes every token singly (asserting each argmax); branch "acc" decodes
// the kept rows as one batch (asserting rows identically); branch "tail"
// decodes the full feed including the rejected tail row, trims it with the
// same trim_verify_tail call the loop uses, then decodes the corrective
// (asserting each step). known holds the sequential argmax after each forced
// token in order. Each branch runs twice on fresh contexts (determinism must
// be bit-exact). fail_why aborts the run; divergences that must NOT abort
// (the verdict comparisons) are reported per-row. rs_seq/nextn select the
// target context flags (failing runs: 4/on).
//
// Continuation (COMPLETE-CAPTURED-REMAINDER): after the corrective, every
// branch teacher-forces the identical captured remainder rem_ids (live
// schedule: B2 pair, single, B3 triple) and records full logits + argmax at
// all 6 boundaries through the output33 decision. Branch "seq" forces them
// all singly (canonical control separating suffix batching effects).
// Mismatches against rem_known are LOGGED per boundary, never fatal: a
// differing diagnostic branch is the signal. Only hard errors (decode
// failure, missing logits) abort.
//
// Width/trim isolation controls (ISOLATE-WIDTH-AND-TRIM), same
// prefix/config/model/kernels, same remainder afterwards:
// - Branch "d": the block's first three ids plus the corrective as ONE
//   accepted B4 (no trim). Tests whether width4 without rollback suffices.
// - Branch "e": accepted B3, then the rejected id as a single, then the
//   identical trim keeping 3, then the corrective single. Tests the
//   trim/redecode sequence without a B4 graph.
// NOTE (rev73 correction, amended by the rev76 audit — E inference
// RETRACTED): B-vs-C alone does NOT isolate trim: the 4-wide rows already
// differ pre-trim, so B-vs-C conflates width4 + trim. D-vs-C isolates width4
// (no trim ever runs). E does NOT isolate the trim sequence and its outcome
// must NOT be read as trim-alone evidence: E's single-11 decode rewrites
// only snapshot slot 0 ("slot s = s tokens back"; single-token decodes leave
// older slots caller-owned), so the post-trim restore reads stale slot 1
// (through-143, missing token 144) — E tests trim + stale restore. What
// stands: D (width4, no trim) near-flips without reproducing; B (width4 +
// trim, correct slot-1 restore) reproduces. Whether trim adds anything over
// width alone is decided by B-vs-D, not by E.
//
// Rows per rep, in order: seq, acc, tail, seqc0..5, accc0..5, tailc0..5,
// d0..d3, drem0..5, e0..e2, etail, ecorr, erem0..5, accb0..2, btail
// (46 rows; 92 total). accb/btail are the committed-block reference rows so
// every D/E-vs-C CSV comparison uses same-token+position rows. The first
// three keep the original layout so existing evidence columns never shift.
void run_replay_measure(const std::string& model_name, const std::string& prompt_text,
                        const std::vector<int32_t>& ctx_ids, const std::vector<int32_t>& feed_ids,
                        const std::vector<int32_t>& known_next, const std::vector<int32_t>& rem_ids,
                        const std::vector<int32_t>& rem_known, int gpu_layers, int n_ctx,
                        int n_threads, int rs_seq, bool nextn_on, std::vector<ReplayLogits>* out,
                        std::string* fail_why);

} // namespace xllama
