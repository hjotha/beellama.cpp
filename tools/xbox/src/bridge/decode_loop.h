// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Internal: the llama.cpp prefill and generation loops, written in exactly one
// place. Needs llama.h, so it stays out of the public include/ tree.
#pragma once

#include "llama.h"
#include "xllama/chat_prompt.h"  // apply_stop_sequences
#include "xllama/decode_trace.h" // boundary-diagnosis knob (plan 004)
#include "xllama/ggml_d3d12.h"   // shape histogram scope tags
#include "xllama/mtp_draft.h"    // MtpDrafter
#include "xllama/path_utils.h"   // resolve_local_path (diag knob)
#include "xllama/platform.h"     // log_output
#include "xllama/speculative.h"  // prompt_lookup_draft (#210)
#include "xllama/utf8_utils.h"   // utf8_to_wstring (diag knob, UWP)

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace xllama {

// Why this is shared rather than copied: run_inference (CLI, bench) and
// LlamaSession (chat UI, LAN API) ran two hand-maintained copies of the same
// prefill and generation loops, and the copies had already drifted.
//
//   * #193 — a prompt past the logical batch aborted the process, because
//     llama_decode ASSERTS on an oversized batch instead of returning an error.
//     The fix had to be written twice; inference.cpp still carried the comment
//     "same fix as LlamaSession::generate" as evidence.
//   * The stop-sequence token count diverged silently: run_inference counted the
//     token that triggered the stop, LlamaSession did not, so n_eval — and the
//     published decode_tok_s derived from it — differed by one between the two
//     paths for the same generation.
//
// The project had already drawn this conclusion for sampling (#125/#141,
// sampler_chain.h) and stop sequences (chat_prompt.h). This is the piece that
// was left out.
//
// Phase 15 W2 (#210) adds optional draft-free prompt-lookup inside this same
// loop so CLI and Session cannot drift on speculation either.

// Feed |n_tokens| tokens through the context, chunked at the model's logical
// batch. Returns the number of rows the FINAL chunk decoded, or -1 on failure;
// the caller owns what a failure means for its cache, since the two callers
// answer that differently.
//
// The chunking is NOT optional: llama_decode does not return an error for a
// batch larger than n_batch, it trips GGML_ASSERT(n_tokens_all <= n_batch) and
// aborts — in Release too. n_batch defaults to min(n_ctx, 2048) while a 4096-token
// coding session's trimmer ceiling is 3846, so a long paste used to kill the
// process. This is the LOGICAL batch only; the physical ubatch stays at the #172
// optimum of 512, which is what every published prefill rate was measured on.
//
// The row count of the final chunk is load-bearing for MTP (plan 003, F3): the
// drafter indexes the last-batch nextn buffer by row, and deriving it from the
// absolute position (pos_max % n_batch + 1) is wrong whenever the prompt is a
// delta appended after a reused prefix — the positions keep counting while the
// buffer only describes the last chunk.
//
// |after_chunk|, when set, runs right after each chunk's decode with the chunk
// offset, its row count and its first position. The MTP callers use it to feed
// the target's hidden rows of that chunk into the draft context (F2 catch-up):
// the target's nextn buffer describes the LAST batch only, so the replay has
// to happen chunk by chunk, exactly like the reference's process() does.
inline int
prefill_chunked(llama_context* ctx, const llama_token* tokens, int n_tokens,
                const std::function<void(int off, int n_rows, llama_pos pos0)>& after_chunk = {},
                bool profile = false) {
    const int n_batch = std::max(1, static_cast<int>(llama_n_batch(ctx)));
    int last_rows = -1;
    llama_pos pos0 = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
    for (int off = 0; off < n_tokens; off += n_batch) {
        const int chunk = std::min(n_batch, n_tokens - off);
        llama_batch batch = llama_batch_get_one(const_cast<llama_token*>(tokens) + off, chunk);
        // Same-boundary prefill bracket (owner 599): DSTEP kind=prefill per
        // chunk with the same wall/d3w/d3g/cpu/calls/mm accounting as the
        // decode steps, so the ubatch effect is attributable at stage level.
        // Self-contained (the detail::StepSplit definition lives further down
        // in this header, after this function).
        const bool prof = profile;
        const auto t0 = prof ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
        const double w0 = prof ? d3d12_wall_ms() : 0.0;
        const double g0 = prof ? d3d12_gpu_ms() : 0.0;
        const double c0 = prof ? process_cpu_ms() : 0.0;
        const std::uint64_t k0 = prof ? d3d12_graph_calls() : 0;
        const std::uint64_t m0 = prof ? d3d12_matmul_count() : 0;
        const int rc = llama_decode(ctx, batch);
        if (prof) {
            const double wall = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0)
                                    .count();
            char b[224];
            snprintf(b, sizeof(b),
                     "[xllama] DSTEP kind=prefill width=%d wall=%.2f d3w=%.2f d3g=%.2f cpu=%.2f "
                     "calls=%llu mm=%llu\n",
                     chunk, wall, d3d12_wall_ms() - w0, d3d12_gpu_ms() - g0,
                     process_cpu_ms() - c0,
                     static_cast<unsigned long long>(d3d12_graph_calls() - k0),
                     static_cast<unsigned long long>(d3d12_matmul_count() - m0));
            log_output(b);
        }
        if (rc != 0)
            return -1;
        last_rows = chunk;
        if (after_chunk)
            after_chunk(off, chunk, pos0);
        pos0 += static_cast<llama_pos>(chunk);
    }
    return last_rows;
}

// Rows in the batch the nextn embeddings were last written by, or 0.
//
// Unmasked nextn rows are indexed within the last batch (see decode_loop), so a
// drafter starting straight after a chunked prefill needs the size of the FINAL
// chunk, not the whole prompt: the prefill splits at n_batch and the buffer only
// describes whatever llama_decode ran last. This is the fallback for callers
// that do not know the real count; prefill_chunked now returns it, and
// DecodeLoopParams.mtp_prefill_rows carries it into the loop (plan 003, F3).
// Deriving from the absolute position (pos_max % n_batch + 1) is wrong for a
// delta appended after a reused prefix: positions keep counting past the chunk.
inline int last_nextn_rows(llama_context* ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    const llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
    if (pos_max < 0)
        return 0;
    const int n_batch = std::max(1, static_cast<int>(llama_n_batch(ctx)));
    const int pos = static_cast<int>(pos_max);
    // The final chunk starts at the largest multiple of n_batch at or below pos.
    return pos % n_batch + 1;
}

// The message for a prompt that cannot fit the context at all. Chunking makes an
// oversized BATCH safe, not an oversized CONTEXT — without this the run would
// fail somewhere inside the loop with a bare "decode failed".
inline std::string prompt_too_long_message(int n_tokens, int n_ctx) {
    return "prompt too long: " + std::to_string(n_tokens) +
           " tokens exceed n_ctx=" + std::to_string(n_ctx);
}

struct DecodeLoopParams {
    llama_context* ctx = nullptr;
    llama_sampler* sampler = nullptr;
    const llama_vocab* vocab = nullptr;
    int n_predict = 0;
    const std::vector<std::string>* stop_sequences = nullptr;
    const std::atomic<bool>* abort_flag = nullptr;
    // Streamed piece-by-piece to the caller (chat UI token stream, CLI echo).
    std::function<void(std::string_view)> on_token;
    // CLI only: mirror the stream to stdout as it decodes.
    bool echo_stdout = false;
    // Called after a token has been accepted into the cache. LlamaSession uses it
    // to keep m_kv_tokens in step with the KV cells, which is what the #170a
    // prefix diff and the #170b snapshot fingerprint both read.
    std::function<void(llama_token)> on_accepted;

    // Collect every token accepted into the KV, in order (plan 003, stage 1:
    // per-run token parity sidecar). Null = off. Prompt-lookup's own history is
    // token_history; this is an independent collector so the CLI path can dump
    // tokens without enabling drafting.
    std::vector<llama_token>* tokens_out = nullptr;

    // Start of the decode phase, used to expose time-to-first-token.
    std::chrono::steady_clock::time_point decode_start{};

    // Phase 15 W2 (#210): draft-free prompt-lookup speculative decoding.
    // Default OFF. Requires token_history seeded with the prefill tokens.
    // History ownership: when on_accepted is set it must update the same
    // vector (Session: on_accepted pushes m_kv_tokens); when null the loop
    // appends to token_history itself (CLI path).
    bool prompt_lookup = false;
    std::vector<llama_token>* token_history = nullptr;
    int spec_n_gram = kSpecNgramDefault;
    int spec_k = kSpecDraftKDefault;

    // MTP drafting. When set and ready(), it replaces prompt-lookup as the draft
    // source; both feed the same verify path below, so acceptance and the
    // rewind rules are shared. Null leaves decoding exactly as it was.
    class MtpDrafter* mtp = nullptr;
    int mtp_n_embd = 0;

    // Real row count of the last prefill chunk (plan 003, F3). 0 = unknown:
    // fall back to last_nextn_rows(), which is wrong for a delta appended after
    // a reused prefix. The caller gets it from prefill_chunked().
    int mtp_prefill_rows = 0;

    // Bench only: keep decoding through end-of-generation so every run decodes
    // exactly n_predict tokens. Two backends whose arithmetic differs (CPU q8
    // activations vs d3d12 f32) reach EOG at different points, and a decode
    // tok/s over a handful of tokens is noise (D2b). Default off.
    bool ignore_eog = false;

    // Phase instrumentation on/off (InferenceParams::profile_phases). With it off
    // the per-phase chrono snapshots are skipped while every counter still runs, so
    // an ON/OFF pair measures the instrumentation's own cost.
    bool profile_phases = true;

    // Parity/diagnostic sink for the ids this loop accepted, in emission order.
    // Null in every production path, so the hot loop pays nothing; the Session
    // sets it when a caller asks for a per-turn token dump (plan 003 stage 1:
    // greedy parity must compare integral ids, not text).
    std::vector<llama_token>* out_token_ids = nullptr;

    // Plan 009 S3 probe: byte capture of per-step decisions. Empty in every
    // production path (the Session arms it only when LocalState\chat-dump.flag
    // exists), so the hot loop pays one empty-string check per emit. Each
    // emitted token appends: int32 token id (native endian) followed by
    // n_vocab f32 logits of the target context's LAST row at emission (the
    // row the sample was read from on the classic path; MTP-verify rounds
    // emit from their own verify rows, so the row index is part of the
    // compare, never assumed). Fail-closed: any failure logs once and
    // disables the capture; generation never depends on it.
    std::string logits_dump_path;
    mutable bool logits_dump_failed = false;
    // Snapshot of the logits row that DECIDED the token about to be emitted.
    // Captured at each sample site (row -1 for classic/anchor, row i-1 for
    // verify-accepted candidates), because emission of the anchor happens
    // after a decode has already overwritten the context's last row.
    mutable std::vector<float> capture_row;
    mutable bool capture_valid = false;
};

struct DecodeLoopResult {
    int n_generated = 0;
    double first_token_ms = 0.0;
    bool ended_with_stop = false; // a textual stop sequence matched
    // Termination-state gate (plan 003): which path ended generation. The
    // EOG token id + branch record a natural end directly from the result
    // (never inferred from output length); the stop branch + loop round
    // prove which path/round a stop-sequence landed in. -1/nullptr = the
    // run ended without that event (n_predict cap or abort).
    int eog_token = -1;
    const char* eog_branch = nullptr;
    const char* stop_branch = nullptr;
    int stop_round = -1;
    // Speculative counters (zero when prompt_lookup is off or never drafted).
    int n_drafted = 0;  // draft tokens proposed (not counting the lead sample)
    int n_accepted = 0; // draft tokens that matched the target sample
    // Per-source split of the two counters above. The aggregate is still what the
    // CSV reports, but "MTP drafted something" is only answerable from these:
    // with both sources on, n_drafted mixes n-gram and MTP proposals.
    int n_mtp_drafted = 0;     // proposals that came from the MTP head
    int n_lookup_drafted = 0;  // proposals that came from the n-gram lookup
    int n_mtp_accepted = 0;    // accepted from MTP proposals
    int n_lookup_accepted = 0; // accepted from lookup proposals
    // Phase accounting (plan 003, stage 2): how the MTP budget is spent, so the
    // cost can be attributed before any backend rewrite. All values are summed
    // over the whole generation; the drafter's per-call stats feed them.
    int n_mtp_rounds = 0;         // MTP rounds with a non-empty draft batch
    int n_mtp_decodes = 0;        // draft-context llama_decode calls (draft + catch-up)
    int n_mtp_discarded = 0;      // draft candidates rejected by the p_min gate
    int n_catchup_tokens = 0;     // target tokens replayed into the draft context
    double mtp_draft_ms = 0.0;    // draft-context decode time
    double mtp_sample_ms = 0.0;   // draft sampler time (incl. the p_min gate)
    double mtp_top_prob_ms = 0.0; // confidence softmax time
    double mtp_catchup_ms = 0.0;  // prefix replay decode time
    double verify_ms = 0.0;       // target verify-batch decode time
    // Per-verify-round split (plan003: attribute verify's share of decode).
    // Profile-only, empty when OFF. wall = the verify llama_decode's own
    // wall (same clock as verify_ms, so sum(vr_wall) <= verify_ms with the
    // bracket granularity); d3w = the D3D12 backend's graph_compute wall
    // inside that call (submission+wait+driver), d3g = its GPU timestamps —
    // nesting is gpu <= d3w <= wall <= verify_ms, so nothing is counted
    // twice; wall - d3w is CPU-splits + scheduler + cross-backend sync (the
    // lump current instrumentation cannot split further). acc0 =
    // n_accepted BEFORE the round's accept walk; the final round's accepted
    // count comes from out.n_accepted at loop end.
    std::vector<int> vr_width;
    // Process-CPU delta (all threads) of the same verify call, ms
    // (process_cpu_ms). Read the limit with the value: process-wide, so it
    // may exceed vr_wall under parallel threads; it counts CPU burned only —
    // blocked/waiting time is absent and scheduler-vs-kernel work is not
    // separable from this number alone.
    std::vector<double> vr_cpu;
    // Graph-reuse count inside the same verify call (llama_perf_context
    // n_reused delta; needs cparams.no_perf=false). 1 = previous graph
    // reused (prep = set_inputs only), 0 = graph rebuilt (reset + build_graph
    // + sched_alloc_graph). Profile-only, plan004 rebuild-vs-reuse split.
    std::vector<int> vr_reuse;
    std::vector<double> vr_wall;
    std::vector<double> vr_d3w;
    std::vector<double> vr_d3g;
    std::vector<int> vr_acc0;
    // Same-boundary D3D12 dispatch counts (plan 006 scheduler-split
    // attribution): graph_compute calls and matmul dispatches inside the same
    // verify bracket as vr_d3w. Profile-only; counts, never timing claims.
    std::vector<std::uint64_t> vr_calls;
    std::vector<std::uint64_t> vr_mm;
    // Per-round proposal IDs (the verify feed: anchor + drafts) so OFF/ON arms
    // can be compared by proposal identity, not just accepted counts.
    // Profile-only.
    std::vector<std::vector<llama_token>> vr_feed;
    double corrective_ms = 0.0; // immediate corrective decode time (reject path)
    double t_lookup_ms = 0.0;   // n-gram lookup drafting (only when enabled)
    // Plan 005: same-boundary whole-decode split. Deltas of the D3D12 backend
    // wall/GPU counters and of process CPU over THIS loop's interval, so the
    // sequential and MTP paths are ranked on identical boundaries (prefill and
    // model load are outside the loop by construction). Nesting: d3g <= d3w <=
    // t_decode_ms; cpu is process-wide and may exceed the wall. Profile-only,
    // zero reads when OFF.
    double win_d3w_ms = 0.0;
    double win_d3g_ms = 0.0;
    double win_cpu_ms = 0.0;
    // Whole-decode D3D12 dispatch counts over the same window (profile-only).
    std::uint64_t win_calls = 0;
    std::uint64_t win_mm = 0;
    // Classic single-token step sub-split on the same clocks; the sequential
    // path has no verify bracket, so this is its per-token d3w/d3g/cpu view.
    // Profile-only.
    double classic_d3w_ms = 0.0;
    double classic_d3g_ms = 0.0;
    double classic_cpu_ms = 0.0;
    int classic_steps = 0;
    // Per-token costs that live OUTSIDE any llama_decode: sampling from the target,
    // token emission/detokenisation, and the maintenance around an accepted token
    // (KV bookkeeping, stop-sequence scan). They are the bulk of what was previously
    // unattributed, and they are disjoint from the decode timers by construction.
    double t_sample_target_ms = 0.0;   // llama_sampler_sample on the TARGET context
    double t_emit_ms = 0.0;            // emit_token: detokenise + append + stops
    double t_maintain_ms = 0.0;        // accept_token + per-token bookkeeping
    double t_classic_ms = 0.0;         // classic single-token DECODE only (decode_one).
                                       // It does NOT include the MTP catch-up below.
    double t_classic_catchup_ms = 0.0; // MTP replay triggered by a classic token
    // Wall clock of THIS loop and the sum of the phase timers inside it, over the
    // same interval. Their difference is the uninstrumented residual; without
    // both, "the phases look small" could just mean the phases are incomplete.
    double t_decode_ms = 0.0;
    double t_decode_accounted_ms = 0.0;
    // True when a needed llama_memory_seq_rm refused after a multi-token verify
    // batch. Callers must treat the generation as failed and drop the KV —
    // continuing would desync history from cells (hybrid/LFM caches).
    bool rewind_failed = false;
};

// Rollback window for speculative decoding on a hybrid (recurrent + attention)
// cache. The recurrent R/S rows must cover the draft depth: llama_memory_seq_rm
// refuses a tail rewind on a hybrid cache otherwise, which is what aborted every
// MTP run deeper than one token. Mirrors common_params_speculative::need_n_rs_seq
// and the n_ubatch clamp from common.cpp.
inline uint32_t speculative_n_rs_seq(bool mtp, int mtp_n_max) {
    if (!mtp || mtp_n_max <= 0)
        return 0;
    return static_cast<uint32_t>(mtp_n_max);
}

// The window plus the token being decoded has to fit one micro-batch, or the
// recurrent memory cannot keep it and the fork falls back to KV checkpoints.
// Same condition as the original: effective n_ubatch must exceed n_rs_seq + 1.
inline void clamp_speculative_n_rs_seq(llama_context_params& cparams, uint32_t n_rs_seq) {
    const uint32_t n_batch_eff = static_cast<uint32_t>(
        cparams.n_ctx > 0 ? std::min<uint32_t>(cparams.n_batch, cparams.n_ctx) : cparams.n_batch);
    const uint32_t n_ubatch_eff =
        cparams.n_ubatch == 0 ? n_batch_eff
                              : std::min(n_batch_eff, static_cast<uint32_t>(cparams.n_ubatch));
    if (n_rs_seq > 0 && n_ubatch_eff <= n_rs_seq + 1) {
        log_output("[xllama] speculative rollback window (" + std::to_string(n_rs_seq) +
                   " + 1) does not fit micro-batch " + std::to_string(n_ubatch_eff) +
                   "; disabling it (raise -ub and -b to " + std::to_string(n_rs_seq + 2) +
                   " to enable)\n");
        cparams.n_rs_seq = 0;
        return;
    }
    cparams.n_rs_seq = n_rs_seq;
}

namespace detail {

// Emit a sampled token into the output stream. Returns true if a stop sequence
// ended generation (caller should still count the token).
inline bool emit_token(const DecodeLoopParams& p, llama_token token, std::string& output_text,
                       DecodeLoopResult* out = nullptr) {
    if (p.out_token_ids)
        p.out_token_ids->push_back(token);
    // Plan 009 S3 probe: per-step byte capture (armed by LocalState\chat-dump.flag).
    if (!p.logits_dump_path.empty() && !p.logits_dump_failed) {
        const int32_t n_vocab = llama_vocab_n_tokens(p.vocab);
        if (p.capture_valid && n_vocab > 0 &&
            static_cast<int>(p.capture_row.size()) == n_vocab) {
            if (FILE* df = std::fopen(p.logits_dump_path.c_str(), "ab")) {
                const int32_t tok = static_cast<int32_t>(token);
                const size_t w1 = std::fwrite(&tok, sizeof(tok), 1, df);
                const size_t w2 =
                    std::fwrite(p.capture_row.data(), sizeof(float),
                                static_cast<size_t>(n_vocab), df);
                std::fclose(df);
                if (w1 != 1 || w2 != static_cast<size_t>(n_vocab)) {
                    log_output("[xllama] WARN: S3 logits capture short write; disabled\n");
                    p.logits_dump_failed = true;
                }
            } else {
                log_output("[xllama] WARN: S3 logits capture open failed; disabled\n");
                p.logits_dump_failed = true;
            }
        } else {
            log_output("[xllama] WARN: S3 logits row snapshot missing at emit; "
                       "capture disabled\n");
            p.logits_dump_failed = true;
        }
    }
    // Detokenisation is real per-token work that no decode timer covered.
    const bool prof = out && p.profile_phases;
    const auto t_emit0 =
        prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    char buf[256] = {};
    const int len = llama_token_to_piece(p.vocab, token, buf, sizeof(buf) - 1, 0, false);
    if (len > 0) {
        buf[len] = '\0';
        output_text += buf;
        if (p.on_token)
            p.on_token(std::string_view(buf, static_cast<size_t>(len)));
        if (p.echo_stdout) {
            std::fputs(buf, stdout);
            std::fflush(stdout);
        }
    }
    if (prof)
        out->t_emit_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_emit0)
                .count();
    return p.stop_sequences && apply_stop_sequences(output_text, *p.stop_sequences);
}

// Record an accepted token in history / session bookkeeping.
inline void accept_token(const DecodeLoopParams& p, llama_token token) {
    if (p.tokens_out)
        p.tokens_out->push_back(token);
    if (p.on_accepted)
        p.on_accepted(token);
    else if (p.token_history)
        p.token_history->push_back(token);
}

// The one end-of-generation decision for every branch of decode_loop (classic,
// pre-draft, draft commit, spec reject), so ignore_eog cannot drift between them.
inline bool stops_at_eog(const DecodeLoopParams& p, llama_token token) {
    return llama_vocab_is_eog(p.vocab, token) && !p.ignore_eog;
}

// Plan 003 stage 2: RAII tag so the shape histogram can attribute each graph to
// the context and phase that issued it. The drafter's private context and the
// target share this thread, so a tag set for one call must not leak into the next —
// the destructor restores whatever was in force before.
struct ScopeTag {
    const char* prev_ctx;
    const char* prev_phase;
    ScopeTag(const char* ctx, const char* phase) {
        d3d12_get_scope(&prev_ctx, &prev_phase);
        d3d12_set_scope(ctx, phase);
    }
    ~ScopeTag() {
        d3d12_set_scope(prev_ctx, prev_phase);
    }
    ScopeTag(const ScopeTag&) = delete;
    ScopeTag& operator=(const ScopeTag&) = delete;
};

// Plan 005: same-boundary per-step split for decode calls that have no VROUND
// bracket (classic is CSTEP, verify is VROUND, this covers the draft context's
// draft/catch-up). One DSTEP line per call, profile only; zero clock reads when
// OFF. Nesting: d3g <= d3w <= wall; cpu is process-wide and may exceed wall.
struct StepSplit {
    const char* kind;
    int width;
    bool on;
    std::chrono::steady_clock::time_point t0{};
    double d3w0 = 0.0, d3g0 = 0.0, cpu0 = 0.0;
    std::uint64_t calls0 = 0, mm0 = 0;
    StepSplit(bool profile, const char* k, int w) : kind(k), width(w), on(profile) {
        if (!on)
            return;
        t0 = std::chrono::steady_clock::now();
        d3w0 = d3d12_wall_ms();
        d3g0 = d3d12_gpu_ms();
        cpu0 = process_cpu_ms();
        calls0 = d3d12_graph_calls();
        mm0 = d3d12_matmul_count();
    }
    ~StepSplit() {
        if (!on)
            return;
        const double wall =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count();
        char b[224];
        snprintf(b, sizeof(b),
                 "[xllama] DSTEP kind=%s width=%d wall=%.2f d3w=%.2f d3g=%.2f cpu=%.2f "
                 "calls=%llu mm=%llu\n",
                 kind, width, wall, d3d12_wall_ms() - d3w0, d3d12_gpu_ms() - d3g0,
                 process_cpu_ms() - cpu0,
                 static_cast<unsigned long long>(d3d12_graph_calls() - calls0),
                 static_cast<unsigned long long>(d3d12_matmul_count() - mm0));
        log_output(b);
    }
    StepSplit(const StepSplit&) = delete;
    StepSplit& operator=(const StepSplit&) = delete;
};

// Per-token maintenance around an accepted token: history bookkeeping only.
//
// The stop-sequence scan is NOT repeated here: emit_token already calls
// apply_stop_sequences, and calling it twice was duplicated work introduced by the
// instrumentation — it would also mutate output_text twice per token. Measurement
// stays observation-only.
inline void account_maintain(const DecodeLoopParams& p, llama_token token, DecodeLoopResult& out,
                             bool profile) {
    const auto t_m0 =
        profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    accept_token(p, token);
    if (profile)
        out.t_maintain_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_m0)
                .count();
}

// Decode one already-sampled token (classic path).
inline bool decode_one(llama_context* ctx, llama_token token) {
    llama_batch next = llama_batch_get_one(&token, 1);
    return llama_decode(ctx, next) == 0;
}

// Multi-token verify batch: lead sample + draft, all positions request logits
// so the same sampler chain can sample at each batch index (#210 W2.2).
// Positions are explicit from the current seq_pos_max so auto-tracking cannot
// disagree with seq_rm on a partial accept.
inline bool decode_verify_batch(llama_context* ctx, const std::vector<llama_token>& feed) {
    if (feed.empty())
        return true;
    llama_batch batch = llama_batch_init(static_cast<int32_t>(feed.size()), 0, 1);
    llama_memory_t mem = llama_get_memory(ctx);
    const llama_pos pos0 = llama_memory_seq_pos_max(mem, 0) + 1;
    for (size_t i = 0; i < feed.size(); ++i) {
        batch.token[i] = feed[i];
        batch.pos[i] = pos0 + static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1; // need a logit row per position for verification
    }
    batch.n_tokens = static_cast<int32_t>(feed.size());
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

// Drop KV cells past the first |n_keep| tokens of a verify batch that began
// after |pos_before|. Returns false if the memory refuses (no mutation on
// hybrid caches — #183 / #210 W2.3).
inline bool trim_verify_tail(llama_memory_t mem, llama_pos pos_before, int n_keep, int n_feed) {
    if (n_keep >= n_feed)
        return true;
    const llama_pos keep = pos_before + static_cast<llama_pos>(n_keep) + 1;
    return llama_memory_seq_rm(mem, 0, keep, -1);
}

// Trace helper (plan 004 boundary diagnosis): top-5 token:logit pairs and the
// top-2 margin of one logits row, as a single log-safe line. Only called when
// the trace knob names the output index being decided.
inline std::string trace_topk(llama_context* ctx, const llama_vocab* vocab, int32_t row) {
    if (!ctx || !vocab)
        return "ctx-or-vocab-null";
    const int32_t n = llama_vocab_n_tokens(vocab);
    const float* logits = llama_get_logits_ith(ctx, row);
    if (!logits || n <= 0)
        return "logits-unavailable";
    int top[5] = {-1, -1, -1, -1, -1};
    for (int32_t i = 0; i < n; ++i) {
        const float v = logits[i];
        for (int k = 0; k < 5; ++k) {
            if (top[k] < 0 || v > logits[top[k]]) {
                for (int s = 4; s > k; --s)
                    top[s] = top[s - 1];
                top[k] = i;
                break;
            }
        }
    }
    char buf[256]{};
    size_t len = 0;
    for (int k = 0; k < 5 && top[k] >= 0; ++k) {
        const int written = snprintf(buf + len, sizeof(buf) - len, "%s%d:%.4f", k ? " " : "",
                                     top[k], static_cast<double>(logits[top[k]]));
        if (written < 0 || static_cast<size_t>(written) >= sizeof(buf) - len)
            return buf;
        len += static_cast<size_t>(written);
    }
    const double margin = top[1] >= 0 ? static_cast<double>(logits[top[0]]) - logits[top[1]] : 0.0;
    snprintf(buf + len, sizeof(buf) - len, " margin=%.6f", margin);
    return buf;
}

// History for n-gram search = prefill/prior accepted tokens + the lead sample
// not yet recorded.
inline std::vector<int32_t> history_for_draft(const DecodeLoopParams& p, llama_token lead) {
    std::vector<int32_t> h;
    if (p.token_history) {
        h.reserve(p.token_history->size() + 1);
        for (llama_token t : *p.token_history)
            h.push_back(static_cast<int32_t>(t));
    }
    h.push_back(static_cast<int32_t>(lead));
    return h;
}

// MTP catch-up for one target batch (plan 003, F2): replay |n_rows| tokens with
// the target's hidden rows (shifted by one, first row carrying the drafter's
// pending state) into the private draft context. Called from the prefill
// after_chunk hook and after every target verify batch, exactly when the
// target's nextn buffer still describes the batch that just decoded.
// |profile| is deliberately mandatory: a defaulted value silently enabled the
// profile-only DSTEP clocks on the prefill path while profile was OFF (rev129
// measurement-integrity defect), so every caller must pass its own flag.
inline bool mtp_catchup_batch(class MtpDrafter* mtp, llama_context* ctx, const llama_token* tokens,
                              int n_rows, llama_pos pos0, int n_embd, bool profile) {
    if (!mtp || !mtp->ready() || n_rows <= 0 || !tokens)
        return true;
    detail::StepSplit step(profile, "catchup", n_rows);
    // The target's buffer describes the LAST decode — this batch. Copy the rows
    // out before anything else can overwrite them.
    std::vector<float> h_rows(static_cast<size_t>(n_rows) * static_cast<size_t>(n_embd));
    for (int i = 0; i < n_rows; ++i) {
        const float* hr = llama_get_embeddings_nextn_ith(ctx, i);
        if (!hr)
            return false;
        std::memcpy(h_rows.data() + static_cast<size_t>(i) * n_embd, hr,
                    static_cast<size_t>(n_embd) * sizeof(float));
    }
    // |profile| reaches the drafter's own chrono snapshots: the OFF arm has to be
    // real at BOTH levels or it still measures instrumentation-ON.
    return mtp->process(tokens, h_rows.data(), nullptr, n_rows, pos0, profile);
}

// Classic single-token step. Returns false to stop the outer loop.
// Sets |stop| when a stop sequence matched; |decode_ok| false on decode error.
inline bool classic_step(const DecodeLoopParams& p, llama_token token, std::string& output_text,
                         DecodeLoopResult& out, bool& stop, bool& decode_ok) {
    stop = false;
    decode_ok = true;
    if (emit_token(p, token, output_text, &out)) {
        out.ended_with_stop = true;
        ++out.n_generated;
        stop = true;
        log_output("[xllama] stop sequence after " + std::to_string(out.n_generated) + " tokens\n");
        return false;
    }
    const auto t_cls0 = p.profile_phases ? std::chrono::steady_clock::now()
                                         : std::chrono::steady_clock::time_point{};
    const double cls_d3w0 = p.profile_phases ? d3d12_wall_ms() : 0.0;
    const double cls_d3g0 = p.profile_phases ? d3d12_gpu_ms() : 0.0;
    const double cls_cpu0 = p.profile_phases ? process_cpu_ms() : 0.0;
    ScopeTag scope("target", "classic");
    const bool dec_ok = decode_one(p.ctx, token);
    if (p.profile_phases) {
        const double cls_wall =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cls0)
                .count();
        const double d3w = d3d12_wall_ms() - cls_d3w0;
        const double d3g = d3d12_gpu_ms() - cls_d3g0;
        const double cpu = process_cpu_ms() - cls_cpu0;
        out.t_classic_ms += cls_wall;
        out.classic_d3w_ms += d3w;
        out.classic_d3g_ms += d3g;
        out.classic_cpu_ms += cpu;
        ++out.classic_steps;
        char clb[192];
        snprintf(clb, sizeof(clb),
                 "[xllama] CSTEP kind=classic width=1 wall=%.2f d3w=%.2f d3g=%.2f cpu=%.2f\n",
                 cls_wall, d3w, d3g, cpu);
        log_output(clb);
    }
    if (!dec_ok) {
        log_output("[xllama] decode failed at token, stopping generation\n");
        decode_ok = false;
        return false;
    }
    account_maintain(p, token, out, p.profile_phases);
    ++out.n_generated;
    // A classic token commits to the shared KV without a verify batch, so the
    // drafter's pending row — and its mirror of the committed prefix — would
    // drift one position behind the target. Replay the single row now, the
    // same F2 catch-up the verify path runs, so the next draft round (or the
    // next session turn) starts from the right carry. Fails closed: a refused
    // replay leaves the drafter stale and the next round's catch-up disables
    // it (F2.3).
    if (p.mtp && p.mtp->ready()) {
        const llama_pos pos = llama_memory_seq_pos_max(llama_get_memory(p.ctx), 0);
        // Snapshot delta: this catch-up's own decodes and time. Read as a
        // cumulative total it would double-count the preceding draft and then be
        // erased by the next reset_stats().
        const MtpDraftStats before_cls = p.mtp->stats();
        ScopeTag scope("draft", "catchup");
        if (!detail::mtp_catchup_batch(p.mtp, p.ctx, &token, 1, pos, p.mtp_n_embd,
                                       p.profile_phases)) {
            log_output("[xllama] mtp: classic catch-up failed at pos " + std::to_string(pos) +
                       "\n");
        } else {
            const MtpDraftStats after_cls = p.mtp->stats();
            out.n_mtp_decodes += after_cls.n_decodes - before_cls.n_decodes;
            // process() charges its llama_decode to decode_ms and the catch-up to
            // catchup_ms, so a delta here lands in draft_ms — otherwise that
            // second decode would be counted nowhere and inflate the residual.
            out.mtp_draft_ms += after_cls.decode_ms - before_cls.decode_ms;
            out.mtp_catchup_ms += after_cls.catchup_ms - before_cls.catchup_ms;
            out.n_catchup_tokens += 1;
            out.t_classic_catchup_ms += (after_cls.catchup_ms - before_cls.catchup_ms) +
                                        (after_cls.decode_ms - before_cls.decode_ms);
        }
    }
    return true;
}

} // namespace detail

// Generate up to |n_predict| tokens, appending decoded text to |output_text|.
//
// n_generated counts every token the model produced, INCLUDING the one that
// triggered a stop sequence. That token was sampled and rendered — the cost is
// real and t_eval contains it — so leaving it out understates decode_tok_s.
// run_inference already counted it and LlamaSession did not; this unifies on the
// counting version, which is why LlamaSession's figure moves by one token on a
// stop-sequence finish.
//
// When prompt_lookup is on, the lead token is always committed with the same
// classic path as non-spec (so greedy output matches). Drafts are then verified
// with a multi-token batch of *only* the draft tokens (W2.2). Rejected tails
// use llama_memory_seq_rm; a refused rewind sets rewind_failed (W2.3).
//
// Correctness note (2026-08-07): feeding [lead, draft...] in one batch and
// sampling index 0 for the first draft diverged from sequential greedy even
// with n_spec_accepted=0 — batch logits after lead were not equivalent to a
// single-token decode of lead on this stack. Commit lead first, then speculate.
inline DecodeLoopResult decode_loop(const DecodeLoopParams& p, std::string& output_text) {
    DecodeLoopResult out;
    // Same-interval accounting (plan 003 stage 2): every phase timer below is
    // summed into t_decode_accounted_ms and compared against t_decode_ms, so a
    // missing path shows up as residual instead of being invisible.
    //
    // This ONE clock, and t_first_token_ms, are deliberately NOT gated on
    // profile_phases: they are the denominator and the published TTFT, and an OFF
    // arm that dropped them could not be compared against the ON arm at all. The
    // OFF arm removes the per-phase snapshots only, which is what the perturbation
    // measurement is about.
    const auto t_loop0 = std::chrono::steady_clock::now();
    const double win_d3w0 = p.profile_phases ? d3d12_wall_ms() : 0.0;
    const double win_d3g0 = p.profile_phases ? d3d12_gpu_ms() : 0.0;
    const double win_cpu0 = p.profile_phases ? process_cpu_ms() : 0.0;
    const std::uint64_t win_calls0 = p.profile_phases ? d3d12_graph_calls() : 0;
    const std::uint64_t win_mm0 = p.profile_phases ? d3d12_matmul_count() : 0;
    const bool mtp_enabled = p.mtp != nullptr && p.mtp->ready();
    // Mutable: a catch-up or seq_rm failure leaves the private draft context in
    // an unknown state, and drafting on top of it is worse than not drafting
    // (plan 003, F2.3). The loop then falls back to single-token decoding for
    // the rest of the generation.
    bool mtp_ok = mtp_enabled;
    // The target enables nextn embeddings unmasked, and in that mode
    // llama_get_embeddings_nextn_ith is indexed by ROW WITHIN THE LAST BATCH, not by
    // token position: llama-context.cpp writes the rows at n_tokens_prev, a local that
    // resets to 0 on every llama_decode, and reads them back as a dense index.
    // common/speculative.cpp relies on exactly this -- it asks for
    // i_batch_beg[seq_id] + i and carries forward row n_rows - 1.
    //
    // The draft needs the hidden row of the LAST DECODED token, i.e. the last row of
    // whichever batch ran last. Track that row count rather than a position:
    // deriving it from seq_pos_max is wrong, because that is a position rather than
    // a row, so subtracting 1 lands two tokens back.
    // Seed from the prefill: the first draft would otherwise read row 0, which is
    // the FIRST prompt token rather than the last one decoded.
    //
    // F3 (plan 003): the caller reports the REAL row count of the last prefill
    // chunk (prefill_chunked returns it). last_nextn_rows() is only a fallback:
    // pos_max % n_batch + 1 is wrong for a delta appended after a reused prefix.
    const int prefill_rows = p.mtp_prefill_rows > 0 ? p.mtp_prefill_rows : last_nextn_rows(p.ctx);
    int mtp_carry_row = mtp_enabled ? prefill_rows - 1 : 0;
    if (mtp_carry_row < 0)
        mtp_carry_row = 0;
    auto set_carry_rows = [&](int n_rows) { mtp_carry_row = n_rows > 0 ? n_rows - 1 : 0; };
    // Speculation is ON when either source is usable. The two are independent: the
    // n-gram lookup is opt-in (p.prompt_lookup) and must stay opt-in even when MTP
    // is active — with only this flag, MTP proposing nothing fell through to
    // prompt_lookup_draft() and its tokens were counted as MTP drafts.
    const bool lookup_enabled = p.prompt_lookup && p.token_history != nullptr && p.spec_k > 0;
    bool spec_enabled = mtp_enabled || lookup_enabled;

    // Plan 009 (single-token variant of replay_committed below): replay one
    // just-decoded token so the carry covers it. Function scope (used both
    // inside the walk and at the F4.1 loop-end flush): |tok| at |pos| with
    // its row live (caller guarantees: single decode just ran). Same
    // fail-closed contract and accounting shape; no sampling, commits, or
    // output change.
    auto replay_one = [&](llama_token tok, llama_pos pos) {
        if (!(mtp_ok && p.mtp != nullptr && !out.rewind_failed))
            return;
        const MtpDraftStats before_1 = p.mtp->stats();
        detail::ScopeTag scope("draft", "catchup");
        if (!detail::mtp_catchup_batch(p.mtp, p.ctx, &tok, 1, pos, p.mtp_n_embd,
                                       p.profile_phases)) {
            log_output("[xllama] mtp: single-row catch-up failed — falling back to classic "
                       "decoding\n");
            mtp_ok = false;
            spec_enabled = false;
        } else {
            const MtpDraftStats after_1 = p.mtp->stats();
            out.n_catchup_tokens += 1;
            out.n_mtp_decodes += after_1.n_decodes - before_1.n_decodes;
            out.mtp_catchup_ms += after_1.catchup_ms - before_1.catchup_ms;
        }
    };

    // F4.1 (plan 003): a corrected token from a rejected MTP draft is emitted
    // immediately but NOT decoded right away — it becomes the anchor of the next
    // round, so the correction shares that round's verify batch instead of
    // costing its own target decode. The token was already sampled (do not
    // sample it again) and already counted; only its KV insertion is deferred.
    // The pending hidden row is captured now, before any decode overwrites the
    // buffer. If the loop ends with a pending correction, the final flush
    // decodes it so the Session contract (m_kv_tokens == KV cells) holds.
    bool mtp_pending_anchor = false;
    llama_token mtp_pending_tok = LLAMA_TOKEN_NULL;
    llama_pos mtp_pending_pos = -1;
    std::vector<float> mtp_pending_h;

    // Termination-state gate: 1-based round counter for stop/abort records.
    int loop_iter = 0;
    while (out.n_generated < p.n_predict) {
        ++loop_iter;
        if (p.abort_flag && p.abort_flag->load()) {
            // Termination-state gate (plan 003): one line per armed abort so a
            // headless run can prove WHERE the cancel landed (round boundary
            // vs mid-round trim below). Log-only.
            log_output("[xllama] abort: round boundary n_generated=" +
                       std::to_string(out.n_generated) + "\n");
            break;
        }

        // Boundary trace (plan 004): every round's entry state when armed. The
        // idx-specific decision traces below show WHAT was decided; these lines
        // show the KV end each round started from, so untrimmed rejects or
        // over-appends (kv growing past committed outputs) are visible as a
        // drift between kv and the output index across rounds.
        if (xllama::decode_trace::output_idx() >= 0) {
            char rb[192];
            snprintf(rb, sizeof(rb), "[xllama] TRACE round idx=%d anchor=%d kv=%d mtp_ok=%d\n",
                     out.n_generated, mtp_pending_anchor ? 1 : 0,
                     static_cast<int>(llama_memory_seq_pos_max(llama_get_memory(p.ctx), 0)),
                     mtp_ok ? 1 : 0);
            log_output(rb);
        }

        // The anchor of a pending correction was sampled from the target logits
        // of the previous batch; sampling again would consume the sampler's
        // state twice. Use it verbatim.
        llama_token token = LLAMA_TOKEN_NULL;
        const bool anchor_pending = mtp_pending_anchor;
        if (anchor_pending) {
            token = mtp_pending_tok;
        } else {
            const auto t_sm0 = p.profile_phases ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
            token = llama_sampler_sample(p.sampler, p.ctx, -1);

            if (!p.logits_dump_path.empty()) {
                const int32_t nv = llama_vocab_n_tokens(p.vocab);
                const float* sr = llama_get_logits_ith(p.ctx, -1);
                if (sr && nv > 0) {
                    p.capture_row.assign(sr, sr + nv);
                    p.capture_valid = true;
                }
            }            if (p.profile_phases)
                out.t_sample_target_ms += std::chrono::duration<double, std::milli>(
                                              std::chrono::steady_clock::now() - t_sm0)
                                              .count();
            if (detail::stops_at_eog(p, token)) {
                out.eog_token = static_cast<int>(token);
                out.eog_branch = "classic";
                log_output("[xllama] EOG after " + std::to_string(out.n_generated) +
                           " tokens tok=" + std::to_string(token) + " branch=classic\n");
                break;
            }
            // Boundary trace (plan 004): the logits just sampled for the output
            // token about to be decided. Same row (-1) feeds the classic path
            // and the MTP anchor alike, so this one point covers both.
            if (!anchor_pending && out.n_generated == xllama::decode_trace::output_idx() &&
                xllama::decode_trace::output_idx() >= 0) {
                char tb[384];
                snprintf(tb, sizeof(tb), "[xllama] TRACE pre idx=%d tok=%d top5(row -1)=%s\n",
                         out.n_generated, token, detail::trace_topk(p.ctx, p.vocab, -1).c_str());
                log_output(tb);
            }
        }

        // MTP drafts from the anchor *before* the anchor is decoded, exactly like
        // the original common/speculative.cpp: |token| was sampled from the
        // previous logits but never fed to the target, and it pairs with the
        // hidden row of the last decoded position. That lets the anchor and the
        // whole draft share ONE target decode. Decoding the anchor first (what
        // this did before) cost two target decodes per round to gain at most
        // n_max tokens, which cannot beat the single-token path even at 100%
        // acceptance.
        bool lead_in_batch = false;
        std::vector<llama_token> feed;
        if (mtp_ok) {
            // seq_pos_max is the last written position, so the anchor sits one past
            // it and is the position the first draft would occupy. A pending
            // correction already occupies its slot.
            const llama_pos P = anchor_pending
                                    ? mtp_pending_pos
                                    : llama_memory_seq_pos_max(llama_get_memory(p.ctx), 0) + 1;
            const float* h = anchor_pending ? mtp_pending_h.data()
                                            : llama_get_embeddings_nextn_ith(p.ctx, mtp_carry_row);
            if (h) {
                // F4.2: never draft past the output budget or the context end.
                // The anchor plus each draft occupies one position; a draft token
                // that could not be emitted would only be verified and discarded.
                const int budget_left = p.n_predict - out.n_generated;
                const int ctx_left = static_cast<int>(llama_n_ctx(p.ctx)) - static_cast<int>(P);
                int n_max_eff = p.mtp->params().n_max;
                n_max_eff = std::min(n_max_eff, std::max(0, budget_left - 1));
                n_max_eff = std::min(n_max_eff, std::max(0, ctx_left - 1));
                // MtpDraftStats is CUMULATIVE until the next reset_stats(), so every read has to
                // be a snapshot delta around the one call it belongs to. Adding the
                // raw cumulative value at two points double-counted the draft
                // decodes and then reset_stats() erased the catch-up's contribution
                // before anything read it.
                p.mtp->reset_stats();
                detail::StepSplit draft_step(p.profile_phases, "draft", n_max_eff);
                detail::ScopeTag scope("draft", "draft");
                const std::vector<llama_token> md =
                    p.mtp->draft(token, P, h, p.mtp_n_embd, n_max_eff, p.profile_phases);
                const MtpDraftStats st = p.mtp->stats(); // snapshot AFTER draft()
                out.n_mtp_decodes += st.n_decodes;
                out.n_mtp_discarded += st.n_discarded;
                out.mtp_draft_ms += st.decode_ms;
                out.mtp_sample_ms += st.sample_ms;
                out.mtp_top_prob_ms += st.top_prob_ms;
                if (!md.empty()) {
                    ++out.n_mtp_rounds;
                    out.n_mtp_drafted += static_cast<int>(md.size());
                    out.n_drafted += static_cast<int>(md.size());
                    feed.reserve(md.size() + 1);
                    feed.push_back(token); // anchor: sampled, not yet decoded
                    for (llama_token t : md)
                        feed.push_back(t);
                    lead_in_batch = true;
                }
            } else {
                log_output("[xllama] mtp: no nextn row for carry row " +
                           std::to_string(mtp_carry_row) + "; declining to draft\n");
            }
        }

        // |first| is the pre-batch sample of the classic path. Hoisted so the
        // post-batch fallbacks can still reach it when the lead is in the batch.
        llama_token first = LLAMA_TOKEN_NULL;
        if (!lead_in_batch) {
            // A pending correction with no draft batch still has to enter the KV
            // (it was emitted and counted already): decode it classically, then
            // continue with a fresh sample on the next iteration.
            if (anchor_pending) {
                const auto t_c0 = p.profile_phases ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};
                const bool ok = detail::decode_one(p.ctx, token);
                if (p.profile_phases)
                    out.corrective_ms += std::chrono::duration<double, std::milli>(
                                             std::chrono::steady_clock::now() - t_c0)
                                             .count();
                if (!ok) {
                    log_output("[xllama] decode failed after speculative reject\n");
                    break;
                }
                detail::account_maintain(p, token, out, p.profile_phases);
                mtp_pending_anchor = false;
                mtp_pending_tok = LLAMA_TOKEN_NULL;
                if (mtp_ok) {
                    detail::ScopeTag scope("draft", "catchup");
                    detail::mtp_catchup_batch(
                        p.mtp, p.ctx, &token, 1,
                        mtp_pending_pos >= 0 ? mtp_pending_pos
                                             : llama_memory_seq_pos_max(llama_get_memory(p.ctx), 0),
                        p.mtp_n_embd, p.profile_phases);
                }
                set_carry_rows(1);
                if (out.n_generated >= p.n_predict)
                    break;
                continue;
            }
            // Always commit the sampled token via the classic path first. Spec never
            // changes this step — that is what keeps greedy text identical.
            if (out.n_generated == 0 && out.first_token_ms == 0.0 &&
                p.decode_start != std::chrono::steady_clock::time_point{}) {
                out.first_token_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - p.decode_start)
                                         .count();
            }
            bool stop = false, decode_ok = true;
            if (!detail::classic_step(p, token, output_text, out, stop, decode_ok)) {
                if (stop) { // stop-sequence, not a decode failure
                    out.stop_branch = "classic";
                    out.stop_round = loop_iter;
                }
                break;
            }
            set_carry_rows(1); // classic_step is a one-row decode
            if (out.n_generated >= p.n_predict)
                break;

            if (!spec_enabled)
                continue;

            // Lead is already in token_history via accept_token. Draft from that
            // alone (no extra lead argument). Reached only when the lookup is
            // explicitly enabled: with MTP alone, an empty proposal must fall
            // through to the next classic step, never into a second drafter.
            if (!lookup_enabled)
                continue;

            std::vector<int32_t> draft32;
            if (p.token_history && static_cast<int>(p.token_history->size()) >= p.spec_n_gram) {
                std::vector<int32_t> hist;
                hist.reserve(p.token_history->size());
                for (llama_token t : *p.token_history)
                    hist.push_back(static_cast<int>(t));
                const auto t_look0 = p.profile_phases ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point{};
                draft32 = prompt_lookup_draft(hist, p.spec_n_gram, p.spec_k);
                if (p.profile_phases)
                    out.t_lookup_ms += std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() - t_look0)
                                           .count();
            }
            if (draft32.empty())
                continue; // decline: no draft evidence, no extra cost

            // Counted separately from MTP: n_drafted_aggregate alone cannot say
            // which source proposed, and a gate that reads it as "MTP drafted"
            // would pass on n-gram tokens alone.
            out.n_lookup_drafted += static_cast<int>(draft32.size());

            // Verify draft[0] against the logits we already have after the lead
            // (same sample classic would take next). If it disagrees, that sample
            // *is* the true next token — classic_step it and skip the batch. When the
            // lead is already inside the batch there is nothing to pre-check: row 0 of
            // that batch is the first real verification.
            if (!lead_in_batch) {
                const auto t_sm1 = p.profile_phases ? std::chrono::steady_clock::now()
                                                    : std::chrono::steady_clock::time_point{};
                first = llama_sampler_sample(p.sampler, p.ctx, -1);

            if (!p.logits_dump_path.empty()) {
                const int32_t nv = llama_vocab_n_tokens(p.vocab);
                const float* sr = llama_get_logits_ith(p.ctx, -1);
                if (sr && nv > 0) {
                    p.capture_row.assign(sr, sr + nv);
                    p.capture_valid = true;
                }
            }                if (p.profile_phases)
                    out.t_sample_target_ms += std::chrono::duration<double, std::milli>(
                                                  std::chrono::steady_clock::now() - t_sm1)
                                                  .count();
                if (detail::stops_at_eog(p, first)) {
                    out.eog_token = static_cast<int>(first);
                    out.eog_branch = "pre-draft";
                    log_output("[xllama] EOG after " + std::to_string(out.n_generated) +
                               " tokens (pre-draft) tok=" + std::to_string(first) +
                               " branch=pre-draft\n");
                    break;
                }
                if (first != static_cast<llama_token>(draft32[0])) {
                    bool stop = false, decode_ok = true;
                    if (!detail::classic_step(p, first, output_text, out, stop, decode_ok)) {
                        if (stop) {
                            out.stop_branch = "pre-draft";
                            out.stop_round = loop_iter;
                        }
                        break;
                    }
                    continue;
                }
            }

            // first == draft[0]: at least one draft token is free. Batch-decode all
            // drafts; logits[i] predict the token after draft[i].
            feed.reserve(draft32.size());
            for (int32_t d : draft32)
                feed.push_back(static_cast<llama_token>(d));
        } // !lead_in_batch

        // One target decode now covers the committed lead plus every draft:
        // logits[i] predict feed[i+1], and the last row predicts the bonus.
        const int n_feed = static_cast<int>(feed.size());

        llama_memory_t mem = llama_get_memory(p.ctx);
        const llama_pos pos_before = llama_memory_seq_pos_max(mem, 0);

        {
            const auto t_v0 = p.profile_phases ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
            // Per-round backend split (plan003): snapshot the D3D12
            // backend's own wall/GPU around this verify call only — no other
            // backend work interleaves inside the synchronous llama_decode.
            double d3w0 = 0.0, d3g0 = 0.0, cpu0 = 0.0;
            int32_t reuse0 = 0;
            std::uint64_t calls0 = 0, mm0 = 0;
            if (p.profile_phases) {
                d3w0 = d3d12_wall_ms();
                d3g0 = d3d12_gpu_ms();
                cpu0 = process_cpu_ms();
                reuse0 = llama_perf_context(p.ctx).n_reused; // rebuild vs reuse
                calls0 = d3d12_graph_calls();
                mm0 = d3d12_matmul_count();
            }
            detail::ScopeTag scope("target", "verify");
            const bool vok = detail::decode_verify_batch(p.ctx, feed);
            if (p.profile_phases) {
                const double wall_v = std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - t_v0)
                                          .count();
                out.verify_ms += wall_v;
                out.vr_width.push_back(n_feed);
                out.vr_wall.push_back(wall_v);
                out.vr_d3w.push_back(d3d12_wall_ms() - d3w0);
                out.vr_d3g.push_back(d3d12_gpu_ms() - d3g0);
                out.vr_cpu.push_back(process_cpu_ms() - cpu0);
                out.vr_reuse.push_back(llama_perf_context(p.ctx).n_reused - reuse0);
                out.vr_calls.push_back(d3d12_graph_calls() - calls0);
                out.vr_mm.push_back(d3d12_matmul_count() - mm0);
                out.vr_feed.push_back(feed);
                out.vr_acc0.push_back(out.n_accepted);
            }
            if (!vok) {
                log_output("[xllama] speculative draft batch failed — classic for first "
                           "match\n");
                if (!detail::trim_verify_tail(mem, pos_before, /*n_keep=*/0, n_feed)) {
                    out.rewind_failed = true;
                    break;
                }
                // Nothing entered the KV, so take one classic token. With the lead in
                // the batch the trimmed logits are the pre-batch ones and |first| is
                // stale, so resample; otherwise |first| already matched draft[0].
                const llama_token next = [&] {
                    const auto t_sm2 = p.profile_phases ? std::chrono::steady_clock::now()
                                                        : std::chrono::steady_clock::time_point{};
                    const llama_token t =
                        lead_in_batch ? llama_sampler_sample(p.sampler, p.ctx, -1) : first;

            if (!p.logits_dump_path.empty()) {
                const int32_t nv = llama_vocab_n_tokens(p.vocab);
                const float* sr = llama_get_logits_ith(p.ctx, -1);
                if (sr && nv > 0) {
                    p.capture_row.assign(sr, sr + nv);
                    p.capture_valid = true;
                }
            }                    if (p.profile_phases)
                        out.t_sample_target_ms += std::chrono::duration<double, std::milli>(
                                                      std::chrono::steady_clock::now() - t_sm2)
                                                      .count();
                    return t;
                }();
                if (detail::stops_at_eog(p, next)) {
                    out.eog_token = static_cast<int>(next);
                    out.eog_branch = "spec-verify";
                    log_output("[xllama] EOG after " + std::to_string(out.n_generated) +
                               " tokens (spec-verify) tok=" + std::to_string(next) +
                               " branch=spec-verify\n");
                    break;
                }
                bool stop = false, decode_ok = true;
                if (!detail::classic_step(p, next, output_text, out, stop, decode_ok)) {
                    if (stop) {
                        out.stop_branch = "spec-verify";
                        out.stop_round = loop_iter;
                    }
                    break;
                }
                continue;
            }
        }

        // Boundary trace (plan 004): the exact verify-batch inputs behind the
        // outputs about to be decided. Positions are explicit (pos_before+1+i,
        // same formula the batch uses), so a replay harness can teacher-force
        // the identical block instead of guessing drafts.
        if (xllama::decode_trace::output_idx() >= 0) {
            char fb[320];
            int flen = snprintf(fb, sizeof(fb),
                                "[xllama] TRACE feed idx=%d n=%d pos0=%d toks=", out.n_generated,
                                n_feed, static_cast<int>(pos_before) + 1);
            for (int fi = 0; fi < n_feed && flen < 280; ++fi)
                flen += snprintf(fb + flen, sizeof(fb) - static_cast<size_t>(flen), "%s%d",
                                 fi ? "," : "", feed[static_cast<size_t>(fi)]);
            log_output((std::string(fb) + "\n").c_str());
        }

        auto fail_rewind = [&]() {
            log_output("[xllama] speculative: seq_rm refused — aborting generation "
                       "(hybrid/SWA caches cannot tail-rewind; disable prompt_lookup "
                       "for this model)\n");
            out.rewind_failed = true;
            spec_enabled = false;
        };

        // feed[0] is already committed in both paths: with lead_in_batch it is the
        // anchor sampled from the target itself, otherwise draft[0] just verified
        // equal to |first|. Either way row 0 of the batch is decided, so the walk
        // starts at i = 1 with one row kept.
        int n_keep = 1;
        auto commit_draft = [&](llama_token tok) -> bool {
            // returns false → stop outer loop
            if (detail::stops_at_eog(p, tok)) {
                out.eog_token = static_cast<int>(tok);
                out.eog_branch = "spec";
                log_output("[xllama] EOG after " + std::to_string(out.n_generated) +
                           " tokens (spec) tok=" + std::to_string(tok) + " branch=spec\n");
                return false;
            }
            if (detail::emit_token(p, tok, output_text, &out)) {
                out.ended_with_stop = true;
                out.stop_branch = "spec";
                out.stop_round = loop_iter;
                ++out.n_generated;
                return false;
            }
            detail::account_maintain(p, tok, out, p.profile_phases);
            ++out.n_generated;
            return true;
        };

        // feed[0] is the anchor when MTP put it there, and the lead sample otherwise;
        // either way this batch verified the tokens this round proposed, so the
        // acceptance is attributed to whichever source filled the feed.
        const bool batch_from_mtp = lead_in_batch;
        if (!lead_in_batch) {
            ++out.n_accepted;
            ++out.n_lookup_accepted;
            if (!commit_draft(first)) {
                if (!detail::trim_verify_tail(mem, pos_before, n_keep, n_feed))
                    fail_rewind();
                break;
            }
        } else {
            // The anchor joins the batch, so feed[0] is the first token this round can
            // emit and the classic path above never ran. Stamp first_token_ms here or
            // a leading MTP round leaves it at 0 and the latency reads as unset.
            if (out.first_token_ms == 0.0 &&
                p.decode_start != std::chrono::steady_clock::time_point{}) {
                out.first_token_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - p.decode_start)
                                         .count();
            }
            if (anchor_pending) {
                // The anchor of a pending correction was emitted and counted in the
                // previous round; this batch put it in the KV. Record it in the
                // session bookkeeping without emitting or counting it again.
                detail::account_maintain(p, feed[0], out, p.profile_phases);
                mtp_pending_anchor = false;
                mtp_pending_tok = LLAMA_TOKEN_NULL;
            } else if (!commit_draft(feed[0])) {
                if (!detail::trim_verify_tail(mem, pos_before, n_keep, n_feed))
                    fail_rewind();
                break;
            }
        }

        bool stop_all = false;
        // Set once the reject path has already trimmed and then appended the
        // correction. The generic tail trim at the bottom would otherwise remove
        // the very token that was just decoded: it recomputes the same `keep`
        // boundary and deletes |cand| along with the rejected drafts.
        bool tail_already_trimmed = false;
        // Plan 009 turn-end carry: replay the committed prefix into the draft
        // mirror so the pending row covers the final committed token. This is
        // the F2 body factored — the normal path below calls it too, with
        // identical behavior. The committed count is read back from the KV
        // ground truth, so EOG/stop cut-offs are covered; rows must be live
        // (callers guarantee: the verify batch just decoded, nothing decoded
        // since). Fail-closed exactly like the normal path (F2.3). No
        // sampling, no commits, no output change. NOT used on error paths
        // (decode/trim failure: rows untrustworthy) nor for the corrective
        // single decode (its token is not feed[]-addressable; left for a
        // follow-up if ever needed — the turn still ends correctly, just
        // without a refreshed carry).
        auto replay_committed = [&] {
            if (!(mtp_ok && p.mtp != nullptr && !out.rewind_failed))
                return;
            const int n_committed = static_cast<int>(llama_memory_seq_pos_max(mem, 0) - pos_before);
            // Plan 009 diagnostic (behind kv_reuse_diag.txt): replay span vs
            // ground truth, so an off-by-one between committed extents and
            // replayed rows is visible instead of inferred. Temporary.
            {
                FILE* df = nullptr;
                {
                    const std::string dp = ::xllama::resolve_local_path("kv_reuse_diag.txt");
#ifdef XLLAMA_UWP
                    df = _wfopen(::xllama::utf8_to_wstring(dp).c_str(), L"rb");
#else
                    df = std::fopen(dp.c_str(), "rb");
#endif
                }
                if (df) {
                    const bool on = std::fgetc(df) == '1';
                    std::fclose(df);
                    if (on) {
                        char db[256];
                        snprintf(db, sizeof(db),
                                 "[xllama] mtp: replay n=%d pos_before=%d kv_end=%d mirror_end=%d "
                                 "gen=%d nkeep=%d nfeed=%d\n",
                                 n_committed, static_cast<int>(pos_before),
                                 static_cast<int>(llama_memory_seq_pos_max(mem, 0)),
                                 p.mtp ? static_cast<int>(llama_memory_seq_pos_max(
                                             llama_get_memory(p.mtp->ctx()), 0))
                                       : -2,
                                 out.n_generated, n_keep, n_feed);
                        log_output(db);
                    }
                }
            }
            const MtpDraftStats before_cu = p.mtp->stats();
            detail::ScopeTag scope("draft", "catchup");
            if (!detail::mtp_catchup_batch(p.mtp, p.ctx, feed.data(), n_committed, pos_before + 1,
                                           p.mtp_n_embd, p.profile_phases)) {
                log_output("[xllama] mtp: catch-up failed — falling back to classic "
                           "decoding\n");
                mtp_ok = false;
                spec_enabled = false;
            } else if (n_committed > 0) {
                const MtpDraftStats after_cu = p.mtp->stats();
                out.n_catchup_tokens += n_committed;
                out.n_mtp_decodes += after_cu.n_decodes - before_cu.n_decodes;
                out.mtp_catchup_ms += after_cu.catchup_ms - before_cu.catchup_ms;
            }
        };
        for (size_t i = 1; i < feed.size(); ++i) {
            if (out.n_generated >= p.n_predict || (p.abort_flag && p.abort_flag->load())) {
                // Termination-state gate: fires only for an armed abort, so
                // "cancelled during a speculative round" is evidenced by this
                // line plus the trim that follows (log-only).
                if (p.abort_flag && p.abort_flag->load())
                    log_output("[xllama] abort: mid-round trim i=" + std::to_string(i) + "/" +
                               std::to_string(feed.size()) +
                               " n_generated=" + std::to_string(out.n_generated) + "\n");
                if (!detail::trim_verify_tail(mem, pos_before, n_keep, n_feed))
                    fail_rewind();
                // Plan 009: replay the committed prefix before exiting, so the
                // carry covers the final token (no-op on error paths via the
                // lambda's own guards).
                replay_committed();
                stop_all = true;
                break;
            }
            // Logits after feed[i-1] (batch index i-1) predict feed[i].
            const llama_token cand = [&] {
                const auto t_sm3 = p.profile_phases ? std::chrono::steady_clock::now()
                                                    : std::chrono::steady_clock::time_point{};
                const llama_token t =
                    llama_sampler_sample(p.sampler, p.ctx, static_cast<int32_t>(i - 1));
                    if (!p.logits_dump_path.empty()) {
                        const int32_t nv = llama_vocab_n_tokens(p.vocab);
                        const float* sr = llama_get_logits_ith(
                            p.ctx, static_cast<int32_t>(i - 1));
                        if (sr && nv > 0) {
                            p.capture_row.assign(sr, sr + nv);
                            p.capture_valid = true;
                        }
                    }
                if (p.profile_phases)
                    out.t_sample_target_ms += std::chrono::duration<double, std::milli>(
                                                  std::chrono::steady_clock::now() - t_sm3)
                                                  .count();
                return t;
            }();
            const llama_token drafted = feed[i];
            const bool accepted = (cand == drafted);
            // Diagnostic-only (bench_swiglu_mtp_diag.txt): target row vs draft
            // at every verify step, with the committed-history accounting.
            if (xllama::d3d12_swiglu_mtp_diag_enabled()) {
                char vb[512];
                snprintf(vb, sizeof(vb),
                         "[xllama] MTPDIAG verify row=%d drafted=%d cand=%d accepted=%d "
                         "top5=%s n_feed=%d n_keep=%d kv_end=%d rollback=%d rng=greedy\n",
                         static_cast<int>(i) - 1, drafted, cand, accepted ? 1 : 0,
                         detail::trace_topk(p.ctx, p.vocab, static_cast<int32_t>(i) - 1).c_str(),
                         n_feed, n_keep,
                         static_cast<int>(llama_memory_seq_pos_max(llama_get_memory(p.ctx), 0)),
                         accepted ? 0 : (n_feed - n_keep));
                log_output(vb);
            }
            // Boundary trace (plan 004): the verify decision behind the output
            // token about to be emitted (out.n_generated is exactly its index:
            // every earlier feed slot was already emitted or counted). Carries
            // the batch row, the draft, the sampled candidate, the verdict,
            // and the row's top candidates with margin, so a mismatch against
            // the sequential pre-trace attributes to batching numerics (same
            // top-1, or different top-1 with a margin dwarfing drift) vs
            // accept/rollback/state (verdict contradicts its own row).
            // kv_end is read here, mid-walk: it must equal pos_before + n_feed
            // (the batch fully appended, trim not yet run). The post-trim
            // value is logged after the walk below; the two together prove KV
            // accounting round-trips through accept AND trim.
            if (out.n_generated == xllama::decode_trace::output_idx() &&
                xllama::decode_trace::output_idx() >= 0) {
                char tb[512];
                snprintf(tb, sizeof(tb),
                         "[xllama] TRACE verify idx=%d batch_row=%d tok=%d drafted=%d "
                         "accepted=%d top5=%s feed_n=%d pos_before=%d n_keep=%d kv_end=%d\n",
                         out.n_generated, static_cast<int>(i) - 1, cand, drafted, accepted ? 1 : 0,
                         detail::trace_topk(p.ctx, p.vocab, static_cast<int32_t>(i) - 1).c_str(),
                         n_feed, static_cast<int>(pos_before), n_keep,
                         static_cast<int>(llama_memory_seq_pos_max(llama_get_memory(p.ctx), 0)));
                log_output(tb);
            }
            if (accepted) {
                ++out.n_accepted;
                if (batch_from_mtp)
                    ++out.n_mtp_accepted;
                else
                    ++out.n_lookup_accepted;
                ++n_keep;
                if (!commit_draft(cand)) {
                    // Plan 009: this cand hit EOG/stop — it was counted above
                    // but never mirrored (commit failed before the push), yet
                    // it sits decoded in the KV tail. Exclude it from the
                    // extents BEFORE trim+replay, or trim keeps its row, the
                    // replay covers it, and the stash describes a token the
                    // mirror lacks — breaking can_carry on every EOG turn.
                    --n_keep;
                    if (!detail::trim_verify_tail(mem, pos_before, n_keep, n_feed))
                        fail_rewind();
                    // Plan 009: EOG/stop inside the walk still leaves committed
                    // tokens behind — replay them so the carry is fresh.
                    replay_committed();
                    stop_all = true;
                    break;
                }
                continue;
            }
            // Reject: keep accepted drafts, drop the rest, commit |cand|.
            if (!detail::trim_verify_tail(mem, pos_before, n_keep, n_feed)) {
                fail_rewind();
                stop_all = true;
                break;
            }
            if (detail::stops_at_eog(p, cand)) {
                out.eog_token = static_cast<int>(cand);
                out.eog_branch = "spec-reject";
                log_output("[xllama] EOG after " + std::to_string(out.n_generated) +
                           " tokens (spec reject) tok=" + std::to_string(cand) +
                           " branch=spec-reject\n");
                replay_committed();
                stop_all = true;
                break;
            }
            if (detail::emit_token(p, cand, output_text, &out)) {
                out.ended_with_stop = true;
                out.stop_branch = "spec-reject";
                out.stop_round = loop_iter;
                ++out.n_generated;
                replay_committed();
                stop_all = true;
                break;
            }
            // cand is not yet in the KV (only accepted drafts are).
            if (mtp_ok && p.mtp != nullptr) {
                // F4.1: defer the correction decode to the next round's anchor
                // slot; the token was already sampled from these logits, so the
                // next round must not sample it again. Capture the hidden row of
                // the last accepted line NOW — the next decode overwrites the
                // target's nextn buffer.
                mtp_pending_anchor = true;
                mtp_pending_tok = cand;
                mtp_pending_pos = pos_before + static_cast<llama_pos>(n_keep) + 1;
                const float* h_keep = llama_get_embeddings_nextn_ith(p.ctx, n_keep - 1);
                if (h_keep) {
                    mtp_pending_h.assign(h_keep, h_keep + p.mtp_n_embd);
                } else {
                    mtp_pending_h.clear();
                }
            } else {
                const auto t_c0 = p.profile_phases ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};
                const bool ok = detail::decode_one(p.ctx, cand);
                if (p.profile_phases)
                    out.corrective_ms += std::chrono::duration<double, std::milli>(
                                             std::chrono::steady_clock::now() - t_c0)
                                             .count();
                if (!ok) {
                    log_output("[xllama] decode failed after speculative reject\n");
                    stop_all = true;
                    break;
                }
                detail::account_maintain(p, cand, out, p.profile_phases);
            }
            ++out.n_generated;
            tail_already_trimmed = true;
            // Plan 009 note: no replay here — the normal F2 catch-up below
            // covers this correction (KV ground truth includes it), and an
            // extra replay would add a trim-refusal disable path that legacy
            // never had (its refusal kills drafting for the rest of the
            // turn). The F4.1 loop-end replay stays: nothing follows it.
            break;
        }

        if (out.rewind_failed)
            break;
        if (stop_all)
            break;
        if (n_keep < n_feed && !tail_already_trimmed) {
            if (!detail::trim_verify_tail(mem, pos_before, n_keep, n_feed)) {
                fail_rewind();
                break;
            }
        }
        // Carry the hidden row of the last token this round left in the KV.
        // tail_already_trimmed means the reject path appended |cand| with its own
        // single-token decode, so the carry is that one row; otherwise the whole
        // verify batch stands and its last row is the freshest.
        set_carry_rows(tail_already_trimmed ? 1 : n_feed);

        // F2 catch-up (plan 003): replay into the private draft context the
        // tokens this round actually committed — the verify batch may have
        // rejected a tail, and replaying rejected tokens would make the draft
        // head attend to a history the target never committed. The committed
        // count is read back from the KV (the ground truth), so EOG/stop
        // cut-offs inside the walk are covered too. The first replay row
        // carries the drafter's pending row, which the last process() call
        // left at the last committed position; a failure disables drafting for
        // the rest of the generation (F2.3: never speculate on an unknown
        // state). Body factored as replay_committed() (plan 009) — identical.
        replay_committed();
    }

    // F4.1 contract: a pending correction was emitted and counted but never
    // decoded. Satisfy the Session state (KV + m_kv_tokens) before returning.
    if (mtp_pending_anchor) {
        const auto t_c0 = p.profile_phases ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};
        const bool ok = detail::decode_one(p.ctx, mtp_pending_tok);
        if (p.profile_phases)
            out.corrective_ms +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_c0)
                    .count();
        if (ok)
            detail::account_maintain(p, mtp_pending_tok, out, p.profile_phases);
        else
            log_output("[xllama] decode failed for pending correction at end of "
                       "generation\n");
        // Plan 009: replay the flushed correction for the same reason as
        // above (its row is live: single decode just ran). Fail-closed.
        if (ok)
            replay_one(mtp_pending_tok, llama_memory_seq_pos_max(llama_get_memory(p.ctx), 0));
        mtp_pending_anchor = false;
        mtp_pending_tok = LLAMA_TOKEN_NULL;
    }

    // Plan 009 turn-end carry commit: install the snapshot (last successfully
    // replayed prefix end) as the pending carry when it describes exactly the
    // last committed token and the mirror holds it. Draft rounds after the
    // last replay leave pending on future positions; without this, every
    // multi-turn continuation pays a full re-prefill even with a fully
    // matching prefix. Fail-closed (returns false, nothing touched): the
    // next turn then clears and rebuilds as today. No sampling, no commits,
    // no output change. The session's turn-end drafter-state line shows the
    // outcome (pending_valid/pos after this call).
    if (p.mtp && p.mtp->ready()) {
        const llama_pos committed_end = llama_memory_seq_pos_max(llama_get_memory(p.ctx), 0);
        if (committed_end >= 0)
            p.mtp->commit_carry(committed_end);
    }

    // Wall clock of the loop is always measured: t_decode_ms is the denominator the
    // phases are reconciled against, so it must not depend on profile_phases.
    out.t_decode_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_loop0)
            .count();
    if (p.profile_phases) {
        out.win_d3w_ms = d3d12_wall_ms() - win_d3w0;
        out.win_d3g_ms = d3d12_gpu_ms() - win_d3g0;
        out.win_cpu_ms = process_cpu_ms() - win_cpu0;
        out.win_calls = d3d12_graph_calls() - win_calls0;
        out.win_mm = d3d12_matmul_count() - win_mm0;
        char wlb[224];
        snprintf(wlb, sizeof(wlb),
                 "[xllama] DSPLIT wall=%.2f d3w=%.2f d3g=%.2f cpu=%.2f calls=%llu mm=%llu "
                 "classic_steps=%d\n",
                 out.t_decode_ms, out.win_d3w_ms, out.win_d3g_ms, out.win_cpu_ms,
                 static_cast<unsigned long long>(out.win_calls),
                 static_cast<unsigned long long>(out.win_mm), out.classic_steps);
        log_output(wlb);
    }
    // draft + sampling + top_prob + catch-up (verify AND classic) + verify +
    // corrective + lookup + classic decode are disjoint by construction: each
    // timer wraps exactly one call. t_classic_catchup_ms is a BREAKDOWN of the
    // catch-up total, not an extra term, so it is deliberately not added here —
    // adding both would double-count. The difference from t_decode_ms is the
    // uninstrumented residual and is reported, not hidden.
    out.t_decode_accounted_ms = out.mtp_draft_ms + out.mtp_sample_ms + out.mtp_top_prob_ms +
                                out.mtp_catchup_ms + out.verify_ms + out.corrective_ms +
                                out.t_lookup_ms + out.t_classic_ms + out.t_sample_target_ms +
                                out.t_emit_ms + out.t_maintain_ms;
    // Per-verify-round split (plan003): one line per verify round, profile
    // only (zero work when OFF). Units: ms, nesting gpu <= d3w <= wall <=
    // verify_ms — no double counting across the PHASE aggregates. accepted is
    // derived from the cumulative acc0 snapshots (exact: only round i's
    // accept walk lies between snapshot i and i+1).
    if (p.profile_phases && !out.vr_width.empty()) {
        for (size_t i = 0; i < out.vr_width.size(); ++i) {
            const int acc_before = out.vr_acc0[i];
            const int acc_after =
                (i + 1 < out.vr_acc0.size()) ? out.vr_acc0[i + 1] : out.n_accepted;
            char vlb[320];
            char feed_buf[96] = {};
            if (i < out.vr_feed.size()) {
                size_t off = 0;
                for (size_t k = 0; k < out.vr_feed[i].size() && off + 12 < sizeof(feed_buf); ++k)
                    off += static_cast<size_t>(snprintf(feed_buf + off, sizeof(feed_buf) - off,
                                                        "%s%d", k ? "," : "", out.vr_feed[i][k]));
            }
            snprintf(vlb, sizeof(vlb),
                     "[xllama] VROUND width=%d wall=%.2f d3w=%.2f d3g=%.2f cpu=%.2f "
                     "reuse=%d accepted=%d calls=%llu mm=%llu feed=%s gen=%d\n",
                     out.vr_width[i], out.vr_wall[i], out.vr_d3w[i], out.vr_d3g[i], out.vr_cpu[i],
                     out.vr_reuse[i], acc_after - acc_before,
                     static_cast<unsigned long long>(out.vr_calls[i]),
                     static_cast<unsigned long long>(out.vr_mm[i]), feed_buf, out.n_generated);
            log_output(vlb);
        }
    }
    return out;
}

} // namespace xllama
