// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Controlled single-vs-batch replay (plan 004 boundary diagnosis). See
// include/xllama/replay.h for the method contract.
//
// Prefix construction is DERIVED, not assumed: prefill the prompt, then
// teacher-force committed output ids one single at a time, asserting each
// argmax against the known sequential trajectory, until the KV ends at the
// observed boundary position. Whatever composition falls out (prompt length,
// committed count) is reported; any mismatch aborts loudly instead of
// comparing across different positions. Only then do the two branches run:
// singles vs one batch over the same final block.
#include "xllama/replay.h"

#include "decode_loop.h" // clamp_speculative_n_rs_seq (same call the session makes)
#include "llama.h"
#include "llama_gpu.h"
#include "xllama/chat_prompt.h"
#include "xllama/ggml_d3d12.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"
#include "xllama/tttarget.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace xllama {
namespace {

struct Top5 {
    int id[5] = {-1, -1, -1, -1, -1};
    double margin = 0.0;
};

Top5 top5_of(const float* logits, int32_t n) {
    Top5 t;
    for (int32_t i = 0; i < n; ++i) {
        const float v = logits[i];
        for (int k = 0; k < 5; ++k) {
            if (t.id[k] < 0 || v > logits[t.id[k]]) {
                for (int s = 4; s > k; --s)
                    t.id[s] = t.id[s - 1];
                t.id[k] = i;
                break;
            }
        }
    }
    if (t.id[0] >= 0 && t.id[1] >= 0)
        t.margin = static_cast<double>(logits[t.id[0]]) - logits[t.id[1]];
    return t;
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x))
            return false;
    return true;
}

int raw_argmax(const float* logits, int32_t n) {
    int best = 0;
    for (int32_t i = 1; i < n; ++i)
        if (logits[i] > logits[best])
            best = i;
    return best;
}

} // namespace

ReplayCmp compare_replay_rows(const std::vector<float>& a, const std::vector<float>& b) {
    ReplayCmp c;
    if (a.size() != b.size() || a.empty())
        return c;
    double max_a = 0.0;
    int arg_a = -1, arg_b = -1;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
            return ReplayCmp{}; // non-finite input: no comparison, never a match
        const double d = std::fabs(static_cast<double>(a[i]) - b[i]);
        if (d > c.maxabs)
            c.maxabs = d;
        const double fa = std::fabs(static_cast<double>(a[i]));
        if (fa > max_a)
            max_a = fa;
        // Raw argmax (largest value), NOT largest magnitude: logits go
        // negative, and max-abs picks the most negative entry instead.
        if (arg_a < 0 || a[i] > a[static_cast<size_t>(arg_a)])
            arg_a = static_cast<int>(i);
        if (arg_b < 0 || b[i] > b[static_cast<size_t>(arg_b)])
            arg_b = static_cast<int>(i);
    }
    c.maxrel = max_a > 0.0 ? c.maxabs / max_a : c.maxabs;
    c.argmax_match = (arg_a >= 0 && arg_a == arg_b);
    return c;
}

void run_replay_measure(const std::string& model_name, const std::string& prompt_text,
                        const std::vector<int32_t>& ctx_ids, const std::vector<int32_t>& feed_ids,
                        const std::vector<int32_t>& known_next, const std::vector<int32_t>& rem_ids,
                        const std::vector<int32_t>& rem_known, int gpu_layers, int n_ctx,
                        int n_threads, int rs_seq, bool nextn_on, std::vector<ReplayLogits>* out,
                        std::string* fail_why) {
    if (!out || !fail_why)
        return;
    auto fail = [&](const std::string& msg) {
        *fail_why = msg;
        log_output(("[xllama] replay FAIL: " + msg + "\n").c_str());
    };
    // ctx_ids are the committed outputs forming the prefix tail (23 ids,
    // indices 0..22); feed_ids is the observed verify feed (4 ids: 3 kept +
    // 1 rejected); known_next holds the sequential argmax after feed[0..2]
    // and after the corrective, in order. All real observed ids, no dummies.
    if (ctx_ids.size() != 23 || feed_ids.size() != 4 || known_next.size() != 4) {
        fail("replay files must hold: ctx 23 committed ids (0..22), feed 4 ids, known 4 ids");
        return;
    }
    // Captured remainder through the output33 decision (verified against the
    // feedcap TRACE schedule and the sequential baseline sidecars, 3/3
    // identical): B2 pair, single, B3 triple. rem_ids[0] must continue the
    // corrective (known_next[3]); rem_known holds the sequential argmax after
    // each remainder id, ending at the sequential output33.
    if (rem_ids.size() != 6 || rem_known.size() != 6) {
        fail("replay remainder files must hold: rem 6 ids, rem_known 6 ids");
        return;
    }
    if (rem_ids[0] != known_next[3]) {
        fail("replay remainder must continue the corrective's predicted next");
        return;
    }
    // The kept feed must continue the committed prefix (feed[0..2] are outputs
    // 23..25, already accepted when observed): verified live below by the
    // per-step asserts (each expects the next committed id), so a staged-file
    // mixup fails loudly instead of forging the boundary.

    const std::string model_dir = resolve_model_path(model_name);
    const std::string abs_model_path = first_gguf_in_dir(model_dir);
    if (abs_model_path.empty()) {
        fail("no .gguf in model dir: " + model_dir);
        return;
    }
    std::uint64_t gguf_bytes = 0;
    const std::string gguf_hash = fnv1a64_file(abs_model_path, &gguf_bytes);
    {
        char lb[256];
        snprintf(lb, sizeof(lb), "[xllama] replay model=%s gguf=%s bytes=%llu gpu_layers=%d\n",
                 abs_model_path.c_str(), gguf_hash.c_str(),
                 static_cast<unsigned long long>(gguf_bytes), gpu_layers);
        log_output(lb);
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.load_mtp = true; // same model object as the failing runs (head inert: never drafted)
    const int applied = apply_gguf_gpu_layers(gpu_layers, mparams);
    llama_model* model = llama_model_load_from_file(abs_model_path.c_str(), mparams);
    if (!model) {
        fail("model load failed: " + abs_model_path);
        return;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const ChatFormat fmt = chat_format_for(model_name);
    const std::string full_prompt =
        fmt.render_prompt("You are a helpful AI assistant.", {}, prompt_text);
    int32_t n_tok =
        llama_tokenize(vocab, full_prompt.c_str(), static_cast<int32_t>(full_prompt.size()),
                       nullptr, 0, true, false);
    if (n_tok >= 0) {
        fail("prompt tokenize size query failed");
        llama_model_free(model);
        return;
    }
    std::vector<llama_token> prompt_ids(static_cast<size_t>(-n_tok));
    n_tok = llama_tokenize(vocab, full_prompt.c_str(), static_cast<int32_t>(full_prompt.size()),
                           prompt_ids.data(), static_cast<int32_t>(prompt_ids.size()), true, false);
    if (n_tok <= 0) {
        fail("prompt tokenization failed");
        llama_model_free(model);
        return;
    }
    prompt_ids.resize(static_cast<size_t>(n_tok));
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    // Tripartite composition asserts (evidence-backed, fail loudly instead of
    // adjusting): 119 prompt + 23 committed = 142 prefix tokens, KV ends 141
    // = observed feed pos0 - 1. The feed's kept part must continue the
    // committed prefix: feed[i] for i<3 is output 23+i (verified below by the
    // per-step asserts, which compare against known, not against feed).
    if (prompt_ids.size() != 119 || ctx_ids.size() != 23) {
        char lb[200];
        snprintf(lb, sizeof(lb),
                 "replay prefix mismatch: prompt=%d (want 119) ctx=%zu (want 23 outputs 0..22)",
                 n_tok, ctx_ids.size());
        fail(lb);
        llama_model_free(model);
        return;
    }
    const llama_pos p_block = static_cast<llama_pos>(prompt_ids.size() + ctx_ids.size());
    if (p_block != 142) {
        fail("replay block position is not the observed 142");
        llama_model_free(model);
        return;
    }
    // Captured remainder starts right after the corrective (pos 145).
    const llama_pos p_rem = p_block + 4;
    if (p_rem != 146) {
        fail("replay remainder position is not the observed 146");
        llama_model_free(model);
        return;
    }
    {
        std::uint64_t ph = 1469598103934665603ull;
        for (llama_token t : prompt_ids) {
            ph ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(t));
            ph *= 1099511628211ull;
        }
        for (int32_t t : ctx_ids) {
            ph ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(t));
            ph *= 1099511628211ull;
        }
        char lb[320];
        snprintf(lb, sizeof(lb),
                 "[xllama] replay prefix n=%d fnv=%016llx feed=[%d,%d,%d,%d] at pos=%d "
                 "rem=[%d,%d,%d,%d,%d,%d] at rempos=%d vocab=%d\n",
                 static_cast<int>(prompt_ids.size() + ctx_ids.size()),
                 static_cast<unsigned long long>(ph), feed_ids[0], feed_ids[1], feed_ids[2],
                 feed_ids[3], static_cast<int>(p_block), rem_ids[0], rem_ids[1], rem_ids[2],
                 rem_ids[3], rem_ids[4], rem_ids[5], static_cast<int>(p_rem), n_vocab);
        log_output(lb);
    }

    const int n_thr = n_threads > 0 ? n_threads : detect_threads_llama();
    auto make_ctx = [&]() {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = static_cast<std::uint32_t>(n_ctx > 0 ? n_ctx : 2048);
        cp.n_threads = n_thr;
        cp.n_threads_batch = n_thr;
        clamp_speculative_n_rs_seq(cp, static_cast<std::uint32_t>(rs_seq < 0 ? 0 : rs_seq));
        apply_gguf_gpu_context(applied, cp);
        llama_context* ctx = llama_init_from_model(model, cp);
        if (ctx && nextn_on)
            llama_set_embeddings_nextn(ctx, true, /*masked=*/false);
        return ctx;
    };

    auto single_decode = [&](llama_context* ctx, llama_token tok, llama_pos pos,
                             std::vector<float>* logits_out) {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = tok;
        b.pos[0] = pos;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = 1;
        b.n_tokens = 1;
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0)
            return false;
        const float* lg = llama_get_logits_ith(ctx, 0);
        if (!lg)
            return false;
        logits_out->assign(lg, lg + n_vocab);
        return true;
    };

    auto kv_end = [&](llama_context* ctx) {
        return static_cast<llama_pos>(llama_memory_seq_pos_max(llama_get_memory(ctx), 0));
    };

    auto top_row = [&](const std::vector<float>& lg, ReplayLogits* row, const char* branch,
                       int rep) {
        row->branch = branch;
        row->rep = rep;
        row->full = lg;
        row->finite = all_finite(lg);
        Top5 t = top5_of(lg.data(), n_vocab);
        row->argmax = t.id[0];
        row->margin = t.margin;
        for (int k = 0; k < 5; ++k)
            row->top[k] = t.id[k];
    };

    auto log_row = [&](const char* what, const ReplayLogits& row) {
        char lb[320];
        snprintf(lb, sizeof(lb),
                 "[xllama] replay %s rep=%d argmax=%d margin=%.6f top=%d:%d:%d:%d:%d finite=%d\n",
                 what, row.rep, row.argmax, row.margin, row.top[0], row.top[1], row.top[2],
                 row.top[3], row.top[4], row.finite ? 1 : 0);
        log_output(lb);
    };

    // Per-boundary continuation record: want/got are evidence, logged, never
    // fatal. A differing diagnostic branch is the signal, not a staging error.
    auto log_cont = [&](int rep, int k, llama_pos pos, const ReplayLogits& row, int32_t want) {
        char lb[320];
        snprintf(
            lb, sizeof(lb),
            "[xllama] replay cont rep=%d k=%d pos=%d %s arg=%d want=%d margin=%.4f finite=%d\n",
            rep, k, static_cast<int>(pos), row.branch.c_str(), row.argmax, want, row.margin,
            row.finite ? 1 : 0);
        log_output(lb);
    };

    auto decode_block = [&](llama_context* ctx, const int32_t* toks, llama_pos pos0, int n) {
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            b.token[i] = toks[i];
            b.pos[i] = pos0 + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = 1;
        }
        b.n_tokens = n;
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        return rc == 0;
    };

    auto record_batch = [&](llama_context* ctx, int base_k, int n, const char* tag, int rep,
                            std::vector<ReplayLogits>* dest) {
        for (int i = 0; i < n; ++i) {
            const float* rlg = llama_get_logits_ith(ctx, i);
            if (!rlg)
                return false;
            const std::vector<float> brow(rlg, rlg + n_vocab);
            ReplayLogits crow;
            top_row(brow, &crow, tag, rep);
            crow.branch += std::to_string(base_k + i);
            log_cont(rep, base_k + i, p_rem + base_k + i, crow,
                     rem_known[static_cast<size_t>(base_k + i)]);
            dest->push_back(std::move(crow));
        }
        return true;
    };

    auto log_mem = [&](const char* what, int rep, llama_context* ctx) {
        char lb[160];
        snprintf(lb, sizeof(lb), "[xllama] replay mem rep=%d %s kv_end=%d\n", rep, what,
                 static_cast<int>(kv_end(ctx)));
        log_output(lb);
    };

    // Identical remainder used by C, B, D and E (B2 pair, single, B3 triple):
    // only the row-label tag and destination differ. Returns false with err
    // set on hard errors only; argmax differences are logged, never fatal.
    auto run_remainder = [&](llama_context* ctx, const char* tag, int rep,
                             std::vector<ReplayLogits>* dest, const char* who, std::string* err) {
        if (!decode_block(ctx, rem_ids.data(), p_rem, 2)) {
            *err = std::string("branch ") + who + " remainder B2 failed";
            return false;
        }
        if (!record_batch(ctx, 0, 2, tag, rep, dest)) {
            *err = std::string("branch ") + who + " remainder B2 logits missing";
            return false;
        }
        std::vector<float> rlg;
        if (!single_decode(ctx, rem_ids[2], p_rem + 2, &rlg)) {
            *err = std::string("branch ") + who + " remainder single failed";
            return false;
        }
        ReplayLogits crow;
        top_row(rlg, &crow, (std::string(tag) + "2").c_str(), rep);
        log_cont(rep, 2, p_rem + 2, crow, rem_known[2]);
        dest->push_back(std::move(crow));
        if (!decode_block(ctx, rem_ids.data() + 3, p_rem + 3, 3)) {
            *err = std::string("branch ") + who + " remainder B3 failed";
            return false;
        }
        if (!record_batch(ctx, 3, 3, tag, rep, dest)) {
            *err = std::string("branch ") + who + " remainder B3 logits missing";
            return false;
        }
        return true;
    };

    // Teacher-force the committed prefix identically in a fresh context,
    // asserting every argmax against the known sequential trajectory.
    // next_after_last is the expected argmax after the final forced id.
    // Returns false with fail_why set on any divergence: a diverged prefix
    // cannot test batching.
    auto build_prefix = [&](llama_context* ctx, int32_t next_after_last, std::string* why) {
        llama_batch pre = llama_batch_init(static_cast<int32_t>(prompt_ids.size()), 0, 1);
        for (size_t i = 0; i < prompt_ids.size(); ++i) {
            pre.token[i] = prompt_ids[i];
            pre.pos[i] = static_cast<llama_pos>(i);
            pre.n_seq_id[i] = 1;
            pre.seq_id[i][0] = 0;
            pre.logits[i] = 0;
        }
        pre.n_tokens = static_cast<int32_t>(prompt_ids.size());
        if (llama_decode(ctx, pre) != 0) {
            *why = "prefix prefill failed";
            llama_batch_free(pre);
            return false;
        }
        llama_batch_free(pre);
        for (size_t i = 0; i < ctx_ids.size(); ++i) {
            const llama_pos pos = static_cast<llama_pos>(prompt_ids.size() + i);
            std::vector<float> lg;
            if (!single_decode(ctx, ctx_ids[i], pos, &lg)) {
                *why = "prefix force-decode failed";
                return false;
            }
            const int want = (i + 1 < ctx_ids.size()) ? ctx_ids[i + 1] : next_after_last;
            if (raw_argmax(lg.data(), n_vocab) != want) {
                char lb[160];
                snprintf(lb, sizeof(lb),
                         "prefix diverged forcing id %zu (pos %d): argmax=%d want=%d", i,
                         static_cast<int>(pos), raw_argmax(lg.data(), n_vocab), want);
                *why = lb;
                return false;
            }
        }
        if (kv_end(ctx) != p_block - 1) {
            char lb[160];
            snprintf(lb, sizeof(lb), "prefix KV ends at %d, want %d", static_cast<int>(kv_end(ctx)),
                     static_cast<int>(p_block) - 1);
            *why = lb;
            return false;
        }
        return true;
    };

    // The kept feed must continue the committed prefix (feed[0..2] are outputs
    // 23..25): verified here once per run against the dumps, so a staged-file
    // mixup cannot forge the boundary. (The rejected feed[3] is exempt: it
    // never committed.)
    // NOTE: verified per-branch below via the known chain instead (tighter:
    // checks values, not just continuity).

    for (int rep = 0; rep < 2; ++rep) {
        // acc_rows holds the accepted 3-batch rows for this rep so branch B
        // can compare its 4-wide rows 0..2 against the 3-wide equivalents
        // BEFORE any trim: that isolates batch-width numerics from trim
        // residue. The final B-vs-C comparison (in the caller) then covers
        // width + trim combined.
        std::vector<std::vector<float>> acc_rows;
        // Continuation holders: the six existing rows keep their exact order
        // (seq, acc, tail per rep); continuation rows append after, so prior
        // evidence columns never shift.
        ReplayLogits seq_final, acc_final, tail_final;
        std::vector<ReplayLogits> seqc, accc, tailc;
        // Width/trim isolation holders (D/E run after B in the same rep).
        std::vector<float> b_tail_full;
        std::vector<ReplayLogits> d_batch, drem, e_batch, erem;
        ReplayLogits e_tail_single, e_corr;
        // Reference rows for the committed block boundaries, so every
        // D/E-vs-C comparison in the CSV uses same-token+position rows:
        // accb0..2 (accepted 3-batch rows) and btail (rejected batch row).
        std::vector<ReplayLogits> accb;
        ReplayLogits btail_row;
        // Branch A (sequential): prefix, then feed[0..2] + corrective as
        // singles, asserting known[0..3]. Records the final row.
        {
            llama_context* ctx = make_ctx();
            if (!ctx) {
                fail("branch A context creation failed");
                llama_model_free(model);
                return;
            }
            std::string why;
            bool ok = build_prefix(ctx, feed_ids[0], &why);
            const int32_t steps[4] = {feed_ids[0], feed_ids[1], feed_ids[2], known_next[2]};
            std::vector<float> lg;
            for (int s = 0; ok && s < 4; ++s) {
                if (!single_decode(ctx, steps[s], p_block + s, &lg)) {
                    fail("branch A single decode failed");
                    ok = false;
                    break;
                }
                if (raw_argmax(lg.data(), n_vocab) != known_next[s]) {
                    char lb[160];
                    snprintf(lb, sizeof(lb), "branch A diverged at step %d: argmax=%d want=%d", s,
                             raw_argmax(lg.data(), n_vocab), known_next[s]);
                    fail(lb);
                    ok = false;
                    break;
                }
            }
            if (!ok || lg.size() != static_cast<size_t>(n_vocab)) {
                if (ok)
                    fail("branch A logits missing");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            ReplayLogits row;
            top_row(lg, &row, "seq", rep);
            log_row("A-final", row);
            seq_final = std::move(row);
            if (kv_end(ctx) != p_block + 3) {
                fail("branch A KV is not at the corrective before remainder");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            // Canonical all-single remainder: the control separating suffix
            // batching effects. All 6 boundaries recorded; argmax differences
            // are logged, never fatal.
            for (int k = 0; k < 6; ++k) {
                std::vector<float> clg;
                if (!single_decode(ctx, rem_ids[static_cast<size_t>(k)], p_rem + k, &clg)) {
                    fail("branch A remainder single failed");
                    llama_free(ctx);
                    llama_model_free(model);
                    return;
                }
                ReplayLogits crow;
                top_row(clg, &crow, "seqc", rep);
                crow.branch += std::to_string(k);
                log_cont(rep, k, p_rem + k, crow, rem_known[static_cast<size_t>(k)]);
                seqc.push_back(std::move(crow));
            }
            llama_free(ctx);
        }
        // Branch C (accepted batch): prefix, then feed[0..2] as one batch
        // (asserting rows identically), then the corrective single. Records
        // the final row. No rejected tail, no trim.
        {
            llama_context* ctx = make_ctx();
            if (!ctx) {
                fail("branch C context creation failed");
                llama_model_free(model);
                return;
            }
            std::string why;
            if (!build_prefix(ctx, feed_ids[0], &why)) {
                fail(std::string("branch C prefix: ") + why);
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            llama_batch b = llama_batch_init(3, 0, 1);
            for (int i = 0; i < 3; ++i) {
                b.token[i] = feed_ids[i];
                b.pos[i] = p_block + i;
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = 1;
            }
            b.n_tokens = 3;
            bool ok = llama_decode(ctx, b) == 0;
            llama_batch_free(b);
            if (!ok) {
                fail("branch C batch decode failed");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            for (int i = 0; i < 3; ++i) {
                const float* lg = llama_get_logits_ith(ctx, i);
                if (!lg) {
                    fail("branch C logits missing");
                    llama_free(ctx);
                    llama_model_free(model);
                    return;
                }
                Top5 t = top5_of(lg, n_vocab);
                if (t.id[0] != known_next[i]) {
                    char lb[160];
                    snprintf(lb, sizeof(lb), "branch C row %d diverged: argmax=%d want=%d", i,
                             t.id[0], known_next[i]);
                    fail(lb);
                    llama_free(ctx);
                    llama_model_free(model);
                    return;
                }
                acc_rows.emplace_back(lg, lg + n_vocab);
                ReplayLogits bref;
                top_row(acc_rows.back(), &bref, "accb", rep);
                bref.branch += std::to_string(i);
                log_row("C-batch", bref);
                accb.push_back(std::move(bref));
            }
            std::vector<float> lg;
            if (!single_decode(ctx, known_next[2], p_block + 3, &lg)) {
                fail("branch C corrective single failed");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            top_row(lg, &acc_final, "acc", rep);
            log_row("C-final", acc_final);
            if (acc_final.argmax != known_next[3]) {
                char lb[160];
                snprintf(lb, sizeof(lb), "branch C final argmax=%d want=%d", acc_final.argmax,
                         known_next[3]);
                fail(lb);
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            if (kv_end(ctx) != p_block + 3) {
                fail("branch C KV is not at the corrective before remainder");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            // Live-schedule remainder on the clean path: B2 pair, single,
            // B3 triple. Same teacher-forced ids as every branch.
            if (!decode_block(ctx, rem_ids.data(), p_rem, 2)) {
                fail("branch C remainder B2 failed");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            if (!record_batch(ctx, 0, 2, "accc", rep, &accc)) {
                fail("branch C remainder B2 logits missing");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            {
                std::vector<float> rlg;
                if (!single_decode(ctx, rem_ids[2], p_rem + 2, &rlg)) {
                    fail("branch C remainder single failed");
                    llama_free(ctx);
                    llama_model_free(model);
                    return;
                }
                ReplayLogits crow;
                top_row(rlg, &crow, "accc2", rep);
                log_cont(rep, 2, p_rem + 2, crow, rem_known[2]);
                accc.push_back(std::move(crow));
            }
            if (!decode_block(ctx, rem_ids.data() + 3, p_rem + 3, 3)) {
                fail("branch C remainder B3 failed");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            if (!record_batch(ctx, 3, 3, "accc", rep, &accc)) {
                fail("branch C remainder B3 logits missing");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            llama_free(ctx);
        }
        // Branch B (tail + trim): prefix, then the full observed feed
        // including the rejected tail row, trim it with the same
        // trim_verify_tail call the loop uses, then the corrective single.
        // Records the final row. B-vs-C does NOT isolate trim on its own:
        // the 4-wide rows already differ pre-trim (rev73), so B-vs-C
        // conflates width4 + trim; D-vs-C and E-vs-C separate them.
        {
            llama_context* ctx = make_ctx();
            if (!ctx) {
                fail("branch B context creation failed");
                llama_model_free(model);
                return;
            }
            std::string why;
            if (!build_prefix(ctx, feed_ids[0], &why)) {
                fail(std::string("branch B prefix: ") + why);
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            llama_batch b = llama_batch_init(4, 0, 1);
            for (int i = 0; i < 4; ++i) {
                b.token[i] = feed_ids[i];
                b.pos[i] = p_block + i;
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = 1;
            }
            b.n_tokens = 4;
            bool ok = llama_decode(ctx, b) == 0;
            llama_batch_free(b);
            if (!ok) {
                fail("branch B batch decode failed");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            if (acc_rows.size() != 3) {
                fail("branch B missing accepted rows for width comparison");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            for (int i = 0; i < 3; ++i) {
                const float* lg = llama_get_logits_ith(ctx, i);
                if (!lg) {
                    fail("branch B logits missing");
                    llama_free(ctx);
                    llama_model_free(model);
                    return;
                }
                Top5 t = top5_of(lg, n_vocab);
                if (t.id[0] != known_next[i]) {
                    char lb[160];
                    snprintf(lb, sizeof(lb), "branch B row %d diverged: argmax=%d want=%d", i,
                             t.id[0], known_next[i]);
                    fail(lb);
                    llama_free(ctx);
                    llama_model_free(model);
                    return;
                }
                // Width isolation: same position, 4-wide vs 3-wide, pre-trim.
                // Zero here + nonzero final B-vs-C pins trim residue; nonzero
                // here implicates batch-width numerics (before any trim call).
                const std::vector<float> b_row(lg, lg + n_vocab);
                const ReplayCmp w = compare_replay_rows(b_row, acc_rows[static_cast<size_t>(i)]);
                char lb[320];
                snprintf(lb, sizeof(lb),
                         "[xllama] replay bw rep=%d row=%d b_arg=%d c_arg=%d maxabs=%.6g "
                         "maxrel=%.6g match=%d\n",
                         rep, i, t.id[0],
                         raw_argmax(acc_rows[static_cast<size_t>(i)].data(), n_vocab), w.maxabs,
                         w.maxrel, w.argmax_match ? 1 : 0);
                log_output(lb);
            }
            // Rejected tail row (same token+position as E's rejected single):
            // saved so E can compare single-vs-batch for this exact row.
            {
                const float* tlg = llama_get_logits_ith(ctx, 3);
                if (!tlg) {
                    fail("branch B tail row logits missing");
                    llama_free(ctx);
                    llama_model_free(model);
                    return;
                }
                b_tail_full.assign(tlg, tlg + n_vocab);
                top_row(b_tail_full, &btail_row, "btail", rep);
                log_row("B-tailrow", btail_row);
            }
            log_mem("B-pre-trim", rep, ctx);
            llama_memory_t mem = llama_get_memory(ctx);
            if (!detail::trim_verify_tail(mem, p_block - 1, 3, 4)) {
                fail("branch B tail trim refused");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            log_mem("B-post-trim", rep, ctx);
            if (kv_end(ctx) != p_block + 2) {
                char lb[160];
                snprintf(lb, sizeof(lb), "branch B KV ends at %d, want %d after trim",
                         static_cast<int>(kv_end(ctx)), static_cast<int>(p_block) + 2);
                fail(lb);
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            std::vector<float> lg;
            if (!single_decode(ctx, known_next[2], p_block + 3, &lg)) {
                fail("branch B corrective single failed");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            top_row(lg, &tail_final, "tail", rep);
            log_row("B-final", tail_final);
            if (tail_final.argmax != known_next[3]) {
                char lb[160];
                snprintf(lb, sizeof(lb), "branch B final argmax=%d want=%d", tail_final.argmax,
                         known_next[3]);
                fail(lb);
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            if (kv_end(ctx) != p_block + 3) {
                fail("branch B KV is not at the corrective before remainder");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            // Identical live-schedule remainder on the tail+trim path: B2
            // pair, single, B3 triple. Whether the width4-shifted history
            // reproduces the 1204 inversion is read off tailc5 vs seqc5.
            if (!decode_block(ctx, rem_ids.data(), p_rem, 2)) {
                fail("branch B remainder B2 failed");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            if (!record_batch(ctx, 0, 2, "tailc", rep, &tailc)) {
                fail("branch B remainder B2 logits missing");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            {
                std::vector<float> rlg;
                if (!single_decode(ctx, rem_ids[2], p_rem + 2, &rlg)) {
                    fail("branch B remainder single failed");
                    llama_free(ctx);
                    llama_model_free(model);
                    return;
                }
                ReplayLogits crow;
                top_row(rlg, &crow, "tailc2", rep);
                log_cont(rep, 2, p_rem + 2, crow, rem_known[2]);
                tailc.push_back(std::move(crow));
            }
            if (!decode_block(ctx, rem_ids.data() + 3, p_rem + 3, 3)) {
                fail("branch B remainder B3 failed");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            if (!record_batch(ctx, 3, 3, "tailc", rep, &tailc)) {
                fail("branch B remainder B3 logits missing");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            llama_free(ctx);
        }
        // Branch D (width4, no trim): the block's first three ids plus the
        // corrective as ONE accepted B4 at 142..145. No trim ever runs, so
        // D-vs-C isolates width4. The identical remainder follows.
        {
            llama_context* ctx = make_ctx();
            if (!ctx) {
                fail("branch D context creation failed");
                llama_model_free(model);
                return;
            }
            std::string why;
            if (!build_prefix(ctx, feed_ids[0], &why)) {
                fail(std::string("branch D prefix: ") + why);
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            const int32_t dblk[4] = {feed_ids[0], feed_ids[1], feed_ids[2], known_next[2]};
            if (!decode_block(ctx, dblk, p_block, 4)) {
                fail("branch D B4 failed");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            log_mem("D-post-B4", rep, ctx);
            if (kv_end(ctx) != p_block + 3) {
                fail("branch D KV is not 145 after B4");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            for (int i = 0; i < 4; ++i) {
                const float* dlg = llama_get_logits_ith(ctx, i);
                if (!dlg) {
                    fail("branch D B4 logits missing");
                    llama_free(ctx);
                    llama_model_free(model);
                    return;
                }
                const std::vector<float> drow(dlg, dlg + n_vocab);
                ReplayLogits rec;
                top_row(drow, &rec, "d", rep);
                rec.branch += std::to_string(i);
                // Same token+position as the accepted path: rows 0..2 vs the
                // 3-batch rows, row 3 (corrective-in-batch) vs the corrective
                // single. Logged only, never fatal.
                const std::vector<float>& ref =
                    (i < 3) ? acc_rows[static_cast<size_t>(i)] : acc_final.full;
                const ReplayCmp w = compare_replay_rows(drow, ref);
                const int32_t want = (i < 3) ? known_next[static_cast<size_t>(i)] : known_next[3];
                char lb[320];
                snprintf(lb, sizeof(lb),
                         "[xllama] replay dw rep=%d row=%d d_arg=%d want=%d maxabs=%.6g "
                         "maxrel=%.6g match=%d\n",
                         rep, i, rec.argmax, want, w.maxabs, w.maxrel, w.argmax_match ? 1 : 0);
                log_output(lb);
                d_batch.push_back(std::move(rec));
            }
            std::string rerr;
            if (!run_remainder(ctx, "drem", rep, &drem, "D", &rerr)) {
                fail(rerr);
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            llama_free(ctx);
        }
        // Branch E (trim without B4): accepted B3, the rejected id as a
        // single, the identical trim keeping 3, then the corrective single.
        // E-vs-C does NOT isolate the trim sequence (see NOTE in replay.h:
        // single-11 rewrites only snapshot slot 0, so post-trim restore reads
        // stale slot 1; E tests trim + stale restore). Same remainder after.
        {
            llama_context* ctx = make_ctx();
            if (!ctx) {
                fail("branch E context creation failed");
                llama_model_free(model);
                return;
            }
            std::string why;
            if (!build_prefix(ctx, feed_ids[0], &why)) {
                fail(std::string("branch E prefix: ") + why);
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            if (!decode_block(ctx, feed_ids.data(), p_block, 3)) {
                fail("branch E B3 failed");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            for (int i = 0; i < 3; ++i) {
                const float* elg = llama_get_logits_ith(ctx, i);
                if (!elg) {
                    fail("branch E B3 logits missing");
                    llama_free(ctx);
                    llama_model_free(model);
                    return;
                }
                const std::vector<float> erow(elg, elg + n_vocab);
                ReplayLogits rec;
                top_row(erow, &rec, "e", rep);
                rec.branch += std::to_string(i);
                // Determinism sanity: same computation as the accepted batch.
                const ReplayCmp w = compare_replay_rows(erow, acc_rows[static_cast<size_t>(i)]);
                char lb[320];
                snprintf(lb, sizeof(lb),
                         "[xllama] replay eb rep=%d row=%d e_arg=%d want=%d maxabs=%.6g "
                         "maxrel=%.6g match=%d\n",
                         rep, i, rec.argmax, known_next[static_cast<size_t>(i)], w.maxabs, w.maxrel,
                         w.argmax_match ? 1 : 0);
                log_output(lb);
                e_batch.push_back(std::move(rec));
            }
            std::vector<float> tlg;
            if (!single_decode(ctx, feed_ids[3], p_block + 3, &tlg)) {
                fail("branch E rejected single failed");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            top_row(tlg, &e_tail_single, "etail", rep);
            {
                // Single-vs-batch for the exact tail token+position, both
                // pre-trim: isolates batch-width numerics on that row.
                const ReplayCmp w = compare_replay_rows(tlg, b_tail_full);
                Top5 bt = top5_of(b_tail_full.data(), n_vocab);
                char lb[320];
                snprintf(lb, sizeof(lb),
                         "[xllama] replay et rep=%d e_arg=%d b_arg=%d maxabs=%.6g maxrel=%.6g "
                         "match=%d\n",
                         rep, e_tail_single.argmax, bt.id[0], w.maxabs, w.maxrel,
                         w.argmax_match ? 1 : 0);
                log_output(lb);
            }
            log_mem("E-pre-trim", rep, ctx);
            llama_memory_t emem = llama_get_memory(ctx);
            if (!detail::trim_verify_tail(emem, p_block - 1, 3, 4)) {
                fail("branch E tail trim refused");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            log_mem("E-post-trim", rep, ctx);
            if (kv_end(ctx) != p_block + 2) {
                fail("branch E KV is not 144 after trim");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            std::vector<float> elg2;
            if (!single_decode(ctx, known_next[2], p_block + 3, &elg2)) {
                fail("branch E corrective single failed");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            top_row(elg2, &e_corr, "ecorr", rep);
            {
                // Trim-sequence effect on the corrective row itself: same
                // token+position as the clean-path corrective single.
                const ReplayCmp w = compare_replay_rows(elg2, acc_final.full);
                char lb[320];
                snprintf(lb, sizeof(lb),
                         "[xllama] replay ec rep=%d e_arg=%d c_arg=%d maxabs=%.6g maxrel=%.6g "
                         "match=%d\n",
                         rep, e_corr.argmax, acc_final.argmax, w.maxabs, w.maxrel,
                         w.argmax_match ? 1 : 0);
                log_output(lb);
            }
            std::string rerr;
            if (!run_remainder(ctx, "erem", rep, &erem, "E", &rerr)) {
                fail(rerr);
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            llama_free(ctx);
        }
        // Six existing rows keep exact order; D/E/reference rows appended
        // after in fixed order: d0..d3, drem0..5, e0..e2, etail, ecorr,
        // erem0..5, accb0..2, btail (46 per rep, 92 total).
        out->push_back(std::move(seq_final));
        out->push_back(std::move(acc_final));
        out->push_back(std::move(tail_final));
        for (auto& r : seqc)
            out->push_back(std::move(r));
        for (auto& r : accc)
            out->push_back(std::move(r));
        for (auto& r : tailc)
            out->push_back(std::move(r));
        for (auto& r : d_batch)
            out->push_back(std::move(r));
        for (auto& r : drem)
            out->push_back(std::move(r));
        for (auto& r : e_batch)
            out->push_back(std::move(r));
        out->push_back(std::move(e_tail_single));
        out->push_back(std::move(e_corr));
        for (auto& r : erem)
            out->push_back(std::move(r));
        for (auto& r : accb)
            out->push_back(std::move(r));
        out->push_back(std::move(btail_row));
    }
    llama_model_free(model);
}

} // namespace xllama
