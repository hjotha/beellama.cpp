// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// T_target(B) implementation. See include/xllama/tttarget.h for the method.
//
// What this bench is, and is not:
//   * it drives a REAL llama_context built with the same params the MTP gate uses
//     (D3D12 layers, same n_ctx/n_batch/n_ubatch/n_rs_seq, load_mtp=true so the
//     target carries the same flags as the verify path);
//   * the state is rebuilt by RECREATING the context and re-decoding the same
//     prefix for every measurement — the only option that rules out recurrent-state
//     carry-over between widths;
//   * the timed interval is exactly one llama_decode of B candidate ids; prefix,
//     context creation, validation and the shape drain are all outside it;
//   * equivalence is checked on the LOGITS, not on the argmax alone: finite rows,
//     maxabs/maxrel against the sequential reference and the top-2 margin per row,
//     with a declared tolerance. Divergence is reported, never smoothed over.
#include "xllama/tttarget.h"

#include <cstring>

#include "decode_loop.h" // speculative_n_rs_seq / clamp_speculative_n_rs_seq
#include "llama.h"
#include "llama_gpu.h"
#include "xllama/ggml_d3d12.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <random>

namespace xllama {
namespace {

// Declared tolerance for the logits comparison. Same weights, same positions, same
// FP32 path; only the GEMM batch shape changes, so the reference and the measured
// row must agree far inside these bounds. Anything past them is reported as a
// divergence with its measured value, not clamped.
constexpr float kTolMaxAbs = 0.5f;  // absolute logit difference
constexpr float kTolMaxRel = 0.05f; // relative to max |logit| of the row
constexpr int kRefIds = 6;          // 5 candidates + the id after the batch

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Argmax of an ALREADY FETCHED row. Pure over the pointer, so a comparison reads the
// logits exactly once: read_row() decides whether the row exists, this decides what
// is in it.
bool argmax_ptr(const float* logits, int n_vocab, llama_token* out, float* margin) {
    if (!logits)
        return false;
    int best = 0, second = -1;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            second = best;
            best = i;
        } else if (second < 0 || logits[i] > logits[second]) {
            second = i;
        }
    }
    *out = static_cast<llama_token>(best);
    if (margin)
        *margin = second >= 0 ? logits[best] - logits[second] : 0.0f;
    return true;
}

// Row |row| of the LAST decode, or null with |err| set. |asked| is the list of batch
// indices that requested logits in that decode, recorded when the batch was built —
// so an index that was never requested is caught HERE instead of reaching
// llama_get_logits_ith, which resolves it through output_ids[i] == -1 and throws
// "batch.logits[i] != true" (aborting the Debug build). Nothing below depends on the
// fork remembering what was asked: the caller states it.
const float* read_row(const llama_context* ctx, const std::vector<int>& asked, int row,
                      int* n_vocab, std::string* err) {
    const auto it = std::find(asked.begin(), asked.end(), row);
    if (it == asked.end()) {
        if (err) {
            std::string rows;
            for (int a : asked) {
                rows += std::to_string(a);
                rows += '|';
            }
            *err =
                "row " + std::to_string(row) + " has no logits (this decode asked: " + rows + ")";
        }
        return nullptr;
    }
    const float* logits = llama_get_logits_ith(const_cast<llama_context*>(ctx), row);
    if (!logits) {
        if (err)
            *err = "get_logits_ith(" + std::to_string(row) + ") refused the row";
        return nullptr;
    }
    *n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    return logits;
}

// Builds the batch for |tokens[first .. first+count)| at continuing positions on
// sequence 0, requesting logits on the rows flagged in |logits_on| (null = none).
// Allocation and filling stop HERE, so a caller can keep them outside a timed
// interval and run llama_decode itself — the measurement times one call and nothing
// else. |asked| receives the batch indices that asked for logits.
llama_batch build_batch(llama_context* ctx, const std::vector<llama_token>& tokens, int first,
                        int count, const int* logits_on, std::vector<int>* asked) {
    llama_batch b = llama_batch_init(count, 0, 1);
    const llama_pos pos0 = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
    if (asked)
        asked->clear();
    for (int i = 0; i < count; ++i) {
        b.token[i] = tokens[static_cast<std::size_t>(first + i)];
        b.pos[i] = pos0 + static_cast<llama_pos>(i);
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        const bool want = logits_on != nullptr && logits_on[i] != 0;
        b.logits[i] = want ? 1 : 0;
        if (want && asked)
            asked->push_back(i);
    }
    b.n_tokens = count;
    return b;
}

// Decodes the prefix in n_batch chunks with logits on the LAST ROW of the last chunk
// only, reported in |asked|: get_logits_ith takes a batch index, and only the rows
// that requested logits resolve.
bool decode_prefix(llama_context* ctx, const std::vector<llama_token>& prefix,
                   std::vector<int>* asked) {
    const int n_batch = std::max(1, static_cast<int>(llama_n_batch(ctx)));
    if (asked)
        asked->clear();
    for (int off = 0; off < static_cast<int>(prefix.size()); off += n_batch) {
        const int chunk = std::min(n_batch, static_cast<int>(prefix.size()) - off);
        const bool last = off + chunk == static_cast<int>(prefix.size());
        std::vector<int> on(static_cast<std::size_t>(chunk), 0);
        if (last)
            on.back() = 1;
        llama_batch b = build_batch(ctx, prefix, off, chunk, on.data(), last ? asked : nullptr);
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0)
            return false;
    }
    return asked == nullptr || !asked->empty();
}

// One-shot decode of |tokens|: build, run, free. Returns false on failure and, when
// |asked| is given, reports the batch indices that requested logits — the set
// read_row() accepts afterwards. |n_requested| receives how many there are, i.e. the
// decode's n_outputs, so a caller can assert "the row I am about to read is the last
// one this decode produced" without an API the fork does not export.
bool decode_one(llama_context* ctx, const std::vector<llama_token>& tokens, const int* logits_on,
                std::vector<int>* asked, int* n_requested, std::string* err) {
    if (tokens.empty()) {
        if (err)
            *err = "empty batch";
        return false;
    }
    const int n_batch = std::max(1, static_cast<int>(llama_n_batch(ctx)));
    if (static_cast<int>(tokens.size()) > n_batch) {
        if (err)
            *err = "batch of " + std::to_string(tokens.size()) + " exceeds n_batch " +
                   std::to_string(n_batch);
        return false;
    }
    llama_batch b = build_batch(ctx, tokens, 0, static_cast<int>(tokens.size()), logits_on, asked);
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    if (rc != 0) {
        if (err)
            *err = "llama_decode returned " + std::to_string(rc);
        return false;
    }
    if (n_requested)
        *n_requested = asked ? static_cast<int>(asked->size()) : 0;
    if (asked && asked->empty()) {
        if (err)
            *err = "decode produced no logits row (logits_on was all zero)";
        return false;
    }
    return true;
}

std::string join_ids(const std::vector<llama_token>& ids) {
    std::string s;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i)
            s += '|';
        s += std::to_string(ids[i]);
    }
    return s;
}

} // namespace

// FNV-1a 64 over the whole file, streamed. A cheap identity fingerprint for the
// GGUF when a SHA-256 would mean a multi-GB transfer; labelled as such in the CSV.
std::string fnv1a64_file(const std::string& path, std::uint64_t* size_out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return "unreadable";
    std::uint64_t h = 1469598103934665603ull;
    std::uint64_t total = 0;
    std::vector<unsigned char> buf(1 << 20);
    std::size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) {
        total += n;
        for (std::size_t i = 0; i < n; ++i) {
            h ^= buf[i];
            h *= 1099511628211ull;
        }
    }
    std::fclose(f);
    if (size_out)
        *size_out = total;
    char out[32];
    std::snprintf(out, sizeof(out), "%016llx", static_cast<unsigned long long>(h));
    return out;
}

void measure_ttarget(const std::string& model_path, int n_gpu_layers, int n_ctx, int n_batch,
                     int n_ubatch, int n_rs_seq, int n_threads, int reps,
                     const std::vector<int>& widths, const std::vector<std::int32_t>& seed_ids,
                     const std::string& msix_sha, std::vector<TtargetRow>* out) {
    if (!out)
        return;
    if (!widths.empty()) {
        bool bad = false;
        for (int w : widths)
            bad = bad || w <= 0;
        if (bad) {
            TtargetRow r;
            r.error = "invalid width in schedule";
            out->push_back(std::move(r));
            return;
        }
    }

    // The caller passes the model NAME, which resolves to a directory; llama
    // wants the file. Same resolution the inference path uses, so both load the
    // same GGUF when a directory holds more than one.
    const std::string model_dir = resolve_model_path(model_path);
    const std::string abs_model_path = first_gguf_in_dir(model_dir);
    if (abs_model_path.empty()) {
        TtargetRow r;
        r.error = "no .gguf in model dir: " + model_dir;
        out->push_back(std::move(r));
        return;
    }

    llama_model_params mparams = llama_model_default_params();
    // Same flags the MTP verify path loads the target with: the head lives in the
    // same GGUF and loading it makes graph/tensor placement identical, so the cost
    // measured here is the cost the gate pays. A load without it is a different
    // target and would be labelled as such.
    mparams.load_mtp = true;
    const int applied = apply_gguf_gpu_layers(n_gpu_layers, mparams);
    llama_model* model = llama_model_load_from_file(abs_model_path.c_str(), mparams);
    if (!model) {
        TtargetRow r;
        r.error = "model load failed: " + abs_model_path;
        out->push_back(std::move(r));
        return;
    }

    std::vector<llama_token> prefix(seed_ids.begin(), seed_ids.end());
    if (prefix.empty()) {
        const char* text = "The capital of France is";
        const llama_vocab* vocab = llama_model_get_vocab(model);
        int n = llama_tokenize(vocab, text, static_cast<int>(std::strlen(text)), nullptr, 0, true,
                               true);
        if (n < 0)
            n = -n;
        prefix.resize(static_cast<std::size_t>(n));
        n = llama_tokenize(vocab, text, static_cast<int>(std::strlen(text)), prefix.data(), n, true,
                           true);
        if (n <= 0) {
            TtargetRow r;
            r.error = "prefix tokenization failed";
            out->push_back(std::move(r));
            llama_model_free(model);
            return;
        }
        prefix.resize(static_cast<std::size_t>(n));
    }

    auto make_ctx = [&]() {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = static_cast<std::uint32_t>(n_ctx > 0 ? n_ctx : 2048);
        cp.n_threads = n_threads;
        cp.n_threads_batch = n_threads;
        if (n_batch > 0)
            cp.n_batch = static_cast<std::uint32_t>(n_batch);
        if (n_ubatch > 0)
            cp.n_ubatch = static_cast<std::uint32_t>(n_ubatch);
        clamp_speculative_n_rs_seq(cp, static_cast<std::uint32_t>(n_rs_seq));
        apply_gguf_gpu_context(applied, cp);
        return llama_init_from_model(model, cp);
    };

    std::uint64_t gguf_size = 0;
    const std::string gguf_hash = fnv1a64_file(abs_model_path, &gguf_size);

    // Stage markers: the host runner polls the console log for these instead of
    // guessing from a missing .done file.
    log_output("[xllama] ttarget stage=reference start\n");
    // ---- Sequential reference, on its own context --------------------------
    // Produces kRefIds ids and keeps the logits that predict each of them, so every
    // measured row can be compared numerically and not just by argmax.
    std::vector<llama_token> ref_ids;
    std::vector<std::vector<float>> ref_logits;
    std::string cfg_line;
    {
        llama_context* rc = make_ctx();
        if (!rc) {
            TtargetRow r;
            r.error = "reference context creation failed";
            out->push_back(std::move(r));
            llama_model_free(model);
            return;
        }
        char cb[256];
        std::snprintf(cb, sizeof(cb),
                      "[xllama] ttarget effective config n_ctx=%u n_batch=%u n_ubatch=%u "
                      "n_rs_seq=%d backend=%s",
                      llama_n_ctx(rc), llama_n_batch(rc), llama_n_ubatch(rc), n_rs_seq,
                      applied > 0 ? "d3d12" : "cpu");
        cfg_line = cb;
        log_output(cfg_line + "\n");
        // The asked-list is reset by every decode below: it describes the decode that
        // produced the row, never a previous one.
        std::vector<int> asked;
        std::string why;
        bool ok = decode_prefix(rc, prefix, &asked);
        // The row index belongs to the decode that PRODUCED it: the prefix batch put
        // its single logits on the last row, every one-token decode after that puts
        // them on row 0. read_row() checks it against the asked-list of that same
        // decode, so a stale index is an error message and not a fork exception.
        int row_idx = asked.empty() ? -1 : asked.back();
        for (int step = 0; ok && step < kRefIds; ++step) {
            int nv = 0;
            const float* row = read_row(rc, asked, row_idx, &nv, &why);
            llama_token next = LLAMA_TOKEN_NULL;
            if (!row || !argmax_ptr(row, nv, &next, nullptr)) {
                if (why.empty())
                    why = "argmax failed on row " + std::to_string(row_idx);
                ok = false;
                break;
            }
            // Copied BEFORE the next decode: the pointer aliases the context's logits
            // buffer, which the following llama_decode overwrites in place.
            ref_logits.emplace_back(row, row + nv);
            ref_ids.push_back(next);
            // The batch the NEXT step reads is the one built here, so its logits must
            // be requested explicitly: the helper maps a null |logits_on| to
            // b.logits[]=false, and read_row() would then (correctly) refuse row 0.
            std::vector<llama_token> one = {next};
            const int one_on[1] = {1};
            int n_out = 0;
            why.clear();
            ok = decode_one(rc, one, one_on, &asked, &n_out, &why);
            if (!ok)
                break;
            // n_outputs of a one-token batch with logits on it is exactly 1, at batch
            // index 0 — asserted rather than assumed, because this is the index the
            // following iteration dereferences.
            if (n_out != 1 || asked.size() != 1 || asked.back() != 0) {
                why = "unit decode requested n_outputs=" + std::to_string(n_out) +
                      " rows=" + std::to_string(asked.size());
                ok = false;
                break;
            }
            row_idx = asked.back();
        }
        llama_free(rc);
        if (!ok || ref_ids.size() < static_cast<std::size_t>(kRefIds)) {
            TtargetRow r;
            r.error = "sequential reference failed at step " + std::to_string(ref_ids.size()) +
                      (why.empty() ? "" : ": " + why);
            out->push_back(std::move(r));
            llama_model_free(model);
            return;
        }
    }
    log_output("[xllama] ttarget stage=reference done ids=" + std::to_string(ref_ids.size()) +
               "\n");

    // ---- Measurement schedule: randomized B order, warm-up + reps ----------
    const unsigned order_seed = 0x5eedu;
    std::mt19937 rng(order_seed);
    std::vector<int> bs = {1, 2, 3, 5};
    if (!widths.empty())
        bs = widths;
    // The reference above produced kRefIds ids; a wider B would compare past it.
    for (int w : bs) {
        if (w >= kRefIds) {
            TtargetRow r;
            r.error = "width " + std::to_string(w) + " exceeds the " + std::to_string(kRefIds - 1) +
                      " reference ids produced";
            out->push_back(std::move(r));
            llama_model_free(model);
            return;
        }
    }
    int sched = 0; // TRUE sequential position in the shuffled schedule
    for (int pass = -1; pass < reps; ++pass) {
        std::vector<int> order(bs.size());
        for (std::size_t i = 0; i < order.size(); ++i)
            order[i] = static_cast<int>(i);
        std::shuffle(order.begin(), order.end(), rng);
        for (int oi : order) {
            const int B = bs[static_cast<std::size_t>(oi)];
            ++sched;
            log_output("[xllama] ttarget stage=measure b=" + std::to_string(B) + " rep=" +
                       std::to_string(pass) + " order=" + std::to_string(sched) + " start\n");
            // Fresh context per measurement: no recurrent-state carry-over between
            // widths, at the cost of a rebuild inside the untimed part.
            llama_context* ctx = make_ctx();
            TtargetRow r;
            r.b = B;
            r.rep = pass;
            r.order = sched;
            r.order_seed = static_cast<int>(order_seed);
            r.prefix_len = static_cast<int>(prefix.size());
            r.prefix_ids = join_ids(prefix);
            r.config = cfg_line;
            r.msix_sha = msix_sha;
            r.gguf_fnv1a64 = gguf_hash;
            r.gguf_bytes = gguf_size;
            r.load_mtp = 1;
            if (!ctx) {
                r.error = "context creation failed";
                out->push_back(std::move(r));
                continue;
            }
            std::vector<int> p_asked;
            if (!decode_prefix(ctx, prefix, &p_asked)) {
                r.error = "prefix rebuild failed";
                llama_free(ctx);
                out->push_back(std::move(r));
                continue;
            }
            std::string row_err;
            // State check on the LOGITS of the rebuilt prefix vs the reference's
            // first row: finite, maxabs/maxrel and argmax must all agree. A
            // mismatch voids this measurement.
            {
                int nv = 0;
                const int rebuild_row = p_asked.empty() ? -1 : p_asked.back();
                const float* row = read_row(ctx, p_asked, rebuild_row, &nv, &row_err);
                bool finite = row != nullptr;
                float maxabs = 0.0f, maxref = 1.0f;
                for (int i = 0; finite && i < nv; ++i) {
                    const float d = std::fabs(row[i] - ref_logits[0][static_cast<std::size_t>(i)]);
                    if (!std::isfinite(row[i]) ||
                        !std::isfinite(ref_logits[0][static_cast<std::size_t>(i)]))
                        finite = false;
                    maxabs = std::max(maxabs, d);
                    maxref =
                        std::max(maxref, std::fabs(ref_logits[0][static_cast<std::size_t>(i)]));
                }
                llama_token got = LLAMA_TOKEN_NULL, want = ref_ids[0];
                const bool have_arg = finite && argmax_ptr(row, nv, &got, nullptr);
                if (!have_arg || got != want || maxabs > kTolMaxAbs ||
                    maxabs / maxref > kTolMaxRel) {
                    char eb[256];
                    std::snprintf(eb, sizeof(eb),
                                  "prefix state mismatch: maxabs=%.4f maxrel=%.4f finite=%d "
                                  "argmax=%d want=%d%s%s",
                                  maxabs, maxabs / maxref, finite ? 1 : 0, static_cast<int>(got),
                                  static_cast<int>(want), row_err.empty() ? "" : " | ",
                                  row_err.c_str());
                    r.error = eb;
                    llama_free(ctx);
                    out->push_back(std::move(r));
                    continue;
                }
            }

            // Preparation — candidate vector, logits flags, llama_batch_init and its
            // fill — sits BEFORE t0: the interval times one llama_decode and nothing
            // else. The batch object survives t1 and is freed after it, again outside.
            std::vector<llama_token> cand(ref_ids.begin(), ref_ids.begin() + B);
            std::vector<int> logon(static_cast<std::size_t>(B), 1);
            std::vector<int> timed_asked;
            llama_batch timed = build_batch(ctx, cand, 0, B, logon.data(), &timed_asked);

            const std::uint64_t calls0 = d3d12_graph_calls();
            const std::uint64_t mm0 = d3d12_matmul_count();
            const double gpu0 = d3d12_gpu_ms();
            double wall = 0.0;
            int got_n = -1;
            std::string dec_err;
            {
                // ScopeTag is constructed before t0 as well: only llama_decode sits
                // inside the interval.
                detail::ScopeTag scope("target", "tttarget");
                const double t0 = now_ms();
                const int drc = llama_decode(ctx, timed);
                wall = now_ms() - t0;
                if (drc == 0)
                    got_n = B;
                else
                    dec_err = "llama_decode returned " + std::to_string(drc);
            }
            const double gpu = d3d12_gpu_ms() - gpu0;
            const std::uint64_t calls = d3d12_graph_calls() - calls0;
            const std::uint64_t mm = d3d12_matmul_count() - mm0;
            llama_batch_free(timed);       // outside the interval
            d3d12_shape_drain("tttarget"); // outside the interval, own label

            r.wall_ms = wall;
            r.gpu_ms = gpu;
            r.calls = calls;
            r.matmuls = mm;
            r.candidate_ids = join_ids(cand);
            r.n_decoded = got_n;
            if (got_n != B) {
                r.error = "timed decode failed" + (dec_err.empty() ? "" : ": " + dec_err);
                llama_free(ctx);
                out->push_back(std::move(r));
                continue;
            }
            // n_outputs of the timed batch, stated rather than assumed: every row the
            // comparison reads must be one this decode produced.
            if (static_cast<int>(timed_asked.size()) != B) {
                r.error = "timed decode asked for " + std::to_string(timed_asked.size()) +
                          " logits rows, expected " + std::to_string(B);
                llama_free(ctx);
                out->push_back(std::move(r));
                continue;
            }
            // Numerical comparison per row: row i predicts ref_ids[i+1].
            r.argmax_match = 1;
            for (int i = 0; i < B; ++i) {
                int nv = 0;
                std::string rerr;
                const int row_idx = timed_asked[static_cast<std::size_t>(i)];
                const float* row = read_row(ctx, timed_asked, row_idx, &nv, &rerr);
                if (!row) {
                    r.argmax_match = 0;
                    r.first_mismatch = i;
                    r.error = "no logits row " + std::to_string(i) + ": " + rerr;
                    break;
                }
                const std::vector<float>& ref = ref_logits[static_cast<std::size_t>(i + 1)];
                float maxabs = 0.0f, maxref = 1.0f;
                bool finite = true;
                for (int k = 0; k < nv; ++k) {
                    const float d = std::fabs(row[k] - ref[static_cast<std::size_t>(k)]);
                    if (!std::isfinite(row[k]) || !std::isfinite(ref[static_cast<std::size_t>(k)]))
                        finite = false;
                    maxabs = std::max(maxabs, d);
                    maxref = std::max(maxref, std::fabs(ref[static_cast<std::size_t>(k)]));
                }
                llama_token got = LLAMA_TOKEN_NULL, want = ref_ids[static_cast<std::size_t>(i + 1)];
                float margin = 0.0f;
                const bool have_arg = finite && argmax_ptr(row, nv, &got, &margin);
                r.logit_maxabs = std::max(r.logit_maxabs, maxabs);
                r.logit_maxrel = std::max(r.logit_maxrel, maxabs / maxref);
                r.margin_min = r.margin_min < 0.0f ? margin : std::min(r.margin_min, margin);
                if (!finite || !have_arg || got != want || maxabs > kTolMaxAbs ||
                    maxabs / maxref > kTolMaxRel) {
                    r.argmax_match = 0;
                    r.first_mismatch = i;
                    char eb[224];
                    std::snprintf(eb, sizeof(eb),
                                  "row %d: maxabs=%.4f maxrel=%.4f finite=%d argmax=%d want=%d "
                                  "margin=%.4f",
                                  i, maxabs, maxabs / maxref, finite ? 1 : 0, static_cast<int>(got),
                                  static_cast<int>(want), margin);
                    r.error = eb;
                    break;
                }
            }
            llama_free(ctx);
            out->push_back(std::move(r));
            log_output("[xllama] ttarget stage=measure b=" + std::to_string(B) +
                       " rep=" + std::to_string(pass) + " order=" + std::to_string(sched) +
                       (r.error.empty() ? " done" : " error=" + r.error) + "\n");
        }
    }

    llama_model_free(model);
}

const char* ttarget_csv_header() {
    return "b,rep,order,order_seed,prefix_len,n_decoded,argmax_match,first_mismatch,"
           "logit_maxabs,logit_maxrel,margin_min,wall_ms,gpu_ms,calls,matmuls,load_mtp,"
           "prefix_ids,candidate_ids,config,msix_sha256,gguf_fnv1a64,gguf_bytes,host,date,error\n";
}

std::string format_ttarget_row(const TtargetRow& r, const char* host_label) {
    char date_buf[32];
    std::time_t now = std::time(nullptr);
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    char buf[512];
    std::snprintf(
        buf, sizeof(buf),
        "%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.3f,%.3f,%llu,%llu,"
        "%d,%s,%s,%s,%s,%s,%llu,%s,%s,%s\n",
        r.b, r.rep, r.order, r.order_seed, r.prefix_len, r.n_decoded, r.argmax_match,
        r.first_mismatch, r.logit_maxabs, r.logit_maxrel, r.margin_min, r.wall_ms, r.gpu_ms,
        static_cast<unsigned long long>(r.calls), static_cast<unsigned long long>(r.matmuls),
        r.load_mtp, r.prefix_ids.c_str(), r.candidate_ids.c_str(), r.config.c_str(),
        r.msix_sha.c_str(), r.gguf_fnv1a64.c_str(), static_cast<unsigned long long>(r.gguf_bytes),
        host_label ? host_label : "unknown", date_buf, r.error.empty() ? "-" : r.error.c_str());
    return buf;
}

} // namespace xllama
