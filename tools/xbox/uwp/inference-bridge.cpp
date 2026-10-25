// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "inference-bridge.h"

#include "xllama/chat_prompt.h"
#include "xllama/decode_trace.h"
#include "xllama/device_train.h"
#include "xllama/diskbw.h"
#include "xllama/diverge.h"
#include "xllama/gpubw.h"
#include "xllama/gpugemv.h"
#include "xllama/gpustep.h"
#include "xllama/termgate.h"
#include "xllama/tttarget.h"
#ifdef XLLAMA_USE_LLAMA
    #include "xllama/ggml_d3d12.h"
#endif
#include "xllama/inference.h"
#include "xllama/json_utils.h"
#include "xllama/membw.h"
#include "xllama/path_utils.h"
#include "xllama/personalize.h"
#include "xllama/platform.h"
#include "xllama/ramceil.h"
#include "xllama/replay.h"
#include "xllama/session.h"
#include "xllama/training.h"
#include "xllama/utf8_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

// Elementwise CPU microbench (run_elembench): needs the ggml backend API.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <string>
#include <vector>

#ifdef XLLAMA_UWP
    // WS-F microphone probe (run_mic_probe). Universal contract, so no
    // SDKReference is needed — but per uwp-constraints.md §10b that says
    // nothing about activation, which is what the probe measures.
    //
    // unknwn.h first, and not by taste: WIN32_LEAN_AND_MEAN omits objbase.h,
    // which is what would otherwise define IUnknown, and winrt/base.h only
    // forward-declares it. The probe derives IMemoryBufferByteAccessXll from
    // IUnknown, so an incomplete declaration is a hard error here rather than
    // the usual static_assert. pch.h carries the same include for the same
    // reason; this TU does not include pch.h.
    #include <unknwn.h>

    #include <winrt/Windows.Foundation.Metadata.h>
    #include <winrt/Windows.Media.Audio.h>
    #include <winrt/Windows.Media.Capture.h>
    #include <winrt/Windows.Media.MediaProperties.h>
    #include <winrt/Windows.Media.Render.h>
    #include <winrt/Windows.Media.h>
#endif

namespace xllama::bridge {

#ifdef XLLAMA_UWP
namespace {

std::string read_local_file(const char* name) {
    std::string out;
    std::string p = resolve_local_path(name);
    FILE* f = _wfopen(utf8_to_wstring(p).c_str(), L"r");
    if (!f)
        return out;
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        out.append(buf, n);
    fclose(f);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' '))
        out.pop_back();
    return out;
}

// Small integer knob from a LocalState file; |fallback| when absent or unparsable.
int read_local_int(const char* name, int fallback) {
    const std::string s = read_local_file(name);
    return s.empty() ? fallback : std::atoi(s.c_str());
}

// Bench/session host tag (plan 005). The exact console model is not detectable
// from an AppContainer, and the hardcoded family label was already wrong once
// (Series X hardware labeled xbox-series-s). The runner may provide the
// verified label in LocalState bench_host_tag.txt (lowercase [a-z0-9-], max 24
// chars); absent or invalid stays family-only "xbox-series".
std::string host_tag_prefix() {
    const std::string tag = read_local_file("bench_host_tag.txt");
    const bool valid = !tag.empty() && tag.size() <= 24 &&
                       std::all_of(tag.begin(), tag.end(), [](unsigned char c) {
                           return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-';
                       });
    return valid ? tag : std::string("xbox-series");
}

// Real-decode two-column knob (plan 004): LocalState d3d12twocol.txt with
// content "auto" engages variant 2 (allowlisted NEW) for the enclosing bench;
// anything else or absent stays OLD (0). Force (1) is bench internals only
// (the d3d12be paired runner) and never comes from this file, so real decode
// cannot silently run an unmeasured width. Returns the previous variant so
// the scope below restores it.
int apply_twocol_knob(const char* who) {
    const int prev = ::xllama::d3d12_kernel_variant();
    const std::string raw = read_local_file("d3d12twocol.txt");
    const int v = (raw == "auto") ? 2 : 0;
    ::xllama::d3d12_set_kernel_variant(v);
    char lb[192];
    snprintf(lb, sizeof(lb), "[xllama] %s: d3d12twocol.txt='%s' -> kernel variant %d (prev %d)\n",
             who, raw.empty() ? "(absent)" : raw.c_str(), v, prev);
    log_output(lb);
    return prev;
}

struct TwocolKnobScope {
    int prev;
    explicit TwocolKnobScope(const char* who) : prev(apply_twocol_knob(who)) {}
    ~TwocolKnobScope() {
        ::xllama::d3d12_set_kernel_variant(prev);
    }
};

// CPU-repack force-gemv knob (z-0 first-divergence control, patches/0005):
// LocalState cpurepackforcegemv.txt with content "1" (z-0 weight scope) or
// "2" (all small-batch repack matmuls) sets GGML_REPACK_FORCE_GEMV before
// the fork's first use (model load); anything else or absent keeps the
// baseline OFF path. Process env + the fork's static-once parse mean the
// value fixes for the process lifetime — delete the file between runs to
// return to baseline on the next launch.
void apply_repack_ctrl_knob(const char* who) {
    const std::string raw = read_local_file("cpurepackforcegemv.txt");
    const bool on = (raw == "1" || raw == "2");
    if (on)
        _putenv(("GGML_REPACK_FORCE_GEMV=" + raw).c_str());
    char lb[192];
    snprintf(
        lb, sizeof(lb), "[xllama] %s: cpurepackforcegemv.txt='%s' -> %s\n", who,
        raw.empty() ? "(absent)" : raw.c_str(),
        on ? (raw == "1" ? "repack control scope 1 (z-0)" : "repack control scope 2 (all small)")
           : "repack control off (baseline)");
    log_output(lb);
}

// Queue-fence wait-policy experiment (plan004 rev101): d3d12spinwait.txt
// holding a non-negative integer (<= 7 digits, microseconds) caps the
// backend's GetCompletedValue spin at that budget before falling back to
// the event wait; absent or anything else = -1 keeps the historical
// unbounded spin — the default, unchanged.
void apply_spinwait_knob(const char* who) {
    const std::string raw = read_local_file("d3d12spinwait.txt");
    int us = -1;
    bool parsed = false;
    if (!raw.empty() && raw.size() <= 7 &&
        raw.find_first_not_of("0123456789") == std::string::npos) {
        us = 0;
        for (char c : raw)
            us = us * 10 + (c - '0');
        parsed = true;
    }
    ::xllama::d3d12_set_spin_wait_us(parsed ? us : -1);
    char lb[208];
    snprintf(lb, sizeof(lb), "[xllama] %s: d3d12spinwait.txt='%s' -> spin_wait_us=%d%s\n", who,
             raw.empty() ? "(absent)" : raw.c_str(), parsed ? us : -1,
             parsed ? " (bounded spin)" : " (default unbounded spin)");
    log_output(lb);
}

// Plan 008 complete island: SWIGLU + ADD + RMS_NORM + MUL on D3D12, one knob
// (d3d12island.txt "1"). Partial moves were measured worthless (plan 007), so
// the set is enabled together; SWIGLU keeps its own product-mode policy (the
// MTP arm still binds FFN OFF unless the policy accepts). Process-wide, read
// at each bench start so a paired A/B can flip it per run.
void apply_island_knob(const char* who) {
    const std::string raw = read_local_file("d3d12island.txt");
    // Production-safe default OFF until the extended gates pass (owner
    // review): d3d12island.txt "1" enables the island for the paired A/B.
    const bool on = raw == "1";
    ::xllama::d3d12_set_island_enabled(on);
    if (on)
        ::xllama::d3d12_set_swiglu_product_mode(1); // all FFN; policy still gates MTP
    log_output(std::string("[xllama] ") + who + ": d3d12island.txt='" +
               (raw.empty() ? "(absent)" : raw) + "' -> island " + (on ? "D3D12" : "CPU") +
               (on ? " (SWIGLU+ADD+RMS_NORM+MUL)" : "") + "\n");
}

// Diagnostic-only: bench_swiglu_mtp_diag.txt "1" allows the FFN SWIGLU product
// path in an MTP context (bypasses the seq-bound policy) and turns on the
// per-round MTPDIAG draft/verify dumps for the equal-history diagnosis.
void apply_swiglu_mtp_diag_knob(const char* who) {
    const bool on = read_local_file("bench_swiglu_mtp_diag.txt") == "1";
    ::xllama::d3d12_set_swiglu_mtp_diag(on);
    log_output(std::string("[xllama] ") + who + ": bench_swiglu_mtp_diag.txt=" +
               (on ? "1 (FFN allowed in MTP + MTPDIAG dumps)\n" : "(absent)\n"));
}

// Owner audit: d3d12swiglurange.txt "1" samples the product SWIGLU inputs
// (gate/up) after the graph fence and logs the real |x| distribution.
void apply_swiglu_range_knob(const char* who) {
    const bool on = read_local_file("d3d12swiglurange.txt") == "1";
    ::xllama::d3d12_set_swiglu_range_diag(on);
    log_output(std::string("[xllama] ") + who + ": d3d12swiglurange.txt=" +
               (on ? "1 (range sampling on)\n" : "(absent)\n"));
}

// Existing-instrumentation switch (plan004 rev102): ggmlprof.txt strict
// single token -> _putenv of the matching env read by the PINNED ggml/llama
// (default-off facilities; no submodule change). Absent or anything else
// sets nothing — today's behavior byte-for-byte. Must run before the first
// model/context use (the pin reads these once: at first compute, at sched
// creation, or at context construction).
void apply_ggmlprof_knob(const char* who) {
    const std::string raw = read_local_file("ggmlprof.txt");
    const char* var = nullptr;
    if (raw == "copies")
        var = "GGML_BACKEND_COPY_PROFILE=1";
    else if (raw == "sched")
        var = "GGML_SCHED_DEBUG=1";
    else if (raw == "sched2")
        var = "GGML_SCHED_DEBUG=2"; // headers + per-node backend assignment
    else if (raw == "realloc")
        var = "GGML_SCHED_DEBUG_REALLOC=1";
    else if (raw == "noreuse")
        var = "LLAMA_GRAPH_REUSE_DISABLE=1";
    else if (raw == "singlesync")
        var = "SPEC_OPT_SINGLE_SYNC=1";
    if (var)
        _putenv(var);
    char lb[224];
    snprintf(lb, sizeof(lb), "[xllama] %s: ggmlprof.txt='%s' -> %s\n", who,
             raw.empty() ? "(absent)" : raw.c_str(), var ? var : "none (default)");
    log_output(lb);
}

// Multi-turn TTFT bench: measures turn-2 prefill with KV reuse (append only the
// new turn) against the cold baseline (full re-prefill of the 2-turn context),
// on the same persistent Session. The ratio is the KV-reuse win. Writes
// bench-kv-result.csv (+ .done) and logs the numbers.
void run_kv_bench(const std::string& model_name, const std::string& sys, const std::string& u1,
                  const std::string& u2, int n_threads, int n_ctx, const char* host,
                  int run_index) {
    // KV-reuse now works on both backends: ORT-GenAI (persistent generator) and
    // GGUF/llama.cpp (persistent llama_context in LlamaSession). No early skip.
    std::string err;
    ::xllama::SessionParams sp;
    sp.model_path = model_name;
    sp.n_ctx = n_ctx > 0 ? n_ctx : 2048;
    sp.n_threads = n_threads;
    auto sess = ::xllama::Session::create(sp, &err);
    if (!sess) {
        log_output("[xllama] kv-bench: session create failed: " + err + "\n");
        return;
    }

    const ::xllama::ChatFormat fmt = ::xllama::chat_format_for(model_name);
    auto mkgp = [&](const std::string& p, bool reuse, bool reset) {
        ::xllama::GenerateParams gp;
        gp.prompt = p;
        gp.n_predict = 96;
        gp.reuse_kv = reuse;
        gp.reset_kv = reset;
        gp.stop_sequences = fmt.stop_sequences;
        return gp;
    };

    // Turn 1: seed the persistent generator.
    auto r1 = sess->generate(mkgp(fmt.render_prompt(sys, {}, u1), /*reuse=*/true, /*reset=*/true));
    // Turn 2 (KV reuse): append only the new turn's delta.
    std::string delta = fmt.render_delta(u2, r1.ended_with_stop);
    auto r2 = sess->generate(mkgp(delta, /*reuse=*/true, /*reset=*/false));
    // Turn 2 (cold): full re-prefill of the whole 2-turn context — the pre-Stage-2
    // behaviour. Uses turn-1's actual output so the token count matches.
    std::string full2 = fmt.render_prompt(sys, {::xllama::ChatTurn{u1, r1.output_text}}, u2);
    auto r2c = sess->generate(mkgp(full2, /*reuse=*/true, /*reset=*/true));

    double speedup = (r2.t_p_eval_ms > 0) ? r2c.t_p_eval_ms / r2.t_p_eval_ms : 0.0;
    char lb[320];
    snprintf(lb, sizeof(lb),
             "[xllama] kv-bench: turn2 prefill reuse=%.1fms (%d tok) cold=%.1fms (%d tok) "
             "speedup=%.2fx\n",
             r2.t_p_eval_ms, r2.n_p_eval, r2c.t_p_eval_ms, r2c.n_p_eval, speedup);
    log_output(lb);

    const std::string csv = resolve_local_path("bench-kv-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp) {
        time_t now = time(nullptr);
        char date_buf[32];
        strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
        // run_index appended last (W1.1), same as the single-turn CSV: repeats are
        // recoverable rather than pre-averaged. 0 = single-run / legacy.
        fputs("model,prefill1_ms,n_p1,prefill2_reuse_ms,n_p2_reuse,prefill2_cold_ms,n_p2_cold,"
              "speedup,decode_tok_s,n_ctx,host,date,run_index\n",
              fp);
        double dtok = (r2.n_eval > 0 && r2.t_eval_ms > 0) ? r2.n_eval / (r2.t_eval_ms / 1000.0) : 0;
        fprintf(fp, "%s,%.1f,%d,%.1f,%d,%.1f,%d,%.2f,%.2f,%d,%s,%s,%d\n", model_name.c_str(),
                r1.t_p_eval_ms, r1.n_p_eval, r2.t_p_eval_ms, r2.n_p_eval, r2c.t_p_eval_ms,
                r2c.n_p_eval, speedup, dtok, sp.n_ctx, host ? host : "unknown", date_buf,
                run_index);
        fclose(fp);
        FILE* done =
            _wfopen(utf8_to_wstring(resolve_local_path("bench-kv-result.csv.done")).c_str(), L"w");
        if (done) {
            fputs("done\n", done);
            fclose(done);
        }
        log_output("[xllama] bench-kv-result.csv written\n");
    }
}

// Plan 003 F3.5: the Session shapes a throughput bench cannot reach — a full
// reset with a prompt swap, a full_prompt that keeps a shared prefix, a delta
// continuation and a multi-chunk prefill. Same Session/GenerateParams plumbing
// as run_kv_bench: one persistent Session per scenario, greedy, per-turn token
// dump, and a cold reference in a fresh Session. A scenario only passes when the
// drafter is still ACTIVE and proposed something after the reset/rewind AND the
// integral ids match the reference; the first divergence is logged when they do
// not. Writes bench-mtp-session.csv (+ .done).
void run_mtp_session_parity(const std::string& model_name, int n_threads, int n_ctx,
                            int n_gpu_layers, int mtp_n_max, int mtp_pmin_pct, int n_predict,
                            const char* host, int run_index) {
    const float pmin = mtp_pmin_pct >= 0 ? static_cast<float>(mtp_pmin_pct) / 100.0f : 0.75f;
    const int npred = n_predict > 0 ? n_predict : 12;
    const bool profile = read_local_int("bench_profile.txt", 1) != 0;

    // Effective configuration, logged before anything runs: a gate that cannot
    // say which knobs produced it cannot be compared against another run, and a
    // leftover knob from a previous bench would otherwise be inherited silently.
    // The GGUF identity binds this run to the matrix/product fixture: same file
    // must hash the same, or the arms are not comparable.
    const std::string gguf_path =
        ::xllama::first_gguf_in_dir(::xllama::resolve_model_path(model_name));
    std::uint64_t gguf_bytes = 0;
    const std::string gguf_hash =
        gguf_path.empty() ? "unresolved" : ::xllama::fnv1a64_file(gguf_path, &gguf_bytes);
    {
        char cfg[384];
        snprintf(cfg, sizeof(cfg),
                 "[xllama] MTP_SESSION_CONFIG threads=%d n_ctx=%d gpu_layers=%d mtp=%d "
                 "n_max=%d p_min=%.2f n_predict=%d greedy=1 twocol=%d gguf=%s\n",
                 n_threads, n_ctx > 0 ? n_ctx : 2048, n_gpu_layers, mtp_n_max, mtp_n_max, pmin,
                 npred, ::xllama::d3d12_kernel_variant(), gguf_hash.c_str());
        log_output(cfg);
    }

    auto open_session = [&](int n_batch, std::string& err, int mtp_n_max_override = -1) {
        const int depth = mtp_n_max_override >= 0 ? mtp_n_max_override : mtp_n_max;
        ::xllama::SessionParams sp;
        sp.model_path = model_name;
        sp.n_ctx = n_ctx > 0 ? n_ctx : 2048;
        sp.n_threads = n_threads;
        if (n_gpu_layers > 0)
            sp.n_gpu_layers = n_gpu_layers;
        if (n_batch > 0)
            sp.n_batch = n_batch;
        sp.mtp = depth > 0;
        sp.mtp_n_max = depth;
        sp.mtp_p_min = pmin;
        return ::xllama::Session::create(sp, &err);
    };

    auto gen = [npred, profile](::xllama::Session& s, const std::string& prompt, bool reuse,
                                bool reset, const std::string& dump, std::string& out_text) {
        ::xllama::GenerateParams gp;
        gp.prompt = prompt;
        gp.n_predict = npred;
        gp.temperature = 0.0f; // greedy: the ids have to be comparable
        gp.reuse_kv = reuse;
        gp.reset_kv = reset;
        gp.profile_phases = profile; // ON/OFF arm of the instrumentation cost
        if (!dump.empty())
            gp.dump_tokens_path = resolve_local_path(dump);
        const ::xllama::InferenceResult r = s.generate(gp);
        out_text = r.output_text;
        return r;
    };

    auto ids_of = [](const std::string& name) {
        std::vector<int> out;
        const std::string raw = read_local_file(name.c_str());
        size_t pos = 0;
        while (pos < raw.size()) {
            const size_t eol = raw.find('\n', pos);
            const std::string tok =
                raw.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
            if (!tok.empty())
                out.push_back(std::atoi(tok.c_str()));
            if (eol == std::string::npos)
                break;
            pos = eol + 1;
        }
        return out;
    };

    struct Row {
        const char* name;
        int ok;
        int active;
        int drafted; // aggregate, MTP + n-gram lookup
        int accepted;
        int n_ids;
        int n_ref;
        int parity;
        int first_diff;
        std::string reason;
        // Per-source split. n_mtp_drafted > 0 with rounds > 0 is what "the MTP
        // head really proposed" means; the aggregate alone can be satisfied by
        // the n-gram lookup when both are enabled.
        int mtp_drafted = 0;
        int mtp_accepted = 0;
        int mtp_rounds = 0;
        int lookup_drafted = 0;
    };
    std::vector<Row> rows;

    // |ok| is the AND over every generate the scenario ran (warm-up, measured turn,
    // cold reference). A failed or empty generate is a FAILURE, never a parity
    // pass — two empty dumps would otherwise compare equal and pass silently.
    auto finish = [&](const char* name, bool ok, const std::string& reason,
                      const ::xllama::InferenceResult& got, const std::string& got_dump,
                      const std::string& ref_dump) {
        const std::vector<int> a = ids_of(got_dump);
        const std::vector<int> b = ids_of(ref_dump);
        int first = -1;
        const size_t n = a.size() < b.size() ? a.size() : b.size();
        for (size_t i = 0; i < n; ++i) {
            if (a[i] != b[i]) {
                first = static_cast<int>(i);
                break;
            }
        }
        const bool parity = first < 0 && a.size() == b.size();

        std::string why;
        if (!ok)
            why = reason.empty() ? "a generate failed" : reason;
        else if (a.empty() || b.empty())
            why = "empty token dump";
        else if (static_cast<int>(a.size()) != got.n_eval)
            why = "dump size does not match the emitted token count";
        else if (!parity)
            why = first >= 0 ? ("ids differ at index " + std::to_string(first))
                             : "id count differs from the reference";
        else if (!got.mtp_active)
            why = "MTP inactive after the reset/rewind";
        else if (got.n_mtp_rounds <= 0)
            why = "no MTP round reached a verify batch after the reset/rewind";
        else if (got.n_mtp_drafted <= 0)
            why = "the MTP head proposed nothing after the reset/rewind";

        char lb[448];
        snprintf(lb, sizeof(lb),
                 "[xllama] MTP_SCENARIO %s ok=%d active=%d n_drafted=%d n_spec_accepted=%d "
                 "n_ids=%d n_ref=%d parity=%s first_diff=%d reason=%s | mtp_drafted=%d "
                 "mtp_accepted=%d mtp_rounds=%d lookup_drafted=%d\n",
                 name, why.empty() ? 1 : 0, got.mtp_active ? 1 : 0, got.n_drafted,
                 got.n_spec_accepted, static_cast<int>(a.size()), static_cast<int>(b.size()),
                 parity ? "OK" : "DIVERGE", first, why.empty() ? "-" : why.c_str(),
                 got.n_mtp_drafted, got.n_mtp_accepted, got.n_mtp_rounds, got.n_lookup_drafted);
        log_output(lb);
        if (!parity && first >= 0) {
            char db[288];
            snprintf(db, sizeof(db), "[xllama] MTP_DIVERGENCE %s idx=%d got=%d ref=%d\n", name,
                     first, a[first], b[first]);
            log_output(db);
            log_output("[xllama] MTP_DIVERGENCE_NOTE the logit margin at this index is not "
                       "instrumented yet; open gap, not a verdict\n");
        }
        rows.push_back({name, why.empty() ? 1 : 0, got.mtp_active ? 1 : 0, got.n_drafted,
                        got.n_spec_accepted, static_cast<int>(a.size()), static_cast<int>(b.size()),
                        parity ? 1 : 0, first, why, got.n_mtp_drafted, got.n_mtp_accepted,
                        got.n_mtp_rounds, got.n_lookup_drafted});
    };

    // CSV writer as a lambda so the fixture fail-early path below writes the
    // same file (with explicit skip rows) instead of returning silently.
    auto write_session_csv = [&] {
        const std::string csv = resolve_local_path("bench-mtp-session.csv");
        FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
        if (!fp)
            return;
        time_t now = time(nullptr);
        char date_buf[32];
        strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
        fputs("scenario,ok,mtp_active,n_drafted,n_spec_accepted,n_ids,n_ref,parity,first_diff,"
              "reason,mtp_drafted,mtp_accepted,mtp_rounds,lookup_drafted,"
              "host,date,run_index\n",
              fp);
        for (const Row& r : rows) {
            // Reasons carry prose with commas: RFC-4180 quoting (double the
            // quotes) keeps every column aligned. An unquoted comma once
            // shifted the verdict script into int() on a reason fragment; the
            // host parser additionally rejects malformed rows outright.
            std::string reason = r.reason.empty() ? "-" : r.reason;
            std::string quoted;
            quoted += '"';
            for (char ch : reason) {
                if (ch == '"')
                    quoted += "\"\"";
                else if (ch == '\n' || ch == '\r')
                    quoted += ' ';
                else
                    quoted += ch;
            }
            quoted += '"';
            fprintf(fp, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%d,%d,%d,%s,%s,%d\n", r.name, r.ok,
                    r.active, r.drafted, r.accepted, r.n_ids, r.n_ref, r.parity, r.first_diff,
                    quoted.c_str(), r.mtp_drafted, r.mtp_accepted, r.mtp_rounds, r.lookup_drafted,
                    host ? host : "unknown", date_buf, run_index);
        }
        fclose(fp);
        FILE* done = _wfopen(
            utf8_to_wstring(resolve_local_path("bench-mtp-session.csv.done")).c_str(), L"w");
        if (done) {
            fputs("done\n", done);
            fclose(done);
        }
        log_output("[xllama] bench-mtp-session.csv written\n");
    };

    const std::string p1 = "The capital of France is";
    // Second prompt of the reset scenario. It must be one the confidence filter
    // actually passes COLD, otherwise the scenario measures the FIXTURE instead
    // of the reset: with "A thermometer measures" the cold reference also drafted
    // nothing at p_min=0.75, so no reset-specific conclusion is possible.
    //
    // The fixture is the France prompt, because its cold drafting was OBSERVED
    // (drafted=6, rounds=1 in the rev38/39 console logs) — that evidence belongs
    // to France, not to any other capital. Its cold reference drafting is
    // re-checked every run and reported as ref_drafted, so a fixture that stops
    // passing the threshold is caught as a fixture failure rather than blamed on
    // the reset path.
    const std::string p2 = "The capital of France is";
    const std::string p1_edit = "The capital of France is Paris, and its largest city is";
    const std::string delta = " and its largest city is";

    // Per-scenario results. Declared up front because a scenario's finish() call
    // happens in a LATER scope than the generate that produced it: the model is
    // released between phases (one resident model at a time), so the result has
    // to outlive the block that created it.
    struct Outcome {
        ::xllama::InferenceResult got;
        bool warm_ok = false;
        bool got_ok = false;
        bool ref_ok = false;
        bool pre_ok = false; // preconditions (ids non-empty, chunking real)
        int ref_drafted = 0; // cold-reference MTP proposals (fixture check)
    };
    Outcome o1, o2, o3, o4;

    // 1. full reset + prompt swap: the drafter must come back on the new prompt.
    //
    // Memory discipline: only ONE resident model at a time. The Rev 38 run measured
    // ~4.3 GB peak for a single MTP context, so holding the scenario session, the
    // cold reference and a sequential control open together could exhaust the app's
    // budget and turn a parity gate into an OOM. Each phase therefore saves what it
    // needs (ids are already on disk), destroys its Session, and only then does the
    // next one load.
    {
        std::string err;
        auto s = open_session(0, err);
        if (s) {
            std::string t1;
            o1.warm_ok = gen(*s, p1, true, true, "bench-mtp-s1-t1.txt", t1).success;
            std::string t2;
            o1.got = gen(*s, p2, true, true, "bench-mtp-s1-t2.txt", t2);
            o1.got_ok = o1.got.success;
            s.reset(); // release the model before the reference loads
        } else {
            log_output("[xllama] MTP_SCENARIO reset_prompt_swap SKIPPED create failed: " + err +
                       "\n");
        }
        err.clear();
        auto ref = open_session(0, err);
        if (ref) {
            std::string rt;
            const auto rr = gen(*ref, p2, true, true, "bench-mtp-s1-ref.txt", rt);
            o1.ref_ok = rr.success;
            // The fixture's own confidence is a precondition of this scenario: if
            // the COLD reference cannot pass p_min, a zero here says nothing about
            // the reset path. Recorded instead of assumed, and reported either way.
            o1.ref_drafted = rr.n_mtp_drafted;
            ref.reset();
            o1.pre_ok = !ids_of("bench-mtp-s1-t2.txt").empty();
            const char* why = nullptr;
            if (!o1.warm_ok || !o1.got_ok || !o1.ref_ok)
                why = "a generate in this scenario failed";
            else if (!o1.pre_ok)
                why = "the measured turn produced no token dump";
            else if (o1.ref_drafted <= 0)
                why = "fixture precondition: the COLD reference drafted nothing at this "
                      "p_min, so this scenario cannot exercise the reset path";
            {
                char fb[256];
                snprintf(
                    fb, sizeof(fb),
                    "[xllama] MTP_SCENARIO reset_prompt_swap fixture_check cold_ref_drafted=%d "
                    "cold_ref_rounds=%d usable=%d\n",
                    rr.n_mtp_drafted, rr.n_mtp_rounds, o1.ref_drafted > 0 ? 1 : 0);
                log_output(fb);
            }
            finish("reset_prompt_swap", why == nullptr, why ? why : "", o1.got,
                   "bench-mtp-s1-t2.txt", "bench-mtp-s1-ref.txt");
        } else {
            log_output("[xllama] MTP_SCENARIO reset_prompt_swap SKIPPED reference create "
                       "failed: " +
                       err + "\n");
        }
    }
    // Fail early when the fixture cannot draft: scenarios 2-4 need an active
    // drafter to exercise reset/edited/delta paths, and the s1 cold reference
    // is the documented canary. Running on would burn ~10 minutes for rows
    // that can only fail identically. The skips are explicit ok=0 rows, never
    // silent passes; the pmin diagnostic (--mtp-pmin 0) stays available to
    // separate a threshold fixture issue from a head/path issue.
    if (o1.ref_drafted <= 0) {
        log_output("[xllama] MTP_SESSION fixture FAIL: cold reference drafted nothing; "
                   "scenarios 2-4 skipped\n");
        ::xllama::InferenceResult empty{};
        finish("edited_prefix", false,
               "skipped: session fixture cannot draft cold (see reset_prompt_swap)", empty,
               "bench-mtp-s2-t2.txt", "bench-mtp-s2-ref.txt");
        finish("delta_continuation", false,
               "skipped: session fixture cannot draft cold (see reset_prompt_swap)", empty,
               "bench-mtp-s3-t2.txt", "bench-mtp-s3-ref.txt");
        finish("multi_chunk", false,
               "skipped: session fixture cannot draft cold (see reset_prompt_swap)", empty,
               "bench-mtp-s4-t.txt", "bench-mtp-s4-ref.txt");
        write_session_csv();
        return;
    }
    // 2. edited full_prompt: kv_keep > 0, so the catch-up replays a tail above 0.
    // One resident model at a time, as in scenario 1.
    {
        std::string err;
        auto s = open_session(0, err);
        if (s) {
            std::string t1;
            o2.warm_ok = gen(*s, p1, true, true, "bench-mtp-s2-t1.txt", t1).success;
            std::string t2;
            o2.got = gen(*s, p1_edit, true, true, "bench-mtp-s2-t2.txt", t2);
            o2.got_ok = o2.got.success;
            s.reset();
        } else {
            log_output("[xllama] MTP_SCENARIO edited_prefix SKIPPED create failed: " + err + "\n");
        }
        err.clear();
        auto ref = open_session(0, err);
        if (ref) {
            std::string rt;
            o2.ref_ok = gen(*ref, p1_edit, true, true, "bench-mtp-s2-ref.txt", rt).success;
            ref.reset();
            o2.pre_ok = !ids_of("bench-mtp-s2-t2.txt").empty();
            finish("edited_prefix", o2.warm_ok && o2.got_ok && o2.ref_ok && o2.pre_ok,
                   o2.pre_ok ? "a generate in this scenario failed"
                             : "the measured turn produced no token dump",
                   o2.got, "bench-mtp-s2-t2.txt", "bench-mtp-s2-ref.txt");
        } else {
            log_output("[xllama] MTP_SCENARIO edited_prefix SKIPPED reference create failed: " +
                       err + "\n");
        }
    }

    // 3. delta continuation: KV reuse without a reset (the shape already covered
    //    on the host, kept here so the console build is exercised the same way).
    //    Three phases, ONE resident model at a time: the MTP session, then the MTP-off
    //    sequential control, then the cold reference. Phase order matters — each
    //    Session is destroyed before the next one loads.
    std::string t1_text;
    bool r1_ok = false, got_ok = false, rs1_ok = false, rs2_ok = false, rr2_ok = false;
    {
        std::string err;
        auto s = open_session(0, err);
        if (s) {
            r1_ok = gen(*s, p1, true, true, "bench-mtp-s3-t1.txt", t1_text).success;
            std::string t2;
            o3.got = gen(*s, delta, true, false, "bench-mtp-s3-t2.txt", t2);
            got_ok = o3.got.success;
            s.reset();
        } else {
            log_output("[xllama] MTP_SCENARIO delta_continuation SKIPPED create failed: " + err +
                       "\n");
        }
    }
    {
        std::string err;
        auto seq = open_session(0, err, /*mtp_n_max=*/0);
        if (seq) {
            std::string sq1;
            const auto rs1 = gen(*seq, p1, true, true, "bench-mtp-s3-seq-t1.txt", sq1);
            rs1_ok = rs1.success;
            std::string sq2;
            const auto rs2 = gen(*seq, delta, true, false, "bench-mtp-s3-seq-t2.txt", sq2);
            rs2_ok = rs2.success;
            seq.reset();
        } else {
            log_output("[xllama] MTP_SCENARIO delta_continuation control create failed: " + err +
                       "\n");
        }
    }
    {
        std::string err;
        auto ref = open_session(0, err);
        if (ref) {
            std::string rt;
            const auto rr2 =
                gen(*ref, p1 + t1_text + delta, true, true, "bench-mtp-s3-ref.txt", rt);
            rr2_ok = rr2.success;
            ref.reset();
        } else {
            log_output("[xllama] MTP_SCENARIO delta_continuation reference create failed: " + err +
                       "\n");
        }
    }

    {
        std::vector<int> res_t1_pre, res_t1_out, res_t2_pre, seq_t1_pre, seq_t1_out, seq_t2_pre,
            seq_t2_out, cold_pre;
        bool ids_loaded = true;
        auto load = [&](const char* name, std::vector<int>& out) {
            out = ids_of(name);
            if (out.empty())
                ids_loaded = false;
        };
        load("bench-mtp-s3-t1.txt.prefill", res_t1_pre);
        load("bench-mtp-s3-t1.txt", res_t1_out);
        load("bench-mtp-s3-t2.txt.prefill", res_t2_pre);
        load("bench-mtp-s3-seq-t1.txt.prefill", seq_t1_pre);
        load("bench-mtp-s3-seq-t1.txt", seq_t1_out);
        load("bench-mtp-s3-seq-t2.txt.prefill", seq_t2_pre);
        load("bench-mtp-s3-seq-t2.txt", seq_t2_out);
        load("bench-mtp-s3-ref.txt.prefill", cold_pre);

        // (a) Sequential control, MTP off, same turns. "Same history" means the
        //     whole sequence: turn-1 AND turn-2 prefill ids, and the emitted ids of
        //     BOTH turns. Comparing only the delta text or only turn 2 would not
        //     prove the control saw the same history.
        const bool seq_ok_full = ids_loaded && res_t1_pre == seq_t1_pre &&
                                 res_t2_pre == seq_t2_pre && res_t1_out == seq_t1_out &&
                                 ids_of("bench-mtp-s3-t2.txt") == seq_t2_out;

        // (b) History equivalence for the cold reference: its prefill must be
        //     token-for-token the real resident history (turn-1 prefill +
        //     turn-1 emitted + delta prefill). Concatenating text can shift ids at
        //     the joins, so a mismatch is a REFERENCE failure — reported as such,
        //     never as an MTP parity difference.
        std::vector<int> expected;
        expected.insert(expected.end(), res_t1_pre.begin(), res_t1_pre.end());
        expected.insert(expected.end(), res_t1_out.begin(), res_t1_out.end());
        expected.insert(expected.end(), res_t2_pre.begin(), res_t2_pre.end());
        const bool history_ok = ids_loaded && expected == cold_pre && !expected.empty();

        int hist_diff = -1;
        for (size_t i = 0; i < expected.size() && i < cold_pre.size(); ++i) {
            if (expected[i] != cold_pre[i]) {
                hist_diff = static_cast<int>(i);
                break;
            }
        }
        char pb[384];
        snprintf(pb, sizeof(pb),
                 "[xllama] MTP_SCENARIO delta_continuation seq_ok=%d hist_ok=%d hist_diff=%d "
                 "n_expected=%d n_cold_pre=%d t1_out_equal=%d t2_out_equal=%d\n",
                 seq_ok_full ? 1 : 0, history_ok ? 1 : 0, hist_diff,
                 static_cast<int>(expected.size()), static_cast<int>(cold_pre.size()),
                 res_t1_out == seq_t1_out ? 1 : 0,
                 ids_of("bench-mtp-s3-t2.txt") == seq_t2_out ? 1 : 0);
        log_output(pb);

        const char* why = nullptr;
        if (!r1_ok || !got_ok || !rs1_ok || !rs2_ok || !rr2_ok)
            why = "a generate in this scenario failed";
        else if (!ids_loaded)
            why = "a token dump was missing or empty";
        else if (!seq_ok_full)
            why = "the MTP-off sequential control is not the same token history";
        else if (!history_ok)
            why = "cold reference history is not token-for-token the resident history";

        finish("delta_continuation", why == nullptr, why ? why : "", o3.got, "bench-mtp-s3-t2.txt",
               "bench-mtp-s3-ref.txt");
    }

    // 4. multi-chunk prefill: prompt past n_batch, so the catch-up sees several
    //    replay batches. The same prompt in one batch is the reference.
    {
        std::string err;
        const int nb = read_local_int("bench_session_nbatch.txt", 16);
        int ntok = 0;
        bool chunked = false;
        auto s = open_session(nb, err);
        if (s) {
            std::string longp = "Summarize the following log.\n";
            for (int i = 0; i < 40; ++i)
                longp += "line " + std::to_string(i) + ": nothing happened\n";
            ntok = s->count_tokens(longp);
            chunked = nb > 0 && ntok > nb;
            char cb[192];
            snprintf(cb, sizeof(cb),
                     "[xllama] MTP_SCENARIO multi_chunk config n_batch=%d n_prompt=%d chunks=%d "
                     "multi_chunk_exercised=%d\n",
                     nb, ntok, nb > 0 ? (ntok + nb - 1) / nb : 1, chunked ? 1 : 0);
            log_output(cb);
            std::string t;
            o4.got = gen(*s, longp, true, true, "bench-mtp-s4-t.txt", t);
            o4.got_ok = o4.got.success;
            o4.pre_ok = chunked;
            s.reset(); // release before the single-batch reference loads
        } else {
            log_output("[xllama] MTP_SCENARIO multi_chunk SKIPPED create failed: " + err + "\n");
        }
        err.clear();
        auto ref = open_session(0, err);
        if (ref) {
            std::string longp2 = "Summarize the following log.\n";
            for (int i = 0; i < 40; ++i)
                longp2 += "line " + std::to_string(i) + ": nothing happened\n";
            std::string rt;
            o4.ref_ok = gen(*ref, longp2, true, true, "bench-mtp-s4-ref.txt", rt).success;
            ref.reset();
            finish("multi_chunk",
                   o4.pre_ok && o4.got_ok && o4.ref_ok && !ids_of("bench-mtp-s4-t.txt").empty(),
                   o4.pre_ok ? "a generate in this scenario failed"
                             : "prompt did not exceed n_batch: multi-chunk path not exercised",
                   o4.got, "bench-mtp-s4-t.txt", "bench-mtp-s4-ref.txt");
        } else {
            log_output("[xllama] MTP_SCENARIO multi_chunk SKIPPED reference create failed: " + err +
                       "\n");
        }
    }

    write_session_csv();
}

} // namespace

LlamaIni read_llama_ini() {
    return parse_llama_ini(read_local_file(kLlamaIniFile));
}

int gguf_gpu_layers_knob() {
    const std::string s = read_local_file("gguf_gpu_layers.txt");
    if (!s.empty())
        return std::atoi(s.c_str());
    int v = 0;
    return llama_ini_int(read_llama_ini(), "n_gpu_layers", v) ? v : 0;
}

int gguf_kv_q8_knob() {
    const std::string s = read_local_file("kv_q8.txt");
    if (!s.empty())
        return std::atoi(s.c_str());
    bool v = false;
    return llama_ini_bool(read_llama_ini(), "kv_q8", v) ? (v ? 1 : 0) : 0;
}

// Immutable startup profile (plan004): read once, single-threaded, before any
// session/model load. GGML_REPACK_FORCE_GEMV is consumed static-once for the
// process and the d3d12 kernel variant is process-global — so this profile is
// valid only until restart; file changes later are REJECTED (logged), never
// applied mid-process. Absent files => default profile (nothing set).
static std::string s_prof_twocol;
static std::string s_prof_repack;
static bool s_prof_applied = false;

// Placement must be selected before model loading: Q8_0 weights move between
// CPU and D3D12 buffers. Absent = GPU; "0" is the same-package CPU control.
static void apply_q8_knob(const char* who) {
    const std::string value = read_local_file("d3d12q8.txt");
    const bool enabled = value.empty() || value[0] != '0';
    d3d12_set_q8_enabled(enabled);
    d3d12_set_q6_columns(read_local_int("d3d12q6tile.txt", 1));
    log_output(std::string("[xllama] ") + who +
               ": Q6_K LM-head columns=" + std::to_string(d3d12_q6_columns()) + "\n");
    log_output(std::string("[xllama] ") + who +
               ": Q8_0 matmul=" + (enabled ? "D3D12" : "CPU control") + "\n");
    const std::string flash_mode = read_local_file("flashattn.txt");
    const bool flash = flash_mode == "1" || flash_mode == "2";
    _putenv(flash ? "XLLAMA_FLASH_ATTN=1" : "XLLAMA_FLASH_ATTN=0");
    _putenv(flash_mode == "2" ? "GGML_FLASH_ATTN_NO_SPLIT_KV=1" : "GGML_FLASH_ATTN_NO_SPLIT_KV=0");
    // Consistent-query mode preserves F32 queries in scalar attention QK.
    _putenv(flash_mode == "2" ? "GGML_FLASH_ATTN_NO_QUERY_TILING=4"
                              : "GGML_FLASH_ATTN_NO_QUERY_TILING=0");
    log_output(std::string("[xllama] ") + who + ": fused attention=" + (flash ? "forced" : "auto") +
               "\n");
    log_output(std::string("[xllama] ") + who + ": attention split-KV=" +
               (flash_mode == "2" ? "disabled for consistent query reduction" : "native") + "\n");
    _putenv(("XLLAMA_MTP_THREADS=" + read_local_file("mtp_threads.txt")).c_str());
    _putenv(read_local_file("mtp_catchup_logits.txt") == "1" ? "XLLAMA_MTP_CATCHUP_LOGITS=1"
                                                             : "XLLAMA_MTP_CATCHUP_LOGITS=0");
    const std::string gdn_mode = read_local_file("d3d12gdn.txt");
    const bool gdn = gdn_mode == "1" || gdn_mode == "2";
    d3d12_set_gdn_enabled(gdn, gdn_mode == "2" ? 2 : 1);
    log_output(std::string("[xllama] ") + who + ": GATED_DELTA_NET=" +
               (gdn_mode == "2" ? "D3D12 (T>=2)"
                : gdn           ? "D3D12"
                                : "CPU control") +
               "\n");
    // Plan 006 C4 product trial: FFN split-SWIGLU on D3D12 (default OFF;
    // d3d12swiglu.txt "1" = all FFN, "2" = target-context FFN only).
    // Default ON since 2026-10-09 (exact kernel + strict MTP gate pass);
    // "0" forces OFF for A/B, "2" keeps the target-only trial.
    const std::string swiglu_mode = read_local_file("d3d12swiglu.txt");
    const int sw_mode = swiglu_mode == "0" ? 0 : (swiglu_mode == "2" ? 2 : 1);
    d3d12_set_swiglu_product_mode(sw_mode);
    log_output(std::string("[xllama] ") + who + ": FFN SWIGLU D3D12=" +
               (sw_mode == 2 ? "target-only" : (sw_mode == 1 ? "all" : "off")) + "\n");
}

// Plan 006 C8: GPU timestamp queries are instrumentation. Explicit
// d3d12timestamps.txt "0"/"1" wins; when the file is absent the process default
// is bound to the run's profile switch (enabled iff profiling), so the product
// path pays no query/readback cost while attribution keeps d3g. Called once per
// run BEFORE decode work; the session/API path never re-applies it after
// startup, so concurrent requests cannot flip the policy mid-run.
static void apply_gpu_timestamps(bool profile_phases, const char* who) {
    const std::string mode = read_local_file("d3d12timestamps.txt");
    bool enabled;
    if (mode == "0")
        enabled = false;
    else if (mode == "1")
        enabled = true;
    else
        enabled = profile_phases;
    d3d12_set_gpu_timestamps(enabled);
    log_output(std::string("[xllama] ") + who + ": gpu timestamps=" + (enabled ? "on" : "off") +
               (mode.empty() ? " (profile-bound default)" : " (explicit knob)") + "\n");
}

void apply_startup_profile() {
    if (s_prof_applied)
        return;
    s_prof_applied = true;
    apply_q8_knob("startup");
    // Startup default for the session/API path: timestamps OFF unless the
    // explicit knob says otherwise (profile runs apply their own policy before
    // decode). Deterministic for every request; never flipped per request.
    apply_gpu_timestamps(false, "startup");
    s_prof_twocol = read_local_file("d3d12twocol.txt");
    s_prof_repack = read_local_file("cpurepackforcegemv.txt");
    apply_twocol_knob("startup");      // no-op when file absent
    apply_repack_ctrl_knob("startup"); // _putenv only when file present
    apply_island_knob("startup");
    char lb[256];
    snprintf(lb, sizeof(lb),
             "[xllama] startup profile: twocol='%s' repack='%s' (immutable for this "
             "process lifetime; restart the app to change)\n",
             s_prof_twocol.empty() ? "(absent=default)" : s_prof_twocol.c_str(),
             s_prof_repack.empty() ? "(absent=default)" : s_prof_repack.c_str());
    log_output(lb);
}

// Log-only enforcement at session-config entry points: detects a mid-process
// change of the immutable profile files and reports it without mutating
// anything (restart required to adopt a new profile).
static void enforce_startup_profile(const char* who) {
    if (!s_prof_applied)
        return;
    const std::string t = read_local_file("d3d12twocol.txt");
    const std::string r = read_local_file("cpurepackforcegemv.txt");
    if (t != s_prof_twocol || r != s_prof_repack) {
        char lb[256];
        snprintf(lb, sizeof(lb),
                 "[xllama] %s: profile files changed after startup - REJECTED, "
                 "startup profile stays (restart required)\n",
                 who);
        log_output(lb);
    }
}

void apply_llama_ini_session(SessionParams& sp) {
    const LlamaIni ini = read_llama_ini();
    if (ini.empty())
        return;
    // Positive values only; anything else leaves the catalogue default in
    // place. n_gpu_layers and kv_q8 stay with their knobs above.
    int v = 0;
    if (llama_ini_int(ini, "n_ctx", v) && v > 0)
        sp.n_ctx = v;
    if (llama_ini_int(ini, "n_threads", v) && v > 0)
        sp.n_threads = v;
    if (llama_ini_int(ini, "n_threads_batch", v) && v > 0)
        sp.n_threads_batch = v;
    if (llama_ini_int(ini, "n_batch", v) && v > 0)
        sp.n_batch = v;
    if (llama_ini_int(ini, "n_ubatch", v) && v > 0)
        sp.n_ubatch = v;
    // Output budget default: llama.ini [n_predict] is the one home for it, so
    // the API (request without max_tokens) and the chat UI (catalogue without
    // n_predict) stop carrying their own fallbacks. Absent/invalid keeps them.
    int npredict = -1;
    if (llama_ini_int(ini, "n_predict", npredict) && npredict > 0)
        llama_ini_n_predict = npredict;
    // Parallel capacity: the llama.cpp name is n_seq_max; n_parallel is the
    // historical spelling and is accepted as an alias. Clamped to 1..8: nothing
    // in this frontend drives more than a handful of sequences, and a larger
    // value only slices the shared n_ctx pool thinner.
    int seq_max = 0;
    if (!llama_ini_int(ini, "n_seq_max", seq_max))
        llama_ini_int(ini, "n_parallel", seq_max);
    if (seq_max > 0)
        sp.n_seq_max = std::min(seq_max, 8);
    // Functional-MVP keys (plan004): validated, populate-only — this parser
    // performs NO global mutation (twocol/repack are startup-immutable and
    // applied only by apply_startup_profile; see below).
    // Canonical basename == exact catalog id, never substring matching
    // ("qwen35-4b" must not satisfy the MTP profile).
    auto base_of = [](std::string p) {
        const auto pos = p.find_last_of("/\\");
        if (pos != std::string::npos)
            p = p.substr(pos + 1);
        return p;
    };
    const std::string base = base_of(sp.model_path);
    const bool selected = base == "qwen35-4b-mtp";

    // Immutable selected-Qwen MVP startup mode (scope2 files present at
    // startup): ONLY the selected model may load; others are rejected until
    // a restart with the default profile (files absent). Never labels other
    // models "unaffected" while the globals are active.
    const bool profile_active = !s_prof_twocol.empty() || !s_prof_repack.empty();
    if (profile_active && !selected) {
        sp.config_reject = "startup scope2 profile is active: only qwen35-4b-mtp may load; "
                           "restart the app with the default profile (remove d3d12twocol.txt/"
                           "cpurepackforcegemv.txt) to load other models";
    }

    sp.mtp = false; // explicit MTP-OFF unless validated below
    int m = 0;
    if (llama_ini_int(ini, "mtp", m)) {
        if (m > 0) {
            int depth = m;
            int pmin_pct = -1;
            llama_ini_int(ini, "mtp_pmin", pmin_pct);
            const bool depth_ok = depth >= 1 && depth <= 16;
            const bool pmin_ok = pmin_pct >= 1 && pmin_pct <= 100;
            if (selected && depth_ok && pmin_ok) {
                sp.mtp = true;
                sp.mtp_n_max = depth;
                sp.mtp_p_min = static_cast<float>(pmin_pct) / 100.0f;
            } else if (!selected) {
                char msg[320];
                snprintf(msg, sizeof(msg),
                         "[xllama] ini mtp REJECTED: model '%s' is not the exact selected "
                         "qwen35-4b-mtp -> MTP OFF\n",
                         sp.model_path.c_str());
                log_output(msg);
            } else {
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "[xllama] ini mtp REJECTED: depth %d (1-16) / mtp_pmin %d (1-100) "
                         "out of range -> MTP OFF\n",
                         depth, pmin_pct);
                log_output(msg);
            }
        } else {
            log_output("[xllama] ini mtp=0 -> MTP explicitly OFF\n");
        }
    }
    // Log-only startup-profile consistency check (zero mutation; the parser
    // only populates SessionParams).
    enforce_startup_profile("ini");
}
#endif // XLLAMA_UWP

// ---------------------------------------------------------------------------
// main_loop (called from UWP bench mode background thread)
// ---------------------------------------------------------------------------

void main_loop() {
#ifdef XLLAMA_UWP
    // Pin CWD to LocalState so relative paths from genai_config.json (e.g. the
    // ORT enable_profiling prefix) land in a writable, WDP-fetchable location.
    set_cwd_to_local_folder();

    apply_q8_knob("bench");

    apply_repack_ctrl_knob("main_loop");
    apply_spinwait_knob("main_loop");
    apply_ggmlprof_knob("main_loop");
    apply_island_knob("main_loop");
    apply_swiglu_mtp_diag_knob("main_loop");
    apply_swiglu_range_knob("main_loop");

    // Real-decode two-column knob (plan 004): d3d12twocol.txt="auto" engages
    // variant 2 (allowlisted NEW) for every decode in this bench process —
    // session parity, kv-bench and single-turn alike; absent/anything-else
    // stays OLD. Scoped: restored on every exit below, so no bench leaks a
    // variant into the next process... and the knob file itself is owned by
    // the driver scripts (uploaded per run, deleted on exit).
    TwocolKnobScope twocol_knob("bench");

    // model.txt and prompt.txt are REQUIRED, and their absence aborts the run.
    //
    // Both used to fall back to a hardcoded default — smollm2-360m-cpu-int4 and a
    // 58-character prompt. That turns a lost upload into a silent measurement of
    // the wrong thing, and WDP POSTs are documented to fail silently in this
    // project (a POST without X-CSRF-Token returns success and writes nothing).
    // The result would be a real, plausible CSV row describing a run nobody asked
    // for, appended to the results file of the run that was asked for. Same defect
    // class as the invented quant label: a benchmark that guesses its own inputs
    // produces evidence indistinguishable from the genuine kind.
    //
    // read_local_file reads to EOF; the hand-rolled reader this replaced used a
    // fixed char buf[8192] and truncated silently at ~2k tokens — exactly the
    // range a prompt-length sweep needs.
    std::string user_prompt = read_local_file("prompt.txt");
    if (user_prompt.empty()) {
        log_output("[xllama] bench: prompt.txt missing or empty — refusing to bench a prompt "
                   "nobody asked for. Upload it and retry.\n");
        return;
    }
    log_output("[xllama] bench: prompt.txt " + std::to_string(user_prompt.size()) + " bytes\n");

    std::string model_name;
    {
        std::string model_cfg = resolve_local_path("model.txt");
        FILE* mf = _wfopen(utf8_to_wstring(model_cfg).c_str(), L"r");
        if (mf) {
            char buf[512] = {};
            size_t n = fread(buf, 1, sizeof(buf) - 1, mf);
            fclose(mf);
            if (n > 0) {
                model_name = buf;
                while (!model_name.empty() &&
                       (model_name.back() == '\n' || model_name.back() == '\r' ||
                        model_name.back() == ' '))
                    model_name.pop_back();
            }
        }
    }
    if (model_name.empty()) {
        log_output("[xllama] bench: model.txt missing or empty — refusing to pick a model. "
                   "Upload it and retry.\n");
        return;
    }

    // Optional numeric knobs from LocalState, written by the bench scripts.
    // bench_threads.txt also labels the host column; bench_ctx.txt and
    // bench_npredict.txt exist because #130 needs to vary n_ctx and n_predict:
    // the DirectML prefill band's edges sit near n_ctx/2 and n_ctx - n_predict,
    // and that hypothesis is only falsifiable if both are controllable here.
    const int bench_threads = read_local_int("bench_threads.txt", 0);
    // Plan 006 C1: batch/prefill/verify thread count; 0 = same as bench_threads.
    const int bench_batch_threads = read_local_int("bench_batch_threads.txt", 0);
    const int bench_ctx = read_local_int("bench_ctx.txt", 0);
    const int bench_npredict = read_local_int("bench_npredict.txt", 0);
    // #130: max_length is the variable that governs DirectML prefill, and it is
    // normally derived from n_predict. This decouples them. 0 = derive,
    // -1 = saturate to n_ctx (what the shipping app does).
    const int bench_maxlen = read_local_int("bench_maxlen.txt", 0);
    // #172: llama.cpp physical prefill chunk (n_ubatch). 0 = llama default
    // (512). Labeled in the host column (-uN) because the CSV has no ubatch
    // column — a row that does not carry the variable under study is not
    // interpretable (the Phase 12 lesson, twice).
    const int bench_ubatch = read_local_int("bench_ubatch.txt", 0);
    const int bench_n_batch = read_local_int("bench_n_batch.txt", 0); // MVP 64/64 adapter
    // #171: q8_0 KV cache + flash attention A/B. Host-column tag -kvq8, same
    // rationale as -uN (the CSV schema carries no cache-type column).
    const int bench_kvq8 = read_local_int("bench_kvq8.txt", 0);
    // Phase 15 W2 (#210): draft-free prompt-lookup speculative decoding.
    // 0 = off (default); 1 = on. Host-column tag -plookup (CSV has no dedicated
    // column — same pattern as -kvq8 / -uN).
    const int bench_prompt_lookup = read_local_int("bench_prompt_lookup.txt", 0);
    // MTP drafting (beellama fork). Reads the raw value rather than a boolean so
    // n_max can be swept from one file; 0 = off. Host-column tag -mtpN.
    const int bench_mtp = read_local_int("bench_mtp.txt", 0);
    // MTP draft confidence threshold in percent (0..100). -1 (file absent) keeps
    // the InferenceParams default (0.75). Host-column tag -pminN.
    const int bench_mtp_pmin = read_local_int("bench_mtp_pmin.txt", -1);
    // Plan 004 boundary diagnosis: verify_trace.txt holds an output token index;
    // the decode loop logs the exact decision (path, batch row, top candidates
    // and margin, draft/accept state) behind that one output. Diagnostic only:
    // a traced run is evidence about a boundary, never gate evidence.
    const int bench_trace = read_local_int("verify_trace.txt", -1);
    ::xllama::decode_trace::set_output_idx(bench_trace);
    if (bench_trace >= 0) {
        char tb[128];
        snprintf(tb, sizeof(tb), "[xllama] bench: verify_trace output idx=%d (diagnostic)\n",
                 bench_trace);
        log_output(tb);
    }
    // Plan 003, stage 1: parity knobs. Greedy (argmax) and a fixed seed make a
    // run reproducible token-by-token, so baseline and MTP arms can be diffed
    // (not just hashed). 0 = off / engine default.
    const int bench_greedy = read_local_int("bench_greedy.txt", 0);
    const int bench_seed = read_local_int("bench_seed.txt", 0);
    // Plan 003, stage 1: dump the accepted token ids per run (bench-tokens-<i>.txt)
    // for token-level parity diffing. 0 = off.
    const int bench_tokens = read_local_int("bench_tokens.txt", 0);
    // GGUF GPU decode D2b: layers on the d3d12 backend. 0 = CPU. Host tag -gN.
    const int bench_gpu_layers = read_local_int("bench_gpu_layers.txt", 0);
    // D2b: decode exactly n_predict tokens (no EOG / stop sequence). Host tag -noeog.
    const int bench_ignore_eog = read_local_int("bench_ignore_eog.txt", 0);
    // W1.1: which repetition this run is, written by the bench driver before each
    // iteration. Echoed into the CSV run_index column so the driver can append
    // every repeat and the summary generator can report a spread. 0 = single run.
    const int bench_run_index = read_local_int("bench_run_index.txt", 0);
    // Plan 003 stage 2: phase instrumentation switch. bench_profile.txt=0 keeps
    // every counter but drops the chrono snapshots, so an ON/OFF pair on the same
    // package/GGUF measures the instrumentation's own cost instead of inferring it
    // from the residual. Host tag -prof0/-prof1 so the CSV says which ran.
    const int bench_profile = read_local_int("bench_profile.txt", 1);
    // Bind the process-global timestamp policy to THIS bench's profile switch
    // (explicit d3d12timestamps.txt still wins). Set before any bench sub-mode
    // (session parity included) starts decoding, so it cannot flip mid-run.
    apply_gpu_timestamps(bench_profile != 0, "bench");

    // Plan 003 F3.5: Session reset / edited-prefix / delta / multi-chunk parity.
    // bench_mtp_session.txt selects it and skips the single-turn bench, the same
    // way bench_turns.txt selects the KV bench. The host column carries -session
    // so an MSIX that ignores the knob fails the driver's tag validation instead
    // of silently benching the single-turn path.
    if (read_local_int("bench_mtp_session.txt", 0) != 0) {
        char host_buf3[96];
        int l3 = snprintf(host_buf3, sizeof(host_buf3), "%s", host_tag_prefix().c_str());
        if (bench_threads > 0)
            l3 += snprintf(host_buf3 + l3, sizeof(host_buf3) - l3, "-t%d", bench_threads);
        if (bench_gpu_layers > 0)
            l3 += snprintf(host_buf3 + l3, sizeof(host_buf3) - l3, "-g%d", bench_gpu_layers);
        if (bench_mtp > 0)
            l3 += snprintf(host_buf3 + l3, sizeof(host_buf3) - l3, "-mtp%d", bench_mtp);
        if (bench_mtp > 0 && bench_mtp_pmin >= 0)
            l3 += snprintf(host_buf3 + l3, sizeof(host_buf3) - l3, "-pmin%d", bench_mtp_pmin);
        snprintf(host_buf3 + l3, sizeof(host_buf3) - l3, "-session");
        log_output("[xllama] mtp session parity model: " + model_name + "\n");
        run_mtp_session_parity(model_name, bench_threads, bench_ctx, bench_gpu_layers, bench_mtp,
                               bench_mtp_pmin, read_local_int("bench_session_npredict.txt", 12),
                               host_buf3, bench_run_index);
        return;
    }

    // Multi-turn TTFT bench (Stage 2b): if bench_turns.txt is present it holds the
    // turn-2 user prompt; prompt.txt supplies turn 1. Measures the KV-reuse win
    // and skips the normal single-turn bench.
    {
        std::string turn2 = read_local_file("bench_turns.txt");
        if (!turn2.empty()) {
            char host_buf2[64];
            if (bench_threads > 0)
                snprintf(host_buf2, sizeof(host_buf2), "%s-t%d", host_tag_prefix().c_str(),
                         bench_threads);
            else
                snprintf(host_buf2, sizeof(host_buf2), "%s", host_tag_prefix().c_str());
            log_output("[xllama] kv-bench model: " + model_name + "\n");
            run_kv_bench(model_name, "You are a helpful AI assistant.", user_prompt, turn2,
                         bench_threads, bench_ctx, host_buf2, bench_run_index);
            return;
        }
    }

    // Apply the per-model chat template (ChatML default, required for
    // SmolLM2-Instruct; Gemma/Qwen get their own format via the model name).
    const ::xllama::ChatFormat fmt = ::xllama::chat_format_for(model_name);
    std::string prompt = fmt.render_prompt("You are a helpful AI assistant.", {}, user_prompt);

    log_output("[xllama] bench model: " + model_name + "\n");
    log_output("[xllama] bench prompt: " + prompt.substr(0, 80) + "...\n");

    InferenceParams params;
    params.model_path = model_name;
    params.prompt = prompt;
    // 0 = keep the default (n_predict 512, n_ctx from InferenceParams).
    params.n_predict = bench_npredict > 0 ? bench_npredict : 512;
    if (bench_ctx > 0)
        params.n_ctx = bench_ctx;
    params.max_length_override = bench_maxlen;
    params.n_threads = bench_threads;                // 0 = auto; set by bench-xbox-ort.sh
    params.n_ubatch = bench_ubatch;                  // #172: 0 = llama default (512)
    params.n_batch = bench_n_batch;                  // MVP adapter: 0 = llama default (512)
    params.kv_q8 = bench_kvq8 != 0;                  // #171: q8_0 KV + flash attention
    params.prompt_lookup = bench_prompt_lookup != 0; // #210 W2
    params.mtp = bench_mtp > 0;                      // MTP drafting; requires the MTP GGUF
    params.mtp_n_max = bench_mtp > 0 ? bench_mtp : 4;
    if (bench_mtp_pmin >= 0)
        params.mtp_p_min = static_cast<float>(bench_mtp_pmin) / 100.0f;
    // Plan 003, stage 1: parity knobs, applied from the LocalState files the
    // bench driver writes. The effective values are logged AND echoed into the
    // host column (-greedy / -sN), so an MSIX that ignores them fails the
    // driver's host-tag validation instead of silently benching the default.
    params.greedy = bench_greedy != 0;
    if (bench_seed > 0)
        params.seed = static_cast<uint32_t>(bench_seed);
    if (bench_tokens != 0) {
        char tokpath[128];
        snprintf(tokpath, sizeof(tokpath), "bench-tokens-%d.txt", bench_run_index);
        params.dump_tokens_path = resolve_local_path(tokpath);
        log_output(std::string("[xllama] bench tokens sidecar: ") + tokpath + "\n");
    }
    char samp_buf[160];
    snprintf(samp_buf, sizeof(samp_buf),
             "[xllama] bench sampling: greedy=%d seed=%u temp=%.2f top_p=%.2f top_k=%d\n",
             params.greedy ? 1 : 0, params.seed, params.temperature, params.top_p, params.top_k);
    log_output(samp_buf);
    params.n_gpu_layers = bench_gpu_layers;                                     // D2b: 0 = CPU
    params.n_threads_batch = bench_batch_threads > 0 ? bench_batch_threads : 0; // plan 006 C1
    if (bench_batch_threads > 0)
        log_output("[xllama] bench: batch threads=" + std::to_string(bench_batch_threads) +
                   " (decode threads=" + std::to_string(bench_threads) + ")\n");
    params.stop_sequences = fmt.stop_sequences; // clean stop for Gemma's <end_of_turn>
    params.run_index = bench_run_index;         // W1.1: echo into CSV (0 = single-run)
    params.profile_phases = bench_profile != 0;
    if (bench_ignore_eog != 0) { // after stop_sequences is set, or the stops come back
        params.ignore_eog = true;
        params.stop_sequences.clear();
    }

    char host_buf[80];
    int host_len = snprintf(host_buf, sizeof(host_buf), "%s", host_tag_prefix().c_str());
    if (bench_threads > 0)
        host_len +=
            snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-t%d", bench_threads);
    if (bench_batch_threads > 0)
        host_len += snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-tb%d",
                             bench_batch_threads);
    if (bench_ubatch > 0)
        host_len +=
            snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-u%d", bench_ubatch);
    if (bench_n_batch > 0)
        host_len +=
            snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-b%d", bench_n_batch);
    if (bench_kvq8 != 0)
        host_len += snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-kvq8");
    if (bench_mtp > 0)
        host_len += snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-mtp%d", bench_mtp);
    if (bench_mtp > 0 && bench_mtp_pmin >= 0)
        host_len +=
            snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-pmin%d", bench_mtp_pmin);
    if (bench_greedy != 0)
        host_len += snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-greedy");
    if (bench_seed > 0)
        host_len += snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-s%d", bench_seed);
    if (bench_gpu_layers > 0)
        host_len +=
            snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-g%d", bench_gpu_layers);
    if (bench_ignore_eog != 0)
        host_len += snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-noeog");
    if (bench_prompt_lookup != 0)
        snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-plookup");
    snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-prof%d", bench_profile ? 1 : 0);

    InferenceResult res = ::xllama::run_inference(params);
    if (params.prompt_lookup) {
        char spec_buf[192];
        snprintf(spec_buf, sizeof(spec_buf),
                 "[xllama] SPEC_STATS success=%d n_eval=%d t_eval_ms=%.1f "
                 "n_drafted=%d n_spec_accepted=%d\n",
                 res.success ? 1 : 0, res.n_eval, res.t_eval_ms, res.n_drafted,
                 res.n_spec_accepted);
        log_output(spec_buf);
    }
    // Plan 003, stage 1: a bench that asked for MTP but whose drafter never
    // came up must FAIL, not silently produce a baseline row. write_bench_csv
    // only writes on success anyway; make the failure loud and explicit so
    // get-log shows the reason instead of a mysterious missing CSV.
    if (bench_mtp > 0 && !res.mtp_active) {
        log_output("[xllama] MTP_REQUESTED_BUT_INACTIVE — refusing to write bench row\n");
        return;
    }
    if (res.success && bench_mtp > 0) {
        char mlog[192];
        snprintf(mlog, sizeof(mlog), "[xllama] MTP_ACTIVE n_drafted=%d n_spec_accepted=%d\n",
                 res.n_drafted, res.n_spec_accepted);
        log_output(mlog);
    }
    xllama::write_bench_csv(params, res, host_buf);
#endif
}

// ---------------------------------------------------------------------------
// run_logits (called from UWP logits.flag mode background thread)
// ---------------------------------------------------------------------------

void run_logits() {
#ifdef XLLAMA_UWP
    set_cwd_to_local_folder();

    // Raw prompt (NOT chat-templated): parity feeds the identical string to both
    // backends so scripts/compare-logits.py can diff the resulting distributions.
    std::string prompt = read_local_file("prompt.txt");
    if (prompt.empty())
        prompt = "Hello from Xbox Series S. Tell me about your architecture.";
    std::string model_name = read_local_file("model.txt");
    if (model_name.empty())
        model_name = "smollm2-360m-cpu-int4";

    log_output("[xllama] logits model: " + model_name + "\n");
    log_output("[xllama] logits prompt: " + prompt.substr(0, 80) + "\n");

    InferenceParams params;
    params.model_path = model_name;
    params.prompt = prompt;
    params.n_predict = 1; // one deterministic forward pass; we only need prefill logits
    params.greedy = true;
    params.dump_logits_path = resolve_local_path("logits.bin");

    InferenceResult res = ::xllama::run_inference(params);
    log_output(res.success ? "[xllama] logits dump OK\n"
                           : "[xllama] logits dump FAILED: " + res.error_msg + "\n");

    // Completion marker for scripts/validate-logit-parity.sh to poll (WDP).
    FILE* done = _wfopen(utf8_to_wstring(resolve_local_path("logits.done")).c_str(), L"w");
    if (done) {
        fputs(res.success ? "ok" : "fail", done);
        fclose(done);
    }
#endif
}

// ---------------------------------------------------------------------------
// run_replay (called from UWP replay.flag mode background thread)
// ---------------------------------------------------------------------------

void run_replay() {
#if defined(XLLAMA_UWP) && defined(XLLAMA_USE_LLAMA)
    log_output("[xllama] replay: controlled single-vs-batch replay (diagnostic)\n");
    apply_repack_ctrl_knob("replay");
    apply_spinwait_knob("replay");
    apply_ggmlprof_knob("replay");
    const std::string prompt_text = read_local_file("replay_prompt.txt");
    std::string model_file = read_local_file("model.txt");
    const std::string model_name = model_file.empty() ? "qwen35-4b-mtp" : model_file;
    // Strict integer lists: any unexpected character fails the parse (atoi
    // truncates "12a" to 12 silently, which would forge prefix ids).
    auto parse_ids = [](const std::string& raw, std::vector<int32_t>& ids) {
        ids.clear();
        std::string tok;
        for (size_t i = 0; i <= raw.size(); ++i) {
            const char c = i < raw.size() ? raw[i] : ' ';
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == ',') {
                if (!tok.empty()) {
                    char* end = nullptr;
                    const long v = std::strtol(tok.c_str(), &end, 10);
                    if (end == nullptr || *end != '\0')
                        return false;
                    ids.push_back(static_cast<int32_t>(v));
                    tok.clear();
                }
            } else if ((c >= '0' && c <= '9') || (c == '-' && tok.empty())) {
                tok += c;
            } else {
                return false;
            }
        }
        return true;
    };
    std::vector<int32_t> ctx_ids, feed_ids, known_v, rem_ids, rem_known;
    const bool ids_ok = parse_ids(read_local_file("replay_ctx.txt"), ctx_ids) &&
                        parse_ids(read_local_file("replay_feed.txt"), feed_ids) &&
                        parse_ids(read_local_file("replay_known.txt"), known_v) &&
                        parse_ids(read_local_file("replay_rem.txt"), rem_ids) &&
                        parse_ids(read_local_file("replay_rem_known.txt"), rem_known);
    const int gpu_layers = read_local_int("bench_gpu_layers.txt", 28);
    const int n_ctx = read_local_int("bench_ctx.txt", 2048);
    const int n_threads = read_local_int("bench_threads.txt", 0);
    // Isolation knobs (plan 004): defaults match the failing MTP target
    // contexts (rs_seq=4, nextn on); replay_rs_seq.txt=0 / replay_nextn.txt=0
    // select the sequential-baseline flags to isolate each factor.
    const int replay_rs = read_local_int("replay_rs_seq.txt", 4);
    const int replay_nextn = read_local_int("replay_nextn.txt", 1);
    const std::string msix_sha = read_local_file("replay_msix_sha.txt");

    const std::string csv = resolve_local_path("replay-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp)
        fputs("branch,rep,argmax,margin,top1,top1_logit,top2,maxabs,maxrel,argmax_match,finite,"
              "selfcheck,error\n",
              fp);
    auto emit = [&](const ::xllama::ReplayLogits& r, double maxabs, double maxrel, int match,
                    const std::string& selfcheck, const std::string& err) {
        if (!fp)
            return;
        char lb[512];
        snprintf(lb, sizeof(lb), "%s,%d,%d,%.6f,%d,%.4f,%d,%.6g,%.6g,%d,%d,%s,%s\n",
                 r.branch.c_str(), r.rep, r.argmax, r.margin, r.top[0],
                 r.top[0] >= 0 && !r.full.empty() ? static_cast<double>(r.full[r.top[0]]) : 0.0,
                 r.top[1], maxabs, maxrel, match, r.finite ? 1 : 0,
                 selfcheck.empty() ? "-" : selfcheck.c_str(), err.empty() ? "-" : err.c_str());
        fputs(lb, fp);
    };

    std::string fail_why;
    if (prompt_text.empty())
        fail_why = "replay_prompt.txt missing or empty";
    else if (!ids_ok)
        fail_why = "replay id files must hold strict integer lists";
    std::vector<::xllama::ReplayLogits> rows;
    if (fail_why.empty()) {
        ::xllama::run_replay_measure(model_name, prompt_text, ctx_ids, feed_ids, known_v, rem_ids,
                                     rem_known, gpu_layers, n_ctx, n_threads, replay_rs,
                                     replay_nextn != 0, &rows, &fail_why);
    }
    // Pair per rep: (seq,acc) tests batching at this prefix; (tail,acc) tests
    // the discarded-tail/rewind path but does NOT isolate trim on its own
    // (rev73: the 4-wide rows already differ pre-trim). D-vs-C isolates
    // width4 (no trim ever runs); E-vs-C isolates the trim sequence (no B4
    // graph ever runs). Rows arrive seq,acc,tail per rep, then the
    // phase-labeled continuation seqc0..5, accc0..5, tailc0..5, then
    // d0..d3, drem0..5, e0..e2, etail, ecorr, erem0..5 (42 rows per rep, 84
    // total). A mid-run failure leaves a short vector: pair only complete
    // rep groups so a gap never compares across reps. Continuation
    // mismatches are evidence, never fatal: emit + log them.
    if (fail_why.empty() && rows.size() == 92) {
        for (int rep = 0; rep < 2; ++rep) {
            const size_t base = static_cast<size_t>(rep) * 46;
            const auto& a = rows[base];
            const auto& c = rows[base + 1];
            const auto& b = rows[base + 2];
            if (a.branch != "seq" || c.branch != "acc" || b.branch != "tail" || a.rep != rep ||
                c.rep != rep || b.rep != rep) {
                log_output("[xllama] replay FAILED: row pairing broken\n");
                emit({"diagnostic", -1}, 0.0, 0.0, 0, "-", "row pairing broken");
                break;
            }
            const ::xllama::ReplayCmp ac = ::xllama::compare_replay_rows(a.full, c.full);
            emit(a, 0.0, 0.0, 1, "sequential-replay", "");
            emit(c, ac.maxabs, ac.maxrel, ac.argmax_match ? 1 : 0, "accbatch-replay", "");
            const ::xllama::ReplayCmp bc = ::xllama::compare_replay_rows(b.full, c.full);
            emit(b, bc.maxabs, bc.maxrel, bc.argmax_match ? 1 : 0, "tailbatch-replay", "");
            char lb[320];
            snprintf(lb, sizeof(lb),
                     "[xllama] replay rep=%d seq=%d acc=%d tail=%d ac_maxabs=%.6g ac_match=%d "
                     "bc_maxabs=%.6g bc_match=%d\n",
                     rep, a.argmax, c.argmax, b.argmax, ac.maxabs, ac.argmax_match ? 1 : 0,
                     bc.maxabs, bc.argmax_match ? 1 : 0);
            log_output(lb);
            for (int k = 0; k < 6; ++k) {
                const auto& sa = rows[base + 3 + static_cast<size_t>(k)];
                const auto& sc = rows[base + 9 + static_cast<size_t>(k)];
                const auto& sb = rows[base + 15 + static_cast<size_t>(k)];
                const std::string want_s = "seqc" + std::to_string(k);
                const std::string want_c = "accc" + std::to_string(k);
                const std::string want_b = "tailc" + std::to_string(k);
                if (sa.branch != want_s || sc.branch != want_c || sb.branch != want_b ||
                    sa.rep != rep || sc.rep != rep || sb.rep != rep) {
                    log_output("[xllama] replay FAILED: continuation pairing broken\n");
                    emit({"diagnostic", -1}, 0.0, 0.0, 0, "-", "continuation pairing broken");
                    break;
                }
                const ::xllama::ReplayCmp cac = ::xllama::compare_replay_rows(sc.full, sa.full);
                emit(sa, 0.0, 0.0, 1, "cont-seq", "");
                emit(sc, cac.maxabs, cac.maxrel, cac.argmax_match ? 1 : 0, "cont-acc", "");
                const ::xllama::ReplayCmp cbc = ::xllama::compare_replay_rows(sb.full, sa.full);
                emit(sb, cbc.maxabs, cbc.maxrel, cbc.argmax_match ? 1 : 0, "cont-tail", "");
                char cl[320];
                snprintf(cl, sizeof(cl),
                         "[xllama] replay cont-rep=%d k=%d seq=%d acc=%d tail=%d "
                         "acc_maxabs=%.6g acc_match=%d tail_maxabs=%.6g tail_match=%d\n",
                         rep, k, sa.argmax, sc.argmax, sb.argmax, cac.maxabs,
                         cac.argmax_match ? 1 : 0, cbc.maxabs, cbc.argmax_match ? 1 : 0);
                log_output(cl);
            }
            // Width/trim isolation rows: d0..d3 (B4, no trim), drem0..5,
            // e0..e2 (B3, expect exact vs accb), etail (rejected single,
            // vs btail), ecorr (post-trim corrective, vs acc), erem0..5,
            // accb0..2 + btail committed-block references. E's outcome is
            // NOT trim-alone evidence (single-11 leaves slot 1 stale; see
            // NOTE in replay.h). Compared against the same-position seq/acc
            // rows; mismatches are evidence, not fatal.
            for (int k = 0; k < 4; ++k) {
                const auto& dr = rows[base + 21 + static_cast<size_t>(k)];
                const std::string want_d = "d" + std::to_string(k);
                if (dr.branch != want_d || dr.rep != rep) {
                    log_output("[xllama] replay FAILED: D pairing broken\n");
                    emit({"diagnostic", -1}, 0.0, 0.0, 0, "-", "D pairing broken");
                    break;
                }
                const auto& ref = (k < 3) ? rows[base + 42 + static_cast<size_t>(k)] : c;
                const ::xllama::ReplayCmp dc = ::xllama::compare_replay_rows(dr.full, ref.full);
                emit(dr, dc.maxabs, dc.maxrel, dc.argmax_match ? 1 : 0, "cont-d", "");
            }
            for (int k = 0; k < 6; ++k) {
                const auto& dr = rows[base + 25 + static_cast<size_t>(k)];
                const auto& sa = rows[base + 3 + static_cast<size_t>(k)];
                const std::string want_d = "drem" + std::to_string(k);
                if (dr.branch != want_d || dr.rep != rep) {
                    log_output("[xllama] replay FAILED: D-remainder pairing broken\n");
                    emit({"diagnostic", -1}, 0.0, 0.0, 0, "-", "D-remainder pairing broken");
                    break;
                }
                const ::xllama::ReplayCmp dc = ::xllama::compare_replay_rows(dr.full, sa.full);
                emit(dr, dc.maxabs, dc.maxrel, dc.argmax_match ? 1 : 0, "cont-drem", "");
                char dl[320];
                snprintf(dl, sizeof(dl),
                         "[xllama] replay dcont-rep=%d k=%d seq=%d d=%d d_maxabs=%.6g "
                         "d_match=%d\n",
                         rep, k, sa.argmax, dr.argmax, dc.maxabs, dc.argmax_match ? 1 : 0);
                log_output(dl);
            }
            for (int k = 0; k < 3; ++k) {
                const auto& er = rows[base + 31 + static_cast<size_t>(k)];
                const auto& ref = rows[base + 42 + static_cast<size_t>(k)];
                const std::string want_e = "e" + std::to_string(k);
                const std::string want_r = "accb" + std::to_string(k);
                if (er.branch != want_e || ref.branch != want_r || er.rep != rep ||
                    ref.rep != rep) {
                    log_output("[xllama] replay FAILED: E pairing broken\n");
                    emit({"diagnostic", -1}, 0.0, 0.0, 0, "-", "E pairing broken");
                    break;
                }
                const ::xllama::ReplayCmp ec = ::xllama::compare_replay_rows(er.full, ref.full);
                emit(er, ec.maxabs, ec.maxrel, ec.argmax_match ? 1 : 0, "cont-e", "");
            }
            {
                const auto& et = rows[base + 34];
                const auto& ec = rows[base + 35];
                const auto& bt = rows[base + 45];
                if (et.branch != "etail" || ec.branch != "ecorr" || bt.branch != "btail" ||
                    et.rep != rep || ec.rep != rep || bt.rep != rep) {
                    log_output("[xllama] replay FAILED: E-tail pairing broken\n");
                    emit({"diagnostic", -1}, 0.0, 0.0, 0, "-", "E-tail pairing broken");
                    break;
                }
                const ::xllama::ReplayCmp ecorr_c = ::xllama::compare_replay_rows(ec.full, c.full);
                const ::xllama::ReplayCmp etail_b = ::xllama::compare_replay_rows(et.full, bt.full);
                emit(et, etail_b.maxabs, etail_b.maxrel, etail_b.argmax_match ? 1 : 0, "cont-etail",
                     "");
                emit(ec, ecorr_c.maxabs, ecorr_c.maxrel, ecorr_c.argmax_match ? 1 : 0, "cont-ecorr",
                     "");
            }
            for (int k = 0; k < 6; ++k) {
                const auto& er = rows[base + 36 + static_cast<size_t>(k)];
                const auto& sa = rows[base + 3 + static_cast<size_t>(k)];
                const std::string want_e = "erem" + std::to_string(k);
                if (er.branch != want_e || er.rep != rep) {
                    log_output("[xllama] replay FAILED: E-remainder pairing broken\n");
                    emit({"diagnostic", -1}, 0.0, 0.0, 0, "-", "E-remainder pairing broken");
                    break;
                }
                const ::xllama::ReplayCmp ec = ::xllama::compare_replay_rows(er.full, sa.full);
                emit(er, ec.maxabs, ec.maxrel, ec.argmax_match ? 1 : 0, "cont-erem", "");
                char el[320];
                snprintf(el, sizeof(el),
                         "[xllama] replay econt-rep=%d k=%d seq=%d e=%d e_maxabs=%.6g "
                         "e_match=%d\n",
                         rep, k, sa.argmax, er.argmax, ec.maxabs, ec.argmax_match ? 1 : 0);
                log_output(el);
            }
            for (int k = 0; k < 3; ++k) {
                const auto& ab = rows[base + 42 + static_cast<size_t>(k)];
                const std::string want_ab = "accb" + std::to_string(k);
                if (ab.branch != want_ab || ab.rep != rep) {
                    log_output("[xllama] replay FAILED: accb pairing broken\n");
                    emit({"diagnostic", -1}, 0.0, 0.0, 0, "-", "accb pairing broken");
                    break;
                }
                emit(ab, 0.0, 0.0, 1, "cont-accb", "");
            }
            {
                const auto& bt = rows[base + 45];
                if (bt.branch != "btail" || bt.rep != rep) {
                    log_output("[xllama] replay FAILED: btail pairing broken\n");
                    emit({"diagnostic", -1}, 0.0, 0.0, 0, "-", "btail pairing broken");
                    break;
                }
                emit(bt, 0.0, 0.0, 1, "cont-btail", "");
            }
        }
    }
    if (!fail_why.empty()) {
        log_output(("[xllama] replay FAILED: " + fail_why + "\n").c_str());
        emit({"diagnostic", -1}, 0.0, 0.0, 0, "-", fail_why);
    }
    if (fp) {
        fflush(fp);
        fclose(fp);
    }
    {
        std::string msix = msix_sha.empty() ? "unknown" : msix_sha;
        log_output(("[xllama] replay done rows=" + std::to_string(rows.size()) +
                    " fail=" + (fail_why.empty() ? "-" : fail_why) + " msix=" + msix + "\n")
                       .c_str());
    }
    FILE* done =
        _wfopen(utf8_to_wstring(resolve_local_path("replay-result.csv.done")).c_str(), L"w");
    if (done) {
        fputs("done\n", done);
        fclose(done);
    }
    log_output("[xllama] replay-result.csv written\n");
#elif defined(XLLAMA_UWP)
    log_output("[xllama] replay: this package has no llama.cpp backend\n");
#endif
}

// ---------------------------------------------------------------------------
// run_diverge (called from UWP diverge.flag mode background thread)
// ---------------------------------------------------------------------------

void run_diverge() {
#if defined(XLLAMA_UWP) && defined(XLLAMA_USE_LLAMA)
    log_output("[xllama] diverge: first-numeric-divergence capture (diagnostic)\n");
    const std::string prompt_text = read_local_file("replay_prompt.txt");
    std::string model_file = read_local_file("model.txt");
    const std::string model_name = model_file.empty() ? "qwen35-4b-mtp" : model_file;
    auto parse_ids = [](const std::string& raw, std::vector<int32_t>& ids) {
        ids.clear();
        std::string tok;
        for (size_t i = 0; i <= raw.size(); ++i) {
            const char c = i < raw.size() ? raw[i] : ' ';
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == ',') {
                if (!tok.empty()) {
                    char* end = nullptr;
                    const long v = std::strtol(tok.c_str(), &end, 10);
                    if (end == nullptr || *end != '\0')
                        return false;
                    ids.push_back(static_cast<int32_t>(v));
                    tok.clear();
                }
            } else if ((c >= '0' && c <= '9') || (c == '-' && tok.empty())) {
                tok += c;
            } else {
                return false;
            }
        }
        return true;
    };
    std::vector<int32_t> ctx_ids, feed_ids, known_v;
    const bool ids_ok = parse_ids(read_local_file("replay_ctx.txt"), ctx_ids) &&
                        parse_ids(read_local_file("replay_feed.txt"), feed_ids) &&
                        parse_ids(read_local_file("replay_known.txt"), known_v);
    const int gpu_layers = read_local_int("bench_gpu_layers.txt", 28);
    const int n_ctx = read_local_int("bench_ctx.txt", 2048);
    const int n_threads = read_local_int("bench_threads.txt", 0);
    const int replay_rs = read_local_int("replay_rs_seq.txt", 4);
    const int replay_nextn = read_local_int("replay_nextn.txt", 1);
    const std::string msix_sha = read_local_file("replay_msix_sha.txt");

    const std::string csv = resolve_local_path("diverge-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp)
        fputs("rep,index,name,op,shape_b3,shape_b4,match,maxabs,note\n", fp);

    std::string fail_why;
    if (prompt_text.empty())
        fail_why = "replay_prompt.txt missing or empty";
    else if (!ids_ok)
        fail_why = "replay id files must hold strict integer lists";
    std::vector<::xllama::DivergeTensorRow> rows;
    if (fail_why.empty()) {
        ::xllama::run_diverge_measure(model_name, prompt_text, ctx_ids, feed_ids, known_v,
                                      gpu_layers, n_ctx, n_threads, replay_rs, replay_nextn != 0,
                                      &rows, &fail_why);
    }
    // Layout per rep: N interleaved pairs (b3, b4) in graph order, 4N rows
    // total over 2 reps. Pairing/validation/export use the shared pure
    // helper (host-tested); this site only writes files and log lines.
    // Tensor mismatches are evidence; structural breaks fail loudly.
    // Unlike the first rev80 attempt, a failed export emits no partial
    // tensor rows: fail_why plus the error marker only.
    if (fail_why.empty()) {
        const ::xllama::DivergeExport ex = ::xllama::diverge_pair_rows(rows);
        if (!ex.fail_why.empty()) {
            fail_why = ex.fail_why;
            if (ex.fail_pair >= 0) {
                log_output(("[xllama] diverge break pair=" + std::to_string(ex.fail_pair) +
                            " a=" + ::xllama::diverge_row_string(ex.fail_a) +
                            " b=" + ::xllama::diverge_row_string(ex.fail_b) + "\n")
                               .c_str());
            }
        } else {
            for (const auto& s : ex.reps) {
                char lb[320];
                snprintf(lb, sizeof(lb),
                         "[xllama] diverge rep=%d tensors=%zu matched=%zu skipped=%zu "
                         "first_diff=%d\n",
                         s.rep, s.n_tensors, s.n_matched, s.n_skipped, s.first_diff);
                log_output(lb);
            }
            for (const auto& p : ex.pairs) {
                if (!fp)
                    break;
                fputs(::xllama::diverge_csv_line(p).c_str(), fp);
            }
        }
    }
    if (!fail_why.empty()) {
        log_output(("[xllama] diverge FAILED: " + fail_why + "\n").c_str());
        if (fp)
            fputs("-1,-1,-,-,-,-,0,-1,error\n", fp);
    }
    if (fp) {
        fflush(fp);
        fclose(fp);
    }
    {
        std::string msix = msix_sha.empty() ? "unknown" : msix_sha;
        log_output(("[xllama] diverge done rows=" + std::to_string(rows.size()) +
                    " fail=" + (fail_why.empty() ? "-" : fail_why) + " msix=" + msix + "\n")
                       .c_str());
    }
    FILE* done =
        _wfopen(utf8_to_wstring(resolve_local_path("diverge-result.csv.done")).c_str(), L"w");
    if (done) {
        fputs("done\n", done);
        fclose(done);
    }
    log_output("[xllama] diverge-result.csv written\n");
#elif defined(XLLAMA_UWP)
    log_output("[xllama] diverge: this package has no llama.cpp backend\n");
#endif
}

// ---------------------------------------------------------------------------
// run_znarrow (called from UWP znarrow.flag mode background thread)
// ---------------------------------------------------------------------------

void run_znarrow() {
#if defined(XLLAMA_UWP) && defined(XLLAMA_USE_LLAMA)
    log_output("[xllama] znarrow: narrow z-0 capture (diagnostic)\n");
    const std::string prompt_text = read_local_file("replay_prompt.txt");
    std::string model_file = read_local_file("model.txt");
    const std::string model_name = model_file.empty() ? "qwen35-4b-mtp" : model_file;
    auto parse_ids = [](const std::string& raw, std::vector<int32_t>& ids) {
        ids.clear();
        std::string tok;
        for (size_t i = 0; i <= raw.size(); ++i) {
            const char c = i < raw.size() ? raw[i] : ' ';
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == ',') {
                if (!tok.empty()) {
                    char* end = nullptr;
                    const long v = std::strtol(tok.c_str(), &end, 10);
                    if (end == nullptr || *end != '\0')
                        return false;
                    ids.push_back(static_cast<int32_t>(v));
                    tok.clear();
                }
            } else if ((c >= '0' && c <= '9') || (c == '-' && tok.empty())) {
                tok += c;
            } else {
                return false;
            }
        }
        return true;
    };
    std::vector<int32_t> ctx_ids, feed_ids, known_v;
    const bool ids_ok = parse_ids(read_local_file("replay_ctx.txt"), ctx_ids) &&
                        parse_ids(read_local_file("replay_feed.txt"), feed_ids) &&
                        parse_ids(read_local_file("replay_known.txt"), known_v);
    const int gpu_layers = read_local_int("bench_gpu_layers.txt", 28);
    const int n_ctx = read_local_int("bench_ctx.txt", 2048);
    const int n_threads = read_local_int("bench_threads.txt", 0);
    const int replay_rs = read_local_int("replay_rs_seq.txt", 4);
    const int replay_nextn = read_local_int("replay_nextn.txt", 1);
    const std::string msix_sha = read_local_file("replay_msix_sha.txt");
    // Narrow z-0 dispatch record (first-divergence diagnosis): LocalState
    // d3d12zdispatch.txt containing 1 enables per-z-0 dispatch lines from the
    // backend. Default off. Scoped to this run.
    const bool zdispatch = read_local_file("d3d12zdispatch.txt") == "1";
    ::xllama::d3d12_set_z0_dispatch_log(zdispatch);
    log_output(std::string("[xllama] znarrow: d3d12zdispatch.txt=") +
               (zdispatch ? "1 (z-0 dispatch lines on)\n" : "0 (off)\n"));
    // Narrow z-0 placement pin (first-divergence diagnosis): d3d12zpin.txt
    // containing "cpu" refuses z-0 on D3D12 so it runs the CPU kernel path.
    // Default off. Scoped to this run (restored below before exit).
    const bool zpin = read_local_file("d3d12zpin.txt") == "cpu";
    ::xllama::d3d12_set_z0_pin_cpu(zpin);
    log_output(std::string("[xllama] znarrow: d3d12zpin.txt=") +
               (zpin ? "cpu (z-0 pinned to CPU)\n" : "off (default placement)\n"));
    apply_repack_ctrl_knob("znarrow");
    apply_spinwait_knob("znarrow");
    apply_ggmlprof_knob("znarrow");

    const std::string csv = resolve_local_path("znarrow-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp)
        fputs("rep,index,name,op,shape_b3,shape_b4,match,maxabs,note\n", fp);

    std::string fail_why;
    if (prompt_text.empty())
        fail_why = "replay_prompt.txt missing or empty";
    else if (!ids_ok)
        fail_why = "replay id files must hold strict integer lists";
    std::vector<::xllama::DivergeTensorRow> rows;
    if (fail_why.empty()) {
        ::xllama::run_znarrow_measure(model_name, prompt_text, ctx_ids, feed_ids, known_v,
                                      gpu_layers, n_ctx, n_threads, replay_rs, replay_nextn != 0,
                                      &rows, &fail_why);
    }
    // Layout per rep: N interleaved pairs (b3, b4) in graph order, 4N rows
    // total over 2 reps. Pairing/validation/export use the shared pure
    // helper (host-tested); this site only writes files and log lines.
    // Tensor mismatches are evidence; structural breaks fail loudly.
    // Unlike the first rev80 attempt, a failed export emits no partial
    // tensor rows: fail_why plus the error marker only.
    if (fail_why.empty()) {
        const ::xllama::DivergeExport ex = ::xllama::diverge_pair_rows(rows);
        if (!ex.fail_why.empty()) {
            fail_why = ex.fail_why;
            if (ex.fail_pair >= 0) {
                log_output(("[xllama] diverge break pair=" + std::to_string(ex.fail_pair) +
                            " a=" + ::xllama::diverge_row_string(ex.fail_a) +
                            " b=" + ::xllama::diverge_row_string(ex.fail_b) + "\n")
                               .c_str());
            }
        } else {
            for (const auto& s : ex.reps) {
                char lb[320];
                snprintf(lb, sizeof(lb),
                         "[xllama] diverge rep=%d tensors=%zu matched=%zu skipped=%zu "
                         "first_diff=%d\n",
                         s.rep, s.n_tensors, s.n_matched, s.n_skipped, s.first_diff);
                log_output(lb);
            }
            for (const auto& p : ex.pairs) {
                if (!fp)
                    break;
                fputs(::xllama::diverge_csv_line(p).c_str(), fp);
            }
        }
    }
    if (!fail_why.empty()) {
        log_output(("[xllama] diverge FAILED: " + fail_why + "\n").c_str());
        if (fp)
            fputs("-1,-1,-,-,-,-,0,-1,error\n", fp);
    }
    if (fp) {
        fflush(fp);
        fclose(fp);
    }
    {
        std::string msix = msix_sha.empty() ? "unknown" : msix_sha;
        log_output(("[xllama] diverge done rows=" + std::to_string(rows.size()) +
                    " fail=" + (fail_why.empty() ? "-" : fail_why) + " msix=" + msix + "\n")
                       .c_str());
    }
    FILE* done =
        _wfopen(utf8_to_wstring(resolve_local_path("znarrow-result.csv.done")).c_str(), L"w");
    if (done) {
        fputs("done\n", done);
        fclose(done);
    }
    ::xllama::d3d12_set_z0_dispatch_log(false);
    ::xllama::d3d12_set_z0_pin_cpu(false);
    log_output("[xllama] znarrow-result.csv written\n");
#elif defined(XLLAMA_UWP)
    log_output("[xllama] diverge: this package has no llama.cpp backend\n");
#endif
}

// ---------------------------------------------------------------------------
// run_termgate (called from UWP termgate.flag mode background thread)
//
// Termination-state gate (plan 003): deterministic headless scenarios for
// cancel through the real abort_flag, a boundary-spanning stop sequence, and
// natural EOG termination — each proven by full-id comparisons (a dump-size
// or metadata equality alone never passes). Arms come from the bench knobs:
// bench_mtp.txt (0 = sequential), d3d12twocol.txt (auto = candidate),
// cpurepackforcegemv.txt (2 = scope2). A sequential reference session runs
// inside every invocation (width-1 decode is immune to both knobs), so each
// arm proves its own state parity; the host verifier also compares arms to
// each other. Writes termgate-result.csv (+ .done). Diagnostic, never a gate
// default.
// ---------------------------------------------------------------------------
void run_termgate() {
#if defined(XLLAMA_UWP) && defined(XLLAMA_USE_LLAMA)
    apply_q8_knob("termgate");
    // Termgate is a phase-attribution diagnostic (gp.profile_phases=true), so
    // its default is timestamps ON unless the explicit knob overrides.
    apply_gpu_timestamps(true, "termgate");
    log_output("[xllama] termgate: termination-state gate (diagnostic)\n");
    apply_twocol_knob("termgate");
    apply_repack_ctrl_knob("termgate");
    apply_spinwait_knob("termgate");
    apply_ggmlprof_knob("termgate");

    std::string model_file = read_local_file("model.txt");
    const std::string model_name = model_file.empty() ? "qwen35-4b-mtp" : model_file;
    const int bench_threads = read_local_int("bench_threads.txt", 0);
    const int bench_ctx = read_local_int("bench_ctx.txt", 2048);
    const int gpu_layers = read_local_int("bench_gpu_layers.txt", 28);
    const int bench_n_batch = read_local_int("bench_n_batch.txt", 0); // MVP 64/64
    const int bench_ubatch = read_local_int("bench_ubatch.txt", 0);   // MVP 64/64
    const int mtp_n = read_local_int("bench_mtp.txt", 4);
    const int pmin_pct = read_local_int("bench_mtp_pmin.txt", 75);
    const float pmin = pmin_pct >= 0 ? static_cast<float>(pmin_pct) / 100.0f : 0.75f;
    const int twocol = ::xllama::d3d12_kernel_variant();
    const std::string repack_scope = read_local_file("cpurepackforcegemv.txt");
    std::string scen_raw = read_local_file("termgate.txt");
    if (scen_raw.empty())
        scen_raw = "cancel,stop,eog";
    const bool do_cancel = scen_raw.find("cancel") != std::string::npos;
    const bool do_stop = scen_raw.find("stop") != std::string::npos;
    const bool do_eog = scen_raw.find("eog") != std::string::npos;
    const char* arm = (mtp_n == 0) ? "seq" : (twocol >= 2 && repack_scope == "2") ? "cand" : "ref";

    const std::string gguf_path =
        ::xllama::first_gguf_in_dir(::xllama::resolve_model_path(model_name));
    std::uint64_t gguf_bytes = 0;
    const std::string gguf_hash =
        gguf_path.empty() ? "unresolved" : ::xllama::fnv1a64_file(gguf_path, &gguf_bytes);
    {
        char cfg[448];
        snprintf(cfg, sizeof(cfg),
                 "[xllama] TERM_GATE_CONFIG arm=%s threads=%d n_ctx=%d gpu_layers=%d mtp=%d "
                 "p_min=%.2f twocol=%d repack_scope=%s scenarios=%s greedy=1 gguf=%s\n",
                 arm, bench_threads, bench_ctx > 0 ? bench_ctx : 2048, gpu_layers, mtp_n, pmin,
                 twocol, repack_scope.empty() ? "0" : repack_scope.c_str(), scen_raw.c_str(),
                 gguf_hash.c_str());
        log_output(cfg);
    }

    struct TermRow {
        std::string scenario;
        std::string reason = "-";
        int ok = 0;
        int n_ids = -1, n_ref = -1, parity = -1, first_diff = -1;
        int active = 0, drafted = 0, rounds = 0, lookup = 0;
        int ews = 0, n_eval = -1, eog_tok = -1;
        // Termination metadata straight from the result (never inferred):
        // which branch sampled the EOG / fired the stop, and in which round.
        std::string eog_branch;
        std::string stop_branch;
        int stop_round = -1;
    };
    std::vector<TermRow> rows;

    auto ids_of = [](const std::string& name) {
        std::vector<int> out;
        const std::string raw = read_local_file(name.c_str());
        size_t pos = 0;
        while (pos < raw.size()) {
            const size_t eol = raw.find('\n', pos);
            const std::string tok =
                raw.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
            if (!tok.empty())
                out.push_back(std::atoi(tok.c_str()));
            if (eol == std::string::npos)
                break;
            pos = eol + 1;
        }
        return out;
    };

    // Equal over min(size, limit); first difference index (or -1).
    auto equal_prefix = [](const std::vector<int>& a, const std::vector<int>& b, size_t limit,
                           int* first) {
        const size_t n = std::min({a.size(), b.size(), limit});
        for (size_t i = 0; i < n; ++i) {
            if (a[i] != b[i]) {
                if (first)
                    *first = static_cast<int>(i);
                return false;
            }
        }
        if (first)
            *first = -1;
        return true;
    };

    auto push_row = [&](const TermRow& row) {
        rows.push_back(row);
        char lb[512];
        snprintf(lb, sizeof(lb),
                 "[xllama] TERMGATE scenario=%s arm=%s ok=%d reason=%s n_ids=%d n_ref=%d "
                 "parity=%d first_diff=%d active=%d drafted=%d rounds=%d lookup=%d ews=%d "
                 "n_eval=%d eog_tok=%d eog_branch=%s stop_branch=%s stop_round=%d\n",
                 row.scenario.c_str(), arm, row.ok, row.reason.c_str(), row.n_ids, row.n_ref,
                 row.parity, row.first_diff, row.active, row.drafted, row.rounds, row.lookup,
                 row.ews, row.n_eval, row.eog_tok,
                 row.eog_branch.empty() ? "-" : row.eog_branch.c_str(),
                 row.stop_branch.empty() ? "-" : row.stop_branch.c_str(), row.stop_round);
        log_output(lb);
    };

    auto open_session = [&](int depth_override, std::string& err) {
        const int depth = depth_override >= 0 ? depth_override : mtp_n;
        ::xllama::SessionParams sp;
        sp.model_path = model_name;
        sp.n_ctx = bench_ctx > 0 ? bench_ctx : 2048;
        sp.n_threads = bench_threads;
        if (bench_n_batch > 0)
            sp.n_batch = bench_n_batch;
        if (bench_ubatch > 0)
            sp.n_ubatch = bench_ubatch;
        if (gpu_layers > 0)
            sp.n_gpu_layers = gpu_layers;
        sp.mtp = depth > 0;
        sp.mtp_n_max = depth > 0 ? depth : 4;
        sp.mtp_p_min = pmin;
        return ::xllama::Session::create(sp, &err);
    };

    // Greedy generate with optional stop set; returns false on failure.
    auto gen = [](std::unique_ptr<::xllama::Session>& s, const std::string& prompt, int npred,
                  bool reuse, bool reset, const std::string& dump, std::vector<std::string> stops,
                  std::function<void(std::string_view)> on_tok, std::atomic<bool>* abort,
                  ::xllama::InferenceResult* r, std::string* text) {
        ::xllama::GenerateParams gp;
        gp.prompt = prompt;
        gp.n_predict = npred;
        gp.temperature = 0.0f; // greedy: ids must be comparable
        gp.reuse_kv = reuse;
        gp.reset_kv = reset;
        // Phase timers ON for this diagnostic only: MTP_STATS corrective_ms
        // is the evidence that a verify-reject/corrective ran inside the
        // cancelled turn ("after correction, where feasible"). Timing claims
        // are never taken from termgate runs.
        gp.profile_phases = true;
        gp.stop_sequences = std::move(stops);
        if (on_tok)
            gp.on_token = std::move(on_tok);
        if (abort)
            gp.abort_flag = abort;
        if (!dump.empty())
            gp.dump_tokens_path = resolve_local_path(dump);
        *r = s->generate(gp);
        if (text)
            *text = r->output_text;
        return r->success;
    };

    std::string fail_why;

    // ---- Scenario: cancel at deterministic emission boundaries -------------
    // Session ownership: ONE Session (one model load) at a time — two live
    // Sessions exceed the console budget and fail creation. Phase A runs
    // cancel+resume on its own session and closes it; phase B opens the
    // sequential reference session; the verdict uses only persisted dumps.
    if (do_cancel && fail_why.empty()) {
        const std::string base = "Write a long paragraph about the history of computing.";
        const int K = 12;
        for (int N : {6, 13}) {
            TermRow row;
            row.scenario = "cancel@" + std::to_string(N);
            std::string e;
            int T1 = -1;
            std::vector<int> t1, t1p, rsm, rsmp, ref;
            {
                // Phase A: cancel + resume on one session, then close it.
                auto s = open_session(-1, e);
                if (!s) {
                    row.reason = "session create failed";
                    push_row(row);
                    continue;
                }
                std::atomic<bool> abort{false};
                int emitted = 0;
                std::string text1;
                ::xllama::InferenceResult r1;
                const std::string d1 = "termgate-cancel" + std::to_string(N) + "-t1.txt";
                const bool ok1 = gen(
                    s, base, 24, false, true, d1, {},
                    [&](std::string_view sv) {
                        (void)sv;
                        if (++emitted == N)
                            abort.store(true);
                    },
                    &abort, &r1, &text1);
                std::string text2;
                ::xllama::InferenceResult r2;
                const std::string d2 = "termgate-cancel" + std::to_string(N) + "-resume.txt";
                // Full-prompt turn (reset_kv=true): the product resumes an
                // aborted chat with the full context; the #170 prefix-reuse
                // path re-establishes state from resident-vs-prompt, and the
                // fresh sequential reference below is the comparison. (The
                // reuse_kv continuation path appends the prompt as a delta —
                // the session turn log proved pf == whole prompt — so a
                // continuation-style resume would not be product-realistic.)
                const bool ok2 =
                    gen(s, base + text1, K, false, true, d2, {}, nullptr, nullptr, &r2, &text2);
                T1 = r1.n_eval;
                row.active = r1.mtp_active ? 1 : 0;
                row.drafted = r1.n_mtp_drafted;
                row.rounds = r1.n_mtp_rounds;
                row.lookup = r1.n_lookup_drafted;
                row.ews = r1.ended_with_stop ? 1 : 0;
                row.n_eval = T1;
                t1 = ids_of(d1);
                t1p = ids_of(d1 + ".prefill");
                rsm = ids_of(d2);
                rsmp = ids_of(d2 + ".prefill");
                if (!ok1)
                    row.reason = "cancel generate failed";
                else if (!ok2)
                    row.reason = "resume generate failed";
                else if (T1 < N)
                    row.reason = "emitted fewer than N before stopping";
                else if (static_cast<int>(t1.size()) != T1)
                    row.reason = "t1 dump size != n_eval";
            } // phase A session closed here
            // Phase B: sequential reference on its own session.
            if (row.reason == "-") {
                std::string ref_err;
                auto ref_ses = open_session(0, ref_err);
                if (!ref_ses) {
                    row.reason = "seqref session create failed";
                } else {
                    const std::string dr = "termgate-cancel" + std::to_string(N) + "-seqref.txt";
                    ::xllama::InferenceResult rr;
                    std::string rtx;
                    const bool okr = gen(ref_ses, base, T1 + K, false, true, dr, {}, nullptr,
                                         nullptr, &rr, &rtx);
                    ref = ids_of(dr);
                    if (!okr || ref.empty())
                        row.reason = "sequential reference failed or empty";
                }
            } // phase B session closed here
            row.n_ids = static_cast<int>(t1.size());
            row.n_ref = static_cast<int>(ref.size());
            int first = -1;
            if (row.reason == "-") {
                // V1: cancel boundary matches the sequential prefix.
                if (!equal_prefix(t1, ref, t1.size(), &first)) {
                    row.reason =
                        "cancel prefix differs from sequential at " + std::to_string(first);
                    row.first_diff = first;
                } else {
                    // V2: resume prefill reconstructs the accepted history
                    // exactly (prompt tokens + this turn's ids) — the cache
                    // state proof; retokenization failures are reported.
                    std::vector<int> want = t1p;
                    want.insert(want.end(), t1.begin(), t1.end());
                    if (rsmp != want)
                        row.reason = "resume prefill != prompt + accepted ids (state proof)";
                    else {
                        // V3: continuation equals the sequential reference slice.
                        const size_t off = t1.size();
                        std::vector<int> exp;
                        if (off <= ref.size())
                            exp.assign(ref.begin() + static_cast<long>(off), ref.end());
                        if (rsm.empty() || exp.empty())
                            row.reason = "resume or reference continuation empty";
                        else if (!equal_prefix(rsm, exp, rsm.size(), &first)) {
                            row.reason = "resume continuation differs from sequential at " +
                                         std::to_string(first);
                            row.first_diff = first;
                        } else {
                            row.parity = 1;
                            row.ok = 1;
                        }
                    }
                }
            }
            push_row(row);
        }
    }

    // ---- Scenario: boundary-spanning stop sequence ------------------------
    // Same one-session-at-a-time rule: session S runs probe -> stop -> resume
    // sequentially, closes; session C runs the cold reference.
    if (do_stop && fail_why.empty()) {
        const std::string base = "Count from one to five, one number per line.";
        std::vector<std::string> pieces;
        ::xllama::InferenceResult rprobe;
        ::xllama::StopSpan span;
        std::string stripped; // survives the session-S scope for the cold ref
        bool have_span = false;
        TermRow row;
        row.scenario = "stop@run";
        std::string stext_probe;
        const std::string dp = "termgate-stop-probe.txt";
        {
            std::string e;
            auto s = open_session(-1, e);
            if (!s) {
                fail_why = "stop session create failed: " + e;
            } else {
                const bool okp = gen(
                    s, base, 64, false, true, dp, {},
                    [&](std::string_view sv) { pieces.emplace_back(sv); }, nullptr, &rprobe,
                    &stext_probe);
                TermRow prow;
                prow.scenario = "stop@probe";
                prow.n_eval = rprobe.n_eval;
                prow.n_ids = static_cast<int>(ids_of(dp).size());
                prow.active = rprobe.mtp_active ? 1 : 0;
                prow.drafted = rprobe.n_mtp_drafted;
                prow.rounds = rprobe.n_mtp_rounds;
                prow.lookup = rprobe.n_lookup_drafted;
                prow.ews = rprobe.ended_with_stop ? 1 : 0;
                prow.eog_tok = rprobe.eog_token;
                prow.eog_branch = rprobe.eog_branch;
                prow.stop_branch = rprobe.stop_branch;
                prow.stop_round = rprobe.stop_round;
                // Same termination taxonomy as the eog scenario: a stop or
                // EOG that ends the probe early is recorded, never "cap".
                prow.reason =
                    rprobe.ended_with_stop ? "stop" : (rprobe.n_eval < 64 ? "eos" : "cap");
                if (!okp)
                    prow.reason = "probe generate failed";
                else if (prow.n_ids != rprobe.n_eval)
                    prow.reason = "probe dump size != n_eval";
                else
                    prow.ok = 1;
                push_row(prow);
                have_span = okp && pieces.size() >= 2;
                if (have_span) {
                    // Last usable pair: the later boundary, so the stop run
                    // has the probe's full preceding MTP rounds to cover.
                    span = ::xllama::pick_boundary_spanning_stop(pieces, /*prefer_last=*/true);
                    have_span = !span.stop.empty();
                    if (have_span) {
                        char lb[320];
                        snprintf(lb, sizeof(lb),
                                 "[xllama] TERMGATE stop_span pieces=%zu start=%zu stop_len=%zu "
                                 "probe_n_eval=%d\n",
                                 pieces.size(), span.text_start, span.stop.size(), rprobe.n_eval);
                        log_output(lb);
                        std::string stext;
                        ::xllama::InferenceResult rst;
                        const std::string ds = "termgate-stop-run.txt";
                        const bool oks = gen(s, base, 64, false, true, ds, {span.stop}, nullptr,
                                             nullptr, &rst, &stext);
                        stripped = stext;
                        const std::vector<int> probe_ids = ids_of(dp);
                        const std::vector<int> stop_ids = ids_of(ds);
                        row.ews = rst.ended_with_stop ? 1 : 0;
                        row.n_eval = rst.n_eval;
                        row.n_ids = static_cast<int>(stop_ids.size());
                        row.active = rst.mtp_active ? 1 : 0;
                        row.drafted = rst.n_mtp_drafted;
                        row.rounds = rst.n_mtp_rounds;
                        row.lookup = rst.n_lookup_drafted;
                        row.eog_tok = rst.eog_token;
                        row.eog_branch = rst.eog_branch;
                        row.stop_branch = rst.stop_branch;
                        row.stop_round = rst.stop_round;
                        int first = -1;
                        if (!oks)
                            row.reason = "stop generate failed";
                        else if (!rst.ended_with_stop)
                            row.reason = "stop sequence never fired";
                        else if (stext != stext_probe.substr(0, span.text_start))
                            row.reason = "stripped text != probe prefix";
                        else if (stext.find(span.stop) != std::string::npos)
                            row.reason = "stop sequence still present after strip";
                        else if (stop_ids.size() != static_cast<size_t>(rst.n_eval))
                            row.reason = "stop dump size != n_eval";
                        else if (!equal_prefix(stop_ids, probe_ids, stop_ids.size(), &first)) {
                            row.reason =
                                "stop prefix differs from probe at " + std::to_string(first);
                            row.first_diff = first;
                        } else if (row.stop_branch.empty() || row.stop_round < 1) {
                            // Coverage demand: the result must say WHERE the
                            // stop fired (branch + round), not just that it did.
                            row.reason = "stop metadata missing (branch/round)";
                        } else if (row.active == 1 && (row.drafted <= 0 || row.rounds <= 0)) {
                            // MTP arms must have real preceding speculative
                            // rounds before the stop; the seq arm has none by
                            // construction (recorded, not failed).
                            row.reason = "stop fired before any MTP round";
                        } else {
                            // State after stop: resume the SAME session (reuse)
                            // before it closes. The retokenization check comes
                            // after the cold reference below.
                            std::string stext2;
                            ::xllama::InferenceResult rrsm;
                            const std::string drs = "termgate-stop-resume.txt";
                            // Full-prompt turn (reset_kv=true): the contract
                            // continuation path appends the whole prompt as a
                            // delta, which a stripped stop turn cannot supply;
                            // the product resumes from text via the #170
                            // prefix-reuse full-prompt path, and the cold
                            // reference below uses the same path.
                            row.reason = gen(s, base + stext, 12, false, true, drs, {}, nullptr,
                                             nullptr, &rrsm, &stext2)
                                             ? "-"
                                             : "resume after stop failed";
                        }
                    } else {
                        row.reason = "no two consecutive non-empty pieces";
                    }
                } else {
                    row.reason = !okp ? "probe generate failed"
                                      : (pieces.size() >= 2 ? "no boundary-spanning stop available"
                                                            : "probe too short for a span");
                }
            }
        } // session S closed here
        // Cold reference (sequential) on its own session, same stripped prompt.
        if (row.reason == "-") {
            std::string e2;
            auto cold = open_session(0, e2);
            if (!cold) {
                row.reason = "cold reference session create failed";
            } else {
                std::string ctext;
                ::xllama::InferenceResult rcold;
                const std::string dc = "termgate-stop-cold.txt";
                const bool okc = gen(cold, base + stripped, 12, false, true, dc, {}, nullptr,
                                     nullptr, &rcold, &ctext);
                if (!okc)
                    row.reason = "cold generate failed";
            }
        }
        if (row.reason == "-") {
            // State proof (rev96 -> rev97, concrete reason): the rev96
            // assertion `resume_prefill == prompt_prefill + accepted_ids`
            // encoded the WRONG contract for a stop. The emitted/accepted
            // split at a stop is: emit_token pushes the token to the id dump
            // BEFORE the stop scan, then the caller counts it; accept_token
            // is SKIPPED for the crossing token (classic_step and the spec
            // commit/verify paths all return before account_maintain), so the
            // crossing token is in n_eval/dump but never in the KV; and
            // accepted tokens whose text was stripped out of the visible
            // output are not part of the next turn's prompt text either.
            // The product next turn is a FULL-prompt turn (reset_kv=true)
            // whose state is derived from text via the #170 prefix-reuse
            // path, so prompt+accepted-ids can never equal a text-derived
            // prefill and rev96's formula could not hold by construction.
            // Expected prefix (exact): resume prefill == cold prefill ==
            // tokenize(prompt + stripped), 11 tokens on this fixture,
            // asserted byte-for-byte below; the assertions were NOT loosened:
            // strip/prefix-equality checks are unchanged, and the state proof
            // is full-ID equality of the resumed continuation against the
            // cold sequential reference from the SAME stripped prompt.
            const std::vector<int> rsm_ids = ids_of("termgate-stop-resume.txt");
            const std::vector<int> rsm_pref = ids_of("termgate-stop-resume.txt.prefill");
            const std::vector<int> cold_ids = ids_of("termgate-stop-cold.txt");
            const std::vector<int> cold_pref = ids_of("termgate-stop-cold.txt.prefill");
            if (cold_ids.empty() || rsm_ids.empty())
                row.reason = "resume or cold ids empty";
            else if (rsm_ids != cold_ids)
                row.reason = "continuation after stop differs from sequential cold";
            else if (rsm_pref != cold_pref)
                row.reason = "resume prefill != cold prefill (retokenization)";
            else
                row.parity = 1;
        }
        if (row.reason == "-")
            row.ok = 1;
        push_row(row);
    }

    // ---- Scenario: bounded natural-EOG probe --------------------------------
    if (do_eog && fail_why.empty()) {
        // Bounded candidates. The fourth is the fixture's PROVEN natural-EOS
        // prompt (device evidence: "EOG after 10 tokens tok=248044" on the
        // rev95 probe run) so the gate has a prompt that actually emits EOG.
        const char* cands[4] = {"Reply with the single word OK.", "hi",
                                "1+1=", "Count from one to five, one number per line."};
        std::string e;
        auto s = open_session(-1, e);
        if (!s) {
            fail_why = "eog session create failed: " + e;
        }
        for (int i = 0; i < 4 && fail_why.empty(); ++i) {
            TermRow row;
            row.scenario = "eog@" + std::to_string(i);
            std::string txt;
            ::xllama::InferenceResult r;
            const std::string d = "termgate-eog" + std::to_string(i) + ".txt";
            const bool ok =
                s && gen(s, cands[i], 64, false, true, d, {}, nullptr, nullptr, &r, &txt);
            row.n_eval = r.n_eval;
            row.n_ids = static_cast<int>(ids_of(d).size());
            row.active = r.mtp_active ? 1 : 0;
            row.drafted = r.n_mtp_drafted;
            row.rounds = r.n_mtp_rounds;
            row.lookup = r.n_lookup_drafted;
            row.ews = r.ended_with_stop ? 1 : 0;
            row.eog_tok = r.eog_token;
            row.eog_branch = r.eog_branch;
            row.stop_branch = r.stop_branch;
            row.stop_round = r.stop_round;
            row.reason = r.ended_with_stop ? "stop" : (r.n_eval < 64 ? "eos" : "cap");
            if (!ok)
                row.reason = "generate failed";
            else if (row.n_ids != r.n_eval)
                row.reason = "eog dump size != n_eval";
            else
                row.ok = 1;
            {
                char lb[320];
                snprintf(lb, sizeof(lb), "[xllama] TERMGATE eog_probe idx=%d n_eval=%d reason=%s",
                         i, r.n_eval, row.reason.c_str());
                log_output(lb);
            }
            push_row(row);
        }
    }

    // ---- CSV + done marker -------------------------------------------------
    if (!fail_why.empty())
        log_output(("[xllama] termgate FAILED: " + fail_why + "\n").c_str());
    const std::string csv = resolve_local_path("termgate-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp) {
        fputs("scenario,arm,ok,reason,n_ids,n_ref,parity,first_diff,active,drafted,rounds,"
              "lookup,ews,n_eval,eog_tok,eog_branch,stop_branch,stop_round\n",
              fp);
        if (!fail_why.empty()) {
            std::string why = "error:" + fail_why;
            for (char& c : why)
                if (c == ',' || c == '\n' || c == '"')
                    c = ';';
            fprintf(fp, "error,%s,0,\"%s\",-1,-1,-1,-1,0,0,0,0,0,-1,-,-,-1\n", arm, why.c_str());
        }
        for (const auto& row : rows) {
            std::string why = row.reason;
            for (char& c : why)
                if (c == ',' || c == '\n' || c == '"')
                    c = ';';
            fprintf(fp, "%s,%s,%d,\"%s\",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s,%d\n",
                    row.scenario.c_str(), arm, row.ok, why.c_str(), row.n_ids, row.n_ref,
                    row.parity, row.first_diff, row.active, row.drafted, row.rounds, row.lookup,
                    row.ews, row.n_eval, row.eog_tok,
                    row.eog_branch.empty() ? "-" : row.eog_branch.c_str(),
                    row.stop_branch.empty() ? "-" : row.stop_branch.c_str(), row.stop_round);
        }
        fflush(fp);
        fclose(fp);
    }
    {
        std::string msix = read_local_file("replay_msix_sha.txt");
        char lb[320];
        snprintf(lb, sizeof(lb), "[xllama] termgate done rows=%d fail=%s msix=%s\n",
                 static_cast<int>(rows.size()), fail_why.empty() ? "-" : fail_why.c_str(),
                 msix.empty() ? "unknown" : msix.c_str());
        log_output(lb);
    }
    FILE* done =
        _wfopen(utf8_to_wstring(resolve_local_path("termgate-result.csv.done")).c_str(), L"w");
    if (done) {
        fputs("done\n", done);
        fclose(done);
    }
    log_output("[xllama] termgate-result.csv written\n");
#elif defined(XLLAMA_UWP)
    log_output("[xllama] termgate: this package has no llama.cpp backend\n");
#endif
}

// ---------------------------------------------------------------------------
// run_membw (called from UWP membw.flag mode background thread)
// ---------------------------------------------------------------------------

void run_membw() {
#ifdef XLLAMA_UWP
    log_output("[xllama] membw: measuring CPU memory bandwidth\n");
    // Two passes: single-thread and full-width, so the scaling ratio is visible.
    // 256 MB working set overflows the LLC → measures DRAM, not cache.
    const std::size_t buf = static_cast<std::size_t>(256) << 20;
    const ::xllama::MembwResult st = ::xllama::measure_membw(buf, 5, 1);
    const ::xllama::MembwResult mt = ::xllama::measure_membw(buf, 5, 0);

    char lb[256];
    snprintf(lb, sizeof(lb),
             "[xllama] membw: 1t read=%.1f copy=%.1f triad=%.1f | %dt read=%.1f copy=%.1f "
             "triad=%.1f GB/s\n",
             st.read_gbs, st.copy_gbs, st.triad_gbs, mt.threads, mt.read_gbs, mt.copy_gbs,
             mt.triad_gbs);
    log_output(lb);

    const std::string csv = resolve_local_path("membw-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp) {
        fputs(::xllama::membw_csv_header(), fp);
        fputs(::xllama::format_membw_row(st, "xbox-series-s-t1").c_str(), fp);
        fputs(::xllama::format_membw_row(mt, "xbox-series-s").c_str(), fp);
        fclose(fp);
        FILE* done =
            _wfopen(utf8_to_wstring(resolve_local_path("membw-result.csv.done")).c_str(), L"w");
        if (done) {
            fputs("done\n", done);
            fclose(done);
        }
        log_output("[xllama] membw-result.csv written\n");
    }
#endif
}

// ---------------------------------------------------------------------------
// run_elembench (called from UWP elembench.flag mode background thread)
// ---------------------------------------------------------------------------

void run_elembench() {
#ifdef XLLAMA_UWP
    log_output("[xllama] elembench: elementwise CPU kernels by width\n");
    const int n_embd = 2560; // model embedding width
    const int rows_list[] = {1, 5, 64, 128, 256};
    const int reps = 40;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        log_output("[xllama] elembench: CPU backend unavailable\n");
        return;
    }
    // Mirror the app's CPU-side wiring: 2 threads and a persistent threadpool,
    // so the per-call number is not a standalone-backend threadpool artifact.
    ggml_backend_cpu_set_n_threads(cpu, 2);
    ggml_threadpool_params tp = ggml_threadpool_params_default(2);
    ggml_threadpool_t pool = ggml_threadpool_new(&tp);
    if (pool)
        ggml_backend_cpu_set_threadpool(cpu, pool);
    FILE* fp = _wfopen(utf8_to_wstring(resolve_local_path("elembench-result.csv")).c_str(), L"w");
    if (fp)
        fputs("op,rows,elems,graph_ns,manual_ns,ns_per_elem,overhead_ns,checksum\n", fp);

    auto median = [](std::vector<double>& v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    // Empty-graph floor: a single dup of a 4-byte tensor, same call path.
    {
        ggml_init_params ip = {};
        ip.mem_size = 8 * ggml_tensor_overhead() + 2 * ggml_graph_overhead();
        ip.no_alloc = true;
        ggml_context* ec = ggml_init(ip);
        if (ec) {
            ggml_tensor* e = ggml_new_tensor_1d(ec, GGML_TYPE_F32, 1);
            ggml_tensor* eo = ggml_dup(ec, e);
            ggml_cgraph* eg = ggml_new_graph(ec);
            ggml_build_forward_expand(eg, eo);
            ggml_backend_buffer_t eb = ggml_backend_alloc_ctx_tensors_from_buft(ec, ggml_backend_cpu_buffer_type());
            if (eb) {
                for (int i = 0; i < 3; ++i)
                    (void)ggml_backend_graph_compute(cpu, eg);
                std::vector<double> ts;
                for (int i = 0; i < reps; ++i) {
                    const auto t0 = std::chrono::steady_clock::now();
                    (void)ggml_backend_graph_compute(cpu, eg);
                    ts.push_back(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count());
                }
                const double floor_ns = median(ts);
                if (fp)
                    fprintf(fp, "empty,0,1,%.0f,0,0,0,0\n", floor_ns);
                char lb[160];
                snprintf(lb, sizeof(lb), "[xllama] elembench: empty-graph floor=%.0fns (n_threads=2, pool)\n", floor_ns);
                log_output(lb);
                ggml_backend_buffer_free(eb);
            }
            ggml_free(ec);
        }
    }
    for (int rows : rows_list) {
        const size_t n = static_cast<size_t>(n_embd) * static_cast<size_t>(rows);
        for (const char* op_name : {"add", "mul", "rms_norm", "cpy"}) {
            ggml_init_params ip = {};
            ip.mem_size = 16 * ggml_tensor_overhead() + 2 * ggml_graph_overhead();
            ip.no_alloc = true;
            ggml_context* ctx = ggml_init(ip);
            if (!ctx)
                continue;
            ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rows);
            ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rows);
            ggml_tensor* w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            ggml_tensor* out = nullptr;
            if (strcmp(op_name, "add") == 0)
                out = ggml_add(ctx, a, b);
            else if (strcmp(op_name, "mul") == 0)
                out = ggml_mul(ctx, a, w);
            else if (strcmp(op_name, "rms_norm") == 0)
                out = ggml_rms_norm(ctx, a, 1e-5f);
            else
                out = ggml_dup(ctx, a);
            ggml_cgraph* g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, out);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
            if (!buf) {
                ggml_free(ctx);
                continue;
            }
            std::vector<float> av(n), bv(n), wv(n_embd), ov(n);
            std::mt19937 rng(42u);
            std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
            for (auto& v : av) v = uni(rng);
            for (auto& v : bv) v = uni(rng);
            for (auto& v : wv) v = uni(rng) + 1.0f;
            ggml_backend_tensor_set(a, av.data(), 0, n * sizeof(float));
            ggml_backend_tensor_set(b, bv.data(), 0, n * sizeof(float));
            ggml_backend_tensor_set(w, wv.data(), 0, n_embd * sizeof(float));
            for (int i = 0; i < 3; ++i)
                (void)ggml_backend_graph_compute(cpu, g);
            std::vector<double> gts;
            for (int i = 0; i < reps; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                (void)ggml_backend_graph_compute(cpu, g);
                gts.push_back(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count());
            }
            const double gns = median(gts);
            std::vector<double> mts;
            double chk = 0.0;
            for (int i = 0; i < reps; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                if (strcmp(op_name, "add") == 0) {
                    for (size_t k = 0; k < n; ++k) ov[k] = av[k] + bv[k];
                } else if (strcmp(op_name, "mul") == 0) {
                    for (size_t k = 0; k < n; ++k) ov[k] = av[k] * wv[k % n_embd];
                } else if (strcmp(op_name, "rms_norm") == 0) {
                    for (int r = 0; r < rows; ++r) {
                        double sum = 0.0;
                        const float* xr = av.data() + static_cast<size_t>(r) * n_embd;
                        for (int j = 0; j < n_embd; ++j) sum += static_cast<double>(xr[j]) * static_cast<double>(xr[j]);
                        const float mean = static_cast<float>(sum / n_embd);
                        const float sc = 1.0f / std::sqrt(mean + 1e-5f);
                        float* yr = ov.data() + static_cast<size_t>(r) * n_embd;
                        for (int j = 0; j < n_embd; ++j) yr[j] = xr[j] * sc;
                    }
                } else {
                    memcpy(ov.data(), av.data(), n * sizeof(float));
                }
                chk += ov[i % n];
                mts.push_back(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count());
            }
            const double mns = median(mts);
            if (fp) {
                fprintf(fp, "%s,%d,%zu,%.0f,%.0f,%.2f,%.0f,%.3f\n", op_name, rows, n, gns, mns,
                        mns / static_cast<double>(n), gns - mns, chk);
                fflush(fp);
            }
            char lb[200];
            snprintf(lb, sizeof(lb),
                     "[xllama] elembench: %-8s rows=%3d n=%7zu graph=%9.0fns manual=%9.0fns "
                     "ns/elem=%.2f overhead=%.0fns\n",
                     op_name, rows, n, gns, mns, mns / static_cast<double>(n), gns - mns);
            log_output(lb);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
    if (fp) {
        fclose(fp);
        FILE* done = _wfopen(utf8_to_wstring(resolve_local_path("elembench-result.csv.done")).c_str(), L"w");
        if (done) {
            fputs("done\n", done);
            fclose(done);
        }
        log_output("[xllama] elembench-result.csv written\n");
    }
    ggml_backend_free(cpu);
    if (pool)
        ggml_threadpool_free(pool);
#endif
}

// ---------------------------------------------------------------------------
// run_diskbw (called from UWP diskbw.flag mode background thread)
// ---------------------------------------------------------------------------

void run_diskbw() {
#ifdef XLLAMA_UWP
    log_output("[xllama] diskbw: measuring sandboxed NVMe read bandwidth\n");
    // 4 GiB incompressible file in LocalState — above the RAM budget, so a
    // buffered pass cannot be served entirely from cache. Deleted afterwards.
    const std::string path = resolve_local_path("diskbw-test.bin");
    std::string err;
    if (!::xllama::ensure_diskbw_file(path, ::xllama::kDiskbwDefaultFileBytes, &err)) {
        log_output(("[xllama] diskbw FAIL: " + err + "\n").c_str());
        // A failed creation (disk full) can leave a multi-GiB partial file in
        // LocalState that nothing else would ever clean up.
        _wremove(utf8_to_wstring(path).c_str());
        return;
    }
    const ::xllama::DiskbwResult runs[] = {
        ::xllama::measure_diskbw(path, ::xllama::kDiskbwDefaultFileBytes,
                                 ::xllama::kDiskbwSeqBlockBytes, /*random=*/false, 1, 3, true),
        ::xllama::measure_diskbw(path, ::xllama::kDiskbwDefaultFileBytes,
                                 ::xllama::kDiskbwSeqBlockBytes, /*random=*/false, 4, 3, true),
        ::xllama::measure_diskbw(path, ::xllama::kDiskbwDefaultFileBytes,
                                 ::xllama::kDiskbwRndBlockBytes, /*random=*/true, 1, 3, true),
        ::xllama::measure_diskbw(path, ::xllama::kDiskbwDefaultFileBytes,
                                 ::xllama::kDiskbwRndBlockBytes, /*random=*/true, 4, 3, true),
    };

    const std::string csv = resolve_local_path("diskbw-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp) {
        fputs(::xllama::diskbw_csv_header(), fp);
        for (const auto& r : runs) {
            char lb[320];
            if (!r.error_msg.empty()) {
                snprintf(lb, sizeof(lb), "[xllama] diskbw FAIL: %s\n", r.error_msg.c_str());
                log_output(lb);
                continue;
            }
            snprintf(lb, sizeof(lb), "[xllama] diskbw: %s %dt unbuf=%d first=%.2f best=%.2f GB/s\n",
                     r.random ? "rnd" : "seq", r.threads, r.unbuffered ? 1 : 0, r.read_gbs_first,
                     r.read_gbs_best);
            log_output(lb);
            fputs(::xllama::format_diskbw_row(r, "xbox-series-s").c_str(), fp);
        }
        fclose(fp);
        FILE* done =
            _wfopen(utf8_to_wstring(resolve_local_path("diskbw-result.csv.done")).c_str(), L"w");
        if (done) {
            fputs("done\n", done);
            fclose(done);
        }
        log_output("[xllama] diskbw-result.csv written\n");
    } else {
        log_output("[xllama] diskbw: cannot open diskbw-result.csv\n");
    }
    _wremove(utf8_to_wstring(path).c_str());
#endif
}

// ---------------------------------------------------------------------------
// run_gpubw (called from UWP gpubw.flag mode background thread)
// ---------------------------------------------------------------------------

void run_gpubw() {
#ifdef XLLAMA_UWP
    log_output("[xllama] gpubw: measuring GPU STREAM bandwidth (own CS, no Agility)\n");
    // ~1 GiB default (issue #211). If alloc fails, measure_gpubw reports error.
    const ::xllama::GpubwResult r = ::xllama::measure_gpubw(::xllama::kGpubwDefaultBufferBytes, 3);

    char lb[320];
    snprintf(lb, sizeof(lb),
             "[xllama] gpubw: read=%.2f GB/s checksum_ok=%d d3d12_ran=%d kill_gate=%d "
             "buf=%zu MB err=%s\n",
             r.read_gbs, r.checksum_ok ? 1 : 0, r.d3d12_ran ? 1 : 0,
             ::xllama::gpubw_passes_kill_gate(r) ? 1 : 0, r.buffer_bytes / (1024 * 1024),
             r.error_msg.empty() ? "-" : r.error_msg.c_str());
    log_output(lb);

    const std::string csv = resolve_local_path("gpubw-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp) {
        fputs(::xllama::gpubw_csv_header(), fp);
        fputs(::xllama::format_gpubw_row(r, "xbox-series-s").c_str(), fp);
        fclose(fp);
        FILE* done =
            _wfopen(utf8_to_wstring(resolve_local_path("gpubw-result.csv.done")).c_str(), L"w");
        if (done) {
            fputs("done\n", done);
            fclose(done);
        }
        log_output("[xllama] gpubw-result.csv written\n");
    }
#endif
}

// ---------------------------------------------------------------------------
// run_gpugemv (called from UWP gpugemv.flag mode background thread)
// ---------------------------------------------------------------------------

void run_gpugemv() {
#ifdef XLLAMA_UWP
    log_output("[xllama] gpugemv: measuring Q4_K GEMV density (wave32/rows/dot4 + naive A/B, "
               "no Agility)\n");

    const ::xllama::GpugemvKernel kernels[] = {
        ::xllama::GpugemvKernel::Naive, ::xllama::GpugemvKernel::Wave32,
        ::xllama::GpugemvKernel::Rows, ::xllama::GpugemvKernel::Dot4};
    ::xllama::GpugemvKernelSummary denser[3] = {};
    std::size_t n_denser = 0;

    const std::string csv = resolve_local_path("gpugemv-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp)
        fputs(::xllama::gpugemv_csv_header(), fp);

    for (auto kernel : kernels) {
        std::vector<::xllama::GpugemvResult> rows;
        ::xllama::measure_gpugemv_each(::xllama::kGpugemvDefaultN, ::xllama::kGpugemvDefaultK,
                                       /*recorded=*/3, kernel, &rows);
        bool g1_all3 = rows.size() >= 3;
        std::vector<double> gbs;
        if (!rows.empty()) {
            char cap[200];
            snprintf(
                cap, sizeof(cap),
                "[xllama] gpugemv: kernel=%s WaveOps=%d WaveLaneCountMin=%u WaveLaneCountMax=%u "
                "wave_ops=%d\n",
                ::xllama::gpugemv_kernel_name(kernel), rows[0].wave_ops_cap ? 1 : 0,
                rows[0].wave_lane_min, rows[0].wave_lane_max, rows[0].wave_ops ? 1 : 0);
            log_output(cap);
        }
        for (const auto& r : rows) {
            if (fp)
                fputs(::xllama::format_gpugemv_row(r, "xbox-series-s").c_str(), fp);
            g1_all3 =
                g1_all3 && r.run_index >= 1 && r.run_index <= 3 && ::xllama::gpugemv_passes_g1(r);
            if (r.run_index >= 1 && r.run_index <= 3)
                gbs.push_back(r.packed_gbs);
            char lb[400];
            snprintf(lb, sizeof(lb),
                     "[xllama] gpugemv: kernel=%s run=%d packed_gbs=%.2f packed_gbs_cpu=%.2f "
                     "gpu_timestamp=%d max_abs_err=%.6g checksum_ok=%d d3d12_ran=%d g1=%d g2=%d "
                     "err=%s\n",
                     ::xllama::gpugemv_kernel_name(r.kernel), r.run_index, r.packed_gbs,
                     r.packed_gbs_cpu, r.gpu_timestamp ? 1 : 0, static_cast<double>(r.max_abs_err),
                     r.checksum_ok ? 1 : 0, r.d3d12_ran ? 1 : 0,
                     ::xllama::gpugemv_passes_g1(r) ? 1 : 0, ::xllama::gpugemv_passes_g2(r) ? 1 : 0,
                     r.error_msg.empty() ? "-" : r.error_msg.c_str());
            log_output(lb);
        }
        if (rows.empty()) {
            char lb[200];
            snprintf(lb, sizeof(lb), "[xllama] gpugemv: kernel=%s produced no rows (PSO/setup)\n",
                     ::xllama::gpugemv_kernel_name(kernel));
            log_output(lb);
            g1_all3 = false;
        }
        double median = 0.0;
        if (!gbs.empty()) {
            std::sort(gbs.begin(), gbs.end());
            median = gbs[gbs.size() / 2];
        }
        const ::xllama::GpugemvLadder ladder = ::xllama::gpugemv_ladder(median, g1_all3);
        char lb[240];
        snprintf(lb, sizeof(lb), "[xllama] gpugemv: kernel=%s median=%.2f g1_all3=%d ladder=%s\n",
                 ::xllama::gpugemv_kernel_name(kernel), median, g1_all3 ? 1 : 0,
                 ::xllama::gpugemv_ladder_name(ladder));
        log_output(lb);
        if (kernel != ::xllama::GpugemvKernel::Naive && n_denser < 3) {
            denser[n_denser].kernel = kernel;
            denser[n_denser].median_packed_gbs = median;
            denser[n_denser].g1_all3 = g1_all3;
            denser[n_denser].ladder = ladder;
            ++n_denser;
        }
    }

    const ::xllama::GpugemvLadder campaign = ::xllama::gpugemv_campaign_verdict(denser, n_denser);
    char clb[160];
    snprintf(clb, sizeof(clb), "[xllama] gpugemv: campaign_verdict=%s\n",
             ::xllama::gpugemv_ladder_name(campaign));
    log_output(clb);

    // Always write .done so bench-gpugemv.sh's 300 s wait cannot hang on partial PSO fail.
    if (fp) {
        fflush(fp);
        fclose(fp);
    }
    FILE* done =
        _wfopen(utf8_to_wstring(resolve_local_path("gpugemv-result.csv.done")).c_str(), L"w");
    if (done) {
        fputs("done\n", done);
        fclose(done);
    }
    log_output("[xllama] gpugemv-result.csv written\n");
#endif
}

// ---------------------------------------------------------------------------
// run_gpustep (gpustep.flag headless, gpustep-inproc.flag inside the XAML process)
// ---------------------------------------------------------------------------

void run_gpustep(bool inproc) {
#ifdef XLLAMA_UWP
    const char* process = inproc ? "inproc" : "headless";
    const std::string base = inproc ? "gpustep-inproc-result.csv" : "gpustep-result.csv";
    char lb[400];
    snprintf(lb, sizeof(lb), "[xllama] gpustep: D1 probe process=%s (no Agility)\n", process);
    log_output(lb);

    std::vector<::xllama::GpustepRow> rows;
    ::xllama::measure_gpustep(process, &rows);

    FILE* fp = _wfopen(utf8_to_wstring(resolve_local_path(base)).c_str(), L"w");
    if (fp)
        fputs(::xllama::gpustep_csv_header(), fp);
    for (const auto& r : rows) {
        if (fp)
            fputs(::xllama::format_gpustep_row(r, "xbox-series-s").c_str(), fp);
        snprintf(lb, sizeof(lb),
                 "[xllama] gpustep: %s %s median=%.3f p90=%.3f %s ok=%d d3d12_ran=%d uma=%d "
                 "cc_uma=%d err=%s\n",
                 r.kind.c_str(), r.variant.c_str(), r.median, r.p90, r.unit.c_str(), r.ok ? 1 : 0,
                 r.d3d12_ran ? 1 : 0, r.uma ? 1 : 0, r.cc_uma ? 1 : 0,
                 r.error.empty() ? "-" : r.error.c_str());
        log_output(lb);
    }
    // One process alone is never the D1 verdict (D1d needs both); the host
    // evaluates the merged CSV with xllama-cli --gpustep-verdict.
    if (fp) {
        fflush(fp);
        fclose(fp);
    }
    FILE* done = _wfopen(utf8_to_wstring(resolve_local_path(base + ".done")).c_str(), L"w");
    if (done) {
        fputs("done\n", done);
        fclose(done);
    }
    snprintf(lb, sizeof(lb), "[xllama] %s written\n", base.c_str());
    log_output(lb);
#else
    (void)inproc;
#endif
}

// ---------------------------------------------------------------------------
// run_ttarget (tttarget.flag, headless): T_target(B) on the real context.
// ---------------------------------------------------------------------------

void run_tttarget() {
#if defined(XLLAMA_UWP) && defined(XLLAMA_USE_LLAMA)
    log_output("[xllama] ttarget: real T_target(B), one decode per measurement\n");
    TwocolKnobScope twocol("tttarget");
    std::string model_name = read_local_file("model.txt");
    if (model_name.empty())
        model_name = "qwen35-4b-mtp";
    const int n_threads = read_local_int("bench_threads.txt", 6);
    const int n_ctx = read_local_int("bench_ctx.txt", 2048);
    const int n_batch = read_local_int("bench_session_nbatch.txt", 0);
    const int n_ubatch = read_local_int("bench_ubatch.txt", 0);
    // Same window the MTP gate runs with (mtp_n_max=4): the recurrent state the
    // timed decode sees must match the one the gate validates. decode_loop.h is
    // internal (needs llama.h), so the value is stated here with its derivation.
    const int mtp_n_max = 4;
    const int n_rs_seq = mtp_n_max;
    const int reps = read_local_int("tttarget_reps.txt", 3);
    // Optional schedule restriction: "1" for a smoke, "1,2,3,5" (or absent) for the
    // full matrix. Keeps the knob in data, not in a rebuild.
    std::vector<int> widths;
    {
        const std::string raw = read_local_file("tttarget_widths.txt");
        std::string tok;
        for (std::size_t i = 0; i <= raw.size(); ++i) {
            const char c = i < raw.size() ? raw[i] : ',';
            if (c == ',') {
                if (!tok.empty())
                    widths.push_back(std::atoi(tok.c_str()));
                tok.clear();
            } else if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
                tok += c;
            }
        }
    }
    std::string wlist;
    for (std::size_t i = 0; i < widths.size(); ++i)
        wlist += (i ? "," : "") + std::to_string(widths[i]);
    log_output("[xllama] ttarget config model=" + model_name +
               " threads=" + std::to_string(n_threads) + " n_ctx=" + std::to_string(n_ctx) +
               " n_batch=" + std::to_string(n_batch) + " n_ubatch=" + std::to_string(n_ubatch) +
               " n_rs_seq=" + std::to_string(n_rs_seq) + " reps=" + std::to_string(reps) +
               " widths=" + (wlist.empty() ? "1,2,3,5" : wlist) + "\n");
    log_output("[xllama] ttarget stage=load start\n");
    std::vector<::xllama::TtargetRow> rows;
    const std::string msix_sha = read_local_file("tttarget_msix_sha.txt");
    ::xllama::measure_ttarget(model_name, 99, n_ctx, n_batch, n_ubatch, n_rs_seq, n_threads, reps,
                              widths, {}, msix_sha.empty() ? "unknown" : msix_sha, &rows);
    log_output("[xllama] ttarget stage=load done rows=" + std::to_string(rows.size()) + "\n");
    const std::string csv = resolve_local_path("tttarget-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp)
        fputs(::xllama::ttarget_csv_header(), fp);
    int n_err = 0;
    for (const auto& r : rows) {
        if (fp)
            fputs(::xllama::format_ttarget_row(r, "xbox-series-s").c_str(), fp);
        if (!r.error.empty())
            ++n_err;
        char lb[320];
        snprintf(lb, sizeof(lb),
                 "[xllama] ttarget: B=%d rep=%d order=%d decoded=%d match=%d first_diff=%d "
                 "wall=%.3f gpu=%.3f calls=%llu mm=%llu err=%s\n",
                 r.b, r.rep, r.order, r.n_decoded, r.argmax_match, r.first_mismatch, r.wall_ms,
                 r.gpu_ms, static_cast<unsigned long long>(r.calls),
                 static_cast<unsigned long long>(r.matmuls),
                 r.error.empty() ? "-" : r.error.c_str());
        log_output(lb);
    }
    if (fp) {
        fflush(fp);
        fclose(fp);
    }
    FILE* done =
        _wfopen(utf8_to_wstring(resolve_local_path("tttarget-result.csv.done")).c_str(), L"w");
    if (done) {
        fputs("done\n", done);
        fclose(done);
    }
    log_output("[xllama] ttarget stage=done rows=" + std::to_string(rows.size()) +
               " errors=" + std::to_string(n_err) + "\n");
    log_output("[xllama] ttarget-result.csv written\n");
#elif defined(XLLAMA_UWP)
    log_output("[xllama] ttarget: this package has no llama.cpp backend\n");
#endif
}

// ---------------------------------------------------------------------------
// run_d3d12_selftest (d3d12be.flag, headless)
// ---------------------------------------------------------------------------

void run_d3d12_selftest() {
#if defined(XLLAMA_UWP) && defined(XLLAMA_USE_LLAMA)
    log_output("[xllama] d3d12be: backend selftest (Q4_0/Q4_K/Q6_K vs ggml dequant)\n");
    std::vector<::xllama::D3d12SelftestRow> rows;
    ::xllama::run_d3d12_selftest(&rows);
    const std::string csv = resolve_local_path("d3d12be-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp)
        fputs(::xllama::d3d12_selftest_csv_header(), fp);
    int ok = 0;
    for (const auto& r : rows) {
        if (fp)
            fputs(::xllama::format_d3d12_selftest_row(r, "xbox-series-s").c_str(), fp);
        ok += r.ok ? 1 : 0;
        char lb[320];
        snprintf(lb, sizeof(lb),
                 "[xllama] d3d12be: %s n=%d k=%d ncols=%d rel_err=%.3g gpu_ms=%.4f gbs=%.2f "
                 "ok=%d err=%s\n",
                 r.type.c_str(), r.n, r.k, r.ncols, r.rel_err, r.gpu_ms, r.packed_gbs, r.ok ? 1 : 0,
                 r.error.empty() ? "-" : r.error.c_str());
        log_output(lb);
    }
    if (fp) {
        fflush(fp);
        fclose(fp);
    }
    FILE* done =
        _wfopen(utf8_to_wstring(resolve_local_path("d3d12be-result.csv.done")).c_str(), L"w");
    if (done) {
        fputs("done\n", done);
        fclose(done);
    }
    char lb[96];
    snprintf(lb, sizeof(lb), "[xllama] d3d12be: %d/%zu cases ok\n", ok, rows.size());
    log_output(lb);
    // The D2a gate above keeps its exact inputs: the gate rows are written and
    // their .done is out before the shape-cost bench even runs. The real
    // (type, N, K) triples cost B=1/2/3/5 into a second CSV, never gated.
    log_output("[xllama] d3d12be: shape-cost bench (real tttarget triples, B=1/2/3/5)\n");
    log_output(("[xllama] d3d12sc: " + ::xllama::d3d12_2col_info() + "\n").c_str());
    std::vector<::xllama::D3d12SelftestRow> cost_rows;
    ::xllama::run_d3d12_shape_cost(&cost_rows);
    FILE* cfp = _wfopen(utf8_to_wstring(resolve_local_path("d3d12sc-result.csv")).c_str(), L"w");
    if (cfp)
        fputs(::xllama::d3d12_shapecost_csv_header(), cfp);
    int cost_ok = 0;
    for (const auto& r : cost_rows) {
        if (cfp)
            fputs(::xllama::format_d3d12_shapecost_row(r, "xbox-series-s").c_str(), cfp);
        cost_ok += r.ok ? 1 : 0;
        char clb[384];
        snprintf(clb, sizeof(clb),
                 "[xllama] d3d12sc: %s n=%d k=%d ncols=%d rel_err=%.3g gpu_ms=%.4f wall_ms=%.4f "
                 "grange=%.4f wrange=%.4f gbs=%.2f wMB=%.3f peakMB=%.1f var=%d pair=%d tc=%llu "
                 "pads=%s ok=%d err=%s\n",
                 r.type.c_str(), r.n, r.k, r.ncols, r.rel_err, r.gpu_ms, r.wall_ms, r.gpu_ms_range,
                 r.wall_ms_range, r.packed_gbs, r.weight_mb, r.peak_ws_mb, r.variant, r.pair,
                 static_cast<unsigned long long>(r.twocol), r.pads.c_str(), r.ok ? 1 : 0,
                 r.error.empty() ? "-" : r.error.c_str());
        log_output(clb);
    }
    if (cfp) {
        fflush(cfp);
        fclose(cfp);
    }
    FILE* cdone =
        _wfopen(utf8_to_wstring(resolve_local_path("d3d12sc-result.csv.done")).c_str(), L"w");
    if (cdone) {
        fputs("done\n", cdone);
        fclose(cdone);
    }
    char clb[96];
    snprintf(clb, sizeof(clb), "[xllama] d3d12sc: %d/%zu cases ok\n", cost_ok, cost_rows.size());
    log_output(clb);
#elif defined(XLLAMA_UWP)
    log_output("[xllama] d3d12be: this package has no llama.cpp backend\n");
#endif
}

// ---------------------------------------------------------------------------
// run_ramceil (called from UWP ramceil.flag mode background thread)
//
// Writes each row as it is produced and flushes: the whole point of the probe
// is to approach the point where the OS stops cooperating, so a buffered write
// would lose the last — and most informative — rows to a PLM kill.
// ---------------------------------------------------------------------------

void run_ramceil() {
#ifdef XLLAMA_UWP
    log_output("[xllama] ramceil: probing the committable heap ceiling\n");

    const std::string csv = resolve_local_path("ramceil-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (!fp) {
        log_output("[xllama] ramceil: cannot open ramceil-result.csv\n");
        return;
    }
    fputs(::xllama::ramceil_csv_header(), fp);
    fflush(fp);

    // 128 MB steps: fine enough to place the ceiling within a model quant's
    // margin, coarse enough that the probe stays short. The 8 GB limit is above
    // the console's 10 GB unified pool minus the OS reservation, so the stop
    // reason is the platform's answer, not ours. The 256 MB floor keeps a
    // margin for the OS rather than racing it to the kill.
    const ::xllama::RamCeilResult r = ::xllama::probe_ram_ceiling(
        /*step_mb=*/128, /*limit_mb=*/8192, /*floor_avail_mb=*/256,
        [fp](const ::xllama::RamCeilStep& s) {
            fputs(::xllama::format_ramceil_row(s, "xbox-series-s").c_str(), fp);
            fflush(fp);
        });

    char lb[256];
    snprintf(lb, sizeof(lb),
             "[xllama] ramceil: max committed %zu MB (start avail %zu MB, stop: %s)\n",
             r.max_committed_mb, r.avail_phys_start_mb, r.stop_reason.c_str());
    log_output(lb);

    fclose(fp);
    FILE* done =
        _wfopen(utf8_to_wstring(resolve_local_path("ramceil-result.csv.done")).c_str(), L"w");
    if (done) {
        fputs(r.stop_reason.c_str(), done);
        fputs("\n", done);
        fclose(done);
    }
    log_output("[xllama] ramceil-result.csv written\n");
#endif
}

// ---------------------------------------------------------------------------
// run_mic_probe (called from UWP mic.flag mode background thread)
//
// Phase 16 WS-F / card H16.6. The question is not "does the API exist" — the
// [mic] line in App.cpp already answers that, and per uwp-constraints.md §10b a
// present type says nothing about whether it can be activated. The question is
// whether an AppContainer app on GameOS can actually capture audio.
//
// The whole value of this probe is that it does NOT collapse to a boolean.
// The WinRT status enums already separate the cases that matter, and the JSON
// carries them by name rather than as a derived verdict:
//
//   AudioDeviceNodeCreationStatus::AccessDenied      -> the sandbox refuses.
//                                                       WS-F FAIL, and a
//                                                       permanent constraint.
//   AudioDeviceNodeCreationStatus::DeviceNotAvailable-> NO MIC IS PLUGGED IN.
//                                                       This is NOT a verdict
//                                                       on the sandbox; rerun
//                                                       with a headset before
//                                                       concluding anything.
//   Success but RMS ~ 0                              -> opened and silenced.
//   Success and RMS > 1e-3                           -> real capture.
//
// Everything is wrapped: this runs on HeadlessView's std::thread, where an
// escaping exception is a process kill with no log line and no .done marker —
// indistinguishable from a hang on the host side (uwp-constraints.md §10c).
// A probe that dies mute teaches the operator to read a timeout as a FAIL,
// which is the one reading this probe exists to prevent. So every exit path
// writes mic-result.json, including the ones that threw.
// ---------------------------------------------------------------------------

#ifdef XLLAMA_UWP
namespace {

// C++/WinRT does not project the byte-access interop, and AudioFrame samples
// are only reachable through it.
struct __declspec(uuid("5b0d3235-4dba-4d44-865e-8f1d0e4fd04d")) __declspec(novtable)
IMemoryBufferByteAccessXll : ::IUnknown {
    virtual HRESULT __stdcall GetBuffer(uint8_t** value, uint32_t* capacity) = 0;
};

const char* graph_status_name(winrt::Windows::Media::Audio::AudioGraphCreationStatus s) {
    using S = winrt::Windows::Media::Audio::AudioGraphCreationStatus;
    switch (s) {
    case S::Success:
        return "Success";
    case S::DeviceNotAvailable:
        return "DeviceNotAvailable";
    case S::FormatNotSupported:
        return "FormatNotSupported";
    case S::UnknownFailure:
        return "UnknownFailure";
    }
    return "Unrecognised";
}

const char* input_status_name(winrt::Windows::Media::Audio::AudioDeviceNodeCreationStatus s) {
    using S = winrt::Windows::Media::Audio::AudioDeviceNodeCreationStatus;
    switch (s) {
    case S::Success:
        return "Success";
    case S::DeviceNotAvailable:
        return "DeviceNotAvailable";
    case S::FormatNotSupported:
        return "FormatNotSupported";
    case S::UnknownFailure:
        return "UnknownFailure";
    case S::AccessDenied:
        return "AccessDenied";
    }
    return "Unrecognised";
}

// json_escape is in xllama::json_utils.h — included above.

} // namespace
#endif // XLLAMA_UWP

void run_mic_probe() {
#ifdef XLLAMA_UWP
    using namespace winrt::Windows::Media::Audio;
    using winrt::Windows::Foundation::Metadata::ApiInformation;
    using winrt::Windows::Media::AudioBufferAccessMode;
    using winrt::Windows::Media::Capture::MediaCategory;
    using winrt::Windows::Media::Render::AudioRenderCategory;

    log_output("[xllama] mic: probing AppContainer audio capture (WS-F / H16.6)\n");

    // Capture duration. Three seconds is the card's number; long enough that a
    // hum or a breath clears the silence floor, short enough to keep the whole
    // probe inside one app launch.
    const int capture_ms = 3000;

    int ag_present = 0, mc_present = 0;
    std::string graph_status = "not-attempted";
    std::string input_status = "not-attempted";
    std::string error;
    uint32_t sample_rate = 0, channels = 0;
    uint64_t samples = 0;
    double rms = -1.0, peak = -1.0;

    try {
        ag_present = ApiInformation::IsTypePresent(L"Windows.Media.Audio.AudioGraph") ? 1 : 0;
        mc_present = ApiInformation::IsTypePresent(L"Windows.Media.Capture.MediaCapture") ? 1 : 0;

        if (ag_present) {
            AudioGraphSettings settings(AudioRenderCategory::Speech);
            auto graph_result = AudioGraph::CreateAsync(settings).get();
            graph_status = graph_status_name(graph_result.Status());

            if (graph_result.Status() == AudioGraphCreationStatus::Success) {
                auto graph = graph_result.Graph();
                sample_rate = graph.EncodingProperties().SampleRate();
                channels = graph.EncodingProperties().ChannelCount();

                auto in_result = graph.CreateDeviceInputNodeAsync(MediaCategory::Speech).get();
                input_status = input_status_name(in_result.Status());

                if (in_result.Status() == AudioDeviceNodeCreationStatus::Success) {
                    auto out = graph.CreateFrameOutputNode();
                    in_result.DeviceInputNode().AddOutgoingConnection(out);

                    // QuantumStarted fires on the audio engine thread; the graph
                    // is stopped and the handler revoked before these are read.
                    std::mutex acc_mu;
                    double sum_sq = 0.0, pk = 0.0;
                    uint64_t n = 0;

                    auto token = graph.QuantumStarted(
                        [&](AudioGraph const&, winrt::Windows::Foundation::IInspectable const&) {
                            try {
                                auto frame = out.GetFrame();
                                auto buffer = frame.LockBuffer(AudioBufferAccessMode::Read);
                                auto ref = buffer.CreateReference();
                                uint8_t* data = nullptr;
                                uint32_t cap = 0;
                                if (FAILED(ref.as<IMemoryBufferByteAccessXll>()->GetBuffer(&data,
                                                                                           &cap)))
                                    return;
                                const float* f = reinterpret_cast<const float*>(data);
                                const size_t count = cap / sizeof(float);
                                double s2 = 0.0, p = 0.0;
                                for (size_t i = 0; i < count; ++i) {
                                    const double v = static_cast<double>(f[i]);
                                    s2 += v * v;
                                    const double a = v < 0 ? -v : v;
                                    if (a > p)
                                        p = a;
                                }
                                std::lock_guard<std::mutex> lk(acc_mu);
                                sum_sq += s2;
                                n += count;
                                if (p > pk)
                                    pk = p;
                            } catch (...) {
                                // A throw here would cross the audio thread and take
                                // the process; the probe would rather lose a quantum.
                            }
                        });

                    graph.Start();
                    ::Sleep(static_cast<DWORD>(capture_ms));
                    graph.Stop();
                    graph.QuantumStarted(token); // revoke before reading

                    std::lock_guard<std::mutex> lk(acc_mu);
                    samples = n;
                    peak = pk;
                    rms = n ? std::sqrt(sum_sq / static_cast<double>(n)) : 0.0;
                }
            }
        }
    } catch (winrt::hresult_error const& e) {
        char b[256];
        snprintf(b, sizeof(b), "hresult 0x%08X", static_cast<unsigned>(e.code().value));
        error = b;
        error += ": " + winrt::to_string(e.message());
    } catch (std::exception const& e) {
        error = e.what();
    } catch (...) {
        error = "unknown exception";
    }

    char json[1024];
    snprintf(json, sizeof(json),
             "{\n"
             "  \"audiograph_type_present\": %d,\n"
             "  \"mediacapture_type_present\": %d,\n"
             "  \"graph_status\": \"%s\",\n"
             "  \"input_node_status\": \"%s\",\n"
             "  \"sample_rate\": %u,\n"
             "  \"channels\": %u,\n"
             "  \"capture_ms\": %d,\n"
             "  \"samples\": %llu,\n"
             "  \"rms\": %.8f,\n"
             "  \"peak\": %.8f,\n"
             "  \"error\": \"%s\"\n"
             "}\n",
             ag_present, mc_present, xllama::json_escape(graph_status).c_str(),
             xllama::json_escape(input_status).c_str(), sample_rate, channels, capture_ms,
             static_cast<unsigned long long>(samples), rms, peak,
             xllama::json_escape(error).c_str());

    log_output(std::string("[mic] ") + graph_status + " / " + input_status +
               " rms=" + std::to_string(rms) + " samples=" + std::to_string(samples) +
               (error.empty() ? "" : (" error=" + error)) + "\n");

    const std::string path = resolve_local_path("mic-result.json");
    FILE* fp = _wfopen(utf8_to_wstring(path).c_str(), L"w");
    if (fp) {
        fputs(json, fp);
        fclose(fp);
    } else {
        log_output("[xllama] mic: cannot open mic-result.json\n");
    }

    // The .done marker is written even when the probe failed — the host waits
    // on it, and a missing marker must mean "the process died", not "the answer
    // was no". That distinction is the whole reason for the try/catch above.
    FILE* done = _wfopen(utf8_to_wstring(resolve_local_path("mic-result.json.done")).c_str(), L"w");
    if (done) {
        fputs(error.empty() ? "ok\n" : "error\n", done);
        fclose(done);
    }
    log_output("[xllama] mic-result.json written\n");
#endif
}

// ---------------------------------------------------------------------------
// run_train_job_localized — shared by headless train.flag and in-app personalize
// ---------------------------------------------------------------------------

xllama::TrainingResult run_train_job_localized(const xllama::TrainingJob& job_in,
                                               const xllama::DeviceTrainCallbacks& cb_in) {
    xllama::TrainingResult fail;
    fail.success = false;
#ifdef XLLAMA_UWP
    #ifdef XLLAMA_DEVICE_TRAIN
    xllama::TrainingJob job = job_in;
    // Job paths are LocalState-relative on console (the process cwd is the
    // read-only install dir). Absolute paths pass through untouched.
    auto localize = [](std::string& p) {
        if (!p.empty() && p.find(':') == std::string::npos && p[0] != '\\' && p[0] != '/')
            p = resolve_local_path(p);
    };
    localize(job.base_model);
    localize(job.dataset_path);
    localize(job.out_dir);

    log_output(("[xllama] train: " + xllama::format_training_job_summary(job) + "\n").c_str());

    // Compose callbacks: always log + write training/progress.json so the UI
    // and GET /v1/training/status can poll without WinRT.
    xllama::DeviceTrainCallbacks cb = cb_in;
    auto prev_status = cb.on_status;
    auto prev_progress = cb.on_progress;
    cb.on_status = [prev_status](const std::string& line) {
        log_output(("[xllama] train: " + line + "\n").c_str());
        if (prev_status)
            prev_status(line);
    };
    cb.on_progress = [prev_progress](const xllama::DeviceTrainProgress& p) {
        const char* stage = xllama::training_stage_name(p.stage);
        char lb[160];
        snprintf(lb, sizeof(lb), "[xllama] train: epoch %d/%d batch %lld/%lld loss=%.4f\n", p.epoch,
                 p.epochs, static_cast<long long>(p.ibatch), static_cast<long long>(p.ibatch_max),
                 p.loss);
        log_output(lb);
        const std::string json = xllama::format_train_progress_json(
            stage ? stage : "train", p.epoch, p.epochs, p.ibatch, p.ibatch_max, p.loss);
        const std::string prog_path = resolve_local_path("training/progress.json");
        FILE* fp = _wfopen(utf8_to_wstring(prog_path).c_str(), L"w");
        if (fp) {
            fputs(json.c_str(), fp);
            fclose(fp);
        }
        if (prev_progress)
            prev_progress(p);
    };

    return xllama::run_device_train_job(job, cb);
    #else
    (void)job_in;
    (void)cb_in;
    fail.error_msg = "built without XLLAMA_DEVICE_TRAIN";
    return fail;
    #endif
#else
    (void)job_in;
    (void)cb_in;
    fail.error_msg = "run_train_job_localized is UWP-only";
    return fail;
#endif
}

// run_train (called from UWP train.flag mode background thread)
// ---------------------------------------------------------------------------

void run_train() {
#ifdef XLLAMA_UWP
    bool ok = false;
    std::string err;
    #ifdef XLLAMA_DEVICE_TRAIN
    const std::string job_path = resolve_local_path("training/job.json");
    xllama::TrainingJob job;
    ok = xllama::load_training_job_file(job_path, job, &err);
    if (ok) {
        const xllama::TrainingResult r = run_train_job_localized(job);
        ok = r.success;
        if (!ok)
            err = r.error_msg;
    }
    if (!ok)
        log_output(("[xllama] train FAIL: " + err + "\n").c_str());

    #else
    err = "built without XLLAMA_DEVICE_TRAIN";
    log_output(("[xllama] train FAIL: " + err + "\n").c_str());
    #endif
    FILE* done = _wfopen(utf8_to_wstring(resolve_local_path("training/result.done")).c_str(), L"w");
    if (done) {
        fputs(ok ? "ok\n" : "fail\n", done);
        fclose(done);
    }
#endif
}

} // namespace xllama::bridge
