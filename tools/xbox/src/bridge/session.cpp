// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// xllama::Session — persistent model session for multi-turn applications.
// ORT GenAI path (UWP): model + tokenizer loaded once, reused per generate().
// llama.cpp path (Linux): model kept alive; context rebuilt per generate().

#include "xllama/session.h"
#include "xllama/chat_prompt.h"
#include "xllama/ggml_d3d12.h"
#include "xllama/inference_params.h"
#include "xllama/mtp_draft.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"
#include "xllama/routing_policy.h"
#include "xllama/session_hub.h"
#include "xllama/utf8_utils.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

// Backend availability. A build defines XLLAMA_USE_ORT when ORT GenAI is linked.
// llama.cpp is available on Linux (always) and on the UWP llamacpp variant
// (XllamaBackend=llamacpp → XLLAMA_USE_ORT undefined). The unified UWP build
// defines BOTH explicitly; this shim derives the llama flag so exactly one path
// matches every current single-backend build without touching the build files.
#if !defined(XLLAMA_USE_LLAMA)
    #if defined(XLLAMA_LINUX) || !defined(XLLAMA_USE_ORT)
        #define XLLAMA_USE_LLAMA 1
    #endif
#endif

#if !defined(XLLAMA_USE_ORT) && !defined(XLLAMA_USE_LLAMA)
    #error "no inference backend compiled (define XLLAMA_USE_ORT and/or XLLAMA_USE_LLAMA)"
#endif

namespace xllama {
namespace detail {
// Backend factories, each defined in its own #ifdef block below. The public
// Session::create dispatches to one of these.
#ifdef XLLAMA_USE_ORT
std::unique_ptr<Session> create_ort(const SessionParams& sp, std::string* err);
#endif
#ifdef XLLAMA_USE_LLAMA
std::unique_ptr<Session> create_llama(const SessionParams& sp, std::string* err);
#endif
} // namespace detail
} // namespace xllama

// ---------------------------------------------------------------------------
// ONNX Runtime GenAI path (UWP / Xbox Series S)
// ---------------------------------------------------------------------------
#ifdef XLLAMA_USE_ORT

// clang-format off
#include <windows.h>
#include <eh.h>
#include "ort_genai_c.h"

#include "decode_loop_ort.h"
#include "ort_common.h"
#include "ort_sampling.h" // shared ORT search-param builder (#125); needs ort_genai_c.h
#include "xllama/ort_raii.h"
// clang-format on

namespace xllama {

class OrtSession final : public Session {
  public:
    OgaModelPtr m_model;
    OgaTokenizerPtr m_tok;
    int m_n_ctx;

    // Persistent chat state for KV-cache reuse (continuous decoding). Kept alive
    // between generate() calls when GenerateParams::reuse_kv is set. m_chat_params
    // must outlive m_chat_gen.
    OgaGeneratorParamsPtr m_chat_params;
    OgaGeneratorPtr m_chat_gen;
    OgaTokenizerStreamPtr m_chat_stream;
    bool m_chat_valid = false;
    // Sampling signature bound into m_chat_gen — the generator can only be reused
    // while these are unchanged (ORT binds sampling at generator creation).
    float m_b_temp = 0.0f, m_b_top_p = 0.0f, m_b_rep = 0.0f;
    int m_b_top_k = 0;

    explicit OrtSession(OgaModelPtr model, OgaTokenizerPtr tok, int n_ctx)
        : m_model(std::move(model)), m_tok(std::move(tok)), m_n_ctx(n_ctx) {}

    void reset_chat_state() {
        m_chat_gen.reset(); // generator before its params
        m_chat_params.reset();
        m_chat_stream.reset();
        m_chat_valid = false;
    }

    bool sampling_matches(const GenerateParams& gp) const {
        return m_chat_valid && m_b_temp == gp.temperature && m_b_top_p == gp.top_p &&
               m_b_top_k == gp.top_k && m_b_rep == gp.repetition_penalty;
    }

    OgaGeneratorParamsPtr make_params(const GenerateParams& gp) {
        OgaGeneratorParams* raw_params = nullptr;
        oga_check(OgaCreateGeneratorParams(m_model.get(), &raw_params), "OgaCreateGeneratorParams");
        OgaGeneratorParamsPtr gparams(raw_params);
        // Session ALWAYS saturates max_length to n_ctx via the shared #130
        // ladder (resolve_max_length, inference_params.h) — the interior band
        // is a measured DirectML valley (§5c; table at the stateless caller).
        // The n_predict bound is enforced by run_decode's cap instead.
        const int max_length = resolve_max_length(m_n_ctx, 0, 0, /*override_v=*/-1);
        oga_check(OgaGeneratorParamsSetSearchNumber(gparams.get(), "max_length",
                                                    static_cast<double>(max_length)),
                  "SetSearchNumber max_length");
        // #125 follow-up: temperature and the greedy guard via the shared helper,
        // so this path and run_inference cannot drift apart again. This is the
        // fix for the greedy gap — make_params used to set the full chain
        // unconditionally, running the repetition penalty before argmax at
        // temperature 0.
        apply_ort_sampling(gparams.get(), gp.sampling());
        return gparams;
    }

    // Decode loop shared by the stateless and chat paths. The caller has already
    // appended the prompt tokens and captured t_prefill_start immediately before
    // AppendTokenSequences; prefill-end is marked after the first
    // GenerateNextToken (in ORT GenAI the prompt prefill runs during the first
    // compute step), so decode timing excludes prefill. n_predict_cap > 0 caps
    // per-turn generation (chat mode); 0 relies on max_length (stateless mode).
    //
    // Consolidated into decode_loop_ort.h to eliminate duplication with
    // run_inference_ort (inference.cpp).
    void run_decode(OgaGenerator* gen, OgaTokenizerStream* stream, const GenerateParams& gp,
                    InferenceResult& res, std::chrono::steady_clock::time_point t_prefill_start,
                    int n_prompt_tok, int n_predict_cap) {
        detail::run_decode_loop_ort(gen, stream, gp, res, t_prefill_start, n_prompt_tok,
                                    n_predict_cap);
    }

    // Legacy stateless turn: fresh generator, full prompt, destroyed at return.
    void generate_stateless(const GenerateParams& gp, InferenceResult& res) {
        if (gp.on_status)
            gp.on_status("tokenizing");

        OgaTokenizerStream* raw_stream = nullptr;
        oga_check(OgaCreateTokenizerStream(m_tok.get(), &raw_stream), "OgaCreateTokenizerStream");
        OgaTokenizerStreamPtr stream(raw_stream);

        OgaSequences* raw_seqs = nullptr;
        oga_check(OgaCreateSequences(&raw_seqs), "OgaCreateSequences");
        OgaSequencesPtr seqs(raw_seqs);
        oga_check(OgaTokenizerEncode(m_tok.get(), gp.prompt.c_str(), seqs.get()),
                  "OgaTokenizerEncode");
        int n_prompt_tok = static_cast<int>(OgaSequencesGetSequenceCount(seqs.get(), 0));

        // #130: request the FULL context as max_length and bound generation with
        // the n_predict cap in run_decode — exactly what generate_chat already
        // does. The obvious alternative, max_length = min(n_ctx, prompt +
        // n_predict), is what this used to do, and on DirectML it is 3-4x slower.
        //
        // Measured 2026-07-21, Series S, one byte-identical 1289-token prompt,
        // only n_predict varied (bench/results/phase12-maxlen-band.csv):
        //
        //   max_length  1297  1400  1545  1650  1801  1950  2048
        //   prefill t/s   515   475   170   215   130   212   611
        //
        // There is a valley in max_length between ~1400 and n_ctx, deepest near
        // 1800, with both edges clean. The prompt never changed, so this is not a
        // prompt-length effect. A control run at n_ctx 3072 holding max_length at
        // 1801 reproduced the slow figure (132 t/s), so n_ctx has no effect of its
        // own — max_length alone controls it. Saturating also costs LESS memory
        // here (1968 MB against 2483 at max_length 1950).
        //
        // The shipping default (n_predict 256) landed at 1545 — inside the valley.
        // Mechanism still unknown; suspected shape-bucketed DML kernel selection.
        OgaGeneratorParamsPtr gparams = make_params(gp);
        OgaGenerator* raw_gen = nullptr;
        oga_check(OgaCreateGenerator(m_model.get(), gparams.get(), &raw_gen), "OgaCreateGenerator");
        OgaGeneratorPtr gen(raw_gen);

        if (gp.on_status)
            gp.on_status("generating");
        auto t0 = std::chrono::steady_clock::now();
        oga_check(OgaGenerator_AppendTokenSequences(gen.get(), seqs.get()), "AppendTokenSequences");
        // Reproduce the old allowance exactly. The previous max_length of
        // min(n_ctx, prompt + n_predict) permitted min(n_ctx - prompt, n_predict)
        // new tokens; max_length is no longer the bound, so the cap has to carry
        // that arithmetic or a long prompt would now generate past the context.
        // Clamped to >= 1 because run_decode treats a cap of 0 as "no cap": a
        // prompt at or beyond the context would otherwise flip from a hard stop
        // to unbounded generation. Such a prompt fails inside ORT first
        // ("input_ids size exceeds max length"), which is the intended error.
        const int n_predict_cap = std::max(1, std::min(m_n_ctx - n_prompt_tok, gp.n_predict));
        run_decode(gen.get(), stream.get(), gp, res, t0, n_prompt_tok, n_predict_cap);
        log_gen(res, false);
    }

    // Continuation turn: append `gp.prompt` (the new turn's tokens) to the
    // persistent generator; on reset_kv or a sampling change, rebuild it first
    // (then `gp.prompt` is the full context).
    void generate_chat(const GenerateParams& gp, InferenceResult& res) {
        const bool reuse = !gp.reset_kv && sampling_matches(gp);
        if (!reuse) {
            reset_chat_state();
            m_chat_params = make_params(gp);
            OgaGenerator* raw_gen = nullptr;
            oga_check(OgaCreateGenerator(m_model.get(), m_chat_params.get(), &raw_gen),
                      "OgaCreateGenerator(chat)");
            m_chat_gen.reset(raw_gen);
            OgaTokenizerStream* raw_stream = nullptr;
            oga_check(OgaCreateTokenizerStream(m_tok.get(), &raw_stream),
                      "OgaCreateTokenizerStream(chat)");
            m_chat_stream.reset(raw_stream);
            m_b_temp = gp.temperature;
            m_b_top_p = gp.top_p;
            m_b_top_k = gp.top_k;
            m_b_rep = gp.repetition_penalty;
            m_chat_valid = true;
        }

        if (gp.on_status)
            gp.on_status("tokenizing");
        OgaSequences* raw_seqs = nullptr;
        oga_check(OgaCreateSequences(&raw_seqs), "OgaCreateSequences");
        OgaSequencesPtr seqs(raw_seqs);
        oga_check(OgaTokenizerEncode(m_tok.get(), gp.prompt.c_str(), seqs.get()),
                  "OgaTokenizerEncode");
        int n_prompt_tok = static_cast<int>(OgaSequencesGetSequenceCount(seqs.get(), 0));

        // #173: a continuation that cannot fit (KV + delta + >=1 generated token
        // over n_ctx) is predictable from state we already hold — fail before
        // touching the generator instead of letting the append discover it. The
        // throw lands in generate()'s catch, which resets the chat state; the
        // caller sees n_eval == 0 and falls back to a trimmed full prefill.
        if (reuse) {
            const int kv = static_cast<int>(OgaGenerator_GetSequenceCount(m_chat_gen.get(), 0));
            if (kv + n_prompt_tok + 1 > m_n_ctx)
                throw std::runtime_error("context full: kv=" + std::to_string(kv) +
                                         " + delta=" + std::to_string(n_prompt_tok) +
                                         " exceeds n_ctx=" + std::to_string(m_n_ctx));
        }

        if (gp.on_status)
            gp.on_status("generating");
        auto t0 = std::chrono::steady_clock::now();
        oga_check(OgaGenerator_AppendTokenSequences(m_chat_gen.get(), seqs.get()),
                  "AppendTokenSequences(chat)");
        run_decode(m_chat_gen.get(), m_chat_stream.get(), gp, res, t0, n_prompt_tok, gp.n_predict);

        size_t kv = OgaGenerator_GetSequenceCount(m_chat_gen.get(), 0);
        char lb[192];
        snprintf(lb, sizeof(lb),
                 "[xllama] chat turn: reuse=%d prefill=%d tok decode=%d tok kv_len=%zu\n",
                 reuse ? 1 : 0, res.n_p_eval, res.n_eval, kv);
        log_output(lb);
    }

    void log_gen(const InferenceResult& res, bool chat) {
        double dt =
            (res.n_eval > 0 && res.t_eval_ms > 0) ? res.n_eval / (res.t_eval_ms / 1000.0) : 0.0;
        // Prefill rate too: without it the per-turn TTFT (and the #130
        // cold-vs-warm gap) cannot be reconstructed from the log.
        double pt = (res.n_p_eval > 0 && res.t_p_eval_ms > 0)
                        ? res.n_p_eval / (res.t_p_eval_ms / 1000.0)
                        : 0.0;
        char log_buf[256];
        snprintf(log_buf, sizeof(log_buf),
                 "[xllama] session generate%s: prefill=%.1f tok/s (%d tok, %.0f ms) "
                 "decode=%.1f tok/s n=%d\n",
                 chat ? " (chat)" : "", pt, res.n_p_eval, res.t_p_eval_ms, dt, res.n_eval);
        log_output(log_buf);
    }

    InferenceResult generate(const GenerateParams& gp) override {
        InferenceResult res;
        install_se_translator();
        try {
            if (gp.reuse_kv)
                generate_chat(gp, res);
            else
                generate_stateless(gp, res);
        } catch (const std::exception& e) {
            res.success = false;
            res.error_msg = e.what();
            if (gp.reuse_kv)
                reset_chat_state(); // poisoned generator — force a fresh one next turn
            log_output(("[xllama] session generate error: " + res.error_msg + "\n").c_str());
            if (gp.on_status)
                gp.on_status("error: " + res.error_msg);
        }
        return res;
    }

    int count_tokens(const std::string& prompt) override {
        OgaSequences* raw_seqs = nullptr;
        oga_check(OgaCreateSequences(&raw_seqs), "OgaCreateSequences");
        OgaSequencesPtr seqs(raw_seqs);
        oga_check(OgaTokenizerEncode(m_tok.get(), prompt.c_str(), seqs.get()),
                  "OgaTokenizerEncode");
        return static_cast<int>(OgaSequencesGetSequenceCount(seqs.get(), 0));
    }

    int context_length() const override {
        return m_n_ctx;
    }
};

namespace detail {
std::unique_ptr<Session> create_ort(const SessionParams& sp, std::string* err) {
    install_se_translator();

    const std::string model_dir = resolve_model_path(sp.model_path);
    try {
        // Redirect ORT GenAI internal log messages into xllama.log.
        register_oga_logging();

        OgaModel* raw_model = nullptr;
        oga_check(OgaCreateModel(model_dir.c_str(), &raw_model), "OgaCreateModel");
        OgaModelPtr model(raw_model);

        OgaTokenizer* raw_tok = nullptr;
        oga_check(OgaCreateTokenizer(model.get(), &raw_tok), "OgaCreateTokenizer");
        OgaTokenizerPtr tok(raw_tok);

        log_output("[xllama] Session: model loaded (persistent)\n");
        GpuMemInfo gpu = gpu_mem_info();
        if (gpu.available) {
            char gpu_buf[128];
            snprintf(gpu_buf, sizeof(gpu_buf),
                     "[xllama] gpu-mem post-load: current=%zuMB budget=%zuMB\n", gpu.current_mb,
                     gpu.budget_mb);
            log_output(gpu_buf);
        }
        int n_ctx = sp.n_ctx > 0 ? sp.n_ctx : 2048;

        // #130: DirectML pays a large one-time per-process cost on the first
        // generate (§5e: cold→warm prefill 1.64-1.72×; CPU control 1.00× —
        // suspected lazy kernel compilation). Pay it here, inside the load
        // phase the UI already presents as such, instead of on the user's
        // first turn. The throwaway generator uses max_length = n_ctx exactly
        // like every real turn: the §5c valley is keyed by max_length, so
        // warming a smaller shape could warm the wrong regime. Failure is
        // non-fatal — the session is still usable, just cold.
        if (sp.dml_warmup && model_is_dml(sp.model_path)) {
            try {
                const auto t_w0 = std::chrono::steady_clock::now();
                OgaGeneratorParams* raw_wp = nullptr;
                oga_check(OgaCreateGeneratorParams(model.get(), &raw_wp),
                          "OgaCreateGeneratorParams warmup");
                OgaGeneratorParamsPtr wparams(raw_wp);
                oga_check(OgaGeneratorParamsSetSearchNumber(wparams.get(), "max_length",
                                                            static_cast<double>(n_ctx)),
                          "SetSearchNumber warmup");
                OgaSequences* raw_wseqs = nullptr;
                oga_check(OgaCreateSequences(&raw_wseqs), "OgaCreateSequences warmup");
                OgaSequencesPtr wseqs(raw_wseqs);
                // A ~2-token warm-up left turn-1 at ~76% of the warm prefill
                // rate (validated on-console, PR #158 build 678): a tiny
                // prompt never exercises the long-sequence prefill GEMMs. A
                // few hundred tokens lands in the same regime as real turns.
                std::string wtext;
                wtext.reserve(2048);
                for (int i = 0; i < 256; ++i)
                    wtext += "warmup ";
                oga_check(OgaTokenizerEncode(tok.get(), wtext.c_str(), wseqs.get()),
                          "OgaTokenizerEncode warmup");
                OgaGenerator* raw_wgen = nullptr;
                oga_check(OgaCreateGenerator(model.get(), wparams.get(), &raw_wgen),
                          "OgaCreateGenerator warmup");
                OgaGeneratorPtr wgen(raw_wgen);
                oga_check(OgaGenerator_AppendTokenSequences(wgen.get(), wseqs.get()),
                          "AppendTokenSequences warmup");
                // 3 steps: the first IS the prefill compute; only later ones
                // run the seq=1 decode kernels (turn-1 decode stayed 17% slow
                // with a single step — decode was never warmed).
                for (int i = 0; i < 3; ++i)
                    oga_check(OgaGenerator_GenerateNextToken(wgen.get()),
                              "GenerateNextToken warmup");
                const double w_ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - t_w0)
                                        .count();
                char w_buf[96];
                snprintf(w_buf, sizeof(w_buf), "[xllama] Session: DML warm-up %.0f ms\n", w_ms);
                log_output(w_buf);
            } catch (const std::exception& we) {
                log_output(("[xllama] Session: DML warm-up failed (non-fatal): " +
                            std::string(we.what()) + "\n")
                               .c_str());
            }
        }
        return std::make_unique<OrtSession>(std::move(model), std::move(tok), n_ctx);

    } catch (const std::exception& e) {
        if (err)
            *err = e.what();
        log_output(("[xllama] Session::create error: " + std::string(e.what()) + "\n").c_str());
        return nullptr;
    }
}
} // namespace detail

} // namespace xllama

#endif // XLLAMA_USE_ORT

// ---------------------------------------------------------------------------
// llama.cpp path (Linux dev + UWP llamacpp/unified)
// ---------------------------------------------------------------------------
#ifdef XLLAMA_USE_LLAMA

    #include "ggml-backend.h"
    #include "llama.h"
    #include "state_identity.h"

    #include "decode_loop.h" // shared prefill + generation loops; needs llama.h
    #include "llama_gpu.h"
    #include "sampler_chain.h" // shared sampler chain (#125); needs llama.h
    #include "xllama/llama_raii.h"

    #include <vector>

namespace xllama {

class LlamaSession final : public Session {
  public:
    LlamaModelPtr m_model;
    LlamaAdapterLoraPtr m_adapter; // optional runtime LoRA (freed before model)
    float m_lora_scale = 1.0f;
    int m_n_ctx;
    int m_n_threads;
    int m_n_threads_batch; // graphs with >1 token (prefill/verify/catch-up)
    int m_n_batch;         // 0 = llama.cpp default
    int m_n_ubatch;        // 0 = llama.cpp default
    int m_n_seq_max = 1;   // llama.cpp context sequence capacity (1 = single session)
    // Persistent context across turns: the KV cache lives here so a reuse turn
    // (reuse_kv && !reset_kv) can append only the delta instead of re-prefilling
    // the whole conversation. Created lazily on the first generate().
    GgufCpuThreadpools m_cpu_pools; // declared before m_ctx: must outlive it
    LlamaContextPtr m_ctx;
    // #175 decision: sampler state follows the KV lifecycle, mirroring the ORT
    // persistent generator — the penalty window (and the dist RNG) live as long
    // as the conversation, so the tail of the previous reply stays penalized at
    // the start of the next turn. Rebuilt on reset_kv or a sampling change
    // (same_chain guard). The 64-token window itself is a deliberate divergence
    // from ORT's whole-sequence penalty — see sampling.h.
    LlamaSamplerPtr m_sampler;
    SamplingConfig m_sampler_cfg;

    // #170a: the tokens whose KV is currently resident (prefilled + generated,
    // in position order). A full-prompt turn diffs its tokenization against
    // this and rewinds only the divergent tail (llama_memory_seq_rm) instead
    // of clearing — regenerate, an edited last message, and LAN-API requests
    // that extend the previous one re-prefill only the difference. Cleared
    // whenever the KV state stops being trustworthy (decode failure).
    std::vector<llama_token> m_kv_tokens;

    bool m_kv_q8 = false;         // #171: q8_0 KV + forced flash attention
    bool m_prompt_lookup = false; // #210: draft-free speculative decoding
    bool m_mtp = false;           // MTP drafting against the beellama fork
    bool m_mtp_active = false;    // drafter came up (plan 003, stage 1 gate)
    int m_mtp_n_max = 4;
    float m_mtp_p_min = 0.75f;
    int m_gpu_layers = 0; // layers on the d3d12 backend (llama_gpu.h)

    // MTP drafter, created lazily with the context. Owned here rather than
    // passed into decode_loop so the draft context shares the session lifetime
    // and is freed before the model it was built from.
    std::unique_ptr<MtpDrafter> m_mtp_drafter;

    // #169: whether the resident KV supports front-drop eviction + RoPE shift.
    // Known once the lazy context exists. Gated on llama_memory_can_shift —
    // seq_add on an unsupported arch (mrope) is a GGML_ASSERT abort, not an
    // error — and on the model not using SWA, where the resident window is
    // not [0, kv_len) and the shift arithmetic below would lie.
    bool m_can_shift = false;

    explicit LlamaSession(LlamaModelPtr model, LlamaAdapterLoraPtr adapter, float lora_scale,
                          int n_ctx, int n_threads, int n_threads_batch, int n_batch, int n_ubatch,
                          bool kv_q8, bool prompt_lookup, int gpu_layers, bool mtp, int mtp_n_max,
                          float mtp_p_min, int n_seq_max)
        : m_model(std::move(model)), m_adapter(std::move(adapter)), m_lora_scale(lora_scale),
          m_n_ctx(n_ctx), m_n_threads(n_threads), m_n_threads_batch(n_threads_batch),
          m_n_batch(n_batch), m_n_ubatch(n_ubatch), m_kv_q8(kv_q8), m_prompt_lookup(prompt_lookup),
          m_gpu_layers(gpu_layers), m_mtp(mtp), m_mtp_n_max(mtp_n_max), m_mtp_p_min(mtp_p_min),
          m_n_seq_max(n_seq_max > 0 ? n_seq_max : 1) {}

    // Plan 009 diagnostic: one bounded first-attention capture per target context.
    // Dormant unless fa-capture.flag is exactly "1"; never installed by default.
    bool m_fa_captured = false;
    int m_fa_query_policy = 0;
    static bool capture_first_fa(ggml_tensor* t, bool ask, void* opaque) {
        auto* self = static_cast<LlamaSession*>(opaque);
        if (self->m_fa_captured || t->op != GGML_OP_FLASH_ATTN_EXT || !t->src[0] ||
            t->src[0]->ne[1] <= 1)
            return false;
        if (ask)
            return true;
        self->m_fa_captured = true;
        FILE* fp = open_state_file(resolve_local_path("fa-probe.bin"), "wb");
        if (!fp) {
            log_output("[xllama] fa-probe: open failed\n");
            return true;
        }
        bool ok = true;
        auto write = [&](const void* p, size_t n) {
            if (ok && std::fwrite(p, 1, n, fp) != n)
                ok = false;
        };
        const uint32_t header[2] = {0x31414658, 5}; // XFA1, q/k/v/mask/output
        write(header, sizeof(header));
        for (int i = 0; i < 5 && ok; ++i) {
            ggml_tensor* node = i < 4 ? t->src[i] : t;
            if (!node || !node->data || ggml_nbytes(node) > (8u << 20)) {
                ok = false;
                break;
            }
            const int32_t fields[3] = {i, static_cast<int32_t>(node->op),
                                       static_cast<int32_t>(node->type)};
            const uint64_t bytes = ggml_nbytes(node);
            uint64_t strides[4];
            for (int a = 0; a < 4; ++a)
                strides[a] = node->nb[a];
            write(fields, sizeof(fields));
            write(node->ne, sizeof(node->ne));
            write(strides, sizeof(strides));
            write(&bytes, sizeof(bytes));
            write(node->name, sizeof(node->name));
            std::vector<uint8_t> raw(static_cast<size_t>(bytes));
            ggml_backend_tensor_get(node, raw.data(), 0, raw.size());
            write(raw.data(), raw.size());
        }
        if (std::fclose(fp) != 0)
            ok = false;
        log_output(ok ? "[xllama] fa-probe: complete\n" : "[xllama] fa-probe: FAILED\n");
        return true;
    }

    // Lazy context creation, shared by generate() and the state-file entry
    // points (#170b needs a context before the first turn). Returns false and
    // sets *err on failure; m_ctx stays null.
    bool ensure_ctx(std::string* err, bool for_embedding = false) {
        if (m_ctx)
            return true;
        {
            // Plan 006 C4: setup-time target layer count for the target-only
            // SWIGLU whitelist (never a mid-graph toggle).
            // Conflict check FIRST: a rejected creation must not mutate any
            // backend state an existing context depends on.
            if (!d3d12_swiglu_profile_accepts(m_mtp)) {
                if (err)
                    *err = "FFN SWIGLU profile is bound to the other MTP mode; restart required";
                return false;
            }
            d3d12_set_swiglu_target_layers(llama_model_n_layer(m_model.get()));
            d3d12_finalize_swiglu_mode(m_mtp);
            llama_context_params cparams = llama_context_default_params();
            cparams.n_ctx = m_n_ctx;
            cparams.n_threads = m_n_threads;
            // Built-in perf counters (numerics-neutral): llama_perf_context()
            // stays zero while the default no_perf=true — we need n_reused per
            // verify call to split rebuild-vs-reuse in the wall-d3w lump
            // (plan004 rev102). Cost is a ggml_time_us() per synchronize().
            cparams.no_perf = false;
            // Prefill (any ubatch > 1 token) runs on n_threads_batch, whose
            // default is GGML_DEFAULT_N_THREADS (4) regardless of n_threads —
            // left unset it caps prefill at 4 threads while decode gets 6 (#168).
            // Plan 006 C1: the batch count is separately selectable; it equals
            // n_threads unless the caller resolved a different value.
            cparams.n_threads_batch = m_n_threads_batch;
            if (m_n_batch > 0)
                cparams.n_batch = static_cast<uint32_t>(m_n_batch);
            if (m_n_ubatch > 0)
                cparams.n_ubatch = static_cast<uint32_t>(m_n_ubatch);
            // One context, up to n_seq_max sequences (llama.ini [n_seq_max] /
            // [n_parallel]). The default 1 is exactly the pre-existing
            // single-sequence context; sequences share the same n_ctx cells.
            cparams.n_seq_max = static_cast<uint32_t>(m_n_seq_max);
            if (m_n_seq_max != 1)
                log_output("[xllama] session: n_seq_max=" + std::to_string(m_n_seq_max) +
                           " (sequences share n_ctx=" + std::to_string(m_n_ctx) + ")\n");
            clamp_speculative_n_rs_seq(cparams, speculative_n_rs_seq(m_mtp, m_mtp_n_max));
            if (m_mtp && cparams.n_rs_seq > 0)
                log_output("[xllama] session: speculative n_rs_seq=" +
                           std::to_string(cparams.n_rs_seq) + "\n");

            apply_gguf_gpu_context(m_gpu_layers, cparams);
            const char* no_tile = std::getenv("GGML_FLASH_ATTN_NO_QUERY_TILING");
            m_fa_query_policy = no_tile && std::strcmp(no_tile, "4") == 0   ? 4
                                : no_tile && std::strcmp(no_tile, "1") == 0 ? 1
                                                                            : 0;
            if (for_embedding) {
                // Non-causal embedding graphs cannot split a sequence into
                // physical micro-batches (llama.cpp asserts instead of failing).
                cparams.n_batch = std::min(cparams.n_batch, cparams.n_ctx);
                cparams.n_ubatch = cparams.n_batch;
                cparams.embeddings = true;
            }
            if (m_kv_q8) {
                // #171: quantized V requires flash attention (the pin throws at
                // context creation with FA disabled, and AUTO may resolve to
                // disabled) — force it and fall back below if the arch refuses.
                cparams.type_k = GGML_TYPE_Q8_0;
                cparams.type_v = GGML_TYPE_Q8_0;
                cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            }
            if (m_mtp) {
                FILE* flag = open_state_file(resolve_local_path("fa-capture.flag"), "rb");
                if (flag) {
                    const bool enabled = std::fgetc(flag) == '1' && std::fgetc(flag) == EOF;
                    std::fclose(flag);
                    if (enabled) {
                        m_fa_captured = false;
                        cparams.cb_eval = capture_first_fa;
                        cparams.cb_eval_user_data = this;
                    }
                }
            }
            m_ctx.reset(llama_init_from_model(m_model.get(), cparams));
            if (!m_ctx && m_kv_q8) {
                log_output("[xllama] session: q8_0 KV context failed — falling back "
                           "to default cache types (#171)\n");
                cparams.type_k = llama_context_default_params().type_k;
                cparams.type_v = llama_context_default_params().type_v;
                cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
                m_kv_q8 = false;
                m_ctx.reset(llama_init_from_model(m_model.get(), cparams));
            }
            if (m_ctx)
                log_output("[xllama] session effective threads: decode=" +
                           std::to_string(llama_n_threads(m_ctx.get())) +
                           " batch=" + std::to_string(llama_n_threads_batch(m_ctx.get())) + "\n");
            if (m_ctx && m_mtp) {
                // The target has to expose nextn hidden rows; that is what the
                // drafter reads to seed each step. Enabled only here so a
                // session without MTP pays nothing for it.
                llama_set_embeddings_nextn(m_ctx.get(), true, /*masked=*/false);
                MtpDraftParams mp;
                mp.n_max = m_mtp_n_max;
                mp.p_min = m_mtp_p_min;
                m_mtp_drafter = std::make_unique<MtpDrafter>();
                if (!m_mtp_drafter->init(m_model.get(), m_ctx.get(), cparams, mp,
                                         llama_model_n_embd_out(m_model.get()))) {
                    // Not fatal: MTP is an accelerator, so a failure here must
                    // leave a working single-token session behind.
                    m_mtp_drafter.reset();
                } else {
                    // Shared with the target, used strictly sequentially.
                    m_cpu_pools.attach(m_gpu_layers, m_mtp_drafter->ctx(),
                                       static_cast<int>(cparams.n_threads),
                                       static_cast<int>(cparams.n_threads_batch));
                    m_mtp_active = true;
                }
            }
            m_cpu_pools.attach(m_gpu_layers, m_ctx.get(), m_n_threads, m_n_threads_batch);
            if (!m_ctx) {
                if (err)
                    *err = "failed to create context";
                log_output("[xllama] session: failed to create context\n");
                return false;
            }
            m_can_shift = llama_memory_can_shift(llama_get_memory(m_ctx.get())) &&
                          llama_model_n_swa(m_model.get()) == 0;
            // Logged because it is a per-ARCH capability that decides whether a long
            // chat shifts (#169) or fail-fasts (#173), and docs used to state it from
            // assumption: imrope and SWA architectures cannot shift, and which is
            // which is not something to guess per model. One line makes it a
            // measurement anyone can read off the device log.
            {
                char cb[128];
                snprintf(cb, sizeof(cb),
                         "[xllama] session: can_shift=%d (memory_can_shift=%d n_swa=%d)\n",
                         m_can_shift ? 1 : 0,
                         llama_memory_can_shift(llama_get_memory(m_ctx.get())) ? 1 : 0,
                         (int)llama_model_n_swa(m_model.get()));
                log_output(cb);
            }
            // Apply runtime LoRA after context exists (llama_set_adapters_lora is
            // context-scoped). Must re-apply if context is ever recreated.
            if (m_adapter) {
                llama_adapter_lora* arr[1] = {m_adapter.get()};
                float scales[1] = {m_lora_scale};
                if (llama_set_adapters_lora(m_ctx.get(), arr, 1, scales) != 0) {
                    if (err)
                        *err = "llama_set_adapters_lora failed";
                    log_output("[xllama] session: set_adapters_lora failed\n");
                    m_ctx.reset();
                    return false;
                }
                log_output("[xllama] session: runtime LoRA adapter applied\n");
            }
        }
        return true;
    }

    InferenceResult generate(const GenerateParams& gp) override {
        InferenceResult res;
        // Wall clock for the WHOLE call, started here so it covers load + prefill +
        // decode. Reported as res.t_total_ms only: res.t_eval_ms stays the decode
        // interval because published throughput divides n_eval by it, and the phase
        // line needs the wider interval to attribute the difference.
        const auto t_total0 = std::chrono::steady_clock::now();

        if (!ensure_ctx(&res.error_msg))
            return res;
        llama_context* ctx = m_ctx.get();

        // KV-cache reuse: a continuation turn (reuse_kv && !reset_kv) keeps the
        // existing cache and appends the delta; otherwise re-prefill the full
        // prompt — rewinding to the common token prefix when one is resident
        // (#170a) instead of always clearing. add_bos only on a full prompt —
        // a delta must not carry a BOS mid-conversation.
        const bool full_prompt = gp.reset_kv || !gp.reuse_kv;

        const llama_vocab* vocab = llama_model_get_vocab(m_model.get());

        // parse_special=true: every Session caller (chat UI, API endpoint, bench)
        // passes a TEMPLATED prompt, whose <|im_start|>/<|im_end|> markers must
        // become the special ids the model was trained on. As plain text the
        // model sees an alien template and its next-token distribution goes flat
        // (LFM2.5: top-2 gap 0.52 vs 3.91 with specials) — flat enough that the
        // Zen2-vs-desktop kernel jitter (max |Δlogit| ~1.6 measured) flipped the
        // greedy token on-device ("User\n\n<|end|>" instead of a greeting).
        int32_t n_tokens =
            llama_tokenize(vocab, gp.prompt.c_str(), static_cast<int32_t>(gp.prompt.size()),
                           nullptr, 0, full_prompt, true);
        if (n_tokens == INT32_MIN) {
            res.error_msg = "tokenize overflow";
            return res;
        }
        // Size query returns the negated required count (see llama.h) — not an error.
        n_tokens = -n_tokens;

        std::vector<llama_token> tokens(static_cast<size_t>(n_tokens));
        n_tokens = llama_tokenize(vocab, gp.prompt.c_str(), static_cast<int32_t>(gp.prompt.size()),
                                  tokens.data(), n_tokens, full_prompt, true);
        if (n_tokens < 0) {
            res.error_msg = "tokenize failed";
            return res;
        }
        tokens.resize(static_cast<size_t>(n_tokens));

        // #170a: on a full prompt, rewind the resident KV to the common token
        // prefix and prefill only the divergent tail. Capped at size-1 so at
        // least one token is prefilled — the last token's logits seed the first
        // sample. Regenerate keeps everything but the final token; an edited
        // last message keeps everything before the edit; an unrelated prompt
        // diverges immediately and degrades to the old full clear.
        llama_memory_t mem = llama_get_memory(ctx);
        size_t kv_keep = 0;
        if (full_prompt) {
            const size_t max_keep = tokens.empty() ? 0 : tokens.size() - 1;
            while (kv_keep < m_kv_tokens.size() && kv_keep < max_keep &&
                   m_kv_tokens[kv_keep] == tokens[kv_keep])
                ++kv_keep;
            // Plan 009 diagnostic: `kv_reuse_diag.txt` containing `1` logs
            // the common-prefix seam (resident vs prompt token ids + pieces
            // + prompt tail) for every full-prompt turn. Default off;
            // log-only — the reuse decision below is untouched. Absolute
            // LocalState path: the process CWD is not LocalState.
            {
                FILE* df = open_state_file(resolve_local_path("kv_reuse_diag.txt"), "rb");
                bool diag = false;
                if (df) {
                    diag = std::fgetc(df) == '1';
                    std::fclose(df);
                }
                if (diag) {
                    std::string d =
                        "[xllama] session: reuse-diag common=" + std::to_string(kv_keep) +
                        " resident=" + std::to_string(m_kv_tokens.size()) +
                        " prompt=" + std::to_string(tokens.size());
                    auto show = [&](const char* tag, const std::vector<llama_token>& v, size_t from,
                                    size_t count) {
                        d += tag;
                        for (size_t i = from; i < from + count && i < v.size(); ++i) {
                            char tmp[64] = {};
                            llama_token_to_piece(vocab, v[i], tmp, sizeof(tmp) - 1, 0, true);
                            for (char* c = tmp; *c; ++c)
                                if (*c < 0x20 || *c > 0x7e)
                                    *c = '.';
                            d += " " + std::to_string(v[i]) + ":'" + tmp + "'";
                        }
                    };
                    const size_t r0 = kv_keep >= 3 ? kv_keep - 3 : 0;
                    show(" resident_tail", m_kv_tokens, r0, kv_keep + 3 - r0);
                    const size_t p0 = kv_keep >= 3 ? kv_keep - 3 : 0;
                    show(" prompt_at_seam", tokens, p0, 6);
                    const size_t tail = gp.prompt.size() > 100 ? gp.prompt.size() - 100 : 0;
                    std::string pt = gp.prompt.substr(tail);
                    for (char& c : pt)
                        c = (c == '\n') ? '|' : ((c < 0x20 || c > 0x7e) ? '.' : c);
                    d += " prompt_tail='" + pt + "'\n";
                    log_output(d.c_str());
                }
            }
            if (kv_keep > 0) {
                // Pure extension (no divergent tail) needs no removal. When a
                // tail must go, honor seq_rm's result: hybrid caches (LFM2's
                // attn+recurrent) refuse a partial tail erase — the recurrent
                // state cannot be rewound — and mutate NOTHING on failure, so
                // decoding on top would corrupt the output. Degrade to a full
                // clear + re-prefill instead.
                if (kv_keep < m_kv_tokens.size() &&
                    !llama_memory_seq_rm(mem, 0, static_cast<llama_pos>(kv_keep), -1)) {
                    char rb[256];
                    snprintf(rb, sizeof(rb),
                             "[xllama] session: KV rewind unsupported (hybrid cache) — "
                             "full re-prefill (#170): common=%zu resident=%zu prompt=%zu\n",
                             kv_keep, m_kv_tokens.size(), tokens.size());
                    log_output(rb);
                    llama_memory_clear(mem, true);
                    kv_keep = 0;
                } else {
                    char pb[128];
                    snprintf(pb, sizeof(pb),
                             "[xllama] session: KV prefix reuse — kept %zu of %zu tokens (#170)\n",
                             kv_keep, tokens.size());
                    log_output(pb);
                }
            } else {
                // #216: when a #170b restore just landed, "restored" + full
                // re-prefill means the saved token list and the re-rendered
                // prompt share no common prefix — log enough to diagnose.
                if (!m_kv_tokens.empty()) {
                    char pb[192];
                    snprintf(pb, sizeof(pb),
                             "[xllama] session: KV prefix reuse none — resident %zu tok, "
                             "prompt %zu tok (first mismatch at 0) (#170)\n",
                             m_kv_tokens.size(), tokens.size());
                    log_output(pb);
                }
                llama_memory_clear(mem, true);
            }
            // Plan 003 F3.5: a real rewind leaves the mirror without the hidden row
            // the divergent tail needs, and those kept rows cannot be rebuilt from
            // a different prefix. Re-prefill the whole prompt in the target too:
            // mirror and target then both start at 0, the prefill catch-up rebuilds
            // them from validated state, and MTP stays active instead of being
            // dropped. Only reached when the mirror cannot present the carry at
            // |kv_keep|; a full clear (kv_keep == 0) needs none of this.
            if (m_mtp_drafter && m_mtp_drafter->ready() && kv_keep > 0 &&
                !m_mtp_drafter->can_carry_from(static_cast<llama_pos>(kv_keep))) {
                // Plan 009 note: the committed carry (MtpDrafter::commit_carry,
                // installed at the previous turn end) is what normally avoids
                // this clear; reaching here means no valid committed carry
                // (divergent tail, empty/fresh mirror, post-failure, or a turn
                // that never established one). Fall through to the clear.
                llama_memory_clear(mem, true);
                m_kv_tokens.clear();
                kv_keep = 0;
                log_output("[xllama] mtp: draft mirror cannot carry the reused prefix; "
                           "full re-prefill so the catch-up rebuilds it\n");
            }

            m_kv_tokens.resize(kv_keep);

            // Plan 003 F3.5: the private draft mirror must follow the target's
            // rewind. The prefill catch-up below replays the divergent tail
            // starting at |kv_keep|; when the mirror still holds the discarded
            // prefix, that replay targets positions it already occupies and
            // llama_decode refuses it, which used to disable drafting for the
            // rest of the session. Aligning here keeps the positions and the
            // carry consistent, and lets the catch-up rebuild the mirror from
            // the new prefix instead of dropping the drafter.
            if (m_mtp_drafter && m_mtp_drafter->ready() &&
                !m_mtp_drafter->align_to(static_cast<llama_pos>(kv_keep))) {
                log_output("[xllama] mtp: private draft context could not follow the KV "
                           "rewind; drafting disabled\n");
                m_mtp_drafter.reset();
            }
        }

        // #173: the KV length is exact and free to read (seq_pos_max is -1 on
        // an empty cache). Predict the overflow — KV + delta + at least one
        // generated token — and fail before the decode attempt; the caller
        // retries with the trimmed full prompt. The same arithmetic clamps the
        // generation loop below so hitting the context end is a clean stop,
        // not a mid-generation decode failure.
        int kv_len = full_prompt ? static_cast<int>(kv_keep)
                                 : static_cast<int>(llama_memory_seq_pos_max(mem, 0)) + 1;
        // The slice that is not yet resident: the divergent tail on a full
        // prompt, the whole delta on a continuation.
        llama_token* pf = tokens.data() + (full_prompt ? kv_keep : 0);
        const int n_pf = static_cast<int>(tokens.size()) - (full_prompt ? (int)kv_keep : 0);
        {
            // Termination-state gates (plan 003): one line disambiguating
            // prefix-reuse (pf = tail) from continuation-append (pf = whole
            // prompt) — the ambiguity that made a stop-resume verdict
            // unreadable. Log-only.
            char tl[160];
            snprintf(tl, sizeof(tl),
                     "[xllama] session: turn full=%d kv_before=%d pf=%d prompt=%d\n",
                     full_prompt ? 1 : 0, kv_len, n_pf, static_cast<int>(tokens.size()));
            log_output(tl);
        }
        if (n_pf <= 0) {
            // Both callers always render at least one token (a delta carries the
            // turn close + user turn; a full prompt keeps size-1 at most), so
            // this is a programming error, not a user-reachable state. Fail
            // rather than sample on whatever logits the last decode left.
            res.error_msg = "nothing to prefill";
            log_output("[xllama] session generate: nothing to prefill\n");
            return res;
        }
        if (full_prompt && kv_len + n_pf + 1 > m_n_ctx) {
            // A full prompt has nothing older to evict — the UI trimmer
            // (kMaxPromptTokens / max_prompt_tokens_for_n_ctx) owns that, and
            // its char-based estimate can undershoot on dense code or CJK.
            // Fail here: reaching llama_decode with more tokens than the
            // context is not an error code but a GGML_ASSERT abort.
            res.error_msg = "prompt too long: " + std::to_string(n_pf) +
                            " tokens exceed n_ctx=" + std::to_string(m_n_ctx) +
                            " — shorten the message or start a new chat";
            log_output(("[xllama] session generate: " + res.error_msg + "\n").c_str());
            return res;
        }
        if (!full_prompt && kv_len + n_pf + 1 > m_n_ctx) {
            // #169: context shift — evict the oldest tokens past the pinned
            // head (gp.n_keep, the system prompt) and RoPE-shift the survivors
            // down, so a long chat stays in the reuse regime instead of
            // falling back to trimmed ~n_ctx re-prefills every turn. Eviction
            // is upstream-style (server-context.cpp): at least half the
            // movable region, or more when the delta + requested generation
            // need it. The recurrent half of a hybrid cache holds no cells in
            // the evicted range (its absorbed history survives — a documented
            // approximation), so the front-drop seq_rm succeeds where #170a's
            // tail rewind cannot.
            bool shifted = false;
            if (m_can_shift && kv_len > 0 && static_cast<int>(m_kv_tokens.size()) == kv_len) {
                const int keep = std::min(std::max(gp.n_keep, 0), kv_len - 1);
                const int movable = kv_len - keep;
                const int need = kv_len + n_pf + std::max(gp.n_predict, 1) - m_n_ctx;
                const int n_discard = std::min(std::max(need, movable / 2), movable);
                if (n_discard > 0 && kv_len + n_pf + 1 - n_discard <= m_n_ctx &&
                    llama_memory_seq_rm(mem, 0, keep, keep + n_discard)) {
                    llama_memory_seq_add(mem, 0, keep + n_discard, -1, -n_discard);
                    m_kv_tokens.erase(m_kv_tokens.begin() + keep,
                                      m_kv_tokens.begin() + keep + n_discard);
                    kv_len -= n_discard;
                    shifted = true;
                    char sb[160];
                    snprintf(sb, sizeof(sb),
                             "[xllama] session: context shift — evicted %d tokens past "
                             "keep=%d, kv now %d of %d (#169)\n",
                             n_discard, keep, kv_len, m_n_ctx);
                    log_output(sb);
                }
            }
            if (!shifted) {
                // #173 fail-fast: the caller retries with the trimmed full
                // prompt. Reached when the cache cannot shift, the bookkeeping
                // is out of sync, or even a full eviction would not fit.
                res.error_msg = "context full: kv=" + std::to_string(kv_len) +
                                " + delta=" + std::to_string(n_pf) +
                                " exceeds n_ctx=" + std::to_string(m_n_ctx);
                log_output(("[xllama] session generate: " + res.error_msg + "\n").c_str());
                return res;
            }
        }

        // Prefill: positions continue automatically from the KV cache end, so a
        // reuse turn appends after the retained context.
        //
        // Chunked at llama_n_batch: llama_decode does not return an error for an
        // oversized batch, it ASSERTS (GGML_ASSERT(n_tokens_all <= n_batch) in
        // llama-context.cpp → abort, Release included). n_batch defaults to
        // min(n_ctx, 2048) while the trimmer ceiling of a 4096-token coding
        // session is 3846, so a long paste used to kill the process. This is the
        // LOGICAL batch only — the physical ubatch stays 512 (#172 optimum),
        // which is what the prefill rate was measured on.
        const auto t_prefill0 = std::chrono::steady_clock::now();
        // Plan 003 stage 2: snapshot the d3d12 counters around THIS prefill. The
        // lifetime counters are zeroed in backend_free, so they cannot attribute
        // time to a phase; a delta around the real prefill can. No extra decode and
        // no state change — only observation, and only when profiling is on.
        const bool profile_t = gp.profile_phases && m_ctx != nullptr;
        detail::ScopeTag prefill_scope("target", "prefill");
        // The per-shape histogram builds strings inside graph_compute, so it has
        // its own switch tied to the same bench knob: OFF means no aggregation work
        // in the hot path at all, which is what the cost arm must measure.
        d3d12_set_shape_log(!gp.profile_phases);
        const std::uint64_t pf_calls0 = profile_t ? d3d12_graph_calls() : 0;
        const std::uint64_t pf_mm0 = profile_t ? d3d12_matmul_count() : 0;
        const double pf_gpu0 = profile_t ? d3d12_gpu_ms() : 0.0;
        // MTP catch-up (plan 003, F2): replay each prefill chunk into the
        // private draft context, right after the target decoded it (the nextn
        // buffer describes only the last batch). The drafter's own n_batch is
        // n_max+2, so the replay is chunked inside process().
        int mtp_prefill_rows = 0;
        const int pf_rows = prefill_chunked(
            ctx, pf, n_pf,
            [&](int off, int n_rows, llama_pos pos0) {
                if (m_mtp_drafter && m_mtp_drafter->ready()) {
                    if (!detail::mtp_catchup_batch(m_mtp_drafter.get(), ctx, pf + off, n_rows, pos0,
                                                   llama_model_n_embd_out(m_model.get()),
                                                   gp.profile_phases)) {
                        log_output("[xllama] mtp: session prefill catch-up failed; drafting "
                                   "disabled\n");
                        m_mtp_drafter.reset();
                    }
                }
            },
            gp.profile_phases);
        if (pf_rows < 0) {
            res.error_msg = "prompt decode failed";
            log_output("[xllama] session generate: prompt decode failed\n");
            // The cache now holds a partial batch — nothing about it is
            // trustworthy, so drop the record AND the cells. Clearing both keeps
            // a continuation turn (which reads the cache length, not
            // m_kv_tokens) from decoding on top of half a prefill.
            m_kv_tokens.clear();
            llama_memory_clear(mem, true);
            return res;
        }
        mtp_prefill_rows = pf_rows;

        if (profile_t) {
            log_output("[xllama] PHASE_PREFILL n_prompt=" + std::to_string(n_pf) +
                       " rows_kept=" + std::to_string(mtp_prefill_rows) +
                       " n_ubatch=" + std::to_string(llama_n_ubatch(ctx)) +
                       " calls=" + std::to_string(d3d12_graph_calls() - pf_calls0) +
                       " matmuls=" + std::to_string(d3d12_matmul_count() - pf_mm0) +
                       " gpu_ms=" + std::to_string(d3d12_gpu_ms() - pf_gpu0) + "\n");
        }

        res.t_p_eval_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_prefill0)
                .count();
        res.n_p_eval = n_pf;

        // The resident-token record now covers everything prefilled.
        if (full_prompt)
            m_kv_tokens.assign(tokens.begin(), tokens.end());
        else
            m_kv_tokens.insert(m_kv_tokens.end(), tokens.begin(), tokens.end());

        // Shared with run_inference — see src/bridge/sampler_chain.h and #125.
        // SamplingConfig::is_greedy() also covers temperature 0, where the full
        // chain must not run (the repetition penalty can flip the argmax).
        // #175: reuse the chain on a continuation turn with unchanged sampling,
        // so penalty/RNG state crosses turn boundaries exactly as the KV does.
        const SamplingConfig sc = gp.sampling();
        if (full_prompt || !m_sampler || !same_chain(m_sampler_cfg, sc)) {
            const llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
            m_sampler.reset(llama_sampler_chain_init(sparams));
            add_sampler_stages(m_sampler.get(), sc, vocab);
            m_sampler_cfg = sc;
        }
        llama_sampler* sampler_chain = m_sampler.get();

        // Plan 009 S3 probe: post-prefill state bytes, armed by
        // LocalState\chat-dump.flag (same flag as the logits capture). Dumps
        // the TARGET seq-0 KV blob, the MTP mirror's position extent and the
        // drafter pending row, so fresh vs post-restore runs can be compared
        // state-for-state and a logits difference can be attributed to state
        // provenance vs execution non-determinism. Fail-closed: any failure
        // logs and leaves generation untouched.
        {
            FILE* kf = open_state_file(resolve_local_path("chat-dump.flag"), "rb");
            if (kf) {
                std::fclose(kf);
                // Size the buffer with the fork's own query (get_data(nullptr,0)
                // returns 0 here — do NOT use it), then dump the target seq-0
                // state and the MTP mirror state, both post-prefill/pre-decode.
                size_t kv_bytes = 0, mirror_bytes = 0;
                try {
                    const size_t need = llama_state_seq_get_size(m_ctx.get(), 0);
                    if (need > 0) {
                        std::vector<uint8_t> kbuf(need);
                        kv_bytes =
                            llama_state_seq_get_data(m_ctx.get(), kbuf.data(), kbuf.size(), 0);
                        if (FILE* out = open_state_file(resolve_local_path("chat-dump.kv"), "wb")) {
                            const size_t w = std::fwrite(kbuf.data(), 1, kv_bytes, out);
                            std::fclose(out);
                            if (w != kv_bytes)
                                log_output("[xllama] WARN: S3 kv dump short write\n");
                        }
                    }
                    if (m_mtp_drafter && m_mtp_drafter->ready()) {
                        llama_context* dctx = m_mtp_drafter->ctx();
                        const size_t mneed = llama_state_seq_get_size(dctx, 0);
                        if (mneed > 0) {
                            std::vector<uint8_t> mbuf(mneed);
                            mirror_bytes =
                                llama_state_seq_get_data(dctx, mbuf.data(), mbuf.size(), 0);
                            if (FILE* out =
                                    open_state_file(resolve_local_path("chat-dump.mirror"), "wb")) {
                                const size_t w = std::fwrite(mbuf.data(), 1, mirror_bytes, out);
                                std::fclose(out);
                                if (w != mirror_bytes)
                                    log_output("[xllama] WARN: S3 mirror dump short write\n");
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    log_output(std::string("[xllama] WARN: S3 state dump failed: ") + e.what() +
                               "\n");
                }
                llama_pos mirror_max = -1;
                const float* pend = nullptr;
                int pend_n = 0;
                if (m_mtp_drafter && m_mtp_drafter->ready()) {
                    mirror_max = m_mtp_drafter->mirror_pos_max();
                    pend = m_mtp_drafter->pending_row_data();
                    pend_n = llama_model_n_embd_out(m_model.get());
                }
                if (FILE* out = open_state_file(resolve_local_path("chat-dump.mtp"), "wb")) {
                    char meta[224];
                    const int mn = std::snprintf(
                        meta, sizeof(meta),
                        "kv_tokens=%zu mirror_pos_max=%d pending_valid=%d n_embd=%d\n",
                        m_kv_tokens.size(), static_cast<int>(mirror_max), pend ? 1 : 0, pend_n);
                    std::fwrite(meta, 1, static_cast<size_t>(mn), out);
                    if (pend && pend_n > 0)
                        std::fwrite(pend, sizeof(float), static_cast<size_t>(pend_n), out);
                    const uint32_t ntok = static_cast<uint32_t>(m_kv_tokens.size());
                    std::fwrite(&ntok, sizeof(ntok), 1, out);
                    if (ntok)
                        std::fwrite(m_kv_tokens.data(), sizeof(llama_token), ntok, out);
                    std::fclose(out);
                }
                char lb[256];
                std::snprintf(lb, sizeof(lb),
                              "[xllama] S3: state dumped — kv_bytes=%zu mirror_bytes=%zu "
                              "mirror_pos=%d pending=%d kv_tokens=%zu\n",
                              kv_bytes, mirror_bytes, static_cast<int>(mirror_max), pend ? 1 : 0,
                              m_kv_tokens.size());
                log_output(lb);
            }
        }

        const auto t_decode0 = std::chrono::steady_clock::now();

        // Clean stop at the context end (#173): each generated token needs a KV
        // slot, so the loop must not outrun n_ctx - (kv_len + prompt). Clamped
        // at 0 — an oversized full prompt yields a clean zero-token result.
        const int n_predict_eff = std::max(0, std::min(gp.n_predict, m_n_ctx - kv_len - n_pf));

        // The loop itself lives in decode_loop.h, shared with run_inference —
        // see there for why. What stays here is what is genuinely this session's:
        // keeping m_kv_tokens in step with the KV cells, which the #170a prefix
        // diff and the #170b snapshot fingerprint both read.
        DecodeLoopParams dlp;
        dlp.ctx = ctx;
        dlp.sampler = sampler_chain;
        dlp.vocab = vocab;
        dlp.n_predict = n_predict_eff;
        dlp.stop_sequences = &gp.stop_sequences;
        dlp.abort_flag = gp.abort_flag;
        dlp.on_token = gp.on_token;
        dlp.on_accepted = [this](llama_token t) { m_kv_tokens.push_back(t); };
        dlp.decode_start = t_decode0;
        // W2: seed + live history is m_kv_tokens (prefill already recorded above).
        dlp.prompt_lookup = m_prompt_lookup;
        dlp.token_history = &m_kv_tokens;
        dlp.profile_phases = gp.profile_phases;
        // Leaving the prefill window: drain what the prefill graphs accumulated and
        // restore the tag so the decode-phase graphs are classified as decode.
        d3d12_set_scope("target", "decode");
        d3d12_shape_drain("session_prefill");
        if (m_mtp_drafter) {
            dlp.mtp = m_mtp_drafter.get();
            dlp.mtp_n_embd = llama_model_n_embd_out(m_model.get());
            dlp.mtp_prefill_rows = mtp_prefill_rows;
        }
        // Parity/diagnostic: collect this turn's accepted ids so a caller can
        // compare an integral sequence (plan 003 stage 1). Empty request = no
        // sink, no cost.
        std::vector<llama_token> turn_ids;
        std::string dump_tokens_path = gp.dump_tokens_path;
        if (!dump_tokens_path.empty())
            dlp.out_token_ids = &turn_ids;

        // Plan 009 S3 probe: LocalState\chat-dump.flag (presence arms it)
        // captures per-step logits bytes + sampled ids AND the per-turn token
        // ids, so a fresh run and a post-restore run can be compared byte for
        // byte. Default OFF: no flag, no file, no hot-path cost beyond the
        // existence check done once per turn here. Fail-closed: the capture
        // disables itself on any I/O failure inside emit_token.
        {
            FILE* cf = open_state_file(resolve_local_path("chat-dump.flag"), "rb");
            if (cf) {
                std::fclose(cf);
                const std::string lp = resolve_local_path("chat-dump.logits");
                std::remove(lp.c_str()); // one turn's capture per generate
                dlp.logits_dump_path = lp;
                if (dump_tokens_path.empty()) {
                    const std::string tp = resolve_local_path("chat-dump.tokens");
                    std::remove(tp.c_str());
                    dump_tokens_path = tp; // ids + .prefill ids of THIS turn
                    dlp.out_token_ids = &turn_ids;
                }
                log_output("[xllama] S3: chat-dump.flag armed -> " + lp + "\n");
            }
        }

        const DecodeLoopResult dlr = decode_loop(dlp, res.output_text);

        if (!dump_tokens_path.empty()) {
            if (FILE* tf = std::fopen(dump_tokens_path.c_str(), "w")) {
                for (llama_token t : turn_ids)
                    std::fprintf(tf, "%d\n", static_cast<int>(t));
                std::fclose(tf);
            } else {
                log_output("[xllama] WARN: per-turn token dump failed: " + dump_tokens_path + "\n");
            }
            // The ids THIS turn decoded. Together with the previous turns' dumps they
            // reconstruct the exact resident history, token for token — which is what
            // a cold reference has to reproduce. Comparing output ids alone cannot
            // prove that: text concatenation may retokenize at the boundaries.
            const std::string pf_path = dump_tokens_path + ".prefill";
            if (FILE* pf2 = std::fopen(pf_path.c_str(), "w")) {
                for (int i = 0; i < n_pf; ++i)
                    std::fprintf(pf2, "%d\n", static_cast<int>(pf[i]));
                std::fclose(pf2);
            }
        }
        const int n_generated = dlr.n_generated;
        const bool stopped_by_seq = dlr.ended_with_stop;
        res.eog_token = dlr.eog_token;
        res.eog_branch = dlr.eog_branch ? dlr.eog_branch : "";
        res.stop_branch = dlr.stop_branch ? dlr.stop_branch : "";
        res.stop_round = dlr.stop_round;

        // t_eval_ms KEEPS its published meaning (decode interval only): bench.cpp,
        // main.cpp and the CSV's decode_tok_s all divide n_eval by it, so widening
        // it to the whole generate() would silently fold prefill and load into
        // every throughput number. The generate-wide clock goes to t_total_ms.
        res.t_eval_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_decode0)
                .count();
        res.t_total_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_total0)
                .count();
        res.t_first_token_ms =
            dlr.first_token_ms > 0.0 ? res.t_p_eval_ms + dlr.first_token_ms : 0.0;
        res.n_eval = n_generated;
        res.ended_with_stop = stopped_by_seq;
        res.n_drafted = dlr.n_drafted;
        res.n_spec_accepted = dlr.n_accepted;
        // Per-source split: n_drafted alone cannot answer "did MTP propose".
        res.n_mtp_drafted = dlr.n_mtp_drafted;
        res.n_mtp_accepted = dlr.n_mtp_accepted;
        res.n_lookup_drafted = dlr.n_lookup_drafted;
        res.n_lookup_accepted = dlr.n_lookup_accepted;
        res.n_mtp_rounds = dlr.n_mtp_rounds;
        res.t_decode_ms = dlr.t_decode_ms;
        res.t_decode_accounted_ms = dlr.t_decode_accounted_ms;
        res.mtp_active = m_mtp_active;
        if (dlr.rewind_failed) {
            // History and KV are untrustworthy — same class as a decode failure.
            m_kv_tokens.clear();
            if (m_ctx)
                llama_memory_clear(llama_get_memory(m_ctx.get()), /*data=*/true);
            res.success = false;
            res.error_msg =
                "speculative KV rewind unsupported (disable prompt_lookup for this model)";
            log_output(("[xllama] " + res.error_msg + "\n").c_str());
            return res;
        }
        res.success = true;

        // Plan 009 turn-end drafter state (log-only): what the NEXT turn's
        // F3.5 can_carry check will see (pending carry + mirror end vs the
        // committed token count). Diagnoses carry staleness without touching
        // the reuse decision.
        if (m_mtp_drafter && m_mtp_drafter->ready()) {
            const llama_pos mirror_end =
                llama_memory_seq_pos_max(llama_get_memory(m_mtp_drafter->ctx()), 0);
            char db[224];
            snprintf(db, sizeof(db),
                     "[xllama] mtp: turn-end drafter state: pending_valid=%d pending_pos=%d "
                     "mirror_end=%d kv_tokens=%zu\n",
                     m_mtp_drafter->pending_valid() ? 1 : 0,
                     static_cast<int>(m_mtp_drafter->pending_pos()), static_cast<int>(mirror_end),
                     m_kv_tokens.size());
            log_output(db);
        }

        char log_buf[320];
        snprintf(log_buf, sizeof(log_buf),
                 "[xllama] session generate: n=%d prefill=%.1fms (%d tok) reuse=%d "
                 "drafted=%d (mtp=%d lookup=%d) spec_accept=%d (mtp=%d lookup=%d)\n",
                 n_generated, res.t_p_eval_ms, res.n_p_eval, gp.reuse_kv && !gp.reset_kv,
                 dlr.n_drafted, dlr.n_mtp_drafted, dlr.n_lookup_drafted, dlr.n_accepted,
                 dlr.n_mtp_accepted, dlr.n_lookup_accepted);
        log_output(log_buf);
        // Same-interval accounting: total covers this generate() call, decode is
        // the loop alone, and the phase timers sum to accounted. The three gaps
        // are named so nothing is hidden: load+setup, prefill, uninstrumented.
        {
            const double total = res.t_total_ms; // generate-wide, NOT the decode interval
            const double prefill = res.t_p_eval_ms;
            const double dec = dlr.t_decode_ms;
            const double acct = dlr.t_decode_accounted_ms;
            // Verify sub-cost split (plan003): sums of the per-round backend
            // snapshots. Units ms; nesting gpu <= d3w <= verify_ms keeps the
            // aggregates disjoint from verify_ms (never counted twice).
            // rounds_split=0 with profile OFF or without MTP verify rounds.
            double v_d3w = 0.0, v_d3g = 0.0, v_cpu = 0.0;
            for (size_t i = 0; i < dlr.vr_d3w.size(); ++i) {
                v_d3w += dlr.vr_d3w[i];
                v_d3g += dlr.vr_d3g[i];
                v_cpu += dlr.vr_cpu[i];
            }
            char tlog[640];
            snprintf(tlog, sizeof(tlog),
                     "[xllama] PHASE total_ms=%.1f prefill_ms=%.1f decode_ms=%.1f "
                     "accounted_ms=%.1f residual_decode_ms=%.1f load_plus_setup_ms=%.1f "
                     "draft_ms=%.1f sample_ms=%.1f topprob_ms=%.1f catchup_ms=%.1f "
                     "verify_ms=%.1f verify_d3w=%.1f verify_d3g=%.1f verify_cpu=%.1f "
                     "rounds_split=%zu "
                     "corrective_ms=%.1f classic_decode_ms=%.1f "
                     "classic_catchup_ms=%.1f lookup_ms=%.1f sample_target_ms=%.1f "
                     "emit_ms=%.1f maintain_ms=%.1f profile=%d\n",
                     total, prefill, dec, acct, dec - acct, total - prefill - dec, dlr.mtp_draft_ms,
                     dlr.mtp_sample_ms, dlr.mtp_top_prob_ms, dlr.mtp_catchup_ms, dlr.verify_ms,
                     v_d3w, v_d3g, v_cpu, dlr.vr_width.size(), dlr.corrective_ms, dlr.t_classic_ms,
                     dlr.t_classic_catchup_ms, dlr.t_lookup_ms, dlr.t_sample_target_ms,
                     dlr.t_emit_ms, dlr.t_maintain_ms, gp.profile_phases ? 1 : 0);
            log_output(tlog);
            // Reconciliation stated, not asserted: the phases sum to `accounted`,
            // the loop to `decode`, and the leftover is the instrumentation's own
            // residual plus any path not yet timed. It is printed so a reader can
            // see the gap instead of inferring coverage from the small numbers.
            if (dec - acct > 0.05 * dec && dec > 1.0) {
                char rlog[256];
                snprintf(rlog, sizeof(rlog),
                         "[xllama] PHASE_RESIDUAL decode_ms=%.1f accounted_ms=%.1f "
                         "unaccounted_ms=%.1f (%.1f%% of decode)\n",
                         dec, acct, dec - acct, 100.0 * (dec - acct) / dec);
                log_output(rlog);
            }
        }
        d3d12_shape_drain("session_decode");
        if (m_mtp) {
            char mlog[384];
            snprintf(mlog, sizeof(mlog),
                     "[xllama] MTP_STATS rounds=%d decodes=%d discarded=%d catchup_tok=%d"
                     " draft_ms=%.1f sample_ms=%.1f topprob_ms=%.1f catchup_ms=%.1f"
                     " verify_ms=%.1f corrective_ms=%.1f\n",
                     dlr.n_mtp_rounds, dlr.n_mtp_decodes, dlr.n_mtp_discarded, dlr.n_catchup_tokens,
                     dlr.mtp_draft_ms, dlr.mtp_sample_ms, dlr.mtp_top_prob_ms, dlr.mtp_catchup_ms,
                     dlr.verify_ms, dlr.corrective_ms);
            log_output(mlog);
        }

        return res;
    }

    int count_tokens(const std::string& prompt) override {
        const llama_vocab* vocab = llama_model_get_vocab(m_model.get());
        // parse_special=true to match generate(): counting template markers as
        // plain text overcounts a chat prompt by ~70% (50 vs 29 tokens on a
        // 2-turn ChatML render), which skews the routing threshold.
        int32_t n = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                   nullptr, 0, true, true);
        if (n == INT32_MIN)
            return 0;
        return -n;
    }

    EmbeddingResult embed(const EmbeddingParams& params) override {
        EmbeddingResult result;
        if (params.input.empty()) {
            result.error_msg = "input must not be empty";
            return result;
        }
        std::string ctx_error;
        if (!ensure_ctx(&ctx_error, true)) {
            result.error_msg = "failed to create embedding context: " + ctx_error;
            return result;
        }

        if (llama_model_has_encoder(m_model.get()) && llama_model_has_decoder(m_model.get())) {
            result.error_msg = "encoder-decoder models do not provide supported text embeddings";
            return result;
        }
        const llama_vocab* vocab = llama_model_get_vocab(m_model.get());
        int32_t n =
            llama_tokenize(vocab, params.input.data(), static_cast<int32_t>(params.input.size()),
                           nullptr, 0, true, true);
        if (n == INT32_MIN || n == 0) {
            result.error_msg = "input tokenization failed";
            return result;
        }
        std::vector<llama_token> tokens(static_cast<size_t>(-n));
        n = llama_tokenize(vocab, params.input.data(), static_cast<int32_t>(params.input.size()),
                           tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
        if (n <= 0) {
            result.error_msg = "input tokenization failed";
            return result;
        }
        tokens.resize(static_cast<size_t>(n));
        result.n_tokens = n;

        llama_context* ctx = m_ctx.get();
        const enum llama_pooling_type pooling = llama_pooling_type(ctx);
        if (pooling == LLAMA_POOLING_TYPE_RANK) {
            result.error_msg = "ranking GGUF models do not provide text embeddings";
            return result;
        }

        // Append EOS if the GGUF requests it and the sequence does not already end
        // in SEP or EOS. llama.cpp/examples/embedding/embedding.cpp warns when the
        // last token is neither. The tokenizer.ggml.add_eos_token flag is
        // authoritative; we append only when the tokenizer didn't already.
        const llama_token sep_token = llama_vocab_sep(vocab);
        const llama_token eos_token = llama_vocab_eos(vocab);
        if (llama_vocab_get_add_eos(vocab) && !tokens.empty()) {
            const llama_token last = tokens.back();
            if (last != sep_token && last != eos_token) {
                // Only append if it still fits in context after truncation.
                if (static_cast<int>(tokens.size()) < m_n_ctx) {
                    tokens.push_back(eos_token);
                }
            }
        }

        const int max_input_tokens = std::min(m_n_ctx, static_cast<int>(llama_n_batch(ctx)));
        if (static_cast<int>(tokens.size()) > max_input_tokens) {
            if (!params.truncate) {
                result.error_msg = "input exceeds embedding batch/context limit of " +
                                   std::to_string(max_input_tokens) +
                                   " tokens and truncate is false";
                return result;
            }
            // Trim to fit, keeping the tokens that matter for this pooling type.
            const PoolingType pool = static_cast<PoolingType>(static_cast<int>(pooling));
            trim_tokens_for_pooling(tokens, pool, max_input_tokens);
        }
        result.n_tokens = static_cast<int>(tokens.size());

        // Embedding requests share this context with chat. Drop any chat KV and
        // invalidate its token mirror; the next chat turn will prefill cleanly.
        m_kv_tokens.clear();
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_set_embeddings(ctx, true);
        struct EmbeddingContextReset {
            llama_context* ctx;
            ~EmbeddingContextReset() {
                llama_memory_clear(llama_get_memory(ctx), true);
                llama_set_embeddings(ctx, false);
            }
        } context_reset{ctx};

        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
        if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            result.error_msg = "failed to allocate embedding batch";
            return result;
        }
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = tokens[static_cast<size_t>(i)];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == batch.n_tokens - 1) ? 1 : 0;
        }
        const int32_t rc = llama_model_has_encoder(m_model.get()) ? llama_encode(ctx, batch)
                                                                  : llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            result.error_msg =
                "llama.cpp failed to encode the embedding input (code " + std::to_string(rc) + ")";
            return result;
        }

        const int n_embd = llama_model_n_embd_out(m_model.get());
        const float* values = pooling == LLAMA_POOLING_TYPE_NONE
                                  ? llama_get_embeddings_ith(ctx, batch.n_tokens - 1)
                                  : llama_get_embeddings_seq(ctx, 0);
        if (!values || n_embd <= 0) {
            result.error_msg = "model/backend did not produce a sequence embedding";
            return result;
        }
        result.embedding.assign(values, values + n_embd);
        if (!normalize_embedding(result.embedding, params.dimensions, &result.error_msg))
            return result;
        result.success = true;
        return result;
    }

    int context_length() const override {
        return m_n_ctx;
    }

    bool can_context_shift() const override {
        return m_can_shift;
    }

    // --- #170b: KV state files ---------------------------------------------
    //
    // Layout: one text header line, then the resident token ids, then the raw
    // sequence state. Everything the pin does NOT check lives in the header —
    // it validates cache shape only, so a re-quantized build of the same model
    // or the same model with a LoRA applied would load this file and generate
    // silent garbage.

    static constexpr const char* kStateMagic = "xllama-kv";
    // Plan 009 complete-state trailer (additive; a v2 reader stops at the
    // target blob and never looks here). Carries the private draft KV + the
    // real pending row + the mirror extent + a prefix hash that proves the
    // mirror's rows correspond to the restored tokens. Refused unless every
    // field validates; anything else falls back to the v2 prime path.
    static constexpr const char* kMtpMagic = "XLMTPST1"; // 8 bytes, no NUL
    static constexpr uint32_t kMtpTrailerVersion = 1;
    // v3 binds the resolved RoPE/YaRN parameters with exact float identity.
    // Older v1/v2 snapshots fail closed before native state installation.
    static constexpr int kStateVersion = 3;
    // AppContainer dislikes very large single reads: ORT's 1 GB
    // ReadFileIntoBuffer had to come down to 16 MB before errcode 1450
    // (ERROR_NO_SYSTEM_RESOURCES) stopped appearing (uwp-constraints §9), and
    // llama_file's own path would move this blob in 64 MB slices. Stay well
    // under the known-good bound.
    static constexpr size_t kIoChunk = 8u << 20;
    // A sane ceiling for a declared blob size: refuse to allocate for a
    // corrupt or hostile header instead of trying and dying.
    static constexpr size_t kMaxStateBytes = 512u << 20;

    static FILE* open_state_file(const std::string& path, const char* mode) {
    #ifdef XLLAMA_UWP
        const std::string m(mode);
        return _wfopen(utf8_to_wstring(path).c_str(), std::wstring(m.begin(), m.end()).c_str());
    #else
        return std::fopen(path.c_str(), mode);
    #endif
    }

    static bool io_chunked(FILE* fp, void* data, size_t n, bool write,
                           size_t* transferred = nullptr) {
        if (transferred)
            *transferred = 0;
        auto* p = static_cast<uint8_t*>(data);
        for (size_t off = 0; off < n;) {
            const size_t want = std::min(kIoChunk, n - off);
            const size_t got =
                write ? std::fwrite(p + off, 1, want, fp) : std::fread(p + off, 1, want, fp);
            if (transferred)
                *transferred += got;
            if (got != want)
                return false;
            off += want;
        }
        return true;
    }

    // TEST-ONLY mid-write fault probe (plan 009 / T18, authorized msg 1414).
    // Armed by the fixed-name LOCALSTATE file "slot-fault.target.txt" whose
    // content is "<target-basename> <bytes> <nonce>". OFF by default: no
    // file, a malformed/legacy knob, or any failed match returns 0 with zero
    // behaviour change. The knob is opened read-only on every *.kv save; the
    // nonce AND the slot basename are verified BEFORE the atomic claim, so a
    // save with no nonce, a divergent nonce, or another slot never claims,
    // never consumes, and never fires the arm. A pre-existing ".claimed"
    // blocks the fire fail-closed (knob untouched, save proceeds normally):
    // it belongs to a previous round and is never replaced or deleted here,
    // and destination-replace semantics differ per OS, so the claim never
    // depends on them. The claim is a single rename() — of any number of
    // concurrent matching saves exactly one wins and only that one injects.
    // Returns the fault offset to write (strictly inside the blob) or 0 = off.
    static size_t probe_fault_bytes_for(const std::string& path, const std::string& nonce,
                                        size_t n_state) {
        if (nonce.empty() || n_state == 0)
            return 0; // no request identity -> never match, knob untouched
        const size_t slash = path.find_last_of("/\\");
        const std::string target = slash == std::string::npos ? path : path.substr(slash + 1);
        if (target.size() < 4 || target.compare(target.size() - 3, 3, ".kv") != 0)
            return 0; // *.tmp and friends are disarmed before any I/O
        const std::string knob = resolve_local_path("slot-fault.target.txt");
        FILE* k = open_state_file(knob, "rb"); // read-only; the read itself is harmless
        if (!k)
            return 0;
        char b[256] = {};
        const size_t n = std::fread(b, 1, sizeof(b) - 1, k);
        std::fclose(k);
        if (n == 0)
            return 0;
        // arm layout: exactly three whitespace-separated tokens
        // "<target-basename> <bytes> <nonce>" and nothing else — a missing
        // token, a fourth token, or a second line never arms (plain buffers,
        // no sscanf into std::string, no format-string pitfalls on MSVC).
        const char* cur = b;
        auto next_token = [&cur](char* out, size_t cap) -> bool {
            while (*cur == ' ' || *cur == '\t' || *cur == '\r' || *cur == '\n')
                ++cur;
            if (*cur == '\0')
                return false;
            size_t i = 0;
            while (*cur != '\0' && *cur != ' ' && *cur != '\t' && *cur != '\r' && *cur != '\n') {
                if (i + 1 >= cap)
                    return false;
                out[i++] = *cur++;
            }
            out[i] = '\0';
            return true;
        };
        char armed_target[80] = {}, bytes_txt[80] = {}, armed_nonce[80] = {};
        if (!next_token(armed_target, sizeof(armed_target)) ||
            !next_token(bytes_txt, sizeof(bytes_txt)) ||
            !next_token(armed_nonce, sizeof(armed_nonce)))
            return 0;
        while (*cur == ' ' || *cur == '\t' || *cur == '\r' || *cur == '\n')
            ++cur;
        if (*cur != '\0')
            return 0; // trailing junk or a second line: never arms
        if (nonce != armed_nonce)
            return 0; // divergent nonce -> no claim, arm kept
        if (target != armed_target)
            return 0; // other slot -> no claim, arm kept
        for (const char* nc = armed_nonce; *nc; ++nc) {
            const bool okc = (*nc >= 'a' && *nc <= 'z') || (*nc >= 'A' && *nc <= 'Z') ||
                             (*nc >= '0' && *nc <= '9') || *nc == '-' || *nc == '_';
            if (!okc)
                return 0;
        }
        size_t v = 0;
        for (const char* c = bytes_txt; *c; ++c) {
            if (*c < '0' || *c > '9')
                return 0;
            const size_t d = static_cast<size_t>(*c - '0');
            // Rejected BEFORE the multiply: v*10+d <= kMaxStateBytes is
            // enforced exactly, so a long decimal string can never overflow
            // size_t — and the check happens pre-claim, so a malformed arm
            // is neither fired nor consumed.
            if (v > (kMaxStateBytes - d) / 10)
                return 0;
            v = v * 10 + d;
        }
        if (v == 0 || v >= n_state)
            return 0; // must land strictly inside the blob; checked pre-claim
        // FAIL-CLOSED on a pre-existing claim marker: a ".claimed" left by a
        // previous round belongs to that round — never replace or delete it
        // here, and never fire into it. Destination-replace semantics differ
        // per OS (POSIX rename(2) replaces; the UWP STL behavior is not
        // provable from the Linux host), so the claim must not depend on
        // them: with a marker standing, this save proceeds normally, the arm
        // stays armed and untouched, and exactly-once per arm is preserved on
        // every platform. The device runbook clears ".claimed" before
        // re-arming; nothing here re-arms or consumes while a marker stands.
        std::error_code ec1;
        if (std::filesystem::exists(knob + ".claimed", ec1))
            return 0;
        // ATOMIC CLAIM: rename is one namespace operation (POSIX rename(2),
        // MoveFileEx on NTFS) — of concurrent matching saves exactly one
        // moves the knob; every other contender fails here and injects
        // nothing. Non-matching saves never reach this line.
        std::error_code ec0;
        std::filesystem::rename(knob, knob + ".claimed", ec0);
        if (ec0)
            return 0; // lost the claim (or stale/absent knob)
        return v;
    }

    // Target context identity; the MTP trailer validates its own state.
    // RoPE/YaRN values come from the resolved context, not rounded metadata.
    // This is configuration identity, not a cryptographic weight checksum.
    std::string state_fingerprint() const {
        char desc[192] = {};
        llama_model_desc(m_model.get(), desc, sizeof(desc) - 1);
        const int type_k =
            static_cast<int>(m_kv_q8 ? GGML_TYPE_Q8_0 : llama_context_default_params().type_k);
        const int type_v =
            static_cast<int>(m_kv_q8 ? GGML_TYPE_Q8_0 : llama_context_default_params().type_v);
        const auto rope = state_rope_identity(llama_get_context_rope_params(m_ctx.get()));
        char buf[768];
        snprintf(buf, sizeof(buf),
                 "v%d n_ctx=%d n_seq_max=%d kv_q8=%d type_k=%d type_v=%d rope_type=%d "
                 "%s fa_query=%d lora=%08x params=%llu %s",
                 kStateVersion, m_n_ctx, m_n_seq_max, m_kv_q8 ? 1 : 0, type_k, type_v,
                 static_cast<int>(llama_model_rope_type(m_model.get())), rope.c_str(),
                 m_fa_query_policy, state_float_bits(m_adapter ? m_lora_scale : 0.0f),
                 static_cast<unsigned long long>(llama_model_n_params(m_model.get())), desc);
        return buf;
    }

    // FNV-1a 64 over a byte range — the prefix proof for the MTP trailer.
    static uint64_t fnv1a(const void* data, size_t n) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        uint64_t h = 1469598103934665603ULL;
        for (size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

    // Plan 009: append the drafter trailer into an OPEN tmp file (already
    // holding header+tokens+target blob). Returns false on any write
    // failure so the caller removes the tmp instead of shipping a silently
    // v2 file. No trailer when the drafter is absent (legit v2 file).
    bool write_mtp_trailer(FILE* fp, const std::vector<llama_token>& tokens, bool* wrote) {
        *wrote = false;
        if (!m_mtp_drafter || !m_mtp_drafter->ready())
            return true; // MTP off: a v2 file is the honest artifact
        const llama_pos pmin = m_mtp_drafter->mirror_pos_min();
        const llama_pos pmax = m_mtp_drafter->mirror_pos_max();
        if (pmin < 0 || pmax < pmin)
            return false;
        const uint32_t pos_min = static_cast<uint32_t>(pmin);
        const uint32_t cells = static_cast<uint32_t>(pmax - pmin + 1);
        if (pos_min > tokens.size() || cells != tokens.size() - pos_min)
            return false; // mirror extent must mirror the resident tokens
        const float* pend = m_mtp_drafter->pending_row_data();
        const llama_pos ppos = m_mtp_drafter->pending_pos();
        if (!pend || ppos != static_cast<llama_pos>(tokens.size()) - 1)
            return false; // the carry must describe the last token
        std::vector<uint8_t> draft;
        if (!m_mtp_drafter->get_state(draft) || draft.empty())
            return false;
        const uint32_t n_embd = static_cast<uint32_t>(llama_model_n_embd_out(m_model.get()));
        const uint64_t prefix_hash =
            fnv1a(tokens.data() + pos_min, (tokens.size() - pos_min) * sizeof(llama_token));
        const uint32_t pending_pos = static_cast<uint32_t>(ppos);
        const uint64_t draft_size = draft.size();
        std::vector<float> pend_row(pend, pend + n_embd); // io_chunked needs void*
        bool ok = std::fwrite(kMtpMagic, 1, 8, fp) == 8;
        ok = ok && std::fwrite(&kMtpTrailerVersion, sizeof(kMtpTrailerVersion), 1, fp) == 1;
        ok = ok && std::fwrite(&pos_min, sizeof(pos_min), 1, fp) == 1;
        ok = ok && std::fwrite(&cells, sizeof(cells), 1, fp) == 1;
        ok = ok && std::fwrite(&draft_size, sizeof(draft_size), 1, fp) == 1;
        ok = ok && io_chunked(fp, draft.data(), draft.size(), /*write=*/true);
        ok = ok && std::fwrite(&pending_pos, sizeof(pending_pos), 1, fp) == 1;
        ok = ok && std::fwrite(&n_embd, sizeof(n_embd), 1, fp) == 1;
        ok = ok && io_chunked(fp, pend_row.data(), static_cast<size_t>(n_embd) * sizeof(float),
                              /*write=*/true);
        ok = ok && std::fwrite(&prefix_hash, sizeof(prefix_hash), 1, fp) == 1;
        *wrote = ok;
        return ok;
    }

    bool save_state(const std::string& path, std::string* err,
                    const std::string& request_nonce) override {
        if (!ensure_ctx(err))
            return false;
        if (m_kv_tokens.empty()) {
            if (err)
                *err = "no resident KV to save";
            return false;
        }

        std::vector<uint8_t> blob;
        size_t n_state = 0;
        try {
            blob.resize(llama_state_seq_get_size(m_ctx.get(), 0));
            n_state = llama_state_seq_get_data(m_ctx.get(), blob.data(), blob.size(), 0);
        } catch (const std::exception& e) {
            if (err)
                *err = std::string("state_seq_get_data threw: ") + e.what();
            return false;
        }
        if (n_state == 0) {
            if (err)
                *err = "llama_state_seq_get_data failed";
            return false;
        }

        // Write to a sibling temp and rename: a half-written file is tens of
        // MB of garbage that the loader can only reject after reading it all,
        // and a crash mid-write must not leave one behind.
        const std::string tmp = path + ".tmp";
        FILE* fp = open_state_file(tmp, "wb");
        if (!fp) {
            if (err)
                *err = "cannot open " + tmp;
            return false;
        }
        char hdr[1024];
        const int hn = snprintf(hdr, sizeof(hdr), "%s\t%zu\t%zu\t%s\n", kStateMagic, n_state,
                                m_kv_tokens.size(), state_fingerprint().c_str());
        bool ok = hn > 0 && static_cast<size_t>(hn) < sizeof(hdr) &&
                  std::fwrite(hdr, 1, static_cast<size_t>(hn), fp) == static_cast<size_t>(hn);
        ok = ok && io_chunked(fp, m_kv_tokens.data(), m_kv_tokens.size() * sizeof(llama_token),
                              /*write=*/true);
        // T18 fault probe (test-only): claim only after a healthy header +
        // token write and only on an exact nonce+slot match; a losing or
        // non-matching save writes the whole blob as always.
        size_t fault = 0;
        if (ok)
            fault = probe_fault_bytes_for(path, request_nonce, n_state);
        if (fault > 0) {
            // Persist only up to the armed offset, then take the normal
            // failure path below (remove tmp, false -> HTTP 400). The final
            // <id>.kv is only ever touched by the rename, which is skipped.
            size_t written = 0;
            const bool write_ok = io_chunked(fp, blob.data(), fault, true, &written);
            const int flush_result = std::fflush(fp);
            std::error_code size_error;
            const auto observed_size = std::filesystem::file_size(tmp, size_error);
            log_output("[xllama] session: fault probe — abort save after " + std::to_string(fault) +
                       " bytes (test-only); fwrite_bytes=" + std::to_string(written) +
                       " write_ok=" + std::to_string(write_ok) +
                       " flush_rc=" + std::to_string(flush_result) + " tmp_size=" +
                       (size_error ? std::string("unavailable") : std::to_string(observed_size)) +
                       " expected_size=" +
                       std::to_string(static_cast<size_t>(hn) +
                                      m_kv_tokens.size() * sizeof(llama_token) + written) +
                       " (observed after fflush; not fsync durability)\n");
            ok = false;
        } else {
            ok = ok && io_chunked(fp, blob.data(), n_state, /*write=*/true);
        }
        bool wrote_trailer = false;
        ok = ok && write_mtp_trailer(fp, m_kv_tokens, &wrote_trailer);
        ok = (std::fclose(fp) == 0) && ok;

        std::error_code ec;
        if (!ok) {
            std::filesystem::remove(tmp, ec);
            if (err)
                *err = "write failed (disk full?)";
            return false;
        }
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            std::filesystem::remove(tmp, ec);
            if (err)
                *err = "rename failed: " + ec.message();
            return false;
        }
        char lb[192];
        snprintf(lb, sizeof(lb),
                 "[xllama] session: KV state saved — %zu tokens, %.1f MB%s (#170b)\n",
                 m_kv_tokens.size(), (double)n_state / (1024.0 * 1024.0),
                 wrote_trailer ? " + MTP mirror trailer" : " (no MTP trailer: drafter absent)");
        log_output(lb);
        return true;
    }

    // Plan 009: try to restore the drafter state from the slot trailer. The
    // file pointer sits right after the target blob. Returns true only when
    // EVERY field validates and the draft context accepted the blob; any
    // failure logs the reason and returns false so the caller keeps the v2
    // behavior (clear + prime) instead of claiming a mirror it does not have.
    bool try_restore_mtp_trailer(FILE* fp, const std::vector<llama_token>& tokens) {
        auto refuse = [](const char* why) {
            log_output(std::string("[xllama] MTP trailer refused (") + why +
                       ") — falling back to v2 restore (clear + prime)\n");
            return false;
        };
        if (!m_mtp_drafter || !m_mtp_drafter->ready())
            return false; // nothing to restore into; v2 semantics
        char magic[8] = {};
        if (std::fread(magic, 1, 8, fp) != 8)
            return false; // clean EOF: v2 file without a trailer
        if (std::memcmp(magic, kMtpMagic, 8) != 0)
            return refuse("bad magic");
        uint32_t version = 0, pos_min = 0, cells = 0, pending_pos = 0, n_embd = 0;
        uint64_t draft_size = 0, stored_hash = 0;
        bool ok = std::fread(&version, sizeof(version), 1, fp) == 1;
        ok = ok && std::fread(&pos_min, sizeof(pos_min), 1, fp) == 1;
        ok = ok && std::fread(&cells, sizeof(cells), 1, fp) == 1;
        ok = ok && std::fread(&draft_size, sizeof(draft_size), 1, fp) == 1;
        if (!ok)
            return refuse("header truncated");
        if (version != kMtpTrailerVersion)
            return refuse("version");
        if (draft_size == 0 || draft_size > kMaxStateBytes)
            return refuse("draft blob size");
        if (pos_min > tokens.size() || cells != tokens.size() - pos_min)
            return refuse("mirror extent vs tokens");
        std::vector<uint8_t> draft(static_cast<size_t>(draft_size));
        if (!io_chunked(fp, draft.data(), draft.size(), /*write=*/false))
            return refuse("draft blob truncated");
        if (std::fread(&pending_pos, sizeof(pending_pos), 1, fp) != 1 ||
            std::fread(&n_embd, sizeof(n_embd), 1, fp) != 1)
            return refuse("carry header truncated");
        if (pending_pos != tokens.size() - 1)
            return refuse("carry pos != last token");
        if (n_embd != static_cast<uint32_t>(llama_model_n_embd_out(m_model.get())))
            return refuse("carry width");
        std::vector<float> pend(static_cast<size_t>(n_embd));
        if (!io_chunked(fp, pend.data(), pend.size() * sizeof(float), /*write=*/false))
            return refuse("carry row truncated");
        if (std::fread(&stored_hash, sizeof(stored_hash), 1, fp) != 1)
            return refuse("prefix proof truncated");
        const uint64_t hash =
            fnv1a(tokens.data() + pos_min, (tokens.size() - pos_min) * sizeof(llama_token));
        if (hash != stored_hash)
            return refuse("prefix proof mismatch");
        if (!m_mtp_drafter->set_state(draft.data(), draft.size()))
            return refuse("draft set_data rejected");
        if (!m_mtp_drafter->prime_pending(pend.data(), static_cast<llama_pos>(pending_pos)))
            return refuse("carry install failed");
        log_output("[xllama] MTP trailer restored — drafter mirror cells=" + std::to_string(cells) +
                   " pos_min=" + std::to_string(pos_min) + " carry@" + std::to_string(pending_pos) +
                   " (prefix proof OK)\n");
        return true;
    }

    bool load_state(const std::string& path, std::string* err) override {
        if (!ensure_ctx(err))
            return false;
        FILE* fp = open_state_file(path, "rb");
        if (!fp) {
            if (err)
                *err = "no state file at " + path;
            return false;
        }
        struct Closer {
            FILE* f;
            ~Closer() {
                if (f)
                    std::fclose(f);
            }
        } closer{fp};

        char hdr[1024] = {};
        if (!std::fgets(hdr, sizeof(hdr), fp)) {
            if (err)
                *err = "empty state file";
            return false;
        }
        std::string line(hdr);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
            line.pop_back();
        // magic \t n_state \t n_tokens \t fingerprint(rest of line)
        std::string field[3];
        size_t pos = 0;
        for (auto& f : field) {
            const size_t t = line.find('\t', pos);
            if (t == std::string::npos) {
                if (err)
                    *err = "malformed state header";
                return false;
            }
            f = line.substr(pos, t - pos);
            pos = t + 1;
        }
        if (field[0] != kStateMagic) {
            if (err)
                *err = "not an xllama KV state file";
            return false;
        }
        if (line.substr(pos) != state_fingerprint()) {
            // The common, expected case: model / n_ctx / KV-quant / LoRA
            // changed since the file was written. Not an error worth shouting
            // about — the caller just prefills.
            if (err)
                *err = "state file belongs to a different model or context";
            return false;
        }
        size_t n_state = 0, n_tokens = 0;
        try {
            n_state = std::stoull(field[1]);
            n_tokens = std::stoull(field[2]);
        } catch (const std::exception&) {
            if (err)
                *err = "malformed state header";
            return false;
        }
        if (n_state == 0 || n_state > kMaxStateBytes || n_tokens == 0 ||
            n_tokens > static_cast<size_t>(m_n_ctx)) {
            if (err)
                *err = "state header out of range";
            return false;
        }

        std::vector<llama_token> tokens(n_tokens);
        std::vector<uint8_t> blob(n_state);
        if (!io_chunked(fp, tokens.data(), n_tokens * sizeof(llama_token), /*write=*/false) ||
            !io_chunked(fp, blob.data(), n_state, /*write=*/false)) {
            if (err)
                *err = "state file truncated";
            return false;
        }

        const auto t_load0 = std::chrono::steady_clock::now();
        size_t used = 0;
        try {
            used = llama_state_seq_set_data(m_ctx.get(), blob.data(), blob.size(), 0);
        } catch (const std::exception& e) {
            used = 0;
            if (err)
                *err = std::string("state_seq_set_data threw: ") + e.what();
        }
        if (used == 0) {
            // The pin rolls its own read back, but the bookkeeping is ours.
            llama_memory_clear(llama_get_memory(m_ctx.get()), true);
            m_kv_tokens.clear();
            if (err && err->empty())
                *err = "llama_state_seq_set_data rejected the file";
            return false;
        }
        // Plan 009 identical-prefix fast path: when the loaded tokens are
        // byte-identical to the resident ones, no divergent turn ran between
        // save and restore (every mutation moves m_kv_tokens in lockstep),
        // so the mirror already holds this exact prefix — skip its clear and
        // keep a valid pending carry if one survived. F3.5 re-verifies via
        // can_carry before any use, so even a pathological interleaving
        // (e.g. a zero-generate divergent prefill, which moves neither token
        // vector) still clears safely there instead of here.
        const bool same_prefix = (m_kv_tokens == tokens);
        m_kv_tokens = std::move(tokens);
        // Plan 009: a slot with a drafter trailer restores the mirror for
        // real (only meaningful when the resident prefix differs; the
        // same-prefix path keeps the live mirror anyway).
        const bool mirror_restored = !same_prefix && try_restore_mtp_trailer(fp, m_kv_tokens);
        // The native blob restores the TARGET context only. A live MTP
        // drafter still mirrors the pre-load prefix plus a pending row for
        // it; letting that survive would feed the draft head a hidden state
        // from dropped positions. align_to(0) clears the mirror and the
        // pending row so the next generate rebuilds it via catch-up. Fail
        // closed: a mirror that refuses the clear must not coexist with the
        // restored target.
        if (same_prefix) {
            char lb2[160];
            snprintf(lb2, sizeof(lb2),
                     "[xllama] session: KV state loaded — mirror kept (identical %zu-token "
                     "prefix, carry preserved if valid) (#170b)\n",
                     m_kv_tokens.size());
            log_output(lb2);
        } else if (mirror_restored) {
            char lb3[192];
            snprintf(lb3, sizeof(lb3),
                     "[xllama] session: KV state loaded — MTP mirror restored from trailer "
                     "(%zu tokens, drafting continues on real state) (#170c)\n",
                     m_kv_tokens.size());
            log_output(lb3);
        } else if (m_mtp_drafter && m_mtp_drafter->ready() && !m_mtp_drafter->align_to(0)) {
            llama_memory_clear(llama_get_memory(m_ctx.get()), true);
            m_kv_tokens.clear();
            if (err)
                *err = "MTP draft mirror refused the post-restore clear";
            return false;
        } else if (m_mtp_drafter && m_mtp_drafter->ready() && !m_kv_tokens.empty()) {
            // Plan 009: with the mirror cleared, F3.5 would refuse the carry
            // on the next turn and full-prefill it, so the restored prefix
            // was dead weight for MTP users. Prime a zero pending row at the
            // last restored token so the tail-only turn keeps drafting; the
            // decoy row's only effect is one draft-mirror row's input and
            // the target still verifies every candidate (contract in
            // mtp_draft.h:prime_zero_carry). Fail-closed: a failed prime
            // leaves the cleared mirror = the pre-009 full-prefill path.
            const llama_pos prime_pos = static_cast<llama_pos>(m_kv_tokens.size()) - 1;
            if (m_mtp_drafter->prime_zero_carry(prime_pos)) {
                log_output("[xllama] mtp: post-restore carry primed at pos=" +
                           std::to_string(static_cast<int>(prime_pos)) +
                           " (zero decoy row; drafting stays on, target verifies)\n");
            } else {
                log_output("[xllama] mtp: post-restore carry prime failed — full "
                           "prefill on the next turn (fail-closed)\n");
            }
        }
        const double load_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0)
                .count();
        char lb[192];
        snprintf(lb, sizeof(lb),
                 "[xllama] session: KV state loaded — %zu tokens, %.1f MB, %.1f ms (#170b)\n",
                 m_kv_tokens.size(), (double)n_state / (1024.0 * 1024.0), load_ms);
        log_output(lb);
        return true;
    }
};

namespace detail {
std::unique_ptr<Session> create_llama(const SessionParams& sp, std::string* err) {
    // resolve_model_path yields a FILE path on Linux (a direct .gguf) but the
    // model DIRECTORY on UWP (catalogue layout LocalState\models\<name>\). llama
    // loads a file, so descend into a directory and pick the single .gguf inside.
    // Exclude LoRA basename so dirs with model.gguf + adapter.gguf load the base.
    std::string lora_base;
    if (!sp.lora_path.empty()) {
        lora_base = std::filesystem::path(sp.lora_path).filename().string();
    }
    std::string abs_path = first_gguf_in_dir(resolve_model_path(sp.model_path), lora_base);
    if (abs_path.empty()) {
        if (err)
            *err = "no .gguf file in model dir: " + resolve_model_path(sp.model_path);
        return nullptr;
    }

    llama_model_params mparams = llama_model_default_params();
    const int gpu_layers = apply_gguf_gpu_layers(sp.n_gpu_layers, mparams);
    // MTP: the draft head lives in the same GGUF, so loading it costs context
    // memory and draft compute but no second file and no second mmap. Requesting
    // it on a GGUF without the head is harmless -- the loader just finds nothing.
    mparams.load_mtp = sp.mtp;
    // Required alongside load_mtp: the fork's MTP speculative path gates on
    // llama_model_mtp_weights_get_info().managed, which mirrors this flag.
    // split_mtp_weights stays FALSE on Xbox. The fork requires the MTP head's
    // host backing to live in a CPU or CUDA buffer (llama-model.cpp: the
    // ctx_map check allows only GGML_BACKEND_DEVICE_TYPE_CPU or a backend
    // registered as CUDA). With the head offloaded to D3D12_Weights the load
    // is rejected outright:
    //   MTP host backing requires standard CPU/CUDA buffers, got D3D12_Weights
    // D3D12 is neither, so the flag is unsatisfiable on this platform.
    // Left false; the draft context below is the non-split MTP path, which
    // is the one the fork's is_ready() treats as supported when !managed.

    llama_model* raw_model = llama_model_load_from_file(abs_path.c_str(), mparams);
    if (!raw_model) {
        if (err)
            *err = "failed to load model: " + abs_path;
        return nullptr;
    }

    LlamaAdapterLoraPtr adapter;
    if (!sp.lora_path.empty()) {
        llama_adapter_lora* raw_ad = llama_adapter_lora_init(raw_model, sp.lora_path.c_str());
        if (!raw_ad) {
            llama_model_free(raw_model);
            if (err)
                *err = "failed to load LoRA adapter: " + sp.lora_path;
            return nullptr;
        }
        adapter.reset(raw_ad);
        log_output("[xllama] Session: GGUF LoRA adapter loaded (runtime)\n");
    }

    int n_threads = sp.n_threads > 0 ? sp.n_threads : detect_threads_llama();
    // Plan 006 C1: batch-graph thread count; 0/invalid keeps parity with the
    // decode count (historical behavior). The persistent pool wrapper creates a
    // second pool when the two differ.
    int n_threads_batch = sp.n_threads_batch > 0 ? sp.n_threads_batch : n_threads;
    int n_ctx = sp.n_ctx > 0 ? sp.n_ctx : kDefaultNCtx;
    log_output("[xllama] Session: GGUF model loaded via llama.cpp (persistent)\n");
    return std::make_unique<LlamaSession>(
        LlamaModelPtr(raw_model), std::move(adapter), sp.lora_scale, n_ctx, n_threads,
        n_threads_batch, sp.n_batch, sp.n_ubatch, sp.kv_q8, sp.prompt_lookup, gpu_layers, sp.mtp,
        sp.mtp_n_max, sp.mtp_p_min, sp.n_seq_max);
}
} // namespace detail

} // namespace xllama

#endif // XLLAMA_USE_LLAMA

// ---------------------------------------------------------------------------
// Public factory: dispatch to the compiled backend(s).
// ---------------------------------------------------------------------------
namespace xllama {

std::unique_ptr<Session> Session::create(const SessionParams& sp, std::string* err) {
    // Rejected configuration (set by the config parser, e.g. immutable scope2
    // startup profile + a different model): fail fast with the reason.
    if (!sp.config_reject.empty()) {
        if (err)
            *err = sp.config_reject;
        log_output("[xllama] session config REJECTED: " + sp.config_reject + "\n");
        return nullptr;
    }
    // Effective per-session configuration (no secrets): shared by GUI/API/
    // hub entry points; printed before any load so the log proves what the
    // session actually got (plan004 normal-session MTP delivery).
    {
        char cfg[320];
        snprintf(cfg, sizeof(cfg),
                 "[xllama] session config: model=%s mtp=%d depth=%d pmin=%.2f n_ctx=%d "
                 "n_batch=%d n_ubatch=%d gpu=%d kv_q8=%d\n",
                 sp.model_path.c_str(), sp.mtp ? 1 : 0, sp.mtp_n_max,
                 static_cast<double>(sp.mtp_p_min), static_cast<int>(sp.n_ctx),
                 static_cast<int>(sp.n_batch), static_cast<int>(sp.n_ubatch), sp.n_gpu_layers,
                 sp.kv_q8 ? 1 : 0);
        log_output(cfg);
    }
#if defined(XLLAMA_USE_ORT) && defined(XLLAMA_USE_LLAMA)
    Backend b = sp.backend;
    if (b == Backend::Auto) {
        // Layout-aware: uses suffix fast-path + resolve + directory scan so that
        // bare catalogue names (e.g. "qwen35-0.8b" pointing at a dir with .gguf)
        // are classified correctly. Explicit Backend from manifest (MainPage) or
        // tests still takes precedence.
        b = model_uses_llama_backend(sp.model_path) ? Backend::LlamaCpp : Backend::OrtGenAI;
    }
    return b == Backend::LlamaCpp ? detail::create_llama(sp, err) : detail::create_ort(sp, err);
#elif defined(XLLAMA_USE_ORT)
    return detail::create_ort(sp, err);
#else // XLLAMA_USE_LLAMA
    return detail::create_llama(sp, err);
#endif
}

SessionHub& session_hub() {
    static SessionHub hub;
    return hub;
}

} // namespace xllama
