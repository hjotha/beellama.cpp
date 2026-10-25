// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// xllama::Session — persistent model session for multi-turn applications.
// Keeps the model and tokenizer loaded between generate() calls,
// eliminating the ~1-2s per-call reload overhead of run_inference().
#pragma once

#include "xllama/embedding.h"
#include "xllama/inference_params.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace xllama {

// Which inference backend a session uses. Auto inspects the model identifier
// (suffix or resolved on-disk layout: *.gguf -> LlamaCpp, else OrtGenAI).
// Explicit values are honored when a build links both backends; single-backend
// builds ignore this field.
enum class Backend { Auto, OrtGenAI, LlamaCpp };

struct SessionParams {
    std::string model_path;   // same semantics as InferenceParams::model_path
    int n_ctx = kDefaultNCtx; // one home, inference_params.h (#171)
    int n_threads = 0;        // 0 = auto
    int n_threads_batch = 0;  // llama.cpp only; 0 = same as n_threads (plan 006 C1)
    int n_batch = 0;          // llama.cpp only; 0 = default (2048). Logical prefill batch.
    int n_ubatch = 0;         // llama.cpp only; 0 = default (512). Physical prefill chunk.
    // llama.cpp only: how many sequences one context may hold at once
    // (llama_context_params::n_seq_max). From llama.ini [n_seq_max], with
    // [n_parallel] accepted as the historical llama.cpp/llama-server spelling.
    // 1 = the single-sequence context to date; the sequences share the same
    // n_ctx cell pool. This is context capacity, not a request queue: the
    // session hub still serves one request at a time.
    int n_seq_max = 1;
    Backend backend = Backend::Auto;
    int n_gpu_layers = 0; // llama.cpp only; 0 = CPU; > 0 = d3d12 backend (docs/gguf-gpu-decode.md)

    // Optional GGUF LoRA adapter (llama.cpp only). Empty = base model only.
    // Loaded via llama_adapter_lora_init + llama_set_adapters_lora (inference-time;
    // not training). Scale defaults to 1.0. ORT GenAI sessions ignore this field.
    std::string lora_path;
    float lora_scale = 1.0f;

    // #171: quantize the llama.cpp KV cache to q8_0 (halves its footprint) and
    // force flash attention, which quantized V requires — the pin throws at
    // context creation otherwise. On any context-creation failure the session
    // retries with default cache types, so an architecture without flash
    // attention support degrades to F16 KV instead of failing to load.
    // Default off until the on-console measurement decides (#171).
    bool kv_q8 = false;

    // #130: run a throwaway generate at load for DirectML models, paying the
    // one-time per-process cost (§5e: cold→warm prefill 1.64-1.72×) inside the
    // "loading model" phase instead of on the user's first turn. Ignored for
    // CPU models (§5e control: 1.00×) and by the llama.cpp backend.
    bool dml_warmup = true;

    // Phase 15 W2 (#210): draft-free prompt-lookup speculative decoding on the
    // GGUF path. Default OFF — product default is decided after console M3.
    // ORT sessions ignore this field. Hybrid caches that refuse tail rewind
    // disable speculation at runtime for that generation (no KV corruption).
    bool prompt_lookup = false;

    // MTP (multi-token prediction) drafting against the beellama fork. Requires a
    // model whose GGUF carries the MTP head (llama_model_params::load_mtp), which
    // the session sets when this is on. Default OFF: the console measurement decides,
    // and a model without an MTP head cannot use it at all.
    bool mtp = false;
    int mtp_n_max = 4;       // tokens proposed per step
    float mtp_p_min = 0.75f; // stop drafting below this candidate probability

    // Set by the config parser on a rejected combination (e.g. another model
    // while the immutable scope2 startup profile is active). Session::create
    // fails with this message; empty = admissible. Populate-only — the parser
    // never mutates global state.
    std::string config_reject;
};

struct GenerateParams {
    std::string prompt;
    int n_predict = 96;
    // Phase instrumentation on/off; see InferenceParams::profile_phases. Counters
    // stay on either way — this only drops the chrono snapshots, so the ON/OFF
    // comparison measures the instrumentation cost.
    bool profile_phases = true;
    // Shared with InferenceParams via xllama/sampling.h — see #125. The two
    // surfaces ran different samplers until these were made one source.
    float temperature = sampling_defaults::kTemperature;
    float top_p = sampling_defaults::kTopP;
    int top_k = sampling_defaults::kTopK;
    float repetition_penalty = sampling_defaults::kRepetitionPenalty;
    uint32_t seed = sampling_defaults::kSeed;

    SamplingConfig sampling() const {
        return SamplingConfig{temperature, top_p, top_k, repetition_penalty, seed, false};
    }

    // Stop sequences: checked as substrings of accumulated output.
    // Generation stops and the matching sequence is stripped.
    std::vector<std::string> stop_sequences;

    // Continuous decoding / KV-cache reuse. Honored by BOTH backends now: the
    // ORT path (persistent generator) and the llama.cpp path (persistent
    // llama_context, KV cache retained across turns — turn-2 prefill 4.07×).
    //   reuse_kv = false           → legacy stateless turn: a fresh generator is
    //                                created and destroyed; `prompt` is the full
    //                                context. Proven default.
    //   reuse_kv = true,  reset_kv = false → continuation turn: `prompt` is only
    //                                the NEW turn's tokens; they are appended to
    //                                the generator kept alive from the previous
    //                                turn, so the per-turn prefill covers just the
    //                                delta (the multi-turn TTFT win).
    //   reuse_kv = true,  reset_kv = true  → start/refresh the persistent
    //                                generator (new conversation, context
    //                                eviction, or sampling change); `prompt` is
    //                                the full context. Subsequent turns pass
    //                                reset_kv = false to reuse it.
    // On any failure of a continuation turn the session drops its chat state and
    // reports !success so the caller can retry with reset_kv + the full prompt.
    // Exception (#169): when the session supports context shift
    // (can_context_shift()), a continuation turn that would overflow n_ctx
    // evicts the oldest tokens past n_keep from the resident KV instead of
    // failing, so long chats stay in the reuse regime past the token budget.
    bool reuse_kv = false;
    bool reset_kv = false;

    // #169: number of leading prompt tokens pinned across context shifts —
    // the system prompt (plus BOS), which must survive any eviction. 0 = no
    // pinned head. Callers compute it once per conversation via
    // count_tokens() on the system-only render. Ignored by sessions that
    // cannot shift (ORT, and any arch where llama_memory_can_shift is
    // false — those keep the fail-fast path). Hybrid caches accept the
    // front-drop erase (unlike #170a's tail rewind): the recurrent state
    // holds no cells in the evicted range, only its absorbed history —
    // an approximation the quality gates measure.
    int n_keep = 0;

    // Parity/diagnostic: write this turn's accepted token ids, one per line, to
    // this path. Empty = no dump. Exists because the Session decode path does
    // not go through run_inference(), so InferenceParams::dump_tokens_path does
    // not reach it; plan 003 stage 1 compares integral ids per turn.
    std::string dump_tokens_path;

    // on_token receives a view into a per-iteration buffer: copy it before
    // the callback returns.
    std::function<void(std::string_view)> on_token;
    std::function<void(const std::string&)> on_status;
    std::atomic<bool>* abort_flag = nullptr;
};

// Session owns the loaded model and tokenizer.
// Thread safety: generate() must not be called concurrently.
struct Session {
    // Create a session by loading the model once.
    // On failure returns nullptr and sets *err.
    static std::unique_ptr<Session> create(const SessionParams& params, std::string* err = nullptr);

    virtual InferenceResult generate(const GenerateParams& params) = 0;

    // Token count for routing/heuristics (encode-only; no generation).
    virtual int count_tokens(const std::string& prompt) = 0;

    // Encode one input with an embedding-capable backend. Text-generation-only
    // backends report an explicit unsupported error by default.
    virtual EmbeddingResult embed(const EmbeddingParams& params) {
        (void)params;
        EmbeddingResult result;
        result.error_msg = "embeddings are not supported by this backend";
        return result;
    }

    virtual int context_length() const {
        return 0;
    }

    // #169: whether a continuation turn that would overflow n_ctx evicts the
    // oldest tokens (RoPE shift) instead of failing. False for ORT and for
    // llama archs whose cache cannot shift (llama_memory_can_shift). Callers
    // use it to keep the KV-reuse path for long chats instead of falling back
    // to trimmed full prefills.
    virtual bool can_context_shift() const {
        return false;
    }

    // #170b: persist / restore the resident KV (sequence 0) so switching away
    // from a conversation and back — or restarting the app — does not re-read
    // the whole history. The file carries a fingerprint of the model and the
    // context configuration: load_state refuses one that does not match,
    // because the llama.cpp per-sequence state path validates cache SHAPE only
    // (layer count, cell count, K/V types) — two different models of the same
    // shape would load each other's cache and generate silent garbage.
    // Both return false with *err on any failure, "unsupported backend"
    // included; a caller must treat that as "no cache, prefill normally",
    // never as a hard error.
    // request_nonce (T18 fault probe, plan 009): test-only identity of the
    // caller's request, threaded from the LAN handler's X-Test-Nonce header
    // through to save_state so an armed mid-write fault can bind to the exact
    // disposable test request. Empty (the default) never matches any arm, so
    // GUI snapshots and ordinary saves are unaffected.
    virtual bool save_state(const std::string& path, std::string* err = nullptr,
                            const std::string& request_nonce = "") {
        (void)path;
        (void)request_nonce;
        if (err)
            *err = "state persistence not supported by this backend";
        return false;
    }
    virtual bool load_state(const std::string& path, std::string* err = nullptr) {
        (void)path;
        if (err)
            *err = "state persistence not supported by this backend";
        return false;
    }

    virtual ~Session() = default;
};

} // namespace xllama
