// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#include "xllama/sampling.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace xllama {

/// The shipping context size.
///
/// Every surface that opens a context (chat UI, LAN API, Session default, CLI
/// default) reads this constant. The trimmer budget `kMaxPromptTokens`
/// (routing_policy.h) is sized against it.
inline constexpr int kDefaultNCtx = 2048;

// ---------------------------------------------------------------------------
// Inference configuration
// ---------------------------------------------------------------------------

/// Configuration for a single inference call.
///
/// Used by the CLI, benchmarks, and the `run_inference()` function. Most fields
/// have defaults so callers only set what they want to change.
struct InferenceParams {
    /// Path to model. Linux: absolute path; UWP: filename in LocalFolder.
    std::string model_path;

    /// Input prompt text.
    std::string prompt;

    /// Max tokens to generate (default: 128).
    int n_predict = 128;

    /// Context size (default: kDefaultNCtx = 2048).
    int n_ctx = kDefaultNCtx;

    /// Override for the engine's max_length.
    ///   0  = derive as min(n_ctx, n_prompt + n_predict) (bench default)
    ///  -1  = saturate to n_ctx (what Session::generate ships)
    ///  >0  = explicit, clamped to (n_prompt+1, n_ctx]
    int max_length_override = 0;

    /// Thread count (0 = auto-detect).
    int n_threads = 0;

    /// Batch thread count for graphs with more than one token (prefill chunks,
    /// verify batches, catch-up). 0 = same as n_threads (unchanged default);
    /// single-token decode always uses n_threads (plan 006 C1).
    int n_threads_batch = 0;

    /// Which repetition of a repeated bench this run is (0 = not a bench).
    /// Written verbatim into the bench CSV's run_index column.
    int run_index = 0;

    /// llama.cpp prefill batching (GGUF path only; 0 = default).
    int n_batch = 0;
    int n_ubatch = 0;

    /// Quantize KV cache to q8_0 (GGUF path only).
    bool kv_q8 = false;

    /// GGUF layers whose weight matmuls run on the d3d12 backend (0 = CPU,
    /// docs/gguf-gpu-decode.md). Ignored where D3D12 is unavailable.
    int n_gpu_layers = 0;

    /// Bench only: decode exactly n_predict tokens, through end-of-generation
    /// (stop sequences should be cleared too). Default off.
    bool ignore_eog = false;

    /// Phase instrumentation on/off (plan 003 stage 2). ON by default. OFF removes
    /// the per-phase chrono snapshots while keeping every counter, so the ON/OFF
    /// delta measures the cost of the instrumentation itself rather than guessing
    /// it from the residual. Does NOT change decoding or sampling.
    bool profile_phases = true;

    /// Sampling defaults from `xllama/sampling.h`.
    float temperature = sampling_defaults::kTemperature;
    float top_p = sampling_defaults::kTopP;
    int top_k = sampling_defaults::kTopK;
    float repetition_penalty = sampling_defaults::kRepetitionPenalty;
    uint32_t seed = sampling_defaults::kSeed;

    /// Deterministic argmax decode (prerequisite for cross-backend logit parity).
    bool greedy = false;

    /// Build a `SamplingConfig` from the sampling fields.
    SamplingConfig sampling() const {
        return SamplingConfig{temperature, top_p, top_k, repetition_penalty, seed, greedy};
    }

    /// System message for chat template (empty = built-in default).
    std::string system_prompt;

    /// Dump last prefill-token logit vector to this path (+ `.json` sidecar).
    std::string dump_logits_path;

    /// Dump the accepted token ids of this run to this path, one per line, with
    /// a trailing fnv1a of the generated text (plan 003, stage 1: per-run token
    /// parity sidecar). Empty = off.
    std::string dump_tokens_path;

    /// Stop strings; generation ends when output ends with any of these.
    std::vector<std::string> stop_sequences;

    /// Wrap `prompt` with the model's chat template before inference.
    bool chat_template = false;

    /// GGUF LoRA adapter path (llama.cpp path only; empty = off).
    std::string lora_path;
    float lora_scale = 1.0f;

    /// Draft-free prompt-lookup speculative decoding (GGUF path only; default OFF).
    bool prompt_lookup = false;

    /// MTP drafting against the beellama MTP head (GGUF path only; default OFF).
    /// Needs a model whose GGUF carries the head; a no-op on any other model.
    bool mtp = false;
    int mtp_n_max = 4;
    float mtp_p_min = 0.75f;

    /// Probe CPU memory bandwidth and exit (no model load).
    bool run_membw = false;

    /// Probe disk (NVMe) read bandwidth and exit (no model load).
    bool run_diskbw = false;

    /// Probe GPU STREAM bandwidth and exit (D3D12; non-Windows reports unavailable).
    bool run_gpubw = false;

    /// Probe Q4_K GEMV density and exit (D3D12 CS; not a backend).
    bool run_gpugemv = false;

    /// GGUF GPU decode probe D1: print the cost-model projection and exit.
    bool run_gpustep = false;

    /// Evaluate this gpustep CSV against the D1 gates and exit (empty = off).
    std::string gpustep_verdict_csv;

    /// Probe heap ceiling and exit (no model load).
    bool run_ramceil = false;

    /// Validate a TrainingJob JSON and exit (no model).
    bool run_validate_train_job = false;

    /// Shell out to the host training runner (from a TrainingJob JSON).
    bool run_train_job = false;

    /// Print training capability matrix and exit.
    bool run_training_capabilities = false;

    /// Run embedding smoke test and exit (load model, embed inputs, report metrics).
    bool run_embed = false;

    /// Embedding input strings for --embed mode.
    std::vector<std::string> embed_inputs;

    /// Requested embedding dimensions for --embed mode (0 = native).
    int embed_dimensions = 0;

    /// Path to TrainingJob JSON (for --train-job / --validate-train-job).
    std::string train_job_path;

    /// UI callback: status changes (e.g. "loading model"). Called from inference thread.
    std::function<void(const std::string&)> on_status;

    /// UI callback: per-token text piece. Called from inference thread; copy before return.
    std::function<void(std::string_view)> on_token;

    /// Stream generated pieces to stdout (interactive CLI; off by default).
    bool echo_stdout = false;

    /// Atomic flag set from UI thread to request early termination.
    std::atomic<bool>* abort_flag = nullptr;
};

/// Resolve the max_length for inference.
///
/// @param n_ctx        Session context size.
/// @param n_prompt     Number of prompt tokens.
/// @param n_predict    Max tokens to generate.
/// @param override_v   Override value (<0=saturate to n_ctx, 0=derive, >0=explicit).
/// @return             The effective max_length.
inline int resolve_max_length(int n_ctx, int n_prompt, int n_predict, int override_v) {
    if (override_v < 0)
        return n_ctx;
    if (override_v > 0)
        return std::clamp(override_v, n_prompt + 1, n_ctx);
    return std::min(n_ctx, n_prompt + n_predict);
}

/// Result of an inference call.
struct InferenceResult {
    /// Whether generation succeeded.
    bool success = false;

    /// Model load time (ms).
    double t_load_ms = 0.0;

    /// Prefill time (ms).
    double t_p_eval_ms = 0.0;

    /// Decode time (ms).
    double t_eval_ms = 0.0;

    /// Time from the start of prefill until the first generated token is ready.
    /// This is the interactive time-to-first-token (TTFT), in milliseconds.
    double t_first_token_ms = 0.0;

    /// Number of prefill tokens evaluated.
    int n_p_eval = 0;

    /// Number of decode tokens generated.
    int n_eval = 0;

    /// True if stopped on a stop sequence; false if capped by n_predict/EOS.
    bool ended_with_stop = false;

    /// Termination-state gate (plan 003): natural-EOG evidence straight from
    /// the result — the sampled token id and the branch that sampled it — so
    /// a gate never infers EOG from a short output or from hitting the cap.
    /// Stop-sequence evidence: the branch and loop round the stop fired in.
    /// Defaults: -1 / empty = the run ended without that event (cap, abort).
    int eog_token = -1;
    std::string eog_branch;
    std::string stop_branch;
    int stop_round = -1;

    /// Max length actually requested of the engine.
    /// On DirectML this controls prefill throughput. 0 = N/A (GGUF).
    int max_length = 0;

    /// Peak working set size (MB).
    size_t peak_ws_mb = 0;

    /// Per-process GPU memory usage after model load (0 = N/A).
    size_t gpu_mem_mb = 0;

    /// OS-granted GPU budget for this process (0 = N/A).
    size_t gpu_budget_mb = 0;

    /// GGUF layers actually placed on the d3d12 backend (0 = CPU run).
    int gpu_layers = 0;

    /// Speculative drafts generated, MTP + n-gram lookup combined (0 when both are
    /// off). Read n_mtp_drafted when the question is "did MTP propose anything":
    /// this aggregate cannot distinguish the two sources.
    int n_drafted = 0;

    /// Speculative drafts accepted, both sources combined.
    int n_spec_accepted = 0;

    /// Proposals that came from the MTP head, and the ones accepted from them.
    /// Zero whenever MTP is off, even if the n-gram lookup drafted.
    int n_mtp_drafted = 0;
    int n_mtp_accepted = 0;

    /// Proposals that came from the n-gram lookup (only when it is enabled).
    int n_lookup_drafted = 0;
    int n_lookup_accepted = 0;

    /// MTP rounds that carried a non-empty draft batch. With this at 0 the MTP
    /// head never reached a verify batch, whatever n_drafted says.
    int n_mtp_rounds = 0;

    /// Wall-clock accounting over the SAME interval, so the phases can be
    /// reconciled against a total instead of being compared to nothing
    /// (plan 003 stage 2). t_total_ms covers load + prefill + decode;
    /// t_decode_ms is the decode loop alone and t_decode_accounted_ms is the sum
    /// of the per-phase timers inside it, so their difference is the residual.
    double t_total_ms = 0.0;
    double t_decode_ms = 0.0;
    double t_decode_accounted_ms = 0.0;
    double t_lookup_ms = 0.0;
    double t_classic_ms = 0.0;

    /// True when MTP was requested and the draft context actually came up.
    /// A bench that asked for MTP must fail when this is false (plan 003,
    /// stage 1: "a drafter that did not activate is a validation failure").
    bool mtp_active = false;

    /// Generated text output.
    std::string output_text;

    /// Error description on failure.
    std::string error_msg;
};

} // namespace xllama
