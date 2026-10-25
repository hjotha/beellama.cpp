// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#include "xllama/device_train.h"
#include "xllama/inference.h"
#include "xllama/inference_params.h"
#include "xllama/llama_ini.h"
#include "xllama/session.h"
#include "xllama/training_params.h"

namespace xllama::bridge {

// Thin wrappers kept for backward compatibility with existing UWP code.
using InferenceParams = xllama::InferenceParams;
using InferenceResult = xllama::InferenceResult;

inline InferenceResult run_inference(const InferenceParams& params) {
    return xllama::run_inference(params);
}

// Called from UWP App on a background thread (bench mode).
void main_loop();

// CPU memory-bandwidth micro-bench. Triggered by LocalFolder\membw.flag; writes
// membw-result.csv (+ .done marker) to LocalState. Pins the DRAM-bandwidth
// ceiling behind the bandwidth-bound decode number (see docs/benchmarks.md).
void run_membw();

// Elementwise CPU kernel microbenchmark (owner 599): single-op ggml graphs
// vs pure reference loops at the workload widths, on the console CPU.
// Triggered by LocalFolder\elembench.flag; writes elembench-result.csv.
void run_elembench();

// Disk (NVMe) read-bandwidth micro-bench (SSD-inference assessment). Triggered
// by LocalFolder\diskbw.flag; writes diskbw-result.csv (+ .done) to LocalState.
// Pins the sandboxed file-read rate any weight-streaming scheme would divide by.
void run_diskbw();

// Phase 15 W3 (#211): own D3D12 compute STREAM read (~1 GB VRAM) + checksum.
// Triggered by LocalFolder\gpubw.flag; writes gpubw-result.csv (+ .done).
// Kill criterion 100 GB/s is documented in docs/phase15-re-opt.md — not invented here.
void run_gpubw();

// Phase 15 H6.2 (#228): Q4_K GEMV density probe (wave32 + naive A/B).
// Triggered by LocalFolder\gpugemv.flag; writes gpugemv-result.csv (+ .done even
// on partial PSO failure). Soft G2 40 GB/s packed, K1=8; not a product backend.
void run_gpugemv();

// GGUF GPU decode probe D1 (docs/gguf-gpu-decode.md). inproc=false: headless
// gpustep.flag → gpustep-result.csv. inproc=true: gpustep-inproc.flag, run after
// the XAML window is up → gpustep-inproc-result.csv. Both write a .done marker.
void run_gpustep(bool inproc);

// d3d12 ggml backend selftest (docs/gguf-gpu-decode.md, D2a). Triggered by
// LocalFolder\d3d12be.flag; writes d3d12be-result.csv (+ .done). Every Q4_0 /
// Q4_K / Q6_K shape against ggml's dequantizers. llama builds only.
void run_d3d12_selftest();
void run_tttarget();

// GGUF GPU decode D2b: n_gpu_layers for interactive sessions (GUI and LAN API)
// from LocalState\gguf_gpu_layers.txt, else LocalState\llama.ini [n_gpu_layers],
// else 0. One home for both front ends; experimental, no UI
// (docs/gguf-gpu-decode.md).
int gguf_gpu_layers_knob();

// #171: q8_0 KV cache (+ forced flash attention where supported) for
// interactive GGUF sessions (GUI and LAN API) from LocalState\kv_q8.txt,
// else LocalState\llama.ini [kv_q8], else 0. Same one-home-per-knob pattern.
int gguf_kv_q8_knob();

// LocalState\llama.ini (xllama/llama_ini.h) as parsed session defaults.
// Empty map when the file is absent.
LlamaIni read_llama_ini();

// Overwrite the SessionParams fields the llama.ini carries (n_ctx, n_threads,
// n_batch, n_ubatch). n_gpu_layers and kv_q8 stay with their knobs above, so
// the explicit single-purpose files keep overriding the .ini. No-op when the
// file is absent.
void apply_llama_ini_session(SessionParams& sp);
// Apply the immutable twocol/repack startup profile once (App launch).
void apply_startup_profile();

// Heap-ceiling probe. Triggered by LocalFolder\ramceil.flag; writes
// ramceil-result.csv (+ .done marker holding the stop reason) to LocalState.
// Measures how much heap the process can actually commit under GameOS — the
// number that decides which model quants are admissible, since GGUF weights
// are read into the heap and not mapped (see include/xllama/ramceil.h).
void run_ramceil();

// Phase 16 WS-F (card H16.6): can an AppContainer app on GameOS capture audio?
// Triggered by LocalFolder\mic.flag; writes mic-result.json (+ .done) holding
// the WinRT status enums BY NAME, not a boolean — AccessDenied (sandbox says
// no) and DeviceNotAvailable (no headset plugged in) are different answers, and
// only the first is a verdict on WS-F. See docs/uwp-constraints.md.
void run_mic_probe();

// Diffusion pipeline (SD-Turbo on plain ORT DirectML). Triggered by
// LocalFolder\diffuse.flag (headless) or diffuse-inproc.flag (in-process
// experiment) — see diffuse.cpp for the model contract.
void run_diffuse();

// Logit-parity dump: greedy 1-token forward pass on the ORT backend, writing the
// last prefill-token logits to LocalState\logits.bin (+ .json sidecar, + a
// logits.done marker). Triggered by LocalFolder\logits.flag. Reads the raw prompt
// from prompt.txt and the model from model.txt (same config files as bench).
// scripts/validate-logit-parity.sh pulls the dump and diffs it against the
// llama.cpp golden via scripts/compare-logits.py.
void run_logits();

// Controlled single-vs-batch replay (plan 004 boundary diagnosis): knob files
// replay_prompt.txt (prompt text), replay_ctx.txt (committed output ids),
// replay_feed.txt (observed verify feed), replay_known.txt (sequential argmax
// chain), replay_rem.txt (captured remainder ids through the output33
// decision), replay_rem_known.txt (sequential argmax after each remainder
// id). Writes replay-result.csv (+ .done): 46 rows per rep (seq, acc, tail,
// seqc/accc/tailc x6, d0..d3, drem x6, e0..e2, etail, ecorr, erem x6,
// accb x3, btail). UWP llama builds only; diagnostic, never a gate.
void run_replay();

// First-numeric-divergence capture (plan 004): same replay_* inputs as
// run_replay (rem files not required). Triggered by diverge.flag; writes
// diverge-result.csv (+ .done) with per-tensor B3-vs-B4 first-3-row
// comparisons in graph order. Diagnostic, never a gate.
void run_diverge();

// Narrow z-0 capture (NARROW-TO-Z0): same replay_* inputs (rem files not
// required). Triggered by znarrow.flag; writes znarrow-result.csv (+ .done)
// with per-tensor B3-vs-B4 first-3-row comparisons for the z-0 neighborhood
// only (z-0, norm-0, attn_norm-0), validated by the same shared exporter.
// Diagnostic, never a gate.
void run_znarrow();

// Termination-state gate (plan 003): deterministic headless scenarios for
// cancel through the real abort_flag, a boundary-spanning stop sequence and
// natural EOG, each with full-id state checks. Same replay/bench knob files
// select the arm (bench_mtp.txt / d3d12twocol.txt / cpurepackforcegemv.txt)
// and termgate.txt selects scenarios ("cancel,stop,eog"; default all).
// Writes termgate-result.csv (+ .done). Diagnostic, never a gate default.
void run_termgate();

// Single-op CPU-vs-DML repro (#111): loads LocalState\repro.onnx with a plain
// ORT CPU session and a DML session, feeds repro-input.bin and writes
// repro-out-cpu.bin / repro-out-dml.bin (+ repro.done marker). Triggered by
// LocalFolder\oprepro.flag; driven by scripts/validate-op-repro.sh. Used to
// isolate broken DML kernels (first target: (Skip)SimplifiedLayerNormalization
// -> MVN2 UseMean=false, the #91 root cause).
void run_oprepro();

// Lane B on-device training (ggml-opt partial FT). Triggered by
// LocalFolder\train.flag; reads the job from LocalState\training\job.json
// (paths inside the job are resolved relative to LocalState), runs
// prepare → train → export → evaluate in-process on the CPU, and writes
// <out_dir>\result.json plus a training\result.done marker. See
// docs/training-architecture.md Lane B and scripts/validate-console-training.sh
// device-train.
void run_train();

// Shared runner used by headless train.flag and the in-app personalize path
// (#116). Localizes relative job paths to LocalState, writes
// training/progress.json on each progress tick, and returns the engine result.
// Does NOT exit the process — safe to call while XAML is up.
// XLLAMA_DEVICE_TRAIN builds only; otherwise returns success=false.
xllama::TrainingResult run_train_job_localized(const xllama::TrainingJob& job,
                                               const xllama::DeviceTrainCallbacks& cb = {});

} // namespace xllama::bridge
