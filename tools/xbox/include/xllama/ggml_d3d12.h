// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// ggml backend "d3d12" — GGUF GPU decode D2 (docs/gguf-gpu-decode.md).
//
// A GPU-type ggml device running MUL_MAT with Q4_0/Q4_K/Q5_K/Q6_K/Q8_0 weights
// and f32 activations, plus the opt-in narrow GATED_DELTA_NET contract below.
// Two buffer types:
//   D3D12_Weights  DEFAULT heap, holds matmul weights (exposed as an extra buft)
//   D3D12_Host     CUSTOM WRITE_BACK heap, is_host — the device default buft, so
//                  the scheduler's activations are CPU-visible: CPU<->GPU copies
//                  are memcpy and the CPU reads results in place.
// supports_op accepts a MUL_MAT only when its weight already lives in
// D3D12_Weights, which steers llama.cpp's per-weight buft probe past the host
// buft. GDN uses the host buft; unsupported ops and weights fall back to CPU.
//
// This header is WinRT- and D3D12-free: the pure rules below are host-tested on
// Linux, where ggml_d3d12_register() returns false.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"

namespace xllama {

// Rows per thread group and threads per group of every mmv kernel
// (shaders/ggml_d3d12_mmv_*.hlsl).
inline constexpr int kD3d12MmvRows = 4;
// Threads per group: 64 or 128, chosen per matmul by d3d12_mm_threads(). 16
// threads share one 256-element chunk, so threads/16 chunks are in flight.
inline constexpr int kD3d12MmvThreadsShort = 64;
inline constexpr int kD3d12MmvThreadsLong = 128;
inline constexpr int kD3d12LongKChunks = 16; // K >= 4096 → 128 threads
// Every kernel walks K in 256-element chunks (one Q4_K/Q5_K/Q6_K super-block, eight
// Q4_0/Q8_0 blocks).
inline constexpr int kD3d12Chunk = 256;
inline constexpr std::uint32_t kD3d12MaxGroups = 65535;

// Weight types the backend runs.
bool d3d12_weight_type_supported(ggml_type t);

// Same-package Q8_0 CPU/GPU comparison. Default ON; set before loading the
// model so the loader's buffer probe and execution use the same placement.
void d3d12_set_q8_enabled(bool enabled);
bool d3d12_q8_enabled();

// Narrow Qwen3.5 recurrent-op experiment, default OFF. Set before context
// creation. F32, S=128, Q/K heads=16, V heads=32, one sequence, T<=64,
// gathered state, scalar activated gates, 1..17 rollback snapshots only.
void d3d12_set_gdn_enabled(bool enabled, int min_tokens = 1);
bool d3d12_gdn_enabled();

// Plan 006 C4 selftest-only gate: lets the d3d12be selftest run GGML_OP_SILU
// through the D3D12 backend for the CPU-vs-GPU corpus. Default OFF; no product
// path ever enables it.
void d3d12_set_silu_test_enabled(bool enabled);
bool d3d12_silu_test_enabled();

// Plan 006 C4 product trial (default OFF): FFN split-SWIGLU on D3D12, narrow
// name/shape lineage whitelist. Mode 0 = off, 1 = all FFN, 2 = target-context
// FFN only (layer index < target layer count, i.e. nextn/draft excluded).
// d3d12_set_swiglu_target_layers is setup-time only (model load), never a
// mid-graph toggle.
void d3d12_set_swiglu_product_mode(int mode);
int d3d12_swiglu_product_mode();
// Applies the requested mode filtered by the process profile latch; called at
// every context creation. The first MTP-capable context makes the process
// MTP-profiled and FFN stays OFF from then on (fail-closed; no multi-context
// concurrency guarantee is claimed).
void d3d12_finalize_swiglu_mode(bool mtp_active);
// Gate every context creation: false means this context's MTP mode conflicts
// with the bound FFN profile; fail the request and require a restart instead
// of changing capability under a live context.
bool d3d12_swiglu_profile_accepts(bool mtp_active);
void d3d12_set_swiglu_target_layers(int n);
int d3d12_swiglu_target_layers();

// Plan 006 C4: FFN SWIGLU is a sequential-only optimization. An active MTP
// drafter always forces it off, so a seq-configured process cannot leak the
// capability into an MTP session. Pure and host-testable.
inline int effective_swiglu_mode(int requested_mode, bool mtp_active) {
    // Owner decision 2026-10-09: FFN SWIGLU + MTP enabled together. The exact
    // kernel (bit-exact vs the CPU on the real activation range, strict
    // per-round trace gate passing on chat256/code256 MTP) removed the reason
    // the MTP profile used to force FFN OFF; MTP no longer constrains the mode.
    (void)mtp_active;
    return requested_mode;
}

// Process profile latch: the capability is bound by the FIRST context that
// uses it and never changes while the process lives. A later context with the
// other MTP mode is rejected BEFORE creation (restart required) by accepts(),
// so no already-scheduled graph ever loses its backend. Pure and
// host-testable; the backend holds one instance.
struct SwigluModePolicy {
    int mode = 0;             // effective mode seen by the backend predicate
    bool bound = false;       // first context fixed the profile
    bool mtp_capable = false; // profile of that first context
    // Bind once; already-bound calls leave the mode untouched (conflicting
    // requests are refused earlier by accepts()).
    void finalize(int requested_mode, bool mtp_active) {
        if (bound)
            return;
        bound = true;
        mtp_capable = mtp_active;
        mode = effective_swiglu_mode(requested_mode, mtp_active);
    }
    // May a context with |mtp_active| be created at all? With the knob OFF
    // capability is disabled and any context is fine; with it ON only contexts
    // matching the bound profile are accepted, and only before binding.
    // Once bound, the decision uses the BOUND effective mode/profile only; a
    // later knob change (e.g. requested -> 0) cannot open the gate while the
    // bound capability is still ON.
    bool accepts(int requested_mode, bool mtp_active) const {
        (void)requested_mode;
        (void)mtp_active;
        // The effective mode is the same for every context profile now, so no
        // context can conflict with a bound capability; the mode itself is
        // still bound once by the first finalize().
        return true;
    }
};
bool d3d12_gdn_supported(const ggml_tensor* op);
void d3d12_gdn_emulate(const ggml_tensor* op, float* out);

// LM-head Q6_K experiment: 1 (default OLD), 2/4 columns, or 0 (auto:
// B2 uses two columns, B3/B5 use four; both kernels measured on the Xbox).
// Production selection is restricted to N=248320/K=2560/B=2,3,5. B1 stays OLD.
void d3d12_set_q6_columns(int columns);
int d3d12_q6_columns();
bool d3d12_q6_allowlisted(int n, int k, int ncols);
int d3d12_q6_columns_for(int n, int k, int ncols);
bool d3d12_q6_tiled_available(int columns);
std::uint64_t d3d12_q6_tiled_matmuls();

// What supports_op sees, as plain data (no ggml_tensor needed in tests).
struct D3d12MatmulDesc {
    ggml_type src0_type = GGML_TYPE_F32;
    std::int64_t ne00 = 0, ne01 = 0, ne02 = 1, ne03 = 1; // weight [K, N]
    std::int64_t ne10 = 0, ne11 = 0, ne12 = 1, ne13 = 1; // activations [K, M]
    ggml_type src1_type = GGML_TYPE_F32;
    ggml_type dst_type = GGML_TYPE_F32;
    bool src0_contiguous = true;
    bool src1_contiguous = true;
    bool src0_in_weight_buffer = false; // weight lives in a D3D12_Weights buffer
};

// The MUL_MAT rules of supports_op. An owned supported weight with zero
// activation columns stays on GPU; graph_compute skips the empty node.
bool d3d12_mm_supported(const D3d12MatmulDesc& d);

struct D3d12Dispatch {
    std::uint32_t groups_x = 0; // ceil(N / kD3d12MmvRows)
    std::uint32_t groups_y = 0; // one per activation column
    bool ok = false;            // within the 65535 group limit
};

D3d12Dispatch d3d12_mm_dispatch(std::int64_t n, std::int64_t ncols);

// Narrow z-0 weight identity (first-divergence diagnosis): the exact gate
// weight feeding z-0, by stable loader name + shape + quant. Used for both
// the placement pin and the dispatch record so a renamed graph node cannot
// silently dodge either, without broadening to unrelated matmuls.
// Host-tested like the other pure rules.
inline bool d3d12_is_z0_weight(const char* name, std::int64_t n0, std::int64_t n1, bool is_q4k) {
    return name != nullptr && std::strcmp(name, "blk.0.attn_gate.weight") == 0 && n0 == 2560 &&
           n1 == 4096 && is_q4k;
}

// Two-column tile planner (plan 004, q4_k experiment): same row groups as
// above, but one group covers two activation columns, so groups_y is
// ceil(ncols/2). Same group-limit rule. Host-tested like the base planner;
// the backend uses it only for the NEW variant (q4_k, ncols >= 2).
D3d12Dispatch d3d12_mm_dispatch_2col(std::int64_t n, std::int64_t ncols);
// Four-column sibling, using the same row groups and two-column planner.
D3d12Dispatch d3d12_mm_dispatch_4col(std::int64_t n, std::int64_t ncols);

// Kernel variant switch (plan 004): 0 = one-column kernels (default, the only
// path B=1 ever takes), 1 = q4_k two-column tile wherever applicable (bench
// experiment: the paired runner forces both variants to compare them),
// 2 = auto: two-column only on the explicit allowlist below (real decode).
// Setter clamps to 0/1/2. Same-package OLD/NEW comparison rides on this;
// production decode never changes it (stays 0) until the experiment reports.
void d3d12_set_kernel_variant(int v);
int d3d12_kernel_variant();

// Narrow z-0 dispatch record (first-divergence diagnosis): when enabled, the
// matmul record path logs one line per z-0 node with node/source names,
// buffer placement flags, shapes/strides, selected variant/threads, dispatch
// dims and root constants. Log-only, default off, gated to the exact tensor
// name so all other dispatches are untouched.
void d3d12_set_z0_dispatch_log(bool on);

// Narrow z-0 placement pin (first-divergence diagnosis): when enabled, the
// z-0 node is refused D3D12 execution (falls back to the CPU kernel the
// refused f32 shapes already use). Tests whether the B3/B4 width variation
// follows placement (split D3D12/CPU execution) or persists on one backend.
// Log-only sibling above tells which backend ran it. Default off.
void d3d12_set_z0_pin_cpu(bool on);
// Plan 008 complete island: SWIGLU + ADD + RMS_NORM + MUL on D3D12. Default off.
void d3d12_set_island_enabled(bool on);
// Diagnostic-only (bench): allow FFN SWIGLU in MTP for the fixed-history repro.
void d3d12_set_swiglu_mtp_diag(bool on);
// Owner audit: sample the product SWIGLU inputs' real range (diagnostic-only).
void d3d12_set_swiglu_range_diag(bool on);
bool d3d12_swiglu_mtp_diag_enabled();

// Queue-fence wait-policy knob (plan004 rev101 experiment): -1 (default,
// knob file absent) keeps today's unbounded spin exactly as-is. A value >= 0
// caps the GetCompletedValue spin at that many microseconds before falling
// back to the event wait (0 = event wait immediately). Fence completion
// semantics and numerics are identical on every path; set once before
// decode, read per graph sync. Device knob: d3d12spinwait.txt (decimal
// microseconds; anything else = default).
void d3d12_set_spin_wait_us(int spin_us);
int d3d12_spin_wait_us();
// Explicit allowlist for variant 2 (real decode): measured winning q4_k
// (type, N, K, B) combinations from the rev57 same-run pairs, each with pair
// deltas far from zero. Anything unlisted — B=1, B=4, prefill widths,
// small shapes, other types — stays OLD until measured and validated.
// A hardcoded table, not a rule: broadening it is a code change with fresh
// evidence, never a knob edit.
bool d3d12_2col_allowlisted(ggml_type t, int n, int k, int ncols);
// True when both two-column q4_k blobs created their PSOs. A paired case
// with no NEW PSO reports an error row instead of silently running OLD.
bool d3d12_2col_available();
// Actual two-column dispatches since backend start (resets with the other
// lifetime counters). Rows take deltas around their timed runs: that is what
// really ran, regardless of the requested variant.
std::uint64_t d3d12_2col_matmuls();
// One-line resource record for the log: 2col blob bytes, cbuffer dwords,
// groupshared bytes per width, per-thread accumulator floats old vs new.
std::string d3d12_2col_info();

// Read-only accessors for the backend's lifetime counters, so a caller can take a
// delta around one phase (prefill, verify, draft) instead of reading a total that
// backend_free later zeroes. Cheap and side-effect free; 0 on a CPU host.
std::uint64_t d3d12_graph_calls();

// Lifetime wall time of graph_compute on this backend (submission + wait +
// driver), ms. Snapshotted as deltas around a phase call so it never overlaps
// the phase's own wall timer twice: gpu_ms <= wall_ms <= caller's verify_ms.
double d3d12_wall_ms();
std::uint64_t d3d12_matmul_count();
double d3d12_gpu_ms();

// Plan 006 C8: GPU timestamp queries are instrumentation and run on every
// graph_compute when enabled. `enabled=false` skips query/resolve/readback
// entirely; the backend then reports the timing as unavailable instead of a
// measured zero. Default is enabled, which preserves the historical behavior
// until an A/B proves the disable is worth changing.
void d3d12_set_gpu_timestamps(bool enabled);
bool d3d12_gpu_timestamps_enabled();

// Valid-sample accumulation state for one backend timeline: exactly one
// begin_call() per graph_compute, then exactly one add_sample(ms) when a
// timestamp was captured for THAT call or add_unavailable() when queries are
// disabled or the readback failed. Never accumulates a previous call's sample
// (the rev130 stale-sample defect). Pure and host-testable.
struct GpuTimingState {
    double last_ms = 0.0;          // valid sample of the most recent call, else 0
    double total_ms = 0.0;         // sum of valid samples only
    std::uint64_t valid = 0;       // calls that produced a sample
    std::uint64_t unavailable = 0; // calls with no sample (disabled or Map failed)
    void begin_call() {
        last_ms = 0.0;
    }
    void add_sample(double ms) {
        last_ms = ms;
        total_ms += ms;
        ++valid;
    }
    void add_unavailable() {
        last_ms = 0.0;
        ++unavailable;
    }
    void reset() {
        last_ms = 0.0;
        total_ms = 0.0;
        valid = 0;
        unavailable = 0;
    }
};

// Per-shape histogram (plan 003 stage 2). Aggregates inside graph_compute into a
// buffer keyed by (context, phase, type, N, K, B, threads); d3d12_shape_drain()
// emits it at a phase/generation boundary, outside the compute lock and the decode
// loop. Emission is PAGINATED: every key is written across as many records as it
// takes, each record carrying part=i/n and keys_total, so a consumer verifies
// completeness from the lines instead of trusting an arbitrary cap. ON by default;
// `off` disables collection entirely for the instrumentation-cost arm.
void d3d12_set_shape_log(bool off);
bool d3d12_shape_drain(const char* label);

// Context/phase tags for the histogram. set_scope is called around each target and
// drafter call and restored by the caller (RAII), so a tag never leaks from one
// call site into the next.
void d3d12_set_scope(const char* ctx, const char* phase);
void d3d12_get_scope(const char** ctx, const char** phase);

// Kernel width for a K: 128 threads (8 chunks in flight) once K has at least
// kD3d12LongKChunks chunks, else 64 — measured both ways in D2a runs 1 and 2.
int d3d12_mm_threads(std::int64_t k);

// Host emulation of the shaders' per-thread lane mapping, unaligned loads and
// algebra: y[c*y_stride + r] = sum_k W[r,k] * x[c*x_stride + k]. `w` holds N
// rows of `w_row_bytes` packed ggml blocks. Tests compare it with ggml's own
// dequantizers; the console selftest compares the GPU with the same reference.
void d3d12_mmv_emulate(ggml_type t, const std::uint8_t* w, std::size_t w_row_bytes, const float* x,
                       std::size_t x_stride, float* y, std::size_t y_stride, int n, int k,
                       int ncols);

// Mirrors Q6 tile row/column groups, tail guards and OLD per-column FP32 sums.
void d3d12_q6_tile_emulate(const std::uint8_t* w, std::size_t w_row_bytes, const float* x,
                           std::size_t x_stride, float* y, std::size_t y_stride, int n, int k,
                           int ncols, int columns);

// Median and max-min range of one timed block (plan 004 variance): the
// shape-cost bench reports medians over consecutive timed runs, not single
// exploratory samples. Empty input yields 0. Even counts take the lower
// middle; the bench only uses odd repeat counts.
double d3d12_block_median(const std::vector<double>& v);
double d3d12_block_range(const std::vector<double>& v);

// Bounded weight generation for the shape-cost bench (plan 004): quantizes
// the N x K weight matrix into `dst` (ggml_nbytes of them) from one
// std::mt19937 seeded only by the shape, holding at most
// kD3d12GenScratchBytes of fp32 source at a time. A full n*k source never has
// to exist: the Q6_K 248320x2560 lm_head would need 2542.8 MB of it. The
// bytes are identical to one ggml_quantize_chunk(0, n, k) over the whole
// matrix whenever k % ggml_blck_size(type) == 0, because every supported
// quantizer with a null imatrix walks the flat nrow*n_per_row stream, and
// row boundaries are super-block boundaries exactly then. The rng ends in
// the same state as the one-shot path, so the draws that follow the weights
// (activations, replicates) stay the same numbers. False on a bad type, a
// bad shape, a null dst, or a k that does not divide into blocks.
inline constexpr std::size_t kD3d12GenScratchBytes = 16u << 20;
bool d3d12_gen_weights(ggml_type t, int n, int k, std::mt19937& rng, void* dst);

// Control for the test: the original one-shot generation (a full n*k fp32
// vector, then one ggml_quantize_chunk). Same contract, same seed encoding.
bool d3d12_gen_weights_reference(ggml_type t, int n, int k, std::mt19937& rng, void* dst);

// Register the backend with ggml (idempotent). False when D3D12 is unavailable
// (always on non-Windows). The device appears as a GPU device named "D3D12".
bool ggml_d3d12_register();

// --- Console selftest (d3d12be.flag) ---

struct D3d12SelftestRow {
    std::string type; // q4_0 | q4_k | q5_k | q6_k | q8_0
    int n = 0, k = 0, ncols = 0;
    double rel_err = 0.0; // max |gpu - ref| / max |ref|
    double gpu_ms = 0.0;  // GPU timestamp time of the last compute, ms
    double packed_gbs = 0.0;
    bool ok = false;
    bool d3d12_ran = false;
    std::string error;
    // Shape-cost diagnostics (plan 004), ignored by the D2a formatter: wall
    // time of the timed compute next to the GPU timestamp (the gap is queue
    // and submission overhead), the packed weight size, and the process peak
    // working set at this row. The peak is monotonic — a high-water mark —
    // so only a rise means this case added memory; an early small row just
    // reports whatever an earlier phase left behind.
    double wall_ms = 0.0;
    double cpu_ms = 0.0; // GDN only: six-thread CPU-op median, logged outside the MMV CSV schema
    double weight_mb = 0.0;
    double peak_ws_mb = 0.0;
    // Timed-block dispersion (plan 004 variance): max-min across the timed
    // runs of this row. Zero on single-sample rows (all D2a gate rows).
    double gpu_ms_range = 0.0;
    double wall_ms_range = 0.0;
    // Kernel experiment (plan 004): variant 0 = OLD one-column, 1 = NEW
    // two-column; pair is the pair index inside a paired block, -1 on summary
    // rows (gate rows and single-variant medians). Pair rows carry one timed
    // sample each with zero ranges; the script aggregates pairs.
    int variant = 0;
    int pair = -1;
    // Actual NEW tile dispatches during this row's timed runs (counter delta,
    // not the request): Q4 two-column or Q6 two/four-column. B1 rows read 0;
    // a NEW pair row reads exactly its timed runs. Column name stays stable.
    std::uint64_t twocol = 0;
    // Stride padding (plan 004 padded gate): "x_pad/y_pad/w_pad" — floats of
    // X/Y column padding, bytes of weight row padding. "0/0/0" = contiguous.
    std::string pads = "0/0/0";
};

inline constexpr double kD3d12SelftestRelTol = 1e-2;

const char* d3d12_selftest_csv_header();
std::string format_d3d12_selftest_row(const D3d12SelftestRow& r, const char* host_label);

// Second CSV (d3d12sc-result.csv): the D2a rows above keep their exact
// schema, the shape-cost rows carry wall/weight/peak next to them.
const char* d3d12_shapecost_csv_header();
std::string format_d3d12_shapecost_row(const D3d12SelftestRow& r, const char* host_label);

// Windows: run every type x shape through the backend and append rows.
// Non-Windows: one row with d3d12_ran=false.
void run_d3d12_selftest(std::vector<D3d12SelftestRow>* out);

// GDN outputs and every written snapshot vs the actual ggml CPU op. Linux
// checks the shader's host emulation; Windows dispatches the GPU. Rows are
// also appended to the reported shape-cost CSV, outside the MMV bandwidth gate.
void run_d3d12_gdn_selftest(std::vector<D3d12SelftestRow>* out);

// The 11 real (type, N, K) triples of the tttarget histogram, costed at
// batch widths 1/2/3/5: smallest weights first, the 248320x2560 lm_head last.
// Each row is the median of kD3d12ShapeCostRepeats consecutive timed runs
// with the max-min range next to it. Reported, never gated; the D2a gate
// above keeps its exact inputs.
inline constexpr int kD3d12ShapeCostRepeats = 5;
inline constexpr int kD3d12ShapeCostPairs = 4;
void run_d3d12_shape_cost(std::vector<D3d12SelftestRow>* out);

} // namespace xllama
