// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// T_target(B): the cost of ONE target decode of B tokens, measured on the real
// context (same D3D12 config, same KV and recurrent state as the gate), not on a
// synthetic matmul. Plan 003 stage 2.
//
// Method, fixed by the plan and not by convenience:
//   * a fixed token prefix is decoded to rebuild the state; the state is rebuilt
//     by RECREATING the context for every measurement, and prefix, context
//     creation, validation and drain are all OUTSIDE the timed interval;
//   * a sequential reference produces the candidate ids, one token per decode, and
//     keeps the logits that predict each of them;
//   * for each B the timed call is ONE llama_decode of the first B candidate ids
//     at their positions with logits requested on every row;
//   * equivalence is checked on the LOGITS, not the argmax alone: finite rows,
//     maxabs/maxrel against the reference and the top-2 margin per row, under a
//     declared tolerance. Divergence is reported with its measured values.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace xllama {

// FNV-1a 64 over the whole file, streamed. A cheap identity fingerprint for
// the GGUF when a SHA-256 would mean a multi-GB transfer; labelled as such
// wherever recorded. Shared so every bench binds the same fixture the same way.
std::string fnv1a64_file(const std::string& path, std::uint64_t* size_out);

struct TtargetRow {
    int b = 0;     // batch width of the timed call (1, 2, 3 or 5)
    int rep = 0;   // repetition index within this B (-1 = warm-up)
    int order = 0; // TRUE sequential position in the shuffled schedule (1-based)
    int order_seed = 0;
    int prefix_len = 0;
    int n_decoded = 0;         // candidate ids actually decoded in the timed call
    int argmax_match = 0;      // 1 when every row passed argmax AND the logits tolerance
    int first_mismatch = -1;   // row index of the first disagreement, else -1
    float logit_maxabs = 0.0f; // worst |Δlogit| over all compared rows
    float logit_maxrel = 0.0f; // worst |Δlogit| / max|ref logit|
    float margin_min = -1.0f;  // smallest top-2 margin seen (reference side)
    double wall_ms = 0.0;      // steady_clock around the single llama_decode
    double gpu_ms = 0.0;       // d3d12 GPU-time delta over the same call
    std::uint64_t calls = 0;   // d3d12 graph_compute delta
    std::uint64_t matmuls = 0; // d3d12 matmul delta
    int load_mtp = 1;          // 1 = target loaded with the MTP head, like the gate
    std::string prefix_ids;
    std::string candidate_ids;
    std::string config;   // effective n_ctx/n_batch/n_ubatch/n_rs_seq/backend
    std::string msix_sha; // from the host, recorded for provenance
    std::string gguf_fnv1a64;
    std::uint64_t gguf_bytes = 0;
    std::string error;
};

// Measures B in {1,2,3,5} with |reps| independent repeats each, in a shuffled
// order per pass, one warm-up pass first. |n_rs_seq| mirrors the speculative
// window the gate uses. |seed_ids|, when non-empty, replaces the tokenized prefix.
// |msix_sha| is recorded in every row for provenance (empty becomes "unknown").
// |widths| restricts the schedule to a subset (a B1 smoke before the full matrix);
// empty means {1,2,3,5}.
void measure_ttarget(const std::string& model_path, int n_gpu_layers, int n_ctx, int n_batch,
                     int n_ubatch, int n_rs_seq, int n_threads, int reps,
                     const std::vector<int>& widths, const std::vector<std::int32_t>& seed_ids,
                     const std::string& msix_sha, std::vector<TtargetRow>* out);

const char* ttarget_csv_header();
std::string format_ttarget_row(const TtargetRow& r, const char* host_label);

} // namespace xllama
