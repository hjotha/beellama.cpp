// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// First-numeric-divergence measurement (plan 004): compare the captured B3
// block ([8772,279,84810] at 142..144) against the B4 block (same plus the
// rejected 11 at 145) from an identical prefix, tensor by tensor in graph
// order, using the llama_context_params cb_eval scheduler callback only
// around those two block decodes. Only rows for the first three tokens
// (identical inputs, causal prefix) are compared, so any difference is batch
// width/composition arithmetic, never content.
//
// Comparability contract (checked loudly, never assumed): both runs must
// produce the same node count; per index the op, type and all extents except
// exactly one token axis (3 vs 4) must match. Tensors without a token axis
// (weights, scales) are skipped: they are equal by construction (same model,
// same prefix). Whole-model dumps are never written; only first-3-row
// slices (hash + optional F32 maxabs) and bounded summaries leave the device.
//
// Contamination control: each block also runs once WITHOUT the callback
// (full-graph scheduling). If callback-on differs from callback-off, the
// capture is discarded and reported, never used.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xllama {

// FNV-1a 64 over raw bytes.
uint64_t diverge_fnv(const void* data, size_t len);

// Token-axis rule over a B3/B4 shape pair: every differing axis must be a
// clean (3, 4) token pair; static axes (any equal value, including static 3
// or 4 extents) are ignored. Returns a bitmask of token axes (> 0;
// multi-axis tensors like [heads, T, T] compare the causal 3x3 block); -2
// when all four axes are equal (static tensor: skip without comparing); -1
// for genuine shape drift (caller fails loudly, never compares).
int diverge_token_axes(const int64_t n3[4], const int64_t n4[4]);

// Byte offsets (in increasing order) of the elements with index in
// [nstart, nstart+nkeep) along EVERY masked axis, for a tensor with extents
// ne and ggml-style byte strides nb. Element size esz is nb[0] for contiguous
// leading dims; the walk covers the full tensor and keeps matching offsets.
// Pure arithmetic, host-testable. nstart selects shifted windows (snapshot
// slots need B4 rows 1..3 for the same logical positions as B3 rows 0..2).
std::vector<size_t> diverge_slice_offsets(const int64_t ne[4], const size_t nb[4], size_t esz,
                                          int axes_mask, int64_t nstart, int64_t nkeep);

// Max |a-b| over float buffers.
double diverge_maxabs_f32(const float* a, const float* b, size_t n);

// One captured graph node (device side produces these during the cb_eval
// capture; raw tensor bytes are freed inline, only slices retained).
struct DivergeCapNode {
    int index = -1; // graph node order (ask sequence)
    std::string name;
    int op = -1; // ggml_op enum value
    std::string op_name;
    int64_t ne[4] = {0, 0, 0, 0};
    int type = -1; // ggml_type enum value
    struct AxisSlice {
        int axes = 0;       // bitmask compared (singleton or full candidate combo)
        int64_t nstart = 0; // window start along masked axes (1 for B4 shifted)
        uint64_t hash = 0;
        std::vector<float> floats; // F32 tensors only (for maxabs)
    };
    std::vector<AxisSlice> slices; // one per candidate mask (+shifted on B4)
};

// One export row (device side produces these, exporter validates them).
struct DivergeTensorRow {
    std::string branch; // "b3" or "b4"
    int rep = 0;
    int index = -1;      // graph node order
    std::string name;    // ggml tensor name (often empty: builders don't set names)
    int op = -1;         // ggml_op enum value
    std::string op_name; // ggml_op_name(op), resolved device-side
    int64_t ne[4] = {0, 0, 0, 0};
    int type = -1;            // ggml_type enum value
    int axes = 0;             // compared token-axis bitmask (diverge_token_axes)
    std::string skip_why;     // nonempty when skipped
    uint64_t hash = 0;        // FNV over the unshifted slice bytes
    std::vector<float> slice; // F32 slice floats (cleared on device after verdict)
    double maxabs = -1.0;     // unshifted maxabs (device-computed); -1 = none
    uint64_t hash_alt = 0;    // FNV over the shifted slice (B4 rows 1..3); B3 repeats hash
    double maxabs_alt = -1.0; // shifted maxabs (device-computed); -1 = none
};

struct DivergeCmp {
    double maxabs = 0.0;
    bool hash_match = false;
    bool argmax_match = false; // reserved: intermediates carry no argmax
};

// Pure comparison of two captured slices (same layout by contract).
DivergeCmp compare_diverge_rows(const DivergeTensorRow& a, const DivergeTensorRow& b);

// Pair one B3/B4 capture node into two export rows (b3 then b4) carrying the
// unshifted + shifted measurements. Structural/slice problems are marked as
// skips (the exporter fails loudly on those); value mismatches are evidence.
// Pure over the node structs: host-testable with hand-built fixtures.
void diverge_emit_pair(int rep, const DivergeCapNode& n3, const DivergeCapNode& n4,
                       std::vector<DivergeTensorRow>* out);

// (Tier verdict helper removed: min-rule retired in favor of provenance
// tiers below. A hash match IS value equality; alignment choice must come
// from tensor meaning, never from which alignment happens to match.)

// Narrow z-0 capture set (NARROW-TO-Z0): the layer-0 z projection plus its
// likely input activations, by stable builder name. Pure: host-tested.
bool diverge_want_narrow(const char* name);

// Tensor identity for src-meta lines (no data read: safe against graph
// memory reuse). Pure: host-tested byte-for-byte.
std::string diverge_tensor_id(const char* name, const int64_t ne[4], const char* type);

// Provenance tier for first-divergence ranking, by ggml op name:
// 1 = per-token/causal compute whose same-token rows are directly comparable
//     (MUL_MAT, MUL, ADD, RMS_NORM, UNARY, GLU, ROPE, SCALE);
// 2 = state-family or uncertain ops (VIEW, CPY, RESHAPE, TRANSPOSE, GET_ROWS,
//     SSM_CONV, fused recurrent, scatter, custom, anything unlisted):
//     presentation ops can re-present snapshot planes (proven rev82), so they
//     are never ranked; both alignments are retained as diagnostic rows.
// Conservative default: unknown names are tier 2 (excluded from ranking,
// never marked pass). Pure: host-tested.
int diverge_tier_by_name(const std::string& op_name);

// One validated export pair (device rows already carry the verdict; the
// exporter only validates layout/discipline and formats).
struct DivergePairOut {
    int rep = 0;
    int pair = 0;   // position within the rep
    int index = -1; // graph node index (sparse: never assumed == pair)
    std::string name;
    std::string op_name;
    int64_t ne_a[4] = {0, 0, 0, 0};
    int64_t ne_b[4] = {0, 0, 0, 0};
    int type = -1;
    int match = 0;        // 1 = stored hashes equal
    double maxabs = -1.0; // stored device verdict
    std::string note;     // "" or "skipped:<reason>"
};

struct DivergeRepSummary {
    int rep = 0;
    size_t n_tensors = 0;
    size_t n_matched = 0;
    size_t n_skipped = 0;
    int first_diff = -1; // pair position, -1 = none
};

struct DivergeExport {
    std::string fail_why;    // empty when ok
    int fail_pair = -1;      // pair position of the break, -1 when count-level
    DivergeTensorRow fail_a; // offending b3 row (valid iff fail_pair >= 0)
    DivergeTensorRow fail_b; // offending b4 row (valid iff fail_pair >= 0)
    std::vector<DivergeRepSummary> reps;
    std::vector<DivergePairOut> pairs; // emission order
};

// Validate the interleaved layout (per rep: N (b3,b4) pairs, 4N rows total
// over 2 reps) and build the export. Tensor value mismatches are evidence;
// layout/discipline breaks fail via fail_why. Pure: host-testable with the
// exact production logic.
DivergeExport diverge_pair_rows(const std::vector<DivergeTensorRow>& rows);

// Exact CSV field formats (header: rep,index,name,op,shape_b3,shape_b4,
// match,maxabs,note; shape fields RFC-4180 quoted). Pure: host-tested
// byte-for-byte.
std::string diverge_shape_string(const int64_t ne[4], int type);
std::string diverge_csv_line(const DivergePairOut& p);

// One-line row identity for failure logs. Pure: host-tested.
std::string diverge_row_string(const DivergeTensorRow& r);

// Runs the measurement: prefix = tokenized prompt + ctx_ids, then the B3
// block and the B4 block, each twice (callback off for contamination control,
// callback on for capture), on fresh contexts with identical flags
// (rs_seq/nextn as given). known_next guards the B3 rows (fail loudly).
// feed_ids[0..2] is the B3 block, feed_ids[3] the rejected 4th token.
// Rows arrive per rep: b3off, b3on, b4off, b4on logit rows, then the captured
// b3 tensor rows (in graph order), then the captured b4 tensor rows.
// fail_why aborts; tensor mismatches never abort (they are the result).
void run_diverge_measure(const std::string& model_name, const std::string& prompt_text,
                         const std::vector<int32_t>& ctx_ids, const std::vector<int32_t>& feed_ids,
                         const std::vector<int32_t>& known_next, int gpu_layers, int n_ctx,
                         int n_threads, int rs_seq, bool nextn_on,
                         std::vector<DivergeTensorRow>* out, std::string* fail_why);

// Narrow z-0 measurement (NARROW-TO-Z0): same fixture/flags/prefix/rows as
// run_diverge_measure, but capture takes only the z-0 neighborhood by name
// (z-0, norm-0, attn_norm-0) plus z-0 src identities from tensor metadata.
// Bounded by construction (<=3 small tensors per run); same interleaved
// layout, so the shared exporter validates it unchanged.
void run_znarrow_measure(const std::string& model_name, const std::string& prompt_text,
                         const std::vector<int32_t>& ctx_ids, const std::vector<int32_t>& feed_ids,
                         const std::vector<int32_t>& known_next, int gpu_layers, int n_ctx,
                         int n_threads, int rs_seq, bool nextn_on,
                         std::vector<DivergeTensorRow>* out, std::string* fail_why);

} // namespace xllama
