#pragma once

// One-way PURE KVarN -> MIXED (KVarN local + standard Qx remote) prefix
// handoff engine. Pure GGML: depends only on stable ggml APIs and the
// reviewed llama_kv_qx converter/materializer modules; no llama context or
// memory ABI types. The server glue (tools/server/server-mixed-kv-handoff)
// supplies authoritative tensors, descriptors and index maps; this module
// validates, plans, converts (bounded) and stages the transaction.
//
// Contract:
//   - source prefix is the evaluated checkpoint length (n_tokens <= source
//     LCP), parsed by the CALLER into a CPU hybrid memory (authoritative
//     legacy parser for KV + recurrent state). Only borrowed leaf tensors
//     are passed here.
//   - local KVarN layers are copied COMPRESSED (records/stage/exact tail
//     bytes) without requantizing; remote layers are converted with the
//     reviewed materializer + convert_to_sink (bounded blocks <= 1024
//     tokens, no full Qx output buffer) into private candidate buffers.
//   - one-way only: destination kvarn bits must match the source descriptor;
//     a destination without standard layers (mixed->pure) or with changed
//     Qx types is rejected, never silently requantized.
//   - the Qx payload layout is rawstorage per TOKEN row
//     ([head_dim*n_heads] elements per row, quantized), matching the
//     standard-cache rawstorage row stride; no permuted attention view is
//     written.
//   - transaction: prepare() validates and runs the remote conversions
//     (writes go only into private candidate tensors); commit() applies the
//     staged local copies in one step; discard() drops everything. A failed
//     prepare leaves the candidate uncommitted (caller frees it); the source
//     compressed snapshot stays untouched and reusable for retry.
//   - memory: no full F32 prefix, no full Qx output, no duplicate of the
//     compressed source (staged copies are borrowed; apply uses bounded
//     chunks).

#include "ggml.h"
#include "llama-kv-qx-convert.h"
#include "llama-kv-kvarn-materializer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llama_kv_handoff {

enum class handoff_status {
    ok = 0,
    invalid_args,
    invalid_geometry,
    invalid_type,
    invalid_domain,
    invalid_indices,
    no_compatible_prefix,
    mixed_to_pure_rejected,   // destination has no standard remote layers
    changed_qx_rejected,      // destination Qx types differ from the recorded layout
    changed_kvarn_rejected,   // destination KVarN bits/params differ from the source
    destination_missing,
    payload_overflow,
    workspace_overflow,
    callback_failure,
    sink_failure,
    compute_failure,
};

const char * handoff_status_name(handoff_status st);

// ---------------------------------------------------------------------------
// Source descriptor: recorded from the OLD (pure KVarN) context BEFORE it is
// destroyed. KVarN bits are recorded explicitly -- never derived from the F16
// metadata profile.
// ---------------------------------------------------------------------------

struct handoff_source_desc {
    uint32_t n_tokens = 0;        // evaluated prefix to transfer (checkpoint length)
    uint32_t n_ctx_seq = 0;       // source capacity (old context)
    uint32_t n_seq_max = 1;
    bool kv_unified = true;
    int32_t kvarn_type = 0;       // llama_kvarn_type as int32
    int32_t key_bits = 0;         // recorded KVarN bits (NOT derived from metadata)
    int32_t value_bits = 0;
    int32_t stage_groups = 0;
    int32_t tail_groups = 0;
    uint32_t exact_tail_tokens = 0; // intrinsic 128 (+rollback) exact tail
    bool has_cell_ext = false;
    uint64_t source_state_bytes = 0; // compressed snapshot payload bytes (diagnostic)
};

// Per-layer source KVarN geometry (mirror of kvarn_layer_layout).
struct handoff_source_layer {
    uint32_t il = 0;
    uint32_t record_dim_k = 0;
    uint32_t record_dim_v = 0;
    uint32_t k_slices = 0;
    uint32_t v_slices = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim_k = 0;
    uint32_t head_dim_v = 0;
};

// Per-layer remote (standard Qx) destination configuration. Rotation widths
// come from the candidate's authoritative standard layer layout -- read, not
// guessed (tested Qwen D256: K 256 / V 64; 0 keeps the original domain).
struct handoff_remote_layer {
    uint32_t il = 0;
    int32_t qx_type_k = 0;   // ggml_type as int32 (whitelisted)
    int32_t qx_type_v = 0;
    uint32_t rotation_k = 0;
    uint32_t rotation_v = 0;
    bool v_transposed = false;
    uint32_t head_dim_k = 0;
    uint32_t head_dim_v = 0;
    uint32_t n_head_kv = 0;
    size_t row_bytes_k = 0;  // quantized bytes per token row (from layout strides)
    size_t row_bytes_v = 0;
};

// ---------------------------------------------------------------------------
// Prefix selection: pick the best eligible host checkpoint <= lcp.
// Candidates are evaluated in order; the LAST compatible prefix wins (longest
// eligible). Eligible: host_only, non-empty, n_tokens <= lcp, pos_max ==
// n_tokens-1. Returns the selected token count (0 = no compatible prefix).
// ---------------------------------------------------------------------------

struct handoff_prefix_candidate {
    uint32_t n_tokens = 0;
    int32_t pos_max = -1;
    bool host_only = false;
};

uint32_t handoff_select_prefix(
        const handoff_prefix_candidate * candidates,
        size_t count,
        uint32_t lcp);

// ---------------------------------------------------------------------------
// One-way plan: decides, per attention layer of the destination, whether it
// is local (compressed copy) or remote (Qx conversion), and validates the
// whole configuration before any conversion runs.
// ---------------------------------------------------------------------------

struct handoff_plan {
    handoff_status st = handoff_status::ok;
    std::vector<uint32_t> local_layers;
    std::vector<handoff_remote_layer> remote_layers;
    uint32_t n_tokens = 0;
    uint64_t qx_bytes = 0; // total remote Qx payload bytes (diagnostic)
};

handoff_plan handoff_plan_build(
        const handoff_source_desc & source,
        const std::vector<handoff_source_layer> & source_layers,
        const std::vector<handoff_remote_layer> & remote_layers,
        uint32_t dest_kvarn_key_bits,
        uint32_t dest_kvarn_value_bits,
        bool dest_has_standard_cache);

// ---------------------------------------------------------------------------
// Transaction.
// ---------------------------------------------------------------------------

// Exact-tail overlay source: supplies ORIGINAL-domain rows for tail tokens.
// rows[d + h*head_dim] head-major; return false to keep the materialized
// body value for that token. Called per K and V with the same token set.
using handoff_tail_fn = bool (*)(void * user, uint32_t token, uint32_t head_dim,
        uint32_t n_heads, float * rows);

struct handoff_convert_params {
    // source (CPU hybrid memory) layer tensors -- borrowed leaves
    ggml_tensor * records = nullptr;
    ggml_tensor * stage = nullptr;
    ggml_backend_t source_backend = nullptr;
    // effective physical index map (one entry per logical token)
    const int64_t * index_map = nullptr;
    uint32_t n_tokens = 0;
    // exact tail overlay (original domain); may be null
    handoff_tail_fn tail = nullptr;
    void * tail_user = nullptr;
    // destination private candidate tensor (standard-cache rawstorage,
    // token rows) + offset base (usually 0)
    ggml_tensor * dst = nullptr;
    size_t dst_offset = 0;
    // Optional destination row map: dst_row_of_token[logical token] = physical
    // row in the destination storage (null requires the DENSE mapping
    // physical row == logical token, which the engine enforces with bounds).
    const uint32_t * dst_row_of_token = nullptr;
    // geometry (from the authoritative layouts)
    uint32_t head_dim = 0;
    uint32_t n_heads = 0;
    bool value = false;
    uint32_t record_dim = 128;
    uint32_t head_slices = 1;
    int kvarn_bits = 4;
    int stage_groups = 3;
    int32_t qx_type = GGML_TYPE_Q4_0; // ggml_type as int32
    uint32_t rotation_wht = 0;        // destination standard rotation width
    size_t row_bytes = 0;             // quantized bytes per token row
    uint32_t max_window = 64;         // bounded conversion window (<= 1024)
};

class handoff_transaction {
public:
    handoff_transaction() = default;
    ~handoff_transaction() { discard(); }

    handoff_transaction(const handoff_transaction &) = delete;
    handoff_transaction & operator=(const handoff_transaction &) = delete;

    // Stages a compressed copy (records/stage/tail/RS bytes): borrowed source
    // range applied to the destination tensor in one commit, bounded chunks.
    handoff_status add_copy(ggml_tensor * src, size_t src_offset,
            ggml_tensor * dst, size_t dst_offset, size_t bytes);

    // Runs a remote layer conversion during prepare(): bounded windows
    // through the reviewed materializer + convert_to_sink into the private
    // candidate tensor. The sink writes at logical row offsets.
    handoff_status add_convert(const handoff_convert_params & params);

    // Validates every staged op and executes all conversions (sink writes go
    // only into private candidate tensors). On failure no staged copy has
    // been applied and discard() drops the transaction.
    handoff_status prepare();

    // Applies all staged copies (bounded chunks) in one step. Must only be
    // called after a successful prepare().
    handoff_status commit();

    // Drops staged ops. The caller owns the candidate tensors and frees them
    // on failure.
    void discard();

    // Diagnostics are split honestly: source_bytes counts the COMPRESSED source
// bytes actually read (stage/records/tail slices + RS), f32_traffic_bytes
// counts the materialized F32 windows of the conversions, and
// scratch_bytes is an ESTIMATE of the bounded peak scratch.
uint64_t source_bytes() const { return source_bytes_; }
    uint64_t f32_traffic_bytes() const { return f32_traffic_bytes_; }
    uint64_t scratch_bytes() const { return scratch_bytes_; }
    double convert_ms() const { return convert_ms_; }
    handoff_status last_status() const { return last_; }
    size_t op_count() const { return copies_.size() + converts_.size(); }
    size_t convert_count() const { return converts_.size(); }
    size_t copy_count() const { return copies_.size(); }
    bool prepared() const { return prepared_; }
    bool committed() const { return committed_; }
    bool failed() const { return failed_; }

private:
    struct copy_op {
        ggml_tensor * src = nullptr;
        size_t src_offset = 0;
        ggml_tensor * dst = nullptr;
        size_t dst_offset = 0;
        size_t bytes = 0;
    };
    struct convert_op {
        handoff_convert_params params;
    };

    std::vector<copy_op> copies_;
    std::vector<convert_op> converts_;
    bool prepared_ = false;
    bool committed_ = false;
    bool failed_ = false; // terminal: further mutation/compute is rejected
    handoff_status last_ = handoff_status::ok;
    uint64_t source_bytes_ = 0;
    uint64_t f32_traffic_bytes_ = 0;
    uint64_t scratch_bytes_ = 0;
    double convert_ms_ = 0.0;
};

// Diagnostic marker line (completed commit only).
std::string handoff_marker(
        uint64_t reused_tokens,
        uint32_t converted_layers,
        uint64_t source_bytes,
        uint64_t scratch_bytes,
        double convert_ms);

// Failure / no-source diagnostic line.
std::string handoff_marker_failed(const char * reason);
std::string handoff_marker_no_source();

// Bounded recurrent-state transfer budget derived from the source's ACTUALLY
// allocated recurrent geometry plus a metadata margin, clamped by the given
// physical RAM budget (no fixed 4B-model threshold).
uint64_t handoff_rs_budget(uint64_t rs_allocated_bytes, uint64_t physical_ram_budget);

} // namespace llama_kv_handoff