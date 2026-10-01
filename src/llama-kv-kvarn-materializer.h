#pragma once

// Backend-aware bounded read_block adapter for llama_kv_qx::convert.
//
// This adapter feeds the KVarN -> Qx converter from REAL KVarN records and
// stage tensors: it runs a bounded ggml_kvarn_materialize(+cast) graph on
// the source backend for one token window at a time and hands the converter
// the window as [head_dim, n_heads, n_window] F32 in the declared source
// (rotated) domain. Only the requested window goes to host memory; the
// records/stage tensors stay on their owning backend.
//
// Scope of this first contract:
//   - one layer / one source sequence, single stream, non-SWA;
//   - the owner supplies the EXACT ordered encoded cell IDs for the whole
//     prefix (one entry per logical token): plain PHYSICAL cell IDs for
//     cells served from sealed records, explicit F16 stage-slot encodings
//     (llama_kvarn_encode_stage_cell) for cells served from the stage, or -1
//     for holes (output zeros; use for fixed-shape padding only). The map
//     may reference sparse/high physical cells (e.g. a compact read plan
//     after rollback); the adapter does NOT infer stage ownership by position
//     modulo anything, the map is the authority, and the sink must be encoded
//     explicitly (stage slot 0) by the owner;
//   - the map is validated (physical cells < records groups x 128 -- the
//     physical capacity -- and explicit slots < stage_groups) BEFORE any
//     backend compute, so invalid input cannot reach the materializer's
//     GGML_ABORT;
//   - source leaves must be contiguous, allocated, and their buffer types
//     must be accepted by the source backend (native
//     ggml_backend_supports_buft check, not backend-name based: pinned or
//     shared host buffers the device genuinely accepts keep working). The
//     bounded graph's nodes are preflighted with ggml_backend_supports_op
//     before any allocation/compute; unsupported nodes return
//     unsupported_backend instead of aborting;
//   - windows are served with size <= max_window <= 1024; the materialize
//     graph is built once per window size and reused across windows;
//   - the adapter only reads. It never submits producer/store graphs through
//     the borrowed source tensors and requires them to be plain leaves
//     (op == GGML_OP_NONE), so no old writes can re-execute.
//
// Precision-tail contract (IMPORTANT)
// -----------------------------------
// KVarN attention keeps an exact overlay / intrinsic 128-token tail in
// addition to the F16 stage and the sealed records. The body materializer
// alone dequantizes sealed records even when effective attention uses
// exact-tail rows for those cells. This adapter does NOT claim that bare
// records + stage preserve every effective attention value: it materializes
// exactly what the owner's index map + tensors represent. If effective
// attention reads exact rows (e.g. the intrinsic tail) that are not present
// in records/stage, the owner MUST supply the optional exact-row overlay
// callback; rows it returns (ORIGINAL domain) are rotated by the adapter
// into the declared source (rotated) domain before being returned, keeping
// the whole window in one consistent domain for the converter.
//
// Lifetime and synchronization (IMPORTANT)
// ----------------------------------------
// The caller guarantees that records/stage/backend are immutable and
// synchronized for the materializer's lifetime (no producer writes in
// flight) and that they outlive it. The adapter allocates only its own
// window graphs/buffers; it never frees the source tensors or their buffers,
// never writes into records/stage, and never touches any live cache.
//
// Status and exceptions (IMPORTANT)
// ---------------------------------
// The interface is status-based and never throws: std::exception escaping
// the implementation (e.g. allocation failure) is caught and reported as
// workspace_overflow. A FAILED set_prefix invalidates the installed prefix
// map (the map is cleared and serve() returns false with invalid_args until
// a successful set_prefix); a failed serve() keeps the installed map (the
// call is retryable) but leaves last_status set. The staged window written
// into req.out_f32 before a failure is only ever the converter's staged
// buffer, never a live cache.

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-kv-qx-convert.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llama_kv_qx {

enum class mat_status {
    ok = 0,
    invalid_args,          // null tensor/backend/index map, non-leaf source, bad window
    invalid_geometry,      // shape/type/contiguity/allocation/buft mismatch
    invalid_indices,       // index map out of bounds (position/slot/group/hole misuse)
    unsupported_backend,   // the source backend cannot run the bounded graph nodes
    workspace_overflow,    // allocation failure or too many window sizes
    compute_failure,       // backend graph compute failed
    non_finite_input,      // materialized window (or exact overlay) is non-finite
};

const char * mat_status_name(mat_status st);

constexpr uint32_t LLAMA_KV_MATERIALIZER_MAX_WINDOW = 1024;

struct materializer_config {
    uint32_t head_dim = 0;       // full head width (record_dim * head_slices)
    uint32_t n_heads = 0;        // LOGICAL heads; physical = n_heads * head_slices
    bool     value = false;      // false = K, true = V
    uint32_t record_dim = 128;   // KVarN record width (64 or 128)
    uint32_t head_slices = 1;    // head_dim / record_dim, in {1, 2, 4}
    int      bits = 4;           // KVarN record bits (2..8)
    int      stage_groups = 3;   // stage depth (>= 2)
    uint32_t max_window = 256;   // max tokens per served window, 1..1024
};

struct materializer_source {
    ggml_tensor * records = nullptr;  // I8 leaf [record_bytes, phys_heads, groups_per_stream]
    ggml_tensor * stage = nullptr;    // F16 leaf [record_dim, phys_heads, 128*stage_groups]
    ggml_backend_t backend = nullptr; // owning backend (CPU mandatory, CUDA/Vulkan via API)
};

// Optional exact-row overlay for precision-tail cells. Called once per token
// of a served window (before it is returned). `rows` is [head_dim * n_heads]
// floats, ORIGINAL domain, head-major (rows[d + h*head_dim]). Return true to
// override the materialized body for that token; return false to keep the
// body value. The adapter rotates returned rows into the declared source
// (rotated) domain. Must not throw.
using exact_token_fn = bool (*)(void * user, uint32_t token, float * rows);

class materializer {
public:
    // Validates the configuration and the source tensors (shapes, types,
    // leaf-ness, single stream) without touching backend memory. Check
    // `valid()`/`last_status()` before use.
    materializer(const materializer_config & cfg, const materializer_source & src);

    ~materializer();

    materializer(const materializer &) = delete;
    materializer & operator=(const materializer &) = delete;

    bool valid() const { return last_ == mat_status::ok; }
    mat_status last_status() const { return last_; }
    const char * last_status_name() const { return mat_status_name(last_); }

    // Installs the effective prefix index map (copied) and the optional
    // exact-row overlay. Every entry is validated eagerly (positions <
    // n_tokens, explicit slots < stage_groups, record groups within the
    // records capacity, no store-cell encodings) so a bad map fails before
    // any backend compute. `exact`/`exact_user` are kept by reference and
    // must outlive the serving calls.
    mat_status set_prefix(const int64_t * indices, uint32_t n_tokens,
            exact_token_fn exact = nullptr, void * exact_user = nullptr);

    // read_block adapter body for llama_kv_qx::convert. Serves the window
    // [req.t0, req.t0 + req.n_tokens) from the installed prefix map.
    // Returns false (with last_status set) when the window cannot be served;
    // the converter then aborts with its staged prefix intact. Never touches
    // the live cache.
    bool serve(void * user, const block_request & req);

    // Diagnostic: number of cached per-size materialize graphs (reuse count).
    size_t graph_count() const;

private:
    mat_status validate_source() const;
    mat_status ensure_graph(uint32_t n_window);
    bool serve_impl(const block_request & req);

    materializer_config cfg_;
    materializer_source src_;

    std::vector<int64_t> prefix_;  // effective index map (copied)
    uint32_t n_tokens_ = 0;
    exact_token_fn exact_ = nullptr;
    void * exact_user_ = nullptr;

    struct graph_entry {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor * indices = nullptr;  // I64 [n_window]
        ggml_tensor * out = nullptr;      // F32 [head_dim, n_heads, n_window]
        ggml_cgraph * graph = nullptr;
        uint32_t n_window = 0;
    };
    std::vector<graph_entry> graphs_;
    std::vector<int64_t> win_indices_;  // per-window index buffer (reused)

    std::vector<float> exact_scratch_;
    mat_status last_ = mat_status::ok;
};

// Static adapter for llama_kv_qx::convert: pass `&materializer` as user and
// this function as read_block.
bool materializer_read_block(void * user, const block_request & req);

} // namespace llama_kv_qx