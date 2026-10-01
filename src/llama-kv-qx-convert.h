#pragma once

// Bounded CPU conversion of a materialized KVarN KV prefix into standard
// row-quantized Q4_0/Q5_0/Q6_0/Q8_0 payloads for the remote-layer handoff.
//
// The converter never materializes the whole prefix in F32: it requests one
// token window at a time through a caller-provided callback and quantizes
// window rows incrementally. The wrapper (cache layer) keeps the KVarN
// cells/positions; remote layers own a standard Qx cache with their own
// Hadamard and write fresh K/V directly, without an intermediate KVarN
// quantization step. This module only produces the historical-prefix payload.
//
// Scope of this first contract:
//   - single sequence, single stream, non-SWA (everything else is rejected
//     explicitly);
//   - KVarN source materialized in the rotated domain (the default emitted by
//     llama_kv_cache_kvarn::materialize, op_params[4] = 1) or, explicitly
//     declared, already in the original domain;
//   - destination payload rows are TOKENS: row t holds all heads
//     concatenated head-major ([head_dim*n_heads] floats), quantized with the
//     canonical Qx row quantizer (block size 32, e.g. Q4_0/Q5_0/Q6_0/Q8_0).
//     The attention views of that storage are permuted; this module only
//     touches the raw contiguous storage described here.
//
// Source callback contract (IMPORTANT)
// ------------------------------------
// `read_block` must return the FULL KVarN materialization of its token window
// in the rotated domain, exactly as ggml_compute_forward_kvarn_materialize
// produces it for those cells: sealed records (completed groups), the
// permanent group-0 sink, the stage tail groups, and the intrinsic 128-token
// tail resolved through the index map (including explicit stage-slot
// encodings). It is NOT sufficient to return only the F16 staging rows: cells
// whose values live in sealed records would be wrong. In the serving
// integration the wrapper materializes each window through the existing KVarN
// materializer using its index map; this header only fixes the window layout
// and the domain contract.
//
// The records/stage/index adapter that feeds `read_block` from the live cache
// is NOT part of this module and is not implemented here; it belongs to the
// cache integration. This module is the standalone conversion primitive only:
// it converts whatever full materialization the caller supplies.
//
// `read_block` is expected not to throw. If an exception escapes it, the
// converter catches std::exception and returns callback_failure without
// publishing the failing window. All size computations (payload, bounded
// scratch) are checked before any allocation or callback call; a request that
// cannot be represented (arithmetic overflow, workspace too large) returns
// payload_overflow, and a workspace allocation failure also returns
// payload_overflow.
//
// The block layout matches the reshaped materializer output used by the
// attention path: out_f32 is [head_dim, n_heads, n_tokens] contiguous
// (slice-major within head_dim for multi-slice KVarN heads), in the source
// domain. See block_request below.

#include "ggml.h"

#include <cstddef>
#include <cstdint>

namespace llama_kv_qx {

enum class status {
    ok = 0,
    invalid_args,          // null callback / null payload with capacity / n_tokens == 0
    invalid_type,          // dst_type is not Q4_0/Q5_0/Q6_0/Q8_0 (whitelisted)
    invalid_geometry,      // head/n_heads/record_dim/row alignment mismatch
    invalid_domain,        // source or destination rotation metadata invalid
    unsupported_swa,       // SWA is out of scope for this first contract
    unsupported_multi_stream, // only single-stream conversion is supported
    payload_overflow,      // payload/workspace cannot be represented or allocated
    non_finite_input,      // the KVarN source produced a non-finite value
    callback_failure,      // read_block returned false or threw std::exception
    sink_failure,          // sink returned false or threw std::exception
};

const char * status_name(status st);

// KVarN source rotation domain. The materializer emits the rotated domain:
// every 128-dim slice of a head is transformed by the normalized WHT-128 and
// the corresponding lanes across slices are mixed by a second WHT step
// (matching ggml_cuda_kvarn_store / ggml_compute_forward_kvarn_materialize).
// The transform is self-inverse; the converter applies the canonical inverse
// per head when `rotated` is true. `record_dim` is the KVarN record width
// (64 for D64 V heads, 128 otherwise); `head_dim` must equal
// record_dim * head_slices with head_slices in {1, 2, 4}.
struct source_domain {
    uint32_t record_dim = 128;
    bool     rotated = true;
};

// Destination rotation: the remote standard Qx cache applies its own
// Hadamard per block of `wht_width` elements within each head, using the
// standard normalized WHT (the same transform family as the KVarN rot
// matrices). 0 keeps the original (de-rotated) domain. `wht_width` must be a
// power of two in {64, 128, 256, 512} and divide head_dim. The converter
// does not assume the source and destination domains are equal: the source
// rotation is always undone first, then the destination rotation is applied.
struct dest_domain {
    uint32_t wht_width = 0;
};

struct config {
    // One of the GGML_TYPE_Q4_0/Q5_0/Q6_0/Q8_0 enumerators. Stored as int32_t
    // so hostile values (negative, GGML_TYPE_COUNT+1, UINT_MAX) can flow into
    // the converter without enum-UB and be rejected by the explicit whitelist
    // before any ggml_type_traits lookup.
    int32_t       dst_type = GGML_TYPE_Q4_0;
    uint32_t      head_dim = 0;              // per-head dimension D (KVarN head width)
    uint32_t      n_heads  = 0;              // heads in the reshaped materializer output
    bool          value    = false;          // false = K, true = V
    source_domain src;
    dest_domain   dst;
    uint32_t      n_streams = 1;             // must be 1 (explicit rejection otherwise)
    bool          swa       = false;         // must be false (explicit rejection otherwise)
    uint32_t      max_tokens_per_block = 0;  // window size limit; 0 => 64
};

// One token window of the materialized KVarN source.
// out_f32 is contiguous with layout [head_dim, n_heads, n_tokens]:
//   out_f32[d + h*head_dim + t*head_dim*n_heads]
// for d in [0, head_dim), h in [0, n_heads), t in [0, n_tokens), in the
// source domain declared by config.src. For multi-slice heads the dim axis is
// slice-major (d = slice*record_dim + lane), matching the reshaped output of
// ggml_kvarn_materialize used by the attention path.
struct block_request {
    uint32_t t0;        // first token of the window (absolute prefix position)
    uint32_t n_tokens;  // window size (<= max_tokens_per_block)
    uint32_t head_dim;  // copy of config.head_dim
    uint32_t n_heads;   // copy of config.n_heads
    bool     value;     // copy of config.value
    float  * out_f32;   // head_dim * n_heads * n_tokens floats
};

using read_block_fn = bool (*)(void * user, const block_request & req);

// One quantized block delivered to the sink. `data` holds `n_tokens` logical
// token rows (row per token, head-major, the same row layout as convert's
// payload), `row_bytes` per row. `t0` is the GLOBAL first token of the block;
// `head_dim`/`n_heads`/`value` mirror the config. The sink owns
// staging/rollback/publication and may write only part of the window before
// returning false; no atomicity of any live cache is promised. The block is
// only valid during the call.
struct sink_block {
    uint32_t      t0;
    uint32_t      n_tokens;
    const uint8_t * data;
    size_t        row_bytes;
    uint32_t      head_dim;
    uint32_t      n_heads;
    bool          value;
};

using sink_fn = bool (*)(void * user, const sink_block & block);

struct result {
    status   st = status::ok;
    size_t   payload_bytes = 0;     // valid bytes written when st == ok
    uint32_t tokens_completed = 0;  // prefix converted when st == ok; on
                                    // failure, the staged payload is valid up
                                    // to this token (inclusive-exclusive)
};

// Quantized bytes per token row for a valid config (0 if the config is
// invalid). row = head_dim * n_heads floats.
size_t row_bytes(const config & cfg);

// Total quantized payload bytes for n_tokens (0 if invalid).
size_t payload_bytes(const config & cfg, uint32_t n_tokens);

// Converts `n_tokens` tokens of the KVarN source into `payload`
// (payload_capacity bytes, caller-owned STAGED buffer -- never the live
// cache). Processing is bounded: windows of at most max_tokens_per_block
// tokens (clamped to n_tokens) are read through `read_block`, de-rotated
// from the KVarN domain, rotated into the destination domain per the declared
// metadata, validated (finite) and only then quantized row by row. On any
// failure the function stops before publishing the failing window;
// tokens_completed reports the valid staged prefix. All
// geometry/type/overflow/domain checks run before the first window is read
// and before any allocation.
//
// On failure the staged buffer may contain a valid prefix up to
// tokens_completed. That is a property of the staged buffer only -- the
// caller decides whether to publish it -- and never an atomicity guarantee
// about any live cache. The live cache is never touched by this module.
// Streaming variant of convert: bounds BOTH the F32 window and the quantized
// output memory. Each window is fully validated (finite/domain) before its
// quantized block is handed to `sink`; on sink failure or std::exception the
// conversion stops with tokens_completed covering only confirmed earlier
// windows. A rejecting sink may already have written part of its window;
// the caller must discard or roll back that staged output. All
// geometry/type/overflow/domain checks run before any source or sink
// callback, and the full output is never allocated by this function.
// read_block receives the GLOBAL token offset exactly (block_request.t0).
// The final short window carries its true n_tokens and global t0.
result convert_to_sink(
        const config & cfg,
        uint32_t       n_tokens,
        read_block_fn  read_block,
        void         * read_user,
        sink_fn        sink,
        void         * sink_user);

// Bounded-window conversion into a caller-owned full-payload STAGED buffer
// (see convert_to_sink for the streaming contract). Implemented as an
// adapter over convert_to_sink with an internal writer sink, so both APIs
// share the same validation, transforms and quantization; outputs are
// byte-identical. The payload_capacity check runs before any callback.
result convert(
        const config & cfg,
        uint32_t       n_tokens,
        read_block_fn  read_block,
        void         * user,
        void         * payload,
        size_t         payload_capacity);

} // namespace llama_kv_qx
