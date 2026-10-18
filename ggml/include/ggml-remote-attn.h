#pragma once

// ggml-remote-attn: backend that executes GGML_OP_REMOTE_ATTN on a remote
// KV cache + attention accelerator (Xbox Series X, RKVA protocol).
//
// The backend buffers are plain host memory, so the graph scheduler copies
// Q/K/V/pos out of CUDA and the attention output back with the same
// machinery used by the offload_kqv CPU pinning path. Exactly one RKVA call
// is in flight per node; the 16 full-attention layers are sequential.

#include "ggml-backend.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Extension slots of GGML_OP_REMOTE_ATTN; the legacy slots remain in ggml.h.
enum ggml_remote_attn_range_op_param {
    GGML_REMOTE_ATTN_PARAM_RANGE  = 6,
    GGML_REMOTE_ATTN_PARAM_ORIGIN = 7,
    GGML_REMOTE_ATTN_PARAM_N_LIVE = 8,
};

// Query-only v2 overflow attention. Sources 1/2 are the companion SET_ROWS
// results (Q4_0 or Q8_0 [D*HKV, capacity]); source3 contains global P+i IDs,
// with optional trailing -1 padding. Output rows pack OUT[D*H] then LSE[H]
// for each query. The op uploads only the changed/invalidated KV suffix.
GGML_API struct ggml_tensor * ggml_remote_attn_range(
        struct ggml_context * ctx,
        struct ggml_tensor * q,
        struct ggml_tensor * k_written,
        struct ggml_tensor * v_written,
        struct ggml_tensor * overflow_cells,
        struct ggml_tensor * query_pos,
        int layer_id, uint32_t origin, uint32_t n_live, float kq_scale);

// KVarN attention domain negotiated in HELLO. Values mirror rkva_domain and
// GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_*; AUTO lets the server reproduce the host
// KVarN rotated-domain plan for the configured cache type.
enum ggml_remote_attn_domain {
    GGML_REMOTE_ATTN_DOMAIN_AUTO       = 0,
    GGML_REMOTE_ATTN_DOMAIN_ROTATED    = 1,
    GGML_REMOTE_ATTN_DOMAIN_ORIGINAL_V = 2,
};

// Geometry negotiated in RKVA HELLO. Values come from the live hparams/cparams
// and the --cache-type/--kv-tail flags; the server must reproduce the host
// KVarN semantics for these parameters.
struct ggml_remote_attn_geometry {
    uint32_t n_layer_remote;  // full-attention layers owned by the server
    uint32_t n_head;          // Q heads
    uint32_t n_head_kv;       // KV heads
    uint32_t head_dim;
    uint32_t max_ctx;         // per-session position capacity
    uint32_t cache_bits_k;    // 4 = kvarn4
    uint32_t cache_bits_v;
    uint32_t group_tokens;    // KVarN record group (128)
    uint32_t sinkhorn_iters;
    uint32_t tail_tokens;     // exact precision tail (0 = server policy)
    uint32_t tail_groups;
    uint32_t tail_type;       // 0 = f16, 1 = bf16
    uint32_t domain;          // rkva_domain (matches GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_*)
    uint32_t has_sinks;
    uint32_t swa;             // 0 = full attention
    float    kq_scale;
};

// Creates the backend (does not connect). Returns NULL on failure.
GGML_API ggml_backend_t ggml_backend_remote_attn_init(const char * host, uint16_t port);

// Connect + HELLO handshake; fails fast with a log message. The geometry must
// be set before connecting.
GGML_API bool ggml_backend_remote_attn_set_geometry(
        ggml_backend_t backend, const struct ggml_remote_attn_geometry * geo);
GGML_API bool ggml_backend_remote_attn_connect(ggml_backend_t backend);
GGML_API bool ggml_backend_remote_attn_connected(ggml_backend_t backend);

// Session lifecycle. session_id is host-assigned and opaque to the server
// except for routing; seq_id is informational (llama sequence).
GGML_API bool ggml_backend_remote_attn_create_session(
        ggml_backend_t backend, uint32_t session_id, uint32_t seq_id, uint32_t capacity);
// Range-specific creation: native-quant servers defer KV allocation until
// CONFIG selects its storage. Old v2 F16 servers use their existing creation.
GGML_API bool ggml_backend_remote_attn_create_range_session(
        ggml_backend_t backend, uint32_t session_id, uint32_t seq_id, uint32_t capacity);

GGML_API bool ggml_backend_remote_attn_destroy_session(
        ggml_backend_t backend, uint32_t session_id);
GGML_API bool ggml_backend_remote_attn_reset(
        ggml_backend_t backend, uint32_t session_id);
// Remove every stored position >= pos0 (trim and speculative rewind).
GGML_API bool ggml_backend_remote_attn_trim(
        ggml_backend_t backend, uint32_t session_id, int32_t pos0);

// v2 compact overflow cache. Requires active session and the three RANGE ops.
// Matching Q4_0/Q8_0 storage is selected when advertised, otherwise the same
// quantized source expands to F16. An explicit native format must match source.
// Config resets its range histories; invalidation forces full per-layer resync
// from the authoritative host companion on the next range attention call.
GGML_API bool ggml_backend_remote_attn_configure_range(
        ggml_backend_t backend, uint32_t origin, uint32_t capacity, enum ggml_type cache_type);
GGML_API void ggml_backend_remote_attn_invalidate_range(ggml_backend_t backend);

// Explicit experimental native producer (GGML_RKVA_RANGE_STORAGE=kvarn4 and
// GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL=1). Append-only original projected F32
// sources; existing quantized overflow snapshots cannot initialize it.
GGML_API bool ggml_backend_remote_attn_native_kvarn_enabled(ggml_backend_t backend);
GGML_API bool ggml_backend_remote_attn_native_kvarn_has_state(ggml_backend_t backend);
GGML_API void ggml_remote_attn_range_set_native_sources(
        struct ggml_tensor * node, struct ggml_tensor * keys, struct ggml_tensor * values);

// Optional disk checkpoint accelerator. Opt in with GGML_RKVA_DISK_SNAPSHOTS=1;
// unsupported servers return false without sending any checkpoint request.
// Key is SHA256 of the fully verified authoritative HOST native file. The host
// CPU companion must always be restored first; a disk hit only avoids its LAN
// reupload. Missing/corrupt snapshots retain the normal full-upload fallback.
GGML_API bool ggml_backend_remote_attn_disk_snapshot_enabled(ggml_backend_t backend);
GGML_API bool ggml_backend_remote_attn_disk_snapshot(
        ggml_backend_t backend, const uint8_t host_state_sha256[32],
        uint32_t live_rows, bool restore);

// Selects the session used by subsequent graph_compute calls (the graph nodes
// carry only the layer id).
GGML_API void ggml_backend_remote_attn_set_active_session(
        ggml_backend_t backend, uint32_t session_id);

// True when the connection died or the server invalidated the session; the
// next compute aborts the graph instead of producing wrong results.
GGML_API bool ggml_backend_remote_attn_failed(ggml_backend_t backend);

// Profiling counters (--remote-attn-stats). The returned pointer is valid
// until the next call on the same backend; debug/verbose use only.
GGML_API const char * ggml_backend_remote_attn_stats_json(ggml_backend_t backend);
GGML_API void ggml_backend_remote_attn_reset_stats(ggml_backend_t backend);

// --- CPU-pinned execution path -------------------------------------------
// The remote attention op is computed on the CPU backend (pinned like the
// offload_kqv path) so the scheduler's proven CUDA<->CPU copies feed Q/K/V/pos
// and drain the output; the RPC runs inside the CPU compute. The backend object
// still owns the connection but is NOT registered with the scheduler.
//
// Registers the backend whose connection ggml_remote_attn_exec() should use
// (MVP: one active remote connection per process). Pass NULL to clear.
GGML_API void ggml_remote_attn_set_active(ggml_backend_t backend);

// Executes one GGML_OP_REMOTE_ATTN node over the active connection. Called from
// the CPU backend's compute_forward. The node's sources are host tensors (CPU
// buffer), read via ggml_backend_tensor_get. Returns false on RPC/GPU failure.
GGML_API bool ggml_remote_attn_exec(struct ggml_tensor * node);

#ifdef __cplusplus
}
#endif
