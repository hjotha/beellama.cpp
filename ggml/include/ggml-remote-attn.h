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
    // Opt-in native KVarN4 bulk range: the range node submits the prepared RPC
    // to the single ordered worker and returns without waiting for the reply.
    GGML_REMOTE_ATTN_PARAM_ASYNC  = 9,
    // Companion completion node: src[0] is the async range node; its output is
    // a copy of that node's OUT/LSE once the worker has completed.
    GGML_REMOTE_ATTN_PARAM_WAIT   = 10,
    // Unique monotonic ticket and epoch for safe completion binding & ABA protection
    GGML_REMOTE_ATTN_PARAM_TICKET_LO = 11,
    GGML_REMOTE_ATTN_PARAM_TICKET_HI = 12,
    GGML_REMOTE_ATTN_PARAM_EPOCH_LO  = 13,
    GGML_REMOTE_ATTN_PARAM_EPOCH_HI  = 14,
    // Optional cohort: count in bits0..7, index in bits8..15. Bit16 is
    // host-local WF CUDA ownership metadata; it is never encoded on the wire.
    GGML_REMOTE_ATTN_PARAM_COHORT    = 15,
};

enum ggml_remote_attn_range_local_flag {
    GGML_REMOTE_ATTN_COHORT_WIRE_MASK = 0xffffu,
    GGML_REMOTE_ATTN_WAVEFRONT_SHARED_CUDA = 1u << 16,
};

// Query-only v2 overflow attention. Sources 1/2 are the companion SET_ROWS
// results (Q4_0, Q8_0 or F16 [D*HKV, capacity]); source3 contains global P+i IDs,
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

// Opt-in overlap plumbing. Marking a range node async is only valid when
// ggml_backend_remote_attn_async_enabled() is true (native KVarN4 bulk RPC).
// The completion node must be scheduled after the local attention work and
// before anything that reads the remote OUT/LSE. Both nodes execute on the
// CPU backend through ggml_remote_attn_exec().
GGML_API void ggml_remote_attn_range_set_async(struct ggml_tensor * node);
// Distinct-layer ranges in one wave, prepared consecutively before any WAIT.
// First member requires an empty queue. Tickets bind each runtime invocation.
// Cohort4 transport is unavailable in this integration; enabled() returns false.
GGML_API void ggml_remote_attn_range_set_cohort(struct ggml_tensor * node, uint32_t count, uint32_t index);
// Host-local opt-in for ranges built by the causal WF planner. Sharing does
// not change wire batching or FP8 arithmetic. Marking/cohort setter order is
// immaterial. Untagged nodes retain private context ownership.
GGML_API void ggml_remote_attn_range_set_wavefront(struct ggml_tensor * node);
GGML_API bool ggml_backend_remote_attn_cohort_enabled(ggml_backend_t backend);
GGML_API struct ggml_tensor * ggml_remote_attn_range_wait(
        struct ggml_context * ctx, struct ggml_tensor * range);

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
// Authorize a prefix-preserving suffix trim after host memory removal or a
// validated resident partial restore. Standard ranges send at most one global
// REWIND and retain acknowledged prefix coverage; arbitrary rewrites must use
// invalidate_range instead. A false result makes the remote owner unusable.
GGML_API bool ggml_backend_remote_attn_trim(
        ggml_backend_t backend, uint32_t session_id, int32_t pos0);

// v2 compact overflow cache. Requires active session and the three RANGE ops.
// Matching Q4_0/Q8_0 storage is selected when advertised, otherwise the same
// quantized source expands to F16. F16 companions upload their original half
// bytes directly. An explicit native quantized format must match its source.
// Config resets its range histories; invalidation forces full per-layer resync
// from the authoritative host companion on the next range attention call.
GGML_API bool ggml_backend_remote_attn_configure_range(
        ggml_backend_t backend, uint32_t origin, uint32_t capacity, enum ggml_type cache_type);
GGML_API void ggml_backend_remote_attn_invalidate_range(ggml_backend_t backend);

// Explicit experimental native producer (GGML_RKVA_RANGE_STORAGE=kvarn4 and
// GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL=1). Original projected F32 sources;
// negotiated op21 supports <=128-row resident rollback. FULL overflow snapshots
// cannot initialize it. Capability does not imply a native persistence codec.
GGML_API bool ggml_backend_remote_attn_native_kvarn_enabled(ggml_backend_t backend);
// True when the opt-in GGML_RKVA_ASYNC_LOCAL_REMOTE=1 overlap was accepted by a
// native KVarN4 bulk-capable server. Always false for synchronous/local modes.
GGML_API bool ggml_backend_remote_attn_async_enabled(ggml_backend_t backend);
// Native FP8 serving can retain CUDA operands and output in device memory.
// Queried by the graph builder/CUDA backend for the active remote connection.
GGML_API bool ggml_backend_remote_attn_cuda_enabled(void);
// Shape-specific policy for a complete range (or its WAIT), before segmentation.
// GGML_RKVA_CUDA_FP8=prefill enables CUDA only for the existing nq>9 bucket;
// 0 disables it, while 1/unset preserve the full CUDA capability behavior.
// mapped explicitly selects v3 mapped-host kernels for 1..9 queries and the
// ordinary helper for larger ranges; unsupported small operations fail closed.
// Keep this process configuration fixed while graphs/requests are alive.
GGML_API bool ggml_backend_remote_attn_cuda_enabled_for_node(const struct ggml_tensor * node);
// CUDA dispatcher supplies a recorded producer fence, retained until return.
GGML_API bool ggml_remote_attn_exec_cuda(struct ggml_tensor * node, void * producer_event);
// Terminal graph boundary: wait and discard an unconsumed completion before
// graph tensors are recycled. Ordinary backend synchronize preserves completion.
GGML_API void ggml_backend_remote_attn_drain(ggml_backend_t backend);
// Async queue diagnostics: in-flight count and max queue capacity (2 slots).
GGML_API size_t ggml_backend_remote_attn_async_in_flight(ggml_backend_t backend);
GGML_API size_t ggml_backend_remote_attn_async_capacity(ggml_backend_t backend);
GGML_API bool ggml_backend_remote_attn_native_kvarn_can_rewind(ggml_backend_t backend);
GGML_API bool ggml_backend_remote_attn_native_kvarn_can_restore(ggml_backend_t backend, int32_t pos0);
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

// --- Host-orchestrated execution -----------------------------------------
// CPU execution uses scheduler copies. Negotiated native FP8 can dispatch from
// CUDA, retaining device Q/OUT and host cache metadata in their owning buffers.
// The connection object itself is not registered with the scheduler.
//
// Registers the backend whose connection ggml_remote_attn_exec() should use
// (MVP: one active remote connection per process). Pass NULL to clear.
GGML_API void ggml_remote_attn_set_active(ggml_backend_t backend);

// Executes one GGML_OP_REMOTE_ATTN node over the active connection. Called from
// the CPU backend's compute_forward. The node's sources are host tensors (CPU
// buffer), read via ggml_backend_tensor_get. Returns false on RPC/GPU failure.
GGML_API bool ggml_remote_attn_exec(struct ggml_tensor * node);

// Test-only deterministic handshake for the post-transaction validation
// window. Enable, wait until the worker reaches the barrier, inject the fault,
// then release. The worker's internal wait is bounded (10 s); wait_reached()
// uses timeout_ms for the test-side observation. No effect unless enabled.
GGML_API void ggml_backend_remote_attn_test_handshake_enable(bool enable);
GGML_API bool ggml_backend_remote_attn_test_handshake_wait_reached(int timeout_ms);
GGML_API void ggml_backend_remote_attn_test_handshake_release();

#ifdef __cplusplus
}
#endif
