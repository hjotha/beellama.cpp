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
GGML_API bool ggml_backend_remote_attn_destroy_session(
        ggml_backend_t backend, uint32_t session_id);
GGML_API bool ggml_backend_remote_attn_reset(
        ggml_backend_t backend, uint32_t session_id);
// Remove every stored position >= pos0 (trim and speculative rewind).
GGML_API bool ggml_backend_remote_attn_trim(
        ggml_backend_t backend, uint32_t session_id, int32_t pos0);

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

#ifdef __cplusplus
}
#endif
