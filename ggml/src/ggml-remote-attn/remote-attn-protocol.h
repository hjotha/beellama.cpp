#pragma once

#include <cstdint>

// RKVA v1 — remote KVarN attention wire protocol (host <-> Xbox Series X).
// Little-endian x64 on both ends, one persistent TCP connection per context,
// strict request/response ordering (the 16 full-attention layers are
// sequential, so exactly one call is in flight at a time).
//
// This header is duplicated verbatim on the Xbox side
// (xllama uwp/dflash-rpc/remote-attn-protocol.h). Keep both in sync and
// bump RKVA_VERSION on any wire change.

static constexpr uint32_t RKVA_MAGIC   = 0x41564B52; // "RKVA"
static constexpr uint16_t RKVA_VERSION = 1;

enum rkva_op : uint16_t {
    RKVA_HELLO           = 1,
    RKVA_CREATE_SESSION  = 2,
    RKVA_DESTROY_SESSION = 3,
    RKVA_ATTN_DECODE     = 4,  // n_tokens == 1 hot path
    RKVA_ATTN_PREFILL    = 5,  // n_tokens >= 1 batched
    RKVA_RESET           = 6,  // clear all KV of a session
    RKVA_TRIM            = 7,  // remove positions >= pos0
    RKVA_REWIND          = 8,  // alias of TRIM (speculative rollback)
    RKVA_PING            = 9,
    RKVA_GET_STATS       = 10, // response payload: UTF-8 JSON
    RKVA_PROBE           = 11, // kernel parity harness (diagnostics only)
};

// Wire element types for Q/K/V/OUT payloads.
enum rkva_wire : uint32_t {
    RKVA_WIRE_F16 = 0,
    RKVA_WIRE_F32 = 1,
};

// KVarN attention domain — values must match
// GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_* on the host.
enum rkva_domain : uint32_t {
    RKVA_DOMAIN_AUTO        = 0,
    RKVA_DOMAIN_ROTATED     = 1,
    RKVA_DOMAIN_ORIGINAL_V  = 2, // rotated K, original V
};

// rkva_request.session_id == RKVA_SESSION_NONE for connection-global ops.
static constexpr uint32_t RKVA_SESSION_NONE = 0xFFFFFFFFu;

#pragma pack(push, 1)

struct rkva_request {
    uint32_t magic;
    uint16_t version;
    uint16_t op;
    uint64_t cycle_id;
    uint32_t session_id;
    uint32_t layer_id;   // remote-relative: 0 .. n_layer_remote-1
    uint32_t n_tokens;
    uint32_t payload_bytes;
};

struct rkva_response {
    uint32_t magic;
    uint16_t version;
    uint16_t op;
    uint64_t cycle_id;
    int32_t  status;     // 0 = ok, negative = error (rkva_status)
    uint32_t payload_bytes;
    uint64_t xbox_receive_us;
    uint64_t xbox_prepare_us;
    uint64_t xbox_compute_us;
    uint64_t xbox_response_us;
};

enum rkva_status : int32_t {
    RKVA_OK                 =  0,
    RKVA_ERR_PROTOCOL       = -1,
    RKVA_ERR_UNSUPPORTED_OP = -2,
    RKVA_ERR_BAD_SESSION    = -3,
    RKVA_ERR_BAD_GEOMETRY   = -4,
    RKVA_ERR_MEMORY         = -5,
    RKVA_ERR_GPU            = -6,
    RKVA_ERR_BAD_POSITION   = -7,
    RKVA_ERR_BAD_PAYLOAD    = -8,
    RKVA_ERR_STATE          = -9,  // e.g. attention on a corrupted session
};

// HELLO request payload: the host proposes the model/cache geometry taken
// from the live llama_hparams/cparams and the --cache-type/--kv-tail flags.
struct rkva_hello {
    uint32_t proto_flags;      // bit0: host supports F16 wire, bit1: F32 wire
    uint32_t n_layer_remote;   // full-attention layers owned by the server
    uint32_t n_head;           // Q heads (24)
    uint32_t n_head_kv;        // KV heads (4)
    uint32_t head_dim;         // 256
    uint32_t max_ctx;          // per-session position capacity
    uint32_t cache_bits_k;     // 4 for kvarn4
    uint32_t cache_bits_v;     // 4 for kvarn4
    uint32_t group_tokens;     // KVarN record group (128)
    uint32_t sinkhorn_iters;
    uint32_t tail_tokens;      // --kv-tail-tokens equivalent (0 = policy default)
    uint32_t tail_groups;      // rollback insurance groups (2)
    uint32_t tail_type;        // 0 = f16, 1 = bf16
    uint32_t domain;           // rkva_domain
    uint32_t has_sinks;        // 0 (qwen35 passes no sinks)
    uint32_t swa;              // 0 = full attention (no sliding window)
    float    kq_scale;
    uint64_t reserved[2];
};

// HELLO response payload.
struct rkva_hello_ack {
    uint32_t supported_ops;    // bitmask of rkva_op (bit op = supported)
    uint32_t chosen_wire;      // rkva_wire
    uint32_t max_sessions;
    uint32_t server_version;   // informational (UWP package version)
    uint64_t mem_budget_bytes; // server-side KV budget (<= ~5 GiB game heap)
    uint64_t reserved[2];
};

struct rkva_session_req {
    uint32_t session_id;
    uint32_t seq_id;
    uint32_t capacity;         // positions; 0 = hello.max_ctx
    uint32_t flags;
};

// ATTN_DECODE / ATTN_PREFILL request payload:
//   rkva_attn_req
//   int32_t positions[n_tokens]        (absolute, per session)
//   Q [n_tokens][n_head][head_dim]     (chosen wire type)
//   K [n_tokens][n_head_kv][head_dim]
//   V [n_tokens][n_head_kv][head_dim]
// K/V are post-norm/post-RoPE, pre-Hadamard: the server owns the KVarN
// rotation (WHT), record compression, stage/tail and attention.
// Response payload:
//   OUT [n_tokens][n_head][head_dim]   (chosen wire type, post inverse-WHT
//                                       when the domain requires it)
struct rkva_attn_req {
    uint32_t wire;             // rkva_wire actually used in this call
    uint32_t flags;            // bit0: skip KV append (attention-only, probes)
    uint32_t n_query_pad;      // reserved, 0
    uint32_t reserved;
};

// TRIM / REWIND request payload: remove every stored position >= pos0.
struct rkva_trim_req {
    int32_t pos0;
};

// PROBE request payload: kernel-level parity harness.
//   RKVA_PROBE_STORE:   blob = K/V F32 rows + positions; response = packed
//                       record bytes for the completed group(s).
//   RKVA_PROBE_ATTENTION: blob = server-side session state reference +
//                       Q F32 rows; response = OUT F32 rows.
struct rkva_probe_req {
    uint32_t probe_id;         // RKVA_PROBE_*
    uint32_t mode;             // probe-specific
    uint32_t blob_bytes;
    uint32_t reserved;
};

enum rkva_probe : uint32_t {
    RKVA_PROBE_STORE     = 1,
    RKVA_PROBE_ATTENTION = 2,
    RKVA_PROBE_WHT       = 3,
    RKVA_PROBE_MATERIALIZE = 4,
};

#pragma pack(pop)

static_assert(sizeof(rkva_request)      == 32, "RKVA request wire size");
static_assert(sizeof(rkva_response)     == 56, "RKVA response wire size");
static_assert(sizeof(rkva_hello)        == 84, "RKVA hello wire size");
static_assert(sizeof(rkva_hello_ack)    == 40, "RKVA hello ack wire size");
static_assert(sizeof(rkva_session_req)  == 16, "RKVA session wire size");
static_assert(sizeof(rkva_attn_req)     == 16, "RKVA attn req wire size");
static_assert(sizeof(rkva_trim_req)     ==  4, "RKVA trim wire size");
static_assert(sizeof(rkva_probe_req)    == 16, "RKVA probe wire size");
