#pragma once

#include <cstdint>

// RKVA v1 — remote KVarN attention wire protocol (host <-> Xbox Series X).
// Little-endian x64 on both ends, one persistent TCP connection per context,
// strict request/response ordering (the 16 full-attention layers are
// sequential, so exactly one call is in flight at a time).
//
// Shared by the host backend and tools/xbox/uwp/rkva.
// Bump RKVA_VERSION on any wire change.

static constexpr uint32_t RKVA_MAGIC = 0x41564B52; // "RKVA"
static constexpr uint16_t RKVA_VERSION = 3;
// Version 2 remains the default and the authenticated-snapshot contract.
static constexpr uint16_t RKVA_VERSION_V2 = 2;
static constexpr uint16_t RKVA_VERSION_LEGACY = 1;

static constexpr bool rkva_range_version(uint16_t version) {
    return version == RKVA_VERSION_V2 || version == RKVA_VERSION;
}

enum rkva_op : uint16_t {
    RKVA_HELLO = 1,
    RKVA_CREATE_SESSION = 2,
    RKVA_DESTROY_SESSION = 3,
    RKVA_ATTN_DECODE = 4,  // n_tokens == 1 hot path
    RKVA_ATTN_PREFILL = 5, // n_tokens >= 1 batched
    RKVA_RESET = 6,        // clear all KV of a session
    RKVA_TRIM = 7,         // remove positions >= pos0
    RKVA_REWIND = 8,       // alias of TRIM (speculative rollback)
    RKVA_PING = 9,
    RKVA_GET_STATS = 10,             // response payload: UTF-8 JSON
    RKVA_PROBE = 11,                 // kernel parity harness (diagnostics only)
    RKVA_RANGE_CONFIG = 12,          // session-global origin/capacity, clears cached rows
    RKVA_RANGE_UPLOAD = 13,          // per-layer append/replace rows, K then V
    RKVA_KVARN_RANGE_CONFIG = 15,    // descriptor-qualified native KVarN4
    RKVA_KVARN_RANGE_UPLOAD = 16,    // native record delta + complete stage/tail snapshot
    RKVA_SNAPSHOT_SAVE = 17,         // atomic all-layer local-disk checkpoint
    RKVA_SNAPSHOT_RESTORE = 18,      // validated all-layer local-disk restore
    RKVA_RANGE_UPLOAD_ATTN = 19,     // last upload chunk plus attention in one round trip
    RKVA_RANGE_PROFILE = 20,         // optional profiling/control, UTF-8 JSON response
    RKVA_KVARN_RANGE_REWIND = 21,    // atomic bounded resident native rewind
    RKVA_KVARN_RANGE_BULK_ATTN = 22, // ordered native upload/query segments, one reply
    RKVA_KVARN_RANGE_COHORT_ATTN = 23, // negotiated independent-layer envelopes
    // Reserved managed-snapshot extension. Nothing in this tree advertises,
    // returns or dispatches these numbers from a live listener: the Phase 4
    // codec is compiled only behind RKVA_MANAGED_SNAPSHOT_PHASE4 and gated at
    // run time by RKVA_MANAGED_SNAPSHOT_PHASE4=1. The only negotiated managed
    // framing is the v3 single-frame contract (see docs/rkva-managed-snapshot-
    // phase4.md); the older E2E v2 offer/reply shape is offline-only.
    RKVA_REGISTRY_PREPARE = 24,
    RKVA_REGISTRY_COMMIT = 25,
    RKVA_REGISTRY_ABORT = 26,
    RKVA_REGISTRY_RELEASE = 27,
    RKVA_REGISTRY_STATUS = 28,
    RKVA_MANAGED_AUTH_HELLO = 29,
    RKVA_MANAGED_AUTH_CHALLENGE = 30,
    RKVA_MANAGED_ATTEST = 31,
    RKVA_NATIVE_SNAPSHOT_STORE = 32, // negotiated quota status / keyed retirement
    RKVA_RANGE_ATTN = 14,            // query-only F32, returns OUT F32 then LSE F32
};

// Wire element types for Q/K/V/OUT payloads.
enum rkva_wire : uint32_t {
    RKVA_WIRE_F16 = 0,
    RKVA_WIRE_F32 = 1,
    RKVA_WIRE_Q4_0 = 2, // RANGE_UPLOAD only: ggml block_q4_0, 18 bytes / 32 values
    RKVA_WIRE_Q8_0 = 3, // RANGE_UPLOAD only: ggml block_q8_0, 34 bytes / 32 values
    RKVA_WIRE_FP8 = 4, // Option A: 1-byte FP8 E5M2 for queries and outputs
    RKVA_WIRE_SPLITZIP = 5, // Option B: lossless exponent-compressed FP16
    RKVA_WIRE_OSCAR_INT2 = 6, // RANGE_UPLOAD only: OSCAR block_oscar_int2, 12 bytes / 32 values
    RKVA_WIRE_FP8_E4M3FN = 7, // v3 op22: layout-specific negotiated E4M3FN Q and OUT
};

// v2 range extensions. Native storage capabilities use ack.reserved[0].
static constexpr uint64_t RKVA_CAP_RANGE_Q4_0 = 1ull << 0;
static constexpr uint64_t RKVA_CAP_RANGE_Q8_0 = 1ull << 1;
static constexpr uint64_t RKVA_CAP_RANGE_KVARN4 = 1ull << 2;
static constexpr uint64_t RKVA_CAP_DISK_SNAPSHOT = 1ull << 3;
static constexpr uint64_t RKVA_CAP_RANGE_UPLOAD_ATTN = 1ull << 4;
static constexpr uint64_t RKVA_CAP_RANGE_PROFILE = 1ull << 5;
static constexpr uint64_t RKVA_CAP_KVARN_REWIND = 1ull << 6;
static constexpr uint64_t RKVA_CAP_KVARN_SERIAL_PREFIX = 1ull << 7;
static constexpr uint64_t RKVA_CAP_KVARN_BULK_ATTN = 1ull << 8;
static constexpr uint64_t RKVA_CAP_KVARN_BULK_F16 = 1ull << 9;
static constexpr uint64_t RKVA_CAP_KVARN_BULK_FP8 = 1ull << 10;
static constexpr uint64_t RKVA_CAP_KVARN_BULK_SPLITZIP = 1ull << 11;
// Codec revision: Q rounds directly from F32; finite Q/O saturate, ties to even.
static constexpr uint64_t RKVA_CAP_KVARN_BULK_FP8_RNE = 1ull << 12;
// Upload flag OMIT_STAGE is supported for serial-prefix overflow with an exact 128-row tail.
static constexpr uint64_t RKVA_CAP_KVARN_COMPACT_STAGE = 1ull << 13;
// layout 4: native Q4_0 overflow rows over op16/op22 (bulk + async capable server)
static constexpr uint64_t RKVA_CAP_NATIVE_Q4_BULK = 1ull << 14;
// layout 5: native Q8_0 overflow rows over op16/op22 (bulk + async capable server)
static constexpr uint64_t RKVA_CAP_NATIVE_Q8_BULK = 1ull << 15;
// layout 6: native SAW-INT4 / BDR-INT4 overflow rows over op16/op22 (136 bytes per head row)
static constexpr uint64_t RKVA_CAP_NATIVE_SAW_INT4_BULK = 1ull << 23;
// layout 7: native OSCAR INT2 overflow rows over op16/op22 (96 bytes per head row)
static constexpr uint64_t RKVA_CAP_NATIVE_OSCAR_INT2_BULK = 1ull << 22;
// layout 8: TurboQuant NC, two 132-byte-per-head planes (K has two zero pad bytes)
static constexpr uint64_t RKVA_CAP_NATIVE_TURBOQUANT_NC_BULK = 1ull << 24;
static constexpr uint64_t RKVA_CAP_KVARN_BULK_E4M3FN = 1ull << 25;
static constexpr uint64_t RKVA_CAP_KVARN_FLEX_TAIL = 1ull << 26;
static constexpr uint64_t RKVA_CAP_NATIVE_SNAPSHOT_STORE = 1ull << 27;
// E4 is not implied by the E5 native-row caps or the KVarN-only bit 25.
// An old server must never receive E4 for layouts it only decodes as E5.
static constexpr uint64_t RKVA_CAP_NATIVE_Q4_BULK_E4M3FN = 1ull << 28;
static constexpr uint64_t RKVA_CAP_NATIVE_SAW_INT4_BULK_E4M3FN = 1ull << 29;
static constexpr uint64_t RKVA_CAP_NATIVE_TURBOQUANT_NC_BULK_E4M3FN = 1ull << 30;
// Experimental native standard 4-bit rows; never reinterpret as layout 4.
// Additive descriptors carried by existing op15/op16/op22 framing (no ABI change).
static constexpr uint64_t RKVA_CAP_NATIVE_Q4_1_BULK = 1ull << 31;
static constexpr uint64_t RKVA_CAP_NATIVE_IQ4_NL_BULK = 1ull << 32;
static constexpr uint64_t RKVA_CAP_NATIVE_Q4_1_BULK_E4M3FN = 1ull << 33;
static constexpr uint64_t RKVA_CAP_NATIVE_IQ4_NL_BULK_E4M3FN = 1ull << 34;
static constexpr uint64_t RKVA_CAP_NATIVE_Q5_0_BULK = 1ull << 35;
static constexpr uint64_t RKVA_CAP_NATIVE_Q5_1_BULK = 1ull << 36;
static constexpr uint64_t RKVA_CAP_NATIVE_Q6_0_BULK = 1ull << 37;
static constexpr uint64_t RKVA_CAP_NATIVE_Q6_1_BULK = 1ull << 38;
static constexpr uint64_t RKVA_CAP_NATIVE_Q3_0_BULK = 1ull << 39;
static constexpr uint64_t RKVA_CAP_NATIVE_Q3_1_BULK = 1ull << 40;
static constexpr uint64_t RKVA_CAP_NATIVE_Q2_1_BULK = 1ull << 41;
static constexpr uint64_t RKVA_CAP_NATIVE_BF16_BULK = 1ull << 42;
static constexpr uint64_t RKVA_CAP_NATIVE_F32_BULK  = 1ull << 43;
static constexpr uint64_t RKVA_CAP_NATIVE_Q5_0_BULK_E4M3FN = 1ull << 44;
static constexpr uint64_t RKVA_CAP_NATIVE_Q5_1_BULK_E4M3FN = 1ull << 45;
static constexpr uint64_t RKVA_CAP_NATIVE_Q6_0_BULK_E4M3FN = 1ull << 46;
static constexpr uint64_t RKVA_CAP_NATIVE_Q6_1_BULK_E4M3FN = 1ull << 47;
static constexpr uint64_t RKVA_CAP_NATIVE_Q3_0_BULK_E4M3FN = 1ull << 48;
static constexpr uint64_t RKVA_CAP_NATIVE_Q3_1_BULK_E4M3FN = 1ull << 49;
static constexpr uint64_t RKVA_CAP_NATIVE_OSCAR_INT2_BULK_E4M3FN = 1ull << 50;
static constexpr uint64_t RKVA_CAP_NATIVE_Q2_1_BULK_E4M3FN = 1ull << 51;
static constexpr uint64_t RKVA_CAP_NATIVE_BF16_BULK_E4M3FN = 1ull << 52;
static constexpr uint64_t RKVA_CAP_NATIVE_F32_BULK_E4M3FN = 1ull << 53;
// Capability allocation in HELLO ack.reserved[0]:
//   0..13  range/KVarN storage and bulk features
//   14     RKVA_CAP_NATIVE_Q4_BULK (descriptor layout 4)
//   15     RKVA_CAP_NATIVE_Q8_BULK (descriptor layout 5)
//   16..27 existing cohort, snapshot and native-storage extensions
//   28..30 native Q4_0/SAW/NC E4M3FN
//   31..34 experimental Q4_1/IQ4_NL bulk and their separate E4M3FN capabilities
//   35+    reserved; do not reuse allocated bits for a different descriptor.
// Reserved experimental identity only. This client refuses RPC_BATCH=4 until
// a matching server implementation has been qualified; bits14/15 stay native.
static constexpr uint64_t RKVA_CAP_KVARN_COHORT_ATTN = 1ull << 16;
// Native KVarN4 bulk supports up to 512 queries / 64 serial-prefix segments.
static constexpr uint64_t RKVA_CAP_KVARN_BULK_512 = 1ull << 17;
// Descriptor revision-3 FULL native overflow snapshots (ops 17/18 revision 2).
// Additive: old servers reject revision 2 before touching state, and old hosts
// never send it because the HELLO bit/capability are both absent.
static constexpr uint64_t RKVA_CAP_NATIVE_KVARN_SNAPSHOT = 1ull << 18;
// Reserved managed-snapshot capabilities. These are never set by the default
// listener (the Phase 4 activation gate is off) and a peer that requires them
// must fail closed instead of falling back to an unauthenticated path.
static constexpr uint64_t RKVA_CAP_MANAGED_REGISTRY_V1 = 1ull << 19;
static constexpr uint64_t RKVA_CAP_MANAGED_AUTH_V2 = 1ull << 20;
static constexpr uint64_t RKVA_CAP_MANAGED_FRAMING_V3 = 1ull << 21;

// v1 cohort extension; each entry immediately precedes its old op22 body.
// Reply mirrors the envelope with OUT/LSE bodies and exact ticket/epoch echoes.
struct rkva_kvarn_cohort_header {
    uint32_t version;
    uint32_t entry_count;
    uint32_t total_queries;
    uint32_t reserved;
};
struct rkva_kvarn_cohort_entry {
    uint32_t layer;
    uint32_t n_queries;
    uint32_t body_bytes;
    uint32_t reserved;
    uint64_t ticket;
    uint64_t epoch;
};
static_assert(sizeof(rkva_kvarn_cohort_header) == 16, "cohort header ABI");
static_assert(sizeof(rkva_kvarn_cohort_entry) == 32, "cohort entry ABI");
static constexpr uint32_t RKVA_KVARN_UPLOAD_RESET = 1u;
static constexpr uint32_t RKVA_KVARN_UPLOAD_OMIT_STAGE = 2u;
static constexpr uint32_t RKVA_PROTO_NATIVE_REWIND_V2 = 1u << 16;
// Version 3 only; separate from the legacy HELLO chosen_wire field.
static constexpr uint32_t RKVA_PROTO_BULK_E4M3FN = 1u << RKVA_WIRE_FP8_E4M3FN;
static constexpr uint32_t RKVA_PROTO_NATIVE_SERIAL_PREFIX_V3 = 1u << 17;
static constexpr uint32_t RKVA_PROTO_NATIVE_SNAPSHOT_V4 = 1u << 18;
static constexpr uint32_t RKVA_PROTO_KVARN_FLEX_TAIL = 1u << 26;
static constexpr uint32_t RKVA_PROTO_NATIVE_SNAPSHOT_STORE = 1u << 27;
// Reserved managed-snapshot protocol flags. Advertised only by a Phase 4 build
// that has passed the activation gate; inert otherwise.
static constexpr uint32_t RKVA_PROTO_MANAGED_REGISTRY_V1 = 1u << 19;
static constexpr uint32_t RKVA_PROTO_MANAGED_AUTH_V2 = 1u << 20;
static constexpr uint32_t RKVA_PROTO_MANAGED_FRAMING_V3 = 1u << 21;
// The E4 experiment is tied to one v3 HELLO; it never aliases an E5 payload.
static constexpr bool rkva_e4_wire_allowed(uint16_t request_version, uint16_t hello_version,
                                            bool offered, bool available, uint32_t wire) {
    return wire == RKVA_WIRE_FP8_E4M3FN ?
        request_version == RKVA_VERSION && hello_version == request_version && offered && available :
        !(hello_version == RKVA_VERSION && offered);
}
static constexpr uint32_t RKVA_SNAPSHOT_REVISION_NATIVE = 2;
static constexpr uint32_t RKVA_NATIVE_SNAPSHOT_FORMAT = 1;
static constexpr uint32_t RKVA_PROFILE_ENABLE = 1u << 0;
static constexpr uint32_t RKVA_PROFILE_RESET = 1u << 1;
static constexpr uint32_t RKVA_PROFILE_TRACE = 1u << 2;
static constexpr uint32_t RKVA_PROFILE_REPORT_ONLY = 1u << 3;
static constexpr uint32_t RKVA_RANGE_STORAGE_KVARN4 = 1u << 2;
static constexpr uint32_t RKVA_RANGE_STORAGE_Q4_0 = 1u << 0; // CONFIG.reserved[0]
static constexpr uint32_t RKVA_RANGE_STORAGE_Q8_0 = 1u << 1; // CONFIG.reserved[0]
static constexpr uint32_t RKVA_RANGE_STORAGE_SAW_INT4 = 1u << 3;
static constexpr uint32_t RKVA_RANGE_STORAGE_OSCAR_INT2 = 1u << 4;
static constexpr uint32_t RKVA_RANGE_STORAGE_TURBOQUANT_NC = 1u << 5;
static constexpr uint32_t RKVA_RANGE_STORAGE_Q4_1 = 1u << 6;
static constexpr uint32_t RKVA_RANGE_STORAGE_IQ4_NL = 1u << 7;
static constexpr uint32_t RKVA_RANGE_STORAGE_Q5_0 = 1u << 8;
static constexpr uint32_t RKVA_RANGE_STORAGE_Q5_1 = 1u << 9;
static constexpr uint32_t RKVA_RANGE_STORAGE_Q6_0 = 1u << 10;
static constexpr uint32_t RKVA_RANGE_STORAGE_Q6_1 = 1u << 11;
static constexpr uint32_t RKVA_RANGE_STORAGE_Q3_0 = 1u << 12;
static constexpr uint32_t RKVA_RANGE_STORAGE_Q3_1 = 1u << 13;
static constexpr uint32_t RKVA_RANGE_STORAGE_Q2_1 = 1u << 14;
static constexpr uint32_t RKVA_RANGE_STORAGE_BF16 = 1u << 15;
static constexpr uint32_t RKVA_RANGE_STORAGE_F32  = 1u << 16;
static constexpr uint32_t RKVA_LAYOUT_Q4_1 = 9;
static constexpr uint32_t RKVA_LAYOUT_IQ4_NL = 10;
static constexpr uint32_t RKVA_LAYOUT_Q5_0 = 11;
static constexpr uint32_t RKVA_LAYOUT_Q5_1 = 12;
static constexpr uint32_t RKVA_LAYOUT_Q6_0 = 13;
static constexpr uint32_t RKVA_LAYOUT_Q6_1 = 14;
static constexpr uint32_t RKVA_LAYOUT_Q3_0 = 15;
static constexpr uint32_t RKVA_LAYOUT_Q3_1 = 16;
static constexpr uint32_t RKVA_LAYOUT_Q2_1 = 17;
static constexpr uint32_t RKVA_LAYOUT_BF16 = 18;
static constexpr uint32_t RKVA_LAYOUT_F32  = 19;
// Exact descriptor identity, not a bitwise storage test. Unknown layouts fail closed.
static constexpr uint64_t rkva_e4_layout_capability(uint32_t storage, uint32_t layout) {
    return storage == RKVA_RANGE_STORAGE_KVARN4 && layout == 3 ? RKVA_CAP_KVARN_BULK_E4M3FN :
           storage == RKVA_RANGE_STORAGE_Q4_0 && layout == 4 ? RKVA_CAP_NATIVE_Q4_BULK_E4M3FN :
           storage == RKVA_RANGE_STORAGE_SAW_INT4 && layout == 6 ? RKVA_CAP_NATIVE_SAW_INT4_BULK_E4M3FN :
           storage == RKVA_RANGE_STORAGE_OSCAR_INT2 && layout == 7 ? RKVA_CAP_NATIVE_OSCAR_INT2_BULK_E4M3FN :
           storage == RKVA_RANGE_STORAGE_TURBOQUANT_NC && layout == 8 ? RKVA_CAP_NATIVE_TURBOQUANT_NC_BULK_E4M3FN :
           storage == RKVA_RANGE_STORAGE_Q4_1 && layout == RKVA_LAYOUT_Q4_1 ? RKVA_CAP_NATIVE_Q4_1_BULK_E4M3FN :
           storage == RKVA_RANGE_STORAGE_IQ4_NL && layout == RKVA_LAYOUT_IQ4_NL ? RKVA_CAP_NATIVE_IQ4_NL_BULK_E4M3FN :
           storage == RKVA_RANGE_STORAGE_Q5_0 && layout == RKVA_LAYOUT_Q5_0 ? RKVA_CAP_NATIVE_Q5_0_BULK_E4M3FN :
           storage == RKVA_RANGE_STORAGE_Q5_1 && layout == RKVA_LAYOUT_Q5_1 ? RKVA_CAP_NATIVE_Q5_1_BULK_E4M3FN :
           storage == RKVA_RANGE_STORAGE_Q6_0 && layout == RKVA_LAYOUT_Q6_0 ? RKVA_CAP_NATIVE_Q6_0_BULK_E4M3FN :
           storage == RKVA_RANGE_STORAGE_Q6_1 && layout == RKVA_LAYOUT_Q6_1 ? RKVA_CAP_NATIVE_Q6_1_BULK_E4M3FN :
           storage == RKVA_RANGE_STORAGE_Q3_0 && layout == RKVA_LAYOUT_Q3_0 ? RKVA_CAP_NATIVE_Q3_0_BULK_E4M3FN :
           storage == RKVA_RANGE_STORAGE_Q3_1 && layout == RKVA_LAYOUT_Q3_1 ? RKVA_CAP_NATIVE_Q3_1_BULK_E4M3FN :
           storage == RKVA_RANGE_STORAGE_Q2_1 && layout == RKVA_LAYOUT_Q2_1 ? RKVA_CAP_NATIVE_Q2_1_BULK_E4M3FN :
           storage == RKVA_RANGE_STORAGE_BF16 && layout == RKVA_LAYOUT_BF16 ? RKVA_CAP_NATIVE_BF16_BULK_E4M3FN :
           storage == RKVA_RANGE_STORAGE_F32  && layout == RKVA_LAYOUT_F32  ? RKVA_CAP_NATIVE_F32_BULK_E4M3FN : 0;
}
static constexpr bool rkva_e4_layout_allowed(uint64_t caps, uint32_t storage, uint32_t layout) {
    const uint64_t required = rkva_e4_layout_capability(storage, layout);
    return required != 0 && (caps & required) == required;
}
static constexpr uint32_t RKVA_SESSION_RANGE =
    1u << 0; // CREATE_SESSION.flags; defer allocation until CONFIG

// KVarN attention domain — values must match
// GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_* on the host.
enum rkva_domain : uint32_t {
    RKVA_DOMAIN_AUTO = 0,
    RKVA_DOMAIN_ROTATED = 1,
    RKVA_DOMAIN_ORIGINAL_V = 2, // rotated K, original V
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
    uint32_t layer_id; // remote-relative: 0 .. n_layer_remote-1
    uint32_t n_tokens;
    uint32_t payload_bytes;
};

struct rkva_response {
    uint32_t magic;
    uint16_t version;
    uint16_t op;
    uint64_t cycle_id;
    int32_t status; // 0 = ok, negative = error (rkva_status)
    uint32_t payload_bytes;
    uint64_t xbox_receive_us;
    uint64_t xbox_prepare_us;
    uint64_t xbox_compute_us;
    uint64_t xbox_response_us;
};

enum rkva_status : int32_t {
    RKVA_OK = 0,
    RKVA_ERR_PROTOCOL = -1,
    RKVA_ERR_UNSUPPORTED_OP = -2,
    RKVA_ERR_BAD_SESSION = -3,
    RKVA_ERR_BAD_GEOMETRY = -4,
    RKVA_ERR_MEMORY = -5,
    RKVA_ERR_GPU = -6,
    RKVA_ERR_BAD_POSITION = -7,
    RKVA_ERR_BAD_PAYLOAD = -8,
    RKVA_ERR_SNAPSHOT_MISS = -10, // absent/incompatible/corrupt; host reuploads
    RKVA_ERR_SNAPSHOT_IO = -11,   // disk/quota error; live state unchanged by SAVE
    RKVA_ERR_STATE = -9,          // e.g. attention on a corrupted session
};

// HELLO request payload: the host proposes the model/cache geometry taken
// from the live llama_hparams/cparams and the --cache-type/--kv-tail flags.
struct rkva_hello {
    uint32_t
        proto_flags; // bit0: F16, bit1: F32, bit16: native rewind v2, bit17: serial query prefix v3
    uint32_t n_layer_remote; // full-attention layers owned by the server
    uint32_t n_head;         // Q heads (24)
    uint32_t n_head_kv;      // KV heads (4)
    uint32_t head_dim;       // 256
    uint32_t max_ctx;        // per-session position capacity
    uint32_t cache_bits_k;   // 4 for kvarn4
    uint32_t cache_bits_v;   // 4 for kvarn4
    uint32_t group_tokens;   // KVarN record group (128)
    uint32_t sinkhorn_iters;
    uint32_t tail_tokens; // --kv-tail-tokens equivalent (0 = policy default)
    uint32_t tail_groups; // rollback insurance groups (2)
    uint32_t tail_type;   // 0 = f16, 1 = bf16
    uint32_t domain;      // rkva_domain
    uint32_t has_sinks;   // 0 (qwen35 passes no sinks)
    uint32_t swa;         // 0 = full attention (no sliding window)
    float kq_scale;
    uint64_t reserved[2];
};

// HELLO response payload.
struct rkva_hello_ack {
    uint32_t supported_ops; // bitmask of rkva_op (bit op = supported)
    uint32_t chosen_wire;   // rkva_wire
    uint32_t max_sessions;
    uint32_t server_version;   // informational (UWP package version)
    uint64_t mem_budget_bytes; // server-side KV budget (<= ~5 GiB game heap)
    uint64_t reserved[2];
};

struct rkva_session_req {
    uint32_t session_id;
    uint32_t seq_id;
    uint32_t capacity; // positions; 0 = hello.max_ctx
    uint32_t flags;    // RKVA_SESSION_RANGE when ACK advertises native Q4 or Q8 ranges
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
    uint32_t wire;        // rkva_wire actually used in this call
    uint32_t flags;       // bit0: skip KV append (attention-only, probes)
    uint32_t n_query_pad; // reserved, 0
    uint32_t reserved;
};

// v2 range mode: absolute query positions and compact physical KV rows.
// HELLO uses cache_bits_k/v=0, domain=AUTO, max_ctx=capacity. CONFIG follows
// CREATE_SESSION. Reissuing CONFIG clears ALL layer histories in that session.
// CONFIG reserved[0]: 0=F16, STORAGE_Q4_0 or STORAGE_Q8_0 (mutually exclusive).
// CONFIG reserved[1] stays 0.
struct rkva_range_config {
    uint32_t origin;   // first absolute position stored remotely (P)
    uint32_t capacity; // compact row capacity (C-P)
    uint32_t reserved[2];
};

// Payload after header: ALL K rows followed by ALL V rows, matching CONFIG.
// Native Q4_0/Q8_0 rows are original packed ggml blocks: F16 scale followed
// by 16 nibble bytes / 32 signed bytes, 32 values/block; no requantization.
// start_row==0 replaces the layer's history; otherwise it must append at the
// current row count. Uploads contain no gaps, and KV positions are origin+row.
struct rkva_range_upload {
    uint32_t start_row;
    uint32_t n_rows;
    uint32_t wire; // RKVA_WIRE_F16, RKVA_WIRE_Q4_0 or RKVA_WIRE_Q8_0
    uint32_t reserved;
};

// Payload: positions[n_tokens] I32 (absolute), Q[n_tokens][n_head][D] F32.
// Response: ALL OUT[n_tokens][n_head][D] F32, then ALL LSE[n_tokens][n_head]
// F32 (natural log, with identical scale/causal semantics; empty => O=0,LSE=-inf).
// No append occurs here; RANGE_UPLOAD must precede queries for new positions.
struct rkva_range_attn {
    uint32_t wire;  // must be RKVA_WIRE_F32
    uint32_t flags; // must be 0
    uint32_t reserved[2];
};

// Payload: this header, K plane, V plane, query positions, F32 Q. The request
// n_tokens counts queries; upload.n_rows counts the appended KV chunk.
struct rkva_range_upload_attn {
    rkva_range_upload upload;
    rkva_range_attn attention;
};

struct rkva_range_profile {
    uint32_t flags;
    uint32_t q4_kernel_mode; // 0 reference, 1 balanced, 2 fine, 3 coarse optimized
    uint64_t reserved;
};

// Native KVarN4 revision 1: exact host-packed records, not a Q4_0 alias.
// All integers little endian. Descriptor validation is mandatory before allocation.
// Record layout: 8192 payload bytes, scale[128] F16, zero[128] F16,
// other[128] F16. K indices [dimension][token], V [token][dimension].
// Record group i is GLOBAL group origin/128+i; only GLOBAL group 0 is a sink.
// Rotated stage [slot*128+token][KV head*2+slice][128] F16; canonical slot
// is global_group==0 ? 0 : 1+(global_group-1)%2. Host explicit stage maps
// must be reordered into these canonical slots before transport.
// Exact precision tail is ORIGINAL-domain F16 [token][KV head][256],
// containing the most recent min(live_count,tail_tokens) rows in order.
struct rkva_kvarn_range_config {
    uint32_t origin, capacity;
    uint32_t storage;        // KVARN4, or Q4_0/Q8_0 for layout 4/5 native rows
    uint32_t layout_version; // 1: legacy; 2: same layout + generation ACK for resident rewind;
                             // 4: native Q4_0 rows; 5: native Q8_0 rows (no records/stage/tail)
    uint32_t token_group, record_dim, head_dim, head_slices;
    uint32_t key_bits, value_bits, record_bytes;
    uint32_t stage_groups, tail_groups, sink_tokens;
    uint32_t precision_tail_tokens; // 0, 128, or negotiated flexible length
    uint32_t wht_version;           // 1: H128 normalized, then normalized cross-slice mix
    uint32_t domain;                // 1: K and V records/stage rotated; Q/OUT wire original
    uint32_t record_policy;         // 1: eager committed records; delayed/SWA unsupported
};

// Payload planes: K records[record_groups][physical_heads][8960], V records,
// then K stage rows, V stage rows, K precision tail rows, V precision tail rows.
// RESET snapshots contain all 3*128 physical stage rows and
// min(live_count,precision_tail_tokens) chronological original-domain tail rows.
// When COMPACT_STAGE was negotiated, OMIT_STAGE may omit both stage planes only
// with descriptor revision 3, origin >= 128 and precision_tail_tokens >= 128.
// Every incomplete group is then entirely covered by the exact tail, and no
// global group-0 sink can read stage. Records and all per-prefix tail rows stay.
// Append payloads otherwise contain append_rows stage and tail rows at positions
// [append_begin,live_count); the server maps these to canonical stage/tail rings.
// RESET_SNAPSHOT flag=1: discard history and rebuild with record_begin=0.
// Otherwise live_count may only advance, and records must extend the existing
// contiguous coverage. After RESET/TRIM/REWIND a RESET_SNAPSHOT is mandatory;
// attention fails closed until then, even when old buffers remain allocated.
// Bulk v1 retains the exact op16/upload then op14/query image ordering.
// Header followed by segment_count segments: segment header, upload_bytes
// bytes (rkva_kvarn_range_upload + planes), then rkva_range_attn, positions
// and Q (F32 by default; F16 when RKVA_CAP_KVARN_BULK_F16 is advertised and
// every segment sets attention.wire == RKVA_WIRE_F16). Response is ALL OUT
// (matching attention.wire, F32 or F16), followed by ALL LSE (F32), in order.
// Descriptor revision 3, or layouts 4/5 (native Q4_0/Q8_0 rows); <=256 total
// queries, <=32 segments, 1..9 queries per segment (8 for prefill). Envelope is
// validated before any GPU mutation.
struct rkva_kvarn_range_bulk {
    uint32_t version, segment_count, total_queries, reserved;
};
struct rkva_kvarn_range_bulk_segment {
    uint32_t upload_bytes, n_queries;
};

struct rkva_kvarn_range_upload {
    uint32_t live_count, record_begin, record_groups, flags;
    uint32_t append_begin, append_rows;
    uint32_t reserved[2];
};

// Native resident rewind: header, then one layer header and chronological
// K/V ORIGINAL-domain F16 rows evicted from the retained tail for every layer:
// [max(0,new_live-tail_tokens), min(new_live,max(0,expected_live-tail_tokens))).
// Overlapping ring rows stay resident; a depth-one rejection repairs one row.
// All expected counts and <=128 removal bounds are validated before GPU writes;
// one final fence precedes publication/ACK. No native records are transcoded.
struct rkva_kvarn_range_rewind {
    uint64_t expected_generation; // v2 config ACK seed; advances on every native mutation
    uint32_t pos0, n_layers, tail_tokens, reserved;
};
struct rkva_kvarn_layer_rewind {
    uint32_t expected_live, new_live;
};

// Revision 1 key is SHA256 of the COMPLETE native host sequence file (header,
// token IDs, KV payload), not a path or user label. Revision 2 native snapshots
// key by the SHA256 of the complete host prefix through the end of the RKVK
// frame, so server SCKP checkpoint bytes appended after the llama state do not
// change it. Host
// validates/restores its authoritative snapshot before RESTORE and never skips
// its CPU mirror. Only call at a quiescent evaluated checkpoint, with all
// layers at the same live_rows.
// SAVE/RESTORE are optional, NEVER sent for live_rows==0 or positions below P.
// Revision 1 has no per-token save; host must opt in explicitly.
struct rkva_snapshot_req {
    uint8_t host_state_sha256[32];
    uint32_t live_rows;
    uint32_t revision; // 1
    uint32_t flags;    // 0
    uint32_t reserved; // 0
};
struct rkva_snapshot_ack {
    uint8_t host_state_sha256[32];
    uint32_t live_rows;
    uint32_t n_layers;
    uint64_t disk_bytes; // native stored payload bytes, no LAN KV transfer
};

// Revision 2 native extension, appended after rkva_snapshot_req/ack. Old
// servers reject revision 2 before touching state. expected_generation
// authenticates the currently configured resident session; saved_generation
// binds the on-disk manifest to the host trailer that was restored.
struct rkva_native_snapshot_req {
    uint64_t expected_generation;
    uint64_t saved_generation;
    uint32_t format_version; // RKVA_NATIVE_SNAPSHOT_FORMAT
    uint32_t flags;          // 0
};
struct rkva_native_snapshot_ack {
    uint64_t generation;       // new resident generation after RESTORE
    uint64_t payload_checksum; // canonical manifest checksum
    uint32_t layers;
    uint32_t reserved;
};

// Status matches only keys already supplied by the host; it does not expose
// unrelated snapshot keys. Retire accepts one such 256-bit key.
struct rkva_native_store_request {
    uint32_t action; // 0=status, 1=retire, 2=verify complete, 3=verify recovery
    uint32_t keys;
};
struct rkva_native_store_status {
    uint64_t used_bytes;
    uint64_t limit_bytes;
    uint32_t objects;
    uint32_t max_objects;
    uint32_t matches;
    uint32_t reserved;
};
struct rkva_native_store_match {
    uint32_t key_index;
    uint32_t reserved;
    uint64_t bytes;
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
    uint32_t probe_id; // RKVA_PROBE_*
    uint32_t mode;     // probe-specific
    uint32_t blob_bytes;
    uint32_t reserved;
};

enum rkva_probe : uint32_t {
    RKVA_PROBE_STORE = 1,
    RKVA_PROBE_ATTENTION = 2,
    RKVA_PROBE_WHT = 3,
    RKVA_PROBE_MATERIALIZE = 4,
};

#pragma pack(pop)

static_assert(sizeof(rkva_request) == 32, "RKVA request wire size");
static_assert(sizeof(rkva_response) == 56, "RKVA response wire size");
static_assert(sizeof(rkva_native_store_request) == 8, "native store request wire size");
static_assert(sizeof(rkva_native_store_status) == 32, "native store status wire size");
static_assert(sizeof(rkva_native_store_match) == 16, "native store match wire size");
static_assert(sizeof(rkva_hello) == 84, "RKVA hello wire size");
static_assert(sizeof(rkva_hello_ack) == 40, "RKVA hello ack wire size");
static_assert(sizeof(rkva_session_req) == 16, "RKVA session wire size");
static_assert(sizeof(rkva_attn_req) == 16, "RKVA attn req wire size");
static_assert(sizeof(rkva_trim_req) == 4, "RKVA trim wire size");
static_assert(sizeof(rkva_probe_req) == 16, "RKVA probe wire size");
static_assert(sizeof(rkva_range_config) == 16, "RKVA range config wire size");
static_assert(sizeof(rkva_range_upload) == 16, "RKVA range upload wire size");
static_assert(sizeof(rkva_range_attn) == 16, "RKVA range attention wire size");

static_assert(sizeof(rkva_kvarn_range_config) == 72, "KVarN descriptor wire size");
static_assert(sizeof(rkva_kvarn_range_upload) == 32, "KVarN upload wire size");
static_assert(sizeof(rkva_kvarn_range_rewind) == 24, "KVarN rewind wire size");
static_assert(sizeof(rkva_kvarn_layer_rewind) == 8, "KVarN rewind layer wire size");

static_assert(sizeof(rkva_snapshot_req) == 48, "snapshot request wire size");
static_assert(sizeof(rkva_snapshot_ack) == 48, "snapshot response wire size");
static_assert(sizeof(rkva_native_snapshot_req) == 24, "native snapshot request wire size");
static_assert(sizeof(rkva_native_snapshot_ack) == 24, "native snapshot response wire size");

static_assert(sizeof(rkva_range_upload_attn) == 32, "fused range request wire size");
static_assert(sizeof(rkva_range_profile) == 16, "range profile control wire size");
