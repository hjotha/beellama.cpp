#pragma once

// Versioned, self-contained codec for mixed KVarN/standard-Qx KV-cache
// snapshots. Independent of llama-kv-cache / llama-kv-cache-kvarn internals:
// cache and cell data travel as opaque caller-provided blobs, so the codec
// does not serialize any class currently under ABI-editing by other fronts.
//
// Wire format (explicit little-endian, never raw C structs), version 1:
//
//   [0]   magic           u32   "KMS1" (0x31534D4B)
//   [4]   version         u32   1
//   [8]   flags           u32   full | partial | body_only
//   [12]  owner_epoch     u64   required != 0 for partial snapshots
//   [20]  owner_id        u64   owner tag for partial snapshots
//   [28]  n_layers        u32   layer descriptor count
//   [32]  n_used          u32   shared occupied cell prefix length
//   [36]  metadata_version u32  wrapper-defined version of the metadata blob
//   [40]  metadata_bytes  u64   opaque metadata blob size
//   [48]  metadata_checksum u64 XXH64(seed 0) over the metadata blob
//   [56]  reserved        u32[8] all zero (space for future extension)
//   [88]  header_crc      u64   XXH64(seed 0) over bytes [0, 96) with the crc
//                               field itself zeroed
//   [96]  metadata blob (metadata_bytes)          once, full/partial only
//        per layer, in order:
//          layer descriptor (fixed 101 bytes, see layer_desc wire order)
//          payload bytes (payload_bytes)          per payload_mode; see below
//        shared cell summary, one entry per cell of the prefix [0, n_used):
//          pos i32, x i32, y i32, tok i32, n_seq u16, seq ids i32 * n_seq
//          (pos == -1 encodes a hole inside the prefix; summary only, the
//          authoritative shared-cache state is the metadata blob)
//        tail_crc u64   XXH64(seed 0) over every byte preceding it
//
// The cell summary is a validated, lightweight representation used for
// capacity/prefix checks only. ALL authoritative shared-cache state
// (allocation groups, stage slots, checkpoints, cell ownership, tail
// generations, remaps) must travel in the versioned opaque metadata blob
// provided by the wrapper. This codec never invents a complete serialization
// of llama_kv_cache metadata; pos+seq alone is NOT sufficient state.
//
// Integrity: header_crc + per-payload checksum + metadata checksum +
// whole-file tail_crc, all XXH64 seed 0 (same as src/llama-io-file.h).
// Corruption detection, not cryptographic authentication.
//
// Snapshot flags contract:
//   full      - metadata blob + cells + descriptors + complete payloads
//               (every layer payload_mode == mode_full); replayable anywhere.
//   partial   - metadata blob + cells + descriptors + per-layer payloads
//               according to each layer's payload_mode:
//                 mode_partial_overlay    -> overlay bytes included (KVarN
//                    stages/intrinsic tail/record fixups must be preserved)
//                 mode_resident_reference -> no bytes; Qx append-only body
//                    referenced while still resident in the originating owner
//               Owner/epoch-bound: validate_expected_snapshot() requires an
//               exact owner match before commit is allowed.
//   body_only - descriptors + complete payloads only (mode_full); no metadata
//               blob and no cells (restore keeps the destination cell layout).
//               partial|body_only is rejected.
//
// Limits (documented, enforced):
//   - Shared cells are a contiguous prefix [0, n_used); holes inside the
//     prefix are serialized with pos == -1. Non-prefix cells are not
//     representable in this version.
//   - n_used, layer ids and cell indices are u32; payload/metadata lengths
//     are u64 with checked bounds against the remaining input and against
//     caller-supplied caps in parse_options, so malicious/truncated lengths
//     fail before any large allocation.
//   - n_layers <= 4096, n_seq per cell <= 256 (LLAMA_MAX_SEQ).
//   - n_stream field is stored for exact-match validation but only 1 stream is
//     a valid layout in this version (multi-stream/SWA ring layouts reserved).
//
// API flow (staged, no partial application):
//   1. parse(data, size, opts)        - full validation, no side effects.
//   2. validate_expected_snapshot(expected, opts)
//      - exact structural match (no wildcards; every expected field must
//        equal the snapshot field), capacity bound, partial owner contract.
//        Single-shot: a second call fails.
//   3. commit(handlers)               - transactional: stage everything first
//      (destination untouched), then finish() swaps staged state in; any
//      stage failure invokes abort() and leaves the destination untouched.
//      Exactly one commit attempt per snapshot; the ticket is consumed by any
//      attempt, including failed ones.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace llama_kv_mixed {

// numeric ids are part of the wire format; do not renumber
enum class cache_kind : uint8_t {
    unknown     = 0,
    kvarn       = 1, // structured joint 128-token tile records
    standard_qx = 2, // conventional ggml row cache
};

// snapshot flags (bit values are part of the wire format)
enum class snapshot_flags : uint32_t {
    none      = 0,
    full      = 0, // default: blob + cells + descriptors + payloads
    partial   = 1, // blob + cells + descriptors; payloads omitted; owner-bound
    body_only = 2, // descriptors + payloads; blob and cells omitted
};

inline snapshot_flags operator|(snapshot_flags a, snapshot_flags b) {
    return snapshot_flags(uint32_t(a) | uint32_t(b));
}
inline snapshot_flags operator&(snapshot_flags a, snapshot_flags b) {
    return snapshot_flags(uint32_t(a) & uint32_t(b));
}

// KVarN domain (wire). Numeric values are identical to
// GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_* in ggml/include/ggml.h.
enum class kvarn_domain : uint8_t {
    auto_domain = 0, // GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO
    rotated     = 1, // GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED
    original    = 2, // GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ORIGINAL
    rotated_k_original_v = 3, // GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V
};

// Per-tensor pre-rotation (wire). Standard caches record the rotation applied
// to K/V before quantization: llama_kv_cache attn_rot_k / attn_rot_v
// (LLAMA_ATTN_ROT_DISABLE), hadamard width in elements (Qwen3.5/3.6: K 256,
// V 64). This differentiates unrotated from rotated data, which is not
// recoverable from the quantized bytes.
enum class rotation : uint8_t {
    none     = 0,
    hadamard = 1,
};

// Payload layout ids (part of the wire format). KVarN stages/tail/records and
// Qx rows are different physical layouts and must never be interchanged.
enum layout_id : uint16_t {
    layout_kvarn_records_stage_tail = 1, // KVarN: record groups + stage rows + optional exact tail
    layout_standard_qx_rows         = 2, // standard: contiguous Qx rows per stream
};

// Per-layer payload mode (part of the wire format). Determines whether and
// how payload bytes travel with the snapshot:
//   mode_full               - complete payload bytes included; replayable
//                             anywhere (mandatory for full/body_only).
//   mode_partial_overlay    - mutable overlay bytes included (KVarN stages/
//                             intrinsic tail/record fixups must be preserved:
//                             llama_kv_cache_kvarn::requires_state_for_partial_restore()
//                             is true). Still owner/epoch-bound: the snapshot
//                             is partial and only restoreable under the exact
//                             originating owner.
//   mode_resident_reference - no payload bytes; references body still
//                             resident in the originating owner (standard Qx
//                             append-only body). Owner/epoch-bound.
enum payload_mode : uint8_t {
    mode_full               = 1,
    mode_partial_overlay    = 2,
    mode_resident_reference = 3,
};

// Logical owner ids (part of the wire format)
enum logical_owner : uint8_t {
    owner_target    = 0, // target model cache
    owner_draft_mtp = 1, // draft-owned MTP cache
    owner_auxiliary = 2, // other auxiliary context
};

// Standard cache type ids (wire). Numeric values are identical to the
// ggml_type enum in ggml/include/ggml.h for the fork's cache-facing formats;
// translation to ggml_type lives in the wrapper, this codec only stores and
// matches ids.
enum standard_qx_type : uint16_t {
    sx_f16   = 1,  // GGML_TYPE_F16
    sx_q4_0  = 2,  // GGML_TYPE_Q4_0
    sx_q4_1  = 3,  // GGML_TYPE_Q4_1
    sx_q5_0  = 6,  // GGML_TYPE_Q5_0
    sx_q5_1  = 7,  // GGML_TYPE_Q5_1
    sx_q8_0  = 8,  // GGML_TYPE_Q8_0
    sx_bf16  = 30, // GGML_TYPE_BF16
    sx_q6_0  = 43, // GGML_TYPE_Q6_0
    sx_q6_1  = 44, // GGML_TYPE_Q6_1
    sx_q3_0  = 45, // GGML_TYPE_Q3_0
    sx_q3_1  = 46, // GGML_TYPE_Q3_1
    sx_q2_0s = 47, // GGML_TYPE_Q2_0S (Bee cache-facing Q2_0)
    sx_q2_1  = 48, // GGML_TYPE_Q2_1
};

// KVarN joint type ids (wire). Numeric values are identical to the
// llama_kvarn_type enum in include/llama.h
// (LLAMA_KVARN_K2V2_G128 = 1 ... LLAMA_KVARN_K8V8_G128 = 36).
enum kvarn_type : uint16_t {
    kv2v2 = 1,  kv2v3 = 2,  kv2v4 = 3,
    kv3v2 = 4,  kv3v3 = 5,  kv3v4 = 6,
    kv4v2 = 7,  kv4v3 = 8,  kv4v4 = 9,
    kv2v5 = 10, kv2v6 = 11, kv2v8 = 12,
    kv3v5 = 13, kv3v6 = 14, kv3v8 = 15,
    kv4v5 = 16, kv4v6 = 17, kv4v8 = 18,
    kv5v2 = 19, kv5v3 = 20, kv5v4 = 21,
    kv5v5 = 22, kv5v6 = 23, kv5v8 = 24,
    kv6v2 = 25, kv6v3 = 26, kv6v4 = 27,
    kv6v5 = 28, kv6v6 = 29, kv6v8 = 30,
    kv8v2 = 31, kv8v3 = 32, kv8v4 = 33,
    kv8v5 = 34, kv8v6 = 35, kv8v8 = 36,
};

struct layer_desc {
    uint32_t layer_id = 0; // model layer index; must be unique
    cache_kind kind = cache_kind::unknown;
    uint16_t k_type = 0; // standard_qx_type or kvarn_type wire id
    uint16_t v_type = 0;
    uint8_t k_bits = 0; // kvarn bit width; 0 for standard
    uint8_t v_bits = 0;
    uint8_t kvarn_domain = 0; // kvarn_domain wire id; must be 0 for standard
    uint8_t k_rot = 0;        // rotation wire id for K (standard caches; 0 for kvarn)
    uint8_t v_rot = 0;        // rotation wire id for V
    uint32_t k_rot_width = 0; // hadamard width in elements, 0 when rotation is none
    uint32_t v_rot_width = 0;
    uint16_t layout = 0;
    uint8_t owner = owner_target;
    uint8_t v_trans = 0; // standard cache V-transposed flag
    uint16_t tail_type = 0; // kvarn exact-tail ggml type id, 0 = none
    uint32_t token_group = 0; // kvarn group (128) / standard block
    uint32_t record_dim = 0;  // kvarn record dim
    uint32_t head_dim_k = 0;  // n_embd_head_k
    uint32_t head_dim_v = 0;  // n_embd_head_v
    uint32_t head_slices = 0; // kvarn head slices
    uint32_t n_head_kv = 0;   // kv heads (Qwen: 4)
    uint32_t n_stream = 1;    // only 1 valid in this version
    uint64_t k_stride = 0;    // row/record bytes
    uint64_t v_stride = 0;
    uint8_t payload_mode = mode_full; // payload_mode wire id (see enum)
    uint32_t payload_cells = 0; // cells covered by the payload
    uint64_t payload_rows = 0;  // rows (kvarn stage rows / record groups; qx token rows)
    uint64_t payload_bytes = 0; // exact payload size; 0 = derive from blob size at serialize
    uint64_t payload_checksum = 0; // XXH64(seed 0) over payload bytes (set by codec)
};

// validated cell summary: lightweight, used for capacity/prefix checks only
struct cell_entry {
    int32_t pos = -1; // -1 = hole inside the prefix
    int32_t x = 0;
    int32_t y = 0;
    int32_t tok = 0; // LLAMA_TOKEN_NULL-compatible (-1)
    std::vector<int32_t> seq_ids; // up to 256 entries
};

// Opaque caller-provided payload blob for one layer
struct layer_blob {
    layer_desc desc;
    const uint8_t * data = nullptr;
    size_t size = 0;
};

struct parse_options {
    std::function<void(const std::string & message)> on_fault; // invoked before throwing
    // caller-provided hard caps, enforced BEFORE copying any large blob:
    uint64_t max_total_bytes = 0;   // whole snapshot; 0 = input size
    uint64_t max_payload_bytes = 0; // per-layer payload; 0 = remaining input
    uint64_t max_metadata_bytes = 0;// metadata blob; 0 = remaining input
};

// Strict expected descriptor: NO wildcards. Every field must equal the
// snapshot's field; compatibility fails closed on any difference. Use
// from_desc() to seed from the destination's known layer.
struct expected_layer {
    uint32_t layer_id;
    cache_kind kind;
    uint16_t k_type;
    uint16_t v_type;
    uint8_t k_bits;
    uint8_t v_bits;
    uint8_t kvarn_domain;
    uint8_t k_rot;
    uint8_t v_rot;
    uint32_t k_rot_width;
    uint32_t v_rot_width;
    uint16_t layout;
    uint8_t owner;
    uint8_t v_trans;
    uint16_t tail_type;
    uint32_t token_group;
    uint32_t record_dim;
    uint32_t head_dim_k;
    uint32_t head_dim_v;
    uint32_t head_slices;
    uint32_t n_head_kv;
    uint32_t n_stream;
    uint64_t k_stride;
    uint64_t v_stride;
    uint8_t payload_mode;
    uint32_t payload_cells;
    uint64_t payload_rows;
    uint64_t payload_bytes;

    static expected_layer from_desc(const layer_desc & d);
};

struct expected_snapshot {
    std::vector<expected_layer> layers; // exact set; duplicate layer ids rejected
    // destination capacity; REQUIRED (validate fails if unset): n_used <= capacity
    std::optional<uint64_t> capacity_cells;
    // expected metadata schema version of the opaque blob. REQUIRED for
    // full/partial snapshots (checked before commit is reachable); ignored
    // for body_only (no blob).
    std::optional<uint32_t> metadata_version;
    // partial snapshots REQUIRE both owner fields set and exactly equal to the
    // snapshot's; for full/body_only they are optional exact checks.
    std::optional<uint64_t> owner_epoch;
    std::optional<uint64_t> owner_id;
};

// Transactional commit: no publication happens before finish(). stage_*
// handlers must NOT touch the destination; they only stage into caller-owned
// buffers. finish() swaps the staged state into the live destination and must
// be infallible by contract (no rollback exists after it). If any stage
// handler returns false or throws, abort() is invoked and the destination was
// never touched by the codec. Atomicity against the destination is a contract
// of these handlers; the codec guarantees ordering, single-shot tickets and
// abort-on-stage-failure.
// Required handlers (absent => the commit attempt fails BEFORE any staging and
// consumes the ticket): stage_layer, finish and abort always; stage_metadata
// whenever the snapshot carries the metadata blob (full/partial).
struct commit_handlers {
    std::function<bool(uint32_t layer_id, const layer_desc &, const uint8_t * payload, size_t size)> stage_layer;
    std::function<bool(uint32_t metadata_version, const uint8_t * metadata, size_t size)> stage_metadata;
    std::function<void()> finish; // swap staged state in; infallible by contract
    std::function<void()> abort;  // roll back anything staged by this handler set
};

class kv_mixed_error : public std::runtime_error {
public:
    explicit kv_mixed_error(const std::string & message) : std::runtime_error(message) {}
};

class snapshot {
public:
    // serialized fields
    uint32_t version() const { return version_; }
    snapshot_flags flags() const { return flags_; }
    uint64_t owner_epoch() const { return owner_epoch_; }
    uint64_t owner_id() const { return owner_id_; }
    uint32_t n_used() const { return n_used_; }
    uint32_t metadata_version() const { return metadata_version_; }
    const uint8_t * metadata_blob() const { return metadata_.data(); }
    size_t metadata_size() const { return metadata_.size(); }
    const std::vector<layer_desc> & layers() const { return layers_; }
    const std::vector<cell_entry> & cells() const { return cells_; }
    const uint8_t * payload(size_t layer_index) const { return payloads_[layer_index].data; }
    size_t payload_size(size_t layer_index) const { return payloads_[layer_index].size; }

    // Serialize a complete, validated snapshot. For full/partial the opaque
    // metadata blob (version + bytes) is mandatory; for body_only it must be
    // absent. Per-layer payload_mode must be consistent with the flags
    // (full/body_only: mode_full; partial: mode_partial_overlay with the
    // caller-provided exact overlay bytes or mode_resident_reference with
    // none). Rejects duplicate layer ids, unknown kinds/types/domains/
    // rotations/layouts/modes/owners, invalid flag combinations, partial
    // without an owner epoch/id, payload size mismatches, and body_only
    // snapshots carrying blob/cell metadata.
    // Throws kv_mixed_error on any invalid input (opts.on_fault is invoked
    // once before the throw); returns true on success.
    static bool serialize(const std::vector<layer_blob> & layers,
                          const std::vector<cell_entry> & cells,
                          uint32_t metadata_version,
                          const uint8_t * metadata,
                          size_t metadata_size,
                          snapshot_flags flags,
                          uint64_t owner_epoch,
                          uint64_t owner_id,
                          std::vector<uint8_t> & out,
                          const parse_options & opts = {});

    // Parse + fully validate a snapshot (owning mode). No callbacks and no
    // state publication happen here; the result is a staged snapshot.
    // Payload bytes are copied into the snapshot (self-contained, independent
    // of the input buffer). Peak parse memory is bounded by the caller caps:
    // header copy + metadata blob + sum of payloads + cell summary +
    // descriptors. Throws kv_mixed_error on any malformed/truncated/unknown
    // input.
    static snapshot parse(const uint8_t * data, size_t size, const parse_options & opts = {});

    // Borrowed, read-only view parse for large payloads (e.g. 6.8 GiB of Q8
    // KV at 200K): payload()/payload_size() point into the caller's immutable
    // backing instead of copying every layer, so parse memory stays bounded
    // (header copy + metadata blob + cell summaries + descriptors; payload
    // bytes are zero-copy). The metadata blob and cell summaries are still
    // copied (small). Strict validation, checksums and transaction semantics
    // are identical to parse().
    //
    // Lifetime: shared_lifetime_owner must be a NON-NULL std::shared_ptr that
    // OWNS the immutable backing (e.g. a shared_ptr<kv_mixed_file_backing> or
    // a shared_ptr to the buffer holding the snapshot bytes). The snapshot
    // RETAINS that shared_ptr for its whole lifetime, so the backing stays
    // alive until the snapshot (and every move of it) is destroyed; the raw
    // pointer alone would not retain anything. The caller must still keep the
    // backing bytes IMMUTABLE: commit() re-hashes every borrowed payload and
    // compares it with the stored checksum before any stage handler runs, so
    // an in-place mutation of the backing after parse fails the commit — but
    // this is detection, not protection: use stable atomic snapshot files
    // (write to a temp file, then rename) and never rewrite the backing. The
    // codec does not implement file mapping — the integration caller supplies
    // the backing. This is NOT end-to-end streaming: the input is scanned and
    // hashed once during parse. A null shared_lifetime_owner is rejected.
    static snapshot parse_view(const uint8_t * data, size_t size,
                               std::shared_ptr<const void> shared_lifetime_owner,
                               const parse_options & opts = {});

    // true when the payloads are borrowed views into the caller backing
    bool is_borrowed() const { return borrowed_; }
    // the retained backing owner (empty in owning mode)
    const std::shared_ptr<const void> & backing() const { return backing_; }

    // Exact structural match (no wildcards) plus the partial-snapshot owner
    // contract, the required capacity bound and the required metadata schema
    // version (full/partial). Must pass before commit() may run. Single-shot:
    // a second call FAILS and INVALIDATES the ticket (commit afterwards is
    // blocked). Throws kv_mixed_error on mismatch.
    void validate_expected_snapshot(const expected_snapshot & expected, const parse_options & opts = {});

    // Transactional commit (see commit_handlers). Exactly one attempt per
    // snapshot: the ticket is consumed by any attempt, successful or not, and
    // commit() is unreachable unless validation passed. For borrowed
    // snapshots every payload is re-hashed and compared with its stored
    // checksum BEFORE any stage handler runs; a mutated backing fails the
    // commit (ticket consumed, nothing staged, abort not invoked because
    // nothing was staged).
    void commit(const commit_handlers & handlers);

    // snapshot is move-only: payload pointers must stay attached to the
    // objects that own the bytes, and the validation/commit ticket must never
    // be duplicated. A move transfers the ticket (validated/committed state)
    // to the destination and invalidates the moved-from object.
    snapshot() = default;
    snapshot(snapshot && other) noexcept;
    snapshot & operator=(snapshot && other) noexcept;
    snapshot(const snapshot &) = delete;
    snapshot & operator=(const snapshot &) = delete;

    // minimum/maximum version the codec understands
    static constexpr uint32_t format_version() { return 1; }

private:
    struct payload_ref {
        const uint8_t * data = nullptr;
        size_t size = 0;
    };

    uint32_t version_ = 1;
    snapshot_flags flags_ = snapshot_flags::full;
    uint64_t owner_epoch_ = 0;
    uint64_t owner_id_ = 0;
    uint32_t n_used_ = 0;
    uint32_t metadata_version_ = 0;
    std::vector<uint8_t> metadata_;
    std::vector<layer_desc> layers_;
    std::vector<cell_entry> cells_;
    // payloads_[i] always points at the final payload bytes: into
    // payload_owned_[i] in owning mode, into the caller backing in view mode.
    // payload_owned_ keeps owning-mode bytes alive and is empty in view mode.
    std::vector<payload_ref> payloads_;
    std::vector<std::vector<uint8_t>> payload_owned_;
    bool borrowed_ = false;
    std::shared_ptr<const void> backing_; // retained owner (borrowed mode)
    bool validated_ = false;
    bool committed_ = false;
    bool moved_from_ = false; // moved-from snapshots are permanently inert

    // shared parse path; a non-empty backing selects borrowed (view) mode
    static snapshot parse_impl(const uint8_t * data, size_t size,
                               const parse_options & opts,
                               std::shared_ptr<const void> backing);
};

} // namespace llama_kv_mixed