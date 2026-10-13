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
// Wire format version 2 (F4 position_split), explicit little-endian:
//   Same leading layout as v1, then the v1 header grows a position_split block.
//   v2 is a strict superset of v1 so a v1 reader fails closed on the magic:
//
//   [0]   magic           u32   "KMS2" (0x32534D4B)
//   [4]   version         u32   2
//   [8]   flags           u32   full | partial | body_only
//   [12]  owner_epoch     u64
//   [20]  owner_id        u64
//   [28]  n_layers        u32   descriptor count (2 per full layer when split)
//   [32]  n_used          u32   shared occupied cell prefix length
//   [36]  metadata_version u32
//   [40]  metadata_bytes  u64
//   [48]  metadata_checksum u64
//   [56]  split_p         u32   local boundary P
//   [60]  split_c         u32   total logical capacity C
//   [64]  model_id        u64   caller-defined model identity hash
//   [72]  rope_id         u64   caller-defined RoPE identity hash
//   [80]  reserved        u32[4] all zero
//   [96]  header_crc      u64   XXH64(seed 0) over bytes [0, 104) with the
//                               crc field itself zeroed
//   [104] metadata blob (metadata_bytes)          once, full/partial only
//        per descriptor, in order:
//          layer descriptor (fixed 113 bytes = the 101 v1 bytes followed by
//            range_id u32, range_start u32, range_end u32)
//          payload bytes (payload_bytes)          per payload_mode
//        shared cell summary, one entry per cell of the prefix [0, n_used):
//          pos i32, x i32, y i32, tok i32, n_seq u16, seq ids i32 * n_seq
//        tail_crc u64   XXH64(seed 0) over every byte preceding it
//
// v2 range contract: descriptor identity is the PAIR (layer_id, range_id).
// Every full layer carries exactly two descriptors, range_id 0 = local
// [0,P) and range_id 1 = overflow [P,C), with no gap and no overlap at P.
// An empty range may omit its payload bytes but MUST keep its descriptor.
// P/C/model_id/rope_id are matched EXACTLY against the destination before any
// staging is reachable, so a mismatched restore fails without mutating state.
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
    layout_standard_qx_compact_rows = 3, // split v2 overflow: physical row = global cell - P
};

// Shared by native compact Qx storage and its codec tests. The logical cell
// must belong to the owned interval [origin,capacity); never wrap or clamp it.
uint32_t qx_payload_row_for_cell(uint32_t origin, uint32_t capacity, uint32_t global_cell);

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
    uint32_t layer_id = 0; // model layer index; must be unique (v1) / paired with range_id (v2)
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
    // F4 v2 range identity. Wire-serialized ONLY in version 2; the v1 codec
    // ignores these fields on read and rejects them on write, so a v1 frame can
    // never carry range identity it cannot validate. Uniqueness is on the PAIR
    // (layer_id, range_id): one full layer owns range_id 0 = local [0,P) and
    // range_id 1 = overflow [P,C).
    uint32_t range_id = 0;  // 0 = local [0,P), 1 = overflow [P,C)
    uint32_t range_start = 0; // inclusive logical cell index (v2 only)
    uint32_t range_end = 0;   // exclusive logical cell index (v2 only)
};

// F4 v2 envelope: position_split identity + occupancy of the two ranges.
// All fields are REQUIRED in a v2 snapshot (no defaults are inferred) and must
// match the destination exactly in validate_expected_snapshot(); a mismatch
// fails BEFORE any staging, so a P/C/model/rope divergence never mutates state.
struct position_split {
    uint32_t p = 0;  // local capacity boundary [0,P)
    uint32_t c = 0;  // total logical capacity [0,C)
    uint64_t model_id = 0; // caller-defined stable model identity hash
    uint64_t rope_id = 0;  // caller-defined stable RoPE/positional identity hash
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

// Serialized snapshot identity. v1 leaves position_split empty and travels in
// the v1 header. v2 REQUIRES a populated position_split and travels in the v2
// header, which is a strict superset (the same first 96 bytes plus the
// position_split block) so v1 readers fail closed on the v2 magic/version.
// ORIGINAL WRITER identity of a PURE (non-split) frame, serialized in the KMS3
// header and covered by the header checksum (integrity, not authenticity). It
// travels WITH the payload, so it is bound to the exact validated bytes and
// cannot be supplied, forged from the destination, or left stale by a previous
// load: each parse produces its own value and the frame fails closed without it.
struct source_identity {
    uint64_t model_id  = 0; // llama_mixed_state_model_id of the WRITER
    uint64_t rope_id   = 0; // llama_mixed_state_rope_id of the WRITER
    uint64_t config_id = 0; // descriptor_config_id of the WRITER's layer set
};

struct serialize_options {
    // empty => emit a v1/v3 frame; populated => emit a v2 frame. A v2 frame
    // with p == 0 or c <= p is rejected before any byte is produced.
    std::optional<position_split> split;
    // populated => emit a KMS3 frame carrying the ORIGINAL WRITER identity
    // (used by pure, non-split snapshots so a split destination can verify the
    // source separately from itself). Ignored when `split` is set: a split
    // frame already carries identity in its envelope.
    std::optional<source_identity> source;
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
    // F4 v2 range identity; both descriptors of one full layer share layer_id
    // and are told apart by range_id only.
    uint32_t range_id;
    uint32_t range_start;
    uint32_t range_end;

    static expected_layer from_desc(const layer_desc & d);
};

struct expected_snapshot {
    std::vector<expected_layer> layers; // exact set; duplicates rejected on (layer_id, range_id)
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
    // F4 v2: REQUIRED when the snapshot carries a position_split envelope (the
    // caller must know its own P/C/model/rope) and rejected as a mismatch when
    // it does not. P/C/model/rope identity is compared EXACTLY: a v2 snapshot
    // can never be replayed into a differently placed cache.
    std::optional<position_split> split;
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
    // F4 v2: layer_id is no longer unique on its own. Handlers receive the
    // range_id alongside it so the two descriptors of one full layer stage into
    // their OWN range buffers and cannot overwrite each other. range_id is 0 for
    // every v1 frame, which keeps existing v1 handlers correct unchanged.
    std::function<bool(uint32_t layer_id, uint32_t range_id, const layer_desc &, const uint8_t * payload, size_t size)> stage_layer;
    std::function<bool(uint32_t metadata_version, const uint8_t * metadata, size_t size)> stage_metadata;
    std::function<void()> finish; // swap staged state in; infallible by contract
    std::function<void()> abort;  // roll back anything staged by this handler set
    // Optional F4 v2 hook invoked BEFORE any layer staging, carrying the
    // validated position_split identity (p, c, model_id, rope_id) so the caller
    // can stage DeltaNet/recurrent state and the logical cell table for both
    // ranges as ONE unit. A false return aborts with nothing published, which is
    // what makes the KV + table + DeltaNet restore atomic. It is declared LAST so
    // the historical positional initialization order of the handlers above stays
    // source compatible; a v2 commit REQUIRES it and fails closed without it.
    std::function<bool(const position_split &)> stage_split;
};


// Canonical hash over the CONFIG-relevant, range-geometry-independent
// descriptor fields of a layer set. Written into a KMS3 frame as
// `config_id` and recomputed on read, so a source config is compared against
// the destination's separately from model/RoPE.
uint64_t descriptor_config_id(const std::vector<layer_desc> & layers);

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
    // F4 v2: the position_split envelope of a parsed frame; empty for v1.
    // v1 frames can never present one, so a caller can branch on this to tell
    // which layout it is holding.
    const std::optional<position_split> & split() const { return split_; }
    bool has_split() const { return split_.has_value(); }
    // KMS3: ORIGINAL WRITER identity of a pure frame. Absent on legacy v1
    // frames, which therefore fail closed for a pure->split conversion.
    const source_identity & frame_source_identity() const { return source_; }
    bool has_source_identity() const { return source_present_; }
    const std::vector<layer_desc> & layers() const { return layers_; }
    const std::vector<cell_entry> & cells() const { return cells_; }
    const uint8_t * payload(size_t layer_index) const { return payloads_[layer_index].data; }
    size_t payload_size(size_t layer_index) const { return payloads_[layer_index].size; }

    // Plan 3.6 (lines 187-191): authorized import of a PURE KVarN (non-split)
    // snapshot into a position_split destination. The source carries ONE
    // descriptor per layer; the split destination owns two (local + overflow).
    // This adopts the destination envelope and materializes the missing
    // descriptors as EMPTY ranges -- bytes omitted, description kept -- so the
    // existing v2 integrity check, staged reader and atomic commit run
    // UNCHANGED. Model/RoPE/geometry identity is established by the caller's
    // layout predicate (common_prompt_cache_layout_pure_to_split); this call
    // only guarantees that no synthesized range carries payload and that the
    // source is fully covered by the destination descriptor set.
    //
    // Contract enforced HERE, before anything is rebound:
    //  * source identity must be established by the caller's earlier trusted
    //    check (server-context.cpp: common_prompt_cache_layout_pure_to_split)
    //    and must equal the destination's model/RoPE identity;
    //  * every invariant descriptor field (formats, dimensions, strides,
    //    domain, layout, owner, tail, payload shape) must already agree with
    //    the authorized conversion; equal payload LENGTH alone is not
    //    compatibility;
    //  * payload_bytes must match the mapped payload actually present;
    //  * only range_start/range_end are rewritten, and payload_checksum of the
    //    source is preserved verbatim as authenticated payload provenance.
    void adopt_split_for_import(const position_split & want,
                                const std::vector<layer_desc> & destination_descs,
                                const source_identity & source);

    // Explicit inverse import, full snapshots only. Validate the original
    // authenticated split envelope and BOTH source ranges before projecting
    // local descriptors into a pure KVarN destination. Occupancy must fit
    // min(source P, destination capacity); every overflow payload must be
    // empty. All local representation/tail/stride invariants match exactly.
    // This does not edit the source bytes or opaque I/O manifest. The native
    // caller must separately consume and validate the empty overflow trailer
    // before registering the existing deferred KV/table/recurrent commit.
    // Failure leaves this snapshot's descriptor/payload ownership unchanged.
    void project_split_to_pure_for_import(
            uint32_t destination_capacity,
            uint64_t destination_model_id,
            uint64_t destination_rope_id,
            const std::vector<layer_desc> & destination_descs);

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
                          const parse_options & opts = {},
                          const serialize_options & ser = {});

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
    // F4 v2 envelope; unset means this is a v1 frame.
    std::optional<position_split> split_;
    source_identity source_;
    bool source_present_ = false;
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
