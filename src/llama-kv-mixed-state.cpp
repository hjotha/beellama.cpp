// Versioned, self-contained codec for mixed KVarN/standard-Qx KV-cache
// snapshots. See llama-kv-mixed-state.h for the wire format and contract.

#define XXH_INLINE_ALL
#include "../vendor/hash/xxhash/xxhash.h"

#include "llama-kv-mixed-state.h"

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <unordered_set>

namespace llama_kv_mixed {

namespace {

constexpr uint32_t KMS_MAGIC        = 0x31534D4B; // "KMS1"
constexpr uint32_t KMS_HEADER_BYTES = 96;
constexpr uint32_t KMS_DESC_BYTES   = 101;
constexpr uint32_t KMS_MAX_LAYERS   = 4096;
constexpr uint32_t KMS_MAX_SEQ      = 256; // LLAMA_MAX_SEQ
constexpr uint32_t KMS_MAX_CELLS    = 268435456u; // 1 << 28
// KMS_DESC_BYTES is the exact serialized descriptor size; the parse-time
// check below uses it as the minimum required before read_desc().
static_assert(KMS_DESC_BYTES == 101, "descriptor size must match write_desc/read_desc");

// canonical kvarn_type wire id -> (key_bits, value_bits). Translation of the
// llama_kvarn_type enum order in include/llama.h (ids 1..36).
struct kvarn_bits {
    uint8_t k;
    uint8_t v;
};
constexpr kvarn_bits KVARn_BITS[36] = {
    {2,2},{2,3},{2,4},{3,2},{3,3},{3,4},{4,2},{4,3},{4,4},
    {2,5},{2,6},{2,8},{3,5},{3,6},{3,8},{4,5},{4,6},{4,8},
    {5,2},{5,3},{5,4},{5,5},{5,6},{5,8},{6,2},{6,3},{6,4},
    {6,5},{6,6},{6,8},{8,2},{8,3},{8,4},{8,5},{8,6},{8,8},
};

bool is_kvarn_type(uint16_t id) {
    return id >= uint16_t(kv2v2) && id <= uint16_t(kv8v8);
}

bool is_standard_type(uint16_t id) {
    switch (id) {
        case sx_f16:
        case sx_q4_0:
        case sx_q4_1:
        case sx_q5_0:
        case sx_q5_1:
        case sx_q8_0:
        case sx_bf16:
        case sx_q6_0:
        case sx_q6_1:
        case sx_q3_0:
        case sx_q3_1:
        case sx_q2_0s:
        case sx_q2_1:
            return true;
        default:
            return false;
    }
}

void fault(const parse_options & opts, const std::string & message) {
    if (opts.on_fault) {
        opts.on_fault(message);
    }
    throw kv_mixed_error(message);
}

// little-endian explicit encoding
void put_u16(std::vector<uint8_t> & out, uint16_t v) {
    out.push_back(uint8_t(v & 0xFF));
    out.push_back(uint8_t((v >> 8) & 0xFF));
}
void put_u32(std::vector<uint8_t> & out, uint32_t v) {
    out.push_back(uint8_t(v & 0xFF));
    out.push_back(uint8_t((v >> 8) & 0xFF));
    out.push_back(uint8_t((v >> 16) & 0xFF));
    out.push_back(uint8_t((v >> 24) & 0xFF));
}
void put_u64(std::vector<uint8_t> & out, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(uint8_t((v >> (8 * i)) & 0xFF));
    }
}

struct reader {
    const uint8_t * data;
    size_t size;
    size_t pos = 0;
    const parse_options * opts;
    XXH64_state_t hash;

    reader(const uint8_t * data_, size_t size_, const parse_options * opts_)
        : data(data_), size(size_), opts(opts_) {
        XXH64_reset(&hash, 0);
    }

    void read(void * dst, size_t n) {
        if (pos > size || n > size - pos) {
            fault(*opts, "parse: truncated snapshot");
        }
        std::memcpy(dst, data + pos, n);
        XXH64_update(&hash, data + pos, n);
        pos += n;
    }

    uint16_t read_u16() {
        uint8_t b[2];
        read(b, 2);
        return uint16_t(b[0]) | (uint16_t(b[1]) << 8);
    }
    uint32_t read_u32() {
        uint8_t b[4];
        read(b, 4);
        return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
    }
    uint64_t read_u64() {
        uint8_t b[8];
        read(b, 8);
        uint64_t v = 0;
        for (int i = 7; i >= 0; --i) {
            v = (v << 8) | b[i];
        }
        return v;
    }
    uint8_t read_u8() {
        uint8_t b;
        read(&b, 1);
        return b;
    }
    uint64_t digest() const {
        return XXH64_digest(&hash);
    }
};

// field validation shared by serialize and parse (both fail closed).
// kvarn bit fields are derived from the joint type id when both are zero and
// cross-checked otherwise.
void validate_desc(layer_desc & d, bool serializing) {
    const std::string where = serializing ? "serialize" : "parse";
    switch (d.kind) {
        case cache_kind::kvarn:
            if (!is_kvarn_type(d.k_type) || !is_kvarn_type(d.v_type)) {
                throw kv_mixed_error(where + ": unknown kvarn type ids");
            }
            if (d.k_type != d.v_type) {
                throw kv_mixed_error(where + ": kvarn k_type must equal v_type (joint type id)");
            }
            {
                const kvarn_bits & b = KVARn_BITS[d.k_type - 1];
                if (d.k_bits == 0 && d.v_bits == 0) {
                    d.k_bits = b.k;
                    d.v_bits = b.v;
                } else if (d.k_bits != b.k || d.v_bits != b.v) {
                    throw kv_mixed_error(where + ": kvarn bit fields disagree with the type id");
                }
            }
            if (d.layout != layout_kvarn_records_stage_tail) {
                throw kv_mixed_error(where + ": unknown kvarn layout id");
            }
            if (d.v_trans != 0) {
                throw kv_mixed_error(where + ": kvarn layers cannot set v_trans");
            }
            if (d.tail_type != 0 && !is_standard_type(d.tail_type)) {
                throw kv_mixed_error(where + ": unknown kvarn exact-tail type");
            }
            if (d.kvarn_domain > uint8_t(kvarn_domain::rotated_k_original_v)) {
                throw kv_mixed_error(where + ": unknown kvarn domain id");
            }
            if (d.k_rot != 0 || d.v_rot != 0 || d.k_rot_width != 0 || d.v_rot_width != 0) {
                throw kv_mixed_error(where + ": kvarn layers cannot carry standard rotations");
            }
            break;
        case cache_kind::standard_qx:
            if (!is_standard_type(d.k_type) || !is_standard_type(d.v_type)) {
                throw kv_mixed_error(where + ": unknown standard qx type id");
            }
            if (d.layout != layout_standard_qx_rows) {
                throw kv_mixed_error(where + ": unknown standard layout id");
            }
            if (d.tail_type != 0) {
                throw kv_mixed_error(where + ": standard caches cannot carry a kvarn tail");
            }
            if (d.v_trans > 1) {
                throw kv_mixed_error(where + ": invalid v_trans flag");
            }
            if (d.kvarn_domain != 0) {
                throw kv_mixed_error(where + ": standard caches cannot carry a kvarn domain");
            }
            break;
        default:
            throw kv_mixed_error(where + ": unknown cache kind");
    }
    if (d.k_rot > uint8_t(rotation::hadamard) || d.v_rot > uint8_t(rotation::hadamard)) {
        throw kv_mixed_error(where + ": unknown rotation id");
    }
    if ((d.k_rot == 0) != (d.k_rot_width == 0)) {
        throw kv_mixed_error(where + ": k rotation width must be 0 iff rotation is none");
    }
    if ((d.v_rot == 0) != (d.v_rot_width == 0)) {
        throw kv_mixed_error(where + ": v rotation width must be 0 iff rotation is none");
    }
    if (d.owner > owner_auxiliary) {
        throw kv_mixed_error(where + ": unknown logical owner");
    }
    if (d.n_stream != 1) {
        throw kv_mixed_error(where + ": multi-stream layouts are not supported in this version");
    }
    if (d.n_head_kv == 0 || d.head_dim_k == 0 || d.head_dim_v == 0) {
        throw kv_mixed_error(where + ": missing head geometry (n_head_kv, head_dim_k, head_dim_v)");
    }
    if (d.payload_mode != mode_full && d.payload_mode != mode_partial_overlay &&
            d.payload_mode != mode_resident_reference) {
        throw kv_mixed_error(where + ": unknown payload mode");
    }
    if (d.payload_mode == mode_resident_reference && d.payload_bytes != 0) {
        throw kv_mixed_error(where + ": resident reference cannot declare payload bytes");
    }
}

void check_flags(snapshot_flags flags, uint64_t owner_epoch, bool serializing) {
    const std::string where = serializing ? "serialize" : "parse";
    const uint32_t raw = uint32_t(flags);
    if ((raw & uint32_t(snapshot_flags::partial)) != 0 && (raw & uint32_t(snapshot_flags::body_only)) != 0) {
        throw kv_mixed_error(where + ": partial|body_only is an invalid flag combination");
    }
    if (raw & ~(uint32_t(snapshot_flags::partial) | uint32_t(snapshot_flags::body_only))) {
        throw kv_mixed_error(where + ": unknown snapshot flags");
    }
    if ((raw & uint32_t(snapshot_flags::partial)) != 0 && owner_epoch == 0) {
        throw kv_mixed_error(where + ": partial snapshots require a nonzero owner epoch");
    }
}

// descriptor wire order, 96 bytes exactly
void write_desc(std::vector<uint8_t> & out, const layer_desc & d) {
    put_u32(out, d.layer_id);
    out.push_back(uint8_t(d.kind));
    put_u16(out, d.k_type);
    put_u16(out, d.v_type);
    out.push_back(d.k_bits);
    out.push_back(d.v_bits);
    out.push_back(d.kvarn_domain);
    out.push_back(d.k_rot);
    out.push_back(d.v_rot);
    put_u32(out, d.k_rot_width);
    put_u32(out, d.v_rot_width);
    put_u16(out, d.layout);
    out.push_back(d.owner);
    out.push_back(d.v_trans);
    put_u16(out, d.tail_type);
    put_u32(out, d.token_group);
    put_u32(out, d.record_dim);
    put_u32(out, d.head_dim_k);
    put_u32(out, d.head_dim_v);
    put_u32(out, d.head_slices);
    put_u32(out, d.n_head_kv);
    put_u32(out, d.n_stream);
    put_u64(out, d.k_stride);
    put_u64(out, d.v_stride);
    out.push_back(d.payload_mode);
    put_u32(out, d.payload_cells);
    put_u64(out, d.payload_rows);
    put_u64(out, d.payload_bytes);
    put_u64(out, d.payload_checksum);
}

layer_desc read_desc(reader & r) {
    layer_desc d;
    d.layer_id      = r.read_u32();
    d.kind          = cache_kind(r.read_u8());
    d.k_type        = r.read_u16();
    d.v_type        = r.read_u16();
    d.k_bits        = r.read_u8();
    d.v_bits        = r.read_u8();
    d.kvarn_domain  = r.read_u8();
    d.k_rot         = r.read_u8();
    d.v_rot         = r.read_u8();
    d.k_rot_width   = r.read_u32();
    d.v_rot_width   = r.read_u32();
    d.layout        = r.read_u16();
    d.owner         = r.read_u8();
    d.v_trans       = r.read_u8();
    d.tail_type     = r.read_u16();
    d.token_group   = r.read_u32();
    d.record_dim    = r.read_u32();
    d.head_dim_k    = r.read_u32();
    d.head_dim_v    = r.read_u32();
    d.head_slices   = r.read_u32();
    d.n_head_kv     = r.read_u32();
    d.n_stream      = r.read_u32();
    d.k_stride      = r.read_u64();
    d.v_stride      = r.read_u64();
    d.payload_mode  = r.read_u8();
    d.payload_cells = r.read_u32();
    d.payload_rows  = r.read_u64();
    d.payload_bytes = r.read_u64();
    d.payload_checksum = r.read_u64();
    return d;
}

} // namespace

expected_layer expected_layer::from_desc(const layer_desc & d) {
    expected_layer e;
    e.layer_id      = d.layer_id;
    e.kind          = d.kind;
    e.k_type        = d.k_type;
    e.v_type        = d.v_type;
    e.k_bits        = d.k_bits;
    e.v_bits        = d.v_bits;
    e.kvarn_domain  = d.kvarn_domain;
    e.k_rot         = d.k_rot;
    e.v_rot         = d.v_rot;
    e.k_rot_width   = d.k_rot_width;
    e.v_rot_width   = d.v_rot_width;
    e.layout        = d.layout;
    e.owner         = d.owner;
    e.v_trans       = d.v_trans;
    e.tail_type     = d.tail_type;
    e.token_group   = d.token_group;
    e.record_dim    = d.record_dim;
    e.head_dim_k    = d.head_dim_k;
    e.head_dim_v    = d.head_dim_v;
    e.head_slices   = d.head_slices;
    e.n_head_kv     = d.n_head_kv;
    e.n_stream      = d.n_stream;
    e.k_stride      = d.k_stride;
    e.v_stride      = d.v_stride;
    e.payload_mode  = d.payload_mode;
    e.payload_cells = d.payload_cells;
    e.payload_rows  = d.payload_rows;
    e.payload_bytes = d.payload_bytes;
    return e;
}

bool snapshot::serialize(const std::vector<layer_blob> & layers,
                         const std::vector<cell_entry> & cells,
                         uint32_t metadata_version,
                         const uint8_t * metadata,
                         size_t metadata_size,
                         snapshot_flags flags,
                         uint64_t owner_epoch,
                         uint64_t owner_id,
                         std::vector<uint8_t> & out,
                         const parse_options & opts) {
    check_flags(flags, owner_epoch, true);
    if (layers.size() > KMS_MAX_LAYERS) {
        fault(opts, "serialize: too many layers");
    }
    const bool partial   = (uint32_t(flags & snapshot_flags::partial)) != 0;
    const bool body_only = (uint32_t(flags & snapshot_flags::body_only)) != 0;

    if (!body_only && (metadata == nullptr || metadata_size == 0)) {
        fault(opts, "serialize: full/partial snapshots require the opaque metadata blob");
    }
    if (body_only && (metadata != nullptr || metadata_size > 0)) {
        fault(opts, "serialize: body_only snapshots cannot carry a metadata blob");
    }

    std::unordered_set<uint32_t> seen;
    std::vector<layer_desc> descs;
    descs.reserve(layers.size());
    for (const layer_blob & blob : layers) {
        layer_desc d = blob.desc;
        validate_desc(d, true);
        if (!seen.insert(d.layer_id).second) {
            fault(opts, "serialize: duplicate layer id");
        }
        if (blob.data == nullptr && blob.size > 0) {
            fault(opts, "serialize: null payload with nonzero size");
        }
        if (partial) {
            // per-layer payload mode decides what travels in a partial snapshot
            if (d.payload_mode == mode_partial_overlay) {
                if (d.payload_bytes == 0) {
                    d.payload_bytes = blob.size;
                } else if (d.payload_bytes != blob.size) {
                    fault(opts, "serialize: declared overlay payload_bytes does not match the blob size");
                }
                if (d.payload_bytes == 0) {
                    fault(opts, "serialize: partial overlay cannot be empty");
                }
                d.payload_checksum = XXH64(blob.data, blob.size, 0);
            } else if (d.payload_mode == mode_resident_reference) {
                if (blob.size != 0) {
                    fault(opts, "serialize: resident reference cannot carry a payload blob");
                }
                d.payload_bytes = 0;
                d.payload_checksum = 0;
            } else {
                fault(opts, "serialize: partial snapshots require overlay or reference payload mode");
            }
        } else {
            // full/body_only: complete payloads only
            if (d.payload_mode != mode_full) {
                fault(opts, "serialize: full/body_only snapshots require mode_full payloads");
            }
            if (d.payload_bytes == 0) {
                d.payload_bytes = blob.size;
            } else if (d.payload_bytes != blob.size) {
                fault(opts, "serialize: declared payload_bytes does not match the blob size");
            }
            d.payload_checksum = blob.size == 0 ? 0 : XXH64(blob.data, blob.size, 0);
        }
        descs.push_back(d);
    }
    if (body_only && !cells.empty()) {
        fault(opts, "serialize: body_only snapshots cannot carry cell metadata");
    }
    if (cells.size() > KMS_MAX_CELLS) {
        fault(opts, "serialize: cell prefix exceeds the supported bound");
    }
    for (const cell_entry & c : cells) {
        if (c.seq_ids.size() > KMS_MAX_SEQ) {
            fault(opts, "serialize: cell has too many sequence ids");
        }
        if (c.pos == -1 && !c.seq_ids.empty()) {
            fault(opts, "serialize: empty cell carries sequence ids");
        }
    }
    const uint32_t n_used = uint32_t(cells.size());

    out.clear();
    out.reserve(KMS_HEADER_BYTES + metadata_size +
                descs.size() * (KMS_DESC_BYTES + 64) + cells.size() * 32 + 8);

    // header with placeholder crc (exactly KMS_HEADER_BYTES)
    std::vector<uint8_t> header;
    header.reserve(KMS_HEADER_BYTES);
    put_u32(header, KMS_MAGIC);
    put_u32(header, 1); // version
    put_u32(header, uint32_t(flags));
    put_u64(header, owner_epoch);
    put_u64(header, owner_id);
    put_u32(header, uint32_t(descs.size()));
    put_u32(header, n_used);
    put_u32(header, metadata_version);
    put_u64(header, body_only ? 0 : metadata_size);
    put_u64(header, body_only || metadata_size == 0 ? 0 : XXH64(metadata, metadata_size, 0));
    for (int i = 0; i < 8; ++i) {
        put_u32(header, 0); // reserved[8]
    }
    for (int i = 0; i < 8; ++i) {
        header.push_back(0); // header_crc placeholder
    }
    const uint64_t header_crc = XXH64(header.data(), header.size(), 0);
    for (int i = 0; i < 8; ++i) {
        header[88 + i] = uint8_t((header_crc >> (8 * i)) & 0xFF);
    }
    out.insert(out.end(), header.begin(), header.end());

    // opaque metadata blob, once
    if (!body_only && metadata_size > 0) {
        out.insert(out.end(), metadata, metadata + metadata_size);
    }

    // per-layer descriptor + inline payload (per payload_mode)
    for (size_t i = 0; i < layers.size(); ++i) {
        write_desc(out, descs[i]);
        const bool carries_bytes = descs[i].payload_mode == mode_full ||
                descs[i].payload_mode == mode_partial_overlay;
        if (carries_bytes && layers[i].size > 0) {
            out.insert(out.end(), layers[i].data, layers[i].data + layers[i].size);
        }
    }

    // shared cell summary, once (omitted for body_only)
    if (!body_only) {
        for (const cell_entry & c : cells) {
            put_u32(out, uint32_t(c.pos));
            put_u32(out, uint32_t(c.x));
            put_u32(out, uint32_t(c.y));
            put_u32(out, uint32_t(c.tok));
            put_u16(out, uint16_t(c.seq_ids.size()));
            for (int32_t s : c.seq_ids) {
                put_u32(out, uint32_t(s));
            }
        }
    }

    // whole-file checksum over everything preceding it
    const uint64_t tail_crc = XXH64(out.data(), out.size(), 0);
    put_u64(out, tail_crc);
    return true;
}

snapshot snapshot::parse(const uint8_t * data, size_t size, const parse_options & opts) {
    return snapshot::parse_impl(data, size, opts, {});
}

snapshot snapshot::parse_view(const uint8_t * data, size_t size,
                              std::shared_ptr<const void> shared_lifetime_owner,
                              const parse_options & opts) {
    if (!shared_lifetime_owner) {
        fault(opts, "parse_view: a non-null shared lifetime owner is required");
    }
    return snapshot::parse_impl(data, size, opts, std::move(shared_lifetime_owner));
}

snapshot snapshot::parse_impl(const uint8_t * data, size_t size,
                              const parse_options & opts,
                              std::shared_ptr<const void> backing) {
    const bool borrowed = static_cast<bool>(backing);
    if (data == nullptr || size < KMS_HEADER_BYTES) {
        fault(opts, "parse: truncated header");
    }
    if (opts.max_total_bytes != 0 && size > opts.max_total_bytes) {
        fault(opts, "parse: snapshot exceeds the caller total byte cap");
    }
    // header_crc covers the header with the crc field itself zeroed
    std::vector<uint8_t> header_bytes(data, data + KMS_HEADER_BYTES);
    std::memset(header_bytes.data() + 88, 0, 8);
    const uint64_t header_crc = XXH64(header_bytes.data(), KMS_HEADER_BYTES, 0);
    reader r(data, size, &opts);
    const uint32_t magic = r.read_u32();
    const uint32_t version = r.read_u32();
    const uint32_t raw_flags = r.read_u32();
    const uint64_t owner_epoch = r.read_u64();
    const uint64_t owner_id = r.read_u64();
    const uint32_t n_layers = r.read_u32();
    const uint32_t n_used = r.read_u32();
    const uint32_t metadata_version = r.read_u32();
    const uint64_t metadata_bytes = r.read_u64();
    const uint64_t metadata_checksum = r.read_u64();
    uint32_t reserved[8];
    for (int i = 0; i < 8; ++i) {
        reserved[i] = r.read_u32();
    }
    const uint64_t stored_header_crc = r.read_u64();
    if (stored_header_crc != header_crc) {
        fault(opts, "parse: header checksum mismatch");
    }
    if (magic != KMS_MAGIC) {
        fault(opts, "parse: bad magic");
    }
    if (version != 1) {
        fault(opts, "parse: unsupported snapshot version");
    }
    for (uint32_t v : reserved) {
        if (v != 0) {
            fault(opts, "parse: nonzero reserved header field");
        }
    }
    check_flags(snapshot_flags(raw_flags), owner_epoch, false);
    if (n_layers > KMS_MAX_LAYERS) {
        fault(opts, "parse: layer count exceeds the supported bound");
    }
    if (n_used > KMS_MAX_CELLS) {
        fault(opts, "parse: cell prefix exceeds the supported bound");
    }
    const bool body_only = (raw_flags & uint32_t(snapshot_flags::body_only)) != 0;
    if (body_only && n_used != 0) {
        fault(opts, "parse: body_only snapshot carries cell metadata");
    }
    if (body_only && metadata_bytes != 0) {
        fault(opts, "parse: body_only snapshot carries a metadata blob");
    }

    snapshot s;
    s.version_ = version;
    s.flags_ = snapshot_flags(raw_flags);
    s.owner_epoch_ = owner_epoch;
    s.owner_id_ = owner_id;
    s.n_used_ = n_used;
    s.metadata_version_ = metadata_version;
    s.borrowed_ = borrowed;
    s.backing_ = std::move(backing);
    s.layers_.reserve(n_layers);
    s.payloads_.reserve(n_layers);
    s.payload_owned_.reserve(n_layers);

    // metadata blob: cap and bound checks BEFORE any allocation
    if (!body_only && metadata_bytes > 0) {
        if (opts.max_metadata_bytes != 0 && metadata_bytes > opts.max_metadata_bytes) {
            fault(opts, "parse: metadata blob exceeds the caller cap");
        }
        if (metadata_bytes > r.size - r.pos) {
            fault(opts, "parse: metadata length exceeds the remaining input");
        }
        s.metadata_.resize(size_t(metadata_bytes));
        r.read(s.metadata_.data(), s.metadata_.size());
        if (XXH64(s.metadata_.data(), s.metadata_.size(), 0) != metadata_checksum) {
            fault(opts, "parse: metadata checksum mismatch");
        }
    }

    std::unordered_set<uint32_t> seen;
    for (uint32_t i = 0; i < n_layers; ++i) {
        if (r.size - r.pos < KMS_DESC_BYTES) {
            fault(opts, "parse: truncated layer descriptor");
        }
        layer_desc d = read_desc(r);
        validate_desc(d, false);
        const bool partial = (raw_flags & uint32_t(snapshot_flags::partial)) != 0;
        if (partial) {
            if (d.payload_mode == mode_full) {
                fault(opts, "parse: full payload mode in a partial snapshot");
            }
            if (d.payload_mode == mode_partial_overlay && d.payload_bytes == 0) {
                fault(opts, "parse: partial overlay with zero payload bytes");
            }
        } else if (d.payload_mode != mode_full) {
            fault(opts, "parse: non-full payload mode in a full/body_only snapshot");
        }
        if (!seen.insert(d.layer_id).second) {
            fault(opts, "parse: duplicate layer id");
        }
        if (opts.max_payload_bytes != 0 && d.payload_bytes > opts.max_payload_bytes) {
            fault(opts, "parse: payload exceeds the caller cap");
        }
        if (d.payload_bytes > r.size - r.pos) {
            fault(opts, "parse: payload length exceeds the remaining input");
        }
        const uint8_t * payload_ptr = nullptr;
        std::vector<uint8_t> owned;
        if (d.payload_bytes > 0) {
            if (borrowed) {
                // zero-copy: payload points into the caller's immutable backing
                payload_ptr = r.data + r.pos;
                XXH64_update(&r.hash, payload_ptr, size_t(d.payload_bytes));
                r.pos += size_t(d.payload_bytes);
            } else {
                owned.resize(size_t(d.payload_bytes));
                r.read(owned.data(), owned.size());
                payload_ptr = owned.data();
            }
            if (XXH64(payload_ptr, size_t(d.payload_bytes), 0) != d.payload_checksum) {
                fault(opts, "parse: payload checksum mismatch");
            }
        }
        s.layers_.push_back(d);
        s.payload_owned_.push_back(std::move(owned));
        s.payloads_.push_back({ payload_ptr, size_t(d.payload_bytes) });
    }

    if (!body_only) {
        // bound the reservation by what the remaining input can possibly hold
        s.cells_.reserve(std::min<size_t>(size_t(n_used), (r.size - r.pos) / 18 + 1));
        for (uint32_t i = 0; i < n_used; ++i) {
            if (r.size - r.pos < 18) {
                fault(opts, "parse: truncated cell metadata");
            }
            cell_entry c;
            c.pos = int32_t(r.read_u32());
            c.x   = int32_t(r.read_u32());
            c.y   = int32_t(r.read_u32());
            c.tok = int32_t(r.read_u32());
            const uint16_t n_seq = r.read_u16();
            if (n_seq > KMS_MAX_SEQ) {
                fault(opts, "parse: cell carries too many sequence ids");
            }
            if (c.pos == -1 && n_seq != 0) {
                fault(opts, "parse: empty cell carries sequence ids");
            }
            if (r.size - r.pos < size_t(n_seq) * 4) {
                fault(opts, "parse: truncated cell sequence ids");
            }
            c.seq_ids.reserve(n_seq);
            for (uint16_t j = 0; j < n_seq; ++j) {
                c.seq_ids.push_back(int32_t(r.read_u32()));
            }
            s.cells_.push_back(std::move(c));
        }
    }

    if (r.size - r.pos < 8) {
        fault(opts, "parse: truncated tail checksum");
    }
    const uint64_t tail_crc = r.digest();
    const uint64_t stored_tail = r.read_u64();
    if (stored_tail != tail_crc) {
        fault(opts, "parse: whole-file checksum mismatch");
    }
    if (r.pos != r.size) {
        fault(opts, "parse: trailing garbage after the tail checksum");
    }
    return s;
}

void snapshot::validate_expected_snapshot(const expected_snapshot & expected, const parse_options & opts) {
    if (moved_from_) {
        fault(opts, "validate: snapshot was moved-from");
    }
    if (committed_) {
        fault(opts, "validate: snapshot already committed");
    }
    if (validated_) {
        // a second validation attempt FAILS and invalidates the ticket
        validated_ = false;
        fault(opts, "validate: snapshot already validated; ticket invalidated");
    }
    if (!expected.capacity_cells.has_value()) {
        fault(opts, "validate: destination capacity must be specified");
    }
    const bool partial = (uint32_t(flags_ & snapshot_flags::partial)) != 0;
    const bool body_only = (uint32_t(flags_ & snapshot_flags::body_only)) != 0;
    if (!body_only) {
        if (!expected.metadata_version.has_value()) {
            fault(opts, "validate: expected metadata schema version must be specified");
        }
        if (metadata_version_ != *expected.metadata_version) {
            fault(opts, "validate: metadata schema version mismatch; blob must not reach commit");
        }
    }
    if (expected.layers.size() != layers_.size()) {
        fault(opts, "validate: layer count does not match the expected descriptor set");
    }
    std::unordered_set<uint32_t> expected_ids;
    for (const expected_layer & e : expected.layers) {
        if (!expected_ids.insert(e.layer_id).second) {
            fault(opts, "validate: duplicate expected layer id");
        }
    }
    for (const layer_desc & actual : layers_) {
        const expected_layer * found = nullptr;
        for (const expected_layer & e : expected.layers) {
            if (e.layer_id == actual.layer_id) {
                found = &e;
                break;
            }
        }
        if (found == nullptr) {
            fault(opts, "validate: unexpected layer id in snapshot");
        }
        const expected_layer & e = *found;
        if (e.kind != actual.kind || e.k_type != actual.k_type || e.v_type != actual.v_type ||
                e.k_bits != actual.k_bits || e.v_bits != actual.v_bits ||
                e.kvarn_domain != actual.kvarn_domain ||
                e.k_rot != actual.k_rot || e.v_rot != actual.v_rot ||
                e.k_rot_width != actual.k_rot_width || e.v_rot_width != actual.v_rot_width ||
                e.layout != actual.layout || e.owner != actual.owner ||
                e.v_trans != actual.v_trans || e.tail_type != actual.tail_type ||
                e.token_group != actual.token_group || e.record_dim != actual.record_dim ||
                e.head_dim_k != actual.head_dim_k || e.head_dim_v != actual.head_dim_v ||
                e.head_slices != actual.head_slices || e.n_head_kv != actual.n_head_kv ||
                e.n_stream != actual.n_stream || e.k_stride != actual.k_stride ||
                e.v_stride != actual.v_stride || e.payload_mode != actual.payload_mode ||
                e.payload_cells != actual.payload_cells ||
                e.payload_rows != actual.payload_rows || e.payload_bytes != actual.payload_bytes) {
            fault(opts, "validate: layer descriptor mismatch");
        }
    }
    if (n_used_ > *expected.capacity_cells) {
        fault(opts, "validate: occupied cell prefix exceeds the destination capacity");
    }
    if (partial) {
        if (!expected.owner_epoch.has_value() || !expected.owner_id.has_value()) {
            fault(opts, "validate: partial snapshots require an exact owner epoch/id");
        }
        if (owner_epoch_ != *expected.owner_epoch || owner_id_ != *expected.owner_id) {
            fault(opts, "validate: partial snapshot owner/epoch mismatch; not replayable here");
        }
    } else {
        if (expected.owner_epoch.has_value() && owner_epoch_ != *expected.owner_epoch) {
            fault(opts, "validate: snapshot owner epoch mismatch");
        }
        if (expected.owner_id.has_value() && owner_id_ != *expected.owner_id) {
            fault(opts, "validate: snapshot owner id mismatch");
        }
    }
    validated_ = true;
}

void snapshot::commit(const commit_handlers & handlers) {
    if (moved_from_) {
        throw kv_mixed_error("commit: snapshot was moved-from");
    }
    if (!validated_) {
        throw kv_mixed_error("commit: snapshot was not validated");
    }
    if (committed_) {
        throw kv_mixed_error("commit: snapshot already committed");
    }
    // the ticket is consumed by any attempt
    committed_ = true;
    validated_ = false;
    const bool body_only = (uint32_t(flags_ & snapshot_flags::body_only)) != 0;
    // required handlers must be present BEFORE any staging starts; a missing
    // handler drops state silently, so the attempt fails and the ticket is
    // consumed without running anything
    if (!handlers.stage_layer || !handlers.finish || !handlers.abort ||
            (!body_only && !handlers.stage_metadata)) {
        throw kv_mixed_error("commit: required handlers missing; ticket consumed");
    }
    // borrowed payload integrity recheck BEFORE any stage handler runs: the
    // caller contract keeps the backing immutable; a mutation since parse is
    // detected here (nothing staged yet, so no abort is needed)
    if (borrowed_) {
        for (size_t i = 0; i < layers_.size(); ++i) {
            const layer_desc & d = layers_[i];
            if (d.payload_bytes > 0 &&
                    XXH64(payloads_[i].data, payloads_[i].size, 0) != d.payload_checksum) {
                throw kv_mixed_error("commit: borrowed payload changed since parse; backing must be immutable");
            }
        }
    }
    auto rollback = [&handlers]() {
        handlers.abort();
    };
    try {
        if (!body_only) {
            if (!handlers.stage_metadata(metadata_version_, metadata_.data(), metadata_.size())) {
                throw kv_mixed_error("commit: metadata staging failed; destination untouched");
            }
        }
        for (size_t i = 0; i < layers_.size(); ++i) {
            const layer_desc & d = layers_[i];
            if (!handlers.stage_layer(d.layer_id, d, payloads_[i].data, payloads_[i].size)) {
                throw kv_mixed_error("commit: layer staging failed; destination untouched");
            }
        }
    } catch (...) {
        // any failure during the stage phase: destination never touched,
        // abort() rolls back whatever the handler set staged
        rollback();
        throw;
    }
    // contract: infallible swap; no rollback exists after this point
    handlers.finish();
}

snapshot::snapshot(snapshot && other) noexcept
    : version_(other.version_), flags_(other.flags_),
      owner_epoch_(other.owner_epoch_), owner_id_(other.owner_id_),
      n_used_(other.n_used_), metadata_version_(other.metadata_version_),
      metadata_(std::move(other.metadata_)), layers_(std::move(other.layers_)),
      cells_(std::move(other.cells_)), payloads_(std::move(other.payloads_)),
      payload_owned_(std::move(other.payload_owned_)), borrowed_(other.borrowed_),
      backing_(std::move(other.backing_)), validated_(other.validated_),
      committed_(other.committed_), moved_from_(other.moved_from_) {
    if (!borrowed_) {
        // owning mode: payloads_ must point into the moved owned buffers
        for (size_t i = 0; i < payloads_.size(); ++i) {
            payloads_[i].data = payload_owned_[i].empty() ? nullptr : payload_owned_[i].data();
        }
    }
    // moved-from: permanently inert, ticket invalidated, backing detached
    other.moved_from_ = true;
    other.borrowed_ = false;
    other.backing_.reset();
    other.validated_ = false;
    other.committed_ = false;
    other.payloads_.clear();
    other.payload_owned_.clear();
}

snapshot & snapshot::operator=(snapshot && other) noexcept {
    if (this != &other) {
        version_ = other.version_;
        flags_ = other.flags_;
        owner_epoch_ = other.owner_epoch_;
        owner_id_ = other.owner_id_;
        n_used_ = other.n_used_;
        metadata_version_ = other.metadata_version_;
        metadata_ = std::move(other.metadata_);
        layers_ = std::move(other.layers_);
        cells_ = std::move(other.cells_);
        payload_owned_ = std::move(other.payload_owned_);
        payloads_ = std::move(other.payloads_);
        borrowed_ = other.borrowed_;
        backing_ = std::move(other.backing_);
        validated_ = other.validated_;
        committed_ = other.committed_;
        moved_from_ = other.moved_from_;
        if (!borrowed_) {
            for (size_t i = 0; i < payloads_.size(); ++i) {
                payloads_[i].data = payload_owned_[i].empty() ? nullptr : payload_owned_[i].data();
            }
        }
        other.moved_from_ = true;
        other.borrowed_ = false;
        other.backing_.reset();
        other.validated_ = false;
        other.committed_ = false;
        other.payloads_.clear();
        other.payload_owned_.clear();
    }
    return *this;
}

static_assert(!std::is_copy_constructible<snapshot>::value, "snapshot must be move-only");
static_assert(!std::is_copy_assignable<snapshot>::value, "snapshot must be move-only");
static_assert(std::is_nothrow_move_constructible<snapshot>::value, "snapshot move ctor must be noexcept");
static_assert(std::is_nothrow_move_assignable<snapshot>::value, "snapshot move assignment must be noexcept");

} // namespace llama_kv_mixed