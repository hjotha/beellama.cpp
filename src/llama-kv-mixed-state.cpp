// Versioned, self-contained codec for mixed KVarN/standard-Qx KV-cache
// snapshots. See llama-kv-mixed-state.h for the wire format and contract.

#define XXH_INLINE_ALL
#include "../vendor/hash/xxhash/xxhash.h"

#include "llama-kv-mixed-state.h"
#include "llama-position-split.h"
#include "llama-impl.h"

#include <cstdarg>
#include <cstdio>

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace llama_kv_mixed {

uint32_t qx_payload_row_for_cell(uint32_t origin, uint32_t capacity, uint32_t global_cell) {
    if (origin >= capacity || global_cell < origin || global_cell >= capacity) {
        throw kv_mixed_error("Qx payload cell is outside its owned logical interval");
    }
    return global_cell - origin;
}

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

// Local variadic formatter. This TU is also compiled standalone by
// test-kv-mixed-io (ggml-only link), so it cannot depend on llama-impl.cpp's
// auto_format(); mirrors its two-pass vsnprintf semantics.
static std::string auto_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    const int size = vsnprintf(nullptr, 0, fmt, ap);
    if (size < 0) {
        va_end(ap2);
        va_end(ap);
        return std::string(fmt);
    }
    std::vector<char> buf((size_t) size + 1);
    const int size2 = vsnprintf(buf.data(), (size_t) size + 1, fmt, ap2);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size2 < 0 ? 0 : (size_t) size2);
}

// F4 v2 wire constants. The v2 header keeps the whole v1 prefix and appends the
// position_split block, so a v1 reader trips on the magic/version long before
// it can misinterpret a split descriptor as a single-range one.
constexpr uint32_t KMS2_MAGIC        = 0x32534D4B; // "KMS2"
constexpr uint32_t KMS2_VERSION      = 2;
// 96 (v1 prefix) + 4 split_p + 4 split_c + 8 model_id + 8 rope_id
// + 16 reserved[4] + 8 header_crc = 104
constexpr uint32_t KMS2_HEADER_BYTES = 104;
constexpr uint32_t KMS2_DESC_BYTES   = KMS_DESC_BYTES + 12; // + range_id/start/end
constexpr uint32_t KMS2_CRC_OFFSET   = 96;
static_assert(KMS2_HEADER_BYTES == KMS2_CRC_OFFSET + 8, "v2 crc must close the header");
static_assert(KMS2_DESC_BYTES == 113, "v2 descriptor size must match write_desc_v2/read_desc_v2");

// F4 KMS3: PURE (non-split) frame that carries the ORIGINAL WRITER identity.
// It reuses the whole v1 prefix verbatim (so an old reader still trips on the
// magic, and the v1 header checksum keeps covering the prefix) and appends the
// identity block plus its own checksum. Integrity only -- not authenticity.
constexpr uint32_t KMS3_MAGIC         = 0x33534D4B; // "KMS3"
constexpr uint32_t KMS3_VERSION       = 3;
// 96 (v1 prefix) + 8 model_id + 8 rope_id + 8 config_id + 8 identity_crc = 128
constexpr uint32_t KMS3_HEADER_BYTES  = 128;
constexpr uint32_t KMS3_PREFIX_BYTES  = KMS_HEADER_BYTES; // 96, same as v1
constexpr uint32_t KMS3_IDENTITY_OFF  = KMS_HEADER_BYTES;  // 96
constexpr uint32_t KMS3_IDENTITY_CRC  = KMS_HEADER_BYTES + 24; // 120
static_assert(KMS3_HEADER_BYTES == KMS3_IDENTITY_CRC + 8, "v3 identity crc closes the header");

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
            if (d.layout != layout_standard_qx_rows && d.layout != layout_standard_qx_compact_rows) {
                throw kv_mixed_error(where + ": unknown standard layout id");
            }
            if (d.tail_type != 0) {
                throw kv_mixed_error(where + ": standard caches cannot carry a kvarn tail");
            }
            if (d.v_trans > 1) {
                throw kv_mixed_error(where + ": invalid v_trans flag");
            }
            if (d.layout == layout_standard_qx_compact_rows && d.v_trans != 0) {
                throw kv_mixed_error(where + ": compact standard payload cannot transpose V");
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

// F4 v2: a v1 frame cannot carry range identity it would have no way to
// validate, so reject it at write time instead of silently dropping the fields.
void validate_no_range_v1(const layer_desc & d, const char * where) {
    if (d.layout == layout_standard_qx_compact_rows) {
        throw kv_mixed_error(std::string(where) + ": compact Qx rows require the v2 position_split envelope");
    }
    if (d.range_id != 0 || d.range_start != 0 || d.range_end != 0) {
        throw kv_mixed_error(std::string(where) +
                ": range identity requires the v2 position_split envelope");
    }
}

// F4 v2: a v1 frame has no range block on the wire, so a v1 descriptor can
// never carry a non-default range identity. Reject it rather than ignoring it.
void validate_no_range_v2_fields(const layer_desc & d, const char * where) {
    if (d.layout == layout_standard_qx_compact_rows) {
        throw kv_mixed_error(std::string(where) + ": compact Qx rows require the v2 position_split envelope");
    }
    if (d.range_id != 0 || d.range_start != 0 || d.range_end != 0) {
        throw kv_mixed_error(std::string(where) +
                ": v1 snapshot cannot carry range identity");
    }
}

// F4 v2: per-descriptor range checks. The range_id contract is taken from
// llama_position_split so the snapshot and the runtime allocator cannot drift:
// local is [0,P) and must be the KVarN/CUDA side, overflow is [P,C) and must be
// the standard-Qx side. Fail closed on unknown range ids or inverted bounds.
void validate_range_v2(const layer_desc & d, const position_split & split,
                       const char * where) {
    if (d.range_id > llama_position_split::kRangeOverflow) {
        throw kv_mixed_error(std::string(where) + ": unknown range_id");
    }
    if (d.range_end <= d.range_start) {
        throw kv_mixed_error(std::string(where) + ": inverted or empty range bounds");
    }
    if (d.range_end > split.c) {
        throw kv_mixed_error(std::string(where) + ": range end exceeds the split capacity");
    }
    if (d.range_id == llama_position_split::kRangeLocal) {
        if (d.range_start != 0 || d.range_end != split.p) {
            throw kv_mixed_error(std::string(where) + ": local range must be exactly [0,P)");
        }
        if (d.kind != cache_kind::kvarn) {
            throw kv_mixed_error(std::string(where) + ": local range must be a kvarn cache");
        }
    } else {
        if (d.range_start != split.p || d.range_end != split.c) {
            throw kv_mixed_error(std::string(where) + ": overflow range must be exactly [P,C)");
        }
        if (d.kind != cache_kind::standard_qx) {
            throw kv_mixed_error(std::string(where) + ": overflow range must be a standard qx cache");
        }
        if (d.layout == layout_standard_qx_compact_rows &&
                (d.payload_cells != d.payload_rows || d.payload_rows > uint64_t(split.c - split.p))) {
            throw kv_mixed_error(std::string(where) + ": compact Qx payload rows exceed C-P or disagree with cell count");
        }
    }
}

// The shared table uses global cell IDs even though compact Qx owns only the
// overflow rows. Validate its declared row count against that original table.
void validate_compact_qx_cells(const std::vector<layer_desc> & descs,
        const std::vector<cell_entry> & cells, const position_split & split, const char * where) {
    for (const auto & d : descs) {
        if (d.layout != layout_standard_qx_compact_rows) { continue; }
        if (cells.size() > split.c) {
            throw kv_mixed_error(std::string(where) + ": compact Qx cell table exceeds C");
        }
        uint32_t rows = 0;
        for (size_t i = 0; i < cells.size(); ++i) {
            if (cells[i].pos >= 0) {
                if ((i < split.p) != (uint64_t(cells[i].pos) < split.p) ||
                        uint64_t(cells[i].pos) >= split.c || cells[i].seq_ids.empty()) {
                    throw kv_mixed_error(std::string(where) + ": compact Qx cell has invalid range, position or ownership");
                }
                if (i >= split.p) {
                    (void) qx_payload_row_for_cell(split.p, split.c, uint32_t(i));
                    ++rows;
                }
            }
        }
        if (d.payload_rows != rows) {
            throw kv_mixed_error(std::string(where) + ": compact Qx row count differs from occupied overflow cells");
        }
    }
}

void validate_split_envelope(const position_split & split, const char * where) {
    if (split.p == 0 || split.c == 0 || split.p >= split.c) {
        throw kv_mixed_error(std::string(where) + ": invalid position_split bounds (require 0 < P < C)");
    }
    if (split.p % llama_position_split::kAlignTokens != 0) {
        throw kv_mixed_error(std::string(where) + ": position_split P must be a multiple of the alignment");
    }
}

// F4 v2 set-level check: uniqueness is on (layer_id, range_id), and every
// layer must own BOTH ranges with no gap or overlap at the P boundary. This is
// the snapshot-side mirror of llama_position_split::validate_ranges().
void validate_range_set_v2(const std::vector<layer_desc> & descs,
                           const position_split & split, const char * where) {
    for (size_t i = 0; i < descs.size(); ++i) {
        for (size_t j = i + 1; j < descs.size(); ++j) {
            if (descs[i].layer_id == descs[j].layer_id &&
                    descs[i].range_id == descs[j].range_id) {
                throw kv_mixed_error(std::string(where) +
                        ": duplicate (layer_id, range_id) descriptor pair");
            }
        }
    }
    for (const layer_desc & d : descs) {
        if (d.range_id != llama_position_split::kRangeLocal) {
            continue;
        }
        const layer_desc * overflow = nullptr;
        for (const layer_desc & other : descs) {
            if (other.layer_id == d.layer_id &&
                    other.range_id == llama_position_split::kRangeOverflow) {
                overflow = &other;
                break;
            }
        }
        if (overflow == nullptr) {
            throw kv_mixed_error(std::string(where) +
                    ": layer " + std::to_string(d.layer_id) +
                    " has a local range but no matching overflow range");
        }
        // the two ranges must tile [0,C) exactly: no hole and no overlap at P
        if (d.range_end != overflow->range_start || d.range_start != 0 ||
                overflow->range_end != split.c) {
            throw kv_mixed_error(std::string(where) +
                    ": layer " + std::to_string(d.layer_id) +
                    " ranges do not tile [0,C) without gap or overlap");
        }
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

// F4 v2 descriptor: the v1 wire followed by the range identity. Written and
// read as one fixed 113-byte unit so the frame stays self-describing.
void write_desc_v2(std::vector<uint8_t> & out, const layer_desc & d) {
    write_desc(out, d);
    put_u32(out, d.range_id);
    put_u32(out, d.range_start);
    put_u32(out, d.range_end);
}

layer_desc read_desc_v2(reader & r) {
    layer_desc d = read_desc(r);
    d.range_id    = r.read_u32();
    d.range_start = r.read_u32();
    d.range_end   = r.read_u32();
    return d;
}

// F4 v2 header: v1 prefix, then split_p/split_c/model_id/rope_id, then the
// reserved block and the crc at the v2 offset. The crc covers the whole
// KMS2_HEADER_BYTES with its own field zeroed.
std::vector<uint8_t> build_header_v2(snapshot_flags flags, uint64_t owner_epoch,
        uint64_t owner_id, uint32_t n_layers, uint32_t n_used,
        uint32_t metadata_version, uint64_t metadata_bytes, uint64_t metadata_checksum,
        const position_split & split) {
    std::vector<uint8_t> header;
    header.reserve(KMS2_HEADER_BYTES);
    put_u32(header, KMS2_MAGIC);
    put_u32(header, KMS2_VERSION);
    put_u32(header, uint32_t(flags));
    put_u64(header, owner_epoch);
    put_u64(header, owner_id);
    put_u32(header, n_layers);
    put_u32(header, n_used);
    put_u32(header, metadata_version);
    put_u64(header, metadata_bytes);
    put_u64(header, metadata_checksum);
    put_u32(header, split.p);
    put_u32(header, split.c);
    put_u64(header, split.model_id);
    put_u64(header, split.rope_id);
    for (int i = 0; i < 4; ++i) {
        put_u32(header, 0); // reserved[4]
    }
    for (int i = 0; i < 8; ++i) {
        header.push_back(0); // header_crc placeholder
    }
    const uint64_t header_crc = XXH64(header.data(), header.size(), 0);
    for (int i = 0; i < 8; ++i) {
        header[KMS2_CRC_OFFSET + i] = uint8_t((header_crc >> (8 * i)) & 0xFF);
    }
    return header;
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
    e.range_id     = d.range_id;
    e.range_start  = d.range_start;
    e.range_end    = d.range_end;
    return e;
}

// Canonical config identity over the range-geometry-INDEPENDENT descriptor
// fields, in (layer_id, range_id) order. Written into a KMS3 frame and
// recomputed on read: the source's config is then compared against the
// destination's separately from model/RoPE identity.
uint64_t descriptor_config_id(const std::vector<layer_desc> & layers) {
    std::vector<const layer_desc *> ordered;
    ordered.reserve(layers.size());
    for (const layer_desc & d : layers) { ordered.push_back(&d); }
    std::sort(ordered.begin(), ordered.end(), [](const layer_desc * a, const layer_desc * b) {
        if (a->layer_id != b->layer_id) { return a->layer_id < b->layer_id; }
        return a->range_id < b->range_id;
    });
    XXH64_state_t h;
    XXH64_reset(&h, 0);
    const auto put = [&](const void * data, size_t n) { XXH64_update(&h, data, n); };
    for (const layer_desc * dp : ordered) {
        const layer_desc & d = *dp;
        put(&d.layer_id, sizeof(d.layer_id));
        put(&d.kind, sizeof(d.kind));
        put(&d.k_type, sizeof(d.k_type));
        put(&d.v_type, sizeof(d.v_type));
        put(&d.k_bits, sizeof(d.k_bits));
        put(&d.v_bits, sizeof(d.v_bits));
        put(&d.kvarn_domain, sizeof(d.kvarn_domain));
        put(&d.k_rot, sizeof(d.k_rot));
        put(&d.v_rot, sizeof(d.v_rot));
        put(&d.k_rot_width, sizeof(d.k_rot_width));
        put(&d.v_rot_width, sizeof(d.v_rot_width));
        put(&d.layout, sizeof(d.layout));
        put(&d.owner, sizeof(d.owner));
        put(&d.v_trans, sizeof(d.v_trans));
        put(&d.tail_type, sizeof(d.tail_type));
        put(&d.token_group, sizeof(d.token_group));
        put(&d.record_dim, sizeof(d.record_dim));
        put(&d.head_dim_k, sizeof(d.head_dim_k));
        put(&d.head_dim_v, sizeof(d.head_dim_v));
        put(&d.head_slices, sizeof(d.head_slices));
        put(&d.n_head_kv, sizeof(d.n_head_kv));
        put(&d.n_stream, sizeof(d.n_stream));
        put(&d.k_stride, sizeof(d.k_stride));
        put(&d.v_stride, sizeof(d.v_stride));
        put(&d.payload_mode, sizeof(d.payload_mode));
        put(&d.payload_cells, sizeof(d.payload_cells));
        put(&d.payload_rows, sizeof(d.payload_rows));
        put(&d.range_id, sizeof(d.range_id));
    }
    return XXH64_digest(&h);
}

// KMS3 header: the v1 prefix byte-for-byte (magic/version swapped) plus the
// writer identity block and its own checksum.
static std::vector<uint8_t> build_header_v3(snapshot_flags flags, uint64_t owner_epoch,
        uint64_t owner_id, uint32_t n_layers, uint32_t n_used,
        uint32_t metadata_version, uint64_t metadata_bytes, uint64_t metadata_checksum,
        const source_identity & source) {
    std::vector<uint8_t> header;
    header.reserve(KMS3_HEADER_BYTES);
    put_u32(header, KMS3_MAGIC);
    put_u32(header, KMS3_VERSION);
    put_u32(header, uint32_t(flags));
    put_u64(header, owner_epoch);
    put_u64(header, owner_id);
    put_u32(header, n_layers);
    put_u32(header, n_used);
    put_u32(header, metadata_version);
    put_u64(header, metadata_bytes);
    put_u64(header, metadata_checksum);
    for (int i = 0; i < 8; ++i) {
        put_u32(header, 0); // reserved[8], same as the v1 prefix
    }
    // v1 prefix checksum at offset 88 (over the first 96 bytes, field zeroed)
    const size_t prefix_crc_off = 88;
    std::vector<uint8_t> prefix = header;
    prefix.resize(KMS3_PREFIX_BYTES, 0);
    std::memset(prefix.data() + prefix_crc_off, 0, 8);
    const uint64_t prefix_crc = XXH64(prefix.data(), KMS3_PREFIX_BYTES, 0);
    for (int i = 0; i < 8; ++i) {
        header.push_back(uint8_t((prefix_crc >> (8 * i)) & 0xFF));
    }
    const size_t identity_off = header.size();
    put_u64(header, source.model_id);
    put_u64(header, source.rope_id);
    put_u64(header, source.config_id);
    const uint64_t identity_crc =
            XXH64(header.data() + identity_off, KMS3_IDENTITY_CRC - KMS3_IDENTITY_OFF, 0);
    put_u64(header, identity_crc);
    return header;
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
                         const parse_options & opts,
                         const serialize_options & ser) {
    check_flags(flags, owner_epoch, true);
    // F4 v2: validate the envelope before touching any caller data so an
    // invalid P/C never produces a frame.
    const bool v2 = ser.split.has_value();
    const bool v3 = !v2 && ser.source.has_value();
    if (v2) {
        validate_split_envelope(*ser.split, "serialize");
    }
    if (v3 && (ser.source->model_id == 0 || ser.source->rope_id == 0 ||
               ser.source->config_id == 0)) {
        fault(opts, "serialize: KMS3 source identity is incomplete");
    }
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

    std::unordered_set<uint64_t> seen;
    std::vector<layer_desc> descs;
    descs.reserve(layers.size());
    for (const layer_blob & blob : layers) {
        layer_desc d = blob.desc;
        validate_desc(d, true);
        if (v2) {
            validate_range_v2(d, *ser.split, "serialize");
            // identity is the PAIR in v2: one full layer legitimately carries
            // two descriptors and the old layer_id-only key would reject it
            // (the exact F4 save failure: duplicate layer ownership).
            const uint64_t key = (uint64_t(d.layer_id) << 32) | d.range_id;
            if (!seen.insert(key).second) {
                fault(opts, "serialize: duplicate (layer_id, range_id) descriptor pair");
            }
        } else {
            validate_no_range_v1(d, "serialize");
            if (!seen.insert(d.layer_id).second) {
                fault(opts, "serialize: duplicate layer id");
            }
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
    if (v2) {
        // set-level range contract, checked before a single byte is emitted
        validate_range_set_v2(descs, *ser.split, "serialize");
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

    if (v2) {
        validate_compact_qx_cells(descs, cells, *ser.split, "serialize");
    }

    out.clear();
    const uint32_t header_bytes = v2 ? KMS2_HEADER_BYTES
            : (v3 ? KMS3_HEADER_BYTES : KMS_HEADER_BYTES);
    out.reserve(header_bytes + metadata_size +
                descs.size() * ((v2 ? KMS2_DESC_BYTES : KMS_DESC_BYTES) + 64) + cells.size() * 32 + 8);

    if (v2) {
        const std::vector<uint8_t> header = build_header_v2(flags, owner_epoch, owner_id,
                uint32_t(descs.size()), n_used, metadata_version,
                body_only ? 0 : metadata_size,
                body_only || metadata_size == 0 ? 0 : XXH64(metadata, metadata_size, 0),
                *ser.split);
        out.insert(out.end(), header.begin(), header.end());
    } else if (v3) {
        const std::vector<uint8_t> header = build_header_v3(flags, owner_epoch, owner_id,
                uint32_t(descs.size()), n_used, metadata_version,
                body_only ? 0 : metadata_size,
                body_only || metadata_size == 0 ? 0 : XXH64(metadata, metadata_size, 0),
                *ser.source);
        out.insert(out.end(), header.begin(), header.end());
    } else {
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
    }

    // opaque metadata blob, once
    if (!body_only && metadata_size > 0) {
        out.insert(out.end(), metadata, metadata + metadata_size);
    }

    // per-layer descriptor + inline payload (per payload_mode)
    for (size_t i = 0; i < layers.size(); ++i) {
        if (v2) {
            write_desc_v2(out, descs[i]);
        } else {
            write_desc(out, descs[i]);
        }
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
    if (data == nullptr || size < 8) {
        fault(opts, "parse: truncated header");
    }
    if (opts.max_total_bytes != 0 && size > opts.max_total_bytes) {
        fault(opts, "parse: snapshot exceeds the caller total byte cap");
    }

    // Dispatch on the magic BEFORE assuming any layout. A v2 frame must never be
    // read with v1 field offsets, and an unknown magic must never be guessed.
    const uint32_t peek_magic = uint32_t(data[0]) | (uint32_t(data[1]) << 8) |
        (uint32_t(data[2]) << 16) | (uint32_t(data[3]) << 24);
    const bool v2 = (peek_magic == KMS2_MAGIC);
    const bool v3 = (peek_magic == KMS3_MAGIC);
    const uint32_t header_bytes = v2 ? KMS2_HEADER_BYTES
            : (v3 ? KMS3_HEADER_BYTES : KMS_HEADER_BYTES);
    const uint32_t desc_bytes  = v2 ? KMS2_DESC_BYTES  : KMS_DESC_BYTES;
    if (size < header_bytes) {
        fault(opts, "parse: truncated header");
    }

    // header_crc covers the header with the crc field itself zeroed. KMS3 keeps
    // the v1 prefix intact (and its own checksum) and adds a separate checksum
    // over the appended identity block.
    const uint32_t crc_span = v2 ? KMS2_HEADER_BYTES : KMS3_PREFIX_BYTES;
    std::vector<uint8_t> header_bytes_copy(data, data + crc_span);
    const size_t crc_off = v2 ? KMS2_CRC_OFFSET : 88;
    std::memset(header_bytes_copy.data() + crc_off, 0, 8);
    const uint64_t header_crc = XXH64(header_bytes_copy.data(), crc_span, 0);
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
    uint32_t n_reserved = 8;
    position_split split;
    if (v2) {
        split.p        = r.read_u32();
        split.c        = r.read_u32();
        split.model_id = r.read_u64();
        split.rope_id  = r.read_u64();
        n_reserved = 4;
    }
    for (uint32_t i = 0; i < n_reserved; ++i) {
        reserved[i] = r.read_u32();
    }
    const uint64_t stored_header_crc = r.read_u64();
    if (stored_header_crc != header_crc) {
        fault(opts, "parse: header checksum mismatch");
    }
    if (magic != KMS_MAGIC && magic != KMS2_MAGIC && magic != KMS3_MAGIC) {
        fault(opts, "parse: bad magic");
    }
    // fail closed on any unknown version; never attempt a best-effort read
    if (version != 1 && version != 2 && version != 3) {
        fault(opts, "parse: unsupported snapshot version");
    }
    const uint32_t expect_version = v2 ? 2u : (v3 ? 3u : 1u);
    if (version != expect_version) {
        fault(opts, "parse: magic/version mismatch");
    }
    source_identity source;
    bool source_present = false;
    if (v3) {
        // identity block + its own checksum (integrity, not authenticity)
        source.model_id  = r.read_u64();
        source.rope_id   = r.read_u64();
        source.config_id = r.read_u64();
        const uint64_t stored_identity_crc = r.read_u64();
        const uint64_t identity_crc = XXH64(data + KMS3_IDENTITY_OFF,
                KMS3_IDENTITY_CRC - KMS3_IDENTITY_OFF, 0);
        if (stored_identity_crc != identity_crc) {
            fault(opts, "parse: source identity checksum mismatch");
        }
        if (source.model_id == 0 || source.rope_id == 0 || source.config_id == 0) {
            fault(opts, "parse: KMS3 source identity is incomplete");
        }
        source_present = true;
    }
    for (uint32_t i = 0; i < n_reserved; ++i) {
        if (reserved[i] != 0) {
            fault(opts, "parse: nonzero reserved header field");
        }
    }
    if (v2) {
        validate_split_envelope(split, "parse");
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
    if (v2) {
        s.split_ = split;
    }
    if (source_present) {
        s.source_ = source;
        s.source_present_ = true;
    }
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

    std::unordered_set<uint64_t> seen;
    for (uint32_t i = 0; i < n_layers; ++i) {
        if (r.size - r.pos < desc_bytes) {
            fault(opts, "parse: truncated layer descriptor");
        }
        layer_desc d = v2 ? read_desc_v2(r) : read_desc(r);
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
        if (v2) {
            validate_range_v2(d, split, "parse");
            // pair-keyed identity: two ranges of one layer are legal, the same
            // pair twice is not
            const uint64_t key = (uint64_t(d.layer_id) << 32) | d.range_id;
            if (!seen.insert(key).second) {
                fault(opts, "parse: duplicate (layer_id, range_id) descriptor pair");
            }
        } else {
            validate_no_range_v2_fields(d, "parse");
            if (!seen.insert(d.layer_id).second) {
                fault(opts, "parse: duplicate layer id");
            }
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
    if (v2) {
        // set-level range contract (both ranges per layer, no gap/overlap)
        validate_range_set_v2(s.layers_, split, "parse");
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
    if (v2) {
        validate_compact_qx_cells(s.layers_, s.cells_, split, "parse");
    }
    if (source_present && source.config_id != descriptor_config_id(s.layers_)) {
        fault(opts, "parse: source config identity does not match its descriptors");
    }
    return s;
}

void snapshot::adopt_split_for_import(const position_split & want,
                                      const std::vector<layer_desc> & destination_descs,
                                      const source_identity & source) {
    if (split_.has_value()) {
        throw kv_mixed_error("pure KVarN import: snapshot already carries a position_split envelope");
    }
    if (moved_from_ || committed_ || validated_) {
        throw kv_mixed_error("pure KVarN import: snapshot is not in an importable state");
    }
    if (destination_descs.empty()) {
        throw kv_mixed_error("pure KVarN import: destination descriptor set is empty");
    }
    // SOURCE identity comes from the frame itself (KMS3, checksum-protected and
    // bound to these exact bytes). A legacy v1 frame carries none and fails
    // closed here -- it is never substituted with the destination's own ids.
    if (source.model_id == 0 || source.rope_id == 0 || source.config_id == 0) {
        throw kv_mixed_error(
                "pure KVarN import: frame carries no source identity (legacy v1 frame); "
                "conversion is unverifiable");
    }
    if (source.model_id != want.model_id) {
        throw kv_mixed_error(auto_format(
                "pure KVarN import: source model identity does not match the destination "
                "(source %llu vs destination %llu)",
                (unsigned long long) source.model_id, (unsigned long long) want.model_id));
    }
    if (source.rope_id != want.rope_id) {
        throw kv_mixed_error(auto_format(
                "pure KVarN import: source RoPE identity does not match the destination "
                "(source %llu vs destination %llu)",
                (unsigned long long) source.rope_id, (unsigned long long) want.rope_id));
    }
    if (source.config_id != descriptor_config_id(layers_)) {
        throw kv_mixed_error(auto_format(
                "pure KVarN import: source config identity does not match its descriptors "
                "(%llu vs %llu)",
                (unsigned long long) source.config_id,
                (unsigned long long) descriptor_config_id(layers_)));
    }
    const auto key_of = [](const layer_desc & d) {
        return uint64_t((uint64_t(d.layer_id) << 32) | d.range_id);
    };
    std::unordered_set<uint64_t> want_keys;
    want_keys.reserve(destination_descs.size());
    for (const layer_desc & d : destination_descs) {
        if (d.layout == layout_standard_qx_compact_rows) {
            auto copy = d;
            validate_desc(copy, false);
            validate_range_v2(copy, want, "pure KVarN import");
            if (d.payload_cells != 0 || d.payload_rows != 0 || d.payload_bytes != 0 || d.payload_checksum != 0) {
                throw kv_mixed_error("pure KVarN import: compact synthesized overflow must be empty");
            }
            validate_compact_qx_cells({d}, cells_, want, "pure KVarN import");
        }
        if (!want_keys.insert(key_of(d)).second) {
            throw kv_mixed_error("pure KVarN import: duplicate destination descriptor");
        }
    }
    std::unordered_set<uint64_t> have;
    have.reserve(layers_.size());
    // SOURCE vs DESTINATION compatibility, checked field by field BEFORE any
    // rebinding: equal payload LENGTH alone is never compatibility.
    const auto invariant_ok = [](const layer_desc & a, const layer_desc & b) {
        return a.kind == b.kind && a.k_type == b.k_type && a.v_type == b.v_type &&
               a.k_bits == b.k_bits && a.v_bits == b.v_bits &&
               a.kvarn_domain == b.kvarn_domain &&
               a.k_rot == b.k_rot && a.v_rot == b.v_rot &&
               a.k_rot_width == b.k_rot_width && a.v_rot_width == b.v_rot_width &&
               a.layout == b.layout && a.owner == b.owner && a.v_trans == b.v_trans &&
               a.tail_type == b.tail_type && a.token_group == b.token_group &&
               a.record_dim == b.record_dim &&
               a.head_dim_k == b.head_dim_k && a.head_dim_v == b.head_dim_v &&
               a.head_slices == b.head_slices && a.n_head_kv == b.n_head_kv &&
               a.n_stream == b.n_stream && a.k_stride == b.k_stride &&
               a.v_stride == b.v_stride && a.payload_mode == b.payload_mode &&
               a.payload_cells == b.payload_cells && a.payload_rows == b.payload_rows &&
               a.payload_bytes == b.payload_bytes && a.layer_id == b.layer_id &&
               a.range_id == b.range_id;
    };
    std::unordered_map<uint64_t, size_t> dest_index;
    dest_index.reserve(destination_descs.size());
    for (size_t i = 0; i < destination_descs.size(); ++i) {
        dest_index.emplace(key_of(destination_descs[i]), i);
    }
    for (size_t i = 0; i < layers_.size(); ++i) {
        const layer_desc & src = layers_[i];
        if (!have.insert(key_of(src)).second) {
            throw kv_mixed_error(auto_format(
                    "pure KVarN import: duplicate source descriptor for layer %u range %u",
                    src.layer_id, src.range_id));
        }
        const auto it = dest_index.find(key_of(src));
        if (it == dest_index.end()) {
            throw kv_mixed_error(auto_format(
                    "pure KVarN import: source layer %u range %u is not owned by the destination",
                    src.layer_id, src.range_id));
        }
        const layer_desc & dst = destination_descs[it->second];
        if (!invariant_ok(src, dst)) {
            std::string diff;
            const auto cmp = [&](const char * n, auto a, auto b) {
                if (a != b) {
                    diff += auto_format("%s:%llu!=%llu ", n,
                            (unsigned long long) a, (unsigned long long) b);
                }
            };
            cmp("kind", (unsigned) src.kind, (unsigned) dst.kind);
            cmp("k_type", src.k_type, dst.k_type);
            cmp("v_type", src.v_type, dst.v_type);
            cmp("k_bits", src.k_bits, dst.k_bits);
            cmp("v_bits", src.v_bits, dst.v_bits);
            cmp("domain", src.kvarn_domain, dst.kvarn_domain);
            cmp("k_rot", src.k_rot, dst.k_rot);
            cmp("v_rot", src.v_rot, dst.v_rot);
            cmp("k_rot_width", src.k_rot_width, dst.k_rot_width);
            cmp("v_rot_width", src.v_rot_width, dst.v_rot_width);
            cmp("layout", src.layout, dst.layout);
            cmp("owner", src.owner, dst.owner);
            cmp("v_trans", src.v_trans, dst.v_trans);
            cmp("tail_type", src.tail_type, dst.tail_type);
            cmp("token_group", src.token_group, dst.token_group);
            cmp("record_dim", src.record_dim, dst.record_dim);
            cmp("head_dim_k", src.head_dim_k, dst.head_dim_k);
            cmp("head_dim_v", src.head_dim_v, dst.head_dim_v);
            cmp("head_slices", src.head_slices, dst.head_slices);
            cmp("n_head_kv", src.n_head_kv, dst.n_head_kv);
            cmp("n_stream", src.n_stream, dst.n_stream);
            cmp("k_stride", src.k_stride, dst.k_stride);
            cmp("v_stride", src.v_stride, dst.v_stride);
            cmp("payload_mode", src.payload_mode, dst.payload_mode);
            cmp("payload_cells", src.payload_cells, dst.payload_cells);
            cmp("payload_rows", src.payload_rows, dst.payload_rows);
            cmp("payload_bytes", src.payload_bytes, dst.payload_bytes);
            throw kv_mixed_error(auto_format(
                    "pure KVarN import: layer %u range %u descriptor is incompatible with the "
                    "destination conversion contract [%s]",
                    src.layer_id, src.range_id, diff.c_str()));
        }
        // The mapped payload the codec authenticated must still be there.
        if (payloads_.size() <= i || payloads_[i].size != src.payload_bytes) {
            throw kv_mixed_error(auto_format(
                    "pure KVarN import: layer %u range %u payload bytes do not match its descriptor",
                    src.layer_id, src.range_id));
        }
        if (src.payload_bytes != 0 && payloads_[i].data == nullptr) {
            throw kv_mixed_error(auto_format(
                    "pure KVarN import: layer %u range %u payload is missing or corrupt",
                    src.layer_id, src.range_id));
        }
        // ONLY the explicitly permitted range geometry moves; everything else,
        // including the authenticated payload_checksum, is preserved verbatim.
        const uint64_t provenance = src.payload_checksum;
        layers_[i] = dst;
        layers_[i].payload_checksum = provenance;
    }
    for (const layer_desc & d : destination_descs) {
        if (have.count(key_of(d))) { continue; }
        if (d.payload_bytes != 0) {
            throw kv_mixed_error(auto_format(
                    "pure KVarN import: synthesized layer %u range %u must be empty (payload_bytes=%llu)",
                    d.layer_id, d.range_id, (unsigned long long) d.payload_bytes));
        }
        layers_.push_back(d);
        payloads_.push_back(payload_ref{});
        payload_owned_.push_back({});
    }
    split_ = want;
}

void snapshot::project_split_to_pure_for_import(
        uint32_t destination_capacity, uint64_t destination_model_id,
        uint64_t destination_rope_id, const std::vector<layer_desc> & destination_descs) {
    if (!split_ || moved_from_ || committed_ || validated_) {
        throw kv_mixed_error("split-to-pure import: snapshot is not an unvalidated split frame");
    }
    if (flags_ != snapshot_flags::full || destination_capacity == 0 || destination_descs.empty()) {
        throw kv_mixed_error("split-to-pure import: requires a full snapshot and nonempty pure destination");
    }
    const auto source_split = *split_;
    if (!llama_position_split::is_valid_p(source_split.p, source_split.c) ||
            !destination_model_id || !destination_rope_id ||
            source_split.model_id != destination_model_id || source_split.rope_id != destination_rope_id) {
        throw kv_mixed_error("split-to-pure import: source model/RoPE identity or split geometry differs");
    }
    validate_range_set_v2(layers_, source_split, "split-to-pure import");
    const uint64_t limit = std::min<uint64_t>(source_split.p, destination_capacity);
    uint64_t extent = 0;
    for (const auto & cell : cells_) {
        if (cell.pos >= 0) { extent = std::max(extent, uint64_t(cell.pos)+1u); }
    }
    if (extent > limit) {
        throw kv_mixed_error(auto_format(
                "split-to-pure import refused: snapshot extent %llu exceeds min(P=%u, capacity=%u)",
                (unsigned long long) extent, source_split.p, destination_capacity));
    }
    if (n_used_ != cells_.size() || cells_.size() > destination_capacity) {
        throw kv_mixed_error("split-to-pure import: source cell table exceeds destination capacity");
    }
    for (size_t i = 0; i < cells_.size(); ++i) {
        const auto & cell = cells_[i];
        if (cell.pos >= 0 && i >= source_split.p) {
            throw kv_mixed_error("split-to-pure import: occupied physical overflow cell despite an in-range logical extent");
        }
    }
    // Compare the full descriptor invariant set before rebinding any field.
    // Only local range geometry is allowed to disappear. Payload checksums
    // remain those authenticated by the original full split frame.
    const auto invariant_ok = [](const layer_desc & a, const layer_desc & b) {
        return a.kind == b.kind && a.k_type == b.k_type && a.v_type == b.v_type &&
            a.k_bits == b.k_bits && a.v_bits == b.v_bits && a.kvarn_domain == b.kvarn_domain &&
            a.k_rot == b.k_rot && a.v_rot == b.v_rot &&
            a.k_rot_width == b.k_rot_width && a.v_rot_width == b.v_rot_width &&
            a.layout == b.layout && a.owner == b.owner && a.v_trans == b.v_trans &&
            a.tail_type == b.tail_type && a.token_group == b.token_group && a.record_dim == b.record_dim &&
            a.head_dim_k == b.head_dim_k && a.head_dim_v == b.head_dim_v &&
            a.head_slices == b.head_slices && a.n_head_kv == b.n_head_kv && a.n_stream == b.n_stream &&
            a.k_stride == b.k_stride && a.v_stride == b.v_stride &&
            a.payload_mode == b.payload_mode && a.payload_cells == b.payload_cells &&
            a.payload_rows == b.payload_rows && a.payload_bytes == b.payload_bytes &&
            a.layer_id == b.layer_id && a.range_id == b.range_id;
    };
    std::unordered_map<uint32_t, size_t> destination_by_layer;
    for (size_t i = 0; i < destination_descs.size(); ++i) {
        const auto & d = destination_descs[i];
        if (d.kind != cache_kind::kvarn || d.range_id != llama_position_split::kRangeLocal ||
                d.range_start != 0 || d.range_end != 0 || d.payload_mode != mode_full ||
                !destination_by_layer.emplace(d.layer_id, i).second) {
            throw kv_mixed_error("split-to-pure import: destination is not a unique pure KVarN layer set");
        }
    }
    std::vector<size_t> local_indices;
    std::unordered_set<uint32_t> local_layers, overflow_layers;
    std::vector<layer_desc> projected;
    std::vector<payload_ref> projected_payloads;
    std::vector<std::vector<uint8_t>> projected_owned;
    local_indices.reserve(destination_descs.size());
    projected.reserve(destination_descs.size());
    projected_payloads.reserve(destination_descs.size());
    projected_owned.reserve(destination_descs.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto & d = layers_[i];
        if (i >= payloads_.size() || i >= payload_owned_.size() || payloads_[i].size != d.payload_bytes ||
                (d.payload_bytes && !payloads_[i].data)) {
            throw kv_mixed_error("split-to-pure import: authenticated payload extent/backing is missing");
        }
        const auto destination = destination_by_layer.find(d.layer_id);
        if (destination == destination_by_layer.end()) {
            throw kv_mixed_error("split-to-pure import: source layer is not owned by the destination");
        }
        const auto & want = destination_descs[destination->second];
        if (d.range_id == llama_position_split::kRangeLocal) {
            if (d.range_start != 0 || d.range_end != source_split.p ||
                    !local_layers.insert(d.layer_id).second || !invariant_ok(d, want)) {
                throw kv_mixed_error("split-to-pure import: local formats/domain/tail/geometry/payload differ");
            }
            auto projected_desc = want;
            projected_desc.payload_checksum = d.payload_checksum;
            projected.push_back(projected_desc);
            projected_payloads.push_back(payloads_[i]);
            local_indices.push_back(i);
        } else if (d.range_id == llama_position_split::kRangeOverflow) {
            if (!overflow_layers.insert(d.layer_id).second || d.kind != cache_kind::standard_qx ||
                    d.range_start != source_split.p || d.range_end != source_split.c ||
                    d.payload_mode != mode_full || d.payload_bytes != 0 || d.payload_checksum != 0 ||
                    d.head_dim_k != want.head_dim_k || d.head_dim_v != want.head_dim_v ||
                    d.n_head_kv != want.n_head_kv || d.n_stream != want.n_stream || d.owner != want.owner ||
                    (d.layout == layout_standard_qx_compact_rows
                        ? (d.payload_cells != 0 || d.payload_rows != 0)
                        : (d.payload_cells != want.payload_cells || d.payload_rows != want.payload_rows))) {
                throw kv_mixed_error("split-to-pure import: overflow descriptor is not an empty compatible range");
            }
        } else {
            throw kv_mixed_error("split-to-pure import: unknown source range");
        }
    }
    if (local_layers.size() != destination_descs.size() || overflow_layers != local_layers) {
        throw kv_mixed_error("split-to-pure import: source does not carry both ranges of every destination layer");
    }
    const uint64_t projected_config = descriptor_config_id(projected);
    // All checks and allocations finished. Moving owned vectors preserves
    // their payload addresses; borrowed mappings retain the original backing.
    for (size_t index : local_indices) { projected_owned.push_back(std::move(payload_owned_[index])); }
    layers_ = std::move(projected);
    payloads_ = std::move(projected_payloads);
    payload_owned_ = std::move(projected_owned);
    split_.reset();
    source_ = {source_split.model_id, source_split.rope_id, projected_config};
    source_present_ = true;
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
    // F4 v2: the position_split identity is matched EXACTLY, and it is a hard
    // requirement in both directions. A snapshot that carries an envelope must
    // be validated against a known destination P/C/model/rope; a snapshot
    // without one must not be replayed into a split cache. This runs before any
    // staging, so a P mismatch never mutates the destination.
    if (split_.has_value() != expected.split.has_value()) {
        fault(opts, "validate: position_split envelope presence does not match the destination");
    }
    if (split_.has_value() && expected.split.has_value()) {
        const position_split & a = *split_;
        const position_split & b = *expected.split;
        if (a.p != b.p || a.c != b.c) {
            fault(opts, "validate: position_split P/C does not match the destination");
        }
        if (a.model_id != b.model_id) {
            fault(opts, "validate: position_split model identity does not match the destination");
        }
        if (a.rope_id != b.rope_id) {
            fault(opts, "validate: position_split RoPE identity does not match the destination");
        }
    }
    std::unordered_set<uint64_t> expected_ids;
    for (const expected_layer & e : expected.layers) {
        // v2 identity is the pair; v1 stays layer_id-only
        const uint64_t key = split_.has_value()
                ? ((uint64_t(e.layer_id) << 32) | e.range_id)
                : uint64_t(e.layer_id);
        if (!expected_ids.insert(key).second) {
            fault(opts, "validate: duplicate expected layer id");
        }
    }
    for (const layer_desc & actual : layers_) {
        const uint64_t actual_key = split_.has_value()
                ? ((uint64_t(actual.layer_id) << 32) | actual.range_id)
                : uint64_t(actual.layer_id);
        const expected_layer * found = nullptr;
        for (const expected_layer & e : expected.layers) {
            const uint64_t key = split_.has_value()
                    ? ((uint64_t(e.layer_id) << 32) | e.range_id)
                    : uint64_t(e.layer_id);
            if (key == actual_key) {
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
                e.payload_rows != actual.payload_rows || e.payload_bytes != actual.payload_bytes ||
                e.range_id != actual.range_id || e.range_start != actual.range_start ||
                e.range_end != actual.range_end) {
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
        // F4 v2: stage the position_split unit (DeltaNet/recurrent state plus
        // the logical cell table for BOTH ranges) before any KV byte lands, so a
        // failure here aborts with the destination entirely untouched.
        if (split_.has_value()) {
            if (!handlers.stage_split || !handlers.stage_split(*split_)) {
                throw kv_mixed_error("commit: position_split staging failed; destination untouched");
            }
        }
        for (size_t i = 0; i < layers_.size(); ++i) {
            const layer_desc & d = layers_[i];
            if (!handlers.stage_layer(d.layer_id, d.range_id, d, payloads_[i].data, payloads_[i].size)) {
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
    : version_(other.version_), split_(other.split_),
      source_(other.source_), source_present_(other.source_present_),
      flags_(other.flags_),
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
    // the source identity belongs to the frame, not to a moved-from shell
    other.source_present_ = false;
    other.source_ = source_identity{};
}

snapshot & snapshot::operator=(snapshot && other) noexcept {
    if (this != &other) {
        version_ = other.version_;
        split_ = other.split_;
        source_ = other.source_;
        source_present_ = other.source_present_;
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
        other.source_present_ = false;
        other.source_ = source_identity{};
    }
    return *this;
}

static_assert(!std::is_copy_constructible<snapshot>::value, "snapshot must be move-only");
static_assert(!std::is_copy_assignable<snapshot>::value, "snapshot must be move-only");
static_assert(std::is_nothrow_move_constructible<snapshot>::value, "snapshot move ctor must be noexcept");
static_assert(std::is_nothrow_move_assignable<snapshot>::value, "snapshot move assignment must be noexcept");

} // namespace llama_kv_mixed
