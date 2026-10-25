// Bounded streaming encoder and file-backing helpers for the codec wire
// format. See llama-kv-mixed-state-stream.h for the contract.

#define XXH_INLINE_ALL
#include "../vendor/hash/xxhash/xxhash.h"

#include "llama-kv-mixed-state-stream.h"
#include "llama-position-split.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <unordered_set>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace llama_kv_mixed {

namespace {

// wire constants, mirrored from llama-kv-mixed-state.cpp (the encoder must
// produce byte-identical output to snapshot::serialize; byte equality is
// enforced by the tests)
constexpr uint32_t KMS_MAGIC        = 0x31534D4B; // "KMS1"
constexpr uint32_t KMS_HEADER_BYTES = 96;
constexpr uint32_t KMS_DESC_BYTES   = 101;
constexpr uint32_t KMS_MAX_LAYERS   = 4096;
constexpr uint32_t KMS_MAX_SEQ      = 256; // LLAMA_MAX_SEQ
constexpr uint32_t KMS_MAX_CELLS    = 268435456u; // 1 << 28
constexpr uint32_t KMS2_MAGIC        = 0x32534D4B; // "KMS2"
constexpr uint32_t KMS2_HEADER_BYTES = 104;
constexpr uint32_t KMS2_CRC_OFFSET   = 96;
constexpr uint32_t KMS2_DESC_BYTES   = KMS_DESC_BYTES + 12; // + range_id/start/end
// KMS3: v1 prefix (96) + source model/rope/config (24) + identity crc (8) = 128
constexpr uint32_t KMS3_MAGIC         = 0x33534D4B; // "KMS3"
constexpr uint32_t KMS3_HEADER_BYTES  = 128;
constexpr uint32_t KMS3_IDENTITY_OFF  = KMS_HEADER_BYTES;
constexpr uint32_t KMS3_IDENTITY_CRC  = KMS_HEADER_BYTES + 24;
constexpr uint64_t KMS_MAX_TOTAL    = uint64_t(1) << 42; // 4 TiB sanity bound
constexpr size_t   KMS_MAX_TRANSFER = 8u << 20; // 8 MiB

// canonical kvarn_type wire id -> (key_bits, value_bits), ids 1..36
// (translation of the llama_kvarn_type enum order in include/llama.h)
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

// plain throw helper: on_fault is handled by the single outer exception
// boundary in stream_serialize (exactly once per failure)
void fail(const std::string & message) {
    throw kv_mixed_error(message);
}

#if !defined(__linux__)
// used by the file-backing ctor path only (its own exactly-once boundary).
// Templated on the options type: the file-backing ctor reports through
// parse_options while stream_serialize reports through stream_options, and
// both carry the same on_fault callback. A single stream_options overload
// fails to compile on every non-Linux platform, where only this ctor exists.
template <typename Opts>
void fault(const Opts & opts, const std::string & message) {
    if (opts.on_fault) {
        opts.on_fault(message);
    }
    throw kv_mixed_error(message);
}
#endif

// little-endian explicit encoding (mirrors the codec)
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

// field validation mirrored from the codec (fail closed)
void validate_desc(layer_desc & d, bool serializing) {
    const std::string where = serializing ? "serialize" : "parse";
    switch (d.kind) {
        case cache_kind::kvarn:
            if (!is_kvarn_type(d.k_type) || !is_kvarn_type(d.v_type)) {
                fail(where + ": unknown kvarn type ids");
            }
            if (d.k_type != d.v_type) {
                fail(where + ": kvarn k_type must equal v_type (joint type id)");
            }
            {
                const kvarn_bits & b = KVARn_BITS[d.k_type - 1];
                if (d.k_bits == 0 && d.v_bits == 0) {
                    d.k_bits = b.k;
                    d.v_bits = b.v;
                } else if (d.k_bits != b.k || d.v_bits != b.v) {
                    fail(where + ": kvarn bit fields disagree with the type id");
                }
            }
            if (d.layout != layout_kvarn_records_stage_tail) {
                fail(where + ": unknown kvarn layout id");
            }
            if (d.v_trans != 0) {
                fail(where + ": kvarn layers cannot set v_trans");
            }
            if (d.tail_type != 0 && !is_standard_type(d.tail_type)) {
                fail(where + ": unknown kvarn exact-tail type");
            }
            if (d.kvarn_domain > uint8_t(kvarn_domain::rotated_k_original_v)) {
                fail(where + ": unknown kvarn domain id");
            }
            if (d.k_rot != 0 || d.v_rot != 0 || d.k_rot_width != 0 || d.v_rot_width != 0) {
                fail(where + ": kvarn layers cannot carry standard rotations");
            }
            break;
        case cache_kind::standard_qx:
            if (!is_standard_type(d.k_type) || !is_standard_type(d.v_type)) {
                fail(where + ": unknown standard qx type id");
            }
            if (d.layout != layout_standard_qx_rows && d.layout != layout_standard_qx_compact_rows) {
                fail(where + ": unknown standard layout id");
            }
            if (d.tail_type != 0) {
                fail(where + ": standard caches cannot carry a kvarn tail");
            }
            if (d.v_trans > 1) {
                fail(where + ": invalid v_trans flag");
            }
            if (d.layout == layout_standard_qx_compact_rows && d.v_trans != 0) {
                fail(where + ": compact standard payload cannot transpose V");
            }
            if (d.kvarn_domain != 0) {
                fail(where + ": standard caches cannot carry a kvarn domain");
            }
            break;
        default:
            fail(where + ": unknown cache kind");
    }
    if (d.k_rot > uint8_t(rotation::hadamard) || d.v_rot > uint8_t(rotation::hadamard)) {
        fail(where + ": unknown rotation id");
    }
    if ((d.k_rot == 0) != (d.k_rot_width == 0)) {
        fail(where + ": rotation width must be 0 iff rotation is none");
    }
    if ((d.v_rot == 0) != (d.v_rot_width == 0)) {
        fail(where + ": v rotation width must be 0 iff rotation is none");
    }
    if (d.owner > owner_auxiliary) {
        fail(where + ": unknown logical owner");
    }
    if (d.n_stream != 1) {
        fail(where + ": multi-stream layouts are not supported in this version");
    }
    if (d.n_head_kv == 0 || d.head_dim_k == 0 || d.head_dim_v == 0) {
        fail(where + ": missing head geometry (n_head_kv, head_dim_k, head_dim_v)");
    }
    if (d.payload_mode != mode_full && d.payload_mode != mode_partial_overlay &&
            d.payload_mode != mode_resident_reference) {
        fail(where + ": unknown payload mode");
    }
    if (d.payload_mode == mode_resident_reference && d.payload_bytes != 0) {
        fail(where + ": resident reference cannot declare payload bytes");
    }
}

void check_flags(snapshot_flags flags, uint64_t owner_epoch, bool serializing) {
    const std::string where = serializing ? "serialize" : "parse";
    const uint32_t raw = uint32_t(flags);
    if ((raw & uint32_t(snapshot_flags::partial)) != 0 && (raw & uint32_t(snapshot_flags::body_only)) != 0) {
        fail(where + ": partial|body_only is an invalid flag combination");
    }
    if (raw & ~(uint32_t(snapshot_flags::partial) | uint32_t(snapshot_flags::body_only))) {
        fail(where + ": unknown snapshot flags");
    }
    if ((raw & uint32_t(snapshot_flags::partial)) != 0 && owner_epoch == 0) {
        fail(where + ": partial snapshots require a nonzero owner epoch");
    }
}

// descriptor wire order, 101 bytes exactly (mirrors the codec)
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

// checked cumulative accounting: invariant total_bytes <= KMS_MAX_TOTAL at
// every step; `add` is a u64 (size_t inputs must be widened BEFORE calling)
void add_budget(uint64_t & total, uint64_t add, const char * what) {
    if (add > KMS_MAX_TOTAL - total) {
        fail(std::string("stream: ") + what + " exceeds the sanity bound");
    }
    total += add;
}

} // namespace

namespace {

// shared validation + accounting for stream_serialize and
// stream_serialized_size: fail-closed, no payload reads, no callbacks.
struct stream_prepared {
    std::vector<layer_desc> descs;
    uint64_t total_bytes = 0;
    uint32_t n_used = 0;
    size_t transfer = 0;
};

stream_prepared stream_prepare(const stream_input & input, const stream_options & opts) {
    check_flags(input.flags, input.owner_epoch, true);
    const bool partial   = (uint32_t(input.flags & snapshot_flags::partial)) != 0;
    const bool body_only = (uint32_t(input.flags & snapshot_flags::body_only)) != 0;
    // F4 v2: the same envelope rules the owning codec applies, so the two
    // encoders cannot disagree about what a valid v2 frame is.
    const bool v2 = input.split.has_value();
    if (v2) {
        if (input.split->p == 0 || input.split->c == 0 || input.split->p >= input.split->c) {
            fail("stream: invalid position_split bounds (require 0 < P < C)");
        }
        if (input.split->p % llama_position_split::kAlignTokens != 0) {
            fail("stream: position_split P must be a multiple of the alignment");
        }
    }

    if (opts.transfer_buffer_bytes == 0 || opts.transfer_buffer_bytes > KMS_MAX_TRANSFER) {
        fail("stream: transfer buffer must be within 1..8 MiB");
    }
    if (input.layers.size() > KMS_MAX_LAYERS) {
        fail("stream: too many layers");
    }
    if (!body_only && (input.metadata == nullptr || input.metadata_size == 0)) {
        fail("stream: full/partial snapshots require the opaque metadata blob");
    }
    if (body_only && (input.metadata != nullptr || input.metadata_size > 0)) {
        fail("stream: body_only snapshots cannot carry a metadata blob");
    }
    if (input.cells.size() > KMS_MAX_CELLS) {
        fail("stream: cell prefix exceeds the supported bound");
    }
    if (body_only && !input.cells.empty()) {
        // without this guard, n_used in the header would be nonzero while
        // the cells are never emitted, producing bytes the parser rejects
        fail("stream: body_only snapshots cannot carry cell metadata");
    }
    if (opts.max_metadata_bytes != 0 && uint64_t(input.metadata_size) > opts.max_metadata_bytes) {
        fail("stream: metadata blob exceeds the caller cap");
    }

    // checked cumulative accounting, u64 widened before any arithmetic so
    // 32-bit size_t cannot wrap; invariant: total <= KMS_MAX_TOTAL
    uint64_t total_bytes = 0;
        const bool v3 = !v2 && input.source.has_value();
        add_budget(total_bytes, v2 ? KMS2_HEADER_BYTES
                : (v3 ? KMS3_HEADER_BYTES : KMS_HEADER_BYTES), "header");
    add_budget(total_bytes, uint64_t(input.metadata_size), "metadata blob");
    add_budget(total_bytes, 8, "tail checksum");

    stream_prepared out;
    out.descs.reserve(input.layers.size());
    std::unordered_set<uint64_t> seen;
    for (const layer_desc & raw : input.layers) {
        layer_desc d = raw;
        validate_desc(d, true);
        if (v2) {
            // F4 v2: identity is the (layer_id, range_id) pair and each range
            // must be exactly [0,P) local (kvarn) or [P,C) overflow (standard).
            if (d.range_id > llama_position_split::kRangeOverflow) {
                fail("stream: unknown range_id");
            }
            if (d.range_end <= d.range_start || d.range_end > input.split->c) {
                fail("stream: invalid range bounds for the split capacity");
            }
            if (d.range_id == llama_position_split::kRangeLocal) {
                if (d.range_start != 0 || d.range_end != input.split->p ||
                        d.kind != cache_kind::kvarn) {
                    fail("stream: local range must be [0,P) and a kvarn cache");
                }
            } else if (d.range_start != input.split->p || d.range_end != input.split->c ||
                    d.kind != cache_kind::standard_qx) {
                fail("stream: overflow range must be [P,C) and a standard qx cache");
            }
            if (d.layout == layout_standard_qx_compact_rows &&
                    (d.payload_cells != d.payload_rows ||
                     d.payload_rows > uint64_t(input.split->c - input.split->p))) {
                fail("stream: compact Qx payload rows exceed C-P or disagree with cell count");
            }
            const uint64_t key = (uint64_t(d.layer_id) << 32) | d.range_id;
            if (!seen.insert(key).second) {
                fail("stream: duplicate (layer_id, range_id) descriptor pair");
            }
        } else {
            if (d.layout == layout_standard_qx_compact_rows) {
                fail("stream: compact Qx rows require the v2 position_split envelope");
            }
            if (d.range_id != 0 || d.range_start != 0 || d.range_end != 0) {
                fail("stream: range identity requires the v2 position_split envelope");
            }
            if (!seen.insert(d.layer_id).second) {
                fail("stream: duplicate layer id");
            }
        }
        if (partial) {
            if (d.payload_mode == mode_partial_overlay) {
                if (d.payload_bytes == 0) {
                    fail("stream: partial overlay cannot be empty");
                }
                if (opts.max_payload_bytes != 0 && d.payload_bytes > opts.max_payload_bytes) {
                    fail("stream: overlay payload exceeds the caller cap");
                }
            } else if (d.payload_mode == mode_resident_reference) {
                // no payload bytes
            } else {
                fail("stream: partial snapshots require overlay or reference payload mode");
            }
        } else {
            if (d.payload_mode != mode_full) {
                fail("stream: full/body_only snapshots require mode_full payloads");
            }
            if (opts.max_payload_bytes != 0 && d.payload_bytes > opts.max_payload_bytes) {
                fail("stream: payload exceeds the caller cap");
            }
        }
        // descriptor room must be reserved BEFORE the payload, so a
        // payload that exactly fills the bound without its descriptor
        // fails instead of pushing total past the bound
        add_budget(total_bytes, v2 ? KMS2_DESC_BYTES : KMS_DESC_BYTES, "layer descriptor");
        add_budget(total_bytes, d.payload_bytes, "payload");
        out.descs.push_back(d);
    }
    if (v2) {
        // every layer that owns a local range must also own the matching
        // overflow range, and the two must tile [0,C) with no gap or overlap
        for (const layer_desc & d : out.descs) {
            if (d.range_id != llama_position_split::kRangeLocal) {
                continue;
            }
            const layer_desc * ovf = nullptr;
            for (const layer_desc & other : out.descs) {
                if (other.layer_id == d.layer_id &&
                        other.range_id == llama_position_split::kRangeOverflow) {
                    ovf = &other;
                    break;
                }
            }
            if (ovf == nullptr) {
                fail("stream: layer " + std::to_string(d.layer_id) +
                        " has a local range but no matching overflow range");
            }
            if (d.range_end != ovf->range_start || ovf->range_end != input.split->c) {
                fail("stream: layer " + std::to_string(d.layer_id) +
                        " ranges do not tile [0,C) without gap or overlap");
            }
        }
    }
    // cells: validate sequence counts and account (incl. per-cell seq ids)
    // before summation
    uint64_t cells_bytes = 0;
    for (const cell_entry & c : input.cells) {
        if (c.seq_ids.size() > KMS_MAX_SEQ) {
            fail("stream: cell has too many sequence ids");
        }
        if (c.pos == -1 && !c.seq_ids.empty()) {
            fail("stream: empty cell carries sequence ids");
        }
        cells_bytes += 18 + uint64_t(c.seq_ids.size()) * 4;
    }
    if (v2) {
        // Mirror the vector codec's compact-cell validation. The cell summary
        // retains global IDs; only occupied [P,C) cells count as Qx rows.
        for (const auto & d : out.descs) {
            if (d.layout != layout_standard_qx_compact_rows) { continue; }
            const auto & split = *input.split;
            if (input.cells.size() > split.c) {
                fail("stream: compact Qx cell table exceeds C");
            }
            uint32_t rows = 0;
            for (size_t i = 0; i < input.cells.size(); ++i) {
                const auto & cell = input.cells[i];
                if (cell.pos >= 0) {
                    if ((i < split.p) != (uint64_t(cell.pos) < split.p) ||
                            uint64_t(cell.pos) >= split.c || cell.seq_ids.empty()) {
                        fail("stream: compact Qx cell has invalid range, position or ownership");
                    }
                    if (i >= split.p) {
                        (void) qx_payload_row_for_cell(split.p, split.c, uint32_t(i));
                        ++rows;
                    }
                }
            }
            if (d.payload_rows != rows) {
                fail("stream: compact Qx row count differs from occupied overflow cells");
            }
        }
    }
    add_budget(total_bytes, cells_bytes, "cell metadata");
    if (opts.max_total_bytes != 0 && total_bytes > opts.max_total_bytes) {
        fail("stream: total output exceeds the caller cap");
    }

    out.total_bytes = total_bytes;
    out.n_used = uint32_t(input.cells.size());
    out.transfer = opts.transfer_buffer_bytes;
    return out;
}

} // namespace

bool stream_serialize(const stream_input & input,
                      const stream_read_callbacks & reads,
                      const stream_write_callbacks & writes,
                      const stream_options & opts) {
    try {
        if (!writes.write_bytes) {
            fail("stream: write_bytes callback is required");
        }
        bool any_payload = false;
        for (const layer_desc & d : input.layers) {
            if (d.payload_bytes > 0) {
                any_payload = true;
                break;
            }
        }
        if (any_payload && !reads.read_payload) {
            fail("stream: read_payload callback is required when payloads are present");
        }
        const stream_prepared p = stream_prepare(input, opts);
        const auto & descs = p.descs;
        const uint32_t n_used = p.n_used;
        const size_t transfer = p.transfer;
        const bool body_only = (uint32_t(input.flags & snapshot_flags::body_only)) != 0;

        // --- pass 1: bounded checksum pass over every payload ---
        std::vector<uint8_t> buf(transfer);
        std::vector<uint64_t> pass1_checksums(descs.size(), 0);
        for (size_t i = 0; i < descs.size(); ++i) {
            const layer_desc & d = descs[i];
            const bool carries = d.payload_mode == mode_full || d.payload_mode == mode_partial_overlay;
            if (!carries || d.payload_bytes == 0) {
                continue;
            }
            XXH64_state_t h;
            XXH64_reset(&h, 0);
            uint64_t remaining = d.payload_bytes;
            uint64_t offset = 0;
            while (remaining > 0) {
                const size_t chunk = size_t(std::min<uint64_t>(remaining, transfer));
                const size_t got = reads.read_payload(d.layer_id, d.range_id, offset, buf.data(), chunk);
                if (got != chunk) {
                    fail("stream: producer truncated payload (layer " + std::to_string(d.layer_id) + ")");
                }
                XXH64_update(&h, buf.data(), chunk);
                offset += chunk;
                remaining -= chunk;
            }
            pass1_checksums[i] = XXH64_digest(&h);
        }

        // --- header with placeholder crc ---
        std::vector<uint8_t> header;
        const bool v2 = input.split.has_value();
        const bool v3 = !v2 && input.source.has_value();
        const uint32_t header_bytes = v2 ? KMS2_HEADER_BYTES
                : (v3 ? KMS3_HEADER_BYTES : KMS_HEADER_BYTES);
        header.reserve(header_bytes);
        put_u32(header, v2 ? KMS2_MAGIC : (v3 ? KMS3_MAGIC : KMS_MAGIC));
        put_u32(header, v2 ? 2 : (v3 ? 3 : 1)); // version
        put_u32(header, uint32_t(input.flags));
        put_u64(header, input.owner_epoch);
        put_u64(header, input.owner_id);
        put_u32(header, uint32_t(descs.size()));
        put_u32(header, n_used);
        put_u32(header, input.metadata_version);
        put_u64(header, body_only ? 0 : input.metadata_size);
        put_u64(header, body_only || input.metadata_size == 0 ? 0 : XXH64(input.metadata, input.metadata_size, 0));
        if (v2) {
            put_u32(header, input.split->p);
            put_u32(header, input.split->c);
            put_u64(header, input.split->model_id);
            put_u64(header, input.split->rope_id);
            for (int i = 0; i < 4; ++i) {
                put_u32(header, 0); // reserved[4]
            }
        } else {
            for (int i = 0; i < 8; ++i) {
                put_u32(header, 0); // reserved[8]
            }
        }
        for (int i = 0; i < 8; ++i) {
            header.push_back(0); // header_crc placeholder
        }
        const uint64_t header_crc = XXH64(header.data(), header.size(), 0);
        const size_t crc_off = v2 ? KMS2_CRC_OFFSET : 88;
        for (int i = 0; i < 8; ++i) {
            header[crc_off + i] = uint8_t((header_crc >> (8 * i)) & 0xFF);
        }
        if (v3) {
            if (input.source->model_id == 0 || input.source->rope_id == 0 ||
                    input.source->config_id == 0) {
                fail("stream: KMS3 source identity is incomplete");
            }
            const size_t identity_off = header.size();
            put_u64(header, input.source->model_id);
            put_u64(header, input.source->rope_id);
            put_u64(header, input.source->config_id);
            const uint64_t identity_crc = XXH64(header.data() + identity_off,
                    KMS3_IDENTITY_CRC - KMS3_IDENTITY_OFF, 0);
            put_u64(header, identity_crc);
        }

        // --- pass 2: write pass ---
        XXH64_state_t tail_hash;
        XXH64_reset(&tail_hash, 0);
        uint64_t written = 0;
        const auto emit = [&](const uint8_t * bytes, size_t count) {
            if (opts.max_total_bytes != 0 && written + count > opts.max_total_bytes) {
                fail("stream: total output exceeds the caller cap");
            }
            if (!writes.write_bytes(bytes, count)) {
                fail("stream: sink write failed");
            }
            XXH64_update(&tail_hash, bytes, count);
            written += count;
            return true;
        };

        // writer-side section map (diagnostic): one entry per emitted region,
        // absolute frame offsets, same schema as stream_describe().
        auto note = [&](const char * kind, const char * field, uint32_t layer_id,
                        uint32_t range_id, uint64_t count, uint64_t k_stride,
                        uint64_t v_stride, uint64_t checksum,
                        uint64_t start, uint64_t end) {
            if (!opts.on_section) {
                return;
            }
            stream_section s;
            s.start = start; s.end = end;
            s.kind = kind; s.field = field;
            s.layer_id = layer_id; s.range_id = range_id;
            s.count = count; s.k_stride = k_stride; s.v_stride = v_stride;
            s.checksum = checksum;
            opts.on_section(s);
        };

        emit(header.data(), header.size());
        note("header", "header", UINT32_MAX, UINT32_MAX, 0, 0, 0, 0, 0, written);
        if (!body_only && input.metadata_size > 0) {
            const uint64_t at = written;
            emit(input.metadata, input.metadata_size);
            note("metadata", "manifest", UINT32_MAX, UINT32_MAX, 0, 0, 0,
                 XXH64(input.metadata, input.metadata_size, 0), at, written);
        }
        std::vector<uint64_t> pass2_checksums(descs.size(), 0);
        for (size_t i = 0; i < descs.size(); ++i) {
            const layer_desc & d = descs[i];
            std::vector<uint8_t> desc_bytes;
            desc_bytes.reserve(v2 ? KMS2_DESC_BYTES : KMS_DESC_BYTES);
            layer_desc dw = d;
            dw.payload_checksum = pass1_checksums[i];
            write_desc(desc_bytes, dw);
            if (v2) {
                // F4 v2: the range identity follows the v1 descriptor bytes
                put_u32(desc_bytes, dw.range_id);
                put_u32(desc_bytes, dw.range_start);
                put_u32(desc_bytes, dw.range_end);
            }
            const uint64_t desc_at = written;
            emit(desc_bytes.data(), desc_bytes.size());
            note("descriptor", "descriptor", d.layer_id, d.range_id, d.payload_cells,
                 d.k_stride, d.v_stride, pass1_checksums[i], desc_at, written);

            const bool carries = d.payload_mode == mode_full || d.payload_mode == mode_partial_overlay;
            if (!carries || d.payload_bytes == 0) {
                continue;
            }
            const uint64_t payload_at = written;
            XXH64_state_t h;
            XXH64_reset(&h, 0);
            uint64_t remaining = d.payload_bytes;
            uint64_t offset = 0;
            while (remaining > 0) {
                const size_t chunk = size_t(std::min<uint64_t>(remaining, transfer));
                const size_t got = reads.read_payload(d.layer_id, d.range_id, offset, buf.data(), chunk);
                if (got != chunk) {
                    fail("stream: producer truncated payload on write pass (layer " +
                         std::to_string(d.layer_id) + ")");
                }
                XXH64_update(&h, buf.data(), chunk);
                emit(buf.data(), chunk);
                offset += chunk;
                remaining -= chunk;
            }
            note("payload", "payload_kv", d.layer_id, d.range_id, d.payload_cells,
                 d.k_stride, d.v_stride, pass1_checksums[i], payload_at, written);
            pass2_checksums[i] = XXH64_digest(&h);
        }
        if (!body_only) {
            const uint64_t cells_at = written;
            for (const cell_entry & c : input.cells) {
                std::vector<uint8_t> cell_bytes;
                cell_bytes.reserve(18 + size_t(c.seq_ids.size()) * 4);
                put_u32(cell_bytes, uint32_t(c.pos));
                put_u32(cell_bytes, uint32_t(c.x));
                put_u32(cell_bytes, uint32_t(c.y));
                put_u32(cell_bytes, uint32_t(c.tok));
                put_u16(cell_bytes, uint16_t(c.seq_ids.size()));
                for (int32_t s : c.seq_ids) {
                    put_u32(cell_bytes, uint32_t(s));
                }
                emit(cell_bytes.data(), cell_bytes.size());
            }
            note("cells", "cell_entry", UINT32_MAX, UINT32_MAX, input.cells.size(),
                 0, 0, 0, cells_at, written);
        }

        // producer mutation detection: pass 2 checksums must equal pass 1
        for (size_t i = 0; i < descs.size(); ++i) {
            if (pass1_checksums[i] != pass2_checksums[i]) {
                fail("stream: payload changed between passes (layer " +
                     std::to_string(descs[i].layer_id) + "); producer mutation");
            }
        }

        const uint64_t tail_crc = XXH64_digest(&tail_hash);
        std::vector<uint8_t> tail(8);
        for (int i = 0; i < 8; ++i) {
            tail[i] = uint8_t((tail_crc >> (8 * i)) & 0xFF);
        }
        const uint64_t tail_at = written;
        emit(tail.data(), tail.size());
        note("tail", "tail_crc", UINT32_MAX, UINT32_MAX, 0, 0, 0, tail_crc, tail_at, written);
        return true;
    } catch (const kv_mixed_error & e) {
        // single outer boundary: on_fault exactly once, error preserved
        if (opts.on_fault) {
            opts.on_fault(e.what());
        }
        throw;
    } catch (const std::exception & e) {
        const std::string msg = std::string("stream: callback failure: ") + e.what();
        if (opts.on_fault) {
            opts.on_fault(msg);
        }
        throw kv_mixed_error(msg);
    } catch (...) {
        const std::string msg = "stream: unknown callback failure";
        if (opts.on_fault) {
            opts.on_fault(msg);
        }
        throw kv_mixed_error(msg);
    }
}

uint64_t stream_serialized_size(const stream_input & input, const stream_options & opts) {
    try {
        // pure sizing: identical validation and accounting to stream_serialize,
        // no payload reads, no hashing, no callbacks
        return stream_prepare(input, opts).total_bytes;
    } catch (const kv_mixed_error & e) {
        if (opts.on_fault) {
            opts.on_fault(e.what());
        }
        throw;
    } catch (const std::exception & e) {
        const std::string msg = std::string("stream: ") + e.what();
        if (opts.on_fault) {
            opts.on_fault(msg);
        }
        throw kv_mixed_error(msg);
    } catch (...) {
        const std::string msg = "stream: unknown sizing failure";
        if (opts.on_fault) {
            opts.on_fault(msg);
        }
        throw kv_mixed_error(msg);
    }
}

// ---------------------------------------------------------------------------
// Reader-side section map: walk the REAL frame bytes.

namespace {

struct byte_view {
    const uint8_t * p = nullptr;
    size_t n = 0;

    void need(size_t off, size_t len, const char * what) const {
        if (p == nullptr || off > n || len > n - off) {
            throw kv_mixed_error(std::string("describe: truncated ") + what);
        }
    }
    uint16_t u16(size_t off) const {
        need(off, 2, "u16");
        return uint16_t(uint16_t(p[off]) | (uint16_t(p[off + 1]) << 8));
    }
    uint32_t u32(size_t off) const {
        need(off, 4, "u32");
        return uint32_t(p[off]) | (uint32_t(p[off + 1]) << 8) |
               (uint32_t(p[off + 2]) << 16) | (uint32_t(p[off + 3]) << 24);
    }
    uint64_t u64(size_t off) const {
        return uint64_t(u32(off)) | (uint64_t(u32(off + 4)) << 32);
    }
};

// descriptor wire field offsets (mirror of write_desc / write_desc_v2)
constexpr size_t DESC_LAYER_ID   = 0;
constexpr size_t DESC_K_STRIDE   = 56;
constexpr size_t DESC_V_STRIDE   = 64;
constexpr size_t DESC_MODE       = 72;
constexpr size_t DESC_PAYLOAD_CELLS = 73;
constexpr size_t DESC_PAYLOAD_ROWS  = 77;
constexpr size_t DESC_PAYLOAD_BYTES = 85;
constexpr size_t DESC_PAYLOAD_CKSUM = 93;
constexpr size_t DESC_RANGE_ID   = 101; // v2 only

} // namespace

std::vector<stream_section> stream_describe(const uint8_t * data, size_t size) {
    std::vector<stream_section> out;
    const byte_view r{data, size};
    if (data == nullptr || size < 8) {
        throw kv_mixed_error("describe: frame smaller than the magic");
    }
    const uint32_t magic = r.u32(0);
    const bool v2 = magic == KMS2_MAGIC;
    const bool v3 = magic == KMS3_MAGIC;
    if (!v2 && !v3 && magic != KMS_MAGIC) {
        throw kv_mixed_error("describe: unknown frame magic");
    }
    const uint32_t header_bytes = v2 ? KMS2_HEADER_BYTES
            : (v3 ? KMS3_HEADER_BYTES : KMS_HEADER_BYTES);
    const uint32_t desc_bytes   = v2 ? KMS2_DESC_BYTES   : KMS_DESC_BYTES;
    r.need(0, header_bytes, "header");
    const uint32_t version = r.u32(4);
    if (version != (v2 ? 2u : (v3 ? 3u : 1u))) {
        throw kv_mixed_error("describe: version/magic mismatch");
    }
    if (v2 || v3) {
        // This walk must respect compact geometry and pure-source identity
        // exactly as the real reader does. Borrow synchronously: no payload
        // copy or escaping view; the caller keeps the input alive throughout.
        std::shared_ptr<const void> backing(data, [](const void *) {});
        (void) snapshot::parse_view(data, size, std::move(backing));
    }
    // KMS3 retains the 96-byte prefix CRC and a separate identity-block CRC.
    // The borrowed parser above checked that identity; do not hash 128 bytes
    // against its prefix CRC here.
    const size_t crc_off = v2 ? KMS2_CRC_OFFSET : 88;
    const uint32_t crc_span = v2 ? KMS2_HEADER_BYTES : KMS_HEADER_BYTES;
    std::vector<uint8_t> hdr(data, data + crc_span);
    std::memset(hdr.data() + crc_off, 0, 8);
    if (XXH64(hdr.data(), crc_span, 0) != r.u64(crc_off)) {
        throw kv_mixed_error("describe: header checksum mismatch");
    }

    const uint32_t flags     = r.u32(8);
    const uint32_t n_layers  = r.u32(28);
    const uint32_t n_used    = r.u32(32);
    const uint64_t meta_size = r.u64(40);
    const bool body_only = (flags & uint32_t(snapshot_flags::body_only)) != 0;
    if (n_layers > KMS_MAX_LAYERS) {
        throw kv_mixed_error("describe: layer count exceeds the bound");
    }

    auto push = [&](uint64_t start, uint64_t end, const char * kind, const char * field) {
        stream_section s;
        s.start = start; s.end = end; s.kind = kind; s.field = field;
        out.push_back(s);
        return &out.back();
    };

    uint64_t cur = 0;
    push(0, header_bytes, "header", "header");
    cur = header_bytes;
    if (!body_only && meta_size > 0) {
        r.need(cur, meta_size, "metadata");
        auto * s = push(cur, cur + meta_size, "metadata", "manifest");
        s->checksum = r.u64(48);
        cur += meta_size;
    }
    for (uint32_t i = 0; i < n_layers; ++i) {
        r.need(cur, desc_bytes, "layer descriptor");
        auto * d = push(cur, cur + desc_bytes, "descriptor", "descriptor");
        d->layer_id  = r.u32(cur + DESC_LAYER_ID);
        d->range_id  = v2 ? r.u32(cur + DESC_RANGE_ID) : 0;
        d->k_stride  = r.u64(cur + DESC_K_STRIDE);
        d->v_stride  = r.u64(cur + DESC_V_STRIDE);
        d->count     = r.u32(cur + DESC_PAYLOAD_CELLS);
        d->checksum  = r.u64(cur + DESC_PAYLOAD_CKSUM);
        const uint8_t mode = data[cur + DESC_MODE];
        const uint64_t payload_bytes = r.u64(cur + DESC_PAYLOAD_BYTES);
        cur += desc_bytes;
        const bool carries = mode == mode_full || mode == mode_partial_overlay;
        if (carries && payload_bytes > 0) {
            r.need(cur, payload_bytes, "payload");
            auto * p = push(cur, cur + payload_bytes, "payload", "payload_kv");
            p->layer_id  = d->layer_id;
            p->range_id  = d->range_id;
            p->count     = d->count;
            p->k_stride  = d->k_stride;
            p->v_stride  = d->v_stride;
            p->checksum  = d->checksum;
            cur += payload_bytes;
        }
    }
    if (!body_only && n_used > 0) {
        const uint64_t cells_start = cur;
        for (uint32_t i = 0; i < n_used; ++i) {
            r.need(cur, 18, "cell entry");
            const uint16_t nseq = r.u16(cur + 16);
            cur += 18;
            r.need(cur, uint64_t(nseq) * 4, "cell seq ids");
            cur += uint64_t(nseq) * 4;
        }
        auto * c = push(cells_start, cur, "cells", "cell_entry");
        c->count = n_used;
    }
    if (cur + 8 > size) {
        throw kv_mixed_error("describe: truncated tail checksum");
    }
    push(cur, cur + 8, "tail", "tail_crc");
    cur += 8;
    if (cur > size) {
        throw kv_mixed_error("describe: frame claims " + std::to_string(cur) +
                             " bytes but only " + std::to_string(size) + " are present");
    }
    if (cur < size) {
        // The sequence blob is [mixed attention frame][whatever the hybrid
        // memory appends afterwards] - for this model the DeltaNet recurrent
        // state. Report it as its own region instead of pretending the frame
        // owns those bytes: an unattributed tail is exactly how a layer table
        // that only lists attention rows ends up being read as "the 32 layers".
        auto * t = push(cur, size, "trailing", "post_frame_state");
        t->count = size - cur;
    }
    return out;
}

std::string stream_section_text(const stream_section & s) {
    std::string t = "[" + std::to_string(s.start) + "," + std::to_string(s.end) + ") " +
                    s.kind + "/" + s.field;
    if (s.layer_id != UINT32_MAX) {
        t += " layer=" + std::to_string(s.layer_id);
    }
    if (s.range_id != UINT32_MAX) {
        t += " stream=" + std::to_string(s.range_id);
    }
    if (s.count != 0) {
        t += " count=" + std::to_string(s.count);
    }
    if (s.k_stride != 0 || s.v_stride != 0) {
        t += " stride=" + std::to_string(s.k_stride) + "/" + std::to_string(s.v_stride);
    }
    if (s.checksum != 0) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) s.checksum);
        t += std::string(" checksum=0x") + buf;
    }
    return t;
}

#if defined(__linux__)

kv_mixed_file_backing::kv_mixed_file_backing(const std::string & path, const parse_options & opts) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw kv_mixed_error("kv_mixed_file_backing: cannot open '" + path + "': " + std::strerror(errno));
    }
    struct stat st;
    if (::fstat(fd, &st) != 0) {
        const std::string err = std::strerror(errno);
        ::close(fd);
        throw kv_mixed_error("kv_mixed_file_backing: cannot stat '" + path + "': " + err);
    }
    if (!S_ISREG(st.st_mode)) {
        ::close(fd);
        throw kv_mixed_error("kv_mixed_file_backing: not a regular file: '" + path + "'");
    }
    // widen before any cast; reject beyond the sanity bound and beyond what a
    // size_t can represent on this host (32-bit portability)
    const uint64_t fsize = uint64_t(st.st_size);
    if (fsize > KMS_MAX_TOTAL) {
        ::close(fd);
        throw kv_mixed_error("kv_mixed_file_backing: file exceeds the 4 TiB sanity bound");
    }
    if (fsize > uint64_t(std::numeric_limits<size_t>::max())) {
        ::close(fd);
        throw kv_mixed_error("kv_mixed_file_backing: file exceeds the host size_t range");
    }
    if (fsize < KMS_HEADER_BYTES) {
        ::close(fd);
        throw kv_mixed_error("kv_mixed_file_backing: file is smaller than the snapshot header");
    }
    if (opts.max_total_bytes != 0 && fsize > opts.max_total_bytes) {
        ::close(fd);
        throw kv_mixed_error("kv_mixed_file_backing: file exceeds the caller size cap");
    }
    void * mapped = ::mmap(nullptr, size_t(fsize), PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        const std::string err = std::strerror(errno);
        ::close(fd);
        throw kv_mixed_error("kv_mixed_file_backing: mmap failed: " + err);
    }
    ::close(fd); // MAP_PRIVATE keeps the pages; the fd is no longer needed
    data_ = static_cast<uint8_t *>(mapped);
    size_ = size_t(fsize);
}

kv_mixed_file_backing::~kv_mixed_file_backing() {
    if (data_ != nullptr) {
        ::munmap(data_, size_);
        data_ = nullptr;
        size_ = 0;
    }
}

kv_mixed_file_backing::kv_mixed_file_backing(kv_mixed_file_backing && other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

kv_mixed_file_backing & kv_mixed_file_backing::operator=(kv_mixed_file_backing && other) noexcept {
    if (this != &other) {
        if (data_ != nullptr) {
            ::munmap(data_, size_);
        }
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

#else // !defined(__linux__)

kv_mixed_file_backing::kv_mixed_file_backing(const std::string &, const parse_options & opts) {
    fault(opts, "kv_mixed_file_backing: unsupported on this platform");
}

#endif // defined(__linux__)

} // namespace llama_kv_mixed
