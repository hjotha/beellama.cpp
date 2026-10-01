// Bounded streaming encoder and file-backing helpers for the codec wire
// format. See llama-kv-mixed-state-stream.h for the contract.

#define XXH_INLINE_ALL
#include "../vendor/hash/xxhash/xxhash.h"

#include "llama-kv-mixed-state-stream.h"

#include <algorithm>
#include <cerrno>
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
// used by the file-backing ctor path only (its own exactly-once boundary)
void fault(const stream_options & opts, const std::string & message) {
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
            if (d.layout != layout_standard_qx_rows) {
                fail(where + ": unknown standard layout id");
            }
            if (d.tail_type != 0) {
                fail(where + ": standard caches cannot carry a kvarn tail");
            }
            if (d.v_trans > 1) {
                fail(where + ": invalid v_trans flag");
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
    add_budget(total_bytes, KMS_HEADER_BYTES, "header");
    add_budget(total_bytes, uint64_t(input.metadata_size), "metadata blob");
    add_budget(total_bytes, 8, "tail checksum");

    stream_prepared out;
    out.descs.reserve(input.layers.size());
    std::unordered_set<uint32_t> seen;
    for (const layer_desc & raw : input.layers) {
        layer_desc d = raw;
        validate_desc(d, true);
        if (!seen.insert(d.layer_id).second) {
            fail("stream: duplicate layer id");
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
        add_budget(total_bytes, KMS_DESC_BYTES, "layer descriptor");
        add_budget(total_bytes, d.payload_bytes, "payload");
        out.descs.push_back(d);
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
                const size_t got = reads.read_payload(d.layer_id, offset, buf.data(), chunk);
                if (got != chunk) {
                    fail("stream: producer truncated payload (layer " + std::to_string(d.layer_id) + ")");
                }
                XXH64_update(&h, buf.data(), chunk);
                offset += chunk;
                remaining -= chunk;
            }
            pass1_checksums[i] = XXH64_digest(&h);
        }

        // --- header with placeholder crc (exactly KMS_HEADER_BYTES) ---
        std::vector<uint8_t> header;
        header.reserve(KMS_HEADER_BYTES);
        put_u32(header, KMS_MAGIC);
        put_u32(header, 1); // version
        put_u32(header, uint32_t(input.flags));
        put_u64(header, input.owner_epoch);
        put_u64(header, input.owner_id);
        put_u32(header, uint32_t(descs.size()));
        put_u32(header, n_used);
        put_u32(header, input.metadata_version);
        put_u64(header, body_only ? 0 : input.metadata_size);
        put_u64(header, body_only || input.metadata_size == 0 ? 0 : XXH64(input.metadata, input.metadata_size, 0));
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

        emit(header.data(), header.size());
        if (!body_only && input.metadata_size > 0) {
            emit(input.metadata, input.metadata_size);
        }
        std::vector<uint64_t> pass2_checksums(descs.size(), 0);
        for (size_t i = 0; i < descs.size(); ++i) {
            const layer_desc & d = descs[i];
            std::vector<uint8_t> desc_bytes;
            desc_bytes.reserve(KMS_DESC_BYTES);
            layer_desc dw = d;
            dw.payload_checksum = pass1_checksums[i];
            write_desc(desc_bytes, dw);
            emit(desc_bytes.data(), desc_bytes.size());

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
                const size_t got = reads.read_payload(d.layer_id, offset, buf.data(), chunk);
                if (got != chunk) {
                    fail("stream: producer truncated payload on write pass (layer " +
                         std::to_string(d.layer_id) + ")");
                }
                XXH64_update(&h, buf.data(), chunk);
                emit(buf.data(), chunk);
                offset += chunk;
                remaining -= chunk;
            }
            pass2_checksums[i] = XXH64_digest(&h);
        }
        if (!body_only) {
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
        emit(tail.data(), tail.size());
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