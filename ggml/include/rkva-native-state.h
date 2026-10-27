#pragma once

// Canonical host-side state for the native KVarN4 overflow producer.
//
// This module is deliberately independent of ggml, D3D12 and the llama state
// reader. It serializes only authoritative values: normalized completed KVarN4
// records (relative to the range origin group), the rotated F16 stage, the
// logical original-F16 precision tail, the original-F32 rollback ring and its
// monotonic positions. Transient preparation buffers and live GPU handles are
// never part of the format.
//
// All integers are little-endian. Every multibyte value is stored in IEEE bit
// form. The set is validated in full (header, derived sizes, ring invariants,
// SHA-256 footer) before a caller may install it into a producer.

#include "rkva-snapshot-digest.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace rkva_native_state {

constexpr uint32_t kMagic = 0x4b564b52; // "RKVK" little-endian
constexpr uint16_t kVersion = 1;
constexpr uint16_t kHeaderBytes = 80;
constexpr uint32_t kMaxLayers = 256;
constexpr uint32_t kStageRows = 384;
constexpr uint32_t kTailRingRows = 256;
constexpr uint32_t kOriginalRows = 256;
constexpr uint32_t kMaxTailTokens = 128;
constexpr uint32_t kMaxHeads = 4096;
constexpr uint64_t kMaxPayloadBytes = 4ull << 30;
constexpr uint64_t kMaxLayerBytes = 1ull << 30;

inline bool little_endian() {
    const uint16_t probe = 1;
    return *reinterpret_cast<const uint8_t*>(&probe) == 1;
}

inline void put_u16(uint8_t* dst, uint16_t value) {
    dst[0] = uint8_t(value);
    dst[1] = uint8_t(value >> 8);
}
inline void put_u32(uint8_t* dst, uint32_t value) {
    for (int i = 0; i < 4; ++i)
        dst[i] = uint8_t(value >> (8 * i));
}
inline void put_u64(uint8_t* dst, uint64_t value) {
    for (int i = 0; i < 8; ++i)
        dst[i] = uint8_t(value >> (8 * i));
}
inline uint16_t get_u16(const uint8_t* src) {
    return uint16_t(src[0]) | uint16_t(uint16_t(src[1]) << 8);
}
inline uint32_t get_u32(const uint8_t* src) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i)
        value |= uint32_t(src[i]) << (8 * i);
    return value;
}
inline uint64_t get_u64(const uint8_t* src) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i)
        value |= uint64_t(src[i]) << (8 * i);
    return value;
}

class Sink {
  public:
    virtual ~Sink() = default;
    virtual bool write(const void* data, size_t bytes) = 0;
};

class Source {
  public:
    virtual ~Source() = default;
    virtual bool read(void* data, size_t bytes) = 0;
};

class VectorSink final : public Sink {
  public:
    std::vector<uint8_t> bytes;
    bool write(const void* data, size_t size) override {
        try {
            const auto* begin = static_cast<const uint8_t*>(data);
            bytes.insert(bytes.end(), begin, begin + size);
            return true;
        } catch (...) {
            return false;
        }
    }
};

class VectorSource final : public Source {
  public:
    explicit VectorSource(const std::vector<uint8_t>& data) : data_(data) {}
    bool read(void* dst, size_t size) override {
        if (size > data_.size() - offset_)
            return false;
        std::memcpy(dst, data_.data() + offset_, size);
        offset_ += size;
        return true;
    }
    size_t remaining() const { return data_.size() - offset_; }

  private:
    const std::vector<uint8_t>& data_;
    size_t offset_ = 0;
};

struct Layout {
    uint32_t origin = 0;
    uint32_t capacity = 0;
    uint32_t heads = 0;
    uint32_t iterations = 0;
    uint32_t precision_tail_tokens = 0;
    uint32_t compact_stage = 0;
    uint32_t rows = 0;
    uint32_t record_base_group = 0;
    uint64_t saved_generation = 0;
};

struct View {
    Layout layout;
    const uint8_t* records_k = nullptr;
    const uint8_t* records_v = nullptr;
    const uint8_t* stage_k = nullptr;
    const uint8_t* stage_v = nullptr;
    const uint8_t* tail_k = nullptr;
    const uint8_t* tail_v = nullptr;
    const uint8_t* original_k = nullptr;
    const uint8_t* original_v = nullptr;
    const uint32_t* original_pos = nullptr;
};

struct Storage {
    Layout layout;
    // Completed groups are normalized to origin/128, independent of the
    // source producer's compact/global addressing.
    std::vector<uint8_t> records_k;
    std::vector<uint8_t> records_v;
    std::vector<uint16_t> stage_k;
    std::vector<uint16_t> stage_v;
    std::vector<uint16_t> tail_k;
    std::vector<uint16_t> tail_v;
    std::vector<float> original_k;
    std::vector<float> original_v;
    std::vector<uint32_t> original_pos;
    bool valid = false;
};

inline uint64_t width(const Layout& layout) {
    return uint64_t(layout.heads) * 256ull;
}
inline uint64_t record_group_bytes(const Layout& layout) {
    return uint64_t(layout.heads) * 2ull * 8960ull;
}
inline uint32_t complete_groups(const Layout& layout) {
    return layout.rows / 128;
}
inline uint64_t records_bytes(const Layout& layout) {
    return uint64_t(complete_groups(layout)) * record_group_bytes(layout);
}
inline uint64_t stage_bytes(const Layout& layout) {
    return uint64_t(kStageRows) * width(layout) * sizeof(uint16_t);
}
inline uint64_t tail_bytes(const Layout& layout) {
    return uint64_t(layout.precision_tail_tokens) * width(layout) * sizeof(uint16_t);
}
inline uint64_t original_bytes(const Layout& layout) {
    return uint64_t(kOriginalRows) * width(layout) * sizeof(float);
}
inline uint64_t positions_bytes() {
    return uint64_t(kOriginalRows) * sizeof(uint32_t);
}
inline uint64_t layer_bytes(const Layout& layout) {
    return 2 * (records_bytes(layout) + stage_bytes(layout) + tail_bytes(layout) +
                original_bytes(layout)) +
           positions_bytes();
}
inline bool generation_supported(uint64_t generation) {
    return generation != 0 && uint32_t(generation) != UINT32_MAX;
}
inline uint64_t payload_bytes(const Layout& layout, uint32_t layers) {
    return uint64_t(layers) * layer_bytes(layout);
}

inline bool valid_layout(const Layout& layout, std::string* error) {
    const auto reject = [&](const char* message) {
        if (error)
            *error = message;
        return false;
    };
    if (!little_endian())
        return reject("native state: big-endian hosts are unsupported");
    if (!layout.origin || !layout.capacity || !layout.heads ||
        !layout.iterations || layout.origin % 128 ||
        uint64_t(layout.origin) + layout.capacity > uint32_t(INT32_MAX) ||
        layout.heads > kMaxHeads ||
        (layout.precision_tail_tokens != 0 && layout.precision_tail_tokens != kMaxTailTokens) ||
        layout.compact_stage > 1 || layout.rows > layout.capacity ||
        (layout.record_base_group != 0 && layout.record_base_group != layout.origin / 128) ||
        !generation_supported(layout.saved_generation))
        return reject("native state: invalid layout");
    const uint64_t layer = layer_bytes(layout);
    if (!layer || layer > kMaxLayerBytes || layer > kMaxPayloadBytes)
        return reject("native state: payload exceeds bounded store");
    return true;
}

// The aggregate budget depends on the real layer count; using kMaxLayers here
// would reject valid 16-layer production snapshots. Division avoids overflow.
inline bool valid_payload_size(const Layout& layout, uint32_t layers) {
    if (!layers || layers > kMaxLayers)
        return false;
    const uint64_t layer = layer_bytes(layout);
    return layer != 0 && layer <= kMaxLayerBytes && layer <= kMaxPayloadBytes / layers;
}

inline bool valid_ring(const Storage& storage, std::string* error) {
    const auto reject = [&](const char* message) {
        if (error)
            *error = message;
        return false;
    };
    const Layout& layout = storage.layout;
    if (storage.original_pos.size() != kOriginalRows ||
        storage.original_k.size() * sizeof(float) != original_bytes(layout) ||
        storage.original_v.size() * sizeof(float) != original_bytes(layout))
        return reject("native state: original rollback ring size mismatch");
    // Every non-sentinel slot must still obey the ring's slot invariant. A
    // post-rewind ring may retain older/future stale values; only the contiguous
    // newest window is authoritative.
    for (uint32_t slot = 0; slot < kOriginalRows; ++slot) {
        const uint32_t position = storage.original_pos[slot];
        if (position == UINT32_MAX) {
            if (layout.rows >= kOriginalRows)
                return reject("native state: missing original rollback row");
            continue;
        }
        if (position % kOriginalRows != slot)
            return reject("native state: original rollback position mismatch");
    }
    // Walk backwards from the newest row. A rewind can only consume this
    // contiguous window; at least the last 128 rows are required for the
    // advertised <=128-row rollback bound.
    uint32_t available = layout.rows;
    while (available > 0) {
        const uint32_t position = available - 1;
        if (storage.original_pos[position % kOriginalRows] != position)
            break;
        --available;
    }
    const uint32_t needed = layout.rows > kMaxTailTokens ? layout.rows - kMaxTailTokens : 0;
    if (available > needed)
        return reject("native state: original rollback ring is incomplete");
    return true;
}

struct SetHeader {
    uint32_t layers = 0;
    uint32_t flags = 0;
    Layout layout;
    uint64_t payload_bytes = 0;
    uint64_t layer_bytes = 0;
};

inline bool encode_header(const SetHeader& header, uint8_t out[kHeaderBytes], std::string* error) {
    if (!valid_layout(header.layout, error) || !valid_payload_size(header.layout, header.layers) ||
        header.flags || header.payload_bytes != payload_bytes(header.layout, header.layers) ||
        header.layer_bytes != layer_bytes(header.layout))
        return false;
    std::memset(out, 0, kHeaderBytes);
    put_u32(out + 0, kMagic);
    put_u16(out + 4, kVersion);
    put_u16(out + 6, kHeaderBytes);
    put_u32(out + 8, header.layers);
    put_u32(out + 12, header.flags);
    put_u32(out + 16, header.layout.origin);
    put_u32(out + 20, header.layout.capacity);
    put_u32(out + 24, header.layout.heads);
    put_u32(out + 28, header.layout.iterations);
    put_u32(out + 32, header.layout.precision_tail_tokens);
    put_u32(out + 36, header.layout.rows);
    put_u32(out + 40, header.layout.record_base_group);
    put_u32(out + 44, header.layout.compact_stage);
    put_u64(out + 48, header.layout.saved_generation);
    put_u64(out + 56, header.payload_bytes);
    put_u64(out + 64, header.layer_bytes);
    // 72..79 reserved, zero
    return true;
}

inline bool decode_header(const uint8_t in[kHeaderBytes], SetHeader& header, std::string* error) {
    if (get_u32(in + 0) != kMagic || get_u16(in + 4) != kVersion ||
        get_u16(in + 6) != kHeaderBytes || get_u32(in + 12) != 0 || get_u32(in + 44) > 1 ||
        get_u64(in + 72) != 0)
        return false;
    header.flags = get_u32(in + 12);
    header.layers = get_u32(in + 8);
    header.layout.origin = get_u32(in + 16);
    header.layout.capacity = get_u32(in + 20);
    header.layout.heads = get_u32(in + 24);
    header.layout.iterations = get_u32(in + 28);
    header.layout.precision_tail_tokens = get_u32(in + 32);
    header.layout.rows = get_u32(in + 36);
    header.layout.record_base_group = get_u32(in + 40);
    header.layout.compact_stage = get_u32(in + 44);
    header.layout.saved_generation = get_u64(in + 48);
    header.payload_bytes = get_u64(in + 56);
    header.layer_bytes = get_u64(in + 64);
    return valid_layout(header.layout, error) &&
           valid_payload_size(header.layout, header.layers) &&
           header.payload_bytes == payload_bytes(header.layout, header.layers) &&
           header.layer_bytes == layer_bytes(header.layout);
}

inline bool same_layout(const Layout& a, const Layout& b) {
    return a.origin == b.origin && a.capacity == b.capacity && a.heads == b.heads &&
           a.iterations == b.iterations && a.precision_tail_tokens == b.precision_tail_tokens &&
           a.compact_stage == b.compact_stage && a.rows == b.rows &&
           a.saved_generation == b.saved_generation;
}

inline bool write_section(Sink& sink, rkva_snapshot::Sha256& hash, const uint8_t* data,
                          size_t bytes, std::string* error) {
    if (!data || !bytes) {
        if (bytes)
            return false;
        return true;
    }
    hash.update(data, bytes);
    if (!sink.write(data, bytes)) {
        if (error)
            *error = "native state: write failed";
        return false;
    }
    return true;
}

inline bool read_section(Source& source, rkva_snapshot::Sha256& hash, uint8_t* data, size_t bytes,
                         std::string* error) {
    if (!data || !bytes) {
        if (bytes)
            return false;
        return true;
    }
    if (!source.read(data, bytes)) {
        if (error)
            *error = "native state: truncated source";
        return false;
    }
    hash.update(data, bytes);
    return true;
}

// Writes one self-contained set. The caller supplies one view per remote layer;
// all views must share the same layout and live row count. No view is mutated.
inline bool write_set(const std::vector<View>& views, Sink& sink, std::string* error,
                      rkva_snapshot::Digest* out_digest = nullptr) {
    if (views.empty() || views.size() > kMaxLayers) {
        if (error)
            *error = "native state: invalid layer count";
        return false;
    }
    SetHeader header;
    header.layers = uint32_t(views.size());
    header.layout = views.front().layout;
    if (!valid_layout(header.layout, error))
        return false;
    for (const auto& view : views) {
        if (!same_layout(view.layout, header.layout) || view.layout.record_base_group !=
                header.layout.record_base_group)
            return false;
    }
    header.payload_bytes = payload_bytes(header.layout, header.layers);
    header.layer_bytes = layer_bytes(header.layout);
    uint8_t encoded[kHeaderBytes];
    if (!encode_header(header, encoded, error))
        return false;
    rkva_snapshot::Sha256 hash;
    hash.update(encoded, sizeof(encoded));
    if (!sink.write(encoded, sizeof(encoded))) {
        if (error)
            *error = "native state: header write failed";
        return false;
    }
    for (const auto& view : views) {
        if (view.records_k == nullptr || view.records_v == nullptr ||
            view.stage_k == nullptr || view.stage_v == nullptr ||
            view.original_k == nullptr || view.original_v == nullptr ||
            view.original_pos == nullptr || (header.layout.precision_tail_tokens &&
                (view.tail_k == nullptr || view.tail_v == nullptr)))
            return false;
        const uint64_t record_bytes = records_bytes(header.layout);
        const uint64_t stage = stage_bytes(header.layout);
        const uint64_t tail = tail_bytes(header.layout);
        const uint64_t original = original_bytes(header.layout);
        if (!write_section(sink, hash, view.records_k, size_t(record_bytes), error) ||
            !write_section(sink, hash, view.records_v, size_t(record_bytes), error) ||
            !write_section(sink, hash, view.stage_k, size_t(stage), error) ||
            !write_section(sink, hash, view.stage_v, size_t(stage), error) ||
            !write_section(sink, hash, view.tail_k, size_t(tail), error) ||
            !write_section(sink, hash, view.tail_v, size_t(tail), error) ||
            !write_section(sink, hash, view.original_k, size_t(original), error) ||
            !write_section(sink, hash, view.original_v, size_t(original), error) ||
            !write_section(sink, hash, reinterpret_cast<const uint8_t*>(view.original_pos),
                           size_t(positions_bytes()), error))
            return false;
    }
    const auto digest = hash.finish();
    if (!sink.write(digest.data(), digest.size())) {
        if (error)
            *error = "native state: footer write failed";
        return false;
    }
    if (out_digest)
        *out_digest = digest;
    return true;
}

inline bool read_set(Source& source, std::vector<Storage>& out, std::string* error,
                     rkva_snapshot::Digest* out_digest = nullptr) {
    out.clear();
    uint8_t encoded[kHeaderBytes];
    if (!source.read(encoded, sizeof(encoded))) {
        if (error)
            *error = "native state: truncated header";
        return false;
    }
    SetHeader header;
    if (!decode_header(encoded, header, error))
        return false;
    rkva_snapshot::Sha256 hash;
    hash.update(encoded, sizeof(encoded));
    std::vector<Storage> candidate(header.layers);
    for (auto& storage : candidate) {
        storage.layout = header.layout;
        storage.records_k.resize(size_t(records_bytes(header.layout)));
        storage.records_v.resize(size_t(records_bytes(header.layout)));
        storage.stage_k.resize(size_t(stage_bytes(header.layout) / sizeof(uint16_t)));
        storage.stage_v.resize(size_t(stage_bytes(header.layout) / sizeof(uint16_t)));
        storage.tail_k.resize(size_t(tail_bytes(header.layout) / sizeof(uint16_t)));
        storage.tail_v.resize(size_t(tail_bytes(header.layout) / sizeof(uint16_t)));
        storage.original_k.resize(size_t(original_bytes(header.layout) / sizeof(float)));
        storage.original_v.resize(size_t(original_bytes(header.layout) / sizeof(float)));
        storage.original_pos.resize(kOriginalRows);
        if (!read_section(source, hash, storage.records_k.data(), storage.records_k.size(), error) ||
            !read_section(source, hash, storage.records_v.data(), storage.records_v.size(), error) ||
            !read_section(source, hash, reinterpret_cast<uint8_t*>(storage.stage_k.data()),
                          storage.stage_k.size() * sizeof(uint16_t), error) ||
            !read_section(source, hash, reinterpret_cast<uint8_t*>(storage.stage_v.data()),
                          storage.stage_v.size() * sizeof(uint16_t), error) ||
            !read_section(source, hash, reinterpret_cast<uint8_t*>(storage.tail_k.data()),
                          storage.tail_k.size() * sizeof(uint16_t), error) ||
            !read_section(source, hash, reinterpret_cast<uint8_t*>(storage.tail_v.data()),
                          storage.tail_v.size() * sizeof(uint16_t), error) ||
            !read_section(source, hash, reinterpret_cast<uint8_t*>(storage.original_k.data()),
                          storage.original_k.size() * sizeof(float), error) ||
            !read_section(source, hash, reinterpret_cast<uint8_t*>(storage.original_v.data()),
                          storage.original_v.size() * sizeof(float), error) ||
            !read_section(source, hash, reinterpret_cast<uint8_t*>(storage.original_pos.data()),
                          size_t(positions_bytes()), error))
            return false;
        if (!valid_ring(storage, error))
            return false;
    }
    rkva_snapshot::Digest stored{};
    if (!source.read(stored.data(), stored.size())) {
        if (error)
            *error = "native state: truncated footer";
        return false;
    }
    const auto digest = hash.finish();
    if (digest != stored) {
        if (error)
            *error = "native state: checksum mismatch";
        return false;
    }
    if (out_digest)
        *out_digest = digest;
    for (auto& storage : candidate)
        storage.valid = true;
    out = std::move(candidate);
    return true;
}

// Buffer helper used by CPU tests and adapters that already own a bounded blob.
inline bool write_set(const std::vector<View>& views, std::vector<uint8_t>& out,
                      std::string* error, rkva_snapshot::Digest* out_digest = nullptr) {
    VectorSink sink;
    if (!write_set(views, sink, error, out_digest))
        return false;
    out = std::move(sink.bytes);
    return true;
}
inline bool read_set(const std::vector<uint8_t>& bytes, std::vector<Storage>& out,
                     std::string* error, rkva_snapshot::Digest* out_digest = nullptr) {
    VectorSource source(bytes);
    return read_set(source, out, error, out_digest);
}

} // namespace rkva_native_state
