#pragma once

// Canonical manifest and chunk layout for the Xbox LocalState native KVarN4
// snapshot. Pure CPU: no D3D12 types, no engine pointers, no host file paths.
//
// The manifest binds a snapshot to the complete host sequence-state SHA-256,
// the negotiated geometry/descriptor, the evaluated row count and the
// generation that produced it. The chunk descriptor list is reconstructed
// deterministically from those values; restore rejects any different order or
// extent instead of trusting the directory listing.

#include "remote-attn-protocol.h"
#include "rkva-snapshot-digest.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace rkva_native_snapshot {

constexpr uint32_t kMagic = 0x53564b52; // "RKVS"
constexpr uint16_t kVersion = 1;
constexpr uint16_t kHeaderBytes = 176;
constexpr uint32_t kMaxLayers = 128;
constexpr uint32_t kMaxChunks = 4096;
constexpr uint64_t kChunkBytes = 8ull << 20;
constexpr uint64_t kMaxPayloadBytes = 4ull << 30;
constexpr uint64_t kMaxManifestBytes = 1ull << 20;
constexpr uint32_t kRecordBytes = 8960;
constexpr uint32_t kStageRows = 384;
constexpr uint32_t kTailRingRows = 256;

enum region : uint32_t {
    kRecordsK = 1,
    kRecordsV = 2,
    kStageK = 3,
    kStageV = 4,
    kTailK = 5,
    kTailV = 6,
};

struct Geometry {
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t head_slices = 0;
    uint32_t max_ctx = 0;
    uint32_t n_layers = 0;
};

struct Descriptor {
    uint32_t origin = 0;
    uint32_t capacity = 0;
    uint32_t storage = 0;
    uint32_t layout_version = 0;
    uint32_t token_group = 0;
    uint32_t record_dim = 0;
    uint32_t head_dim = 0;
    uint32_t head_slices = 0;
    uint32_t key_bits = 0;
    uint32_t value_bits = 0;
    uint32_t record_bytes = 0;
    uint32_t stage_groups = 0;
    uint32_t tail_groups = 0;
    uint32_t sink_tokens = 0;
    uint32_t precision_tail_tokens = 0;
    uint32_t wht_version = 0;
    uint32_t domain = 0;
    uint32_t record_policy = 0;
};

struct LayerCoverage {
    uint32_t tail_begin = 0;
    uint32_t stage_begin = 0;
    uint32_t record_groups = 0;
    uint32_t ready = 0;
};

struct Chunk {
    uint32_t layer = 0;
    uint32_t region = 0;
    uint64_t offset = 0;
    uint32_t bytes = 0;
    rkva_snapshot::Digest digest{};
};

struct Manifest {
    uint8_t host_state_sha256[32] = {};
    uint64_t generation = 0;
    uint32_t live_rows = 0;
    uint32_t flags = 0;
    Geometry geometry;
    Descriptor descriptor;
    std::vector<LayerCoverage> layers;
    std::vector<Chunk> chunks;
    uint64_t payload_bytes = 0;
};

inline void put_u16(uint8_t* dst, uint16_t value) {
    dst[0] = uint8_t(value);
    dst[1] = uint8_t(value >> 8);
}
inline uint16_t get_u16(const uint8_t* src) {
    return uint16_t(src[0]) | uint16_t(uint16_t(src[1]) << 8);
}
inline void put_u32(uint8_t* dst, uint32_t value) {
    for (int i = 0; i < 4; ++i)
        dst[i] = uint8_t(value >> (8 * i));
}
inline void put_u64(uint8_t* dst, uint64_t value) {
    for (int i = 0; i < 8; ++i)
        dst[i] = uint8_t(value >> (8 * i));
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

inline bool valid_descriptor(const Descriptor& d, const Geometry& g) {
    return g.n_head && g.n_head_kv && g.n_head % g.n_head_kv == 0 && g.n_head_kv <= 256 &&
           g.head_dim == 256 && g.head_slices == 2 && g.max_ctx && g.n_layers &&
           g.n_layers <= kMaxLayers && d.storage == RKVA_RANGE_STORAGE_KVARN4 &&
           d.layout_version == 3 && d.origin % 128 == 0 && d.capacity > 0 &&
           d.capacity <= uint32_t(INT32_MAX) && d.origin <= uint32_t(INT32_MAX) - d.capacity &&
           d.token_group == 128 && d.record_dim == 128 && d.head_dim == 256 &&
           d.head_slices == 2 && d.key_bits == 4 && d.value_bits == 4 &&
           d.record_bytes == kRecordBytes && d.stage_groups == 3 && d.tail_groups == 2 &&
           d.sink_tokens == 128 &&
           (d.precision_tail_tokens == 0 || d.precision_tail_tokens == 128) &&
           d.wht_version == 1 && d.domain == 1 && d.record_policy == 1;
}

inline bool verify_chunk(const Chunk& chunk, const void* data, size_t bytes,
                         std::string* error) {
    if (!data || bytes != chunk.bytes) {
        if (error)
            *error = "native snapshot: chunk extent mismatch";
        return false;
    }
    if (rkva_snapshot::digest(data, bytes) != chunk.digest) {
        if (error)
            *error = "native snapshot: chunk checksum mismatch";
        return false;
    }
    return true;
}

struct StoreBudget {
    uint64_t bytes = 0;
    uint32_t snapshots = 0;
};

// Native snapshots are content-addressed pairs with a host file. Evicting an
// older directory before the new pair commits can leave the previous host slot
// pointing at deleted Xbox data. Admission therefore never selects a victim:
// a full store fails explicitly and the operator must garbage-collect after
// preserving the host/Xbox pair they intend to keep.
inline bool admit_snapshot_store(const StoreBudget& used, uint64_t requested,
                                 uint64_t limit, uint32_t max_snapshots,
                                 std::string* error) {
    if (!max_snapshots || used.snapshots >= max_snapshots ||
        requested > limit || used.bytes > limit - requested) {
        if (error)
            *error = "native snapshot: quota exhausted; refusing to evict a valid pair";
        return false;
    }
    return true;
}

inline bool same_geometry(const Geometry& a, const Geometry& b) {
    return a.n_head == b.n_head && a.n_head_kv == b.n_head_kv && a.head_dim == b.head_dim &&
           a.head_slices == b.head_slices && a.max_ctx == b.max_ctx && a.n_layers == b.n_layers;
}
inline bool same_descriptor(const Descriptor& a, const Descriptor& b) {
    return a.origin == b.origin && a.capacity == b.capacity && a.storage == b.storage &&
           a.layout_version == b.layout_version && a.token_group == b.token_group &&
           a.record_dim == b.record_dim && a.head_dim == b.head_dim &&
           a.head_slices == b.head_slices && a.key_bits == b.key_bits &&
           a.value_bits == b.value_bits && a.record_bytes == b.record_bytes &&
           a.stage_groups == b.stage_groups && a.tail_groups == b.tail_groups &&
           a.sink_tokens == b.sink_tokens &&
           a.precision_tail_tokens == b.precision_tail_tokens && a.wht_version == b.wht_version &&
           a.domain == b.domain && a.record_policy == b.record_policy;
}

inline uint64_t record_group_bytes(const Geometry& g) {
    return uint64_t(g.n_head_kv) * 2ull * kRecordBytes;
}
inline uint64_t record_stride_bytes(const Descriptor& d, const Geometry& g) {
    return uint64_t((uint64_t(d.capacity) + 127) / 128) * record_group_bytes(g);
}
inline uint64_t stage_bytes(const Geometry& g) {
    return uint64_t(kStageRows) * uint64_t(g.n_head_kv) * 256ull * sizeof(uint16_t);
}
inline uint32_t physical_tail_rows(const Descriptor& d) {
    return d.precision_tail_tokens ? kTailRingRows : 0;
}
inline uint64_t tail_bytes(const Descriptor& d, const Geometry& g) {
    return uint64_t(physical_tail_rows(d)) * uint64_t(g.n_head_kv) * 256ull * sizeof(uint16_t);
}
inline uint64_t half_stride_bytes(const Descriptor& d, const Geometry& g) {
    return record_stride_bytes(d, g) + stage_bytes(g) + tail_bytes(d, g);
}
inline uint64_t layer_stride_bytes(const Descriptor& d, const Geometry& g) {
    return 2 * half_stride_bytes(d, g);
}

inline bool valid_coverage(const Descriptor& d, uint32_t live,
                           const LayerCoverage& coverage, std::string* error) {
    const auto reject = [&](const char* message) {
        if (error)
            *error = message;
        return false;
    };
    if (!live || live > d.capacity || (coverage.ready ? 0u : 1u) ||
        coverage.record_groups != live / 128 || coverage.stage_begin % 128 ||
        coverage.stage_begin > (live ? (live - 1) / 128 * 128 : 0) ||
        coverage.tail_begin > live)
        return reject("native snapshot: invalid layer coverage");
    if (d.precision_tail_tokens &&
        coverage.tail_begin > (live > d.precision_tail_tokens ? live - d.precision_tail_tokens : 0))
        return reject("native snapshot: tail coverage does not reach the live prefix");
    return true;
}

// Save-side gate. A snapshot SAVE requires every resident layer at the same
// evaluated checkpoint. Restore never calls this: an empty fresh session has
// zero counts, and the manifest is validated instead.
inline bool validate_resident_layers(const Descriptor& d, uint32_t live,
                                     const std::vector<uint32_t>& counts,
                                     const std::vector<bool>& ready,
                                     const std::vector<uint32_t>& record_groups,
                                     const std::vector<LayerCoverage>& coverage,
                                     std::string* error) {
    if (!live || live > d.capacity || counts.empty() || ready.size() != counts.size() ||
        record_groups.size() != counts.size() || coverage.size() != counts.size())
        return false;
    for (size_t il = 0; il < counts.size(); ++il) {
        if (counts[il] != live || !ready[il] || record_groups[il] != live / 128 ||
            !valid_coverage(d, live, coverage[il], error))
            return false;
    }
    return true;
}

inline bool add_region(const Geometry& g, const Descriptor& d, uint32_t live, uint32_t layer,
                       uint32_t region, uint64_t base, uint64_t bytes, Manifest& manifest,
                       std::string* error) {
    uint64_t offset = 0;
    while (offset < bytes) {
        if (manifest.chunks.size() >= kMaxChunks)
            return false;
        Chunk chunk;
        chunk.layer = layer;
        chunk.region = region;
        chunk.offset = base + offset;
        chunk.bytes = uint32_t(std::min<uint64_t>(bytes - offset, kChunkBytes));
        manifest.chunks.push_back(chunk);
        manifest.payload_bytes += chunk.bytes;
        offset += chunk.bytes;
    }
    (void) g;
    (void) d;
    (void) live;
    if (error)
        error->clear();
    return true;
}

// Builds the canonical manifest and deterministic chunk layout. The caller
// supplies per-layer coverage captured under the same fence as the resident
// counts. Every layer must be at the same live row count.
inline bool build(const Geometry& g, const Descriptor& d, uint32_t live, uint64_t generation,
                  const uint8_t host_sha256[32], const std::vector<LayerCoverage>& layers,
                  Manifest& manifest, std::string* error) {
    manifest = Manifest{};
    if (!host_sha256 || !valid_descriptor(d, g) || !live || live > d.capacity ||
        layers.size() != g.n_layers || generation == 0 || uint32_t(generation) == UINT32_MAX) {
        if (error)
            *error = "native snapshot: invalid geometry/descriptor/coverage";
        return false;
    }
    for (const auto& coverage : layers)
        if (!valid_coverage(d, live, coverage, error))
            return false;
    manifest.geometry = g;
    manifest.descriptor = d;
    manifest.live_rows = live;
    manifest.generation = generation;
    std::memcpy(manifest.host_state_sha256, host_sha256, 32);
    manifest.layers = layers;
    const uint64_t groups = live / 128;
    const uint64_t record = record_group_bytes(g);
    const uint64_t record_base = record_stride_bytes(d, g);
    const uint64_t stage = stage_bytes(g);
    const uint64_t tail = tail_bytes(d, g);
    const uint64_t half = half_stride_bytes(d, g);
    for (uint32_t layer = 0; layer < g.n_layers; ++layer) {
        for (uint32_t kind = 0; kind < 2; ++kind) {
            const uint64_t plane = uint64_t(kind) * half;
            const uint32_t rk = kind ? kRecordsV : kRecordsK;
            const uint32_t sk = kind ? kStageV : kStageK;
            const uint32_t tk = kind ? kTailV : kTailK;
            if (!add_region(g, d, live, layer, rk, plane, groups * record, manifest, error) ||
                !add_region(g, d, live, layer, sk, plane + record_base, stage, manifest, error) ||
                !add_region(g, d, live, layer, tk, plane + record_base + stage, tail, manifest,
                            error))
                return false;
        }
    }
    if (!manifest.chunks.size() || manifest.payload_bytes > kMaxPayloadBytes) {
        if (error)
            *error = "native snapshot: payload exceeds bounded store";
        return false;
    }
    return true;
}

inline bool encode(const Manifest& manifest, std::vector<uint8_t>& out, std::string* error) {
    if (manifest.layers.size() != manifest.geometry.n_layers ||
        manifest.chunks.empty() || manifest.chunks.size() > kMaxChunks ||
        !valid_descriptor(manifest.descriptor, manifest.geometry))
        return false;
    const uint64_t bytes = uint64_t(kHeaderBytes) + 16ull * manifest.layers.size() +
                           56ull * manifest.chunks.size() + 32ull;
    if (bytes > kMaxManifestBytes)
        return false;
    out.assign(size_t(bytes), 0);
    uint8_t* p = out.data();
    put_u32(p + 0, kMagic);
    p[4] = uint8_t(kVersion);
    p[5] = uint8_t(kVersion >> 8);
    p[6] = uint8_t(kHeaderBytes);
    p[7] = uint8_t(kHeaderBytes >> 8);
    put_u32(p + 8, manifest.live_rows);
    put_u32(p + 12, manifest.geometry.n_layers);
    put_u32(p + 16, uint32_t(manifest.chunks.size()));
    put_u32(p + 20, manifest.flags);
    put_u64(p + 24, manifest.generation);
    std::memcpy(p + 32, manifest.host_state_sha256, 32);
    const Geometry& g = manifest.geometry;
    put_u32(p + 64, g.n_head);
    put_u32(p + 68, g.n_head_kv);
    put_u32(p + 72, g.head_dim);
    put_u32(p + 76, g.head_slices);
    put_u32(p + 80, g.max_ctx);
    put_u32(p + 84, g.n_layers);
    const Descriptor& d = manifest.descriptor;
    const uint32_t fields[18] = {
        d.origin,        d.capacity,         d.storage,        d.layout_version,
        d.token_group,   d.record_dim,       d.head_dim,       d.head_slices,
        d.key_bits,      d.value_bits,       d.record_bytes,   d.stage_groups,
        d.tail_groups,   d.sink_tokens,      d.precision_tail_tokens, d.wht_version,
        d.domain,        d.record_policy,
    };
    for (uint32_t i = 0; i < 18; ++i)
        put_u32(p + 88 + 4 * i, fields[i]);
    put_u64(p + 160, manifest.payload_bytes);
    // 168..175 reserved
    p += kHeaderBytes;
    for (const auto& coverage : manifest.layers) {
        put_u32(p + 0, coverage.tail_begin);
        put_u32(p + 4, coverage.stage_begin);
        put_u32(p + 8, coverage.record_groups);
        put_u32(p + 12, coverage.ready);
        p += 16;
    }
    for (const auto& chunk : manifest.chunks) {
        put_u32(p + 0, chunk.layer);
        put_u32(p + 4, chunk.region);
        put_u64(p + 8, chunk.offset);
        put_u32(p + 16, chunk.bytes);
        put_u32(p + 20, 0);
        std::memcpy(p + 24, chunk.digest.data(), 32);
        p += 56;
    }
    const auto digest = rkva_snapshot::digest(out.data(), size_t(bytes) - 32);
    std::memcpy(out.data() + bytes - 32, digest.data(), 32);
    if (error)
        error->clear();
    return true;
}

inline bool decode(const std::vector<uint8_t>& in, Manifest& manifest, std::string* error) {
    const auto reject = [&](const char* message) {
        if (error)
            *error = message;
        manifest = Manifest{};
        return false;
    };
    if (in.size() < kHeaderBytes + 32 || in.size() > kMaxManifestBytes)
        return reject("native snapshot: invalid manifest size");
    const uint8_t* p = in.data();
    if (get_u32(p + 0) != kMagic || p[4] != uint8_t(kVersion) || p[5] != 0 ||
        get_u16(p + 6) != kHeaderBytes || get_u32(p + 20) != 0)
        return reject("native snapshot: unsupported manifest version");
    manifest = Manifest{};
    manifest.live_rows = get_u32(p + 8);
    manifest.geometry.n_layers = get_u32(p + 12);
    const uint32_t chunks = get_u32(p + 16);
    manifest.generation = get_u64(p + 24);
    std::memcpy(manifest.host_state_sha256, p + 32, 32);
    manifest.geometry.n_head = get_u32(p + 64);
    manifest.geometry.n_head_kv = get_u32(p + 68);
    manifest.geometry.head_dim = get_u32(p + 72);
    manifest.geometry.head_slices = get_u32(p + 76);
    manifest.geometry.max_ctx = get_u32(p + 80);
    manifest.geometry.n_layers = get_u32(p + 84);
    uint32_t fields[18];
    for (uint32_t i = 0; i < 18; ++i)
        fields[i] = get_u32(p + 88 + 4 * i);
    Descriptor& d = manifest.descriptor;
    d.origin = fields[0];
    d.capacity = fields[1];
    d.storage = fields[2];
    d.layout_version = fields[3];
    d.token_group = fields[4];
    d.record_dim = fields[5];
    d.head_dim = fields[6];
    d.head_slices = fields[7];
    d.key_bits = fields[8];
    d.value_bits = fields[9];
    d.record_bytes = fields[10];
    d.stage_groups = fields[11];
    d.tail_groups = fields[12];
    d.sink_tokens = fields[13];
    d.precision_tail_tokens = fields[14];
    d.wht_version = fields[15];
    d.domain = fields[16];
    d.record_policy = fields[17];
    manifest.payload_bytes = get_u64(p + 160);
    if (get_u64(p + 168) != 0 || !chunks || chunks > kMaxChunks ||
        manifest.geometry.n_layers == 0 || manifest.geometry.n_layers > kMaxLayers ||
        !valid_descriptor(d, manifest.geometry))
        return reject("native snapshot: invalid manifest fields");
    const uint64_t expected = uint64_t(kHeaderBytes) + 16ull * manifest.geometry.n_layers +
                              56ull * chunks + 32ull;
    if (expected != in.size())
        return reject("native snapshot: manifest extent mismatch");
    const auto stored = rkva_snapshot::digest(p, size_t(in.size()) - 32);
    if (std::memcmp(stored.data(), p + in.size() - 32, 32) != 0)
        return reject("native snapshot: manifest checksum mismatch");
    p += kHeaderBytes;
    manifest.layers.resize(manifest.geometry.n_layers);
    for (auto& coverage : manifest.layers) {
        coverage.tail_begin = get_u32(p + 0);
        coverage.stage_begin = get_u32(p + 4);
        coverage.record_groups = get_u32(p + 8);
        coverage.ready = get_u32(p + 12);
        p += 16;
    }
    manifest.chunks.resize(chunks);
    for (auto& chunk : manifest.chunks) {
        chunk.layer = get_u32(p + 0);
        chunk.region = get_u32(p + 4);
        chunk.offset = get_u64(p + 8);
        chunk.bytes = get_u32(p + 16);
        if (get_u32(p + 20) != 0)
            return reject("native snapshot: noncanonical chunk");
        std::memcpy(chunk.digest.data(), p + 24, 32);
        p += 56;
    }
    if (error)
        error->clear();
    return true;
}

// Rejects any manifest that does not exactly match the negotiated live state
// and the canonical chunk layout derived from it.
inline bool validate(const Manifest& manifest, const Geometry& geometry, const Descriptor& descriptor,
                     uint32_t live_rows, uint64_t saved_generation,
                     const uint8_t host_sha256[32], std::string* error) {
    const auto reject = [&](const char* message) {
        if (error)
            *error = message;
        return false;
    };
    if (!host_sha256 || !valid_descriptor(descriptor, geometry) || !live_rows ||
        live_rows != manifest.live_rows || live_rows > descriptor.capacity ||
        saved_generation != manifest.generation || saved_generation == 0 ||
        uint32_t(saved_generation) == UINT32_MAX ||
        manifest.flags || !same_geometry(geometry, manifest.geometry) ||
        !same_descriptor(descriptor, manifest.descriptor) ||
        std::memcmp(host_sha256, manifest.host_state_sha256, 32) != 0 ||
        manifest.layers.size() != geometry.n_layers)
        return reject("native snapshot: identity/geometry/descriptor/generation mismatch");
    Manifest expected;
    if (!build(geometry, descriptor, live_rows, saved_generation, host_sha256, manifest.layers,
               expected, error))
        return false;
    if (expected.chunks.size() != manifest.chunks.size() ||
        expected.payload_bytes != manifest.payload_bytes)
        return reject("native snapshot: chunk layout mismatch");
    for (size_t i = 0; i < expected.chunks.size(); ++i) {
        const auto& want = expected.chunks[i];
        const auto& got = manifest.chunks[i];
        if (want.layer != got.layer || want.region != got.region ||
            want.offset != got.offset || want.bytes != got.bytes)
            return reject("native snapshot: chunk order/extent mismatch");
    }
    return true;
}

} // namespace rkva_native_snapshot
