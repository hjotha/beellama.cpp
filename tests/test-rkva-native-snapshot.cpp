// CPU regression for the canonical Xbox native snapshot manifest and the exact
// revision-3 chunk layout. No D3D12 types or GPU calls are compiled here.
#include "../ggml/include/rkva-native-snapshot.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace rkva_native_snapshot;

Geometry geometry(uint32_t layers = 2) {
    return Geometry{8, 2, 256, 2, 4096, layers};
}
Descriptor descriptor() {
    return Descriptor{1024, 1024, RKVA_RANGE_STORAGE_KVARN4, 3, 128, 128, 256, 2,
                      4, 4, 8960, 3, 2, 128, 128, 1, 1, 1};
}
std::vector<uint8_t> host_sha() {
    std::vector<uint8_t> sha(32);
    for (size_t i = 0; i < sha.size(); ++i) sha[i] = uint8_t(0xa0 + i);
    return sha;
}
std::vector<LayerCoverage> coverage(uint32_t layers, uint32_t live) {
    std::vector<LayerCoverage> result(layers);
    for (auto & layer : result) {
        layer.tail_begin = live - 128;
        layer.stage_begin = (live - 1) / 128 * 128 - 128;
        layer.record_groups = live / 128;
        layer.ready = 1;
    }
    return result;
}

Manifest build_ok(uint32_t live = 513) {
    Manifest manifest;
    std::string error;
    const auto sha = host_sha();
    assert(build(geometry(), descriptor(), live, 0x700000001ull, sha.data(),
                 coverage(2, live), manifest, &error));
    return manifest;
}

void test_layout() {
    auto manifest = build_ok();
    assert(manifest.chunks.size() == 12);
    assert(manifest.payload_bytes == 3194880ull);
    const uint64_t record = 143360;
    const uint64_t record_stride = 286720;
    const uint64_t stage = 393216;
    const uint64_t tail = 262144;
    const uint64_t half = 942080;
    assert(manifest.chunks[0].layer == 0 && manifest.chunks[0].region == kRecordsK);
    assert(manifest.chunks[0].offset == 0 && manifest.chunks[0].bytes == record);
    assert(manifest.chunks[1].region == kStageK && manifest.chunks[1].offset == record_stride);
    assert(manifest.chunks[1].bytes == stage);
    assert(manifest.chunks[2].region == kTailK &&
           manifest.chunks[2].offset == record_stride + stage);
    assert(manifest.chunks[2].bytes == tail);
    assert(manifest.chunks[3].region == kRecordsV && manifest.chunks[3].offset == half);
    assert(manifest.chunks[4].region == kStageV &&
           manifest.chunks[4].offset == half + record_stride);
    assert(manifest.chunks[5].region == kTailV &&
           manifest.chunks[5].offset == half + record_stride + stage);
    assert(manifest.chunks[6].layer == 1 && manifest.chunks[6].region == kRecordsK);
    assert(manifest.chunks[6].offset == 0);

    // tail=0 removes both tail regions and shortens the half stride.
    auto no_tail_descriptor = descriptor();
    no_tail_descriptor.precision_tail_tokens = 0;
    Manifest no_tail;
    std::string error;
    const auto sha = host_sha();
    auto no_tail_coverage = coverage(2, 513);
    for (auto & layer : no_tail_coverage) layer.tail_begin = 0;
    assert(build(geometry(), no_tail_descriptor, 513, 0x700000001ull, sha.data(),
                 no_tail_coverage, no_tail, &error));
    assert(no_tail.chunks.size() == 8);
    for (const auto & chunk : no_tail.chunks)
        assert(chunk.region != kTailK && chunk.region != kTailV);
}

void test_roundtrip_and_validate() {
    auto manifest = build_ok();
    for (size_t i = 0; i < manifest.chunks.size(); ++i) {
        std::vector<uint8_t> chunk(i + 1, uint8_t(i));
        manifest.chunks[i].digest = rkva_snapshot::digest(chunk.data(), chunk.size());
    }
    std::vector<uint8_t> encoded;
    std::string error;
    assert(encode(manifest, encoded, &error));
    Manifest decoded;
    assert(decode(encoded, decoded, &error));
    assert(std::memcmp(decoded.host_state_sha256, manifest.host_state_sha256, 32) == 0);
    assert(decoded.generation == manifest.generation && decoded.live_rows == 513);
    assert(decoded.geometry.n_head == 8 && decoded.geometry.n_layers == 2);
    assert(decoded.descriptor.origin == 1024 && decoded.descriptor.layout_version == 3);
    assert(decoded.layers.size() == 2);
    assert(decoded.layers[0].tail_begin == manifest.layers[0].tail_begin);
    assert(decoded.chunks.size() == manifest.chunks.size());
    for (size_t i = 0; i < decoded.chunks.size(); ++i) {
        assert(decoded.chunks[i].layer == manifest.chunks[i].layer);
        assert(decoded.chunks[i].region == manifest.chunks[i].region);
        assert(decoded.chunks[i].offset == manifest.chunks[i].offset);
        assert(decoded.chunks[i].bytes == manifest.chunks[i].bytes);
        assert(decoded.chunks[i].digest == manifest.chunks[i].digest);
    }
    const auto sha = host_sha();
    assert(validate(decoded, geometry(), descriptor(), 513, 0x700000001ull, sha.data(), &error));

    auto bad = decoded;
    bad.host_state_sha256[0] ^= 1;
    assert(!validate(bad, geometry(), descriptor(), 513, 0x700000001ull, sha.data(), &error));
    bad = decoded;
    bad.generation++;
    assert(!validate(bad, geometry(), descriptor(), 513, 0x700000001ull, sha.data(), &error));
    bad = decoded;
    bad.live_rows++;
    assert(!validate(bad, geometry(), descriptor(), 513, 0x700000001ull, sha.data(), &error));
    bad = decoded;
    bad.geometry.n_head++;
    assert(!validate(bad, geometry(), descriptor(), 513, 0x700000001ull, sha.data(), &error));
    bad = decoded;
    bad.descriptor.origin += 128;
    assert(!validate(bad, geometry(), descriptor(), 513, 0x700000001ull, sha.data(), &error));
    bad = decoded;
    bad.layers[0].tail_begin = 400;
    assert(!validate(bad, geometry(), descriptor(), 513, 0x700000001ull, sha.data(), &error));
    bad = decoded;
    bad.layers[0].stage_begin = 640;
    assert(!validate(bad, geometry(), descriptor(), 513, 0x700000001ull, sha.data(), &error));
    bad = decoded;
    bad.layers[0].record_groups--;
    assert(!validate(bad, geometry(), descriptor(), 513, 0x700000001ull, sha.data(), &error));
    bad = decoded;
    std::swap(bad.chunks[0], bad.chunks[3]);
    assert(!validate(bad, geometry(), descriptor(), 513, 0x700000001ull, sha.data(), &error));
    bad = decoded;
    bad.chunks[0].bytes--;
    assert(!validate(bad, geometry(), descriptor(), 513, 0x700000001ull, sha.data(), &error));
    assert(!validate(decoded, geometry(), descriptor(), 513, 0x700000002ull, sha.data(), &error));
    auto other_sha = sha;
    other_sha[0] ^= 1;
    assert(!validate(decoded, geometry(), descriptor(), 513, 0x700000001ull, other_sha.data(), &error));
}

void test_encoded_corruption() {
    auto manifest = build_ok();
    std::vector<uint8_t> encoded;
    std::string error;
    assert(encode(manifest, encoded, &error));
    Manifest decoded;

    auto bad_magic = encoded;
    bad_magic[0] ^= 0xff;
    assert(!decode(bad_magic, decoded, &error));
    auto bad_version = encoded;
    bad_version[4] = 2;
    assert(!decode(bad_version, decoded, &error));
    auto bad_footer = encoded;
    bad_footer.back() ^= 1;
    assert(!decode(bad_footer, decoded, &error));
    auto truncated = encoded;
    truncated.resize(truncated.size() - 1);
    assert(!decode(truncated, decoded, &error));
    auto trailing = encoded;
    trailing.push_back(0);
    assert(!decode(trailing, decoded, &error));

    // A legacy RVKS header must never be interpreted as a native manifest.
    std::vector<uint8_t> legacy(encoded.size(), 0);
    legacy[0] = 0x52;
    legacy[1] = 0x56;
    legacy[2] = 0x4b;
    legacy[3] = 0x53;
    assert(!decode(legacy, decoded, &error));
}

void test_build_rejections() {
    auto bad_geometry = geometry();
    bad_geometry.head_dim = 128;
    auto bad_descriptor = descriptor();
    bad_descriptor.layout_version = 2;
    auto sha = host_sha();
    std::string error;
    Manifest manifest;
    assert(!build(bad_geometry, descriptor(), 513, 0x700000001ull, sha.data(),
                  coverage(2, 513), manifest, &error));
    assert(!build(geometry(), bad_descriptor, 513, 0x700000001ull, sha.data(),
                  coverage(2, 513), manifest, &error));
    auto bad_coverage = coverage(2, 513);
    bad_coverage[0].tail_begin = 400;
    assert(!build(geometry(), descriptor(), 513, 0x700000001ull, sha.data(),
                  bad_coverage, manifest, &error));
    bad_coverage = coverage(2, 513);
    bad_coverage[0].tail_begin = 200;
    bad_coverage[0].stage_begin = 640;
    assert(!build(geometry(), descriptor(), 513, 0x700000001ull, sha.data(),
                  bad_coverage, manifest, &error));
    assert(!build(geometry(), descriptor(), 513, 0, sha.data(),
                  coverage(2, 513), manifest, &error));
    const uint8_t zero_sha[32] = {};
    // All-zero host state would be a repeated placeholder; the engine request
    // path rejects it separately. The codec accepts a caller-supplied binding.
    assert(build(geometry(), descriptor(), 513, 0x700000001ull, zero_sha,
                 coverage(2, 513), manifest, &error));
}

void test_chunk_bitflip_same_size() {
    auto manifest = build_ok();
    std::vector<uint8_t> payload(manifest.chunks[0].bytes, 0x5a);
    manifest.chunks[0].digest = rkva_snapshot::digest(payload.data(), payload.size());
    std::vector<uint8_t> encoded;
    std::string error;
    assert(encode(manifest, encoded, &error));
    Manifest decoded;
    assert(decode(encoded, decoded, &error));
    assert(verify_chunk(decoded.chunks[0], payload.data(), payload.size(), &error));
    payload[payload.size() / 2] ^= 0x01; // same size, different content
    assert(!verify_chunk(decoded.chunks[0], payload.data(), payload.size(), &error));
    assert(!verify_chunk(decoded.chunks[0], payload.data(), payload.size() - 1, &error));
}

void test_quota_admission_never_evicts() {
    std::string error;
    constexpr uint64_t limit = 8ull << 30;
    StoreBudget used;
    used.bytes = limit - 1024;
    used.snapshots = 4;
    // A fifth object cannot be admitted and no victim is selected.
    assert(!admit_snapshot_store(used, 4096, limit, 4, &error));
    assert(used.bytes == limit - 1024 && used.snapshots == 4);
    used.bytes = (1ull << 30);
    used.snapshots = 3;
    assert(admit_snapshot_store(used, 512ull << 20, limit, 4, &error));
    assert(!admit_snapshot_store(used, UINT64_MAX, limit, 4, &error));
    assert(!admit_snapshot_store(used, 1, 0, 4, &error));
    assert(!admit_snapshot_store(used, 1, limit, 0, &error));
    // Refusal is a pure decision: no victim list exists and old entries stay.
    std::vector<uint64_t> existing{101, 202, 303, 404};
    const auto before = existing;
    assert(!admit_snapshot_store(StoreBudget{limit, 4}, 1, limit, 4, &error));
    if (admit_snapshot_store(StoreBudget{limit, 4}, 1, limit, 4, &error))
        existing.erase(existing.begin()); // unreachable under admission refusal
    assert(existing == before);
}

void test_save_gate_vs_empty_restore() {
    std::string error;
    const auto cov = coverage(2, 513);
    const std::vector<uint32_t> empty_counts{0, 0};
    const std::vector<bool> empty_ready{false, false};
    const std::vector<uint32_t> empty_groups{0, 0};
    assert(!validate_resident_layers(descriptor(), 513, empty_counts, empty_ready,
                                     empty_groups, cov, &error));
    const std::vector<uint32_t> counts{513, 513};
    const std::vector<bool> ready{true, true};
    const std::vector<uint32_t> groups{4, 4};
    assert(validate_resident_layers(descriptor(), 513, counts, ready, groups, cov, &error));
    auto bad_groups = groups;
    bad_groups[1] = 3;
    assert(!validate_resident_layers(descriptor(), 513, counts, ready, bad_groups, cov, &error));
    // A fresh empty Xbox session has no counts/ready/groups; the manifest is
    // the authority and must validate independently of the live counters.
    auto manifest = build_ok();
    const auto sha = host_sha();
    assert(validate(manifest, geometry(), descriptor(), 513, 0x700000001ull, sha.data(), &error));
}

void test_chunk_splitting() {
    auto wide = descriptor();
    wide.capacity = 65536;
    auto geo = geometry(1);
    geo.max_ctx = 131072;
    const uint32_t live = 65536;
    auto cov = coverage(1, live);
    cov[0].tail_begin = live - 128;
    cov[0].stage_begin = (live - 1) / 128 * 128 - 128;
    cov[0].record_groups = live / 128;
    const auto sha = host_sha();
    Manifest manifest;
    std::string error;
    assert(build(geo, wide, live, 0x700000002ull, sha.data(), cov, manifest, &error));
    assert(manifest.chunks.size() > 3);
    for (const auto & chunk : manifest.chunks)
        assert(chunk.bytes && chunk.bytes <= kChunkBytes);
    std::vector<uint8_t> encoded;
    assert(encode(manifest, encoded, &error));
    Manifest decoded;
    assert(decode(encoded, decoded, &error));
    assert(validate(decoded, geo, wide, live, 0x700000002ull, sha.data(), &error));
}

} // namespace

int main() {
    test_layout();
    test_roundtrip_and_validate();
    test_encoded_corruption();
    test_build_rejections();
    test_chunk_bitflip_same_size();
    test_quota_admission_never_evicts();
    test_save_gate_vs_empty_restore();
    test_chunk_splitting();
    std::printf("native snapshot manifest CPU regression: ok\n");
    return 0;
}
