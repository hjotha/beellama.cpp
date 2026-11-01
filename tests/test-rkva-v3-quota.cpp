// Phase 2 POSIX v3 quota accounting test.
//
// Builds one canonical v3 object from the existing native snapshot codec,
// then checks conservative charging, orphan-stage preservation, fail-closed
// handling of unknown/corrupt entries and legacy namespace isolation. No
// SQLite admission, writer, server, GPU, Xbox or power-loss claim is involved.

#include "rkva-v3-quota.h"
#undef NDEBUG
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace snap = rkva_native_snapshot;
namespace quota = rkva_v3_quota;

static std::string hex_key(const uint8_t * key) {
    return rkva_snapshot::hex(key);
}

static void write_file(const fs::path & path, const std::string & data) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(data.data(), std::streamsize(data.size()));
    assert(output.good());
}

static void make_object(const fs::path & root, const snap::Manifest & manifest,
                        uint8_t key_byte) {
    uint8_t host_key[32];
    std::fill(std::begin(host_key), std::end(host_key), key_byte);
    const auto dir = root / hex_key(host_key);
    assert(fs::create_directory(dir));
    snap::Manifest final_manifest = manifest;
    for (size_t i = 0; i < final_manifest.chunks.size(); ++i) {
        auto & chunk = final_manifest.chunks[i];
        std::string data(chunk.bytes, '\0');
        for (size_t j = 0; j < data.size(); ++j) {
            data[j] = char((uint32_t(i) * 71u + uint32_t(j) * 13u + key_byte) & 0xffu);
        }
        const auto digest = rkva_snapshot::digest(data.data(), data.size());
        chunk.digest = digest;
        write_file(dir / ("n-" + std::to_string(i) + ".kv"), data);
    }
    std::vector<uint8_t> encoded;
    std::string why;
    assert(snap::encode(final_manifest, encoded, &why));
    write_file(dir / "manifest-v2.bin",
               std::string(reinterpret_cast<const char *>(encoded.data()), encoded.size()));
}

int main() {
    char tmp[] = "/tmp/rkva-phase2-quota-XXXXXX";
    char * name = mkdtemp(tmp);
    assert(name != nullptr);
    const fs::path parent(name);
    const fs::path root = parent / "rkva-snapshots-v3";
    const fs::path legacy = parent / "rkva-snapshots-v2";
    assert(fs::create_directory(root));
    assert(fs::create_directory(legacy));
    write_file(legacy / "sentinel", "never touch");

    // Missing namespace fails closed instead of reporting zero usage.
    bool rejected = false;
    try {
        (void) quota::scan(parent / "absent");
    } catch (const std::exception &) {
        rejected = true;
    }
    assert(rejected);

    const snap::Geometry geometry{8, 2, 256, 2, 4096, 1};
    const snap::Descriptor descriptor{1024, 1024, RKVA_RANGE_STORAGE_KVARN4, 3, 128, 128,
                                      256, 2, 4, 4, 8960, 3, 2, 128, 128, 1, 1, 1};
    const snap::LayerCoverage coverage{385, 384, 4, 1};
    uint8_t host_key[32];
    std::fill(std::begin(host_key), std::end(host_key), 0x44);
    snap::Manifest manifest;
    std::string why;
    assert(snap::build(geometry, descriptor, 513, 0x700000001ull, host_key, {coverage},
                       manifest, &why));
    make_object(root, manifest, 0x44);

    const auto empty = quota::scan(parent);
    assert(empty.objects == 1 && empty.bytes > manifest.payload_bytes);

    // A partial orphan stage is charged and preserved.
    const fs::path stage = root / ("tmp-" + std::string(64, '7'));
    assert(fs::create_directory(stage));
    write_file(stage / "n-0.kv", std::string(4096, 'x'));
    const auto staged = quota::scan(parent);
    assert(staged.objects == 2 && staged.bytes >= empty.bytes + 4096);
    assert(fs::exists(stage / "n-0.kv"));

    // Removing the stage returns to the published-object usage (the root
    // directory allocation may legitimately grow and not shrink).
    fs::remove_all(stage);
    const auto restored = quota::scan(parent);
    assert(restored.objects == 1 && restored.bytes >= empty.bytes);

    // Unknown names, symlinks and corrupt published objects fail closed.
    assert(fs::create_directory(root / "junk"));
    rejected = false;
    try {
        (void) quota::scan(parent);
    } catch (const std::exception &) {
        rejected = true;
    }
    assert(rejected);
    fs::remove_all(root / "junk");

    const fs::path object_dir = root / hex_key(host_key);
    fs::create_symlink(legacy / "sentinel", object_dir / "n-99.kv");
    rejected = false;
    try {
        (void) quota::scan(parent);
    } catch (const std::exception &) {
        rejected = true;
    }
    assert(rejected);
    fs::remove(object_dir / "n-99.kv");

    const fs::path chunk = object_dir / "n-0.kv";
    std::ifstream input(chunk, std::ios::binary);
    std::string original((std::istreambuf_iterator<char>(input)), {});
    input.close();
    std::string corrupt = original;
    corrupt[0] ^= 1;
    write_file(chunk, corrupt);
    rejected = false;
    try {
        (void) quota::scan(parent);
    } catch (const std::exception &) {
        rejected = true;
    }
    assert(rejected);
    write_file(chunk, original);
    assert(quota::scan(parent).objects == 1);

    assert(fs::exists(legacy / "sentinel"));
    std::ifstream sentinel(legacy / "sentinel");
    std::string value;
    std::getline(sentinel, value);
    assert(value == "never touch");

    fs::remove_all(parent);
    std::cout << "v3 quota accounting: charging, orphan stages, corruption and "
                 "legacy isolation: PASS\n";
    return 0;
}
