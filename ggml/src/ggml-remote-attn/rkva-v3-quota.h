#pragma once

// Isolated Phase 2 quota accounting for the managed v3 object namespace.
//
// This is the POSIX side of the admission scan that was developed inside the
// SQLite PREPARE experiment. It is not an OS-level physical reservation and it
// never deletes, repairs or imports an object. The caller must already hold the
// concrete cross-process store authority (rkva_managed_store::lease) for the
// parent directory; this helper does not take that lock itself.
//
// Charging rules (deliberately conservative):
//   * every directory is charged its allocated blocks (st_blocks * 512);
//   * every regular file is charged max(logical size, allocated blocks);
//   * partial tmp-<tx> stages are charged and preserved, including orphans;
//   * published key directories must contain a canonical, self-consistent
//     manifest and exactly the declared chunk files with matching extents and
//     SHA-256 digests.
// Unknown names, symlinks, hardlinked files, malformed publications, missing
// roots and arithmetic overflow fail closed by throwing std::runtime_error.
//
// The Windows FileStandardInfo scanner is not ported in Phase 2 (see
// docs/rkva-managed-snapshot-phase2.md). Other platforms are unavailable.
// Legacy rkva-snapshots-v1/v2 namespaces are never visited.

#include "rkva-native-snapshot.h"
#include "rkva-snapshot-digest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/stat.h>
#endif

namespace rkva_v3_quota {

struct usage {
    uint64_t bytes = 0;
    size_t objects = 0;
};

#if defined(__linux__)

namespace detail {
inline void need(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
inline bool hex_name(const std::string & name) {
    return name.size() == 64 && std::all_of(name.begin(), name.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
} // namespace detail

// parent must be the private managed parent that contains the existing
// rkva-snapshots-v3 directory. directory_margin exists for API parity with the
// Windows scanner; the POSIX scan charges actual directory blocks instead.
inline usage scan(const std::filesystem::path & parent, uint64_t directory_margin = 0) {
    (void) directory_margin;
    namespace disk = std::filesystem;
    namespace snap = rkva_native_snapshot;
    constexpr int64_t max_amount = std::numeric_limits<int64_t>::max();
    const disk::path root = parent / "rkva-snapshots-v3";
    std::error_code ec;
    const auto root_status = disk::symlink_status(root, ec);
    detail::need(!ec && disk::is_directory(root_status) && !disk::is_symlink(root_status),
                 "v3 namespace unavailable");
    uint64_t total = 0;
    size_t objects = 0;
    auto add = [&](uint64_t amount) {
        detail::need(amount <= uint64_t(max_amount) - total, "v3 accounting overflow");
        total += amount;
    };
    auto directory_size = [&](const disk::path & path) {
        struct stat st {};
        detail::need(::lstat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode) && st.st_blocks >= 0,
                     "unsafe v3 directory");
        const auto blocks = uint64_t(st.st_blocks);
        detail::need(blocks <= uint64_t(max_amount) / 512, "v3 allocation overflow");
        add(blocks * 512);
    };
    auto size_file = [&](const disk::path & path) {
        struct stat st {};
        detail::need(::lstat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode) && st.st_nlink == 1 &&
                         st.st_size >= 0 && st.st_blocks >= 0,
                     "unsafe v3 file");
        const auto blocks = uint64_t(st.st_blocks);
        detail::need(blocks <= uint64_t(max_amount) / 512, "v3 allocation overflow");
        const auto size = uint64_t(st.st_size);
        add(std::max(size, blocks * 512));
        return size;
    };
    directory_size(root);
    for (disk::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        detail::need(++objects <= 50000, "v3 object count exceeded");
        const auto dir = it->path();
        const auto name = dir.filename().string();
        const bool staged = name.size() == 68 && name.compare(0, 4, "tmp-") == 0 &&
            detail::hex_name(name.substr(4));
        detail::need(staged || detail::hex_name(name), "unknown v3 object name");
        const auto dir_status = it->symlink_status(ec);
        detail::need(!ec && disk::is_directory(dir_status) && !disk::is_symlink(dir_status),
                     "unsafe v3 object directory");
        directory_size(dir);
        std::set<std::string> found;
        std::map<std::string, uint64_t> sizes;
        for (disk::directory_iterator file_it(dir, ec), file_end; !ec && file_it != file_end;
             file_it.increment(ec)) {
            const auto entry = file_it->path().filename().string();
            detail::need(found.size() < snap::kMaxChunks + 1, "v3 object entry limit exceeded");
            const bool manifest = entry == "manifest-v2.bin";
            bool chunk = false;
            if (entry.size() >= 6 && entry.compare(0, 2, "n-") == 0 &&
                entry.compare(entry.size() - 3, 3, ".kv") == 0) {
                const auto number = entry.substr(2, entry.size() - 5);
                chunk = !number.empty() && (number.size() == 1 || number[0] != '0') &&
                    number.size() <= 4 &&
                    std::all_of(number.begin(), number.end(),
                                [](char c) { return c >= '0' && c <= '9'; }) &&
                    std::stoul(number) < snap::kMaxChunks;
            }
            detail::need(manifest || chunk, "unknown v3 object entry");
            detail::need(found.insert(entry).second, "duplicate v3 entry");
            const auto bytes = size_file(file_it->path());
            detail::need(bytes <= (manifest ? snap::kMaxManifestBytes : snap::kChunkBytes),
                         "v3 file extent invalid");
            sizes.emplace(entry, bytes);
        }
        detail::need(!ec, "v3 object listing failed");
        if (staged) continue;
        detail::need(found.count("manifest-v2.bin") != 0, "published v3 manifest missing");
        std::vector<uint8_t> bytes(size_t(sizes.at("manifest-v2.bin")));
        std::ifstream input(dir / "manifest-v2.bin", std::ios::binary);
        detail::need(bool(input) &&
                         (bytes.empty() ||
                          bool(input.read(reinterpret_cast<char *>(bytes.data()),
                                          std::streamsize(bytes.size())))),
                     "v3 manifest read failed");
        snap::Manifest manifest;
        std::string why;
        std::vector<uint8_t> canonical;
        detail::need(snap::decode(bytes, manifest, &why) && snap::encode(manifest, canonical, &why) &&
                         bytes == canonical && rkva_snapshot::hex(manifest.host_state_sha256) == name &&
                         snap::validate(manifest, manifest.geometry, manifest.descriptor,
                                        manifest.live_rows, manifest.generation,
                                        manifest.host_state_sha256, &why),
                     "published v3 manifest invalid");
        detail::need(found.size() == manifest.chunks.size() + 1,
                     "published v3 chunk count mismatch");
        std::array<uint8_t, 65536> buffer {};
        for (size_t i = 0; i < manifest.chunks.size(); ++i) {
            const auto entry = "n-" + std::to_string(i) + ".kv";
            const auto size = sizes.find(entry);
            detail::need(size != sizes.end() && size->second == manifest.chunks[i].bytes,
                         "published v3 chunk missing or truncated");
            std::ifstream chunk(dir / entry, std::ios::binary);
            detail::need(bool(chunk), "published v3 chunk unavailable");
            rkva_snapshot::Sha256 hash;
            uint64_t remaining = size->second;
            while (remaining) {
                const auto count = size_t(std::min<uint64_t>(remaining, buffer.size()));
                detail::need(bool(chunk.read(reinterpret_cast<char *>(buffer.data()),
                                             std::streamsize(count))),
                             "published v3 chunk read failed");
                hash.update(buffer.data(), count);
                remaining -= count;
            }
            detail::need(hash.finish() == manifest.chunks[i].digest,
                         "published v3 chunk corrupt");
        }
    }
    detail::need(!ec, "v3 namespace listing failed");
    return {total, objects};
}

#else

// The Windows FileStandardInfo implementation and the SQLite admission coupling
// remain deferred (Phase 3). Failing here keeps every caller fail-closed.
inline usage scan(const std::filesystem::path & parent, uint64_t directory_margin = 0) {
    (void) parent;
    (void) directory_margin;
    throw std::runtime_error("physical v3 accounting unavailable on this platform");
}

#endif

} // namespace rkva_v3_quota
