#pragma once

// Internal helpers shared by the fail-closed production writer/verifier and by
// the test-only staging seam. This is not a public interface and it does not
// make any operation reachable. The production translation unit includes it;
// tests/test-rkva-managed-object-v3-staging.cpp includes it indirectly through
// rkva-managed-object-v3-write.inc.
//
// Keeping the canonical manifest/path checks in one place prevents the test
// seam from drifting away from the production fail-closed path. None of these
// helpers performs a durable operation or authorizes publication.

#include "rkva-managed-object-v3.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <set>
#include <stdexcept>
#include <system_error>

namespace rkva_managed_object_v3 {
namespace snap = rkva_native_snapshot;
namespace detail {

inline void need(bool value, const std::string & error) {
    if (!value) throw std::runtime_error(error);
}

inline bool nonzero(const rkva_snapshot::Digest & x) {
    return std::any_of(x.begin(), x.end(), [](uint8_t b) { return b != 0; });
}

inline bool valid_pin(const pin & p) {
    return nonzero(p.owner) && nonzero(p.transaction) && nonzero(p.reference) && nonzero(p.key);
}

inline fs::file_status path_status(const fs::path & path) {
    std::error_code ec;
    const auto s = fs::symlink_status(path, ec);
    if (ec == std::errc::no_such_file_or_directory) return fs::file_status(fs::file_type::not_found);
    need(!ec, "object path inspection failed");
    return s;
}

inline void validate_root(const fs::path & root) {
    need(!root.empty() && root.filename() == "rkva-snapshots-v3", "v3 namespace required");
    const auto s = path_status(root);
    need(fs::is_directory(s) && !fs::is_symlink(s), "invalid v3 store root");
    // Roots nested in legacy namespaces are never used, even if the final name
    // happens to be v3. Trusted path ancestors are still a caller obligation.
    for (const auto & part : root.lexically_normal())
        need(part != "rkva-snapshots-v1" && part != "rkva-snapshots-v2",
             "legacy namespace forbidden");
}

inline void hold(pin_proof * proof, const pin & p, const expected & e,
                 const std::shared_ptr<rkva_managed_store::lease> & store) {
    need(proof && proof->pending(p, e, store), "v3 durable pending pin unavailable");
}

inline void regular(const fs::path & path, uint64_t size) {
    const auto s = path_status(path);
    need(fs::is_regular_file(s) && !fs::is_symlink(s), "v3 object entry is not a regular file");
    std::error_code ec;
    const auto actual = fs::file_size(path, ec);
    need(!ec && actual == size, "v3 object extent differs");
    const auto links = fs::hard_link_count(path, ec);
    need(!ec && links == 1, "v3 object file has unsafe hardlinks");
}

inline std::set<std::string> names(const fs::path & dir) {
    std::set<std::string> found;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        need(found.size() < snap::kMaxChunks + 1, "v3 object entry limit exceeded");
        need(found.insert(it->path().filename().string()).second, "duplicate v3 object entry");
    }
    need(!ec, "v3 object listing failed");
    return found;
}

inline std::vector<uint8_t> load_manifest(const fs::path & path) {
    const auto s = path_status(path);
    need(fs::is_regular_file(s) && !fs::is_symlink(s), "invalid v3 manifest");
    std::error_code ec;
    auto n = fs::file_size(path, ec);
    need(!ec && n <= snap::kMaxManifestBytes, "v3 manifest too large");
    regular(path, n);
    std::vector<uint8_t> bytes(size_t(n), uint8_t{});
    std::ifstream input(path, std::ios::binary);
    need(bool(input) && (bytes.empty() ||
                         bool(input.read(reinterpret_cast<char *>(bytes.data()),
                                         std::streamsize(bytes.size())))),
         "v3 manifest read failed");
    regular(path, n);
    return bytes;
}

inline void validate_directory(const fs::path & dir, const pin & p, const expected & e,
                               std::string & digest) {
    const auto s = path_status(dir);
    need(fs::is_directory(s) && !fs::is_symlink(s), "invalid v3 object directory");
    const auto bytes = load_manifest(dir / "manifest-v2.bin");
    snap::Manifest m;
    std::string why;
    if (!snap::decode(bytes, m, &why)) throw std::runtime_error("v3 manifest decode failed: " + why);
    std::vector<uint8_t> canonical;
    need(snap::encode(m, canonical, &why) && canonical == bytes, "v3 manifest is noncanonical");
    if (!snap::validate(m, e.geometry, e.descriptor, e.rows, e.generation, p.key.data(), &why))
        throw std::runtime_error("v3 manifest identity mismatch: " + why);
    need(m.layers.size() == e.layers.size(), "v3 layer coverage count differs");
    for (size_t i = 0; i < m.layers.size(); i++) {
        const auto & a = m.layers[i];
        const auto & b = e.layers[i];
        need(a.tail_begin == b.tail_begin && a.stage_begin == b.stage_begin &&
                 a.record_groups == b.record_groups && a.ready == b.ready,
             "v3 layer coverage differs");
    }
    std::set<std::string> expected_names{"manifest-v2.bin"};
    for (size_t i = 0; i < m.chunks.size(); i++)
        expected_names.insert("n-" + std::to_string(i) + ".kv");
    need(names(dir) == expected_names, "v3 object has missing or unexpected entries");
    std::array<uint8_t, 64 * 1024> buffer{};
    for (size_t i = 0; i < m.chunks.size(); i++) {
        const auto & c = m.chunks[i];
        const auto path = dir / ("n-" + std::to_string(i) + ".kv");
        need(c.bytes && c.bytes <= snap::kChunkBytes, "v3 chunk extent invalid");
        regular(path, c.bytes);
        std::ifstream input(path, std::ios::binary);
        need(bool(input), "v3 chunk cannot be read");
        rkva_snapshot::Sha256 hash;
        uint64_t remaining = c.bytes;
        while (remaining) {
            const size_t n = size_t(std::min<uint64_t>(remaining, buffer.size()));
            need(bool(input.read(reinterpret_cast<char *>(buffer.data()), std::streamsize(n))),
                 "v3 chunk read failed");
            hash.update(buffer.data(), n);
            remaining -= n;
        }
        regular(path, c.bytes);
        need(hash.finish() == c.digest, "v3 chunk checksum mismatch");
    }
    need(names(dir) == expected_names &&
             load_manifest(dir / "manifest-v2.bin") == bytes,
         "v3 object changed during verification");
    const auto hash = rkva_snapshot::digest(bytes.data(), bytes.size());
    digest = rkva_snapshot::hex(hash.data());
}

} // namespace detail
} // namespace rkva_managed_object_v3
