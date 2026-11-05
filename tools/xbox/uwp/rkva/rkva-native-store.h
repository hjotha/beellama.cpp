// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include "rkva-native-snapshot.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Native v2 is a trusted-local-writer store. Every writer must use this guard;
// it does not exclude older binaries or outside filesystem modification.
namespace rkva_native_store {
namespace fs = std::filesystem;

constexpr uint64_t byte_limit = 8ull << 30;
constexpr uint64_t object_limit = (4ull << 30) + (1ull << 20);
constexpr uint32_t object_count_limit = 8;
constexpr uint32_t file_count_limit = 4097;

inline bool opaque_name(const std::string& name) {
    return name.size() == 64 && name.find_first_not_of("0123456789abcdef") == std::string::npos;
}

// Atomic create-directory is the cross-process exclusion primitive. A crash
// leaves the lock behind: never infer ownership or remove it automatically.
class WriteGuard {
public:
    explicit WriteGuard(const fs::path& root) : lock_(root / ".native-writer-lock") {
        std::error_code ec;
        const bool created = fs::create_directory(lock_, ec);
        if (ec || !created)
            throw std::runtime_error("snapshot: native writer lock unavailable; explicit recovery may be required");
        owned_ = true;
    }
    WriteGuard(const WriteGuard&) = delete;
    WriteGuard& operator=(const WriteGuard&) = delete;
    ~WriteGuard() {
        if (owned_) {
            // remove only our empty lock directory; never recurse into it.
            std::error_code ignored;
            fs::remove(lock_, ignored);
        }
    }
private:
    fs::path lock_;
    bool owned_ = false;
};

struct entry {
    std::string name;
    uint64_t bytes;
    fs::file_time_type modified;
    bool published;
};

// Called with WriteGuard held. Staging entries remain accounted and protected.
inline std::vector<entry> inspect(const fs::path& root, uint64_t object_limit, uint32_t max_files) {
    std::vector<entry> result;
    size_t scanned = 0;
    for (const auto& item : fs::directory_iterator(root)) {
        if (++scanned > 64)
            throw std::runtime_error("snapshot: directory entry limit");
        const std::string name = item.path().filename().string();
        if (name == ".native-writer-lock")
            continue;
        const bool published = opaque_name(name);
        const bool temporary = name.size() == 68 && name.compare(0, 4, "tmp-") == 0 &&
                               opaque_name(name.substr(4));
        if (!published && !temporary)
            throw std::runtime_error("snapshot: unknown native store entry");
        if (fs::is_symlink(item.symlink_status()) || !item.is_directory())
            throw std::runtime_error("snapshot: invalid native store entry");
        uint64_t size = 0;
        uint32_t files = 0;
        for (const auto& file : fs::directory_iterator(item.path())) {
            if (++files > max_files || fs::is_symlink(file.symlink_status()) || !file.is_regular_file())
                throw std::runtime_error("snapshot: invalid native object entry");
            const uint64_t bytes = file.file_size();
            if (bytes > object_limit || size > object_limit - bytes)
                throw std::runtime_error("snapshot: oversized native object");
            size += bytes;
        }
        result.push_back({name, size, item.last_write_time(), published});
    }
    std::sort(result.begin(), result.end(), [](const entry& a, const entry& b) {
        return a.modified != b.modified ? a.modified < b.modified : a.name < b.name;
    });
    return result;
}

inline void check_room(const fs::path& root, uint64_t requested, uint64_t store_limit,
                       uint64_t object_limit, uint32_t max_snapshots, uint32_t max_files) {
    rkva_native_snapshot::StoreBudget used{};
    for (const auto& item : inspect(root, object_limit, max_files)) {
        if (used.bytes > UINT64_MAX - item.bytes || used.snapshots == UINT32_MAX)
            throw std::runtime_error("snapshot: native store size overflow");
        used.bytes += item.bytes;
        ++used.snapshots;
    }
    std::string error;
    if (!rkva_native_snapshot::admit_snapshot_store(used, requested, store_limit, max_snapshots, &error))
        throw std::runtime_error(error);
}

struct retirement_plan {
    bool present = false;
    std::vector<fs::path> files;
};

// A missing object is already retired; every present file is checked before
// the host removes its reference or the Xbox removes any object bytes.
inline retirement_plan plan_retirement(const fs::path& root, const std::string& key,
                                       uint64_t object_limit, uint32_t max_files, bool allow_partial) {
    if (!opaque_name(key)) throw std::runtime_error("snapshot: invalid retirement key");
    const auto entries = inspect(root, object_limit, max_files);
    const auto found = std::find_if(entries.begin(), entries.end(), [&](const entry& item) {
        return item.published && item.name == key;
    });
    if (found == entries.end()) {
        if (!allow_partial) throw std::runtime_error("snapshot: retirement object missing");
        return {};
    }
    retirement_plan plan;
    plan.present = true;
    const fs::path object = root / key;
    const fs::path manifest_path = object / "manifest-v2.bin";
    std::error_code ec;
    const auto manifest_status = fs::symlink_status(manifest_path, ec);
    if (ec == std::errc::no_such_file_or_directory ||
        (!ec && manifest_status.type() == fs::file_type::not_found)) {
        if (!allow_partial || !fs::is_empty(object))
            throw std::runtime_error("snapshot: retirement manifest missing");
        return plan;
    }
    if (ec || fs::is_symlink(manifest_status) || !fs::is_regular_file(manifest_status) ||
        fs::file_size(manifest_path) > rkva_native_snapshot::kMaxManifestBytes)
        throw std::runtime_error("snapshot: invalid retirement manifest");
    const auto manifest_size = fs::file_size(manifest_path);
    std::vector<uint8_t> bytes(size_t(manifest_size), 0);
    std::ifstream input(manifest_path, std::ios::binary);
    if (!input || !input.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size())))
        throw std::runtime_error("snapshot: unreadable retirement manifest");
    input.close();
    if (!input) throw std::runtime_error("snapshot: retirement manifest close failed");
    rkva_native_snapshot::Manifest manifest;
    std::string error;
    std::array<uint8_t, 32> digest_key{};
    const auto digit = [](char c) { return uint8_t(c <= '9' ? c - '0' : c - 'a' + 10); };
    for (size_t i = 0; i < digest_key.size(); ++i)
        digest_key[i] = uint8_t((digit(key[2*i]) << 4) | digit(key[2*i + 1]));
    if (!rkva_native_snapshot::decode(bytes, manifest, &error) ||
        !rkva_native_snapshot::validate(manifest, manifest.geometry, manifest.descriptor,
                                       manifest.live_rows, manifest.generation, digest_key.data(), &error))
        throw std::runtime_error("snapshot: retirement manifest identity mismatch");
    std::set<std::string> expected{"manifest-v2.bin"};
    for (size_t i = 0; i < manifest.chunks.size(); ++i) expected.insert("n-" + std::to_string(i) + ".kv");
    for (const auto& item : fs::directory_iterator(object)) {
        if (fs::is_symlink(item.symlink_status()) || !item.is_regular_file() ||
            !expected.count(item.path().filename().string()))
            throw std::runtime_error("snapshot: unknown retirement entry");
    }
    std::vector<char> buffer(1u << 20);
    for (size_t i = 0; i < manifest.chunks.size(); ++i) {
        const auto path = object / ("n-" + std::to_string(i) + ".kv");
        ec.clear();
        const auto stat = fs::symlink_status(path, ec);
        if (ec == std::errc::no_such_file_or_directory ||
            (!ec && stat.type() == fs::file_type::not_found)) {
            if (!allow_partial) throw std::runtime_error("snapshot: retirement chunk missing");
            continue;
        }
        if (ec || fs::is_symlink(stat) || !fs::is_regular_file(stat) ||
            fs::file_size(path) != manifest.chunks[i].bytes)
            throw std::runtime_error("snapshot: invalid retirement chunk");
        std::ifstream chunk(path, std::ios::binary);
        rkva_snapshot::Sha256 hash;
        uint64_t remaining = manifest.chunks[i].bytes;
        while (remaining) {
            const size_t n = size_t(std::min<uint64_t>(remaining, buffer.size()));
            if (!chunk.read(buffer.data(), std::streamsize(n)))
                throw std::runtime_error("snapshot: unreadable retirement chunk");
            hash.update(buffer.data(), n); remaining -= n;
        }
        if (hash.finish() != manifest.chunks[i].digest)
            throw std::runtime_error("snapshot: retirement chunk checksum mismatch");
        plan.files.push_back(path);
    }
    plan.files.push_back(manifest_path);
    return plan;
}

inline void erase_published(const fs::path& root, const std::string& key,
                            uint64_t object_limit, uint32_t max_files) {
    const auto plan = plan_retirement(root, key, object_limit, max_files, true);
    if (!plan.present) return;
    std::error_code ec;
    for (const auto& path : plan.files) {
        ec.clear();
        if (!fs::remove(path, ec) && ec != std::errc::no_such_file_or_directory && ec)
            throw std::runtime_error("snapshot: retirement unlink failed");
    }
    if (!fs::remove(root / key)) throw std::runtime_error("snapshot: retirement directory removal failed");
}

inline void create_stage(const fs::path& stage) {
    std::error_code ec;
    if (!fs::create_directory(stage, ec) || ec)
        throw std::runtime_error("snapshot: native staging directory already exists or cannot be created");
}
} // namespace rkva_native_store
