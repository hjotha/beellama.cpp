// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include "rkva-native-snapshot.h"
#include <filesystem>
#include <stdexcept>
#include <string>

// Native v2 is a trusted-local-writer store. Every writer must use this guard;
// it does not exclude older binaries or outside filesystem modification.
namespace rkva_native_store {
namespace fs = std::filesystem;

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

// Called with WriteGuard held. Both complete and incomplete generated objects
// count against quota. Never recover or delete a staging directory here.
inline void check_room(const fs::path& root, uint64_t requested, uint64_t store_limit,
                       uint64_t object_limit, uint32_t max_snapshots, uint32_t max_files) {
    rkva_native_snapshot::StoreBudget used{};
    size_t scanned = 0;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (++scanned > 64)
            throw std::runtime_error("snapshot: directory entry limit");
        const std::string name = entry.path().filename().string();
        if (name == ".native-writer-lock")
            continue;
        const bool published = opaque_name(name);
        const bool temporary = name.size() == 68 && name.compare(0, 4, "tmp-") == 0 &&
                               opaque_name(name.substr(4));
        if (!published && !temporary)
            throw std::runtime_error("snapshot: unknown native store entry");
        if (fs::is_symlink(entry.symlink_status()) || !entry.is_directory())
            throw std::runtime_error("snapshot: invalid native store entry");
        uint64_t size = 0;
        uint32_t files = 0;
        for (const auto& file : fs::directory_iterator(entry.path())) {
            if (++files > max_files || fs::is_symlink(file.symlink_status()) || !file.is_regular_file())
                throw std::runtime_error("snapshot: invalid native object entry");
            const uint64_t bytes = file.file_size();
            if (bytes > object_limit || size > object_limit - bytes)
                throw std::runtime_error("snapshot: oversized native object");
            size += bytes;
        }
        if (used.bytes > UINT64_MAX - size || used.snapshots == UINT32_MAX)
            throw std::runtime_error("snapshot: native store size overflow");
        used.bytes += size;
        ++used.snapshots;
    }
    std::string error;
    if (!rkva_native_snapshot::admit_snapshot_store(used, requested, store_limit, max_snapshots, &error))
        throw std::runtime_error(error);
}

inline void create_stage(const fs::path& stage) {
    std::error_code ec;
    if (!fs::create_directory(stage, ec) || ec)
        throw std::runtime_error("snapshot: native staging directory already exists or cannot be created");
}
} // namespace rkva_native_store
