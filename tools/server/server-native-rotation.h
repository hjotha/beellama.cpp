#pragma once

#include "server-native-refs.h"
#include "server-route-state.h"
#include "ggml-remote-attn.h"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace server_native_rotation {

inline bool enabled(const char * option, bool peer_supported) {
    return peer_supported && option && std::strcmp(option, "1") == 0;
}

struct candidate {
    std::string key;
    std::array<uint8_t, 32> bytes{};
    server_native_refs::retire_plan retirement;
    std::string timestamp;
    std::vector<std::string> pending_files;
};

inline std::array<uint8_t, 32> decode_key(const std::string & key) {
    std::array<uint8_t, 32> result{};
    for (size_t i = 0; i < result.size(); ++i) {
        const auto value = [](char c) { return uint8_t(c <= '9' ? c - '0' : c - 'a' + 10); };
        result[i] = uint8_t((value(key[2*i]) << 4) | value(key[2*i + 1]));
    }
    return result;
}

inline std::vector<candidate> candidates(const std::vector<server_native_refs::record> & journal,
                                         const std::string & incoming) {
    std::set<std::string> seen;
    std::vector<candidate> result;
    for (const auto & entry : journal) {
        if (entry.key == incoming || !seen.insert(entry.key).second) continue;
        auto plan = server_native_refs::plan_retire(entry.key, journal, false,
                                                    server_native_refs::timestamp_now());
        std::vector<std::string> pending_files;
        if (plan.refusal == server_native_refs::retire_refusal::pending) {
            std::set<std::string> pending, committed;
            bool pinned = false;
            for (const auto & ref : journal) if (ref.key == entry.key) {
                if (ref.st == server_native_refs::state::pinned) pinned = true;
                if (ref.st == server_native_refs::state::pending) pending.insert(ref.host_file);
                if (ref.st == server_native_refs::state::committed) committed.insert(ref.host_file);
            }
            if (!pinned && !pending.empty()) {
                pending_files.assign(pending.begin(), pending.end());
                plan.host_files.assign(committed.begin(), committed.end());
                plan.entry.host_file = !plan.host_files.empty() ? plan.host_files.back() : pending_files.back();
                plan.execute = true;
                plan.refusal = server_native_refs::retire_refusal::none;
            }
        }
        if (!plan.execute && plan.refusal != server_native_refs::retire_refusal::already_retired) continue;
        result.push_back({entry.key, decode_key(entry.key), std::move(plan), entry.timestamp, std::move(pending_files)});
    }
    std::sort(result.begin(), result.end(), [](const candidate & a, const candidate & b) {
        if (a.retirement.execute != b.retirement.execute) return a.retirement.execute;
        return a.timestamp != b.timestamp ? a.timestamp < b.timestamp : a.key < b.key;
    });
    return result;
}

// Status matches are in remote age order. A missing remote object can be
// retired first when only the host quota is exhausted.
inline int select(const ggml_remote_native_store_info & info, size_t candidate_count, bool host_fits) {
    if (candidate_count > 63 || info.matches > 64) return -2;
    std::array<bool, 64> present{};
    for (uint32_t i = 0; i < info.matches; ++i) {
        if (info.match[i].key_index > candidate_count) return -2;
        present[info.match[i].key_index] = true;
    }
    const bool remote_fits = present[0] ||
        (info.objects < info.max_objects && info.incoming_bytes <= info.limit_bytes &&
         info.used_bytes <= info.limit_bytes - info.incoming_bytes);
    if (remote_fits && host_fits) return -1;
    if (!present[0] && info.incoming_bytes > info.limit_bytes) return -2;
    if (!host_fits) {
        for (size_t i = 0; i < candidate_count; ++i) if (!present[i + 1]) return int(i);
    }
    for (uint32_t i = 0; i < info.matches; ++i) {
        const uint32_t index = info.match[i].key_index;
        if (index) return int(index - 1);
    }
    return -2;
}

enum class retirement_result { retired, skipped, failed };

inline bool sync_directory(const std::string & directory) {
#ifndef _WIN32
    int flags = O_RDONLY;
#ifdef O_DIRECTORY
    flags |= O_DIRECTORY;
#endif
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int fd = ::open(directory.c_str(), flags);
    if (fd < 0) return false;
    const bool ok = ::fsync(fd) == 0;
    return ::close(fd) == 0 && ok;
#else
    (void) directory;
    return true;
#endif
}

template <class Verify, class Preflight, class Remote, class Sync>
retirement_result retire(const candidate & victim, const std::string & store_dir,
                         server_route_state_store_lock & store, Verify && verify, Preflight && preflight,
                         Remote && remote, Sync && sync,
                         size_t & removed, std::string & error) {
    removed = 0;
    const bool already_retired = victim.retirement.refusal == server_native_refs::retire_refusal::already_retired;
    if ((!victim.retirement.execute && !already_retired) || !server_native_refs::is_hex_key(victim.key) ||
        victim.retirement.entry.key != victim.key) return retirement_result::skipped;
    for (const auto & name : victim.pending_files) {
        if (!server_native_refs::valid_host_file(name)) return retirement_result::skipped;
        const auto path = std::filesystem::u8path(store_dir) / name;
        if (!store.owns_exclusive_for(path.string())) return retirement_result::skipped;
        std::error_code ec;
        const auto stat = std::filesystem::symlink_status(path, ec);
        if ((ec && ec != std::errc::no_such_file_or_directory) ||
            (!ec && stat.type() != std::filesystem::file_type::not_found))
            return retirement_result::skipped;
    }
    std::vector<std::unique_ptr<server_route_state_lease>> leases;
    std::vector<std::filesystem::path> paths;
    for (const auto & name : victim.retirement.host_files) {
        if (!server_native_refs::valid_host_file(name)) return retirement_result::skipped;
        const auto path = std::filesystem::u8path(store_dir) / name;
        if (!store.owns_exclusive_for(path.string())) return retirement_result::skipped;
        std::error_code ec;
        const auto stat = std::filesystem::symlink_status(path, ec);
        if (ec == std::errc::no_such_file_or_directory ||
            (!ec && stat.type() == std::filesystem::file_type::not_found)) continue;
        if (ec || !std::filesystem::is_regular_file(stat) || std::filesystem::is_symlink(stat))
            return retirement_result::skipped;
        auto lease = std::make_unique<server_route_state_lease>(path.string(),
                server_route_state_lock_mode::eviction);
        if (!lease->acquired()) return retirement_result::skipped;
        const auto links = std::filesystem::hard_link_count(path, ec);
        if (ec || links != 1) return retirement_result::skipped;
        for (const char * suffix : {".logits", ".meta", ".ckpt", ".ckpts"}) {
            ec.clear();
            const auto sidecar = std::filesystem::symlink_status(path.string() + suffix, ec);
            if ((ec && ec != std::errc::no_such_file_or_directory) ||
                (!ec && sidecar.type() != std::filesystem::file_type::not_found))
                return retirement_result::skipped;
        }
        if (!verify(path, victim.key)) return retirement_result::skipped;
        paths.push_back(path);
        leases.push_back(std::move(lease));
    }
    if (!preflight(victim.bytes.data(), paths.empty())) return retirement_result::skipped;
    // Remove host paths first so an interrupted retirement cannot leave a
    // restorable host file pointing at a deleted Xbox object.
    for (const auto & path : paths) {
        std::error_code ec;
        if (!std::filesystem::remove(path, ec) || ec) {
            error = "host removal failed: " + ec.message();
            return retirement_result::failed;
        }
        ++removed;
    }
    if (!sync(store_dir)) {
        error = "host directory sync failed; Xbox object retained for retry";
        return retirement_result::failed;
    }
    if (!remote(victim.bytes.data())) {
        error = "Xbox retirement failed; journal retained for retry";
        return retirement_result::failed;
    }
    const auto held = [&store](const std::string & path) { return store.owns_exclusive_for(path); };
    if (!already_retired && !server_native_refs::append(store_dir, victim.retirement.entry, held, &error))
        return retirement_result::failed;
    return retirement_result::retired;
}
} // namespace server_native_rotation
