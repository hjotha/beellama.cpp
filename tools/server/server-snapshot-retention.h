#pragma once

// Reversible staging for independently classified, local-only snapshot units.
// The caller MUST hold the exclusive route-state store lock and supply proof
// of the existing eviction leases and a conservative classifier. This utility does
// not classify manifests or remote companions and never deletes remote data.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace server_snapshot_retention {
namespace fs = std::filesystem;

enum class classification { unknown, local_only, native_pair };
enum class outcome { rejected, rolled_back, rollback_incomplete, committed, cleanup_incomplete };
struct result {
    outcome state = outcome::rejected;
    fs::path staging_directory;
    std::string reason;
    std::vector<fs::path> pending; // staging paths still present after an incomplete operation
};

// Implementations return errors via error_code, including injected failures.
struct filesystem_ops {
    bool create_directory(const fs::path & p, std::error_code & ec) { return fs::create_directory(p, ec); }
    void rename(const fs::path & a, const fs::path & b, std::error_code & ec) { fs::rename(a, b, ec); }
    bool remove(const fs::path & p, std::error_code & ec) { return fs::remove(p, ec); }
};

struct item { fs::path original, staged; };

// Call under the exclusive store lock on every quota admission, including
// admissions needing no victims. An orphaned staging directory still occupies
// disk even if ordinary snapshot enumeration excludes temporary names.
inline bool has_unresolved_staging(const fs::path & dir, std::error_code & ec) {
    ec.clear();
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().filename().string().find("retention.tmp-") == 0) return true;
    }
    return false;
}

// `store.owns_exclusive_for(dir / filename)` must prove the real store lock is held;
// `has_lease(path)` must confirm that the caller retains the acquired exclusive
// eviction lease for this exact candidate throughout the whole call, and
// `classify(path)` must return local_only only when an authoritative,
// integrity-checked manifest proves the state has no native companion, or a
// recognized sidecar is proven to belong to that state. Classifying the state
// alone does not authorize removing any adjacent file.
// Unknown and native_pair are refused. These are trust contracts of the caller,
// not inferred from names or an untrusted marker. The classifier runs under both
// store lock and lease; it must not reacquire the shared store lock.
//
// `publish()` runs after all renames while every lock is still held. Its false
// result or exception rolls all renames back. Its true result is the commit
// point: subsequent cleanup failures return cleanup_incomplete, NEVER rollback
// the already-published object. A caller must propagate that distinction rather
// than translating cleanup_incomplete into a publication failure.
//
// No journal or fsync is implemented. A process/power failure can leave staged
// files and interrupted publication; these must be inspected manually. Do not
// scan/delete staging directories automatically. Only cooperating processes in
// a trusted, exclusively administered store are covered.
template <class Store, class LeaseCheck, class Classify, class Publish, class Ops>
result run(const fs::path & dir, Store & store, const std::vector<std::string> & states,
           LeaseCheck has_lease, Classify classify, Publish publish, Ops & ops) {
    result out;
    if (!store.owns_exclusive_for((dir / ".retention-authority").string())) { out.reason = "exclusive store lock required"; return out; }
    if (states.empty() || states.size() > 4096) { out.reason = "invalid victim count"; return out; }
    std::vector<item> entries;
    std::set<std::string> used;
    std::error_code ec;
    bool committed = false;
    size_t renamed = 0;
    auto rollback = [&]() {
        bool failed = false;
        for (size_t i = renamed; i > 0; --i) {
            auto & e = entries[i - 1];
            try {
            std::error_code check;
            const auto status = fs::symlink_status(e.original, check);
            if (check && check != std::errc::no_such_file_or_directory) { failed = true; continue; }
            if (!check && status.type() != fs::file_type::not_found) { failed = true; continue; }
            check.clear(); ops.rename(e.staged, e.original, check);
            if (check) { failed = true; }
            } catch (...) { failed = true; }
        }
        if (!failed && !out.staging_directory.empty()) {
            std::error_code cleanup;
            try {
                if (!ops.remove(out.staging_directory, cleanup) || cleanup) failed = true;
            } catch (...) { failed = true; }
        }
        out.state = failed ? outcome::rollback_incomplete : outcome::rolled_back;
        if (failed) {
            for (size_t i = 0; i < renamed; ++i) {
                std::error_code exists_ec;
                if (fs::exists(entries[i].staged, exists_ec) || exists_ec) out.pending.push_back(entries[i].staged);
            }
        }
    };
    try {
        entries.reserve(states.size() * 5);
        out.pending.reserve(states.size() * 5);
        // An interrupted prior operation may still consume quota. Never start
        // a new one while it is present; the store planner must independently
        // enforce this even when it would otherwise need no victims.
        if (has_unresolved_staging(dir, ec)) { out.reason = "unresolved retention staging exists"; return out; }
        if (ec) { out.reason = "store scan failed"; return out; }
        for (const auto & name : states) {
            const fs::path base(name);
            const auto extension = base.extension().string();
            const bool reserved = extension == ".logits" || extension == ".meta" ||
                extension == ".ckpt" || extension == ".ckpts" || extension == ".lease" ||
                extension == ".ref" || extension == ".lock" || extension == ".tmp";
            if (name.empty() || base.is_absolute() || base.filename() != base || name == "." || name == ".." ||
                    reserved || name.find(".tmp-") != std::string::npos || !used.insert(name).second) {
                out.reason = "invalid or duplicate snapshot filename"; return out;
            }
            if (!has_lease(dir / base)) { out.reason = "snapshot eviction lease unavailable"; return out; }
            for (const auto & suffix : {std::string(), std::string(".logits"), std::string(".meta"),
                                        std::string(".ckpt"), std::string(".ckpts")}) {
                const std::string component = name + suffix;
                if (suffix.empty() == false && !used.insert(component).second) {
                    out.reason = "duplicate snapshot component"; return out;
                }
                const fs::path original = dir / component;
                ec.clear(); const auto status = fs::symlink_status(original, ec);
                if (ec == std::errc::no_such_file_or_directory || (!ec && status.type() == fs::file_type::not_found)) {
                    if (suffix.empty()) { out.reason = "snapshot missing"; return out; }
                    continue;
                }
                if (ec || !fs::is_regular_file(status)) { out.reason = "component is not a regular file"; return out; }
                ec.clear(); const auto links = fs::hard_link_count(original, ec);
                if (ec || links != 1) { out.reason = "component has unsafe hard links"; return out; }
                if (classify(original) != classification::local_only) {
                    out.reason = "snapshot component is unknown or has a native companion"; return out;
                }
                entries.push_back({original, fs::path(component)});
            }
        }
        // Reserve all metadata before the first mutation. mkdir is atomic and
        // cannot overwrite an existing staging directory, even after restart.
        static std::atomic<uint64_t> nonce{0};
        for (int attempt = 0; attempt < 64; ++attempt) {
            const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
            const auto name = std::string("retention.tmp-") + std::to_string(tick) + "-" +
                std::to_string(nonce.fetch_add(1, std::memory_order_relaxed));
            out.staging_directory = dir / name;
            ec.clear();
            if (ops.create_directory(out.staging_directory, ec)) break;
            if (ec && ec != std::errc::file_exists) { out.reason = "staging directory creation failed"; out.staging_directory.clear(); return out; }
            out.staging_directory.clear();
        }
        if (out.staging_directory.empty()) { out.reason = "staging directory collision"; return out; }
        for (auto & e : entries) e.staged = out.staging_directory / e.staged;
        for (const auto & e : entries) {
            ec.clear(); ops.rename(e.original, e.staged, ec);
            if (ec) { out.reason = "staging rename failed: " + ec.message(); rollback(); return out; }
            ++renamed;
        }
        if (!publish()) { out.reason = "publication refused"; rollback(); return out; }
        committed = true;
        out.state = outcome::committed;
        for (const auto & e : entries) {
            ec.clear();
            if (!ops.remove(e.staged, ec) || ec) {
                out.state = outcome::cleanup_incomplete;
                out.pending.push_back(e.staged);
            }
        }
        if (out.state == outcome::committed) {
            ec.clear();
            if (!ops.remove(out.staging_directory, ec) || ec) out.state = outcome::cleanup_incomplete;
        }
        if (out.state == outcome::cleanup_incomplete) out.reason = "committed; staging cleanup incomplete";
        return out;
    } catch (...) {
        if (committed) {
            out.state = outcome::cleanup_incomplete; out.reason = "committed; staging cleanup threw";
            for (const auto & e : entries) {
                std::error_code check;
                if (fs::exists(e.staged, check) || check) out.pending.push_back(e.staged);
            }
        } else if (!out.staging_directory.empty()) {
            out.reason = "staging or publication threw";
            rollback();
        } else {
            out.reason = "preflight threw";
        }
        return out;
    }
}
} // namespace server_snapshot_retention
