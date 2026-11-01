#pragma once

// Test-only, compile-guarded fault injector for the v3 durable-filesystem
// contract (sync_file / sync_directory / rename_noreplace_and_sync).
//
// This header is NEVER included by production code. The test target must
// define RKVA_DURABLE_FAULT_TESTING; including it without that macro is a
// build error so an accidental production include fails closed. Even with the
// macro defined, nothing here is linked into llama-server, the UWP app or the
// Phase 2/3/4 libraries: only tests/test-rkva-durable-fault.cpp compiles it.
//
// Scope and honest limits:
//  * It implements the abstract rkva_managed_object_v3::durable_filesystem
//    interface over a real POSIX directory, so the injected failures act on
//    real bytes, real directory entries and real fsync/rename calls.
//  * It does not make the closed v3 writer reachable. write()/verify() still
//    refuse before any backend call because no concrete authenticated
//    pin/metadata proof exists (see docs/rkva-durable-fault-injection.md for
//    the test-seam proposal and the exact line references).
//  * The "ack lost" and "crash after apply" modes model a rename that reached
//    the directory entry but whose caller may not claim durability. They are
//    deterministic models, not real power-loss evidence.
//
// Fault queues are consumed one entry per call of the matching kind. An empty
// queue means "no injected fault" for that call, so a test can schedule a
// failure at a precise position and let every other call behave normally.

#ifndef RKVA_DURABLE_FAULT_TESTING
#error "tests/rkva-durable-fault.h is test-only; define RKVA_DURABLE_FAULT_TESTING"
#endif

#include "../tools/xbox/uwp/rkva/rkva-managed-object-v3.h"

#include <cerrno>
#include <cstring>
#include <deque>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace rkva_durable_fault {

namespace fs = std::filesystem;
namespace v3 = rkva_managed_object_v3;

enum class fault {
    none,               // no injection
    eio,                // the operation fails with EIO
    refuse_existing,    // no-replace: a competing object already exists
    os_error,           // rename fails with a benign error, nothing applied
    rename_eio,         // rename reports EIO; the applied state is unknown
    ack_lost,           // rename applied, caller sees "uncertain"
    crash_after_apply,  // rename applied and the model process dies
};

inline const char * name(fault value) {
    switch (value) {
        case fault::none: return "none";
        case fault::eio: return "eio";
        case fault::refuse_existing: return "refuse_existing";
        case fault::os_error: return "os_error";
        case fault::rename_eio: return "rename_eio";
        case fault::ack_lost: return "ack_lost";
        case fault::crash_after_apply: return "crash_after_apply";
    }
    return "unknown";
}

// One queue per operation kind. Order within a queue is the call order.
struct plan {
    bool qualified = true;
    std::deque<fault> file_sync;
    std::deque<fault> dir_sync;
    std::deque<fault> rename;
    long file_sync_calls = 0;
    long dir_sync_calls = 0;
    long rename_calls = 0;
    bool crashed = false;

    static fault take(std::deque<fault> & queue) {
        if (queue.empty()) {
            return fault::none;
        }
        const fault value = queue.front();
        queue.pop_front();
        return value;
    }
};

namespace detail {

inline bool is_regular_nofollow(int fd) {
    struct stat st {};
    return ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_nlink == 1;
}

inline bool is_directory_nofollow(int fd) {
    struct stat st {};
    return ::fstat(fd, &st) == 0 && S_ISDIR(st.st_mode);
}

inline bool destination_exists(const fs::path & path) {
    std::error_code ec;
    const auto status = fs::symlink_status(path, ec);
    if (ec && ec != std::errc::no_such_file_or_directory) {
        return true; // unknown destination state: refuse, never overwrite
    }
    return status.type() != fs::file_type::not_found;
}

// Linux renameat2(RENAME_NOREPLACE). Returns 0, EEXIST, or a real errno.
// Filesystems without RENAME_NOREPLACE fall back to a checked rename; the
// harness is CPU test code, not a durability implementation.
inline int rename_noreplace(const fs::path & source, const fs::path & destination) {
#if defined(SYS_renameat2)
    if (::syscall(SYS_renameat2, AT_FDCWD, source.c_str(), AT_FDCWD,
                  destination.c_str(), 1u /* RENAME_NOREPLACE */) == 0) {
        return 0;
    }
    if (errno != ENOSYS && errno != EINVAL) {
        return -1;
    }
#endif
    if (destination_exists(destination)) {
        errno = EEXIST;
        return -1;
    }
    return ::rename(source.c_str(), destination.c_str());
}

} // namespace detail

class injected_filesystem final : public v3::durable_filesystem {
public:
    injected_filesystem(std::weak_ptr<rkva_managed_store::lease> lease,
                        fs::path root, plan * schedule)
        : durable_filesystem(std::move(lease), std::move(root)),
          schedule_(schedule) {}

    bool qualified() const override { return schedule_ != nullptr && schedule_->qualified; }

    bool sync_file(const fs::path & path, std::string & error) override {
        if (schedule_ == nullptr) {
            error = "no fault schedule";
            return false;
        }
        ++schedule_->file_sync_calls;
        const fault injected = plan::take(schedule_->file_sync);
        if (injected == fault::eio) {
            error = "injected file sync failure";
            return false;
        }
        if (injected != fault::none) {
            error = std::string("injected unsupported file-sync fault: ") + name(injected);
            return false;
        }
        const int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
        if (fd < 0) {
            error = std::string("file sync open failed: ") + std::strerror(errno);
            return false;
        }
        const bool regular = detail::is_regular_nofollow(fd);
        const bool synced = regular && ::fsync(fd) == 0;
        const int saved = errno;
        ::close(fd);
        if (!regular) {
            error = "file sync target is not a regular single-link file";
            return false;
        }
        if (!synced) {
            error = std::string("file sync failed: ") + std::strerror(saved);
            return false;
        }
        error.clear();
        return true;
    }

    bool sync_directory(const fs::path & path, std::string & error) override {
        if (schedule_ == nullptr) {
            error = "no fault schedule";
            return false;
        }
        ++schedule_->dir_sync_calls;
        const fault injected = plan::take(schedule_->dir_sync);
        if (injected == fault::eio) {
            error = "injected directory sync failure";
            return false;
        }
        if (injected != fault::none) {
            error = std::string("injected unsupported directory-sync fault: ") + name(injected);
            return false;
        }
        const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (fd < 0) {
            error = std::string("directory sync open failed: ") + std::strerror(errno);
            return false;
        }
        const bool directory = detail::is_directory_nofollow(fd);
        const bool synced = directory && ::fsync(fd) == 0;
        const int saved = errno;
        ::close(fd);
        if (!directory) {
            error = "directory sync target is not a directory";
            return false;
        }
        if (!synced) {
            error = std::string("directory sync failed: ") + std::strerror(saved);
            return false;
        }
        error.clear();
        return true;
    }

    v3::move_result rename_noreplace_and_sync(const fs::path & source,
                                              const fs::path & destination,
                                              std::string & error) override {
        if (schedule_ == nullptr) {
            error = "no fault schedule";
            return v3::move_result::no_change;
        }
        ++schedule_->rename_calls;
        const fault injected = plan::take(schedule_->rename);
        if (injected == fault::refuse_existing || detail::destination_exists(destination)) {
            // A real no-replace move must leave both entries untouched. The
            // injected variant also models "another writer won the race".
            error = "injected or actual rename destination already exists";
            return v3::move_result::no_change;
        }
        if (injected == fault::os_error) {
            error = "injected rename failure";
            return v3::move_result::no_change;
        }
        if (injected == fault::eio) {
            error = "unsupported rename fault: eio; use rename_eio";
            return v3::move_result::no_change;
        }
        if (injected == fault::rename_eio) {
            // A rename that reports EIO may or may not have reached the
            // directory entry. This deterministic model leaves both entries in
            // place but still reports the caller-visible ambiguity: never
            // durable, and no automatic retry/cleanup.
            error = "injected rename EIO after indeterminate outcome";
            return v3::move_result::uncertain;
        }
        if (detail::rename_noreplace(source, destination) != 0) {
            const int saved = errno;
            if (saved == EEXIST) {
                error = "rename destination already exists";
                return v3::move_result::no_change;
            }
            error = std::string("rename failed: ") + std::strerror(saved);
            // The entry was not observed to move. EIO is the only code where
            // the platform contract leaves the outcome genuinely ambiguous.
            return saved == EIO ? v3::move_result::uncertain : v3::move_result::no_change;
        }
        if (injected == fault::ack_lost || injected == fault::crash_after_apply) {
            if (injected == fault::crash_after_apply) {
                schedule_->crashed = true;
            }
            // The rename reached the directory entry but no parent sync was
            // performed. The caller must never turn this into "durable".
            error = std::string("injected ") + name(injected) + " after rename apply";
            return v3::move_result::uncertain;
        }
        std::string sync_error;
        if (!sync_directory(destination.parent_path(), sync_error)) {
            error = "rename applied but parent sync failed: " + sync_error;
            return v3::move_result::uncertain;
        }
        error.clear();
        return v3::move_result::durable;
    }

private:
    plan * schedule_ = nullptr;
};

} // namespace rkva_durable_fault
