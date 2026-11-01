#include "server-rkva-managed-authority.h"

#include <system_error>
#include <utility>

#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace rkva_managed_authority {

struct store::impl {
    std::filesystem::path parent;
    std::shared_ptr<rkva_managed_store::lease> lease;
#if defined(__linux__)
    int journal_fd = -1;
    int orchestrator_fd = -1;
    dev_t journal_dev = 0, orchestrator_dev = 0;
    ino_t journal_ino = 0, orchestrator_ino = 0;
    ~impl() {
        if (journal_fd >= 0) ::close(journal_fd);
        if (orchestrator_fd >= 0) ::close(orchestrator_fd);
    }
#endif
};

#if defined(__linux__)
namespace {

// Opens and exclusively locks `<parent>/<name>` without following a symlink and
// without ever waiting. Every failure is reported to the caller; nothing is
// repaired, created twice or retried.
bool acquire_lock(const std::filesystem::path & path, const char * busy, int & out_fd,
                  dev_t & out_dev, ino_t & out_ino, std::string & error) {
    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) {
        error = errno == ELOOP ? "managed lock is a symlink" : "managed lock unavailable";
        return false;
    }
    struct stat opened {};
    if (::fstat(fd, &opened) || !S_ISREG(opened.st_mode) || opened.st_nlink != 1 ||
        opened.st_uid != ::geteuid() || (opened.st_mode & 0022)) {
        ::close(fd);
        error = "invalid managed lock file";
        return false;
    }
    if (::flock(fd, LOCK_EX | LOCK_NB)) {
        const int saved = errno;
        ::close(fd);
        error = saved == EWOULDBLOCK || saved == EAGAIN ? busy : "managed lock failed";
        return false;
    }
    struct stat path_state {};
    if (::lstat(path.c_str(), &path_state) || !S_ISREG(path_state.st_mode) ||
        path_state.st_dev != opened.st_dev || path_state.st_ino != opened.st_ino) {
        ::close(fd);
        error = "managed lock replaced during acquisition";
        return false;
    }
    out_fd = fd;
    out_dev = opened.st_dev;
    out_ino = opened.st_ino;
    return true;
}

bool lock_still_held(int fd, const std::filesystem::path & path, dev_t dev, ino_t ino) {
    if (fd < 0) return false;
    struct stat held {}, current {};
    if (::fstat(fd, &held) || ::lstat(path.c_str(), &current)) return false;
    return S_ISREG(held.st_mode) && S_ISREG(current.st_mode) &&
           held.st_nlink == 1 && current.st_nlink == 1 &&
           held.st_uid == ::geteuid() && current.st_uid == held.st_uid &&
           !(held.st_mode & 0022) && !(current.st_mode & 0022) &&
           held.st_dev == dev && held.st_ino == ino &&
           current.st_dev == held.st_dev && current.st_ino == held.st_ino;
}

} // namespace
#endif

store::store() : data_(std::make_shared<impl>()) {}
store::~store() = default;

std::shared_ptr<store> store::try_acquire(const std::filesystem::path & parent, std::string & error) {
    error.clear();
#if defined(__linux__)
    try {
        std::error_code ec;
        const auto resolved = std::filesystem::canonical(parent, ec);
        if (ec) { error = "managed store parent unavailable"; return {}; }
        auto lease = rkva_managed_store::lease::try_acquire(resolved, error);
        if (!lease) {
            if (error.empty()) error = "managed store busy";
            return {};
        }
        auto out = std::shared_ptr<store>(new store());
        auto & d = *out->data_;
        d.parent = resolved;
        d.lease = std::move(lease);
        // Hold the concrete token while the remaining locks are acquired so a
        // concurrent legacy SQLite/object writer cannot interleave.
        auto section = d.lease->enter();
        if (!section.owns_lock()) { error = "managed store authority fenced"; return {}; }
        if (!d.lease->valid_for(resolved)) { error = "managed store authority changed"; return {}; }
        struct stat state {};
        if (::lstat(resolved.c_str(), &state) || !S_ISDIR(state.st_mode) ||
            state.st_uid != ::geteuid() || (state.st_mode & 0022)) {
            error = "store must be a private owned directory";
            return {};
        }
        if (!acquire_lock(resolved / "host-journal.lock", "host journal busy",
                          d.journal_fd, d.journal_dev, d.journal_ino, error)) return {};
        if (!acquire_lock(resolved / ".managed-orchestrator.lock", "managed orchestrator busy",
                          d.orchestrator_fd, d.orchestrator_dev, d.orchestrator_ino, error)) return {};
        if (!out->valid()) { error = "store authority changed during acquisition"; return {}; }
        return out;
    } catch (const std::exception & e) {
        error = e.what();
        return {};
    }
#else
    (void) parent;
    error = "managed store authority composition unsupported on this platform";
    return {};
#endif
}

bool store::valid() const {
#if defined(__linux__)
    if (!data_ || !data_->lease) return false;
    if (!data_->lease->valid_for(data_->parent)) return false;
    return lock_still_held(data_->journal_fd, data_->parent / "host-journal.lock",
                           data_->journal_dev, data_->journal_ino) &&
           lock_still_held(data_->orchestrator_fd, data_->parent / ".managed-orchestrator.lock",
                           data_->orchestrator_dev, data_->orchestrator_ino);
#else
    return false;
#endif
}

bool store::valid_for(const std::filesystem::path & parent) const {
#if defined(__linux__)
    if (!data_) return false;
    std::error_code ec;
    const auto resolved = std::filesystem::canonical(parent, ec);
    if (ec || resolved != data_->parent) return false;
    return valid();
#else
    (void) parent;
    return false;
#endif
}

std::unique_lock<std::recursive_mutex> store::enter() const {
    if (!data_ || !data_->lease) return {};
    auto section = data_->lease->enter();
    if (!section.owns_lock() || !valid()) return {};
    return section;
}

std::unique_lock<std::recursive_mutex> store::enter(std::string & error) const {
    error.clear();
    if (!data_ || !data_->lease) { error = "managed store authority unavailable"; return {}; }
    auto section = data_->lease->enter();
    if (!section.owns_lock()) { error = "managed store authority fenced (fork or lease lost)"; return {}; }
    if (!valid()) { error = "managed store authority changed"; return {}; }
    return section;
}

bool store::valid_for(const std::filesystem::path & parent, std::string & error) const {
    if (valid_for(parent)) return true;
    error = "managed store authority unavailable for this directory";
    return false;
}

const std::filesystem::path & store::directory() const {
    static const std::filesystem::path empty;
    return data_ ? data_->parent : empty;
}

std::shared_ptr<rkva_managed_store::lease> store::concrete() const {
    return data_ ? data_->lease : std::shared_ptr<rkva_managed_store::lease>{};
}

rkva_host_journal::journal store::journal(const std::string & owner) {
    return rkva_host_journal::journal(data_->parent.string(), owner, shared_from_this());
}

} // namespace rkva_managed_authority
