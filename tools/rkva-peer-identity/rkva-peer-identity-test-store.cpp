#include "rkva-peer-identity.h"

#include "../../ggml/src/ggml-remote-attn/rkva-managed-auth.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#ifndef _WIN32
#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace rkva_peer_identity {
#ifdef RKVA_PEER_IDENTITY_TESTING
uint64_t (*test_clock)() = nullptr;
bool (*test_step)(const char * step) = nullptr;
#endif

#ifdef RKVA_PEER_IDENTITY_TESTING
namespace {

status memory_read(bool present, bool fail, std::string & error) {
    if (fail) {
        error = "test store read failure";
        return status::unavailable;
    }
    if (!present) {
        error = "test store has no record";
        return status::unknown;
    }
    error.clear();
    return status::ok;
}

#ifndef _WIN32
namespace fs = std::filesystem;

struct fd_guard {
    int fd = -1;
    explicit fd_guard(int value = -1) : fd(value) {}
    ~fd_guard() {
        if (fd >= 0) ::close(fd);
    }
    fd_guard(const fd_guard &) = delete;
    fd_guard & operator=(const fd_guard &) = delete;
    int release() {
        const int value = fd;
        fd = -1;
        return value;
    }
};

bool absolute_path(const std::string & path) {
    return !path.empty() && path.front() == '/';
}

bool directory_without_symlink(const fs::path & path, bool & exists, std::string & error) {
    struct stat info {};
    if (::lstat(path.c_str(), &info) != 0) {
        if (errno == ENOENT) {
            exists = false;
            error.clear();
            return true;
        }
        error = "test store path is unavailable";
        return false;
    }
    exists = true;
    if (!S_ISDIR(info.st_mode) || S_ISLNK(info.st_mode)) {
        error = "test store path is not a directory";
        return false;
    }
    return true;
}

enum class namespace_state : uint8_t { missing, empty, lock_only, complete, ambiguous };

namespace_state inspect_namespace(const fs::path & root, std::string & error,
                                  bool allow_transient = false) {
    error = "test store namespace is ambiguous";
    DIR * directory = ::opendir(root.c_str());
    if (!directory) return namespace_state::missing;
    bool lock = false;
    bool record = false;
    bool pending = false;
    bool staging = false;
    bool unknown = false;
    while (const dirent * entry = ::readdir(directory)) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        if (name == lock_name) {
            lock = true;
        } else if (name == record_name) {
            record = true;
        } else if (name == pending_name) {
            pending = true;
        } else if (name == staging_name) {
            staging = true;
        } else {
            unknown = true;
        }
    }
    ::closedir(directory);
    if (unknown) return namespace_state::ambiguous;
    if ((pending || staging) && !allow_transient) return namespace_state::ambiguous;
    if (record && !lock) return namespace_state::ambiguous;
    if (record && lock) return namespace_state::complete;
    if (lock) return namespace_state::lock_only;
    return namespace_state::empty;
}

bool flush_directory(const fs::path & root, std::string & error) {
    fd_guard directory(::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (directory.fd < 0 || ::fsync(directory.fd) != 0) {
        error = "test store directory sync failed";
        return false;
    }
    return true;
}

bool step(const char * name, std::string & error) {
    if (test_step && !test_step(name)) {
        error = "test store step was refused";
        return false;
    }
    return true;
}

bool write_all(int fd, const std::string & bytes) {
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t written = ::write(fd, bytes.data() + offset, bytes.size() - offset);
        if (written <= 0) return false;
        offset += size_t(written);
    }
    return ::fsync(fd) == 0;
}

class posix_file_store final : public store {
public:
    explicit posix_file_store(std::string local_state) : local_state_(std::move(local_state)) {}

    status read(std::string & bytes, std::string & error) override {
        if (!absolute_path(local_state_)) {
            error = "test store root must be absolute";
            return status::unavailable;
        }
        const fs::path root = fs::path(local_state_) / namespace_name;
        bool exists = false;
        if (!directory_without_symlink(root, exists, error)) return status::corrupt;
        if (!exists) {
            error = "test store has no record";
            return status::unknown;
        }
        const namespace_state state = inspect_namespace(root, error);
        if (state == namespace_state::missing) {
            error = "test store has no record";
            return status::unknown;
        }
        if (state == namespace_state::ambiguous) return status::corrupt;
        if (state != namespace_state::complete) {
            error = "test store has no record";
            return status::unknown;
        }

        fd_guard lock(::open((root / lock_name).c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
        if (lock.fd < 0) {
            error = "test store lock file is unavailable";
            return status::corrupt;
        }
        if (::flock(lock.fd, LOCK_EX | LOCK_NB) != 0) {
            error = "test store is locked";
            return status::unavailable;
        }
        fd_guard file(::open((root / record_name).c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
        if (file.fd < 0) {
            error = "test store record is unavailable";
            return status::corrupt;
        }
        struct stat info {};
        if (::fstat(file.fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_nlink != 1 ||
            info.st_size <= 0 || uint64_t(info.st_size) > max_record_bytes) {
            error = "test store record is not a regular file";
            return status::corrupt;
        }
        bytes.resize(size_t(info.st_size));
        size_t offset = 0;
        while (offset < bytes.size()) {
            const ssize_t used = ::read(file.fd, bytes.data() + offset, bytes.size() - offset);
            if (used <= 0) {
                bytes.clear();
                error = "test store record read failed";
                return status::corrupt;
            }
            offset += size_t(used);
        }
        if (inspect_namespace(root, error) != namespace_state::complete) {
            bytes.clear();
            error = "test store namespace changed during read";
            return status::corrupt;
        }
        error.clear();
        return status::ok;
    }

    status write(const std::string & bytes, std::string & error) override {
        if (!absolute_path(local_state_) || bytes.empty() || bytes.size() > max_record_bytes) {
            error = "test store write argument is invalid";
            return status::invalid_argument;
        }
        bool local_exists = false;
        if (!directory_without_symlink(fs::path(local_state_), local_exists, error)) {
            return status::unavailable;
        }
        if (!local_exists) {
            error = "test store LocalState directory does not exist";
            return status::unavailable;
        }
        const fs::path root = fs::path(local_state_) / namespace_name;
        bool root_exists = false;
        if (!directory_without_symlink(root, root_exists, error)) return status::corrupt;
        if (!root_exists) {
            if (::mkdir(root.c_str(), 0700) != 0) {
                error = "test store namespace could not be created";
                return status::unavailable;
            }
        }

        fd_guard lock(::open((root / lock_name).c_str(),
                            O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0600));
        if (lock.fd < 0) {
            error = "test store lock file is unavailable";
            return status::corrupt;
        }
        if (::flock(lock.fd, LOCK_EX | LOCK_NB) != 0) {
            error = "test store is locked";
            return status::unavailable;
        }
        const namespace_state before = inspect_namespace(root, error);
        if (before == namespace_state::ambiguous || before == namespace_state::missing) {
            error = "test store namespace is ambiguous before write";
            return status::corrupt;
        }

        fd_guard pending(::open((root / pending_name).c_str(),
                                O_CREAT | O_EXCL | O_WRONLY | O_NOFOLLOW | O_CLOEXEC, 0600));
        if (pending.fd < 0 || !write_all(pending.fd, "pending\n")) {
            error = "test store pending marker could not be written";
            return status::corrupt;
        }
        if (!flush_directory(root, error) || !step("after_pending", error)) {
            if (error.empty()) error = "test store write was interrupted";
            return status::corrupt;
        }

        fd_guard staging(::open((root / staging_name).c_str(),
                                O_CREAT | O_EXCL | O_WRONLY | O_NOFOLLOW | O_CLOEXEC, 0600));
        if (staging.fd < 0 || !write_all(staging.fd, bytes)) {
            error = "test store staging file could not be written";
            return status::corrupt;
        }
        if (::fchmod(staging.fd, 0600) != 0) {
            error = "test store staging file permissions could not be set";
            return status::corrupt;
        }
        if (!step("after_staging", error)) {
            if (error.empty()) error = "test store write was interrupted";
            return status::corrupt;
        }

        if (::rename((root / staging_name).c_str(), (root / record_name).c_str()) != 0) {
            error = "test store rename failed";
            return status::corrupt;
        }
        if (!flush_directory(root, error)) return status::corrupt;
        if (!step("after_rename", error)) {
            if (error.empty()) error = "test store write was interrupted";
            return status::corrupt;
        }
        if (::unlink((root / pending_name).c_str()) != 0) {
            error = "test store pending marker could not be removed";
            return status::corrupt;
        }
        if (!flush_directory(root, error)) return status::corrupt;
        if (!step("after_pending_unlink", error)) {
            if (error.empty()) error = "test store write was interrupted";
            return status::corrupt;
        }
        if (inspect_namespace(root, error) != namespace_state::complete) {
            error = "test store namespace is incomplete after write";
            return status::corrupt;
        }
        error.clear();
        return status::ok;
    }

    status erase(std::string & error) override {
        if (!absolute_path(local_state_)) {
            error = "test store root must be absolute";
            return status::unavailable;
        }
        const fs::path root = fs::path(local_state_) / namespace_name;
        bool exists = false;
        if (!directory_without_symlink(root, exists, error)) return status::corrupt;
        if (!exists) {
            error.clear();
            return status::ok;
        }
        const namespace_state state = inspect_namespace(root, error, true);
        if (state == namespace_state::ambiguous) return status::corrupt;
        fd_guard lock(::open((root / lock_name).c_str(),
                            O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0600));
        if (lock.fd < 0 || ::flock(lock.fd, LOCK_EX | LOCK_NB) != 0) {
            error = "test store is locked";
            return status::unavailable;
        }
        for (const char * name : {record_name, pending_name, staging_name}) {
            if (::unlink((root / name).c_str()) != 0 && errno != ENOENT) {
                error = "test store entry could not be removed";
                return status::corrupt;
            }
        }
        if (!flush_directory(root, error)) return status::corrupt;
        const namespace_state after = inspect_namespace(root, error);
        if (after != namespace_state::empty && after != namespace_state::lock_only) {
            error = "test store namespace is ambiguous after wipe";
            return status::corrupt;
        }
        error.clear();
        return status::ok;
    }

    bool new_key(key & out, std::string & error) override {
        for (size_t i = 0; i < out.size(); ++i) out[i] = uint8_t(0xa5 ^ i);
        error.clear();
        return true;
    }

    bool seal_key(const key & value, std::string & sealed, std::string & error) override {
        sealed = to_hex(value.data(), value.size());
        error.clear();
        return true;
    }

    bool unseal_key(const std::string & sealed, key & out, std::string & error) override {
        if (!from_hex(sealed, out.data(), out.size())) {
            error = "test store protected key is invalid";
            return false;
        }
        error.clear();
        return true;
    }

private:
    std::string local_state_;
};
#endif  // !_WIN32
}  // namespace

status test_memory_store::read(std::string & out, std::string & error) {
    const status result = memory_read(present, fail_read, error);
    if (result == status::ok) out = bytes;
    return result;
}

status test_memory_store::write(const std::string & value, std::string & error) {
    if (fail_write) {
        error = "test store write failure";
        return status::unavailable;
    }
    bytes = value;
    present = true;
    error.clear();
    return status::ok;
}

status test_memory_store::erase(std::string & error) {
    bytes.clear();
    present = false;
    error.clear();
    return status::ok;
}

bool test_memory_store::new_key(key & out, std::string & error) {
    for (size_t i = 0; i < out.size(); ++i) out[i] = uint8_t(0xa5 ^ i);
    stored_key = out;
    has_key = true;
    error.clear();
    return true;
}

bool test_memory_store::seal_key(const key & value, std::string & sealed,
                                 std::string & error) {
    if (has_key && !std::equal(value.begin(), value.end(), stored_key.begin())) {
        error = "test store key does not match the stored key";
        return false;
    }
    stored_key = value;
    has_key = true;
    sealed = to_hex(value.data(), value.size());
    error.clear();
    return true;
}

bool test_memory_store::unseal_key(const std::string & sealed, key & out,
                                   std::string & error) {
    if (!from_hex(sealed, out.data(), out.size())) {
        error = "test store protected key is invalid";
        return false;
    }
    error.clear();
    return true;
}

#ifndef _WIN32
std::unique_ptr<store> open_test_directory(const std::string & local_state,
                                           std::string & error) {
    if (!absolute_path(local_state)) {
        error = "test store root must be absolute";
        return nullptr;
    }
    error.clear();
    return std::unique_ptr<store>(new posix_file_store(local_state));
}
#endif
#endif  // RKVA_PEER_IDENTITY_TESTING

}  // namespace rkva_peer_identity
