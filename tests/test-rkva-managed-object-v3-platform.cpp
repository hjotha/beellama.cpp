// Platform backend contract tests for the v3 durable-filesystem skeleton.
//
// The Windows/UWP class in rkva-managed-object-v3-windows.{h,cpp} is compiled
// with RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST. On Linux its Win32 body is
// not compiled and every method must fail closed; this test asserts that and
// statically checks that the guarded source still contains the required
// FlushFileBuffers/no-replace/parent-flush sequence.
//
// To keep the *semantic* contract executable on this host, the same test body
// runs against a POSIX reference backend with the same conservative policy:
//
//   * file fsync and directory fsync report success only when the syscall does;
//   * an existing destination is never replaced and the source is preserved;
//   * a successful rename + parent fsync is still `uncertain`, because a host
//     CPU test cannot qualify directory power-loss durability;
//   * qualified() is false.
//
// This is parity of the contract, not a Windows execution. The Windows body is
// compile/static evidence only and remains unqualified for power loss.

#include "../tools/xbox/uwp/rkva/rkva-managed-object-v3-windows.h"

#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
namespace authority = rkva_managed_store;
namespace v3 = rkva_managed_object_v3;

namespace {

std::string read_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

void write_file(const fs::path & path, const std::string & bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), std::streamsize(bytes.size()));
    assert(bool(output));
}

#ifndef _WIN32

bool destination_exists(const fs::path & path) {
    std::error_code ec;
    const auto status = fs::symlink_status(path, ec);
    if (ec && ec != std::errc::no_such_file_or_directory) return true;
    return status.type() != fs::file_type::not_found;
}

int rename_noreplace(const fs::path & source, const fs::path & destination) {
#if defined(SYS_renameat2)
    if (::syscall(SYS_renameat2, AT_FDCWD, source.c_str(), AT_FDCWD,
                  destination.c_str(), 1u /* RENAME_NOREPLACE */) == 0)
        return 0;
    if (errno != ENOSYS && errno != EINVAL) return -1;
#endif
    if (destination_exists(destination)) {
        errno = EEXIST;
        return -1;
    }
    return ::rename(source.c_str(), destination.c_str());
}

// POSIX reference implementation of the same skeleton contract. Test-only; it
// is not a production backend and cannot qualify power loss.
class posix_filesystem final : public v3::durable_filesystem {
public:
    posix_filesystem(std::shared_ptr<authority::lease> store, fs::path root)
        : durable_filesystem(store, std::move(root)), store_(std::move(store)) {}

    bool acquire(const fs::path & root, std::string & error) {
        if (!root_.empty()) {
            error = "backend already acquired";
            return false;
        }
        if (!store_ || !store_->valid_for(root.parent_path())) {
            error = "store authority unavailable";
            return false;
        }
        const auto normalized = root.lexically_normal();
        if (normalized != this->root()) {
            error = "filesystem root mismatch";
            return false;
        }
        if (!normalized.is_absolute() || normalized.filename() != "rkva-snapshots-v3") {
            error = "absolute v3 root required";
            return false;
        }
        std::error_code ec;
        const auto status = fs::symlink_status(normalized, ec);
        if (ec || !fs::is_directory(status) || fs::is_symlink(status)) {
            error = "invalid v3 root";
            return false;
        }
        for (const auto & part : normalized)
            if (part == "rkva-snapshots-v1" || part == "rkva-snapshots-v2") {
                error = "legacy namespace forbidden";
                return false;
            }
        root_ = normalized;
        error.clear();
        return true;
    }

    bool qualified() const override { return false; }

    bool sync_file(const fs::path & path, std::string & error) override {
        if (!lock_held()) { error = "lock unavailable"; return false; }
        const auto target = path.lexically_normal();
        if (target.parent_path().parent_path() != root_ ||
            target.parent_path().filename().string().rfind("tmp-", 0) != 0) {
            error = "staged file path invalid";
            return false;
        }
        const auto name = target.filename().string();
        const bool chunk = name.size() > 5 && name.rfind("n-", 0) == 0 &&
                           name.compare(name.size() - 3, 3, ".kv") == 0;
        if (name != "manifest-v2.bin" && !chunk) {
            error = "staged filename invalid";
            return false;
        }
        const int fd = ::open(target.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
        if (fd < 0) { error = "open failed"; return false; }
        struct stat st {};
        const bool regular = ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_nlink == 1;
        const bool synced = regular && ::fsync(fd) == 0;
        ::close(fd);
        if (!regular) { error = "not a regular single-link file"; return false; }
        if (!synced) { error = "file sync failed"; return false; }
        error.clear();
        return true;
    }

    bool sync_directory(const fs::path & path, std::string & error) override {
        if (!lock_held()) { error = "lock unavailable"; return false; }
        const auto target = path.lexically_normal();
        if (target != root_ &&
            (target.parent_path() != root_ ||
             target.filename().string().rfind("tmp-", 0) != 0)) {
            error = "staged directory path invalid";
            return false;
        }
        const int fd = ::open(target.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (fd < 0) { error = "open failed"; return false; }
        struct stat st {};
        const bool dir = ::fstat(fd, &st) == 0 && S_ISDIR(st.st_mode);
        const bool synced = dir && ::fsync(fd) == 0;
        ::close(fd);
        if (!dir) { error = "not a directory"; return false; }
        if (!synced) { error = "directory sync failed"; return false; }
        error.clear();
        return true;
    }

    v3::move_result rename_noreplace_and_sync(const fs::path & from,
                                              const fs::path & to,
                                              std::string & error) override {
        if (!lock_held()) { error = "lock unavailable"; return v3::move_result::no_change; }
        const auto source = from.lexically_normal();
        const auto destination = to.lexically_normal();
        if (source.parent_path() != root_ || destination.parent_path() != root_) {
            error = "rename paths invalid";
            return v3::move_result::no_change;
        }
        if (destination_exists(destination)) {
            error = "destination already exists";
            return v3::move_result::no_change;
        }
        if (rename_noreplace(source, destination) != 0) {
            const int saved = errno;
            error = std::string("rename failed: ") + std::strerror(saved);
            return saved == EEXIST ? v3::move_result::no_change
                                   : v3::move_result::uncertain;
        }
        if (!sync_directory(root_, error)) {
            error = "rename applied but parent sync failed: " + error;
            return v3::move_result::uncertain;
        }
        // A successful file/dir fsync is not a power-loss qualification.
        error = "rename/flush observed, but directory power-loss durability is unqualified";
        return v3::move_result::uncertain;
    }

private:
    bool lock_held() const {
        return store_ && !root_.empty() && store_->valid_for(root_.parent_path());
    }
    std::shared_ptr<authority::lease> store_;
    fs::path root_;
};

#endif // !_WIN32

// Runs the same contract against a backend. Kept as a template so the Windows
// class can reuse it when built and run on a Windows host.
template <class Backend>
void exercise_contract(Backend & backend, const fs::path & root) {
    std::string already;
    assert(!backend.acquire(root, already)); // already acquired
    assert(!backend.qualified());

    const auto stage = root / (std::string("tmp-") + std::string(64, 'b'));
    assert(fs::create_directory(stage));
    write_file(stage / "n-0.kv", "stage-data");
    std::string error;
    assert(backend.sync_file(stage / "n-0.kv", error));
    assert(backend.sync_directory(stage, error));
    assert(backend.sync_directory(root, error));

    const auto target = root / std::string(64, 'b');
    assert(fs::create_directory(target));
    write_file(target / "sentinel", "keep");
    assert(backend.rename_noreplace_and_sync(stage, target, error) ==
           v3::move_result::no_change);
    assert(fs::is_directory(stage));
    assert(read_file(target / "sentinel") == "keep");

    assert(fs::remove(target / "sentinel"));
    assert(fs::remove(target));
    assert(backend.rename_noreplace_and_sync(stage, target, error) ==
           v3::move_result::uncertain);
    assert(!fs::exists(stage));
    assert(read_file(target / "n-0.kv") == "stage-data");

#ifndef _WIN32
    // A symlink is never followed, and a path outside the exact stage layout is
    // refused.
    const auto link = root / "not-a-stage-link";
    assert(::symlink(target.c_str(), link.c_str()) == 0);
    assert(!backend.sync_file(link / "n-0.kv", error));
    assert(!backend.sync_directory(link, error));
    fs::remove(link);
#endif
}

void static_windows_skeleton_checks() {
#ifdef RKVA_PLATFORM_SOURCE_DIR
    const fs::path source = fs::path(RKVA_PLATFORM_SOURCE_DIR) /
                            "tools/xbox/uwp/rkva/rkva-managed-object-v3-windows.cpp";
    std::ifstream input(source, std::ios::binary);
    assert(bool(input));
    const std::string text((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    assert(text.find("FlushFileBuffers") != std::string::npos);
    assert(text.find("MoveFileExW") != std::string::npos);
    assert(text.find("FILE_FLAG_OPEN_REPARSE_POINT") != std::string::npos);
    assert(text.find("MOVEFILE_REPLACE_EXISTING") == std::string::npos);
    assert(text.find("MOVEFILE_COPY_ALLOWED") == std::string::npos);
    assert(text.find("RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST") != std::string::npos);
    assert(text.find("return false;") != std::string::npos); // qualified()
    assert(text.find("directory power-loss durability is unqualified") !=
           std::string::npos);
#endif
}

struct fixture {
    fs::path parent;
    fs::path root;
    std::shared_ptr<authority::lease> token;

    fixture() {
        const auto stamp = std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count();
        parent = fs::temp_directory_path() /
                 ("rkva-v3-platform-" + std::to_string(stamp));
        assert(fs::create_directory(parent));
        root = parent / "rkva-snapshots-v3";
        assert(fs::create_directory(root));
        std::string error;
        token = authority::lease::try_acquire(parent, error);
        assert(token);
    }
    ~fixture() {
        token.reset();
        std::error_code ec;
        fs::remove_all(parent, ec);
    }
};

} // namespace

int main() {
    static_windows_skeleton_checks();
    fixture f;

    v3::windows_filesystem windows(f.token, f.root);
    assert(!windows.qualified());
    std::string error;
    // Linux (and any non-experimental build) must fail closed without creating
    // a lock, a stage or touching the root.
    assert(!windows.acquire(f.root, error));
    assert(!windows.sync_file(f.root / "x", error));
    assert(!windows.sync_directory(f.root, error));
    assert(windows.rename_noreplace_and_sync(f.root / "a", f.root / "b", error) ==
           v3::move_result::no_change);
    assert(fs::is_empty(f.root));

#ifdef _WIN32
    v3::windows_filesystem acquired(f.token, f.root);
    assert(acquired.acquire(f.root, error));
    exercise_contract(acquired, f.root);
    std::cout << "Windows backend contract: PASS\n";
#else
    posix_filesystem posix(f.token, f.root);
    assert(posix.acquire(f.root, error));
    exercise_contract(posix, f.root);
    std::cout << "POSIX backend contract parity: PASS (Windows skeleton not executed)\n";
#endif

    return 0;
}
