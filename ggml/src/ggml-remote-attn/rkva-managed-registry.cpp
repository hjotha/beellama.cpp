#include "rkva-managed-registry.h"
#include "rkva-snapshot-digest.h"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <limits>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace rkva_managed_registry {
#ifdef RKVA_MANAGED_REGISTRY_TESTING
void (*test_persist_hook)(const char *) = nullptr;
#endif
namespace {
constexpr size_t max_file = 16 * 1024 * 1024;
constexpr size_t max_records = 50000;
bool hex64(const std::string & s) {
    return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
bool number(std::string_view s, uint64_t & n) {
    if (s.empty() || (s.size() > 1 && s[0] == '0')) return false;
    auto result = std::from_chars(s.data(), s.data() + s.size(), n);
    return result.ec == std::errc{} && result.ptr == s.data() + s.size();
}
char phase_char(phase p) {
    switch (p) { case phase::pending: return 'P'; case phase::committed: return 'C';
        case phase::aborted: return 'A'; case phase::released: return 'R'; }
    return '?';
}
std::string encode(const std::string & owner, const snapshot & s) {
    std::string body = "RKVA-MANAGED-1\n" + owner + "\n" + std::to_string(s.version) + "\n";
    for (const auto & t : s.transactions)
        body += t.id + " " + t.reference + " " + t.key + " " + phase_char(t.state) + " " +
                std::to_string(t.committed_version) + "\n";
    const auto hash = rkva_snapshot::digest(body.data(), body.size());
    return body + rkva_snapshot::hex(hash.data()) + "\n";
}
bool decode(const std::string & bytes, const std::string & owner, snapshot & s, std::string & err) {
    auto bad = [&]() { err = "registry corrupt, incompatible, or owner mismatch"; return false; };
    if (bytes.size() < 65 || bytes.size() > max_file || bytes.back() != '\n') return bad();
    const size_t end = bytes.size() - 65;
    if (end == 0 || bytes[end - 1] != '\n') return bad();
    const auto hash = rkva_snapshot::digest(bytes.data(), end);
    if (bytes.compare(end, 64, rkva_snapshot::hex(hash.data())) != 0) return bad();
    std::istringstream input(bytes.substr(0, end));
    std::string line;
    if (!std::getline(input, line) || line != "RKVA-MANAGED-1" ||
        !std::getline(input, line) || line != owner || !std::getline(input, line) ||
        !number(line, s.version)) return bad();
    s.transactions.clear();
    std::set<std::string> txs, refs;
    while (std::getline(input, line)) {
        if (line.size() < 198 || line[64] != ' ' || line[129] != ' ' || line[194] != ' ' || line[196] != ' ') return bad();
        transaction t;
        t.id = line.substr(0,64); t.reference = line.substr(65,64); t.key = line.substr(130,64);
        if (!hex64(t.id) || !hex64(t.reference) || !hex64(t.key) ||
            !number(std::string_view(line).substr(197), t.committed_version) ||
            !txs.insert(t.id).second || !refs.insert(t.reference).second) return bad();
        switch (line[195]) { case 'P': t.state = phase::pending; break; case 'C': t.state = phase::committed; break;
            case 'A': t.state = phase::aborted; break; case 'R': t.state = phase::released; break; default: return bad(); }
        if (((t.state == phase::committed || t.state == phase::released) &&
             (t.committed_version == 0 || t.committed_version > s.version)) ||
            ((t.state == phase::pending || t.state == phase::aborted) && t.committed_version != 0) ||
            s.transactions.size() >= max_records) return bad();
        s.transactions.push_back(std::move(t));
    }
    return true;
}
#if defined(__unix__) || defined(__APPLE__)
struct fd_guard { int fd = -1; ~fd_guard() { if (fd >= 0) ::close(fd); } };
bool sync_fd(int fd) { int r; do { r = ::fsync(fd); } while (r < 0 && errno == EINTR); return r == 0; }
class locked_store {
    fd_guard dir, lock;
public:
    bool open(const std::string & path, const std::string & owner, std::string & error) {
        if (!hex64(owner)) { error = "invalid explicit single owner"; return false; }
        std::error_code path_error;
        const auto resolved = std::filesystem::canonical(path, path_error);
        if (path_error) { error = "cannot resolve managed directory"; return false; }
        for (const auto & part : resolved) {
            if (part == "rkva-snapshots-v1" || part == "rkva-snapshots-v2") {
                error = "legacy snapshot namespaces are forbidden"; return false;
            }
        }
        dir.fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (dir.fd < 0) { error = "cannot open managed directory"; return false; }
        lock.fd = ::openat(dir.fd, "registry.lock", O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
        struct stat st{};
        if (lock.fd < 0 || ::fstat(lock.fd, &st) || !S_ISREG(st.st_mode) || st.st_nlink != 1) {
            error = "invalid registry lock"; return false;
        }
        int result; do { result = ::flock(lock.fd, LOCK_EX); } while (result < 0 && errno == EINTR);
        if (result) { error = "cannot lock registry"; return false; }
        // A prior rename may have succeeded while its directory sync failed.
        // Re-sync before allowing an idempotent retry to report success.
        if (!sync_fd(dir.fd)) { error = "registry directory sync failed"; return false; }
        return true;
    }
    bool read(const std::string & owner, snapshot & s, std::string & error) {
        fd_guard file; file.fd = ::openat(dir.fd, "registry", O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
        struct stat st{};
        if (file.fd < 0 || ::fstat(file.fd, &st) || !S_ISREG(st.st_mode) || st.st_nlink != 1 ||
            st.st_size < 0 || uint64_t(st.st_size) > max_file) { error = "missing or invalid registry"; return false; }
        std::string bytes(size_t(st.st_size), '\0'); size_t used = 0;
        while (used < bytes.size()) {
            ssize_t n = ::read(file.fd, &bytes[used], bytes.size() - used);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { error = "registry read failed"; return false; }
            used += size_t(n);
        }
        return decode(bytes, owner, s, error);
    }
    bool write(const std::string & owner, const snapshot & s, bool create, std::string & error) {
        const auto bytes = encode(owner, s);
        if (bytes.size() > max_file) { error = "registry capacity exceeded"; return false; }
        if (create) {
            struct stat st{};
            if (::fstatat(dir.fd, "registry", &st, AT_SYMLINK_NOFOLLOW) == 0 || errno != ENOENT) {
                error = "registry already exists or cannot be inspected"; return false;
            }
        }
        fd_guard file; std::string name;
        for (unsigned i = 0; i < 1024; ++i) {
            name = "registry.tmp." + std::to_string(::getpid()) + "." + std::to_string(i);
            file.fd = ::openat(dir.fd, name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
            if (file.fd >= 0 || errno != EEXIST) break;
        }
        if (file.fd < 0) { error = "cannot create private registry staging file"; return false; }
        size_t used = 0;
        while (used < bytes.size()) {
            ssize_t n = ::write(file.fd, bytes.data() + used, bytes.size() - used);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { error = "registry write failed"; ::unlinkat(dir.fd, name.c_str(), 0); return false; }
            used += size_t(n);
        }
        if (!sync_fd(file.fd)) { error = "registry file sync failed"; ::unlinkat(dir.fd, name.c_str(), 0); return false; }
#ifdef RKVA_MANAGED_REGISTRY_TESTING
        if (test_persist_hook) test_persist_hook("before_rename");
#endif
        if (::renameat(dir.fd, name.c_str(), dir.fd, "registry")) {
            error = "registry rename failed"; ::unlinkat(dir.fd, name.c_str(), 0); return false;
        }
#ifdef RKVA_MANAGED_REGISTRY_TESTING
        if (test_persist_hook) test_persist_hook("after_rename");
#endif
        if (!sync_fd(dir.fd)) { error = "registry directory sync failed; outcome uncertain, retry same request"; return false; }
        return true;
    }
 };
#elif defined(_WIN32) && defined(RKVA_MANAGED_REGISTRY_WINDOWS_EXPERIMENTAL)
struct handle_guard {
    HANDLE handle = INVALID_HANDLE_VALUE;
    ~handle_guard() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
    void close() { if (handle != INVALID_HANDLE_VALUE) { CloseHandle(handle); handle = INVALID_HANDLE_VALUE; } }
};
bool regular_handle(HANDLE h) {
    BY_HANDLE_FILE_INFORMATION info{};
    return GetFileType(h) == FILE_TYPE_DISK && GetFileInformationByHandle(h, &info) &&
        !(info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) &&
        info.nNumberOfLinks == 1;
}
class locked_store {
    handle_guard dir, lock;
    OVERLAPPED lock_offset{};
    std::filesystem::path root;
    bool locked = false;
public:
    ~locked_store() { if (locked) UnlockFileEx(lock.handle, 0, 1, 0, &lock_offset); }
    bool open(const std::string & path, const std::string & owner, std::string & error) {
        if (!hex64(owner)) { error = "invalid explicit single owner"; return false; }
        std::error_code ec;
        const auto resolved = std::filesystem::canonical(std::filesystem::u8path(path), ec);
        if (ec) { error = "cannot resolve managed directory"; return false; }
        for (const auto & part : resolved) {
            if (_wcsicmp(part.c_str(), L"rkva-snapshots-v1") == 0 ||
                _wcsicmp(part.c_str(), L"rkva-snapshots-v2") == 0) {
                error = "legacy snapshot namespaces are forbidden"; return false;
            }
        }
        root = std::filesystem::u8path(path);
        dir.handle = CreateFileW(root.c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        BY_HANDLE_FILE_INFORMATION info{};
        if (dir.handle == INVALID_HANDLE_VALUE || !GetFileInformationByHandle(dir.handle, &info) ||
            !(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ||
            (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
            error = "cannot open trusted managed directory without reparse point"; return false;
        }
        const auto lock_path = root / L"registry.lock";
        lock.handle = CreateFileW(lock_path.c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        if (lock.handle == INVALID_HANDLE_VALUE || !regular_handle(lock.handle)) {
            error = "invalid registry lock"; return false;
        }
        if (!LockFileEx(lock.handle, LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, &lock_offset)) {
            error = "cannot lock registry"; return false;
        }
        locked = true;
        // Required before even an idempotent success after an uncertain rename.
        // This directory flush is supported by the tested desktop system; its
        // durability and UWP availability require separate platform validation.
        if (!FlushFileBuffers(dir.handle)) { error = "registry directory flush unavailable"; return false; }
        return true;
    }
    bool read(const std::string & owner, snapshot & s, std::string & error) {
        handle_guard file;
        file.handle = CreateFileW((root / L"registry").c_str(), GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        LARGE_INTEGER size{};
        if (file.handle == INVALID_HANDLE_VALUE || !regular_handle(file.handle) ||
            !GetFileSizeEx(file.handle, &size) || size.QuadPart < 0 || uint64_t(size.QuadPart) > max_file) {
            error = "missing or invalid registry"; return false;
        }
        std::string bytes(size_t(size.QuadPart), '\0'); size_t used = 0;
        while (used < bytes.size()) {
            DWORD count = 0;
            if (!ReadFile(file.handle, &bytes[used], DWORD(bytes.size() - used), &count, nullptr) || count == 0) {
                error = "registry read failed"; return false;
            }
            used += count;
        }
        return decode(bytes, owner, s, error);
    }
    bool write(const std::string & owner, const snapshot & s, bool create, std::string & error) {
        const auto bytes = encode(owner, s);
        if (bytes.size() > max_file) { error = "registry capacity exceeded"; return false; }
        const auto destination = root / L"registry";
        handle_guard existing;
        existing.handle = CreateFileW(destination.c_str(), GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        if (create) {
            if (existing.handle != INVALID_HANDLE_VALUE || GetLastError() != ERROR_FILE_NOT_FOUND) {
                error = "registry already exists or cannot be inspected"; return false;
            }
        } else if (existing.handle == INVALID_HANDLE_VALUE || !regular_handle(existing.handle)) {
            error = "missing or invalid registry"; return false;
        }
        existing.close();
        handle_guard file; std::filesystem::path temporary;
        for (unsigned i = 0; i < 1024; ++i) {
            temporary = root / (L"registry.tmp." + std::to_wstring(GetCurrentProcessId()) + L"." + std::to_wstring(i));
            file.handle = CreateFileW(temporary.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_WRITE_THROUGH, nullptr);
            if (file.handle != INVALID_HANDLE_VALUE || GetLastError() != ERROR_FILE_EXISTS) break;
        }
        if (file.handle == INVALID_HANDLE_VALUE || !regular_handle(file.handle)) {
            error = "cannot create private registry staging file"; return false;
        }
        size_t used = 0;
        while (used < bytes.size()) {
            DWORD count = 0;
            if (!WriteFile(file.handle, bytes.data() + used, DWORD(bytes.size() - used), &count, nullptr) || count == 0) {
                error = "registry write failed"; file.close(); DeleteFileW(temporary.c_str()); return false;
            }
            used += count;
        }
        if (!FlushFileBuffers(file.handle)) {
            error = "registry file flush failed"; file.close(); DeleteFileW(temporary.c_str()); return false;
        }
        file.close();
#ifdef RKVA_MANAGED_REGISTRY_TESTING
        if (test_persist_hook) test_persist_hook("before_rename");
#endif
        // REPLACEFILE_WRITE_THROUGH is unsupported. The replacement file and
        // directory are flushed explicitly; no power-loss guarantee is assumed.
        const bool ok = create ? MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_WRITE_THROUGH) != 0
                               : ReplaceFileW(destination.c_str(), temporary.c_str(), nullptr, 0, nullptr, nullptr) != 0;
        if (!ok) { error = "registry replacement failed; outcome uncertain, inspect before retry"; return false; }
#ifdef RKVA_MANAGED_REGISTRY_TESTING
        if (test_persist_hook) test_persist_hook("after_rename");
#endif
        if (!FlushFileBuffers(dir.handle)) { error = "registry directory flush failed; outcome uncertain, retry same request"; return false; }
        return true;
    }
};
#endif
}
registry::registry(std::string dir, std::string owner) : dir_(std::move(dir)), owner_(std::move(owner)) {}
#if defined(__unix__) || defined(__APPLE__) || (defined(_WIN32) && defined(RKVA_MANAGED_REGISTRY_WINDOWS_EXPERIMENTAL))
bool registry::initialize(std::string & error) {
    error.clear(); locked_store store; snapshot s;
    return store.open(dir_, owner_, error) && store.write(owner_, s, true, error);
}
bool registry::inspect(snapshot & out, std::string & error) const {
    error.clear(); out = {}; locked_store store; snapshot s;
    if (!store.open(dir_, owner_, error) || !store.read(owner_, s, error)) return false;
    out = std::move(s); return true;
}
bool registry::prepare(const std::string & tx, const std::string & ref, const std::string & key, std::string & error) {
    error.clear();
    if (!hex64(tx) || !hex64(ref) || !hex64(key)) { error = "invalid identity"; return false; }
    locked_store store; snapshot s;
    if (!store.open(dir_, owner_, error) || !store.read(owner_, s, error)) return false;
    for (const auto & t : s.transactions) {
        if (t.id == tx) {
            if (t.reference == ref && t.key == key && (t.state == phase::pending || t.state == phase::committed)) return true;
            error = "transaction conflict or already terminal"; return false;
        }
        if (t.reference == ref) { error = "reference already used"; return false; }
    }
    if (s.transactions.size() >= max_records || s.version == UINT64_MAX) { error = "registry capacity exceeded"; return false; }
    s.transactions.push_back({tx, ref, key, phase::pending, 0}); ++s.version;
    return store.write(owner_, s, false, error);
}
bool registry::commit(const std::string & tx, uint64_t & version, std::string & error) {
    error.clear(); version = 0; locked_store store; snapshot s;
    if (!hex64(tx)) { error = "invalid transaction"; return false; }
    if (!store.open(dir_, owner_, error) || !store.read(owner_, s, error)) return false;
    for (auto & t : s.transactions) if (t.id == tx) {
        if (t.state == phase::committed) { version = t.committed_version; return true; }
        if (t.state != phase::pending || s.version == UINT64_MAX) { error = "transaction cannot commit"; return false; }
        t.state = phase::committed; t.committed_version = ++s.version;
        if (!store.write(owner_, s, false, error)) return false;
        version = t.committed_version; return true;
    }
    error = "unknown transaction"; return false;
}
bool registry::abort(const std::string & tx, std::string & error) {
    error.clear(); locked_store store; snapshot s;
    if (!hex64(tx)) { error = "invalid transaction"; return false; }
    if (!store.open(dir_, owner_, error) || !store.read(owner_, s, error)) return false;
    for (auto & t : s.transactions) if (t.id == tx) {
        if (t.state == phase::aborted) return true;
        if (t.state != phase::pending || s.version == UINT64_MAX) { error = "transaction cannot abort"; return false; }
        t.state = phase::aborted; ++s.version; return store.write(owner_, s, false, error);
    }
    error = "unknown transaction"; return false;
}
bool registry::release(const std::string & ref, uint64_t version, std::string & error) {
    error.clear(); locked_store store; snapshot s;
    if (!hex64(ref) || !version) { error = "invalid reference or version"; return false; }
    if (!store.open(dir_, owner_, error) || !store.read(owner_, s, error)) return false;
    for (auto & t : s.transactions) if (t.reference == ref) {
        if (t.committed_version != version) { error = "stale reference version"; return false; }
        if (t.state == phase::released) return true;
        if (t.state != phase::committed || s.version == UINT64_MAX) { error = "reference cannot be released"; return false; }
        t.state = phase::released; ++s.version; return store.write(owner_, s, false, error);
    }
    error = "unknown reference"; return false;
}
#else
bool registry::initialize(std::string & e) { e = "persistent registry backend unavailable on this platform"; return false; }
bool registry::inspect(snapshot & out, std::string & e) const { out = {}; e = "persistent registry backend unavailable on this platform"; return false; }
bool registry::prepare(const std::string &, const std::string &, const std::string &, std::string & e) { e = "persistent registry backend unavailable on this platform"; return false; }
bool registry::commit(const std::string &, uint64_t & version, std::string & e) { version = 0; e = "persistent registry backend unavailable on this platform"; return false; }
bool registry::abort(const std::string &, std::string & e) { e = "persistent registry backend unavailable on this platform"; return false; }
bool registry::release(const std::string &, uint64_t, std::string & e) { e = "persistent registry backend unavailable on this platform"; return false; }
#endif
bool registry::candidates(std::vector<candidate> & out, std::string & error) const {
    out.clear(); snapshot s; if (!inspect(s, error)) return false;
    std::set<std::string> known, pinned;
    for (const auto & t : s.transactions) {
        known.insert(t.key);
        if (t.state == phase::pending || t.state == phase::committed) pinned.insert(t.key);
    }
    for (const auto & key : known) if (!pinned.count(key)) out.push_back({key, s.version});
    return true;
}
}
