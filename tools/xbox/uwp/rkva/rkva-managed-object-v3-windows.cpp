#include "rkva-managed-object-v3-windows.h"

#include <algorithm>
#include <cwctype>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif
#include <windows.h>
#endif

namespace rkva_managed_object_v3 {

#ifdef RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST
void (*windows_test_hook)(const char * phase) = nullptr;
#endif

#ifdef _WIN32
namespace {

struct handle {
    HANDLE h = INVALID_HANDLE_VALUE;
    ~handle() {
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
    }
    void reset(HANDLE value) {
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
        h = value;
    }
    handle() = default;
    handle(const handle &) = delete;
    handle & operator=(const handle &) = delete;
};

std::string error_code(const char * prefix, DWORD error) {
    return std::string(prefix) + " (Win32 " + std::to_string(error) + ")";
}

HANDLE open_path(const fs::path & path, DWORD access, DWORD share, DWORD disposition,
                 DWORD flags) {
    CREATEFILE2_EXTENDED_PARAMETERS params{};
    params.dwSize = sizeof(params);
    params.dwFileAttributes = FILE_ATTRIBUTE_NORMAL;
    params.dwFileFlags = flags | FILE_FLAG_OPEN_REPARSE_POINT;
    return CreateFile2(path.wstring().c_str(), access, share, disposition, &params);
}

bool regular(HANDLE h) {
    BY_HANDLE_FILE_INFORMATION info{};
    return GetFileType(h) == FILE_TYPE_DISK && GetFileInformationByHandle(h, &info) &&
           !(info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) &&
           info.nNumberOfLinks == 1;
}

bool directory(HANDLE h) {
    BY_HANDLE_FILE_INFORMATION info{};
    return GetFileType(h) == FILE_TYPE_DISK && GetFileInformationByHandle(h, &info) &&
           (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
           !(info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT);
}

bool hex_name(const std::wstring & value) {
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](wchar_t c) {
               return (c >= L'0' && c <= L'9') || (c >= L'a' && c <= L'f');
           });
}

bool stage_name(const std::wstring & value) {
    return value.size() == 68 && value.compare(0, 4, L"tmp-") == 0 &&
           hex_name(value.substr(4));
}

bool chunk_name(const std::wstring & value) {
    return value.size() > 5 && value.compare(0, 2, L"n-") == 0 &&
           value.compare(value.size() - 3, 3, L".kv") == 0 &&
           std::all_of(value.begin() + 2, value.end() - 3, [](wchar_t c) {
               return c >= L'0' && c <= L'9';
           });
}

void hook(const char * phase) {
#ifdef RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST
    if (windows_test_hook) windows_test_hook(phase);
#else
    (void) phase;
#endif
}

} // namespace
#endif

struct windows_filesystem::state {
    std::shared_ptr<rkva_managed_store::lease> store;
    fs::path root;
#ifdef _WIN32
    handle root_handle;
#endif
};

windows_filesystem::windows_filesystem(std::shared_ptr<rkva_managed_store::lease> store,
                                       fs::path root)
    : durable_filesystem(store, std::move(root)), state_(std::make_unique<state>()) {
    state_->store = std::move(store);
}

windows_filesystem::~windows_filesystem() = default;

bool windows_filesystem::lock_held() const {
#if defined(_WIN32) && defined(RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST)
    return state_ && state_->store && !state_->root.empty() &&
           state_->store->valid_for(state_->root.parent_path());
#else
    return false;
#endif
}

bool windows_filesystem::qualified() const {
    // Even the experimental Win32 body cannot claim power-loss qualification.
    return false;
}

bool windows_filesystem::acquire(const fs::path & root, std::string & error) {
#if !defined(_WIN32) || !defined(RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST)
    (void) root;
    error = "Windows durable backend requires _WIN32 and "
            "RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST";
    return false;
#else
    if (!state_ || !state_->store || !state_->root.empty()) {
        error = "shared store lease and an unused backend are required";
        return false;
    }
    try {
        auto guard = state_->store->enter();
        const auto normalized = root.lexically_normal();
        if (normalized != this->root()) {
            error = "filesystem root mismatch";
            return false;
        }
        if (!normalized.is_absolute() || normalized.filename() != L"rkva-snapshots-v3") {
            error = "absolute v3 root required";
            return false;
        }
        if (!guard.owns_lock() ||
            !state_->store->valid_for(normalized.parent_path())) {
            error = "store authority unavailable or root mismatch";
            return false;
        }
        for (const auto & part : normalized) {
            std::wstring value = part.wstring();
            std::transform(value.begin(), value.end(), value.begin(),
                           [](wchar_t c) { return wchar_t(std::towlower(c)); });
            if (value == L"rkva-snapshots-v1" || value == L"rkva-snapshots-v2") {
                error = "legacy namespace forbidden";
                return false;
            }
        }
        state_->root_handle.reset(open_path(
            normalized, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
            OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS));
        if (state_->root_handle.h == INVALID_HANDLE_VALUE ||
            !directory(state_->root_handle.h)) {
            error = error_code("cannot open v3 root without reparse point", GetLastError());
            return false;
        }
        state_->root = normalized;
        error.clear();
        return true;
    } catch (const std::exception & ex) {
        error = ex.what();
        return false;
    }
#endif
}

bool windows_filesystem::sync_file(const fs::path & path, std::string & error) {
#if !defined(_WIN32) || !defined(RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST)
    (void) path;
    error = "Windows durable backend requires _WIN32 and "
            "RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST";
    return false;
#else
    if (!lock_held()) {
        error = "v3 lock unavailable";
        return false;
    }
    const auto target = path.lexically_normal();
    const auto parent = target.parent_path();
    if (parent.parent_path() != state_->root ||
        !stage_name(parent.filename().wstring())) {
        error = "v3 staged file path invalid";
        return false;
    }
    const auto name = target.filename().wstring();
    if (name != L"manifest-v2.bin" && !chunk_name(name)) {
        error = "v3 staged filename invalid";
        return false;
    }
    handle file;
    file.h = open_path(target, GENERIC_READ | GENERIC_WRITE,
                       FILE_SHARE_READ | FILE_SHARE_WRITE, OPEN_EXISTING, 0);
    if (file.h == INVALID_HANDLE_VALUE || !regular(file.h)) {
        error = error_code("cannot open staged regular file", GetLastError());
        return false;
    }
    if (!FlushFileBuffers(file.h)) {
        error = error_code("staged file flush failed", GetLastError());
        return false;
    }
    error.clear();
    return true;
#endif
}

bool windows_filesystem::sync_directory(const fs::path & path, std::string & error) {
#if !defined(_WIN32) || !defined(RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST)
    (void) path;
    error = "Windows durable backend requires _WIN32 and "
            "RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST";
    return false;
#else
    if (!lock_held()) {
        error = "v3 lock unavailable";
        return false;
    }
    const auto target = path.lexically_normal();
    if (target == state_->root) {
        if (!FlushFileBuffers(state_->root_handle.h)) {
            error = error_code("v3 root flush failed", GetLastError());
            return false;
        }
        error.clear();
        return true;
    }
    if (target.parent_path() != state_->root ||
        !stage_name(target.filename().wstring())) {
        error = "v3 staged directory path invalid";
        return false;
    }
    handle dir;
    dir.h = open_path(target, GENERIC_READ | GENERIC_WRITE,
                      FILE_SHARE_READ | FILE_SHARE_WRITE, OPEN_EXISTING,
                      FILE_FLAG_BACKUP_SEMANTICS);
    if (dir.h == INVALID_HANDLE_VALUE || !directory(dir.h)) {
        error = error_code("cannot open staged directory", GetLastError());
        return false;
    }
    if (!FlushFileBuffers(dir.h)) {
        error = error_code("staged directory flush unavailable", GetLastError());
        return false;
    }
    error.clear();
    return true;
#endif
}

move_result windows_filesystem::rename_noreplace_and_sync(const fs::path & from,
                                                          const fs::path & to,
                                                          std::string & error) {
#if !defined(_WIN32) || !defined(RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST)
    (void) from;
    (void) to;
    error = "Windows durable backend requires _WIN32 and "
            "RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST";
    return move_result::no_change;
#else
    if (!lock_held()) {
        error = "v3 lock unavailable";
        return move_result::no_change;
    }
    const auto source = from.lexically_normal();
    const auto destination = to.lexically_normal();
    if (source.parent_path() != state_->root ||
        destination.parent_path() != state_->root ||
        !stage_name(source.filename().wstring()) ||
        !hex_name(destination.filename().wstring())) {
        error = "v3 rename paths invalid";
        return move_result::no_change;
    }
    handle dir;
    dir.h = open_path(source, GENERIC_READ,
                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                      OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS);
    if (dir.h == INVALID_HANDLE_VALUE || !directory(dir.h)) {
        error = error_code("invalid staged directory", GetLastError());
        return move_result::no_change;
    }
    hook("before_move");
    // Do not request REPLACE_EXISTING or COPY_ALLOWED. A no-replace failure
    // leaves both trees untouched.
    if (!MoveFileExW(source.wstring().c_str(), destination.wstring().c_str(), 0)) {
        const DWORD code = GetLastError();
        error = error_code("v3 rename refused or failed", code);
        if (code == ERROR_ALREADY_EXISTS || code == ERROR_FILE_EXISTS)
            return move_result::no_change;
        return move_result::uncertain;
    }
    hook("after_move");
    if (!FlushFileBuffers(state_->root_handle.h)) {
        error = error_code("v3 parent flush failed after rename; outcome uncertain",
                           GetLastError());
        return move_result::uncertain;
    }
    hook("after_parent_flush");
    error = "v3 rename/flush observed, but directory power-loss durability is unqualified";
    return move_result::uncertain;
#endif
}

} // namespace rkva_managed_object_v3
