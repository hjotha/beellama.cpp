#include "rkva-peer-identity.h"

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <aclapi.h>
#include <bcrypt.h>
#include <sddl.h>
#include <wincrypt.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if defined(RKVA_PEER_IDENTITY_UWP)
#include <winrt/Windows.Storage.h>
#endif

namespace rkva_peer_identity {
namespace {

namespace fs = std::filesystem;

constexpr wchar_t namespace_directory[] = L"rkva-peer-identity-v3";
constexpr wchar_t lock_file[] = L"identity.lock";
constexpr wchar_t record_file[] = L"identity.json";
constexpr wchar_t pending_file[] = L"identity.pending";
constexpr wchar_t staging_file[] = L"identity.new";
constexpr char dpapi_entropy[] = "RKVA-PEER-IDENTITY-STORE-V1";

struct handle_guard {
    HANDLE value = INVALID_HANDLE_VALUE;
    handle_guard() = default;
    explicit handle_guard(HANDLE handle) : value(handle) {}
    ~handle_guard() {
        if (value != INVALID_HANDLE_VALUE) CloseHandle(value);
    }
    handle_guard(const handle_guard &) = delete;
    handle_guard & operator=(const handle_guard &) = delete;
};

HANDLE open_path(const fs::path & path, DWORD access, DWORD share, DWORD disposition,
                 DWORD flags) {
    CREATEFILE2_EXTENDED_PARAMETERS extended{};
    extended.dwSize = sizeof(extended);
    extended.dwFileAttributes = FILE_ATTRIBUTE_NORMAL;
    extended.dwFileFlags = flags | FILE_FLAG_OPEN_REPARSE_POINT;
    return CreateFile2(path.c_str(), access, share, disposition, &extended);
}

bool regular_file(HANDLE handle, std::string & error) {
    BY_HANDLE_FILE_INFORMATION info{};
    if (handle == INVALID_HANDLE_VALUE || GetFileType(handle) != FILE_TYPE_DISK ||
        !GetFileInformationByHandle(handle, &info) ||
        (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) ||
        (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || info.nNumberOfLinks != 1) {
        error = "peer identity file is not a single regular file";
        return false;
    }
    return true;
}

bool directory_handle(HANDLE handle, std::string & error) {
    BY_HANDLE_FILE_INFORMATION info{};
    if (handle == INVALID_HANDLE_VALUE || !GetFileInformationByHandle(handle, &info) ||
        (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) ||
        !(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
        error = "peer identity directory is unavailable";
        return false;
    }
    return true;
}

struct namespace_state {
    bool lock = false;
    bool record = false;
    bool pending = false;
    bool staging = false;
    bool unknown = false;
    bool exists = false;
};

bool inspect_namespace(const fs::path & root, namespace_state & state, std::string & error) {
    error = "peer identity namespace is ambiguous";
    const DWORD attributes = GetFileAttributesW(root.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES) {
        if (GetLastError() == ERROR_FILE_NOT_FOUND || GetLastError() == ERROR_PATH_NOT_FOUND) {
            state = namespace_state{};
            error.clear();
            return true;
        }
        return false;
    }
    if ((attributes & FILE_ATTRIBUTE_DIRECTORY) == 0 ||
        (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
        return false;
    }
    state = namespace_state{};
    state.exists = true;
    std::error_code code;
    for (const auto & entry : fs::directory_iterator(root, code)) {
        const std::wstring name = entry.path().filename().wstring();
        const DWORD entry_attributes = GetFileAttributesW(entry.path().c_str());
        if (entry_attributes == INVALID_FILE_ATTRIBUTES ||
            (entry_attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
            return false;
        }
        if (name == lock_file) {
            state.lock = true;
        } else if (name == record_file) {
            state.record = true;
        } else if (name == pending_file) {
            state.pending = true;
        } else if (name == staging_file) {
            state.staging = true;
        } else {
            state.unknown = true;
        }
    }
    if (code) return false;
    error.clear();
    return true;
}

bool flush_directory(const fs::path & root, std::string & error) {
    handle_guard directory(open_path(root, GENERIC_READ | GENERIC_WRITE,
                                     FILE_SHARE_READ | FILE_SHARE_WRITE, OPEN_EXISTING,
                                     FILE_FLAG_BACKUP_SEMANTICS));
    if (!directory_handle(directory.value, error) || !FlushFileBuffers(directory.value)) {
        error = "peer identity directory sync failed";
        return false;
    }
    return true;
}

bool write_all(HANDLE handle, const std::string & bytes) {
    size_t offset = 0;
    while (offset < bytes.size()) {
        DWORD written = 0;
        if (!WriteFile(handle, bytes.data() + offset, DWORD(bytes.size() - offset), &written,
                       nullptr) ||
            written == 0) {
            return false;
        }
        offset += written;
    }
    return FlushFileBuffers(handle) != 0;
}

bool step(const char * name, std::string & error) {
#ifdef RKVA_PEER_IDENTITY_TESTING
    if (test_step && !test_step(name)) {
        error = "peer identity write step was refused";
        return false;
    }
#else
    (void)name;
    (void)error;
#endif
    return true;
}

bool add_token_sid(HANDLE token, TOKEN_INFORMATION_CLASS information,
                   std::vector<std::vector<uint8_t>> & storage,
                   std::vector<PSID> & sids) {
    DWORD size = 0;
    GetTokenInformation(token, information, nullptr, 0, &size);
    if (size == 0) return false;
    std::vector<uint8_t> buffer(size);
    if (!GetTokenInformation(token, information, buffer.data(), size, &size)) return false;
    PSID sid = nullptr;
    if (information == TokenUser) {
        sid = reinterpret_cast<TOKEN_USER *>(buffer.data())->User.Sid;
    } else if (information == TokenAppContainerSid) {
        sid = reinterpret_cast<TOKEN_APPCONTAINER_INFORMATION *>(buffer.data())->TokenAppContainer;
    }
    if (sid == nullptr || !IsValidSid(sid)) return false;
    const DWORD length = GetLengthSid(sid);
    std::vector<uint8_t> copy(length);
    if (!CopySid(length, copy.data(), sid)) return false;
    storage.push_back(std::move(copy));
    sids.push_back(storage.back().data());
    return true;
}

// A protected (non-inherited) DACL with only the current user SID and, in an
// AppContainer, the package SID. No Everyone, Users or inherited ACE is kept.
bool apply_acl(HANDLE handle, bool directory, std::string & error) {
    error = "peer identity ACL could not be applied";
    HANDLE raw_token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw_token)) return false;
    handle_guard token(raw_token);
    std::vector<std::vector<uint8_t>> storage;
    std::vector<PSID> sids;
    if (!add_token_sid(token.value, TokenUser, storage, sids)) return false;
    // A desktop process has no package SID. An AppContainer process must have
    // one; failing to obtain it would silently leave the package SID unlisted,
    // so the ACL is refused instead.
    DWORD is_app_container = 0;
    DWORD is_app_size = 0;
    if (!GetTokenInformation(token.value, TokenIsAppContainer, &is_app_container,
                             sizeof(is_app_container), &is_app_size)) {
        return false;
    }
    if (is_app_container != 0 &&
        !add_token_sid(token.value, TokenAppContainerSid, storage, sids)) {
        return false;
    }

    std::vector<EXPLICIT_ACCESS_W> entries(sids.size());
    for (size_t i = 0; i < sids.size(); ++i) {
        EXPLICIT_ACCESS_W & entry = entries[i];
        ZeroMemory(&entry, sizeof(entry));
        entry.grfAccessPermissions = FILE_ALL_ACCESS;
        entry.grfAccessMode = SET_ACCESS;
        entry.grfInheritance = directory ? (SUB_CONTAINERS_AND_OBJECTS_INHERIT) : NO_INHERITANCE;
        entry.Trustee.TrusteeForm = TRUSTEE_IS_SID;
        entry.Trustee.TrusteeType = TRUSTEE_IS_UNKNOWN;
        entry.Trustee.ptstrName = reinterpret_cast<LPWSTR>(sids[i]);
    }
    PACL acl = nullptr;
    if (SetEntriesInAclW(DWORD(entries.size()), entries.data(), nullptr, &acl) != ERROR_SUCCESS) {
        return false;
    }
    const DWORD result = SetSecurityInfo(
        handle, SE_FILE_OBJECT,
        DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION, nullptr, nullptr, acl,
        nullptr);
    LocalFree(acl);
    if (result != ERROR_SUCCESS) return false;
    error.clear();
    return true;
}

std::string dpapi_seal(const key & value, std::string & error) {
    DATA_BLOB input{};
    input.cbData = DWORD(value.size());
    input.pbData = const_cast<BYTE *>(value.data());
    DATA_BLOB entropy{};
    entropy.cbData = sizeof(dpapi_entropy) - 1;
    entropy.pbData = reinterpret_cast<BYTE *>(const_cast<char *>(dpapi_entropy));
    DATA_BLOB output{};
    if (!CryptProtectData(&input, L"rkva-peer-identity-store", &entropy, nullptr, nullptr,
                          CRYPTPROTECT_UI_FORBIDDEN, &output)) {
        error = "peer identity integrity key could not be protected";
        return {};
    }
    std::string sealed = to_hex(output.pbData, output.cbData);
    SecureZeroMemory(output.pbData, output.cbData);
    LocalFree(output.pbData);
    error.clear();
    return sealed;
}

bool dpapi_unseal(const std::string & sealed, key & out, std::string & error) {
    error = "peer identity protected integrity key is unavailable";
    if (sealed.empty() || sealed.size() % 2 != 0 || sealed.size() > 8192) return false;
    std::vector<BYTE> input(sealed.size() / 2);
    if (!from_hex(sealed, input.data(), input.size())) return false;
    DATA_BLOB encrypted{};
    encrypted.cbData = DWORD(input.size());
    encrypted.pbData = input.data();
    DATA_BLOB entropy{};
    entropy.cbData = sizeof(dpapi_entropy) - 1;
    entropy.pbData = reinterpret_cast<BYTE *>(const_cast<char *>(dpapi_entropy));
    DATA_BLOB output{};
    if (!CryptUnprotectData(&encrypted, nullptr, &entropy, nullptr, nullptr,
                            CRYPTPROTECT_UI_FORBIDDEN, &output)) {
        return false;
    }
    bool ok = output.cbData == out.size();
    if (ok) std::memcpy(out.data(), output.pbData, out.size());
    SecureZeroMemory(output.pbData, output.cbData);
    LocalFree(output.pbData);
    SecureZeroMemory(input.data(), input.size());
    if (!ok) {
        error = "peer identity protected integrity key has the wrong size";
        return false;
    }
    error.clear();
    return true;
}

class windows_store final : public store {
public:
    explicit windows_store(fs::path local_state) : local_state_(std::move(local_state)) {}

    status read(std::string & bytes, std::string & error) override {
        const fs::path root = root_path();
        namespace_state state{};
        if (!inspect_namespace(root, state, error)) return status::corrupt;
        if (!state.exists) {
            error = "peer identity record is absent";
            return status::unknown;
        }
        if (state.unknown || state.pending || state.staging || (state.record && !state.lock)) {
            error = "peer identity namespace is ambiguous";
            return status::corrupt;
        }
        if (!state.record) {
            error = "peer identity record is absent";
            return status::unknown;
        }

        handle_guard lock(open_path(root / lock_file, GENERIC_READ,
                                     FILE_SHARE_READ | FILE_SHARE_WRITE, OPEN_EXISTING, 0));
        if (lock.value == INVALID_HANDLE_VALUE || !regular_file(lock.value, error)) {
            error = "peer identity lock file is unavailable";
            return status::corrupt;
        }
        OVERLAPPED overlap{};
        if (!LockFileEx(lock.value, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1,
                        0, &overlap)) {
            error = "peer identity store is locked";
            return status::unavailable;
        }
        handle_guard record(open_path(root / record_file, GENERIC_READ, FILE_SHARE_READ,
                                      OPEN_EXISTING, 0));
        LARGE_INTEGER size{};
        if (!regular_file(record.value, error) || !GetFileSizeEx(record.value, &size) ||
            size.QuadPart <= 0 || uint64_t(size.QuadPart) > max_record_bytes) {
            UnlockFileEx(lock.value, 0, 1, 0, &overlap);
            error = "peer identity record size is invalid";
            return status::corrupt;
        }
        bytes.resize(size_t(size.QuadPart));
        size_t offset = 0;
        while (offset < bytes.size()) {
            DWORD used = 0;
            if (!ReadFile(record.value, bytes.data() + offset, DWORD(bytes.size() - offset),
                          &used, nullptr) ||
                used == 0) {
                UnlockFileEx(lock.value, 0, 1, 0, &overlap);
                bytes.clear();
                error = "peer identity record read failed";
                return status::corrupt;
            }
            offset += used;
        }
        namespace_state after{};
        const bool stable = inspect_namespace(root, after, error) && after.record && after.lock &&
                            !after.pending && !after.staging && !after.unknown;
        UnlockFileEx(lock.value, 0, 1, 0, &overlap);
        if (!stable) {
            bytes.clear();
            error = "peer identity namespace changed during read";
            return status::corrupt;
        }
        error.clear();
        return status::ok;
    }

    status write(const std::string & bytes, std::string & error) override {
        if (bytes.empty() || bytes.size() > max_record_bytes) {
            error = "peer identity record size is invalid";
            return status::invalid_argument;
        }
        std::error_code code;
        if (!fs::is_directory(local_state_, code) || code) {
            error = "AppContainer LocalState directory is unavailable";
            return status::unavailable;
        }
        const fs::path root = root_path();
        namespace_state existing{};
        if (!inspect_namespace(root, existing, error)) return status::corrupt;
        if (!existing.exists) {
            if (!CreateDirectoryW(root.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) {
                error = "peer identity namespace could not be created";
                return status::unavailable;
            }
            if (!inspect_namespace(root, existing, error)) return status::corrupt;
        }
        handle_guard root_handle(open_path(root, FILE_READ_ATTRIBUTES | WRITE_DAC,
                                           FILE_SHARE_READ | FILE_SHARE_WRITE, OPEN_EXISTING,
                                           FILE_FLAG_BACKUP_SEMANTICS));
        if (!directory_handle(root_handle.value, error)) return status::unavailable;
        if (!apply_acl(root_handle.value, true, error)) return status::corrupt;

        handle_guard lock(open_path(root / lock_file, GENERIC_READ | GENERIC_WRITE | WRITE_DAC,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE, OPEN_ALWAYS, 0));
        if (lock.value == INVALID_HANDLE_VALUE || !regular_file(lock.value, error) ||
            !apply_acl(lock.value, false, error)) {
            error = "peer identity lock file is unavailable";
            return status::corrupt;
        }
        OVERLAPPED overlap{};
        if (!LockFileEx(lock.value, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1,
                        0, &overlap)) {
            error = "peer identity store is locked";
            return status::unavailable;
        }
        namespace_state before{};
        if (!inspect_namespace(root, before, error) || before.pending || before.staging ||
            before.unknown) {
            UnlockFileEx(lock.value, 0, 1, 0, &overlap);
            error = "peer identity namespace is ambiguous before write";
            return status::corrupt;
        }

        handle_guard pending(open_path(root / pending_file,
                                       GENERIC_READ | GENERIC_WRITE | WRITE_DAC,
                                       FILE_SHARE_READ | FILE_SHARE_WRITE, CREATE_NEW, 0));
        if (pending.value == INVALID_HANDLE_VALUE || !regular_file(pending.value, error) ||
            !apply_acl(pending.value, false, error) || !write_all(pending.value, "pending\n")) {
            UnlockFileEx(lock.value, 0, 1, 0, &overlap);
            error = "peer identity pending marker could not be written";
            return status::corrupt;
        }
        if (!step("after_pending", error)) {
            UnlockFileEx(lock.value, 0, 1, 0, &overlap);
            if (error.empty()) error = "peer identity write was interrupted";
            return status::corrupt;
        }

        handle_guard staging(open_path(root / staging_file,
                                       GENERIC_READ | GENERIC_WRITE | WRITE_DAC,
                                       FILE_SHARE_READ | FILE_SHARE_WRITE, CREATE_NEW, 0));
        if (staging.value == INVALID_HANDLE_VALUE || !regular_file(staging.value, error) ||
            !apply_acl(staging.value, false, error) || !write_all(staging.value, bytes)) {
            UnlockFileEx(lock.value, 0, 1, 0, &overlap);
            error = "peer identity staging file could not be written";
            return status::corrupt;
        }
        if (!step("after_staging", error)) {
            UnlockFileEx(lock.value, 0, 1, 0, &overlap);
            if (error.empty()) error = "peer identity write was interrupted";
            return status::corrupt;
        }
        // Windows rename/delete require the handles to be closed: they were
        // opened without FILE_SHARE_DELETE. Leaving them open would make the
        // atomic step fail after the staging file was already flushed.
        CloseHandle(staging.value);
        staging.value = INVALID_HANDLE_VALUE;
        CloseHandle(pending.value);
        pending.value = INVALID_HANDLE_VALUE;

        if (!MoveFileExW((root / staging_file).c_str(), (root / record_file).c_str(),
                         MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            UnlockFileEx(lock.value, 0, 1, 0, &overlap);
            error = "peer identity atomic rename failed";
            return status::corrupt;
        }
        if (!flush_directory(root, error)) {
            UnlockFileEx(lock.value, 0, 1, 0, &overlap);
            return status::corrupt;
        }
        if (!step("after_rename", error)) {
            UnlockFileEx(lock.value, 0, 1, 0, &overlap);
            if (error.empty()) error = "peer identity write was interrupted";
            return status::corrupt;
        }
        if (!DeleteFileW((root / pending_file).c_str())) {
            UnlockFileEx(lock.value, 0, 1, 0, &overlap);
            error = "peer identity pending marker could not be removed";
            return status::corrupt;
        }
        if (!flush_directory(root, error)) {
            UnlockFileEx(lock.value, 0, 1, 0, &overlap);
            return status::corrupt;
        }
        if (!step("after_pending_unlink", error)) {
            UnlockFileEx(lock.value, 0, 1, 0, &overlap);
            if (error.empty()) error = "peer identity write was interrupted";
            return status::corrupt;
        }
        namespace_state after{};
        if (!inspect_namespace(root, after, error) || after.pending || after.staging ||
            after.unknown || !after.record || !after.lock) {
            UnlockFileEx(lock.value, 0, 1, 0, &overlap);
            error = "peer identity namespace is incomplete after write";
            return status::corrupt;
        }
        UnlockFileEx(lock.value, 0, 1, 0, &overlap);
        error.clear();
        return status::ok;
    }

    status erase(std::string & error) override {
        const fs::path root = root_path();
        namespace_state state{};
        if (!inspect_namespace(root, state, error)) return status::corrupt;
        if (!state.exists) {
            error.clear();
            return status::ok;
        }
        if (state.unknown) {
            error = "peer identity namespace contains an unknown entry";
            return status::corrupt;
        }
        handle_guard lock(open_path(root / lock_file, GENERIC_READ | GENERIC_WRITE | WRITE_DAC,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE, OPEN_ALWAYS, 0));
        if (lock.value == INVALID_HANDLE_VALUE || !regular_file(lock.value, error)) {
            error = "peer identity lock file is unavailable";
            return status::corrupt;
        }
        OVERLAPPED overlap{};
        if (!LockFileEx(lock.value, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1,
                        0, &overlap)) {
            error = "peer identity store is locked";
            return status::unavailable;
        }
        for (const wchar_t * name : {record_file, pending_file, staging_file}) {
            if (!DeleteFileW((root / name).c_str()) && GetLastError() != ERROR_FILE_NOT_FOUND) {
                UnlockFileEx(lock.value, 0, 1, 0, &overlap);
                error = "peer identity entry could not be removed";
                return status::corrupt;
            }
        }
        if (!flush_directory(root, error)) {
            UnlockFileEx(lock.value, 0, 1, 0, &overlap);
            return status::corrupt;
        }
        namespace_state after{};
        if (!inspect_namespace(root, after, error) || after.record || after.pending ||
            after.staging || after.unknown) {
            UnlockFileEx(lock.value, 0, 1, 0, &overlap);
            error = "peer identity namespace is ambiguous after wipe";
            return status::corrupt;
        }
        UnlockFileEx(lock.value, 0, 1, 0, &overlap);
        error.clear();
        return status::ok;
    }

    bool new_key(key & out, std::string & error) override {
        if (BCryptGenRandom(nullptr, out.data(), ULONG(out.size()),
                            BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
            error = "peer identity randomness source failed";
            return false;
        }
        error.clear();
        return true;
    }

    bool seal_key(const key & value, std::string & sealed, std::string & error) override {
        sealed = dpapi_seal(value, error);
        return !sealed.empty();
    }

    bool unseal_key(const std::string & sealed, key & out, std::string & error) override {
        return dpapi_unseal(sealed, out, error);
    }

private:
    fs::path root_path() const { return local_state_ / namespace_directory; }
    fs::path local_state_;
};

}  // namespace

std::unique_ptr<store> open_local_state(std::string & error) {
#if defined(RKVA_PEER_IDENTITY_UWP)
    try {
        const auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
        const std::wstring path(folder.Path().c_str());
        if (path.empty()) {
            error = "AppContainer LocalState path is empty";
            return nullptr;
        }
        error.clear();
        return std::unique_ptr<store>(new windows_store(fs::path(path)));
    } catch (...) {
        error = "AppContainer LocalState is unavailable";
        return nullptr;
    }
#else
    error = "LocalState requires the UWP AppContainer build (RKVA_PEER_IDENTITY_UWP)";
    return nullptr;
#endif
}

#ifdef RKVA_PEER_IDENTITY_TESTING
std::unique_ptr<store> open_test_directory(const std::string & local_state,
                                           std::string & error) {
    if (local_state.empty()) {
        error = "test store root is empty";
        return nullptr;
    }
    error.clear();
    return std::unique_ptr<store>(new windows_store(fs::u8path(local_state)));
}
#endif

}  // namespace rkva_peer_identity

#endif  // _WIN32
