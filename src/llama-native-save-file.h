#pragma once

// Atomic publication of host native-overflow slot files. Kept platform-local so
// the state code does not duplicate race-prone rename logic and so the UTF-8
// path conversion can be unit-tested on the host.

#include "ggml.h"
#include "llama-mmap.h"
#include "rkva-snapshot-digest.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace llama_native_save {

inline bool hash_file_prefix(llama_file & file, size_t end, uint8_t out_sha256[32]) {
    const size_t position = file.tell();
    const size_t size = file.size();
    if (end > size || position > size) {
        return false;
    }
    rkva_snapshot::Sha256 hash;
    std::vector<uint8_t> buffer(1u << 20);
    file.seek(0, SEEK_SET);
    for (size_t left = end; left;) {
        const size_t count = std::min(left, buffer.size());
        file.read_raw(buffer.data(), count);
        hash.update(buffer.data(), count);
        left -= count;
    }
    file.seek(position, SEEK_SET);
    const auto digest = hash.finish();
    std::memcpy(out_sha256, digest.data(), digest.size());
    return true;
}

inline std::string temporary_path(const char * filepath) {
    static std::atomic<uint64_t> nonce{0};
    return std::string(filepath) + ".tmp-native-" +
           std::to_string((unsigned long long) ggml_time_us()) + "-" +
           std::to_string((unsigned long long) nonce.fetch_add(1));
}

#ifndef _WIN32
inline bool publish(const std::string & temporary, const char * destination) {
    std::error_code ec;
    std::filesystem::rename(temporary, destination, ec);
    return !ec;
}
#else
inline bool utf8_to_wide(const std::string & text, std::wstring & out) {
    if (text.empty()) {
        out.clear();
        return true;
    }
    const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                                         int(text.size()), nullptr, 0);
    if (size <= 0) return false;
    out.resize(size_t(size));
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                               int(text.size()), out.data(), size) == size;
}

inline bool publish(const std::string & temporary, const char * destination) {
    std::wstring wide_temporary;
    std::wstring wide_destination;
    if (!utf8_to_wide(temporary, wide_temporary) ||
        !utf8_to_wide(destination, wide_destination))
        return false;
    return MoveFileExW(wide_temporary.c_str(), wide_destination.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
}
#endif

inline bool remove_path(const std::string & path) {
#ifndef _WIN32
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return !ec;
#else
    std::wstring wide;
    if (!utf8_to_wide(path, wide))
        return false;
    return DeleteFileW(wide.c_str()) != 0 || GetLastError() == ERROR_FILE_NOT_FOUND;
#endif
}

struct remove_guard {
    std::string path;
    bool released = false;
    ~remove_guard() {
        if (!released && !path.empty())
            (void) remove_path(path);
    }
};

} // namespace llama_native_save
