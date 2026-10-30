#pragma once
#include <array>
#include <filesystem>
#include <string>
#include <system_error>

// Filesystem preflight only; caller must hold the matching exclusive store
// lock until publication/cleanup completes. A suffix does not prove ownership:
// block every present sidecar and every target which aliases a present base.
// A dangling symlink is present. Unknown/error entries fail closed. The caller
// separately validates the target and must prevent noncooperating writers.
namespace server_slot_path_collision {
inline bool absent(const std::string & path) {
    std::error_code ec;
    const auto state = std::filesystem::symlink_status(std::filesystem::path(path), ec);
    if (ec == std::errc::no_such_file_or_directory) return true;
    return !ec && state.type() == std::filesystem::file_type::not_found;
}
inline bool has_unresolved_staging(const std::string & path) {
    const auto parent = std::filesystem::path(path).parent_path();
    const auto directory = parent.empty() ? std::filesystem::path(".") : parent;
    std::error_code ec;
    std::filesystem::directory_iterator it(directory, ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        const auto name = it->path().filename().string();
        if (name.find(".tmp-rollback-") != std::string::npos ||
                name.rfind(".legacy-slot-stage-", 0) == 0 ||
                name.rfind("retention.tmp-", 0) == 0) return true;
    }
    return bool(ec); // unreadable or incomplete enumeration fails closed
}
inline bool clear(const std::string & path) {
    if (path.empty() || has_unresolved_staging(path)) return false;
    constexpr std::array<const char *, 4> suffixes = {{".logits", ".meta", ".ckpt", ".ckpts"}};
    for (const char * suffix : suffixes) {
        const std::string ending(suffix);
        if (!absent(path + ending)) return false;
        if (path.size() > ending.size() &&
                path.compare(path.size() - ending.size(), ending.size(), ending) == 0 &&
                !absent(path.substr(0, path.size() - ending.size()))) return false;
    }
    return true;
}
} // namespace server_slot_path_collision
