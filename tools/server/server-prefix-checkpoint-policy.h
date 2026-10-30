#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>

namespace server_prefix_checkpoint_policy {

enum class action { proceed, save };

// New split-prefix names encode P and C. This is only a conservative index
// prefilter, not authentication: the state reader validates the actual frame.
// Non-matching names are existing cache files and retain their old behavior.
inline bool checkpoint_geometry_compatible(std::string_view path, uint32_t p, uint32_t c) {
    const auto slash = path.find_last_of("/\\");
    const auto name = slash == std::string_view::npos ? path : path.substr(slash + 1);
    const auto is_hex = [](std::string_view s) {
        if (s.size() != 16) return false;
        for (const char ch : s) {
            if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
        }
        return true;
    };
    const auto parse_uint = [](std::string_view s, uint32_t & out) {
        if (s.empty()) return false;
        uint32_t value = 0;
        for (const char ch : s) {
            if (ch < '0' || ch > '9') return false;
            const uint32_t digit = uint32_t(ch - '0');
            if (value > (std::numeric_limits<uint32_t>::max() - digit) / 10) return false;
            value = value * 10 + digit;
        }
        out = value;
        return true;
    };
    if (name.size() < 42 || name.substr(0, 5) != "auto-" || name[21] != '-' ||
            name[38] != '-' || !is_hex(name.substr(5, 16)) || !is_hex(name.substr(22, 16))) {
        return true;
    }
    const auto marker = name.find("-split-", 39);
    if (marker == std::string_view::npos) return true;
    uint32_t count = 0, source_p = 0, source_c = 0;
    const auto p_end = name.find('-', marker + 7);
    const auto c_end = p_end == std::string_view::npos ? p_end : name.find('-', p_end + 1);
    if (p_end == std::string_view::npos || c_end == std::string_view::npos ||
            !parse_uint(name.substr(39, marker - 39), count) ||
            !parse_uint(name.substr(marker + 7, p_end - marker - 7), source_p) ||
            !parse_uint(name.substr(p_end + 1, c_end - p_end - 1), source_c) ||
            name.size() != c_end + 21 || name.substr(name.size() - 4) != ".bin" ||
            !is_hex(name.substr(c_end + 1, 16))) {
        return true;
    }
    return source_p == p && source_c == c;
}

// Only save when prefill has naturally finished a batch exactly at P. Never
// split a batch to create a checkpoint. The caller checks the live memory.
inline action decide(const char * opt_in, bool eligible, uint32_t p,
        std::size_t prompt_tokens, std::size_t queued_tokens,
        std::size_t total_tokens, std::size_t block) {
    if (!opt_in || std::strcmp(opt_in, "1") != 0 || !eligible || !p ||
            block > p || total_tokens <= p || prompt_tokens != p || queued_tokens != 0) {
        return action::proceed;
    }
    return action::save;
}

} // namespace server_prefix_checkpoint_policy
