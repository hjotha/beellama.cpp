#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace server_direct_disk_policy {

// The preset takes precedence over the legacy environment variable. Invalid
// environment values leave the optimization off instead of wrapping a P.
inline uint32_t configured_split(uint32_t preset, const char * env) {
    if (preset != 0) {
        return preset;
    }
    if (!env || !*env) {
        return 0;
    }
    uint64_t p = 0;
    for (const char * c = env; *c; ++c) {
        if (*c < '0' || *c > '9') {
            return 0;
        }
        const uint32_t digit = uint32_t(*c - '0');
        if (p > (std::numeric_limits<uint32_t>::max() - digit) / 10) {
            return 0;
        }
        p = p * 10 + digit;
    }
    return p != 0 && p % 256 == 0 ? uint32_t(p) : 0;
}

inline bool eligible(const char * opt_in, bool is_xxxl, std::size_t pure_prefix, uint32_t split_p) {
    return opt_in && std::strcmp(opt_in, "1") == 0 && is_xxxl &&
           pure_prefix > 0 && split_p > 0 && pure_prefix <= split_p;
}

} // namespace server_direct_disk_policy
