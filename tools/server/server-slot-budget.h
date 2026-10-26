#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <istream>
#include <limits>
#include <sstream>
#include <string>

// A valid zero means no memory is available, not that measurement failed.
inline bool server_slot_parse_mem_available(std::istream & input, uint64_t & bytes) {
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string key, value, unit, extra;
        if (!(row >> key) || key != "MemAvailable:") continue;
        if (!(row >> value >> unit) || unit != "kB" || (row >> extra)) return false;
        uint64_t kb = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), kb);
        if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size() ||
                kb > std::numeric_limits<uint64_t>::max() / 1024ULL) return false;
        bytes = kb * 1024ULL;
        return true;
    }
    return false;
}

inline uint64_t server_slot_cap_ram_budget(uint64_t available, uint64_t configured, bool measured) {
    if (measured) available -= available / 4;
    return configured ? std::min(configured, available) : available;
}
