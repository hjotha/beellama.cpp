#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

// A RESET carries only its newest tail rows, so its first query batch must fit
// inside that tail. Later APPEND segments retain the full query history.
inline std::vector<size_t> rkva_bulk_segments(size_t queries, bool serial_kvarn,
                                              bool bootstrap, uint32_t tail, size_t segment = 8) {
    segment = std::max<size_t>(1, segment);
    // The first native parent establishes the serial-prefix session. Every
    // segment of that parent must stay at the qualified MQ2 width (8); only
    // the following parents may use the wider segment.
    const size_t effective_segment = (serial_kvarn && bootstrap) ? 8 : segment;
    const size_t base = serial_kvarn ? (queries > 9 ? std::min(effective_segment, queries) : queries) : 1;
    const size_t first_base = serial_kvarn ? (queries > 9 ? 8 : queries) : 1;
    size_t first = std::min(first_base, base);
    if (serial_kvarn && bootstrap && tail && tail < first) first = tail;
    const size_t rest = first < base && queries - first > 63 * base ? 9 : base;
    std::vector<size_t> result;
    if (!queries) return result;
    result.push_back(first);
    for (size_t remaining = queries - first; remaining; ) {
        const size_t chunk = std::min(rest, remaining);
        result.push_back(chunk);
        remaining -= chunk;
    }
    return result;
}
