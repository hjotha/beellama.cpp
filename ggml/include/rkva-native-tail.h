#pragma once

#include <cstdint>

// The history ring must cover the exact tail and up to 128 rows of rewind or
// in-flight serial-prefix queries. Zero indicates an unrepresentable size.
inline uint32_t rkva_native_history_rows(uint32_t tail_tokens) {
    const uint64_t expanded = uint64_t(tail_tokens) + 128;
    const uint64_t needed = expanded < 256 ? 256 : expanded;
    uint64_t rows = 256;
    while (rows < needed && rows <= UINT32_MAX / 2u) rows *= 2;
    return rows >= needed && rows <= UINT32_MAX ? uint32_t(rows) : 0;
}
