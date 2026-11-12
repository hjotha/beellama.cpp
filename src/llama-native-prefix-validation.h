#pragma once

#include "llama-kv-mixed-state.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

// A native snapshot represents the complete evaluated prefix. A matching
// RKVK row count is insufficient if its memory body has missing or duplicate
// logical positions, or if its cells belong to different sequences.
inline void llama_validate_native_prefix(const std::vector<llama_kv_mixed::cell_entry> & cells,
        uint32_t count, uint32_t capacity, uint32_t split, llama_seq_id expected_seq = 0) {
    if (count <= split || count > capacity) {
        throw std::runtime_error("native FULL body has no valid evaluated prefix extent");
    }
    std::vector<uint8_t> seen(count, 0);
    uint32_t occupied = 0;
    llama_seq_id source_seq = expected_seq;
    for (const auto & cell : cells) {
        if (cell.pos == -1 && cell.seq_ids.empty()) continue;
        if (cell.pos < 0) {
            throw std::runtime_error("native FULL body has an invalid empty cell");
        }
        const auto pos = uint32_t(cell.pos);
        if (cell.seq_ids.size() != 1 || cell.seq_ids[0] < 0) {
            throw std::runtime_error("native FULL body cells do not match the evaluated prefix");
        }
        if (source_seq < 0) source_seq = cell.seq_ids[0];
        if (pos >= count || seen[pos] || cell.seq_ids[0] != source_seq) {
            throw std::runtime_error("native FULL body cells do not match the evaluated prefix");
        }
        seen[pos] = 1;
        ++occupied;
    }
    if (occupied != count) {
        std::string missing_str;
        for (uint32_t i = 0; i < count; ++i) {
            if (!seen[i]) {
                missing_str += " " + std::to_string(i);
                if (missing_str.size() > 200) { missing_str += "..."; break; }
            }
        }
        throw std::runtime_error("native FULL body has a gap in the evaluated prefix: occupied=" +
            std::to_string(occupied) + " count=" + std::to_string(count) + " missing=" + missing_str);
    }
}
