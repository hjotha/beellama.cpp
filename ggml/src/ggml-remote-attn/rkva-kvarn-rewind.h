#pragma once
#include "remote-attn-protocol.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <utility>
#include <vector>

// Process-lifetime configuration epochs survive engine/session recreation.
// Each configuration reserves 2^32 mutation tickets, then fails closed instead
// of carrying its counter into another configuration's namespace.
inline bool rkva_kvarn_generation_mutable(uint64_t generation) {
    return generation != 0 && uint32_t(generation) != UINT32_MAX;
}
inline uint64_t rkva_kvarn_new_generation_seed() {
    static std::atomic<uint64_t> next_epoch{1};
    uint64_t epoch = next_epoch.load(std::memory_order_relaxed);
    while (epoch <= UINT32_MAX) {
        if (next_epoch.compare_exchange_weak(epoch, epoch + 1, std::memory_order_relaxed))
            return epoch << 32;
    }
    return 0;
}

// Shared CPU parser for the atomic all-layer native rewind transaction. No
// live state is touched until this complete plan has passed validation.
struct rkva_kvarn_rewind_tail_span {
    uint32_t begin, rows;
};
inline rkva_kvarn_rewind_tail_span rkva_kvarn_rewind_tail(uint32_t before, uint32_t keep,
                                                          uint32_t tail) {
    const uint32_t begin = keep > tail ? keep - tail : 0;
    const uint32_t end = std::min(keep, before > tail ? before - tail : 0);
    return {begin, tail && end > begin ? end - begin : 0};
}
struct rkva_kvarn_rewind_plan {
    std::vector<rkva_kvarn_layer_rewind> layers;
    std::vector<size_t> tail_offsets;
};
inline bool rkva_kvarn_plan_rewind(const rkva_kvarn_range_rewind& request, uint32_t origin,
                                   uint32_t capacity, uint32_t heads, uint32_t tail_tokens,
                                   uint64_t generation, const std::vector<uint32_t>& counts,
                                   const std::vector<bool>& ready, const uint8_t* data,
                                   size_t bytes, rkva_kvarn_rewind_plan& plan) {
    if (!rkva_kvarn_generation_mutable(generation) || request.expected_generation != generation ||
        !data || !heads || request.reserved || request.pos0 > uint32_t(INT32_MAX) ||
        request.n_layers != counts.size() || ready.size() != counts.size() ||
        request.tail_tokens != tail_tokens || (tail_tokens != 0 && tail_tokens != 128))
        return false;
    const uint32_t keep = request.pos0 <= origin ? 0 : std::min(request.pos0 - origin, capacity);
    rkva_kvarn_rewind_plan candidate;
    candidate.layers.resize(counts.size());
    candidate.tail_offsets.resize(counts.size());
    size_t cursor = 0;
    for (size_t il = 0; il < counts.size(); ++il) {
        if (bytes - cursor < sizeof(rkva_kvarn_layer_rewind))
            return false;
        auto& layer = candidate.layers[il];
        std::memcpy(&layer, data + cursor, sizeof(layer));
        cursor += sizeof(layer);
        if (layer.expected_live != counts[il] || layer.expected_live > capacity ||
            layer.new_live != std::min(layer.expected_live, keep) ||
            layer.expected_live - layer.new_live > 128 || (layer.expected_live && !ready[il]))
            return false;
        const uint64_t tail_bytes =
            uint64_t(
                rkva_kvarn_rewind_tail(layer.expected_live, layer.new_live, tail_tokens).rows) *
            heads * 256 * 2 * 2;
        if (tail_bytes > bytes - cursor)
            return false;
        candidate.tail_offsets[il] = cursor;
        cursor += size_t(tail_bytes);
    }
    if (cursor != bytes)
        return false;
    plan = std::move(candidate);
    return true;
}
