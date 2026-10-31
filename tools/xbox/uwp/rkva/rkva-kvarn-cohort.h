// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include "rkva-kvarn-bulk.h"
#include "rkva-kvarn-rewind.h"
#include <array>

namespace rkva_kvarn_cohort {
inline constexpr uint32_t max_entries = 4, max_queries = 256, max_segments = 32;
inline constexpr uint64_t max_bytes = 256ull << 20;

// Snapshot copied by the existing serial engine owner. No GPU resources or
// mutable references escape admission; ready=false permits an initial RESET.
struct LayerState {
    uint32_t live = 0, groups = 0;
    bool ready = false;
    rkva_kvarn_range::prefix_coverage coverage{};
};
struct Entry {
    rkva_kvarn_cohort_entry wire{};
    size_t body_offset = 0;
    uint32_t format = 0, segments = 0, live_after = 0, reply_bytes = 0;
    uint64_t generation_before = 0, generation_after = 0;
};
struct Plan {
    rkva_kvarn_cohort_header header{};
    std::array<Entry, max_entries> entries{};
    uint32_t segments = 0;
    size_t reply_bytes = 0;
    uint64_t generation_after = 0;
};

// Snapshot(layer, descriptor, state) must be read-only and return false for
// an invalid session/layer. Only a fully validated plan is published. Scratch
// capacity may grow, but no resident state or output completion is changed.
template <class Snapshot>
bool validate(const uint8_t* data, size_t bytes, uint32_t outer_queries, uint32_t n_layers,
              uint32_t n_head, uint32_t n_head_kv, uint64_t generation, Snapshot snapshot,
              Plan& plan, std::vector<rkva_kvarn_bulk::Segment>& scratch) {
    plan = {};
    if (!data || bytes < sizeof(rkva_kvarn_cohort_header) || bytes > max_bytes || !n_layers ||
        n_layers > 128 || !n_head || n_head > 256 || !n_head_kv || n_head_kv > 256 ||
        n_head % n_head_kv || !rkva_kvarn_generation_mutable(generation))
        return false;
    Plan candidate;
    std::memcpy(&candidate.header, data, sizeof(candidate.header));
    const auto& h = candidate.header;
    if (h.version != 1 || !h.entry_count || h.entry_count > max_entries || h.reserved ||
        !h.total_queries || h.total_queries > max_queries || h.total_queries != outer_queries)
        return false;
    size_t cursor = sizeof(h);
    uint32_t queries = 0;
    uint64_t reply_bytes = sizeof(h), ticket = 0, epoch = 0;
    for (uint32_t i = 0; i < h.entry_count; ++i) {
        if (bytes - cursor < sizeof(rkva_kvarn_cohort_entry))
            return false;
        auto& e = candidate.entries[i];
        std::memcpy(&e.wire, data + cursor, sizeof(e.wire));
        cursor += sizeof(e.wire);
        const auto& w = e.wire;
        if (w.layer >= n_layers || !w.n_queries || w.n_queries > h.total_queries - queries ||
            w.reserved || !w.ticket || w.ticket <= ticket || !w.epoch || (i && w.epoch != epoch) ||
            w.body_bytes > bytes - cursor)
            return false;
        for (uint32_t j = 0; j < i; ++j)
            if (candidate.entries[j].wire.layer == w.layer)
                return false;
        ticket = w.ticket;
        epoch = w.epoch;
        e.body_offset = cursor;
        rkva_kvarn_range_config descriptor{};
        LayerState resident;
        if (!snapshot(w.layer, descriptor, resident) ||
            !rkva_kvarn_bulk::peek_wire(data + cursor, w.body_bytes, &e.format) ||
            (e.format != RKVA_WIRE_F32 && e.format != RKVA_WIRE_F16 && e.format != RKVA_WIRE_FP8) ||
            !rkva_kvarn_bulk::validate_wire(data + cursor, w.body_bytes, w.n_queries, descriptor,
                                            n_head, n_head_kv, resident.live, resident.groups,
                                            resident.ready, resident.coverage, generation, e.format,
                                            scratch))
            return false;
        e.segments = static_cast<uint32_t>(scratch.size());
        if (e.segments > max_segments - candidate.segments ||
            e.segments > uint64_t(UINT32_MAX) - uint32_t(generation))
            return false;
        e.generation_before = generation;
        generation += e.segments;
        e.generation_after = generation;
        e.live_after = scratch.back().upload.live_count;
        const uint64_t element_bytes = e.format == RKVA_WIRE_FP8   ? 1
                                       : e.format == RKVA_WIRE_F16 ? 2
                                                                   : 4;
        const uint64_t heads = uint64_t(w.n_queries) * n_head;
        const uint64_t result_bytes = heads * (256 * element_bytes + sizeof(float));
        reply_bytes += sizeof(w) + result_bytes;
        if (reply_bytes > max_bytes)
            return false;
        e.reply_bytes = static_cast<uint32_t>(result_bytes);
        candidate.segments += e.segments;
        queries += w.n_queries;
        cursor += w.body_bytes;
    }
    if (cursor != bytes || queries != h.total_queries)
        return false;
    candidate.reply_bytes = static_cast<size_t>(reply_bytes);
    candidate.generation_after = generation;
    plan = candidate;
    return true;
}

// Extra cohort publication gate. The existing engine retains the exact OUT
// conversion contract; this only rejects malformed/nonfinite success bodies.
inline bool valid_reply(const Entry& e, const uint8_t* data, size_t bytes, uint32_t n_head) {
    if (!data || bytes != e.reply_bytes || !e.wire.n_queries || e.wire.n_queries > max_queries ||
        !n_head || n_head > 256 ||
        (e.format != RKVA_WIRE_FP8 && e.format != RKVA_WIRE_F16 && e.format != RKVA_WIRE_F32))
        return false;
    const uint64_t elements = uint64_t(e.wire.n_queries) * n_head * 256;
    const size_t elem_bytes = e.format == RKVA_WIRE_FP8 ? 1 : e.format == RKVA_WIRE_F16 ? 2 : 4;
    const uint64_t out_bytes = elements * elem_bytes;
    if (out_bytes + uint64_t(e.wire.n_queries) * n_head * sizeof(float) != bytes)
        return false;
    for (size_t offset = 0; offset < out_bytes; offset += elem_bytes) {
        if (e.format == RKVA_WIRE_FP8) {
            if ((data[offset] & 0x7cu) == 0x7cu)
                return false;
        } else if (e.format == RKVA_WIRE_F16) {
            uint16_t value;
            std::memcpy(&value, data + offset, sizeof(value));
            if (!rkva_kvarn_bulk::f16_bits_finite(value))
                return false;
        } else if (e.format == RKVA_WIRE_F32) {
            float value;
            std::memcpy(&value, data + offset, sizeof(value));
            if (!std::isfinite(value))
                return false;
        } else {
            return false;
        }
    }
    for (size_t offset = static_cast<size_t>(out_bytes); offset < bytes; offset += sizeof(float)) {
        float value;
        std::memcpy(&value, data + offset, sizeof(value));
        if (!std::isfinite(value))
            return false;
    }
    return true;
}
} // namespace rkva_kvarn_cohort
