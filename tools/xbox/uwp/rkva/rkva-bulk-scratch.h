// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include "rkva-kvarn-bulk.h"
#include <array>
#include <string>

namespace rkva_kvarn_bulk {
// One instance per existing serial engine/listener owner. Capacity survives calls;
// nothing here is global, shared with a second request, or retained by the GPU.
struct Scratch {
    std::vector<Segment> segments;
    std::vector<float> queries, lse;
    std::vector<uint16_t> output_f16;
    struct Shape {
        uint64_t calls = 0, growth_calls = 0, reallocations = 0, grown_bytes = 0;
    };
    std::array<std::array<Shape, 257>, 3> shapes{}; // F16, F32, FP8; bounded nq
    std::array<size_t, 4> capacities() const {
        return {segments.capacity() * sizeof(Segment), queries.capacity() * sizeof(float),
                lse.capacity() * sizeof(float), output_f16.capacity() * sizeof(uint16_t)};
    }
    void record(uint32_t nq, uint32_t wire, const std::array<size_t, 4>& before) {
        if (!nq || nq > 256 ||
            (wire != RKVA_WIRE_F16 && wire != RKVA_WIRE_F32 && wire != RKVA_WIRE_FP8))
            return;
        auto& shape = shapes[wire == RKVA_WIRE_FP8 ? 2 : wire == RKVA_WIRE_F32 ? 1 : 0][nq];
        ++shape.calls;
        bool grew = false;
        const auto after = capacities();
        for (size_t i = 0; i < before.size(); ++i) {
            if (after[i] > before[i]) {
                ++shape.reallocations;
                shape.grown_bytes += after[i] - before[i];
                grew = true;
            }
        }
        shape.growth_calls += grew;
    }
    void reset_counters() {
        shapes = {};
    }
    std::string json() const {
        std::string out = "[";
        bool first = true;
        for (size_t w = 0; w < shapes.size(); ++w) {
            for (size_t nq = 1; nq < shapes[w].size(); ++nq) {
                const auto& s = shapes[w][nq];
                if (!s.calls)
                    continue;
                if (!first)
                    out += ',';
                first = false;
                out += "{\"wire\":" +
                       std::to_string(w == 2   ? RKVA_WIRE_FP8
                                      : w == 1 ? RKVA_WIRE_F32
                                               : RKVA_WIRE_F16) +
                       ",\"nq\":" + std::to_string(nq) + ",\"calls\":" + std::to_string(s.calls) +
                       ",\"growth_calls\":" + std::to_string(s.growth_calls) +
                       ",\"reallocations\":" + std::to_string(s.reallocations) +
                       ",\"grown_bytes\":" + std::to_string(s.grown_bytes) + "}";
            }
        }
        return out + "]";
    }
};
struct ScratchCall {
    Scratch& scratch;
    uint32_t nq, wire;
    std::array<size_t, 4> before;
    ScratchCall(Scratch& s, uint32_t n, uint32_t w)
        : scratch(s), nq(n), wire(w), before(s.capacities()) {}
    ~ScratchCall() {
        scratch.record(nq, wire, before);
    }
};
} // namespace rkva_kvarn_bulk
