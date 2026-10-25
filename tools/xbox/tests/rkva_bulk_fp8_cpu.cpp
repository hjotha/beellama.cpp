// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "../uwp/rkva/rkva-bulk-fp8.h"
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

static uint8_t nearest_fp8_f16(uint16_t h) {
    const bool negative = (h & 0x8000u) != 0;
    const double magnitude = rkva_kvarn_bulk::f16_bits_to_f32(h & 0x7fffu);
    double best_distance = std::numeric_limits<double>::infinity();
    unsigned best = 0;
    for (unsigned code = 0; code <= 0x7b; ++code) {
        const double value = rkva_kvarn_bulk::f16_bits_to_f32(uint16_t(code << 8));
        const double distance = std::abs(magnitude - value);
        if (distance < best_distance || (distance == best_distance && !(code & 1))) {
            best_distance = distance;
            best = code;
        }
    }
    return uint8_t(best | (negative ? 0x80 : 0));
}

static std::vector<uint8_t> probe(uint32_t heads) {
    const size_t source = size_t(heads) * 257 * 4;
    std::vector<uint8_t> bytes(sizeof(rkva_probe_req) + 8 + source);
    const rkva_probe_req req{5, 0, uint32_t(source + 8), 0};
    std::memcpy(bytes.data(), &req, sizeof(req));
    std::memcpy(bytes.data() + sizeof(req), &heads, 4);
    return bytes;
}

int main() {
    unsigned finite = 0, rejected = 0;
    for (unsigned bits = 0; bits <= 65535; ++bits) {
        const uint16_t h = uint16_t(bits);
        if (!rkva_kvarn_bulk::f16_bits_finite(h)) {
            ++rejected;
            continue;
        }
        assert(rkva_kvarn_bulk::f16_to_fp8_bits(h) == nearest_fp8_f16(h));
        ++finite;
    }
    assert(finite == 63488 && rejected == 2048);
    // A direct F32->FP8 conversion rounds this value UP. The deployed two-step
    // contract first rounds it to the F16 midpoint and then chooses even 0x3c.
    const float above_midpoint = std::nextafter(1.125f, 2.0f);
    const uint16_t half = rkva_kvarn_bulk::f32_to_f16_bits(above_midpoint);
    assert(half == 0x3c80);
    assert(rkva_kvarn_bulk::f16_to_fp8_bits(half) == 0x3c);
    assert(!rkva_kvarn_bulk::f16_bits_finite(rkva_kvarn_bulk::f32_to_f16_bits(65520.0f)));
    assert(rkva_kvarn_bulk::f16_bits_finite(
        rkva_kvarn_bulk::f32_to_f16_bits(std::nextafter(65520.0f, 0.0f))));
    for (float invalid :
         {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::quiet_NaN()})
        assert(!rkva_kvarn_bulk::f16_bits_finite(rkva_kvarn_bulk::f32_to_f16_bits(invalid)));
    for (uint32_t heads : {1u, 2u, 3u, 4u, 24u, 25u, 1023u, 1024u}) {
        auto bytes = probe(heads);
        rkva_bulk_fp8::ProbeLayout layout;
        assert(rkva_bulk_fp8::validate_probe(bytes.data(), bytes.size(), layout));
        assert(layout.groups == (heads + 3) / 4);
        assert(layout.readback_bytes == size_t(heads) * 260 + size_t(layout.groups) * 4);
        assert(layout.source_offset + layout.source_bytes == bytes.size());
        assert(!rkva_bulk_fp8::validate_probe(bytes.data(), bytes.size() - 1, layout));
        bytes.push_back(0);
        assert(!rkva_bulk_fp8::validate_probe(bytes.data(), bytes.size(), layout));
    }
    for (uint32_t heads : {0u, 1025u}) {
        auto bytes = probe(heads);
        rkva_bulk_fp8::ProbeLayout layout;
        assert(!rkva_bulk_fp8::validate_probe(bytes.data(), bytes.size(), layout));
    }
    auto good = probe(1);
    for (size_t field : {0u, 4u, 8u, 12u, 20u}) {
        auto bad = good;
        bad[field] ^= 0x80;
        rkva_bulk_fp8::ProbeLayout layout;
        assert(!rkva_bulk_fp8::validate_probe(bad.data(), bad.size(), layout));
    }
    for (size_t bytes = 0; bytes < sizeof(rkva_probe_req) + 8; ++bytes) {
        rkva_bulk_fp8::ProbeLayout layout;
        assert(!rkva_bulk_fp8::validate_probe(good.data(), bytes, layout));
    }
    std::puts("rkva_bulk_fp8_cpu: 63488 finite F16 patterns, 2048 rejected patterns; nearest-even "
              "composition/overflow and bounded diagnostic layouts passed");
}
