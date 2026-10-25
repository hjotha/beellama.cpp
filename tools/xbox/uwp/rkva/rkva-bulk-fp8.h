// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once
#include "rkva-kvarn-bulk.h"
#include <algorithm>

namespace rkva_bulk_fp8 {
inline uint32_t pack_groups(uint32_t elements, uint32_t heads) {
    return (std::max(elements / 4, heads) + 255u) / 256u;
}
inline uint64_t readback_bytes(uint32_t elements, uint32_t heads) {
    return uint64_t(elements) + uint64_t(heads) * 4 + uint64_t(pack_groups(elements, heads)) * 4;
}
struct ProbeLayout {
    uint32_t heads = 0, elements = 0, groups = 0;
    size_t source_offset = 0, source_bytes = 0, readback_bytes = 0;
};
// Diagnostic op11/id5: rkva_probe_req then {heads,reserved} then OUT F32 + LSE F32.
// No session/KV reference; bounded at 1024 independent 256-element vectors.
inline bool validate_probe(const uint8_t* data, size_t bytes, ProbeLayout& out) {
    out = {};
    if (!data || bytes < sizeof(rkva_probe_req) + 8)
        return false;
    rkva_probe_req request{};
    uint32_t header[2]{};
    std::memcpy(&request, data, sizeof(request));
    std::memcpy(header, data + sizeof(request), sizeof(header));
    if (request.probe_id != 5 || request.mode || request.reserved || header[1] || !header[0] ||
        header[0] > 1024)
        return false;
    const uint64_t source = uint64_t(header[0]) * (256 + 1) * 4;
    if (request.blob_bytes != source + sizeof(header) ||
        bytes != sizeof(request) + sizeof(header) + source)
        return false;
    out.heads = header[0];
    out.elements = out.heads * 256;
    out.groups = pack_groups(out.elements, out.heads);
    out.source_offset = sizeof(request) + sizeof(header);
    out.source_bytes = size_t(source);
    out.readback_bytes = size_t(readback_bytes(out.elements, out.heads));
    return true;
}
} // namespace rkva_bulk_fp8
