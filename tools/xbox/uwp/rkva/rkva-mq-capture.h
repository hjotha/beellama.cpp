#pragma once
// One bounded native MQ2 snapshot. File fields are little-endian on Xbox.
#include <cstdint>
#include <ostream>
#include "rkva-snapshot-digest.h"

namespace rkva_mq_capture {
constexpr uint64_t max_bytes = 512ull << 20;
struct Selection {
    uint32_t layer = 0, position = 0, segment = 0;
    bool matches(uint32_t l, uint32_t p, uint32_t s) const {
        return layer == l && position == p && segment == s;
    }
};
struct Extents {
    uint64_t query = 0, kv = 0, partial = 0, output = 0, total = 0;
};
inline bool extents(uint64_t q, uint64_t kv, uint64_t partial, uint64_t output, Extents& e) {
    if (!q || !kv || !output || q > max_bytes || kv > max_bytes || partial > max_bytes ||
        output > max_bytes || q > max_bytes - kv || q + kv > max_bytes - partial ||
        q + kv + partial > max_bytes - output) return false;
    e = {q, kv, partial, output, q + kv + partial + output};
    return true;
}
inline bool write(std::ostream& out, const void* header, size_t header_bytes,
                  const void* payload, size_t payload_bytes) {
    if (!header || !header_bytes || !payload || !payload_bytes || payload_bytes > max_bytes)
        return false;
    rkva_snapshot::Sha256 hash;
    hash.update(header, header_bytes);
    hash.update(payload, payload_bytes);
    const auto digest = hash.finish();
    out.write(static_cast<const char*>(header), std::streamsize(header_bytes));
    out.write(static_cast<const char*>(payload), std::streamsize(payload_bytes));
    out.write(reinterpret_cast<const char*>(digest.data()), digest.size());
    return bool(out);
}
}
