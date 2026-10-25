#pragma once
#include "remote-attn-protocol.h"
#include <algorithm>
#include <cstdint>

// Native Bee KVarN4 descriptor helpers. Revision 1/2 keep the resident-image
// semantics (every query reads the image of active_count; 128-row ORIGINAL-F16
// tail ring). Revision 3 (serial query prefix, negotiated by root with
// RKVA_CAP_KVARN_SERIAL_PREFIX) makes every query read the image of its own
// prefix q_count = position - origin + 1 and doubles the physical tail ring to
// 256 rows; the wire struct, the upload payload and the rewind payload are
// unchanged.

namespace rkva_kvarn_range {
inline bool valid_descriptor(const rkva_kvarn_range_config& c, uint32_t dimension) {
    return c.storage == RKVA_RANGE_STORAGE_KVARN4 &&
           (c.layout_version == 1 || c.layout_version == 2 || c.layout_version == 3) &&
           c.origin % 128 == 0 && c.capacity > 0 && c.capacity <= uint32_t(INT32_MAX) &&
           c.origin <= uint32_t(INT32_MAX) - c.capacity && dimension == 256 &&
           c.token_group == 128 && c.record_dim == 128 && c.head_dim == 256 && c.head_slices == 2 &&
           c.key_bits == 4 && c.value_bits == 4 && c.record_bytes == 8960 && c.stage_groups == 3 &&
           c.tail_groups == 2 && c.sink_tokens == 128 &&
           (c.precision_tail_tokens == 0 || c.precision_tail_tokens == 128) && c.wht_version == 1 &&
           c.domain == 1 && c.record_policy == 1;
}
inline uint32_t required_record_groups(uint32_t live) {
    return live / 128; // eager: every completed overflow group, including relative group 0
}

// layout_version 4: native Q4_0 overflow rows. Same op16/op22 envelope as the
// native KVarN4 path, but the payload planes are plain Q4_0 rows (K then V,
// head-major, 18 bytes per 32-value block) and there is no record/stage/tail
// or sinkhorn machinery. Storage is the standard Q4_0 range buffer.
inline bool valid_q4_descriptor(const rkva_kvarn_range_config& c, uint32_t dimension) {
    return c.storage == RKVA_RANGE_STORAGE_Q4_0 && c.layout_version == 4 &&
           c.origin % 256 == 0 && c.capacity > 0 && c.capacity <= uint32_t(INT32_MAX) &&
           c.origin <= uint32_t(INT32_MAX) - c.capacity && dimension == 256 &&
           c.token_group == 1 && c.record_dim == 0 && c.head_dim == 256 && c.head_slices == 2 &&
           c.key_bits == 4 && c.value_bits == 4 && c.record_bytes == 144 &&
           c.stage_groups == 0 && c.tail_groups == 0 && c.sink_tokens == 0 &&
           c.precision_tail_tokens == 0 && c.wht_version == 0 && c.domain == 0 &&
           c.record_policy == 1;
}

inline uint64_t q4_row_bytes(uint32_t head_dim, uint32_t heads) {
    return uint64_t(heads) * uint64_t(head_dim / 32) * 18ull;
}

inline bool valid_q4_upload(const rkva_kvarn_range_upload& u, uint32_t capacity,
                            uint32_t previous_live) {
    if (!u.live_count || u.live_count > capacity || (u.flags & ~RKVA_KVARN_UPLOAD_RESET) ||
        u.record_begin || u.record_groups || u.reserved[0] || u.reserved[1])
        return false;
    if (u.flags & RKVA_KVARN_UPLOAD_RESET) {
        return u.append_begin == 0 && u.append_rows == u.live_count;
    }
    return u.append_begin == previous_live && u.append_rows == u.live_count - previous_live;
}

// layout_version 5: native Q8_0 overflow rows. Identical op16/op22 envelope and
// row ordering to layout 4, but every head is 8 q8_0 blocks (34 bytes) and the
// storage is the standard Q8_0 range buffer.
inline bool valid_q8_descriptor(const rkva_kvarn_range_config& c, uint32_t dimension) {
    return c.storage == RKVA_RANGE_STORAGE_Q8_0 && c.layout_version == 5 &&
           c.origin % 256 == 0 && c.capacity > 0 && c.capacity <= uint32_t(INT32_MAX) &&
           c.origin <= uint32_t(INT32_MAX) - c.capacity && dimension == 256 &&
           c.token_group == 1 && c.record_dim == 0 && c.head_dim == 256 && c.head_slices == 2 &&
           c.key_bits == 8 && c.value_bits == 8 && c.record_bytes == 272 &&
           c.stage_groups == 0 && c.tail_groups == 0 && c.sink_tokens == 0 &&
           c.precision_tail_tokens == 0 && c.wht_version == 0 && c.domain == 0 &&
           c.record_policy == 1;
}

inline uint64_t q8_row_bytes(uint32_t head_dim, uint32_t heads) {
    return uint64_t(heads) * uint64_t(head_dim / 32) * 34ull;
}

inline bool valid_q8_upload(const rkva_kvarn_range_upload& u, uint32_t capacity,
                            uint32_t previous_live) {
    if (!u.live_count || u.live_count > capacity || (u.flags & ~RKVA_KVARN_UPLOAD_RESET) ||
        u.record_begin || u.record_groups || u.reserved[0] || u.reserved[1])
        return false;
    if (u.flags & RKVA_KVARN_UPLOAD_RESET) {
        return u.append_begin == 0 && u.append_rows == u.live_count;
    }
    return u.append_begin == previous_live && u.append_rows == u.live_count - previous_live;
}
// OMIT_STAGE is safe only when the exact tail covers every incomplete group
// and global group 0 (whose sink uses stage) is outside this overflow range.
inline bool compact_stage_supported(const rkva_kvarn_range_config& c) {
    return c.layout_version == 3 && c.origin >= 128 && c.precision_tail_tokens == 128;
}
inline uint32_t upload_stage_rows(const rkva_kvarn_range_upload& u) {
    return (u.flags & RKVA_KVARN_UPLOAD_OMIT_STAGE)
               ? 0u
               : ((u.flags & RKVA_KVARN_UPLOAD_RESET) ? 384u : u.append_rows);
}
inline bool valid_upload(const rkva_kvarn_range_upload& u, uint32_t capacity,
                         uint32_t previous_live, uint32_t previous_groups, bool ready,
                         bool allow_compact_stage = false) {
    if (!u.live_count || u.live_count > capacity ||
        (u.flags & ~(RKVA_KVARN_UPLOAD_RESET | RKVA_KVARN_UPLOAD_OMIT_STAGE)) ||
        ((u.flags & RKVA_KVARN_UPLOAD_OMIT_STAGE) && !allow_compact_stage) || u.reserved[0] ||
        u.reserved[1])
        return false;
    const uint32_t total_groups = (capacity + 127) / 128;
    if (u.record_begin > total_groups || u.record_groups > total_groups - u.record_begin)
        return false;
    const bool reset = (u.flags & 1) != 0;
    if (reset ? u.record_begin != 0
              : (!ready || u.live_count < previous_live || u.record_begin > previous_groups))
        return false;
    if (reset ? (u.append_begin || u.append_rows)
              : (u.append_begin != previous_live || u.append_rows != u.live_count - previous_live))
        return false;
    if (u.record_begin + u.record_groups > required_record_groups(u.live_count))
        return false;
    const uint32_t covered =
        reset ? u.record_groups : std::max(previous_groups, u.record_begin + u.record_groups);
    return covered >= required_record_groups(u.live_count);
}
template <typename Copy> void stage_chunks(uint32_t begin, uint32_t rows, Copy copy) {
    for (uint32_t row = 0; row < rows;) {
        const uint32_t token = begin + row;
        const uint32_t group = token / 128;
        const uint32_t slot = group == 0 ? 0 : 1 + ((group - 1) % 2);
        const uint32_t chunk = std::min(rows - row, 128 - token % 128);
        copy(row, slot * 128 + token % 128, chunk);
        row += chunk;
    }
}
template <typename Copy> void tail_chunks(uint32_t begin, uint32_t rows, Copy copy) {
    for (uint32_t row = rows > 128 ? rows - 128 : 0; row < rows;) {
        const uint32_t token = begin + row;
        const uint32_t chunk = std::min(rows - row, 128 - token % 128);
        copy(row, token % 128, chunk);
        row += chunk;
    }
}

// ---- Revision 3: serial query prefix ----
inline bool serial_prefix(const rkva_kvarn_range_config& c) {
    return c.layout_version == 3;
}
constexpr uint32_t kPrefixTailRing = 256;
constexpr uint32_t kPrefixMaxQueries = 128;
// Physical ORIGINAL-F16 tail rows per kind; the logical window stays
// precision_tail_tokens (0 or 128).
inline uint32_t physical_tail_rows(const rkva_kvarn_range_config& c) {
    return serial_prefix(c) && c.precision_tail_tokens ? kPrefixTailRing : c.precision_tail_tokens;
}
// Newest min(rows, ring) rows of [begin, begin + rows) at ring slot token % ring.
template <typename Copy>
void tail_ring_chunks(uint32_t begin, uint32_t rows, uint32_t ring, Copy copy) {
    for (uint32_t row = rows > ring ? rows - ring : 0; row < rows;) {
        const uint32_t token = begin + row;
        const uint32_t chunk = std::min(rows - row, ring - token % ring);
        copy(row, token % ring, chunk);
        row += chunk;
    }
}

// Per-layer revision-3 coverage: rows [tail_begin, live) are held by the
// ORIGINAL-F16 tail ring and every stage row of a group starting at or after
// stage_begin is held by its stage slot. Published only after the GPU copy that
// establishes it has been fenced.
struct prefix_coverage {
    uint32_t tail_begin = 0;
    uint32_t stage_begin = 0;
};
// RESET carries min(live, tail_tokens) tail rows and the full 384-row stage
// image (sink slot + both ping-pong groups, the stage_groups=3/tail_groups=2
// contract the revision-2 rewind already relies on).
inline prefix_coverage coverage_after_reset(uint32_t live, uint32_t tail_tokens) {
    const uint32_t live_group = live ? (live - 1) / 128 : 0;
    return {live - std::min(live, tail_tokens), live_group ? (live_group - 1) * 128 : 0};
}
// Append [previous, live): the ring keeps the newest 256 rows; a new stage group
// overwrites the slot of the group two before it.
inline prefix_coverage coverage_after_append(prefix_coverage c, uint32_t live,
                                             uint32_t tail_tokens) {
    const uint32_t live_group = live ? (live - 1) / 128 : 0;
    if (tail_tokens)
        c.tail_begin = std::max(c.tail_begin, live > kPrefixTailRing ? live - kPrefixTailRing : 0);
    c.stage_begin = std::max(c.stage_begin, live_group ? (live_group - 1) * 128 : 0);
    return c;
}
// Rewind to keep rows; the original-F16 repair span [repair_begin,
// repair_begin + repair_rows) (rkva_kvarn_rewind_tail) extends tail coverage
// downwards only when it is contiguous with the surviving rows. Rewind never
// rewrites stage slots, so stage coverage is unchanged (a <=128-row rollback
// keeps the live group's slot intact).
inline prefix_coverage coverage_after_rewind(prefix_coverage c, uint32_t keep,
                                             uint32_t repair_begin, uint32_t repair_rows) {
    c.tail_begin = std::min(c.tail_begin, keep);
    if (repair_rows && repair_begin < c.tail_begin && repair_begin + repair_rows >= c.tail_begin)
        c.tail_begin = repair_begin;
    return c;
}
// Revision-3 query batch: 1..128 contiguous positions ending at origin + live -
// 1 (leading positions below origin give empty results), every non-empty
// prefix covered by the stage slots (tail 0) or by the ORIGINAL-F16 ring
// (tail 128). False rejects the call before any mutation.
inline bool valid_prefix_queries(const int32_t* positions, uint32_t n, uint32_t origin,
                                 uint32_t live, prefix_coverage c, uint32_t tail_tokens) {
    if (!positions || !n || n > kPrefixMaxQueries || !live)
        return false;
    for (uint32_t i = 1; i < n; ++i) {
        if (int64_t(positions[i]) != int64_t(positions[i - 1]) + 1)
            return false;
    }
    if (int64_t(positions[n - 1]) != int64_t(origin) + live - 1)
        return false;
    for (uint32_t i = 0; i < n; ++i) {
        const int64_t q_count = int64_t(live) - (n - 1 - i);
        if (q_count <= 0)
            continue;
        const uint32_t count = uint32_t(q_count);
        if (tail_tokens) {
            // Rows [count - tail_tokens, live) for the first and therefore every query.
            return c.tail_begin <= (count > tail_tokens ? count - tail_tokens : 0);
        }
        if (count % 128 && c.stage_begin > count / 128 * 128)
            return false;
    }
    return true;
}

// 256-point normalized Walsh-Hadamard transform (normalized H128 then cross-slice mix).
// Bit-exact with CSWht (rkva_wht.hlsl) and wht256x6 (rkva_range_attn_kvarn4_mq.hlsl).
inline void butterfly_wht256(const float* src, float* dst) {
    float t[256];
    for (int i = 0; i < 256; ++i)
        t[i] = src[i];
    for (int slice = 0; slice < 2; ++slice) {
        float* s = t + slice * 128;
        for (int stride = 1; stride < 128; stride <<= 1) {
            for (int i = 0; i < 128; i += 2 * stride) {
                for (int j = 0; j < stride; ++j) {
                    const float u = s[i + j];
                    const float v = s[i + j + stride];
                    s[i + j] = u + v;
                    s[i + j + stride] = u - v;
                }
            }
        }
        for (int lane = 0; lane < 128; ++lane) {
            s[lane] *= 0.08838834764831845f;
        }
    }
    for (int lane = 0; lane < 128; ++lane) {
        const float a = t[lane];
        const float b = t[lane + 128];
        dst[lane] = (a + b) * 0.7071067811865475f;
        dst[lane + 128] = (a - b) * 0.7071067811865475f;
    }
}
} // namespace rkva_kvarn_range
