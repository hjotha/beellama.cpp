#pragma once

#include "rkva-kvarn-range.h"
#include <cmath>
#include <cstring>
#include <vector>

// Pure, bounded admission. No live metadata or GPU resource is changed until
// every segment has passed descriptor, coverage, size and finite-query checks.
namespace rkva_kvarn_bulk {
struct Segment {
    rkva_kvarn_range_upload upload{};
    uint32_t n_queries = 0;
    size_t data_offset = 0, data_bytes = 0, positions_offset = 0, query_offset = 0;
};

inline bool f16_bits_finite(uint16_t bits) {
    return (bits & 0x7c00u) != 0x7c00u;
}

inline uint16_t f32_to_f16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return uint16_t(sign | (mantissa ? 0x7e00u : 0x7c00u));
    }
    int32_t half_exp = int32_t(exponent) - 127 + 15;
    if (half_exp <= 0) {
        if (half_exp < -10) {
            return uint16_t(sign);
        }
        mantissa |= 0x800000u;
        const uint32_t shift = uint32_t(14 - half_exp);
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) {
            ++rounded;
        }
        return uint16_t(sign | rounded);
    }
    if (half_exp >= 31) {
        return uint16_t(sign | 0x7c00u);
    }
    uint32_t rounded = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        ++rounded;
    }
    if (rounded == 0x400u) {
        rounded = 0;
        ++half_exp;
    }
    if (half_exp >= 31) {
        return uint16_t(sign | 0x7c00u);
    }
    return uint16_t(sign | (uint32_t(half_exp) << 10) | rounded);
}

inline float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t m = mant;
            int32_t e = -14;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03ffu;
            bits = sign | (uint32_t(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline uint8_t f16_to_fp8_bits(uint16_t h) {
    if ((h & 0x7c00u) == 0x7c00u) {
        const uint8_t sign_byte = (h & 0x8000u) ? 0x80u : 0x00u;
        if (h & 0x03ffu) {
            uint8_t m = uint8_t((h >> 8) & 3u);
            return sign_byte | 0x7cu | (m ? m : 1u);
        }
        return sign_byte | 0x7cu;
    }
    const uint16_t mag = h & 0x7fffu;
    if (mag > 0x7b00u) {
        return (h & 0x8000u) ? 0xfbu : 0x7bu;
    }
    const uint8_t sign_byte = (h & 0x8000u) ? 0x80u : 0x00u;
    const uint8_t rounded = uint8_t((mag + 0x007fu + ((mag >> 8) & 1u)) >> 8);
    return sign_byte | rounded;
}

inline uint16_t fp8_to_f16_bits(uint8_t b) {
    return uint16_t(b) << 8;
}

// Option B: SplitZip FP16 lossless exponent-split codec
static const uint8_t kSplitZipTopExponents[15] = {14, 15, 13, 12, 16, 11, 10, 17,
                                                  9,  18, 8,  19, 7,  20, 6};

struct rkva_splitzip_header {
    uint32_t n_elements;
    uint32_t n_escapes;
    uint32_t compressed_bytes;
};

inline size_t splitzip_encode(const uint16_t* src, size_t n, std::vector<uint8_t>& out) {
    uint8_t exp_to_code[32];
    std::memset(exp_to_code, 0x0F, sizeof(exp_to_code));
    for (uint8_t code = 0; code < 15; ++code) {
        exp_to_code[kSplitZipTopExponents[code]] = code;
    }

    std::vector<uint8_t> escapes;
    out.resize(sizeof(rkva_splitzip_header) + n * 4 + 16);
    auto* hdr = reinterpret_cast<rkva_splitzip_header*>(out.data());
    hdr->n_elements = (uint32_t)n;
    hdr->n_escapes = 0;

    uint8_t* p = out.data() + sizeof(rkva_splitzip_header);
    for (size_t i = 0; i < n; ++i) {
        p[i] = (uint8_t)(src[i] & 0xFF);
    }
    p += n;
    for (size_t i = 0; i < n; i += 2) {
        uint8_t e0 = (src[i] >> 10) & 0x1F;
        uint8_t c0 = exp_to_code[e0];
        uint8_t e1 = (i + 1 < n) ? ((src[i + 1] >> 10) & 0x1F) : 0;
        uint8_t c1 = (i + 1 < n) ? exp_to_code[e1] : 0;
        *p++ = (c0 << 4) | (c1 & 0x0F);
        if (c0 == 0x0F)
            escapes.push_back((uint8_t)((src[i] >> 8) & 0xFF));
        if (c1 == 0x0F && i + 1 < n)
            escapes.push_back((uint8_t)((src[i + 1] >> 8) & 0xFF));
    }
    for (size_t i = 0; i < n; i += 2) {
        uint8_t s0 = ((src[i] >> 15) & 1) << 2 | ((src[i] >> 8) & 3);
        uint8_t s1 = (i + 1 < n) ? (((src[i + 1] >> 15) & 1) << 2 | ((src[i + 1] >> 8) & 3)) : 0;
        *p++ = (s0 << 4) | (s1 & 0x0F);
    }
    if (!escapes.empty()) {
        std::memcpy(p, escapes.data(), escapes.size());
        p += escapes.size();
    }
    hdr->n_escapes = (uint32_t)escapes.size();
    hdr->compressed_bytes = (uint32_t)(p - out.data());
    out.resize(hdr->compressed_bytes);
    return out.size();
}

inline bool splitzip_decode(const uint8_t* in, size_t in_bytes, uint16_t* dst, size_t n) {
    if (!in || (!dst && n > 0) || in_bytes < sizeof(rkva_splitzip_header))
        return false;
    rkva_splitzip_header header{};
    std::memcpy(&header, in, sizeof(header));
    const auto* hdr = &header;
    if (hdr->n_elements != n || hdr->n_escapes > n)
        return false;
    const uint64_t count = hdr->n_elements;
    const uint64_t nibble_bytes = (count + 1) / 2;
    const uint64_t required_bytes = sizeof(header) + count + 2 * nibble_bytes + hdr->n_escapes;
    if (hdr->compressed_bytes != required_bytes || in_bytes < required_bytes)
        return false;

    const uint8_t* p = in + sizeof(rkva_splitzip_header);
    const uint8_t* low_bytes = p;
    p += n;
    const uint8_t* exp_nibbles = p;
    p += (n + 1) / 2;
    const uint8_t* sm_nibbles = p;
    p += (n + 1) / 2;
    const uint8_t* esc_ptr = p;

    size_t esc_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        uint8_t lo = low_bytes[i];
        uint8_t exp_pair = exp_nibbles[i / 2];
        uint8_t code = (i % 2 == 0) ? (exp_pair >> 4) : (exp_pair & 0x0F);

        uint8_t sm_pair = sm_nibbles[i / 2];
        uint8_t sm3 = (i % 2 == 0) ? (sm_pair >> 4) : (sm_pair & 0x0F);
        uint16_t sign = (sm3 >> 2) & 1;
        uint16_t top_m = sm3 & 3;

        uint16_t exp;
        if (code == 0x0F) {
            if (esc_idx >= hdr->n_escapes)
                return false;
            uint8_t raw_hi = esc_ptr[esc_idx++];
            exp = (raw_hi >> 2) & 0x1F;
        } else {
            exp = kSplitZipTopExponents[code];
        }

        uint16_t h = (sign << 15) | (exp << 10) | (top_m << 8) | lo;
        dst[i] = h;
    }
    if (esc_idx != hdr->n_escapes)
        return false;
    return true;
}

inline bool peek_wire(const uint8_t* data, size_t bytes, uint32_t* out_wire) {
    if (!data || !out_wire ||
        bytes < sizeof(rkva_kvarn_range_bulk) + sizeof(rkva_kvarn_range_bulk_segment) +
                    sizeof(rkva_kvarn_range_upload) + sizeof(rkva_range_attn)) {
        return false;
    }
    rkva_kvarn_range_bulk header{};
    std::memcpy(&header, data, sizeof(header));
    if (header.version != 1 || !header.segment_count || header.segment_count > 32 ||
        header.reserved) {
        return false;
    }
    rkva_kvarn_range_bulk_segment seg{};
    std::memcpy(&seg, data + sizeof(header), sizeof(seg));
    const size_t upload_offset = sizeof(header) + sizeof(seg);
    if (seg.upload_bytes < sizeof(rkva_kvarn_range_upload) ||
        seg.upload_bytes > bytes - upload_offset ||
        bytes - upload_offset - seg.upload_bytes < sizeof(rkva_range_attn)) {
        return false;
    }
    rkva_range_attn attention{};
    std::memcpy(&attention, data + upload_offset + seg.upload_bytes, sizeof(attention));
    if (attention.wire != RKVA_WIRE_F32 && attention.wire != RKVA_WIRE_F16 &&
        attention.wire != RKVA_WIRE_FP8) {
        return false;
    }
    *out_wire = attention.wire;
    return true;
}

inline bool validate_wire(const uint8_t* data, size_t bytes, uint32_t n_queries,
                          const rkva_kvarn_range_config& descriptor, uint32_t n_head,
                          uint32_t n_head_kv, uint32_t live, uint32_t groups, bool ready,
                          rkva_kvarn_range::prefix_coverage coverage, uint64_t generation,
                          uint32_t expected_wire, std::vector<Segment>& segments) {
    segments.clear();
    if (!data || bytes < sizeof(rkva_kvarn_range_bulk) || bytes > (256u << 20) ||
        (expected_wire != RKVA_WIRE_F32 && expected_wire != RKVA_WIRE_F16 &&
         expected_wire != RKVA_WIRE_FP8) ||
        !rkva_kvarn_range::valid_descriptor(descriptor, 256) ||
        !rkva_kvarn_range::serial_prefix(descriptor) || !n_head || !n_head_kv || n_head > 256 ||
        n_head_kv > 256 || !n_queries || n_queries > 256)
        return false;
    rkva_kvarn_range_bulk header{};
    std::memcpy(&header, data, sizeof(header));
    if (header.version != 1 || !header.segment_count || header.segment_count > 32 ||
        header.total_queries != n_queries || header.reserved ||
        generation > UINT64_MAX - header.segment_count)
        return false;
    const uint64_t query_elem_bytes =
        (expected_wire == RKVA_WIRE_FP8)                                          ? sizeof(uint8_t)
        : (expected_wire == RKVA_WIRE_F16 || expected_wire == RKVA_WIRE_SPLITZIP) ? sizeof(uint16_t)
                                                                                  : sizeof(float);
    size_t offset = sizeof(header);
    uint32_t total = 0;
    for (uint32_t i = 0; i < header.segment_count; ++i) {
        if (bytes - offset < sizeof(rkva_kvarn_range_bulk_segment))
            return false;
        rkva_kvarn_range_bulk_segment wire{};
        std::memcpy(&wire, data + offset, sizeof(wire));
        offset += sizeof(wire);
        if (!wire.n_queries || wire.n_queries > 9 || wire.n_queries > n_queries - total ||
            wire.upload_bytes < sizeof(rkva_kvarn_range_upload) ||
            wire.upload_bytes > bytes - offset)
            return false;
        Segment part;
        part.n_queries = wire.n_queries;
        std::memcpy(&part.upload, data + offset, sizeof(part.upload));
        const auto& u = part.upload;
        if (!rkva_kvarn_range::valid_upload(u, descriptor.capacity, live, groups, ready,
                                            rkva_kvarn_range::compact_stage_supported(descriptor)))
            return false;
        const bool reset = (u.flags & 1) != 0;
        if ((reset && i) || (!reset && u.append_rows != wire.n_queries))
            return false;
        const uint64_t row_bytes = uint64_t(n_head_kv) * 256 * 2;
        const uint64_t records = uint64_t(u.record_groups) * n_head_kv * 2 * 8960;
        const uint64_t stage = uint64_t(rkva_kvarn_range::upload_stage_rows(u)) * row_bytes;
        const uint64_t tail_rows =
            descriptor.precision_tail_tokens
                ? (reset ? std::min(u.live_count, descriptor.precision_tail_tokens) : u.append_rows)
                : 0;
        const uint64_t expected = 2 * (records + stage + tail_rows * row_bytes);
        if (uint64_t(wire.upload_bytes) != sizeof(u) + expected)
            return false;
        part.data_offset = offset + sizeof(u);
        part.data_bytes = size_t(expected);
        offset += wire.upload_bytes;
        const uint64_t query_bytes = uint64_t(wire.n_queries) * n_head * 256 * query_elem_bytes;
        const uint64_t attention_bytes =
            sizeof(rkva_range_attn) + uint64_t(wire.n_queries) * 4 + query_bytes;
        if (attention_bytes > bytes - offset)
            return false;
        rkva_range_attn attention{};
        std::memcpy(&attention, data + offset, sizeof(attention));
        if (attention.wire != expected_wire || attention.flags || attention.reserved[0] ||
            attention.reserved[1])
            return false;
        part.positions_offset = offset + sizeof(attention);
        part.query_offset = part.positions_offset + wire.n_queries * 4;
        int32_t positions[9];
        std::memcpy(positions, data + part.positions_offset, wire.n_queries * 4);
        for (uint32_t j = 0; j < wire.n_queries; ++j)
            if (positions[j] < 0)
                return false;
        coverage = reset ? rkva_kvarn_range::coverage_after_reset(u.live_count,
                                                                  descriptor.precision_tail_tokens)
                         : rkva_kvarn_range::coverage_after_append(
                               coverage, u.live_count, descriptor.precision_tail_tokens);
        if (!rkva_kvarn_range::valid_prefix_queries(positions, wire.n_queries, descriptor.origin,
                                                    u.live_count, coverage,
                                                    descriptor.precision_tail_tokens))
            return false;
        if (expected_wire == RKVA_WIRE_FP8) {
            for (size_t j = 0; j < query_bytes; ++j) {
                uint8_t value;
                std::memcpy(&value, data + part.query_offset + j, sizeof(value));
                if ((value & 0x7cu) == 0x7cu)
                    return false;
            }
        } else if (expected_wire == RKVA_WIRE_F16) {
            for (size_t j = 0; j < query_bytes; j += sizeof(uint16_t)) {
                uint16_t value;
                std::memcpy(&value, data + part.query_offset + j, sizeof(value));
                if (!f16_bits_finite(value))
                    return false;
            }
        } else if (expected_wire == RKVA_WIRE_SPLITZIP) {
            if (query_bytes < sizeof(rkva_splitzip_header))
                return false;
        } else {
            for (size_t j = 0; j < query_bytes; j += sizeof(float)) {
                float value;
                std::memcpy(&value, data + part.query_offset + j, sizeof(value));
                if (!std::isfinite(value))
                    return false;
            }
        }
        offset += size_t(attention_bytes);
        groups = reset ? u.record_groups : std::max(groups, u.record_begin + u.record_groups);
        live = u.live_count;
        ready = true;
        total += wire.n_queries;
        segments.push_back(part);
    }
    return offset == bytes && total == n_queries;
}

inline bool validate(const uint8_t* data, size_t bytes, uint32_t n_queries,
                     const rkva_kvarn_range_config& descriptor, uint32_t n_head, uint32_t n_head_kv,
                     uint32_t live, uint32_t groups, bool ready,
                     rkva_kvarn_range::prefix_coverage coverage, uint64_t generation,
                     std::vector<Segment>& segments) {
    return validate_wire(data, bytes, n_queries, descriptor, n_head, n_head_kv, live, groups, ready,
                         coverage, generation, RKVA_WIRE_F32, segments);
}
} // namespace rkva_kvarn_bulk
