#pragma once

#include "rkva-kvarn-range.h"
#include "../../../../ggml/src/ggml-remote-attn/rkva-fp8-e4m3.h"
#include "../../../../ggml/include/ggml-remote-attn-policy.h"
#include <cmath>
#include <cstring>
#include <vector>

// Pure, bounded admission. No live metadata or GPU resource is changed until
// every segment has passed descriptor, coverage, size and finite-query checks.
namespace rkva_kvarn_bulk {
// Native-row CPU E4 admission/packing. Shared with host tests; unlike the
// KVarN kernel helpers, this accepts the complete bounded 256-query envelope.
inline bool native_e4_queries_valid(const uint8_t* positions, const uint8_t* query,
                                    uint32_t nq, uint32_t heads, uint32_t dim) {
    if (!positions || !query || !nq || nq > 256 || !heads || heads > 256 || dim != 256) return false;
    for (uint32_t q = 0; q < nq; ++q) {
        int32_t pos;
        std::memcpy(&pos, positions + size_t(q) * sizeof(pos), sizeof(pos));
        if (pos < 0) return false;
    }
    for (size_t e = 0; e < size_t(nq) * heads * dim; ++e)
        if ((query[e] & 0x7fu) == 0x7fu) return false;
    return true;
}
inline bool native_e4_results_valid(const float* out, const float* lse, const uint8_t* positions,
                                    uint32_t nq, uint32_t heads, uint32_t dim, uint32_t origin) {
    if (!out || !lse || !positions || !nq || nq > 256 || !heads || heads > 256 || dim != 256) return false;
    const size_t width = size_t(heads) * dim;
    for (uint32_t q = 0; q < nq; ++q) {
        int32_t position;
        std::memcpy(&position, positions + size_t(q) * sizeof(position), sizeof(position));
        if (position < 0) return false;
        const bool empty = ggml_remote_attn_policy::e4_empty_prefix(position, origin);
        for (size_t e = 0; e < width; ++e)
            if (!std::isfinite(out[size_t(q) * width + e]) ||
                (empty && out[size_t(q) * width + e] != 0.0f)) return false;
        for (uint32_t h = 0; h < heads; ++h)
            if (!ggml_remote_attn_policy::e4_lse_valid(lse[size_t(q) * heads + h], position, origin)) return false;
    }
    return true;
}
inline bool native_e4_pack_reply(const float* out, const float* lse, const int32_t* positions,
                                 uint32_t nq, uint32_t heads, uint32_t dim, uint32_t origin,
                                 std::vector<uint8_t>& reply) {
    // Validate before publishing any bytes, including empty-prefix semantics.
    if (!native_e4_results_valid(out, lse, reinterpret_cast<const uint8_t*>(positions), nq, heads, dim, origin)) return false;
    const size_t elements = size_t(nq) * heads * dim;
    reply.resize(elements + size_t(nq) * heads * sizeof(float));
    for (size_t e = 0; e < elements; ++e) {
        // Direct F32 RNE: no intermediate F16 double rounding or overflow.
        if (!rkva_e4::encode_f32(out[e], reply[e])) return false;
    }
    std::memcpy(reply.data() + elements, lse, size_t(nq) * heads * sizeof(float));
    return true;
}
// Status 2 may only be provisionally accepted when a per-query prefix
// validation follows, since the status also covers NaN/+inf LSE.
inline bool fp8_gpu_status_valid(uint32_t status, bool validate_prefix) {
    return status == 0 || (validate_prefix && status == 2);
}
// The FP8 pack shader reports every non-finite LSE. Negative infinity is
// nevertheless the defined result for a query with no visible remote rows.
// Validate against the original positions before accepting that status.
inline bool q8_fp8_lse_valid(const uint8_t* lse, const uint8_t* positions,
                              uint32_t n_queries, uint32_t n_heads,
                              uint32_t origin, uint32_t live_rows) {
    for (uint32_t q = 0; q < n_queries; ++q) {
        int32_t pos;
        std::memcpy(&pos, positions + size_t(q) * 4, 4);
        const bool empty = pos >= 0 && (uint32_t(pos) < origin || live_rows == 0);
        for (uint32_t h = 0; h < n_heads; ++h) {
            uint32_t bits;
            std::memcpy(&bits, lse + (size_t(q) * n_heads + h) * 4, 4);
            if ((bits & 0x7f800000u) == 0x7f800000u && !(empty && bits == 0xff800000u))
                return false;
        }
    }
    return true;
}

// Empty serial prefixes must return an exact zero OUT and -infinity LSE.
inline bool compact_f16_prefix_results_valid(const uint16_t* out, const float* lse,
        const uint8_t* positions, uint32_t n_queries, uint32_t n_heads, uint32_t dimension,
        uint32_t origin) {
    if (!out || !lse || !positions || !n_queries || n_queries > 9 ||
            !n_heads || n_heads > 256 || dimension != 256) return false;
    const uint64_t width = uint64_t(n_heads) * dimension;
    for (uint32_t q = 0; q < n_queries; ++q) {
        int32_t position;
        std::memcpy(&position, positions + size_t(q) * sizeof(position), sizeof(position));
        for (uint32_t h = 0; h < n_heads; ++h) {
            if (!ggml_remote_attn_policy::e4_lse_valid(lse[uint64_t(q) * n_heads + h], position, origin))
                return false;
        }
        if (ggml_remote_attn_policy::e4_empty_prefix(position, origin)) {
            for (uint64_t d = 0; d < width; ++d) {
                if ((out[uint64_t(q) * width + d] & 0x7fffu) != 0) return false;
            }
        }
    }
    return true;
}

// The pack shader validates finite OUT; empty prefixes additionally require
// zero packed OUT and negative-infinity LSE.
inline bool compact_fp8_prefix_results_valid(const uint8_t* out, const float* lse,
        const uint8_t* positions, uint32_t n_queries, uint32_t n_heads, uint32_t dimension,
        uint32_t origin) {
    if (!out || !lse || !positions || !n_queries || n_queries > 9 ||
            !n_heads || n_heads > 256 || dimension != 256) return false;
    const uint64_t width = uint64_t(n_heads) * dimension;
    for (uint32_t q = 0; q < n_queries; ++q) {
        int32_t position;
        std::memcpy(&position, positions + size_t(q) * sizeof(position), sizeof(position));
        for (uint32_t h = 0; h < n_heads; ++h)
            if (!ggml_remote_attn_policy::e4_lse_valid(lse[uint64_t(q) * n_heads + h], position, origin))
                return false;
        if (ggml_remote_attn_policy::e4_empty_prefix(position, origin))
            for (uint64_t d = 0; d < width; ++d)
                if ((out[uint64_t(q) * width + d] & 0x7fu) != 0) return false;
    }
    return true;
}
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

// Validate both planes before touching the live GPU buffers. Metadata is
// little-endian FP16 and keys have a mandatory two-byte zero pad.
inline bool turboquant_nc_planes_valid(const uint8_t* k, const uint8_t* v, size_t records) {
    if ((!k || !v) && records) return false;
    for (size_t i = 0; i < records; ++i) {
        const uint8_t* kb = k + i * 132;
        const uint8_t* vb = v + i * 132;
        const uint16_t norm_bits = uint16_t(kb[128]) | uint16_t(kb[129] << 8);
        const uint16_t scale_bits = uint16_t(vb[128]) | uint16_t(vb[129] << 8);
        const uint16_t min_bits = uint16_t(vb[130]) | uint16_t(vb[131] << 8);
        if (kb[130] || kb[131] || !f16_bits_finite(norm_bits) ||
            !f16_bits_finite(scale_bits) || !f16_bits_finite(min_bits)) return false;
        const float norm = f16_bits_to_f32(norm_bits);
        const float scale = f16_bits_to_f32(scale_bits);
        const float minimum = f16_bits_to_f32(min_bits);
        if (norm < 0.0f || scale < 0.0f ||
            !f16_bits_finite(f32_to_f16_bits(15.0f * scale + minimum))) return false;
    }
    return true;
}

// The CPU E4 native-row path admits complete envelopes before any upload.
// Current coverage is authoritative in the engine; the handler can validate
// framing/metadata without inventing a per-layer resident count.
inline bool native_e4_planes_valid(uint32_t storage, const uint8_t* k, const uint8_t* v, size_t heads) {
    if (rkva_kvarn_range::low4_storage(storage)) return rkva_kvarn_range::valid_low4_planes(storage, k, v, heads);
    if (storage == RKVA_RANGE_STORAGE_TURBOQUANT_NC) return turboquant_nc_planes_valid(k, v, heads);
    const size_t head_bytes = rkva_kvarn_range::native_storage_head_bytes(storage);
    if (!head_bytes) return false;
    if (!heads) return true;
    if (!k || !v) return false;
    for (const auto* plane : {k, v}) {
        for (size_t h = 0; h < heads; ++h) {
            const uint8_t* row = plane + h * head_bytes;
            if (storage == RKVA_RANGE_STORAGE_SAW_INT4) {
                float scale, zero;
                std::memcpy(&scale, row + 128, sizeof(scale));
                std::memcpy(&zero, row + 132, sizeof(zero));
                if (!std::isfinite(scale) || !std::isfinite(zero) || scale < 0.0f ||
                    !std::isfinite(scale * (0.0f - zero)) || !std::isfinite(scale * (15.0f - zero))) return false;
            } else if (storage == RKVA_RANGE_STORAGE_BF16) {
                for (size_t d = 0; d < 256; ++d) {
                    uint16_t val;
                    std::memcpy(&val, row + d * 2, sizeof(val));
                    if ((val & 0x7f80u) == 0x7f80u) return false;
                }
            } else if (storage == RKVA_RANGE_STORAGE_F32) {
                for (size_t d = 0; d < 256; ++d) {
                    float val;
                    std::memcpy(&val, row + d * 4, sizeof(val));
                    if (!std::isfinite(val)) return false;
                }
            } else {
                const size_t block_bytes = head_bytes / 8;
                const bool has_min = (storage == RKVA_RANGE_STORAGE_OSCAR_INT2 ||
                                      storage == RKVA_RANGE_STORAGE_Q2_1 ||
                                      storage == RKVA_RANGE_STORAGE_Q3_1 ||
                                      storage == RKVA_RANGE_STORAGE_Q5_1 ||
                                      storage == RKVA_RANGE_STORAGE_Q6_1);
                for (size_t block = 0; block < 8; ++block) {
                    uint16_t scale;
                    std::memcpy(&scale, row + block * block_bytes, sizeof(scale));
                    if (!f16_bits_finite(scale)) return false;
                    if (has_min) {
                        uint16_t min_val;
                        std::memcpy(&min_val, row + block * block_bytes + 2, sizeof(min_val));
                        if (!f16_bits_finite(min_val)) return false;
                    }
                }
            }
        }
    }
    return true;
}
inline bool validate_native_e4(const uint8_t* data, size_t bytes, uint32_t nq, uint32_t storage,
                               uint32_t heads, uint32_t kv_heads, uint32_t capacity,
                               uint32_t live, bool check_live) {
    const size_t row = rkva_kvarn_range::native_storage_head_bytes(storage);
    if (!data || bytes < sizeof(rkva_kvarn_range_bulk) || !row || !nq || nq > 256 ||
        !heads || heads > 256 || !kv_heads || kv_heads > 256 || !capacity || live > capacity) return false;
    rkva_kvarn_range_bulk header;
    std::memcpy(&header, data, sizeof(header));
    if (header.version != 1 || header.reserved || !header.segment_count || header.segment_count > 32 ||
        header.total_queries != nq) return false;
    size_t off = sizeof(header);
    uint32_t queries = 0;
    for (uint32_t i = 0; i < header.segment_count; ++i) {
        rkva_kvarn_range_bulk_segment segment;
        if (bytes - off < sizeof(segment)) return false;
        std::memcpy(&segment, data + off, sizeof(segment)); off += sizeof(segment);
        if (segment.upload_bytes < sizeof(rkva_kvarn_range_upload) || segment.upload_bytes > bytes - off ||
            !segment.n_queries || segment.n_queries > nq - queries) return false;
        rkva_kvarn_range_upload upload;
        std::memcpy(&upload, data + off, sizeof(upload));
        const bool valid = rkva_kvarn_range::low4_storage(storage) ? rkva_kvarn_range::valid_low4_upload(upload, capacity, upload.append_begin) :
            storage == RKVA_RANGE_STORAGE_Q8_0 ? rkva_kvarn_range::valid_q8_upload(upload, capacity, upload.append_begin) :
            storage == RKVA_RANGE_STORAGE_SAW_INT4 ? rkva_kvarn_range::valid_saw_int4_upload(upload, capacity, upload.append_begin) :
            storage == RKVA_RANGE_STORAGE_TURBOQUANT_NC ? rkva_kvarn_range::valid_turboquant_nc_upload(upload, capacity, upload.append_begin) :
            storage == RKVA_RANGE_STORAGE_OSCAR_INT2 ? rkva_kvarn_range::valid_oscar_int2_upload(upload, capacity, upload.append_begin) :
            (storage == RKVA_RANGE_STORAGE_Q4_0 || rkva_kvarn_range::is_extended_native_storage(storage))
                ? rkva_kvarn_range::valid_q4_upload(upload, capacity, upload.append_begin) : false;
        const bool reset = (upload.flags & RKVA_KVARN_UPLOAD_RESET) != 0;
        const uint64_t plane = uint64_t(upload.append_rows) * kv_heads * row;
        if (!valid || uint64_t(upload.append_begin) + upload.append_rows != upload.live_count ||
            (reset ? (i != 0 || upload.append_begin != 0) : ((i || check_live) && upload.append_begin != live)) ||
            uint64_t(segment.upload_bytes) != sizeof(upload) + 2 * plane) return false;
        const uint8_t* k = data + off + sizeof(upload);
        if (!native_e4_planes_valid(storage, k, k + size_t(plane), size_t(upload.append_rows) * kv_heads)) return false;
        live = upload.live_count; off += segment.upload_bytes;
        rkva_range_attn attn;
        if (bytes - off < sizeof(attn)) return false;
        std::memcpy(&attn, data + off, sizeof(attn)); off += sizeof(attn);
        const size_t pos_bytes = size_t(segment.n_queries) * sizeof(int32_t);
        const size_t query_bytes = size_t(segment.n_queries) * heads * 256;
        if (attn.wire != RKVA_WIRE_FP8_E4M3FN || attn.flags || attn.reserved[0] || attn.reserved[1] ||
            bytes - off < pos_bytes + query_bytes ||
            !native_e4_queries_valid(data + off, data + off + pos_bytes, segment.n_queries, heads, 256)) return false;
        off += pos_bytes + query_bytes; queries += segment.n_queries;
    }
    return off == bytes && queries == nq;
}

inline bool turboquant_nc_output_valid(const uint8_t* out, const uint8_t* lse,
                                       const uint8_t* positions, uint32_t queries,
                                       uint32_t heads, uint32_t origin, uint32_t live,
                                       bool narrow_output) {
    if (!out || !lse || !positions || !queries || !heads ||
        !q8_fp8_lse_valid(lse, positions, queries, heads, origin, live)) return false;
    const uint64_t elements = uint64_t(queries) * heads * 256;
    for (uint64_t i = 0; i < elements; ++i) {
        float value;
        std::memcpy(&value, out + i * sizeof(float), sizeof(value));
        if (!std::isfinite(value) || (narrow_output &&
            !f16_bits_finite(f32_to_f16_bits(value)))) return false;
    }
    return true;
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
    if (header.version != 1 || !header.segment_count || header.segment_count > 64 ||
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
        attention.wire != RKVA_WIRE_FP8 && attention.wire != RKVA_WIRE_FP8_E4M3FN) {
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
         expected_wire != RKVA_WIRE_FP8 && expected_wire != RKVA_WIRE_FP8_E4M3FN) ||
        !rkva_kvarn_range::valid_descriptor(descriptor, 256) ||
        !rkva_kvarn_range::serial_prefix(descriptor) || !n_head || !n_head_kv || n_head > 256 ||
        n_head_kv > 256 || !n_queries || n_queries > 512)
        return false;
    rkva_kvarn_range_bulk header{};
    std::memcpy(&header, data, sizeof(header));
    if (header.version != 1 || !header.segment_count || header.segment_count > 64 ||
        header.total_queries != n_queries || header.reserved ||
        generation > UINT64_MAX - header.segment_count)
        return false;
    const uint64_t query_elem_bytes =
        (expected_wire == RKVA_WIRE_FP8 || expected_wire == RKVA_WIRE_FP8_E4M3FN) ? sizeof(uint8_t)
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
        const uint64_t records = uint64_t(u.record_groups) * n_head_kv * 2 * descriptor.record_bytes;
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
        if (expected_wire == RKVA_WIRE_FP8_E4M3FN) {
            for (size_t j = 0; j < query_bytes; ++j) {
                if ((data[part.query_offset + j] & 0x7fu) == 0x7fu)
                    return false;
            }
        } else if (expected_wire == RKVA_WIRE_FP8) {
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
