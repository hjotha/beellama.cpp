#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#if !defined(GGML_MASK_PACK_DISABLE_SSE2) && (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#    include <emmintrin.h>
#    define GGML_MASK_PACK_HAS_SSE2 1
#else
#    define GGML_MASK_PACK_HAS_SSE2 0
#endif

// One bit per raw F16 value: +0 (0x0000) -> 0, -inf (0xfc00) -> 1.
// The final word's unused high bits are zero. Source and destination may be
// unaligned; destination writes use memcpy rather than uint32_t dereferences.
// On failure the destination can contain a valid partial prefix: callers must
// not consume/launch it unless the whole call succeeds.
static inline size_t ggml_mask_pack_f16_words(size_t n) {
    return n/32 + (n%32 != 0); // unlike (n + 31)/32, cannot overflow
}

namespace ggml_mask_pack_detail {

static inline bool valid_args(const void * data, size_t n, const uint32_t * dst) {
    return n <= std::numeric_limits<size_t>::max()/sizeof(uint16_t) &&
            (n == 0 || (data != nullptr && dst != nullptr));
}

static inline bool scalar_word(const uint8_t * src, size_t n, uint32_t & word) {
    word = 0;
    for (size_t lane = 0; lane < n; ++lane) {
        uint16_t raw;
        std::memcpy(&raw, src + lane*sizeof(raw), sizeof(raw));
        if (raw == uint16_t(0xfc00)) {
            word |= uint32_t(1) << lane;
        } else if (raw != 0) {
            return false;
        }
    }
    return true;
}

static inline bool scalar(const void * data, size_t n, uint32_t * dst) {
    if (!valid_args(data, n, dst)) return false;
    const auto * src = static_cast<const uint8_t *>(data);
    auto * output = reinterpret_cast<uint8_t *>(dst);
    while (n) {
        const size_t count = n < 32 ? n : 32;
        uint32_t word;
        if (!scalar_word(src, count, word)) return false;
        std::memcpy(output, &word, sizeof(word));
        src += count*sizeof(uint16_t);
        output += sizeof(word);
        n -= count;
    }
    return true;
}

#if GGML_MASK_PACK_HAS_SSE2
static inline bool sse2_word16(const uint8_t * src, uint32_t & bits) {
    const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 16));
    const __m128i zero = _mm_setzero_si128();
    const __m128i negative_inf = _mm_set1_epi16(-1024); // raw 0xfc00
    const __m128i inf_a = _mm_cmpeq_epi16(a, negative_inf);
    const __m128i inf_b = _mm_cmpeq_epi16(b, negative_inf);
    const __m128i valid_a = _mm_or_si128(inf_a, _mm_cmpeq_epi16(a, zero));
    const __m128i valid_b = _mm_or_si128(inf_b, _mm_cmpeq_epi16(b, zero));
    if (_mm_movemask_epi8(_mm_and_si128(valid_a, valid_b)) != 0xffff) return false;
    // Each comparison lane is 0 or -1, so signed saturation keeps its bit.
    bits = static_cast<uint32_t>(_mm_movemask_epi8(_mm_packs_epi16(inf_a, inf_b)));
    return true;
}

static inline bool sse2(const void * data, size_t n, uint32_t * dst) {
    if (!valid_args(data, n, dst)) return false;
    const auto * src = static_cast<const uint8_t *>(data);
    auto * output = reinterpret_cast<uint8_t *>(dst);
    while (n >= 32) {
        uint32_t lo, hi;
        if (!sse2_word16(src, lo) || !sse2_word16(src + 32, hi)) return false;
        const uint32_t word = lo | (hi << 16);
        std::memcpy(output, &word, sizeof(word));
        src += 32*sizeof(uint16_t);
        output += sizeof(word);
        n -= 32;
    }
    if (n) {
        uint32_t word;
        if (!scalar_word(src, n, word)) return false;
        std::memcpy(output, &word, sizeof(word));
    }
    return true;
}
#endif

} // namespace ggml_mask_pack_detail

static inline bool ggml_mask_pack_f16(const void * data, size_t n, uint32_t * dst) {
#if GGML_MASK_PACK_HAS_SSE2
    return ggml_mask_pack_detail::sse2(data, n, dst);
#else
    return ggml_mask_pack_detail::scalar(data, n, dst);
#endif
}
