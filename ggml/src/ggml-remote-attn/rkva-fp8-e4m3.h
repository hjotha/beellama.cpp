#pragma once

#include <cstdint>
#include <cstring>
#include <limits>

// Activation wire codec: E4M3FN, bias 7, round-to-nearest-even, finite
// saturation, subnormals and signed zero. Non-finite inputs are rejected.
namespace rkva_e4 {

static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559,
              "E4M3FN wire requires IEEE-754 binary32");

namespace detail {
inline uint32_t round_shift_even(uint32_t value, unsigned shift) {
    const uint32_t low = value & ((uint32_t(1) << shift) - 1u);
    const uint32_t half = uint32_t(1) << (shift - 1u);
    const uint32_t base = value >> shift;
    return base + (low > half || (low == half && (base & 1u)));
}
} // namespace detail

// Q is encoded directly from F32; invalid input leaves output unchanged.
inline bool encode_f32(float value, uint8_t & output) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t magnitude = bits & 0x7fffffffu;
    const uint8_t sign = uint8_t((bits >> 24u) & 0x80u);
    if (magnitude >= 0x7f800000u) return false;
    if (magnitude >= 0x43e00000u) { output = uint8_t(sign | 0x7eu); return true; }
    if (magnitude <= 0x3a800000u) { output = sign; return true; }
    const uint32_t exponent = magnitude >> 23u;
    const uint32_t mantissa = magnitude & 0x7fffffu;
    // The preceding thresholds bound the shifts to [20,24].
    const uint32_t result = magnitude < 0x3c800000u
        ? detail::round_shift_even(mantissa | 0x800000u, 141u - exponent)
        : ((exponent - 120u) << 3u) + detail::round_shift_even(mantissa, 20u);
    output = uint8_t(sign | result);
    return true;
}

// OUT has already been rounded from F32 to F16; invalid F16 is rejected.
inline bool encode_f16(uint16_t half, uint8_t & output) {
    const uint32_t sign = uint32_t(half & 0x8000u) << 16u;
    const uint32_t exponent = (half >> 10u) & 31u;
    const uint32_t mantissa = half & 1023u;
    if (exponent == 31u) return false;
    if (exponent == 0) { output = uint8_t(sign >> 24u); return true; }
    const uint32_t bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return encode_f32(value, output);
}

// E4M3FN -> F32 is exact. Only 0x7f/0xff are invalid.
inline bool decode_f32(uint8_t byte, float & output) {
    const uint8_t magnitude = byte & 0x7fu;
    if (magnitude == 0x7fu) return false;
    const uint32_t sign = uint32_t(byte & 0x80u) << 24u;
    const unsigned exponent = magnitude >> 3u, mantissa = magnitude & 7u;
    uint32_t bits = sign;
    if (exponent != 0) {
        bits |= ((exponent + 120u) << 23u) | (mantissa << 20u);
    } else if (mantissa != 0) {
        const unsigned power = mantissa >= 4 ? 2u : mantissa >= 2 ? 1u : 0u;
        bits |= ((power + 118u) << 23u) | ((mantissa << (23u - power)) & 0x7fffffu);
    }
    std::memcpy(&output, &bits, sizeof(bits));
    return true;
}

// Every finite E4M3FN value is exactly representable in F16.
inline bool decode_f16(uint8_t byte, uint16_t & output) {
    const uint8_t magnitude = byte & 0x7fu;
    if (magnitude == 0x7fu) return false;
    const uint16_t sign = (byte & 0x80u) ? 0x8000u : 0u;
    const unsigned exponent = magnitude >> 3u, mantissa = magnitude & 7u;
    if (exponent == 0) {
        if (mantissa == 0) { output = sign; return true; }
        const unsigned power = mantissa >= 4 ? 2u : mantissa >= 2 ? 1u : 0u;
        output = uint16_t(sign | ((power + 6u) << 10u) |
                          ((mantissa << (10u - power)) & 0x3ffu));
    } else {
        output = uint16_t(sign | ((exponent + 8u) << 10u) | (mantissa << 7u));
    }
    return true;
}
} // namespace rkva_e4
