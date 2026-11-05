#include "../../../ggml/src/ggml-remote-attn/rkva-fp8-e4m3.h"

#undef NDEBUG
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

using uint = uint32_t;
#include "../shaders/rkva/rkva_e4m3_codec.hlsli"

static std::array<double, 127> reference_levels() {
    std::array<double, 127> result{};
    for (unsigned i = 0; i < result.size(); ++i) {
        const unsigned exponent = i / 8, mantissa = i % 8;
        result[i] = exponent == 0 ? std::ldexp(double(mantissa), -9)
                                  : (1.0 + double(mantissa) / 8.0) * std::ldexp(1.0, int(exponent) - 7);
    }
    return result;
}

static uint8_t reference_encode(float value, const std::array<double, 127> & levels) {
    const double magnitude = std::abs(double(value));
    auto it = std::lower_bound(levels.begin(), levels.end(), magnitude);
    size_t upper = size_t(it - levels.begin());
    if (upper == levels.size()) upper = levels.size() - 1;
    const size_t lower = upper ? upper - 1 : 0;
    const double below = magnitude - levels[lower], above = levels[upper] - magnitude;
    const size_t code = (above < below || (above == below && !(upper & 1))) ? upper : lower;
    return uint8_t(code | (std::signbit(value) ? 0x80u : 0u));
}

static float reference_half(uint16_t half) {
    const unsigned exponent = (half >> 10u) & 31u, mantissa = half & 1023u;
    const float value = exponent == 0 ? std::ldexp(float(mantissa), -24)
                                      : std::ldexp(float(1024u + mantissa), int(exponent) - 25);
    return (half & 0x8000u) ? std::copysign(value, -1.0f) : value;
}

static void check_f32(float value, const std::array<double, 127> & levels) {
    uint8_t got = 0xa5;
    const bool ok = rkva_e4::encode_f32(value, got);
    if (!std::isfinite(value)) { assert(!ok && got == 0xa5); return; }
    assert(ok && got == reference_encode(value, levels));
}

int main() {
    const auto levels = reference_levels();
    assert(levels[1] == std::ldexp(1.0, -9) && levels[126] == 448.0);
    unsigned decoded = 0, rejected_codes = 0;
    for (unsigned code = 0; code < 256; ++code) {
        float value = 123.0f;
        uint16_t half = 0x3555;
        const bool valid_f32 = rkva_e4::decode_f32(uint8_t(code), value);
        const bool valid_f16 = rkva_e4::decode_f16(uint8_t(code), half);
        if ((code & 0x7fu) == 0x7fu) {
            assert(!valid_f32 && !valid_f16 && value == 123.0f && half == 0x3555);
            ++rejected_codes;
            continue;
        }
        const double expected = (code & 0x80u) ? -levels[code & 0x7fu] : levels[code];
        const float from_half = reference_half(half);
        assert(valid_f32 && valid_f16 && double(value) == expected);
        assert(e4m3fn_to_half(code) == half);
        assert(value == from_half && std::signbit(value) == std::signbit(from_half));
        uint8_t encoded = 0xff;
        assert(rkva_e4::encode_f32(value, encoded) && encoded == code);
        assert(rkva_e4::encode_f16(half, encoded) && encoded == code);
        ++decoded;
    }
    assert(decoded == 254 && rejected_codes == 2);

    unsigned finite_half = 0, rejected_half = 0;
    for (unsigned half = 0; half < 65536; ++half) {
        uint8_t got = 0xa5;
        const bool ok = rkva_e4::encode_f16(uint16_t(half), got);
        if ((half & 0x7c00u) == 0x7c00u) {
            assert(!ok && got == 0xa5);
            ++rejected_half;
        } else {
            const float value = reference_half(uint16_t(half));
            assert(ok && got == reference_encode(value, levels));
            assert(e4m3fn_from_half(half) == got);
            check_f32(value, levels);
            ++finite_half;
        }
    }
    assert(finite_half == 63488 && rejected_half == 2048);

    for (size_t i = 0; i + 1 < levels.size(); ++i) {
        const float midpoint = float((levels[i] + levels[i + 1]) / 2.0);
        for (float value : {std::nextafter(midpoint, 0.0f), midpoint,
                            std::nextafter(midpoint, std::numeric_limits<float>::infinity())}) {
            check_f32(value, levels);
            check_f32(-value, levels);
        }
    }
    std::mt19937 rng(12345);
    for (unsigned i = 0; i < 262144; ++i) {
        const uint32_t bits = rng();
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        check_f32(value, levels);
    }
    for (float value : {std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) check_f32(value, levels);

    // A value just above the midpoint distinguishes direct Q from staged OUT.
    uint8_t direct = 0, staged = 0;
    const float above_midpoint = std::nextafter(1.0625f, 2.0f);
    assert(rkva_e4::encode_f32(above_midpoint, direct) && direct == 0x39);
    assert(rkva_e4::encode_f16(0x3c40, staged) && staged == 0x38);
    assert(!rkva_e4::encode_f16(0x7c00, staged) && staged == 0x38);

    std::printf("E4M3FN: 256 codes, 65536 F16 patterns, 126 midpoint triplets "
                "for each sign, 262144 sampled F32 values and staged OUT passed\n");
}
