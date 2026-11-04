#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

// Shared by context admission, the RKVA client and the Xbox CPU validator.
namespace ggml_remote_attn_policy {
inline bool e4_requested(const char * wire, const char * bulk_wire, const char * bulk) {
    const char * selected = wire ? wire : bulk_wire ? bulk_wire : bulk;
    return selected && std::strcmp(selected, "fp8_e4m3fn") == 0;
}

inline bool e4_remote_mode_allowed(bool remote_requested, bool tcp_position_split, bool requested) {
    return !requested || !remote_requested || tcp_position_split;
}

inline bool e4_empty_prefix(int32_t position, uint32_t origin) {
    return position >= 0 && uint32_t(position) < origin;
}

inline bool e4_lse_valid(float lse, int32_t position, uint32_t origin) {
    if (position < 0) return false;
    uint32_t bits;
    std::memcpy(&bits, &lse, sizeof(bits));
    return e4_empty_prefix(position, origin) ? bits == 0xff800000u :
            (bits & 0x7f800000u) != 0x7f800000u;
}

// One complete op22 reply: OUT E4 bytes then F32 LSE, row-major.
inline bool e4_reply_valid(const uint8_t * data, size_t bytes, size_t nq, size_t width,
                           size_t heads, const int32_t * positions, uint32_t origin) {
    constexpr size_t max = std::numeric_limits<size_t>::max();
    if (!data || !positions || !nq || !width || !heads || width % heads ||
            nq > max / width || nq > max / heads || nq * heads > max / sizeof(float)) return false;
    const size_t out_bytes = nq * width, lse_bytes = nq * heads * sizeof(float);
    if (out_bytes > max - lse_bytes || bytes != out_bytes + lse_bytes) return false;
    for (size_t row = 0; row < nq; ++row) {
        const bool empty = e4_empty_prefix(positions[row], origin);
        if (positions[row] < 0) return false;
        for (size_t col = 0; col < width; ++col) {
            const uint8_t magnitude = data[row * width + col] & 0x7fu;
            if (magnitude == 0x7fu || (empty && magnitude != 0)) return false;
        }
        for (size_t h = 0; h < heads; ++h) {
            float lse;
            std::memcpy(&lse, data + out_bytes + (row * heads + h) * sizeof(float), sizeof(lse));
            if (!e4_lse_valid(lse, positions[row], origin)) return false;
        }
    }
    return true;
}
} // namespace ggml_remote_attn_policy
