#pragma once

#include "rkva-native-state.h"
#include "rkva-snapshot-digest.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace llama_native_ram_frame {
constexpr uint32_t magic = 0x4d524b52;
constexpr uint32_t version = 1;
constexpr size_t header_bytes = 24;

inline void write_header(uint8_t * out, uint32_t seq, uint32_t evaluated, uint64_t size) {
    rkva_native_state::put_u32(out, magic);
    rkva_native_state::put_u32(out + 4, version);
    rkva_native_state::put_u32(out + 8, seq);
    rkva_native_state::put_u32(out + 12, evaluated);
    rkva_native_state::put_u64(out + 16, size);
}

inline bool validate(const uint8_t * data, size_t size, uint32_t evaluated, uint32_t max_seq,
        const uint8_t sha256[32]) {
    if (!data || !sha256 || size < header_bytes) return false;
    const auto digest = rkva_snapshot::digest(data, size);
    return !std::memcmp(digest.data(), sha256, digest.size()) &&
           rkva_native_state::get_u32(data) == magic &&
           rkva_native_state::get_u32(data + 4) == version &&
           rkva_native_state::get_u32(data + 8) < max_seq &&
           rkva_native_state::get_u32(data + 12) == evaluated &&
           rkva_native_state::get_u64(data + 16) == size;
}
}
