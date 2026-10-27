#pragma once

// Shared framing decision for the bytes that follow the llama sequence-state
// payload. Legacy FULL files may carry the server SCKP checkpoint appendix;
// native FULL files recognize only EOF or SCKP and fail closed otherwise.

#include <cstddef>
#include <cstdint>

namespace llama_state_framing {

constexpr uint32_t slot_checkpoint_magic = 0x504b4353u; // "SCKP"

enum class trailing {
    none,
    slot_checkpoint,
    truncated,
    unknown,
};

inline trailing classify(const uint8_t * data, size_t size) {
    if (size == 0) {
        return trailing::none;
    }
    if (size < sizeof(uint32_t)) {
        return trailing::truncated;
    }
    const uint32_t magic = uint32_t(data[0]) | (uint32_t(data[1]) << 8) |
                           (uint32_t(data[2]) << 16) | (uint32_t(data[3]) << 24);
    return magic == slot_checkpoint_magic ? trailing::slot_checkpoint : trailing::unknown;
}

} // namespace llama_state_framing
