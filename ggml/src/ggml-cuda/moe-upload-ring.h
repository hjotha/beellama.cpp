#pragma once

#include <cstddef>

struct ggml_moe_upload_ring {
    static constexpr size_t slots = 2;
    static constexpr size_t chunk_bytes = 4*1024*1024;

    size_t next = 0;

    static constexpr size_t chunk(size_t remaining) {
        return remaining < chunk_bytes ? remaining : chunk_bytes;
    }

    constexpr size_t take() {
        const size_t slot = next;
        next = (slot + 1) % slots;
        return slot;
    }
};
