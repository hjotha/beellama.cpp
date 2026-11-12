#pragma once

#include "src/llama-ext.h"

enum class server_native_ram_transaction_result { restored, rejected_clean, invalidated };

inline bool server_native_ram_cache_route(uint32_t sequences, uint32_t split, int64_t position, bool native_records) {
    return sequences > 1 && split > 0 && position >= int64_t(split) && native_records;
}

template<typename Slots, typename Publish, typename Abort>
void server_native_ram_invalidate_shared_context(Slots & slots, bool & unavailable, Publish publish, Abort abort) {
    for (auto & slot : slots) slot.prompt_cleanup_failed = true;
    unavailable = true;
    publish();
    abort();
}

template<typename Failed, typename Invalidate>
bool server_native_ram_idle_clear_safe(bool cleared, Failed failed, Invalidate invalidate) {
    if (!cleared && failed()) {
        invalidate();
        return false;
    }
    return true;
}

// The caller prepares and reserves both frames and all scratch before entry.
// A failed clear or rollback can leave state partial and is always invalidated.
template<typename Validate, typename Clear, typename Restore>
server_native_ram_transaction_result server_native_ram_transaction(
        bool occupied, Validate validate, Clear clear, Restore restore) {
    auto source = validate(false);
    if (source == LLAMA_NATIVE_RAM_FAILED_INVALIDATED) return server_native_ram_transaction_result::invalidated;
    if (source != LLAMA_NATIVE_RAM_OK) return server_native_ram_transaction_result::rejected_clean;
    if (occupied) {
        auto backup = validate(true);
        if (backup == LLAMA_NATIVE_RAM_FAILED_INVALIDATED) return server_native_ram_transaction_result::invalidated;
        if (backup != LLAMA_NATIVE_RAM_OK) return server_native_ram_transaction_result::rejected_clean;
        if (!clear()) return server_native_ram_transaction_result::invalidated;
    }
    auto result = restore(false);
    if (result == LLAMA_NATIVE_RAM_FAILED_INVALIDATED) return server_native_ram_transaction_result::invalidated;
    if (result == LLAMA_NATIVE_RAM_OK) return server_native_ram_transaction_result::restored;
    if (occupied && restore(true) != LLAMA_NATIVE_RAM_OK) return server_native_ram_transaction_result::invalidated;
    return server_native_ram_transaction_result::rejected_clean;
}
