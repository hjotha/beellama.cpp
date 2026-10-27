#pragma once

// Pure eligibility contract for the explicit target-only native streaming slot
// API. The server handlers use this so the same decisions can be regression
// tested without a model, GPU or remote peer.

#include "rkva-native-state.h"

#include <cstdint>
#include <string>

namespace server_native_slot {

struct Eligibility {
    bool native_stream = false;
    bool adaptive = false;
    bool profile_xxxl = false;
    bool one_sequence = false;
    bool snapshot_supported = false;
    bool native_active = false; // SAVE only
    bool draft_present = false;
    bool can_speculate = false;
    bool incompatible_route_flags = false;
    bool slot_empty = false;        // RESTORE only
    bool all_sequences_empty = false; // RESTORE only
    bool media = false;
    bool identity_available = false;
    bool unmodified_model = false;
    bool text_only = false;
};

inline bool save_allowed(const Eligibility & e, std::string * error) {
    const auto reject = [&](const char * message) {
        if (error) *error = message;
        return false;
    };
    if (!e.native_stream)
        return reject("native_stream is required");
    if (!e.adaptive)
        return reject("native streaming slots require the adaptive server");
    if (!e.profile_xxxl)
        return reject("native streaming slots require the xxxl adaptive profile (select it with a small completion budget, then erase the target slot)");
    if (e.draft_present || e.can_speculate)
        return reject("native streaming slots are target-only; MTP/draft state is unsupported");
    if (!e.one_sequence)
        return reject("native streaming slots require exactly one sequence");
    if (!e.snapshot_supported)
        return reject("native streaming peer/geometry is not qualified");
    if (!e.native_active)
        return reject("native overflow is not resident in this slot");
    if (!e.identity_available)
        return reject("native streaming slots require a verified model identity");
    if (!e.unmodified_model)
        return reject("native streaming slots require an unmodified model without LoRA or control vectors");
    if (!e.text_only)
        return reject("native streaming slots support text-only requests");
    if (e.incompatible_route_flags)
        return reject("native_stream cannot be combined with route_state_transfer, route_target_no_mtp or route_state_reuse");
    if (e.media)
        return reject("native streaming slots support text-only prompts");
    return true;
}

inline bool restore_allowed(const Eligibility & e, std::string * error) {
    const auto reject = [&](const char * message) {
        if (error) *error = message;
        return false;
    };
    if (!e.native_stream)
        return reject("native_stream is required");
    if (!e.adaptive)
        return reject("native streaming slots require the adaptive server");
    if (!e.profile_xxxl)
        return reject("native streaming slots require the xxxl adaptive profile (select it with a small completion budget, then erase the target slot)");
    if (e.draft_present || e.can_speculate)
        return reject("native streaming slots are target-only; MTP/draft state is unsupported");
    if (!e.one_sequence)
        return reject("native streaming slots require exactly one sequence");
    if (!e.snapshot_supported)
        return reject("native streaming peer/geometry is not qualified");
    if (e.native_active)
        return reject("native streaming restore requires cleared native owner state");
    if (!e.identity_available)
        return reject("native streaming slots require a verified model identity");
    if (!e.unmodified_model)
        return reject("native streaming slots require an unmodified model without LoRA or control vectors");
    if (!e.text_only)
        return reject("native streaming slots support text-only requests");
    if (e.incompatible_route_flags)
        return reject("native_stream cannot be combined with route_state_transfer, route_target_no_mtp or route_state_reuse");
    if (!e.slot_empty || !e.all_sequences_empty)
        return reject("native streaming restore requires an empty slot and empty sequence state");
    if (e.media)
        return reject("native streaming slots support text-only prompts");
    return true;
}

inline uint64_t restore_peak_bytes(const rkva_native_state::SetHeader & header) {
    return rkva_native_state::restore_peak_bytes(header);
}

inline bool restore_fits(uint64_t peak, uint64_t budget) {
    return peak != UINT64_MAX && peak <= budget;
}

// Cumulative admission for the real restore peak: local bounded-streaming
// workspace plus the native staged/owner shadow. The server budget is the
// smaller of the configured cache-ram limit and measured MemAvailable after
// the existing 25% reserve.
inline uint64_t save_workspace_bytes(uint64_t n_ctx) {
    constexpr uint64_t base = 16ull << 20;
    if (n_ctx > (UINT64_MAX - base) / 16ull)
        return UINT64_MAX;
    return base + n_ctx * 16ull;
}

inline bool cumulative_fits(uint64_t workspace, uint64_t native_peak,
                            uint64_t budget, uint64_t * total = nullptr) {
    if (native_peak == UINT64_MAX || workspace > UINT64_MAX - native_peak)
        return false;
    const uint64_t sum = workspace + native_peak;
    if (total)
        *total = sum;
    return sum <= budget;
}

} // namespace server_native_slot
