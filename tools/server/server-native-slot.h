#pragma once

// Pure eligibility contract for the explicit target-only native streaming slot
// API. The server handlers use this so the same decisions can be regression
// tested without a model, GPU or remote peer.

#include "rkva-native-state.h"

#include <cstdint>
#include <string>
#include <algorithm>

namespace server_native_slot {

enum class automatic_restore_outcome { no_retry, native_preflight_rejected };

inline bool local_fallback_eligible(bool split_active, bool has_marker, bool native,
        size_t tokens, uint32_t split_p, bool legacy_pure) {
    return split_active && (has_marker ? !native : legacy_pure) &&
        split_p > 0 && tokens > 0 && tokens <= split_p;
}

// Mark entry before invoking the loader: zero, an exception, and success are
// all potentially mutating once the function has been called.
template <class Loader>
auto invoke_native_loader(bool * called, Loader loader) -> decltype(loader()) {
    if (called) *called = true;
    return loader();
}

// The caller decides the native outcome immediately around the loader. A
// second lookup is allowed only before that loader was ever entered, and must
// return a different, non-native candidate into a still-empty destination.
template <class Candidate, class Attempt, class Lookup, class StillEmpty>
void retry_local_after_preflight(const Candidate & selected, Attempt attempt,
        Lookup lookup_local, StillEmpty still_empty) {
    const auto outcome = attempt(selected);
    if (!selected.native_candidate || outcome != automatic_restore_outcome::native_preflight_rejected ||
            !still_empty()) return;
    const auto local = lookup_local(selected.state_path);
    if (!local || local->native_candidate || local->state_path == selected.state_path) return;
    (void) attempt(*local);
}

// The native owner cannot be arbitrarily rewound. Require an entire snapshot
// and at least one unevaluated request token, even for attention-only models.
template <class Tokens>
bool automatic_prefix(const Tokens & saved, const Tokens & request) {
    return !saved.empty() && saved.size() < request.size() &&
           std::equal(saved.begin(), saved.end(), request.begin());
}

// Legacy mixed-layout envelopes have no recorded origin P. A native owner
// from another split might fit below this P, so only disjoint pure/convertible
// source layouts are eligible without the new checksummed metadata.
inline bool legacy_auto_importable(size_t tokens, uint32_t split_p, bool pure, bool convertible_q4) {
    return split_p > 0 && tokens <= split_p && (pure || convertible_q4);
}

inline bool generic_route_eligible(bool has_marker, bool native_remote, bool legacy_mixed) {
    return has_marker ? !native_remote : !legacy_mixed;
}

// Native streaming runs only with a single target sequence. Once a restore
// has mutated it, clear the entire memory exactly once (which invokes the
// remote reset hook), then forget the slot's bookkeeping. Calling the normal
// slot prompt_clear first would also remove the sequence and send a second
// reset to the peer.
template <class Slot, class ClearMemory>
void clear_failed_restore(Slot & slot, ClearMemory clear_memory) {
    try {
        clear_memory();
    } catch (...) {
        // A throwing clear may already have changed memory. Never leave slot
        // metadata claiming the old state is still usable.
        slot.prompt_forget_memory();
        throw;
    }
    slot.prompt_forget_memory();
}

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
