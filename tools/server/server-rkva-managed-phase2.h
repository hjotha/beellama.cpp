#pragma once

// Phase 2 managed snapshot lifecycle core: activation gate.
//
// The lifecycle core (managed registry, host journal, orchestrator, store
// authority, v3 object component and quota scan) is compiled only when the
// build defines RKVA_MANAGED_SNAPSHOT_PHASE2. The runtime gate is a second,
// independent opt-in and defaults to false.
//
// Phase 2 deliberately has NO server, RPC, capability, GC or request-handler
// call site. This header exists so a future Phase 3 handler must pass both
// gates explicitly; including it changes no existing behaviour, and
// enabled() returns false in every default build.

#include <cstdlib>
#include <cstring>

namespace rkva_managed_phase2 {

#if defined(RKVA_MANAGED_SNAPSHOT_PHASE2)
inline constexpr bool compiled_in = true;
#else
inline constexpr bool compiled_in = false;
#endif

// Runtime opt-in. Unset, empty, "0" or any value other than "1" means off.
// A compiled-out build always reports false, even if the variable is set.
inline bool enabled() {
    if (!compiled_in) {
        return false;
    }
    const char * value = std::getenv("RKVA_MANAGED_SNAPSHOT_PHASE2");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

} // namespace rkva_managed_phase2
