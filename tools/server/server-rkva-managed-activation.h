#pragma once

// Phase 5 managed snapshot activation gate (single store authority, successful
// COMMIT/STATUS reply, listener session authority and lost-ACK recovery).
//
// The activation layer is compiled only when the build defines
// RKVA_MANAGED_SNAPSHOT_ACTIVATION (itself only allowed with Phase 4 on). The
// runtime gate is a second, independent opt-in and defaults to false.
//
// There is deliberately NO server, listener, RPC, capability, HELLO, dispatch
// or GC call site. Nothing is advertised in HELLO, no managed operation number
// is returned in `supported_ops`, and no connection is minted outside tests.
// Including this header changes no existing behaviour, and enabled() returns
// false in every default build.
//
// Enabling this gate still does not authorize publication: there is no concrete
// authenticated pin/metadata proof (item 2), no qualified durable sync/rename
// backend (item 3) and no UWP/console qualification (item 7).

#include <cstdlib>
#include <cstring>

namespace rkva_managed_activation {

#if defined(RKVA_MANAGED_SNAPSHOT_ACTIVATION)
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
    const char * value = std::getenv("RKVA_MANAGED_SNAPSHOT_ACTIVATION");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

} // namespace rkva_managed_activation
