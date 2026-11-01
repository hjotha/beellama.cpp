#pragma once

// Phase 3 managed snapshot PREPARE/COMMIT/receipt core: activation gate.
//
// The Phase 3 core (host journal anchor/nonce lifecycle, managed auth command
// contract, offline registry reconciliation gate and the anchored SQLite
// pending-pin/receipt backend) is compiled only when the build defines
// RKVA_MANAGED_SNAPSHOT_PHASE3. The runtime gate is a second, independent
// opt-in and defaults to false.
//
// Phase 3 deliberately has NO server, RPC, capability, dispatch, GC or
// request-handler call site. Nothing is advertised in HELLO and no operation
// number is returned in `supported_ops`. Including this header changes no
// existing behaviour, and enabled() returns false in every default build.
//
// Enabling this gate still does not authorize publication: the v3 object
// writer/verifier and the SQLite managed `apply` transition remain
// unconditionally fail-closed until a concrete authenticated pin/metadata
// proof and a qualified durable sync/rename backend exist.

#include <cstdlib>
#include <cstring>

namespace rkva_managed_phase3 {

#if defined(RKVA_MANAGED_SNAPSHOT_PHASE3)
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
    const char * value = std::getenv("RKVA_MANAGED_SNAPSHOT_PHASE3");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

} // namespace rkva_managed_phase3
