#pragma once

// Phase 4 managed snapshot framing/ATTEST/peer-identity activation gate.
//
// The Phase 4 layer (single v3 authenticated frame, pre-HELLO challenge,
// anchored HELLO/HELLO ACK, ATTEST evidence, resident-attestation metadata,
// durable v3 nonce ledger and the authenticated negative-ACK host guard) is
// compiled only when the build defines RKVA_MANAGED_SNAPSHOT_PHASE4. The
// runtime gate is a second, independent opt-in and defaults to false.
//
// Phase 4 deliberately has NO listener, RPC, capability, HELLO, dispatch or
// request-handler call site. Nothing is advertised in HELLO, no operation
// number is added to `supported_ops`, and no frame is parsed before a peer
// explicitly negotiates RKVA_PROTO_MANAGED_REGISTRY_V1 |
// RKVA_PROTO_MANAGED_AUTH_V2 | RKVA_PROTO_MANAGED_FRAMING_V3 plus every
// reserved op through `rkva_registry_wire::negotiated()`. Including this
// header changes no existing behaviour, and enabled() returns false in every
// default build.
//
// Enabling this gate still does not authorize publication: the object v3
// writer/verifier and the SQLite managed `apply` transition remain
// unconditionally fail-closed, and there is no authenticated successful
// COMMIT/ATTEST reply guard or durable sync/rename backend.

#include <cstdlib>
#include <cstring>

namespace rkva_managed_phase4 {

#if defined(RKVA_MANAGED_SNAPSHOT_PHASE4)
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
    const char * value = std::getenv("RKVA_MANAGED_SNAPSHOT_PHASE4");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

// A frame may only be decoded when the build is compiled in, the runtime gate
// is open and the peer negotiated every managed bit and op. This is a pure
// predicate; it grants no authority and performs no I/O.
template <typename Negotiated>
inline bool accepts_negotiated(Negotiated negotiated) {
    return enabled() && negotiated;
}

} // namespace rkva_managed_phase4
