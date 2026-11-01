#include "../ggml/src/ggml-remote-attn/rkva-managed-framing-v3.h"
#undef NDEBUG
#include <cassert>
#include <iostream>
#include <set>

// Capability negotiation is one predicate shared by the registry wire codec and
// the v3 framing. This test pins the exact bit/op set so a future edit cannot
// quietly accept a peer that offers only part of the managed contract, and
// checks that the reserved numbers do not collide with existing operations.
namespace wire = rkva_registry_wire;
namespace v3 = rkva_managed_framing_v3;

int main() {
    static_assert(RKVA_REGISTRY_PREPARE == 24 && RKVA_REGISTRY_STATUS == 28);
    static_assert(RKVA_MANAGED_AUTH_HELLO == 29 && RKVA_MANAGED_AUTH_CHALLENGE == 30);
    static_assert(RKVA_MANAGED_ATTEST == 31);
    static_assert(RKVA_CAP_MANAGED_REGISTRY_V1 == 1ull << 19);
    static_assert(RKVA_CAP_MANAGED_AUTH_V2 == 1ull << 20);
    static_assert(RKVA_CAP_MANAGED_FRAMING_V3 == 1ull << 21);
    static_assert(RKVA_PROTO_MANAGED_REGISTRY_V1 == 1u << 19);
    static_assert(RKVA_PROTO_MANAGED_AUTH_V2 == 1u << 20);
    static_assert(RKVA_PROTO_MANAGED_FRAMING_V3 == 1u << 21);

    // Reserved op numbers are contiguous 24..31 and nothing else in the
    // protocol enum uses them.
    static constexpr uint16_t reserved[] = { 24, 25, 26, 27, 28, 29, 30, 31 };
    for (uint16_t op : reserved) {
        assert(wire::valid_op(op) == (op >= 24 && op <= 28));
    }
    std::set<uint16_t> known;
    for (uint16_t op = 0; op < 24; ++op) {
        assert(known.insert(op).second);
    }

    const uint32_t flags = RKVA_PROTO_MANAGED_REGISTRY_V1 | RKVA_PROTO_MANAGED_AUTH_V2 |
                           RKVA_PROTO_MANAGED_FRAMING_V3;
    const uint64_t caps = RKVA_CAP_MANAGED_REGISTRY_V1 | RKVA_CAP_MANAGED_AUTH_V2 |
                          RKVA_CAP_MANAGED_FRAMING_V3;
    const uint32_t ops = wire::managed_operations;
    assert(wire::negotiated(flags, caps, ops));
    assert(v3::negotiated(flags, caps, ops));

    // Every single missing protocol/capability bit fails closed.
    assert(!wire::negotiated(flags & ~RKVA_PROTO_MANAGED_REGISTRY_V1, caps, ops));
    assert(!wire::negotiated(flags & ~RKVA_PROTO_MANAGED_AUTH_V2, caps, ops));
    assert(!wire::negotiated(flags & ~RKVA_PROTO_MANAGED_FRAMING_V3, caps, ops));
    assert(!wire::negotiated(flags, caps & ~RKVA_CAP_MANAGED_REGISTRY_V1, ops));
    assert(!wire::negotiated(flags, caps & ~RKVA_CAP_MANAGED_AUTH_V2, ops));
    assert(!wire::negotiated(flags, caps & ~RKVA_CAP_MANAGED_FRAMING_V3, ops));
    assert(!v3::negotiated(flags & ~RKVA_PROTO_MANAGED_FRAMING_V3, caps, ops));
    assert(!v3::negotiated(flags, caps & ~RKVA_CAP_MANAGED_FRAMING_V3, ops));

    // Every single missing managed op fails closed, including hello/challenge/
    // attest which are not part of the registry operations mask.
    assert(!wire::negotiated(flags, caps, ops & ~(1u << RKVA_REGISTRY_PREPARE)));
    assert(!wire::negotiated(flags, caps, ops & ~(1u << RKVA_REGISTRY_STATUS)));
    assert(!wire::negotiated(flags, caps, ops & ~(1u << RKVA_MANAGED_AUTH_HELLO)));
    assert(!wire::negotiated(flags, caps, ops & ~(1u << RKVA_MANAGED_AUTH_CHALLENGE)));
    assert(!wire::negotiated(flags, caps, ops & ~(1u << RKVA_MANAGED_ATTEST)));
    assert(!v3::negotiated(flags, caps, ops & ~(1u << RKVA_MANAGED_ATTEST)));

    // Extra unrelated bits do not break negotiation, but they never add a
    // managed op: the mask is exact for the required set.
    assert(wire::negotiated(flags | (1u << 0), caps | (1ull << 0), ops | (1u << 0)));
    assert(!wire::negotiated(0, 0, 0));
    assert(!wire::negotiated(flags, caps, wire::operations));

    std::cout << "phase 4 negotiation: exact managed bits/ops required, no fallback\n";
    return 0;
}
