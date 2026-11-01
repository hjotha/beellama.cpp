#pragma once

// Managed-registry wire codec (Phase 3 scaffolding, Phase 4 negotiation).
// Nothing in a default build advertises, negotiates or dispatches this
// extension. IDs and snapshot keys are 32 opaque bytes (the prototype
// represents the same bytes as lowercase hex). These identifiers are
// scope/replay keys, never authentication credentials.
//
// PHASE 4 OP ALLOCATION: the reserved numbers live in
// remote-attn-protocol.h (`RKVA_REGISTRY_*`, `RKVA_MANAGED_AUTH_*`) together
// with the managed capability/protocol bits. `negotiated()` below is the only
// capability gate the Phase 4 codec may consult; there is no fallback: a peer
// missing any managed bit or op is refused before any managed payload is read.
#include "remote-attn-protocol.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace rkva_registry_wire {

// Reserved operation numbers, identical to the protocol allocation. No server
// returns them in `supported_ops` and no client sends them while the Phase 4
// gate is closed.
namespace op {
constexpr uint16_t prepare = RKVA_REGISTRY_PREPARE;
constexpr uint16_t commit  = RKVA_REGISTRY_COMMIT;
constexpr uint16_t abort   = RKVA_REGISTRY_ABORT;
constexpr uint16_t release = RKVA_REGISTRY_RELEASE;
constexpr uint16_t status  = RKVA_REGISTRY_STATUS;
} // namespace op

using id = std::array<uint8_t, 32>;
constexpr uint32_t revision = 1;
constexpr size_t request_bytes = 144;
constexpr size_t response_bytes = 152;
using request_wire = std::array<uint8_t, request_bytes>;
using response_wire = std::array<uint8_t, response_bytes>;
constexpr uint32_t operations = (1u << op::prepare) | (1u << op::commit) | (1u << op::abort) |
    (1u << op::release) | (1u << op::status);

// The single managed capability gate. A managed connection must offer the
// registry, auth-v2 and framing-v3 protocol flags, the matching capabilities
// and every reserved operation including the handshake and ATTEST ops.
// Nothing here negotiates a weaker fallback: any missing bit returns false and
// the caller must refuse the frame before reading it.
constexpr uint32_t managed_protocols = RKVA_PROTO_MANAGED_REGISTRY_V1 |
    RKVA_PROTO_MANAGED_AUTH_V2 | RKVA_PROTO_MANAGED_FRAMING_V3;
constexpr uint64_t managed_capabilities = RKVA_CAP_MANAGED_REGISTRY_V1 |
    RKVA_CAP_MANAGED_AUTH_V2 | RKVA_CAP_MANAGED_FRAMING_V3;
constexpr uint32_t managed_operations = operations | (1u << RKVA_MANAGED_AUTH_HELLO) |
    (1u << RKVA_MANAGED_AUTH_CHALLENGE) | (1u << RKVA_MANAGED_ATTEST);
inline bool negotiated(uint32_t protocol_flags, uint64_t capabilities, uint32_t supported_ops) {
    return (protocol_flags & managed_protocols) == managed_protocols &&
           (capabilities & managed_capabilities) == managed_capabilities &&
           (supported_ops & managed_operations) == managed_operations;
}

inline bool valid_op(uint16_t value) {
    return value >= op::prepare && value <= op::status;
}
inline bool all_zero(const id & value) {
    return std::all_of(value.begin(), value.end(), [](uint8_t b) { return b == 0; });
}
inline void put32(uint8_t * p, uint32_t v) {
    for (unsigned i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i));
}
inline void put64(uint8_t * p, uint64_t v) {
    for (unsigned i = 0; i < 8; ++i) p[i] = uint8_t(v >> (8 * i));
}
inline uint32_t get32(const uint8_t * p) {
    uint32_t v = 0; for (unsigned i = 0; i < 4; ++i) v |= uint32_t(p[i]) << (8 * i); return v;
}
inline uint64_t get64(const uint8_t * p) {
    uint64_t v = 0; for (unsigned i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i); return v;
}
inline void put_id(uint8_t * p, const id & v) { std::copy(v.begin(), v.end(), p); }
inline id get_id(const uint8_t * p) { id v; std::copy(p, p + v.size(), v.begin()); return v; }

inline bool from_hex(const std::string & s, id & out) {
    if (s.size() != 64) return false;
    id value{};
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    for (size_t i = 0; i < value.size(); ++i) {
        const int hi = nibble(s[2*i]), lo = nibble(s[2*i + 1]);
        if (hi < 0 || lo < 0) return false;
        value[i] = uint8_t((hi << 4) | lo);
    }
    out = value;
    return true;
}
inline std::string to_hex(const id & v) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string s(64, '0');
    for (size_t i = 0; i < v.size(); ++i) {
        s[2*i] = digits[v[i] >> 4]; s[2*i + 1] = digits[v[i] & 15];
    }
    return s;
}

struct request {
    id owner{}, tx{}, reference{}, key{};
    uint64_t committed_version = 0;
};
// Unused fields must be zero. Required identifiers can contain any 32 bytes,
// including all zero, matching the prototype's full 64-hex-character domain.
inline bool valid_request(uint16_t operation, const request & r) {
    switch (operation) {
    case op::prepare: return r.committed_version == 0;
    case op::commit:
    case op::abort:
    case op::status: return all_zero(r.reference) && all_zero(r.key) && r.committed_version == 0;
    case op::release: return all_zero(r.tx) && all_zero(r.key) && r.committed_version != 0;
    default: return false;
    }
}
inline bool encode_request(uint16_t operation, const request & r, request_wire & out) {
    if (!valid_request(operation, r)) return false;
    request_wire b{};
    put32(b.data(), revision); // bytes 4..7 reserved and zero
    put_id(b.data() + 8, r.owner); put_id(b.data() + 40, r.tx);
    put_id(b.data() + 72, r.reference); put_id(b.data() + 104, r.key);
    put64(b.data() + 136, r.committed_version);
    out = b; return true;
}
inline bool decode_request(uint16_t operation, const uint8_t * bytes, size_t len, request & out) {
    if (!bytes || len != request_bytes || get32(bytes) != revision || get32(bytes + 4) != 0) return false;
    request r; r.owner = get_id(bytes + 8); r.tx = get_id(bytes + 40);
    r.reference = get_id(bytes + 72); r.key = get_id(bytes + 104);
    r.committed_version = get64(bytes + 136);
    if (!valid_request(operation, r)) return false;
    out = r; return true;
}

enum class phase : uint32_t { pending = 1, committed = 2, aborted = 3, released = 4 };
struct response {
    phase state = phase::pending;
    uint64_t registry_version = 0, committed_version = 0;
    id owner{}, tx{}, reference{}, key{};
};
// Success describes one record, including STATUS. Unknown/conflicting IDs,
// persistence errors and unsupported operations use nonzero RKVA status and
// no success body. Server must obtain this record atomically with its update.
inline bool valid_response(uint16_t operation, const request & q, const response & r) {
    if (!valid_request(operation, q) || !r.registry_version || r.owner != q.owner) return false;
    const bool committed = r.state == phase::committed || r.state == phase::released;
    if (r.state != phase::pending && r.state != phase::committed &&
            r.state != phase::aborted && r.state != phase::released) return false;
    if (committed ? (!r.committed_version || r.committed_version > r.registry_version) : r.committed_version != 0) return false;
    switch (operation) {
    case op::prepare:
        return (r.state == phase::pending || r.state == phase::committed) &&
            r.tx == q.tx && r.reference == q.reference && r.key == q.key;
    case op::commit: return r.state == phase::committed && r.tx == q.tx;
    case op::abort: return r.state == phase::aborted && r.tx == q.tx;
    case op::release:
        return r.state == phase::released && r.reference == q.reference && r.committed_version == q.committed_version;
    case op::status: return r.tx == q.tx;
    default: return false;
    }
}
inline bool encode_response(uint16_t operation, const request & q, const response & r, response_wire & out) {
    if (!valid_response(operation, q, r)) return false;
    response_wire b{}; put32(b.data(), revision); put32(b.data() + 4, uint32_t(r.state));
    put64(b.data() + 8, r.registry_version); put64(b.data() + 16, r.committed_version);
    put_id(b.data() + 24, r.owner); put_id(b.data() + 56, r.tx);
    put_id(b.data() + 88, r.reference); put_id(b.data() + 120, r.key);
    out = b; return true;
}
inline bool decode_response(uint16_t operation, const request & q, const uint8_t * bytes, size_t len, response & out) {
    if (!bytes || len != response_bytes || get32(bytes) != revision) return false;
    response r; r.state = phase(get32(bytes + 4)); r.registry_version = get64(bytes + 8);
    r.committed_version = get64(bytes + 16); r.owner = get_id(bytes + 24); r.tx = get_id(bytes + 56);
    r.reference = get_id(bytes + 88); r.key = get_id(bytes + 120);
    if (!valid_response(operation, q, r)) return false;
    out = r; return true;
}

// Phase 4 defines the authenticated negotiation above. The outer
// rkva_request/rkva_response envelope is bound by `rkva-managed-framing-v3.h`;
// this header deliberately has no socket, listener or dispatch call site.
} // namespace rkva_registry_wire
