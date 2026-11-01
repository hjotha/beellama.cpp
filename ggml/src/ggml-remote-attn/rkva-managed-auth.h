#pragma once

// Standalone authenticated-command contract. No runtime advertises or dispatches
// it. Do not enable it without durable atomic replay admission and response
// authentication, secure key provisioning, managed object transactions and UWP
// qualification. HMAC authenticates but does not encrypt the command.
#include "rkva-managed-registry-wire.h"
#include "rkva-snapshot-digest.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rkva_managed_auth {
using digest = rkva_snapshot::Digest;
using secret = std::array<uint8_t, 32>;
using nonce = std::array<uint8_t, 32>;
constexpr size_t request_bytes = rkva_registry_wire::request_bytes + 32 + 32;
using wire = std::array<uint8_t, request_bytes>;

// Wipe temporary HMAC key blocks, including keyed hash state. This does not
// erase caller-owned key copies, registers, OS pages or compiler-created copies.
inline void wipe(void * p, size_t n) {
    volatile uint8_t * b = static_cast<volatile uint8_t *>(p);
    while (n--) *b++ = 0;
}
inline bool hmac_sha256(const uint8_t * key, size_t key_len, const uint8_t * data,
                        size_t data_len, digest & out) {
    if ((!key && key_len) || (!data && data_len) || key_len > 4096 || data_len > (16u << 20)) return false;
    std::array<uint8_t, 64> block{}, ipad{}, opad{};
    if (key_len > block.size()) {
        auto kh = rkva_snapshot::digest(key, key_len);
        std::copy(kh.begin(), kh.end(), block.begin()); wipe(kh.data(), kh.size());
    } else if (key_len) {
        std::copy(key, key + key_len, block.begin());
    }
    for (size_t i = 0; i < block.size(); ++i) { ipad[i] = block[i] ^ 0x36; opad[i] = block[i] ^ 0x5c; }
    rkva_snapshot::Sha256 inner; inner.update(ipad.data(), ipad.size()); inner.update(data, data_len);
    auto mid = inner.finish();
    rkva_snapshot::Sha256 outer; outer.update(opad.data(), opad.size()); outer.update(mid.data(), mid.size());
    out = outer.finish();
    wipe(&inner, sizeof(inner)); wipe(&outer, sizeof(outer)); wipe(mid.data(), mid.size());
    wipe(block.data(), block.size()); wipe(ipad.data(), ipad.size()); wipe(opad.data(), opad.size());
    return true;
}
// Always visits all 32 bytes; volatile accumulator prevents an early-exit
// comparison. Constant-time machine code remains a platform/compiler property
// that must be audited on each qualified build, including UWP.
inline bool equal_tag(const uint8_t * a, const uint8_t * b) {
    volatile uint8_t diff = 0;
    for (size_t i = 0; i < 32; ++i) diff = uint8_t(diff | (a[i] ^ b[i]));
    return diff == 0;
}
inline void put16(uint8_t * p, uint16_t n) { p[0] = uint8_t(n); p[1] = uint8_t(n >> 8); }
inline bool valid_header(const rkva_request & h) {
    return h.magic == RKVA_MAGIC && h.version == RKVA_VERSION && rkva_registry_wire::valid_op(h.op) &&
        h.session_id == RKVA_SESSION_NONE && h.layer_id == 0 && h.n_tokens == 0 && h.payload_bytes == request_bytes;
}
// The MAC input is explicit and endian-stable: domain (no NUL), the complete
// 32-byte outer RKVA header serialized little-endian, then 144-byte canonical
// managed request and 32-byte nonce. The request contains owner/tx/ref/key and
// commit version. This is a distinct envelope from the reserved unauthenticated
// 144-byte codec, and MUST get separate explicit authenticated negotiation.
inline bool tag_request(const secret & key, const rkva_request & h, const uint8_t * body,
                        const nonce & n, digest & tag) {
    if (!valid_header(h) || !body || rkva_registry_wire::all_zero(key) || rkva_registry_wire::all_zero(n)) return false;
    static constexpr char domain[] = "RKVA-MANAGED-COMMAND-HMAC-SHA256-V1";
    constexpr size_t prefix = sizeof(domain) - 1;
    std::array<uint8_t, prefix + 32 + rkva_registry_wire::request_bytes + 32> input{};
    std::memcpy(input.data(), domain, prefix);
    auto * p = input.data() + prefix;
    rkva_registry_wire::put32(p, h.magic); put16(p + 4, h.version); put16(p + 6, h.op);
    rkva_registry_wire::put64(p + 8, h.cycle_id); rkva_registry_wire::put32(p + 16, h.session_id);
    rkva_registry_wire::put32(p + 20, h.layer_id); rkva_registry_wire::put32(p + 24, h.n_tokens);
    rkva_registry_wire::put32(p + 28, h.payload_bytes);
    std::memcpy(p + 32, body, rkva_registry_wire::request_bytes);
    std::copy(n.begin(), n.end(), p + 32 + rkva_registry_wire::request_bytes);
    return hmac_sha256(key.data(), key.size(), input.data(), input.size(), tag);
}
inline bool encode(const secret & key, const rkva_request & h, const rkva_registry_wire::request & q,
                   const nonce & n, wire & out) {
    rkva_registry_wire::request_wire body{}; digest tag{};
    if (!valid_header(h) || !rkva_registry_wire::encode_request(h.op, q, body) ||
        !tag_request(key, h, body.data(), n, tag)) return false;
    wire result{}; std::copy(body.begin(), body.end(), result.begin());
    std::copy(n.begin(), n.end(), result.begin() + body.size());
    std::copy(tag.begin(), tag.end(), result.begin() + body.size() + n.size());
    out = result; return true;
}

enum class nonce_result { recorded, replay, unavailable };
// MUST perform a cross-process atomic check-and-insert and persist it durably
// before returning recorded; scope by provisioned owner and key epoch. Never
// discard nonce history while that key is accepted. An uncertain commit must
// return unavailable. This abstract interface has no production implementation.
struct durable_nonce_guard {
    virtual ~durable_nonce_guard() = default;
    virtual nonce_result record(const rkva_registry_wire::id & owner, const nonce & value) = 0;
};
enum class result { accepted, invalid, replay, unavailable };
inline result verify(const secret & key, const rkva_registry_wire::id & provisioned_owner,
                     const rkva_request & h, const uint8_t * bytes, size_t len,
                     durable_nonce_guard & guard, rkva_registry_wire::request & out) {
    if (!valid_header(h) || !bytes || len != request_bytes || rkva_registry_wire::all_zero(key)) return result::invalid;
    rkva_registry_wire::request parsed{};
    if (!rkva_registry_wire::decode_request(h.op, bytes, rkva_registry_wire::request_bytes, parsed) ||
        !equal_tag(parsed.owner.data(), provisioned_owner.data())) return result::invalid;
    nonce n{}; std::copy(bytes + rkva_registry_wire::request_bytes,
                         bytes + rkva_registry_wire::request_bytes + n.size(), n.begin());
    digest expected{};
    if (!tag_request(key, h, bytes, n, expected) ||
        !equal_tag(expected.data(), bytes + rkva_registry_wire::request_bytes + n.size())) return result::invalid;
    try {
        const auto recorded = guard.record(parsed.owner, n);
        if (recorded == nonce_result::replay) return result::replay;
        if (recorded != nonce_result::recorded) return result::unavailable;
    } catch (...) { return result::unavailable; }
    out = parsed; return result::accepted;
}
} // namespace rkva_managed_auth
