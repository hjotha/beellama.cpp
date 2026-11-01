#include "server-rkva-host-v3-success.h"

#include "../../ggml/include/rkva-snapshot-digest.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <utility>

namespace rkva_host_journal {
namespace v3 = rkva_managed_framing_v3;
namespace auth = rkva_managed_auth_e2e;
namespace wire = rkva_registry_wire;

namespace {

bool same_session(const v3::session & a, const v3::session & b) {
    return a.authenticated && b.authenticated && a.owner == b.owner &&
           a.ledger_epoch == b.ledger_epoch && a.transcript == b.transcript &&
           a.peer_challenge == b.peer_challenge && a.auth_epoch == b.auth_epoch;
}

// The receipt is canonical wire data; sign exactly the fields put_receipt
// serializes before the MAC (352 of its 384 bytes).
auth_id receipt_tag(const v3::secret & key, const v3::receipt & value) {
    auth_id out{};
    std::array<uint8_t, v3::receipt_bytes> body{};
    v3::receipt copy = value;
    copy.mac = {};
    v3::put_receipt(body.data(), copy);
    static constexpr char domain[] = "RKVA-V3-COMMIT-RECEIPT-V1";
    std::vector<uint8_t> input(domain, domain + sizeof(domain) - 1);
    input.insert(input.end(), body.begin(), body.begin() + (v3::receipt_bytes - 32));
    if (!rkva_managed_auth::hmac_sha256(key.data(), key.size(), input.data(), input.size(), out)) {
        return {};
    }
    return out;
}

struct success_guard final : v3::reply_ack_guard {
    success_guard(journal & host, const v3::secret & key, const v3::session & expected,
                  v3_connection_authority * connection,
                  const std::shared_ptr<rkva_managed_store::lease> & concrete,
                  v3_success_proof * proof, std::string & error, bool & recovered)
        : host_(host), key_(key), expected_(expected), connection_(connection),
          concrete_(concrete), proof_(proof), error_(error), recovered_(recovered) {}

    auth::guard_result commit(const v3::session & session, uint16_t op, const wire::request & query,
                              const v3::nonce & nonce, bool ok, const v3::acknowledgment & ack,
                              const rkva_request & header, const std::vector<uint8_t> & request,
                              const rkva_response & response, const uint8_t * reply, size_t bytes) override {
        (void) header;
        if (called_) { error_ = "reply guard called twice"; return auth::guard_result::unavailable; }
        called_ = true;
        if (!ok || response.status != RKVA_OK) {
            error_ = "authenticated successful reply required";
            return auth::guard_result::unavailable;
        }
        if (op != RKVA_REGISTRY_COMMIT && op != RKVA_REGISTRY_STATUS) {
            error_ = "only COMMIT and STATUS publish a receipt";
            return auth::guard_result::unavailable;
        }
        if (!connection_ || !connection_->held_and_bound(expected_)) {
            error_ = "listener session authority not held and bound";
            return auth::guard_result::unavailable;
        }
        if (!same_session(session, expected_)) {
            error_ = "reply session differs from the accepted session";
            return auth::guard_result::unavailable;
        }
        if (!ack.has_receipt || ack.record.state != wire::phase::committed) {
            error_ = "successful reply is not a committed receipt";
            return auth::guard_result::unavailable;
        }
        const auto & receipt = ack.value;
        if (ack.record.owner != session.owner || ack.record.tx != query.tx ||
            ack.record.committed_version != receipt.commit_version) {
            error_ = "acknowledgement record and receipt differ";
            return auth::guard_result::unavailable;
        }
        if (!valid_receipt_mac(key_, receipt)) {
            error_ = "receipt MAC invalid";
            return auth::guard_result::unavailable;
        }
        if (!v3::valid_receipt(receipt, session, ack.record, ack.state_digest)) {
            error_ = "receipt does not bind the replied commit";
            return auth::guard_result::unavailable;
        }
        snapshot state;
        std::string why;
        if (!host_.inspect(state, why)) {
            error_ = "host journal unavailable: " + why;
            return auth::guard_result::unavailable;
        }
        const auto tx = wire::to_hex(query.tx);
        const auto it = std::find_if(state.transactions.begin(), state.transactions.end(),
                                     [&](const transaction & t) { return t.tx == tx; });
        if (it == state.transactions.end()) { error_ = "unknown host transaction"; return auth::guard_result::unavailable; }
        // COMMIT and STATUS requests carry only the transaction id; the record
        // and the receipt carry the reference and key.
        const auto & reference = it->reference;
        const auto & key = it->key;
        if ((!wire::all_zero(query.reference) && wire::to_hex(query.reference) != reference) ||
            (!wire::all_zero(query.key) && wire::to_hex(query.key) != key) ||
            wire::to_hex(receipt.reference) != reference || wire::to_hex(receipt.key) != key) {
            error_ = "host transaction phase or binding differs";
            return auth::guard_result::unavailable;
        }
        if (it->state != phase::host_published && it->state != phase::commit_ack) {
            error_ = "host transaction phase or binding differs";
            return auth::guard_result::unavailable;
        }
        if (!state.anchor.configured || !state.anchor.v3_configured) {
            error_ = "authenticated v3 anchor not configured";
            return auth::guard_result::unavailable;
        }
        if (state.anchor.key_epoch != session.auth_epoch || state.anchor.ledger_epoch != session.ledger_epoch) {
            error_ = "journal epoch or ledger differs from the session";
            return auth::guard_result::unavailable;
        }
        const auto issued = std::find_if(state.anchor.v3_nonces.begin(), state.anchor.v3_nonces.end(),
                                         [&](const issued_v3_nonce & n) { return n.nonce == nonce; });
        if (issued == state.anchor.v3_nonces.end()) {
            error_ = "reply nonce was never issued by this journal";
            return auth::guard_result::replay;
        }
        if (issued->consumed) {
            error_ = "reply nonce already consumed";
            return auth::guard_result::replay;
        }
        if (receipt.previous_version != state.anchor.version || receipt.previous_digest != state.anchor.digest) {
            error_ = "receipt does not match the durable host anchor";
            return auth::guard_result::unavailable;
        }
        if (state.anchor.version == UINT64_MAX || receipt.successor_version != state.anchor.version + 1) {
            error_ = "successor anchor does not advance exactly one version";
            return auth::guard_result::unavailable;
        }
        if (receipt.successor_digest != ack.state_digest) {
            error_ = "successor anchor digest differs from the replied state";
            return auth::guard_result::unavailable;
        }
        if (issued->op != op || issued->tx != tx || issued->reference != reference || issued->key != key) {
            error_ = "issued v3 nonce binds a different operation or transaction";
            return auth::guard_result::unavailable;
        }
        if (issued->transcript != session.transcript || issued->challenge != session.peer_challenge) {
            error_ = "issued v3 nonce binds a different session or challenge";
            return auth::guard_result::unavailable;
        }
        // The journal persists the SHA-256 of the complete frame (32-byte outer
        // header plus the sealed body); the guard receives them separately.
        const auto header_bytes = v3::bytes(header);
        std::vector<uint8_t> full_request(header_bytes.begin(), header_bytes.end());
        full_request.insert(full_request.end(), request.begin(), request.end());
        const auto request_hash = rkva_snapshot::digest(full_request.data(), full_request.size());
        if (issued->request_hash != request_hash) {
            error_ = "reply request frame differs from the issued request";
            return auth::guard_result::unavailable;
        }
        if (!proof_) {
            error_ = "concrete authenticated pin and metadata proof unavailable (item 2)";
            return auth::guard_result::unavailable;
        }
        if (!concrete_) {
            error_ = "single store authority unavailable (item 6)";
            return auth::guard_result::unavailable;
        }
        why.clear();
        if (!proof_->verified(receipt, concrete_, why)) {
            error_ = why.empty() ? "object/pin proof refused the receipt (item 2)" : why;
            return auth::guard_result::unavailable;
        }
        const auto response_head = v3::bytes(response);
        rkva_snapshot::Sha256 hash;
        hash.update(response_head.data(), response_head.size());
        hash.update(reply, bytes);
        const auto response_hash = hash.finish();
        if (!host_.commit_v3_success(nonce, request_hash, response_hash, receipt.commit_version,
                                     receipt.successor_version, receipt.successor_digest, tx,
                                     reference, key, wire::to_hex(session.owner), session.auth_epoch, why)) {
            error_ = "v3 success publication refused: " + why;
            return auth::guard_result::unavailable;
        }
        recovered_ = op == RKVA_REGISTRY_STATUS;
        return auth::guard_result::recorded;
    }

    const std::string & reason() const { return error_; }

private:
    journal & host_;
    const v3::secret & key_;
    const v3::session & expected_;
    v3_connection_authority * connection_;
    std::shared_ptr<rkva_managed_store::lease> concrete_;
    v3_success_proof * proof_;
    std::string & error_;
    bool & recovered_;
    bool called_ = false;
};

} // namespace

bool sign_receipt_mac(const v3::secret & key, v3::receipt & value) {
    if (wire::all_zero(key)) return false;
    const auto tag = receipt_tag(key, value);
    if (wire::all_zero(tag)) return false;
    value.mac = tag;
    return true;
}

bool valid_receipt_mac(const v3::secret & key, const v3::receipt & value) {
    if (wire::all_zero(key) || wire::all_zero(value.mac)) return false;
    const auto tag = receipt_tag(key, value);
    return !wire::all_zero(tag) && rkva_managed_auth::equal_tag(tag.data(), value.mac.data());
}

v3::status accept_v3_success(journal & host, const v3::secret & key, v3::session & session,
                             v3_connection_authority * connection,
                             const std::shared_ptr<rkva_managed_store::delegation> & authority,
                             v3_success_proof * proof, const std::vector<uint8_t> & exact_request,
                             const rkva_response & response, const uint8_t * reply, size_t bytes,
                             bool & recovered, std::string & error) {
    error.clear();
    recovered = false;
    try {
        if (!authority) {
            session.authenticated = false;
            error = "single store authority unavailable (item 6)";
            return v3::status::unavailable;
        }
        if (!reply || exact_request.size() < 32 || exact_request.size() > 32 + v3::max_frame ||
            bytes > v3::max_frame) {
            session.authenticated = false;
            error = "v3 success context unavailable";
            return v3::status::unavailable;
        }
        if (response.status != RKVA_OK) {
            session.authenticated = false;
            error = "authenticated successful reply required";
            return v3::status::unavailable;
        }
        auto section = authority->enter();
        if (!section.owns_lock() || !authority->valid_for(authority->directory())) {
            session.authenticated = false;
            error = "single store authority changed or is fenced";
            return v3::status::unavailable;
        }
        rkva_request header{};
        if (!v3::parse_request_header(exact_request.data(), 32, header) ||
            header.payload_bytes != exact_request.size() - 32) {
            session.authenticated = false;
            error = "v3 success request framing invalid";
            return v3::status::invalid;
        }
        std::vector<uint8_t> body(exact_request.begin() + 32, exact_request.end());
        success_guard guard(host, key, session, connection, authority->concrete(), proof, error, recovered);
        v3::acknowledgment ignored;
        const auto result = v3::accept_reply(key, session, header, body, response, reply, bytes, &guard, ignored);
        if (result != v3::status::accepted) {
            session.authenticated = false;
            if (error.empty()) error = "authenticated success was not published";
        }
        return result;
    } catch (...) {
        session.authenticated = false;
        error = "v3 success acceptance unavailable";
        return v3::status::unavailable;
    }
}

bool lost_ack(journal & host, const std::string & tx, lost_ack_state & out, std::string & error) {
    error.clear();
    out = {};
    out.tx = tx;
    try {
        snapshot state;
        if (!host.inspect(state, error)) return false;
        const auto it = std::find_if(state.transactions.begin(), state.transactions.end(),
                                     [&](const transaction & t) { return t.tx == tx; });
        if (it == state.transactions.end()) { error = "unknown journal transaction"; return false; }
        out.found = true;
        out.state = it->state;
        out.published = it->state == phase::commit_ack || it->state == phase::release_intent ||
                        it->state == phase::release_ack;
        out.anchor_version = state.anchor.version;
        out.anchor_digest = state.anchor.digest;
        if (out.published) {
            out.reason = "commit acknowledged durably";
            return true;
        }
        for (const auto & n : state.anchor.v3_nonces) {
            if (n.consumed || n.tx != tx) continue;
            out.ambiguous = true;
            out.pending_op = n.op;
            const auto digest = rkva_snapshot::digest(n.nonce.data(), n.nonce.size());
            out.pending_nonce_hash = digest;
        }
        if (out.ambiguous) {
            out.reason = "lost ACK: peer commit is unknown; the transaction stays pinned and "
                         "charged and no anchor may be replaced without an authenticated STATUS receipt";
        } else if (it->state == phase::host_published) {
            out.reason = "host published without an issued v3 command; explicit reconciliation required";
        } else {
            out.reason = "not yet host published";
        }
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}

bool issue_status_recovery(journal & host, const v3::secret & key, const v3::session & session,
                           v3_connection_authority * connection, const v3::anchor & host_anchor,
                           const wire::request & query, uint64_t cycle_id, v3_issued_request & issued,
                           std::string & error) {
    error.clear();
    if (query.owner != session.owner) { error = "STATUS owner differs from the session"; return false; }
    const v3::secret secret = key;
    const v3::session bound = session;
    const v3::anchor anchor = host_anchor;
    const wire::request request = query;
    const v3_frame_builder builder = [secret, bound, anchor, request, cycle_id](
                                         const auth_id & nonce, std::vector<uint8_t> & frame) {
        wire::request_wire body{};
        if (!wire::encode_request(RKVA_REGISTRY_STATUS, request, body)) return false;
        v3::fields fields;
        fields.type = v3::kind::command;
        fields.owner = bound.owner;
        fields.auth_epoch = bound.auth_epoch;
        fields.ledger_epoch = bound.ledger_epoch;
        fields.transcript = bound.transcript;
        fields.nonce = nonce;
        fields.host_anchor = anchor;
        fields.payload.assign(body.begin(), body.end());
        rkva_request header{};
        header.magic = RKVA_MAGIC;
        header.version = RKVA_VERSION;
        header.op = RKVA_REGISTRY_STATUS;
        header.cycle_id = cycle_id;
        header.session_id = RKVA_SESSION_NONE;
        header.payload_bytes = uint32_t(v3::prefix_bytes + body.size() + v3::tag_bytes);
        const auto header_bytes = v3::bytes(header);
        std::vector<uint8_t> sealed;
        if (!v3::seal(secret, header_bytes.data(), header_bytes.size(), fields, sealed)) return false;
        frame.assign(header_bytes.begin(), header_bytes.end());
        frame.insert(frame.end(), sealed.begin(), sealed.end());
        return true;
    };
    return host.issue_v3_command(key, session, connection, builder, issued, error);
}

} // namespace rkva_host_journal
