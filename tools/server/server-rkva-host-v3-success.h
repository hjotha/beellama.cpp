#pragma once

// Phase 5 host side of the authenticated *successful* COMMIT/STATUS reply
// (activation item 1) and receipt-based lost-ACK recovery (item 5). Default
// off, CPU tests only, no listener/RPC/dispatch call site.
//
// The reply guard verifies, in order:
//   1. the listener session authority is held and bound to this exact session
//      (item 4: accepted connection + challenge + state mutex);
//   2. the reply is an authenticated success for COMMIT or STATUS;
//   3. the acknowledgement is a committed record with a receipt whose canonical
//      host MAC verifies;
//   4. the receipt's previous anchor equals the durable host anchor exactly and
//      the successor advances it by exactly one version with the replied state
//      digest;
//   5. an independent object/pin proof accepts this exact receipt under the one
//      concrete store authority (items 2 and 3);
//   6. only then is commit_ack published atomically: the issued v3 nonce is
//      consumed with the authenticated response hash and the durable anchor is
//      advanced in one journal publication.
//
// There is no production `v3_success_proof` in this tree. A null proof, a
// refusal, or a missing single authority fails closed *before* any durable
// change with a precise reason. Nothing here fakes the pin/metadata proof or
// the qualified durable sync/rename, and nothing replaces an anchor
// automatically. A COMMIT retry after a lost ACK is refused by the journal; the
// only recovery is an authenticated STATUS reply carrying the receipt, which
// this same guard publishes once.

#include "../../ggml/include/rkva-managed-store-authority.h"
#include "server-rkva-host-journal.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rkva_host_journal {

// Canonical host MAC over the complete receipt (all fields except `mac`).
// Returns false when the key is all zero or on any internal failure.
bool sign_receipt_mac(const rkva_managed_framing_v3::secret & key,
                      rkva_managed_framing_v3::receipt & value);
bool valid_receipt_mac(const rkva_managed_framing_v3::secret & key,
                       const rkva_managed_framing_v3::receipt & value);

// Independent evidence for a successful COMMIT/STATUS reply. An implementation
// must prove, for this exact receipt and the same concrete lease token, that
// the pending pin and canonical metadata are durable and that the v3 object is
// durably published (items 2 and 3). A boolean fixture is not evidence and no
// implementation exists in this tree: production callers must pass nullptr and
// fail closed until the real UWP/store proof and qualified sync/rename exist.
struct v3_success_proof {
    virtual ~v3_success_proof() = default;
    virtual bool verified(const rkva_managed_framing_v3::receipt & value,
                          const std::shared_ptr<rkva_managed_store::lease> & authority,
                          std::string & error) = 0;
};

// Verify and publish one authenticated successful COMMIT (or STATUS recovery)
// reply. `authority` must be the single composed store authority; its section is
// held across verification and publication. On any failure the session is
// invalidated, `error` carries the precise reason and no durable state changes.
// `recovered` is set when the publication closed a lost ACK through STATUS.
rkva_managed_framing_v3::status accept_v3_success(
    journal & host, const rkva_managed_framing_v3::secret & key,
    rkva_managed_framing_v3::session & session, v3_connection_authority * connection,
    const std::shared_ptr<rkva_managed_store::delegation> & authority,
    v3_success_proof * proof, const std::vector<uint8_t> & exact_request,
    const rkva_response & response, const uint8_t * reply, size_t bytes,
    bool & recovered, std::string & error);

// Read-only lost-ACK classification. A transaction with an unconsumed v3
// COMMIT/STATUS nonce and no recorded commit_ack is ambiguous: the peer may or
// may not have committed. It stays pinned and charged, no anchor is replaced and
// no automatic cleanup follows. `published` means commit_ack is already durable.
struct lost_ack_state {
    std::string tx;
    bool found = false;
    bool published = false;
    bool ambiguous = false;
    phase state = phase::prepared;
    uint32_t pending_op = 0;
    auth_id pending_nonce_hash{};
    uint64_t anchor_version = 0;
    auth_id anchor_digest{};
    std::string reason;
};
bool lost_ack(journal & host, const std::string & tx, lost_ack_state & out, std::string & error);

// Build and durably issue the authenticated STATUS request used to recover a
// lost COMMIT ACK. The journal refuses a second COMMIT for the same transaction
// while an unconsumed COMMIT nonce exists, so this is the only retry path.
bool issue_status_recovery(journal & host, const rkva_managed_framing_v3::secret & key,
                           const rkva_managed_framing_v3::session & session,
                           v3_connection_authority * connection,
                           const rkva_managed_framing_v3::anchor & host_anchor,
                           const rkva_registry_wire::request & query, uint64_t cycle_id,
                           v3_issued_request & issued, std::string & error);

} // namespace rkva_host_journal
