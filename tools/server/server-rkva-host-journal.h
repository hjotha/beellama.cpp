#pragma once
#include <cstdint>
#include <array>
#include <functional>
#include <string>
#include <vector>
#include "../../ggml/src/ggml-remote-attn/rkva-managed-framing-v3.h"

// Standalone host transaction journal. Not connected to a server, RPC, host
// file management or GC. All paths and identities are supplied by trusted
// caller; SHA-256 is corruption detection, not authentication or rollback
// protection. Directory must exist and be exclusively administered.
namespace rkva_host_journal {
class committed_response_guard;
class v3_error_guard;
using auth_id = std::array<uint8_t,32>;
// Phase 4: the caller must hold a concrete socket/session lease bound to this
// peer and challenge. No production implementation is supplied in this tree.
struct v3_connection_authority {
    virtual ~v3_connection_authority() = default;
    virtual bool held_and_bound(const rkva_managed_framing_v3::session &) = 0;
};
struct v3_issued_request { auth_id nonce{}, request_hash{}; std::vector<uint8_t> frame; };
using v3_frame_builder = std::function<bool(const auth_id &, std::vector<uint8_t> &)>;
struct issued_nonce { std::string tx; auth_id nonce{}; bool consumed=false; };
// Phase 4 durable v3 request ledger. The full-frame SHA-256 is stored before
// the frame may leave the journal so a re-signed different request that reuses
// the nonce is rejected by the persisted hash, not only by the HMAC.
struct issued_v3_nonce {
    std::string tx, reference, key;
    uint32_t op=0;
    auth_id nonce{}, request_hash{}, transcript{}, challenge{}, anchor_digest{}, response_hash{};
    uint64_t anchor_version=0;
    bool consumed=false;
};
struct authenticated_anchor {
    bool configured=false;
    uint64_t key_epoch=0, version=0;
    auth_id digest{};
    std::vector<issued_nonce> nonces;
    bool v3_configured=false;
    auth_id ledger_epoch{};
    std::vector<issued_v3_nonce> v3_nonces;
};
enum class phase : uint8_t { prepared = 1, host_published, commit_ack, release_intent, release_ack, aborted };
struct transaction {
    std::string tx, reference, key;
    phase state = phase::prepared;
    std::string host_path, host_sha256;
    uint64_t host_bytes = 0, peer_commit_version = 0;
};
struct snapshot {
    uint64_t revision = 0;
    std::vector<transaction> transactions;
    authenticated_anchor anchor;
};
#ifdef RKVA_HOST_JOURNAL_TESTING
extern void (*test_persist_hook)(const char * stage);
#endif
// owner, tx, reference, key and host_sha256 are 64 lowercase hex digits.
// All operations reacquire the interprocess lock and reload the journal.
// Initialization is explicit; a missing/corrupt journal is never re-created
// by any other method. Methods never inspect, mutate or delete host files.
//
// Two modes share one file format (v2 only when an anchor is provisioned):
//  - unanchored: the plain prepare/host_published/commit_ack/release/abort
//    prototype used by the Phase 2 orchestrator;
//  - anchored: after provision_anchor(), a raw commit_ack and every release or
//    abort transition are refused. The only commit path is the atomic
//    commit_verified() publication that consumes a previously issued nonce. In
//    this mode the journal checksum is still corruption detection, not
//    authentication or rollback protection.
//
// Phase 4 adds a third, opt-in mode on top of an anchored journal: after
// configure_v3_epoch(), the journal persists the RKHJNL03 v3 nonce ledger and
// only issue_v3_command()/consume_v3_error() may touch v3 request nonces. The
// v2 commit-nonce path is then refused. Enabling v3 does not implement the
// authenticated successful COMMIT/ATTEST reply: no call site exists and the
// object/receipt proof it would need does not exist in this tree.
class journal {
public:
    journal(std::string directory, std::string owner);
    bool initialize(std::string & error);
    bool inspect(snapshot & result, std::string & error) const;
    bool prepare(const std::string & tx, const std::string & reference, const std::string & key, std::string & error);
    bool host_published(const std::string & tx, const std::string & path,
                        const std::string & whole_file_sha256, uint64_t bytes, std::string & error);
    bool commit_ack(const std::string & tx, uint64_t peer_version, std::string & error);
    // One-time provisioning from an independently trusted host anchor. Never
    // bootstrap by signing/accepting a state supplied only by the peer.
    bool provision_anchor(uint64_t key_epoch, uint64_t version, const auth_id & digest, std::string & error);
    // Record a fresh command nonce durably before sending. No automatic retry
    // or cleanup: a lost ACK leaves it issued until explicit reconciliation.
    bool issue_commit_nonce(const std::string & tx, const auth_id & nonce,
                            const std::string & owner, uint64_t epoch, std::string & error);
    // Explicit trusted owner/epoch mapping bound to the pre-existing anchor.
    bool configure_v3_epoch(uint64_t auth_epoch, const auth_id & ledger_epoch,
                            uint64_t anchor_version, const auth_id & anchor_digest, std::string & error);
    // Builds locally under lock with a fresh CSPRNG nonce, verifies the entire
    // V3 request/HMAC and the concrete connection authority, and durably stores
    // the complete-frame hash before returning bytes. On any failure after the
    // ledger insert the nonce stays pending; output is never partially written.
    bool issue_v3_command(const rkva_managed_framing_v3::secret & secret,
                          const rkva_managed_framing_v3::session & session,
                          v3_connection_authority * connection, const v3_frame_builder & builder,
                          v3_issued_request & output, std::string & error);
    bool release_intent(const std::string & tx, uint64_t peer_version, std::string & error);
    bool release_ack(const std::string & tx, uint64_t peer_version, std::string & error);
    // Caller must have independently proved that the host file was never
    // published. Journal state alone cannot prove this across a crash.
    bool abort_unpublished(const std::string & tx, std::string & error);
#ifdef RKVA_HOST_JOURNAL_TESTING
    // Test-only direct access to the atomic verified-COMMIT publication. The
    // production caller is the authenticated COMMIT reply guard (Phase 4),
    // which is the only component allowed to establish the peer version and
    // successor anchor. Do not build a shipping target with this hook defined.
    bool test_commit_verified(const std::string & tx, const auth_id & nonce, uint64_t peer_version,
                              uint64_t anchor_version, const auth_id & digest,
                              const std::string & reference, const std::string & key,
                              const std::string & owner, uint64_t epoch, std::string & error) {
        return commit_verified(tx, nonce, peer_version, anchor_version, digest, reference, key,
                               owner, epoch, error);
    }
#endif
private:
    friend class committed_response_guard;
    friend class v3_error_guard;
    bool consume_v3_error(const auth_id & nonce, const auth_id & request_hash, const auth_id & response_hash,
                          const std::string & owner, uint64_t epoch, const auth_id & ledger_epoch, std::string & error);
    bool commit_verified(const std::string & tx, const auth_id & nonce, uint64_t peer_version,
                         uint64_t anchor_version, const auth_id & digest,
                         const std::string & reference, const std::string & key,
                         const std::string & owner, uint64_t epoch, std::string & error);
    std::string dir_, owner_;
};
} // namespace rkva_host_journal
