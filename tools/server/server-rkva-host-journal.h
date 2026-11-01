#pragma once
#include <cstdint>
#include <array>
#include <string>
#include <vector>

// Standalone host transaction journal. Not connected to a server, RPC, host
// file management or GC. All paths and identities are supplied by trusted
// caller; SHA-256 is corruption detection, not authentication or rollback
// protection. Directory must exist and be exclusively administered.
namespace rkva_host_journal {
class committed_response_guard;
using auth_id = std::array<uint8_t,32>;
struct issued_nonce { std::string tx; auth_id nonce{}; bool consumed=false; };
struct authenticated_anchor {
    bool configured=false;
    uint64_t key_epoch=0, version=0;
    auth_id digest{};
    std::vector<issued_nonce> nonces;
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
    bool commit_verified(const std::string & tx, const auth_id & nonce, uint64_t peer_version,
                         uint64_t anchor_version, const auth_id & digest,
                         const std::string & reference, const std::string & key,
                         const std::string & owner, uint64_t epoch, std::string & error);
    std::string dir_, owner_;
};
} // namespace rkva_host_journal
