#pragma once
#include <array>
#include <cstdint>
#include <mutex>
#include <memory>
#include <string>
#include <vector>

namespace rkva_prepare_v3 { class concrete_guard; }
#include "rkva-managed-store-authority.h"

// Isolated backend experiment, not a network endpoint. Caller owns a private
// parent directory and provisions the secret and trusted host independently.
// No capability, garbage collection, v2 snapshot access or blob I/O exists here.
namespace rkva_anchored_sqlite {
using id = std::array<uint8_t,32>;
using secret = std::array<uint8_t,32>;
struct state { uint64_t version = 0; id digest{}; };
struct anchor { id owner{}, epoch{}, challenge{}; state accepted{}; id mac{}; };
enum class operation : uint8_t { prepare=1, commit=2, abort=3, release=4, status=5, hello_claim=6 };
struct command { operation op; id tx{}, reference{}, key{}, nonce{}; uint64_t commit_version=0; };
enum class result { ok, unavailable, invalid_anchor, stale_anchor, replay, denied };
// Canonical data must already be authenticated against the live session by
// the concrete friend guard. Reservation is conservative accounting, not an
// OS filesystem-space guarantee.
struct pending_request { id owner{}, tx{}, reference{}, key{}, nonce{}, identity{};
    std::array<uint8_t,80> semantic_hello{}; std::vector<uint8_t> metadata; uint64_t reservation=0; };
struct record { id tx{}, reference{}, key{}; uint32_t phase=0; uint64_t committed_version=0; };
// Caller must hold the v3 object's cross-process authority and establish
// durable object identity throughout this callback and transaction. No real
// Xbox implementation is supplied; null or false fails closed.
struct object_proof {
    virtual ~object_proof() = default;
    virtual bool verify(operation op, const record & value,
                        const std::shared_ptr<rkva_managed_store::lease> & authority) = 0;
};
// Host-side MAC encoding helper. A host must sign only a separately durable,
// authenticated anchor, never blindly sign a state supplied by this peer.
bool sign_anchor(const secret & key, anchor & value);
class backend {
public:
    backend(std::string private_parent, id owner, id epoch, secret anchor_key, uint64_t quota_bytes = (1ull << 40));
    ~backend();
    backend(const backend &) = delete;
    backend & operator=(const backend &) = delete;
    // One-time explicit provisioning; fails if the managed namespace exists.
    bool provision(std::shared_ptr<rkva_managed_store::lease> authority, std::string & error);
    // The locally generated random challenge is single-use; an absent/invalid
    // anchor or failed request consumes it. No fallback to old anchor/backup.
    bool challenge(std::shared_ptr<rkva_managed_store::lease> authority, id & out, std::string & error);
    result apply(std::shared_ptr<rkva_managed_store::lease> authority, const anchor * host, const command & cmd, state & out, std::string & error,
                 uint64_t * committed_version = nullptr, record * returned_record = nullptr,
                 object_proof * proof = nullptr);
    // Offline diagnostic only; this is NOT an authenticated host anchor.
    bool inspect(std::shared_ptr<rkva_managed_store::lease> authority, state & out, std::string & error);
#ifdef RKVA_ANCHORED_SQLITE_TESTING
    struct test_receipt { id owner{},epoch{},tx{},reference{},key{},object_digest{},nonce{},mac{}; state prior{},next{}; };
    struct test_status { test_receipt receipt{}; id challenge{},mac{}; };
    result test_commit(std::shared_ptr<rkva_managed_store::lease>,const anchor *,const command &,const id &,test_receipt &,std::string &);
    result test_status_read(std::shared_ptr<rkva_managed_store::lease>,const anchor *,const command &,const id &,test_status &,std::string &);
    static id test_sign_status(const secret &,const anchor &,const command &);
    static bool test_validate_status(const secret &,const anchor &,const command &,const test_status &);
    result test_prepare(std::shared_ptr<rkva_managed_store::lease> authority, const anchor * host,
                        const pending_request & request, state & out, std::string & error) {
        return prepare_pending(std::move(authority),host,request,out,error);
    }
#endif
private:
    friend class rkva_prepare_v3::concrete_guard;
    result prepare_pending(std::shared_ptr<rkva_managed_store::lease>, const anchor *,
                           const pending_request &, state &, std::string &);
    std::string parent_;
    id owner_{}, epoch_{}, challenge_{};
    secret key_{};
    uint64_t quota_bytes_ = 0;
    bool pending_ = false;
    std::weak_ptr<rkva_managed_store::lease> challenge_authority_;
    std::mutex mutex_;
};
#ifdef RKVA_ANCHORED_SQLITE_TESTING
extern void (*test_hook)(const char * stage);
#endif
}
