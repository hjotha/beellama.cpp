// Phase 3 anchored PREPARE/COMMIT/receipt state machine (CPU only).
//
// This exercises the real SQLite pending-pin backend behind the Phase 3 gate:
// challenge/anchor admission, canonical metadata validation, conservative quota
// reservation, durable nonce admission, the test-only MACed receipt and STATUS
// transition, crash injection around the receipt commit, and the deliberately
// fail-closed managed `apply` path.
//
// It is NOT an object verifier, network handler, UWP or power-loss test. The
// `fixture-object` file is not an RKVA v3 object and its digest is not the host
// prefix key. No callback in this file may authorize production publication.

#include "../ggml/include/rkva-anchored-sqlite.h"
#include "../ggml/include/rkva-native-snapshot.h"
#undef NDEBUG
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sqlite3.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace db = rkva_anchored_sqlite;
namespace snap = rkva_native_snapshot;
namespace fs = std::filesystem;

static fs::path scratch() {
    char pattern[] = "/tmp/rkva-phase3-receipt-XXXXXX";
    char * name = mkdtemp(pattern);
    assert(name != nullptr);
    return fs::path(name);
}
static void write_file(const fs::path & path, const std::string & data) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(data.data(), std::streamsize(data.size()));
    assert(output.good());
}
static std::string payload() { return std::string("synthetic phase3 object bytes"); }
static db::id object_digest() {
    const auto data = payload();
    rkva_snapshot::Sha256 hash;
    hash.update(data.data(), data.size());
    return hash.finish();
}
static db::id fill_id(uint8_t value) {
    db::id out{};
    out.fill(value);
    return out;
}
struct manifest_fixture {
    std::vector<uint8_t> canonical;
    uint64_t payload_bytes = 0;
};

// Canonical metadata for a synthetic resident snapshot. Only the manifest
// binding (host_state_sha256 == key) and canonical encoding are used here.
static manifest_fixture make_metadata(const db::id & key) {
    const snap::Geometry geometry{8, 2, 256, 2, 4096, 1};
    const snap::Descriptor descriptor{1024, 1024, RKVA_RANGE_STORAGE_KVARN4, 3, 128, 128,
                                      256, 2, 4, 4, 8960, 3, 2, 128, 128, 1, 1, 1};
    const snap::LayerCoverage coverage{385, 384, 4, 1};
    snap::Manifest manifest;
    std::string why;
    assert(snap::build(geometry, descriptor, 513, 0x700000001ull, key.data(), {coverage},
                       manifest, &why));
    manifest_fixture out;
    assert(snap::encode(manifest, out.canonical, &why));
    out.payload_bytes = manifest.payload_bytes;
    return out;
}
static std::array<uint8_t, 80> semantic_hello(uint8_t base) {
    std::array<uint8_t, 80> hello{};
    for (size_t i = 0; i < hello.size(); ++i) hello[i] = uint8_t(base + i);
    return hello;
}
static db::id identity(const db::id & owner, const db::id & tx, const db::id & reference,
                       const db::id & key, const std::array<uint8_t, 80> & hello,
                       const std::vector<uint8_t> & metadata) {
    static constexpr char domain[] = "RKVA-PREPARE-V3-METADATA-V1";
    rkva_snapshot::Sha256 sha;
    sha.update(domain, sizeof(domain) - 1);
    sha.update(owner.data(), 32); sha.update(tx.data(), 32);
    sha.update(reference.data(), 32); sha.update(key.data(), 32);
    sha.update(hello.data(), hello.size());
    sha.update(metadata.data(), metadata.size());
    return sha.finish();
}
static db::anchor issue_anchor(db::backend & store,
                               const std::shared_ptr<rkva_managed_store::lease> & token,
                               const db::id & owner, const db::id & epoch, const db::secret & key,
                               const db::state & accepted, std::string & error) {
    db::anchor anchor;
    anchor.owner = owner;
    anchor.epoch = epoch;
    anchor.accepted = accepted;
    assert(store.challenge(token, anchor.challenge, error));
    assert(db::sign_anchor(key, anchor));
    return anchor;
}
static db::pending_request make_request(const db::id & owner, const db::id & tx,
                                        const db::id & reference, const db::id & key,
                                        uint8_t nonce_byte, const std::vector<uint8_t> & metadata,
                                        uint64_t payload_bytes, uint64_t extra = 0) {
    db::pending_request request;
    request.owner = owner; request.tx = tx; request.reference = reference; request.key = key;
    request.nonce = fill_id(nonce_byte);
    request.semantic_hello = semantic_hello(nonce_byte);
    request.metadata = metadata;
    request.identity = identity(owner, tx, reference, key, request.semantic_hello, metadata);
    request.reservation = payload_bytes + metadata.size() + (1ull << 20) + extra;
    return request;
}

#ifndef _WIN32
struct true_object_proof final : db::object_proof {
    mutable int calls = 0;
    bool verify(db::operation, const db::record &,
                const std::shared_ptr<rkva_managed_store::lease> &) override {
        ++calls;
        return true;
    }
};
static std::string crash_stage;
static void crash_hook(const char * stage) {
    if (crash_stage == stage) _exit(73);
}
#endif

int main() {
#ifdef _WIN32
    std::cout << "POSIX SQLite/lease fixture skipped\n";
    return 0;
#else
    const auto root = scratch();
    std::string error;
    const auto owner = fill_id(0x11), epoch = fill_id(0x22), secret_seed = fill_id(0x33);
    db::secret secret = secret_seed;
    const auto tx = fill_id(0x44), reference = fill_id(0x55), key = fill_id(0x66);
    const auto metadata = make_metadata(key);

    // Provisioning needs a real lease bound to this exact private parent.
    {
        db::backend store(root.string(), owner, epoch, secret);
        assert(!store.provision({}, error));
        assert(!fs::exists(root / "rkva-managed-v3"));
        const auto other = scratch(); fs::create_directory(other);
        auto unrelated = rkva_managed_store::lease::try_acquire(other, error);
        assert(unrelated);
        assert(!store.provision(unrelated, error));
        assert(!fs::exists(root / "rkva-managed-v3"));
        db::state untouched{};
        untouched.version = 999;
        assert(!store.inspect({}, untouched, error) && untouched.version == 999);
        fs::remove_all(other);
    }
    auto token = rkva_managed_store::lease::try_acquire(root, error);
    assert(token);
    db::backend store(root.string(), owner, epoch, secret);
    assert(store.provision(token, error));
    assert(!store.provision(token, error)); // provisioned once
    db::state state{};
    assert(store.inspect(token, state, error) && state.version == 0);

    // The challenge and the anchor must come from this exact live lease.
    db::id no_challenge{};
    assert(!store.challenge(nullptr, no_challenge, error));
    db::pending_request pending = make_request(owner, tx, reference, key, 0x71, metadata.canonical,
                                               metadata.payload_bytes);
    {
        db::state out{};
        assert(store.test_prepare(token, nullptr, pending, out, error) == db::result::invalid_anchor);
        // No challenge was issued, so a correctly signed anchor is still refused.
        db::anchor unsigned_anchor;
        unsigned_anchor.owner = owner; unsigned_anchor.epoch = epoch; unsigned_anchor.accepted = state;
        assert(!db::sign_anchor(secret, unsigned_anchor)); // zero challenge
        assert(store.test_prepare(token, &unsigned_anchor, pending, out, error) != db::result::ok);
    }
    // Canonical metadata, key binding, reservation and identity digest are all
    // mandatory; each failure leaves the store unchanged.
    {
        auto broken = pending; broken.key = fill_id(0x99);
        auto anchor = issue_anchor(store, token, owner, epoch, secret, state, error);
        db::state out{};
        assert(store.test_prepare(token, &anchor, broken, out, error) != db::result::ok);
        assert(error.find("invalid canonical metadata") != std::string::npos);
        auto cheap = pending; cheap.reservation = metadata.payload_bytes + metadata.canonical.size();
        anchor = issue_anchor(store, token, owner, epoch, secret, state, error);
        assert(store.test_prepare(token, &anchor, cheap, out, error) != db::result::ok);
        assert(error.find("invalid reservation") != std::string::npos);
        auto forged = pending; forged.identity = fill_id(0x77);
        anchor = issue_anchor(store, token, owner, epoch, secret, state, error);
        assert(store.test_prepare(token, &anchor, forged, out, error) != db::result::ok);
        assert(error.find("identity digest mismatch") != std::string::npos);
        db::state unchanged{};
        assert(store.inspect(token, unchanged, error) && unchanged.version == 0);
        assert(!fs::exists(root / "rkva-managed-v3" / "receipts.db"));
    }
    // PREPARE persists exactly one pending pin and one reservation.
    {
        auto anchor = issue_anchor(store, token, owner, epoch, secret, state, error);
        db::state out{};
        assert(store.test_prepare(token, &anchor, pending, out, error) == db::result::ok);
        assert(out.version == state.version + 1);
        state = out;
    }
    // Nonce replay is refused even with a fresh challenge and anchor.
    {
        auto anchor = issue_anchor(store, token, owner, epoch, secret, state, error);
        db::state out{};
        assert(store.test_prepare(token, &anchor, pending, out, error) == db::result::replay);
    }
    // A retry of the same reference/metadata/reservation needs a fresh nonce
    // and is idempotent for the pin and reservation.
    {
        auto retry = pending; retry.nonce = fill_id(0x72);
        auto anchor = issue_anchor(store, token, owner, epoch, secret, state, error);
        db::state out{};
        assert(store.test_prepare(token, &anchor, retry, out, error) == db::result::ok);
        assert(out.version == state.version + 1);
        state = out;
    }
    // A different identity for the same transaction is a conflict, not a
    // replacement of the pending pin.
    {
        auto conflict = make_request(owner, tx, fill_id(0x56), fill_id(0x67), 0x73,
                                     make_metadata(fill_id(0x67)).canonical, metadata.payload_bytes);
        auto anchor = issue_anchor(store, token, owner, epoch, secret, state, error);
        db::state out{};
        assert(store.test_prepare(token, &anchor, conflict, out, error) != db::result::ok);
        db::state unchanged{};
        assert(store.inspect(token, unchanged, error) && unchanged.version == state.version);
    }

    // Conservative quota: the second pending reservation is refused without
    // evicting or shrinking the first one.
    {
        const auto quota_dir = scratch();
        auto quota_token = rkva_managed_store::lease::try_acquire(quota_dir, error);
        assert(quota_token);
        const uint64_t quota = metadata.payload_bytes + metadata.canonical.size() + (1ull << 20);
        db::backend limited(quota_dir.string(), owner, epoch, secret, quota);
        assert(limited.provision(quota_token, error));
        db::state zero{}, out{};
        assert(limited.inspect(quota_token, zero, error));
        auto anchor = issue_anchor(limited, quota_token, owner, epoch, secret, zero, error);
        assert(limited.test_prepare(quota_token, &anchor, pending, out, error) == db::result::ok);
        auto other_tx = fill_id(0x88), other_ref = fill_id(0x89), other_key = fill_id(0x8a);
        auto other_meta = make_metadata(other_key);
        auto second = make_request(owner, other_tx, other_ref, other_key, 0x74, other_meta.canonical,
                                   other_meta.payload_bytes);
        anchor = issue_anchor(limited, quota_token, owner, epoch, secret, out, error);
        db::state ignored{};
        assert(limited.test_prepare(quota_token, &anchor, second, ignored, error) != db::result::ok);
        db::state unchanged{};
        assert(limited.inspect(quota_token, unchanged, error) && unchanged.version == out.version);
        quota_token.reset();
        fs::remove_all(quota_dir);
    }

    // Test-only COMMIT: it consumes the nonce, transitions the exact pin and
    // stores the MACed receipt in the same SQLite commit. The fixture file is
    // not an RKVA object and cannot authorize a production transition.
    db::state predecessor{};
    {
        write_file(root / "fixture-object", payload());
        auto anchor = issue_anchor(store, token, owner, epoch, secret, state, error);
        db::command commit{db::operation::commit};
        commit.tx = tx; commit.reference = reference; commit.key = key;
        commit.nonce = fill_id(0x75);
        db::backend::test_receipt receipt;
        assert(store.test_commit(token, &anchor, commit, object_digest(), receipt, error) == db::result::ok);
        assert(receipt.tx == tx && receipt.reference == reference && receipt.key == key);
        assert(receipt.prior.version == state.version && receipt.next.version == state.version + 1);
        assert(receipt.object_digest == object_digest());
        db::state committed{};
        assert(store.inspect(token, committed, error) && committed.version == receipt.next.version);
        // STATUS binds the receipt to its immediate predecessor anchor, so the
        // pre-COMMIT state is kept alongside the advanced store state.
        predecessor = state;
        state = committed;
        // Replay needs a fresh anchor and is refused. The committed pin is
        // checked before the nonce ledger, so the duplicate receipt nonce is
        // rejected as a pin mismatch rather than reported as a replay.
        anchor = issue_anchor(store, token, owner, epoch, secret, state, error);
        db::backend::test_receipt replay{};
        assert(store.test_commit(token, &anchor, commit, object_digest(), replay, error) != db::result::ok);
        // A different object under the same receipt is denied.
        auto wrong = commit; wrong.nonce = fill_id(0x76);
        anchor = issue_anchor(store, token, owner, epoch, secret, state, error);
        assert(store.test_commit(token, &anchor, wrong, fill_id(0x99), replay, error) == db::result::denied);
        // Corrupting the fixture bytes is detected before the commit.
        write_file(root / "fixture-object", payload() + "tampered");
        anchor = issue_anchor(store, token, owner, epoch, secret, state, error);
        assert(store.test_commit(token, &anchor, wrong, object_digest(), replay, error) == db::result::denied);
        write_file(root / "fixture-object", payload());
        db::state unchanged{};
        assert(store.inspect(token, unchanged, error) && unchanged.version == state.version);
    }

    // Test-only STATUS: fresh challenge, request MAC, exact predecessor and
    // receipt binding. It never mutates SQLite.
    {
        db::command query{db::operation::status};
        query.tx = tx; query.reference = reference; query.key = key;
        auto anchor = issue_anchor(store, token, owner, epoch, secret, predecessor, error);
        db::backend::test_status reply;
        assert(store.test_status_read(token, &anchor, query, db::backend::test_sign_status(secret, anchor, query), reply, error) == db::result::ok);
        assert(db::backend::test_validate_status(secret, anchor, query, reply));
        assert(reply.receipt.next.digest == state.digest && reply.receipt.prior.digest == predecessor.digest);
        // The challenge is single-use.
        assert(store.test_status_read(token, &anchor, query, db::backend::test_sign_status(secret, anchor, query), reply, error) == db::result::invalid_anchor);
        // A tampered reply cannot be validated, and a wrong request MAC is denied.
        auto bad = reply; bad.receipt.next.digest[0] ^= 1;
        assert(!db::backend::test_validate_status(secret, anchor, query, bad));
        auto fresh = issue_anchor(store, token, owner, epoch, secret, predecessor, error);
        auto bad_mac = db::backend::test_sign_status(secret, fresh, query); bad_mac[0] ^= 1;
        db::backend::test_status rejected{};
        assert(store.test_status_read(token, &fresh, query, bad_mac, rejected, error) == db::result::denied);
        // Tampering the stored receipt must fail STATUS closed.
        sqlite3 * raw = nullptr;
        const auto path = root / "rkva-managed-v3" / "peer.db";
        assert(sqlite3_open(path.string().c_str(), &raw) == SQLITE_OK);
        assert(sqlite3_exec(raw, "UPDATE receipts SET mac=zeroblob(32)", nullptr, nullptr, nullptr) == SQLITE_OK);
        sqlite3_close(raw);
        fresh = issue_anchor(store, token, owner, epoch, secret, predecessor, error);
        assert(store.test_status_read(token, &fresh, query, db::backend::test_sign_status(secret, fresh, query), rejected, error) == db::result::unavailable);
    }

    // The managed apply path stays unconditionally fail-closed: a backing
    // callback that always returns true cannot authorize a transition, and it
    // is never even called.
    {
        true_object_proof proof;
        db::command prepare{db::operation::prepare};
        prepare.tx = fill_id(0x91); prepare.reference = fill_id(0x92); prepare.key = fill_id(0x93);
        prepare.nonce = fill_id(0x94);
        for (const auto op : {db::operation::prepare, db::operation::commit, db::operation::abort,
                              db::operation::release, db::operation::status}) {
            auto anchor = issue_anchor(store, token, owner, epoch, secret, state, error);
            auto command = prepare; command.op = op;
            db::state out{};
            assert(store.apply(token, &anchor, command, out, error, nullptr, nullptr, &proof) == db::result::denied);
            assert(error.find("concrete authenticated pin and metadata proof unavailable") != std::string::npos);
        }
        assert(proof.calls == 0);
        // hello_claim is the only admitted operation and still consumes its
        // nonce durably.
        auto anchor = issue_anchor(store, token, owner, epoch, secret, state, error);
        db::command claim{db::operation::hello_claim};
        claim.nonce = fill_id(0x95);
        db::state out{};
        assert(store.apply(token, &anchor, claim, out, error) == db::result::ok);
        assert(out.version == state.version + 1);
        state = out;
        anchor = issue_anchor(store, token, owner, epoch, secret, state, error);
        assert(store.apply(token, &anchor, claim, out, error) == db::result::replay);
    }

    // A replaced lock pathname invalidates the authority before SQLite opens.
    {
        fs::rename(root / ".rkva-managed-store.lock", root / "old-lock");
        write_file(root / ".rkva-managed-store.lock", "replacement");
        db::state ignored{};
        assert(!token->valid_for(root));
        assert(!store.inspect(token, ignored, error));
    }
    token.reset();

    // Crash injection: a process death around the single receipt COMMIT leaves
    // either an ambiguous hot journal (never auto-recovered) or the committed
    // receipt. The fixture does not claim power-loss durability.
    for (const char * stage : {"receipt_before_commit", "receipt_after_commit"}) {
        const auto dir = scratch();
        write_file(dir / "fixture-object", payload());
        auto crash_token = rkva_managed_store::lease::try_acquire(dir, error);
        assert(crash_token);
        const auto crash_owner = fill_id(0x62), crash_epoch = fill_id(0x63);
        const auto crash_tx = fill_id(0x64), crash_ref = fill_id(0x65), crash_object = fill_id(0x66);
        db::secret crash_secret = fill_id(0x67);
        const auto crash_meta = make_metadata(crash_object);
        db::backend crash_store(dir.string(), crash_owner, crash_epoch, crash_secret);
        assert(crash_store.provision(crash_token, error));
        db::state zero{}, prepared{};
        assert(crash_store.inspect(crash_token, zero, error));
        auto request = make_request(crash_owner, crash_tx, crash_ref, crash_object, 0x68,
                                    crash_meta.canonical, crash_meta.payload_bytes);
        auto anchor = issue_anchor(crash_store, crash_token, crash_owner, crash_epoch, crash_secret, zero, error);
        assert(crash_store.test_prepare(crash_token, &anchor, request, prepared, error) == db::result::ok);
        const auto live = dir / "rkva-managed-v3" / "peer.db";
        const auto precommit = dir / "fixture-precommit.db";
        fs::copy_file(live, precommit);
        crash_token.reset();
        pid_t pid = fork();
        assert(pid >= 0);
        if (pid == 0) {
            auto child_token = rkva_managed_store::lease::try_acquire(dir, error);
            if (!child_token) _exit(20);
            db::backend child(dir.string(), crash_owner, crash_epoch, crash_secret);
            auto child_anchor = issue_anchor(child, child_token, crash_owner, crash_epoch, crash_secret, prepared, error);
            db::command commit{db::operation::commit};
            commit.tx = crash_tx; commit.reference = crash_ref; commit.key = crash_object;
            commit.nonce = fill_id(0x69);
            crash_stage = stage; db::test_hook = crash_hook;
            db::backend::test_receipt receipt;
            (void) child.test_commit(child_token, &child_anchor, commit, object_digest(), receipt, error);
            _exit(22);
        }
        int status = 0;
        assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 73);
        db::test_hook = nullptr;
        crash_token = rkva_managed_store::lease::try_acquire(dir, error);
        assert(crash_token);
        db::backend recovered(dir.string(), crash_owner, crash_epoch, crash_secret);
        db::command query{db::operation::status};
        query.tx = crash_tx; query.reference = crash_ref; query.key = crash_object;
        auto fresh = issue_anchor(recovered, crash_token, crash_owner, crash_epoch, crash_secret, prepared, error);
        db::backend::test_status reply;
        if (std::string(stage) == "receipt_before_commit") {
            assert(recovered.test_status_read(crash_token, &fresh, query, db::backend::test_sign_status(crash_secret, fresh, query), reply, error) == db::result::unavailable);
            assert(error.find("ambiguous managed store file") != std::string::npos);
        } else {
            assert(recovered.test_status_read(crash_token, &fresh, query, db::backend::test_sign_status(crash_secret, fresh, query), reply, error) == db::result::ok);
            assert(db::backend::test_validate_status(crash_secret, fresh, query, reply));
            assert(reply.receipt.prior.digest == prepared.digest);
            assert(reply.receipt.next.version == prepared.version + 1);
        }
        crash_token.reset();
        fs::remove_all(dir);
    }
    fs::remove_all(root);
    std::cout << "anchored PREPARE/COMMIT/receipt, quota, crash and fail-closed apply: PASS\n";
    return 0;
#endif
}
