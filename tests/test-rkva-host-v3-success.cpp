// Phase 5 authenticated successful COMMIT/STATUS reply (activation item 1) and
// receipt-based lost-ACK recovery (item 5). CPU-only; publication is exercised
// with a test-only proof double because no production pin/metadata or durable
// sync proof exists (items 2 and 3).
#include "../tools/server/server-rkva-host-v3-success.h"
#include "../tools/server/server-rkva-managed-authority.h"
#include "../tools/server/server-rkva-session-authority.h"
#undef NDEBUG
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace j = rkva_host_journal;
namespace v3 = rkva_managed_framing_v3;
namespace w = rkva_registry_wire;
namespace au = rkva_managed_authority;
namespace sa = rkva_session_authority;

static std::string crash_stage;
static void crash_hook(const char * stage) {
    if (crash_stage == stage) _exit(77);
}

static v3::id id(uint8_t n) { v3::id value{}; value.fill(n); return value; }
static std::string hex(const v3::id & value) { return w::to_hex(value); }

static rkva_request request(uint16_t op, uint64_t cycle, size_t size) {
    rkva_request h{};
    h.magic = RKVA_MAGIC;
    h.version = RKVA_VERSION_V2;
    h.op = op;
    h.cycle_id = cycle;
    h.session_id = RKVA_SESSION_NONE;
    h.payload_bytes = uint32_t(size);
    return h;
}

static rkva_response response(const rkva_request & q, size_t size) {
    rkva_response h{};
    h.magic = RKVA_MAGIC;
    h.version = RKVA_VERSION_V2;
    h.op = q.op;
    h.cycle_id = q.cycle_id;
    h.status = RKVA_OK;
    h.payload_bytes = uint32_t(size);
    h.xbox_compute_us = 9;
    return h;
}

struct test_proof final : j::v3_success_proof {
    v3::id expected_object{};
    bool accept = true;
    bool durable = true;
    int calls = 0;
    std::string refusal = "qualified durable sync/rename unavailable (item 3)";
    bool verified(const v3::receipt & value,
                  const std::shared_ptr<rkva_managed_store::lease> & authority,
                  std::string & why) override {
        ++calls;
        if (!authority || !authority->valid_for(authority->directory())) {
            why = "concrete store authority unavailable";
            return false;
        }
        if (!durable) { why = refusal; return false; }
        if (!accept || value.object_digest != expected_object) {
            why = "object proof does not match the receipt";
            return false;
        }
        return true;
    }
};

struct scenario {
    fs::path root;
    fs::path dir;
    std::shared_ptr<au::store> store;
    std::unique_ptr<j::journal> host;
    std::recursive_mutex mutex;
    int fds[2] = {-1, -1};
    std::shared_ptr<sa::authority> connection;
    v3::secret secret{};
    v3::id owner{}, ledger{}, tx{}, ref{}, key{}, h0{}, h1{}, object{}, identity{}, commit_nonce{};
    v3::session session{};
    v3::anchor anchor{};
    std::string error;

    explicit scenario(const fs::path & base) : root(base) {
        assert(fs::create_directories(root));
        dir = root;
        secret.fill(0xa5);
        owner = id(0x11);
        ledger = id(0x77);
        tx = id(0x22);
        ref = id(0x33);
        key = id(0x44);
        h0 = id(0x60);
        h1 = id(0x61);
        object = id(0x70);
        identity = id(0x71);
        commit_nonce = id(0x72);
        store = au::store::try_acquire(dir, error);
        assert(store);
        host = std::make_unique<j::journal>(store->journal(hex(owner)));
        assert(host->initialize(error));
        assert(host->provision_anchor(7, 3, h0, error));
        assert(host->configure_v3_epoch(7, ledger, 3, h0, error));
        assert(host->prepare(hex(tx), hex(ref), hex(key), error));
        assert(host->host_published(hex(tx), "/fixture-managed-host", std::string(64, 'c'), 4096, error));
        session = v3::session{owner, ledger, id(0x91), id(0x92), 7,
                              RKVA_PROTO_MANAGED_REGISTRY_V1, true};
        anchor.owner = owner;
        anchor.ledger_epoch = ledger;
        anchor.challenge = session.peer_challenge;
        anchor.version = 3;
        anchor.digest = h0;
        anchor.mac = id(0x93);
        assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
        sa::authority::binding binding;
        binding.accepted_fd = fds[0];
        binding.connection_cookie = 0x77;
        binding.peer = "fixture-peer";
        binding.challenge = id(0x92);
        binding.state_mutex = &mutex;
        connection = sa::authority::mint(binding, error);
        assert(connection);
        assert(connection->bind_session(session, error));
    }

    ~scenario() {
        if (fds[0] >= 0) close(fds[0]);
        if (fds[1] >= 0) close(fds[1]);
    }

    bool try_issue(uint16_t op, const w::request & query, uint64_t cycle, j::v3_issued_request & issued) {
        const auto builder = [this, op, query, cycle](const j::auth_id & nonce,
                                                      std::vector<uint8_t> & out) {
            w::request_wire body{};
            if (!w::encode_request(op, query, body)) return false;
            v3::fields fields;
            fields.type = v3::kind::command;
            fields.owner = session.owner;
            fields.auth_epoch = session.auth_epoch;
            fields.ledger_epoch = session.ledger_epoch;
            fields.transcript = session.transcript;
            fields.nonce = nonce;
            fields.host_anchor = anchor;
            fields.payload.assign(body.begin(), body.end());
            rkva_request header{};
            header.magic = RKVA_MAGIC;
            header.version = RKVA_VERSION_V2;
            header.op = op;
            header.cycle_id = cycle;
            header.session_id = RKVA_SESSION_NONE;
            header.payload_bytes = uint32_t(v3::prefix_bytes + body.size() + v3::tag_bytes);
            const auto header_bytes = v3::bytes(header);
            std::vector<uint8_t> sealed;
            if (!v3::seal(secret, header_bytes.data(), header_bytes.size(), fields, sealed)) return false;
            out.assign(header_bytes.begin(), header_bytes.end());
            out.insert(out.end(), sealed.begin(), sealed.end());
            return true;
        };
        return host->issue_v3_command(secret, session, connection.get(), builder, issued, error);
    }

    j::v3_issued_request issue(uint16_t op, const w::request & query, uint64_t cycle) {
        j::v3_issued_request issued;
        assert(try_issue(op, query, cycle, issued));
        return issued;
    }

    // A challenge is single-use, so a lost ACK is recovered on a fresh
    // authenticated session (new HELLO transcript and challenge) that is bound
    // to the same accepted connection authority.
    void reconnect(const v3::id & challenge) {
        session.transcript = id(0x95);
        session.peer_challenge = challenge;
        anchor.challenge = challenge;
        assert(connection->bind_session(session, error));
    }

    v3::acknowledgment ack(uint64_t registry_version, uint64_t committed_version,
                           const v3::id & successor_digest, const v3::id & next_challenge,
                           uint64_t previous_version, const v3::id & previous_digest,
                           uint64_t successor_version) {
        v3::acknowledgment value;
        value.record = w::response{w::phase::committed, registry_version, committed_version,
                                   owner, tx, ref, key};
        value.state_digest = successor_digest;
        value.next_challenge = next_challenge;
        value.has_receipt = true;
        auto & receipt = value.value;
        receipt.owner = owner;
        receipt.ledger_epoch = ledger;
        receipt.tx = tx;
        receipt.reference = ref;
        receipt.key = key;
        receipt.identity_digest = identity;
        receipt.object_digest = object;
        receipt.commit_nonce = commit_nonce;
        receipt.previous_version = previous_version;
        receipt.previous_digest = previous_digest;
        receipt.successor_version = successor_version;
        receipt.successor_digest = successor_digest;
        receipt.commit_version = committed_version;
        assert(j::sign_receipt_mac(secret, receipt));
        return value;
    }

    std::vector<uint8_t> reply(const rkva_request & header, const j::v3_issued_request & issued,
                               const rkva_response & outer, const v3::acknowledgment & value) {
        std::vector<uint8_t> sealed(issued.frame.begin() + 32, issued.frame.end());
        v3::session peer = session;
        std::vector<uint8_t> out;
        assert(v3::make_reply(secret, peer, header, sealed, outer, &value, out) == v3::status::accepted);
        return out;
    }

    void expect_pending() {
        j::snapshot state;
        assert(host->inspect(state, error));
        assert(state.anchor.version == 3 && state.anchor.digest == h0);
        assert(state.anchor.v3_nonces.size() == 1 && !state.anchor.v3_nonces[0].consumed);
        assert(state.transactions.size() == 1 && state.transactions[0].state == j::phase::host_published &&
               state.transactions[0].peer_commit_version == 0);
    }
};

int main() {
    alarm(240);
    assert(setenv("RKVA_MANAGED_SNAPSHOT_ACTIVATION", "1", 1) == 0);
    char tmp[] = "/tmp/rkva-host-v3-success-XXXXXX";
    char * name = mkdtemp(tmp);
    assert(name);
    const fs::path base(name);

    {
        scenario s(base / "main");
        const w::request commit_query{id(0x11), id(0x22), {}, {}, 0};
        const auto issued = s.issue(RKVA_REGISTRY_COMMIT, commit_query, 100);
        rkva_request header{};
        assert(v3::parse_request_header(issued.frame.data(), 32, header));
        const auto outer = response(header, v3::prefix_bytes + 220 + v3::receipt_bytes + v3::tag_bytes);
        const auto good_ack = s.ack(4, 4, s.h1, id(0x81), 3, s.h0, 4);
        const auto good_reply = s.reply(header, issued, outer, good_ack);
        test_proof proof;
        proof.expected_object = s.object;
        bool recovered = true;

        // No single store authority: refused before any durable change.
        {
            auto copy = s.session;
            assert(j::accept_v3_success(*s.host, s.secret, copy, s.connection.get(), nullptr, &proof,
                                        issued.frame, outer, good_reply.data(), good_reply.size(),
                                        recovered, s.error) == v3::status::unavailable);
            assert(s.error.find("single store authority") != std::string::npos);
            assert(!recovered);
        }
        // No pin/metadata proof: precise fail-closed, nothing published.
        {
            auto copy = s.session;
            assert(j::accept_v3_success(*s.host, s.secret, copy, s.connection.get(), s.store, nullptr,
                                        issued.frame, outer, good_reply.data(), good_reply.size(),
                                        recovered, s.error) == v3::status::unavailable);
            assert(s.error.find("pin and metadata proof") != std::string::npos);
        }
        // Durability proof unavailable: precise reason, nothing published.
        {
            auto weak = proof;
            weak.durable = false;
            auto copy = s.session;
            assert(j::accept_v3_success(*s.host, s.secret, copy, s.connection.get(), s.store, &weak,
                                        issued.frame, outer, good_reply.data(), good_reply.size(),
                                        recovered, s.error) == v3::status::unavailable);
            assert(s.error.find("durable sync/rename") != std::string::npos);
        }
        s.expect_pending();

        // Tampered receipt MAC with a recomputed outer tag (key holder).
        {
            auto forged = good_reply;
            forged[v3::prefix_bytes + 220 + 352] ^= 1;
            const auto head = v3::bytes(outer);
            v3::id tag{};
            assert(v3::body_mac(s.secret, "RKVA-V3-RESPONSE", head.data(), head.size(),
                                forged.data(), forged.size() - 32, tag));
            std::copy(tag.begin(), tag.end(), forged.end() - 32);
            auto copy = s.session;
            const auto result = j::accept_v3_success(*s.host, s.secret, copy, s.connection.get(), s.store,
                                                     &proof, issued.frame, outer, forged.data(), forged.size(),
                                                     recovered, s.error);
            assert(result != v3::status::accepted && s.error.find("receipt MAC") != std::string::npos);
            s.expect_pending();
        }
        // Wrong nonce: a re-sealed reply that names a nonce this journal never
        // issued is refused by the envelope before any proof runs.
        {
            auto forged = good_reply;
            forged[120] ^= 1; // sealed prefix command nonce
            const auto head = v3::bytes(outer);
            v3::id tag{};
            assert(v3::body_mac(s.secret, "RKVA-V3-RESPONSE", head.data(), head.size(),
                                forged.data(), forged.size() - 32, tag));
            std::copy(tag.begin(), tag.end(), forged.end() - 32);
            auto copy = s.session;
            assert(j::accept_v3_success(*s.host, s.secret, copy, s.connection.get(), s.store, &proof,
                                        issued.frame, outer, forged.data(), forged.size(), recovered,
                                        s.error) != v3::status::accepted);
            s.expect_pending();
        }
        // Receipt that does not match the durable anchor.
        {
            auto ack = s.ack(4, 4, s.h1, id(0x81), 3, id(0xee), 4);
            const auto forged = s.reply(header, issued, outer, ack);
            auto copy = s.session;
            assert(j::accept_v3_success(*s.host, s.secret, copy, s.connection.get(), s.store, &proof,
                                        issued.frame, outer, forged.data(), forged.size(), recovered,
                                        s.error) == v3::status::unavailable);
            assert(s.error.find("durable host anchor") != std::string::npos);
            s.expect_pending();
        }
        // Successor that skips a version.
        {
            auto ack = s.ack(5, 5, s.h1, id(0x81), 3, s.h0, 5);
            const auto forged = s.reply(header, issued, outer, ack);
            auto copy = s.session;
            assert(j::accept_v3_success(*s.host, s.secret, copy, s.connection.get(), s.store, &proof,
                                        issued.frame, outer, forged.data(), forged.size(), recovered,
                                        s.error) == v3::status::unavailable);
            assert(s.error.find("advance exactly one") != std::string::npos);
            s.expect_pending();
        }
        // Object proof rejects the exact receipt.
        {
            auto other = proof;
            other.expected_object = id(0x7f);
            auto copy = s.session;
            assert(j::accept_v3_success(*s.host, s.secret, copy, s.connection.get(), s.store, &other,
                                        issued.frame, outer, good_reply.data(), good_reply.size(),
                                        recovered, s.error) == v3::status::unavailable);
            assert(s.error.find("object proof") != std::string::npos);
            assert(other.calls == 1);
            s.expect_pending();
        }
        // Wrong session transcript: rejected by the authenticated envelope.
        {
            auto wrong = s.session;
            wrong.transcript = id(0x99);
            assert(j::accept_v3_success(*s.host, s.secret, wrong, s.connection.get(), s.store, &proof,
                                        issued.frame, outer, good_reply.data(), good_reply.size(),
                                        recovered, s.error) != v3::status::accepted);
            s.expect_pending();
        }
        // Valid success publishes exactly once.
        {
            auto copy = s.session;
            assert(j::accept_v3_success(*s.host, s.secret, copy, s.connection.get(), s.store, &proof,
                                        issued.frame, outer, good_reply.data(), good_reply.size(),
                                        recovered, s.error) == v3::status::accepted);
            assert(!recovered && copy.peer_challenge == id(0x81));
            j::snapshot state;
            assert(s.host->inspect(state, s.error));
            assert(state.anchor.version == 4 && state.anchor.digest == s.h1);
            assert(state.anchor.v3_nonces.size() == 1 && state.anchor.v3_nonces[0].consumed &&
                   !w::all_zero(state.anchor.v3_nonces[0].response_hash));
            assert(state.transactions[0].state == j::phase::commit_ack &&
                   state.transactions[0].peer_commit_version == 4);
        }
        // Replay of the same successful reply is refused and changes nothing.
        {
            auto copy = s.session;
            assert(j::accept_v3_success(*s.host, s.secret, copy, s.connection.get(), s.store, &proof,
                                        issued.frame, outer, good_reply.data(), good_reply.size(),
                                        recovered, s.error) == v3::status::replay);
            j::snapshot state;
            assert(s.host->inspect(state, s.error) && state.anchor.version == 4);
        }
        // Ledger-side binding: a different request hash or transaction cannot
        // consume the same record.
        {
            j::snapshot state;
            assert(s.host->inspect(state, s.error));
            const auto & nonce = state.anchor.v3_nonces[0].nonce;
            assert(!s.host->commit_v3_success(nonce, id(0xaa), id(0xbb), 4, 5, s.h1, hex(s.tx),
                                              hex(s.ref), hex(s.key), hex(s.owner), 7, s.error));
            assert(!s.host->commit_v3_success(nonce, state.anchor.v3_nonces[0].request_hash, id(0xbb),
                                              4, 4, s.h1, hex(s.tx), hex(s.ref), hex(s.key),
                                              hex(s.owner), 7, s.error));
            assert(s.host->inspect(state, s.error) && state.anchor.version == 4 &&
                   state.anchor.v3_nonces[0].consumed);
        }
    }

    // Lost ACK: the peer committed, the host never recorded the ACK. The
    // transaction stays pinned and ambiguous until an authenticated STATUS
    // receipt proves the commit.
    {
        scenario s(base / "lost");
        const w::request commit_query{id(0x11), id(0x22), {}, {}, 0};
        const auto commit_issued = s.issue(RKVA_REGISTRY_COMMIT, commit_query, 200);
        j::lost_ack_state lost;
        assert(j::lost_ack(*s.host, hex(s.tx), lost, s.error));
        assert(lost.found && lost.ambiguous && !lost.published && lost.pending_op == RKVA_REGISTRY_COMMIT);
        assert(lost.anchor_version == 3 && lost.anchor_digest == s.h0);
        assert(lost.reason.find("lost ACK") != std::string::npos);

        // A second COMMIT for the same transaction is refused: a fresh nonce
        // could advance the anchor twice for one peer commit.
        {
            j::v3_issued_request refused;
            assert(!s.try_issue(RKVA_REGISTRY_COMMIT, commit_query, 201, refused));
            assert(s.error.find("lost-ACK") != std::string::npos);
        }
        // Without a proof the STATUS recovery reply publishes nothing and the
        // transaction stays ambiguous.
        {
            s.reconnect(id(0x94));
            const w::request status_query{id(0x11), id(0x22), {}, {}, 0};
            j::v3_issued_request status_issued;
            assert(j::issue_status_recovery(*s.host, s.secret, s.session, s.connection.get(),
                                            s.anchor, status_query, 300, status_issued, s.error));
            rkva_request header{};
            assert(v3::parse_request_header(status_issued.frame.data(), 32, header));
            const auto outer = response(header, v3::prefix_bytes + 220 + v3::receipt_bytes + v3::tag_bytes);
            const auto ack = s.ack(4, 4, s.h1, id(0x82), 3, s.h0, 4);
            const auto valid = s.reply(header, status_issued, outer, ack);
            auto copy = s.session;
            bool recovered = false;
            assert(j::accept_v3_success(*s.host, s.secret, copy, s.connection.get(), s.store, nullptr,
                                        status_issued.frame, outer, valid.data(), valid.size(),
                                        recovered, s.error) == v3::status::unavailable);
            assert(s.error.find("pin and metadata proof") != std::string::npos);
            j::lost_ack_state after;
            assert(j::lost_ack(*s.host, hex(s.tx), after, s.error) && after.ambiguous &&
                   after.anchor_version == 3);
        }
    }

    // The recovery path itself, on a clean scenario.
    {
        scenario s(base / "recovery");
        const w::request commit_query{id(0x11), id(0x22), {}, {}, 0};
        (void) s.issue(RKVA_REGISTRY_COMMIT, commit_query, 400);
        s.reconnect(id(0x94));
        const w::request status_query{id(0x11), id(0x22), {}, {}, 0};
        j::v3_issued_request status_issued;
        assert(j::issue_status_recovery(*s.host, s.secret, s.session, s.connection.get(), s.anchor,
                                        status_query, 500, status_issued, s.error));
        rkva_request header{};
        assert(v3::parse_request_header(status_issued.frame.data(), 32, header));
        const auto outer = response(header, v3::prefix_bytes + 220 + v3::receipt_bytes + v3::tag_bytes);
        // Forged recovery receipt: previous anchor differs.
        {
            const auto ack = s.ack(4, 4, s.h1, id(0x82), 3, id(0xef), 4);
            const auto forged = s.reply(header, status_issued, outer, ack);
            test_proof proof;
            proof.expected_object = s.object;
            auto copy = s.session;
            bool recovered = false;
            assert(j::accept_v3_success(*s.host, s.secret, copy, s.connection.get(), s.store, &proof,
                                        status_issued.frame, outer, forged.data(), forged.size(),
                                        recovered, s.error) == v3::status::unavailable);
            assert(!recovered);
            j::lost_ack_state lost;
            assert(j::lost_ack(*s.host, hex(s.tx), lost, s.error) && lost.ambiguous);
        }
        // Valid recovery publishes the same commit exactly once.
        {
            const auto ack = s.ack(4, 4, s.h1, id(0x82), 3, s.h0, 4);
            const auto valid = s.reply(header, status_issued, outer, ack);
            test_proof proof;
            proof.expected_object = s.object;
            auto copy = s.session;
            bool recovered = false;
            assert(j::accept_v3_success(*s.host, s.secret, copy, s.connection.get(), s.store, &proof,
                                        status_issued.frame, outer, valid.data(), valid.size(),
                                        recovered, s.error) == v3::status::accepted);
            assert(recovered && copy.peer_challenge == id(0x82));
            j::lost_ack_state lost;
            assert(j::lost_ack(*s.host, hex(s.tx), lost, s.error));
            assert(lost.published && !lost.ambiguous && lost.anchor_version == 4);
            j::snapshot state;
            assert(s.host->inspect(state, s.error) && state.transactions[0].state == j::phase::commit_ack);
            auto replay = s.session;
            bool again = false;
            assert(j::accept_v3_success(*s.host, s.secret, replay, s.connection.get(), s.store, &proof,
                                        status_issued.frame, outer, valid.data(), valid.size(),
                                        again, s.error) == v3::status::replay);
        }
    }

    // Crash in the middle of the atomic success publication: a reopen sees the
    // complete old record or the complete new one, never a partial file.
    {
        scenario source(base / "crash-source");
        const w::request commit_query{id(0x11), id(0x22), {}, {}, 0};
        const auto issued = source.issue(RKVA_REGISTRY_COMMIT, commit_query, 600);
        j::snapshot state;
        assert(source.host->inspect(state, source.error));
        const auto nonce = state.anchor.v3_nonces[0].nonce;
        const auto request_hash = state.anchor.v3_nonces[0].request_hash;
        const auto response_hash = id(0xd1);
        for (const char * stage : {"before_rename", "after_rename"}) {
            const fs::path copy = base / (std::string("crash-") + stage);
            fs::copy(source.dir, copy, fs::copy_options::recursive);
            pid_t child = fork();
            assert(child >= 0);
            if (child == 0) {
                std::string why;
                auto store = au::store::try_acquire(copy, why);
                if (!store) _exit(90);
                auto journal = store->journal(hex(source.owner));
                crash_stage = stage;
                j::test_persist_hook = crash_hook;
                (void) journal.commit_v3_success(nonce, request_hash, response_hash, 4, 4,
                                                 source.h1, hex(source.tx), hex(source.ref),
                                                 hex(source.key), hex(source.owner), 7, why);
                _exit(91);
            }
            int status = 0;
            assert(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 77);
            std::string why;
            auto store = au::store::try_acquire(copy, why);
            assert(store);
            auto journal = store->journal(hex(source.owner));
            j::snapshot reopened;
            assert(journal.inspect(reopened, why));
            const bool published = reopened.anchor.version == 4;
            assert(published ? (reopened.anchor.digest == source.h1 &&
                                reopened.anchor.v3_nonces[0].consumed &&
                                reopened.transactions[0].state == j::phase::commit_ack)
                             : (reopened.anchor.digest == source.h0 &&
                                !reopened.anchor.v3_nonces[0].consumed &&
                                reopened.transactions[0].state == j::phase::host_published));
            // Retry is safe in both outcomes: idempotent publication or the
            // single missing publication. Never a second advance.
            assert(journal.commit_v3_success(nonce, request_hash, response_hash, 4, 4, source.h1,
                                             hex(source.tx), hex(source.ref), hex(source.key),
                                             hex(source.owner), 7, why));
            assert(journal.inspect(reopened, why) && reopened.anchor.version == 4 &&
                   reopened.anchor.v3_nonces[0].consumed);
        }
    }

    fs::remove_all(base);
    std::cout << "v3 success reply: receipt MAC, anchor successor, object proof, replay, wrong session, "
                 "lost-ACK STATUS recovery and crash: PASS\n";
    return 0;
}
