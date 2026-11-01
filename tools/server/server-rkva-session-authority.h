#pragma once

// Phase 5 listener-side session authority (activation item 4), default off.
//
// A listener mints exactly one `authority` per accepted managed connection and
// binds it to:
//
//   * the concrete accepted socket (device/inode identity plus a listener
//     connection cookie) -- a closed, replaced or reused descriptor fails;
//   * the durable pre-HELLO challenge issued for that connection;
//   * the exact recursive mutex that serializes the session's managed state --
//     a contended mutex fails closed instead of waiting;
//   * the accepted HELLO ACK session (owner, ledger epoch, auth epoch,
//     transcript and current peer challenge).
//
// This is a test-seam implementation: there is no listener call site, no
// HELLO/capability/operation advertisement, and `available()` is false unless
// the activation build and the `RKVA_MANAGED_SNAPSHOT_ACTIVATION=1` runtime gate
// are both on. Enabling it still advertises nothing and dispatches nothing.

#include "server-rkva-host-journal.h"
#include "server-rkva-managed-activation.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace rkva_session_authority {

class authority final : public rkva_host_journal::v3_connection_authority {
public:
    struct binding {
        int accepted_fd = -1;            // listener-owned accepted connection
        std::uint64_t connection_cookie = 0; // non-zero listener accept identity
        std::string peer;                // stable peer description (addr:port)
        rkva_managed_framing_v3::nonce challenge{}; // issued pre-HELLO challenge
        std::recursive_mutex * state_mutex = nullptr; // exact session state mutex
    };

    ~authority() override;
    authority(const authority &) = delete;
    authority & operator=(const authority &) = delete;

    // Both the compile-time activation gate and the runtime opt-in must be on.
    static bool available();

    // Mint for one accepted connection. Fails closed on an unavailable gate, a
    // non-socket descriptor, a disconnected peer, a zero cookie or challenge,
    // or a missing state mutex.
    static std::shared_ptr<authority> mint(const binding & value, std::string & error);

    // Bind the accepted HELLO ACK session. The pre-HELLO challenge must have
    // been issued for this connection and the session must be authenticated.
    bool bind_session(const rkva_managed_framing_v3::session & session, std::string & error);

    // v3_connection_authority: true only while the connection identity and the
    // exact bound session still hold and the state mutex could be taken without
    // blocking. The mutex stays held until release()/invalidate()/destruction
    // so the publication that follows cannot race another writer.
    bool held_and_bound(const rkva_managed_framing_v3::session & session) override;

    void release();
    void invalidate();
    bool live() const;

private:
    struct impl;
    std::shared_ptr<impl> data_;
    authority();
};

} // namespace rkva_session_authority
