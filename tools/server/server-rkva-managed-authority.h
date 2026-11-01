#pragma once

// One RAII authority for the whole managed snapshot store (Phase 5, default
// off). Nothing in the server, listener, HELLO, capability or dispatch path
// references it.
//
// Before this layer the managed store had four independent locks:
//
//   * rkva_managed_store::lease            <parent>/.rkva-managed-store.lock
//     guards the anchored SQLite store, the v3 object writer/verifier and the
//     read-only v3 quota scan;
//   * rkva_host_journal::journal           <parent>/host-journal.lock
//   * rkva_managed_orchestrator            <parent>/.managed-orchestrator.lock
//   * the sidecar id/state mutexes owned by the caller.
//
// `store` acquires the three filesystem locks exactly once, in the documented
// order below, and keeps them for its whole lifetime. Components constructed
// from it run in *delegated* mode: the journal and the orchestrator never take
// a second, independent lock, and they revalidate this authority before every
// operation. The anchored SQLite backend and the v3 object writer/verifier
// receive the same concrete lease token through `concrete()`.
//
// Documented acquisition order (outermost first; never inverted):
//
//   1. <parent>/.rkva-managed-store.lock   (lease; SQLite + object v3 + quota)
//   2. <parent>/host-journal.lock          (host journal)
//   3. <parent>/.managed-orchestrator.lock (orchestrator)
//
// No component may acquire a lower-numbered lock while holding a higher one,
// and none may block. Acquisition is non-blocking and every contention, unsafe
// path, wrong owner, symlink, replaced inode or lost lease fails closed and
// releases everything already acquired.
//
// This is still not a durability or authentication proof: it does not make the
// three files one atomic transaction, it does not qualify rename/fsync across
// power loss (item 3), and it does not prove a pin or metadata (item 2). The
// journal publishes its own file atomically; a failure between the journal and
// SQLite publication leaves the transaction pinned/quarantined and charged.

#include "../../ggml/include/rkva-managed-store-authority.h"
#include "server-rkva-host-journal.h"

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

namespace rkva_managed_authority {

class store final : public std::enable_shared_from_this<store>,
                    public rkva_managed_store::delegation {
public:
    ~store() override;
    store(const store &) = delete;
    store & operator=(const store &) = delete;

    // Non-blocking acquisition of the three lock files in the documented order.
    // The parent must already exist, be owned by this UID and not be group or
    // world writable (the lease enforces this) and must not be inside the
    // legacy rkva-snapshots-v1/v2 namespaces.
    static std::shared_ptr<store> try_acquire(const std::filesystem::path & parent, std::string & error);

    // rkva_managed_store::delegation
    std::unique_lock<std::recursive_mutex> enter() const override;
    bool valid_for(const std::filesystem::path & parent) const override;
    const std::filesystem::path & directory() const override;
    std::shared_ptr<rkva_managed_store::lease> concrete() const override;

    // Convenience overloads that report the precise failure instead of an
    // empty lock. They behave exactly like the overrides above.
    std::unique_lock<std::recursive_mutex> enter(std::string & error) const;
    bool valid_for(const std::filesystem::path & parent, std::string & error) const;

    // Revalidates every held lock and inode against the filesystem.
    bool valid() const;

    // Delegated components. The returned objects keep this authority (and every
    // lock file descriptor) alive through their own shared ownership.
    rkva_host_journal::journal journal(const std::string & owner);

private:
    struct impl;
    std::shared_ptr<impl> data_;
    store();
};

} // namespace rkva_managed_authority
