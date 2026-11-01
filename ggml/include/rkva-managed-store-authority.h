#pragma once
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

// Experimental authority for a trusted, private managed store.
// Every SQLite operation and every v3 object writer/reader must retain this
// exact concrete token. No network authentication, pin proof, GC, production
// or UWP durability is implied. Non-cooperating writers are outside scope.
// On Windows the caller must control the parent and all ancestors; the lock
// does not establish ownership/ACLs or defend against non-cooperating processes.
namespace rkva_managed_store {
class lease;
// Proof that ONE already-held authority covers a whole managed directory and
// every lock inside it. A component constructed with a delegation must never
// take its own lock; it enters this authority for the whole operation and
// refuses to proceed when the authority was lost, replaced or forked. The
// concrete implementation is the composed activation authority; there is no
// production binding of it to a listener yet.
class delegation {
public:
    virtual ~delegation() = default;
    // Returns a held section or an empty lock; never waits on a second lock.
    virtual std::unique_lock<std::recursive_mutex> enter() const = 0;
    // Revalidates every lock, inode, owner and mode the authority holds.
    virtual bool valid_for(const std::filesystem::path & parent) const = 0;
    virtual const std::filesystem::path & directory() const = 0;
    // The exact concrete token the anchored SQLite store and the v3 object
    // writer/verifier require. Never null while enter() succeeds.
    virtual std::shared_ptr<lease> concrete() const = 0;
};
class lease final : public std::enable_shared_from_this<lease> {
public:
    ~lease();
    lease(const lease &)=delete;lease & operator=(const lease &)=delete;
    // Nonblocking exclusive lock; the parent must already exist and be a
    // private directory owned by this UID. Never initializes or repairs data.
    static std::shared_ptr<lease> try_acquire(const std::filesystem::path & parent,std::string & error);
    // Holds the per-token recursive mutex for the entire operation. A copy of
    // the shared_ptr must also remain alive. Forked children cannot reuse it.
    std::unique_lock<std::recursive_mutex> enter() const;
    bool valid_for(const std::filesystem::path & parent) const;
    // Reject a different control block aliasing this token.
    bool owns(const std::shared_ptr<lease> & candidate) const;
    const std::filesystem::path & directory() const;
private:
    struct impl;std::unique_ptr<impl> data_;
    lease();
};
}
