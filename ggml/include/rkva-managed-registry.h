#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Standalone prototype. This registry is not connected to the RKVA protocol,
// snapshot storage, host journals, or restore readers. It never deletes blobs.
namespace rkva_managed_registry {

#ifdef RKVA_MANAGED_REGISTRY_TESTING
// Fault-injection hook, compiled only into the standalone test binary.
extern void (*test_persist_hook)(const char * stage);
#endif

enum class phase { pending, committed, aborted, released };
struct transaction {
    std::string id, reference, key;
    phase state = phase::pending;
    uint64_t committed_version = 0;
};
struct snapshot {
    uint64_t version = 0;
    std::vector<transaction> transactions;
};
struct candidate {
    std::string key;
    uint64_t observed_version;
};

// Explicit trust boundary, not authentication: the caller must exclusively
// control the directory and authenticate the owner outside this library. All
// identities are 64-character lowercase hex strings, including the SHA-256 key.
// The directory must already exist. Initialization is explicit and never
// replaces an existing or damaged registry. Never use the legacy v2 directory.
class registry {
public:
    registry(std::string directory, std::string single_owner);
    bool initialize(std::string & error);
    bool inspect(snapshot & out, std::string & error) const;
    bool prepare(const std::string & tx, const std::string & ref, const std::string & key, std::string & error);
    bool commit(const std::string & tx, uint64_t & committed_version, std::string & error);
    bool abort(const std::string & tx, std::string & error);
    bool release(const std::string & ref, uint64_t committed_version, std::string & error);
    // Observations only, never deletion authorization: a new pin can appear
    // immediately afterwards and this prototype cannot observe active readers,
    // legacy copies, unmanaged files or remote owners. Only keys introduced by
    // this registry and with no pending/committed reference are reported.
    bool candidates(std::vector<candidate> & out, std::string & error) const;
private:
    std::string dir_, owner_;
};
}
