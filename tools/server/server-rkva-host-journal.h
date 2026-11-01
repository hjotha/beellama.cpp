#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Standalone host transaction journal. Not connected to a server, RPC, host
// file management or GC. All paths and identities are supplied by trusted
// caller; SHA-256 is corruption detection, not authentication or rollback
// protection. Directory must exist and be exclusively administered.
namespace rkva_host_journal {
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
};
#ifdef RKVA_HOST_JOURNAL_TESTING
extern void (*test_persist_hook)(const char * stage);
#endif
// owner, tx, reference, key and host_sha256 are 64 lowercase hex digits.
// All operations reacquire the interprocess lock and reload the journal.
// Initialization is explicit; a missing/corrupt journal is never re-created
// by any other method. Methods never inspect, mutate or delete host files.
class journal {
public:
    journal(std::string directory, std::string owner);
    bool initialize(std::string & error);
    bool inspect(snapshot & result, std::string & error) const;
    bool prepare(const std::string & tx, const std::string & reference, const std::string & key, std::string & error);
    bool host_published(const std::string & tx, const std::string & path,
                        const std::string & whole_file_sha256, uint64_t bytes, std::string & error);
    bool commit_ack(const std::string & tx, uint64_t peer_version, std::string & error);
    bool release_intent(const std::string & tx, uint64_t peer_version, std::string & error);
    bool release_ack(const std::string & tx, uint64_t peer_version, std::string & error);
    // Caller must have independently proved that the host file was never
    // published. Journal state alone cannot prove this across a crash.
    bool abort_unpublished(const std::string & tx, std::string & error);
private:
    std::string dir_, owner_;
};
} // namespace rkva_host_journal
