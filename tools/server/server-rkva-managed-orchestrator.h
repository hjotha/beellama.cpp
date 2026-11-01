#pragma once
#include "server-rkva-host-journal.h"
#include "../../ggml/include/rkva-managed-registry.h"
#include "../../ggml/include/rkva-managed-store-authority.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

// Standalone POSIX-only composition prototype. It is deliberately not wired to
// server or RKVA. The caller exclusively owns both stores and both durable
// libraries; identifiers are scope keys, not authentication.
namespace rkva_managed_orchestrator {
// Additional read-only source check. It does not replace the existing
// full-file hash contract or establish peer object presence.
using source_preflight = std::function<bool(int fd, std::string & key, uint64_t & bytes, std::string & error)>;
#ifdef RKVA_MANAGED_ORCHESTRATOR_TESTING
extern void (*test_hook)(const char * stage);
#endif
class orchestrator {
public:
    // peer_objects contains a verified immutable <key>.blob; host_root is a
    // distinct, dedicated managed directory. This prototype defines key as
    // the SHA-256 of the complete file, NOT Xbox's native prefix digest.
    orchestrator(rkva_host_journal::journal & journal, rkva_managed_registry::registry & peer,
                 std::string host_root, std::string peer_objects, uint64_t max_bytes,
                 source_preflight check_source = {},
                 std::shared_ptr<rkva_managed_store::delegation> authority = {});
    // Resume the same tx/ref/key; source may be empty if the host is already
    // staged or published. Never aborts or removes a peer object on failure.
    bool publish(const std::string & tx, const std::string & ref, const std::string & key,
                 const std::string & source, std::string & error);
    // Release only after recording a durable deletion intent, validating and
    // durably unlinking this managed host file. No v2 object is touched.
    bool release(const std::string & tx, std::string & error);
private:
    rkva_host_journal::journal & journal_;
    rkva_managed_registry::registry & peer_;
    std::string host_root_, peer_objects_;
    uint64_t max_bytes_;
    source_preflight check_source_;
    std::shared_ptr<rkva_managed_store::delegation> authority_;
};
}
