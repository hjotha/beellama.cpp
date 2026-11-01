#pragma once

// Windows/UWP peer identity store and certificate pin for RKVA blocker 7.
//
// This is test-only scaffolding until an MSVC/UWP build and the console
// restart/power-loss rows exist. It is default off: the library is only built
// with -DRKVA_PEER_IDENTITY_STORE=ON and every entry point additionally
// requires RKVA_PEER_IDENTITY_STORE=1 at run time. No listener, capability,
// HELLO change or dispatch path calls it.
//
// The store is a single canonical JSON record under
// <LocalState>/rkva-peer-identity-v3 with an exclusive lock file. The record
// carries the pinned package/certificate identity and is sealed with an
// HMAC-SHA256 whose key is protected by the platform (DPAPI on Windows). The
// MAC detects corruption and off-machine/off-user copies; it is not proof of
// who authored a bundle. A bundle is accepted only from an explicitly
// authorized provisioning session whose certificate digest was obtained out of
// band. TOFU is allowed only for the first enrollment of a new host identity,
// and a later certificate change is a hard refusal until an explicit rotation.
//
// This header and its implementation contain no logging. Error strings are
// static field descriptions and never contain keys, MACs or certificate bytes.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rkva_peer_identity {

constexpr const char * schema_name = "rkva-peer-identity-v1";
constexpr uint32_t schema_version = 1;
constexpr const char * namespace_name = "rkva-peer-identity-v3";
constexpr const char * lock_name = "identity.lock";
constexpr const char * record_name = "identity.json";
constexpr const char * pending_name = "identity.pending";
constexpr const char * staging_name = "identity.new";
constexpr size_t max_record_bytes = 8192;
constexpr size_t max_rotation_history = 8;

using key = std::array<uint8_t, 32>;
using sha256 = std::array<uint8_t, 32>;
using thumbprint = std::array<uint8_t, 20>;

enum class status : uint8_t {
    ok,
    disabled,          // compile option or runtime gate is closed
    unknown,           // no record; explicit enrollment is required
    corrupt,           // malformed, non-canonical, ambiguous or bad MAC
    unavailable,       // locked, unreadable or platform failure
    mismatch,          // pinned identity differs from the presented peer
    revoked,           // record is explicitly revoked
    already_pinned,    // enrollment/provisioning would replace a record
    unauthorized,      // missing/incorrect explicit operator authorization
    invalid_argument,  // caller-supplied field is invalid
};

const char * status_name(status value);

// A peer identity as observed at connection time. The certificate values are
// digests of the DER leaf certificate, never the certificate or a private key.
struct peer {
    std::string pfn;             // exact PackageFullName
    std::string package_family;  // optional; derived from pfn when empty
    std::string publisher;       // WDP Publisher string (X.500 subject)
    std::string host;            // peer host name or IP
    uint16_t port = 0;
    sha256 cert_sha256{};        // SHA-256 over the DER leaf certificate
    thumbprint cert_thumbprint{};  // SHA-1 thumbprint (Windows convention)
};

// Explicit first-use enrollment. Allowed only when no record exists. The
// out-of-band digest must have been obtained from the console UI or a physical
// label, never from the connection being enrolled.
struct enrollment {
    bool explicit_operator_confirmation = false;
    std::string operator_label;  // bounded, non-secret evidence label
    sha256 out_of_band_cert_sha256{};
};

// Explicit rotation to a new certificate/package identity. The retired
// certificate stays in a bounded history and may not be re-pinned.
struct rotation {
    bool explicit_operator_confirmation = false;
    std::string operator_label;
    sha256 out_of_band_cert_sha256{};
};

struct revocation {
    bool explicit_operator_confirmation = false;
    std::string operator_label;
    std::string reason;  // bounded, non-secret reason token
};

// Provisioning from an explicitly authorized WDP session. Never TOFU: the
// certificate digest still has to come from an out-of-band source.
struct provisioning {
    bool authorized_wdp_session = false;
    std::string operator_label;
    sha256 out_of_band_cert_sha256{};
    uint64_t first_seen = 0;  // 0 uses the store clock
    peer observed;
};

struct wipe_request {
    bool explicit_operator_confirmation = false;
    std::string operator_label;
};

// Non-secret inspection view for evidence. It deliberately has no key, MAC or
// protected-key field.
struct view {
    std::string pfn;
    std::string package_family;
    std::string publisher;
    std::string publisher_id;
    std::string peer_host;
    uint16_t peer_port = 0;
    sha256 cert_sha256{};
    thumbprint cert_thumbprint{};
    uint64_t first_seen = 0;
    bool pinned = false;
    bool revoked = false;
    bool tofu_enrolled = false;
    std::string tofu_boundary;
    uint64_t tofu_enrolled_at = 0;
    uint32_t rotation_generation = 0;
    uint64_t rotated_at = 0;
    size_t rotation_history = 0;
    uint64_t revoked_at = 0;
    std::string revocation_reason;
};

std::string to_hex(const uint8_t * data, size_t size);
bool from_hex(const std::string & text, uint8_t * out, size_t size);
bool parse_pfn(const std::string & pfn, std::string & package_family,
               std::string & publisher_id);
std::string canonical_host(const std::string & host);

// Platform storage. read() returns ok, unknown (no record), corrupt
// (ambiguous/partial namespace) or unavailable. write() is an atomic replace
// that fails closed when a pending marker or unknown entry is present.
class store {
public:
    virtual ~store() = default;
    virtual status read(std::string & bytes, std::string & error) = 0;
    virtual status write(const std::string & bytes, std::string & error) = 0;
    virtual status erase(std::string & error) = 0;
    virtual bool new_key(key & out, std::string & error) = 0;
    virtual bool seal_key(const key & value, std::string & sealed,
                          std::string & error) = 0;
    virtual bool unseal_key(const std::string & sealed, key & out,
                            std::string & error) = 0;
};

// The run-time gate is open only for the exact value "1" in a build that
// defines RKVA_PEER_IDENTITY_STORE.
bool enabled();

// Connection hook: call after the peer certificate is observed and before any
// authenticated frame, credential or managed payload is accepted. It never
// enrolls; a missing record is an unknown refusal.
status verify(store & backend, const peer & observed, std::string & error);

// First-use-only enrollment. Refuses when any record exists, when the
// confirmation is not explicit, or when the out-of-band digest differs.
status enroll_tofu(store & backend, const peer & observed,
                   const enrollment & request, std::string & error);

// Explicit replacement of the pinned identity. Revoked records cannot rotate.
status rotate(store & backend, const peer & observed,
              const rotation & request, std::string & error);

status revoke(store & backend, const revocation & request, std::string & error);

status provision(store & backend, const provisioning & request,
                 std::string & error);

status wipe(store & backend, const wipe_request & request, std::string & error);

// Reads and validates the record, but returns only non-secret fields.
status inspect(store & backend, view & out, std::string & error);

#ifdef RKVA_PEER_IDENTITY_TESTING
// Test-only in-memory backend. It performs no platform protection; seal_key()
// is plain hex so CPU tests can inspect and tamper with record bytes.
class test_memory_store final : public store {
public:
    status read(std::string & bytes, std::string & error) override;
    status write(const std::string & bytes, std::string & error) override;
    status erase(std::string & error) override;
    bool new_key(key & out, std::string & error) override;
    bool seal_key(const key & value, std::string & sealed,
                  std::string & error) override;
    bool unseal_key(const std::string & sealed, key & out,
                    std::string & error) override;

    std::string bytes;
    bool present = false;
    bool fail_read = false;
    bool fail_write = false;
    key stored_key{};
    bool has_key = false;
};

// Test-only directory backend. On POSIX it mirrors the Windows sequence
// (pending marker, staging file, rename, directory fsync, exact namespace) but
// makes no AppContainer/ACL claim. On Windows it returns the real Windows
// backend rooted at the supplied LocalState directory.
std::unique_ptr<store> open_test_directory(const std::string & local_state,
                                           std::string & error);

extern uint64_t (* test_clock)();
extern bool (* test_step)(const char * step);
#endif

#ifdef _WIN32
// Production UWP entry point: uses ApplicationData::Current().LocalFolder()
// when RKVA_PEER_IDENTITY_UWP is defined. A desktop build refuses instead of
// guessing a path.
std::unique_ptr<store> open_local_state(std::string & error);
#endif

}  // namespace rkva_peer_identity
