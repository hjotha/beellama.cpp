#pragma once
#include "rkva-native-snapshot.h"
#include "../../../../ggml/include/rkva-managed-store-authority.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

// Isolated v3 object component, not connected to RKVA, registry, GPU or UWP
// runtime. Caller must supply actual durable authority, exclusive store lock,
// pin, quota reservation and platform-specific durable publication. No default
// backend exists. None of these interfaces authorizes COMMIT, RELEASE or GC.
namespace rkva_managed_object_v3 {
namespace fs = std::filesystem;
struct pin {
    rkva_snapshot::Digest owner{}, transaction{}, reference{}, key{};
};
struct expected {
    rkva_native_snapshot::Geometry geometry{};
    rkva_native_snapshot::Descriptor descriptor{};
    uint32_t rows = 0;
    uint64_t generation = 0;
    std::vector<rkva_native_snapshot::LayerCoverage> layers;
};
struct pin_proof {
    virtual ~pin_proof() = default;
    // Proves durable pin and metadata; NOT a replacement for the separately
    // checked concrete store token. No production proof implementation exists.
    virtual bool pending(const pin &, const expected &,
                         const std::shared_ptr<rkva_managed_store::lease> &) = 0;
    virtual bool pinned(const pin &, const expected &,
                        const std::shared_ptr<rkva_managed_store::lease> &) = 0;
};
struct quota_lease {
    virtual ~quota_lease() = default;
    // Includes staged and published bytes, filesystem allocation overhead and
    // other writers. Lease must remain held throughout.
    virtual bool held_for(uint64_t worst_case_bytes) = 0;
};
enum class move_result { durable, no_change, uncertain };
struct durable_filesystem {
protected:
    // Fixed, inspectable binding; the writer compares concrete lease identity
    // and root before invoking any backend method or object filesystem I/O.
    durable_filesystem(std::weak_ptr<rkva_managed_store::lease> lease, fs::path root)
        : lease_(std::move(lease)), root_(root.lexically_normal()) {}
private:
    const std::weak_ptr<rkva_managed_store::lease> lease_;
    const fs::path root_;
public:
    durable_filesystem(const durable_filesystem &) = delete;
    durable_filesystem & operator=(const durable_filesystem &) = delete;
    const std::weak_ptr<rkva_managed_store::lease> & authority() const { return lease_; }
    const fs::path & root() const { return root_; }
    virtual ~durable_filesystem() = default;
    // No default true implementation. Caller must establish that these exact
    // platform/filesystem semantics are actually available and qualified.
    virtual bool qualified() const = 0;
    virtual bool sync_file(const fs::path &, std::string &) = 0;
    virtual bool sync_directory(const fs::path &, std::string &) = 0;
    // Must atomically refuse an existing destination, then durably sync the
    // parent. An uncertain outcome must NEVER be reported as durable.
    virtual move_result rename_noreplace_and_sync(const fs::path &, const fs::path &, std::string &) = 0;
};
struct source {
    virtual ~source() = default;
    // Must independently prove the frozen GPU/host snapshot belongs to this
    // exact pending pin and metadata for the entire write. No default proof.
    virtual bool bound_to(const pin &, const expected &) = 0;
    // Read bytes from that frozen snapshot; chunk is the canonical
    // layer/region/offset geometry.
    virtual bool read(const rkva_native_snapshot::Chunk &, uint64_t within_chunk,
                      void * dst, size_t bytes, std::string & error) = 0;
};
enum class publication { rejected, staging_preserved, publication_uncertain, published_pending };
struct result {
    publication state = publication::rejected;
    fs::path staging;
    fs::path object;
    uint64_t payload_bytes = 0;
    std::string manifest_sha256;
    std::string error;
};
// root must be an existing, trusted, separately provisioned directory named
// exactly rkva-snapshots-v3. A given transaction may not reuse a stale stage.
// Never deletes any object, stage, pin or legacy v1/v2 state. published_pending
// is NOT a managed registry COMMIT and conveys no GC authority.
result write(const fs::path & root, std::shared_ptr<rkva_managed_store::lease>, const pin &, const expected &, pin_proof *, quota_lease *,
             durable_filesystem *, source *);
// Inspection likewise requires externally held authority and a durable pin. It
// validates canonical manifest, exact names, extents and every chunk checksum.
bool verify(const fs::path & root, std::shared_ptr<rkva_managed_store::lease>, const pin &, const expected &, pin_proof *,
            std::string & manifest_sha256, std::string & error);
} // namespace rkva_managed_object_v3
