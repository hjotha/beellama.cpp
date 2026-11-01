// Test-only staging body of the v3 object writer.
//
// The production rkva-managed-object-v3.cpp intentionally stops at
// `need(false, "concrete authenticated pin and metadata proof unavailable")`
// and never calls anything in this file. This .inc is compiled only into
// tests/test-rkva-managed-object-v3-staging.cpp, which defines
// RKVA_DURABLE_FAULT_TESTING and supplies test-only pin/quota/source/backend
// implementations.
//
// Including this file without the test macro is a build error. It is not part
// of the rkva-managed-phase2 library, llama-server, the UWP app or any Phase
// 2/3/4 target, and a `published_pending` result here is NOT a managed COMMIT
// and conveys no GC or registry authority.
//
// The body is a copy of the production staging sequence (same order, same
// checks, same result states) minus the fail-closed gate. It exists so the
// durable-filesystem contract can be exercised against real bytes on a real
// POSIX directory; it is a deterministic model, never power-loss evidence.

#ifndef RKVA_DURABLE_FAULT_TESTING
#error "rkva-managed-object-v3-write.inc is test-only; define RKVA_DURABLE_FAULT_TESTING"
#endif

#ifndef RKVA_MANAGED_OBJECT_V3_WRITE_INC
#define RKVA_MANAGED_OBJECT_V3_WRITE_INC

#include "rkva-managed-object-v3-internal.h"

namespace rkva_managed_object_v3 {
using namespace detail;

// Preconditions are revalidated here so a test cannot bypass the concrete
// lease/root/namespace checks merely by calling this function directly. The
// only difference from write() is that the authenticated-pin refusal is
// replaced by the caller-supplied pin_proof, which is a test fixture.
inline result write_prepared(const fs::path & root,
                             std::shared_ptr<rkva_managed_store::lease> store,
                             const pin & p, const expected & e, pin_proof * proof,
                             quota_lease * quota, durable_filesystem * disk,
                             source * input) {
    result r;
    bool move_started = false;
    try {
        // Compare shared ownership as well as the pointee: a separately owned
        // alias must not impersonate the concrete lease. No backend callbacks
        // or object-path I/O occur before these checks.
        need(store && disk, "v3 filesystem/store authority unavailable");
        const auto bound = disk->authority().lock();
        need(bound && bound.get() == store.get() && !bound.owner_before(store) &&
                 !store.owner_before(bound),
             "v3 filesystem/store lease mismatch or expired");
        need(!root.empty() && disk->root() == root.lexically_normal(),
             "v3 filesystem root mismatch");
        auto operation =
            store && store->owns(store) ? store->enter() : std::unique_lock<std::recursive_mutex>{};
        need(operation.owns_lock() && store->valid_for(root.parent_path()),
             "concrete store authority unavailable");
        validate_root(root);
        need(valid_pin(p), "invalid v3 pin");
        hold(proof, p, e, store);
        need(quota && disk && input && disk->qualified(),
             "v3 durable backend or reservation unavailable");
        need(input->bound_to(p, e), "v3 source binding unavailable");
        snap::Manifest manifest;
        std::string why;
        if (!snap::build(e.geometry, e.descriptor, e.rows, e.generation, p.key.data(), e.layers,
                         manifest, &why))
            throw std::runtime_error("v3 snapshot geometry rejected: " + why);
        std::vector<uint8_t> encoded;
        need(snap::encode(manifest, encoded, &why), "v3 manifest sizing failed");
        constexpr uint64_t overhead = 1ull << 20;
        need(encoded.size() <= UINT64_MAX - overhead &&
                 manifest.payload_bytes <= UINT64_MAX - encoded.size() - overhead,
             "v3 quota arithmetic overflow");
        const uint64_t reserved = manifest.payload_bytes + encoded.size() + overhead;
        need(quota->held_for(reserved), "v3 quota reservation unavailable");
        r.object = root / rkva_snapshot::hex(p.key.data());
        r.staging = root / ("tmp-" + rkva_snapshot::hex(p.transaction.data()));
        need(path_status(r.object).type() == fs::file_type::not_found,
             "v3 object already exists; refusing overwrite");
        std::error_code ec;
        need(fs::create_directory(r.staging, ec) && !ec,
             "v3 staging already exists or cannot be created");
        r.state = publication::staging_preserved;
        hold(proof, p, e, store);
        if (!disk->sync_directory(root, why))
            throw std::runtime_error("v3 staging directory sync failed: " + why);
        std::array<uint8_t, 64 * 1024> buffer{};
        for (size_t i = 0; i < manifest.chunks.size(); i++) {
            const auto path = r.staging / ("n-" + std::to_string(i) + ".kv");
            need(path_status(path).type() == fs::file_type::not_found,
                 "v3 staging chunk already exists");
            auto & c = manifest.chunks[i];
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            need(bool(output), "v3 chunk create failed");
            rkva_snapshot::Sha256 hash;
            uint64_t pos = 0;
            while (pos < c.bytes) {
                hold(proof, p, e, store);
                need(quota->held_for(reserved), "v3 quota lease lost");
                need(input->bound_to(p, e), "v3 source binding lost");
                const auto n = size_t(std::min<uint64_t>(c.bytes - pos, buffer.size()));
                if (!input->read(c, pos, buffer.data(), n, why))
                    throw std::runtime_error("v3 source read failed: " + why);
                need(bool(output.write(reinterpret_cast<const char *>(buffer.data()),
                                       std::streamsize(n))),
                     "v3 chunk write failed");
                hash.update(buffer.data(), n);
                pos += n;
            }
            output.flush();
            need(bool(output), "v3 chunk flush failed");
            output.close();
            need(bool(output), "v3 chunk close failed");
            c.digest = hash.finish();
            regular(path, c.bytes);
            if (!disk->sync_file(path, why))
                throw std::runtime_error("v3 chunk durable sync failed: " + why);
        }
        need(snap::encode(manifest, encoded, &why), "v3 manifest encoding failed");
        const auto path = r.staging / "manifest-v2.bin";
        need(path_status(path).type() == fs::file_type::not_found,
             "v3 staging manifest already exists");
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        need(bool(output.write(reinterpret_cast<const char *>(encoded.data()),
                               std::streamsize(encoded.size()))),
             "v3 manifest write failed");
        output.flush();
        need(bool(output), "v3 manifest flush failed");
        output.close();
        need(bool(output), "v3 manifest close failed");
        if (!disk->sync_file(path, why) || !disk->sync_directory(r.staging, why))
            throw std::runtime_error("v3 staged object sync failed: " + why);
        hold(proof, p, e, store);
        need(quota->held_for(reserved), "v3 quota lease lost");
        std::string digest;
        validate_directory(r.staging, p, e, digest);
        hold(proof, p, e, store);
        need(quota->held_for(reserved), "v3 quota lease lost");
        need(input->bound_to(p, e), "v3 source binding lost");
        need(store->valid_for(root.parent_path()), "concrete store authority lost");
        move_started = true;
        const auto moved = disk->rename_noreplace_and_sync(r.staging, r.object, why);
        if (moved == move_result::no_change) {
            move_started = false;
            throw std::runtime_error("v3 publication refused: " + why);
        }
        if (moved != move_result::durable) {
            r.state = publication::publication_uncertain;
            r.error = "v3 publication outcome uncertain: " + why;
            return r;
        }
        // An authority loss or verification failure after publication is an
        // uncertain pair, never permission to delete or claim commit.
        hold(proof, p, e, store);
        need(quota->held_for(reserved), "v3 quota lease lost");
        std::string verify_digest;
        validate_directory(r.object, p, e, verify_digest);
        need(verify_digest == digest, "v3 published manifest changed");
        need(store->valid_for(root.parent_path()), "concrete store authority lost");
        hold(proof, p, e, store);
        r.state = publication::published_pending;
        r.payload_bytes = manifest.payload_bytes;
        r.manifest_sha256 = digest;
        r.staging.clear();
        return r;
    } catch (const std::exception & ex) {
        if (move_started) r.state = publication::publication_uncertain;
        r.error = ex.what();
        return r;
    } catch (...) {
        if (move_started) r.state = publication::publication_uncertain;
        r.error = "v3 write failed";
        return r;
    }
}

} // namespace rkva_managed_object_v3

#endif // RKVA_MANAGED_OBJECT_V3_WRITE_INC
