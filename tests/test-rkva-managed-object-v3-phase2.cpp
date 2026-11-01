// Phase 2 lease-bound object v3 test: the component is deliberately closed.
//
// The v3 writer/verifier requires a concrete authenticated pin/metadata proof
// (a durable pending pin bound to the same store lease and metadata). Phase 2
// does not supply one, so write() and verify() must refuse before any v3
// filesystem I/O. This test proves the refusal, the lease/root binding checks
// and that no object, stage or legacy file is created or removed.

#include "../tools/xbox/uwp/rkva/rkva-managed-object-v3.h"
#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;
namespace authority = rkva_managed_store;
namespace v3 = rkva_managed_object_v3;

struct pin_fixture final : v3::pin_proof {
    int calls = 0;
    bool pending(const v3::pin &, const v3::expected &,
                 const std::shared_ptr<authority::lease> &) override {
        ++calls;
        return true;
    }
    bool pinned(const v3::pin &, const v3::expected &,
                const std::shared_ptr<authority::lease> &) override {
        ++calls;
        return true;
    }
};

struct quota_fixture final : v3::quota_lease {
    int calls = 0;
    bool held_for(uint64_t) override {
        ++calls;
        return true;
    }
};

// Fixture backend only: reports success without implementing durability.
struct disk_fixture final : v3::durable_filesystem {
    mutable int calls = 0;
    disk_fixture(std::weak_ptr<authority::lease> token, fs::path root)
        : durable_filesystem(std::move(token), std::move(root)) {}
    bool qualified() const override {
        ++calls;
        return true;
    }
    bool sync_file(const fs::path &, std::string &) override {
        ++calls;
        return true;
    }
    bool sync_directory(const fs::path &, std::string &) override {
        ++calls;
        return true;
    }
    v3::move_result rename_noreplace_and_sync(const fs::path &, const fs::path &,
                                              std::string &) override {
        ++calls;
        return v3::move_result::durable;
    }
};

struct source_fixture final : v3::source {
    int calls = 0;
    bool bound_to(const v3::pin &, const v3::expected &) override {
        ++calls;
        return true;
    }
    bool read(const rkva_native_snapshot::Chunk &, uint64_t, void *, size_t,
              std::string &) override {
        ++calls;
        return true;
    }
};

int main() {
    char tmp[] = "/tmp/rkva-phase2-v3-XXXXXX";
    char * name = mkdtemp(tmp);
    assert(name != nullptr);
    const fs::path parent(name);
    const fs::path root = parent / "rkva-snapshots-v3";
    const fs::path other = parent / "other";
    assert(fs::create_directory(root));
    assert(fs::create_directory(other));
    {
        std::ofstream sentinel(root / "sentinel");
        sentinel << "preserve";
    }

    std::string error;
    auto token = authority::lease::try_acquire(parent, error);
    assert(token);

    v3::pin pin{};
    pin.owner.fill(1);
    pin.transaction.fill(2);
    pin.reference.fill(3);
    pin.key.fill(4);
    v3::expected expected{};
    pin_fixture proof;
    quota_fixture quota;
    source_fixture source;
    disk_fixture disk(token, root);
    disk_fixture other_root(token, other);

    // Missing or mismatched concrete authority is refused before callbacks.
    auto result = v3::write(root, {}, pin, expected, &proof, &quota, &disk, &source);
    assert(result.state == v3::publication::rejected);
    assert(result.error.find("filesystem/store authority") != std::string::npos);

    auto alias = std::shared_ptr<authority::lease>(token.get(), [](authority::lease *) {});
    disk_fixture forged(alias, root);
    result = v3::write(root, token, pin, expected, &proof, &quota, &forged, &source);
    assert(result.state == v3::publication::rejected);
    assert(result.error.find("lease mismatch") != std::string::npos);

    result = v3::write(root, token, pin, expected, &proof, &quota, &other_root, &source);
    assert(result.state == v3::publication::rejected);
    assert(result.error.find("root mismatch") != std::string::npos);

    auto unrelated = authority::lease::try_acquire(other, error);
    assert(unrelated);
    disk_fixture unrelated_disk(unrelated, root);
    result = v3::write(root, token, pin, expected, &proof, &quota, &unrelated_disk, &source);
    assert(result.state == v3::publication::rejected);
    assert(result.error.find("lease mismatch") != std::string::npos);

    // With a matching lease/root the component still fails closed: no concrete
    // authenticated pin/metadata proof exists in Phase 2.
    result = v3::write(root, token, pin, expected, &proof, &quota, &disk, &source);
    assert(result.state == v3::publication::rejected);
    assert(result.error.find("concrete authenticated pin and metadata proof unavailable") !=
           std::string::npos);
    assert(proof.calls == 0);
    assert(quota.calls == 0);
    assert(source.calls == 0);
    assert(disk.calls == 0);

    std::string digest = "unchanged";
    assert(!v3::verify(root, token, pin, expected, &proof, digest, error));
    assert(digest == "unchanged");
    assert(error.find("concrete authenticated pin and metadata proof unavailable") !=
           std::string::npos);
    assert(!v3::verify(root, {}, pin, expected, &proof, digest, error));

    // Nothing was staged, published or deleted; the sentinel still exists.
    for (const auto & entry : fs::directory_iterator(root)) {
        assert(entry.path().filename() == "sentinel");
    }
    std::ifstream sentinel(root / "sentinel");
    std::string value;
    sentinel >> value;
    assert(value == "preserve");

    token.reset();
    unrelated.reset();
    fs::remove_all(parent);
    std::cout << "managed v3 object closed under concrete lease/root: PASS\n";
    return 0;
}
