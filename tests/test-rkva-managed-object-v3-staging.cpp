// CPU contract tests for the test-only v3 staging seam.
//
// This translation unit is the ONLY place that compiles
// tools/xbox/uwp/rkva/rkva-managed-object-v3-write.inc. The production
// rkva-managed-object-v3.cpp still refuses at
// "concrete authenticated pin and metadata proof unavailable" and is covered by
// test-rkva-managed-object-v3-phase2.
//
// The tests drive the real staging sequence over a real temporary POSIX
// directory with the injectable durable_filesystem from rkva-durable-fault.h:
//
//   * every pre-rename fault leaves the stage, the committed pair and the
//     legacy v1/v2 namespaces byte-identical;
//   * a no-replace refusal or rename EIO never reports durable/published;
//   * ack loss and crash-after-apply are "publication_uncertain", never durable;
//   * a parent-sync failure after apply is uncertain;
//   * a stale stage is never adopted by a retry, even after the lease is
//     released and reacquired;
//   * symlinks and existing destinations are refused;
//   * fault queues are one-shot and independent per operation kind.
//
// No console, WDP, UWP, service, GPU or real power loss is involved.

#ifndef RKVA_DURABLE_FAULT_TESTING
#define RKVA_DURABLE_FAULT_TESTING
#endif

#include "rkva-durable-fault.h"
#include "../tools/xbox/uwp/rkva/rkva-managed-object-v3-write.inc"

#undef NDEBUG
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
namespace authority = rkva_managed_store;
namespace fi = rkva_durable_fault;
namespace v3 = rkva_managed_object_v3;
namespace snap = rkva_native_snapshot;

namespace {

std::string read_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

void write_file(const fs::path & path, const std::string & bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), std::streamsize(bytes.size()));
    assert(bool(output));
}

// Deterministic digest over names, sizes and bytes. Used only to prove that a
// tree did not change across a failed publication attempt.
std::string tree_digest(const fs::path & dir) {
    std::vector<fs::path> entries;
    if (fs::exists(dir)) {
        for (const auto & entry : fs::recursive_directory_iterator(dir))
            entries.push_back(entry.path());
    }
    std::sort(entries.begin(), entries.end());
    rkva_snapshot::Sha256 hash;
    for (const auto & path : entries) {
        const auto relative = path.lexically_relative(dir).generic_string();
        hash.update(relative.data(), relative.size());
        if (fs::is_regular_file(path)) {
            const auto bytes = read_file(path);
            const uint64_t size = bytes.size();
            hash.update(&size, sizeof(size));
            hash.update(bytes.data(), bytes.size());
        }
    }
    const auto digest = hash.finish();
    return rkva_snapshot::hex(digest.data());
}

v3::pin make_pin(uint8_t transaction = 0x22, uint8_t key = 0x44) {
    v3::pin pin{};
    pin.owner.fill(0x11);
    pin.transaction.fill(transaction);
    pin.reference.fill(0x33);
    pin.key.fill(key);
    return pin;
}

// Distinct transaction/key per scenario: stage names depend on the transaction
// and object names on the key, so a test cannot accidentally observe state left
// by an earlier scenario.
v3::pin distinct_pin() {
    static uint8_t counter = 0x30;
    ++counter;
    return make_pin(counter, uint8_t(counter + 0x30));
}

v3::expected make_expected() {
    v3::expected expected{};
    expected.geometry = snap::Geometry{8, 2, 256, 2, 4096, 2};
    expected.descriptor = snap::Descriptor{1024, 1024, RKVA_RANGE_STORAGE_KVARN4, 3, 128,
                                           128, 256, 2, 4, 4, 8960, 3, 2, 128, 128,
                                           1, 1, 1};
    expected.rows = 513;
    expected.generation = 0x700000001ull;
    expected.layers = {snap::LayerCoverage{385, 384, 4, 1},
                       snap::LayerCoverage{385, 384, 4, 1}};
    return expected;
}

struct proof_fixture final : v3::pin_proof {
    long pending_calls = 0;
    long pinned_calls = 0;
    long fail_pending_at = -1;
    bool pending(const v3::pin &, const v3::expected &,
                 const std::shared_ptr<authority::lease> &) override {
        ++pending_calls;
        return fail_pending_at < 0 || pending_calls < fail_pending_at;
    }
    bool pinned(const v3::pin &, const v3::expected &,
                const std::shared_ptr<authority::lease> &) override {
        ++pinned_calls;
        return true;
    }
};

struct quota_fixture final : v3::quota_lease {
    long calls = 0;
    bool allow = true;
    uint64_t last_requested = 0;
    bool held_for(uint64_t bytes) override {
        ++calls;
        last_requested = bytes;
        return allow;
    }
};

struct source_fixture final : v3::source {
    long bound_calls = 0;
    long read_calls = 0;
    long fail_read_at = -1;
    bool bound_to(const v3::pin &, const v3::expected &) override {
        ++bound_calls;
        return true;
    }
    bool read(const snap::Chunk & chunk, uint64_t within, void * dst, size_t bytes,
              std::string &) override {
        ++read_calls;
        if (fail_read_at >= 0 && read_calls >= fail_read_at) return false;
        auto * out = static_cast<uint8_t *>(dst);
        for (size_t i = 0; i < bytes; ++i) {
            const uint64_t index = within + i;
            out[i] = uint8_t((chunk.layer * 31u + chunk.region * 17u + index) & 0xffu);
        }
        return true;
    }
};

struct fixture {
    fs::path parent;
    fs::path root;
    fs::path legacy_v1;
    fs::path legacy_v2;
    fs::path committed;
    std::string committed_digest;
    std::shared_ptr<authority::lease> token;

    fixture() {
        char pattern[] = "/tmp/rkva-durable-seam-XXXXXX";
        char * made = mkdtemp(pattern);
        assert(made != nullptr);
        parent = fs::path(made);
        root = parent / "rkva-snapshots-v3";
        legacy_v1 = parent / "rkva-snapshots-v1";
        legacy_v2 = parent / "rkva-snapshots-v2";
        assert(fs::create_directory(root));
        assert(fs::create_directory(legacy_v1));
        assert(fs::create_directory(legacy_v2));
        write_file(legacy_v1 / "sentinel", "legacy-v1");
        write_file(legacy_v2 / "sentinel", "legacy-v2");
        // A previously committed pair is represented by bytes the writer must
        // never touch; the key is different from the pin under test.
        committed = root / std::string(64, 'a');
        assert(fs::create_directory(committed));
        write_file(committed / "manifest-v2.bin", "committed-manifest");
        write_file(committed / "n-0.kv", "committed-chunk");
        committed_digest = tree_digest(committed);
        std::string error;
        token = authority::lease::try_acquire(parent, error);
        assert(token);
    }
    ~fixture() {
        token.reset();
        std::error_code ec;
        fs::remove_all(parent, ec);
    }

    void assert_untouched() const {
        assert(tree_digest(committed) == committed_digest);
        assert(read_file(legacy_v1 / "sentinel") == "legacy-v1");
        assert(read_file(legacy_v2 / "sentinel") == "legacy-v2");
    }
};

struct outcome {
    v3::result result;
    fs::path staging;
    bool object_exists = false;
    bool staging_exists = false;
};

outcome run(fixture & f, fi::plan & schedule, const v3::pin & pin,
            const v3::expected & expected, proof_fixture & proof,
            quota_fixture & quota, source_fixture & source) {
    fi::injected_filesystem disk(f.token, f.root, &schedule);
    outcome out;
    out.staging = f.root / ("tmp-" + rkva_snapshot::hex(pin.transaction.data()));
    out.result = v3::write_prepared(f.root, f.token, pin, expected, &proof, &quota, &disk,
                                    &source);
    out.object_exists = fs::exists(out.result.object);
    out.staging_exists = fs::exists(out.staging);
    return out;
}

size_t chunk_count(const v3::expected & expected) {
    snap::Manifest manifest;
    std::string why;
    const auto pin = make_pin();
    assert(snap::build(expected.geometry, expected.descriptor, expected.rows,
                       expected.generation, pin.key.data(), expected.layers, manifest, &why));
    return manifest.chunks.size();
}

void expect_staging_failure(const fixture & f, const outcome & out,
                            const std::string & error_fragment) {
    static int sequence = 0;
    ++sequence;
    if (out.result.state != v3::publication::staging_preserved)
        std::cerr << "case " << sequence << " (" << error_fragment
                  << "): expected staging_preserved, got state="
                  << int(out.result.state) << " error=" << out.result.error << "\n";
    assert(out.result.state == v3::publication::staging_preserved);
    assert(!out.object_exists);
    assert(out.staging_exists);
    assert(out.result.error.find(error_fragment) != std::string::npos);
    f.assert_untouched();
}

void expect_uncertain(const fixture & f, const outcome & out,
                      const std::string & error_fragment) {
    assert(out.result.state == v3::publication::publication_uncertain);
    assert(out.result.state != v3::publication::published_pending);
    assert(out.result.error.find(error_fragment) != std::string::npos);
    f.assert_untouched();
}

// Static proof that the seam stays test-only. This is intentionally a source
// check, not a build-graph guess: the production writer must keep the exact
// fail-closed refusal and must never include the .inc.
void static_seam_checks() {
#ifndef RKVA_DURABLE_SOURCE_DIR
#error "RKVA_DURABLE_SOURCE_DIR must point at the repository root"
#endif
    const fs::path repo = fs::path(RKVA_DURABLE_SOURCE_DIR);
    std::ifstream production(repo / "tools/xbox/uwp/rkva/rkva-managed-object-v3.cpp",
                             std::ios::binary);
    assert(bool(production));
    const std::string writer((std::istreambuf_iterator<char>(production)),
                             std::istreambuf_iterator<char>());
    assert(writer.find("need(false,\"concrete authenticated pin and metadata proof unavailable\");") !=
           std::string::npos);
    assert(writer.find("rkva-managed-object-v3-write.inc") == std::string::npos);
    assert(writer.find("write_prepared") == std::string::npos);

    std::ifstream inc(repo / "tools/xbox/uwp/rkva/rkva-managed-object-v3-write.inc",
                      std::ios::binary);
    assert(bool(inc));
    const std::string seam((std::istreambuf_iterator<char>(inc)),
                           std::istreambuf_iterator<char>());
    assert(seam.find("#ifndef RKVA_DURABLE_FAULT_TESTING") != std::string::npos);
    assert(seam.find("#error") != std::string::npos);

    std::ifstream cmake(repo / "tests/CMakeLists.txt", std::ios::binary);
    assert(bool(cmake));
    const std::string targets((std::istreambuf_iterator<char>(cmake)),
                              std::istreambuf_iterator<char>());
    assert(targets.find("test-rkva-managed-object-v3-staging") != std::string::npos);
    assert(targets.find("RKVA_DURABLE_FAULT_TESTING") != std::string::npos);
    // The .inc must not appear in the library source list.
    const auto library = targets.find("add_library(rkva-managed-phase2");
    const auto library_end = targets.find("target_compile_features(rkva-managed-phase2");
    assert(library != std::string::npos && library_end != std::string::npos);
    assert(targets.substr(library, library_end - library)
               .find("rkva-managed-object-v3-write.inc") == std::string::npos);
}

} // namespace

int main() {
    static_seam_checks();
    fixture f;
    const auto pin = make_pin();
    const auto expected = make_expected();
    const size_t chunks = chunk_count(expected);

    // 0. The seam reaches published_pending only in a test-only build, and the
    // published bytes are canonical. The production writer remains closed;
    // test-rkva-managed-object-v3-phase2 asserts that separately.
    {
        fi::plan schedule;
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        const auto out = run(f, schedule, pin, expected, proof, quota, source);
        assert(out.result.state == v3::publication::published_pending);
        assert(out.result.staging.empty());
        assert(out.object_exists);
        assert(!out.staging_exists);
        assert(out.result.payload_bytes > 0);
        assert(out.result.manifest_sha256.size() == 64);
        assert(quota.calls > 1 && quota.last_requested > out.result.payload_bytes);
        assert(proof.pending_calls > 1);
        assert(source.read_calls > 0);
        assert(schedule.file_sync_calls == long(chunks + 1)); // chunks + manifest
        assert(schedule.dir_sync_calls == 3); // root, staging, rename parent
        assert(schedule.rename_calls == 1);
        assert(!schedule.crashed);
        std::string digest;
        v3::detail::validate_directory(out.result.object, pin, expected, digest);
        assert(digest == out.result.manifest_sha256);
        f.assert_untouched();
    }

    // 1. Chunk sync failure at k: stage and the three least chunks remain,
    // nothing is renamed, and a retry with the same transaction cannot adopt it.
    {
        const auto case_pin = distinct_pin();
        fi::plan schedule;
        schedule.file_sync = {fi::fault::none, fi::fault::none, fi::fault::eio};
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        const auto out = run(f, schedule, case_pin, expected, proof, quota, source);
        expect_staging_failure(f, out, "chunk durable sync failed");
        assert(schedule.rename_calls == 0);
        assert(schedule.dir_sync_calls == 1);
        assert(fs::exists(out.staging / "n-0.kv"));
        assert(fs::exists(out.staging / "n-1.kv"));
        assert(fs::exists(out.staging / "n-2.kv"));
        assert(!fs::exists(out.staging / "manifest-v2.bin"));
        const auto stage_digest = tree_digest(out.staging);

        fi::plan retry_schedule;
        proof_fixture retry_proof;
        quota_fixture retry_quota;
        source_fixture retry_source;
        const auto retry = run(f, retry_schedule, case_pin, expected, retry_proof,
                               retry_quota, retry_source);
        assert(retry.result.state == v3::publication::rejected);
        assert(retry.result.error.find("staging already exists") != std::string::npos);
        assert(tree_digest(out.staging) == stage_digest);
        assert(!retry.object_exists);
        assert(retry_schedule.rename_calls == 0);
        f.assert_untouched();
    }

    // 2. Manifest sync failure: every chunk exists, the manifest is present but
    // unsynced, and no rename is attempted.
    {
        const auto case_pin = distinct_pin();
        fi::plan schedule;
        schedule.file_sync.assign(chunks, fi::fault::none);
        schedule.file_sync.push_back(fi::fault::eio);
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        const auto out = run(f, schedule, case_pin, expected, proof, quota, source);
        expect_staging_failure(f, out, "staged object sync failed");
        assert(fs::exists(out.staging / "manifest-v2.bin"));
        for (size_t i = 0; i < chunks; ++i)
            assert(fs::exists(out.staging / ("n-" + std::to_string(i) + ".kv")));
        assert(schedule.rename_calls == 0);
    }

    // 3. Staging-directory sync failure after the manifest write: preserved,
    // no rename.
    {
        const auto case_pin = distinct_pin();
        fi::plan schedule;
        schedule.dir_sync = {fi::fault::none, fi::fault::eio};
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        const auto out = run(f, schedule, case_pin, expected, proof, quota, source);
        expect_staging_failure(f, out, "staged object sync failed");
        assert(fs::exists(out.staging / "manifest-v2.bin"));
        assert(schedule.rename_calls == 0);
        assert(schedule.dir_sync_calls == 2);
    }

    // 4. Injected racing writer: no-replace refuses, the stage stays intact.
    {
        const auto case_pin = distinct_pin();
        fi::plan schedule;
        schedule.rename = {fi::fault::refuse_existing};
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        const auto out = run(f, schedule, case_pin, expected, proof, quota, source);
        expect_staging_failure(f, out, "publication refused");
        assert(schedule.rename_calls == 1);
        assert(!fs::exists(out.result.object));
    }

    // 5. A destination that already exists is refused before the stage is
    // created; its bytes are never overwritten.
    {
        const auto case_pin = distinct_pin();
        const auto object = f.root / rkva_snapshot::hex(case_pin.key.data());
        assert(fs::create_directory(object));
        write_file(object / "n-0.kv", "preexisting");
        fi::plan schedule;
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        const auto out = run(f, schedule, case_pin, expected, proof, quota, source);
        assert(out.result.state == v3::publication::rejected);
        assert(out.result.error.find("refusing overwrite") != std::string::npos);
        assert(!out.staging_exists);
        assert(read_file(object / "n-0.kv") == "preexisting");
        assert(schedule.rename_calls == 0);
        fs::remove_all(object);
    }

    // 6. Rename EIO: the applied state is unknown, so the result is uncertain
    // and the stage is preserved. Never durable.
    {
        const auto case_pin = distinct_pin();
        fi::plan schedule;
        schedule.rename = {fi::fault::rename_eio};
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        const auto out = run(f, schedule, case_pin, expected, proof, quota, source);
        expect_uncertain(f, out, "uncertain");
        assert(out.staging_exists);
        assert(!out.object_exists);
        assert(schedule.rename_calls == 1);
    }

    // 7. Ack lost: the rename was applied but the parent sync was not, so the
    // object bytes exist while the result must stay uncertain.
    {
        const auto case_pin = distinct_pin();
        fi::plan schedule;
        schedule.rename = {fi::fault::ack_lost};
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        const auto out = run(f, schedule, case_pin, expected, proof, quota, source);
        expect_uncertain(f, out, "ack_lost");
        assert(out.object_exists);
        assert(!out.staging_exists);
        assert(!schedule.crashed);
        std::string digest;
        v3::detail::validate_directory(out.result.object, case_pin, expected, digest);
    }

    // 8. Crash after apply: same bytes, crash marker set, uncertain.
    {
        const auto case_pin = distinct_pin();
        fi::plan schedule;
        schedule.rename = {fi::fault::crash_after_apply};
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        const auto out = run(f, schedule, case_pin, expected, proof, quota, source);
        expect_uncertain(f, out, "crash_after_apply");
        assert(out.object_exists);
        assert(!out.staging_exists);
        assert(schedule.crashed);
        const auto object_digest = tree_digest(out.result.object);

        // A retry with the same key refuses to overwrite the uncertain object.
        fi::plan retry_schedule;
        proof_fixture retry_proof;
        quota_fixture retry_quota;
        source_fixture retry_source;
        const auto retry = run(f, retry_schedule, case_pin, expected, retry_proof,
                               retry_quota, retry_source);
        assert(retry.result.state == v3::publication::rejected);
        assert(retry.result.error.find("refusing overwrite") != std::string::npos);
        assert(tree_digest(out.result.object) == object_digest);
        f.assert_untouched();
    }

    // 9. Parent-directory sync failure after apply: uncertain, object exists,
    // never durable.
    {
        const auto case_pin = distinct_pin();
        fi::plan schedule;
        schedule.dir_sync = {fi::fault::none, fi::fault::none, fi::fault::eio};
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        const auto out = run(f, schedule, case_pin, expected, proof, quota, source);
        expect_uncertain(f, out, "parent sync failed");
        assert(out.object_exists);
        assert(!out.staging_exists);
        assert(schedule.dir_sync_calls == 3);
    }

    // 10. Source read failure: the partial stage is preserved and not renamed.
    {
        const auto case_pin = distinct_pin();
        fi::plan schedule;
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        source.fail_read_at = 2;
        const auto out = run(f, schedule, case_pin, expected, proof, quota, source);
        expect_staging_failure(f, out, "source read failed");
        assert(fs::exists(out.staging));
        assert(schedule.rename_calls == 0);
    }

    // 11. Pending-pin loss mid-write: the stage is preserved, no rename.
    {
        const auto case_pin = distinct_pin();
        fi::plan schedule;
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        proof.fail_pending_at = 5;
        const auto out = run(f, schedule, case_pin, expected, proof, quota, source);
        expect_staging_failure(f, out, "durable pending pin unavailable");
        assert(schedule.rename_calls == 0);
    }

    // 12. Quota refusal and unqualified backend fail before creating a stage.
    {
        fi::plan schedule;
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        quota.allow = false;
        auto out = run(f, schedule, distinct_pin(), expected, proof, quota, source);
        assert(out.result.state == v3::publication::rejected);
        assert(out.result.error.find("quota reservation unavailable") != std::string::npos);
        assert(!out.staging_exists);
        assert(!out.object_exists);

        fi::plan unqualified;
        unqualified.qualified = false;
        proof_fixture proof2;
        quota_fixture quota2;
        source_fixture source2;
        out = run(f, unqualified, distinct_pin(), expected, proof2, quota2, source2);
        assert(out.result.state == v3::publication::rejected);
        assert(out.result.error.find("durable backend or reservation unavailable") !=
               std::string::npos);
        assert(!out.staging_exists);
        f.assert_untouched();
    }

    // 13. Symlink object and symlink stage are refused without following them.
    {
        const auto object_pin = distinct_pin();
        const auto object = f.root / rkva_snapshot::hex(object_pin.key.data());
        assert(::symlink((f.legacy_v1 / "sentinel").c_str(), object.c_str()) == 0);
        fi::plan schedule;
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        auto out = run(f, schedule, object_pin, expected, proof, quota, source);
        assert(out.result.state == v3::publication::rejected);
        assert(out.result.error.find("refusing overwrite") != std::string::npos);
        assert(!out.staging_exists);
        assert(read_file(f.legacy_v1 / "sentinel") == "legacy-v1");
        fs::remove(object);

        const auto stage_pin = distinct_pin();
        const auto stage =
            f.root / ("tmp-" + rkva_snapshot::hex(stage_pin.transaction.data()));
        assert(::symlink(f.legacy_v2.c_str(), stage.c_str()) == 0);
        fi::plan stage_schedule;
        proof_fixture stage_proof;
        quota_fixture stage_quota;
        source_fixture stage_source;
        out = run(f, stage_schedule, stage_pin, expected, stage_proof, stage_quota,
                  stage_source);
        assert(out.result.state == v3::publication::rejected);
        assert(out.result.error.find("staging already exists") != std::string::npos);
        assert(read_file(f.legacy_v2 / "sentinel") == "legacy-v2");
        fs::remove(stage);
        f.assert_untouched();
    }

    // 14. Queue semantics: one-shot, independent per operation kind, and an
    // unsupported fault kind is refused rather than ignored.
    {
        fi::plan schedule;
        fi::injected_filesystem disk(f.token, f.root, &schedule);
        const auto path = f.root / "queue-file";
        write_file(path, "queue");
        schedule.file_sync = {fi::fault::eio, fi::fault::none};
        std::string error;
        assert(!disk.sync_file(path, error));
        assert(disk.sync_file(path, error));
        assert(schedule.file_sync_calls == 2);
        assert(schedule.dir_sync_calls == 0);
        assert(schedule.rename_calls == 0);

        schedule.file_sync.push_back(fi::fault::ack_lost); // wrong queue kind
        assert(!disk.sync_file(path, error));
        assert(error.find("unsupported file-sync fault") != std::string::npos);
        fs::remove(path);
    }

    // 15. A second writer cannot enter while the lease is held, and a released
    // lease does not make the stale stage adoptable.
    {
        std::string error;
        auto blocked = authority::lease::try_acquire(f.parent, error);
        assert(!blocked);

        const auto case_pin = distinct_pin();
        fi::plan schedule;
        schedule.file_sync = {fi::fault::eio};
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        const auto out = run(f, schedule, case_pin, expected, proof, quota, source);
        expect_staging_failure(f, out, "chunk durable sync failed");
        const auto stage = out.staging;
        const auto stage_digest = tree_digest(stage);

        f.token.reset();
        auto reopened = authority::lease::try_acquire(f.parent, error);
        assert(reopened);
        f.token = reopened;

        fi::plan retry_schedule;
        proof_fixture retry_proof;
        quota_fixture retry_quota;
        source_fixture retry_source;
        auto retry = run(f, retry_schedule, case_pin, expected, retry_proof,
                         retry_quota, retry_source);
        assert(retry.result.state == v3::publication::rejected);
        assert(retry.result.error.find("staging already exists") != std::string::npos);
        assert(tree_digest(stage) == stage_digest);
        f.assert_untouched();
    }

    // 16. The real production writer is still fail-closed after this change.
    {
        fi::plan schedule;
        proof_fixture proof;
        quota_fixture quota;
        source_fixture source;
        fi::injected_filesystem disk(f.token, f.root, &schedule);
        const auto result = v3::write(f.root, f.token, pin, expected, &proof, &quota, &disk,
                                      &source);
        assert(result.state == v3::publication::rejected);
        assert(result.error.find("concrete authenticated pin and metadata proof unavailable") !=
               std::string::npos);
        assert(proof.pending_calls == 0);
        assert(quota.calls == 0);
        assert(source.read_calls == 0);
        assert(schedule.file_sync_calls == 0);
        assert(schedule.dir_sync_calls == 0);
        assert(schedule.rename_calls == 0);
    }

    std::cout << "v3 durable staging seam: every fault path PASS\n";
    return 0;
}
