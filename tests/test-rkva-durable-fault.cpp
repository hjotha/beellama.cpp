// CPU unit tests for the test-only durable sync/rename/staging fault injector.
//
// This test never signs, deploys or contacts a console. It exercises the
// injected_filesystem contract over a real temporary POSIX directory:
//
//   * fault queues are consumed exactly once per matching call;
//   * a failed file/directory sync never reports success;
//   * a no-replace rename never overwrites an existing object and leaves the
//     staging tree intact;
//   * a rename EIO is explicitly uncertain, never durable;
//   * an applied-but-unacknowledged rename (ack lost / crash model) returns
//     "uncertain" and never "durable", while the published bytes are intact;
//   * a parent-directory sync failure after apply is "uncertain", not durable;
//   * one-shot injection does not leak into later calls.
//
// The closed v3 writer is deliberately NOT reachable here (see
// docs/rkva-durable-fault-injection.md). These tests prove the injector's
// semantics, not production publication or power-loss durability.

#ifndef RKVA_DURABLE_FAULT_TESTING
#define RKVA_DURABLE_FAULT_TESTING
#endif
#include "rkva-durable-fault.h"

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

fs::path make_staging(const fs::path & root, const std::string & name,
                      const std::string & payload) {
    const fs::path staging = root / name;
    assert(fs::create_directory(staging));
    write_file(staging / "manifest-v2.bin", "manifest");
    write_file(staging / "n-0.kv", payload);
    return staging;
}

struct fixture {
    fs::path parent;
    fs::path root;
    std::shared_ptr<authority::lease> token;

    fixture() {
        char pattern[] = "/tmp/rkva-durable-fault-XXXXXX";
        char * made = mkdtemp(pattern);
        assert(made != nullptr);
        parent = fs::path(made);
        root = parent / "rkva-snapshots-v3";
        assert(fs::create_directory(root));
        std::string error;
        token = authority::lease::try_acquire(parent, error);
        assert(token);
    }
    ~fixture() {
        token.reset();
        std::error_code ec;
        fs::remove_all(parent, ec);
    }
};

} // namespace

int main() {
    fixture f;

    // 1. qualified() is explicit; an unqualified backend is visible to callers.
    {
        fi::plan schedule;
        schedule.qualified = false;
        fi::injected_filesystem disk(f.token, f.root, &schedule);
        assert(!disk.qualified());
        schedule.qualified = true;
        assert(disk.qualified());
    }

    // 2. Happy path: a real no-replace move is durable and leaves no staging.
    {
        fi::plan schedule;
        fi::injected_filesystem disk(f.token, f.root, &schedule);
        const auto staging = make_staging(f.root, "tmp-happy", "payload-happy");
        const auto object = f.root / "object-happy";
        std::string error;
        assert(disk.sync_file(staging / "n-0.kv", error));
        assert(disk.sync_directory(staging, error));
        assert(disk.rename_noreplace_and_sync(staging, object, error) ==
               v3::move_result::durable);
        assert(!fs::exists(staging));
        assert(read_file(object / "n-0.kv") == "payload-happy");
        assert(schedule.file_sync_calls == 1);
        assert(schedule.dir_sync_calls == 2); // explicit staging + rename parent
        assert(schedule.rename_calls == 1);
        assert(!schedule.crashed);
    }

    // 3. Existing destination: refuse, keep both sides byte-identical.
    {
        fi::plan schedule;
        fi::injected_filesystem disk(f.token, f.root, &schedule);
        const auto staging = make_staging(f.root, "tmp-existing", "new");
        const auto object = f.root / "object-existing";
        assert(fs::create_directory(object));
        write_file(object / "n-0.kv", "old");
        std::string error;
        assert(disk.rename_noreplace_and_sync(staging, object, error) ==
               v3::move_result::no_change);
        assert(!error.empty());
        assert(fs::exists(staging / "n-0.kv"));
        assert(read_file(object / "n-0.kv") == "old");
        assert(schedule.rename_calls == 1);
    }

    // 4. Injected racing writer with no actual destination: still refused.
    {
        fi::plan schedule;
        schedule.rename.push_back(fi::fault::refuse_existing);
        fi::injected_filesystem disk(f.token, f.root, &schedule);
        const auto staging = make_staging(f.root, "tmp-race", "race");
        const auto object = f.root / "object-race";
        std::string error;
        assert(disk.rename_noreplace_and_sync(staging, object, error) ==
               v3::move_result::no_change);
        assert(fs::exists(staging));
        assert(!fs::exists(object));
    }

    // 5. Injected benign rename error: nothing moves, outcome is no_change.
    {
        fi::plan schedule;
        schedule.rename.push_back(fi::fault::os_error);
        fi::injected_filesystem disk(f.token, f.root, &schedule);
        const auto staging = make_staging(f.root, "tmp-oserror", "oserror");
        const auto object = f.root / "object-oserror";
        std::string error;
        assert(disk.rename_noreplace_and_sync(staging, object, error) ==
               v3::move_result::no_change);
        assert(fs::exists(staging / "n-0.kv"));
        assert(!fs::exists(object));
    }

    // 5b. Rename EIO: the applied state is unknown, so the call is uncertain,
    // the source stays intact in this model, and no object is reported durable.
    {
        fi::plan schedule;
        schedule.rename.push_back(fi::fault::rename_eio);
        fi::injected_filesystem disk(f.token, f.root, &schedule);
        const auto staging = make_staging(f.root, "tmp-renameeio", "renameeio");
        const auto object = f.root / "object-renameeio";
        std::string error;
        assert(disk.rename_noreplace_and_sync(staging, object, error) ==
               v3::move_result::uncertain);
        assert(error.find("rename EIO") != std::string::npos);
        assert(fs::exists(staging / "n-0.kv"));
        assert(!fs::exists(object));
        assert(schedule.rename_calls == 1);
    }

    // 6. Ack lost: applied bytes exist, but the call must return uncertain.
    {
        fi::plan schedule;
        schedule.rename.push_back(fi::fault::ack_lost);
        fi::injected_filesystem disk(f.token, f.root, &schedule);
        const auto staging = make_staging(f.root, "tmp-acklost", "acklost");
        const auto object = f.root / "object-acklost";
        std::string error;
        assert(disk.rename_noreplace_and_sync(staging, object, error) ==
               v3::move_result::uncertain);
        assert(error.find("ack_lost") != std::string::npos);
        assert(!fs::exists(staging));
        assert(read_file(object / "n-0.kv") == "acklost");
        assert(!schedule.crashed);
    }

    // 7. Crash after apply: the model process is marked dead, bytes remain.
    {
        fi::plan schedule;
        schedule.rename.push_back(fi::fault::crash_after_apply);
        fi::injected_filesystem disk(f.token, f.root, &schedule);
        const auto staging = make_staging(f.root, "tmp-crash", "crash");
        const auto object = f.root / "object-crash";
        std::string error;
        assert(disk.rename_noreplace_and_sync(staging, object, error) ==
               v3::move_result::uncertain);
        assert(schedule.crashed);
        assert(read_file(object / "n-0.kv") == "crash");
    }

    // 8. Parent directory sync failure after apply: uncertain, never durable.
    {
        fi::plan schedule;
        schedule.dir_sync.push_back(fi::fault::eio); // first dir sync is the parent
        fi::injected_filesystem disk(f.token, f.root, &schedule);
        const auto staging = make_staging(f.root, "tmp-dirsync", "dirsync");
        const auto object = f.root / "object-dirsync";
        std::string error;
        assert(disk.rename_noreplace_and_sync(staging, object, error) ==
               v3::move_result::uncertain);
        assert(error.find("parent sync failed") != std::string::npos);
        assert(fs::exists(object / "n-0.kv"));
    }

    // 9. File sync failure: false, one-shot, bytes readable, next call works.
    {
        fi::plan schedule;
        schedule.file_sync.push_back(fi::fault::eio);
        fi::injected_filesystem disk(f.token, f.root, &schedule);
        const auto staging = make_staging(f.root, "tmp-filesync", "filesync");
        std::string error;
        assert(!disk.sync_file(staging / "n-0.kv", error));
        assert(error.find("injected file sync failure") != std::string::npos);
        assert(read_file(staging / "n-0.kv") == "filesync");
        assert(disk.sync_file(staging / "n-0.kv", error));
        assert(schedule.file_sync_calls == 2);
    }

    // 10. Directory sync failure: false, one-shot, later call succeeds.
    {
        fi::plan schedule;
        schedule.dir_sync.push_back(fi::fault::eio);
        fi::injected_filesystem disk(f.token, f.root, &schedule);
        std::string error;
        assert(!disk.sync_directory(f.root, error));
        assert(error.find("injected directory sync failure") != std::string::npos);
        assert(disk.sync_directory(f.root, error));
        assert(schedule.dir_sync_calls == 2);
    }

    // 11. Symlinks are refused even when no fault is scheduled.
    {
        fi::plan schedule;
        fi::injected_filesystem disk(f.token, f.root, &schedule);
        const auto target = f.root / "link-target";
        write_file(target, "x");
        const auto link = f.root / "link";
        assert(::symlink(target.c_str(), link.c_str()) == 0);
        std::string error;
        assert(!disk.sync_file(link, error));
        assert(error.find("file sync") != std::string::npos);
    }

    // 12. Unsupported fault kinds for a queue are refused, not ignored.
    {
        fi::plan schedule;
        schedule.file_sync.push_back(fi::fault::ack_lost);
        fi::injected_filesystem disk(f.token, f.root, &schedule);
        std::string error;
        assert(!disk.sync_file(f.root / "does-not-exist", error));
        assert(error.find("unsupported file-sync fault") != std::string::npos);
    }

    std::cout << "durable fault injection semantics: PASS\n";
    return 0;
}
