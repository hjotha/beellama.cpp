// Phase 5 single store authority (activation item 6). One RAII authority holds
// the journal, orchestrator, anchored SQLite and object-v3/quota locks in the
// documented order, and every composed component runs delegated (no second
// lock). CPU-only POSIX test.
#include "../tools/server/server-rkva-managed-authority.h"
#include "../tools/server/server-rkva-managed-orchestrator.h"
#include "../ggml/include/rkva-anchored-sqlite.h"
#include "../ggml/include/rkva-snapshot-digest.h"
#include "../ggml/src/ggml-remote-attn/rkva-v3-quota.h"
#include "../tools/xbox/uwp/rkva/rkva-managed-object-v3.h"
#undef NDEBUG
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace au = rkva_managed_authority;
namespace lease = rkva_managed_store;
namespace hj = rkva_host_journal;
namespace mr = rkva_managed_registry;
namespace mo = rkva_managed_orchestrator;
namespace db = rkva_anchored_sqlite;
namespace v3obj = rkva_managed_object_v3;

static void write_file(const fs::path & path, const std::string & data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(data.data(), std::streamsize(data.size()));
    assert(out.good());
}

static std::string hex_id(uint8_t seed) {
    std::array<uint8_t, 32> value{};
    value.fill(seed);
    return rkva_snapshot::hex(value.data());
}

int main(int argc, char ** argv) {
    alarm(180);
    std::string error;
    if (argc == 3 && std::string(argv[1]) == "--try") {
        auto authority = au::store::try_acquire(argv[2], error);
        return authority ? 0 : 3;
    }

    char tmp[] = "/tmp/rkva-managed-authority-XXXXXX";
    char * name = mkdtemp(tmp);
    assert(name);
    const fs::path base(name);
    const fs::path store_dir = base / "store";
    assert(fs::create_directory(store_dir));
    const std::string owner = hex_id(1);

    auto authority = au::store::try_acquire(store_dir, error);
    assert(authority && error.empty());
    assert(authority->valid());
    assert(authority->directory() == fs::canonical(store_dir));
    assert(authority->concrete() && authority->concrete()->valid_for(store_dir));
    assert(authority->valid_for(store_dir));

    // The same authority token serializes threads through the shared section.
    {
        auto held = authority->enter(error);
        assert(held.owns_lock());
        std::promise<void> started, finished;
        auto a = started.get_future(), b = finished.get_future();
        std::thread thread([&] {
            started.set_value();
            std::string why;
            auto next = authority->enter(why);
            assert(next.owns_lock());
            finished.set_value();
        });
        a.wait();
        assert(b.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout);
        held.unlock();
        b.wait();
        thread.join();
    }

    // Only one holder exists per store, in this process and across processes.
    assert(!au::store::try_acquire(store_dir, error));
    assert(error.find("busy") != std::string::npos);
    assert(!lease::lease::try_acquire(store_dir, error));
    {
        // A legacy journal writer would block on this exact lock file; the
        // authority already holds it, so a non-blocking probe must fail.
        int fd = open((store_dir / "host-journal.lock").c_str(), O_RDWR | O_NOFOLLOW | O_CLOEXEC);
        assert(fd >= 0);
        assert(flock(fd, LOCK_EX | LOCK_NB) != 0 && (errno == EWOULDBLOCK || errno == EAGAIN));
        close(fd);
        fd = open((store_dir / ".managed-orchestrator.lock").c_str(), O_RDWR | O_NOFOLLOW | O_CLOEXEC);
        assert(fd >= 0);
        assert(flock(fd, LOCK_EX | LOCK_NB) != 0 && (errno == EWOULDBLOCK || errno == EAGAIN));
        close(fd);
    }

    // Delegated components never take their own lock: these calls would deadlock
    // against the authority-held lock files if delegation were broken.
    auto journal = std::make_unique<hj::journal>(authority->journal(owner));
    assert(journal->delegated());
    assert(journal->initialize(error));
    assert(journal->prepare(hex_id(2), hex_id(3), hex_id(4), error));
    hj::snapshot snapshot;
    assert(journal->inspect(snapshot, error) && snapshot.transactions.size() == 1);
    assert(journal->host_published(hex_id(2), "/fixture-managed", std::string(64, 'c'), 4096, error));

    // Anchored SQLite uses the very same concrete lease token.
    db::id db_owner{}, db_epoch{};
    db::secret db_key{};
    db_owner.fill(1);
    db_epoch.fill(2);
    db_key.fill(3);
    db::backend sqlite(store_dir.string(), db_owner, db_epoch, db_key);
    assert(sqlite.provision(authority->concrete(), error));
    db::state state;
    assert(sqlite.inspect(authority->concrete(), state, error) && state.version == 0);
    assert(!sqlite.inspect({}, state, error));

    // The read-only v3 quota scan is charged under the same section.
    assert(fs::create_directory(store_dir / "rkva-snapshots-v3"));
    {
        auto held = authority->enter(error);
        assert(held.owns_lock());
        const auto usage = rkva_v3_quota::scan(authority->directory());
        assert(usage.objects == 0);
    }

    // Object v3 accepts the same token (it still refuses without a real pin,
    // filesystem and source; that is item 2/3, not this test's subject).
    v3obj::pin pin{};
    v3obj::expected geometry{};
    const fs::path objects = store_dir / "rkva-snapshots-v3";
    auto rejected = v3obj::write(objects, authority->concrete(), pin, geometry, nullptr, nullptr, nullptr, nullptr);
    assert(rejected.state == v3obj::publication::rejected);
    assert(rejected.error.find("filesystem/store authority") != std::string::npos);

    // Losing a lock pathname invalidates the whole authority before any
    // SQLite/journal/object operation can proceed.
    assert(unlink((store_dir / "host-journal.lock").c_str()) == 0);
    assert(!authority->valid());
    assert(!authority->enter(error).owns_lock());
    assert(!authority->valid_for(store_dir));
    assert(!journal->inspect(snapshot, error));
    assert(authority->concrete()->valid_for(store_dir));
    // The delegated journal keeps the authority (and its lock files) alive.
    journal.reset();
    authority.reset();

    // A fresh acquisition creates a new, valid lock and reuses no old state.
    authority = au::store::try_acquire(store_dir, error);
    assert(authority);
    {
        auto reopened = authority->journal(owner);
        assert(reopened.inspect(snapshot, error) && snapshot.transactions.size() == 1);
    }
    authority.reset();

    // Symlinked and non-private stores fail closed without touching anything.
    const fs::path symlinked = base / "symlinked";
    assert(fs::create_directory(symlinked));
    assert(symlink("/etc/passwd", (symlinked / ".rkva-managed-store.lock").c_str()) == 0);
    assert(!au::store::try_acquire(symlinked, error));
    const fs::path journal_symlink = base / "journal-symlink";
    assert(fs::create_directory(journal_symlink));
    assert(symlink("/etc/passwd", (journal_symlink / "host-journal.lock").c_str()) == 0);
    assert(!au::store::try_acquire(journal_symlink, error));
    assert(error.find("symlink") != std::string::npos);
    assert(!au::store::try_acquire(base / "missing", error));
    error.clear();
    auto world = au::store::try_acquire(base, error); // 0700 temporary root: private
    assert(world);
    world.reset();
    const fs::path public_dir = base / "public";
    assert(fs::create_directory(public_dir));
    fs::permissions(public_dir, fs::perms::owner_all | fs::perms::group_all | fs::perms::others_all);
    assert((fs::status(public_dir).permissions() & fs::perms::others_write) == fs::perms::others_write);
    assert(!au::store::try_acquire(public_dir, error));

    // A forked child cannot reuse the inherited token and cannot acquire it.
    authority = au::store::try_acquire(store_dir, error);
    assert(authority);
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        std::string why;
        auto inherited = authority->enter(why);
        _exit(!inherited.owns_lock() && !authority->valid() ? 0 : 5);
    }
    int status = 0;
    assert(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    child = fork();
    assert(child >= 0);
    if (child == 0) {
        execl(argv[0], argv[0], "--try", store_dir.c_str(), (char *) nullptr);
        _exit(127);
    }
    status = 0;
    assert(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 3);
    authority.reset();

    // Process death leaves no stale lock: the kernel releases every descriptor.
    const fs::path crash_dir = base / "crash";
    assert(fs::create_directory(crash_dir));
    child = fork();
    assert(child >= 0);
    if (child == 0) {
        auto held = au::store::try_acquire(crash_dir, error);
        _exit(held ? 0 : 6);
    }
    status = 0;
    assert(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    auto recovered = au::store::try_acquire(crash_dir, error);
    assert(recovered);
    recovered.reset();

    // End-to-end orchestrator publication under the one authority: journal,
    // orchestrator, peer registry and host staging compose without a second
    // independent managed lock.
    const fs::path host_dir = base / "orch-host";
    const fs::path peer_dir = base / "orch-peer";
    const fs::path peer_objects = base / "orch-objects";
    for (const auto & dir : {host_dir, peer_dir, peer_objects}) assert(fs::create_directory(dir));
    write_file(peer_objects / "sentinel", "preserve");
    auto orch_authority = au::store::try_acquire(host_dir, error);
    assert(orch_authority);
    auto orch_journal = std::make_unique<hj::journal>(orch_authority->journal(owner));
    assert(orch_journal->initialize(error));
    mr::registry peer(peer_dir.string(), owner);
    assert(peer.initialize(error));
    const std::string payload = "managed snapshot payload bytes";
    const auto digest = rkva_snapshot::digest(payload.data(), payload.size());
    const std::string key = rkva_snapshot::hex(digest.data());
    const fs::path source = base / "source.bin";
    write_file(source, payload);
    write_file(peer_objects / (key + ".blob"), payload);
    mo::orchestrator orchestrator(*orch_journal, peer, host_dir.string(), peer_objects.string(),
                                  1u << 20, nullptr, orch_authority);
    const std::string tx = hex_id(5), reference = hex_id(6);
    assert(orchestrator.publish(tx, reference, key, source.string(), error));
    hj::snapshot published;
    assert(orch_journal->inspect(published, error));
    assert(published.transactions.size() == 1 && published.transactions[0].state == hj::phase::commit_ack);
    assert(fs::exists(host_dir / (tx + ".managed")));
    assert(orchestrator.release(tx, error));
    assert(!fs::exists(host_dir / (tx + ".managed")));
    assert(fs::exists(host_dir / ("." + tx + ".release")));
    assert(fs::exists(peer_objects / "sentinel"));
    {
        std::ifstream input(peer_objects / "sentinel");
        assert(std::string(std::istreambuf_iterator<char>(input), {}) == "preserve");
    }
    orch_journal.reset();
    orch_authority.reset();

    fs::remove_all(base);
    std::cout << "single store authority: order, delegation, contention, fork, crash, errno, orchestration: PASS\n";
    return 0;
}
