// Phase 2 concrete managed-store lease test (Linux/POSIX only).
//
// Exercises the cross-process authority token in isolation: private-directory
// requirement, process exclusion, shared-ownership identity, operation mutex,
// reacquisition and lock-path replacement. No SQLite, v3 pin proof, server,
// GPU, Xbox or power-loss claim is involved.

#include "../ggml/include/rkva-managed-store-authority.h"
#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
namespace authority = rkva_managed_store;

int main(int argc, char ** argv) {
    std::string error;
    if (argc == 3 && std::string(argv[1]) == "--try") {
        auto token = authority::lease::try_acquire(argv[2], error);
        return token ? 0 : 3;
    }

    char tmp[] = "/tmp/rkva-phase2-lease-XXXXXX";
    char * name = mkdtemp(tmp);
    assert(name != nullptr);
    const fs::path parent(name);
    const fs::path other = parent / "other";
    const fs::path insecure = parent / "insecure";
    assert(fs::create_directory(other));
    assert(fs::create_directory(insecure));
    fs::permissions(insecure, fs::perms::owner_all | fs::perms::group_all | fs::perms::others_all);

    // A group/other-writable directory is not an acceptable private store.
    assert(!authority::lease::try_acquire(insecure, error));
    assert(!authority::lease::try_acquire(parent / "absent", error));

    auto token = authority::lease::try_acquire(parent, error);
    assert(token && token->valid_for(parent));
    assert(token->directory() == fs::canonical(parent));

    // A separately owned shared_ptr alias with the same raw pointer must not
    // be accepted as the concrete token.
    auto alias = std::shared_ptr<authority::lease>(token.get(), [](authority::lease *) {});
    assert(!token->owns(alias));
    assert(token->owns(token));
    assert(!token->valid_for(other));

    // Threads sharing the same open-file-description still serialize through
    // the token's recursive operation mutex, rather than relying on flock.
    {
        auto held = token->enter();
        assert(held.owns_lock());
        std::promise<void> started;
        std::promise<void> finished;
        auto started_future = started.get_future();
        auto finished_future = finished.get_future();
        std::thread thread([&]() {
            started.set_value();
            auto next = token->enter();
            assert(next.owns_lock());
            finished.set_value();
        });
        started_future.wait();
        assert(finished_future.wait_for(std::chrono::milliseconds(50)) ==
               std::future_status::timeout);
        held.unlock();
        finished_future.wait();
        thread.join();
    }

    // A different process cannot acquire the same store while it is held.
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        execl(argv[0], argv[0], "--try", parent.c_str(), (char *) nullptr);
        _exit(127);
    }
    int status = 0;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 3);

    // A different directory has an independent token.
    auto unrelated = authority::lease::try_acquire(other, error);
    assert(unrelated && unrelated->valid_for(other));
    assert(!unrelated->valid_for(parent));

    // Replacing the lock pathname invalidates the outstanding authority even
    // though the old open descriptor still exists.
    fs::rename(parent / ".rkva-managed-store.lock", parent / "old-lock");
    {
        std::ofstream replacement(parent / ".rkva-managed-store.lock");
        replacement << "new";
    }
    assert(!token->valid_for(parent));

    // Releasing the token lets a separate process acquire the same store.
    token.reset();
    unrelated.reset();
    child = fork();
    assert(child >= 0);
    if (child == 0) {
        execl(argv[0], argv[0], "--try", parent.c_str(), (char *) nullptr);
        _exit(127);
    }
    status = 0;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);

    fs::remove_all(parent);
    std::cout << "managed store lease authority: privacy, exclusion, identity, "
                 "mutex and reacquisition: PASS\n";
    return 0;
}
