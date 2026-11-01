// Phase 5 listener session authority (activation item 4). Test seam only: the
// authority is minted for one accepted socket, its issued challenge and the
// exact state mutex. There is no listener call site and no advertisement.
#include "../tools/server/server-rkva-session-authority.h"
#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace sa = rkva_session_authority;
namespace v3 = rkva_managed_framing_v3;

static v3::id id(uint8_t n) { v3::id value{}; value.fill(n); return value; }

int main() {
    std::string error;
    std::recursive_mutex state_mutex;

    // Default off: the compile gate is in, but the runtime opt-in is not.
    assert(!sa::authority::available());
    int pair[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
    sa::authority::binding binding;
    binding.accepted_fd = pair[0];
    binding.connection_cookie = 0x51;
    binding.peer = "test-peer";
    binding.challenge = id(0x31);
    binding.state_mutex = &state_mutex;
    assert(!sa::authority::mint(binding, error) && !error.empty());

    assert(setenv("RKVA_MANAGED_SNAPSHOT_ACTIVATION", "1", 1) == 0);
    assert(sa::authority::available());

    // A pipe is not a connected socket and is refused.
    int pipe_fds[2];
    assert(pipe(pipe_fds) == 0);
    auto bad = binding;
    bad.accepted_fd = pipe_fds[0];
    assert(!sa::authority::mint(bad, error) && error.find("socket") != std::string::npos);
    close(pipe_fds[0]);
    close(pipe_fds[1]);

    bad = binding;
    bad.state_mutex = nullptr;
    assert(!sa::authority::mint(bad, error));
    bad = binding;
    bad.connection_cookie = 0;
    assert(!sa::authority::mint(bad, error));
    bad = binding;
    bad.challenge = {};
    assert(!sa::authority::mint(bad, error));
    bad = binding;
    bad.peer.clear();
    assert(!sa::authority::mint(bad, error));
    bad = binding;
    bad.accepted_fd = -1;
    assert(!sa::authority::mint(bad, error));

    auto authority = sa::authority::mint(binding, error);
    assert(authority && authority->live());
    v3::session session{id(0x11), id(0x77), id(0x71), id(0x72), 7,
                        RKVA_PROTO_MANAGED_REGISTRY_V1, true};
    // Not bound yet.
    assert(!authority->held_and_bound(session));
    assert(!authority->bind_session(v3::session{}, error));
    v3::session closed = session;
    closed.authenticated = false;
    assert(!authority->bind_session(closed, error));
    assert(authority->bind_session(session, error));

    // The mutex stays held after a successful check, so the publication that
    // follows cannot race another writer.
    assert(authority->held_and_bound(session));
    {
        std::promise<void> started, finished;
        auto a = started.get_future(), b = finished.get_future();
        std::thread thread([&] {
            started.set_value();
            std::unique_lock<std::recursive_mutex> wait(state_mutex);
            finished.set_value();
        });
        a.wait();
        assert(b.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout);
        authority->release();
        b.wait();
        thread.join();
    }
    assert(authority->held_and_bound(session));
    authority->release();

    // A different session, transcript or challenge is not the bound session.
    v3::session other = session;
    other.transcript = id(0x79);
    assert(!authority->held_and_bound(other));
    other = session;
    other.peer_challenge = id(0x73);
    assert(!authority->held_and_bound(other));
    other = session;
    other.auth_epoch = 8;
    assert(!authority->held_and_bound(other));
    // A failed check must not leave the mutex held.
    {
        std::unique_lock<std::recursive_mutex> free_now(state_mutex, std::try_to_lock);
        assert(free_now.owns_lock());
    }

    // Contention fails closed: another thread holding the exact state mutex
    // makes the authority unusable instead of waiting. A recursive mutex can
    // only be contended from a different thread.
    {
        std::promise<void> locked, done;
        auto wait_locked = locked.get_future();
        auto wait_done = done.get_future();
        std::thread thread([&] {
            std::unique_lock<std::recursive_mutex> hold(state_mutex);
            locked.set_value();
            wait_done.wait();
        });
        wait_locked.wait();
        assert(!authority->held_and_bound(session));
        done.set_value();
        thread.join();
    }
    assert(authority->held_and_bound(session));
    authority->release();

    // A closed descriptor invalidates the binding even if the number is later
    // reused by an unrelated resource.
    close(pair[0]);
    assert(!authority->held_and_bound(session));
    const int reused = open("/dev/null", O_RDONLY);
    assert(reused >= 0);
    assert(!authority->held_and_bound(session));
    close(reused);

    // A forked child may not reuse the inherited authority.
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        _exit(!authority->held_and_bound(session) && !authority->live() ? 0 : 5);
    }
    int status = 0;
    assert(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);

    authority->invalidate();
    assert(!authority->live());
    assert(!authority->held_and_bound(session));
    close(pair[1]);

    // The authority never advertises anything: it has no capability/op API and
    // the managed negotiation predicate stays closed without the full set.
    assert(!v3::negotiated(0, 0, 0));
    rkva_hello hello{};
    assert(hello.proto_flags == 0);
    assert(unsetenv("RKVA_MANAGED_SNAPSHOT_ACTIVATION") == 0);
    assert(!sa::authority::available());

    std::cout << "session authority: accept/challenge/mutex binding, contention, fork, fd reuse, default off: PASS\n";
    return 0;
}
