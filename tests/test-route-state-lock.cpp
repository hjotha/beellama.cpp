#include "server-route-state.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

static void require_lock_test(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

#ifndef _WIN32
static void write_byte(int fd) {
    const char value = '1';
    require_lock_test(::write(fd, &value, 1) == 1, "lock test ready write failed");
}

static void read_byte(int fd, char expected = '1') {
    char value = 0;
    require_lock_test(::read(fd, &value, 1) == 1 && value == expected, "lock test ready read failed");
}

static pid_t spawn_reader(const std::string & path, int ready_write, int release_read, bool die) {
    const pid_t pid = ::fork();
    require_lock_test(pid >= 0, "lock test fork failed");
    if (pid == 0) {
        server_route_state_lease reader(path, server_route_state_lock_mode::reference);
        if (!reader.acquired()) {
            _exit(2);
        }
        write_byte(ready_write);
        if (die) {
            _exit(0); // kernel closes the lock descriptors and releases the reference
        }
        read_byte(release_read);
        _exit(0); // the descriptor is released by process exit after the live-reader assertion
    }
    return pid;
}

static void wait_ok(pid_t pid) {
    int status = 0;
    require_lock_test(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "lock test child failed");
}
#endif

int main() {
    try {
        const char * root_raw = std::getenv("TMPDIR");
        require_lock_test(root_raw && *root_raw, "TMPDIR must be an explicit disk directory");
        const auto run_id = std::chrono::steady_clock::now().time_since_epoch().count();
        const std::filesystem::path root = std::filesystem::path(root_raw) /
                ("route-state-lock-test-" + std::to_string(run_id));
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
        require_lock_test(!ec && std::filesystem::create_directories(root, ec) && !ec,
                "lock test directory creation failed");

        const std::string live_path = (root / "live.bin").string();
        std::ofstream(live_path).put('x');

#ifndef _WIN32
        int ready[2] = {-1, -1};
        int release[2] = {-1, -1};
        require_lock_test(::pipe(ready) == 0 && ::pipe(release) == 0, "lock test pipe failed");
        const pid_t live = spawn_reader(live_path, ready[1], release[0], false);
        ::close(ready[1]);
        ::close(release[0]);
        read_byte(ready[0]);
        ::close(ready[0]);
        require_lock_test(!server_route_state_remove_if_unreferenced(live_path),
                "live reader did not block eviction");
        write_byte(release[1]);
        ::close(release[1]);
        wait_ok(live);
        require_lock_test(server_route_state_remove_if_unreferenced(live_path),
                "released reader did not allow eviction");

        const std::string dead_path = (root / "dead.bin").string();
        std::ofstream(dead_path).put('x');
        int dead_ready[2] = {-1, -1};
        int dead_release[2] = {-1, -1};
        require_lock_test(::pipe(dead_ready) == 0 && ::pipe(dead_release) == 0, "dead-reader pipe failed");
        const pid_t dead = spawn_reader(dead_path, dead_ready[1], dead_release[0], true);
        ::close(dead_ready[1]);
        ::close(dead_ready[0]);
        ::close(dead_release[0]);
        ::close(dead_release[1]);
        wait_ok(dead);
        require_lock_test(server_route_state_remove_if_unreferenced(dead_path),
                "dead reader left an unrecoverable lock");

        const std::string exec_path = (root / "exec.bin").string();
        std::ofstream(exec_path).put('x');
        int exec_ready[2] = {-1, -1};
        require_lock_test(::pipe(exec_ready) == 0, "exec pipe failed");
        const pid_t exec_pid = ::fork();
        require_lock_test(exec_pid >= 0, "exec fork failed");
        if (exec_pid == 0) {
            server_route_state_lease reader(exec_path, server_route_state_lock_mode::reference);
            if (!reader.acquired()) { _exit(2); }
            write_byte(exec_ready[1]);
            const std::string command = "printf 2 >&" + std::to_string(exec_ready[1]) + "; sleep 2";
            ::execl("/bin/sh", "sh", "-c", command.c_str(), (char *) nullptr);
            _exit(3);
        }
        ::close(exec_ready[1]);
        read_byte(exec_ready[0]);
        read_byte(exec_ready[0], '2'); // written by the post-exec shell
        ::close(exec_ready[0]);
        // O_CLOEXEC is required: sleep remains alive, but it must not inherit the lock.
        require_lock_test(server_route_state_remove_if_unreferenced(exec_path),
                "lock descriptor leaked across exec; O_CLOEXEC missing");
        wait_ok(exec_pid);

        const std::string first_path = (root / "first.bin").string();
        const std::string second_path = (root / "second.bin").string();
        std::ofstream(first_path).put('x');
        std::ofstream(second_path).put('x');
        int ready_a[2] = {-1, -1}, release_a[2] = {-1, -1};
        int ready_b[2] = {-1, -1}, release_b[2] = {-1, -1};
        require_lock_test(::pipe(ready_a) == 0 && ::pipe(release_a) == 0 &&
                ::pipe(ready_b) == 0 && ::pipe(release_b) == 0, "limit pipes failed");
        const pid_t reader_a = spawn_reader(first_path, ready_a[1], release_a[0], false);
        const pid_t reader_b = spawn_reader(second_path, ready_b[1], release_b[0], false);
        ::close(ready_a[1]); ::close(release_a[0]);
        ::close(ready_b[1]); ::close(release_b[0]);
        read_byte(ready_a[0]); read_byte(ready_b[0]);
        ::close(ready_a[0]); ::close(ready_b[0]);
        // With a count budget of two, no protected candidate is removable; the
        // caller must reject/retain the new publication rather than evicting a
        // live snapshot. This uses the exact helper called by the LRU.
        require_lock_test(!server_route_state_remove_if_unreferenced(first_path) &&
                !server_route_state_remove_if_unreferenced(second_path),
                "protected candidates were evictable under saturated limits");
        write_byte(release_a[1]); write_byte(release_b[1]);
        ::close(release_a[1]); ::close(release_b[1]);
        wait_ok(reader_a); wait_ok(reader_b);
        require_lock_test(server_route_state_remove_if_unreferenced(first_path) &&
                server_route_state_remove_if_unreferenced(second_path),
                "released saturated candidates did not recover");

        const std::filesystem::path bad_store = root / "bad-store";
        std::filesystem::create_directories(bad_store);
        std::filesystem::create_directory(bad_store / ".llama-kv-snapshot-store.lock");
        const std::string bad_state = (bad_store / "unpublished.bin").string();
        std::ofstream(bad_state).put('x');
        server_route_state_store_lock failed_store(bad_store.string(),
                server_route_state_store_lock_mode::exclusive_try);
        require_lock_test(!failed_store.acquired(), "invalid store lock path reported acquired");
        require_lock_test(!server_route_state_remove_if_unreferenced(bad_state) &&
                std::filesystem::exists(bad_state),
                "invalid store lock path removed a snapshot");

        for (int i = 0; i < 128; ++i) {
            const std::string path = (root / ("stripe-" + std::to_string(i) + ".bin")).string();
            std::ofstream(path).put('x');
            require_lock_test(server_route_state_remove_if_unreferenced(path),
                    "striped lock cleanup probe failed");
        }
        size_t lock_files = 0;
        for (const auto & entry : std::filesystem::directory_iterator(root)) {
            const std::string name = entry.path().filename().string();
            if (name.find(".llama-kv-snapshot-ref-") == 0) {
                ++lock_files;
            }
        }
        require_lock_test(lock_files <= 64, "per-snapshot lock identities were not striped");
#else
        const char * pass_message = "PASS: Windows build gate; kernel lock process assertions are POSIX-only\n";
#endif

#ifndef _WIN32
        const char * pass_message = "PASS: live-reader eviction block, dead-reader recovery, CLOEXEC exec recovery, "
                      "saturated protected candidates, and bounded striped lock identities\n";
#endif
        std::filesystem::remove_all(root, ec);
        require_lock_test(!ec, "lock test directory cleanup failed");
        std::cout << pass_message;
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
