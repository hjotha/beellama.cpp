// Real filesystem tests for native v2 quota, stale staging and writer exclusion.
#include "../tools/xbox/uwp/rkva/rkva-native-store.h"
#undef NDEBUG
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdlib>
#endif
namespace store = rkva_native_store;
namespace fs = std::filesystem;

template <typename Fn>
static bool rejected(const Fn & fn) {
    try { fn(); return false; } catch (const std::exception &) { return true; }
}
static void write(const fs::path & p, const std::string & s) {
    std::ofstream f(p, std::ios::binary); f << s; assert(f.good());
}
#ifdef _WIN32
static DWORD run_child(const fs::path & root, const char * operation) {
    char exe[32768];
    const DWORD length = GetModuleFileNameA(nullptr, exe, sizeof(exe));
    assert(length > 0 && length < sizeof(exe));
    std::string command = "\"" + std::string(exe, length) + "\" " + operation + " \"" + root.string() + "\"";
    STARTUPINFOA info{}; info.cb = sizeof(info); PROCESS_INFORMATION process{};
    assert(CreateProcessA(nullptr, &command[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                          nullptr, nullptr, &info, &process));
    const DWORD result = WaitForSingleObject(process.hProcess, 30000);
    if (result != WAIT_OBJECT_0) {
        TerminateProcess(process.hProcess, 90);
        WaitForSingleObject(process.hProcess, 10000);
    }
    assert(result == WAIT_OBJECT_0);
    DWORD code = 0; assert(GetExitCodeProcess(process.hProcess, &code));
    CloseHandle(process.hThread); CloseHandle(process.hProcess);
    return code;
}
#endif
int main(int argc, char ** argv) {
#ifdef _WIN32
    if (argc == 3 && std::string(argv[1]) == "--try-lock") {
        return rejected([&] { store::WriteGuard guard(argv[2]); }) ? 0 : 1;
    }
    if (argc == 3 && std::string(argv[1]) == "--crash-stage") {
        const fs::path root(argv[2]);
        store::WriteGuard guard(root);
        const auto stage = root / ("tmp-" + std::string(64, 'c'));
        store::create_stage(stage); write(stage / "n-0.kv", "crash");
        std::_Exit(0); // terminate the process without the guard destructor
    }
#else
    (void) argc; (void) argv;
#endif
    const auto name = "rkva-store-test-" + std::to_string(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const auto root = fs::temp_directory_path() / name;
    assert(fs::create_directory(root));
    const std::string key1(64, 'a'), key2(64, 'b');
    const auto published = root / key1, stage = root / ("tmp-" + key2);
    assert(fs::create_directory(published));
    assert(fs::create_directory(stage));
    write(published / "n-0.kv", "1234");
    write(stage / "n-0.kv", "12345");
    {
        store::WriteGuard guard(root);
        store::check_room(root, 1, 10, 10, 3, 10);
        assert(rejected([&] { store::check_room(root, 2, 10, 10, 3, 10); }));
        assert(rejected([&] { store::check_room(root, 1, 100, 100, 2, 10); }));
        assert(rejected([&] { store::create_stage(stage); }));
        assert(fs::file_size(stage / "n-0.kv") == 5);
        assert(fs::file_size(published / "n-0.kv") == 4);
        assert(rejected([&] { store::WriteGuard other(root); }));
#ifdef _WIN32
        assert(run_child(root, "--try-lock") == 0);
#endif
#ifndef _WIN32
        const pid_t pid = fork();
        assert(pid >= 0);
        if (pid == 0) {
            bool busy = rejected([&] { store::WriteGuard other(root); });
            _exit(busy ? 0 : 1);
        }
        int status = 0;
        assert(waitpid(pid, &status, 0) == pid);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
#endif
    }
    assert(!fs::exists(root / ".native-writer-lock"));
    assert(fs::exists(stage / "n-0.kv"));
#ifndef _WIN32
    // A process dying while it owns a stage must leave both its lock and data.
    const auto crashed_stage = root / ("tmp-" + std::string(64, 'c'));
    const pid_t crashed = fork();
    assert(crashed >= 0);
    if (crashed == 0) {
        store::WriteGuard guard(root);
        store::create_stage(crashed_stage);
        write(crashed_stage / "n-0.kv", "crash");
        _exit(0); // simulate process death without C++ destructors
    }
    int crash_status = 0;
    assert(waitpid(crashed, &crash_status, 0) == crashed);
    assert(WIFEXITED(crash_status) && WEXITSTATUS(crash_status) == 0);
    assert(fs::exists(root / ".native-writer-lock"));
    assert(fs::file_size(crashed_stage / "n-0.kv") == 5);
    assert(rejected([&] { store::WriteGuard guard(root); }));
    // Explicit test recovery after waitpid proved the writer has exited.
    fs::remove(root / ".native-writer-lock");
    {
        store::WriteGuard guard(root);
        assert(rejected([&] { store::check_room(root, 1, 100, 100, 3, 10); }));
    }
    assert(fs::file_size(crashed_stage / "n-0.kv") == 5);
    fs::remove_all(crashed_stage);
#else
    const auto crashed_stage = root / ("tmp-" + std::string(64, 'c'));
    assert(run_child(root, "--crash-stage") == 0);
    assert(fs::exists(root / ".native-writer-lock"));
    assert(fs::file_size(crashed_stage / "n-0.kv") == 5);
    assert(rejected([&] { store::WriteGuard guard(root); }));
    fs::remove(root / ".native-writer-lock"); // explicit recovery after child exited
    {
        store::WriteGuard guard(root);
        assert(rejected([&] { store::check_room(root, 1, 100, 100, 3, 10); }));
    }
    assert(fs::file_size(crashed_stage / "n-0.kv") == 5);
    fs::remove_all(crashed_stage);
#endif
    // Stale lock is never stolen, even after the original process is gone.
    assert(fs::create_directory(root / ".native-writer-lock"));
    assert(rejected([&] { store::WriteGuard guard(root); }));
    assert(fs::exists(root / ".native-writer-lock"));
    fs::remove(root / ".native-writer-lock");
    {
        store::WriteGuard guard(root);
        write(root / "unexpected", "x");
        assert(rejected([&] { store::check_room(root, 1, 100, 100, 8, 10); }));
        fs::remove(root / "unexpected");
        fs::create_directory(stage / "unexpected");
        assert(rejected([&] { store::check_room(root, 1, 100, 100, 8, 10); }));
        fs::remove(stage / "unexpected");
    }
    fs::remove_all(root);
    std::cout << "native filesystem quota, preserved staging and writer exclusion: PASS\n";
}
