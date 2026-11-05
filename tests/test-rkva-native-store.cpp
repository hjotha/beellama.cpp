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
static void write_native_object(const fs::path & object, uint8_t key_byte) {
    using namespace rkva_native_snapshot;
    const Geometry geometry{1, 1, 256, 2, 128, 1};
    const Descriptor descriptor{1024, 128, RKVA_RANGE_STORAGE_KVARN4, 3, 128, 128, 256, 2,
                                4, 4, 8960, 3, 2, 128, 0, 1, 1, 1};
    const std::vector<LayerCoverage> coverage{{0, 0, 0, 1}};
    std::array<uint8_t, 32> key{}; key.fill(key_byte);
    Manifest manifest; std::string error;
    assert(build(geometry, descriptor, 1, 0x100000001ull, key.data(), coverage, manifest, &error));
    for (size_t i = 0; i < manifest.chunks.size(); ++i) {
        std::vector<uint8_t> data(manifest.chunks[i].bytes, uint8_t(i + 1));
        manifest.chunks[i].digest = rkva_snapshot::digest(data.data(), data.size());
        std::ofstream file(object / ("n-" + std::to_string(i) + ".kv"), std::ios::binary);
        file.write(reinterpret_cast<const char *>(data.data()), std::streamsize(data.size()));
        assert(file.good());
    }
    std::vector<uint8_t> encoded;
    assert(encode(manifest, encoded, &error));
    std::ofstream file(object / "manifest-v2.bin", std::ios::binary);
    file.write(reinterpret_cast<const char *>(encoded.data()), std::streamsize(encoded.size()));
    assert(file.good());
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
        fs::remove(published / "n-0.kv");
        write_native_object(published, 0xaa);
        const auto now = fs::file_time_type::clock::now();
        fs::last_write_time(published, now - std::chrono::seconds(10));
        fs::last_write_time(stage, now);
        const auto entries = store::inspect(root, 1u << 20, 8);
        assert(entries.size() == 2 && entries[0].name == key1 && entries[0].published);
        assert(entries[1].name == "tmp-" + key2 && !entries[1].published);
        const auto verified = store::plan_retirement(root, key1, 1u << 20, 8, false);
        assert(verified.present && verified.files.size() == 3 && fs::exists(published / "n-0.kv"));
        assert(rejected([&] { store::erase_published(root, "tmp-" + key2, 1u << 20, 8); }));
        assert(rejected([&] { (void) store::plan_retirement(root, std::string(64, 'd'), 1u << 20, 8, false); }));
        store::erase_published(root, std::string(64, 'd'), 1u << 20, 8);
        write(published / "unrelated", "preserve");
        assert(rejected([&] { store::erase_published(root, key1, 1u << 20, 8); }));
        assert(fs::exists(published / "n-0.kv") && fs::exists(published / "unrelated"));
        fs::remove(published / "unrelated");
        {
            std::fstream file(published / "n-0.kv", std::ios::binary | std::ios::in | std::ios::out);
            file.put(char(99)); assert(file.good());
        }
        assert(rejected([&] { store::erase_published(root, key1, 1u << 20, 8); }));
        assert(fs::exists(published / "manifest-v2.bin"));
        {
            std::fstream file(published / "n-0.kv", std::ios::binary | std::ios::in | std::ios::out);
            file.put(char(1)); assert(file.good());
        }
#ifndef _WIN32
        fs::create_symlink(stage / "n-0.kv", published / "link");
        assert(rejected([&] { store::erase_published(root, key1, 1u << 20, 8); }));
        assert(fs::exists(published / "n-0.kv"));
        fs::remove(published / "link");
#endif
        fs::remove(published / "n-0.kv"); // a previous interrupted retirement
        assert(rejected([&] { (void) store::plan_retirement(root, key1, 1u << 20, 8, false); }));
        assert(store::plan_retirement(root, key1, 1u << 20, 8, true).present);
        store::erase_published(root, key1, 1u << 20, 8);
        assert(!fs::exists(published) && fs::file_size(stage / "n-0.kv") == 5);
        store::erase_published(root, key1, 1u << 20, 8);
    }
    fs::remove_all(root);
    std::cout << "native filesystem quota, preserved staging and writer exclusion: PASS\n";
}
