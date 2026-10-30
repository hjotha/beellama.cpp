#include "../tools/server/server-snapshot-retention.h"

#undef NDEBUG
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>

using namespace server_snapshot_retention;

struct lock {
    fs::path dir;
    int fd = -1;
    explicit lock(const fs::path & p) : dir(p) {
        fd = ::open((p / ".llama-kv-snapshot-store.lock").c_str(), O_CREAT|O_RDWR, 0600);
        if (fd >= 0 && ::flock(fd, LOCK_EX|LOCK_NB) != 0) { ::close(fd); fd = -1; }
    }
    ~lock() { if (fd >= 0) { ::flock(fd, LOCK_UN); ::close(fd); } }
    bool owns_exclusive_for(const std::string & p) const { return fd >= 0 && fs::equivalent(dir, fs::path(p).parent_path()); }
};
struct pin {
    int fd = -1;
    explicit pin(const fs::path & p) {
        auto name = p.string() + ".test-pin";
        fd = ::open(name.c_str(), O_CREAT|O_RDWR, 0600);
        if (fd >= 0 && ::flock(fd, LOCK_EX|LOCK_NB) != 0) { ::close(fd); fd = -1; }
    }
    ~pin() { if (fd >= 0) { ::flock(fd, LOCK_UN); ::close(fd); } }
    bool acquired() const { return fd >= 0; }
};
struct injected_ops : filesystem_ops {
    int renames = 0, removals = 0, fail_rename = -1, fail_rename2 = -1, fail_remove = -1;
    void rename(const fs::path & a, const fs::path & b, std::error_code & ec) {
        ++renames;
        if (renames == fail_rename || renames == fail_rename2) { ec = std::make_error_code(std::errc::permission_denied); return; }
        filesystem_ops::rename(a, b, ec);
    }
    bool remove(const fs::path & p, std::error_code & ec) {
        ++removals;
        if (removals == fail_remove) { ec = std::make_error_code(std::errc::permission_denied); return false; }
        return filesystem_ops::remove(p, ec);
    }
};

static std::vector<std::string> names{"a.bin", "b.bin"};
static std::vector<std::string> components{"a.bin", "a.bin.logits", "a.bin.meta", "a.bin.ckpt", "a.bin.ckpts",
                                           "b.bin", "b.bin.logits", "b.bin.meta", "b.bin.ckpt", "b.bin.ckpts"};
static auto acquire = [](const fs::path & p) { return std::make_unique<pin>(p); };
static auto ordinary = [](const fs::path &) { return classification::local_only; };
// Mirror the planner: acquire and retain each victim lease before run, then
// check the actual retained lock associated with each path without reacquiring.
template <class Factory, class Classify, class Publish>
static result run_test(const fs::path & dir, lock & store, const std::vector<std::string> & victims,
        Factory factory, Classify classify, Publish publish, injected_ops & io) {
    std::vector<std::pair<fs::path, std::unique_ptr<pin>>> pins;
    for (const auto & name : victims) {
        const fs::path part(name);
        if (part.filename() != part || part.is_absolute()) continue;
        const fs::path path = dir / part;
        bool seen = false;
        for (const auto & value : pins) if (value.first == path) seen = true;
        if (!seen) pins.emplace_back(path, factory(path));
    }
    auto holds = [&](const fs::path & path) {
        for (const auto & value : pins) if (value.first == path) return value.second && value.second->acquired();
        return false;
    };
    return server_snapshot_retention::run(dir, store, victims, holds, classify, publish, io);
}
static void seed(const fs::path & dir) {
    fs::create_directories(dir);
    for (const auto & x : components) { std::ofstream f(dir / x); f << x; assert(f.good()); }
}
static bool contents(const fs::path & p, const std::string & s) { std::ifstream f(p); std::string v; f >> v; return v == s; }
static void check_restored(const fs::path & dir) {
    for (const auto & x : components) assert(contents(dir / x, x));
}
static void check_absent(const fs::path & dir) {
    for (const auto & x : components) assert(!fs::exists(dir / x));
}

int main() {
    char templ[] = "/tmp/retention-stage-test-XXXXXX";
    char * p = ::mkdtemp(templ); assert(p); const fs::path root(p);
    for (int i = 1; i <= 10; ++i) {
        const auto dir = root / ("rename-" + std::to_string(i)); seed(dir); lock store(dir); assert(store.fd >= 0);
        injected_ops io; io.fail_rename = i; bool published = false;
        const auto r = run_test(dir, store, names, acquire, ordinary, [&] { published = true; return true; }, io);
        assert(r.state == outcome::rolled_back && !published && r.pending.empty());
        check_restored(dir); assert(!fs::exists(r.staging_directory));
    }
    {
        const auto dir = root / "rollback-error"; seed(dir); lock store(dir); injected_ops io;
        io.fail_rename = 4; io.fail_rename2 = 5; // fourth staging rename; first rollback rename
        const auto r = run_test(dir, store, names, acquire, ordinary, [] { return true; }, io);
        assert(r.state == outcome::rollback_incomplete && r.pending.size() == 1);
        assert(fs::exists(r.pending[0]) && fs::exists(r.staging_directory));
        assert(contents(r.pending[0], r.pending[0].filename().string()));
    }
    for (int i = 1; i <= 11; ++i) {
        const auto dir = root / ("remove-" + std::to_string(i)); seed(dir); lock store(dir); injected_ops io;
        io.fail_remove = i; bool published = false;
        const auto r = run_test(dir, store, names, acquire, ordinary, [&] { published = true; return true; }, io);
        assert(published && r.state == outcome::cleanup_incomplete && fs::exists(r.staging_directory));
        check_absent(dir);
        if (i <= 10) { assert(r.pending.size() == 1 && fs::exists(r.pending[0])); }
        else { assert(r.pending.empty() && fs::is_empty(r.staging_directory)); }
        const auto next = run_test(dir, store, names, acquire, ordinary, [] { return true; }, io);
        assert(next.state == outcome::rejected && next.reason == "unresolved retention staging exists");
    }
    for (int mode = 0; mode < 2; ++mode) {
        const auto dir = root / ("publish-" + std::to_string(mode)); seed(dir); lock store(dir); injected_ops io;
        const auto r = run_test(dir, store, names, acquire, ordinary, [mode]() -> bool {
            if (mode) throw std::runtime_error("publication");
            return false;
        }, io);
        assert(r.state == outcome::rolled_back); check_restored(dir); assert(!fs::exists(r.staging_directory));
    }
    {
        const auto dir = root / "success"; seed(dir); lock store(dir); injected_ops io;
        const auto r = run_test(dir, store, names, acquire, ordinary, [] { return true; }, io);
        assert(r.state == outcome::committed && r.pending.empty()); check_absent(dir); assert(!fs::exists(r.staging_directory));
    }
    {
        const auto dir = root / "unknown"; seed(dir); lock store(dir); injected_ops io;
        for (auto kind : {classification::unknown, classification::native_pair}) {
            const auto r = run_test(dir, store, names, acquire, [kind](const fs::path &) { return kind; }, [] { return true; }, io);
            assert(r.state == outcome::rejected && io.renames == 0); check_restored(dir);
        }
        const auto duplicate = run_test(dir, store, std::vector<std::string>{"a.bin", "a.bin"}, acquire, ordinary, [] { return true; }, io);
        assert(duplicate.state == outcome::rejected && io.renames == 0); check_restored(dir);
        const auto escape = run_test(dir, store, std::vector<std::string>{"../a.bin"}, acquire, ordinary, [] { return true; }, io);
        assert(escape.state == outcome::rejected && io.renames == 0); check_restored(dir);
        for (const auto & suffix : {".logits", ".meta", ".ckpt", ".ckpts", ".lease", ".ref", ".lock", ".tmp"}) {
            const auto reserved = run_test(dir, store, std::vector<std::string>{std::string("unknown") + suffix}, acquire, ordinary, [] { return true; }, io);
            assert(reserved.state == outcome::rejected && io.renames == 0);
        }
        const auto unproven = run_test(dir, store, names, acquire, [](const fs::path & p) {
            return p.extension() == ".ckpts" ? classification::unknown : classification::local_only;
        }, [] { return true; }, io);
        assert(unproven.state == outcome::rejected && io.renames == 0); check_restored(dir);
    }
    {
        const auto dir = root / "generic"; fs::create_directories(dir);
        { std::ofstream(dir / "explicit.slot") << "slot"; std::ofstream(dir / "route") << "route"; }
        lock store(dir); injected_ops io;
        const auto r = run_test(dir, store, std::vector<std::string>{"explicit.slot", "route"}, acquire, ordinary, [] { return true; }, io);
        assert(r.state == outcome::committed && !fs::exists(dir / "explicit.slot") && !fs::exists(dir / "route"));
    }
    {
        const auto dir = root / "link"; seed(dir); fs::remove(dir / "a.bin.meta"); fs::create_symlink("missing", dir / "a.bin.meta");
        lock store(dir); injected_ops io; const auto r = run_test(dir, store, names, acquire, ordinary, [] { return true; }, io);
        assert(r.state == outcome::rejected && io.renames == 0);
    }
    {
        const auto dir = root / "hardlink"; seed(dir); fs::create_hard_link(dir / "a.bin", dir / "outside-link");
        lock store(dir); injected_ops io; const auto r = run_test(dir, store, names, acquire, ordinary, [] { return true; }, io);
        assert(r.state == outcome::rejected && io.renames == 0); check_restored(dir);
    }
    {
        const auto dir = root / "busy"; seed(dir); lock store(dir); injected_ops io;
        pin held(dir / "a.bin"); assert(held.acquired());
        const auto r = run_test(dir, store, names, acquire, ordinary, [] { return true; }, io);
        assert(r.state == outcome::rejected && io.renames == 0); check_restored(dir);
        lock wrong(root); const auto wrong_r = run_test(dir, wrong, names, acquire, ordinary, [] { return true; }, io);
        assert(wrong_r.state == outcome::rejected); check_restored(dir);
    }
    {
        // A crash after staging leaves the staging directory behind. This test
        // intentionally does not call that recovery: no journal exists.
        const auto dir = root / "crash"; seed(dir);
        const pid_t child = fork(); assert(child >= 0);
        if (child == 0) { lock store(dir); injected_ops io; (void) run_test(dir, store, names, acquire, ordinary, []() -> bool { _exit(73); }, io); _exit(74); }
        int status = 0; assert(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 73);
        lock after_restart(dir); assert(after_restart.fd >= 0); check_absent(dir);
        int staged_dirs = 0;
        for (const auto & entry : fs::directory_iterator(dir)) {
            if (entry.is_directory() && entry.path().filename().string().find("retention.tmp-") == 0) {
                ++staged_dirs; for (const auto & x : components) assert(contents(entry.path() / x, x));
            }
        }
        assert(staged_dirs == 1);
        injected_ops io;
        const auto next = run_test(dir, after_restart, names, acquire, ordinary, [] { return true; }, io);
        assert(next.state == outcome::rejected && next.reason == "unresolved retention staging exists" && io.renames == 0);
    }
    fs::remove_all(root);
    std::cout << "snapshot retention staging: ok\n";
}
