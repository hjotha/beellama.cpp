#include "../tools/server/server-slot-path-collision.h"
#undef NDEBUG
#include <array>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
namespace fs = std::filesystem;
namespace policy = server_slot_path_collision;
static void write(const fs::path & path, const std::string & data) {
    std::ofstream out(path, std::ios::binary); out << data; assert(out.good());
}
static std::string read(const fs::path & path) {
    std::ifstream in(path, std::ios::binary); assert(in.good());
    return std::string(std::istreambuf_iterator<char>(in), {});
}
int main() {
    const auto dir=fs::temp_directory_path()/("slot-path-collision-"+
        std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    assert(fs::create_directory(dir));
    const std::array<std::string,4> suffixes={{".logits", ".meta", ".ckpt", ".ckpts"}};
    for (const auto & suffix : suffixes) {
        const auto base=dir/("file"+suffix);
        for (bool existing : {false,true}) {
            if (existing) write(base,"existing snapshot");
            else fs::remove(base);
            assert(policy::clear(base.string()));
            const auto sidecar=fs::path(base.string()+suffix);
            for (const auto & content : {std::string("unknown-independent"), std::string("RKVK-native-independent")}) {
                write(sidecar, content);
                assert(!policy::clear(base.string()));
                assert(read(sidecar)==content);
                if (existing) assert(read(base)=="existing snapshot");
                fs::remove(sidecar);
            }
#ifndef _WIN32
            fs::create_symlink(dir/"missing-target",sidecar);
            assert(!policy::clear(base.string()));
            assert(fs::is_symlink(fs::symlink_status(sidecar)));
            fs::remove(sidecar);
#endif
            fs::create_directory(sidecar);
            assert(!policy::clear(base.string()));
            fs::remove(sidecar);
        }
        fs::remove(base);
        const auto parent=dir/("parent"+suffix);
        const auto alias=fs::path(parent.string()+suffix);
        write(parent,"another state");
        assert(!policy::clear(alias.string()));
        write(alias,"independent content");
        assert(!policy::clear(alias.string()));
        assert(read(parent)=="another state" && read(alias)=="independent content");
        fs::remove(parent); fs::remove(alias);
#ifndef _WIN32
        fs::create_symlink(dir/"missing-base", parent);
        assert(!policy::clear(alias.string()));
        assert(fs::is_symlink(fs::symlink_status(parent)));
        fs::remove(parent);
#endif
    }
    const auto target=dir/"clean";
    for (const auto & name : {std::string("unrelated.bin.tmp-rollback-123"), std::string(".legacy-slot-stage-123"), std::string("retention.tmp-123")}) {
        const auto artifact=dir/name;
        write(artifact,"unknown staged data");
        assert(policy::has_unresolved_staging(target.string()));
        assert(!policy::clear(target.string()));
        assert(read(artifact)=="unknown staged data");
        fs::remove(artifact);
        fs::create_directory(artifact);
        assert(!policy::clear(target.string()));
        fs::remove(artifact);
#ifndef _WIN32
        fs::create_symlink(dir/"missing-stage-target",artifact);
        assert(!policy::clear(target.string()));
        assert(fs::is_symlink(fs::symlink_status(artifact)));
        fs::remove(artifact);
#endif
    }
    assert(policy::clear(target.string()));
    assert(!policy::clear(""));
    fs::remove_all(dir);
    std::cout << "snapshot path collision preservation: PASS\n";
}
