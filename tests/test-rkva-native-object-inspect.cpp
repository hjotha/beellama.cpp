#include "../tools/rkva-native-object-inspect/native-object-inspect.h"
#undef NDEBUG
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
namespace fs = std::filesystem;
namespace snap = rkva_native_snapshot;
namespace inspect = rkva_native_object_inspect;
static fs::path race_dir;
static std::string race_mode;
static void write(const fs::path & path,const std::vector<uint8_t> & bytes) {
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    out.write(reinterpret_cast<const char *>(bytes.data()),std::streamsize(bytes.size())); assert(out.good());
}
static std::vector<uint8_t> read(const fs::path & path) {
    std::ifstream in(path,std::ios::binary); assert(in.good());
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in),{});
}
static void on_read(const char * phase,uint32_t index) {
    if (race_mode=="manifest" && std::string(phase)=="after_manifest") {
        auto bytes=read(race_dir/"manifest-v2.bin"); bytes.back()^=1; write(race_dir/"manifest-v2.bin",bytes); race_mode.clear();
    } else if (race_mode=="chunk" && std::string(phase)=="after_chunk" && index==0) {
        auto bytes=read(race_dir/"n-0.kv"); bytes[0]^=1; write(race_dir/"n-0.kv",bytes); race_mode.clear();
    } else if (race_mode=="rename" && std::string(phase)=="after_chunk" && index==0) {
        const auto path=race_dir/"n-0.kv"; auto bytes=read(path);
        fs::rename(path,race_dir/"moved"); write(path,bytes); fs::remove(race_dir/"moved"); race_mode.clear();
    } else if (race_mode=="extra" && std::string(phase)=="after_chunk" && index==0) {
        write(race_dir/"injected",{1}); race_mode.clear();
    }
}
struct fixture {
    inspect::expected want;
    fs::path dir;
    std::vector<uint8_t> manifest;
    std::vector<std::vector<uint8_t>> chunks;
    fixture(const fs::path & root) {
        want.key.fill(0xab); want.geometry={8,2,256,2,4096,1};
        want.descriptor={1024,1024,RKVA_RANGE_STORAGE_KVARN4,3,128,128,256,2,4,4,8960,3,2,128,128,1,1,1};
        want.live_rows=513; want.saved_generation=0x700000001ull;
        dir=root/rkva_snapshot::hex(want.key.data()); assert(fs::create_directory(dir));
        snap::LayerCoverage cov{385,384,4,1};
        snap::Manifest data; std::string error;
        assert(snap::build(want.geometry,want.descriptor,want.live_rows,want.saved_generation,
                           want.key.data(),{cov},data,&error));
        for(size_t i=0;i<data.chunks.size();i++) {
            chunks.emplace_back(data.chunks[i].bytes,uint8_t(0x31+i));
            data.chunks[i].digest=rkva_snapshot::digest(chunks.back().data(),chunks.back().size());
            write(dir/("n-"+std::to_string(i)+".kv"),chunks.back());
        }
        assert(snap::encode(data,manifest,&error)); write(dir/"manifest-v2.bin",manifest);
    }
    void restore() {
        for(const auto & entry:fs::directory_iterator(dir)) fs::remove_all(entry.path());
        write(dir/"manifest-v2.bin",manifest);
        for(size_t i=0;i<chunks.size();i++) write(dir/("n-"+std::to_string(i)+".kv"),chunks[i]);
    }
};
static void rejected(const fixture & f,const inspect::expected & want) {
    inspect::result out; out.key="sentinel"; out.chunks=99; std::string error;
    assert(!inspect::inspect(f.dir.string(),want,out,error));
    assert(!error.empty() && out.key=="sentinel" && out.chunks==99);
}
int main(int argc, char ** argv) {
#ifdef _WIN32
    (void)argc; (void)argv;
    inspect::expected e; inspect::result r; std::string error;
    assert(!inspect::inspect("x",e,r,error)); return 0;
#else
    const auto root=fs::temp_directory_path()/("native-object-inspect-"+
        std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    assert(fs::create_directory(root)); fixture f(root);
    inspect::result out; std::string error;
    assert(inspect::inspect(f.dir.string(),f.want,out,error));
    assert(out.key==rkva_snapshot::hex(f.want.key.data()) && out.chunks==f.chunks.size() &&
           out.manifest_bytes==f.manifest.size() && out.payload_bytes>0);
    if (argc==2) {
        // Exercise the real command-line parser against the same actual
        // directory, with independent caller-supplied identity JSON.
        const auto config=root/"expected.json";
        auto write_config = [&](uint64_t generation) {
            std::ofstream json(config);
            json << "{\"key\":\"" << rkva_snapshot::hex(f.want.key.data())
                 << "\",\"live_rows\":513,\"saved_generation\":" << generation
                 << ",\"geometry\":{\"n_head\":8,\"n_head_kv\":2,\"head_dim\":256,\"head_slices\":2,\"max_ctx\":4096,\"n_layers\":1}"
                 << ",\"descriptor\":{\"origin\":1024,\"capacity\":1024,\"storage\":4,\"layout_version\":3,"
                 << "\"token_group\":128,\"record_dim\":128,\"head_dim\":256,\"head_slices\":2,\"key_bits\":4,"
                 << "\"value_bits\":4,\"record_bytes\":8960,\"stage_groups\":3,\"tail_groups\":2,\"sink_tokens\":128,"
                 << "\"precision_tail_tokens\":128,\"wht_version\":1,\"domain\":1,\"record_policy\":1}}";
            assert(json.good());
        };
        auto call_cli = [&]() {
            const pid_t pid=fork(); assert(pid>=0);
            if(pid==0) { execl(argv[1],argv[1],f.dir.c_str(),config.c_str(),nullptr); _exit(127); }
            int status=0; assert(waitpid(pid,&status,0)==pid && WIFEXITED(status)); return WEXITSTATUS(status);
        };
        write_config(f.want.saved_generation); assert(call_cli()==0);
        write_config(f.want.saved_generation+1); assert(call_cli()==1);
        fs::remove(config);
    }
    assert(read(f.dir/"manifest-v2.bin")==f.manifest);
    for(size_t i=0;i<f.chunks.size();i++) assert(read(f.dir/("n-"+std::to_string(i)+".kv"))==f.chunks[i]);
    auto bad=f.want; bad.key[0]^=1; rejected(f,bad);
    bad=f.want; bad.geometry.max_ctx++; rejected(f,bad);
    bad=f.want; bad.descriptor.origin+=128; rejected(f,bad);
    bad=f.want; bad.live_rows--; rejected(f,bad);
    bad=f.want; bad.saved_generation++; rejected(f,bad);
    bad=f.want; bad.key.fill(0); rejected(f,bad);
    auto manifest=f.manifest;
    // Duplicate layer counts must match even with a recomputed SHA-256.
    snap::put_u32(manifest.data()+12,2);
    const auto forged_hash=rkva_snapshot::digest(manifest.data(),manifest.size()-32);
    std::copy(forged_hash.begin(),forged_hash.end(),manifest.end()-32);
    write(f.dir/"manifest-v2.bin",manifest); rejected(f,f.want); f.restore();
    manifest=f.manifest; manifest[32]^=1;
    const auto wrong_key_hash=rkva_snapshot::digest(manifest.data(),manifest.size()-32);
    std::copy(wrong_key_hash.begin(),wrong_key_hash.end(),manifest.end()-32);
    write(f.dir/"manifest-v2.bin",manifest); rejected(f,f.want); f.restore();
    manifest=f.manifest; manifest.back()^=1; write(f.dir/"manifest-v2.bin",manifest); rejected(f,f.want); f.restore();
    manifest=f.manifest; manifest.pop_back(); write(f.dir/"manifest-v2.bin",manifest); rejected(f,f.want); f.restore();
    manifest=f.manifest; manifest.push_back(0); write(f.dir/"manifest-v2.bin",manifest); rejected(f,f.want); f.restore();
    {
        std::ofstream out(f.dir/"manifest-v2.bin",std::ios::binary|std::ios::trunc);
        out.seekp(snap::kMaxManifestBytes); out.put('x');
    }
    rejected(f,f.want); f.restore();
    auto chunk=f.chunks[0]; chunk[chunk.size()/2]^=1; write(f.dir/"n-0.kv",chunk); rejected(f,f.want); f.restore();
    chunk=f.chunks[0]; chunk.pop_back(); write(f.dir/"n-0.kv",chunk); rejected(f,f.want); f.restore();
    chunk=f.chunks[0]; chunk.push_back(0); write(f.dir/"n-0.kv",chunk); rejected(f,f.want); f.restore();
    fs::resize_file(f.dir/"n-0.kv",snap::kChunkBytes+1); rejected(f,f.want); f.restore();
    fs::remove(f.dir/"n-0.kv"); rejected(f,f.want); f.restore();
    write(f.dir/"n-00.kv",{1}); rejected(f,f.want); f.restore();
    fs::create_directory(f.dir/"unexpected"); rejected(f,f.want); f.restore();
    fs::remove(f.dir/"n-0.kv"); fs::create_symlink(root/"absent",f.dir/"n-0.kv"); rejected(f,f.want); f.restore();
    fs::remove(f.dir/"manifest-v2.bin"); fs::create_symlink(root/"absent",f.dir/"manifest-v2.bin"); rejected(f,f.want); f.restore();
    const auto outside=root/"hardlink-source"; write(outside,f.chunks[0]);
    fs::remove(f.dir/"n-0.kv"); fs::create_hard_link(outside,f.dir/"n-0.kv"); rejected(f,f.want); f.restore(); fs::remove(outside);
    const auto other=root/"symlink-parent"; assert(fs::create_directory(other));
    const auto alias=other/rkva_snapshot::hex(f.want.key.data()); fs::create_directory_symlink(f.dir,alias);
    inspect::result keep; keep.key="sentinel";
    assert(!inspect::inspect(alias.string(),f.want,keep,error) && keep.key=="sentinel");
    // Entry cap is enforced before allocating unbounded names or opening chunks.
    for(uint32_t i=0;i<snap::kMaxChunks+2;i++) write(f.dir/("extra-"+std::to_string(i)),{});
    rejected(f,f.want); f.restore();
    race_dir=f.dir; inspect::test_hook=on_read;
    for(const auto & mode : {"manifest","chunk","rename","extra"}) {
        race_mode=mode; rejected(f,f.want); assert(race_mode.empty()); f.restore();
    }
    inspect::test_hook=nullptr;
    assert(inspect::inspect(f.dir.string(),f.want,out,error));
    fs::remove_all(root);
    std::cout << "native local object manifest, chunks, identity, races: PASS\n";
#endif
}
