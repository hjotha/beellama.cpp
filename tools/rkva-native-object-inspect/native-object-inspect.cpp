#include "native-object-inspect.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>
#ifndef _WIN32
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace rkva_native_object_inspect {
#ifdef RKVA_NATIVE_OBJECT_INSPECT_TESTING
void (*test_hook)(const char *, uint32_t) = nullptr;
#endif
namespace {
void require(bool condition, const char * error) { if (!condition) throw std::runtime_error(error); }
void hook(const char * phase, uint32_t index) {
#ifdef RKVA_NATIVE_OBJECT_INSPECT_TESTING
    if (test_hook) test_hook(phase,index);
#else
    (void) phase; (void) index;
#endif
}
#ifndef _WIN32
struct fd {
    int value = -1;
    explicit fd(int f=-1):value(f) {}
    ~fd() { if (value>=0) ::close(value); }
    fd(const fd &) = delete;
    fd & operator=(const fd &) = delete;
};
struct dir_close { void operator()(DIR * p) const { if (p) ::closedir(p); } };
bool same(const struct stat & a, const struct stat & b) {
    return a.st_dev==b.st_dev && a.st_ino==b.st_ino && a.st_mode==b.st_mode &&
           a.st_nlink==b.st_nlink && a.st_size==b.st_size &&
           a.st_mtim.tv_sec==b.st_mtim.tv_sec && a.st_mtim.tv_nsec==b.st_mtim.tv_nsec &&
           a.st_ctim.tv_sec==b.st_ctim.tv_sec && a.st_ctim.tv_nsec==b.st_ctim.tv_nsec;
}
struct entry_info { std::string name; struct stat st{}; };
int open_readonly(int dir, const char * name) {
    int flags=O_RDONLY|O_CLOEXEC|O_NOFOLLOW;
#ifdef O_NOATIME
    int opened=::openat(dir,name,flags|O_NOATIME);
    if(opened>=0) return opened;
    if(errno != EPERM && errno != EACCES) return -1;
#endif
    return ::openat(dir,name,flags);
}
struct stat path_stat(int dir, const std::string & name) {
    struct stat result{};
    require(::fstatat(dir,name.c_str(),&result,AT_SYMLINK_NOFOLLOW)==0, "object entry missing or inaccessible");
    require(S_ISREG(result.st_mode) && result.st_nlink==1, "object entry is not an exclusive regular file");
    return result;
}
std::set<std::string> list(int dir) {
    // openat(".") creates a fresh directory description: dup would share the
    // seek position with the previous scan.
    const int copy=::openat(dir,".",O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);
    require(copy>=0,"cannot open object directory for enumeration");
    DIR * raw=::fdopendir(copy);
    if (!raw) { ::close(copy); throw std::runtime_error("cannot enumerate object directory"); }
    std::unique_ptr<DIR,dir_close> entries(raw);
    std::set<std::string> result;
    for (;;) {
        errno=0; const dirent * entry=::readdir(entries.get());
        if(!entry) { require(errno==0,"object directory enumeration failed"); break; }
        const std::string name(entry->d_name);
        if(name=="."||name=="..") continue;
        require(result.size() < rkva_native_snapshot::kMaxChunks + 1, "object entry limit exceeded");
        require(result.insert(name).second, "duplicate object entry");
    }
    return result;
}
void read_all(int file, void * data, size_t bytes) {
    auto * p=static_cast<uint8_t *>(data); size_t done=0;
    while(done<bytes) {
        const auto n=::read(file,p+done,bytes-done);
        if(n<0 && errno==EINTR) continue;
        require(n>0,"object file truncated or unreadable");
        done+=size_t(n);
    }
}
entry_info checked_open(int dir, const std::string & name, fd & file, uint64_t expected_bytes) {
    entry_info info; info.name=name; info.st=path_stat(dir,name);
    require(info.st.st_size>=0 && uint64_t(info.st.st_size)==expected_bytes, "object file extent mismatch");
    file.value=open_readonly(dir,name.c_str());
    require(file.value>=0, "cannot open object file read-only");
    struct stat opened{};
    require(::fstat(file.value,&opened)==0 && same(info.st,opened), "object file changed during open");
    return info;
}
void check_after_read(int file, const entry_info & info) {
    struct stat after{};
    require(::fstat(file,&after)==0 && same(info.st,after), "object file changed while reading");
}
#endif
} // namespace
bool inspect(const std::string & directory, const expected & want, result & out, std::string & error) {
#ifdef _WIN32
    (void)directory; (void)want; (void)out;
    error="native object inspection unsupported on Windows"; return false;
#else
    try {
        require(!directory.empty() && directory.find('\0')==std::string::npos, "invalid object path");
        require(std::any_of(want.key.begin(),want.key.end(),[](uint8_t v){return v!=0;}), "invalid all-zero expected key");
        const std::string key=rkva_snapshot::hex(want.key.data());
        const auto name=std::filesystem::path(directory).lexically_normal().filename().string();
        require(name==key,"object directory name does not match expected key");
        fd root(::open(directory.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));
        require(root.value>=0,"cannot open object directory read-only");
        struct stat initial_dir{};
        require(::fstat(root.value,&initial_dir)==0 && S_ISDIR(initial_dir.st_mode),"invalid object directory");
        const auto manifest_path=path_stat(root.value,"manifest-v2.bin");
        require(manifest_path.st_size>=0 && uint64_t(manifest_path.st_size)<=rkva_native_snapshot::kMaxManifestBytes,
                "manifest exceeds bounded size");
        fd manifest_file;
        auto manifest_info=checked_open(root.value,"manifest-v2.bin",manifest_file,uint64_t(manifest_path.st_size));
        std::vector<uint8_t> manifest_bytes(size_t(manifest_path.st_size));
        read_all(manifest_file.value,manifest_bytes.data(),manifest_bytes.size());
        check_after_read(manifest_file.value,manifest_info);
        hook("after_manifest",0);
        rkva_native_snapshot::Manifest manifest;
        std::string codec_error;
        if (!rkva_native_snapshot::decode(manifest_bytes,manifest,&codec_error))
            throw std::runtime_error(codec_error);
        // The shared decoder overwrites the duplicated layer count at offset
        // 12 with the geometry count at offset 84. Re-encoding and comparing
        // every byte also rejects inconsistent duplicate or reserved fields.
        std::vector<uint8_t> canonical;
        if (!rkva_native_snapshot::encode(manifest,canonical,&codec_error) || canonical != manifest_bytes)
            throw std::runtime_error("native snapshot: noncanonical manifest encoding");
        if (!rkva_native_snapshot::validate(manifest,want.geometry,want.descriptor,want.live_rows,
                    want.saved_generation,want.key.data(),&codec_error))
            throw std::runtime_error(codec_error);
        std::set<std::string> names{"manifest-v2.bin"};
        for(size_t i=0;i<manifest.chunks.size();i++) names.insert("n-"+std::to_string(i)+".kv");
        require(list(root.value)==names,"object contains missing or unexpected entries");
        std::vector<entry_info> observed; observed.reserve(manifest.chunks.size()+1); observed.push_back(manifest_info);
        std::array<uint8_t,64*1024> buffer{};
        uint64_t total=0;
        for(size_t i=0;i<manifest.chunks.size();i++) {
            const auto & chunk=manifest.chunks[i];
            require(chunk.bytes>0 && chunk.bytes<=rkva_native_snapshot::kChunkBytes, "chunk exceeds bounded size");
            fd file;
            auto info=checked_open(root.value,"n-"+std::to_string(i)+".kv",file,chunk.bytes);
            rkva_snapshot::Sha256 hash;
            uint64_t left=chunk.bytes;
            while(left) {
                const auto n=size_t(std::min<uint64_t>(left,buffer.size()));
                read_all(file.value,buffer.data(),n); hash.update(buffer.data(),n); left-=n;
            }
            check_after_read(file.value,info);
            require(hash.finish()==chunk.digest,"native snapshot: chunk checksum mismatch");
            observed.push_back(info);
            total+=chunk.bytes;
            hook("after_chunk",uint32_t(i));
        }
        require(total==manifest.payload_bytes,"object payload total mismatch");
        require(list(root.value)==names,"object directory changed during inspection");
        for(const auto & entry:observed) require(same(entry.st,path_stat(root.value,entry.name)),"object entry changed during inspection");
        struct stat final_dir{}, path_dir{};
        require(::fstat(root.value,&final_dir)==0 && same(initial_dir,final_dir),"object directory changed during inspection");
        require(::lstat(directory.c_str(),&path_dir)==0 && S_ISDIR(path_dir.st_mode) && same(initial_dir,path_dir),
                "object directory pathname changed during inspection");
        result success; success.key=key; success.chunks=uint32_t(manifest.chunks.size());
        success.payload_bytes=total; success.manifest_bytes=manifest_bytes.size();
        const auto digest=rkva_snapshot::digest(manifest_bytes.data(),manifest_bytes.size());
        success.manifest_sha256=rkva_snapshot::hex(digest.data());
        out=std::move(success); error.clear(); return true;
    } catch(const std::exception & exception) {
        error=exception.what(); return false;
    }
#endif
}
} // namespace rkva_native_object_inspect
