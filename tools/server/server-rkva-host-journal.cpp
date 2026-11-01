#include "server-rkva-host-journal.h"
#include "../../ggml/include/rkva-snapshot-digest.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#ifndef _WIN32
#include <atomic>
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace rkva_host_journal {
#ifdef RKVA_HOST_JOURNAL_TESTING
void (*test_persist_hook)(const char *) = nullptr;
#endif
namespace {
constexpr size_t max_size = 16 * 1024 * 1024, max_records = 8192, max_path = 4096;
const std::string zero_hash(64, '0');
bool hex64(const std::string & s) {
    return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string::npos && s != zero_hash;
}
bool valid_path(const std::string & s) {
    return !s.empty() && s[0] == '/' && s.size() <= max_path && s.find('\0') == std::string::npos;
}
void u32(std::string & s, uint32_t n) { for (int i=0;i<4;i++) s.push_back(char(n >> (8*i))); }
void u64(std::string & s, uint64_t n) { for (int i=0;i<8;i++) s.push_back(char(n >> (8*i))); }
struct reader {
    const std::string & s; size_t pos = 0, end;
    explicit reader(const std::string & data) : s(data), end(data.size() - 32) {}
    std::string get(size_t n) { if (n > end - pos) throw std::runtime_error("journal truncated"); auto v=s.substr(pos,n); pos+=n; return v; }
    uint32_t get32() { auto b=get(4); uint32_t v=0; for(int i=0;i<4;i++) v|=uint32_t(uint8_t(b[i])) << (8*i); return v; }
    uint64_t get64() { auto b=get(8); uint64_t v=0; for(int i=0;i<8;i++) v|=uint64_t(uint8_t(b[i])) << (8*i); return v; }
};
void validate(const transaction & r) {
    if (!hex64(r.tx) || !hex64(r.reference) || !hex64(r.key)) throw std::runtime_error("invalid journal identity");
    bool plain = r.state == phase::prepared || r.state == phase::aborted;
    bool published = r.state == phase::host_published || r.state == phase::commit_ack ||
                     r.state == phase::release_intent || r.state == phase::release_ack;
    if (!plain && !published) throw std::runtime_error("invalid journal phase");
    if (plain && (!r.host_path.empty() || !r.host_sha256.empty() || r.host_bytes || r.peer_commit_version))
        throw std::runtime_error("invalid unpublished journal record");
    if (published && (!valid_path(r.host_path) || !hex64(r.host_sha256) || !r.host_bytes))
        throw std::runtime_error("invalid published journal record");
    if (r.state == phase::host_published && r.peer_commit_version) throw std::runtime_error("unexpected commit version");
    if ((r.state == phase::commit_ack || r.state == phase::release_intent || r.state == phase::release_ack) && !r.peer_commit_version)
        throw std::runtime_error("missing commit version");
}
std::string encode(const snapshot & snap, const std::string & owner) {
    if (!hex64(owner) || snap.transactions.size() > max_records) throw std::runtime_error("invalid journal owner or capacity");
    std::string out = "RKHJNL01"; out += owner; u64(out,snap.revision); u32(out,uint32_t(snap.transactions.size()));
    std::set<std::string> txs, refs;
    for (const auto & r : snap.transactions) {
        validate(r);
        if (!txs.insert(r.tx).second || !refs.insert(r.reference).second) throw std::runtime_error("duplicate journal identity");
        out += r.tx; out += r.reference; out += r.key; out.push_back(char(r.state));
        u64(out,r.host_bytes); u64(out,r.peer_commit_version); out += r.host_sha256.empty() ? zero_hash : r.host_sha256;
        u32(out,uint32_t(r.host_path.size())); out += r.host_path;
        if (out.size() > max_size - 32) throw std::runtime_error("journal capacity exceeded");
    }
    auto hash=rkva_snapshot::digest(out.data(),out.size()); out.append(reinterpret_cast<const char *>(hash.data()),hash.size());
    return out;
}
snapshot decode(const std::string & s, const std::string & owner) {
    if (s.size() < 8+64+8+4+32 || s.size() > max_size) throw std::runtime_error("invalid journal size");
    const auto hash=rkva_snapshot::digest(s.data(),s.size()-32);
    if (std::memcmp(hash.data(),s.data()+s.size()-32,32)) throw std::runtime_error("journal checksum mismatch");
    reader rd(s); if(rd.get(8)!="RKHJNL01" || rd.get(64)!=owner || !hex64(owner)) throw std::runtime_error("journal format or owner mismatch");
    snapshot result; result.revision=rd.get64(); auto n=rd.get32(); if(n>max_records) throw std::runtime_error("journal count exceeded");
    std::set<std::string> txs,refs;
    for(uint32_t i=0;i<n;i++) {
        transaction r; r.tx=rd.get(64); r.reference=rd.get(64); r.key=rd.get(64); r.state=phase(uint8_t(rd.get(1)[0]));
        r.host_bytes=rd.get64(); r.peer_commit_version=rd.get64(); r.host_sha256=rd.get(64);
        if (r.host_sha256 == zero_hash) r.host_sha256.clear();
        auto len=rd.get32(); if(len>max_path) throw std::runtime_error("journal path too long"); r.host_path=rd.get(len);
        validate(r);
        if(!txs.insert(r.tx).second || !refs.insert(r.reference).second) throw std::runtime_error("duplicate journal identity");
        result.transactions.push_back(std::move(r));
    }
    if(rd.pos != rd.end) throw std::runtime_error("journal trailing data");
    return result;
}
#ifndef _WIN32
struct fd {
    int n=-1;
    fd() = default; explicit fd(int value):n(value) {}
    ~fd() { if(n>=0) ::close(n); }
    fd(const fd &) = delete; fd & operator=(const fd &) = delete;
};
void check_regular(int value) {
    struct stat st{};
    if (::fstat(value,&st) || !S_ISREG(st.st_mode) || st.st_nlink != 1) throw std::runtime_error("journal invalid file or link count");
}
void sync_dir(int value) { if (::fsync(value)) throw std::runtime_error("journal directory sync failed; outcome may be uncertain"); }
struct locked_store {
    fd dir, lock;
    explicit locked_store(const std::string & path) {
        dir.n=::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);
        if(dir.n<0) throw std::runtime_error("journal directory unavailable");
        lock.n=::openat(dir.n,"host-journal.lock",O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW,0600);
        if(lock.n<0) throw std::runtime_error("journal lock unavailable");
        check_regular(lock.n);
        if(::flock(lock.n,LOCK_EX)) throw std::runtime_error("journal lock failed");
        // A prior rename may have returned an uncertain outcome. Synchronize
        // before returning any read or replay success.
        sync_dir(dir.n);
    }
};
std::string read_all(locked_store & store) {
    fd f(::openat(store.dir.n,"host-journal.bin",O_RDONLY|O_CLOEXEC|O_NOFOLLOW));
    if(f.n<0) throw std::runtime_error("journal missing or inaccessible; explicit recovery required");
    check_regular(f.n); struct stat st{};
    if(::fstat(f.n,&st) || st.st_size<0 || uint64_t(st.st_size)>max_size) throw std::runtime_error("journal invalid size");
    std::string s(size_t(st.st_size),'\0'); size_t pos=0;
    while(pos<s.size()) { const ssize_t n=::read(f.n,&s[pos],s.size()-pos); if(n<0 && errno==EINTR) continue; if(n<=0) throw std::runtime_error("journal read failed"); pos+=size_t(n); }
    return s;
}
void hook(const char * stage) {
#ifdef RKVA_HOST_JOURNAL_TESTING
    if (test_persist_hook) test_persist_hook(stage);
#else
    (void) stage;
#endif
}
void persist(locked_store & store, const std::string & content, bool initialize) {
    static std::atomic<uint64_t> nonce{0};
    std::string name; fd temp;
    for(int i=0;i<128;i++) {
        name=".host-journal.tmp-"+std::to_string(::getpid())+"-"+std::to_string(nonce.fetch_add(1));
        temp.n=::openat(store.dir.n,name.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
        if(temp.n>=0) break;
        if(errno != EEXIST) throw std::runtime_error("journal staging creation failed");
    }
    if(temp.n<0) throw std::runtime_error("journal staging namespace exhausted");
    bool renamed=false;
    try {
        hook("before_write"); size_t pos=0;
        while(pos<content.size()) {
            size_t requested = content.size()-pos;
#ifdef RKVA_HOST_JOURNAL_TESTING
            // Exercise a failure after a genuine partial staging write.
            if (pos == 0 && requested > 1) requested /= 2;
#endif
            const ssize_t n=::write(temp.n,content.data()+pos,requested);
            if(n<0 && errno==EINTR) continue;
            if(n<=0) throw std::runtime_error("journal staging write failed");
            pos+=size_t(n);
            if (pos < content.size()) hook("after_partial_write");
        }
        hook("after_write"); hook("before_file_sync"); if(::fsync(temp.n)) throw std::runtime_error("journal staging sync failed");
        hook("after_file_sync");
        hook("before_rename");
        if(initialize) {
            // Hard-link publication is no-replace even if a noncooperating
            // creator races initialization. Both files share the synced inode.
            if(::linkat(store.dir.n,name.c_str(),store.dir.n,"host-journal.bin",0)) throw std::runtime_error("journal already exists or cannot initialize");
            renamed=true;
            hook("after_link");
            if(::unlinkat(store.dir.n,name.c_str(),0)) throw std::runtime_error("journal temporary unlink failed; outcome uncertain");
        } else {
            if(::renameat(store.dir.n,name.c_str(),store.dir.n,"host-journal.bin")) throw std::runtime_error("journal rename failed");
            renamed=true;
        }
        hook("after_rename"); hook("before_directory_sync"); sync_dir(store.dir.n); hook("after_directory_sync");
    } catch (...) {
        if (!renamed) {
            ::unlinkat(store.dir.n,name.c_str(),0);
            throw;
        }
        throw std::runtime_error("journal publication outcome uncertain; reopen and replay");
    }
}
template<class Fn>
bool run(const std::string & dir, const std::string & owner, std::string & error, Fn fn) {
    try { if(!hex64(owner)) throw std::runtime_error("invalid journal owner"); locked_store lock(dir); fn(lock); error.clear(); return true; }
    catch(const std::exception & e) { error=e.what(); return false; }
}
template<class Fn>
bool change(const std::string & dir, const std::string & owner, std::string & error, Fn fn) {
    return run(dir,owner,error,[&](locked_store & lock) {
        snapshot s=decode(read_all(lock),owner);
        if (!fn(s)) return; // idempotent replay; directory sync already performed
        if (s.revision==UINT64_MAX) throw std::runtime_error("journal revision exhausted");
        ++s.revision; persist(lock,encode(s,owner),false);
    });
}
transaction & get(snapshot & s, const std::string & tx) {
    auto it=std::find_if(s.transactions.begin(),s.transactions.end(),[&](const transaction & r){ return r.tx==tx; });
    if(it==s.transactions.end()) throw std::runtime_error("unknown journal transaction");
    return *it;
}
#else
template<class... T> bool unsupported(std::string & e,T&&...) { e="host journal unsupported on Windows; no durability backend qualified"; return false; }
#endif
} // namespace
journal::journal(std::string directory, std::string owner):dir_(std::move(directory)),owner_(std::move(owner)) {}
#ifndef _WIN32
bool journal::initialize(std::string & e) { return run(dir_,owner_,e,[&](locked_store & l) { struct stat s{}; if(!::fstatat(l.dir.n,"host-journal.bin",&s,AT_SYMLINK_NOFOLLOW) || errno!=ENOENT) throw std::runtime_error("journal exists or cannot inspect"); persist(l,encode({},owner_),true); }); }
bool journal::inspect(snapshot & out, std::string & e) const { return run(dir_,owner_,e,[&](locked_store & l){ out=decode(read_all(l),owner_); }); }
bool journal::prepare(const std::string & tx,const std::string & ref,const std::string & key,std::string & e) {
    return change(dir_,owner_,e,[&](snapshot & s){
        if(!hex64(tx)||!hex64(ref)||!hex64(key)) throw std::runtime_error("invalid journal identity");
        for(const auto & r:s.transactions) {
            if(r.tx==tx) { if(r.reference!=ref||r.key!=key||r.state==phase::aborted||r.state==phase::release_ack) throw std::runtime_error("transaction reuse or terminal state"); return false; }
            if(r.reference==ref) throw std::runtime_error("reference reuse");
        }
        if(s.transactions.size()>=max_records) throw std::runtime_error("journal capacity exceeded");
        transaction r; r.tx=tx; r.reference=ref; r.key=key; s.transactions.push_back(std::move(r)); return true;
    });
}
bool journal::host_published(const std::string & tx,const std::string & path,const std::string & sha,uint64_t bytes,std::string & e) {
    return change(dir_,owner_,e,[&](snapshot & s){ auto & r=get(s,tx);
        if(!valid_path(path)||!hex64(sha)||!bytes) throw std::runtime_error("invalid published host identity");
        if(r.state==phase::prepared) { r.state=phase::host_published; r.host_path=path; r.host_sha256=sha; r.host_bytes=bytes; return true; }
        if(r.state==phase::aborted||r.state==phase::release_ack||r.host_path!=path||r.host_sha256!=sha||r.host_bytes!=bytes)
            throw std::runtime_error("conflicting or terminal host publication");
        return false;
    });
}
bool journal::commit_ack(const std::string & tx,uint64_t ver,std::string & e) { return change(dir_,owner_,e,[&](snapshot & s){ auto & r=get(s,tx); if(!ver) throw std::runtime_error("invalid commit version"); if(r.state==phase::host_published) { r.state=phase::commit_ack; r.peer_commit_version=ver; return true; } if((r.state==phase::commit_ack || r.state==phase::release_intent) && r.peer_commit_version==ver) return false; throw std::runtime_error("invalid commit transition"); }); }
bool journal::release_intent(const std::string & tx,uint64_t ver,std::string & e) { return change(dir_,owner_,e,[&](snapshot & s){ auto & r=get(s,tx); if(!ver || r.peer_commit_version!=ver) throw std::runtime_error("release version mismatch"); if(r.state==phase::commit_ack) { r.state=phase::release_intent; return true; } if(r.state==phase::release_intent || r.state==phase::release_ack) return false; throw std::runtime_error("invalid release intent transition"); }); }
bool journal::release_ack(const std::string & tx,uint64_t ver,std::string & e) { return change(dir_,owner_,e,[&](snapshot & s){ auto & r=get(s,tx); if(!ver || r.peer_commit_version!=ver) throw std::runtime_error("release version mismatch"); if(r.state==phase::release_intent) { r.state=phase::release_ack; return true; } if(r.state==phase::release_ack) return false; throw std::runtime_error("invalid release ack transition"); }); }
bool journal::abort_unpublished(const std::string & tx,std::string & e) { return change(dir_,owner_,e,[&](snapshot & s){ auto & r=get(s,tx); if(r.state==phase::prepared) { r.state=phase::aborted; return true; } if(r.state==phase::aborted) return false; throw std::runtime_error("published transaction cannot be aborted"); }); }
#else
bool journal::initialize(std::string & e) { return unsupported(e); }
bool journal::inspect(snapshot &,std::string & e) const { return unsupported(e); }
bool journal::prepare(const std::string &,const std::string &,const std::string &,std::string & e) { return unsupported(e); }
bool journal::host_published(const std::string &,const std::string &,const std::string &,uint64_t,std::string & e) { return unsupported(e); }
bool journal::commit_ack(const std::string &,uint64_t,std::string & e) { return unsupported(e); }
bool journal::release_intent(const std::string &,uint64_t,std::string & e) { return unsupported(e); }
bool journal::release_ack(const std::string &,uint64_t,std::string & e) { return unsupported(e); }
bool journal::abort_unpublished(const std::string &,std::string & e) { return unsupported(e); }
#endif
} // namespace rkva_host_journal
