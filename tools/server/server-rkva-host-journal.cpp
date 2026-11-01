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
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace rkva_host_journal {
#ifdef RKVA_HOST_JOURNAL_TESTING
void (*test_persist_hook)(const char *) = nullptr;
#endif
namespace {
constexpr size_t max_size = 16 * 1024 * 1024, max_records = 8192, max_path = 4096, max_nonces = 50000;
const std::string zero_hash(64, '0');
bool hex64(const std::string & s) {
    return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string::npos && s != zero_hash;
}
bool nonzero(const auth_id & id) { return std::any_of(id.begin(),id.end(),[](uint8_t b){return b!=0;}); }
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
    std::string out = snap.anchor.v3_configured ? "RKHJNL03" : snap.anchor.configured ? "RKHJNL02" : "RKHJNL01";
    out += owner; u64(out,snap.revision); u32(out,uint32_t(snap.transactions.size()));
    std::set<std::string> txs, refs;
    for (const auto & r : snap.transactions) {
        validate(r);
        if (!txs.insert(r.tx).second || !refs.insert(r.reference).second) throw std::runtime_error("duplicate journal identity");
        out += r.tx; out += r.reference; out += r.key; out.push_back(char(r.state));
        u64(out,r.host_bytes); u64(out,r.peer_commit_version); out += r.host_sha256.empty() ? zero_hash : r.host_sha256;
        u32(out,uint32_t(r.host_path.size())); out += r.host_path;
        if (out.size() > max_size - 32) throw std::runtime_error("journal capacity exceeded");
    }
    if(snap.anchor.configured) {
        if(!snap.anchor.key_epoch || !nonzero(snap.anchor.digest) || snap.anchor.nonces.size()>max_nonces)
            throw std::runtime_error("invalid authenticated anchor");
        u64(out,snap.anchor.key_epoch);u64(out,snap.anchor.version);
        out.append(reinterpret_cast<const char *>(snap.anchor.digest.data()),snap.anchor.digest.size());
        u32(out,uint32_t(snap.anchor.nonces.size()));std::set<auth_id> nonces;
        for(const auto & n:snap.anchor.nonces) {
            if(!hex64(n.tx) || !txs.count(n.tx) || !nonzero(n.nonce) || !nonces.insert(n.nonce).second)
                throw std::runtime_error("invalid issued nonce");
            out+=n.tx;out.append(reinterpret_cast<const char *>(n.nonce.data()),n.nonce.size());out.push_back(n.consumed?1:0);
        }
        if(snap.anchor.v3_configured) {
            if(!nonzero(snap.anchor.ledger_epoch) || snap.anchor.v3_nonces.size()>max_nonces)
                throw std::runtime_error("invalid v3 nonce ledger");
            out.append(reinterpret_cast<const char *>(snap.anchor.ledger_epoch.data()),32);
            u32(out,uint32_t(snap.anchor.v3_nonces.size()));
            std::set<auth_id> v3_seen;
            for(const auto & n:snap.anchor.v3_nonces) {
                if(!hex64(n.tx)||!txs.count(n.tx)||!hex64(n.reference)||!hex64(n.key)||
                   n.op>UINT16_MAX || !rkva_registry_wire::valid_op(uint16_t(n.op))||!nonzero(n.nonce)||!nonzero(n.request_hash)||
                   !nonzero(n.transcript)||!nonzero(n.challenge)||!nonzero(n.anchor_digest)||
                   (n.consumed != nonzero(n.response_hash)) || !v3_seen.insert(n.nonce).second)
                    throw std::runtime_error("invalid v3 issued nonce");
                for(const auto & old:snap.anchor.nonces)if(old.nonce==n.nonce)throw std::runtime_error("nonce reused across versions");
                auto tx=std::find_if(snap.transactions.begin(),snap.transactions.end(),[&](const transaction & t){return t.tx==n.tx;});
                if(tx==snap.transactions.end()||tx->reference!=n.reference||tx->key!=n.key)throw std::runtime_error("v3 transaction binding differs");
                out+=n.tx;out+=n.reference;out+=n.key;u32(out,n.op);
                for(const auto * id:{&n.nonce,&n.request_hash,&n.transcript,&n.challenge,&n.anchor_digest,&n.response_hash})
                    out.append(reinterpret_cast<const char *>(id->data()),32);
                u64(out,n.anchor_version);out.push_back(n.consumed?1:0);
            }
        }
        if(out.size()>max_size-32)throw std::runtime_error("journal capacity exceeded");
    }
    auto hash=rkva_snapshot::digest(out.data(),out.size()); out.append(reinterpret_cast<const char *>(hash.data()),hash.size());
    return out;
}
snapshot decode(const std::string & s, const std::string & owner) {
    if (s.size() < 8+64+8+4+32 || s.size() > max_size) throw std::runtime_error("invalid journal size");
    const auto hash=rkva_snapshot::digest(s.data(),s.size()-32);
    if (std::memcmp(hash.data(),s.data()+s.size()-32,32)) throw std::runtime_error("journal checksum mismatch");
    reader rd(s);const auto magic=rd.get(8);
    if((magic!="RKHJNL01" && magic!="RKHJNL02" && magic!="RKHJNL03") || rd.get(64)!=owner || !hex64(owner)) throw std::runtime_error("journal format or owner mismatch");
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
    if(magic=="RKHJNL02" || magic=="RKHJNL03") {
        result.anchor.configured=true;result.anchor.key_epoch=rd.get64();result.anchor.version=rd.get64();
        const auto digest=rd.get(32);std::copy(digest.begin(),digest.end(),result.anchor.digest.begin());
        const auto count=rd.get32();
        if(!result.anchor.key_epoch || !nonzero(result.anchor.digest) || count>max_nonces || count>(rd.end-rd.pos)/97)
            throw std::runtime_error("invalid authenticated anchor");
        std::set<auth_id> nonces;
        for(uint32_t i=0;i<count;i++) {
            issued_nonce item;item.tx=rd.get(64);const auto bytes=rd.get(32);
            std::copy(bytes.begin(),bytes.end(),item.nonce.begin());const auto flag=uint8_t(rd.get(1)[0]);
            if(!txs.count(item.tx) || !nonzero(item.nonce) || flag>1 || !nonces.insert(item.nonce).second)
                throw std::runtime_error("invalid issued nonce");
            item.consumed=flag!=0;result.anchor.nonces.push_back(std::move(item));
        }
        if(magic=="RKHJNL03") {
            result.anchor.v3_configured=true;
            auto ledger=rd.get(32);std::copy(ledger.begin(),ledger.end(),result.anchor.ledger_epoch.begin());
            const auto count_v3=rd.get32();constexpr size_t record_bytes=192+4+6*32+8+1;
            if(!nonzero(result.anchor.ledger_epoch)||count_v3>max_nonces||count_v3>(rd.end-rd.pos)/record_bytes)
                throw std::runtime_error("invalid v3 nonce ledger");
            std::set<auth_id> v3_seen;
            for(uint32_t i=0;i<count_v3;i++) {
                issued_v3_nonce item;item.tx=rd.get(64);item.reference=rd.get(64);item.key=rd.get(64);item.op=rd.get32();
                for(auto * value:{&item.nonce,&item.request_hash,&item.transcript,&item.challenge,&item.anchor_digest,&item.response_hash}) {
                    const auto raw=rd.get(32);std::copy(raw.begin(),raw.end(),value->begin());
                }
                item.anchor_version=rd.get64();const auto flag=uint8_t(rd.get(1)[0]);item.consumed=flag!=0;
                const auto tx=std::find_if(result.transactions.begin(),result.transactions.end(),[&](const transaction & t){return t.tx==item.tx;});
                if(flag>1||tx==result.transactions.end()||tx->reference!=item.reference||tx->key!=item.key||
                   item.op>UINT16_MAX || !rkva_registry_wire::valid_op(uint16_t(item.op))||!nonzero(item.nonce)||!nonzero(item.request_hash)||
                   !nonzero(item.transcript)||!nonzero(item.challenge)||!nonzero(item.anchor_digest)||
                   (item.consumed != nonzero(item.response_hash))||!v3_seen.insert(item.nonce).second)
                    throw std::runtime_error("invalid v3 nonce record");
                for(const auto & old:result.anchor.nonces)if(old.nonce==item.nonce)throw std::runtime_error("nonce reused across versions");
                result.anchor.v3_nonces.push_back(std::move(item));
            }
        }
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
bool journal::commit_ack(const std::string & tx,uint64_t ver,std::string & e) { return change(dir_,owner_,e,[&](snapshot & s){
    if(s.anchor.configured)throw std::runtime_error("authenticated journal requires verified atomic commit");
    auto & r=get(s,tx); if(!ver) throw std::runtime_error("invalid commit version"); if(r.state==phase::host_published) { r.state=phase::commit_ack; r.peer_commit_version=ver; return true; } if((r.state==phase::commit_ack || r.state==phase::release_intent) && r.peer_commit_version==ver) return false; throw std::runtime_error("invalid commit transition"); }); }
bool journal::provision_anchor(uint64_t epoch,uint64_t version,const auth_id & digest,std::string & e) {
    return change(dir_,owner_,e,[&](snapshot & s){
        if(s.anchor.configured || !s.transactions.empty() || !epoch || !nonzero(digest))
            throw std::runtime_error("anchor provisioning refused");
        s.anchor.configured=true;s.anchor.key_epoch=epoch;s.anchor.version=version;s.anchor.digest=digest;return true;
    });
}
bool journal::issue_commit_nonce(const std::string & tx,const auth_id & nonce,const std::string & owner,uint64_t epoch,std::string & e) {
    return change(dir_,owner_,e,[&](snapshot & s){
        const auto & r=get(s,tx);
        if(!s.anchor.configured || s.anchor.v3_configured || owner_!=owner || s.anchor.key_epoch!=epoch || r.state!=phase::host_published ||
                !nonzero(nonce) || s.anchor.nonces.size()>=max_nonces)
            throw std::runtime_error("commit nonce issuance refused");
        for(const auto & prior:s.anchor.nonces)if(prior.nonce==nonce)throw std::runtime_error("nonce reuse rejected");
        s.anchor.nonces.push_back({tx,nonce,false});return true;
    });
}
bool journal::commit_verified(const std::string & tx,const auth_id & nonce,uint64_t peer_version,
        uint64_t anchor_version,const auth_id & digest,const std::string & reference,const std::string & key,
        const std::string & owner,uint64_t epoch,std::string & e) {
    return change(dir_,owner_,e,[&](snapshot & s){
        auto & r=get(s,tx);
        if(!s.anchor.configured || s.anchor.v3_configured || owner_!=owner || s.anchor.key_epoch!=epoch ||
                r.state!=phase::host_published || r.reference!=reference || r.key!=key ||
                !peer_version || !nonzero(digest) || s.anchor.version==UINT64_MAX ||
                anchor_version!=s.anchor.version+1 || peer_version>anchor_version)
            throw std::runtime_error("verified commit state mismatch");
        auto it=std::find_if(s.anchor.nonces.begin(),s.anchor.nonces.end(),[&](const issued_nonce & x){return x.nonce==nonce;});
        if(it==s.anchor.nonces.end() || it->tx!=tx || it->consumed)throw std::runtime_error("nonce absent or already consumed");
        it->consumed=true;s.anchor.version=anchor_version;s.anchor.digest=digest;
        r.state=phase::commit_ack;r.peer_commit_version=peer_version;return true;
    });
}
bool journal::configure_v3_epoch(uint64_t epoch,const auth_id & ledger,uint64_t version,const auth_id & digest,std::string & e) {
    return change(dir_,owner_,e,[&](snapshot & s){
        if(!s.anchor.configured||s.anchor.v3_configured||s.anchor.key_epoch!=epoch||
           s.anchor.version!=version||s.anchor.digest!=digest||!nonzero(ledger)||
           std::any_of(s.anchor.nonces.begin(),s.anchor.nonces.end(),[](const issued_nonce & n){return !n.consumed;}))
            throw std::runtime_error("trusted v3 epoch binding refused");
        s.anchor.v3_configured=true;s.anchor.ledger_epoch=ledger;return true;
    });
}
bool journal::issue_v3_command(const rkva_managed_framing_v3::secret & secret,
        const rkva_managed_framing_v3::session & session,v3_connection_authority * connection,
        const v3_frame_builder & builder,v3_issued_request & output,std::string & e) {
    namespace v3=rkva_managed_framing_v3;
    v3_issued_request local;
    const bool ok=change(dir_,owner_,e,[&](snapshot & s){
        if(!s.anchor.configured||!s.anchor.v3_configured||!v3::valid_session(session)||
           rkva_registry_wire::to_hex(session.owner)!=owner_||session.auth_epoch!=s.anchor.key_epoch||
           session.ledger_epoch!=s.anchor.ledger_epoch||!connection||!connection->held_and_bound(session)||!builder||
           s.anchor.v3_nonces.size()>=max_nonces)
            throw std::runtime_error("v3 issue authority or epoch unavailable");
        size_t used=0;
        while(used<local.nonce.size()) {
            const auto n=::getrandom(local.nonce.data()+used,local.nonce.size()-used,0);
            if(n<0&&errno==EINTR)continue;
            if(n<=0)throw std::runtime_error("v3 CSPRNG unavailable");
            used+=size_t(n);
        }
        if(!nonzero(local.nonce))throw std::runtime_error("invalid random nonce");
        if(!builder(local.nonce,local.frame)||local.frame.size()<32||local.frame.size()>32+v3::max_frame)
            throw std::runtime_error("v3 local frame construction failed");
        rkva_request h{};if(!v3::parse_request_header(local.frame.data(),32,h) || h.payload_bytes!=local.frame.size()-32 ||
            (h.op!=RKVA_REGISTRY_PREPARE && h.op!=RKVA_REGISTRY_COMMIT && h.op!=RKVA_REGISTRY_STATUS))
            throw std::runtime_error("v3 operation or exact frame length refused");
        std::vector<uint8_t> body(local.frame.begin()+32,local.frame.end());v3::fields f;rkva_registry_wire::request q{};
        if(!v3::command_from_wire(secret,session,h,body,f,q)||f.nonce!=local.nonce||
           f.host_anchor.version!=s.anchor.version||f.host_anchor.digest!=s.anchor.digest||
           !connection->held_and_bound(session))throw std::runtime_error("v3 frame authentication or anchor mismatch");
        const std::string tx=rkva_registry_wire::to_hex(q.tx);auto & record=get(s,tx);
        if((h.op==RKVA_REGISTRY_PREPARE && (record.state!=phase::prepared||
            record.reference!=rkva_registry_wire::to_hex(q.reference)||record.key!=rkva_registry_wire::to_hex(q.key)))||
           (h.op==RKVA_REGISTRY_COMMIT && record.state!=phase::host_published)||
           (h.op==RKVA_REGISTRY_STATUS && record.state!=phase::host_published && record.state!=phase::commit_ack))
            throw std::runtime_error("v3 transaction phase or identity differs");
        for(const auto & old:s.anchor.nonces)if(old.nonce==local.nonce)throw std::runtime_error("nonce reuse rejected");
        for(const auto & old:s.anchor.v3_nonces)
            if(old.nonce==local.nonce || old.challenge==session.peer_challenge)throw std::runtime_error("v3 nonce or session challenge reuse rejected");
        local.request_hash=rkva_snapshot::digest(local.frame.data(),local.frame.size());
        issued_v3_nonce entry;entry.tx=tx;entry.reference=record.reference;entry.key=record.key;entry.op=h.op;
        entry.nonce=local.nonce;entry.request_hash=local.request_hash;entry.transcript=session.transcript;
        entry.challenge=session.peer_challenge;entry.anchor_version=s.anchor.version;entry.anchor_digest=s.anchor.digest;
        s.anchor.v3_nonces.push_back(std::move(entry));return true;
    });
    if(ok) {
        try {
            if(!connection || !connection->held_and_bound(session)) {e="connection lost after issuance; nonce remains pending";return false;}
            output=std::move(local);
        } catch(...) {e="connection check failed after issuance; nonce remains pending";return false;}
    }
    return ok;
}
bool journal::consume_v3_error(const auth_id & nonce,const auth_id & request_hash,const auth_id & response_hash,
        const std::string & owner,uint64_t epoch,const auth_id & ledger,std::string & e) {
    return change(dir_,owner_,e,[&](snapshot & s){
        if(!s.anchor.configured||!s.anchor.v3_configured||owner_!=owner||s.anchor.key_epoch!=epoch||
           s.anchor.ledger_epoch!=ledger||!nonzero(response_hash))throw std::runtime_error("v3 error ledger identity differs");
        auto it=std::find_if(s.anchor.v3_nonces.begin(),s.anchor.v3_nonces.end(),[&](const issued_v3_nonce & x){return x.nonce==nonce;});
        if(it==s.anchor.v3_nonces.end()||it->consumed||it->request_hash!=request_hash)
            throw std::runtime_error("v3 error nonce absent, consumed or request differs");
        it->consumed=true;it->response_hash=response_hash;return true;
    });
}
bool journal::release_intent(const std::string & tx,uint64_t ver,std::string & e) { return change(dir_,owner_,e,[&](snapshot & s){ if(s.anchor.configured)throw std::runtime_error("authenticated release unsupported");auto & r=get(s,tx); if(!ver || r.peer_commit_version!=ver) throw std::runtime_error("release version mismatch"); if(r.state==phase::commit_ack) { r.state=phase::release_intent; return true; } if(r.state==phase::release_intent || r.state==phase::release_ack) return false; throw std::runtime_error("invalid release intent transition"); }); }
bool journal::release_ack(const std::string & tx,uint64_t ver,std::string & e) { return change(dir_,owner_,e,[&](snapshot & s){ if(s.anchor.configured)throw std::runtime_error("authenticated release unsupported");auto & r=get(s,tx); if(!ver || r.peer_commit_version!=ver) throw std::runtime_error("release version mismatch"); if(r.state==phase::release_intent) { r.state=phase::release_ack; return true; } if(r.state==phase::release_ack) return false; throw std::runtime_error("invalid release ack transition"); }); }
bool journal::abort_unpublished(const std::string & tx,std::string & e) { return change(dir_,owner_,e,[&](snapshot & s){ if(s.anchor.configured)throw std::runtime_error("authenticated abort unsupported");auto & r=get(s,tx); if(r.state==phase::prepared) { r.state=phase::aborted; return true; } if(r.state==phase::aborted) return false; throw std::runtime_error("published transaction cannot be aborted"); }); }
#else
bool journal::initialize(std::string & e) { return unsupported(e); }
bool journal::inspect(snapshot &,std::string & e) const { return unsupported(e); }
bool journal::prepare(const std::string &,const std::string &,const std::string &,std::string & e) { return unsupported(e); }
bool journal::host_published(const std::string &,const std::string &,const std::string &,uint64_t,std::string & e) { return unsupported(e); }
bool journal::commit_ack(const std::string &,uint64_t,std::string & e) { return unsupported(e); }
bool journal::provision_anchor(uint64_t,uint64_t,const auth_id &,std::string & e) { return unsupported(e); }
bool journal::issue_commit_nonce(const std::string &,const auth_id &,const std::string &,uint64_t,std::string & e) { return unsupported(e); }
bool journal::commit_verified(const std::string &,const auth_id &,uint64_t,uint64_t,const auth_id &,const std::string &,const std::string &,const std::string &,uint64_t,std::string & e) { return unsupported(e); }
bool journal::configure_v3_epoch(uint64_t,const auth_id &,uint64_t,const auth_id &,std::string & e) { return unsupported(e); }
bool journal::issue_v3_command(const rkva_managed_framing_v3::secret &,const rkva_managed_framing_v3::session &,v3_connection_authority *,const v3_frame_builder &,v3_issued_request &,std::string & e) { return unsupported(e); }
bool journal::consume_v3_error(const auth_id &,const auth_id &,const auth_id &,const std::string &,uint64_t,const auth_id &,std::string & e) { return unsupported(e); }
bool journal::release_intent(const std::string &,uint64_t,std::string & e) { return unsupported(e); }
bool journal::release_ack(const std::string &,uint64_t,std::string & e) { return unsupported(e); }
bool journal::abort_unpublished(const std::string &,std::string & e) { return unsupported(e); }
#endif
} // namespace rkva_host_journal
