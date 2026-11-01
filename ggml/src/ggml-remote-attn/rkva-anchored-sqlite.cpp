#include "rkva-anchored-sqlite.h"
#include "rkva-snapshot-digest.h"
#include "rkva-native-snapshot.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <winsqlite/winsqlite3.h>
#elif defined(__linux__)
#include <cerrno>
#include <sys/random.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sqlite3.h>
#else
#include <sqlite3.h>
#endif
namespace rkva_anchored_sqlite {
#ifdef RKVA_ANCHORED_SQLITE_TESTING
void (*test_hook)(const char *) = nullptr;
#endif
namespace {
constexpr int64_t max_version = std::numeric_limits<int64_t>::max();
constexpr int max_refs = 50000, max_nonces = 100000;
using fs = std::filesystem::path;
void wipe(void * ptr,size_t n) { volatile uint8_t * p=static_cast<volatile uint8_t *>(ptr); while(n--) *p++=0; }
bool empty(const id & a) { uint8_t c=0; for(auto b:a)c|=b; return c==0; }
bool equal(const id & a,const id & b) { volatile uint8_t c=0; for(size_t i=0;i<a.size();++i)c=uint8_t(c|(a[i]^b[i])); return c==0; }
void put64(uint8_t * p,uint64_t n) { for(unsigned i=0;i<8;i++)p[i]=uint8_t(n>>(8*i)); }
id hmac(const secret & key,const uint8_t * data,size_t n) {
    std::array<uint8_t,64> in{},out{}; for(size_t i=0;i<64;i++) { const uint8_t b=i<key.size()?key[i]:0; in[i]=b^0x36;out[i]=b^0x5c; }
    rkva_snapshot::Sha256 a;a.update(in.data(),in.size());a.update(data,n);auto mid=a.finish();
    rkva_snapshot::Sha256 b;b.update(out.data(),out.size());b.update(mid.data(),mid.size());auto mac=b.finish();
    wipe(&a,sizeof a);wipe(&b,sizeof b);wipe(mid.data(),mid.size());wipe(in.data(),in.size());wipe(out.data(),out.size());return mac;
}
id anchor_tag(const secret & key,const anchor & a) {
    static constexpr char domain[]="RKVA-ANCHOR-STATE-V1";
    std::array<uint8_t,sizeof(domain)-1+32*4+8> buf{};size_t n=0;
    auto add=[&](const void * p,size_t s) { std::memcpy(buf.data()+n,p,s);n+=s; };
    add(domain,sizeof(domain)-1);add(a.owner.data(),32);add(a.epoch.data(),32);add(a.challenge.data(),32);
    uint8_t version[8];put64(version,a.accepted.version);add(version,8);add(a.accepted.digest.data(),32);
    return hmac(key,buf.data(),n);
}
bool random_bytes(id & out) {
#if defined(_WIN32)
    return BCryptGenRandom(nullptr,out.data(),ULONG(out.size()),BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#elif defined(__linux__)
    size_t n=0;while(n<out.size()) { ssize_t k=getrandom(out.data()+n,out.size()-n,0);if(k<0&&errno==EINTR)continue;if(k<=0)return false;n+=size_t(k); }return true;
#else
    (void)out;return false;
#endif
}
struct db_guard { sqlite3 * p=nullptr; ~db_guard(){if(p)sqlite3_close(p);} };
struct statement { sqlite3_stmt * p=nullptr; ~statement(){if(p)sqlite3_finalize(p);} };
void need(bool b,const char * msg) { if(!b)throw std::runtime_error(msg); }
void exec(sqlite3 * db,const char * q) { need(sqlite3_exec(db,q,nullptr,nullptr,nullptr)==SQLITE_OK,"SQLite operation failed"); }
statement * prep(sqlite3 * db,const char * q,statement & s) { need(sqlite3_prepare_v2(db,q,-1,&s.p,nullptr)==SQLITE_OK,"SQLite prepare failed");return &s; }
void bind_id(sqlite3_stmt * s,int i,const id & a) { need(sqlite3_bind_blob(s,i,a.data(),32,SQLITE_TRANSIENT)==SQLITE_OK,"SQLite bind failed"); }
void bind_ver(sqlite3_stmt * s,int i,uint64_t n) { need(n<=uint64_t(max_version) && sqlite3_bind_int64(s,i,sqlite3_int64(n))==SQLITE_OK,"invalid version"); }
id column_id(sqlite3_stmt * s,int col) { need(sqlite3_column_type(s,col)==SQLITE_BLOB && sqlite3_column_bytes(s,col)==32,"invalid SQLite identifier");id x{};std::memcpy(x.data(),sqlite3_column_blob(s,col),32);return x; }
uint64_t column_num(sqlite3_stmt * s,int col) { need(sqlite3_column_type(s,col)==SQLITE_INTEGER && sqlite3_column_int64(s,col)>=0,"invalid SQLite integer");return uint64_t(sqlite3_column_int64(s,col)); }
void done(sqlite3_stmt * s) { need(sqlite3_step(s)==SQLITE_DONE,"SQLite statement failed"); }
std::string utf8(const fs & path) { const auto text=path.u8string(); return std::string(reinterpret_cast<const char *>(text.data()),text.size()); }
// C++17 UTF-8 -> filesystem::path. std::filesystem::u8path is deprecated in
// newer standard libraries, so the Windows branch decodes UTF-8 explicitly and
// keeps the same narrow/UTF-8 contract the prototype had.
fs path_from_utf8(const std::string & p) {
#if defined(_WIN32)
    if (p.empty()) return {};
    const int needed=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,p.data(),int(p.size()),nullptr,0);
    if (needed<=0) return {};
    std::wstring wide(size_t(needed),L'\0');
    MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,p.data(),int(p.size()),wide.data(),needed);
    return fs(wide);
#else
    return fs(p);
#endif
}
fs root_path(const std::string & p) {
    std::error_code ec;const auto resolved=std::filesystem::canonical(path_from_utf8(p),ec);
    need(!ec,"private parent unavailable");
    for(const auto & component:resolved) {
#if defined(_WIN32)
        need(_wcsicmp(component.c_str(),L"rkva-snapshots-v1")!=0 &&
             _wcsicmp(component.c_str(),L"rkva-snapshots-v2")!=0,"legacy namespace forbidden");
#else
        need(component!="rkva-snapshots-v1" && component!="rkva-snapshots-v2","legacy namespace forbidden");
#endif
    }
    return resolved/"rkva-managed-v3";
}
void safe_namespace(const fs & root) {
    std::error_code ec;auto st=std::filesystem::symlink_status(root,ec);
    need(!ec && std::filesystem::is_directory(st) && !std::filesystem::is_symlink(st),"managed namespace unavailable");
    // Any backup, journal, WAL or unknown path is ambiguous. Never invoke
    // SQLite automatic recovery before an independent operator reconciles it.
    for(std::filesystem::directory_iterator i(root,ec), end; !ec && i!=end; i.increment(ec)) {
        need(i->path().filename()=="peer.db","ambiguous managed store file");
        const auto status=i->symlink_status(ec);need(!ec && std::filesystem::is_regular_file(status) && !std::filesystem::is_symlink(status),"unsafe database path");
        const auto links=std::filesystem::hard_link_count(i->path(),ec);need(!ec && links==1,"unsafe database hardlinks");
    }
    need(!ec,"managed namespace scan failed");
}
void open_existing(const fs & root,db_guard & db) {
    safe_namespace(root);
    const auto path=utf8(root/"peer.db");need(sqlite3_open_v2(path.c_str(),&db.p,SQLITE_OPEN_READWRITE,nullptr)==SQLITE_OK,"managed database unavailable");
    sqlite3_busy_timeout(db.p,0);
    statement mode;prep(db.p,"PRAGMA journal_mode",mode);need(sqlite3_step(mode.p)==SQLITE_ROW,"journal mode unavailable");
    const auto p=reinterpret_cast<const char *>(sqlite3_column_text(mode.p,0));need(p && std::strcmp(p,"delete")==0,"unsafe SQLite journal mode");
    exec(db.p,"PRAGMA synchronous=EXTRA");statement sync;prep(db.p,"PRAGMA synchronous",sync);
    need(sqlite3_step(sync.p)==SQLITE_ROW && sqlite3_column_int(sync.p,0)==3,"SQLite sync mode unavailable");
}
struct txn { sqlite3 * p; bool active=false; explicit txn(sqlite3 * d):p(d){exec(p,"BEGIN IMMEDIATE");active=true;} ~txn(){if(active)sqlite3_exec(p,"ROLLBACK",nullptr,nullptr,nullptr);} void commit(){exec(p,"COMMIT");active=false;} };
state snapshot(sqlite3 * db,const id & owner,const id & epoch) {
    statement meta;prep(db,"SELECT owner,epoch,version,quota FROM meta",meta);need(sqlite3_step(meta.p)==SQLITE_ROW,"registry metadata missing");
    need(equal(column_id(meta.p,0),owner) && equal(column_id(meta.p,1),epoch),"owner or epoch mismatch");
    state s;s.version=column_num(meta.p,2);const auto quota=column_num(meta.p,3);need(quota>0,"invalid quota");need(sqlite3_step(meta.p)==SQLITE_DONE,"registry metadata not unique");
    rkva_snapshot::Sha256 hash;static constexpr char domain[]="RKVA-SQLITE-STATE-V1";hash.update(domain,sizeof(domain)-1);hash.update(owner.data(),32);hash.update(epoch.data(),32);
    uint8_t number[8];put64(number,s.version);hash.update(number,8);put64(number,quota);hash.update(number,8);
    statement refs;prep(db,"SELECT tx,ref,obj,phase,commit_version FROM refs ORDER BY tx",refs);int rc=0,count=0;
    while((rc=sqlite3_step(refs.p))==SQLITE_ROW) { need(++count<=max_refs,"registry capacity exceeded");
        const auto tx=column_id(refs.p,0),ref=column_id(refs.p,1),obj=column_id(refs.p,2);const auto phase=column_num(refs.p,3),ver=column_num(refs.p,4);
        need(phase>=1&&phase<=4 && ((phase==2||phase==4)?ver>0&&ver<=s.version:ver==0),"invalid registry phase");
        const uint8_t tag='R';hash.update(&tag,1);hash.update(tx.data(),32);hash.update(ref.data(),32);hash.update(obj.data(),32);put64(number,phase);hash.update(number,8);put64(number,ver);hash.update(number,8);
    } need(rc==SQLITE_DONE,"registry scan failed");
    statement pins;prep(db,"SELECT tx,identity,hello,metadata,reservation FROM pins ORDER BY tx",pins);count=0;uint64_t total=0;
    while((rc=sqlite3_step(pins.p))==SQLITE_ROW) {
        need(++count<=max_refs,"pin capacity exceeded");const auto tx=column_id(pins.p,0),ident=column_id(pins.p,1);
        need(sqlite3_column_type(pins.p,2)==SQLITE_BLOB && sqlite3_column_bytes(pins.p,2)==80,"invalid pin hello");
        need(sqlite3_column_type(pins.p,3)==SQLITE_BLOB && sqlite3_column_bytes(pins.p,3)>0 && uint64_t(sqlite3_column_bytes(pins.p,3))<=rkva_native_snapshot::kMaxManifestBytes,"invalid pin metadata");
        const auto reservation=column_num(pins.p,4);need(reservation>0 && reservation<=quota-total,"invalid quota reservation");total+=reservation;
        const uint8_t tag='P';hash.update(&tag,1);hash.update(tx.data(),32);hash.update(ident.data(),32);
        hash.update(sqlite3_column_blob(pins.p,2),80);const auto len=uint64_t(sqlite3_column_bytes(pins.p,3));put64(number,len);hash.update(number,8);
        hash.update(sqlite3_column_blob(pins.p,3),size_t(len));put64(number,reservation);hash.update(number,8);
    } need(rc==SQLITE_DONE,"pin scan failed");
    statement nonces;prep(db,"SELECT nonce FROM nonces ORDER BY nonce",nonces);count=0;
    while((rc=sqlite3_step(nonces.p))==SQLITE_ROW) { need(++count<=max_nonces,"nonce capacity exceeded");const auto n=column_id(nonces.p,0);const uint8_t tag='N';hash.update(&tag,1);hash.update(n.data(),32); }
    need(rc==SQLITE_DONE,"nonce scan failed");s.digest=hash.finish();return s;
}
struct internal_record { bool found=false; id tx{},ref{},obj{};uint64_t phase=0,ver=0; };
internal_record lookup(sqlite3 * db,const id & key,bool by_ref) {
    statement s;prep(db,by_ref?"SELECT tx,ref,obj,phase,commit_version FROM refs WHERE ref=?":"SELECT tx,ref,obj,phase,commit_version FROM refs WHERE tx=?",s);bind_id(s.p,1,key);
    int rc=sqlite3_step(s.p);if(rc==SQLITE_DONE)return {};need(rc==SQLITE_ROW,"registry lookup failed");internal_record r;r.found=true;r.tx=column_id(s.p,0);r.ref=column_id(s.p,1);r.obj=column_id(s.p,2);r.phase=column_num(s.p,3);r.ver=column_num(s.p,4);need(sqlite3_step(s.p)==SQLITE_DONE,"duplicate registry entry");return r;
}
}
bool sign_anchor(const secret & key,anchor & a) { if(empty(key)||empty(a.owner)||empty(a.epoch)||empty(a.challenge))return false;a.mac=anchor_tag(key,a);return true; }
backend::backend(std::string p,id o,id e,secret k,uint64_t quota):parent_(std::move(p)),owner_(o),epoch_(e),key_(k),quota_bytes_(quota){}
backend::~backend(){wipe(key_.data(),key_.size());}
bool backend::provision(std::shared_ptr<rkva_managed_store::lease> authority,std::string & error) {
    auto operation=authority?authority->enter():std::unique_lock<std::recursive_mutex>{};
    if(!operation.owns_lock() || !authority->valid_for(parent_)){error="store authority unavailable";return false;}
    std::lock_guard<std::mutex> lk(mutex_);error.clear();
    try { need(!empty(owner_)&&!empty(epoch_)&&!empty(key_) && quota_bytes_>0 && quota_bytes_<=uint64_t(max_version),"invalid provisioning identity or quota");
        const auto parent=path_from_utf8(parent_);std::error_code ec;const auto ps=std::filesystem::symlink_status(parent,ec);
        need(!ec&&std::filesystem::is_directory(ps)&&!std::filesystem::is_symlink(ps),"private parent unavailable");
        const auto root=root_path(parent_);need(std::filesystem::create_directory(root,ec)&&!ec,"managed namespace already exists or cannot be created");
        db_guard db;const auto path=utf8(root/"peer.db");need(sqlite3_open_v2(path.c_str(),&db.p,SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE,nullptr)==SQLITE_OK,"SQLite provisioning failed");
        exec(db.p,"PRAGMA journal_mode=DELETE; PRAGMA synchronous=EXTRA; BEGIN IMMEDIATE; CREATE TABLE meta(owner BLOB NOT NULL,epoch BLOB NOT NULL,version INTEGER NOT NULL,quota INTEGER NOT NULL); CREATE TABLE refs(tx BLOB PRIMARY KEY,ref BLOB NOT NULL UNIQUE,obj BLOB NOT NULL,phase INTEGER NOT NULL,commit_version INTEGER NOT NULL); CREATE TABLE nonces(nonce BLOB PRIMARY KEY); CREATE TABLE pins(tx BLOB PRIMARY KEY,identity BLOB NOT NULL,hello BLOB NOT NULL,metadata BLOB NOT NULL,reservation INTEGER NOT NULL); CREATE TABLE receipts(tx BLOB PRIMARY KEY,ref BLOB NOT NULL,obj BLOB NOT NULL,obj_digest BLOB NOT NULL,nonce BLOB NOT NULL,prev_version INTEGER NOT NULL,prev_digest BLOB NOT NULL,next_version INTEGER NOT NULL,next_digest BLOB NOT NULL,mac BLOB NOT NULL);");
        statement s;prep(db.p,"INSERT INTO meta VALUES(?,?,0,?)",s);bind_id(s.p,1,owner_);bind_id(s.p,2,epoch_);bind_ver(s.p,3,quota_bytes_);done(s.p);
        need(authority->valid_for(parent_),"store authority lost");exec(db.p,"COMMIT");return true;
    }catch(const std::exception & ex){error=ex.what();return false;} }
bool backend::challenge(std::shared_ptr<rkva_managed_store::lease> authority,id & out,std::string & error) {
    auto operation=authority?authority->enter():std::unique_lock<std::recursive_mutex>{};
    if(!operation.owns_lock() || !authority->valid_for(parent_)){error="store authority unavailable";return false;}
    std::lock_guard<std::mutex> lk(mutex_);pending_=false;challenge_authority_.reset();error.clear();
    if(!random_bytes(challenge_)||empty(challenge_)){error="secure challenge unavailable";return false;}
    pending_=true;challenge_authority_=authority;out=challenge_;return true;
}
bool backend::inspect(std::shared_ptr<rkva_managed_store::lease> authority,state & out,std::string & error) {
    auto operation=authority?authority->enter():std::unique_lock<std::recursive_mutex>{};
    if(!operation.owns_lock() || !authority->valid_for(parent_)){error="store authority unavailable";return false;}
    std::lock_guard<std::mutex> lk(mutex_);error.clear();try{db_guard db;open_existing(root_path(parent_),db);txn t(db.p);const auto verified=snapshot(db.p,owner_,epoch_);need(authority->valid_for(parent_),"store authority lost");t.commit();out=verified;return true;}catch(const std::exception & ex){error=ex.what();return false;}}
result backend::prepare_pending(std::shared_ptr<rkva_managed_store::lease> authority,const anchor * a,const pending_request & c,state & out,std::string & error) {
    auto lock=authority?authority->enter():std::unique_lock<std::recursive_mutex>{};
    if(!lock.owns_lock() || !authority->valid_for(parent_)){error="store authority unavailable";return result::unavailable;}
    std::lock_guard<std::mutex> lk(mutex_);error.clear();const bool had=pending_;pending_=false;
    const auto issued=challenge_authority_.lock();challenge_authority_.reset();
    if(!had || issued!=authority || !a || empty(key_) || !equal(a->owner,owner_) || !equal(a->epoch,epoch_) ||
       !equal(a->challenge,challenge_) || !equal(a->mac,anchor_tag(key_,*a))) {error="missing or invalid authenticated anchor/authority";return result::invalid_anchor;}
    try {
        need(equal(c.owner,owner_) && !empty(c.tx)&&!empty(c.reference)&&!empty(c.key)&&!empty(c.nonce)&&!empty(c.identity),"invalid pending identity");
        rkva_native_snapshot::Manifest manifest;std::vector<uint8_t> canonical;std::string why;
        need(rkva_native_snapshot::decode(c.metadata,manifest,&why) && rkva_native_snapshot::encode(manifest,canonical,&why) && canonical==c.metadata &&
             std::memcmp(manifest.host_state_sha256,c.key.data(),32)==0,"invalid canonical metadata");
        constexpr uint64_t overhead=1ull<<20;
        need(manifest.payload_bytes<=uint64_t(max_version)-overhead-c.metadata.size() &&
             c.reservation>=manifest.payload_bytes+c.metadata.size()+overhead && c.reservation<=uint64_t(max_version),"invalid reservation");
        static constexpr char domain[]="RKVA-PREPARE-V3-METADATA-V1";rkva_snapshot::Sha256 sha;sha.update(domain,sizeof(domain)-1);
        sha.update(c.owner.data(),32);sha.update(c.tx.data(),32);sha.update(c.reference.data(),32);sha.update(c.key.data(),32);
        sha.update(c.semantic_hello.data(),c.semantic_hello.size());sha.update(c.metadata.data(),c.metadata.size());need(equal(sha.finish(),c.identity),"identity digest mismatch");
        db_guard db;open_existing(root_path(parent_),db);txn t(db.p);const auto before=snapshot(db.p,owner_,epoch_);
        if(before.version!=a->accepted.version||!equal(before.digest,a->accepted.digest)){error="anchor and managed state differ";return result::stale_anchor;}
        need(before.version<uint64_t(max_version),"registry version exhausted");
        statement cnt;prep(db.p,"SELECT count(*) FROM nonces",cnt);need(sqlite3_step(cnt.p)==SQLITE_ROW && sqlite3_column_int64(cnt.p,0)<max_nonces,"nonce capacity exhausted");
        statement ns;prep(db.p,"INSERT INTO nonces VALUES(?)",ns);bind_id(ns.p,1,c.nonce);const int nr=sqlite3_step(ns.p);
        if(nr==SQLITE_CONSTRAINT){error="nonce replay";return result::replay;}need(nr==SQLITE_DONE,"nonce admission failed");
        auto r=lookup(db.p,c.tx,false);
        if(r.found) {
            need(equal(r.ref,c.reference)&&equal(r.obj,c.key)&&r.phase==1,"transaction conflict");
            statement pin;prep(db.p,"SELECT identity,hello,metadata,reservation FROM pins WHERE tx=?",pin);bind_id(pin.p,1,c.tx);
            need(sqlite3_step(pin.p)==SQLITE_ROW && equal(column_id(pin.p,0),c.identity) &&
                 sqlite3_column_type(pin.p,1)==SQLITE_BLOB && sqlite3_column_bytes(pin.p,1)==80 &&
                 std::memcmp(sqlite3_column_blob(pin.p,1),c.semantic_hello.data(),80)==0 &&
                 sqlite3_column_type(pin.p,2)==SQLITE_BLOB && size_t(sqlite3_column_bytes(pin.p,2))==c.metadata.size() &&
                 std::memcmp(sqlite3_column_blob(pin.p,2),c.metadata.data(),c.metadata.size())==0 && column_num(pin.p,3)==c.reservation &&
                 sqlite3_step(pin.p)==SQLITE_DONE,"pending metadata conflict");
        } else {
            statement capacity;prep(db.p,"SELECT count(*),coalesce(sum(reservation),0) FROM pins",capacity);
            need(sqlite3_step(capacity.p)==SQLITE_ROW && column_num(capacity.p,0)<uint64_t(max_refs),"pin capacity exhausted");
            const auto total=column_num(capacity.p,1);statement q;prep(db.p,"SELECT quota FROM meta",q);need(sqlite3_step(q.p)==SQLITE_ROW,"quota unavailable");
            const auto quota=column_num(q.p,0);need(total<=quota && c.reservation<=quota-total,"quota exhausted");
            statement ins;prep(db.p,"INSERT INTO refs VALUES(?,?,?,1,0)",ins);bind_id(ins.p,1,c.tx);bind_id(ins.p,2,c.reference);bind_id(ins.p,3,c.key);done(ins.p);
            statement pin;prep(db.p,"INSERT INTO pins VALUES(?,?,?,?,?)",pin);bind_id(pin.p,1,c.tx);bind_id(pin.p,2,c.identity);
            need(sqlite3_bind_blob(pin.p,3,c.semantic_hello.data(),80,SQLITE_TRANSIENT)==SQLITE_OK &&
                 sqlite3_bind_blob(pin.p,4,c.metadata.data(),int(c.metadata.size()),SQLITE_TRANSIENT)==SQLITE_OK,"metadata bind failed");
            bind_ver(pin.p,5,c.reservation);done(pin.p);
        }
        statement mv;prep(db.p,"UPDATE meta SET version=?",mv);bind_ver(mv.p,1,before.version+1);done(mv.p);const auto after=snapshot(db.p,owner_,epoch_);
#ifdef RKVA_ANCHORED_SQLITE_TESTING
        if(test_hook)test_hook("prepare_before_commit");
#endif
        need(authority->valid_for(parent_),"store authority lost");t.commit();
#ifdef RKVA_ANCHORED_SQLITE_TESTING
        if(test_hook)test_hook("prepare_after_commit");
#endif
        out=after;return result::ok;
    }catch(const std::exception & ex){error=ex.what();return result::unavailable;}
}
#if defined(RKVA_ANCHORED_SQLITE_TESTING) && defined(__linux__)
// Deliberately test-only: fixture file is not an RKVA v3 object. The separate
// digest is NOT the host prefix key. Do not export this verifier to runtime.
namespace {
bool fixture_object(const std::string & parent,const id & digest) {
    const auto path=std::filesystem::path(parent)/"fixture-object";
    int fd=open(path.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC);if(fd<0)return false;
    struct stat a{},b{};bool ok=fstat(fd,&a)==0 && S_ISREG(a.st_mode) && a.st_nlink==1 && a.st_size>0 && a.st_size<=1024*1024;
    rkva_snapshot::Sha256 sha;std::array<uint8_t,4096> buf{};off_t count=0;
    while(ok && count<a.st_size) {const auto n=read(fd,buf.data(),std::min(size_t(a.st_size-count),buf.size()));if(n<=0){ok=false;break;}sha.update(buf.data(),size_t(n));count+=n;}
    ok=ok && fstat(fd,&b)==0 && a.st_dev==b.st_dev && a.st_ino==b.st_ino && a.st_size==b.st_size &&
       a.st_mtim.tv_sec==b.st_mtim.tv_sec && a.st_mtim.tv_nsec==b.st_mtim.tv_nsec && equal(sha.finish(),digest);
    close(fd);return ok;
}
using receipt=backend::test_receipt;
id receipt_mac(const secret & key,const receipt & r) {
    static constexpr char domain[]="RKVA-TEST-RECEIPT-V1";std::vector<uint8_t> b(domain,domain+sizeof(domain)-1);
    auto add=[&](const id & v){b.insert(b.end(),v.begin(),v.end());};
    add(r.owner);add(r.epoch);add(r.tx);add(r.reference);add(r.key);add(r.object_digest);add(r.nonce);
    uint8_t n[8];put64(n,r.prior.version);b.insert(b.end(),n,n+8);add(r.prior.digest);
    put64(n,r.next.version);b.insert(b.end(),n,n+8);add(r.next.digest);return hmac(key,b.data(),b.size());
}
id status_mac(const secret & key,const backend::test_status & s) {
    static constexpr char domain[]="RKVA-TEST-STATUS-V1";std::vector<uint8_t> b(domain,domain+sizeof(domain)-1);
    b.insert(b.end(),s.challenge.begin(),s.challenge.end());b.insert(b.end(),s.receipt.mac.begin(),s.receipt.mac.end());
    return hmac(key,b.data(),b.size());
}
}
result backend::test_commit(std::shared_ptr<rkva_managed_store::lease> authority,const anchor * a,const command & c,const id & object_digest,test_receipt & output,std::string & error) {
    auto lock=authority?authority->enter():std::unique_lock<std::recursive_mutex>{};
    if(!lock.owns_lock()||!authority->valid_for(parent_)){error="store authority unavailable";return result::unavailable;}
    std::lock_guard<std::mutex> lk(mutex_);error.clear();const bool had=pending_;pending_=false;
    auto issued=challenge_authority_.lock();challenge_authority_.reset();
    if(!had||issued!=authority||!a||!equal(a->owner,owner_)||!equal(a->epoch,epoch_)||!equal(a->challenge,challenge_)||!equal(a->mac,anchor_tag(key_,*a))){error="invalid anchor";return result::invalid_anchor;}
    if(c.op!=operation::commit||empty(c.nonce)||!fixture_object(parent_,object_digest)){error="fixture object unavailable";return result::denied;}
    try {db_guard db;open_existing(root_path(parent_),db);txn t(db.p);const auto before=snapshot(db.p,owner_,epoch_);
        if(before.version!=a->accepted.version||!equal(before.digest,a->accepted.digest)){error="stale anchor";return result::stale_anchor;}
        auto r=lookup(db.p,c.tx,false);need(r.found && r.phase==1 && equal(r.ref,c.reference)&&equal(r.obj,c.key),"pending pin mismatch");
        statement pin;prep(db.p,"SELECT count(*) FROM pins WHERE tx=?",pin);bind_id(pin.p,1,c.tx);need(sqlite3_step(pin.p)==SQLITE_ROW && column_num(pin.p,0)==1,"missing pin");
        need(before.version<uint64_t(max_version),"version exhausted");
        statement ns;prep(db.p,"INSERT INTO nonces VALUES(?)",ns);bind_id(ns.p,1,c.nonce);auto rc=sqlite3_step(ns.p);if(rc==SQLITE_CONSTRAINT)return result::replay;need(rc==SQLITE_DONE,"nonce failed");
        statement u;prep(db.p,"UPDATE refs SET phase=2,commit_version=? WHERE tx=?",u);bind_ver(u.p,1,before.version+1);bind_id(u.p,2,c.tx);done(u.p);
        statement m;prep(db.p,"UPDATE meta SET version=?",m);bind_ver(m.p,1,before.version+1);done(m.p);
        receipt rcp;rcp.owner=owner_;rcp.epoch=epoch_;rcp.tx=c.tx;rcp.reference=c.reference;rcp.key=c.key;rcp.object_digest=object_digest;rcp.nonce=c.nonce;
        rcp.prior=before;rcp.next=snapshot(db.p,owner_,epoch_);rcp.mac=receipt_mac(key_,rcp);
        statement ins;prep(db.p,"INSERT INTO receipts VALUES(?,?,?,?,?,?,?,?,?,?)",ins);bind_id(ins.p,1,c.tx);bind_id(ins.p,2,c.reference);bind_id(ins.p,3,c.key);
        bind_id(ins.p,4,object_digest);bind_id(ins.p,5,c.nonce);bind_ver(ins.p,6,before.version);bind_id(ins.p,7,before.digest);
        bind_ver(ins.p,8,rcp.next.version);bind_id(ins.p,9,rcp.next.digest);bind_id(ins.p,10,rcp.mac);done(ins.p);
        // Receipts are authenticated independently and excluded from the state
        // digest to avoid a digest/receipt cycle. Missing or altered receipt
        // fails closed at STATUS; it cannot authorize a different state.
        if(test_hook)test_hook("receipt_before_commit");
        need(authority->valid_for(parent_) && fixture_object(parent_,object_digest),"fixture object changed");t.commit();
        if(test_hook)test_hook("receipt_after_commit");
        output=rcp;return result::ok;
    }catch(const std::exception & ex){error=ex.what();return result::unavailable;}
}
result backend::test_status_read(std::shared_ptr<rkva_managed_store::lease> authority,const anchor * a,const command & c,const id & request_mac,test_status & output,std::string & error) {
    auto lock=authority?authority->enter():std::unique_lock<std::recursive_mutex>{};
    if(!lock.owns_lock()||!authority->valid_for(parent_)){error="store authority unavailable";return result::unavailable;}
    std::lock_guard<std::mutex> lk(mutex_);error.clear();const bool had=pending_;pending_=false;
    auto issued=challenge_authority_.lock();challenge_authority_.reset();
    if(!had||issued!=authority||!a||!equal(a->owner,owner_)||!equal(a->epoch,epoch_)||!equal(a->challenge,challenge_)||!equal(a->mac,anchor_tag(key_,*a))){error="invalid anchor";return result::invalid_anchor;}
    if(c.op!=operation::status||empty(c.tx)||empty(c.reference)||empty(c.key)||!equal(request_mac,test_sign_status(key_,*a,c))){error="invalid STATUS";return result::denied;}
    try {db_guard db;open_existing(root_path(parent_),db);txn t(db.p);auto now=snapshot(db.p,owner_,epoch_);
        statement q;prep(db.p,"SELECT ref,obj,obj_digest,nonce,prev_version,prev_digest,next_version,next_digest,mac FROM receipts WHERE tx=?",q);bind_id(q.p,1,c.tx);
        need(sqlite3_step(q.p)==SQLITE_ROW,"receipt missing");test_status reply;auto & r=reply.receipt;r.owner=owner_;r.epoch=epoch_;r.tx=c.tx;
        r.reference=column_id(q.p,0);r.key=column_id(q.p,1);r.object_digest=column_id(q.p,2);r.nonce=column_id(q.p,3);
        r.prior.version=column_num(q.p,4);r.prior.digest=column_id(q.p,5);r.next.version=column_num(q.p,6);r.next.digest=column_id(q.p,7);r.mac=column_id(q.p,8);
        need(sqlite3_step(q.p)==SQLITE_DONE && equal(r.reference,c.reference)&&equal(r.key,c.key) &&
             r.prior.version==a->accepted.version && equal(r.prior.digest,a->accepted.digest) && r.next.version==now.version && equal(r.next.digest,now.digest) &&
             equal(r.mac,receipt_mac(key_,r)) && fixture_object(parent_,r.object_digest),"receipt or fixture proof mismatch");
        reply.challenge=a->challenge;reply.mac=status_mac(key_,reply);
        need(authority->valid_for(parent_),"store authority lost");t.commit();output=reply;return result::ok;
    }catch(const std::exception & ex){error=ex.what();return result::unavailable;}
}
id backend::test_sign_status(const secret & key,const anchor & a,const command & c) {
    static constexpr char domain[]="RKVA-TEST-STATUS-REQUEST-V1";std::vector<uint8_t> b(domain,domain+sizeof(domain)-1);
    b.insert(b.end(),a.mac.begin(),a.mac.end());b.insert(b.end(),c.tx.begin(),c.tx.end());
    b.insert(b.end(),c.reference.begin(),c.reference.end());b.insert(b.end(),c.key.begin(),c.key.end());return hmac(key,b.data(),b.size());
}
bool backend::test_validate_status(const secret & key,const anchor & a,const command & c,const test_status & s) {
    const auto & r=s.receipt;
    return c.op==operation::status && equal(a.owner,r.owner)&&equal(a.epoch,r.epoch)&&equal(a.challenge,s.challenge)&&
        equal(c.tx,r.tx)&&equal(c.reference,r.reference)&&equal(c.key,r.key)&&r.prior.version==a.accepted.version&&
        equal(r.prior.digest,a.accepted.digest)&&r.next.version==r.prior.version+1&&equal(r.mac,receipt_mac(key,r))&&equal(s.mac,status_mac(key,s));
}
#endif
result backend::apply(std::shared_ptr<rkva_managed_store::lease> authority,const anchor * a,const command & c,state & out,std::string & error,uint64_t * committed_version,record * returned_record,object_proof * proof) {
    auto operation=authority?authority->enter():std::unique_lock<std::recursive_mutex>{};
    if(!operation.owns_lock() || !authority->valid_for(parent_)){error="store authority unavailable";return result::unavailable;}
    std::lock_guard<std::mutex> lk(mutex_);error.clear();if(committed_version)*committed_version=0;const bool had=pending_;pending_=false;
    const auto issued=challenge_authority_.lock();challenge_authority_.reset();
    if(!had || issued!=authority || !a||empty(key_)||!equal(a->owner,owner_)||!equal(a->epoch,epoch_)||!equal(a->challenge,challenge_)||!equal(a->mac,anchor_tag(key_,*a))) {error="missing or invalid authenticated anchor/authority";return result::invalid_anchor;}
    if(empty(c.nonce)||unsigned(c.op)<1||unsigned(c.op)>6){error="invalid command";return result::denied;}
    try {db_guard db;open_existing(root_path(parent_),db);txn t(db.p);const auto before=snapshot(db.p,owner_,epoch_);
        if(before.version!=a->accepted.version||!equal(before.digest,a->accepted.digest)){error="anchor and managed state differ";return result::stale_anchor;}
        // The new store lock is real, but the copied prototype's boolean
        // callback does not prove authenticated metadata or a durable pin.
        // Until that concrete binding exists, no reference/object transition
        // (including mutable STATUS) is permitted in this managed variant.
        if(c.op!=operation::hello_claim){error="concrete authenticated pin and metadata proof unavailable";return result::denied;}
        need(before.version<uint64_t(max_version),"registry version exhausted");
        statement cnt;prep(db.p,"SELECT count(*) FROM nonces",cnt);need(sqlite3_step(cnt.p)==SQLITE_ROW && sqlite3_column_int64(cnt.p,0)<max_nonces,"nonce capacity exhausted");
        statement ns;prep(db.p,"INSERT INTO nonces VALUES(?)",ns);bind_id(ns.p,1,c.nonce);const int nr=sqlite3_step(ns.p);if(nr==SQLITE_CONSTRAINT){error="nonce replay";return result::replay;}need(nr==SQLITE_DONE,"nonce admission failed");
        const auto next=before.version+1;internal_record r;
        if(c.op==operation::prepare){need(!empty(c.tx)&&!empty(c.reference)&&!empty(c.key),"invalid prepare");r=lookup(db.p,c.tx,false);
            const rkva_anchored_sqlite::record candidate{c.tx,c.reference,c.key,uint32_t(r.found?r.phase:1),r.found?r.ver:0};
            if(!proof || !proof->verify(c.op,candidate,authority)){error="managed object proof unavailable";return result::denied;}
            if(r.found)need(equal(r.ref,c.reference)&&equal(r.obj,c.key)&&(r.phase==1||r.phase==2),"transaction conflict");
            else {statement count;prep(db.p,"SELECT count(*) FROM refs",count);need(sqlite3_step(count.p)==SQLITE_ROW && sqlite3_column_int64(count.p,0)<max_refs,"registry capacity exhausted");
                statement ins;prep(db.p,"INSERT INTO refs VALUES(?,?,?,1,0)",ins);bind_id(ins.p,1,c.tx);bind_id(ins.p,2,c.reference);bind_id(ins.p,3,c.key);done(ins.p);}}
        else if(c.op!=operation::hello_claim) {const bool byref=c.op==operation::release;r=lookup(db.p,byref?c.reference:c.tx,byref);need(r.found,"unknown transaction");uint64_t phase=r.phase,cv=r.ver;
            if(c.op==operation::commit){
                const rkva_anchored_sqlite::record candidate{r.tx,r.ref,r.obj,uint32_t(r.phase),r.ver};
                if(!proof || !proof->verify(c.op,candidate,authority)){error="managed object proof unavailable";return result::denied;}
                if(phase==1){phase=2;cv=next;}else need(phase==2,"commit conflict");}
            if(c.op==operation::abort || c.op==operation::release) {
                const rkva_anchored_sqlite::record candidate{r.tx,r.ref,r.obj,uint32_t(r.phase),r.ver};
                if(!proof || !proof->verify(c.op,candidate,authority)){error="managed transition proof unavailable";return result::denied;}
            }
            if(c.op==operation::abort){if(phase==1)phase=3;else need(phase==3,"abort conflict");}
            if(c.op==operation::release){need(c.commit_version && c.commit_version==cv,"release version conflict");if(phase==2)phase=4;else need(phase==4,"release conflict");}
            if(phase!=r.phase||cv!=r.ver){statement update;prep(db.p,"UPDATE refs SET phase=?,commit_version=? WHERE tx=?",update);bind_ver(update.p,1,phase);bind_ver(update.p,2,cv);bind_id(update.p,3,r.tx);done(update.p);}}
        statement mv;prep(db.p,"UPDATE meta SET version=?",mv);bind_ver(mv.p,1,next);done(mv.p);const auto after=snapshot(db.p,owner_,epoch_);
#ifdef RKVA_ANCHORED_SQLITE_TESTING
        if(test_hook)test_hook("before_commit");
#endif
        record result_record{};
        if(c.op!=operation::hello_claim) {
            const auto found=lookup(db.p,c.op==operation::release?c.reference:c.tx,c.op==operation::release);
            need(found.found,"committed record unavailable");
            result_record={found.tx,found.ref,found.obj,uint32_t(found.phase),found.ver};
        }
        need(authority->valid_for(parent_),"store authority lost");t.commit();
#ifdef RKVA_ANCHORED_SQLITE_TESTING
        if(test_hook)test_hook("after_commit");
#endif
        out=after;if(committed_version)*committed_version=result_record.committed_version;
        if(returned_record)*returned_record=result_record;
        return result::ok;
    }catch(const std::exception & ex){error=ex.what();return result::unavailable;}
}
}
