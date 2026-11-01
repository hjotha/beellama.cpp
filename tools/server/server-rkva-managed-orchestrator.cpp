#include "server-rkva-managed-orchestrator.h"
#include "../../ggml/include/rkva-snapshot-digest.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace rkva_managed_orchestrator {
#ifdef RKVA_MANAGED_ORCHESTRATOR_TESTING
void (*test_hook)(const char *) = nullptr;
#endif
namespace {
void hook(const char * name) {
#ifdef RKVA_MANAGED_ORCHESTRATOR_TESTING
    if (test_hook) test_hook(name);
#else
    (void) name;
#endif
}
bool hex64(const std::string & s) { return s.size()==64 && s.find_first_not_of("0123456789abcdef")==std::string::npos && s!=std::string(64,'0'); }
#ifndef _WIN32
struct fd {
    int n=-1; explicit fd(int f=-1):n(f) {} ~fd(){if(n>=0) ::close(n);}
    fd(const fd&)=delete; fd & operator=(const fd&)=delete;
};
void need(bool yes, const std::string & message) { if(!yes) throw std::runtime_error(message); }
void must(bool yes, const char * what, const std::string & error) {
    if (!yes) throw std::runtime_error(std::string(what) + ": " + error);
}
std::string trusted_dir(const std::string & p) {
    char * raw=::realpath(p.c_str(),nullptr); need(raw!=nullptr,"managed directory unavailable");
    std::string result(raw); std::free(raw);
    for (const auto & part:std::filesystem::path(result)) {
        need(part!="rkva-snapshots-v1" && part!="rkva-snapshots-v2","legacy snapshot namespace forbidden");
    }
    return result;
}
void regular(int f, unsigned max_links=1) {
    struct stat s{}; need(!::fstat(f,&s) && S_ISREG(s.st_mode) && s.st_nlink>0 && s.st_nlink<=max_links,"unsafe file or links");
}
void syncdir(int d) { need(::fsync(d)==0,"directory fsync failed; outcome uncertain"); }
struct store {
    fd host, peer, lock; std::string root;
    std::unique_lock<std::recursive_mutex> section;
    store(const std::string & h,const std::string & p,const std::shared_ptr<rkva_managed_store::delegation> & authority) {
        root=trusted_dir(h); const std::string objects=trusted_dir(p);
        need(root!=objects,"host and peer object stores must be distinct");
        host.n=::open(root.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);
        peer.n=::open(objects.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);
        need(host.n>=0 && peer.n>=0,"managed store open failed");
        if(authority) {
            // Single-authority composition: the authority already holds this
            // store's lock file. Never take a second, independent lock.
            section=authority->enter();
            need(section.owns_lock() && authority->valid_for(root) && authority->directory()==root,
                 "delegated store authority unavailable");
            syncdir(host.n); // settle a prior ambiguous rename/unlink before replay
            return;
        }
        lock.n=::openat(host.n,".managed-orchestrator.lock",O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW,0600);
        need(lock.n>=0,"managed store lock unavailable"); regular(lock.n);
        need(::flock(lock.n,LOCK_EX)==0,"managed store lock failed");
        syncdir(host.n); // settle a prior ambiguous rename/unlink before replay
    }
};
bool exists(int dir,const std::string & name) {
    struct stat s{}; if(::fstatat(dir,name.c_str(),&s,AT_SYMLINK_NOFOLLOW)==0) return true;
    need(errno==ENOENT,"cannot inspect managed file"); return false;
}
std::pair<std::string,uint64_t> hash_fd(int f,uint64_t max,unsigned links=1) {
    regular(f,links); struct stat before{},after{};
    need(!::fstat(f,&before) && before.st_size>0 && uint64_t(before.st_size)<=max,"managed file size invalid");
    need(::lseek(f,0,SEEK_SET)==0,"managed file seek failed");
    rkva_snapshot::Sha256 hash; std::array<char,65536> buf{}; uint64_t total=0;
    for(;;) { const ssize_t n=::read(f,buf.data(),buf.size()); if(n<0&&errno==EINTR) continue;
        need(n>=0,"managed file read failed"); if(!n) break;
        need(uint64_t(n)<=max-total,"managed file exceeds limit"); total+=uint64_t(n); hash.update(buf.data(),size_t(n)); }
    need(::fstat(f,&after)==0,"managed file reinspection failed");
#if defined(__APPLE__)
    const auto before_mtime = before.st_mtimespec, after_mtime = after.st_mtimespec;
#else
    const auto before_mtime = before.st_mtim, after_mtime = after.st_mtim;
#endif
    need(before.st_dev==after.st_dev && before.st_ino==after.st_ino &&
         before.st_size==after.st_size && uint64_t(after.st_size)==total &&
         before_mtime.tv_sec==after_mtime.tv_sec && before_mtime.tv_nsec==after_mtime.tv_nsec,
         "managed file changed during verification");
    const auto d=hash.finish(); return {rkva_snapshot::hex(d.data()),total};
}
std::pair<std::string,uint64_t> hash_at(int dir,const std::string & name,uint64_t max,unsigned links=1) {
    fd f(::openat(dir,name.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW)); need(f.n>=0,"managed file unavailable");
    return hash_fd(f.n,max,links);
}
void write_all(int f,const void * data,size_t size) {
    const char * p=static_cast<const char *>(data);
    while(size) { ssize_t n=::write(f,p,size); if(n<0&&errno==EINTR) continue; need(n>0,"managed write failed"); p+=n; size-=size_t(n); }
}
void unlink_file(int dir,const std::string & n) { need(::unlinkat(dir,n.c_str(),0)==0,"managed unlink failed"); }
void peer_object(store & s,const std::string & key,uint64_t max) {
    const auto actual=hash_at(s.peer.n,key+".blob",max); need(actual.first==key,"peer object digest mismatch");
}
rkva_host_journal::transaction lookup(rkva_host_journal::journal & j,const std::string & tx) {
    rkva_host_journal::snapshot snap; std::string error; must(j.inspect(snap,error),"journal inspect",error);
    for(const auto & r:snap.transactions) if(r.tx==tx) return r;
    throw std::runtime_error("unknown journal transaction");
}
void peer_check(rkva_managed_registry::registry & p,const rkva_host_journal::transaction & h,
                rkva_managed_registry::phase state,uint64_t version=0) {
    rkva_managed_registry::snapshot snap; std::string error; must(p.inspect(snap,error),"peer inspect",error);
    for(const auto & r:snap.transactions) if(r.id==h.tx) {
        need(r.reference==h.reference && r.key==h.key && r.state==state && (!version || r.committed_version==version),"peer transaction mismatch"); return;
    }
    throw std::runtime_error("peer transaction missing");
}
// Copies an input to a disposable partial. A fully synced stage is published
// by rename under the exclusive store lock. A partial left by crash is ignored
// and replaced only while no durable stage or final exists.
void stage_source(store & s,const std::string & tx,const std::string & key,const std::string & source,uint64_t max) {
    const std::string part="."+tx+".partial", stage="."+tx+".stage", final=tx+".managed";
    if(exists(s.host.n,stage) || exists(s.host.n,final)) return;
    need(!source.empty(),"source required to resume incomplete staging");
    fd in(::open(source.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW)); need(in.n>=0,"source unavailable");
    const auto expected=hash_fd(in.n,max); need(expected.first==key,"source digest differs from key");
    if(exists(s.host.n,part)) { fd old(::openat(s.host.n,part.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW)); need(old.n>=0,"unsafe partial"); regular(old.n); unlink_file(s.host.n,part); syncdir(s.host.n); }
    fd out(::openat(s.host.n,part.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600)); need(out.n>=0,"host staging create failed");
    need(::lseek(in.n,0,SEEK_SET)==0,"source seek failed"); std::array<char,65536> buf{};
    for(;;) { ssize_t n=::read(in.n,buf.data(),buf.size()); if(n<0&&errno==EINTR) continue; need(n>=0,"source read failed"); if(!n) break; write_all(out.n,buf.data(),size_t(n)); }
    hook("host_partial_written"); need(::fsync(out.n)==0,"host staging fsync failed");
    const auto copied=hash_at(s.host.n,part,max); need(copied==expected,"source changed during staging");
    need(::renameat(s.host.n,part.c_str(),s.host.n,stage.c_str())==0,"host stage rename failed");
    hook("host_stage_rename"); syncdir(s.host.n); hook("host_stage_durable");
}
void publish_stage(store & s,const std::string & tx,const std::string & key,uint64_t max) {
    const std::string stage="."+tx+".stage", final=tx+".managed";
    if(exists(s.host.n,final)) {
        // A crash between link and unlink leaves the same inode under both names.
        if(exists(s.host.n,stage)) {
            struct stat a{},b{};
            need(!::fstatat(s.host.n,final.c_str(),&a,AT_SYMLINK_NOFOLLOW) && !::fstatat(s.host.n,stage.c_str(),&b,AT_SYMLINK_NOFOLLOW) &&
                 S_ISREG(a.st_mode) && S_ISREG(b.st_mode) && a.st_dev==b.st_dev && a.st_ino==b.st_ino && a.st_nlink==2,
                 "conflicting final and staged host files");
            need(hash_at(s.host.n,final,max,2).first==key,"published host digest mismatch");
            unlink_file(s.host.n,stage); syncdir(s.host.n);
        }
        need(hash_at(s.host.n,final,max).first==key,"published host digest mismatch"); return;
    }
    need(hash_at(s.host.n,stage,max).first==key,"host staging digest mismatch");
    need(::linkat(s.host.n,stage.c_str(),s.host.n,final.c_str(),0)==0,"host publish failed");
    hook("host_linked"); unlink_file(s.host.n,stage); hook("host_stage_unlinked");
    syncdir(s.host.n); hook("host_published_durable");
}
std::string tomb_value(const rkva_host_journal::transaction & t) {
    return std::string("RKVARELEASE1\n")+t.tx+"\n"+t.reference+"\n"+t.key+"\n"+t.host_sha256+"\n"+
        std::to_string(t.host_bytes)+"\n"+std::to_string(t.peer_commit_version)+"\n";
}
void verify_tomb(int d,const std::string & path,const std::string & expected) {
    fd f(::openat(d,path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW)); need(f.n>=0,"release evidence missing"); regular(f.n);
    std::string actual(expected.size()+1,'\0'); ssize_t n=::read(f.n,&actual[0],actual.size());
    need(n>=0 && size_t(n)==expected.size() && actual.compare(0,size_t(n),expected)==0,"release evidence invalid");
}
void ensure_tomb(store & s,const rkva_host_journal::transaction & t,uint64_t max) {
    const std::string name="."+t.tx+".release", part="."+t.tx+".release-partial", final=t.tx+".managed";
    const std::string value=tomb_value(t);
    if(!exists(s.host.n,name)) {
        const auto actual=hash_at(s.host.n,final,max); need(actual.first==t.host_sha256 && actual.second==t.host_bytes,"host identity differs before release");
        if(exists(s.host.n,part)) { fd old(::openat(s.host.n,part.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW)); need(old.n>=0,"unsafe partial evidence"); regular(old.n); unlink_file(s.host.n,part); syncdir(s.host.n); }
        fd f(::openat(s.host.n,part.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600)); need(f.n>=0,"release evidence create failed");
        write_all(f.n,value.data(),value.size()); hook("release_partial_written"); need(::fsync(f.n)==0,"release evidence sync failed");
        need(::renameat(s.host.n,part.c_str(),s.host.n,name.c_str())==0,"release evidence publish failed");
        hook("release_tomb_renamed"); syncdir(s.host.n); hook("release_tomb_durable");
    }
    verify_tomb(s.host.n,name,value); syncdir(s.host.n);
    if(exists(s.host.n,final)) {
        const auto actual=hash_at(s.host.n,final,max); need(actual.first==t.host_sha256 && actual.second==t.host_bytes,"host identity differs at removal");
        unlink_file(s.host.n,final); hook("release_host_unlinked");
    }
    syncdir(s.host.n); hook("release_host_absence_durable");
}
#endif
}
orchestrator::orchestrator(rkva_host_journal::journal & j,rkva_managed_registry::registry & p,
        std::string host,std::string objects,uint64_t max,source_preflight check,
        std::shared_ptr<rkva_managed_store::delegation> authority):journal_(j),peer_(p),
        host_root_(std::move(host)),peer_objects_(std::move(objects)),max_bytes_(max),check_source_(std::move(check)),
        authority_(std::move(authority)) {}
bool orchestrator::publish(const std::string & tx,const std::string & ref,const std::string & key,const std::string & source,std::string & error) {
#ifndef _WIN32
    try {
        need(hex64(tx)&&hex64(ref)&&hex64(key)&&max_bytes_,"invalid managed identity or limit");
        store s(host_root_,peer_objects_,authority_); std::string e;
        need(!exists(s.host.n,"."+tx+".release") && !exists(s.host.n,"."+tx+".release-partial"),
             "release is in progress or terminal; publication forbidden");
        if (check_source_ && !source.empty()) {
            fd input(::open(source.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));
            need(input.n>=0,"source preflight open failed");regular(input.n);
            struct stat st{};need(::fstat(input.n,&st)==0 && st.st_size>0 && uint64_t(st.st_size)<=max_bytes_,"source preflight size invalid");
            std::string source_key;uint64_t bytes=0;
            must(check_source_(input.n,source_key,bytes,e),"source preflight",e);
            need(source_key==key && bytes==uint64_t(st.st_size),"source preflight identity mismatch");
        }
        must(journal_.prepare(tx,ref,key,e),"journal prepare",e); hook("journal_prepared");
        auto record=lookup(journal_,tx);
        need(record.reference==ref && record.key==key && (record.state==rkva_host_journal::phase::prepared ||
             record.state==rkva_host_journal::phase::host_published || record.state==rkva_host_journal::phase::commit_ack),"journal transaction cannot publish");
        peer_object(s,key,max_bytes_);
        must(peer_.prepare(tx,ref,key,e),"peer prepare",e); hook("peer_prepared");
        if(record.state==rkva_host_journal::phase::prepared) {
            stage_source(s,tx,key,source,max_bytes_); publish_stage(s,tx,key,max_bytes_);
            const auto actual=hash_at(s.host.n,tx+".managed",max_bytes_); need(actual.first==key,"host digest mismatch");
            must(journal_.host_published(tx,s.root+"/"+tx+".managed",actual.first,actual.second,e),"journal host publication",e);
            hook("journal_host_published"); record=lookup(journal_,tx);
        } else {
            const auto actual=hash_at(s.host.n,tx+".managed",max_bytes_);
            need(record.host_path==s.root+"/"+tx+".managed" && actual.first==record.host_sha256 &&
                 actual.first==key && actual.second==record.host_bytes,"published host identity mismatch");
        }
        peer_object(s,key,max_bytes_);
        uint64_t version=0; must(peer_.commit(tx,version,e),"peer commit",e); hook("peer_committed");
        must(journal_.commit_ack(tx,version,e),"journal commit acknowledgment",e); hook("journal_commit_ack");
        record=lookup(journal_,tx); peer_check(peer_,record,rkva_managed_registry::phase::committed,version);
        error.clear(); return true;
    } catch(const std::exception & ex) { error=ex.what(); return false; }
#else
    (void)tx;(void)ref;(void)key;(void)source;error="managed orchestrator unsupported on Windows";return false;
#endif
}
bool orchestrator::release(const std::string & tx,std::string & error) {
#ifndef _WIN32
    try {
        need(hex64(tx)&&max_bytes_,"invalid managed identity or limit"); store s(host_root_,peer_objects_,authority_); std::string e;
        auto record=lookup(journal_,tx);
        need(record.state==rkva_host_journal::phase::commit_ack || record.state==rkva_host_journal::phase::release_intent || record.state==rkva_host_journal::phase::release_ack,"journal transaction cannot release");
        need(record.host_path==s.root+"/"+tx+".managed" && record.host_sha256==record.key,"host identity mismatch");
        if(record.state==rkva_host_journal::phase::commit_ack) {
            peer_check(peer_,record,rkva_managed_registry::phase::committed,record.peer_commit_version);
            ensure_tomb(s,record,max_bytes_);
        } else {
            verify_tomb(s.host.n,"."+record.tx+".release",tomb_value(record));
            need(!exists(s.host.n,record.tx+".managed"),"released host path reappeared; explicit recovery required");
            syncdir(s.host.n);
        }
        if(record.state!=rkva_host_journal::phase::release_ack) {
            must(journal_.release_intent(tx,record.peer_commit_version,e),"journal release intent",e); hook("journal_release_intent");
            must(peer_.release(record.reference,record.peer_commit_version,e),"peer release",e); hook("peer_released");
            must(journal_.release_ack(tx,record.peer_commit_version,e),"journal release acknowledgment",e); hook("journal_release_ack");
        }
        record=lookup(journal_,tx); peer_check(peer_,record,rkva_managed_registry::phase::released,record.peer_commit_version);
        error.clear();return true;
    } catch(const std::exception & ex) { error=ex.what(); return false; }
#else
    (void)tx;error="managed orchestrator unsupported on Windows";return false;
#endif
}
}
