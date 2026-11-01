#include "../tools/server/server-rkva-reconcile-gate.h"
#undef NDEBUG
#include <cassert>
#include <filesystem>
#include <iostream>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif
using namespace rkva_reconcile_gate;
namespace fs = std::filesystem;
#ifndef _WIN32
static std::string id(char c) { return std::string(64,c); }
#endif
int main() {
#ifdef _WIN32
    std::cout << "POSIX fixture skipped\n"; return 0;
#else
    char name[]="/tmp/rkva-reconcile-XXXXXX"; assert(mkdtemp(name)); fs::path root(name);
    fs::create_directory(root/"host"); fs::create_directory(root/"peer");
    const auto owner=id('a'), t1=id('1'), t2=id('2'), r1=id('3'), r2=id('4'), key=id('5');
    rkva_host_journal::journal j((root/"host").string(),owner);
    rkva_managed_registry::registry p((root/"peer").string(),owner); std::string error;
    assert(j.initialize(error)); assert(p.initialize(error));
    assert(j.prepare(t1,r1,key,error)); assert(p.prepare(t1,r1,key,error));
    assert(j.host_published(t1,(root/"host"/"one").string(),key,123,error));
    uint64_t v1=0; assert(p.commit(t1,v1,error)); assert(j.commit_ack(t1,v1,error));
    fs::copy_file(root/"peer"/"registry", root/"old-registry");
    assert(j.prepare(t2,r2,key,error)); assert(p.prepare(t2,r2,key,error));
    assert(j.host_published(t2,(root/"host"/"two").string(),key,456,error));
    // Peer commit succeeds but process exits before host ACK: ambiguous.
    pid_t child=fork(); assert(child>=0);
    if (!child) { uint64_t v=0; std::string e; _exit(p.commit(t2,v,e) ? 77 : 1); }
    int st=0; assert(waitpid(child,&st,0)==child && WIFEXITED(st) && WEXITSTATUS(st)==77);
    rkva_host_journal::snapshot host; rkva_managed_registry::snapshot peer;
    assert(j.inspect(host,error)); assert(p.inspect(peer,error));
    rkva_managed_auth::secret secret{}; secret.fill(9); challenge nonce{}; nonce.fill(7);
    signed_view proof{owner,nonce,peer,{}}; assert(seal(secret,proof));
    assert(evaluate(&host,&proof,owner,nonce,secret)==verdict::ambiguous);
    const auto v2=peer.transactions.back().committed_version; assert(v2>v1);
    assert(j.commit_ack(t2,v2,error)); assert(j.inspect(host,error));
    assert(evaluate(&host,&proof,owner,nonce,secret)==verdict::matched);
    assert(evaluate(nullptr,&proof,owner,nonce,secret)==verdict::unavailable);
    assert(evaluate(&host,nullptr,owner,nonce,secret)==verdict::unavailable);
    auto forged=proof; forged.peer.version++; assert(evaluate(&host,&forged,owner,nonce,secret)==verdict::unauthenticated);
    auto wrong_nonce=nonce; wrong_nonce[0]^=1; assert(evaluate(&host,&proof,owner,wrong_nonce,secret)==verdict::unauthenticated);
    auto wrong_key=secret; wrong_key[0]^=1; assert(evaluate(&host,&proof,owner,nonce,wrong_key)==verdict::unauthenticated);
    fs::copy_file(root/"peer"/"registry",root/"new-registry");
    // A real old registry file can be structurally valid and freshly MACed;
    // durable host ACK v2 still proves that this peer view rolled back.
    fs::copy_file(root/"old-registry",root/"peer"/"registry",fs::copy_options::overwrite_existing);
    assert(p.inspect(peer,error)); assert(peer.version < v2);
    nonce[0]++; signed_view old{owner,nonce,peer,{}}; assert(seal(secret,old));
    assert(evaluate(&host,&old,owner,nonce,secret)==verdict::rollback);
    assert(p.inspect(peer,error)); assert(peer.version == old.peer.version); // no mutation or automatic recovery
    fs::copy_file(root/"new-registry",root/"peer"/"registry",fs::copy_options::overwrite_existing);
    assert(p.inspect(peer,error)); signed_view restored{owner,nonce,peer,{}}; assert(seal(secret,restored));
    assert(evaluate(&host,&restored,owner,nonce,secret)==verdict::matched);
    // A peer release without host ACK cannot authorize anything either.
    assert(p.release(r1,v1,error)); assert(p.inspect(peer,error));
    nonce[0]++; signed_view released{owner,nonce,peer,{}}; assert(seal(secret,released));
    assert(evaluate(&host,&released,owner,nonce,secret)==verdict::ambiguous);
    fs::copy_file(root/"new-registry",root/"peer"/"registry",fs::copy_options::overwrite_existing);
    assert(p.inspect(peer,error)); restored={owner,nonce,peer,{}}; assert(seal(secret,restored));
    // A host release intent without ACK is ambiguous.
    assert(j.release_intent(t1,v1,error)); assert(j.inspect(host,error));
    assert(evaluate(&host,&restored,owner,nonce,secret)==verdict::ambiguous);
    fs::rename(root/"host"/"host-journal.bin",root/"host"/"hidden");
    assert(!j.inspect(host,error)); assert(evaluate(nullptr,&restored,owner,nonce,secret)==verdict::unavailable);
    fs::rename(root/"peer"/"registry",root/"peer"/"hidden");
    assert(!p.inspect(peer,error)); assert(evaluate(&host,nullptr,owner,nonce,secret)==verdict::unavailable);
    fs::remove_all(root); std::cout << "offline reconcile gate tests passed\n";
#endif
}
