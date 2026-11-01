#include "../tools/server/server-rkva-host-journal.h"
#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif
namespace hj = rkva_host_journal;
namespace fs = std::filesystem;
static const std::string owner(64,'a'), key(64,'b'), digest(64,'c');
static std::string id(unsigned n) { char buf[65]; std::snprintf(buf,sizeof(buf),"%064x",n); return buf; }
static std::string fault_stage;
static void fail_stage(const char * stage) { if(fault_stage==stage) throw std::runtime_error("injected failure"); }
#ifndef _WIN32
static void crash_stage(const char * stage) { if(fault_stage==stage) _exit(77); }
#endif
static void transition(hj::journal & j, unsigned n, std::string & e) {
    const auto tx=id(1), ref=id(2);
    bool ok=false;
    switch(n) {
        case 0: ok=j.prepare(tx,ref,key,e); break;
        case 1: ok=j.host_published(tx,"/trusted/store/object.slot",digest,123,e); break;
        case 2: ok=j.commit_ack(tx,13,e); break;
        case 3: ok=j.release_intent(tx,13,e); break;
        case 4: ok=j.release_ack(tx,13,e); break;
        case 5: ok=j.abort_unpublished(tx,e); break;
    }
    assert(ok);
}
static bool do_transition(hj::journal & j, unsigned n, std::string & e) {
    switch(n) {
        case 0: return j.prepare(id(1),id(2),key,e);
        case 1: return j.host_published(id(1),"/trusted/store/object.slot",digest,123,e);
        case 2: return j.commit_ack(id(1),13,e);
        case 3: return j.release_intent(id(1),13,e);
        case 4: return j.release_ack(id(1),13,e);
        case 5: return j.abort_unpublished(id(1),e);
    }
    return false;
}
#ifndef _WIN32
static void wait_success(pid_t pid) { int status=0; assert(waitpid(pid,&status,0)==pid); assert(WIFEXITED(status)&&WEXITSTATUS(status)==0); }
#endif
int main() {
#ifdef _WIN32
    hj::journal j("x",owner); std::string e; assert(!j.initialize(e)); assert(e.find("unsupported")!=std::string::npos);
    std::puts("Windows fail-closed: PASS"); return 0;
#else
    const auto root=fs::temp_directory_path()/("rkva-host-journal-test-"+std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    assert(fs::create_directory(root));
    std::string e;
    const auto baseline=root/"baseline"; assert(fs::create_directory(baseline));
    hj::journal j(baseline.string(),owner);
    hj::snapshot snap;
    assert(!j.inspect(snap,e)); assert(j.initialize(e)); assert(!j.initialize(e));
    for(unsigned i=0;i<5;i++) { transition(j,i,e); assert(do_transition(j,i,e)); }
    assert(j.inspect(snap,e) && snap.transactions.size()==1 && snap.revision==5 && snap.transactions[0].state==hj::phase::release_ack);
    assert(!j.prepare(id(1),id(2),key,e)); assert(!j.prepare(id(3),id(2),key,e));
    assert(!j.commit_ack(id(1),13,e)); assert(!j.release_ack(id(1),14,e));
    hj::journal reopened(baseline.string(),owner); assert(reopened.inspect(snap,e) && snap.revision==5);
    assert(!hj::journal(baseline.string(),id(99)).inspect(snap,e));
    const auto aborted=root/"aborted"; assert(fs::create_directory(aborted)); hj::journal a(aborted.string(),owner); assert(a.initialize(e));
    transition(a,0,e); transition(a,5,e); transition(a,5,e); assert(!a.prepare(id(1),id(2),key,e)); assert(!a.host_published(id(1),"/x",digest,1,e));
    const auto invalid=root/"invalid"; assert(fs::create_directory(invalid)); hj::journal q(invalid.string(),owner); assert(q.initialize(e));
    assert(q.prepare(id(1),id(2),key,e)); assert(!q.commit_ack(id(1),13,e));
    assert(!q.host_published(id(1),"relative",digest,123,e)); assert(q.host_published(id(1),"/x",digest,123,e));
    assert(!q.abort_unpublished(id(1),e)); assert(!q.host_published(id(1),"/changed",digest,123,e));
    const std::vector<std::string> stages={"before_write","after_partial_write","after_write","before_file_sync","after_file_sync","before_rename","after_rename","before_directory_sync","after_directory_sync"};
    for(unsigned op=0;op<6;op++) for(size_t st=0;st<stages.size();st++) {
        const auto dir=root/("fault-"+std::to_string(op)+"-"+std::to_string(st)); assert(fs::create_directory(dir)); hj::journal f(dir.string(),owner); assert(f.initialize(e));
        if(op==5) transition(f,0,e); else for(unsigned n=0;n<op;n++) transition(f,n,e);
        fault_stage=stages[st]; hj::test_persist_hook=fail_stage;
        assert(!do_transition(f,op,e)); hj::test_persist_hook=nullptr;
        hj::journal recovered(dir.string(),owner); assert(recovered.inspect(snap,e));
        assert(do_transition(recovered,op,e)); assert(do_transition(recovered,op,e));
        assert(recovered.inspect(snap,e));
        assert(snap.transactions.size()==1);
        const auto want=op==5 ? hj::phase::aborted : hj::phase(op+1);
        assert(snap.transactions[0].state==want);
    }
    // Initialization can fail with an ambiguous published inode; recovery
    // must inspect it and must never reinitialize or silently discard it.
    for(const auto & stage : stages) {
        const auto dir=root/("init-"+stage); assert(fs::create_directory(dir)); hj::journal init(dir.string(),owner);
        fault_stage=stage; hj::test_persist_hook=fail_stage; assert(!init.initialize(e)); hj::test_persist_hook=nullptr;
        if(stage=="after_rename" || stage=="before_directory_sync" || stage=="after_directory_sync") {
            assert(init.inspect(snap,e)); assert(!init.initialize(e));
        } else {
            assert(!init.inspect(snap,e)); assert(init.initialize(e));
        }
    }
    {
        const auto dir=root/"init-after-link"; assert(fs::create_directory(dir)); hj::journal init(dir.string(),owner);
        fault_stage="after_link"; hj::test_persist_hook=fail_stage; assert(!init.initialize(e)); hj::test_persist_hook=nullptr;
        assert(!init.inspect(snap,e)); assert(!init.initialize(e)); // two hardlinks: explicit recovery
    }
    // A process death before and after rename must leave an old or a new valid
    // journal. A replay of the same transaction settles either outcome.
    for(const auto & stage : stages) {
        const auto dir=root/("crash-"+stage); assert(fs::create_directory(dir)); hj::journal f(dir.string(),owner); assert(f.initialize(e));
        assert(f.prepare(id(10), id(11), key, e));
        pid_t pid=fork(); assert(pid>=0);
        if(pid==0) { fault_stage=stage; hj::test_persist_hook=crash_stage; (void) f.prepare(id(1),id(2),key,e); _exit(1); }
        int status=0; assert(waitpid(pid,&status,0)==pid); assert(WIFEXITED(status)&&WEXITSTATUS(status)==77);
        hj::journal reopened_crash(dir.string(),owner); assert(reopened_crash.inspect(snap,e));
        assert(reopened_crash.prepare(id(1),id(2),key,e)); assert(reopened_crash.inspect(snap,e));
        assert(snap.transactions.size()==2);
        assert(snap.transactions[0].tx == id(10) && snap.transactions[0].state==hj::phase::prepared);
        assert(snap.transactions[1].tx == id(1) && snap.transactions[1].state==hj::phase::prepared);
    }
    // Concurrent distinct writers preserve both updates; conflicting refs are rejected.
    const auto concurrent=root/"concurrent"; assert(fs::create_directory(concurrent)); hj::journal c(concurrent.string(),owner); assert(c.initialize(e));
    std::vector<pid_t> children;
    for(unsigned i=0;i<8;i++) { pid_t pid=fork(); assert(pid>=0); if(pid==0) { std::string err; _exit(c.prepare(id(100+i),id(200+i),key,err)?0:1); } children.push_back(pid); }
    for(auto pid:children) wait_success(pid);
    assert(c.inspect(snap,e) && snap.transactions.size()==8 && snap.revision==8);
    int pipefd[2]; assert(pipe(pipefd)==0); children.clear();
    for(unsigned i=0;i<2;i++) { pid_t pid=fork(); assert(pid>=0); if(pid==0) { close(pipefd[0]); std::string err; char answer=c.prepare(id(300+i),id(400),key,err)?1:0; (void)!write(pipefd[1],&answer,1); _exit(0); } children.push_back(pid); }
    close(pipefd[1]); int success=0; for(int i=0;i<2;i++) { char b=0; assert(read(pipefd[0],&b,1)==1); success+=b; } close(pipefd[0]); for(auto pid:children) wait_success(pid); assert(success==1);
    assert(c.inspect(snap,e) && snap.transactions.size()==9);
    // Untrusted changes, malformed contents, and symlink file must fail closed.
    const auto corrupt=root/"corrupt"; assert(fs::create_directory(corrupt)); hj::journal d(corrupt.string(),owner); assert(d.initialize(e));
    { std::fstream file(corrupt/"host-journal.bin",std::ios::binary|std::ios::in|std::ios::out); file.seekp(1); file.put('X'); }
    assert(!d.inspect(snap,e)); assert(!d.prepare(id(1),id(2),key,e)); assert(!d.initialize(e));
    const auto symlink=root/"symlink"; assert(fs::create_directory(symlink)); fs::create_symlink(corrupt/"host-journal.bin",symlink/"host-journal.bin"); hj::journal x(symlink.string(),owner); assert(!x.inspect(snap,e)); assert(!x.initialize(e));
    // ------------------------------------------------------------------
    // Phase 3 anchored journal: provisioned anchor, durable command nonces and
    // one atomic verified-COMMIT publication. The journal checksum is still
    // corruption detection, not authentication or rollback protection.
    // ------------------------------------------------------------------
    hj::auth_id digest0{}, digest1{}, nonce1{}, other{};
    digest0.fill(0xa5); digest1.fill(0x5a); nonce1.fill(0x11); other.fill(0x22);
    {
        const auto dir=root/"anchored"; assert(fs::create_directory(dir));
        hj::journal a(dir.string(),owner); assert(a.initialize(e));
        assert(!a.provision_anchor(0,0,digest0,e));            // epoch 0 refused
        assert(!a.provision_anchor(7,0,hj::auth_id{},e));      // zero digest refused
        assert(a.provision_anchor(7,0,digest0,e));
        assert(!a.provision_anchor(7,0,digest0,e));            // provisioned once
        assert(!a.issue_commit_nonce(id(1),nonce1,owner,7,e)); // unknown tx
        assert(a.prepare(id(1),id(2),key,e));
        assert(a.host_published(id(1),"/store/one",digest,10,e));
        // An authenticated journal refuses the legacy unauthenticated ACK.
        assert(!a.commit_ack(id(1),1,e));
        assert(!a.issue_commit_nonce(id(1),nonce1,id(98),7,e));       // wrong owner
        assert(!a.issue_commit_nonce(id(1),nonce1,owner,8,e));        // wrong epoch
        assert(!a.issue_commit_nonce(id(1),hj::auth_id{},owner,7,e)); // zero nonce
        assert(a.issue_commit_nonce(id(1),nonce1,owner,7,e));
        assert(!a.issue_commit_nonce(id(1),nonce1,owner,7,e));        // nonce reuse
        // A peer cannot move the anchor by itself: the version must be the
        // immediate successor and the nonce must have been issued here first.
        assert(!a.test_commit_verified(id(1),other,1,1,digest1,id(2),key,owner,7,e));
        assert(!a.test_commit_verified(id(1),nonce1,1,2,digest1,id(2),key,owner,7,e));
        assert(!a.test_commit_verified(id(1),nonce1,0,1,digest1,id(2),key,owner,7,e));
        assert(!a.test_commit_verified(id(1),nonce1,1,1,digest1,id(3),key,owner,7,e));
        assert(!a.test_commit_verified(id(1),nonce1,1,1,hj::auth_id{},id(2),key,owner,7,e));
        assert(!a.test_commit_verified(id(1),nonce1,1,1,digest1,id(2),key,owner,8,e));
        assert(!a.test_commit_verified(id(1),nonce1,1,1,digest1,id(2),id(55),owner,7,e));
        assert(a.inspect(snap,e) && snap.anchor.configured && snap.anchor.version==0 &&
               snap.anchor.digest==digest0 && snap.anchor.nonces.size()==1 &&
               !snap.anchor.nonces[0].consumed && snap.transactions[0].state==hj::phase::host_published);
        assert(a.test_commit_verified(id(1),nonce1,1,1,digest1,id(2),key,owner,7,e));
        assert(a.inspect(snap,e) && snap.anchor.version==1 && snap.anchor.digest==digest1 &&
               snap.anchor.nonces[0].consumed && snap.transactions[0].state==hj::phase::commit_ack &&
               snap.transactions[0].peer_commit_version==1 && snap.revision==5);
        // Replay after a lost ACK is not an automatic success; the caller must
        // reconcile against the durable anchor before reissuing anything.
        assert(!a.test_commit_verified(id(1),nonce1,1,1,digest1,id(2),key,owner,7,e));
        // Authenticated release/abort needs independent durable evidence that
        // this tree does not have, so every such transition fails closed.
        assert(!a.release_intent(id(1),1,e)); assert(!a.release_ack(id(1),1,e));
        assert(!a.abort_unpublished(id(1),e));
        assert(a.prepare(id(7),id(8),key,e)); // transaction creation is not gated by the anchor
        hj::journal reopened(dir.string(),owner); assert(reopened.inspect(snap,e));
        assert(snap.anchor.configured && snap.anchor.key_epoch==7 && snap.anchor.version==1 &&
               snap.anchor.nonces.size()==1 && snap.anchor.nonces[0].consumed);
        // Anchor provisioning is refused once the journal holds history, so a
        // compromised peer cannot bootstrap a trusted anchor after the fact.
        const auto late=root/"anchored-late"; assert(fs::create_directory(late));
        hj::journal l(late.string(),owner); assert(l.initialize(e));
        assert(l.prepare(id(20),id(21),key,e));
        assert(!l.provision_anchor(7,0,digest0,e));
        // Corrupting the anchor block is detected exactly like any other byte.
        const auto rec=root/"anchored-corrupt"; assert(fs::create_directory(rec));
        hj::journal c(rec.string(),owner); assert(c.initialize(e)); assert(c.provision_anchor(7,0,digest0,e));
        { std::fstream file(rec/"host-journal.bin",std::ios::binary|std::ios::in|std::ios::out);
          file.seekp(std::streamoff(8+64+8+4+8)); file.put('Z'); }
        assert(!c.inspect(snap,e)) ; assert(!c.provision_anchor(7,0,digest0,e));
    }
    // Process death around the single atomic commit publication leaves either
    // the old or the new journal, never a partial nonce/anchor/ACK triple.
    for(const auto & stage : stages) {
        if(stage=="before_write"||stage=="after_partial_write"||stage=="after_write"||
           stage=="before_file_sync"||stage=="after_file_sync"||stage=="after_link") continue;
        const auto dir=root/("anchored-crash-"+stage); assert(fs::create_directory(dir));
        hj::journal a(dir.string(),owner); assert(a.initialize(e)); assert(a.provision_anchor(7,0,digest0,e));
        assert(a.prepare(id(1),id(2),key,e)); assert(a.host_published(id(1),"/store/one",digest,10,e));
        assert(a.issue_commit_nonce(id(1),nonce1,owner,7,e));
        pid_t pid=fork(); assert(pid>=0);
        if(pid==0) { fault_stage=stage; hj::test_persist_hook=crash_stage;
                     (void)a.test_commit_verified(id(1),nonce1,1,1,digest1,id(2),key,owner,7,e); _exit(1); }
        int status=0; assert(waitpid(pid,&status,0)==pid); assert(WIFEXITED(status)&&WEXITSTATUS(status)==77);
        hj::journal reopened(dir.string(),owner); assert(reopened.inspect(snap,e));
        const bool committed=snap.transactions[0].state==hj::phase::commit_ack;
        assert((committed && snap.anchor.nonces[0].consumed && snap.anchor.version==1 && snap.anchor.digest==digest1) ||
               (!committed && !snap.anchor.nonces[0].consumed && snap.anchor.version==0 &&
                snap.anchor.digest==digest0 && snap.transactions[0].state==hj::phase::host_published));
        const bool retry=reopened.test_commit_verified(id(1),nonce1,1,1,digest1,id(2),key,owner,7,e);
        assert(retry==!committed);
        assert(reopened.inspect(snap,e) && snap.transactions[0].state==hj::phase::commit_ack &&
               snap.anchor.nonces[0].consumed && snap.anchor.version==1 && snap.anchor.digest==digest1);
    }
    fs::remove_all(root);
    std::puts("host journal replay, faults, process death, corruption, concurrency, anchor/nonce commit: PASS");
#endif
}
