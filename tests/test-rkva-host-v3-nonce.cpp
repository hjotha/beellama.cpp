#include "../tools/server/server-rkva-host-v3-error.h"
#undef NDEBUG
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>
namespace fs=std::filesystem;namespace j=rkva_host_journal;namespace v3=rkva_managed_framing_v3;namespace w=rkva_registry_wire;
static v3::id id(uint8_t n){v3::id x{};x.fill(n);return x;}
static std::string crash_at;
static void crash(const char * stage){if(crash_at==stage)_exit(77);}
struct connection final:j::v3_connection_authority {
    bool valid=true;
    bool held_and_bound(const v3::session & s) override {return valid && s.authenticated;}
};
static rkva_request request(uint16_t op,uint64_t cycle,size_t size) {
    rkva_request h{};h.magic=RKVA_MAGIC;h.version=RKVA_VERSION_V2;h.op=op;h.cycle_id=cycle;h.session_id=RKVA_SESSION_NONE;h.payload_bytes=uint32_t(size);return h;
}
static j::v3_frame_builder builder(const v3::secret & key,const v3::session & session,const v3::anchor & anchor,
                                   const w::request & query,uint64_t cycle,bool corrupt=false) {
    return [=](const j::auth_id & nonce,std::vector<uint8_t> & output){
        w::request_wire q{};if(!w::encode_request(RKVA_REGISTRY_STATUS,query,q))return false;
        v3::fields f;f.type=v3::kind::command;f.owner=session.owner;f.auth_epoch=session.auth_epoch;
        f.ledger_epoch=session.ledger_epoch;f.transcript=session.transcript;f.nonce=nonce;f.host_anchor=anchor;f.payload.assign(q.begin(),q.end());
        const auto h=request(RKVA_REGISTRY_STATUS,cycle,v3::prefix_bytes+q.size()+v3::tag_bytes);const auto hb=v3::bytes(h);
        std::vector<uint8_t> body;if(!v3::seal(key,hb.data(),hb.size(),f,body))return false;
        output.assign(hb.begin(),hb.end());output.insert(output.end(),body.begin(),body.end());
        if(corrupt)output.back()^=1;
        return true;
    };
}
static std::pair<rkva_response,std::vector<uint8_t>> error_reply(const v3::secret & key,v3::session session,
                                                                 const j::v3_issued_request & issued) {
    rkva_request q{};assert(v3::parse_request_header(issued.frame.data(),32,q));
    std::vector<uint8_t> body(issued.frame.begin()+32,issued.frame.end());
    rkva_response r{};r.magic=RKVA_MAGIC;r.version=RKVA_VERSION_V2;r.op=q.op;r.cycle_id=q.cycle_id;
    r.status=RKVA_ERR_STATE;r.payload_bytes=v3::prefix_bytes+220+v3::tag_bytes;
    std::vector<uint8_t> out;assert(v3::make_reply(key,session,q,body,r,nullptr,out)==v3::status::accepted);return {r,out};
}
int main() {
    char pattern[]="/tmp/rkva-host-v3-nonce-XXXXXX";const char * name=mkdtemp(pattern);assert(name);const fs::path root(name);
    const auto dir=root/"journal";assert(fs::create_directory(dir));
    v3::secret secret{};secret.fill(9);const auto owner=id(1),tx=id(2),ref=id(3),key=id(4),ledger=id(5),digest=id(6);
    v3::session session{owner,ledger,id(7),id(8),7,0,true};
    v3::anchor anchor;anchor.owner=owner;anchor.ledger_epoch=ledger;anchor.challenge=session.peer_challenge;anchor.version=1;anchor.digest=digest;anchor.mac=id(9);
    w::request query{};query.owner=owner;query.tx=tx;
    j::journal journal(dir.string(),w::to_hex(owner));std::string error;
    assert(journal.initialize(error));assert(journal.provision_anchor(7,1,digest,error));
    assert(journal.prepare(w::to_hex(tx),w::to_hex(ref),w::to_hex(key),error));
    assert(journal.host_published(w::to_hex(tx),"/fixture",std::string(64,'c'),3,error));
    assert(journal.configure_v3_epoch(7,ledger,1,digest,error));assert(!journal.configure_v3_epoch(7,ledger,1,digest,error));
    connection conn;j::v3_issued_request issued;issued.frame={99};
    assert(!journal.issue_v3_command(secret,session,nullptr,builder(secret,session,anchor,query,10),issued,error));assert(issued.frame==std::vector<uint8_t>{99});
    assert(!journal.issue_v3_command(secret,session,&conn,builder(secret,session,anchor,query,10,true),issued,error));
    j::snapshot view;assert(journal.inspect(view,error)&&view.anchor.v3_nonces.empty());
    assert(journal.issue_v3_command(secret,session,&conn,builder(secret,session,anchor,query,10),issued,error));
    assert(issued.frame.size()>32 && issued.request_hash==rkva_snapshot::digest(issued.frame.data(),issued.frame.size()));
    assert(journal.inspect(view,error)&&view.anchor.v3_nonces.size()==1&&!view.anchor.v3_nonces[0].consumed&&
           view.anchor.v3_nonces[0].request_hash==issued.request_hash && view.anchor.v3_nonces[0].nonce==issued.nonce);
    assert(!journal.issue_v3_command(secret,session,&conn,builder(secret,session,anchor,query,11),issued,error));
    assert(!journal.issue_commit_nonce(w::to_hex(tx),id(88),w::to_hex(owner),7,error));
    auto response=error_reply(secret,session,issued);auto original_session=session;
    // A different but validly HMAC-signed request carrying the same nonce is
    // still rejected by the persisted full-frame hash, not only by HMAC.
    auto forged=issued;assert(builder(secret,session,anchor,query,11)(issued.nonce,forged.frame));
    auto forged_reply=error_reply(secret,session,forged);
    assert(j::accept_v3_error(journal,secret,session,&conn,forged.frame,forged_reply.first,
                             forged_reply.second.data(),forged_reply.second.size(),error)!=v3::status::peer_error);
    assert(journal.inspect(view,error)&&!view.anchor.v3_nonces[0].consumed);session=original_session;
    // A timeout keeps its nonce pending; a fresh connection challenge may
    // issue a distinct nonce without erasing the earlier unknown outcome.
    {
        const auto timeout_dir=root/"timeout";fs::copy(dir,timeout_dir,fs::copy_options::recursive);
        j::journal timed(timeout_dir.string(),w::to_hex(owner));auto fresh=session;fresh.peer_challenge=id(40);
        auto fresh_anchor=anchor;fresh_anchor.challenge=fresh.peer_challenge;j::v3_issued_request second;connection c;
        assert(timed.issue_v3_command(secret,fresh,&c,builder(secret,fresh,fresh_anchor,query,21),second,error));
        assert(second.nonce!=issued.nonce && timed.inspect(view,error)&&view.anchor.v3_nonces.size()==2 &&
               !view.anchor.v3_nonces[0].consumed && !view.anchor.v3_nonces[1].consumed);
    }
    // Before/after journal rename on authenticated-error acceptance, reopen
    // sees either the complete old state or the complete consumed record.
    for(const auto * stage:{"before_rename","after_rename"}) {
        const auto copy=root/(std::string("ack-crash-")+stage);fs::copy(dir,copy,fs::copy_options::recursive);
        pid_t child=fork();assert(child>=0);
        if(child==0){j::journal cj(copy.string(),w::to_hex(owner));connection c;auto local=session;
            crash_at=stage;j::test_persist_hook=crash;
            (void)j::accept_v3_error(cj,secret,local,&c,issued.frame,response.first,response.second.data(),response.second.size(),error);_exit(1);}
        int status=0;assert(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==77);
        j::journal reopened(copy.string(),w::to_hex(owner));assert(reopened.inspect(view,error));
        const bool consumed=view.anchor.v3_nonces[0].consumed;
        assert(view.anchor.version==1&&view.anchor.digest==digest&&view.transactions[0].state==j::phase::host_published);
        auto retry_session=session;connection c;auto retried=j::accept_v3_error(reopened,secret,retry_session,&c,issued.frame,
            response.first,response.second.data(),response.second.size(),error);
        assert(consumed ? retried!=v3::status::peer_error : retried==v3::status::peer_error);
        assert(reopened.inspect(view,error)&&view.anchor.v3_nonces[0].consumed);
    }
    auto fake_request=issued.frame;fake_request.back()^=1;
    assert(j::accept_v3_error(journal,secret,session,&conn,fake_request,response.first,response.second.data(),response.second.size(),error)!=v3::status::peer_error);
    assert(!session.authenticated);session=original_session;
    auto altered=response.second;altered.back()^=1;
    assert(j::accept_v3_error(journal,secret,session,&conn,issued.frame,response.first,altered.data(),altered.size(),error)!=v3::status::peer_error);
    assert(journal.inspect(view,error)&&!view.anchor.v3_nonces[0].consumed);
    session=original_session;
    assert(j::accept_v3_error(journal,secret,session,&conn,issued.frame,response.first,response.second.data(),response.second.size(),error)==v3::status::peer_error);
    assert(!session.authenticated&&journal.inspect(view,error)&&view.anchor.v3_nonces[0].consumed&&
           view.anchor.version==1&&view.anchor.digest==digest&&view.transactions[0].state==j::phase::host_published);
    session=original_session;assert(j::accept_v3_error(journal,secret,session,&conn,issued.frame,response.first,response.second.data(),response.second.size(),error)!=v3::status::peer_error);
    // Independent process writers using the same challenge must not both issue.
    const auto race_dir=root/"race";fs::copy(dir,race_dir,fs::copy_options::recursive);
    v3::session race_session=original_session;race_session.peer_challenge=id(42);v3::anchor race_anchor=anchor;race_anchor.challenge=race_session.peer_challenge;
    int channel[2];assert(pipe(channel)==0);pid_t kids[2];
    for(int i=0;i<2;i++){kids[i]=fork();assert(kids[i]>=0);if(kids[i]==0){close(channel[0]);j::journal child(race_dir.string(),w::to_hex(owner));j::v3_issued_request item;std::string err;connection c;
        char ok=child.issue_v3_command(secret,race_session,&c,builder(secret,race_session,race_anchor,query,20),item,err)?1:0;(void)!write(channel[1],&ok,1);_exit(0);}}
    close(channel[1]);int success=0;for(int i=0;i<2;i++){char c=0;assert(read(channel[0],&c,1)==1);success+=c;}close(channel[0]);
    for(auto kid:kids){int status=0;assert(waitpid(kid,&status,0)==kid&&WIFEXITED(status)&&WEXITSTATUS(status)==0);}assert(success==1);
    // Process death either side of rename leaves old or complete issued record.
    for(const auto * stage:{"before_rename","after_rename"}) {
        const auto copy=root/(std::string("crash-")+stage);fs::copy(dir,copy,fs::copy_options::recursive);
        pid_t child=fork();assert(child>=0);if(child==0){j::journal cj(copy.string(),w::to_hex(owner));connection c;j::v3_issued_request x;
            crash_at=stage;j::test_persist_hook=crash;(void)cj.issue_v3_command(secret,race_session,&c,builder(secret,race_session,race_anchor,query,30),x,error);_exit(1);}
        int status=0;assert(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==77);
        j::journal reopened(copy.string(),w::to_hex(owner));assert(reopened.inspect(view,error));
        assert(view.anchor.v3_nonces.size()==1 || view.anchor.v3_nonces.size()==2);
        assert(view.anchor.version==1&&view.anchor.digest==digest);
    }
    // Missing/corrupt journal cannot be re-created or used.
    const auto corrupt=root/"corrupt";fs::copy(dir,corrupt,fs::copy_options::recursive);
    {std::fstream f(corrupt/"host-journal.bin",std::ios::binary|std::ios::in|std::ios::out);f.seekp(5);f.put('X');}
    j::journal broken(corrupt.string(),w::to_hex(owner));assert(!broken.inspect(view,error));
    assert(!broken.issue_v3_command(secret,race_session,&conn,builder(secret,race_session,race_anchor,query,50),issued,error));
    const auto missing=root/"missing";fs::copy(dir,missing,fs::copy_options::recursive);
    assert(fs::remove(missing/"host-journal.bin"));
    j::journal absent(missing.string(),w::to_hex(owner));assert(!absent.inspect(view,error));
    assert(!absent.issue_v3_command(secret,race_session,&conn,builder(secret,race_session,race_anchor,query,51),issued,error));
    assert(!fs::exists(missing/"host-journal.bin"));
    fs::remove_all(root);std::cout<<"durable V3 request hash/nonce, signed error, replay and crash: PASS\n";
}
