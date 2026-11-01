#include "../tools/xbox/uwp/rkva/rkva-prepare-v3.h"
#undef NDEBUG
#include <cassert>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <set>
#include <tuple>
#ifdef RKVA_SQLITE_COMPOSITION_TEST
#include "../ggml/include/rkva-anchored-sqlite.h"
#include "../ggml/src/ggml-remote-attn/rkva-managed-framing-v3.h"
#include <filesystem>
#include <unistd.h>
#include <sys/wait.h>
#include <sqlite3.h>
#endif
namespace a=rkva_resident_attestation;namespace auth=rkva_managed_auth_e2e;
namespace wire=rkva_registry_wire;namespace v3=rkva_prepare_v3;
struct handshake_guard : auth::nonce_guard {
    std::set<std::tuple<wire::id,uint64_t,auth::purpose,auth::nonce>> seen;
    auth::guard_result claim(const wire::id & o,uint64_t e,auth::purpose p,const auth::nonce & n) override {
        return seen.emplace(o,e,p,n).second?auth::guard_result::recorded:auth::guard_result::replay;
    }
    auth::guard_result consume(const wire::id & o,uint64_t e,auth::purpose from,auth::purpose to,const auth::nonce & n) override {
        if(!seen.count({o,e,from,n}))return auth::guard_result::unavailable;
        return seen.emplace(o,e,to,n).second?auth::guard_result::recorded:auth::guard_result::replay;
    }
};
// In-memory fixture is not durable and never qualifies a real SQLite pin.
struct prepare_fixture : v3::prepare_guard {
    std::mutex mutex;size_t calls=0;bool unavailable=false,throws=false;std::set<wire::id> used;
    v3::persist_result persist(const v3::metadata & m) override {
        std::lock_guard<std::mutex> lock(mutex);++calls;if(throws)throw 1;if(unavailable)return v3::persist_result::unavailable;
        return used.insert(m.nonce).second?v3::persist_result::committed:v3::persist_result::replay;
    }
};
#ifdef RKVA_SQLITE_COMPOSITION_TEST
static void make_pending_identity(rkva_anchored_sqlite::pending_request & request) {
    namespace framing=rkva_managed_framing_v3;namespace native=rkva_native_state;
    rkva_native_snapshot::Manifest manifest;std::string error;
    assert(rkva_native_snapshot::decode(request.metadata,manifest,&error));
    native::SetHeader h;h.layers=manifest.geometry.n_layers;h.layout.origin=manifest.descriptor.origin;
    h.layout.capacity=manifest.descriptor.capacity;h.layout.heads=manifest.geometry.n_head_kv;
    h.layout.iterations=wire::get32(request.semantic_hello.data()+32);
    h.layout.precision_tail_tokens=manifest.descriptor.precision_tail_tokens;h.layout.rows=manifest.live_rows;
    h.layout.saved_generation=manifest.generation;h.payload_bytes=native::payload_bytes(h.layout,h.layers);
    h.layer_bytes=native::layer_bytes(h.layout);assert(native::encode_header(h,request.native_header.data(),&error));
    wire::request q{request.owner,request.tx,request.reference,request.key,0};
    assert(framing::semantic_identity(q,request.native_header,request.semantic_hello,request.metadata,request.identity));
}
// Negative integration fixture only: a valid wire frame is forwarded to the
// actual SQLite backend, whose transition gate must reject it. No callback in
// this adapter claims resident GPU/source or host RKVK proof.
struct blocked_sqlite_guard : v3::prepare_guard {
    rkva_anchored_sqlite::backend & backend;
    std::shared_ptr<rkva_managed_store::lease> authority;
    rkva_anchored_sqlite::anchor anchor;
    rkva_anchored_sqlite::result actual = rkva_anchored_sqlite::result::ok;
    std::string error;
    blocked_sqlite_guard(rkva_anchored_sqlite::backend & b,
                         std::shared_ptr<rkva_managed_store::lease> l,
                         rkva_anchored_sqlite::anchor a):backend(b),authority(std::move(l)),anchor(a) {}
    v3::persist_result persist(const v3::metadata & m) override {
        rkva_anchored_sqlite::command c{rkva_anchored_sqlite::operation::prepare};
        c.tx=m.tx;c.reference=m.reference;c.key=m.key;c.nonce=m.nonce;
        rkva_anchored_sqlite::state out;
        actual=backend.apply(authority,&anchor,c,out,error);
        return actual==rkva_anchored_sqlite::result::replay ? v3::persist_result::replay : v3::persist_result::unavailable;
    }
};
static const char * crash_stage=nullptr;
static void crash_hook(const char * stage) { if(crash_stage && std::strcmp(crash_stage,stage)==0) _exit(73); }
struct sqlite_hello_fixture final:rkva_managed_framing_v3::hello_guard {
    rkva_anchored_sqlite::backend & db;std::shared_ptr<rkva_managed_store::lease> lease;
    wire::id owner,ledger;uint64_t epoch;rkva_anchored_sqlite::state state{};std::string error;
    sqlite_hello_fixture(rkva_anchored_sqlite::backend & b,std::shared_ptr<rkva_managed_store::lease> l,
        wire::id o,wire::id e,uint64_t n):db(b),lease(std::move(l)),owner(o),ledger(e),epoch(n) {}
    auth::guard_result admit(const wire::id & o,uint64_t e,const wire::id & le,const rkva_managed_framing_v3::anchor & a,
                             const wire::id & n) override {
        if(o!=owner||e!=epoch||le!=ledger)return auth::guard_result::unavailable;
        rkva_anchored_sqlite::anchor anchor{a.owner,a.ledger_epoch,a.challenge,{a.version,a.digest},a.mac};
        rkva_anchored_sqlite::command cmd{rkva_anchored_sqlite::operation::hello_claim};cmd.nonce=n;
        auto r=db.apply(lease,&anchor,cmd,state,error);
        return r==rkva_anchored_sqlite::result::ok?auth::guard_result::recorded:
               r==rkva_anchored_sqlite::result::replay?auth::guard_result::replay:auth::guard_result::unavailable;
    }
};
struct hello_ack_fixture final:rkva_managed_framing_v3::hello_ack_guard {
    handshake_guard & issued;explicit hello_ack_fixture(handshake_guard & g):issued(g) {}
    auth::guard_result commit(const wire::id & o,uint64_t e,const wire::id &,const rkva_managed_framing_v3::anchor &,
        const wire::id & n,const rkva_managed_framing_v3::hello_state &,const wire::id &,const wire::id &,
        const rkva_request &,const std::vector<uint8_t> &,const rkva_response &,const uint8_t *,size_t) override {
        return issued.consume(o,e,auth::purpose::client_hello,auth::purpose::client_hello_reply,n);
    }
};
#endif
// Test-only re-signing proves semantic checks still reject malformed data even
// when the frame's outer MAC is otherwise valid.
static void resign(const auth::secret & key,const rkva_request & h,std::vector<uint8_t> & frame) {
    static constexpr char domain[]="RKVA-PREPARE-V3-FRAME-V1";std::vector<uint8_t> data(domain,domain+sizeof(domain)-1);
    std::array<uint8_t,32> outer{};auth::req_header_bytes(h,outer.data());data.insert(data.end(),outer.begin(),outer.end());
    data.insert(data.end(),frame.begin(),frame.end()-v3::tag_bytes);wire::id tag{};
    assert(rkva_managed_auth::hmac_sha256(key.data(),key.size(),data.data(),data.size(),tag));
    std::copy(tag.begin(),tag.end(),frame.end()-v3::tag_bytes);
}
int main() {
    auth::secret secret{};secret.fill(9);wire::id owner{},tx{},ref{},object{},nonce{};
    owner.fill(1);tx.fill(2);ref.fill(3);object.fill(4);nonce.fill(5);
    handshake_guard client,peer;a::authenticated_hello session;
    session.offered_flags=1u|RKVA_PROTO_MANAGED_REGISTRY_V1|RKVA_PROTO_MANAGED_AUTH_V2;
    session.negotiated.reserved[0]=RKVA_CAP_MANAGED_REGISTRY_V1|RKVA_CAP_MANAGED_AUTH_V2;
    session.negotiated.supported_ops=wire::operations|(1u<<RKVA_MANAGED_AUTH_HELLO);session.negotiated.server_version=1;
    session.request.magic=RKVA_MAGIC;session.request.version=RKVA_VERSION;session.request.op=RKVA_MANAGED_AUTH_HELLO;
    session.request.cycle_id=9;session.request.session_id=RKVA_SESSION_NONE;session.request.payload_bytes=auth::offer_bytes;
    session.response.magic=RKVA_MAGIC;session.response.version=RKVA_VERSION;session.response.op=session.request.op;
    session.response.cycle_id=9;session.response.status=RKVA_OK;session.response.payload_bytes=auth::hello_reply_bytes;
    auth::nonce client_nonce{},peer_nonce{};client_nonce.fill(6);peer_nonce.fill(7);
    assert(auth::make_offer(secret,owner,3,session.offered_flags,session.negotiated,session.request,client_nonce,&client,session.offer)==auth::status::accepted);
    assert(auth::accept_offer(secret,owner,3,session.offered_flags,session.negotiated,session.request,session.response,session.offer,peer_nonce,&peer,session.reply,session.live)==auth::status::accepted);
    rkva_hello hello{};hello.proto_flags=session.offered_flags;hello.n_layer_remote=1;hello.n_head=8;hello.n_head_kv=2;
    hello.head_dim=256;hello.max_ctx=4096;hello.cache_bits_k=16;hello.cache_bits_v=16;hello.group_tokens=128;
    hello.sinkhorn_iters=3;hello.tail_tokens=128;hello.tail_groups=2;hello.domain=0;hello.kq_scale=0.0625f;
    a::input in;in.session_id=9;in.live_rows=513;in.generation=0x700000001ull;in.submitted_fence=11;in.completed_fence=11;in.host_key=object;
    in.geometry={8,2,256,2,4096,1};in.descriptor={1024,1024,RKVA_RANGE_STORAGE_KVARN4,3,128,128,256,2,4,4,8960,3,2,128,128,1,1,1};
    in.counts={513};in.record_groups={4};in.ready={true};in.layers={{385,384,4,1}};
    a::evidence proof;std::string error;assert(a::make(in,session,secret,proof,error));
    rkva_request header{};header.magic=RKVA_MAGIC;header.version=RKVA_VERSION;header.op=RKVA_REGISTRY_PREPARE;
    header.session_id=RKVA_SESSION_NONE;header.cycle_id=42;
    header.payload_bytes=uint32_t(v3::prefix_bytes+proof.canonical_metadata.size()+v3::tag_bytes);
    std::vector<uint8_t> frame;assert(v3::make(secret,session,header,tx,ref,object,hello,in.session_id,proof,nonce,frame));
    assert(frame.size()==header.payload_bytes);
#ifdef RKVA_SQLITE_COMPOSITION_TEST
    {
        namespace f=rkva_managed_framing_v3;namespace db=rkva_anchored_sqlite;
        char dir[]="/tmp/rkva-v3-hello-real-sqlite-XXXXXX";assert(mkdtemp(dir));
        db::id ledger{};ledger.fill(0x61);db::secret anchor_key{};anchor_key.fill(0x62);
        auto token=rkva_managed_store::lease::try_acquire(dir,error);assert(token);
        db::backend store(dir,owner,ledger,anchor_key);assert(store.provision(token,error));db::state state{};assert(store.inspect(token,state,error));
        const db::state before_hello=state;
        db::anchor old;old.owner=owner;old.epoch=ledger;old.accepted=state;assert(store.challenge(token,old.challenge,error));
        rkva_response challenge_header{};challenge_header.magic=RKVA_MAGIC;challenge_header.version=RKVA_VERSION;
        challenge_header.op=RKVA_MANAGED_AUTH_CHALLENGE;challenge_header.payload_bytes=uint32_t(f::prefix_bytes+f::tag_bytes);
        std::vector<uint8_t> challenge;assert(f::make_prehello_challenge(secret,challenge_header,owner,3,ledger,old.challenge,challenge)==f::status::accepted);
        wire::id received{};assert(f::accept_prehello_challenge(secret,challenge_header,challenge.data(),challenge.size(),owner,3,ledger,received)==f::status::accepted);
        assert(received==old.challenge && db::sign_anchor(anchor_key,old));
        f::anchor host_anchor{old.owner,old.epoch,old.challenge,old.accepted.version,old.accepted.digest,old.mac};
        f::hello info;info.proposed=hello;info.proposed.proto_flags|=RKVA_PROTO_MANAGED_FRAMING_V3;
        info.caps=RKVA_CAP_MANAGED_REGISTRY_V1|RKVA_CAP_MANAGED_AUTH_V2|RKVA_CAP_MANAGED_FRAMING_V3;
        info.ops=wire::operations|(1u<<RKVA_MANAGED_AUTH_HELLO)|(1u<<RKVA_MANAGED_AUTH_CHALLENGE)|(1u<<RKVA_MANAGED_ATTEST);info.server_version=1;
        rkva_request q{};q.magic=RKVA_MAGIC;q.version=RKVA_VERSION;q.op=RKVA_MANAGED_AUTH_HELLO;q.session_id=RKVA_SESSION_NONE;
        q.payload_bytes=uint32_t(f::prefix_bytes+f::hello_bytes+f::tag_bytes);q.cycle_id=1;
        handshake_guard host_nonce;hello_ack_fixture ack_guard(host_nonce);std::vector<uint8_t> offer;wire::id hello_nonce{};hello_nonce.fill(0x63);
        assert(f::make_offer(secret,q,info,owner,3,ledger,host_anchor,hello_nonce,&host_nonce,offer)==f::status::accepted);
        sqlite_hello_fixture peer(store,token,owner,ledger,3);f::fields accepted;
        assert(f::accept_offer(secret,q,offer.data(),offer.size(),info,owner,3,ledger,received,&peer,accepted)==f::status::accepted);
        assert(peer.state.version==state.version+1 && store.inspect(token,state,error) && state.digest==peer.state.digest);
        wire::id next{};assert(store.challenge(token,next,error));
        rkva_response rh{};rh.magic=RKVA_MAGIC;rh.version=RKVA_VERSION;rh.op=q.op;rh.cycle_id=q.cycle_id;
        rh.payload_bytes=uint32_t(f::prefix_bytes+f::hello_ack_bytes+f::tag_bytes);
        f::session peer_session{},host_session{};std::vector<uint8_t> ack;f::hello_state committed{state.version,state.digest},host_state{};
        assert(f::make_hello_ack(secret,q,offer,rh,accepted,next,committed,ack,peer_session)==f::status::accepted);
        assert(f::accept_hello_ack(secret,q,offer,rh,ack.data(),ack.size(),&ack_guard,host_session,host_state)==f::status::accepted);
        assert(host_state.version==state.version && host_state.digest==state.digest && host_session.transcript==peer_session.transcript);
        // On reconnect, the same HELLO nonce cannot be admitted even under a
        // fresh challenge and correctly signed anchor.
        old.accepted=state;old.challenge=next;assert(db::sign_anchor(anchor_key,old));
        host_anchor={old.owner,old.epoch,old.challenge,old.accepted.version,old.accepted.digest,old.mac};
        handshake_guard second_host;std::vector<uint8_t> replay;q.cycle_id++;
        assert(f::make_offer(secret,q,info,owner,3,ledger,host_anchor,hello_nonce,&second_host,replay)==f::status::accepted);
        assert(f::accept_offer(secret,q,replay.data(),replay.size(),info,owner,3,ledger,next,&peer,accepted)==f::status::replay);
        db::state unchanged{};assert(store.inspect(token,unchanged,error) && unchanged.version==state.version && unchanged.digest==state.digest);
        // If the committed HELLO ACK was lost, the old independently held
        // host anchor is stale; no diagnostic state may silently replace it.
        old.accepted=before_hello;assert(store.challenge(token,old.challenge,error)&&db::sign_anchor(anchor_key,old));
        host_anchor={old.owner,old.epoch,old.challenge,old.accepted.version,old.accepted.digest,old.mac};
        handshake_guard lost_host;hello_nonce[0]++;q.cycle_id++;std::vector<uint8_t> lost_retry;
        assert(f::make_offer(secret,q,info,owner,3,ledger,host_anchor,hello_nonce,&lost_host,lost_retry)==f::status::accepted);
        assert(f::accept_offer(secret,q,lost_retry.data(),lost_retry.size(),info,owner,3,ledger,old.challenge,&peer,accepted)==f::status::unavailable);
        assert(store.inspect(token,unchanged,error) && unchanged.version==state.version && unchanged.digest==state.digest);
        token.reset();std::filesystem::remove_all(dir);
    }
    {
        namespace db=rkva_anchored_sqlite;
        char tmp[]="/tmp/rkva-prepare-sqlite-negative-XXXXXX";
        char * path=mkdtemp(tmp);assert(path);
        auto lease=rkva_managed_store::lease::try_acquire(path,error);assert(lease);
        db::id epoch{};epoch.fill(0x23);db::secret anchor_key{};anchor_key.fill(0x24);
        db::backend sqlite(path,owner,epoch,anchor_key);assert(sqlite.provision(lease,error));
        db::state before{},after{};assert(sqlite.inspect(lease,before,error));
        db::anchor anchor;anchor.owner=owner;anchor.epoch=epoch;anchor.accepted=before;
        assert(sqlite.challenge(lease,anchor.challenge,error)&&db::sign_anchor(anchor_key,anchor));
        blocked_sqlite_guard blocked(sqlite,lease,anchor);v3::metadata result;result.tx.fill(0xee);
        assert(v3::accept(secret,session,hello,header,frame.data(),frame.size(),&blocked,result)==v3::result::unavailable);
        assert(blocked.actual==db::result::denied && result.tx[0]==0xee);
        assert(sqlite.inspect(lease,after,error)&&after.version==before.version && after.digest==before.digest);
        // Reopening the backend still observes the unchanged durable state.
        db::backend reopened(path,owner,epoch,anchor_key);db::state recovered{};
        assert(reopened.inspect(lease,recovered,error) && recovered.version==before.version && recovered.digest==before.digest);
        // The rejected PREPARE must not consume its nonce. The hello claim is a
        // diagnostic-only operation; it proves the nonce can still be inserted.
        anchor.accepted=before;assert(reopened.challenge(lease,anchor.challenge,error)&&db::sign_anchor(anchor_key,anchor));
        db::command probe{db::operation::hello_claim};probe.nonce=nonce;
        assert(reopened.apply(lease,&anchor,probe,after,error)==db::result::ok && after.version==before.version+1);
        lease.reset();std::filesystem::remove_all(path);
    }
#endif
#ifdef RKVA_SQLITE_COMPOSITION_TEST
    {
        namespace db=rkva_anchored_sqlite;namespace snap=rkva_native_snapshot;
        snap::Manifest manifest;assert(snap::decode(proof.canonical_metadata,manifest,&error));
        db::pending_request request;request.owner=owner;request.tx=tx;request.reference=ref;request.key=object;request.nonce=nonce;
        std::copy_n(frame.data()+140,80,request.semantic_hello.data());request.metadata=proof.canonical_metadata;make_pending_identity(request);
        request.reservation=manifest.payload_bytes+request.metadata.size()+(1u<<20);
        char tmp[]="/tmp/rkva-prepare-sqlite-atomic-XXXXXX";char * path=mkdtemp(tmp);assert(path);
        db::id epoch{};epoch.fill(0x41);db::secret ak{};ak.fill(0x42);
        auto lease=rkva_managed_store::lease::try_acquire(path,error);assert(lease);
        db::backend sqlite(path,owner,epoch,ak,request.reservation+1);assert(sqlite.provision(lease,error));
        db::state state{},saved{};assert(sqlite.inspect(lease,state,error));
        auto run=[&](db::pending_request & req,db::state & current) {
            db::anchor a;a.owner=owner;a.epoch=epoch;a.accepted=current;
            assert(sqlite.challenge(lease,a.challenge,error)&&db::sign_anchor(ak,a));
            db::state out{};const auto res=sqlite.test_prepare(lease,&a,req,out,error);
            if(res==db::result::ok)current=out;
            return res;
        };
        assert(run(request,state)==db::result::ok);saved=state;assert(state.version==1);
        assert(run(request,state)==db::result::replay && state.digest==saved.digest);
        request.nonce[0]++;assert(run(request,state)==db::result::ok && state.version==2);saved=state;
        auto changed=request;changed.nonce[1]++;changed.metadata.back()^=1;
        assert(run(changed,state)!=db::result::ok && state.digest==saved.digest);
        changed=request;changed.nonce[1]++;changed.reservation++;
        assert(run(changed,state)!=db::result::ok && state.digest==saved.digest);
        changed=request;changed.nonce[1]++;changed.tx[0]++;changed.reference[0]++;
        // Recompute a valid stable identity for a second transaction; quota
        // must reject it rather than reuse or evict the existing reservation.
        auto tmpframe=frame;std::copy(changed.tx.begin(),changed.tx.end(),tmpframe.begin()+40);
        std::copy(changed.reference.begin(),changed.reference.end(),tmpframe.begin()+72);
        make_pending_identity(changed);
        assert(run(changed,state)!=db::result::ok && state.digest==saved.digest);
        // Failed quota admission rolls back its nonce and leaves the first pin
        // available for an exact retry with that nonce.
        request.nonce=changed.nonce;assert(run(request,state)==db::result::ok);saved=state;
        db::backend reopened(path,owner,epoch,ak,1);db::state recovered{};
        assert(reopened.inspect(lease,recovered,error) && recovered.version==state.version && recovered.digest==state.digest);
        // Corrupt a persisted reservation and verify the prior signed anchor is
        // rejected; inspect is only a diagnostic and cannot bless the new data.
        sqlite3 * raw=nullptr;std::string filename=std::string(path)+"/rkva-managed-v3/peer.db";
        assert(sqlite3_open(filename.c_str(),&raw)==SQLITE_OK);
        assert(sqlite3_exec(raw,"UPDATE pins SET reservation=reservation+1",nullptr,nullptr,nullptr)==SQLITE_OK);sqlite3_close(raw);
        request.nonce[2]++;assert(run(request,state)==db::result::stale_anchor && state.digest==saved.digest);
        lease.reset();std::filesystem::remove_all(path);
        // Kill a real writer immediately before and after SQLite COMMIT. A
        // hot journal is deliberately fail-closed, never auto-recovered.
        for(const char * stage:{"prepare_before_commit","prepare_after_commit"}) {
            char crashdir[]="/tmp/rkva-prepare-sqlite-crash-XXXXXX";assert(mkdtemp(crashdir));
            auto initial=rkva_managed_store::lease::try_acquire(crashdir,error);assert(initial);
            db::backend seed(crashdir,owner,epoch,ak,request.reservation+1);assert(seed.provision(initial,error));
            db::state zero{};assert(seed.inspect(initial,zero,error));initial.reset();
            pid_t pid=fork();assert(pid>=0);
            if(pid==0) {
                auto childlease=rkva_managed_store::lease::try_acquire(crashdir,error);if(!childlease)_exit(10);
                db::backend child(crashdir,owner,epoch,ak,request.reservation+1);db::anchor a;
                a.owner=owner;a.epoch=epoch;a.accepted=zero;
                if(!child.challenge(childlease,a.challenge,error)||!db::sign_anchor(ak,a))_exit(11);
                crash_stage=stage;db::test_hook=crash_hook;db::state out{};
                (void)child.test_prepare(childlease,&a,request,out,error);_exit(12);
            }
            int status=0;assert(waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==73);
            auto again=rkva_managed_store::lease::try_acquire(crashdir,error);assert(again);
            db::backend after_crash(crashdir,owner,epoch,ak,request.reservation+1);db::state out{};
            if(std::strcmp(stage,"prepare_after_commit")==0)assert(after_crash.inspect(again,out,error) && out.version==1);
            else assert(!after_crash.inspect(again,out,error) && error.find("ambiguous managed store file")!=std::string::npos);
            again.reset();std::filesystem::remove_all(crashdir);
        }
    }
#endif
    prepare_fixture guard;v3::metadata decoded;decoded.tx.fill(0xee);
    // Every byte is authenticated or structurally rejected; no invalid frame
    // may call the guard or publish partial metadata.
    for(size_t i=0;i<frame.size();i++) {
        auto altered=frame;altered[i]^=1;v3::metadata out;out.tx.fill(0xee);
        assert(v3::accept(secret,session,hello,header,altered.data(),altered.size(),&guard,out)==v3::result::invalid);
        assert(out.tx[0]==0xee && guard.calls==0);
    }
    for(size_t n=0;n<frame.size();n++)assert(v3::accept(secret,session,hello,header,frame.data(),n,&guard,decoded)==v3::result::invalid);
    assert(v3::accept(secret,session,hello,header,nullptr,frame.size(),&guard,decoded)==v3::result::invalid);
    auto hh=header;hh.cycle_id++;assert(v3::accept(secret,session,hello,hh,frame.data(),frame.size(),&guard,decoded)==v3::result::invalid);
    hh=header;hh.payload_bytes--;assert(v3::accept(secret,session,hello,hh,frame.data(),frame.size(),&guard,decoded)==v3::result::invalid);
    hh=header;hh.op=RKVA_REGISTRY_COMMIT;assert(v3::accept(secret,session,hello,hh,frame.data(),frame.size(),&guard,decoded)==v3::result::invalid);
    auto wronghello=hello;wronghello.n_head++;assert(v3::accept(secret,session,wronghello,header,frame.data(),frame.size(),&guard,decoded)==v3::result::invalid);
    // Even with a correctly recomputed outer HMAC, inconsistent geometry,
    // reserved HELLO bits and a forged nested coverage proof fail closed.
    auto forged=frame;forged[144] ^= 1;resign(secret,header,forged);auto forged_hello=v3::get_hello(forged.data()+136);
    assert(v3::accept(secret,session,forged_hello,header,forged.data(),forged.size(),&guard,decoded)==v3::result::invalid);
    forged=frame;forged[204] ^= 1;resign(secret,header,forged);forged_hello=v3::get_hello(forged.data()+136);
    assert(v3::accept(secret,session,forged_hello,header,forged.data(),forged.size(),&guard,decoded)==v3::result::invalid);
    forged=frame;forged[240]^=1;resign(secret,header,forged);
    assert(v3::accept(secret,session,hello,header,forged.data(),forged.size(),&guard,decoded)==v3::result::invalid);
    assert(guard.calls==0);
    auto badsession=session;badsession.live.transcript[0]^=1;assert(v3::accept(secret,badsession,hello,header,frame.data(),frame.size(),&guard,decoded)==v3::result::invalid);
    auto badkey=secret;badkey[0]^=1;assert(v3::accept(badkey,session,hello,header,frame.data(),frame.size(),&guard,decoded)==v3::result::invalid);
    assert(v3::accept(secret,session,hello,header,frame.data(),frame.size(),nullptr,decoded)==v3::result::unavailable);
    guard.unavailable=true;assert(v3::accept(secret,session,hello,header,frame.data(),frame.size(),&guard,decoded)==v3::result::unavailable);
    guard.unavailable=false;guard.throws=true;assert(v3::accept(secret,session,hello,header,frame.data(),frame.size(),&guard,decoded)==v3::result::unavailable);guard.throws=false;
    assert(v3::accept(secret,session,hello,header,frame.data(),frame.size(),&guard,decoded)==v3::result::accepted);
    assert(decoded.owner==owner && decoded.tx==tx && decoded.reference==ref && decoded.key==object && decoded.rows==513 && decoded.generation==in.generation);
    assert(decoded.coverage.size()==1 && decoded.coverage[0].stage_begin==384 && decoded.canonical_attestation==proof.canonical_metadata);
    const auto identity=decoded.identity_digest;
    assert(v3::accept(secret,session,hello,header,frame.data(),frame.size(),&guard,decoded)==v3::result::replay);
    nonce[0]++;header.cycle_id++;std::vector<uint8_t> retry;assert(v3::make(secret,session,header,tx,ref,object,hello,in.session_id,proof,nonce,retry));
    assert(v3::accept(secret,session,hello,header,retry.data(),retry.size(),&guard,decoded)==v3::result::accepted && decoded.identity_digest==identity);
    // The real guard must be cross-process durable; this test exercises only
    // the callback's atomicity with an in-memory synchronized fixture.
    nonce[1]++;header.cycle_id++;std::vector<uint8_t> raced;assert(v3::make(secret,session,header,tx,ref,object,hello,in.session_id,proof,nonce,raced));
    std::array<v3::result,8> results{};std::vector<std::thread> threads;
    for(size_t i=0;i<results.size();i++)threads.emplace_back([&,i]{v3::metadata local;results[i]=v3::accept(secret,session,hello,header,raced.data(),raced.size(),&guard,local);});
    for(auto & t:threads)t.join();
    assert(std::count(results.begin(),results.end(),v3::result::accepted)==1);
    assert(std::count(results.begin(),results.end(),v3::result::replay)==7);
    // A new authenticated connection and newly observed fence may produce a
    // different attestation MAC while retaining the same durable identity.
    auto next_session=session;next_session.request.cycle_id++;
    next_session.response.cycle_id=next_session.request.cycle_id;
    next_session.offered_flags|=2u; // unrelated F32 wire offer may change on reconnect
    auto next_hello=hello;next_hello.proto_flags=next_session.offered_flags;
    auth::nonce new_client{},new_peer{};new_client.fill(0x71);new_peer.fill(0x72);
    assert(auth::make_offer(secret,owner,3,next_session.offered_flags,next_session.negotiated,next_session.request,new_client,&client,next_session.offer)==auth::status::accepted);
    assert(auth::accept_offer(secret,owner,3,next_session.offered_flags,next_session.negotiated,next_session.request,next_session.response,next_session.offer,new_peer,&peer,next_session.reply,next_session.live)==auth::status::accepted);
    auto recaptured=in;recaptured.submitted_fence=12;recaptured.completed_fence=12;a::evidence new_proof;
    assert(a::make(recaptured,next_session,secret,new_proof,error));assert(new_proof.mac!=proof.mac && new_proof.canonical_metadata==proof.canonical_metadata);
    header.cycle_id++;nonce[2]++;std::vector<uint8_t> reconnected;
    assert(v3::make(secret,next_session,header,tx,ref,object,next_hello,in.session_id,new_proof,nonce,reconnected));
    assert(v3::accept(secret,next_session,next_hello,header,reconnected.data(),reconnected.size(),&guard,decoded)==v3::result::accepted);
    assert(decoded.identity_digest==identity);
#ifdef RKVA_SQLITE_COMPOSITION_TEST
    {
        namespace db=rkva_anchored_sqlite;namespace snap=rkva_native_snapshot;
        char dir[]="/tmp/rkva-prepare-reconnect-XXXXXX";assert(mkdtemp(dir));
        db::id ep{};ep.fill(0x51);db::secret key{};key.fill(0x52);
        auto token=rkva_managed_store::lease::try_acquire(dir,error);assert(token);
        db::backend store(dir,owner,ep,key);assert(store.provision(token,error));db::state st{};assert(store.inspect(token,st,error));
        auto persist=[&](const std::vector<uint8_t> & bytes,const std::vector<uint8_t> & meta) {
            db::pending_request req;req.owner=wire::get_id(bytes.data()+8);req.tx=wire::get_id(bytes.data()+40);
            req.reference=wire::get_id(bytes.data()+72);req.key=wire::get_id(bytes.data()+104);req.nonce=wire::get_id(bytes.data()+344);
            std::copy_n(bytes.data()+140,80,req.semantic_hello.data());req.metadata=meta;make_pending_identity(req);
            snap::Manifest m;assert(snap::decode(meta,m,&error));req.reservation=m.payload_bytes+meta.size()+(1u<<20);
            db::anchor anchor;anchor.owner=owner;anchor.epoch=ep;anchor.accepted=st;
            assert(store.challenge(token,anchor.challenge,error)&&db::sign_anchor(key,anchor));
            db::state out{};assert(store.test_prepare(token,&anchor,req,out,error)==db::result::ok);st=out;
        };
        persist(frame,proof.canonical_metadata);persist(reconnected,new_proof.canonical_metadata);assert(st.version==2);
        token.reset();std::filesystem::remove_all(dir);
    }
#endif
    assert(v3::accept(secret,session,hello,header,reconnected.data(),reconnected.size(),&guard,decoded)==v3::result::invalid);
    auto badproof=proof;badproof.mac[0]^=1;std::vector<uint8_t> out;
    assert(!v3::make(secret,session,header,tx,ref,object,hello,in.session_id,badproof,nonce,out));
    auto incompatible=hello;incompatible.n_head++;assert(!v3::make(secret,session,header,tx,ref,object,incompatible,in.session_id,proof,nonce,out));
    std::cout<<"PREPARE v3 metadata frame fixture passed; guard is not durable\n";
}
