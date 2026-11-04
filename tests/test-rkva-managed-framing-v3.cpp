#include "../ggml/src/ggml-remote-attn/rkva-managed-framing-v3.h"
#include "../ggml/src/ggml-remote-attn/rkva-managed-attest-v3.h"
#undef NDEBUG
#include <cassert>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>
#ifndef _WIN32
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
namespace v3=rkva_managed_framing_v3;
namespace auth=rkva_managed_auth_e2e;
namespace wire=rkva_registry_wire;
namespace snap=rkva_native_snapshot;
namespace native=rkva_native_state;
static wire::id id(uint8_t n){wire::id x{};x.fill(n);return x;}
struct guard final:auth::nonce_guard, v3::hello_guard, v3::hello_ack_guard {
    std::mutex mu;std::set<std::tuple<wire::id,uint64_t,auth::purpose,auth::nonce>> seen;
    auth::guard_result claim(const wire::id & o,uint64_t e,auth::purpose p,const auth::nonce & n) override {
        std::lock_guard<std::mutex> lock(mu);return seen.emplace(o,e,p,n).second?auth::guard_result::recorded:auth::guard_result::replay; }
    auth::guard_result consume(const wire::id & o,uint64_t e,auth::purpose issued,auth::purpose used,const auth::nonce & n) override {
        std::lock_guard<std::mutex> lock(mu);if(!seen.count({o,e,issued,n}))return auth::guard_result::unavailable;
        return seen.emplace(o,e,used,n).second?auth::guard_result::recorded:auth::guard_result::replay; }
    auth::guard_result admit(const wire::id & o,uint64_t e,const wire::id &,const v3::anchor &,const wire::id & n) override {
        return claim(o,e,auth::purpose::server_hello,n);
    }
    auth::guard_result commit(const wire::id & o,uint64_t e,const wire::id &,const v3::anchor &,const wire::id & n,
        const v3::hello_state &,const wire::id &,const wire::id &,const rkva_request &,const std::vector<uint8_t> &,
        const rkva_response &,const uint8_t *,size_t) override {
        return consume(o,e,auth::purpose::client_hello,auth::purpose::client_hello_reply,n);
    }
};
struct peer_guard final:v3::command_guard {
    std::set<wire::id> seen;int calls=0,effects=0;std::map<wire::id,wire::id> transactions;
    auth::guard_result admit(const v3::session & s,const v3::anchor & a,uint16_t op,const wire::request & q,
                             const v3::preparation * p,const v3::nonce & n) override {
        ++calls;if(!seen.insert(n).second)return auth::guard_result::replay;
        if(a.owner!=s.owner || a.ledger_epoch!=s.ledger_epoch || !v3::nz(a.mac))return auth::guard_result::unavailable;
        if(op==RKVA_REGISTRY_PREPARE){if(!p||!v3::nz(p->identity_digest))return auth::guard_result::unavailable;
            const auto it=transactions.find(q.tx);if(it==transactions.end()){transactions[q.tx]=q.key;++effects;}else if(it->second!=q.key)return auth::guard_result::unavailable;}
        return auth::guard_result::recorded;
    }
};
struct receipt_proof final:v3::reply_ack_guard {
    guard & client;bool allow=true;int calls=0;
    explicit receipt_proof(guard & g):client(g) {}
    auth::guard_result commit(const v3::session & s,uint16_t,const wire::request &,const v3::nonce & n,bool,
        const v3::acknowledgment & a,const rkva_request &,const std::vector<uint8_t> &,
        const rkva_response &,const uint8_t *,size_t) override {
        ++calls;if(!allow || (a.has_receipt && a.value.mac!=id(0x4f)))return auth::guard_result::unavailable;
        return client.consume(s.owner,s.auth_epoch,auth::purpose::client_command,auth::purpose::client_command_reply,n);
    }
};
namespace at=rkva_managed_attest_v3;
// Synthetic engine/ledger for wire tests only; cannot authorize runtime.
struct fixture_attest_peer:at::peer_guard {
    v3::secret key{};std::set<wire::id> nonces;bool open=true;uint64_t generation=0x700000001ull;
    explicit fixture_attest_peer(v3::secret value):key(value) {}
    auth::guard_result admit(const v3::session & s,const v3::anchor &,const at::request & q,const at::nonce & n,at::evidence & out) override {
        if(!open||q.generation!=generation)return auth::guard_result::unavailable;
        if(!nonces.insert(n).second)return auth::guard_result::replay;
        rkva_resident_attestation::input in;in.session_id=q.session_id;in.live_rows=q.rows;in.generation=q.generation;in.host_key=q.key;
        in.submitted_fence=20;in.completed_fence=21;in.geometry={8,2,256,2,4096,1};
        in.descriptor={1024,1024,RKVA_RANGE_STORAGE_KVARN4,3,128,128,256,2,4,4,8960,3,2,128,128,1,1,1};
        in.counts={513};in.record_groups={4};in.ready={true};in.layers={{385,384,4,1}};
        auth::session context{s.owner,s.auth_epoch,s.transcript,true};rkva_resident_attestation::evidence proof;std::string error;
        if(!rkva_resident_attestation::make_metadata(in,context,key,proof,error))return auth::guard_result::unavailable;
        out.requested=q;out.submitted=proof.submitted_fence;out.completed=proof.completed_fence;out.mac=proof.mac;out.metadata=proof.canonical_metadata;
        out.coverage=rkva_prepare_v3::attestation_digest(out.metadata,out.mac,q.session_id,out.submitted,out.completed);
        out.version=3;out.state_digest=id(0x65);out.next_challenge=id(0x2f);return auth::guard_result::recorded;
    }
};
struct fixture_attest_host:at::host_guard {
    guard & client;explicit fixture_attest_host(guard & c):client(c) {}
    auth::guard_result commit(const v3::session & s,const at::request &,const at::nonce & n,const at::evidence &,
       const rkva_request &,const std::vector<uint8_t> &,const rkva_response &,const uint8_t *,size_t) override {
        return client.consume(s.owner,s.auth_epoch,auth::purpose::client_command,auth::purpose::client_command_reply,n);
    }
};
static rkva_request req(uint16_t op,uint64_t cycle,size_t size) {
    rkva_request h{};h.magic=RKVA_MAGIC;h.version=RKVA_VERSION_V2;h.op=op;h.cycle_id=cycle;
    h.session_id=RKVA_SESSION_NONE;h.payload_bytes=uint32_t(size);return h;
}
static rkva_response rsp(const rkva_request & q,size_t size,int32_t status=0) {
    rkva_response h{};h.magic=RKVA_MAGIC;h.version=RKVA_VERSION_V2;h.op=q.op;h.cycle_id=q.cycle_id;
    h.status=status;h.payload_bytes=uint32_t(size);h.xbox_compute_us=9;return h;
}
static v3::hello hello() {
    v3::hello info;auto & h=info.proposed;
    h.proto_flags=RKVA_PROTO_MANAGED_REGISTRY_V1|RKVA_PROTO_MANAGED_AUTH_V2|RKVA_PROTO_MANAGED_FRAMING_V3;
    h.n_layer_remote=1;h.n_head=8;h.n_head_kv=2;h.head_dim=256;h.max_ctx=4096;
    h.cache_bits_k=16;h.cache_bits_v=16;h.group_tokens=128;h.sinkhorn_iters=16;h.tail_groups=2;
    h.domain=1;h.kq_scale=0.0625f;
    info.caps=RKVA_CAP_MANAGED_REGISTRY_V1|RKVA_CAP_MANAGED_AUTH_V2|RKVA_CAP_MANAGED_FRAMING_V3;
    info.ops=wire::operations|(1u<<RKVA_MANAGED_AUTH_HELLO)|(1u<<RKVA_MANAGED_AUTH_CHALLENGE)|(1u<<RKVA_MANAGED_ATTEST);
    info.server_version=436;return info;
}
static v3::preparation preparation(const v3::session & session,const wire::request & q) {
    v3::preparation p;p.request=q;p.hello=hello().proposed;p.session_id=7;p.submitted_fence=20;p.completed_fence=21;
    p.attestation_mac=id(0x91);p.coverage_digest=id(0x92);
    snap::Geometry geo{8,2,256,2,4096,1};snap::Descriptor desc{1024,1024,RKVA_RANGE_STORAGE_KVARN4,3,128,128,256,2,4,4,8960,3,2,128,128,1,1,1};
    snap::Manifest m;std::string error;const uint64_t generation=0x700000001ull;
    assert(snap::build(geo,desc,513,generation,q.key.data(),{{385,384,4,1}},m,&error));
    assert(snap::encode(m,p.metadata,&error));
    native::SetHeader nh;nh.layers=1;nh.layout.origin=1024;nh.layout.capacity=1024;nh.layout.heads=2;
    nh.layout.iterations=16;nh.layout.precision_tail_tokens=128;nh.layout.rows=513;nh.layout.record_base_group=0;
    nh.layout.saved_generation=generation;nh.payload_bytes=native::payload_bytes(nh.layout,nh.layers);
    nh.layer_bytes=native::layer_bytes(nh.layout);assert(native::encode_header(nh,p.native_header.data(),&error));
    std::vector<uint8_t> raw;assert(v3::canonical_prepare(p,session,raw));return p;
}
#ifndef _WIN32
static void send_all(int fd,const uint8_t * data,size_t len){while(len){const size_t take=std::min<size_t>(len,7);auto n=send(fd,data,take,0);assert(n>0);data+=n;len-=size_t(n);}}
static void stream_check(int read_fd,int write_fd,const std::vector<uint8_t> & frame,bool reply,bool complete=true) {
    std::thread sender([&]{send_all(write_fd,frame.data(),frame.size());shutdown(write_fd,SHUT_WR);});
    v3::stream_decoder parser(reply);std::array<uint8_t,13> buf{};
    for(;;){const auto n=recv(read_fd,buf.data(),buf.size(),0);assert(n>=0);if(!n)break;size_t used=0;assert(parser.feed(buf.data(),size_t(n),used)&&used==size_t(n));}
    sender.join();assert(parser.finish()==complete && parser.data()==frame);
}
#endif
int main(){
    v3::secret key{};key.fill(0xa1);const auto owner=id(0x11),ledger=id(0x77),tx=id(0x22),ref=id(0x33),object=id(0x44);
    guard client,server;peer_guard peer;receipt_proof proof(client);auto info=hello();
    assert(v3::negotiated(info.proposed.proto_flags,info.caps,info.ops));
    assert(!v3::negotiated(info.proposed.proto_flags&~RKVA_PROTO_MANAGED_FRAMING_V3,info.caps,info.ops));
    assert(!v3::negotiated(info.proposed.proto_flags,info.caps&~RKVA_CAP_MANAGED_FRAMING_V3,info.ops));
    assert(!v3::negotiated(info.proposed.proto_flags,info.caps,info.ops&~(1u<<RKVA_REGISTRY_STATUS)));
    assert(!v3::negotiated(info.proposed.proto_flags,info.caps,info.ops&~(1u<<RKVA_MANAGED_ATTEST)));
    assert(!v3::negotiated(info.proposed.proto_flags,info.caps,info.ops&~(1u<<RKVA_MANAGED_AUTH_CHALLENGE)));
    rkva_response challenge_header{};challenge_header.magic=RKVA_MAGIC;challenge_header.version=RKVA_VERSION_V2;
    challenge_header.op=RKVA_MANAGED_AUTH_CHALLENGE;challenge_header.payload_bytes=uint32_t(v3::prefix_bytes+v3::tag_bytes);
    std::vector<uint8_t> challenge;wire::id issued{};
    assert(v3::make_prehello_challenge(key,challenge_header,owner,7,ledger,id(0x31),challenge)==v3::status::accepted);
    assert(v3::accept_prehello_challenge(key,challenge_header,challenge.data(),challenge.size(),owner,7,ledger,issued)==v3::status::accepted && issued==id(0x31));
    for(size_t i=0;i<challenge.size();++i){auto bad=challenge;bad[i]^=1;assert(v3::accept_prehello_challenge(key,challenge_header,bad.data(),bad.size(),owner,7,ledger,issued)==v3::status::invalid);}
#ifndef _WIN32
    auto challenge_wire=v3::bytes(challenge_header);std::vector<uint8_t> challenge_stream(challenge_wire.begin(),challenge_wire.end());
    challenge_stream.insert(challenge_stream.end(),challenge.begin(),challenge.end());
    int challenge_socket[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,challenge_socket)==0);
    stream_check(challenge_socket[0],challenge_socket[1],challenge_stream,true);close(challenge_socket[0]);close(challenge_socket[1]);
#endif
    v3::anchor hello_anchor;hello_anchor.owner=owner;hello_anchor.ledger_epoch=ledger;hello_anchor.challenge=issued;
    hello_anchor.version=1;hello_anchor.digest=id(0x60);hello_anchor.mac=id(0x61);
    v3::hello_state hello_committed{2,id(0x62)},host_hello_state{};
    auto offer_h=req(RKVA_MANAGED_AUTH_HELLO,1,v3::prefix_bytes+v3::hello_bytes+v3::tag_bytes);
    std::vector<uint8_t> offer;assert(v3::make_offer(key,offer_h,info,owner,7,ledger,hello_anchor,id(1),nullptr,offer)==v3::status::unavailable);
    auto wrong=offer_h;wrong.payload_bytes--;assert(v3::make_offer(key,wrong,info,owner,7,ledger,hello_anchor,id(1),&client,offer)==v3::status::invalid);
    assert(v3::make_offer(key,offer_h,info,owner,7,ledger,hello_anchor,id(1),&client,offer)==v3::status::accepted);
    assert(offer.size()==offer_h.payload_bytes);
    v3::fields accepted;
    for(size_t i=0;i<offer.size();i++){auto bad=offer;bad[i]^=1;assert(v3::accept_offer(key,offer_h,bad.data(),bad.size(),info,owner,7,ledger,issued,&server,accepted)==v3::status::invalid);}
    for(size_t i=0;i<sizeof(offer_h);i++) {
        auto changed=offer_h;reinterpret_cast<uint8_t *>(&changed)[i]^=1;
        assert(v3::accept_offer(key,changed,offer.data(),offer.size(),info,owner,7,ledger,issued,&server,accepted)==v3::status::invalid);
    }
    // A different connection's issued challenge cannot admit this offer.
    assert(v3::accept_offer(key,offer_h,offer.data(),offer.size(),info,owner,7,ledger,id(0x32),&server,accepted)==v3::status::invalid);
    assert(v3::accept_offer(key,offer_h,offer.data(),offer.size(),info,owner,7,ledger,issued,&server,accepted)==v3::status::accepted);
    assert(v3::accept_offer(key,offer_h,offer.data(),offer.size(),info,owner,7,ledger,issued,&server,accepted)==v3::status::replay);
    auto hello_h=rsp(offer_h,v3::prefix_bytes+v3::hello_ack_bytes+v3::tag_bytes);std::vector<uint8_t> hello_frame;
    v3::session peer_session,host_session;
    assert(v3::make_hello_ack(key,offer_h,offer,hello_h,accepted,id(2),hello_committed,hello_frame,peer_session)==v3::status::accepted);
    for(size_t i=0;i<hello_frame.size();i++){auto bad=hello_frame;bad[i]^=1;assert(v3::accept_hello_ack(key,offer_h,offer,hello_h,bad.data(),bad.size(),&client,host_session,host_hello_state)==v3::status::invalid);}
    for(size_t i=0;i<sizeof(hello_h);i++) {
        auto changed=hello_h;reinterpret_cast<uint8_t *>(&changed)[i]^=1;
        assert(v3::accept_hello_ack(key,offer_h,offer,changed,hello_frame.data(),hello_frame.size(),&client,host_session,host_hello_state)==v3::status::invalid);
    }
    assert(v3::accept_hello_ack(key,offer_h,offer,hello_h,hello_frame.data(),hello_frame.size(),&client,host_session,host_hello_state)==v3::status::accepted);
    assert(host_session.transcript==peer_session.transcript && host_session.ledger_epoch==ledger && host_session.auth_epoch==7);
    assert(v3::accept_hello_ack(key,offer_h,offer,hello_h,hello_frame.data(),hello_frame.size(),&client,host_session,host_hello_state)==v3::status::replay);
    {
        auto peer_att=peer_session,host_att=host_session;fixture_attest_peer fake_peer(key);fixture_attest_host fake_host(client);
        v3::anchor att_anchor;att_anchor.owner=owner;att_anchor.ledger_epoch=ledger;att_anchor.challenge=host_att.peer_challenge;
        att_anchor.version=hello_committed.version;att_anchor.digest=hello_committed.digest;att_anchor.mac=id(0x55);
        at::request q{7,513,0x700000001ull,object},decoded_q;at::evidence result{};
        auto ah=req(RKVA_MANAGED_ATTEST,11,v3::prefix_bytes+at::request_bytes+v3::tag_bytes);std::vector<uint8_t> af;
        assert(at::make_request(key,host_att,ah,att_anchor,id(0x21),q,&client,af)==v3::status::accepted);
#ifndef _WIN32
        const auto ah_bytes=v3::bytes(ah);std::vector<uint8_t> att_stream(ah_bytes.begin(),ah_bytes.end());
        att_stream.insert(att_stream.end(),af.begin(),af.end());int att_socket[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,att_socket)==0);
        stream_check(att_socket[0],att_socket[1],att_stream,false);close(att_socket[0]);close(att_socket[1]);
#endif
        auto closed=peer_att;closed.authenticated=false;
        assert(at::accept_request(key,closed,ah,af.data(),af.size(),&fake_peer,decoded_q,result)==v3::status::invalid);
        for(size_t i=0;i<af.size();++i){auto bad=af;bad[i]^=1;
            assert(at::accept_request(key,peer_att,ah,bad.data(),bad.size(),&fake_peer,decoded_q,result)==v3::status::invalid);}
        assert(fake_peer.nonces.empty());
        fake_peer.generation++;assert(at::accept_request(key,peer_att,ah,af.data(),af.size(),&fake_peer,decoded_q,result)==v3::status::unavailable);
        fake_peer.generation--;assert(at::accept_request(key,peer_att,ah,af.data(),af.size(),nullptr,decoded_q,result)==v3::status::unavailable);
        assert(at::accept_request(key,peer_att,ah,af.data(),af.size(),&fake_peer,decoded_q,result)==v3::status::accepted);
        assert(at::accept_request(key,peer_att,ah,af.data(),af.size(),&fake_peer,decoded_q,result)==v3::status::replay);
        auto ar=rsp(ah,v3::prefix_bytes+at::response_fixed_bytes+result.metadata.size()+v3::tag_bytes);std::vector<uint8_t> reply;
        assert(at::make_reply(key,peer_att,ah,af,ar,result,reply)==v3::status::accepted);
        at::evidence host_result{};for(size_t i=0;i<reply.size();++i){auto bad=reply;bad[i]^=1;
            assert(at::accept_reply(key,host_att,ah,af,ar,bad.data(),bad.size(),&fake_host,host_result)==v3::status::invalid);}
        assert(at::accept_reply(key,host_att,ah,af,ar,reply.data(),reply.size(),nullptr,host_result)==v3::status::unavailable);
        assert(at::accept_reply(key,host_att,ah,af,ar,reply.data(),reply.size(),&fake_host,host_result)==v3::status::accepted);
        assert(host_result.metadata==result.metadata && host_att.peer_challenge==result.next_challenge);
        assert(at::accept_reply(key,host_att,ah,af,ar,reply.data(),reply.size(),&fake_host,host_result)!=v3::status::accepted);
    }
    wire::request query{owner,tx,ref,object,0};auto prep=preparation(host_session,query);
    v3::anchor anchor;anchor.owner=owner;anchor.ledger_epoch=ledger;anchor.challenge=id(2);anchor.version=3;anchor.digest=id(0x66);anchor.mac=id(0x77);
    std::vector<uint8_t> prep_body;assert(v3::canonical_prepare(prep,host_session,prep_body));
    auto h=req(RKVA_REGISTRY_PREPARE,2,v3::prefix_bytes+prep_body.size()+v3::tag_bytes);
    std::vector<uint8_t> command;assert(v3::make_command(key,host_session,h,anchor,id(3),query,&prep,&client,command)==v3::status::accepted);
    assert(command.size()==h.payload_bytes);
    wire::request decoded{};v3::preparation decoded_p;v3::fields decoded_f;
    for(size_t i=0;i<command.size();i++){auto bad=command;bad[i]^=1;assert(v3::accept_command(key,peer_session,h,bad.data(),bad.size(),&peer,decoded,decoded_p,decoded_f)==v3::status::invalid);}
    assert(peer.calls==0);
    for(size_t i=0;i<sizeof(h);i++) {
        auto changed=h;reinterpret_cast<uint8_t *>(&changed)[i]^=1;
        assert(v3::accept_command(key,peer_session,changed,command.data(),command.size(),&peer,decoded,decoded_p,decoded_f)==v3::status::invalid);
    }
    for(size_t n=0;n<command.size();n++)
        assert(v3::accept_command(key,peer_session,h,command.data(),n,&peer,decoded,decoded_p,decoded_f)==v3::status::invalid);
    // Even an outer MAC recomputed by a key holder cannot hide a mismatched
    // nested binding: the caller must still prove the original anchor.
    auto bad_binding=command;bad_binding[184+64]^=1;
    auto raw_header=v3::bytes(h);v3::id forged{};
    assert(v3::body_mac(key,"RKVA-V3-REQUEST",raw_header.data(),raw_header.size(),bad_binding.data(),bad_binding.size()-32,forged));
    std::copy(forged.begin(),forged.end(),bad_binding.end()-32);
    assert(v3::accept_command(key,peer_session,h,bad_binding.data(),bad_binding.size(),&peer,decoded,decoded_p,decoded_f)==v3::status::invalid);
    assert(peer.calls==0);
    auto corrupted_header=h;corrupted_header.cycle_id++;assert(v3::accept_command(key,peer_session,corrupted_header,command.data(),command.size(),&peer,decoded,decoded_p,decoded_f)==v3::status::invalid);
    assert(v3::accept_command(key,peer_session,h,command.data(),command.size(),nullptr,decoded,decoded_p,decoded_f)==v3::status::unavailable);
    assert(v3::accept_command(key,peer_session,h,command.data(),command.size(),&peer,decoded,decoded_p,decoded_f)==v3::status::accepted);
    assert(peer.effects==1 && decoded_p.native_header==prep.native_header && decoded_p.session_id==7);
    assert(v3::accept_command(key,peer_session,h,command.data(),command.size(),&peer,decoded,decoded_p,decoded_f)==v3::status::replay);
    const auto stable=decoded_p.identity_digest;
    auto changed=prep;changed.session_id=8;changed.submitted_fence=24;changed.completed_fence=25;changed.attestation_mac=id(0x93);changed.coverage_digest=id(0x94);
    std::vector<uint8_t> volatile_change;assert(v3::canonical_prepare(changed,host_session,volatile_change));v3::preparation inspect;
    assert(v3::decode_prepare(volatile_change,host_session,inspect) && inspect.identity_digest==stable);
    // No external/peer proof is inferred from the host RKVK or MACs.
    auto invalid=prep;invalid.native_header[16]^=1;std::vector<uint8_t> discard;assert(!v3::canonical_prepare(invalid,host_session,discard));
    v3::acknowledgment ack;ack.record={wire::phase::pending,4,0,owner,tx,ref,object};ack.state_digest=id(0x60);ack.next_challenge=id(0x80);
    auto rh=rsp(h,v3::prefix_bytes+220+v3::tag_bytes);std::vector<uint8_t> reply;
    assert(v3::make_reply(key,peer_session,h,command,rh,&ack,reply)==v3::status::accepted && reply.size()==rh.payload_bytes);
    v3::acknowledgment result;
    for(size_t i=0;i<reply.size();i++){auto bad=reply;bad[i]^=1;assert(v3::accept_reply(key,host_session,h,command,rh,bad.data(),bad.size(),&proof,result)==v3::status::invalid);}
    assert(proof.calls==0);
    for(size_t i=0;i<sizeof(rh);i++) {
        auto changed=rh;reinterpret_cast<uint8_t *>(&changed)[i]^=1;
        assert(v3::accept_reply(key,host_session,h,command,changed,reply.data(),reply.size(),&proof,result)==v3::status::invalid);
    }
    assert(proof.calls==0);
    assert(v3::accept_reply(key,host_session,h,command,rh,reply.data(),reply.size(),nullptr,result)==v3::status::unavailable);
    assert(v3::accept_reply(key,host_session,h,command,rh,reply.data(),reply.size(),&proof,result)==v3::status::accepted);
    assert(result.record.registry_version==4 && result.state_digest==id(0x60));
    assert(v3::accept_reply(key,host_session,h,command,rh,reply.data(),reply.size(),&proof,result)!=v3::status::accepted);
    assert(host_session.peer_challenge==id(0x80) && peer_session.peer_challenge==id(0x80));
    // Issue an idempotent retry and lose its ACK. The peer challenge advances,
    // so stale-session reuse is refused; reconnect with a fresh HELLO.
    anchor.challenge=id(0x80);auto lost_h=req(RKVA_REGISTRY_PREPARE,3,h.payload_bytes);std::vector<uint8_t> lost_cmd;
    assert(v3::make_command(key,host_session,lost_h,anchor,id(4),query,&prep,&client,lost_cmd)==v3::status::accepted);
    assert(v3::accept_command(key,peer_session,lost_h,lost_cmd.data(),lost_cmd.size(),&peer,decoded,decoded_p,decoded_f)==v3::status::accepted && peer.effects==1);
    auto lost_rh=rsp(lost_h,rh.payload_bytes);std::vector<uint8_t> lost_reply;ack.next_challenge=id(0x81);
    assert(v3::make_reply(key,peer_session,lost_h,lost_cmd,lost_rh,&ack,lost_reply)==v3::status::accepted);
#ifndef _WIN32
    // Send an ACK into a real socket but drop it without consumption. The
    // retry below reconnects and uses a fresh challenge and nonce.
    int lost_socket[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,lost_socket)==0);
    const auto lost_header=v3::bytes(lost_rh);send_all(lost_socket[0],lost_header.data(),lost_header.size());
    send_all(lost_socket[0],lost_reply.data(),lost_reply.size());close(lost_socket[1]);close(lost_socket[0]);
#endif
    assert(v3::accept_command(key,peer_session,lost_h,lost_cmd.data(),lost_cmd.size(),&peer,decoded,decoded_p,decoded_f)==v3::status::invalid);
    auto new_oh=req(RKVA_MANAGED_AUTH_HELLO,6,offer_h.payload_bytes);std::vector<uint8_t> new_offer;
    issued=id(0x32);hello_anchor.challenge=issued;hello_anchor.version=2;hello_committed.version=3;
    assert(v3::make_offer(key,new_oh,info,owner,7,ledger,hello_anchor,id(7),&client,new_offer)==v3::status::accepted);
    v3::fields accepted2;assert(v3::accept_offer(key,new_oh,new_offer.data(),new_offer.size(),info,owner,7,ledger,issued,&server,accepted2)==v3::status::accepted);
    auto new_hr=rsp(new_oh,hello_h.payload_bytes);std::vector<uint8_t> new_hello;
    assert(v3::make_hello_ack(key,new_oh,new_offer,new_hr,accepted2,id(8),hello_committed,new_hello,peer_session)==v3::status::accepted);
    assert(v3::accept_hello_ack(key,new_oh,new_offer,new_hr,new_hello.data(),new_hello.size(),&client,host_session,host_hello_state)==v3::status::accepted);
    anchor.challenge=id(8);auto h2=req(RKVA_REGISTRY_PREPARE,7,h.payload_bytes);std::vector<uint8_t> retry;
    assert(v3::make_command(key,host_session,h2,anchor,id(9),query,&prep,&client,retry)==v3::status::accepted);
    assert(v3::accept_command(key,peer_session,h2,retry.data(),retry.size(),&peer,decoded,decoded_p,decoded_f)==v3::status::accepted && peer.effects==1);
    auto rh2=rsp(h2,rh.payload_bytes);std::vector<uint8_t> reply2;ack.next_challenge=id(0x82);
    assert(v3::make_reply(key,peer_session,h2,retry,rh2,&ack,reply2)==v3::status::accepted);
    assert(v3::accept_reply(key,host_session,h2,retry,rh2,lost_reply.data(),lost_reply.size(),&proof,result)==v3::status::invalid);
    assert(v3::accept_reply(key,host_session,h2,retry,rh2,reply2.data(),reply2.size(),&proof,result)==v3::status::accepted);
    // Commit receipt stays in the single authenticated payload, not a suffix.
    anchor.challenge=host_session.peer_challenge;
    wire::request commit{};commit.owner=owner;commit.tx=tx;
    auto hc=req(RKVA_REGISTRY_COMMIT,4,v3::prefix_bytes+wire::request_bytes+v3::tag_bytes);std::vector<uint8_t> commit_frame;
    assert(v3::make_command(key,host_session,hc,anchor,id(5),commit,nullptr,&client,commit_frame)==v3::status::accepted);
    assert(v3::accept_command(key,peer_session,hc,commit_frame.data(),commit_frame.size(),&peer,decoded,decoded_p,decoded_f)==v3::status::accepted);
    v3::acknowledgment committed;committed.record={wire::phase::committed,5,5,owner,tx,ref,object};committed.state_digest=id(0x61);committed.next_challenge=id(0x83);committed.has_receipt=true;
    auto & rec=committed.value;rec.owner=owner;rec.ledger_epoch=ledger;rec.tx=tx;rec.reference=ref;rec.key=object;
    rec.identity_digest=stable;rec.object_digest=id(0x70);rec.commit_nonce=id(0x71);rec.previous_version=4;rec.previous_digest=id(0x60);
    rec.successor_version=5;rec.successor_digest=committed.state_digest;rec.commit_version=5;rec.mac=id(0x4f);
    auto rc=rsp(hc,v3::prefix_bytes+220+v3::receipt_bytes+v3::tag_bytes);std::vector<uint8_t> cr;
    assert(v3::make_reply(key,peer_session,hc,commit_frame,rc,&committed,cr)==v3::status::accepted);
    // A fresh outer MAC does not turn an unverified receipt into evidence.
    auto forged_receipt=cr;forged_receipt[v3::prefix_bytes+220+352]^=1;
    auto receipt_header=v3::bytes(rc);v3::id forged_ack{};
    assert(v3::body_mac(key,"RKVA-V3-RESPONSE",receipt_header.data(),receipt_header.size(),forged_receipt.data(),forged_receipt.size()-32,forged_ack));
    std::copy(forged_ack.begin(),forged_ack.end(),forged_receipt.end()-32);
    assert(v3::accept_reply(key,host_session,hc,commit_frame,rc,forged_receipt.data(),forged_receipt.size(),&proof,result)==v3::status::unavailable);
    proof.allow=false;assert(v3::accept_reply(key,host_session,hc,commit_frame,rc,cr.data(),cr.size(),&proof,result)==v3::status::unavailable);
    proof.allow=true;assert(v3::accept_reply(key,host_session,hc,commit_frame,rc,cr.data(),cr.size(),&proof,result)==v3::status::accepted);
    assert(result.has_receipt && result.value.object_digest==id(0x70) && result.value.commit_nonce==id(0x71));
    for(size_t i=0;i<cr.size();i++){auto bad=cr;bad[i]^=1;assert(v3::accept_reply(key,host_session,hc,commit_frame,rc,bad.data(),bad.size(),&proof,result)==v3::status::invalid);}
    // Signed negative reply is an error, never evidence of absent commit.
    anchor.challenge=host_session.peer_challenge;
    auto he=req(RKVA_REGISTRY_STATUS,5,v3::prefix_bytes+wire::request_bytes+v3::tag_bytes);std::vector<uint8_t> er;
    assert(v3::make_command(key,host_session,he,anchor,id(6),commit,nullptr,&client,er)==v3::status::accepted);
    auto unsupported=rsp(he,v3::prefix_bytes+220+v3::tag_bytes,-12);std::vector<uint8_t> ep;
    assert(v3::make_reply(key,peer_session,he,er,unsupported,nullptr,ep)==v3::status::invalid);
    auto re=rsp(he,v3::prefix_bytes+220+v3::tag_bytes,RKVA_ERR_STATE);
    assert(v3::make_reply(key,peer_session,he,er,re,nullptr,ep)==v3::status::accepted);
#ifndef _WIN32
    const auto error_header=v3::bytes(re);std::vector<uint8_t> error_stream(error_header.begin(),error_header.end());
    error_stream.insert(error_stream.end(),ep.begin(),ep.end());
    int error_socket[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,error_socket)==0);
    stream_check(error_socket[0],error_socket[1],error_stream,true);close(error_socket[0]);close(error_socket[1]);
#endif
    assert(v3::accept_reply(key,host_session,he,er,re,ep.data(),ep.size(),&proof,result)==v3::status::peer_error);
    // Stream parser: every truncated boundary fails; overlong length rejected.
    const auto hb=v3::bytes(h);std::vector<uint8_t> stream(hb.begin(),hb.end());stream.insert(stream.end(),command.begin(),command.end());
    for(size_t i=0;i<stream.size();i++){v3::stream_decoder d(false);size_t used=0;assert(d.feed(stream.data(),i,used));assert(!d.finish());}
    v3::stream_decoder d(false);size_t used=0;std::vector<uint8_t> concat=stream;concat.insert(concat.end(),stream.begin(),stream.end());
    assert(d.feed(concat.data(),concat.size(),used)&&d.finish()&&used==stream.size()&&d.data()==stream);
    rkva_request parsed_header{};assert(d.request(parsed_header)&&parsed_header.payload_bytes==command.size()&&d.payload_size()==command.size());
    const auto rbytes=v3::bytes(rc);std::vector<uint8_t> response_stream(rbytes.begin(),rbytes.end());response_stream.insert(response_stream.end(),cr.begin(),cr.end());
    v3::stream_decoder response_decoder(true);assert(response_decoder.feed(response_stream.data(),response_stream.size(),used)&&response_decoder.finish());
    rkva_response parsed_response{};assert(response_decoder.response(parsed_response)&&parsed_response.payload_bytes==cr.size());
    for(size_t i=0;i<response_stream.size();i++){v3::stream_decoder partial(true);size_t part=0;assert(partial.feed(response_stream.data(),i,part)&&!partial.finish());}
    auto huge=hb;wire::put32(huge.data()+28,UINT32_MAX);v3::stream_decoder reject(false);assert(!reject.feed(huge.data(),huge.size(),used)&&reject.failed());
#ifndef _WIN32
    int pair[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,pair)==0);stream_check(pair[0],pair[1],response_stream,true);close(pair[0]);close(pair[1]);
    int truncated[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,truncated)==0);
    const std::vector<uint8_t> partial(stream.begin(),stream.end()-1);
    stream_check(truncated[0],truncated[1],partial,false,false);close(truncated[0]);close(truncated[1]);
    int listener=socket(AF_INET,SOCK_STREAM,0);assert(listener>=0);sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);address.sin_port=0;
    assert(bind(listener,reinterpret_cast<sockaddr *>(&address),sizeof(address))==0 && listen(listener,1)==0);
    socklen_t len=sizeof(address);assert(getsockname(listener,reinterpret_cast<sockaddr *>(&address),&len)==0);
    int client_socket=socket(AF_INET,SOCK_STREAM,0);assert(client_socket>=0 && connect(client_socket,reinterpret_cast<sockaddr *>(&address),sizeof(address))==0);
    int accepted_socket=accept(listener,nullptr,nullptr);assert(accepted_socket>=0);stream_check(accepted_socket,client_socket,response_stream,true);
    close(accepted_socket);close(client_socket);
    client_socket=socket(AF_INET,SOCK_STREAM,0);assert(client_socket>=0 && connect(client_socket,reinterpret_cast<sockaddr *>(&address),sizeof(address))==0);
    accepted_socket=accept(listener,nullptr,nullptr);assert(accepted_socket>=0);stream_check(accepted_socket,client_socket,partial,false,false);
    close(accepted_socket);close(client_socket);close(listener);
#endif
    std::cout<<"managed framing v3 bounded authenticated streams, receipts, replay and retry: PASS\n";
}
