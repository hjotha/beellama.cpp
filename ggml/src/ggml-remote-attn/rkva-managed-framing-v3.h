#pragma once
// Experimental single v3 framing. No runtime advertises/dispatches this.
// Guards, real peer authority, object proof and durable nonce/anchor store are
// external and mandatory. This is not COMMIT, GC or production authorization.
#include "rkva-managed-auth-e2e.h"
#include "../../include/rkva-native-state.h"
#include "../../../tools/xbox/uwp/rkva/rkva-prepare-v3.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <vector>

namespace rkva_managed_framing_v3 {
namespace auth=rkva_managed_auth_e2e;
namespace wire=rkva_registry_wire;
namespace snap=rkva_native_snapshot;
namespace native=rkva_native_state;
using id=wire::id;
using secret=auth::secret;
using nonce=auth::nonce;
constexpr uint32_t revision=3;
constexpr size_t prefix_bytes=384, tag_bytes=32, anchor_bytes=168;
constexpr size_t receipt_bytes=384, hello_bytes=100, hello_ack_bytes=140, prepare_fixed_bytes=396;
constexpr size_t max_payload=snap::kMaxManifestBytes+prepare_fixed_bytes;
constexpr size_t max_frame=prefix_bytes+max_payload+tag_bytes;
enum class kind:uint32_t { offer=1, hello_ack=2, command=3, reply=4, challenge=5, attest=6, attest_reply=7 };
enum class status { accepted, peer_error, invalid, replay, unavailable };
struct anchor { id owner{},ledger_epoch{},challenge{}; uint64_t version=0; id digest{},mac{}; };
struct receipt {
    id owner{},ledger_epoch{},tx{},reference{},key{},identity_digest{},object_digest{},commit_nonce{};
    uint64_t previous_version=0,successor_version=0,commit_version=0;
    id previous_digest{},successor_digest{},mac{};
    uint32_t result=0;
};
struct session {
    id owner{},ledger_epoch{},transcript{},peer_challenge{};
    uint64_t auth_epoch=0;
    uint32_t hello_flags=0;
    bool authenticated=false;
};
struct fields { kind type=kind::offer; id owner{},ledger_epoch{},transcript{},nonce{},request_hash{},binding{};
                uint64_t auth_epoch=0; anchor host_anchor{}; std::vector<uint8_t> payload; };
struct hello { rkva_hello proposed{}; uint64_t caps=0; uint32_t ops=0,server_version=0; };
struct preparation {
    wire::request request{}; std::array<uint8_t,80> native_header{}; rkva_hello hello{};
    uint32_t session_id=0; uint64_t submitted_fence=0,completed_fence=0;
    id attestation_mac{},coverage_digest{},identity_digest{}; std::vector<uint8_t> metadata;
};
struct acknowledgment { wire::response record{}; id state_digest{},next_challenge{}; bool has_receipt=false; receipt value{}; };
struct hello_state { uint64_t version=0; id digest{}; };
struct hello_guard {
    virtual ~hello_guard()=default;
    virtual auth::guard_result admit(const id &,uint64_t,const id &,const anchor &,const nonce &)=0;
};
// The client must consume its issued nonce and persist the accepted state and
// challenge atomically before the codec exposes the new session or ACK.
struct hello_ack_guard {
    virtual ~hello_ack_guard()=default;
    virtual auth::guard_result commit(const id &,uint64_t,const id &,const anchor &,const nonce &,const hello_state &,
                                      const nonce &,const id &,const rkva_request &,const std::vector<uint8_t> &,
                                      const rkva_response &,const uint8_t *,size_t)=0;
};
struct reply_ack_guard {
    virtual ~reply_ack_guard()=default;
    virtual auth::guard_result commit(const session &,uint16_t,const wire::request &,const nonce &,
                                      bool,const acknowledgment &,const rkva_request &,const std::vector<uint8_t> &,
                                      const rkva_response &,const uint8_t *,size_t)=0;
};
// Guard verifies independently held durable anchor, epoch mapping, peer engine,
// frozen object and policy; the codec never substitutes HMAC for peer proof.
struct command_guard {
    virtual ~command_guard()=default;
    virtual auth::guard_result admit(const session &,const anchor &,uint16_t,const wire::request &,
                                     const preparation *,const nonce &)=0;
};
struct receipt_guard { virtual ~receipt_guard()=default; virtual bool validate(const session &,uint16_t,
                                      const wire::request &,const acknowledgment &)=0; };
inline bool nz(const id & v){return !wire::all_zero(v);}
inline bool negotiated(uint32_t flags,uint64_t caps,uint32_t ops) {
    return wire::negotiated(flags,caps,ops);
}
inline void put_anchor(uint8_t * p,const anchor & a) {
    wire::put_id(p,a.owner);wire::put_id(p+32,a.ledger_epoch);wire::put_id(p+64,a.challenge);
    wire::put64(p+96,a.version);wire::put_id(p+104,a.digest);wire::put_id(p+136,a.mac);
}
inline anchor get_anchor(const uint8_t * p) {
    anchor a;a.owner=wire::get_id(p);a.ledger_epoch=wire::get_id(p+32);a.challenge=wire::get_id(p+64);
    a.version=wire::get64(p+96);a.digest=wire::get_id(p+104);a.mac=wire::get_id(p+136);return a;
}
inline bool valid_anchor(const anchor & a,const session & s) {
    return a.owner==s.owner && a.ledger_epoch==s.ledger_epoch && a.challenge==s.peer_challenge &&
           nz(a.challenge) && nz(a.digest) && nz(a.mac);
}
inline void put_receipt(uint8_t * p,const receipt & r) {
    const id * ids[]={&r.owner,&r.ledger_epoch,&r.tx,&r.reference,&r.key,&r.identity_digest,&r.object_digest,&r.commit_nonce};
    for(size_t i=0;i<8;i++)wire::put_id(p+i*32,*ids[i]);
    wire::put64(p+256,r.previous_version);wire::put_id(p+264,r.previous_digest);
    wire::put64(p+296,r.successor_version);wire::put_id(p+304,r.successor_digest);
    wire::put64(p+336,r.commit_version);wire::put32(p+344,r.result);wire::put32(p+348,0);wire::put_id(p+352,r.mac);
}
inline receipt get_receipt(const uint8_t * p) {
    receipt r;id * ids[]={&r.owner,&r.ledger_epoch,&r.tx,&r.reference,&r.key,&r.identity_digest,&r.object_digest,&r.commit_nonce};
    for(size_t i=0;i<8;i++)*ids[i]=wire::get_id(p+i*32);
    r.previous_version=wire::get64(p+256);r.previous_digest=wire::get_id(p+264);
    r.successor_version=wire::get64(p+296);r.successor_digest=wire::get_id(p+304);
    r.commit_version=wire::get64(p+336);r.result=wire::get32(p+344);r.mac=wire::get_id(p+352);return r;
}
inline bool valid_receipt(const receipt & r,const session & s,const wire::response & record,const id & state_digest) {
    return r.owner==s.owner && r.ledger_epoch==s.ledger_epoch && r.tx==record.tx && r.reference==record.reference &&
        r.key==record.key && nz(r.tx)&&nz(r.reference)&&nz(r.key)&&nz(r.identity_digest)&&nz(r.object_digest)&&
        nz(r.commit_nonce)&&nz(r.previous_digest)&&nz(r.successor_digest)&&nz(r.mac)&&r.result==0 &&
        r.successor_version>r.previous_version && r.commit_version && r.commit_version<=r.successor_version &&
        r.successor_version<=record.registry_version && r.commit_version==record.committed_version &&
        (r.successor_version!=record.registry_version || r.successor_digest==state_digest);
}
inline std::vector<uint8_t> encode_hello(const hello & h) {
    std::vector<uint8_t> p(hello_bytes);rkva_prepare_v3::put_hello(h.proposed,p.data());
    wire::put64(p.data()+84,h.caps);wire::put32(p.data()+92,h.ops);wire::put32(p.data()+96,h.server_version);return p;
}
inline bool decode_hello(const std::vector<uint8_t> & p,hello & out) {
    if(p.size()!=hello_bytes)return false;
    hello h;h.proposed=rkva_prepare_v3::get_hello(p.data());
    h.caps=wire::get64(p.data()+84);h.ops=wire::get32(p.data()+92);h.server_version=wire::get32(p.data()+96);
    if(!negotiated(h.proposed.proto_flags,h.caps,h.ops))return false;
    out=h;return true;
}
inline bool valid_request_fields(uint16_t op,const wire::request & q,const id & owner) {
    if(q.owner!=owner || !wire::valid_request(op,q))return false;
    if(op==RKVA_REGISTRY_PREPARE)return nz(q.tx)&&nz(q.reference)&&nz(q.key);
    if(op==RKVA_REGISTRY_RELEASE)return nz(q.reference);
    return nz(q.tx);
}
inline bool canonical_prepare(const preparation & in,const session & s,std::vector<uint8_t> & out) {
    try {
        if(!valid_request_fields(RKVA_REGISTRY_PREPARE,in.request,s.owner)||in.metadata.empty()||
           in.metadata.size()>snap::kMaxManifestBytes||in.session_id==RKVA_SESSION_NONE||!in.submitted_fence||
           in.submitted_fence==UINT64_MAX||in.completed_fence==UINT64_MAX||in.completed_fence<in.submitted_fence||
           !nz(in.attestation_mac)||!nz(in.coverage_digest))return false;
        native::SetHeader rh{};std::string error;
        if(!native::decode_header(in.native_header.data(),rh,&error))return false;
        snap::Manifest m;std::vector<uint8_t> canonical;
        if(!snap::decode(in.metadata,m,&error)||!snap::encode(m,canonical,&error)||canonical!=in.metadata||
           !snap::validate(m,m.geometry,m.descriptor,m.live_rows,m.generation,in.request.key.data(),&error)||
           rh.layers!=m.geometry.n_layers||rh.layout.origin!=m.descriptor.origin||rh.layout.capacity!=m.descriptor.capacity||
           rh.layout.heads!=m.geometry.n_head_kv||rh.layout.precision_tail_tokens!=m.descriptor.precision_tail_tokens||
           rh.layout.rows!=m.live_rows||rh.layout.saved_generation!=m.generation||
           rh.layout.iterations!=in.hello.sinkhorn_iters||!rkva_prepare_v3::valid_hello(in.hello,m,s.hello_flags))return false;
        wire::request_wire q{};if(!wire::encode_request(RKVA_REGISTRY_PREPARE,in.request,q))return false;
        std::vector<uint8_t> p(prepare_fixed_bytes+in.metadata.size());std::copy(q.begin(),q.end(),p.begin());
        std::copy(in.native_header.begin(),in.native_header.end(),p.begin()+144);rkva_prepare_v3::put_hello(in.hello,p.data()+224);
        wire::put32(p.data()+308,in.session_id);wire::put64(p.data()+312,in.submitted_fence);wire::put64(p.data()+320,in.completed_fence);
        wire::put_id(p.data()+328,in.attestation_mac);wire::put_id(p.data()+360,in.coverage_digest);
        wire::put32(p.data()+392,uint32_t(in.metadata.size()));std::copy(in.metadata.begin(),in.metadata.end(),p.begin()+396);out=std::move(p);return true;
    } catch(...){return false;}
}
inline bool semantic_identity(const wire::request & request,const std::array<uint8_t,80> & header,
                              const std::array<uint8_t,80> & hello,const std::vector<uint8_t> & metadata,id & out) {
    wire::request_wire q{};if(!wire::encode_request(RKVA_REGISTRY_PREPARE,request,q))return false;
    rkva_snapshot::Sha256 hash;static constexpr char domain[]="RKVA-V3-SEMANTIC-IDENTITY";hash.update(domain,sizeof(domain)-1);
    hash.update(q.data(),q.size());hash.update(header.data(),header.size());hash.update(hello.data(),hello.size());
    hash.update(metadata.data(),metadata.size());out=hash.finish();return true;
}
inline bool decode_prepare(const std::vector<uint8_t> & p,const session & s,preparation & out) {
    try {
        if(p.size()<prepare_fixed_bytes||p.size()>max_payload||wire::get32(p.data()+392)!=p.size()-prepare_fixed_bytes)return false;
        preparation a;if(!wire::decode_request(RKVA_REGISTRY_PREPARE,p.data(),144,a.request))return false;
        std::copy_n(p.data()+144,80,a.native_header.data());a.hello=rkva_prepare_v3::get_hello(p.data()+224);
        a.session_id=wire::get32(p.data()+308);a.submitted_fence=wire::get64(p.data()+312);a.completed_fence=wire::get64(p.data()+320);
        a.attestation_mac=wire::get_id(p.data()+328);a.coverage_digest=wire::get_id(p.data()+360);
        a.metadata.assign(p.begin()+396,p.end());std::vector<uint8_t> encoded;if(!canonical_prepare(a,s,encoded)||encoded!=p)return false;
        // Stable semantic identity excludes flags, nonce, transcript, session,
        // fences and MACs. RKVK contains no resident coverage.
        std::array<uint8_t,80> semantic_hello{};std::copy_n(p.data()+228,80,semantic_hello.data());
        if(!semantic_identity(a.request,a.native_header,semantic_hello,a.metadata,a.identity_digest))return false;
        out=std::move(a);return true;
    }catch(...){return false;}
}
inline bool request_header(const rkva_request & h,size_t n,kind k) {
    return n>=prefix_bytes+tag_bytes && n<=max_frame && h.magic==RKVA_MAGIC && h.version==RKVA_VERSION_V2 &&
        h.session_id==RKVA_SESSION_NONE && !h.layer_id && !h.n_tokens && h.payload_bytes==n &&
        ((k==kind::offer && h.op==RKVA_MANAGED_AUTH_HELLO)||(k==kind::command && wire::valid_op(h.op))||
         (k==kind::attest && h.op==RKVA_MANAGED_ATTEST));
}
inline bool response_header(const rkva_response & h,size_t n,kind k) {
    return n>=prefix_bytes+tag_bytes && n<=max_frame && h.magic==RKVA_MAGIC && h.version==RKVA_VERSION_V2 &&
        h.payload_bytes==n && h.status<=RKVA_OK && h.status>=RKVA_ERR_SNAPSHOT_IO && ((k==kind::challenge && h.op==RKVA_MANAGED_AUTH_CHALLENGE && h.cycle_id==0 && h.status==RKVA_OK)||
                                               (k==kind::hello_ack && h.op==RKVA_MANAGED_AUTH_HELLO && h.status==RKVA_OK)||
                                               (k==kind::attest_reply && h.op==RKVA_MANAGED_ATTEST && h.status==RKVA_OK)||
                                               (k==kind::reply && wire::valid_op(h.op)));
}
inline std::array<uint8_t,32> bytes(const rkva_request & h) { std::array<uint8_t,32> b{};auth::req_header_bytes(h,b.data());return b; }
inline std::array<uint8_t,56> bytes(const rkva_response & h) { std::array<uint8_t,56> b{};auth::resp_header_bytes(h,b.data());return b; }
inline bool body_mac(const secret & key,const char * domain,const uint8_t * hdr,size_t hs,
                     const uint8_t * body,size_t size,id & result) {
    if(!hdr||!body||wire::all_zero(key)||size>max_frame||hs>56)return false;
    try {const size_t n=std::strlen(domain);std::vector<uint8_t> input;input.reserve(n+hs+size);
        input.insert(input.end(),domain,domain+n);input.insert(input.end(),hdr,hdr+hs);input.insert(input.end(),body,body+size);
        return rkva_managed_auth::hmac_sha256(key.data(),key.size(),input.data(),input.size(),result);
    }catch(...){return false;}
}
inline bool seal(const secret & key,const uint8_t * hdr,size_t hs,const fields & f,std::vector<uint8_t> & out) {
    try {
        if(!nz(f.owner)||!nz(f.ledger_epoch)||!f.auth_epoch||!nz(f.nonce)||f.payload.size()>max_payload)return false;
        std::vector<uint8_t> b(prefix_bytes+f.payload.size()+tag_bytes,0);
        wire::put32(b.data(),revision);wire::put32(b.data()+4,uint32_t(f.type));wire::put32(b.data()+12,uint32_t(f.payload.size()));
        wire::put_id(b.data()+16,f.owner);wire::put64(b.data()+48,f.auth_epoch);wire::put_id(b.data()+56,f.ledger_epoch);
        wire::put_id(b.data()+88,f.transcript);wire::put_id(b.data()+120,f.nonce);wire::put_id(b.data()+152,f.request_hash);
        put_anchor(b.data()+184,f.host_anchor);std::copy(f.payload.begin(),f.payload.end(),b.begin()+prefix_bytes);
        if(f.type==kind::command || f.type==kind::attest) {
            id binding{};if(!body_mac(key,"RKVA-V3-BINDING",hdr,hs,b.data(),b.size()-tag_bytes,binding))return false;
            wire::put_id(b.data()+352,binding);
        } else if(nz(f.binding))return false;
        id tag{};const char * domain=hs==32?"RKVA-V3-REQUEST":"RKVA-V3-RESPONSE";
        if(!body_mac(key,domain,hdr,hs,b.data(),b.size()-tag_bytes,tag))return false;
        std::copy(tag.begin(),tag.end(),b.end()-tag_bytes);out=std::move(b);return true;
    }catch(...){return false;}
}
inline bool open(const secret & key,const uint8_t * hdr,size_t hs,const uint8_t * p,size_t n,kind expected,fields & out) {
    try {
        if(!p||n<prefix_bytes+tag_bytes||n>max_frame||wire::get32(p)!=revision||wire::get32(p+4)!=uint32_t(expected)||
           wire::get32(p+8)!=0||wire::get32(p+12)!=n-prefix_bytes-tag_bytes)return false;
        id mac{};const char * domain=hs==32?"RKVA-V3-REQUEST":"RKVA-V3-RESPONSE";
        if(!body_mac(key,domain,hdr,hs,p,n-tag_bytes,mac)||!rkva_managed_auth::equal_tag(mac.data(),p+n-tag_bytes))return false;
        fields f;f.type=expected;f.owner=wire::get_id(p+16);f.auth_epoch=wire::get64(p+48);f.ledger_epoch=wire::get_id(p+56);
        f.transcript=wire::get_id(p+88);f.nonce=wire::get_id(p+120);f.request_hash=wire::get_id(p+152);
        f.host_anchor=get_anchor(p+184);f.binding=wire::get_id(p+352);
        if(!nz(f.owner)||!f.auth_epoch||!nz(f.ledger_epoch)||!nz(f.nonce))return false;
        if(expected==kind::command || expected==kind::attest) {
            std::vector<uint8_t> canonical(p,p+n-tag_bytes);std::fill(canonical.begin()+352,canonical.begin()+384,0);
            id binding{};if(!body_mac(key,"RKVA-V3-BINDING",hdr,hs,canonical.data(),canonical.size(),binding)||
                !rkva_managed_auth::equal_tag(binding.data(),f.binding.data()))return false;
        } else if(nz(f.binding))return false;
        f.payload.assign(p+prefix_bytes,p+n-tag_bytes);out=std::move(f);return true;
    }catch(...){return false;}
}
inline bool empty_anchor(const anchor & a) {return !nz(a.owner)&&!nz(a.ledger_epoch)&&!nz(a.challenge)&&!a.version&&!nz(a.digest)&&!nz(a.mac);}
inline id request_digest(const rkva_request & h,const std::vector<uint8_t> & b) {
    const auto header=bytes(h);rkva_snapshot::Sha256 hash;static constexpr char domain[]="RKVA-V3-REQUEST-DIGEST";
    hash.update(domain,sizeof(domain)-1);hash.update(header.data(),header.size());hash.update(b.data(),b.size());return hash.finish();
}
inline id transcript(const rkva_request & q,const std::vector<uint8_t> & request,const rkva_response & a,const std::vector<uint8_t> & reply) {
    auto qb=bytes(q);auto ab=bytes(a);rkva_snapshot::Sha256 h;static constexpr char d[]="RKVA-V3-SESSION";
    h.update(d,sizeof(d)-1);h.update(qb.data(),qb.size());h.update(request.data(),request.size());
    h.update(ab.data(),ab.size());h.update(reply.data(),reply.size());return h.finish();
}
inline status convert(auth::status x) {
    return x==auth::status::accepted?status::accepted:x==auth::status::replay?status::replay:
           x==auth::status::unavailable?status::unavailable:status::invalid;
}
// Challenge comes from the SQLite backend bound to this accepted connection.
inline status make_prehello_challenge(const secret & key,const rkva_response & h,const id & owner,uint64_t epoch,
                                      const id & ledger,const nonce & challenge,std::vector<uint8_t> & out) {
    if(!response_header(h,prefix_bytes+tag_bytes,kind::challenge))return status::invalid;
    fields f;f.type=kind::challenge;f.owner=owner;f.auth_epoch=epoch;f.ledger_epoch=ledger;f.nonce=challenge;
    const auto hdr=bytes(h);return seal(key,hdr.data(),hdr.size(),f,out)?status::accepted:status::invalid;
}
inline status accept_prehello_challenge(const secret & key,const rkva_response & h,const uint8_t * p,size_t n,
                                        const id & owner,uint64_t epoch,const id & ledger,nonce & out) {
    fields f;const auto hdr=bytes(h);
    if(!response_header(h,n,kind::challenge)||!open(key,hdr.data(),hdr.size(),p,n,kind::challenge,f)||
       f.owner!=owner||f.auth_epoch!=epoch||f.ledger_epoch!=ledger||nz(f.transcript)||nz(f.request_hash)||
       !empty_anchor(f.host_anchor)||!f.payload.empty())return status::invalid;
    out=f.nonce;return status::accepted;
}
inline status make_offer(const secret & key,const rkva_request & h,const hello & info,const id & owner,uint64_t epoch,
                         const id & ledger,const anchor & host,const nonce & challenge,auth::nonce_guard * guard,std::vector<uint8_t> & output) {
    try {const auto payload=encode_hello(info);hello validated;
        if(!decode_hello(payload,validated)||!request_header(h,prefix_bytes+payload.size()+tag_bytes,kind::offer))return status::invalid;
        if(host.owner!=owner||host.ledger_epoch!=ledger||!nz(host.challenge)||!nz(host.digest)||!nz(host.mac))return status::invalid;
        fields f;f.type=kind::offer;f.owner=owner;f.auth_epoch=epoch;f.ledger_epoch=ledger;f.nonce=challenge;f.host_anchor=host;f.payload=payload;
        std::vector<uint8_t> b;const auto hdr=bytes(h);if(!seal(key,hdr.data(),hdr.size(),f,b))return status::invalid;
        const auto r=convert(auth::claim(guard,owner,epoch,auth::purpose::client_hello,challenge));
        if(r!=status::accepted)return r;
        output=std::move(b);return status::accepted;
    }catch(...){return status::unavailable;}
}
inline status accept_offer(const secret & key,const rkva_request & h,const uint8_t * p,size_t n,const hello & expected,
                           const id & owner,uint64_t epoch,const id & ledger,const nonce & issued,hello_guard * guard,fields & out) {
    try {fields f;hello got;const auto hdr=bytes(h);
        if(!request_header(h,n,kind::offer)||!open(key,hdr.data(),hdr.size(),p,n,kind::offer,f)||
           f.owner!=owner||f.auth_epoch!=epoch||f.ledger_epoch!=ledger||nz(f.transcript)||nz(f.request_hash)||
           f.host_anchor.owner!=owner||f.host_anchor.ledger_epoch!=ledger||!nz(issued)||f.host_anchor.challenge!=issued||
           !nz(f.host_anchor.digest)||!nz(f.host_anchor.mac)||!decode_hello(f.payload,got)||f.payload!=encode_hello(expected))return status::invalid;
        if(!guard)return status::unavailable;
        const auto r=guard->admit(owner,epoch,ledger,f.host_anchor,f.nonce);
        if(r==auth::guard_result::replay)return status::replay;
        if(r!=auth::guard_result::recorded)return status::unavailable;
        out=std::move(f);return status::accepted;
    }catch(...){return status::unavailable;}
}
inline status make_hello_ack(const secret & key,const rkva_request & h,const std::vector<uint8_t> & offer,
                             const rkva_response & rh,const fields & accepted,const nonce & challenge,
                             const hello_state & committed,std::vector<uint8_t> & output,session & server) {
    try {fields original;const auto request_bytes=bytes(h);
        if(!request_header(h,offer.size(),kind::offer)||!response_header(rh,prefix_bytes+hello_ack_bytes+tag_bytes,kind::hello_ack)||
          rh.cycle_id!=h.cycle_id||!nz(challenge)||challenge==accepted.nonce||
          !open(key,request_bytes.data(),request_bytes.size(),offer.data(),offer.size(),kind::offer,original)||
          original.owner!=accepted.owner||original.auth_epoch!=accepted.auth_epoch||original.ledger_epoch!=accepted.ledger_epoch||
          original.nonce!=accepted.nonce||original.payload!=accepted.payload||nz(original.transcript)||nz(original.request_hash)||
          original.host_anchor.owner!=accepted.owner||original.host_anchor.ledger_epoch!=accepted.ledger_epoch||
          !nz(original.host_anchor.challenge)||challenge==original.host_anchor.challenge||!nz(original.host_anchor.digest)||
          !nz(original.host_anchor.mac)||!nz(committed.digest)||committed.version<=original.host_anchor.version)return status::invalid;
        fields f;f.type=kind::hello_ack;f.owner=accepted.owner;f.ledger_epoch=accepted.ledger_epoch;f.auth_epoch=accepted.auth_epoch;
        f.nonce=challenge;f.request_hash=request_digest(h,offer);f.payload=accepted.payload;
        f.payload.resize(hello_ack_bytes);wire::put64(f.payload.data()+hello_bytes,committed.version);
        wire::put_id(f.payload.data()+hello_bytes+8,committed.digest);
        std::vector<uint8_t> b;auto hdr=bytes(rh);if(!seal(key,hdr.data(),hdr.size(),f,b))return status::invalid;
        hello info;if(!decode_hello(accepted.payload,info))return status::invalid;
        session next{f.owner,f.ledger_epoch,transcript(h,offer,rh,b),challenge,f.auth_epoch,info.proposed.proto_flags,true};
        output=std::move(b);server=next;return status::accepted;
    }catch(...){return status::unavailable;}
}
inline status accept_hello_ack(const secret & key,const rkva_request & h,const std::vector<uint8_t> & offer,
                               const rkva_response & rh,const uint8_t * p,size_t n,hello_ack_guard * guard,session & out,hello_state & committed) {
    try {fields o,a;hello info;auto qh=bytes(h);auto ah=bytes(rh);
        if(!request_header(h,offer.size(),kind::offer)||!response_header(rh,n,kind::hello_ack)||rh.cycle_id!=h.cycle_id||
           !open(key,qh.data(),qh.size(),offer.data(),offer.size(),kind::offer,o)||!decode_hello(o.payload,info)||
           nz(o.transcript)||nz(o.request_hash)||o.host_anchor.owner!=o.owner||o.host_anchor.ledger_epoch!=o.ledger_epoch||
           !nz(o.host_anchor.challenge)||!nz(o.host_anchor.digest)||!nz(o.host_anchor.mac)||
           !open(key,ah.data(),ah.size(),p,n,kind::hello_ack,a)||a.owner!=o.owner||a.ledger_epoch!=o.ledger_epoch||
           a.auth_epoch!=o.auth_epoch||nz(a.transcript)||!empty_anchor(a.host_anchor)||a.nonce==o.nonce||
           a.payload.size()!=hello_ack_bytes||!std::equal(o.payload.begin(),o.payload.end(),a.payload.begin())||
           a.nonce==o.host_anchor.challenge||a.request_hash!=request_digest(h,offer))return status::invalid;
        hello_state state{wire::get64(a.payload.data()+hello_bytes),wire::get_id(a.payload.data()+hello_bytes+8)};
        if(state.version<=o.host_anchor.version||!nz(state.digest)||!guard)return status::unavailable;
        std::vector<uint8_t> reply(p,p+n);
        const auto trans=transcript(h,offer,rh,reply);
        const auto r=guard->commit(o.owner,o.auth_epoch,o.ledger_epoch,o.host_anchor,o.nonce,state,a.nonce,trans,h,offer,rh,p,n);
        if(r==auth::guard_result::replay)return status::replay;
        if(r!=auth::guard_result::recorded)return status::unavailable;
        out=session{o.owner,o.ledger_epoch,trans,a.nonce,o.auth_epoch,info.proposed.proto_flags,true};committed=state;return status::accepted;
    }catch(...){return status::unavailable;}
}
inline bool valid_session(const session & s) {return s.authenticated&&nz(s.owner)&&nz(s.ledger_epoch)&&nz(s.transcript)&&nz(s.peer_challenge)&&s.auth_epoch;}
inline status make_command(const secret & key,const session & s,const rkva_request & h,const anchor & a,const nonce & n,
                           const wire::request & q,const preparation * prep,auth::nonce_guard * guard,std::vector<uint8_t> & output) {
    try {if(!valid_session(s)||!valid_anchor(a,s)||!valid_request_fields(h.op,q,s.owner))return status::invalid;
        std::vector<uint8_t> payload;
        if(h.op==RKVA_REGISTRY_PREPARE) {if(!prep||prep->request.owner!=q.owner||prep->request.tx!=q.tx||
            prep->request.reference!=q.reference||prep->request.key!=q.key||!canonical_prepare(*prep,s,payload))return status::invalid;}
        else {wire::request_wire w{};if(prep||!wire::encode_request(h.op,q,w))return status::invalid;payload.assign(w.begin(),w.end());}
        if(!request_header(h,prefix_bytes+payload.size()+tag_bytes,kind::command))return status::invalid;
        fields f;f.type=kind::command;f.owner=s.owner;f.ledger_epoch=s.ledger_epoch;f.auth_epoch=s.auth_epoch;
        f.transcript=s.transcript;f.nonce=n;f.host_anchor=a;f.payload=std::move(payload);
        std::vector<uint8_t> b;auto hdr=bytes(h);if(!seal(key,hdr.data(),hdr.size(),f,b))return status::invalid;
        const auto r=convert(auth::claim(guard,s.owner,s.auth_epoch,auth::purpose::client_command,n));
        if(r!=status::accepted)return r;
        output=std::move(b);return status::accepted;
    }catch(...){return status::unavailable;}
}
inline status accept_command(const secret & key,const session & s,const rkva_request & h,const uint8_t * p,size_t n,
                             command_guard * guard,wire::request & output,preparation & prep_out,fields & frame_out) {
    try {fields f;auto hdr=bytes(h);wire::request q{};preparation prep;
        if(!valid_session(s)||!request_header(h,n,kind::command)||!open(key,hdr.data(),hdr.size(),p,n,kind::command,f)||
           f.owner!=s.owner||f.ledger_epoch!=s.ledger_epoch||f.auth_epoch!=s.auth_epoch||f.transcript!=s.transcript||
           nz(f.request_hash)||!valid_anchor(f.host_anchor,s))return status::invalid;
        if(h.op==RKVA_REGISTRY_PREPARE) {if(!decode_prepare(f.payload,s,prep))return status::invalid;q=prep.request;}
        else if(!wire::decode_request(h.op,f.payload.data(),f.payload.size(),q)||!valid_request_fields(h.op,q,s.owner))return status::invalid;
        if(!guard)return status::unavailable;
        const auto result=guard->admit(s,f.host_anchor,h.op,q,h.op==RKVA_REGISTRY_PREPARE?&prep:nullptr,f.nonce);
        if(result==auth::guard_result::replay)return status::replay;
        if(result!=auth::guard_result::recorded)return status::unavailable;
        output=q;if(h.op==RKVA_REGISTRY_PREPARE)prep_out=std::move(prep);frame_out=std::move(f);return status::accepted;
    }catch(...){return status::unavailable;}
}
inline bool encode_ack(uint16_t op,const wire::request & q,const session & s,const acknowledgment & a,std::vector<uint8_t> & out) {
    try {wire::response_wire record{};if(!wire::encode_response(op,q,a.record,record)||!nz(a.state_digest))return false;
        const bool receipt_required=a.record.state==wire::phase::committed||a.record.state==wire::phase::released;
        if(receipt_required!=a.has_receipt)return false;
        if(a.has_receipt&&!valid_receipt(a.value,s,a.record,a.state_digest))return false;
        if(!nz(a.next_challenge)||a.next_challenge==s.peer_challenge)return false;
        std::vector<uint8_t> p(220+(a.has_receipt?receipt_bytes:0),0);std::copy(record.begin(),record.end(),p.begin());
        wire::put_id(p.data()+152,a.state_digest);wire::put32(p.data()+184,a.has_receipt?receipt_bytes:0);
        wire::put_id(p.data()+188,a.next_challenge);
        if(a.has_receipt)put_receipt(p.data()+220,a.value);
        out=std::move(p);return true;
    }catch(...){return false;}
}
inline bool decode_ack(uint16_t op,const wire::request & q,const session & s,const std::vector<uint8_t> & p,acknowledgment & out) {
    try {if(p.size()!=220 && p.size()!=220+receipt_bytes)return false;
        const auto size=wire::get32(p.data()+184);if(size!=p.size()-220)return false;
        acknowledgment a;if(!wire::decode_response(op,q,p.data(),152,a.record))return false;
        a.state_digest=wire::get_id(p.data()+152);a.next_challenge=wire::get_id(p.data()+188);a.has_receipt=size!=0;
        if(a.has_receipt) {if(size!=receipt_bytes||wire::get32(p.data()+220+348)!=0)return false;a.value=get_receipt(p.data()+220);}
        std::vector<uint8_t> encoded;if(!encode_ack(op,q,s,a,encoded)||encoded!=p)return false;out=std::move(a);return true;
    }catch(...){return false;}
}
inline bool command_from_wire(const secret & key,const session & s,const rkva_request & h,const std::vector<uint8_t> & command,
                              fields & f,wire::request & q) {
    auto hb=bytes(h);preparation prep;
    if(!valid_session(s)||!request_header(h,command.size(),kind::command)||!open(key,hb.data(),hb.size(),command.data(),command.size(),kind::command,f)||
       f.owner!=s.owner||f.ledger_epoch!=s.ledger_epoch||f.auth_epoch!=s.auth_epoch||f.transcript!=s.transcript||
       nz(f.request_hash)||!valid_anchor(f.host_anchor,s))return false;
    if(h.op==RKVA_REGISTRY_PREPARE) {if(!decode_prepare(f.payload,s,prep))return false;q=prep.request;}
    else if(!wire::decode_request(h.op,f.payload.data(),f.payload.size(),q)||!valid_request_fields(h.op,q,s.owner))return false;
    return true;
}
// Peer must call accept_command and complete its authoritative transaction
// before signing an ACK. This framing library cannot enforce dispatch order.
inline status make_reply(const secret & key,session & s,const rkva_request & h,const std::vector<uint8_t> & command,
                         const rkva_response & rh,const acknowledgment * a,std::vector<uint8_t> & output) {
    try {fields request_frame;wire::request q{};if(!command_from_wire(key,s,h,command,request_frame,q))return status::invalid;
        std::vector<uint8_t> payload;
        if(rh.status==RKVA_OK) {if(!a||!encode_ack(h.op,q,s,*a,payload))return status::invalid;}
        else {if(a)return status::invalid;payload.assign(220,0);}
        if(!response_header(rh,prefix_bytes+payload.size()+tag_bytes,kind::reply)||rh.op!=h.op||rh.cycle_id!=h.cycle_id)return status::invalid;
        fields f;f.type=kind::reply;f.owner=s.owner;f.ledger_epoch=s.ledger_epoch;f.auth_epoch=s.auth_epoch;
        f.transcript=s.transcript;f.nonce=request_frame.nonce;f.request_hash=request_digest(h,command);f.payload=std::move(payload);
        auto rhb=bytes(rh);std::vector<uint8_t> result;if(!seal(key,rhb.data(),rhb.size(),f,result))return status::invalid;
        output=std::move(result);
        if(rh.status==RKVA_OK)s.peer_challenge=a->next_challenge;else s.authenticated=false;
        return status::accepted;
    }catch(...){return status::unavailable;}
}
inline status accept_reply(const secret & key,session & s,const rkva_request & h,const std::vector<uint8_t> & command,
                           const rkva_response & rh,const uint8_t * p,size_t n,reply_ack_guard * guard,acknowledgment & output) {
    try {fields request_frame,reply;wire::request q{};auto hdr=bytes(rh);
        if(!command_from_wire(key,s,h,command,request_frame,q)||!response_header(rh,n,kind::reply)||rh.op!=h.op||rh.cycle_id!=h.cycle_id||
           !open(key,hdr.data(),hdr.size(),p,n,kind::reply,reply)||reply.owner!=s.owner||reply.ledger_epoch!=s.ledger_epoch||
           reply.auth_epoch!=s.auth_epoch||reply.transcript!=s.transcript||reply.nonce!=request_frame.nonce||
           reply.request_hash!=request_digest(h,command)||!empty_anchor(reply.host_anchor))return status::invalid;
        acknowledgment a;
        if(rh.status==RKVA_OK) {if(!decode_ack(h.op,q,s,reply.payload,a))return status::invalid;}
        else if(reply.payload.size()!=220||!std::all_of(reply.payload.begin(),reply.payload.end(),[](uint8_t b){return b==0;}))return status::invalid;
        if(!guard)return status::unavailable;
        const auto r=guard->commit(s,h.op,q,request_frame.nonce,rh.status==RKVA_OK,a,h,command,rh,p,n);
        if(r==auth::guard_result::replay)return status::replay;
        if(r!=auth::guard_result::recorded)return status::unavailable;
        if(rh.status!=RKVA_OK){s.authenticated=false;return status::peer_error;}
        s.peer_challenge=a.next_challenge;output=std::move(a);return status::accepted;
    }catch(...){return status::unavailable;}
}
inline bool parse_request_header(const uint8_t * p,size_t n,rkva_request & out) {
    if(!p||n!=32)return false;
    rkva_request h{};
    h.magic=wire::get32(p);h.version=uint16_t(p[4])|uint16_t(p[5])<<8;h.op=uint16_t(p[6])|uint16_t(p[7])<<8;
    h.cycle_id=wire::get64(p+8);h.session_id=wire::get32(p+16);h.layer_id=wire::get32(p+20);
    h.n_tokens=wire::get32(p+24);h.payload_bytes=wire::get32(p+28);out=h;return true;
}
inline bool parse_response_header(const uint8_t * p,size_t n,rkva_response & out) {
    if(!p||n!=56)return false;
    rkva_response h{};
    h.magic=wire::get32(p);h.version=uint16_t(p[4])|uint16_t(p[5])<<8;h.op=uint16_t(p[6])|uint16_t(p[7])<<8;
    h.cycle_id=wire::get64(p+8);h.status=static_cast<int32_t>(wire::get32(p+16));h.payload_bytes=wire::get32(p+20);
    h.xbox_receive_us=wire::get64(p+24);h.xbox_prepare_us=wire::get64(p+32);h.xbox_compute_us=wire::get64(p+40);
    h.xbox_response_us=wire::get64(p+48);out=h;return true;
}
// Incremental, bounded parser. It never authenticates; caller must validate the
// decoded header/body. `used` leaves concatenated bytes for the next decoder.
class stream_decoder {
public:
    explicit stream_decoder(bool reply):reply_(reply),header_size_(reply?56:32) { bytes_.reserve(header_size_); }
    bool feed(const uint8_t * p,size_t n,size_t & used) {
        used=0;if(failed_||(!p&&n))return false;
        try {while(used<n && !complete()) {
            const size_t limit=expected_?expected_:header_size_;const size_t count=std::min(n-used,limit-bytes_.size());
            bytes_.insert(bytes_.end(),p+used,p+used+count);used+=count;
            if(!expected_ && bytes_.size()==header_size_) {
                const uint32_t magic=wire::get32(bytes_.data()),len=wire::get32(bytes_.data()+(reply_?20:28));
                const uint16_t version=uint16_t(bytes_[4])|uint16_t(bytes_[5])<<8;
                const uint16_t op=uint16_t(bytes_[6])|uint16_t(bytes_[7])<<8;
                const bool invalid_request=!reply_ && (wire::get32(bytes_.data()+16)!=RKVA_SESSION_NONE ||
                    wire::get32(bytes_.data()+20)!=0 || wire::get32(bytes_.data()+24)!=0);
                const auto response_status=static_cast<int32_t>(wire::get32(bytes_.data()+16));
                const bool invalid_response=reply_ && (response_status>RKVA_OK || response_status<RKVA_ERR_SNAPSHOT_IO);
                if(magic!=RKVA_MAGIC||version!=RKVA_VERSION_V2||(op!=RKVA_MANAGED_AUTH_HELLO&&op!=RKVA_MANAGED_ATTEST&&!(reply_&&op==RKVA_MANAGED_AUTH_CHALLENGE)&&!wire::valid_op(op))||
                   invalid_request||invalid_response||len<prefix_bytes+tag_bytes||len>max_frame) {failed_=true;return false;}
                expected_=header_size_+len;bytes_.reserve(expected_);
            }
        }return true;}catch(...){failed_=true;return false;}
    }
    bool complete() const {return !failed_&&expected_&&bytes_.size()==expected_;}
    bool finish() const {return complete();}
    bool failed() const {return failed_;}
    bool request(rkva_request & out) const {return !reply_&&complete()&&parse_request_header(bytes_.data(),32,out);}
    bool response(rkva_response & out) const {return reply_&&complete()&&parse_response_header(bytes_.data(),56,out);}
    const uint8_t * payload() const {return complete()?bytes_.data()+header_size_:nullptr;}
    size_t payload_size() const {return complete()?bytes_.size()-header_size_:0;}
    const std::vector<uint8_t> & data() const {return bytes_;}
private:
    bool reply_,failed_=false;size_t header_size_,expected_=0;std::vector<uint8_t> bytes_;
};
} // namespace rkva_managed_framing_v3
