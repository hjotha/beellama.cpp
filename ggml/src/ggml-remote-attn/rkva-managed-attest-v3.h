#pragma once
// Experimental ATTEST framing only. A real listener guard must capture the
// engine under its live connection/mutex and atomically admit nonce+anchor.
#include "rkva-managed-framing-v3.h"
namespace rkva_managed_attest_v3 {
namespace f=rkva_managed_framing_v3;
namespace a=rkva_resident_attestation;
using f::id;using f::nonce;using f::secret;using f::status;
constexpr size_t request_bytes=48, response_fixed_bytes=204;
struct request {uint32_t session_id=0,rows=0;uint64_t generation=0;id key{};};
struct evidence {request requested{};uint64_t submitted=0,completed=0;id mac{},coverage{};
    uint64_t version=0;id state_digest{},next_challenge{};std::vector<uint8_t> metadata;};
struct peer_guard {
    virtual ~peer_guard()=default;
    virtual f::auth::guard_result admit(const f::session &,const f::anchor &,const request &,const nonce &,evidence &)=0;
};
struct host_guard {
    virtual ~host_guard()=default;
    virtual f::auth::guard_result commit(const f::session &,const request &,const nonce &,const evidence &,
        const rkva_request &,const std::vector<uint8_t> &,const rkva_response &,const uint8_t *,size_t)=0;
};
inline bool encode_request(const request & q,std::vector<uint8_t> & b) {
    if(q.session_id==RKVA_SESSION_NONE||!q.rows||!q.generation||!f::nz(q.key))return false;
    b.assign(request_bytes,0);f::wire::put32(b.data(),q.session_id);f::wire::put32(b.data()+4,q.rows);
    f::wire::put64(b.data()+8,q.generation);f::wire::put_id(b.data()+16,q.key);return true;
}
inline bool decode_request(const std::vector<uint8_t> & b,request & q) {
    if(b.size()!=request_bytes)return false;
    q.session_id=f::wire::get32(b.data());q.rows=f::wire::get32(b.data()+4);q.generation=f::wire::get64(b.data()+8);q.key=f::wire::get_id(b.data()+16);
    std::vector<uint8_t> check;return encode_request(q,check)&&check==b;
}
inline bool verify(const secret & key,const f::session & s,const evidence & e) {
    if(!f::valid_session(s)||!e.version||!f::nz(e.state_digest)||!f::nz(e.next_challenge)||e.next_challenge==s.peer_challenge||
       !e.submitted||e.submitted==UINT64_MAX||e.completed==UINT64_MAX||e.completed<e.submitted||e.metadata.empty()||
       e.metadata.size()>f::snap::kMaxManifestBytes)return false;
    f::snap::Manifest m;std::string error;std::vector<uint8_t> canonical;
    if(!f::snap::decode(e.metadata,m,&error)||!f::snap::encode(m,canonical,&error)||canonical!=e.metadata||
       m.live_rows!=e.requested.rows||m.generation!=e.requested.generation||
       !rkva_managed_auth::equal_tag(m.host_state_sha256,e.requested.key.data()))return false;
    a::input in;in.session_id=e.requested.session_id;in.live_rows=m.live_rows;in.generation=m.generation;
    in.submitted_fence=e.submitted;in.completed_fence=e.completed;in.host_key=e.requested.key;
    in.geometry=m.geometry;in.descriptor=m.descriptor;in.layers=m.layers;
    for(const auto & l:m.layers){in.counts.push_back(m.live_rows);in.record_groups.push_back(l.record_groups);in.ready.push_back(l.ready!=0);}
    f::auth::session context{s.owner,s.auth_epoch,s.transcript,true};a::evidence captured;
    return a::make_metadata(in,context,key,captured,error)&&captured.canonical_metadata==e.metadata&&
       rkva_managed_auth::equal_tag(captured.mac.data(),e.mac.data())&&
       rkva_prepare_v3::attestation_digest(e.metadata,e.mac,e.requested.session_id,e.submitted,e.completed)==e.coverage;
}
inline bool encode_evidence(const secret & key,const f::session & s,const evidence & e,std::vector<uint8_t> & b) {
    if(!verify(key,s,e))return false;
    std::vector<uint8_t> q;if(!encode_request(e.requested,q))return false;
    b.assign(response_fixed_bytes+e.metadata.size(),0);std::copy(q.begin(),q.end(),b.begin());
    f::wire::put64(b.data()+48,e.submitted);f::wire::put64(b.data()+56,e.completed);f::wire::put_id(b.data()+64,e.mac);
    f::wire::put_id(b.data()+96,e.coverage);f::wire::put64(b.data()+128,e.version);f::wire::put_id(b.data()+136,e.state_digest);
    f::wire::put_id(b.data()+168,e.next_challenge);f::wire::put32(b.data()+200,uint32_t(e.metadata.size()));
    std::copy(e.metadata.begin(),e.metadata.end(),b.begin()+204);return true;
}
inline bool decode_evidence(const secret & key,const f::session & s,const std::vector<uint8_t> & b,evidence & e) {
    if(b.size()<response_fixed_bytes||b.size()>response_fixed_bytes+f::snap::kMaxManifestBytes||
       f::wire::get32(b.data()+200)!=b.size()-response_fixed_bytes)return false;
    evidence v;std::vector<uint8_t> q(b.begin(),b.begin()+request_bytes);if(!decode_request(q,v.requested))return false;
    v.submitted=f::wire::get64(b.data()+48);v.completed=f::wire::get64(b.data()+56);v.mac=f::wire::get_id(b.data()+64);
    v.coverage=f::wire::get_id(b.data()+96);v.version=f::wire::get64(b.data()+128);v.state_digest=f::wire::get_id(b.data()+136);
    v.next_challenge=f::wire::get_id(b.data()+168);v.metadata.assign(b.begin()+response_fixed_bytes,b.end());if(!verify(key,s,v))return false;e=std::move(v);return true;
}
inline status make_request(const secret & key,const f::session & s,const rkva_request & h,const f::anchor & anchor,
        const nonce & n,const request & q,f::auth::nonce_guard * guard,std::vector<uint8_t> & out) {
    try {std::vector<uint8_t> payload;if(!f::valid_session(s)||!f::valid_anchor(anchor,s)||!encode_request(q,payload)||
        !f::request_header(h,f::prefix_bytes+payload.size()+f::tag_bytes,f::kind::attest))return status::invalid;
        f::fields x;x.type=f::kind::attest;x.owner=s.owner;x.ledger_epoch=s.ledger_epoch;x.auth_epoch=s.auth_epoch;
        x.transcript=s.transcript;x.host_anchor=anchor;x.nonce=n;x.payload=std::move(payload);std::vector<uint8_t> frame;
        auto header=f::bytes(h);if(!f::seal(key,header.data(),header.size(),x,frame))return status::invalid;
        const auto r=f::convert(f::auth::claim(guard,s.owner,s.auth_epoch,f::auth::purpose::client_command,n));
        if(r!=status::accepted)return r;
        out=std::move(frame);return status::accepted;
    }catch(...){return status::unavailable;}
}
inline status accept_request(const secret & key,const f::session & s,const rkva_request & h,const uint8_t * p,size_t n,
        peer_guard * guard,request & out,evidence & proof) {
    try {f::fields x;request q;auto header=f::bytes(h);
        if(!f::valid_session(s)||!f::request_header(h,n,f::kind::attest)||!f::open(key,header.data(),header.size(),p,n,f::kind::attest,x)||
           x.owner!=s.owner||x.ledger_epoch!=s.ledger_epoch||x.auth_epoch!=s.auth_epoch||x.transcript!=s.transcript||
           f::nz(x.request_hash)||!f::valid_anchor(x.host_anchor,s)||!decode_request(x.payload,q))return status::invalid;
        if(!guard)return status::unavailable;
        evidence candidate;auto r=guard->admit(s,x.host_anchor,q,x.nonce,candidate);
        if(r==f::auth::guard_result::replay)return status::replay;
        if(r!=f::auth::guard_result::recorded||x.host_anchor.version==UINT64_MAX||candidate.version!=x.host_anchor.version+1||
           !verify(key,s,candidate)||candidate.requested.session_id!=q.session_id||
           candidate.requested.rows!=q.rows||candidate.requested.generation!=q.generation||candidate.requested.key!=q.key)return status::unavailable;
        out=q;proof=std::move(candidate);return status::accepted;
    }catch(...){return status::unavailable;}
}
inline status make_reply(const secret & key,f::session & s,const rkva_request & h,const std::vector<uint8_t> & command,
        const rkva_response & rh,const evidence & proof,std::vector<uint8_t> & out) {
    try {f::fields x;request q;auto header=f::bytes(h);std::vector<uint8_t> payload;
        if(!f::valid_session(s)||!f::request_header(h,command.size(),f::kind::attest)||
           !f::open(key,header.data(),header.size(),command.data(),command.size(),f::kind::attest,x)||
           x.owner!=s.owner||x.ledger_epoch!=s.ledger_epoch||x.auth_epoch!=s.auth_epoch||x.transcript!=s.transcript||
           f::nz(x.request_hash)||!f::valid_anchor(x.host_anchor,s)||!decode_request(x.payload,q)||
           q.session_id!=proof.requested.session_id||q.rows!=proof.requested.rows||q.generation!=proof.requested.generation||
           q.key!=proof.requested.key||x.host_anchor.version==UINT64_MAX||proof.version!=x.host_anchor.version+1||
           !encode_evidence(key,s,proof,payload)||
           !f::response_header(rh,f::prefix_bytes+payload.size()+f::tag_bytes,f::kind::attest_reply)||rh.cycle_id!=h.cycle_id)return status::invalid;
        f::fields r;r.type=f::kind::attest_reply;r.owner=s.owner;r.ledger_epoch=s.ledger_epoch;r.auth_epoch=s.auth_epoch;
        r.transcript=s.transcript;r.nonce=x.nonce;r.request_hash=f::request_digest(h,command);r.payload=std::move(payload);
        std::vector<uint8_t> frame;auto response_header=f::bytes(rh);
        if(!f::seal(key,response_header.data(),response_header.size(),r,frame))return status::invalid;
        out=std::move(frame);s.peer_challenge=proof.next_challenge;return status::accepted;
    }catch(...){return status::unavailable;}
}
inline status accept_reply(const secret & key,f::session & s,const rkva_request & h,const std::vector<uint8_t> & command,
        const rkva_response & rh,const uint8_t * p,size_t n,host_guard * guard,evidence & out) {
    try {f::fields x,r;request q;evidence proof;auto req_header=f::bytes(h);auto resp_header=f::bytes(rh);
        if(!f::valid_session(s)||!f::request_header(h,command.size(),f::kind::attest)||
           !f::open(key,req_header.data(),req_header.size(),command.data(),command.size(),f::kind::attest,x)||
           x.owner!=s.owner||x.ledger_epoch!=s.ledger_epoch||x.auth_epoch!=s.auth_epoch||x.transcript!=s.transcript||
           f::nz(x.request_hash)||!f::valid_anchor(x.host_anchor,s)||!decode_request(x.payload,q)||
           !f::response_header(rh,n,f::kind::attest_reply)||rh.cycle_id!=h.cycle_id||
           !f::open(key,resp_header.data(),resp_header.size(),p,n,f::kind::attest_reply,r)||r.owner!=s.owner||
           r.ledger_epoch!=s.ledger_epoch||r.auth_epoch!=s.auth_epoch||r.transcript!=s.transcript||r.nonce!=x.nonce||
           r.request_hash!=f::request_digest(h,command)||!f::empty_anchor(r.host_anchor)||!decode_evidence(key,s,r.payload,proof)||
           x.host_anchor.version==UINT64_MAX||proof.version!=x.host_anchor.version+1||
           proof.requested.session_id!=q.session_id||proof.requested.rows!=q.rows||proof.requested.generation!=q.generation||proof.requested.key!=q.key)return status::invalid;
        if(!guard)return status::unavailable;
        const auto admitted=guard->commit(s,q,x.nonce,proof,h,command,rh,p,n);
        if(admitted==f::auth::guard_result::replay)return status::replay;
        if(admitted!=f::auth::guard_result::recorded)return status::unavailable;
        s.peer_challenge=proof.next_challenge;out=std::move(proof);return status::accepted;
    }catch(...){return status::unavailable;}
}
} // namespace rkva_managed_attest_v3
