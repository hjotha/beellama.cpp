#pragma once
// Experimental protocol-only construction. NO runtime listener/client uses or
// advertises this. Guards and secure randomness/provisioning are mandatory;
// no production implementation is supplied. HMAC does not encrypt payloads.
//
// PHASE 4 CONTRACT NOTE: the managed vocabulary below (secret/nonce/digest,
// session, nonce_guard, purpose, req/resp header bytes, mac) is shared by the
// v3 framing and the resident-attestation metadata encoder. The E2E v2
// offer/hello-reply/command/reply functions in this header are the *older*
// two-frame shape; they are offline-only in this tree and are NOT the
// negotiated contract. `rkva_registry_wire::negotiated` plus
// `rkva-managed-framing-v3.h` are the only Phase 4 gate/framing. Nothing may
// dispatch an op on the strength of `rkva_managed_auth_e2e::negotiated` alone,
// because that predicate does not require RKVA_PROTO/CAP_MANAGED_FRAMING_V3.
#include "rkva-managed-auth.h"
#include <array>
#include <cstring>
#include <string>

namespace rkva_managed_auth_e2e {
using rkva_managed_auth::secret;
using rkva_managed_auth::nonce;
using rkva_managed_auth::digest;
using rkva_registry_wire::id;
using rkva_registry_wire::request;
using rkva_registry_wire::response;
constexpr uint32_t revision = 2;
constexpr size_t offer_body_bytes = 100;
constexpr size_t offer_bytes = offer_body_bytes + 32;
constexpr size_t hello_reply_bytes = offer_bytes + 32 + 32;
constexpr size_t command_body_bytes = rkva_registry_wire::request_bytes + 8 + 32 + 32;
constexpr size_t command_bytes = command_body_bytes + 32;
constexpr size_t reply_body_bytes = rkva_registry_wire::response_bytes;
constexpr size_t reply_bytes = reply_body_bytes + 32;
using offer_wire = std::array<uint8_t, offer_bytes>;
using hello_reply_wire = std::array<uint8_t, hello_reply_bytes>;
using command_wire = std::array<uint8_t, command_bytes>;
using reply_wire = std::array<uint8_t, reply_bytes>;

enum class status { accepted, peer_error, invalid, replay, unavailable };
enum class purpose : uint8_t { client_hello, server_hello, client_hello_reply, client_command, server_command, client_command_reply };
enum class guard_result { recorded, replay, unavailable };
// Implementations must atomically and durably claim each tuple across processes
// and restarts, bind it to the provisioned key epoch, and fail closed on loss,
// rollback, corruption, full capacity, or uncertain persistence. Never prune
// while the key epoch remains usable. The interface does not implement this.
struct nonce_guard {
    virtual ~nonce_guard() = default;
    virtual guard_result claim(const id & owner, uint64_t epoch, purpose kind, const nonce & n) = 0;
    // Atomically require the issued record and persist its first consumption.
    // A missing/uncertain issued record is unavailable, never accepted.
    virtual guard_result consume(const id & owner, uint64_t epoch, purpose issued,
                                 purpose consumed, const nonce & n) = 0;
};
inline status claim(nonce_guard * guard,const id & owner,uint64_t epoch,purpose kind,const nonce & n) {
    if (!guard) return status::unavailable;
    try {
        const auto r=guard->claim(owner,epoch,kind,n);
        return r==guard_result::recorded ? status::accepted :
               r==guard_result::replay ? status::replay : status::unavailable;
    } catch (...) { return status::unavailable; }
}
inline status consume(nonce_guard * guard,const id & owner,uint64_t epoch,purpose issued,purpose consumed,const nonce & n) {
    if (!guard) return status::unavailable;
    try {
        const auto r=guard->consume(owner,epoch,issued,consumed,n);
        return r==guard_result::recorded ? status::accepted :
               r==guard_result::replay ? status::replay : status::unavailable;
    } catch (...) { return status::unavailable; }
}
// Strict managed-auth-only handshake: stripping any flag or op results in
// failure, never a fallback to an unauthenticated managed operation.
inline bool negotiated(uint16_t protocol, uint32_t offered, const rkva_hello_ack & ack) {
    const uint32_t offers=RKVA_PROTO_MANAGED_REGISTRY_V1|RKVA_PROTO_MANAGED_AUTH_V2;
    const uint64_t caps=RKVA_CAP_MANAGED_REGISTRY_V1|RKVA_CAP_MANAGED_AUTH_V2;
    const uint32_t ops=rkva_registry_wire::operations|(1u<<RKVA_MANAGED_AUTH_HELLO);
    return protocol==RKVA_VERSION && (offered&offers)==offers && (ack.reserved[0]&caps)==caps &&
           (ack.supported_ops&ops)==ops;
}
struct session {
    id owner{};
    uint64_t key_epoch=0;
    digest transcript{};
    bool authenticated=false;
};
inline bool valid_session(const session & s) {
    return s.authenticated && s.key_epoch && !rkva_registry_wire::all_zero(s.owner) &&
           !rkva_registry_wire::all_zero(s.transcript);
}
// Domain-separated HMAC over exact canonical bytes, no NUL terminator.
template<size_t N>
inline bool mac(const secret & key,const char * domain,size_t domain_bytes,const std::array<uint8_t,N> & data,digest & out) {
    if (rkva_registry_wire::all_zero(key) || domain_bytes>64) return false;
    std::array<uint8_t,64+N> buffer{};
    std::memcpy(buffer.data(),domain,domain_bytes);
    std::copy(data.begin(),data.end(),buffer.begin()+domain_bytes);
    return rkva_managed_auth::hmac_sha256(key.data(),key.size(),buffer.data(),domain_bytes+N,out);
}
inline void req_header_bytes(const rkva_request & h,uint8_t * p) {
    rkva_registry_wire::put32(p,h.magic); rkva_managed_auth::put16(p+4,h.version); rkva_managed_auth::put16(p+6,h.op);
    rkva_registry_wire::put64(p+8,h.cycle_id); rkva_registry_wire::put32(p+16,h.session_id);
    rkva_registry_wire::put32(p+20,h.layer_id); rkva_registry_wire::put32(p+24,h.n_tokens); rkva_registry_wire::put32(p+28,h.payload_bytes);
}
inline void resp_header_bytes(const rkva_response & h,uint8_t * p) {
    rkva_registry_wire::put32(p,h.magic); rkva_managed_auth::put16(p+4,h.version); rkva_managed_auth::put16(p+6,h.op);
    rkva_registry_wire::put64(p+8,h.cycle_id); rkva_registry_wire::put32(p+16,uint32_t(h.status));
    rkva_registry_wire::put32(p+20,h.payload_bytes); rkva_registry_wire::put64(p+24,h.xbox_receive_us);
    rkva_registry_wire::put64(p+32,h.xbox_prepare_us); rkva_registry_wire::put64(p+40,h.xbox_compute_us);
    rkva_registry_wire::put64(p+48,h.xbox_response_us);
}
inline bool hello_request_header(const rkva_request & h) {
    return h.magic==RKVA_MAGIC && h.version==RKVA_VERSION && h.op==RKVA_MANAGED_AUTH_HELLO &&
           h.session_id==RKVA_SESSION_NONE && !h.layer_id && !h.n_tokens && h.payload_bytes==offer_bytes;
}
inline bool hello_response_header(const rkva_request & q,const rkva_response & r) {
    return hello_request_header(q) && r.magic==RKVA_MAGIC && r.version==q.version && r.op==q.op &&
           r.cycle_id==q.cycle_id && r.status==RKVA_OK && r.payload_bytes==hello_reply_bytes;
}
// Safe transport framing: exact-size copy only; authentication follows.
template<size_t N>
inline bool copy_wire(const uint8_t * data,size_t length,std::array<uint8_t,N> & out) {
    if (!data || length!=N) return false;
    std::array<uint8_t,N> tmp{}; std::copy_n(data,N,tmp.data()); out=tmp; return true;
}
// Caller supplies a fresh CSPRNG challenge and a durable local guard. No
// default keys, automatic random source, negotiation fallback or credentials.
inline status make_offer(const secret & key,const id & owner,uint64_t epoch,uint32_t flags,
                         const rkva_hello_ack & ack,const rkva_request & h,const nonce & client,nonce_guard * local,offer_wire & out) {
    if (!hello_request_header(h) || !negotiated(RKVA_VERSION,flags,ack) || !epoch || rkva_registry_wire::all_zero(owner) ||
        rkva_registry_wire::all_zero(client) || rkva_registry_wire::all_zero(key)) return status::invalid;
    std::array<uint8_t,offer_body_bytes> body{};
    rkva_registry_wire::put32(body.data(),revision); rkva_registry_wire::put32(body.data()+4,RKVA_VERSION);
    rkva_registry_wire::put_id(body.data()+8,owner); rkva_registry_wire::put64(body.data()+40,epoch);
    rkva_registry_wire::put32(body.data()+48,flags); rkva_registry_wire::put64(body.data()+52,ack.reserved[0]);
    rkva_registry_wire::put32(body.data()+60,ack.supported_ops); rkva_registry_wire::put32(body.data()+64,ack.server_version);
    std::copy(client.begin(),client.end(),body.begin()+68);
    std::array<uint8_t,32+offer_body_bytes> auth{}; req_header_bytes(h,auth.data());
    std::copy(body.begin(),body.end(),auth.begin()+32);
    static constexpr char domain[]="RKVA-AUTH-V2-HELLO-CLIENT"; digest tag{};
    if (!mac(key,domain,sizeof(domain)-1,auth,tag)) return status::invalid;
    const auto admitted=claim(local,owner,epoch,purpose::client_hello,client);
    if (admitted!=status::accepted) return admitted;
    offer_wire temp{}; std::copy(body.begin(),body.end(),temp.begin()); std::copy(tag.begin(),tag.end(),temp.begin()+offer_body_bytes);
    out=temp; return status::accepted;
}
inline bool offer_fields(const offer_wire & offer,const id & owner,uint64_t epoch,uint32_t flags,const rkva_hello_ack & ack) {
    const uint8_t * p=offer.data();
    return negotiated(RKVA_VERSION,flags,ack) && rkva_registry_wire::get32(p)==revision &&
        rkva_registry_wire::get32(p+4)==RKVA_VERSION && rkva_registry_wire::get_id(p+8)==owner &&
        rkva_registry_wire::get64(p+40)==epoch && epoch && !rkva_registry_wire::all_zero(owner) &&
        rkva_registry_wire::get32(p+48)==flags && rkva_registry_wire::get64(p+52)==ack.reserved[0] &&
        rkva_registry_wire::get32(p+60)==ack.supported_ops && rkva_registry_wire::get32(p+64)==ack.server_version;
}
inline bool offer_mac_ok(const secret & key,const rkva_request & h,const offer_wire & offer) {
    if (!hello_request_header(h)) return false;
    std::array<uint8_t,32+offer_body_bytes> body{}; req_header_bytes(h,body.data());
    std::copy_n(offer.data(),offer_body_bytes,body.data()+32);
    static constexpr char domain[]="RKVA-AUTH-V2-HELLO-CLIENT"; digest tag{};
    return mac(key,domain,sizeof(domain)-1,body,tag) && rkva_managed_auth::equal_tag(tag.data(),offer.data()+offer_body_bytes);
}
inline session make_session(const rkva_request & h,const rkva_response & r,const offer_wire & offer,const hello_reply_wire & reply) {
    std::array<uint8_t,32> req{}; std::array<uint8_t,56> resp{};
    req_header_bytes(h,req.data()); resp_header_bytes(r,resp.data());
    rkva_snapshot::Sha256 hash; hash.update(req.data(),req.size()); hash.update(offer.data(),offer.size());
    hash.update(resp.data(),resp.size()); hash.update(reply.data(),reply.size());
    session s; s.owner=rkva_registry_wire::get_id(offer.data()+8); s.key_epoch=rkva_registry_wire::get64(offer.data()+40);
    s.transcript=hash.finish(); s.authenticated=true; return s;
}
inline bool hello_reply_mac(const secret & key,const rkva_request & h,const rkva_response & r,
                            const uint8_t * reply_body,digest & tag) {
    if (!hello_response_header(h,r) || !reply_body) return false;
    std::array<uint8_t,32+56+offer_bytes+32> auth{};
    req_header_bytes(h,auth.data()); resp_header_bytes(r,auth.data()+32);
    std::copy_n(reply_body,offer_bytes+32,auth.data()+32+56);
    static constexpr char domain[]="RKVA-AUTH-V2-HELLO-SERVER";
    return mac(key,domain,sizeof(domain)-1,auth,tag);
}
inline status accept_offer(const secret & key,const id & owner,uint64_t epoch,uint32_t flags,const rkva_hello_ack & ack,
                           const rkva_request & h,const rkva_response & r,const offer_wire & offer,const nonce & peer,nonce_guard * remote,hello_reply_wire & out,session & server) {
    if (!hello_response_header(h,r) || !offer_fields(offer,owner,epoch,flags,ack) || !offer_mac_ok(key,h,offer) || rkva_registry_wire::all_zero(peer)) return status::invalid;
    nonce client=rkva_registry_wire::get_id(offer.data()+68);
    if (rkva_registry_wire::all_zero(client) || client==peer) return status::invalid;
    std::array<uint8_t,offer_bytes+32> body{}; std::copy(offer.begin(),offer.end(),body.begin());
    std::copy(peer.begin(),peer.end(),body.begin()+offer_bytes);
    digest tag{};
    if (!hello_reply_mac(key,h,r,body.data(),tag)) return status::invalid;
    const auto admitted=claim(remote,owner,epoch,purpose::server_hello,client);
    if (admitted!=status::accepted) return admitted;
    hello_reply_wire temp{}; std::copy(body.begin(),body.end(),temp.begin()); std::copy(tag.begin(),tag.end(),temp.begin()+body.size());
    out=temp; server=make_session(h,r,offer,temp); return status::accepted;
}
inline status finish_hello(const secret & key,const id & owner,uint64_t epoch,uint32_t flags,const rkva_hello_ack & ack,
                           const rkva_request & h,const rkva_response & r,const offer_wire & offer,const hello_reply_wire & reply,nonce_guard * local,session & client) {
    if (!hello_response_header(h,r) || !offer_fields(offer,owner,epoch,flags,ack) || !offer_mac_ok(key,h,offer) ||
        !std::equal(offer.begin(),offer.end(),reply.begin())) return status::invalid;
    const nonce client_nonce=rkva_registry_wire::get_id(offer.data()+68);
    const nonce peer=rkva_registry_wire::get_id(reply.data()+offer_bytes);
    if (rkva_registry_wire::all_zero(client_nonce) || rkva_registry_wire::all_zero(peer) || peer==client_nonce) return status::invalid;
    std::array<uint8_t,offer_bytes+32> body{}; std::copy_n(reply.data(),body.size(),body.data());
    digest tag{};
    if (!hello_reply_mac(key,h,r,body.data(),tag) || !rkva_managed_auth::equal_tag(tag.data(),reply.data()+body.size())) return status::invalid;
    const auto admitted=consume(local,owner,epoch,purpose::client_hello,purpose::client_hello_reply,client_nonce);
    if (admitted!=status::accepted) return admitted;
    client=make_session(h,r,offer,reply); return status::accepted;
}
inline bool command_header(const rkva_request & h) {
    return h.magic==RKVA_MAGIC && h.version==RKVA_VERSION && rkva_registry_wire::valid_op(h.op) &&
           h.session_id==RKVA_SESSION_NONE && !h.layer_id && !h.n_tokens && h.payload_bytes==command_bytes;
}
inline bool command_mac(const secret & key,const rkva_request & h,const uint8_t * body,digest & out) {
    if (!command_header(h) || !body) return false;
    std::array<uint8_t,32+command_body_bytes> data{}; req_header_bytes(h,data.data());
    std::copy_n(body,command_body_bytes,data.data()+32);
    static constexpr char domain[]="RKVA-AUTH-V2-COMMAND"; return mac(key,domain,sizeof(domain)-1,data,out);
}
inline bool inspect_command(const secret & key,const session & s,const rkva_request & h,const command_wire & wire,request & q,nonce & n) {
    if (!valid_session(s) || !command_header(h) || !rkva_registry_wire::decode_request(h.op,wire.data(),rkva_registry_wire::request_bytes,q) ||
        q.owner!=s.owner || rkva_registry_wire::get64(wire.data()+144)!=s.key_epoch ||
        !rkva_managed_auth::equal_tag(wire.data()+152,s.transcript.data())) return false;
    n=rkva_registry_wire::get_id(wire.data()+184); digest tag{};
    return !rkva_registry_wire::all_zero(n) && command_mac(key,h,wire.data(),tag) &&
           rkva_managed_auth::equal_tag(tag.data(),wire.data()+command_body_bytes);
}
inline status make_command(const secret & key,const session & s,const rkva_request & h,const request & q,
                           const nonce & n,nonce_guard * local,command_wire & out) {
    rkva_registry_wire::request_wire body{};
    if (!valid_session(s) || !command_header(h) || q.owner!=s.owner || rkva_registry_wire::all_zero(n) ||
        !rkva_registry_wire::encode_request(h.op,q,body)) return status::invalid;
    command_wire tmp{}; std::copy(body.begin(),body.end(),tmp.begin()); rkva_registry_wire::put64(tmp.data()+144,s.key_epoch);
    std::copy(s.transcript.begin(),s.transcript.end(),tmp.begin()+152); std::copy(n.begin(),n.end(),tmp.begin()+184);
    digest tag{}; if (!command_mac(key,h,tmp.data(),tag)) return status::invalid;
    const auto admitted=claim(local,s.owner,s.key_epoch,purpose::client_command,n); if(admitted!=status::accepted) return admitted;
    std::copy(tag.begin(),tag.end(),tmp.begin()+command_body_bytes); out=tmp; return status::accepted;
}
inline status accept_command(const secret & key,const session & s,const rkva_request & h,const command_wire & wire,
                             nonce_guard * remote,request & out) {
    request parsed{}; nonce n{}; if(!inspect_command(key,s,h,wire,parsed,n)) return status::invalid;
    const auto admitted=claim(remote,s.owner,s.key_epoch,purpose::server_command,n);
    if(admitted!=status::accepted) return admitted;
    out=parsed; return status::accepted;
}
inline bool response_header(const rkva_request & q,const rkva_response & r) {
    return command_header(q) && r.magic==RKVA_MAGIC && r.version==RKVA_VERSION && r.op==q.op &&
           r.cycle_id==q.cycle_id && r.status<=0 && r.payload_bytes==reply_bytes;
}
inline bool response_mac(const secret & key,const rkva_request & h,const command_wire & command,
                         const rkva_response & r,const uint8_t * body,digest & tag) {
    if(!response_header(h,r) || !body) return false;
    std::array<uint8_t,32+command_bytes+56+reply_body_bytes> data{};
    req_header_bytes(h,data.data()); std::copy(command.begin(),command.end(),data.begin()+32);
    resp_header_bytes(r,data.data()+32+command_bytes);
    std::copy_n(body,reply_body_bytes,data.data()+32+command_bytes+56);
    static constexpr char domain[]="RKVA-AUTH-V2-RESPONSE"; return mac(key,domain,sizeof(domain)-1,data,tag);
}
// Error replies use an all-zero canonical body; status remains HMAC-bound.
inline bool make_reply(const secret & key,const session & s,const rkva_request & h,const command_wire & command,
                       const rkva_response & outer,const response * value,reply_wire & out) {
    request q{}; nonce n{}; if(!inspect_command(key,s,h,command,q,n) || !response_header(h,outer)) return false;
    rkva_registry_wire::response_wire body{};
    if(outer.status==RKVA_OK) { if(!value || !rkva_registry_wire::encode_response(h.op,q,*value,body)) return false; }
    else if(value) return false;
    digest tag{}; if(!response_mac(key,h,command,outer,body.data(),tag)) return false;
    reply_wire tmp{}; std::copy(body.begin(),body.end(),tmp.begin()); std::copy(tag.begin(),tag.end(),tmp.begin()+reply_body_bytes);
    out=tmp; return true;
}
inline status accept_reply(const secret & key,const session & s,const rkva_request & h,const command_wire & command,
                           const rkva_response & outer,const reply_wire & wire,nonce_guard * local,response & out) {
    request q{}; nonce n{}; digest tag{}; response parsed{};
    if(!inspect_command(key,s,h,command,q,n) || !response_header(h,outer) ||
       !response_mac(key,h,command,outer,wire.data(),tag) ||
       !rkva_managed_auth::equal_tag(tag.data(),wire.data()+reply_body_bytes)) return status::invalid;
    if(outer.status==RKVA_OK) {
        if(!rkva_registry_wire::decode_response(h.op,q,wire.data(),reply_body_bytes,parsed)) return status::invalid;
    } else if(!std::all_of(wire.begin(),wire.begin()+reply_body_bytes,[](uint8_t b){return b==0;})) return status::invalid;
    const auto admitted=consume(local,s.owner,s.key_epoch,purpose::client_command,purpose::client_command_reply,n);
    if(admitted!=status::accepted) return admitted;
    if(outer.status==RKVA_OK) { out=parsed; return status::accepted; }
    return status::peer_error;
}
} // namespace rkva_managed_auth_e2e
