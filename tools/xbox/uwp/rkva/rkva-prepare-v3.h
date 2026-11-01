#pragma once
// Experimental PREPARE metadata frame. No runtime capability/GC or v2 changes.
// Attestation authenticates resident metadata, not GPU bytes or durable blobs.
#include "rkva-resident-attestation.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
namespace rkva_prepare_v3 {
namespace auth=rkva_managed_auth_e2e;
namespace attest=rkva_resident_attestation;
namespace snap=rkva_native_snapshot;
using id=rkva_registry_wire::id;
constexpr uint32_t revision=3;
constexpr size_t hello_bytes=84, prefix_bytes=376, tag_bytes=32;
constexpr size_t max_metadata=snap::kMaxManifestBytes;
struct metadata {
    id owner{},tx{},reference{},key{},coverage_digest{},identity_digest{},nonce{},transcript{};
    uint64_t epoch=0,generation=0;
    rkva_hello hello{};
    snap::Geometry geometry{};snap::Descriptor descriptor{};uint32_t rows=0;
    std::vector<snap::LayerCoverage> coverage;
    std::vector<uint8_t> canonical_attestation;
};
enum class result { accepted, invalid, replay, unavailable };
enum class persist_result { committed, replay, unavailable };
// MUST hold the real peer engine/store authority, independently revalidate the
// live resident attestation, HELLO and host RKVK, verify the host anchor,
// atomically consume this nonce and persist quota/pending pin + exact metadata.
// The same HMAC key is held by the host, so nested MACs alone do not prove
// peer-resident coverage. Return committed only after durable COMMIT; fail
// closed otherwise. No production authority/SQLite implementation is supplied.
struct prepare_guard { virtual ~prepare_guard()=default; virtual persist_result persist(const metadata & m)=0; };
inline void put_hello(const rkva_hello & h,uint8_t * p) {
    const uint32_t v[]={h.proto_flags,h.n_layer_remote,h.n_head,h.n_head_kv,h.head_dim,h.max_ctx,h.cache_bits_k,h.cache_bits_v,
        h.group_tokens,h.sinkhorn_iters,h.tail_tokens,h.tail_groups,h.tail_type,h.domain,h.has_sinks,h.swa};
    for(size_t i=0;i<16;i++)rkva_registry_wire::put32(p+4*i,v[i]);
    uint32_t f=0;std::memcpy(&f,&h.kq_scale,4);
    rkva_registry_wire::put32(p+64,f);rkva_registry_wire::put64(p+68,h.reserved[0]);rkva_registry_wire::put64(p+76,h.reserved[1]);
}
inline rkva_hello get_hello(const uint8_t * p) {
    rkva_hello h{};uint32_t v[16];for(size_t i=0;i<16;i++)v[i]=rkva_registry_wire::get32(p+4*i);
    h.proto_flags=v[0];h.n_layer_remote=v[1];h.n_head=v[2];h.n_head_kv=v[3];h.head_dim=v[4];h.max_ctx=v[5];
    h.cache_bits_k=v[6];h.cache_bits_v=v[7];h.group_tokens=v[8];h.sinkhorn_iters=v[9];h.tail_tokens=v[10];
    h.tail_groups=v[11];h.tail_type=v[12];h.domain=v[13];h.has_sinks=v[14];h.swa=v[15];
    const uint32_t f=rkva_registry_wire::get32(p+64);std::memcpy(&h.kq_scale,&f,4);
    h.reserved[0]=rkva_registry_wire::get64(p+68);h.reserved[1]=rkva_registry_wire::get64(p+76);return h;
}
inline bool valid_hello(const rkva_hello & h,const snap::Manifest & m,uint32_t flags) {
    return std::numeric_limits<float>::is_iec559 && sizeof(float)==4 && h.proto_flags==flags &&
        h.n_layer_remote==m.geometry.n_layers && h.n_head==m.geometry.n_head && h.n_head_kv==m.geometry.n_head_kv &&
        h.head_dim==m.geometry.head_dim && h.max_ctx==m.geometry.max_ctx && h.head_dim==256 &&
        m.geometry.head_slices==h.head_dim/128 && (h.group_tokens==0 || h.group_tokens==128) &&
        (h.cache_bits_k==0 || h.cache_bits_k==16) && (h.cache_bits_v==0 || h.cache_bits_v==16) &&
        h.sinkhorn_iters>0 && h.tail_type<=1 && h.domain<=2 &&
        !h.has_sinks && !h.swa && !h.reserved[0] && !h.reserved[1] && std::isfinite(h.kq_scale) && h.kq_scale>0;
}
inline bool header(const rkva_request & h,size_t n) {
    return n<=UINT32_MAX && h.magic==RKVA_MAGIC && h.version==RKVA_VERSION && h.op==RKVA_REGISTRY_PREPARE &&
        h.session_id==RKVA_SESSION_NONE && !h.layer_id && !h.n_tokens && h.payload_bytes==n;
}
inline bool nonzero(const id & v){return !rkva_registry_wire::all_zero(v);}
inline id attestation_digest(const std::vector<uint8_t> & meta,const id & mac,uint32_t session_id,uint64_t submitted,uint64_t completed) {
    static constexpr char domain[]="RKVA-COVERAGE-ATTESTATION-DIGEST-V1";rkva_snapshot::Sha256 h;h.update(domain,sizeof(domain)-1);
    std::array<uint8_t,20> nums{};rkva_registry_wire::put32(nums.data(),session_id);rkva_registry_wire::put64(nums.data()+4,submitted);
    rkva_registry_wire::put64(nums.data()+12,completed);h.update(nums.data(),nums.size());h.update(meta.data(),meta.size());h.update(mac.data(),mac.size());return h.finish();
}
inline bool check_attestation(const attest::authenticated_hello & session,const auth::secret & key,
                              uint32_t sid,uint64_t submitted,uint64_t completed,const id & expected_key,
                              const std::vector<uint8_t> & canonical,const id & att_mac,snap::Manifest & m) {
    std::string error;std::vector<uint8_t> encoded;
    if(!snap::decode(canonical,m,&error) || !snap::encode(m,encoded,&error) || encoded!=canonical ||
       !rkva_managed_auth::equal_tag(m.host_state_sha256,expected_key.data()))return false;
    attest::input in;in.session_id=sid;in.live_rows=m.live_rows;in.generation=m.generation;
    in.submitted_fence=submitted;in.completed_fence=completed;in.host_key=expected_key;
    in.geometry=m.geometry;in.descriptor=m.descriptor;in.layers=m.layers;
    for(const auto & l:m.layers){in.counts.push_back(m.live_rows);in.record_groups.push_back(l.record_groups);in.ready.push_back(l.ready!=0);}
    attest::evidence proof;
    return attest::make(in,session,key,proof,error) && proof.canonical_metadata==canonical &&
           rkva_managed_auth::equal_tag(proof.mac.data(),att_mac.data());
}
inline id identity_digest(const uint8_t * prefix,const uint8_t * metadata,size_t n) {
    static constexpr char domain[]="RKVA-PREPARE-V3-METADATA-V1";rkva_snapshot::Sha256 sha;sha.update(domain,sizeof(domain)-1);
    // Stable across reconnect: owner/tx/ref/key + semantic HELLO fields +
    // canonical metadata (including coverage). Negotiation flags can change
    // without changing snapshot identity; frame HMAC still binds the complete
    // HELLO, transcript, attestation MAC, fences, session ID and nonce.
    sha.update(prefix+8,128);sha.update(prefix+140,80);sha.update(metadata,n);return sha.finish();
}
inline bool make(const auth::secret & secret,const attest::authenticated_hello & session,const rkva_request & h,
                 const id & tx,const id & ref,const id & object,const rkva_hello & hello,uint32_t sid,
                 const attest::evidence & proof,const id & nonce,std::vector<uint8_t> & out) {
    try {
        if(!attest::valid_hello(session,secret)||!nonzero(tx)||!nonzero(ref)||!nonzero(object)||!nonzero(nonce)||
           proof.canonical_metadata.empty()||proof.canonical_metadata.size()>max_metadata||
           !header(h,prefix_bytes+proof.canonical_metadata.size()+tag_bytes))return false;
        snap::Manifest manifest;
        if(!check_attestation(session,secret,sid,proof.submitted_fence,proof.completed_fence,object,proof.canonical_metadata,proof.mac,manifest)||
           proof.generation!=manifest.generation||!valid_hello(hello,manifest,session.offered_flags))return false;
        std::vector<uint8_t> bytes(prefix_bytes+proof.canonical_metadata.size()+tag_bytes,0);auto * p=bytes.data();
        rkva_registry_wire::put32(p,revision);rkva_registry_wire::put32(p+4,uint32_t(proof.canonical_metadata.size()));
        std::copy(session.live.owner.begin(),session.live.owner.end(),p+8);std::copy(tx.begin(),tx.end(),p+40);
        std::copy(ref.begin(),ref.end(),p+72);std::copy(object.begin(),object.end(),p+104);put_hello(hello,p+136);
        rkva_registry_wire::put32(p+220,sid);rkva_registry_wire::put64(p+224,proof.submitted_fence);
        rkva_registry_wire::put64(p+232,proof.completed_fence);std::copy(proof.mac.begin(),proof.mac.end(),p+240);
        const auto coverage=attestation_digest(proof.canonical_metadata,proof.mac,sid,proof.submitted_fence,proof.completed_fence);
        std::copy(coverage.begin(),coverage.end(),p+272);rkva_registry_wire::put64(p+304,session.live.key_epoch);
        std::copy(session.live.transcript.begin(),session.live.transcript.end(),p+312);std::copy(nonce.begin(),nonce.end(),p+344);
        std::copy(proof.canonical_metadata.begin(),proof.canonical_metadata.end(),p+prefix_bytes);
        static constexpr char domain[]="RKVA-PREPARE-V3-FRAME-V1";std::vector<uint8_t> data;
        data.reserve(sizeof(domain)-1+32+bytes.size()-tag_bytes);data.insert(data.end(),domain,domain+sizeof(domain)-1);
        std::array<uint8_t,32> outer{};auth::req_header_bytes(h,outer.data());data.insert(data.end(),outer.begin(),outer.end());
        data.insert(data.end(),bytes.begin(),bytes.end()-tag_bytes);id mac{};
        if(!rkva_managed_auth::hmac_sha256(secret.data(),secret.size(),data.data(),data.size(),mac))return false;
        std::copy(mac.begin(),mac.end(),bytes.end()-tag_bytes);out=std::move(bytes);return true;
    } catch(...) {return false;}
}
inline result accept(const auth::secret & secret,const attest::authenticated_hello & session,const rkva_hello & expected_hello,
                     const rkva_request & h,const uint8_t * data,size_t n,prepare_guard * guard,metadata & output) {
    try {
        if(!data || n<prefix_bytes+tag_bytes || n>prefix_bytes+max_metadata+tag_bytes || !header(h,n) ||
           rkva_registry_wire::get32(data)!=revision || rkva_registry_wire::get32(data+4)!=n-prefix_bytes-tag_bytes ||
           !attest::valid_hello(session,secret))return result::invalid;
        std::array<uint8_t,hello_bytes> hello_wire{};put_hello(expected_hello,hello_wire.data());
        if(!std::equal(hello_wire.begin(),hello_wire.end(),data+136))return result::invalid;
        metadata m;m.owner=rkva_registry_wire::get_id(data+8);m.tx=rkva_registry_wire::get_id(data+40);
        m.reference=rkva_registry_wire::get_id(data+72);m.key=rkva_registry_wire::get_id(data+104);
        m.epoch=rkva_registry_wire::get64(data+304);m.transcript=rkva_registry_wire::get_id(data+312);m.nonce=rkva_registry_wire::get_id(data+344);
        if(m.owner!=session.live.owner||m.epoch!=session.live.key_epoch||
           !rkva_managed_auth::equal_tag(m.transcript.data(),session.live.transcript.data())||
           !nonzero(m.tx)||!nonzero(m.reference)||!nonzero(m.key)||!nonzero(m.nonce))return result::invalid;
        static constexpr char domain[]="RKVA-PREPARE-V3-FRAME-V1";std::vector<uint8_t> body;
        body.reserve(sizeof(domain)-1+32+n-tag_bytes);body.insert(body.end(),domain,domain+sizeof(domain)-1);
        std::array<uint8_t,32> outer{};auth::req_header_bytes(h,outer.data());body.insert(body.end(),outer.begin(),outer.end());
        body.insert(body.end(),data,data+n-tag_bytes);id tag{};
        if(!rkva_managed_auth::hmac_sha256(secret.data(),secret.size(),body.data(),body.size(),tag)||
           !rkva_managed_auth::equal_tag(tag.data(),data+n-tag_bytes))return result::invalid;
        const auto sid=rkva_registry_wire::get32(data+220);const auto submitted=rkva_registry_wire::get64(data+224),completed=rkva_registry_wire::get64(data+232);
        const id att_mac=rkva_registry_wire::get_id(data+240);m.coverage_digest=rkva_registry_wire::get_id(data+272);
        m.canonical_attestation.assign(data+prefix_bytes,data+n-tag_bytes);snap::Manifest parsed;
        if(attestation_digest(m.canonical_attestation,att_mac,sid,submitted,completed)!=m.coverage_digest ||
           !check_attestation(session,secret,sid,submitted,completed,m.key,m.canonical_attestation,att_mac,parsed))return result::invalid;
        m.hello=get_hello(data+136);if(!valid_hello(m.hello,parsed,session.offered_flags))return result::invalid;
        m.geometry=parsed.geometry;m.descriptor=parsed.descriptor;m.rows=parsed.live_rows;m.generation=parsed.generation;m.coverage=parsed.layers;
        m.identity_digest=identity_digest(data,data+prefix_bytes,m.canonical_attestation.size());
        if(!guard)return result::unavailable;
        const auto committed=guard->persist(m);
        if(committed==persist_result::replay)return result::replay;
        if(committed!=persist_result::committed)return result::unavailable;
        output=std::move(m);return result::accepted;
    }catch(...){return result::unavailable;}
}
} // namespace rkva_prepare_v3
