#pragma once
// Pure metadata encoder. The engine caller must hold the real session authority
// and prove an already-completed GPU fence; this helper cannot do either. The
// result authenticates metadata, not GPU bytes, a blob, a durable pin or GC.
#include "rkva-native-snapshot.h"
#include "../../../../ggml/src/ggml-remote-attn/rkva-managed-auth-e2e.h"
#include <array>
#include <string>
#include <vector>
namespace rkva_resident_attestation {
namespace snap = rkva_native_snapshot;
namespace auth = rkva_managed_auth_e2e;
struct input {
    uint32_t session_id = 0, live_rows = 0;
    uint64_t generation = 0, submitted_fence = 0, completed_fence = 0;
    rkva_snapshot::Digest host_key{};
    snap::Geometry geometry{};
    snap::Descriptor descriptor{};
    std::vector<uint32_t> counts, record_groups;
    std::vector<bool> ready;
    std::vector<snap::LayerCoverage> layers;
};
struct authenticated_hello {
    auth::session live{};
    uint32_t offered_flags = 0;
    rkva_hello_ack negotiated{};
    rkva_request request{};
    rkva_response response{};
    auth::offer_wire offer{};
    auth::hello_reply_wire reply{};
};
// Verifies the actual E2E offer and reply before accepting its transcript.
// The caller must also retain the accepted live session and durable nonce
// admission; an old signed exchange alone cannot establish liveness.
inline bool valid_hello(const authenticated_hello & h,const auth::secret & key) {
    if(!auth::valid_session(h.live) || !auth::hello_response_header(h.request,h.response) ||
       !auth::offer_fields(h.offer,h.live.owner,h.live.key_epoch,h.offered_flags,h.negotiated) ||
       !auth::offer_mac_ok(key,h.request,h.offer) ||
       !std::equal(h.offer.begin(),h.offer.end(),h.reply.begin())) return false;
    const auto client=rkva_registry_wire::get_id(h.offer.data()+68);
    const auto peer=rkva_registry_wire::get_id(h.reply.data()+auth::offer_bytes);
    if(rkva_registry_wire::all_zero(client)||rkva_registry_wire::all_zero(peer)||client==peer)return false;
    std::array<uint8_t,auth::offer_bytes+32> body{};
    std::copy_n(h.reply.data(),body.size(),body.data());auth::digest tag{};
    if(!auth::hello_reply_mac(key,h.request,h.response,body.data(),tag) ||
       !rkva_managed_auth::equal_tag(tag.data(),h.reply.data()+body.size())) return false;
    const auto derived=auth::make_session(h.request,h.response,h.offer,h.reply);
    return derived.owner==h.live.owner && derived.key_epoch==h.live.key_epoch &&
           rkva_managed_auth::equal_tag(derived.transcript.data(),h.live.transcript.data());
}
struct evidence {
    std::vector<uint8_t> canonical_metadata;
    rkva_snapshot::Digest mac{};
    uint64_t generation = 0, submitted_fence = 0, completed_fence = 0;
};
// This helper encodes metadata only. Its arguments do not establish session
// liveness or authorize a registry operation; the concrete guard must do so.
inline bool make_metadata(const input & in,const auth::session & session,const auth::secret & key,
                          evidence & output,std::string & error) {
    const auto reject=[&](const char * why) { error=why; return false; };
    if(!auth::valid_session(session) || rkva_registry_wire::all_zero(key) ||
       rkva_registry_wire::all_zero(in.host_key)) return reject("attestation identity unavailable");
    if(!in.generation || !in.submitted_fence || in.submitted_fence==UINT64_MAX ||
       in.completed_fence==UINT64_MAX || in.completed_fence<in.submitted_fence)
        return reject("attestation fence or generation unavailable");
    if(in.geometry.n_layers != in.layers.size() ||
       !snap::validate_resident_layers(in.descriptor,in.live_rows,in.counts,in.ready,
                                       in.record_groups,in.layers,&error))
        return reject("attestation resident coverage differs");
    snap::Manifest manifest;std::string why;
    if(!snap::build(in.geometry,in.descriptor,in.live_rows,in.generation,in.host_key.data(),in.layers,manifest,&why))
        return reject("attestation geometry rejected");
    std::vector<uint8_t> bytes;
    if(!snap::encode(manifest,bytes,&why) || bytes.size()>snap::kMaxManifestBytes)
        return reject("attestation metadata encoding failed");
    static constexpr char domain[]="RKVA-RESIDENT-COVERAGE-V1";
    std::vector<uint8_t> body;body.reserve(sizeof(domain)-1+32+32+32+8+4+8+8+4+bytes.size());
    body.insert(body.end(),domain,domain+sizeof(domain)-1);
    body.insert(body.end(),session.owner.begin(),session.owner.end());
    body.insert(body.end(),session.transcript.begin(),session.transcript.end());
    body.insert(body.end(),in.host_key.begin(),in.host_key.end());
    std::array<uint8_t,32> numbers{};
    rkva_registry_wire::put64(numbers.data(),session.key_epoch);
    rkva_registry_wire::put32(numbers.data()+8,in.session_id);
    rkva_registry_wire::put64(numbers.data()+12,in.submitted_fence);
    rkva_registry_wire::put64(numbers.data()+20,in.completed_fence);
    rkva_registry_wire::put32(numbers.data()+28,uint32_t(bytes.size()));
    body.insert(body.end(),numbers.begin(),numbers.end());body.insert(body.end(),bytes.begin(),bytes.end());
    evidence result; if(!rkva_managed_auth::hmac_sha256(key.data(),key.size(),body.data(),body.size(),result.mac))
        return reject("attestation MAC failed");
    result.canonical_metadata=std::move(bytes);result.generation=in.generation;
    result.submitted_fence=in.submitted_fence;result.completed_fence=in.completed_fence;
    output=std::move(result);error.clear();return true;
}
inline bool make(const input & in,const authenticated_hello & hello,const auth::secret & key,
                 evidence & output,std::string & error) {
    if(!valid_hello(hello,key)) {error="attestation hello invalid";return false;}
    return make_metadata(in,hello.live,key,output,error);
}
} // namespace rkva_resident_attestation
