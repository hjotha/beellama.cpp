#include "../tools/xbox/uwp/rkva/rkva-resident-attestation.h"
#undef NDEBUG
#include <cassert>
#include <iostream>
#include <set>
#include <tuple>
namespace a=rkva_resident_attestation;
namespace auth=rkva_managed_auth_e2e;
namespace wire=rkva_registry_wire;
struct fixture_guard : auth::nonce_guard {
    std::set<std::tuple<wire::id,uint64_t,auth::purpose,auth::nonce>> values;
    auth::guard_result claim(const wire::id& o,uint64_t e,auth::purpose p,const auth::nonce& n) override {
        return values.emplace(o,e,p,n).second?auth::guard_result::recorded:auth::guard_result::replay;
    }
    auth::guard_result consume(const wire::id& o,uint64_t e,auth::purpose from,auth::purpose to,const auth::nonce& n) override {
        if(!values.count({o,e,from,n}))return auth::guard_result::unavailable;
        return values.emplace(o,e,to,n).second?auth::guard_result::recorded:auth::guard_result::replay;
    }
};
int main() {
    auth::secret secret{};secret.fill(9);wire::id owner{};owner.fill(1);auth::nonce client{},peer{};client.fill(2);peer.fill(3);
    fixture_guard host_guard,peer_guard;a::authenticated_hello hello;
    hello.offered_flags=RKVA_PROTO_MANAGED_REGISTRY_V1|RKVA_PROTO_MANAGED_AUTH_V2;
    hello.negotiated.reserved[0]=RKVA_CAP_MANAGED_REGISTRY_V1|RKVA_CAP_MANAGED_AUTH_V2;
    hello.negotiated.supported_ops=wire::operations|(1u<<RKVA_MANAGED_AUTH_HELLO);hello.negotiated.server_version=1;
    hello.request.magic=RKVA_MAGIC;hello.request.version=RKVA_VERSION_V2;hello.request.op=RKVA_MANAGED_AUTH_HELLO;
    hello.request.cycle_id=5;hello.request.session_id=RKVA_SESSION_NONE;hello.request.payload_bytes=auth::offer_bytes;
    hello.response.magic=RKVA_MAGIC;hello.response.version=RKVA_VERSION_V2;hello.response.op=hello.request.op;
    hello.response.cycle_id=hello.request.cycle_id;hello.response.status=RKVA_OK;hello.response.payload_bytes=auth::hello_reply_bytes;
    assert(auth::make_offer(secret,owner,7,hello.offered_flags,hello.negotiated,hello.request,client,&host_guard,hello.offer)==auth::status::accepted);
    assert(auth::accept_offer(secret,owner,7,hello.offered_flags,hello.negotiated,hello.request,hello.response,hello.offer,peer,&peer_guard,hello.reply,hello.live)==auth::status::accepted);
    a::input in;in.session_id=9;in.live_rows=513;in.generation=0x700000001ull;in.submitted_fence=11;in.completed_fence=11;in.host_key.fill(4);
    in.geometry={8,2,256,2,4096,1};in.descriptor={1024,1024,RKVA_RANGE_STORAGE_KVARN4,3,128,128,256,2,4,4,8960,3,2,128,128,1,1,1};
    in.counts={513};in.record_groups={4};in.ready={true};in.layers={{385,384,4,1}};
    std::string error;a::evidence baseline;assert(a::make(in,hello,secret,baseline,error));
    assert(baseline.generation==in.generation && baseline.submitted_fence==11 && !baseline.canonical_metadata.empty());
    auto changed=in;changed.host_key[0]^=1;a::evidence other;assert(a::make(changed,hello,secret,other,error) && other.mac!=baseline.mac);
    changed=in;changed.layers[0].stage_begin=256;assert(a::make(changed,hello,secret,other,error) && other.mac!=baseline.mac);
    changed=in;changed.completed_fence=12;assert(a::make(changed,hello,secret,other,error) && other.mac!=baseline.mac);
    changed=in;changed.generation++;assert(a::make(changed,hello,secret,other,error) && other.mac!=baseline.mac);
    changed=in;changed.session_id++;assert(a::make(changed,hello,secret,other,error) && other.mac!=baseline.mac);
    changed=in;changed.counts[0]--;assert(!a::make(changed,hello,secret,other,error));
    changed=in;changed.ready[0]=false;assert(!a::make(changed,hello,secret,other,error));
    changed=in;changed.record_groups[0]--;assert(!a::make(changed,hello,secret,other,error));
    changed=in;changed.layers.clear();assert(!a::make(changed,hello,secret,other,error));
    changed=in;changed.completed_fence=10;assert(!a::make(changed,hello,secret,other,error));
    changed=in;changed.completed_fence=UINT64_MAX;assert(!a::make(changed,hello,secret,other,error));
    changed=in;changed.submitted_fence=0;assert(!a::make(changed,hello,secret,other,error));
    changed=in;changed.host_key={};assert(!a::make(changed,hello,secret,other,error));
    auto forged=hello;forged.live.transcript[0]^=1;assert(!a::make(in,forged,secret,other,error));
    forged=hello;forged.reply[1]^=1;assert(!a::make(in,forged,secret,other,error));
    forged=hello;forged.negotiated.supported_ops=0;assert(!a::make(in,forged,secret,other,error));
    auto badkey=secret;badkey[0]^=1;assert(!a::make(in,hello,badkey,other,error));
    auth::session host{};assert(auth::finish_hello(secret,owner,7,hello.offered_flags,hello.negotiated,hello.request,hello.response,hello.offer,hello.reply,&host_guard,host)==auth::status::accepted);
    assert(host.transcript==hello.live.transcript);
    std::cout<<"resident attestation metadata fixture passed; no GPU/lease proof\n";
}
