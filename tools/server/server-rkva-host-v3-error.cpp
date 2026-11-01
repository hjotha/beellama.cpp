#include "server-rkva-host-v3-error.h"
namespace rkva_host_journal {
namespace v3=rkva_managed_framing_v3;
namespace auth=rkva_managed_auth_e2e;
namespace wire=rkva_registry_wire;
// The v3 reply contract hands the guard the exact request nonce and the
// authenticated negative ACK. The guard must not fire for a successful reply
// (no independent receipt/object proof exists) and must not fire twice.
class v3_error_guard final : public v3::reply_ack_guard {
public:
    v3_error_guard(journal & j,const v3::session & s,v3_connection_authority * c,const auth_id & req,const auth_id & reply,std::string & e)
        :journal_(j),session_(s),connection_(c),request_(req),response_(reply),error_(e) {}
    auth::guard_result commit(const v3::session & s,uint16_t,const wire::request &,const v3::nonce & nonce,bool ok,
            const v3::acknowledgment &,const rkva_request &,const std::vector<uint8_t> &,
            const rkva_response &,const uint8_t *,size_t) override {
        if(called_||ok||s.owner!=session_.owner||s.auth_epoch!=session_.auth_epoch||
           s.ledger_epoch!=session_.ledger_epoch||!connection_||!connection_->held_and_bound(session_))
            return auth::guard_result::unavailable;
        called_=true;
        return journal_.consume_v3_error(nonce,request_,response_,rkva_registry_wire::to_hex(s.owner),
            s.auth_epoch,s.ledger_epoch,error_)?auth::guard_result::recorded:auth::guard_result::unavailable;
    }
private:
    journal & journal_;const v3::session & session_;v3_connection_authority * connection_;
    auth_id request_,response_;std::string & error_;bool called_=false;
};
v3::status accept_v3_error(journal & j,const v3::secret & key,v3::session & session,v3_connection_authority * connection,
        const std::vector<uint8_t> & exact_request,const rkva_response & response,const uint8_t * reply,size_t n,std::string & error) {
    error.clear();
    try {
        if(!connection||!connection->held_and_bound(session)||!reply||exact_request.size()<32||
           response.status>=0||n>v3::max_frame) {session.authenticated=false;error="v3 error context unavailable";return v3::status::unavailable;}
        rkva_request header{};
        if(!v3::parse_request_header(exact_request.data(),32,header)||header.payload_bytes!=exact_request.size()-32) {
            session.authenticated=false;error="v3 request framing invalid";return v3::status::invalid;
        }
        const auto request_hash=rkva_snapshot::digest(exact_request.data(),exact_request.size());
        auto response_header=v3::bytes(response);rkva_snapshot::Sha256 hash;
        hash.update(response_header.data(),response_header.size());hash.update(reply,n);const auto response_hash=hash.finish();
        std::vector<uint8_t> body(exact_request.begin()+32,exact_request.end());
        v3_error_guard guard(j,session,connection,request_hash,response_hash,error);
        v3::acknowledgment ignored;
        const auto result=v3::accept_reply(key,session,header,body,response,reply,n,&guard,ignored);
        if(result!=v3::status::peer_error) {
            session.authenticated=false;
            if(error.empty())error="v3 authenticated error was not committed";
        }
        return result;
    } catch(...) { session.authenticated=false;error="v3 error acceptance unavailable";return v3::status::unavailable; }
}
} // namespace rkva_host_journal
