#pragma once
#include "server-rkva-host-journal.h"
namespace rkva_host_journal {
// Only authenticated negative ACKs are supported. Success/receipt/anchor
// update and HELLO remain unavailable until an independently verified v3
// receipt/object proof and the final atomic framing contract exist. This is
// the Phase 4 host side of the single v3 framing contract: the reply guard
// receives the fully authenticated request/reply frames and consumes the
// previously issued v3 nonce in one journal publication.
rkva_managed_framing_v3::status accept_v3_error(journal &, const rkva_managed_framing_v3::secret &,
    rkva_managed_framing_v3::session &, v3_connection_authority *, const std::vector<uint8_t> & exact_request,
    const rkva_response &, const uint8_t * reply, size_t bytes, std::string & error);
} // namespace rkva_host_journal
