#pragma once
// Offline evidence checker only. Not wired to host/server/Xbox or GC. A
// matching result is never permission to release: caller must authenticate a
// fresh challenge and hold one authority across checks and subsequent actions.
#include "server-rkva-host-journal.h"
#include "../../ggml/include/rkva-managed-registry.h"
#include "../../ggml/src/ggml-remote-attn/rkva-managed-auth.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace rkva_reconcile_gate {
using challenge = std::array<uint8_t, 32>;
struct signed_view {
    std::string owner;
    challenge request_challenge{};
    rkva_managed_registry::snapshot peer;
    rkva_snapshot::Digest mac{};
};
enum class verdict { matched, unavailable, unauthenticated, ambiguous, rollback };
// Fixed-size names, bounded count, and sorted records give one MAC encoding.
inline bool hex64(const std::string & s) {
    return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string::npos;
}
inline bool mac_for(const rkva_managed_auth::secret & key, const signed_view & view, rkva_snapshot::Digest & out) {
    if (!hex64(view.owner) || rkva_registry_wire::all_zero(key) ||
        rkva_registry_wire::all_zero(view.request_challenge) || view.peer.transactions.size() > 50000) return false;
    std::vector<rkva_managed_registry::transaction> entries = view.peer.transactions;
    std::sort(entries.begin(), entries.end(), [](const auto & a, const auto & b) { return a.id < b.id; });
    std::set<std::string> ids, refs;
    static constexpr char domain[] = "RKVA-PEER-STATE-PROOF-V1";
    std::vector<uint8_t> bytes(domain, domain + sizeof(domain)-1);
    bytes.insert(bytes.end(), view.owner.begin(), view.owner.end());
    bytes.insert(bytes.end(), view.request_challenge.begin(), view.request_challenge.end());
    std::array<uint8_t, 12> numbers{};
    rkva_registry_wire::put64(numbers.data(), view.peer.version);
    rkva_registry_wire::put32(numbers.data()+8, uint32_t(entries.size()));
    bytes.insert(bytes.end(), numbers.begin(), numbers.end());
    for (const auto & r : entries) {
        if (!hex64(r.id) || !hex64(r.reference) || !hex64(r.key) ||
            !ids.insert(r.id).second || !refs.insert(r.reference).second) return false;
        uint8_t state = 0;
        switch (r.state) {
        case rkva_managed_registry::phase::pending: state=1; break;
        case rkva_managed_registry::phase::committed: state=2; break;
        case rkva_managed_registry::phase::aborted: state=3; break;
        case rkva_managed_registry::phase::released: state=4; break;
        default: return false;
        }
        if ((state==2 || state==4) ? (!r.committed_version || r.committed_version>view.peer.version) : r.committed_version!=0) return false;
        bytes.insert(bytes.end(), r.id.begin(), r.id.end()); bytes.insert(bytes.end(), r.reference.begin(), r.reference.end());
        bytes.insert(bytes.end(), r.key.begin(), r.key.end()); bytes.push_back(state);
        std::array<uint8_t,8> version{}; rkva_registry_wire::put64(version.data(), r.committed_version);
        bytes.insert(bytes.end(),version.begin(),version.end());
    }
    return rkva_managed_auth::hmac_sha256(key.data(),key.size(),bytes.data(),bytes.size(),out);
}
// Only the peer/authenticated transport may create the proof. In this offline
// experiment it is invoked on a real registry snapshot by the test fixture.
inline bool seal(const rkva_managed_auth::secret & key, signed_view & view) {
    rkva_snapshot::Digest tag{}; if (!mac_for(key,view,tag)) return false; view.mac=tag; return true;
}
inline verdict evaluate(const rkva_host_journal::snapshot * host, const signed_view * view,
                        const std::string & owner, const challenge & expected,
                        const rkva_managed_auth::secret & key) {
    if (!host || !view) return verdict::unavailable;
    try {
        if (!hex64(owner) || rkva_registry_wire::all_zero(expected) || view->owner != owner ||
            !rkva_managed_auth::equal_tag(expected.data(),view->request_challenge.data())) return verdict::unauthenticated;
        rkva_snapshot::Digest mac{};
        if (!mac_for(key,*view,mac) || !rkva_managed_auth::equal_tag(mac.data(),view->mac.data())) return verdict::unauthenticated;
        std::map<std::string, const rkva_managed_registry::transaction *> peer;
        for (const auto & r : view->peer.transactions) peer.emplace(r.id,&r);
        std::set<std::string> seen;
        for (const auto & h : host->transactions) {
            if (!hex64(h.tx) || !hex64(h.reference) || !hex64(h.key) || !seen.insert(h.tx).second) return verdict::ambiguous;
            using H = rkva_host_journal::phase; using P = rkva_managed_registry::phase;
            // Any unacknowledged phase has ambiguous cross-device outcome.
            if (h.state != H::commit_ack && h.state != H::release_ack) return verdict::ambiguous;
            if (!h.peer_commit_version) return verdict::ambiguous;
            if (view->peer.version < h.peer_commit_version) return verdict::rollback;
            auto it=peer.find(h.tx);
            if (it==peer.end()) return verdict::rollback;
            const auto & p=*it->second;
            if (p.reference!=h.reference || p.key!=h.key || p.committed_version!=h.peer_commit_version) return verdict::ambiguous;
            if ((h.state==H::commit_ack && p.state!=P::committed) ||
                (h.state==H::release_ack && p.state!=P::released)) return verdict::ambiguous;
        }
        if (seen.size()!=peer.size()) return verdict::ambiguous; // unknown owner pins remain protected
        return verdict::matched;
    } catch (...) { return verdict::unavailable; }
}
} // namespace rkva_reconcile_gate
