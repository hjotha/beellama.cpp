#include "server-rkva-session-authority.h"

#include <algorithm>
#include <cstring>
#include <utility>

#if defined(__linux__)
#include <cerrno>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace rkva_session_authority {
namespace v3 = rkva_managed_framing_v3;

struct authority::impl {
    int fd = -1;
    std::uint64_t cookie = 0;
    std::string peer;
    v3::nonce issued_challenge{};
    std::recursive_mutex * mutex = nullptr;
    std::mutex meta;
    std::unique_lock<std::recursive_mutex> held;
    v3::session bound{};
    bool has_bound = false;
    bool live = true;
    bool has_identity = false;
#if defined(__linux__)
    pid_t owner = 0;
    dev_t dev = 0;
    ino_t ino = 0;
#endif
};

namespace {

bool same_session(const v3::session & a, const v3::session & b) {
    return a.authenticated && b.authenticated && a.owner == b.owner &&
           a.ledger_epoch == b.ledger_epoch && a.transcript == b.transcript &&
           a.peer_challenge == b.peer_challenge && a.auth_epoch == b.auth_epoch;
}

} // namespace

authority::authority() : data_(std::make_shared<impl>()) {}

authority::~authority() { invalidate(); }

bool authority::available() {
    return rkva_managed_activation::enabled();
}

std::shared_ptr<authority> authority::mint(const binding & value, std::string & error) {
    error.clear();
    if (!available()) { error = "managed session authority unavailable (activation gate closed)"; return {}; }
    if (value.accepted_fd < 0) { error = "accepted connection required"; return {}; }
    if (!value.state_mutex) { error = "session state mutex required"; return {}; }
    if (!value.connection_cookie) { error = "connection cookie required"; return {}; }
    if (value.peer.empty()) { error = "peer identity required"; return {}; }
    if (v3::wire::all_zero(value.challenge)) { error = "issued challenge required"; return {}; }
#if defined(__linux__)
    try {
        struct stat state {};
        if (::fstat(value.accepted_fd, &state) || !S_ISSOCK(state.st_mode)) {
            error = "accepted connection is not a socket";
            return {};
        }
        sockaddr_storage peer {};
        socklen_t length = sizeof(peer);
        if (::getpeername(value.accepted_fd, reinterpret_cast<sockaddr *>(&peer), &length) != 0 ||
            length == 0) {
            error = "accepted connection has no connected peer";
            return {};
        }
        auto out = std::shared_ptr<authority>(new authority());
        auto & d = *out->data_;
        d.fd = value.accepted_fd;
        d.cookie = value.connection_cookie;
        d.peer = value.peer;
        d.issued_challenge = value.challenge;
        d.mutex = value.state_mutex;
        d.owner = ::getpid();
        d.dev = state.st_dev;
        d.ino = state.st_ino;
        d.has_identity = true;
        return out;
    } catch (const std::exception & e) {
        error = e.what();
        return {};
    }
#else
    (void) value;
    error = "managed session authority unsupported on this platform";
    return {};
#endif
}

bool authority::bind_session(const v3::session & session, std::string & error) {
    error.clear();
    std::lock_guard<std::mutex> guard(data_->meta);
    if (!data_->live) { error = "managed session authority invalidated"; return false; }
#if defined(__linux__)
    if (data_->owner != ::getpid()) { error = "managed session authority forked"; return false; }
#endif
    if (!data_->has_identity) { error = "managed session authority has no connection"; return false; }
    if (!session.authenticated) { error = "authenticated session required"; return false; }
    if (v3::wire::all_zero(session.transcript) || v3::wire::all_zero(session.peer_challenge)) {
        error = "session transcript and challenge required";
        return false;
    }
    data_->bound = session;
    data_->has_bound = true;
    return true;
}

bool authority::held_and_bound(const v3::session & session) {
    std::lock_guard<std::mutex> guard(data_->meta);
    if (!data_->live || !data_->has_bound || !data_->has_identity) return false;
#if defined(__linux__)
    if (data_->owner != ::getpid()) return false;
    struct stat state {};
    if (::fstat(data_->fd, &state) || !S_ISSOCK(state.st_mode) ||
        state.st_dev != data_->dev || state.st_ino != data_->ino) {
        return false;
    }
    sockaddr_storage peer {};
    socklen_t length = sizeof(peer);
    if (::getpeername(data_->fd, reinterpret_cast<sockaddr *>(&peer), &length) != 0) return false;
#endif
    if (!data_->held.owns_lock()) {
        std::unique_lock<std::recursive_mutex> next(*data_->mutex, std::try_to_lock);
        if (!next.owns_lock()) return false;
        data_->held = std::move(next);
    }
    if (!same_session(session, data_->bound)) {
        if (data_->held.owns_lock()) data_->held.unlock();
        return false;
    }
    return true;
}

void authority::release() {
    std::lock_guard<std::mutex> guard(data_->meta);
    if (data_->held.owns_lock()) data_->held.unlock();
}

void authority::invalidate() {
    std::lock_guard<std::mutex> guard(data_->meta);
    if (data_->held.owns_lock()) data_->held.unlock();
    data_->live = false;
    data_->has_bound = false;
    data_->has_identity = false;
}

bool authority::live() const {
    std::lock_guard<std::mutex> guard(data_->meta);
#if defined(__linux__)
    if (data_->owner != ::getpid()) return false;
#endif
    return data_->live;
}

} // namespace rkva_session_authority
