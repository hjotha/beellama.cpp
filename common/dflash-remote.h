#pragma once

#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <arpa/inet.h>
#include <cerrno>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace dflash_remote {

// Must match xllama/uwp/dflash-rpc/dedicated_protocol.h, version 2.
constexpr uint32_t magic = 0x324c4644;
constexpr uint16_t version = 2;
enum op : uint16_t { hello = 1, reset = 2, sync = 3, draft = 4, trim = 5, mock = 6, sync_trim = 7 };

#pragma pack(push, 1)
struct request {
    uint32_t magic;
    uint16_t version;
    uint16_t op;
    uint64_t cycle_id;
    uint32_t n_tokens;
    uint32_t payload_bytes;
};
struct response {
    uint32_t magic;
    uint16_t version;
    uint16_t op;
    uint64_t cycle_id;
    int32_t status;
    uint32_t payload_bytes;
    uint64_t xbox_receive_us;
    uint64_t xbox_prepare_us;
    uint64_t xbox_compute_us;
    uint64_t xbox_response_us;
};
struct hello_data {
    uint32_t n_ctx;
    uint32_t n_embd_enc;
    uint32_t n_embd_dec;
    uint32_t selector_top_k;
    int32_t mask_token_id;
    uint32_t cache_bits_k;
    uint32_t cache_bits_v;
    uint64_t model_bytes;
};
struct draft_data { int32_t pos0; int32_t id_last; int32_t n_max; };
struct trim_data  { int32_t pos0; };
// Optional v2 extension: an empty call probes support without changing KV.
// Otherwise followed by the usual SYNC rows. reset=1 subsumes the suffix trim.
struct sync_trim_data { int32_t pos0; uint32_t reset; };
#pragma pack(pop)

static_assert(sizeof(request) == 24 && sizeof(response) == 56 && sizeof(hello_data) == 36,
              "DFlash RPC v2 wire layout changed");
static_assert(sizeof(sync_trim_data) == 8, "DFlash fused SYNC layout changed");

class server_error : public std::runtime_error {
public:
    const int32_t status;
    server_error(int32_t status, op operation)
        : std::runtime_error("DFlash RPC server error " + std::to_string(status) +
                             " on op " + std::to_string(operation)), status(status) {}
};

struct timing {
    uint64_t header_send_us = 0;
    uint64_t payload_send_us = 0;
    uint64_t header_wait_us = 0;
    uint64_t payload_read_us = 0;
    uint64_t send_us = 0;
    uint64_t roundtrip_us = 0;
    response server = {};
};

class client {
#ifndef _WIN32
    int fd = -1;

    static void write_all(int fd, const void * src, size_t len, int flags = 0) {
        const auto * p = static_cast<const uint8_t *>(src);
        while (len) {
            const ssize_t n = ::send(fd, p, len, MSG_NOSIGNAL | flags);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("DFlash RPC send failed");
            p += n;
            len -= (size_t) n;
        }
    }

    static void read_all(int fd, void * dst, size_t len) {
        auto * p = static_cast<uint8_t *>(dst);
        while (len) {
            const ssize_t n = ::recv(fd, p, len, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("DFlash RPC receive failed");
            p += n;
            len -= (size_t) n;
        }
    }
#endif

public:
    client(const std::string & host, uint16_t port) {
#ifndef _WIN32
        fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) throw std::runtime_error("DFlash RPC socket failed");
        const int one = 1;
        if (::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0) {
            ::close(fd);
            fd = -1;
            throw std::runtime_error("DFlash RPC TCP_NODELAY failed");
        }
        timeval timeout{15, 0};
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(port);
        if (::inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1 ||
                ::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
            ::close(fd);
            fd = -1;
            throw std::runtime_error("DFlash RPC connect failed: " + host + ":" + std::to_string(port));
        }
#else
        (void) host;
        (void) port;
        throw std::runtime_error("DFlash RPC client currently requires a POSIX host");
#endif
    }

    client(const client &) = delete;
    client & operator=(const client &) = delete;
    ~client() {
#ifndef _WIN32
        if (fd >= 0) ::close(fd);
#endif
    }

    timing call(op operation, uint64_t cycle_id, uint32_t n_tokens,
                const void * payload, uint32_t payload_bytes,
                std::vector<uint8_t> & output, uint32_t expected_bytes) {
#ifndef _WIN32
        const request req{magic, version, operation, cycle_id, n_tokens, payload_bytes};
        const auto start = std::chrono::steady_clock::now();
        // Keep the tiny header with its payload, without copying large SYNC rows.
        // TCP_NODELAY also prevents delayed-ACK stalls on the remaining tail.
        int header_flags = 0;
#ifdef MSG_MORE
        if (payload_bytes) header_flags = MSG_MORE;
#endif
        write_all(fd, &req, sizeof(req), header_flags);
        const auto header_sent = std::chrono::steady_clock::now();
        if (payload_bytes) write_all(fd, payload, payload_bytes);
        const auto sent = std::chrono::steady_clock::now();
        response rsp{};
        read_all(fd, &rsp, sizeof(rsp));
        const auto header_received = std::chrono::steady_clock::now();
        if (rsp.magic != magic || rsp.version != version || rsp.op != operation ||
                rsp.cycle_id != cycle_id || rsp.payload_bytes != expected_bytes) {
            throw std::runtime_error("DFlash RPC response header mismatch");
        }
        if (rsp.status != 0) {
            throw server_error(rsp.status, operation);
        }
        output.resize(expected_bytes);
        if (expected_bytes) read_all(fd, output.data(), expected_bytes);
        const auto end = std::chrono::steady_clock::now();
        const auto us = [](auto a, auto b) -> uint64_t {
            return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
        };
        return {us(start, header_sent), us(header_sent, sent),
                us(sent, header_received), us(header_received, end),
                us(start, sent), us(start, end), rsp};
#else
        (void) operation; (void) cycle_id; (void) n_tokens; (void) payload;
        (void) payload_bytes; (void) output; (void) expected_bytes;
        throw std::runtime_error("DFlash RPC client currently requires a POSIX host");
#endif
    }
};

} // namespace dflash_remote
