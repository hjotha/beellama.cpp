// ggml-remote-attn backend implementation.
//
// Executes GGML_OP_REMOTE_ATTN by forwarding Q/K/V/pos over a persistent TCP
// connection to a remote KVarN attention server (RKVA protocol). All buffers
// are plain host memory; the graph scheduler moves data between CUDA and this
// backend using the same host-copy path as the offload_kqv CPU pinning.
//
// Only POSIX hosts are supported (the .57 station is Linux). On any other
// platform ggml_backend_remote_attn_init returns NULL and the feature is
// disabled, leaving the local baseline untouched.

#include "ggml-remote-attn.h"
#include "ggml-local-split.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "remote-attn-protocol.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
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

#define GGML_REMOTE_ATTN_NAME "RemoteAttn"

namespace {

// ---------------------------------------------------------------------------
// host f16 helpers (software, portable)
// ---------------------------------------------------------------------------

static inline uint16_t f32_to_f16_bits(float value) {
    return (uint16_t) ggml_fp32_to_fp16(value);
}

static inline float f16_bits_to_f32(uint16_t bits) {
    return ggml_fp16_to_fp32((ggml_fp16_t) bits);
}

// ---------------------------------------------------------------------------
// persistent socket
// ---------------------------------------------------------------------------

struct rkva_socket {
#ifndef _WIN32
    int fd = -1;
#endif

    bool open() const {
#ifndef _WIN32
        return fd >= 0;
#else
        return false;
#endif
    }

    void close() {
#ifndef _WIN32
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
#endif
    }

    bool write_all(const void * src, size_t len, int flags = 0) {
#ifndef _WIN32
        const auto * p = static_cast<const uint8_t *>(src);
        while (len) {
            const ssize_t n = ::send(fd, p, len, MSG_NOSIGNAL | flags);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n <= 0) {
                return false;
            }
            p += n;
            len -= (size_t) n;
        }
        return true;
#else
        (void) src; (void) len; (void) flags;
        return false;
#endif
    }

    bool read_all(void * dst, size_t len) {
#ifndef _WIN32
        auto * p = static_cast<uint8_t *>(dst);
        while (len) {
            const ssize_t n = ::recv(fd, p, len, 0);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n <= 0) {
                return false;
            }
            p += n;
            len -= (size_t) n;
        }
        return true;
#else
        (void) dst; (void) len;
        return false;
#endif
    }
};

// ---------------------------------------------------------------------------
// profiling counters
// ---------------------------------------------------------------------------

struct rkva_stats {
    uint64_t calls       = 0;
    uint64_t bytes_tx    = 0;
    uint64_t bytes_rx    = 0;
    uint64_t rpc_us_sum  = 0;
    uint64_t rpc_us_max  = 0;
    uint64_t xbox_compute_us_sum = 0;
    uint64_t xbox_prepare_us_sum = 0;
    std::vector<uint64_t> rpc_us;  // for p50/p95
};

// ---------------------------------------------------------------------------
// backend context
// ---------------------------------------------------------------------------

struct ggml_backend_remote_attn_context {
    std::string host;
    uint16_t    port = 0;

    std::mutex  mutex;                 // serializes the single in-flight call
    rkva_socket sock;
    bool        connected = false;
    bool        failed = false;
    uint64_t    cycle_id = 0;

    ggml_remote_attn_geometry geo {};
    bool        geo_set = false;
    rkva_hello_ack ack {};

    uint32_t    active_session = RKVA_SESSION_NONE;

    // reusable pinned-ish staging (grow-only; never freed per token)
    std::vector<uint8_t> payload_tx;
    std::vector<uint8_t> payload_rx;

    rkva_stats stats;
    std::string stats_json;

    // scratch for the last positions block (int32)
    std::vector<int32_t> pos_scratch;
    // reusable host staging for reading Q/K/V out of their backend buffer
    // (they may live in CUDA or be views of CUDA tensors co-located by the
    // scheduler, so we must read via ggml_backend_tensor_get, never ->data).
    std::vector<float> f32_scratch;
};

static uint64_t now_us() {
#ifndef _WIN32
    timeval tv {};
    gettimeofday(&tv, nullptr);
    return (uint64_t) tv.tv_sec * 1000000ull + (uint64_t) tv.tv_usec;
#else
    return 0;
#endif
}

// low-level request/response over the persistent socket
static bool rkva_transact(
        ggml_backend_remote_attn_context * ctx,
        rkva_op op, uint32_t session_id, uint32_t layer_id, uint32_t n_tokens,
        const void * payload, uint32_t payload_bytes,
        std::vector<uint8_t> & out, uint32_t expected_bytes) {

    const rkva_request req {
        RKVA_MAGIC, RKVA_VERSION, (uint16_t) op, ++ctx->cycle_id,
        session_id, layer_id, n_tokens, payload_bytes
    };

    const uint64_t t0 = now_us();
    int header_flags = 0;
#ifdef MSG_MORE
    if (payload_bytes) {
        header_flags = MSG_MORE;
    }
#endif
    if (!ctx->sock.write_all(&req, sizeof(req), header_flags)) {
        return false;
    }
    if (payload_bytes && !ctx->sock.write_all(payload, payload_bytes)) {
        return false;
    }

    rkva_response rsp {};
    if (!ctx->sock.read_all(&rsp, sizeof(rsp))) {
        return false;
    }
    if (rsp.magic != RKVA_MAGIC || rsp.version != RKVA_VERSION ||
        rsp.op != (uint16_t) op || rsp.cycle_id != req.cycle_id) {
        return false;
    }
    if (rsp.status != RKVA_OK) {
        GGML_LOG_ERROR("%s: server error status=%d op=%d layer=%u\n",
                       GGML_REMOTE_ATTN_NAME, rsp.status, (int) op, layer_id);
        return false;
    }
    if (expected_bytes && rsp.payload_bytes != expected_bytes) {
        return false;
    }
    out.resize(rsp.payload_bytes);
    if (rsp.payload_bytes && !ctx->sock.read_all(out.data(), rsp.payload_bytes)) {
        return false;
    }
    const uint64_t t1 = now_us();

    ctx->stats.calls++;
    ctx->stats.bytes_tx += sizeof(req) + payload_bytes;
    ctx->stats.bytes_rx += sizeof(rsp) + rsp.payload_bytes;
    const uint64_t rtt = t1 - t0;
    ctx->stats.rpc_us_sum += rtt;
    ctx->stats.rpc_us_max = std::max(ctx->stats.rpc_us_max, rtt);
    ctx->stats.xbox_compute_us_sum += rsp.xbox_compute_us;
    ctx->stats.xbox_prepare_us_sum += rsp.xbox_prepare_us;
    if (ctx->stats.rpc_us.size() < 1000000) {
        ctx->stats.rpc_us.push_back(rtt);
    }
    return true;
}

// ---------------------------------------------------------------------------
// REMOTE_ATTN execution
// ---------------------------------------------------------------------------

// Serialize Q/K/V to the chosen wire type and one RKVA attn call.
static bool rkva_run_attn(
        ggml_backend_remote_attn_context * ctx,
        ggml_tensor * node) {

    ggml_tensor * q   = node->src[0];   // [n_embd_head, n_head,    n_tokens] F32
    ggml_tensor * k   = node->src[1];   // [n_embd_head, n_head_kv, n_tokens] F32
    ggml_tensor * v   = node->src[2];   // [n_embd_head, n_head_kv, n_tokens] F32
    ggml_tensor * pos = node->src[3];   // [n_tokens] I32

    const int layer_id  = ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_LAYER_ID);
    const int n_head    = ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_N_HEAD);
    const int n_head_kv = ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_N_HEAD_KV);
    const int n_embd_head = ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_N_EMBD_HEAD);
    const int n_tokens  = (int) q->ne[2];

    if (ctx->active_session == RKVA_SESSION_NONE) {
        GGML_LOG_ERROR("%s: no active session for remote attention\n", GGML_REMOTE_ATTN_NAME);
        return false;
    }

    const bool wire_f16 = (ctx->ack.chosen_wire == RKVA_WIRE_F16);
    const size_t esz = wire_f16 ? 2 : 4;

    if (pos->type != GGML_TYPE_I32) {
        return false;  // defensive: pos contract is I32
    }

    const size_t q_elems = (size_t) n_tokens * n_head * n_embd_head;
    const size_t kv_elems = (size_t) n_tokens * n_head_kv * n_embd_head;
    const size_t payload_bytes = sizeof(rkva_attn_req)
        + sizeof(int32_t) * n_tokens
        + (q_elems + 2 * kv_elems) * esz;

    ctx->payload_tx.resize(payload_bytes);
    uint8_t * p = ctx->payload_tx.data();

    rkva_attn_req areq { (uint32_t) ctx->ack.chosen_wire, 0, 0, 0 };
    std::memcpy(p, &areq, sizeof(areq)); p += sizeof(areq);

    // Positions: read the first n_tokens i32 (M-RoPE dim 0 == scalar position).
    // IMPORTANT: read every source through ggml_backend_tensor_get, never via
    // ->data: the scheduler may co-locate a view/reshape of a CUDA tensor in
    // this (host) backend's split, so ->data can be a device pointer. tensor_get
    // dispatches to the source buffer's backend (cudaMemcpy D2H when needed).
    ctx->pos_scratch.resize(n_tokens);
    ggml_backend_tensor_get(pos, ctx->pos_scratch.data(), 0, sizeof(int32_t) * n_tokens);
    std::memcpy(p, ctx->pos_scratch.data(), sizeof(int32_t) * n_tokens);
    p += sizeof(int32_t) * n_tokens;

    // Q/K/V: ggml tensors are contiguous [n_embd_head, heads, n_tokens] whose
    // linear order equals the wire [n_tokens][heads][n_embd_head] layout.
    if (!wire_f16) {
        ggml_backend_tensor_get(q, p, 0, q_elems * 4);  p += q_elems * 4;
        ggml_backend_tensor_get(k, p, 0, kv_elems * 4); p += kv_elems * 4;
        ggml_backend_tensor_get(v, p, 0, kv_elems * 4); p += kv_elems * 4;
    } else {
        ctx->f32_scratch.resize(std::max(q_elems, kv_elems));
        auto emit16 = [&](const ggml_tensor * t, size_t elems) {
            ggml_backend_tensor_get(t, ctx->f32_scratch.data(), 0, elems * 4);
            uint16_t * dst = (uint16_t *) p;
            for (size_t i = 0; i < elems; i++) {
                dst[i] = f32_to_f16_bits(ctx->f32_scratch[i]);
            }
            p += elems * 2;
        };
        emit16(q, q_elems);
        emit16(k, kv_elems);
        emit16(v, kv_elems);
    }

    const uint32_t out_elems = (uint32_t) q_elems;
    const uint32_t expected_bytes = out_elems * (uint32_t) esz;

    std::vector<uint8_t> out;
    const rkva_op op = (n_tokens == 1) ? RKVA_ATTN_DECODE : RKVA_ATTN_PREFILL;
    if (!rkva_transact(ctx, op, ctx->active_session, (uint32_t) layer_id,
                       (uint32_t) n_tokens, ctx->payload_tx.data(),
                       (uint32_t) payload_bytes, out, expected_bytes)) {
        return false;
    }

    // write output into node (host buffer), [n_embd_head*n_head, n_tokens] F32
    float * dst = (float *) node->data;
    if (wire_f16) {
        const uint16_t * s = (const uint16_t *) out.data();
        for (uint32_t i = 0; i < out_elems; i++) {
            dst[i] = f16_bits_to_f32(s[i]);
        }
    } else {
        std::memcpy(dst, out.data(), (size_t) out_elems * 4);
    }
    return true;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// buffer
// ---------------------------------------------------------------------------

namespace {

struct ggml_backend_remote_attn_buffer_context {
    void * base = nullptr;
    size_t size = 0;
};

static void ggml_backend_remote_attn_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_remote_attn_buffer_context *) buffer->context;
    if (ctx->base) {
        free(ctx->base);
    }
    delete ctx;
}

static void * ggml_backend_remote_attn_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_backend_remote_attn_buffer_context *) buffer->context)->base;
}

static void ggml_backend_remote_attn_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_remote_attn_buffer_context *) buffer->context;
    memset((char *) tensor->data + offset, value, size);
    (void) ctx;
}

static void ggml_backend_remote_attn_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
    (void) buffer;
}

static void ggml_backend_remote_attn_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    (void) buffer;
}

static bool ggml_backend_remote_attn_buffer_cpy_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    (void) buffer;
    return false;
}

static void ggml_backend_remote_attn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_remote_attn_buffer_context *) buffer->context;
    if (ctx->base) {
        memset(ctx->base, value, ctx->size);
    }
}

} // anonymous namespace

static ggml_backend_buffer_i ggml_backend_remote_attn_buffer_iface = {
    /* .free_buffer   = */ ggml_backend_remote_attn_buffer_free,
    /* .get_base      = */ ggml_backend_remote_attn_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_remote_attn_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_remote_attn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_remote_attn_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ggml_backend_remote_attn_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_remote_attn_buffer_clear,
    /* .reset         = */ nullptr,
};

// ---------------------------------------------------------------------------
// buffer type
// ---------------------------------------------------------------------------

namespace {

static const char * ggml_backend_remote_attn_buft_name(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return GGML_REMOTE_ATTN_NAME;
}

static ggml_backend_buffer_t ggml_backend_remote_attn_buft_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = new ggml_backend_remote_attn_buffer_context();
    ctx->size = size;
    if (size) {
        // 64-byte aligned host allocation
        void * ptr = nullptr;
        if (posix_memalign(&ptr, 64, size) != 0 || ptr == nullptr) {
            delete ctx;
            return nullptr;
        }
        memset(ptr, 0, size);
        ctx->base = ptr;
    }
    return ggml_backend_buffer_init(buft, ggml_backend_remote_attn_buffer_iface, ctx, size);
}

static size_t ggml_backend_remote_attn_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return 64;
}

static size_t ggml_backend_remote_attn_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return SIZE_MAX;
}

static size_t ggml_backend_remote_attn_buft_get_alloc_size(
        ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    (void) buft;
    return ggml_nbytes(tensor);
}

static bool ggml_backend_remote_attn_buft_is_host(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return true;
}

} // anonymous namespace

// buffer type + device singletons are stored in the backend context; we keep a
// single global buffer type / device for simplicity (one remote accelerator).
namespace {
ggml_backend_buffer_type ggml_backend_remote_attn_buft_global;
ggml_backend_device      ggml_backend_remote_attn_dev_global;
bool                     ggml_backend_remote_attn_globals_init = false;

ggml_backend_buffer_type_i ggml_backend_remote_attn_buft_iface = {
    /* .get_name      = */ ggml_backend_remote_attn_buft_name,
    /* .alloc_buffer  = */ ggml_backend_remote_attn_buft_alloc_buffer,
    /* .get_alignment = */ ggml_backend_remote_attn_buft_get_alignment,
    /* .get_max_size  = */ ggml_backend_remote_attn_buft_get_max_size,
    /* .get_alloc_size= */ ggml_backend_remote_attn_buft_get_alloc_size,
    /* .is_host       = */ ggml_backend_remote_attn_buft_is_host,
};
} // anonymous namespace

// ---------------------------------------------------------------------------
// backend
// ---------------------------------------------------------------------------

namespace {

static const char * ggml_backend_remote_attn_get_name(ggml_backend_t backend) {
    (void) backend;
    return GGML_REMOTE_ATTN_NAME;
}

static void ggml_backend_remote_attn_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ctx->sock.close();
        ctx->connected = false;
    }
    delete ctx;
    delete backend;
}

static void ggml_backend_remote_attn_set_tensor_async(
        ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
    (void) backend;
}

static void ggml_backend_remote_attn_get_tensor_async(
        ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    (void) backend;
}

static bool ggml_backend_remote_attn_cpy_tensor_async(
        ggml_backend_t backend_src, ggml_backend_t backend_dst,
        const ggml_tensor * src, ggml_tensor * dst) {
    (void) backend_src; (void) backend_dst; (void) src; (void) dst;
    return false;  // let the scheduler fall back to host copies
}

static void ggml_backend_remote_attn_synchronize(ggml_backend_t backend) {
    (void) backend;  // fully synchronous
}

static enum ggml_status ggml_backend_remote_attn_graph_compute(
        ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        // The scheduler may co-locate pure layout ops (views/reshapes of the
        // Q/K/V/pos inputs) in this backend's split. On a host buffer these are
        // no-ops: ggml-alloc already set their data pointers from the source.
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                continue;
            case GGML_OP_REMOTE_ATTN:
                break;
            default:
                // Anything else is a wiring bug: this backend only computes the
                // remote attention op. Fail loudly rather than silently skip.
                GGML_LOG_ERROR("%s: unexpected op %s in remote graph\n",
                               GGML_REMOTE_ATTN_NAME, ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
        }

        std::lock_guard<std::mutex> lock(ctx->mutex);
        if (ctx->failed || !ctx->connected) {
            GGML_LOG_ERROR("%s: backend not connected (failed=%d)\n",
                           GGML_REMOTE_ATTN_NAME, (int) ctx->failed);
            return GGML_STATUS_ABORTED;
        }
        if (!rkva_run_attn(ctx, node)) {
            ctx->failed = true;
            ctx->connected = false;
            ctx->sock.close();
            return GGML_STATUS_ABORTED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

} // anonymous namespace

static ggml_backend_i ggml_backend_remote_attn_iface = {
    /* .get_name          = */ ggml_backend_remote_attn_get_name,
    /* .free              = */ ggml_backend_remote_attn_free,
    /* .set_tensor_async  = */ ggml_backend_remote_attn_set_tensor_async,
    /* .get_tensor_async  = */ ggml_backend_remote_attn_get_tensor_async,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async  = */ ggml_backend_remote_attn_cpy_tensor_async,
    /* .synchronize       = */ ggml_backend_remote_attn_synchronize,
    /* .graph_plan_create = */ nullptr,
    /* .graph_plan_free   = */ nullptr,
    /* .graph_plan_update = */ nullptr,
    /* .graph_plan_compute= */ nullptr,
    /* .graph_compute     = */ ggml_backend_remote_attn_graph_compute,
    /* .event_record      = */ nullptr,
    /* .event_wait        = */ nullptr,
    /* .graph_optimize    = */ nullptr,
};

// ---------------------------------------------------------------------------
// device
// ---------------------------------------------------------------------------

namespace {

static const char * ggml_backend_remote_attn_dev_get_name(ggml_backend_dev_t dev) {
    (void) dev;
    return GGML_REMOTE_ATTN_NAME;
}

static const char * ggml_backend_remote_attn_dev_get_description(ggml_backend_dev_t dev) {
    (void) dev;
    return "Remote KVarN attention accelerator (RKVA)";
}

static void ggml_backend_remote_attn_dev_get_memory(
        ggml_backend_dev_t dev, size_t * free, size_t * total) {
    (void) dev;
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_remote_attn_dev_get_type(ggml_backend_dev_t dev) {
    (void) dev;
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_remote_attn_dev_get_props(
        ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name         = ggml_backend_remote_attn_dev_get_name(dev);
    props->description  = ggml_backend_remote_attn_dev_get_description(dev);
    props->type         = ggml_backend_remote_attn_dev_get_type(dev);
    props->memory_free  = 0;
    props->memory_total = 0;
    props->device_id    = nullptr;
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t ggml_backend_remote_attn_dev_init_backend(
        ggml_backend_dev_t dev, const char * params) {
    (void) dev; (void) params;
    // The backend is created explicitly via ggml_backend_remote_attn_init so
    // the host/port are known; dynamic device init is not used.
    return nullptr;
}

static ggml_backend_buffer_type_t ggml_backend_remote_attn_dev_get_buffer_type(ggml_backend_dev_t dev) {
    (void) dev;
    return &ggml_backend_remote_attn_buft_global;
}

static bool ggml_backend_remote_attn_dev_supports_op(
        ggml_backend_dev_t dev, const ggml_tensor * op) {
    (void) dev;
    switch (op->op) {
        case GGML_OP_REMOTE_ATTN:
            return true;
        // Pure layout ops the scheduler co-locates with the pinned remote node
        // (views/reshapes of Q/K/V/pos). No-ops on a host buffer.
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

static bool ggml_backend_remote_attn_dev_supports_buft(
        ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void) dev;
    return buft == &ggml_backend_remote_attn_buft_global;
}

static bool ggml_backend_remote_attn_dev_offload_op(
        ggml_backend_dev_t dev, const ggml_tensor * op) {
    (void) dev;
    return op->op == GGML_OP_REMOTE_ATTN;
}

static void ggml_backend_remote_attn_init_globals() {
    if (ggml_backend_remote_attn_globals_init) {
        return;
    }
    ggml_backend_remote_attn_buft_global = {
        /* .iface   = */ ggml_backend_remote_attn_buft_iface,
        /* .device  = */ &ggml_backend_remote_attn_dev_global,
        /* .context = */ nullptr,
    };
    ggml_backend_remote_attn_dev_global = {
        /* .iface   = */ {},
        /* .reg     = */ nullptr,
        /* .context = */ nullptr,
    };
    ggml_backend_remote_attn_dev_global.iface = {
        /* .get_name        = */ ggml_backend_remote_attn_dev_get_name,
        /* .get_description = */ ggml_backend_remote_attn_dev_get_description,
        /* .get_memory      = */ ggml_backend_remote_attn_dev_get_memory,
        /* .get_type        = */ ggml_backend_remote_attn_dev_get_type,
        /* .get_props       = */ ggml_backend_remote_attn_dev_get_props,
        /* .init_backend    = */ ggml_backend_remote_attn_dev_init_backend,
        /* .get_buffer_type = */ ggml_backend_remote_attn_dev_get_buffer_type,
        /* .get_host_buffer_type = */ nullptr,
        /* .buffer_from_host_ptr = */ nullptr,
        /* .supports_op     = */ ggml_backend_remote_attn_dev_supports_op,
        /* .supports_buft   = */ ggml_backend_remote_attn_dev_supports_buft,
        /* .offload_op      = */ ggml_backend_remote_attn_dev_offload_op,
        /* .event_new       = */ nullptr,
        /* .event_free      = */ nullptr,
        /* .event_synchronize = */ nullptr,
    };
    ggml_backend_remote_attn_globals_init = true;
}

} // anonymous namespace

// ===========================================================================
// public API implementation
// ===========================================================================

ggml_backend_t ggml_backend_remote_attn_init(const char * host, uint16_t port) {
#ifdef _WIN32
    (void) host; (void) port;
    GGML_LOG_ERROR("%s: remote attention requires a POSIX host\n", GGML_REMOTE_ATTN_NAME);
    return nullptr;
#else
    ggml_backend_remote_attn_init_globals();

    auto * ctx = new ggml_backend_remote_attn_context();
    ctx->host = host ? host : "";
    ctx->port = port;

    auto * backend = new ggml_backend {
        /* .guid   = */ {},
        /* .iface  = */ ggml_backend_remote_attn_iface,
        /* .device = */ &ggml_backend_remote_attn_dev_global,
        /* .context= */ ctx,
    };
    return backend;
#endif
}

bool ggml_backend_remote_attn_set_geometry(
        ggml_backend_t backend, const struct ggml_remote_attn_geometry * geo) {
    if (!backend || !geo) {
        return false;
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    ctx->geo = *geo;
    ctx->geo_set = true;
    return true;
}

bool ggml_backend_remote_attn_connect(ggml_backend_t backend) {
#ifdef _WIN32
    (void) backend;
    return false;
#else
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (!ctx->geo_set) {
        GGML_LOG_ERROR("%s: geometry not set before connect\n", GGML_REMOTE_ATTN_NAME);
        return false;
    }
    ctx->sock.close();
    ctx->connected = false;

    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        GGML_LOG_ERROR("%s: socket failed: %s\n", GGML_REMOTE_ATTN_NAME, strerror(errno));
        return false;
    }
    ctx->sock.fd = fd;
    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    timeval tv { 30, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    // large socket buffers: prefill messages can be multi-MB
    int sndbuf = 8 << 20, rcvbuf = 8 << 20;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, (const char *) &sndbuf, sizeof(sndbuf));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, (const char *) &rcvbuf, sizeof(rcvbuf));

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(ctx->port);
    if (::inet_pton(AF_INET, ctx->host.c_str(), &addr.sin_addr) != 1) {
        GGML_LOG_ERROR("%s: invalid host '%s' (numeric IPv4 required)\n",
                       GGML_REMOTE_ATTN_NAME, ctx->host.c_str());
        ctx->sock.close();
        return false;
    }
    if (::connect(fd, (sockaddr *) &addr, sizeof(addr)) != 0) {
        GGML_LOG_ERROR("%s: connect to %s:%u failed: %s\n",
                       GGML_REMOTE_ATTN_NAME, ctx->host.c_str(), (unsigned) ctx->port, strerror(errno));
        ctx->sock.close();
        return false;
    }

    // HELLO
    rkva_hello hello {};
    hello.proto_flags    = (1u << 0) | (1u << 1);  // host supports F16 and F32
    hello.n_layer_remote = ctx->geo.n_layer_remote;
    hello.n_head         = ctx->geo.n_head;
    hello.n_head_kv      = ctx->geo.n_head_kv;
    hello.head_dim       = ctx->geo.head_dim;
    hello.max_ctx        = ctx->geo.max_ctx;
    hello.cache_bits_k   = ctx->geo.cache_bits_k;
    hello.cache_bits_v   = ctx->geo.cache_bits_v;
    hello.group_tokens   = ctx->geo.group_tokens;
    hello.sinkhorn_iters = ctx->geo.sinkhorn_iters;
    hello.tail_tokens    = ctx->geo.tail_tokens;
    hello.tail_groups    = ctx->geo.tail_groups;
    hello.tail_type      = ctx->geo.tail_type;
    hello.domain         = ctx->geo.domain;
    hello.has_sinks      = ctx->geo.has_sinks;
    hello.swa            = ctx->geo.swa;
    hello.kq_scale       = ctx->geo.kq_scale;

    ctx->connected = true;  // allow transact
    std::vector<uint8_t> ack_bytes;
    if (!rkva_transact(ctx, RKVA_HELLO, RKVA_SESSION_NONE, 0, 0,
                       &hello, sizeof(hello), ack_bytes, sizeof(rkva_hello_ack))) {
        GGML_LOG_ERROR("%s: HELLO handshake failed\n", GGML_REMOTE_ATTN_NAME);
        ctx->connected = false;
        ctx->sock.close();
        return false;
    }
    std::memcpy(&ctx->ack, ack_bytes.data(), sizeof(rkva_hello_ack));
    if (!(ctx->ack.supported_ops & (1u << RKVA_ATTN_DECODE)) ||
        !(ctx->ack.supported_ops & (1u << RKVA_ATTN_PREFILL))) {
        GGML_LOG_ERROR("%s: server lacks ATTN ops (mask=0x%x)\n",
                       GGML_REMOTE_ATTN_NAME, ctx->ack.supported_ops);
        ctx->connected = false;
        ctx->sock.close();
        return false;
    }
    GGML_LOG_INFO("%s: connected to %s:%u, %u remote layers, wire=%s, budget=%llu MiB\n",
                  GGML_REMOTE_ATTN_NAME, ctx->host.c_str(), (unsigned) ctx->port,
                  ctx->geo.n_layer_remote,
                  ctx->ack.chosen_wire == RKVA_WIRE_F16 ? "f16" : "f32",
                  (unsigned long long) (ctx->ack.mem_budget_bytes >> 20));
    return true;
#endif
}

bool ggml_backend_remote_attn_connected(ggml_backend_t backend) {
    if (!backend) return false;
    if (ggml_backend_is_local_split(backend)) {
        return ggml_backend_local_split_is_ready(backend);
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    return ctx->connected && !ctx->failed;
}

bool ggml_backend_remote_attn_create_session(
        ggml_backend_t backend, uint32_t session_id, uint32_t seq_id, uint32_t capacity) {
    if (!backend) return false;
    if (ggml_backend_is_local_split(backend)) {
        return ggml_backend_local_split_create_session(backend, session_id, seq_id, capacity);
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (!ctx->connected || ctx->failed) {
        return false;
    }
    rkva_session_req req { session_id, seq_id, capacity, 0 };
    std::vector<uint8_t> out;
    return rkva_transact(ctx, RKVA_CREATE_SESSION, session_id, 0, 0,
                         &req, sizeof(req), out, 0);
}

bool ggml_backend_remote_attn_destroy_session(ggml_backend_t backend, uint32_t session_id) {
    if (!backend) return false;
    if (ggml_backend_is_local_split(backend)) {
        return ggml_backend_local_split_destroy_session(backend, session_id);
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (!ctx->connected || ctx->failed) {
        return false;
    }
    std::vector<uint8_t> out;
    return rkva_transact(ctx, RKVA_DESTROY_SESSION, session_id, 0, 0, nullptr, 0, out, 0);
}

bool ggml_backend_remote_attn_reset(ggml_backend_t backend, uint32_t session_id) {
    if (!backend) return false;
    if (ggml_backend_is_local_split(backend)) {
        return ggml_backend_local_split_reset(backend, session_id);
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (!ctx->connected || ctx->failed) {
        return false;
    }
    std::vector<uint8_t> out;
    return rkva_transact(ctx, RKVA_RESET, session_id, 0, 0, nullptr, 0, out, 0);
}

bool ggml_backend_remote_attn_trim(ggml_backend_t backend, uint32_t session_id, int32_t pos0) {
    if (!backend) return false;
    if (ggml_backend_is_local_split(backend)) {
        return ggml_backend_local_split_trim(backend, session_id, pos0);
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (!ctx->connected || ctx->failed) {
        return false;
    }
    rkva_trim_req req { pos0 };
    std::vector<uint8_t> out;
    return rkva_transact(ctx, RKVA_REWIND, session_id, 0, 0, &req, sizeof(req), out, 0);
}

void ggml_backend_remote_attn_set_active_session(ggml_backend_t backend, uint32_t session_id) {
    if (!backend) return;
    if (ggml_backend_is_local_split(backend)) {
        ggml_backend_local_split_set_active_session(backend, session_id);
        return;
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    ctx->active_session = session_id;
}

bool ggml_backend_remote_attn_failed(ggml_backend_t backend) {
    if (!backend) return true;
    if (ggml_backend_is_local_split(backend)) {
        return ggml_backend_local_split_failed(backend);
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    return ctx->failed;
}

const char * ggml_backend_remote_attn_stats_json(ggml_backend_t backend) {
    if (!backend) return "{}";
    if (ggml_backend_is_local_split(backend)) {
        return ggml_backend_local_split_stats_json(backend);
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    auto & s = ctx->stats;
    double avg = s.calls ? (double) s.rpc_us_sum / (double) s.calls : 0.0;
    double p50 = 0.0, p95 = 0.0;
    if (!s.rpc_us.empty()) {
        std::vector<uint64_t> v = s.rpc_us;
        std::sort(v.begin(), v.end());
        p50 = (double) v[v.size() / 2];
        p95 = (double) v[(size_t) (v.size() * 0.95)];
    }
    double tx_per_call = s.calls ? (double) s.bytes_tx / (double) s.calls : 0.0;
    double rx_per_call = s.calls ? (double) s.bytes_rx / (double) s.calls : 0.0;
    char buf[1024];
    snprintf(buf, sizeof(buf),
        "{\"calls\":%llu,\"bytes_tx\":%llu,\"bytes_rx\":%llu,"
        "\"avg_tx_bytes_per_call\":%.1f,\"avg_rx_bytes_per_call\":%.1f,"
        "\"rpc_us_avg\":%.1f,\"rpc_us_p50\":%.1f,\"rpc_us_p95\":%.1f,\"rpc_us_max\":%llu,"
        "\"xbox_compute_us_avg\":%.1f,\"xbox_prepare_us_avg\":%.1f}",
        (unsigned long long) s.calls, (unsigned long long) s.bytes_tx,
        (unsigned long long) s.bytes_rx, tx_per_call, rx_per_call,
        avg, p50, p95, (unsigned long long) s.rpc_us_max,
        s.calls ? (double) s.xbox_compute_us_sum / (double) s.calls : 0.0,
        s.calls ? (double) s.xbox_prepare_us_sum / (double) s.calls : 0.0);
    ctx->stats_json = buf;
    return ctx->stats_json.c_str();
}

void ggml_backend_remote_attn_reset_stats(ggml_backend_t backend) {
    if (!backend) return;
    if (ggml_backend_is_local_split(backend)) {
        ggml_backend_local_split_reset_stats(backend);
        return;
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->stats = rkva_stats {};
}

// ---------------------------------------------------------------------------
// CPU-pinned execution path (PATH B): the op is computed on the CPU backend and
// calls into the active connection here. One active connection per process (MVP).
// ---------------------------------------------------------------------------

static ggml_backend_t g_remote_attn_active = nullptr;

void ggml_remote_attn_set_active(ggml_backend_t backend) {
    g_remote_attn_active = backend;
    if (ggml_backend_is_local_split(backend)) {
        ggml_backend_local_split_set_active(backend);
    }
}

bool ggml_remote_attn_exec(struct ggml_tensor * node) {
    ggml_backend_t backend = g_remote_attn_active;
    if (backend == nullptr) {
        GGML_LOG_ERROR("%s: exec with no active remote connection\n", GGML_REMOTE_ATTN_NAME);
        return false;
    }
    if (ggml_backend_is_local_split(backend)) {
        return ggml_local_split_exec(node);
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (ctx->failed || !ctx->connected) {
        GGML_LOG_ERROR("%s: exec while not connected (failed=%d)\n",
                       GGML_REMOTE_ATTN_NAME, (int) ctx->failed);
        return false;
    }
    if (!rkva_run_attn(ctx, node)) {
        ctx->failed = true;
        ctx->connected = false;
        ctx->sock.close();
        return false;
    }
    return true;
}
