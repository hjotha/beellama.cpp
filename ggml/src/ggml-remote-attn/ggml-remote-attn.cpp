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
#include "rkva-kvarn-producer.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
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
    uint64_t range_upload_calls = 0;
    uint64_t range_upload_rows = 0;
    uint64_t range_upload_bytes = 0;
    uint64_t range_attn_calls = 0;
    uint64_t range_fused_calls = 0;
    int64_t range_query_min = -1;
    int64_t range_query_max = -1;
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
    bool        range_configured = false;
    ggml_type   range_source_type = GGML_TYPE_COUNT;
    rkva_wire   range_wire = RKVA_WIRE_F16;
    uint32_t    range_origin = 0;
    uint32_t    range_capacity = 0;
    uint64_t    range_epoch = 1;
    bool        range_fused = false;
    bool        range_profile = false;
    bool        range_profile_controlled = false;
    uint32_t    range_kernel_mode = 0;
    struct range_layer {
        uint64_t epoch = 0;
        uint32_t rows = 0;
    };
    std::vector<range_layer> range_layers;
    bool range_native_kvarn = false;
    ggml_backend_cpu_kvarn_store_direct_v1_t native_store = nullptr;
    std::vector<std::unique_ptr<rkva_kvarn_producer>> native_layers;

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
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

// low-level request/response over the persistent socket
static bool rkva_transact(
        ggml_backend_remote_attn_context * ctx,
        rkva_op op, uint32_t session_id, uint32_t layer_id, uint32_t n_tokens,
        const void * payload, uint32_t payload_bytes,
        std::vector<uint8_t> & out, uint32_t expected_bytes, int32_t * server_status = nullptr) {

    // A partial frame cannot be retried on the same ordered byte stream. Valid
    // server errors with an empty body remain recoverable (snapshot fallback).
    const auto stream_failure = [&]() {
        ctx->failed = true;
        ctx->connected = false;
        ctx->sock.close();
        out.clear();
        return false;
    };
    if (ctx->failed || !ctx->sock.open()) {
        return stream_failure();
    }
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
        return stream_failure();
    }
    if (payload_bytes && !ctx->sock.write_all(payload, payload_bytes)) {
        return stream_failure();
    }

    rkva_response rsp {};
    if (!ctx->sock.read_all(&rsp, sizeof(rsp))) {
        return stream_failure();
    }
    if (rsp.magic != RKVA_MAGIC || rsp.version != RKVA_VERSION ||
        rsp.op != (uint16_t) op || rsp.cycle_id != req.cycle_id) {
        return stream_failure();
    }
    if (server_status) *server_status = rsp.status;
    if (rsp.status != RKVA_OK) {
        // Optional cache failures carry no payload. Refuse an invalid envelope
        // instead of desynchronizing the stream before fallback uploads.
        if (rsp.payload_bytes) { return stream_failure(); }
        GGML_LOG_ERROR("%s: server error status=%d op=%d layer=%u\n",
                       GGML_REMOTE_ATTN_NAME, rsp.status, (int) op, layer_id);
        return false;
    }
    if (rsp.payload_bytes > 256u * 1024u * 1024u || (expected_bytes && rsp.payload_bytes != expected_bytes)) {
        return stream_failure();
    }
    out.resize(rsp.payload_bytes);
    if (rsp.payload_bytes && !ctx->sock.read_all(out.data(), rsp.payload_bytes)) {
        return stream_failure();
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

static bool rkva_run_native_range(ggml_backend_remote_attn_context * ctx, ggml_tensor * node,
                                  uint32_t layer, uint32_t live, size_t nq, size_t out_width) {
    const auto * keys = node->src[5];
    const auto * values = node->src[6];
    const size_t width = size_t(ctx->geo.n_head_kv) * 256;
    if (!keys || !values || keys->type != GGML_TYPE_F32 || values->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(keys) || !ggml_is_contiguous(values) ||
            !ggml_are_same_shape(keys, values) || keys->ne[0] != 256 ||
            keys->ne[1] != ctx->geo.n_head_kv || keys->ne[2] != int64_t(nq) ||
            keys->ne[3] != 1 || nq > 256 || !ctx->native_store) return false;
    auto & cached = ctx->range_layers[layer];
    auto & owner = ctx->native_layers[layer];
    if ((owner && cached.epoch != ctx->range_epoch) ||
            live != (owner ? owner->rows() : 0u) + nq) {
        GGML_LOG_ERROR("%s: native KVarN4 requires fresh append-only overflow; quantized snapshot bootstrap/rewind is unsupported\n",
                       GGML_REMOTE_ATTN_NAME);
        return false;
    }
    try {
        if (!owner) {
            owner.reset(new rkva_kvarn_producer(ctx->range_origin, ctx->range_capacity,
                ctx->geo.n_head_kv, ctx->geo.sinkhorn_iters, ctx->native_store));
        }
        std::vector<float> original_k(nq * width), original_v(nq * width);
        ggml_backend_tensor_get(keys, original_k.data(), 0, original_k.size() * sizeof(float));
        ggml_backend_tensor_get(values, original_v.data(), 0, original_v.size() * sizeof(float));
        if (!owner->append(original_k.data(), original_v.data(), ctx->pos_scratch.data(),
                            uint32_t(nq), cached.rows == 0, ctx->payload_tx)) return false;
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("%s: native KVarN4 producer failed: %s\n", GGML_REMOTE_ATTN_NAME, error.what());
        return false;
    }
    std::vector<uint8_t> reply;
    if (ctx->payload_tx.size() > (256u << 20) ||
            !rkva_transact(ctx, RKVA_KVARN_RANGE_UPLOAD, ctx->active_session, layer, 0,
                ctx->payload_tx.data(), uint32_t(ctx->payload_tx.size()), reply, 0) || !reply.empty()) return false;
    ++ctx->stats.range_upload_calls;
    ctx->stats.range_upload_rows += nq;
    ctx->stats.range_upload_bytes += ctx->payload_tx.size() - sizeof(rkva_kvarn_range_upload);
    cached.epoch = ctx->range_epoch;
    cached.rows = live;
    const rkva_range_attn request {RKVA_WIRE_F32, 0, {0, 0}};
    const size_t query_bytes = nq * (sizeof(int32_t) + out_width * sizeof(float));
    const size_t output_bytes = nq * (out_width + ctx->geo.n_head) * sizeof(float);
    ctx->payload_tx.resize(sizeof(request) + query_bytes);
    std::memcpy(ctx->payload_tx.data(), &request, sizeof(request));
    std::memcpy(ctx->payload_tx.data() + sizeof(request), ctx->pos_scratch.data(), nq * sizeof(int32_t));
    ggml_backend_tensor_get(node->src[0], ctx->payload_tx.data() + sizeof(request) + nq * sizeof(int32_t),
                            0, nq * out_width * sizeof(float));
    if (!rkva_transact(ctx, RKVA_RANGE_ATTN, ctx->active_session, layer, uint32_t(nq),
            ctx->payload_tx.data(), uint32_t(ctx->payload_tx.size()), reply, uint32_t(output_bytes))) return false;
    for (size_t offset = 0; offset < reply.size(); offset += sizeof(float)) {
        float value;
        std::memcpy(&value, reply.data() + offset, sizeof(value));
        if (!std::isfinite(value)) return false;
    }
    ++ctx->stats.range_attn_calls;
    if (ctx->stats.range_query_min < 0) ctx->stats.range_query_min = ctx->pos_scratch.front();
    ctx->stats.range_query_max = std::max<int64_t>(ctx->stats.range_query_max, ctx->pos_scratch.back());
    auto * dst = static_cast<uint8_t *>(node->data);
    for (size_t query = 0; query < nq; ++query) {
        std::memcpy(dst, reply.data() + query * out_width * sizeof(float), out_width * sizeof(float));
        dst += out_width * sizeof(float);
        std::memcpy(dst, reply.data() + (nq * out_width + query * ctx->geo.n_head) * sizeof(float),
                    ctx->geo.n_head * sizeof(float));
        dst += ctx->geo.n_head * sizeof(float);
    }
    return true;
}

static bool rkva_run_range(ggml_backend_remote_attn_context * ctx, ggml_tensor * node) {
    const auto * q = node->src[0];
    const auto * k = node->src[1];
    const auto * v = node->src[2];
    const auto * cells = node->src[3];
    const auto * pos = node->src[4];
    const int layer = ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_LAYER_ID);
    const int n_live = ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_N_LIVE);
    const int origin = ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_ORIGIN);
    const size_t nq = q ? (size_t) q->ne[2] : 0;
    const size_t width = (size_t) ctx->geo.n_head_kv * ctx->geo.head_dim;
    const size_t out_width = (size_t) ctx->geo.n_head * ctx->geo.head_dim;
    if (!ctx->range_configured || ctx->active_session == RKVA_SESSION_NONE ||
            layer < 0 || (size_t) layer >= ctx->range_layers.size() ||
            origin < 0 || (uint32_t) origin != ctx->range_origin || n_live <= 0 ||
            (uint32_t) n_live > ctx->range_capacity || !q || !k || !v || !cells || !pos ||
            q->type != GGML_TYPE_F32 || !ggml_is_contiguous(q) || q->ne[3] != 1 || nq == 0 ||
            q->ne[0] != ctx->geo.head_dim || q->ne[1] != ctx->geo.n_head ||
            k->type != ctx->range_source_type || v->type != ctx->range_source_type ||
            !ggml_are_same_shape(k, v) || k->ne[0] != (int64_t) width ||
            k->ne[2] != 1 || k->ne[3] != 1 ||
            cells->type != GGML_TYPE_I32 || pos->type != GGML_TYPE_I32 ||
            !ggml_is_contiguous(cells) || !ggml_is_contiguous(pos) ||
            ggml_nelements(cells) < n_live || ggml_nelements(pos) < (int64_t) nq ||
            node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node) ||
            node->ne[0] != (int64_t) (out_width + ctx->geo.n_head) || node->ne[1] != (int64_t) nq ||
            ggml_get_op_params_f32(node, GGML_REMOTE_ATTN_PARAM_KQ_SCALE) != ctx->geo.kq_scale) {
        return false;
    }
    std::vector<int32_t> ids((size_t) n_live);
    ggml_backend_tensor_get(cells, ids.data(), 0, ids.size() * sizeof(int32_t));
    uint32_t count = 0;
    bool padding = false;
    for (const auto id : ids) {
        if (id == -1) {
            padding = true;
        } else {
            if (padding || id < 0 || (uint32_t) id != ctx->range_origin + count) {
                return false; // holes/reordering are not append-only range histories
            }
            ++count;
        }
    }
    if (count == 0 || count > (uint64_t) k->ne[1] || count > (uint64_t) v->ne[1]) {
        return false;
    }
    ctx->pos_scratch.resize(nq);
    ggml_backend_tensor_get(pos, ctx->pos_scratch.data(), 0, nq * sizeof(int32_t));
    for (size_t i = 0; i < nq; ++i) {
        const int64_t p = ctx->pos_scratch[i];
        if (p < ctx->range_origin || p >= (int64_t) ctx->range_origin + count ||
                (i && p != (int64_t) ctx->pos_scratch[i-1] + 1)) {
            return false;
        }
    }
    if ((uint32_t) ctx->pos_scratch.back() + 1 != ctx->range_origin + count) {
        return false;
    }
    if (ctx->range_native_kvarn) {
        return rkva_run_native_range(ctx, node, uint32_t(layer), count, nq, out_width);
    }
    auto & cached = ctx->range_layers[(size_t) layer];
    uint32_t start = cached.epoch == ctx->range_epoch ? cached.rows : 0;
    if (start > count || (uint32_t) ctx->pos_scratch.front() < ctx->range_origin + start) {
        start = 0; // rewind or rewritten rows: replace this layer, never reuse stale KV
    }
    const size_t row_bytes = ggml_row_size(ctx->range_source_type, width);
    if (k->nb[1] < row_bytes || v->nb[1] < row_bytes || width > (16u << 20) / 4) {
        return false;
    }
    const bool native_quant = ctx->range_wire != RKVA_WIRE_F16;
    std::vector<uint8_t> packed(native_quant ? 0 : row_bytes);
    std::vector<float> decoded(native_quant ? 0 : width);
    const auto to_float = ggml_get_type_traits(ctx->range_source_type)->to_float;
    const size_t upload_row_bytes = native_quant ? row_bytes * 2 : width * 4;
    const uint32_t chunk_rows = std::max<uint32_t>(1, (16u << 20) / (uint32_t) upload_row_bytes);
    const uint64_t output_bytes = nq * (out_width + ctx->geo.n_head) * sizeof(float);
    const uint64_t query_bytes = nq * (sizeof(int32_t) + out_width * sizeof(float));
    if (output_bytes > (256u << 20) || query_bytes + sizeof(rkva_range_attn) > (256u << 20)) {
        return false;
    }
    std::vector<uint8_t> reply;
    bool attended = false;
    while (start < count) {
        const uint32_t rows = std::min(chunk_rows, count - start);
        rkva_range_upload upload { start, rows, ctx->range_wire, 0 };
        const bool combine = ctx->range_fused && start + rows == count &&
                sizeof(rkva_range_upload_attn) + (uint64_t) rows * upload_row_bytes + query_bytes <= (256u << 20);
        const size_t prefix = combine ? sizeof(rkva_range_upload_attn) : sizeof(upload);
        ctx->payload_tx.resize(prefix + (size_t) rows * upload_row_bytes + (combine ? query_bytes : 0));
        if (combine) {
            const rkva_range_upload_attn request {upload, {RKVA_WIRE_F32, 0, {0, 0}}};
            std::memcpy(ctx->payload_tx.data(), &request, sizeof(request));
        } else {
            std::memcpy(ctx->payload_tx.data(), &upload, sizeof(upload));
        }
        uint8_t * dst = ctx->payload_tx.data() + prefix;
        for (const auto * source : {k, v}) {
            for (uint32_t row = start; row < start + rows; ++row) {
                if (native_quant) {
                    ggml_backend_tensor_get(source, dst, (size_t) row * source->nb[1], row_bytes);
                    dst += row_bytes;
                    continue;
                }
                ggml_backend_tensor_get(source, packed.data(), (size_t) row * source->nb[1], row_bytes);
                to_float(packed.data(), decoded.data(), width);
                for (const float value : decoded) {
                    const uint16_t half = f32_to_f16_bits(value);
                    std::memcpy(dst, &half, sizeof(half));
                    dst += sizeof(half);
                }
            }
        }
        if (combine) {
            std::memcpy(dst, ctx->pos_scratch.data(), nq * sizeof(int32_t));
            ggml_backend_tensor_get(q, dst + nq * sizeof(int32_t), 0, nq * out_width * sizeof(float));
        }
        if (!rkva_transact(ctx, combine ? RKVA_RANGE_UPLOAD_ATTN : RKVA_RANGE_UPLOAD,
                ctx->active_session, (uint32_t) layer, combine ? (uint32_t) nq : rows,
                ctx->payload_tx.data(), (uint32_t) ctx->payload_tx.size(), reply,
                combine ? (uint32_t) output_bytes : 0) || (!combine && !reply.empty())) {
            return false;
        }
        if (combine) {
            attended = true;
            ++ctx->stats.range_fused_calls;
        }
        ++ctx->stats.range_upload_calls;
        ctx->stats.range_upload_rows += rows;
        ctx->stats.range_upload_bytes += (size_t) rows * upload_row_bytes;
        start += rows;
    }
    cached.epoch = ctx->range_epoch;
    cached.rows = count;
    if (!attended) {
        const rkva_range_attn request {RKVA_WIRE_F32, 0, {0, 0}};
        ctx->payload_tx.resize(sizeof(request) + (size_t) query_bytes);
        std::memcpy(ctx->payload_tx.data(), &request, sizeof(request));
        std::memcpy(ctx->payload_tx.data() + sizeof(request), ctx->pos_scratch.data(), nq * sizeof(int32_t));
        ggml_backend_tensor_get(q, ctx->payload_tx.data() + sizeof(request) + nq * sizeof(int32_t),
                0, nq * out_width * sizeof(float));
        if (!rkva_transact(ctx, RKVA_RANGE_ATTN, ctx->active_session, (uint32_t) layer, (uint32_t) nq,
                ctx->payload_tx.data(), (uint32_t) ctx->payload_tx.size(), reply, (uint32_t) output_bytes)) {
            return false;
        }
    }
    for (size_t offset = 0; offset < reply.size(); offset += sizeof(float)) {
        float value;
        std::memcpy(&value, reply.data() + offset, sizeof(value));
        if (!std::isfinite(value)) {
            return false; // each query sees at least its own uploaded key
        }
    }
    ++ctx->stats.range_attn_calls;
    if (ctx->stats.range_query_min < 0 || ctx->pos_scratch.front() < ctx->stats.range_query_min) {
        ctx->stats.range_query_min = ctx->pos_scratch.front();
    }
    ctx->stats.range_query_max = std::max<int64_t>(ctx->stats.range_query_max, ctx->pos_scratch.back());
    // Wire response is two planes. The graph node packs OUT+LSE per query so
    // views can expose both outputs without a second op or another RPC.
    auto * dst = (uint8_t *) node->data;
    for (size_t query = 0; query < nq; ++query) {
        std::memcpy(dst, reply.data() + query * out_width * sizeof(float), out_width * sizeof(float));
        dst += out_width * sizeof(float);
        std::memcpy(dst, reply.data() + (nq * out_width + query * ctx->geo.n_head) * sizeof(float),
                ctx->geo.n_head * sizeof(float));
        dst += ctx->geo.n_head * sizeof(float);
    }
    return true;
}

// Serialize Q/K/V to the chosen wire type and one RKVA attn call.
static bool rkva_run_attn(
        ggml_backend_remote_attn_context * ctx,
        ggml_tensor * node) {

    if (ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_RANGE)) {
        return rkva_run_range(ctx, node);
    }

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
    ctx->failed = false;
    ctx->range_configured = false;
    ctx->range_layers.clear();
    ctx->range_native_kvarn = false;
    ctx->native_layers.clear();
    ctx->native_store = nullptr;
    ctx->range_fused = false;
    ctx->range_profile = false;
    ctx->range_profile_controlled = false;
    ctx->range_kernel_mode = 0;

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

static bool rkva_create_session(
        ggml_backend_t backend, uint32_t session_id, uint32_t seq_id, uint32_t capacity, bool range) {
    if (!backend) return false;
    if (ggml_backend_is_local_split(backend)) {
        return !range && ggml_backend_local_split_create_session(backend, session_id, seq_id, capacity);
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (!ctx->connected || ctx->failed) {
        return false;
    }
    const uint32_t flags = range && (ctx->ack.reserved[0] & (RKVA_CAP_RANGE_Q4_0 | RKVA_CAP_RANGE_Q8_0)) ?
            RKVA_SESSION_RANGE : 0;
    rkva_session_req req { session_id, seq_id, capacity, flags };
    std::vector<uint8_t> out;
    return rkva_transact(ctx, RKVA_CREATE_SESSION, session_id, 0, 0,
                         &req, sizeof(req), out, 0);
}

bool ggml_backend_remote_attn_create_session(
        ggml_backend_t backend, uint32_t session_id, uint32_t seq_id, uint32_t capacity) {
    return rkva_create_session(backend, session_id, seq_id, capacity, false);
}

bool ggml_backend_remote_attn_create_range_session(
        ggml_backend_t backend, uint32_t session_id, uint32_t seq_id, uint32_t capacity) {
    return rkva_create_session(backend, session_id, seq_id, capacity, true);
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
    ++ctx->range_epoch;
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
    if (ctx->range_configured) {
        if (ctx->range_native_kvarn) {
            if (session_id != ctx->active_session) return false;
            for (const auto & owner : ctx->native_layers) {
                if (owner && int64_t(pos0) < int64_t(ctx->range_origin) + owner->rows()) {
                    ++ctx->range_epoch; // force subsequent compute to fail closed
                    return false;
                }
            }
            return true; // trimming an unoccupied suffix must not invalidate native records
        }
        // The host companion is authoritative; the next range op replaces each
        // layer's history, including changes that preserve its row count.
        ++ctx->range_epoch;
        return session_id == ctx->active_session;
    }
    rkva_trim_req req { pos0 };
    std::vector<uint8_t> out;
    return rkva_transact(ctx, RKVA_REWIND, session_id, 0, 0, &req, sizeof(req), out, 0);
}

// Generic opt-in is separate from the historical Q4-only setting. A server
// must positively advertise the selected storage kernel in its control reply.
static bool rkva_range_kernel_options(uint32_t & mode, bool & profile, bool & generic) {
    const char * selected = std::getenv("GGML_RKVA_RANGE_KERNEL");
    const char * legacy = std::getenv("GGML_RKVA_Q4_KERNEL");
    const auto parse = [](const char * value, uint32_t & result) {
        if (!value || !std::strcmp(value, "reference") || !std::strcmp(value, "0")) result = 0;
        else if (!std::strcmp(value, "optimized") || !std::strcmp(value, "balanced") || !std::strcmp(value, "1")) result = 1;
        else if (!std::strcmp(value, "fine") || !std::strcmp(value, "2")) result = 2;
        else if (!std::strcmp(value, "coarse") || !std::strcmp(value, "3")) result = 3;
        else return false;
        return true;
    };
    generic = selected != nullptr;
    if (!parse(selected ? selected : legacy, mode)) return false;
    if (selected && legacy) {
        uint32_t old_mode = 0;
        if (!parse(legacy, old_mode) || old_mode != mode) return false;
    }
    profile = false;
    if (const char * env = std::getenv("GGML_RKVA_PROFILE")) {
        if (!std::strcmp(env, "1")) profile = true;
        else if (std::strcmp(env, "0")) return false;
    }
    return true;
}

static bool rkva_range_apply_control(ggml_backend_remote_attn_context * ctx,
                                     const char * storage_name, bool generic) {
    const bool previous = ctx->range_profile_controlled;
    ctx->range_profile_controlled = ctx->range_profile || ctx->range_kernel_mode != 0;
    if (!ctx->range_profile_controlled && !previous) return true;
    const rkva_range_profile control {
        uint32_t((ctx->range_profile ? RKVA_PROFILE_ENABLE : 0) | RKVA_PROFILE_RESET),
        ctx->range_kernel_mode, 0};
    std::vector<uint8_t> reply;
    if (!rkva_transact(ctx, RKVA_RANGE_PROFILE, ctx->active_session, 0, 0,
            &control, sizeof(control), reply, 0) || reply.empty() || reply.size() > (1u << 20)) {
        ctx->failed = true;
        return false;
    }
    if (generic && ctx->range_kernel_mode) {
        // The compact diagnostic writer emits this fixed capability object.
        // Missing, unavailable or differently encoded reports fail closed;
        // older servers must never silently run a reference-format benchmark.
        const std::string report(reply.begin(), reply.end());
        const size_t begin = report.find("\"optimized_kernels\":{");
        const size_t end = begin == std::string::npos ? begin : report.find("},\"buckets\":{", begin);
        const std::string field = std::string("\"") + storage_name + "\":{\"available\":true,";
        const size_t found = begin == std::string::npos ? begin : report.find(field, begin);
        if (found == std::string::npos || end == std::string::npos || found >= end) {
            GGML_LOG_ERROR("%s: server does not advertise an optimized %s range kernel\n", GGML_REMOTE_ATTN_NAME, storage_name);
            ctx->failed = true;
            return false;
        }
    }
    return true;
}

static bool rkva_configure_native_kvarn(ggml_backend_remote_attn_context * ctx,
                                       uint32_t origin, uint32_t capacity, ggml_type source_type) {
    const char * experimental = std::getenv("GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    constexpr uint32_t required = (1u << RKVA_KVARN_RANGE_CONFIG) |
                                  (1u << RKVA_KVARN_RANGE_UPLOAD) | (1u << RKVA_RANGE_ATTN);
    if (!experimental || std::strcmp(experimental, "1") || !ctx->connected || ctx->failed ||
            ctx->active_session == RKVA_SESSION_NONE || !(ctx->ack.reserved[0] & RKVA_CAP_RANGE_KVARN4) ||
            (ctx->ack.supported_ops & required) != required || origin == 0 || origin % 128 || !capacity ||
            uint64_t(origin) + capacity > INT32_MAX || ctx->geo.max_ctx != capacity ||
            ctx->geo.head_dim != 256 || ctx->geo.n_head_kv == 0 ||
            ctx->geo.group_tokens != 128 || ctx->geo.sinkhorn_iters == 0 ||
            ctx->geo.tail_type != 0 || ctx->geo.has_sinks || ctx->geo.swa ||
            (source_type != GGML_TYPE_Q4_0 && source_type != GGML_TYPE_Q8_0)) return false;
    if (const char * fuse = std::getenv("GGML_RKVA_RANGE_FUSE")) {
        if (std::strcmp(fuse, "auto") && std::strcmp(fuse, "0")) return false;
    }
    uint32_t mode = 0;
    bool profile = false, generic = false;
    if (!rkva_range_kernel_options(mode, profile, generic) || (mode && !generic)) return false;
    const bool can_profile = (ctx->ack.reserved[0] & RKVA_CAP_RANGE_PROFILE) &&
                             (ctx->ack.supported_ops & (1u << RKVA_RANGE_PROFILE));
    if ((mode || profile) && !can_profile) return false;
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) return false;
    auto store = reinterpret_cast<ggml_backend_cpu_kvarn_store_direct_v1_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu), "ggml_backend_cpu_kvarn_store_direct_v1"));
    if (!store) return false;
    const rkva_kvarn_range_config descriptor {
        origin, capacity, RKVA_RANGE_STORAGE_KVARN4, 1,
        128, 128, 256, 2, 4, 4, 8960, 3, 2, 128, 128, 1, 1, 1};
    std::vector<uint8_t> reply;
    if (!rkva_transact(ctx, RKVA_KVARN_RANGE_CONFIG, ctx->active_session, 0, 0,
            &descriptor, sizeof(descriptor), reply, 0) || !reply.empty()) return false;
    ctx->range_configured = true;
    ctx->range_native_kvarn = true;
    ctx->native_store = store;
    ctx->native_layers.clear();
    ctx->native_layers.resize(ctx->geo.n_layer_remote);
    ctx->range_source_type = source_type;
    ctx->range_wire = RKVA_WIRE_F32; // Q/OUT only; op16 carries structured native bytes
    ctx->range_origin = origin;
    ctx->range_capacity = capacity;
    ++ctx->range_epoch;
    ctx->range_layers.assign(ctx->geo.n_layer_remote, {});
    ctx->range_fused = false;
    ctx->range_profile = profile;
    ctx->range_kernel_mode = mode;
    if (!rkva_range_apply_control(ctx, "kvarn4", generic)) return false;
    GGML_LOG_WARN("%s: EXPERIMENTAL append-only native KVarN4; original F32 producer, CPU native owner, no native snapshot/rewind support\n",
                  GGML_REMOTE_ATTN_NAME);
    return true;
}

bool ggml_backend_remote_attn_configure_range(
        ggml_backend_t backend, uint32_t origin, uint32_t capacity, ggml_type cache_type) {
    if (!backend || ggml_backend_is_local_split(backend)) {
        return false;
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (const char * storage = std::getenv("GGML_RKVA_RANGE_STORAGE")) {
        if (!std::strcmp(storage, "kvarn4")) return rkva_configure_native_kvarn(ctx, origin, capacity, cache_type);
    }
    constexpr uint32_t required = (1u << RKVA_RANGE_CONFIG) | (1u << RKVA_RANGE_UPLOAD) | (1u << RKVA_RANGE_ATTN);
    if (!ctx->connected || ctx->failed || ctx->active_session == RKVA_SESSION_NONE ||
            (ctx->ack.supported_ops & required) != required || capacity == 0 ||
            origin > INT32_MAX || capacity > (uint32_t) INT32_MAX - origin ||
            ctx->geo.max_ctx != capacity || ctx->geo.n_layer_remote == 0 ||
            ctx->geo.cache_bits_k != 0 || ctx->geo.cache_bits_v != 0 ||
            ctx->geo.domain != GGML_REMOTE_ATTN_DOMAIN_AUTO || ctx->geo.has_sinks || ctx->geo.swa ||
            (cache_type != GGML_TYPE_Q4_0 && cache_type != GGML_TYPE_Q8_0)) {
        return false;
    }
    const bool source_q4 = cache_type == GGML_TYPE_Q4_0;
    const uint64_t capability = source_q4 ? RKVA_CAP_RANGE_Q4_0 : RKVA_CAP_RANGE_Q8_0;
    const bool supports_native = (ctx->ack.reserved[0] & capability) != 0;
    const char * requested = std::getenv("GGML_RKVA_RANGE_STORAGE");
    bool native_quant = supports_native;
    if (requested && std::strcmp(requested, "auto") != 0) {
        if (std::strcmp(requested, "f16") == 0) {
            native_quant = false;
        } else if (std::strcmp(requested, "q4_0") == 0 || std::strcmp(requested, "q8_0") == 0) {
            if (std::strcmp(requested, ggml_type_name(cache_type)) != 0 || !supports_native) {
                GGML_LOG_ERROR("%s: GGML_RKVA_RANGE_STORAGE=%s requires a matching host cache and server capability\n",
                        GGML_REMOTE_ATTN_NAME, requested);
                return false;
            }
            native_quant = true;
        } else {
            GGML_LOG_ERROR("%s: invalid GGML_RKVA_RANGE_STORAGE='%s' (expected auto, f16, q4_0 or q8_0)\n",
                    GGML_REMOTE_ATTN_NAME, requested);
            return false;
        }
    }
    const bool can_fuse = (ctx->ack.reserved[0] & RKVA_CAP_RANGE_UPLOAD_ATTN) &&
                          (ctx->ack.supported_ops & (1u << RKVA_RANGE_UPLOAD_ATTN));
    bool fused = can_fuse;
    if (const char * env = std::getenv("GGML_RKVA_RANGE_FUSE")) {
        if (std::strcmp(env, "0") == 0) fused = false;
        else if (std::strcmp(env, "1") == 0) {
            if (!can_fuse) return false;
        } else if (std::strcmp(env, "auto") != 0) return false;
    }
    uint32_t kernel_mode = 0;
    bool profile = false, generic_kernel = false;
    if (!rkva_range_kernel_options(kernel_mode, profile, generic_kernel)) return false;
    const bool can_profile = (ctx->ack.reserved[0] & RKVA_CAP_RANGE_PROFILE) &&
                            (ctx->ack.supported_ops & (1u << RKVA_RANGE_PROFILE));
    if ((profile || kernel_mode) && (!can_profile ||
            (kernel_mode && !generic_kernel && (!source_q4 || !native_quant)))) {
        return false;
    }
    const uint32_t storage = native_quant ? (source_q4 ? RKVA_RANGE_STORAGE_Q4_0 : RKVA_RANGE_STORAGE_Q8_0) : 0;
    const rkva_range_config req {origin, capacity, {storage, 0}};
    std::vector<uint8_t> reply;
    if (!rkva_transact(ctx, RKVA_RANGE_CONFIG, ctx->active_session, 0, 0,
            &req, sizeof(req), reply, 0) || !reply.empty()) {
        ctx->failed = true;
        return false;
    }
    ctx->range_configured = true;
    ctx->range_native_kvarn = false;
    ctx->native_layers.clear();
    ctx->range_source_type = cache_type;
    ctx->range_wire = native_quant ? (source_q4 ? RKVA_WIRE_Q4_0 : RKVA_WIRE_Q8_0) : RKVA_WIRE_F16;
    ctx->range_origin = origin;
    ctx->range_capacity = capacity;
    ++ctx->range_epoch;
    ctx->range_layers.assign(ctx->geo.n_layer_remote, {});
    ctx->range_fused = fused;
    ctx->range_profile = profile;
    ctx->range_kernel_mode = kernel_mode;
    const char * storage_name = native_quant ? (source_q4 ? "q4_0" : "q8_0") : "f16";
    if (!rkva_range_apply_control(ctx, storage_name, generic_kernel)) return false;
    return true;
}

bool ggml_backend_remote_attn_disk_snapshot_enabled(ggml_backend_t backend) {
    const char * enabled = std::getenv("GGML_RKVA_DISK_SNAPSHOTS");
    if (!backend || ggml_backend_is_local_split(backend) || !enabled || std::strcmp(enabled, "1")) {
        return false;
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    constexpr uint32_t ops = (1u << RKVA_SNAPSHOT_SAVE) | (1u << RKVA_SNAPSHOT_RESTORE);
    return ctx->connected && !ctx->failed && ctx->range_configured && !ctx->range_native_kvarn &&
        ctx->active_session != RKVA_SESSION_NONE && (ctx->ack.supported_ops & ops) == ops &&
        (ctx->ack.reserved[0] & RKVA_CAP_DISK_SNAPSHOT);
}

bool ggml_backend_remote_attn_disk_snapshot(
        ggml_backend_t backend, const uint8_t host_state_sha256[32], uint32_t live_rows, bool restore) {
    if (!ggml_backend_remote_attn_disk_snapshot_enabled(backend) || !host_state_sha256 || !live_rows) {
        return false; // absolutely no snapshot RPC below P
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (live_rows > ctx->range_capacity) { return false; }
    if (!restore) {
        for (const auto & layer : ctx->range_layers) {
            if (layer.epoch != ctx->range_epoch || layer.rows != live_rows) {
                return false; // a checkpoint cannot combine different evaluated prefixes
            }
        }
    } else {
        // Host state has ALREADY committed. Every failure must make the next
        // graph replace all server layers from the authoritative CPU companion.
        ++ctx->range_epoch;
    }
#ifndef _WIN32
    struct snapshot_timeout {
        int fd;
        timeval previous {};
        bool valid = false;
        explicit snapshot_timeout(int descriptor) : fd(descriptor) {
            socklen_t length = sizeof(previous);
            timeval extended { 600, 0 };
            valid = getsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &previous, &length) == 0 &&
                    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &extended, sizeof(extended)) == 0;
        }
        ~snapshot_timeout() {
            if (valid) setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &previous, sizeof(previous));
        }
    } timeout(ctx->sock.fd);
    if (!timeout.valid) return false;
#endif
    rkva_snapshot_req req {};
    std::memcpy(req.host_state_sha256, host_state_sha256, 32);
    req.live_rows = live_rows; req.revision = 1;
    std::vector<uint8_t> reply;
    int32_t status = INT32_MIN;
    if (!rkva_transact(ctx, restore ? RKVA_SNAPSHOT_RESTORE : RKVA_SNAPSHOT_SAVE,
            ctx->active_session, 0, 0, &req, sizeof(req), reply, sizeof(rkva_snapshot_ack), &status)) {
        if (status == RKVA_ERR_STATE && restore && !ctx->failed) {
            // A disk read failing after GPU mutation invalidates all layers.
            // RESET clears readiness before the host reuploads everything.
            std::vector<uint8_t> reset_reply;
            if (!rkva_transact(ctx, RKVA_RESET, ctx->active_session, 0, 0,
                    nullptr, 0, reset_reply, 0) || !reset_reply.empty()) { ctx->failed = true; }
        } else if (status != RKVA_ERR_SNAPSHOT_MISS && status != RKVA_ERR_SNAPSHOT_IO) {
            ctx->failed = true; // transport, invalid envelope, or unexpected server state
        }
        return false;
    }
    rkva_snapshot_ack ack {};
    std::memcpy(&ack, reply.data(), sizeof(ack));
    if (std::memcmp(ack.host_state_sha256, host_state_sha256, 32) ||
            ack.live_rows != live_rows || ack.n_layers != ctx->geo.n_layer_remote ||
            !ack.disk_bytes || ack.disk_bytes > (4ull << 30)) {
        ctx->failed = true; return false;
    }
    if (restore) {
        for (auto & layer : ctx->range_layers) { layer.epoch = ctx->range_epoch; layer.rows = live_rows; }
    }
    return true;
}

void ggml_backend_remote_attn_invalidate_range(ggml_backend_t backend) {
    if (!backend || ggml_backend_is_local_split(backend)) {
        return;
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ++ctx->range_epoch;
}

bool ggml_backend_remote_attn_native_kvarn_enabled(ggml_backend_t backend) {
    if (!backend || ggml_backend_is_local_split(backend)) return false;
    const auto * ctx = static_cast<ggml_backend_remote_attn_context *>(backend->context);
    return ctx->range_configured && ctx->range_native_kvarn;
}

bool ggml_backend_remote_attn_native_kvarn_has_state(ggml_backend_t backend) {
    if (!ggml_backend_remote_attn_native_kvarn_enabled(backend)) return false;
    auto * ctx = static_cast<ggml_backend_remote_attn_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->mutex);
    for (const auto & owner : ctx->native_layers) if (owner && owner->rows()) return true;
    return false;
}

void ggml_remote_attn_range_set_native_sources(ggml_tensor * node, ggml_tensor * keys, ggml_tensor * values) {
    GGML_ASSERT(node && node->op == GGML_OP_REMOTE_ATTN && keys && values);
    GGML_ASSERT(ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_RANGE));
    GGML_ASSERT(keys->type == GGML_TYPE_F32 && values->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(keys) && ggml_is_contiguous(values));
    GGML_ASSERT(ggml_are_same_shape(keys, values) && keys->ne[3] == 1);
    node->src[5] = keys;
    node->src[6] = values;
}

void ggml_backend_remote_attn_set_active_session(ggml_backend_t backend, uint32_t session_id) {
    if (!backend) return;
    if (ggml_backend_is_local_split(backend)) {
        ggml_backend_local_split_set_active_session(backend, session_id);
        return;
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    if (ctx->active_session != session_id) {
        ctx->range_configured = false;
        ++ctx->range_epoch;
    }
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
    if (ctx->range_profile_controlled && ctx->connected && !ctx->failed) {
        const rkva_range_profile report {RKVA_PROFILE_REPORT_ONLY, 0, 0};
        std::vector<uint8_t> profile;
        if (rkva_transact(ctx, RKVA_RANGE_PROFILE, ctx->active_session, 0, 0,
                &report, sizeof(report), profile, 0) && !profile.empty() && profile.size() <= (1u << 20)) {
            const std::string json(profile.begin(), profile.end());
            GGML_LOG_INFO("remote_attn gpu_profile %s\n", json.c_str());
        }
    }
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
    char buf[2048];
    snprintf(buf, sizeof(buf),
        "{\"calls\":%llu,\"bytes_tx\":%llu,\"bytes_rx\":%llu,"
        "\"avg_tx_bytes_per_call\":%.1f,\"avg_rx_bytes_per_call\":%.1f,"
        "\"rpc_us_avg\":%.1f,\"rpc_us_p50\":%.1f,\"rpc_us_p95\":%.1f,\"rpc_us_max\":%llu,"
        "\"xbox_compute_us_avg\":%.1f,\"xbox_prepare_us_avg\":%.1f,"
        "\"range_fused_calls\":%llu,\"range_kernel_mode\":%u,"
        "\"range_upload_calls\":%llu,\"range_upload_rows\":%llu,\"range_attn_calls\":%llu,"
        "\"range_origin\":%u,\"range_capacity\":%u,\"range_query_min\":%lld,\"range_query_max\":%lld,"
        "\"range_cache_type\":\"%s\",\"range_source_cache_type\":\"%s\",\"range_upload_bytes\":%llu}",
        (unsigned long long) s.calls, (unsigned long long) s.bytes_tx,
        (unsigned long long) s.bytes_rx, tx_per_call, rx_per_call,
        avg, p50, p95, (unsigned long long) s.rpc_us_max,
        s.calls ? (double) s.xbox_compute_us_sum / (double) s.calls : 0.0,
        s.calls ? (double) s.xbox_prepare_us_sum / (double) s.calls : 0.0,
        (unsigned long long) s.range_fused_calls, ctx->range_kernel_mode,
        (unsigned long long) s.range_upload_calls, (unsigned long long) s.range_upload_rows,
        (unsigned long long) s.range_attn_calls, ctx->range_origin, ctx->range_capacity,
        (long long) s.range_query_min, (long long) s.range_query_max,
        ctx->range_native_kvarn ? "kvarn4" : ctx->range_wire == RKVA_WIRE_Q4_0 ? "q4_0" : ctx->range_wire == RKVA_WIRE_Q8_0 ? "q8_0" : "f16",
        ctx->range_native_kvarn ? "projected_f32" : ctx->range_configured ? ggml_type_name(ctx->range_source_type) : "none",
        (unsigned long long) s.range_upload_bytes);
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
