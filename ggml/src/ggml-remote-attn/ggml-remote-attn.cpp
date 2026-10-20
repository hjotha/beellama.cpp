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
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <arpa/inet.h>
#include <cerrno>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

#define GGML_REMOTE_ATTN_NAME "RemoteAttn"

#if defined(__AVX2__) && defined(__F16C__)
#include <immintrin.h>
#endif

namespace {

// ---------------------------------------------------------------------------
// host f16 helpers (software, portable + AVX2/F16C vectorized batch)
// ---------------------------------------------------------------------------

static inline uint16_t f32_to_f16_bits(float value) {
    return (uint16_t) ggml_fp32_to_fp16(value);
}

static inline bool f16_bits_finite(uint16_t bits) {
    return (bits & 0x7c00u) != 0x7c00u;
}

static inline float f16_bits_to_f32(uint16_t bits) {
    return ggml_fp16_to_fp32((ggml_fp16_t) bits);
}

static inline bool rkva_f32_to_f16_batch(const float * src, uint16_t * dst, size_t n) {
#if defined(__AVX2__) && defined(__F16C__)
    size_t i = 0;
    const __m256 max_f16 = _mm256_set1_ps(65504.0f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 abs_v = _mm256_andnot_ps(sign_mask, v);
        // Finite in FP16: abs(v) <= 65504.0f (filters out Inf, >65504, and NaN)
        __m256 is_finite = _mm256_cmp_ps(abs_v, max_f16, _CMP_LE_OQ);
        int mask = _mm256_movemask_ps(is_finite);
        if (__builtin_expect(mask != 0xFF, 0)) {
            return false;
        }
        __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i*)(dst + i), h);
    }
    for (; i < n; ++i) {
        const float val = src[i];
        if (!std::isfinite(val) || std::abs(val) > 65504.0f) {
            return false;
        }
        const uint16_t half = f32_to_f16_bits(val);
        if (!f16_bits_finite(half)) {
            return false;
        }
        dst[i] = half;
    }
    return true;
#else
    for (size_t i = 0; i < n; ++i) {
        const float val = src[i];
        if (!std::isfinite(val) || std::abs(val) > 65504.0f) {
            return false;
        }
        const uint16_t half = f32_to_f16_bits(val);
        if (!f16_bits_finite(half)) {
            return false;
        }
        dst[i] = half;
    }
    return true;
#endif
}

static inline bool rkva_f16_to_f32_batch(const uint16_t * src, float * dst, size_t n) {
#if defined(__AVX2__) && defined(__F16C__)
    size_t i = 0;
    const __m128i exp_mask = _mm_set1_epi16(0x7c00);
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i*)(src + i));
        __m128i exps = _mm_and_si128(h, exp_mask);
        __m128i is_inf_nan = _mm_cmpeq_epi16(exps, exp_mask);
        if (__builtin_expect(!_mm_testz_si128(is_inf_nan, is_inf_nan), 0)) {
            return false;
        }
        __m256 f = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, f);
    }
    for (; i < n; ++i) {
        const uint16_t half = src[i];
        if (!f16_bits_finite(half)) {
            return false;
        }
        dst[i] = f16_bits_to_f32(half);
    }
    return true;
#else
    for (size_t i = 0; i < n; ++i) {
        const uint16_t half = src[i];
        if (!f16_bits_finite(half)) {
            return false;
        }
        dst[i] = f16_bits_to_f32(half);
    }
    return true;
#endif
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

    bool write_all_iov(const struct iovec * iov_in, int count) {
#ifndef _WIN32
        struct iovec iov[4];
        if (count <= 0 || count > 4) return false;
        for (int i = 0; i < count; ++i) iov[i] = iov_in[i];
        int cur = 0;
        while (cur < count) {
            const ssize_t n = ::writev(fd, iov + cur, count - cur);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n <= 0) {
                return false;
            }
            size_t rem = (size_t) n;
            while (cur < count && rem >= iov[cur].iov_len) {
                rem -= iov[cur].iov_len;
                ++cur;
            }
            if (cur < count && rem > 0) {
                iov[cur].iov_base = static_cast<char *>(iov[cur].iov_base) + rem;
                iov[cur].iov_len -= rem;
            }
        }
        return true;
#else
        (void) iov_in; (void) count;
        return false;
#endif
    }

    bool read_all(void * dst, size_t len) {
#ifndef _WIN32
#if defined(__linux__) && defined(TCP_QUICKACK)
        if (len > 4096) {
            int quickack = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &quickack, sizeof(quickack));
        }
#endif
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
// bounded async overlap job (native KVarN4 bulk range only)
// ---------------------------------------------------------------------------

enum class rkva_async_state {
    IDLE,
    PENDING,   // submitted, worker has not started the RPC
    RUNNING,   // worker owns the persistent stream
    DONE_OK,
    DONE_ERR,
};

struct rkva_async_job {
    std::vector<uint8_t> payload;
    uint8_t * output = nullptr;
    size_t    output_stride = 0;
    uint32_t  layer = 0;
    uint32_t  nq = 0;
    uint32_t  segments = 0;
    uint32_t  out_width = 0;
    uint64_t  upload_bytes = 0;
    int32_t   first_pos = 0;
    int32_t   last_pos = 0;
    bool      bulk_f16 = false;
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

    // One ordered owner of the persistent stream. The worker is the only
    // thread that runs a submitted job; async_state is protected by
    // async_mutex, while the job itself runs under mutex.
    std::mutex              async_mutex;
    std::condition_variable async_cv;
    std::thread             async_worker;
    rkva_async_state        async_state = rkva_async_state::IDLE;
    bool                    async_stop = false;
    std::unique_ptr<rkva_async_job> async_job;
    std::unique_ptr<rkva_async_job> async_job_reusable;
    std::vector<uint8_t>            async_reply_buffer;
    struct ggml_tensor *    async_node = nullptr;
    bool                    async_local_remote = false;

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
    uint32_t    prefill_query_chunk = 0;
    bool        range_profile = false;
    bool        range_profile_controlled = false;
    uint32_t    range_kernel_mode = 0;
    struct range_layer {
        uint64_t epoch = 0;
        uint32_t rows = 0;
    };
    std::vector<range_layer> range_layers;
    bool range_native_kvarn = false;
    bool native_serial_prefix = false;
    bool native_bulk = false;
    uint64_t native_generation = 0;
    uint32_t native_tail_tokens = 128;
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
    if (payload_bytes) {
        struct iovec iov[2];
        iov[0].iov_base = const_cast<rkva_request *>(&req);
        iov[0].iov_len  = sizeof(req);
        iov[1].iov_base = const_cast<void *>(payload);
        iov[1].iov_len  = payload_bytes;
        if (!ctx->sock.write_all_iov(iov, 2)) {
            return stream_failure();
        }
    } else {
        if (!ctx->sock.write_all(&req, sizeof(req))) {
            return stream_failure();
        }
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
// bounded async overlap (native KVarN4 bulk range only)
// ---------------------------------------------------------------------------

// Caller owns ctx->mutex and the single job slot. This is the only part of a
// submitted native range that touches the ordered RPC stream.
static bool rkva_finish_native_bulk_job(
        ggml_backend_remote_attn_context * ctx, const rkva_async_job & job) {
    if (job.layer >= ctx->range_layers.size() ||
            job.layer >= ctx->native_layers.size() || !job.output) {
        return false;
    }
    auto & cached = ctx->range_layers[job.layer];
    auto * owner = ctx->native_layers[job.layer].get();
    if (!owner) {
        return false;
    }
    const size_t out_elem_bytes = job.bulk_f16 ? sizeof(uint16_t) : sizeof(float);
    const size_t out_bytes = size_t(job.nq) * job.out_width * out_elem_bytes;
    const size_t lse_bytes = size_t(job.nq) * ctx->geo.n_head * sizeof(float);
    const size_t row_bytes = (size_t(job.out_width) + ctx->geo.n_head) * sizeof(float);
    if (job.nq == 0 || job.out_width == 0 || job.output_stride < row_bytes ||
            job.payload.size() > (256u << 20) ||
            out_bytes + lse_bytes > (256u << 20)) {
        return false;
    }
    auto & reply = ctx->async_reply_buffer;
    if (reply.capacity() < (4u << 20)) {
        reply.reserve(4u << 20);
    }
    if (!rkva_transact(ctx, RKVA_KVARN_RANGE_BULK_ATTN, ctx->active_session,
                job.layer, job.nq, job.payload.data(), uint32_t(job.payload.size()),
                reply, uint32_t(out_bytes + lse_bytes))) {
        return false;
    }
    if (job.bulk_f16) {
        const auto * src_f16 = reinterpret_cast<const uint16_t *>(reply.data());
        const auto * src_lse = reinterpret_cast<const float *>(reply.data() + out_bytes);
        for (size_t row = 0; row < job.nq; ++row) {
            auto * dst = static_cast<float *>(static_cast<void *>(job.output + row * job.output_stride));
            if (!rkva_f16_to_f32_batch(src_f16 + row * job.out_width, dst, job.out_width)) {
                return false;
            }
            for (size_t h = 0; h < ctx->geo.n_head; ++h) {
                const float lse_val = src_lse[row * ctx->geo.n_head + h];
                if (!std::isfinite(lse_val)) return false;
                dst[job.out_width + h] = lse_val;
            }
        }
    } else {
        for (size_t offset = 0; offset < reply.size(); offset += sizeof(float)) {
            float value;
            std::memcpy(&value, reply.data() + offset, sizeof(value));
            if (!std::isfinite(value)) {
                return false;
            }
        }
        for (size_t row = 0; row < job.nq; ++row) {
            auto * dst = job.output + row * job.output_stride;
            std::memcpy(dst, reply.data() + row * job.out_width * sizeof(float),
                    job.out_width * sizeof(float));
            std::memcpy(dst + job.out_width * sizeof(float),
                    reply.data() + out_bytes + row * ctx->geo.n_head * sizeof(float),
                    ctx->geo.n_head * sizeof(float));
        }
    }
    // Publish host coverage only after the single ordered server ACK. Any
    // failure above poisons the context before this point.
    if (ctx->native_generation > UINT64_MAX - job.segments) {
        return false;
    }
    ctx->native_generation += job.segments;
    cached.epoch = ctx->range_epoch;
    cached.rows = owner->rows();
    ctx->stats.range_upload_calls += job.segments;
    ctx->stats.range_upload_rows += job.nq;
    ctx->stats.range_upload_bytes += job.upload_bytes;
    ctx->stats.range_attn_calls += job.segments;
    if (ctx->stats.range_query_min < 0 || job.first_pos < ctx->stats.range_query_min) {
        ctx->stats.range_query_min = job.first_pos;
    }
    ctx->stats.range_query_max = std::max<int64_t>(ctx->stats.range_query_max, job.last_pos);
    return true;
}

static void rkva_async_worker_main(ggml_backend_remote_attn_context * ctx) {
    std::unique_lock<std::mutex> lock(ctx->async_mutex);
    for (;;) {
        ctx->async_cv.wait(lock, [ctx] {
            return ctx->async_stop || ctx->async_state == rkva_async_state::PENDING;
        });
        if (ctx->async_stop) {
            return;
        }
        std::unique_ptr<rkva_async_job> job = std::move(ctx->async_job);
        ctx->async_state = rkva_async_state::RUNNING;
        lock.unlock();

        bool ok = false;
        {
            std::lock_guard<std::mutex> ctx_lock(ctx->mutex);
            if (!ctx->failed && ctx->connected && job) {
                try {
                    ok = rkva_finish_native_bulk_job(ctx, *job);
                } catch (const std::exception & error) {
                    GGML_LOG_ERROR("%s: async native KVarN4 RPC failed: %s\n",
                            GGML_REMOTE_ATTN_NAME, error.what());
                } catch (...) {
                    // Fall through to the fail-closed path.
                }
            }
            if (!ok) {
                // An uncertain ordered-stream/GPU mutation must fail closed.
                ctx->failed = true;
                ctx->connected = false;
                ctx->sock.close();
            }
        }

        lock.lock();
        ctx->async_state = ok ? rkva_async_state::DONE_OK : rkva_async_state::DONE_ERR;
        ctx->async_cv.notify_all();
    }
}

// Wait for any submitted job and discard a completed-but-unconsumed result.
// The remote session is only mutated after this returns.
static void rkva_async_drain(ggml_backend_remote_attn_context * ctx) {
    std::unique_lock<std::mutex> lock(ctx->async_mutex);
    ctx->async_cv.wait(lock, [ctx] {
        return ctx->async_state != rkva_async_state::PENDING &&
               ctx->async_state != rkva_async_state::RUNNING;
    });
    if (ctx->async_state == rkva_async_state::DONE_OK ||
            ctx->async_state == rkva_async_state::DONE_ERR) {
        ctx->async_state = rkva_async_state::IDLE;
        if (ctx->async_job) {
            ctx->async_job->payload.clear();
            ctx->async_job_reusable = std::move(ctx->async_job);
        }
        ctx->async_node = nullptr;
    }
    lock.unlock();
    ctx->async_cv.notify_all();
}

// Acquire ctx->mutex with no async work in flight. The recheck under ctx->mutex
// closes the race with a submit that was already waiting for the lock.
static std::unique_lock<std::mutex> rkva_lock_ctx_drained(
        ggml_backend_remote_attn_context * ctx) {
    for (;;) {
        std::unique_lock<std::mutex> lock(ctx->mutex);
        bool busy = false;
        bool done = false;
        {
            std::lock_guard<std::mutex> alock(ctx->async_mutex);
            busy = ctx->async_state == rkva_async_state::PENDING ||
                   ctx->async_state == rkva_async_state::RUNNING;
            done = ctx->async_state == rkva_async_state::DONE_OK ||
                   ctx->async_state == rkva_async_state::DONE_ERR;
            if (done) {
                ctx->async_state = rkva_async_state::IDLE;
                if (ctx->async_job) {
                    ctx->async_job->payload.clear();
                    ctx->async_job_reusable = std::move(ctx->async_job);
                }
                ctx->async_node = nullptr;
            }
        }
        if (!busy) {
            if (done) {
                ctx->async_cv.notify_all();
            }
            return lock;
        }
        lock.unlock();
        rkva_async_drain(ctx);
    }
}

static bool rkva_async_wait_node(
        ggml_backend_remote_attn_context * ctx, ggml_tensor * node) {
    if (!node || !node->src[0] || node->src[0]->op != GGML_OP_REMOTE_ATTN ||
            !ggml_get_op_params_i32(node->src[0], GGML_REMOTE_ATTN_PARAM_ASYNC)) {
        return false;
    }
    std::unique_lock<std::mutex> lock(ctx->async_mutex);
    ctx->async_cv.wait(lock, [ctx] {
        return ctx->async_state != rkva_async_state::PENDING &&
               ctx->async_state != rkva_async_state::RUNNING;
    });
    ggml_tensor * source = node->src[0];
    bool ok = ctx->async_state == rkva_async_state::DONE_OK &&
              ctx->async_node == source &&
              node->data != nullptr && source->data != nullptr &&
              source->type == node->type &&
              source->ne[0] == node->ne[0] && source->ne[1] == node->ne[1] &&
              source->ne[2] == node->ne[2] && source->ne[3] == node->ne[3];
    if (ok) {
        const size_t row_bytes = size_t(node->ne[0]) * sizeof(float);
        for (int64_t row = 0; row < node->ne[1]; ++row) {
            std::memcpy(static_cast<uint8_t *>(node->data) + row * node->nb[1],
                    static_cast<const uint8_t *>(source->data) + row * source->nb[1],
                    row_bytes);
        }
    }
    ctx->async_state = rkva_async_state::IDLE;
    if (ctx->async_job) {
        ctx->async_job->payload.clear();
        ctx->async_job_reusable = std::move(ctx->async_job);
    }
    ctx->async_node = nullptr;
    lock.unlock();
    ctx->async_cv.notify_all();
    if (!ok) {
        std::lock_guard<std::mutex> ctx_lock(ctx->mutex);
        ctx->failed = true;
        ctx->connected = false;
        ctx->sock.close();
    }
    return ok;
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
        GGML_LOG_ERROR("%s: native KVarN4 requires original append sources and current resident ownership; "
                       "quantized snapshot bootstrap is unsupported\n", GGML_REMOTE_ATTN_NAME);
        return false;
    }
    try {
        if (!owner) {
            owner.reset(new rkva_kvarn_producer(ctx->range_origin, ctx->range_capacity,
                ctx->geo.n_head_kv, ctx->geo.sinkhorn_iters, ctx->native_store, ctx->native_tail_tokens));
        }
        std::vector<float> original_k(nq * width), original_v(nq * width);
        ggml_backend_tensor_get(keys, original_k.data(), 0, original_k.size() * sizeof(float));
        const float * dev_k = static_cast<const float *>(keys->data);
        const float * dev_v = static_cast<const float *>(values->data);
        if (!owner->prepare(original_k.data(), original_v.data(), ctx->pos_scratch.data(), uint32_t(nq), dev_k, dev_v)) {
            ctx->failed = true; return false;
        }
        // Descriptor revisions 1/2 select stage, sealed records and the exact
        // tail from the resident high-water mark. Uploading an entire query
        // batch first would change earlier queries across seals/tail eviction.
        // Revision 3 explicitly negotiates query-prefix semantics. Keep its
        // chunks within the qualified optimized kernel's nine-query bound;
        // prefill uses eight, verification batches keep their actual shape.
        const size_t chunk_limit = ctx->native_serial_prefix ? (nq > 9 ? 8 : nq) : 1;
        std::vector<uint8_t> bulk_payload;
        std::vector<float> bulk_queries;
        const uint32_t segments = uint32_t((nq + chunk_limit - 1) / chunk_limit);
        if (ctx->native_bulk) {
            if (ctx->native_generation > UINT64_MAX - segments) { ctx->failed = true; return false; }
            const rkva_kvarn_range_bulk header {1, segments, uint32_t(nq), 0};
            bulk_payload.resize(sizeof(header));
            std::memcpy(bulk_payload.data(), &header, sizeof(header));
            bulk_queries.resize(nq * out_width);
            ggml_backend_tensor_get(node->src[0], bulk_queries.data(), 0, bulk_queries.size() * sizeof(float));
        }
        uint64_t bulk_upload_bytes = 0;
        for (size_t first = 0; first < nq; first += chunk_limit) {
            const size_t chunk = std::min(chunk_limit, nq - first);
            if (!owner->append(original_k.data() + first * width, original_v.data() + first * width,
                    ctx->pos_scratch.data() + first, uint32_t(chunk), first == 0 && cached.rows == 0, ctx->payload_tx)) {
                ctx->failed = true; return false;
            }
            if (ctx->native_bulk) {
                const rkva_kvarn_range_bulk_segment segment {uint32_t(ctx->payload_tx.size()), uint32_t(chunk)};
                const bool bulk_f16 = ctx->range_wire == RKVA_WIRE_F16;
                const size_t q_elem_bytes = bulk_f16 ? sizeof(uint16_t) : sizeof(float);
                const rkva_range_attn attention {bulk_f16 ? RKVA_WIRE_F16 : RKVA_WIRE_F32, 0, {0, 0}};
                const size_t offset = bulk_payload.size();
                const size_t query_bytes = chunk * (sizeof(int32_t) + out_width * q_elem_bytes);
                bulk_payload.resize(offset + sizeof(segment) + ctx->payload_tx.size() + sizeof(attention) + query_bytes);
                auto * dst = bulk_payload.data() + offset;
                std::memcpy(dst, &segment, sizeof(segment)); dst += sizeof(segment);
                std::memcpy(dst, ctx->payload_tx.data(), ctx->payload_tx.size()); dst += ctx->payload_tx.size();
                std::memcpy(dst, &attention, sizeof(attention)); dst += sizeof(attention);
                std::memcpy(dst, ctx->pos_scratch.data() + first, chunk * sizeof(int32_t)); dst += chunk * sizeof(int32_t);
                if (bulk_f16) {
                    uint16_t * dst_f16 = reinterpret_cast<uint16_t *>(dst);
                    if (!rkva_f32_to_f16_batch(bulk_queries.data() + first * out_width, dst_f16, chunk * out_width)) {
                        ctx->failed = true; return false;
                    }
                } else {
                    std::memcpy(dst, bulk_queries.data() + first * out_width, chunk * out_width * sizeof(float));
                }
                bulk_upload_bytes += ctx->payload_tx.size() - sizeof(rkva_kvarn_range_upload);
                continue;
            }
            std::vector<uint8_t> reply;
            if (ctx->payload_tx.size() > (256u << 20) ||
                    !rkva_transact(ctx, RKVA_KVARN_RANGE_UPLOAD, ctx->active_session, layer, 0,
                        ctx->payload_tx.data(), uint32_t(ctx->payload_tx.size()), reply, 0) || !reply.empty()) {
                ctx->failed = true; return false;
            }
            if (ctx->native_generation == UINT64_MAX) { ctx->failed = true; return false; }
            ++ctx->native_generation;
            ++ctx->stats.range_upload_calls;
            ctx->stats.range_upload_rows += chunk;
            ctx->stats.range_upload_bytes += ctx->payload_tx.size() - sizeof(rkva_kvarn_range_upload);
            cached.epoch = ctx->range_epoch;
            cached.rows = owner->rows();
            const rkva_range_attn request {RKVA_WIRE_F32, 0, {0, 0}};
            const size_t query_bytes = chunk * (sizeof(int32_t) + out_width * sizeof(float));
            const size_t output_bytes = chunk * (out_width + ctx->geo.n_head) * sizeof(float);
            ctx->payload_tx.resize(sizeof(request) + query_bytes);
            std::memcpy(ctx->payload_tx.data(), &request, sizeof(request));
            std::memcpy(ctx->payload_tx.data() + sizeof(request), ctx->pos_scratch.data() + first, chunk * sizeof(int32_t));
            ggml_backend_tensor_get(node->src[0], ctx->payload_tx.data() + sizeof(request) + chunk * sizeof(int32_t),
                    first * out_width * sizeof(float), chunk * out_width * sizeof(float));
            if (!rkva_transact(ctx, RKVA_RANGE_ATTN, ctx->active_session, layer, uint32_t(chunk),
                    ctx->payload_tx.data(), uint32_t(ctx->payload_tx.size()), reply, uint32_t(output_bytes))) {
                ctx->failed = true; return false;
            }
            for (size_t offset = 0; offset < reply.size(); offset += sizeof(float)) {
                float value;
                std::memcpy(&value, reply.data() + offset, sizeof(value));
                if (!std::isfinite(value)) { ctx->failed = true; return false; }
            }
            ++ctx->stats.range_attn_calls;
            const size_t out_row_bytes = out_width * sizeof(float);
            const size_t lse_row_bytes = ctx->geo.n_head * sizeof(float);
            for (size_t row = 0; row < chunk; ++row) {
                auto * dst = static_cast<uint8_t *>(node->data) + (first + row) * node->nb[1];
                std::memcpy(dst, reply.data() + row * out_row_bytes, out_row_bytes);
                std::memcpy(dst + out_row_bytes,
                        reply.data() + chunk * out_row_bytes + row * lse_row_bytes, lse_row_bytes);
            }
        }
        if (ctx->native_bulk) {
            const bool bulk_f16 = ctx->range_wire == RKVA_WIRE_F16;
            std::vector<uint8_t> reply;
            const size_t out_elem_bytes = bulk_f16 ? sizeof(uint16_t) : sizeof(float);
            const size_t out_bytes = nq * out_width * out_elem_bytes;
            const size_t lse_bytes = nq * ctx->geo.n_head * sizeof(float);
            if (bulk_payload.size() > (256u << 20) ||
                !rkva_transact(ctx, RKVA_KVARN_RANGE_BULK_ATTN, ctx->active_session, layer, uint32_t(nq),
                    bulk_payload.data(), uint32_t(bulk_payload.size()), reply, uint32_t(out_bytes + lse_bytes))) {
                ctx->failed = true; return false;
            }
            if (bulk_f16) {
                const auto * src_f16 = reinterpret_cast<const uint16_t *>(reply.data());
                const auto * src_lse = reinterpret_cast<const float *>(reply.data() + out_bytes);
                for (size_t row = 0; row < nq; ++row) {
                    auto * dst = static_cast<float *>(static_cast<void *>(static_cast<uint8_t *>(node->data) + row * node->nb[1]));
                    if (!rkva_f16_to_f32_batch(src_f16 + row * out_width, dst, out_width)) {
                        ctx->failed = true; return false;
                    }
                    for (size_t h = 0; h < ctx->geo.n_head; ++h) {
                        const float lse_val = src_lse[row * ctx->geo.n_head + h];
                        if (!std::isfinite(lse_val)) { ctx->failed = true; return false; }
                        dst[out_width + h] = lse_val;
                    }
                }
            } else {
                for (size_t offset = 0; offset < reply.size(); offset += sizeof(float)) {
                    float value; std::memcpy(&value, reply.data() + offset, sizeof(value));
                    if (!std::isfinite(value)) { ctx->failed = true; return false; }
                }
                for (size_t row = 0; row < nq; ++row) {
                    auto * dst = static_cast<uint8_t *>(node->data) + row * node->nb[1];
                    std::memcpy(dst, reply.data() + row * out_width * sizeof(float), out_width * sizeof(float));
                    std::memcpy(dst + out_width * sizeof(float), reply.data() + out_bytes + row * ctx->geo.n_head * sizeof(float),
                                ctx->geo.n_head * sizeof(float));
                }
            }
            // The producer is speculative until the single ordered server ACK.
            // Any error above poisons the context; never publish partial history.
            ctx->native_generation += segments;
            cached.epoch = ctx->range_epoch;
            cached.rows = owner->rows();
            ctx->stats.range_upload_calls += segments;
            ctx->stats.range_upload_rows += nq;
            ctx->stats.range_upload_bytes += bulk_upload_bytes;
            ctx->stats.range_attn_calls += segments;
        }
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("%s: native KVarN4 append/query failed: %s\n", GGML_REMOTE_ATTN_NAME, error.what());
        ctx->failed = true;
        return false;
    } catch (...) {
        ctx->failed = true;
        return false;
    }
    if (ctx->stats.range_query_min < 0) ctx->stats.range_query_min = ctx->pos_scratch.front();
    ctx->stats.range_query_max = std::max<int64_t>(ctx->stats.range_query_max, ctx->pos_scratch.back());
    return cached.rows == live;
}

// Prepare the full native bulk payload on the scheduler thread (all tensor
// reads and the optional CUDA producer happen here, never on the worker).
// Only the ordered socket transaction is deferred.
static bool rkva_prepare_native_bulk_job(
        ggml_backend_remote_attn_context * ctx, ggml_tensor * node,
        uint32_t layer, uint32_t live, size_t nq, size_t out_width,
        rkva_async_job & job) {
    const auto * keys = node->src[5];
    const auto * values = node->src[6];
    const size_t width = size_t(ctx->geo.n_head_kv) * 256;
    if (!ctx->native_bulk || !keys || !values || keys->type != GGML_TYPE_F32 ||
            values->type != GGML_TYPE_F32 || !ggml_is_contiguous(keys) ||
            !ggml_is_contiguous(values) || !ggml_are_same_shape(keys, values) ||
            keys->ne[0] != 256 || keys->ne[1] != ctx->geo.n_head_kv ||
            keys->ne[2] != int64_t(nq) || keys->ne[3] != 1 || nq == 0 || nq > 256 ||
            !ctx->native_store ||
            (size_t) layer >= ctx->range_layers.size() ||
            (size_t) layer >= ctx->native_layers.size() || live == 0) {
        return false;
    }
    auto & cached = ctx->range_layers[(size_t) layer];
    auto & owner = ctx->native_layers[(size_t) layer];
    if ((owner && cached.epoch != ctx->range_epoch) ||
            uint32_t(live) != (owner ? owner->rows() : 0u) + nq) {
        GGML_LOG_ERROR("%s: async native KVarN4 requires original append sources and current resident ownership\n",
                GGML_REMOTE_ATTN_NAME);
        return false;
    }
    try {
        if (!owner) {
            owner.reset(new rkva_kvarn_producer(ctx->range_origin, ctx->range_capacity,
                    ctx->geo.n_head_kv, ctx->geo.sinkhorn_iters, ctx->native_store,
                    ctx->native_tail_tokens));
        }
        std::vector<float> original_k(nq * width), original_v(nq * width);
        ggml_backend_tensor_get(keys, original_k.data(), 0, original_k.size() * sizeof(float));
        ggml_backend_tensor_get(values, original_v.data(), 0, original_v.size() * sizeof(float));
        const float * dev_k = static_cast<const float *>(keys->data);
        const float * dev_v = static_cast<const float *>(values->data);
        if (!owner->prepare(original_k.data(), original_v.data(), ctx->pos_scratch.data(),
                    uint32_t(nq), dev_k, dev_v)) {
            ctx->failed = true;
            return false;
        }
        const size_t chunk_limit = ctx->native_serial_prefix ? (nq > 9 ? 8 : nq) : 1;
        const uint32_t segments = uint32_t((nq + chunk_limit - 1) / chunk_limit);
        if (ctx->native_generation > UINT64_MAX - segments) {
            ctx->failed = true;
            return false;
        }
        const bool bulk_f16 = ctx->range_wire == RKVA_WIRE_F16;
        const size_t q_elem_bytes = bulk_f16 ? sizeof(uint16_t) : sizeof(float);
        const rkva_kvarn_range_bulk header {1, segments, uint32_t(nq), 0};
        job.payload.resize(sizeof(header));
        std::memcpy(job.payload.data(), &header, sizeof(header));
        std::vector<float> bulk_queries(nq * out_width);
        ggml_backend_tensor_get(node->src[0], bulk_queries.data(), 0,
                bulk_queries.size() * sizeof(float));
        for (size_t first = 0; first < nq; first += chunk_limit) {
            const size_t chunk = std::min(chunk_limit, nq - first);
            if (!owner->append(original_k.data() + first * width,
                    original_v.data() + first * width,
                    ctx->pos_scratch.data() + first, uint32_t(chunk),
                    first == 0 && cached.rows == 0, ctx->payload_tx)) {
                ctx->failed = true;
                return false;
            }
            const rkva_kvarn_range_bulk_segment segment {
                uint32_t(ctx->payload_tx.size()), uint32_t(chunk)};
            const rkva_range_attn attention {bulk_f16 ? RKVA_WIRE_F16 : RKVA_WIRE_F32, 0, {0, 0}};
            const size_t offset = job.payload.size();
            const size_t query_bytes = chunk * (sizeof(int32_t) + out_width * q_elem_bytes);
            job.payload.resize(offset + sizeof(segment) + ctx->payload_tx.size() +
                    sizeof(attention) + query_bytes);
            auto * dst = job.payload.data() + offset;
            std::memcpy(dst, &segment, sizeof(segment)); dst += sizeof(segment);
            std::memcpy(dst, ctx->payload_tx.data(), ctx->payload_tx.size());
            dst += ctx->payload_tx.size();
            std::memcpy(dst, &attention, sizeof(attention)); dst += sizeof(attention);
            std::memcpy(dst, ctx->pos_scratch.data() + first, chunk * sizeof(int32_t));
            dst += chunk * sizeof(int32_t);
            if (bulk_f16) {
                uint16_t * dst_f16 = reinterpret_cast<uint16_t *>(dst);
                if (!rkva_f32_to_f16_batch(bulk_queries.data() + first * out_width, dst_f16, chunk * out_width)) {
                    ctx->failed = true; return false;
                }
            } else {
                std::memcpy(dst, bulk_queries.data() + first * out_width,
                        chunk * out_width * sizeof(float));
            }
            job.upload_bytes += ctx->payload_tx.size() - sizeof(rkva_kvarn_range_upload);
        }
        if (job.payload.size() > (256u << 20)) {
            ctx->failed = true;
            return false;
        }
        job.output = static_cast<uint8_t *>(node->data);
        job.output_stride = node->nb[1];
        job.layer = uint32_t(layer);
        job.nq = uint32_t(nq);
        job.segments = segments;
        job.out_width = uint32_t(out_width);
        job.first_pos = ctx->pos_scratch.front();
        job.last_pos = ctx->pos_scratch.back();
        job.bulk_f16 = bulk_f16;
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("%s: async native KVarN4 preparation failed: %s\n",
                GGML_REMOTE_ATTN_NAME, error.what());
        ctx->failed = true;
        return false;
    } catch (...) {
        ctx->failed = true;
        return false;
    }
    return true;
}

// Caller holds ctx->mutex. Submit at most one bound job; any unconsumed or
// in-flight slot is a graph ordering bug and fails closed.
static bool rkva_async_submit_locked(
        ggml_backend_remote_attn_context * ctx, ggml_tensor * node,
        uint32_t layer, uint32_t live, size_t nq, size_t out_width) {
    {
        std::lock_guard<std::mutex> alock(ctx->async_mutex);
        if (ctx->async_state != rkva_async_state::IDLE) {
            GGML_LOG_ERROR("%s: async range submitted while a previous job is not consumed\n",
                    GGML_REMOTE_ATTN_NAME);
            ctx->failed = true;
            return false;
        }
    }
    if (!ctx->async_worker.joinable()) {
        try {
            ctx->async_worker = std::thread(rkva_async_worker_main, ctx);
        } catch (...) {
            ctx->failed = true;
            return false;
        }
    }
    std::unique_ptr<rkva_async_job> job;
    if (ctx->async_job_reusable) {
        job = std::move(ctx->async_job_reusable);
        job->payload.clear();
        job->output = nullptr;
        job->output_stride = 0;
        job->upload_bytes = 0;
    } else {
        try {
            job = std::make_unique<rkva_async_job>();
            job->payload.reserve(8u << 20);
        } catch (...) {
            ctx->failed = true;
            return false;
        }
    }
    if (!rkva_prepare_native_bulk_job(ctx, node, layer, live, nq, out_width, *job)) {
        ctx->failed = true;
        return false;
    }
    {
        std::lock_guard<std::mutex> alock(ctx->async_mutex);
        if (ctx->async_state != rkva_async_state::IDLE) {
            ctx->failed = true;
            return false;
        }
        ctx->async_state = rkva_async_state::PENDING;
        ctx->async_job = std::move(job);
        ctx->async_node = node;
    }
    ctx->async_cv.notify_all();
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
        if (ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_ASYNC)) {
            // The scheduler executes local attention between this submit and
            // the companion wait node; only the ordered bulk RPC is deferred.
            return ctx->async_local_remote && rkva_async_submit_locked(
                    ctx, node, uint32_t(layer), count, nq, out_width);
        }
        try {
            if (rkva_run_native_range(ctx, node, uint32_t(layer), count, nq, out_width)) return true;
        } catch (...) {
            // An original-domain append may have advanced the native owner or
            // GPU before allocation/transport failed. Never resume implicitly.
        }
        ctx->failed = true;
        return false;
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
    const bool split_queries = ctx->prefill_query_chunk != 0 && nq > 9;
    if (output_bytes > (256u << 20) || query_bytes + sizeof(rkva_range_attn) > (256u << 20)) {
        return false;
    }
    std::vector<uint8_t> reply;
    bool attended = false;
    while (start < count) {
        const uint32_t rows = std::min(chunk_rows, count - start);
        rkva_range_upload upload { start, rows, ctx->range_wire, 0 };
        const bool combine = !split_queries && ctx->range_fused && start + rows == count &&
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
                if (native_quant || ctx->range_source_type == GGML_TYPE_F16) {
                    // Direct F16 is already the wire representation. Preserve
                    // its bits; never route original F16 through a Q4 decoder.
                    ggml_backend_tensor_get(source, dst, (size_t) row * source->nb[1], row_bytes);
                    dst += row_bytes;
                    continue;
                }
                ggml_backend_tensor_get(source, packed.data(), (size_t) row * source->nb[1], row_bytes);
                to_float(packed.data(), decoded.data(), width);
                if (!rkva_f32_to_f16_batch(decoded.data(), reinterpret_cast<uint16_t *>(dst), width)) {
                    ctx->failed = true;
                    return false;
                }
                dst += width * sizeof(uint16_t);
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
    if (split_queries) {
        // Keep the local CUDA graph's large batch, but bound remote query
        // batches to the console's optimized decode kernels. Upload the whole
        // KV delta once; per-query absolute positions mask its future rows.
        // Native KVarN owners use the separate path above, without this option.
        for (size_t first = 0; first < nq; first += ctx->prefill_query_chunk) {
            const size_t queries = std::min<size_t>(ctx->prefill_query_chunk, nq - first);
            const rkva_range_attn request {RKVA_WIRE_F32, 0, {0, 0}};
            const size_t bytes = queries * (sizeof(int32_t) + out_width * sizeof(float));
            ctx->payload_tx.resize(sizeof(request) + bytes);
            std::memcpy(ctx->payload_tx.data(), &request, sizeof(request));
            std::memcpy(ctx->payload_tx.data() + sizeof(request), ctx->pos_scratch.data() + first,
                        queries * sizeof(int32_t));
            ggml_backend_tensor_get(q, ctx->payload_tx.data() + sizeof(request) + queries * sizeof(int32_t),
                                    first * out_width * sizeof(float), queries * out_width * sizeof(float));
            const size_t result_bytes = queries * (out_width + ctx->geo.n_head) * sizeof(float);
            if (!rkva_transact(ctx, RKVA_RANGE_ATTN, ctx->active_session, uint32_t(layer), uint32_t(queries),
                    ctx->payload_tx.data(), uint32_t(ctx->payload_tx.size()), reply, uint32_t(result_bytes))) {
                return false;
            }
            for (size_t offset = 0; offset < reply.size(); offset += sizeof(float)) {
                float value;
                std::memcpy(&value, reply.data() + offset, sizeof(value));
                if (!std::isfinite(value)) { return false; }
            }
            auto * dst = static_cast<uint8_t *>(node->data) + first * node->nb[1];
            for (size_t query = 0; query < queries; ++query) {
                std::memcpy(dst, reply.data() + query * out_width * sizeof(float), out_width * sizeof(float));
                dst += out_width * sizeof(float);
                std::memcpy(dst, reply.data() + (queries * out_width + query * ctx->geo.n_head) * sizeof(float),
                            ctx->geo.n_head * sizeof(float));
                dst += ctx->geo.n_head * sizeof(float);
            }
            ++ctx->stats.range_attn_calls;
        }
        if (ctx->stats.range_query_min < 0 || ctx->pos_scratch.front() < ctx->stats.range_query_min) {
            ctx->stats.range_query_min = ctx->pos_scratch.front();
        }
        ctx->stats.range_query_max = std::max<int64_t>(ctx->stats.range_query_max, ctx->pos_scratch.back());
        return true;
    }
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
        auto emit16 = [&](const ggml_tensor * t, size_t elems) -> bool {
            ggml_backend_tensor_get(t, ctx->f32_scratch.data(), 0, elems * 4);
            uint16_t * dst = (uint16_t *) p;
            if (!rkva_f32_to_f16_batch(ctx->f32_scratch.data(), dst, elems)) {
                return false;
            }
            p += elems * 2;
            return true;
        };
        if (!emit16(q, q_elems) || !emit16(k, kv_elems) || !emit16(v, kv_elems)) {
            return false;
        }
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
        if (!rkva_f16_to_f32_batch(s, dst, out_elems)) {
            return false;
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
    rkva_async_drain(ctx);
    {
        std::lock_guard<std::mutex> alock(ctx->async_mutex);
        ctx->async_stop = true;
    }
    ctx->async_cv.notify_all();
    if (ctx->async_worker.joinable()) {
        ctx->async_worker.join();
    }
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
    const char * prefill_chunk = std::getenv("GGML_RKVA_PREFILL_QUERY_CHUNK");
    if (prefill_chunk && std::strcmp(prefill_chunk, "0") != 0) {
        if (std::strcmp(prefill_chunk, "8") != 0) {
            GGML_LOG_ERROR("%s: GGML_RKVA_PREFILL_QUERY_CHUNK must be 0 or 8\n", GGML_REMOTE_ATTN_NAME);
            delete ctx;
            return nullptr;
        }
        ctx->prefill_query_chunk = 8;
    }
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
    auto lock = rkva_lock_ctx_drained(ctx);
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
    auto lock = rkva_lock_ctx_drained(ctx);
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
    ctx->native_serial_prefix = false;
    ctx->native_bulk = false;
    ctx->async_local_remote = false;
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
    hello.proto_flags    = (1u << 0) | (1u << 1) | RKVA_PROTO_NATIVE_REWIND_V2 |
                           RKVA_PROTO_NATIVE_SERIAL_PREFIX_V3;
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
    auto lock = rkva_lock_ctx_drained(ctx);
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
    auto lock = rkva_lock_ctx_drained(ctx);
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
    auto lock = rkva_lock_ctx_drained(ctx);
    if (!ctx->connected || ctx->failed) {
        return false;
    }
    ++ctx->range_epoch;
    std::vector<uint8_t> out;
    if (!rkva_transact(ctx, RKVA_RESET, session_id, 0, 0, nullptr, 0, out, 0) || !out.empty()) {
        ctx->failed = true;
        return false;
    }
    if (ctx->range_native_kvarn) {
        if (ctx->native_generation == UINT64_MAX) { ctx->failed = true; return false; }
        ++ctx->native_generation;
        for (auto & owner : ctx->native_layers) owner.reset();
        ctx->range_layers.assign(ctx->native_layers.size(), {});
    }
    return true;
}

// Caller holds ctx->mutex. A successful suffix-removal/owner-bound partial
// restore guarantees bytes below pos0 are unchanged. Rewind is session-wide;
// adjust every layer only after ACK so later layers cannot retrim earlier ones.
static bool rkva_trim_range_locked(ggml_backend_remote_attn_context * ctx,
        uint32_t session_id, int32_t pos0) {
    if (session_id != ctx->active_session || pos0 < 0) return false;
    const uint32_t keep = uint32_t(pos0) <= ctx->range_origin ? 0 :
            std::min(uint32_t(pos0) - ctx->range_origin, ctx->range_capacity);
    bool needed = false;
    for (const auto & layer : ctx->range_layers) {
        needed |= layer.epoch == ctx->range_epoch && layer.rows > keep;
    }
    if (!needed) return true;
    if (!(ctx->ack.supported_ops & (1u << RKVA_REWIND))) {
        ++ctx->range_epoch; // older server: authoritative full-upload fallback
        return true;
    }
    rkva_trim_req request { pos0 };
    std::vector<uint8_t> reply;
    if (!rkva_transact(ctx, RKVA_REWIND, session_id, 0, 0,
                &request, sizeof(request), reply, 0)) {
        ctx->failed = true; // ACK failure cannot publish speculative coverage
        return false;
    }
    for (auto & layer : ctx->range_layers) {
        if (layer.epoch == ctx->range_epoch) layer.rows = std::min(layer.rows, keep);
    }
    return true;
}

// Caller owns the context mutex. Never publish host coverage before the
// session-wide GPU tail repair has completed and its ACK was validated.
static bool rkva_native_rewind_locked(ggml_backend_remote_attn_context * ctx,
        uint32_t session_id, int32_t pos0) {
    if (session_id != ctx->active_session || pos0 < 0) return false;
    const uint32_t keep = uint32_t(pos0) <= ctx->range_origin ? 0 :
            std::min(uint32_t(pos0) - ctx->range_origin, ctx->range_capacity);
    bool needed = false;
    for (size_t il = 0; il < ctx->native_layers.size(); ++il) {
        const auto & owner = ctx->native_layers[il];
        if (!owner) continue;
        if (ctx->range_layers[il].epoch != ctx->range_epoch) return false;
        if (owner->rows() > keep) {
            needed = true;
            if (!owner->can_rewind(keep)) return false;
        }
    }
    if (!needed) return true;
    if (!(ctx->ack.reserved[0] & RKVA_CAP_KVARN_REWIND) ||
            !(ctx->ack.supported_ops & (1u << RKVA_KVARN_RANGE_REWIND))) return false;
    if (!rkva_kvarn_generation_mutable(ctx->native_generation)) return false;
    try {
        const rkva_kvarn_range_rewind request {ctx->native_generation, uint32_t(pos0), uint32_t(ctx->native_layers.size()), ctx->native_tail_tokens, 0};
        std::vector<uint8_t> payload(sizeof(request));
        std::memcpy(payload.data(), &request, sizeof(request));
        for (const auto & owner : ctx->native_layers) {
            const uint32_t old = owner ? owner->rows() : 0;
            const rkva_kvarn_layer_rewind layer {old, std::min(old, keep)};
            const auto * bytes = reinterpret_cast<const uint8_t *>(&layer);
            payload.insert(payload.end(), bytes, bytes + sizeof(layer));
            if (owner && !owner->rewind_tail(layer.new_live, payload)) return false;
        }
        std::vector<uint8_t> reply;
        if (payload.size() > (256u << 20) || !rkva_transact(ctx, RKVA_KVARN_RANGE_REWIND,
                    session_id, 0, 0, payload.data(), uint32_t(payload.size()), reply, 0) || !reply.empty()) {
            ctx->failed = true;
            return false;
        }
        ++ctx->native_generation;
        for (size_t il = 0; il < ctx->native_layers.size(); ++il) {
            auto & owner = ctx->native_layers[il];
            if (owner && !owner->rewind(std::min(owner->rows(), keep))) {
                ctx->failed = true;
                return false;
            }
            ctx->range_layers[il].rows = owner ? owner->rows() : 0;
        }
        return true;
    } catch (...) {
        ctx->failed = true;
        return false;
    }
}

bool ggml_backend_remote_attn_trim(ggml_backend_t backend, uint32_t session_id, int32_t pos0) {
    if (!backend) return false;
    if (ggml_backend_is_local_split(backend)) {
        return ggml_backend_local_split_trim(backend, session_id, pos0);
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    auto lock = rkva_lock_ctx_drained(ctx);
    if (!ctx->connected || ctx->failed) {
        return false;
    }
    if (ctx->range_configured) {
        if (ctx->range_native_kvarn) {
            return rkva_native_rewind_locked(ctx, session_id, pos0);
        }
        return rkva_trim_range_locked(ctx, session_id, pos0);
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

static bool rkva_async_local_remote_option(bool & async_overlap) {
    async_overlap = false;
    if (const char * option = std::getenv("GGML_RKVA_ASYNC_LOCAL_REMOTE")) {
        if (!std::strcmp(option, "1")) async_overlap = true;
        else if (std::strcmp(option, "0")) async_overlap = false;
        else {
            GGML_LOG_ERROR("%s: GGML_RKVA_ASYNC_LOCAL_REMOTE must be 0 or 1\n",
                    GGML_REMOTE_ATTN_NAME);
            return false;
        }
    }
    return true;
}

static bool rkva_configure_native_kvarn(ggml_backend_remote_attn_context * ctx,
        uint32_t origin, uint32_t capacity, ggml_type source_type, bool async_overlap) {
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
    uint32_t precision_tail_tokens = 128;
    if (const char * tail = std::getenv("GGML_RKVA_NATIVE_TAIL_TOKENS")) {
        if (!std::strcmp(tail, "0")) precision_tail_tokens = 0;
        else if (std::strcmp(tail, "128")) {
            GGML_LOG_ERROR("%s: GGML_RKVA_NATIVE_TAIL_TOKENS must be 0 or 128\n", GGML_REMOTE_ATTN_NAME);
            return false;
        }
    }
    const bool rewind_v2 = (ctx->ack.reserved[0] & RKVA_CAP_KVARN_REWIND) &&
                           (ctx->ack.supported_ops & (1u << RKVA_KVARN_RANGE_REWIND));
    const bool serial_prefix = rewind_v2 && (ctx->ack.reserved[0] & RKVA_CAP_KVARN_SERIAL_PREFIX);
    bool bulk = false;
    rkva_wire bulk_wire = RKVA_WIRE_F32;
    if (const char * option = std::getenv("GGML_RKVA_NATIVE_BULK")) {
        if (!std::strcmp(option, "1")) {
            bulk = true;
        } else if (!std::strcmp(option, "f16") || !std::strcmp(option, "fp16")) {
            bulk = true;
            bulk_wire = RKVA_WIRE_F16;
        } else if (std::strcmp(option, "0")) {
            return false;
        }
    }
    if (const char * wire_env = std::getenv("GGML_RKVA_NATIVE_BULK_WIRE")) {
        if (!std::strcmp(wire_env, "f16") || !std::strcmp(wire_env, "fp16")) {
            bulk_wire = RKVA_WIRE_F16;
        } else if (!std::strcmp(wire_env, "f32") || !std::strcmp(wire_env, "fp32")) {
            bulk_wire = RKVA_WIRE_F32;
        } else {
            return false;
        }
    }
    if (bulk && (!serial_prefix || !(ctx->ack.reserved[0] & RKVA_CAP_KVARN_BULK_ATTN) ||
                 !(ctx->ack.supported_ops & (1u << RKVA_KVARN_RANGE_BULK_ATTN)))) {
        GGML_LOG_ERROR("%s: requested native bulk RPC is not supported by the negotiated server\n", GGML_REMOTE_ATTN_NAME);
        return false;
    }
    if (bulk && bulk_wire == RKVA_WIRE_F16 && !(ctx->ack.reserved[0] & RKVA_CAP_KVARN_BULK_F16)) {
        GGML_LOG_ERROR("%s: requested native bulk FP16 RPC is not supported by the negotiated server\n", GGML_REMOTE_ATTN_NAME);
        return false;
    }
    if (async_overlap && !bulk) {
        GGML_LOG_ERROR("%s: GGML_RKVA_ASYNC_LOCAL_REMOTE requires "
                "GGML_RKVA_NATIVE_BULK=1 with a bulk-capable native KVarN4 server\n",
                GGML_REMOTE_ATTN_NAME);
        return false;
    }
    const rkva_kvarn_range_config descriptor {
        origin, capacity, RKVA_RANGE_STORAGE_KVARN4, serial_prefix ? 3u : rewind_v2 ? 2u : 1u,
        128, 128, 256, 2, 4, 4, 8960, 3, 2, 128, precision_tail_tokens, 1, 1, 1};
    std::vector<uint8_t> reply;
    if (!rkva_transact(ctx, RKVA_KVARN_RANGE_CONFIG, ctx->active_session, 0, 0,
            &descriptor, sizeof(descriptor), reply, rewind_v2 ? sizeof(uint64_t) : 0) ||
            reply.size() != (rewind_v2 ? sizeof(uint64_t) : 0)) return false;
    ctx->native_generation = 0;
    if (rewind_v2) {
        std::memcpy(&ctx->native_generation, reply.data(), sizeof(uint64_t));
        if (!rkva_kvarn_generation_mutable(ctx->native_generation)) { ctx->failed = true; return false; }
    }
    ctx->range_configured = true;
    ctx->range_native_kvarn = true;
    ctx->native_serial_prefix = serial_prefix;
    ctx->native_bulk = bulk;
    ctx->async_local_remote = async_overlap;
    ctx->native_tail_tokens = precision_tail_tokens;
    ctx->native_store = store;
    ctx->native_layers.clear();
    ctx->native_layers.resize(ctx->geo.n_layer_remote);
    ctx->range_source_type = source_type;
    ctx->range_wire = bulk ? bulk_wire : RKVA_WIRE_F32; // Q/OUT only; op16 carries structured native bytes
    ctx->range_origin = origin;
    ctx->range_capacity = capacity;
    ++ctx->range_epoch;
    ctx->range_layers.assign(ctx->geo.n_layer_remote, {});
    ctx->range_fused = false;
    ctx->range_profile = profile;
    ctx->range_kernel_mode = mode;
    if (!rkva_range_apply_control(ctx, "kvarn4", generic)) return false;
    GGML_LOG_WARN("%s: EXPERIMENTAL native KVarN4 precision_tail=%u descriptor=%u serial_prefix=%d async_overlap=%d; original F32 producer, bounded resident rewind requires negotiated capability; FULL snapshots unsupported\n",
                  GGML_REMOTE_ATTN_NAME, precision_tail_tokens, descriptor.layout_version, int(serial_prefix), int(async_overlap));
    return true;
}

static const char * rkva_get_upload_format() {
    if (const char * fmt = std::getenv("GGML_RKVA_UPLOAD_FORMAT")) return fmt;
    if (const char * fmt = std::getenv("GGML_RKVA_FORMAT")) return fmt;
    if (const char * fmt = std::getenv("GGML_RKVA_RANGE_STORAGE")) return fmt;
    return nullptr;
}

bool ggml_backend_remote_attn_configure_range(
        ggml_backend_t backend, uint32_t origin, uint32_t capacity, ggml_type cache_type) {
    if (!backend || ggml_backend_is_local_split(backend)) {
        return false;
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    auto lock = rkva_lock_ctx_drained(ctx);
    bool async_overlap = false;
    if (!rkva_async_local_remote_option(async_overlap)) return false;
    const char * format_req = rkva_get_upload_format();
    if (format_req && (!std::strcmp(format_req, "kvarn4") || !std::strcmp(format_req, "kvarn"))) {
        return rkva_configure_native_kvarn(ctx, origin, capacity, cache_type, async_overlap);
    }
    if (async_overlap) {
        GGML_LOG_ERROR("%s: GGML_RKVA_ASYNC_LOCAL_REMOTE requires "
                "upload format kvarn4\n", GGML_REMOTE_ATTN_NAME);
        return false;
    }
    constexpr uint32_t required = (1u << RKVA_RANGE_CONFIG) | (1u << RKVA_RANGE_UPLOAD) | (1u << RKVA_RANGE_ATTN);
    if (!ctx->connected || ctx->failed || ctx->active_session == RKVA_SESSION_NONE ||
            (ctx->ack.supported_ops & required) != required || capacity == 0 ||
            origin > INT32_MAX || capacity > (uint32_t) INT32_MAX - origin ||
            ctx->geo.max_ctx != capacity || ctx->geo.n_layer_remote == 0 ||
            ctx->geo.cache_bits_k != 0 || ctx->geo.cache_bits_v != 0 ||
            ctx->geo.domain != GGML_REMOTE_ATTN_DOMAIN_AUTO || ctx->geo.has_sinks || ctx->geo.swa ||
            (cache_type != GGML_TYPE_Q4_0 && cache_type != GGML_TYPE_Q8_0 && cache_type != GGML_TYPE_F16)) {
        return false;
    }
    const bool source_q4 = cache_type == GGML_TYPE_Q4_0;
    const uint64_t capability = source_q4 ? RKVA_CAP_RANGE_Q4_0 : RKVA_CAP_RANGE_Q8_0;
    const bool supports_native = cache_type != GGML_TYPE_F16 && (ctx->ack.reserved[0] & capability) != 0;
    const char * requested = format_req;
    bool native_quant = supports_native;
    if (requested && std::strcmp(requested, "auto") != 0) {
        if (!std::strcmp(requested, "f16") || !std::strcmp(requested, "fp16")) {
            native_quant = false;
        } else if (!std::strcmp(requested, "f32") || !std::strcmp(requested, "fp32")) {
            native_quant = false;
        } else if (!std::strcmp(requested, "q4_0") || !std::strcmp(requested, "q4")) {
            if (std::strcmp(ggml_type_name(cache_type), "q4_0") != 0 || !supports_native) {
                GGML_LOG_ERROR("%s: upload format %s requires a matching host cache and server capability\n",
                        GGML_REMOTE_ATTN_NAME, requested);
                return false;
            }
            native_quant = true;
        } else if (!std::strcmp(requested, "q8_0") || !std::strcmp(requested, "q8")) {
            if (std::strcmp(ggml_type_name(cache_type), "q8_0") != 0 || !supports_native) {
                GGML_LOG_ERROR("%s: upload format %s requires a matching host cache and server capability\n",
                        GGML_REMOTE_ATTN_NAME, requested);
                return false;
            }
            native_quant = true;
        } else {
            GGML_LOG_ERROR("%s: invalid upload format '%s' (expected auto, kvarn4/kvarn, fp16/f16, fp32/f32, q4/q4_0 or q8/q8_0)\n",
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
    ctx->native_serial_prefix = false;
    ctx->native_bulk = false;
    ctx->async_local_remote = false;
    ctx->native_layers.clear();
    const bool wire_f32 = requested && (!std::strcmp(requested, "f32") || !std::strcmp(requested, "fp32"));
    ctx->range_source_type = cache_type;
    ctx->range_wire = wire_f32 ? RKVA_WIRE_F32 : (native_quant ? (source_q4 ? RKVA_WIRE_Q4_0 : RKVA_WIRE_Q8_0) : RKVA_WIRE_F16);
    ctx->range_origin = origin;
    ctx->range_capacity = capacity;
    ++ctx->range_epoch;
    ctx->range_layers.assign(ctx->geo.n_layer_remote, {});
    ctx->range_fused = fused;
    ctx->range_profile = profile;
    ctx->range_kernel_mode = kernel_mode;
    const char * storage_name = wire_f32 ? "f32" : (native_quant ? (source_q4 ? "q4_0" : "q8_0") : "f16");
    if (!rkva_range_apply_control(ctx, storage_name, generic_kernel)) return false;
    if (cache_type == GGML_TYPE_F16) {
        GGML_LOG_INFO("%s: range source=f16 storage=f16; direct original K/V half bytes, no quantized intermediate\n",
                      GGML_REMOTE_ATTN_NAME);
    } else if (!native_quant) {
        GGML_LOG_INFO("%s: range source=%s storage=f16; quantized-source expansion does not recover precision\n",
                      GGML_REMOTE_ATTN_NAME, ggml_type_name(cache_type));
    }
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
    auto lock = rkva_lock_ctx_drained(ctx);
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
    auto lock = rkva_lock_ctx_drained(ctx);
    ++ctx->range_epoch;
}

bool ggml_backend_remote_attn_native_kvarn_enabled(ggml_backend_t backend) {
    if (!backend || ggml_backend_is_local_split(backend)) return false;
    const auto * ctx = static_cast<ggml_backend_remote_attn_context *>(backend->context);
    return ctx->range_configured && ctx->range_native_kvarn;
}

bool ggml_backend_remote_attn_async_enabled(ggml_backend_t backend) {
    if (!backend || ggml_backend_is_local_split(backend)) return false;
    const auto * ctx = static_cast<const ggml_backend_remote_attn_context *>(backend->context);
    return ctx->async_local_remote && ctx->connected && !ctx->failed &&
           ctx->range_configured && ctx->range_native_kvarn && ctx->native_bulk;
}

bool ggml_backend_remote_attn_native_kvarn_can_rewind(ggml_backend_t backend) {
    if (!ggml_backend_remote_attn_native_kvarn_enabled(backend)) return false;
    const auto * ctx = static_cast<const ggml_backend_remote_attn_context *>(backend->context);
    return ctx->connected && !ctx->failed && rkva_kvarn_generation_mutable(ctx->native_generation) &&
           (ctx->ack.reserved[0] & RKVA_CAP_KVARN_REWIND) &&
           (ctx->ack.supported_ops & (1u << RKVA_KVARN_RANGE_REWIND));
}

bool ggml_backend_remote_attn_native_kvarn_can_restore(ggml_backend_t backend, int32_t pos0) {
    if (!ggml_backend_remote_attn_native_kvarn_can_rewind(backend) || pos0 < 0) return false;
    auto * ctx = static_cast<ggml_backend_remote_attn_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->mutex);
    const uint32_t keep = uint32_t(pos0) <= ctx->range_origin ? 0 : uint32_t(pos0) - ctx->range_origin;
    if (keep > ctx->range_capacity || !rkva_kvarn_generation_mutable(ctx->native_generation)) return false;
    for (size_t il = 0; il < ctx->native_layers.size(); ++il) {
        const auto & owner = ctx->native_layers[il];
        if (owner && ctx->range_layers[il].epoch != ctx->range_epoch) return false;
        if (keep && (!owner || owner->rows() < keep || !owner->can_rewind(keep))) return false;
        if (owner && !owner->can_rewind(keep)) return false;
    }
    return true;
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

void ggml_remote_attn_range_set_async(ggml_tensor * node) {
    GGML_ASSERT(node && node->op == GGML_OP_REMOTE_ATTN);
    GGML_ASSERT(ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_RANGE));
    GGML_ASSERT(!ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_WAIT));
    ggml_set_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_ASYNC, 1);
}

ggml_tensor * ggml_remote_attn_range_wait(
        ggml_context * ctx, ggml_tensor * range) {
    GGML_ASSERT(ctx && range && range->op == GGML_OP_REMOTE_ATTN);
    GGML_ASSERT(ggml_get_op_params_i32(range, GGML_REMOTE_ATTN_PARAM_RANGE));
    GGML_ASSERT(ggml_get_op_params_i32(range, GGML_REMOTE_ATTN_PARAM_ASYNC));
    ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
            range->ne[0], range->ne[1]);
    result->op = GGML_OP_REMOTE_ATTN;
    result->src[0] = range;
    std::memcpy(result->op_params, range->op_params, sizeof(result->op_params));
    ggml_set_op_params_i32(result, GGML_REMOTE_ATTN_PARAM_ASYNC, 0);
    ggml_set_op_params_i32(result, GGML_REMOTE_ATTN_PARAM_WAIT, 1);
    return result;
}

void ggml_backend_remote_attn_set_active_session(ggml_backend_t backend, uint32_t session_id) {
    if (!backend) return;
    if (ggml_backend_is_local_split(backend)) {
        ggml_backend_local_split_set_active_session(backend, session_id);
        return;
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    auto lock = rkva_lock_ctx_drained(ctx);
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
    bool async_idle = false;
    {
        std::lock_guard<std::mutex> alock(ctx->async_mutex);
        async_idle = ctx->async_state == rkva_async_state::IDLE;
    }
    if (async_idle && ctx->range_profile_controlled && ctx->connected && !ctx->failed) {
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
        "\"range_cache_type\":\"%s\",\"range_source_cache_type\":\"%s\",\"range_upload_bytes\":%llu,\"native_precision_tail_tokens\":%u}",
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
        (unsigned long long) s.range_upload_bytes, ctx->range_native_kvarn ? ctx->native_tail_tokens : 0);
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
    if (ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_WAIT)) {
        return rkva_async_wait_node(ctx, node);
    }
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
