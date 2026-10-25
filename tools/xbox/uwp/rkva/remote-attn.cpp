// RKVA remote KVarN attention service — Xbox Series X side.
//
// Control-plane listener skeleton. The attention data plane (D3D12 KVarN
// store + flash-attention-over-records kernels) is implemented separately in
// remote-attn-engine.* and wired in once it passes the kernel-parity gate; until
// then ATTN_DECODE/ATTN_PREFILL are deliberately NOT advertised in HELLO and
// return RKVA_ERR_UNSUPPORTED_OP, so the host fails fast instead of consuming
// silently-wrong attention.
//
// Modeled on the proven DFL2 dedicated server (dedicated.cpp): single persistent
// client, strict request/response ordering, jsonl event log, TCP_NODELAY.

#include "remote-attn.h"
#include "remote-attn-engine.h"
#include "remote-attn-protocol.h"
#include "rkva-bulk-scratch.h"
#include "rkva-kvarn-bulk.h"
#include "rkva-range-profile.h"

#ifndef NOMINMAX
    #define NOMINMAX
#endif

#include <immintrin.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include "xllama/session_hub.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#pragma comment(lib, "Ws2_32.lib")

namespace {

void evict_local_session() {
    std::lock_guard<std::mutex> lk(xllama::session_hub().mtx);
    xllama::session_hub().reset_locked();
}

using clock_type = std::chrono::steady_clock;

static inline float f16_to_f32(uint16_t h) {
    __m128i in = _mm_cvtsi32_si128(static_cast<int>(h));
    __m128 out = _mm_cvtph_ps(in);
    return _mm_cvtss_f32(out);
}

static inline uint16_t f32_to_f16(float f) {
    __m128 in = _mm_set_ss(f);
    __m128i out = _mm_cvtps_ph(in, 0);
    return static_cast<uint16_t>(_mm_cvtsi128_si32(out));
}

static inline void convert_f16_to_f32(const uint16_t* src, float* dst, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256 out = _mm256_cvtph_ps(in);
        _mm256_storeu_ps(dst + i, out);
    }
    for (; i < count; ++i) {
        dst[i] = f16_to_f32(src[i]);
    }
}

static inline void convert_f32_to_f16(const float* src, uint16_t* dst, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 in = _mm256_loadu_ps(src + i);
        __m128i out = _mm256_cvtps_ph(in, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), out);
    }
    for (; i < count; ++i) {
        dst[i] = f32_to_f16(src[i]);
    }
}

uint64_t elapsed_us(clock_type::time_point start) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(clock_type::now() -
                                                                           start)
        .count();
}

void event(const std::string& path, const std::string& message) {
    std::ofstream out(path, std::ios::app);
    if (out) {
        out << message << "\n";
    }
}

// Log path for the active connection, so handle_request (which has no path
// argument) can record engine/GPU errors.
std::string g_log_path;
std::string g_snapshot_local_folder;

void event_engine_error(const std::string& message) {
    std::string safe;
    safe.reserve(message.size());
    for (char c : message) {
        if (c == '"' || c == '\\') {
            safe += '\\';
        }
        safe += (c == '\n' || c == '\r') ? ' ' : c;
    }
    event(g_log_path, "{\"phase\":\"engine_error\",\"error\":\"" + safe + "\"}");
}

static int q4_bad(const char* what, uint32_t segment, uint64_t at, uint64_t total) {
    event_engine_error("native bulk envelope rejected: " + std::string(what) +
                       " segment=" + std::to_string(segment) +
                       " at=" + std::to_string(at) + "/" + std::to_string(total));
    return RKVA_ERR_BAD_PAYLOAD;
}


bool recv_all(SOCKET sock, void* dst, size_t size) {
    auto* p = static_cast<uint8_t*>(dst);
    while (size) {
        const int n = ::recv(sock, reinterpret_cast<char*>(p), (int)size, 0);
        if (n <= 0) {
            return false;
        }
        p += n;
        size -= (size_t)n;
    }
    return true;
}

bool send_all(SOCKET sock, const void* src, size_t size) {
    const auto* p = static_cast<const uint8_t*>(src);
    while (size) {
        const int n = ::send(sock, reinterpret_cast<const char*>(p), (int)size, 0);
        if (n <= 0) {
            return false;
        }
        p += n;
        size -= (size_t)n;
    }
    return true;
}

// One host context's KV state. The D3D12 engine (next phase) owns the actual
// record/stage/tail buffers; this skeleton tracks logical session state so the
// control plane (create/destroy/reset/trim/rewind) is fully exercisable.
struct remote_session {
    uint32_t session_id = 0;
    uint32_t seq_id = 0;
    uint32_t capacity = 0;
    int32_t high_water = -1; // highest stored position, -1 = empty
    uint64_t trims = 0;
    uint64_t resets = 0;
    bool valid = true;
    bool range_configured = false;
    bool range_session = false;
    uint32_t range_storage = 0;
    uint32_t range_origin = 0;
    bool native_q4 = false; // layout 4: native Q4_0 bulk rows
    bool native_q8 = false; // layout 5: native Q8_0 bulk rows
};

struct rpc_profile_counter {
    uint64_t calls = 0, request_bytes = 0, response_bytes = 0;
    uint64_t payload_receive_us = 0, handle_us = 0, send_enqueue_us = 0;
};

struct bulk_dispatch_sample {
    uint64_t cycle_id = 0;
    uint32_t session = 0, layer = 0, nq = 0, wire = 0;
    uint64_t scratch_us = 0, engine_us = 0, transcode_us = 0;
};

struct remote_state {
    bool hello_done = false;
    bool native_rewind_v2_client = false;
    bool native_serial_prefix_v3_client = false;
    rkva_hello geo{};
    std::map<uint32_t, remote_session> sessions;
    std::unique_ptr<RemoteAttnEngine> engine;
    bool engine_f16 = false; // cache_bits 0/16 -> F16 path; 2..8 -> KVarN (pending)
    std::mutex mutex;
    bool profile_enabled = false;
    bool trace_requests = false;
    bool fp8_gpu_enabled = true;
    rpc_profile_counter rpc_profile[32]{};
    rkva_kvarn_bulk::Scratch bulk_scratch;
    uint64_t bulk_scratch_prepare_us = 0, bulk_engine_us = 0, bulk_output_transcode_us = 0;
    std::array<bulk_dispatch_sample, 256> bulk_dispatch_ring{};
    std::array<size_t, 2> bulk_dispatch_next{}, bulk_dispatch_stored{};
};

std::string rpc_profile_json(const remote_state& state) {
    std::string text = "{\"enabled\":" + std::string(state.profile_enabled ? "true" : "false") +
                       ",\"trace_requests\":" + (state.trace_requests ? "true" : "false") +
                       ",\"fp8_gpu_enabled\":" + (state.fp8_gpu_enabled ? "true" : "false") +
                       ",\"ops\":[";
    bool first = true;
    for (uint32_t op = 0; op < 32; ++op) {
        const auto& c = state.rpc_profile[op];
        if (!c.calls)
            continue;
        if (!first)
            text += ",";
        first = false;
        text += "{\"op\":" + std::to_string(op) + ",\"calls\":" + std::to_string(c.calls) +
                ",\"request_bytes\":" + std::to_string(c.request_bytes) +
                ",\"response_bytes\":" + std::to_string(c.response_bytes) +
                ",\"payload_receive_us\":" + std::to_string(c.payload_receive_us) +
                ",\"handle_us\":" + std::to_string(c.handle_us) +
                ",\"send_enqueue_us\":" + std::to_string(c.send_enqueue_us) + "}";
    }
    std::string phases = "{";
    for (unsigned phase = 0; phase < 2; ++phase) {
        if (phase)
            phases += ',';
        phases += phase ? "\"multi_query\":{" : "\"decode\":{";
        std::vector<double> scratch, engine, transcode;
        for (size_t i = 0; i < state.bulk_dispatch_stored[phase]; ++i) {
            const auto& b = state.bulk_dispatch_ring[phase * 128 + i];
            scratch.push_back(double(b.scratch_us));
            engine.push_back(double(b.engine_us));
            transcode.push_back(double(b.transcode_us));
        }
        rkva_rprof::Recorder::append_stats(phases, "scratch_prepare_us", scratch);
        phases += ',';
        rkva_rprof::Recorder::append_stats(phases, "engine_us", engine);
        phases += ',';
        rkva_rprof::Recorder::append_stats(phases, "output_transcode_us", transcode);
        phases += '}';
    }
    phases += '}';
    std::string recent = "[";
    bool first_recent = true;
    for (unsigned phase = 0; phase < 2; ++phase) {
        const size_t recent_count = std::min<size_t>(8, state.bulk_dispatch_stored[phase]);
        for (size_t i = 0; i < recent_count; ++i) {
            const auto& b =
                state.bulk_dispatch_ring[phase * 128 + (state.bulk_dispatch_next[phase] + 128 -
                                                        recent_count + i) %
                                                           128];
            if (!first_recent)
                recent += ',';
            first_recent = false;
            recent += "{\"cycle_id\":" + std::to_string(b.cycle_id) +
                      ",\"session\":" + std::to_string(b.session) +
                      ",\"layer\":" + std::to_string(b.layer) + ",\"nq\":" + std::to_string(b.nq) +
                      ",\"wire\":" + std::to_string(b.wire) +
                      ",\"scratch_prepare_us\":" + std::to_string(b.scratch_us) +
                      ",\"engine_us\":" + std::to_string(b.engine_us) +
                      ",\"output_transcode_us\":" + std::to_string(b.transcode_us) + "}";
        }
    }
    return text + "],\"bulk_dispatch_phases\":" + phases + ",\"bulk_dispatch_recent\":" + recent +
           "],\"bulk_scratch\":" + state.bulk_scratch.json() +
           ",\"bulk_dispatch_us\":{\"scratch_prepare\":" +
           std::to_string(state.bulk_scratch_prepare_us) +
           ",\"engine\":" + std::to_string(state.bulk_engine_us) +
           ",\"output_transcode\":" + std::to_string(state.bulk_output_transcode_us) + "}}";
}

// Ops this server actually serves. The F16 attention path is live; the KVarN
// native path (cache_bits 2..8) advertises ATTN too but returns UNSUPPORTED at
// call time until its D3D12 kernels land, so a KVarN host fails fast rather
// than consuming wrong output.
constexpr uint32_t supported_ops_mask() {
    return (1u << RKVA_HELLO) | (1u << RKVA_CREATE_SESSION) | (1u << RKVA_DESTROY_SESSION) |
           (1u << RKVA_RESET) | (1u << RKVA_TRIM) | (1u << RKVA_REWIND) | (1u << RKVA_PING) |
           (1u << RKVA_GET_STATS) | (1u << RKVA_ATTN_DECODE) | (1u << RKVA_ATTN_PREFILL) |
           (1u << RKVA_RANGE_CONFIG) | (1u << RKVA_RANGE_UPLOAD) | (1u << RKVA_RANGE_ATTN) |
           (1u << RKVA_KVARN_RANGE_CONFIG) | (1u << RKVA_KVARN_RANGE_UPLOAD) |
           (1u << RKVA_SNAPSHOT_SAVE) | (1u << RKVA_SNAPSHOT_RESTORE) |
           (1u << RKVA_RANGE_UPLOAD_ATTN) | (1u << RKVA_RANGE_PROFILE) |
           (1u << RKVA_KVARN_RANGE_BULK_ATTN) | (1u << RKVA_KVARN_RANGE_REWIND);
}

int handle_request(remote_state& state, const rkva_request& req, const std::vector<uint8_t>& input,
                   std::vector<uint8_t>& output, uint64_t& prepare_us, uint64_t& compute_us) {
    const auto t0 = clock_type::now();
    std::lock_guard<std::mutex> lock(state.mutex);
    // No HELLO replacement, destroy, reset or other control mutation after an
    // uncertain submission; the connection owner will stop accepting frames.
    if (state.engine && state.engine->submission_unconfirmed())
        return RKVA_ERR_GPU;

    if (req.op >= RKVA_RANGE_CONFIG && req.version != RKVA_VERSION) {
        return RKVA_ERR_UNSUPPORTED_OP;
    }
    switch (req.op) {
    case RKVA_HELLO: {
        if (req.payload_bytes != sizeof(rkva_hello)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        rkva_hello hello{};
        std::memcpy(&hello, input.data(), sizeof(hello));
        // Validate the geometry we can actually serve.
        if (hello.head_dim == 0 || hello.n_head == 0 || hello.n_head_kv == 0 ||
            hello.n_head % hello.n_head_kv != 0 || hello.max_ctx == 0 ||
            (hello.head_dim != 128 && hello.head_dim != 256) || !hello.n_layer_remote ||
            hello.n_layer_remote > 128 || hello.n_head > 256 || hello.n_head_kv > 256 ||
            hello.max_ctx > uint32_t(INT32_MAX) || hello.has_sinks || hello.swa) {
            return RKVA_ERR_BAD_GEOMETRY;
        }
        state.sessions.clear();
        state.profile_enabled = state.trace_requests = false;
        for (auto& counter : state.rpc_profile)
            counter = {};
        state.bulk_scratch.reset_counters();
        state.bulk_scratch_prepare_us = state.bulk_engine_us = state.bulk_output_transcode_us = 0;
        state.bulk_dispatch_next = {};
        state.bulk_dispatch_stored = {};
        state.fp8_gpu_enabled =
            !std::ifstream(g_snapshot_local_folder + "\\rkva-fp8-gpu-off.flag").good();
        state.geo = hello;
        state.hello_done = true;
        state.native_rewind_v2_client = (hello.proto_flags & RKVA_PROTO_NATIVE_REWIND_V2) != 0;
        state.native_serial_prefix_v3_client =
            state.native_rewind_v2_client &&
            (hello.proto_flags & RKVA_PROTO_NATIVE_SERIAL_PREFIX_V3) != 0;

        evict_local_session();

        // cache_bits 0 or 16 -> F16 bring-up path; 2..8 -> KVarN native.
        state.engine_f16 = (hello.cache_bits_k == 0 || hello.cache_bits_k >= 16);

        RemoteAttnEngine::Geometry g{};
        g.n_head = hello.n_head;
        g.n_head_kv = hello.n_head_kv;
        g.head_dim = hello.head_dim;
        g.head_slices = hello.head_dim / 128;
        g.cache_bits_k = state.engine_f16 ? 16 : hello.cache_bits_k;
        g.cache_bits_v = state.engine_f16 ? 16 : hello.cache_bits_v;
        g.group_tokens = hello.group_tokens ? hello.group_tokens : 128;
        g.sinkhorn_iters = hello.sinkhorn_iters;
        g.tail_tokens = hello.tail_tokens;
        g.tail_groups = hello.tail_groups ? hello.tail_groups : 2;
        g.stage_groups = g.tail_groups + 1;
        g.domain = hello.domain;
        g.kq_scale = hello.kq_scale;
        g.max_ctx = hello.max_ctx;
        g.n_layers = hello.n_layer_remote;

        state.engine = std::make_unique<RemoteAttnEngine>();
        std::string engine_error;
        if (!state.engine->initialize(g, &engine_error)) {
            state.engine.reset();
            state.hello_done = false;
            event_engine_error(engine_error);
            return RKVA_ERR_GPU;
        }

        rkva_hello_ack ack{};
        ack.supported_ops = supported_ops_mask();
        if (req.version == RKVA_VERSION_LEGACY || !state.engine_f16 ||
            hello.domain != RKVA_DOMAIN_AUTO) {
            ack.supported_ops &=
                ~((1u << RKVA_RANGE_CONFIG) | (1u << RKVA_RANGE_UPLOAD) | (1u << RKVA_RANGE_ATTN) |
                  (1u << RKVA_KVARN_RANGE_CONFIG) | (1u << RKVA_KVARN_RANGE_UPLOAD) |
                  (1u << RKVA_SNAPSHOT_SAVE) | (1u << RKVA_SNAPSHOT_RESTORE) |
                  (1u << RKVA_RANGE_UPLOAD_ATTN) | (1u << RKVA_RANGE_PROFILE) |
                  (1u << RKVA_KVARN_RANGE_REWIND) | (1u << RKVA_KVARN_RANGE_BULK_ATTN));
        }
        // Choose wire format: if host supports F16 wire (proto_flags bit 0), negotiate F16 for 2x
        // bandwidth savings
        ack.chosen_wire =
            (hello.proto_flags & (1u << RKVA_WIRE_F16)) ? RKVA_WIRE_F16 : RKVA_WIRE_F32;
        if (req.version == RKVA_VERSION && state.engine_f16 && hello.domain == RKVA_DOMAIN_AUTO) {
            ack.reserved[0] = RKVA_CAP_RANGE_Q4_0 | RKVA_CAP_RANGE_Q8_0 | RKVA_CAP_RANGE_KVARN4 |
                              RKVA_CAP_DISK_SNAPSHOT | RKVA_CAP_RANGE_UPLOAD_ATTN |
                              RKVA_CAP_RANGE_PROFILE;
            if (state.native_rewind_v2_client)
                ack.reserved[0] |= RKVA_CAP_KVARN_REWIND;
            if (state.native_serial_prefix_v3_client &&
                state.engine->native_serial_prefix_available())
                ack.reserved[0] |= RKVA_CAP_KVARN_SERIAL_PREFIX | RKVA_CAP_KVARN_BULK_ATTN |
                                   RKVA_CAP_KVARN_BULK_F16 | RKVA_CAP_KVARN_BULK_FP8 |
                                   RKVA_CAP_KVARN_BULK_FP8_RNE | RKVA_CAP_KVARN_COMPACT_STAGE |
                                   RKVA_CAP_NATIVE_Q4_BULK | RKVA_CAP_NATIVE_Q8_BULK;
        }
        if (!state.native_rewind_v2_client)
            ack.supported_ops &= ~(1u << RKVA_KVARN_RANGE_REWIND);
        ack.max_sessions = 64;
        ack.server_version = 278; // bumped with the UWP package
        // ~5 GiB game heap; leave headroom for the D3D12 runtime + buffers.
        ack.mem_budget_bytes = 4ull * 1024 * 1024 * 1024;
        output.resize(sizeof(ack));
        std::memcpy(output.data(), &ack, sizeof(ack));
        return RKVA_OK;
    }
    case RKVA_PING: {
        return RKVA_OK;
    }
    case RKVA_CREATE_SESSION: {
        if (req.payload_bytes != sizeof(rkva_session_req)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        evict_local_session();
        rkva_session_req sr{};
        std::memcpy(&sr, input.data(), sizeof(sr));
        if (!state.hello_done || !state.engine) {
            return RKVA_ERR_STATE;
        }
        if (req.version == RKVA_VERSION && (sr.flags & ~RKVA_SESSION_RANGE)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        const bool defer = req.version == RKVA_VERSION && (sr.flags & RKVA_SESSION_RANGE);
        if (defer && (!state.engine_f16 || state.geo.domain != RKVA_DOMAIN_AUTO)) {
            return RKVA_ERR_BAD_GEOMETRY;
        }
        remote_session s;
        s.range_session = defer;
        s.session_id = sr.session_id;
        s.seq_id = sr.seq_id;
        s.capacity = sr.capacity ? sr.capacity : state.geo.max_ctx;
        std::string engine_error;
        if (!state.engine->create_session(sr.session_id, s.capacity, &engine_error, defer)) {
            event_engine_error(engine_error);
            return RKVA_ERR_MEMORY;
        }
        state.sessions[sr.session_id] = s;
        return RKVA_OK;
    }
    case RKVA_DESTROY_SESSION: {
        if (state.engine && state.sessions.count(req.session_id)) {
            try {
                event(g_log_path, "{\"event\":\"memory_before_destroy\",\"session\":" +
                                      std::to_string(req.session_id) + ",\"memory\":" +
                                      state.engine->get_memory_profile_json() + "}");
            } catch (...) { /* Observation must never block session release. */
            }
        }
        state.sessions.erase(req.session_id);
        if (state.engine) {
            state.engine->destroy_session(req.session_id);
        }
        return RKVA_OK;
    }
    case RKVA_RESET: {
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end()) {
            return RKVA_ERR_BAD_SESSION;
        }
        it->second.high_water = -1;
        it->second.resets++;
        std::string engine_error;
        if (state.engine && !state.engine->reset_session(req.session_id, &engine_error)) {
            event_engine_error(engine_error);
            return RKVA_ERR_GPU;
        }
        // Disk restore can fail after partial GPU copies. A successful RESET
        // makes all old rows unreachable before authoritative host reupload.
        it->second.valid = true;
        return RKVA_OK;
    }
    case RKVA_TRIM:
    case RKVA_REWIND: {
        if (req.payload_bytes != sizeof(rkva_trim_req)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        rkva_trim_req tr{};
        std::memcpy(&tr, input.data(), sizeof(tr));
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end()) {
            return RKVA_ERR_BAD_SESSION;
        }
        if (tr.pos0 < 0) {
            return RKVA_ERR_BAD_POSITION;
        }
        // Remove every stored position >= pos0 (logical, mirrors DFL2 trim).
        if (it->second.high_water >= tr.pos0) {
            it->second.high_water = tr.pos0 - 1;
        }
        it->second.trims++;
        std::string engine_error;
        if (state.engine && !state.engine->trim_session(req.session_id, tr.pos0, &engine_error)) {
            event_engine_error(engine_error);
            return RKVA_ERR_GPU;
        }
        return RKVA_OK;
    }
    case RKVA_GET_STATS: {
        uint64_t trims = 0, resets = 0;
        for (const auto& kv : state.sessions) {
            trims += kv.second.trims;
            resets += kv.second.resets;
        }
        const unsigned long long engine_bytes =
            state.engine ? (unsigned long long)state.engine->allocated_bytes() : 0ull;
        char buf[640];
        snprintf(buf, sizeof(buf),
                 "{\"service\":\"rkva\",\"hello\":%s,\"sessions\":%zu,\"trims\":%llu,"
                 "\"resets\":%llu,\"engine\":\"%s\",\"engine_kv_bytes\":%llu}",
                 state.hello_done ? "true" : "false", state.sessions.size(),
                 (unsigned long long)trims, (unsigned long long)resets,
                 state.engine_f16 ? "d3d12-f16" : "d3d12-kvarn", engine_bytes);
        const size_t n = std::strlen(buf);
        output.assign(buf, buf + n);
        return RKVA_OK;
    }
    case RKVA_ATTN_DECODE:
    case RKVA_ATTN_PREFILL: {
        if (!state.hello_done || !state.engine) {
            return RKVA_ERR_STATE;
        }
        // KVarN (cache_bits 2..8) and F16 paths both run the D3D12 engine.
        if (state.sessions.find(req.session_id) == state.sessions.end()) {
            return RKVA_ERR_BAD_SESSION;
        }
        if (state.sessions.at(req.session_id).range_configured ||
            state.sessions.at(req.session_id).range_session) {
            return RKVA_ERR_STATE;
        }
        if (input.size() < sizeof(rkva_attn_req)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        const uint32_t n_tokens = req.n_tokens;
        const uint32_t n_head = state.geo.n_head;
        const uint32_t n_head_kv = state.geo.n_head_kv;
        const uint32_t head_dim = state.geo.head_dim;
        if (n_tokens == 0) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        const uint64_t q_elems = uint64_t(n_tokens) * n_head * head_dim;
        const uint64_t kv_elems = uint64_t(n_tokens) * n_head_kv * head_dim;

        rkva_attn_req areq{};
        std::memcpy(&areq, input.data(), sizeof(areq));
        const bool is_f16 = (areq.wire == RKVA_WIRE_F16);
        if (areq.wire != RKVA_WIRE_F32 && areq.wire != RKVA_WIRE_F16) {
            return RKVA_ERR_UNSUPPORTED_OP;
        }
        const uint64_t elem_size = is_f16 ? 2ull : 4ull;
        const uint64_t expected =
            sizeof(rkva_attn_req) + uint64_t(n_tokens) * 4 + (q_elems + 2 * kv_elems) * elem_size;
        if (req.payload_bytes != expected) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        const uint8_t* p = input.data() + sizeof(rkva_attn_req);
        const int32_t* positions = reinterpret_cast<const int32_t*>(p);
        p += uint64_t(n_tokens) * 4;

        std::vector<float> q_f32, k_f32, v_f32;
        const float* q_ptr = nullptr;
        const float* k_ptr = nullptr;
        const float* v_ptr = nullptr;

        if (is_f16) {
            const uint16_t* q_f16 = reinterpret_cast<const uint16_t*>(p);
            p += q_elems * 2;
            const uint16_t* k_f16 = reinterpret_cast<const uint16_t*>(p);
            p += kv_elems * 2;
            const uint16_t* v_f16 = reinterpret_cast<const uint16_t*>(p);

            q_f32.resize(q_elems);
            k_f32.resize(kv_elems);
            v_f32.resize(kv_elems);

            convert_f16_to_f32(q_f16, q_f32.data(), q_elems);
            convert_f16_to_f32(k_f16, k_f32.data(), kv_elems);
            convert_f16_to_f32(v_f16, v_f32.data(), kv_elems);
            q_ptr = q_f32.data();
            k_ptr = k_f32.data();
            v_ptr = v_f32.data();
        } else {
            q_ptr = reinterpret_cast<const float*>(p);
            p += q_elems * 4;
            k_ptr = reinterpret_cast<const float*>(p);
            p += kv_elems * 4;
            v_ptr = reinterpret_cast<const float*>(p);
        }

        std::vector<float> out_f32_buf;
        float* out_target = nullptr;
        if (is_f16) {
            out_f32_buf.resize(q_elems);
            out_target = out_f32_buf.data();
        } else {
            output.resize(q_elems * 4);
            out_target = reinterpret_cast<float*>(output.data());
        }

        const auto compute_start = clock_type::now();
        std::string engine_error;
        const bool ok = state.engine->attend(req.session_id, req.layer_id, n_tokens, positions,
                                             q_ptr, k_ptr, v_ptr, out_target, &engine_error);
        compute_us = elapsed_us(compute_start);
        if (!ok) {
            output.clear();
            event_engine_error(engine_error);
            auto it = state.sessions.find(req.session_id);
            if (it != state.sessions.end()) {
                it->second.valid = false;
            }
            return RKVA_ERR_GPU;
        }

        if (is_f16) {
            output.resize(q_elems * 2);
            uint16_t* out_f16 = reinterpret_cast<uint16_t*>(output.data());
            convert_f32_to_f16(out_target, out_f16, q_elems);
        }
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    case RKVA_RANGE_CONFIG: {
        if (input.size() != sizeof(rkva_range_config)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !state.engine || !state.engine_f16) {
            return RKVA_ERR_BAD_SESSION;
        }
        rkva_range_config config{};
        std::memcpy(&config, input.data(), sizeof(config));
        if ((config.reserved[0] & ~(RKVA_RANGE_STORAGE_Q4_0 | RKVA_RANGE_STORAGE_Q8_0)) ||
            config.reserved[0] == (RKVA_RANGE_STORAGE_Q4_0 | RKVA_RANGE_STORAGE_Q8_0) ||
            config.reserved[1] || config.capacity != it->second.capacity ||
            config.origin > uint32_t(INT32_MAX) - config.capacity) {
            return RKVA_ERR_BAD_POSITION;
        }
        std::string error;
        if (!state.engine->configure_range(req.session_id, config.origin, config.capacity, &error,
                                           config.reserved[0])) {
            event_engine_error(error);
            return RKVA_ERR_STATE;
        }
        it->second.range_configured = true;
        it->second.range_storage = config.reserved[0];
        it->second.range_origin = config.origin;
        return RKVA_OK;
    }
    case RKVA_RANGE_UPLOAD: {
        if (input.size() < sizeof(rkva_range_upload)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !it->second.range_configured || !state.engine ||
            !it->second.valid) {
            return RKVA_ERR_BAD_SESSION;
        }
        rkva_range_upload upload{};
        std::memcpy(&upload, input.data(), sizeof(upload));
        if (it->second.range_storage == RKVA_RANGE_STORAGE_KVARN4 ||
            upload.wire != (it->second.range_storage == RKVA_RANGE_STORAGE_Q8_0 ? RKVA_WIRE_Q8_0
                            : it->second.range_storage == RKVA_RANGE_STORAGE_Q4_0
                                ? RKVA_WIRE_Q4_0
                                : RKVA_WIRE_F16) ||
            upload.reserved || !upload.n_rows || upload.start_row > it->second.capacity ||
            upload.n_rows > it->second.capacity - upload.start_row ||
            req.layer_id >= state.geo.n_layer_remote) {
            return RKVA_ERR_BAD_POSITION;
        }
        const uint64_t plane_bytes =
            uint64_t(upload.n_rows) * state.geo.n_head_kv *
            (it->second.range_storage
                 ? uint64_t(state.geo.head_dim / 32) *
                       (it->second.range_storage == RKVA_RANGE_STORAGE_Q4_0 ? 18 : 34)
                 : uint64_t(state.geo.head_dim) * 2);
        if (input.size() != sizeof(upload) + 2 * plane_bytes) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        const uint8_t* data = input.data() + sizeof(upload);
        std::string error;
        const auto start = clock_type::now();
        if (!state.engine->upload_range(req.session_id, req.layer_id, upload.start_row,
                                        upload.n_rows, reinterpret_cast<const uint16_t*>(data),
                                        reinterpret_cast<const uint16_t*>(data + plane_bytes),
                                        &error)) {
            event_engine_error(error);
            return state.engine->session_invalid(req.session_id) ? RKVA_ERR_GPU
                                                                 : RKVA_ERR_BAD_POSITION;
        }
        compute_us = elapsed_us(start);
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    case RKVA_KVARN_RANGE_CONFIG: {
        if (req.version != RKVA_VERSION || input.size() != sizeof(rkva_kvarn_range_config))
            return RKVA_ERR_BAD_PAYLOAD;
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !state.engine || !state.engine_f16 || !it->second.valid)
            return RKVA_ERR_BAD_SESSION;
        rkva_kvarn_range_config c{};
        std::memcpy(&c, input.data(), sizeof(c));
        if ((c.layout_version >= 2 && c.layout_version <= 3 && !state.native_rewind_v2_client) ||
            (c.layout_version == 3 && !state.native_serial_prefix_v3_client))
            return RKVA_ERR_BAD_PAYLOAD;
        std::string error;
        if (c.layout_version == 4) {
            // Native Q4_0 overflow rows: the storage is the standard Q4_0 range
            // buffer, so the existing range upload/attention kernels apply.
            if (!rkva_kvarn_range::valid_q4_descriptor(c, state.geo.head_dim))
                return RKVA_ERR_BAD_GEOMETRY;
            if (!state.engine->configure_range(req.session_id, c.origin, c.capacity, &error,
                                               RKVA_RANGE_STORAGE_Q4_0)) {
                event_engine_error(error);
                return RKVA_ERR_BAD_GEOMETRY;
            }
            it->second.range_configured = true;
            it->second.range_storage = RKVA_RANGE_STORAGE_Q4_0;
            it->second.native_q4 = true;
            it->second.native_q8 = false;
            it->second.range_origin = c.origin;
            return RKVA_OK;
        }
        if (c.layout_version == 5) {
            // Native Q8_0 overflow rows: same envelope as layout 4 over the
            // standard Q8_0 range buffer (34-byte blocks, 272 B per 256-dim head).
            if (!rkva_kvarn_range::valid_q8_descriptor(c, state.geo.head_dim))
                return RKVA_ERR_BAD_GEOMETRY;
            if (!state.engine->configure_range(req.session_id, c.origin, c.capacity, &error,
                                               RKVA_RANGE_STORAGE_Q8_0)) {
                event_engine_error(error);
                return RKVA_ERR_BAD_GEOMETRY;
            }
            it->second.range_configured = true;
            it->second.range_storage = RKVA_RANGE_STORAGE_Q8_0;
            it->second.native_q8 = true;
            it->second.native_q4 = false;
            it->second.range_origin = c.origin;
            return RKVA_OK;
        }
        if (!state.engine->configure_kvarn_range(req.session_id, c, &error)) {
            event_engine_error(error);
            return RKVA_ERR_BAD_GEOMETRY;
        }
        it->second.range_configured = true;
        it->second.range_storage = RKVA_RANGE_STORAGE_KVARN4;
        it->second.native_q4 = false;
        it->second.native_q8 = false;
        if (c.layout_version >= 2) {
            const uint64_t generation = state.engine->native_range_generation(req.session_id);
            output.resize(sizeof(generation));
            std::memcpy(output.data(), &generation, sizeof(generation));
        }
        return RKVA_OK;
    }
    case RKVA_KVARN_RANGE_UPLOAD: {
        if (req.version != RKVA_VERSION || input.size() < sizeof(rkva_kvarn_range_upload))
            return RKVA_ERR_BAD_PAYLOAD;
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !state.engine || !it->second.valid ||
            !it->second.range_configured ||
            (it->second.range_storage != RKVA_RANGE_STORAGE_KVARN4 &&
             !(it->second.native_q4 && it->second.range_storage == RKVA_RANGE_STORAGE_Q4_0) &&
             !(it->second.native_q8 && it->second.range_storage == RKVA_RANGE_STORAGE_Q8_0)))
            return RKVA_ERR_BAD_SESSION;
        rkva_kvarn_range_upload u{};
        std::memcpy(&u, input.data(), sizeof(u));
        std::string error;
        if (it->second.native_q4 || it->second.native_q8) {
            const bool q8 = it->second.native_q8;
            const uint64_t row = q8
                ? rkva_kvarn_range::q8_row_bytes(state.geo.head_dim, state.geo.n_head_kv)
                : rkva_kvarn_range::q4_row_bytes(state.geo.head_dim, state.geo.n_head_kv);
            const bool valid = q8
                ? rkva_kvarn_range::valid_q8_upload(u, it->second.capacity, u.append_begin)
                : rkva_kvarn_range::valid_q4_upload(u, it->second.capacity, u.append_begin);
            if (!valid || uint64_t(u.append_begin) + u.append_rows != u.live_count)
                return RKVA_ERR_BAD_PAYLOAD;
            const uint64_t need = uint64_t(u.append_rows) * row * 2;
            if (input.size() != sizeof(u) + need)
                return RKVA_ERR_BAD_PAYLOAD;
            const uint8_t* kp = input.data() + sizeof(u);
            const uint8_t* vp = kp + uint64_t(u.append_rows) * row;
            const auto start = clock_type::now();
            if (!state.engine->upload_range(req.session_id, req.layer_id, u.append_begin,
                                            u.append_rows,
                                            reinterpret_cast<const uint16_t*>(kp),
                                            reinterpret_cast<const uint16_t*>(vp), &error)) {
                event_engine_error(error);
                return state.engine->session_invalid(req.session_id) ? RKVA_ERR_GPU
                                                                     : RKVA_ERR_BAD_POSITION;
            }
            compute_us = elapsed_us(start);
            prepare_us = elapsed_us(t0);
            return RKVA_OK;
        }
        const auto start = clock_type::now();
        if (!state.engine->upload_kvarn_range(req.session_id, req.layer_id, u,
                                              input.data() + sizeof(u), input.size() - sizeof(u),
                                              &error)) {
            event_engine_error(error);
            return state.engine->session_invalid(req.session_id) ? RKVA_ERR_GPU
                                                                 : RKVA_ERR_BAD_PAYLOAD;
        }
        compute_us = elapsed_us(start);
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    case RKVA_KVARN_RANGE_BULK_ATTN: {
        if (req.version != RKVA_VERSION || !req.n_tokens || req.n_tokens > 256 ||
            req.layer_id >= state.geo.n_layer_remote)
            return RKVA_ERR_BAD_PAYLOAD;
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !it->second.range_configured ||
            (it->second.range_storage != RKVA_RANGE_STORAGE_KVARN4 && !it->second.native_q4 &&
             !it->second.native_q8) ||
            !state.engine || !state.engine_f16 || !it->second.valid)
            return RKVA_ERR_BAD_SESSION;
        if (!it->second.native_q4 && !it->second.native_q8 &&
            !state.native_serial_prefix_v3_client)
            return RKVA_ERR_BAD_PAYLOAD;
        if (it->second.native_q4 && it->second.range_storage != RKVA_RANGE_STORAGE_Q4_0)
            return RKVA_ERR_BAD_SESSION;
        if (it->second.native_q8 && it->second.range_storage != RKVA_RANGE_STORAGE_Q8_0)
            return RKVA_ERR_BAD_SESSION;
        if (it->second.native_q4 || it->second.native_q8) {
            // Layouts 4/5: ordered native row uploads plus queries in one round
            // trip. The host defers only the socket transaction (async overlap),
            // so the upload and the attention keep their wire order. Q/OUT may be
            // F32, F16 or FP8; LSE is always F32.
            const bool q8_rows = it->second.native_q8;
            const uint64_t row = q8_rows
                ? rkva_kvarn_range::q8_row_bytes(state.geo.head_dim, state.geo.n_head_kv)
                : rkva_kvarn_range::q4_row_bytes(state.geo.head_dim, state.geo.n_head_kv);
            const auto valid_rows_upload = [&](const rkva_kvarn_range_upload& vu) {
                return q8_rows
                    ? rkva_kvarn_range::valid_q8_upload(vu, it->second.capacity, vu.append_begin)
                    : rkva_kvarn_range::valid_q4_upload(vu, it->second.capacity, vu.append_begin);
            };
            const uint64_t heads = uint64_t(req.n_tokens) * state.geo.n_head;
            const uint64_t total = heads * state.geo.head_dim;
            rkva_kvarn_range_bulk header{};
            if (input.size() < sizeof(header))
                return RKVA_ERR_BAD_PAYLOAD;
            std::memcpy(&header, input.data(), sizeof(header));
            if (header.version != 1 || !header.segment_count || header.segment_count > 32 ||
                header.reserved || header.total_queries != req.n_tokens)
                return RKVA_ERR_BAD_PAYLOAD;
            // Peek the wire of the first segment to size the reply.
            uint32_t reply_wire = RKVA_WIRE_F32;
            {
                const size_t first = sizeof(header) + sizeof(rkva_kvarn_range_bulk_segment);
                if (input.size() < first + sizeof(rkva_kvarn_range_upload))
                    return RKVA_ERR_BAD_PAYLOAD;
                rkva_kvarn_range_upload u0{};
                std::memcpy(&u0, input.data() + first, sizeof(u0));
                const size_t attn_off = first + sizeof(u0) +
                        size_t(u0.append_rows) * row * 2;
                if (attn_off + sizeof(rkva_range_attn) > input.size())
                    return RKVA_ERR_BAD_PAYLOAD;
                rkva_range_attn a0{};
                std::memcpy(&a0, input.data() + attn_off, sizeof(a0));
                reply_wire = a0.wire;
            }
            const uint64_t out_elem = reply_wire == RKVA_WIRE_FP8 ? 1ull :
                                      reply_wire == RKVA_WIRE_F16 ? 2ull : 4ull;
            // Validation pass: a malformed envelope must never mutate KV.
            {
                size_t v = sizeof(header);
                uint32_t vq = 0;
                uint32_t expected_live = 0;
                bool first_segment = true;
                for (uint32_t si = 0; si < header.segment_count; ++si) {
                    rkva_kvarn_range_bulk_segment vseg{};
                    if (input.size() - v < sizeof(vseg))
                        return q4_bad("segment header", si, v, input.size());
                    std::memcpy(&vseg, input.data() + v, sizeof(vseg));
                    v += sizeof(vseg);
                    if (vseg.upload_bytes < sizeof(rkva_kvarn_range_upload) ||
                        vseg.upload_bytes > input.size() - v)
                        return q4_bad("segment upload size", si, v, input.size());
                    rkva_kvarn_range_upload vu{};
                    std::memcpy(&vu, input.data() + v, sizeof(vu));
                    // Continuity inside the envelope: every segment after the
                    // first must start exactly at the previous live count, and
                    // a non-RESET bootstrap is left to the engine's coverage
                    // check (the handler has no per-layer counter).
                    const bool reset = (vu.flags & RKVA_KVARN_UPLOAD_RESET) != 0;
                    if (!valid_rows_upload(vu) ||
                        uint64_t(vu.append_begin) + vu.append_rows != vu.live_count ||
                        (first_segment ? (reset && vu.append_begin != 0)
                                       : vu.append_begin != expected_live))
                        return q4_bad("upload header", si, v, input.size());
                    expected_live = vu.live_count;
                    first_segment = false;
                    if (vseg.upload_bytes != sizeof(vu) + uint64_t(vu.append_rows) * row * 2)
                        return q4_bad("upload payload", si, v, input.size());
                    v += vseg.upload_bytes;
                    rkva_range_attn va{};
                    if (input.size() - v < sizeof(va))
                        return q4_bad("attn header", si, v, input.size());
                    std::memcpy(&va, input.data() + v, sizeof(va));
                    v += sizeof(va);
                    if (!vseg.n_queries || vseg.n_queries > req.n_tokens - vq ||
                        va.wire != reply_wire || va.flags || va.reserved[0] || va.reserved[1])
                        return q4_bad("attn fields", si, v, input.size());
                    const uint64_t ve = uint64_t(vseg.n_queries) * state.geo.n_head *
                                        state.geo.head_dim;
                    const uint64_t vb = uint64_t(vseg.n_queries) * sizeof(int32_t) + ve * out_elem;
                    if (ve > (64ull << 20) || input.size() - v < vb)
                        return q4_bad("query payload", si, v, input.size());
                    v += vb;
                    vq += vseg.n_queries;
                }
                if (v != input.size())
                    return q4_bad("trailing bytes", header.segment_count, v, input.size());
                if (vq != req.n_tokens)
                    return q4_bad("query total", header.segment_count, vq, req.n_tokens);
            }
            // One command list / one submit for every segment of the layer.
            std::vector<float> out_f32(size_t(total), 0.0f);
            std::vector<float> lse_f32(size_t(heads), 0.0f);
            std::string error;
            const auto start = clock_type::now();
            const bool bulk_ok = q8_rows
                ? state.engine->attend_q8_bulk(req.session_id, req.layer_id, req.n_tokens,
                                               input.data(), input.size(), out_f32.data(),
                                               lse_f32.data(), &error)
                : state.engine->attend_q4_bulk(req.session_id, req.layer_id, req.n_tokens,
                                               input.data(), input.size(), out_f32.data(),
                                               lse_f32.data(), &error);
            if (!bulk_ok) {
                event_engine_error(error);
                return state.engine->session_invalid(req.session_id) ? RKVA_ERR_GPU
                                                                     : RKVA_ERR_BAD_PAYLOAD;
            }
            compute_us = elapsed_us(start);
            output.resize(size_t(total * out_elem + heads * sizeof(float)));
            uint8_t * out_bytes = output.data();
            std::memcpy(output.data() + total * out_elem, lse_f32.data(), heads * sizeof(float));
            if (reply_wire == RKVA_WIRE_FP8) {
                for (uint64_t e = 0; e < total; ++e)
                    out_bytes[e] = rkva_kvarn_bulk::f16_to_fp8_bits(
                            rkva_kvarn_bulk::f32_to_f16_bits(out_f32[e]));
            } else if (reply_wire == RKVA_WIRE_F16) {
                uint16_t * o16 = reinterpret_cast<uint16_t*>(out_bytes);
                for (uint64_t e = 0; e < total; ++e)
                    o16[e] = rkva_kvarn_bulk::f32_to_f16_bits(out_f32[e]);
            } else {
                std::memcpy(out_bytes, out_f32.data(), total * sizeof(float));
            }
            prepare_us = elapsed_us(t0);
            return RKVA_OK;
        }
        uint32_t wire = RKVA_WIRE_F32;
        if (!rkva_kvarn_bulk::peek_wire(input.data(), input.size(), &wire))
            return RKVA_ERR_BAD_PAYLOAD;
        const bool wire_f16 = (wire == RKVA_WIRE_F16);
        const bool wire_fp8 = (wire == RKVA_WIRE_FP8);
        const bool wire_sz = (wire == RKVA_WIRE_SPLITZIP);
        const uint64_t heads = uint64_t(req.n_tokens) * state.geo.n_head;
        const uint64_t total_elems = heads * state.geo.head_dim;
        const uint64_t out_bytes = total_elems * (wire_fp8                ? 1ull
                                                  : (wire_f16 || wire_sz) ? 2ull
                                                                          : 4ull);
        // Include cold optional PSO setup in preparation, not an unreported gap.
        const auto t_scratch = state.profile_enabled ? clock_type::now() : clock_type::time_point{};
        const bool gpu_fp8 =
            wire_fp8 && state.fp8_gpu_enabled && state.engine->ensure_bulk_fp8_gpu_pipelines();
        rkva_kvarn_bulk::ScratchCall scratch_call(state.bulk_scratch, req.n_tokens, wire);
        auto& temp_out_f16 = state.bulk_scratch.output_f16;
        auto& temp_lse = state.bulk_scratch.lse;
        if ((wire_fp8 && !gpu_fp8) || wire_sz) {
            temp_out_f16.resize(total_elems);
            temp_lse.resize(heads);
        } else {
            output.resize(out_bytes + heads * 4);
        }
        std::string error;
        bulk_dispatch_sample dispatch;
        dispatch.cycle_id = req.cycle_id;
        dispatch.session = req.session_id;
        dispatch.layer = req.layer_id;
        dispatch.nq = req.n_tokens;
        dispatch.wire = wire;
        if (state.profile_enabled) {
            dispatch.scratch_us = elapsed_us(t_scratch);
            state.bulk_scratch_prepare_us += dispatch.scratch_us;
        }
        const auto start = clock_type::now();
        const bool ok =
            gpu_fp8 ? state.engine->attend_kvarn_bulk_fp8(
                          req.session_id, req.layer_id, req.n_tokens, input.data(), input.size(),
                          output.data(), reinterpret_cast<float*>(output.data() + out_bytes),
                          &error, req.cycle_id)
            : (wire_f16 || wire_fp8 || wire_sz)
                ? state.engine->attend_kvarn_bulk_f16(
                      req.session_id, req.layer_id, req.n_tokens, input.data(), input.size(),
                      (wire_fp8 || wire_sz) ? temp_out_f16.data()
                                            : reinterpret_cast<uint16_t*>(output.data()),
                      (wire_fp8 || wire_sz) ? temp_lse.data()
                                            : reinterpret_cast<float*>(output.data() + out_bytes),
                      &error, req.cycle_id)
                : state.engine->attend_kvarn_bulk(
                      req.session_id, req.layer_id, req.n_tokens, input.data(), input.size(),
                      reinterpret_cast<float*>(output.data()),
                      reinterpret_cast<float*>(output.data() + out_bytes), &error, req.cycle_id);
        if (state.profile_enabled) {
            dispatch.engine_us = elapsed_us(start);
            state.bulk_engine_us += dispatch.engine_us;
        }
        const auto t_transcode =
            state.profile_enabled ? clock_type::now() : clock_type::time_point{};
        if (!ok) {
            output.clear();
            event_engine_error(error);
            return state.engine->session_invalid(req.session_id) ? RKVA_ERR_GPU
                                                                 : RKVA_ERR_BAD_PAYLOAD;
        }
        if (wire_fp8 && !gpu_fp8) {
            output.resize(total_elems + heads * sizeof(float));
            uint8_t* dst_fp8 = output.data();
            for (size_t e = 0; e < total_elems; ++e) {
                dst_fp8[e] = rkva_kvarn_bulk::f16_to_fp8_bits(temp_out_f16[e]);
            }
            std::memcpy(output.data() + total_elems, temp_lse.data(), heads * sizeof(float));
        } else if (wire_sz) {
            const uint64_t f16_bytes = total_elems * sizeof(uint16_t);
            output.resize(f16_bytes + heads * sizeof(float));
            std::vector<uint8_t> sz_out;
            rkva_kvarn_bulk::splitzip_encode(temp_out_f16.data(), temp_out_f16.size(), sz_out);
            if (sz_out.size() <= f16_bytes) {
                std::memcpy(output.data(), sz_out.data(), sz_out.size());
                if (sz_out.size() < f16_bytes) {
                    std::memset(output.data() + sz_out.size(), 0, f16_bytes - sz_out.size());
                }
            } else {
                std::memcpy(output.data(), temp_out_f16.data(), f16_bytes);
            }
            std::memcpy(output.data() + f16_bytes, temp_lse.data(), heads * sizeof(float));
        }
        if (state.profile_enabled) {
            dispatch.transcode_us = elapsed_us(t_transcode);
            state.bulk_output_transcode_us += dispatch.transcode_us;
            const unsigned phase = req.n_tokens == 1 ? 0 : 1;
            state.bulk_dispatch_ring[phase * 128 + state.bulk_dispatch_next[phase]] = dispatch;
            state.bulk_dispatch_next[phase] = (state.bulk_dispatch_next[phase] + 1) % 128;
            state.bulk_dispatch_stored[phase] =
                std::min<size_t>(state.bulk_dispatch_stored[phase] + 1, 128);
        }
        compute_us = elapsed_us(start);
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    case RKVA_RANGE_ATTN: {
        if (input.size() < sizeof(rkva_range_attn)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !it->second.range_configured || !state.engine ||
            !it->second.valid) {
            return RKVA_ERR_BAD_SESSION;
        }
        rkva_range_attn attn{};
        std::memcpy(&attn, input.data(), sizeof(attn));
        if (attn.wire != RKVA_WIRE_F32 || attn.flags || attn.reserved[0] || attn.reserved[1] ||
            !req.n_tokens || req.n_tokens > 1024 || req.layer_id >= state.geo.n_layer_remote) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        const uint64_t q_heads = uint64_t(req.n_tokens) * state.geo.n_head;
        const uint64_t q_bytes = q_heads * state.geo.head_dim * 4;
        if (input.size() != sizeof(attn) + uint64_t(req.n_tokens) * 4 + q_bytes) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        const auto* positions = reinterpret_cast<const int32_t*>(input.data() + sizeof(attn));
        const auto* q = reinterpret_cast<const float*>(input.data() + sizeof(attn) +
                                                       uint64_t(req.n_tokens) * 4);
        for (uint32_t t = 0; t < req.n_tokens; ++t) {
            if (positions[t] < 0)
                return RKVA_ERR_BAD_POSITION;
        }
        output.resize(q_bytes + q_heads * 4);
        auto* out = reinterpret_cast<float*>(output.data());
        auto* lse = reinterpret_cast<float*>(output.data() + q_bytes);
        std::string error;
        const auto start = clock_type::now();
        if (!state.engine->attend_range(req.session_id, req.layer_id, req.n_tokens, positions, q,
                                        out, lse, &error)) {
            output.clear();
            event_engine_error(error);
            return state.engine->session_invalid(req.session_id) ? RKVA_ERR_GPU
                                                                 : RKVA_ERR_BAD_PAYLOAD;
        }
        compute_us = elapsed_us(start);
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    case RKVA_KVARN_RANGE_REWIND: {
        if (!state.hello_done || !state.native_rewind_v2_client ||
            input.size() < sizeof(rkva_kvarn_range_rewind) || req.layer_id || req.n_tokens ||
            !state.engine || !state.engine_f16)
            return RKVA_ERR_BAD_PAYLOAD;
        const auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !it->second.valid || !it->second.range_configured)
            return RKVA_ERR_BAD_SESSION;
        rkva_kvarn_range_rewind rewind{};
        std::memcpy(&rewind, input.data(), sizeof(rewind));
        std::string error;
        const auto start = clock_type::now();
        if (!state.engine->rewind_kvarn_range(req.session_id, rewind, input.data() + sizeof(rewind),
                                              input.size() - sizeof(rewind), &error)) {
            event_engine_error(error);
            return state.engine->session_invalid(req.session_id) ? RKVA_ERR_GPU
                                                                 : RKVA_ERR_BAD_PAYLOAD;
        }
        compute_us = elapsed_us(start);
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    case RKVA_RANGE_PROFILE: {
        if (!state.hello_done || !state.engine || !state.engine_f16 ||
            input.size() != sizeof(rkva_range_profile) || req.n_tokens || req.layer_id)
            return RKVA_ERR_BAD_PAYLOAD;
        rkva_range_profile control{};
        std::memcpy(&control, input.data(), sizeof(control));
        constexpr uint32_t allowed = RKVA_PROFILE_ENABLE | RKVA_PROFILE_RESET | RKVA_PROFILE_TRACE |
                                     RKVA_PROFILE_REPORT_ONLY;
        if (control.reserved || (control.flags & ~allowed) || control.q4_kernel_mode > 3 ||
            ((control.flags & RKVA_PROFILE_REPORT_ONLY) &&
             (control.flags != RKVA_PROFILE_REPORT_ONLY || control.q4_kernel_mode)))
            return RKVA_ERR_BAD_PAYLOAD;
        if (!(control.flags & RKVA_PROFILE_REPORT_ONLY)) {
            std::string error;
            if (!state.engine->set_range_profiling((control.flags & RKVA_PROFILE_ENABLE) != 0,
                                                   (control.flags & RKVA_PROFILE_RESET) != 0,
                                                   control.q4_kernel_mode, error)) {
                event_engine_error(error);
                return RKVA_ERR_STATE;
            }
            state.profile_enabled = (control.flags & RKVA_PROFILE_ENABLE) != 0;
            state.trace_requests = (control.flags & RKVA_PROFILE_TRACE) != 0;
            if (control.flags & RKVA_PROFILE_RESET) {
                for (auto& counter : state.rpc_profile)
                    counter = {};
                state.bulk_scratch.reset_counters();
                state.bulk_scratch_prepare_us = state.bulk_engine_us =
                    state.bulk_output_transcode_us = 0;
                state.bulk_dispatch_next = {};
                state.bulk_dispatch_stored = {};
            }
        }
        const std::string engine = state.engine->get_range_profile_json();
        const std::string report = "{\"engine\":" + (engine.empty() ? "{}" : engine) +
                                   ",\"transport\":" + rpc_profile_json(state) + "}";
        if (report.size() > (1u << 20))
            return RKVA_ERR_MEMORY;
        output.assign(report.begin(), report.end());
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    case RKVA_RANGE_UPLOAD_ATTN: {
        if (input.size() < sizeof(rkva_range_upload_attn))
            return RKVA_ERR_BAD_PAYLOAD;
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !it->second.range_configured || !state.engine ||
            !it->second.valid || it->second.range_storage == RKVA_RANGE_STORAGE_KVARN4)
            return RKVA_ERR_BAD_SESSION;
        rkva_range_upload_attn combined{};
        std::memcpy(&combined, input.data(), sizeof(combined));
        const auto& upload = combined.upload;
        const auto& attn = combined.attention;
        const uint32_t wire = it->second.range_storage == RKVA_RANGE_STORAGE_Q8_0   ? RKVA_WIRE_Q8_0
                              : it->second.range_storage == RKVA_RANGE_STORAGE_Q4_0 ? RKVA_WIRE_Q4_0
                                                                                    : RKVA_WIRE_F16;
        if (upload.wire != wire || upload.reserved || !upload.n_rows ||
            upload.start_row > it->second.capacity ||
            upload.n_rows > it->second.capacity - upload.start_row || attn.wire != RKVA_WIRE_F32 ||
            attn.flags || attn.reserved[0] || attn.reserved[1] || !req.n_tokens ||
            req.n_tokens > 1024 || req.layer_id >= state.geo.n_layer_remote)
            return RKVA_ERR_BAD_PAYLOAD;
        const uint64_t row_bytes = uint64_t(state.geo.n_head_kv) *
                                   (wire == RKVA_WIRE_F16 ? uint64_t(state.geo.head_dim) * 2
                                                          : uint64_t(state.geo.head_dim / 32) *
                                                                (wire == RKVA_WIRE_Q4_0 ? 18 : 34));
        const uint64_t plane_bytes = uint64_t(upload.n_rows) * row_bytes;
        const uint64_t q_heads = uint64_t(req.n_tokens) * state.geo.n_head;
        const uint64_t q_bytes = q_heads * state.geo.head_dim * sizeof(float);
        const uint64_t positions_bytes = uint64_t(req.n_tokens) * sizeof(int32_t);
        if (input.size() != sizeof(combined) + 2 * plane_bytes + positions_bytes + q_bytes)
            return RKVA_ERR_BAD_PAYLOAD;
        const auto* k = input.data() + sizeof(combined);
        const auto* v = k + plane_bytes;
        const auto* positions = reinterpret_cast<const int32_t*>(v + plane_bytes);
        const auto* q = reinterpret_cast<const float*>(v + plane_bytes + positions_bytes);
        const uint64_t end = uint64_t(it->second.range_origin) + upload.start_row + upload.n_rows;
        for (uint32_t t = 0; t < req.n_tokens; ++t) {
            if (positions[t] < int64_t(it->second.range_origin) || uint64_t(positions[t]) >= end ||
                (t && int64_t(positions[t]) != int64_t(positions[t - 1]) + 1))
                return RKVA_ERR_BAD_POSITION;
        }
        if (uint64_t(positions[req.n_tokens - 1]) + 1 != end)
            return RKVA_ERR_BAD_POSITION;
        output.resize(q_bytes + q_heads * sizeof(float));
        auto* out = reinterpret_cast<float*>(output.data());
        auto* lse = reinterpret_cast<float*>(output.data() + q_bytes);
        std::string error;
        const auto start = clock_type::now();
        // First isolate network batching: retain the qualified upload/attention
        // engine calls. Queue/fence coalescing is measured as a separate change.
        if (!state.engine->upload_range(req.session_id, req.layer_id, upload.start_row,
                                        upload.n_rows, reinterpret_cast<const uint16_t*>(k),
                                        reinterpret_cast<const uint16_t*>(v), &error) ||
            !state.engine->attend_range(req.session_id, req.layer_id, req.n_tokens, positions, q,
                                        out, lse, &error)) {
            output.clear();
            event_engine_error(error);
            return state.engine->session_invalid(req.session_id) ? RKVA_ERR_GPU
                                                                 : RKVA_ERR_BAD_POSITION;
        }
        compute_us = elapsed_us(start);
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    case RKVA_SNAPSHOT_SAVE:
    case RKVA_SNAPSHOT_RESTORE: {
        if (input.size() != sizeof(rkva_snapshot_req) || req.n_tokens || req.layer_id)
            return RKVA_ERR_BAD_PAYLOAD;
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !state.engine || !it->second.valid ||
            !it->second.range_configured)
            return RKVA_ERR_BAD_SESSION;
        rkva_snapshot_req snapshot{};
        std::memcpy(&snapshot, input.data(), sizeof(snapshot));
        if (snapshot.revision != 1 || snapshot.flags || snapshot.reserved || !snapshot.live_rows ||
            snapshot.live_rows > it->second.capacity)
            return RKVA_ERR_BAD_PAYLOAD;
        rkva_snapshot_ack ack{};
        std::string error;
        const bool restoring = req.op == RKVA_SNAPSHOT_RESTORE;
        const auto start = clock_type::now();
        if (!state.engine->disk_snapshot(req.session_id, g_snapshot_local_folder, snapshot,
                                         restoring, &ack, &error)) {
            event_engine_error(error);
            if (state.engine->session_invalid(req.session_id)) {
                it->second.valid = false;
                return RKVA_ERR_STATE;
            }
            return restoring ? RKVA_ERR_SNAPSHOT_MISS : RKVA_ERR_SNAPSHOT_IO;
        }
        // Engine publishes counters only after all GPU transfers complete.
        it->second.high_water = -1; // absolute range origin is owned by the engine
        compute_us = elapsed_us(start);
        prepare_us = elapsed_us(t0);
        output.resize(sizeof(ack));
        std::memcpy(output.data(), &ack, sizeof(ack));
        return RKVA_OK;
    }
    case RKVA_PROBE: {
        if (req.version != RKVA_VERSION)
            return RKVA_ERR_UNSUPPORTED_OP;
        rkva_probe_req probe{};
        if (input.size() < sizeof(probe))
            return RKVA_ERR_BAD_PAYLOAD;
        std::memcpy(&probe, input.data(), sizeof(probe));
        if (probe.probe_id != 5)
            return RKVA_ERR_UNSUPPORTED_OP;
        rkva_bulk_fp8::ProbeLayout layout;
        if (req.session_id != RKVA_SESSION_NONE || req.layer_id || req.n_tokens ||
            !rkva_bulk_fp8::validate_probe(input.data(), input.size(), layout))
            return RKVA_ERR_BAD_PAYLOAD;
        if (!state.hello_done || !state.engine)
            return RKVA_ERR_STATE;
        if (!state.engine->ensure_bulk_fp8_gpu_pipelines())
            return RKVA_ERR_UNSUPPORTED_OP;
        std::string error;
        const auto start = clock_type::now();
        if (!state.engine->probe_bulk_fp8_pack(input.data(), input.size(), output, &error)) {
            output.clear();
            event_engine_error(error);
            return RKVA_ERR_GPU;
        }
        compute_us = elapsed_us(start);
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    default:
        // Parity harness lands with the KVarN kernels; explicit failure now.
        (void)compute_us;
        prepare_us = elapsed_us(t0);
        return RKVA_ERR_UNSUPPORTED_OP;
    }
    prepare_us = elapsed_us(t0);
}

void client_loop(SOCKET sock, remote_state& state, const std::string& log_path) {
    uint64_t previous_response_us = 0;
    std::vector<uint8_t> input, output;
    for (;;) {
        const auto start = clock_type::now();
        rkva_request req{};
        if (!recv_all(sock, &req, sizeof(req))) {
            break;
        }
        const uint64_t idle_us = elapsed_us(start);
        // Generous cap: prefill batches can be multi-MB (Q/K/V rows).
        static constexpr uint32_t MAX_PAYLOAD = 256u * 1024u * 1024u;
        if (req.magic != RKVA_MAGIC ||
            (req.version != RKVA_VERSION && req.version != RKVA_VERSION_LEGACY) ||
            req.payload_bytes > MAX_PAYLOAD) {
            break;
        }
        input.resize(req.payload_bytes);
        const auto payload_start = clock_type::now();
        if (req.payload_bytes && !recv_all(sock, input.data(), input.size())) {
            break;
        }
        const uint64_t payload_receive_us = elapsed_us(payload_start);
        output.clear();
        uint64_t prepare_us = 0, compute_us = 0;
        const auto handle_start = clock_type::now();
        int rc = handle_request(state, req, input, output, prepare_us, compute_us);
        if (state.engine && state.engine->submission_unconfirmed()) {
            // Do not send even an error header on a blocking socket: an
            // unresponsive peer must not delay owner-side GPU retirement.
            event(log_path, "{\"phase\":\"request\",\"status\":\"gpu_submission_unconfirmed\","
                            "\"connection_closed_without_response\":true}");
            break;
        }
        const uint64_t handle_us = elapsed_us(handle_start);

        const rkva_response rsp{
            RKVA_MAGIC, req.version, req.op,     req.cycle_id,        rc, (uint32_t)output.size(),
            idle_us,    prepare_us,  compute_us, previous_response_us};
        const auto response_start = clock_type::now();
        if (!send_all(sock, &rsp, sizeof(rsp)) ||
            (output.size() && !send_all(sock, output.data(), output.size()))) {
            break;
        }
        previous_response_us = elapsed_us(response_start);
        if (state.profile_enabled && req.op != RKVA_RANGE_PROFILE && req.op < 32) {
            auto& counter = state.rpc_profile[req.op];
            ++counter.calls;
            counter.request_bytes += sizeof(req) + input.size();
            counter.response_bytes += sizeof(rsp) + output.size();
            counter.payload_receive_us += payload_receive_us;
            counter.handle_us += handle_us;
            counter.send_enqueue_us += previous_response_us;
        }
        if (state.trace_requests)
            event(log_path, "{\"phase\":\"request\",\"cycleId\":" + std::to_string(req.cycle_id) +
                                ",\"op\":" + std::to_string(req.op) +
                                ",\"session\":" + std::to_string(req.session_id) +
                                ",\"layer\":" + std::to_string(req.layer_id) +
                                ",\"nTokens\":" + std::to_string(req.n_tokens) +
                                ",\"payloadBytes\":" + std::to_string(req.payload_bytes) +
                                ",\"responseBytes\":" + std::to_string(output.size()) +
                                ",\"status\":" + std::to_string(rc) +
                                ",\"idleUs\":" + std::to_string(idle_us) +
                                ",\"prepareUs\":" + std::to_string(prepare_us) +
                                ",\"computeUs\":" + std::to_string(compute_us) +
                                ",\"handleUs\":" + std::to_string(handle_us) +
                                ",\"responseUs\":" + std::to_string(previous_response_us) + "}");
    }
}

// A single poisoned engine remains owned by its original worker stack. This
// predicate is deliberately never signaled: process restart releases the OS
// context. No busy polling, global leak registry, or repeated engine creation.
[[noreturn]] void retain_engine_until_process_restart() {
    std::mutex mutex;
    std::condition_variable parked;
    std::unique_lock<std::mutex> lock(mutex);
    parked.wait(lock, [] { return false; });
    std::terminate(); // defensive: never unwind the owning state on this path
}

} // namespace

void serve_remote_attn(const std::string& local_folder) {
    const std::string log_path = local_folder + "\\remote-attn.jsonl";
    // App::OnLaunched can start another detached listener on reactivation.
    // The owner lock is retained with a poisoned engine, so closing its socket
    // cannot allow a second UI activation to create another GPU engine.
    // Intentionally process-lifetime: a parked worker may hold this at exit,
    // so static destruction must never destroy an owned mutex. One allocation,
    // no engine registry or accumulating retention state.
    static std::mutex* const listener_owner = new std::mutex;
    std::unique_lock<std::mutex> owner_lock(*listener_owner, std::try_to_lock);
    if (!owner_lock.owns_lock()) {
        event(log_path,
              "{\"phase\":\"listen\",\"status\":\"owner_already_active_or_restart_required\"}");
        return;
    }
    g_log_path = log_path;
    g_snapshot_local_folder = local_folder;
    constexpr uint16_t port = 50054;

    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa)) {
        event(log_path, "{\"phase\":\"listen\",\"status\":\"wsa_failed\"}");
        return;
    }
    SOCKET listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listener == INVALID_SOCKET) {
        event(log_path, "{\"phase\":\"listen\",\"status\":\"socket_failed\"}");
        WSACleanup();
        return;
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);
    if (bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == SOCKET_ERROR ||
        listen(listener, 1) == SOCKET_ERROR) {
        event(log_path, "{\"phase\":\"listen\",\"status\":\"bind_failed\",\"port\":" +
                            std::to_string(port) + "}");
        closesocket(listener);
        WSACleanup();
        return;
    }
    event(log_path,
          "{\"phase\":\"listen\",\"status\":\"ready\",\"port\":" + std::to_string(port) + "}");

    remote_state state;
    for (;;) {
        if (state.engine && state.engine->submission_unconfirmed())
            break;
        SOCKET client = accept(listener, nullptr, nullptr);
        if (client == INVALID_SOCKET) {
            break;
        }
        const int one = 1;
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one),
                   sizeof(one));
        int rcvbuf = 8 << 20, sndbuf = 8 << 20;
        setsockopt(client, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&rcvbuf),
                   sizeof(rcvbuf));
        setsockopt(client, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&sndbuf),
                   sizeof(sndbuf));
        event(log_path, "{\"phase\":\"accept\",\"status\":\"connected\"}");
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.hello_done = false;
            state.sessions.clear();
            state.engine.reset(); // freed and re-created at the next HELLO
            state.engine_f16 = false;
        }
        client_loop(client, state, log_path);
        closesocket(client);
        event(log_path, "{\"phase\":\"accept\",\"status\":\"disconnected\"}");
    }
    closesocket(listener);
    WSACleanup();
    if (state.engine && !state.engine->confirm_execution_retired()) {
        event(log_path, "{\"phase\":\"gpu_retirement\",\"status\":\"unconfirmed_engine_retained\","
                        "\"restart_required\":true}");
        // state.engine and owner_lock both stay alive; no delete/reset occurs.
        retain_engine_until_process_restart();
    }
}
