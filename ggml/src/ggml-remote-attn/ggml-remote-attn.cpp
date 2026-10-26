// ggml-remote-attn backend implementation.
//
// Executes GGML_OP_REMOTE_ATTN by forwarding Q/K/V/pos over a persistent TCP
// connection to a remote KVarN attention server (RKVA protocol). Native FP8 can
// retain CUDA Q/OUT and use pinned byte staging; other modes use host buffers.
//
// Only POSIX hosts are supported (the .57 station is Linux). On any other
// platform ggml_backend_remote_attn_init returns NULL and the feature is
// disabled, leaving the local baseline untouched.

#include "ggml-remote-attn.h"
#include "ggml-remote-attn-cuda.h"
#include "ggml-remote-attn-trace.h"
#include "ggml-local-split.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "remote-attn-protocol.h"
#include "rkva-kvarn-producer.h"
#include "rkva-q4-producer.h"
#include "rkva-q8-producer.h"

#if defined(_WIN32)
#include <malloc.h> // _aligned_malloc / _aligned_free
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
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

namespace {
thread_local ggml_remote_attn_trace_event rkva_trace_tls{};
std::atomic<uint64_t> rkva_trace_graph_counter{0};
struct rkva_trace_buffer {
    std::mutex mutex;
    std::vector<ggml_remote_attn_trace_event> events;
    size_t capacity = 262144;
    uint64_t dropped = 0, written = 0;
    std::string path;
    rkva_trace_buffer() {
        const char * prefix = std::getenv("GGML_RKVA_TRACE_FILE");
        if (!prefix || !*prefix) return;
        if (const char * count = std::getenv("GGML_RKVA_TRACE_CAPACITY"))
            capacity = std::clamp<size_t>(std::strtoull(count, nullptr, 10), 1024, 1048576);
#ifndef _WIN32
        path = std::string(prefix) + "." + std::to_string(::getpid()) + ".jsonl";
#else
        path = std::string(prefix) + ".jsonl";
#endif
        try { events.reserve(capacity); } catch (...) { capacity = 0; }
    }
};
rkva_trace_buffer & rkva_trace_storage() { static rkva_trace_buffer buffer; return buffer; }
std::string rkva_trace_escape(const char * text) {
    std::string out;
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(text); *p; ++p) {
        if (*p == '"' || *p == '\\') out += '\\';
        out += *p < 32 ? ' ' : char(*p);
    }
    return out;
}
}

bool ggml_remote_attn_trace_enabled(void) {
    static const bool enabled = [] { const char * p = std::getenv("GGML_RKVA_TRACE_FILE"); return p && *p; }();
    return enabled;
}
void ggml_remote_attn_trace_request(uint64_t id) {
    if (ggml_remote_attn_trace_enabled()) rkva_trace_tls.request_id = id;
}
void ggml_remote_attn_trace_graph(uint64_t uid) {
    if (!ggml_remote_attn_trace_enabled()) return;
    rkva_trace_tls.graph_run = ++rkva_trace_graph_counter;
    rkva_trace_tls.graph_uid = uid;
    rkva_trace_tls.ticket = rkva_trace_tls.epoch = rkva_trace_tls.cycle = 0;
    rkva_trace_tls.first_pos = rkva_trace_tls.last_pos = -1;
    rkva_trace_tls.model_layer = -1;
    rkva_trace_tls.nq = 0;
    rkva_trace_tls.layer = UINT32_MAX;
}
ggml_remote_attn_trace_event ggml_remote_attn_trace_identity(const char * phase, const ggml_tensor * node) {
    if (!ggml_remote_attn_trace_enabled()) return {};
    ggml_remote_attn_trace_event out = rkva_trace_tls;
    out.begin_us = out.end_us = out.bytes = out.value = out.flags = 0;
    out.gpu_us = 0;
    std::snprintf(out.phase, sizeof(out.phase), "%s", phase ? phase : "");
    if (node) {
        std::snprintf(out.node, sizeof(out.node), "%s", node->name);
        if (node->op == GGML_OP_REMOTE_ATTN) {
            const auto * range = ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_WAIT) ? node->src[0] : node;
            out.layer = uint32_t(ggml_get_op_params_i32(range, GGML_REMOTE_ATTN_PARAM_LAYER_ID));
            if (range->src[0]) out.nq = uint32_t(range->src[0]->ne[2]);
        } else if (const char * dash = std::strrchr(node->name, '-')) {
            char * end = nullptr;
            const long layer = std::strtol(dash + 1, &end, 10);
            if (end != dash + 1 && !*end && layer >= 0) out.model_layer = int32_t(layer);
        }
    }
    return out;
}
ggml_remote_attn_trace_event ggml_remote_attn_trace_context(const ggml_remote_attn_trace_event * context) {
    const auto old = rkva_trace_tls;
    if (context) rkva_trace_tls = *context;
    return old;
}
void ggml_remote_attn_trace_record(const ggml_remote_attn_trace_event * event) {
    if (!ggml_remote_attn_trace_enabled() || !event) return;
    auto & trace = rkva_trace_storage();
    std::lock_guard<std::mutex> lock(trace.mutex);
    if (trace.events.size() < trace.capacity) trace.events.push_back(*event);
    else ++trace.dropped;
}
void ggml_remote_attn_trace_flush(void) {
    if (!ggml_remote_attn_trace_enabled()) return;
    auto & trace = rkva_trace_storage();
    std::lock_guard<std::mutex> lock(trace.mutex);
    FILE * file = std::fopen(trace.path.c_str(), trace.written ? "a" : "w");
    if (!file) { GGML_LOG_ERROR("RKVA trace: cannot open %s\n", trace.path.c_str()); return; }
    for (const auto & e : trace.events) {
        const auto phase = rkva_trace_escape(e.phase), node = rkva_trace_escape(e.node);
        std::fprintf(file, "{\"schema\":\"rkva-host-trace-v1\",\"phase\":\"%s\",\"node\":\"%s\","
            "\"begin_us\":%llu,\"end_us\":%llu,\"request\":%llu,\"graph_run\":%llu,\"graph_uid\":%llu,"
            "\"cycle\":%llu,\"ticket\":%llu,\"epoch\":%llu,\"session\":%u,\"layer\":%u,\"nq\":%u,"
            "\"first_pos\":%d,\"last_pos\":%d,\"model_layer\":%d,\"bytes\":%llu,\"value\":%llu,\"flags\":%u,\"gpu_us\":%.6f}\n",
            phase.c_str(), node.c_str(), (unsigned long long)e.begin_us, (unsigned long long)e.end_us,
            (unsigned long long)e.request_id, (unsigned long long)e.graph_run, (unsigned long long)e.graph_uid,
            (unsigned long long)e.cycle, (unsigned long long)e.ticket, (unsigned long long)e.epoch,
            e.session, e.layer, e.nq, e.first_pos, e.last_pos, e.model_layer, (unsigned long long)e.bytes,
            (unsigned long long)e.value, e.flags, e.gpu_us);
    }
    trace.written += trace.events.size();
    std::fprintf(file, "{\"schema\":\"rkva-host-trace-status-v1\",\"events_written\":%llu,\"dropped\":%llu}\n",
        (unsigned long long)trace.written, (unsigned long long)trace.dropped);
    std::fclose(file);
    trace.events.clear();
}

#if defined(__AVX2__) && defined(__F16C__)
#include <immintrin.h>
#endif

namespace {

static bool rkva_cuda_fp8_bucket_permitted(int64_t nq) {
    const char * option = std::getenv("GGML_RKVA_CUDA_FP8");
    // Only the new explicit mode restricts the existing device-pointer path.
    // Preserve the legacy runtime behavior of0/1/unset; graph placement for0
    // continues to use the CPU backend through cuda_enabled().
    return !option || std::strcmp(option, "prefill") || nq > 9;
}

static const ggml_remote_attn_cuda_fp8_interface_v2 * rkva_cuda_fp8_api() {
    static std::mutex mutex;
    static const ggml_remote_attn_cuda_fp8_interface_v2 * api = nullptr;
    std::lock_guard<std::mutex> lock(mutex);
    if (!api) {
        auto * reg = ggml_backend_reg_by_name("CUDA");
        auto getter = reg ? reinterpret_cast<ggml_backend_cuda_remote_attn_fp8_interface_v2_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_remote_attn_fp8_interface_v2")) : nullptr;
        if (getter) api = getter();
    }
    return api;
}

static const ggml_remote_attn_cuda_fp8_interface_v3 * rkva_cuda_fp8_small_api() {
    static std::mutex mutex;
    static const ggml_remote_attn_cuda_fp8_interface_v3 * api = nullptr;
    std::lock_guard<std::mutex> lock(mutex);
    if (!api) {
        auto * reg = ggml_backend_reg_by_name("CUDA");
        auto getter = reg ? reinterpret_cast<ggml_backend_cuda_remote_attn_fp8_interface_v3_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_remote_attn_fp8_interface_v3")) : nullptr;
        if (getter) api = getter();
    }
    return api;
}

static bool rkva_cuda_fp8_mapped_requested() {
    const char * option = std::getenv("GGML_RKVA_CUDA_FP8");
    return option && !std::strcmp(option, "mapped");
}

static const ggml_remote_attn_cuda_retirement_interface_v1 * rkva_cuda_retirement_api() {
    auto * reg = ggml_backend_reg_by_name("CUDA");
    auto getter = reg ? reinterpret_cast<ggml_backend_cuda_remote_attn_retirement_interface_v1_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_remote_attn_retirement_interface_v1")) : nullptr;
    return getter ? getter() : nullptr;
}

static thread_local void * rkva_cuda_producer_event = nullptr;

// A slot owns its pinned network/staging allocations until retirement. CPU-only
// operation retains the ordinary vector implementation and no CUDA dependency.
class rkva_byte_buffer {
    std::vector<uint8_t> host;
    const ggml_remote_attn_cuda_fp8_interface_v2 * api = nullptr;
    uint8_t * pinned = nullptr;
    size_t used = 0, allocated = 0;
    uint64_t zeroed = 0;
public:
    rkva_byte_buffer() = default;
    rkva_byte_buffer(const rkva_byte_buffer &) = delete;
    rkva_byte_buffer & operator=(const rkva_byte_buffer &) = delete;
    ~rkva_byte_buffer() { if (pinned) api->free_pinned(pinned); }
    void use_pinned(const ggml_remote_attn_cuda_fp8_interface_v2 * value) {
        if (api) return;
        if (!host.empty()) throw std::logic_error("active RKVA buffer cannot change allocator");
        api = value;
        const size_t initial = host.capacity();
        reserve(initial);
        std::vector<uint8_t>().swap(host);
    }
    size_t size() const { return api ? used : host.size(); }
    size_t capacity() const { return api ? allocated : host.capacity(); }
    uint64_t zeroed_bytes() const { return zeroed; }
    uint8_t * data() { return api ? pinned : host.data(); }
    const uint8_t * data() const { return api ? pinned : host.data(); }
    void reserve(size_t bytes) {
        if (!api) { host.reserve(bytes); return; }
        if (bytes <= allocated) return;
        auto * replacement = static_cast<uint8_t *>(api->alloc_pinned(bytes));
        if (!replacement) throw std::bad_alloc();
        if (used) std::memcpy(replacement, pinned, used);
        if (pinned) api->free_pinned(pinned);
        pinned = replacement;
        allocated = bytes;
    }
    void resize(size_t bytes) {
        if (!api) { if (bytes > host.size()) zeroed += bytes - host.size(); host.resize(bytes); return; }
        if (bytes > allocated) reserve(std::max(bytes, allocated <= SIZE_MAX / 2 ? allocated * 2 : bytes));
        if (bytes > used) { std::memset(pinned + used, 0, bytes - used); zeroed += bytes - used; }
        used = bytes;
    }
    void clear() { if (api) used = 0; else host.clear(); }
    // Device failure may leave a DMA targeting these pages. Detach ownership
    // only during destruction, when successful retirement cannot be established.
    size_t quarantine_pinned() {
        const size_t bytes = pinned ? allocated : 0;
        pinned = nullptr;
        used = allocated = 0;
        return bytes;
    }
};

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

static inline uint32_t rkva_round_shift_even(uint32_t value, unsigned shift) {
    const uint32_t base = value >> shift;
    const uint32_t remainder = value & ((uint32_t(1) << shift) - 1);
    const uint32_t halfway = uint32_t(1) << (shift - 1);
    return base + (remainder > halfway || (remainder == halfway && (base & 1)));
}

// Direct F32 to finite E5M2: nearest-even, finite saturation, signed zero.
static inline bool rkva_f32_to_fp8_scalar(float value, uint8_t & result) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t magnitude = bits & 0x7fffffffu;
    const uint8_t sign = uint8_t((bits >> 24) & 0x80u);
    if (magnitude >= 0x7f800000u) return false;
    if (magnitude >= 0x47600000u) {
        result = sign | 0x7bu;
        return true;
    }
    if (magnitude <= 0x37000000u) {
        result = sign;
        return true;
    }
    const uint32_t exponent = magnitude >> 23;
    const uint32_t mantissa = magnitude & 0x7fffffu;
    const uint32_t encoded = magnitude < 0x38800000u ?
        rkva_round_shift_even(mantissa | 0x800000u, 134u - exponent) :
        ((exponent - 112u) << 2) + rkva_round_shift_even(mantissa, 21);
    result = sign | uint8_t(encoded);
    return true;
}

// FP8 E5M2 wire codec: finite saturation and direct nearest-even rounding.
static inline bool rkva_f32_to_fp8_batch(const float * src, uint8_t * dst, size_t n) {
#if defined(__AVX2__) && defined(__F16C__)
    size_t i = 0;
    const __m256 max_fp8  = _mm256_set1_ps(57344.0f);
    const __m256 min_fp8  = _mm256_set1_ps(-57344.0f);
    const __m256 inf_ps   = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 abs_v = _mm256_andnot_ps(sign_mask, v);
        __m256 is_finite = _mm256_cmp_ps(abs_v, inf_ps, _CMP_LT_OQ);
        int mask = _mm256_movemask_ps(is_finite);
        if (__builtin_expect(mask != 0xFF, 0)) return false;
        // Explicit clamp finite values exceeding 57344.0f to 57344.0f (0x7b / 0xfb)
        v = _mm256_min_ps(v, max_fp8);
        v = _mm256_max_ps(v, min_fp8);
        __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m128i retained_lsb = _mm_and_si128(_mm_srli_epi16(h, 8), _mm_set1_epi16(1));
        __m128i rounded = _mm_add_epi16(h, _mm_add_epi16(_mm_set1_epi16(0x007f), retained_lsb));
        __m128i shifted = _mm_srli_epi16(rounded, 8);
        __m128i packed = _mm_packus_epi16(shifted, shifted);
        _mm_storel_epi64((__m128i*)(dst + i), packed);
        // A first rounding can only change the FP8 decision at an F16 midpoint.
        const __m128i midpoint = _mm_cmpeq_epi16(_mm_and_si128(h, _mm_set1_epi16(0x00ff)),
                                               _mm_set1_epi16(0x0080));
        const int midpoint_mask = _mm_movemask_epi8(midpoint);
        if (__builtin_expect(midpoint_mask != 0, 0)) {
            for (size_t lane = 0; lane < 8; ++lane) {
                if ((midpoint_mask & (3 << (2 * lane))) &&
                    !rkva_f32_to_fp8_scalar(src[i + lane], dst[i + lane])) return false;
            }
        }
    }
    for (; i < n; ++i) {
        if (!rkva_f32_to_fp8_scalar(src[i], dst[i])) return false;
    }
    return true;
#else
    for (size_t i = 0; i < n; ++i) {
        if (!rkva_f32_to_fp8_scalar(src[i], dst[i])) return false;
    }
    return true;
#endif
}

static inline bool rkva_fp8_to_f32_batch(const uint8_t * src, float * dst, size_t n) {
#if defined(__AVX2__) && defined(__F16C__)
    size_t i = 0;
    const __m128i exp_mask = _mm_set1_epi16(0x7c00);
    for (; i + 8 <= n; i += 8) {
        __m128i b = _mm_loadl_epi64((const __m128i*)(src + i));
        __m128i h = _mm_slli_epi16(_mm_cvtepu8_epi16(b), 8);
        __m128i exps = _mm_and_si128(h, exp_mask);
        __m128i is_inf_nan = _mm_cmpeq_epi16(exps, exp_mask);
        if (__builtin_expect(!_mm_testz_si128(is_inf_nan, is_inf_nan), 0)) return false;
        __m256 f = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, f);
    }
    for (; i < n; ++i) {
        uint16_t h = uint16_t(src[i]) << 8;
        if ((h & 0x7c00u) == 0x7c00u) return false;
        dst[i] = f16_bits_to_f32(h);
    }
    return true;
#else
    for (size_t i = 0; i < n; ++i) {
        uint16_t h = uint16_t(src[i]) << 8;
        if ((h & 0x7c00u) == 0x7c00u) return false;
        dst[i] = f16_bits_to_f32(h);
    }
    return true;
#endif
}

// Option B: SplitZip FP16 lossless exponent-split codec
static const uint8_t kSplitZipTopExponents[15] = {
    14, 15, 13, 12, 16, 11, 10, 17, 9, 18, 8, 19, 7, 20, 6
};

struct rkva_splitzip_header {
    uint32_t n_elements;
    uint32_t n_escapes;
    uint32_t compressed_bytes;
};

static size_t rkva_splitzip_encode(const uint16_t * src, size_t n, std::vector<uint8_t> & out) {
    uint8_t exp_to_code[32];
    std::memset(exp_to_code, 0x0F, sizeof(exp_to_code));
    for (uint8_t code = 0; code < 15; ++code) {
        exp_to_code[kSplitZipTopExponents[code]] = code;
    }

    std::vector<uint8_t> escapes;
    out.resize(sizeof(rkva_splitzip_header) + n * 4 + 16);
    auto * hdr = reinterpret_cast<rkva_splitzip_header*>(out.data());
    hdr->n_elements = (uint32_t)n;
    hdr->n_escapes = 0;

    uint8_t * p = out.data() + sizeof(rkva_splitzip_header);
    for (size_t i = 0; i < n; ++i) {
        p[i] = (uint8_t)(src[i] & 0xFF);
    }
    p += n;
    for (size_t i = 0; i < n; i += 2) {
        uint8_t e0 = (src[i] >> 10) & 0x1F;
        uint8_t c0 = exp_to_code[e0];
        uint8_t e1 = (i + 1 < n) ? ((src[i + 1] >> 10) & 0x1F) : 0;
        uint8_t c1 = (i + 1 < n) ? exp_to_code[e1] : 0;
        *p++ = (c0 << 4) | (c1 & 0x0F);
        if (c0 == 0x0F) escapes.push_back((uint8_t)((src[i] >> 8) & 0xFF));
        if (c1 == 0x0F && i + 1 < n) escapes.push_back((uint8_t)((src[i + 1] >> 8) & 0xFF));
    }
    for (size_t i = 0; i < n; i += 2) {
        uint8_t s0 = ((src[i] >> 15) & 1) << 2 | ((src[i] >> 8) & 3);
        uint8_t s1 = (i + 1 < n) ? (((src[i + 1] >> 15) & 1) << 2 | ((src[i + 1] >> 8) & 3)) : 0;
        *p++ = (s0 << 4) | (s1 & 0x0F);
    }
    if (!escapes.empty()) {
        std::memcpy(p, escapes.data(), escapes.size());
        p += escapes.size();
    }
    hdr->n_escapes = (uint32_t)escapes.size();
    hdr->compressed_bytes = (uint32_t)(p - out.data());
    out.resize(hdr->compressed_bytes);
    return out.size();
}

static bool rkva_splitzip_decode(const uint8_t * in, size_t in_bytes, uint16_t * dst, size_t n) {
    if (!in || (!dst && n > 0) || in_bytes < sizeof(rkva_splitzip_header)) return false;
    rkva_splitzip_header header{};
    std::memcpy(&header, in, sizeof(header));
    const auto * hdr = &header;
    if (hdr->n_elements != n || hdr->n_escapes > n) return false;
    const uint64_t count = hdr->n_elements;
    const uint64_t nibble_bytes = (count + 1) / 2;
    const uint64_t required_bytes = sizeof(header) + count + 2 * nibble_bytes + hdr->n_escapes;
    if (hdr->compressed_bytes != required_bytes || in_bytes < required_bytes) return false;

    const uint8_t * p = in + sizeof(rkva_splitzip_header);
    const uint8_t * low_bytes = p; p += n;
    const uint8_t * exp_nibbles = p; p += (n + 1) / 2;
    const uint8_t * sm_nibbles = p; p += (n + 1) / 2;
    const uint8_t * esc_ptr = p;

    size_t esc_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        uint8_t lo = low_bytes[i];
        uint8_t exp_pair = exp_nibbles[i / 2];
        uint8_t code = (i % 2 == 0) ? (exp_pair >> 4) : (exp_pair & 0x0F);

        uint8_t sm_pair = sm_nibbles[i / 2];
        uint8_t sm3 = (i % 2 == 0) ? (sm_pair >> 4) : (sm_pair & 0x0F);
        uint16_t sign = (sm3 >> 2) & 1;
        uint16_t top_m = sm3 & 3;

        uint16_t exp;
        if (code == 0x0F) {
            if (esc_idx >= hdr->n_escapes) return false;
            uint8_t raw_hi = esc_ptr[esc_idx++];
            exp = (raw_hi >> 2) & 0x1F;
        } else {
            exp = kSplitZipTopExponents[code];
        }

        uint16_t h = (sign << 15) | (exp << 10) | (top_m << 8) | lo;
        dst[i] = h;
    }
    if (esc_idx != hdr->n_escapes) return false;
    return true;
}

// ---------------------------------------------------------------------------
// persistent socket
// ---------------------------------------------------------------------------

struct rkva_socket {
    std::mutex lifetime_mutex;
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
        std::lock_guard<std::mutex> lock(lifetime_mutex);
#ifndef _WIN32
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
#endif
    }

    // Cancellation wakes the owning recv without releasing/reusing its fd.
    void interrupt() {
        std::lock_guard<std::mutex> lock(lifetime_mutex);
#ifndef _WIN32
        if (fd >= 0) ::shutdown(fd, SHUT_RDWR);
#endif
    }

    void assign(int value) {
        std::lock_guard<std::mutex> lock(lifetime_mutex);
#ifndef _WIN32
        fd = value;
#else
        (void) value;
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
            msghdr message {};
            message.msg_iov = iov + cur;
            message.msg_iovlen = size_t(count - cur);
            const ssize_t n = ::sendmsg(fd, &message, MSG_NOSIGNAL);
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
    uint64_t cuda_pack_calls = 0;
    uint64_t cuda_unpack_calls = 0;
    uint64_t cuda_small_pack_calls = 0;
    uint64_t cuda_small_unpack_calls = 0;
    uint64_t cohort_calls = 0;
    uint64_t cohort_entries = 0;
    uint64_t cuda_context_creates = 0;
    uint64_t cuda_context_phase_releases = 0; // slot-owner drops, not physical frees
    uint64_t cuda_context_shared_acquires = 0;
    uint64_t cuda_context_shared_releases = 0;
    // Last strong owner dropped during an explicit phase/mode transition.
    // Teardown drops and CUDA destroy success are deliberately not inferred.
    uint64_t cuda_context_shared_last_releases = 0;
    uint64_t q_f32_readback_bytes_avoided = 0;
    uint64_t out_f32_upload_bytes_avoided = 0;
    int64_t range_query_min = -1;
    int64_t range_query_max = -1;
    std::vector<uint64_t> rpc_us;  // for p50/p95
};

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 2-slot async queue with single TCP owner (native KVarN4 bulk range only)
// ---------------------------------------------------------------------------

enum class rkva_slot_state {
    FREE,
    PREPARING, // slot reserved by submitter, preparing payload
    PENDING,   // submitted, worker has not started the RPC
    RUNNING,   // worker owns the persistent TCP stream and runs RPC / decodes reply
    DONE_OK,   // reply decoded into output; ready for consumer wait_node
    DONE_ERR,  // RPC or validation failed; fail closed
};

struct rkva_async_slot {
    uint32_t        id = 0;
    uint32_t        session = RKVA_SESSION_NONE; // captured at prepare (worker must not read ctx->active_session)
    rkva_slot_state state = rkva_slot_state::FREE;
    uint64_t        epoch = 0;
    uint64_t        ticket = 0;
    uint32_t        cohort_count = 0;
    uint32_t        cohort_index = 0;
    ggml_remote_attn_trace_event trace{};
    uint64_t pending_us = 0;
    uint64_t response_cycle = 0; // worker writes before DONE; wait reads after acquire

    rkva_byte_buffer payload;              // pinned TX when CUDA FP8 is active
    rkva_byte_buffer reply;                // pinned RX when CUDA FP8 is active
    rkva_byte_buffer query_fp8;
    const ggml_remote_attn_cuda_fp8_interface_v2 * cuda_api = nullptr;
    const ggml_remote_attn_cuda_fp8_interface_v3 * cuda_small_api = nullptr;
    bool cuda_small = false; // captured for this submission; never reread policy on worker
    void * cuda_context = nullptr; // alias; strong lease below owns shared contexts
    std::shared_ptr<void> cuda_shared_lease;
    bool cuda_shared_requested = false; // captured from this submission's local bit
    int cuda_device = -1;
    // A failed helper cannot establish successful retirement for future reuse.
    // Keep this across reset/reconnect, including failures that were numerical.
    bool cuda_context_retired = true;
    std::vector<uint8_t>  segment_scratch;  // per-slot scratch for owner->append
    std::vector<int32_t>  pos_scratch;      // positions scratch for this slot
    std::vector<float>    f32_scratch;      // queries scratch for this slot
    std::vector<float>    original_k;       // keys scratch for this slot
    std::vector<float>    original_v;       // values scratch for this slot
    std::vector<uint16_t> decode_f16;       // intermediate f16 decode scratch

    struct ggml_tensor * node = nullptr;   // submitted range node
    uint8_t *            output = nullptr;
    size_t               output_stride = 0;
    uint32_t             layer = 0;
    uint32_t             nq = 0;
    uint32_t             segments = 0;
    uint32_t             start_rows = 0;
    uint32_t             end_rows = 0;
    uint32_t             out_width = 0;
    uint64_t             upload_bytes = 0;
    int32_t              first_pos = 0;
    int32_t              last_pos = 0;
    bool                 bulk_f16 = false;
    rkva_wire            wire = RKVA_WIRE_F16;

    ~rkva_async_slot() {
        bool retired = cuda_context_retired;
        if (!retired && cuda_context) {
            const auto * retirement = rkva_cuda_retirement_api();
            retired = retirement && retirement->base == cuda_api &&
                      retirement->retire_context(cuda_context);
        }
        if (!retired) {
            const size_t quarantined = payload.quarantine_pinned() + reply.quarantine_pinned() +
                                       query_fp8.quarantine_pinned();
            GGML_LOG_ERROR("RKVA: quarantined %zu pinned bytes after unconfirmed CUDA retirement\n", quarantined);
        }
        release_cuda_owner();
    }

    void release_cuda_owner() {
        void * previous = cuda_context;
        cuda_context = nullptr;
        cuda_device = -1;
        if (cuda_shared_lease) {
            // A RUNNING peer retains its own strong lease. The last owner alone
            // invokes the API captured in the deleter; no backend pointer lives
            // in that deleter or survives the backend's mutex/stat destruction.
            cuda_shared_lease.reset();
        } else if (previous) {
            cuda_api->destroy_context(previous);
        }
    }

    bool prepare_cuda(int device) {
        if (!cuda_context_retired || cuda_shared_lease) return false;
        if (cuda_context && device != cuda_device) release_cuda_owner();
        if (!cuda_context) {
            cuda_api = rkva_cuda_fp8_api();
            if (!cuda_api) return false;
            cuda_context = cuda_api->create_context(device);
            if (!cuda_context) return false;
            cuda_device = device;
            payload.use_pinned(cuda_api);
            reply.use_pinned(cuda_api);
            query_fp8.use_pinned(cuda_api);
        }
        return true;
    }

    void reset() {
        if (!cuda_context_retired) {
            state = rkva_slot_state::DONE_ERR;
            return; // Do not clear/recycle pinned storage after uncertain work.
        }
        state = rkva_slot_state::FREE;
        epoch = 0;
        ticket = 0;
        cohort_count = 0;
        cohort_index = 0;
        trace = {};
        pending_us = 0;
        response_cycle = 0;
        node = nullptr;
        output = nullptr;
        output_stride = 0;
        layer = 0;
        nq = 0;
        segments = 0;
        start_rows = 0;
        end_rows = 0;
        out_width = 0;
        upload_bytes = 0;
        first_pos = 0;
        last_pos = 0;
        bulk_f16 = false;
        cuda_small = false;
        cuda_shared_requested = false;
        cuda_small_api = nullptr;
        wire = RKVA_WIRE_F16;
        payload.clear();
        reply.clear();
        segment_scratch.clear();
        pos_scratch.clear();
        f32_scratch.clear();
        original_k.clear();
        original_v.clear();
        decode_f16.clear();
        query_fp8.clear();
    }
};

// ---------------------------------------------------------------------------
// backend context
// ---------------------------------------------------------------------------

struct rkva_shared_cuda_context_entry {
    const ggml_remote_attn_cuda_fp8_interface_v2 * api = nullptr;
    std::weak_ptr<void> context;
};

struct ggml_backend_remote_attn_context {
    std::string host;
    uint16_t    port = 0;

    std::mutex  mutex;                 // serializes non-async calls and ACK commits
    rkva_socket sock;
    std::atomic<bool> connected {false};
    std::atomic<bool> failed {false};
    // Logical poison survives until every in-flight job is drained; only then
    // are owners/vectors destroyed. Never destroy state under a job that may
    // still reference range_layers/native_layers.
    bool        remote_state_poisoned = false;
    // Holds the descriptor for an entire frame. Never wait for this while
    // holding async_mutex; the worker releases it before publishing its ACK.
    std::mutex socket_mutex;
    uint64_t    cycle_id = 0;

    // 2-slot async queue managed by worker thread with single TCP owner.
    // Slots decouple host preparation, staging and socket IO with deterministic backpressure.
    static constexpr size_t RKVA_ASYNC_QUEUE_CAPACITY = 4;
    // Backpressure reads this after dropping mutex, while control can reconfigure.
    std::atomic<size_t> async_capacity {2};
    bool cohort_enabled = false;
    std::mutex              async_mutex;
    std::condition_variable async_cv;
    std::thread             async_worker;
    std::atomic<bool>        async_stop {false};
    uint64_t                async_ticket_counter = 0;
    rkva_async_slot         async_slots[RKVA_ASYNC_QUEUE_CAPACITY];
    rkva_async_slot         synchronous_slot;
    // Protected only by mutex. Weak entries do not prolong CUDA resources;
    // slots own every live lease until terminal retirement and teardown.
    std::unordered_map<int, rkva_shared_cuda_context_entry> shared_cuda_contexts;
    rkva_byte_buffer        cohort_payload;
    rkva_byte_buffer        cohort_reply;
    bool                    async_local_remote = false;
    std::mutex              stats_mutex;

    ggml_remote_attn_geometry geo {};
    bool        geo_set = false;
    rkva_hello_ack ack {};

    uint32_t    active_session = RKVA_SESSION_NONE;
    bool        range_configured = false;
    ggml_type   range_source_type = GGML_TYPE_COUNT;
    rkva_wire   range_wire = RKVA_WIRE_F16;
    uint32_t    range_origin = 0;
    uint32_t    range_capacity = 0;
    // WAIT observes the epoch under async_mutex; invalidation increments it
    // under mutex before draining. The two lock domains share this atomic.
    std::atomic<uint64_t> range_epoch {1};
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
    bool native_q4 = false;            // layout 4: Q4_0 rows in the native bulk pipeline
    bool native_q8 = false;            // layout 5: Q8_0 rows in the native bulk pipeline
    bool native_serial_prefix = false;
    bool native_bulk = false;
    uint64_t native_generation = 0;
    uint32_t native_tail_tokens = 128;
    ggml_backend_cpu_kvarn_store_direct_v1_t native_store = nullptr;
    std::vector<std::unique_ptr<rkva_native_owner>> native_layers;

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

    ggml_backend_remote_attn_context() {
        for (size_t i = 0; i < RKVA_ASYNC_QUEUE_CAPACITY; ++i) {
            async_slots[i].id = uint32_t(i);
            if (i >= async_capacity) continue;
            try {
                async_slots[i].payload.reserve(8u << 20);
                async_slots[i].reply.reserve(4u << 20);
                async_slots[i].segment_scratch.reserve(1u << 20);
                async_slots[i].pos_scratch.reserve(256);
                async_slots[i].f32_scratch.reserve(256 * 256);
            } catch (...) {
                // Ignore reservation failure at init; will grow on demand if needed
            }
        }
    }

    void close_socket() {
        std::lock_guard<std::mutex> lock(socket_mutex);
        sock.close();
    }
};

// Caller owns ctx->mutex and this retired slot: PREPARING for async, or the
// synchronous slot. Explicit FREE slot release also requires async_mutex.
// Active other slots remain untouched; pinned network storage stays with slots.
static bool rkva_release_cuda_phase_context(ggml_backend_remote_attn_context * ctx,
        rkva_async_slot & slot, const ggml_tensor * node, bool free_idle_slot = false) {
    if (!slot.cuda_context_retired) return false;
    if (!slot.cuda_context) return true;
    if (ctx->failed || ctx->async_stop || !slot.cuda_api || slot.cuda_device < 0 ||
            (&slot != &ctx->synchronous_slot && slot.state != rkva_slot_state::PREPARING &&
             !(free_idle_slot && slot.state == rkva_slot_state::FREE))) return false;
    ggml_remote_attn_trace_scope timing("cuda_context_phase_release", node);
    timing.event.flags = 0x40000000u;
    timing.event.value = 1;
    // Successful pack/unpack already retired this slot's DMA. Shared peers
    // keep their own lease: dropping this owner is not a physical free claim.
    const bool shared = bool(slot.cuda_shared_lease);
    const bool last = shared && slot.cuda_shared_lease.use_count() == 1;
    slot.release_cuda_owner();
    {
        std::lock_guard<std::mutex> slock(ctx->stats_mutex);
        ++ctx->stats.cuda_context_phase_releases;
        ctx->stats.cuda_context_shared_releases += shared;
        ctx->stats.cuda_context_shared_last_releases += last;
    }
    if (shared) {
        auto event = ggml_remote_attn_trace_identity("cuda_context_lease_release", node);
        event.value = last ? 2 : 1; // last owner vs ordinary lease drop, never bytes freed
        event.begin_us = event.end_us = ggml_time_us();
        ggml_remote_attn_trace_record(&event);
    }
    return true;
}

// Caller owns ctx->mutex and this PREPARING/synchronous slot. Shared v2
// helpers serialize their complete synchronous operation in the opaque mutex.
// Pack/unpack scratch and status never outlive a call; wire buffers remain
// private to the slot. Backend failure/reconnect barriers are unchanged.
static bool rkva_prepare_cuda_context(ggml_backend_remote_attn_context * ctx,
        rkva_async_slot & slot, const ggml_tensor * node, int device,
        const ggml_remote_attn_cuda_fp8_interface_v2 * api, bool & created) {
    created = false;
    if (!slot.cuda_context_retired || !api || (slot.cuda_api && slot.cuda_api != api)) return false;
    const bool shared = slot.cuda_shared_requested;
    if (slot.cuda_context && (slot.cuda_device != device || slot.cuda_api != api ||
            bool(slot.cuda_shared_lease) != shared)) {
        if (!rkva_release_cuda_phase_context(ctx, slot, node)) return false;
    }
    if (!shared) {
        const bool fresh = slot.cuda_context == nullptr;
        if (!slot.prepare_cuda(device) || slot.cuda_api != api) return false;
        created = fresh;
        return true;
    }
    if (slot.cuda_context) return true;
    auto & entry = ctx->shared_cuda_contexts[device];
    auto lease = entry.context.lock();
    if (lease && entry.api != api) return false;
    if (!lease) {
        void * context = api->create_context(device);
        if (!context) return false;
        // shared_ptr invokes this deleter on construction failure too. Capturing
        // only api avoids touching an already-destroyed backend/stats mutex.
        lease = std::shared_ptr<void>(context, [api](void * owned) { api->destroy_context(owned); });
        entry.api = api;
        entry.context = lease;
        created = true;
    }
    slot.cuda_api = api;
    slot.cuda_device = device;
    slot.cuda_shared_lease = std::move(lease);
    slot.cuda_context = slot.cuda_shared_lease.get();
    slot.payload.use_pinned(api);
    slot.reply.use_pinned(api);
    slot.query_fp8.use_pinned(api);
    {
        std::lock_guard<std::mutex> slock(ctx->stats_mutex);
        ++ctx->stats.cuda_context_shared_acquires;
    }
    auto event = ggml_remote_attn_trace_identity("cuda_context_lease_acquire", node);
    event.value = 1;
    event.begin_us = event.end_us = ggml_time_us();
    ggml_remote_attn_trace_record(&event);
    return true;
}

// A tiled prefill may use every context, whereas decode reuses just one slot.
// Caller owns mutex; async_mutex excludes new owners of other FREE contexts.
static bool rkva_release_idle_cuda_peers(ggml_backend_remote_attn_context * ctx,
        rkva_async_slot & current, const ggml_tensor * node) {
    std::lock_guard<std::mutex> alock(ctx->async_mutex);
    for (auto & slot : ctx->async_slots) {
        if (&slot != &current && slot.state == rkva_slot_state::FREE &&
                !rkva_release_cuda_phase_context(ctx, slot, node, true)) return false;
    }
    return true;
}

static bool rkva_release_cuda_cpu_phase(ggml_backend_remote_attn_context * ctx,
        rkva_async_slot & current, const ggml_tensor * node) {
    return rkva_release_cuda_phase_context(ctx, current, node) &&
           rkva_release_idle_cuda_peers(ctx, current, node);
}

static uint64_t now_us() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Transport circuit breaker for the optional Xbox RKVA backend.
//
// A llama context owns one remote-attn backend and is rebuilt per request, so
// the breaker must be process-wide. It is keyed by endpoint (host:port) so a
// reconfigured server starts closed. Only transport failures (connect, HELLO,
// session and range RPCs) arm the cooldown; validation and capability
// rejections keep returning false exactly as before. When the cooldown expires
// the next request probes the endpoint normally: fixed interval, no backoff.
// ---------------------------------------------------------------------------
namespace {
struct rkva_breaker_state {
    std::mutex mutex;
    std::string endpoint;
    int64_t disabled_until_us = 0; // 0 = closed (split enabled)
    int64_t last_log_us = 0;
    uint64_t failures = 0;
    uint64_t fast_rejects = 0;
};

rkva_breaker_state g_rkva_breaker;

// Caller holds g_rkva_breaker.mutex. A different endpoint starts from a clean
// closed state so the counters and cooldown never leak across servers.
void rkva_breaker_bind_endpoint_locked(const ggml_backend_remote_attn_context * ctx) {
    std::string endpoint = ctx->host;
    endpoint += ':';
    endpoint += std::to_string((unsigned) ctx->port);
    if (endpoint == g_rkva_breaker.endpoint) {
        return;
    }
    g_rkva_breaker.endpoint = endpoint;
    g_rkva_breaker.disabled_until_us = 0;
    g_rkva_breaker.last_log_us = 0;
    g_rkva_breaker.failures = 0;
    g_rkva_breaker.fast_rejects = 0;
}

int64_t rkva_breaker_cooldown_us() {
    int64_t seconds = 30;
    if (const char * env = std::getenv("GGML_RKVA_RETRY_COOLDOWN_S")) {
        char * end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end != env && *end == '\0') {
            seconds = parsed;
        } else {
            GGML_LOG_WARN("%s: invalid GGML_RKVA_RETRY_COOLDOWN_S='%s' (expected seconds); using 30\n",
                    GGML_REMOTE_ATTN_NAME, env);
        }
    }
    return std::clamp<int64_t>(seconds, 5, 300) * 1000000;
}

// False while the cooldown is active: the caller must return without opening a
// socket or waiting. The diagnostic is throttled to one line per second.
bool rkva_breaker_allows_attempt(const ggml_backend_remote_attn_context * ctx) {
    std::lock_guard<std::mutex> lock(g_rkva_breaker.mutex);
    rkva_breaker_bind_endpoint_locked(ctx);
    const int64_t now = (int64_t) now_us();
    if (g_rkva_breaker.disabled_until_us == 0 || now >= g_rkva_breaker.disabled_until_us) {
        return true;
    }
    ++g_rkva_breaker.fast_rejects;
    if (now - g_rkva_breaker.last_log_us >= 1000000) {
        g_rkva_breaker.last_log_us = now;
        const int64_t remaining_s = (g_rkva_breaker.disabled_until_us - now + 999999) / 1000000;
        GGML_LOG_WARN("%s: Xbox backend unavailable; tier above the split disabled for %llds "
                "(fast fail, failures=%llu, rejects=%llu)\n",
                GGML_REMOTE_ATTN_NAME, (long long) remaining_s,
                (unsigned long long) g_rkva_breaker.failures,
                (unsigned long long) g_rkva_breaker.fast_rejects);
    }
    return false;
}

// Transport-level failure: arm (or re-arm) the fixed cooldown. Re-arming while
// it is already open stays silent; each probe after expiry may report once.
void rkva_breaker_mark_down(const ggml_backend_remote_attn_context * ctx) {
    const int64_t cooldown_us = rkva_breaker_cooldown_us();
    std::lock_guard<std::mutex> lock(g_rkva_breaker.mutex);
    rkva_breaker_bind_endpoint_locked(ctx);
    const int64_t now = (int64_t) now_us();
    const bool was_closed = g_rkva_breaker.disabled_until_us <= now;
    g_rkva_breaker.disabled_until_us = now + cooldown_us;
    ++g_rkva_breaker.failures;
    if (!was_closed) return;
    g_rkva_breaker.last_log_us = now;
    GGML_LOG_ERROR("%s: Xbox backend marked down; retrying automatically in %llds\n",
            GGML_REMOTE_ATTN_NAME, (long long) (cooldown_us / 1000000));
}

// A successful split configuration proves the endpoint usable again. The
// recovery line is emitted only on the open -> closed transition.
void rkva_breaker_note_configured(const ggml_backend_remote_attn_context * ctx) {
    std::lock_guard<std::mutex> lock(g_rkva_breaker.mutex);
    rkva_breaker_bind_endpoint_locked(ctx);
    if (g_rkva_breaker.disabled_until_us == 0) return;
    g_rkva_breaker.disabled_until_us = 0;
    g_rkva_breaker.last_log_us = 0;
    GGML_LOG_WARN("%s: Xbox backend recovered; split enabled\n", GGML_REMOTE_ATTN_NAME);
}
} // namespace

// low-level request/response over the persistent socket
template<class ReplyBuffer>
static bool rkva_transact(
        ggml_backend_remote_attn_context * ctx,
        rkva_op op, uint32_t session_id, uint32_t layer_id, uint32_t n_tokens,
        const void * payload, uint32_t payload_bytes,
        ReplyBuffer & out, uint32_t expected_bytes, int32_t * server_status = nullptr,
        uint64_t * out_cycle = nullptr) {

    std::lock_guard<std::mutex> socket_lock(ctx->socket_mutex);

    // A partial frame cannot be retried on the same ordered byte stream. Valid
    // server errors with an empty body remain recoverable (snapshot fallback).
    // A live stream failure also means the Xbox endpoint is unreachable, so it
    // arms the process-wide circuit breaker; an already-failed context only
    // repeats the earlier verdict and must not extend the cooldown.
    const auto stream_failure = [&](bool transport = true) {
        ctx->failed = true;
        ctx->connected = false;
        ctx->sock.close();
        out.clear();
        if (transport) {
            rkva_breaker_mark_down(ctx);
        }
        return false;
    };
    if (ctx->failed || !ctx->sock.open()) {
        return stream_failure(false);
    }
    const rkva_request req {
        RKVA_MAGIC, RKVA_VERSION, (uint16_t) op, ++ctx->cycle_id,
        session_id, layer_id, n_tokens, payload_bytes
    };
    if (out_cycle) *out_cycle = req.cycle_id;
    auto trace_context = ggml_remote_attn_trace_identity("rpc", nullptr);
    trace_context.cycle = req.cycle_id;
    trace_context.session = session_id;
    trace_context.layer = layer_id;
    trace_context.nq = n_tokens;
    ggml_remote_attn_trace_context_scope trace_owner(trace_context);
    ggml_remote_attn_trace_scope trace_total("rpc_total");
    trace_total.event.flags = 0x40000000u;

    const uint64_t t0 = now_us();
    {
    ggml_remote_attn_trace_scope trace_send("rpc_send");
    trace_send.event.bytes = sizeof(req) + payload_bytes;
    if (payload_bytes) {
#ifndef _WIN32
        struct iovec iov[2];
        iov[0].iov_base = const_cast<rkva_request *>(&req);
        iov[0].iov_len  = sizeof(req);
        iov[1].iov_base = const_cast<void *>(payload);
        iov[1].iov_len  = payload_bytes;
        if (!ctx->sock.write_all_iov(iov, 2)) {
            return stream_failure();
        }
#else
        if (!ctx->sock.write_all(&req, sizeof(req)) || !ctx->sock.write_all(payload, payload_bytes)) {
            return stream_failure();
        }
#endif
    } else {
        if (!ctx->sock.write_all(&req, sizeof(req))) {
            return stream_failure();
        }
    }
    }

    rkva_response rsp {};
    {
    ggml_remote_attn_trace_scope trace_header("rpc_header_wait");
    if (!ctx->sock.read_all(&rsp, sizeof(rsp))) {
        return stream_failure();
    }
    trace_header.event.bytes = sizeof(rsp);
    trace_header.event.value = rsp.xbox_prepare_us;
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
    {
    ggml_remote_attn_trace_scope trace_resize("rpc_rx_resize");
    trace_resize.event.bytes = rsp.payload_bytes;
    trace_resize.event.value = rsp.payload_bytes > out.size() ? rsp.payload_bytes - out.size() : 0;
    out.resize(rsp.payload_bytes);
    }
    {
    ggml_remote_attn_trace_scope trace_receive("rpc_body_recv");
    trace_receive.event.bytes = rsp.payload_bytes;
    if (rsp.payload_bytes && !ctx->sock.read_all(out.data(), rsp.payload_bytes)) {
        return stream_failure();
    }
    }
    trace_total.event.value = rsp.xbox_prepare_us;
    trace_total.event.gpu_us = rsp.xbox_compute_us;
    const uint64_t t1 = now_us();

    {
        std::lock_guard<std::mutex> slock(ctx->stats_mutex);
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
    }
    return true;
}

// ---------------------------------------------------------------------------
// bounded async overlap (native KVarN4 bulk range only)
// ---------------------------------------------------------------------------

static bool rkva_has_free_slot(const ggml_backend_remote_attn_context * ctx) {
    for (const auto & s : ctx->async_slots) {
        if (s.id < ctx->async_capacity && s.state == rkva_slot_state::FREE) return true;
    }
    return false;
}

// Test-only deterministic barrier for the window between a successful
// transaction and the epoch/publication checks. The worker signals "reached"
// and waits (bounded) for an explicit release; production never enables it.
struct rkva_test_handshake_state {
    std::mutex mutex;
    std::condition_variable cv;
    bool enabled = false;
    bool reached = false;
    bool released = false;
};
rkva_test_handshake_state g_rkva_test_handshake;

static void rkva_test_handshake_barrier() {
    std::unique_lock<std::mutex> lock(g_rkva_test_handshake.mutex);
    if (!g_rkva_test_handshake.enabled) return;
    g_rkva_test_handshake.reached = true;
    g_rkva_test_handshake.cv.notify_all();
    g_rkva_test_handshake.cv.wait_for(lock, std::chrono::seconds(10), [] {
        return !g_rkva_test_handshake.enabled || g_rkva_test_handshake.released;
    });
}

// The worker thread has sole ownership of ctx->sock. Network transmit/receive
// and reply decoding run without ctx->mutex to prevent host CPU stall.
// ctx->mutex is only acquired briefly to commit generation and ACK coverage.
static bool rkva_finish_native_bulk_job(
        ggml_backend_remote_attn_context * ctx, rkva_async_slot & slot, bool ctx_locked = false,
        bool received = false) {
    ggml_remote_attn_trace_context_scope trace_owner(slot.trace);
    ggml_remote_attn_trace_scope trace_finish("job_finish");
    trace_finish.event.flags = 0x40000000u;
    if (slot.layer >= ctx->range_layers.size() ||
            slot.layer >= ctx->native_layers.size() || !slot.output) {
        return false;
    }
    auto & cached = ctx->range_layers[slot.layer];
    auto * owner = ctx->native_layers[slot.layer].get();
    if (!owner) {
        return false;
    }
    const rkva_wire wire = slot.wire;
    const size_t out_elem_bytes = (wire == RKVA_WIRE_FP8) ? sizeof(uint8_t) :
                                  (wire == RKVA_WIRE_F16 || wire == RKVA_WIRE_SPLITZIP) ? sizeof(uint16_t) :
                                  sizeof(float);
    const size_t out_bytes = size_t(slot.nq) * slot.out_width * out_elem_bytes;
    const size_t lse_bytes = size_t(slot.nq) * ctx->geo.n_head * sizeof(float);
    const size_t row_bytes = (size_t(slot.out_width) + ctx->geo.n_head) * sizeof(float);
    if (slot.nq == 0 || slot.out_width == 0 || slot.output_stride < row_bytes ||
            slot.payload.size() > (256u << 20) ||
            out_bytes + lse_bytes > (256u << 20)) {
        return false;
    }
    auto & reply = slot.reply;
    if (reply.capacity() < (4u << 20)) {
        reply.reserve(4u << 20);
    }
    uint64_t cycle = slot.response_cycle;
    const bool transacted = received ? reply.size() == out_bytes + lse_bytes :
        rkva_transact(ctx, RKVA_KVARN_RANGE_BULK_ATTN, slot.session,
                slot.layer, slot.nq, slot.payload.data(), uint32_t(slot.payload.size()),
                reply, uint32_t(out_bytes + lse_bytes), nullptr, &cycle);
    slot.response_cycle = cycle;
    if (!transacted) {
        return false;
    }
    // Test-only deterministic barrier; no-op unless explicitly enabled.
    rkva_test_handshake_barrier();
    // Validate before unpack so an invalidated job cannot publish. A poison
    // after this check is contained: buffers stay alive until retirement and
    // WAIT/commit reject the result (no publication of invalidated output).
    if (slot.epoch != ctx->range_epoch || ctx->failed) {
        return false;
    }
    auto response_trace = slot.trace;
    response_trace.cycle = cycle;
    if (ggml_remote_attn_trace_enabled()) ggml_remote_attn_trace_context(&response_trace);
    trace_finish.event.cycle = cycle;
    const bool cuda_output = wire == RKVA_WIRE_FP8 && slot.cuda_api &&
        slot.cuda_api->device_for_pointer(slot.output) >= 0;
    {
    ggml_remote_attn_trace_scope trace_unpack("out_decode");
    trace_unpack.event.bytes = reply.size();
    if (cuda_output) {
        if (!slot.cuda_context_retired || !slot.cuda_context) return false;
        slot.cuda_context_retired = false;
        const bool unpacked = slot.cuda_small ? slot.cuda_small_api->unpack_small(slot.cuda_context, reply.data(),
                reinterpret_cast<const float *>(reply.data() + out_bytes),
                slot.nq, slot.out_width, ctx->geo.n_head, slot.output, slot.output_stride) :
            slot.cuda_api->unpack_out(slot.cuda_context, reply.data(),
                reinterpret_cast<const float *>(reply.data() + out_bytes),
                slot.nq, slot.out_width, ctx->geo.n_head, slot.output, slot.output_stride);
        slot.cuda_context_retired = unpacked;
        if (!unpacked) return false;
    } else if (wire == RKVA_WIRE_FP8) {
        const auto * src_fp8 = reinterpret_cast<const uint8_t *>(reply.data());
        const auto * src_lse = reinterpret_cast<const float *>(reply.data() + out_bytes);
        for (size_t row = 0; row < slot.nq; ++row) {
            auto * dst = static_cast<float *>(static_cast<void *>(slot.output + row * slot.output_stride));
            if (!rkva_fp8_to_f32_batch(src_fp8 + row * slot.out_width, dst, slot.out_width)) {
                return false;
            }
            for (size_t h = 0; h < ctx->geo.n_head; ++h) {
                const float lse_val = src_lse[row * ctx->geo.n_head + h];
                if (!std::isfinite(lse_val)) return false;
                dst[slot.out_width + h] = lse_val;
            }
        }
    } else if (wire == RKVA_WIRE_SPLITZIP) {
        slot.decode_f16.resize(slot.nq * slot.out_width);
        if (!rkva_splitzip_decode(reply.data(), out_bytes, slot.decode_f16.data(), slot.decode_f16.size())) {
            std::memcpy(slot.decode_f16.data(), reply.data(), out_bytes);
        }
        const auto * src_lse = reinterpret_cast<const float *>(reply.data() + out_bytes);
        for (size_t row = 0; row < slot.nq; ++row) {
            auto * dst = static_cast<float *>(static_cast<void *>(slot.output + row * slot.output_stride));
            if (!rkva_f16_to_f32_batch(slot.decode_f16.data() + row * slot.out_width, dst, slot.out_width)) {
                return false;
            }
            for (size_t h = 0; h < ctx->geo.n_head; ++h) {
                const float lse_val = src_lse[row * ctx->geo.n_head + h];
                if (!std::isfinite(lse_val)) return false;
                dst[slot.out_width + h] = lse_val;
            }
        }
    } else if (wire == RKVA_WIRE_F16) {
        const auto * src_f16 = reinterpret_cast<const uint16_t *>(reply.data());
        const auto * src_lse = reinterpret_cast<const float *>(reply.data() + out_bytes);
        for (size_t row = 0; row < slot.nq; ++row) {
            auto * dst = static_cast<float *>(static_cast<void *>(slot.output + row * slot.output_stride));
            if (!rkva_f16_to_f32_batch(src_f16 + row * slot.out_width, dst, slot.out_width)) {
                return false;
            }
            for (size_t h = 0; h < ctx->geo.n_head; ++h) {
                const float lse_val = src_lse[row * ctx->geo.n_head + h];
                if (!std::isfinite(lse_val)) return false;
                dst[slot.out_width + h] = lse_val;
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
        for (size_t row = 0; row < slot.nq; ++row) {
            auto * dst = slot.output + row * slot.output_stride;
            std::memcpy(dst, reply.data() + row * slot.out_width * sizeof(float),
                    slot.out_width * sizeof(float));
            std::memcpy(dst + slot.out_width * sizeof(float),
                    reply.data() + out_bytes + row * ctx->geo.n_head * sizeof(float),
                    ctx->geo.n_head * sizeof(float));
        }
    }
    }

    // Publish host coverage only after the single ordered server ACK.
    ggml_remote_attn_trace_scope trace_commit("ack_commit");
    std::unique_lock<std::mutex> ctx_lock(ctx->mutex, std::defer_lock);
    if (!ctx_locked) ctx_lock.lock();
    if (slot.epoch != ctx->range_epoch) {
        return false;
    }
    if (ctx->native_generation > UINT64_MAX - slot.segments) {
        return false;
    }
    ctx->native_generation += slot.segments;
    cached.epoch = ctx->range_epoch;
    cached.rows = slot.end_rows;
    {
        std::lock_guard<std::mutex> slock(ctx->stats_mutex);
        ctx->stats.range_upload_calls += slot.segments;
        ctx->stats.range_upload_rows += slot.nq;
        ctx->stats.range_upload_bytes += slot.upload_bytes;
        ctx->stats.range_attn_calls += slot.segments;
        if (cuda_output) {
            ++ctx->stats.cuda_unpack_calls;
            ctx->stats.cuda_small_unpack_calls += slot.cuda_small;
            ctx->stats.out_f32_upload_bytes_avoided += uint64_t(slot.nq) * slot.out_width * sizeof(float);
        }
        if (ctx->stats.range_query_min < 0 || slot.first_pos < ctx->stats.range_query_min) {
            ctx->stats.range_query_min = slot.first_pos;
        }
        ctx->stats.range_query_max = std::max<int64_t>(ctx->stats.range_query_max, slot.last_pos);
    }
    return true;
}

// Caller holds ctx->mutex. LOGICAL invalidation only: safe even while another
// async job is still running and may reference range_layers/native_layers.
// Physical teardown is deferred to rkva_destroy_poisoned_state_locked() after a
// drain. The socket is interrupted (non-blocking cancel), not waited on.
static void rkva_poison_remote_state_locked(ggml_backend_remote_attn_context * ctx) {
    ++ctx->range_epoch;
    ctx->remote_state_poisoned = true;
    ctx->failed = true;
    ctx->connected = false;
    ctx->range_configured = false;
    ctx->range_native_kvarn = false;
    ctx->range_fused = false;
    ctx->range_profile = false;
    ctx->range_profile_controlled = false;
    ctx->range_kernel_mode = 0;
    ctx->active_session = RKVA_SESSION_NONE;
    if (ctx->native_generation != UINT64_MAX) {
        ++ctx->native_generation;
    } else {
        ctx->failed = true;
    }
    ctx->sock.interrupt();
}

// Caller holds ctx->mutex AND guarantees no async job is in flight
// (rkva_lock_ctx_drained or rkva_async_drain returned). Only here may owners,
// vectors and the socket be destroyed. Idempotent.
static void rkva_destroy_poisoned_state_locked(ggml_backend_remote_attn_context * ctx) {
    if (!ctx->remote_state_poisoned) return;
    ctx->close_socket();
    for (auto & owner : ctx->native_layers) {
        owner.reset();
    }
    ctx->native_layers.clear();
    ctx->range_layers.clear();
    ctx->native_store = nullptr;
    ctx->native_q4 = false;
    ctx->native_q8 = false;
    ctx->native_serial_prefix = false;
    ctx->native_bulk = false;
    ctx->async_local_remote = false;
    ctx->remote_state_poisoned = false;
}

// async_mutex is held. A partial cohort never runs, and later tickets cannot
// bypass it. Tensor sources have already been copied by each PREPARE node.
static size_t rkva_ready_group(ggml_backend_remote_attn_context * ctx, rkva_async_slot ** group) {
    rkva_async_slot * first = nullptr;
    for (auto & s : ctx->async_slots) {
        if ((s.state == rkva_slot_state::PENDING || s.state == rkva_slot_state::PREPARING) &&
                (!first || s.ticket < first->ticket)) first = &s;
    }
    if (!first || first->state != rkva_slot_state::PENDING) return 0;
    if (!first->cohort_count) { group[0] = first; return 1; }
    if (first->cohort_index || first->cohort_count > 4) return 0;
    for (uint32_t i = 0; i < first->cohort_count; ++i) {
        group[i] = nullptr;
        for (auto & s : ctx->async_slots) {
            if (s.state == rkva_slot_state::PENDING && s.ticket == first->ticket + i &&
                    s.epoch == first->epoch && s.cohort_count == first->cohort_count && s.cohort_index == i)
                group[i] = &s;
        }
        if (!group[i]) return 0;
    }
    return first->cohort_count;
}

static bool rkva_finish_cohort(ggml_backend_remote_attn_context * ctx,
        rkva_async_slot ** group, size_t count) {
    if (!count || count > 4 || !ctx->cohort_enabled) return false;
    ggml_remote_attn_trace_context_scope trace_owner(group[0]->trace);
    ggml_remote_attn_trace_scope trace_cohort("cohort_total");
    trace_cohort.event.flags = 0x40000000u;
    trace_cohort.event.value = count;
    size_t bytes = sizeof(rkva_kvarn_cohort_header), expected = bytes;
    uint32_t nq = 0, segments = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto & s = *group[i];
        if (bytes > (256u << 20) - sizeof(rkva_kvarn_cohort_entry) ||
                s.wire != RKVA_WIRE_FP8 || s.nq > 256 - nq || s.segments > 32 - segments ||
                s.payload.size() > (256u << 20) - bytes - sizeof(rkva_kvarn_cohort_entry)) return false;
        bytes += sizeof(rkva_kvarn_cohort_entry) + s.payload.size();
        expected += sizeof(rkva_kvarn_cohort_entry) + size_t(s.nq) * (s.out_width + ctx->geo.n_head * sizeof(float));
        nq += s.nq; segments += s.segments;
    }
    if (expected > (256u << 20)) return false;
    auto & tx = ctx->cohort_payload;
    auto & rx = ctx->cohort_reply;
    tx.resize(bytes);
    const rkva_kvarn_cohort_header header {1, uint32_t(count), nq, 0};
    std::memcpy(tx.data(), &header, sizeof(header));
    size_t offset = sizeof(header);
    for (size_t i = 0; i < count; ++i) {
        const auto & s = *group[i];
        const rkva_kvarn_cohort_entry entry {s.layer, s.nq, uint32_t(s.payload.size()), 0, s.ticket, s.epoch};
        std::memcpy(tx.data() + offset, &entry, sizeof(entry)); offset += sizeof(entry);
        std::memcpy(tx.data() + offset, s.payload.data(), s.payload.size()); offset += s.payload.size();
    }
    uint64_t cycle = 0;
    if (!rkva_transact(ctx, RKVA_KVARN_RANGE_COHORT_ATTN, ctx->active_session, 0, nq,
            tx.data(), uint32_t(bytes), rx, uint32_t(expected), nullptr, &cycle)) return false;
    trace_cohort.event.cycle = cycle;
    trace_cohort.event.bytes = bytes + expected;
    rkva_kvarn_cohort_header reply_header{};
    std::memcpy(&reply_header, rx.data(), sizeof(reply_header));
    if (std::memcmp(&reply_header, &header, sizeof(header))) return false;
    // Check every envelope/echo/length before publishing any OUT or ACK.
    size_t offsets[4]{};
    offset = sizeof(header);
    for (size_t i = 0; i < count; ++i) {
        auto & s = *group[i];
        if (rx.size() - offset < sizeof(rkva_kvarn_cohort_entry)) return false;
        rkva_kvarn_cohort_entry e{};
        std::memcpy(&e, rx.data() + offset, sizeof(e)); offset += sizeof(e);
        const size_t body = size_t(s.nq) * (s.out_width + ctx->geo.n_head * sizeof(float));
        if (e.layer != s.layer || e.n_queries != s.nq || e.body_bytes != body || e.reserved ||
                e.ticket != s.ticket || e.epoch != s.epoch || body > rx.size() - offset) return false;
        offsets[i] = offset; offset += body;
    }
    if (offset != rx.size()) return false;
    for (size_t i = 0; i < count; ++i) {
        const auto & s = *group[i];
        const size_t out_bytes = size_t(s.nq) * s.out_width;
        const uint8_t * body = rx.data() + offsets[i];
        for (size_t j = 0; j < out_bytes; ++j)
            if ((body[j] & 0x7cu) == 0x7cu) return false;
        for (size_t j = 0; j < size_t(s.nq) * ctx->geo.n_head; ++j) {
            float lse;
            std::memcpy(&lse, body + out_bytes + j * sizeof(float), sizeof(lse));
            if (!std::isfinite(lse)) return false;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        auto & s = *group[i];
        const size_t body = size_t(s.nq) * (s.out_width + ctx->geo.n_head * sizeof(float));
        s.reply.resize(body);
        std::memcpy(s.reply.data(), rx.data() + offsets[i], body);
        s.response_cycle = cycle;
        if (!rkva_finish_native_bulk_job(ctx, s, false, true)) return false;
    }
    std::lock_guard<std::mutex> slock(ctx->stats_mutex);
    ++ctx->stats.cohort_calls;
    ctx->stats.cohort_entries += count;
    return true;
}

static void rkva_async_worker_main(ggml_backend_remote_attn_context * ctx) {
    std::unique_lock<std::mutex> lock(ctx->async_mutex);
    for (;;) {
        ctx->async_cv.wait(lock, [ctx] {
            if (ctx->async_stop) return true;
            rkva_async_slot * ready[4]{};
            if (rkva_ready_group(ctx, ready)) return true;
            for (const auto & s : ctx->async_slots) {
                if (ctx->failed && s.state == rkva_slot_state::PENDING) return true;
            }
            return false;
        });
        if (ctx->async_stop) {
            return;
        }

        if (ctx->failed) {
            for (auto & s : ctx->async_slots) if (s.state == rkva_slot_state::PENDING) s.state = rkva_slot_state::DONE_ERR;
            ctx->async_cv.notify_all();
            continue;
        }
        rkva_async_slot * group[4]{};
        const size_t count = rkva_ready_group(ctx, group);
        if (!count) continue;
        auto * slot = group[0];
        for (size_t i = 0; i < count; ++i) group[i]->state = rkva_slot_state::RUNNING;
        if (ggml_remote_attn_trace_enabled()) for (size_t i = 0; i < count; ++i) {
            auto e = group[i]->trace;
            std::snprintf(e.phase, sizeof(e.phase), "rpc_queue");
            e.begin_us = group[i]->pending_us;
            e.end_us = ggml_time_us();
            ggml_remote_attn_trace_record(&e);
        }
        lock.unlock();

        bool ok = false;
        if (!ctx->failed && ctx->connected) {
            try {
                ok = slot->cohort_count ? rkva_finish_cohort(ctx, group, count) : rkva_finish_native_bulk_job(ctx, *slot);
            } catch (const std::exception & error) {
                GGML_LOG_ERROR("%s: async native KVarN4 RPC failed: %s\n",
                        GGML_REMOTE_ATTN_NAME, error.what());
            } catch (...) {
                // Fall through to fail-closed
            }
        }
        if (!ok) {
            // An uncertain ordered-stream/GPU mutation must fail closed. Poison
            // is logical only: this worker must not destroy owners/vectors that
            // another job may still reference; the next drain tears them down.
            std::lock_guard<std::mutex> ctx_lock(ctx->mutex);
            rkva_poison_remote_state_locked(ctx);
        }

        lock.lock();
        for (size_t i = 0; i < count; ++i) group[i]->state = ok ? rkva_slot_state::DONE_OK : rkva_slot_state::DONE_ERR;
        if (!ok) {
            // Error barrier: cancel any remaining pending or preparing slots
            for (auto & s : ctx->async_slots) {
                if (s.state == rkva_slot_state::PENDING ||
                    s.state == rkva_slot_state::PREPARING) {
                    s.state = rkva_slot_state::DONE_ERR;
                }
            }
        }
        ctx->async_cv.notify_all();
    }
}

// A terminal boundary with only part of a cohort cannot wait for future graph
// nodes. No RPC has started for that cohort. Poison its producer state and
// retire the pending members, without resetting a PREPARING/RUNNING owner.
static void rkva_cancel_partial_cohort_locked(ggml_backend_remote_attn_context * ctx) {
    bool partial = false;
    for (const auto & s : ctx->async_slots) {
        if (s.state == rkva_slot_state::PREPARING || s.state == rkva_slot_state::RUNNING) return;
        partial |= s.state == rkva_slot_state::PENDING && s.cohort_count;
    }
    rkva_async_slot * ready[4]{};
    if (!partial || rkva_ready_group(ctx, ready)) return;
    ctx->failed = true;
    ctx->connected = false;
    for (auto & s : ctx->async_slots)
        if (s.state == rkva_slot_state::PENDING) s.state = rkva_slot_state::DONE_ERR;
    ctx->async_cv.notify_all();
}

// Wait for any submitted jobs and discard completed-but-unconsumed results.
// The remote session is only mutated after this returns.
static void rkva_async_drain(ggml_backend_remote_attn_context * ctx) {
    std::unique_lock<std::mutex> lock(ctx->async_mutex);
    ctx->async_cv.wait(lock, [ctx] {
        rkva_cancel_partial_cohort_locked(ctx);
        for (const auto & s : ctx->async_slots) {
            if (s.state == rkva_slot_state::PENDING ||
                s.state == rkva_slot_state::RUNNING ||
                s.state == rkva_slot_state::PREPARING) {
                return false;
            }
        }
        return true;
    });
    for (auto & s : ctx->async_slots) {
        if (s.state == rkva_slot_state::DONE_OK ||
            s.state == rkva_slot_state::DONE_ERR) {
            s.reset();
        }
    }
    lock.unlock();
    ctx->async_cv.notify_all();
    if (ctx->failed) ctx->close_socket();
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
            for (auto & s : ctx->async_slots) {
                if (s.state == rkva_slot_state::PENDING ||
                    s.state == rkva_slot_state::RUNNING ||
                    s.state == rkva_slot_state::PREPARING) {
                    busy = true;
                }
                if (s.state == rkva_slot_state::DONE_OK ||
                    s.state == rkva_slot_state::DONE_ERR) {
                    done = true;
                    s.reset();
                }
            }
        }
        if (!busy) {
            if (done) {
                ctx->async_cv.notify_all();
            }
            rkva_destroy_poisoned_state_locked(ctx);
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
    ggml_tensor * source = node->src[0];
    const uint64_t ticket = (uint64_t(uint32_t(ggml_get_op_params_i32(source, GGML_REMOTE_ATTN_PARAM_TICKET_HI))) << 32) |
                             uint32_t(ggml_get_op_params_i32(source, GGML_REMOTE_ATTN_PARAM_TICKET_LO));
    const uint64_t epoch  = (uint64_t(uint32_t(ggml_get_op_params_i32(source, GGML_REMOTE_ATTN_PARAM_EPOCH_HI))) << 32) |
                             uint32_t(ggml_get_op_params_i32(source, GGML_REMOTE_ATTN_PARAM_EPOCH_LO));
    if (ticket == 0) {
        std::lock_guard<std::mutex> ctx_lock(ctx->mutex);
        rkva_poison_remote_state_locked(ctx);
        return false;
    }

    std::unique_lock<std::mutex> lock(ctx->async_mutex);

    rkva_async_slot * slot = nullptr;
    for (auto & s : ctx->async_slots) {
        if (s.node == source && s.ticket == ticket && s.epoch == epoch && s.state != rkva_slot_state::FREE) {
            slot = &s;
            break;
        }
    }
    if (!slot) {
        // A WAIT with no live ticket is a terminal inconsistency, not a benign
        // no-op: never let the decode fall back to the batch-retry path.
        lock.unlock();
        std::lock_guard<std::mutex> ctx_lock(ctx->mutex);
        rkva_poison_remote_state_locked(ctx);
        return false;
    }

    ggml_remote_attn_trace_context_scope trace_owner(slot->trace);
    ggml_remote_attn_trace_scope trace_wait("range_wait", node);

    ctx->async_cv.wait(lock, [&] {
        // Every member must have been prepared before any WAIT in the graph.
        // A wiring error must not wait forever for nodes this thread cannot run.
        if (slot->cohort_count) rkva_cancel_partial_cohort_locked(ctx);
        if (slot->ticket != ticket || slot->epoch != epoch || slot->node != source) {
            return true;
        }
        return slot->state == rkva_slot_state::DONE_OK ||
               slot->state == rkva_slot_state::DONE_ERR ||
               ctx->async_stop;
    });

    if (slot->ticket != ticket || slot->epoch != epoch || slot->node != source) {
        // The slot was retired/reused under us: terminal for this request.
        lock.unlock();
        std::lock_guard<std::mutex> ctx_lock(ctx->mutex);
        rkva_poison_remote_state_locked(ctx);
        return false;
    }

    trace_wait.event.cycle = slot->response_cycle;

    const bool ok = (slot->state == rkva_slot_state::DONE_OK) && !ctx->failed &&
                    (slot->epoch == ctx->range_epoch) &&
                    (node->data != nullptr) && (source->data != nullptr) &&
                    (source->type == node->type) &&
                    (source->ne[0] == node->ne[0]) && (source->ne[1] == node->ne[1]) &&
                    (source->ne[2] == node->ne[2]) && (source->ne[3] == node->ne[3]);

    if (ok) {
        if (node->data != source->data) {
            const size_t row_bytes = size_t(node->ne[0]) * sizeof(float);
            for (int64_t row = 0; row < node->ne[1]; ++row) {
                auto * api = rkva_cuda_fp8_api();
                if (api && (api->device_for_pointer(node->data) >= 0 || api->device_for_pointer(source->data) >= 0)) {
                    ggml_backend_tensor_copy(source, node);
                    break;
                }
                std::memcpy(static_cast<uint8_t *>(node->data) + row * node->nb[1],
                    static_cast<const uint8_t *>(source->data) + row * source->nb[1], row_bytes);
            }
        }
    }

    slot->reset();
    lock.unlock();
    ctx->async_cv.notify_all();

    if (!ok) {
        std::lock_guard<std::mutex> ctx_lock(ctx->mutex);
        rkva_poison_remote_state_locked(ctx);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// REMOTE_ATTN execution
// ---------------------------------------------------------------------------

static bool rkva_prepare_native_bulk_job(
        ggml_backend_remote_attn_context * ctx, ggml_tensor * node,
        uint32_t layer, uint32_t live, size_t nq, size_t out_width,
        rkva_async_slot & slot);

// One authoritative host owner per layer. Layouts 4/5 pack ordinary Q4_0/Q8_0
// rows; layouts 1..3 keep the KVarN record/stage/tail producer untouched.
static std::unique_ptr<rkva_native_owner> rkva_new_native_owner(
        ggml_backend_remote_attn_context * ctx) {
    if (ctx->native_q4) {
        return std::unique_ptr<rkva_native_owner>(new rkva_q4_producer(
                ctx->range_origin, ctx->range_capacity, ctx->geo.n_head_kv));
    }
    if (ctx->native_q8) {
        return std::unique_ptr<rkva_native_owner>(new rkva_q8_producer(
                ctx->range_origin, ctx->range_capacity, ctx->geo.n_head_kv));
    }
    return std::unique_ptr<rkva_native_owner>(new rkva_kvarn_producer(
            ctx->range_origin, ctx->range_capacity, ctx->geo.n_head_kv,
            ctx->geo.sinkhorn_iters, ctx->native_store, ctx->native_tail_tokens,
            (ctx->ack.reserved[0] & RKVA_CAP_KVARN_COMPACT_STAGE) && ctx->native_serial_prefix));
}

static bool rkva_run_native_range(ggml_backend_remote_attn_context * ctx, ggml_tensor * node,
                                  uint32_t layer, uint32_t live, size_t nq, size_t out_width) {
    // Includes the QCPU/outputCPU fallback, which does not prepare a slot.
    if (!ctx->synchronous_slot.cuda_context_retired) return false;
    auto * cuda_api = ctx->range_wire == RKVA_WIRE_FP8 && ctx->native_bulk ? rkva_cuda_fp8_api() : nullptr;
    if (cuda_api && !rkva_cuda_fp8_bucket_permitted(int64_t(nq)) &&
            cuda_api->device_for_pointer(node->data) >= 0) {
        // Placement must agree with the CPU codec policy. Never let CPU decode
        // write a raw VRAM address when a caller bypasses the scheduler.
        return false;
    }
    if (cuda_api && !rkva_cuda_fp8_bucket_permitted(int64_t(nq)) &&
            cuda_api->device_for_pointer(node->data) == -1 &&
            !rkva_release_cuda_phase_context(ctx, ctx->synchronous_slot, node)) return false;
    if (cuda_api && node->src[5] && node->src[6] &&
            (cuda_api->device_for_pointer(node->src[0]->data) != -1 || cuda_api->device_for_pointer(node->data) != -1)) {
        auto & slot = ctx->synchronous_slot;
        slot.reset();
        slot.epoch = ctx->range_epoch;
        slot.session = ctx->active_session;
        const bool ok = rkva_prepare_native_bulk_job(ctx, node, layer, live, nq, out_width, slot) &&
            rkva_finish_native_bulk_job(ctx, slot, true);
        slot.reset();
        return ok;
    }
    const auto * keys = node->src[5];
    const auto * values = node->src[6];
    const size_t width = size_t(ctx->geo.n_head_kv) * 256;
    if (!keys || !values || keys->type != GGML_TYPE_F32 || values->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(keys) || !ggml_is_contiguous(values) ||
            !ggml_are_same_shape(keys, values) || keys->ne[0] != 256 ||
            keys->ne[1] != ctx->geo.n_head_kv || keys->ne[2] != int64_t(nq) ||
            keys->ne[3] != 1 || nq > 256 || (!ctx->native_store && !ctx->native_q4 && !ctx->native_q8)) return false;
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
            owner = rkva_new_native_owner(ctx);
        }
        std::vector<float> original_k(nq * width), original_v(nq * width);
        ggml_backend_tensor_get(keys, original_k.data(), 0, original_k.size() * sizeof(float));
        ggml_backend_tensor_get(values, original_v.data(), 0, original_v.size() * sizeof(float));
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
        const size_t chunk_limit = (ctx->native_q4 || ctx->native_q8) ? (nq > 9 ? 8 : nq) :
            ctx->native_serial_prefix ? (nq > 9 ? 8 : nq) : 1;
        std::vector<uint8_t> bulk_payload;
        const uint32_t segments = uint32_t((nq + chunk_limit - 1) / chunk_limit);
        if (ctx->native_bulk) {
            if (ctx->native_generation > UINT64_MAX - segments) { ctx->failed = true; return false; }
            const rkva_kvarn_range_bulk header {1, segments, uint32_t(nq), 0};
            bulk_payload.resize(sizeof(header));
            std::memcpy(bulk_payload.data(), &header, sizeof(header));
            ctx->f32_scratch.resize(nq * out_width);
            ggml_backend_tensor_get(node->src[0], ctx->f32_scratch.data(), 0, ctx->f32_scratch.size() * sizeof(float));
        }
        const auto * bulk_queries = ctx->f32_scratch.data();
        uint64_t bulk_upload_bytes = 0;
        for (size_t first = 0; first < nq; first += chunk_limit) {
            const size_t chunk = std::min(chunk_limit, nq - first);
            if (!owner->append(original_k.data() + first * width, original_v.data() + first * width,
                    ctx->pos_scratch.data() + first, uint32_t(chunk), first == 0 && cached.rows == 0, ctx->payload_tx)) {
                ctx->failed = true; return false;
            }
            if (ctx->native_bulk) {
                const rkva_kvarn_range_bulk_segment segment {uint32_t(ctx->payload_tx.size()), uint32_t(chunk)};
                const rkva_wire wire = ctx->range_wire;
                const size_t q_elem_bytes = (wire == RKVA_WIRE_FP8) ? sizeof(uint8_t) :
                                            (wire == RKVA_WIRE_F16 || wire == RKVA_WIRE_SPLITZIP) ? sizeof(uint16_t) :
                                            sizeof(float);
                const rkva_range_attn attention {wire, 0, {0, 0}};
                const size_t offset = bulk_payload.size();
                const size_t query_bytes = chunk * (sizeof(int32_t) + out_width * q_elem_bytes);
                bulk_payload.resize(offset + sizeof(segment) + ctx->payload_tx.size() + sizeof(attention) + query_bytes);
                auto * dst = bulk_payload.data() + offset;
                std::memcpy(dst, &segment, sizeof(segment)); dst += sizeof(segment);
                std::memcpy(dst, ctx->payload_tx.data(), ctx->payload_tx.size()); dst += ctx->payload_tx.size();
                std::memcpy(dst, &attention, sizeof(attention)); dst += sizeof(attention);
                std::memcpy(dst, ctx->pos_scratch.data() + first, chunk * sizeof(int32_t)); dst += chunk * sizeof(int32_t);
                if (wire == RKVA_WIRE_FP8) {
                    uint8_t * dst_fp8 = dst;
                    if (!rkva_f32_to_fp8_batch(bulk_queries + first * out_width, dst_fp8, chunk * out_width)) {
                        ctx->failed = true; return false;
                    }
                } else if (wire == RKVA_WIRE_SPLITZIP) {
                    std::vector<uint16_t> tmp_f16(chunk * out_width);
                    if (!rkva_f32_to_f16_batch(bulk_queries + first * out_width, tmp_f16.data(), chunk * out_width)) {
                        ctx->failed = true; return false;
                    }
                    std::vector<uint8_t> sz_out;
                    rkva_splitzip_encode(tmp_f16.data(), tmp_f16.size(), sz_out);
                    if (sz_out.size() <= chunk * out_width * sizeof(uint16_t)) {
                        std::memcpy(dst, sz_out.data(), sz_out.size());
                    } else {
                        std::memcpy(dst, tmp_f16.data(), tmp_f16.size() * sizeof(uint16_t));
                    }
                } else if (wire == RKVA_WIRE_F16) {
                    uint16_t * dst_f16 = reinterpret_cast<uint16_t *>(dst);
                    if (!rkva_f32_to_f16_batch(bulk_queries + first * out_width, dst_f16, chunk * out_width)) {
                        ctx->failed = true; return false;
                    }
                } else {
                    std::memcpy(dst, bulk_queries + first * out_width, chunk * out_width * sizeof(float));
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
            const rkva_wire wire = ctx->range_wire;
            std::vector<uint8_t> reply;
            const size_t out_elem_bytes = (wire == RKVA_WIRE_FP8) ? sizeof(uint8_t) :
                                          (wire == RKVA_WIRE_F16 || wire == RKVA_WIRE_SPLITZIP) ? sizeof(uint16_t) :
                                          sizeof(float);
            const size_t out_bytes = nq * out_width * out_elem_bytes;
            const size_t lse_bytes = nq * ctx->geo.n_head * sizeof(float);
            if (bulk_payload.size() > (256u << 20) ||
                !rkva_transact(ctx, RKVA_KVARN_RANGE_BULK_ATTN, ctx->active_session, layer, uint32_t(nq),
                    bulk_payload.data(), uint32_t(bulk_payload.size()), reply, uint32_t(out_bytes + lse_bytes))) {
                ctx->failed = true; return false;
            }
            if (wire == RKVA_WIRE_FP8) {
                const auto * src_fp8 = reinterpret_cast<const uint8_t *>(reply.data());
                const auto * src_lse = reinterpret_cast<const float *>(reply.data() + out_bytes);
                for (size_t row = 0; row < nq; ++row) {
                    auto * dst = static_cast<float *>(static_cast<void *>(static_cast<uint8_t *>(node->data) + row * node->nb[1]));
                    if (!rkva_fp8_to_f32_batch(src_fp8 + row * out_width, dst, out_width)) {
                        ctx->failed = true; return false;
                    }
                    for (size_t h = 0; h < ctx->geo.n_head; ++h) {
                        const float lse_val = src_lse[row * ctx->geo.n_head + h];
                        if (!std::isfinite(lse_val)) { ctx->failed = true; return false; }
                        dst[out_width + h] = lse_val;
                    }
                }
            } else if (wire == RKVA_WIRE_SPLITZIP) {
                std::vector<uint16_t> tmp_f16(nq * out_width);
                if (!rkva_splitzip_decode(reply.data(), out_bytes, tmp_f16.data(), tmp_f16.size())) {
                    std::memcpy(tmp_f16.data(), reply.data(), out_bytes);
                }
                const auto * src_lse = reinterpret_cast<const float *>(reply.data() + out_bytes);
                for (size_t row = 0; row < nq; ++row) {
                    auto * dst = static_cast<float *>(static_cast<void *>(static_cast<uint8_t *>(node->data) + row * node->nb[1]));
                    if (!rkva_f16_to_f32_batch(tmp_f16.data() + row * out_width, dst, out_width)) {
                        ctx->failed = true; return false;
                    }
                    for (size_t h = 0; h < ctx->geo.n_head; ++h) {
                        const float lse_val = src_lse[row * ctx->geo.n_head + h];
                        if (!std::isfinite(lse_val)) { ctx->failed = true; return false; }
                        dst[out_width + h] = lse_val;
                    }
                }
            } else if (wire == RKVA_WIRE_F16) {
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
        rkva_async_slot & slot) {
    // Before readback, resize or envelope writes, even after reconnect and for
    // CPU-only jobs: an unsuccessful helper requires a fresh backend context.
    const auto cuda_failure = [&](const char * stage) {
        GGML_LOG_ERROR("%s: native CUDA preparation failed stage=%s layer=%u nq=%zu slot=%u retired=%d shared=%d small=%d\n",
                GGML_REMOTE_ATTN_NAME, stage, layer, nq, slot.id,
                int(slot.cuda_context_retired), int(bool(slot.cuda_shared_lease)), int(slot.cuda_small));
        return false;
    };
    if (!slot.cuda_context_retired) return cuda_failure("unretired_slot");
    if (ggml_remote_attn_trace_enabled()) {
        slot.trace = ggml_remote_attn_trace_identity("job", node);
        slot.trace.ticket = slot.ticket;
        slot.trace.epoch = slot.epoch;
        slot.trace.session = ctx->active_session;
        slot.trace.layer = layer;
        slot.trace.nq = uint32_t(nq);
        if (ctx->pos_scratch.size() == nq) {
            slot.trace.first_pos = ctx->pos_scratch.front();
            slot.trace.last_pos = ctx->pos_scratch.back();
        }
    }
    ggml_remote_attn_trace_context_scope trace_owner(slot.trace);
    ggml_remote_attn_trace_scope trace_prepare("job_prepare");
    trace_prepare.event.flags = 0x40000000u;
    const auto * keys = node->src[5];
    const auto * values = node->src[6];
    const size_t width = size_t(ctx->geo.n_head_kv) * 256;
    if (!ctx->native_bulk || !keys || !values || keys->type != GGML_TYPE_F32 ||
            values->type != GGML_TYPE_F32 || !ggml_is_contiguous(keys) ||
            !ggml_is_contiguous(values) || !ggml_are_same_shape(keys, values) ||
            keys->ne[0] != 256 || keys->ne[1] != ctx->geo.n_head_kv ||
            keys->ne[2] != int64_t(nq) || keys->ne[3] != 1 || nq == 0 || nq > 256 ||
            (!ctx->native_store && !ctx->native_q4 && !ctx->native_q8) ||
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
            owner = rkva_new_native_owner(ctx);
            cached.epoch = ctx->range_epoch;
        }
        slot.start_rows = owner->rows();
        {
        ggml_remote_attn_trace_scope trace_k("k_readback");
        trace_k.event.bytes = nq * width * sizeof(float);
        slot.original_k.resize(nq * width);
        ggml_backend_tensor_get(keys, slot.original_k.data(), 0, slot.original_k.size() * sizeof(float));
        }
        {
        ggml_remote_attn_trace_scope trace_v("v_readback");
        trace_v.event.bytes = nq * width * sizeof(float);
        slot.original_v.resize(nq * width);
        ggml_backend_tensor_get(values, slot.original_v.data(), 0, slot.original_v.size() * sizeof(float));
        }
        const float * dev_k = static_cast<const float *>(keys->data);
        const float * dev_v = static_cast<const float *>(values->data);
        {
        ggml_remote_attn_trace_scope trace_pos("positions_readback");
        trace_pos.event.bytes = nq * sizeof(int32_t);
        slot.pos_scratch.resize(nq);
        ggml_backend_tensor_get(node->src[4], slot.pos_scratch.data(), 0, nq * sizeof(int32_t));
        }
        slot.trace.first_pos = slot.pos_scratch.front();
        slot.trace.last_pos = slot.pos_scratch.back();
        if (ggml_remote_attn_trace_enabled()) ggml_remote_attn_trace_context(&slot.trace);
        {
        ggml_remote_attn_trace_scope trace_producer("kv_producer");
        if (!owner->prepare(slot.original_k.data(), slot.original_v.data(), slot.pos_scratch.data(),
                    uint32_t(nq), dev_k, dev_v)) {
            ctx->failed = true;
            return false;
        }
        }
        const size_t chunk_limit = (ctx->native_q4 || ctx->native_q8) ? (nq > 9 ? 8 : nq) :
            ctx->native_serial_prefix ? (nq > 9 ? 8 : nq) : 1;
        const uint32_t segments = uint32_t((nq + chunk_limit - 1) / chunk_limit);
        if (ctx->native_generation > UINT64_MAX - segments) {
            ctx->failed = true;
            return false;
        }
        const rkva_wire wire = ctx->range_wire;
        const size_t q_elem_bytes = (wire == RKVA_WIRE_FP8) ? sizeof(uint8_t) :
                                    (wire == RKVA_WIRE_F16 || wire == RKVA_WIRE_SPLITZIP) ? sizeof(uint16_t) :
                                    sizeof(float);
        const rkva_kvarn_range_bulk header {1, segments, uint32_t(nq), 0};
        slot.payload.resize(sizeof(header));
        std::memcpy(slot.payload.data(), &header, sizeof(header));
        const auto * api = wire == RKVA_WIRE_FP8 ? rkva_cuda_fp8_api() : nullptr;
        const int q_device = api ? api->device_for_pointer(node->src[0]->data) : -1;
        const int out_device = api ? api->device_for_pointer(node->data) : -1;
        // Managed pointers and CUDA attribute failures have no supported host
        // coherency contract here; never reinterpret them as ordinary CPU data.
        if (q_device < -1 || out_device < -1) return cuda_failure("pointer_device");
        const bool cuda_allowed = rkva_cuda_fp8_bucket_permitted(int64_t(nq));
        if (!cuda_allowed && out_device >= 0) return cuda_failure("bucket_device_output");
        if (!cuda_allowed && !rkva_release_cuda_cpu_phase(ctx, slot, node)) return cuda_failure("cpu_phase_release");
        const bool cuda_pack_q = cuda_allowed && q_device >= 0;
        slot.cuda_small = rkva_cuda_fp8_mapped_requested() && nq <= 9 &&
                          (cuda_pack_q || out_device >= 0);
        slot.cuda_small_api = slot.cuda_small ? rkva_cuda_fp8_small_api() : nullptr;
        // An explicit experimental selection is not a request to silently use
        // the old GPU helper if the extension/device cannot map host memory.
        if (slot.cuda_small && (!slot.cuda_small_api || slot.cuda_small_api->base != api ||
                !slot.cuda_small_api->supports_small(q_device >= 0 ? q_device : out_device))) return cuda_failure("small_support");
        if (slot.cuda_small && !rkva_release_idle_cuda_peers(ctx, slot, node)) return cuda_failure("idle_peer_release");
        // Change allocators before assembling the envelope, never while active.
        if (cuda_pack_q || out_device >= 0) {
            slot.payload.clear();
            bool creating = false;
            ggml_remote_attn_trace_scope timing("cuda_context_prepare", node);
            timing.event.flags = 0x40000000u;
            if ((q_device >= 0 && out_device >= 0 && q_device != out_device) ||
                    !rkva_prepare_cuda_context(ctx, slot, node, q_device >= 0 ? q_device : out_device, api, creating)) return cuda_failure("context_prepare");
            if (creating) {
                timing.event.value = 1;
                std::lock_guard<std::mutex> slock(ctx->stats_mutex);
                ++ctx->stats.cuda_context_creates;
            }
            slot.payload.resize(sizeof(header));
            std::memcpy(slot.payload.data(), &header, sizeof(header));
        }
        if (cuda_pack_q) {
            ggml_remote_attn_trace_scope trace_q("q_pack");
            trace_q.event.bytes = nq * out_width;
            slot.query_fp8.resize(nq * out_width);
            slot.cuda_context_retired = false;
            const bool packed = slot.cuda_small ? slot.cuda_small_api->pack_small(slot.cuda_context,
                    node->src[0]->data, false, nq, out_width, slot.query_fp8.data(), rkva_cuda_producer_event) :
                api->pack_q(slot.cuda_context, node->src[0]->data, false, nq * out_width,
                    slot.query_fp8.data(), rkva_cuda_producer_event);
            slot.cuda_context_retired = packed;
            if (!packed) return cuda_failure(slot.cuda_small ? "small_pack" : "pack");
            std::lock_guard<std::mutex> slock(ctx->stats_mutex);
            ++ctx->stats.cuda_pack_calls;
            ctx->stats.cuda_small_pack_calls += slot.cuda_small;
            ctx->stats.q_f32_readback_bytes_avoided += uint64_t(nq) * out_width * sizeof(float);
        } else {
            ggml_remote_attn_trace_scope trace_q("q_readback");
            trace_q.event.bytes = nq * out_width * sizeof(float);
            slot.f32_scratch.resize(nq * out_width);
            ggml_backend_tensor_get(node->src[0], slot.f32_scratch.data(), 0,
                    slot.f32_scratch.size() * sizeof(float));
        }
        const auto * bulk_queries = slot.f32_scratch.data();
        ggml_remote_attn_trace_scope trace_envelope("envelope");
        trace_envelope.event.flags = 0x40000000u;
        const uint64_t zeros_before = slot.payload.zeroed_bytes();
        for (size_t first = 0; first < nq; first += chunk_limit) {
            const size_t chunk = std::min(chunk_limit, nq - first);
            const bool is_bootstrap = (first == 0) && (slot.start_rows == 0);
            {
            ggml_remote_attn_trace_scope trace_append("kv_append");
            if (!owner->append(slot.original_k.data() + first * width,
                    slot.original_v.data() + first * width,
                    slot.pos_scratch.data() + first, uint32_t(chunk),
                    is_bootstrap, slot.segment_scratch)) {
                ctx->failed = true;
                return false;
            }
            }
            const rkva_kvarn_range_bulk_segment segment {
                uint32_t(slot.segment_scratch.size()), uint32_t(chunk)};
            const rkva_range_attn attention {wire, 0, {0, 0}};
            const size_t offset = slot.payload.size();
            const size_t query_bytes = chunk * (sizeof(int32_t) + out_width * q_elem_bytes);
            slot.payload.resize(offset + sizeof(segment) + slot.segment_scratch.size() +
                    sizeof(attention) + query_bytes);
            auto * dst = slot.payload.data() + offset;
            std::memcpy(dst, &segment, sizeof(segment)); dst += sizeof(segment);
            std::memcpy(dst, slot.segment_scratch.data(), slot.segment_scratch.size());
            dst += slot.segment_scratch.size();
            std::memcpy(dst, &attention, sizeof(attention)); dst += sizeof(attention);
            std::memcpy(dst, slot.pos_scratch.data() + first, chunk * sizeof(int32_t));
            dst += chunk * sizeof(int32_t);
            if (wire == RKVA_WIRE_FP8) {
                uint8_t * dst_fp8 = dst;
                if (cuda_pack_q) {
                    ggml_remote_attn_trace_scope trace_copy("q_payload_copy");
                    trace_copy.event.bytes = chunk * out_width;
                    std::memcpy(dst_fp8, slot.query_fp8.data() + first * out_width, chunk * out_width);
                } else {
                    ggml_remote_attn_trace_scope trace_cpu_pack("q_cpu_pack");
                    trace_cpu_pack.event.bytes = chunk * out_width;
                    if (!rkva_f32_to_fp8_batch(bulk_queries + first * out_width, dst_fp8, chunk * out_width)) {
                        ctx->failed = true; return false;
                    }
                }
            } else if (wire == RKVA_WIRE_SPLITZIP) {
                slot.decode_f16.resize(chunk * out_width);
                if (!rkva_f32_to_f16_batch(bulk_queries + first * out_width, slot.decode_f16.data(), chunk * out_width)) {
                    ctx->failed = true; return false;
                }
                std::vector<uint8_t> sz_out;
                rkva_splitzip_encode(slot.decode_f16.data(), slot.decode_f16.size(), sz_out);
                if (sz_out.size() <= chunk * out_width * sizeof(uint16_t)) {
                    std::memcpy(dst, sz_out.data(), sz_out.size());
                } else {
                    std::memcpy(dst, slot.decode_f16.data(), slot.decode_f16.size() * sizeof(uint16_t));
                }
            } else if (wire == RKVA_WIRE_F16) {
                uint16_t * dst_f16 = reinterpret_cast<uint16_t *>(dst);
                if (!rkva_f32_to_f16_batch(bulk_queries + first * out_width, dst_f16, chunk * out_width)) {
                    ctx->failed = true; return false;
                }
            } else {
                std::memcpy(dst, bulk_queries + first * out_width,
                        chunk * out_width * sizeof(float));
            }
            slot.upload_bytes += slot.segment_scratch.size() - sizeof(rkva_kvarn_range_upload);
        }
        trace_envelope.event.value = slot.payload.zeroed_bytes() - zeros_before;
        trace_envelope.event.bytes = slot.payload.size();
        slot.end_rows = owner->rows();
        if (slot.payload.size() > (256u << 20)) {
            ctx->failed = true;
            return false;
        }
        slot.output = static_cast<uint8_t *>(node->data);
        slot.output_stride = node->nb[1];
        slot.layer = uint32_t(layer);
        slot.nq = uint32_t(nq);
        slot.segments = segments;
        slot.out_width = uint32_t(out_width);
        slot.first_pos = slot.pos_scratch.front();
        slot.last_pos = slot.pos_scratch.back();
        trace_prepare.event.first_pos = slot.first_pos;
        trace_prepare.event.last_pos = slot.last_pos;
        slot.wire = wire;
        slot.bulk_f16 = (wire == RKVA_WIRE_F16);
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

// 2-slot queue submitter. Caller holds ctx->mutex. If the queue is saturated,
// releases ctx->mutex to allow the worker thread to commit ACKs while waiting
// for deterministic backpressure.
static bool rkva_async_submit_locked(
        ggml_backend_remote_attn_context * ctx, ggml_tensor * node,
        uint32_t layer, uint32_t live, size_t nq, size_t out_width,
        std::unique_lock<std::mutex> & ctx_lock) {
    const uint32_t metadata = uint32_t(ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_COHORT));
    if (metadata & ~(uint32_t(GGML_REMOTE_ATTN_COHORT_WIRE_MASK) | uint32_t(GGML_REMOTE_ATTN_WAVEFRONT_SHARED_CUDA))) {
        GGML_LOG_ERROR("%s: invalid async metadata=0x%08x layer=%u nq=%zu\n",
                GGML_REMOTE_ATTN_NAME, metadata, layer, nq);
        return false;
    }
    const bool shared_cuda = (metadata & GGML_REMOTE_ATTN_WAVEFRONT_SHARED_CUDA) != 0;
    const uint32_t cohort = metadata & GGML_REMOTE_ATTN_COHORT_WIRE_MASK;
    const uint32_t cohort_count = cohort & 0xffu, cohort_index = (cohort >> 8) & 0xffu;
    if (cohort && (!ctx->cohort_enabled || !cohort_count ||
            cohort_count > 4 || cohort_index >= cohort_count || !nq || nq > 256)) return false;
    if (!ctx->async_worker.joinable()) {
        try {
            ctx->async_worker = std::thread(rkva_async_worker_main, ctx);
        } catch (...) {
            ctx->failed = true;
            return false;
        }
    }

    const uint64_t submitted_epoch = ctx->range_epoch;
    const uint32_t submitted_session = ctx->active_session;
    const auto timeout = std::chrono::milliseconds([] {
        const char * env = std::getenv("GGML_RKVA_ASYNC_TIMEOUT_MS");
        return env ? std::max(1, std::atoi(env)) : 10000;
    }());
    const auto deadline = std::chrono::steady_clock::now() + timeout;

    rkva_async_slot * slot = nullptr;
    for (;;) {
        if (ctx->failed || ctx->async_stop || ctx->range_epoch != submitted_epoch || ctx->active_session != submitted_session) {
            return false;
        }

        {
            std::unique_lock<std::mutex> alock(ctx->async_mutex);
            if (cohort) {
                uint32_t members = 0, queries = uint32_t(nq);
                uint32_t segments = uint32_t(nq > 9 ? (nq + 7) / 8 : 1);
                uint64_t base = 0;
                for (const auto & s : ctx->async_slots) {
                    if (s.state == rkva_slot_state::FREE) continue;
                    if (!cohort_index || s.state != rkva_slot_state::PENDING ||
                            s.cohort_count != cohort_count || s.cohort_index >= cohort_index ||
                            s.epoch != submitted_epoch || s.layer == layer) return false;
                    if (!members) base = s.ticket - s.cohort_index;
                    if (s.ticket != base + s.cohort_index) return false;
                    ++members; queries += s.nq; segments += s.segments;
                }
                if (members != cohort_index || queries > 256 || segments > 32 ||
                        (cohort_index && ctx->async_ticket_counter != base + cohort_index - 1)) return false;
            } else {
                for (const auto & s : ctx->async_slots)
                    if (s.state != rkva_slot_state::FREE && s.cohort_count) return false;
            }
            for (auto & s : ctx->async_slots) {
                if (s.id < ctx->async_capacity && s.state == rkva_slot_state::FREE) {
                    slot = &s;
                    break;
                }
            }
            if (slot) {
                if (ctx->async_ticket_counter == UINT64_MAX) return false;
                slot->state = rkva_slot_state::PREPARING;
                slot->ticket = ++ctx->async_ticket_counter;
                slot->epoch = ctx->range_epoch;
                slot->session = submitted_session;
                slot->node = node;
                slot->cohort_count = cohort_count;
                slot->cohort_index = cohort_index;
                slot->cuda_shared_requested = shared_cuda;
                ggml_set_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_TICKET_LO, int32_t(slot->ticket & 0xffffffffu));
                ggml_set_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_TICKET_HI, int32_t((slot->ticket >> 32) & 0xffffffffu));
                ggml_set_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_EPOCH_LO,  int32_t(slot->epoch & 0xffffffffu));
                ggml_set_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_EPOCH_HI,  int32_t((slot->epoch >> 32) & 0xffffffffu));
                break;
            }
        }

        // Apply deterministic backpressure: unlock ctx_lock before waiting on async_mutex
        ctx_lock.unlock();
        bool acquired = true;
        {
            std::unique_lock<std::mutex> wait_lock(ctx->async_mutex);
            acquired = ctx->async_cv.wait_until(wait_lock, deadline, [ctx] {
                return ctx->failed || ctx->async_stop || rkva_has_free_slot(ctx);
            });
        } // wait_lock is completely unlocked before acquiring ctx_lock!
        ctx_lock.lock();

        if (!acquired || ctx->failed || ctx->async_stop || ctx->range_epoch != submitted_epoch || ctx->active_session != submitted_session) {
            GGML_LOG_ERROR("%s: async slot backpressure timeout or failure\n", GGML_REMOTE_ATTN_NAME);
            ctx->failed = true;
            return false;
        }
    }

    if (!rkva_prepare_native_bulk_job(ctx, node, layer, live, nq, out_width, *slot)) {
        std::lock_guard<std::mutex> alock(ctx->async_mutex);
        slot->reset();
        ctx->failed = true;
        ctx->async_cv.notify_all();
        return false;
    }

    {
        std::lock_guard<std::mutex> alock(ctx->async_mutex);
        if (ctx->failed || ctx->async_stop || slot->epoch != ctx->range_epoch ||
                slot->state != rkva_slot_state::PREPARING) {
            GGML_LOG_ERROR("%s: async preparation invalidated layer=%u nq=%zu slot=%u state=%d failed=%d stop=%d\n",
                    GGML_REMOTE_ATTN_NAME, layer, nq, slot->id, int(slot->state), int(ctx->failed.load()), int(ctx->async_stop.load()));
            slot->state = rkva_slot_state::DONE_ERR;
            ctx->async_cv.notify_all();
            return false;
        }
        if (ggml_remote_attn_trace_enabled()) slot->pending_us = ggml_time_us();
        slot->state = rkva_slot_state::PENDING;
    }
    ctx->async_cv.notify_all();
    return true;
}

static bool rkva_run_range(
        ggml_backend_remote_attn_context * ctx, ggml_tensor * node,
        std::unique_lock<std::mutex> & ctx_lock) {
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
    const auto invalid_range = [&](const char * stage) {
        GGML_LOG_ERROR("%s: range validation failed stage=%s layer=%d nq=%zu live=%d origin=%d\n",
                GGML_REMOTE_ATTN_NAME, stage, layer, nq, n_live, origin);
        return false;
    };
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
        GGML_LOG_ERROR("%s: range geometry configured=%d session=%u source_type=%d q_type=%d q_contiguous=%d q=%lld/%lld/%lld/%lld k_type=%d k=%lld/%lld/%lld/%lld out_contiguous=%d out=%lld/%lld capacity=%u\n",
                GGML_REMOTE_ATTN_NAME, int(ctx->range_configured), ctx->active_session,
                int(ctx->range_source_type), q ? int(q->type) : -1, q ? int(ggml_is_contiguous(q)) : 0,
                q ? (long long)q->ne[0] : -1LL, q ? (long long)q->ne[1] : -1LL,
                q ? (long long)q->ne[2] : -1LL, q ? (long long)q->ne[3] : -1LL,
                k ? int(k->type) : -1, k ? (long long)k->ne[0] : -1LL,
                k ? (long long)k->ne[1] : -1LL, k ? (long long)k->ne[2] : -1LL,
                k ? (long long)k->ne[3] : -1LL, int(ggml_is_contiguous(node)),
                (long long)node->ne[0], (long long)node->ne[1], ctx->range_capacity);
        return invalid_range("geometry");
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
                return invalid_range("cell_history"); // holes/reordering are not append-only range histories
            }
            ++count;
        }
    }
    if (count == 0 || count > (uint64_t) k->ne[1] || count > (uint64_t) v->ne[1]) {
        return invalid_range("coverage");
    }
    ctx->pos_scratch.resize(nq);
    ggml_backend_tensor_get(pos, ctx->pos_scratch.data(), 0, nq * sizeof(int32_t));
    for (size_t i = 0; i < nq; ++i) {
        const int64_t p = ctx->pos_scratch[i];
        if (p < ctx->range_origin || p >= (int64_t) ctx->range_origin + count ||
                (i && p != (int64_t) ctx->pos_scratch[i-1] + 1)) {
            return invalid_range("query_positions");
        }
    }
    if ((uint32_t) ctx->pos_scratch.back() + 1 != ctx->range_origin + count) {
        return invalid_range("query_suffix");
    }
    if (ctx->range_native_kvarn) {
        if (ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_ASYNC)) {
            // The scheduler executes local attention between this submit and
            // the companion wait node; only the ordered bulk RPC is deferred.
            return ctx->async_local_remote && rkva_async_submit_locked(
                    ctx, node, uint32_t(layer), count, nq, out_width, ctx_lock);
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
        ggml_tensor * node,
        std::unique_lock<std::mutex> & ctx_lock) {

    if (ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_RANGE)) {
        return rkva_run_range(ctx, node, ctx_lock);
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
        // Must pair with the allocator: the MSVC CRT requires _aligned_free
        // on _aligned_malloc memory, and free() on it is undefined behaviour.
#if defined(_WIN32)
        _aligned_free(ctx->base);
#else
        free(ctx->base);
#endif
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
#if defined(_WIN32)
        void * ptr = _aligned_malloc(size, 64);
        if (ptr == nullptr) {
            delete ctx;
            return nullptr;
        }
#else
        void * ptr = nullptr;
        if (posix_memalign(&ptr, 64, size) != 0 || ptr == nullptr) {
            delete ctx;
            return nullptr;
        }
#endif
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
    /* .alloc_buffer_n= */ nullptr,
    /* .get_alignment = */ ggml_backend_remote_attn_buft_get_alignment,
    /* .get_max_size  = */ ggml_backend_remote_attn_buft_get_max_size,
    /* .get_alloc_size= */ ggml_backend_remote_attn_buft_get_alloc_size,
    /* .get_alloc_size_n= */ nullptr,
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
        ctx->close_socket();
        ctx->connected = false;
    }
    delete ctx;
    ggml_remote_attn_trace_flush();
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
    auto * ctx = static_cast<ggml_backend_remote_attn_context *>(backend->context);
    std::unique_lock<std::mutex> lock(ctx->async_mutex);
    ctx->async_cv.wait(lock, [ctx] {
        rkva_cancel_partial_cohort_locked(ctx);
        for (const auto & s : ctx->async_slots) {
            if (s.state == rkva_slot_state::PENDING ||
                s.state == rkva_slot_state::RUNNING ||
                s.state == rkva_slot_state::PREPARING) {
                return false;
            }
        }
        return true;
    });
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

        if (ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_WAIT)) {
            if (!rkva_async_wait_node(ctx, node)) {
                return GGML_STATUS_ABORTED;
            }
            continue;
        }

        const bool is_async = ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_ASYNC) != 0;
        auto lock = is_async ?
            std::unique_lock<std::mutex>(ctx->mutex) : rkva_lock_ctx_drained(ctx);
        if (ctx->failed || !ctx->connected) {
            GGML_LOG_ERROR("%s: backend not connected (failed=%d)\n",
                           GGML_REMOTE_ATTN_NAME, (int) ctx->failed);
            return GGML_STATUS_ABORTED;
        }
        if (!rkva_run_attn(ctx, node, lock)) {
            rkva_poison_remote_state_locked(ctx);
            if (!is_async) {
                // Drained lock: no job can reference owners/vectors now.
                rkva_destroy_poisoned_state_locked(ctx);
            }
            ctx->async_cv.notify_all();
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
    // Fast fail while the endpoint is marked down: no socket, no timeout.
    if (!rkva_breaker_allows_attempt(ctx)) {
        return false;
    }
    auto lock = rkva_lock_ctx_drained(ctx);
    // connect() normally clears failed. Never revive storage whose codec
    // helper failed to establish successful retirement; require a new backend.
    bool retired = ctx->synchronous_slot.cuda_context_retired;
    for (const auto & slot : ctx->async_slots) retired = retired && slot.cuda_context_retired;
    if (!retired) { ctx->failed = true; return false; }
    if (!ctx->geo_set) {
        GGML_LOG_ERROR("%s: geometry not set before connect\n", GGML_REMOTE_ATTN_NAME);
        return false;
    }
    ctx->close_socket();
    ctx->connected = false;
    ctx->failed = false;
    ctx->range_configured = false;
    ctx->range_layers.clear();
    ctx->range_native_kvarn = false;
    ctx->native_q4 = false;
    ctx->native_q8 = false;
    ctx->cohort_enabled = false;
    ctx->async_capacity = 2;
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
    ctx->sock.assign(fd);
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
        ctx->close_socket();
        return false;
    }
    if (::connect(fd, (sockaddr *) &addr, sizeof(addr)) != 0) {
        GGML_LOG_ERROR("%s: connect to %s:%u failed: %s\n",
                       GGML_REMOTE_ATTN_NAME, ctx->host.c_str(), (unsigned) ctx->port, strerror(errno));
        ctx->close_socket();
        rkva_breaker_mark_down(ctx);
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
        ctx->close_socket();
        return false;
    }
    std::memcpy(&ctx->ack, ack_bytes.data(), sizeof(rkva_hello_ack));
    if (!(ctx->ack.supported_ops & (1u << RKVA_ATTN_DECODE)) ||
        !(ctx->ack.supported_ops & (1u << RKVA_ATTN_PREFILL))) {
        GGML_LOG_ERROR("%s: server lacks ATTN ops (mask=0x%x)\n",
                       GGML_REMOTE_ATTN_NAME, ctx->ack.supported_ops);
        ctx->connected = false;
        ctx->close_socket();
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
    // A failed/refused reset must retire every derived state: a later request
    // must not match a stale session, range layer or native owner.
    if (!ctx->connected || ctx->failed) {
        rkva_poison_remote_state_locked(ctx);
        rkva_destroy_poisoned_state_locked(ctx); // drained lock: safe
        return false;
    }
    ++ctx->range_epoch;
    std::vector<uint8_t> out;
    if (!rkva_transact(ctx, RKVA_RESET, session_id, 0, 0, nullptr, 0, out, 0) || !out.empty()) {
        rkva_poison_remote_state_locked(ctx);
        rkva_destroy_poisoned_state_locked(ctx);
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
        rkva_poison_remote_state_locked(ctx); // ACK failure cannot publish speculative coverage
        rkva_destroy_poisoned_state_locked(ctx);
        return false;
    }
    for (auto & layer : ctx->range_layers) {
        if (layer.epoch == ctx->range_epoch) layer.rows = std::min(layer.rows, keep);
    }
    return true;
}

// Caller owns the context mutex. Never publish host coverage before the
// session-wide GPU tail repair has completed and its ACK was validated.
// Layouts 4/5 keep plain Q4_0/Q8_0 rows in the server's standard range buffer,
// so a suffix removal uses the session-wide range trim (op8): no exact F16 tail
// and no native rewind payload are involved. Owners only mirror the row count.
static bool rkva_trim_native_q4_locked(ggml_backend_remote_attn_context * ctx,
        uint32_t session_id, int32_t pos0) {
    if (session_id != ctx->active_session || pos0 < 0) return false;
    const uint32_t keep = uint32_t(pos0) <= ctx->range_origin ? 0 :
            std::min(uint32_t(pos0) - ctx->range_origin, ctx->range_capacity);
    bool needed = false;
    for (size_t il = 0; il < ctx->native_layers.size(); ++il) {
        const auto & owner = ctx->native_layers[il];
        if (!owner) continue;
        if (ctx->range_layers[il].epoch != ctx->range_epoch) return false;
        needed |= owner->rows() > keep;
    }
    if (!needed) return true;
    if (!(ctx->ack.supported_ops & (1u << RKVA_REWIND))) {
        // Older server: drop host ownership and rebuild on the next graph.
        ++ctx->range_epoch;
        for (auto & owner : ctx->native_layers) owner.reset();
        ctx->range_layers.assign(ctx->native_layers.size(), {});
        return true;
    }
    rkva_trim_req request { pos0 };
    std::vector<uint8_t> reply;
    if (!rkva_transact(ctx, RKVA_REWIND, session_id, 0, 0,
                &request, sizeof(request), reply, 0) || !reply.empty()) {
        ctx->failed = true; // ACK failure cannot publish speculative coverage
        return false;
    }
    for (size_t il = 0; il < ctx->native_layers.size(); ++il) {
        auto & owner = ctx->native_layers[il];
        if (owner && !owner->rewind(std::min(owner->rows(), keep))) {
            ctx->failed = true;
            return false;
        }
        ctx->range_layers[il].epoch = ctx->range_epoch;
        ctx->range_layers[il].rows = owner ? owner->rows() : 0;
    }
    return true;
}

static bool rkva_native_rewind_locked(ggml_backend_remote_attn_context * ctx,
        uint32_t session_id, int32_t pos0) {
    if (ctx->native_q4 || ctx->native_q8) return rkva_trim_native_q4_locked(ctx, session_id, pos0);
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
            rkva_poison_remote_state_locked(ctx);
            rkva_destroy_poisoned_state_locked(ctx);
            return false;
        }
        ++ctx->native_generation;
        for (size_t il = 0; il < ctx->native_layers.size(); ++il) {
            auto & owner = ctx->native_layers[il];
            if (owner && !owner->rewind(std::min(owner->rows(), keep))) {
                rkva_poison_remote_state_locked(ctx);
                rkva_destroy_poisoned_state_locked(ctx);
                return false;
            }
            ctx->range_layers[il].rows = owner ? owner->rows() : 0;
        }
        return true;
    } catch (...) {
        rkva_poison_remote_state_locked(ctx);
        rkva_destroy_poisoned_state_locked(ctx);
        return false;
    }
}

bool ggml_backend_remote_attn_trim(ggml_backend_t backend, uint32_t session_id, int32_t pos0) {
    ggml_remote_attn_trace_scope trace_trim("control_trim");
    trace_trim.event.session = session_id;
    trace_trim.event.first_pos = pos0;
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

static bool rkva_async_local_remote_option(bool & async_overlap, bool & explicitly_set) {
    // Overlap is the default for the native bulk path; set the env to 0 to
    // serialize local compute after the socket transaction.
    async_overlap = true;
    explicitly_set = false;
    if (const char * option = std::getenv("GGML_RKVA_ASYNC_LOCAL_REMOTE")) {
        explicitly_set = true;
        if (!std::strcmp(option, "1")) async_overlap = true;
        else if (!std::strcmp(option, "0")) async_overlap = false;
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
    // Native bulk RPC is the default: one ordered envelope per layer with a
    // single submit/fence, and FP8 activations when the server advertises the
    // RNE codec (F16 otherwise). No legacy fallback: a server without the bulk
    // capabilities is rejected by the check below.
    bool bulk = true;
    rkva_wire bulk_wire = RKVA_WIRE_F16;
    bool bulk_wire_auto = true;
    if (const char * option = std::getenv("GGML_RKVA_NATIVE_BULK")) {
        if (!std::strcmp(option, "1")) {
            bulk = true;
            bulk_wire = RKVA_WIRE_F16;
        } else if (!std::strcmp(option, "f16") || !std::strcmp(option, "fp16")) {
            bulk = true;
            bulk_wire = RKVA_WIRE_F16;
        } else if (!std::strcmp(option, "fp8") || !std::strcmp(option, "fp8_e5m2")) {
            bulk = true;
            bulk_wire = RKVA_WIRE_FP8;
        } else if (!std::strcmp(option, "auto")) {
            bulk = true;
            bulk_wire_auto = true;
        } else if (!std::strcmp(option, "splitzip")) {
            GGML_LOG_ERROR("%s: SplitZip wire format is deactivated; use lossless 'f16' instead\n", GGML_REMOTE_ATTN_NAME);
            return false;
        } else if (!std::strcmp(option, "0")) {
            // Explicit opt-out for legacy-layout compatibility tests. Production
            // default remains bulk; a non-bulk server is only used when asked.
            bulk = false;
        }
    }
    if (const char * wire_env = std::getenv("GGML_RKVA_WIRE_FORMAT")) {
        bulk = true;
        if (!std::strcmp(wire_env, "f16") || !std::strcmp(wire_env, "fp16")) {
            bulk_wire = RKVA_WIRE_F16;
            bulk_wire_auto = false;
        } else if (!std::strcmp(wire_env, "fp8") || !std::strcmp(wire_env, "fp8_e5m2")) {
            bulk_wire = RKVA_WIRE_FP8;
            bulk_wire_auto = false;
        } else if (!std::strcmp(wire_env, "auto")) {
            bulk_wire_auto = true;
        } else if (!std::strcmp(wire_env, "splitzip")) {
            GGML_LOG_ERROR("%s: SplitZip wire format is deactivated; use lossless 'f16' instead\n", GGML_REMOTE_ATTN_NAME);
            return false;
        } else if (!std::strcmp(wire_env, "f32") || !std::strcmp(wire_env, "fp32")) {
            bulk_wire = RKVA_WIRE_F32;
            bulk_wire_auto = false;
        } else {
            GGML_LOG_ERROR("%s: unrecognized GGML_RKVA_WIRE_FORMAT '%s' (expected f16, fp8, auto, f32)\n",
                GGML_REMOTE_ATTN_NAME, wire_env);
            return false;
        }
    } else if (const char * wire_env = std::getenv("GGML_RKVA_NATIVE_BULK_WIRE")) {
        if (!std::strcmp(wire_env, "f16") || !std::strcmp(wire_env, "fp16")) {
            bulk_wire = RKVA_WIRE_F16;
            bulk_wire_auto = false;
        } else if (!std::strcmp(wire_env, "fp8") || !std::strcmp(wire_env, "fp8_e5m2")) {
            bulk_wire = RKVA_WIRE_FP8;
            bulk_wire_auto = false;
        } else if (!std::strcmp(wire_env, "auto")) {
            bulk_wire_auto = true;
        } else if (!std::strcmp(wire_env, "splitzip")) {
            GGML_LOG_ERROR("%s: SplitZip wire format is deactivated; use lossless 'f16' instead\n", GGML_REMOTE_ATTN_NAME);
            return false;
        } else if (!std::strcmp(wire_env, "f32") || !std::strcmp(wire_env, "fp32")) {
            bulk_wire = RKVA_WIRE_F32;
            bulk_wire_auto = false;
        } else {
            return false;
        }
    }
    if (bulk && (!serial_prefix || !(ctx->ack.reserved[0] & RKVA_CAP_KVARN_BULK_ATTN) ||
                 !(ctx->ack.supported_ops & (1u << RKVA_KVARN_RANGE_BULK_ATTN)))) {
        GGML_LOG_ERROR("%s: requested native bulk RPC is not supported by the negotiated server\n", GGML_REMOTE_ATTN_NAME);
        return false;
    }
    const uint64_t fp8_required = RKVA_CAP_KVARN_BULK_FP8 | RKVA_CAP_KVARN_BULK_FP8_RNE;
    if (bulk && bulk_wire_auto) {
        if ((ctx->ack.reserved[0] & fp8_required) == fp8_required) {
            bulk_wire = RKVA_WIRE_FP8;
        } else if (ctx->ack.reserved[0] & RKVA_CAP_KVARN_BULK_F16) {
            GGML_LOG_WARN("%s: remote server does not advertise FP8 RNE capabilities; falling back to negotiated F16 wire (auto mode)\n", GGML_REMOTE_ATTN_NAME);
            bulk_wire = RKVA_WIRE_F16;
        } else {
            GGML_LOG_ERROR("%s: auto wire selection failed: remote server does not advertise FP8 or F16 bulk capability\n", GGML_REMOTE_ATTN_NAME);
            return false;
        }
    } else if (bulk && bulk_wire == RKVA_WIRE_FP8 &&
               (ctx->ack.reserved[0] & fp8_required) != fp8_required) {
        GGML_LOG_ERROR("%s: requested native bulk FP8 RNE RPC is not supported by the negotiated server (missing FP8/RNE capabilities)\n", GGML_REMOTE_ATTN_NAME);
        return false;
    }
    if (bulk && bulk_wire == RKVA_WIRE_SPLITZIP) {
        GGML_LOG_ERROR("%s: SplitZip wire format is deactivated; use lossless 'f16' instead\n", GGML_REMOTE_ATTN_NAME);
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
    bool cohort_enabled = false;
    if (const char * option = std::getenv("GGML_RKVA_WAVEFRONT_RPC_BATCH")) {
        if (!std::strcmp(option, "4")) {
            GGML_LOG_ERROR("%s: WF RPC batch 4 is unavailable: no qualified cohort server; use RPC batch 1\n", GGML_REMOTE_ATTN_NAME);
            return false;
        }
        else if (std::strcmp(option, "1")) {
            GGML_LOG_ERROR("%s: GGML_RKVA_WAVEFRONT_RPC_BATCH requires 1 or 4\n", GGML_REMOTE_ATTN_NAME);
            return false;
        }
    }
    if (cohort_enabled && (!async_overlap || !bulk || !serial_prefix || bulk_wire != RKVA_WIRE_FP8 ||
            !(ctx->ack.reserved[0] & RKVA_CAP_KVARN_COHORT_ATTN) ||
            !(ctx->ack.supported_ops & (1u << RKVA_KVARN_RANGE_COHORT_ATTN)))) {
        GGML_LOG_ERROR("%s: explicit cohort batching requires negotiated FP8 op23 and async serial-prefix\n", GGML_REMOTE_ATTN_NAME);
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
    ctx->native_q4 = false;
    ctx->native_q8 = false;
    ctx->native_serial_prefix = serial_prefix;
    ctx->native_bulk = bulk;
    ctx->async_local_remote = async_overlap;
    ctx->cohort_enabled = cohort_enabled;
    ctx->async_capacity = cohort_enabled ? 4 : 2;
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
    const char * wire_str = bulk ? (bulk_wire == RKVA_WIRE_FP8 ? "fp8_e5m2" :
                                    bulk_wire == RKVA_WIRE_F16 ? "f16" : "f32") : "f32";
    GGML_LOG_WARN("%s: EXPERIMENTAL native KVarN4 precision_tail=%u descriptor=%u serial_prefix=%d async_overlap=%d wire=%s; original F32 producer, bounded resident rewind requires negotiated capability; FULL snapshots unsupported\n",
                  GGML_REMOTE_ATTN_NAME, precision_tail_tokens, descriptor.layout_version, int(serial_prefix), int(async_overlap), wire_str);
    return true;
}

// Descriptor layout 4: native Q4_0 overflow rows through the same op16/op22
// envelope and async overlap as KVarN4. The KV payload is ordinary ggml Q4_0
// blocks (no sinkhorn/WHT/records/stage/tail) while Q/OUT stay on the FP8
// activation wire. Negotiated only when the server advertises
// RKVA_CAP_RANGE_Q4_0 plus the existing bulk caps; otherwise the caller keeps
// the synchronous Q4 range path.
static bool rkva_configure_native_q4(ggml_backend_remote_attn_context * ctx,
        uint32_t origin, uint32_t capacity, ggml_type source_type, bool async_overlap) {
    constexpr uint64_t caps = RKVA_CAP_RANGE_Q4_0 | RKVA_CAP_NATIVE_Q4_BULK |
                              RKVA_CAP_KVARN_BULK_ATTN | RKVA_CAP_KVARN_BULK_FP8 |
                              RKVA_CAP_KVARN_BULK_FP8_RNE;
    constexpr uint32_t required = (1u << RKVA_KVARN_RANGE_CONFIG) |
                                  (1u << RKVA_KVARN_RANGE_UPLOAD) |
                                  (1u << RKVA_KVARN_RANGE_BULK_ATTN);
    if (!ctx->connected || ctx->failed || ctx->active_session == RKVA_SESSION_NONE ||
            source_type != GGML_TYPE_Q4_0 || !capacity ||
            origin % 256 || uint64_t(origin) + capacity > INT32_MAX ||
            ctx->geo.max_ctx != capacity || ctx->geo.head_dim != rkva_q4_producer::head_dim ||
            ctx->geo.n_head_kv == 0 || ctx->geo.n_layer_remote == 0 ||
            ctx->geo.domain != GGML_REMOTE_ATTN_DOMAIN_AUTO ||
            ctx->geo.has_sinks || ctx->geo.swa) {
        return false;
    }
    if ((ctx->ack.reserved[0] & caps) != caps || (ctx->ack.supported_ops & required) != required) {
        GGML_LOG_INFO("%s: native Q4_0 bulk is not advertised by the negotiated server; "
                "using the synchronous range path\n", GGML_REMOTE_ATTN_NAME);
        return false;
    }
    // Layout 4 mandates the FP8 Q/OUT activation wire. Follow the same option
    // precedence as the KVarN4 bulk path; an explicit non-FP8 request keeps
    // the previous synchronous path instead of downgrading.
    const char * wire_option = std::getenv("GGML_RKVA_WIRE_FORMAT");
    const char * bulk_wire_option = std::getenv("GGML_RKVA_NATIVE_BULK_WIRE");
    const char * bulk_option = std::getenv("GGML_RKVA_NATIVE_BULK");
    const char * selected_wire = wire_option ? wire_option :
            bulk_wire_option ? bulk_wire_option : bulk_option;
    const bool selected_is_bulk = !wire_option && !bulk_wire_option;
    const bool fp8_selected = !selected_wire ||
            !std::strcmp(selected_wire, "fp8") || !std::strcmp(selected_wire, "fp8_e5m2") ||
            !std::strcmp(selected_wire, "auto") ||
            (selected_is_bulk && !std::strcmp(selected_wire, "1"));
    if (!fp8_selected) {
        GGML_LOG_INFO("%s: native Q4_0 bulk requires the FP8 activation wire; "
                "using the synchronous range path\n", GGML_REMOTE_ATTN_NAME);
        return false;
    }
    uint32_t mode = 0;
    bool profile = false, generic = false;
    if (!rkva_range_kernel_options(mode, profile, generic)) return false;
    const bool can_profile = (ctx->ack.reserved[0] & RKVA_CAP_RANGE_PROFILE) &&
                             (ctx->ack.supported_ops & (1u << RKVA_RANGE_PROFILE));
    if ((mode || profile) && !can_profile) return false;
    const rkva_kvarn_range_config descriptor {
        origin, capacity, RKVA_RANGE_STORAGE_Q4_0, 4u,
        1, 0, rkva_q4_producer::head_dim, 2, 4, 4, rkva_q4_producer::record_bytes,
        0, 0, 0, 0, 0, 0, 1};
    std::vector<uint8_t> reply;
    if (!rkva_transact(ctx, RKVA_KVARN_RANGE_CONFIG, ctx->active_session, 0, 0,
            &descriptor, sizeof(descriptor), reply, 0) || !reply.empty()) return false;
    ctx->range_configured = true;
    ctx->range_native_kvarn = true;
    ctx->native_q4 = true;
    ctx->native_q8 = false;
    ctx->native_serial_prefix = false;
    ctx->native_bulk = true;
    ctx->async_local_remote = async_overlap;
    ctx->native_generation = 0;
    ctx->native_tail_tokens = 0;
    ctx->native_store = nullptr;
    ctx->native_layers.clear();
    ctx->native_layers.resize(ctx->geo.n_layer_remote);
    ctx->range_source_type = source_type;
    ctx->range_wire = RKVA_WIRE_FP8; // Q/OUT only; op16 carries Q4_0 rows
    ctx->range_origin = origin;
    ctx->range_capacity = capacity;
    ++ctx->range_epoch;
    ctx->range_layers.assign(ctx->geo.n_layer_remote, {});
    ctx->range_fused = false;
    ctx->range_profile = profile;
    ctx->range_kernel_mode = mode;
    if (!rkva_range_apply_control(ctx, "q4_0", generic)) return false;
    GGML_LOG_WARN("%s: native Q4_0 bulk descriptor=4 origin=%u capacity=%u async_overlap=%d "
            "wire=fp8_e5m2; host CPU q4_0 producer, no KVarN records/stage/tail, "
            "suffix rollback uses the range trim (op8)\n",
            GGML_REMOTE_ATTN_NAME, origin, capacity, int(async_overlap));
    return true;
}

// Descriptor layout 5: native Q8_0 overflow rows through the same op16/op22
// envelope and async overlap as KVarN4. The KV payload is ordinary ggml Q8_0
// blocks (no sinkhorn/WHT/records/stage/tail) while Q/OUT stay on the FP8
// activation wire. Negotiated only when the server advertises
// RKVA_CAP_RANGE_Q8_0 plus RKVA_CAP_NATIVE_Q8_BULK and the existing bulk caps;
// otherwise the caller keeps the synchronous Q8 range path.
static bool rkva_configure_native_q8(ggml_backend_remote_attn_context * ctx,
        uint32_t origin, uint32_t capacity, ggml_type source_type, bool async_overlap) {
    constexpr uint64_t caps = RKVA_CAP_RANGE_Q8_0 | RKVA_CAP_NATIVE_Q8_BULK |
                              RKVA_CAP_KVARN_BULK_ATTN | RKVA_CAP_KVARN_BULK_FP8 |
                              RKVA_CAP_KVARN_BULK_FP8_RNE;
    constexpr uint32_t required = (1u << RKVA_KVARN_RANGE_CONFIG) |
                                  (1u << RKVA_KVARN_RANGE_UPLOAD) |
                                  (1u << RKVA_KVARN_RANGE_BULK_ATTN);
    if (!ctx->connected || ctx->failed || ctx->active_session == RKVA_SESSION_NONE ||
            source_type != GGML_TYPE_Q8_0 || !capacity ||
            origin % 256 || uint64_t(origin) + capacity > INT32_MAX ||
            ctx->geo.max_ctx != capacity || ctx->geo.head_dim != rkva_q8_producer::head_dim ||
            ctx->geo.n_head_kv == 0 || ctx->geo.n_layer_remote == 0 ||
            ctx->geo.domain != GGML_REMOTE_ATTN_DOMAIN_AUTO ||
            ctx->geo.has_sinks || ctx->geo.swa) {
        return false;
    }
    if ((ctx->ack.reserved[0] & caps) != caps || (ctx->ack.supported_ops & required) != required) {
        GGML_LOG_INFO("%s: native Q8_0 bulk is not advertised by the negotiated server; "
                "using the synchronous range path\n", GGML_REMOTE_ATTN_NAME);
        return false;
    }
    // Layout 5 mandates the FP8 Q/OUT activation wire. Follow the same option
    // precedence as the KVarN4 bulk path; an explicit non-FP8 request keeps
    // the previous synchronous path instead of downgrading.
    const char * wire_option = std::getenv("GGML_RKVA_WIRE_FORMAT");
    const char * bulk_wire_option = std::getenv("GGML_RKVA_NATIVE_BULK_WIRE");
    const char * bulk_option = std::getenv("GGML_RKVA_NATIVE_BULK");
    const char * selected_wire = wire_option ? wire_option :
            bulk_wire_option ? bulk_wire_option : bulk_option;
    const bool selected_is_bulk = !wire_option && !bulk_wire_option;
    const bool fp8_selected = !selected_wire ||
            !std::strcmp(selected_wire, "fp8") || !std::strcmp(selected_wire, "fp8_e5m2") ||
            !std::strcmp(selected_wire, "auto") ||
            (selected_is_bulk && !std::strcmp(selected_wire, "1"));
    if (!fp8_selected) {
        GGML_LOG_INFO("%s: native Q8_0 bulk requires the FP8 activation wire; "
                "using the synchronous range path\n", GGML_REMOTE_ATTN_NAME);
        return false;
    }
    uint32_t mode = 0;
    bool profile = false, generic = false;
    if (!rkva_range_kernel_options(mode, profile, generic)) return false;
    const bool can_profile = (ctx->ack.reserved[0] & RKVA_CAP_RANGE_PROFILE) &&
                             (ctx->ack.supported_ops & (1u << RKVA_RANGE_PROFILE));
    if ((mode || profile) && !can_profile) return false;
    const rkva_kvarn_range_config descriptor {
        origin, capacity, RKVA_RANGE_STORAGE_Q8_0, 5u,
        1, 0, rkva_q8_producer::head_dim, 2, 8, 8, rkva_q8_producer::record_bytes,
        0, 0, 0, 0, 0, 0, 1};
    std::vector<uint8_t> reply;
    if (!rkva_transact(ctx, RKVA_KVARN_RANGE_CONFIG, ctx->active_session, 0, 0,
            &descriptor, sizeof(descriptor), reply, 0) || !reply.empty()) return false;
    ctx->range_configured = true;
    ctx->range_native_kvarn = true;
    ctx->native_q4 = false;
    ctx->native_q8 = true;
    ctx->native_serial_prefix = false;
    ctx->native_bulk = true;
    ctx->async_local_remote = async_overlap;
    ctx->native_generation = 0;
    ctx->native_tail_tokens = 0;
    ctx->native_store = nullptr;
    ctx->native_layers.clear();
    ctx->native_layers.resize(ctx->geo.n_layer_remote);
    ctx->range_source_type = source_type;
    ctx->range_wire = RKVA_WIRE_FP8; // Q/OUT only; op16 carries Q8_0 rows
    ctx->range_origin = origin;
    ctx->range_capacity = capacity;
    ++ctx->range_epoch;
    ctx->range_layers.assign(ctx->geo.n_layer_remote, {});
    ctx->range_fused = false;
    ctx->range_profile = profile;
    ctx->range_kernel_mode = mode;
    if (!rkva_range_apply_control(ctx, "q8_0", generic)) return false;
    GGML_LOG_WARN("%s: native Q8_0 bulk descriptor=5 origin=%u capacity=%u async_overlap=%d "
            "wire=fp8_e5m2; host CPU q8_0 producer, no KVarN records/stage/tail, "
            "suffix rollback uses the range trim (op8)\n",
            GGML_REMOTE_ATTN_NAME, origin, capacity, int(async_overlap));
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
    // The split tier is denied immediately while the endpoint cooldown is open.
    if (!rkva_breaker_allows_attempt(ctx)) {
        return false;
    }
    auto lock = rkva_lock_ctx_drained(ctx);
    bool async_overlap = false;
    bool async_explicit = false;
    if (!rkva_async_local_remote_option(async_overlap, async_explicit)) return false;
    if (!async_explicit) {
        // No bulk path => no overlap. This also covers the KVarN4 branch, which
        // returns before the generic non-bulk adjustment below.
        const char * bulk_opt = std::getenv("GGML_RKVA_NATIVE_BULK");
        if (bulk_opt && !std::strcmp(bulk_opt, "0")) {
            async_overlap = false;
        }
    }
    const char * format_req = rkva_get_upload_format();
    if (format_req && (!std::strcmp(format_req, "kvarn4") || !std::strcmp(format_req, "kvarn"))) {
        if (!rkva_configure_native_kvarn(ctx, origin, capacity, cache_type, async_overlap)) return false;
        rkva_breaker_note_configured(ctx);
        return true;
    }
    if (format_req && (!std::strcmp(format_req, "q4") || !std::strcmp(format_req, "q4_0"))) {
        if (rkva_configure_native_q4(ctx, origin, capacity, cache_type, async_overlap)) {
            rkva_breaker_note_configured(ctx);
            return true;
        }
        if (ctx->failed) return false;
        // Not negotiated (old server or missing bulk caps): keep the current
        // synchronous range upload path below, unchanged.
    }
    if (format_req && (!std::strcmp(format_req, "q8") || !std::strcmp(format_req, "q8_0"))) {
        if (rkva_configure_native_q8(ctx, origin, capacity, cache_type, async_overlap)) {
            rkva_breaker_note_configured(ctx);
            return true;
        }
        if (ctx->failed) return false;
        // Not negotiated (old server or missing bulk caps): keep the current
        // synchronous range upload path below, unchanged.
    }
    if (async_overlap && !async_explicit) {
        // Auto default: overlap only on a negotiated bulk path. Non-bulk
        // formats (f16/legacy) keep the synchronous path instead of failing.
        async_overlap = false;
    }
    if (async_overlap) {
        GGML_LOG_ERROR("%s: GGML_RKVA_ASYNC_LOCAL_REMOTE requires "
                "upload format kvarn4 or a native Q4_0/Q8_0 bulk server\n", GGML_REMOTE_ATTN_NAME);
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
    ctx->native_q4 = false;
    ctx->native_q8 = false;
    ctx->cohort_enabled = false;
    ctx->async_capacity = 2;
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
    rkva_breaker_note_configured(ctx);
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
    ggml_remote_attn_trace_scope trace_invalidate("control_invalidate");
    if (!backend || ggml_backend_is_local_split(backend)) {
        return;
    }
    auto * ctx = (ggml_backend_remote_attn_context *) backend->context;
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ++ctx->range_epoch;
    }
    {
        std::lock_guard<std::mutex> alock(ctx->async_mutex);
        for (auto & s : ctx->async_slots) {
            if (s.state == rkva_slot_state::PENDING) {
                if (s.cohort_count) {
                    // Producer state advanced while preparing these entries,
                    // but no complete cohort was acknowledged on the peer.
                    ctx->failed = true;
                    ctx->connected = false;
                }
                s.state = rkva_slot_state::DONE_ERR;
            }
        }
    }
    ctx->async_cv.notify_all();
    rkva_async_drain(ctx);
}

bool ggml_backend_remote_attn_native_kvarn_enabled(ggml_backend_t backend) {
    if (!backend || ggml_backend_is_local_split(backend)) return false;
    const auto * ctx = static_cast<ggml_backend_remote_attn_context *>(backend->context);
    // A failed/down context must never advertise native state, even before the
    // retire path clears range_configured: the caller would build a split graph
    // that can only abort after touching stale owners.
    return ctx->connected && !ctx->failed && ctx->range_configured && ctx->range_native_kvarn;
}

bool ggml_backend_remote_attn_async_enabled(ggml_backend_t backend) {
    if (!backend || ggml_backend_is_local_split(backend)) return false;
    const auto * ctx = static_cast<const ggml_backend_remote_attn_context *>(backend->context);
    return ctx->async_local_remote && ctx->connected && !ctx->failed &&
           ctx->range_configured && ctx->range_native_kvarn && ctx->native_bulk;
}

bool ggml_backend_remote_attn_cohort_enabled(ggml_backend_t backend) {
    if (!ggml_backend_remote_attn_async_enabled(backend)) return false;
    return static_cast<ggml_backend_remote_attn_context *>(backend->context)->cohort_enabled;
}

void ggml_backend_remote_attn_drain(ggml_backend_t backend) {
    if (!backend) return;
    if (ggml_backend_is_local_split(backend)) {
        ggml_backend_synchronize(backend);
        return;
    }
    auto * ctx = static_cast<ggml_backend_remote_attn_context *>(backend->context);
    // rkva_lock_ctx_drained revalidates "no job in flight" under both locks and
    // destroys a poisoned state only after that revalidation. A plain drain +
    // lock would allow a submission to slip in between the two operations.
    auto lock = rkva_lock_ctx_drained(ctx);
}

extern "C" void ggml_backend_remote_attn_test_handshake_enable(bool enable) {
    std::lock_guard<std::mutex> lock(g_rkva_test_handshake.mutex);
    g_rkva_test_handshake.enabled = enable;
    g_rkva_test_handshake.reached = false;
    g_rkva_test_handshake.released = false;
    g_rkva_test_handshake.cv.notify_all();
}

extern "C" bool ggml_backend_remote_attn_test_handshake_wait_reached(int timeout_ms) {
    std::unique_lock<std::mutex> lock(g_rkva_test_handshake.mutex);
    return g_rkva_test_handshake.cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [] {
        return g_rkva_test_handshake.reached;
    });
}

extern "C" void ggml_backend_remote_attn_test_handshake_release() {
    std::lock_guard<std::mutex> lock(g_rkva_test_handshake.mutex);
    g_rkva_test_handshake.released = true;
    g_rkva_test_handshake.cv.notify_all();
}

size_t ggml_backend_remote_attn_async_in_flight(ggml_backend_t backend) {
    if (!backend || ggml_backend_is_local_split(backend)) return 0;
    auto * ctx = static_cast<ggml_backend_remote_attn_context *>(backend->context);
    std::lock_guard<std::mutex> alock(ctx->async_mutex);
    size_t count = 0;
    for (const auto & s : ctx->async_slots) {
        if (s.state != rkva_slot_state::FREE) {
            count++;
        }
    }
    return count;
}

size_t ggml_backend_remote_attn_async_capacity(ggml_backend_t backend) {
    if (!backend || ggml_backend_is_local_split(backend)) return 0;
    return static_cast<ggml_backend_remote_attn_context *>(backend->context)->async_capacity;
}

bool ggml_backend_remote_attn_native_kvarn_can_rewind(ggml_backend_t backend) {
    if (!ggml_backend_remote_attn_native_kvarn_enabled(backend)) return false;
    const auto * ctx = static_cast<const ggml_backend_remote_attn_context *>(backend->context);
    if (ctx->native_q4 || ctx->native_q8) {
        // Layouts 4/5 rows live in the standard Q4_0/Q8_0 range buffer; op8
        // trims the suffix exactly, without the KVarN resident-tail contract.
        return ctx->connected && !ctx->failed &&
               (ctx->ack.supported_ops & (1u << RKVA_REWIND));
    }
    return ctx->connected && !ctx->failed && !ctx->native_q4 && !ctx->native_q8 &&
           rkva_kvarn_generation_mutable(ctx->native_generation) &&
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

void ggml_remote_attn_range_set_cohort(ggml_tensor * node, uint32_t count, uint32_t index) {
    GGML_ASSERT(count >= 1 && count <= 4 && index < count);
    ggml_remote_attn_range_set_async(node);
    const uint32_t local = uint32_t(ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_COHORT)) &
                           ~uint32_t(GGML_REMOTE_ATTN_COHORT_WIRE_MASK);
    ggml_set_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_COHORT, int32_t(local | count | (index << 8)));
}

void ggml_remote_attn_range_set_wavefront(ggml_tensor * node) {
    ggml_remote_attn_range_set_async(node);
    const uint32_t metadata = uint32_t(ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_COHORT));
    ggml_set_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_COHORT,
                          int32_t(metadata | GGML_REMOTE_ATTN_WAVEFRONT_SHARED_CUDA));
}

ggml_tensor * ggml_remote_attn_range_wait(
        ggml_context * ctx, ggml_tensor * range) {
    GGML_ASSERT(ctx && range && range->op == GGML_OP_REMOTE_ATTN);
    GGML_ASSERT(ggml_get_op_params_i32(range, GGML_REMOTE_ATTN_PARAM_RANGE));
    GGML_ASSERT(ggml_get_op_params_i32(range, GGML_REMOTE_ATTN_PARAM_ASYNC));
    ggml_tensor * result = ggml_backend_remote_attn_cuda_enabled_for_node(range) ?
        ggml_view_tensor(ctx, range) : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, range->ne[0], range->ne[1]);
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
    bool async_idle = true;
    {
        std::lock_guard<std::mutex> alock(ctx->async_mutex);
        for (const auto & s : ctx->async_slots) {
            if (s.state != rkva_slot_state::FREE) {
                async_idle = false;
                break;
            }
        }
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
    std::lock_guard<std::mutex> slock(ctx->stats_mutex);
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
        ctx->native_q4 ? "q4_0" : ctx->native_q8 ? "q8_0" : ctx->range_native_kvarn ? "kvarn4" : ctx->range_wire == RKVA_WIRE_Q4_0 ? "q4_0" : ctx->range_wire == RKVA_WIRE_Q8_0 ? "q8_0" : "f16",
        ctx->range_native_kvarn ? "projected_f32" : ctx->range_configured ? ggml_type_name(ctx->range_source_type) : "none",
        (unsigned long long) s.range_upload_bytes, ctx->range_native_kvarn ? ctx->native_tail_tokens : 0);
    ctx->stats_json = buf;
    ctx->stats_json.pop_back();
    ctx->stats_json += ",\"wire_format\":\"" + std::string(ctx->range_wire == RKVA_WIRE_FP8 ? "fp8_e5m2" :
        ctx->range_wire == RKVA_WIRE_F16 ? "f16" : "f32") + "\",\"remote_layers\":" + std::to_string(ctx->geo.n_layer_remote) +
        ",\"n_head\":" + std::to_string(ctx->geo.n_head) + ",\"n_head_kv\":" + std::to_string(ctx->geo.n_head_kv) +
        ",\"server_version\":" + std::to_string(ctx->ack.server_version) +
        ",\"compact_stage\":" + std::string(ctx->range_native_kvarn && ctx->native_serial_prefix &&
            (ctx->ack.reserved[0] & RKVA_CAP_KVARN_COMPACT_STAGE) && ctx->range_origin >= 128 &&
            ctx->native_tail_tokens == 128 ? "true" : "false") +
        ",\"cuda_pack_calls\":" + std::to_string(s.cuda_pack_calls) +
        ",\"cuda_unpack_calls\":" + std::to_string(s.cuda_unpack_calls) +
        ",\"cuda_small_pack_calls\":" + std::to_string(s.cuda_small_pack_calls) +
        ",\"cuda_small_unpack_calls\":" + std::to_string(s.cuda_small_unpack_calls) +
        ",\"cohort_calls\":" + std::to_string(s.cohort_calls) +
        ",\"cohort_entries\":" + std::to_string(s.cohort_entries) +
        ",\"cuda_context_creates\":" + std::to_string(s.cuda_context_creates) +
        ",\"cuda_context_phase_releases\":" + std::to_string(s.cuda_context_phase_releases) +
        ",\"cuda_context_shared_acquires\":" + std::to_string(s.cuda_context_shared_acquires) +
        ",\"cuda_context_shared_releases\":" + std::to_string(s.cuda_context_shared_releases) +
        ",\"cuda_context_shared_last_releases\":" + std::to_string(s.cuda_context_shared_last_releases) +
        ",\"q_f32_readback_bytes_avoided\":" + std::to_string(s.q_f32_readback_bytes_avoided) +
        ",\"out_f32_upload_bytes_avoided\":" + std::to_string(s.out_f32_upload_bytes_avoided) + "}";
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
    std::lock_guard<std::mutex> slock(ctx->stats_mutex);
    ctx->stats = rkva_stats {};
}

// ---------------------------------------------------------------------------
// CPU-pinned execution path (PATH B): the op is computed on the CPU backend and
// calls into the active connection here. One active connection per process (MVP).
// ---------------------------------------------------------------------------

static ggml_backend_t g_remote_attn_active = nullptr;

bool ggml_backend_remote_attn_cuda_enabled(void) {
    auto * backend = g_remote_attn_active;
    if (!backend || ggml_backend_is_local_split(backend)) return false;
    const char * option = std::getenv("GGML_RKVA_CUDA_FP8");
    if (option && !std::strcmp(option, "0")) return false;
    auto * ctx = static_cast<ggml_backend_remote_attn_context *>(backend->context);
    return ctx->connected && !ctx->failed && ctx->range_configured &&
        ctx->range_native_kvarn && ctx->native_bulk && ctx->range_wire == RKVA_WIRE_FP8 && rkva_cuda_fp8_api();
}

bool ggml_backend_remote_attn_cuda_enabled_for_node(const ggml_tensor * node) {
    if (!node || node->op != GGML_OP_REMOTE_ATTN) return false;
    const auto * range = ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_WAIT) ? node->src[0] : node;
    if (!range || range->op != GGML_OP_REMOTE_ATTN ||
            !ggml_get_op_params_i32(range, GGML_REMOTE_ATTN_PARAM_RANGE) || range->ne[1] <= 0) return false;
    // This is the native serial-prefix protocol bucket, not a measured or
    // adaptive threshold. A prefill segment of eight queries still belongs to
    // its original whole range, e.g.256, for this decision.
    if (!rkva_cuda_fp8_bucket_permitted(range->ne[1])) return false;
    return ggml_backend_remote_attn_cuda_enabled();
}

bool ggml_remote_attn_exec_cuda(struct ggml_tensor * node, void * producer_event) {
    struct restore_event {
        void * previous;
        ~restore_event() { rkva_cuda_producer_event = previous; }
    } restore {rkva_cuda_producer_event};
    rkva_cuda_producer_event = producer_event;
    return ggml_remote_attn_exec(node);
}

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
    const bool is_async = ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_ASYNC) != 0;
    auto lock = is_async ?
        std::unique_lock<std::mutex>(ctx->mutex) : rkva_lock_ctx_drained(ctx);
    if (ctx->failed || !ctx->connected) {
        GGML_LOG_ERROR("%s: exec while not connected (failed=%d)\n",
                       GGML_REMOTE_ATTN_NAME, (int) ctx->failed);
        return false;
    }
    if (!rkva_run_attn(ctx, node, lock)) {
        rkva_poison_remote_state_locked(ctx);
        if (!is_async) {
            rkva_destroy_poisoned_state_locked(ctx);
        }
        ctx->async_cv.notify_all();
        return false;
    }
    if (ggml_remote_attn_trace_enabled()) {
        auto e = ggml_remote_attn_trace_identity("active_range", node);
        if (!ctx->pos_scratch.empty()) {
            e.first_pos = ctx->pos_scratch.front();
            e.last_pos = ctx->pos_scratch.back();
        }
        e.session = ctx->active_session;
        e.ticket = uint64_t(uint32_t(ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_TICKET_LO))) |
                   (uint64_t(uint32_t(ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_TICKET_HI))) << 32);
        e.epoch = ctx->range_epoch;
        ggml_remote_attn_trace_context(&e);
    }
    return true;
}
