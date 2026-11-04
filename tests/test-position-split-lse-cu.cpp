// F1b — CUDA FA-LSE route state (plan §3.4).
//
// CUDA LSE export is implemented only for the KVarN-native routes
// (decode-split/vec, windowed, portable, tail). The generic FA path must
// decline an LSE request instead of silently producing a wrong buffer, so this
// test pins that contract:
//
//   1. generic Q4_0 and F16 K/V without LSE stay supported,
//   2. the same shapes with an LSE request are declined as a whole
//      (supports_op(FA) == false), which is the fail-closed behavior,
//   3. the GGML_OP_FLASH_ATTN_EXT_LSE side-output node is supported by CUDA
//      and the scheduler co-locates it with an FA node pinned to CUDA, so the
//      KVarN routes can publish their LSE through the same contract.
//
// Values for the KVarN routes are proven by the F3 integration run (4B,
// P-small, per-route logging), which drives real KVarN records.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <random>
#include <vector>

static int g_checks = 0;
static int g_failures = 0;
static const char * g_current = "";

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::printf("  FAIL [%s] line %d: %s\n", g_current, __LINE__, #cond); \
    } \
} while (0)

static const int D = 64, NQ = 1, NKV = 512, NQH = 4, NKH = 2;

// supports_op for the FA op and for the LSE side-output node, with and without
// the LSE request.
static void probe_support(ggml_backend_t backend, ggml_type kv_type, bool with_lse,
        bool * fa_supported, bool * lse_node_supported) {
    struct ggml_init_params params = { 8 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, NQ, NQH, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, kv_type, D, NKV, NKH, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, kv_type, D, NKV, NKH, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, NKV, NQ);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);
    ggml_tensor * lse = with_lse ? ggml_flash_attn_ext_lse_out(ctx, fa) : nullptr;

    *fa_supported = ggml_backend_supports_op(backend, fa);
    *lse_node_supported = lse != nullptr && ggml_backend_supports_op(backend, lse);
    ggml_free(ctx);
}

// The scheduler must place the LSE node on the FA node's backend even when the
// FA node is pinned by the user (the position-split placement).
static void probe_colocation(ggml_backend_t backend, bool * colocated) {
    struct ggml_init_params params = { 8 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, NQ, NQH, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, NKV, NKH, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, NKV, NKH, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, NKV, NQ);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);
    ggml_tensor * lse = ggml_flash_attn_ext_lse_out(ctx, fa);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[2] = { backend, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 2, 256, false, true);
    ggml_backend_sched_set_tensor_backend(sched, fa, backend);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, lse);

    *colocated = false;
    if (ggml_backend_sched_alloc_graph(sched, gf)) {
        *colocated = ggml_backend_sched_get_tensor_backend(sched, fa) == backend &&
            ggml_backend_sched_get_tensor_backend(sched, lse) == backend;
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
}

// Compare both CUDA routes against FP64 attention over the actual SAW bytes,
// after the same F16 materialization that the CUDA FlashAttention path uses.
static void probe_saw_attention(ggml_backend_t backend, int nkv, int nq, bool view, bool empty_row,
                                int nqh = 4, int nkh = 2, bool full_visibility = false, bool tqnc = false) {
    constexpr int d = 256;
    const int stride = view ? 2*nkv : nkv;
    const int offset = view ? 3 : 0;
    const ggml_type type_k = tqnc ? GGML_TYPE_TQNC_K : GGML_TYPE_SAW_INT4;
    const ggml_type type_v = tqnc ? GGML_TYPE_TQNC_V : GGML_TYPE_SAW_INT4;
    const size_t key_bytes = ggml_row_size(type_k, d);
    const size_t value_bytes = ggml_row_size(type_v, d);
    std::mt19937 rng(61743 + nkv + nq);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> query(d*nq*nqh), input(d), kd(d*nkv*nkh), vd(d*nkv*nkh);
    std::vector<uint8_t> keys(stride*nkh*key_bytes), values(stride*nkh*value_bytes);
    std::vector<ggml_fp16_t> mask(nkv*nq);
    for (float & v : query) v = distribution(rng);
    for (int h = 0; h < nkh; ++h) for (int t = 0; t < nkv; ++t) {
        const size_t decoded_offset = size_t(h*nkv + t)*d;
        const auto encode = [&](ggml_type type, std::vector<uint8_t> & bytes,
                                std::vector<float> & decoded, size_t row_bytes) {
            const auto * traits = ggml_get_type_traits(type);
            GGML_ASSERT(traits && traits->to_float);
            const size_t packed_offset = size_t(h*stride + t + offset)*row_bytes;
            for (float & x : input) x = distribution(rng);
            ggml_quantize_chunk(type, input.data(), bytes.data() + packed_offset, 0, 1, d, nullptr);
            traits->to_float(bytes.data() + packed_offset, decoded.data() + decoded_offset, d);
            for (int j = 0; j < d; ++j) {
                float & x = decoded[decoded_offset + j];
                x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
            }
        };
        encode(type_k, keys, kd, key_bytes);
        encode(type_v, values, vd, value_bytes);
    }
    for (int q = 0; q < nq; ++q) for (int t = 0; t < nkv; ++t) {
        const bool masked = (empty_row && q == nq - 1) || (!full_visibility && t > (q + 1)*nkv/nq - 1);
        mask[q*nkv + t] = ggml_fp32_to_fp16(masked ? -INFINITY : 0.0f);
    }

    ggml_init_params params = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nq, nqh, 1);
    ggml_tensor * ks = ggml_new_tensor_4d(ctx, type_k, d, stride, nkh, 1);
    ggml_tensor * vs = ggml_new_tensor_4d(ctx, type_v, d, stride, nkh, 1);
    ggml_tensor * k = view ? ggml_view_4d(ctx, ks, d, nkv, nkh, 1,
            ks->nb[1], ks->nb[2], ks->nb[3], offset*key_bytes) : ks;
    ggml_tensor * v = view ? ggml_view_4d(ctx, vs, d, nkv, nkh, 1,
            vs->nb[1], vs->nb[2], vs->nb[3], offset*value_bytes) : vs;
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nkv, nq);
    ggml_tensor * plain = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/16.0f, 0.0f, 0.0f);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/16.0f, 0.0f, 0.0f);
    ggml_prec_set_acc(plain, GGML_PREC_F32);
    ggml_prec_set_acc(fa, GGML_PREC_F32);
    ggml_tensor * lse = ggml_flash_attn_ext_lse_out(ctx, fa);
    const bool supported = ggml_backend_supports_op(backend, plain) &&
        ggml_backend_supports_op(backend, fa) && ggml_backend_supports_op(backend, lse);
    CHECK(supported);
    if (!supported) { ggml_free(ctx); return; }

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    CHECK(buf != nullptr);
    if (!buf) { ggml_free(ctx); return; }
    ggml_backend_tensor_set(q, query.data(), 0, query.size()*sizeof(float));
    ggml_backend_tensor_set(ks, keys.data(), 0, keys.size());
    ggml_backend_tensor_set(vs, values.data(), 0, values.size());
    ggml_backend_tensor_set(m, mask.data(), 0, mask.size()*sizeof(mask[0]));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, plain);
    ggml_build_forward_expand(graph, lse);
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    CHECK(computed);
    if (!computed) { ggml_free(ctx); return; }
    std::vector<float> out(d*nq*nqh), out_plain(out.size()), lse_values(nq*nqh);
    ggml_backend_tensor_get(fa, out.data(), 0, out.size()*sizeof(float));
    ggml_backend_tensor_get(plain, out_plain.data(), 0, out_plain.size()*sizeof(float));
    ggml_backend_tensor_get(lse, lse_values.data(), 0, lse_values.size()*sizeof(float));

    double max_o = 0, max_lse = 0, max_plain_delta = 0;
    bool finite = true, empty_ok = true;
    for (int qi = 0; qi < nq; ++qi) for (int h = 0; h < nqh; ++h) {
        const bool empty = empty_row && qi == nq - 1;
        const int kh = h/(nqh/nkh);
        std::vector<double> scores(nkv, -INFINITY);
        double mx = -INFINITY;
        for (int t = 0; t < nkv; ++t) {
            if (empty || (!full_visibility && t > (qi + 1)*nkv/nq - 1)) continue;
            double dot = 0;
            for (int j = 0; j < d; ++j) dot += double(query[(h*nq + qi)*d + j])*kd[(kh*nkv + t)*d + j];
            scores[t] = dot/16.0;
            mx = std::max(mx, scores[t]);
        }
        std::vector<double> weights(nkv, 0.0);
        double denom = 0;
        if (!empty) for (int t = 0; t < nkv; ++t) {
            weights[t] = std::exp(scores[t] - mx);
            denom += weights[t];
        }
        if (!empty) for (double & weight : weights) weight /= denom;
        const float lse_gpu = lse_values[qi*nqh + h];
        if (empty) empty_ok &= std::isinf(lse_gpu) && lse_gpu < 0;
        else if (!std::isfinite(lse_gpu)) finite = false;
        else max_lse = std::max(max_lse, std::abs(double(lse_gpu) - (mx + std::log(denom))));
        for (int j = 0; j < d; ++j) {
            double ref = 0;
            if (!empty) for (int t = 0; t < nkv; ++t) ref += weights[t]*vd[(kh*nkv+t)*d+j];
            const size_t idx = size_t(qi*nqh + h)*d + j;
            finite &= std::isfinite(out[idx]) && std::isfinite(out_plain[idx]);
            if (empty) empty_ok &= out[idx] == 0.0f && out_plain[idx] == 0.0f;
            max_o = std::max(max_o, std::abs(double(out[idx]) - ref));
            max_plain_delta = std::max(max_plain_delta, std::abs(double(out[idx]) - out_plain[idx]));
        }
    }
    std::printf("  %s D=256 nkv=%d nq=%d heads=%d/%d view=%d empty=%d full=%d max_O=%g max_LSE=%g plain_delta=%g\n",
            tqnc ? "TQNC" : "SAW", nkv, nq, nqh, nkh, int(view), int(empty_row), int(full_visibility), max_o, max_lse, max_plain_delta);
    CHECK(finite && empty_ok && max_o < 0.01 && max_lse < 0.01 && max_plain_delta < 1e-6);
    ggml_free(ctx);
}

// Q4 SET_ROWS must honor source/destination views and I32/I64 row indices.
// Binary-exact inputs avoid conflating a row mapping bug with rounding ties.
template<typename I>
static void probe_q4_set_rows(ggml_backend_t backend, ggml_type index_type) {
    constexpr int width = 256*4, src_rows = 9, dst_rows = 11, n_write = 4;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, width);
    ggml_init_params params = { 2*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * src_base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, src_rows);
    ggml_tensor * src = ggml_view_2d(ctx, src_base, width, n_write,
            2*src_base->nb[1], src_base->nb[1]);
    ggml_tensor * idx_base = ggml_new_tensor_1d(ctx, index_type, 6);
    ggml_tensor * idx = ggml_view_1d(ctx, idx_base, n_write, sizeof(I));
    ggml_tensor * dst_base = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, width, dst_rows);
    ggml_tensor * dst = ggml_view_2d(ctx, dst_base, width, 7, dst_base->nb[1], 2*row_bytes);
    ggml_tensor * store = ggml_set_rows(ctx, dst, src, idx);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, store);
    CHECK(ggml_backend_supports_op(backend, store));
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    CHECK(buf != nullptr);
    if (!buf) { ggml_free(ctx); return; }

    std::vector<float> input(width*src_rows);
    for (int r = 0; r < src_rows; ++r) for (int d = 0; d < width; ++d) {
        // The first value of every 32-element block is the unique max abs.
        input[r*width + d] = d % 32 == 0 ? -1.0f : float((d*17 + r*3) % 15 - 7)/8.0f;
    }
    const I indices[] = {0, 2, 5, 1, 6, 0};
    std::vector<uint8_t> initial(row_bytes*dst_rows, 0x5a), expected = initial, got(initial.size());
    for (int r = 0; r < n_write; ++r) {
        const int dst_row = 2 + int(indices[r + 1]);
        const int src_row = 1 + 2*r;
        CHECK(ggml_quantize_chunk(GGML_TYPE_Q4_0, input.data() + src_row*width,
                expected.data() + dst_row*row_bytes, 0, 1, width, nullptr) == row_bytes);
    }
    ggml_backend_tensor_set(src_base, input.data(), 0, input.size()*sizeof(float));
    ggml_backend_tensor_set(idx_base, indices, 0, sizeof(indices));
    ggml_backend_tensor_set(dst_base, initial.data(), 0, initial.size());
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    CHECK(ok);
    if (ok) {
        ggml_backend_tensor_get(dst_base, got.data(), 0, got.size());
        const size_t diff = std::mismatch(got.begin(), got.end(), expected.begin()).first - got.begin();
        std::printf("  Q4 SET_ROWS %s: bytes=%zu first_difference=%zu\n",
                    ggml_type_name(index_type), got.size(), diff);
        CHECK(got == expected);
    }
    ggml_free(ctx);
}

// Q4 split-prefix attention over the exact stored bytes, comparing O and LSE
// with an FP64 oracle. Interleaved views match the model's token/head layout.
static void probe_q4_lse_attention(ggml_backend_t backend, int nq, bool view, bool empty_row) {
    constexpr int d = 256, nkv = 256, nqh = 24, nkh = 4;
    const int stride = view ? nkv + 8 : nkv;
    const int offset = view ? 3 : 0;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, d);
    std::mt19937 rng(74391 + nq);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> query(d*nq*nqh), input(d), kd(d*nkv*nkh), vd(d*nkv*nkh);
    std::vector<uint8_t> keys(stride*nkh*row_bytes), values(keys.size());
    std::vector<ggml_fp16_t> mask(nkv*nq);
    for (float & x : query) x = distribution(rng);
    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q4_0);
    for (int h = 0; h < nkh; ++h) for (int t = 0; t < nkv; ++t) {
        const size_t decoded_offset = size_t(h*nkv + t)*d;
        const size_t physical_row = view ? size_t((t + offset)*nkh + h) : size_t(h*stride + t);
        const auto encode = [&](std::vector<uint8_t> & bytes, std::vector<float> & decoded) {
            for (float & x : input) x = distribution(rng);
            CHECK(ggml_quantize_chunk(GGML_TYPE_Q4_0, input.data(), bytes.data() + physical_row*row_bytes,
                                      0, 1, d, nullptr) == row_bytes);
            traits->to_float(bytes.data() + physical_row*row_bytes, decoded.data() + decoded_offset, d);
            for (int j = 0; j < d; ++j) {
                float & x = decoded[decoded_offset + j];
                x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
            }
        };
        encode(keys, kd);
        encode(values, vd);
    }
    for (int qi = 0; qi < nq; ++qi) for (int t = 0; t < nkv; ++t) {
        const bool masked = (empty_row && qi == nq - 1) || t >= (qi + 1)*nkv/nq;
        mask[qi*nkv + t] = ggml_fp32_to_fp16(masked ? -INFINITY : 0.0f);
    }

    ggml_init_params params = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nq, nqh, 1);
    ggml_tensor * ks = view ? ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, d, nkh, stride, 1) :
                             ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, d, stride, nkh, 1);
    ggml_tensor * vs = view ? ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, d, nkh, stride, 1) :
                             ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, d, stride, nkh, 1);
    ggml_tensor * k = ks;
    ggml_tensor * v = vs;
    if (view) {
        k = ggml_permute(ctx, ks, 0, 2, 1, 3);
        v = ggml_permute(ctx, vs, 0, 2, 1, 3);
        k = ggml_view_4d(ctx, k, d, nkv, nkh, 1, k->nb[1], k->nb[2], k->nb[3], offset*nkh*row_bytes);
        v = ggml_view_4d(ctx, v, d, nkv, nkh, 1, v->nb[1], v->nb[2], v->nb[3], offset*nkh*row_bytes);
    }
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nkv, nq);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/16.0f, 0.0f, 0.0f);
    ggml_prec_set_acc(fa, GGML_PREC_F32);
    ggml_tensor * lse = ggml_flash_attn_ext_lse_out(ctx, fa);
    const bool supported = ggml_backend_supports_op(backend, fa) && ggml_backend_supports_op(backend, lse);
    CHECK(supported);
    if (!supported) { ggml_free(ctx); return; }
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    CHECK(buf != nullptr);
    if (!buf) { ggml_free(ctx); return; }
    ggml_backend_tensor_set(q, query.data(), 0, query.size()*sizeof(float));
    ggml_backend_tensor_set(ks, keys.data(), 0, keys.size());
    ggml_backend_tensor_set(vs, values.data(), 0, values.size());
    ggml_backend_tensor_set(m, mask.data(), 0, mask.size()*sizeof(mask[0]));
    std::vector<float> lse_values(nq*nqh, -123.0f);
    ggml_backend_tensor_set(lse, lse_values.data(), 0, lse_values.size()*sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, lse);
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    CHECK(computed);
    if (!computed) { ggml_free(ctx); return; }
    std::vector<float> out(d*nq*nqh);
    ggml_backend_tensor_get(fa, out.data(), 0, out.size()*sizeof(float));
    ggml_backend_tensor_get(lse, lse_values.data(), 0, lse_values.size()*sizeof(float));

    double max_o = 0, max_lse = 0;
    bool finite = true, empty_ok = true;
    for (int qi = 0; qi < nq; ++qi) for (int h = 0; h < nqh; ++h) {
        const bool empty = empty_row && qi == nq - 1;
        const int kh = h/(nqh/nkh);
        std::vector<double> scores(nkv, -INFINITY), weights(nkv, 0.0);
        double mx = -INFINITY, denom = 0;
        if (!empty) for (int t = 0; t < (qi + 1)*nkv/nq; ++t) {
            double dot = 0;
            for (int j = 0; j < d; ++j) dot += double(query[(h*nq + qi)*d + j])*kd[(kh*nkv + t)*d + j];
            scores[t] = dot/16.0;
            mx = std::max(mx, scores[t]);
        }
        if (!empty) for (int t = 0; t < nkv; ++t) { weights[t] = std::exp(scores[t] - mx); denom += weights[t]; }
        if (!empty) for (double & weight : weights) weight /= denom;
        const float got_lse = lse_values[qi*nqh + h];
        if (empty) empty_ok &= std::isinf(got_lse) && got_lse < 0;
        else if (!std::isfinite(got_lse)) finite = false;
        else max_lse = std::max(max_lse, std::abs(double(got_lse) - (mx + std::log(denom))));
        for (int j = 0; j < d; ++j) {
            double ref = 0;
            if (!empty) for (int t = 0; t < nkv; ++t) ref += weights[t]*vd[(kh*nkv + t)*d + j];
            const float got = out[(qi*nqh + h)*d + j];
            finite &= std::isfinite(got);
            if (empty) empty_ok &= got == 0.0f;
            max_o = std::max(max_o, std::abs(double(got) - ref));
        }
    }
    std::printf("  Q4 D256 nkv=%d nq=%d GQA=24/4 view=%d empty=%d max_O=%g max_LSE=%g\n",
                nkv, nq, int(view), int(empty_row), max_o, max_lse);
    CHECK(finite && empty_ok && max_o < 0.01 && max_lse < 0.01);
    ggml_free(ctx);
}

static void probe_q4_lse_refusal(ggml_backend_t backend) {
    constexpr int d = 256, nkv = 256, nqh = 24, nkh = 4;
    ggml_init_params params = { 8*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, 1, nqh, 1);
    ggml_tensor * kb = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, d, 2*nkv, nkh, 1);
    ggml_tensor * vb = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, d, 2*nkv, nkh, 1);
    const size_t offset = ggml_type_size(GGML_TYPE_Q4_0);
    ggml_tensor * k = ggml_view_4d(ctx, kb, d, nkv, nkh, 1, kb->nb[1], kb->nb[2], kb->nb[3], offset);
    ggml_tensor * v = ggml_view_4d(ctx, vb, d, nkv, nkh, 1, vb->nb[1], vb->nb[2], vb->nb[3], offset);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nkv, 1);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/16.0f, 0.0f, 0.0f);
    ggml_tensor * lse = ggml_flash_attn_ext_lse_out(ctx, fa);
    CHECK(ggml_backend_supports_op(backend, fa)); // aligned 18-byte offset control
    const size_t old_offset = k->view_offs;
    k->view_offs = 1;
    CHECK(!ggml_backend_supports_op(backend, fa));
    k->view_offs = old_offset;
    const size_t old_stride = k->nb[1];
    k->nb[1] += 1;
    CHECK(!ggml_backend_supports_op(backend, fa));
    k->nb[1] = old_stride;
    const int64_t old_heads = v->ne[2];
    v->ne[2] = nkh - 1;
    CHECK(!ggml_backend_supports_op(backend, fa));
    v->ne[2] = old_heads;
    const size_t old_span_stride = k->nb[1];
    k->nb[1] = kb->nb[2];
    CHECK(!ggml_backend_supports_op(backend, fa)); // stride is aligned, but span escapes the base
    k->nb[1] = old_span_stride;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, lse);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    CHECK(buf != nullptr);
    if (!buf) { ggml_free(ctx); return; }
    ggml_backend_buffer_clear(buf.get(), 0);
    std::vector<float> observed(nqh, -123.0f);
    ggml_backend_tensor_set(lse, observed.data(), 0, observed.size()*sizeof(float));
    mask->ne[0] = nkv - 1;
    CHECK(!ggml_backend_supports_op(backend, fa));
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    ggml_backend_tensor_get(lse, observed.data(), 0, observed.size()*sizeof(float));
    CHECK(std::all_of(observed.begin(), observed.end(), [](float x) { return x == -123.0f; }));
    mask->ne[0] = nkv;
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(lse, observed.data(), 0, observed.size()*sizeof(float));
    CHECK(std::all_of(observed.begin(), observed.end(), [](float x) {
        return std::isfinite(x) && std::abs(x - std::log(256.0f)) < 0.01f;
    }));
    ggml_free(ctx);
}

// FP64 oracle over the real quantized body bytes (decoded to the F16 domain
// consumed by CUDA) and original-domain F16 tail. The compact tail includes
// current rows, so 128+32 stays indexed-small while 256+1/32 uses packed FA.
static void probe_custom_tail_attention(ggml_backend_t backend, bool nc, int tail, int nq,
                                        bool view, bool segmented, bool empty_row) {
    constexpr int d = 256, nkv = 256, nqh = 24, nkh = 4;
    const int tail_stride = tail + nq;
    const int history = segmented ? tail : tail_stride;
    const int body_stride = view ? nkv + 8 : nkv;
    const int body_offset = view ? 3 : 0;
    const ggml_type type_k = nc ? GGML_TYPE_TQNC_K : GGML_TYPE_SAW_INT4;
    const ggml_type type_v = nc ? GGML_TYPE_TQNC_V : GGML_TYPE_SAW_INT4;
    const size_t k_row_bytes = ggml_row_size(type_k, d), v_row_bytes = ggml_row_size(type_v, d);
    std::mt19937 rng(23841 + tail + nq + int(nc));
    std::uniform_real_distribution<float> random(-0.75f, 0.75f);
    std::vector<float> query(d*nq*nqh), input(d), kd(d*nkv*nkh), vd(d*nkv*nkh);
    std::vector<uint8_t> keys(body_stride*nkh*k_row_bytes), values(body_stride*nkh*v_row_bytes);
    for (float & x : query) x = random(rng);
    for (int h = 0; h < nkh; ++h) for (int t = 0; t < nkv; ++t) {
        const size_t logical = size_t(h*nkv + t)*d;
        const size_t physical = view ? size_t((t + body_offset)*nkh + h) : size_t(h*body_stride + t);
        const auto encode = [&](ggml_type type, size_t row_bytes, std::vector<uint8_t> & bytes,
                                std::vector<float> & decoded) {
            for (float & x : input) x = random(rng);
            CHECK(ggml_quantize_chunk(type, input.data(), bytes.data() + physical*row_bytes,
                                      0, 1, d, nullptr) == row_bytes);
            ggml_get_type_traits(type)->to_float(bytes.data() + physical*row_bytes, decoded.data() + logical, d);
            for (int j = 0; j < d; ++j) {
                float & x = decoded[logical + j];
                x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
            }
        };
        encode(type_k, k_row_bytes, keys, kd);
        encode(type_v, v_row_bytes, values, vd);
    }
    std::vector<ggml_fp16_t> tk(d*history*nkh), tv(tk.size());
    std::vector<ggml_fp16_t> ck(segmented ? d*nq*nkh : 0), cv(ck.size());
    for (auto & x : tk) x = ggml_fp32_to_fp16(random(rng));
    for (auto & x : tv) x = ggml_fp32_to_fp16(random(rng));
    for (auto & x : ck) x = ggml_fp32_to_fp16(random(rng));
    for (auto & x : cv) x = ggml_fp32_to_fp16(random(rng));
    std::vector<ggml_fp16_t> bm(nkv*nq), tm(tail_stride*nq);
    for (int qi = 0; qi < nq; ++qi) {
        const bool empty = empty_row && qi == nq - 1;
        for (int t = 0; t < nkv; ++t) {
            bm[qi*nkv + t] = ggml_fp32_to_fp16(empty || t >= (qi + 1)*nkv/nq ? -INFINITY : 0.0f);
        }
        for (int t = 0; t < tail_stride; ++t) {
            tm[qi*tail_stride + t] = ggml_fp32_to_fp16(empty || t >= tail + qi + 1 ? -INFINITY : 0.0f);
        }
    }

    ggml_init_params params = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nq, nqh, 1);
    auto * ks = view ? ggml_new_tensor_4d(ctx, type_k, d, nkh, body_stride, 1) :
                       ggml_new_tensor_4d(ctx, type_k, d, body_stride, nkh, 1);
    auto * vs = view ? ggml_new_tensor_4d(ctx, type_v, d, nkh, body_stride, 1) :
                       ggml_new_tensor_4d(ctx, type_v, d, body_stride, nkh, 1);
    ggml_tensor * k = ks, * v = vs;
    if (view) {
        k = ggml_permute(ctx, ks, 0, 2, 1, 3);
        v = ggml_permute(ctx, vs, 0, 2, 1, 3);
        k = ggml_view_4d(ctx, k, d, nkv, nkh, 1, k->nb[1], k->nb[2], k->nb[3], body_offset*nkh*k_row_bytes);
        v = ggml_view_4d(ctx, v, d, nkv, nkh, 1, v->nb[1], v->nb[2], v->nb[3], body_offset*nkh*v_row_bytes);
    }
    auto * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nkv, nq);
    auto * kt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, history, nkh, 1);
    auto * vt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, history, nkh, 1);
    auto * kc = segmented ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, nq, nkh, 1) : nullptr;
    auto * vc = segmented ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, nq, nkh, 1) : nullptr;
    auto * mt = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, tail_stride, nq);
    auto * qo = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nq, 1);
    auto * rd = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6 + tail_stride + (view ? nkv : 0), 1);
    const auto attach_tail = [&](ggml_tensor * fa) {
        if (segmented) {
            fa = ggml_kv_tail_attention_merge_segmented(ctx, fa, kt, vt, kc, vc, mt, qo, rd);
            ggml_flash_attn_ext_set_kv_tail_history_slots(fa, history);
        } else {
            fa = ggml_kv_tail_attention_merge(ctx, fa, kt, vt, mt, qo, rd);
        }
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        return fa;
    };
    auto * plain = attach_tail(ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/16.0f, 0.0f, 0.0f));
    auto * fa = attach_tail(ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/16.0f, 0.0f, 0.0f));
    auto * lse = ggml_flash_attn_ext_lse_out(ctx, fa);
    const bool supported = ggml_backend_supports_op(backend, plain) &&
                           ggml_backend_supports_op(backend, fa) && ggml_backend_supports_op(backend, lse);
    CHECK(supported);
    if (!supported) { ggml_free(ctx); return; }
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    CHECK(buf != nullptr);
    if (!buf) { ggml_free(ctx); return; }
    std::vector<int32_t> order(nq), desc(6 + tail_stride + (view ? nkv : 0), -1);
    for (int i = 0; i < nq; ++i) order[i] = i;
    desc[4] = tail_stride;
    for (int i = 0; i < tail_stride; ++i) desc[6 + i] = i;
    if (view) { desc[5] = nkv; for (int i = 0; i < nkv; ++i) desc[6 + tail_stride + i] = i; }
    ggml_backend_tensor_set(q, query.data(), 0, query.size()*sizeof(float));
    ggml_backend_tensor_set(ks, keys.data(), 0, keys.size());
    ggml_backend_tensor_set(vs, values.data(), 0, values.size());
    ggml_backend_tensor_set(m, bm.data(), 0, bm.size()*sizeof(bm[0]));
    ggml_backend_tensor_set(kt, tk.data(), 0, tk.size()*sizeof(tk[0]));
    ggml_backend_tensor_set(vt, tv.data(), 0, tv.size()*sizeof(tv[0]));
    if (segmented) {
        ggml_backend_tensor_set(kc, ck.data(), 0, ck.size()*sizeof(ck[0]));
        ggml_backend_tensor_set(vc, cv.data(), 0, cv.size()*sizeof(cv[0]));
    }
    ggml_backend_tensor_set(mt, tm.data(), 0, tm.size()*sizeof(tm[0]));
    ggml_backend_tensor_set(qo, order.data(), 0, order.size()*sizeof(order[0]));
    ggml_backend_tensor_set(rd, desc.data(), 0, desc.size()*sizeof(desc[0]));
    std::vector<float> out(d*nq*nqh), out_plain(out.size()), lse_values(nq*nqh, 123.0f);
    ggml_backend_tensor_set(lse, lse_values.data(), 0, lse_values.size()*sizeof(float));
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, plain);
    ggml_build_forward_expand(graph, lse);
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    CHECK(computed);
    if (!computed) { ggml_free(ctx); return; }
    ggml_backend_tensor_get(fa, out.data(), 0, out.size()*sizeof(float));
    ggml_backend_tensor_get(plain, out_plain.data(), 0, out_plain.size()*sizeof(float));
    ggml_backend_tensor_get(lse, lse_values.data(), 0, lse_values.size()*sizeof(float));

    double max_o = 0, max_lse = 0, max_plain = 0;
    bool finite = true, empty_ok = true;
    for (int qi = 0; qi < nq; ++qi) for (int h = 0; h < nqh; ++h) {
        const bool empty = empty_row && qi == nq - 1;
        const int kh = h/(nqh/nkh);
        std::vector<double> scores(nkv + tail_stride, -INFINITY), weights(scores.size(), 0.0);
        double mx = -INFINITY, denom = 0;
        if (!empty) for (int t = 0; t < nkv + tail_stride; ++t) {
            if ((t < nkv && t >= (qi + 1)*nkv/nq) || (t >= nkv && t - nkv >= tail + qi + 1)) continue;
            double dot = 0;
            for (int j = 0; j < d; ++j) {
                float kval;
                if (t < nkv) kval = kd[(kh*nkv + t)*d + j];
                else {
                    const int ti = t - nkv;
                    const bool current = segmented && ti >= history;
                    kval = current ? ggml_fp16_to_fp32(ck[(kh*nq + ti - history)*d + j]) :
                                     ggml_fp16_to_fp32(tk[(kh*history + ti)*d + j]);
                }
                dot += double(query[(h*nq + qi)*d + j])*kval;
            }
            scores[t] = dot/16.0;
            mx = std::max(mx, scores[t]);
        }
        if (!empty) for (size_t t = 0; t < scores.size(); ++t) { weights[t] = std::exp(scores[t] - mx); denom += weights[t]; }
        if (!empty) for (double & weight : weights) weight /= denom;
        const float got_lse = lse_values[qi*nqh + h];
        if (empty) empty_ok &= std::isinf(got_lse) && got_lse < 0;
        else if (!std::isfinite(got_lse)) finite = false;
        else max_lse = std::max(max_lse, std::abs(double(got_lse) - (mx + std::log(denom))));
        for (int j = 0; j < d; ++j) {
            double ref = 0;
            if (!empty) for (int t = 0; t < nkv + tail_stride; ++t) {
                float value;
                if (t < nkv) value = vd[(kh*nkv + t)*d + j];
                else {
                    const int ti = t - nkv;
                    const bool current = segmented && ti >= history;
                    value = current ? ggml_fp16_to_fp32(cv[(kh*nq + ti - history)*d + j]) :
                                      ggml_fp16_to_fp32(tv[(kh*history + ti)*d + j]);
                }
                ref += weights[t]*value;
            }
            const size_t idx = size_t(qi*nqh + h)*d + j;
            finite &= std::isfinite(out[idx]) && std::isfinite(out_plain[idx]);
            if (empty) empty_ok &= out[idx] == 0.0f && out_plain[idx] == 0.0f;
            max_o = std::max(max_o, std::abs(double(out[idx]) - ref));
            max_plain = std::max(max_plain, std::abs(double(out[idx]) - out_plain[idx]));
        }
    }
    std::printf("  %s tail=%d stride=%d nq=%d view=%d segmented=%d empty=%d maxO=%g maxLSE=%g plain=%g\n",
                nc ? "TQNC" : "SAW", tail, tail_stride, nq, int(view), int(segmented), int(empty_row),
                max_o, max_lse, max_plain);
    CHECK(finite && empty_ok && max_o < 0.01 && max_lse < 0.01 && max_plain < 1e-6);
    ggml_free(ctx);
}

// IQ4_NL currently takes plain FlashAttention after K/V materialization to
// F16. Compare against FP64 attention over the same quantized bytes and the
// same F16 rounding; generic IQ4 LSE export remains unsupported.
static void probe_iq4_attention(ggml_backend_t backend, int nkv, int nq, bool view, bool empty_row) {
    constexpr int d = 256, nqh = 24, nkh = 4;
    const int stride = view ? 2*nkv : nkv;
    const int offset = view ? 3 : 0;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_IQ4_NL, d);
    std::mt19937 rng(71359 + nkv + nq);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> query(d*nq*nqh), input(d), kd(d*nkv*nkh), vd(d*nkv*nkh);
    std::vector<uint8_t> keys(size_t(stride)*nkh*row_bytes), values(keys.size());
    std::vector<ggml_fp16_t> mask(nkv*nq);
    for (float & x : query) x = distribution(rng);
    const auto * traits = ggml_get_type_traits(GGML_TYPE_IQ4_NL);
    CHECK(traits && traits->to_float);
    if (!traits || !traits->to_float) return;
    for (int h = 0; h < nkh; ++h) for (int t = 0; t < nkv; ++t) {
        const size_t packed_offset = size_t(h*stride + t + offset)*row_bytes;
        const size_t decoded_offset = size_t(h*nkv + t)*d;
        for (auto pair : {std::pair<std::vector<uint8_t> *, std::vector<float> *>(&keys, &kd),
                          std::pair<std::vector<uint8_t> *, std::vector<float> *>(&values, &vd)}) {
            for (float & x : input) x = distribution(rng);
            ggml_quantize_chunk(GGML_TYPE_IQ4_NL, input.data(), pair.first->data() + packed_offset, 0, 1, d, nullptr);
            traits->to_float(pair.first->data() + packed_offset, pair.second->data() + decoded_offset, d);
            for (int j = 0; j < d; ++j) {
                float & x = (*pair.second)[decoded_offset + j];
                x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
            }
        }
    }
    for (int qi = 0; qi < nq; ++qi) for (int t = 0; t < nkv; ++t) {
        const bool masked = (empty_row && qi == nq - 1) || t > (qi + 1)*nkv/nq - 1;
        mask[qi*nkv + t] = ggml_fp32_to_fp16(masked ? -INFINITY : 0.0f);
    }

    ggml_init_params params = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nq, nqh, 1);
    ggml_tensor * ks = ggml_new_tensor_4d(ctx, GGML_TYPE_IQ4_NL, d, stride, nkh, 1);
    ggml_tensor * vs = ggml_new_tensor_4d(ctx, GGML_TYPE_IQ4_NL, d, stride, nkh, 1);
    ggml_tensor * k = view ? ggml_view_4d(ctx, ks, d, nkv, nkh, 1,
            ks->nb[1], ks->nb[2], ks->nb[3], offset*row_bytes) : ks;
    ggml_tensor * v = view ? ggml_view_4d(ctx, vs, d, nkv, nkh, 1,
            vs->nb[1], vs->nb[2], vs->nb[3], offset*row_bytes) : vs;
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nkv, nq);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/16.0f, 0.0f, 0.0f);
    ggml_tensor * with_lse = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/16.0f, 0.0f, 0.0f);
    ggml_prec_set_acc(fa, GGML_PREC_F32);
    ggml_flash_attn_ext_lse_out(ctx, with_lse);
    const bool supported = ggml_backend_supports_op(backend, fa);
    CHECK(supported);
    CHECK(!ggml_backend_supports_op(backend, with_lse));
    if (!supported) { ggml_free(ctx); return; }

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    CHECK(buf != nullptr);
    if (!buf) { ggml_free(ctx); return; }
    ggml_backend_tensor_set(q, query.data(), 0, query.size()*sizeof(float));
    ggml_backend_tensor_set(ks, keys.data(), 0, keys.size());
    ggml_backend_tensor_set(vs, values.data(), 0, values.size());
    ggml_backend_tensor_set(m, mask.data(), 0, mask.size()*sizeof(mask[0]));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, fa);
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    CHECK(computed);
    if (!computed) { ggml_free(ctx); return; }
    std::vector<float> out(d*nq*nqh);
    ggml_backend_tensor_get(fa, out.data(), 0, out.size()*sizeof(float));
    double squared = 0.0, reference_squared = 0.0, max_o = 0.0;
    bool finite = true, empty_ok = true;
    for (int qi = 0; qi < nq; ++qi) for (int h = 0; h < nqh; ++h) {
        const bool empty = empty_row && qi == nq - 1;
        const int kh = h/(nqh/nkh);
        std::vector<double> scores(nkv, -INFINITY), weights(nkv, 0.0);
        double mx = -INFINITY;
        for (int t = 0; t < nkv; ++t) {
            if (empty || t > (qi + 1)*nkv/nq - 1) continue;
            double dot = 0.0;
            for (int j = 0; j < d; ++j) dot += double(query[(h*nq + qi)*d + j])*kd[(kh*nkv + t)*d + j];
            scores[t] = dot/16.0;
            mx = std::max(mx, scores[t]);
        }
        double denom = 0.0;
        if (!empty) for (int t = 0; t < nkv; ++t) {
            weights[t] = std::exp(scores[t] - mx);
            denom += weights[t];
        }
        if (!empty) for (double & weight : weights) weight /= denom;
        for (int j = 0; j < d; ++j) {
            double ref = 0.0;
            if (!empty) for (int t = 0; t < nkv; ++t) ref += weights[t]*vd[(kh*nkv+t)*d+j];
            const float actual = out[size_t(qi*nqh + h)*d + j];
            finite &= std::isfinite(actual);
            if (empty) empty_ok &= actual == 0.0f;
            const double delta = double(actual) - ref;
            squared += delta*delta;
            reference_squared += ref*ref;
            max_o = std::max(max_o, std::fabs(delta));
        }
    }
    const double nrmse = std::sqrt(squared/std::max(reference_squared, 1e-30));
    std::printf("  IQ4 D=256 nkv=%d nq=%d view=%d empty=%d max_O=%g nrmse=%g\n",
            nkv, nq, int(view), int(empty_row), max_o, nrmse);
    CHECK(finite && empty_ok && max_o < 0.01 && nrmse < 0.01);
    ggml_free(ctx);
}

// These graphs must be declined before dispatch: conversion requires block
// strides and the shared K/V materialization requires identical addressing.
static void probe_iq4_unsafe_views(ggml_backend_t backend) {
    constexpr int d = 256, nkv = 256, nqh = 24, nkh = 4, batches = 2;
    ggml_init_params params = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    const size_t row = ggml_row_size(GGML_TYPE_IQ4_NL, d);
    CHECK(ggml_type_size(GGML_TYPE_IQ4_NL) == 18 && row == 144);
    const size_t head = row*(2*nkv + 2);
    const size_t batch = head*nkh;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, 1, nqh, batches);
    ggml_tensor * kb = ggml_new_tensor_4d(ctx, GGML_TYPE_IQ4_NL, d, 4*nkv + 32, nkh, batches);
    ggml_tensor * vb = ggml_new_tensor_4d(ctx, GGML_TYPE_IQ4_NL, d, 4*nkv + 32, nkh, batches);
    const auto make_view = [&](ggml_tensor * base, size_t s1, size_t s2, size_t s3) {
        return ggml_view_4d(ctx, base, d, nkv, nkh, batches, s1, s2, s3, 0);
    };
    ggml_tensor * k = make_view(kb, row, head, batch);
    ggml_tensor * v = make_view(vb, row, head, batch);
    const auto supports = [&](ggml_tensor * key, ggml_tensor * value) {
        return ggml_backend_supports_op(backend,
                ggml_flash_attn_ext(ctx, q, key, value, nullptr, 1.0f/16.0f, 0.0f, 0.0f));
    };
    CHECK(supports(k, v)); // the padded but block-aligned control is valid
    for (int axis = 1; axis <= 3; ++axis) {
        size_t strides[] = {row, head, batch};
        strides[axis - 1] += 2;
        ggml_tensor * bad_k = make_view(kb, strides[0], strides[1], strides[2]);
        ggml_tensor * bad_v = make_view(vb, strides[0], strides[1], strides[2]);
        CHECK(!supports(bad_k, v));
        CHECK(!supports(k, bad_v));
    }
    ggml_tensor * q4 = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, d, nkv, nkh, batches);
    CHECK(!supports(k, q4));
    CHECK(!supports(q4, v));

    // 192/128 is an admitted asymmetric shape at GQA8. The V view begins one
    // 18-byte block into K; it must never be reused as K's unshifted F16 data.
    ggml_tensor * aq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 192, 1, 8, 1);
    ggml_tensor * ak_owned = ggml_new_tensor_4d(ctx, GGML_TYPE_IQ4_NL, 192, nkv, 1, 1);
    ggml_tensor * av_offset = ggml_view_4d(ctx, ak_owned, 128, nkv, 1, 1,
            ak_owned->nb[1], ak_owned->nb[2], ak_owned->nb[3], ggml_type_size(GGML_TYPE_IQ4_NL));
    ggml_tensor * ak_base = ggml_new_tensor_4d(ctx, GGML_TYPE_IQ4_NL, 192, 2*nkv, 1, 1);
    ggml_tensor * ak = ggml_view_4d(ctx, ak_base, 192, nkv, 1, 1,
            ak_base->nb[1], ak_base->nb[2], ak_base->nb[3], 0);
    ggml_tensor * av_stride = ggml_view_4d(ctx, ak, 128, nkv, 1, 1,
            ak->nb[1] + ggml_type_size(GGML_TYPE_IQ4_NL), ak->nb[2], ak->nb[3], 0);
    ggml_tensor * am = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nkv, 1);
    CHECK(!ggml_backend_supports_op(backend,
            ggml_flash_attn_ext(ctx, aq, ak_owned, av_offset, am, 1.0f/std::sqrt(192.0f), 0.0f, 0.0f)));
    CHECK(!ggml_backend_supports_op(backend,
            ggml_flash_attn_ext(ctx, aq, ak, av_stride, am, 1.0f/std::sqrt(192.0f), 0.0f, 0.0f)));
    ggml_free(ctx);
}

int main() {
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (!cuda) {
        std::printf("position-split-lse-cu: no CUDA backend\n");
        return 1;
    }

    for (ggml_type kv_type : { GGML_TYPE_Q4_0, GGML_TYPE_F16 }) {
        g_current = ggml_type_name(kv_type);
        bool fa_sup = false, lse_node_sup = false;

        // 1. plain FA keeps working
        probe_support(cuda, kv_type, false, &fa_sup, &lse_node_sup);
        CHECK(fa_sup);

        // 2. an LSE request on the generic route is declined as a whole
        probe_support(cuda, kv_type, true, &fa_sup, &lse_node_sup);
        CHECK(!fa_sup);

        // 3. the side-output node itself is supported by CUDA
        CHECK(lse_node_sup);
    }

    g_current = "lse_colocation";
    {
        bool colocated = false;
        probe_colocation(cuda, &colocated);
        CHECK(colocated);
    }

    g_current = "q4_set_rows";
    probe_q4_set_rows<int32_t>(cuda, GGML_TYPE_I32);
    probe_q4_set_rows<int64_t>(cuda, GGML_TYPE_I64);
    g_current = "q4_attention_lse";
    probe_q4_lse_attention(cuda, 1, false, false);
    probe_q4_lse_attention(cuda, 1, false, true);
    probe_q4_lse_attention(cuda, 2, true, false);
    probe_q4_lse_attention(cuda, 8, true, true);
    probe_q4_lse_attention(cuda, 32, true, false);
    probe_q4_lse_attention(cuda, 256, true, false);
    g_current = "q4_lse_refusal";
    probe_q4_lse_refusal(cuda);

    for (bool nc : { false, true }) {
        g_current = nc ? "turboquant_nc_tail_attention" : "saw_tail_attention";
        for (int tail : { 128, 256 }) {
            probe_custom_tail_attention(cuda, nc, tail, 1, false, false, false);
            probe_custom_tail_attention(cuda, nc, tail, 2, true, true, false);
            probe_custom_tail_attention(cuda, nc, tail, 32, true, true, true);
        }
    }

    g_current = "saw_attention_lse";
    probe_saw_attention(cuda, 256, 1, false, false);
    probe_saw_attention(cuda, 512, 2, true, false);
    probe_saw_attention(cuda, 512, 7, true, true);
    probe_saw_attention(cuda, 256, 64, false, false);
    probe_saw_attention(cuda, 256, 1, true, false, 24, 4);
    probe_saw_attention(cuda, 512, 8, true, false, 24, 4, true);
    probe_saw_attention(cuda, 256, 256, true, false, 24, 4);
    probe_saw_attention(cuda, 1024, 1, true, false, 24, 4);
    probe_saw_attention(cuda, 20480, 1, true, false, 24, 4, true);
    g_current = "turboquant_nc_attention_lse";
    probe_saw_attention(cuda, 256, 1, false, false, 24, 4, true, true);
    probe_saw_attention(cuda, 512, 8, true, false, 24, 4, true, true);
    probe_saw_attention(cuda, 256, 256, true, false, 24, 4, false, true);
    probe_saw_attention(cuda, 20480, 1, true, false, 24, 4, true, true);

    g_current = "iq4_attention";
    probe_iq4_attention(cuda, 256, 1, false, false);
    probe_iq4_attention(cuda, 256, 7, true, true);
    probe_iq4_attention(cuda, 512, 32, true, false);
    g_current = "iq4_unsafe_views";
    probe_iq4_unsafe_views(cuda);

    ggml_backend_free(cuda);
    std::printf("position-split-lse-cu: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
