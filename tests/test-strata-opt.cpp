#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-cuda/moe-upload-ring.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static ggml_context_ptr make_context() {
    return ggml_context_ptr(ggml_init({ 65536, nullptr, true }));
}

static void test_indexer(ggml_backend_t backend, int pools, int queries) {
    constexpr int dim = 128;
    constexpr int streams = 2;
    auto ctx = make_context();
    auto q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, dim, 4, queries, streams);
    auto k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_Q8_0, dim, 1, pools, streams);
    auto w = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 4, queries, 1, streams);
    auto mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, pools, queries, 1, streams);
    auto bounds = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_I32, 2, queries, 1, streams);
    auto dense = ggml_lightning_indexer(ctx.get(), q, k, w, mask);
    auto compact = ggml_lightning_indexer(ctx.get(), q, k, w, bounds);
    auto dense_top = ggml_top_k(ctx.get(), dense, 4);
    auto compact_top = ggml_top_k(ctx.get(), compact, 4);
    auto graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, dense_top);
    ggml_build_forward_expand(graph, compact_top);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer && ggml_backend_supports_op(backend, compact));

    std::vector<float> qdata(ggml_nelements(q), 0.125f);
    std::vector<float> weights(ggml_nelements(w), 0.25f);
    std::vector<float> keys(ggml_nelements(k));
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = std::sin(0.0011f*i)*0.5f + 0.5f;
    }
    std::vector<char> quant(ggml_nbytes(k));
    GGML_ASSERT(ggml_quantize_chunk(k->type, keys.data(), quant.data(), 0, pools*streams, dim, nullptr) == quant.size());

    std::vector<int32_t> ranges(2*queries*streams);
    std::vector<ggml_fp16_t> full(pools*queries*streams, ggml_fp32_to_fp16(-INFINITY));
    for (int s = 0; s < streams; ++s) {
        for (int t = 0; t < queries; ++t) {
            const int i = s*queries + t;
            const int first = s*pools/2;
            const int end = t == 1 ? first : first + 5 + t*9;
            ranges[2*i] = first;
            ranges[2*i + 1] = end;
            for (int p = first; p < end; ++p) {
                full[i*pools + p] = ggml_fp32_to_fp16(0.0f);
            }
        }
    }

    ggml_backend_tensor_set(q, qdata.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(k, quant.data(), 0, quant.size());
    ggml_backend_tensor_set(mask, full.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(bounds, ranges.data(), 0, ggml_nbytes(bounds));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> a(full.size()), b(full.size());
    ggml_backend_tensor_get(dense, a.data(), 0, ggml_nbytes(dense));
    ggml_backend_tensor_get(compact, b.data(), 0, ggml_nbytes(compact));
    for (int i = 0; i < queries*streams; ++i) {
        for (int p = 0; p < pools; ++p) {
            const size_t idx = i*pools + p;
            if (p >= ranges[2*i] && p < ranges[2*i + 1]) {
                GGML_ASSERT(std::isfinite(b[idx]) && a[idx] == b[idx]);
            } else {
                GGML_ASSERT(b[idx] == -INFINITY && b[idx] == a[idx]);
            }
        }
    }
    std::vector<int32_t> ids_a(ggml_nelements(dense_top)), ids_b(ids_a.size());
    ggml_backend_tensor_get(dense_top, ids_a.data(), 0, ggml_nbytes(dense_top));
    ggml_backend_tensor_get(compact_top, ids_b.data(), 0, ggml_nbytes(compact_top));
    GGML_ASSERT(ids_a == ids_b);
    std::printf("QSA pools=%d queries=%d streams=%d mask=%zu bounds=%zu bytes score/IDs equal\n",
            pools, queries, streams, ggml_nbytes(mask), ggml_nbytes(bounds));
}

static void test_gdn_epilogue(ggml_backend_t backend, int tokens) {
    constexpr int dim = 128;
    constexpr int channels = 4*dim;
    constexpr int n_qk = 2*dim;
    constexpr int kernel = 4;
    constexpr int seqs = 2;
    constexpr float eps = 1e-6f;
    auto ctx = make_context();
    auto x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, tokens + kernel - 1, channels, seqs);
    auto w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kernel, channels);
    auto plain = ggml_silu(ctx.get(), ggml_ssm_conv(ctx.get(), x, w));
    auto qk = ggml_view_4d(ctx.get(), plain, dim, 2, tokens, seqs, dim*sizeof(float), plain->nb[1], plain->nb[2], 0);
    auto normalized = ggml_scale(ctx.get(), ggml_rms_norm(ctx.get(), qk, eps/dim), 1.0f/std::sqrt(float(dim)));
    auto fused = ggml_ssm_conv(ctx.get(), x, w);
    ggml_ssm_conv_set_silu_l2(fused, dim, n_qk, eps);
    auto graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, normalized);
    ggml_build_forward_expand(graph, fused);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer && ggml_backend_supports_op(backend, fused));
    std::vector<float> xdata(ggml_nelements(x)), wdata(ggml_nelements(w));
    for (size_t i = 0; i < xdata.size(); ++i) {
        xdata[i] = std::sin(0.031f*i);
    }
    for (size_t i = 0; i < wdata.size(); ++i) {
        wdata[i] = 0.2f*std::cos(0.069f*i);
    }
    // One head per sequence is exactly zero, including its normalization denominator.
    std::fill(wdata.begin(), wdata.begin() + dim*kernel, 0.0f);
    ggml_backend_tensor_set(x, xdata.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(w, wdata.data(), 0, ggml_nbytes(w));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> a(ggml_nelements(plain)), b(a.size()), qk_ref(ggml_nelements(normalized));
    ggml_backend_tensor_get(plain, a.data(), 0, ggml_nbytes(plain));
    ggml_backend_tensor_get(fused, b.data(), 0, ggml_nbytes(fused));
    ggml_backend_tensor_get(normalized, qk_ref.data(), 0, ggml_nbytes(normalized));
    float max_qk = 0.0f, max_v = 0.0f;
    for (int t = 0; t < seqs*tokens; ++t) {
        for (int c = 0; c < channels; ++c) {
            const int i = t*channels + c;
            GGML_ASSERT(std::isfinite(b[i]));
            if (c < n_qk) {
                max_qk = std::max(max_qk, std::abs(qk_ref[t*n_qk + c] - b[i]));
            } else {
                max_v = std::max(max_v, std::abs(a[i] - b[i]));
            }
        }
    }
    GGML_ASSERT(max_qk < 1e-6f && max_v < 1e-6f);
    std::printf("GDN tokens=%d seqs=%d max |delta| Q/K=%g V=%g\n", tokens, seqs, max_qk, max_v);
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    const char * name = argc > 1 ? argv[1] : "CPU";
    ggml_backend_ptr backend(ggml_backend_init_by_name(name, nullptr));
    GGML_ASSERT(backend);
    if (ggml_backend_is_cpu(backend.get())) {
        const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        const auto set_threads = reinterpret_cast<decltype(ggml_backend_cpu_set_n_threads) *>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        GGML_ASSERT(set_threads);
        set_threads(backend.get(), 3);
    }
    ggml_moe_upload_ring ring;
    const size_t mib = 1024*1024;
    GGML_ASSERT(ring.slots*ring.chunk_bytes == 8*mib);
    size_t left = 2*ring.chunk_bytes + 17;
    for (size_t expected : {0, 1, 0}) {
        GGML_ASSERT(ring.take() == expected);
        left -= ring.chunk(left);
    }
    GGML_ASSERT(left == 0 && ring.chunk(0) == 0 && ring.take() == 1);
    for (int pools : {4096/4, 28672/4}) {
        for (int queries : {1, 9}) {
            test_indexer(backend.get(), pools, queries);
        }
    }
    for (int tokens : {1, 32, 33, 65}) {
        test_gdn_epilogue(backend.get(), tokens);
    }
}
