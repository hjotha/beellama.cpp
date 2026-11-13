#include "../ggml/src/ggml-prefill-reference.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

constexpr int64_t S = 128, HK = 16, H = 48, MAX_T = 512, D = S * S * H;

struct inputs {
    std::vector<float> q, k, v, g, b, state;
};

static inputs make_inputs() {
    inputs x;
    x.q.resize(S * HK * MAX_T);
    x.k.resize(x.q.size());
    x.v.resize(S * H * MAX_T);
    x.g.resize(H * MAX_T);
    x.b.resize(H * MAX_T);
    x.state.resize(D);
    for (int64_t t = 0; t < MAX_T; ++t) {
        for (int64_t h = 0; h < HK; ++h) {
            double        sq = 0, sk = 0;
            const int64_t off = (t * HK + h) * S;
            for (int64_t i = 0; i < S; ++i) {
                x.q[off + i] = std::sin(float((i + 1) * (h + 3) + (t + 1) * 7) * 0.013f);
                x.k[off + i] = std::cos(float((i + 2) * (h + 5) + (t + 1) * 3) * 0.011f);
                sq += double(x.q[off + i]) * x.q[off + i];
                sk += double(x.k[off + i]) * x.k[off + i];
            }
            for (int64_t i = 0; i < S; ++i) {
                x.q[off + i] /= float(std::sqrt(sq));
                x.k[off + i] /= float(std::sqrt(sk));
            }
        }
    }
    for (size_t i = 0; i < x.v.size(); ++i) {
        x.v[i] = 0.4f * std::sin(float(i % 10007) * 0.017f);
    }
    for (size_t i = 0; i < x.g.size(); ++i) {
        x.g[i] = -0.01f - 0.0001f * float(i % 37);
        x.b[i] = 0.1f + 0.006f * float(i % 80);
    }
    for (size_t i = 0; i < x.state.size(); ++i) {
        x.state[i] = 0.01f * std::cos(float(i % 10009) * 0.007f);
    }
    return x;
}

static void compare(const char * label, const float * a, const float * b, size_t n, bool exact) {
    double sq = 0, den = 0, max = 0;
    size_t diff = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            throw std::runtime_error("nonfinite");
        }
        double d = double(a[i]) - b[i];
        sq += d * d;
        den += double(b[i]) * b[i];
        max = std::max(max, std::abs(d));
        diff += std::memcmp(a + i, b + i, sizeof(float)) != 0;
    }
    const double nmse = sq / std::max(den, 1e-300);
    std::printf("%s different=%zu/%zu max_abs=%.9g nmse=%.9g\n", label, diff, n, max, nmse);
    std::fflush(stdout);
    if ((exact && diff) || (!exact && nmse > 1e-5)) {
        throw std::runtime_error("reference mismatch");
    }
}

static std::vector<float> run_slice(ggml_backend_t             backend,
                                    const inputs &             x,
                                    int64_t                    offset,
                                    int64_t                    T,
                                    int64_t                    K,
                                    const std::vector<float> & initial,
                                    int                        plan) {
    if (offset < 0 || offset + T > MAX_T || initial.size() != size_t(D)) {
        throw std::runtime_error("slice");
    }
    auto ctx = ggml_init({ 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true });
    if (!ctx) {
        throw std::runtime_error("context");
    }
    auto q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HK, T, 1);
    auto k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HK, T, 1);
    auto v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, 1);
    auto g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, 1);
    auto b     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, 1);
    auto state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, 1);
    auto y     = ggml_gated_delta_net(ctx, q, k, v, g, b, state, K);
    ggml_prefill_reference_set(y, static_cast<ggml_prefill_reference_plan>(plan));
    auto graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, y);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        throw std::runtime_error("buffer");
    }
    auto put = [](ggml_tensor * t, const float * a) {
        ggml_backend_tensor_set(t, a, 0, ggml_nbytes(t));
    };
    put(q, x.q.data() + offset * S * HK);
    put(k, x.k.data() + offset * S * HK);
    put(v, x.v.data() + offset * S * H);
    put(g, x.g.data() + offset * H);
    put(b, x.b.data() + offset * H);
    put(state, initial.data());
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("graph");
    }
    ggml_backend_synchronize(backend);
    std::vector<float> ydata(ggml_nelements(y));
    ggml_backend_tensor_get(y, ydata.data(), 0, ggml_nbytes(y));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ydata;
}

int main() {
    try {
#if defined(_WIN32)
        _putenv_s("GGML_CUDA_GDN_PREFIX_MMA", "2");
#else
        setenv("GGML_CUDA_GDN_PREFIX_MMA", "2", 1);
#endif
        ggml_backend_load_all();
        auto dev = ggml_backend_dev_by_name("CUDA0");
        if (!dev) {
            std::puts("SKIP: CUDA0 unavailable");
            return 0;
        }
        auto reg            = ggml_backend_dev_backend_reg(dev);
        using capability_fn = bool (*)(ggml_backend_dev_t);
        auto cap = reg ? reinterpret_cast<capability_fn>(
                             ggml_backend_reg_get_proc_address(reg, "ggml_backend_prefill_reference_supported")) :
                         nullptr;
        if (!cap || !cap(dev)) {
            std::puts("SKIP: reference prefill unsupported on this device");
            return 0;
        }
        auto gpu = ggml_backend_dev_init(dev, nullptr);
        auto cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!gpu || !cpu) {
            throw std::runtime_error("backend");
        }
        auto               x        = make_inputs();
        auto               prefix   = run_slice(gpu, x, 0, 128, 1, x.state, 0);
        const size_t       score128 = size_t(S * H * 128), score256 = size_t(S * H * 256);
        std::vector<float> state128(prefix.begin() + score128, prefix.begin() + score128 + D);
        auto               cold0 = run_slice(gpu, x, 0, 256, 7, x.state, 0);
        std::vector<float> state256(cold0.begin() + score256, cold0.begin() + score256 + D);
        auto               cold1   = run_slice(gpu, x, 256, 256, 7, state256, 0);
        auto               shifted = run_slice(gpu, x, 128, 256, 7, state128, 4);
        compare("shift_first_vs_cold_suffix", shifted.data(), cold0.data() + score128, score128, true);
        compare("shift_second_vs_cold_prefix", shifted.data() + score128, cold1.data(), score128, true);
        auto cpu_shift = run_slice(cpu, x, 128, 256, 7, state128, 0);
        compare("shift_scores_vs_cpu", shifted.data(), cpu_shift.data(), score256, false);
        for (int j = 0; j < 7; ++j) {
            auto         ref     = run_slice(gpu, x, 256, 128 - j, 1, state256, 0);
            const size_t ref_off = size_t(S * H * (128 - j));
            const size_t off     = score256 + size_t(j) * D;
            char         name[64];
            std::snprintf(name, sizeof(name), "shift_snapshot%d_vs_mma_prefix", j);
            compare(name, shifted.data() + off, ref.data() + ref_off, D, true);
            std::snprintf(name, sizeof(name), "shift_snapshot%d_vs_cpu", j);
            compare(name, shifted.data() + off, cpu_shift.data() + off, D, false);
        }
        auto first = run_slice(gpu, x, 0, 128, 7, x.state, 2);
        compare("aligned_first_scores", first.data(), cold0.data(), score128, true);
        for (int j = 0; j < 7; ++j) {
            auto ref = run_slice(gpu, x, 0, 128 - j, 1, x.state, 0);
            char name[64];
            std::snprintf(name, sizeof(name), "aligned_first_snapshot%d", j);
            compare(name, first.data() + score128 + size_t(j) * D, ref.data() + size_t(S * H * (128 - j)), D, true);
        }
        auto second = run_slice(gpu, x, 128, 128, 7, state128, 3);
        compare("aligned_second_scores", second.data(), cold0.data() + score128, score128, true);
        for (int j = 0; j < 7; ++j) {
            char name[64];
            std::snprintf(name, sizeof(name), "aligned_second_snapshot%d", j);
            compare(name, second.data() + score128 + size_t(j) * D, cold0.data() + score256 + size_t(j) * D, D, true);
        }
        ggml_backend_free(cpu);
        ggml_backend_free(gpu);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
