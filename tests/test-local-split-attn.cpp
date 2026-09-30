// Native attention numerical regression; the legacy zero-output stub must reject execution.
#include "ggml-local-split.h"
#include "ggml-backend.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "test-local-split-attn FAIL: %s\n", message);
        std::exit(1);
    }
}
static constexpr int dim = 256, hq = 24, hkv = 4;

static void numerical_case(ggml_backend_t backend, int nk, int nq) {
    ggml_init_params params = {4 * 1024 * 1024, nullptr, true};
    auto * ctx = ggml_init(params);
    require(ctx != nullptr, "context initialization");
    // Inputs [dimension,tokens,heads,streams]; output [dimension,heads,tokens,streams].
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, nq, hq, 1);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dim, nk, hkv, 1);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dim, nk, hkv, 1);
    auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nk, nq, 1, 1);
    const float scale = 1.0f / std::sqrt(float(dim));
    auto * out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    require(ggml_backend_supports_op(backend, out), "native FLASH_ATTN_EXT unsupported");
    auto * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "backend tensor allocation");
    std::vector<float> qd(dim * nq * hq);
    std::vector<ggml_fp16_t> kd(dim * nk * hkv), vd(kd.size()), md(nk * nq);
    for (int h = 0; h < hq; ++h) {
        for (int t = 0; t < nq; ++t) {
            for (int d = 0; d < dim; ++d) {
                qd[(h * nq + t) * dim + d] = 0.7f * std::sin((1 + d + 11*h + 17*t) * 0.073f);
            }
        }
    }
    for (int h = 0; h < hkv; ++h) {
        for (int t = 0; t < nk; ++t) {
            for (int d = 0; d < dim; ++d) {
                const int i = (h * nk + t) * dim + d;
                kd[i] = ggml_fp32_to_fp16(0.6f * std::cos((3 + d + 19*h + 23*t) * 0.061f));
                // Nonzero head/token-dependent V catches zeros, GQA mapping and missing masking.
                vd[i] = ggml_fp32_to_fp16(0.4f + 0.2f*h + 0.15f*t + 0.3f * std::sin(d * 0.037f));
            }
        }
    }
    for (int qt = 0; qt < nq; ++qt) {
        for (int kt = 0; kt < nk; ++kt) {
            md[qt * nk + kt] = ggml_fp32_to_fp16(
                kt <= nk - nq + qt ? 0.0f : -std::numeric_limits<float>::infinity());
        }
    }
    ggml_backend_tensor_set(q, qd.data(), 0, qd.size() * sizeof(float));
    ggml_backend_tensor_set(k, kd.data(), 0, kd.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v, vd.data(), 0, vd.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(ggml_fp16_t));
    require(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS, "native attention compute");
    std::vector<float> actual(dim * nq * hq);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    // Independent double-precision causal GQA oracle over the rounded uploaded F16 inputs.
    // One visible token must return that head's V regardless of nontrivial Q and K.
    double max_error = 0.0, sum_error_sq = 0.0;
    for (int qt = 0; qt < nq; ++qt) {
        const int visible = nk - nq + qt + 1;
        for (int h = 0; h < hq; ++h) {
            const int kh = h / (hq / hkv);
            std::vector<double> weights(visible);
            for (int kt = 0; kt < visible; ++kt) {
                double dot = 0.0;
                for (int d = 0; d < dim; ++d) {
                    dot += double(qd[(h * nq + qt) * dim + d]) *
                           ggml_fp16_to_fp32(kd[(kh * nk + kt) * dim + d]);
                }
                weights[kt] = dot * scale;
            }
            const double maximum = *std::max_element(weights.begin(), weights.end());
            double total = 0.0;
            for (double & w : weights) { w = std::exp(w - maximum); total += w; }
            for (int d = 0; d < dim; ++d) {
                double expected = 0.0;
                for (int kt = 0; kt < visible; ++kt) {
                    expected += weights[kt] / total * ggml_fp16_to_fp32(vd[(kh * nk + kt) * dim + d]);
                }
                const float value = actual[(qt * hq + h) * dim + d];
                require(std::isfinite(value), "non-finite attention output");
                const double error = std::abs(double(value) - expected);
                max_error = std::max(max_error, error);
                sum_error_sq += error * error;
            }
        }
    }
    std::printf("%s KV=%d Q=%d causal GQA24/4 D256 max_error=%.8g rms_error=%.8g\n",
                ggml_backend_name(backend), nk, nq, max_error, std::sqrt(sum_error_sq / actual.size()));
    require(max_error < 0.005, "numerical mismatch against independent causal GQA reference");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::printf("PASS numerical attention\n");
}

int main(int argc, char ** argv) {
    ggml_init_params params = {1024 * 1024, nullptr, false};
    auto * ctx = ggml_init(params);
    require(ctx != nullptr, "legacy guard context");
    auto * node = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, hq, 1);
    node->op = GGML_OP_REMOTE_ATTN;
    require(!ggml_local_split_exec(node), "legacy local-split execution must fail closed");
    ggml_free(ctx);
    std::printf("PASS legacy local-split execution rejected\n");
    ggml_backend_load_all();
    auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(cpu != nullptr, "CPU backend initialization");
    numerical_case(cpu, 1, 1);
    numerical_case(cpu, 4, 4);
    numerical_case(cpu, 7, 1);
    ggml_backend_free(cpu);
    ggml_backend_dev_t device = nullptr;
    if (argc > 1) {
        device = ggml_backend_dev_by_name(argv[1]);
        require(device != nullptr, "requested backend device unavailable");
    } else {
        auto * reg = ggml_backend_reg_by_name("Vulkan");
        if (reg && ggml_backend_reg_dev_count(reg) > 0) { device = ggml_backend_reg_dev_get(reg, 0); }
    }
    if (!device) {
        std::printf("SKIP GPU: Vulkan unavailable; CPU numerical and legacy guards passed\n");
        return 0;
    }
    auto * gpu = ggml_backend_dev_init(device, nullptr);
    require(gpu != nullptr, "native GPU backend initialization");
    numerical_case(gpu, 1, 1);
    numerical_case(gpu, 4, 4);
    numerical_case(gpu, 7, 1);
    ggml_backend_free(gpu);
    std::printf("PASS all native attention numerical regressions\n");
    return 0;
}
