// F1 — position-split LSE under the real backend scheduler (plan §3.4,
// external review C).
//
// The FA-LSE contract must survive ggml_backend_sched, not just a direct
// ggml_backend_graph_compute call. The previous design (LSE smuggled in as
// src[4]) had no FA -> LSE graph edge: the scheduler could stage a leaf copy of
// the LSE tensor on the FA backend, the kernel wrote that copy, and a later
// consumer read the original buffer instead.
//
// This test pins the whole chain on a GPU + CPU scheduler:
//   1. the LSE node is a graph node (not a leaf) with src[0] == FA,
//   2. FA is ordered before LSE,
//   3. the LSE node is co-located with the FA node's backend (scheduler pin),
//   4. a CPU-pinned consumer of the LSE sees the values the GPU kernel wrote,
//      which proves the cross-backend copy reads the FA-written buffer,
//   5. the values match the CPU FA reference within the plan gate.
//
// Declines are fail-closed: if the GPU backend declines the LSE route the test
// fails, it never reports a pass by skipping.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)

// records the evaluation order of the FA and LSE nodes
struct eval_order {
    ggml_tensor * fa = nullptr;
    ggml_tensor * lse = nullptr;
    int n_fa = 0;
    int n_lse = 0;
};

static bool eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    eval_order * eo = (eval_order *) user_data;
    if (ask) {
        return true; // observe every node
    }
    if (t->op == GGML_OP_FLASH_ATTN_EXT) {
        eo->fa = t;
        eo->n_fa++;
    } else if (t->op == GGML_OP_FLASH_ATTN_EXT_LSE) {
    // only the pinned (final) evaluation is recorded: the scheduler may ask
    // for a node before a split boundary and evaluate it early
        if (eo->fa != nullptr) {
            eo->lse = t;
            eo->n_lse++;
        }
    }
    return true;
}

static void dequant_q4_0_row(const void * row, float * out, int D) {
    const uint8_t * b = (const uint8_t *) row;
    for (int j = 0; j < D; ++j) {
        const uint8_t * blk = b + (j / 32) * 18;
        const float d = ggml_fp16_to_fp32(*(const ggml_fp16_t *) (const void *) blk);
        const uint8_t byte = blk[2 + (j % 32) % 16];
        const uint8_t nib = ((j % 32) < 16) ? (byte & 0x0F) : (byte >> 4);
        out[j] = (float(int(nib)) - 8.0f) * d;
    }
}

int main() {
    ggml_backend_t gpu = ggml_backend_vk_init(0);
    if (!gpu) {
        std::printf("position-split-lse-sched: no Vulkan backend\n");
        return 1;
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::printf("position-split-lse-sched: no CPU backend\n");
        ggml_backend_free(gpu);
        return 1;
    }

    const int D = 64, nq = 1, nkv = 256, nqh = 4, nkh = 2;

    // host data
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> Q((size_t) D * nq * nqh);
    std::vector<float> K((size_t) D * nkv * nkh);
    std::vector<float> V((size_t) D * nkv * nkh);
    for (auto & x : Q) { x = dist(rng); }
    for (auto & x : K) { x = dist(rng); }
    for (auto & x : V) { x = dist(rng); }

    const size_t q4_row = ggml_row_size(GGML_TYPE_Q4_0, D);
    std::vector<char> Kq(q4_row * nkv * nkh), Vq(q4_row * nkv * nkh);
    std::vector<float> Kd((size_t) D * nkv * nkh), Vd((size_t) D * nkv * nkh);
    std::vector<float> row(D);
    for (int kk = 0; kk < nkv; ++kk) {
        for (int kh = 0; kh < nkh; ++kh) {
            const size_t base = (size_t) (kh * nkv + kk) * D;
            for (int d = 0; d < D; ++d) { row[d] = K[base + d]; }
            char * dst = &Kq[(kh * nkv + kk) * q4_row];
            ggml_quantize_chunk(GGML_TYPE_Q4_0, row.data(), dst, 0, 1, D, NULL);
            dequant_q4_0_row(dst, &Kd[base], D);
            for (int d = 0; d < D; ++d) { row[d] = V[base + d]; }
            char * dstv = &Vq[(kh * nkv + kk) * q4_row];
            ggml_quantize_chunk(GGML_TYPE_Q4_0, row.data(), dstv, 0, 1, D, NULL);
            dequant_q4_0_row(dstv, &Vd[base], D);
        }
    }
    std::vector<ggml_fp16_t> mask((size_t) nkv * nq);
    for (int qq = 0; qq < nq; ++qq) {
        for (int kk = 0; kk < nkv; ++kk) {
            mask[(size_t) qq * nkv + kk] = ggml_fp32_to_fp16(kk > qq ? -INFINITY : 0.0f);
        }
    }

    // FP64 reference for LSE (plan gate |dLSE| <= 1e-2) and the LSE sum
    std::vector<double> LSEref((size_t) nqh * nq, -INFINITY);
    double sum_ref = 0.0;
    for (int qq = 0; qq < nq; ++qq) {
        for (int h = 0; h < nqh; ++h) {
            const int kh = h / (nqh / nkh);
            std::vector<double> s(nkv, -INFINITY);
            for (int k = 0; k < nkv; ++k) {
                if (k > qq) {
                    continue;
                }
                double dot = 0;
                for (int d = 0; d < D; ++d) {
                    dot += double(Q[((size_t) h * nq + qq) * D + d]) * double(Kd[((size_t) kh * nkv + k) * D + d]);
                }
                s[k] = dot;
            }
            double m = -INFINITY;
            for (double v : s) { m = std::max(m, v); }
            double denom = 0;
            for (double v : s) { denom += std::exp(v - m); }
            LSEref[(size_t) qq * nqh + h] = m + std::log(denom);
            sum_ref += LSEref[(size_t) qq * nqh + h];
        }
    }

    ggml_backend_t backends[2] = { gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 2, 4096, false, true);

    struct ggml_init_params params = { 64 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, nq, nqh, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, D, nkv, nkh, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, D, nkv, nkh, 1);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nkv, nq);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f, 0.0f, 0.0f);
    ggml_set_name(fa, "fa");
    ggml_tensor * lse = ggml_flash_attn_ext_lse_out(ctx, fa);
    // production placement: the FA range runs on the GPU, the merge side runs
    // on the CPU. The LSE node is *not* pinned: it must follow the FA node.
    ggml_backend_sched_set_tensor_backend(sched, fa, gpu);
    // CPU-pinned consumer: forces a cross-backend read of the LSE buffer, which
    // is exactly the path the old src[4] design broke
    ggml_tensor * lse_sum = ggml_sum(ctx, lse);
    ggml_backend_sched_set_tensor_backend(sched, lse_sum, cpu);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, lse_sum);

    // graph structure: LSE must be a real node reachable from the FA output,
    // not a detached leaf
    CHECK(lse->op == GGML_OP_FLASH_ATTN_EXT_LSE);
    CHECK(lse->src[0] == fa);
    CHECK(ggml_graph_get_tensor(gf, "fa") == fa);
    CHECK(ggml_graph_get_tensor(gf, "fa_lse") == lse);

    eval_order eo;
    ggml_backend_sched_set_eval_callback(sched, eval_cb, &eo);

    // the GPU route must support LSE, otherwise this test cannot prove anything
    const bool supported = ggml_backend_supports_op(gpu, fa) && ggml_backend_supports_op(gpu, lse);
    if (!supported) {
        std::printf("position-split-lse-sched: GPU backend declines the FA-LSE route\n");
        ggml_free(ctx);
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
        ggml_backend_free(gpu);
        return 1;
    }
    CHECK(supported);

    // allocate with the CPU pin in place (ggml_backend_sched_alloc_graph runs
    // the split; a compute call would reset the assignments first)
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        CHECK(false);
        ggml_free(ctx);
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
        ggml_backend_free(gpu);
        return 1;
    }

    // co-location: the LSE node must live on the FA node's backend
    ggml_backend_t fa_backend = ggml_backend_sched_get_tensor_backend(sched, fa);
    ggml_backend_t lse_backend = ggml_backend_sched_get_tensor_backend(sched, lse);
    CHECK(fa_backend == gpu);
    CHECK(lse_backend == fa_backend);
    // and the CPU consumer must have produced a separate split
    std::printf("  backends: fa=%s lse=%s lse_sum=%s splits=%d copies=%d\n",
        ggml_backend_name(fa_backend), ggml_backend_name(lse_backend),
        ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched, lse_sum)),
        ggml_backend_sched_get_n_splits(sched), ggml_backend_sched_get_n_copies(sched));
    CHECK(ggml_backend_sched_get_tensor_backend(sched, lse_sum) == cpu);
    CHECK(ggml_backend_sched_get_n_splits(sched) >= 2);

    // feed the inputs and run for real
    ggml_backend_tensor_set(q, Q.data(), 0, Q.size() * sizeof(float));
    ggml_backend_tensor_set(k, Kq.data(), 0, Kq.size());
    ggml_backend_tensor_set(v, Vq.data(), 0, Vq.size());
    ggml_backend_tensor_set(m, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        CHECK(false);
    } else {
        std::vector<float> LSE((size_t) nqh * nq);
        ggml_backend_tensor_get(lse, LSE.data(), 0, LSE.size() * sizeof(float));
        float sum_gpu = 0.0f;
        float max_l = 0.0f;
        for (size_t i = 0; i < LSE.size(); ++i) {
            CHECK(std::isfinite(LSE[i]));
            max_l = std::max(max_l, std::fabs(LSE[i] - float(LSEref[i])));
            sum_gpu += LSE[i];
        }
        std::printf("  lse(vs-fp64)=%.3e sum_gpu=%.6f sum_ref=%.6f splits=%d\n",
            max_l, sum_gpu, sum_ref, ggml_backend_sched_get_n_splits(sched));
        CHECK(max_l <= 1e-2f);

        // the CPU-pinned consumer must see the GPU-written values
        float sum_cpu = 0.0f;
        ggml_backend_tensor_get(lse_sum, &sum_cpu, 0, sizeof(float));
        std::printf("  cpu consumer sum=%.6f (ref %.6f)\n", sum_cpu, sum_ref);
        CHECK(std::fabs(sum_cpu - float(sum_ref)) <= 1e-2f * std::max(1.0f, std::fabs(float(sum_ref))));
        // a stale/zero LSE buffer (the old leaf-copy hazard) would show up here
        CHECK(std::fabs(sum_cpu - sum_gpu) <= 1e-3f * std::max(1.0f, std::fabs(sum_gpu)));

        // evaluation order: the FA node must be evaluated before the LSE node
        CHECK(eo.n_fa >= 1);
        CHECK(eo.n_lse >= 1);
        CHECK(eo.fa == fa);
        CHECK(eo.lse == lse);
    }

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    std::printf("position-split-lse-sched: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}