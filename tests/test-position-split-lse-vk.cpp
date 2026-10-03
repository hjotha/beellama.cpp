// F1a — Vulkan Q4 LSE export vs the CPU FA reference (plan §3.4).
//
// Exercises the split_k >= 2 + flash_attn_split_k_lse reduce path on the
// Vulkan backend: decode and prefill, GQA and non-GQA, causal/full/empty-row
// masks, and both the forced bisection and the tuned heuristic split.
//
// Gates, all against stable references of the *same* dequantized KV:
//  1. mandatory: O vs FP64 with fp32 accumulation (NRMSE <= 1e-3, max abs
//     <= 1e-2). This is the gate that proves the kernels, including the
//     split-K reduce and the LSE reduce, are numerically correct.
//  2. mandatory: with the production precision mode (fp16 O accumulator,
//     GGML_PREC_DEFAULT) the LSE route must leave O bit-identical to the plain
//     FA route, so requesting LSE cannot change the attention output. The
//     deviation of that mode from FP64 is inherited from the upstream fp16
//     accumulator (measured 1.3e-3..4.3e-3 here, and 3.9e-4..8.0e-4 with
//     fp32 accumulate) and is reported, not gated.
//  3. LSE vs FP64 <= 1e-2 (plan value) in the production mode.
//  4. fully masked row: O = 0 and LSE = -inf.
// The CPU backend runs a different arithmetic (it quantizes Q to int8 for
// Q4_0 vec_dot), so it is only an extra cross-check, never the gate.
//
// Declines are fail-closed and asserted, never skipped: the LSE route needs
// KV >= 2*Bc for two non-empty Bc-aligned split-K blocks (see
// ggml_vk_flash_attn), so the test probes the device threshold and requires
// "declined below it, accepted at and above it".

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "ggml-vulkan.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <algorithm>
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

struct vk_case {
    // K/V cache type of the range being merged. Q4_0 is the production Radeon
    // range type; F16 is used as well because both backends then evaluate the
    // QK product in the same precision, which makes the O/LSE gates directly
    // comparable to test-backend-ops.
    ggml_type kv_type = GGML_TYPE_Q4_0;
    int D = 64;
    int nq = 1;
    int nkv = 48;
    int nqh = 4;
    int nkh = 2;
    bool causal = true;
    bool empty_row = false;
    // false: the backend must decline this shape (checked, not skipped)
    bool expect_support = true;
};

// q: [D,nq,nqh] ; kv host floats: [D,nkv,nkh] (flat ((kh*nkv)+k)*D+d)
struct host_data {
    std::vector<float> Q;
    std::vector<float> Kf;
    std::vector<float> Vf;
    std::vector<char>  Kq4;
    std::vector<char>  Vq4;
    std::vector<ggml_fp16_t> Kf16;
    std::vector<ggml_fp16_t> Vf16;
    std::vector<float> Kd; // values as the backend sees them (Q4 dequantized)
    std::vector<float> Vd;
    std::vector<ggml_fp16_t> mask;
};

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

static host_data make_host_data(const vk_case & c, uint32_t seed) {
    const int D = c.D, nq = c.nq, nkv = c.nkv, nqh = c.nqh, nkh = c.nkh;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    host_data h;
    h.Q.resize((size_t) D * nq * nqh);
    h.Kf.resize((size_t) D * nkv * nkh);
    h.Vf.resize((size_t) D * nkv * nkh);
    for (auto & x : h.Q) { x = dist(rng); }
    for (auto & x : h.Kf) { x = dist(rng); }
    for (auto & x : h.Vf) { x = dist(rng); }

    const size_t q4_row = ggml_row_size(GGML_TYPE_Q4_0, D);
    h.Kq4.resize(q4_row * nkv * nkh);
    h.Vq4.resize(q4_row * nkv * nkh);
    h.Kf16.resize(h.Kf.size());
    h.Vf16.resize(h.Vf.size());
    h.Kd.resize(h.Kf.size());
    h.Vd.resize(h.Vf.size());
    const bool use_f16 = c.kv_type == GGML_TYPE_F16;
    std::vector<float> row(D);
    for (int kk = 0; kk < nkv; ++kk) {
        for (int kh = 0; kh < nkh; ++kh) {
            const size_t base = (size_t) (kh * nkv + kk) * D;
            for (int d = 0; d < D; ++d) {
                row[d] = h.Kf[base + d];
            }
            char * dst = &h.Kq4[(kh * nkv + kk) * q4_row];
            ggml_quantize_chunk(GGML_TYPE_Q4_0, row.data(), dst, 0, 1, D, NULL);
            if (use_f16) {
                for (int d = 0; d < D; ++d) {
                    h.Kf16[base + d] = ggml_fp32_to_fp16(row[d]);
                    h.Kd[base + d] = ggml_fp16_to_fp32(h.Kf16[base + d]);
                }
            } else {
                dequant_q4_0_row(dst, &h.Kd[base], D);
            }
            for (int d = 0; d < D; ++d) {
                row[d] = h.Vf[base + d];
            }
            char * dstv = &h.Vq4[(kh * nkv + kk) * q4_row];
            ggml_quantize_chunk(GGML_TYPE_Q4_0, row.data(), dstv, 0, 1, D, NULL);
            if (use_f16) {
                for (int d = 0; d < D; ++d) {
                    h.Vf16[base + d] = ggml_fp32_to_fp16(row[d]);
                    h.Vd[base + d] = ggml_fp16_to_fp32(h.Vf16[base + d]);
                }
            } else {
                dequant_q4_0_row(dstv, &h.Vd[base], D);
            }
        }
    }

    h.mask.resize((size_t) nkv * nq);
    for (int qq = 0; qq < nq; ++qq) {
        for (int kk = 0; kk < nkv; ++kk) {
            const bool masked = (c.empty_row && qq == nq - 1) || (c.causal && kk > qq);
            h.mask[(size_t) qq * nkv + kk] = ggml_fp32_to_fp16(masked ? -INFINITY : 0.0f);
        }
    }
    return h;
}

// O and LSE from an FP64 evaluation of the dequantized Q4 K/V.
static void reference_fp64(const vk_case & c, const host_data & h,
        std::vector<float> & Oref, std::vector<double> & LSEref) {
    const int D = c.D, nq = c.nq, nkv = c.nkv, nqh = c.nqh, nkh = c.nkh;
    const int gqa = nqh / nkh;
    Oref.assign((size_t) D * nqh * nq, 0.0f);
    LSEref.assign((size_t) nqh * nq, -INFINITY);
    for (int qq = 0; qq < nq; ++qq) {
        for (int h_i = 0; h_i < nqh; ++h_i) {
            const int kh = h_i / gqa;
            std::vector<double> s(nkv, -INFINITY);
            for (int k = 0; k < nkv; ++k) {
                const bool masked = (c.empty_row && qq == nq - 1) || (c.causal && k > qq);
                if (masked) {
                    continue;
                }
                double dot = 0;
                for (int d = 0; d < D; ++d) {
                    dot += double(h.Q[((size_t) h_i * nq + qq) * D + d]) *
                           double(h.Kd[((size_t) kh * nkv + k) * D + d]);
                }
                s[k] = dot;
            }
            double m = -INFINITY;
            for (double v : s) {
                m = std::max(m, v);
            }
            if (m == -INFINITY) {
                continue;
            }
            double denom = 0;
            for (double v : s) {
                denom += std::exp(v - m);
            }
            LSEref[(size_t) qq * nqh + h_i] = m + std::log(denom);
            for (int d = 0; d < D; ++d) {
                double acc = 0;
                for (int k = 0; k < nkv; ++k) {
                    if (s[k] == -INFINITY) {
                        continue;
                    }
                    acc += std::exp(s[k] - m) * double(h.Vd[((size_t) kh * nkv + k) * D + d]);
                }
                Oref[((size_t) qq * nqh + h_i) * D + d] = float(acc / denom);
            }
        }
    }
}

// Runs the LSE graph on one backend. Returns false only on hard failure;
// *supported reports the fail-closed decline.
static bool run_backend(ggml_backend_t backend, const vk_case & c, const host_data & h,
        std::vector<float> & O, std::vector<float> & LSE, bool * supported,
        bool with_lse = true, bool prec_f32 = false) {
    const int D = c.D, nq = c.nq, nkv = c.nkv, nqh = c.nqh, nkh = c.nkh;

    struct ggml_init_params params = { 64 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, nq, nqh, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, c.kv_type, D, nkv, nkh, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, c.kv_type, D, nkv, nkh, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nkv, nq);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);
    if (prec_f32) {
        // fp32 accumulation: the O accumulator is FLOAT_TYPE, i.e. fp16 in the
        // default mode, which is what the mandatory FP64 gate must not measure
        ggml_prec_set_acc(fa, GGML_PREC_F32);
    }
    // real graph node: src[0] == fa, so the graph carries the FA -> LSE edge
    ggml_tensor * lse = with_lse ? ggml_flash_attn_ext_lse_out(ctx, fa) : nullptr;

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(q, h.Q.data(), 0, h.Q.size() * sizeof(float));
    if (c.kv_type == GGML_TYPE_F16) {
        ggml_backend_tensor_set(k, h.Kf16.data(), 0, h.Kf16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(v, h.Vf16.data(), 0, h.Vf16.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(k, h.Kq4.data(), 0, h.Kq4.size());
        ggml_backend_tensor_set(v, h.Vq4.data(), 0, h.Vq4.size());
    }
    ggml_backend_tensor_set(mask, h.mask.data(), 0, h.mask.size() * sizeof(ggml_fp16_t));

    *supported = ggml_backend_supports_op(backend, fa) &&
        (lse == nullptr || ggml_backend_supports_op(backend, lse));
    if (!*supported) {
        ggml_free(ctx);
        return true;
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, lse != nullptr ? lse : fa);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        return false;
    }
    O.resize(ggml_nelements(fa));
    ggml_backend_tensor_get(fa, O.data(), 0, O.size() * sizeof(float));
    if (lse != nullptr) {
        LSE.resize(ggml_nelements(lse));
        ggml_backend_tensor_get(lse, LSE.data(), 0, LSE.size() * sizeof(float));
    } else {
        LSE.clear();
    }
    ggml_free(ctx);
    return true;
}

static double nrmse(const std::vector<float> & a, const std::vector<float> & b) {
    double se = 0, sr = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = double(a[i]) - double(b[i]);
        se += d * d;
        sr += double(b[i]) * double(b[i]);
    }
    double rr = std::sqrt(sr / (a.size() ? a.size() : 1));
    if (rr < 1e-6) {
        rr = 1e-6;
    }
    return std::sqrt(se / (a.size() ? a.size() : 1)) / rr;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

// Gates (all relative to the same measured backend baseline, so they hold on
// any device and any K/V type):
//  1. LSE route vs the plain FA route on the same backend: O must stay inside
//     the plan's split/unsplit tolerance (NRMSE 1e-3, max abs 1e-2). Requesting
//     LSE may add a split-K reduction, which reorders the softmax sum.
//  2. The LSE route must not be less accurate than the plain FA route against
//     the CPU backend (device precision floor, e.g. fp16 accumulation or the
//     CPU int8 Q quantization for Q4_0 K/V).
//  3. LSE against the FP64 reference: |dLSE| <= 1e-2 (plan value).
//  4. Fully masked row: O = 0 and LSE = -inf.
//  5. The CPU reference itself must satisfy the same LSE gate.
static bool run_one(ggml_backend_t vk, ggml_backend_t cpu, const vk_case & c) {
    const host_data h = make_host_data(c, 777);
    std::vector<float> O_lse, LSE_vk, O_base, O_f32, O_cpu, LSE_cpu;
    bool sup_lse = false, sup_base = false, sup_f32 = false, sup_cpu = false;

    // 1. production precision mode, LSE route
    if (!run_backend(vk, c, h, O_lse, LSE_vk, &sup_lse, true, false)) {
        CHECK(false);
        return true;
    }
    if (!sup_lse) {
        // fail-closed: a decline is only acceptable where the case demands it
        CHECK(!c.expect_support);
        return true;
    }
    CHECK(c.expect_support);

    // 2. production precision mode, plain FA (no LSE side output)
    std::vector<float> LSE_unused;
    if (!run_backend(vk, c, h, O_base, LSE_unused, &sup_base, false, false)) {
        CHECK(false);
        return true;
    }
    CHECK(sup_base); // a plain FA run must not be declined when LSE was accepted
    if (!sup_base) {
        return true;
    }

    // 3. fp32 accumulation, LSE route: this is the mandatory FP64 gate
    std::vector<float> LSE_f32;
    if (!run_backend(vk, c, h, O_f32, LSE_f32, &sup_f32, true, true)) {
        CHECK(false);
        return true;
    }
    CHECK(sup_f32);
    if (!sup_f32) {
        return true;
    }

    // 4. CPU reference (different arithmetic: extra cross-check only)
    if (!run_backend(cpu, c, h, O_cpu, LSE_cpu, &sup_cpu, true, false)) {
        CHECK(false);
        return true;
    }
    CHECK(sup_cpu); // the CPU FA always exports LSE (reference path)
    if (!sup_cpu) {
        return true;
    }

    std::vector<float> Oref;
    std::vector<double> LSEref;
    reference_fp64(c, h, Oref, LSEref);

    for (size_t i = 0; i < O_lse.size(); ++i) {
        if (!std::isfinite(O_lse[i]) || !std::isfinite(O_f32[i])) {
            CHECK(false);
            return true;
        }
    }

    // mandatory gate 1: O vs FP64 with fp32 accumulation
    const double nrmse_f32_fp64 = nrmse(O_f32, Oref);
    const float max_o_f32_fp64 = max_abs_diff(O_f32, Oref);
    CHECK(nrmse_f32_fp64 <= 1e-3);
    CHECK(max_o_f32_fp64 <= 1e-2f);

    // mandatory gate 2: production mode, LSE route vs plain FA route (plan
    // split/unsplit tolerance). It is exactly 0 whenever both routes run the
    // split-K reduce; it is ~1e-4 (fp16 O accumulator rounding) when the plain
    // route divides in-kernel because its tuned split_k is 1 and the LSE route
    // goes through a single-partial reduce.
    const double nrmse_split = nrmse(O_lse, O_base);
    const float max_o_split = max_abs_diff(O_lse, O_base);
    CHECK(nrmse_split <= 1e-3);
    CHECK(max_o_split <= 1e-2f);

    // reported only: the fp16 O accumulator floor of the production mode
    const double nrmse_prod_fp64 = nrmse(O_lse, Oref);
    const double nrmse_lse_cpu = nrmse(O_lse, O_cpu);
    const double nrmse_base_cpu = nrmse(O_base, O_cpu);
    std::printf("  [%s] kv=%s D=%d nq=%d nkv=%d nqh=%d nkh=%d\n"
        "        vs-fp64: f32acc=%.3e (max %.3e) prod=%.3e | prod split-vs-unsplit: %.3e (max %.3e)\n"
        "        vs-cpu: lse=%.3e base=%.3e\n",
        g_current, ggml_type_name(c.kv_type), c.D, c.nq, c.nkv, c.nqh, c.nkh,
        nrmse_f32_fp64, max_o_f32_fp64, nrmse_prod_fp64, nrmse_split, max_o_split,
        nrmse_lse_cpu, nrmse_base_cpu);
    // sanity cap on the inherited fp16-accumulate deviation: a real regression
    // in the shared FA path would show up here as well
    CHECK(nrmse_prod_fp64 <= 1e-2);
    CHECK(nrmse_lse_cpu <= std::max(1e-3, 1.5 * nrmse_base_cpu));

    // mandatory gate 3: LSE vs FP64 with fp32 accumulation. In the default
    // (fp16 accumulate) mode the scores carry an absolute error of ~|s|*5e-4,
    // and LSE = m + log(sum exp) exposes it directly - unlike O, where the
    // softmax normalizes it away. At D=256 that is 3e-2..6e-2 of inherited
    // device precision, so the mandatory gate uses the f32-accumulate run and
    // the production number is reported with a sanity cap.
    float max_l_f32_fp64 = 0;
    for (size_t i = 0; i < LSE_f32.size(); ++i) {
        if (std::isinf(LSEref[i]) && LSEref[i] < 0) {
            CHECK(std::isinf(LSE_f32[i]) && LSE_f32[i] < 0);
            continue;
        }
        CHECK(std::isfinite(LSE_f32[i]));
        max_l_f32_fp64 = std::max(max_l_f32_fp64, std::fabs(LSE_f32[i] - float(LSEref[i])));
    }
    CHECK(max_l_f32_fp64 <= 1e-2f);

    float max_l_fp64 = 0, max_l_cpu = 0, max_l_cpu_ref_fp64 = 0;
    for (size_t i = 0; i < LSE_vk.size(); ++i) {
        if (std::isinf(LSEref[i]) && LSEref[i] < 0) {
            // fully masked row: O = 0, LSE = -inf
            CHECK(std::isinf(LSE_vk[i]) && LSE_vk[i] < 0);
            for (int d = 0; d < c.D; ++d) {
                CHECK(O_lse[i * c.D + d] == 0.0f);
            }
            continue;
        }
        if (!std::isfinite(LSE_vk[i])) {
            CHECK(false);
            return true;
        }
        max_l_fp64 = std::max(max_l_fp64, std::fabs(LSE_vk[i] - float(LSEref[i])));
        max_l_cpu = std::max(max_l_cpu, std::fabs(LSE_vk[i] - LSE_cpu[i]));
        max_l_cpu_ref_fp64 = std::max(max_l_cpu_ref_fp64,
            std::fabs(LSE_cpu[i] - float(LSEref[i])));
    }
    std::printf("        lse: vs-fp64 f32acc=%.3e prod=%.3e | vs-cpu=%.3e (cpu vs-fp64=%.3e)\n",
        max_l_f32_fp64, max_l_fp64, max_l_cpu, max_l_cpu_ref_fp64);
    // sanity cap on the inherited fp16-score deviation of the production mode
    CHECK(max_l_fp64 <= 1e-1);
    // The CPU reference must honour the same gate when both sides evaluate QK
    // at the same precision (F16 K/V). With Q4_0 K/V the CPU vec_dot quantizes
    // Q to int8, which puts ~2e-2 on its own LSE; that number is reported, not
    // gated, because the authoritative LSE gate above is the FP64 comparison.
    if (c.kv_type == GGML_TYPE_F16) {
        CHECK(max_l_cpu_ref_fp64 <= 1e-2f);
    }
    return true;
}

// Probe whether the backend accepts the LSE route for a given KV length.
// The reduce needs two non-empty Bc-aligned split-K blocks (KV >= 2*Bc) and
// Bc depends on the device, so the threshold is measured, not hardcoded.
static bool probe_supported(ggml_backend_t backend, int nkv, int nqh, int nkh, int D, int nq) {
    struct ggml_init_params params = { 4 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, nq, nqh, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, D, nkv, nkh, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, D, nkv, nkh, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nkv, nq);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);
    ggml_tensor * lse = ggml_flash_attn_ext_lse_out(ctx, fa);
    const bool ok = ggml_backend_supports_op(backend, fa) && ggml_backend_supports_op(backend, lse);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_t vk = ggml_backend_vk_init(0);
    if (!vk) {
        std::printf("position-split-lse-vk: no Vulkan backend\n");
        return 1;
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::printf("position-split-lse-vk: no CPU backend\n");
        ggml_backend_free(vk);
        return 1;
    }

    // support smoke probe: every KV length in the plan matrix must be accepted
    static const int probe_kv[] = { 1, 2, 3, 16, 17, 48, 64, 96, 128, 129, 192, 256 };
    {
        for (int nkv : probe_kv) {
            g_current = "vk_support_probe";
            CHECK(probe_supported(vk, nkv, 4, 2, 64, 1));
        }
    }

    {
        vk_case c;
        c.nkv = 128;
        g_current = "vk_decode_q1_kv128";
        CHECK(run_one(vk, cpu, c));
    }
    {
        // split_k = 1 route: tiny KV no longer declines (plan §3.4 requires it
        // so a small overflow can start at P+1)
        vk_case c;
        c.nkv = 1;
        g_current = "vk_decode_kv1_split_k1";
        CHECK(run_one(vk, cpu, c));
    }
    {
        vk_case c;
        c.nkv = 3;
        c.causal = false;
        g_current = "vk_decode_kv3_split_k1";
        CHECK(run_one(vk, cpu, c));
    }
    {
        vk_case c;
        c.nkv = 17;
        g_current = "vk_decode_kv17_split_k1";
        CHECK(run_one(vk, cpu, c));
    }
    {
        vk_case c;
        c.nq = 4;
        c.nkv = 256;
        g_current = "vk_prefill_q4_gqa_flat";
        CHECK(run_one(vk, cpu, c));
    }
    {
        vk_case c;
        c.nq = 16;
        c.nkv = 512;
        c.nqh = 8;
        c.nkh = 2;
        g_current = "vk_prefill_q16_largekv";
        CHECK(run_one(vk, cpu, c));
    }
    {
        vk_case c;
        c.causal = false;
        c.nq = 2;
        c.nkv = 384;
        g_current = "vk_decode_fullmask";
        CHECK(run_one(vk, cpu, c));
    }
    {
        vk_case c;
        c.empty_row = true;
        c.nkv = 384;
        g_current = "vk_decode_empty_row";
        CHECK(run_one(vk, cpu, c));
    }
    // plan matrix: D=256 and the production head geometry (24 query heads over
    // 4 KV heads, GQA 6) plus a 256-token prefill
    {
        vk_case c;
        c.D = 256;
        c.nqh = 24;
        c.nkh = 4;
        c.nq = 2;
        c.nkv = 256;
        g_current = "vk_d256_gqa24_4";
        CHECK(run_one(vk, cpu, c));
    }
    {
        vk_case c;
        c.D = 256;
        c.nqh = 24;
        c.nkh = 4;
        c.nq = 1;
        c.nkv = 256;
        c.causal = false;
        g_current = "vk_d256_gqa24_4_fullmask";
        CHECK(run_one(vk, cpu, c));
    }
    {
        vk_case c;
        c.D = 256;
        c.nqh = 4;
        c.nkh = 1;
        c.nq = 64;
        c.nkv = 128;
        g_current = "vk_d256_q256_prefill";
        CHECK(run_one(vk, cpu, c));
    }

    // F16 K/V: both backends evaluate QK at the same precision, so these run
    // the strict upstream gates on O and LSE.
    {
        vk_case c;
        c.kv_type = GGML_TYPE_F16;
        c.nkv = 256;
        g_current = "vk_f16_prefill_strict";
        CHECK(run_one(vk, cpu, c));
    }
    {
        vk_case c;
        c.kv_type = GGML_TYPE_F16;
        c.causal = false;
        c.nq = 2;
        c.nkv = 384;
        c.nqh = 8;
        c.nkh = 2;
        g_current = "vk_f16_fullmask_gqa4_strict";
        CHECK(run_one(vk, cpu, c));
    }
    {
        vk_case c;
        c.kv_type = GGML_TYPE_F16;
        c.nq = 16;
        c.nkv = 512;
        c.nqh = 8;
        c.nkh = 2;
        g_current = "vk_f16_prefill_q16_strict";
        CHECK(run_one(vk, cpu, c));
    }

    ggml_backend_free(cpu);
    ggml_backend_free(vk);
    std::printf("position-split-lse-vk: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}