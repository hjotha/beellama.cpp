// Deterministic op-level diagnostic for the Q4_0 Vulkan FLASH_ATTN_EXT path.
//
// Context (F5 packed overflow investigation): a "native packed" attention that
// feeds the raw Q4_0 overflow storage directly to Vulkan FA must be numerically
// equivalent to the supported F16 reference (GET_ROWS_AS(F16) gather) before it
// can be enabled. This test isolates the op from any model or KV-cache code.
//
// It compares, on the same Vulkan device and with model-like shapes
// (head_dim 128, 64 query heads, 8 KV heads, scale = 1/sqrt(head_dim)):
//   * O and LSE with Q4_0 K/V vs the SAME dequantized values as F16, at the
//     SAME live extent -> must match (these are hard assertions);
//   * the same comparison with a PADDED extent (live rows + masked -inf tail),
//     which is the layout the gather reference uses -> must also match;
//   * the CROSS pairs (valid Q4 vs padded F16 and vice versa) -> reported only:
//     they measure how much the valid-vs-padded extent difference alone moves O
//     and LSE, without a reference to call either one "wrong".
//
// This is a diagnostic, not an acceptance gate. Keep the packed path disabled
// by default until a numeric gate over this comparison passes.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static ggml_backend_t open_vulkan0() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        if (name && strstr(name, "Vulkan0")) {
            printf("device: %s | %s\n", name, ggml_backend_dev_description(dev));
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

static void max_abs_diff(const std::vector<float> & a, const std::vector<float> & b,
                         double & max_abs, double & rms, int64_t & at) {
    max_abs = 0.0; rms = 0.0; at = -1;
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        rms += d*d;
        if (d > max_abs) { max_abs = d; at = (int64_t) i; }
    }
    rms = std::sqrt(rms / (double) n);
}

struct result {
    std::vector<float> o_q4v, o_f16v, o_q4p, o_f16p;
    std::vector<float> l_q4v, l_f16v, l_q4p, l_f16p;
};

static bool run_case(ggml_backend_t backend, int64_t kv_valid, int64_t nq, int64_t pad,
                     double tol_same, int & failures, bool folded = false,
                     int64_t hsk = 128, int64_t nh = 8, int64_t nhq = 64) {
    // Defaults are the synthetic 128/8/64 shape; the model's overflow layers use
    // hsk=256, 4 KV heads, 24 query heads (GQA 6) -- the shapes of the integrated
    // first divergence at il=7, eval 14/15.
    const int64_t kv_pad = kv_valid + pad;

    ggml_init_params ip = { 768u*1024u*1024u, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * kq4_st = folded
        ? ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hsk*nh, kv_pad)
        : ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, hsk, kv_pad, nh, 1);
    ggml_tensor * vq4_st = folded
        ? ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hsk*nh, kv_pad)
        : ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, hsk, kv_pad, nh, 1);
    ggml_tensor * kf16_st = folded
        ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hsk*nh, kv_pad)
        : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hsk, kv_pad, nh, 1);
    ggml_tensor * vf16_st = folded
        ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hsk*nh, kv_pad)
        : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hsk, kv_pad, nh, 1);

    // Folded layout = the model's: a [head_dim * n_head, rows] storage reshaped to
    // [head_dim, n_head, rows, 1], i.e. heads interleaved inside each token row.
    auto fold = [&](ggml_tensor * st, int64_t rows) {
        // [head_dim*n_head, rows] -> [head_dim, n_head, rows, 1] -> permute to the
        // FA layout [head_dim, rows, n_head, 1], exactly like build_attn_mha does.
        return ggml_permute(ctx, ggml_reshape_4d(ctx, st, hsk, nh, rows, 1), 0, 2, 1, 3);
    };
    auto view_valid = [&](ggml_tensor * st) {
        if (folded) {
            ggml_tensor * v2 = ggml_view_2d(ctx, st, hsk*nh, kv_valid, st->nb[1], 0);
            return ggml_permute(ctx, ggml_reshape_4d(ctx, v2, hsk, nh, kv_valid, 1), 0, 2, 1, 3);
        }
        return ggml_view_4d(ctx, st, hsk, kv_valid, nh, 1, st->nb[1], st->nb[2], st->nb[3], 0);
    };
    ggml_tensor * kq4_v = view_valid(kq4_st);
    ggml_tensor * vq4_v = view_valid(vq4_st);
    ggml_tensor * kf16_v = view_valid(kf16_st);
    ggml_tensor * vf16_v = view_valid(vf16_st);
    // Padded extent (= full storage) as 4D for the FA call.
    ggml_tensor * kq4_p = folded ? fold(kq4_st, kv_pad) : kq4_st;
    ggml_tensor * vq4_p = folded ? fold(vq4_st, kv_pad) : vq4_st;
    ggml_tensor * kf16_p = folded ? fold(kf16_st, kv_pad) : kf16_st;
    ggml_tensor * vf16_p = folded ? fold(vf16_st, kv_pad) : vf16_st;

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsk, nq, nhq, 1);

    ggml_tensor * mask_v = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_valid, nq);
    ggml_tensor * mask_p = nullptr;
    if (pad > 0) {
        mask_p = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_pad, nq);
    }

    const float scale = 1.0f / std::sqrt((float) hsk);
    ggml_tensor * o_q4v = ggml_flash_attn_ext(ctx, q, kq4_v, vq4_v, mask_v, scale, 0.0f, 0.0f);
    ggml_tensor * o_f16v = ggml_flash_attn_ext(ctx, q, kf16_v, vf16_v, mask_v, scale, 0.0f, 0.0f);
    ggml_tensor * l_q4v = ggml_flash_attn_ext_lse_out(ctx, o_q4v);
    ggml_tensor * l_f16v = ggml_flash_attn_ext_lse_out(ctx, o_f16v);

    ggml_tensor * o_q4p = nullptr, * o_f16p = nullptr, * l_q4p = nullptr, * l_f16p = nullptr;
    if (pad > 0) {
        o_q4p  = ggml_flash_attn_ext(ctx, q, kq4_p, vq4_p, mask_p, scale, 0.0f, 0.0f);
        o_f16p = ggml_flash_attn_ext(ctx, q, kf16_p, vf16_p, mask_p, scale, 0.0f, 0.0f);
        l_q4p  = ggml_flash_attn_ext_lse_out(ctx, o_q4p);
        l_f16p = ggml_flash_attn_ext_lse_out(ctx, o_f16p);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { printf("INVALID alloc failed kv=%lld\n", (long long) kv_valid); ggml_free(ctx); return false; }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int64_t row_len = folded ? hsk*nh : hsk;
    const int64_t n_rows  = folded ? kv_pad : kv_pad*nh;
    const size_t  n_vals  = size_t(row_len) * size_t(n_rows);
    std::vector<float> kf(n_vals), vf(n_vals);
    for (auto & x : kf) x = dist(rng);
    for (auto & x : vf) x = dist(rng);

    // Poison a few INACTIVE tail rows (masked -inf) with non-finite values: the
    // raw view reads whatever is in storage there, while the gather reference reads
    // a duplicated finite row. If masking is neutral even for NaN/Inf, O and LSE
    // must still match; if this fails, non-finite stale storage is the cause.
    if (pad > 0 && getenv("FA_Q4_NAN_TAIL") != nullptr) {
        for (int64_t r = kv_valid; r < std::min<int64_t>(kv_valid + 8, kv_pad); ++r) {
            const size_t base = (folded ? size_t(r) : size_t(r)*nh) * size_t(row_len);
            if (folded) {
                kf[base + 5] = NAN; kf[base + 7] = INFINITY;
                vf[base + 4] = NAN; vf[base + 9] = -INFINITY;
            } else {
                for (int64_t h = 0; h < nh; ++h) {
                    const size_t o = size_t(row_len) * size_t(r + kv_pad*h);
                    kf[o + 5] = NAN; kf[o + 7] = INFINITY;
                    vf[o + 4] = NAN; vf[o + 9] = -INFINITY;
                }
            }
        }
    }
    std::vector<uint8_t> kq_bytes(ggml_row_size(GGML_TYPE_Q4_0, row_len) * n_rows);
    std::vector<uint8_t> vq_bytes(ggml_row_size(GGML_TYPE_Q4_0, row_len) * n_rows);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, kf.data(), kq_bytes.data(), 0, n_rows, row_len, nullptr);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, vf.data(), vq_bytes.data(), 0, n_rows, row_len, nullptr);

    std::vector<float> kd(n_vals), vd(n_vals);
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q4_0);
    tt->to_float(kq_bytes.data(), kd.data(), (int64_t) n_vals);
    tt->to_float(vq_bytes.data(), vd.data(), (int64_t) n_vals);
    std::vector<ggml_fp16_t> kd16(kd.size()), vd16(vd.size());
    for (size_t i = 0; i < kd.size(); ++i) {
        kd16[i] = ggml_fp32_to_fp16(kd[i]);
        vd16[i] = ggml_fp32_to_fp16(vd[i]);
    }
    // Model-faithful tail VALUES: the gather reference reads a duplicated cell-0
    // row for the inactive tail (or zeros), while a raw storage view reads whatever
    // stale bytes are there. Reproduce that difference on the F16 reference side so
    // the comparison tests mask neutrality itself, not just the extent.
    if (pad > 0) {
        const size_t row = size_t(row_len);
        if (folded) {
            for (int64_t r = kv_valid; r < kv_pad; ++r) {
                std::memcpy(&kd16[row*size_t(r)], &kd16[0], row*sizeof(ggml_fp16_t));
                std::memcpy(&vd16[row*size_t(r)], &vd16[0], row*sizeof(ggml_fp16_t));
            }
        } else {
            for (int64_t h = 0; h < nh; ++h) {
                for (int64_t r = kv_valid; r < kv_pad; ++r) {
                    const size_t dst = row * size_t(r + kv_pad*h);
                    const size_t src = row * size_t(0 + kv_pad*h);
                    std::memcpy(&kd16[dst], &kd16[src], row*sizeof(ggml_fp16_t));
                    std::memcpy(&vd16[dst], &vd16[src], row*sizeof(ggml_fp16_t));
                }
            }
        }
    }
    ggml_backend_tensor_set(kq4_st,  kq_bytes.data(), 0, kq_bytes.size());
    ggml_backend_tensor_set(vq4_st,  vq_bytes.data(), 0, vq_bytes.size());
    ggml_backend_tensor_set(kf16_st, kd16.data(),    0, kd16.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(vf16_st, vd16.data(),    0, vd16.size()*sizeof(ggml_fp16_t));

    std::vector<float> qv(hsk * nq * nhq);
    for (auto & x : qv) x = dist(rng) * 4.0f;
    ggml_backend_tensor_set(q, qv.data(), 0, qv.size()*sizeof(float));

    static const bool dead_row = [] {
        const char * v = getenv("FA_Q4_DEAD_ROW");
        return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
    }();
    auto fill_mask = [&](ggml_tensor * m, int64_t kv_ext) {
        std::vector<ggml_fp16_t> mv(size_t(kv_ext) * nq);
        for (int64_t iq = 0; iq < nq; ++iq) {
            const int64_t visible = kv_valid - nq + iq; // causal over the LIVE rows
            for (int64_t ik = 0; ik < kv_ext; ++ik) {
                const bool all_masked_row = dead_row && iq == 0;
                mv[size_t(iq)*kv_ext + ik] = ggml_fp32_to_fp16(
                        (!all_masked_row && ik <= visible) ? 0.0f : -INFINITY);
            }
        }
        ggml_backend_tensor_set(m, mv.data(), 0, mv.size()*sizeof(ggml_fp16_t));
    };
    fill_mask(mask_v, kv_valid);
    if (pad > 0) { fill_mask(mask_p, kv_pad); }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o_q4v);
    ggml_build_forward_expand(gf, o_f16v);
    if (pad > 0) {
        ggml_build_forward_expand(gf, o_q4p);
        ggml_build_forward_expand(gf, o_f16p);
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("INVALID compute failed kv=%lld\n", (long long) kv_valid);
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }

    const size_t o_n = size_t(hsk) * nq * nhq;
    const size_t l_n = size_t(nhq) * nq;
    result r;
    r.o_q4v.resize(o_n); r.o_f16v.resize(o_n);
    r.l_q4v.resize(l_n); r.l_f16v.resize(l_n);
    ggml_backend_tensor_get(o_q4v,  r.o_q4v.data(),  0, o_n*sizeof(float));
    ggml_backend_tensor_get(o_f16v, r.o_f16v.data(), 0, o_n*sizeof(float));
    ggml_backend_tensor_get(l_q4v,  r.l_q4v.data(),  0, l_n*sizeof(float));
    ggml_backend_tensor_get(l_f16v, r.l_f16v.data(), 0, l_n*sizeof(float));
    if (pad > 0) {
        r.o_q4p.resize(o_n); r.o_f16p.resize(o_n);
        r.l_q4p.resize(l_n); r.l_f16p.resize(l_n);
        ggml_backend_tensor_get(o_q4p,  r.o_q4p.data(),  0, o_n*sizeof(float));
        ggml_backend_tensor_get(o_f16p, r.o_f16p.data(), 0, o_n*sizeof(float));
        ggml_backend_tensor_get(l_q4p,  r.l_q4p.data(),  0, l_n*sizeof(float));
        ggml_backend_tensor_get(l_f16p, r.l_f16p.data(), 0, l_n*sizeof(float));
    }

    double mo, ro, ml, rl; int64_t io, il;
    max_abs_diff(r.o_q4v, r.o_f16v, mo, ro, io);
    max_abs_diff(r.l_q4v, r.l_f16v, ml, rl, il);
    printf("hsk=%3lld nh=%lld nhq=%2lld kv_valid=%4lld kv_pad=%4lld nq=%3lld %-6s | q4v-vs-f16v: O max=%.3e rms=%.3e | LSE max=%.3e | %s\n",
           (long long) hsk, (long long) nh, (long long) nhq,
           (long long) kv_valid, (long long) kv_pad, (long long) nq, folded ? "folded" : "4d",
           mo, ro, ml, (mo <= tol_same && ml <= tol_same) ? "OK" : "FAIL");
    if (mo > tol_same || ml > tol_same) { failures++; }

    if (pad > 0) {
        double mo2, ro2, ml2, rl2; int64_t io2, il2;
        max_abs_diff(r.o_q4p, r.o_f16p, mo2, ro2, io2);
        max_abs_diff(r.l_q4p, r.l_f16p, ml2, rl2, il2);
        printf("         padded extent            : q4p-vs-f16p: O max=%.3e | LSE max=%.3e | %s\n",
               mo2, ml2, (mo2 <= tol_same && ml2 <= tol_same) ? "OK" : "FAIL");
        if (mo2 > tol_same || ml2 > tol_same) { failures++; }

        double mc1, rc1, mc2, rc2; int64_t ic1, ic2;
        max_abs_diff(r.o_q4v, r.o_f16p, mc1, rc1, ic1);
        max_abs_diff(r.o_q4p, r.o_f16v, mc2, rc2, ic2);
        printf("         CROSS valid-vs-padded O : q4v-vs-f16p max=%.3e | q4p-vs-f16v max=%.3e (diagnostic)\n",
               mc1, mc2);
        // q4p reads stale tail bytes, f16p reads the duplicated row-0 tail: if the
        // -inf mask is neutral, these must still agree despite the value difference.
        double mt, rt, lt; int64_t it;
        max_abs_diff(r.o_q4p, r.o_f16p, mt, rt, it);
        max_abs_diff(r.l_q4p, r.l_f16p, lt, rt, it);
        printf("         TAIL-VALUE difference O  : q4p(stale) vs f16p(row0-dup) max=%.3e | LSE max=%.3e | %s\n",
               mt, lt, (mt <= tol_same && lt <= tol_same) ? "OK(mask neutral)" : "DIFFERS(mask not neutral?)");
        if (mt > tol_same || lt > tol_same) { failures++; }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main() {
    ggml_backend_t backend = open_vulkan0();
    if (!backend) {
        printf("SKIP: no Vulkan0 device\n");
        return 0;
    }
    int failures = 0;
    const double tol_same = 2e-5;
    for (int64_t kv : { 640, 952, 1024, 1152, 1536 }) {
        for (int64_t nq : { 1, 2, 64 }) {
            run_case(backend, kv, nq, 0,   tol_same, failures, false);
            run_case(backend, kv, nq, 128, tol_same, failures, false);
            run_case(backend, kv, nq, 128, tol_same, failures, true);
        }
    }
    // Model-shape cases matching the integrated overflow layers (il=7 eval 14/15):
    // head_dim 256, 4 KV heads, 24 query heads, padded extent 1024 (952 live).
    {
        const double tol = 2e-5;
        run_case(backend,  952,  56, 72, tol, failures, true, 256, 4, 24);
        run_case(backend,  952, 256, 72, tol, failures, true, 256, 4, 24);
        run_case(backend, 1024, 256,  0, tol, failures, true, 256, 4, 24);
        run_case(backend,  896, 256,128, tol, failures, true, 256, 4, 24);
    }
    ggml_backend_free(backend);
    if (failures != 0) {
        printf("FAILED: %d same-extent comparison(s) above tolerance %.0e\n", failures, tol_same);
        return 1;
    }
    printf("OK: Q4_0 and F16 agree at equal extent; cross-extent deltas reported above\n");
    return 0;
}
