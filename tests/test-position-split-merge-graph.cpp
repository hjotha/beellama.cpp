// F3 core — two FlashAttention ranges merged by log-sum-exp in one GGML graph
// (plan §3.3 and §3.4).
//
// Per full-attention layer the product shape is:
//
//   o1, lse1 = FA(Q, K_local[0,P),  V_local,  mask_local)      -> CUDA (KVarN)
//   o2, lse2 = FA(Q, K_ovf[P,C),    V_ovf,    mask_causal_ovf) -> Vulkan (Q4)
//   o  = (o1*w1 + o2*w2) / (w1 + w2),  w = exp(lse - max(lse1,lse2))
//
// Layout contract (ggml/src/ggml.c):
//   Q is ne = {D, n_tokens, n_heads}, K/V are ne = {D, n_kv, n_kv_heads},
//   O is ne = {D, n_q_heads, n_q, n_batch}, LSE is ne = {n_q_heads, n_q, n_batch}.
// Host buffers are laid out exactly like those contiguous tensors and the FP64
// oracle reads them with the same strides (review 5, R3).
//
// The merge uses only operations that already exist. The weights are the two
// rows of a softmax over the concatenated LSE, so an empty range (LSE = -inf)
// gets exactly zero weight with no infinity arithmetic, and the merged LSE is
// mx + log(w1 + w2), which is -inf exactly when both ranges are empty while a
// single empty range reduces to the other range's LSE (review 5, R5).
//
// Every requirement is a gate: a violated limit, a missing required backend or a
// skipped mandatory shape makes this test exit non-zero (review 5, R6).

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"
#include "llama-kvarn.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

static int g_checks = 0;
static int g_failures = 0;
static std::string g_case;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::printf("  FAIL [%s] line %d: %s\n", g_case.c_str(), __LINE__, #cond); \
    } \
} while (0)

// The product reads LLAMA_OVF_QWHT_CUDA at graph-build time; the test sets it
// per variant so the binary is hermetic to the caller's environment.
static void set_env_llama_ovf_qwht_cuda(const char * value) {
#ifdef _WIN32
    _putenv_s("LLAMA_OVF_QWHT_CUDA", value != nullptr ? value : "");
#else
    if (value != nullptr) {
        setenv("LLAMA_OVF_QWHT_CUDA", value, 1);
    } else {
        unsetenv("LLAMA_OVF_QWHT_CUDA");
    }
#endif
}

static const double kGateNrmseO       = 1e-3;   // plan 4.1: enforced cumulatively
                                                  // on the merged result (R4); the
                                                  // base/ovf reports attribute
                                                  // backend fidelity separately
static const double kGateMergeExactO   = 1e-4;   // merge arithmetic vs backend outputs
static const double kGateMergeExactLse = 1e-4;
static const double kGateMaxAbsO  = 1e-2;   // plan 4.1
static const double kGateMaxAbsLse = 1e-2;  // plan 4.1
// No invented LSE bound (review 6, R4): with max subtraction every exp arg is
// <= 0, so exp can underflow (exact 0, the correct empty weight) but never
// overflow, for any finite LSE. -inf inputs stay -inf through the shifts.

// ---------------------------------------------------------------- oracle ----

struct ref_out {
    std::vector<double> o;        // ne {D, H, Q, B}
    std::vector<double> lse;      // ne {H, Q, B}
    std::vector<char>   empty;    // 1 => fully masked row: O = 0, LSE = -inf
};

// FP64 attention over the exact bytes each range consumed, with the same
// visibility rules as the graph.
static ref_out oracle(int D, int n_q, int n_qh, int n_kvh,
        const std::vector<float> & q,        // ne {D, n_q, n_qh, 1}
        const std::vector<float> & k1,       // ne {D, n_kv1, n_kvh, 1}
        const std::vector<float> & v1,
        int n_kv1,
        const std::vector<float> & k2,       // ne {D, n_kv2, n_kvh, 1}
        const std::vector<float> & v2,
        int n_kv2,
        bool mask1_visible,
        int mask2_mode) {     // 0 = all, 1 = causal (s <= t + n_kv2 - n_q), 2 = none
    const int gqa = n_qh / n_kvh;
    const double scale = 1.0 / std::sqrt(double(D));
    const size_t n_scores = size_t(n_kv1) + size_t(n_kv2);
    ref_out out;
    out.o.assign(size_t(D) * n_qh * n_q, 0.0);
    out.lse.assign(size_t(n_qh) * n_q, -INFINITY);
    out.empty.assign(size_t(n_qh) * n_q, 1);

    std::vector<double> scores(n_scores);
    for (int h = 0; h < n_qh; ++h) {
        const int kh = h / gqa;
        for (int t = 0; t < n_q; ++t) {
            const size_t qoff = (size_t(h) * n_q + t) * D;
            double max_score = -INFINITY;
            int visible = 0;
            for (int s = 0; s < n_kv1; ++s) {
                if (!mask1_visible) {
                    scores[size_t(s)] = -INFINITY;
                    continue;
                }
                const size_t koff = (size_t(s) + size_t(kh) * n_kv1) * D;
                double acc = 0.0;
                for (int d = 0; d < D; ++d) {
                    acc += double(q[qoff + d]) * double(k1[koff + d]);
                }
                scores[size_t(s)] = acc * scale;
                max_score = std::max(max_score, scores[size_t(s)]);
                ++visible;
            }
            for (int s = 0; s < n_kv2; ++s) {
                const bool vis = mask2_mode == 0 ? true
                                : mask2_mode == 2 ? false
                                : s <= t + n_kv2 - n_q;
                if (!vis) {
                    scores[size_t(n_kv1) + size_t(s)] = -INFINITY;
                    continue;
                }
                const size_t koff = (size_t(s) + size_t(kh) * n_kv2) * D;
                double acc = 0.0;
                for (int d = 0; d < D; ++d) {
                    acc += double(q[qoff + d]) * double(k2[koff + d]);
                }
                scores[size_t(n_kv1) + size_t(s)] = acc * scale;
                max_score = std::max(max_score, scores[size_t(n_kv1) + size_t(s)]);
                ++visible;
            }
            // O and LSE share the FA axis order ne = {D, n_qh, n_q}: the head
            // axis is outermost, so a row is h + t*n_qh.
            const size_t row = size_t(h) + size_t(t) * n_qh;
            if (visible == 0) {
                continue;
            }
            double denom = 0.0;
            for (size_t i = 0; i < n_scores; ++i) {
                if (scores[i] == -INFINITY) {
                    continue;
                }
                denom += std::exp(scores[i] - max_score);
            }
            for (int d = 0; d < D; ++d) {
                double acc = 0.0;
                for (size_t i = 0; i < n_scores; ++i) {
                    if (scores[i] == -INFINITY) {
                        continue;
                    }
                    const bool local = i < size_t(n_kv1);
                    const int s = local ? int(i) : int(i - size_t(n_kv1));
                    const std::vector<float> & vv = local ? v1 : v2;
                    const size_t voff = (size_t(s) + size_t(kh) * (local ? n_kv1 : n_kv2)) * D;
                    acc += std::exp(scores[i] - max_score) * double(vv[voff + d]);
                }
                out.o[(size_t(h) + size_t(t) * n_qh) * D + d] = acc / denom;
            }
            out.lse[row] = max_score + std::log(denom);
            out.empty[row] = 0;
        }
    }
    return out;
}

static double nrmse_of(const std::vector<float> & got, const std::vector<double> & ref) {
    if (got.size() != ref.size() || ref.empty()) {
        return INFINITY;
    }
    double se = 0.0;
    double sr = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double diff = double(got[i]) - ref[i];
        se += diff * diff;
        sr += ref[i] * ref[i];
    }
    const double rms_ref = std::max(std::sqrt(sr / double(ref.size())), 1e-6);
    return std::sqrt(se / double(got.size())) / rms_ref;
}

static double nrmse_ff(const std::vector<float> & got, const std::vector<float> & ref) {
    if (got.size() != ref.size() || ref.empty()) {
        return INFINITY;
    }
    double se = 0.0;
    double sr = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double diff = double(got[i]) - double(ref[i]);
        se += diff * diff;
        sr += double(ref[i]) * double(ref[i]);
    }
    const double rms_ref = std::max(std::sqrt(sr / double(ref.size())), 1e-6);
    return std::sqrt(se / double(got.size())) / rms_ref;
}

static double max_abs_of(const std::vector<float> & got, const std::vector<double> & ref) {
    double m = 0.0;
    for (size_t i = 0; i < got.size() && i < ref.size(); ++i) {
        m = std::max(m, std::fabs(double(got[i]) - ref[i]));
    }
    return m;
}

// R4: shift invariance of the merge formula, evaluated in fp32 mirroring the
// graph ops (clamp/abs-max/exp/log), over large positive/negative LSEs, -inf,
// and different per-pair weights (standing in for per-head/query weights).
// O must be invariant under a common shift, LSE must shift by it, empties
// must reduce exactly, and nothing may produce NaN. A direct FP64
// max-subtracted reference validates the fp32 rounding.
static void check_merge_formula_invariance() {
    g_case = "merge formula invariance";
    const float pairs[][2] = {
        {5.0f, 3.0f}, {100.0f, 90.0f}, {-100.0f, -90.0f},
        {1000.0f, 990.0f}, {-1000.0f, -990.0f},
        {0.0f, -INFINITY}, {-INFINITY, 5.0f}, {-INFINITY, -INFINITY},
        {88.0f, 87.0f}, {-88.0f, -87.0f},
    };
    const float shifts[] = {0.0f, 50.0f, -50.0f, 500.0f, -500.0f};
    const float weights[][2] = {{0.7f, -0.3f}, {-1.3f, 2.1f}};
    auto merge_f32 = [](float a, float b, float x1, float x2, float & o, float & lse) {
        const float c1 = std::fmin(std::fmax(a, -1e6f), 1e30f);
        const float c2 = std::fmin(std::fmax(b, -1e6f), 1e30f);
        const float mx = (c1 + c2 + std::fabs(c1 - c2)) * 0.5f;
        const float w1 = std::exp(a - mx);
        const float w2 = std::exp(b - mx);
        const float den = w1 + w2;
        o = (x1 * w1 + x2 * w2) / (den + 1e-30f);
        lse = mx + std::log(den);
    };
    auto merge_f64 = [](double a, double b, double x1, double x2, double & o, double & lse) {
        const double mx = std::max(a, b);
        const double w1 = std::exp(a - mx);
        const double w2 = std::exp(b - mx);
        const double den = w1 + w2;
        o = (x1 * w1 + x2 * w2) / (den + 1e-30);
        lse = mx + std::log(den);
    };
    for (const auto & pr : pairs) {
        for (const auto & w : weights) {
            float o0 = 0.0f, lse0 = 0.0f;
            merge_f32(pr[0], pr[1], w[0], w[1], o0, lse0);
            const bool both_empty =
                std::isinf(pr[0]) && std::isinf(pr[1]) && pr[0] < 0.0f && pr[1] < 0.0f;
            if (both_empty) {
                CHECK(o0 == 0.0f);
                CHECK(std::isinf(lse0) && lse0 < 0.0f);
            } else {
                CHECK(std::isfinite(o0));
                CHECK(std::isfinite(lse0));
                double ro = 0.0, rlse = 0.0;
                merge_f64(pr[0], pr[1], w[0], w[1], ro, rlse);
                CHECK(std::fabs(double(o0) - ro) <= 1e-5);
                CHECK(std::fabs(double(lse0) - rlse) <= 1e-4);
            }
            for (float c : shifts) {
                float os = 0.0f, lses = 0.0f;
                merge_f32(pr[0] + c, pr[1] + c, w[0], w[1], os, lses);
                if (both_empty) {
                    CHECK(os == 0.0f);
                    CHECK(std::isinf(lses) && lses < 0.0f);
                    continue;
                }
                CHECK(std::isfinite(os));
                CHECK(std::isfinite(lses));
                CHECK(std::fabs(double(os) - double(o0)) <= 1e-5);
                CHECK(std::fabs(double(lses) - (double(lse0) + double(c))) <= 1e-3);
            }
        }
    }
}

// R2: exact merge arithmetic as a pure function over the three backend
// outputs, so the same logic gates the product and proves the negative
// controls below. O is ne = {D, H, Q, B}, so row (t,h) starts at
// (t*n_qh + h)*D (the old checker used (h*n_q + t)*D, which only coincides
// for n_q == 1). Every input element must be finite here: the callers only
// invoke this on fully visible ranges, and any NaN/+Inf must fail loudly
// instead of being skipped or averaged away.
struct exactness_result {
    double max_o = 0.0;
    double max_lse = 0.0;
    bool all_finite = true;
};

static exactness_result check_merge_exactness(int D, int n_q, int n_qh,
        const std::vector<float> & base_o, const std::vector<float> & base_lse,
        const std::vector<float> & ovf_o, const std::vector<float> & ovf_lse,
        const std::vector<float> & m_o, const std::vector<float> & m_lse,
        const std::vector<char> & merged_empty) {
    exactness_result r;
    if (base_o.size() != m_o.size() || ovf_o.size() != m_o.size() ||
            base_lse.size() != m_lse.size() || ovf_lse.size() != m_lse.size() ||
            m_o.empty() || m_lse.empty() || merged_empty.size() != m_lse.size()) {
        r.all_finite = false;
        return r;
    }
    for (int t = 0; t < n_q; ++t) {
        for (int h = 0; h < n_qh; ++h) {
            const size_t li = size_t(h) + size_t(t) * n_qh;
            const double l1 = base_lse[li];
            const double l2 = ovf_lse[li];
            // LSE is exactly -inf iff its range is empty for this row: a
            // non-empty row always has a finite log-sum-exp, and backends
            // publish exactly -inf (not just very negative) for empty rows.
            const bool e1 = std::isinf(l1) && l1 < 0.0;
            const bool e2 = std::isinf(l2) && l2 < 0.0;
            if (!std::isfinite(l1) && !e1) { r.all_finite = false; return r; }
            if (!std::isfinite(l2) && !e2) { r.all_finite = false; return r; }
            if (merged_empty[li]) {
                // Both ranges empty: exact contract, no arithmetic.
                if (!e1 || !e2) { r.all_finite = false; return r; }
                for (int d = 0; d < D; ++d) {
                    const size_t oi = (size_t(t) * n_qh + h) * D + d;
                    if (m_o[oi] != 0.0f) { r.all_finite = false; return r; }
                }
                if (!(std::isinf(m_lse[li]) && m_lse[li] < 0.0f)) {
                    r.all_finite = false;
                    return r;
                }
                continue;
            }
            if (e1 && e2) { r.all_finite = false; return r; }
            if (!std::isfinite(m_lse[li])) { r.all_finite = false; return r; }
            // At least one range is live, so mx is finite and lse - mx can
            // only be finite or -inf: no inf-inf anywhere.
            const double mx = std::max(l1, l2);
            if (!std::isfinite(mx)) { r.all_finite = false; return r; }
            const double w1 = std::exp(l1 - mx);
            const double w2 = std::exp(l2 - mx);
            const double den = w1 + w2;
            if (!(den > 0.0) || !std::isfinite(den)) { r.all_finite = false; return r; }
            const double le = mx + std::log(den);
            if (!std::isfinite(le)) { r.all_finite = false; return r; }
            r.max_lse = std::max(r.max_lse, std::fabs(le - double(m_lse[li])));
            for (int d = 0; d < D; ++d) {
                // O is ne = {D, H, Q, B}: row (t,h) starts at (t*n_qh + h)*D.
                const size_t oi = (size_t(t) * n_qh + h) * D + d;
                // An empty range contributes exactly nothing: its O row must
                // be exactly 0 (plan contract for fully masked rows), so even
                // 0 * garbage cannot leak NaN into the numerator.
                if (e1 && base_o[oi] != 0.0f) { r.all_finite = false; return r; }
                if (e2 && ovf_o[oi] != 0.0f) { r.all_finite = false; return r; }
                if (!e1 && !std::isfinite(base_o[oi])) { r.all_finite = false; return r; }
                if (!e2 && !std::isfinite(ovf_o[oi])) { r.all_finite = false; return r; }
                if (!std::isfinite(m_o[oi])) { r.all_finite = false; return r; }
                const double bo = e1 ? 0.0 : double(base_o[oi]);
                const double vo = e2 ? 0.0 : double(ovf_o[oi]);
                const double ex = (bo * w1 + vo * w2) / den;
                if (!std::isfinite(ex)) { r.all_finite = false; return r; }
                r.max_o = std::max(r.max_o, std::fabs(ex - double(m_o[oi])));
            }
        }
    }
    if (!std::isfinite(r.max_o) || !std::isfinite(r.max_lse)) {
        r.all_finite = false;
    }
    return r;
}

// ------------------------------------------------------------ quantisation ----

// Packs a host range into `type` and returns the dequantized values of exactly
// those bytes. Rows advance by ggml_row_size(type, D) - one *block* advance
// corrupts every row after the first (review 5, R5).
static void pack_range(ggml_type type, int D, const std::vector<float> & src,
        std::vector<uint8_t> & bytes, std::vector<float> & dequant) {
    const size_t n_rows = src.size() / size_t(D);
    const size_t row_bytes = ggml_row_size(type, D);
    bytes.assign(n_rows * row_bytes, 0);
    dequant.assign(src.size(), 0.0f);
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            tmp[i] = ggml_fp32_to_fp16(src[i]);
            dequant[i] = ggml_fp16_to_fp32(tmp[i]);
        }
        std::memcpy(bytes.data(), tmp.data(), bytes.size());
        return;
    }
    ggml_quantize_chunk(type, src.data(), bytes.data(), 0, int64_t(n_rows), D, nullptr);
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    for (size_t r = 0; r < n_rows; ++r) {
        tr->to_float(bytes.data() + r * row_bytes, dequant.data() + r * size_t(D), D);
    }
}

// ------------------------------------------------------------------- run ----

struct merge_result {
    bool declined = false;
    std::string reason;
    std::vector<float> o;
    std::vector<float> lse;
    int  splits = 0;
    int  overflow_splits = 0;
    std::vector<float> q_rot;
    std::vector<uint8_t> k2_final;
    std::vector<uint8_t> v2_final;
    int  attn_nodes = 0;
    int  attn_nodes_on_overflow = 0;
    int  nodes_on_overflow = 0;
    bool fa1_on_requested = false;
    bool fa2_on_requested = false;
};

struct eval_counters {
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_t overflow = nullptr;
    int total = 0;
    int on_overflow = 0;
    int attn_on_overflow = 0;
};

static bool eval_count_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    eval_counters * c = (eval_counters *) user_data;
    if (ask || c->sched == nullptr) {
        return true;
    }
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(c->sched, t);
    ++c->total;
    if (backend == c->overflow) {
        ++c->on_overflow;
        if (t->op == GGML_OP_FLASH_ATTN_EXT) {
            ++c->attn_on_overflow;
        }
    }
    return true;
}

// Builds and runs the two-range graph. `mask1_visible`/`mask2_mode` drive the
// visibility of each range; when the overflow range is absent (n_kv2 == 0) the
// graph contains a single attention node and no merge arithmetic at all.
static merge_result run_merge(ggml_backend_t local, ggml_backend_t overflow,
        int D, int n_q, int n_qh, int n_kvh,
        int n_kv1, int n_kv2, ggml_type ovf_type,
        bool mask1_visible, int mask2_mode,
        const std::vector<float> & q,
        const std::vector<float> & k1, const std::vector<float> & v1,
        const std::vector<float> & k2, const std::vector<float> & v2,
        const std::vector<uint8_t> & k1_bytes, const std::vector<uint8_t> & v1_bytes,
        const std::vector<uint8_t> & k2_bytes, const std::vector<uint8_t> & v2_bytes,
        ggml_type local_type = GGML_TYPE_F16) {
    // In the single-range form the caller passes one set twice; only the range
    // actually built is read.
    (void) k1; (void) v1; (void) k2; (void) v2;
    merge_result res;
    const bool with_overflow = n_kv2 > 0;

    ggml_init_params params = { 128 * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        res.declined = true;
        res.reason = "context allocation failed";
        return res;
    }

    // ggml contract: Q ne = {D, n_tokens, n_heads} and mask ne =
    // {n_kv, n_head_kv, n_tokens, 1}; ggml asserts
    // q->ne[2] % mask->ne[2] == 0, so ne[2] is the head axis.
    ggml_tensor * q_t  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_q, n_qh, 1);
    ggml_tensor * k1_t = ggml_new_tensor_4d(ctx, local_type, D, n_kv1, n_kvh, 1);
    ggml_tensor * v1_t = ggml_new_tensor_4d(ctx, local_type, D, n_kv1, n_kvh, 1);
    // R1: never reinterpret bytes across formats. The caller selects the exact
    // local-range storage type (F16 normally, Q4_0 for the overflow baseline),
    // and every host upload is size-checked before touching the backend.
    if (q.size() * sizeof(float) != ggml_nbytes(q_t) ||
            k1_bytes.size() != ggml_nbytes(k1_t) ||
            v1_bytes.size() != ggml_nbytes(v1_t)) {
        ggml_free(ctx);
        res.declined = true;
        res.reason = "host upload byte size does not match local range tensors";
        return res;
    }
    // mask ne = {n_kv, n_q, n_kv_heads, 1}: ggml asserts q->ne[2] % mask->ne[2]
    // == 0, so ne[2] must be a head axis.
    ggml_tensor * m1_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv1, n_q, n_kvh, 1);
    ggml_tensor * fa1 = ggml_flash_attn_ext(ctx, q_t, k1_t, v1_t, m1_t,
            1.0f / std::sqrt(float(D)), 0.0f, 0.0f);
    // The plan gates are measured against an FP64 oracle, so the accumulator is
    // F32 (the product sets the same precision for the split graph).
    ggml_prec_set_acc(fa1, GGML_PREC_F32);
    ggml_tensor * lse1 = ggml_flash_attn_ext_lse_out(ctx, fa1);

    ggml_tensor * o = nullptr;
    ggml_tensor * lse_merged = nullptr;

    if (with_overflow) {
        ggml_tensor * k2_t = ggml_new_tensor_4d(ctx, ovf_type, D, n_kv2, n_kvh, 1);
        ggml_tensor * v2_t = ggml_new_tensor_4d(ctx, ovf_type, D, n_kv2, n_kvh, 1);
        ggml_tensor * m2_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv2, n_q, n_kvh, 1);
        ggml_tensor * fa2 = ggml_flash_attn_ext(ctx, q_t, k2_t, v2_t, m2_t,
                1.0f / std::sqrt(float(D)), 0.0f, 0.0f);
        ggml_prec_set_acc(fa2, GGML_PREC_F32);
        ggml_tensor * lse2 = ggml_flash_attn_ext_lse_out(ctx, fa2);

        const bool ok1 = ggml_backend_supports_op(local, fa1) &&
                         ggml_backend_supports_op(local, lse1);
        const bool ok2 = ggml_backend_supports_op(overflow, fa2) &&
                         ggml_backend_supports_op(overflow, lse2);
        if (!ok1 || !ok2) {
            ggml_free(ctx);
            res.declined = true;
            res.reason = std::string("supports_op local=") + (ok1 ? "1" : "0") +
                         " overflow=" + (ok2 ? "1" : "0");
            return res;
        }

        // Stable merge (review 6, R4): shift by the max so every exp arg is
        // <= 0 and can never overflow, using only existing ops. M is the max
        // of clamped LSEs, hence always finite, so the abs-trick max never
        // sees inf-inf (which is NaN); weights use the ORIGINAL LSEs, so -inf
        // stays -inf, empty ranges get exactly zero weight with den == 0, and
        // the exact O=0/LSE=-inf contract holds with no epsilon in LSE.
        // The lower clamp is -1e6 (not -1e30): the abs trick
        // (a+b+|a-b|)/2 absorbs the smaller operand in fp32 when the operands
        // differ by more than 2^24, so the clamp bound must stay within fp32
        // exact range of real LSEs; anything below -1e6 with visible keys is
        // unrepresentable weight anyway (exp underflows identically). An M
        // rounding error cancels exactly (common factor of both weights).
        ggml_tensor * c1 = ggml_clamp(ctx, lse1, -1e6f, 1e30f);
        ggml_tensor * c2 = ggml_clamp(ctx, lse2, -1e6f, 1e30f);
        ggml_tensor * mx = ggml_scale(ctx,
                ggml_add(ctx, ggml_add(ctx, c1, c2), ggml_abs(ctx, ggml_sub(ctx, c1, c2))),
                0.5f);
        ggml_tensor * e1 = ggml_exp(ctx, ggml_sub(ctx, lse1, mx));
        ggml_tensor * e2 = ggml_exp(ctx, ggml_sub(ctx, lse2, mx));
        // LSE is ne = {H, Q, B}; O is ne = {D, H, Q, B}. A reshape to
        // {1, H, Q, B} keeps the same flat order (the leading axis is 1), so the
        // weights broadcast over D with the tensor's own layout (review 5, R4).
        ggml_tensor * w1 = ggml_reshape_4d(ctx, e1, 1, n_qh, n_q, 1);
        ggml_tensor * w2 = ggml_reshape_4d(ctx, e2, 1, n_qh, n_q, 1);
        ggml_tensor * num = ggml_add(ctx, ggml_mul(ctx, fa1, w1), ggml_mul(ctx, fa2, w2));
        ggml_tensor * den = ggml_add(ctx, e1, e2);
        // O needs a strictly positive denominator so two empty ranges give 0
        // instead of NaN; the LSE must NOT have the epsilon, so that two empty
        // ranges publish exactly -inf (review 5, R5).
        ggml_tensor * eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        const float eps_value = 1e-30f;
        ggml_tensor * den_o = ggml_reshape_4d(ctx,
                ggml_add(ctx, den, eps), 1, n_qh, n_q, 1);
        o = ggml_div(ctx, num, den_o);
        // M is finite by construction (max of clamped LSEs), so M + log(den)
        // is exactly -inf iff den == 0, i.e. iff both ranges are empty.
        lse_merged = ggml_add(ctx, mx, ggml_log(ctx, den));

        ggml_backend_t backends[3] = { overflow, local, ggml_backend_cpu_init() };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 3, 8192, false, true);
        ggml_backend_sched_set_tensor_backend(sched, fa1, local);
        ggml_backend_sched_set_tensor_backend(sched, lse1, local);
        ggml_backend_sched_set_tensor_backend(sched, k1_t, local);
        ggml_backend_sched_set_tensor_backend(sched, v1_t, local);
        ggml_backend_sched_set_tensor_backend(sched, q_t,  local);
        ggml_backend_sched_set_tensor_backend(sched, m1_t, local);
        ggml_backend_sched_set_tensor_backend(sched, fa2, overflow);
        ggml_backend_sched_set_tensor_backend(sched, lse2, overflow);
        ggml_backend_sched_set_tensor_backend(sched, k2_t, overflow);
        ggml_backend_sched_set_tensor_backend(sched, v2_t, overflow);
        ggml_backend_sched_set_tensor_backend(sched, m2_t, overflow);
        ggml_backend_sched_set_tensor_backend(sched, o, local);
        ggml_backend_sched_set_tensor_backend(sched, lse_merged, local);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);
        ggml_build_forward_expand(gf, lse_merged);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
            res.declined = true;
            res.reason = "sched_alloc_graph failed";
            return res;
        }

        eval_counters counters;
        counters.sched = sched;
        counters.overflow = overflow;
        ggml_backend_sched_set_eval_callback(sched, eval_count_cb, &counters);

        ggml_backend_tensor_set(eps, &eps_value, 0, sizeof(float));
        ggml_backend_tensor_set(q_t, q.data(), 0, ggml_nbytes(q_t));
        if (k2_bytes.size() != ggml_nbytes(k2_t) || v2_bytes.size() != ggml_nbytes(v2_t)) {
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
            res.declined = true;
            res.reason = "host upload byte size does not match overflow range tensors";
            return res;
        }
        ggml_backend_tensor_set(k1_t, k1_bytes.data(), 0, ggml_nbytes(k1_t));
        ggml_backend_tensor_set(v1_t, v1_bytes.data(), 0, ggml_nbytes(v1_t));
        ggml_backend_tensor_set(k2_t, k2_bytes.data(), 0, ggml_nbytes(k2_t));
        ggml_backend_tensor_set(v2_t, v2_bytes.data(), 0, ggml_nbytes(v2_t));
        {
            std::vector<ggml_fp16_t> m1_data(size_t(n_kv1) * n_kvh * n_q, ggml_fp32_to_fp16(-INFINITY));
            for (int t = 0; t < n_q; ++t) {
                for (int kh = 0; kh < n_kvh; ++kh) {
                    for (int s = 0; s < n_kv1; ++s) {
                        const size_t off = (size_t(kh) * n_q + t) * n_kv1 + s;
                        m1_data[off] = ggml_fp32_to_fp16(mask1_visible ? 0.0f : -INFINITY);
                    }
                }
            }
            ggml_backend_tensor_set(m1_t, m1_data.data(), 0, ggml_nbytes(m1_t));
            std::vector<ggml_fp16_t> m2_data(size_t(n_kv2) * n_kvh * n_q, ggml_fp32_to_fp16(-INFINITY));
            for (int t = 0; t < n_q; ++t) {
                for (int kh = 0; kh < n_kvh; ++kh) {
                    for (int s = 0; s < n_kv2; ++s) {
                        // R3: absolute causal position. Queries sit at the end
                        // of the concatenated [local, overflow] sequence, so
                        // query t sees overflow keys s <= t + n_kv2 - n_q --
                        // exactly the oracle rule. The old `s <= t` measured a
                        // different rule (NRMSE 0.5) and is not a backend bug.
                        const bool vis = mask2_mode == 0 ? true
                                        : mask2_mode == 2 ? false
                                        : s <= t + n_kv2 - n_q;
                        const size_t off = (size_t(kh) * n_q + t) * n_kv2 + s;
                        m2_data[off] = ggml_fp32_to_fp16(vis ? 0.0f : -INFINITY);
                    }
                }
            }
            ggml_backend_tensor_set(m2_t, m2_data.data(), 0, ggml_nbytes(m2_t));
        }

        int attn = 0;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            if (ggml_graph_node(gf, i)->op == GGML_OP_FLASH_ATTN_EXT) {
                ++attn;
                const ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched, ggml_graph_node(gf, i));
                if (ggml_graph_node(gf, i) == fa1) {
                    res.fa1_on_requested = (be == local);
                } else if (ggml_graph_node(gf, i) == fa2) {
                    res.fa2_on_requested = (be == overflow);
                }
            }
        }
        res.attn_nodes = attn;

        if (ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS) {
            res.o.assign(ggml_nelements(o), 0.0f);
            res.lse.assign(ggml_nelements(lse_merged), 0.0f);
            ggml_backend_tensor_get(o, res.o.data(), 0, ggml_nbytes(o));
            ggml_backend_tensor_get(lse_merged, res.lse.data(), 0, ggml_nbytes(lse_merged));
            res.attn_nodes_on_overflow = counters.attn_on_overflow;
            res.nodes_on_overflow = counters.on_overflow;
        }
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return res;
    }

    // Single range: no weights, no merge arithmetic, no second attention node.
    const bool ok1 = ggml_backend_supports_op(local, fa1) &&
                     ggml_backend_supports_op(local, lse1);
    if (!ok1) {
        ggml_free(ctx);
        res.declined = true;
        res.reason = "supports_op local=0 (single range)";
        return res;
    }
    o = fa1;
    lse_merged = lse1;
    ggml_backend_t backends[3] = { overflow, local, ggml_backend_cpu_init() };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 3, 8192, false, true);
    ggml_backend_sched_set_tensor_backend(sched, fa1, local);
    ggml_backend_sched_set_tensor_backend(sched, lse1, local);
    ggml_backend_sched_set_tensor_backend(sched, k1_t, local);
    ggml_backend_sched_set_tensor_backend(sched, v1_t, local);
    ggml_backend_sched_set_tensor_backend(sched, q_t,  local);
    ggml_backend_sched_set_tensor_backend(sched, m1_t, local);
    ggml_backend_sched_set_tensor_backend(sched, o, local);
    ggml_backend_sched_set_tensor_backend(sched, lse_merged, local);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    ggml_build_forward_expand(gf, lse_merged);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        res.declined = true;
        res.reason = "sched_alloc_graph failed (single range)";
        return res;
    }
    eval_counters counters;
    counters.sched = sched;
    counters.overflow = overflow;
    ggml_backend_sched_set_eval_callback(sched, eval_count_cb, &counters);
    ggml_backend_tensor_set(q_t, q.data(), 0, ggml_nbytes(q_t));
    ggml_backend_tensor_set(k1_t, k1_bytes.data(), 0, ggml_nbytes(k1_t));
    ggml_backend_tensor_set(v1_t, v1_bytes.data(), 0, ggml_nbytes(v1_t));
    {
        std::vector<ggml_fp16_t> m1_data(size_t(n_kv1) * n_kvh * n_q, ggml_fp32_to_fp16(-INFINITY));
        for (int t = 0; t < n_q; ++t) {
            for (int kh = 0; kh < n_kvh; ++kh) {
                for (int s = 0; s < n_kv1; ++s) {
                    // In the single-range case mask2_mode selects the visibility
                    // with the same absolute positions as the two-range graph:
                    // 0 = all, 1 = causal (s <= t + n_kv1 - n_q), 2 = none.
                    const bool vis = mask1_visible
                        ? (mask2_mode == 0 ? true : mask2_mode == 2 ? false : s <= t + n_kv1 - n_q)
                        : false;
                    const size_t off = (size_t(kh) * n_q + t) * n_kv1 + s;
                    m1_data[off] = ggml_fp32_to_fp16(vis ? 0.0f : -INFINITY);
                }
            }
        }
        ggml_backend_tensor_set(m1_t, m1_data.data(), 0, ggml_nbytes(m1_t));
    }
    int attn = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (ggml_graph_node(gf, i)->op == GGML_OP_FLASH_ATTN_EXT) {
            ++attn;
        }
    }
    res.attn_nodes = attn;
    if (ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS) {
        res.o.assign(ggml_nelements(o), 0.0f);
        res.lse.assign(ggml_nelements(lse_merged), 0.0f);
        ggml_backend_tensor_get(o, res.o.data(), 0, ggml_nbytes(o));
        ggml_backend_tensor_get(lse_merged, res.lse.data(), 0, ggml_nbytes(lse_merged));
        res.attn_nodes_on_overflow = counters.attn_on_overflow;
        res.nodes_on_overflow = counters.on_overflow;
    }
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return res;
}

// Real-pair composition (plan 4.1, msg57): the LOCAL range uses the product
// representation -- KVarN records (bits=4) stored on CUDA in the rotated
// domain and read back through the native record view, exactly like the
// llama-graph split builder, un-rotating O before the merge -- while the
// OVERFLOW range stays Q4_0 on Vulkan. The FP64 oracle is fed by the records
// decoded through a CPU `ggml_kvarn_materialize` side-path in the ORIGINAL
// domain, so the same kGateNrmseO applies to the composition the product runs.
// Requires CUDA (local) + Vulkan (overflow) + CPU (materialize side-path).
//
// schedule_mode:
//   0 = single expansion (no order/placement forcing)
//   1 = legacy expansion order + consolidated (forced) placement
//   2 = consolidated order + forced placement (the production decode); the Q
//       WHT stays on the Vulkan overflow backend, bit-identical to the
//       pre-consolidation graph
//   3 = legacy order with NO manual placements (scheduler's own placement)
//   4 = mode 2 with the Q WHT pinned to the local CUDA backend, mirroring
//       LLAMA_OVF_QWHT_CUDA=1; requires the env (set by the caller) and is
//       gated by tolerance, never bitwise
static merge_result run_merge_kvarn(ggml_backend_t local, ggml_backend_t overflow,
        int D, int n_q, int n_qh, int n_kvh,
        int n_kv1, int n_kv2,
        bool mask1_visible, int mask2_mode,
        const std::vector<float> & q,
        const std::vector<float> & k1, const std::vector<float> & v1,
        const std::vector<uint8_t> & k2_bytes, const std::vector<uint8_t> & v2_bytes,
        int schedule_mode = 0, int overflow_layout = 0, int store_mode = 0) {
    merge_result res;
    if (n_kv1 <= 0) {
        res.declined = true;
        res.reason = "KVarN real-pair run requires the local range";
        return res;
    }
    const bool with_overflow = n_kv2 > 0;
    const int bits = 4;
    const int stage_groups = 5;
    const int record_dim = D == 64 ? 64 : 128;
    const int slices = D / record_dim;
    const int record_heads = n_kvh * slices;
    const int groups_per_stream = std::max(4, (n_kv1 + 127) / 128);
    const int k_record_bytes = int(llama_kvarn_make_record_layout(record_dim, bits, false).record_bytes);
    const int v_record_bytes = int(llama_kvarn_make_record_layout(record_dim, bits, true).record_bytes);
    const int n_stream = 1;

    ggml_init_params params = { 512 * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        res.declined = true;
        res.reason = "context allocation failed";
        return res;
    }
    if (q.size() != size_t(D) * n_q * n_qh ||
            k1.size() != size_t(D) * n_kv1 * n_kvh ||
            v1.size() != size_t(D) * n_kv1 * n_kvh) {
        ggml_free(ctx);
        res.declined = true;
        res.reason = "host float size does not match the KVarN range shapes";
        return res;
    }

    ggml_tensor * q_t   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_q, n_qh, 1);
    ggml_tensor * q_rot = ggml_kvarn_wht(ctx, q_t, D);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_kv1);
    ggml_tensor * current_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, record_dim, record_heads, n_kv1 * n_stream);
    ggml_tensor * current_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, record_dim, record_heads, n_kv1 * n_stream);
    ggml_tensor * k_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, record_dim, record_heads, 128 * stage_groups * n_stream);
    ggml_tensor * v_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, record_dim, record_heads, 128 * stage_groups * n_stream);
    ggml_tensor * k_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, k_record_bytes, record_heads, groups_per_stream * n_stream);
    ggml_tensor * v_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, v_record_bytes, record_heads, groups_per_stream * n_stream);

    ggml_tensor * stored_k = ggml_kvarn_store(ctx, current_k, indices, k_stage, k_records, bits, 16, false, stage_groups);
    ggml_tensor * stored_v = ggml_kvarn_store(ctx, current_v, indices, v_stage, v_records, bits, 16, true, stage_groups);
    stored_k->op_params[3] = n_kv1;
    stored_v->op_params[3] = n_kv1;
    stored_k->op_params[5] = slices;
    stored_v->op_params[5] = slices;

    ggml_tensor * k_v = ggml_kvarn_view(ctx, k_records, stored_k, indices, n_kv1, 0, n_stream, bits, false, stage_groups);
    ggml_tensor * v_v = ggml_kvarn_view(ctx, v_records, stored_v, indices, n_kv1, 0, n_stream, bits, true, stage_groups);
    if (slices > 1) {
        k_v = ggml_reshape_4d(ctx, k_v, D, n_kvh, n_kv1, n_stream);
        v_v = ggml_reshape_4d(ctx, v_v, D, n_kvh, n_kv1, n_stream);
    }
    k_v = ggml_permute(ctx, k_v, 0, 2, 1, 3);
    v_v = ggml_permute(ctx, v_v, 0, 2, 1, 3);

    // Local all-visible F16 mask with ne[2] = 1, mirroring the native KVarN FA
    // runner (tests/test-kvarn.cpp); the overflow keeps the run_merge causal
    // mask shape.
    ggml_tensor * m1_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv1, n_q, 1, 1);
    ggml_tensor * fa1 = ggml_flash_attn_ext(ctx, q_rot, k_v, v_v, m1_t,
            1.0f / std::sqrt(float(D)), 0.0f, 0.0f);
    ggml_prec_set_acc(fa1, GGML_PREC_F32);
    ggml_tensor * lse1 = ggml_flash_attn_ext_lse_out(ctx, fa1);
    fa1->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN] = GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED;
    // Same domain rule as llama-graph.cpp: the rotated-domain O goes back to
    // the original V domain before merging with the (original-domain) overflow O.
    ggml_tensor * o1 = ggml_kvarn_wht(ctx, fa1, D);

    if (schedule_mode) { ggml_set_output(q_rot); }

    ggml_tensor * k2_t = nullptr;
    ggml_tensor * v2_t = nullptr;
    ggml_tensor * m2_t = nullptr;
    ggml_tensor * overflow_rows = nullptr;
    std::vector<ggml_tensor *> overflow_prep;
    ggml_tensor * store_k = nullptr;
    ggml_tensor * store_v = nullptr;
    ggml_tensor * cur_store_k = nullptr;
    ggml_tensor * cur_store_v = nullptr;
    ggml_tensor * store_idxs = nullptr;
    ggml_tensor * fa2 = nullptr;
    ggml_tensor * lse2 = nullptr;
    ggml_tensor * o = o1;
    ggml_tensor * lse_merged = lse1;
    ggml_tensor * eps = nullptr;
    const float eps_value = 1e-30f;
    if (!with_overflow) {
        const bool ok1s = ggml_backend_supports_op(local, fa1) && ggml_backend_supports_op(local, lse1);
        if (!ok1s) {
            ggml_free(ctx);
            res.declined = true;
            res.reason = "supports_op local=0 (single KVarN range)";
            return res;
        }
    } else {
        k2_t = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, D, n_kv2, n_kvh, 1);
        v2_t = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, D, n_kv2, n_kvh, 1);
        m2_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv2 + (schedule_mode ? 64 : 0), n_q, n_kvh, 1);
        if (k2_bytes.size() != ggml_nbytes(k2_t) || v2_bytes.size() != ggml_nbytes(v2_t)) {
            ggml_free(ctx);
            res.declined = true;
            res.reason = "host upload byte size does not match overflow range tensors";
            return res;
        }
        if (store_mode) {
            // The current token written into the overflow storage by an
            // in-graph store (SET_ROWS into k2_t/v2_t, exactly like the
            // product's Q4 cache write); fa2's read path still consumes the
            // storage tensor itself, so the store is ordered only by the
            // scheduler/split barriers -- the property under test.
            cur_store_k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, 1, n_kvh, 1);
            cur_store_v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, 1, n_kvh, 1);
            store_idxs = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 1, 1);
            store_k = ggml_set_rows(ctx, k2_t, cur_store_k, store_idxs);
            store_v = ggml_set_rows(ctx, v2_t, cur_store_v, store_idxs);
            ggml_set_name(store_k, "attn_ovf_store_k");
            ggml_set_name(store_v, "attn_ovf_store_v");
        }
        ggml_tensor * k2_read = k2_t;
        ggml_tensor * v2_read = v2_t;
        ggml_tensor * m2_read = m2_t;
        if (schedule_mode) {
            // Capacity-sized host mask, cropped/packed on Vulkan, just like
            // the occupied and stable-256 product paths.
            m2_read = ggml_cont(ctx, ggml_view_4d(ctx, m2_t, n_kv2, n_q, n_kvh, 1,
                    m2_t->nb[1], m2_t->nb[2], m2_t->nb[3], 0));
            overflow_prep.push_back(m2_read);
            if (overflow_layout == 1 || overflow_layout == 2) {
                k2_read = ggml_cont(ctx, k2_t);
                overflow_prep.push_back(k2_read);
                if (overflow_layout == 2) {
                    v2_read = ggml_cont(ctx, v2_t);
                    overflow_prep.push_back(v2_read);
                }
            } else if (overflow_layout >= 3) {
                overflow_rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv2, n_kvh);
                const ggml_type type = overflow_layout == 3 ? GGML_TYPE_F16 : GGML_TYPE_F32;
                k2_read = ggml_get_rows_as(ctx, k2_t, overflow_rows, type);
                v2_read = ggml_get_rows_as(ctx, v2_t, overflow_rows, type);
                overflow_prep.push_back(k2_read);
                overflow_prep.push_back(v2_read);
                if (type == GGML_TYPE_F32) {
                    k2_read = ggml_cast(ctx, k2_read, GGML_TYPE_F16);
                    v2_read = ggml_cast(ctx, v2_read, GGML_TYPE_F16);
                    overflow_prep.push_back(k2_read);
                    overflow_prep.push_back(v2_read);
                }
            }
        }
        fa2 = ggml_flash_attn_ext(ctx, q_t, k2_read, v2_read, m2_read,
                1.0f / std::sqrt(float(D)), 0.0f, 0.0f);
        ggml_prec_set_acc(fa2, GGML_PREC_F32);
        lse2 = ggml_flash_attn_ext_lse_out(ctx, fa2);

        const bool ok1 = ggml_backend_supports_op(local, fa1) && ggml_backend_supports_op(local, lse1);
        const bool ok2 = ggml_backend_supports_op(overflow, fa2) && ggml_backend_supports_op(overflow, lse2);
        if (!ok1 || !ok2) {
            ggml_free(ctx);
            res.declined = true;
            res.reason = std::string("supports_op local=") + (ok1 ? "1" : "0") + " overflow=" + (ok2 ? "1" : "0");
            return res;
        }

        // Merge formula: byte-for-byte the same arithmetic as run_merge
        // (validated by the merge-exactness gate there); only the local
        // operand differs.
        ggml_tensor * c1 = ggml_clamp(ctx, lse1, -1e6f, 1e30f);
        ggml_tensor * c2 = ggml_clamp(ctx, lse2, -1e6f, 1e30f);
        ggml_tensor * mx = ggml_scale(ctx,
                ggml_add(ctx, ggml_add(ctx, c1, c2), ggml_abs(ctx, ggml_sub(ctx, c1, c2))),
                0.5f);
        ggml_tensor * e1 = ggml_exp(ctx, ggml_sub(ctx, lse1, mx));
        ggml_tensor * e2 = ggml_exp(ctx, ggml_sub(ctx, lse2, mx));
        ggml_tensor * w1 = ggml_reshape_4d(ctx, e1, 1, n_qh, n_q, 1);
        ggml_tensor * w2 = ggml_reshape_4d(ctx, e2, 1, n_qh, n_q, 1);
        ggml_tensor * num = ggml_add(ctx, ggml_mul(ctx, o1, w1), ggml_mul(ctx, fa2, w2));
        ggml_tensor * den = ggml_add(ctx, e1, e2);
        eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_tensor * den_o = ggml_reshape_4d(ctx, ggml_add(ctx, den, eps), 1, n_qh, n_q, 1);
        o = ggml_div(ctx, num, den_o);
        lse_merged = ggml_add(ctx, mx, ggml_log(ctx, den));
    }

    // Reproduce the product's output ownership, including the CUDA LSE add.
    ggml_tensor * merged_div = o;
    if (schedule_mode) {
        o = ggml_cont(ctx, o);
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cpu == nullptr) {
        ggml_free(ctx);
        res.declined = true;
        res.reason = "cpu backend unavailable for the materialize side-path";
        return res;
    }
    ggml_backend_t backends[3] = { overflow, local, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 3, 8192, false, true);
    ggml_backend_sched_set_tensor_backend(sched, q_t, local);
    if (schedule_mode == 4) {
        // LLAMA_OVF_QWHT_CUDA=1 arm: mirror llama-graph.cpp's opt-in pin and
        // move the Q WHT to the local (CUDA) backend. Fail closed if the env
        // is not enabled or CUDA cannot run the transform; this variant exists
        // to exercise that exact production configuration.
        const char * env = std::getenv("LLAMA_OVF_QWHT_CUDA");
        if (env == nullptr || std::strcmp(env, "0") == 0) {
            ggml_backend_sched_free(sched);
            ggml_backend_free(cpu);
            ggml_free(ctx);
            res.declined = true;
            res.reason = "schedule_mode 4 requires LLAMA_OVF_QWHT_CUDA (non-\"0\")";
            return res;
        }
        if (!ggml_backend_supports_op(local, q_rot)) {
            ggml_backend_sched_free(sched);
            ggml_backend_free(cpu);
            ggml_free(ctx);
            res.declined = true;
            res.reason = "CUDA WHT not supported for this shape (LLAMA_OVF_QWHT_CUDA=1)";
            return res;
        }
        ggml_backend_sched_set_tensor_backend(sched, q_rot, local);
    } else if (schedule_mode != 3) {
        // Mode 3 leaves the Q WHT to the scheduler on purpose (legacy order
        // reproduces the pre-consolidation placement naturally).
        ggml_backend_sched_set_tensor_backend(sched, q_rot, schedule_mode ? overflow : local);
    }
    ggml_backend_sched_set_tensor_backend(sched, indices, local);
    ggml_backend_sched_set_tensor_backend(sched, current_k, local);
    ggml_backend_sched_set_tensor_backend(sched, current_v, local);
    ggml_backend_sched_set_tensor_backend(sched, k_stage, local);
    ggml_backend_sched_set_tensor_backend(sched, v_stage, local);
    ggml_backend_sched_set_tensor_backend(sched, k_records, local);
    ggml_backend_sched_set_tensor_backend(sched, v_records, local);
    ggml_backend_sched_set_tensor_backend(sched, stored_k, local);
    ggml_backend_sched_set_tensor_backend(sched, stored_v, local);
    ggml_backend_sched_set_tensor_backend(sched, k_v, local);
    ggml_backend_sched_set_tensor_backend(sched, v_v, local);
    ggml_backend_sched_set_tensor_backend(sched, m1_t, local);
    ggml_backend_sched_set_tensor_backend(sched, fa1, local);
    ggml_backend_sched_set_tensor_backend(sched, lse1, local);
    if (schedule_mode != 3) {
        // Mode 3 leaves the local O WHT to the scheduler on purpose.
        ggml_backend_sched_set_tensor_backend(sched, o1, schedule_mode ? overflow : local);
    }
    if (with_overflow) {
        ggml_backend_sched_set_tensor_backend(sched, k2_t, overflow);
        ggml_backend_sched_set_tensor_backend(sched, v2_t, overflow);
        ggml_backend_sched_set_tensor_backend(sched, m2_t, schedule_mode ? cpu : overflow);
        if (overflow_rows) { ggml_backend_sched_set_tensor_backend(sched, overflow_rows, cpu); }
        for (ggml_tensor * node : overflow_prep) {
            ggml_backend_sched_set_tensor_backend(sched, node, overflow);
        }
        if (store_k) {
            // The store writes the overflow storage, so it belongs to the same
            // backend as k2_t/v2_t by contract, not by scheduling choice.
            ggml_backend_sched_set_tensor_backend(sched, store_k, overflow);
            ggml_backend_sched_set_tensor_backend(sched, store_v, overflow);
            ggml_backend_sched_set_tensor_backend(sched, cur_store_k, overflow);
            ggml_backend_sched_set_tensor_backend(sched, cur_store_v, overflow);
            ggml_backend_sched_set_tensor_backend(sched, store_idxs, overflow);
        }
        ggml_backend_sched_set_tensor_backend(sched, fa2, overflow);
        ggml_backend_sched_set_tensor_backend(sched, lse2, overflow);
    }
    ggml_backend_sched_set_tensor_backend(sched, o, local);
    ggml_backend_sched_set_tensor_backend(sched, lse_merged, local);

    if (schedule_mode) {
        // Model inputs are recomputed/uploaded and KV storage is externally
        // allocated in production. Keep these fixture leaves alive across both
        // evaluations instead of letting the graph allocator recycle them.
        for (ggml_tensor * leaf : { q_t, indices, current_k, current_v, k_stage, v_stage,
                k_records, v_records, m1_t, k2_t, v2_t, m2_t, overflow_rows, eps,
                cur_store_k, cur_store_v, store_idxs }) {
            if (leaf) { ggml_set_output(leaf); }
        }
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    if (schedule_mode) {
        // Modes 1 and 3 reproduce the pre-consolidation (legacy) expansion
        // order: the overflow stores run first, then fa1, fa2, lse1, lse2.
        // Mode 2 follows the consolidated order: fa1/lse1 finish first, then
        // the moved overflow stores join fa2. Modes 1/2 force the consolidated
        // arm's placement of the Q WHT, the local O WHT and the merge steps;
        // mode 3 keeps the legacy order WITHOUT those manual placements, so
        // bit-equality with mode 2 proves the consolidated schedule changed no
        // bit under the scheduler's natural (legacy) placement too. Q WHT
        // stays on Vulkan in the forced arms: at D=256 its F32 results differ
        // from CUDA, so that pre-local split is unavoidable. Mode 4 is the
        // mode-2 schedule with the Q WHT forced onto the local CUDA backend
        // (LLAMA_OVF_QWHT_CUDA=1), which drops that pre-local split.
        ggml_build_forward_expand(gf, stored_k);
        ggml_build_forward_expand(gf, stored_v);
        if (schedule_mode == 1 || schedule_mode == 3) {
            if (store_k) {
                ggml_build_forward_expand(gf, store_k);
                ggml_build_forward_expand(gf, store_v);
            }
            ggml_build_forward_expand(gf, q_rot);
        }
        ggml_build_forward_expand(gf, fa1);
        if (schedule_mode == 2 || schedule_mode == 4) {
            ggml_build_forward_expand(gf, lse1);
            if (store_k) {
                ggml_build_forward_expand(gf, store_k);
                ggml_build_forward_expand(gf, store_v);
            }
        }
        ggml_build_forward_expand(gf, fa2);
        ggml_build_forward_expand(gf, lse1);
        ggml_build_forward_expand(gf, lse2);
        const int merge_begin = ggml_graph_n_nodes(gf);
        ggml_build_forward_expand(gf, merged_div);
        ggml_build_forward_expand(gf, lse_merged);
        ggml_build_forward_expand(gf, o);
        if (schedule_mode != 3) {
            // The production graph's unpinned intermediates followed lse2 onto
            // Vulkan already. Pin both forced arms identically to isolate
            // ordering; mode 3 lets the scheduler reproduce that placement.
            for (int i = merge_begin; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_tensor * node = ggml_graph_node(gf, i);
                if (node != o && node != lse_merged && !node->view_src) {
                    ggml_backend_sched_set_tensor_backend(sched, node, overflow);
                }
            }
        }
    } else {
        ggml_build_forward_expand(gf, o);
        ggml_build_forward_expand(gf, lse_merged);
    }
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        res.declined = true;
        res.reason = "sched_alloc_graph failed";
        return res;
    }

    res.splits = ggml_backend_sched_get_n_splits(sched);
    ggml_backend_t previous = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->view_src) { continue; }
        ggml_backend_t current = ggml_backend_sched_get_tensor_backend(sched, node);
        if (current != previous && current == overflow) { ++res.overflow_splits; }
        previous = current;
    }

    eval_counters counters;
    counters.sched = sched;
    counters.overflow = overflow;
    ggml_backend_sched_set_eval_callback(sched, eval_count_cb, &counters);

    // Uploads ---------------------------------------------------------------
    std::vector<float> ck((size_t) record_dim * record_heads * n_kv1 * n_stream);
    std::vector<float> cv((size_t) record_dim * record_heads * n_kv1 * n_stream);
    for (int s = 0; s < n_kv1; ++s) {
        for (int kh = 0; kh < n_kvh; ++kh) {
            for (int slice = 0; slice < slices; ++slice) {
                for (int d = 0; d < record_dim; ++d) {
                    const size_t src = (size_t(s) * n_kvh + kh) * D + size_t(slice) * record_dim + d;
                    const size_t dst = (size_t(s) * record_heads + (kh * slices + slice)) * record_dim + d;
                    ck[dst] = k1[src];
                    cv[dst] = v1[src];
                }
            }
        }
    }
    std::vector<int64_t> idx(n_kv1 * n_stream);
    for (int i = 0; i < n_kv1 * n_stream; ++i) {
        idx[i] = i;
    }
    std::vector<uint8_t> k_stage_zeros(ggml_nbytes(k_stage), 0);
    std::vector<uint8_t> v_stage_zeros(ggml_nbytes(v_stage), 0);
    std::vector<uint8_t> k_record_zeros(ggml_nbytes(k_records), 0);
    std::vector<uint8_t> v_record_zeros(ggml_nbytes(v_records), 0);
    if (eps != nullptr) {
        ggml_backend_tensor_set(eps, &eps_value, 0, sizeof(float));
    }
    ggml_backend_tensor_set(q_t, q.data(), 0, ggml_nbytes(q_t));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(current_k, ck.data(), 0, ggml_nbytes(current_k));
    ggml_backend_tensor_set(current_v, cv.data(), 0, ggml_nbytes(current_v));
    ggml_backend_tensor_set(k_stage, k_stage_zeros.data(), 0, k_stage_zeros.size());
    ggml_backend_tensor_set(v_stage, v_stage_zeros.data(), 0, v_stage_zeros.size());
    ggml_backend_tensor_set(k_records, k_record_zeros.data(), 0, k_record_zeros.size());
    ggml_backend_tensor_set(v_records, v_record_zeros.data(), 0, v_record_zeros.size());
    {
        std::vector<ggml_fp16_t> m1_data(size_t(n_kv1) * n_q, ggml_fp32_to_fp16(0.0f));
        if (!mask1_visible) {
            std::fill(m1_data.begin(), m1_data.end(), ggml_fp32_to_fp16(-INFINITY));
        }
        ggml_backend_tensor_set(m1_t, m1_data.data(), 0, ggml_nbytes(m1_t));
    }
    if (with_overflow) {
        std::vector<ggml_fp16_t> m2_data(size_t(m2_t->ne[0]) * n_kvh * n_q, ggml_fp32_to_fp16(-INFINITY));
        for (int t = 0; t < n_q; ++t) {
            for (int kh = 0; kh < n_kvh; ++kh) {
                for (int s = 0; s < n_kv2; ++s) {
                    const bool vis = mask2_mode == 0 ? true
                                    : mask2_mode == 2 ? false
                                    : s <= t + n_kv2 - n_q;
                    const size_t off = (size_t(kh) * n_q + t) * m2_t->ne[0] + s;
                    m2_data[off] = ggml_fp32_to_fp16(vis ? 0.0f : -INFINITY);
                }
            }
        }
        ggml_backend_tensor_set(m2_t, m2_data.data(), 0, ggml_nbytes(m2_t));
        ggml_backend_tensor_set(k2_t, k2_bytes.data(), 0, ggml_nbytes(k2_t));
        ggml_backend_tensor_set(v2_t, v2_bytes.data(), 0, ggml_nbytes(v2_t));
    }
    if (store_k) {
        // Deterministic current-token values, distinct from any storage row.
        std::vector<float> store_row_k(size_t(D) * size_t(n_kvh));
        std::vector<float> store_row_v(size_t(D) * size_t(n_kvh));
        for (size_t i = 0; i < store_row_k.size(); ++i) {
            store_row_k[i] = 0.5f * float(int(i % 97)) - 20.0f;
            store_row_v[i] = 0.25f * float(int(i % 61)) - 8.0f;
        }
        const int64_t store_row_idx = n_kv2 - 1;
        ggml_backend_tensor_set(cur_store_k, store_row_k.data(), 0, ggml_nbytes(cur_store_k));
        ggml_backend_tensor_set(cur_store_v, store_row_v.data(), 0, ggml_nbytes(cur_store_v));
        ggml_backend_tensor_set(store_idxs, &store_row_idx, 0, ggml_nbytes(store_idxs));
    }

    if (overflow_rows) {
        std::vector<int32_t> rows(size_t(n_kv2) * n_kvh);
        for (size_t i = 0; i < rows.size(); ++i) { rows[i] = i % n_kv2; }
        ggml_backend_tensor_set(overflow_rows, rows.data(), 0, ggml_nbytes(overflow_rows));
    }

    int attn = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (ggml_graph_node(gf, i)->op == GGML_OP_FLASH_ATTN_EXT) {
            ++attn;
            const ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched, ggml_graph_node(gf, i));
            if (ggml_graph_node(gf, i) == fa1) {
                res.fa1_on_requested = (be == local);
            } else if (ggml_graph_node(gf, i) == fa2) {
                res.fa2_on_requested = (be == overflow);
            }
        }
    }
    res.attn_nodes = attn;

    // Reuse the allocated graph with changed Q to catch stale cross-backend
    // copies and lifetime mistakes, not just a successful first allocation.
    for (int pass = 0; pass < (schedule_mode ? 2 : 1); ++pass) {
        if (pass) {
            std::vector<float> next_q = q;
            for (float & x : next_q) { x = -x; }
            ggml_backend_tensor_set(q_t, next_q.data(), 0, ggml_nbytes(q_t));
        }
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            res.declined = true;
            res.reason = "sched_graph_compute failed";
            break;
        }
        if (schedule_mode) {
            const size_t offset = res.q_rot.size();
            res.q_rot.resize(offset + ggml_nelements(q_rot));
            ggml_backend_tensor_get(q_rot, res.q_rot.data() + offset, 0, ggml_nbytes(q_rot));
        }
        const size_t o_offset = res.o.size();
        const size_t l_offset = res.lse.size();
        res.o.resize(o_offset + ggml_nelements(o));
        res.lse.resize(l_offset + ggml_nelements(lse_merged));
        ggml_backend_tensor_get(o, res.o.data() + o_offset, 0, ggml_nbytes(o));
        ggml_backend_tensor_get(lse_merged, res.lse.data() + l_offset, 0, ggml_nbytes(lse_merged));
        res.attn_nodes_on_overflow = counters.attn_on_overflow;
        res.nodes_on_overflow = counters.on_overflow;
    }
    if (store_k && !res.declined) {
        // Post-run storage bytes: the caller re-runs the same graph with these
        // bytes pre-filled and no store to prove fa2 consumed the stored row.
        res.k2_final.resize(ggml_nbytes(k2_t));
        res.v2_final.resize(ggml_nbytes(v2_t));
        ggml_backend_tensor_get(k2_t, res.k2_final.data(), 0, res.k2_final.size());
        ggml_backend_tensor_get(v2_t, res.v2_final.data(), 0, res.v2_final.size());
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return res;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_backend_load_all();

    // The fork exposes the Radeon device through ggml_backend_vk_init; the
    // generic name-based init returns a backend that declines the FA-LSE route.
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t vk  = ggml_backend_vk_init(0);
    ggml_backend_t cuda = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (cuda == nullptr) {
        for (int i = 0; i < (int) ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                    std::strstr(ggml_backend_dev_name(dev), "CUDA") != nullptr) {
                cuda = ggml_backend_dev_init(dev, nullptr);
                break;
            }
        }
    }
    if (cpu) {
        // The plan gates are numerical, so the CPU FA must use the stable
        // reference accumulation instead of the tiled/split-KV float paths.
        ggml_backend_cpu_set_use_ref(cpu, true);
    }
    if (!cpu || !vk || !cuda) {
        std::printf("  FAIL: required backends missing (cpu=%d vulkan=%d cuda=%d)\n",
                    cpu != nullptr, vk != nullptr, cuda != nullptr);
        return 1;
    }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // R4: formula-level stability first (no backend): shift invariance over
    // large LSEs, exact empties, no NaN.
    check_merge_formula_invariance();

    // Mandatory shapes: Q=1 and Q=2 decode, Q=256 prefill, GQA 24/4 at D=256,
    // plus D=128 GQA 6/2 as the smaller geometry.
    struct shape { int D, n_q, n_qh, n_kvh, n_kv1, n_kv2; };
    const shape shapes[] = {
        { 128,   1,  6, 2, 512, 128 },
        { 128,   2,  6, 2, 512, 128 },
        { 256,   1, 24, 4, 512, 128 },
        { 256,   2, 24, 4, 512, 128 },
        { 256, 256, 24, 4, 512, 128 },
    };

    int executed = 0;
    for (const shape & sh : shapes) {
        const int D = sh.D, n_q = sh.n_q, n_qh = sh.n_qh, n_kvh = sh.n_kvh;
        const size_t q_elems  = size_t(D) * n_q * n_qh;
        const size_t kv1_elems = size_t(D) * sh.n_kv1 * n_kvh;
        const size_t kv2_elems = size_t(D) * sh.n_kv2 * n_kvh;

        // Host buffers follow the tensor layout exactly:
        //   Q is ne = {D, n_q, n_qh}     -> q[(h*n_q + t)*D + d]
        //   K/V ne = {D, n_kv, n_kvh}   -> k[(s + kh*n_kv)*D + d]
        std::vector<float> q(q_elems);
        std::vector<float> k1(kv1_elems);
        std::vector<float> v1(kv1_elems);
        std::vector<float> k2(kv2_elems);
        std::vector<float> v2(kv2_elems);
        for (auto & x : q)  { x = dist(rng); }
        for (auto & x : k1) { x = dist(rng); }
        for (auto & x : v1) { x = dist(rng); }
        for (auto & x : k2) { x = dist(rng); }
        for (auto & x : v2) { x = dist(rng); }

        std::vector<uint8_t> k1b, v1b, k2b, v2b;
        std::vector<float>   k1d, v1d, k2d, v2d;
        pack_range(GGML_TYPE_F16, D, k1, k1b, k1d);
        pack_range(GGML_TYPE_F16, D, v1, v1b, v1d);
        pack_range(GGML_TYPE_Q4_0, D, k2, k2b, k2d);
        pack_range(GGML_TYPE_Q4_0, D, v2, v2b, v2d);

        char tag[128];
        std::snprintf(tag, sizeof(tag), "merge D%d Q%d GQA%d/%d", D, n_q, n_qh, n_kvh);
        g_case = tag;

        // Causal overflow mask, matching the oracle exactly (review 6, R3):
        // queries sit at the end of the concatenated sequence, so the product
        // causal rule s <= t + n_kv2 - n_q applies. Both ranges on the Radeon:
        // the CUDA FA declines LSE for non-KVarN tensors by design, and the
        // CUDA/KVarN local range is gated by tests/test-kvarn
        // (NRMSE <= 1.7e-4). This prototype therefore proves the merge
        // composition, the scheduler copies and the empty-range contract on
        // the backend pair it can drive.
        if (n_q == 1) {
            const auto same_bits = [](const std::vector<float> & a, const std::vector<float> & b) {
                return !a.empty() && a.size() == b.size() &&
                        std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
            };
            const auto pass_bits = [](const std::vector<float> & v, bool second) {
                const size_t h = v.size() / 2;
                return std::vector<float>(v.begin() + (second ? h : 0),
                        v.begin() + (second ? 2 * h : h));
            };
            for (int layout = 0; layout < 5; ++layout) {
                // Mode 1: legacy expansion order + the consolidated arm's forced
                // (pinned) placement. Mode 2: consolidated order + forced
                // placement (the production decode). Mode 3: legacy order with
                // NO manual placement of the Q WHT, the local O WHT nor the
                // merge steps, so the scheduler reproduces the legacy
                // placement by itself. All three carry the in-graph overflow
                // store of the current token.
                const merge_result before = run_merge_kvarn(cuda, vk, D, n_q, n_qh, n_kvh,
                        sh.n_kv1, sh.n_kv2, true, 1, q, k1, v1, k2b, v2b, 1, layout, 1);
                const merge_result after = run_merge_kvarn(cuda, vk, D, n_q, n_qh, n_kvh,
                        sh.n_kv1, sh.n_kv2, true, 1, q, k1, v1, k2b, v2b, 2, layout, 1);
                const merge_result plain = run_merge_kvarn(cuda, vk, D, n_q, n_qh, n_kvh,
                        sh.n_kv1, sh.n_kv2, true, 1, q, k1, v1, k2b, v2b, 3, layout, 1);
                CHECK(!before.declined && !after.declined && !plain.declined);
                if (before.declined || after.declined || plain.declined) {
                    continue;
                }
                CHECK(same_bits(before.q_rot, after.q_rot));
                CHECK(same_bits(before.o, after.o));
                CHECK(same_bits(before.lse, after.lse));
                // The legacy order under the scheduler's own placement must be
                // bit-identical to the consolidated arm as well.
                CHECK(same_bits(plain.q_rot, after.q_rot));
                CHECK(same_bits(plain.o, after.o));
                CHECK(same_bits(plain.lse, after.lse));
                CHECK(before.overflow_splits == 3);
                CHECK(after.overflow_splits == 2);
                CHECK(after.splits == before.splits - 2);

                // The in-graph overflow store must be visible to fa2's read
                // path. First prove it executed (the stored token row changed
                // in the post-run storage bytes), then re-run the same graph
                // with those bytes pre-filled and NO store: identical O/LSE
                // proves the first run's fa2 consumed the stored row.
                bool store_changed = false;
                CHECK(after.k2_final.size() == k2b.size() &&
                      after.v2_final.size() == v2b.size());
                if (after.k2_final.size() == k2b.size()) {
                    const size_t row = ggml_row_size(GGML_TYPE_Q4_0, D);
                    const size_t plane = row * size_t(sh.n_kv2);
                    for (int kh = 0; kh < n_kvh && !store_changed; ++kh) {
                        const size_t off = size_t(sh.n_kv2 - 1) * row + size_t(kh) * plane;
                        if (std::memcmp(after.k2_final.data() + off, k2b.data() + off, row) != 0) {
                            store_changed = true;
                        }
                    }
                }
                CHECK(store_changed);
                const merge_result ctl = run_merge_kvarn(cuda, vk, D, n_q, n_qh, n_kvh,
                        sh.n_kv1, sh.n_kv2, true, 1, q, k1, v1,
                        after.k2_final, after.v2_final, 2, layout, 0);
                CHECK(!ctl.declined);
                if (!ctl.declined) {
                    CHECK(same_bits(ctl.o, after.o));
                    CHECK(same_bits(ctl.lse, after.lse));
                }

                // LLAMA_OVF_QWHT_CUDA=1 arm (mode 4): the same consolidated
                // schedule with the Q WHT on the local CUDA backend instead of
                // Vulkan. The test sets the env itself, so the binary behaves
                // identically regardless of the caller's environment. The Q
                // WHT bits then differ from the Vulkan arm (at D=256), so the
                // comparison against the default arm and its oracle-adjacent
                // gates is TOLERANCE-based (nrmse), never bitwise; the reused
                // graph's second pass stays bitwise against a fresh graph of
                // the same arm.
                set_env_llama_ovf_qwht_cuda("1");
                const merge_result cuda_arm = run_merge_kvarn(cuda, vk, D, n_q, n_qh, n_kvh,
                        sh.n_kv1, sh.n_kv2, true, 1, q, k1, v1, k2b, v2b, 4, layout, 1);
                std::vector<float> q_neg4 = q;
                for (float & x : q_neg4) { x = -x; }
                const merge_result cuda_fresh = run_merge_kvarn(cuda, vk, D, n_q, n_qh, n_kvh,
                        sh.n_kv1, sh.n_kv2, true, 1, q_neg4, k1, v1, k2b, v2b, 4, layout, 1);
                set_env_llama_ovf_qwht_cuda(nullptr);
                CHECK(!cuda_arm.declined);
                CHECK(!cuda_fresh.declined);
                double cuda_q_nrmse = INFINITY;
                double cuda_o_nrmse = INFINITY;
                double cuda_lse_nrmse = INFINITY;
                bool cuda_fresh_bits = false;
                if (!cuda_arm.declined && !cuda_fresh.declined) {
                    // Plan gates: nrmse Q/O <= 1e-3 and LSE <= 1e-2 against
                    // the default (Vulkan-WHT) arm, not bitwise.
                    cuda_q_nrmse = nrmse_ff(cuda_arm.q_rot, after.q_rot);
                    cuda_o_nrmse = nrmse_ff(cuda_arm.o, after.o);
                    cuda_lse_nrmse = nrmse_ff(cuda_arm.lse, after.lse);
                    CHECK(cuda_q_nrmse <= 1e-3);
                    CHECK(cuda_o_nrmse <= 1e-3);
                    CHECK(cuda_lse_nrmse <= 1e-2);
                    // One fewer Vulkan cluster per layer: the Q WHT no longer
                    // runs as its own pre-local split.
                    CHECK(cuda_arm.overflow_splits == after.overflow_splits - 1);
                    CHECK(cuda_arm.splits == after.splits - 2);
                    cuda_fresh_bits =
                        same_bits(pass_bits(cuda_arm.o, false), pass_bits(cuda_fresh.o, true)) &&
                        same_bits(pass_bits(cuda_arm.o, true), pass_bits(cuda_fresh.o, false)) &&
                        same_bits(pass_bits(cuda_arm.lse, false), pass_bits(cuda_fresh.lse, true)) &&
                        same_bits(pass_bits(cuda_arm.lse, true), pass_bits(cuda_fresh.lse, false)) &&
                        same_bits(pass_bits(cuda_arm.q_rot, false), pass_bits(cuda_fresh.q_rot, true)) &&
                        same_bits(pass_bits(cuda_arm.q_rot, true), pass_bits(cuda_fresh.q_rot, false));
                    CHECK(cuda_fresh_bits);
                }
                std::printf("  cuda-qwht D%d layout%d: splits %d vs %d, Vulkan clusters %d vs %d, "
                        "nrmse Q=%.3e O=%.3e LSE=%.3e, fresh-negQ=%d\n",
                        D, layout, after.splits, cuda_arm.splits,
                        after.overflow_splits, cuda_arm.overflow_splits,
                        cuda_q_nrmse, cuda_o_nrmse, cuda_lse_nrmse, int(cuda_fresh_bits));

                // The reused graph's pass 2 (Q negated in place) must match a
                // FRESH graph allocated with the negated Q up front, per arm.
                std::vector<float> q_neg = q;
                for (float & x : q_neg) { x = -x; }
                bool fresh_ok = true;
                for (int mode = 1; mode <= 3; ++mode) {
                    const merge_result & base_ref = mode == 1 ? before : (mode == 2 ? after : plain);
                    const merge_result fresh = run_merge_kvarn(cuda, vk, D, n_q, n_qh, n_kvh,
                            sh.n_kv1, sh.n_kv2, true, 1, q_neg, k1, v1, k2b, v2b, mode, layout, 1);
                    CHECK(!fresh.declined);
                    if (fresh.declined) {
                        fresh_ok = false;
                        continue;
                    }
                    CHECK(same_bits(pass_bits(base_ref.o, false), pass_bits(fresh.o, true)));
                    CHECK(same_bits(pass_bits(base_ref.o, true), pass_bits(fresh.o, false)));
                    CHECK(same_bits(pass_bits(base_ref.lse, false), pass_bits(fresh.lse, true)));
                    CHECK(same_bits(pass_bits(base_ref.lse, true), pass_bits(fresh.lse, false)));
                    CHECK(same_bits(pass_bits(base_ref.q_rot, false), pass_bits(fresh.q_rot, true)));
                    CHECK(same_bits(pass_bits(base_ref.q_rot, true), pass_bits(fresh.q_rot, false)));
                }
                std::printf("  decode scheduling D%d layout%d: splits %d -> %d, Vulkan clusters %d -> %d, "
                        "bitwise Q/O/LSE=%d/%d/%d, legacy-order-no-pin Q/O/LSE=%d/%d/%d, "
                        "store-to-fa2=%d, fresh-negQ=%d\n", D, layout, before.splits, after.splits,
                        before.overflow_splits, after.overflow_splits,
                        int(same_bits(before.q_rot, after.q_rot)), int(same_bits(before.o, after.o)),
                        int(same_bits(before.lse, after.lse)),
                        int(same_bits(plain.q_rot, after.q_rot)), int(same_bits(plain.o, after.o)),
                        int(same_bits(plain.lse, after.lse)),
                        int(store_changed && !ctl.declined), int(fresh_ok));
            }
        }
        const merge_result m = run_merge(vk, vk, D, n_q, n_qh, n_kvh,
                sh.n_kv1, sh.n_kv2, GGML_TYPE_Q4_0, true, 1,
                q, k1, v1, k2, v2, k1b, v1b, k2b, v2b);
        CHECK(!m.declined);
        if (m.declined) {
            std::printf("  declined: %s\n", m.reason.c_str());
            continue;
        }
        const ref_out ref = oracle(D, n_q, n_qh, n_kvh, q, k1d, v1d, sh.n_kv1,
                k2d, v2d, sh.n_kv2, true, 1);
        // Same-backend single-range baseline over the *local* range only. The
        // plan requires merge error to be separated from representation/backend
        // error (section 4.1), so the merge is also gated against what the same
        // backend produces for a single range over the same bytes.
        const merge_result base = run_merge(vk, vk, D, n_q, n_qh, n_kvh,
                sh.n_kv1, 0, GGML_TYPE_Q4_0, true, 0,
                q, k1, v1, k2, v2, k1b, v1b, k2b, v2b);
        CHECK(!base.declined);
        const ref_out ref_base = oracle(D, n_q, n_qh, n_kvh, q, k1d, v1d, sh.n_kv1,
                k2d, v2d, sh.n_kv2, true, 2);
        // Same-backend single-range baseline over the *overflow* range only,
        // with the same causal mask the merged graph feeds into the combine,
        // so it is the reference for merge exactness.
        const merge_result ovf = run_merge(vk, vk, D, n_q, n_qh, n_kvh,
                sh.n_kv2, 0, GGML_TYPE_Q4_0, true, 1,
                q, k2, v2, k2, v2, k2b, v2b, k2b, v2b, GGML_TYPE_Q4_0);
        CHECK(!ovf.declined);
        // The oracle models the single range as the SECOND range (n_kv1 = 0)
        // so mask2_mode = 1 applies the same causal rule the merged graph
        // feeds its overflow range; a causal first range does not exist.
        const ref_out ref_ovf = oracle(D, n_q, n_qh, n_kvh, q, k2d, v2d, 0,
                k2d, v2d, sh.n_kv2, true, 1);
        const double nrmse = nrmse_of(m.o, ref.o);
        const double max_abs = max_abs_of(m.o, ref.o);
        // R2: sizes must match exactly before any comparison; nrmse_of already
        // fails closed on size mismatch (INFINITY), max_abs_of does not.
        CHECK(m.o.size() == ref.o.size() && !ref.o.empty());
        CHECK(m.lse.size() == ref.lse.size() && !ref.lse.empty());
        CHECK(ref.empty.size() == ref.lse.size());
        double max_lse = 0.0;
        // R2: finiteness over EVERY O element (the old loop only covered the
        // first lse.size() elements) and finite LSE on every valid row. Empty
        // rows keep the exact contract: O exactly 0, LSE exactly -inf.
        for (size_t i = 0; i < m.o.size(); ++i) {
            CHECK(std::isfinite(m.o[i]));
        }
        for (int t = 0; t < n_q; ++t) {
            for (int h = 0; h < n_qh; ++h) {
                const size_t li = size_t(h) + size_t(t) * n_qh;
                if (ref.empty[li]) {
                    for (int d = 0; d < D; ++d) {
                        CHECK(m.o[(size_t(t) * n_qh + h) * D + d] == 0.0f);
                    }
                    CHECK(std::isinf(m.lse[li]) && m.lse[li] < 0.0f);
                    continue;
                }
                CHECK(std::isfinite(m.lse[li]));
                CHECK(std::isfinite(ref.lse[li]));
                max_lse = std::max(max_lse, std::fabs(double(m.lse[li]) - ref.lse[li]));
            }
        }
        const double base_nrmse = base.declined ? -1.0 : nrmse_of(base.o, ref_base.o);
        const double ovf_nrmse_print = ovf.declined ? -1.0 : nrmse_of(ovf.o, ref_ovf.o);
        std::printf("  %s: nrmse=%.3e maxO=%.3e maxLSE=%.3e base_nrmse=%.3e ovf_nrmse=%.3e "
                "attn=%d attn_on_ovf=%d nodes_on_ovf=%d\n",
                tag, nrmse, max_abs, max_lse, base_nrmse, ovf_nrmse_print,
                m.attn_nodes, m.attn_nodes_on_overflow, m.nodes_on_overflow);
        CHECK(!base.declined);

        CHECK(m.attn_nodes == 2);
        CHECK(m.fa1_on_requested);
        CHECK(m.fa2_on_requested);
        CHECK(m.attn_nodes_on_overflow >= 1);   // the overflow FA runs on the Radeon

        // Gate 1 - merge exactness: combining the two single-range backend
        // outputs on the host must reproduce the graph's merged output up to
        // fp32 rounding. This gates the merge arithmetic itself, independent of
        // backend fidelity.
        //
        // Gate 2 - no regression: the merged result may not be materially worse
        // than the same backend's single-range error. The absolute fidelity gate
        // (NRMSE <= 1e-3) belongs to the tests owning the backends the product
        // uses: tests/test-kvarn (CUDA KVarN, <= 1.7e-4) and
        // tests/test-position-split-lse-vk (Vulkan Q4, 7.6e-4).
        CHECK(!ovf.declined);
        if (!ovf.declined) {
            const exactness_result ex = check_merge_exactness(D, n_q, n_qh,
                    base.o, base.lse, ovf.o, ovf.lse, m.o, m.lse, ref.empty);
            std::printf("    merge exactness: maxO=%.3e maxLSE=%.3e finite=%d\n",
                    ex.max_o, ex.max_lse, int(ex.all_finite));
            CHECK(ex.all_finite);
            CHECK(ex.max_o <= kGateMergeExactO);
            CHECK(ex.max_lse <= kGateMergeExactLse);
            const double ovf_nrmse = nrmse_of(ovf.o, ref_ovf.o);
            CHECK(nrmse <= std::max(1.05 * std::max(base_nrmse, ovf_nrmse), 1e-4));
            // R4 (msg57): the mandatory cumulative gate runs on the PRODUCT's
            // per-range representation -- local range as KVarN4 records on CUDA
            // (rotated domain, un-rotated before the merge, exactly like
            // llama-graph.cpp) and overflow as Q4_0 on Vulkan. The (vk,vk) pair
            // measured above is a stand-in: the CUDA FA declines LSE for
            // non-KVarN tensors by design, and the Vulkan-F16 local route
            // measures ~1.0e-3, above this gate BY CONSTRUCTION of the stand-in
            // (not a product route). Its numbers remain printed above and in
            // docs/occupancy-progress-20261003.md for the record.
            if (!(nrmse <= kGateNrmseO)) {
                std::printf("    note: stand-in (Vulkan-F16 local) nrmse=%.3e exceeds 1e-3 by construction; "
                        "not a product route; the mandatory gate below measures the real pair\n", nrmse);
            }
            {
                const merge_result mk = run_merge_kvarn(cuda, vk, D, n_q, n_qh, n_kvh,
                        sh.n_kv1, sh.n_kv2, true, 1, q, k1, v1, k2b, v2b);
                CHECK(!mk.declined);
                // Structural reference (plan 4.1): the SAME per-range
                // representation executed by the stable CPU attention -- the
                // same KVarN store/view graph on the CPU backend (it decodes the
                // very records the product reads) plus the same Q4 overflow and
                // the same merge arithmetic.
                const merge_result mref = run_merge_kvarn(cpu, cpu, D, n_q, n_qh, n_kvh,
                        sh.n_kv1, sh.n_kv2, true, 1, q, k1, v1, k2b, v2b);
                CHECK(!mref.declined);
                if (!mk.declined && !mref.declined) {
                    CHECK(mk.o.size() == mref.o.size() && !mref.o.empty());
                    CHECK(mk.lse.size() == mref.lse.size());
                    const std::vector<double> mref_o(mref.o.begin(), mref.o.end());
                    const double nrmse_kv = nrmse_of(mk.o, mref_o);
                    const double max_abs_kv = max_abs_of(mk.o, mref_o);
                    bool finite_kv = true;
                    for (size_t i = 0; i < mk.o.size(); ++i) {
                        if (!std::isfinite(mk.o[i]) || !std::isfinite(mref.o[i])) {
                            finite_kv = false;
                            break;
                        }
                    }
                    double max_lse_kv = 0.0;
                    for (size_t li = 0; li < mk.lse.size(); ++li) {
                        if (!std::isfinite(mk.lse[li]) || !std::isfinite(mref.lse[li])) {
                            finite_kv = false;
                            break;
                        }
                        max_lse_kv = std::max(max_lse_kv,
                                std::fabs(double(mk.lse[li]) - mref.lse[li]));
                    }
                    std::printf("    REAL PAIR (CUDA/KVarN4 + Vulkan/Q4) vs CPU ref: nrmse=%.3e maxO=%.3e maxLSE=%.3e "
                            "attn=%d attn_on_ovf=%d fa1_local=%d fa2_ovf=%d\n",
                            nrmse_kv, max_abs_kv, max_lse_kv, mk.attn_nodes,
                            mk.attn_nodes_on_overflow, int(mk.fa1_on_requested), int(mk.fa2_on_requested));
                    CHECK(finite_kv);
                    CHECK(mk.attn_nodes == 2);
                    CHECK(mk.fa1_on_requested);
                    CHECK(mk.fa2_on_requested);
                    CHECK(mk.attn_nodes_on_overflow >= 1);
                    CHECK(nrmse_kv <= kGateNrmseO);
                    CHECK(max_abs_kv <= kGateMaxAbsO);
                    CHECK(max_lse_kv <= kGateMaxAbsLse);
                }
            }
            if (executed == 0) {
                // Negative controls (review 6, R2): the verifier must fail on
                // NaN/+Inf and on a perturbation confined to a single
                // head/query row, proving it does not average errors away.
                g_case = std::string(tag) + " negative control";
                {
                    std::vector<float> bad_o = m.o;
                    const size_t bad_oi = (size_t(0) * n_qh + 1 % n_qh) * D + 5 % D;
                    bad_o[bad_oi] = std::numeric_limits<float>::quiet_NaN();
                    const exactness_result bad = check_merge_exactness(D, n_q, n_qh,
                            base.o, base.lse, ovf.o, ovf.lse, bad_o, m.lse, ref.empty);
                    CHECK(!bad.all_finite);
                }
                {
                    std::vector<float> bad_lse = m.lse;
                    bad_lse[0] = std::numeric_limits<float>::infinity();
                    const exactness_result bad = check_merge_exactness(D, n_q, n_qh,
                            base.o, base.lse, ovf.o, ovf.lse, m.o, bad_lse, ref.empty);
                    CHECK(!bad.all_finite);
                }
                {
                    std::vector<float> bad_o = m.o;
                    // Single head/query row (t=0,h=0), first element: a +1.0
                    // perturbation must surface almost unattenuated in max_o.
                    bad_o[0] += 1.0f;
                    const exactness_result bad = check_merge_exactness(D, n_q, n_qh,
                            base.o, base.lse, ovf.o, ovf.lse, bad_o, m.lse, ref.empty);
                    CHECK(bad.all_finite);
                    CHECK(bad.max_o > 0.5);
                    CHECK(bad.max_o <= kGateMergeExactO + 1.0 + 1e-3);
                }
                g_case = tag;
            }
        }
        CHECK(max_abs <= kGateMaxAbsO);
        CHECK(max_lse <= kGateMaxAbsLse);
        ++executed;

        // One range entirely masked: the merged output must equal the local
        // range alone and the merged LSE must equal the local LSE.
        {
            g_case = std::string(tag) + " overflow masked";
            const merge_result mm = run_merge(vk, vk, D, n_q, n_qh, n_kvh,
                    sh.n_kv1, sh.n_kv2, GGML_TYPE_Q4_0, true, 2,
                    q, k1, v1, k2, v2, k1b, v1b, k2b, v2b);
            CHECK(!mm.declined);
            if (!mm.declined) {
                const ref_out r1 = oracle(D, n_q, n_qh, n_kvh, q, k1d, v1d, sh.n_kv1,
                        k2d, v2d, sh.n_kv2, true, 2);
                for (size_t i = 0; i < mm.o.size(); ++i) {
                    CHECK(std::isfinite(mm.o[i]));
                }
                const double n1 = nrmse_of(mm.o, r1.o);
                double l1 = 0.0;
                for (size_t i = 0; i < mm.lse.size(); ++i) {
                    // The local range is fully visible here, so every row is
                    // valid: non-finite values fail instead of being skipped.
                    CHECK(i < r1.lse.size());
                    if (i >= r1.lse.size()) {
                        break;
                    }
                    CHECK(std::isfinite(mm.lse[i]));
                    CHECK(std::isfinite(r1.lse[i]));
                    l1 = std::max(l1, std::fabs(double(mm.lse[i]) - r1.lse[i]));
                }
                std::printf("    overflow masked: nrmse=%.3e maxLSE=%.3e\n", n1, l1);
                // One range masked out: O and LSE must match the local range's
                // own single-range output on the same backend.
                double l_exact = 0.0;
                for (size_t i = 0; i < base.lse.size() && i < mm.lse.size(); ++i) {
                    CHECK(std::isfinite(base.lse[i]));
                    CHECK(std::isfinite(mm.lse[i]));
                    l_exact = std::max(l_exact, std::fabs(double(base.lse[i]) - double(mm.lse[i])));
                }
                std::printf("    overflow masked vs local alone: maxLSE=%.3e\n", l_exact);
                CHECK(n1 <= std::max(1.05 * std::max(base_nrmse, 1e-4), 1e-4));
                CHECK(l1 <= kGateMaxAbsLse);
                CHECK(l_exact <= kGateMergeExactLse);
                CHECK(mm.attn_nodes == 2);          // the second FA really ran
                CHECK(mm.fa1_on_requested);
                CHECK(mm.fa2_on_requested);
            }
        }

        // Both ranges entirely masked: O must be exactly zero and LSE exactly
        // -inf (plan 3.4).
        {
            g_case = std::string(tag) + " both masked";
            const merge_result mb = run_merge(vk, vk, D, n_q, n_qh, n_kvh,
                    sh.n_kv1, sh.n_kv2, GGML_TYPE_Q4_0, false, 2,
                    q, k1, v1, k2, v2, k1b, v1b, k2b, v2b);
            CHECK(!mb.declined);
            if (!mb.declined) {
                double max_o = 0.0;
                bool all_neg_inf = true;
                for (size_t i = 0; i < mb.o.size(); ++i) {
                    max_o = std::max(max_o, std::fabs(double(mb.o[i])));
                }
                for (size_t i = 0; i < mb.lse.size(); ++i) {
                    if (!(std::isinf(mb.lse[i]) && mb.lse[i] < 0.0f)) {
                        all_neg_inf = false;
                    }
                }
                std::printf("    both masked: maxO=%.3e all_lse_neg_inf=%d\n", max_o, int(all_neg_inf));
                CHECK(max_o == 0.0);                 // exact zero
                CHECK(all_neg_inf);                  // exact -inf
            }
        }
    }

    // Below P: a single range, no second attention node and nothing executed on
    // the overflow backend. Local range on the CPU so "no work on the second
    // backend" is observable.
    static const int bp_nkv[] = { 8, 128, 512 };
    {
        g_case = "below P";
        for (const int n_kv : bp_nkv) {
        const int D = 256, n_q = 2, n_qh = 24, n_kvh = 4;
        std::vector<float> q(size_t(D) * n_q * n_qh);
        std::vector<float> k(size_t(D) * n_kv * n_kvh);
        std::vector<float> v(size_t(D) * n_kv * n_kvh);
        for (auto & x : q) { x = dist(rng); }
        for (auto & x : k) { x = dist(rng); }
        for (auto & x : v) { x = dist(rng); }
        std::vector<uint8_t> kb, vb;
        std::vector<float>   kd, vd;
        pack_range(GGML_TYPE_F16, D, k, kb, kd);
        pack_range(GGML_TYPE_F16, D, v, vb, vd);
        std::vector<float> k2, v2;
        // Local range on the CPU and the "overflow" backend is Vulkan: below P
        // the graph must contain one attention node and must not touch Vulkan at
        // all (plan 3.8), which is only observable with distinct backends.
        const merge_result m = run_merge(cpu, vk, D, n_q, n_qh, n_kvh,
                n_kv, 0, GGML_TYPE_Q4_0, true, 0, q, k, v, k2, v2, kb, vb, kb, vb);
        CHECK(!m.declined);
        if (!m.declined) {
            const ref_out ref = oracle(D, n_q, n_qh, n_kvh, q, kd, vd, n_kv,
                    kd, vd, 0, true, 0);
            for (size_t i = 0; i < m.o.size(); ++i) {
                CHECK(std::isfinite(m.o[i]));
            }
            const double nrmse = nrmse_of(m.o, ref.o);
            double max_lse = 0.0;
            for (size_t i = 0; i < m.lse.size(); ++i) {
                CHECK(i < ref.lse.size());
                if (i >= ref.lse.size()) {
                    break;
                }
                CHECK(std::isfinite(m.lse[i]));
                CHECK(std::isfinite(ref.lse[i]));
                max_lse = std::max(max_lse, std::fabs(double(m.lse[i]) - ref.lse[i]));
            }
            std::printf("  below P nkv=%d: nrmse=%.3e maxLSE=%.3e attn=%d attn_on_ovf=%d nodes_on_ovf=%d\n",
                    n_kv,
                    nrmse, max_lse, m.attn_nodes, m.attn_nodes_on_overflow, m.nodes_on_overflow);
            CHECK(m.attn_nodes == 1);          // a single FA node, no merge
            CHECK(m.attn_nodes_on_overflow == 0);
            CHECK(m.nodes_on_overflow == 0);   // no copy or kernel on the Radeon
            // The CPU FA is a stand-in for the local range here; its fidelity
            // gate lives in tests/test-kvarn for the product's KVarN route.
            CHECK(std::isfinite(nrmse));
            CHECK(std::isfinite(max_lse));
            ++executed;
        }
        }
    }

    // Every mandatory shape must have executed: the merge shapes plus each
    // below-P size.
    CHECK(executed == int(sizeof(shapes)/sizeof(shapes[0])) + int(sizeof(bp_nkv)/sizeof(bp_nkv[0])));

    ggml_backend_free(cuda);
    ggml_backend_free(vk);
    ggml_backend_free(cpu);

    std::printf("position-split-merge-graph: %d checks, %d failures, %d mandatory shapes\n",
            g_checks, g_failures, executed);
    return g_failures == 0 ? 0 : 1;
}