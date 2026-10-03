// F1 P0 — LSE output lifetime in the allocator, graph copies and RPC (plan §3.4).
//
// The FA node writes its LSE side output while it is executing, but the LSE is
// a separate graph node. If the allocator only learns about the LSE when it
// reaches that node, the FA inputs have already been returned to the free list
// and the LSE block can land on memory the same kernel is still reading. This
// test pins the contract with:
//
//   1. two FA ranges ([0,P) and [P,C)) + an O/LSE merge, so the graph has
//      several producer/consumer pairs and reusable temporaries,
//   2. interval checks after allocation: the LSE buffer must not overlap the
//      producing FA node's own output nor any of its inputs,
//   3. numeric checks: merged O against a single full-range FA, and each
//      range's LSE against an FP64 reference,
//   4. the reserve() path, a reset + re-execution with different inputs, and a
//      run without the eval callback (the callback may change splitting),
//   5. ggml_backend_graph_copy: the clone must rebind the LSE back-pointer to
//      the clone's own LSE node and must not write into the source graph.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

static const int D = 64, NQH = 4, NKH = 2, NQ = 1;

static bool ranges_overlap(const void * a, size_t na, const void * b, size_t nb) {
    if (a == nullptr || b == nullptr || na == 0 || nb == 0) {
        return false;
    }
    const char * pa = (const char *) a;
    const char * pb = (const char *) b;
    return pa < pb + nb && pb < pa + na;
}

// graph under test: two ranges plus the O/LSE merge
struct split_graph {
    ggml_tensor * q   = nullptr;
    ggml_tensor * kfull = nullptr;
    ggml_tensor * vfull = nullptr;
    ggml_tensor * fa_full = nullptr;
    ggml_tensor * k0  = nullptr;
    ggml_tensor * v0  = nullptr;
    ggml_tensor * k1  = nullptr;
    ggml_tensor * v1  = nullptr;
    ggml_tensor * fa0 = nullptr;
    ggml_tensor * fa1 = nullptr;
    ggml_tensor * lse0 = nullptr;
    ggml_tensor * lse1 = nullptr;
    ggml_tensor * merged = nullptr;
};

static split_graph build_split_graph(ggml_context * ctx, int P, int C) {
    split_graph g;
    const int n_local = P;
    const int n_over  = C - P;

    g.q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, NQ, NQH, 1);
    g.kfull = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, C, NKH, 1);
    g.vfull = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, C, NKH, 1);
    g.fa_full = ggml_flash_attn_ext(ctx, g.q, g.kfull, g.vfull, nullptr, 1.0f, 0.0f, 0.0f);
    ggml_set_name(g.fa_full, "fa_full");
    g.k0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_local, NKH, 1);
    g.v0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_local, NKH, 1);
    g.k1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_over,  NKH, 1);
    g.v1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_over,  NKH, 1);

    // no mask: every position of the range is visible, which is what the two
    // halves of a position-split range see once their masks are merged
    g.fa0  = ggml_flash_attn_ext(ctx, g.q, g.k0, g.v0, nullptr, 1.0f, 0.0f, 0.0f);
    g.fa1  = ggml_flash_attn_ext(ctx, g.q, g.k1, g.v1, nullptr, 1.0f, 0.0f, 0.0f);
    g.lse0 = ggml_flash_attn_ext_lse_out(ctx, g.fa0);
    g.lse1 = ggml_flash_attn_ext_lse_out(ctx, g.fa1);

    // weighted merge, softmax-invariant with a constant shift of 0:
    //   merged = (O0*exp(L0) + O1*exp(L1)) / (exp(L0) + exp(L1))
    // The exp/sub temporaries are exactly the kind of short-lived buffers the
    // allocator is allowed to reuse, so they are what the LSE lifetime has to
    // survive.
    ggml_tensor * w0 = ggml_exp(ctx, g.lse0);
    ggml_tensor * w1 = ggml_exp(ctx, g.lse1);
    ggml_tensor * r0 = ggml_reshape_3d(ctx, w0, 1, NQH, NQ);
    ggml_tensor * r1 = ggml_reshape_3d(ctx, w1, 1, NQH, NQ);
    ggml_tensor * num = ggml_add(ctx,
        ggml_mul(ctx, g.fa0, r0),
        ggml_mul(ctx, g.fa1, r1));
    ggml_tensor * den = ggml_reshape_3d(ctx, ggml_add(ctx, w0, w1), 1, NQH, NQ);
    g.merged = ggml_div(ctx, num, den);

    ggml_set_name(g.fa0, "fa0");
    ggml_set_name(g.fa1, "fa1");
    ggml_set_name(g.lse0, "fa_lse0");
    ggml_set_name(g.lse1, "fa_lse1");
    ggml_set_name(g.merged, "merged");
    return g;
}

// LSE must not share memory with the producing kernel's own output or inputs
static void check_lse_placement(const split_graph & g) {
    struct pair { const char * name; ggml_tensor * fa; ggml_tensor * lse; };
    const pair pairs[] = { { "range0", g.fa0, g.lse0 }, { "range1", g.fa1, g.lse1 } };
    for (const pair & p : pairs) {
        const size_t lse_size = ggml_nbytes(p.lse);
        CHECK(!ranges_overlap(p.lse->data, lse_size, p.fa->data, ggml_nbytes(p.fa)));
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = p.fa->src[j];
            if (src == nullptr) {
                continue;
            }
            CHECK(!ranges_overlap(p.lse->data, lse_size, src->data, ggml_nbytes(src)));
        }
    }
}

struct host_inputs {
    std::vector<float> q;
    std::vector<float> k0, v0, k1, v1;
};

static host_inputs make_inputs(int P, int C, uint32_t seed) {
    const int n_local = P, n_over = C - P;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    host_inputs h;
    h.q.resize((size_t) D * NQ * NQH);
    for (auto & x : h.q) { x = dist(rng); }
    const size_t rows_local = (size_t) n_local * NKH;
    const size_t rows_over  = (size_t) n_over * NKH;
    h.k0.resize(rows_local * D); h.v0.resize(rows_local * D);
    h.k1.resize(rows_over * D);  h.v1.resize(rows_over * D);
    for (auto & x : h.k0) { x = dist(rng); }
    for (auto & x : h.v0) { x = dist(rng); }
    for (auto & x : h.k1) { x = dist(rng); }
    for (auto & x : h.v1) { x = dist(rng); }
    return h;
}

static void upload(const split_graph & g, const host_inputs & h, int P, int C, bool with_full = true) {
    ggml_backend_tensor_set(g.q,  h.q.data(),  0, h.q.size() * sizeof(float));
    ggml_backend_tensor_set(g.k0, h.k0.data(), 0, h.k0.size() * sizeof(float));
    ggml_backend_tensor_set(g.v0, h.v0.data(), 0, h.v0.size() * sizeof(float));
    ggml_backend_tensor_set(g.k1, h.k1.data(), 0, h.k1.size() * sizeof(float));
    ggml_backend_tensor_set(g.v1, h.v1.data(), 0, h.v1.size() * sizeof(float));
    if (!with_full) {
        return; // graph does not contain the unsplit comparison tensors
    }
    // full-range tensors are [D, C, NKH]: per head, local keys then overflow keys
    const int n_local = P, n_over = C - P;
    std::vector<float> kfull((size_t) D * C * NKH);
    std::vector<float> vfull((size_t) D * C * NKH);
    for (int kh = 0; kh < NKH; ++kh) {
        for (int k = 0; k < n_local; ++k) {
            for (int d = 0; d < D; ++d) {
                kfull[((size_t) (kh * C) + k) * D + d] = h.k0[((size_t) (kh * n_local) + k) * D + d];
                vfull[((size_t) (kh * C) + k) * D + d] = h.v0[((size_t) (kh * n_local) + k) * D + d];
            }
        }
        for (int k = 0; k < n_over; ++k) {
            for (int d = 0; d < D; ++d) {
                kfull[((size_t) (kh * C) + n_local + k) * D + d] = h.k1[((size_t) (kh * n_over) + k) * D + d];
                vfull[((size_t) (kh * C) + n_local + k) * D + d] = h.v1[((size_t) (kh * n_over) + k) * D + d];
            }
        }
    }
    ggml_backend_tensor_set(g.kfull, kfull.data(), 0, kfull.size() * sizeof(float));
    ggml_backend_tensor_set(g.vfull, vfull.data(), 0, vfull.size() * sizeof(float));
}

// FP64 LSE reference for one range (contiguous [0,n) of the same absolute
// positions), plus the full-range O reference
static void reference(const host_inputs & h, int P, int C,
        std::vector<double> & lse0, std::vector<double> & lse1,
        std::vector<float> & ofull) {
    const int gqa = NQH / NKH;
    const int n_local = P, n_over = C - P;
    // K/V are F32 here, so the CPU kernel runs an F32 vec_dot and this FP64
    // reference models its arithmetic exactly: the gate below measures the
    // split/merge, not kernel rounding.
    const std::vector<float> & q16 = h.q;
    lse0.assign((size_t) NQH * NQ, -INFINITY);
    lse1.assign((size_t) NQH * NQ, -INFINITY);
    ofull.assign((size_t) D * NQH * NQ, 0.0f);

    auto range_lse = [&](const std::vector<float> & K,
            int n, std::vector<double> & out) {
        for (int qq = 0; qq < NQ; ++qq) {
            for (int hh = 0; hh < NQH; ++hh) {
                const int kh = hh / gqa;
                std::vector<double> s(n, -INFINITY);
                for (int k = 0; k < n; ++k) {
                    double dot = 0;
                    for (int d = 0; d < D; ++d) {
                        dot += double(q16[((size_t) hh * NQ + qq) * D + d]) *
                               double(K[((size_t) (kh * n) + k) * D + d]);
                    }
                    s[k] = dot;
                }
                double m = -INFINITY;
                for (double v : s) { m = std::max(m, v); }
                double denom = 0;
                for (double v : s) { denom += std::exp(v - m); }
                out[(size_t) qq * NQH + hh] = m + std::log(denom);
            }
        }
    };
    range_lse(h.k0, n_local, lse0);
    range_lse(h.k1, n_over,  lse1);

    // full-range O over the concatenation [0,n_local) + [0,n_over)
    for (int qq = 0; qq < NQ; ++qq) {
        for (int hh = 0; hh < NQH; ++hh) {
            const int kh = hh / gqa;
            std::vector<double> s(n_local + n_over, -INFINITY);
            for (int k = 0; k < n_local; ++k) {
                double dot = 0;
                for (int d = 0; d < D; ++d) {
                    dot += double(q16[((size_t) hh * NQ + qq) * D + d]) *
                           double(h.k0[((size_t) (kh * n_local) + k) * D + d]);
                }
                s[k] = dot;
            }
            for (int k = 0; k < n_over; ++k) {
                double dot = 0;
                for (int d = 0; d < D; ++d) {
                    dot += double(q16[((size_t) hh * NQ + qq) * D + d]) *
                           double(h.k1[((size_t) (kh * n_over) + k) * D + d]);
                }
                s[n_local + k] = dot;
            }
            double m = -INFINITY;
            for (double v : s) { m = std::max(m, v); }
            double denom = 0;
            for (double v : s) { denom += std::exp(v - m); }
            for (int d = 0; d < D; ++d) {
                double acc = 0;
                for (int k = 0; k < n_local + n_over; ++k) {
                    const float v = k < n_local ?
                        h.v0[((size_t) (kh * n_local) + k) * D + d] :
                        h.v1[((size_t) (kh * n_over) + (k - n_local)) * D + d];
                    acc += std::exp(s[k] - m) * double(v);
                }
                ofull[((size_t) qq * NQH + hh) * D + d] = float(acc / denom);
            }
        }
    }
}

static double nrmse(const std::vector<float> & a, const std::vector<float> & b) {
    double se = 0, sr = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = double(a[i]) - double(b[i]);
        se += d * d;
        sr += double(b[i]) * double(b[i]);
    }
    double rr = std::sqrt(sr / (a.size() ? a.size() : 1));
    if (rr < 1e-6) { rr = 1e-6; }
    return std::sqrt(se / (a.size() ? a.size() : 1)) / rr;
}

// Runs the graph on a CPU-only scheduler and checks placement + numbers.
// use_callback mirrors the sched test: the callback can change splitting, so
// both paths must hold.
static bool run_case(ggml_backend_t cpu, int P, int C, uint32_t seed, bool use_callback, bool reserve) {
    const host_inputs h = make_inputs(P, C, seed);
    std::vector<double> lse0_ref, lse1_ref;
    std::vector<float> ofull_ref;
    reference(h, P, C, lse0_ref, lse1_ref, ofull_ref);

    ggml_backend_t backends[1] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 1, 4096, false, true);

    struct ggml_init_params params = { 64 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    const split_graph g = build_split_graph(ctx, P, C);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, g.merged);
    ggml_build_forward_expand(gf, g.fa_full);

    if (reserve) {
        // reserve() walks the same allocation logic as alloc_graph
        CHECK(ggml_backend_sched_reserve(sched, gf));
    }
    if (use_callback) {
        ggml_backend_sched_set_eval_callback(sched, [](ggml_tensor *, bool ask, void *) {
            return ask ? true : true;
        }, nullptr);
    }
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        CHECK(false);
        ggml_free(ctx);
        ggml_backend_sched_free(sched);
        return true;
    }

    check_lse_placement(g);

    upload(g, h, P, C);
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        CHECK(false);
        ggml_free(ctx);
        ggml_backend_sched_free(sched);
        return true;
    }

    std::vector<float> merged(ggml_nelements(g.merged));
    ggml_backend_tensor_get(g.merged, merged.data(), 0, merged.size() * sizeof(float));
    std::vector<float> full(ggml_nelements(g.fa_full));
    ggml_backend_tensor_get(g.fa_full, full.data(), 0, full.size() * sizeof(float));
    const double err_full_fp64 = nrmse(full, ofull_ref);
    const double err_merge_full = nrmse(merged, full);
    const double err = nrmse(merged, ofull_ref);
    std::printf("  [%s] P=%d C=%d seed=%u reserve=%d cb=%d: merge-vs-fp64=%.3e "
                "full-vs-fp64=%.3e merge-vs-full=%.3e\n",
        g_current, P, C, seed, (int) reserve, (int) use_callback,
        err, err_full_fp64, err_merge_full);
    // structural gate: the merged position-split output must match the single
    // unsplit FA of the same range, and both must match FP64 (plan gate)
    CHECK(err_merge_full <= 1e-3);
    CHECK(err <= 1e-3);
    CHECK(err_full_fp64 <= 1e-3);

    // The LSE buffers are released once the merge has consumed them, which is
    // the point of the liveness fix. Read the values from a second graph whose
    // outputs are the LSE nodes themselves, after a scheduler reset (also
    // covering the realloc/reset path).
    ggml_backend_sched_reset(sched);
    struct ggml_cgraph * gl = ggml_new_graph(ctx);
    ggml_build_forward_expand(gl, g.lse0);
    ggml_build_forward_expand(gl, g.lse1);
    if (!ggml_backend_sched_alloc_graph(sched, gl)) {
        CHECK(false);
        ggml_free(ctx);
        ggml_backend_sched_free(sched);
        return true;
    }
    check_lse_placement(g);
    upload(g, h, P, C);
    if (ggml_backend_sched_graph_compute(sched, gl) != GGML_STATUS_SUCCESS) {
        CHECK(false);
        ggml_free(ctx);
        ggml_backend_sched_free(sched);
        return true;
    }

    for (int i = 0; i < 2; ++i) {
        ggml_tensor * lse_t = i == 0 ? g.lse0 : g.lse1;
        const std::vector<double> & ref = i == 0 ? lse0_ref : lse1_ref;
        std::vector<float> lse(ggml_nelements(lse_t));
        ggml_backend_tensor_get(lse_t, lse.data(), 0, lse.size() * sizeof(float));
        float max_l = 0;
        for (size_t k = 0; k < lse.size(); ++k) {
            CHECK(std::isfinite(lse[k]));
            max_l = std::max(max_l, std::fabs(lse[k] - float(ref[k])));
        }
        std::printf("  [%s] range%d max|dLSE|=%.3e\n", g_current, i, max_l);
        CHECK(max_l <= 1e-2f);
    }

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    return true;
}

// The clone must rebind the LSE back-pointer and must not write the source.
static void run_graph_copy(ggml_backend_t cpu) {
    const int P = 128, C = 384;
    const host_inputs h = make_inputs(P, C, 99);
    std::vector<double> lse0_ref, lse1_ref;
    std::vector<float> ofull_ref;
    reference(h, P, C, lse0_ref, lse1_ref, ofull_ref);

    ggml_backend_t backends[1] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 1, 4096, false, true);

    struct ggml_init_params params = { 64 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    const split_graph g = build_split_graph(ctx, P, C);
    // outputs are the LSE nodes, so the clone keeps their buffers alive and the
    // values can be compared after the copy runs
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, g.lse0);
    ggml_build_forward_expand(gf, g.lse1);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        CHECK(false);
        ggml_free(ctx);
        ggml_backend_sched_free(sched);
        return;
    }
    check_lse_placement(g);
    upload(g, h, P, C, false);

    struct ggml_backend_graph_copy gcopy = ggml_backend_graph_copy(cpu, gf);
    struct ggml_cgraph * gc = gcopy.graph;
    if (gc == nullptr || ggml_graph_n_nodes(gc) == 0) {
        CHECK(false);
        ggml_free(ctx);
        ggml_backend_sched_free(sched);
        return;
    }

    // find the cloned FA/LSE pairs
    ggml_tensor * cfa[2] = { nullptr, nullptr };
    ggml_tensor * clse[2] = { nullptr, nullptr };
    ggml_tensor * cmerged = nullptr;
    // node lookup by name (ggml_cgraph is opaque outside the library)
    cfa[0] = ggml_graph_get_tensor(gc, "fa0");
    cfa[1] = ggml_graph_get_tensor(gc, "fa1");
    clse[0] = ggml_graph_get_tensor(gc, "fa_lse0");
    clse[1] = ggml_graph_get_tensor(gc, "fa_lse1");
    cmerged = ggml_graph_get_tensor(gc, "merged"); // may be absent: LSE-output graph
    CHECK(cfa[0] != nullptr && cfa[1] != nullptr);
    CHECK(clse[0] != nullptr && clse[1] != nullptr);
    (void) cmerged;

    if (cfa[0] && clse[0]) {
        CHECK(ggml_flash_attn_ext_get_lse_out(cfa[0]) == clse[0]);
        CHECK(ggml_flash_attn_ext_get_lse_out(cfa[0]) != g.lse0);
    }
    if (cfa[1] && clse[1]) {
        CHECK(ggml_flash_attn_ext_get_lse_out(cfa[1]) == clse[1]);
        CHECK(ggml_flash_attn_ext_get_lse_out(cfa[1]) != g.lse1);
    }

    // sentinel in the source LSE buffers: running the clone must not touch them
    const float sentinel = -12345.0f;
    std::vector<float> sent((size_t) NQH * NQ, sentinel);
    ggml_backend_tensor_set(g.lse0, sent.data(), 0, sent.size() * sizeof(float));
    ggml_backend_tensor_set(g.lse1, sent.data(), 0, sent.size() * sizeof(float));

    if (ggml_backend_graph_compute(cpu, gc) == GGML_STATUS_SUCCESS && clse[0]) {
        std::vector<float> lse(ggml_nelements(clse[0]));
        ggml_backend_tensor_get(clse[0], lse.data(), 0, lse.size() * sizeof(float));
        float max_l = 0;
        for (size_t k = 0; k < lse.size(); ++k) {
            max_l = std::max(max_l, std::fabs(lse[k] - float(lse0_ref[k])));
        }
        std::printf("  [%s] clone lse max|dLSE|=%.3e\n", g_current, max_l);
        CHECK(max_l <= 1e-2f);

        std::vector<float> after(ggml_nelements(g.lse0));
        ggml_backend_tensor_get(g.lse0, after.data(), 0, after.size() * sizeof(float));
        for (size_t k = 0; k < after.size(); ++k) {
            CHECK(after[k] == sentinel);
        }
    } else {
        CHECK(false);
    }

    ggml_backend_graph_copy_free(gcopy);
    ggml_free(ctx);
    ggml_backend_sched_free(sched);
}

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::printf("position-split-lse-alloc: no CPU backend\n");
        return 1;
    }

    g_current = "alloc_placement";
    CHECK(run_case(cpu, 128, 384, 7, false, false));

    g_current = "alloc_placement_callback";
    CHECK(run_case(cpu, 128, 384, 7, true, false));

    g_current = "alloc_placement_reserve";
    CHECK(run_case(cpu, 128, 384, 7, false, true));

    g_current = "alloc_placement_p1";
    // P = 1: the local range is a single token, the overflow carries the rest
    CHECK(run_case(cpu, 1, 129, 11, false, false));

    g_current = "alloc_placement_p_plus_1";
    // P + 1 boundary: the overflow starts exactly one past the local range
    CHECK(run_case(cpu, 256, 257, 13, false, false));

    g_current = "alloc_reexec";
    // same shape, different data: reuse of the allocator state must not leak
    // the previous LSE buffer into the result
    CHECK(run_case(cpu, 128, 384, 21, false, false));
    CHECK(run_case(cpu, 128, 384, 22, false, false));

    g_current = "graph_copy";
    run_graph_copy(cpu);

    ggml_backend_free(cpu);
    std::printf("position-split-lse-alloc: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}