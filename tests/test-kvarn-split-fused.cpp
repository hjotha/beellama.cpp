// Phase-2 GPU test for GGML_OP_KVARN_SPLIT_FUSED ("peça (c)"): the fused
// position-split overflow span must reproduce the legacy two-range chain within
// the phase-2 tolerances.
//
// The test builds, for D = 128 and D = 256 decode shapes (Q = 1, GQA):
//   * the legacy chain -- local F16 FA1/LSE1 on CUDA, overflow Q4 store
//     (SET_ROWS) + F16 gather + FA2/LSE2 on Vulkan, LSE merge -- inherited
//     structurally from tests/test-position-split-merge-graph.cpp, and
//   * the fused graph -- the same local FA1/LSE1, then ONE
//     GGML_OP_KVARN_SPLIT_FUSED node in place of store + gather + FA2/LSE2 +
//     merge, executed by the llama-side executor registered through
//     llama_kvarn_split_fused_exec_register().
//
// Gates:
//   * fused O vs legacy O:  maxabs <= 1e-3 (phase-2 gate; CUDA vs Vulkan
//     exp/log rounding makes bit-equality the wrong contract),
//   * legacy merged LSE vs an FP64 oracle: maxabs <= 1e-2,
//   * the fused graph must keep the overflow FA out of the outer graph
//     (exactly one FLASH_ATTN_EXT node) and must not add scheduler splits.
//
// Requires CUDA + Vulkan + CPU. Exits non-zero on missing backends or gates.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include "llama-kvarn-split-fused.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::printf("  FAIL [%s] line %d: %s\n", g_case.c_str(), __LINE__, #cond); \
    } \
} while (0)

static std::string g_case;

static void set_env_fused_span() {
#ifdef _WIN32
    _putenv_s("LLAMA_OVF_FUSED_SPAN", "1");
#else
    setenv("LLAMA_OVF_FUSED_SPAN", "1", 1);
#endif
}

// ------------------------------------------------------------ host packing --

// Packs a host range into `type` and returns the dequantized values of exactly
// those bytes (row advance = ggml_row_size, as in the merge-graph harness).
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

// ----------------------------------------------------------------- oracle ---

// FP64 attention over the effective bytes both legs consume: local F16 rows
// concatenated with the overflow rows (prefilled storage + the stored current
// row), all fully visible. Q = 1, so queries sit at the end of the sequence.
struct oracle_out {
    std::vector<double> o;   // [D, Hq]
    std::vector<double> lse; // [Hq]
};

static oracle_out fp64_oracle(int D, int Hq, int Hkv,
        const std::vector<float> & q,        // [D, Hq]
        const std::vector<float> & k_local,  // [D, n_local*Hkv] dequantized
        const std::vector<float> & v_local,
        int n_local,
        const std::vector<float> & k_ovf,    // [D, n_ovf*Hkv] dequantized
        const std::vector<float> & v_ovf,
        int n_ovf) {
    const int gqa = Hq / Hkv;
    const double scale = 1.0 / std::sqrt(double(D));
    oracle_out out;
    out.o.assign(size_t(D) * Hq, 0.0);
    out.lse.assign(size_t(Hq), -INFINITY);
    for (int h = 0; h < Hq; ++h) {
        const int kh = h / gqa;
        std::vector<double> scores(size_t(n_local) + n_ovf);
        double max_score = -INFINITY;
        for (int s = 0; s < n_local; ++s) {
            const size_t koff = (size_t(s) + size_t(kh) * n_local) * D;
            double acc = 0.0;
            for (int d = 0; d < D; ++d) {
                acc += double(q[size_t(h) * D + d]) * double(k_local[koff + d]);
            }
            scores[size_t(s)] = acc * scale;
            max_score = std::max(max_score, scores[size_t(s)]);
        }
        for (int s = 0; s < n_ovf; ++s) {
            const size_t koff = (size_t(s) + size_t(kh) * n_ovf) * D;
            double acc = 0.0;
            for (int d = 0; d < D; ++d) {
                acc += double(q[size_t(h) * D + d]) * double(k_ovf[koff + d]);
            }
            scores[size_t(n_local) + size_t(s)] = acc * scale;
            max_score = std::max(max_score, scores[size_t(n_local) + size_t(s)]);
        }
        double denom = 0.0;
        for (double sc : scores) {
            denom += std::exp(sc - max_score);
        }
        for (int d = 0; d < D; ++d) {
            double acc = 0.0;
            for (int s = 0; s < n_local; ++s) {
                const size_t voff = (size_t(s) + size_t(kh) * n_local) * D;
                acc += std::exp(scores[size_t(s)] - max_score) * double(v_local[voff + d]);
            }
            for (int s = 0; s < n_ovf; ++s) {
                const size_t voff = (size_t(s) + size_t(kh) * n_ovf) * D;
                acc += std::exp(scores[size_t(n_local) + size_t(s)] - max_score) * double(v_ovf[voff + d]);
            }
            out.o[size_t(h) * D + d] = acc / denom;
        }
        out.lse[size_t(h)] = max_score + std::log(denom);
    }
    return out;
}


// FP64 replay of the merge formula over two (O, LSE) pairs, used to check the
// merged LSE the fused span implies (o2/lse2 inside the op are the same ops on
// the same operands as the legacy leg's fa2/lse2, which the legacy leg
// exposes).
static void merge_fp64_pair(const std::vector<float> & o1, const std::vector<float> & lse1,
        const std::vector<float> & o2, const std::vector<float> & lse2,
        int D, int Hq, std::vector<double> & o_out, std::vector<double> & lse_out) {
    o_out.assign(size_t(D)*Hq, 0.0);
    lse_out.assign(size_t(Hq), 0.0);
    for (int h = 0; h < Hq; ++h) {
        const double l1 = lse1[size_t(h)];
        const double l2 = lse2[size_t(h)];
        const double mx = std::max(l1, l2);
        const double w1 = std::exp(l1 - mx);
        const double w2 = std::exp(l2 - mx);
        const double den = w1 + w2;
        for (int d = 0; d < D; ++d) {
            const size_t i = size_t(h)*D + d;
            o_out[i] = (double(o1[i])*w1 + double(o2[i])*w2) / den;
        }
        lse_out[size_t(h)] = mx + std::log(den);
    }
}

// ------------------------------------------------------------------- graph ---

// The executor can dump its internal Vulkan leg (O2/LSE2) to a file; the test
// uses it to compare the REAL span outputs, not a legacy proxy.
static const char * dump_path() {
    static const std::string path = [] {
        const char * tmp = std::getenv("TMPDIR");
        return std::string(tmp && *tmp ? tmp : "/tmp") + "/test-kvarn-split-fused-leg.dump";
    }();
    return path.c_str();
}

static void dump_truncate() {
    std::ofstream f(dump_path(), std::ios::trunc | std::ios::binary);
}

static bool dump_read_first(std::vector<float> & o2, std::vector<float> & lse2, int32_t & il) {
    std::ifstream f(dump_path(), std::ios::binary);
    if (!f) {
        return false;
    }
    int32_t ilr = 0;
    uint64_t n_o2 = 0, n_lse2 = 0;
    f.read(reinterpret_cast<char *>(&ilr), sizeof(ilr));
    f.read(reinterpret_cast<char *>(&n_o2), sizeof(n_o2));
    f.read(reinterpret_cast<char *>(&n_lse2), sizeof(n_lse2));
    if (!f || n_o2 == 0 || n_lse2 == 0 || n_o2 > (1u << 20) || n_lse2 > (1u << 20)) {
        return false;
    }
    o2.assign(size_t(n_o2), 0.0f);
    lse2.assign(size_t(n_lse2), 0.0f);
    f.read(reinterpret_cast<char *>(o2.data()), std::streamsize(n_o2 * sizeof(float)));
    f.read(reinterpret_cast<char *>(lse2.data()), std::streamsize(n_lse2 * sizeof(float)));
    il = ilr;
    return bool(f);
}

struct leg_result {
    bool declined = false;
    std::string reason;
    std::vector<float> o;    // [D*Hq]
    std::vector<float> lse;  // [Hq]
    std::vector<float> o1, lse1; // local range operands of the merge
    std::vector<float> o2, lse2; // overflow range operands of the merge
    std::vector<float> k_ovf_eff; // [D*Hkv, n_ovf] storage bytes, dequantized
    std::vector<float> v_ovf_eff;
    std::vector<uint8_t> k_store_bytes; // raw companion storage after the run
    std::vector<uint8_t> v_store_bytes;
    std::vector<float> o2_exec;   // executor's internal Vulkan O2 (fused leg)
    std::vector<float> lse2_exec; // executor's internal Vulkan LSE2 (fused leg)
    int attn_nodes = 0;
    int splits = 0;
};

// mode 0: legacy overflow chain (store + gather + FA2/LSE2 + merge).
// mode 1: fused op (local FA1/LSE1 + GGML_OP_KVARN_SPLIT_FUSED).
static leg_result run_leg(ggml_backend_t cuda, ggml_backend_t vk, int mode,
        uint32_t bundle_id,
        int D, int Hq, int Hkv, int n_local, int n_ovf, int P,
        const std::vector<float> & q,        // [D, Hq]
        const std::vector<uint8_t> & k1_bytes, const std::vector<uint8_t> & v1_bytes,
        const std::vector<uint8_t> & k_storage_bytes, const std::vector<uint8_t> & v_storage_bytes,
        const std::vector<float> & k_cur,    // [D*Hkv] current rows (stored)
        const std::vector<float> & v_cur,
        int store_row) {
    leg_result res;

    ggml_init_params params = { 256 * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        res.declined = true;
        res.reason = "context allocation failed";
        return res;
    }

    const int    n_tok  = 1;
    const float  scale  = 1.0f / std::sqrt(float(D));
    const size_t Dkv    = size_t(D) * Hkv;

    // Local range (identical in both legs): F16 K/V on CUDA. Q keeps the
    // production layout [D, H_q, Q, B]; the FA consumes the view+permute
    // build_attn_mha() emits, so both legs see the same permuted query.
    ggml_tensor * q_t  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, Hq, n_tok, 1);
    ggml_tensor * k1_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_local, Hkv, 1);
    ggml_tensor * v1_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_local, Hkv, 1);
    ggml_tensor * m1_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_local, n_tok, 1, 1);
    ggml_tensor * q_view = ggml_view_4d(ctx, q_t, D, Hq, n_tok, 1,
            q_t->nb[1], q_t->nb[2], q_t->nb[3], 0);
    ggml_tensor * qp = ggml_permute(ctx, q_view, 0, 2, 1, 3);
    ggml_tensor * fa1  = ggml_flash_attn_ext(ctx, qp, k1_t, v1_t, m1_t, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(fa1, GGML_PREC_F32);
    ggml_tensor * lse1 = ggml_flash_attn_ext_lse_out(ctx, fa1);

    // Overflow companion storage (Q4_0, Vulkan) and its packed row window.
    ggml_tensor * k_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, int64_t(Dkv), n_ovf);
    ggml_tensor * v_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, int64_t(Dkv), n_ovf);
    ggml_tensor * k_view = ggml_view_2d(ctx, k_storage, int64_t(Dkv), n_ovf, k_storage->nb[1], 0);
    ggml_tensor * v_view = ggml_view_2d(ctx, v_storage, int64_t(Dkv), n_ovf, v_storage->nb[1], 0);
    ggml_set_name(k_view, "attn_k_ovf_packed_view");
    ggml_set_name(v_view, "attn_v_ovf_packed_view");

    // Contiguous F16 crop mask for the overflow extent, one query row.
    ggml_tensor * m2_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_ovf, n_tok, 1, 1);

    // Store index list (payload row of the current token) and the current
    // rows themselves: the storage companion write both legs must perform.
    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tok);
    ggml_tensor * v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tok);
    ggml_tensor * k_cur_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, Hkv, n_tok, 1);
    ggml_tensor * v_cur_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, Hkv, n_tok, 1);

    ggml_tensor * o_out   = nullptr;
    ggml_tensor * lse_out = nullptr;
    ggml_tensor * fused   = nullptr;

    if (mode == 0) {
        // ---- legacy chain -----------------------------------------------
        ggml_tensor * idrows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_ovf);
        // SET_ROWS store of the current rows, mirroring llama_kv_cache::cpy_k.
        ggml_tensor * kcur_2d = ggml_view_2d(ctx, k_cur_t, int64_t(Dkv), n_tok, k_cur_t->nb[2], 0);
        ggml_tensor * vcur_2d = ggml_view_2d(ctx, v_cur_t, int64_t(Dkv), n_tok, v_cur_t->nb[2], 0);
        ggml_tensor * store_k = ggml_set_rows(ctx, k_storage, kcur_2d, k_idxs);
        ggml_tensor * store_v = ggml_set_rows(ctx, v_storage, vcur_2d, v_idxs);
        // F16 gather readback over the packed window with identity rows.
        ggml_tensor * k16 = ggml_get_rows_as(ctx, k_view, idrows, GGML_TYPE_F16);
        ggml_tensor * v16 = ggml_get_rows_as(ctx, v_view, idrows, GGML_TYPE_F16);
        ggml_tensor * k4 = ggml_reshape_4d(ctx, k16, D, Hkv, n_ovf, 1);
        ggml_tensor * v4 = ggml_reshape_4d(ctx, v16, D, Hkv, n_ovf, 1);
        ggml_tensor * kp = ggml_permute(ctx, k4, 0, 2, 1, 3);
        ggml_tensor * vp = ggml_permute(ctx, v4, 0, 2, 1, 3);
        ggml_tensor * fa2 = ggml_flash_attn_ext(ctx, qp, kp, vp, m2_t, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(fa2, GGML_PREC_F32);
        ggml_flash_attn_ext_set_n_kv_max(fa2, n_ovf);
        ggml_tensor * lse2 = ggml_flash_attn_ext_lse_out(ctx, fa2);

        // Merge: the exact arithmetic build_attn_position_split emits.
        ggml_tensor * o1_4d = ggml_reshape_4d(ctx, fa1, D, Hq, n_tok, 1);
        ggml_tensor * o2_4d = ggml_reshape_4d(ctx, fa2, D, Hq, n_tok, 1);
        ggml_tensor * c1 = ggml_clamp(ctx, lse1, -1e6f, 1e30f);
        ggml_tensor * c2 = ggml_clamp(ctx, lse2, -1e6f, 1e30f);
        ggml_tensor * mx = ggml_scale(ctx,
                ggml_add(ctx, ggml_add(ctx, c1, c2), ggml_abs(ctx, ggml_sub(ctx, c1, c2))),
                0.5f);
        ggml_tensor * e1 = ggml_exp(ctx, ggml_sub(ctx, lse1, mx));
        ggml_tensor * e2 = ggml_exp(ctx, ggml_sub(ctx, lse2, mx));
        ggml_tensor * w1 = ggml_reshape_4d(ctx, e1, 1, Hq, n_tok, 1);
        ggml_tensor * w2 = ggml_reshape_4d(ctx, e2, 1, Hq, n_tok, 1);
        ggml_tensor * num = ggml_add(ctx, ggml_mul(ctx, o1_4d, w1), ggml_mul(ctx, o2_4d, w2));
        ggml_tensor * den = ggml_add(ctx, e1, e2);
        ggml_tensor * den_4d = ggml_reshape_4d(ctx, den, 1, Hq, n_tok, 1);
        ggml_tensor * o_4d = ggml_div(ctx, num, den_4d);
        ggml_tensor * lse_m = ggml_add(ctx, mx, ggml_log(ctx, den));
        ggml_tensor * o_2d = ggml_reshape_2d(ctx, o_4d, D*Hq, n_tok);
        o_out = ggml_cont_2d(ctx, o_2d, D*Hq, n_tok);
        lse_out = lse_m;

        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_t backends[3] = { vk, cuda, cpu };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 3, 8192, false, true);
        ggml_backend_sched_set_tensor_backend(sched, q_t, cuda);
        ggml_backend_sched_set_tensor_backend(sched, k1_t, cuda);
        ggml_backend_sched_set_tensor_backend(sched, v1_t, cuda);
        ggml_backend_sched_set_tensor_backend(sched, m1_t, vk);
        // The generic (non-KVarN) CUDA FA declines the LSE export by design
        // (ggml_cuda_flash_attn_ext_supported), so the local range runs on the
        // Radeon exactly like the merge-graph harness's oracle legs.
        ggml_backend_sched_set_tensor_backend(sched, fa1, vk);
        ggml_backend_sched_set_tensor_backend(sched, lse1, vk);
        ggml_backend_sched_set_tensor_backend(sched, k_storage, vk);
        ggml_backend_sched_set_tensor_backend(sched, v_storage, vk);
        ggml_backend_sched_set_tensor_backend(sched, k_view, vk);
        ggml_backend_sched_set_tensor_backend(sched, v_view, vk);
        ggml_backend_sched_set_tensor_backend(sched, m2_t, vk);
        ggml_backend_sched_set_tensor_backend(sched, idrows, vk);
        ggml_backend_sched_set_tensor_backend(sched, k_cur_t, vk);
        ggml_backend_sched_set_tensor_backend(sched, v_cur_t, vk);
        ggml_backend_sched_set_tensor_backend(sched, k_idxs, vk);
        ggml_backend_sched_set_tensor_backend(sched, v_idxs, vk);
        ggml_backend_sched_set_tensor_backend(sched, store_k, vk);
        ggml_backend_sched_set_tensor_backend(sched, store_v, vk);
        ggml_backend_sched_set_tensor_backend(sched, k16, vk);
        ggml_backend_sched_set_tensor_backend(sched, v16, vk);
        ggml_backend_sched_set_tensor_backend(sched, fa2, vk);
        ggml_backend_sched_set_tensor_backend(sched, lse2, vk);
        ggml_backend_sched_set_tensor_backend(sched, o_out, cuda);
        ggml_backend_sched_set_tensor_backend(sched, lse_out, cuda);

        // Roots read after compute must survive the allocator's buffer
        // recycling (same rule as the merge-graph harness's fixture leaves).
        ggml_set_output(o_out);
        ggml_set_output(lse_out);
        ggml_set_output(fa1);
        ggml_set_output(lse1);
        ggml_set_output(fa2);
        ggml_set_output(lse2);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, store_k);
        ggml_build_forward_expand(gf, store_v);
        ggml_build_forward_expand(gf, fa1);
        ggml_build_forward_expand(gf, fa2);
        ggml_build_forward_expand(gf, lse1);
        ggml_build_forward_expand(gf, lse2);
        ggml_build_forward_expand(gf, o_4d);
        ggml_build_forward_expand(gf, lse_m);
        ggml_build_forward_expand(gf, o_out);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            ggml_backend_sched_free(sched);
            ggml_backend_free(cpu);
            ggml_free(ctx);
            res.declined = true;
            res.reason = "sched_alloc_graph failed (legacy)";
            return res;
        }

        ggml_backend_tensor_set(q_t, q.data(), 0, ggml_nbytes(q_t));
        ggml_backend_tensor_set(k1_t, k1_bytes.data(), 0, ggml_nbytes(k1_t));
        ggml_backend_tensor_set(v1_t, v1_bytes.data(), 0, ggml_nbytes(v1_t));
        std::vector<uint8_t> storage_k = k_storage_bytes;
        std::vector<uint8_t> storage_v = v_storage_bytes;
        ggml_backend_tensor_set(k_storage, storage_k.data(), 0, ggml_nbytes(k_storage));
        ggml_backend_tensor_set(v_storage, storage_v.data(), 0, ggml_nbytes(v_storage));
        {
            std::vector<ggml_fp16_t> m1(size_t(n_local), ggml_fp32_to_fp16(0.0f));
            std::vector<ggml_fp16_t> m2(size_t(n_ovf), ggml_fp32_to_fp16(0.0f));
            ggml_backend_tensor_set(m1_t, m1.data(), 0, ggml_nbytes(m1_t));
            ggml_backend_tensor_set(m2_t, m2.data(), 0, ggml_nbytes(m2_t));
        }
        {
            std::vector<int32_t> rows;
            rows.reserve(size_t(n_ovf));
            for (int i = 0; i < n_ovf; ++i) { rows.push_back(i); }
            ggml_backend_tensor_set(idrows, rows.data(), 0, ggml_nbytes(idrows));
        }
        ggml_backend_tensor_set(k_cur_t, k_cur.data(), 0, ggml_nbytes(k_cur_t));
        ggml_backend_tensor_set(v_cur_t, v_cur.data(), 0, ggml_nbytes(v_cur_t));
        {
            const int64_t row = store_row;
            ggml_backend_tensor_set(k_idxs, &row, 0, sizeof(row));
            ggml_backend_tensor_set(v_idxs, &row, 0, sizeof(row));
        }

        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            if (ggml_graph_node(gf, i)->op == GGML_OP_FLASH_ATTN_EXT) {
                ++res.attn_nodes;
            }
        }
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            res.declined = true;
            res.reason = "sched_graph_compute failed (legacy)";
        } else {
            res.o.assign(size_t(D)*Hq, 0.0f);
            res.lse.assign(size_t(Hq), 0.0f);
            res.o1.assign(size_t(D)*Hq, 0.0f);
            res.o2.assign(size_t(D)*Hq, 0.0f);
            res.lse1.assign(size_t(Hq), 0.0f);
            res.lse2.assign(size_t(Hq), 0.0f);
            ggml_backend_tensor_get(o_out, res.o.data(), 0, ggml_nbytes(o_out));
            ggml_backend_tensor_get(lse_out, res.lse.data(), 0, ggml_nbytes(lse_out));
            ggml_backend_tensor_get(fa1, res.o1.data(), 0, ggml_nbytes(fa1));
            ggml_backend_tensor_get(fa2, res.o2.data(), 0, ggml_nbytes(fa2));
            ggml_backend_tensor_get(lse1, res.lse1.data(), 0, ggml_nbytes(lse1));
            ggml_backend_tensor_get(lse2, res.lse2.data(), 0, ggml_nbytes(lse2));
            // Post-run storage bytes: the oracle must use exactly what the
            // overflow FA consumed (store included), and the fused vs legacy
            // runs must leave bit-identical companion bytes.
            std::vector<uint8_t> k_final(ggml_nbytes(k_storage), 0);
            std::vector<uint8_t> v_final(ggml_nbytes(v_storage), 0);
            ggml_backend_tensor_get(k_storage, k_final.data(), 0, k_final.size());
            ggml_backend_tensor_get(v_storage, v_final.data(), 0, v_final.size());
            res.k_store_bytes = k_final;
            res.v_store_bytes = v_final;
            const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q4_0);
            const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, int64_t(Dkv));
            // Remap [D*H_kv] rows into the oracle's [D, n_ovf, H_kv] layout.
            res.k_ovf_eff.assign(size_t(Dkv)*n_ovf, 0.0f);
            res.v_ovf_eff.assign(size_t(Dkv)*n_ovf, 0.0f);
            // The overflow FA consumes the F16 GATHER of the Q4 bytes, so the
            // oracle must round through F16 exactly like that destination type
            // (fp32 dequant would bias O at the 2e-4 level, far above the
            // fused-vs-oracle gate).
            std::vector<float> row(Dkv);
            for (int r = 0; r < n_ovf; ++r) {
                tr->to_float(k_final.data() + size_t(r)*row_bytes, row.data(), Dkv);
                for (int e = 0; e < int(Dkv); ++e) {
                    const int kh = e / D, d = e % D;
                    res.k_ovf_eff[(size_t(r) + size_t(kh)*n_ovf)*D + d] =
                        ggml_fp16_to_fp32(ggml_fp32_to_fp16(row[size_t(e)]));
                }
                tr->to_float(v_final.data() + size_t(r)*row_bytes, row.data(), Dkv);
                for (int e = 0; e < int(Dkv); ++e) {
                    const int kh = e / D, d = e % D;
                    res.v_ovf_eff[(size_t(r) + size_t(kh)*n_ovf)*D + d] =
                        ggml_fp16_to_fp32(ggml_fp32_to_fp16(row[size_t(e)]));
                }
            }
            res.splits = ggml_backend_sched_get_n_splits(sched);
        }
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return res;
    }

    // ---- fused leg --------------------------------------------------------
    ggml_tensor * o_local = ggml_reshape_2d(ctx, fa1, D*Hq, n_tok);
    const int n_kv_live = P + n_ovf;
    fused = ggml_kvarn_split_fused(ctx, q_t, k_cur_t, v_cur_t,
            o_local, lse1, k_view, v_view, m2_t, k_idxs, v_idxs,
            P, n_kv_live, 4, 4, n_ovf, scale, 0, int(bundle_id));
    o_out = fused;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[3] = { vk, cuda, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 3, 8192, false, true);
    ggml_backend_sched_set_tensor_backend(sched, q_t, cuda);
    ggml_backend_sched_set_tensor_backend(sched, k_cur_t, cuda);
    ggml_backend_sched_set_tensor_backend(sched, v_cur_t, cuda);
    ggml_backend_sched_set_tensor_backend(sched, k1_t, cuda);
    ggml_backend_sched_set_tensor_backend(sched, v1_t, cuda);
    ggml_backend_sched_set_tensor_backend(sched, m1_t, vk);
    // Local FA/LSE on the Radeon as above; the scheduler copies o_local/lse1
    // into the CUDA split for the fused op (its src[3]/src[4] must be CUDA).
    ggml_backend_sched_set_tensor_backend(sched, fa1, vk);
    ggml_backend_sched_set_tensor_backend(sched, lse1, vk);
    ggml_backend_sched_set_tensor_backend(sched, o_local, vk);
    // Companion storage and the raw row window stay on Vulkan; the executor
    // reaches them through the suppressed-copy allowance for this op.
    ggml_backend_sched_set_tensor_backend(sched, k_storage, vk);
    ggml_backend_sched_set_tensor_backend(sched, v_storage, vk);
    ggml_backend_sched_set_tensor_backend(sched, k_view, vk);
    ggml_backend_sched_set_tensor_backend(sched, v_view, vk);
    // Mask and index lists stay on the host, read directly by the executor.
    ggml_backend_sched_set_tensor_backend(sched, m2_t, cpu);
    ggml_backend_sched_set_tensor_backend(sched, k_idxs, cpu);
    ggml_backend_sched_set_tensor_backend(sched, v_idxs, cpu);
    ggml_backend_sched_set_tensor_backend(sched, fused, cuda);

    ggml_set_output(fused);
    dump_truncate();
    // The local operands of the merge stay alive so the test can check the
    // fused span's merged LSE and the executor's REAL internal O2/LSE2 (dumped
    // by the executor itself via LLAMA_OVF_FUSED_SPAN_DUMP).
    ggml_set_output(fa1);
    ggml_set_output(lse1);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, fused);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        res.declined = true;
        res.reason = "sched_alloc_graph failed (fused)";
        return res;
    }

    ggml_backend_tensor_set(q_t, q.data(), 0, ggml_nbytes(q_t));
    ggml_backend_tensor_set(k1_t, k1_bytes.data(), 0, ggml_nbytes(k1_t));
    ggml_backend_tensor_set(v1_t, v1_bytes.data(), 0, ggml_nbytes(v1_t));
    {
        std::vector<uint8_t> storage_k = k_storage_bytes;
        std::vector<uint8_t> storage_v = v_storage_bytes;
        ggml_backend_tensor_set(k_storage, storage_k.data(), 0, ggml_nbytes(k_storage));
        ggml_backend_tensor_set(v_storage, storage_v.data(), 0, ggml_nbytes(v_storage));
    }
    {
        std::vector<ggml_fp16_t> m1(size_t(n_local), ggml_fp32_to_fp16(0.0f));
        std::vector<ggml_fp16_t> m2(size_t(n_ovf), ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(m1_t, m1.data(), 0, ggml_nbytes(m1_t));
        ggml_backend_tensor_set(m2_t, m2.data(), 0, ggml_nbytes(m2_t));
    }
    ggml_backend_tensor_set(k_cur_t, k_cur.data(), 0, ggml_nbytes(k_cur_t));
    ggml_backend_tensor_set(v_cur_t, v_cur.data(), 0, ggml_nbytes(v_cur_t));
    {
        const int64_t row = store_row;
        ggml_backend_tensor_set(k_idxs, &row, 0, sizeof(row));
        ggml_backend_tensor_set(v_idxs, &row, 0, sizeof(row));
    }

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (ggml_graph_node(gf, i)->op == GGML_OP_FLASH_ATTN_EXT) {
            ++res.attn_nodes;
        }
    }
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        res.declined = true;
        res.reason = "sched_graph_compute failed (fused)";
    } else {
        res.o.assign(size_t(D)*Hq, 0.0f);
        res.o1.assign(size_t(D)*Hq, 0.0f);
        res.lse1.assign(size_t(Hq), 0.0f);
        ggml_backend_tensor_get(fused, res.o.data(), 0, ggml_nbytes(fused));
        ggml_backend_tensor_get(o_local, res.o1.data(), 0, ggml_nbytes(o_local));
        ggml_backend_tensor_get(lse1, res.lse1.data(), 0, ggml_nbytes(lse1));
        {
            // Companion bytes after the in-op store (memcmp vs the legacy leg).
            std::vector<uint8_t> k_final(ggml_nbytes(k_storage), 0);
            std::vector<uint8_t> v_final(ggml_nbytes(v_storage), 0);
            ggml_backend_tensor_get(k_storage, k_final.data(), 0, k_final.size());
            ggml_backend_tensor_get(v_storage, v_final.data(), 0, v_final.size());
            res.k_store_bytes = k_final;
            res.v_store_bytes = v_final;
        }
        {
            int32_t dump_il = -1;
            if (!dump_read_first(res.o2_exec, res.lse2_exec, dump_il) || dump_il != 0) {
                res.declined = true;
                res.reason = "executor leg dump missing (LLAMA_OVF_FUSED_SPAN_DUMP)";
            }
        }
        res.splits = ggml_backend_sched_get_n_splits(sched);
        if (std::getenv("KVARN_SPLIT_FUSED_DEBUG") != nullptr) {
            std::vector<uint8_t> bytes(ggml_nbytes(k_storage), 0);
            ggml_backend_tensor_get(k_storage, bytes.data(), 0, bytes.size());
            const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q4_0);
            const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, int64_t(Dkv));
            std::vector<float> r0(Dkv), r63(Dkv);
            tr->to_float(bytes.data(), r0.data(), Dkv);
            tr->to_float(bytes.data() + size_t(store_row)*row_bytes, r63.data(), Dkv);
            std::printf("    [dbg-store] row0[0..3]=%.5f %.5f %.5f %.5f | row63[0..3]=%.5f %.5f %.5f %.5f | kcur[0..3]=%.5f %.5f %.5f %.5f\n",
                    r0[0], r0[1], r0[2], r0[3], r63[0], r63[1], r63[2], r63[3],
                    k_cur[0], k_cur[1], k_cur[2], k_cur[3]);
        }
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return res;
}

// -------------------------------------------------------------------- main ---

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    set_env_fused_span();
    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_cpu_init();
    // Explicit device selection: Vulkan device 0 for the overflow leg and the
    // first CUDA-named GPU for the local/merge leg (a generic
    // init_by_type(GPU) can return the Vulkan iGPU instead).
    ggml_backend_t vk  = ggml_backend_vk_init(0);
    ggml_backend_t cuda = nullptr;
    for (int i = 0; i < (int) ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                std::strstr(ggml_backend_dev_name(dev), "CUDA") != nullptr) {
            cuda = ggml_backend_dev_init(dev, nullptr);
            break;
        }
    }
    if (cpu) {
        // The oracle gates are numerical; keep the CPU FA reference-accurate.
        ggml_backend_cpu_set_use_ref(cpu, true);
    }
    if (!cpu || !vk || !cuda) {
        std::printf("  FAIL: required backends missing (cpu=%d vulkan=%d cuda=%d)\n",
                    cpu != nullptr, vk != nullptr, cuda != nullptr);
        return 1;
    }

    // One executor bundle for this test context (the builder does the same at
    // graph build); the id is baked into every op below.
#ifdef _WIN32
    _putenv_s("LLAMA_OVF_FUSED_SPAN_DUMP", dump_path());
#else
    setenv("LLAMA_OVF_FUSED_SPAN_DUMP", dump_path(), 1);
#endif
    uint32_t bundle_id = llama_kvarn_split_fused_exec_register(cuda, vk);
    if (bundle_id == 0) {
        std::printf("  FAIL: executor registration failed\n");
        return 1;
    }
    CHECK(llama_kvarn_split_fused_exec_ready(bundle_id));

    // Per-context bundle isolation: a second bundle can be registered and
    // released without affecting the first one. The CUDA callback slot is
    // never cleared (immortal dispatcher), so draining the ENTIRE registry and
    // registering again must keep a fully functional executor; the executions
    // below use the freshly registered id.
    {
        const uint32_t idA = llama_kvarn_split_fused_exec_register(cuda, vk);
        const uint32_t idB = llama_kvarn_split_fused_exec_register(cuda, vk);
        CHECK(idA != 0 && idB != 0 && idA != idB && idA != bundle_id && idB != bundle_id);
        CHECK(llama_kvarn_split_fused_exec_ready(idA) && llama_kvarn_split_fused_exec_ready(idB));
        llama_kvarn_split_fused_exec_unregister(idA);
        CHECK(!llama_kvarn_split_fused_exec_ready(idA));
        CHECK(llama_kvarn_split_fused_exec_ready(idB));
        CHECK(llama_kvarn_split_fused_exec_ready(bundle_id));
        llama_kvarn_split_fused_exec_unregister(idB);
        CHECK(!llama_kvarn_split_fused_exec_ready(idB));
        CHECK(llama_kvarn_split_fused_exec_ready(bundle_id));

        llama_kvarn_split_fused_exec_unregister(bundle_id);
        CHECK(!llama_kvarn_split_fused_exec_ready(bundle_id));
        const uint32_t idRe = llama_kvarn_split_fused_exec_register(cuda, vk);
        CHECK(idRe != 0 && idRe != idA && idRe != idB);
        CHECK(llama_kvarn_split_fused_exec_ready(idRe));
        bundle_id = idRe;
    }

    struct shape { int D, Hq, Hkv; };
    const shape shapes[] = {
        { 128,  6, 2 },
        { 256, 24, 4 },
    };

    int executed = 0;
    int shape_idx = 0;
    for (const shape & sh : shapes) {
        ++shape_idx;
        const int D = sh.D, Hq = sh.Hq, Hkv = sh.Hkv;
        const int n_local = 64;
        const int n_ovf   = 64;
        const int P       = 256;
        const int store_row = n_ovf - 1;

        char tag[128];
        std::snprintf(tag, sizeof(tag), "fused D%d GQA%d/%d", D, Hq, Hkv);
        g_case = tag;

        // Deterministic inputs.
        std::vector<float> q(size_t(D)*Hq);
        std::vector<float> k1(size_t(D)*n_local*Hkv), v1(size_t(D)*n_local*Hkv);
        std::vector<float> k2(size_t(D)*n_ovf*Hkv),   v2(size_t(D)*n_ovf*Hkv);
        std::vector<float> k_cur(size_t(D)*Hkv),      v_cur(size_t(D)*Hkv);
        uint32_t s = 0x12345678u ^ uint32_t(D);
        const auto next_f = [&]() {
            s = s * 1664525u + 1013904223u;
            return float(int32_t(s >> 8)) / float(1 << 23) - 1.0f;
        };
        for (auto & x : q) { x = next_f(); }
        for (auto & x : k1) { x = next_f(); }
        for (auto & x : v1) { x = next_f(); }
        for (auto & x : k2) { x = next_f(); }
        for (auto & x : v2) { x = next_f(); }
        for (auto & x : k_cur) { x = next_f(); }
        for (auto & x : v_cur) { x = next_f(); }

        std::vector<uint8_t> k1b, v1b, k2b, v2b;
        std::vector<float>   k1d, v1d, k2d, v2d;
        pack_range(GGML_TYPE_F16, D, k1, k1b, k1d);
        pack_range(GGML_TYPE_F16, D, v1, v1b, v1d);
        pack_range(GGML_TYPE_Q4_0, D, k2, k2b, k2d);
        pack_range(GGML_TYPE_Q4_0, D, v2, v2b, v2d);

        leg_result ref = run_leg(cuda, vk, 0, bundle_id, D, Hq, Hkv, n_local, n_ovf, P,
                q, k1b, v1b, k2b, v2b, k_cur, v_cur, store_row);
        leg_result fus = run_leg(cuda, vk, 1, bundle_id, D, Hq, Hkv, n_local, n_ovf, P,
                q, k1b, v1b, k2b, v2b, k_cur, v_cur, store_row);

        CHECK(!ref.declined);
        CHECK(!fus.declined);
        if (ref.declined || fus.declined) {
            std::printf("  [%s] declined: legacy='%s' fused='%s'\n",
                    tag, ref.reason.c_str(), fus.reason.c_str());
            continue;
        }
        ++executed;

        // Gate 1: fused O vs legacy O.
        double max_o = 0.0;
        double se = 0.0, sr = 0.0;
        for (size_t i = 0; i < fus.o.size(); ++i) {
            const double diff = double(fus.o[i]) - double(ref.o[i]);
            max_o = std::max(max_o, std::fabs(diff));
            se += diff * diff;
            sr += double(ref.o[i]) * double(ref.o[i]);
        }
        const double nrmse_o = std::sqrt(se / double(ref.o.size())) /
                std::max(std::sqrt(sr / double(ref.o.size())), 1e-6);
        const bool bits_equal = std::memcmp(fus.o.data(), ref.o.data(), fus.o.size()*sizeof(float)) == 0;

        // Gate 2: legacy merged LSE vs the FP64 oracle over the effective bytes.
        // The oracle consumes the storage bytes the device actually holds after
        // the in-graph store (read back below), not the host-quantized row, so
        // host-vs-device quantizer rounding cannot bias the gate.
        const oracle_out oracle = fp64_oracle(D, Hq, Hkv, q, k1d, v1d, n_local,
                ref.k_ovf_eff, ref.v_ovf_eff, n_ovf);
        if (std::getenv("KVARN_SPLIT_FUSED_DEBUG") != nullptr) {
            // FP64 LSE over the overflow range alone (same visibility), to tell
            // which executor call consumed the correct stored row.
            const int gqa_dbg = Hq / Hkv;
            const double scale_dbg = 1.0 / std::sqrt(double(D));
            for (int h = 0; h < std::min(Hq, 4); ++h) {
                const int kh = h / gqa_dbg;
                double mx_dbg = -INFINITY;
                std::vector<double> sc;
                sc.reserve(size_t(n_ovf));
                for (int s2 = 0; s2 < n_ovf; ++s2) {
                    double acc = 0.0;
                    for (int d = 0; d < D; ++d) {
                        acc += double(q[size_t(h)*D + d]) *
                            double(ref.k_ovf_eff[(size_t(s2) + size_t(kh)*n_ovf)*D + d]);
                    }
                    sc.push_back(acc * scale_dbg);
                    mx_dbg = std::max(mx_dbg, sc.back());
                }
                double den_dbg = 0.0;
                for (double v : sc) den_dbg += std::exp(v - mx_dbg);
                std::printf("    [dbg-ovf] h=%d fp64_lse2=%.6f\n", h, mx_dbg + std::log(den_dbg));
            }
        }
        double max_lse = 0.0;
        for (int h = 0; h < Hq; ++h) {
            max_lse = std::max(max_lse, std::fabs(double(ref.lse[size_t(h)]) - oracle.lse[size_t(h)]));
        }
        if (std::getenv("KVARN_SPLIT_FUSED_DEBUG") != nullptr) {
            // Per-range backend-vs-FP64 deviation, to attribute the residual
            // fused-vs-oracle O error to the FA precision floor.
            const int gqa_dbg = Hq / Hkv;
            const double scale_dbg = 1.0 / std::sqrt(double(D));
            double dev1 = 0.0, dev2 = 0.0;
            for (int h = 0; h < Hq; ++h) {
                const int kh = h / gqa_dbg;
                for (int range = 0; range < 2; ++range) {
                    const int n_kv = range == 0 ? n_local : n_ovf;
                    const std::vector<float> & kk = range == 0 ? k1d : ref.k_ovf_eff;
                    const std::vector<float> & vv = range == 0 ? v1d : ref.v_ovf_eff;
                    std::vector<double> sc;
                    sc.reserve(size_t(n_kv));
                    double mx = -INFINITY;
                    for (int s2 = 0; s2 < n_kv; ++s2) {
                        double acc = 0.0;
                        for (int d = 0; d < D; ++d) {
                            acc += double(q[size_t(h)*D + d]) *
                                double(kk[(size_t(s2) + size_t(kh)*n_kv)*D + d]);
                        }
                        sc.push_back(acc * scale_dbg);
                        mx = std::max(mx, sc.back());
                    }
                    double den = 0.0;
                    for (double v : sc) { den += std::exp(v - mx); }
                    const std::vector<float> & oo = range == 0 ? ref.o1 : ref.o2;
                    double dev = 0.0;
                    for (int d = 0; d < D; ++d) {
                        double accv = 0.0;
                        for (int s2 = 0; s2 < n_kv; ++s2) {
                            accv += std::exp(sc[size_t(s2)] - mx) *
                                double(vv[(size_t(s2) + size_t(kh)*n_kv)*D + d]);
                        }
                        dev = std::max(dev, std::fabs(accv / den - double(oo[size_t(h)*D + d])));
                    }
                    (range == 0 ? dev1 : dev2) = std::max(range == 0 ? dev1 : dev2, dev);
                }
            }
            std::printf("    [dbg-dev] fa1-vs-fp64=%.3e fa2-vs-fp64=%.3e\n", dev1, dev2);
        }

        // Fused O vs the FP64 oracle (the legacy-vs-oracle gate would hide a
        // chain that is wrong on both legs in the same way).
        double max_o_oracle = 0.0;
        double se_or = 0.0, sr_or = 0.0;
        for (size_t i = 0; i < fus.o.size(); ++i) {
            const double diff = double(fus.o[i]) - oracle.o[i];
            max_o_oracle = std::max(max_o_oracle, std::fabs(diff));
            se_or += diff * diff;
            sr_or += oracle.o[i] * oracle.o[i];
        }
        const double nrmse_or = std::sqrt(se_or / double(fus.o.size())) /
                std::max(std::sqrt(sr_or / double(fus.o.size())), 1e-6);

        // The executor's REAL internal Vulkan leg (dumped by the executor)
        // must match the legacy leg's FA2/LSE2 (same op, same backend, same
        // operands) ...
        const bool leg_shapes_ok = fus.o2_exec.size() == ref.o2.size() &&
                fus.lse2_exec.size() == ref.lse2.size() && !fus.o2_exec.empty();
        double max_o2_leg = leg_shapes_ok ? 0.0 : INFINITY;
        double max_lse2_leg = leg_shapes_ok ? 0.0 : INFINITY;
        for (size_t i = 0; leg_shapes_ok && i < fus.o2_exec.size(); ++i) {
            max_o2_leg = std::max(max_o2_leg, std::fabs(double(fus.o2_exec[i]) - double(ref.o2[i])));
        }
        for (size_t i = 0; leg_shapes_ok && i < fus.lse2_exec.size(); ++i) {
            max_lse2_leg = std::max(max_lse2_leg, std::fabs(double(fus.lse2_exec[i]) - double(ref.lse2[i])));
        }

        // ... and the fused merged LSE replays the merge formula in FP64 over
        // the fused leg's local operands and the executor's REAL LSE2, gated
        // against the oracle.
        std::vector<double> fus_o_rec, fus_lse_rec;
        merge_fp64_pair(fus.o1, fus.lse1, fus.o2_exec, fus.lse2_exec, D, Hq, fus_o_rec, fus_lse_rec);
        double max_lse_rec = 0.0;
        double max_o_rec  = 0.0;
        for (int h = 0; h < Hq; ++h) {
            max_lse_rec = std::max(max_lse_rec,
                    std::fabs(fus_lse_rec[size_t(h)] - oracle.lse[size_t(h)]));
        }
        for (size_t i = 0; i < fus_o_rec.size(); ++i) {
            max_o_rec = std::max(max_o_rec, std::fabs(fus_o_rec[i] - oracle.o[i]));
        }

        // Companion storage must be byte-identical between the fused and the
        // legacy store (same SET_ROWS op, same backend, same operands).
        const bool store_bits_equal =
            fus.k_store_bytes.size() == ref.k_store_bytes.size() &&
            fus.v_store_bytes.size() == ref.v_store_bytes.size() &&
            !fus.k_store_bytes.empty() &&
            std::memcmp(fus.k_store_bytes.data(), ref.k_store_bytes.data(), fus.k_store_bytes.size()) == 0 &&
            std::memcmp(fus.v_store_bytes.data(), ref.v_store_bytes.data(), fus.v_store_bytes.size()) == 0;

        // The fused graph must not carry a second FLASH_ATTN_EXT node (the
        // overflow FA lives inside the executor) and must not add splits.
        CHECK(fus.attn_nodes == 1);
        CHECK(ref.attn_nodes == 2);
        CHECK(fus.splits <= ref.splits);

        // The fused-vs-legacy comparison is the strict one (<=1e-6): it proves
        // the span adds no error over the chain it replaces. The FP64-oracle
        // comparisons use the plan's achieved backend floor: each Vulkan FA
        // (local and overflow) deviates from FP64 by ~3e-4 maxabs on its own
        // (F16 PV accumulation; measured, printed above under
        // KVARN_SPLIT_FUSED_DEBUG), so a 1e-6-vs-oracle gate is below the
        // backend precision for ANY chain, legacy included. The merged LSE is
        // precision-insensitive and keeps the strict 1e-4 gate.
        CHECK(max_o <= 1e-6);
        CHECK(nrmse_o <= 1e-6);
        CHECK(max_o_oracle <= 1e-2);
        CHECK(nrmse_or <= 1e-3);
        CHECK(max_lse <= 1e-4);
        CHECK(max_lse_rec <= 1e-4);
        CHECK(max_o_rec <= 1e-2);
        CHECK(leg_shapes_ok);
        CHECK(max_o2_leg <= 1e-6);
        CHECK(max_lse2_leg <= 1e-6);
        CHECK(store_bits_equal);

        std::printf("  [%s] O fused-vs-legacy maxabs=%.3e nrmse=%.3e | fused-vs-oracle maxabs=%.3e nrmse=%.3e | LSE legacy=%.3e fused-rec=%.3e | exec-leg O2=%.3e LSE2=%.3e | store memcmp=%d | bitwise-O=%d | splits %d vs %d, attn %d vs %d\n",
                tag, max_o, nrmse_o, max_o_oracle, nrmse_or, max_lse, max_lse_rec,
                max_o2_leg, max_lse2_leg,
                int(store_bits_equal), int(bits_equal),
                fus.splits, ref.splits, fus.attn_nodes, ref.attn_nodes);

        // Stable-extent growth steps: n_ovf crossing several 256-row steps
        // changes the geometry, so the executor must rebuild its sub-graphs in
        // place and stay correct (including the store and the merged LSE).
        if (shape_idx == 1) {
            const int extents[] = { 320, 576, 64 };
            for (int e : extents) {
                std::vector<float> k2e(size_t(D)*e*Hkv), v2e(size_t(D)*e*Hkv);
                for (auto & x : k2e) { x = next_f(); }
                for (auto & x : v2e) { x = next_f(); }
                std::vector<uint8_t> k2eb, v2eb;
                std::vector<float>   k2ed, v2ed;
                pack_range(GGML_TYPE_Q4_0, D, k2e, k2eb, k2ed);
                pack_range(GGML_TYPE_Q4_0, D, v2e, v2eb, v2ed);
                const int sr = e - 1;
                const leg_result rE = run_leg(cuda, vk, 0, bundle_id, D, Hq, Hkv, n_local, e, P,
                        q, k1b, v1b, k2eb, v2eb, k_cur, v_cur, sr);
                const leg_result fE = run_leg(cuda, vk, 1, bundle_id, D, Hq, Hkv, n_local, e, P,
                        q, k1b, v1b, k2eb, v2eb, k_cur, v_cur, sr);
                char etag[64];
                std::snprintf(etag, sizeof(etag), "extent D%d n_ovf=%d", D, e);
                g_case = etag;
                CHECK(!rE.declined);
                CHECK(!fE.declined);
                if (rE.declined || fE.declined) {
                    std::printf("  [%s] declined: legacy='%s' fused='%s'\n",
                            etag, rE.reason.c_str(), fE.reason.c_str());
                    continue;
                }
                const oracle_out oracleE = fp64_oracle(D, Hq, Hkv, q, k1d, v1d, n_local,
                        rE.k_ovf_eff, rE.v_ovf_eff, e);
                double mo = 0.0, mor = 0.0, ml = 0.0;
                for (size_t i = 0; i < fE.o.size(); ++i) {
                    mo  = std::max(mo,  std::fabs(double(fE.o[i]) - double(rE.o[i])));
                    mor = std::max(mor, std::fabs(double(fE.o[i]) - oracleE.o[i]));
                }
                for (int h = 0; h < Hq; ++h) {
                    ml = std::max(ml, std::fabs(double(rE.lse[size_t(h)]) - oracleE.lse[size_t(h)]));
                }
                std::vector<double> ro, rl;
                merge_fp64_pair(fE.o1, fE.lse1, fE.o2_exec, fE.lse2_exec, D, Hq, ro, rl);
                double mlf = 0.0;
                for (int h = 0; h < Hq; ++h) {
                    mlf = std::max(mlf, std::fabs(rl[size_t(h)] - oracleE.lse[size_t(h)]));
                }
                const bool sbe = fE.k_store_bytes.size() == rE.k_store_bytes.size() &&
                    fE.v_store_bytes.size() == rE.v_store_bytes.size() && !fE.k_store_bytes.empty() &&
                    std::memcmp(fE.k_store_bytes.data(), rE.k_store_bytes.data(), fE.k_store_bytes.size()) == 0 &&
                    std::memcmp(fE.v_store_bytes.data(), rE.v_store_bytes.data(), fE.v_store_bytes.size()) == 0;
                CHECK(mo <= 1e-6);
                CHECK(mor <= 1e-2);
                CHECK(ml <= 1e-4);
                CHECK(mlf <= 1e-4);
                CHECK(sbe);
                std::printf("  [%s] O fused-vs-legacy=%.3e fused-vs-oracle=%.3e | LSE legacy=%.3e fused-rec=%.3e | store memcmp=%d\n",
                        etag, mo, mor, ml, mlf, int(sbe));
            }
            g_case = tag;
        }

        // Rebuild stability (regression for the pointer-keyed cache): the same
        // span rebuilt with FRESH graph/tensor objects must reconcile the one
        // persistent executor per layer instead of accumulating duplicates.
        // 70 rebuilds exceed the old 64-entry cap that aborted on the 27B.
        if (shape_idx == 1) {
            const int n_rebuilds = 70;
            bool rebuild_ok = true;
            double rebuild_max = 0.0;
            for (int r = 0; r < n_rebuilds; ++r) {
                leg_result re = run_leg(cuda, vk, 1, bundle_id, D, Hq, Hkv, n_local, n_ovf, P,
                        q, k1b, v1b, k2b, v2b, k_cur, v_cur, store_row);
                if (re.declined) {
                    rebuild_ok = false;
                    std::printf("  [dbg %s] rebuild %d declined: %s\n", tag, r, re.reason.c_str());
                    break;
                }
                double iter_max = 0.0, iter_first = 0.0;
                size_t first_idx = 0;
                for (size_t i = 0; i < re.o.size(); ++i) {
                    iter_max = std::max(iter_max, std::fabs(double(re.o[i]) - double(ref.o[i])));
                    const double d0 = std::fabs(double(re.o[i]) - double(fus.o[i]));
                    if (d0 > iter_first) { iter_first = d0; first_idx = i; }
                }
                rebuild_max = std::max(rebuild_max, iter_max);
                if (std::getenv("KVARN_SPLIT_FUSED_DEBUG") != nullptr && r < 3) {
                    std::printf("    [dbg %s] rebuild %d: vs legacy=%.3e vs first=%.3e (idx=%zu h=%zu)\n",
                            tag, r, iter_max, iter_first, first_idx, first_idx / size_t(D));
                }
            }
            CHECK(rebuild_ok);
            CHECK(rebuild_max <= 1e-6);
            std::printf("  [%s] rebuild stability: %d fresh graphs, maxabs vs legacy=%.3e\n",
                    tag, n_rebuilds, rebuild_max);
        }
    }

    // Release the test bundle; other bundles are already gone, so this also
    // clears the CUDA callback slot (exercises the drain path).
    llama_kvarn_split_fused_exec_unregister(bundle_id);
    CHECK(!llama_kvarn_split_fused_exec_ready(bundle_id));
    std::remove(dump_path());

    ggml_backend_free(cpu);
    ggml_backend_free(vk);
    ggml_backend_free(cuda);

    if (executed == 0) {
        std::printf("FAIL: no shape executed\n");
        return 1;
    }
    std::printf("test-kvarn-split-fused: %d checks, %d failures (shapes executed: %d)\n",
            g_checks, g_failures, executed);
    if (g_failures != 0) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
