// F4 real-pair O/LSE oracle.
//
// Contract (from ggml.c):
//   q  -> [D, n_q,  n_qh, n_batch]
//   FA -> [D, n_qh, n_q,  n_batch]   => flat index ((t*n_qh + h)*D + d)
//   lse-> [n_qh, n_q, n_batch]        => flat index  h + t*n_qh
//   mask-> [n_kv, n_q, 1, 1]          => flat index  p + t*n_kv
//
// K/V row order seen by the oracle is identical for both codecs:
//   element (d, p, kh)  ->  k[(p + n_kv*kh)*D + d]
// because the native KVarN view is permuted (0,2,1,3) into [D, n_kv, n_kvh]
// and the Q4_0 overflow tensor is built directly as [D, n_kv, n_kvh].
//
// The KVarN oracle rows are materialized from the EXACT records/stage bytes the
// CUDA store wrote, read back from the device -- never from a CPU-side store.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-vulkan.h"
#include "llama.h"
#include "llama-kvarn.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <map>
#include <set>
#include <vector>

// ---------------------------------------------------------------------------
// tiny check harness
// ---------------------------------------------------------------------------
static int g_checks = 0, g_failures = 0, g_skipped = 0;
static std::string g_case;
// A/A2: the last measured merged O/LSE, so main() can re-run the same case and
// require bit-identical results before interpreting anything else about it.
static std::vector<float> g_last_o, g_last_lse;
// Alternates A/A2 so the two runs poison CONTRACTUALLY UNUSED storage bytes
// with DIFFERENT patterns: if a valid logical row moves with the poison, the
// output depends on bytes the store never wrote.
static int g_run_index = 0;
// Sentinel actually written into the stage leaf for the current case, and
// whether any sentinel was written at all; published so the post-store
// initialization assertion can see them.
static uint8_t g_case_poison = 0;
// The exact index descriptor uploaded into idx1 for the current case. The
// initialization/read-set assertions derive from THIS, not from a re-derived
// idealisation of it.
static std::vector<int64_t> g_case_indices;

// msg57 F5 closure: the PRODUCTION prefill route runs the local KVarN FA in
// KVARN_DOMAIN_ROTATED_K_ORIGINAL_V (K rotated, V original, Q rotated, O in
// the original domain). Default OFF keeps every existing case byte-identical.
//   F4_RP_DOMAIN_ORIGINAL_V=1 -> build that domain (CUDA route) with an oracle
//     whose body V is UN-rotated (inverse of the store's WHT) and whose tail V
//     is the stored original one.
//   F4_RP_DORO3_CPU_TOO=1    -> also run those cases with the local FA on the
//     CPU (reference route). The CPU KVarN FA is domain-agnostic and emits a
//     ROTATED O, so that route keeps the domain-1 tail V and the domain-1
//     oracle; the report says which oracle was used.
static bool g_domain_original_v = false;
static bool g_force_cpu_fa1     = false;
static bool g_emulate_p_f16     = false;   // F4_RP_EMULATE_P_F16=1
// Whether the current case pre-filled the leaves with the byte-level sentinel
// pair (K pattern, V = complement) rather than with the production zero contract.
static bool g_case_init_sentinel = false;
// Byte-level store-coverage hits, per A/A2 parity. A half that still holds the
// pre-store sentinel can be either an unwritten byte OR a value collision (a
// legitimately stored fp16 that happens to equal the pattern). The two runs use
// DIFFERENT patterns, so only a genuinely unwritten half can appear in BOTH
// parities: the intersection is collision-free, which a single-run threshold
// cannot give. (Requires the default alternating patterns, not
// F4_RP_POISON_FIXED, which would defeat the intersection.)
static std::vector<uint64_t> g_sent_hits[2];
static std::string          g_sent_tag[2];
static bool                 g_sent_seen[2] = {false, false};
static bool   g_case_no_poison = true;
// Per-intermediate snapshots so A/A2 can name WHICH tensor first differs
// (body O/LSE -> local merged O/LSE -> final merged O/LSE) instead of only
// reporting that the last one did.
static std::vector<float> g_last_local, g_last_local_lse;
static std::vector<float> g_last_body,  g_last_body_lse;
static std::vector<int32_t> g_last_desc;
static std::string g_first_diff_level;

static void check(bool cond, const char * what, int line) {
    ++g_checks;
    if (!cond) { ++g_failures; std::printf("  FAIL [%s] line %d: %s\n", g_case.c_str(), line, what); }
}
#define CHECK(x) check((x), #x, __LINE__)

static const double kGateNrmseO    = 1e-3;
static const double kGateMaxAbsO   = 1e-2;
static const double kGateMaxAbsLse = 1e-2;


// --------------------------------------------------------------- backends ---
struct env_t { ggml_backend_t cuda = nullptr, vk = nullptr, cpu = nullptr; };

static bool env_init(env_t & e) {
    ggml_backend_dev_t cd = nullptr, vd = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d  = ggml_backend_dev_get(i);
        const char *       nm = ggml_backend_dev_name(d);
        std::printf("  device[%zu]: name=%s desc=%s\n", i, nm, ggml_backend_dev_description(d));
        if (std::strncmp(nm, "CUDA",   4) == 0 && !cd) { cd = d; }
        if (std::strncmp(nm, "Vulkan", 6) == 0 && !vd) { vd = d; }
    }
    if (!cd || !vd) { return false; }
    e.cuda = ggml_backend_dev_init(cd, nullptr);
    e.vk   = ggml_backend_dev_init(vd, nullptr);
    e.cpu  = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!e.cuda || !e.vk || !e.cpu) { return false; }
    std::printf("  chosen: local=%s overflow=%s\n",
                ggml_backend_name(e.cuda), ggml_backend_name(e.vk));
    return true;
}

// ---------------------------------------------------------------- oracle ----
struct ref_o { std::vector<double> o, lse; };

static bool all_finite(const std::vector<float> & v) {
    for (float x : v) { if (!std::isfinite(x)) { return false; } }
    return true;
}

// FP64 attention over exactly the rows the backend read.
// mask is indexed [p + n_kv*t]; -INFINITY means masked.
static ref_o oracle(int D, int n_q, int n_qh, int n_kvh,
                    const std::vector<float> & q,
                    const std::vector<float> & k, const std::vector<float> & v,
                    int n_kv, const std::vector<float> & mask, double scale) {
    ref_o r;
    const int gqa = n_qh / n_kvh;
    r.o.assign(size_t(D) * n_qh * n_q, 0.0);
    r.lse.assign(size_t(n_qh) * n_q, 0.0);
    for (int h = 0; h < n_qh; ++h) {
        const int kh = h / gqa;
        for (int t = 0; t < n_q; ++t) {
            std::vector<double> s(size_t(n_kv), 0.0);
            double m = -INFINITY;
            for (int p = 0; p < n_kv; ++p) {
                if (std::isinf(mask[size_t(p) + size_t(n_kv) * t])) { s[size_t(p)] = 0.0; continue; }
                double acc = 0.0;
                for (int d = 0; d < D; ++d) {
                    acc += double(q[(size_t(t) + size_t(n_q) * h) * D + d]) *
                           double(k[(size_t(p) + size_t(n_kv) * kh) * D + d]);
                }
                // FA applies the scale BEFORE the softmax, so the max must be
                // taken over the scaled score, not the raw accumulator.
                s[size_t(p)] = acc * scale;
                if (s[size_t(p)] > m) { m = s[size_t(p)]; }
            }
            const size_t li = size_t(h) + size_t(n_qh) * t;
            const size_t oi = (size_t(t) * n_qh + size_t(h)) * D;
            if (m == -INFINITY) { r.lse[li] = -INFINITY; continue; }  // O stays exactly 0
            double den = 0.0;
            for (int p = 0; p < n_kv; ++p) {
                if (std::isinf(mask[size_t(p) + size_t(n_kv) * t])) { continue; }
                s[size_t(p)] = std::exp(s[size_t(p)] - m);
                den += s[size_t(p)];
            }
            // F4_RP_EMULATE_P_F16=1: the F16 MMA path stores the softmax weights
            // in a half2 matrix, i.e. every weight is rounded to F16 before the
            // V accumulation. This variant reproduces exactly that rounding.
            if (g_emulate_p_f16) {
                for (int p = 0; p < n_kv; ++p) {
                    if (std::isinf(mask[size_t(p) + size_t(n_kv) * t])) { continue; }
                    s[size_t(p)] = ggml_fp16_to_fp32(ggml_fp32_to_fp16((float) s[size_t(p)]));
                }
            }
            for (int d = 0; d < D; ++d) {
                double acc = 0.0;
                for (int p = 0; p < n_kv; ++p) {
                    if (s[size_t(p)] == 0.0) { continue; }
                    acc += s[size_t(p)] * double(v[(size_t(p) + size_t(n_kv) * kh) * D + d]);
                }
                r.o[oi + d] = acc / den;
            }
            r.lse[li] = m + std::log(den);
        }
    }
    return r;
}

static double nrmse_of(const std::vector<float> & got, const std::vector<double> & ref) {
    if (got.size() != ref.size() || ref.empty()) { return INFINITY; }
    double se = 0.0, sr = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) { return INFINITY; }
        const double d = double(got[i]) - ref[i];
        se += d * d;
        sr += ref[i] * ref[i];
    }
    return std::sqrt(se / double(ref.size())) /
           std::max(std::sqrt(sr / double(ref.size())), 1e-6);
}

static double maxdiff(const std::vector<float> & got, const std::vector<double> & ref) {
    if (got.size() != ref.size() || ref.empty()) { return INFINITY; }
    double m = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = double(got[i]), b = ref[i];
        if (std::isinf(a) || std::isinf(b)) { return (std::isnan(a) || std::isnan(b)) ? INFINITY : INFINITY; }
        m = std::max(m, std::fabs(a - b));
    }
    return m;
}

static size_t first_diff(const std::vector<float> & got, const std::vector<double> & ref) {
    if (got.size() != ref.size()) { return 0; }
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i]) || std::fabs(double(got[i]) - ref[i]) > kGateMaxAbsO) { return i; }
    }
    return ref.size();
}

// Shared acceptance verifier: every measured tensor goes through this, so the
// negative controls exercise the same code path the real gate uses.
static bool verify_o(const std::vector<float> & got, const std::vector<double> & ref,
                     double & nr, double & md) {
    nr = nrmse_of(got, ref);
    md = maxdiff(got, ref);
    return got.size() == ref.size() && !ref.empty() && all_finite(got) &&
           nr <= kGateNrmseO && md <= kGateMaxAbsO;
}

static bool verify_lse(const std::vector<float> & got, const std::vector<double> & ref,
                       double & md) {
    if (got.size() != ref.size() || ref.empty()) { md = INFINITY; return false; }
    md = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = double(got[i]), b = ref[i];
        // NaN is never acceptable on either side
        if (std::isnan(a) || std::isnan(b)) { return false; }
        // an all-masked row must be -inf on both sides; a matching inf pair is
        // NOT a reason to stop -- the remaining elements must still be checked
        if (std::isinf(b)) {
            if (!std::isinf(a) || (a < 0.0) != (b < 0.0)) { return false; }
            continue;
        }
        if (std::isinf(a)) { return false; }
        md = std::max(md, std::fabs(a - b));
    }
    return md <= kGateMaxAbsLse;
}

// Per-range fully-masked contract: O == 0 exactly and LSE == -inf exactly.
static bool masked_rows_ok(int D, int n_q, int n_qh, const ref_o & r,
                           const std::vector<float> & ho, const std::vector<float> & hl) {
    if (ho.size() != r.o.size() || hl.size() != r.lse.size()) { return false; }
    for (int t = 0; t < n_q; ++t) {
        for (int h = 0; h < n_qh; ++h) {
            const size_t li = size_t(h) + size_t(n_qh) * t;
            const size_t oi = (size_t(t) * n_qh + size_t(h)) * D;
            if (r.lse[li] != -INFINITY) { continue; }
            for (int d = 0; d < D; ++d) { if (ho[oi + d] != 0.0f) { return false; } }
            if (!(std::isinf(hl[li]) && hl[li] < 0.0f)) { return false; }
        }
    }
    return true;
}

// --------------------------------------------------- codec dequantization ---
// KVarN rows materialized from the EXACT records/stage the CUDA store wrote.
static bool kvarn_rows_from_device(const env_t & env, const std::vector<uint8_t> & rec_h,
                                   const std::vector<ggml_fp16_t> & stg_h,
                                   int rec_bytes, int group_rows, int n_kv,
                                   int bits, bool is_value, int stage_groups,
                                   int head_slices, bool emit_rotated,
                                   std::vector<float> & out) {
    ggml_init_params dp = { ggml_tensor_overhead() * 512 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(dp);

    // records are [record_bytes, phys_heads, GROUPS] and one group holds
    // KVAR_N_GROUP cells: a hard-coded single slot only ever held kv1 <= 128,
    // and the P=512 cases wrote groups 1..3 past the end of it.
    const int rec_slots = std::max(1, (n_kv + int(KVAR_N_GROUP) - 1) / int(KVAR_N_GROUP));
    ggml_tensor * rec = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, rec_bytes, group_rows, rec_slots);
    ggml_tensor * stg = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, KVAR_N_GROUP, group_rows, 128 * stage_groups);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_kv);
    // the materialize op inherits head_slices / tail_groups from the stage op
    stg->op_params[5] = head_slices;
    stg->op_params[8] = stage_groups - 1;
    ggml_tensor * mat = ggml_kvarn_materialize(ctx, rec, stg, idx, n_kv, 0, 1, bits, is_value, stage_groups);
    mat->op_params[5] = head_slices;
    // op_params[4] == 1 keeps the rotated domain (llama-kv-cache-kvarn.cpp
    // materialize(): "stays in the same rotated domain as the persistent body").
    // K is consumed rotated by ROTATED FA; V is un-rotated to produce O.
    mat->op_params[4] = emit_rotated ? 1 : 0;
    // Mirror production llama_kv_cache_kvarn::materialize(), which sets
    // op_params[6]=swa, [8]=tail_groups, [9]=eager_records=1 and [10]=read_indirect.
    // ggml_kvarn_materialize() leaves [9] at its default, so the oracle decoded
    // with a DIFFERENT stage-slot mapping than the graph (which runs the
    // production materialize): identical store bytes, different reference. That
    // was the entire eager discrepancy -- a fixture lifecycle gap, not a
    // production defect.
    mat->op_params[6] = 0;
    mat->op_params[8] = stage_groups - 1;
    mat->op_params[9] = 1;
    mat->op_params[10] = 0;
    ggml_tensor * f32 = ggml_cast(ctx, mat, GGML_TYPE_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, f32);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, env.cpu);
    if (!buf) { ggml_free(ctx); return false; }
    if (ggml_nbytes(rec) != rec_h.size() || ggml_nbytes(stg) != stg_h.size() * sizeof(ggml_fp16_t)) {
        std::printf("  kvarn oracle: device bytes size mismatch rec=%zu/%zu stg=%zu/%zu\n",
                    ggml_nbytes(rec), rec_h.size(), ggml_nbytes(stg), stg_h.size() * 2);
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    ggml_backend_tensor_set(rec, rec_h.data(), 0, ggml_nbytes(rec));
    ggml_backend_tensor_set(stg, stg_h.data(), 0, ggml_nbytes(stg));
    std::vector<int64_t> hi(static_cast<size_t>(n_kv));
    for (int i = 0; i < n_kv; ++i) { hi[size_t(i)] = i; }
    ggml_backend_tensor_set(idx, hi.data(), 0, ggml_nbytes(idx));

    if (ggml_backend_graph_compute(env.cpu, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    out.assign(size_t(ggml_nelements(f32)), 0.0f);
    ggml_backend_tensor_get(f32, out.data(), 0, ggml_nbytes(f32));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Diagnostic decoder for COMPLETE sealed 128-token groups only. Input is the
// actual device record readback, never a CPU quantization of the source rows.
// Separate-record axes match portable_load_resolved: K[row=dim,col=token],
// V[row=token,col=dim]. Keep float arithmetic; materialize() would round to F16.
static bool raw_record_rows_f32(const std::vector<uint8_t> & records,
        int record_bytes, int D, int heads, int local, int body, int bits,
        bool value, std::vector<float> & out) {
    const int slices = D / 128;
    const auto layout = llama_kvarn_make_record_layout(128, bits, value);
    if (D % 128 || local % 128 || body % 128 || body < 0 || body > local ||
            record_bytes != int(layout.record_bytes) ||
            records.size() != size_t(local / 128) * heads * slices * layout.record_bytes) {
        return false;
    }
    out.assign(size_t(D) * local * heads, 0.0f);
    for (int kh = 0; kh < heads; ++kh)
        for (int pos = 0; pos < body; ++pos)
            for (int d = 0; d < D; ++d) {
                const int row = value ? pos % 128 : d % 128;
                const int col = value ? d % 128 : pos % 128;
                const size_t ri = (size_t(pos / 128) * heads * slices + kh * slices + d / 128);
                const uint8_t * rec = records.data() + ri * layout.record_bytes;
                auto axis = [&](size_t offset, int i) {
                    ggml_fp16_t h;
                    std::memcpy(&h, rec + offset + size_t(i) * sizeof(h), sizeof(h));
                    return ggml_fp16_to_fp32(h);
                };
                const uint8_t q = llama_kvarn_unpack_bits_value(rec, row * 128 + col, bits);
                out[(size_t(pos) + size_t(local) * kh) * D + d] =
                    (float(q) * axis(layout.scale_off, row) + axis(layout.zp_off, row)) *
                    axis(layout.other_off, col);
            }
    return true;
}

// Q4_0 rows: official quantize, then a REAL CPU dequant graph.
// NOTE: ggml_nbytes(Q4_0) is 18 bytes per 32 values = 0.5625 B/value, so the raw
// packed bytes cannot be memcpy'd into a float vector (it would UNDER-fill and
// leave garbage). The row must be dequantized by an actual op.
static bool q4_oracle_rows(const env_t & env, const std::vector<float> & rows,
                           int D, int n_kv, int n_kvh,
                           std::vector<float> & out, std::vector<uint8_t> & bytes_out) {
    const int64_t nrows = int64_t(n_kv) * n_kvh;
    const size_t  rsz   = size_t(ggml_row_size(GGML_TYPE_Q4_0, D));
    std::vector<uint8_t> qb(rsz * size_t(nrows));
    if (ggml_quantize_chunk(GGML_TYPE_Q4_0, rows.data(), qb.data(), 0, nrows, D, nullptr) == 0) {
        std::printf("  q4: ggml_quantize_chunk returned 0\n"); return false;
    }
    bytes_out = qb;

    ggml_init_params dp = { ggml_tensor_overhead() * 128 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(dp);
    ggml_tensor * q4  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, D, nrows);
    ggml_tensor * f32 = ggml_cast(ctx, q4, GGML_TYPE_F32);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, f32);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, env.cpu);
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(q4, qb.data(), 0, ggml_nbytes(q4));
    if (ggml_backend_graph_compute(env.cpu, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    std::vector<uint8_t> rt(qb.size());
    ggml_backend_tensor_get(q4, rt.data(), 0, ggml_nbytes(q4));
    if (rt != qb) { std::printf("  q4: backend byte roundtrip MISMATCH\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return false; }
    if (ggml_nbytes(f32) != size_t(ggml_nelements(f32)) * sizeof(float)) {
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    out.assign(size_t(ggml_nelements(f32)), 0.0f);
    ggml_backend_tensor_get(f32, out.data(), 0, ggml_nbytes(f32));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Host-side Q rotation using the SAME ggml op the graph uses, so the oracle's
// Q is bit-comparable with the WHT the CUDA route applies.
static bool wht_rows(const env_t & env, const std::vector<float> & in, int D,
                    std::vector<float> & out) {
    ggml_init_params dp = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(dp);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in.size());
    ggml_tensor * r = ggml_kvarn_wht(ctx, a, D);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, env.cpu);
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(a, in.data(), 0, ggml_nbytes(a));
    if (ggml_backend_graph_compute(env.cpu, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    out.assign(size_t(ggml_nelements(r)), 0.0f);
    ggml_backend_tensor_get(r, out.data(), 0, ggml_nbytes(r));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Same WHT as the graph applies to the exact tail: ggml_kvarn_wht PRESERVES the
// input dtype, so a tail stored as F16 must be rotated as F16. Rotating it as
// F32 (wht_rows) silently drops the graph's F16 intermediate rounding and the
// reference then disagrees with a precision that was never in play.
static bool wht_rows_f16(const env_t & env, const std::vector<ggml_fp16_t> & in, int D,
                         std::vector<float> & out) {
    ggml_init_params dp = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(dp);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, in.size());
    ggml_tensor * r = ggml_kvarn_wht(ctx, a, D);
    if (r->type != GGML_TYPE_F16) { ggml_free(ctx); return false; }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, env.cpu);
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(a, in.data(), 0, ggml_nbytes(a));
    if (ggml_backend_graph_compute(env.cpu, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    std::vector<ggml_fp16_t> rb(ggml_nelements(r));
    ggml_backend_tensor_get(r, rb.data(), 0, ggml_nbytes(r));
    out.assign(rb.size(), 0.0f);
    ggml_fp16_to_fp32_row(rb.data(), out.data(), (int) rb.size());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static bool readback(const ggml_tensor * t, std::vector<float> & out) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    out.assign(size_t(ggml_nelements(t)), 0.0f);
    if (!buf) { std::printf("  readback MISSING buffer: %s\n", t->name); return false; }
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    return true;
}


// Raw bytes of the CPU store's records / stage, captured so the caller can diff
// them against the device readback BEFORE materialization. Without this we only
// see the post-decode row difference and cannot tell "different records" apart
// from "different stage slot mapping".
static std::vector<uint8_t>  g_cpu_rec_raw[2];
static std::vector<uint16_t> g_cpu_stg_raw[2];

// KVarN rows from a CPU-side store of the SAME input rows. Used only to upload an
// F16 K/V pair for the local diagnostic; cross-checked against the device
// records/stage readback so a CPU/device store divergence cannot pass silently.
static bool kvarn_rows_from_cpu(const env_t & env, const std::vector<float> & k_cur,
                                const std::vector<float> & v_cur,
                                int group_rows, int kv1, int bits, bool is_value,
                                int stage_groups, int head_slices, std::vector<float> & out) {
    ggml_init_params dp = { ggml_tensor_overhead() * 512 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(dp);
    const int rb = int(llama_kvarn_make_record_layout(KVAR_N_GROUP, bits, is_value).record_bytes);
    ggml_tensor * cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, KVAR_N_GROUP, group_rows, kv1);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, kv1);
    ggml_tensor * stg = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, KVAR_N_GROUP, group_rows, 128*stage_groups);
    const int rec_slots = std::max(1, (kv1 + int(KVAR_N_GROUP) - 1) / int(KVAR_N_GROUP));
    ggml_tensor * rec = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, rb, group_rows, rec_slots);
    ggml_tensor * st  = ggml_kvarn_store(ctx, cur, idx, stg, rec, bits, 16, is_value, stage_groups);
    st->op_params[5] = head_slices;
    ggml_tensor * mat = ggml_kvarn_materialize(ctx, rec, st, idx, kv1, 0, 1, bits, is_value, stage_groups);
    // The CPU store must use the SAME eager contract as the device store, or the
    // two seal into different stage slots and materialize[9]=1 exposes it
    // (cpu-vs-device blew up with inf/-372 for the first row of group 1).
    st->op_params[3] = kv1;
    st->op_params[4] = 0;
    st->op_params[9] = 1;
    mat->op_params[5] = head_slices;
    mat->op_params[4] = 0;   // this path always produces the un-rotated rows
    // Mirror production llama_kv_cache_kvarn::materialize(), which sets
    // op_params[6]=swa, [8]=tail_groups, [9]=eager_records=1 and [10]=read_indirect.
    // ggml_kvarn_materialize() leaves [9] at its default, so the oracle decoded
    // with a DIFFERENT stage-slot mapping than the graph (which runs the
    // production materialize): identical store bytes, different reference. That
    // was the entire eager discrepancy -- a fixture lifecycle gap, not a
    // production defect.
    mat->op_params[6] = 0;
    mat->op_params[8] = stage_groups - 1;
    mat->op_params[9] = 1;
    mat->op_params[10] = 0;
    ggml_tensor * f32 = ggml_cast(ctx, mat, GGML_TYPE_F32);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, f32);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, env.cpu);
    if (!buf) { ggml_free(ctx); return false; }
    // ggml_backend_alloc_ctx_tensors() does NOT zero host memory. The eager
    // materialize reads stage/record cells the CPU store may leave untouched, so
    // without this the oracle compared against uninitialized malloc bytes -- which
    // is why every case reported the SAME maxdiff (a stable allocation residue).
    if (stg->data) std::memset(stg->data, 0, ggml_nbytes(stg));
    if (rec->data) std::memset(rec->data, 0, ggml_nbytes(rec));
    const std::vector<float> & src = is_value ? v_cur : k_cur;
    ggml_backend_tensor_set(cur, src.data(), 0, ggml_nbytes(cur));
    std::vector<int64_t> hi(static_cast<size_t>(kv1));
    for (int i = 0; i < kv1; ++i) { hi[size_t(i)] = i; }
    ggml_backend_tensor_set(idx, hi.data(), 0, ggml_nbytes(idx));
    if (ggml_backend_graph_compute(env.cpu, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    out.assign(size_t(ggml_nelements(f32)), 0.0f);
    ggml_backend_tensor_get(f32, out.data(), 0, ggml_nbytes(f32));
    {
        const int s = is_value ? 1 : 0;
        const uint8_t * rp = static_cast<const uint8_t *>(rec->data);
        g_cpu_rec_raw[s].assign(rp, rp + ggml_nbytes(rec));
        const uint16_t * sp = static_cast<const uint16_t *>(stg->data);
        g_cpu_stg_raw[s].assign(sp, sp + ggml_nbytes(stg) / 2);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// ---------------------------------------------------------------- case ----
struct cfg_t {
    int D, n_q, n_qh, n_kvh, n_local, n_ovf, bits;
    // Exact KVarN tail: the LAST n_tail keys of the local range are served by
    // the tail tensor instead of the body, so the local FA must cover
    // body UNION tail exactly once. 0 = body only, n_local = tail only.
    int n_tail;
    // 0 = causal overflow mask (production), 1 = every overflow key visible
    // for every query. 2 = all local/overflow keys masked (opt-in control).
    // Same operands, different mask: isolates mask handling from arithmetic.
    int mask_mode;
    // 1 = the local FA uses the PRODUCTION native exact tail: a body FA whose
    // tail rows are masked out, plus ggml_kv_tail_attention_merge() fed by
    // INDEPENDENTLY STORED exact F16 tail K/V with its own tail mask, ring
    // order and run descriptor. 0 = generic body/tail partition over the same
    // materialized tensor (kept as a separate gate).
    int native_tail;
    const char * tag;
};

// Same stable LSE merge the production path uses (llama-graph.cpp:4623-4643):
//   M = max(la, lb), o = (oa*e^(la-M) + ob*e^(lb-M)) / (e^(la-M) + e^(lb-M))
//   lse = M + log(e^(la-M) + e^(lb-M))
// An all-masked side carries -inf and contributes nothing (no 0/0).
struct pair_out { ggml_tensor * o; ggml_tensor * lse; };

static pair_out merge_two(ggml_context * gctx, ggml_tensor * oa, ggml_tensor * la,
                          ggml_tensor * ob, ggml_tensor * lb,
                          int64_t n_qh, int64_t n_q) {
    ggml_tensor * c1 = ggml_clamp(gctx, la, -1e6f, 1e30f);
    ggml_tensor * c2 = ggml_clamp(gctx, lb, -1e6f, 1e30f);
    ggml_tensor * mx = ggml_scale(gctx,
            ggml_add(gctx, ggml_add(gctx, c1, c2), ggml_abs(gctx, ggml_sub(gctx, c1, c2))), 0.5f);
    ggml_tensor * e1 = ggml_exp(gctx, ggml_sub(gctx, la, mx));
    ggml_tensor * e2 = ggml_exp(gctx, ggml_sub(gctx, lb, mx));
    ggml_tensor * w1 = ggml_reshape_4d(gctx, e1, 1, n_qh, n_q, 1);
    ggml_tensor * w2 = ggml_reshape_4d(gctx, e2, 1, n_qh, n_q, 1);
    ggml_tensor * den = ggml_add(gctx, e1, e2);
    pair_out r;
    // Plan 3.4 (lines 143-145): an empty/fully-masked range is O = 0,
    // LSE = -inf, and BOTH ranges empty returns zero. With both sides masked,
    // den == 0 and the numerator is also 0, so the raw division is 0/0 = NaN.
    // Clamp only the DENOMINATOR: 0 -> 1e-30 turns that into 0/eps = 0. It is
    // invisible for every row with a visible key, because the winning side
    // contributes e = exp(lse - max) = 1, so den >= 1 and 1 + 1e-30 == 1.0f in
    // F32. LSE keeps the UNCLAMPED den so it is exactly -inf when both are
    // masked, and the clamp/max above already resolved -inf - -inf before any
    // subtraction, per the same section.
    ggml_tensor * den_safe = ggml_clamp(gctx, den, 1e-30f, 1e38f);
    r.o   = ggml_div(gctx,
             ggml_add(gctx, ggml_mul(gctx, oa, w1), ggml_mul(gctx, ob, w2)),
             ggml_reshape_4d(gctx, den_safe, 1, n_qh, n_q, 1));
    r.lse = ggml_add(gctx, mx, ggml_log(gctx, den));
    return r;
}

// Number of valid tail-merge diagnostic records read by libggml-cuda
// (-1 = missing), so a missing capture fails as INCOMPLETE.
extern "C" int ggml_cuda_kdisc_rec_count(void);

static void run_case(const env_t & env, const cfg_t & c) {
    g_case = c.tag;
    const bool long_rotated_control = g_case.find("LONGROTATED") != std::string::npos;
    // Fixed deterministic contrast prevents 128 exact-tail keys being diluted
    // by a 102400-key body in the duplicate/drop sensitivity controls. It only
    // changes independent native V-tail input, never KVarN store inputs/bytes.
    const float exact_tail_v_contrast = long_rotated_control ? 16.0f : 1.0f;
    ++g_run_index;
    const int D = c.D, n_q = c.n_q, n_qh = c.n_qh, n_kvh = c.n_kvh;
    const int n_local = c.n_local, n_ovf = c.n_ovf, bits = c.bits;
    const int head_slices  = D / KVAR_N_GROUP;
    const int group_rows   = n_kvh * head_slices;   // == stage->ne[1] == records->ne[1]
    const float scale      = 1.0f / std::sqrt(float(D));

    const int kv1 = std::max(n_local, 1);   // a 0-length range cannot build store/view
    // The stage rings hold 128 cells per group, so the store needs enough groups
    // for the WHOLE local range: a fixed 3 covered only kv1 <= 384 and made
    // ggml_compute_forward_kvarn_store abort outright on the P=512 cases.
    const int stage_groups = std::max(3,
            int((kv1 + int(KVAR_N_GROUP) - 1) / int(KVAR_N_GROUP)) + 1);
    // Local body/exact-tail partition (plan 3.1/3.4). Computed here because the
    // contiguous body/tail masks must exist before the input tensors are
    // allocated -- ggml_flash_attn_ext requires a CONTIGUOUS mask, so a strided
    // row slice of mask1 is not an admissible operand.
    const int n_tail_rows = std::max(0, std::min(c.n_tail, kv1));
    const int n_body_rows = kv1 - n_tail_rows;
    // The ACTUAL exact-F16 tail bytes read back from the backend after upload.
    // The reference must be built from these, not from a regenerated unrounded
    // val(5/6): the graph stores fp16(val(...)) and a reference that skips that
    // rounding is comparing against numbers the backend never held.
    std::vector<float> kn_exact_rb, vn_exact_rb;
    std::vector<ggml_fp16_t> kn_exact_f16;
    // Host-side copies of the two masks the graph is fed, for the explicit
    // effective-key membership readout below.
    std::vector<float> body_mask_f32, tail_mask_f32;
    std::vector<int32_t> desc_snapshot;
    // The NATIVE tail mask, [n_tail_rows x n_q] -- the layout the merge reads.
    std::vector<float> tail_nat_f32;
    const int kv2 = std::max(n_ovf, 1);

    // Distinct value per (head, token, dim): any head/token permutation shows
    // up immediately instead of cancelling in the average.
    auto val = [](int kind, int h, int p, int d) {
        const double x = std::sin(0.7*(kind + 1) + 0.31*h + 0.113*p + 0.017*d);
        return float(0.5 + 0.4*x);
    };

    std::vector<float> q(size_t(D) * n_q * n_qh);
    for (int h = 0; h < n_qh; ++h)
        for (int t = 0; t < n_q; ++t)
            for (int d = 0; d < D; ++d) { q[(size_t(t) + size_t(n_q)*h)*D + d] = val(0, h, t, d); }

    // KVarN 'current' is [KVAR_N_GROUP, group_rows, n_kv] -- head_wide, matching
    // llama-kv-cache-kvarn::store (ne[0]=head_dim, ne[1]=n_head_kv*slices).
    std::vector<float> k_cur(size_t(KVAR_N_GROUP) * group_rows * kv1);
    std::vector<float> v_cur(size_t(KVAR_N_GROUP) * group_rows * kv1);
    for (int p = 0; p < kv1; ++p)
        for (int gr = 0; gr < group_rows; ++gr)
            for (int d = 0; d < KVAR_N_GROUP; ++d) {
                const size_t o = (size_t(p)*group_rows + gr)*KVAR_N_GROUP + d;
                k_cur[o] = val(1, gr, p, d);
                v_cur[o] = val(2, gr, p, d);
            }

    // Q4_0 K/V are [D, n_kv, n_kvh]: element (d,p,kh) at (p + kv*n_kvh)*D + d.
    std::vector<float> k2f(size_t(D) * kv2 * n_kvh);
    std::vector<float> v2f(size_t(D) * kv2 * n_kvh);
    for (int kh = 0; kh < n_kvh; ++kh)
        for (int p = 0; p < kv2; ++p)
            for (int d = 0; d < D; ++d) {
                const size_t o = (size_t(p) + size_t(kv2)*kh)*D + d;
                k2f[o] = val(3, kh, p, d);
                v2f[o] = val(4, kh, p, d);
            }

    // Absolute causal masks, stored [n_kv, n_q] and indexed p + n_kv*t.
    //   local    : query t sits at absolute (n_local - n_q + t); key p is at p.
    //              visible iff p <= n_local - n_q + t  (never exposes padding)
    //   overflow : key p is at absolute (n_local + p); query t at
    //              (n_local + n_ovf - n_q + t)  =>  p <= n_ovf - n_q + t
    std::vector<float> m1(size_t(kv1) * n_q, -INFINITY);
    std::vector<float> m2(size_t(kv2) * n_q, -INFINITY);
    for (int t = 0; t < n_q; ++t) {
        for (int p = 0; p < kv1; ++p) {
            if (n_local > 0 && p <= n_local - n_q + t && p < n_local) {
                m1[size_t(p) + size_t(kv1)*t] = 0.0f;
            }
        }
        for (int p = 0; p < kv2; ++p) {
            if (n_ovf > 0 && p <= n_ovf - n_q + t && p < n_ovf) {
                m2[size_t(p) + size_t(kv2)*t] = 0.0f;
            }
        }
    }
    if (c.mask_mode == 1) {
        // all-visible: the SAME operand tensors, only the mask changes, so a
        // failure here is mask/index handling rather than Q4 arithmetic.
        for (size_t i = 0; i < m2.size(); ++i) { m2[i] = 0.0f; }
    }
    if (c.mask_mode == 2) {
        std::fill(m1.begin(), m1.end(), -INFINITY);
        std::fill(m2.begin(), m2.end(), -INFINITY);
    }
    int vis1 = 0, vis2 = 0;
    for (int t = 0; t < n_q; ++t)
        for (int p = 0; p < kv1; ++p) { if (m1[size_t(p)+size_t(kv1)*t] == 0.0f) { ++vis1; } }
    for (int t = 0; t < n_q; ++t)
        for (int p = 0; p < kv2; ++p) { if (m2[size_t(p)+size_t(kv2)*t] == 0.0f) { ++vis2; } }
    // PER-QUERY census: an aggregate hides exactly the rows that fail. Query t
    // sits at absolute position (n_local + n_ovf - n_q + t) and at n=513
    // (n_ovf=1) every query but the last legitimately has ZERO visible
    // overflow keys.
    int ovf_zero = 0, ovf_some = 0;
    for (int t = 0; t < n_q; ++t) {
        int vis = 0;
        for (int p = 0; p < kv2; ++p) { if (m2[size_t(p) + size_t(kv2) * t] == 0.0f) { ++vis; } }
        if (vis == 0) { ++ovf_zero; } else { ++ovf_some; }
        if (n_q <= 16 || vis == 0 || t == 0 || t == n_q - 1) {
            std::printf("  [%s] ovf query t=%d abs_pos=%d visible_keys=%d/%d\n",
                        g_case.c_str(), t, n_local + n_ovf - n_q + t, vis, kv2);
        }
    }
    // One-empty and mixed visibility are ACCEPTED cases: prove they are present
    // in the matrix instead of leaving them incidental. A fully-masked query is
    // checked for O=0 / LSE=-inf by masked_rows_ok(r2, h2, h2l) below.
    std::printf("  [%s] ovf per-query visibility: %d fully-masked, %d with keys%s\n",
                g_case.c_str(), ovf_zero, ovf_some,
                (ovf_zero > 0 && ovf_some > 0) ? " [MIXED per-query]" :
                (ovf_zero == n_q ? " [ALL masked -> O=0, LSE=-inf]" :
                 (ovf_some == n_q ? " [ALL visible]" : "")));
    if (n_ovf > 0 && n_q > 1) { CHECK(ovf_zero + ovf_some == n_q); }

    // ---- Q4_0 oracle rows (official quantize + official dequant) ---------
    std::vector<float> k2d, v2d;
    std::vector<uint8_t> k2bytes, v2bytes;
    if (!q4_oracle_rows(env, k2f, D, kv2, n_kvh, k2d, k2bytes) ||
        !q4_oracle_rows(env, v2f, D, kv2, n_kvh, v2d, v2bytes)) {
        std::printf("  [%s] q4 oracle rows failed\n", g_case.c_str());
        ++g_checks; ++g_failures; return;
    }

    const ref_o r2 = oracle(D, n_q, n_qh, n_kvh, q, k2d, v2d, kv2, m2, scale);

    // rows first, so the F16 K/V can be uploaded as a leaf (CUDA FA declines the
    // in-graph materialize output geometry)
    std::vector<float> kraw_cpu, vraw_cpu;
    if (!kvarn_rows_from_cpu(env, k_cur, v_cur, group_rows, kv1, bits, false,
                             stage_groups, head_slices, kraw_cpu) ||
        !kvarn_rows_from_cpu(env, k_cur, v_cur, group_rows, kv1, bits, true,
                             stage_groups, head_slices, vraw_cpu)) {
        std::printf("  [%s] local CPU store/materialize failed\n", g_case.c_str());
        ++g_checks; ++g_failures;
        return;
    }
    auto to_head_rows_host = [&](const std::vector<float> & in, std::vector<float> & out) {
        out.assign(size_t(D) * size_t(kv1) * n_kvh, 0.0f);
        for (int p = 0; p < kv1; ++p)
            for (int kh = 0; kh < n_kvh; ++kh)
                for (int s2 = 0; s2 < head_slices; ++s2)
                    for (int d = 0; d < KVAR_N_GROUP; ++d) {
                        const int gr = kh * head_slices + s2;
                        out[(size_t(p) + size_t(kv1)*kh)*D + s2*KVAR_N_GROUP + d] =
                            in[(size_t(p)*group_rows + gr)*KVAR_N_GROUP + d];
                    }
    };
    std::vector<float> k1h_cpu, v1h_cpu;
    to_head_rows_host(kraw_cpu, k1h_cpu);
    to_head_rows_host(vraw_cpu, v1h_cpu);


    // ---- data contexts (one per backend; a context is allocated once) ----
    ggml_init_params dpar = { ggml_tensor_overhead() * 2048, nullptr, true };
    ggml_context * dctx_c = ggml_init(dpar);
    ggml_context * dctx_v = ggml_init(dpar);

    ggml_tensor * q_t  = ggml_new_tensor_4d(dctx_c, GGML_TYPE_F32, D, n_q, n_qh, 1);
    ggml_tensor * kc   = ggml_new_tensor_3d(dctx_c, GGML_TYPE_F32, KVAR_N_GROUP, group_rows, kv1);
    ggml_tensor * vc   = ggml_new_tensor_3d(dctx_c, GGML_TYPE_F32, KVAR_N_GROUP, group_rows, kv1);
    ggml_tensor * idx1 = ggml_new_tensor_1d(dctx_c, GGML_TYPE_I64, kv1);
    ggml_tensor * k_stg = ggml_new_tensor_3d(dctx_c, GGML_TYPE_F16, KVAR_N_GROUP, group_rows, 128*stage_groups);
    ggml_tensor * v_stg = ggml_new_tensor_3d(dctx_c, GGML_TYPE_F16, KVAR_N_GROUP, group_rows, 128*stage_groups);
    const int k_rb = int(llama_kvarn_make_record_layout(KVAR_N_GROUP, bits, false).record_bytes);
    const int v_rb = int(llama_kvarn_make_record_layout(KVAR_N_GROUP, bits, true ).record_bytes);
    const int rec_slots = std::max(1, (kv1 + int(KVAR_N_GROUP) - 1) / int(KVAR_N_GROUP));
    ggml_tensor * k_rec = ggml_new_tensor_3d(dctx_c, GGML_TYPE_I8, k_rb, group_rows, rec_slots);
    ggml_tensor * v_rec = ggml_new_tensor_3d(dctx_c, GGML_TYPE_I8, v_rb, group_rows, rec_slots);
    ggml_tensor * mask1 = ggml_new_tensor_4d(dctx_c, GGML_TYPE_F16, kv1, n_q, 1, 1);
    // Production native exact tail operands: their OWN exact-F16 storage, not
    // rows of the materialized KVarN tensor the body reads from.
    const int ntail_max = std::max(n_tail_rows, 1);
    ggml_tensor * kt_nat = ggml_new_tensor_4d(dctx_c, GGML_TYPE_F16, D, ntail_max, n_kvh, 1);
    ggml_tensor * vt_nat = ggml_new_tensor_4d(dctx_c, GGML_TYPE_F16, D, ntail_max, n_kvh, 1);
    ggml_tensor * tail_mask_nat = ggml_new_tensor_4d(dctx_c, GGML_TYPE_F16, ntail_max, n_q, 1, 1);
    ggml_tensor * tail_qorder   = ggml_new_tensor_2d(dctx_c, GGML_TYPE_I32, std::max(n_q, 1), 1);
    ggml_tensor * tail_rundesc  = ggml_new_tensor_2d(dctx_c, GGML_TYPE_I32, 6 + ntail_max, 1);
    ggml_tensor * mask1b = ggml_new_tensor_4d(dctx_c, GGML_TYPE_F16, kv1, n_q, 1, 1);
    ggml_tensor * mask1t = ggml_new_tensor_4d(dctx_c, GGML_TYPE_F16, kv1, n_q, 1, 1);


    ggml_tensor * k2_t   = ggml_new_tensor_3d(dctx_v, GGML_TYPE_Q4_0, D, kv2, n_kvh);
    ggml_tensor * v2_t   = ggml_new_tensor_3d(dctx_v, GGML_TYPE_Q4_0, D, kv2, n_kvh);
    ggml_tensor * mask2  = ggml_new_tensor_4d(dctx_v, GGML_TYPE_F16, kv2, n_q, 1, 1);

    // ---- geometry contract asserted BEFORE any upload or access ----------
    CHECK(q_t->ne[0] == D && q_t->ne[1] == n_q && q_t->ne[2] == n_qh);
    CHECK(kc->ne[0] == KVAR_N_GROUP && kc->ne[1] == group_rows && kc->ne[2] == kv1);
    CHECK(vc->ne[1] == group_rows && vc->ne[2] == kv1);
    CHECK(k_stg->ne[1] == group_rows && v_stg->ne[1] == group_rows);
    CHECK(k_rec->ne[1] == group_rows && v_rec->ne[1] == group_rows);
    CHECK(k_cur.size() == size_t(ggml_nelements(kc)));
    CHECK(v_cur.size() == size_t(ggml_nelements(vc)));
    CHECK(k2f.size() == size_t(ggml_nelements(k2_t)));
    CHECK(v2f.size() == size_t(ggml_nelements(v2_t)));
    CHECK(k2d.size() == size_t(ggml_nelements(k2_t)));
    CHECK(v2d.size() == size_t(ggml_nelements(v2_t)));
    CHECK(m1.size() == size_t(kv1) * n_q && m2.size() == size_t(kv2) * n_q);

    // F4_RP_ADDR_PERTURB=<bytes>: hold a dummy CUDA buffer allocated BEFORE the
    // fixture's own tensors. Not one byte of any tensor changes -- only where
    // subsequent device allocations land. Held for the whole process (static)
    // so A and A2 are shifted identically. If the gate moves under this alone,
    // the fixture's numbers depend on placement, which no initialization route
    // can explain and which points at an out-of-bounds read or a host-side
    // lifetime/alias bug rather than at missing initialization.
    static ggml_context * g_perturb_ctx = nullptr;
    static ggml_backend_buffer_t g_perturb_buf = nullptr;
    if (const char * ap = getenv("F4_RP_ADDR_PERTURB"); ap && *ap) {
        const long pbytes = std::strtol(ap, nullptr, 10);
        if (pbytes > 0 && !g_perturb_buf) {
            ggml_init_params ip = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
            g_perturb_ctx = ggml_init(ip);
            ggml_new_tensor_1d(g_perturb_ctx, GGML_TYPE_I8, pbytes);
            g_perturb_buf = ggml_backend_alloc_ctx_tensors(g_perturb_ctx, env.cuda);
            std::printf("  [%s] ADDR_PERTURB holding %ld device bytes (buffer=%p)\n",
                        g_case.c_str(), pbytes, (void *) g_perturb_buf);
        }
    }

    ggml_backend_buffer_t bcuda = ggml_backend_alloc_ctx_tensors(dctx_c, env.cuda);
    ggml_backend_buffer_t bvk   = ggml_backend_alloc_ctx_tensors(dctx_v, env.vk);
    // F4_RP_INIT_PROBE=1: compare the two initialization routes BYTE FOR BYTE on
    // this very allocation, before any upload has touched the leaves:
    //   route 1 = ggml_backend_buffer_clear(bcuda, 0)   (CUDA)
    //   route 2 = host zero upload into the four persistent leaves
    // The gate swung 1 -> 1060 failures between these two routes. If both leave
    // identical bytes here, initialization content cannot be the cause and the
    // swing has to come from the clear op's side effects (placement) instead.
    if (bcuda && getenv("F4_RP_INIT_PROBE")) {
        // NOTE: these leaves live in a CUDA buffer, so t->data is a DEVICE
        // pointer -- dereferencing it on the host segfaults (that was the crash
        // in f4-probe-20261006-060739.log, before any probe line was printed).
        // Everything goes through ggml_backend_tensor_get.
        auto nonzero = [](const void * pv, size_t n) {
            const uint8_t * b = (const uint8_t *) pv; size_t c = 0;
            for (size_t i = 0; i < n; ++i) { if (b[i]) { ++c; } }
            return c;
        };
        auto snapshot = [&](const char * label) {
            size_t nk = 0, nv = 0, nkr = 0, nvr = 0, sk = 0, sv = 0, skr = 0, svr = 0;
            std::vector<uint8_t> tmp;
            auto grab = [&](ggml_tensor * t, size_t & cnt, size_t & tot) {
                if (!t || !t->buffer) { return; }
                tot = size_t(ggml_nbytes(t));
                tmp.assign(tot, 0);
                ggml_backend_tensor_get(t, tmp.data(), 0, tot);
                cnt = nonzero(tmp.data(), tot);
            };
            grab(k_stg, nk, sk); grab(v_stg, nv, sv);
            grab(k_rec, nkr, skr); grab(v_rec, nvr, svr);
            std::printf("  [%s] INIT_PROBE %s: k_stg nonzero=%zu/%zu v_stg=%zu/%zu "
                        "k_rec=%zu/%zu v_rec=%zu/%zu\n", g_case.c_str(), label,
                        nk, sk, nv, sv, nkr, skr, nvr, svr);
            return nk + nv + nkr + nvr;
        };
        snapshot("after-alloc");
        ggml_backend_buffer_clear(bcuda, 0);
        ggml_backend_synchronize(env.cuda);
        const size_t route1 = snapshot("route1-cuda-buffer_clear");
        for (ggml_tensor * t : { k_stg, v_stg, k_rec, v_rec }) {
            if (!t || !t->data) { continue; }
            std::vector<uint8_t> z(size_t(ggml_nbytes(t)), 0);
            ggml_backend_tensor_set(t, z.data(), 0, z.size());
        }
        ggml_backend_synchronize(env.cuda);
        const size_t route2 = snapshot("route2-host-zero-upload");
        CHECK(route1 == 0);
        CHECK(route2 == 0);
        std::printf("  [%s] INIT_PROBE verdict: cuda_clear and host_zero %s\n",
                    g_case.c_str(), route1 == route2 ? "leave IDENTICAL bytes"
                                                     : "leave DIFFERENT bytes");
    }
    if (!bcuda || !bvk) {
        std::printf("  [%s] alloc failed (cuda=%p vk=%p)\n", g_case.c_str(), (void*)bcuda, (void*)bvk);
        ++g_checks; ++g_failures;
        if (bcuda) { ggml_backend_buffer_free(bcuda); }
        if (bvk)   { ggml_backend_buffer_free(bvk); }
        ggml_free(dctx_c); ggml_free(dctx_v); return;
    }

    {
        // Leaf initialization.
        //
        // PRODUCTION SAFE-INIT CONTRACT (verified in src/llama-kv-cache-kvarn.cpp):
        //   ggml_backend_buffer_clear(buf, 0) runs immediately after
        //   ggml_backend_alloc_ctx_tensors_from_buft() for the KVarN cache buffer
        //   (line 3321) and for the exact-tail buffer (line 3521), again after
        //   reallocation (line 3622) and from clear() (line 3994). Every stage and
        //   record byte is ZERO before the first store. The fixture must do the
        //   same: leaving the leaf to the allocator produced local NRMSE 1.185e-02
        //   where a defined leaf gives 2.049e-04, with identical inputs.
        //
        // F4_RP_INIT_SENTINEL=1 switches the leaf to two DISTINCT non-zero
        // patterns (K, and V = K's complement) so store write-coverage can be
        // proved byte-by-byte. That proof cannot be done with zeros: a legitimate
        // stored 0.0f is bit-equal to 0x0000. F4_RP_POISON_FIXED=<hex> pins the
        // value so A and A2 run with the same contents (if A/A2 is then
        // bit-identical, an earlier difference came from the leaf contents; if it
        // still differs, the difference is independent of them).
        const bool init_sentinel =
            getenv("F4_RP_INIT_SENTINEL") && getenv("F4_RP_INIT_SENTINEL")[0] != '\0';
        uint8_t poison = init_sentinel ? uint8_t((g_run_index & 1) ? 0x55 : 0xAA) : 0x00;
        if (const char * fx = getenv("F4_RP_POISON_FIXED"); fx && *fx) {
            poison = (uint8_t) std::strtoul(fx, nullptr, 0);
        }
        g_case_no_poison = getenv("F4_RP_NO_POISON") && getenv("F4_RP_NO_POISON")[0] != '\0';
        g_case_init_sentinel = init_sentinel;
        g_case_poison = poison;
        std::printf("  [%s] storage poison pattern=0x%02X (run_index=%d fixed=%s)\n",
                    g_case.c_str(), poison, g_run_index,
                    getenv("F4_RP_POISON_FIXED") ? getenv("F4_RP_POISON_FIXED") : "no");
        // RETRACTED (was wrong): "[kv1, 128*stage_groups) are never read by a
        // materializer asked for n_kv = kv1". That range is NOT dead. The stage
        // is indexed by slot(g)*128+p, not by logical cell, and every live group
        // is assignable any slot in 1..tail_groups (llama_kvarn_reconcile_stage_slots),
        // so the assignable region is the whole leaf. Measured: leaving
        // [kv1, capacity) uninitialized flips the native-tail cases between
        // local NRMSE 2.049e-04 and 1.185e-02 with identical inputs. The
        // records tensor is exactly rec_slots wide and is the store's own
        // output, so it is left alone and remains under initcheck.
        const bool no_poison = getenv("F4_RP_NO_POISON") && getenv("F4_RP_NO_POISON")[0] != '\0';
        g_case_no_poison = no_poison;
        if (no_poison) {
            std::printf("  [%s] poison DISABLED (F4_RP_NO_POISON)\n", g_case.c_str());
        }
        // F4_RP_POISON_CELLS=<lo>-<hi>   restrict the poison to a cell range
        // F4_RP_POISON_TENSOR=k|v        restrict to one stage leaf
        // Narrowing the range is for evidence collection only; the default no
        // longer assumes any dead range (see PRODUCTION-DERIVED VALID SLOT SET).
        // PRODUCTION-DERIVED VALID SLOT SET
        // All three production consumers use one identical mapping:
        //   store        ggml/src/ggml-cuda/kvarn.cu kvarn_store_kernel_*   (stage_slot)
        //   materialize  ggml/src/ggml-cuda/kvarn.cu ggml_cuda_op_kvarn_materialize
        //   FA inline    ggml/src/ggml-cuda/fattn-mma-kvarn.cuh ggml_cuda_fattn_kvarn_stage_pos
        //     slot(0) = 0 ; slot(g) = 1 + ((g-1) % tail_groups)  for g >= 1
        //     (or slot = assigned_slot when the ring descriptor carries one)
        // Slot allocation is llama_kvarn_reconcile_stage_slots() in
        // src/llama-kvarn.cpp: any live group may be assigned ANY slot in
        // 1..tail_groups, group 0 always slot 0, and
        //   llama-kv-cache-kvarn.cpp: stage_groups = tail_groups + 1 (non-SWA).
        // So the assignable region is exactly [0, stage_groups*KVAR_N_GROUP)
        // per stream -- i.e. the whole stage leaf. The complement (the only
        // poisonable region) is therefore EMPTY for this tensor shape, and the
        // old `[kv1, capacity)` range was NOT a dead region: it cut through
        // assignable slots and produced pure poison sensitivity.
        const long valid_cells = long(stage_groups) * long(KVAR_N_GROUP);
        long cell_lo = -1, cell_hi = -1;
        if (const char * cr = getenv("F4_RP_POISON_CELLS"); cr && *cr) {
            cell_lo = std::strtol(cr, nullptr, 10);
            const char * dash = std::strchr(cr, '-');
            cell_hi = dash ? std::strtol(dash + 1, nullptr, 10) : -1;
        }
        const char * pten = getenv("F4_RP_POISON_TENSOR");
        auto poison_stage_tail = [&](ggml_tensor * t, char which) {
            if (!t || !t->buffer || t->nb[2] == 0) { return; }
            if (pten && *pten && pten[0] != which) { return; }
            const size_t cell_bytes = size_t(t->nb[2]);
            const size_t total = ggml_nbytes(t);
            const long n_cells = long(total / cell_bytes);
            // DEFAULT: define the WHOLE stage leaf before the store.
            // The complement of the production-assignable region is EMPTY
            // (valid_cells == n_cells), so there is nothing to poison; what the
            // fixture must NOT do is leave allocator garbage behind, because the
            // native-tail graph reads it (measured: 377 failures with
            // uninitialized leaves vs 1 with a defined sentinel, same inputs and
            // same code). F4_RP_POISON_CELLS narrows the range for evidence.
            long lo = cell_lo >= 0 ? cell_lo : 0;
            long hi = cell_hi >= 0 ? cell_hi : n_cells;
            if (lo < 0) lo = 0;
            if (hi > n_cells) hi = n_cells;
            if (lo >= hi) { return; }
            if (cell_lo < 0) {
                std::printf("  [%s] %c-stage: production-assignable [0,%ld) == leaf [0,%ld), "
                            "complement EMPTY, leaf fully defined (kv1=%d)\n",
                            g_case.c_str(), which, valid_cells, n_cells, kv1);
            }
            const size_t off = size_t(lo) * cell_bytes;
            // ALWAYS allocate the whole tail, then upload only the requested
            // sub-range. A sub-range-sized vector changes the host heap footprint
            // and this fixture is address-sensitive (documented: adding one
            // observability node moved fa1/oc and flipped 20 checks), which would
            // make a cell-range A/B compare layout instead of value.
            // TWO DISTINCT defined patterns: K uses `poison`, V uses its
            // complement, so a half that still equals one tensor's sentinel can
            // never be masked by the other tensor using the same constant.
            const uint8_t pat = (g_case_init_sentinel && which == 'v')
                ? uint8_t(poison ^ 0xFF) : poison;
            std::vector<uint8_t> z(size_t(n_cells) * cell_bytes, pat);
            ggml_backend_tensor_set(t, z.data() + off, off, size_t(hi - lo) * cell_bytes);
            std::printf("  [%s] defined %c-stage cells [%ld,%ld) of %ld bytes=%zu with 0x%02X "
                        "(kv1=%d capacity=%ld, alloc=%zu)\n",
                        g_case.c_str(), which, lo, hi, n_cells, size_t(hi - lo) * cell_bytes,
                        pat, kv1, n_cells, z.size());
        };
        if (!no_poison) {
            poison_stage_tail(k_stg, 'k');
            poison_stage_tail(v_stg, 'v');
        }
        // PRODUCTION LIFECYCLE, mirrored for the two leaves the fixture never
        // uploads: src/llama-kv-cache-kvarn.cpp clears the whole KVarN cache
        // buffer to 0 right after allocation (line 3321) and the exact-tail
        // buffer (line 3521), so a record cell the store does not write is 0,
        // not allocator garbage. Done with a host upload rather than
        // ggml_backend_buffer_clear(): that CUDA op changes the device
        // allocation pattern, and this fixture is address-sensitive (it flips
        // 1 -> 1060 failures on that alone), so the lifecycle must be mirrored
        // without perturbing layout.
        if (!g_case_no_poison) {
            for (ggml_tensor * rec : { k_rec, v_rec }) {
                std::vector<uint8_t> z(size_t(ggml_nbytes(rec)), 0);
                ggml_backend_tensor_set(rec, z.data(), 0, z.size());
            }
        }
        // Address map: prove the poison range cannot alias another operand.
        {
            auto show = [&](const char * n, const ggml_tensor * t) {
                if (!t) { return; }
                std::printf("  [AMAP] %-14s %p .. %p  nb=%zu n=%lld\n", n, (const void *) t->data,
                        (const void *) ((const char *) t->data + ggml_nbytes(t)),
                        (size_t) ggml_nbytes(t), (long long) ggml_nelements(t));
            };
            show("k_stg", k_stg); show("v_stg", v_stg);
            show("k_rec", k_rec); show("v_rec", v_rec);
            show("kt_nat", kt_nat); show("vt_nat", vt_nat);
            show("kc", kc); show("vc", vc); show("q_t", q_t);
            show("mask1", mask1); show("mask1b", mask1b);
            show("tail_nat", tail_mask_nat);
        }

        ggml_backend_tensor_set(q_t, q.data(), 0, ggml_nbytes(q_t));
        ggml_backend_tensor_set(kc, k_cur.data(), 0, ggml_nbytes(kc));
        ggml_backend_tensor_set(vc, v_cur.data(), 0, ggml_nbytes(vc));
        std::vector<int64_t> hi(static_cast<size_t>(kv1));
        for (int i = 0; i < kv1; ++i) { hi[size_t(i)] = i; }
        ggml_backend_tensor_set(idx1, hi.data(), 0, ggml_nbytes(idx1));
        g_case_indices = hi;

        // The reference and the Vulkan tensor MUST receive identical bytes.
        std::vector<uint8_t> qb(size_t(ggml_nbytes(k2_t)));
        ggml_quantize_chunk(GGML_TYPE_Q4_0, k2f.data(), qb.data(), 0, int64_t(kv2)*n_kvh, D, nullptr);
        CHECK(qb == k2bytes);
        ggml_backend_tensor_set(k2_t, qb.data(), 0, ggml_nbytes(k2_t));
        ggml_quantize_chunk(GGML_TYPE_Q4_0, v2f.data(), qb.data(), 0, int64_t(kv2)*n_kvh, D, nullptr);
        CHECK(qb == v2bytes);
        ggml_backend_tensor_set(v2_t, qb.data(), 0, ggml_nbytes(v2_t));

        std::vector<ggml_fp16_t> h16;
        h16.resize(size_t(kv1) * n_q);
        for (size_t i = 0; i < h16.size(); ++i) { h16[i] = ggml_fp32_to_fp16(m1[i]); }
        ggml_backend_tensor_set(mask1, h16.data(), 0, ggml_nbytes(mask1));
        // Body/tail masks over the SAME full KV row set: the body mask keeps
        // rows [0, n_body) and suppresses the tail, the tail mask does the
        // opposite. That is the exactly-once partition expressed as masks, and
        // it keeps both FAs on the full contiguous K/V view -- CUDA declines a
        // row-sliced KVarN view (ggml_backend_supports_op=0), which would make
        // the local route NOT-EXECUTED instead of measured.
        std::vector<ggml_fp16_t> hb(size_t(kv1) * n_q), ht(size_t(kv1) * n_q);
        for (int t = 0; t < n_q; ++t) {
            for (int p = 0; p < kv1; ++p) {
                const float v = m1[size_t(p) + size_t(kv1) * t];
                const bool in_body = p < n_body_rows;
                hb[size_t(p) + size_t(kv1) * t] =
                    ggml_fp32_to_fp16(in_body ? v : -INFINITY);
                ht[size_t(p) + size_t(kv1) * t] =
                    ggml_fp32_to_fp16(in_body ? -INFINITY : v);
            }
        }
        body_mask_f32.assign(hb.begin(), hb.end());
        tail_mask_f32.assign(ht.begin(), ht.end());
        ggml_backend_tensor_set(mask1b, hb.data(), 0, ggml_nbytes(mask1b));
        ggml_backend_tensor_set(mask1t, ht.data(), 0, ggml_nbytes(mask1t));
        // Exact F16 tail storage: a DIFFERENT value function from the KVarN
        // body rows, so a reference built from it can only pass if the graph
        // really consumed the exact tail rather than a compressed body copy.
        if (n_tail_rows > 0) {
            std::vector<ggml_fp16_t> kn(size_t(D) * n_tail_rows * n_kvh),
                                     vn(size_t(D) * n_tail_rows * n_kvh);
            for (int kh = 0; kh < n_kvh; ++kh)
                for (int j = 0; j < n_tail_rows; ++j)
                    for (int d = 0; d < D; ++d) {
                        const int p = n_body_rows + j;
                        kn[size_t(d) + size_t(D) * (j + n_tail_rows * kh)] =
                            ggml_fp32_to_fp16(val(5, kh, p, d));
                        vn[size_t(d) + size_t(D) * (j + n_tail_rows * kh)] =
                            ggml_fp32_to_fp16(exact_tail_v_contrast * val(6, kh, p, d));
                    }
            if (long_rotated_control) {
                std::printf("  [%s] exact native V-tail contrast=%.0f before F16 storage; store inputs unchanged; negative controls use actual native union\n",
                        g_case.c_str(), exact_tail_v_contrast);
            }
            ggml_backend_tensor_set(kt_nat, kn.data(), 0, ggml_nbytes(kt_nat));
            ggml_backend_tensor_set(vt_nat, vn.data(), 0, ggml_nbytes(vt_nat));
            {
                std::vector<ggml_fp16_t> rb(ggml_nelements(kt_nat));
                ggml_backend_tensor_get(kt_nat, rb.data(), 0, ggml_nbytes(kt_nat));
                kn_exact_f16.assign(rb.begin(), rb.end());
                kn_exact_rb.assign(ggml_nelements(kt_nat), 0.0f);
                ggml_fp16_to_fp32_row(rb.data(), kn_exact_rb.data(), (int) rb.size());
                rb.assign(ggml_nelements(vt_nat), 0);
                ggml_backend_tensor_get(vt_nat, rb.data(), 0, ggml_nbytes(vt_nat));
                vn_exact_rb.assign(ggml_nelements(vt_nat), 0.0f);
                ggml_fp16_to_fp32_row(rb.data(), vn_exact_rb.data(), (int) rb.size());
                std::printf("  [%s] exact tail stored F16 bytes: K=%zu V=%zu "
                            "(read back from the device, dtype=f16)\n",
                            g_case.c_str(), ggml_nbytes(kt_nat), ggml_nbytes(vt_nat));
            }
            std::vector<ggml_fp16_t> tm(size_t(n_tail_rows) * n_q,
                                        ggml_fp32_to_fp16(-INFINITY));
            for (int t = 0; t < n_q; ++t)
                for (int j = 0; j < n_tail_rows; ++j) {
                    const int p = n_body_rows + j;
                    if (m1[size_t(p) + size_t(kv1) * t] == 0.0f) {
                        tm[size_t(j) + size_t(n_tail_rows) * t] = ggml_fp32_to_fp16(0.0f);
                    }
                }
            tail_nat_f32.assign(tm.begin(), tm.end());
            ggml_backend_tensor_set(tail_mask_nat, tm.data(), 0, ggml_nbytes(tail_mask_nat));
            std::vector<int32_t> qo;
            qo.resize(static_cast<size_t>(n_q));
            for (int t = 0; t < n_q; ++t) { qo[size_t(t)] = t; }
            ggml_backend_tensor_set(tail_qorder, qo.data(), 0, ggml_nbytes(tail_qorder));
            // Descriptor derived from the PRODUCTION generators, not guessed:
            //   set_tail_query_plan()      -> desc[0]=seq_id, [1]=order offset,
            //                                 [2]=n_queries, [3]=run length
            //   set_input_tail_body_plan() -> desc[4]=tail slot count,
            //                                 [5]=-1 (no current row),
            //                                 [6+i]=ring slot of tail key i
            // This harness has one sequence, one active query group and a tail
            // tensor whose rows ARE the ring slots, so the four head fields are
            // (0, 0, n_q, 1) and the slots are 0..n_tail-1.
            std::vector<int32_t> rd(size_t(6 + n_tail_rows), -1);
            rd[0] = 0;                    // seq_id
            rd[1] = 0;                    // offset into query_order
            rd[2] = int32_t(n_q);         // n_queries in this active group
            rd[3] = 1;                    // consecutive runs with same seq/count
            rd[4] = int32_t(n_tail_rows); // tail candidates (ring slots)
            rd[5] = -1;                   // no graph-local current row
            for (int j = 0; j < n_tail_rows; ++j) { rd[size_t(6 + j)] = j; }
            ggml_backend_tensor_set(tail_rundesc, rd.data(), 0, ggml_nbytes(tail_rundesc));
            // EXACT-TAIL READ SET, from the actual tail FA operands and the
            // descriptor just written -- not from the generic stage formula.
            // Tail FA consumes:  kt_nat/vt_nat [D, ntail_max, n_kvh, F16] at the
            // ring slots desc[6..6+desc[4]), lanes all kh in [0,n_kvh) x d in
            // [0,D); tail_mask_nat [ntail_max, n_q, F16] at rows [0,desc[4]).
            {
                const int32_t n_slots = rd[4];
                int32_t min_slot = INT32_MAX, max_slot = -1;
                for (int32_t j = 0; j < n_slots; ++j) {
                    const int32_t sl = rd[6 + j];
                    if (sl < 0) { continue; }
                    min_slot = std::min(min_slot, sl); max_slot = std::max(max_slot, sl);
                }
                const size_t elem_nat = size_t(D) * size_t(ntail_max) * size_t(n_kvh);
                const size_t elem_msk = size_t(ntail_max) * size_t(n_q);
                const size_t lo = (max_slot >= 0 ? size_t(min_slot) * size_t(D) * size_t(n_kvh) : 0);
                const size_t hi = (max_slot >= 0 ? (size_t(max_slot) + 1) * size_t(D) * size_t(n_kvh) : 0);
                std::printf("  [%s] TAIL-READ slots=[%d..%d) of ntail_max=%d desc[4]=%d "
                            "heads=[0..%d) lanes=[0..%d) | kt/vt elems %zu..%zu of %zu "
                            "mask rows 0..%d of %d\n",
                            g_case.c_str(), min_slot, max_slot + (max_slot >= 0), ntail_max,
                            n_slots, n_kvh, D, lo, hi, elem_nat, n_slots, elem_msk);
                CHECK(n_slots >= 0 && n_slots <= ntail_max);
                CHECK(max_slot < ntail_max);
                CHECK(hi <= elem_nat);
                CHECK(size_t(n_slots) <= elem_msk);
            }
            desc_snapshot = rd;
            std::printf("  [%s] run_desc stride=%d seq=%d order_off=%d n_queries=%d run=%d "
                        "tail_slots=%d current=%d slots=[0..%d) | query_order[0..%d)=identity\n",
                        g_case.c_str(), 6 + n_tail_rows, rd[0], rd[1], rd[2], rd[3],
                        rd[4], rd[5], n_tail_rows - 1, n_q);
        }
        h16.assign(size_t(kv2) * n_q, ggml_fp32_to_fp16(-INFINITY));
        for (size_t i = 0; i < h16.size(); ++i) { h16[i] = ggml_fp32_to_fp16(m2[i]); }
        ggml_backend_tensor_set(mask2, h16.data(), 0, ggml_nbytes(mask2));
    }

    // ---- graph (no alloc; the scheduler allocates) -----------------------
    ggml_init_params gpar = { ggml_tensor_overhead() * 3072 + ggml_graph_overhead(), nullptr, true };
    ggml_context * gctx = ggml_init(gpar);

    ggml_tensor * s_k = ggml_kvarn_store(gctx, kc, idx1, k_stg, k_rec, bits, 16, false, stage_groups);
    ggml_tensor * s_v = ggml_kvarn_store(gctx, vc, idx1, v_stg, v_rec, bits, 16, true,  stage_groups);
    s_k->op_params[5] = head_slices;
    s_v->op_params[5] = head_slices;
    // Production contract (llama-kv-cache-kvarn.cpp llama_kv_cache_kvarn::store):
    //   result->op_params[9] = 1; // commit every completed record before it
    //                             // leaves the live workspace
    // ggml_kvarn_store() defaults this to 0, which delays sealing and changes
    // which (group, token) boundaries are written; the fixture must run the
    // lifecycle production runs, not a weaker one.
    if (getenv("F4_RP_NO_EAGER") == nullptr || getenv("F4_RP_NO_EAGER")[0] == '\0') {
        s_k->op_params[9] = 1;
        s_v->op_params[9] = 1;
    } else {
        std::printf("  [%s] eager_records DISABLED (ggml_kvarn_store default 0)\n", g_case.c_str());
    }
    // Same call site sets op_params[3] = kvarn_workspace_tokens_per_stream_hint(sinfo).
    // The fixture stores idx1 = [0,1,...,kv1-1] -- one contiguous run over one
    // stream -- so the hint is exactly kv1. Leaving it 0 makes the eager sealer
    // skip its stage-slot lookup (source_token < tokens_per_stream never holds)
    // and fold the WRONG stage rows into the boundary group's scales.
    s_k->op_params[3] = int32_t(kv1);
    s_v->op_params[3] = int32_t(kv1);
    // production: op_params[4] = swa ? 1 : 0 (no SWA here)
    s_k->op_params[4] = 0;
    s_v->op_params[4] = 0;
    ggml_tensor * k_v = ggml_kvarn_view(gctx, k_rec, s_k, idx1, kv1, 0, 1, bits, false, stage_groups);
    ggml_tensor * v_v = ggml_kvarn_view(gctx, v_rec, s_v, idx1, kv1, 0, 1, bits, true,  stage_groups);
    // Keep the RAW view nodes: k_v/v_v are immediately reshaped and permuted
    // into [D, n_kvh, kv1] then [D, kv1, n_kvh], after which ne[2] is n_kvh, not
    // n_kv. Reading n_kv off the permuted handle silently derived a wrong read
    // set for the local-empty cases (n_kv read as 2/4 instead of 1).
    ggml_tensor * k_view = k_v, * v_view = v_v;
    // view -> [128, n_kvh*head_slices, n_kv, 1]; reshape only when head-wide
    if (head_slices > 1) {
        k_v = ggml_reshape_4d(gctx, k_v, D, n_kvh, kv1, 1);
        v_v = ggml_reshape_4d(gctx, v_v, D, n_kvh, kv1, 1);
    }
    // view -> [KVAR_N_GROUP, n_kvh*head_slices, n_kv, 1]
    std::printf("  [%s] view k ne=[%lld,%lld,%lld,%lld]\n", g_case.c_str(),
                (long long) k_v->ne[0], (long long) k_v->ne[1],
                (long long) k_v->ne[2], (long long) k_v->ne[3]);
    // after the optional head-wide reshape the view is [D, n_kvh, n_kv, 1]
    CHECK(k_v->ne[0] == D && k_v->ne[1] == n_kvh && k_v->ne[2] == kv1);
    CHECK(size_t(ggml_nelements(k_v)) == size_t(D) * n_kvh * kv1);
    k_v = ggml_permute(gctx, k_v, 0, 2, 1, 3);
    v_v = ggml_permute(gctx, v_v, 0, 2, 1, 3);
    // after permute: [D, n_kv, n_kvh] -- the layout FA expects
    std::printf("  [%s] perm k ne=[%lld,%lld,%lld,%lld]\n", g_case.c_str(),
                (long long) k_v->ne[0], (long long) k_v->ne[1],
                (long long) k_v->ne[2], (long long) k_v->ne[3]);
    CHECK(k_v->ne[0] == D && k_v->ne[1] == kv1 && k_v->ne[2] == n_kvh);
    CHECK(v_v->ne[0] == D && v_v->ne[1] == kv1 && v_v->ne[2] == n_kvh);

    // ---- local range ------------------------------------------------------
    // The native CUDA KVarN FA route admits ONLY KVARN_DOMAIN_ROTATED
    // (ggml-cuda/fattn-kvarn-dispatch.cu:731). The store rotates the records
    // internally (ggml-cuda/kvarn.cu:832-875), so the graph only has to WHT Q
    // and WHT O back -- both done here, mirroring llama-graph.cpp:4223-4226 and
    // 4604-4609. The oracle rows are read from the device records/stage and the
    // oracle Q is rotated by the same ggml_kvarn_wht op.
    // Production recipe (src/llama-graph.cpp:4223-4226): Q is WHT-rotated, K/V
    // come from the NATIVE kvarn views (the store already rotated the records
    // internally), and the FA domain is ROTATED. With views on src[1]/src[2]
    // the LSE export is admitted (a generic packed FA has no LSE path --
    // ggml-cuda/fattn.cu:1032-1035 declines it fail-closed).
    ggml_tensor * q1r = ggml_kvarn_wht(gctx, q_t, D);

    // ---- local range: body + exact tail, one contribution per key --------
    // Plan 3.1 keeps the exact tail on the local range only and 3.4 requires
    // the LOCAL LSE to normalize the complete local output (body + tail)
    // before the overflow merge. The local range is therefore partitioned into
    // body [0, n_body) and tail [n_body, kv1): disjoint, and their union is the
    // whole local range -- asserted below rather than assumed. The measured
    // local FA is built from those two pieces and combined with the SAME LSE
    // merge the production graph uses, so lse1 IS the body+tail LSE that the
    // independent oracle (r1, over the full local union) is compared against.
    CHECK(n_body_rows >= 0 && n_tail_rows >= 0 && n_body_rows + n_tail_rows == kv1);
    {
        // exactly-once, stated as sets rather than implied by arithmetic
        std::vector<int> seen(size_t(kv1), 0);
        for (int i = 0; i < n_body_rows; ++i) { seen[size_t(i)] += 1; }
        for (int i = n_body_rows; i < kv1; ++i) { seen[size_t(i)] += 1; }
        int uncovered = 0, twice = 0;
        for (int i = 0; i < kv1; ++i) { if (seen[size_t(i)] == 0) ++uncovered; if (seen[size_t(i)] > 1) ++twice; }
        std::printf("  [%s] local union: body=[0,%d) tail=[%d,%d) kv1=%d uncovered=%d duplicated=%d\n",
                    g_case.c_str(), n_body_rows, n_body_rows, kv1, kv1, uncovered, twice);
        CHECK(uncovered == 0);
        CHECK(twice == 0);
    }
    // Domain of the local FA: ROTATED (established contract) or, for the
    // production prefill route, ROTATED_K_ORIGINAL_V (K rotated, V original).
    // The CPU reference route keeps ROTATED because the CPU kernel is
    // domain-agnostic and emits a rotated O (see the globals above).
    auto kvarn_fa = [&](ggml_tensor * kx, ggml_tensor * vx, ggml_tensor * mx) {
        ggml_tensor * fa = ggml_flash_attn_ext(gctx, q1r, kx, vx, mx, scale, 0.0f, 0.0f);
        fa->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN] =
            (g_domain_original_v && !g_force_cpu_fa1)
                ? GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V
                : GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED;
        return fa;
    };

    ggml_tensor * fa1 = nullptr;
    ggml_tensor * lse1 = nullptr;
    ggml_tensor * la_b = nullptr, * la_t = nullptr;   // body/tail LSE, non-native route only
    // Diagnostic-only body/tail observers for the NATIVE route. ggml_kv_tail_attention_merge
    // is ggml_flash_attn_ext_add_kv_tail(): it sets src[5..9] and returns the SAME tensor,
    // so fa1 == fa_b and body+tail are fused in ONE kernel node -- there is no body/tail
    // intermediate in the graph to read. These two extra FAs recompute body-only and
    // tail-only over the SAME operands so the real body/tail O and LSE can be captured.
    // Diagnostic only: they perturb the graph, so red/green must be compared with the
    // SAME binary (which is how they are used).
    ggml_tensor * kdisc_bfa = nullptr, * kdisc_blse = nullptr;
    ggml_tensor * kdisc_tfa = nullptr, * kdisc_tlse = nullptr;
    ggml_tensor * kdisc_bo = nullptr, * kdisc_bl = nullptr;
    ggml_tensor * kdisc_to = nullptr, * kdisc_tl = nullptr;
    ggml_tensor * o1_rot = nullptr;   // combined local O, still ROTATED
    ggml_tensor * obs_body_fa = nullptr, * obs_body_lse = nullptr;
    if (n_tail_rows > 0 && c.native_tail) {
        // Works for tail-only too: with n_body_rows == 0, mask1b is all
        // -INF, so the body FA legitimately contributes O=0/LSE=-inf and the
        // native merge carries the keys -- the real empty/masked-body route
        // rather than a fall-through to the generic partition.
        // PRODUCTION native exact tail (src/llama-graph.cpp build_attn): one
        // body FA over the body rows (tail rows masked out) and then
        // ggml_kv_tail_attention_merge() attaching the exact F16 tail operands
        // plus its ring order and run descriptor. The tail K/V come from their
        // OWN F16 storage, not from the materialized KVarN tensor, and are WHT
        // rotated exactly as the production builder rotates k_tail.
        ggml_tensor * fa_b = kvarn_fa(k_v, v_v, mask1b);
        // Observational copy: the merge MUTATES fa_b in place, so a body-only
        // O/LSE can only be sampled from a second, independent node over the
        // same operands. Used to locate the first differing intermediate.
        // Observational only, and OFF BY DEFAULT because it is not benign:
        // adding this node grew the graph 42 -> 46 nodes and made fa1/oc land at
        // DIFFERENT addresses on the two A/A2 builds (20 failures), while the
        // same case with it absent is 0 failures and bit-identical at 42 nodes
        // with identical addresses. Enable with F4_RP_OBS=1 only when the
        // body-O/LSE intermediate is specifically wanted, accepting that the
        // measurement then runs on a perturbed graph.
        if (getenv("F4_RP_OBS") && getenv("F4_RP_OBS")[0] != '\0') {
            obs_body_fa = kvarn_fa(k_v, v_v, mask1b);
            obs_body_lse = ggml_flash_attn_ext_lse_out(gctx, obs_body_fa);
        }
        ggml_tensor * kt_r = ggml_kvarn_wht(gctx, kt_nat, D);
        // Match production: native tail V is rotated in every body domain.
        // The domain-3 kernel applies its inverse WHT to the tail contribution
        // as well as the body, so passing stored-original V transforms it twice.
        ggml_tensor * vt_r = ggml_kvarn_wht(gctx, vt_nat, D);
        // Same dtype/domain contract as the body operands the FA consumes: the
        // tail stays F16 (the stored exact representation) through the WHT and
        // the FA runs in KVARN_DOMAIN_ROTATED, exactly like fa_b.
        CHECK(kt_r->type == GGML_TYPE_F16 && kt_r->ne[0] == D &&
              kt_r->ne[1] == n_tail_rows && kt_r->ne[2] == n_kvh);
        CHECK(vt_r->type == GGML_TYPE_F16 && vt_r->ne[0] == D &&
              vt_r->ne[1] == n_tail_rows && vt_r->ne[2] == n_kvh);
        std::printf("  [%s] rotated tail operands: K dtype=%s ne=[%lld,%lld,%lld] "
                    "V dtype=%s | FA domain=KVARN_DOMAIN_ROTATED (body and tail)\n",
                    g_case.c_str(), ggml_type_name(kt_r->type),
                    (long long) kt_r->ne[0], (long long) kt_r->ne[1], (long long) kt_r->ne[2],
                    ggml_type_name(vt_r->type));
        fa1 = ggml_kv_tail_attention_merge(gctx, fa_b, kt_r, vt_r,
                tail_mask_nat, tail_qorder, tail_rundesc);
        // Production (llama-graph.cpp build_attn) and the native-tail graph test
        // both force F32 accumulation on the tail-merged node. Without it the
        // merged node keeps the default precision and the run is not bit
        // reproducible: A/A2 differed by exactly one float ULP and the same
        // case moved 1.64e-04 <-> 3.35e-03 across runs.
        ggml_flash_attn_ext_set_prec(fa1, GGML_PREC_F32);
        lse1 = ggml_flash_attn_ext_lse_out(gctx, fa1);
        o1_rot = fa1;
        if (getenv("F4_RP_KDISC")) {
            kdisc_bfa  = kvarn_fa(k_v, v_v, mask1b);
            kdisc_blse = ggml_flash_attn_ext_lse_out(gctx, kdisc_bfa);
            kdisc_tfa  = kvarn_fa(k_v, v_v, mask1t);
            kdisc_tlse = ggml_flash_attn_ext_lse_out(gctx, kdisc_tfa);
            std::printf("  [%s] KDISC-OBSERVER body(tail-masked) + tail(body-masked) FAs "
                        "added for capture (diagnostic graph)\n", g_case.c_str());
        }
        std::printf("  [%s] local route: NATIVE exact-tail FA "
                    "(body=%d tail=%d, exact-F16 tail storage, ring+run descriptor)\n",
                    g_case.c_str(), n_body_rows, n_tail_rows);
    } else if (n_body_rows > 0 && n_tail_rows > 0) {
        // body + tail: two FAs on disjoint row sets, combined with the
        // production LSE merge. The tail rows come from the SAME realized
        // representation as the body, so the union is the single mixed
        // representation and nothing is re-quantized.
        ggml_tensor * fa_b = kvarn_fa(k_v, v_v, mask1b);
        la_b = ggml_flash_attn_ext_lse_out(gctx, fa_b);
        ggml_tensor * fa_t = kvarn_fa(k_v, v_v, mask1t);
        la_t = ggml_flash_attn_ext_lse_out(gctx, fa_t);
        pair_out bt = merge_two(gctx, fa_b, la_b, fa_t, la_t, n_qh, n_q);
        fa1 = fa_b;                 // kept for the fa1 shape CHECK below
        o1_rot = bt.o;
        lse1 = bt.lse;
        std::printf("  [%s] local route: body+tail FA (body=%d tail=%d)\n",
                    g_case.c_str(), n_body_rows, n_tail_rows);
    } else if (n_tail_rows > 0) {
        // tail-only: the body contributes no keys at all
        fa1 = kvarn_fa(k_v, v_v, mask1);
        lse1 = ggml_flash_attn_ext_lse_out(gctx, fa1);
        o1_rot = fa1;
        std::printf("  [%s] local route: tail-only FA (%d keys)\n", g_case.c_str(), n_tail_rows);
    } else {
        fa1 = kvarn_fa(k_v, v_v, mask1);
        lse1 = ggml_flash_attn_ext_lse_out(gctx, fa1);
        o1_rot = fa1;
        std::printf("  [%s] local route: body-only FA (%d keys)\n", g_case.c_str(), n_body_rows);
    }
    // ROTATED returns O in the ROTATED domain. Production restores the original
    // V domain BEFORE the merge (src/llama-graph.cpp:4604-4609); the normalized
    // Hadamard is its own inverse, so the same op restores it. Comparing fa1 raw
    // against original-domain V is what made O land outside the convex hull.
    ggml_tensor * o1 = ggml_kvarn_wht(gctx, o1_rot, D);
    CHECK(o1->ne[0] == D && o1->ne[1] == n_qh && o1->ne[2] == n_q);

    ggml_tensor * fa2 = ggml_flash_attn_ext(gctx, q_t, k2_t, v2_t, mask2, scale, 0.0f, 0.0f);
    ggml_tensor * lse2 = ggml_flash_attn_ext_lse_out(gctx, fa2);

    CHECK(lse1 != nullptr && lse1->ne[0] == n_qh && lse1->ne[1] == n_q);
    CHECK(o1_rot != nullptr && o1_rot->ne[0] == D && o1_rot->ne[1] == n_qh && o1_rot->ne[2] == n_q);

    // stable merge, mirroring src/llama-graph.cpp:4623-4639
    ggml_tensor * c1 = ggml_clamp(gctx, lse1, -1e6f, 1e30f);
    ggml_tensor * c2 = ggml_clamp(gctx, lse2, -1e6f, 1e30f);
    ggml_tensor * mx = ggml_scale(gctx,
            ggml_add(gctx, ggml_add(gctx, c1, c2), ggml_abs(gctx, ggml_sub(gctx, c1, c2))), 0.5f);
    ggml_tensor * e1 = ggml_exp(gctx, ggml_sub(gctx, lse1, mx));
    ggml_tensor * e2 = ggml_exp(gctx, ggml_sub(gctx, lse2, mx));
    ggml_tensor * w1 = ggml_reshape_4d(gctx, e1, 1, n_qh, n_q, 1);
    ggml_tensor * w2 = ggml_reshape_4d(gctx, e2, 1, n_qh, n_q, 1);
    // Same empty-range rule as merge_two (plan 3.4): den == 0 only when BOTH
    // ranges are masked, and the numerator is 0 there, so clamping just the
    // denominator yields O = 0 instead of 0/0 = NaN while LSE keeps -inf.
    ggml_tensor * den = ggml_add(gctx, e1, e2);
    ggml_tensor * den_safe = ggml_clamp(gctx, den, 1e-30f, 1e38f);
    ggml_tensor * o = ggml_div(gctx,
            ggml_add(gctx, ggml_mul(gctx, o1, w1), ggml_mul(gctx, fa2, w2)),
            ggml_reshape_4d(gctx, den_safe, 1, n_qh, n_q, 1));
    ggml_tensor * lse_m = ggml_add(gctx, mx, ggml_log(gctx, den));

    auto cp4 = [&](ggml_tensor * src) {
        return ggml_cpy(gctx, src, ggml_new_tensor_4d(gctx, GGML_TYPE_F32,
                src->ne[0], src->ne[1], src->ne[2], src->ne[3]));
    };
    auto cp3 = [&](ggml_tensor * src) {
        return ggml_cpy(gctx, src, ggml_new_tensor_3d(gctx, GGML_TYPE_F32,
                src->ne[0], src->ne[1], src->ne[2]));
    };
    ggml_tensor * o1c = cp4(o1), * o2c = cp4(fa2), * oc = cp4(o);
    ggml_tensor * l1c = cp3(lse1), * l2c = cp3(lse2), * lmc = cp3(lse_m);

    // Report NOT-EXECUTED only when a route actually did not run.
    if (!ggml_backend_supports_op(env.cuda, fa1)) {
        std::printf("  [%s] NOT-EXECUTED local CUDA KVarN FA "
                    "(ggml_backend_supports_op=0)\n", g_case.c_str());
    }
    if (!ggml_backend_supports_op(env.vk, fa2)) {
        std::printf("  [%s] NOT-EXECUTED overflow Vulkan Q4 FA "
                    "(ggml_backend_supports_op=0)\n", g_case.c_str());
    }
    if (!ggml_backend_supports_op(env.cuda, fa1) || !ggml_backend_supports_op(env.vk, fa2)) {
        std::printf("  [%s] route DECLINED local=%d ovf=%d\n", g_case.c_str(),
                    (int) ggml_backend_supports_op(env.cuda, fa1),
                    (int) ggml_backend_supports_op(env.vk, fa2));
        ++g_checks; ++g_failures; ++g_skipped;
        ggml_free(gctx); ggml_free(dctx_c); ggml_free(dctx_v);
        ggml_backend_buffer_free(bcuda); ggml_backend_buffer_free(bvk);
        return;
    }

    ggml_backend_t order[3] = { env.cuda, env.vk, env.cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(order, NULL, 3, 8192, false, true);
    ggml_backend_sched_set_tensor_backend(sched, k_v, env.cuda);
    ggml_backend_sched_set_tensor_backend(sched, v_v, env.cuda);
    // mat_k/mat_v/fa1/lse1 are left to the scheduler: the KVarN materialize op
    // lives on CUDA, so forcing them elsewhere splits the graph across a backend
    // that never produced their inputs.
    // TEST-ONLY knob (default OFF, production machine untouched): force the
    // LOCAL KVarN FA onto the CPU backend so the same case can be measured
    // against the SAME FP64 oracle on a second, independent route. This is the
    // reference route the plan asks for (CPU attention over the dequantized KV
    // of the same per-range representation, stable accumulation). The scheduler
    // then inserts the copies for the cache views exactly as it does when the
    // CUDA dispatcher declines the op.
    const bool force_cpu_fa1 = getenv("F4_RP_FORCE_CPU_FA") != nullptr;
    if (force_cpu_fa1) {
        ggml_backend_sched_set_tensor_backend(sched, fa1, env.cpu);
        if (o1 != nullptr) ggml_backend_sched_set_tensor_backend(sched, o1, env.cpu);
        if (lse1 != nullptr) ggml_backend_sched_set_tensor_backend(sched, lse1, env.cpu);
        std::printf("  [%s] local FA forced to CPU (reference route)\n", g_case.c_str());
    }
    ggml_backend_sched_set_tensor_backend(sched, fa2, env.vk);
    ggml_backend_sched_set_tensor_backend(sched, lse2, env.vk);
    ggml_backend_sched_set_tensor_backend(sched, o, env.cuda);
    ggml_backend_sched_set_tensor_backend(sched, lse_m, env.cuda);

    ggml_cgraph * gf = ggml_new_graph(gctx);
    for (ggml_tensor * t : { lse_m, o1c, l1c, o2c, l2c, oc, lmc }) { ggml_build_forward_expand(gf, t); }
    ggml_tensor * obc = nullptr, * lbcc = nullptr;
    if (obs_body_fa && obs_body_lse) {
        obc = cp4(obs_body_fa);
        lbcc = cp3(obs_body_lse);
        ggml_build_forward_expand(gf, obc);
        ggml_build_forward_expand(gf, lbcc);
    }
    if (kdisc_bfa && kdisc_blse) {
        kdisc_bo = cp4(kdisc_bfa);  kdisc_bl = cp3(kdisc_blse);
        ggml_build_forward_expand(gf, kdisc_bo); ggml_build_forward_expand(gf, kdisc_bl);
    }
    if (kdisc_tfa && kdisc_tlse) {
        kdisc_to = cp4(kdisc_tfa);  kdisc_tl = cp3(kdisc_tlse);
        ggml_build_forward_expand(gf, kdisc_to); ggml_build_forward_expand(gf, kdisc_tl);
    }
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        std::printf("  [%s] alloc_graph FAILED\n", g_case.c_str());
        ++g_checks; ++g_failures; ++g_skipped;
        ggml_backend_sched_free(sched); ggml_free(gctx);
        ggml_free(dctx_c); ggml_free(dctx_v);
        ggml_backend_buffer_free(bcuda); ggml_backend_buffer_free(bvk);
        return;
    }
    // ---- GRAPH-BUILD MANIFEST (address independent) -------------------------
    // Two independent builds of the same case happen inside one process (A and
    // A2). Emit a canonical, address-independent manifest for each and diff them:
    //   node id, op, type, name, ne, nb, ALL 64 bytes of op_params, src edges,
    //   view_src/view_offs, flags, resolved src shapes, and a buffer id derived
    //   from order of first appearance (so backend/buffer partitioning -- the
    //   scheduled split -- is visible without leaking addresses).
    // F4_RP_OPHASH additionally hashes every consumed operand (device tensors,
    // host masks, run_desc/query_order including their -1 padding).
    static std::vector<std::string> g_mf[2];
    static bool g_mf_seen[2] = {false, false};
    static std::string g_mf_tag[2];
    static std::vector<std::string> g_oph[2];
    if (getenv("F4_RP_MANIFEST")) {
        std::vector<std::string> mf;
        const int nn = ggml_graph_n_nodes(gf);
        std::map<ggml_tensor *, int> nidx;
        for (int i = 0; i < nn; ++i) { nidx[ggml_graph_node(gf, i)] = i; }
        std::map<const void *, int> bidx;
        auto bnum = [&](ggml_backend_buffer_t b) {
            if (!b) { return -1; }
            auto it = bidx.find((const void *) b);
            if (it != bidx.end()) { return it->second; }
            const int id = int(bidx.size());
            bidx[(const void *) b] = id;
            return id;
        };
        auto idof = [&](ggml_tensor * t) {
            auto it = nidx.find(t);
            return it == nidx.end() ? -1 : it->second;
        };
        auto shp = [](const ggml_tensor * t) {
            char b[96];
            snprintf(b, sizeof b, "%lldx%lldx%lldx%lld/%zu,%zu,%zu,%zu",
                     (long long) t->ne[0], (long long) t->ne[1],
                     (long long) t->ne[2], (long long) t->ne[3],
                     (size_t) t->nb[0], (size_t) t->nb[1],
                     (size_t) t->nb[2], (size_t) t->nb[3]);
            return std::string(b);
        };
        {
            // op_params is int32_t[GGML_MAX_OP_PARAMS/4] == 16 slots (ggml.h:760).
            // Reading beyond slot 16 walks into the following struct members
            // (grad / view_src / name ...), which are POINTERS and therefore
            // differ on every build -- that was the source of the spurious
            // "MANIFEST DIFFERS" and must not be compared.
            char h[2 * GGML_MAX_OP_PARAMS + 1];
            char b[4096];
            std::map<int, int> oph;
            for (int i = 0; i < nn; ++i) { oph[(int) ggml_graph_node(gf, i)->op]++; }
            std::string oh;
            for (auto & kv : oph) { oh += std::to_string(kv.first) + "x" + std::to_string(kv.second) + ","; }
            snprintf(b, sizeof b, "HEADER nodes=%d case=%s ops=%s", nn, g_case.c_str(), oh.c_str());
            mf.push_back(b);
            for (int i = 0; i < nn; ++i) {
                const ggml_tensor * t = ggml_graph_node(gf, i);
                const int nslots = GGML_MAX_OP_PARAMS / (int) sizeof(int32_t);
                for (int k = 0; k < nslots; ++k) {
                    snprintf(h + k * 8, 9, "%08x", (unsigned) t->op_params[k]);
                }
                h[8 * nslots] = 0;
                snprintf(b, sizeof b,
                    "M %03d op=%d type=%d name=%.32s ne=%s src=[%d,%d,%d] "
                    "vsrc=%d voffs=%lld flags=%u buf=%d p=%s",
                    i, (int) t->op, (int) t->type,
                    t->name[0] ? t->name : "-", shp(t).c_str(),
                    idof(t->src[0]), idof(t->src[1]), idof(t->src[2]),
                    idof(t->view_src), (long long) t->view_offs,
                    (unsigned) t->flags, bnum(t->buffer), h);
                mf.push_back(b);
                // resolved operands of the KVarN view: records/stage/indices
                if (t->op == GGML_OP_KVARN_VIEW || t->op == GGML_OP_FLASH_ATTN_EXT) {
                    for (int k = 0; k < 3; ++k) {
                        if (!t->src[k]) { continue; }
                        snprintf(b, sizeof b, "  -> src%d %s buf=%d", k,
                                 shp(t->src[k]).c_str(), bnum(t->src[k]->buffer));
                        mf.push_back(b);
                    }
                }
            }
        }
        const int par = g_run_index & 1, oth = par ^ 1;
        if (g_mf_seen[oth] && g_mf_tag[oth] == g_case) {
            size_t first = SIZE_MAX; size_t shown = 0;
            for (size_t i = 0; i < std::max(mf.size(), g_mf[oth].size()); ++i) {
                const std::string a = i < g_mf[oth].size() ? g_mf[oth][i] : "<missing>";
                const std::string c = i < mf.size() ? mf[i] : "<missing>";
                if (a != c) {
                    if (first == SIZE_MAX) { first = i; }
                    if (shown < 8) {
                        std::printf("  [%s] MANIFEST-DIFF line %zu\n      build%d: %s\n"
                                    "      build%d: %s\n",
                                    g_case.c_str(), i, oth, a.c_str(), par, c.c_str());
                        ++shown;
                    }
                }
            }
            std::printf("  [%s] MANIFEST build%d lines=%zu build%d lines=%zu -> %s "
                        "(first diff line %s)\n", g_case.c_str(),
                        oth, g_mf[oth].size(), par, mf.size(),
                        first == SIZE_MAX ? "IDENTICAL" : "DIFFERS",
                        first == SIZE_MAX ? "-" : "present");
            CHECK(first == SIZE_MAX);
        } else {
            std::printf("  [%s] MANIFEST build%d captured (%zu lines), waiting for pair\n",
                        g_case.c_str(), par, mf.size());
        }
        g_mf[par] = std::move(mf);
        g_mf_tag[par] = g_case;
        g_mf_seen[par] = true;
    }
    // ---- KDISC: hash graph-owned outputs/scratch BEFORE first use ---------
    static int g_kdisc_incomplete = 0;
    static uint32_t g_kdisc_pat = 0;
    static std::vector<uint64_t> g_kdisc_pre[2];
    std::map<ggml_tensor *, int> nidx_edge;
    { const int nn0 = ggml_graph_n_nodes(gf);
      for (int i = 0; i < nn0; ++i) { nidx_edge[ggml_graph_node(gf, i)] = i; } }
    static bool g_kdisc_pre_seen[2] = {false, false};
    static std::string g_kdisc_tag[2];
    if (getenv("F4_RP_KDISC")) {
        auto fnv = [](const uint8_t * b, size_t n) -> uint64_t {
            uint64_t h = 1469598103934665603ull;
            for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
            return h;
        };
        auto grab = [&](const char * n, ggml_tensor * t, uint64_t & h) {
            if (!t || !t->buffer || !t->data) { return; }
            std::vector<uint8_t> tmp(size_t(ggml_nbytes(t)));
            ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size());
            h = fnv(tmp.data(), tmp.size());
            std::printf("  [%s] KDISC-PRE %-9s hash=%016llx bytes=%zu buf=%s\n",
                        g_case.c_str(), n, (unsigned long long) h, tmp.size(),
                        ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)));
        };
        ggml_backend_synchronize(env.cuda);
        ggml_backend_synchronize(env.vk);
        // la_b/la_t are only built on the NON-native branch. The native route
        // uses ggml_kv_tail_attention_merge(), which mutates body_attn in place
        // (fa1 == fa_b) -- so there is no separate body/tail LSE node to read.
        // A missing capture must FAIL the diagnostic as incomplete; printing a
        // zero hash would fake a measurement.
        // la_b/la_t exist only on the NON-native body+tail branch: on the native
        // route ggml_kv_tail_attention_merge() fuses body and tail into fa1
        // (fa1 == fa_b), so there is no separate tensor to observe. Their absence
        // is NOT an incomplete capture -- coverage comes from the kernel record
        // (ggml_cuda_kdisc_rec_count), asserted where that record is read.
        const bool observers_present = (la_b != nullptr && la_t != nullptr);
        uint64_t hb = 0, ht = 0, hm = 0, hlo = 0, ho = 0;
        if (observers_present) { grab("bodyLSE", la_b, hb); grab("tailLSE", la_t, ht); }
        grab("mergeLSE", lse1, hm); grab("localLSE", l1c, hlo); grab("mergedO", oc, ho);
        std::printf("  [%s] KDISC-PRE-SUM body=%016llx%s tail=%016llx%s merge=%016llx "
                    "localLSE=%016llx mergedO=%016llx\n", g_case.c_str(),
                    hb, observers_present ? "" : "(NOT-CAPTURED)",
                    ht, observers_present ? "" : "(NOT-CAPTURED)", hm, hlo, ho);
        // ---- write-coverage poison: two finite patterns, one per build -------
        // If any lane still holds the pattern after the producer ran, that lane
        // was never written yet is consumed downstream.
        const uint32_t pat = (g_run_index & 1) ? 0x40400000u /*3.0f*/ : 0x40000000u /*2.0f*/;
        g_kdisc_pat = pat;
        for (ggml_tensor * t : { lse1, l1c, o1c, oc, lmc }) {
            if (!t || !t->buffer || !t->data) { continue; }
            std::vector<uint8_t> z(size_t(ggml_nbytes(t)));
            for (size_t i = 0; i + 4 <= z.size(); i += 4) { std::memcpy(&z[i], &pat, 4); }
            ggml_backend_tensor_set(t, z.data(), 0, z.size());
        }
        ggml_backend_synchronize(env.cuda);
        std::printf("  [%s] KDISC-POISON outputs with %.1ff before first compute\n",
                    g_case.c_str(), pat == 0x40400000u ? 3.0 : 2.0);
        // ---- follow the REAL src edges from the merge output ----------------
        if (lse1) {
            std::vector<ggml_tensor *> stack = { lse1 };
            std::set<ggml_tensor *> seen;
            for (int depth = 0; depth < 4 && !stack.empty(); ++depth) {
                std::vector<ggml_tensor *> next;
                for (ggml_tensor * t : stack) {
                    if (!t || !seen.insert(t).second) { continue; }
                    for (int k = 0; k < GGML_MAX_SRC; ++k) {
                        ggml_tensor * u = t->src[k];
                        if (!u) { continue; }
                        auto it = nidx_edge.find(u);
                        std::printf("  [%s] EDGE d=%d %s(src%d) -> op=%d name=%.24s ne=[%lld,%lld,%lld] "
                                    "node=%d buf=%s data=%p alias_of=%s\n",
                                    g_case.c_str(), depth,
                                    t->name[0] ? t->name : "?", k, (int) u->op,
                                    u->name[0] ? u->name : "-", (long long) u->ne[0],
                                    (long long) u->ne[1], (long long) u->ne[2],
                                    it == nidx_edge.end() ? -1 : it->second,
                                    u->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(u->buffer)) : "(none)",
                                    u->data,
                                    u->view_src ? "yes" : "no");
                        next.push_back(u);
                    }
                }
                stack = next;
            }
        }
        const int par = g_run_index & 1, oth = par ^ 1;
        std::vector<uint64_t> cur = {hb, ht, hm, hlo, ho};
        if (g_kdisc_pre_seen[oth] && g_kdisc_tag[oth] == g_case) {
            size_t d = 0; for (size_t i = 0; i < 5; ++i) { if (cur[i] != g_kdisc_pre[oth][i]) { ++d; } }
            std::printf("  [%s] KDISC-PRE-CMP build%d vs build%d: %zu of 5 differ -> %s\n",
                        g_case.c_str(), oth, par, d, d ? "SCRATCH/OUTPUT STATE DIFFERS" : "identical");
        }
        g_kdisc_pre[par] = cur; g_kdisc_tag[par] = g_case; g_kdisc_pre_seen[par] = true;
        CHECK(g_kdisc_incomplete == 0);   // a missing capture is an incomplete diagnostic
    }
    // ---- LSE_PTR TARGET AUDIT (post-sched allocation / lifetime) ----------
    // LSE_PTR is a raw pointer to the LSE tensor STRUCT, stashed in the FA
    // node's op_params. That bypasses the src[] edges for the reader's eye, so
    // prove the target really is (a) in the graph, (b) allocated by sched,
    // (c) given non-aliasing device storage, (d) topologically before its
    // consumers -- with actual post-alloc mapping, not graph shape.
    if (getenv("F4_RP_LSEAUDIT")) {
        const int nn = ggml_graph_n_nodes(gf);
        std::map<ggml_tensor *, int> nidx;
        for (int i = 0; i < nn; ++i) { nidx[ggml_graph_node(gf, i)] = i; }
        struct Rng { const void * lo; const char * hi; const char * nm; };
        std::vector<Rng> rng;
        for (int i = 0; i < nn; ++i) {
            const ggml_tensor * t = ggml_graph_node(gf, i);
            if (!t->buffer || !t->data || t->view_src) { continue; }
            rng.push_back({ t->data, (const char *) t->data + ggml_nbytes(t), t->name[0] ? t->name : "?" });
        }
        int fa_seen = 0, lse_ok = 0, alias_hits = 0, xbk = 0;
        for (int i = 0; i < nn; ++i) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            if (t->op != GGML_OP_FLASH_ATTN_EXT) { continue; }
            ++fa_seen;
            ggml_tensor * lse = ggml_flash_attn_ext_get_lse_out(t);
            const bool has = ggml_flash_attn_ext_has_lse_out(t);
            int lse_node = -1;
            auto it = lse ? nidx.find(lse) : nidx.end();
            if (it != nidx.end()) { lse_node = it->second; }
            // Backend ownership is the whole question: a CUDA FA writing LSE
            // through LSE_PTR->data is only safe if that tensor lives on CUDA.
            // Vulkan buffers keep data as an OFFSET (addr==0), which is why the
            // audit printed 0x2830 for one of them.
            const char * fa_buf = t->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)) : "(none)";
            const char * lse_bn = (lse && lse->buffer) ? ggml_backend_buft_name(ggml_backend_buffer_get_type(lse->buffer)) : "(none)";
            std::printf("  [%s] LSE-AUDIT fa_node=%d FAbuf=%-9s LSE_OUT=%d lse_op=%d "
                        "lse_ne=[%lld,%lld,%lld] lsebuf=%-9s lse_data=%p node=%d "
                        "nbytes=%zu\n",
                        g_case.c_str(), i, fa_buf, (int) has,
                        lse ? (int) lse->op : -1,
                        lse ? (long long) lse->ne[0] : -1,
                        lse ? (long long) lse->ne[1] : -1,
                        lse ? (long long) lse->ne[2] : -1,
                        lse_bn, lse ? lse->data : nullptr, lse_node,
                        lse ? size_t(ggml_nbytes(lse)) : 0);
            // Cross-backend hidden output: the FA's own buffer must own the LSE.
            if (lse && lse->buffer && t->buffer && lse->buffer != t->buffer) {
                std::printf("  [%s] LSE-AUDIT CROSS-BACKEND fa_node=%d FA on %s but LSE on %s "
                            "(LSE_PTR bypasses src edges, so sched may place it elsewhere)\n",
                            g_case.c_str(), i, fa_buf, lse_bn);
                ++xbk;
            }
            CHECK(has);                                   // flag must be set
            CHECK(lse != nullptr);                        // pointer resolves
            CHECK(lse->op == GGML_OP_FLASH_ATTN_EXT_LSE); // right object type
            CHECK(lse_node >= 0);                         // IN the graph -> sched tracks it
            CHECK(lse_node < i);                          // written before the FA it belongs to? (topo)
            CHECK(lse->buffer != nullptr);                // allocated by sched
            CHECK(lse->data != nullptr);                  // has device storage
            if (lse->data) {
                const char * b = (const char *) lse->data;
                const char * e = b + ggml_nbytes(lse);
                for (const auto & r : rng) {
                    if (r.nm == (lse->name[0] ? lse->name : "?")) { continue; }
                    if (b < r.hi && r.lo < e) {
                        std::printf("  [%s] LSE-AUDIT ALIAS lse overlaps node '%s'\n",
                                    g_case.c_str(), r.nm);
                        ++alias_hits;
                    }
                }
                ++lse_ok;
            }
        }
        std::printf("  [%s] LSE-AUDIT summary fa_nodes=%d resolved_with_storage=%d "
                    "alias_hits=%d graph_nodes=%d cross_backend=%d\n",
                    g_case.c_str(), fa_seen, lse_ok, alias_hits, nn, xbk);
        CHECK(alias_hits == 0);
        CHECK(xbk == 0);   // hidden output must live on the FA's own backend
        // consumers must be topologically after the LSE node
        for (int i = 0; i < nn; ++i) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            if (t->op != GGML_OP_FLASH_ATTN_EXT) { continue; }
            ggml_tensor * lse = ggml_flash_attn_ext_get_lse_out(t);
            if (!lse) { continue; }
            for (int j = 0; j < nn; ++j) {
                ggml_tensor * u = ggml_graph_node(gf, j);
                for (int k = 0; k < GGML_MAX_SRC; ++k) {
                    if (u->src[k] == lse) {
                        std::printf("  [%s] LSE-AUDIT consumer node=%d (op=%d src=%d) reads lse "
                                    "produced at node=%d -> %s\n", g_case.c_str(), j, (int) u->op, k,
                                    [&]{ auto z = nidx.find(lse); return z == nidx.end() ? -1 : z->second; }(),
                                    (j > 0) ? "ordered-after-or-same" : "SUSPECT");
                    }
                }
            }
        }
    }
    if (getenv("F4_RP_OPHASH")) {
        auto fnv = [](const uint8_t * b, size_t n) -> uint64_t {
            uint64_t h = 1469598103934665603ull;
            for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
            return h;
        };
        auto op = [&](const char * n, ggml_tensor * t) {
            if (!t || !t->buffer) {
                std::printf("  [%s] OPH %-14s UNALLOCATED\n", g_case.c_str(), n); return;
            }
            std::vector<uint8_t> tmp(size_t(ggml_nbytes(t)));
            ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size());
            std::printf("  [%s] OPH %-14s %016llx %zu\n", g_case.c_str(), n,
                        (unsigned long long) fnv(tmp.data(), tmp.size()), tmp.size());
        };
        op("q_t", q_t); op("kc", kc); op("vc", vc); op("idx1", idx1);
        op("mask1", mask1); op("mask1b", mask1b); op("mask2", mask2);
        op("k2_t", k2_t); op("v2_t", v2_t);
        op("kt_nat", kt_nat); op("vt_nat", vt_nat);
        op("tail_mask", tail_mask_nat); op("tail_qorder", tail_qorder);
        op("tail_rundesc", tail_rundesc);
        std::printf("  [%s] OPH host m1=%016llx m2=%016llx desc=%016llx (%zu words) "
                    "tail_qorder_bytes=%zu desc_stride_words=%zu\n",
                    g_case.c_str(),
                    (unsigned long long) fnv((const uint8_t *) m1.data(), m1.size() * 4),
                    (unsigned long long) fnv((const uint8_t *) m2.data(), m2.size() * 4),
                    (unsigned long long) fnv((const uint8_t *) desc_snapshot.data(),
                                             desc_snapshot.size() * 4),
                    desc_snapshot.size(), size_t(ggml_nbytes(tail_qorder)),
                    desc_snapshot.size() ? desc_snapshot.size() : 0);
    }
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        std::printf("  [%s] compute FAILED\n", g_case.c_str());
        ++g_checks; ++g_failures; ++g_skipped;
        ggml_backend_sched_free(sched); ggml_free(gctx);
        ggml_free(dctx_c); ggml_free(dctx_v);
        ggml_backend_buffer_free(bcuda); ggml_backend_buffer_free(bvk);
        return;
    }

    if (getenv("F4_RP_KDISC")) {
        // (a) baseline read immediately after compute (what the fixture always
        //     does), then (b) an explicit cross-backend barrier, then read again.
        // Topological index order proves dependency STRUCTURE; only comparing
        // (a) with (b) says anything about observed stream completion.
        auto fnv2 = [](const uint8_t * b, size_t n) -> uint64_t {
            uint64_t h = 1469598103934665603ull;
            for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
            return h;
        };
        auto rd = [&](const char * n, ggml_tensor * t, uint64_t & h, float & v0) {
            h = 0; v0 = 0.0f;
            if (!t || !t->buffer || !t->data) { return; }
            std::vector<uint8_t> tmp(size_t(ggml_nbytes(t)));
            ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size());
            h = fnv2(tmp.data(), tmp.size());
            if (tmp.size() >= 4) { std::memcpy(&v0, tmp.data(), 4); }
            std::printf("  [%s] KDISC-POST %-9s hash=%016llx v0=%.9g\n",
                        g_case.c_str(), n, (unsigned long long) h, v0);
        };
        // Body/tail actuals come from the diagnostic observers: the native route has
        // no separate body/tail tensor (fa1 == fa_b, merge is inside the kernel).
        static const char * nm7[7] = {"bodyO","bodyLSE","tailO","tailLSE",
                                      "mergeLSE","localLSE","mergedO"};
        ggml_tensor * ts7[7] = {kdisc_bo, kdisc_bl, kdisc_to, kdisc_tl, lse1, l1c, oc};
        uint64_t h7[7] = {0,0,0,0,0,0,0}; float v7[7] = {0,0,0,0,0,0,0};
        for (int i = 0; i < 7; ++i) { rd(nm7[i], ts7[i], h7[i], v7[i]); }
        if (!kdisc_bo || !kdisc_bl || !kdisc_to || !kdisc_tl) {
            std::printf("  [%s] KDISC-INCOMPLETE body/tail observer missing -> body/tail "
                        "capture INCOMPLETE\n", g_case.c_str());
            ++g_kdisc_incomplete;
        }
        ggml_backend_synchronize(env.cuda);
        ggml_backend_synchronize(env.vk);
        uint64_t p7[7] = {0,0,0,0,0,0,0}; float q7[7] = {0,0,0,0,0,0,0};
        for (int i = 0; i < 7; ++i) { rd(nm7[i], ts7[i], p7[i], q7[i]); }
        // The tail merge kernel must have actually produced a diagnostic record;
        // missing capture is INCOMPLETE, never a numerical pass.
        {
            const int nrec = ggml_cuda_kdisc_rec_count();
            std::printf("  [%s] KDISC-REC-COUNT %d -> %s\n", g_case.c_str(), nrec,
                        nrec > 0 ? "tail-merge record captured" : "MISSING -> INCOMPLETE");
            if (nrec <= 0) { ++g_kdisc_incomplete; }
        }
        // ---- descriptor actually sitting in VRAM vs the host plan ----------
        if (tail_rundesc && tail_rundesc->buffer && !desc_snapshot.empty()) {
            std::vector<int32_t> dev(size_t(ggml_nelements(tail_rundesc)), 0);
            ggml_backend_tensor_get(tail_rundesc, dev.data(), 0, ggml_nbytes(tail_rundesc));
            size_t nd = std::min(dev.size(), desc_snapshot.size());
            size_t diff = 0;
            for (size_t i = 0; i < nd; ++i) { if (dev[i] != desc_snapshot[i]) { ++diff; } }
            const int nt = dev.size() > 4 ? dev[4] : -1;
            const int nb = dev.size() > 5 ? dev[5] : -1;
            const int body_off = 6 + (nt > 0 ? nt : 0);
            std::string ring;
            for (int i = 0; i < nt && 6 + i < (int) dev.size(); ++i) {
                ring += std::to_string(dev[6 + i]) + (i + 1 < nt ? "," : "");
            }
            // query_order actually resident in VRAM (ring order for the tail pass)
            std::string qord;
            if (tail_qorder && tail_qorder->buffer) {
                std::vector<int32_t> qo(size_t(ggml_nelements(tail_qorder)), -99);
                ggml_backend_tensor_get(tail_qorder, qo.data(), 0, ggml_nbytes(tail_qorder));
                for (size_t i = 0; i < qo.size() && i < 8; ++i) {
                    qord += std::to_string(qo[i]) + (i + 1 < qo.size() && i < 7 ? "," : "");
                }
            }
            std::printf("  [%s] KDISC-DESC device(%zu) vs host_plan(%zu) differing_words=%zu "
                        "n_tail=%d n_body=%d ring_slots=[%s] query_order=[%s] "
                        "body_map_off=%d packed_body=%d\n",
                        g_case.c_str(), dev.size(), desc_snapshot.size(), diff,
                        nt, nb, ring.c_str(), qord.c_str(), body_off,
                        (int) (dev.size() > (size_t) body_off));
            CHECK(diff == 0);
            CHECK(nd == desc_snapshot.size());
            CHECK(nt > 0);
        } else {
            std::printf("  [%s] KDISC-INCOMPLETE tail_rundesc/host plan unavailable -> "
                        "descriptor check INCOMPLETE\n", g_case.c_str());
            ++g_kdisc_incomplete;
        }
        // write coverage: no lane may still hold the pre-compute poison
        {
            size_t untouched = 0, total = 0;
            for (ggml_tensor * t : { lse1, l1c, o1c, oc, lmc }) {
                if (!t || !t->buffer || !t->data) { continue; }
                std::vector<uint8_t> tmp(size_t(ggml_nbytes(t)));
                ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size());
                for (size_t i = 0; i + 4 <= tmp.size(); i += 4) {
                    uint32_t v = 0; std::memcpy(&v, &tmp.data()[i], 4);
                    ++total; if (v == g_kdisc_pat) { ++untouched; }
                }
            }
            std::printf("  [%s] KDISC-COVERAGE poisoned-lanes-still-poisoned=%zu/%zu -> %s\n",
                        g_case.c_str(), untouched, total,
                        untouched ? "UNTOUCHED LANE CONSUMED DOWNSTREAM" : "all lanes rewritten");
            CHECK(untouched == 0);
        }
        // ---- earliest-bad-producer: hash the ACTUAL parents of the merge ----
        // node_9 (op=74) IS the merge (fa1==fa_b), so its src[] are the real
        // producers: rotated Q, K/V views (aliases -> resolve view_src), mask,
        // rotated tail K/V (WHT) and the tail mask. Values, not just identity.
        {
            static std::vector<std::pair<std::string, uint64_t> > g_par[2];
            static bool g_par_seen[2] = {false, false};
            static std::string g_par_tag[2];
            std::vector<std::pair<std::string, uint64_t> > cur;
            ggml_tensor * fa = (lse1 && lse1->src[0]) ? lse1->src[0] : nullptr;
            if (!fa) {
                std::printf("  [%s] KDISC-INCOMPLETE merge parent fa1 is NULL -> cannot "
                            "locate producer\n", g_case.c_str());
                ++g_kdisc_incomplete;
            }
            for (int k = 0; k < GGML_MAX_SRC && fa; ++k) {
                ggml_tensor * u = fa->src[k];
                if (!u) { continue; }
                ggml_tensor * r = u->view_src ? u->view_src : u;
                char lab[48];
                snprintf(lab, sizeof lab, "fa1.src%d/op%d%s", k, (int) u->op,
                         u->view_src ? "+view" : "");
                uint64_t hv = 0;
                if (r && r->buffer && r->data) {
                    std::vector<uint8_t> tmp(size_t(ggml_nbytes(r)));
                    ggml_backend_tensor_get(r, tmp.data(), 0, tmp.size());
                    uint64_t h = 1469598103934665603ull;
                    for (size_t i = 0; i < tmp.size(); ++i) { h ^= tmp[i]; h *= 1099511628211ull; }
                    hv = h;
                    std::printf("  [%s] KDISC-PARENT %-18s hash=%016llx bytes=%zu buf=%s\n",
                                g_case.c_str(), lab, (unsigned long long) h, tmp.size(),
                                r->buffer ? ggml_backend_buft_name(
                                    ggml_backend_buffer_get_type(r->buffer)) : "(none)");
                } else {
                    // KVARN_VIEW parents are virtual by design (data=(nil), resolved
                    // inside the kernel from records/stage) -- expected, not missing.
                    std::printf("  [%s] KDISC-PARENT %-18s VIRTUAL (kvarn view)\n",
                                g_case.c_str(), lab);
                }
                cur.push_back({lab, hv});
            }
            const int par = g_run_index & 1, oth = par ^ 1;
            if (g_par_seen[oth] && g_par_tag[oth] == g_case) {
                size_t n = std::min(cur.size(), g_par[oth].size());
                bool first = true;
                for (size_t i = 0; i < n; ++i) {
                    if (cur[i].second != g_par[oth][i].second) {
                        std::printf("  [%s] KDISC-EARLIEST-BAD build%d vs build%d: PARENT %s "
                                    "a=%016llx b=%016llx\n", g_case.c_str(), oth, par,
                                    cur[i].first.c_str(),
                                    (unsigned long long) g_par[oth][i].second,
                                    (unsigned long long) cur[i].second);
                        first = false;
                    }
                }
                if (first) {
                    std::printf("  [%s] KDISC-EARLIEST-BAD: all %zu parents identical "
                                "-> first divergence is INSIDE the merge kernel\n",
                                g_case.c_str(), n);
                }
            }
            g_par[par] = cur; g_par_tag[par] = g_case; g_par_seen[par] = true;
        }
        int uns = 0; for (int i = 0; i < 7; ++i) { if (h7[i] != p7[i]) { ++uns; } }
        std::printf("  [%s] KDISC-SYNC baseline-vs-barrier differ=%d -> %s\n",
                    g_case.c_str(), uns,
                    uns ? "STREAM COMPLETION ORDERING SENSITIVE" : "stable under barrier");
        static std::vector<uint64_t> g_post[2]; static bool g_seen[2] = {false,false};
        static std::string g_tag[2];
        const int par = g_run_index & 1, oth = par ^ 1;
        std::vector<uint64_t> cur(h7, h7 + 7);
        if (g_seen[oth] && g_tag[oth] == g_case) {
            size_t nd = 0;
            for (size_t i = 0; i < cur.size() && i < g_post[oth].size(); ++i) {
                if (cur[i] != g_post[oth][i]) {
                    if (nd == 0) {
                        std::printf("  [%s] KDISC-FIRST-DIVERGENT build%d vs build%d: %s "
                                    "a=%016llx b=%016llx (order: bodyO bodyLSE tailO tailLSE "
                                    "mergeLSE localLSE mergedO)\n", g_case.c_str(), oth, par,
                                    nm7[i], (unsigned long long) g_post[oth][i],
                                    (unsigned long long) cur[i]);
                    } else {
                        std::printf("  [%s] KDISC-DIVERGENT-ALSO %s\n", g_case.c_str(), nm7[i]);
                    }
                    ++nd;
                }
            }
            if (nd == 0) {
                std::printf("  [%s] KDISC-FIRST-DIVERGENT: bodyO/bodyLSE/tailO/tailLSE/"
                            "mergeLSE/localLSE/mergedO all IDENTICAL between builds\n",
                            g_case.c_str());
            }
        }
        g_post[par] = cur; g_tag[par] = g_case; g_seen[par] = true;
    }
    std::printf("  [%s] graph build: nodes=%d fa1_op_params=[%d,%d,%d,%d,%d,%d,%d,%d] "
                "fa1=%p oc=%p obs=%d\n", g_case.c_str(), ggml_graph_n_nodes(gf),
                fa1->op_params[0], fa1->op_params[1], fa1->op_params[2], fa1->op_params[3],
                fa1->op_params[4], fa1->op_params[5], fa1->op_params[6], fa1->op_params[7],
                (void *) fa1, (void *) oc, obs_body_fa ? 1 : 0);
    std::vector<float> h1, h1l, h2, h2l, ho, hl;
    if (!readback(o1c, h1) || !readback(l1c, h1l) || !readback(o2c, h2) ||
        !readback(l2c, h2l) || !readback(oc, ho) || !readback(lmc, hl)) {
        std::printf("  [%s] readback incomplete\n", g_case.c_str());
        ++g_checks; ++g_failures; ++g_skipped;
        ggml_backend_sched_free(sched); ggml_free(gctx);
        ggml_free(dctx_c); ggml_free(dctx_v);
        ggml_backend_buffer_free(bcuda); ggml_backend_buffer_free(bvk);
        return;
    }

    std::vector<float> h_body, h_body_l;
    if (obc && lbcc) {
        if (!readback(obc, h_body) || !readback(lbcc, h_body_l)) {
            h_body.clear(); h_body_l.clear();
        }
    }

    // ---- repeated compute of the SAME graph -------------------------------
    // Separates kernel/runtime nondeterminism from graph construction: the
    // inputs, descriptors and allocation are frozen here, so any difference is
    // produced by the backends themselves on an identical graph.
    if (const char * rp = getenv("F4_RP_REPEAT"); rp && *rp) {
        const int reps = std::atoi(rp);
        for (int i = 1; i <= reps; ++i) {
            if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
                std::printf("  [%s] repeat compute %d FAILED\n", g_case.c_str(), i);
                ++g_checks; ++g_failures;
                break;
            }
            std::vector<float> ro, rl;
            if (!readback(oc, ro) || !readback(lmc, rl)) { break; }
            auto bits = [](const std::vector<float> & a, const std::vector<float> & b) {
                if (a.size() != b.size()) { return false; }
                for (size_t i = 0; i < a.size(); ++i) {
                    uint32_t x = 0, y = 0;
                    std::memcpy(&x, &a[i], 4); std::memcpy(&y, &b[i], 4);
                    if (x != y) { return false; }
                }
                return true;
            };
            double mx = 0.0;
            for (size_t i = 0; i < ro.size() && i < ho.size(); ++i) {
                mx = std::max(mx, std::fabs(double(ro[i]) - double(ho[i])));
            }
            std::printf("  [%s] SAME-GRAPH repeat %d: O bit-identical=%d max|dO|=%.9g "
                        "LSE bit-identical=%d\n", g_case.c_str(), i,
                        (int) bits(ro, ho), mx, (int) bits(rl, hl));
        }
    }

    // ---- SAME-LAYOUT NEIGHBOR CANARY ---------------------------------------
    // Changes ONLY bytes of a region that is not a valid semantic input, inside
    // the SAME graph, SAME bcuda allocation, SAME upload buffers. Every valid
    // operand and descriptor is hashed before and after and must come back
    // identical; the canary target must actually change. Run it in both the
    // green (base) and the red (4 MiB) placement: if the red placement's output
    // reacts while the green one's does not, the read crosses into that region
    // only when the tensors are laid out that way.
    //   F4_RP_CANARY=rec   k_rec / v_rec. For the smallest native case the
    //                      INIT-PLAN prints rec_reads=0 (no record cell is
    //                      consumed), so these bytes are not semantic input --
    //                      and k_rec sits directly after v_stg, which IS read.
    //   F4_RP_CANARY=stage whole stage leaves (already characterised as
    //                      poison-sensitive; kept as a positive control).
    // F4_RP_GEOM=1 additionally logs operand addresses and dispatch geometry.
    if (const char * cnd = getenv("F4_RP_CANARY"); cnd && *cnd) {
        auto fnv = [](const uint8_t * b, size_t n) -> uint64_t {
            uint64_t h = 1469598103934665603ull;
            for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
            return h;
        };
        auto hash_t = [&](const char * name, ggml_tensor * t) -> uint64_t {
            if (!t || !t->buffer) { return 0; }
            std::vector<uint8_t> tmp(size_t(ggml_nbytes(t)));
            ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size());
            const uint64_t h = fnv(tmp.data(), tmp.size());
            std::printf("  [%s] CANARY-IN %-14s hash=%016llx bytes=%zu\n",
                        g_case.c_str(), name, (unsigned long long) h, tmp.size());
            return h;
        };
        struct Op { const char * n; ggml_tensor * t; };
        const Op valid[] = {
            { "q_t", q_t }, { "kc", kc }, { "vc", vc }, { "idx1", idx1 },
            { "mask1", mask1 }, { "mask1b", mask1b }, { "mask2", mask2 },
            { "k2_t", k2_t }, { "v2_t", v2_t },
            { "kt_nat", kt_nat }, { "vt_nat", vt_nat },
            { "tail_mask", tail_mask_nat }, { "tail_qorder", tail_qorder },
            { "tail_rundesc", tail_rundesc },
        };
        const Op targets[] = { { "k_rec", k_rec }, { "v_rec", v_rec },
                               { "k_stg", k_stg }, { "v_stg", v_stg } };
        std::map<std::string, uint64_t> before, before_tgt;
        for (const auto & o : valid) { before[o.n] = hash_t(o.n, o.t); }
        for (const auto & o : targets) { before_tgt[o.n] = hash_t(o.n, o.t); }
        {
            uint64_t dh = 1469598103934665603ull;
            for (int32_t v : desc_snapshot) { dh ^= uint32_t(v); dh *= 1099511628211ull; }
            std::printf("  [%s] CANARY-IN run_desc hash=%016llx (%zu words) "
                        "k_rec_ne2=%lld v_rec_ne2=%lld stage_cells=%lld\n",
                        g_case.c_str(), (unsigned long long) dh, desc_snapshot.size(),
                        (long long) k_rec->ne[2], (long long) v_rec->ne[2],
                        (long long) k_stg->ne[2]);
            before["run_desc"] = dh;
        }
        // snapshot of every level we can read without changing the graph
        const std::vector<float> b0 = h_body,  bl0 = h_body_l;
        const std::vector<float> l0 = h1,     ll0 = h1l;
        const std::vector<float> m0 = ho,     ml0 = hl;
        // write the canary (two distinct patterns, so K and V cannot mask)
        const bool stage_target = std::string(cnd) == "stage";
        const uint8_t pk = stage_target ? 0x3C : 0xA5;
        const uint8_t pv = stage_target ? 0xC3 : 0x5A;
        for (int ti = 0; ti < 2; ++ti) {
            ggml_tensor * t = stage_target ? (ti ? v_stg : k_stg) : (ti ? v_rec : k_rec);
            const uint8_t pat = ti ? pv : pk;
            std::vector<uint8_t> z(size_t(ggml_nbytes(t)), pat);
            ggml_backend_tensor_set(t, z.data(), 0, z.size());
        }
        ggml_backend_synchronize(env.cuda);
        std::printf("  [%s] CANARY applied target=%s (k=0x%02X v=0x%02X)\n",
                    g_case.c_str(), cnd, pk, pv);
        for (const auto & o : targets) {
            if ((std::string(cnd) == "stage") == (std::string(o.n).find("stg") != std::string::npos)) {
                continue;
            }
            before_tgt[o.n] = hash_t(o.n, o.t);   // re-read: this one MUST differ
        }
        // replay the SAME graph, same allocation
        const int rc2 = ggml_backend_sched_graph_compute(sched, gf);
        ggml_backend_synchronize(env.cuda);
        std::vector<float> b1, bl1, l1, ll1, m1n, ml1;
        if (rc2 == GGML_STATUS_SUCCESS) {
            if (obc && !readback(obc, b1)) { b1.clear(); }
            if (lbcc && !readback(lbcc, bl1)) { bl1.clear(); }
            readback(o1c, l1); readback(l1c, ll1);
            readback(oc, m1n); readback(lmc, ml1);
        }
        auto cmp = [&](const char * name, const std::vector<float> & a,
                       const std::vector<float> & b) {
            if (a.empty() && b.empty()) {
                std::printf("  [%s] CANARY-OUT %-9s not populated (body levels need "
                            "F4_RP_OBS=1)\n", g_case.c_str(), name); return;
            }
            size_t n = std::min(a.size(), b.size());
            double mx = 0.0; size_t fi = SIZE_MAX;
            for (size_t i = 0; i < n; ++i) {
                const double d = std::fabs(double(a[i]) - double(b[i]));
                if (d > mx) { mx = d; }
                if (d > 0.0 && fi == SIZE_MAX) { fi = i; }
            }
            std::printf("  [%s] CANARY-OUT %-9s %s max|d|=%.9g first=%s",
                        g_case.c_str(), name,
                        (a.size() == b.size() && mx == 0.0) ? "IDENTICAL" : "DIFFERS",
                        mx, fi == SIZE_MAX ? "-" : "yes");
            if (fi != SIZE_MAX && fi < n) {
                std::printf(" i=%zu got=%.9g ref=%.9g", fi, b[fi], a[fi]);
            }
            std::printf(" n=%zu/%zu\n", a.size(), b.size());
        };
        cmp("bodyO", b0, b1); cmp("bodyLSE", bl0, bl1);
        cmp("localO", l0, l1); cmp("localLSE", ll0, ll1);
        cmp("mergedO", m0, m1n); cmp("mergedLSE", ml0, ml1);
        bool in_ok = true, tgt_changed = false;
        for (const auto & o : valid) {
            const uint64_t h = hash_t(o.n, o.t);
            if (h != before[o.n]) { in_ok = false;
                std::printf("  [%s] CANARY-PROOF %s CHANGED (valid operand!)\\n",
                            g_case.c_str(), o.n); }
        }
        {
            uint64_t dh = 1469598103934665603ull;
            for (int32_t v : desc_snapshot) { dh ^= uint32_t(v); dh *= 1099511628211ull; }
            if (dh != before["run_desc"]) { in_ok = false;
                std::printf("  [%s] CANARY-PROOF run_desc CHANGED (descriptor!)\\n",
                            g_case.c_str()); }
        }
        for (const auto & o : targets) {
            const bool is_target = (std::string(cnd) == "stage") ==
                                   (std::string(o.n).find("stg") != std::string::npos);
            if (!is_target) { continue; }
            const uint64_t h = hash_t(o.n, o.t);
            if (h != before_tgt[o.n]) { tgt_changed = true; }
        }
        std::printf("  [%s] CANARY-PROOF valid_operands_and_desc_unchanged=%d "
                    "canary_target_actually_changed=%d\\n",
                    g_case.c_str(), (int) in_ok, (int) tgt_changed);
        CHECK(in_ok);
        if (getenv("F4_RP_CANARY_REQUIRE")) { CHECK(tgt_changed); }
    }
    if (getenv("F4_RP_GEOM")) {
        auto pt = [&](const char * n, ggml_tensor * t) {
            if (!t) { return; }
            std::printf("  [%s] GEOM %-10s op=%d data=%p ne=[%lld,%lld,%lld,%lld] "
                        "nb=[%zu,%zu,%zu,%zu] buf=%p\n", g_case.c_str(), n, (int) t->op,
                        (const void *) t->data,
                        (long long) t->ne[0], (long long) t->ne[1],
                        (long long) t->ne[2], (long long) t->ne[3],
                        (size_t) t->nb[0], (size_t) t->nb[1],
                        (size_t) t->nb[2], (size_t) t->nb[3], (void *) t->buffer);
        };
        pt("k_view", k_view); pt("v_view", v_view);
        pt("s_k", s_k); pt("s_v", s_v);
        pt("k_stg", k_stg); pt("v_stg", v_stg);
        pt("k_rec", k_rec); pt("v_rec", v_rec);
        pt("kt_nat", kt_nat); pt("vt_nat", vt_nat);
        pt("bcuda-k", k_stg);
    }

    // ---- KVarN oracle from the device-produced records/stage -------------
    std::vector<uint8_t> krec_h(size_t(ggml_nbytes(k_rec))), vrec_h(size_t(ggml_nbytes(v_rec)));
    std::vector<ggml_fp16_t> kstg_h(size_t(ggml_nelements(k_stg))), vstg_h(size_t(ggml_nelements(v_stg)));
    ggml_backend_tensor_get(k_rec, krec_h.data(), 0, ggml_nbytes(k_rec));
    ggml_backend_tensor_get(v_rec, vrec_h.data(), 0, ggml_nbytes(v_rec));
    ggml_backend_tensor_get(k_stg, kstg_h.data(), 0, ggml_nbytes(k_stg));
    ggml_backend_tensor_get(v_stg, vstg_h.data(), 0, ggml_nbytes(v_stg));

    // [128, group_rows, kv] -> head/token major [D, kv, n_kvh]
    auto to_head_rows = [&](const std::vector<float> & in, std::vector<float> & out) {
        out.assign(size_t(D) * size_t(kv1) * n_kvh, 0.0f);
        for (int p = 0; p < kv1; ++p)
            for (int kh = 0; kh < n_kvh; ++kh)
                for (int s = 0; s < head_slices; ++s)
                    for (int d = 0; d < KVAR_N_GROUP; ++d) {
                        const int gr = kh * head_slices + s;
                        out[(size_t(p) + size_t(kv1)*kh)*D + s*KVAR_N_GROUP + d] =
                            in[(size_t(p)*group_rows + gr)*KVAR_N_GROUP + d];
                    }
    };
    std::vector<float> k_raw, v_raw, k_derot, k1h, v1h, k1h_derot;
    if (!kvarn_rows_from_device(env, krec_h, kstg_h, k_rb, group_rows, kv1, bits, false,
                                stage_groups, head_slices, true, k_raw) ||
        !kvarn_rows_from_device(env, vrec_h, vstg_h, v_rb, group_rows, kv1, bits, true,
                                stage_groups, head_slices, false, v_raw) ||
        !kvarn_rows_from_device(env, krec_h, kstg_h, k_rb, group_rows, kv1, bits, false,
                                stage_groups, head_slices, false, k_derot)) {
        std::printf("  [%s] kvarn materialize-from-device failed\n", g_case.c_str());
        ++g_checks; ++g_failures;
        ggml_backend_sched_free(sched); ggml_free(gctx);
        ggml_free(dctx_c); ggml_free(dctx_v);
        ggml_backend_buffer_free(bcuda); ggml_backend_buffer_free(bvk);
        return;
    }
    to_head_rows(k_raw, k1h);      // rotated K: consumed by ROTATED FA
    to_head_rows(v_raw, v1h);      // un-rotated V: produces O
    to_head_rows(k_derot, k1h_derot);
    // a CPU/device store divergence must not pass silently
    {
        double dk = 0.0, dv = 0.0, rk = 0.0, rv = 0.0, uk = 0.0, uv = 0.0;
        size_t nk = 0, nv = 0;
        auto ulp = [](float x) { float a = std::fabs(x); return a > 0.0f ? std::nextafter(a, 1e30f) - a : 1e-7f; };
        for (size_t i = 0; i < k1h_derot.size(); ++i) {
            double d = std::fabs(double(k1h_derot[i]) - k1h_cpu[i]);
            if (d > 0.0) ++nk;
            dk = std::max(dk, d);
            rk = std::max(rk, d / std::max(1e-6, std::fabs(double(k1h_derot[i]))));
            uk = std::max(uk, d / double(ulp(k1h_derot[i])));
        }
        for (size_t i = 0; i < v1h.size(); ++i) {
            double d = std::fabs(double(v1h[i]) - v1h_cpu[i]);
            if (d > 0.0) ++nv;
            dv = std::max(dv, d);
            rv = std::max(rv, d / std::max(1e-6, std::fabs(double(v1h[i]))));
            uv = std::max(uv, d / double(ulp(v1h[i])));
        }
        std::printf("  [%s] cpu-vs-device store rows: maxdiffK=%.3e maxdiffV=%.3e | "
                    "relK=%.3e relV=%.3e | nzK=%zu/%zu nzV=%zu/%zu\n",
                    g_case.c_str(), dk, dv, rk, rv,
                    nk, k1h_derot.size(), nv, v1h.size());
        {
            size_t ak = 0, av = 0; double bk = 0.0, bv = 0.0;
            for (size_t i = 0; i < k1h_derot.size(); ++i) {
                double d = std::fabs(double(k1h_derot[i]) - k1h_cpu[i]);
                if (d > bk) { bk = d; ak = i; }
            }
            for (size_t i = 0; i < v1h.size(); ++i) {
                double d = std::fabs(double(v1h[i]) - v1h_cpu[i]);
                if (d > bv) { bv = d; av = i; }
            }
            std::printf("  [%s] cpu-vs-device worst K: i=%zu device=%g cpu=%g | worst V: i=%zu device=%g cpu=%g\n",
                        g_case.c_str(), ak, k1h_derot[ak], k1h_cpu[ak],
                        av, v1h[av], v1h_cpu[av]);
        }
        size_t live_stage_k = 1, live_stage_v = 1;   // 1 = "not measured"
        {
            // Raw record / stage bytes, CPU store vs device store. This is the
            // real question: do the two stores write the same bytes?
            auto diffb = [](const void * a, const void * b, size_t n) -> size_t {
                const uint8_t * x = (const uint8_t *) a, * y = (const uint8_t *) b;
                size_t c = 0;
                for (size_t i = 0; i < n; ++i) if (x[i] != y[i]) ++c;
                return c;
            };
            // stage is [KVAR_N_GROUP, group_rows, 128*stage_groups*n_stream]; only
            // cells c < kv1 hold live data, the rest are past-the-end ring slots.
            auto diff_stage = [&](const void * a, const void * b) {
                const uint8_t * x = (const uint8_t *) a, * y = (const uint8_t *) b;
                const size_t nc = size_t(kstg_h.size()) / (size_t(KVAR_N_GROUP) * size_t(group_rows));
                size_t live = 0, dead = 0, livebytes = 0, deadbytes = 0;
                for (size_t c = 0; c < nc; ++c) {
                    const size_t lo = c * size_t(KVAR_N_GROUP) * size_t(group_rows) * 2;
                    const size_t hi = lo + size_t(KVAR_N_GROUP) * size_t(group_rows) * 2;
                    size_t d = 0;
                    for (size_t i = lo; i < hi; ++i) if (x[i] != y[i]) ++d;
                    if (c < size_t(kv1)) { live += d; livebytes += d; } else { dead += d; deadbytes += d; }
                }
                std::printf("  [%s] stage bytes differ: LIVE(cells<kv1)=%zu/%zu  DEAD=%zu/%zu\n",
                            g_case.c_str(), live, livebytes,
                            dead, (nc - size_t(kv1)) * size_t(KVAR_N_GROUP) * size_t(group_rows) * 2);
                return live;
            };
            if (g_cpu_rec_raw[0].size() == krec_h.size()) {
                size_t cr = diffb(g_cpu_rec_raw[0].data(), krec_h.data(), krec_h.size());
                size_t cs = diffb(g_cpu_stg_raw[0].data(), kstg_h.data(),
                                  kstg_h.size() * sizeof(ggml_fp16_t));
                std::printf("  [%s] RAW K records: %zu/%zu bytes differ | stage total %zu\n",
                            g_case.c_str(), cr, krec_h.size(), cs);
                live_stage_k = diff_stage(g_cpu_stg_raw[0].data(), kstg_h.data());
            }
            if (g_cpu_rec_raw[1].size() == vrec_h.size()) {
                size_t cr = diffb(g_cpu_rec_raw[1].data(), vrec_h.data(), vrec_h.size());
                size_t cs = diffb(g_cpu_stg_raw[1].data(), vstg_h.data(),
                                  vstg_h.size() * sizeof(ggml_fp16_t));
                std::printf("  [%s] RAW V records: %zu/%zu bytes differ | stage total %zu\n",
                            g_case.c_str(), cr, vrec_h.size(), cs);
                live_stage_v = diff_stage(g_cpu_stg_raw[1].data(), vstg_h.data());
            }
        }
        {
            // CPU/CUDA KVARN STORE DIAGNOSTIC -- separated from the canonical
            // O/LSE acceptance gates on purpose.
            //
            // Expected quantizer arithmetic. The record is a b-bit uniform
            // quantization of a 128-wide group (src/llama-kvarn.cpp host path,
            // ggml/src/ggml-cuda/kvarn.cu device path): scale/zero-point are
            // computed by an independent Sinkhorn loop per implementation, in
            // fp32 on the host and in fp32/fp16 on the device, and the packed
            // bits plus the fp16 scales are stored separately. Two correct
            // implementations therefore disagree by up to ONE quantization
            // step  s = (max-min)/(2^b - 1)  of the row, plus fp16 rounding of
            // the scale/zp axes. Bit-equality (the old 1e-6 bound) only ever
            // held for the DELAYED seal and is not a property of eager sealing.
            //
            // Gated on three things that a real divergence cannot satisfy:
            //   (1) STRUCTURAL: every LIVE stage byte bit-identical between the
            //       CPU store and the device store. A wrong stage slot rewrites
            //       whole groups; it cannot produce zero live-byte diffs.
            //   (2) ABSOLUTE, per row: |device-cpu| <= one quantization step of
            //       that row + fp16 scale rounding.
            //   (3) WHOLE-SIGNAL: NRMSE over all decoded rows <= 1e-3, the same
            //       order of magnitude the O/LSE gates themselves use.
            // rk/rv/nk/nv remain printed as raw observations, not as gates.
            double store_nrmse_k = 0.0, store_nrmse_v = 0.0, store_abs_k = 0.0, store_abs_v = 0.0;
            size_t store_arg_k = 0, store_arg_v = 0;
            auto accum = [&](const std::vector<float> & dev, const std::vector<float> & cpu,
                             double & nrmse, double & worst_rel_to_step, size_t & argmax) {
                double se = 0.0, sr = 0.0;
                const size_t dim = size_t(D);
                for (size_t i = 0; i < dev.size() && i < cpu.size(); ++i) {
                    const double d = double(dev[i]) - double(cpu[i]);
                    se += d*d; sr += double(dev[i])*double(dev[i]);
                }
                nrmse = sr > 0.0 ? std::sqrt(se/sr) : (se > 0.0 ? 1.0 : 0.0);
                // one quantization step of the ROW this element belongs to
                worst_rel_to_step = 0.0;
                const size_t rows = dev.size() / dim;
                for (size_t r = 0; r < rows; ++r) {
                    float mn = dev[r*dim], mx = dev[r*dim];
                    for (size_t k = 1; k < dim; ++k) {
                        mn = std::min(mn, dev[r*dim+k]); mx = std::max(mx, dev[r*dim+k]);
                    }
                    const double step = (double(mx) - double(mn)) / double((1u << bits) - 1u)
                        + 2.0 * double(0x1p-11) * double(std::max(1.0f, std::fabs(mx)));
                    for (size_t k = 0; k < dim; ++k) {
                        const double d = std::fabs(double(dev[r*dim+k]) - double(cpu[r*dim+k]));
                        if (step > 0.0) {
                            if (d/step > worst_rel_to_step) argmax = r*dim+k;
                            worst_rel_to_step = std::max(worst_rel_to_step, d/step);
                        }
                    }
                }
            };
            accum(k1h_derot, k1h_cpu, store_nrmse_k, store_abs_k, store_arg_k);
            accum(v1h, v1h_cpu, store_nrmse_v, store_abs_v, store_arg_v);
            std::printf("  [%s] STORE-DIAG (not an O/LSE gate): liveStageK=%zu liveStageV=%zu "
                        "nrmseK=%.3e nrmseV=%.3e abs/stepK=%.3f abs/stepV=%.3f "
                        "maxdiffK=%.3e maxdiffV=%.3e nzK=%zu nzV=%zu\n",
                        g_case.c_str(), live_stage_k, live_stage_v,
                        store_nrmse_k, store_nrmse_v, store_abs_k, store_abs_v,
                        dk, dv, nk, nv);
            // Additional evidence only: same decoder for both stores, then
            // one FP64 inverse-Hadamard coordinate. No acceptance substitution.
            if (getenv("F4_RP_RAW_F32_ORACLE") && D % 128 == 0 && kv1 % 128 == 0 &&
                    g_cpu_rec_raw[0].size() == krec_h.size()) {
                const int dim = int(store_arg_k % D);
                const int pos = int((store_arg_k / D) % kv1);
                const int kh = int(store_arg_k / (size_t(D) * kv1));
                const auto kl = llama_kvarn_make_record_layout(128, bits, false);
                double cpu_sum = 0, gpu_sum = 0, q_part = 0, axes_part = 0;
                double max_scale = 0, max_zp = 0, max_other = 0;
                int code_diffs = 0, max_code_diff = 0;
                for (int j = 0; j < D; ++j) {
                    unsigned parity = unsigned(j & dim);
                    int odd = 0;
                    while (parity) { odd ^= int(parity & 1u); parity >>= 1; }
                    const double sign = odd ? -1.0 : 1.0;
                    double cv, gv;
                    if (pos < 128) {
                        const size_t si = (size_t(pos) * group_rows + kh * head_slices + j / 128) * 128 + j % 128;
                        cv = ggml_fp16_to_fp32(g_cpu_stg_raw[0][si]);
                        gv = ggml_fp16_to_fp32(kstg_h[si]);
                    } else {
                        const size_t ri = (size_t(pos / 128) * group_rows + kh * head_slices + j / 128) * kl.record_bytes;
                        const uint8_t * cr = g_cpu_rec_raw[0].data() + ri;
                        const uint8_t * gr = krec_h.data() + ri;
                        const int row = j % 128, col = pos % 128;
                        auto axis = [](const uint8_t * rec, size_t off, int i) {
                            ggml_fp16_t h;
                            std::memcpy(&h, rec + off + size_t(i)*sizeof(h), sizeof(h));
                            return double(ggml_fp16_to_fp32(h));
                        };
                        const int cq = llama_kvarn_unpack_bits_value(cr, row*128+col, bits);
                        const int gq = llama_kvarn_unpack_bits_value(gr, row*128+col, bits);
                        const double cs = axis(cr, kl.scale_off, row), gs = axis(gr, kl.scale_off, row);
                        const double cz = axis(cr, kl.zp_off, row), gz = axis(gr, kl.zp_off, row);
                        const double co = axis(cr, kl.other_off, col), go = axis(gr, kl.other_off, col);
                        cv = (cq*cs+cz)*co; gv = (gq*gs+gz)*go;
                        // Ordered exact decomposition: replace code first at
                        // GPU axes, then CPU->GPU axes with CPU code fixed.
                        q_part += sign*(gq-cq)*gs*go;
                        axes_part += sign*(cq*(gs*go-cs*co)+(gz*go-cz*co));
                        code_diffs += cq != gq;
                        max_code_diff = std::max(max_code_diff, std::abs(gq-cq));
                        max_scale = std::max(max_scale, std::fabs(gs-cs));
                        max_zp = std::max(max_zp, std::fabs(gz-cz));
                        max_other = std::max(max_other, std::fabs(go-co));
                    }
                    cpu_sum += sign*cv; gpu_sum += sign*gv;
                }
                const double inv = 1.0/std::sqrt(double(D));
                cpu_sum *= inv; gpu_sum *= inv; q_part *= inv; axes_part *= inv;
                std::printf("  [%s] STORE-ARGMAX K flat=%zu head=%d pos=%d dim=%d group=%d abs/step=%.9g source=%s; same-decoder FP64WHT cpu=%.12g gpu=%.12g delta=%.12g materializedDelta=%.12g residual=%.12g; codeContribution=%.12g axesContribution=%.12g codeDiffs=%d/%d maxCodeDiff=%d maxScaleDiff=%.9g maxZpDiff=%.9g maxOtherDiff=%.9g; diagnostic only, gates unchanged\n",
                        g_case.c_str(), store_arg_k, kh, pos, dim, pos/128, store_abs_k,
                        pos < 128 ? "permanent-stage0" : "records",
                        cpu_sum, gpu_sum, gpu_sum-cpu_sum,
                        double(k1h_derot[store_arg_k])-k1h_cpu[store_arg_k],
                        (double(k1h_derot[store_arg_k])-k1h_cpu[store_arg_k])-(gpu_sum-cpu_sum),
                        q_part, axes_part, code_diffs, D, max_code_diff, max_scale, max_zp, max_other);
                std::printf("  [%s] STORE-ARGMAX V flat=%zu head=%zu pos=%zu dim=%zu abs/step=%.9g\n",
                        g_case.c_str(), store_arg_v, store_arg_v/(size_t(D)*kv1),
                        (store_arg_v/D)%kv1, store_arg_v%D, store_abs_v);
            }
            // (0) VALID BYTES ARE INITIALIZED -- exact descriptor-derived plan.
            //
            // Derived from the ACTUAL uploaded descriptor (g_case_indices), the
            // actual stage/record tensors, the actual store node op_params and
            // the actual FA view node op_params -- not from the assignable
            // capacity and not from an idealisation of the mapping. It mirrors
            // the three production consumers:
            //   store        ggml-cuda/kvarn.cu  kvarn_store_kernel_*   (write,
            //                  plus kvarn_quantize_stage seal/flush reads)
            //   materialize  ggml-cuda/kvarn.cu  ggml_cuda_op_kvarn_materialize
            //   FA inline    ggml-cuda/fattn-mma-kvarn.cuh ggml_cuda_fattn_kvarn_stage_pos
            // Nothing is skipped: an out-of-range stage_pos is a hard failure,
            // never a `continue`.
            auto decode_cell = [](int64_t e) -> int64_t {
                const uint64_t payload = uint64_t(e < -1 ? -(e + 2) : e);
                return int64_t(uint32_t(payload));
            };
            auto decode_slot = [](int64_t e) -> int {
                const uint64_t payload = uint64_t(e < -1 ? -(e + 2) : e);
                const uint32_t packed = uint32_t(payload >> 32u);
                return packed == 0 ? -1 : int(packed - 1u);
            };
            auto plan_for = [&](const char * tag, int tflag, const ggml_tensor * stg,
                                const ggml_tensor * rec, const ggml_tensor * store_node,
                                const ggml_tensor * view_node,
                                const std::vector<ggml_fp16_t> & leaf,
                                uint16_t sentinel, bool have_sentinel,
                                std::vector<uint64_t> & hits_out) {
                const int64_t n_cells = stg->ne[2];
                const int n_stream = int(stg->ne[2] / (int64_t(KVAR_N_GROUP) * stage_groups));
                const int gps = n_stream > 0 ? int(rec->ne[2]) / n_stream : 0;
                const int tg = store_node->op_params[8] > 0 ? store_node->op_params[8]
                                                            : stage_groups - 1;
                const bool eager    = store_node->op_params[9] != 0;
                const bool swa      = view_node->op_params[4] != 0;
                const bool read_ind = view_node->op_params[10] != 0;
                const int64_t n_kv  = view_node->ne[2];
                CHECK(n_stream > 0);
                CHECK(gps > 0);
                CHECK(tg > 0 && tg <= stage_groups);
                CHECK(n_cells > 0);
                CHECK(n_kv > 0 && n_kv <= int64_t(g_case_indices.size()));
                // ---- live_group / live_pos, as kvarn_materialize_live_kernel does
                int64_t live_local = 0;
                for (int64_t i = 0; i < int64_t(g_case_indices.size()); ++i) {
                    const int64_t enc = g_case_indices[size_t(i)];
                    if (enc == -1) { continue; }
                    const int64_t cell = decode_cell(enc);
                    const int64_t g_g  = cell / int64_t(KVAR_N_GROUP);
                    const int64_t stream = swa ? 0 : g_g / gps;
                    const int64_t group  = swa ? g_g : g_g - stream * gps;
                    live_local = std::max(live_local, group * int64_t(KVAR_N_GROUP) + cell % KVAR_N_GROUP);
                }
                const int64_t live_group = live_local / KVAR_N_GROUP;
                const int64_t live_pos   = live_local % KVAR_N_GROUP;
                // ---- stage cells the STORE writes / reads
                std::vector<char> written(size_t(n_cells), 0);
                std::vector<char> seal_read(size_t(n_cells), 0);
                auto stage_slot_store = [&](int64_t group, int assigned) -> int64_t {
                    if (assigned >= 0) { return assigned; }
                    if (swa) { return group % stage_groups; }
                    if (group == 0) { return 0; }
                    const int64_t m = (group - 1) % tg;
                    return 1 + (m < 0 ? m + tg : m);
                };
                auto stage_slot_quant = [&](int64_t group, int assigned) -> int64_t {
                    if (assigned >= 0) { return assigned; }
                    if (swa) { return group % stage_groups; }
                    const int64_t m = (group - 1) % tg;
                    return 1 + (m < 0 ? m + tg : m);
                };
                auto mark = [&](std::vector<char> & v, int64_t pos, const char * what) {
                    if (pos < 0 || pos >= n_cells) {
                        std::printf("  [%s] INIT-PLAN %s OUT OF RANGE: %s stage_pos=%lld n_cells=%lld\n",
                                    g_case.c_str(), tag, what, (long long) pos, (long long) n_cells);
                        CHECK(pos >= 0 && pos < n_cells);
                        return;
                    }
                    v[size_t(pos)] = 1;
                };
                for (int64_t i = 0; i < int64_t(g_case_indices.size()); ++i) {
                    const int64_t enc = g_case_indices[size_t(i)];
                    if (enc == -1) { continue; }
                    const int assigned = decode_slot(enc);
                    const int64_t cell = decode_cell(enc);
                    const int64_t g_g  = cell / int64_t(KVAR_N_GROUP);
                    const int     pos  = int(cell % KVAR_N_GROUP);
                    const int64_t stream = swa ? 0 : g_g / gps;
                    const int64_t group  = swa ? g_g : g_g - stream * gps;
                    const int64_t base   = stream * int64_t(KVAR_N_GROUP) * stage_groups;
                    if (stream < 0 || stream >= n_stream || group < 0 ||
                        (!swa && group >= gps)) {
                        std::printf("  [%s] INIT-PLAN %s BAD STREAM/GROUP: stream=%lld group=%lld gps=%d\n",
                                    g_case.c_str(), tag, (long long) stream, (long long) group, gps);
                        CHECK(false);
                        continue;
                    }
                    mark(written, base + stage_slot_store(group, assigned) * KVAR_N_GROUP + pos, "store");
                    // seal / flush reads (kvarn_quantize_stage)
                    const bool seal_now = eager && pos == KVAR_N_GROUP - 1 && (swa || group > 0);
                    const bool flush_now = !eager && pos == 0 &&
                        (swa ? group >= tg : group > tg);
                    if (seal_now || flush_now) {
                        const int64_t sg = flush_now ? group - tg : group;
                        const int64_t sslot = stage_slot_quant(sg, assigned);
                        for (int t = 0; t < int(KVAR_N_GROUP); ++t) {
                            mark(seal_read, base + sslot * KVAR_N_GROUP + t, flush_now ? "flush" : "seal");
                        }
                    }
                }
                // ---- stage cells the reader (materialize / FA inline) consumes
                std::vector<char> read(size_t(n_cells), 0);
                size_t rec_reads = 0, rec_oor = 0;   // record-backed cells / out-of-range hits
                for (int64_t cell = 0; cell < n_kv; ++cell) {
                    const int64_t enc = read_ind && !swa ? g_case_indices[size_t(cell)] : cell;
                    if (enc == -1) { continue; }
                    const int64_t abs = decode_cell(enc);
                    const int64_t g_g  = abs / int64_t(KVAR_N_GROUP);
                    const int     pos  = int(abs % KVAR_N_GROUP);
                    const int64_t stream = swa ? 0 : g_g / gps;
                    const int64_t group  = swa ? g_g : g_g - stream * gps;
                    const int64_t base   = stream * int64_t(KVAR_N_GROUP) * stage_groups;
                    bool from_stage = false;
                    if (swa) {
                        from_stage = group >= 0 && group <= live_group;
                    } else if (eager) {
                        from_stage = group == 0 || (group == live_group && live_pos < KVAR_N_GROUP - 1);
                    } else {
                        from_stage = group == 0 ||
                            (group > 0 && group <= live_group && group + (tg - 1) >= live_group);
                    }
                    if (!from_stage) {
                        // from_record: validate the RECORD side too. Stage bounds
                        // were checked before; records live in the SAME cuda
                        // allocation as their neighbours, so an over-read here is
                        // invisible to compute-sanitizer memcheck (which works at
                        // cudaMalloc granularity) and is exactly the kind of
                        // error that changes result with placement only.
                        bool from_record = false;
                        if (eager) {
                            const bool completed = group < live_group ||
                                (group == live_group && live_pos == KVAR_N_GROUP - 1);
                            from_record = completed &&
                                (swa ? (live_group - group) < int64_t(gps)
                                     : (group > 0 && group < gps));
                        } else if (!swa) {
                            from_record = group < live_group && group < gps;
                        }
                        if (from_record) {
                            const int64_t rg = swa ? (stream * gps + group % gps)
                                                   : (stream * gps + group);
                            const int64_t rec_elems = rec->ne[0] * rec->ne[1] * rec->ne[2];
                            const int64_t byte_off = rg * rec->ne[1] * rec->ne[0];
                            // Counted, not CHECK()ed per cell: a per-cell CHECK
                            // adds ~1e5 assertions and, more importantly, more
                            // code than the layout-sensitive fixture tolerates
                            // (it flipped the whole gate on its own).
                            if (rg < 0 || rg >= rec->ne[2]) { rec_oor++; }
                            if (byte_off < 0 ||
                                byte_off + rec->ne[1] * rec->ne[0] > rec_elems) { rec_oor++; }
                            if (rec->ne[1] != group_rows) { rec_oor++; }
                            rec_reads++;
                        }
                        continue;
                    }
                    const int64_t sslot = swa ? (group % stage_groups)
                        : (group == 0 ? 0 : 1 + ((group - 1) % tg));
                    mark(read, base + sslot * KVAR_N_GROUP + pos, "read");
                }
                for (size_t i = 0; i < seal_read.size(); ++i) { if (seal_read[i]) { read[i] = 1; } }
                // ---- account
                size_t cap = 0, wr = 0, rd = 0, rd_unwritten = 0, unwritten = 0;
                for (int64_t i = 0; i < n_cells; ++i) {
                    ++cap;
                    const bool w = written[size_t(i)] != 0;
                    const bool r = read[size_t(i)] != 0;
                    if (w) { ++wr; }
                    if (r) { ++rd; if (!w) { ++rd_unwritten; } }
                    if (!w) { ++unwritten; }
                }
                size_t sentinel_in_read = 0;
                int64_t first_cell = -1;
                if (have_sentinel && size_t(n_cells) * size_t(group_rows) * KVAR_N_GROUP
                        != leaf.size()) {
                    std::printf("  [%s] INIT-PLAN %s leaf size mismatch: %zu vs %lld\n",
                                g_case.c_str(), tag, leaf.size(),
                                (long long) n_cells * group_rows * KVAR_N_GROUP);
                    CHECK(false);
                }
                if (have_sentinel) {
                    const size_t halves_per_cell = size_t(group_rows) * KVAR_N_GROUP;
                    for (int64_t i = 0; i < n_cells; ++i) {
                        if (!read[size_t(i)]) { continue; }
                        for (size_t k = 0; k < halves_per_cell; ++k) {
                            uint16_t hv = 0;
                            const size_t idx = size_t(i) * halves_per_cell + k;
                            std::memcpy(&hv, &leaf[idx], sizeof(hv));
                            if (hv == sentinel) {
                            ++sentinel_in_read;
                            if (first_cell < 0) { first_cell = i; }
                            hits_out.push_back((uint64_t(tflag) << 40) | uint64_t(idx));
                        }
                        }
                    }
                }
                std::printf("  [%s] INIT-PLAN %s: streams=%d gps=%d tail_groups=%d eager=%d "
                            "swa=%d read_ind=%d live=(g%lld,p%lld) | capacity=%zu written=%zu "
                            "read=%zu read_unwritten=%zu unwritten=%zu "
                            "rec_reads=%zu rec_OOR=%zu sentinel_in_read=%zu%s\n",
                            g_case.c_str(), tag, n_stream, gps, tg, (int) eager, (int) swa,
                            (int) read_ind, (long long) live_group, (long long) live_pos,
                            cap, wr, rd, rd_unwritten, unwritten, rec_reads, rec_oor,
                            sentinel_in_read,
                            first_cell >= 0 ? " (first sentinel cell above)" : "");
                // Hard, zero-tolerance, both directions -- no 50% heuristic:
                CHECK(rd > 0);
                CHECK(rec_oor == 0);   // record side, aggregated once per tensor
                // Hard, descriptor-based, zero tolerance, both tensors:
                CHECK(rd_unwritten == 0);          // every read cell was written
                // sentinel_in_read is NOT a gate on its own: a legitimately
                // stored fp16 can equal the pattern (observed: 5-6 halves out of
                // ~1.2e5 read halves, in written cells). The collision-free gate
                // is the A/A2 intersection below, where a half must appear under
                // TWO different patterns to count as unwritten. Reported here so
                // the collision rate stays visible instead of being averaged
                // away by any threshold.
                if (have_sentinel) {
                    std::printf("  [%s] INIT-CONTENT %s: sentinel_in_read=%zu "
                                "(per-run; collisions resolved by A/A2 intersection)\n",
                                g_case.c_str(), tag, sentinel_in_read);
                }
            };
            const bool want_content = !g_case_no_poison && g_case_init_sentinel;
            const uint16_t sk = uint16_t(g_case_poison) | (uint16_t(g_case_poison) << 8);
            const uint16_t sv = want_content
                ? uint16_t(g_case_poison ^ 0xFF) | (uint16_t(g_case_poison ^ 0xFF) << 8)
                : sk;
            std::vector<uint64_t> hits;   // (tensor<<40) | half index
            plan_for("K", 0, k_stg, k_rec, s_k, k_view, kstg_h, sk, want_content, hits);
            plan_for("V", 1, v_stg, v_rec, s_v, v_view, vstg_h, sv, want_content, hits);
            if (want_content) {
                const int par = g_run_index & 1;
                const int oth = par ^ 1;
                const bool paired = g_sent_seen[oth] && g_sent_tag[oth] == g_case;
                if (paired) {
                    std::vector<uint64_t> inter;
                    std::set_intersection(g_sent_hits[par].begin(), g_sent_hits[par].end(),
                                          g_sent_hits[oth].begin(), g_sent_hits[oth].end(),
                                          std::back_inserter(inter));
                    std::printf("  [%s] SENTINEL-A/A2: run%d hits=%zu run%d hits=%zu "
                                "intersection=%zu (non-empty = truly unwritten byte)\n",
                                g_case.c_str(), oth, g_sent_hits[oth].size(),
                                par, g_sent_hits[par].size(), inter.size());
                    CHECK(inter.empty());
                }
                g_sent_hits[par] = std::move(hits);
                g_sent_tag[par] = g_case;
                g_sent_seen[par] = true;
            }
            // (1) structural + (2) one-quantization-step absolute + (3) whole-signal
            CHECK(live_stage_k == 0 && live_stage_v == 0);
            CHECK(store_abs_k <= 1.0 && store_abs_v <= 1.0);
            CHECK(store_nrmse_k <= 1e-3 && store_nrmse_v <= 1e-3);
        }
        // FIRST INCORRECT LOGICAL ROW, mapped explicitly: the row layout is
        // k[(pos + kv1*kh)*D + d], so a flat index decodes to (kh, pos, d) and
        // the position is the model's absolute prompt position.
        auto first_bad = [&](const std::vector<float> & a, const std::vector<float> & b,
                             const char * what) {
            for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                if (a[i] == b[i]) { continue; }
                const size_t flat = i / size_t(D);
                const int kh = int(flat / size_t(kv1));
                const int pos = int(flat % size_t(kv1));
                const int d = int(i % size_t(D));
                std::printf("  [%s] FIRST BAD LOGICAL ROW %s: kv_head=%d pos=%d dim=%d "
                            "device=%.9g cpu=%.9g (flat=%zu)\n",
                            g_case.c_str(), what, kh, pos, d, a[i], b[i], i);
                return;
            }
            std::printf("  [%s] cpu-vs-device: no differing logical row in %s\n",
                        g_case.c_str(), what);
        };
        first_bad(k1h_derot, k1h_cpu, "K");
        first_bad(v1h, v1h_cpu, "V");
    }
    std::vector<float> q_rot;
    if (!wht_rows(env, q, D, q_rot)) {
        std::printf("  [%s] host WHT of Q failed\n", g_case.c_str());
        ++g_checks; ++g_failures;
        ggml_backend_sched_free(sched); ggml_free(gctx);
        ggml_free(dctx_c); ggml_free(dctx_v);
        ggml_backend_buffer_free(bcuda); ggml_backend_buffer_free(bvk);
        return;
    }
    CHECK(q_rot.size() == q.size());
    // Native exact tail: the reference union is built from the ACTUAL body
    // values (device records/stage) plus the ACTUAL exact-F16 tail values, so
    // an effective-key or precision substitution in the graph cannot pass.
    std::vector<float> k1h_use = k1h, v1h_use = v1h;
    // ROTATED_K_ORIGINAL_V: the store writes BOTH sides in the rotated domain
    // and the FA un-rotates V inside the kernel, so the oracle must compare
    // against the INVERSE-WHT (original) body V. The Hadamard is orthogonal
    // and auto-inverse, so wht_rows() applied once more yields the original V
    // with the same F16/4-bit rounding already baked into the stored bytes.
    if (g_domain_original_v && !g_force_cpu_fa1) {
        std::vector<float> v_body_unrot;
        if (!wht_rows(env, v1h, D, v_body_unrot)) {
            std::printf("  [%s] host WHT of the body V failed\n", g_case.c_str());
            ++g_checks; ++g_failures;
            ggml_backend_sched_free(sched); ggml_free(gctx);
            ggml_free(dctx_c); ggml_free(dctx_v);
            ggml_backend_buffer_free(bcuda); ggml_backend_buffer_free(bvk);
            return;
        }
        v1h_use = v_body_unrot;
        // F4_RP_DOMOV3_EMULATE=1: kernel-emulating oracle. The mma original-domain
        // loader writes the un-rotated V into a half2 tile, i.e. it rounds the
        // un-rotated V to F16 before the F32 accumulation. This variant applies
        // that rounding to the oracle V as well, so the residual tells us
        // whether the measured |O| error IS that rounding (and nothing else).
        if (getenv("F4_RP_DOMOV3_EMULATE") != nullptr) {
            // v1h_use is ALREADY the un-rotated V (wht of the stored F16 values);
            // the kernel rounds exactly that quantity to F16 before accumulating.
            std::vector<float> v_emul = v1h_use;
            size_t rounded = 0, total = 0;
            for (auto & x : v_emul) {
                const float y = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
                ++total; rounded += (y != x);
                x = y;
            }
            v1h_use = v_emul;
            std::printf("  [%s] ORACLE EMULATES the kernel: V un-rotated AND rounded to "
                        "F16 (%zu/%zu values changed)\n", g_case.c_str(), rounded, total);
        } else {
            std::printf("  [%s] ORACLE: K rotated, body V un-rotated (inverse WHT), "
                        "tail V stored-original (ROTATED_K_ORIGINAL_V contract)\n",
                        g_case.c_str());
        }
    }
    if (c.native_tail && n_tail_rows > 0) {
        // REFERENCE FROM THE ACTUAL STORED BYTES: kt_nat/vt_nat hold
        // fp16(val(...)) and the backend read-back is what the graph consumes.
        // Regenerating val(5/6) here would silently drop that rounding and is
        // exactly the defect this replaces.
        if (kn_exact_rb.size() != size_t(D) * n_tail_rows * n_kvh ||
            vn_exact_rb.size() != size_t(D) * n_tail_rows * n_kvh) {
            std::printf("  [%s] exact tail read-back missing (K=%zu V=%zu)\n", g_case.c_str(),
                        kn_exact_rb.size(), vn_exact_rb.size());
            ++g_checks; ++g_failures;
            ggml_backend_sched_free(sched); ggml_free(gctx);
            ggml_free(dctx_c); ggml_free(dctx_v);
            ggml_backend_buffer_free(bcuda); ggml_backend_buffer_free(bvk);
            return;
        }
        const std::vector<float> & kn_ex = kn_exact_rb;
        const std::vector<float> & vn_ex = vn_exact_rb;
        std::vector<float> kn_rot;
        if (!wht_rows_f16(env, kn_exact_f16, D, kn_rot)) {
            std::printf("  [%s] host WHT of the exact tail K failed\n", g_case.c_str());
            ++g_checks; ++g_failures;
            ggml_backend_sched_free(sched); ggml_free(gctx);
            ggml_free(dctx_c); ggml_free(dctx_v);
            ggml_backend_buffer_free(bcuda); ggml_backend_buffer_free(bvk);
            return;
        }
        for (int kh = 0; kh < n_kvh; ++kh)
            for (int j = 0; j < n_tail_rows; ++j)
                for (int d = 0; d < D; ++d) {
                    const size_t src = size_t(d) + size_t(D) * (j + n_tail_rows * kh);
                    const size_t dst = size_t(n_body_rows + j + kv1 * kh) * D + d;
                    k1h_use[dst] = kn_rot[src];
                    v1h_use[dst] = vn_ex[src];   // V stays in the original domain
                }
        std::printf("  [%s] native tail union: %d exact-F16 keys substituted into the "
                    "oracle reference (independent of the body rows)\n",
                    g_case.c_str(), n_tail_rows * n_kvh * D);
    }
    if (c.native_tail && n_tail_rows > 0) {
        // Explicit effective-key membership: body keys + tail keys, per query,
        // against the same masks the graph was fed. Every intended key exactly
        // once, no gap, no duplicate.
        for (int t = 0; t < n_q && t < 16; ++t) {
            int body_vis = 0, tail_vis = 0;
            for (int p = 0; p < kv1; ++p) {
                const size_t mi = size_t(p) + size_t(kv1) * t;
                if (p < n_body_rows) {
                    // body rows come from the body mask; tail rows are -INF there
                    if (body_mask_f32[mi] == 0.0f) { ++body_vis; }
                } else {
                    // tail rows come from the tail mask, indexed by (p - n_body)
                    const size_t ti = size_t(p - n_body_rows) + size_t(n_tail_rows) * t;
                    if (!tail_nat_f32.empty() && tail_nat_f32[ti] == 0.0f) { ++tail_vis; }
                }
            }
            int oracle_vis = 0;
            for (int p = 0; p < kv1; ++p) { if (m1[size_t(p) + size_t(kv1) * t] == 0.0f) { ++oracle_vis; } }
            std::printf("  [%s] native keys t=%d abs=%d body=%d tail=%d sum=%d oracle=%d %s\n",
                        g_case.c_str(), t, n_local - n_q + t, body_vis, tail_vis,
                        body_vis + tail_vis, oracle_vis,
                        body_vis + tail_vis == oracle_vis ? "EXACTLY-ONCE" : "MISMATCH");
            CHECK(body_vis + tail_vis == oracle_vis);
        }
    }

    {
        auto fnv = [](const std::vector<float> & v) -> uint64_t {
            uint64_t h = 14695981039346656037ull;
            const auto * p = (const uint8_t *) v.data();
            for (size_t i = 0; i < v.size() * sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ull; }
            return h;
        };
        std::printf("  [%s] oracle input hashes: k1h=%016llx k1h_use=%016llx "
                    "v1h_use=%016llx q_rot=%016llx\n", g_case.c_str(),
                    (unsigned long long) fnv(k1h), (unsigned long long) fnv(k1h_use),
                    (unsigned long long) fnv(v1h_use), (unsigned long long) fnv(q_rot));
    }
    const ref_o r1 = oracle(D, n_q, n_qh, n_kvh, q_rot, k1h_use, v1h_use, kv1, m1, scale);

    // Diagnostic references isolate MMA input rounding. They never replace
    // the independent FP64 reference used by the mandatory gates below.
    if (getenv("F4_RP_EMULATE_Q_F16") || getenv("F4_RP_EMULATE_K_F16")) {
        std::vector<float> q_mma = q_rot, k_mma = k1h_use;
        if (getenv("F4_RP_EMULATE_Q_F16")) {
            const float scale_half = ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(scale)));
            for (float & x : q_mma) {
                const float q_half = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
                x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(q_half * scale_half)) / float(scale);
            }
        }
        if (getenv("F4_RP_EMULATE_K_F16")) {
            for (float & x : k_mma) {
                x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
            }
        }
        const ref_o probe = oracle(D, n_q, n_qh, n_kvh, q_mma, k_mma, v1h_use, kv1, m1, scale);
        double probe_lse = 0.0;
        verify_lse(h1l, probe.lse, probe_lse);
        std::printf("  [%s] MMA-ROUNDING diagnostic Q=%d K=%d NRMSE=%.6e maxO=%.6e maxLSE=%.6e\n",
                    g_case.c_str(), int(getenv("F4_RP_EMULATE_Q_F16") != nullptr),
                    int(getenv("F4_RP_EMULATE_K_F16") != nullptr),
                    nrmse_of(h1, probe.o), maxdiff(h1, probe.o), probe_lse);
    }

    // merged reference, same layout as the measured tensors
    ref_o rm;
    rm.o.assign(size_t(D) * n_qh * n_q, 0.0);
    rm.lse.assign(size_t(n_qh) * n_q, 0.0);
    for (int t = 0; t < n_q; ++t)
        for (int h = 0; h < n_qh; ++h) {
            const size_t li = size_t(h) + size_t(n_qh)*t;
            const size_t oi = (size_t(t)*n_qh + size_t(h))*D;
            const double l1 = r1.lse[li], l2 = r2.lse[li];
            if (l1 == -INFINITY && l2 == -INFINITY) { rm.lse[li] = -INFINITY; continue; }
            const double M  = std::max(l1 == -INFINITY ? -1e300 : l1, l2 == -INFINITY ? -1e300 : l2);
            const double w1 = l1 == -INFINITY ? 0.0 : std::exp(l1 - M);
            const double w2 = l2 == -INFINITY ? 0.0 : std::exp(l2 - M);
            const double den = w1 + w2;
            for (int d = 0; d < D; ++d) {
                const double a = l1 == -INFINITY ? 0.0 : w1 * r1.o[oi + d];
                const double b = l2 == -INFINITY ? 0.0 : w2 * r2.o[oi + d];
                rm.o[oi + d] = (a + b) / den;
            }
            rm.lse[li] = M + std::log(den);
        }

    // ---- gates ------------------------------------------------------------
    std::printf("  [%s] D=%d n_q=%d n_qh=%d n_kvh=%d slices=%d group_rows=%d "
                "vis(local)=%d vis(ovf)=%d mask_mode=%d\n",
                g_case.c_str(), D, n_q, n_qh, n_kvh, head_slices, group_rows, vis1, vis2,
                c.mask_mode);
    // Exact operand geometry: shape, strides and the live addresses the
    // scheduler hands the backend. A stride or alignment mismatch has to be
    // excluded BEFORE anything is attributed to shader arithmetic.
    std::printf("  [%s] mask2 shape=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] type=%s "
                "bytes=%zu data=%p\n", g_case.c_str(),
                (long long) mask2->ne[0], (long long) mask2->ne[1],
                (long long) mask2->ne[2], (long long) mask2->ne[3],
                mask2->nb[0], mask2->nb[1], mask2->nb[2], mask2->nb[3],
                ggml_type_name(mask2->type), ggml_nbytes(mask2), (void *) mask2->data);
    std::printf("  [%s] q=%p k2t=%p v2t=%p (D=%d kv2=%d n_kvh=%d) fa2_supported=%d\n",
                g_case.c_str(), (void *) q_t->data, (void *) k2_t->data, (void *) v2_t->data,
                D, kv2, n_kvh, (int) ggml_backend_supports_op(env.vk, fa2));

    double nr1 = 0, m1a = 0, nr2 = 0, m2a = 0;
    CHECK(verify_o(h1, r1.o, nr1, m1a));
    CHECK(verify_o(h2, r2.o, nr2, m2a));
    double ml1 = 0, ml2 = 0;
    const bool l1ok = verify_lse(h1l, r1.lse, ml1);
    const bool l2ok = verify_lse(h2l, r2.lse, ml2);
    if (!l1ok || !l2ok) {
        // an all-masked range legitimately carries -inf; only require it where
        // the reference is finite
        for (int t = 0; t < n_q; ++t)
            for (int h = 0; h < n_qh; ++h) {
                const size_t li = size_t(h) + size_t(n_qh)*t;
                if (!std::isinf(r1.lse[li])) { CHECK(std::isfinite(h1l[li]) && std::fabs(double(h1l[li]) - r1.lse[li]) <= kGateMaxAbsLse); }
                if (!std::isinf(r2.lse[li])) { CHECK(std::isfinite(h2l[li]) && std::fabs(double(h2l[li]) - r2.lse[li]) <= kGateMaxAbsLse); }
            }
    } else { CHECK(true); CHECK(true); }
    std::printf("  [%s] local(CUDA/KVarN) NRMSE=%.3e maxO=%.3e maxLSE=%.3e | "
                "ovf(Vulkan/Q4) NRMSE=%.3e maxO=%.3e maxLSE=%.3e\n",
                g_case.c_str(), nr1, m1a, ml1, nr2, m2a, ml2);

    // Per-query overflow readout: which query diverges, how many keys it can
    // actually see, and what the FA vs the oracle returned for it.
    for (int t = 0; t < n_q && t < 16; ++t) {
        int vis = 0;
        for (int p = 0; p < kv2; ++p) { if (m2[size_t(p) + size_t(kv2) * t] == 0.0f) { ++vis; } }
        const size_t li0 = size_t(0) + size_t(n_qh) * t;   // head 0
        if (li0 >= h2l.size() || li0 >= r2.lse.size()) { continue; }
        const double got_lse = double(h2l[li0]), ref_lse = r2.lse[li0];
        const bool same = (std::isinf(ref_lse) && std::isinf(got_lse) && (ref_lse < 0) == (got_lse < 0))
                          || (!std::isinf(ref_lse) && !std::isinf(got_lse) &&
                              std::fabs(got_lse - ref_lse) <= kGateMaxAbsLse);
        if (!same) {
            std::printf("  [%s] ovf LSE DIVERGE t=%d abs=%d vis=%d got_lse=%.6g ref_lse=%.6g\n",
                        g_case.c_str(), t, n_local + n_ovf - n_q + t, vis, got_lse, ref_lse);
        }
    }

    const size_t f1 = first_diff(h1, r1.o), f2 = first_diff(h2, r2.o);
    if (f1 < h1.size()) {
        const size_t d = f1 % D, rest = f1 / D;
        std::printf("  [%s] local first diff at d=%zu h=%zu t=%zu got=%.6f ref=%.6f\n",
                    g_case.c_str(), d, rest % n_qh, rest / n_qh, h1[f1], r1.o[f1]);
    }
    if (f2 < h2.size()) {
        const size_t d = f2 % D, rest = f2 / D;
        std::printf("  [%s] ovf   first diff at d=%zu h=%zu t=%zu got=%.6f ref=%.6f\n",
                    g_case.c_str(), d, rest % n_qh, rest / n_qh, h2[f2], r2.o[f2]);
    }

    CHECK(masked_rows_ok(D, n_q, n_qh, r1, h1, h1l));
    CHECK(masked_rows_ok(D, n_q, n_qh, r2, h2, h2l));
    // Merged output against the merged reference, unconditionally: this is the
    // per-query hard assertion for both-masked rows (O must be exactly 0 and
    // LSE exactly -inf) and it covers the both-empty case too.
    CHECK(masked_rows_ok(D, n_q, n_qh, rm, ho, hl));

    // Plan 3.4 is an ACCEPTED case, not an invalid precondition: empty/fully
    // masked -> O=0, LSE=-inf, both empty -> return zero.
    const bool both_empty = (n_local == 0 && n_ovf == 0);
    {
        double nrm = 0, mo = 0, ml = 0;
        const bool ook = verify_o(ho, rm.o, nrm, mo);
        const bool lok = verify_lse(hl, rm.lse, ml);
        std::printf("  [%s] merged NRMSE(O)=%.3e maxO=%.3e maxLSE=%.3e first_bad=%zd/%zu%s\n",
                    g_case.c_str(), nrm, mo, ml, first_diff(ho, rm.o), ho.size(),
                    both_empty ? " [both ranges empty -> must be exactly 0 / -inf]" : "");
        CHECK(ook);
        CHECK(lok);

        if (both_empty) {
            // Hard assertions for the accepted both-empty contract.
            for (float x : ho) { CHECK(x == 0.0f); }
            for (float x : hl) { CHECK(std::isinf(x) && x < 0.0f); }
            CHECK(ho.size() == rm.o.size() && hl.size() == rm.lse.size());
        }

        // single valid range: the MERGED tensors must equal that range
        if (!both_empty && (n_local == 0 || n_ovf == 0)) {
            const std::vector<float> & mo_ = n_local == 0 ? h2 : h1;
            const std::vector<float> & ml_ = n_local == 0 ? h2l : h1l;
            const ref_o & vr = n_local == 0 ? r2 : r1;
            double sn = 0, sm = 0, sl = 0;
            const bool so = verify_o(ho, vr.o, sn, sm);
            const bool sv = verify_lse(hl, vr.lse, sl);
            std::printf("  [%s] single-range-after-merge: merged-vs-range NRMSE=%.3e "
                        "maxO=%.3e maxLSE=%.3e | rawrange NRMSE=%.3e\n",
                        g_case.c_str(), sn, sm, sl, nrmse_of(mo_, vr.o));
            CHECK(so);   // ho vs the valid range -- not only_o
            CHECK(sv);   // hl vs the valid range
            CHECK(nrmse_of(mo_, vr.o) <= kGateNrmseO);
        }
    }

    // ---- negative controls through the SAME verifier ----------------------
    {
        double nr = 0, md = 0;
        std::vector<float> bad = ho;
        if (!bad.empty()) { bad[0] = std::nanf(""); }
        CHECK(!verify_o(bad, rm.o, nr, md));                       // NaN in O
        std::vector<float> bad_lse_v = hl;                         // LSE-shaped
        if (!bad_lse_v.empty()) { bad_lse_v[0] = std::nanf(""); }
        CHECK(!verify_lse(bad_lse_v, rm.lse, md));                 // NaN in LSE
        std::vector<float> bad_ref = hl;                           // NaN in reference
        std::vector<double> nan_ref = rm.lse;
        if (!nan_ref.empty()) { nan_ref[n_qh > 1 ? 1 : 0] = std::nan(""); }
        CHECK(!verify_lse(bad_ref, nan_ref, md));
        // correct -inf first, then a bad value at another head/query
        if (n_qh > 1 && !hl.empty()) {
            std::vector<float> mix = hl;
            mix[0] = -INFINITY;
            std::vector<double> ref_mix = rm.lse;
            ref_mix[0] = -INFINITY;
            CHECK(verify_lse(mix, ref_mix, md));
            mix[n_qh - 1] = std::nanf("");
            CHECK(!verify_lse(mix, ref_mix, md));
            mix[n_qh - 1] = 0.0f;
            CHECK(!verify_lse(mix, ref_mix, md));
            mix[n_qh - 1] = INFINITY;
            CHECK(!verify_lse(mix, ref_mix, md));
        }

        std::vector<float> badl = hl;
        if (!badl.empty() && n_qh > 1) {
            badl[n_qh - 1] = std::nanf("");                        // NaN in a different head-query
            CHECK(!verify_lse(badl, rm.lse, md));
        }
        std::vector<float> pert = ho;
        if (!pert.empty()) { pert[pert.size()/2] += 0.5f; }         // wrong value, right head-query
        CHECK(!verify_o(pert, rm.o, nr, md));

        std::vector<float> shorter = ho;
        if (!shorter.empty()) { shorter.pop_back(); }             // size mismatch
        CHECK(!verify_o(shorter, rm.o, nr, md));

        // a fully-masked row with a finite LSE must be rejected
        if (n_local == 0 || n_ovf == 0) {
            std::vector<float> vm = n_local == 0 ? h1l : h2l;
            std::vector<float> vc = vm;
            for (size_t i = 0; i < vc.size(); ++i) { if (std::isinf(vc[i])) { vc[i] = -3.0f; } }
            CHECK(!verify_lse(vc, (n_local == 0 ? r1 : r2).lse, md));
        }
    }

    // Additional raw-record diagnostic; NEVER replaces r1/rm or any assertion.
    // Also available on the small native DOROV3 cases as a decoder control.
    if (getenv("F4_RP_RAW_F32_ORACLE") && g_domain_original_v && !g_force_cpu_fa1 &&
            c.native_tail && n_local > 0 && n_tail_rows == 128 &&
            n_local % 128 == 0 && n_body_rows % 128 == 0) {
        std::vector<float> raw_k, raw_v, original_v;
        bool decoded = raw_record_rows_f32(krec_h, k_rb, D, n_kvh,
                kv1, n_body_rows, bits, false, raw_k) &&
            raw_record_rows_f32(vrec_h, v_rb, D, n_kvh,
                kv1, n_body_rows, bits, true, raw_v);
        if (decoded) {
            // Eager non-SWA group zero is permanently staged, even when full.
            // Read the actual device stage bytes, matching the production resolver.
            for (int kh = 0; kh < n_kvh; ++kh)
                for (int pos = 0; pos < std::min(128, n_body_rows); ++pos)
                    for (int d = 0; d < D; ++d) {
                        const size_t si = (size_t(pos) * group_rows + kh * head_slices + d / 128) * 128 + d % 128;
                        const size_t oi = (size_t(pos) + size_t(kv1) * kh) * D + d;
                        raw_k[oi] = ggml_fp16_to_fp32(kstg_h[si]);
                        raw_v[oi] = ggml_fp16_to_fp32(vstg_h[si]);
                    }
            decoded = wht_rows(env, raw_v, D, original_v);
        }
        if (!decoded) {
            std::printf("  [%s] RAW-F32 diagnostic unavailable: complete-record geometry/decode failed\n", g_case.c_str());
        } else {
            // Integration audit against actual-byte materialization BEFORE tail
            // replacement. It checks both record/stage selection and V domain.
            std::vector<float> vr_flat, vr_head;
            const bool audit_ok = kvarn_rows_from_device(env, vrec_h, vstg_h, v_rb,
                    group_rows, kv1, bits, true, stage_groups, head_slices, true, vr_flat);
            if (audit_ok) {
                to_head_rows(vr_flat, vr_head);
                double kdiff = 0, vdiff = 0, inverse_vdiff = 0, existing_vdiff = 0;
                for (int kh = 0; kh < n_kvh; ++kh)
                    for (int pos = 0; pos < n_body_rows; ++pos)
                        for (int d = 0; d < D; ++d) {
                            const size_t i = (size_t(pos) + size_t(kv1) * kh) * D + d;
                            const auto halfround = [](float x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); };
                            kdiff = std::max(kdiff, std::fabs(double(halfround(raw_k[i])) - k1h[i]));
                            vdiff = std::max(vdiff, std::fabs(double(halfround(raw_v[i])) - vr_head[i]));
                            inverse_vdiff = std::max(inverse_vdiff, std::fabs(double(halfround(original_v[i])) - v1h[i]));
                            existing_vdiff = std::max(existing_vdiff, std::fabs(double(raw_v[i]) - v1h_use[i]));
                        }
                std::printf("  [%s] RAW-F32 integration audit body-only: F16(rawK)-materializedRotK=%.9g F16(rawV)-materializedRotV=%.9g F16(WHT(rawV))-materializedOriginalV=%.9g rawV-existingOracleV=%.9g; group0=device-stage\n",
                        g_case.c_str(), kdiff, vdiff, inverse_vdiff, existing_vdiff);
            }
            // The fixture applies output WHT even for DOROV3, after the
            // kernel inverse-WHT(V). Their composition is the identity in the
            // independent mathematical oracle (WHT is linear and self-inverse).
            // Comparing only inverse-WHT(V) would use the wrong output domain.
            original_v = raw_v;
            // Preserve the existing exact-tail contract: F16-rotated K and
            // stored-original V from the native device readback, exactly once.
            for (int kh = 0; kh < n_kvh; ++kh)
                for (int pos = n_body_rows; pos < kv1; ++pos)
                    for (int d = 0; d < D; ++d) {
                        const size_t i = (size_t(pos) + size_t(kv1) * kh) * D + d;
                        raw_k[i] = k1h_use[i]; original_v[i] = v1h_use[i];
                    }
            const ref_o raw_local = oracle(D, n_q, n_qh, n_kvh, q_rot,
                    raw_k, original_v, kv1, m1, scale);
            ref_o raw_merged = raw_local;
            for (size_t li = 0; li < raw_local.lse.size(); ++li) {
                const double a = raw_local.lse[li], b = r2.lse[li];
                const double mx = std::max(a, b);
                const double wa = a == -INFINITY ? 0.0 : std::exp(a - mx);
                const double wb = b == -INFINITY ? 0.0 : std::exp(b - mx);
                const double den = wa + wb;
                raw_merged.lse[li] = den > 0 ? mx + std::log(den) : -INFINITY;
                for (int d = 0; d < D; ++d) {
                    const size_t i = li * D + d;
                    raw_merged.o[i] = den > 0 ? (wa * raw_local.o[i] + wb * r2.o[i]) / den : 0.0;
                }
            }
            double ln = 0, lo = 0, ll = 0, mn = 0, mo = 0, ml = 0;
            const bool lok = verify_o(h1, raw_local.o, ln, lo);
            const bool llok = verify_lse(h1l, raw_local.lse, ll);
            const bool mok = verify_o(ho, raw_merged.o, mn, mo);
            const bool mlok = verify_lse(hl, raw_merged.lse, ml);
            std::printf("  [%s] RAW-F32 diagnostic actual-device-records plus permanent-stage0 K[dim,token] V[token,dim], kernel+output WHT, no F16 materialization; local NRMSE=%.6e maxO=%.6e maxLSE=%.6e pass=%d; merged NRMSE=%.6e maxO=%.6e maxLSE=%.6e pass=%d; original gates unchanged\n",
                    g_case.c_str(), ln, lo, ll, int(lok && llok), mn, mo, ml, int(mok && mlok));
        }
    }

    // ---- independent-oracle negative controls: hidden duplicate / dropped key
    // Perturbing the measured output is not enough: the oracle itself must
    // reject a union that double-counts or loses a tail key, otherwise a real
    // duplicate/drop in the production graph would be invisible to it.
    {
        const int n_tail_rows = std::max(0, std::min(c.n_tail, kv1));
        const int n_body_rows = kv1 - n_tail_rows;
        const int n_kv2 = kv1 + n_tail_rows;
        auto dup_rows = [&](const std::vector<float> & src, std::vector<float> & dst) {
            dst.assign(size_t(D) * size_t(n_kv2) * n_kvh, 0.0f);
            for (int kh = 0; kh < n_kvh; ++kh)
                for (int p = 0; p < n_kv2; ++p)
                    for (int d = 0; d < D; ++d) {
                        const int sp = p < kv1 ? p : (p - n_tail_rows);
                        dst[(size_t(p) + size_t(n_kv2) * kh) * D + d] =
                            src[(size_t(sp) + size_t(kv1) * kh) * D + d];
                    }
        };
        auto dup_mask = [&](const std::vector<float> & m, std::vector<float> & dst) {
            dst.assign(size_t(n_kv2) * n_q, -INFINITY);
            for (int t = 0; t < n_q; ++t)
                for (int p = 0; p < n_kv2; ++p) {
                    const int sp = p < kv1 ? p : (p - n_tail_rows);
                    dst[size_t(p) + size_t(n_kv2) * t] = m[size_t(sp) + size_t(kv1) * t];
                }
        };
        auto drop_rows = [&](const std::vector<float> & src, std::vector<float> & dst) {
            dst.assign(size_t(D) * size_t(n_body_rows) * n_kvh, 0.0f);
            for (int kh = 0; kh < n_kvh; ++kh)
                for (int p = 0; p < n_body_rows; ++p)
                    for (int d = 0; d < D; ++d)
                        dst[(size_t(p) + size_t(n_body_rows) * kh) * D + d] =
                            src[(size_t(p) + size_t(kv1) * kh) * D + d];
        };
        if (n_tail_rows > 0 && n_body_rows > 0 && !h1.empty()) {
            std::vector<float> kd, vd, md;
            // Long native controls duplicate/drop the SAME union measured by
            // the positive oracle, including actual exact-tail substitutions.
            const auto & control_k = long_rotated_control ? k1h_use : k1h;
            const auto & control_v = long_rotated_control ? v1h_use : v1h;
            dup_rows(control_k, kd); dup_rows(control_v, vd); dup_mask(m1, md);
            const ref_o rdup = oracle(D, n_q, n_qh, n_kvh, q_rot, kd, vd, n_kv2, md, scale);
            double nd = 0, md_ = 0, nld = 0;
            const bool dup_pass = verify_o(h1, rdup.o, nd, md_);
            verify_lse(h1l, rdup.lse, nld);
            std::printf("  [%s] ORACLE control duplicate-tail: NRMSE(O)=%.3e NRMSE(LSE)=%.3e "
                        "(must EXCEED gate %.1e)\n", g_case.c_str(), nd, nld, kGateNrmseO);
            CHECK(nd > kGateNrmseO);
            CHECK(!dup_pass);
            std::vector<float> kdr, vdr, mdr;
            drop_rows(control_k, kdr); drop_rows(control_v, vdr);
            mdr.assign(size_t(n_body_rows) * n_q, -INFINITY);
            for (int t = 0; t < n_q; ++t)
                for (int p = 0; p < n_body_rows; ++p)
                    mdr[size_t(p) + size_t(n_body_rows) * t] = m1[size_t(p) + size_t(kv1) * t];
            const ref_o rdrop = oracle(D, n_q, n_qh, n_kvh, q_rot, kdr, vdr, n_body_rows, mdr, scale);
            double ndr = 0, mdr_ = 0, nldr = 0;
            const bool drop_pass = verify_o(h1, rdrop.o, ndr, mdr_);
            verify_lse(h1l, rdrop.lse, nldr);
            std::printf("  [%s] ORACLE control dropped-tail: NRMSE(O)=%.3e NRMSE(LSE)=%.3e "
                        "(must EXCEED gate %.1e)\n", g_case.c_str(), ndr, nldr, kGateNrmseO);
            CHECK(ndr > kGateNrmseO);
            CHECK(!drop_pass);
        } else {
            std::printf("  [%s] ORACLE duplicate/drop controls skipped "
                        "(tail=%d body=%d -- nothing to duplicate)\n",
                        g_case.c_str(), n_tail_rows, n_body_rows);
        }
    }

    // A/A2 is a BIT-identity check: a NaN compares equal to itself, so
    // numerical validity is asserted here on its own, never left to A/A2.
    {
        // NaN is never a valid result -- not for the both-empty case either,
        // and a matching NaN must never be allowed to pass A/A2 below.
        bool nan_o = false, inf_o = false;
        for (float x : ho) {
            if (std::isnan(x)) { nan_o = true; break; }
            if (std::isinf(x)) { inf_o = true; break; }
        }
        CHECK(!nan_o);                       // O is never NaN (both-empty must be 0)
        CHECK(!inf_o);                       // O is finite; empty rows are exactly 0
        if (both_empty) {
            for (float x : ho) { CHECK(x == 0.0f); }
        }
        bool nan_lse = false, posinf_lse = false;
        for (float x : hl) {
            if (std::isnan(x)) { nan_lse = true; break; }
            if (x > 0.0f && std::isinf(x)) { posinf_lse = true; break; }
        }
        CHECK(!nan_lse);                     // LSE is never NaN
        CHECK(!posinf_lse);                  // -inf only, never +inf
        if (both_empty) {
            for (float x : hl) { CHECK(std::isinf(x) && x < 0.0f); }
        }
        std::printf("  [%s] nonfinite check: O finite=%d LSE finite-or--inf=%d "
                    "(NaN/+inf always rejected, -inf allowed only where the "
                    "reference is masked -- see verify_lse)\n",
                    g_case.c_str(), (int) !nan_o && (int) !inf_o,
                    (int) !nan_lse && (int) !posinf_lse);
    }

    g_last_o = ho;
    g_last_lse = hl;
    g_last_local = h1;
    g_last_local_lse = h1l;
    g_last_body = h_body;
    g_last_body_lse = h_body_l;
    // Descriptor actually fed to the merge (production layout: seq_id, order
    // offset, n_queries, run, slot count, current, ring slots).
    g_last_desc = desc_snapshot;

    ggml_backend_sched_free(sched);
    ggml_free(gctx);
    ggml_free(dctx_c);
    ggml_free(dctx_v);
    ggml_backend_buffer_free(bcuda);
    ggml_backend_buffer_free(bvk);
}

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    for (int i = 1; i < argc; ++i) { (void) argv[i]; }

    env_t env;
    if (!env_init(env)) {
        std::printf("real-pair O/LSE: SKIPPED (needs CUDA + Vulkan)\n");
        return 77;
    }

    // P = 512 (plan 3.1). The n=511/512/513/641 rows map the SEQUENCE LENGTHS
    // onto the ranges exactly as the cache assigns them: pos < P -> local,
    // pos >= P -> overflow. 641 is the second occupied-overflow length.
    const cfg_t cases[] = {
        //                         D  n_q nqh nkh nloc novf bits ntail mode nat tag
        { 128,  1,  6, 2,  64, 32, 4,   0, 0, 0, "D128 Q1 body-only both ranges" },
        { 128,  1,  6, 2,  64, 32, 4,  32, 0, 0, "D128 Q1 body+tail both ranges" },
        { 128,  1,  6, 2,  64, 32, 4,  64, 0, 0, "D128 Q1 tail-only both ranges" },
        { 128,  1,  6, 2,  64, 32, 4,  32, 0, 1, "D128 Q1 NATIVE exact-tail body+tail" },
        { 128,  2,  6, 2,  64, 32, 4,  32, 0, 0, "D128 Q2(bits4) body+tail" },
        { 128,  1,  6, 2,  64,  0, 4,  32, 0, 0, "D128 Q1 body+tail, EMPTY overflow" },
        { 128,  1,  6, 2,   0, 32, 4,   0, 0, 0, "D128 Q1 overflow only" },
        { 128,  1,  6, 2,   0,  0, 4,   0, 0, 0, "D128 Q1 both empty" },
        { 128, 96,  6, 2, 128, 64, 4,  32, 0, 0, "D128 prefill body+tail" },
        { 256,  1, 24, 4, 128, 64, 4,  32, 0, 0, "D256 Q1 body+tail (2 kv heads)" },
        { 256,  1, 24, 4, 511,  0, 4, 128, 0, 0, "P=512 n=511 body+tail, empty overflow" },
        { 256,  1, 24, 4, 512,  0, 4, 128, 0, 0, "P=512 n=512 body+tail, empty overflow" },
        { 256,  8, 24, 4, 512,  1, 4, 128, 0, 0, "P=512 n=513 body+tail, occupied overflow=1" },
        { 256,  1, 24, 4, 512,  1, 4, 128, 0, 0, "P=512 n=513 Q1 body+tail, occupied overflow=1" },
        { 256,  8, 24, 4, 512,129, 4, 128, 0, 0, "P=512 n=641 body+tail, occupied overflow=129" },
        { 256,  1, 24, 4, 512,129, 4, 128, 0, 0, "P=512 n=641 Q1 body+tail, occupied overflow=129" },
        { 256,  1, 24, 4, 512,129, 4, 512, 0, 0, "P=512 n=641 TAIL-ONLY local, occupied overflow" },
        { 256,  8, 24, 4, 511,  0, 4,   0, 0, 0, "P=512 n=511 body-only, empty overflow" },
        { 256,  1, 24, 4,   0,129, 4,   0, 0, 0, "P=512 n=641 overflow only (local empty)" },
        // --- Q boundary around the failing D256 + occupied overflow=129 ---
        { 256,  2, 24, 4, 512,129, 4, 128, 0, 0, "BOUNDARY D256 ovf=129 Q2" },
        { 256,  7, 24, 4, 512,129, 4, 128, 0, 0, "BOUNDARY D256 ovf=129 Q7" },
        { 256,  9, 24, 4, 512,129, 4, 128, 0, 0, "BOUNDARY D256 ovf=129 Q9" },
        { 256, 16, 24, 4, 512,129, 4, 128, 0, 0, "BOUNDARY D256 ovf=129 Q16" },
        { 256,  2, 24, 4, 512,  1, 4, 128, 0, 0, "BOUNDARY D256 ovf=1 Q2" },
        // --- same operands, mask changed: causal vs all-visible ---
        { 256,  8, 24, 4, 512,129, 4, 128, 1, 0, "MASK D256 ovf=129 Q8 ALL-VISIBLE" },
        { 256,  8, 24, 4, 512,  1, 4, 128, 1, 0, "MASK D256 ovf=1 Q8 ALL-VISIBLE" },
        { 128,  8,  6, 2,  64, 32, 4,  32, 1, 0, "MASK D128 ovf=32 Q8 ALL-VISIBLE" },
        // --- PRODUCTION native exact tail (independent F16 tail storage) ---
        { 256,  1, 24, 4, 512,  0, 4, 128, 0, 1, "NATIVE P=512 n=512 Q1 tail, empty ovf" },
        { 256,  1, 24, 4, 511,  0, 4, 128, 0, 1, "NATIVE P=512 n=511 Q1 tail, empty ovf" },
        { 256,  8, 24, 4, 512,  1, 4, 128, 0, 1, "NATIVE P=512 n=513 Q8 tail, ovf=1" },
        { 256,  1, 24, 4, 512,  1, 4, 128, 0, 1, "NATIVE P=512 n=513 Q1 tail, ovf=1" },
        { 256,  8, 24, 4, 512,129, 4, 128, 0, 1, "NATIVE P=512 n=641 Q8 tail, ovf=129" },
        { 256,  1, 24, 4, 512,129, 4, 128, 0, 1, "NATIVE P=512 n=641 Q1 tail, ovf=129" },
        { 256,  1, 24, 4, 512,129, 4, 512, 0, 1, "NATIVE P=512 n=641 Q1 TAIL-ONLY, ovf=129" },
        { 128,  1,  6, 2,  64, 32, 4,  32, 0, 1, "NATIVE D128 Q1 tail body+tail" },
        // ---- msg57: PRODUCTION prefill route, domain ROTATED_K_ORIGINAL_V ----
        // D256/GQA24-4/P-style: local 512 with a 128-row exact tail, occupied
        // overflow of 1 and 129 rows (the boundary that used to fail), Q=256
        // (a real prefill ubatch) and Q=1 (the single-query side). Selected by
        // F4_RP_DOMAIN_ORIGINAL_V=1; ignored by every other run.
        { 256,  1, 24, 4, 512,  1, 4, 128, 0, 1, "DOROV3 P=512 n=513 Q1 tail, ovf=1" },
        { 256, 256, 24, 4, 512,  1, 4, 128, 0, 1, "DOROV3 P=512 n=513 Q256 prefill tail, ovf=1" },
        { 256, 256, 24, 4, 512,129, 4, 128, 0, 1, "DOROV3 P=512 n=641 Q256 prefill tail, ovf=129" },
        { 256,  8, 24, 4, 512,129, 4, 128, 0, 1, "DOROV3 P=512 n=641 Q8 tail, ovf=129" },
        { 256, 256, 24, 4, 511,  0, 4, 128, 0, 1, "DOROV3 P=512 n=511 Q256 prefill, empty ovf" },
        // body-only / tail-only isolation for the domain-3 divergence
        { 256,  1, 24, 4, 512,  0, 4,   0, 0, 1, "DOROV3 P=512 n=512 Q1 BODY-ONLY" },
        { 256, 256, 24, 4, 512,  0, 4,   0, 0, 1, "DOROV3 P=512 n=512 Q256 BODY-ONLY" },
        { 256,  1, 24, 4, 512,  0, 4, 512, 0, 1, "DOROV3 P=512 n=512 Q1 TAIL-ONLY" },
        // Populated D256/Q1 body with zero visible keys exercises split combine
        // empty-row handling; no tail avoids meaningless duplicate/drop controls.
        { 256, 1, 24, 4, 512, 0, 4, 0, 2, 1, "DOROV3 OPTIN Q1 ALLMASK BODY-ONLY" },
        // Actual decode-domain long controls: explicit filter, domain-3 OFF.
        { 256, 1, 24, 4, 102400, 2600, 4, 128, 0, 1, "LONGROTATED P=102400 n=105000 Q1 tail, ovf=2600" },
        { 256, 1, 24, 4, 102400, 12600, 4, 128, 0, 1, "LONGROTATED P=102400 n=115000 Q1 tail, ovf=12600" },
        // Q2 is the smallest windowed-prefill-eligible query count (Q>1).
        // Explicit LONGPREFILL selection keeps this same-KV chunk comparison opt-in.
        { 256, 2, 24, 4, 102400, 2600, 4, 128, 0, 1, "DOROV3 LONGPREFILL P=102400 n=105000 Q2 tail, ovf=2600" },
        // Long-context oracle cases require an explicit LONG filter and domain 3.
        { 256,  1, 24, 4, 102400,  2600, 4, 128, 0, 1, "DOROV3 LONG P=102400 n=105000 Q1 tail, ovf=2600" },
        { 256,  1, 24, 4, 102400, 12600, 4, 128, 0, 1, "DOROV3 LONG P=102400 n=115000 Q1 tail, ovf=12600" },
    };

    // Optional single-case selection so one tiny failing case can be run in
    // its own process (F4_RP_CASE=<substring of tag>).
    const char * only = getenv("F4_RP_CASE");
    g_domain_original_v = getenv("F4_RP_DOMAIN_ORIGINAL_V") != nullptr;
    g_emulate_p_f16     = getenv("F4_RP_EMULATE_P_F16") != nullptr;
    const bool doro3_cpu_too = getenv("F4_RP_DORO3_CPU_TOO") != nullptr;
    for (const cfg_t & c : cases) {
        if (std::string(c.tag).find("LONGROTATED") != std::string::npos &&
                (!only || std::string(only).find("LONGROTATED") == std::string::npos || g_domain_original_v)) { continue; }
        if (std::string(c.tag).find("DOROV3 OPTIN") != std::string::npos &&
                (!only || std::string(only).find("OPTIN") == std::string::npos || !g_domain_original_v)) { continue; }
        if (std::string(c.tag).find("DOROV3 LONGPREFILL") != std::string::npos &&
                (!only || std::string(only).find("LONGPREFILL") == std::string::npos || !g_domain_original_v)) { continue; }
        if (std::string(c.tag).find("DOROV3 LONG") != std::string::npos &&
                (!only || std::string(only).find("LONG") == std::string::npos || !g_domain_original_v)) { continue; }
        if (only && *only && std::string(c.tag).find(only) == std::string::npos) { continue; }
        if (g_domain_original_v && std::string(c.tag).find("DOROV3") == std::string::npos) { continue; }
        g_force_cpu_fa1 = false;
        run_case(env, c);
        const std::vector<float> a_o = g_last_o, a_l = g_last_lse;
        const std::vector<float> a_loc = g_last_local, a_locl = g_last_local_lse;
        const std::vector<float> a_bd  = g_last_body,  a_bdl = g_last_body_lse;
        const std::vector<int32_t> a_desc = g_last_desc;
        const std::string tag = g_case;
        // second pass: local FA on the CPU (reference route), only when
        // explicitly requested - the checks below are per-pass and the report
        // names the route/oracle used.
        if (doro3_cpu_too) {
            g_force_cpu_fa1 = true;
            run_case(env, c);
            g_force_cpu_fa1 = false;
        }
        run_case(env, c);                      // A2: same config, fresh graph

        // Bit-identity first: matching NaNs must never count as valid
        // determinism, so compare raw bit patterns rather than ==.
        auto bits = [](const std::vector<float> & a, const std::vector<float> & b) {
            if (a.size() != b.size()) { return false; }
            for (size_t i = 0; i < a.size(); ++i) {
                uint32_t x = 0, y = 0;
                std::memcpy(&x, &a[i], sizeof(x));
                std::memcpy(&y, &b[i], sizeof(y));
                if (x != y) { return false; }
            }
            return true;
        };
        auto mag = [](const std::vector<float> & a, const std::vector<float> & b) -> double {
            if (a.size() != b.size()) { return (double) INFINITY; }
            double m = 0.0;
            for (size_t i = 0; i < a.size(); ++i) {
                m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
            }
            return m;
        };
        // A/A2 = two INDEPENDENTLY REBUILT graphs (different node addresses,
        // different tensor placement). SAME-GRAPH repeat, gated elsewhere at
        // max|dO| == 0, is the kernel/backend determinism gate. They are kept
        // apart on purpose: bit-identity across a rebuild is not a property the
        // backend promises, but a few fp32 ULPs in the body+tail merge
        // accumulation is also not a comparison-contract mistake.
        //
        // Storage: ggml_flash_attn_ext emits O and LSE as GGML_TYPE_F32, and
        // the fixture readback keeps them as `float`. The number 1.1920929e-07
        // alone proves nothing -- it is only "one fp32 ULP" for a value inside
        // [0.5,1). What is gated is the measured ULP DISTANCE in that storage.
        auto ulp_dist = [](float a, float b) -> uint64_t {
            if (a == b) { return 0; }
            if (std::isnan(a) || std::isnan(b)) { return UINT64_MAX; }
            uint32_t ia, ib;
            std::memcpy(&ia, &a, sizeof(ia));
            std::memcpy(&ib, &b, sizeof(ib));
            if (ia & 0x80000000u) { ia = 0x80000000u - ia; } else { ia |= 0x80000000u; }
            if (ib & 0x80000000u) { ib = 0x80000000u - ib; } else { ib |= 0x80000000u; }
            return ia > ib ? uint64_t(ia - ib) : uint64_t(ib - ia);
        };
        auto ulps = [&](const std::vector<float> & x, const std::vector<float> & y,
                        size_t & where) -> uint64_t {
            uint64_t mx = 0; where = 0;
            for (size_t i = 0; i < x.size() && i < y.size(); ++i) {
                const uint64_t u = ulp_dist(x[i], y[i]);
                if (u > mx) { mx = u; where = i; }
            }
            return mx;
        };
        const bool desc_ok = a_desc.size() == g_last_desc.size() &&
            std::equal(a_desc.begin(), a_desc.end(), g_last_desc.begin());
        size_t wbo=0, wbl=0, wlo=0, wll=0, woo=0, wol=0;
        const uint64_t u_bo = ulps(a_bd,   g_last_body,      wbo);
        const uint64_t u_bl = ulps(a_bdl,  g_last_body_lse,  wbl);
        const uint64_t u_lo = ulps(a_loc,  g_last_local,     wlo);
        const uint64_t u_ll = ulps(a_locl, g_last_local_lse, wll);
        const uint64_t u_o  = ulps(a_o,    g_last_o,         woo);
        const uint64_t u_l  = ulps(a_l,    g_last_lse,       wol);
        const uint64_t u_max = std::max({u_bo, u_bl, u_lo, u_ll, u_o, u_l});
        // Per-level fp32 ULP budgets for an INDEPENDENTLY REBUILT graph.
        // Measured basis (f4-gate-gate-20261006-054427.log, 35 cases):
        //   desc_equal, bodyO, bodyLSE, localLSE, mergedLSE = 0 ULP in all 35;
        //   the only non-zero was D128 Q1 NATIVE exact-tail:
        //     localO=6 mergedO=2, everything else 0.
        // Attribution: local O is the output of ggml_kv_tail_attention_merge
        // (src/llama-graph.cpp:4623-4639), a SEQUENTIAL fp32 reduction
        //   M=max(la,lb); o=(oa*e^(la-M)+ob*e^(lb-M))/(e^(la-M)+e^(lb-M))
        // whose result depends on the order the two bodies are combined. The
        // LSE levels share the same exponentials and stayed at 0 ULP, which
        // bounds the operand difference to the accumulation, not to the data.
        // Rebuilding the graph changes node/tensor addresses, which changes how
        // ggml_backend_sched splits and therefore the order of that reduction.
        // Same-graph repeats are gated separately at exact 0 (SAME-GRAPH repeat
        // max|dO| == 0), and every original O/LSE NRMSE/max gate is untouched.
        // The 4-ULP budget is NOT supported by measurement: two independent
        // batches observed localO=6 and localO=9 (same case, same binary). A ULP
        // count chosen just above the last observation would be tuning, so the
        // gate is expressed in the SAME units and with the SAME constants the
        // original O/LSE acceptance already uses -- an independently rebuilt
        // graph may never be looser than the oracle gate itself. ULP stays
        // printed as the diagnosis of HOW it diverged.
        //   local O / merged O  -> verify_o(): nrmse <= kGateNrmseO && maxabs <= kGateMaxAbsO
        //   * LSE               -> maxabs <= kGateMaxAbsLse
        //   descriptor          -> exact
        auto nrmse_of = [](const std::vector<float> & a, const std::vector<float> & b) {
            double se = 0.0, sr = 0.0;
            for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                const double d = double(a[i]) - double(b[i]);
                se += d*d; sr += double(b[i])*double(b[i]);
            }
            return sr > 0.0 ? std::sqrt(se/sr) : (se > 0.0 ? 1.0 : 0.0);
        };
        const double ao_local = mag(a_loc, g_last_local);
        const double ao_merged = mag(a_o, g_last_o);
        const double al_local = mag(a_locl, g_last_local_lse);
        const double al_merged = mag(a_l, g_last_lse);
        const double no_local = nrmse_of(a_loc, g_last_local);
        const double no_merged = nrmse_of(a_o, g_last_o);
        const bool pre_ok  = no_local  <= kGateNrmseO && ao_local  <= kGateMaxAbsO &&
                             no_merged <= kGateNrmseO && ao_merged <= kGateMaxAbsO;
        const bool post_ok = al_local <= kGateMaxAbsLse && al_merged <= kGateMaxAbsLse;
        // body levels are only populated under F4_RP_OBS; when present they must
        // be exact (same operands, same kernel, same run -- no merge in between).
        const bool body_ok = (a_bd.empty() && a_bdl.empty()) ||
            (u_bo == 0 && u_bl == 0);
        const bool num_ok  = pre_ok && post_ok && body_ok;
        const bool same = desc_ok && num_ok;
        g_case = tag;
        std::printf("  [%s] A/A2 repeat (independent rebuilds): desc_equal=%d "
                    "ulp{bodyO=%llu bodyLSE=%llu localO=%llu localLSE=%llu "
                    "mergedO=%llu mergedLSE=%llu} "
                    "gate{nrmseO_local=%.3e nrmseO_merged=%.3e vs %.1e "
                    "maxO_merged=%.3e vs %.1e maxLSE_local=%.3e maxLSE_merged=%.3e "
                    "vs %.1e} storage=F32 (%zu O, %zu LSE) -> %s\n",
                    tag.c_str(), (int) desc_ok,
                    (unsigned long long) u_bo, (unsigned long long) u_bl,
                    (unsigned long long) u_lo, (unsigned long long) u_ll,
                    (unsigned long long) u_o, (unsigned long long) u_l,
                    no_local, no_merged, kGateNrmseO,
                    ao_merged, kGateMaxAbsO,
                    al_local, al_merged, kGateMaxAbsLse,
                    a_o.size(), a_l.size(), same ? "ok" : "FAIL");
        if (!num_ok || !desc_ok) {
            if (woo < a_o.size() && woo < g_last_o.size())
                std::printf("      merged O worst: i=%zu got=%.9g ref=%.9g ulp=%llu\n",
                            woo, a_o[woo], g_last_o[woo],
                            (unsigned long long) ulp_dist(a_o[woo], g_last_o[woo]));
            if (wol < a_l.size() && wol < g_last_lse.size())
                std::printf("      merged LSE worst: i=%zu got=%.9g ref=%.9g ulp=%llu\n",
                            wol, a_l[wol], g_last_lse[wol],
                            (unsigned long long) ulp_dist(a_l[wol], g_last_lse[wol]));
            std::printf("      raw maxabs mergedO=%.9g mergedLSE=%.9g (informational)\n",
                        mag(a_o, g_last_o), mag(a_l, g_last_lse));
        }
        CHECK(same);
        CHECK(!a_o.empty() && !a_l.empty());
    }

    std::printf("real-pair O/LSE: %d checks, %d failures, %d not-executed\n",
                g_checks, g_failures, g_skipped);
    ggml_backend_free(env.cuda);
    ggml_backend_free(env.vk);
    ggml_backend_free(env.cpu);
    return g_failures == 0 ? 0 : 1;
}
