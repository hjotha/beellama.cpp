// F1 — LSE export on the CPU backend + split/merge math on real FA outputs.
//
// Covers plan §3.4 (contract) and §4.1 (structural) on CPU:
//   Q=1 (decode: one-chunk + split-KV paths) and prefill Q=2/32 (tiled path),
//   GQA (4 q-heads / 2 kv-heads), D=64 + one D=256 case, F16 and Q4_0 KV,
//   causal mask, full mask, fully-masked row (O=0/LSE=-inf), and end-to-end
//   split(FA[0,k1) + FA[k1,n)) + merge vs single FA reference in FP64.
//
// KVarN-native and body+tail LSE paths share the same write helper and are
// covered in F3 integration (4B P-small run with tail active); their CPU
// write sites were added together with these.
//
#include "ggml.h"
#include "ggml-cpu.h"

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

struct fa_case {
    int D = 64;
    int nq = 1;       // queries
    int nkv = 16;     // kv length
    int nqh = 4;      // q heads
    int nkh = 2;      // kv heads (GQA)
    ggml_type kv_type = GGML_TYPE_F16;
    bool causal = true;
    bool fully_masked_row = false; // one query row fully masked -> O=0/-inf
};

// Tensor-flat index helpers (contiguous GGML order).
// q: ne={D,nq,nqh} -> ((h*nq)+q)*D+d ; kv: ne={D,nkv,nkh} -> ((kh*nkv)+k)*D+d
// o (dst): ne={D,nqh,nq} -> ((q*nqh)+h)*D+d ; lse: ne={nqh,nq} -> q*nqh+h
static inline size_t qi(const fa_case & c, int q, int h, int d) {
    return ((size_t) h * c.nq + q) * c.D + d;
}
static inline size_t ki(const fa_case & c, int k, int kh, int d) {
    return ((size_t) kh * c.nkv + k) * c.D + d;
}
static inline size_t oi(const fa_case & c, int q, int h, int d) {
    return ((size_t) q * c.nqh + h) * c.D + d;
}

// FP64 exact reference: O[q][h][d], LSE[q][h].
static void reference_fp64(const fa_case & c,
        const std::vector<float> & Q, const std::vector<float> & K, const std::vector<float> & V,
        std::vector<float> & Oref, std::vector<double> & LSEref) {
    const int D = c.D, nq = c.nq, nkv = c.nkv, nqh = c.nqh, nkh = c.nkh;
    const int gqa = nqh / nkh;
    Oref.assign((size_t) D * nqh * nq, 0.0f);
    LSEref.assign((size_t) nqh * nq, -INFINITY);
    for (int q = 0; q < nq; ++q) {
        for (int h = 0; h < nqh; ++h) {
            const int kh = h / gqa;
            std::vector<double> s(nkv, -INFINITY);
            for (int k = 0; k < nkv; ++k) {
                bool masked = false;
                if (c.fully_masked_row && q == nq - 1) {
                    masked = true;
                } else if (c.causal && k > q) {
                    masked = true; // toy causal: kv position k visible iff k <= q
                }
                if (masked) {
                    continue;
                }
                double dot = 0;
                for (int d = 0; d < D; ++d) {
                    dot += double(Q[qi(c, q, h, d)]) * double(K[ki(c, k, kh, d)]);
                }
                s[k] = dot; // scale 1.0
            }
            double m = -INFINITY;
            for (double v : s) {
                m = std::max(m, v);
            }
            if (m == -INFINITY) {
                continue; // O=0, LSE=-inf
            }
            double denom = 0;
            for (double v : s) {
                denom += std::exp(v - m);
            }
            LSEref[(size_t) q * nqh + h] = m + std::log(denom);
            for (int d = 0; d < D; ++d) {
                double acc = 0;
                for (int k = 0; k < nkv; ++k) {
                    if (s[k] == -INFINITY) {
                        continue;
                    }
                    acc += std::exp(s[k] - m) * double(V[ki(c, k, kh, d)]);
                }
                Oref[oi(c, q, h, d)] = float(acc / denom);
            }
        }
    }
}

// Effective Q used by the kernel dot (plan §4.1: the structural reference
// uses the same representation as the kernel under test, isolating math
// error from representation error). Verified empirically: the CPU Q4_0 dot
// quantizes Q rows to Q8_0 (kernel LSE matches Q8-Q FP64 reference to 5
// decimals); the F16 dot converts Q to F16.
static void effective_q(const fa_case & c, const std::vector<float> & Q, std::vector<float> & Qeff) {
    Qeff = Q;
    if (c.kv_type == GGML_TYPE_Q4_0) {
        std::vector<char> blk(68);
        for (size_t r = 0; r < Q.size() / c.D; ++r) {
            ggml_quantize_chunk(GGML_TYPE_Q8_0, &Q[r * c.D], blk.data(), 0, 1, c.D, NULL);
            for (int j = 0; j < c.D; ++j) {
                const uint8_t * b = (const uint8_t *) blk.data() + (j / 32) * 34;
                const float d = ggml_fp16_to_fp32(*(const ggml_fp16_t *) b);
                Qeff[r * c.D + j] = float(((const int8_t *) b)[2 + j % 32]) * d;
            }
        }
    } else if (c.kv_type == GGML_TYPE_F16) {
        for (size_t i = 0; i < Q.size(); ++i) {
            Qeff[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(Q[i]));
        }
    }
}

// Dequantize one row of the test KV tensors back to F32 (plan §4.1: the
// structural reference uses the dequantized KV of the same representation,
// isolating merge/math error from format error).
static void dequant_row(ggml_type type, const void * row, float * out, int D) {
    if (type == GGML_TYPE_F16) {
        const ggml_fp16_t * h = (const ggml_fp16_t *) row;
        for (int d = 0; d < D; ++d) {
            out[d] = ggml_fp16_to_fp32(h[d]);
        }
        return;
    }
    if (type == GGML_TYPE_Q4_0) {
        // block_q4_0: 32 values per block: d (f16) + qs[16]; values j<16 use
        // the low nibble of qs[j], values j>=16 the high nibble of qs[j-16].
        const uint8_t * b = (const uint8_t *) row;
        for (int j = 0; j < D; ++j) {
            const uint8_t * blk = b + (j / 32) * 18;
            const float d = ggml_fp16_to_fp32(*(const ggml_fp16_t *) blk);
            const uint8_t byte = blk[2 + (j % 32) % 16];
            const uint8_t nib = ((j % 32) < 16) ? (byte & 0x0F) : (byte >> 4);
            out[j] = (float(int(nib)) - 8.0f) * d;
        }
        return;
    }
    CHECK(false); // unsupported KV type in this test
}

// Run one FA op (with LSE attached) on CPU via legacy graph compute.
static bool run_fa_cpu(const fa_case & c, int kv_len, int kv_offset,
        const std::vector<float> & Qf, const std::vector<float> & Kf, const std::vector<float> & Vf,
        std::vector<float> & Kd, std::vector<float> & Vd,
        std::vector<float> & O, std::vector<float> & LSE) {
    const int D = c.D, nq = c.nq, nqh = c.nqh, nkh = c.nkh;
    struct ggml_init_params params = { 256 * 1024 * 1024, NULL, false };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, nq, nqh, 1);
    ggml_tensor * kt = ggml_new_tensor_4d(ctx, c.kv_type, D, kv_len, nkh, 1);
    ggml_tensor * vt = ggml_new_tensor_4d(ctx, c.kv_type, D, kv_len, nkh, 1);
    // mask F16 [nkv, nq] causal toy: row q allows kv k<=q+offset... build full then mask
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len, nq);
    // copy data
    memcpy(q->data, Qf.data(), Qf.size() * sizeof(float));
    // quantize K/V rows; Kd/Vd receive the dequantized rows actually used
    std::vector<float> krow(D), vrow(D);
    Kd.assign(Kf.size(), 0.0f);
    Vd.assign(Vf.size(), 0.0f);
    const size_t qrow_bytes = ggml_row_size(c.kv_type, D);
    for (int kk = 0; kk < kv_len; ++kk) {
        for (int kh = 0; kh < nkh; ++kh) {
            for (int d = 0; d < D; ++d) {
                krow[d] = Kf[((size_t) kh * c.nkv + (kk + kv_offset)) * D + d];
                vrow[d] = Vf[((size_t) kh * c.nkv + (kk + kv_offset)) * D + d];
            }
            char * kdst = (char *) kt->data + (kh * kv_len + kk) * qrow_bytes;
            char * vdst = (char *) vt->data + (kh * kv_len + kk) * qrow_bytes;
            ggml_quantize_chunk(c.kv_type, krow.data(), kdst, 0, 1, D, NULL);
            ggml_quantize_chunk(c.kv_type, vrow.data(), vdst, 0, 1, D, NULL);
            dequant_row(c.kv_type, kdst, &Kd[((size_t) kh * c.nkv + (kk + kv_offset)) * D], D);
            dequant_row(c.kv_type, vdst, &Vd[((size_t) kh * c.nkv + (kk + kv_offset)) * D], D);
        }
    }
    // mask: F16, row q (ne1), col k (ne0); -inf where masked
    for (int qq = 0; qq < nq; ++qq) {
        for (int kk = 0; kk < kv_len; ++kk) {
            bool masked = false;
            const int kabs = kk + kv_offset;
            if (c.fully_masked_row && qq == nq - 1) {
                masked = true;
            } else if (c.causal && kabs > qq) {
                masked = true;
            }
            ggml_fp16_t mv = masked ? ggml_fp32_to_fp16(-INFINITY) : ggml_fp32_to_fp16(0.0f);
            // mask ne = {kv_len, nq}: element (k, q)
            *(ggml_fp16_t *) ((char *) mask->data + kk * mask->nb[0] + qq * mask->nb[1]) = mv;
        }
    }
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, kt, vt, mask, 1.0f, 0.0f, 0.0f);
    ggml_tensor * lse = ggml_flash_attn_ext_lse_out(ctx, fa);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, lse);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    O.assign((const float *) fa->data, (const float *) fa->data + ggml_nelements(fa));
    LSE.assign((const float *) lse->data, (const float *) lse->data + ggml_nelements(lse));
    ggml_free(ctx);
    return true;
}

static double nrmse_vec(const std::vector<float> & o, const std::vector<float> & ref) {
    double se = 0, sr = 0;
    for (size_t i = 0; i < o.size(); ++i) {
        double d = double(o[i]) - double(ref[i]);
        se += d * d;
        sr += double(ref[i]) * double(ref[i]);
    }
    double rr = std::sqrt(sr / o.size());
    if (rr < 1e-6) {
        rr = 1e-6;
    }
    return std::sqrt(se / o.size()) / rr;
}

static void test_one(const fa_case & c, const char * name) {
    g_current = name;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> Q((size_t) c.nq * c.nqh * c.D), K((size_t) c.nkv * c.nkh * c.D), V((size_t) c.nkv * c.nkh * c.D);
    for (auto & x : Q) {
        x = dist(rng);
    }
    for (auto & x : K) {
        x = dist(rng);
    }
    for (auto & x : V) {
        x = dist(rng);
    }
    std::vector<float> Oref;
    std::vector<double> LSEref;
    std::vector<float> Kd, Vd, O, LSE;
    CHECK(run_fa_cpu(c, c.nkv, 0, Q, K, V, Kd, Vd, O, LSE));
    if (g_failures) {
        return;
    }
    std::vector<float> Qeff;
    effective_q(c, Q, Qeff);
    reference_fp64(c, Qeff, Kd, Vd, Oref, LSEref);
    CHECK(O.size() == Oref.size());
    CHECK(LSE.size() == LSEref.size());
    CHECK(nrmse_vec(O, Oref) <= 1e-3);
    float max_abs_o = 0, max_abs_lse = 0;
    for (size_t i = 0; i < O.size(); ++i) {
        max_abs_o = std::max(max_abs_o, std::fabs(O[i] - Oref[i]));
        if (std::isfinite(O[i]) == false) {
            CHECK(false);
            break;
        }
    }
    for (size_t i = 0; i < LSE.size(); ++i) {
        if (std::isinf(LSEref[i]) && LSEref[i] < 0) {
            CHECK(std::isinf(LSE[i]) && LSE[i] < 0); // exact -inf
            // corresponding O row must be exactly 0
            for (int d = 0; d < c.D; ++d) {
                CHECK(O[i * c.D + d] == 0.0f);
            }
        } else {
            max_abs_lse = std::max(max_abs_lse, std::fabs(LSE[i] - float(LSEref[i])));
            if (std::isfinite(LSE[i]) == false) {
                CHECK(false);
                break;
            }
        }
    }
    CHECK(max_abs_o <= 1e-2f);
    CHECK(max_abs_lse <= 1e-2f);
}

static void test_merge_e2e() {
    g_current = "merge_e2e";
    fa_case c;
    c.nq = 4;
    c.nkv = 12;
    c.kv_type = GGML_TYPE_F16;
    c.causal = false;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> Q((size_t) c.nq * c.nqh * c.D), K((size_t) c.nkv * c.nkh * c.D), V((size_t) c.nkv * c.nkh * c.D);
    for (auto & x : Q) {
        x = dist(rng);
    }
    for (auto & x : K) {
        x = dist(rng);
    }
    for (auto & x : V) {
        x = dist(rng);
    }
    // split at k1=5: FA1 over [0,5), FA2 over [5,12). Note: with causal=false
    // both ranges are valid subsets; merge must reproduce full.
    const int k1 = 5;
    std::vector<float> Kd1, Vd1, O1, LSE1, Kd2, Vd2, O2, LSE2;
    CHECK(run_fa_cpu(c, k1, 0, Q, K, V, Kd1, Vd1, O1, LSE1));
    CHECK(run_fa_cpu(c, c.nkv - k1, k1, Q, K, V, Kd2, Vd2, O2, LSE2));
    if (g_failures) {
        return;
    }
    // full reference over the spliced dequantized KV (same representation)
    std::vector<float> Oref;
    std::vector<double> LSEref;
    {
        std::vector<float> Kd = Kd1, Vd = Vd1;
        for (int kh = 0; kh < c.nkh; ++kh) {
            for (int k = k1; k < c.nkv; ++k) {
                for (int d = 0; d < c.D; ++d) {
                    Kd[((size_t) kh * c.nkv + k) * c.D + d] = Kd2[((size_t) kh * c.nkv + k) * c.D + d];
                    Vd[((size_t) kh * c.nkv + k) * c.D + d] = Vd2[((size_t) kh * c.nkv + k) * c.D + d];
                }
            }
        }
        std::vector<float> Qeff;
        effective_q(c, Q, Qeff);
        reference_fp64(c, Qeff, Kd, Vd, Oref, LSEref);
    }
    // merge per (q,h)
    std::vector<float> OM((size_t) c.nq * c.nqh * c.D, 0.0f);
    for (int q = 0; q < c.nq; ++q) {
        for (int h = 0; h < c.nqh; ++h) {
            const size_t li = (size_t) q * c.nqh + h;
            const float l1 = LSE1[li], l2 = LSE2[li];
            const float m = std::max(l1, l2);
            const float w1 = std::exp(l1 - m), w2 = std::exp(l2 - m);
            for (int d = 0; d < c.D; ++d) {
                const size_t oi = li * c.D + d;
                OM[oi] = (O1[oi] * w1 + O2[oi] * w2) / (w1 + w2);
            }
        }
    }
    CHECK(nrmse_vec(OM, Oref) <= 1e-3);
    float mx = 0;
    for (size_t i = 0; i < OM.size(); ++i) {
        mx = std::max(mx, std::fabs(OM[i] - Oref[i]));
    }
    CHECK(mx <= 1e-2f);
}

int main() {
    fa_case c;
    test_one(c, "decode_q1_f16_causal");
    c.kv_type = GGML_TYPE_Q4_0;
    test_one(c, "decode_q1_q4_causal");
    c.kv_type = GGML_TYPE_F16;
    c.causal = false;
    test_one(c, "decode_q1_f16_full");
    c.nq = 2;
    c.causal = true;
    test_one(c, "prefill_q2_f16_causal");
    c.nq = 32;
    c.nkv = 48;
    test_one(c, "prefill_q32_f16_causal");
    c.nq = 1;
    c.nkv = 16;
    c.fully_masked_row = true;
    test_one(c, "decode_q1_fully_masked");
    c.fully_masked_row = false;
    c.D = 256;
    test_one(c, "decode_q1_d256_f16");
    test_merge_e2e();
    std::printf("position-split-lse: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
