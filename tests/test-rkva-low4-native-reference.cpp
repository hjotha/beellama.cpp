#undef NDEBUG
// CPU execution of the exact scalar routines used by the CUDA kernels, compared
// with independent production CPU codecs. This is NOT a CUDA runtime test.
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-cuda/native-low4-reference.cuh"
#include "../ggml/src/ggml-remote-attn/rkva-low4-producer.h"
#include "../ggml/src/ggml-remote-attn/rkva-saw-int4-producer.h"
#include "../ggml/src/ggml-remote-attn/rkva-turboquant-nc-producer.h"
#include "../ggml/src/ggml-turboquant-nc.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

static void env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

static bool reject_prepare = false;

static bool fake_prepare(int           format,
                         uint32_t      flags,
                         const float * k,
                         const float * v,
                         uint32_t      n,
                         uint32_t      heads,
                         void *        pk,
                         void *        pv) {
    if (reject_prepare) {
        return false;
    }
    size_t record = format == GGML_TYPE_Q4_1     ? 160 :
                    format == GGML_TYPE_IQ4_NL   ? 144 :
                    format == GGML_TYPE_SAW_INT4 ? 136 :
                                                   132;
    for (size_t i = 0; i < size_t(n) * heads; ++i) {
        if (!rkva_native_low4::encode(format, k + 256 * i, static_cast<uint8_t *>(pk) + record * i, true, flags) ||
            !rkva_native_low4::encode(format, v + 256 * i, static_cast<uint8_t *>(pv) + record * i, false, flags)) {
            return false;
        }
    }
    return true;
}

static const char * fake_name(ggml_backend_reg_t) {
    return "CUDA";
}

static size_t fake_count(ggml_backend_reg_t) {
    return 0;
}

static void * fake_proc(ggml_backend_reg_t, const char * name) {
    return std::strcmp(name, "ggml_backend_cuda_low4_native_prepare_v1") ? nullptr :
                                                                           reinterpret_cast<void *>(fake_prepare);
}

int main() {
    ggml_init_params params{ 1024 * 1024, nullptr, false };
    auto *           ctx = ggml_init(params);
    assert(ctx);
    std::mt19937                    rng(20261107);
    std::normal_distribution<float> normal;
    size_t                          cases = 0;
    for (int t = 0; t < 2048; ++t) {
        float x[256];
        for (int j = 0; j < 256; ++j) {
            x[j] = normal(rng) * (t % 8 == 0 ? 1e-6f : t % 8 == 1 ? 100.0f : 1.0f);
            if (t < 8) {
                x[j] = t == 0 ? 0 :
                       t == 1 ? -0.0f :
                       t == 2 ? 1 :
                       t == 3 ? -1 :
                       t == 4 ? float(j - 128) :
                       t == 5 ? (j == 127 ? 100.0f : 0) :
                       t == 6 ? 1e-16f :
                                -1e-16f;
            }
        }
        if (t >= 8 && t < 16) {
            const float tiny[] = { 1e-38f, 1e-35f, 1e-30f, 1e-20f, 1e-15f, 1e-14f, 1e-10f, 1e-5f };
            for (int j = 0; j < 256; ++j) {
                x[j] = (j - 128) * tiny[t - 8];
            }
        }
        for (int format : { GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL, GGML_TYPE_SAW_INT4, GGML_TYPE_TQNC_K }) {
            for (bool key : { true, false }) {
                uint8_t ref[160] = {}, got[160] = {};
                size_t  n = 0;
                if (format == GGML_TYPE_Q4_1 || format == GGML_TYPE_IQ4_NL) {
                    n = ggml_quantize_chunk(ggml_type(format), x, ref, 0, 1, 256, nullptr);
                } else if (format == GGML_TYPE_SAW_INT4) {
                    block_saw_int4 b;
                    rkva_saw_int4::quantize_row_saw_int4(x, &b, 256);
                    n = sizeof b;
                    std::memcpy(ref, &b, n);
                } else if (key) {
                    ggml_tqnc_key b;
                    assert(ggml_tqnc_quantize_key(x, &b));
                    std::memcpy(ref, &b, sizeof b);
                    n = 132;
                } else {
                    ggml_tqnc_value b;
                    assert(ggml_tqnc_quantize_value(x, &b));
                    std::memcpy(ref, &b, sizeof b);
                    n = 132;
                }
                uint32_t flags = 0;
#if defined(__AVX2__)
                if (format == GGML_TYPE_SAW_INT4) {
                    flags = 1;
                }
#endif
                if (!rkva_native_low4::encode(format, x, got, key, flags) || std::memcmp(ref, got, n)) {
                    std::fprintf(stderr, "mismatch case=%d format=%d key=%d\n", t, format, key);
                    return 1;
                }
                ++cases;
            }
        }
    }
    float   bad[256] = {};
    uint8_t dst[160];
    for (int format : { GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL, GGML_TYPE_SAW_INT4, GGML_TYPE_TQNC_K }) {
        for (float invalid : { std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN() }) {
            bad[127] = invalid;
            assert(!rkva_native_low4::encode(format, bad, dst, true));
            assert(!rkva_native_low4::encode(format, bad, dst, false));
        }
    }
    bad[127] = 65505;
    assert(!rkva_native_low4::encode(GGML_TYPE_Q4_1, bad, dst, true));
    assert(!rkva_native_low4::encode(GGML_TYPE_IQ4_NL, bad, dst, true));
    bad[127] = 1e30f;
    assert(!rkva_native_low4::encode(GGML_TYPE_TQNC_K, bad, dst, true));
    bad[126] = bad[127] = std::numeric_limits<float>::max();
    assert(!rkva_native_low4::encode(GGML_TYPE_SAW_INT4, bad, dst, true));
    std::memset(bad, 0, sizeof(bad));
    bad[0] = 16 * std::numeric_limits<float>::denorm_min();
    assert(!rkva_native_low4::encode(GGML_TYPE_Q4_1, bad, dst, true));
    if (!ggml_backend_reg_by_name("CUDA")) {
        // Generic CUDA still rejects LOW4; SAW/NC preserve historical CPU
        // behavior with a warning. Explicit format CUDA must fail without its helper.
        env("GGML_RKVA_NATIVE_PRODUCER", "cuda");
        env("GGML_RKVA_LOW4_PRODUCER", nullptr);
        env("GGML_RKVA_SAW_PRODUCER", nullptr);
        env("GGML_RKVA_TQNC_PRODUCER", nullptr);
        auto rejects = [](auto make) {
            bool threw = false;
            try {
                make();
            } catch (const std::exception &) {
                threw = true;
            }
            assert(threw);
        };
        rejects([] { rkva_low4_producer p(0, 512, 1, GGML_TYPE_Q4_1); });
        rejects([] { rkva_low4_producer p(0, 512, 1, GGML_TYPE_IQ4_NL); });
        rkva_saw_int4_producer      historic_saw(0, 512, 1);
        rkva_turboquant_nc_producer historic_nc(0, 512, 1);
        env("GGML_RKVA_LOW4_PRODUCER", "cuda");
        env("GGML_RKVA_SAW_PRODUCER", "cuda");
        env("GGML_RKVA_TQNC_PRODUCER", "cuda");
        rejects([] { rkva_low4_producer p(0, 512, 1, GGML_TYPE_Q4_1); });
        rejects([] { rkva_low4_producer p(0, 512, 1, GGML_TYPE_IQ4_NL); });
        rejects([] { rkva_saw_int4_producer p(0, 512, 1); });
        rejects([] { rkva_turboquant_nc_producer p(0, 512, 1); });
        env("GGML_RKVA_LOW4_PRODUCER", nullptr);
        env("GGML_RKVA_SAW_PRODUCER", nullptr);
        env("GGML_RKVA_TQNC_PRODUCER", nullptr);
        env("GGML_RKVA_NATIVE_PRODUCER", "bogus");
        rejects([] { rkva_saw_int4_producer p(0, 512, 1); });
        rejects([] { rkva_turboquant_nc_producer p(0, 512, 1); });
        env("GGML_RKVA_NATIVE_PRODUCER", nullptr);
        rkva_low4_producer          p(0, 512, 1, GGML_TYPE_Q4_1);
        rkva_saw_int4_producer      s(0, 512, 1);
        rkva_turboquant_nc_producer n(0, 512, 1);
        // Register a test-only backend after unavailable-backend checks. This runs
        // owner transaction semantics, NOT CUDA execution or device-pointer tests.
        static ggml_backend_reg     fake{
            GGML_BACKEND_API_VERSION, { fake_name, fake_count, nullptr, fake_proc },
             nullptr
        };
        ggml_backend_register(&fake);
        env("GGML_RKVA_NATIVE_PRODUCER", "cuda");
        rkva_saw_int4_producer      historic_saw_with_backend(0, 512, 1);
        rkva_turboquant_nc_producer historic_nc_with_backend(0, 512, 1);
        rejects([] { rkva_low4_producer p(0, 512, 1, GGML_TYPE_Q4_1); });
        float   k[3 * 256], v[3 * 256];
        int32_t pos[3] = { 0, 1, 2 };
        for (int i = 0; i < 3 * 256; ++i) {
            k[i] = float(i % 37) / 37;
            v[i] = -float(i % 53) / 53;
        }
        env("GGML_RKVA_NATIVE_PRODUCER", "cuda");
        std::vector<uint8_t> historic_payload;
        assert(historic_saw.append(k, v, pos, 1, true, historic_payload));
        assert(historic_nc.append(k, v, pos, 1, true, historic_payload));
        assert(historic_saw_with_backend.append(k, v, pos, 1, true, historic_payload));
        assert(historic_nc_with_backend.append(k, v, pos, 1, true, historic_payload));
        env("GGML_RKVA_LOW4_PRODUCER", "cuda");
        env("GGML_RKVA_SAW_PRODUCER", "cuda");
        env("GGML_RKVA_TQNC_PRODUCER", "cuda");
        auto transaction = [&](rkva_native_owner & owner, size_t record) {
            std::vector<uint8_t> out{ 7, 8, 9 }, sentinel = out;
            assert(!owner.append(k, v, pos, 1, true, out) && out == sentinel && owner.rows() == 0);
            assert(owner.prepare(k, v, pos, 3));
            k[0] += 1;
            assert(!owner.append(k, v, pos, 1, true, out) && out == sentinel && owner.rows() == 0);
            k[0] -= 1;
            assert(owner.append(k, v, pos, 1, true, out));
            assert(out.size() == 32 + 2 * record);
            assert(owner.rows() == 1);
            assert(owner.append(k + 256, v + 256, pos + 1, 2, false, out));
            assert(owner.rows() == 3);
            assert(owner.rewind(1));
            sentinel = out;
            assert(!owner.append(k + 256, v + 256, pos + 1, 1, false, out) && out == sentinel && owner.rows() == 1);
            assert(owner.prepare(k + 256, v + 256, pos + 1, 2));
            reject_prepare = true;
            assert(!owner.prepare(k + 256, v + 256, pos + 1, 2));
            reject_prepare = false;
            assert(!owner.append(k + 256, v + 256, pos + 1, 1, false, out) && out == sentinel && owner.rows() == 1);
            assert(owner.prepare(k + 256, v + 256, pos + 1, 2));
            int32_t wrong = 99;
            assert(!owner.prepare(k + 256, v + 256, &wrong, 1));
            assert(!owner.append(k + 256, v + 256, pos + 1, 1, false, out) && out == sentinel && owner.rows() == 1);
            assert(owner.prepare(k + 256, v + 256, pos + 1, 2));
            assert(owner.append(k + 256, v + 256, pos + 1, 2, false, out) && owner.rows() == 3);
        };
        rkva_low4_producer          q41(0, 512, 1, GGML_TYPE_Q4_1), iq(0, 512, 1, GGML_TYPE_IQ4_NL);
        rkva_saw_int4_producer      saw(0, 512, 1);
        rkva_turboquant_nc_producer nc(0, 512, 1);
        transaction(q41, 160);
        transaction(iq, 144);
        transaction(saw, 136);
        transaction(nc, 132);
        // Format-specific CPU overrides must take precedence over generic CUDA.
        env("GGML_RKVA_LOW4_PRODUCER", "cpu");
        env("GGML_RKVA_SAW_PRODUCER", "cpu");
        env("GGML_RKVA_TQNC_PRODUCER", "cpu");
        reject_prepare = true;
        rkva_low4_producer          cpuq(0, 512, 1, GGML_TYPE_Q4_1);
        rkva_saw_int4_producer      cpus(0, 512, 1);
        rkva_turboquant_nc_producer cpun(0, 512, 1);
        std::vector<uint8_t>        out;
        assert(cpuq.append(k, v, pos, 1, true, out));
        assert(cpus.append(k, v, pos, 1, true, out));
        assert(cpun.append(k, v, pos, 1, true, out));
    }
    ggml_free(ctx);
    std::printf("PASS %zu byte-identical host reference comparisons; invalid input and absent-CUDA selection checks\n",
                cases);
}
