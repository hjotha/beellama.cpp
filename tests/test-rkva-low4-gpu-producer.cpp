// Opt-in real CUDA ABI/owner qualification. No opt-in is a CTest skip, not a pass.
#undef NDEBUG
#include "../ggml/src/ggml-remote-attn/rkva-low4-producer.h"
#include "../ggml/src/ggml-remote-attn/rkva-saw-int4-producer.h"
#include "../ggml/src/ggml-remote-attn/rkva-turboquant-nc-producer.h"
#include "ggml-backend.h"
#include "ggml-low4-native-producer.h"
#include "ggml.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

static void mode(const char * value) {
    for (const char * name : { "GGML_RKVA_LOW4_PRODUCER", "GGML_RKVA_SAW_PRODUCER", "GGML_RKVA_TQNC_PRODUCER" }) {
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }
}

static std::unique_ptr<rkva_native_owner> owner(int format) {
    if (format == GGML_TYPE_SAW_INT4) {
        return std::unique_ptr<rkva_native_owner>(new rkva_saw_int4_producer(0, 512, 2));
    }
    if (format == GGML_TYPE_TQNC_K) {
        return std::unique_ptr<rkva_native_owner>(new rkva_turboquant_nc_producer(0, 512, 2));
    }
    return std::unique_ptr<rkva_native_owner>(new rkva_low4_producer(0, 512, 2, ggml_type(format)));
}

int main() {
    const char * enable = std::getenv("GGML_RKVA_GPU_TEST");
    if (!enable || std::strcmp(enable, "1")) {
        std::puts("SKIP: set GGML_RKVA_GPU_TEST=1 for exclusive real CUDA qualification");
        return 77;
    }
    ggml_backend_load_all();
    auto reg = ggml_backend_reg_by_name("CUDA");
    assert(reg);
    auto convert = reinterpret_cast<ggml_backend_cuda_low4_native_prepare_v1_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_low4_native_prepare_v1"));
    assert(convert);
    assert(ggml_backend_reg_dev_count(reg) > 0);
    auto backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    assert(backend);
    for (int format : { GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL, GGML_TYPE_SAW_INT4, GGML_TYPE_TQNC_K }) {
        for (uint32_t count : { 1u, 3u, 128u, 512u }) {
            const size_t width = 512, record = format == GGML_TYPE_Q4_1     ? 160 :
                                               format == GGML_TYPE_IQ4_NL   ? 144 :
                                               format == GGML_TYPE_SAW_INT4 ? 136 :
                                                                              132;
            uint32_t     flags = 0;
#if defined(__AVX2__)
            if (format == GGML_TYPE_SAW_INT4) {
                flags = 1;
            }
#endif
            std::vector<float>   k(count * width), v(k.size());
            std::vector<int32_t> pos(count);
            uint32_t             rng = 0x51b9287;
            for (size_t i = 0; i < k.size(); ++i) {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                k[i] = float(int(rng & 65535) - 32768) / 8192;
                v[i] = float(int((rng >> 16) & 65535) - 32768) / 4096;
            }
            for (size_t i = 0; i < 256; ++i) {
                k[i] = 0;
                v[i] = i & 1 ? 0.5f : -0.5f;
            }
            // Gradual-underflow and weighted-IQ boundaries exercise the precise TU.
            if (count >= 3) {
                const float tiny[] = { 1e-38f, 1e-30f, 1e-20f, 1e-15f };
                for (int h = 0; h < 4; ++h) {
                    for (int j = 0; j < 256; ++j) {
                        k[(h + 1) * 256 + j] = (j - 128) * tiny[h];
                        v[(h + 1) * 256 + j] = (127 - j) * tiny[h];
                    }
                }
            }
            for (uint32_t i = 0; i < count; ++i) {
                pos[i] = i;
            }
            mode("cpu");
            auto                 cpu = owner(format);
            std::vector<uint8_t> reference;
            assert(cpu->append(k.data(), v.data(), pos.data(), count, true, reference));
            const size_t         plane = count * 2 * record;
            std::vector<uint8_t> pk(plane), pv(plane);
            assert(convert(format, flags, k.data(), v.data(), count, 2, pk.data(), pv.data()));
            assert(!std::memcmp(pk.data(), reference.data() + 32, plane));
            assert(!std::memcmp(pv.data(), reference.data() + 32 + plane, plane));
            auto ctx = ggml_init({ 2u * 1024 * 1024, nullptr, true });
            assert(ctx);
            auto dk     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k.size()),
                 dv     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, v.size());
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            assert(buffer);
            ggml_backend_tensor_set(dk, k.data(), 0, k.size() * sizeof(float));
            ggml_backend_tensor_set(dv, v.data(), 0, v.size() * sizeof(float));
            ggml_backend_synchronize(backend);
            assert(convert(format, flags, static_cast<const float *>(dk->data), static_cast<const float *>(dv->data),
                           count, 2, pk.data(), pv.data()));
            assert(!std::memcmp(pk.data(), reference.data() + 32, plane));
            assert(!std::memcmp(pv.data(), reference.data() + 32 + plane, plane));
            mode("cuda");
            auto gpu = owner(format);
            assert(gpu->prepare(k.data(), v.data(), pos.data(), count, static_cast<const float *>(dk->data),
                                static_cast<const float *>(dv->data)));
            mode("cpu");
            auto                 chunked = owner(format);
            std::vector<uint8_t> a, b;
            for (uint32_t at = 0; at < count; ++at) {
                assert(gpu->append(k.data() + at * width, v.data() + at * width, pos.data() + at, 1, at == 0, a));
                assert(chunked->append(k.data() + at * width, v.data() + at * width, pos.data() + at, 1, at == 0, b));
                assert(a == b);
            }
            assert(gpu->rewind(0));
            auto saved = a;
            assert(!gpu->append(k.data(), v.data(), pos.data(), 1, true, a) && a == saved && gpu->rows() == 0);
            assert(gpu->prepare(k.data(), v.data(), pos.data(), count));
            float old = k[0];
            k[0]      = std::numeric_limits<float>::infinity();
            assert(!convert(format, flags, k.data(), v.data(), count, 2, pk.data(), pv.data()));
            assert(!gpu->prepare(k.data(), v.data(), pos.data(), count));
            k[0] = old;
            assert(!gpu->append(k.data(), v.data(), pos.data(), 1, true, a) && a == saved && gpu->rows() == 0);
            assert(!convert(format, flags, k.data(), v.data(), 513, 2, pk.data(), pv.data()));
            assert(!convert(format, 2, k.data(), v.data(), count, 2, pk.data(), pv.data()));
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            std::printf("PASS real CUDA format=%d tokens=%u host/device bytes, invalids, chunks and rewind\n", format,
                        count);
        }
    }
    ggml_backend_free(backend);
}
