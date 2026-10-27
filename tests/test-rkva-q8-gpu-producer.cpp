// Opt-in Q8_0 producer byte parity. Run only with exclusive GPU ownership.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-q8-native-producer.h"
#include "../ggml/src/ggml-remote-attn/rkva-q8-producer.h"
#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main() {
    if (!std::getenv("GGML_RKVA_GPU_TEST")) { std::puts("SKIP: set GGML_RKVA_GPU_TEST=1"); return 0; }
    ggml_backend_load_all();
    auto reg = ggml_backend_reg_by_name("CUDA");
    assert(reg);
    auto convert = reinterpret_cast<ggml_backend_cuda_q8_native_prepare_v1_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_q8_native_prepare_v1"));
    assert(convert);
    assert(ggml_backend_reg_dev_count(reg) > 0);
    auto backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    assert(backend);
    constexpr uint32_t heads = 4;
    constexpr size_t width = heads * 256, packed_width = heads * 272;
    uint32_t rng = 0x91a73e2u;
    for (uint32_t count : {1u, 8u, 128u, 256u}) {
        std::vector<float> k(size_t(count)*width), v(k.size());
        for (size_t i = 0; i < k.size(); ++i) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            k[i] = float(int(rng & 0xffff) - 32768) / 4096.0f;
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            v[i] = float(int(rng & 0xffff) - 32768) / 8192.0f;
        }
        // Zero blocks, exact ties and high/low magnitudes exercise scale/rounding.
        for (size_t i = 0; i < 32; ++i) { k[i] = 0.0f; v[i] = (i & 1) ? 0.5f : -0.5f; }
        std::vector<uint8_t> refk(size_t(count)*packed_width), refv(refk.size()), gotk(refk.size()), gotv(refk.size());
        assert(ggml_quantize_chunk(GGML_TYPE_Q8_0, k.data(), refk.data(), 0, int64_t(count)*heads, 256, nullptr) == refk.size());
        assert(ggml_quantize_chunk(GGML_TYPE_Q8_0, v.data(), refv.data(), 0, int64_t(count)*heads, 256, nullptr) == refv.size());
        assert(convert(k.data(), v.data(), count, heads, gotk.data(), gotv.data()));
        if (gotk != refk || gotv != refv) {
            for (size_t i = 0; i < refk.size(); ++i) if (gotk[i] != refk[i]) {
                std::fprintf(stderr, "count=%u key_byte=%zu block=%zu byte=%zu cpu=%u gpu=%u\n", count, i, i / 34, i % 34, unsigned(refk[i]), unsigned(gotk[i]));
                const size_t block = i / 34;
                for (size_t j = 0; j < 32; ++j) std::fprintf(stderr, " %.9g", k[block*32+j]);
                std::fprintf(stderr, "\n"); break;
            }
            for (size_t i = 0; i < refv.size(); ++i) if (gotv[i] != refv[i]) {
                std::fprintf(stderr, "count=%u value_byte=%zu block=%zu byte=%zu cpu=%u gpu=%u\n", count, i, i / 34, i % 34, unsigned(refv[i]), unsigned(gotv[i])); break;
            }
        }
        assert(gotk == refk && gotv == refv);
        auto ctx = ggml_init({2u*1024*1024, nullptr, true}); assert(ctx);
        auto dk = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k.size());
        auto dv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, v.size());
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); assert(buffer);
        ggml_backend_tensor_set(dk, k.data(), 0, k.size()*sizeof(float));
        ggml_backend_tensor_set(dv, v.data(), 0, v.size()*sizeof(float));
        assert(convert(static_cast<const float *>(dk->data), static_cast<const float *>(dv->data), count, heads, gotk.data(), gotv.data()));
        assert(gotk == refk && gotv == refv);
        auto mode = [](const char * value) {
#ifdef _WIN32
            _putenv_s("GGML_RKVA_Q8_PRODUCER", value);
#else
            setenv("GGML_RKVA_Q8_PRODUCER", value, 1);
#endif
        };
        mode("cpu"); rkva_q8_producer cpu(1024, 512, heads);
        mode("cuda"); rkva_q8_producer gpu(1024, 512, heads);
        std::vector<int32_t> positions(count);
        for (uint32_t i = 0; i < count; ++i) positions[i] = 1024 + i;
        assert(cpu.prepare(k.data(), v.data(), positions.data(), count));
        assert(gpu.prepare(k.data(), v.data(), positions.data(), count,
                           static_cast<const float *>(dk->data), static_cast<const float *>(dv->data)));
        std::vector<uint8_t> a, b;
        for (uint32_t at = 0; at < count; at += 8) {
            const uint32_t n = std::min(8u, count - at);
            assert(cpu.append(k.data() + at*width, v.data() + at*width, positions.data() + at, n, at == 0, a));
            assert(gpu.append(k.data() + at*width, v.data() + at*width, positions.data() + at, n, at == 0, b));
            assert(a == b);
        }
        assert(cpu.rows() == count && gpu.rows() == count);
        assert(gpu.rewind(0));
        assert(!gpu.append(k.data(), v.data(), positions.data(), 1, true, b));
        mode("cpu");
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
        std::printf("Q8 GPU producer: %u tokens, host and device inputs byte-identical\n", count);
    }
    ggml_backend_free(backend);
    return 0;
}
