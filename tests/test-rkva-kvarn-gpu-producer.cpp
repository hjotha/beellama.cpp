// Optional CUDA-native producer parity and rollback qualification. Run with
// GGML_RKVA_GPU_TEST=1 and the CUDA backend library on its normal search path.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-remote-attn/rkva-kvarn-producer.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <string>

static void mode(const char * value) {
#ifdef _WIN32
    _putenv_s("GGML_RKVA_NATIVE_PRODUCER", value);
#else
    setenv("GGML_RKVA_NATIVE_PRODUCER", value, 1);
#endif
}

static float half_at(const uint8_t * p) {
    ggml_fp16_t h; std::memcpy(&h, p, sizeof(h)); return ggml_fp16_to_fp32(h);
}
static float record_value(const uint8_t * p, uint32_t row, uint32_t col) {
    const uint8_t byte = p[row * 64 + col / 2];
    const float q = (byte >> (4 * (col % 2))) & 15;
    return (q * half_at(p + 8192 + row * 2) + half_at(p + 8448 + row * 2)) * half_at(p + 8704 + col * 2);
}

static void test(uint32_t tail) {
    constexpr uint32_t origin = 1024, capacity = 768, heads = 4, total = 640;
    constexpr size_t width = heads * 256, record = 8960, physical = heads * 2;
    const auto store = reinterpret_cast<ggml_backend_cpu_kvarn_store_direct_v1_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(), "ggml_backend_cpu_kvarn_store_direct_v1"));
    assert(store);
    mode("cpu"); rkva_kvarn_producer cpu(origin, capacity, heads, 16, store, tail);
    mode("cuda"); rkva_kvarn_producer gpu(origin, capacity, heads, 16, store, tail);
    std::vector<float> k(total * width), v(k.size()); std::vector<int32_t> pos(total);
    uint32_t rng = 12345;
    for (uint32_t t = 0; t < total; ++t) {
        pos[t] = origin + t;
        for (size_t d = 0; d < width; ++d) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            const float noise = float(int(rng & 65535) - 32768) / 32768.0f;
            k[t * width + d] = 0.7f * noise + 0.5f * std::sin(float(t * 7 + d * 3) * 0.003f);
            v[t * width + d] = 0.9f * noise + 0.6f * std::cos(float(t * 11 + d * 5) * 0.007f);
        }
    }
    const char * fixture_dir = std::getenv("GGML_RKVA_GPU_PACKET_DIR");
    const std::string fixture = fixture_dir ? std::string(fixture_dir) + "/tail-" + std::to_string(tail) : "";
    const auto export_bytes = [&](const std::string & name, const void * data, size_t bytes) {
        if (!fixture_dir) return;
        std::ofstream out(fixture + name, std::ios::binary); assert(out);
        out.write(static_cast<const char *>(data), bytes); assert(out);
    };
    export_bytes("-original-k.f32", k.data(), k.size() * sizeof(float));
    export_bytes("-original-v.f32", v.data(), v.size() * sizeof(float));
    if (fixture_dir) {
        std::ofstream meta(fixture + "-geometry.json"); assert(meta);
        meta << "{\"origin\":" << origin << ",\"capacity\":" << capacity << ",\"heads_kv\":" << heads
             << ",\"head_dim\":256,\"iterations\":16,\"tail\":" << tail << ",\"tokens\":" << total
             << ",\"append_tokens\":8,\"record_bytes\":8960,\"packets\":80}"; assert(meta);
    }
    std::vector<ggml_fp16_t> rotated_k(total * width), rotated_v(rotated_k.size());
    std::vector<uint8_t> a, b, saved;
    double cpu_ms = 0, gpu_ms = 0, cross_sq = 0, norm_sq = 0, cpu_sq = 0, gpu_sq = 0;
    uint64_t compared = 0;
    uint32_t at = 0;
    // Unaligned preparations exercise closing groups that contain both old
    // stage rows and newly rotated rows, and cross both ping-pong slots.
    for (uint32_t batch : {120u, 256u, 256u, 8u}) {
        const auto start = std::chrono::steady_clock::now();
        assert(gpu.prepare(k.data() + at * width, v.data() + at * width, pos.data() + at, batch));
        gpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        for (uint32_t consumed = 0; consumed < batch; consumed += 8) {
            const uint32_t begin = at + consumed;
            auto tick = std::chrono::steady_clock::now();
            assert(cpu.append(k.data() + begin * width, v.data() + begin * width, pos.data() + begin, 8, begin == 0, a));
            cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tick).count();
            tick = std::chrono::steady_clock::now();
            assert(gpu.append(k.data() + begin * width, v.data() + begin * width, pos.data() + begin, 8, begin == 0, b));
            gpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tick).count();
            export_bytes("-packet-" + std::to_string(begin / 8) + ".bin", b.data(), b.size());
            assert(a.size() == b.size());
            rkva_kvarn_range_upload h; std::memcpy(&h, a.data(), sizeof(h));
            assert(std::memcmp(a.data(), b.data(), sizeof(h)) == 0);
            const size_t record_bytes = size_t(h.record_groups) * physical * record;
            const size_t data_offset = sizeof(h) + 2 * record_bytes;
            assert(std::memcmp(a.data() + data_offset, b.data() + data_offset, a.size() - data_offset) == 0);
            // The first packet is a snapshot; extract its active stage slot.
            for (uint32_t kind = 0; kind < 2; ++kind) {
                auto & rotated = kind ? rotated_v : rotated_k;
                const size_t source = data_offset + kind * (begin == 0 ? 384 : 8) * width * 2;
                const uint32_t slot = 1 + (origin / 128 - 1) % 2;
                std::memcpy(rotated.data() + begin * width,
                    a.data() + source + (begin == 0 ? slot * 128 * width * 2 : 0), 8 * width * 2);
            }
            for (uint32_t kind = 0; kind < 2; ++kind) for (uint32_t group = 0; group < h.record_groups; ++group) {
                const auto & rotated = kind ? rotated_v : rotated_k;
                for (uint32_t hs = 0; hs < physical; ++hs) {
                    const size_t off = sizeof(h) + kind * record_bytes + (group * physical + hs) * record;
                    for (uint32_t row = 0; row < 128; ++row) for (uint32_t col = 0; col < 128; ++col) {
                        const float x = record_value(a.data() + off, row, col), y = record_value(b.data() + off, row, col);
                        const uint32_t token = (h.record_begin + group) * 128 + (kind ? row : col);
                        const uint32_t dim = kind ? col : row;
                        const float original = ggml_fp16_to_fp32(rotated[token * width + hs * 128 + dim]);
                        assert(std::isfinite(x) && std::isfinite(y));
                        cross_sq += double(x - y) * (x - y); norm_sq += double(original) * original;
                        cpu_sq += double(x - original) * (x - original); gpu_sq += double(y - original) * (y - original); ++compared;
                    }
                }
            }
            if (begin + 8 == total) saved = b;
        }
        at += batch;
    }
    assert(at == total && compared);
    const double relative = std::sqrt(cross_sq / norm_sq);
    std::printf("tail=%u CPU=%.3fms CUDA=%.3fms speedup=%.2fx record_relative_RMSE=%.7f CPU_error=%.7f CUDA_error=%.7f\n",
        tail, cpu_ms, gpu_ms, cpu_ms / gpu_ms, relative, std::sqrt(cpu_sq / norm_sq), std::sqrt(gpu_sq / norm_sq));
    assert(relative < 0.025 && gpu_sq <= cpu_sq * 1.10 + norm_sq * 1e-6);
    // Roll back across a seal and rebuild through original F32. Cached future
    // rows must be invalidated and the resealed GPU records deterministic.
    assert(gpu.rewind(total - 128));
    for (uint32_t t = total - 128; t < total; t += 8) {
        assert(gpu.append(k.data() + t * width, v.data() + t * width, pos.data() + t, 8, false, b));
    }
    assert(saved == b);
    assert(!gpu.can_rewind(total - 129));
}

// Direct helper benchmark also runs against the frozen initial implementation:
// its machine ABI is unchanged by const inputs. Host stage publication below
// makes both implementations advance identically. Full tests additionally
// require the new helper never to modify the host stage.
static void benchmark_decode(bool read_only_contract) {
    const auto append = reinterpret_cast<ggml_backend_cuda_kvarn_native_append_v1_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_reg_by_name("CUDA"), "ggml_backend_cuda_kvarn_native_append_v1"));
    assert(append);
    constexpr uint32_t heads = 4, origin = 1024, total = 512;
    constexpr size_t width = heads * 256;
    std::vector<float> keys(width), values(width);
    std::vector<ggml_fp16_t> sk(384 * width), sv(sk.size()), ak(width), av(width);
    std::vector<uint8_t> rk(heads * 2 * 8960), rv(rk.size());
    std::vector<double> ordinary, seals;
    for (uint32_t row = 0; row < total; ++row) {
        for (size_t d = 0; d < width; ++d) {
            keys[d] = std::sin(float(row * 13 + d * 7) * 0.003f);
            values[d] = std::cos(float(row * 17 + d * 11) * 0.007f);
        }
        const bool check_stage = read_only_contract && (row == 63 || row == 127 || row == 255);
        const auto previous_k = check_stage ? sk : std::vector<ggml_fp16_t>{};
        const auto previous_v = check_stage ? sv : std::vector<ggml_fp16_t>{};
        const auto begin = std::chrono::steady_clock::now();
        assert(append(keys.data(), values.data(), origin + row, 1, heads, 16,
            sk.data(), sv.data(), ak.data(), av.data(), rk.data(), rv.data()));
        const double elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
        if (check_stage) assert(sk == previous_k && sv == previous_v);
        const uint32_t group = (origin + row) / 128, slot = 1 + (group - 1) % 2;
        const size_t offset = size_t(slot * 128 + (origin + row) % 128) * width;
        std::copy_n(ak.data(), width, sk.data() + offset);
        std::copy_n(av.data(), width, sv.data() + offset);
        if (row >= 8) ((row + 1) % 128 ? ordinary : seals).push_back(elapsed);
    }
    std::sort(ordinary.begin(), ordinary.end());
    double ordinary_sum = 0, seal_sum = 0;
    for (double t : ordinary) ordinary_sum += t;
    for (double t : seals) seal_sum += t;
    std::printf("MICRO native_cuda_decode_1 Hkv=4 D=256 iterations=16 ordinary_n=%zu ordinary_mean_us=%.3f ordinary_p50_us=%.3f ordinary_p95_us=%.3f seal_n=%zu seal_mean_us=%.3f read_only_contract=%u\n",
        ordinary.size(), ordinary_sum / ordinary.size(), ordinary[ordinary.size() / 2],
        ordinary[ordinary.size() * 95 / 100], seals.size(), seal_sum / seals.size(), unsigned(read_only_contract));
}

int main() {
    if (!std::getenv("GGML_RKVA_GPU_TEST")) { std::puts("SKIP: set GGML_RKVA_GPU_TEST=1 for CUDA qualification"); return 0; }
    ggml_backend_load_all();
    assert(ggml_backend_reg_by_name("CUDA"));
    if (std::getenv("GGML_RKVA_GPU_MICRO_ONLY")) { benchmark_decode(false); return 0; }
    test(0); test(128);
    benchmark_decode(true);
    mode("cpu");
}
