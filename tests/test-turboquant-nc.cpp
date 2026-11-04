#include "ggml-turboquant-nc.h"
#include "ggml-backend.h"
#if defined(GGML_USE_CPU) && !defined(GGML_BACKEND_DL)
#include "ggml-cpu.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failures; } } while (0)

#if defined(GGML_USE_CPU) && !defined(GGML_BACKEND_DL)
static void test_cpu_backend_admission() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    CHECK(cpu != nullptr);
    if (!cpu) return;
    for (ggml_type type : {GGML_TYPE_TQNC_K, GGML_TYPE_TQNC_V}) {
        ggml_context * ctx = ggml_init({1u << 20, nullptr, true});
        CHECK(ctx != nullptr);
        if (!ctx) continue;
        constexpr int d = GGML_TQNC_DIM;
        const size_t block = ggml_type_size(type);
        ggml_tensor * source = ggml_new_tensor_2d(ctx, type, d, 4);
        ggml_tensor * target = ggml_new_tensor_2d(ctx, type, d, 4);
        ggml_tensor * fp32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, 4);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
        ggml_tensor * float_rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, 2);
        ggml_tensor * out_prod_arg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
        ggml_tensor * opposite = ggml_new_tensor_2d(ctx,
                type == GGML_TYPE_TQNC_K ? GGML_TYPE_TQNC_V : GGML_TYPE_TQNC_K, d, 4);
        ggml_tensor * to_fp32 = ggml_cpy(ctx, source, fp32);
        ggml_tensor * get_rows = ggml_get_rows(ctx, source, indices);
        ggml_tensor * out_prod = ggml_out_prod(ctx, source, out_prod_arg);
        CHECK(!ggml_backend_supports_op(cpu, to_fp32));
        CHECK(!ggml_backend_supports_op(cpu, ggml_cpy(ctx, fp32, target)));
        CHECK(!ggml_backend_supports_op(cpu, ggml_cpy(ctx, source, opposite)));
        CHECK(!ggml_backend_supports_op(cpu, get_rows));
        CHECK(!ggml_backend_supports_op(cpu, out_prod));
        CHECK(!ggml_backend_supports_op(cpu, ggml_set_rows(ctx, target, float_rows, indices)));
        CHECK(!ggml_backend_supports_op(cpu, ggml_add(ctx, source, fp32)));
        ggml_tensor * full_raw = ggml_cpy(ctx, source, target);
        CHECK(ggml_backend_supports_op(cpu, full_raw));
        CHECK(ggml_backend_supports_op(cpu, ggml_dup(ctx, source)));
        CHECK(ggml_backend_supports_op(cpu, ggml_cont(ctx, source)));
        CHECK(ggml_backend_supports_op(cpu, ggml_dup(ctx, fp32))); // unrelated CPU path

        ggml_tensor * good_view = ggml_view_2d(ctx, source, d, 2, block, block);
        ggml_tensor * good_target = ggml_new_tensor_2d(ctx, type, d, 2);
        ggml_tensor * raw = ggml_cpy(ctx, good_view, good_target);
        CHECK(ggml_backend_supports_op(cpu, good_view));
        CHECK(ggml_backend_supports_op(cpu, raw));
        ggml_tensor * mid_block = ggml_view_2d(ctx, source, d, 2, block, 2);
        ggml_tensor * beyond_base = ggml_view_2d(ctx, source, d, 2, 4*block, 0);
        CHECK(!ggml_backend_supports_op(cpu, ggml_cpy(ctx, mid_block, good_target)));
        CHECK(!ggml_backend_supports_op(cpu, ggml_cpy(ctx, beyond_base, good_target)));
        ggml_tensor * nested = ggml_view_2d(ctx, good_view, d, 1, block, block);
        ggml_tensor * one_row = ggml_new_tensor_2d(ctx, type, d, 1);
        CHECK(nested->view_src == source && nested->view_offs == 2*block); // GGML flattens views
        CHECK(ggml_backend_supports_op(cpu, ggml_cpy(ctx, nested, one_row)));
        ggml_tensor * first_half = ggml_view_2d(ctx, source, d, 2, block, 0);
        ggml_tensor * overlap = ggml_view_2d(ctx, source, d, 2, block, block);
        ggml_tensor * last_half = ggml_view_2d(ctx, source, d, 2, block, 2*block);
        CHECK(!ggml_backend_supports_op(cpu, ggml_cpy(ctx, first_half, overlap)));
        CHECK(ggml_backend_supports_op(cpu, ggml_cpy(ctx, first_half, last_half)));

        // Forged metadata must be rejected before any overflowing ggml_nbytes
        // or ggml_nelements multiplication, without allocating a huge tensor.
        ggml_tensor huge_source = *source, huge_copy = *full_raw;
        huge_source.ne[0] = (std::numeric_limits<int64_t>::max() / d) * d;
        huge_copy.ne[0] = huge_source.ne[0]; huge_copy.src[0] = &huge_source;
        CHECK(!ggml_backend_supports_op(cpu, &huge_copy));
        huge_source = *source; huge_copy = *full_raw;
        huge_source.ne[1] = std::numeric_limits<int64_t>::max();
        huge_copy.ne[1] = huge_source.ne[1]; huge_copy.src[0] = &huge_source;
        CHECK(!ggml_backend_supports_op(cpu, &huge_copy));
        ggml_tensor unflattened = *good_view, chained_copy = *raw;
        unflattened.view_src = good_view; unflattened.view_offs = 0;
        chained_copy.src[0] = &unflattened;
        CHECK(!ggml_backend_supports_op(cpu, &chained_copy));

        for (ggml_tensor * refused : {to_fp32, get_rows, out_prod}) {
            ggml_cgraph * bad_graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(bad_graph, refused);
            // Direct backend callers need not query supports_op first.
            CHECK(ggml_backend_graph_compute(cpu, bad_graph) == GGML_STATUS_FAILED);
            CHECK(ggml_backend_graph_plan_create(cpu, bad_graph) == nullptr);
        }

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, raw);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        CHECK(buffer != nullptr);
        if (buffer) {
            // The copied records deliberately contain invalid metadata. A raw
            // byte copy must neither invoke the decoder nor abort on it.
            std::vector<uint8_t> bytes(4*block, 0xa5), result(2*block);
            for (int row = 0; row < 4; ++row) {
                bytes[row*block + 128] = 0;
                bytes[row*block + 129] = 0x7c; // +infinity in norm or scale
            }
            ggml_backend_tensor_set(source, bytes.data(), 0, bytes.size());
            CHECK(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(good_target, result.data(), 0, result.size());
            CHECK(std::equal(result.begin(), result.end(), bytes.begin() + block));
            ggml_backend_graph_plan_t plan = ggml_backend_graph_plan_create(cpu, graph);
            CHECK(plan != nullptr);
            if (plan) {
                ggml_tensor * original = raw->src[0];
                raw->src[0] = fp32; // a reused plan must revalidate changed nodes
                CHECK(ggml_backend_graph_plan_compute(cpu, plan) == GGML_STATUS_FAILED);
                raw->src[0] = original;
                CHECK(ggml_backend_graph_plan_compute(cpu, plan) == GGML_STATUS_SUCCESS);
                ggml_backend_graph_plan_free(cpu, plan);
            }
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(ctx);
    }
    ggml_backend_free(cpu);
}
#else
static void test_cpu_backend_admission() {
    std::puts("SKIP: CPU backend admission checks require a linked CPU backend");
}
#endif

int main() {
    test_cpu_backend_admission();
    constexpr int d = GGML_TQNC_DIM;
    static_assert(sizeof(ggml_tqnc_key) == 130 && sizeof(ggml_tqnc_value) == 132, "NC layout");
    std::vector<float> input(d, 0.0f), output(d);
    ggml_tqnc_key key = {};
    ggml_tqnc_value value = {};
    CHECK(ggml_tqnc_quantize_key(input.data(), &key));
    CHECK(ggml_tqnc_dequantize_key(&key, output.data()));
    CHECK(std::all_of(output.begin(), output.end(), [](float x) { return x == 0.0f; }));

    input[0] = 1.0f;
    CHECK(ggml_tqnc_quantize_key(input.data(), &key));
    CHECK(key.norm == ggml_fp32_to_fp16(1.0f));
    CHECK(std::all_of(std::begin(key.qs), std::end(key.qs), [](uint8_t x) { return x == 0xbb; }));
    CHECK(ggml_tqnc_dequantize_key(&key, output.data()));
    for (int i = 0; i < d; ++i) CHECK(std::abs(input[i] - output[i]) < 2e-6f);

    for (int i = 0; i < d; ++i) input[i] = float(i % 16);
    CHECK(ggml_tqnc_quantize_value(input.data(), &value));
    CHECK(value.scale == ggml_fp32_to_fp16(1.0f) && value.minimum == ggml_fp32_to_fp16(0.0f));
    for (int i = 0; i < d/2; ++i) CHECK(value.qs[i] == uint8_t((2*i + 1) % 16 * 16 + (2*i) % 16));
    CHECK(ggml_tqnc_dequantize_value(&value, output.data()));
    CHECK(input == output);

    std::vector<float> two_rows(2*d), restored(2*d);
    for (int i = 0; i < d; ++i) {
        two_rows[i] = float(i % 16);
        two_rows[d + i] = float((i % 7) - 3);
    }
    ggml_tqnc_key keys[2];
    ggml_tqnc_value values[2];
    ggml_tqnc_quantize_key_rows(two_rows.data(), keys, 2*d);
    ggml_tqnc_quantize_value_rows(two_rows.data(), values, 2*d);
    ggml_tqnc_dequantize_key_rows(keys, restored.data(), 2*d);
    for (int i = 0; i < 2; ++i) {
        CHECK(ggml_tqnc_dequantize_key(&keys[i], output.data()));
        CHECK(std::equal(output.begin(), output.end(), restored.begin() + i*d));
    }
    ggml_tqnc_dequantize_value_rows(values, restored.data(), 2*d);
    for (int i = 0; i < 2; ++i) {
        CHECK(ggml_tqnc_dequantize_value(&values[i], output.data()));
        CHECK(std::equal(output.begin(), output.end(), restored.begin() + i*d));
    }

    std::mt19937 rng(823716);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int row = 0; row < 64; ++row) {
        for (float & x : input) x = normal(rng);
        CHECK(ggml_tqnc_quantize_key(input.data(), &key));
        CHECK(ggml_tqnc_dequantize_key(&key, output.data()));
        double norm2 = 0;
        for (float x : output) norm2 += double(x)*x;
        CHECK(std::abs(std::sqrt(norm2) - ggml_fp16_to_fp32(key.norm)) < 1e-4);
        CHECK(ggml_tqnc_quantize_value(input.data(), &value));
        CHECK(ggml_tqnc_dequantize_value(&value, output.data()));
        CHECK(std::all_of(output.begin(), output.end(), [](float x) { return std::isfinite(x); }));
    }

    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), 1e30f}) {
        std::fill(input.begin(), input.end(), 1.0f);
        input[17] = bad;
        std::memset(&key, 0xa5, sizeof(key));
        std::memset(&value, 0xa5, sizeof(value));
        const auto old_key = key;
        const auto old_value = value;
        CHECK(!ggml_tqnc_quantize_key(input.data(), &key));
        CHECK(!ggml_tqnc_quantize_value(input.data(), &value));
        CHECK(std::memcmp(&key, &old_key, sizeof(key)) == 0);
        CHECK(std::memcmp(&value, &old_value, sizeof(value)) == 0);
    }
    std::fill(input.begin(), input.end(), -65504.0f);
    input[0] = 65504.0f;
    CHECK(!ggml_tqnc_quantize_value(input.data(), &value));
    std::fill(output.begin(), output.end(), 123.0f);
    key.norm = ggml_fp32_to_fp16(std::numeric_limits<float>::infinity());
    value.scale = ggml_fp32_to_fp16(-1.0f);
    CHECK(!ggml_tqnc_dequantize_key(&key, output.data()));
    CHECK(!ggml_tqnc_dequantize_value(&value, output.data()));
    CHECK(std::all_of(output.begin(), output.end(), [](float x) { return x == 123.0f; }));
    std::printf("TurboQuant NC CPU checks: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
