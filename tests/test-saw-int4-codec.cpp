#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

// Compare with the existing scalar RKVA codec even in native AVX2 builds.
#undef __AVX2__
#include "../tools/xbox/uwp/rkva/rkva-saw-int4.h"

int main() {
    constexpr int dim = 256;
    constexpr int rows = 128;
    if (ggml_blck_size(GGML_TYPE_SAW_INT4) != dim ||
            ggml_type_size(GGML_TYPE_SAW_INT4) != sizeof(block_saw_int4)) return 1;

    const auto * traits = ggml_get_type_traits(GGML_TYPE_SAW_INT4);
    const auto * cpu = ggml_get_type_traits_cpu(GGML_TYPE_SAW_INT4);
    if (!traits->from_float_ref || !traits->to_float || !cpu->from_float || cpu->vec_dot) return 2;

    std::mt19937 gen(931274);
    std::vector<float> input(rows * dim);
    for (float & x : input) x = float(int(gen() % 4097) - 2048) / 64.0f;
    std::fill(input.begin(), input.begin() + dim, 0.0f);
    std::fill(input.begin() + dim, input.begin() + 2*dim, 0.5f);
    input[2*dim + 17] = 1024.0f;

    std::vector<block_saw_int4> expected(rows), actual(rows);
    rkva_saw_int4::quantize_row_saw_int4(input.data(), expected.data(), input.size());
    traits->from_float_ref(input.data(), actual.data(), input.size());
    if (std::memcmp(expected.data(), actual.data(), rows*sizeof(actual[0])) != 0) return 3;

    std::vector<float> old_values(input.size()), new_values(input.size());
    rkva_saw_int4::dequantize_row_saw_int4(expected.data(), old_values.data(), input.size());
    traits->to_float(actual.data(), new_values.data(), input.size());
    if (std::memcmp(old_values.data(), new_values.data(), input.size()*sizeof(float)) != 0) return 4;

    std::fill(actual.begin(), actual.end(), block_saw_int4{});
    const size_t nbytes = ggml_quantize_chunk(GGML_TYPE_SAW_INT4, input.data(), actual.data(),
            2*dim, 2, 2*dim, nullptr);
    if (nbytes != 4*sizeof(actual[0]) ||
            std::memcmp(actual.data() + 2, expected.data() + 2, nbytes) != 0) return 5;
    cpu->from_float(input.data(), actual.data(), input.size());
    if (std::memcmp(expected.data(), actual.data(), rows*sizeof(actual[0])) != 0) return 6;

    if (!ggml_validate_row_data(GGML_TYPE_SAW_INT4, expected.data(), sizeof(expected[0])) ||
            ggml_validate_row_data(GGML_TYPE_SAW_INT4, expected.data(), sizeof(expected[0]) - 1)) return 7;
    expected[0].scale = std::numeric_limits<float>::quiet_NaN();
    if (ggml_validate_row_data(GGML_TYPE_SAW_INT4, expected.data(), sizeof(expected[0]))) return 8;

    ggml_init_params params{4*1024*1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_backend_t backend = ggml_backend_cpu_init();
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, 1, 1, 1);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_SAW_INT4, dim, 8, 1, 1);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_SAW_INT4, dim, 8, 1, 1);
    auto * fa = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f, 0.0f, 0.0f);
    auto * mat = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_SAW_INT4, dim, 2),
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1));
    auto * out_prod = ggml_out_prod(ctx, k, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 8));
    auto * get_rows = ggml_get_rows(ctx, k, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1));
    auto * add = ggml_add(ctx, k, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, 8, 1, 1));
    auto * to_f16 = ggml_cast(ctx, k, GGML_TYPE_F16);
    auto * to_bf16 = ggml_cast(ctx, k, GGML_TYPE_BF16);
    auto * padded = ggml_view_2d(ctx, k, dim, 2, 2*ggml_type_size(GGML_TYPE_SAW_INT4), 0);
    auto * reshaped_copy = ggml_cpy(ctx, padded, ggml_new_tensor_2d(ctx, GGML_TYPE_SAW_INT4, 2*dim, 1));
    bool unsupported = true;
    for (auto * op : {fa, mat, out_prod, get_rows, add, to_f16, to_bf16, reshaped_copy}) {
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, op);
        unsupported = unsupported && !ggml_backend_supports_op(backend, op) &&
                ggml_backend_graph_plan_create(backend, graph) == nullptr &&
                ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED;
    }
    auto * floats = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 2);
    auto * packed = ggml_new_tensor_2d(ctx, GGML_TYPE_SAW_INT4, dim, 2);
    auto * converted = ggml_cpy(ctx, floats, packed);
    auto * packed_copy = ggml_cpy(ctx, converted, ggml_new_tensor_2d(ctx, GGML_TYPE_SAW_INT4, dim, 2));
    auto * unpacked = ggml_cpy(ctx, packed_copy, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 2));
    auto * q_mixed = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, 1, 1, 1);
    auto * k_mixed = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dim, 1, 1, 1);
    auto * v_mixed = ggml_new_tensor_4d(ctx, GGML_TYPE_SAW_INT4, dim, 1, 1, 1);
    auto * mixed_fa = ggml_flash_attn_ext(ctx, q_mixed, k_mixed, v_mixed, nullptr, 1.0f, 0.0f, 0.0f);
    struct row_case { ggml_tensor * values; ggml_tensor * indices; ggml_tensor * dest; ggml_tensor * op; ggml_cgraph * graph; };
    std::vector<row_case> row_cases;
    for (auto value_type : {GGML_TYPE_F32, GGML_TYPE_F16}) for (auto index_type : {GGML_TYPE_I32, GGML_TYPE_I64}) {
        auto * values = ggml_new_tensor_2d(ctx, value_type, dim, 2);
        auto * indices = ggml_new_tensor_1d(ctx, index_type, 2);
        auto * dest = ggml_new_tensor_2d(ctx, GGML_TYPE_SAW_INT4, dim, 4);
        auto * rows_op = ggml_set_rows(ctx, dest, values, indices);
        auto * rows_graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(rows_graph, rows_op);
        row_cases.push_back({values, indices, dest, rows_op, rows_graph});
    }
    struct cast_case { ggml_tensor * source; ggml_tensor * dest; ggml_tensor * op; ggml_cgraph * graph; };
    std::vector<cast_case> cast_cases;
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_BF16}) {
        auto * source = ggml_new_tensor_2d(ctx, type, dim, 1);
        auto * dest = ggml_new_tensor_2d(ctx, GGML_TYPE_SAW_INT4, dim, 1);
        auto * cast_op = ggml_cpy(ctx, source, dest);
        auto * cast_graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cast_graph, cast_op);
        cast_cases.push_back({source, dest, cast_op, cast_graph});
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, unpacked);
    ggml_cgraph * mixed_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(mixed_graph, mixed_fa);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 10;
    ggml_backend_tensor_set(floats, input.data(), 0, 2*dim*sizeof(float));
    std::vector<float> roundtrip(2*dim), reference(2*dim);
    traits->to_float(actual.data(), reference.data(), 2*dim);
    const bool converted_ok = ggml_backend_supports_op(backend, converted) &&
            ggml_backend_supports_op(backend, packed_copy) && ggml_backend_supports_op(backend, unpacked) &&
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (converted_ok) ggml_backend_tensor_get(unpacked, roundtrip.data(), 0, roundtrip.size()*sizeof(float));
    const bool matching = converted_ok && std::memcmp(reference.data(), roundtrip.data(), roundtrip.size()*sizeof(float)) == 0;
    std::vector<float> zeros(dim, 0.0f), mixed_output(dim), mixed_reference(dim);
    std::vector<ggml_fp16_t> zeros_f16(dim, ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(q_mixed, zeros.data(), 0, zeros.size()*sizeof(float));
    ggml_backend_tensor_set(k_mixed, zeros_f16.data(), 0, zeros_f16.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v_mixed, actual.data() + 1, 0, sizeof(actual[1]));
    traits->to_float(actual.data() + 1, mixed_reference.data(), dim);
    const bool mixed_ok = ggml_backend_supports_op(backend, mixed_fa) &&
            ggml_backend_graph_compute(backend, mixed_graph) == GGML_STATUS_SUCCESS;
    if (mixed_ok) ggml_backend_tensor_get(mixed_fa, mixed_output.data(), 0, mixed_output.size()*sizeof(float));
    const bool mixed_matching = mixed_ok && std::equal(mixed_output.begin(), mixed_output.end(),
            mixed_reference.begin(), [](float a, float b) { return std::abs(a - b) < 1e-5f; });
    bool rows_matching = true;
    const std::vector<float> half_f32(2*dim, 0.5f);
    const std::vector<ggml_fp16_t> half_f16(2*dim, ggml_fp32_to_fp16(0.5f));
    const int32_t idx32[2] = {1, -1};
    const int64_t idx64[2] = {1, -1};
    std::vector<uint8_t> row_bytes(4*sizeof(block_saw_int4), 0xa5), got_rows(row_bytes.size());
    for (const auto & item : row_cases) {
        ggml_backend_tensor_set(item.dest, row_bytes.data(), 0, row_bytes.size());
        if (item.values->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(item.values, half_f32.data(), 0, half_f32.size()*sizeof(float));
        } else {
            ggml_backend_tensor_set(item.values, half_f16.data(), 0, half_f16.size()*sizeof(ggml_fp16_t));
        }
        if (item.indices->type == GGML_TYPE_I32) ggml_backend_tensor_set(item.indices, idx32, 0, sizeof(idx32));
        else ggml_backend_tensor_set(item.indices, idx64, 0, sizeof(idx64));
        const bool ok = ggml_backend_supports_op(backend, item.op) &&
                ggml_backend_graph_compute(backend, item.graph) == GGML_STATUS_SUCCESS;
        if (ok) ggml_backend_tensor_get(item.dest, got_rows.data(), 0, got_rows.size());
        std::vector<uint8_t> expected_rows = row_bytes;
        std::memcpy(expected_rows.data() + sizeof(block_saw_int4), actual.data() + 1, sizeof(block_saw_int4));
        rows_matching = rows_matching && ok && got_rows == expected_rows;
    }
    bool casts_matching = true;
    const std::vector<ggml_fp16_t> cast_f16(dim, ggml_fp32_to_fp16(0.5f));
    const std::vector<ggml_bf16_t> cast_bf16(dim, ggml_fp32_to_bf16(0.5f));
    for (const auto & item : cast_cases) {
        if (item.source->type == GGML_TYPE_F16) {
            ggml_backend_tensor_set(item.source, cast_f16.data(), 0, cast_f16.size()*sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(item.source, cast_bf16.data(), 0, cast_bf16.size()*sizeof(ggml_bf16_t));
        }
        block_saw_int4 got{};
        const bool ok = ggml_backend_supports_op(backend, item.op) &&
                ggml_backend_graph_compute(backend, item.graph) == GGML_STATUS_SUCCESS;
        if (ok) ggml_backend_tensor_get(item.dest, &got, 0, sizeof(got));
        casts_matching = casts_matching && ok && std::memcmp(&got, actual.data() + 1, sizeof(got)) == 0;
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    if (!unsupported) return 9;
    if (!matching) return 11;
    if (!mixed_matching) return 12;
    if (!rows_matching) return 13;
    if (!casts_matching) return 14;
    std::puts("SAW INT4 scalar codec and GGML traits: ok");
    return 0;
}
