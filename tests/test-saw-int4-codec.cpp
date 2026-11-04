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

    ggml_init_params params{1024*1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_backend_t backend = ggml_backend_cpu_init();
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, 1, 1, 1);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_SAW_INT4, dim, 8, 1, 1);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_SAW_INT4, dim, 8, 1, 1);
    auto * fa = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f, 0.0f, 0.0f);
    auto * mat = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_SAW_INT4, dim, 2),
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1));
    const bool unsupported = !ggml_backend_supports_op(backend, fa) && !ggml_backend_supports_op(backend, mat);
    ggml_backend_free(backend);
    ggml_free(ctx);
    if (!unsupported) return 9;
    std::puts("SAW INT4 scalar codec and GGML traits: ok");
    return 0;
}
