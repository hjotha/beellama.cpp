#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static bool check_pair(ggml_backend_t backend, ggml_type type, int tokens_per_seq) {
    constexpr int k = 512;
    constexpr int m = 128;
    constexpr int experts = 16;
    constexpr int used = 4;
    constexpr int graph_size = 16;
    const int tokens = 2 * tokens_per_seq;

    ggml_context * ctx = ggml_init({
        /*.mem_size   =*/ 16*ggml_tensor_overhead() + 3*ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    });
    GGML_ASSERT(ctx);
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, tokens);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
    auto * pair_graph = ggml_new_graph_custom(ctx, graph_size, false);

    std::array<ggml_tensor *, 2> weights;
    std::array<ggml_tensor *, 2> paired;
    std::array<ggml_tensor *, 2> reference;
    std::array<ggml_cgraph *, 2> reference_graphs;
    for (int proj = 0; proj < 2; ++proj) {
        weights[proj] = ggml_new_tensor_3d(ctx, type, k, m, experts);
        paired[proj] = ggml_mul_mat_id(ctx, weights[proj], input, ids);
        reference[proj] = ggml_mul_mat_id(ctx, weights[proj], input, ids);
        reference_graphs[proj] = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(pair_graph, paired[proj]);
        ggml_build_forward_expand(reference_graphs[proj], reference[proj]);
    }
    GGML_ASSERT(ggml_graph_n_nodes(pair_graph) == 2);
    GGML_ASSERT(ggml_graph_node(pair_graph, 0) == paired[0]);
    GGML_ASSERT(ggml_graph_node(pair_graph, 1) == paired[1]);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    std::mt19937 rng(42);
    const auto random_value = [&]() {
        return (float(int(rng() % 2049) - 1024) / 1024.0f);
    };
    std::vector<float> weights_f32(k*m*experts);
    std::vector<uint8_t> weights_quant(ggml_nbytes(weights[0]));
    for (auto * weight : weights) {
        std::generate(weights_f32.begin(), weights_f32.end(), random_value);
        const size_t bytes = ggml_quantize_chunk(type, weights_f32.data(), weights_quant.data(), 0, m*experts, k, nullptr);
        GGML_ASSERT(bytes == weights_quant.size());
        ggml_backend_tensor_set(weight, weights_quant.data(), 0, bytes);
    }

    std::vector<float> activation(k*tokens), got(m*used*tokens), want(got.size());
    std::vector<int32_t> routing(used*tokens);
    double max_nmse = 0.0;
    bool bit_exact = true;
    // Reuse the same graphs with fresh inputs to cover capture and replay.
    for (int repeat = 0; repeat < 6; ++repeat) {
        std::generate(activation.begin(), activation.end(), random_value);
        for (int token = 0; token < tokens; ++token) {
            const int seq = token / tokens_per_seq;
            const int hot = (seq + repeat) % 4;
            routing[token*used + 0] = hot;
            routing[token*used + 1] = (hot + 1) % 4;
            routing[token*used + 2] = token % 5 == 0 ? 4 + int(rng() % (experts - 4)) : (hot + 2) % 4;
            routing[token*used + 3] = token % 7 == 0 ? 4 + (routing[token*used + 2] + 1) % (experts - 4) : (hot + 3) % 4;
        }
        ggml_backend_tensor_set(input, activation.data(), 0, ggml_nbytes(input));
        ggml_backend_tensor_set(ids, routing.data(), 0, ggml_nbytes(ids));

        GGML_ASSERT(ggml_backend_graph_compute(backend, pair_graph) == GGML_STATUS_SUCCESS);
        for (int proj = 0; proj < 2; ++proj) {
            GGML_ASSERT(ggml_backend_graph_compute(backend, reference_graphs[proj]) == GGML_STATUS_SUCCESS);
            const size_t bytes = ggml_nbytes(reference[proj]);
            ggml_backend_tensor_get(paired[proj], got.data(), 0, bytes);
            ggml_backend_tensor_get(reference[proj], want.data(), 0, bytes);
            bit_exact &= std::memcmp(got.data(), want.data(), bytes) == 0;

            double error = 0.0;
            double signal = 0.0;
            for (size_t i = 0; i < got.size(); ++i) {
                const double diff = double(got[i]) - want[i];
                error += diff*diff;
                signal += double(want[i])*want[i];
            }
            const double nmse = error / signal;
            max_nmse = std::isfinite(nmse) ? std::max(max_nmse, nmse) : INFINITY;
        }
    }

    std::printf("%s %s tokens=%d seq=2: max NMSE %.9g, bit-exact %s\n",
            ggml_backend_name(backend), ggml_type_name(type), tokens, max_nmse, bit_exact ? "yes" : "no");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return max_nmse <= 1e-6;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    const char * device = argc > 1 ? argv[1] : "CUDA0";
    ggml_backend_t backend = ggml_backend_init_by_name(device, nullptr);
    if (!backend) {
        std::fprintf(stderr, "Cannot initialize %s\n", device);
        return 1;
    }

    bool ok = true;
    for (ggml_type type : { GGML_TYPE_Q2_0, GGML_TYPE_IQ3_S }) {
        for (int tokens_per_seq : { 512, 1024 }) {
            ok &= check_pair(backend, type, tokens_per_seq);
        }
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
