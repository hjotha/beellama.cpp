#include "llama-recurrent-zero.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef TEST_RECURRENT_ZERO_CUDA
#include "ggml-cuda.h"
#else
#include "ggml-cpu.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

static bool run_case(ggml_backend_t backend, int row, int count, std::vector<ggml_op> & ops) {
    ggml_init_params params = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * states = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    const int before = ggml_graph_n_nodes(graph);
    llama_build_recurrent_zero(ctx, graph, states, count, row);
    const int added = ggml_graph_n_nodes(graph) - before;
    const bool has_clear = row >= 0 && count > 0;
    for (int i = before; i < before + added; ++i) {
        ops.push_back(ggml_graph_node(graph, i)->op);
    }
    if (added <= 0) {
        std::fprintf(stderr, "row=%d count=%d: unexpected clear-node count %d\n", row, count, added);
        ggml_free(ctx);
        return false;
    }

    // This downstream copy observes the row after the in-place clear.
    ggml_tensor * observed = ggml_dup(ctx, states);
    ggml_build_forward_expand(graph, observed);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    std::array<float, 8> input = { nan, inf, -inf, -7.0f, 11.0f, 12.0f, 13.0f, 14.0f };
    if (row == 1) {
        std::swap_ranges(input.begin(), input.begin() + 4, input.begin() + 4);
    }
    ggml_backend_tensor_set(states, input.data(), 0, sizeof(input));
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::array<float, 8> got{};
    std::array<float, 8> written{};
    if (computed) {
        ggml_backend_tensor_get(observed, got.data(), 0, sizeof(got));
        ggml_backend_tensor_get(states, written.data(), 0, sizeof(written));
    }

    bool ok = computed;
    for (size_t i = 0; i < input.size() && ok; ++i) {
        const float expected = has_clear && i/4 == size_t(row) ? 0.0f : input[i];
        const auto matches = [expected](float actual) {
            return std::isnan(expected) ? std::isnan(actual) : actual == expected;
        };
        if (!matches(got[i]) || !matches(written[i])) {
            std::fprintf(stderr, "row=%d count=%d i=%zu: expected=%g output=%g state=%g\n",
                         row, count, i, expected, got[i], written[i]);
            ok = false;
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main() {
#ifdef TEST_RECURRENT_ZERO_CUDA
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "CUDA backend unavailable\n"); return 77; }
#else
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { std::fprintf(stderr, "CPU backend unavailable\n"); return 1; }
#endif
    std::vector<ggml_op> ops_zero, ops_one, ops_absent, ops_empty;
    const bool ok = run_case(backend, 0, 4, ops_zero) && run_case(backend, 1, 4, ops_one) &&
                    run_case(backend, -1, 4, ops_absent) && run_case(backend, 0, 0, ops_empty) &&
                    ops_zero == ops_one && ops_zero == ops_absent && ops_zero == ops_empty;
    std::printf("recurrent zero graph nodes: row0=%zu row1=%zu absent=%zu empty=%zu\n",
                ops_zero.size(), ops_one.size(), ops_absent.size(), ops_empty.size());
    std::printf("%s recurrent zero: %s\n", ggml_backend_name(backend), ok ? "PASS" : "FAIL");
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
