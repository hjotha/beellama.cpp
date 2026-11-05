#!/usr/bin/env python3
"""Exercise the production chunk transaction with injected OOM, without CUDA."""

from pathlib import Path
import subprocess
import tempfile


source = (Path(__file__).resolve().parents[1] / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text()
begin = "// BEGIN ggml_cuda_graph_chunk_transaction\n"
end = "// END ggml_cuda_graph_chunk_transaction"
assert source.count(begin) == source.count(end) == 1
transaction = source.split(begin, 1)[1].split(end, 1)[0]
begin = "// BEGIN ggml_cuda_graph_failure_invalidate\n"
end = "// END ggml_cuda_graph_failure_invalidate"
assert source.count(begin) == source.count(end) == 1
invalidate = source.split(begin, 1)[1].split(end, 1)[0]
header = (Path(__file__).resolve().parents[1] / "ggml/src/ggml-cuda/common.cuh").read_text()
begin = "            // BEGIN ggml_cuda_graph_eviction_sweep\n"
end = "            // END ggml_cuda_graph_eviction_sweep"
assert header.count(begin) == header.count(end) == 1
eviction = header.split(begin, 1)[1].split(end, 1)[0]

# Captured chunks must not execute before the transaction has prepared all of
# them; a failed preparation then permits one full direct execution.
evaluator = source.index("static bool ggml_cuda_graph_evaluate_and_capture(")
launch = source.index("        const auto launch = [&]() {", evaluator)
assert "cudaGraphLaunch(" not in source[evaluator:launch]
assert "i >= chunk_end && !is_concurrent_event_active" in source[evaluator:launch]
prepare = source.index("static bool ggml_cuda_graph_prepare_chunk(")
assert "cudaGraphLaunch(" not in source[prepare:source.index("\n#endif // USE_CUDA_GRAPH", prepare)]
failure = source.index("!ggml_cuda_graph_failure_invalidate(*it->second", evaluator)
assert source.rindex("CUDA_CHECK(cudaDeviceSynchronize());", evaluator, failure) < failure
failure_end = source.index("for (auto it = gb10_pool_allocations.rbegin()", failure)
assert "cuda_graphs_memory_cleanup_failed = true" in source[failure:failure_end]
assert "disable_cuda_graphs_due_to_memory_pressure = true" in source[failure:failure_end]

program = r"""
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
using std::size_t;
""" + transaction + invalidate + r"""

static void exercise(int failure) {
    constexpr size_t count = 4;
    std::vector<int> events;
    int effects[count] = {};
    int fallbacks = 0, launches = 0, cleanups = 0;
    const bool ok = ggml_cuda_graph_chunk_transaction(count,
        [&](size_t i) { events.push_back(10 + int(i)); return int(i) != failure; },
        [&]() { events.push_back(20); ++cleanups; },
        [&]() {
            events.push_back(30); ++launches;
            for (size_t i = 0; i < count; ++i) ++effects[i];
            return true;
        },
        [&]() {
            events.push_back(40); ++fallbacks;
            for (size_t i = 0; i < count; ++i) ++effects[i];
            return true;
        });
    assert(ok && cleanups == 1);
    for (int n : effects) assert(n == 1);
    if (failure < 0) {
        assert((events == std::vector<int>{10, 11, 12, 13, 20, 30}));
        assert(launches == 1 && fallbacks == 0);
    } else {
        assert(events.size() == size_t(failure) + 3);
        assert(events[events.size() - 2] == 20 && events.back() == 40);
        assert(launches == 0 && fallbacks == 1);
    }
}

struct graph_state {
    std::vector<void *> instances;
    unsigned long long uid = 17;
    bool warmup_complete = true;
    int warmup_stable_calls = 5;
};

static void exercise_invalidation(bool fail_destroy) {
    int a = 1, b = 2, c = 3;
    graph_state graph{{&a, &b, &c}};
    int calls = 0;
    const bool ok = ggml_cuda_graph_failure_invalidate(graph, [&](void *) {
        // Reset before attempting destruction, including on a destroy failure.
        assert(graph.uid == 0 && !graph.warmup_complete && graph.warmup_stable_calls == 0);
        return ++calls != 2 || !fail_destroy;
    });
    assert(graph.uid == 0 && !graph.warmup_complete && graph.warmup_stable_calls == 0);
    assert(!(graph.uid == 17 && graph.warmup_complete && !graph.instances.empty()));
    if (fail_destroy) {
        assert(!ok && calls == 2 && graph.instances.size() == 3);
        assert(graph.instances[0] == nullptr && graph.instances[1] == &b && graph.instances[2] == &c);
    } else {
        assert(ok && calls == 3 && graph.instances.empty());
    }
}

static void exercise_eviction() {
    struct graph { int64_t last_used_time; };
    int active = 0, stale = 0, boundary = 0, fresh = 0;
    std::unordered_map<const void *, std::unique_ptr<graph>> cuda_graphs;
    cuda_graphs.emplace(&active, std::make_unique<graph>(graph{0}));
    cuda_graphs.emplace(&stale, std::make_unique<graph>(graph{0}));
    cuda_graphs.emplace(&boundary, std::make_unique<graph>(graph{10'000'000}));
    cuda_graphs.emplace(&fresh, std::make_unique<graph>(graph{10'000'001}));
    graph * held = cuda_graphs.at(&active).get();
    const void * first_node_ptr = &active;
    const int64_t time_now = 20'000'000;
""" + eviction + r"""
    assert(cuda_graphs.size() == 2);
    assert(cuda_graphs.at(&active).get() == held);
    assert(cuda_graphs.count(&stale) == 0 && cuda_graphs.count(&boundary) == 0);
    assert(cuda_graphs.count(&fresh) == 1);
}

int main() {
    exercise(-1);
    exercise(0);
    exercise(1);
    exercise(3);
    exercise_invalidation(false);
    exercise_invalidation(true);
    exercise_eviction();
    int launches = 0, cleanups = 0, fallbacks = 0;
    const bool ok = ggml_cuda_graph_chunk_transaction(3,
        [](size_t i) { return i != 1; },
        [&]() { ++cleanups; },
        [&]() { ++launches; return true; },
        [&]() { ++fallbacks; return false; });
    assert(!ok && cleanups == 1 && fallbacks == 1 && launches == 0);
}
"""

with tempfile.TemporaryDirectory(prefix="cuda-graph-transaction-") as directory:
    cpp = Path(directory) / "test.cpp"
    exe = Path(directory) / "test"
    cpp.write_text(program)
    subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
print("CUDA graph chunks: OOM fallback, failure invalidation, and active graph eviction guard passed")
