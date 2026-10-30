#include "graph-policy.h"
#include "llama-cuda-graph-policy.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

static void check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main() {
    using ggml_cuda_graph_policy::config;
    config local;
    config remote;
    check(!remote.set_min_stable_calls(0), "zero rejected");
    check(!remote.set_min_stable_calls(-1), "negative rejected");
    check(remote.set_min_stable_calls(256), "explicit remote policy accepted");
    check(!remote.set_min_stable_calls(0), "invalid update leaves existing policy intact");
    check(local.freeze_and_get_min_stable_calls(4) == 4, "local default preserved");
    check(remote.freeze_and_get_min_stable_calls(4) == 256, "remote policy selected");
    check(remote.freeze_and_get_min_stable_calls(17) == 256, "explicit policy takes precedence over global");
    check(!remote.set_min_stable_calls(4), "post-submission change rejected");
    check(remote.freeze_and_get_min_stable_calls(4) == 256, "frozen policy unchanged");
    check(!local.set_min_stable_calls(256), "local policy frozen independently");

    config inherited;
    check(inherited.freeze_and_get_min_stable_calls(17) == 17, "global override remains fallback");
    config replacement;
    check(replacement.freeze_and_get_min_stable_calls(4) == 4, "new backend has no inherited override");
    check(llama_remote_cuda_graph_min_stable_calls(false, "256") == 0, "local context ignores remote option");
    check(llama_remote_cuda_graph_min_stable_calls(true, nullptr) == 0, "absent option leaves default");
    check(llama_remote_cuda_graph_min_stable_calls(true, "256") == 256, "remote option selected");
    for (const char * invalid : {"", "0", "-1", "+1", "1x", " 1", "2147483648"}) {
        check(llama_remote_cuda_graph_min_stable_calls(true, invalid) == -1, "invalid option rejected");
    }
    const auto is_meta = [](int backend) { return backend == 2; };
    const int direct[] = {1, 0}; // CUDA + CPU
    const int mixed[] = {1, 2};  // CUDA + meta
    const int reversed[] = {2, 1};
    check(llama_remote_cuda_graph_backends_supported(direct, is_meta), "direct CUDA + CPU accepted");
    check(!llama_remote_cuda_graph_backends_supported(mixed, is_meta), "CUDA + meta rejected");
    check(!llama_remote_cuda_graph_backends_supported(reversed, is_meta), "meta + CUDA rejected");
    std::puts("CUDA graph policy tests passed");
}
