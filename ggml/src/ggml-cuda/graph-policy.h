#pragma once

// Per-backend admission policy. Configuration is immutable after the first
// submission so changing it cannot leave existing graph state half-reconfigured.
// This object owns no CUDA resources and never evicts or destroys a graph.
namespace ggml_cuda_graph_policy {
struct config {
    int min_stable_calls_override = 0;
    bool frozen = false;

    bool set_min_stable_calls(int value) {
        if (frozen || value < 1) return false;
        min_stable_calls_override = value;
        return true;
    }

    int freeze_and_get_min_stable_calls(int default_value) {
        frozen = true;
        return min_stable_calls_override ? min_stable_calls_override : default_value;
    }
};
} // namespace ggml_cuda_graph_policy
