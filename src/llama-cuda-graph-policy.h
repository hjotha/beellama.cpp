#pragma once

#include <limits>

// Returns 0 for absence, -1 for invalid input, or a positive override. Only
// a TCP position-split context consumes the opt-in environment variable.
inline int llama_remote_cuda_graph_min_stable_calls(bool tcp_position_split, const char * value) {
    if (!tcp_position_split || value == nullptr) return 0;
    if (*value == '\0') return -1;
    int result = 0;
    for (const char * p = value; *p; ++p) {
        if (*p < '0' || *p > '9') return -1;
        const int digit = *p - '0';
        if (result > (std::numeric_limits<int>::max() - digit) / 10) return -1;
        result = result * 10 + digit;
    }
    return result > 0 ? result : -1;
}

// Preflight the complete set before modifying any backend. In particular, a
// direct CUDA backend must not mask an unsupported composite backend.
template <typename Backends, typename IsMeta>
bool llama_remote_cuda_graph_backends_supported(const Backends & backends, IsMeta is_meta) {
    for (const auto & backend : backends) {
        if (is_meta(backend)) return false;
    }
    return true;
}
