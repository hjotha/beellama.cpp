#pragma once

// F4 TEST-ONLY backend copy / remote-dispatch counters.
//
// Compiled only when LLAMA_F4_TEST_COUNTERS is defined, which the build sets
// exclusively for build-f4. No production build change, no profiling framework.
//
// These counters observe the ACTUAL boundaries:
//   * ggml_backend_tensor_copy issued by the scheduler for a cross-backend input
//     (the one copy site the existing GGML_BACKEND_COPY_PROFILE text trace does
//     not cover -- it only profiles the non-INPUT branch), and
//   * ggml_backend_vk_graph_compute, i.e. real Vulkan graph execution.
//
// Window scoping is explicit: ggml_f4_counters_dump_and_reset() prints one
// F4COUNT line and zeroes the window, so a delta printed at the server's
// capture point covers exactly the work since the previous capture. That
// separates model-load initialization from the measured prompt/decode.

#if defined(LLAMA_F4_TEST_COUNTERS)

#include "ggml-backend.h" // GGML_BACKEND_API (export visibility)

#include <atomic>
#include <cstdint>
#include <cstdio>

struct ggml_f4_counter_state {
    std::atomic<uint64_t> copies_total { 0 };
    std::atomic<uint64_t> copies_bytes { 0 };
    std::atomic<uint64_t> copies_in_vulkan { 0 };
    std::atomic<uint64_t> copies_out_vulkan { 0 };
    std::atomic<uint64_t> copies_vulkan_bytes { 0 };
    std::atomic<uint64_t> vk_graphs { 0 };
    std::atomic<uint64_t> vk_nodes { 0 };
};

inline ggml_f4_counter_state & ggml_f4_counters() {
    static ggml_f4_counter_state c;
    return c;
}

// Records one scheduler copy, tagged by whether it moves data INTO or OUT of the
// remote Vulkan device and by its size. `dst_vk`/`src_vk` are precomputed by the
// call site from ggml_backend_dev_name().
inline void ggml_f4_count_copy(uint64_t bytes, bool src_vk, bool dst_vk) {
    auto & c = ggml_f4_counters();
    c.copies_total.fetch_add(1, std::memory_order_relaxed);
    c.copies_bytes.fetch_add(bytes, std::memory_order_relaxed);
    if (dst_vk) {
        c.copies_in_vulkan.fetch_add(1, std::memory_order_relaxed);
        c.copies_vulkan_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    if (src_vk) {
        c.copies_out_vulkan.fetch_add(1, std::memory_order_relaxed);
        c.copies_vulkan_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
}

inline void ggml_f4_count_vk_graph(int64_t n_nodes) {
    auto & c = ggml_f4_counters();
    c.vk_graphs.fetch_add(1, std::memory_order_relaxed);
    c.vk_nodes.fetch_add((uint64_t) n_nodes, std::memory_order_relaxed);
}

GGML_BACKEND_API void ggml_f4_counters_dump_and_reset(const char * tag);

#else // !LLAMA_F4_TEST_COUNTERS

#define ggml_f4_count_copy(bytes, src_vk, dst_vk) ((void) 0)
#define ggml_f4_count_vk_graph(n_nodes)          ((void) 0)

#endif // LLAMA_F4_TEST_COUNTERS