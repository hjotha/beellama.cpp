#pragma once

// Pure decision helper for the CUDA-graph VRAM headroom guard.
//
// The guard samples cudaMemGetInfo() before every graph submission. On WDDM /
// eGPU hosts that query costs ~30-45 us and a decode token issues ~97 graph
// calls, so sampling every call costs several percent of the token budget.
//
// A sample stays a valid *lower bound* of free device memory while nothing has
// reserved new device memory: every reservation (pool growth, compute buffer,
// CUDA graph executable) strictly reduces free memory. Reservations therefore
// bump a process-wide generation counter, and a sample may only approve a
// submission while the counter is unchanged and the sample is younger than an
// explicit maximum age. Everything else keeps the caller on the exact query,
// so the fast path can never approve a state the per-call sampling would have
// rejected.
//
// The helper is deliberately pure so it is unit testable without CUDA.

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace ggml_cuda_graph_guard {

// Bumped by every device reservation the guard cannot observe from the free
// memory number itself (pools, compute buffers, graph executables). A
// function-local static keeps one instance per process across translation
// units without relying on an inline variable.
inline std::atomic<uint64_t> & device_reservation_generation() {
    static std::atomic<uint64_t> generation {0};
    return generation;
}

inline void note_device_reservation() {
    device_reservation_generation().fetch_add(1, std::memory_order_relaxed);
}

inline uint64_t device_reservation_now() {
    return device_reservation_generation().load(std::memory_order_relaxed);
}

struct sample {
    size_t   free_bytes = 0; // exact cudaMemGetInfo() free bytes
    int64_t  time_us    = 0;
    uint64_t generation = 0; // device reservation generation at sample time
    bool     valid      = false;
};

struct approval {
    bool   approved         = false;
    size_t free_bound_bytes = 0; // conservative lower bound of free device memory
};

// Approve a graph submission without a new device query only when the sampled
// free memory still clears `headroom_bytes`. Every rejection keeps the caller
// on the exact query path.
inline approval approve_without_query(const sample & last, uint64_t generation_now,
                                      int64_t now_us, int64_t max_age_us,
                                      size_t headroom_bytes) {
    approval result;
    if (!last.valid || max_age_us <= 0 || generation_now != last.generation ||
            now_us - last.time_us < 0 || now_us - last.time_us > max_age_us) {
        return result;
    }
    result.free_bound_bytes = last.free_bytes;
    result.approved         = result.free_bound_bytes >= headroom_bytes;
    return result;
}

} // namespace ggml_cuda_graph_guard
