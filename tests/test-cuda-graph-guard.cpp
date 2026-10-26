// Unit test for the pure decision helper behind the CUDA-graph VRAM headroom
// guard (ggml/src/ggml-cuda/graph-guard.h).
//
// The guard is allowed to skip its per-submission cudaMemGetInfo() query only
// when the decision it would take is unchanged. This test pins that contract:
//   * a valid, fresh sample from the same device-reservation generation may
//     approve a submission only when it still clears the headroom,
//   * every other state (stale, invalid, new generation, future timestamp)
//     rejects and therefore falls back to the exact query,
//   * the helper never reports a bound above the sampled free memory.

#include "graph-guard.h"

#include <cstdio>
#include <cstdlib>
#include <limits>

using ggml_cuda_graph_guard::approval;
using ggml_cuda_graph_guard::sample;

namespace {

int failures = 0;

void check(bool condition, const char * what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

constexpr size_t MiB = 1024 * 1024;

sample fresh_sample(size_t free_bytes, int64_t time_us, uint64_t generation) {
    sample s;
    s.free_bytes = free_bytes;
    s.time_us    = time_us;
    s.generation = generation;
    s.valid      = true;
    return s;
}

// Approves without a query: exactly the case the optimisation exists for.
void test_fresh_comfortable_sample_approves() {
    const sample s = fresh_sample(300 * MiB, 1'000'000, 7);
    const approval a = ggml_cuda_graph_guard::approve_without_query(
            s, /*generation_now*/ 7, /*now_us*/ 1'005'000, /*max_age*/ 10'000, /*headroom*/ 128 * MiB);
    check(a.approved, "comfortable fresh sample approves");
    check(a.free_bound_bytes == 300 * MiB, "approved bound equals the sampled free memory");
}

// The headroom comparison must match the exact path's `free_bytes < headroom`.
void test_headroom_boundary_is_inclusive() {
    const sample at_boundary = fresh_sample(128 * MiB, 0, 1);
    check(ggml_cuda_graph_guard::approve_without_query(at_boundary, 1, 0, 10'000, 128 * MiB).approved,
          "free == headroom approves (matches `free < headroom` pressure test)");

    const sample below = fresh_sample(128 * MiB - 1, 0, 1);
    check(!ggml_cuda_graph_guard::approve_without_query(below, 1, 0, 10'000, 128 * MiB).approved,
          "free below headroom falls back to the exact query");
}

// Pressure must never be hidden by a cached sample.
void test_pressure_sample_rejects() {
    const sample s = fresh_sample(1 * MiB, 0, 3);
    const approval a = ggml_cuda_graph_guard::approve_without_query(s, 3, 0, 10'000, 18 * MiB);
    check(!a.approved, "sampled pressure is not approved");
    check(a.free_bound_bytes == 1 * MiB, "rejected bound is still reported");
}

// Every staleness/identity condition must fall back to the exact query.
void test_stale_or_foreign_sample_rejects() {
    const size_t headroom = 18 * MiB;
    const sample s = fresh_sample(300 * MiB, 1'000'000, 11);

    check(!ggml_cuda_graph_guard::approve_without_query(s, 11, 1'010'001, 10'000, headroom).approved,
          "sample exactly one microsecond past the age limit rejects");
    check(ggml_cuda_graph_guard::approve_without_query(s, 11, 1'010'000, 10'000, headroom).approved,
          "sample exactly at the age limit approves");

    check(!ggml_cuda_graph_guard::approve_without_query(s, 12, 1'000'001, 10'000, headroom).approved,
          "a new device reservation forces the exact query");

    sample invalid = s;
    invalid.valid = false;
    check(!ggml_cuda_graph_guard::approve_without_query(invalid, 11, 1'000'001, 10'000, headroom).approved,
          "invalid sample rejects");

    check(!ggml_cuda_graph_guard::approve_without_query(s, 11, 999'000, 10'000, headroom).approved,
          "timestamp going backwards rejects");

    check(!ggml_cuda_graph_guard::approve_without_query(s, 11, 1'000'001, 0, headroom).approved,
          "max_age == 0 restores exact per-call sampling");
}

// A default-constructed sample must never approve the first submission.
void test_default_sample_rejects() {
    const sample s;
    check(!ggml_cuda_graph_guard::approve_without_query(s, 0, 0, 10'000, 0).approved,
          "default sample rejects");
    check(!ggml_cuda_graph_guard::approve_without_query(s, 0, 0, 10'000, 128 * MiB).approved,
          "default sample rejects even with zero headroom");
}

// Size_t extremes must not wrap into an approval.
void test_extremes_do_not_wrap() {
    const size_t max = std::numeric_limits<size_t>::max();
    const sample s = fresh_sample(max, 0, 1);
    const approval a = ggml_cuda_graph_guard::approve_without_query(s, 1, 0, 10'000, max);
    check(a.approved, "maximum free memory clears a maximum headroom");
    check(a.free_bound_bytes == max, "bound stays representable at the maximum");
}

// The generation counter is the only cross-context signal; it must be
// monotonic and cheap to observe.
void test_generation_counter_is_monotonic() {
    const uint64_t before = ggml_cuda_graph_guard::device_reservation_now();
    ggml_cuda_graph_guard::note_device_reservation();
    const uint64_t after = ggml_cuda_graph_guard::device_reservation_now();
    check(after == before + 1, "one reservation bumps the generation by one");

    // A sample taken after the bump approves again.
    const sample s = fresh_sample(64 * MiB, 5, after);
    check(ggml_cuda_graph_guard::approve_without_query(s, after, 5, 10'000, 18 * MiB).approved,
          "sample from the current generation approves");
    check(!ggml_cuda_graph_guard::approve_without_query(s, after + 1, 5, 10'000, 18 * MiB).approved,
          "sample from an older generation rejects");
}

} // namespace

int main() {
    test_fresh_comfortable_sample_approves();
    test_headroom_boundary_is_inclusive();
    test_pressure_sample_rejects();
    test_stale_or_foreign_sample_rejects();
    test_default_sample_rejects();
    test_extremes_do_not_wrap();
    test_generation_counter_is_monotonic();

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("graph-guard: all checks passed\n");
    return 0;
}
