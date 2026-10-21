#include "../ggml/src/ggml-cuda/pool-allocation-plan.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace plan = ggml_cuda_pool_plan;

// Deliberately separate mocked allocator: it removes cache entries, retaining
// all six allocations live, rather than sharing the planner's consumed bitmap.
static size_t mocked_legacy_allocator(std::vector<size_t> cache, const std::vector<size_t> & requests) {
    size_t growth = 0;
    for (size_t requested : requests) {
        size_t best_diff = size_t(1ull << 36);
        size_t chosen = cache.size();
        for (size_t i = 0; i < cache.size(); ++i) {
            if (cache[i] != 0 && cache[i] >= requested && cache[i] - requested < best_diff) {
                best_diff = cache[i] - requested;
                chosen = i;
                if (best_diff == 0) {
                    break;
                }
            }
        }
        if (chosen < cache.size()) {
            cache.erase(cache.begin() + chosen);
        } else {
            // Small generated requests cannot overflow. Reproduce the real
            // allocation policy, independently of the planning helper.
            size_t actual = static_cast<size_t>(1.05 * requested);
            actual = 256 * ((actual + 255) / 256);
            growth += actual;
        }
    }
    return growth;
}

static void test_legacy_cache() {
    size_t growth = 99;
    const size_t one[] = {4096};
    const size_t two_requests[] = {2048, 2048};
    assert(plan::legacy_growth(one, two_requests, 2, growth));
    assert(growth == 2304); // One cached block cannot satisfy K and V both.

    const size_t fragmented[] = {2048, 2048, 2048, 2048};
    const size_t large_request[] = {4096};
    assert(plan::legacy_growth(fragmented, large_request, 1, growth));
    assert(growth == 4352); // Summing free bytes would incorrectly predict zero.

    const size_t best_fit[] = {4096, 2048};
    const size_t ordered[] = {2048, 4096};
    assert(plan::legacy_growth(best_fit, ordered, 2, growth) && growth == 0);
    assert(best_fit[0] == 4096 && best_fit[1] == 2048); // Nonmutating snapshot.

    // A live block is absent from the free snapshot, even if pool_size includes
    // its physical allocation. Only the4096 free block can serve this plan.
    const size_t active_excluded[] = {0, 4096};
    const size_t active_requests[] = {4096, 4096};
    assert(plan::legacy_growth(active_excluded, active_requests, 2, growth));
    assert(growth == 4352);

    const size_t empty[] = {0};
    const size_t round_requests[] = {1, 244, 245, 256, 4096};
    assert(plan::legacy_growth(empty, round_requests, 5, growth));
    assert(growth == 256 + 256 + 512 + 512 + 4352);

    if (sizeof(size_t) >= 8) {
        // Match the allocator's existing64GiB best_diff sentinel exactly.
        const size_t huge[] = {size_t(1ull << 36) + 1024};
        const size_t small[] = {1024};
        assert(plan::legacy_growth(huge, small, 1, growth));
        assert(growth == 1280); // diff==sentinel is not selected by real alloc.
    }
}

static void test_low_free_resident_window() {
    constexpr size_t MiB = 1024 * 1024;
    const size_t cache[] = {1024, 1024, 32 * MiB, 32 * MiB, 9 * MiB, 65536};
    const auto before = std::array<size_t, 6>{cache[0], cache[1], cache[2], cache[3], cache[4], cache[5]};
    const size_t free_bytes = 18 * MiB, safety = 20 * MiB;
    const size_t budget = free_bytes > safety ? free_bytes - safety : 0;
    // Real nq256/Hq24/Hkv4/D256 window shapes: partial8.125MiB, acc48KiB.
    const size_t partial = 128 * (2 * 64 + 64 * 128) * 8;
    assert(partial == 8519680);
    assert(free_bytes < partial + safety); // Previous selector rejected all chunks.
    int selected = 0;
    size_t growth = 0;
    for (int candidate = 65536; candidate >= 256; candidate -= 256) {
        const size_t requests[] = {1024, 1024, size_t(candidate) * 2048,
                                   size_t(candidate) * 2048, partial, 256 * 24 * 8};
        assert(plan::legacy_growth(cache, requests, 6, growth));
        if (growth <= budget) {
            selected = candidate;
            break;
        }
    }
    assert(selected == 16384 && growth == 0);
    assert(std::equal(std::begin(cache), std::end(cache), before.begin()));
}

static void test_legacy_differential() {
    std::mt19937 rng(0x504f4f4c);
    for (int trial = 0; trial < 5000; ++trial) {
        size_t cache[32] = {};
        for (size_t & size : cache) {
            size = rng() % 3 == 0 ? 0 : 1 + rng() % (8 * 1024 * 1024);
        }
        const auto copy = std::vector<size_t>(std::begin(cache), std::end(cache));
        std::vector<size_t> requests;
        for (int i = 0; i < 6; ++i) {
            requests.push_back(1 + rng() % (12 * 1024 * 1024));
        }
        size_t growth = 0;
        assert(plan::legacy_growth(cache, requests.data(), requests.size(), growth));
        assert(growth == mocked_legacy_allocator(copy, requests));
        assert(std::equal(std::begin(cache), std::end(cache), copy.begin()));
    }
}

static void test_vmm() {
    size_t growth = 0;
    const size_t requests[] = {1, 129, 129};
    assert(plan::vmm_growth(0, 512, 256, 4096, requests, 3, growth));
    assert(growth == 256); // Aligned live requests128+256+256 need640 mapped.
    assert(plan::vmm_growth(128, 1024, 256, 4096, requests, 3, growth));
    assert(growth == 0);
    assert(!plan::vmm_growth(0, 512, 256, 512, requests, 3, growth));
    assert(!plan::vmm_growth(1025, 1024, 256, 4096, requests, 3, growth));
    assert(!plan::vmm_growth(0, 512, 0, 4096, requests, 3, growth));
}

static void test_overflow_and_invalid_input() {
    const size_t max = std::numeric_limits<size_t>::max();
    size_t out = 123;
    assert(!plan::add(max, 1, out) && out == 123);
    assert(!plan::multiply(max, 2, out) && out == 123);
    assert(plan::multiply(max, 0, out) && out == 0);
    assert(!plan::align(max, 256, out));
    assert(!plan::legacy_allocation_size(max, out));
    const size_t empty[] = {0};
    const size_t zero[] = {0};
    assert(!plan::legacy_growth(empty, zero, 1, out));
    assert(!plan::legacy_growth(empty, nullptr, 1, out));
    const size_t overflow_sum[] = {max / 2, max / 2, max / 2};
    assert(!plan::legacy_growth(empty, overflow_sum, 3, out));
    assert(!plan::vmm_growth(0, 0, 256, max, &max, 1, out));
    assert(!plan::vmm_growth(0, 0, 256, max, nullptr, 1, out));
}

int main() {
    test_legacy_cache();
    test_low_free_resident_window();
    test_legacy_differential();
    test_vmm();
    test_overflow_and_invalid_input();
    std::puts("CUDA pool allocation planner: cache policy, resident window,5000 differential cases and overflow passed (CPU only)");
    return 0;
}
