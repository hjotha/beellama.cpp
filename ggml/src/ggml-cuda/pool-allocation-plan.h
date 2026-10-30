#pragma once

// Host-only, nonmutating accounting for a sequence of simultaneously live pool
// allocations. Kept independent of CUDA so the cache policy can be tested on CPU.
#include <cstddef>
#include <limits>

namespace ggml_cuda_pool_plan {

inline bool add(size_t a, size_t b, size_t & out) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        return false;
    }
    out = a + b;
    return true;
}

inline bool multiply(size_t a, size_t b, size_t & out) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
        return false;
    }
    out = a * b;
    return true;
}

// Conservative bounds for the additional pool allocations made by the
// stream-K launch_fattn path after K/V have already been materialized.  The
// maximum number of stream-K blocks is occupancy_per_sm * number_of_sms;
// using that bound even when launch_fattn rounds down or needs no fixup keeps
// the preflight independent of its exact scheduling heuristic.
inline bool stream_k_fixup_bytes(size_t blocks, size_t ncols, size_t value_dim,
        size_t meta_element_bytes, size_t & bytes) {
    if (!blocks || !ncols || !value_dim || value_dim % 2 || !meta_element_bytes) {
        return false;
    }
    size_t fields, result;
    if (!add(2, value_dim / 2, fields) || !multiply(blocks, ncols, result) ||
            !multiply(result, fields, result) || !multiply(result, meta_element_bytes, result)) {
        return false;
    }
    bytes = result;
    return true;
}

inline bool stream_k_workspace_upper_bound(size_t query_tiles, size_t sequences,
        size_t number_of_sms, size_t occupancy_per_sm, size_t ncols,
        size_t value_dim, size_t meta_element_bytes, bool scan_mask,
        size_t & mask_bytes, size_t & meta_bytes) {
    if (!query_tiles || !sequences || !number_of_sms || !occupancy_per_sm ||
            !ncols || !value_dim || value_dim % 2 || !meta_element_bytes) {
        return false;
    }
    size_t mask = 0, blocks, meta;
    if ((scan_mask && (!multiply(query_tiles, sequences, mask) ||
                       !multiply(mask, sizeof(int), mask))) ||
            !multiply(number_of_sms, occupancy_per_sm, blocks) ||
            !stream_k_fixup_bytes(blocks, ncols, value_dim, meta_element_bytes, meta)) {
        return false;
    }
    mask_bytes = mask;
    meta_bytes = meta;
    return true;
}

inline bool align(size_t size, size_t alignment, size_t & out) {
    size_t rounded;
    if (alignment == 0 || !add(size, alignment - 1, rounded)) {
        return false;
    }
    out = rounded / alignment * alignment;
    return true;
}

inline bool legacy_allocation_size(size_t requested, size_t & actual) {
    // Match ggml_cuda_pool_leg::alloc's double look-ahead and 256-byte rounding,
    // including the conversion boundary (SIZE_MAX rounds up when cast to double).
    const double look_ahead = 1.05 * requested;
    if (!(look_ahead < static_cast<double>(std::numeric_limits<size_t>::max()))) {
        return false;
    }
    return align(static_cast<size_t>(look_ahead), 256, actual);
}

template <size_t N>
bool legacy_growth(const size_t (&free_blocks)[N], const size_t * requests,
                   size_t n_requests, size_t & growth) {
    if (n_requests > 0 && requests == nullptr) {
        return false;
    }
    bool consumed[N] = {};
    size_t total = 0;
    for (size_t r = 0; r < n_requests; ++r) {
        // All runtime requests are nonzero. A zero-sized cached block cannot be
        // represented by this size-only snapshot, so do not guess its behavior.
        if (requests[r] == 0) {
            return false;
        }
        size_t best_diff = size_t(1ull << 36);
        size_t best = N;
        for (size_t b = 0; b < N; ++b) {
            if (!consumed[b] && free_blocks[b] >= requests[r]) {
                const size_t diff = free_blocks[b] - requests[r];
                if (diff < best_diff) {
                    best_diff = diff;
                    best = b;
                    if (diff == 0) {
                        break;
                    }
                }
            }
        }
        if (best != N) {
            consumed[best] = true;
        } else {
            size_t actual;
            if (!legacy_allocation_size(requests[r], actual) || !add(total, actual, total)) {
                return false;
            }
        }
    }
    growth = total;
    return true;
}

inline bool vmm_growth(size_t used, size_t mapped, size_t granularity, size_t limit,
                       const size_t * requests, size_t n_requests, size_t & growth) {
    if (used > mapped || mapped > limit || granularity == 0 ||
            (n_requests > 0 && requests == nullptr)) {
        return false;
    }
    const size_t initial_mapped = mapped;
    for (size_t r = 0; r < n_requests; ++r) {
        size_t size;
        if (!align(requests[r], 128, size)) {
            return false;
        }
        const size_t available = mapped - used;
        if (size > available) {
            size_t reserve;
            if (!align(size - available, granularity, reserve) ||
                    !add(mapped, reserve, mapped) || mapped > limit) {
                return false;
            }
        }
        if (!add(used, size, used)) {
            return false;
        }
    }
    growth = mapped - initial_mapped;
    return true;
}

} // namespace ggml_cuda_pool_plan
