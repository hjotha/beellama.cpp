// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <limits>

namespace rkva_execution_lifetime {
// Snapshot of a single-owner, monotonically GPU-signaled fence. CPU-signaling
// the fence is never a retirement proof. No waiting or D3D calls in this helper.
struct Snapshot {
    bool unconfirmed = false;
    bool has_fence = false;
    bool has_device = false;
    uint64_t submitted = 0;
    uint64_t completed = 0;
    bool device_removed = false;
};
inline bool releasable(const Snapshot& s) {
    if (!s.unconfirmed)
        return true;
    if (!s.has_fence || !s.has_device)
        return false;
    const auto removed_sentinel = std::numeric_limits<uint64_t>::max();
    if (s.completed == removed_sentinel)
        return s.device_removed;
    if (!s.submitted || s.submitted == removed_sentinel)
        return false;
    return s.completed >= s.submitted;
}
} // namespace rkva_execution_lifetime
