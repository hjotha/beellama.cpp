#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>

// Owner-bound resident snapshots retain bytes only below every dirty suffix
// since capture. Later cuts at a lower position dominate earlier cuts, so the
// history is bounded by the number of distinct positions in the static cache.
class llama_kv_prefix_lineage {
public:
    void reset(uint64_t epoch) noexcept {
        floor = current = epoch;
        changes.clear();
    }

    // Prepare before a deferred restore commits any writes.
    void reserve_change() {
        if (changes.size() == changes.capacity()) {
            changes.reserve(std::max<std::size_t>(16, changes.capacity() * 2));
        }
    }

    void invalidate_from(uint64_t epoch, int64_t pos) {
        if (epoch <= current || pos < 0) {
            throw std::invalid_argument("invalid KV prefix generation");
        }
        reserve_change(); // allocation must precede any history mutation
        while (!changes.empty() && changes.back().pos >= pos) changes.pop_back();
        changes.push_back({epoch, pos});
        current = epoch;
    }

    bool can_restore(uint64_t saved_epoch, int64_t last_pos) const {
        if (saved_epoch < floor || saved_epoch > current) return false;
        const auto changed = std::upper_bound(changes.begin(), changes.end(), saved_epoch,
                [](uint64_t epoch, const change & item) { return epoch < item.epoch; });
        return changed == changes.end() || last_pos < changed->pos;
    }

private:
    struct change { uint64_t epoch; int64_t pos; };
    uint64_t floor = 1;
    uint64_t current = 1;
    std::vector<change> changes;
};
