#include "../src/llama-kv-prefix-lineage.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <random>

int main() {
    llama_kv_prefix_lineage lineage;
    // Saved H=10 includes positions [0,10). Truncate r=4, then reappend the
    // same IDs/tokens with different bytes: the old H ticket must stay invalid.
    const uint64_t saved_h = 1;
    lineage.invalidate_from(2, 4);
    assert(!lineage.can_restore(saved_h, 9));
    assert(lineage.can_restore(saved_h, 2)); // a checkpoint before r survives
    lineage.invalidate_from(3, 4); // repeated restore/reappend does not revive H
    assert(!lineage.can_restore(saved_h, 9));
    assert(lineage.can_restore(saved_h, 2));
    lineage.invalidate_from(4, 20);
    assert(lineage.can_restore(3, 9)); // later append/cut beyond this prefix
    lineage.reset(5); // FULL restore/clear starts another lineage
    assert(!lineage.can_restore(3, 2));
    assert(!lineage.can_restore(6, 2));
    assert(lineage.can_restore(5, 31));

    // Independent per-position oracle: stamp all bytes affected by each cut.
    // This catches history coalescing errors without copying its algorithm.
    std::mt19937 random(102400);
    uint64_t stamps[32] = {};
    uint64_t floor = 5;
    struct ticket { uint64_t epoch; int last; };
    std::vector<ticket> tickets;
    for (uint64_t epoch = 6; epoch < 5000; ++epoch) {
        if (epoch % 101 == 0) {
            lineage.reset(epoch);
            floor = epoch;
        } else {
            const int cut = int(random() % 32);
            lineage.invalidate_from(epoch, cut);
            for (int p = cut; p < 32; ++p) stamps[p] = epoch;
        }
        tickets.push_back({epoch, int(random() % 33) - 1});
        for (const auto & saved : tickets) {
            bool valid = saved.epoch >= floor;
            for (int p = 0; p <= saved.last; ++p) valid &= stamps[p] <= saved.epoch;
            assert(lineage.can_restore(saved.epoch, saved.last) == valid);
        }
        if (tickets.size() > 128) tickets.erase(tickets.begin());
    }
    std::puts("KV prefix lineage: truncated/reused H rejected; earlier anchors retained; randomized oracle passed");
}
