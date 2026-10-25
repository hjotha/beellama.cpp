#include "../uwp/rkva/rkva-kvarn-range.h"
#include "../uwp/rkva/rkva-kvarn-rewind.h"
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace rkva_kvarn_range;

// Revision-3 residency simulator: every resident row is tagged with
// (token, version); a rewind bumps the version of every dropped token so stale
// rows are never mistaken for the re-appended ones.
namespace {
struct PrefixSim {
    uint32_t origin, tail;
    uint32_t live = 0;
    std::vector<uint32_t> version;
    std::array<std::pair<int64_t, uint32_t>, 256> ring;
    std::array<std::pair<int64_t, uint32_t>, 384> stage;
    prefix_coverage cov;
    PrefixSim(uint32_t o, uint32_t t, uint32_t cap) : origin(o), tail(t), version(cap, 0) {
        ring.fill({-1, 0});
        stage.fill({-1, 0});
    }
    std::pair<int64_t, uint32_t> tag(uint32_t t) const {
        return {int64_t(t), version[t]};
    }
    void stage_slot_write(uint32_t t) {
        const uint32_t g = (origin + t) / 128;
        const uint32_t slot = g == 0 ? 0 : 1 + (g - 1) % 2;
        stage[slot * 128 + (origin + t) % 128] = tag(t);
    }
    void reset(uint32_t n) {
        ring.fill({-1, 0});
        stage.fill({-1, 0});
        live = n;
        // RESET stage image: sink slot plus the two newest ping-pong groups.
        const uint32_t lg = (n - 1) / 128;
        for (uint32_t t = 0; t < n; ++t) {
            const uint32_t g = t / 128;
            if ((origin + t) / 128 == 0 || g + 1 >= lg)
                stage_slot_write(t);
        }
        const uint32_t rows = std::min(n, tail);
        tail_ring_chunks(n - rows, rows, 256, [&](uint32_t src, uint32_t dst, uint32_t r) {
            for (uint32_t i = 0; i < r; ++i)
                ring[dst + i] = tag(n - rows + src + i);
        });
        cov = coverage_after_reset(n, tail);
    }
    void append(uint32_t n) {
        const uint32_t prev = live;
        stage_chunks(origin + prev, n - prev, [&](uint32_t src, uint32_t dst, uint32_t r) {
            for (uint32_t i = 0; i < r; ++i)
                stage[dst + i] = tag(prev + src + i);
        });
        if (tail)
            tail_ring_chunks(prev, n - prev, 256, [&](uint32_t src, uint32_t dst, uint32_t r) {
                for (uint32_t i = 0; i < r; ++i)
                    ring[dst + i] = tag(prev + src + i);
            });
        live = n;
        cov = coverage_after_append(cov, n, tail);
    }
    void rewind(uint32_t keep) {
        const auto span = rkva_kvarn_rewind_tail(live, keep, tail);
        for (uint32_t t = keep; t < live; ++t)
            ++version[t];
        tail_ring_chunks(span.begin, span.rows, 256, [&](uint32_t src, uint32_t dst, uint32_t r) {
            for (uint32_t i = 0; i < r; ++i)
                ring[dst + i] = tag(span.begin + src + i);
        });
        cov = coverage_after_rewind(cov, keep, span.begin, span.rows);
        live = keep;
    }
    // Ground truth: every row the revision-3 shader reads for this prefix.
    bool resident(uint32_t count) const {
        if (tail) {
            for (uint32_t t = count - std::min(count, tail); t < count; ++t)
                if (ring[t % 256] != tag(t))
                    return false;
            return true;
        }
        if (count % 128 == 0)
            return true;
        for (uint32_t t = count / 128 * 128; t < count; ++t) {
            const uint32_t g = (origin + t) / 128;
            const uint32_t slot = g == 0 ? 0 : 1 + (g - 1) % 2;
            if (stage[slot * 128 + (origin + t) % 128] != tag(t))
                return false;
        }
        return true;
    }
    // Checks the soundness of valid_prefix_queries for every batch size and
    // returns the largest accepted batch.
    uint32_t check() const {
        uint32_t best = 0;
        for (uint32_t n = 1; n <= kPrefixMaxQueries; ++n) {
            std::vector<int32_t> pos(n);
            for (uint32_t i = 0; i < n; ++i)
                pos[i] = int32_t(origin + live - n + i);
            const bool ok = valid_prefix_queries(pos.data(), n, origin, live, cov, tail);
            bool truth = true;
            for (uint32_t i = 0; i < n; ++i) {
                const int64_t count = int64_t(live) - (n - 1 - i);
                if (count > 0)
                    truth = truth && resident(uint32_t(count));
            }
            assert(!ok || truth); // never accept a prefix that is not resident
            if (ok)
                best = n;
        }
        return best;
    }
};

void prefix_tests(const rkva_kvarn_range_config& base) {
    auto c = base;
    c.layout_version = 3;
    assert(valid_descriptor(c, 256) && serial_prefix(c) && physical_tail_rows(c) == 256);
    c.precision_tail_tokens = 0;
    assert(physical_tail_rows(c) == 0);
    c = base;
    for (uint32_t v : {1u, 2u}) {
        c.layout_version = v;
        assert(!serial_prefix(c) && physical_tail_rows(c) == 128);
    }
    c.layout_version = 4;
    assert(!valid_descriptor(c, 256));
    // Physical ring placement: >256 rows keep only the newest 256, wrapped.
    std::vector<int> ring(256, -1);
    tail_ring_chunks(200, 300, 256, [&](uint32_t src, uint32_t dst, uint32_t r) {
        for (uint32_t i = 0; i < r; ++i)
            ring[dst + i] = int(200 + src + i);
    });
    for (int t = 244; t < 500; ++t)
        assert(ring[t % 256] == t);

    // Query-batch shape gates.
    const uint32_t origin = 72704, live = 300;
    const prefix_coverage full{0, 0};
    int32_t q[3] = {int32_t(origin + live - 3), int32_t(origin + live - 2),
                    int32_t(origin + live - 1)};
    assert(valid_prefix_queries(q, 3, origin, live, full, 128));
    assert(valid_prefix_queries(q, 3, origin, live, full, 0));
    assert(!valid_prefix_queries(q, 0, origin, live, full, 128));
    assert(!valid_prefix_queries(q, 3, origin, 0, full, 128));
    assert(!valid_prefix_queries(q, 3, origin, live + 1, full, 128)); // last != newest
    int32_t gap[3] = {q[0], q[0] + 2, q[2]};
    assert(!valid_prefix_queries(gap, 3, origin, live, full, 128));
    int32_t dup[3] = {q[2], q[2], q[2]};
    assert(!valid_prefix_queries(dup, 3, origin, live, full, 128));
    std::vector<int32_t> big(129);
    for (uint32_t i = 0; i < 129; ++i)
        big[i] = int32_t(origin + live - 129 + i);
    assert(!valid_prefix_queries(big.data(), 129, origin, live, full, 128));
    assert(valid_prefix_queries(big.data() + 1, 128, origin, live, full, 128));
    // Positions below origin are empty queries and need no coverage.
    int32_t early[4] = {int32_t(origin - 2), int32_t(origin - 1), int32_t(origin),
                        int32_t(origin + 1)};
    assert(valid_prefix_queries(early, 4, origin, 2, coverage_after_reset(2, 128), 128));
    int32_t extreme[2] = {INT32_MAX - 1, INT32_MAX};
    assert(!valid_prefix_queries(extreme, 2, origin, live, full, 128));

    // Deep RESET provides only the newest 128 tail rows: a multi-query batch
    // must be refused until appends have extended the ring.
    for (uint32_t tail : {0u, 128u}) {
        for (uint32_t start : {1u, 127u, 128u, 129u, 255u, 256u, 257u, 58000u}) {
            PrefixSim sim(origin, tail, 58368);
            sim.reset(start);
            const uint32_t after_reset = sim.check();
            if (tail)
                assert(after_reset == 1 || start <= 128);
            uint32_t live_now = start;
            for (uint32_t step : {1u, 2u, 3u, 5u, 9u, 9u, 9u, 128u, 1u}) {
                if (live_now + step > 58368)
                    break;
                live_now += step;
                sim.append(live_now);
                const uint32_t best = sim.check();
                assert(best >= step); // the freshly appended chunk is always verifiable
                if (tail && live_now - start >= 128)
                    assert(best == 128); // ring 256 holds every <=128-query prefix
            }
            // Rollback across seals / tail wrap, then re-append.
            for (uint32_t back : {1u, 2u, 4u, 9u, 128u}) {
                if (back > sim.live - 1)
                    continue;
                sim.rewind(sim.live - back);
                sim.check();
                sim.append(std::min(58368u, sim.live + back + 3));
                sim.check();
            }
        }
    }
    // Exhaustive single-step decode across the first three seals.
    for (uint32_t tail : {0u, 128u}) {
        PrefixSim sim(origin, tail, 58368);
        sim.reset(1);
        for (uint32_t n = 2; n <= 520; ++n) {
            sim.append(n);
            const uint32_t best = sim.check();
            assert(best >= std::min<uint32_t>(n, 9));
            if (n % 37 == 0) {
                const uint32_t back = 1 + n % 4; // user rollback N = 2..4
                sim.rewind(n - back);
                sim.check();
                sim.append(n);
                assert(sim.check() >= 1);
            }
        }
    }
    // Rewind keep=129 after before=257 with stage-only (tail 0): queries whose
    // prefix sits in group 0 need the overwritten slot and are refused.
    {
        PrefixSim sim(origin, 0, 58368);
        sim.reset(1);
        sim.append(257);
        sim.rewind(129);
        int32_t span[3] = {int32_t(origin + 126), int32_t(origin + 127), int32_t(origin + 128)};
        assert(!valid_prefix_queries(span, 3, origin, 129, sim.cov, 0)); // count 127
        assert(!sim.resident(127));
        assert(valid_prefix_queries(span + 1, 2, origin, 129, sim.cov, 0)); // counts 128, 129
        int32_t one = int32_t(origin + 128);
        assert(valid_prefix_queries(&one, 1, origin, 129, sim.cov, 0));
    }
}
} // namespace

int main() {
    rkva_kvarn_range_config c{102400, 28672, 4, 1, 128, 128, 256, 2, 4,
                              4,      8960,  3, 2, 128, 128, 1,   1, 1};
    assert(valid_descriptor(c, 256));
    auto generation_descriptor = c;
    generation_descriptor.layout_version = 2;
    assert(valid_descriptor(generation_descriptor, 256));
    generation_descriptor.layout_version = 3; // serial prefix (revision 3)
    assert(valid_descriptor(generation_descriptor, 256));
    generation_descriptor.layout_version = 4;
    assert(!valid_descriptor(generation_descriptor, 256));
    for (unsigned i = 0; i < 18; ++i) {
        auto bad = c;
        auto* words = reinterpret_cast<uint32_t*>(&bad);
        words[i] ^= 1;
        if (i == 1)
            continue;
        assert(!valid_descriptor(bad, 256));
    }
    auto zero = c;
    zero.precision_tail_tokens = 0;
    assert(valid_descriptor(zero, 256));
    zero = c;
    zero.precision_tail_tokens = 64;
    assert(!valid_descriptor(zero, 256));
    assert(!valid_descriptor(c, 128));
    auto oversized = c;
    oversized.capacity = UINT32_MAX;
    assert(!valid_descriptor(oversized, 256));
    rkva_kvarn_range_upload u{512, 0, 4, 1, 0, 0, {0, 0}};
    assert(valid_upload(u, c.capacity, 0, 0, false));
    u.flags = 0;
    assert(!valid_upload(u, c.capacity, 0, 0, false));
    u = {641, 4, 1, 0, 512, 129, {0, 0}};
    assert(valid_upload(u, c.capacity, 512, 4, true));
    u.record_groups = 0;
    assert(!valid_upload(u, c.capacity, 512, 4, true));
    u = {641, 0, 5, 1, 0, 0, {0, 0}};
    assert(valid_upload(u, c.capacity, 0, 0, false));
    u = {513, 0, 4, 1, 0, 0, {0, 0}};
    assert(valid_upload(u, c.capacity, 0, 0, false));
    u.record_groups = 5;
    assert(!valid_upload(u, c.capacity, 0, 0, false)); // no future groups
    u = {512, 0, 4, 0, 641, 0, {0, 0}};
    assert(!valid_upload(u, c.capacity, 641, 4, true));
    u = {641, 0, 5, 0, 0, 641, {0, 0}};
    assert(!valid_upload(u, c.capacity, 0, 0, false)); // rewind needs reset
    std::array<int, 384> stage;
    stage.fill(-1);
    std::array<int, 128> tail;
    tail.fill(-1);
    unsigned previous = 0;
    for (unsigned live : {1u, 127u, 128u, 129u, 255u, 256u, 257u, 512u, 641u, 769u, 1025u}) {
        stage_chunks(102400 + previous, live - previous,
                     [&](unsigned source, unsigned target, unsigned rows) {
                         for (unsigned i = 0; i < rows; ++i)
                             stage[target + i] = previous + source + i;
                     });
        tail_chunks(previous, live - previous,
                    [&](unsigned source, unsigned target, unsigned rows) {
                        for (unsigned i = 0; i < rows; ++i)
                            tail[target + i] = previous + source + i;
                    });
        for (unsigned t = 0; t < live; ++t) {
            unsigned g = (102400 + t) / 128, lg = (102400 + live - 1) / 128;
            if (g == 0 || g + 1 >= lg) {
                unsigned slot = g == 0 ? 0 : 1 + (g - 1) % 2;
                assert(stage[slot * 128 + t % 128] == int(t));
            }
        }
        for (unsigned t = live - std::min(live, 128u); t < live; ++t)
            assert(tail[t % 128] == int(t));
        previous = live;
    }
    // Full restored-tail snapshot at a nonaligned position overwrites both ring halves.
    unsigned restored = 513, begin = restored - 128;
    tail_chunks(begin, 128, [&](unsigned source, unsigned target, unsigned rows) {
        for (unsigned i = 0; i < rows; ++i)
            tail[target + i] = begin + source + i;
    });
    for (unsigned t = begin; t < restored; ++t)
        assert(tail[t % 128] == int(t));
    prefix_tests(c);
    std::puts("KVarN descriptor, coverage, append/stage/tail ring and rewind gates passed");
    std::puts("KVarN revision-3 serial-prefix residency/coverage gates passed");
}
