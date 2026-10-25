// Position-split range mapping (plan §§3.1-3.2, phase F2b).
//
// Every absolute position maps to exactly one range: LOCAL [0,P) on CUDA
// (KVarN4) or OVERFLOW [P,C) on Vulkan0 (Q4). This gate covers the pure range
// table the cache, store and graph paths share: per-position assignment,
// contiguous span splitting, and the §4.1 boundary occupancies
// (P-129, P-128, P-1, P, P+1, P+128). Anything outside [0,C), or any invalid
// P (including P == 0, mode off), fails closed as INVALID.

// Windows CRT headers define OVERFLOW; the range enum must remain usable.
#ifndef OVERFLOW
#define OVERFLOW 3
#define LLAMA_TEST_DEFINED_OVERFLOW
#endif
#include "llama-position-split.h"
#ifdef LLAMA_TEST_DEFINED_OVERFLOW
#undef OVERFLOW
#undef LLAMA_TEST_DEFINED_OVERFLOW
#endif

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::printf("  FAIL line %d: %s\n", __LINE__, #cond); \
    } \
} while (0)

using llama_position_split::range_kind;
using llama_position_split::rm_span_kind;

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    constexpr uint32_t P = 1024;
    constexpr uint32_t C = llama_position_split::kCapacityC;

    // Per-position assignment.
    {
        auto r = llama_position_split::range_for_pos(0, P, C);
        CHECK(r.kind == range_kind::LOCAL && r.index_in_range == 0);
    }
    {
        auto r = llama_position_split::range_for_pos(P - 1, P, C);
        CHECK(r.kind == range_kind::LOCAL && r.index_in_range == P - 1);
    }
    {
        auto r = llama_position_split::range_for_pos(P, P, C);
        CHECK(r.kind == range_kind::OVERFLOW_RANGE && r.index_in_range == 0);
    }
    {
        auto r = llama_position_split::range_for_pos(P + 1, P, C);
        CHECK(r.kind == range_kind::OVERFLOW_RANGE && r.index_in_range == 1);
    }
    {
        auto r = llama_position_split::range_for_pos(C - 1, P, C);
        CHECK(r.kind == range_kind::OVERFLOW_RANGE && r.index_in_range == C - 1 - P);
    }
    {
        auto r = llama_position_split::range_for_pos(C, P, C);
        CHECK(r.kind == range_kind::INVALID);
    }
    {
        auto r = llama_position_split::range_for_pos(-1, P, C);
        CHECK(r.kind == range_kind::INVALID);
    }
    // Invalid P (off, unaligned, out of range) fails closed for every pos.
    for (uint32_t bad_p : {0u, 100u, C, C + 256u}) {
        for (int64_t pos : {int64_t(0), int64_t(P), int64_t(C) - 1}) {
            auto r = llama_position_split::range_for_pos(pos, bad_p, C);
            CHECK(r.kind == range_kind::INVALID);
        }
    }
    // Target production P is itself valid and maps the same way.
    {
        constexpr uint32_t PP = llama_position_split::kTargetP;
        auto r0 = llama_position_split::range_for_pos(0, PP, C);
        auto r1 = llama_position_split::range_for_pos(PP, PP, C);
        CHECK(r0.kind == range_kind::LOCAL && r0.index_in_range == 0);
        CHECK(r1.kind == range_kind::OVERFLOW_RANGE && r1.index_in_range == 0);
    }

    // Contiguous span splitting.
    auto split_ok = [](int64_t start, uint64_t n, uint32_t lc, uint32_t oc) {
        uint32_t got_lc = 0, got_oc = 0;
        std::string err;
        const bool ok = llama_position_split::split_span_at_p(start, n, P, C, got_lc, got_oc, err);
        CHECK(ok);
        CHECK(err.empty());
        CHECK(got_lc == lc);
        CHECK(got_oc == oc);
    };
    split_ok(0, 100, 100, 0);            // fully below P
    split_ok(2000, 100, 0, 100);         // fully above P
    split_ok(1000, 100, 24, 76);         // crosses P
    split_ok(P, 50, 0, 50);              // starts exactly at P
    split_ok(P - 50, 50, 50, 0);         // ends exactly at P
    split_ok(0, P, P, 0);                // [0,P)
    split_ok(0, P + 1, P, 1);            // crosses by one
    split_ok(P - 129, 129, 129, 0);      // §4.1: ends at P-? (P-129+129 = P)
    split_ok(P - 128, 256, 128, 128);    // §4.1: straddles P symmetrically
    split_ok(P - 1, 2, 1, 1);            // §4.1: minimal crossing
    split_ok(P + 1, 127, 0, 127);        // §4.1: just above P
    split_ok(P, 128, 0, 128);            // §4.1: P+128 worth of overflow
    split_ok(C - 128, 128, 0, 128);      // tail of the context
    // Fail-closed cases.
    {
        uint32_t lc = 0, oc = 0;
        std::string err;
        CHECK(!llama_position_split::split_span_at_p(0, 0, P, C, lc, oc, err));
        CHECK(!err.empty());
        CHECK(!llama_position_split::split_span_at_p(-5, 10, P, C, lc, oc, err));
        CHECK(!err.empty());
        CHECK(!llama_position_split::split_span_at_p(C - 10, 11, P, C, lc, oc, err));
        CHECK(!err.empty());
        CHECK(!llama_position_split::split_span_at_p(C, 1, P, C, lc, oc, err));
        CHECK(!err.empty());
        CHECK(!llama_position_split::split_span_at_p(0, 10, 0, C, lc, oc, err));
        CHECK(!err.empty());
    }
    // Span/range consistency: every position of a crossing span agrees with
    // the per-position assignment.
    {
        uint32_t lc = 0, oc = 0;
        std::string err;
        CHECK(llama_position_split::split_span_at_p(P - 10, 20, P, C, lc, oc, err));
        CHECK(lc == 10 && oc == 10);
        for (int64_t pos = P - 10; pos < P + 10; ++pos) {
            auto r = llama_position_split::range_for_pos(pos, P, C);
            if (pos < int64_t(P)) {
                CHECK(r.kind == range_kind::LOCAL && r.index_in_range == uint32_t(pos));
            } else {
                CHECK(r.kind == range_kind::OVERFLOW_RANGE && r.index_in_range == uint32_t(pos - P));
            }
        }
    }

    // Removal/rollback classification (§§3.1-3.2): which ranges a [p0,p1)
    // removal touches, including the §4.1 boundary occupancies. The cache
    // decides KVarN representability/checkpoint from this classification.
    {
        using llama_position_split::classify_removal;
        CHECK(classify_removal(0, P, P, C) == rm_span_kind::LOCAL_ONLY);
        CHECK(classify_removal(P, C, P, C) == rm_span_kind::OVERFLOW_ONLY);
        CHECK(classify_removal(0, C, P, C) == rm_span_kind::CROSSING);
        CHECK(classify_removal(P - 1, P + 1, P, C) == rm_span_kind::CROSSING);
        CHECK(classify_removal(P - 129, P, P, C) == rm_span_kind::LOCAL_ONLY);
        CHECK(classify_removal(P - 128, P + 128, P, C) == rm_span_kind::CROSSING);
        CHECK(classify_removal(P - 1, P, P, C) == rm_span_kind::LOCAL_ONLY);
        CHECK(classify_removal(P, P + 1, P, C) == rm_span_kind::OVERFLOW_ONLY);
        CHECK(classify_removal(P, P + 128, P, C) == rm_span_kind::OVERFLOW_ONLY);
        CHECK(classify_removal(P + 1, P + 128, P, C) == rm_span_kind::OVERFLOW_ONLY);
        CHECK(classify_removal(C - 128, C, P, C) == rm_span_kind::OVERFLOW_ONLY);
        CHECK(classify_removal(0, -1, P, C) == rm_span_kind::CROSSING);
        CHECK(classify_removal(P - 10, -1, P, C) == rm_span_kind::CROSSING);
        CHECK(classify_removal(P, -1, P, C) == rm_span_kind::OVERFLOW_ONLY);
        // Fail-closed: empty, inverted, negative, beyond C, bad P.
        CHECK(classify_removal(P, P, P, C) == rm_span_kind::INVALID);
        CHECK(classify_removal(P + 1, P, P, C) == rm_span_kind::INVALID);
        CHECK(classify_removal(-1, 10, P, C) == rm_span_kind::INVALID);
        CHECK(classify_removal(0, C + 1, P, C) == rm_span_kind::INVALID);
        CHECK(classify_removal(C, C + 1, P, C) == rm_span_kind::INVALID);
        CHECK(classify_removal(0, 10, 0, C) == rm_span_kind::INVALID);
        CHECK(classify_removal(0, 10, 100, C) == rm_span_kind::INVALID);
    }

    // Transactional removal plan (§4.1): the whole plan is validated before any
    // mutation, so a rejected span never leaves partial state. A reserve that
    // cannot represent a step in-place escalates to checkpoint/replay instead
    // of silently dropping tokens.
    {
        using namespace llama_position_split;
        // Large exact reserve: single-range removals are in-place.
        {
            removal_plan plan;
            uint32_t reserve = 0;
            CHECK(plan_removal(0, P, P, C, C, reserve, plan) == removal_disposition::APPLY);
            CHECK(reserve == C);
            CHECK(plan.steps.size() == 1);
            CHECK(plan.steps[0].range_id == kRangeLocal);
            CHECK(plan.steps[0].start_index == 0);
            CHECK(plan.steps[0].n_tokens == P);
            CHECK(plan.steps[0].suffix);
        }
        {
            removal_plan plan;
            uint32_t reserve = 0;
            CHECK(plan_removal(P, C, P, C, C, reserve, plan) == removal_disposition::APPLY);
            CHECK(plan.steps.size() == 1);
            CHECK(plan.steps[0].range_id == kRangeOverflow);
            CHECK(plan.steps[0].start_index == 0);
            CHECK(plan.steps[0].n_tokens == C - P);
            CHECK(plan.steps[0].suffix);
        }
        // Crossing P yields exactly two steps, local first, no overlap/gap.
        {
            removal_plan plan;
            uint32_t reserve = 0;
            CHECK(plan_removal(P - 128, P + 128, P, C, C, reserve, plan) == removal_disposition::APPLY);
            CHECK(plan.steps.size() == 2);
            CHECK(plan.steps[0].range_id == kRangeLocal);
            CHECK(plan.steps[0].start_index == P - 128);
            CHECK(plan.steps[0].n_tokens == 128);
            CHECK(plan.steps[0].suffix);
            CHECK(plan.steps[1].range_id == kRangeOverflow);
            CHECK(plan.steps[1].start_index == 0);
            CHECK(plan.steps[1].n_tokens == 128);
            CHECK(!plan.steps[1].suffix);
            uint32_t total = plan.steps[0].n_tokens + plan.steps[1].n_tokens;
            CHECK(total == 256);
        }
        // Minimal crossing P-1..P+1: one token on each side.
        {
            removal_plan plan;
            uint32_t reserve = 0;
            CHECK(plan_removal(P - 1, P + 1, P, C, C, reserve, plan) == removal_disposition::APPLY);
            CHECK(plan.steps.size() == 2);
            CHECK(plan.steps[0].n_tokens == 1 && plan.steps[0].suffix);
            CHECK(plan.steps[1].n_tokens == 1 && !plan.steps[1].suffix);
        }
        // The llama.cpp p1 < 0 removal sentinel means "through C".
        {
            removal_plan plan;
            uint32_t reserve = 0;
            CHECK(plan_removal(P - 10, -1, P, C, C, reserve, plan) == removal_disposition::APPLY);
            CHECK(plan.shape == rm_span_kind::CROSSING);
            CHECK(plan.steps.size() == 2);
            CHECK(plan.steps[0].start_index == P - 10);
            CHECK(plan.steps[0].n_tokens == 10);
            CHECK(plan.steps[0].suffix);
            CHECK(plan.steps[1].start_index == 0);
            CHECK(plan.steps[1].n_tokens == C - P);
            CHECK(plan.steps[1].suffix);
        }
        // Non-suffix removal from the middle needs replay beyond the reserve.
        {
            removal_plan plan;
            uint32_t reserve = 0;
            CHECK(plan_removal(10, 20, P, C, 8, reserve, plan) == removal_disposition::NEEDS_CHECKPOINT);
            CHECK(reserve == 8);
            CHECK(plan.steps.size() == 1);
            CHECK(!plan.steps[0].suffix);
        }
        {
            removal_plan plan;
            uint32_t reserve = 0;
            CHECK(plan_removal(P + 1, P + 128, P, C, 64, reserve, plan) == removal_disposition::NEEDS_CHECKPOINT);
            CHECK(plan.steps.size() == 1);
            CHECK(plan.steps[0].range_id == kRangeOverflow);
            CHECK(plan.steps[0].start_index == 1);
            CHECK(plan.steps[0].n_tokens == 127);
        }
        // In-place when the non-suffix fits the reserve exactly.
        {
            removal_plan plan;
            uint32_t reserve = 0;
            CHECK(plan_removal(P + 1, P + 128, P, C, 128, reserve, plan) == removal_disposition::APPLY);
        }
        // Reserve is clamped to C and a zero reserve escalates everything.
        {
            removal_plan plan;
            uint32_t reserve = 0;
            CHECK(plan_removal(0, 8, P, C, UINT32_MAX, reserve, plan) == removal_disposition::APPLY);
            CHECK(reserve == C);
            removal_plan plan2;
            uint32_t reserve2 = 0;
            CHECK(plan_removal(0, 8, P, C, 0, reserve2, plan2) == removal_disposition::NEEDS_CHECKPOINT);
            CHECK(reserve2 == 0);
        }
        // Fail-closed plans are empty: no step can be half-applied.
        for (auto bad : {std::pair<int64_t, int64_t>{P, P},
                         {P + 1, P},
                         {-1, 4},
                         {0, int64_t(C) + 1},
                         {C, int64_t(C) + 1}}) {
            removal_plan plan;
            uint32_t reserve = 0;
            CHECK(plan_removal(bad.first, bad.second, P, C, C, reserve, plan) ==
                  removal_disposition::REJECT);
            CHECK(plan.steps.empty());
        }
        for (uint32_t bad_p : {0u, 100u, C}) {
            removal_plan plan;
            uint32_t reserve = 0;
            CHECK(plan_removal(0, 8, bad_p, C, C, reserve, plan) == removal_disposition::REJECT);
            CHECK(plan.steps.empty());
        }
        // Plan steps partition the span exactly for every boundary occupancy.
        for (int64_t start : {int64_t(0), int64_t(P) - 129, int64_t(P) - 128, int64_t(P) - 1,
                              int64_t(P), int64_t(P) + 1, int64_t(P) + 128}) {
            for (uint64_t n : {1ull, 2ull, 127ull, 128ull, 256ull, 1000ull}) {
                if (uint64_t(start) + n > C) {
                    continue;
                }
                removal_plan plan;
                uint32_t reserve = 0;
                CHECK(plan_removal(start, start + int64_t(n), P, C, C, reserve, plan) ==
                      removal_disposition::APPLY);
                uint64_t covered = 0;
                for (const auto & step : plan.steps) {
                    covered += step.n_tokens;
                }
                CHECK(covered == n);
                for (const auto & step : plan.steps) {
                    const uint64_t abs_lo = step.range_id == kRangeLocal
                        ? step.start_index
                        : uint64_t(P) + step.start_index;
                    CHECK(abs_lo >= uint64_t(start));
                    CHECK(abs_lo + step.n_tokens <= uint64_t(start) + n);
                }
            }
        }
    }

    // Envelope v2 descriptor table: exactly two ranges per full layer, no
    // gaps/overlaps, with the backend/format pairing the plan fixes.
    {
        using namespace llama_position_split;
        std::vector<uint32_t> full_layers = {0, 1, 2};
        std::vector<range_desc> descs;
        for (uint32_t lid : full_layers) {
            descs.push_back({lid, kRangeLocal, 0, P, 0, 0});
            descs.push_back({lid, kRangeOverflow, P, C, 1, 1});
        }
        CHECK(validate_ranges(descs, P, C, full_layers).empty());
        {
            auto q8 = descs;
            for (auto & desc : q8) {
                if (desc.range_id == kRangeOverflow) desc.format = 2;
            }
            CHECK(validate_ranges(q8, P, C, full_layers).empty());
            q8[1].format = 3;
            CHECK(validate_ranges(q8, P, C, full_layers).empty());
            q8[1].format = 4;
            CHECK(!validate_ranges(q8, P, C, full_layers).empty());
        }
        // Missing overflow range.
        {
            std::vector<range_desc> bad = descs;
            bad.pop_back();
            CHECK(!validate_ranges(bad, P, C, full_layers).empty());
        }
        // Wrong boundary / duplicate / gap / backend-format swap.
        {
            std::vector<range_desc> bad = descs;
            bad[1].end = P - 1;
            CHECK(!validate_ranges(bad, P, C, full_layers).empty());
        }
        {
            std::vector<range_desc> bad = descs;
            bad[1].start = P + 1;
            CHECK(!validate_ranges(bad, P, C, full_layers).empty());
        }
        {
            std::vector<range_desc> bad = descs;
            bad[2] = bad[1];
            CHECK(!validate_ranges(bad, P, C, full_layers).empty());
        }
        {
            std::vector<range_desc> bad = descs;
            bad[0].backend = 1;
            CHECK(!validate_ranges(bad, P, C, full_layers).empty());
        }
        {
            std::vector<range_desc> bad = descs;
            bad[1].backend = 0;
            bad[1].format = 0;
            CHECK(!validate_ranges(bad, P, C, full_layers).empty());
        }
        {
            std::vector<range_desc> bad = descs;
            bad[1].range_id = 7;
            CHECK(!validate_ranges(bad, P, C, full_layers).empty());
        }
        // Bad P is rejected regardless of the descriptors.
        for (uint32_t bad_p : {0u, 100u, C}) {
            CHECK(!validate_ranges(descs, bad_p, C, full_layers).empty());
        }
        // Target P also validates.
        {
            std::vector<range_desc> big;
            for (uint32_t lid : full_layers) {
                big.push_back({lid, kRangeLocal, 0, kTargetP, 0, 0});
                big.push_back({lid, kRangeOverflow, kTargetP, C, 1, 1});
            }
            CHECK(validate_ranges(big, kTargetP, C, full_layers).empty());
        }
    }

    std::printf("position-split-ranges: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
