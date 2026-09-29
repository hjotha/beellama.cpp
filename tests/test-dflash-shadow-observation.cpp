#include "dflash-shadow-observation.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>

static dflash_shadow_observation observation() {
    dflash_shadow_observation result;
    result.id = 7;
    result.epoch = 9;
    result.pos0 = 100;
    result.anchor = 10;
    return result;
}

static void test_defaults() {
    const dflash_shadow_observation obs;
    assert(obs.id == 0 && obs.epoch == 0);
    assert(obs.pos0 == -1 && obs.anchor == -1);
    assert(obs.proposed.empty() && obs.confirmed.empty());
    assert(obs.done_us == 0 && obs.decision_us == 0);
    assert(obs.next_pos == -1 && obs.next_anchor == -1 && obs.capacity == 0);
    assert(!obs.completed && !obs.committed && !obs.decided && !obs.resolved());
    const auto result = obs.evaluate();
    assert(!result.ready && !result.prefix_match && !result.position_match && !result.usable);
    assert(result.matched == 0 && result.remaining == 0 && result.usable_tokens == 0);
    const dflash_shadow_observation::evaluation empty;
    assert(!empty.ready && !empty.prefix_match && !empty.position_match && !empty.usable);
    assert(empty.matched == 0 && empty.remaining == 0 && empty.usable_tokens == 0);
}

static void test_full_acceptance_and_bonus() {
    auto obs = observation();
    obs.finish({11, 12, 13, 14, 15, 16}, 80);
    // The target accepts all three active draft tokens and emits bonus 14.
    obs.commit({11, 12, 13, 14});
    obs.decide(104, 14, 8, 100);
    const auto result = obs.evaluate();
    assert(obs.resolved());
    assert(result.ready && result.prefix_match && result.position_match && result.usable);
    assert(result.matched == 4 && result.remaining == 2 && result.usable_tokens == 2);
}

static void test_partial_rejection_and_replacement_bonus() {
    auto obs = observation();
    obs.finish({11, 42, 43, 44}, 80);
    // Active draft {11, 12, 13} rejects 12; the actual replacement bonus is 42.
    obs.commit({11, 42});
    obs.decide(102, 42, 8, 100);
    const auto result = obs.evaluate();
    assert(obs.resolved());
    assert(result.ready && result.prefix_match && result.position_match && result.usable);
    assert(result.matched == 2 && result.remaining == 2 && result.usable_tokens == 2);

    auto divergent = observation();
    divergent.finish({11, 12, 13, 14}, 80);
    divergent.commit({11, 42});
    divergent.decide(102, 42, 8, 100);
    const auto rejected = divergent.evaluate();
    assert(rejected.ready && rejected.position_match);
    assert(!rejected.prefix_match && !rejected.usable && rejected.usable_tokens == 0);
    assert(rejected.matched == 1 && rejected.remaining == 2);
}

static void test_divergence_at_bonus() {
    auto obs = observation();
    obs.finish({11, 12, 13, 90, 15}, 80);
    obs.commit({11, 12, 13, 14});
    obs.decide(104, 14, 8, 100);
    const auto result = obs.evaluate();
    assert(obs.resolved() && result.ready && result.position_match);
    assert(!result.prefix_match && !result.usable && result.usable_tokens == 0);
    assert(result.matched == 3 && result.remaining == 1);
}

static void test_all_event_orders() {
    const std::vector<int32_t> proposed = {11, 12, 13, 14};
    const std::vector<int32_t> confirmed = {11, 12};
    int order[] = {0, 1, 2};
    do {
        auto obs = observation();
        for (int step = 0; step < 3; ++step) {
            switch (order[step]) {
                case 0: obs.finish(proposed, 80); break;
                case 1: obs.commit(confirmed); break;
                case 2: obs.decide(102, 12, 8, 100); break;
            }
            const auto result = obs.evaluate();
            assert(obs.resolved() == (step == 2));
            assert(result.usable == (step == 2));
            assert(result.usable_tokens == (step == 2 ? 2 : 0));
            assert(result.ready == (obs.completed && obs.decided));
            // Evaluating an incomplete observation must retain either event's tokens.
            assert(obs.completed ? obs.proposed == proposed : obs.proposed.empty());
            assert(obs.committed ? obs.confirmed == confirmed : obs.confirmed.empty());
            assert(obs.id == 7 && obs.epoch == 9 && obs.pos0 == 100 && obs.anchor == 10);
        }
        const auto result = obs.evaluate();
        assert(result.ready && result.prefix_match && result.position_match);
        assert(result.matched == 2 && result.remaining == 2);
    } while (std::next_permutation(order, order + 3));
}

static void test_completion_before_commit_is_retained() {
    auto obs = observation();
    const std::vector<int32_t> proposed = {11, 12, 13};
    obs.finish(proposed, 80);
    auto result = obs.evaluate();
    assert(!obs.resolved() && !result.ready && !result.usable);
    assert(result.remaining == 3 && obs.proposed == proposed);
    obs.decide(102, 12, 8, 100);
    result = obs.evaluate();
    assert(result.ready && !result.prefix_match && !result.usable);
    assert(!obs.resolved() && obs.proposed == proposed && obs.done_us == 80);
    obs.commit({11, 12});
    result = obs.evaluate();
    assert(obs.resolved() && result.usable && result.usable_tokens == 1);
    assert(obs.proposed == proposed && obs.done_us == 80);
}

static void test_deadline() {
    for (int64_t done_us : {99, 100, 101}) {
        auto obs = observation();
        obs.commit({11, 12});
        obs.decide(102, 12, 1, 100);
        assert(!obs.resolved() && !obs.evaluate().ready);
        // Delivery can follow the decision; readiness uses the completion timestamp.
        obs.finish({11, 12, 13, 14}, done_us);
        const auto result = obs.evaluate();
        assert(obs.resolved() && result.prefix_match && result.position_match);
        assert(result.ready == (done_us <= 100));
        assert(result.usable == (done_us <= 100));
        assert(result.usable_tokens == (done_us <= 100 ? 1 : 0));
        obs.decide(102, 12, 8, 200); // A later poll must not make a late result ready.
        assert(obs.decision_us == 100 && obs.capacity == 1);
        const auto polled = obs.evaluate();
        assert(polled.ready == result.ready && polled.usable == result.usable);
        assert(polled.usable_tokens == result.usable_tokens);
        obs.decide(999, 99, 0, 300);
        assert(obs.next_pos == 102 && obs.next_anchor == 12);
        assert(obs.decision_us == 100 && obs.capacity == 1);
        assert(obs.evaluate().usable == result.usable);
    }
}

static void test_wrong_position_or_anchor() {
    for (int32_t position : {101, 102, 103}) {
        for (int32_t anchor : {10, 12, 99}) {
            auto obs = observation();
            obs.finish({11, 12, 13}, 80);
            obs.commit({11, 12});
            obs.decide(position, anchor, 8, 100);
            const auto result = obs.evaluate();
            const bool matches = position == 102 && anchor == 12;
            assert(obs.resolved() && result.ready && result.prefix_match);
            assert(result.position_match == matches && result.usable == matches);
            assert(result.usable_tokens == (matches ? 1 : 0));
        }
    }
}

static void test_empty_or_exhausted_prefix() {
    auto exhausted = observation();
    exhausted.finish({11, 12}, 80);
    exhausted.commit({11, 12});
    exhausted.decide(102, 12, 8, 100);
    const auto full = exhausted.evaluate();
    assert(exhausted.resolved() && full.ready && full.prefix_match && full.position_match);
    assert(full.matched == 2 && full.remaining == 0 && !full.usable && full.usable_tokens == 0);

    auto longer = observation();
    longer.finish({11}, 80);
    longer.commit({11, 12});
    longer.decide(102, 12, 8, 100);
    const auto short_proposal = longer.evaluate();
    assert(longer.resolved() && short_proposal.ready && short_proposal.position_match);
    assert(!short_proposal.prefix_match && short_proposal.matched == 1 && short_proposal.remaining == 0);
    assert(!short_proposal.usable && short_proposal.usable_tokens == 0);

    auto empty = observation();
    empty.finish({11, 12}, 80);
    empty.commit({});
    empty.decide(100, 10, 8, 100);
    const auto no_prefix = empty.evaluate();
    assert(empty.resolved() && no_prefix.ready);
    assert(!no_prefix.prefix_match && !no_prefix.position_match && !no_prefix.usable);
    assert(no_prefix.matched == 0 && no_prefix.remaining == 2 && no_prefix.usable_tokens == 0);

    auto no_tokens = observation();
    no_tokens.finish({}, 80);
    no_tokens.commit({11});
    no_tokens.decide(101, 11, 8, 100);
    const auto no_proposal = no_tokens.evaluate();
    assert(no_tokens.resolved() && no_proposal.ready && no_proposal.position_match);
    assert(!no_proposal.prefix_match && !no_proposal.usable);
    assert(no_proposal.matched == 0 && no_proposal.remaining == 0 && no_proposal.usable_tokens == 0);
}

static void test_capacity_clamp() {
    for (int32_t capacity : {-1, 0, 1, 2, 8}) {
        auto obs = observation();
        obs.finish({11, 12, 13, 14}, 80);
        obs.commit({11, 12});
        obs.decide(102, 12, capacity, 100);
        const auto result = obs.evaluate();
        assert(obs.resolved() && result.ready && result.prefix_match && result.position_match);
        assert(result.remaining == 2 && result.usable == (capacity > 0));
        assert(result.usable_tokens == (capacity <= 0 ? 0 : capacity == 1 ? 1 : 2));
    }
}

int main() {
    test_defaults();
    test_full_acceptance_and_bonus();
    test_partial_rejection_and_replacement_bonus();
    test_divergence_at_bonus();
    test_all_event_orders();
    test_completion_before_commit_is_retained();
    test_deadline();
    test_wrong_position_or_anchor();
    test_empty_or_exhausted_prefix();
    test_capacity_clamp();
}
