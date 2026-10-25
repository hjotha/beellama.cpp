// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/replay.h"

#include <cmath>
#include <limits>
#include <vector>

using namespace xllama;

TEST_CASE("replay: row comparison reports maxabs/maxrel/argmax") {
    const std::vector<float> a = {1.0f, 3.0f, 2.0f, -4.0f};
    const std::vector<float> b = {1.0f, 3.0f, 2.0f, -4.0f};
    ReplayCmp c = compare_replay_rows(a, b);
    CHECK(c.maxabs == doctest::Approx(0.0));
    CHECK(c.maxrel == doctest::Approx(0.0));
    CHECK(c.argmax_match);

    const std::vector<float> d = {1.0f, 3.0f, 2.0f, 4.0f}; // flipped top-1 sign
    c = compare_replay_rows(a, d);
    CHECK(c.maxabs == doctest::Approx(8.0));
    CHECK(c.maxrel == doctest::Approx(2.0)); // 8 / max|a| = 8/4
    CHECK_FALSE(c.argmax_match);

    const std::vector<float> e = {1.0f, 3.0f, 2.0f, -4.0f + 1e-6f};
    c = compare_replay_rows(a, e);
    CHECK(c.argmax_match);
    CHECK(c.maxabs < 1e-5);

    // Negative-logit control: argmax is the largest VALUE (index 1, value 2),
    // not the largest magnitude (index 0, value -100).
    const std::vector<float> neg = {-100.0f, 2.0f, 1.0f};
    const std::vector<float> neg_same = {-100.0f, 2.0f, 1.0f};
    c = compare_replay_rows(neg, neg_same);
    CHECK(c.argmax_match);
    CHECK(c.maxabs == doctest::Approx(0.0));
    const std::vector<float> neg_flip = {-100.0f, 1.0f, 2.0f};
    c = compare_replay_rows(neg, neg_flip);
    CHECK_FALSE(c.argmax_match);

    // Non-finite input: no comparison, never a match.
    const std::vector<float> inf = {1.0f, std::numeric_limits<float>::infinity(), 2.0f};
    c = compare_replay_rows(inf, inf);
    CHECK(c.maxabs == doctest::Approx(0.0));
    CHECK_FALSE(c.argmax_match);

    // Size mismatch or empty input: zero comparison, never a match.
    c = compare_replay_rows(a, {1.0f, 2.0f});
    CHECK(c.maxabs == doctest::Approx(0.0));
    CHECK_FALSE(c.argmax_match);
    c = compare_replay_rows({}, {});
    CHECK_FALSE(c.argmax_match);
}
