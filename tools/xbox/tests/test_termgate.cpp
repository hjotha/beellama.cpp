// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/termgate.h"

#include <string>
#include <vector>

using namespace xllama;

TEST_CASE("termgate: first adjacent non-empty pair becomes the stop span") {
    const std::vector<std::string> pieces = {"one", " two", "\n"};
    const StopSpan span = pick_boundary_spanning_stop(pieces);
    CHECK(span.stop == "one two"); // spans the boundary between piece 0 and 1
    CHECK(span.text_start == 0);
}

TEST_CASE("termgate: empty pieces never form a span (defensive only)") {
    // emit_token forwards only non-empty pieces, so empties are defensive
    // input. A pair containing an empty piece is unusable: a stop ending in
    // no text could never complete as a suffix match.
    const std::vector<std::string> pieces = {"", "abc", "", "def", "ghi"};
    const StopSpan span = pick_boundary_spanning_stop(pieces);
    CHECK(span.stop == "defghi"); // first strictly adjacent non-empty pair
    CHECK(span.text_start == 3);  // "" + "abc" before it contribute 3
}

TEST_CASE("termgate: no span when the list has no adjacent non-empty pair") {
    const std::vector<std::string> pieces = {"x", "", "yz"};
    const StopSpan span = pick_boundary_spanning_stop(pieces);
    CHECK(span.stop.empty());
    CHECK(span.text_start == 0);
}

TEST_CASE("termgate: no span when fewer than two pieces") {
    CHECK(pick_boundary_spanning_stop({}).stop.empty());
    CHECK(pick_boundary_spanning_stop({"only"}).stop.empty());
    CHECK(pick_boundary_spanning_stop({"", "", ""}).stop.empty());
}

TEST_CASE("termgate: span is deterministic and straddles an emission boundary") {
    const std::vector<std::string> pieces = {"1.", " 2", ". 3"};
    const StopSpan a = pick_boundary_spanning_stop(pieces);
    const StopSpan b = pick_boundary_spanning_stop(pieces);
    CHECK(a.stop == b.stop);
    CHECK(a.text_start == b.text_start);
    CHECK(a.stop == "1. 2");
    CHECK(a.text_start == 0);
    // The stop starts at a piece boundary and ends at a later one: it covers
    // piece 0 fully AND part of piece 1, so the suffix scanner can only fire
    // when piece 1 lands — a boundary-spanning match by construction.
    std::string rebuilt;
    for (const auto& p : pieces)
        rebuilt += p;
    CHECK(rebuilt.substr(a.text_start, a.stop.size()) == a.stop);
    // Strictly inside the stop there is a piece boundary (start(“ 2”)==3):
    CHECK(0 < a.text_start + a.stop.size());
    CHECK(rebuilt.find(a.stop) == a.text_start);
}

TEST_CASE("termgate: prefer_last picks the later boundary for stop coverage") {
    const std::vector<std::string> pieces = {"1.", " 2", ". 3", " 4", ". 5"};
    const StopSpan first = pick_boundary_spanning_stop(pieces);
    const StopSpan last = pick_boundary_spanning_stop(pieces, /*prefer_last=*/true);
    REQUIRE(first.stop == "1. 2");
    CHECK(first.text_start == 0);
    // Last usable pair = (" 4", ". 5"): stop completes at the final piece.
    REQUIRE(last.stop == " 4. 5");
    CHECK(last.text_start == 7);               // "1."(2)+" 2"(2)+". 3"(3)
    CHECK(last.text_start > first.text_start); // strictly later boundary
    // Default unchanged for callers that do not opt in.
    CHECK(pick_boundary_spanning_stop({"a", "b"}).stop == "ab");
    CHECK(pick_boundary_spanning_stop({"a", "b"}, false).stop == "ab");
    // prefer_last with no valid pair: empty stop.
    CHECK(pick_boundary_spanning_stop({"x", "", "y"}, true).stop.empty());
}

TEST_CASE("termgate: prefer_last skips empty-adjacent pairs like the default") {
    const std::vector<std::string> pieces = {"a", "", "b", "c"};
    // Usable pairs: ("b","c") only ((a, "") and ("", "b") are unusable).
    const StopSpan last = pick_boundary_spanning_stop(pieces, true);
    REQUIRE(last.stop == "bc");
    CHECK(last.text_start == 1); // "a" (+ empty contributes 0)
}
