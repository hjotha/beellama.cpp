// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Termination-state gate helper (plan 003): pick a stop string that SPANS
// on_token emission boundaries, deterministically, from recorded pieces.
//
// emit_token appends one token's piece and only stops on a SUFFIX match of
// the accumulated text, so a stop that (a) starts exactly at a piece
// boundary, (b) ends exactly at a later piece boundary, and (c) covers at
// least two consecutive pieces is matched only when its final piece lands
// — i.e. the match spans emissions, never fires on a partial stop. Stripping
// then leaves output_text aligned to the token before the first covered
// piece, which is what the resume/prefill state checks compare against.
//
// Pure: WinRT-free, host-tested; the Xbox headless runner feeds it the
// recorded non-empty pieces.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace xllama {

struct StopSpan {
    std::string stop;      // empty when no valid span exists
    size_t text_start = 0; // index in the full text where |stop| begins
};

// Pair of consecutive NON-EMPTY pieces (k, k+1): stop = piece[k] +
// piece[k+1], text_start = total length of all pieces before k. Default:
// first such pair in list order. prefer_last=true picks the LAST such pair —
// the later boundary a stop must sit behind for the stop gate to have real
// preceding speculative rounds (coverage demand). Returns {"" , 0} when no
// valid pair exists (caller must fail the scenario, never fall back to a
// weaker stop silently).
StopSpan pick_boundary_spanning_stop(const std::vector<std::string>& pieces,
                                     bool prefer_last = false);

} // namespace xllama
