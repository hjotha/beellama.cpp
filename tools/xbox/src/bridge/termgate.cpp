// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/termgate.h"

namespace xllama {

StopSpan pick_boundary_spanning_stop(const std::vector<std::string>& pieces, bool prefer_last) {
    StopSpan out;
    // Candidate indices in scan order; prefer_last keeps only the last hit so
    // the stop completes at the probe's latest boundary (maximum preceding
    // MTP rounds under a deterministic greedy probe).
    std::vector<size_t> cands;
    size_t pos = 0;
    for (size_t k = 0; k + 1 < pieces.size(); ++k) {
        if (pieces[k].empty() || pieces[k + 1].empty()) {
            // A stop spanning an empty piece can never complete as a suffix
            // match (it would end in no text); skip and keep pos honest.
            pos += pieces[k].size();
            continue;
        }
        cands.push_back(pos); // text_start of a usable pair
        if (!prefer_last) {
            out.stop = pieces[k] + pieces[k + 1];
            out.text_start = pos;
            return out;
        }
        pos += pieces[k].size();
    }
    if (prefer_last && !cands.empty()) {
        // Re-walk to pair index (recompute start of the last usable pair).
        const size_t want_start = cands.back();
        size_t p2 = 0;
        for (size_t k = 0; k + 1 < pieces.size(); ++k) {
            if (!pieces[k].empty() && !pieces[k + 1].empty() && p2 == want_start) {
                out.stop = pieces[k] + pieces[k + 1];
                out.text_start = p2;
                return out;
            }
            p2 += pieces[k].size();
        }
    }
    // No valid pair: report the full length for diagnostics.
    out.stop.clear();
    out.text_start = 0;
    return out;
}

} // namespace xllama
