#pragma once

#include <cmath>
#include <cstdint>

// Greedy verification rejects the selector's first choice at the first
// mismatch. Only a different token can be a useful bonus-token hypothesis.
// The prefix-times-rank-2 score is a selector heuristic, not a calibrated
// probability of target acceptance or of a pipeline hit.
struct dflash_pipeline_candidate {
    int32_t accepted = -1;
    int32_t token = -1;
    float score = 0.0f;
};

// A local pre-draft may replace noise KV only when all real rows from the
// current anchor through the predicted last accepted token are available.
inline int32_t dflash_pipeline_prefix_offset(const int32_t * positions, int32_t rows,
                                             int32_t anchor, int32_t accepted) {
    if (!positions || rows <= 0 || anchor < 0 || accepted < 0 || accepted >= rows) {
        return -1;
    }
    int32_t first = 0;
    while (first < rows && positions[first] < anchor) {
        ++first;
    }
    if (rows - first <= accepted) {
        return -1;
    }
    for (int32_t i = 0; i <= accepted; ++i) {
        if (int64_t(positions[first + i]) != int64_t(anchor) + i) {
            return -1;
        }
    }
    return first;
}

inline dflash_pipeline_candidate dflash_pipeline_rank2_step(
        const float * ids, const float * scores, int32_t n, int32_t selected,
        int32_t accepted, double & prefix, dflash_pipeline_candidate best) {
    if (!ids || !scores || n < 1 || selected < 0 || selected >= n ||
        !std::isfinite(ids[selected]) || !std::isfinite(scores[selected]) ||
        prefix <= 0.0) {
        prefix = 0.0;
        return best;
    }

    const int32_t first_id = static_cast<int32_t>(ids[selected]);
    int32_t second = -1;
    double sum = 0.0;
    for (int32_t j = 0; j < n; ++j) {
        if (!std::isfinite(scores[j]) || !std::isfinite(ids[j])) {
            prefix = 0.0;
            return best;
        }
        sum += std::exp(double(scores[j]) - double(scores[selected]));
        if (j != selected && static_cast<int32_t>(ids[j]) != first_id &&
            (second < 0 || scores[j] > scores[second])) {
            second = j;
        }
    }
    if (sum <= 0.0 || !std::isfinite(sum)) {
        prefix = 0.0;
        return best;
    }

    if (second >= 0 && accepted >= 0) {
        const double joint = prefix * std::exp(double(scores[second]) -
                                               double(scores[selected])) / sum;
        if (joint > best.score) {
            best = {accepted, static_cast<int32_t>(ids[second]),
                    static_cast<float>(joint)};
        }
    }
    prefix /= sum;
    return best;
}
