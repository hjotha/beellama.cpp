#pragma once

#include "llama.h"

#include <cmath>
#include <vector>

// Response diagnostics only: sampling has already selected the output token.
// Keep exp/probabilities F32, but retain the small terms of a full vocabulary
// when accumulating its denominator after candidates were sorted descending.
inline void server_normalize_token_probabilities(std::vector<llama_token_data> & candidates, float max_logit) {
    double sum = 0.0;
    for (auto & token : candidates) {
        token.p = expf(token.logit - max_logit);
        sum += double(token.p);
    }
    for (auto & token : candidates) {
        token.p = float(double(token.p) / sum);
    }
}
