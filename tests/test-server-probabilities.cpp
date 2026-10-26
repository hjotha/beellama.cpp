#include "server-probabilities.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>

static void require(bool ok, const char * why) {
    if (!ok) { std::fprintf(stderr, "server probabilities: %s\n", why); std::exit(1); }
}

static double mass(const std::vector<llama_token_data> & values, bool log_roundtrip = false) {
    double sum = 0;
    for (const auto & token : values) {
        require(std::isfinite(token.p) && token.p >= 0 && token.p <= 1, "invalid probability");
        sum += log_roundtrip ? (token.p ? std::exp(double(std::log(token.p))) : 0.0) : double(token.p);
    }
    return sum;
}

int main() {
    {
        std::vector<llama_token_data> values;
        server_normalize_token_probabilities(values, -std::numeric_limits<float>::infinity());
        require(values.empty(), "empty input changed");
    }
    {
        std::vector<llama_token_data> values{{17, 0, 0}, {31, 0, 0}};
        server_normalize_token_probabilities(values, 0);
        require(values[0].p == 0.5f && values[1].p == 0.5f, "two equal logits");
        require(values[0].id == 17 && values[1].id == 31, "candidate IDs reordered");
    }
    {
        std::vector<llama_token_data> values{{0, -2, 0}, {1, -1, 0}, {2, 0, 0}};
        server_normalize_token_probabilities(values, 0);
        const double denominator = std::exp(-2.0) + std::exp(-1.0) + 1.0;
        for (size_t i = 0; i < values.size(); ++i) {
            require(std::abs(values[i].p - std::exp(double(i)-2)/denominator) < 5e-8,
                    "small F64 oracle mismatch");
        }
    }
    constexpr size_t n_vocab = 248320;
    std::vector<llama_token_data> wide(n_vocab);
    for (size_t i = 0; i < n_vocab; ++i) { wide[i] = {llama_token(i), i ? -20.0f : 0.0f, 0}; }
    {
        // Sensitivity control: this exact fixture exposes the old F32 sum.
        float old_sum = 0;
        for (const auto & token : wide) { old_sum += expf(token.logit); }
        double old_mass = 0;
        for (const auto & token : wide) { old_mass += float(expf(token.logit) / old_sum); }
        require(old_sum == 1 && old_mass > 1.0004, "wide fixture no longer exposes the regression");
        server_normalize_token_probabilities(wide, 0);
        require(std::abs(mass(wide)-1) < 1e-6, "full vocabulary is not normalized");
        require(std::abs(mass(wide, true)-1) < 1e-6, "logprob JSON roundtrip mass");
        const double expected_top = 1 / (1 + (n_vocab-1)*std::exp(-20.0));
        require(std::abs(wide[0].p-expected_top) < 5e-8, "full vocabulary oracle top probability");
    }
    {
        auto reversed = wide;
        std::reverse(reversed.begin(), reversed.end());
        server_normalize_token_probabilities(reversed, 0);
        for (size_t i = 0; i < wide.size(); ++i) {
            require(reversed[i].id == wide[wide.size()-1-i].id &&
                    reversed[i].p == wide[wide.size()-1-i].p, "input order changed normalization");
        }
    }
    {
        auto shifted = wide;
        for (auto & token : shifted) { token.logit += 100; }
        server_normalize_token_probabilities(shifted, 100);
        for (size_t i = 0; i < wide.size(); ++i) {
            require(shifted[i].p == wide[i].p, "constant logit offset changed probabilities");
        }
    }
    {
        std::vector<llama_token_data> values{{0, 0, 0}, {1, -std::numeric_limits<float>::infinity(), 0}};
        server_normalize_token_probabilities(values, 0);
        require(values[0].p == 1 && values[1].p == 0, "masked probability not zero");
    }
    std::puts("server probabilities: PASS 7 cases, including 248320-token vocabulary and logprob roundtrip");
    return 0;
}
