// Standalone: c++ -std=c++17 -Icommon -Iinclude -Iggml/include tests/test-dflash-proposal.cpp -o /tmp/test-dflash-proposal && /tmp/test-dflash-proposal
#include "speculative.h"
#include <cassert>
#include <cmath>

int main() {
    const std::vector<float> q{0.8f, 0.2f};
    std::mt19937 rng(1234);
    const auto before = rng;
    assert(common_speculative_dflash_sample(q, 0.81f, rng) == -1);
    assert(rng == before); // A gated step must not consume a random draw.
    int low_probability = 0;
    constexpr int samples = 100000;
    for (int i = 0; i < samples; ++i) {
        const int token = common_speculative_dflash_sample(q, 0.3f, rng);
        assert(token == 0 || token == 1); // Never discard a sampled low-q token.
        low_probability += token == 1;
    }
    assert(std::abs(double(low_probability) / samples - q[1]) < 0.01);
    assert(q[0] == 0.8f && q[1] == 0.2f); // Rejection sampling receives original q.
    assert(common_speculative_dflash_sample(q, 0.8f, rng) >= 0);
}
