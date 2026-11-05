// Standalone: c++ -std=c++17 -Icommon -Iinclude -Iggml/include tests/test-dflash-proposal.cpp -o /tmp/test-dflash-proposal && /tmp/test-dflash-proposal
#include "speculative.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
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

    llama_tokens proposed{11, 12, 13};
    std::vector<common_speculative_token_dist> dists(3);
    std::vector<std::vector<llama_token_data>> result_q(3);
    dists[0].ids = {11};
    dists[1].ids = {12};
    result_q[0].push_back({11, 0.0f, 1.0f});
    result_q[1].push_back({12, 0.0f, 1.0f});
    common_speculative_draft_params params{};
    params.result = &proposed;
    params.dists = &dists;
    params.result_q = &result_q;
    params.n_max = 2;
    common_speculative_truncate_draft(params);
    assert((proposed == llama_tokens{11, 12}));
    assert(dists.size() == 2 && dists[0].ids[0] == 11 && dists[1].ids[0] == 12);
    assert(result_q.size() == 2 && result_q[0][0].id == 11 && result_q[1][0].id == 12);

    proposed.push_back(13);
    dists.resize(1);
    result_q.resize(1);
    params.n_max = 2;
    common_speculative_truncate_draft(params);
    assert(proposed.size() == 2 && dists.size() == 1 && result_q.size() == 1);

    proposed.push_back(13);
    dists.clear();
    result_q.clear();
    params.n_max = 1;
    common_speculative_truncate_draft(params);
    assert(proposed.size() == 1 && dists.empty() && result_q.empty());

    params.n_max = -1;
    proposed.push_back(12);
    common_speculative_truncate_draft(params);
    assert(proposed.size() == 2 && dists.empty() && result_q.empty());
}
