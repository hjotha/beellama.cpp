#include "server-prefix-checkpoint-policy.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "failed: %s line %d\n", #x, __LINE__); std::abort(); } } while (0)

struct batch_fixture {
    std::vector<size_t> sizes;
    size_t saves = 0;
};

static batch_fixture simulate_batches(const char * opt_in, size_t initial, size_t total, size_t batch) {
    using namespace server_prefix_checkpoint_policy;
    batch_fixture result;
    size_t position = initial;
    while (position < total) {
        size_t queued = 0;
        while (position < total && queued < batch) {
            if (decide(opt_in, true, 256, position, queued, total, 128) == action::save) {
                ++result.saves;
            }
            ++position;
            ++queued;
        }
        result.sizes.push_back(queued);
    }
    return result;
}

int main() {
    using namespace server_prefix_checkpoint_policy;
    REQUIRE(decide(nullptr, true, 102400, 102400, 0, 102401, 98304) == action::proceed);
    REQUIRE(decide("true", true, 102400, 102400, 0, 102401, 98304) == action::proceed);
    REQUIRE(decide("1", false, 102400, 102400, 0, 102401, 98304) == action::proceed);
    REQUIRE(decide("1", true, 102400, 102399, 0, 102401, 98304) == action::proceed);
    REQUIRE(decide("1", true, 102400, 102400, 0, 102400, 98304) == action::proceed);
    REQUIRE(decide("1", true, 102400, 102400, 0, 102401, 102401) == action::proceed);
    REQUIRE(decide("1", true, 102400, 102400, 1, 102401, 98304) == action::proceed);
    REQUIRE(decide("1", true, 102400, 102400, 0, 102401, 98304) == action::save);
    REQUIRE(decide("1", true, 102400, 102401, 0, 102500, 98304) == action::proceed);
    // A batch crossing P continues unchanged; queued tokens are not a state.
    REQUIRE(decide("1", true, 256, 256, 17, 300, 128) == action::proceed);
    REQUIRE(decide("1", true, 256, 256, 0, 300, 128) == action::save);
    REQUIRE(decide("1", true, 0, 0, 0, 300, 1) == action::proceed);

    const auto aligned_off = simulate_batches(nullptr, 0, 300, 64);
    const auto aligned_on = simulate_batches("1", 0, 300, 64);
    REQUIRE(aligned_off.sizes == aligned_on.sizes);
    REQUIRE(aligned_on.saves == 1);
    const auto crossed_off = simulate_batches(nullptr, 255, 300, 64);
    const auto crossed_on = simulate_batches("1", 255, 300, 64);
    REQUIRE(crossed_off.sizes == crossed_on.sizes);
    REQUIRE(crossed_on.saves == 0);
    const auto unaligned_off = simulate_batches(nullptr, 0, 300, 100);
    const auto unaligned_on = simulate_batches("1", 0, 300, 100);
    REQUIRE(unaligned_off.sizes == unaligned_on.sizes);
    REQUIRE(unaligned_on.saves == 0);

    constexpr std::string_view pure = "auto-0123456789abcdef-0123456789abcdef-102399.bin";
    constexpr std::string_view compatible = "auto-0123456789abcdef-0123456789abcdef-102400-split-102400-162816-0123456789abcdef.bin";
    constexpr std::string_view wrong_c = "auto-0123456789abcdef-0123456789abcdef-102400-split-102400-163840-0123456789abcdef.bin";
    constexpr std::string_view wrong_p = "auto-0123456789abcdef-0123456789abcdef-98304-split-98304-162816-0123456789abcdef.bin";
    REQUIRE(checkpoint_geometry_compatible(pure, 102400, 162816));
    REQUIRE(checkpoint_geometry_compatible(compatible, 102400, 162816));
    REQUIRE(!checkpoint_geometry_compatible(wrong_c, 102400, 162816));
    REQUIRE(!checkpoint_geometry_compatible(wrong_p, 102400, 162816));
    REQUIRE(checkpoint_geometry_compatible("/test/legacy-split-102400-163840.bin", 102400, 162816));
    REQUIRE(checkpoint_geometry_compatible("C:\\test\\auto-0123456789abcdef-0123456789abcdef-102400-split-102400-162816-0123456789abcdef.bin", 102400, 162816));

    // The incompatible snapshot must never shadow a compatible candidate at
    // the same prefix length, independently of directory/index iteration order.
    struct candidate { std::string_view name; int prefix; };
    const std::array<candidate, 3> candidates{{{pure, 102399}, {compatible, 102400}, {wrong_c, 102400}}};
    std::array<int, 3> order{{0, 1, 2}};
    do {
        int best = -1;
        for (int i : order) {
            if (checkpoint_geometry_compatible(candidates[i].name, 102400, 162816) &&
                    (best < 0 || candidates[i].prefix > candidates[best].prefix)) {
                best = i;
            }
        }
        REQUIRE(best == 1);
        best = -1;
        for (int i : order) {
            if (i != 1 && checkpoint_geometry_compatible(candidates[i].name, 102400, 162816) &&
                    (best < 0 || candidates[i].prefix > candidates[best].prefix)) {
                best = i;
            }
        }
        REQUIRE(best == 0);
    } while (std::next_permutation(order.begin(), order.end()));
}
