#include "server-direct-disk-policy.h"

#include <cstdio>
#include <cstdlib>

#define REQUIRE(expr) do { if (!(expr)) { std::fprintf(stderr, "failed: %s at line %d\n", #expr, __LINE__); std::abort(); } } while (0)
#include <cstdint>

int main() {
    using server_direct_disk_policy::configured_split;
    using server_direct_disk_policy::eligible;
    REQUIRE(configured_split(102400, "90112") == 102400);
    REQUIRE(configured_split(0, "102400") == 102400);
    REQUIRE(configured_split(0, nullptr) == 0);
    REQUIRE(configured_split(0, "") == 0);
    REQUIRE(configured_split(0, "-256") == 0);
    REQUIRE(configured_split(0, "102400junk") == 0);
    REQUIRE(configured_split(0, "102401") == 0);
    REQUIRE(configured_split(0, "4294967296") == 0);
    REQUIRE(eligible("1", true, 102399, 102400));
    REQUIRE(eligible("1", true, 102400, 102400));
    REQUIRE(!eligible(nullptr, true, 102399, 102400));
    REQUIRE(!eligible("0", true, 102399, 102400));
    REQUIRE(!eligible("true", true, 102399, 102400));
    REQUIRE(!eligible("1", false, 102399, 102400));
    REQUIRE(!eligible("1", true, 0, 102400));
    REQUIRE(!eligible("1", true, 102401, 102400));
    REQUIRE(!eligible("1", true, 102399, 0));
}
