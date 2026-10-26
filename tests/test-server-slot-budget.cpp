#include "server-slot-budget.h"

#include <cstdio>
#include <cstdlib>

static int checks = 0;
static void check(bool ok) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "slot budget check %d failed\n", checks); std::exit(1); }
}

int main() {
    for (const auto & row : {std::string("MemAvailable: 0 kB\n"),
                            std::string("MemFree: 1 kB\nMemAvailable: 4096 kB\n")}) {
        std::istringstream input(row); uint64_t bytes = 17;
        check(server_slot_parse_mem_available(input, bytes));
        check(bytes == (row[3] == 'A' ? 0 : 4096ULL * 1024));
    }
    for (const char * row : {"", "MemFree: 4096 kB\n", "MemAvailable: -1 kB\n",
                            "MemAvailable: +1 kB\n", "MemAvailable: -0 kB\n",
                            "MemAvailable: 1 MB\n", "MemAvailable: 1 kB extra\n",
                            "MemAvailable: 18446744073709551616 kB\n",
                            "MemAvailable: 18446744073709551615 kB\n",
                            "MemAvailable: 1x kB\n", "MemAvailable: kB\n"}) {
        std::istringstream input(row); uint64_t bytes = 17;
        check(!server_slot_parse_mem_available(input, bytes)); check(bytes == 17);
    }
    {
        std::istringstream input("MemAvailable: 18014398509481983 kB\n"); uint64_t bytes = 0;
        check(server_slot_parse_mem_available(input, bytes));
        check(bytes == std::numeric_limits<uint64_t>::max() - 1023);
    }
    check(server_slot_cap_ram_budget(0, 100, true) == 0);
    check(server_slot_cap_ram_budget(100, 0, true) == 75);
    check(server_slot_cap_ram_budget(100, 50, true) == 50);
    check(server_slot_cap_ram_budget(100, 1000, true) == 75);
    check(server_slot_cap_ram_budget(3, 0, true) == 3);
    check(server_slot_cap_ram_budget(8ULL << 30, 0, false) == (8ULL << 30));
    check(server_slot_cap_ram_budget(8ULL << 30, 4ULL << 30, false) == (4ULL << 30));
    // A missing/invalid Linux field preserves the existing measured-free fallback.
    std::istringstream missing("MemFree: 1024 kB\n"); uint64_t free_fallback = 4096;
    const bool found = server_slot_parse_mem_available(missing, free_fallback);
    check(!found && server_slot_cap_ram_budget(free_fallback, 0, true) == 3072);
    std::printf("PASS server slot RAM budget: %d checks\n", checks);
}
