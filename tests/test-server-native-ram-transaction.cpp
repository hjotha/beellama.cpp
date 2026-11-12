#include "server-native-ram-transaction.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "native RAM transaction: %s\n", message); std::exit(1); }
}

int main() {
    // Identical split geometry must still use the previous generic cache path
    // when the selected session is Q4/Q8/F16 rather than native KVarN records.
    check(!server_native_ram_cache_route(4, 8192, 9215, false), "non-native range selected native RAM cache");
    check(server_native_ram_cache_route(4, 8192, 9215, true), "native KVarN range did not select native RAM cache");
    check(!server_native_ram_cache_route(1, 8192, 9215, true), "parallel=1 changed cache route");
    check(!server_native_ram_cache_route(4, 8192, 8191, true), "local-only prefix selected native RAM cache");
    struct fake_slot { bool prompt_cleanup_failed = false; bool active = true; };
    std::vector<fake_slot> slots(3);
    bool unavailable = false, published = false, aborted = false;
    auto invalidate = [&] {
        server_native_ram_invalidate_shared_context(slots, unavailable,
            [&] { published = true; },
            [&] {
                aborted = true;
                for (auto & slot : slots) slot.active = false;
            });
    };
    check(server_native_ram_idle_clear_safe(true, [] { return true; }, invalidate), "successful clear invalidated context");
    check(server_native_ram_idle_clear_safe(false, [] { return false; }, invalidate), "local clear refusal invalidated context");
    check(!unavailable && !aborted, "clean clear path changed context");
    check(!server_native_ram_idle_clear_safe(false, [] { return true; }, invalidate), "poisoned clear was accepted");
    check(unavailable && published && aborted, "poisoned clear did not invalidate and publish context");
    for (const auto & slot : slots) check(slot.prompt_cleanup_failed && !slot.active, "poisoned clear left a slot available");
    // Two independent slot images. Rejected transactions must preserve B and
    // every transaction must leave the unrelated slot untouched.
    for (int failure = 0; failure != 7; ++failure) {
        int slot = 22, other = 77, clear_count = 0, rollback_count = 0;
        const auto result = server_native_ram_transaction(true,
            [&](bool backup) {
                if ((!backup && failure == 1) || (backup && failure == 2)) return LLAMA_NATIVE_RAM_REJECTED_CLEAN;
                if (!backup && failure == 3) return LLAMA_NATIVE_RAM_FAILED_INVALIDATED;
                return LLAMA_NATIVE_RAM_OK;
            },
            [&] { ++clear_count; slot = 0; return failure != 4; },
            [&](bool backup) {
                if (backup) { ++rollback_count; if (failure == 6) return LLAMA_NATIVE_RAM_FAILED_INVALIDATED; slot = 22; return LLAMA_NATIVE_RAM_OK; }
                if (failure == 5 || failure == 6) return LLAMA_NATIVE_RAM_REJECTED_CLEAN;
                slot = 11; return LLAMA_NATIVE_RAM_OK;
            });
        check(other == 77, "other slot changed");
        if (failure == 0) check(result == server_native_ram_transaction_result::restored && slot == 11, "successful restore failed");
        if (failure == 1 || failure == 2) check(result == server_native_ram_transaction_result::rejected_clean && slot == 22 && !clear_count, "validation refusal mutated slot");
        if (failure == 3 || failure == 4 || failure == 6) check(result == server_native_ram_transaction_result::invalidated, "ambiguous failure was not invalidated");
        if (failure == 5) check(result == server_native_ram_transaction_result::rejected_clean && slot == 22 && rollback_count == 1, "clean refusal did not restore B");
        if (failure == 6) check(rollback_count == 1, "rollback failure was not exercised");
    }
    int clears = 0;
    const auto empty = server_native_ram_transaction(false,
        [](bool) { return LLAMA_NATIVE_RAM_OK; }, [&] { ++clears; return true; },
        [](bool) { return LLAMA_NATIVE_RAM_OK; });
    check(empty == server_native_ram_transaction_result::restored && clears == 0, "empty destination was cleared");
    std::puts("native RAM transaction: failure and rollback cases passed");
}
