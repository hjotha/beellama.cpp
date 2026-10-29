// CPU/offline contract for the explicit adaptive native streaming slot API and
// its bounded restore RAM estimator. No model, GPU or remote peer.
#include "../tools/server/server-native-slot.h"
#include "../tools/server/server-slot-budget.h"
#include "../ggml/include/rkva-native-state.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void test_failed_restore_cleanup() {
    struct slot {
        bool memory_cleared = false;
        bool expect_memory_cleared = true;
        bool prompt_present = true;
        bool logits_present = true;
        int speculative_resets = 0;
        void prompt_forget_memory() {
            if (expect_memory_cleared) assert(memory_cleared);
            prompt_present = false;
            logits_present = false;
            ++speculative_resets;
        }
    } s;
    int memory_clears = 0;
    int remote_resets = 0;
    server_native_slot::clear_failed_restore(s, [&]() {
        ++memory_clears;
        ++remote_resets; // actual memory clear invokes the remote reset hook
        s.memory_cleared = true;
    });
    assert(memory_clears == 1 && remote_resets == 1);
    assert(!s.prompt_present && !s.logits_present && s.speculative_resets == 1);

    slot failed;
    failed.expect_memory_cleared = false;
    bool thrown = false;
    try {
        server_native_slot::clear_failed_restore(failed, [&]() {
            throw std::runtime_error("memory clear failed");
        });
    } catch (const std::runtime_error &) { thrown = true; }
    assert(thrown && !failed.prompt_present && !failed.logits_present && failed.speculative_resets == 1);
}

server_native_slot::Eligibility valid_save() {
    server_native_slot::Eligibility e;
    e.native_stream = true;
    e.adaptive = true;
    e.profile_xxxl = true;
    e.one_sequence = true;
    e.snapshot_supported = true;
    e.native_active = true;
    e.identity_available = true;
    e.unmodified_model = true;
    e.text_only = true;
    e.slot_empty = true;
    e.all_sequences_empty = true;
    return e;
}

void test_eligibility() {
    std::string error;
    auto e = valid_save();
    assert(server_native_slot::save_allowed(e, &error));
    auto r = e;
    r.native_active = false; // restore does not require resident overflow
    assert(server_native_slot::restore_allowed(r, &error));

    auto reject = [&](auto mutate) {
        auto candidate = valid_save();
        mutate(candidate);
        error.clear();
        return !server_native_slot::save_allowed(candidate, &error) && !error.empty();
    };
    assert(reject([](auto & x) { x.native_stream = false; }));
    assert(reject([](auto & x) { x.adaptive = false; }));
    assert(reject([](auto & x) { x.profile_xxxl = false; }));
    assert(reject([](auto & x) { x.draft_present = true; }));
    assert(reject([](auto & x) { x.can_speculate = true; }));
    assert(reject([](auto & x) { x.one_sequence = false; }));
    assert(reject([](auto & x) { x.snapshot_supported = false; }));
    assert(reject([](auto & x) { x.native_active = false; }));
    assert(reject([](auto & x) { x.incompatible_route_flags = true; }));
    assert(reject([](auto & x) { x.media = true; }));
    assert(reject([](auto & x) { x.identity_available = false; }));
    assert(reject([](auto & x) { x.unmodified_model = false; }));
    assert(reject([](auto & x) { x.text_only = false; }));

    auto restore = valid_save();
    restore.native_active = false;
    error.clear();
    assert(server_native_slot::restore_allowed(restore, &error));
    restore.slot_empty = false;
    assert(!server_native_slot::restore_allowed(restore, &error) && !error.empty());
    restore = valid_save();
    restore.native_active = false;
    restore.all_sequences_empty = false;
    assert(!server_native_slot::restore_allowed(restore, &error));
    restore = valid_save();
    restore.native_active = true;
    assert(!server_native_slot::restore_allowed(restore, &error));
}

void test_restore_peak_16_layers() {
    rkva_native_state::Layout layout;
    layout.origin = 102400;
    layout.capacity = 26624;
    layout.heads = 4;
    layout.iterations = 3;
    layout.precision_tail_tokens = 128;
    layout.rows = layout.capacity;
    layout.record_base_group = 0;
    layout.saved_generation = 0x900000001ull;
    std::string error;
    assert(rkva_native_state::valid_layout(layout, &error));

    rkva_native_state::SetHeader header;
    header.layers = 16;
    header.layout = layout;
    header.payload_bytes = 544227328ull;
    header.layer_bytes = rkva_native_state::layer_bytes(layout);

    const uint64_t owner = rkva_native_state::owner_storage_bytes(layout);
    const uint64_t owner_total = owner * header.layers;
    const uint64_t staged = uint64_t(rkva_native_state::kHeaderBytes) + header.payload_bytes + 32;
    const uint64_t margin = (8ull << 20) + uint64_t(header.layers) * (1ull << 20);
    const uint64_t peak = server_native_slot::restore_peak_bytes(header);
    assert(peak == staged + owner_total + margin);
    assert(owner == 148702208ull);
    assert(peak > header.payload_bytes);
    // The estimator is bounded and fits the 4 GiB production cache budget.
    constexpr uint64_t cache_4gib = 4096ull << 20;
    assert(server_native_slot::restore_fits(peak, cache_4gib));
    assert(!server_native_slot::restore_fits(peak, peak - 1));
    assert(peak < cache_4gib);

    // The estimator accounts staged payload plus owner capacity; it does not
    // apply the adaptive five-copy file multiplier or assume the file is RAM.
    assert(peak < 5ull * (staged + owner_total));

    header.payload_bytes = UINT64_MAX;
    assert(server_native_slot::restore_peak_bytes(header) == UINT64_MAX);
    header.payload_bytes = 544227328ull;
    header.layers = rkva_native_state::kMaxLayers;
    assert(server_native_slot::restore_peak_bytes(header) != 0);
    header.layers = 0;
    assert(server_native_slot::restore_peak_bytes(header) == UINT64_MAX);

    // Cumulative admission uses the smaller of configured cache-ram and
    // measured MemAvailable after the server's 25% reserve. This is the real
    // 16-layer peak plus the bounded streaming workspace.
    constexpr uint64_t mib = 1024ull * 1024ull;
    const uint64_t save_workspace = server_native_slot::save_workspace_bytes(131072);
    assert(save_workspace == 16ull * mib + 131072ull * 16ull);
    assert(server_native_slot::save_workspace_bytes(UINT64_MAX) == UINT64_MAX);
    const uint64_t save_budget_low = server_slot_cap_ram_budget(16ull * mib, 4ull << 30, true);
    assert(save_budget_low == 12ull * mib);
    uint64_t save_total = 0;
    assert(!server_native_slot::cumulative_fits(save_workspace, 0, save_budget_low, &save_total));
    assert(save_total == save_workspace);
    const uint64_t save_budget_fit = server_slot_cap_ram_budget(1ull << 30, 4ull << 30, true);
    assert(server_native_slot::cumulative_fits(save_workspace, 0, save_budget_fit));
    const uint64_t workspace = 48ull * mib + 131072ull * 512ull;
    std::istringstream meminfo("MemAvailable: 1048576 kB\n");
    uint64_t measured_available = 0;
    assert(server_slot_parse_mem_available(meminfo, measured_available));
    assert(measured_available == (1ull << 30));
    const uint64_t budget_low = server_slot_cap_ram_budget(measured_available, 4ull << 30, true);
    assert(budget_low == 3ull * 256ull * mib); // 768 MiB
    uint64_t total = 0;
    assert(!server_native_slot::cumulative_fits(workspace, peak, budget_low, &total));
    assert(total == workspace + peak);
    const uint64_t budget_fit = server_slot_cap_ram_budget(8ull << 30, 4ull << 30, true);
    assert(budget_fit == (4ull << 30));
    assert(server_native_slot::cumulative_fits(workspace, peak, budget_fit));
    assert(!server_native_slot::cumulative_fits(UINT64_MAX - 1, 2, budget_fit));
    assert(!server_native_slot::cumulative_fits(workspace, peak, 0));
    assert(server_slot_cap_ram_budget(0, 4ull << 30, true) == 0);
}

} // namespace

int main() {
    test_eligibility();
    test_failed_restore_cleanup();
    test_restore_peak_16_layers();
    std::printf("server native streaming slot CPU regression: ok\n");
    return 0;
}
