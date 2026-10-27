// CPU/offline contract for the explicit adaptive native streaming slot API and
// its bounded restore RAM estimator. No model, GPU or remote peer.
#include "../tools/server/server-native-slot.h"
#include "../ggml/include/rkva-native-state.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <limits>
#include <string>

namespace {

server_native_slot::Eligibility valid_save() {
    server_native_slot::Eligibility e;
    e.native_stream = true;
    e.adaptive = true;
    e.profile_xxxl = true;
    e.one_sequence = true;
    e.snapshot_supported = true;
    e.native_active = true;
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
}

} // namespace

int main() {
    test_eligibility();
    test_restore_peak_16_layers();
    std::printf("server native streaming slot CPU regression: ok\n");
    return 0;
}
