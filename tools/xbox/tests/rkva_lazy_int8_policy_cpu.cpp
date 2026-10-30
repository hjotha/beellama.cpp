// CPU-only tests for the opt-in control path; no D3D12 or Xbox required.
#include "../uwp/rkva/rkva-lazy-int8-policy.h"

#include <cassert>

struct SimulatedVariant {
    rkva_lazy_int8::Policy policy;
    bool pipeline = false;
    int creations = 0;

    void create(bool available) {
        ++creations;
        policy.mark_attempted();
        pipeline = available;
    }
    void initialize(bool lazy, bool available) {
        if (policy.during_initialize(lazy)) create(available);
    }
    void set(bool enabled, bool initialized, bool lazy, bool available) {
        if (policy.set_requested(enabled, initialized, lazy)) create(available);
    }
    bool selected() const { return policy.requested && pipeline; }
};

int main() {
    using rkva_lazy_int8::Policy;
    // Default remains eager even when disabled; later toggles do not retry.
    Policy eager;
    assert(eager.during_initialize(false));
    eager.mark_attempted();
    assert(!eager.set_requested(true, true, false));
    assert(!eager.set_requested(false, true, false));
    assert(!eager.set_requested(true, true, false));

    // Disabled lazy modes do no initialization work; enable after init
    // attempts exactly once, whether the factory succeeds or fails.
    Policy lazy;
    assert(!lazy.during_initialize(true));
    assert(!lazy.set_requested(false, true, true));
    assert(lazy.set_requested(true, true, true));
    lazy.mark_attempted();
    assert(!lazy.set_requested(true, true, true));
    assert(!lazy.set_requested(false, true, true));
    assert(!lazy.set_requested(true, true, true));

    // A setter called before init stores intent for exactly one init attempt.
    Policy before_init;
    assert(!before_init.set_requested(true, false, true));
    assert(before_init.during_initialize(true));
    before_init.mark_attempted();
    assert(!before_init.during_initialize(true));
    assert(!before_init.set_requested(true, true, true));

    // Each format has its own attempt and disabled status.
    Policy kvarn, q8;
    assert(kvarn.set_requested(true, true, true));
    kvarn.mark_attempted();
    assert(!q8.during_initialize(true));
    assert(q8.set_requested(true, true, true));
    q8.mark_attempted();
    assert(!kvarn.set_requested(true, true, true));
    assert(!q8.set_requested(true, true, true));

    // The F32 fallback remains selected until a successfully created PSO is
    // explicitly enabled. Disabling retains the resource for safe reuse.
    SimulatedVariant enabled;
    enabled.initialize(true, true);
    assert(enabled.creations == 0 && !enabled.selected());
    enabled.set(true, true, true, true);
    assert(enabled.creations == 1 && enabled.selected());
    enabled.set(false, true, true, true);
    assert(enabled.pipeline && !enabled.selected());
    enabled.set(true, true, true, true);
    assert(enabled.creations == 1 && enabled.selected());

    // Missing DXIL, unsupported wave/root or a failed creation all preserve
    // fallback and do not cause an implicit retry on later requests.
    SimulatedVariant unavailable;
    unavailable.initialize(true, false);
    unavailable.set(true, true, true, false);
    assert(unavailable.creations == 1 && !unavailable.selected());
    unavailable.set(false, true, true, true);
    unavailable.set(true, true, true, true);
    assert(unavailable.creations == 1 && !unavailable.selected());
}
