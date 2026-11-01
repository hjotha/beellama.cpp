#include "../tools/server/server-rkva-managed-phase4.h"
#undef NDEBUG
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>

// The gate is compiled in for this test target, so compiled_in must be true and
// every non-"1" value must stay off. The default-off build is checked by the
// phase4-off build in the report, not here.
int main() {
    using namespace rkva_managed_phase4;
    static_assert(compiled_in, "phase4 gate test must be built with the macro");
    assert(!enabled());

    const char * values[] = { "", "0", "true", "01", " 1", "1 ", "yes", "2" };
    for (const char * value : values) {
        assert(setenv("RKVA_MANAGED_SNAPSHOT_PHASE4", value, 1) == 0);
        assert(!enabled());
    }
    assert(setenv("RKVA_MANAGED_SNAPSHOT_PHASE4", "1", 1) == 0);
    assert(enabled());
    assert(unsetenv("RKVA_MANAGED_SNAPSHOT_PHASE4") == 0);
    assert(!enabled());

    // The combined predicate never opens without both the gate and negotiation.
    assert(!accepts_negotiated(true));
    assert(setenv("RKVA_MANAGED_SNAPSHOT_PHASE4", "1", 1) == 0);
    assert(!accepts_negotiated(false));
    assert(accepts_negotiated(true));
    assert(unsetenv("RKVA_MANAGED_SNAPSHOT_PHASE4") == 0);

    std::cout << "phase 4 gate: compiled-in, off by default, opt-in only on 1\n";
    return 0;
}
