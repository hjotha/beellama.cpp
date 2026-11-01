// Phase 2 activation gate test. The core must stay compiled-out by default and
// the runtime opt-in must stay false unless explicitly set to "1".

#include "../tools/server/server-rkva-managed-phase2.h"
#undef NDEBUG
#include <cassert>
#include <cstdlib>
#include <iostream>

int main() {
    assert(rkva_managed_phase2::compiled_in);
    unsetenv("RKVA_MANAGED_SNAPSHOT_PHASE2");
    assert(!rkva_managed_phase2::enabled());
    setenv("RKVA_MANAGED_SNAPSHOT_PHASE2", "", 1);
    assert(!rkva_managed_phase2::enabled());
    setenv("RKVA_MANAGED_SNAPSHOT_PHASE2", "0", 1);
    assert(!rkva_managed_phase2::enabled());
    setenv("RKVA_MANAGED_SNAPSHOT_PHASE2", "true", 1);
    assert(!rkva_managed_phase2::enabled());
    setenv("RKVA_MANAGED_SNAPSHOT_PHASE2", "1", 1);
    assert(rkva_managed_phase2::enabled());
    unsetenv("RKVA_MANAGED_SNAPSHOT_PHASE2");
    assert(!rkva_managed_phase2::enabled());
    std::cout << "managed snapshot Phase 2 gate is default-off: PASS\n";
    return 0;
}
