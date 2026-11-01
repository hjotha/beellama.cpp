// Phase 3 activation gate test. The core must stay compiled-out by default and
// the runtime opt-in must stay false unless explicitly set to "1".

#include "../tools/server/server-rkva-managed-phase3.h"
#undef NDEBUG
#include <cassert>
#include <cstdlib>
#include <iostream>

int main() {
    assert(rkva_managed_phase3::compiled_in);
    unsetenv("RKVA_MANAGED_SNAPSHOT_PHASE3");
    assert(!rkva_managed_phase3::enabled());
    setenv("RKVA_MANAGED_SNAPSHOT_PHASE3", "", 1);
    assert(!rkva_managed_phase3::enabled());
    setenv("RKVA_MANAGED_SNAPSHOT_PHASE3", "0", 1);
    assert(!rkva_managed_phase3::enabled());
    setenv("RKVA_MANAGED_SNAPSHOT_PHASE3", "true", 1);
    assert(!rkva_managed_phase3::enabled());
    setenv("RKVA_MANAGED_SNAPSHOT_PHASE3", "1", 1);
    assert(rkva_managed_phase3::enabled());
    unsetenv("RKVA_MANAGED_SNAPSHOT_PHASE3");
    assert(!rkva_managed_phase3::enabled());
    std::cout << "managed snapshot Phase 3 gate is default-off: PASS\n";
    return 0;
}
