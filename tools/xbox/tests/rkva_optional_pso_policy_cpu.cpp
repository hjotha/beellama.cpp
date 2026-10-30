#include "../uwp/rkva/rkva-optional-pso-policy.h"
#include <cassert>
#include <iostream>

int main() {
    using rkva_optional_pso::Policy;
    Policy disabled;
    assert(!disabled.begin_attempt(true));
    assert(!disabled.attempted);
    disabled.requested = true;
    assert(!disabled.begin_attempt(false)); // busy/uncertain GPU must defer
    assert(!disabled.attempted);
    assert(disabled.begin_attempt(true));
    assert(!disabled.begin_attempt(true)); // success OR failure consumes attempt
    disabled.requested = false;
    assert(!disabled.begin_attempt(true));
    disabled.requested = true;
    assert(!disabled.begin_attempt(true)); // do not recreate after toggle
    Policy default_on{true, false};
    assert(default_on.begin_attempt(true));
    Policy new_engine{true, false};
    assert(new_engine.begin_attempt(true)); // ownership is per-engine
    Policy independent{false, false};
    assert(!independent.attempted); // one variant cannot consume another's attempt
    std::cout << "optional PSO policy: disabled, deferred, once-only, toggle, new owner PASS\n";
}
