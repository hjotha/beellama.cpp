// Default-build guard test: without XLLAMA_QUAL_PROBE the probe action must be
// absent from the autopilot contract and no probe implementation may exist.
//
// Compiled with the macro OFF on purpose. This is the reachability proof for
// "not present in a production build", separate from the console run.

#include "rkva/rkva-qual-probe.h"
#include "xllama/autopilot.h"

#undef NDEBUG
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

int main() {
    static_assert(!rkva_qual_probe::compiled,
                  "this target must be built without XLLAMA_QUAL_PROBE");

    assert(!xllama::autopilot_op_known("rkva_qual_probe"));

    std::vector<xllama::AutopilotAction> actions(1);
    actions[0].op = "rkva_qual_probe";
    actions[0].label = L"0123456789abcdef0123456789abcdef";
    actions[0].arg = L"positive";
    std::string error;
    assert(!xllama::validate_autopilot_script(actions, false, error));
    assert(error.find("unknown op 'rkva_qual_probe'") != std::string::npos);

    std::cout << "rkva-qual-probe-guard-off: action unknown and validator refuses it\n";
    return 0;
}
