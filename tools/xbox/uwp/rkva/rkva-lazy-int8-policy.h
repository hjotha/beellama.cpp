#pragma once

namespace rkva_lazy_int8 {

// Each INT8 variant owns one independent control-path attempt. Default mode
// retains eager initialization, while lazy mode waits for explicit enablement.
struct Policy {
    bool requested = false;
    bool attempted = false;

    bool during_initialize(bool lazy) const {
        return !attempted && (!lazy || requested);
    }
    bool set_requested(bool enabled, bool initialized, bool lazy) {
        requested = enabled;
        return enabled && initialized && lazy && !attempted;
    }
    void mark_attempted() { attempted = true; }
};

} // namespace rkva_lazy_int8
