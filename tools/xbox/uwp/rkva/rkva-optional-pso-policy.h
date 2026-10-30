#pragma once

namespace rkva_optional_pso {
// Serialized control-path policy. A blocked attempt is deferred, not consumed.
// A completed attempt (including failure) is never retried in this engine.
struct Policy {
    bool requested = false;
    bool attempted = false;
    bool begin_attempt(bool safe) {
        if (!requested || attempted || !safe) return false;
        attempted = true;
        return true;
    }
};
} // namespace rkva_optional_pso
