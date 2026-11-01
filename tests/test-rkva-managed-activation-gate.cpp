#include "../tools/server/server-rkva-managed-activation.h"
#include "../tools/server/server-rkva-managed-phase4.h"
#include "../ggml/src/ggml-remote-attn/rkva-managed-registry-wire.h"
#undef NDEBUG
#include <cassert>
#include <cstdlib>
#include <iostream>

// The activation gate is compiled in for this test target, so compiled_in must
// be true and every non-"1" value must stay off. The default-off build is
// checked by the activation-off build in the report, not here.
int main() {
    using namespace rkva_managed_activation;
    static_assert(compiled_in, "activation gate test must be built with the macro");
    assert(!enabled());

    const char * values[] = { "", "0", "true", "01", " 1", "1 ", "yes", "2", "ON", "on" };
    for (const char * value : values) {
        assert(setenv("RKVA_MANAGED_SNAPSHOT_ACTIVATION", value, 1) == 0);
        assert(!enabled());
    }
    assert(setenv("RKVA_MANAGED_SNAPSHOT_ACTIVATION", "1", 1) == 0);
    assert(enabled());
    assert(unsetenv("RKVA_MANAGED_SNAPSHOT_ACTIVATION") == 0);
    assert(!enabled());

    // The Phase 4 gate is independent and also off by default.
    assert(!rkva_managed_phase4::enabled());

    // No advertisement: an unprovisioned HELLO acknowledgement has no managed
    // capability, protocol or operation bit, and cannot negotiate.
    rkva_hello_ack ack{};
    assert(ack.supported_ops == 0);
    assert(ack.reserved[0] == 0 && ack.reserved[1] == 0);
    assert(!rkva_registry_wire::negotiated(0, 0, 0));
    assert(!rkva_registry_wire::negotiated(RKVA_PROTO_MANAGED_REGISTRY_V1,
                                           RKVA_CAP_MANAGED_REGISTRY_V1,
                                           1u << RKVA_REGISTRY_PREPARE));

    std::cout << "activation gate: compiled-in, off by default, opt-in only on 1, no advertisement\n";
    return 0;
}
