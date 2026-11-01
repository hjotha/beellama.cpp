#pragma once
#include "../../ggml/include/rkva-native-snapshot.h"
#include <cstdint>
#include <string>

// Read-only local-object validation only. The caller must hold external store
// and reader authority against all writers across this operation and any later
// use. Success is not remote proof, authentication, a durable pin or GC authority.
namespace rkva_native_object_inspect {
struct expected {
    rkva_snapshot::Digest key{};
    rkva_native_snapshot::Geometry geometry{};
    rkva_native_snapshot::Descriptor descriptor{};
    uint32_t live_rows = 0;
    uint64_t saved_generation = 0;
};
struct result {
    std::string key;
    uint32_t chunks = 0;
    uint64_t payload_bytes = 0;
    uint64_t manifest_bytes = 0;
    std::string manifest_sha256;
};
#ifdef RKVA_NATIVE_OBJECT_INSPECT_TESTING
extern void (*test_hook)(const char * phase, uint32_t index);
#endif
// Requires a directory named exactly by the expected lowercase key. Rejects
// every unexpected entry. `out` is not modified on failure.
bool inspect(const std::string & directory, const expected & want, result & out, std::string & error);
}
