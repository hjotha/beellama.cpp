#pragma once
#include <cstdint>
#include <string>

namespace rkva_native_key_inspect {
#ifdef RKVA_NATIVE_KEY_INSPECT_TESTING
// Modifies a private copy-on-write mapping for offline corruption tests only.
extern void (*test_mutate)(uint8_t * bytes, uint64_t size);
#endif
// Strictly read-only forensic parsing of the explicitly identified Qwen35
// 64-layer / 48-recurrent-layer F32 profile. No context, GPU or RPC is used.
// A success proves the structural checks documented in the .cpp, not peer
// presence, model authenticity, reader pinning or safe deletion authority.
struct result {
    uint32_t tokens=0;
    uint64_t native_offset=0,native_end=0,state_bytes=0,whole_file_bytes=0;
    std::string native_key_sha256,whole_sha256,model,profile;
};
bool inspect(const std::string & path, result & out, std::string & error);
// Inspects a duplicate of an already-open read-only descriptor. The caller
// retains ownership and must hold its own immutability/lease authority across
// the whole operation and subsequent use of the result.
bool inspect_fd(int fd, result & out, std::string & error);
}
