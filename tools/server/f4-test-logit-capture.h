#pragma once

// F4 TEST-ONLY full-logit capture hook.
//
// This hook exists for exactly one reason: the F4 occupancy/position-split
// numerical gate has to compare the PRODUCER's own post-decode logits against
// the TARGET's post-decode logits at the SAME checkpoint, so that a phase
// mismatch can never be mislabelled as a numerical result.
//
// It is compiled ONLY when LLAMA_F4_TEST_LOGIT_CAPTURE is defined. The CMake
// option is OFF by default and is turned on exclusively for build-f4, so this
// code is not present in build-optimized (production) binaries at all.
//
// It adds no configuration surface: every input arrives through the environment
// variables listed in the .cpp, and the call site is the server's EXISTING
// post-decode capture point (auto_capture_prompt_logits), whose slot state is
// the one the auto-save writer already serializes. No restore code is touched.

#if defined(LLAMA_F4_TEST_LOGIT_CAPTURE)

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Everything that must be IDENTICAL for two captures to be comparable, plus
// everything that must be DIFFERENT because the phases differ by construction.
struct f4_test_capture_binding {
    const std::string * run_id           = nullptr; // shared group id for the two phases
    const std::string * phase            = nullptr; // "producer" | "target"
    const std::string * build_id         = nullptr; // build-f4 library/binary hashes
    const std::string * model_fingerprint = nullptr; // adaptive model identity (weights+config)
    const std::string * config_core      = nullptr; // numeric context/model config that MUST match
    const std::string * layout           = nullptr; // KV layout; MUST DIFFER (pure vs split)
    const std::string * remote           = nullptr; // remote-attn placement (recorded, not hashed)
    const std::string * cache_source     = nullptr; // slot.prompt_cache_source
    const std::string * cache_reason     = nullptr; // slot.prompt_cache_reason
    uint64_t n_params     = 0;
    uint32_t n_ctx        = 0;
    uint32_t n_vocab      = 0;
    int64_t  pos_max      = -1;   // llama_memory_seq_pos_max of the slot sequence
    uint64_t state_bytes  = 0;    // llama_state_seq_get_size: restored KV payload size
    int32_t  logits_index = -1;
};

// tokens: the slot's full text-token stream at this exact checkpoint.
// logits: full-vocab logits of `logits_index` (post-decode checkpoint).
// Returns the number of capture files written for this checkpoint.
int f4_test_logit_capture_checkpoint(const f4_test_capture_binding & binding,
                                     const std::vector<int32_t> & tokens,
                                     const float * logits, size_t n_logits);

#endif // LLAMA_F4_TEST_LOGIT_CAPTURE