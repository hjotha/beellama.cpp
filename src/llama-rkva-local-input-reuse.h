#pragma once

#include <cstdint>
#include <cstring>

// GGML_RKVA_LOCAL_INPUT_REUSE policy, shared by the parent attention input
// setup and the wavefront tile views in llama-graph.cpp.
//
// The exact string "1" keeps the original all-batch behavior. The exact string
// "decode" is the stricter opt-in that only admits cache intent for a real
// single-query (N1) decode batch. Absent, empty, "0", and every unknown value
// keep the original OFF behavior.
namespace llama_rkva_local_input_reuse {

enum class mode {
    off,    // variable absent, empty, "0", or any unknown value
    all,    // exact "1": legacy experimental behavior for every batch
    decode, // exact "decode": only a real N1 parent batch
};

inline mode mode_from_value(const char * value) noexcept {
    if (value == nullptr) {
        return mode::off;
    }
    if (std::strcmp(value, "1") == 0) {
        return mode::all;
    }
    if (std::strcmp(value, "decode") == 0) {
        return mode::decode;
    }
    return mode::off;
}

// `parent_n_queries` is the query count of the real (parent) batch. Wavefront
// tile construction must pass the parent ubatch token count, never the tile or
// subview token count: a prefill parent can contain a one-token view, and
// `decode` must still leave that view uncached.
inline bool enabled(const char * value, bool remote_attn_position_split,
        std::int64_t parent_n_queries) noexcept {
    if (!remote_attn_position_split) {
        return false;
    }
    switch (mode_from_value(value)) {
        case mode::all:
            return true;
        case mode::decode:
            return parent_n_queries == 1;
        case mode::off:
        default:
            return false;
    }
}

} // namespace llama_rkva_local_input_reuse
