#pragma once

#include <cstdint>
#include <cstring>

// Use the parent batch size: a prefill tile can contain a single query.
inline bool llama_rkva_local_input_reuse(const char * value, bool remote_split, int64_t parent_queries) {
    return remote_split && value && (std::strcmp(value, "1") == 0 ||
            (std::strcmp(value, "prefill") == 0 && parent_queries > 1));
}
