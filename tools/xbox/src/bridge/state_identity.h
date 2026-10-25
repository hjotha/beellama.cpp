// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once
#include "llama.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace xllama {
inline uint32_t state_float_bits(float value) {
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value), "32-bit float identity");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
// Exact float bits avoid decimal rounding collisions and locale dependence.
inline std::string state_rope_identity(const llama_context_rope_params& p) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "scaling=%d orig=%u base=%08x scale=%08x ext=%08x attn=%08x fast=%08x slow=%08x",
                  p.scaling_type, p.orig_ctx, state_float_bits(p.freq_base),
                  state_float_bits(p.freq_scale), state_float_bits(p.ext_factor),
                  state_float_bits(p.attn_factor), state_float_bits(p.beta_fast),
                  state_float_bits(p.beta_slow));
    return buf;
}
} // namespace xllama
