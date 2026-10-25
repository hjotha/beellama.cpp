// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Decode-trace knob for boundary diagnosis (plan 004): when set to an output
// index N (>= 0), the decode loop logs the exact decision behind output token
// N — which path emitted it, the batch row it was sampled from, the top
// candidates and margin there, and the draft/accept state. Zero overhead and
// zero behavior change when off (-1). Used to attribute an MTP-vs-sequential
// divergence to batching numerics vs accept/rollback/state, never left on in
// a gate (a traced run is diagnostic evidence, not gate evidence).
#pragma once

namespace xllama {
namespace decode_trace {

// Output token index to trace, or -1 (off). Thread-local like the d3d12 scope
// tags: set on the bench thread before generate, read inside the loop.
inline thread_local int g_output_idx = -1;

inline void set_output_idx(int idx) {
    g_output_idx = idx;
}

inline int output_idx() {
    return g_output_idx;
}

} // namespace decode_trace
} // namespace xllama
