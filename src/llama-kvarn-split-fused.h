#pragma once

// Executor for GGML_OP_KVARN_SPLIT_FUSED ("peça (c)" phase 2): the fused
// position-split decode span. ONE scheduler node replaces the overflow half of
// build_attn_position_split()'s two-range decode chain:
//
//   overflow Q4 store (SET_ROWS) + F16 gather + FA2 + LSE2  (Vulkan sub-graph)
//   + LSE merge with the legacy local FA/LSE results        (CUDA sub-graph)
//
// The CUDA dispatcher (ggml-cuda) resolves this executor through the
// "ggml_backend_cuda_set_kvarn_split_fused_exec" backend procedure; the
// executor needs the Vulkan backend and the companion storage views, so its
// body lives on the llama side (src/llama-kvarn-split-fused.cpp).
//
// Gating (all must hold; otherwise the legacy chain is kept intact):
//   * LLAMA_OVF_FUSED_SPAN == "1" (default off -> byte-identical legacy path),
//   * the CUDA supports_op shape contract,
//   * the grouped single-query decode arm with the packed overflow view.
//
// Fail closed: a missing executor, an unsupported operand set or any error in
// the Vulkan/CUDA submission aborts the process with a log message instead of
// producing a stale span output.

#include "ggml.h"
#include "ggml-backend.h"

// Registers ONE executor bundle for the calling context pair (cuda, vulkan)
// and returns its id (> 0), which the graph builder bakes into every fused op.
// Each call creates a new bundle: two contexts never share or clobber each
// other's sub-graphs, and the id routes the CUDA-side dispatch to the right
// bundle. Returns 0 when the CUDA registry does not expose the registration
// procedure or the id space is exhausted.
uint32_t llama_kvarn_split_fused_exec_register(ggml_backend_t cuda, ggml_backend_t vulkan);

// Releases ONLY the bundle `id` (its per-layer sub-graphs and buffers). Other
// contexts' bundles are untouched; an in-flight callback keeps its bundle
// alive through a shared reference. When the last bundle goes away the CUDA
// callback slot is cleared.
void llama_kvarn_split_fused_exec_unregister(uint32_t id);

// Whether bundle `id` is currently registered and executable.
bool llama_kvarn_split_fused_exec_ready(uint32_t id);
