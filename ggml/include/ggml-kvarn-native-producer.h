#pragma once

#include "ggml.h"

// CUDA registry proc "ggml_backend_cuda_kvarn_native_append_v1".
// Original F32 inputs only; 256-wide heads, at most 256 contiguous tokens.
// Host stage arrays contain the canonical 384 rotated-F16 rows and are read
// only; the caller publishes append_* rows after success. append_* receive
// rotated rows, and records_* receive only
// the complete groups closed by this append, in chronological order. Scratch
// is bounded independently of the total KV context. No Q4 decoder is involved.
typedef bool (*ggml_backend_cuda_kvarn_native_append_v1_t)(
    const float * keys, const float * values, uint32_t global_begin,
    uint32_t count, uint32_t heads, uint32_t iterations,
    const ggml_fp16_t * stage_k, const ggml_fp16_t * stage_v,
    ggml_fp16_t * append_k, ggml_fp16_t * append_v,
    void * records_k, void * records_v);
