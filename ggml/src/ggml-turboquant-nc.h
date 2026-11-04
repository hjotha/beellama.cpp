// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// D=256 preset compatible with the 4-bit MSE-key / uniform-value semantics.
enum { GGML_TQNC_DIM = 256, GGML_TQNC_PACKED = GGML_TQNC_DIM/2 };

typedef struct {
    uint8_t qs[GGML_TQNC_PACKED];
    ggml_fp16_t norm;
} ggml_tqnc_key;

typedef struct {
    uint8_t qs[GGML_TQNC_PACKED];
    ggml_fp16_t scale;
    ggml_fp16_t minimum;
} ggml_tqnc_value;

// Return false for invalid inputs/metadata and leave the destination untouched.
GGML_API bool ggml_tqnc_quantize_key(const float * src, ggml_tqnc_key * dst);
GGML_API bool ggml_tqnc_quantize_value(const float * src, ggml_tqnc_value * dst);
GGML_API bool ggml_tqnc_dequantize_key(const ggml_tqnc_key * src, float * dst);
GGML_API bool ggml_tqnc_dequantize_value(const ggml_tqnc_value * src, float * dst);

GGML_API void ggml_tqnc_quantize_key_rows(const float * src, void * dst, int64_t n);
GGML_API void ggml_tqnc_quantize_value_rows(const float * src, void * dst, int64_t n);
GGML_API void ggml_tqnc_dequantize_key_rows(const void * src, float * dst, int64_t n);
GGML_API void ggml_tqnc_dequantize_value_rows(const void * src, float * dst, int64_t n);

#ifdef __cplusplus
}
#endif
