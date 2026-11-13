#pragma once

#include "ggml.h"

// Internal operation metadata; unmarked tensors retain the backend's normal path.
enum ggml_prefill_reference_plan {
    GGML_PREFILL_REFERENCE_NONE = 0,
    GGML_PREFILL_REFERENCE_DEFAULT = 1,
    GGML_PREFILL_REFERENCE_FIRST = 2,
    GGML_PREFILL_REFERENCE_SECOND = 3,
    GGML_PREFILL_REFERENCE_SHIFTED = 4,
    GGML_PREFILL_REFERENCE_SHIFTED_SECOND = 5,
};

static constexpr int32_t GGML_PREFILL_REFERENCE_MARKER = 0x4b565250;

static inline ggml_prefill_reference_plan ggml_prefill_reference_get(const ggml_tensor * tensor) {
    const bool attention = tensor->op == GGML_OP_FLASH_ATTN_EXT;
    if (tensor->op != GGML_OP_MUL_MAT && tensor->op != GGML_OP_GATED_DELTA_NET && !attention) {
        return GGML_PREFILL_REFERENCE_NONE;
    }
    const int marker = attention ? 15 : 5;
    const int plan = attention ? 14 : 4;
    return tensor->op_params[marker] == GGML_PREFILL_REFERENCE_MARKER ?
            static_cast<ggml_prefill_reference_plan>(tensor->op_params[plan]) : GGML_PREFILL_REFERENCE_NONE;
}

static inline void ggml_prefill_reference_set(ggml_tensor * tensor, ggml_prefill_reference_plan plan) {
    const bool attention = tensor->op == GGML_OP_FLASH_ATTN_EXT;
    tensor->op_params[attention ? 15 : 5] = GGML_PREFILL_REFERENCE_MARKER;
    tensor->op_params[attention ? 14 : 4] = plan;
}
