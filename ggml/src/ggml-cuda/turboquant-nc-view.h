#pragma once

#include "../ggml-turboquant-nc.h"

#include <cstddef>
#include <cstdint>
#include <limits>

// Validate the actual strided span, not the contiguous span checked when a view is created.
static bool ggml_cuda_tqnc_span(const ggml_tensor * t, size_t & span) {
    const size_t bytes = ggml_type_size(t->type);
    const int64_t block = ggml_blck_size(t->type);
    if (!bytes || block <= 0 || t->ne[0] <= 0 || t->ne[0] % block) return false;
    int64_t elements = 1;
    span = bytes;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] <= 0 || t->ne[i] > INT64_MAX/elements || t->nb[i] > INT64_MAX) return false;
        elements *= t->ne[i];
        const uint64_t n = uint64_t(i == 0 ? t->ne[0]/block - 1 : t->ne[i] - 1);
        if (n && t->nb[i] > (SIZE_MAX - span)/n) return false;
        span += size_t(n)*t->nb[i];
    }
    return span <= INT64_MAX;
}

static bool ggml_cuda_tqnc_tensor_valid(const ggml_tensor * t) {
    if (!t) return false;
    size_t unit = 0, alignment = 0;
    switch (t->type) {
        case GGML_TYPE_SAW_INT4: unit = 136; alignment = alignof(float); break;
        case GGML_TYPE_TQNC_K: unit = sizeof(ggml_tqnc_key);   alignment = alignof(ggml_tqnc_key); break;
        case GGML_TYPE_TQNC_V: unit = sizeof(ggml_tqnc_value); alignment = alignof(ggml_tqnc_value); break;
        case GGML_TYPE_F32:    unit = sizeof(float);           alignment = alignof(float); break;
        case GGML_TYPE_F16:    unit = sizeof(ggml_fp16_t);     alignment = alignof(ggml_fp16_t); break;
        case GGML_TYPE_I32:    unit = sizeof(int32_t);         alignment = alignof(int32_t); break;
        case GGML_TYPE_I64:    unit = sizeof(int64_t);         alignment = alignof(int64_t); break;
        default: return false;
    }
    size_t span;
    if (!ggml_cuda_tqnc_span(t, span) ||
            (t->data && (uintptr_t(t->data) % alignment || span - 1 > UINTPTR_MAX - uintptr_t(t->data)))) return false;
    const ggml_tensor * base = t;
    size_t offset = 0;
    int depth = 0;
    while (base->view_src) {
        if (++depth > 128 || base->view_offs > SIZE_MAX - offset) return false;
        offset += base->view_offs;
        base = base->view_src;
    }
    size_t base_span;
    if (base->type != t->type || offset % unit || !ggml_cuda_tqnc_span(base, base_span) ||
            offset > base_span || span > base_span - offset) return false;
    if (base->data && t->data) {
        const uintptr_t addr = uintptr_t(base->data);
        if (offset > UINTPTR_MAX - addr || uintptr_t(t->data) != addr + offset) return false;
    }
    return true;
}
