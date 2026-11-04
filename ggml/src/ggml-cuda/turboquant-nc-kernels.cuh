#pragma once

#include "common.cuh"

bool ggml_cuda_tqnc_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                             const ggml_tensor * indices, ggml_tensor * dst, bool key);
bool ggml_cuda_saw_set_rows_checked(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                                    const ggml_tensor * indices, ggml_tensor * dst);
bool ggml_cuda_tqnc_validate_tail(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
void ggml_cuda_tqnc_clear_failure();
bool ggml_cuda_tqnc_has_failure();

#define GGML_TQNC_CONVERSIONS(name, type) \
    void ggml_cuda_tqnc_##name##_cont(const void *, type *, int64_t, cudaStream_t); \
    void ggml_cuda_tqnc_##name##_nc(const void *, type *, int64_t, int64_t, int64_t, int64_t, \
                                   int64_t, int64_t, int64_t, cudaStream_t)
GGML_TQNC_CONVERSIONS(key_f16, half);
GGML_TQNC_CONVERSIONS(value_f16, half);
GGML_TQNC_CONVERSIONS(key_f32, float);
GGML_TQNC_CONVERSIONS(value_f32, float);
#undef GGML_TQNC_CONVERSIONS
