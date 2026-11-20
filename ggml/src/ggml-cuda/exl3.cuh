#pragma once

#include "common.cuh"

bool ggml_cuda_exl3_supports_op(const ggml_tensor * op);
bool ggml_cuda_exl3_mul_mat_id_needs_sync(const ggml_tensor * op);

void ggml_cuda_op_exl3_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_exl3_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_exl3_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
