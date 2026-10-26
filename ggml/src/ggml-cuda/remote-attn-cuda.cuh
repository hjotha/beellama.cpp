#pragma once

#include "ggml-remote-attn-cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

const struct ggml_remote_attn_cuda_fp8_interface * ggml_backend_cuda_remote_attn_fp8_interface(void);
const struct ggml_remote_attn_cuda_fp8_interface_v2 * ggml_backend_cuda_remote_attn_fp8_interface_v2(void);
const struct ggml_remote_attn_cuda_fp8_interface_v3 * ggml_backend_cuda_remote_attn_fp8_interface_v3(void);
const struct ggml_remote_attn_cuda_retirement_interface_v1 * ggml_backend_cuda_remote_attn_retirement_interface_v1(void);

bool ggml_cuda_remote_attn_fp8_pack_q(const void * src_q, bool is_f16, size_t n_elements,
                                      uint8_t * dst_pinned_fp8, void * stream, void * event_done);

bool ggml_cuda_remote_attn_fp8_unpack_out(const uint8_t * src_pinned_fp8, const float * src_lse,
                                         size_t nq, size_t out_width, size_t n_head,
                                         void * dst_out, size_t dst_stride_bytes,
                                         void * stream, void * event_done);

#ifdef __cplusplus
}
#endif
