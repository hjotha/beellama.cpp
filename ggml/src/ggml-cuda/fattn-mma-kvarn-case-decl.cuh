#pragma once

#include "fattn-mma-kvarn.cuh"

#include <cstdlib>

void ggml_cuda_fattn_kvarn_init_descs(
        const ggml_cuda_fattn_kvarn_plan & plan,
        ggml_cuda_fattn_kvarn_desc * k_desc,
        ggml_cuda_fattn_kvarn_desc * v_desc,
        int k_original_domain,
        int v_original_domain,
        cudaStream_t stream);

using ggml_cuda_fattn_kvarn_window_dequant_kernel_t = void (*)(
        const ggml_cuda_fattn_kvarn_desc *, const ggml_cuda_fattn_kvarn_desc *,
        half *, half *, int, int, int);
using ggml_cuda_fattn_kvarn_window_finalize_kernel_t = void (*)(
        float *, const float2 *, float2 *, float *, int);

// Position-split LSE export (plan §3.4). When OP_PARAM_LSE_OUT != 0, src[4]
// carries the LSE output tensor (F32, ne = { n_head_q, n_q, n_batch }) instead
// of sinks; sinks and LSE are mutually exclusive (attach API asserts this).
static inline bool ggml_cuda_fattn_lse_requested(const ggml_tensor * dst) {
    return ((const int32_t *) dst->op_params)[7] != 0;
}

static inline float * ggml_cuda_fattn_lse_ptr(ggml_tensor * dst) {
    // Null-safe: internal sub-passes (body/tail) may carry the flag with a
    // null src[4]; only the top-level op owns the LSE tensor.
    return (ggml_cuda_fattn_lse_requested(dst) && dst->src[4] != nullptr) ?
        (float *) dst->src[4]->data : nullptr;
}

static inline const ggml_tensor * ggml_cuda_fattn_sinks_or_null(const ggml_tensor * dst) {
    return ggml_cuda_fattn_lse_requested(dst) ? nullptr : dst->src[4];
}

// Windowed KVarN prefill gate (documented contract, plan §3.4 matrix). Only
// this generic-MMA sub-path exports LSE; the plain MMA launch below does not.
static inline bool ggml_cuda_fattn_kvarn_window_enabled() {
    const char * env = getenv("GGML_KVARN_WINDOW");
    return env == nullptr || atoi(env) != 0;
}

// Position-split LSE post-pass for single-split kernels that only publish
// (max, denom) metadata: lse = m + log(denom); empty -> -inf.
// Row order matches the LSE layout: ((stream * n_q + q) * n_heads + h).
static __global__ void ggml_cuda_fattn_kvarn_meta_to_lse_kernel(
        const float2 * meta,
        float * lse_out,
        size_t n_rows) {
    const size_t row = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    const float2 m = meta[row];
    lse_out[row] = m.y > 0.0f ? (m.x + logf(m.y)) : -INFINITY;
}

template <int D>
ggml_cuda_fattn_kvarn_window_dequant_kernel_t ggml_cuda_fattn_kvarn_window_dequant_get_kernel();

template<int D>
ggml_cuda_fattn_kvarn_window_finalize_kernel_t ggml_cuda_fattn_kvarn_window_finalize_get_kernel();

static inline enum ggml_flash_attn_ext_kvarn_domain ggml_cuda_fattn_kvarn_domain(const ggml_tensor * dst) {    return (enum ggml_flash_attn_ext_kvarn_domain) ggml_get_op_params_i32(
            dst, GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN);
}

static inline bool ggml_cuda_fattn_kvarn_k_original_domain(const ggml_tensor * dst) {
    switch (ggml_cuda_fattn_kvarn_domain(dst)) {
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ORIGINAL:
            return true;
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED:
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V:
            return false;
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO:
            return dst->src[0]->ne[1] > 1;
    }

    return dst->src[0]->ne[1] > 1;
}

static inline bool ggml_cuda_fattn_kvarn_v_original_domain(const ggml_tensor * dst) {
    switch (ggml_cuda_fattn_kvarn_domain(dst)) {
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ORIGINAL:
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V:
            return true;
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED:
            return false;
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO:
            return dst->src[0]->ne[1] > 1;
    }

    return dst->src[0]->ne[1] > 1;
}

static inline bool ggml_cuda_fattn_kvarn_rotated_decode_domain(const ggml_tensor * dst) {
    switch (ggml_cuda_fattn_kvarn_domain(dst)) {
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ORIGINAL:
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V:
            return false;
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED:
            return true;
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO:
            return dst->src[0]->ne[1] == 1;
    }

    return dst->src[0]->ne[1] == 1;
}

static inline const char * ggml_cuda_fattn_kvarn_domain_name(const ggml_tensor * dst) {
    switch (ggml_cuda_fattn_kvarn_domain(dst)) {
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ORIGINAL:
            return "original_prefill";
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED:
            return "rotated";
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V:
            return "rotated_k_original_v";
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO:
            return "auto";
    }

    return "unknown";
}

// True when the documented windowed gate applies (Q>1, no sinks, MIXED
// domain, supported dims). Used by the MMA launch pre-check to decline LSE
// ops early so the dispatcher reaches the portable route.
static inline bool ggml_cuda_fattn_kvarn_windowed_lse_applies(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const int DKQ = (int) Q->ne[0];
    return ggml_cuda_fattn_kvarn_window_enabled() &&
        Q->ne[1] > 1 &&
        ggml_cuda_fattn_sinks_or_null(dst) == nullptr &&
        ggml_cuda_fattn_kvarn_domain(dst) == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V &&
        (DKQ == 128 || DKQ == 256 || DKQ == 512);
}

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_kvarn_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define DECL_FATTN_MMA_KVARN_CASE_EXTERN(DKQ, DV, ncols1, ncols2)                  \
    extern template void ggml_cuda_flash_attn_ext_mma_kvarn_case                  \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(DKQ, DV, ncols)     \
    DECL_FATTN_MMA_KVARN_CASE_EXTERN(DKQ, DV, (ncols)/ 1, 1);    \
    DECL_FATTN_MMA_KVARN_CASE_EXTERN(DKQ, DV, (ncols)/ 2, 2);    \
    DECL_FATTN_MMA_KVARN_CASE_EXTERN(DKQ, DV, (ncols)/ 4, 4);    \
    DECL_FATTN_MMA_KVARN_CASE_EXTERN(DKQ, DV, (ncols)/ 8, 8);    \

DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(128, 128,  8)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(128, 128, 16)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(128, 128, 32)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(128, 128, 64)

DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(256, 256,  8)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(256, 256, 16)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(256, 256, 32)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(256, 256, 64)

DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(512, 512,  8)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(512, 512, 16)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(512, 512, 32)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(512, 512, 64)

#if !defined(GGML_USE_MUSA)
DECL_FATTN_MMA_KVARN_CASE_EXTERN(128, 128, 16, 8);
DECL_FATTN_MMA_KVARN_CASE_EXTERN(256, 256, 16, 8);

template <int DKQ, int DV, int ncols1, int ncols2>
bool ggml_cuda_fattn_kvarn_wide_mma_supported(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst);

extern template bool ggml_cuda_fattn_kvarn_wide_mma_supported<128, 128, 16, 8>(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst);
extern template bool ggml_cuda_fattn_kvarn_wide_mma_supported<256, 256, 16, 8>(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst);
#endif // !defined(GGML_USE_MUSA)

#undef DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2
#undef DECL_FATTN_MMA_KVARN_CASE_EXTERN
