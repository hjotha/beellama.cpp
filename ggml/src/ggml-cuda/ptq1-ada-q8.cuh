#pragma once

#include "common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// Warp-transposed Q8_1 and exact sums from PrismML-Eng/llama.cpp f70247610.
// Each 128-weight block uses 32 quant words and 4 scale/sum words.
// Grouping 32 such blocks transposes the layout for coalesced warp loads.
#define GGML_CUDA_PTQ1_Q8_GROUP_KB     32
#define GGML_CUDA_PTQ1_Q8_WORDS_PER_KB 36
#define GGML_CUDA_PTQ1_Q8_GROUP_WORDS  (GGML_CUDA_PTQ1_Q8_GROUP_KB * GGML_CUDA_PTQ1_Q8_WORDS_PER_KB)
#define GGML_CUDA_PTQ1_K_PAD           (GGML_CUDA_PTQ1_Q8_GROUP_KB * QK_PTQ1_0)

// Word offset (within one activation column) of word w (0..7 = qs words, 8 = ds) of block_q8_1 ib.
static constexpr __host__ __device__ int ggml_cuda_ptq1_q8_word(int ib, int w) {
    const int kb   = ib >> 2;
    const int sub  = ib & 3;
    const int g    = kb >> 5;
    const int lane = kb & 31;
    const int ww   = w < 8 ? sub * 8 + w : 32 + sub;
    return g * GGML_CUDA_PTQ1_Q8_GROUP_WORDS + ww * GGML_CUDA_PTQ1_Q8_GROUP_KB + lane;
}

static __device__ __forceinline__ void ggml_cuda_store_ptq1_ada_q8(float value, int32_t * y, int col) {
    const float maximum = warp_reduce_max<QK8_1>(fabsf(value));
    const float scale = maximum / 127.0f;
    const int8_t quant = maximum == 0.0f ? 0 : roundf(value / scale);
    const int sum = warp_reduce_sum<QK8_1>(int(quant));

    const int block = col / QK8_1;
    const int element = col % QK8_1;
    reinterpret_cast<int8_t *>(y + ggml_cuda_ptq1_q8_word(block, element / 4))[element % 4] = quant;
    if (element == 0) {
        const half2 scale_sum = make_half2(__float2half(scale), __short_as_half(short(sum)));
        y[ggml_cuda_ptq1_q8_word(block, 8)] = *reinterpret_cast<const int32_t *>(&scale_sum);
    }
}

static inline bool ggml_cuda_ptq1_ada_enabled(const ggml_backend_cuda_context & ctx) {
    static const char * value = std::getenv("GGML_CUDA_PTQ1_ADA");
    return (!value || std::atoi(value) != 0) && !ggml_cuda_batch_invariant() &&
        ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_ADA_LOVELACE;
}
#endif
