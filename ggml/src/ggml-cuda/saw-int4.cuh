#pragma once

#include "common.cuh"

template<int N>
static __device__ __forceinline__ void saw_fwht(float (&v)[N]) {
#pragma unroll
    for (int stride = 1; stride < N; stride *= 2) {
#pragma unroll
        for (int base = 0; base < N; base += 2*stride) {
#pragma unroll
            for (int j = 0; j < stride; ++j) {
                const float a = v[base + j];
                const float b = v[base + j + stride];
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
                v[base + j]          = __fadd_rn(a, b);
                v[base + j + stride] = __fsub_rn(a, b);
#else
                v[base + j]          = a + b;
                v[base + j + stride] = a - b;
#endif
            }
        }
    }
}
