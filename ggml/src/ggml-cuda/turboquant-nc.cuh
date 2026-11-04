// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// D=256, K=4 MSE with norm correction, V=4 uniform in the original domain.
// All 32 lanes must execute each helper with the same input and output row.
#pragma once

#include "common.cuh"
#include "../ggml-turboquant-nc.h"

static __device__ __constant__ float tqnc_cuda_centroids[16] = {
    -1.7068263888e-01f, -1.2927794456e-01f, -1.0111761093e-01f, -7.8516095877e-02f,
    -5.8903016150e-02f, -4.1054993868e-02f, -2.4258611724e-02f, -8.0267284065e-03f,
     8.0267284065e-03f,  2.4258611724e-02f,  4.1054993868e-02f,  5.8903016150e-02f,
     7.8516095877e-02f,  1.0111761093e-01f,  1.2927794456e-01f,  1.7068263888e-01f,
};

static __device__ __forceinline__ float tqnc_cuda_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        x += __shfl_down_sync(0xffffffffu, x, offset, 32);
    }
    return __shfl_sync(0xffffffffu, x, 0, 32);
}

static __device__ __forceinline__ void tqnc_cuda_wht(float (&v)[8]) {
#pragma unroll
    for (int stride = 1; stride < 8; stride *= 2) {
#pragma unroll
        for (int base = 0; base < 8; base += 2*stride) {
#pragma unroll
            for (int j = 0; j < stride; ++j) {
                const float a = v[base + j];
                const float b = v[base + j + stride];
                v[base + j]          = __fadd_rn(a, b);
                v[base + j + stride] = __fsub_rn(a, b);
            }
        }
    }
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int stride = 1; stride < 32; stride *= 2) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float other = __shfl_xor_sync(0xffffffffu, v[j], stride, 32);
            v[j] = (lane & stride) ? __fsub_rn(other, v[j]) : __fadd_rn(v[j], other);
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = __fmul_rn(v[j], 0.0625f);
}

static __device__ __forceinline__ int tqnc_cuda_nearest(float v) {
    int lo = 0, hi = 15;
    while (lo < hi) {
        const int mid = (lo + hi)/2;
        const float boundary = __fmul_rn(__fadd_rn(tqnc_cuda_centroids[mid], tqnc_cuda_centroids[mid + 1]), 0.5f);
        if (v >= boundary) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static __device__ __forceinline__ bool tqnc_cuda_encode_key(const float * src, ggml_tqnc_key * dst) {
    const int lane = threadIdx.x % 32;
    float v[8], norm_sq = 0.0f;
    bool valid = true;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = src[8*lane + j];
        valid = valid && isfinite(v[j]);
        norm_sq = __fadd_rn(norm_sq, __fmul_rn(v[j], v[j]));
    }
    if (!__all_sync(0xffffffffu, valid)) return false;
    const float norm = __fsqrt_rn(tqnc_cuda_sum(norm_sq));
    const half stored_norm = __float2half_rn(norm);
    if (!isfinite(norm) || !isfinite(__half2float(stored_norm))) return false;
    const float denom = __fadd_rn(norm, 1e-8f);
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = __fdiv_rn(v[j], denom);
    tqnc_cuda_wht(v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        dst->qs[4*lane + j] = uint8_t(tqnc_cuda_nearest(v[2*j]) | (tqnc_cuda_nearest(v[2*j + 1]) << 4));
    }
    if (lane == 0) dst->norm = __half_as_ushort(stored_norm);
    return true;
}

static __device__ __forceinline__ bool tqnc_cuda_encode_value(const float * src, ggml_tqnc_value * dst) {
    const int lane = threadIdx.x % 32;
    float v[8], lo = INFINITY, hi = -INFINITY;
    bool valid = true;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = src[8*lane + j];
        valid = valid && isfinite(v[j]);
        lo = fminf(lo, v[j]);
        hi = fmaxf(hi, v[j]);
    }
    if (!__all_sync(0xffffffffu, valid)) return false;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        lo = fminf(lo, __shfl_xor_sync(0xffffffffu, lo, offset, 32));
        hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, offset, 32));
    }
    const float scale = fmaxf(__fdiv_rn(__fsub_rn(hi, lo), 15.0f), 1e-8f);
    const half stored_scale = __float2half_rn(scale);
    const half stored_min = __float2half_rn(lo);
    if (!isfinite(scale) || !isfinite(__half2float(stored_scale)) || !isfinite(__half2float(stored_min))) return false;
    const float decoded_max = __fadd_rn(__fmul_rn(15.0f, __half2float(stored_scale)), __half2float(stored_min));
    if (!isfinite(__half2float(__float2half_rn(decoded_max)))) return false;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float a = __fadd_rn(__fdiv_rn(__fsub_rn(v[2*j], lo), scale), 0.5f);
        const float b = __fadd_rn(__fdiv_rn(__fsub_rn(v[2*j + 1], lo), scale), 0.5f);
        const int q0 = min(15, max(0, int(floorf(a))));
        const int q1 = min(15, max(0, int(floorf(b))));
        dst->qs[4*lane + j] = uint8_t(q0 | (q1 << 4));
    }
    if (lane == 0) {
        dst->scale = __half_as_ushort(stored_scale);
        dst->minimum = __half_as_ushort(stored_min);
    }
    return true;
}

template<typename dst_t>
static __device__ __forceinline__ bool tqnc_cuda_decode_key(const ggml_tqnc_key * src, dst_t * dst) {
    const int lane = threadIdx.x % 32;
    float v[8], norm_sq = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int element = 8*lane + j;
        const int index = (src->qs[element/2] >> (4*(element % 2))) & 15;
        v[j] = tqnc_cuda_centroids[index];
        norm_sq = __fadd_rn(norm_sq, __fmul_rn(v[j], v[j]));
    }
    const float norm = __half2float(__ushort_as_half(src->norm));
    if (!isfinite(norm) || norm < 0.0f) return false;
    const float scale = __fdiv_rn(norm, __fsqrt_rn(__fadd_rn(tqnc_cuda_sum(norm_sq), 1e-16f)));
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = __fmul_rn(v[j], scale);
    tqnc_cuda_wht(v);
    bool valid = true;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        valid = valid && isfinite(v[j]) && isfinite(__half2float(__float2half_rn(v[j])));
    }
    if (!__all_sync(0xffffffffu, valid)) return false;
#pragma unroll
    for (int j = 0; j < 8; ++j) dst[8*lane + j] = dst_t(v[j]);
    return true;
}

template<typename dst_t>
static __device__ __forceinline__ bool tqnc_cuda_decode_value(const ggml_tqnc_value * src, dst_t * dst) {
    const int lane = threadIdx.x % 32;
    const float scale = __half2float(__ushort_as_half(src->scale));
    const float min_value = __half2float(__ushort_as_half(src->minimum));
    if (!isfinite(scale) || scale < 0.0f || !isfinite(min_value)) return false;
    const float decoded_max = __fadd_rn(__fmul_rn(15.0f, scale), min_value);
    if (!isfinite(__half2float(__float2half_rn(decoded_max)))) return false;
    float decoded[8];
    bool valid = true;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int element = 8*lane + j;
        const int index = (src->qs[element/2] >> (4*(element % 2))) & 15;
        decoded[j] = __fadd_rn(__fmul_rn(float(index), scale), min_value);
        valid = valid && isfinite(decoded[j]) && isfinite(__half2float(__float2half_rn(decoded[j])));
    }
    if (!__all_sync(0xffffffffu, valid)) return false;
#pragma unroll
    for (int j = 0; j < 8; ++j) dst[8*lane + j] = dst_t(decoded[j]);
    return true;
}
