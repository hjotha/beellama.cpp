// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// D=256 reference for the 4bit_nc preset; format and centroids are from
// TheTom/vllm-turboquant commit 36fc048255d0bbdab05811d667182a965fe05936.
#include "ggml-turboquant-nc.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

_Static_assert(sizeof(ggml_tqnc_key) == 130, "unexpected TurboQuant NC key layout");
_Static_assert(sizeof(ggml_tqnc_value) == 132, "unexpected TurboQuant NC value layout");

static const float tqnc_centroids[16] = {
    -1.7068263888e-01f, -1.2927794456e-01f, -1.0111761093e-01f, -7.8516095877e-02f,
    -5.8903016150e-02f, -4.1054993868e-02f, -2.4258611724e-02f, -8.0267284065e-03f,
     8.0267284065e-03f,  2.4258611724e-02f,  4.1054993868e-02f,  5.8903016150e-02f,
     7.8516095877e-02f,  1.0111761093e-01f,  1.2927794456e-01f,  1.7068263888e-01f,
};

static void tqnc_wht(float * v) {
    for (int stride = 1; stride < GGML_TQNC_DIM; stride *= 2) {
        for (int base = 0; base < GGML_TQNC_DIM; base += 2*stride) {
            for (int j = 0; j < stride; ++j) {
                const float a = v[base + j];
                const float b = v[base + j + stride];
                v[base + j]          = a + b;
                v[base + j + stride] = a - b;
            }
        }
    }
    for (int i = 0; i < GGML_TQNC_DIM; ++i) v[i] *= 0.0625f;
}

static int tqnc_index(float v) {
    int lo = 0, hi = 15;
    while (lo < hi) {
        const int mid = (lo + hi)/2;
        const float boundary = (tqnc_centroids[mid] + tqnc_centroids[mid + 1])*0.5f;
        if (v >= boundary) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

bool ggml_tqnc_quantize_key(const float * src, ggml_tqnc_key * dst) {
    if (!src || !dst) return false;
    float norm_sq = 0.0f;
    for (int i = 0; i < GGML_TQNC_DIM; ++i) {
        if (!isfinite(src[i])) return false;
        norm_sq += src[i]*src[i];
    }
    const float norm = sqrtf(norm_sq);
    if (!isfinite(norm)) return false;
    ggml_tqnc_key block;
    block.norm = ggml_fp32_to_fp16(norm);
    if (!isfinite(ggml_fp16_to_fp32(block.norm))) return false;

    float rotated[GGML_TQNC_DIM];
    for (int i = 0; i < GGML_TQNC_DIM; ++i) rotated[i] = src[i]/(norm + 1e-8f);
    tqnc_wht(rotated);
    for (int i = 0; i < GGML_TQNC_DIM; i += 2) {
        if (!isfinite(rotated[i]) || !isfinite(rotated[i + 1])) return false;
        block.qs[i/2] = (uint8_t) (tqnc_index(rotated[i]) | (tqnc_index(rotated[i + 1]) << 4));
    }
    *dst = block;
    return true;
}

bool ggml_tqnc_quantize_value(const float * src, ggml_tqnc_value * dst) {
    if (!src || !dst) return false;
    float min = src[0], max = src[0];
    for (int i = 0; i < GGML_TQNC_DIM; ++i) {
        if (!isfinite(src[i])) return false;
        if (src[i] < min) min = src[i];
        if (src[i] > max) max = src[i];
    }
    const float scale = fmaxf((max - min)/15.0f, 1e-8f);
    if (!isfinite(scale)) return false;
    ggml_tqnc_value block;
    block.scale = ggml_fp32_to_fp16(scale);
    block.minimum = ggml_fp32_to_fp16(min);
    if (!isfinite(ggml_fp16_to_fp32(block.scale)) || !isfinite(ggml_fp16_to_fp32(block.minimum))) return false;
    const float reconstructed_max = 15.0f*ggml_fp16_to_fp32(block.scale) + ggml_fp16_to_fp32(block.minimum);
    if (!isfinite(ggml_fp16_to_fp32(ggml_fp32_to_fp16(reconstructed_max)))) return false;
    for (int i = 0; i < GGML_TQNC_DIM; i += 2) {
        const float a = floorf((src[i]     - min)/scale + 0.5f);
        const float b = floorf((src[i + 1] - min)/scale + 0.5f);
        const int q0 = a < 0 ? 0 : a > 15 ? 15 : (int) a;
        const int q1 = b < 0 ? 0 : b > 15 ? 15 : (int) b;
        block.qs[i/2] = (uint8_t) (q0 | (q1 << 4));
    }
    *dst = block;
    return true;
}

bool ggml_tqnc_dequantize_key(const ggml_tqnc_key * src, float * dst) {
    if (!src || !dst) return false;
    const float norm = ggml_fp16_to_fp32(src->norm);
    if (!isfinite(norm) || norm < 0.0f) return false;
    float rotated[GGML_TQNC_DIM];
    float recon_sq = 0.0f;
    for (int i = 0; i < GGML_TQNC_DIM; ++i) {
        const int index = (src->qs[i/2] >> (4*(i % 2))) & 15;
        rotated[i] = tqnc_centroids[index];
        recon_sq += rotated[i]*rotated[i];
    }
    const float scale = norm/sqrtf(recon_sq + 1e-16f);
    for (int i = 0; i < GGML_TQNC_DIM; ++i) rotated[i] *= scale;
    tqnc_wht(rotated);
    for (int i = 0; i < GGML_TQNC_DIM; ++i) {
        if (!isfinite(rotated[i]) || !isfinite(ggml_fp16_to_fp32(ggml_fp32_to_fp16(rotated[i])))) return false;
    }
    memcpy(dst, rotated, sizeof(rotated));
    return true;
}

bool ggml_tqnc_dequantize_value(const ggml_tqnc_value * src, float * dst) {
    if (!src || !dst) return false;
    const float scale = ggml_fp16_to_fp32(src->scale);
    const float min = ggml_fp16_to_fp32(src->minimum);
    if (!isfinite(scale) || scale < 0.0f || !isfinite(min)) return false;
    if (!isfinite(ggml_fp16_to_fp32(ggml_fp32_to_fp16(15.0f*scale + min)))) return false;
    float result[GGML_TQNC_DIM];
    for (int i = 0; i < GGML_TQNC_DIM; ++i) {
        const int index = (src->qs[i/2] >> (4*(i % 2))) & 15;
        result[i] = index*scale + min;
        if (!isfinite(result[i])) return false;
    }
    memcpy(dst, result, sizeof(result));
    return true;
}

void ggml_tqnc_quantize_key_rows(const float * src, void * dst, int64_t n) {
    GGML_ASSERT(n >= 0 && n % GGML_TQNC_DIM == 0);
    for (int64_t i = 0; i < n/GGML_TQNC_DIM; ++i) {
        if (!ggml_tqnc_quantize_key(src + i*GGML_TQNC_DIM, (ggml_tqnc_key *) dst + i)) {
            GGML_ABORT("invalid TurboQuant NC key input");
        }
    }
}

void ggml_tqnc_quantize_value_rows(const float * src, void * dst, int64_t n) {
    GGML_ASSERT(n >= 0 && n % GGML_TQNC_DIM == 0);
    for (int64_t i = 0; i < n/GGML_TQNC_DIM; ++i) {
        if (!ggml_tqnc_quantize_value(src + i*GGML_TQNC_DIM, (ggml_tqnc_value *) dst + i)) {
            GGML_ABORT("invalid TurboQuant NC value input");
        }
    }
}

void ggml_tqnc_dequantize_key_rows(const void * src, float * dst, int64_t n) {
    GGML_ASSERT(n >= 0 && n % GGML_TQNC_DIM == 0);
    for (int64_t i = 0; i < n/GGML_TQNC_DIM; ++i) {
        if (!ggml_tqnc_dequantize_key((const ggml_tqnc_key *) src + i, dst + i*GGML_TQNC_DIM)) {
            GGML_ABORT("invalid TurboQuant NC key metadata");
        }
    }
}

void ggml_tqnc_dequantize_value_rows(const void * src, float * dst, int64_t n) {
    GGML_ASSERT(n >= 0 && n % GGML_TQNC_DIM == 0);
    for (int64_t i = 0; i < n/GGML_TQNC_DIM; ++i) {
        if (!ggml_tqnc_dequantize_value((const ggml_tqnc_value *) src + i, dst + i*GGML_TQNC_DIM)) {
            GGML_ABORT("invalid TurboQuant NC value metadata");
        }
    }
}
