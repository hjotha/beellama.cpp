#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

// RKVA SAW-INT4 (Block-Diagonal Walsh-Hadamard Transform + INT4 KV-cache)
//
// Formulation:
// 1. Block-Diagonal Rotation (BDR-16):
//    Head dimension D (e.g. 256) is partitioned into blocks of N = 16 elements.
//    Each block is rotated via normalized 16-point Sylvester Fast Walsh-Hadamard Transform (FWHT):
//      v_rot = (1 / sqrt(16)) * H_16 * v = 0.25 * H_16 * v.
//    Because H_16 is symmetric and orthogonal with H_16^2 = 16 * I_16,
//    the transform is self-inverse: (0.25 * H_16) * (0.25 * H_16) = I_16.
//
// 2. Token-Wise Affine INT4 Quantization:
//    val_min = min(v_rot), val_max = max(v_rot) across the 256 dimensions.
//    val_range = max(val_max - val_min, 1e-8f)
//    scale = val_range / 15.0f
//    zero  = -val_min / scale
//    q_i   = clamp(round(v_rot[i] / scale + zero), 0, 15)
//
// 3. Split-Half Nibble Packing:
//    For j in [0, 127]:
//      low nibble  (bits 0..3) = q[j]       (elements 0..127)
//      high nibble (bits 4..7) = q[j + 128] (elements 128..255)
//      qs[j] = low | (high << 4)
//
// 4. Memory Layout (136 bytes per 256-dim head row):
//    - Bytes 0..127:   128 bytes qs (packed 4-bit quants)
//    - Bytes 128..131: float scale (4 bytes, aligned)
//    - Bytes 132..135: float zero  (4 bytes, aligned)

#define RKVA_SAW_INT4_DIM 256
#define RKVA_SAW_INT4_BLOCK 16
#define RKVA_SAW_INT4_HALF 128
#define RKVA_SAW_INT4_BYTES 136

#pragma pack(push, 1)
struct block_saw_int4 {
    uint8_t qs[RKVA_SAW_INT4_HALF]; // 128 bytes of packed 4-bit nibbles
    float   scale;                  // affine scale
    float   zero;                   // affine zero point (-val_min / scale)
};
#pragma pack(pop)

static_assert(sizeof(block_saw_int4) == RKVA_SAW_INT4_BYTES, "block_saw_int4 size must be 136 bytes");

namespace rkva_saw_int4 {

// In-place 16-point Sylvester Fast Walsh-Hadamard Transform scaled by 1/sqrt(16) = 0.25f.
// Self-inverse: applying twice reproduces the exact original vector.
inline void fwht16_scaled(float* v) {
    // 4 butterfly stages for N = 16
    // Stage 1: stride 1
    for (int i = 0; i < 16; i += 2) {
        float u = v[i];
        float w = v[i + 1];
        v[i]     = u + w;
        v[i + 1] = u - w;
    }
    // Stage 2: stride 2
    for (int i = 0; i < 16; i += 4) {
        for (int j = 0; j < 2; ++j) {
            float u = v[i + j];
            float w = v[i + j + 2];
            v[i + j]     = u + w;
            v[i + j + 2] = u - w;
        }
    }
    // Stage 3: stride 4
    for (int i = 0; i < 16; i += 8) {
        for (int j = 0; j < 4; ++j) {
            float u = v[i + j];
            float w = v[i + j + 4];
            v[i + j]     = u + w;
            v[i + j + 4] = u - w;
        }
    }
    // Stage 4: stride 8
    for (int j = 0; j < 8; ++j) {
        float u = v[j];
        float w = v[j + 8];
        v[j]     = u + w;
        v[j + 8] = u - w;
    }
    // Scale by 1/sqrt(16) = 0.25f
    for (int i = 0; i < 16; ++i) {
        v[i] *= 0.25f;
    }
}

#if defined(__AVX2__)
// In-register AVX2 8-element stages 1..3 (stride 1, 2, 4) with exact IEEE-754 evaluation order.
static inline __m256 fwht8_stages123_avx2(__m256 v) {
    // Stage 1: stride 1
    __m256 p1 = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m256 s1 = _mm256_blend_ps(_mm256_add_ps(v, p1), _mm256_sub_ps(p1, v), 0xAA);
    // Stage 2: stride 2
    __m256 p2 = _mm256_shuffle_ps(s1, s1, _MM_SHUFFLE(1, 0, 3, 2));
    __m256 s2 = _mm256_blend_ps(_mm256_add_ps(s1, p2), _mm256_sub_ps(p2, s1), 0xCC);
    // Stage 3: stride 4 (cross 128-bit lanes)
    __m256 p3 = _mm256_permute2f128_ps(s2, s2, 0x01);
    return _mm256_blend_ps(_mm256_add_ps(s2, p3), _mm256_sub_ps(p3, s2), 0xF0);
}

static inline void fwht16_unscaled_avx2(__m256 lo, __m256 hi, __m256 & out_lo, __m256 & out_hi) {
    __m256 s3_lo = fwht8_stages123_avx2(lo);
    __m256 s3_hi = fwht8_stages123_avx2(hi);
    out_lo = _mm256_add_ps(s3_lo, s3_hi);
    out_hi = _mm256_sub_ps(s3_lo, s3_hi);
}

static inline void fwht16_avx2(__m256 lo, __m256 hi, __m256 & out_lo, __m256 & out_hi) {
    __m256 u_lo, u_hi;
    fwht16_unscaled_avx2(lo, hi, u_lo, u_hi);
    __m256 v_quarter = _mm256_set1_ps(0.25f);
    out_lo = _mm256_mul_ps(u_lo, v_quarter);
    out_hi = _mm256_mul_ps(u_hi, v_quarter);
}

static inline float hmin_avx2(__m256 v) {
    __m128 l = _mm256_castps256_ps128(v);
    __m128 h = _mm256_extractf128_ps(v, 1);
    l = _mm_min_ps(l, h);
    l = _mm_min_ps(l, _mm_movehl_ps(l, l));
    l = _mm_min_ss(l, _mm_shuffle_ps(l, l, 1));
    return _mm_cvtss_f32(l);
}

static inline float hmax_avx2(__m256 v) {
    __m128 l = _mm256_castps256_ps128(v);
    __m128 h = _mm256_extractf128_ps(v, 1);
    l = _mm_max_ps(l, h);
    l = _mm_max_ps(l, _mm_movehl_ps(l, l));
    l = _mm_max_ss(l, _mm_shuffle_ps(l, l, 1));
    return _mm_cvtss_f32(l);
}
#endif

// Block-Diagonal Rotation (BDR-16) on 256 dimensions (16 independent 16-point FWHT blocks).
// Self-inverse: bdr16_transform_256(bdr16_transform_256(x)) == x.
inline void bdr16_transform_256(const float* src, float* dst) {
#if defined(__AVX2__)
    for (int b = 0; b < RKVA_SAW_INT4_DIM; b += RKVA_SAW_INT4_BLOCK) {
        __m256 lo = _mm256_loadu_ps(src + b);
        __m256 hi = _mm256_loadu_ps(src + b + 8);
        __m256 out_lo, out_hi;
        fwht16_avx2(lo, hi, out_lo, out_hi);
        _mm256_storeu_ps(dst + b, out_lo);
        _mm256_storeu_ps(dst + b + 8, out_hi);
    }
#else
    for (int b = 0; b < RKVA_SAW_INT4_DIM; b += RKVA_SAW_INT4_BLOCK) {
        float tmp[RKVA_SAW_INT4_BLOCK];
        std::memcpy(tmp, src + b, RKVA_SAW_INT4_BLOCK * sizeof(float));
        fwht16_scaled(tmp);
        std::memcpy(dst + b, tmp, RKVA_SAW_INT4_BLOCK * sizeof(float));
    }
#endif
}

// Token-wise quantization of 256 float elements into block_saw_int4.
inline void quantize_row_saw_int4_single(const float* src, block_saw_int4* dst) {
#if defined(__AVX2__)
    alignas(32) float rotated_unscaled[RKVA_SAW_INT4_DIM];
    __m256 vmin_v = _mm256_set1_ps(1e30f);
    __m256 vmax_v = _mm256_set1_ps(-1e30f);

    // 0.25f = 2^-2 is an exact power of 2 in IEEE-754, so deferring it to scale/inv_scale
    // eliminates 32 vmulps instructions per row while remaining 100% bit-exact.
    for (int b = 0; b < RKVA_SAW_INT4_DIM; b += RKVA_SAW_INT4_BLOCK) {
        __m256 lo = _mm256_loadu_ps(src + b);
        __m256 hi = _mm256_loadu_ps(src + b + 8);
        __m256 out_lo, out_hi;
        fwht16_unscaled_avx2(lo, hi, out_lo, out_hi);
        _mm256_store_ps(rotated_unscaled + b, out_lo);
        _mm256_store_ps(rotated_unscaled + b + 8, out_hi);
        vmin_v = _mm256_min_ps(vmin_v, _mm256_min_ps(out_lo, out_hi));
        vmax_v = _mm256_max_ps(vmax_v, _mm256_max_ps(out_lo, out_hi));
    }

    float val_min = hmin_avx2(vmin_v) * 0.25f;
    float val_max = hmax_avx2(vmax_v) * 0.25f;
    float val_range = std::max(val_max - val_min, 1e-8f);
    float scale = val_range / 15.0f;
    float zero  = -val_min / scale;

    dst->scale = scale;
    dst->zero  = zero;

    __m256 v_inv_scale4 = _mm256_set1_ps(0.25f / scale);
    __m256 v_bias       = _mm256_set1_ps(zero + 0.5f);
    __m256i v_zero_i    = _mm256_setzero_si256();
    __m256i v_max_i     = _mm256_set1_epi32(15);

    for (int j = 0; j < RKVA_SAW_INT4_HALF; j += 16) {
        __m256 v1_a = _mm256_load_ps(rotated_unscaled + j);
        __m256 v2_a = _mm256_load_ps(rotated_unscaled + j + RKVA_SAW_INT4_HALF);
        __m256 v1_b = _mm256_load_ps(rotated_unscaled + j + 8);
        __m256 v2_b = _mm256_load_ps(rotated_unscaled + j + RKVA_SAW_INT4_HALF + 8);

        __m256 f1_a = _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(v1_a, v_inv_scale4), v_bias));
        __m256 f2_a = _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(v2_a, v_inv_scale4), v_bias));
        __m256 f1_b = _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(v1_b, v_inv_scale4), v_bias));
        __m256 f2_b = _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(v2_b, v_inv_scale4), v_bias));

        __m256i q1_a = _mm256_min_epi32(v_max_i, _mm256_max_epi32(v_zero_i, _mm256_cvttps_epi32(f1_a)));
        __m256i q2_a = _mm256_min_epi32(v_max_i, _mm256_max_epi32(v_zero_i, _mm256_cvttps_epi32(f2_a)));
        __m256i q1_b = _mm256_min_epi32(v_max_i, _mm256_max_epi32(v_zero_i, _mm256_cvttps_epi32(f1_b)));
        __m256i q2_b = _mm256_min_epi32(v_max_i, _mm256_max_epi32(v_zero_i, _mm256_cvttps_epi32(f2_b)));

        __m256i p32_a = _mm256_or_si256(q1_a, _mm256_slli_epi32(q2_a, 4));
        __m256i p32_b = _mm256_or_si256(q1_b, _mm256_slli_epi32(q2_b, 4));

        __m256i p16 = _mm256_packus_epi32(p32_a, p32_b);
        p16 = _mm256_permute4x64_epi64(p16, _MM_SHUFFLE(3, 1, 2, 0));
        __m128i p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_extracti128_si256(p16, 1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst->qs + j), p8);
    }
#else
    float rotated[RKVA_SAW_INT4_DIM];
    bdr16_transform_256(src, rotated);

    // Compute min and max across all 256 rotated dimensions
    float val_min = rotated[0];
    float val_max = rotated[0];
    for (int i = 1; i < RKVA_SAW_INT4_DIM; ++i) {
        if (rotated[i] < val_min) val_min = rotated[i];
        if (rotated[i] > val_max) val_max = rotated[i];
    }

    float val_range = std::max(val_max - val_min, 1e-8f);
    float scale = val_range / 15.0f;
    float zero  = -val_min / scale;

    dst->scale = scale;
    dst->zero  = zero;

    // Pack split-half nibbles: low nibble from [0..127], high nibble from [128..255]
    for (int j = 0; j < RKVA_SAW_INT4_HALF; ++j) {
        float v1 = rotated[j];
        float v2 = rotated[j + RKVA_SAW_INT4_HALF];

        int q1 = static_cast<int>(std::floor(v1 / scale + zero + 0.5f));
        int q2 = static_cast<int>(std::floor(v2 / scale + zero + 0.5f));

        q1 = std::max(0, std::min(15, q1));
        q2 = std::max(0, std::min(15, q2));

        dst->qs[j] = static_cast<uint8_t>(q1 | (q2 << 4));
    }
#endif
}

// Dequantize block_saw_int4 into rotated domain without inverse BDR transform
// (useful for GPU decode attention where Q is rotated).
inline void dequantize_row_saw_int4_rotated_single(const block_saw_int4* src, float* dst) {
#if defined(__AVX2__)
    const __m256 v_scale = _mm256_set1_ps(src->scale);
    const __m256 v_zero  = _mm256_set1_ps(src->zero);
    const __m256i mask0f = _mm256_set1_epi32(0x0F);

    for (int j = 0; j < RKVA_SAW_INT4_HALF; j += 16) {
        __m128i b16  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src->qs + j));
        __m256i w_lo = _mm256_cvtepu8_epi32(b16);
        __m256i w_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(b16, 8));

        __m256 f1_lo = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_and_si256(w_lo, mask0f)), v_zero), v_scale);
        __m256 f1_hi = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_and_si256(w_hi, mask0f)), v_zero), v_scale);
        __m256 f2_lo = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(w_lo, 4), mask0f)), v_zero), v_scale);
        __m256 f2_hi = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(w_hi, 4), mask0f)), v_zero), v_scale);

        _mm256_storeu_ps(dst + j, f1_lo);
        _mm256_storeu_ps(dst + j + 8, f1_hi);
        _mm256_storeu_ps(dst + j + RKVA_SAW_INT4_HALF, f2_lo);
        _mm256_storeu_ps(dst + j + RKVA_SAW_INT4_HALF + 8, f2_hi);
    }
#else
    const float scale = src->scale;
    const float zero  = src->zero;

    for (int j = 0; j < RKVA_SAW_INT4_HALF; ++j) {
        uint8_t byte = src->qs[j];
        uint32_t q1 = byte & 0x0Fu;
        uint32_t q2 = (byte >> 4) & 0x0Fu;

        dst[j]                      = (static_cast<float>(q1) - zero) * scale;
        dst[j + RKVA_SAW_INT4_HALF] = (static_cast<float>(q2) - zero) * scale;
    }
#endif
}

// Full dequantization: dequantize quants + inverse BDR-16 des-rotation.
inline void dequantize_row_saw_int4_single(const block_saw_int4* src, float* dst) {
#if defined(__AVX2__)
    // Fold the 0.25f (2^-2) FWHT normalization directly into scale (bit-exact in IEEE-754)
    const __m256 v_scale4 = _mm256_set1_ps(src->scale * 0.25f);
    const __m256 v_zero   = _mm256_set1_ps(src->zero);
    const __m256i mask0f  = _mm256_set1_epi32(0x0F);

    // Process 16 elements in [0..127] and 16 elements in [128..255] simultaneously in registers!
    for (int j = 0; j < RKVA_SAW_INT4_HALF; j += 16) {
        __m128i b16  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src->qs + j));
        __m256i w_lo = _mm256_cvtepu8_epi32(b16);
        __m256i w_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(b16, 8));

        __m256 r1_lo = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_and_si256(w_lo, mask0f)), v_zero), v_scale4);
        __m256 r1_hi = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_and_si256(w_hi, mask0f)), v_zero), v_scale4);
        __m256 out1_lo, out1_hi;
        fwht16_unscaled_avx2(r1_lo, r1_hi, out1_lo, out1_hi);
        _mm256_storeu_ps(dst + j, out1_lo);
        _mm256_storeu_ps(dst + j + 8, out1_hi);

        __m256 r2_lo = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(w_lo, 4), mask0f)), v_zero), v_scale4);
        __m256 r2_hi = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(w_hi, 4), mask0f)), v_zero), v_scale4);
        __m256 out2_lo, out2_hi;
        fwht16_unscaled_avx2(r2_lo, r2_hi, out2_lo, out2_hi);
        _mm256_storeu_ps(dst + RKVA_SAW_INT4_HALF + j, out2_lo);
        _mm256_storeu_ps(dst + RKVA_SAW_INT4_HALF + j + 8, out2_hi);
    }
#else
    float rotated[RKVA_SAW_INT4_DIM];
    dequantize_row_saw_int4_rotated_single(src, rotated);
    bdr16_transform_256(rotated, dst); // self-inverse BDR-16
#endif
}

// Multi-row batch quantization (n_elem must be multiple of 256)
inline void quantize_row_saw_int4(const float* src, block_saw_int4* dst, int64_t n_elem) {
    const int64_t n_blocks = n_elem / RKVA_SAW_INT4_DIM;
    for (int64_t b = 0; b < n_blocks; ++b) {
        quantize_row_saw_int4_single(src + b * RKVA_SAW_INT4_DIM, dst + b);
    }
}

// Multi-row batch dequantization (n_elem must be multiple of 256)
inline void dequantize_row_saw_int4(const block_saw_int4* src, float* dst, int64_t n_elem) {
    const int64_t n_blocks = n_elem / RKVA_SAW_INT4_DIM;
    for (int64_t b = 0; b < n_blocks; ++b) {
        dequantize_row_saw_int4_single(src + b, dst + b * RKVA_SAW_INT4_DIM);
    }
}

} // namespace rkva_saw_int4
