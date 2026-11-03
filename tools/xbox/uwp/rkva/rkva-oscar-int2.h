#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

// RKVA OSCAR INT2 (Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache)
//
// Mathematical Formulation (arXiv:2605.17757):
// 1. Spectral Covariance-Aware Rotation:
//    Q' = Q @ M_k,  K' = K @ M_k  (post-RoPE)
//    Because M_k is orthogonal (M_k @ M_k^T = I), the attention dot product is
//    strictly invariant: Q' @ (K')^T = (Q @ M_k) @ (M_k^T @ K^T) = Q @ K^T.
//    For V: V' = V @ M_v; after attention A @ V', the output is de-rotated by M_v^T
//    or absorbed into W_o.
//
// 2. Outlier Clipping:
//    Clamps rotated activations to a percentile threshold (e.g. 0.96 for K, 0.92 for V)
//    to prevent severe kurtotic outliers from dominating the variance.
//
// 3. Zero-Centering:
//    Subtracts mean mu before Lloyd-Max quant to align with the N(0, sigma^2) assumption.
//    The mean is stored in the 16-bit float field 'm' of each block so that every
//    block can be decoded independently on GPU SIMD lanes.
//
// 4. Lloyd-Max Optimal Quantization for N(0, 1):
//    Centroids: {-0.9816, -0.4528, 0.4528, 0.9816}
//    Decision thresholds: [-0.6745, 0.0, 0.6745]
//    Sigma is stored in the 16-bit float field 'd' of each block.
//
// 5. Block Layout (12 bytes per 32 elements, 96 bytes per 256-dim head row):
//    - Bytes 0..1:   uint16_t d (fp16 per-block sigma)
//    - Bytes 2..3:   uint16_t m (fp16 per-block / group mean)
//    - Bytes 4..11:  uint8_t qs[8] (32 elements, 4 packed 2-bit codes per byte)
//    A 256-dim row has exactly 8 blocks = 96 bytes (vs 136 bytes SAW-INT4, 272 bytes Q8_0, 512 bytes F16).

#define RKVA_OSCAR_INT2_DIM 256
#define RKVA_OSCAR_INT2_BLOCK 32
#define RKVA_OSCAR_INT2_NUM_BLOCKS (RKVA_OSCAR_INT2_DIM / RKVA_OSCAR_INT2_BLOCK) // 8
#define RKVA_OSCAR_INT2_BYTES 96

#pragma pack(push, 1)
struct block_oscar_int2 {
    uint16_t d;           // per-block sigma (fp16)
    uint16_t m;           // per-block mean (fp16)
    uint8_t  qs[8];       // 32 packed 2-bit values (4 per byte)
};
#pragma pack(pop)

static_assert(sizeof(block_oscar_int2) == 12, "block_oscar_int2 must be 12 bytes");

#pragma pack(push, 1)
struct rkva_oscar_int2_row {
    block_oscar_int2 blocks[RKVA_OSCAR_INT2_NUM_BLOCKS];
};
#pragma pack(pop)

static_assert(sizeof(rkva_oscar_int2_row) == RKVA_OSCAR_INT2_BYTES, "rkva_oscar_int2_row must be 96 bytes");

namespace rkva_oscar_int2 {

// IEEE 754 half-precision float conversions (matching D3D12 f16tof32 and f32tof16)
inline uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant = (mant | 0x800000) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x0FFF + ((mant >> 13) & 1)) >> 13));
    } else if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x0200 : 0));
    }
    return static_cast<uint16_t>(sign | (exp << 10) | ((mant + 0x0FFF + ((mant >> 13) & 1)) >> 13));
}

inline float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t val = 0;

    if (exp == 0) {
        if (mant != 0) {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FF;
            val = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        } else {
            val = sign;
        }
    } else if (exp == 31) {
        val = sign | 0x7F800000 | (mant << 13);
    } else {
        val = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &val, sizeof(float));
    return f;
}

// Lloyd-Max optimal 4-level centroids for N(0,1)
static const float LM_CENTROIDS[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};

// Quantize one value to a 2-bit Lloyd-Max code against per-block sigma
inline uint8_t lm_quantize(float v, float inv_sigma) {
    float vs = v * inv_sigma;
    if (vs < -0.6745f) return 0;
    if (vs <  0.0f)    return 1;
    if (vs <  0.6745f) return 2;
    return 3;
}

// Outlier clipping on rotated activations
inline void apply_oscar_clip(float* x, int n, float clip_ratio) {
    if (clip_ratio <= 0.0f || clip_ratio >= 1.0f) return;
    float absv[RKVA_OSCAR_INT2_DIM];
    for (int i = 0; i < n; ++i) absv[i] = std::fabs(x[i]);
    std::sort(absv, absv + n);
    int idx = static_cast<int>(clip_ratio * n);
    if (idx >= n) idx = n - 1;
    const float thr = absv[idx];
    for (int i = 0; i < n; ++i) {
        if (x[i] >  thr) x[i] =  thr;
        if (x[i] < -thr) x[i] = -thr;
    }
}

// Rotate 256-dim vector by orthogonal matrix M (dst = src @ M)
inline void rotate_256(const float* src, const float* rot_matrix, float* dst) {
    for (int j = 0; j < RKVA_OSCAR_INT2_DIM; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < RKVA_OSCAR_INT2_DIM; ++i) {
            sum += src[i] * rot_matrix[i * RKVA_OSCAR_INT2_DIM + j];
        }
        dst[j] = sum;
    }
}

// Quantize single 256-dim row into 8 block_oscar_int2 blocks
inline void quantize_row_oscar_int2_single(const float* src, block_oscar_int2* dst, float clip_ratio = 0.96f) {
    float tmp[RKVA_OSCAR_INT2_DIM];
    std::memcpy(tmp, src, RKVA_OSCAR_INT2_DIM * sizeof(float));

    // 1. Zero-centering across the entire 256-dim head
    float mean = 0.0f;
    for (int j = 0; j < RKVA_OSCAR_INT2_DIM; ++j) {
        mean += tmp[j];
    }
    mean /= static_cast<float>(RKVA_OSCAR_INT2_DIM);
    for (int j = 0; j < RKVA_OSCAR_INT2_DIM; ++j) {
        tmp[j] -= mean;
    }

    // 2. OSCAR outlier clip on zero-centered rotated activations
    apply_oscar_clip(tmp, RKVA_OSCAR_INT2_DIM, clip_ratio);

    // 3. Quantize each 32-element block
    const uint16_t m_fp16 = float_to_half(mean);

    for (int ib = 0; ib < RKVA_OSCAR_INT2_NUM_BLOCKS; ++ib) {
        const float* blk = tmp + ib * RKVA_OSCAR_INT2_BLOCK;

        float sum_sq = 0.0f;
        for (int j = 0; j < RKVA_OSCAR_INT2_BLOCK; ++j) {
            sum_sq += blk[j] * blk[j];
        }
        const float sigma = std::sqrt(sum_sq / static_cast<float>(RKVA_OSCAR_INT2_BLOCK));
        const float inv_sigma = (sigma > 1e-8f) ? 1.0f / sigma : 0.0f;

        dst[ib].d = float_to_half(sigma);
        dst[ib].m = m_fp16;

        for (int j = 0; j < RKVA_OSCAR_INT2_BLOCK / 4; ++j) {
            uint8_t packed = 0;
            for (int b = 0; b < 4; ++b) {
                uint8_t code = lm_quantize(blk[j * 4 + b], inv_sigma);
                packed |= static_cast<uint8_t>((code & 0x03) << (2 * b));
            }
            dst[ib].qs[j] = packed;
        }
    }
}

// Dequantize single 256-dim row (golden reference)
inline void dequantize_row_oscar_int2_single(const block_oscar_int2* src, float* dst) {
    for (int ib = 0; ib < RKVA_OSCAR_INT2_NUM_BLOCKS; ++ib) {
        const float sigma = half_to_float(src[ib].d);
        const float mean  = half_to_float(src[ib].m);
        float* blk = dst + ib * RKVA_OSCAR_INT2_BLOCK;

        for (int j = 0; j < RKVA_OSCAR_INT2_BLOCK / 4; ++j) {
            const uint8_t packed = src[ib].qs[j];
            for (int b = 0; b < 4; ++b) {
                const uint8_t code = (packed >> (2 * b)) & 0x03;
                blk[j * 4 + b] = mean + sigma * LM_CENTROIDS[code];
            }
        }
    }
}

// Multi-row batch quantization (n_elem must be multiple of 256)
inline void quantize_row_oscar_int2(const float* src, block_oscar_int2* dst, int64_t n_elem, float clip_ratio = 0.96f) {
    const int64_t n_rows = n_elem / RKVA_OSCAR_INT2_DIM;
    for (int64_t r = 0; r < n_rows; ++r) {
        quantize_row_oscar_int2_single(src + r * RKVA_OSCAR_INT2_DIM,
                                       dst + r * RKVA_OSCAR_INT2_NUM_BLOCKS,
                                       clip_ratio);
    }
}

// Multi-row batch dequantization (n_elem must be multiple of 256)
inline void dequantize_row_oscar_int2(const block_oscar_int2* src, float* dst, int64_t n_elem) {
    const int64_t n_rows = n_elem / RKVA_OSCAR_INT2_DIM;
    for (int64_t r = 0; r < n_rows; ++r) {
        dequantize_row_oscar_int2_single(src + r * RKVA_OSCAR_INT2_NUM_BLOCKS,
                                         dst + r * RKVA_OSCAR_INT2_DIM);
    }
}

} // namespace rkva_oscar_int2
