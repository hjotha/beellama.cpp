#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

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

// Block-Diagonal Rotation (BDR-16) on 256 dimensions (16 independent 16-point FWHT blocks).
// Self-inverse: bdr16_transform_256(bdr16_transform_256(x)) == x.
inline void bdr16_transform_256(const float* src, float* dst) {
    for (int b = 0; b < RKVA_SAW_INT4_DIM; b += RKVA_SAW_INT4_BLOCK) {
        float tmp[RKVA_SAW_INT4_BLOCK];
        std::memcpy(tmp, src + b, RKVA_SAW_INT4_BLOCK * sizeof(float));
        fwht16_scaled(tmp);
        std::memcpy(dst + b, tmp, RKVA_SAW_INT4_BLOCK * sizeof(float));
    }
}

// Token-wise quantization of 256 float elements into block_saw_int4.
inline void quantize_row_saw_int4_single(const float* src, block_saw_int4* dst) {
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
}

// Dequantize block_saw_int4 into rotated domain without inverse BDR transform
// (useful for GPU decode attention where Q is rotated).
inline void dequantize_row_saw_int4_rotated_single(const block_saw_int4* src, float* dst) {
    const float scale = src->scale;
    const float zero  = src->zero;

    for (int j = 0; j < RKVA_SAW_INT4_HALF; ++j) {
        uint8_t byte = src->qs[j];
        uint32_t q1 = byte & 0x0Fu;
        uint32_t q2 = (byte >> 4) & 0x0Fu;

        dst[j]                      = (static_cast<float>(q1) - zero) * scale;
        dst[j + RKVA_SAW_INT4_HALF] = (static_cast<float>(q2) - zero) * scale;
    }
}

// Full dequantization: dequantize quants + inverse BDR-16 des-rotation.
inline void dequantize_row_saw_int4_single(const block_saw_int4* src, float* dst) {
    float rotated[RKVA_SAW_INT4_DIM];
    dequantize_row_saw_int4_rotated_single(src, rotated);
    bdr16_transform_256(rotated, dst); // self-inverse BDR-16
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
