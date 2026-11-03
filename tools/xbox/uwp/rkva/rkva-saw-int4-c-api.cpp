#include "rkva-saw-int4.h"

extern "C" {

void rkva_bdr16_transform_256(const float* src, float* dst) {
    rkva_saw_int4::bdr16_transform_256(src, dst);
}

void rkva_quantize_row_saw_int4(const float* src, block_saw_int4* dst, int64_t n_elem) {
    rkva_saw_int4::quantize_row_saw_int4(src, dst, n_elem);
}

void rkva_dequantize_row_saw_int4(const block_saw_int4* src, float* dst, int64_t n_elem) {
    rkva_saw_int4::dequantize_row_saw_int4(src, dst, n_elem);
}

void rkva_dequantize_row_saw_int4_rotated(const block_saw_int4* src, float* dst, int64_t n_elem) {
    const int64_t n_blocks = n_elem / RKVA_SAW_INT4_DIM;
    for (int64_t b = 0; b < n_blocks; ++b) {
        rkva_saw_int4::dequantize_row_saw_int4_rotated_single(src + b, dst + b * RKVA_SAW_INT4_DIM);
    }
}

}
