#include "rkva-oscar-int2.h"

extern "C" {

void rkva_oscar_rotate_256(const float* src, const float* rot_matrix, float* dst) {
    rkva_oscar_int2::rotate_256(src, rot_matrix, dst);
}

void rkva_quantize_row_oscar_int2(const float* src, block_oscar_int2* dst, int64_t n_elem, float clip_ratio) {
    rkva_oscar_int2::quantize_row_oscar_int2(src, dst, n_elem, clip_ratio);
}

void rkva_dequantize_row_oscar_int2(const block_oscar_int2* src, float* dst, int64_t n_elem) {
    rkva_oscar_int2::dequantize_row_oscar_int2(src, dst, n_elem);
}

}
