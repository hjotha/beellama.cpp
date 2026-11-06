#include <cstdint>
inline void butterfly_wht256(const float* src, float* dst) {
    float t[256];
    for (int i = 0; i < 256; ++i)
        t[i] = src[i];
    for (int slice = 0; slice < 2; ++slice) {
        float* s = t + slice * 128;
        for (int stride = 1; stride < 128; stride <<= 1) {
            for (int i = 0; i < 128; i += 2 * stride) {
                for (int j = 0; j < stride; ++j) {
                    const float u = s[i + j];
                    const float v = s[i + j + stride];
                    s[i + j] = u + v;
                    s[i + j + stride] = u - v;
                }
            }
        }
        for (int lane = 0; lane < 128; ++lane) {
            s[lane] *= 0.08838834764831845f;
        }
    }
    for (int lane = 0; lane < 128; ++lane) {
        const float a = t[lane];
        const float b = t[lane + 128];
        dst[lane] = (a + b) * 0.7071067811865475f;
        dst[lane + 128] = (a - b) * 0.7071067811865475f;
    }
}
extern "C" void reference_wht(const float* src, float* dst, unsigned heads) {
    for (unsigned h=0; h<heads; ++h) butterfly_wht256(src+256*h,dst+256*h);
}
