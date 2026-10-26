#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
namespace rkva_kvarn_range {
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

}
int main() {
    float src[256], dst[256];
    for (int i = 0; i < 256; ++i) src[i] = std::sin(float(i) * 0.37f);
    const int heads = 24;      // Qwen3.8 24 Q heads, head_dim 256
    const int rounds = 20000;
    // warmup
    for (int r = 0; r < rounds; ++r) for (int h = 0; h < heads; ++h) rkva_kvarn_range::butterfly_wht256(src, dst);
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < rounds; ++r) for (int h = 0; h < heads; ++h) rkva_kvarn_range::butterfly_wht256(src, dst);
    auto t1 = std::chrono::steady_clock::now();
    double ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / rounds;
    std::printf("host ns per layer (24 heads, current scalar WHT): %.1f\n", ns);
    std::printf("checksum %.9g\n", dst[7]);
    return 0;
}
