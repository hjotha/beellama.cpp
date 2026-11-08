#pragma once
// Deliberately scalar, one independent head per CUDA thread. This preserves the
// reference reduction order; do not replace with warp reductions without byte
// and quality qualification. Also compiled on the host by the oracle test.
#include "ggml.h"

#include <cmath>
#include <cstdint>
#if defined(__CUDACC__)
#    define RKVA_HD __device__ __forceinline__
#else
#    define RKVA_HD inline
#endif
namespace rkva_native_low4 {
RKVA_HD float add(float a, float b) {
#if defined(__CUDACC__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}

RKVA_HD float sub(float a, float b) {
#if defined(__CUDACC__)
    return __fsub_rn(a, b);
#else
    return a - b;
#endif
}

RKVA_HD float mul(float a, float b) {
#if defined(__CUDACC__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}

RKVA_HD float div(float a, float b) {
#if defined(__CUDACC__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}

RKVA_HD uint16_t half_bits(float x) {
#if defined(__CUDACC__)
    return __half_as_ushort(__float2half_rn(x));
#else
    return ggml_fp32_to_fp16(x);
#endif
}

RKVA_HD float half_value(uint16_t x) {
#if defined(__CUDACC__)
    return __half2float(__ushort_as_half(x));
#else
    return ggml_fp16_to_fp32(x);
#endif
}

RKVA_HD bool finite(float x) {
    return x <= 3.402823466e38f && x >= -3.402823466e38f;
}

RKVA_HD bool put_half(uint8_t * p, float x) {
    const uint16_t h = half_bits(x);
    p[0]             = uint8_t(h);
    p[1]             = uint8_t(h >> 8);
    return (h & 0x7c00u) != 0x7c00u;
}

RKVA_HD int clip(float x) {
    return x < 0 ? 0 : x > 15 ? 15 : int(x);
}

RKVA_HD int iq_index(float x) {
    const int v[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
    if (x <= v[0]) {
        return 0;
    }
    if (x >= v[15]) {
        return 15;
    }
    int lo = 0, hi = 15;
    while (hi - lo > 1) {
        int m = (lo + hi) / 2;
        if (x < v[m]) {
            hi = m;
        } else {
            lo = m;
        }
    }
    return sub(x, float(v[hi - 1])) < sub(float(v[hi]), x) ? hi - 1 : hi;
}

RKVA_HD bool q41(const float * x, uint8_t * p) {
    float lo = x[0], hi = x[0];
    for (int j = 1; j < 32; ++j) {
        if (x[j] < lo) {
            lo = x[j];
        }
        if (x[j] > hi) {
            hi = x[j];
        }
    }
    float d = div(sub(hi, lo), 15.0f), inv = d ? div(1.0f, d) : 0;
    // An overflowed reciprocal has no defined CPU integer-conversion contract.
    if (!finite(inv) || !put_half(p, d) || !put_half(p + 2, lo)) {
        return false;
    }
    for (int j = 0; j < 16; ++j) {
        p[4 + j] =
            uint8_t(clip(add(mul(sub(x[j], lo), inv), 0.5f)) | (clip(add(mul(sub(x[j + 16], lo), inv), 0.5f)) << 4));
    }
    return true;
}

RKVA_HD bool iq4(const float * x, uint8_t * p) {
    const int v[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
    float     amax = 0, vmax = 0, w[32];
    for (int j = 0; j < 32; ++j) {
        w[j] = mul(x[j], x[j]);
        if (fabsf(x[j]) > amax) {
            amax = fabsf(x[j]);
            vmax = x[j];
        }
    }
    float d = 0;
    if (amax >= 1e-15f) {
        d         = div(-vmax, -127.0f);
        float inv = div(1.0f, d), sx = 0, sq = 0;
        for (int j = 0; j < 32; ++j) {
            float q = float(v[iq_index(mul(inv, x[j]))]);
            sx      = add(sx, mul(mul(w[j], q), x[j]));
            sq      = add(sq, mul(mul(w[j], q), q));
        }
        d          = sq > 0 ? div(sx, sq) : 0;
        float best = mul(d, sx);
        for (int t = -7; t <= 7; ++t) {
            inv = div(float(t - 127), vmax);
            sx = sq = 0;
            for (int j = 0; j < 32; ++j) {
                float q = float(v[iq_index(mul(inv, x[j]))]);
                sx      = add(sx, mul(mul(w[j], q), x[j]));
                sq      = add(sq, mul(mul(w[j], q), q));
            }
            if (sq > 0 && mul(sx, sx) > mul(best, sq)) {
                d    = div(sx, sq);
                best = mul(d, sx);
            }
        }
    }
    if (!put_half(p, d)) {
        return false;
    }
    float inv = d ? div(1.0f, d) : 0;
    for (int j = 0; j < 16; ++j) {
        p[2 + j] = uint8_t(iq_index(mul(inv, x[j])) | (iq_index(mul(inv, x[j + 16])) << 4));
    }
    return true;
}

RKVA_HD void wht(float * x, int n, float scale) {
    for (int s = 1; s < n; s *= 2) {
        for (int b = 0; b < n; b += 2 * s) {
            for (int j = 0; j < s; ++j) {
                float a = x[b + j], c = x[b + j + s];
                x[b + j]     = add(a, c);
                x[b + j + s] = sub(a, c);
            }
        }
    }
    for (int j = 0; j < n; ++j) {
        x[j] = mul(x[j], scale);
    }
}

RKVA_HD bool saw(const float * x, uint8_t * p, bool avx) {
    float r[256];
    for (int j = 0; j < 256; ++j) {
        r[j] = x[j];
    }
    for (int j = 0; j < 256; j += 16) {
        wht(r + j, 16, avx ? 1.0f : 0.25f);
    }
    float lo = r[0], hi = r[0];
    for (int j = 0; j < 256; ++j) {
        if (!finite(r[j])) {
            return false;
        }
        if (r[j] < lo) {
            lo = r[j];
        }
        if (r[j] > hi) {
            hi = r[j];
        }
    }
    if (avx) {
        // Match x86 MINPS/MAXPS second-operand tie semantics, including signed zero.
        float lower[8], upper[8];
        for (int lane = 0; lane < 8; ++lane) {
            lower[lane] = 1e30f;
            upper[lane] = -1e30f;
        }
        for (int base = 0; base < 256; base += 16) {
            for (int lane = 0; lane < 8; ++lane) {
                float a = r[base + lane], b = r[base + 8 + lane];
                float l = a < b ? a : b, h = a > b ? a : b;
                lower[lane] = lower[lane] < l ? lower[lane] : l;
                upper[lane] = upper[lane] > h ? upper[lane] : h;
            }
        }
        for (int lane = 0; lane < 4; ++lane) {
            lower[lane] = lower[lane] < lower[lane + 4] ? lower[lane] : lower[lane + 4];
            upper[lane] = upper[lane] > upper[lane + 4] ? upper[lane] : upper[lane + 4];
        }
        for (int lane = 0; lane < 2; ++lane) {
            lower[lane] = lower[lane] < lower[lane + 2] ? lower[lane] : lower[lane + 2];
            upper[lane] = upper[lane] > upper[lane + 2] ? upper[lane] : upper[lane + 2];
        }
        lo = mul(lower[0] < lower[1] ? lower[0] : lower[1], 0.25f);
        hi = mul(upper[0] > upper[1] ? upper[0] : upper[1], 0.25f);
    }
    float range = sub(hi, lo);
    if (range < 1e-8f) {
        range = 1e-8f;
    }
    float scale = div(range, 15), zero = div(-lo, scale);
    if (!finite(scale) || !finite(zero)) {
        return false;
    }
    // Byte stores also work for the packed wire ABI without aligned float casts.
    const auto * a = reinterpret_cast<const unsigned char *>(&scale);
    const auto * b = reinterpret_cast<const unsigned char *>(&zero);
    for (int j = 0; j < 4; ++j) {
        p[128 + j] = a[j];
        p[132 + j] = b[j];
    }
    const float inverse4 = div(0.25f, scale), bias = add(zero, 0.5f);
    for (int j = 0; j < 128; ++j) {
        float a = avx ? add(mul(r[j], inverse4), bias) : add(add(div(r[j], scale), zero), 0.5f);
        float b = avx ? add(mul(r[j + 128], inverse4), bias) : add(add(div(r[j + 128], scale), zero), 0.5f);
        p[j]    = uint8_t(clip(floorf(a)) | (clip(floorf(b)) << 4));
    }
    return true;
}

RKVA_HD int nc_index(float x) {
    const float v[16] = { -1.7068263888e-01f, -1.2927794456e-01f, -1.0111761093e-01f, -7.8516095877e-02f,
                          -5.8903016150e-02f, -4.1054993868e-02f, -2.4258611724e-02f, -8.0267284065e-03f,
                          8.0267284065e-03f,  2.4258611724e-02f,  4.1054993868e-02f,  5.8903016150e-02f,
                          7.8516095877e-02f,  1.0111761093e-01f,  1.2927794456e-01f,  1.7068263888e-01f };
    int         lo = 0, hi = 15;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (x >= mul(add(v[m], v[m + 1]), 0.5f)) {
            lo = m + 1;
        } else {
            hi = m;
        }
    }
    return lo;
}

RKVA_HD bool nc(const float * x, uint8_t * p, bool key) {
    if (key) {
        float sum = 0;
        for (int j = 0; j < 256; ++j) {
            sum = add(sum, mul(x[j], x[j]));
        }
#if defined(__CUDACC__)
        float norm = __fsqrt_rn(sum);
#else
        float norm = sqrtf(sum);
#endif
        if (!finite(norm) || !put_half(p + 128, norm)) {
            return false;
        }
        p[130] = p[131] = 0;
        float r[256];
        for (int j = 0; j < 256; ++j) {
            r[j] = div(x[j], add(norm, 1e-8f));
        }
        wht(r, 256, 0.0625f);
        for (int j = 0; j < 256; j += 2) {
            if (!finite(r[j]) || !finite(r[j + 1])) {
                return false;
            }
            p[j / 2] = uint8_t(nc_index(r[j]) | (nc_index(r[j + 1]) << 4));
        }
    } else {
        float lo = x[0], hi = x[0];
        for (int j = 1; j < 256; ++j) {
            if (x[j] < lo) {
                lo = x[j];
            }
            if (x[j] > hi) {
                hi = x[j];
            }
        }
        float scale = div(sub(hi, lo), 15);
        if (scale < 1e-8f) {
            scale = 1e-8f;
        }
        if (!finite(scale) || !put_half(p + 128, scale) || !put_half(p + 130, lo)) {
            return false;
        }
        float top = add(mul(15, half_value(half_bits(scale))), half_value(half_bits(lo)));
        if ((half_bits(top) & 0x7c00u) == 0x7c00u) {
            return false;
        }
        for (int j = 0; j < 256; j += 2) {
            p[j / 2] = uint8_t(clip(floorf(add(div(sub(x[j], lo), scale), 0.5f))) |
                               (clip(floorf(add(div(sub(x[j + 1], lo), scale), 0.5f))) << 4));
        }
    }
    return true;
}

// format is the GGML type for Q4_1/IQ4_NL/SAW_INT4; TQNC_K selects the K/V pair.
RKVA_HD bool encode(int format, const float * x, uint8_t * p, bool key, uint32_t flags = 0) {
    if ((flags & ~1u) || (flags && format != GGML_TYPE_SAW_INT4)) {
        return false;
    }
    for (int j = 0; j < 256; ++j) {
        if (!finite(x[j])) {
            return false;
        }
    }
    if (format == GGML_TYPE_Q4_1 || format == GGML_TYPE_IQ4_NL) {
        for (int j = 0; j < 256; ++j) {
            if (fabsf(x[j]) > 65504) {
                return false;
            }
        }
        for (int b = 0; b < 8; ++b) {
            if (!(format == GGML_TYPE_Q4_1 ? q41(x + 32 * b, p + 20 * b) : iq4(x + 32 * b, p + 18 * b))) {
                return false;
            }
        }
        return true;
    }
    if (format == GGML_TYPE_SAW_INT4) {
        return saw(x, p, (flags & 1u) != 0);
    }
    if (format == GGML_TYPE_TQNC_K) {
        return nc(x, p, key);
    }
    return false;
}
}  // namespace rkva_native_low4

#undef RKVA_HD
