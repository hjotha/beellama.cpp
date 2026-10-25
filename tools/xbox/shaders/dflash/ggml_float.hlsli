// GGML RMSNorm rounds each square to F32, accumulates in F64, then applies
// 1/sqrtf(F32(sum/width)+eps). D3D12 rsqrt and division are not CPU RNE.
// Inspecting the double bits also prevents DXC narrowing the calculation.
float ggml_round_f32(double value) {
    uint low, high;
    asuint(value, low, high);
    int exponent = int((high >> 20u) & 2047u) - 896;
    if (exponent < 1 || exponent > 254) return (float)value;
    uint mantissa = ((high & 0xfffffu) << 3u) | (low >> 29u);
    const uint remainder = low & 0x1fffffffu;
    if (remainder > 0x10000000u ||
        (remainder == 0x10000000u && (mantissa & 1u))) ++mantissa;
    if (mantissa == 0x800000u) { mantissa = 0u; ++exponent; }
    return asfloat((high & 0x80000000u) | (uint(exponent) << 23u) | mantissa);
}

float ggml_fma_f32(float a, float b, float c) {
#if !defined(DFLASH_FAST_MAD)
    precise double value = (double)a * (double)b + (double)c;
    return ggml_round_f32(value);
#else
    return mad(a, b, c);
#endif
}

// ggml-cpu/vec.h AVX2 SiLU uses this polynomial, not the shader exp intrinsic.
float ggml_exp_avx2(float x) {
    const float z = ggml_fma_f32(x, asfloat(0x3fb8aa3bu), 12582912.0f);
    precise float n = z - 12582912.0f;
    const float b = ggml_fma_f32(-n, asfloat(0x35bfbe8eu),
                    ggml_fma_f32(-n, asfloat(0x3f317200u), x));
    const uint e = asuint(z) << 23u;
    const float k = asfloat(e + 0x3f800000u);
    precise float u = b * b;
    precise float b1 = asfloat(0x3f7ffff6u) * b;
    const float j = ggml_fma_f32(ggml_fma_f32(
        ggml_fma_f32(asfloat(0x3c072010u), b, asfloat(0x3d2b9f17u)), u,
        ggml_fma_f32(asfloat(0x3e2aaf33u), b, asfloat(0x3efffedbu))), u, b1);
    if (abs(n) <= 126.0f) return ggml_fma_f32(j, k, k);
    const uint g = n <= 0.0f ? 0x82000000u : 0u;
    const float s1 = asfloat(g + 0x7f000000u);
    const float s2 = asfloat(e - g);
    precise float result = abs(n) > 192.0f ? s1 * s1 :
                          ggml_fma_f32(s2, j, s2) * s1;
    return result;
}

float ggml_rms_factor(double sum, uint size, float eps) {
    precise float mean = ggml_round_f32(sum / (double)size);
    precise float variance = mean + eps;
    precise double root = (double)sqrt(variance);
    root = 0.5 * (root + (double)variance / root);
    precise float rounded_root = ggml_round_f32(root);
    return ggml_round_f32(1.0 / (double)rounded_root);
}
