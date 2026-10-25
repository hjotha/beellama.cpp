// ggml_d3d12_swiglu_probe.hlsl — SWIGLU probe kernel for the D3D12 selftest
// (plan 006 C4 bounded probe): out = silu(gate) * up, using the same AVX2
// ggml_v_expf polynomial with explicit float mad() chains (dxc -Gis emits
// dx.op.tertiary FMad + precise fdiv). Not wired into any product graph.
//
// Root (gdn_root): u0 = output (param 1), u1 = gate (param 2), u2 = up (param 3).
// No root constants and no bounds guard: the host predicate requires
// nelements % 256 == 0 and the dispatch covers exactly one thread per element.

RWStructuredBuffer<float> ys : register(u0);
RWStructuredBuffer<float> gate : register(u1);
RWStructuredBuffer<float> upv : register(u2);

static const float C_r = asfloat(0x4b400000u); // 0x1.8p23f
static const float C_lo = 0x1.715476p+0f;
static const float C_c1 = 0x1.7f7d1cp-20f;
static const float C_c2 = 0x1.62e4p-1f;
static const float C_p0 = 0x1.0e4020p-7f;
static const float C_p1 = 0x1.573e2ep-5f;
static const float C_p2 = 0x1.555e66p-3f;
static const float C_p3 = 0x1.fffdb6p-2f;
static const float C_p4 = 0x1.ffffecp-1f;

// Exact fused multiply-add emulated in double: a*b is exact in double for
// float operands, the add is one double rounding, and the final cast is the
// second rounding (double-rounding ties are ~2^-29 per element). This makes
// the polynomial result independent of the driver's FMad fusion choice.
// Exact f32->f64 conversion including subnormal inputs: the GPU hardware
// conversion flushes subnormal f32 values while the CPU reference keeps them.
// mant * 2^-149 is exact in double (division by 2^23 and the 2^-126 constant
// are both exact power-of-two scalings).
double to_f64_cr(float x) {
    const float NORM_MIN = 1.17549435e-38f;
    const float ax = x < 0.0f ? -x : x;
    if (!(ax < NORM_MIN))
        return (double)x; // normal range, zero, NaN and infinity
    const uint bits = asuint(x);
    const uint mant = bits & 0x7FFFFFu;
    if (mant == 0)
        return (bits & 0x80000000u) ? -0.0 : 0.0;
    const double m = (double)mant / 8388608.0 * 1.17549435e-38;
    return (bits & 0x80000000u) ? -m : m;
}

// Exact f64->f32 conversion including the subnormal range: the hardware
// conversion flushes subnormal results while the CPU reference preserves
// them. abs(q) * 2^149 is exact (2^149 built from exact powers of two), the
// truncated mantissa and fraction come from an exact double->uint conversion,
// the fraction comparison is exact, and the bits are assembled directly.
float to_f32_cr(double q) {
    const float NORM_MIN = 1.17549435e-38f; // 2^-126
    if (!((q < (double)NORM_MIN) && (q > -(double)NORM_MIN)))
        return (float)q; // normal range, zero, NaN and infinity
    const double TWO149 = 1.0 / (1.17549435e-38 / 8388608.0);
    const bool neg = (q < 0.0) || (1.0 / q < 0.0);
    const double a = neg ? -q : q;
    const double t = a * TWO149;
    uint fi = uint(t); // toward zero = floor (t >= 0, t < 2^23), exact
    const double fr = t - (double)fi; // exact
    if (fr > 0.5 || (fr == 0.5 && (fi & 1u) != 0u))
        ++fi;
    if (fi >= 8388608u)
        return neg ? -NORM_MIN : NORM_MIN;
    return asfloat((neg ? 0x80000000u : 0u) | fi);
}

float fma_exact(float a, float b, float c) {
    return to_f32_cr(to_f64_cr(a) * to_f64_cr(b) + to_f64_cr(c));
}

float expf_avx2(float x) {
    const float r = C_r;
    const float z = fma_exact(x, C_lo, r);
    const float n = z - r;
    const float b = fma_exact(-n, C_c1, fma_exact(-n, C_c2, x));
    const uint e = asuint(z) << 23;
    const float k = asfloat(e + asuint(1.0f));
    const uint c = (abs(n) > 126.0f) ? 1u : 0u;
    const float u = b * b;
    const float j = fma_exact(fma_exact(fma_exact(C_p0, b, C_p1), u, fma_exact(C_p2, b, C_p3)), u, C_p4 * b);
    if (c == 0u)
        return fma_exact(j, k, k);
    const uint g = (n <= 0.0f) ? 0x82000000u : 0u;
    const float s1 = asfloat(g + 0x7f000000u);
    const float s2 = asfloat(e - g);
    const uint d = (abs(n) > 192.0f) ? 1u : 0u;
    return (d != 0u) ? (s1 * s1) : (fma_exact(s2, j, s2) * s1);
}

// Correctly-rounded float division. The RDNA2 fdiv is within ~1 ULP but not
// correctly rounded, while the CPU div_ps reference is; the double path is
// correctly rounded (the float rounding of an exact double quotient equals
// the correctly-rounded float quotient except for double-rounding ties, which
// this corpus checks). Non-finite divisors keep the direct path: x/inf must
// stay +/-0 and NaN must propagate, while the correction math below would
// turn inf into NaN.
float div_cr(float x, float d) {
    if (!(d < 3.0e38f))
        return x / d;
    return to_f32_cr(to_f64_cr(x) / to_f64_cr(d));
}

float silu_avx2(float x) {
    const float neg_x = 0.0f - x;
    const float e = expf_avx2(neg_x);
    return div_cr(x, 1.0f + e);
}

[numthreads(256, 1, 1)] void main(uint3 tid : SV_DispatchThreadID) {
    ys[tid.x] = to_f32_cr(to_f64_cr(silu_avx2(gate[tid.x])) * to_f64_cr(upv[tid.x]));
}
