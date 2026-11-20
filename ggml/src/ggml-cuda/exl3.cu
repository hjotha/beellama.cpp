#include "common.cuh"
#include "exl3.cuh"
#include "../../exl3/exl3.h"

#include <cuda_fp16.h>
#include <algorithm>

#ifdef GGML_USE_EXL3

// ============================================================================
// ExLlamaV3 Codebook & Tile Dequantization Primitives
// Verbatim from ExLlamaV3 (util.cuh, ptx.cuh, ptx_portable.cuh, codebook.cuh,
// exl3_dq.cuh, hadamard_inner.cuh, reconstruct.cu, exl3_gemv_kernel.cuh)
// ============================================================================

struct __align__(8) half4 {
    half2 x;
    half2 y;
    __device__ half4() = default;
    __device__ half4(half2 x_, half2 y_) : x(x_), y(y_) {}
    __device__ half4(half h0, half h1, half h2, half h3)
        : x(__halves2half2(h0, h1)), y(__halves2half2(h2, h3)) {}
};

union half2_uint32 {
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
    __device__ half2_uint32() : as_uint32(0) {}
};

union half_uint16 {
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
    __device__ half_uint16() : as_uint16(0) {}
};

template <typename T, int n>
struct Exl3Vec {
    T elems[n];
    __device__ __forceinline__ T & operator[](int i) { return elems[i]; }
    __device__ __forceinline__ const T & operator[](int i) const { return elems[i]; }
};

using FragA   = Exl3Vec<half2, 4>;
using FragB   = Exl3Vec<half2, 2>;
using FragC   = Exl3Vec<float, 4>;
using FragC_h = Exl3Vec<half2, 2>;

#if defined(GGML_USE_HIP)
    __device__ __forceinline__ uint32_t exl3_lop3_6a(uint32_t a, uint32_t b, uint32_t c) {
        return (a & b) ^ c;
    }
    __device__ __forceinline__ uint32_t exl3_shf_r_wrap(uint32_t lo, uint32_t hi, uint32_t n) {
        uint64_t v = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
        return static_cast<uint32_t>(v >> (n & 31));
    }
    __device__ __forceinline__ uint32_t exl3_bfe_u32_16(uint32_t a, uint32_t pos) {
        return (a >> pos) & 0xffffu;
    }
    #define EXL3_CB_LOP3(x) (x) = exl3_lop3_6a((x), 0x8fff8fffu, 0x3b603b60u)
    #define FSHF_IMM(dst, lo, hi, imm) (dst) = exl3_shf_r_wrap((lo), (hi), (imm))
    #define BFE16_IMM(dst, src, imm) (dst) = exl3_bfe_u32_16((src), (imm))
#else
    #define EXL3_CB_LOP3(x) asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x))
    #define FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
    #define BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))
#endif

// Decode two mul1 (cb 2) codebook entries from precomputed products x0 = idx0 * 0x83DCD12D, x1 = idx1 * 0x83DCD12D
__device__ __forceinline__ half2 decode_mul1_product_2(uint32_t x0, uint32_t x1) {
    const uint32_t acc = 0x6400u;  // 0x6400 -> 1024.0 .. 0x67FF -> 2047.0
    uint32_t sum0 = __dp4a(x0, 0x01010101u, acc);
    uint32_t sum1 = __dp4a(x1, 0x01010101u, acc);
    half2 k_inv_h2  = __half2half2(__ushort_as_half(0x1eee));
    half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));
    half_uint16 h0((uint16_t) sum0);
    half_uint16 h1((uint16_t) sum1);
    return __hfma2(__halves2half2(h0.as_half, h1.as_half), k_inv_h2, k_bias_h2);
}

// Decode two mcg (cb 1) codebook entries from precomputed products
__device__ __forceinline__ half2 decode_mcg_product_2(uint32_t x0, uint32_t x1) {
    EXL3_CB_LOP3(x0);
    EXL3_CB_LOP3(x1);
    half2_uint32 xu0(x0);
    half2_uint32 xu1(x1);
    half2 d0 = __lows2half2(xu0.as_half2, xu1.as_half2);
    half2 d1 = __highs2half2(xu0.as_half2, xu1.as_half2);
    return __hadd2(d0, d1);
}

template <int cb>
__device__ __forceinline__ half decode_3inst(uint32_t x) {
    if constexpr (cb == 0) {
        x *= 89226354u;
        x += 64248484u;
        EXL3_CB_LOP3(x);
        half2_uint32 xu(x);
        return __hadd(__low2half(xu.as_half2), __high2half(xu.as_half2));
    }
    if constexpr (cb == 1) {
        x *= 0xCBAC1FEDu;
        EXL3_CB_LOP3(x);
        half2_uint32 xu(x);
        return __hadd(__low2half(xu.as_half2), __high2half(xu.as_half2));
    }
    if constexpr (cb == 2) {
        x *= 0x83DCD12Du;
        const uint32_t acc = 0x6400u;
        uint32_t sum = __dp4a(x, 0x01010101u, acc);
        const __half k_inv_h  = __ushort_as_half(0x1eee);
        const __half k_bias_h = __ushort_as_half(0xc931);
        half_uint16 h((uint16_t) sum);
        return __hfma(h.as_half, k_inv_h, k_bias_h);
    }
}

template <int cb>
__device__ __forceinline__ half2 decode_3inst_2(uint32_t x0, uint32_t x1) {
    if constexpr (cb == 0) {
        x0 *= 89226354u;
        x1 *= 89226354u;
        x0 += 64248484u;
        x1 += 64248484u;
        EXL3_CB_LOP3(x0);
        EXL3_CB_LOP3(x1);
        half2_uint32 xu0(x0);
        half2_uint32 xu1(x1);
        half2 d0 = __lows2half2(xu0.as_half2, xu1.as_half2);
        half2 d1 = __highs2half2(xu0.as_half2, xu1.as_half2);
        return __hadd2(d0, d1);
    }
    if constexpr (cb == 1) {
        x0 *= 0xCBAC1FEDu;
        x1 *= 0xCBAC1FEDu;
        return decode_mcg_product_2(x0, x1);
    }
    if constexpr (cb == 2) {
        x0 *= 0x83DCD12Du;
        x1 *= 0x83DCD12Du;
        return decode_mul1_product_2(x0, x1);
    }
}

// Verbatim from exl3_dq.cuh
__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift) {
    uint64_t merged = ((uint64_t) a << 32) | (uint64_t) b;
    return (uint32_t) (merged >> shift);
}

template <int bits, int cb>
__device__ __forceinline__ void dq4(const uint32_t * ptr, int t_offset, FragB & frag) {
    int b0 = (t_offset + 257) * bits - 16;
    int b1 = b0 + 3 * bits;
    int b2 = b1 + 16;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;

    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i2 % (bits * 256 / 32)];
    uint32_t w3 = fshift(b, a, s2)            & 0xffff;
    uint32_t w2 = fshift(b, a, s2 + bits)     & 0xffff;
    uint32_t w1 = fshift(b, a, s2 + bits * 2) & 0xffff;
    uint32_t w0 = fshift(b, a, s2 + bits * 3) & 0xffff;
    frag[0] = decode_3inst_2<cb>(w0, w1);
    frag[1] = decode_3inst_2<cb>(w2, w3);
}

template <int bits, int cb>
__device__ __forceinline__ void dq2x2(const uint32_t * ptr, int t_offset, FragB & frag) {
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int b0 = (t_offset + 2 * i + 257) * bits - 16;
        int b1 = b0 + 1 * bits;
        int b2 = b1 + 16;
        int i0 = b0 / 32;
        int i2 = (b2 - 1) / 32;
        int s2 = (i2 + 1) * 32 - b2;

        uint32_t a = ptr[i0 % (bits * 256 / 32)];
        uint32_t b = ptr[i2 % (bits * 256 / 32)];
        uint32_t w1 = fshift(b, a, s2)        & 0xffff;
        uint32_t w0 = fshift(b, a, s2 + bits) & 0xffff;
        frag[i] = decode_3inst_2<cb>(w0, w1);
    }
}

template <int bits, int cb, int align>
__device__ __forceinline__ void dq8(const uint32_t * ptr, int t_offset, FragB & frag0, FragB & frag1) {
    int b1 = (t_offset + 257) * bits;
    int b0 = b1 - 16;
    int b2 = b1 + bits * 7;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;

    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i2 % (bits * 256 / 32)];
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    if constexpr (align == 1) {
        w7 = fshift(b, a, s2);
        w6 = fshift(b, a, s2 + bits);
        w5 = fshift(b, a, s2 + bits * 2);
        w4 = fshift(b, a, s2 + bits * 3);
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = fshift(b, a, s2 + bits * 5);
        w1 = fshift(b, a, s2 + bits * 6);
        w0 = fshift(b, a, s2 + bits * 7);
    }
    if constexpr (align == 2) {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = fshift(b, a, s2 + bits * 2);
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = fshift(b, a, s2 + bits * 6);
        w0 = w1 >> bits;
    }
    if constexpr (align == 4) {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
    }
    if constexpr (align == 8) {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = w4 >> bits;
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
    }
    frag0[0] = decode_3inst_2<cb>(w0 & 0xffff, w1 & 0xffff);
    frag0[1] = decode_3inst_2<cb>(w2 & 0xffff, w3 & 0xffff);
    frag1[0] = decode_3inst_2<cb>(w4 & 0xffff, w5 & 0xffff);
    frag1[1] = decode_3inst_2<cb>(w6 & 0xffff, w7 & 0xffff);
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_4bits(const uint32_t * ptr, int t_offset, FragB & frag0, FragB & frag1) {
    uint32_t i0, i1, a, b, s, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 3;
    i0 = (i1 + 31) & 31;
    a = ptr[i0];
    b = ptr[i1];
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_2bits(const uint32_t * ptr, int t_offset, FragB & frag0, FragB & frag1) {
    uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 4;
    i0 = (i1 + 15) & 15;
    a = ptr[i0];
    b = ptr[i1];
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2);
    BFE16_IMM(w5, b, 4);
    BFE16_IMM(w4, b, 6);
    BFE16_IMM(w3, b, 8);
    BFE16_IMM(w2, b, 10);
    BFE16_IMM(w1, b, 12);
    BFE16_IMM(w0, b, 14);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_1bit(const uint32_t * ptr, int t_offset, FragB & frag0, FragB & frag1) {
    uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 5;
    i0 = (i1 + 7) & 7;
    a = ptr[i0];
    b = ptr[i1];
    b = fshift(b, a, ((~t_offset) & 24));
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 1);
    BFE16_IMM(w5, b, 2);
    BFE16_IMM(w4, b, 3);
    BFE16_IMM(w3, b, 4);
    BFE16_IMM(w2, b, 5);
    BFE16_IMM(w1, b, 6);
    BFE16_IMM(w0, b, 7);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int KA, int cb>
__device__ __forceinline__ void dq8_half(const uint32_t * ptr, int t_offset, FragB & frag0, FragB & frag1) {
    constexpr int bits2 = 2 * KA + 1;
    constexpr int words = 4 * bits2;
    constexpr int gspan = 18 + 3 * KA;
    const int e7 = ((t_offset >> 1) + 4) * bits2 + 128 * bits2;
    const int e3 = e7 - 2 * bits2;
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    {
        int hi = (e7 - 1) / 32, lo = (e7 - gspan) / 32, s = (hi + 1) * 32 - e7;
        uint32_t a = ptr[lo % words], b = ptr[hi % words];
        w7 = fshift(b, a, s); w6 = w7 >> (KA + 1); w5 = w6 >> KA; w4 = w5 >> (KA + 1);
    }
    {
        int hi = (e3 - 1) / 32, lo = (e3 - gspan) / 32, s = (hi + 1) * 32 - e3;
        uint32_t a = ptr[lo % words], b = ptr[hi % words];
        w3 = fshift(b, a, s); w2 = w3 >> (KA + 1); w1 = w2 >> KA; w0 = w1 >> (KA + 1);
    }
    frag0[0] = decode_3inst_2<cb>(w0 & 0xffff, w1 & 0xffff);
    frag0[1] = decode_3inst_2<cb>(w2 & 0xffff, w3 & 0xffff);
    frag1[0] = decode_3inst_2<cb>(w4 & 0xffff, w5 & 0xffff);
    frag1[1] = decode_3inst_2<cb>(w6 & 0xffff, w7 & 0xffff);
}

template <int bits, int cb, bool half_k = false>
__device__ __forceinline__ void dq_dispatch(const uint32_t * ptr, int idx, FragB & frag0, FragB & frag1) {
    if constexpr (half_k) {
        dq8_half<bits, cb>(ptr, idx, frag0, frag1);
    } else if constexpr (bits == 1) {
        dq8_aligned_1bit<cb>(ptr, idx, frag0, frag1);
    } else if constexpr (bits == 2) {
        dq8_aligned_2bits<cb>(ptr, idx, frag0, frag1);
    } else if constexpr (bits == 3) {
        dq8<bits, cb, 4>(ptr, idx, frag0, frag1);
    } else if constexpr (bits == 4) {
        dq8_aligned_4bits<cb>(ptr, idx, frag0, frag1);
    } else if constexpr (bits == 5) {
        dq4<bits, cb>(ptr, idx, frag0);
        dq4<bits, cb>(ptr, idx + 4, frag1);
    } else if constexpr (bits == 6) {
        dq4<bits, cb>(ptr, idx, frag0);
        dq4<bits, cb>(ptr, idx + 4, frag1);
    } else if constexpr (bits == 7) {
        dq2x2<bits, cb>(ptr, idx, frag0);
        dq2x2<bits, cb>(ptr, idx + 4, frag1);
    } else if constexpr (bits == 8) {
        dq4<bits, cb>(ptr, idx, frag0);
        dq4<bits, cb>(ptr, idx + 4, frag1);
    }
}

// Verbatim from exl3_gemv_kernel.cuh
namespace exl3_gemv_ns {

__device__ __forceinline__ void mma_ab_f32(const FragB & a01, const FragB & a23, const FragB & b, float2 & c) {
#if __CUDA_ARCH__ >= 800
    const uint32_t * a0 = reinterpret_cast<const uint32_t *>(&a01);
    const uint32_t * a1 = reinterpret_cast<const uint32_t *>(&a23);
    const uint32_t * bb = reinterpret_cast<const uint32_t *>(&b);
    float c2 = 0.0f, c3 = 0.0f;
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c.x), "+f"(c.y), "+f"(c2), "+f"(c3)
        : "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]),
          "r"(bb[0]), "r"(bb[1])
    );
#else
    GGML_UNUSED(a01); GGML_UNUSED(a23); GGML_UNUSED(b); GGML_UNUSED(c);
#endif
}

__device__ __forceinline__ half2 decode_pair_cb2_dp4a_(uint32_t x0, uint32_t x1) {
    x0 *= 0x83DCD12Du;
    x1 *= 0x83DCD12Du;
    uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
    uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
    half2 k_inv_h2  = __half2half2(__ushort_as_half(0x1eee));
    half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));
    return __hfma2(
        __halves2half2(__ushort_as_half((uint16_t) sum0), __ushort_as_half((uint16_t) sum1)),
        k_inv_h2,
        k_bias_h2);
}

template <int cb>
__device__ __forceinline__ void decode8(
        uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
        uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7,
        FragB & f0, FragB & f1) {
    if constexpr (cb == GGML_EXL3_CODEBOOK_MUL1) {
        f0[0] = decode_pair_cb2_dp4a_(w0, w1);
        f0[1] = decode_pair_cb2_dp4a_(w2, w3);
        f1[0] = decode_pair_cb2_dp4a_(w4, w5);
        f1[1] = decode_pair_cb2_dp4a_(w6, w7);
    } else {
        f0[0] = decode_3inst_2<cb>(w0, w1);
        f0[1] = decode_3inst_2<cb>(w2, w3);
        f1[0] = decode_3inst_2<cb>(w4, w5);
        f1[1] = decode_3inst_2<cb>(w6, w7);
    }
}

template <int cb>
__device__ __forceinline__ void dq8_regs_4bits(uint32_t a, uint32_t b, FragB & f0, FragB & f1) {
    uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
    decode8<cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

template <int cb>
__device__ __forceinline__ void dq8_regs_2bits(uint32_t a, uint32_t b, int t_offset, FragB & f0, FragB & f1) {
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2);
    BFE16_IMM(w5, b, 4);
    BFE16_IMM(w4, b, 6);
    BFE16_IMM(w3, b, 8);
    BFE16_IMM(w2, b, 10);
    BFE16_IMM(w1, b, 12);
    BFE16_IMM(w0, b, 14);
    decode8<cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

template <int cb>
__device__ __forceinline__ void dq8_regs_3bits(uint32_t a, uint32_t b, int s2, FragB & f0, FragB & f1) {
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w7 = fshift(b, a, s2);
    w6 = w7 >> 3;
    w5 = w6 >> 3;
    w4 = w5 >> 3;
    w3 = fshift(b, a, s2 + 12);
    w2 = w3 >> 3;
    w1 = w2 >> 3;
    w0 = w1 >> 3;
    decode8<cb>(w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff,
                w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff, f0, f1);
}

template <int KA, int cb>
__device__ __forceinline__ void dq8_regs_half(
        uint32_t a7, uint32_t b7, int s7,
        uint32_t a3, uint32_t b3, int s3,
        FragB & f0, FragB & f1) {
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w7 = fshift(b7, a7, s7);
    w6 = w7 >> (KA + 1);
    w5 = w6 >> KA;
    w4 = w5 >> (KA + 1);
    w3 = fshift(b3, a3, s3);
    w2 = w3 >> (KA + 1);
    w1 = w2 >> KA;
    w0 = w1 >> (KA + 1);
    decode8<cb>(w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff,
                w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff, f0, f1);
}

} // namespace exl3_gemv_ns

// ============================================================================
// ExLlamaV3 Orthonormal 128-point Hadamard Transform (hadamard_inner.cuh)
// ============================================================================

static constexpr float EXL3_HAD_SCALE = 0.08838834764831845f;

__device__ __forceinline__ void shuffle_had_f4x32(float & h0, float & h1, float & h2, float & h3, int lane_id) {
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        uint64_t h01 = (uint64_t) __float_as_uint(h0) | (((uint64_t) __float_as_uint(h1)) << 32);
        uint64_t h23 = (uint64_t) __float_as_uint(h2) | (((uint64_t) __float_as_uint(h3)) << 32);
        uint64_t h01o = __shfl_xor_sync(0xffffffff, h01, i);
        uint64_t h23o = __shfl_xor_sync(0xffffffff, h23, i);
        if (lane_id & i) { h0 = -h0; h1 = -h1; h2 = -h2; h3 = -h3; }
        h0 += __uint_as_float((uint32_t) h01o);
        h1 += __uint_as_float((uint32_t) (h01o >> 32));
        h2 += __uint_as_float((uint32_t) h23o);
        h3 += __uint_as_float((uint32_t) (h23o >> 32));
    }
}

__device__ __forceinline__ void exl3_had128_r(float & v0, float & v1, float & v2, float & v3, int t) {
    float h0 = v0 + v1 + v2 + v3;
    float h1 = v0 - v1 + v2 - v3;
    float h2 = v0 + v1 - v2 - v3;
    float h3 = v0 - v1 - v2 + v3;
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v0 = h0 * EXL3_HAD_SCALE;
    v1 = h1 * EXL3_HAD_SCALE;
    v2 = h2 * EXL3_HAD_SCALE;
    v3 = h3 * EXL3_HAD_SCALE;
}

__global__ void k_exl3_had_in_f16(
        const char * __restrict__ x_base,
        const half * __restrict__ suh,
        half * __restrict__ xh_out,
        int k,
        int nrows,
        size_t x_nb1,
        size_t x_nb2,
        size_t x_nb3,
        int64_t x_ne1,
        int64_t x_ne2) {
    const int chunks_per_row = k >> 7;
    const int total_chunks = nrows * chunks_per_row;
    const int global_warp = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (global_warp >= total_chunks) return;

    const int lane = threadIdx.x & 31;
    const int row  = global_warp / chunks_per_row;
    const int c128 = (global_warp % chunks_per_row) << 7;
    const int col  = c128 + (lane << 2);

    const int64_t i1 = row % x_ne1;
    const int64_t r  = row / x_ne1;
    const int64_t i2 = r % x_ne2;
    const int64_t i3 = r / x_ne2;
    const float * __restrict__ x_row =
        (const float *) (x_base + (size_t) i1 * x_nb1 + (size_t) i2 * x_nb2 + (size_t) i3 * x_nb3);

    float v0 = x_row[col + 0];
    float v1 = x_row[col + 1];
    float v2 = x_row[col + 2];
    float v3 = x_row[col + 3];
    half4 sv = *reinterpret_cast<const half4 *>(suh + col);

    v0 *= __half2float(sv.x.x);
    v1 *= __half2float(sv.x.y);
    v2 *= __half2float(sv.y.x);
    v3 *= __half2float(sv.y.y);

    exl3_had128_r(v0, v1, v2, v3, lane);

    half4 out_h;
    out_h.x = __floats2half2_rn(v0, v1);
    out_h.y = __floats2half2_rn(v2, v3);
    *reinterpret_cast<half4 *>(xh_out + (int64_t) row * k + col) = out_h;
}

__global__ void k_exl3_had_out_f16_to_f32(
        const half * __restrict__ src_c,
        const half * __restrict__ svh,
        float * __restrict__ dst,
        int n,
        int n0,
        int ns,
        int nrows,
        bool swiglu) {
    const int chunks_per_row = ns >> 7;
    const int total_chunks = nrows * chunks_per_row;
    const int global_warp = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (global_warp >= total_chunks) return;

    const int lane = threadIdx.x & 31;
    const int row  = global_warp / chunks_per_row;
    const int c128 = (global_warp % chunks_per_row) << 7;
    const int local_col = c128 + (lane << 2);
    const int col  = n0 + local_col;

    half4 yh = *reinterpret_cast<const half4 *>(src_c + (int64_t) row * ns + local_col);
    half4 sv = *reinterpret_cast<const half4 *>(svh + col);

    float v0 = __half2float(yh.x.x);
    float v1 = __half2float(yh.x.y);
    float v2 = __half2float(yh.y.x);
    float v3 = __half2float(yh.y.y);

    exl3_had128_r(v0, v1, v2, v3, lane);

    v0 *= __half2float(sv.x.x);
    v1 *= __half2float(sv.x.y);
    v2 *= __half2float(sv.y.x);
    v3 *= __half2float(sv.y.y);

    if (swiglu) {
        v0 = v0 / (1.0f + expf(-v0));
        v1 = v1 / (1.0f + expf(-v1));
        v2 = v2 / (1.0f + expf(-v2));
        v3 = v3 / (1.0f + expf(-v3));
    }

    *reinterpret_cast<float4 *>(dst + (int64_t) row * n + col) = make_float4(v0, v1, v2, v3);
}

__global__ void k_exl3_had_out_inplace_slab_f32(
        float * __restrict__ dst,
        const half * __restrict__ svh,
        int n,
        int n0,
        int ns,
        int nrows,
        bool swiglu) {
    const int chunks_per_row = ns >> 7;
    const int total_chunks = nrows * chunks_per_row;
    const int global_warp = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (global_warp >= total_chunks) return;

    const int lane = threadIdx.x & 31;
    const int row  = global_warp / chunks_per_row;
    const int c128 = (global_warp % chunks_per_row) << 7;
    const int col  = n0 + c128 + (lane << 2);

    float * __restrict__ dst_ptr = dst + (int64_t) row * n + col;
    float4 yv = *reinterpret_cast<const float4 *>(dst_ptr);
    half4  sv = *reinterpret_cast<const half4 *>(svh + col);

    exl3_had128_r(yv.x, yv.y, yv.z, yv.w, lane);

    yv.x *= __half2float(sv.x.x);
    yv.y *= __half2float(sv.x.y);
    yv.z *= __half2float(sv.y.x);
    yv.w *= __half2float(sv.y.y);

    if (swiglu) {
        yv.x = yv.x / (1.0f + expf(-yv.x));
        yv.y = yv.y / (1.0f + expf(-yv.y));
        yv.z = yv.z / (1.0f + expf(-yv.z));
        yv.w = yv.w / (1.0f + expf(-yv.w));
    }

    *reinterpret_cast<float4 *>(dst_ptr) = yv;
}

// ============================================================================
// ExLlamaV3 Reconstruct Kernel (verbatim from reconstruct.cu)
// Grid: (ns / 128, k / 16, 1), Block: 256
// ============================================================================

template <int K, int cb, bool HALF = false>
__global__ void __launch_bounds__(256)
reconstruct_kernel(
        const uint16_t * __restrict__ g_packed,
        half * __restrict__ g_unpacked,
        int packed_blocks_n,
        int packed_n_offset) {
    constexpr int packed_size = 16 * K + (HALF ? 8 : 0);

    int t = threadIdx.x;
    int lane_id = t & 31;
    int warp_id = t >> 5;
    int k = blockIdx.y;
    int n = blockIdx.x * 8;
    int tiles_n = gridDim.x;
    int out_blocks_n = tiles_n * 8;

    __shared__ __align__(16) uint32_t s_packed[8][packed_size / 2];
    g_packed += ((int64_t) k * packed_blocks_n + packed_n_offset + n) * packed_size;
    if (t < packed_size) {
        ((int4 *) s_packed)[t] = ((const int4 *) g_packed)[t];
    }
    __syncthreads();

    FragB frag[2];
    dq_dispatch<K, cb, HALF>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    __shared__ __align__(16) half2 tile[16][8][8];

    half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
    half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
    half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
    half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

    if (!(lane_id & 4)) {
        half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
        half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
        half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
        half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
        half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
        half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
        half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
        half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
        int r0 = (lane_id % 4) * 2;
        int r1 = r0 + 1;
        int r2 = r0 + 8;
        int r3 = r0 + 9;
        int c0 = lane_id / 8;
        int c1 = c0 + 4;
        tile[r0][warp_id][c0] = m0;
        tile[r1][warp_id][c0] = m1;
        tile[r2][warp_id][c0] = m2;
        tile[r3][warp_id][c0] = m3;
        tile[r0][warp_id][c1] = m4;
        tile[r1][warp_id][c1] = m5;
        tile[r2][warp_id][c1] = m6;
        tile[r3][warp_id][c1] = m7;
    }
    __syncthreads();

    int r = t >> 4;
    int c = t & 15;
    int4 * tile_int4 = reinterpret_cast<int4 *>(tile);
    int4 * out_int4 = ((int4 *) g_unpacked) + (int64_t) (k * 16 + r) * 2 * out_blocks_n + n * 2 + c;
    *out_int4 = tile_int4[t];
}

template <int CB>
static bool exl3_launch_reconstruct_cb(
        int tile_words,
        dim3 grid,
        cudaStream_t stream,
        const uint16_t * trellis,
        half * w_slab,
        int n_full,
        int n0) {
    const int packed_blocks_n = n_full >> 4;
    const int packed_n_offset = n0 >> 4;
    switch (tile_words) {
        case 16:  reconstruct_kernel<1, CB, false><<<grid, 256, 0, stream>>>(trellis, w_slab, packed_blocks_n, packed_n_offset); return true;
        case 24:  reconstruct_kernel<1, CB, true ><<<grid, 256, 0, stream>>>(trellis, w_slab, packed_blocks_n, packed_n_offset); return true;
        case 32:  reconstruct_kernel<2, CB, false><<<grid, 256, 0, stream>>>(trellis, w_slab, packed_blocks_n, packed_n_offset); return true;
        case 40:  reconstruct_kernel<2, CB, true ><<<grid, 256, 0, stream>>>(trellis, w_slab, packed_blocks_n, packed_n_offset); return true;
        case 48:  reconstruct_kernel<3, CB, false><<<grid, 256, 0, stream>>>(trellis, w_slab, packed_blocks_n, packed_n_offset); return true;
        case 56:  reconstruct_kernel<3, CB, true ><<<grid, 256, 0, stream>>>(trellis, w_slab, packed_blocks_n, packed_n_offset); return true;
        case 64:  reconstruct_kernel<4, CB, false><<<grid, 256, 0, stream>>>(trellis, w_slab, packed_blocks_n, packed_n_offset); return true;
        case 72:  reconstruct_kernel<4, CB, true ><<<grid, 256, 0, stream>>>(trellis, w_slab, packed_blocks_n, packed_n_offset); return true;
        case 80:  reconstruct_kernel<5, CB, false><<<grid, 256, 0, stream>>>(trellis, w_slab, packed_blocks_n, packed_n_offset); return true;
        case 88:  reconstruct_kernel<5, CB, true ><<<grid, 256, 0, stream>>>(trellis, w_slab, packed_blocks_n, packed_n_offset); return true;
        case 96:  reconstruct_kernel<6, CB, false><<<grid, 256, 0, stream>>>(trellis, w_slab, packed_blocks_n, packed_n_offset); return true;
        case 104: reconstruct_kernel<6, CB, true ><<<grid, 256, 0, stream>>>(trellis, w_slab, packed_blocks_n, packed_n_offset); return true;
        case 112: reconstruct_kernel<7, CB, false><<<grid, 256, 0, stream>>>(trellis, w_slab, packed_blocks_n, packed_n_offset); return true;
        case 120: reconstruct_kernel<7, CB, true ><<<grid, 256, 0, stream>>>(trellis, w_slab, packed_blocks_n, packed_n_offset); return true;
        case 128: reconstruct_kernel<8, CB, false><<<grid, 256, 0, stream>>>(trellis, w_slab, packed_blocks_n, packed_n_offset); return true;
        default: return false;
    }
}

static bool exl3_launch_reconstruct(
        int tile_words,
        int cb,
        dim3 grid,
        cudaStream_t stream,
        const uint16_t * trellis,
        half * w_slab,
        int n_full,
        int n0) {
    if (cb == GGML_EXL3_CODEBOOK_MCG) {
        return exl3_launch_reconstruct_cb<GGML_EXL3_CODEBOOK_MCG>(tile_words, grid, stream, trellis, w_slab, n_full, n0);
    }
    if (cb == GGML_EXL3_CODEBOOK_MUL1) {
        return exl3_launch_reconstruct_cb<GGML_EXL3_CODEBOOK_MUL1>(tile_words, grid, stream, trellis, w_slab, n_full, n0);
    }
    return exl3_launch_reconstruct_cb<GGML_EXL3_CODEBOOK_3INST>(tile_words, grid, stream, trellis, w_slab, n_full, n0);
}

// ============================================================================
// EXL3_GET_ROWS Kernel (Bit-identical to ggml_exl3_decode_embd_row)
// Grid: (n_rows, n_groups, 1), Block: 256 (1 thread per ring element)
// ============================================================================

__global__ void __launch_bounds__(256)
k_exl3_get_rows(
        const char * __restrict__ table_base,
        const char * __restrict__ ids_base,
        const half * __restrict__ signs,
        char * __restrict__ dst_base,
        int64_t n_vocab,
        int K,
        int64_t rw,
        int64_t n_out,
        int64_t n_rows,
        size_t table_nb1,
        size_t ids_nb0,
        size_t dst_nb1) {
    const int64_t row_idx = blockIdx.x;
    const int64_t g       = blockIdx.y;
    const int     i       = threadIdx.x;
    if (row_idx >= n_rows || g * GGML_EXL3_GROUP_DIM >= n_out) return;

    const int32_t id = *(const int32_t *) (ids_base + (size_t) row_idx * ids_nb0);
    if (id < 0 || id >= n_vocab) return;

    const int16_t  * ring   = (const int16_t *) (table_base + (size_t) id * table_nb1) + g * rw;
    const uint16_t * stream = (const uint16_t *) (ring + 1);
    const float      scale  = __half2float(*reinterpret_cast<const half *>(&ring[0]));
    const int        words  = (GGML_EXL3_GROUP_DIM * K) >> 4;

    __shared__ uint32_t s_sym[256];
    __shared__ float    s_x[256];

    const int bit = i * K;
    const int w0  = bit >> 4;
    const int w1  = (w0 + 1 < words) ? (w0 + 1) : 0;
    const uint32_t window = (uint32_t) stream[w0] | ((uint32_t) stream[w1] << 16);
    s_sym[i] = (window >> (bit & 15)) & ((1u << K) - 1u);
    __syncthreads();

    const int nj = (15 + K) / K;
    uint32_t state = 0;
    for (int j = 0; j < nj; ++j) {
        state |= s_sym[(i - j) & 255] << (j * K);
    }

    s_x[i] = __fmul_rn(__half2float(decode_3inst<GGML_EXL3_CODEBOOK_MUL1>((uint16_t) state)), scale);
    __syncthreads();

    for (int h = 1; h < 256; h <<= 1) {
        float val = s_x[i];
        float other = s_x[i ^ h];
        __syncthreads();
        s_x[i] = (i & h) ? __fsub_rn(other, val) : __fadd_rn(val, other);
        __syncthreads();
    }

    const int64_t out_col = g * GGML_EXL3_GROUP_DIM + i;
    if (out_col < n_out) {
        float * dst_row = (float *) (dst_base + (size_t) row_idx * dst_nb1);
        const float rotated = __fmul_rn(s_x[i], 0.0625f);
        dst_row[out_col] = __fmul_rn(rotated, __half2float(signs[out_col]));
    }
}

// ============================================================================
// ExLlamaV3 4-Stage Register-Prefetch Tensor-Core GEMV Kernels (nrows <= 8)
// Verbatim from exl3_gemv_kernel.cuh & exl3_moe_coop_kernel.cuh
// ============================================================================

template <int TW, int CBV, int WK>
__device__ __forceinline__ void exl3_gemv_32_body(
        const uint16_t * __restrict__ trellis,
        const half * __restrict__ xh,
        float * __restrict__ dst,
        int k,
        int n,
        int nrows) {
    constexpr int bits = TW >> 4;
    constexpr bool HALF = (TW & 15) != 0;
    constexpr bool REG = (!HALF && (bits == 2 || bits == 3 || bits == 4)) ||
                         (HALF && (bits >= 1 && bits <= 3 && CBV == 2));
    constexpr int WNT = 2;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = HALF ? 4 * (2 * bits + 1) : 8 * bits;
    constexpr bool TWO_PER_LOAD = REG && (HALF ? bits == 1 : bits == 2);
    constexpr int GWORDS = WNT * TWORDS;
    constexpr int LOADS = REG ? (TWO_PER_LOAD ? WNT / 2 : WNT) : ((GWORDS + 31) / 32);
    constexpr int LSTRIDE = REG ? (TWO_PER_LOAD ? 2 * TWORDS : (TWORDS < 32 ? TWORDS : 32)) : 32;
    constexpr int PF = 4;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int ntiles = n >> 4;
    const int kslices = k >> 4;
    const int group = blockIdx.x;
    if (group * WNT >= ntiles) return;

    const int chunk = (kslices + WK - 1) / WK;
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));

    const uint32_t * B32 = (const uint32_t *) trellis;
    const size_t slice_stride = (size_t) ntiles * TWORDS;
    const half2 * A2 = (const half2 *) xh;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    const int r0 = lane >> 2;
    const bool r0_ok = r0 < nrows;
    const size_t a_row0 = r0_ok ? (size_t) r0 * (k >> 1) : 0;

    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    [[maybe_unused]] uint32_t x_pack = 0;
    if constexpr (REG && HALF) {
        constexpr int bits2 = 2 * bits + 1;
        constexpr int gspan = 18 + 3 * bits;
        const int t_offset = lane << 3;
        const int e7 = ((t_offset >> 1) + 4) * bits2 + 128 * bits2;
        const int e3 = e7 - 2 * bits2;
        const uint32_t hi7 = ((e7 - 1) / 32) % TWORDS, lo7 = ((e7 - gspan) / 32) % TWORDS, s7 = ((e7 - 1) / 32 + 1) * 32 - e7;
        const uint32_t hi3 = ((e3 - 1) / 32) % TWORDS, lo3 = ((e3 - gspan) / 32) % TWORDS, s3 = ((e3 - 1) / 32 + 1) * 32 - e3;
        x_pack = lo7 | (hi7 << 5) | (s7 << 10) | (lo3 << 15) | (hi3 << 20) | (s3 << 25);
    } else if constexpr (REG && bits == 2) {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    } else if constexpr (REG && bits == 3) {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }
    #define EXL3_XP(idx) ((x_pack >> (5 * (idx))) & 31u)

    __shared__ float sh_red[WK][8][COLS];
    [[maybe_unused]] __shared__ uint32_t sh_stage[REG ? 1 : WK][REG ? 1 : WNT * 64];

    const uint32_t * bp = B32 + (size_t) ks0 * slice_stride + group * GWORDS + lane;
    auto ld_b = [&](int i, int l) -> uint32_t {
        if constexpr (REG && LSTRIDE < 32) {
            return lane < LSTRIDE ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0u;
        } else if constexpr (!REG) {
            return (l * 32 + lane < GWORDS) ? __ldcs(bp + (size_t) i * slice_stride + l * 32) : 0u;
        } else {
            return __ldcs(bp + (size_t) i * slice_stride + l * 32);
        }
    };

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);

    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
            }

            if constexpr (!REG) {
                __syncwarp();
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    if (l * 32 + lane < GWORDS)
                        sh_stage[warp][l * 32 + lane] = bw[l];
                __syncwarp();
            }

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                if constexpr (!REG) {
                    dq_dispatch<bits, CBV, HALF>(&sh_stage[warp][t * TWORDS], lane << 3, f0, f1);
                } else if constexpr (HALF) {
                    const uint32_t w = TWO_PER_LOAD ? bw[t >> 1] : bw[t];
                    const int base = TWO_PER_LOAD ? (t & 1) * TWORDS : 0;
                    uint32_t a7 = __shfl_sync(0xffffffffu, w, base + EXL3_XP(0));
                    uint32_t b7 = __shfl_sync(0xffffffffu, w, base + EXL3_XP(1));
                    uint32_t a3 = __shfl_sync(0xffffffffu, w, base + EXL3_XP(3));
                    uint32_t b3 = __shfl_sync(0xffffffffu, w, base + EXL3_XP(4));
                    exl3_gemv_ns::dq8_regs_half<bits, CBV>(a7, b7, EXL3_XP(2), a3, b3, EXL3_XP(5), f0, f1);
                } else if constexpr (bits == 4) {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<CBV>(aw, bw[t], f0, f1);
                } else if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<CBV>(awv, bwv, lane << 3, f0, f1);
                } else {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<CBV>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_f32(a01, a23, f0, acc0[t][0]);
                exl3_gemv_ns::mma_ab_f32(a01, a23, f1, acc0[t][1]);
            }
        }
    }
    #undef EXL3_XP

    if (r0_ok) {
        const int c0 = 2 * (lane & 3);
        #pragma unroll
        for (int t = 0; t < WNT; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = t * 16 + f * 8 + c0;
                sh_red[warp][r0][col + 0] = acc0[t][f].x;
                sh_red[warp][r0][col + 1] = acc0[t][f].y;
            }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS * nrows; idx += WK * 32) {
        const int r = idx / COLS;
        const int c = idx % COLS;
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WK; ++j) sum += sh_red[j][r][c];
        dst[(int64_t) r * n + group * COLS + c] = sum;
    }
}

template <int TW, int CBV, int WK>
__global__ void __launch_bounds__(WK * 32)
k_exl3_gemv_32(
        const uint16_t * __restrict__ trellis,
        const half * __restrict__ xh,
        float * __restrict__ dst,
        int k,
        int n,
        int nrows) {
    exl3_gemv_32_body<TW, CBV, WK>(trellis, xh, dst, k, n, nrows);
}

// 128-column block Tensor-Core GEMV with FUSED 128-point output Hadamard + svh epilogue
// WN = 4 warps along N x WNT = 2 N-tiles/warp = 128 columns/block
// WKK = 4 warps along K -> 16 warps = 512 threads/block
template <int TW, int CBV>
__device__ __forceinline__ void exl3_gemv_128_fused_body(
        const uint16_t * __restrict__ trellis,
        const half * __restrict__ xh,
        const half * __restrict__ svh,
        float * __restrict__ dst,
        int k,
        int n,
        int nrows,
        bool swiglu) {
    constexpr int bits = TW >> 4;
    constexpr bool HALF = (TW & 15) != 0;
    constexpr bool REG = (!HALF && (bits == 2 || bits == 3 || bits == 4)) ||
                         (HALF && (bits >= 1 && bits <= 3 && CBV == 2));
    constexpr int WN = 4;
    constexpr int WNT = 2;
    constexpr int WKK = 4;
    constexpr int COLS = WNT * 16;
    constexpr int TCOLS = WN * COLS; // 128
    constexpr int TWORDS = HALF ? 4 * (2 * bits + 1) : 8 * bits;
    constexpr bool TWO_PER_LOAD = REG && (HALF ? bits == 1 : bits == 2);
    constexpr int GWORDS = WNT * TWORDS;
    constexpr int LOADS = REG ? (TWO_PER_LOAD ? WNT / 2 : WNT) : ((GWORDS + 31) / 32);
    constexpr int LSTRIDE = REG ? (TWO_PER_LOAD ? 2 * TWORDS : (TWORDS < 32 ? TWORDS : 32)) : 32;
    constexpr int PF = 4;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wn = warp % WN;
    const int wk = warp / WN;

    const int ntiles = n >> 4;
    const int kslices = k >> 4;
    const int group = blockIdx.x;
    if (group * (WN * WNT) >= ntiles) return;

    const int chunk = (kslices + WKK - 1) / WKK;
    const int ks0 = wk * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));

    const uint32_t * B32 = (const uint32_t *) trellis;
    const size_t slice_stride = (size_t) ntiles * TWORDS;
    const half2 * A2 = (const half2 *) xh;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    const int r0 = lane >> 2;
    const bool r0_ok = r0 < nrows;
    const size_t a_row0 = r0_ok ? (size_t) r0 * (k >> 1) : 0;

    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    [[maybe_unused]] uint32_t x_pack = 0;
    if constexpr (REG && HALF) {
        constexpr int bits2 = 2 * bits + 1;
        constexpr int gspan = 18 + 3 * bits;
        const int t_offset = lane << 3;
        const int e7 = ((t_offset >> 1) + 4) * bits2 + 128 * bits2;
        const int e3 = e7 - 2 * bits2;
        const uint32_t hi7 = ((e7 - 1) / 32) % TWORDS, lo7 = ((e7 - gspan) / 32) % TWORDS, s7 = ((e7 - 1) / 32 + 1) * 32 - e7;
        const uint32_t hi3 = ((e3 - 1) / 32) % TWORDS, lo3 = ((e3 - gspan) / 32) % TWORDS, s3 = ((e3 - 1) / 32 + 1) * 32 - e3;
        x_pack = lo7 | (hi7 << 5) | (s7 << 10) | (lo3 << 15) | (hi3 << 20) | (s3 << 25);
    } else if constexpr (REG && bits == 2) {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    } else if constexpr (REG && bits == 3) {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }
    #define EXL3_XP(idx) ((x_pack >> (5 * (idx))) & 31u)

    __shared__ __align__(16) union {
        uint32_t sh_stage[REG ? 1 : WN * WKK][REG ? 1 : WNT * 64];
        float s_red[WKK][8][TCOLS];
    } smem;

    const uint32_t * bp = B32 + (size_t) ks0 * slice_stride + (group * WN + wn) * GWORDS + lane;
    auto ld_b = [&](int i, int l) -> uint32_t {
        if constexpr (REG && LSTRIDE < 32) {
            return lane < LSTRIDE ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0u;
        } else if constexpr (!REG) {
            return (l * 32 + lane < GWORDS) ? __ldcs(bp + (size_t) i * slice_stride + l * 32) : 0u;
        } else {
            return __ldcs(bp + (size_t) i * slice_stride + l * 32);
        }
    };

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);

    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
            }

            if constexpr (!REG) {
                __syncwarp();
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    if (l * 32 + lane < GWORDS)
                        smem.sh_stage[warp][l * 32 + lane] = bw[l];
                __syncwarp();
            }

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                if constexpr (!REG) {
                    dq_dispatch<bits, CBV, HALF>(&smem.sh_stage[warp][t * TWORDS], lane << 3, f0, f1);
                } else if constexpr (HALF) {
                    const uint32_t w = TWO_PER_LOAD ? bw[t >> 1] : bw[t];
                    const int base = TWO_PER_LOAD ? (t & 1) * TWORDS : 0;
                    uint32_t a7 = __shfl_sync(0xffffffffu, w, base + EXL3_XP(0));
                    uint32_t b7 = __shfl_sync(0xffffffffu, w, base + EXL3_XP(1));
                    uint32_t a3 = __shfl_sync(0xffffffffu, w, base + EXL3_XP(3));
                    uint32_t b3 = __shfl_sync(0xffffffffu, w, base + EXL3_XP(4));
                    exl3_gemv_ns::dq8_regs_half<bits, CBV>(a7, b7, EXL3_XP(2), a3, b3, EXL3_XP(5), f0, f1);
                } else if constexpr (bits == 4) {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<CBV>(aw, bw[t], f0, f1);
                } else if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<CBV>(awv, bwv, lane << 3, f0, f1);
                } else {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<CBV>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_f32(a01, a23, f0, acc0[t][0]);
                exl3_gemv_ns::mma_ab_f32(a01, a23, f1, acc0[t][1]);
            }
        }
    }
    #undef EXL3_XP

    if (!REG) __syncthreads();

    if (r0_ok) {
        const int c0 = 2 * (lane & 3);
        #pragma unroll
        for (int t = 0; t < WNT; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = wn * COLS + t * 16 + f * 8 + c0;
                smem.s_red[wk][r0][col + 0] = acc0[t][f].x;
                smem.s_red[wk][r0][col + 1] = acc0[t][f].y;
            }
    }
    __syncthreads();

    if (warp < nrows) {
        const int m = warp;
        const int c4 = lane << 2;
        const int global_col = group * TCOLS + c4;

        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        #pragma unroll
        for (int w = 0; w < WKK; ++w) {
            v0 += smem.s_red[w][m][c4 + 0];
            v1 += smem.s_red[w][m][c4 + 1];
            v2 += smem.s_red[w][m][c4 + 2];
            v3 += smem.s_red[w][m][c4 + 3];
        }

        exl3_had128_r(v0, v1, v2, v3, lane);

        half4 sv = *reinterpret_cast<const half4 *>(svh + global_col);
        v0 *= __half2float(sv.x.x);
        v1 *= __half2float(sv.x.y);
        v2 *= __half2float(sv.y.x);
        v3 *= __half2float(sv.y.y);

        if (swiglu) {
            v0 = v0 / (1.0f + expf(-v0));
            v1 = v1 / (1.0f + expf(-v1));
            v2 = v2 / (1.0f + expf(-v2));
            v3 = v3 / (1.0f + expf(-v3));
        }

        *reinterpret_cast<float4 *>(dst + (int64_t) m * n + global_col) = make_float4(v0, v1, v2, v3);
    }
}

template <int TW, int CBV>
__global__ void __launch_bounds__(512)
k_exl3_gemv_128_fused(
        const uint16_t * __restrict__ trellis,
        const half * __restrict__ xh,
        const half * __restrict__ svh,
        float * __restrict__ dst,
        int k,
        int n,
        int nrows,
        bool swiglu) {
    exl3_gemv_128_fused_body<TW, CBV>(trellis, xh, svh, dst, k, n, nrows, swiglu);
}

template <int TW>
static bool exl3_launch_gemv_t(
        int cb,
        bool fused_128,
        dim3 grid,
        cudaStream_t stream,
        const uint16_t * trellis,
        const half * xh,
        const half * svh,
        float * dst,
        int k,
        int n,
        int nrows,
        bool swiglu) {
    if (fused_128) {
        if (cb == GGML_EXL3_CODEBOOK_MCG) {
            k_exl3_gemv_128_fused<TW, GGML_EXL3_CODEBOOK_MCG><<<grid, 512, 0, stream>>>(trellis, xh, svh, dst, k, n, nrows, swiglu);
        } else if (cb == GGML_EXL3_CODEBOOK_MUL1) {
            k_exl3_gemv_128_fused<TW, GGML_EXL3_CODEBOOK_MUL1><<<grid, 512, 0, stream>>>(trellis, xh, svh, dst, k, n, nrows, swiglu);
        } else {
            k_exl3_gemv_128_fused<TW, GGML_EXL3_CODEBOOK_3INST><<<grid, 512, 0, stream>>>(trellis, xh, svh, dst, k, n, nrows, swiglu);
        }
    } else {
        if (cb == GGML_EXL3_CODEBOOK_MCG) {
            k_exl3_gemv_32<TW, GGML_EXL3_CODEBOOK_MCG, 16><<<grid, 512, 0, stream>>>(trellis, xh, dst, k, n, nrows);
        } else if (cb == GGML_EXL3_CODEBOOK_MUL1) {
            k_exl3_gemv_32<TW, GGML_EXL3_CODEBOOK_MUL1, 16><<<grid, 512, 0, stream>>>(trellis, xh, dst, k, n, nrows);
        } else {
            k_exl3_gemv_32<TW, GGML_EXL3_CODEBOOK_3INST, 16><<<grid, 512, 0, stream>>>(trellis, xh, dst, k, n, nrows);
        }
    }
    return true;
}

static bool exl3_launch_gemv(
        int tile_words,
        int cb,
        bool fused_128,
        dim3 grid,
        cudaStream_t stream,
        const uint16_t * trellis,
        const half * xh,
        const half * svh,
        float * dst,
        int k,
        int n,
        int nrows,
        bool swiglu) {
    switch (tile_words) {
        case 16:  return exl3_launch_gemv_t<16 >(cb, fused_128, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 24:  return exl3_launch_gemv_t<24 >(cb, fused_128, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 32:  return exl3_launch_gemv_t<32 >(cb, fused_128, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 40:  return exl3_launch_gemv_t<40 >(cb, fused_128, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 48:  return exl3_launch_gemv_t<48 >(cb, fused_128, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 56:  return exl3_launch_gemv_t<56 >(cb, fused_128, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 64:  return exl3_launch_gemv_t<64 >(cb, fused_128, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 72:  return exl3_launch_gemv_t<72 >(cb, fused_128, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 80:  return exl3_launch_gemv_t<80 >(cb, fused_128, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 88:  return exl3_launch_gemv_t<88 >(cb, fused_128, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 96:  return exl3_launch_gemv_t<96 >(cb, fused_128, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 104: return exl3_launch_gemv_t<104>(cb, fused_128, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 112: return exl3_launch_gemv_t<112>(cb, fused_128, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 120: return exl3_launch_gemv_t<120>(cb, fused_128, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 128: return exl3_launch_gemv_t<128>(cb, fused_128, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        default: return false;
    }
}

// ============================================================================
// ExLlamaV3 Tensor Core MMA GEMM Kernels (8 < nrows <= 64, and MoE c > 8)
// ============================================================================

#if __CUDA_ARCH__ >= 800

static __device__ __forceinline__ void exl3_ldmatrix_x4_a(uint32_t reg[4], const half * smem_ptr) {
    const uint32_t addr = (uint32_t) __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(addr)
    );
}

static __device__ __forceinline__ void exl3_mma_m16n8k16(
        float c[4],
        const uint32_t a[4],
        const FragB & b) {
    const uint32_t * bp = reinterpret_cast<const uint32_t *>(&b);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(bp[0]), "r"(bp[1])
    );
}

#endif // __CUDA_ARCH__ >= 800

static constexpr int EXL3_MMA_STAGES = 3;
static constexpr int EXL3_MMA_BN     = 128;
static constexpr int EXL3_MMA_BK     = 32;
static constexpr int EXL3_MMA_STRIDE = 40;

template <int BM>
struct exl3_mma_stage {
    uint16_t trellis[2][8][128];
    half     a[BM][EXL3_MMA_STRIDE];
};

template <int BM>
union exl3_mma_smem {
    exl3_mma_stage<BM> pipe[EXL3_MMA_STAGES];
    float              c[BM][132];
};

template <int TW, int CBV, int BM>
__device__ __forceinline__ void exl3_mma_gemm_body(
        const uint16_t * __restrict__ trellis,
        const half * __restrict__ xh,
        const half * __restrict__ svh,
        float * __restrict__ dst,
        int k,
        int n,
        int nrows,
        int m_block,
        int n_block,
        bool swiglu) {
#if __CUDA_ARCH__ >= 800
    constexpr int K_BITS = TW >> 4;
    constexpr bool HALF = (TW & 15) != 0;
    constexpr int MT = BM / 16;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int m0 = m_block * BM;
    const int n0 = n_block * EXL3_MMA_BN;
    if (m0 >= nrows || n0 >= n) return;

    const int valid_m = min(BM, nrows - m0);
    const int nt      = n >> 4;
    const int n_tile0 = n0 >> 4;
    const int num_steps = k / EXL3_MMA_BK;

    __shared__ __align__(16) exl3_mma_smem<BM> smem;

    float acc[MT][2][4];
    #pragma unroll
    for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                acc[m][j][r] = 0.0f;

    auto load_stage = [&](int step, int stage) {
        const int kt0 = step * 2;
        const int k0  = step * EXL3_MMA_BK;

        constexpr int WORDS8 = 8 * TW;
        if constexpr ((TW & 7) == 0) {
            constexpr int VECS8 = WORDS8 / 8;
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const uint4 * g_base = reinterpret_cast<const uint4 *>(
                    trellis + ((int64_t) (kt0 + sub) * nt + n_tile0) * TW);
                for (int idx = tid; idx < VECS8; idx += 256) {
                    const int tn = idx / (TW / 8);
                    const int w  = (idx % (TW / 8)) * 8;
                    uint32_t s_addr = (uint32_t) __cvta_generic_to_shared(&smem.pipe[stage].trellis[sub][tn][w]);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s_addr), "l"(g_base + idx));
                }
            }
        } else {
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const uint16_t * g_base = trellis + ((int64_t) (kt0 + sub) * nt + n_tile0) * TW;
                for (int idx = tid; idx < WORDS8; idx += 256) {
                    const int tn = idx / TW;
                    const int w  = idx % TW;
                    smem.pipe[stage].trellis[sub][tn][w] = g_base[idx];
                }
            }
        }

        constexpr int A_VEC_TOTAL = (BM * EXL3_MMA_BK) / 8;
        for (int idx = tid; idx < A_VEC_TOTAL; idx += 256) {
            const int r = idx >> 2;
            const int c = (idx & 3) << 3;
            uint32_t s_addr = (uint32_t) __cvta_generic_to_shared(&smem.pipe[stage].a[r][c]);
            if (r < valid_m) {
                const half * g_ptr = xh + (int64_t) (m0 + r) * k + k0 + c;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s_addr), "l"(g_ptr));
            } else {
                *reinterpret_cast<uint4 *>(&smem.pipe[stage].a[r][c]) = make_uint4(0, 0, 0, 0);
            }
        }
        asm volatile("cp.async.commit_group;\n");
    };

    const int preload = min(EXL3_MMA_STAGES - 1, num_steps);
    for (int s = 0; s < preload; ++s) {
        load_stage(s, s);
    }

    for (int step = 0; step < num_steps; ++step) {
        const int stage = step % EXL3_MMA_STAGES;
        const int next_step = step + EXL3_MMA_STAGES - 1;
        if (next_step < num_steps) {
            load_stage(next_step, next_step % EXL3_MMA_STAGES);
        }

        if (step + EXL3_MMA_STAGES - 1 < num_steps) {
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            FragB frag_b[2];
            dq_dispatch<K_BITS, CBV, HALF>((const uint32_t *) smem.pipe[stage].trellis[sub][warp], lane << 3, frag_b[0], frag_b[1]);

            const int a_row_in_tile = (lane & 15);
            const int a_col = sub * 16 + ((lane >> 4) << 3);

            #pragma unroll
            for (int m = 0; m < MT; ++m) {
                uint32_t frag_a[4];
                exl3_ldmatrix_x4_a(frag_a, &smem.pipe[stage].a[m * 16 + a_row_in_tile][a_col]);
                exl3_mma_m16n8k16(acc[m][0], frag_a, frag_b[0]);
                exl3_mma_m16n8k16(acc[m][1], frag_a, frag_b[1]);
            }
        }
        __syncthreads();
    }

    const int r0 = lane >> 2;
    const int r1 = r0 + 8;
    const int c0 = warp * 16 + ((lane & 3) << 1);
    const int c1 = c0 + 8;

    #pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int mb = m * 16;
        smem.c[mb + r0][c0 + 0] = acc[m][0][0];
        smem.c[mb + r0][c0 + 1] = acc[m][0][1];
        smem.c[mb + r1][c0 + 0] = acc[m][0][2];
        smem.c[mb + r1][c0 + 1] = acc[m][0][3];

        smem.c[mb + r0][c1 + 0] = acc[m][1][0];
        smem.c[mb + r0][c1 + 1] = acc[m][1][1];
        smem.c[mb + r1][c1 + 0] = acc[m][1][2];
        smem.c[mb + r1][c1 + 1] = acc[m][1][3];
    }
    __syncthreads();

    const int c4 = lane << 2;
    half4 sv = *reinterpret_cast<const half4 *>(svh + n0 + c4);
    const float sv0 = __half2float(sv.x.x);
    const float sv1 = __half2float(sv.x.y);
    const float sv2 = __half2float(sv.y.x);
    const float sv3 = __half2float(sv.y.y);

    for (int r = warp; r < valid_m; r += 8) {
        float v0 = smem.c[r][c4 + 0];
        float v1 = smem.c[r][c4 + 1];
        float v2 = smem.c[r][c4 + 2];
        float v3 = smem.c[r][c4 + 3];

        exl3_had128_r(v0, v1, v2, v3, lane);

        v0 *= sv0;
        v1 *= sv1;
        v2 *= sv2;
        v3 *= sv3;

        if (swiglu) {
            v0 = v0 / (1.0f + expf(-v0));
            v1 = v1 / (1.0f + expf(-v1));
            v2 = v2 / (1.0f + expf(-v2));
            v3 = v3 / (1.0f + expf(-v3));
        }

        *reinterpret_cast<float4 *>(dst + (int64_t) (m0 + r) * n + n0 + c4) = make_float4(v0, v1, v2, v3);
    }
#else
    GGML_UNUSED(trellis); GGML_UNUSED(xh); GGML_UNUSED(svh); GGML_UNUSED(dst);
    GGML_UNUSED(k); GGML_UNUSED(n); GGML_UNUSED(nrows); GGML_UNUSED(m_block);
    GGML_UNUSED(n_block); GGML_UNUSED(swiglu);
#endif
}

template <int TW, int CBV, int BM>
__global__ void __launch_bounds__(256)
k_exl3_mma_gemm(
        const uint16_t * __restrict__ trellis,
        const half * __restrict__ xh,
        const half * __restrict__ svh,
        float * __restrict__ dst,
        int k,
        int n,
        int nrows,
        bool swiglu) {
    exl3_mma_gemm_body<TW, CBV, BM>(trellis, xh, svh, dst, k, n, nrows, blockIdx.x, blockIdx.y, swiglu);
}

template <int TW, int CBV>
static void exl3_launch_mma_gemm_cb(
        int bm,
        dim3 grid,
        cudaStream_t stream,
        const uint16_t * trellis,
        const half * xh,
        const half * svh,
        float * dst,
        int k,
        int n,
        int nrows,
        bool swiglu) {
    if (bm <= 16) {
        k_exl3_mma_gemm<TW, CBV, 16><<<grid, 256, 0, stream>>>(trellis, xh, svh, dst, k, n, nrows, swiglu);
    } else if (bm <= 32) {
        k_exl3_mma_gemm<TW, CBV, 32><<<grid, 256, 0, stream>>>(trellis, xh, svh, dst, k, n, nrows, swiglu);
    } else {
        k_exl3_mma_gemm<TW, CBV, 64><<<grid, 256, 0, stream>>>(trellis, xh, svh, dst, k, n, nrows, swiglu);
    }
}

template <int TW>
static bool exl3_launch_mma_gemm_t(
        int cb,
        int bm,
        dim3 grid,
        cudaStream_t stream,
        const uint16_t * trellis,
        const half * xh,
        const half * svh,
        float * dst,
        int k,
        int n,
        int nrows,
        bool swiglu) {
    if (cb == GGML_EXL3_CODEBOOK_MCG) {
        exl3_launch_mma_gemm_cb<TW, GGML_EXL3_CODEBOOK_MCG>(bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
    } else if (cb == GGML_EXL3_CODEBOOK_MUL1) {
        exl3_launch_mma_gemm_cb<TW, GGML_EXL3_CODEBOOK_MUL1>(bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
    } else {
        exl3_launch_mma_gemm_cb<TW, GGML_EXL3_CODEBOOK_3INST>(bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
    }
    return true;
}

static bool exl3_launch_mma_gemm(
        int tile_words,
        int cb,
        int bm,
        dim3 grid,
        cudaStream_t stream,
        const uint16_t * trellis,
        const half * xh,
        const half * svh,
        float * dst,
        int k,
        int n,
        int nrows,
        bool swiglu) {
    switch (tile_words) {
        case 16:  return exl3_launch_mma_gemm_t<16 >(cb, bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 24:  return exl3_launch_mma_gemm_t<24 >(cb, bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 32:  return exl3_launch_mma_gemm_t<32 >(cb, bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 40:  return exl3_launch_mma_gemm_t<40 >(cb, bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 48:  return exl3_launch_mma_gemm_t<48 >(cb, bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 56:  return exl3_launch_mma_gemm_t<56 >(cb, bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 64:  return exl3_launch_mma_gemm_t<64 >(cb, bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 72:  return exl3_launch_mma_gemm_t<72 >(cb, bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 80:  return exl3_launch_mma_gemm_t<80 >(cb, bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 88:  return exl3_launch_mma_gemm_t<88 >(cb, bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 96:  return exl3_launch_mma_gemm_t<96 >(cb, bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 104: return exl3_launch_mma_gemm_t<104>(cb, bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 112: return exl3_launch_mma_gemm_t<112>(cb, bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 120: return exl3_launch_mma_gemm_t<120>(cb, bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        case 128: return exl3_launch_mma_gemm_t<128>(cb, bm, grid, stream, trellis, xh, svh, dst, k, n, nrows, swiglu);
        default: return false;
    }
}

static bool exl3_use_mma(int device) {
    static const int mode = []() {
        const char * env = getenv("GGML_CUDA_EXL3_MMA");
        if (!env || env[0] == '\0') return 1;
        return atoi(env);
    }();
    if (mode <= 0) return false;
    const int cc = ggml_cuda_info().devices[device].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE;
}

// ============================================================================
// Dense EXL3_MUL_MAT Dispatcher
// ============================================================================

static void exl3_mul_mat_dispatch(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * trellis,
        const ggml_tensor * su,
        const ggml_tensor * sv,
        const ggml_tensor * x,
        float * dst_f32,
        int cb,
        bool swiglu) {
    const int tile_words = (int) trellis->ne[0];
    const int nt = (int) trellis->ne[1];
    const int kt = (int) trellis->ne[2];
    const int n  = nt * 16;
    const int k  = kt * 16;
    const int64_t nrows = ggml_nrows(x);
    if (nrows == 0) return;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    ggml_cuda_pool_alloc<half> xh_buf(pool, (size_t) nrows * k);
    const int total_in_chunks = (int) (nrows * (k >> 7));
    k_exl3_had_in_f16<<<(total_in_chunks + 3) / 4, 128, 0, stream>>>(
        (const char *) x->data,
        (const half *) su->data,
        xh_buf.get(),
        k,
        (int) nrows,
        x->nb[1],
        x->nb[2],
        x->nb[3],
        x->ne[1],
        x->ne[2]);

    if (exl3_use_mma(ctx.device) && (k % 32 == 0) && (n % 128 == 0) && nrows <= 64) {
        if (nrows <= 8) {
            const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
            const bool fused_128 = ((n >> 7) >= nsm * 2);
            const dim3 grid(fused_128 ? (uint32_t) (n >> 7) : (uint32_t) (n >> 5), 1, 1);
            if (exl3_launch_gemv(
                    tile_words,
                    cb,
                    fused_128,
                    grid,
                    stream,
                    (const uint16_t *) trellis->data,
                    xh_buf.get(),
                    (const half *) sv->data,
                    dst_f32,
                    k,
                    n,
                    (int) nrows,
                    swiglu)) {
                if (!fused_128) {
                    const int total_out_chunks = (int) (nrows * (n >> 7));
                    k_exl3_had_out_inplace_slab_f32<<<(total_out_chunks + 3) / 4, 128, 0, stream>>>(
                        dst_f32,
                        (const half *) sv->data,
                        n,
                        0,
                        n,
                        (int) nrows,
                        swiglu);
                }
                return;
            }
        }
        const int bm = (nrows <= 16) ? 16 : ((nrows <= 32) ? 32 : 64);
        const dim3 grid(((int) nrows + bm - 1) / bm, n / EXL3_MMA_BN, 1);
        if (exl3_launch_mma_gemm(
                tile_words,
                cb,
                bm,
                grid,
                stream,
                (const uint16_t *) trellis->data,
                xh_buf.get(),
                (const half *) sv->data,
                dst_f32,
                k,
                n,
                (int) nrows,
                swiglu)) {
            return;
        }
    }

    static constexpr size_t MAX_W_SLAB_BYTES = 256ULL << 20;
    int64_t n_slab = (int64_t) (MAX_W_SLAB_BYTES / ((size_t) k * sizeof(half)));
    n_slab = (n_slab / 128) * 128;
    if (n_slab < 128) n_slab = 128;
    if (n_slab > n)   n_slab = n;

    ggml_cuda_pool_alloc<half> w_slab(pool, (size_t) n_slab * k);
    ggml_cuda_pool_alloc<half> c_slab(pool, (size_t) nrows * n_slab);

    cublasHandle_t cublas_h = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(cublas_h, stream));
    const float one  = 1.0f;
    const float zero = 0.0f;

    for (int n0 = 0; n0 < n; n0 += (int) n_slab) {
        const int ns = std::min((int) n_slab, n - n0);
        const dim3 rec_grid(ns / 128, kt, 1);
        const bool ok = exl3_launch_reconstruct(
            tile_words,
            cb,
            rec_grid,
            stream,
            (const uint16_t *) trellis->data,
            w_slab.get(),
            n,
            n0);
        GGML_ASSERT(ok);

        CUBLAS_CHECK(cublasGemmEx(
            cublas_h,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            ns,
            (int) nrows,
            k,
            &one,
            w_slab.get(),
            CUDA_R_16F,
            ns,
            xh_buf.get(),
            CUDA_R_16F,
            k,
            &zero,
            c_slab.get(),
            CUDA_R_16F,
            ns,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const int slab_chunks = (int) (nrows * (ns >> 7));
        k_exl3_had_out_f16_to_f32<<<(slab_chunks + 3) / 4, 128, 0, stream>>>(
            c_slab.get(),
            (const half *) sv->data,
            dst_f32,
            n,
            n0,
            ns,
            (int) nrows,
            swiglu);
    }
}

void ggml_cuda_op_exl3_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * trellis = dst->src[0];
    const ggml_tensor * x       = dst->src[1];
    const ggml_tensor * su      = dst->src[2];
    const ggml_tensor * sv      = dst->src[3];
    const int cb = ggml_get_op_params_i32(dst, 0);

    exl3_mul_mat_dispatch(ctx, trellis, su, sv, x, (float *) dst->data, cb, false);
}

void ggml_cuda_op_exl3_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * table = dst->src[0];
    const ggml_tensor * ids   = dst->src[1];
    const ggml_tensor * signs = dst->src[2];

    const int64_t n_out  = ggml_get_op_params_i32(dst, 0);
    const int64_t groups = signs->ne[1];
    const int     K      = ggml_exl3_embd_k(table->ne[0], groups);
    const int64_t rw     = ggml_exl3_ring_words(GGML_EXL3_GROUP_DIM, K);
    const int64_t n_rows = ids->ne[0];
    if (n_rows == 0 || n_out <= 0) return;

    const int64_t n_groups_active = (n_out + GGML_EXL3_GROUP_DIM - 1) / GGML_EXL3_GROUP_DIM;
    const dim3 grid((uint32_t) n_rows, (uint32_t) n_groups_active, 1);
    k_exl3_get_rows<<<grid, 256, 0, ctx.stream()>>>(
        (const char *) table->data,
        (const char *) ids->data,
        (const half *) signs->data,
        (char *) dst->data,
        table->ne[1],
        K,
        rw,
        n_out,
        n_rows,
        table->nb[1],
        ids->nb[0],
        dst->nb[1]);
}

// ============================================================================
// MoE EXL3_MUL_MAT_ID
// ============================================================================

struct exl3_id_task {
    int32_t e;
    int32_t start;
    int32_t count;
    int32_t m_block;
};

__global__ void k_exl3_id_route(
        const char * __restrict__ ids_base,
        int n_tokens,
        int n_used,
        int x_ne1,
        int n_exp,
        size_t ids_nb0,
        size_t ids_nb1,
        int32_t * __restrict__ sorted_tok,
        int32_t * __restrict__ sorted_slot,
        int32_t * __restrict__ sorted_x_col,
        int32_t * __restrict__ sorted_e,
        exl3_id_task * __restrict__ tasks_small,
        exl3_id_task * __restrict__ tasks_16,
        exl3_id_task * __restrict__ tasks_32,
        exl3_id_task * __restrict__ tasks_64,
        int32_t * __restrict__ task_counts,
        bool force_small) {
    __shared__ int32_t s_counts[512];
    __shared__ int32_t s_offsets[512];
    __shared__ int32_t s_cursor[512];
    __shared__ int32_t s_num_tasks[4];

    const int tid = threadIdx.x;
    for (int e = tid; e < n_exp; e += blockDim.x) {
        s_counts[e] = 0;
    }
    if (tid < 4) {
        s_num_tasks[tid] = 0;
    }
    __syncthreads();

    const int total = n_tokens * n_used;
    for (int idx = tid; idx < total; idx += blockDim.x) {
        const int tok  = idx / n_used;
        const int slot = idx % n_used;
        const int32_t e = *(const int32_t *) (ids_base + (size_t) slot * ids_nb0 + (size_t) tok * ids_nb1);
        if (e >= 0 && e < n_exp) {
            atomicAdd(&s_counts[e], 1);
        }
    }
    __syncthreads();

    if (tid == 0) {
        int32_t sum = 0;
        for (int e = 0; e < n_exp; ++e) {
            const int32_t c = s_counts[e];
            s_offsets[e] = sum;
            s_cursor[e]  = sum;
            sum += c;
            if (force_small) {
                for (int32_t off = 0; off < c; off += 8) {
                    const int ti = s_num_tasks[0]++;
                    tasks_small[ti] = { e, s_offsets[e] + off, min(8, c - off), 0 };
                }
            } else if (c > 0 && c <= 8) {
                const int ti = s_num_tasks[0]++;
                tasks_small[ti] = { e, s_offsets[e], c, 0 };
            } else if (c > 8 && c <= 16) {
                const int ti = s_num_tasks[1]++;
                tasks_16[ti] = { e, s_offsets[e], c, 0 };
            } else if (c > 16 && c <= 32) {
                const int ti = s_num_tasks[2]++;
                tasks_32[ti] = { e, s_offsets[e], c, 0 };
            } else if (c > 32) {
                const int nb = (c + 63) / 64;
                for (int b = 0; b < nb; ++b) {
                    const int ti = s_num_tasks[3]++;
                    tasks_64[ti] = { e, s_offsets[e], c, b };
                }
            }
        }
        for (int i = 0; i < 4; ++i) {
            task_counts[i] = s_num_tasks[i];
        }
    }
    __syncthreads();

    const int max_s  = s_num_tasks[0];
    const int max_16 = s_num_tasks[1];
    const int max_32 = s_num_tasks[2];
    const int max_64 = s_num_tasks[3];
    for (int i = max_s + tid; i < total; i += blockDim.x) {
        tasks_small[i] = { -1, 0, 0, 0 };
    }
    for (int i = max_16 + tid; i < (total / 9) + 1; i += blockDim.x) {
        tasks_16[i] = { -1, 0, 0, 0 };
    }
    for (int i = max_32 + tid; i < (total / 17) + 1; i += blockDim.x) {
        tasks_32[i] = { -1, 0, 0, 0 };
    }
    for (int i = max_64 + tid; i < (total / 33) + 2 + n_exp; i += blockDim.x) {
        tasks_64[i] = { -1, 0, 0, 0 };
    }

    // Parallel scatter across all 256 threads + write sorted_e[pos] for O(1) lookup in k_exl3_id_had_in_f16
    for (int idx = tid; idx < total; idx += blockDim.x) {
        const int tok  = idx / n_used;
        const int slot = idx % n_used;
        const int32_t e = *(const int32_t *) (ids_base + (size_t) slot * ids_nb0 + (size_t) tok * ids_nb1);
        if (e >= 0 && e < n_exp) {
            const int32_t pos = atomicAdd(&s_cursor[e], 1);
            sorted_tok[pos]   = tok;
            sorted_slot[pos]  = slot;
            sorted_x_col[pos] = slot % x_ne1;
            sorted_e[pos]     = e;
        }
    }
}

__global__ void k_exl3_id_had_in_f16(
        const char * __restrict__ x_base,
        const char * __restrict__ su_base,
        const int32_t * __restrict__ sorted_tok,
        const int32_t * __restrict__ sorted_x_col,
        const int32_t * __restrict__ sorted_e,
        half * __restrict__ xh_out,
        int k,
        int P,
        size_t x_nb1,
        size_t x_nb2,
        size_t su_nb1) {
    const int chunks_per_row = k >> 7;
    const int total_chunks = P * chunks_per_row;
    const int global_warp = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (global_warp >= total_chunks) return;

    const int lane = threadIdx.x & 31;
    const int p    = global_warp / chunks_per_row;
    const int c128 = (global_warp % chunks_per_row) << 7;
    const int col  = c128 + (lane << 2);

    const int32_t e     = sorted_e[p];
    const int32_t tok   = sorted_tok[p];
    const int32_t x_col = sorted_x_col[p];
    const float * __restrict__ x_row = (const float *) (x_base + (size_t) x_col * x_nb1 + (size_t) tok * x_nb2);
    const half  * __restrict__ suh   = (const half *)  (su_base + (size_t) e * su_nb1);

    float v0 = x_row[col + 0];
    float v1 = x_row[col + 1];
    float v2 = x_row[col + 2];
    float v3 = x_row[col + 3];
    half4 sv = *reinterpret_cast<const half4 *>(suh + col);

    v0 *= __half2float(sv.x.x);
    v1 *= __half2float(sv.x.y);
    v2 *= __half2float(sv.y.x);
    v3 *= __half2float(sv.y.y);

    exl3_had128_r(v0, v1, v2, v3, lane);

    half4 out_h;
    out_h.x = __floats2half2_rn(v0, v1);
    out_h.y = __floats2half2_rn(v2, v3);
    *reinterpret_cast<half4 *>(xh_out + (int64_t) p * k + col) = out_h;
}

template <int TW, int CBV>
__global__ void __launch_bounds__(512)
k_exl3_id_mma_gemv_fused(
        const char * __restrict__ trellis_base,
        const half * __restrict__ xh,
        const char * __restrict__ sv_base,
        const exl3_id_task * __restrict__ tasks,
        const int32_t * __restrict__ task_counts,
        const int32_t * __restrict__ sorted_tok,
        const int32_t * __restrict__ sorted_slot,
        float * __restrict__ sorted_y,
        char * __restrict__ dst_base,
        int k,
        int n,
        size_t trellis_nb3,
        size_t sv_nb1,
        size_t dst_nb1,
        size_t dst_nb2,
        bool direct_dst) {
    const int task_idx = blockIdx.y;
    if (task_idx >= task_counts[0]) return;
    const exl3_id_task task = tasks[task_idx];
    if (task.e < 0 || task.count <= 0) return;

    constexpr int bits = TW >> 4;
    constexpr bool HALF = (TW & 15) != 0;
    constexpr bool REG = (!HALF && (bits == 2 || bits == 3 || bits == 4)) ||
                         (HALF && (bits >= 1 && bits <= 3 && CBV == 2));
    constexpr int WN = 4;
    constexpr int WNT = 2;
    constexpr int WKK = 4;
    constexpr int COLS = WNT * 16;
    constexpr int TCOLS = WN * COLS; // 128
    constexpr int TWORDS = HALF ? 4 * (2 * bits + 1) : 8 * bits;
    constexpr bool TWO_PER_LOAD = REG && (HALF ? bits == 1 : bits == 2);
    constexpr int GWORDS = WNT * TWORDS;
    constexpr int LOADS = REG ? (TWO_PER_LOAD ? WNT / 2 : WNT) : ((GWORDS + 31) / 32);
    constexpr int LSTRIDE = REG ? (TWO_PER_LOAD ? 2 * TWORDS : (TWORDS < 32 ? TWORDS : 32)) : 32;
    constexpr int PF = 4;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wn = warp % WN;
    const int wk = warp / WN;

    const int ntiles = n >> 4;
    const int kslices = k >> 4;
    const int group = blockIdx.x;
    if (group * (WN * WNT) >= ntiles) return;

    const int chunk = (kslices + WKK - 1) / WKK;
    const int ks0 = wk * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));

    const uint16_t * __restrict__ trellis_e = (const uint16_t *) (trellis_base + (size_t) task.e * trellis_nb3);
    const half     * __restrict__ xh_e      = xh + (int64_t) task.start * k;
    const half     * __restrict__ svh_e     = (const half *) (sv_base + (size_t) task.e * sv_nb1);
    const int c_rows = task.count;

    const uint32_t * B32 = (const uint32_t *) trellis_e;
    const size_t slice_stride = (size_t) ntiles * TWORDS;
    const half2 * A2 = (const half2 *) xh_e;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    const int r0 = lane >> 2;
    const bool r0_ok = r0 < c_rows;
    const size_t a_row0 = r0_ok ? (size_t) r0 * (k >> 1) : 0;

    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    [[maybe_unused]] uint32_t x_pack = 0;
    if constexpr (REG && HALF) {
        constexpr int bits2 = 2 * bits + 1;
        constexpr int gspan = 18 + 3 * bits;
        const int t_offset = lane << 3;
        const int e7 = ((t_offset >> 1) + 4) * bits2 + 128 * bits2;
        const int e3 = e7 - 2 * bits2;
        const uint32_t hi7 = ((e7 - 1) / 32) % TWORDS, lo7 = ((e7 - gspan) / 32) % TWORDS, s7 = ((e7 - 1) / 32 + 1) * 32 - e7;
        const uint32_t hi3 = ((e3 - 1) / 32) % TWORDS, lo3 = ((e3 - gspan) / 32) % TWORDS, s3 = ((e3 - 1) / 32 + 1) * 32 - e3;
        x_pack = lo7 | (hi7 << 5) | (s7 << 10) | (lo3 << 15) | (hi3 << 20) | (s3 << 25);
    } else if constexpr (REG && bits == 2) {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    } else if constexpr (REG && bits == 3) {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }
    #define EXL3_XP(idx) ((x_pack >> (5 * (idx))) & 31u)

    __shared__ __align__(16) union {
        uint32_t sh_stage[REG ? 1 : WN * WKK][REG ? 1 : WNT * 64];
        float s_red[WKK][8][TCOLS];
    } smem;

    const uint32_t * bp = B32 + (size_t) ks0 * slice_stride + (group * WN + wn) * GWORDS + lane;
    auto ld_b = [&](int i, int l) -> uint32_t {
        if constexpr (REG && LSTRIDE < 32) {
            return lane < LSTRIDE ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0u;
        } else if constexpr (!REG) {
            return (l * 32 + lane < GWORDS) ? __ldcs(bp + (size_t) i * slice_stride + l * 32) : 0u;
        } else {
            return __ldcs(bp + (size_t) i * slice_stride + l * 32);
        }
    };

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);

    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
            }

            if constexpr (!REG) {
                __syncwarp();
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    if (l * 32 + lane < GWORDS)
                        smem.sh_stage[warp][l * 32 + lane] = bw[l];
                __syncwarp();
            }

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                if constexpr (!REG) {
                    dq_dispatch<bits, CBV, HALF>(&smem.sh_stage[warp][t * TWORDS], lane << 3, f0, f1);
                } else if constexpr (HALF) {
                    const uint32_t w = TWO_PER_LOAD ? bw[t >> 1] : bw[t];
                    const int base = TWO_PER_LOAD ? (t & 1) * TWORDS : 0;
                    uint32_t a7 = __shfl_sync(0xffffffffu, w, base + EXL3_XP(0));
                    uint32_t b7 = __shfl_sync(0xffffffffu, w, base + EXL3_XP(1));
                    uint32_t a3 = __shfl_sync(0xffffffffu, w, base + EXL3_XP(3));
                    uint32_t b3 = __shfl_sync(0xffffffffu, w, base + EXL3_XP(4));
                    exl3_gemv_ns::dq8_regs_half<bits, CBV>(a7, b7, EXL3_XP(2), a3, b3, EXL3_XP(5), f0, f1);
                } else if constexpr (bits == 4) {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<CBV>(aw, bw[t], f0, f1);
                } else if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<CBV>(awv, bwv, lane << 3, f0, f1);
                } else {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<CBV>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_f32(a01, a23, f0, acc0[t][0]);
                exl3_gemv_ns::mma_ab_f32(a01, a23, f1, acc0[t][1]);
            }
        }
    }
    #undef EXL3_XP

    if (!REG) __syncthreads();

    if (r0_ok) {
        const int c0 = 2 * (lane & 3);
        #pragma unroll
        for (int t = 0; t < WNT; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = wn * COLS + t * 16 + f * 8 + c0;
                smem.s_red[wk][r0][col + 0] = acc0[t][f].x;
                smem.s_red[wk][r0][col + 1] = acc0[t][f].y;
            }
    }
    __syncthreads();

    if (warp < c_rows) {
        const int m = warp;
        const int p = task.start + m;
        const int c4 = lane << 2;
        const int global_col = group * TCOLS + c4;

        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        #pragma unroll
        for (int w = 0; w < WKK; ++w) {
            v0 += smem.s_red[w][m][c4 + 0];
            v1 += smem.s_red[w][m][c4 + 1];
            v2 += smem.s_red[w][m][c4 + 2];
            v3 += smem.s_red[w][m][c4 + 3];
        }

        exl3_had128_r(v0, v1, v2, v3, lane);

        half4 sv = *reinterpret_cast<const half4 *>(svh_e + global_col);
        v0 *= __half2float(sv.x.x);
        v1 *= __half2float(sv.x.y);
        v2 *= __half2float(sv.y.x);
        v3 *= __half2float(sv.y.y);

        float * __restrict__ out_row = direct_dst
            ? (float *) (dst_base + (size_t) sorted_slot[p] * dst_nb1 + (size_t) sorted_tok[p] * dst_nb2)
            : (sorted_y + (int64_t) p * n);

        *reinterpret_cast<float4 *>(out_row + global_col) = make_float4(v0, v1, v2, v3);
    }
}

template <int TW, int CBV, int BM, int TASK_BUCKET>
__global__ void __launch_bounds__(256)
k_exl3_id_mma_gemm(
        const char * __restrict__ trellis_base,
        const half * __restrict__ xh,
        const char * __restrict__ sv_base,
        const exl3_id_task * __restrict__ tasks,
        const int32_t * __restrict__ task_counts,
        float * __restrict__ sorted_y,
        int k,
        int n,
        size_t trellis_nb3,
        size_t sv_nb1) {
    const int task_idx = blockIdx.x;
    if (task_idx >= task_counts[TASK_BUCKET]) return;
    const exl3_id_task task = tasks[task_idx];
    if (task.e < 0 || task.count <= 0) return;

    const uint16_t * trellis_e = (const uint16_t *) (trellis_base + (size_t) task.e * trellis_nb3);
    const half     * svh_e     = (const half *)     (sv_base      + (size_t) task.e * sv_nb1);
    const half     * xh_e      = xh       + (int64_t) task.start * k;
    float          * y_e       = sorted_y + (int64_t) task.start * n;

    exl3_mma_gemm_body<TW, CBV, BM>(
        trellis_e, xh_e, svh_e, y_e,
        k, n, task.count, task.m_block, blockIdx.y, false);
}

__global__ void k_exl3_id_combine_f32(
        const float * __restrict__ sorted_y,
        const int32_t * __restrict__ sorted_tok,
        const int32_t * __restrict__ sorted_slot,
        char * __restrict__ dst_base,
        int n,
        int n_tokens,
        int n_used,
        size_t dst_nb1,
        size_t dst_nb2) {
    const int p = blockIdx.x;
    if (p >= n_tokens * n_used) return;
    const int32_t tok  = sorted_tok[p];
    const int32_t slot = sorted_slot[p];
    const float * __restrict__ src = sorted_y + (int64_t) p * n;
    float * __restrict__ dst = (float *) (dst_base + (size_t) slot * dst_nb1 + (size_t) tok * dst_nb2);

    for (int col = threadIdx.x * 4; col < n; col += blockDim.x * 4) {
        *reinterpret_cast<float4 *>(dst + col) = *reinterpret_cast<const float4 *>(src + col);
    }
}

template <int TW, int CBV>
static void exl3_launch_id_mma_cb(
        cudaStream_t stream,
        const ggml_tensor * trellis,
        const ggml_tensor * sv,
        const half * xh,
        const exl3_id_task * tasks_small,
        const exl3_id_task * tasks_16,
        const exl3_id_task * tasks_32,
        const exl3_id_task * tasks_64,
        const int32_t * task_counts,
        const int32_t * sorted_tok,
        const int32_t * sorted_slot,
        float * sorted_y,
        ggml_tensor * dst,
        int k,
        int n,
        int max_small,
        int max_16,
        int max_32,
        int max_64,
        bool direct_dst) {
    const int n_blocks_128 = n / EXL3_MMA_BN;

    if (max_small > 0) {
        const dim3 grid_v(n_blocks_128, max_small, 1);
        k_exl3_id_mma_gemv_fused<TW, CBV><<<grid_v, 512, 0, stream>>>(
            (const char *) trellis->data, xh, (const char *) sv->data,
            tasks_small, task_counts, sorted_tok, sorted_slot,
            sorted_y, (char *) dst->data,
            k, n, trellis->nb[3], sv->nb[1], dst->nb[1], dst->nb[2], direct_dst);
    }

    if (max_16 > 0) {
        const dim3 grid_16(max_16, n_blocks_128, 1);
        k_exl3_id_mma_gemm<TW, CBV, 16, 1><<<grid_16, 256, 0, stream>>>(
            (const char *) trellis->data, xh, (const char *) sv->data,
            tasks_16, task_counts, sorted_y, k, n, trellis->nb[3], sv->nb[1]);
    }
    if (max_32 > 0) {
        const dim3 grid_32(max_32, n_blocks_128, 1);
        k_exl3_id_mma_gemm<TW, CBV, 32, 2><<<grid_32, 256, 0, stream>>>(
            (const char *) trellis->data, xh, (const char *) sv->data,
            tasks_32, task_counts, sorted_y, k, n, trellis->nb[3], sv->nb[1]);
    }
    if (max_64 > 0) {
        const dim3 grid_64(max_64, n_blocks_128, 1);
        k_exl3_id_mma_gemm<TW, CBV, 64, 3><<<grid_64, 256, 0, stream>>>(
            (const char *) trellis->data, xh, (const char *) sv->data,
            tasks_64, task_counts, sorted_y, k, n, trellis->nb[3], sv->nb[1]);
    }
}

template <int TW>
static bool exl3_launch_id_mma_t(
        int cb,
        cudaStream_t stream,
        const ggml_tensor * trellis,
        const ggml_tensor * sv,
        const half * xh,
        const exl3_id_task * tasks_small,
        const exl3_id_task * tasks_16,
        const exl3_id_task * tasks_32,
        const exl3_id_task * tasks_64,
        const int32_t * task_counts,
        const int32_t * sorted_tok,
        const int32_t * sorted_slot,
        float * sorted_y,
        ggml_tensor * dst,
        int k,
        int n,
        int max_small,
        int max_16,
        int max_32,
        int max_64,
        bool direct_dst) {
    if (cb == GGML_EXL3_CODEBOOK_MCG) {
        exl3_launch_id_mma_cb<TW, GGML_EXL3_CODEBOOK_MCG>(
            stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64,
            task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n,
            max_small, max_16, max_32, max_64, direct_dst);
    } else if (cb == GGML_EXL3_CODEBOOK_MUL1) {
        exl3_launch_id_mma_cb<TW, GGML_EXL3_CODEBOOK_MUL1>(
            stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64,
            task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n,
            max_small, max_16, max_32, max_64, direct_dst);
    } else {
        exl3_launch_id_mma_cb<TW, GGML_EXL3_CODEBOOK_3INST>(
            stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64,
            task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n,
            max_small, max_16, max_32, max_64, direct_dst);
    }
    return true;
}

static bool exl3_launch_id_mma(
        int tile_words,
        int cb,
        cudaStream_t stream,
        const ggml_tensor * trellis,
        const ggml_tensor * sv,
        const half * xh,
        const exl3_id_task * tasks_small,
        const exl3_id_task * tasks_16,
        const exl3_id_task * tasks_32,
        const exl3_id_task * tasks_64,
        const int32_t * task_counts,
        const int32_t * sorted_tok,
        const int32_t * sorted_slot,
        float * sorted_y,
        ggml_tensor * dst,
        int k,
        int n,
        int max_small,
        int max_16,
        int max_32,
        int max_64,
        bool direct_dst) {
    switch (tile_words) {
        case 16:  return exl3_launch_id_mma_t<16 >(cb, stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64, task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n, max_small, max_16, max_32, max_64, direct_dst);
        case 24:  return exl3_launch_id_mma_t<24 >(cb, stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64, task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n, max_small, max_16, max_32, max_64, direct_dst);
        case 32:  return exl3_launch_id_mma_t<32 >(cb, stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64, task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n, max_small, max_16, max_32, max_64, direct_dst);
        case 40:  return exl3_launch_id_mma_t<40 >(cb, stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64, task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n, max_small, max_16, max_32, max_64, direct_dst);
        case 48:  return exl3_launch_id_mma_t<48 >(cb, stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64, task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n, max_small, max_16, max_32, max_64, direct_dst);
        case 56:  return exl3_launch_id_mma_t<56 >(cb, stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64, task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n, max_small, max_16, max_32, max_64, direct_dst);
        case 64:  return exl3_launch_id_mma_t<64 >(cb, stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64, task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n, max_small, max_16, max_32, max_64, direct_dst);
        case 72:  return exl3_launch_id_mma_t<72 >(cb, stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64, task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n, max_small, max_16, max_32, max_64, direct_dst);
        case 80:  return exl3_launch_id_mma_t<80 >(cb, stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64, task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n, max_small, max_16, max_32, max_64, direct_dst);
        case 88:  return exl3_launch_id_mma_t<88 >(cb, stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64, task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n, max_small, max_16, max_32, max_64, direct_dst);
        case 96:  return exl3_launch_id_mma_t<96 >(cb, stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64, task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n, max_small, max_16, max_32, max_64, direct_dst);
        case 104: return exl3_launch_id_mma_t<104>(cb, stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64, task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n, max_small, max_16, max_32, max_64, direct_dst);
        case 112: return exl3_launch_id_mma_t<112>(cb, stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64, task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n, max_small, max_16, max_32, max_64, direct_dst);
        case 120: return exl3_launch_id_mma_t<120>(cb, stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64, task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n, max_small, max_16, max_32, max_64, direct_dst);
        case 128: return exl3_launch_id_mma_t<128>(cb, stream, trellis, sv, xh, tasks_small, tasks_16, tasks_32, tasks_64, task_counts, sorted_tok, sorted_slot, sorted_y, dst, k, n, max_small, max_16, max_32, max_64, direct_dst);
        default: return false;
    }
}

void ggml_cuda_op_exl3_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * trellis = dst->src[0];
    const ggml_tensor * x       = dst->src[1];
    const ggml_tensor * ids     = dst->src[2];
    const ggml_tensor * su      = dst->src[3];
    const ggml_tensor * sv      = dst->src[4];

    const int tile_words = (int) trellis->ne[0];
    const int n          = (int) (trellis->ne[1] * 16);
    const int k          = (int) (trellis->ne[2] * 16);
    const int n_exp      = (int) trellis->ne[3];
    const int n_used     = (int) dst->ne[1];
    const int n_tokens   = (int) dst->ne[2];
    const int cb         = ggml_get_op_params_i32(dst, 0);

    if (n_tokens == 0 || n_used == 0) return;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    const int P = n_tokens * n_used;

    if (exl3_use_mma(ctx.device) && (k % 32 == 0) && (n % 128 == 0) && n_exp <= 512) {
        const bool force_small = (n_tokens <= 8);
        const int max_small = force_small ? std::min(P, n_exp + (P / 8)) : std::min(P, n_exp);
        const int max_16    = (force_small || P < 9)  ? 0 : std::min(P / 9,  n_exp);
        const int max_32    = (force_small || P < 17) ? 0 : std::min(P / 17, n_exp);
        const int max_64    = (force_small || P < 33) ? 0 : ((P / 64) + n_exp);
        const bool direct_dst = (max_16 == 0 && max_32 == 0 && max_64 == 0);

        ggml_cuda_pool_alloc<int32_t>      sorted_tok  (pool, P);
        ggml_cuda_pool_alloc<int32_t>      sorted_slot (pool, P);
        ggml_cuda_pool_alloc<int32_t>      sorted_x_col(pool, P);
        ggml_cuda_pool_alloc<int32_t>      sorted_e    (pool, P);
        ggml_cuda_pool_alloc<exl3_id_task> tasks_small (pool, P);
        ggml_cuda_pool_alloc<exl3_id_task> tasks_16    (pool, (P / 9) + 1);
        ggml_cuda_pool_alloc<exl3_id_task> tasks_32    (pool, (P / 17) + 1);
        ggml_cuda_pool_alloc<exl3_id_task> tasks_64    (pool, (P / 33) + 2 + n_exp);
        ggml_cuda_pool_alloc<int32_t>      task_counts (pool, 4);

        k_exl3_id_route<<<1, 256, 0, stream>>>(
            (const char *) ids->data,
            n_tokens,
            n_used,
            (int) x->ne[1],
            n_exp,
            ids->nb[0],
            ids->nb[1],
            sorted_tok.get(),
            sorted_slot.get(),
            sorted_x_col.get(),
            sorted_e.get(),
            tasks_small.get(),
            tasks_16.get(),
            tasks_32.get(),
            tasks_64.get(),
            task_counts.get(),
            force_small);

        ggml_cuda_pool_alloc<half>  xh_buf  (pool, (size_t) P * k);
        ggml_cuda_pool_alloc<float> sorted_y(pool, direct_dst ? 1 : (size_t) P * n);

        const int total_in_chunks = P * (k >> 7);
        k_exl3_id_had_in_f16<<<(total_in_chunks + 3) / 4, 128, 0, stream>>>(
            (const char *) x->data,
            (const char *) su->data,
            sorted_tok.get(),
            sorted_x_col.get(),
            sorted_e.get(),
            xh_buf.get(),
            k,
            P,
            x->nb[1],
            x->nb[2],
            su->nb[1]);

        if (exl3_launch_id_mma(
                tile_words,
                cb,
                stream,
                trellis,
                sv,
                xh_buf.get(),
                tasks_small.get(),
                tasks_16.get(),
                tasks_32.get(),
                tasks_64.get(),
                task_counts.get(),
                sorted_tok.get(),
                sorted_slot.get(),
                sorted_y.get(),
                dst,
                k,
                n,
                max_small,
                max_16,
                max_32,
                max_64,
                direct_dst)) {
            if (!direct_dst) {
                k_exl3_id_combine_f32<<<P, 128, 0, stream>>>(
                    sorted_y.get(),
                    sorted_tok.get(),
                    sorted_slot.get(),
                    (char *) dst->data,
                    n,
                    n_tokens,
                    n_used,
                    dst->nb[1],
                    dst->nb[2]);
            }
            return;
        }
    }

    const size_t ids_bytes = (size_t) (n_tokens - 1) * ids->nb[1] + (size_t) n_used * ids->nb[0];
    std::vector<char> ids_host(ids_bytes);
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ids_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    struct exp_route { int tok; int slot; int x_col; };
    std::vector<std::vector<exp_route>> routes(n_exp);
    for (int tok = 0; tok < n_tokens; ++tok) {
        for (int slot = 0; slot < n_used; ++slot) {
            const int32_t e = *(const int32_t *) (ids_host.data() + (size_t) slot * ids->nb[0] + (size_t) tok * ids->nb[1]);
            GGML_ASSERT(e >= 0 && e < n_exp);
            const int x_col = slot % (int) x->ne[1];
            routes[e].push_back({ tok, slot, x_col });
        }
    }

    for (int e = 0; e < n_exp; ++e) {
        const int m = (int) routes[e].size();
        if (m == 0) continue;

        ggml_cuda_pool_alloc<float> x_e(pool, (size_t) m * k);
        ggml_cuda_pool_alloc<float> y_e(pool, (size_t) m * n);

        for (int r = 0; r < m; ++r) {
            const char * src_row = (const char *) x->data + (size_t) routes[e][r].x_col * x->nb[1] + (size_t) routes[e][r].tok * x->nb[2];
            CUDA_CHECK(cudaMemcpyAsync(x_e.get() + (size_t) r * k, src_row, (size_t) k * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }

        ggml_tensor t_e  = *trellis;
        t_e.ne[3] = 1;
        t_e.data  = (char *) trellis->data + (size_t) e * trellis->nb[3];

        ggml_tensor su_e = *su;
        su_e.ne[1] = 1;
        su_e.data  = (char *) su->data + (size_t) e * su->nb[1];

        ggml_tensor sv_e = *sv;
        sv_e.ne[1] = 1;
        sv_e.data  = (char *) sv->data + (size_t) e * sv->nb[1];

        ggml_tensor x_sub = *x;
        x_sub.ne[0] = k;
        x_sub.ne[1] = m;
        x_sub.ne[2] = 1;
        x_sub.ne[3] = 1;
        x_sub.nb[0] = sizeof(float);
        x_sub.nb[1] = (size_t) k * sizeof(float);
        x_sub.nb[2] = (size_t) m * k * sizeof(float);
        x_sub.nb[3] = (size_t) m * k * sizeof(float);
        x_sub.data  = x_e.get();

        exl3_mul_mat_dispatch(ctx, &t_e, &su_e, &sv_e, &x_sub, y_e.get(), cb, false);

        for (int r = 0; r < m; ++r) {
            char * dst_row = (char *) dst->data + (size_t) routes[e][r].slot * dst->nb[1] + (size_t) routes[e][r].tok * dst->nb[2];
            CUDA_CHECK(cudaMemcpyAsync(dst_row, y_e.get() + (size_t) r * n, (size_t) n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
    }
}

bool ggml_cuda_exl3_mul_mat_id_needs_sync(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_EXL3_MUL_MAT_ID) return false;
    const ggml_tensor * trellis = op->src[0];
    if (!trellis) return true;
    const int n     = (int) (trellis->ne[1] * 16);
    const int k     = (int) (trellis->ne[2] * 16);
    const int n_exp = (int) trellis->ne[3];
    return !(exl3_use_mma(ggml_cuda_get_device()) && (k % 32 == 0) && (n % 128 == 0) && n_exp <= 512);
}

bool ggml_cuda_exl3_supports_op(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_EXL3_MUL_MAT: {
            const ggml_tensor * trellis = op->src[0];
            const ggml_tensor * x       = op->src[1];
            const ggml_tensor * su      = op->src[2];
            const ggml_tensor * sv      = op->src[3];
            const int cb = ggml_get_op_params_i32(op, 0);
            if (!trellis || !x || !su || !sv) return false;
            if (trellis->type != GGML_TYPE_I16 || su->type != GGML_TYPE_F16 || sv->type != GGML_TYPE_F16) return false;
            if (x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float)) return false;
            if (!ggml_exl3_rate_supported(trellis->ne[0], (ggml_exl3_codebook) cb)) return false;
            const int64_t n = trellis->ne[1] * 16;
            const int64_t k = trellis->ne[2] * 16;
            if ((n % 128) != 0 || (k % 128) != 0) return false;
            return ggml_is_contiguous(trellis) && ggml_is_contiguous(su) && ggml_is_contiguous(sv);
        }
        case GGML_OP_EXL3_MUL_MAT_ID: {
            const ggml_tensor * trellis = op->src[0];
            const ggml_tensor * x       = op->src[1];
            const ggml_tensor * ids     = op->src[2];
            const ggml_tensor * su      = op->src[3];
            const ggml_tensor * sv      = op->src[4];
            const int cb = ggml_get_op_params_i32(op, 0);
            if (!trellis || !x || !ids || !su || !sv) return false;
            if (trellis->type != GGML_TYPE_I16 || su->type != GGML_TYPE_F16 || sv->type != GGML_TYPE_F16) return false;
            if (x->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || op->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float)) return false;
            if (!ggml_exl3_rate_supported(trellis->ne[0], (ggml_exl3_codebook) cb)) return false;
            const int64_t n = trellis->ne[1] * 16;
            const int64_t k = trellis->ne[2] * 16;
            return (n % 128) == 0 && (k % 128) == 0 && ggml_is_contiguous(trellis) && ggml_is_contiguous(su) && ggml_is_contiguous(sv);
        }
        case GGML_OP_EXL3_GET_ROWS: {
            const ggml_tensor * table = op->src[0];
            const ggml_tensor * ids   = op->src[1];
            const ggml_tensor * signs = op->src[2];
            if (!table || !ids || !signs) return false;
            if (table->type != GGML_TYPE_I16 || ids->type != GGML_TYPE_I32 || signs->type != GGML_TYPE_F16 || op->type != GGML_TYPE_F32) return false;
            return ggml_is_contiguous(op) && ggml_exl3_embd_k(table->ne[0], signs->ne[1]) > 0;
        }
        default:
            return false;
    }
}

#endif // GGML_USE_EXL3
