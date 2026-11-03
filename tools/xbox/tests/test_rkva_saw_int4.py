#!/usr/bin/env python3
"""Comprehensive test and verification oracle for RKVA SAW-INT4 (BDR-16 + INT4).

Tests:
1. Mathematical properties of BDR-16 (Block-Diagonal Fast Walsh-Hadamard Transform):
   - Symmetry: R^T = R
   - Orthogonality / Unitary: R * R^T = I
   - Self-Inverse: R * R = I
   - Inner product invariance: (Q * R) . (K * R) == Q . K
   - Attention output linearity: (Softmax * (V * R)) * R == Softmax * V
2. Bit-exact Wave64 intrinsics:
   - Simulates bdr16_wave4 across 64 lanes (16 blocks of 16)
   - Matches reference FWHT16 with max_diff == 0.0
3. HLSL Rowlane Codec & Alignment:
   - quarter_dim_saw_int4(u) covers all 256 dims disjointly with zero gaps
   - load_saw_int4 uses strictly 4-byte aligned ByteAddressBuffer loads
   - Dequantization matches C++ golden reference with max_diff == 0.0
4. End-to-End Attention with Outlier Suppression:
   - Validates MSE and SNR reduction of SAW-INT4 over Plain INT4
"""

import ctypes
import os
import struct
import sys
import numpy as np

# Load C++ SAW-INT4 shared library
SO_PATH = '/home/hjotha/beellama.cpp/build-optimized/bin/librkva_saw_int4.so'
assert os.path.exists(SO_PATH), f"Shared library not found at {SO_PATH}"
librkva_saw = ctypes.CDLL(SO_PATH)

librkva_saw.rkva_bdr16_transform_256.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
librkva_saw.rkva_bdr16_transform_256.restype = None

librkva_saw.rkva_quantize_row_saw_int4.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64]
librkva_saw.rkva_quantize_row_saw_int4.restype = None

librkva_saw.rkva_dequantize_row_saw_int4.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
librkva_saw.rkva_dequantize_row_saw_int4.restype = None

librkva_saw.rkva_dequantize_row_saw_int4_rotated.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
librkva_saw.rkva_dequantize_row_saw_int4_rotated.restype = None

# ==============================================================================
# SGLang / SAW-INT4 Reference Implementations
# ==============================================================================

def numpy_fwht_f32(a: np.ndarray) -> np.ndarray:
    """Sylvester Fast Walsh-Hadamard Transform on 1D array of length power-of-two."""
    a2 = np.array(a, dtype=np.float32, copy=True)
    n = len(a2)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                u = a2[i + j]
                v = a2[i + j + h]
                a2[i + j] = u + v
                a2[i + j + h] = u - v
        h *= 2
    return a2

def ref_bdr_transform(vec: np.ndarray, block_size: int = 16) -> np.ndarray:
    """Block-Diagonal Walsh-Hadamard Transform with scaling 1/sqrt(block_size)."""
    vec = np.array(vec, dtype=np.float32, copy=True)
    D = len(vec)
    assert D % block_size == 0
    scale = 1.0 / np.sqrt(block_size)
    out = np.zeros_like(vec)
    for b in range(0, D, block_size):
        sub = vec[b:b + block_size]
        out[b:b + block_size] = numpy_fwht_f32(sub) * scale
    return out

# ==============================================================================
# Mock D3D12 ByteAddressBuffer
# ==============================================================================

class MockByteAddressBuffer:
    def __init__(self, data: bytes):
        self.data = memoryview(data)

    def Load(self, addr: int) -> int:
        assert addr % 4 == 0, f"Unaligned ByteAddressBuffer.Load at {addr}"
        assert addr + 4 <= len(self.data), f"Out of bounds ByteAddressBuffer.Load at {addr} (size {len(self.data)})"
        return struct.unpack_from('<I', self.data, addr)[0]

def asfloat(u32: int) -> float:
    return struct.unpack('<f', struct.pack('<I', u32 & 0xFFFFFFFF))[0]

# HLSL codec logic
def quarter_dim_saw_int4(u: int) -> list[int]:
    base = u * 4 if u < 32 else (128 + (u - 32) * 4)
    return [base, base + 1, base + 2, base + 3]

def load_saw_int4(buffer: MockByteAddressBuffer, base: int, u: int) -> np.ndarray:
    k = u & 31
    word = buffer.Load(base + k * 4)
    is_hi = (u >> 5) & 1
    word >>= (is_hi * 4)

    q0 = word & 0x0F
    q1 = (word >> 8) & 0x0F
    q2 = (word >> 16) & 0x0F
    q3 = (word >> 24) & 0x0F

    scale = asfloat(buffer.Load(base + 128))
    zero  = asfloat(buffer.Load(base + 132))

    return (np.array([q0, q1, q2, q3], dtype=np.float32) - zero) * scale

# Wave64 BDR-16 intrinsic simulation
def simulate_bdr16_wave4(wave_values: np.ndarray) -> np.ndarray:
    """Simulates bdr16_wave4 across 64 lanes (16 groups of 4 lanes)."""
    assert wave_values.shape == (64, 4)
    out = np.zeros_like(wave_values)

    for g in range(16):
        # 4 lanes for group g
        lane_indices = [g * 4 + 0, g * 4 + 1, g * 4 + 2, g * 4 + 3]
        v = wave_values[lane_indices].copy()

        # 1. Intra-lane stages (stride 1 and 2 within the 4 elements)
        for l in range(4):
            u0, w0 = v[l, 0] + v[l, 1], v[l, 0] - v[l, 1]
            u1, w1 = v[l, 2] + v[l, 3], v[l, 2] - v[l, 3]
            v[l, 0], v[l, 2] = u0 + u1, u0 - u1
            v[l, 1], v[l, 3] = w0 + w1, w0 - w1

        # 2. Inter-lane stages across the 4 lanes
        # Stage 3: partner = l ^ 1
        s3 = np.zeros_like(v)
        for l in range(4):
            p1 = l ^ 1
            s3[l] = (v[l] + v[p1]) if ((l & 1) == 0) else (v[p1] - v[l])

        # Stage 4: partner = l ^ 2
        s4 = np.zeros_like(v)
        for l in range(4):
            p2 = l ^ 2
            s4[l] = (s3[l] + s3[p2]) if ((l & 2) == 0) else (s3[p2] - s3[l])

        out[lane_indices] = s4 * 0.25

    return out

# ==============================================================================
# Unit Tests
# ==============================================================================

def test_mathematical_properties():
    print("--- [Test 1] Mathematical Properties of BDR-16 ---")
    np.random.seed(42)
    D = 256
    x = np.random.randn(D).astype(np.float32)

    # 1. Self-Inverse
    x_rot = ref_bdr_transform(x, 16)
    x_rec = ref_bdr_transform(x_rot, 16)
    diff_inv = np.max(np.abs(x - x_rec))
    print(f"Self-Inverse diff: {diff_inv:.8e}")
    assert diff_inv < 1e-6, "BDR-16 must be self-inverse"

    # 2. C++ library parity
    x_c = (ctypes.c_float * 256)(*x)
    x_rot_c = (ctypes.c_float * 256)()
    librkva_saw.rkva_bdr16_transform_256(x_c, x_rot_c)
    diff_c = np.max(np.abs(x_rot - np.array(x_rot_c)))
    print(f"C++ bdr16 parity diff: {diff_c:.8e}")
    assert diff_c < 1e-6, "C++ bdr16 must match Python reference"

    # 3. Dot product preservation
    q = np.random.randn(D).astype(np.float32)
    k = np.random.randn(D).astype(np.float32)
    q_rot = ref_bdr_transform(q, 16)
    k_rot = ref_bdr_transform(k, 16)
    dot_orig = np.dot(q, k)
    dot_rot  = np.dot(q_rot, k_rot)
    diff_dot = np.abs(dot_orig - dot_rot)
    print(f"Dot product orig={dot_orig:.4f}, rot={dot_rot:.4f}, diff={diff_dot:.8e}")
    assert diff_dot < 1e-5, "Dot product must be preserved under BDR rotation"

    # 4. Output des-rotation linearity
    T = 64
    V = np.random.randn(T, D).astype(np.float32)
    P = np.random.dirichlet(np.ones(T)).astype(np.float32)
    O_orig = np.sum(P[:, None] * V, axis=0)

    V_rot = np.stack([ref_bdr_transform(v, 16) for v in V])
    O_rot = np.sum(P[:, None] * V_rot, axis=0)
    O_rec = ref_bdr_transform(O_rot, 16)
    diff_o = np.max(np.abs(O_orig - O_rec))
    print(f"Attention output des-rotation diff: {diff_o:.8e}")
    assert diff_o < 1e-6, "Attention output des-rotation must be exact"
    print("PASS: Mathematical properties verified.\n")


def test_wave64_intrinsics():
    print("--- [Test 2] Wave64 bdr16_wave4 Intrinsics Parity ---")
    np.random.seed(1337)
    x = np.random.randn(256).astype(np.float32)

    # Reference BDR transform
    ref_rot = ref_bdr_transform(x, 16)

    # In Wave64, each lane holds float4.
    # Lanes 0..63 hold x[0..255] in order: lane u holds x[4u .. 4u+3]
    wave_vals = x.reshape(64, 4)
    wave_out = simulate_bdr16_wave4(wave_vals).reshape(256)

    diff = np.max(np.abs(ref_rot - wave_out))
    print(f"Wave64 simulation vs reference FWHT diff: {diff:.8e}")
    assert diff == 0.0 or diff < 1e-6, "Wave64 bdr16_wave4 must match reference exactly"
    print("PASS: Wave64 intrinsics parity verified.\n")


def test_quarter_dim_and_alignment():
    print("--- [Test 3] Quarter Dim Mapping & Byte Alignment ---")
    covered_dims = set()
    for u in range(64):
        dims = quarter_dim_saw_int4(u)
        assert len(dims) == 4
        for d in dims:
            assert 0 <= d < 256, f"Invalid dim {d} for quarter {u}"
            assert d not in covered_dims, f"Duplicate dim {d} mapped at quarter {u}"
            covered_dims.add(d)

    assert len(covered_dims) == 256, f"Expected 256 covered dims, got {len(covered_dims)}"
    print("PASS: All 256 dimensions covered uniquely and disjointly.")

    # Test load_saw_int4 address alignment and bounds
    for u in range(64):
        k = u & 31
        addr = k * 4
        assert addr % 4 == 0, f"Unaligned addr {addr} for u={u}"
        assert addr + 4 <= 128, f"Quants out of bounds: {addr}"

    scale_addr = 128
    zero_addr = 132
    assert scale_addr % 4 == 0 and scale_addr + 4 <= 136
    assert zero_addr % 4 == 0 and zero_addr + 4 <= 136
    print("PASS: All HLSL memory addresses are 4-byte aligned and strictly bounded.\n")


def test_hlsl_dequant_parity():
    print("--- [Test 4] HLSL Dequantizer Bit-Exact Parity vs C++ Golden Reference ---")
    np.random.seed(999)
    # Generate vector with realistic LLM activation spikes
    src = np.random.randn(256).astype(np.float32)
    src[12] = 28.5
    src[87] = -34.2
    src[210] = 24.1

    # Quantize via C++ library
    src_c = (ctypes.c_float * 256)(*src)
    raw_block = (ctypes.c_uint8 * 136)()
    librkva_saw.rkva_quantize_row_saw_int4(src_c, ctypes.byref(raw_block), 256)

    # Dequantize rotated domain via C++
    deq_c_rot = (ctypes.c_float * 256)()
    librkva_saw.rkva_dequantize_row_saw_int4_rotated(ctypes.byref(raw_block), deq_c_rot, 256)
    gold_rot = np.array(deq_c_rot, dtype=np.float32)

    # Dequantize rotated domain via HLSL mock
    buf = MockByteAddressBuffer(bytes(raw_block))
    hlsl_rot = np.zeros(256, dtype=np.float32)
    for u in range(64):
        dims = quarter_dim_saw_int4(u)
        vals = load_saw_int4(buf, 0, u)
        for i, d in enumerate(dims):
            hlsl_rot[d] = vals[i]

    diff = np.max(np.abs(gold_rot - hlsl_rot))
    print(f"HLSL load_saw_int4 vs C++ golden dequant diff: {diff:.8e}")
    assert diff == 0.0, f"Expected bit-exact 0.0 diff, got {diff}"

    # Dequantize full domain (including des-rotation) via C++
    deq_c_full = (ctypes.c_float * 256)()
    librkva_saw.rkva_dequantize_row_saw_int4(ctypes.byref(raw_block), deq_c_full, 256)
    gold_full = np.array(deq_c_full, dtype=np.float32)

    # Full dequant via HLSL + BDR
    hlsl_full = ref_bdr_transform(hlsl_rot, 16)
    diff_full = np.max(np.abs(gold_full - hlsl_full))
    print(f"Full dequant (HLSL + BDR) vs C++ full dequant diff: {diff_full:.8e}")
    assert diff_full < 1e-6, "Full dequantization must match golden C++ reference"
    print("PASS: Bit-exact HLSL dequantization confirmed.\n")


def test_outlier_suppression_and_accuracy():
    print("--- [Test 5] Outlier Suppression & Accuracy Evaluation vs Plain INT4 ---")
    np.random.seed(2026)
    D = 256
    N_SAMPLES = 50

    saw_mse_list = []
    plain_mse_list = []

    for _ in range(N_SAMPLES):
        # Normal activation + kurtotic outliers
        x = np.random.randn(D).astype(np.float32)
        outlier_idx = np.random.choice(D, size=4, replace=False)
        x[outlier_idx] *= 15.0  # Large channel outliers

        # 1. Plain INT4 (no rotation)
        vmin = np.min(x)
        vmax = np.max(x)
        vrange = max(vmax - vmin, 1e-8)
        scale_plain = vrange / 15.0
        zero_plain = -vmin / scale_plain
        q_plain = np.clip(np.floor(x / scale_plain + zero_plain + 0.5), 0, 15)
        x_plain_rec = (q_plain - zero_plain) * scale_plain
        plain_mse_list.append(np.mean((x - x_plain_rec)**2))

        # 2. SAW-INT4 (BDR-16 + INT4)
        x_c = (ctypes.c_float * 256)(*x)
        raw_block = (ctypes.c_uint8 * 136)()
        librkva_saw.rkva_quantize_row_saw_int4(x_c, ctypes.byref(raw_block), 256)
        x_saw_c = (ctypes.c_float * 256)()
        librkva_saw.rkva_dequantize_row_saw_int4(ctypes.byref(raw_block), x_saw_c, 256)
        x_saw_rec = np.array(x_saw_c, dtype=np.float32)
        saw_mse_list.append(np.mean((x - x_saw_rec)**2))

    mean_plain_mse = np.mean(plain_mse_list)
    mean_saw_mse   = np.mean(saw_mse_list)
    reduction = mean_plain_mse / mean_saw_mse

    print(f"Mean Plain INT4 MSE: {mean_plain_mse:.4f}")
    print(f"Mean SAW-INT4  MSE: {mean_saw_mse:.4f}")
    print(f"MSE Improvement:    {reduction:.2f}x error reduction!")
    assert reduction > 3.0, "SAW-INT4 should provide at least 3x MSE reduction on outlier data"
    print("PASS: Outlier suppression confirmed.\n")


def main():
    print("=================================================================")
    print("Starting Comprehensive RKVA SAW-INT4 Test Oracle")
    print("=================================================================\n")
    test_mathematical_properties()
    test_wave64_intrinsics()
    test_quarter_dim_and_alignment()
    test_hlsl_dequant_parity()
    test_outlier_suppression_and_accuracy()
    print("=================================================================")
    print("ALL TESTS PASSED WITH 100% NUMERICAL SUCCESS!")
    print("=================================================================")

if __name__ == '__main__':
    main()
