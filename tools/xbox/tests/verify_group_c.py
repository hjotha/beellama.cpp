#!/usr/bin/env python3
"""
Numerical parity verification for Group C rowlane dequantization formats for RKVA:
1. iq4_nl  (144 bytes / 256 dims: 8 blocks of 18 bytes. d fp16 + 16 bytes qs)
2. kvarn2 (q2_0s) (80 bytes / 256 dims: 8 blocks of 10 bytes. d fp16 + 8 bytes qs)
3. kvarn3 (q3_0)  (112 bytes / 256 dims: 8 blocks of 14 bytes. d fp16 + 4 bytes qh + 8 bytes qs)
4. kvarn8 (q8_0)  (272 bytes / 256 dims: 8 blocks of 34 bytes. d fp16 + 32 bytes qs int8)

Compares bit-identical Python emulation of the HLSL functions in
tools/xbox/shaders/rkva/rkva_decode_quants_group_c.hlsli against the golden
compiled CPU dequantizers in libggml-base.so (ggml-quants.c).
"""

import os
import sys
import struct
import ctypes
import numpy as np

# ============================================================================
# Ctypes Loader for Golden CPU Dequantizers
# ============================================================================

GOLDEN_LIB = None

def load_golden_lib():
    global GOLDEN_LIB
    if GOLDEN_LIB is not None:
        return GOLDEN_LIB

    candidate_paths = [
        "/home/hjotha/beellama.cpp/build-optimized/bin",
        "/home/hjotha/beellama.cpp/build/bin",
    ]
    for p in candidate_paths:
        base_so = os.path.join(p, "libggml-base.so")
        ggml_so = os.path.join(p, "libggml.so")
        if os.path.exists(base_so):
            try:
                if os.path.exists(ggml_so):
                    ctypes.CDLL(ggml_so, mode=ctypes.RTLD_GLOBAL)
                GOLDEN_LIB = ctypes.CDLL(base_so)
                break
            except Exception as e:
                print(f"Warning: failed loading {base_so}: {e}", file=sys.stderr)

    if GOLDEN_LIB is None:
        raise RuntimeError("Could not load golden libggml-base.so from build paths")

    # Set up argtypes
    GOLDEN_LIB.dequantize_row_iq4_nl.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    GOLDEN_LIB.dequantize_row_q2_0s.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    GOLDEN_LIB.dequantize_row_q3_0.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    GOLDEN_LIB.dequantize_row_q8_0.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]

    return GOLDEN_LIB

# ============================================================================
# HLSL ByteAddressBuffer Emulation
# ============================================================================

class ByteAddressBuffer:
    def __init__(self, data: bytes):
        self.data = bytearray(data)

    def Load(self, addr: int) -> int:
        assert addr % 4 == 0, f"Unaligned Load at address {addr}"
        assert addr + 4 <= len(self.data), f"OOB Load at address {addr}+4 > {len(self.data)}"
        return struct.unpack_from('<I', self.data, addr)[0]

    def Load2(self, addr: int) -> tuple:
        assert addr % 4 == 0, f"Unaligned Load2 at address {addr}"
        assert addr + 8 <= len(self.data), f"OOB Load2 at address {addr}+8 > {len(self.data)}"
        return struct.unpack_from('<II', self.data, addr)

def f16tof32(u16: int) -> float:
    return float(np.frombuffer(struct.pack('<H', u16 & 0xFFFF), dtype=np.float16).astype(np.float32)[0])

def asint32(u32: int) -> int:
    val = u32 & 0xFFFFFFFF
    if val >= 0x80000000:
        val -= 0x100000000
    return val

# ============================================================================
# Group C HLSL Logic (Bit-Identical Emulation of rkva_decode_quants_group_c.hlsli)
# ============================================================================

# 1. IQ4_NL
kvalues_iq4nl = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
       1.0,   13.0,  25.0,  38.0,  53.0,  69.0,  89.0, 113.0
]

def quarter_dim_iq4_nl(u: int) -> int:
    return (u >> 3) * 32 + (u & 1) * 16 + ((u >> 1) & 3) * 4

def load_iq4_nl(buffer: ByteAddressBuffer, base: int, u: int) -> tuple:
    pair_idx = u >> 4
    addr_pair = base + pair_idx * 36
    is_odd = (u >> 3) & 1
    c = (u >> 1) & 3

    scale_word = buffer.Load(addr_pair + is_odd * 16)
    scale = f16tof32((scale_word >> (is_odd * 16)) & 65535)

    if is_odd:
        word = buffer.Load(addr_pair + 20 + c * 4)
    else:
        p_x, p_y = buffer.Load2(addr_pair + c * 4)
        word = ((p_x >> 16) | (p_y << 16)) & 0xFFFFFFFF

    word >>= (u & 1) * 4
    q0 = word & 15
    q1 = (word >> 8) & 15
    q2 = (word >> 16) & 15
    q3 = (word >> 24) & 15

    return (
        kvalues_iq4nl[q0] * scale,
        kvalues_iq4nl[q1] * scale,
        kvalues_iq4nl[q2] * scale,
        kvalues_iq4nl[q3] * scale
    )

# 2. Q2_0S (kvarn2)
def quarter_dim_q2_0s(u: int) -> int:
    return u * 4

def load_q2_0s(buffer: ByteAddressBuffer, base: int, u: int) -> tuple:
    pair_idx = u >> 4
    pair_base = base + pair_idx * 20
    is_odd = (u >> 3) & 1
    plane = (u >> 1) & 3
    h = u & 1

    if is_odd:
        scale_word = buffer.Load(pair_base + 8)
        scale = f16tof32((scale_word >> 16) & 65535)
        word = buffer.Load(pair_base + 12 + h * 4)
    else:
        scale_word = buffer.Load(pair_base)
        scale = f16tof32(scale_word & 65535)
        p_x, p_y = buffer.Load2(pair_base + h * 4)
        word = ((p_x >> 16) | (p_y << 16)) & 0xFFFFFFFF

    w = word >> (plane * 2)
    q0 = w & 3
    q1 = (w >> 8) & 3
    q2 = (w >> 16) & 3
    q3 = (w >> 24) & 3

    return (
        (q0 - 2.0) * scale,
        (q1 - 2.0) * scale,
        (q2 - 2.0) * scale,
        (q3 - 2.0) * scale
    )

# 3. Q3_0 (kvarn3)
def quarter_dim_q3_0(u: int) -> int:
    return u * 4

def load_q3_0(buffer: ByteAddressBuffer, base: int, u: int) -> tuple:
    pair_idx = u >> 4
    pair_base = base + pair_idx * 28
    is_odd = (u >> 3) & 1
    plane = (u >> 1) & 3
    h = u & 1

    if is_odd:
        scale_word = buffer.Load(pair_base + 12)
        scale = f16tof32((scale_word >> 16) & 65535)
        qh = buffer.Load(pair_base + 16)
        word = buffer.Load(pair_base + 20 + h * 4)
    else:
        p0_x, p0_y = buffer.Load2(pair_base)
        scale = f16tof32(p0_x & 65535)
        qh = ((p0_x >> 16) | (p0_y << 16)) & 0xFFFFFFFF
        p_x, p_y = buffer.Load2(pair_base + 4 + h * 4)
        word = ((p_x >> 16) | (p_y << 16)) & 0xFFFFFFFF

    w = word >> (plane * 2)
    q0_low = w & 3
    q1_low = (w >> 8) & 3
    q2_low = (w >> 16) & 3
    q3_low = (w >> 24) & 3

    j0 = (u & 7) * 4
    qh_bits = (qh >> j0) & 15
    q0_high = (qh_bits >> 0) & 1
    q1_high = (qh_bits >> 1) & 1
    q2_high = (qh_bits >> 2) & 1
    q3_high = (qh_bits >> 3) & 1

    q0 = q0_low + q0_high * 4.0
    q1 = q1_low + q1_high * 4.0
    q2 = q2_low + q2_high * 4.0
    q3 = q3_low + q3_high * 4.0

    return (
        (q0 - 4.0) * scale,
        (q1 - 4.0) * scale,
        (q2 - 4.0) * scale,
        (q3 - 4.0) * scale
    )

# 4. Q8_0 (kvarn8)
def quarter_dim_q8_0(u: int) -> int:
    return u * 4

def load_q8_0(buffer: ByteAddressBuffer, base: int, u: int) -> tuple:
    pair_idx = u >> 4
    pair_base = base + pair_idx * 68
    is_odd = (u >> 3) & 1
    c = u & 7

    if is_odd:
        scale_word = buffer.Load(pair_base + 32)
        scale = f16tof32((scale_word >> 16) & 65535)
        word = buffer.Load(pair_base + 36 + c * 4)
    else:
        scale_word = buffer.Load(pair_base)
        scale = f16tof32(scale_word & 65535)
        p_x, p_y = buffer.Load2(pair_base + c * 4)
        word = ((p_x >> 16) | (p_y << 16)) & 0xFFFFFFFF

    q0 = asint32(word << 24) >> 24
    q1 = asint32(word << 16) >> 24
    q2 = asint32(word << 8) >> 24
    q3 = asint32(word) >> 24

    return (
        float(q0) * scale,
        float(q1) * scale,
        float(q2) * scale,
        float(q3) * scale
    )

# ============================================================================
# Verification Tests
# ============================================================================

def verify_quarter_dims():
    """Verify that quarter_dim functions for all formats map 64 units to 256 unique dimensions."""
    print("Verifying quarter dimension mappings (64 units -> 256 unique dims)...")
    for name, fn in [
        ("iq4_nl", quarter_dim_iq4_nl),
        ("q2_0s", quarter_dim_q2_0s),
        ("q3_0", quarter_dim_q3_0),
        ("q8_0", quarter_dim_q8_0),
    ]:
        dims = set()
        for u in range(64):
            d0 = fn(u)
            assert d0 % 4 == 0, f"{name}: quarter_dim({u}) = {d0} is not a multiple of 4"
            for i in range(4):
                dim = d0 + i
                assert 0 <= dim < 256, f"{name}: dim {dim} out of range [0, 255]"
                assert dim not in dims, f"{name}: duplicate dim {dim}"
                dims.add(dim)
        assert len(dims) == 256, f"{name}: only covered {len(dims)} dimensions, expected 256"
        print(f"  [OK] {name}: all 256 dimensions uniquely covered in multiples of 4")

def test_quarter_dims():
    verify_quarter_dims()

def test_iq4_nl_parity():
    lib = load_golden_lib()
    _test_format("iq4_nl", 144, 18, lib.dequantize_row_iq4_nl, quarter_dim_iq4_nl, load_iq4_nl)

def test_q2_0s_parity():
    lib = load_golden_lib()
    _test_format("q2_0s (kvarn2)", 80, 10, lib.dequantize_row_q2_0s, quarter_dim_q2_0s, load_q2_0s)

def test_q3_0_parity():
    lib = load_golden_lib()
    _test_format("q3_0 (kvarn3)", 112, 14, lib.dequantize_row_q3_0, quarter_dim_q3_0, load_q3_0)

def test_q8_0_parity():
    lib = load_golden_lib()
    _test_format("q8_0 (kvarn8)", 272, 34, lib.dequantize_row_q8_0, quarter_dim_q8_0, load_q8_0)

def _test_format(name, row_bytes, block_bytes, golden_fn, dim_fn, load_fn, num_trials=50):
    np.random.seed(1337)
    max_diff_all = 0.0

    for trial in range(num_trials):
        raw = bytearray(np.random.randint(0, 256, row_bytes, dtype=np.uint8))

        for b in range(8):
            if trial == 0:
                scale_val = 0.0
            elif trial == 1:
                scale_val = 1.0
            elif trial == 2:
                scale_val = -1.0
            else:
                scale_val = float(np.random.uniform(-4.0, 4.0))
            scale_fp16 = np.float16(scale_val)
            scale_u16 = np.frombuffer(scale_fp16.tobytes(), dtype=np.uint16)[0]
            struct.pack_into('<H', raw, b * block_bytes, scale_u16)

        y_golden = (ctypes.c_float * 256)()
        raw_c_arr = (ctypes.c_uint8 * row_bytes).from_buffer_copy(raw)
        golden_fn(raw_c_arr, y_golden, 256)
        golden_arr = np.array(y_golden, dtype=np.float32)

        base_offset = (trial % 4) * 64
        padded_buffer = bytearray(base_offset + row_bytes + 128)
        padded_buffer[base_offset:base_offset + row_bytes] = raw
        buf = ByteAddressBuffer(bytes(padded_buffer))

        hlsl_arr = np.zeros(256, dtype=np.float32)
        for u in range(64):
            d0 = dim_fn(u)
            floats4 = load_fn(buf, base_offset, u)
            hlsl_arr[d0:d0+4] = floats4

        diff = np.max(np.abs(golden_arr - hlsl_arr))
        max_diff_all = max(max_diff_all, diff)

        assert diff == 0.0, (
            f"Mismatch in {name} at trial {trial} (base={base_offset}): "
            f"max_diff={diff:.6e}, "
            f"golden={golden_arr[:4]}, hlsl={hlsl_arr[:4]}"
        )

    print(f"  [PASS] {name:<16}: {num_trials} trials passed, max difference = {max_diff_all:.1e}")

def run_tests():
    verify_quarter_dims()
    test_iq4_nl_parity()
    test_q2_0s_parity()
    test_q3_0_parity()
    test_q8_0_parity()
    print("\nSUCCESS: All Group C formats achieved 100% bit-exact parity (max diff = 0.0) across all 256 dimensions!")

if __name__ == "__main__":
    run_tests()
