#!/usr/bin/env python3
"""Bit-identical numerical parity verification for Group B quantization codecs (q4_1, q5_1, q6_1).

Compares HLSL rowlane dequantizer logic against golden CPU dequantizers in
ggml/src/ggml-quants.c via ctypes binding to libggml-base.so.
"""

import argparse
import ctypes
import os
from pathlib import Path
import struct
import sys
import numpy as np


# Find and load ggml shared library
SO_PATH = Path("/home/hjotha/beellama.cpp/build-optimized/bin")
GGML_SO = SO_PATH / "libggml.so"
GGML_BASE_SO = SO_PATH / "libggml-base.so"

if not GGML_BASE_SO.exists():
    raise FileNotFoundError(f"Cannot find libggml-base.so at {GGML_BASE_SO}")

ctypes.CDLL(str(GGML_SO), mode=ctypes.RTLD_GLOBAL)
lib = ctypes.CDLL(str(GGML_BASE_SO))

dequant_row_q4_1 = lib.dequantize_row_q4_1
dequant_row_q4_1.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
dequant_row_q4_1.restype = None

dequant_row_q5_1 = lib.dequantize_row_q5_1
dequant_row_q5_1.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
dequant_row_q5_1.restype = None

dequant_row_q6_1 = lib.dequantize_row_q6_1
dequant_row_q6_1.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
dequant_row_q6_1.restype = None


# --- HLSL Emulation Helpers ---

def f16tof32(u16: int) -> np.float32:
    """Emulate HLSL f16tof32 intrinsic: bitcast uint16 to IEEE-754 half then float32."""
    return np.frombuffer(struct.pack("<H", u16 & 0xFFFF), dtype=np.float16)[0].astype(np.float32)


def hlsl_load(buf: bytes, addr: int) -> int:
    """Emulate ByteAddressBuffer.Load(addr)."""
    assert addr % 4 == 0, f"Unaligned Load at byte address {addr}"
    return struct.unpack_from("<I", buf, addr)[0]


def hlsl_load2(buf: bytes, addr: int) -> tuple[int, int]:
    """Emulate ByteAddressBuffer.Load2(addr)."""
    assert addr % 4 == 0, f"Unaligned Load2 at byte address {addr}"
    return struct.unpack_from("<II", buf, addr)


# --- HLSL Functions (bit-identical to rkva_decode_quants_group_b.hlsli) ---

def quarter_dim(u: int) -> int:
    """uint quarter_dim(uint u) { return (u >> 3) * 32 + (u & 1) * 16 + ((u >> 1) & 3) * 4; }"""
    return (u >> 3) * 32 + (u & 1) * 16 + ((u >> 1) & 3) * 4


def quarter_dim_q4_1(u: int) -> int:
    return quarter_dim(u)


def quarter_dim_q5_1(u: int) -> int:
    return quarter_dim(u)


def quarter_dim_q6_1(u: int) -> int:
    return quarter_dim(u)


def load_q4_1(buffer: bytes, base: int, u: int) -> np.ndarray:
    """Emulate float4 load_q4_1(ByteAddressBuffer buffer, uint base, uint u)."""
    b = u >> 3
    block_addr = base + b * 20
    c = (u >> 1) & 3

    dm = hlsl_load(buffer, block_addr)
    d = f16tof32(dm & 0xFFFF)
    m = f16tof32(dm >> 16)

    qs_word = hlsl_load(buffer, block_addr + 4 + c * 4)
    qs_word >>= (u & 1) * 4

    q0 = qs_word & 0x0F
    q1 = (qs_word >> 8) & 0x0F
    q2 = (qs_word >> 16) & 0x0F
    q3 = (qs_word >> 24) & 0x0F

    return np.array([q0 * d + m, q1 * d + m, q2 * d + m, q3 * d + m], dtype=np.float32)


def load_q5_1(buffer: bytes, base: int, u: int) -> np.ndarray:
    """Emulate float4 load_q5_1(ByteAddressBuffer buffer, uint base, uint u)."""
    b = u >> 3
    block_addr = base + b * 24
    c = (u >> 1) & 3

    dm_qh_x, dm_qh_y = hlsl_load2(buffer, block_addr)
    d = f16tof32(dm_qh_x & 0xFFFF)
    m = f16tof32(dm_qh_x >> 16)
    qh = dm_qh_y

    qs_word = hlsl_load(buffer, block_addr + 8 + c * 4)
    qs_word >>= (u & 1) * 4
    qs_low = [
        qs_word & 0x0F,
        (qs_word >> 8) & 0x0F,
        (qs_word >> 16) & 0x0F,
        (qs_word >> 24) & 0x0F,
    ]

    k = (u & 1) * 16 + c * 4
    qh_bits = qh >> k
    qh_high = [
        ((qh_bits >> 0) & 1) << 4,
        ((qh_bits >> 1) & 1) << 4,
        ((qh_bits >> 2) & 1) << 4,
        ((qh_bits >> 3) & 1) << 4,
    ]

    q = [qs_low[i] | qh_high[i] for i in range(4)]
    return np.array([q[0] * d + m, q[1] * d + m, q[2] * d + m, q[3] * d + m], dtype=np.float32)


def load_q6_1(buffer: bytes, base: int, u: int) -> np.ndarray:
    """Emulate float4 load_q6_1(ByteAddressBuffer buffer, uint base, uint u)."""
    b = u >> 3
    block_addr = base + b * 28
    c = (u >> 1) & 3

    dm = hlsl_load(buffer, block_addr)
    d = f16tof32(dm & 0xFFFF)
    m = f16tof32(dm >> 16)

    qh_word = hlsl_load(buffer, block_addr + 4 + (c & 1) * 4)
    qs_word = hlsl_load(buffer, block_addr + 12 + c * 4)

    qs_word >>= (u & 1) * 4
    qs_low = [
        qs_word & 0x0F,
        (qs_word >> 8) & 0x0F,
        (qs_word >> 16) & 0x0F,
        (qs_word >> 24) & 0x0F,
    ]

    qh_bytes = [
        qh_word & 0xFF,
        (qh_word >> 8) & 0xFF,
        (qh_word >> 16) & 0xFF,
        (qh_word >> 24) & 0xFF,
    ]
    shift = ((c & 2) + (u & 1)) * 2
    qh_high = [((byte >> shift) & 0x03) << 4 for byte in qh_bytes]

    q = [qs_low[i] | qh_high[i] for i in range(4)]
    return np.array([q[0] * d + m, q[1] * d + m, q[2] * d + m, q[3] * d + m], dtype=np.float32)


# --- Verification Tests ---

def test_quarter_dim_mapping():
    """Verify quarter_dim covers [0, 255] exactly once."""
    for fn, name in [(quarter_dim_q4_1, "q4_1"), (quarter_dim_q5_1, "q5_1"), (quarter_dim_q6_1, "q6_1")]:
        dims = sorted([fn(u) + i for u in range(64) for i in range(4)])
        assert dims == list(range(256)), f"{name} quarter mapping failed coverage of 256 dims"
    print("PASS: quarter_dim mappings cover all 256 dimensions exactly once.")


def test_q4_1_parity(iterations: int = 100):
    """Verify load_q4_1 against dequantize_row_q4_1."""
    rng = np.random.default_rng(1001)
    max_err = 0.0

    for i in range(iterations):
        base_offset = int(rng.choice([0, 16, 64, 128, 512, 1024]))
        buffer = bytearray(b"\xAA" * base_offset)

        # Generate 8 blocks of 20 bytes
        for _ in range(8):
            d = np.float16(rng.uniform(-2.5, 2.5) * rng.choice([-1.0, 1.0]))
            m = np.float16(rng.uniform(-5.0, 5.0) * rng.choice([-1.0, 1.0]))
            buffer += d.tobytes()
            buffer += m.tobytes()
            buffer += bytes(rng.integers(0, 256, 16, dtype=np.uint8).tolist())

        # Trailing padding
        buffer += b"\x55" * 64
        raw_bytes = bytes(buffer)

        # Run golden CPU dequantizer on slice at base_offset
        y_golden = np.zeros(256, dtype=np.float32)
        slice_bytes = raw_bytes[base_offset : base_offset + 160]
        dequant_row_q4_1(
            slice_bytes,
            y_golden.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            256,
        )

        # Run HLSL emulation
        y_hlsl = np.zeros(256, dtype=np.float32)
        for u in range(64):
            d_base = quarter_dim_q4_1(u)
            y_hlsl[d_base : d_base + 4] = load_q4_1(raw_bytes, base_offset, u)

        err = float(np.max(np.abs(y_golden - y_hlsl)))
        max_err = max(max_err, err)
        assert err == 0.0, f"Q4_1 iteration {i} mismatch: max diff = {err}"

    print(f"PASS: Q4_1 verified across {iterations} rows ({iterations * 256:,} elements). Max diff: {max_err:.1e}")


def test_q5_1_parity(iterations: int = 100):
    """Verify load_q5_1 against dequantize_row_q5_1."""
    rng = np.random.default_rng(2002)
    max_err = 0.0

    for i in range(iterations):
        base_offset = int(rng.choice([0, 16, 64, 128, 512, 1024]))
        buffer = bytearray(b"\xAA" * base_offset)

        # Generate 8 blocks of 24 bytes
        for _ in range(8):
            d = np.float16(rng.uniform(-2.5, 2.5) * rng.choice([-1.0, 1.0]))
            m = np.float16(rng.uniform(-5.0, 5.0) * rng.choice([-1.0, 1.0]))
            buffer += d.tobytes()
            buffer += m.tobytes()
            buffer += bytes(rng.integers(0, 256, 4, dtype=np.uint8).tolist())   # qh
            buffer += bytes(rng.integers(0, 256, 16, dtype=np.uint8).tolist())  # qs

        buffer += b"\x55" * 64
        raw_bytes = bytes(buffer)

        # Run golden CPU dequantizer on slice at base_offset
        y_golden = np.zeros(256, dtype=np.float32)
        slice_bytes = raw_bytes[base_offset : base_offset + 192]
        dequant_row_q5_1(
            slice_bytes,
            y_golden.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            256,
        )

        # Run HLSL emulation
        y_hlsl = np.zeros(256, dtype=np.float32)
        for u in range(64):
            d_base = quarter_dim_q5_1(u)
            y_hlsl[d_base : d_base + 4] = load_q5_1(raw_bytes, base_offset, u)

        err = float(np.max(np.abs(y_golden - y_hlsl)))
        max_err = max(max_err, err)
        assert err == 0.0, f"Q5_1 iteration {i} mismatch: max diff = {err}"

    print(f"PASS: Q5_1 verified across {iterations} rows ({iterations * 256:,} elements). Max diff: {max_err:.1e}")


def test_q6_1_parity(iterations: int = 100):
    """Verify load_q6_1 against dequantize_row_q6_1."""
    rng = np.random.default_rng(3003)
    max_err = 0.0

    for i in range(iterations):
        base_offset = int(rng.choice([0, 16, 64, 128, 512, 1024]))
        buffer = bytearray(b"\xAA" * base_offset)

        # Generate 8 blocks of 28 bytes
        for _ in range(8):
            d = np.float16(rng.uniform(-2.5, 2.5) * rng.choice([-1.0, 1.0]))
            m = np.float16(rng.uniform(-5.0, 5.0) * rng.choice([-1.0, 1.0]))
            buffer += d.tobytes()
            buffer += m.tobytes()
            buffer += bytes(rng.integers(0, 256, 8, dtype=np.uint8).tolist())   # qh
            buffer += bytes(rng.integers(0, 256, 16, dtype=np.uint8).tolist())  # qs

        buffer += b"\x55" * 64
        raw_bytes = bytes(buffer)

        # Run golden CPU dequantizer on slice at base_offset
        y_golden = np.zeros(256, dtype=np.float32)
        slice_bytes = raw_bytes[base_offset : base_offset + 224]
        dequant_row_q6_1(
            slice_bytes,
            y_golden.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            256,
        )

        # Run HLSL emulation
        y_hlsl = np.zeros(256, dtype=np.float32)
        for u in range(64):
            d_base = quarter_dim_q6_1(u)
            y_hlsl[d_base : d_base + 4] = load_q6_1(raw_bytes, base_offset, u)

        err = float(np.max(np.abs(y_golden - y_hlsl)))
        max_err = max(max_err, err)
        assert err == 0.0, f"Q6_1 iteration {i} mismatch: max diff = {err}"

    print(f"PASS: Q6_1 verified across {iterations} rows ({iterations * 256:,} elements). Max diff: {max_err:.1e}")


def test_edge_cases():
    """Verify edge cases: all-zeros, all-ones, alternating bits, zero scale."""
    for fmt, row_size, fn_hlsl, fn_c, fn_dim in [
        ("q4_1", 160, load_q4_1, dequant_row_q4_1, quarter_dim_q4_1),
        ("q5_1", 192, load_q5_1, dequant_row_q5_1, quarter_dim_q5_1),
        ("q6_1", 224, load_q6_1, dequant_row_q6_1, quarter_dim_q6_1),
    ]:
        # Edge case 1: all zeroes
        buf = bytes([0] * row_size)
        y_c = np.zeros(256, dtype=np.float32)
        fn_c(buf, y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 256)
        y_h = np.zeros(256, dtype=np.float32)
        for u in range(64):
            d = fn_dim(u)
            y_h[d : d + 4] = fn_hlsl(buf, 0, u)
        assert np.array_equal(y_c, y_h), f"{fmt} all-zeros edge case mismatch"

        # Edge case 2: all 0xFF
        buf = bytes([0xFF] * row_size)
        y_c = np.zeros(256, dtype=np.float32)
        fn_c(buf, y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 256)
        y_h = np.zeros(256, dtype=np.float32)
        for u in range(64):
            d = fn_dim(u)
            y_h[d : d + 4] = fn_hlsl(buf, 0, u)
        # Note: 0xFFFF in fp16 is NaN, so check bits or isnan
        assert np.array_equal(y_c.view(np.uint32), y_h.view(np.uint32)), f"{fmt} all-ones edge case mismatch"

        # Edge case 3: non-trivial scales with alternating byte patterns
        pat = bytes([0x55, 0xAA] * (row_size // 2))
        y_c = np.zeros(256, dtype=np.float32)
        fn_c(pat, y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 256)
        y_h = np.zeros(256, dtype=np.float32)
        for u in range(64):
            d = fn_dim(u)
            y_h[d : d + 4] = fn_hlsl(pat, 0, u)
        assert np.array_equal(y_c.view(np.uint32), y_h.view(np.uint32)), f"{fmt} alternating pattern mismatch"

    print("PASS: Edge cases (all-zeros, all-ones NaN, alternating patterns) match bit-identically.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100, help="Number of random rows to test per format")
    args = parser.parse_args()

    print(f"=== Running Group B Quantization Verification ({args.iterations} rows per format) ===")
    test_quarter_dim_mapping()
    test_edge_cases()
    test_q4_1_parity(args.iterations)
    test_q5_1_parity(args.iterations)
    test_q6_1_parity(args.iterations)
    print("=== ALL GROUP B PARITY CHECKS PASSED (Max diff = 0.0) ===")


if __name__ == "__main__":
    main()
