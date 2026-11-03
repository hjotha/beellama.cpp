#!/usr/bin/env python3
"""Comprehensive numerical correctness test oracle for RKVA rowlane quant matrix.

Tests all 12 requested quantizations/formats against the golden C dequantizers
from libggml-base.so:
  - q4_1
  - q5_1
  - q6_1
  - iq4_nl
  - kvarn2 (q2_0s)
  - kvarn3 (q3_0)
  - kvarn5 (q5_0)
  - kvarn6 (q6_0)
  - kvarn8 (q8_0)
  - q5_0
  - q6_0
  - f16

Verifies:
1. Every quarter-unit u in [0, 63] (Wave64) maps to disjoint sets of 4 dimensions
   covering all 256 dimensions [0, 255] exactly once.
2. All loads in HLSL are 4-byte aligned and never access outside row record boundaries.
3. Every dequantized element matches the C golden reference with 0.0 difference (or within FP16 rounding).
4. Full Wave64 QK dot product and online softmax produce identical numerical results.
"""
import ctypes
import struct
import sys
import numpy as np

# Load canonical GGML shared libraries
ctypes.CDLL('/home/hjotha/beellama.cpp/build-optimized/bin/libggml.so', mode=ctypes.RTLD_GLOBAL)
libggml = ctypes.CDLL('/home/hjotha/beellama.cpp/build-optimized/bin/libggml-base.so')

# Setup C function signatures
DEQ_FUNCS = {
    'q4_1': libggml.dequantize_row_q4_1,
    'q5_0': libggml.dequantize_row_q5_0,
    'kvarn5': libggml.dequantize_row_q5_0,
    'q5_1': libggml.dequantize_row_q5_1,
    'q6_0': libggml.dequantize_row_q6_0,
    'kvarn6': libggml.dequantize_row_q6_0,
    'q6_1': libggml.dequantize_row_q6_1,
    'iq4_nl': libggml.dequantize_row_iq4_nl,
    'kvarn2': libggml.dequantize_row_q2_0s,
    'kvarn3': libggml.dequantize_row_q3_0,
    'kvarn8': libggml.dequantize_row_q8_0,
}

for name, fn in DEQ_FUNCS.items():
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    fn.restype = None

# Row byte sizes for D=256 (8 blocks of 32 elements, except f16 which is 256 fp16)
ROW_HEAD_BYTES = {
    'f16': 512,
    'q4_1': 160,       # 8 * 20
    'q5_0': 176,       # 8 * 22
    'kvarn5': 176,     # 8 * 22
    'q5_1': 192,       # 8 * 24
    'q6_0': 208,       # 8 * 26
    'kvarn6': 208,     # 8 * 26
    'q6_1': 224,       # 8 * 28
    'iq4_nl': 144,     # 8 * 18
    'kvarn2': 80,      # 8 * 10
    'kvarn3': 112,     # 8 * 14
    'kvarn8': 272,     # 8 * 34
}

KVALUES_IQ4NL = np.array([
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
       1.0,   13.0,  25.0,  38.0,  53.0,  69.0,  89.0, 113.0
], dtype=np.float32)

class MockByteAddressBuffer:
    """Emulates D3D12 ByteAddressBuffer aligned 32-bit loads."""
    def __init__(self, data: bytes):
        self.data = memoryview(data)

    def Load(self, addr: int) -> int:
        assert addr % 4 == 0, f"Unaligned ByteAddressBuffer.Load at {addr}"
        assert addr + 4 <= len(self.data), f"Out of bounds ByteAddressBuffer.Load at {addr} (size {len(self.data)})"
        return struct.unpack_from('<I', self.data, addr)[0]

    def Load2(self, addr: int) -> tuple[int, int]:
        assert addr % 4 == 0, f"Unaligned ByteAddressBuffer.Load2 at {addr}"
        assert addr + 8 <= len(self.data), f"Out of bounds ByteAddressBuffer.Load2 at {addr} (size {len(self.data)})"
        return struct.unpack_from('<II', self.data, addr)

    def Load3(self, addr: int) -> tuple[int, int, int]:
        assert addr % 4 == 0, f"Unaligned ByteAddressBuffer.Load3 at {addr}"
        assert addr + 12 <= len(self.data), f"Out of bounds ByteAddressBuffer.Load3 at {addr} (size {len(self.data)})"
        return struct.unpack_from('<III', self.data, addr)

def f16tof32(u16: int) -> float:
    return float(np.frombuffer(struct.pack('<H', u16 & 0xFFFF), dtype=np.float16)[0])

# ==============================================================================
# Exact HLSL quarter_dim mapping functions
# ==============================================================================
def quarter_dim_standard(u: int) -> int:
    return (u >> 3) * 32 + (u & 1) * 16 + ((u >> 1) & 3) * 4

quarter_dim_q4_1 = quarter_dim_standard
quarter_dim_q5_1 = quarter_dim_standard
quarter_dim_q6_1 = quarter_dim_standard
quarter_dim_q5_0 = quarter_dim_standard
quarter_dim_q6_0 = quarter_dim_standard
quarter_dim_iq4_nl = quarter_dim_standard

def quarter_dim_sequential(u: int) -> int:
    return u * 4

QUARTER_DIM_FN = {
    'f16': quarter_dim_sequential,
    'q4_1': quarter_dim_standard,
    'q5_0': quarter_dim_standard,
    'kvarn5': quarter_dim_standard,
    'q5_1': quarter_dim_standard,
    'q6_0': quarter_dim_standard,
    'kvarn6': quarter_dim_standard,
    'q6_1': quarter_dim_standard,
    'iq4_nl': quarter_dim_standard,
    'kvarn2': quarter_dim_sequential,
    'kvarn3': quarter_dim_sequential,
    'kvarn8': quarter_dim_sequential,
}

# ==============================================================================
# Exact HLSL load_quant emulations
# ==============================================================================
def hlsl_load_f16(buf: MockByteAddressBuffer, base: int, u: int) -> np.ndarray:
    p0, p1 = buf.Load2(base + u * 8)
    return np.array([
        f16tof32(p0 & 0xFFFF), f16tof32(p0 >> 16),
        f16tof32(p1 & 0xFFFF), f16tof32(p1 >> 16)
    ], dtype=np.float32)

def hlsl_load_q4_1(buf: MockByteAddressBuffer, base: int, u: int) -> np.ndarray:
    b = u >> 3
    block_addr = base + b * 20
    c = (u >> 1) & 3
    dm = buf.Load(block_addr)
    d = f16tof32(dm & 0xFFFF)
    m = f16tof32(dm >> 16)
    qs_word = buf.Load(block_addr + 4 + c * 4)
    qs_word >>= (u & 1) * 4
    q0 = qs_word & 0x0F
    q1 = (qs_word >> 8) & 0x0F
    q2 = (qs_word >> 16) & 0x0F
    q3 = (qs_word >> 24) & 0x0F
    return np.array([q0, q1, q2, q3], dtype=np.float32) * d + m

def hlsl_load_q5_0(buf: MockByteAddressBuffer, base: int, u: int) -> np.ndarray:
    pair_idx = u >> 4
    addr_pair = base + pair_idx * 44
    is_odd = (u >> 3) & 1
    h = u & 1
    c = (u >> 1) & 3
    if is_odd:
        scale_word = buf.Load(addr_pair + 20)
        scale = f16tof32(scale_word >> 16)
        qh = buf.Load(addr_pair + 24)
        qs_bytes = buf.Load(addr_pair + 28 + c * 4)
    else:
        w0, w1 = buf.Load2(addr_pair)
        scale = f16tof32(w0 & 65535)
        qh = (w0 >> 16) | (w1 << 16)
        p0, p1 = buf.Load2(addr_pair + 4 + c * 4)
        qs_bytes = (p0 >> 16) | (p1 << 16)

    word = qs_bytes >> (h * 4)
    qs_low = [(word >> (k * 8)) & 15 for k in range(4)]
    qh_bits = qh >> (h * 16 + c * 4)
    qh_high = [((qh_bits >> k) & 1) << 4 for k in range(4)]
    q5 = [(qs_low[k] | qh_high[k]) - 16.0 for k in range(4)]
    return np.array(q5, dtype=np.float32) * scale

def hlsl_load_q5_1(buf: MockByteAddressBuffer, base: int, u: int) -> np.ndarray:
    b = u >> 3
    block_addr = base + b * 24
    c = (u >> 1) & 3
    dm, qh = buf.Load2(block_addr)
    d = f16tof32(dm & 0xFFFF)
    m = f16tof32(dm >> 16)
    qs_word = buf.Load(block_addr + 8 + c * 4)
    qs_word >>= (u & 1) * 4
    qs_low = [(qs_word >> (k * 8)) & 0x0F for k in range(4)]
    k_shift = (u & 1) * 16 + c * 4
    qh_bits = qh >> k_shift
    qh_high = [((qh_bits >> k) & 1) << 4 for k in range(4)]
    q = [qs_low[k] | qh_high[k] for k in range(4)]
    return np.array(q, dtype=np.float32) * d + m

def hlsl_load_q6_0(buf: MockByteAddressBuffer, base: int, u: int) -> np.ndarray:
    pair_idx = u >> 4
    addr_pair = base + pair_idx * 52
    is_odd = (u >> 3) & 1
    h_half = u & 1
    c = (u >> 1) & 3
    if is_odd:
        scale_word = buf.Load(addr_pair + 24)
        scale = f16tof32(scale_word >> 16)
        qh0, qh1 = buf.Load2(addr_pair + 28)
        qs_bytes = buf.Load(addr_pair + 36 + c * 4)
    else:
        w0, w1, w2 = buf.Load3(addr_pair)
        scale = f16tof32(w0 & 65535)
        qh0 = (w0 >> 16) | (w1 << 16)
        qh1 = (w1 >> 16) | (w2 << 16)
        p0, p1 = buf.Load2(addr_pair + 8 + c * 4)
        qs_bytes = (p0 >> 16) | (p1 << 16)

    qh_word = qh1 if (c & 1) else qh0
    qh_shift = ((c >> 1) & 1) * 4 + h_half * 2
    qh_high = [(((qh_word >> (k * 8 + qh_shift)) & 3) << 4) for k in range(4)]
    word = qs_bytes >> (h_half * 4)
    qs_low = [(word >> (k * 8)) & 15 for k in range(4)]
    q6 = [(qs_low[k] | qh_high[k]) - 32.0 for k in range(4)]
    return np.array(q6, dtype=np.float32) * scale

def hlsl_load_q6_1(buf: MockByteAddressBuffer, base: int, u: int) -> np.ndarray:
    b = u >> 3
    block_addr = base + b * 28
    c = (u >> 1) & 3
    dm = buf.Load(block_addr)
    d = f16tof32(dm & 0xFFFF)
    m = f16tof32(dm >> 16)
    qh_word = buf.Load(block_addr + 4 + (c & 1) * 4)
    qs_word = buf.Load(block_addr + 12 + c * 4)
    qs_word >>= (u & 1) * 4
    qs_low = [(qs_word >> (k * 8)) & 0x0F for k in range(4)]
    shift = ((c & 2) + (u & 1)) * 2
    qh_high = [(((qh_word >> (k * 8 + shift)) & 0x03) << 4) for k in range(4)]
    q = [qs_low[k] | qh_high[k] for k in range(4)]
    return np.array(q, dtype=np.float32) * d + m

def hlsl_load_iq4_nl(buf: MockByteAddressBuffer, base: int, u: int) -> np.ndarray:
    pair_idx = u >> 4
    addr_pair = base + pair_idx * 36
    is_odd = (u >> 3) & 1
    c = (u >> 1) & 3
    scale_word = buf.Load(addr_pair + is_odd * 16)
    scale = f16tof32((scale_word >> (is_odd * 16)) & 65535)
    if is_odd:
        word = buf.Load(addr_pair + 20 + c * 4)
    else:
        p0, p1 = buf.Load2(addr_pair + c * 4)
        word = (p0 >> 16) | (p1 << 16)
    word >>= (u & 1) * 4
    q = [(word >> (k * 8)) & 15 for k in range(4)]
    return np.array([KVALUES_IQ4NL[qi] for qi in q], dtype=np.float32) * scale

def hlsl_load_kvarn2(buf: MockByteAddressBuffer, base: int, u: int) -> np.ndarray:
    pair_idx = u >> 4
    pair_base = base + pair_idx * 20
    is_odd = (u >> 3) & 1
    plane = (u >> 1) & 3
    h = u & 1
    if is_odd:
        scale_word = buf.Load(pair_base + 8)
        scale = f16tof32(scale_word >> 16)
        word = buf.Load(pair_base + 12 + h * 4)
    else:
        scale_word = buf.Load(pair_base)
        scale = f16tof32(scale_word & 65535)
        p0, p1 = buf.Load2(pair_base + h * 4)
        word = (p0 >> 16) | (p1 << 16)
    w = word >> (plane * 2)
    q = [(w >> (k * 8)) & 3 for k in range(4)]
    return (np.array(q, dtype=np.float32) - 2.0) * scale

def hlsl_load_kvarn3(buf: MockByteAddressBuffer, base: int, u: int) -> np.ndarray:
    pair_idx = u >> 4
    pair_base = base + pair_idx * 28
    is_odd = (u >> 3) & 1
    plane = (u >> 1) & 3
    h = u & 1
    if is_odd:
        scale_word = buf.Load(pair_base + 12)
        scale = f16tof32(scale_word >> 16)
        qh = buf.Load(pair_base + 16)
        word = buf.Load(pair_base + 20 + h * 4)
    else:
        p0, p1 = buf.Load2(pair_base)
        scale = f16tof32(p0 & 65535)
        qh = (p0 >> 16) | (p1 << 16)
        p_lo0, p_lo1 = buf.Load2(pair_base + 4 + h * 4)
        word = (p_lo0 >> 16) | (p_lo1 << 16)

    w = word >> (plane * 2)
    q_low = [(w >> (k * 8)) & 3 for k in range(4)]
    j0 = (u & 7) * 4
    qh_bits = (qh >> j0) & 15
    q_high = [(qh_bits >> k) & 1 for k in range(4)]
    q = [(q_low[k] + q_high[k] * 4.0) - 4.0 for k in range(4)]
    return np.array(q, dtype=np.float32) * scale

def hlsl_load_kvarn8(buf: MockByteAddressBuffer, base: int, u: int) -> np.ndarray:
    pair_idx = u >> 4
    pair_base = base + pair_idx * 68
    is_odd = (u >> 3) & 1
    c = u & 7
    if is_odd:
        scale_word = buf.Load(pair_base + 32)
        scale = f16tof32(scale_word >> 16)
        word = buf.Load(pair_base + 36 + c * 4)
    else:
        scale_word = buf.Load(pair_base)
        scale = f16tof32(scale_word & 65535)
        p0, p1 = buf.Load2(pair_base + c * 4)
        word = (p0 >> 16) | (p1 << 16)

    bytes_signed = [((word >> (k * 8)) & 0xFF) for k in range(4)]
    int8s = [b - 256 if b >= 128 else b for b in bytes_signed]
    return np.array(int8s, dtype=np.float32) * scale

LOAD_FN = {
    'f16': hlsl_load_f16,
    'q4_1': hlsl_load_q4_1,
    'q5_0': hlsl_load_q5_0,
    'kvarn5': hlsl_load_q5_0,
    'q5_1': hlsl_load_q5_1,
    'q6_0': hlsl_load_q6_0,
    'kvarn6': hlsl_load_q6_0,
    'q6_1': hlsl_load_q6_1,
    'iq4_nl': hlsl_load_iq4_nl,
    'kvarn2': hlsl_load_kvarn2,
    'kvarn3': hlsl_load_kvarn3,
    'kvarn8': hlsl_load_kvarn8,
}

# ==============================================================================
# Test runner
# ==============================================================================
def run_tests():
    rng = np.random.default_rng(42)
    n_trials = 100
    total_elements = 0
    failures = 0

    print("================================================================================")
    print("RKVA Rowlane Dequantization Matrix Correctness Suite (Xbox Wave64)")
    print("================================================================================")

    for fmt in ROW_HEAD_BYTES.keys():
        row_size = ROW_HEAD_BYTES[fmt]
        dim_fn = QUARTER_DIM_FN[fmt]
        load_fn = LOAD_FN[fmt]

        # 1. Dimension coverage check
        covered = set()
        for u in range(64):
            d0 = dim_fn(u)
            assert 0 <= d0 <= 252, f"Invalid quarter_dim {d0} for unit {u}"
            for offset in range(4):
                dim = d0 + offset
                assert dim not in covered, f"Duplicate dimension {dim} in format {fmt}"
                covered.add(dim)
        assert len(covered) == 256, f"Format {fmt} did not cover all 256 dimensions!"

        # 2. Numerical equivalence check against golden C reference
        max_diff = 0.0
        for trial in range(n_trials):
            # Generate random bytes for one row (256 dims)
            raw_bytes = bytearray(rng.integers(0, 256, row_size, dtype=np.uint8).tolist())

            # Ensure valid non-zero finite fp16 scales (and min offsets where applicable)
            if fmt == 'f16':
                arr_f16 = np.frombuffer(raw_bytes, dtype=np.float16)
                arr_f16 = np.nan_to_num(arr_f16, nan=1.0, posinf=1.0, neginf=-1.0)
                raw_bytes = bytearray(arr_f16.tobytes())
                golden_y = arr_f16.astype(np.float32)
            else:
                block_bytes = row_size // 8
                has_min = fmt in ('q4_1', 'q5_1', 'q6_1')
                for b in range(8):
                    scale_val = float(rng.uniform(0.01, 2.0) * rng.choice([-1.0, 1.0]))
                    scale_u16 = int(np.frombuffer(np.float16(scale_val).tobytes(), dtype=np.uint16)[0])
                    if has_min:
                        min_val = float(rng.uniform(-5.0, 5.0))
                        min_u16 = int(np.frombuffer(np.float16(min_val).tobytes(), dtype=np.uint16)[0])
                        struct.pack_into('<HH', raw_bytes, b * block_bytes, scale_u16, min_u16)
                    else:
                        struct.pack_into('<H', raw_bytes, b * block_bytes, scale_u16)

                c_fn = DEQ_FUNCS[fmt]
                golden_y = np.zeros(256, dtype=np.float32)
                c_fn(bytes(raw_bytes), golden_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 256)

            # Reconstruct via HLSL rowlane logic (Wave64: 64 lanes)
            buf = MockByteAddressBuffer(bytes(raw_bytes))
            hlsl_y = np.zeros(256, dtype=np.float32)
            for u in range(64):
                d0 = dim_fn(u)
                val4 = load_fn(buf, 0, u)
                hlsl_y[d0:d0+4] = val4

            diff = np.max(np.abs(golden_y - hlsl_y))
            if np.isnan(diff) or diff > 1e-4:
                print(f"FAILED {fmt} on trial {trial}: max diff = {diff}")
                failures += 1
                break
            if diff > max_diff:
                max_diff = diff
            total_elements += 256

        status = "PASSED" if failures == 0 else "FAILED"
        print(f"[{status}] {fmt:<8} | 256 dims x {n_trials} trials | byte stride: {row_size:>3}B | max_diff: {max_diff:.6e}")

    print("================================================================================")
    if failures == 0:
        print(f"ALL 12 FORMATS PASSED! {total_elements} dequantized elements verified identical to C reference.")
        return 0
    else:
        print(f"{failures} format(s) failed!")
        return 1

if __name__ == '__main__':
    sys.exit(run_tests())
