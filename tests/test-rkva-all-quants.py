#!/usr/bin/env python3
"""
Test suite verifying that all KV cache quantization formats in beellama.cpp
are supported by Xbox Remote Attention (RKVA), execute efficiently on the GPU,
and produce valid numerical outputs.

Clearly separates:
- Standard GGML 1D 32-element block quantizations:
    q8_0 (272 B/head), q6_1 (224 B/head), q6_0 (208 B/head),
    q5_1 (192 B/head), q5_0 (176 B/head), q4_1 (160 B/head),
    iq4_nl (144 B/head), q4_0 (144 B/head), q3_1 (128 B/head),
    q3_0 (112 B/head), q2_1 (96 B/head), q2_0s (80 B/head)
- Rotated 1D row quantizations:
    saw_int4 (136 B/head), bdr_int4 (136 B/head),
    turboquant_nc (132 B/head), oscar_int2 (96 B/head)
- True KVarN 2D 128x128 WHT + 16-iter Sinkhorn record quantizations:
    kvarn8 (17152 B/rec), kvarn6 (13056 B/rec), kvarn5 (11008 B/rec),
    kvarn4 (8960 B/rec),  kvarn3 (6912 B/rec),  kvarn2 (4864 B/rec)
- Unquantized float formats:
    f16 (512 B/head), bf16 (512 B/head), f32 (1024 B/head)
"""

import argparse
import math
import socket
import struct
import sys
import time

RKVA_MAGIC = 0x41564B52  # "RKVA"
RKVA_VERSION = 3

# Ops
RKVA_HELLO = 1
RKVA_CREATE_SESSION = 2
RKVA_DESTROY_SESSION = 3
RKVA_RANGE_CONFIG = 12
RKVA_RANGE_UPLOAD = 13
RKVA_RANGE_ATTN = 14
RKVA_KVARN_RANGE_CONFIG = 15
RKVA_KVARN_RANGE_UPLOAD = 16
RKVA_KVARN_RANGE_BULK_ATTN = 22

# Wires
RKVA_WIRE_F16 = 0
RKVA_WIRE_F32 = 1
RKVA_WIRE_FP8 = 4           # E5M2
RKVA_WIRE_FP8_E4M3FN = 7

# Session Flags
RKVA_SESSION_RANGE = 1

# Protocol flags for HELLO
RKVA_PROTO_BULK_E4M3FN = 1 << RKVA_WIRE_FP8_E4M3FN
RKVA_PROTO_NATIVE_REWIND_V2 = 1 << 16
RKVA_PROTO_NATIVE_SERIAL_PREFIX_V3 = 1 << 17
RKVA_DOMAIN_AUTO = 0

# Capability bits in ack.reserved[0]
CAP_RANGE_Q4_0         = 1 << 0
CAP_RANGE_Q8_0         = 1 << 1
CAP_RANGE_KVARN4       = 1 << 2
CAP_KVARN_BULK_ATTN    = 1 << 8
CAP_KVARN_BULK_F16     = 1 << 9
CAP_KVARN_BULK_FP8     = 1 << 10
CAP_NATIVE_Q4_BULK     = 1 << 14
CAP_NATIVE_Q8_BULK     = 1 << 15
CAP_NATIVE_OSCAR_INT2  = 1 << 22
CAP_NATIVE_SAW_INT4    = 1 << 23
CAP_NATIVE_TURBOQUANT  = 1 << 24
CAP_KVARN_E4M3FN       = 1 << 25
CAP_Q4_E4M3FN          = 1 << 28
CAP_SAW_E4M3FN         = 1 << 29
CAP_TQNC_E4M3FN        = 1 << 30
CAP_Q4_1_BULK          = 1 << 31
CAP_IQ4_NL_BULK        = 1 << 32
CAP_Q4_1_E4M3FN        = 1 << 33
CAP_IQ4_NL_E4M3FN      = 1 << 34
CAP_Q5_0_BULK          = 1 << 35
CAP_Q5_1_BULK          = 1 << 36
CAP_Q6_0_BULK          = 1 << 37
CAP_Q6_1_BULK          = 1 << 38
CAP_Q3_0_BULK          = 1 << 39
CAP_Q3_1_BULK          = 1 << 40
CAP_Q2_1_BULK          = 1 << 41
CAP_BF16_BULK          = 1 << 42
CAP_F32_BULK           = 1 << 43
CAP_Q5_0_E4M3FN        = 1 << 44
CAP_Q5_1_E4M3FN        = 1 << 45
CAP_Q6_0_E4M3FN        = 1 << 46
CAP_Q6_1_E4M3FN        = 1 << 47
CAP_Q3_0_E4M3FN        = 1 << 48
CAP_Q3_1_E4M3FN        = 1 << 49
CAP_OSCAR_E4M3FN       = 1 << 50
CAP_Q2_1_E4M3FN        = 1 << 51
CAP_BF16_E4M3FN        = 1 << 52
CAP_F32_E4M3FN         = 1 << 53
CAP_Q2_0S_BULK         = 1 << 54
CAP_Q2_0S_E4M3FN       = 1 << 55
CAP_Q8_E4M3FN          = 1 << 56

# Storage values
RKVA_STORAGE_F16           = 0
RKVA_STORAGE_Q4_0          = 1 << 0
RKVA_STORAGE_Q8_0          = 1 << 1
RKVA_STORAGE_KVARN4        = 3
RKVA_STORAGE_SAW_INT4      = 1 << 3
RKVA_STORAGE_OSCAR_INT2    = 1 << 4
RKVA_STORAGE_TURBOQUANT_NC = 1 << 5
RKVA_STORAGE_Q4_1          = 1 << 6
RKVA_STORAGE_IQ4_NL        = 1 << 7
RKVA_STORAGE_Q5_0          = 1 << 8
RKVA_STORAGE_Q5_1          = 1 << 9
RKVA_STORAGE_Q6_0          = 1 << 10
RKVA_STORAGE_Q6_1          = 1 << 11
RKVA_STORAGE_Q3_0          = 1 << 12
RKVA_STORAGE_Q3_1          = 1 << 13
RKVA_STORAGE_Q2_1          = 1 << 14
RKVA_STORAGE_BF16          = 1 << 15
RKVA_STORAGE_F32           = 1 << 16
RKVA_STORAGE_Q2_0S         = 1 << 17

# Layout numbers
RKVA_LAYOUT_F16           = 0
RKVA_LAYOUT_KVARN4        = 3
RKVA_LAYOUT_Q4_0          = 4
RKVA_LAYOUT_Q8_0          = 5
RKVA_LAYOUT_SAW_INT4      = 6
RKVA_LAYOUT_OSCAR_INT2    = 7
RKVA_LAYOUT_TURBOQUANT_NC = 8
RKVA_LAYOUT_Q4_1          = 9
RKVA_LAYOUT_IQ4_NL        = 10
RKVA_LAYOUT_Q5_0          = 11
RKVA_LAYOUT_Q5_1          = 12
RKVA_LAYOUT_Q6_0          = 13
RKVA_LAYOUT_Q6_1          = 14
RKVA_LAYOUT_Q3_0          = 15
RKVA_LAYOUT_Q3_1          = 16
RKVA_LAYOUT_Q2_1          = 17
RKVA_LAYOUT_BF16          = 18
RKVA_LAYOUT_F32           = 19
RKVA_LAYOUT_Q2_0S         = 20

FORMAT_SPECS = [
    # --- Group I: Unquantized Float Formats (3) ---
    {
        "name": "f16",
        "kind": "f16_range",
        "storage": RKVA_STORAGE_F16,
        "layout": RKVA_LAYOUT_F16,
        "head_bytes": 512,
        "key_bits": 16, "val_bits": 16,
        "wht": 0, "domain": 0,
        "cap_bit": None, "cap_e4_bit": None,
    },
    {
        "name": "f32",
        "kind": "native_row",
        "storage": RKVA_STORAGE_F32,
        "layout": RKVA_LAYOUT_F32,
        "head_bytes": 1024,
        "key_bits": 32, "val_bits": 32,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_F32_BULK, "cap_e4_bit": CAP_F32_E4M3FN,
    },
    {
        "name": "bf16",
        "kind": "native_row",
        "storage": RKVA_STORAGE_BF16,
        "layout": RKVA_LAYOUT_BF16,
        "head_bytes": 512,
        "key_bits": 16, "val_bits": 16,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_BF16_BULK, "cap_e4_bit": CAP_BF16_E4M3FN,
    },
    # --- Group II: Standard GGML 1D Block Quantizations (12) ---
    {
        "name": "q8_0",
        "kind": "native_row",
        "storage": RKVA_STORAGE_Q8_0,
        "layout": RKVA_LAYOUT_Q8_0,
        "head_bytes": 272,
        "key_bits": 8, "val_bits": 8,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_NATIVE_Q8_BULK, "cap_e4_bit": CAP_Q8_E4M3FN,
    },
    {
        "name": "q6_1",
        "kind": "native_row",
        "storage": RKVA_STORAGE_Q6_1,
        "layout": RKVA_LAYOUT_Q6_1,
        "head_bytes": 224,
        "key_bits": 6, "val_bits": 6,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q6_1_BULK, "cap_e4_bit": CAP_Q6_1_E4M3FN,
    },
    {
        "name": "q6_0",
        "kind": "native_row",
        "storage": RKVA_STORAGE_Q6_0,
        "layout": RKVA_LAYOUT_Q6_0,
        "head_bytes": 208,
        "key_bits": 6, "val_bits": 6,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q6_0_BULK, "cap_e4_bit": CAP_Q6_0_E4M3FN,
    },
    {
        "name": "q5_1",
        "kind": "native_row",
        "storage": RKVA_STORAGE_Q5_1,
        "layout": RKVA_LAYOUT_Q5_1,
        "head_bytes": 192,
        "key_bits": 5, "val_bits": 5,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q5_1_BULK, "cap_e4_bit": CAP_Q5_1_E4M3FN,
    },
    {
        "name": "q5_0",
        "kind": "native_row",
        "storage": RKVA_STORAGE_Q5_0,
        "layout": RKVA_LAYOUT_Q5_0,
        "head_bytes": 176,
        "key_bits": 5, "val_bits": 5,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q5_0_BULK, "cap_e4_bit": CAP_Q5_0_E4M3FN,
    },
    {
        "name": "q4_1",
        "kind": "native_row",
        "storage": RKVA_STORAGE_Q4_1,
        "layout": RKVA_LAYOUT_Q4_1,
        "head_bytes": 160,
        "key_bits": 4, "val_bits": 4,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q4_1_BULK, "cap_e4_bit": CAP_Q4_1_E4M3FN,
    },
    {
        "name": "iq4_nl",
        "kind": "native_row",
        "storage": RKVA_STORAGE_IQ4_NL,
        "layout": RKVA_LAYOUT_IQ4_NL,
        "head_bytes": 144,
        "key_bits": 4, "val_bits": 4,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_IQ4_NL_BULK, "cap_e4_bit": CAP_IQ4_NL_E4M3FN,
    },
    {
        "name": "q4_0",
        "kind": "native_row",
        "storage": RKVA_STORAGE_Q4_0,
        "layout": RKVA_LAYOUT_Q4_0,
        "head_bytes": 144,
        "key_bits": 4, "val_bits": 4,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_NATIVE_Q4_BULK, "cap_e4_bit": CAP_Q4_E4M3FN,
    },
    {
        "name": "q3_1",
        "kind": "native_row",
        "storage": RKVA_STORAGE_Q3_1,
        "layout": RKVA_LAYOUT_Q3_1,
        "head_bytes": 128,
        "key_bits": 3, "val_bits": 3,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q3_1_BULK, "cap_e4_bit": CAP_Q3_1_E4M3FN,
    },
    {
        "name": "q3_0",
        "kind": "native_row",
        "storage": RKVA_STORAGE_Q3_0,
        "layout": RKVA_LAYOUT_Q3_0,
        "head_bytes": 112,
        "key_bits": 3, "val_bits": 3,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q3_0_BULK, "cap_e4_bit": CAP_Q3_0_E4M3FN,
    },
    {
        "name": "q2_1",
        "kind": "native_row",
        "storage": RKVA_STORAGE_Q2_1,
        "layout": RKVA_LAYOUT_Q2_1,
        "head_bytes": 96,
        "key_bits": 2, "val_bits": 2,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q2_1_BULK, "cap_e4_bit": CAP_Q2_1_E4M3FN,
    },
    {
        "name": "q2_0s",
        "kind": "native_row",
        "storage": RKVA_STORAGE_Q2_0S,
        "layout": RKVA_LAYOUT_Q2_0S,
        "head_bytes": 80,
        "key_bits": 2, "val_bits": 2,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q2_0S_BULK, "cap_e4_bit": CAP_Q2_0S_E4M3FN,
    },
    # --- Group III: Rotated 1D Row Quantizations (4) ---
    {
        "name": "saw_int4",
        "kind": "native_row",
        "storage": RKVA_STORAGE_SAW_INT4,
        "layout": RKVA_LAYOUT_SAW_INT4,
        "head_bytes": 136,
        "key_bits": 4, "val_bits": 4,
        "wht": 16, "domain": 1,
        "cap_bit": CAP_NATIVE_SAW_INT4, "cap_e4_bit": CAP_SAW_E4M3FN,
    },
    {
        "name": "bdr_int4",
        "kind": "native_row",
        "storage": RKVA_STORAGE_SAW_INT4,
        "layout": RKVA_LAYOUT_SAW_INT4,
        "head_bytes": 136,
        "key_bits": 4, "val_bits": 4,
        "wht": 16, "domain": 1,
        "cap_bit": CAP_NATIVE_SAW_INT4, "cap_e4_bit": CAP_SAW_E4M3FN,
    },
    {
        "name": "turboquant_nc",
        "kind": "native_row",
        "storage": RKVA_STORAGE_TURBOQUANT_NC,
        "layout": RKVA_LAYOUT_TURBOQUANT_NC,
        "head_bytes": 132,
        "key_bits": 4, "val_bits": 4,
        "wht": 256, "domain": 2,
        "cap_bit": CAP_NATIVE_TURBOQUANT, "cap_e4_bit": CAP_TQNC_E4M3FN,
    },
    {
        "name": "oscar_int2",
        "kind": "native_row",
        "storage": RKVA_STORAGE_OSCAR_INT2,
        "layout": RKVA_LAYOUT_OSCAR_INT2,
        "head_bytes": 96,
        "key_bits": 2, "val_bits": 2,
        "wht": 0, "domain": 1,
        "cap_bit": CAP_NATIVE_OSCAR_INT2, "cap_e4_bit": CAP_OSCAR_E4M3FN,
    },
    # --- Group IV: True KVarN 2D 128x128 WHT + 16-iter Sinkhorn Formats (6) ---
    {
        "name": "kvarn8",
        "kind": "kvarn_2d",
        "storage": RKVA_STORAGE_KVARN4,
        "layout": RKVA_LAYOUT_KVARN4,
        "head_bytes": 17152,
        "key_bits": 8, "val_bits": 8,
        "wht": 1, "domain": 1,
        "cap_bit": CAP_RANGE_KVARN4, "cap_e4_bit": CAP_KVARN_E4M3FN,
    },
    {
        "name": "kvarn6",
        "kind": "kvarn_2d",
        "storage": RKVA_STORAGE_KVARN4,
        "layout": RKVA_LAYOUT_KVARN4,
        "head_bytes": 13056,
        "key_bits": 6, "val_bits": 6,
        "wht": 1, "domain": 1,
        "cap_bit": CAP_RANGE_KVARN4, "cap_e4_bit": CAP_KVARN_E4M3FN,
    },
    {
        "name": "kvarn5",
        "kind": "kvarn_2d",
        "storage": RKVA_STORAGE_KVARN4,
        "layout": RKVA_LAYOUT_KVARN4,
        "head_bytes": 11008,
        "key_bits": 5, "val_bits": 5,
        "wht": 1, "domain": 1,
        "cap_bit": CAP_RANGE_KVARN4, "cap_e4_bit": CAP_KVARN_E4M3FN,
    },
    {
        "name": "kvarn4",
        "kind": "kvarn_2d",
        "storage": RKVA_STORAGE_KVARN4,
        "layout": RKVA_LAYOUT_KVARN4,
        "head_bytes": 8960,
        "key_bits": 4, "val_bits": 4,
        "wht": 1, "domain": 1,
        "cap_bit": CAP_RANGE_KVARN4, "cap_e4_bit": CAP_KVARN_E4M3FN,
    },
    {
        "name": "kvarn3",
        "kind": "kvarn_2d",
        "storage": RKVA_STORAGE_KVARN4,
        "layout": RKVA_LAYOUT_KVARN4,
        "head_bytes": 6912,
        "key_bits": 3, "val_bits": 3,
        "wht": 1, "domain": 1,
        "cap_bit": CAP_RANGE_KVARN4, "cap_e4_bit": CAP_KVARN_E4M3FN,
    },
    {
        "name": "kvarn2",
        "kind": "kvarn_2d",
        "storage": RKVA_STORAGE_KVARN4,
        "layout": RKVA_LAYOUT_KVARN4,
        "head_bytes": 4864,
        "key_bits": 2, "val_bits": 2,
        "wht": 1, "domain": 1,
        "cap_bit": CAP_RANGE_KVARN4, "cap_e4_bit": CAP_KVARN_E4M3FN,
    },
]

def make_request(op, session_id=0, layer_id=0, n_tokens=0, cycle_id=1, payload=b""):
    hdr = struct.pack("<IHHQIIII", RKVA_MAGIC, RKVA_VERSION, op, cycle_id, session_id, layer_id, n_tokens, len(payload))
    return hdr + payload

def read_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"Socket closed while reading {n} bytes (got {len(buf)})")
        buf.extend(chunk)
    return bytes(buf)

def transact(sock, op, session_id=0, layer_id=0, n_tokens=0, cycle_id=1, payload=b""):
    req = make_request(op, session_id=session_id, layer_id=layer_id, n_tokens=n_tokens, cycle_id=cycle_id, payload=payload)
    sock.sendall(req)
    resp_hdr_bytes = read_exact(sock, 56)
    magic, ver, rop, rcycle_id, status, rpayload_len, rx_us, prep_us, comp_us, resp_us = struct.unpack(
        "<IHHQiIQQQQ", resp_hdr_bytes
    )
    if magic != RKVA_MAGIC:
        raise ValueError(f"Bad magic 0x{magic:08x}")
    payload_data = read_exact(sock, rpayload_len) if rpayload_len > 0 else b""
    return status, rop, payload_data, (rx_us, prep_us, comp_us, resp_us)

def generate_quant_bytes(name, head_bytes, n_heads, n_rows, is_k=True):
    """Generate deterministic, valid quantized KV row bytes with well-conditioned scales and values."""
    total = head_bytes * n_heads * n_rows
    data = bytearray(total)
    fp16_val = struct.pack("<e", 0.05)
    f32_val = struct.pack("<f", 0.05)
    bf16_val = struct.pack("<H", 0x3d4c)

    for r in range(n_rows):
        for h in range(n_heads):
            offset = (r * n_heads + h) * head_bytes
            if name == "f32":
                for elem in range(256):
                    data[offset + elem * 4 : offset + elem * 4 + 4] = f32_val
            elif name == "bf16":
                for elem in range(256):
                    data[offset + elem * 2 : offset + elem * 2 + 2] = bf16_val
            elif name == "f16":
                for elem in range(256):
                    data[offset + elem * 2 : offset + elem * 2 + 2] = fp16_val
            elif name in ("saw_int4", "bdr_int4"):
                data[offset + 128 : offset + 132] = f32_val
                data[offset + 132 : offset + 136] = struct.pack("<f", 0.0)
            elif name == "turboquant_nc":
                data[offset + 128 : offset + 130] = fp16_val
                data[offset + 130 : offset + 132] = b"\x00\x00"
            else:
                block_size = head_bytes // 8
                for b in range(8):
                    boff = offset + b * block_size
                    data[boff : boff + 2] = fp16_val
                    if name in ("q4_1", "q5_1", "q6_1", "q3_1", "q2_1", "oscar_int2"):
                        data[boff + 2 : boff + 4] = b"\x00\x00"
    return bytes(data)

def generate_kvarn_records(bits, record_bytes, n_head_kv, n_groups):
    """Generate valid KVarN 2D WHT + Sinkhorn records (packed bit-planes + s1, s2, mu FP16 metadata)."""
    heads = n_head_kv * 2  # 2 slices of 128 per head_dim=256
    total = n_groups * heads * record_bytes
    data = bytearray(total)
    q_bytes = bits * 2048
    s_fp16 = struct.pack("<e", 0.05)
    mu_fp16 = struct.pack("<e", 0.0)
    # Fill bit-planes with alternating pattern so dequantized values are non-trivial
    for g in range(n_groups):
        for h in range(heads):
            base = (g * heads + h) * record_bytes
            for i in range(0, q_bytes, 4):
                data[base + i : base + i + 4] = b"\x55\xaa\x55\xaa"
            # s1 (128 FP16), s2 (128 FP16), mu (128 FP16) = 768 bytes
            meta = base + q_bytes
            for idx in range(128):
                data[meta + idx * 2 : meta + idx * 2 + 2] = s_fp16
                data[meta + 256 + idx * 2 : meta + 256 + idx * 2 + 2] = s_fp16
                data[meta + 512 + idx * 2 : meta + 512 + idx * 2 + 2] = mu_fp16
    return bytes(data)

def generate_f16_rows(n_rows, n_head_kv, head_dim=256):
    fp16_val = struct.pack("<e", 0.05)
    return fp16_val * (n_rows * n_head_kv * head_dim)

def generate_q_fp8_e4m3(n_tokens, n_heads, head_dim):
    """Generate FP8 E4M3FN Q tensor values (0x38 is 1.0f in E4M3FN)."""
    total = n_tokens * n_heads * head_dim
    return bytes([0x38] * total)

def generate_q_f32(n_tokens, n_heads, head_dim):
    f32_val = struct.pack("<f", 0.1)
    return f32_val * (n_tokens * n_heads * head_dim)

def parse_f32_array(data):
    count = len(data) // 4
    return struct.unpack(f"<{count}f", data)

def test_xbox_all_quants(host="192.168.1.200", port=50054):
    print("=" * 112)
    print(f"Connecting to Xbox RKVA Server at {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(60.0)
    sock.connect((host, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    n_head = 48
    n_head_kv = 8
    head_dim = 256
    capacity = 256
    origin = 0
    n_layer = 1
    max_ctx = 2048

    proto_flags = (
        (1 << RKVA_WIRE_F16) |
        (1 << RKVA_WIRE_FP8) |
        RKVA_PROTO_BULK_E4M3FN |
        RKVA_PROTO_NATIVE_REWIND_V2 |
        RKVA_PROTO_NATIVE_SERIAL_PREFIX_V3
    )
    kq_scale = 1.0 / math.sqrt(head_dim)
    hello_payload = struct.pack(
        "<16If2Q",
        proto_flags, n_layer, n_head, n_head_kv, head_dim, max_ctx,
        16, 16, 128, 0,
        0, 2, 0, RKVA_DOMAIN_AUTO, 0, 0,
        kq_scale, 0, 0
    )
    assert len(hello_payload) == 84

    status, rop, hello_ack_data, _ = transact(sock, RKVA_HELLO, payload=hello_payload)
    if status != 0:
        print(f"HELLO failed with status {status}")
        return False

    supported_ops, chosen_wire, max_sessions, server_version, mem_budget, caps, cap_res1 = struct.unpack(
        "<4I3Q", hello_ack_data
    )
    print(f"[OK] HELLO acknowledged. Protocol version {RKVA_VERSION}, server version {server_version}.")
    print(f"     Capabilities bitmask: 0x{caps:016x}")

    results = []
    sess_id_counter = 1

    for spec in FORMAT_SPECS:
        name = spec["name"]
        kind = spec["kind"]
        storage = spec["storage"]
        layout = spec["layout"]
        head_bytes = spec["head_bytes"]
        cap_bit = spec["cap_bit"]
        cap_e4 = spec["cap_e4_bit"]

        cap_advertised = True
        if cap_bit is not None and not (caps & cap_bit):
            cap_advertised = False
        e4_advertised = True
        if cap_e4 is not None and not (caps & cap_e4):
            e4_advertised = False

        sess_id = sess_id_counter
        sess_id_counter += 1

        sess_payload = struct.pack("<4I", sess_id, 0, capacity, RKVA_SESSION_RANGE)
        st, _, _, _ = transact(sock, RKVA_CREATE_SESSION, session_id=sess_id, payload=sess_payload)
        if st != 0:
            results.append({"name": name, "status": "FAIL", "reason": f"CREATE_SESSION failed ({st})"})
            continue

        try:
            if kind == "f16_range":
                # Standard F16 range path: RKVA_RANGE_CONFIG -> RKVA_RANGE_UPLOAD -> RKVA_RANGE_ATTN
                cfg_payload = struct.pack("<4I", origin, capacity, 0, 0)
                st, _, _, _ = transact(sock, RKVA_RANGE_CONFIG, session_id=sess_id, payload=cfg_payload)
                if st != 0:
                    results.append({"name": name, "status": "FAIL", "reason": f"RANGE_CONFIG failed ({st})"})
                    continue

                k_data = generate_f16_rows(capacity, n_head_kv, head_dim)
                v_data = generate_f16_rows(capacity, n_head_kv, head_dim)
                up_hdr = struct.pack("<4I", 0, capacity, RKVA_WIRE_F16, 0)
                st, _, _, _ = transact(
                    sock, RKVA_RANGE_UPLOAD, session_id=sess_id, layer_id=0, payload=up_hdr + k_data + v_data
                )
                if st != 0:
                    results.append({"name": name, "status": "FAIL", "reason": f"RANGE_UPLOAD failed ({st})"})
                    continue

                # Decode nq=1
                attn_hdr = struct.pack("<4I", RKVA_WIRE_F32, 0, 0, 0)
                pos1 = struct.pack("<i", 255)
                q1 = generate_q_f32(1, n_head, head_dim)
                t0 = time.perf_counter()
                st, _, resp1, _ = transact(
                    sock, RKVA_RANGE_ATTN, session_id=sess_id, layer_id=0, n_tokens=1, payload=attn_hdr + pos1 + q1
                )
                t1 = time.perf_counter()
                decode_us = (t1 - t0) * 1e6
                if st != 0:
                    results.append({"name": name, "status": "FAIL", "reason": f"RANGE_ATTN decode failed ({st})"})
                    continue
                out1_bytes = 1 * n_head * head_dim * 4
                lse_vals = parse_f32_array(resp1[out1_bytes:])
                if not all(math.isfinite(x) for x in lse_vals):
                    results.append({"name": name, "status": "FAIL", "reason": "LSE NaN/Inf"})
                    continue

                # Prefill nq=8
                pos8 = struct.pack("<8i", *range(248, 256))
                q8 = generate_q_f32(8, n_head, head_dim)
                t0 = time.perf_counter()
                st, _, resp8, _ = transact(
                    sock, RKVA_RANGE_ATTN, session_id=sess_id, layer_id=0, n_tokens=8, payload=attn_hdr + pos8 + q8
                )
                t1 = time.perf_counter()
                prefill_us = (t1 - t0) * 1e6
                if st != 0:
                    results.append({"name": name, "status": "FAIL", "reason": f"RANGE_ATTN prefill failed ({st})"})
                    continue

            elif kind == "kvarn_2d":
                # True KVarN 2D 128x128 WHT + 16-iter Sinkhorn record path
                bits = spec["key_bits"]
                rec_bytes = spec["head_bytes"]
                cfg_payload = struct.pack(
                    "<18I",
                    origin, capacity, storage, layout,
                    128, 128, head_dim, 2,         # token_group=128, record_dim=128, head_dim=256, head_slices=2
                    bits, bits, rec_bytes,         # key_bits, value_bits, record_bytes
                    3, 2, 128,                     # stage_groups=3, tail_groups=2, sink_tokens=128
                    128, 1, 1, 1                   # precision_tail_tokens=128, wht_version=1, domain=1, record_policy=1
                )
                st, _, _, _ = transact(sock, RKVA_KVARN_RANGE_CONFIG, session_id=sess_id, payload=cfg_payload)
                if st != 0:
                    results.append({"name": name, "status": "FAIL", "reason": f"KVARN_CONFIG failed ({st})"})
                    continue

                # Initial RESET upload of 247 live tokens (1 sealed record group + 384 stage rows + 128 tail rows)
                init_live = 247
                rec_k0 = generate_kvarn_records(bits, rec_bytes, n_head_kv, 1)
                rec_v0 = generate_kvarn_records(bits, rec_bytes, n_head_kv, 1)
                stg_k0 = generate_f16_rows(384, n_head_kv, head_dim)
                stg_v0 = generate_f16_rows(384, n_head_kv, head_dim)
                tail_k0 = generate_f16_rows(128, n_head_kv, head_dim)
                tail_v0 = generate_f16_rows(128, n_head_kv, head_dim)
                up_hdr0 = struct.pack("<8I", init_live, 0, 1, 1, 0, 0, 0, 0)
                up_payload0 = up_hdr0 + rec_k0 + rec_v0 + stg_k0 + stg_v0 + tail_k0 + tail_v0
                st, _, _, _ = transact(
                    sock, RKVA_KVARN_RANGE_UPLOAD, session_id=sess_id, layer_id=0, payload=up_payload0
                )
                if st != 0:
                    results.append({"name": name, "status": "FAIL", "reason": f"KVARN_UPLOAD failed ({st})"})
                    continue

                # Decode nq=1 via BULK_ATTN (append 1 row -> live_count=248)
                wire = RKVA_WIRE_FP8_E4M3FN
                stg_k1 = generate_f16_rows(1, n_head_kv, head_dim)
                stg_v1 = generate_f16_rows(1, n_head_kv, head_dim)
                tail_k1 = generate_f16_rows(1, n_head_kv, head_dim)
                tail_v1 = generate_f16_rows(1, n_head_kv, head_dim)
                u1_data = stg_k1 + stg_v1 + tail_k1 + tail_v1
                u1_hdr = struct.pack("<8I", 248, 1, 0, 0, 247, 1, 0, 0)
                bulk_hdr1 = struct.pack("<4I", 1, 1, 1, 0)
                seg_hdr1 = struct.pack("<2I", 32 + len(u1_data), 1)
                attn_hdr1 = struct.pack("<4I", wire, 0, 0, 0)
                pos1 = struct.pack("<i", 247)
                q1 = generate_q_fp8_e4m3(1, n_head, head_dim)
                bulk_payload1 = bulk_hdr1 + seg_hdr1 + u1_hdr + u1_data + attn_hdr1 + pos1 + q1

                t0 = time.perf_counter()
                st, _, resp_data, _ = transact(
                    sock, RKVA_KVARN_RANGE_BULK_ATTN, session_id=sess_id, layer_id=0, n_tokens=1, payload=bulk_payload1
                )
                t1 = time.perf_counter()
                decode_us = (t1 - t0) * 1e6
                if st != 0:
                    results.append({"name": name, "status": "FAIL", "reason": f"KVARN_BULK_ATTN decode failed ({st})"})
                    continue

                out_bytes = 1 * n_head * head_dim
                lse_bytes = 1 * n_head * 4
                if len(resp_data) != out_bytes + lse_bytes:
                    results.append({"name": name, "status": "FAIL", "reason": f"bad reply size {len(resp_data)}"})
                    continue
                lse_vals = parse_f32_array(resp_data[out_bytes:])
                if not all(math.isfinite(x) for x in lse_vals):
                    results.append({"name": name, "status": "FAIL", "reason": "LSE contains NaN/Inf"})
                    continue

                # Prefill/MQ nq=8 via BULK_ATTN (append 8 rows -> live_count=256, seals record group 1)
                rec_k8 = generate_kvarn_records(bits, rec_bytes, n_head_kv, 1)
                rec_v8 = generate_kvarn_records(bits, rec_bytes, n_head_kv, 1)
                stg_k8 = generate_f16_rows(8, n_head_kv, head_dim)
                stg_v8 = generate_f16_rows(8, n_head_kv, head_dim)
                tail_k8 = generate_f16_rows(8, n_head_kv, head_dim)
                tail_v8 = generate_f16_rows(8, n_head_kv, head_dim)
                u8_data = rec_k8 + rec_v8 + stg_k8 + stg_v8 + tail_k8 + tail_v8
                u8_hdr = struct.pack("<8I", 256, 1, 1, 0, 248, 8, 0, 0)
                bulk_hdr8 = struct.pack("<4I", 1, 1, 8, 0)
                seg_hdr8 = struct.pack("<2I", 32 + len(u8_data), 8)
                attn_hdr8 = struct.pack("<4I", wire, 0, 0, 0)
                pos8 = struct.pack("<8i", *range(248, 256))
                q8 = generate_q_fp8_e4m3(8, n_head, head_dim)
                bulk_payload8 = bulk_hdr8 + seg_hdr8 + u8_hdr + u8_data + attn_hdr8 + pos8 + q8

                t0 = time.perf_counter()
                st, _, resp_data8, _ = transact(
                    sock, RKVA_KVARN_RANGE_BULK_ATTN, session_id=sess_id, layer_id=0, n_tokens=8, payload=bulk_payload8
                )
                t1 = time.perf_counter()
                prefill_us = (t1 - t0) * 1e6
                if st != 0:
                    results.append({"name": name, "status": "FAIL", "reason": f"KVARN_BULK_ATTN prefill failed ({st})"})
                    continue

            else:
                # 1D rowlane native formats
                cfg_payload = struct.pack(
                    "<18I",
                    origin, capacity, storage, layout,
                    1, 0, head_dim, 2,
                    spec["key_bits"], spec["val_bits"], head_bytes,
                    0, 0, 0,
                    0, spec["wht"], spec["domain"], 1
                )
                assert len(cfg_payload) == 72
                st, _, _, _ = transact(sock, RKVA_KVARN_RANGE_CONFIG, session_id=sess_id, payload=cfg_payload)
                if st != 0:
                    results.append({"name": name, "status": "FAIL", "reason": f"CONFIG failed ({st})"})
                    continue

                k_data = generate_quant_bytes(name, head_bytes, n_head_kv, capacity, is_k=True)
                v_data = generate_quant_bytes(name, head_bytes, n_head_kv, capacity, is_k=False)
                up_hdr = struct.pack("<8I", capacity, 0, 0, 1, 0, capacity, 0, 0)
                up_payload = up_hdr + k_data + v_data
                st, _, _, _ = transact(sock, RKVA_KVARN_RANGE_UPLOAD, session_id=sess_id, layer_id=0, payload=up_payload)
                if st != 0:
                    results.append({"name": name, "status": "FAIL", "reason": f"UPLOAD failed ({st})"})
                    continue

                wire = RKVA_WIRE_FP8_E4M3FN
                bulk_hdr = struct.pack("<4I", 1, 1, 1, 0)
                seg_hdr = struct.pack("<2I", 32, 1)
                u0 = struct.pack("<8I", capacity, 0, 0, 0, capacity, 0, 0, 0)
                attn_hdr = struct.pack("<4I", wire, 0, 0, 0)
                pos_data = struct.pack("<i", 255)
                q_data = generate_q_fp8_e4m3(1, n_head, head_dim)
                bulk_payload = bulk_hdr + seg_hdr + u0 + attn_hdr + pos_data + q_data

                t0 = time.perf_counter()
                st, _, resp_data, _ = transact(
                    sock, RKVA_KVARN_RANGE_BULK_ATTN, session_id=sess_id, layer_id=0, n_tokens=1, payload=bulk_payload
                )
                t1 = time.perf_counter()
                decode_us = (t1 - t0) * 1e6

                if st != 0:
                    results.append({"name": name, "status": "FAIL", "reason": f"ATTN failed ({st})"})
                    continue

                out_bytes = 1 * n_head * head_dim
                lse_bytes = 1 * n_head * 4
                if len(resp_data) != out_bytes + lse_bytes:
                    results.append({"name": name, "status": "FAIL", "reason": f"bad reply size {len(resp_data)} vs {out_bytes+lse_bytes}"})
                    continue

                lse_vals = parse_f32_array(resp_data[out_bytes:])
                if not all(math.isfinite(x) for x in lse_vals):
                    results.append({"name": name, "status": "FAIL", "reason": "LSE contains NaN/Inf"})
                    continue

                bulk_hdr8 = struct.pack("<4I", 1, 1, 8, 0)
                seg_hdr8 = struct.pack("<2I", 32, 8)
                u0_8 = struct.pack("<8I", capacity, 0, 0, 0, capacity, 0, 0, 0)
                attn_hdr8 = struct.pack("<4I", wire, 0, 0, 0)
                pos_data8 = struct.pack("<8i", *range(248, 256))
                q_data8 = generate_q_fp8_e4m3(8, n_head, head_dim)
                bulk_payload8 = bulk_hdr8 + seg_hdr8 + u0_8 + attn_hdr8 + pos_data8 + q_data8

                t0 = time.perf_counter()
                st, _, resp_data8, _ = transact(
                    sock, RKVA_KVARN_RANGE_BULK_ATTN, session_id=sess_id, layer_id=0, n_tokens=8, payload=bulk_payload8
                )
                t1 = time.perf_counter()
                prefill_us = (t1 - t0) * 1e6

                if st != 0:
                    results.append({"name": name, "status": "FAIL", "reason": f"PREFILL failed ({st})"})
                    continue

            results.append({
                "name": name,
                "kind": kind,
                "status": "PASS",
                "layout": layout,
                "storage": storage,
                "head_bytes": head_bytes,
                "cap_advertised": cap_advertised,
                "e4_advertised": e4_advertised,
                "decode_us": decode_us,
                "prefill_us": prefill_us,
                "lse_mean": sum(lse_vals) / len(lse_vals)
            })

        finally:
            transact(sock, RKVA_DESTROY_SESSION, session_id=sess_id)

    sock.close()

    print("\n" + "=" * 112)
    print(f"{'Format':<16} | {'Kind':<10} | {'Layout':<6} | {'Bytes/Unit':<10} | {'Caps':<5} | {'E4':<4} | {'Decode RTT':<12} | {'Prefill 8 RTT':<14} | {'Status'}")
    print("-" * 112)
    all_passed = True
    for r in results:
        status_str = r["status"]
        if status_str != "PASS":
            all_passed = False
            print(f"{r['name']:<16} | {'-':<10} | {'-':<6} | {'-':<10} | {'-':<5} | {'-':<4} | {'-':<12} | {'-':<14} | FAIL ({r['reason']})")
        else:
            cap_str = "YES" if r["cap_advertised"] else "NO"
            e4_str = "YES" if r["e4_advertised"] else "NO"
            unit_str = f"{r['head_bytes']} B/rec" if r["kind"] == "kvarn_2d" else f"{r['head_bytes']} B/hd"
            dec_str = f"{r['decode_us']:.1f} µs"
            pref_str = f"{r['prefill_us']:.1f} µs"
            print(f"{r['name']:<16} | {r['kind']:<10} | {r['layout']:<6} | {unit_str:<10} | {cap_str:<5} | {e4_str:<4} | {dec_str:<12} | {pref_str:<14} | PASS")
    print("=" * 112)

    if all_passed:
        print(f"\n>>> ALL {len(results)} KV CACHE QUANTIZATION FORMATS SUCCESSFULLY VERIFIED ON XBOX SERIES X! <<<")
    else:
        print("\n>>> SOME FORMATS FAILED VERIFICATION. <<<")
    return all_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.200")
    parser.add_argument("--port", type=int, default=50054)
    args = parser.parse_args()
    ok = test_xbox_all_quants(args.host, args.port)
    sys.exit(0 if ok else 1)
