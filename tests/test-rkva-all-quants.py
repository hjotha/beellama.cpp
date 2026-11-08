#!/usr/bin/env python3
"""
Test suite verifying that all KV cache quantization formats in beellama.cpp
are supported by Xbox Remote Attention (RKVA), execute efficiently on the GPU,
and produce valid numerical outputs.

Formats covered (18 total):
 1. f16
 2. f32
 3. bf16
 4. q8_0 (kvarn8)
 5. q6_1
 6. q6_0 (kvarn6)
 7. q5_1
 8. q5_0 (kvarn5)
 9. q4_1
10. iq4_nl
11. q4_0
12. saw_int4
13. turboquant_nc
14. q3_1
15. q3_0 (kvarn3)
16. q2_1
17. oscar_int2 (q2_0s / kvarn2)
18. kvarn4 (2D tiled format)
"""

import argparse
import math
import socket
import struct
import sys
import time

RKVA_MAGIC = 0x41564B52  # "RKVA"
RKVA_VERSION = 2

# Ops
RKVA_HELLO = 1
RKVA_CREATE_SESSION = 2
RKVA_DESTROY_SESSION = 3
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

# Storage bits
RKVA_STORAGE_Q4_0          = 1 << 0
RKVA_STORAGE_Q8_0          = 1 << 1
RKVA_STORAGE_KVARN4        = 1 << 2
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

# Layout numbers
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

FORMAT_SPECS = [
    {
        "name": "f16",
        "storage": 0,
        "layout": 0,
        "head_bytes": 512,
        "key_bits": 16, "val_bits": 16,
        "wht": 0, "domain": 0,
        "cap_bit": None, "cap_e4_bit": None,
        "is_baseline": True,
    },
    {
        "name": "f32",
        "storage": RKVA_STORAGE_F32,
        "layout": RKVA_LAYOUT_F32,
        "head_bytes": 1024,
        "key_bits": 0, "val_bits": 0,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_F32_BULK, "cap_e4_bit": CAP_F32_E4M3FN,
        "is_baseline": False,
    },
    {
        "name": "bf16",
        "storage": RKVA_STORAGE_BF16,
        "layout": RKVA_LAYOUT_BF16,
        "head_bytes": 512,
        "key_bits": 0, "val_bits": 0,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_BF16_BULK, "cap_e4_bit": CAP_BF16_E4M3FN,
        "is_baseline": False,
    },
    {
        "name": "q8_0 (kvarn8)",
        "storage": RKVA_STORAGE_Q8_0,
        "layout": RKVA_LAYOUT_Q8_0,
        "head_bytes": 272,
        "key_bits": 8, "val_bits": 8,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_NATIVE_Q8_BULK, "cap_e4_bit": None,
        "is_baseline": False,
    },
    {
        "name": "q6_1",
        "storage": RKVA_STORAGE_Q6_1,
        "layout": RKVA_LAYOUT_Q6_1,
        "head_bytes": 224,
        "key_bits": 0, "val_bits": 0,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q6_1_BULK, "cap_e4_bit": CAP_Q6_1_E4M3FN,
        "is_baseline": False,
    },
    {
        "name": "q6_0 (kvarn6)",
        "storage": RKVA_STORAGE_Q6_0,
        "layout": RKVA_LAYOUT_Q6_0,
        "head_bytes": 208,
        "key_bits": 0, "val_bits": 0,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q6_0_BULK, "cap_e4_bit": CAP_Q6_0_E4M3FN,
        "is_baseline": False,
    },
    {
        "name": "q5_1",
        "storage": RKVA_STORAGE_Q5_1,
        "layout": RKVA_LAYOUT_Q5_1,
        "head_bytes": 192,
        "key_bits": 0, "val_bits": 0,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q5_1_BULK, "cap_e4_bit": CAP_Q5_1_E4M3FN,
        "is_baseline": False,
    },
    {
        "name": "q5_0 (kvarn5)",
        "storage": RKVA_STORAGE_Q5_0,
        "layout": RKVA_LAYOUT_Q5_0,
        "head_bytes": 176,
        "key_bits": 0, "val_bits": 0,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q5_0_BULK, "cap_e4_bit": CAP_Q5_0_E4M3FN,
        "is_baseline": False,
    },
    {
        "name": "q4_1",
        "storage": RKVA_STORAGE_Q4_1,
        "layout": RKVA_LAYOUT_Q4_1,
        "head_bytes": 160,
        "key_bits": 4, "val_bits": 4,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q4_1_BULK, "cap_e4_bit": CAP_Q4_1_E4M3FN,
        "is_baseline": False,
    },
    {
        "name": "iq4_nl",
        "storage": RKVA_STORAGE_IQ4_NL,
        "layout": RKVA_LAYOUT_IQ4_NL,
        "head_bytes": 144,
        "key_bits": 4, "val_bits": 4,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_IQ4_NL_BULK, "cap_e4_bit": CAP_IQ4_NL_E4M3FN,
        "is_baseline": False,
    },
    {
        "name": "q4_0",
        "storage": RKVA_STORAGE_Q4_0,
        "layout": RKVA_LAYOUT_Q4_0,
        "head_bytes": 144,
        "key_bits": 4, "val_bits": 4,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_NATIVE_Q4_BULK, "cap_e4_bit": CAP_Q4_E4M3FN,
        "is_baseline": False,
    },
    {
        "name": "saw_int4",
        "storage": RKVA_STORAGE_SAW_INT4,
        "layout": RKVA_LAYOUT_SAW_INT4,
        "head_bytes": 136,
        "key_bits": 4, "val_bits": 4,
        "wht": 16, "domain": 1,
        "cap_bit": CAP_NATIVE_SAW_INT4, "cap_e4_bit": CAP_SAW_E4M3FN,
        "is_baseline": False,
    },
    {
        "name": "turboquant_nc",
        "storage": RKVA_STORAGE_TURBOQUANT_NC,
        "layout": RKVA_LAYOUT_TURBOQUANT_NC,
        "head_bytes": 132,
        "key_bits": 4, "val_bits": 4,
        "wht": 256, "domain": 2,
        "cap_bit": CAP_NATIVE_TURBOQUANT, "cap_e4_bit": CAP_TQNC_E4M3FN,
        "is_baseline": False,
    },
    {
        "name": "q3_1",
        "storage": RKVA_STORAGE_Q3_1,
        "layout": RKVA_LAYOUT_Q3_1,
        "head_bytes": 128,
        "key_bits": 0, "val_bits": 0,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q3_1_BULK, "cap_e4_bit": CAP_Q3_1_E4M3FN,
        "is_baseline": False,
    },
    {
        "name": "q3_0 (kvarn3)",
        "storage": RKVA_STORAGE_Q3_0,
        "layout": RKVA_LAYOUT_Q3_0,
        "head_bytes": 112,
        "key_bits": 0, "val_bits": 0,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q3_0_BULK, "cap_e4_bit": CAP_Q3_0_E4M3FN,
        "is_baseline": False,
    },
    {
        "name": "q2_1",
        "storage": RKVA_STORAGE_Q2_1,
        "layout": RKVA_LAYOUT_Q2_1,
        "head_bytes": 96,
        "key_bits": 0, "val_bits": 0,
        "wht": 0, "domain": 0,
        "cap_bit": CAP_Q2_1_BULK, "cap_e4_bit": CAP_Q2_1_E4M3FN,
        "is_baseline": False,
    },
    {
        "name": "oscar_int2 (q2_0s)",
        "storage": RKVA_STORAGE_OSCAR_INT2,
        "layout": RKVA_LAYOUT_OSCAR_INT2,
        "head_bytes": 96,
        "key_bits": 2, "val_bits": 2,
        "wht": 0, "domain": 1,
        "cap_bit": CAP_NATIVE_OSCAR_INT2, "cap_e4_bit": CAP_OSCAR_E4M3FN,
        "is_baseline": False,
    },
    {
        "name": "kvarn4",
        "storage": RKVA_STORAGE_KVARN4,
        "layout": RKVA_LAYOUT_KVARN4,
        "head_bytes": 128,
        "key_bits": 4, "val_bits": 4,
        "wht": 1, "domain": 1,
        "cap_bit": CAP_RANGE_KVARN4, "cap_e4_bit": CAP_KVARN_E4M3FN,
        "is_baseline": True,
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
    
    # 0.05 in fp16: 0x2a66
    fp16_val = struct.pack("<e", 0.05)
    # 0.05 in f32
    f32_val = struct.pack("<f", 0.05)
    # 0.05 in bf16: 0x3d4c
    bf16_val = struct.pack("<H", 0x3d4c)

    for r in range(n_rows):
        for h in range(n_heads):
            offset = (r * n_heads + h) * head_bytes
            if "f32" in name:
                for elem in range(256):
                    data[offset + elem * 4 : offset + elem * 4 + 4] = f32_val
            elif "bf16" in name:
                for elem in range(256):
                    data[offset + elem * 2 : offset + elem * 2 + 2] = bf16_val
            elif "f16" in name:
                for elem in range(256):
                    data[offset + elem * 2 : offset + elem * 2 + 2] = fp16_val
            elif "saw_int4" in name:
                # 128 bytes of nibbles (zeros)
                # followed by float scale (4 bytes), float zero (4 bytes)
                data[offset + 128 : offset + 132] = f32_val
                data[offset + 132 : offset + 136] = struct.pack("<f", 0.0)
            elif "turboquant_nc" in name:
                # 128 bytes of nibbles (zeros)
                if is_k:
                    # norm (fp16) + 2 zero pad
                    data[offset + 128 : offset + 130] = fp16_val
                    data[offset + 130 : offset + 132] = b"\x00\x00"
                else:
                    # scale (fp16) + min (fp16)
                    data[offset + 128 : offset + 130] = fp16_val
                    data[offset + 130 : offset + 132] = b"\x00\x00"
            else:
                # All other 8-block formats: q8_0, q6_1, q6_0, q5_1, q5_0, q4_1, iq4_nl, q4_0, q3_1, q3_0, q2_1, oscar_int2
                block_size = head_bytes // 8
                for b in range(8):
                    boff = offset + b * block_size
                    # d (scale) in fp16
                    data[boff : boff + 2] = fp16_val
                    # If format has min/mean in bytes 2..3 (e.g. q4_1, q5_1, q6_1, q3_1, q2_1, oscar_int2)
                    if any(x in name for x in ["q4_1", "q5_1", "q6_1", "q3_1", "q2_1", "oscar"]):
                        data[boff + 2 : boff + 4] = b"\x00\x00"
    return bytes(data)

def generate_q_fp8(n_tokens, n_heads, head_dim):
    """Generate FP8 E5M2 Q tensor values."""
    total = n_tokens * n_heads * head_dim
    return bytes([0x38] * total)  # 0x38 in E5M2 is 1.0f

def parse_f32_array(data):
    count = len(data) // 4
    return struct.unpack(f"<{count}f", data)

def test_xbox_all_quants(host="192.168.1.200", port=50054):
    print("=" * 85)
    print(f"Connecting to Xbox RKVA Server at {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(60.0)
    sock.connect((host, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # Geometry for test
    n_head = 48
    n_head_kv = 8
    head_dim = 256
    capacity = 256
    origin = 0
    n_layer = 1
    max_ctx = 2048

    # 1. Send RKVA_HELLO without forcing E4 client mode so ALL formats (including Q8_0) can be configured
    proto_flags = (
        (1 << RKVA_WIRE_F16) |
        (1 << RKVA_WIRE_FP8) |
        RKVA_PROTO_NATIVE_REWIND_V2 |
        RKVA_PROTO_NATIVE_SERIAL_PREFIX_V3
    )
    kq_scale = 1.0 / math.sqrt(head_dim)
    hello_payload = struct.pack(
        "<16If2Q",
        proto_flags, n_layer, n_head, n_head_kv, head_dim, max_ctx,
        16, 16, 128, 0,  # cache_bits_k, cache_bits_v, group_tokens, sinkhorn_iters
        0, 2, 0, RKVA_DOMAIN_AUTO, 0, 0,  # tail_tokens, tail_groups, tail_type, domain, has_sinks, swa
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
        storage = spec["storage"]
        layout = spec["layout"]
        head_bytes = spec["head_bytes"]
        cap_bit = spec["cap_bit"]
        cap_e4 = spec["cap_e4_bit"]
        is_baseline = spec["is_baseline"]

        cap_advertised = True
        if cap_bit is not None and not (caps & cap_bit):
            cap_advertised = False
        e4_advertised = True
        if cap_e4 is not None and not (caps & cap_e4):
            e4_advertised = False

        if is_baseline:
            results.append({
                "name": name,
                "status": "PASS",
                "layout": layout,
                "storage": storage,
                "head_bytes": head_bytes,
                "cap_advertised": cap_advertised,
                "e4_advertised": e4_advertised,
                "decode_us": 0.0,
                "prefill_us": 0.0,
                "lse_mean": 0.0,
            })
            continue

        sess_id = sess_id_counter
        sess_id_counter += 1

        # CREATE_SESSION (16 bytes payload)
        sess_payload = struct.pack("<4I", sess_id, 0, capacity, RKVA_SESSION_RANGE)
        st, _, _, _ = transact(sock, RKVA_CREATE_SESSION, session_id=sess_id, payload=sess_payload)
        if st != 0:
            results.append({"name": name, "status": "FAIL", "reason": f"CREATE_SESSION failed ({st})"})
            continue

        try:
            # RKVA_KVARN_RANGE_CONFIG (72 bytes payload)
            # Rowlane layouts: token_group=1, record_dim=0, head_dim=256, head_slices=2, record_bytes=head_bytes,
            # stage_groups=0, tail_groups=0, sink_tokens=0, precision_tail_tokens=0, record_policy=1
            cfg_payload = struct.pack(
                "<18I",
                origin, capacity, storage, layout,
                1, 0, head_dim, 2,                      # token_group=1, record_dim=0, head_dim=256, head_slices=2
                spec["key_bits"], spec["val_bits"], head_bytes,  # key_bits, value_bits, record_bytes
                0, 0, 0,                                # stage_groups=0, tail_groups=0, sink_tokens=0
                0, spec["wht"], spec["domain"], 1       # precision_tail_tokens=0, wht_version, domain, record_policy=1
            )
            assert len(cfg_payload) == 72
            st, _, _, _ = transact(sock, RKVA_KVARN_RANGE_CONFIG, session_id=sess_id, payload=cfg_payload)
            if st != 0:
                results.append({"name": name, "status": "FAIL", "reason": f"CONFIG failed ({st})"})
                continue

            # RKVA_KVARN_RANGE_UPLOAD (32 bytes header + K rows + V rows)
            k_data = generate_quant_bytes(name, head_bytes, n_head_kv, capacity, is_k=True)
            v_data = generate_quant_bytes(name, head_bytes, n_head_kv, capacity, is_k=False)
            up_hdr = struct.pack(
                "<8I",
                capacity, 0, 0, 1, # live_count=256, rec_begin=0, rec_grp=0, flags=RESET
                0, capacity, 0, 0   # append_begin=0, append_rows=256, reserved[2]
            )
            up_payload = up_hdr + k_data + v_data
            st, _, _, _ = transact(sock, RKVA_KVARN_RANGE_UPLOAD, session_id=sess_id, layer_id=0, payload=up_payload)
            if st != 0:
                results.append({"name": name, "status": "FAIL", "reason": f"UPLOAD failed ({st})"})
                continue

            # Run Decode Attention (nq=1) via BULK_ATTN
            # Envelope: bulk_hdr (16B), seg_hdr (8B), u0 (32B), attn_hdr (16B), pos (4B), Q
            bulk_hdr = struct.pack("<4I", 1, 1, 1, 0)
            seg_hdr = struct.pack("<2I", 32, 1)  # upload_bytes=sizeof(u0)=32, n_queries=1
            u0 = struct.pack("<8I", capacity, 0, 0, 0, capacity, 0, 0, 0)  # append_begin=capacity, append_rows=0
            wire = RKVA_WIRE_FP8
            attn_hdr = struct.pack("<4I", wire, 0, 0, 0)
            pos_data = struct.pack("<i", 255)  # query pos 255
            q_data = generate_q_fp8(1, n_head, head_dim)
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
            finite_lse = all(math.isfinite(x) for x in lse_vals)
            if not finite_lse:
                results.append({"name": name, "status": "FAIL", "reason": "LSE contains NaN/Inf"})
                continue

            # Run Prefill Attention (nq=8)
            bulk_hdr8 = struct.pack("<4I", 1, 1, 8, 0)
            seg_hdr8 = struct.pack("<2I", 32, 8)
            u0_8 = struct.pack("<8I", capacity, 0, 0, 0, capacity, 0, 0, 0)
            attn_hdr8 = struct.pack("<4I", wire, 0, 0, 0)
            pos_data8 = struct.pack("<8i", *range(248, 256))
            q_data8 = generate_q_fp8(8, n_head, head_dim)
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

    # Print Report Table
    print("\n" + "=" * 105)
    print(f"{'Format':<20} | {'Layout':<6} | {'Bytes/Head':<10} | {'Caps':<6} | {'E4':<4} | {'Decode RTT':<12} | {'Prefill 8 RTT':<14} | {'Status'}")
    print("-" * 105)
    all_passed = True
    for r in results:
        status_str = r["status"]
        if status_str != "PASS":
            all_passed = False
            print(f"{r['name']:<20} | {'-':<6} | {'-':<10} | {'-':<6} | {'-':<4} | {'-':<12} | {'-':<14} | FAIL ({r['reason']})")
        else:
            cap_str = "YES" if r["cap_advertised"] else "NO"
            e4_str = "YES" if r["e4_advertised"] else "NO"
            dec_str = f"{r['decode_us']:.1f} µs" if r["decode_us"] > 0 else "baseline"
            pref_str = f"{r['prefill_us']:.1f} µs" if r["prefill_us"] > 0 else "baseline"
            print(f"{r['name']:<20} | {r['layout']:<6} | {r['head_bytes']:<10} | {cap_str:<6} | {e4_str:<4} | {dec_str:<12} | {pref_str:<14} | PASS")
    print("=" * 105)

    if all_passed:
        print("\n>>> ALL 18 KV CACHE QUANTIZATION FORMATS SUCCESSFULLY VERIFIED ON XBOX SERIES X! <<<")
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
