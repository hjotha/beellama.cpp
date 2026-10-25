#!/usr/bin/env python3
"""Explicit diagnostic lease: exact deployed F32->F16->FP8 GPU pack + status gates.

No network unless --execute. Op11/id5 is diagnostic, never session/KV attention.
The test deliberately sends invalid F32/LSE values and checks per-group poison.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--host', default='192.168.1.26')
    parser.add_argument('--port', type=int, default=50054)
    parser.add_argument('--output')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(execute=False, plan='Explicit --execute tests all F16 patterns, F32 midpoints and neighbors, finite F16 overflow, NaN/Inf, LSE bits and malformed probe framing.')))
        return
    import numpy as np
    import rkva_kvarn_serial_prefix_probe as p
    from rkva_range_probe import Client
    p.CAPACITY = 2048
    client = Client(args.host, args.port)
    results = []
    try:
        flags = p.PROTO_F16 | p.PROTO_F32 | p.PROTO_NATIVE_REWIND_V2 | p.PROTO_SERIAL_PREFIX
        ack = struct.unpack('<IIIIQQQ', client.call(1, p.hello_payload(1, flags)))
        assert ack[3] == 278, ack
        # Probe has no session reference; op22 generation and resident metadata
        # cannot be accessed by this call. Request mode/reserved must stay zero.
        def request(values, lse_bits=None):
            values = np.asarray(values, dtype='<f4').ravel()
            heads = (values.size + 255) // 256
            padded = np.zeros(heads * 256, dtype='<f4')
            padded[:values.size] = values
            lse = np.zeros(heads, dtype='<u4') if lse_bits is None else np.asarray(lse_bits, dtype='<u4')
            assert lse.size == heads
            blob = struct.pack('<2I', heads, 0) + padded.tobytes() + lse.tobytes()
            return struct.pack('<4I', 5, 0, len(blob), 0) + blob, padded, lse

        def case(name, values, lse_bits=None):
            payload, values, lse = request(values, lse_bits)
            response = bytes(client.call(11, payload))
            n, heads = values.size, lse.size
            groups = (heads + 3) // 4
            assert len(response) == n + heads * 4 + groups * 4
            with np.errstate(over='ignore', invalid='ignore'):
                half = values.astype('<f2').view('<u2').astype(np.uint32)
            invalid = (half & 0x7c00) == 0x7c00
            mag = np.minimum(half & 0x7fff, 0x7b00)
            want = (((half >> 8) & 0x80) | ((mag + 0x7f + ((mag >> 8) & 1)) >> 8)).astype(np.uint8)
            got = np.frombuffer(response, dtype=np.uint8, count=n)
            # Invalid lanes have no serving value. They must poison their group;
            # every finite lane still must match the old two-rounding contract.
            assert np.array_equal(got[~invalid], want[~invalid]), name
            assert response[n:n + heads*4] == lse.tobytes(), (name, 'LSE bits changed')
            status = np.frombuffer(response, dtype='<u4', offset=n + heads*4, count=groups)
            expected = np.zeros(groups, dtype='<u4')
            for i in range(groups):
                if invalid[i*1024:(i+1)*1024].any(): expected[i] |= 1
                if ((lse[i*256:(i+1)*256] & 0x7f800000) == 0x7f800000).any(): expected[i] |= 2
            assert np.array_equal(status, expected), (name, status.tolist(), expected.tolist())
            results.append(dict(name=name, elements=int(n), heads=int(heads), finite_elements=int((~invalid).sum()), rejected_out_elements=int(invalid.sum()), poisoned_groups=int(np.count_nonzero(status)), finite_bytes_equal=True, lse_bits_equal=True))

        case('all 65536 F16 patterns', np.arange(65536, dtype='<u2').view('<f2').astype('<f4'))
        lo = np.arange(0x7bff, dtype='<u2').view('<f2').astype(np.float64)
        hi = np.arange(1, 0x7c00, dtype='<u2').view('<f2').astype(np.float64)
        midpoint = ((lo + hi) * 0.5).astype('<f4')
        neighbors = np.concatenate((np.nextafter(midpoint, np.float32(0)), midpoint, np.nextafter(midpoint, np.float32('inf'))))
        case('F32 half midpoints and immediate neighbors, both signs', np.concatenate((neighbors, -neighbors)))
        fp8_values = (np.arange(124, dtype='<u2') << 8).view('<f2').astype(np.float64)
        fp8_midpoints = ((fp8_values[:-1] + fp8_values[1:]) * 0.5).astype('<f4')
        fp8_neighbors = np.concatenate((np.nextafter(fp8_midpoints, np.float32(0)), fp8_midpoints, np.nextafter(fp8_midpoints, np.float32('inf'))))
        case('every E5M2 midpoint and F32 neighbors, both signs', np.concatenate((fp8_neighbors, -fp8_neighbors)))
        special_bits = np.array([0, 0x80000000, 0x3f900001, 0x7f800000, 0xff800000, 0x7fc01234, 0x7f800001, 0xffc01234, 0x7f7fffff, 0xff7fffff, 1, 0x80000001], dtype='<u4')
        edges = np.concatenate((special_bits.view('<f4'), np.array([65520.0, np.nextafter(np.float32(65520), np.float32(0)), -65520.0, 57344.0, -57344.0], dtype='<f4')))
        case('signed zero, double rounding witness, overflow and specials', edges)
        case('combined OUT and LSE poison in same group', [65520.0], [0x7fc01234])
        # Test final group writes and all LSE classes, with unused lanes padded.
        for heads in (1, 3, 4, 25, 1024):
            lse = np.resize(np.array([0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff, 0xca000000, 0x7f800000, 0xff800000, 0x7fc01234, 0x7f800001], dtype='<u4'), heads)
            values = np.zeros(heads * 256, dtype='<f4')
            values[-1] = np.float32(65520)
            case('LSE classes / last group heads=%d' % heads, values, lse)
        good, _, _ = request(np.zeros(256, dtype='<f4'))
        malformed = [good[:-1], good + b'\0']
        for offset, value in ((4, 1), (8, 0), (12, 1), (16, 0), (16, 1025), (20, 1)):
            payload = bytearray(good)
            struct.pack_into('<I', payload, offset, value)
            malformed.append(payload)
        for payload in malformed:
            client.call(11, payload, success=False)
            assert client.timings[-1]['status'] == -8, client.timings[-1]
        for probe_id in (1, 2, 3, 4, 6):
            payload = bytearray(good)
            struct.pack_into('<I', payload, 0, probe_id)
            client.call(11, payload, success=False)
            assert client.timings[-1]['status'] == -2
        result = dict(ok=True, server_version=ack[3], checks=results, malformed_payloads_rejected=len(malformed), contract='F32->F16 RNE; reject F16 exponent31; finite F16->FP8 RNE/sat; F32 LSE bit copy + finite status', timings=client.timings)
        text = json.dumps(result, indent=2)
        if args.output: Path(args.output).write_text(text + '\n')
        print(text)
    finally:
        client.socket.close()


if __name__ == '__main__':
    main()
