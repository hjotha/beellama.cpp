#!/usr/bin/env python3
"""Explicit-execution candidate RKVA op19/20 qualification; no model TPS claim.

Default prints the plan without opening a socket. --self-test tests wire layout
only. --execute is reserved for when the user says the Xbox is online. Requires
rkva_range_probe.py from this worktree and its independent float64 oracle.
"""
import argparse
import json
import math
import random
import struct
from pathlib import Path


def frame(k, v, positions, q, start, rows, wire):
    assert len(k) == len(v) and len(q) == len(positions) * 24 * 256
    return (struct.pack('<8I', start, rows, wire, 0, 1, 0, 0, 0) + k + v +
            struct.pack('<%di' % len(positions), *positions) +
            struct.pack('<%df' % len(q), *q))


def plane(rows, wire, rng):
    raw, decoded = bytearray(), []
    for _ in range(rows * 4 * 8):
        scale = struct.unpack('<e', struct.pack('<e', rng.uniform(-.08, .08)))[0]
        if wire == 2:
            packed = bytes(rng.randrange(256) for _ in range(16))
            raw += struct.pack('<e', scale) + packed
            decoded.extend(((x & 15) - 8) * scale for x in packed)
            decoded.extend(((x >> 4) - 8) * scale for x in packed)
        elif wire == 3:
            values = [rng.randrange(-128, 128) for _ in range(32)]
            raw += struct.pack('<e32b', scale, *values)
            decoded.extend(x * scale for x in values)
        else:
            values = [struct.unpack('<e', struct.pack('<e', rng.uniform(-1, 1)))[0]
                      for _ in range(32)]
            raw += struct.pack('<32e', *values)
            decoded.extend(values)
    return bytes(raw), decoded


def check_response(payload, expected, nq):
    assert len(payload) == nq * (24 * 256 + 24) * 4
    floats = struct.unpack('<%df' % (len(payload) // 4), payload)
    actual = (floats[:nq * 24 * 256], floats[nq * 24 * 256:])
    errors = []
    for name, values, oracle in zip(('OUT', 'LSE'), actual, expected):
        maximum = 0.
        for got, want in zip(values, oracle):
            assert math.isfinite(got) and math.isfinite(want)
            error = abs(got - want)
            assert error <= 3e-4 + (3e-4 if name == 'OUT' else 3e-5) * abs(want), (name, got, want)
            maximum = max(maximum, error)
        errors.append(maximum)
    return dict(zip(('out_max_abs', 'lse_max_abs'), errors))


def self_test():
    for wire, block in ((0, 64), (2, 18), (3, 34)):
        rng = random.Random(713)
        k, dk = plane(3, wire, rng)
        v, dv = plane(3, wire, rng)
        pos, q = [102402], [0.] * (24 * 256)
        payload = frame(k, v, pos, q, 0, 3, wire)
        assert len(k) == 3 * 4 * 8 * block
        assert len(dk) == len(dv) == 3 * 4 * 256
        assert struct.unpack_from('<8I', payload) == (0, 3, wire, 0, 1, 0, 0, 0)
        assert payload[32:32 + len(k)] == k
        assert payload[32 + len(k):32 + 2 * len(k)] == v
        assert struct.unpack_from('<i', payload, 32 + 2 * len(k)) == (102402,)
        assert len(payload) == 32 + 2 * len(k) + 4 + len(q) * 4
    print(json.dumps({'offline_wire_layout_tests': 'passed', 'gpu_tested': False}))


def execute(args):
    from rkva_range_probe import Client, ORIGIN
    from rkva_range_q4_kernel_probe import oracle
    import numpy as np
    wire = {'f16': 0, 'q4_0': 2, 'q8_0': 3}[args.storage]
    rng = random.Random(713)
    k, dk = plane(args.rows, wire, rng)
    v, dv = plane(args.rows, wire, rng)
    keys = np.asarray(dk, dtype=np.float64).reshape(args.rows, 4, 256)
    values = np.asarray(dv, dtype=np.float64).reshape(args.rows, 4, 256)
    row_bytes = len(k) // args.rows
    client = Client(args.host, args.port)
    checks = []
    try:
        client.hello(args.rows, layers=1)
        assert client.capabilities & (1 << 4) and client.capabilities & (1 << 5), 'candidate op19/20 required'
        client.call(2, struct.pack('<4I', 0, 0, args.rows, 1 if wire else 0), session=0)
        for mode in ([0, 1, 2, 3] if wire == 2 else [0]):
            client.call(20, struct.pack('<IIQ', 3, mode, 0), session=0)
            for nq in (1, min(256, args.rows)):
                client.configure(args.rows, q4=wire == 2, q8=wire == 3)
                # Chunked history, followed by a final fused upload with its own
                # independent row count. In decode this uploads just one new row.
                start = args.rows - nq
                if start:
                    client.call(13, struct.pack('<4I', 0, start, wire, 0) +
                                k[:start * row_bytes] + v[:start * row_bytes], session=0, nq=start)
                positions = list(range(ORIGIN + start, ORIGIN + args.rows))
                q = [struct.unpack('<f', struct.pack('<f', rng.uniform(-.7, .7)))[0]
                     for _ in range(nq * 24 * 256)]
                body = frame(k[start * row_bytes:], v[start * row_bytes:], positions, q, start, nq, wire)
                # Independent FP64 dot/softmax over exactly the decoded wire
                # values; vectorization keeps the nq256 control inexpensive.
                out, lse = [], []
                queries = np.asarray(q, dtype=np.float64).reshape(nq, 24, 256)
                for query, position in zip(queries, positions):
                    o, l = oracle(np, query, keys, values,
                                  min(args.rows, position - ORIGIN + 1), 1 / 16)
                    out.extend(o.ravel().tolist())
                    lse.extend(l.ravel().tolist())
                expected = out, lse
                fused = client.call(19, body, session=0, nq=nq)
                error = check_response(fused, expected, nq)
                legacy = client.attention(q, positions)
                for got, want in zip(struct.unpack('<%df' % (len(fused) // 4), fused), legacy[0] + legacy[1]):
                    assert abs(got - want) <= 1e-6 + 1e-5 * abs(want), ('fused-vs-legacy', mode, nq, got, want)
                # A malformed fused frame must not alter existing authoritative KV.
                for offset, value in ((12, 1), (16, 0), (20, 1), (24, 1)):
                    bad = bytearray(body); struct.pack_into('<I', bad, offset, value)
                    client.call(19, bad, session=0, nq=nq, success=False)
                replay = client.attention(q, positions)
                assert replay == legacy, 'rejected requests mutated KV/output'
                checks.append({'mode': mode, 'nq': nq, 'rows': args.rows, **error})
            report = json.loads(client.call(20, struct.pack('<IIQ', 8, 0, 0), session=0))
            checks.append({'mode': mode, 'profile': report})
        result = {'gpu_tested': True, 'storage': args.storage, 'rows': args.rows,
                  'checks': checks, 'timings': client.timings,
                  'scope': 'synthetic GPU correctness/control; NOT model TPS or capacity'}
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'output': args.output, 'checks': len(checks), 'gpu_tested': True}))
    finally:
        client.socket.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host')
    ap.add_argument('--port', type=int, default=50054)
    ap.add_argument('--storage', choices=('q4_0', 'q8_0', 'f16'), default='q4_0')
    ap.add_argument('--rows', type=int, default=257)
    ap.add_argument('--output')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true')
    group.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    assert 2 <= args.rows <= 1025
    if args.self_test:
        self_test()
    elif args.execute:
        assert args.host and args.output, '--execute requires explicit --host and --output'
        execute(args)
    else:
        print(json.dumps({'state': 'PLAN_ONLY_NO_NETWORK', 'rows': args.rows,
                          'storage': args.storage, 'needs': 'user reports Xbox online before --execute',
                          'cases': 'append op19, nq1/nq256, modes0..3 Q4, FP64 oracle, op14 replay, malformed no-mutation, op20 report'}))

if __name__ == '__main__':
    main()
