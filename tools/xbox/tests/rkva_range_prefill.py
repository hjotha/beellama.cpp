#!/usr/bin/env python3
"""Live analytical Q256 prefill gate for positional F16 and native Q4_0 caches.

K is zero, V varies by KV head/dimension and is constant across positions. The
expected OUT is therefore V (or zero if masked), and LSE is log(causal count).
This exercises the real Q256, H24/HKV4/D256, split32 GPU shape without a costly
CPU dot-product oracle. One layer suffices to validate dispatch/scratch/layout.
"""
import argparse
import json
import math
import struct
from rkva_range_probe import Client, H, HKV, D, ORIGIN


def run(host, port, storage):
    q4, q8 = storage == 1, storage == 2
    capacity, queries = 28672, 256
    client = Client(host, port)
    try:
        client.hello(capacity, layers=1)
        assert not storage or client.capabilities & storage
        client.call(2, struct.pack('<4I', 0, 0, capacity, 1))
        assert json.loads(client.call(10))['engine_kv_bytes'] == 0
        client.configure(capacity, q4=q4, q8=q8)
        expected = [(((d+head) % 16)-8) * ((head+1)/16)
                    for head in range(HKV) for d in range(D)]
        if q8:
            key_row = (b'\0\0' + b'\0'*32) * (HKV*D//32)
            value_row = bytearray()
            for head in range(HKV):
                for block in range(D//32):
                    value_row.extend(struct.pack('<e', (head+1)/16))
                    value_row.extend(struct.pack('<32b', *[(((d+head) % 16)-8) for d in range(32)]))
            value_row = bytes(value_row)
        elif q4:
            key_row = (b'\0\0' + b'\x88'*16) * (HKV*D//32)
            value_row = bytearray()
            for head in range(HKV):
                for block in range(D//32):
                    value_row.extend(struct.pack('<e', (head+1)/16))
                    value_row.extend(bytes((((j+head) % 16) |
                                             (((j+16+head) % 16) << 4)) for j in range(16)))
            value_row = bytes(value_row)
        else:
            key_row = b'\0\0' * (HKV*D)
            value_row = struct.pack('<'+'e'*(HKV*D), *expected)
        for start in range(0, capacity, 512):
            rows = min(512, capacity-start)
            payload = struct.pack('<4I', start, rows, 3 if q8 else 2 if q4 else 0, 0)
            client.call(13, payload + key_row*rows + value_row*rows, session=0)
        positions = [ORIGIN-1, ORIGIN, ORIGIN+1, ORIGIN+capacity-1] * (queries//4)
        q = [((i % 13)-6)/16 for i in range(queries*H*D)]
        actual, lse = client.attention(q, positions)
        timing = client.timings[-1]
        max_out = max_lse = 0.
        for qi, position in enumerate(positions):
            count = min(capacity, max(0, position-ORIGIN+1))
            for head in range(H):
                kv_head = head // (H//HKV)
                offset = (qi*H+head)*D
                for d in range(D):
                    want = expected[kv_head*D+d] if count else 0.
                    got = actual[offset+d]
                    error = abs(got-want)
                    assert math.isfinite(got) and error < 3e-4, (q4, qi, head, d, got, want)
                    max_out = max(max_out, error)
                got = lse[qi*H+head]
                if count:
                    error = abs(got-math.log(count))
                    assert math.isfinite(got) and error < 3e-4, (q4, qi, head, got)
                    max_lse = max(max_lse, error)
                else:
                    assert got == -math.inf, (q4, qi, head, got)
        kv_bytes = capacity * HKV * ((D//32)*(68 if q8 else 36) if q4 or q8 else D*4)
        q_heads = queries*H
        query_bytes = q_heads*D*4 + queries*4
        output_bytes = q_heads*(D+1)*4
        scratch_bytes = query_bytes + 32*q_heads*(D+2)*4 + 2*output_bytes + 2*512*len(key_row)
        stats = json.loads(client.call(10))
        assert stats['engine_kv_bytes'] == kv_bytes, stats
        assert kv_bytes + scratch_bytes < 4*1024**3
        client.call(3, session=0)
        return dict(success=True, storage='q8_0' if q8 else 'q4_0' if q4 else 'f16', upload_wire=3 if q8 else 2 if q4 else 0, n_queries=queries,
                    n_head=H, n_head_kv=HKV, head_dim=D, split_k=32,
                    origin=ORIGIN, rows=capacity, layers=1,
                    out_max_abs=max_out, lse_max_abs=max_lse,
                    kv_bytes=kv_bytes, scratch_allocated_bytes=scratch_bytes,
                    response_bytes=(queries*H*(D+1)*4), attention=timing)
    finally:
        client.socket.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=50054)
    parser.add_argument('--output')
    args = parser.parse_args()
    report = dict(success=True, cases=[run(args.host, args.port, storage) for storage in (0, 1, 2)])
    result = json.dumps(report, indent=2)
    print(result)
    if args.output:
        with open(args.output, 'w') as stream:
            stream.write(result+'\n')


if __name__ == '__main__':
    main()
