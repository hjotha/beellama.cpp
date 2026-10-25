#!/usr/bin/env python3
"""Bounded-memory live GPU capacity/causal gate using analytical constant K/V.

Unlike the random parity gate, this fills all 16 layers of the real 100k..128k
range. K=0 makes every score zero: OUT must equal V and LSE must be log(count).
The CPU holds at most one upload chunk, not a second full KV cache.
"""
import argparse
import json
import math
import struct
import time
from rkva_range_probe import Client, H, HKV, D, ORIGIN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=50054)
    parser.add_argument('--rows', type=int, default=28672)
    parser.add_argument('--layers', type=int, default=16)
    parser.add_argument('--output')
    storage_group = parser.add_mutually_exclusive_group()
    storage_group.add_argument('--q4', action='store_true')
    storage_group.add_argument('--q8', action='store_true')
    parser.add_argument('--verify-f16-budget', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.rows <= 131072 and 1 <= args.layers <= 16
    assert args.q4 or args.q8 or args.rows <= 60000
    assert not args.verify_f16_budget or ((args.q4 or args.q8) and args.layers == 16 and args.rows > 65536)
    client = Client(args.host, args.port)
    checks = []
    start = time.perf_counter()
    try:
        client.hello(args.rows, layers=args.layers)
        if args.q4 or args.q8:
            assert client.capabilities & (2 if args.q8 else 1), "server lacks requested quantized range storage"
        client.call(2, struct.pack('<4I', 0, 0, args.rows, int(bool(client.capabilities & 3))))
        if client.capabilities & 3:
            assert json.loads(client.call(10))['engine_kv_bytes'] == 0
        if args.verify_f16_budget:
            client.configure(args.rows, q4=False, success=False)
            assert json.loads(client.call(10))['engine_kv_bytes'] == 0
        client.configure(args.rows, q4=args.q4, q8=args.q8)
        stats = json.loads(client.call(10))
        expected_bytes = args.rows * HKV * ((D//32)*(68 if args.q8 else 36) if args.q4 or args.q8 else D*4) * args.layers
        assert stats['engine_kv_bytes'] == expected_bytes, stats
        positions = [ORIGIN-1, ORIGIN, ORIGIN+1, ORIGIN+args.rows-1]
        q = [0.] * (len(positions)*H*D)
        for layer in range(args.layers):
            value = (layer+1) / 16
            for row in range(0, args.rows, 512):
                rows = min(512, args.rows-row)
                elements = rows * HKV * D
                payload = struct.pack('<4I', row, rows, 3 if args.q8 else 2 if args.q4 else 0, 0)
                if args.q8:
                    blocks = elements//32
                    payload += (b'\0\0' + b'\0'*32) * blocks
                    payload += (struct.pack('<e', value) + b'\x01'*32) * blocks
                elif args.q4:
                    blocks = elements//32
                    payload += (b'\0\0' + b'\x88'*16) * blocks
                    payload += (struct.pack('<e', value) + b'\x99'*16) * blocks
                else:
                    payload += b'\0\0' * elements
                    payload += struct.pack('<e', value) * elements
                client.call(13, payload, session=0, layer=layer)
            out, lse = client.attention(q, positions, layer=layer)
            max_out = max_lse = 0.
            for qi, position in enumerate(positions):
                count = min(args.rows, max(0, position - ORIGIN + 1))
                want_out = value if count else 0.
                values = out[qi*H*D:(qi+1)*H*D]
                err = max(abs(got-want_out) for got in values)
                assert all(math.isfinite(got) for got in values) and err < 3e-4, (layer, qi, err)
                max_out = max(max_out, err)
                for got in lse[qi*H:(qi+1)*H]:
                    if not count:
                        assert got == -math.inf, (layer, qi, got)
                    else:
                        err = abs(got-math.log(count))
                        assert math.isfinite(got) and err < 3e-4, (layer, qi, got)
                        max_lse = max(max_lse, err)
            checks.append(dict(layer=layer, out_max_abs=max_out, lse_max_abs=max_lse))
        # Decode timing excludes upload and covers all layers per token.
        decode = []
        q_one = [0.] * (H*D)
        for token in range(8):
            step = time.perf_counter()
            for layer in range(args.layers):
                client.attention(q_one, [ORIGIN+args.rows-1], layer=layer)
            decode.append(1000*(time.perf_counter()-step))
        upload_requests = [item for item in client.timings if item['op'] == 13]
        upload_data_bytes = sum(item['request_payload_bytes']-16 for item in upload_requests)
        assert upload_data_bytes == expected_bytes
        upload_payload_bytes = sum(item['request_payload_bytes'] for item in upload_requests)
        client.call(3, session=0)
    finally:
        client.socket.close()
    report = dict(success=True, rows=args.rows, layers=args.layers,
                  kv_bytes=expected_bytes, f16_budget_rejected=args.verify_f16_budget, storage="q8_0" if args.q8 else "q4_0" if args.q4 else "f16",
                  upload_wire=3 if args.q8 else 2 if args.q4 else 0,
                  upload_data_bytes=upload_data_bytes, upload_request_payload_bytes=upload_payload_bytes,
                  upload_calls=len(upload_requests), checks=checks, decode_all_layers_ms=decode,
                  elapsed_s=time.perf_counter()-start,
                  attention_compute_us=[item['compute_us'] for item in client.timings if item['op']==14],
                  note='Kernel/RPC timing only; model TPS requires full host inference.')
    result = json.dumps(report, indent=2)
    print(result)
    if args.output:
        with open(args.output, 'w') as stream:
            stream.write(result+'\n')


if __name__ == '__main__':
    main()
