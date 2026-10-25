#!/usr/bin/env python3
"""Future live disk round-trip gate. NO connection unless --execute is explicit.

Requires a reserved console with the future snapshot candidate deployed. This is
not a model/TPS benchmark and must not run concurrently with performance work.
"""
import argparse
import hashlib
import json
import random
import struct

from rkva_range_probe import Client, D, H, HKV, ORIGIN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--host')
    parser.add_argument('--port', type=int, default=50054)
    parser.add_argument('--storage', choices=['f16', 'q4_0', 'q8_0'], default='q8_0')
    args = parser.parse_args()
    if not args.execute:
        print('Prepared only. Reserve the Xbox and pass --execute --host ADDRESS to connect.')
        return
    if not args.host:
        parser.error('--execute requires --host')
    capacity, rows, layers = 256, 129, 2
    rng = random.Random(817019)
    wire = {'f16': 0, 'q4_0': 2, 'q8_0': 3}[args.storage]
    stride = HKV * (D * 2 if wire == 0 else (D // 32) * (18 if wire == 2 else 34))

    def plane():
        if wire == 0:
            return struct.pack('<' + 'e' * (rows * HKV * D),
                               *(rng.uniform(-1, 1) for _ in range(rows * HKV * D)))
        result = bytearray()
        for _ in range(rows * HKV * (D // 32)):
            scale = (-1 if rng.randrange(2) else 1) / 32
            if wire == 2:
                result += struct.pack('<e', scale) + bytes(rng.randrange(256) for _ in range(16))
            else:
                result += struct.pack('<e32b', scale, *(rng.randrange(-128, 128) for _ in range(32)))
        return bytes(result)

    kv = [(plane(), plane()) for _ in range(layers)]
    # A synthetic immutable fixture identity for this primitive gate. Real Bee
    # uses SHA256 of the entire authoritative native sequence file with tokens.
    key = hashlib.sha256(struct.pack('<4I', capacity, rows, ORIGIN, wire) +
                         b''.join(k + v for k, v in kv)).digest()
    request = key + struct.pack('<4I', rows, 1, 0, 0)
    client = Client(args.host, args.port)
    client.socket.settimeout(600)
    checks = []
    try:
        client.hello(capacity, layers)
        assert client.capabilities & 8, 'candidate lacks disk snapshots'
        client.call(2, struct.pack('<4I', 0, 0, capacity, 1))
        client.configure(capacity, q4=wire == 2, q8=wire == 3)

        def upload(layer, begin=0, count=rows, start=0, zero=False):
            k, v = kv[layer]
            k, v = k[begin * stride:(begin + count) * stride], v[begin * stride:(begin + count) * stride]
            if zero:
                k, v = bytes(len(k)), bytes(len(v))
            client.call(13, struct.pack('<4I', start, count, wire, 0) + k + v, session=0, layer=layer)

        upload(0)
        client.call(17, request, session=0, success=False)  # no mixed-layer snapshots
        upload(1)
        q = [rng.uniform(-1, 1) for _ in range(H * D)]
        original = [client.attention(q, [ORIGIN + rows - 1], layer) for layer in range(layers)]
        ack = struct.unpack('<32sIIQ', client.call(17, request, session=0))
        assert ack[:3] == (key, rows, layers) and ack[3] == 2 * layers * rows * stride
        checks.append('all-layer-save')
        client.call(17, request, session=0)  # immutable repeated save
        client.call(6, session=0)
        client.call(18, request, session=0)
        assert [client.attention(q, [ORIGIN + rows - 1], layer) for layer in range(layers)] == original
        checks.append('reset-local-disk-restore-bitexact')
        # Future rows cannot survive a rewind to a disk checkpoint.
        for layer in range(layers):
            upload(layer, begin=0, count=1, start=rows, zero=True)
        client.call(18, request, session=0)
        assert [client.attention(q, [ORIGIN + rows - 1], layer) for layer in range(layers)] == original
        for layer in range(layers):
            upload(layer, begin=0, count=1, start=rows, zero=True)
        checks.append('restore-publishes-counts-and-discards-stale-future')
        # Miss/incompatible geometry must preserve live state, enabling upload fallback.
        missing = hashlib.sha256(key + b'missing').digest() + request[32:]
        client.call(18, missing, session=0, success=False)
        client.configure(capacity, origin=ORIGIN + 128, q4=wire == 2, q8=wire == 3)
        client.call(18, request, session=0, success=False)
        checks.append('missing-key-and-incompatible-geometry-fail-closed')
        client.call(17, key + struct.pack('<4I', 0, 1, 0, 0), session=0, success=False)
        client.call(18, key + struct.pack('<4I', rows, 2, 0, 0), session=0, success=False)
        checks.append('zero-rows-and-unknown-revision-rejected')
        print(json.dumps({'storage': args.storage, 'checks': checks, 'timings': client.timings}, indent=2))
    finally:
        client.socket.close()


if __name__ == '__main__':
    main()
