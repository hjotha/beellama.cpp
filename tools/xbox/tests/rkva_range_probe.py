#!/usr/bin/env python3
"""Live Xbox RKVA v2 GPU parity test against independent CPU float64 attention.

No model or Xbox credentials needed: the dedicated RKVA TCP listener must be
running. Inputs are exact F16, queries F32. This tests the negotiated 24/4/256
geometry, global causal positions, split-K OUT+LSE, per-layer counts, replacement,
append, trim, reset, reconfiguration, malformed requests, and v1 compatibility.
"""
import argparse
import json
import math
import random
import socket
import struct
import time

MAGIC = 0x41564B52
NONE = 0xFFFFFFFF
H, HKV, D = 24, 4, 256
ORIGIN = 102400


class Client:
    def __init__(self, host, port, version=2):
        self.socket = socket.create_connection((host, port), timeout=120)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.version, self.cycle = version, 0
        self.timings = []

    def read(self, length):
        data = bytearray()
        while len(data) < length:
            part = self.socket.recv(length - len(data))
            if not part:
                raise RuntimeError('RKVA disconnected')
            data.extend(part)
        return data

    def call(self, op, payload=b'', session=NONE, layer=0, nq=0, success=True):
        self.cycle += 1
        request = struct.pack('<IHHQIIII', MAGIC, self.version, op, self.cycle,
                              session, layer, nq, len(payload))
        start = time.perf_counter()
        self.socket.sendall(request + payload)
        response = struct.unpack('<IHHQiIQQQQ', self.read(56))
        body = self.read(response[5])
        assert response[:4] == (MAGIC, self.version, op, self.cycle), response
        self.timings.append(dict(op=op, elapsed_ms=1000*(time.perf_counter()-start),
                                 compute_us=response[8], status=response[4],
                                 request_payload_bytes=len(payload), response_payload_bytes=len(body)))
        if success:
            assert response[4] == 0, (op, response[4], body)
        else:
            assert response[4] < 0, (op, response[4])
        return body

    def hello(self, capacity, layers=2):
        payload = struct.pack('<16IfQQ', 3, layers, H, HKV, D, capacity,
                              0, 0, 128, 0, 0, 2, 0, 0, 0, 0,
                              1/math.sqrt(D), 0, 0)
        ack = struct.unpack('<IIIIQQQ', self.call(1, payload))
        self.ack = ack
        self.capabilities = ack[5]
        if self.version == 2:
            assert ack[0] & sum(1 << op for op in (12, 13, 14)) == sum(1 << op for op in (12, 13, 14))
        else:
            # Legacy peers must not be offered any v2-only extension.
            assert not ack[0] & sum(1 << op for op in range(12, 21))

    def configure(self, capacity, origin=ORIGIN, success=True, q4=False, q8=False):
        return self.call(12, struct.pack('<4I', origin, capacity, 2 if q8 else int(q4), 0), session=0,
                         success=success)

    def upload(self, k, v, start=0, layer=0, success=True):
        rows = len(k) // (HKV * D)
        payload = struct.pack('<4I', start, rows, 0, 0)
        payload += struct.pack('<' + 'e' * len(k), *k) + struct.pack('<' + 'e' * len(v), *v)
        return self.call(13, payload, session=0, layer=layer, success=success)

    def upload_packed(self, k, v, start=0, layer=0, success=True, wire=2):
        assert wire in (2, 3)
        row_bytes = HKV * (D//32) * (18 if wire == 2 else 34)
        rows = len(k) // row_bytes
        assert len(k) == len(v) == rows * row_bytes
        return self.call(13, struct.pack('<4I', start, rows, wire, 0) + k + v,
                         session=0, layer=layer, success=success)

    def attention(self, q, positions, layer=0):
        payload = struct.pack('<4I', 1, 0, 0, 0)
        payload += struct.pack('<' + 'i' * len(positions), *positions)
        payload += struct.pack('<' + 'f' * len(q), *q)
        response = self.call(14, payload, session=0, layer=layer, nq=len(positions))
        count = len(positions) * H * D
        assert len(response) == (count + len(positions) * H) * 4
        floats = struct.unpack('<' + 'f' * (len(response)//4), response)
        return floats[:count], floats[count:]


def reference(q, positions, k, v, origin=ORIGIN):
    rows = len(k) // (HKV * D)
    output, lse = [], []
    scale = 1 / math.sqrt(D)
    for qi, position in enumerate(positions):
        limit = max(0, min(rows, position - origin + 1))
        for head in range(H):
            if not limit:
                output.extend([0.] * D)
                lse.append(-math.inf)
                continue
            query = q[(qi*H+head)*D:(qi*H+head+1)*D]
            kh = head // (H//HKV)
            scores = []
            for row in range(limit):
                offset = (row * HKV + kh) * D
                scores.append(sum(query[d] * k[offset+d] for d in range(D)) * scale)
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            denominator = sum(weights)
            result = [0.] * D
            for row, weight in enumerate(weights):
                offset = (row * HKV + kh) * D
                for d in range(D):
                    result[d] += weight * v[offset+d]
            output.extend(value / denominator for value in result)
            lse.append(maximum + math.log(denominator))
    return output, lse


def check(client, q, positions, k, v, name, layer=0, origin=ORIGIN):
    actual = client.attention(q, positions, layer)
    expected = reference(q, positions, k, v, origin)
    max_out, max_lse = 0., 0.
    out_error2 = out_ref2 = lse_error2 = lse_ref2 = 0.
    for i, (got, want) in enumerate(zip(actual[0], expected[0])):
        error = abs(got-want)
        max_out = max(max_out, error)
        out_error2 += error*error
        out_ref2 += want*want
        assert math.isfinite(got) and error <= 3e-4 + 3e-4 * abs(want), (name, 'OUT', i, got, want)
    for i, (got, want) in enumerate(zip(actual[1], expected[1])):
        if want == -math.inf:
            assert got == -math.inf, (name, 'LSE-empty', i, got)
        else:
            error = abs(got-want)
            max_lse = max(max_lse, error)
            lse_error2 += error*error
            lse_ref2 += want*want
            assert math.isfinite(got) and error <= 3e-4 + 3e-5 * abs(want), (name, 'LSE', i, got, want)
    return dict(name=name, out_max_abs=max_out, lse_max_abs=max_lse,
                out_nrms=math.sqrt(out_error2/max(out_ref2, 1e-30)),
                lse_nrms=math.sqrt(lse_error2/max(lse_ref2, 1e-30)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=50054)
    parser.add_argument('--rows', type=int, default=1025)
    parser.add_argument('--output')
    storage_group = parser.add_mutually_exclusive_group()
    storage_group.add_argument('--q4', action='store_true')
    storage_group.add_argument('--q8', action='store_true')
    args = parser.parse_args()
    assert 514 <= args.rows <= 28672
    rng = random.Random(416029)
    # Float round-trips make the CPU oracle consume exactly the wire values.
    def f16(value): return struct.unpack('<e', struct.pack('<e', value))[0]
    def f32(value): return struct.unpack('<f', struct.pack('<f', value))[0]
    k = [f16(rng.uniform(-1, 1)) for _ in range(args.rows*HKV*D)]
    v = [f16(rng.uniform(-2, 2)) for _ in range(args.rows*HKV*D)]
    raw_k = raw_v = None
    if args.q4 or args.q8:
        def packed_plane():
            raw, values = bytearray(), []
            for block in range(args.rows * HKV * (D//32)):
                scale = f16((1 if block % 2 else -1) * rng.uniform(0.01, 0.5))
                if args.q8:
                    scale = f16(scale / 16)
                    quants = [rng.randrange(-128, 128) for _ in range(32)]
                    quants[:6] = [-128, 127, 0, -127, 1, -1]
                    raw.extend(struct.pack('<e32b', scale, *quants))
                    values.extend(quant*scale for quant in quants)
                else:
                    quants = bytearray(rng.randrange(256) for _ in range(16))
                    quants[:3] = bytes([0xf0, 0x0f, 0x88])
                    raw.extend(struct.pack('<e', scale))
                    raw.extend(quants)
                    # GGML dequantize_row_q4_0: low 16, then high 16.
                    values.extend(((byte & 15)-8)*scale for byte in quants)
                    values.extend(((byte >> 4)-8)*scale for byte in quants)
            return bytes(raw), values
        raw_k, k = packed_plane()
        raw_v, v = packed_plane()
    positions = [ORIGIN-1, ORIGIN, ORIGIN+1, ORIGIN+args.rows-1]
    q = [f32(rng.uniform(-1, 1)) for _ in range(len(positions)*H*D)]
    client = Client(args.host, args.port)
    checks = []
    try:
        client.hello(args.rows)
        if args.q4 or args.q8:
            assert client.capabilities & (2 if args.q8 else 1), "server lacks requested quantized range storage"
        client.call(2, struct.pack('<4I', 0, 0, args.rows, int(bool(client.capabilities & 3))))
        if client.capabilities & 3:
            assert json.loads(client.call(10))['engine_kv_bytes'] == 0
        client.configure(args.rows, q4=args.q4, q8=args.q8)
        checks.append(check(client, q, positions, [], [], 'initial-empty'))
        def upload(begin, count, start=0, success=True):
            if args.q4 or args.q8:
                stride = HKV * (D//32) * (34 if args.q8 else 18)
                client.upload_packed(raw_k[begin*stride:(begin+count)*stride],
                                     raw_v[begin*stride:(begin+count)*stride],
                                     start=start, success=success, wire=3 if args.q8 else 2)
            else:
                stride = HKV * D
                client.upload(k[begin*stride:(begin+count)*stride],
                              v[begin*stride:(begin+count)*stride],
                              start=start, success=success)
        upload(0, 513)
        upload(513, args.rows-513, start=513)
        if client.capabilities & 2:
            client.call(12, struct.pack('<4I', ORIGIN, args.rows, 3, 0), session=0, success=False)
        if args.q4 or args.q8:
            stride = HKV*(D//32)*(34 if args.q8 else 18)
            row = raw_k[:stride] + raw_v[:stride]
            client.call(13, struct.pack('<4I', 0, 1, 0, 0)+row, session=0, success=False)
            client.call(13, struct.pack('<4I', 0, 1, 3 if args.q8 else 2, 0)+row[:-1],
                        session=0, success=False)
        checks.append(check(client, q, positions, k, v, 'split-k-boundaries'))
        checks.append(check(client, q, positions, [], [], 'other-layer-empty', layer=1))
        upload(0, 1, start=1, success=False)
        upload(0, 3)
        checks.append(check(client, q, positions, k[:3*HKV*D], v[:3*HKV*D], 'replace-shrinks'))
        client.call(7, struct.pack('<i', ORIGIN+2), session=0)
        checks.append(check(client, q, positions, k[:2*HKV*D], v[:2*HKV*D], 'trim-global-position'))
        client.call(6, session=0)
        checks.append(check(client, q, positions, [], [], 'reset-preserves-origin'))
        upload(0, 1)
        checks.append(check(client, q, positions, k[:HKV*D], v[:HKV*D], 'upload-after-reset'))
        client.configure(args.rows, origin=2*ORIGIN, q4=args.q4, q8=args.q8)
        checks.append(check(client, q, positions, [], [], 'configure-clears-and-rebases', origin=2*ORIGIN))
        client.configure(args.rows+1, success=False)
        if client.capabilities & 3:
            current = 'q8_0' if args.q8 else 'q4_0' if args.q4 else 'f16'
            for name, flag, row_bytes in [('f16', 0, HKV*D*2),
                                          ('q4_0', 1, HKV*(D//32)*18),
                                          ('q8_0', 2, HKV*(D//32)*34)]:
                if name == current or (flag and not client.capabilities & flag):
                    continue
                client.configure(args.rows, q4=flag == 1, q8=flag == 2)
                checks.append(check(client, q, positions, [], [], 'format-switch-'+name+'-clears'))
                assert json.loads(client.call(10))['engine_kv_bytes'] == args.rows*row_bytes*2*2
                client.configure(args.rows, q4=args.q4, q8=args.q8)
                checks.append(check(client, q, positions, [], [], 'format-switch-back-'+current+'-clears'))
        client.call(3, session=0)
    finally:
        client.socket.close()
    legacy = Client(args.host, args.port, version=1)
    try:
        legacy.call(9)
        legacy.hello(args.rows)
        legacy.call(12, struct.pack('<4I', ORIGIN, args.rows, 0, 0), success=False)
    finally:
        legacy.socket.close()
    report = dict(success=True, geometry=dict(n_head=H, n_head_kv=HKV, head_dim=D),
                  origin=ORIGIN, rows=args.rows, storage="q8_0" if args.q8 else "q4_0" if args.q4 else "f16",
                  upload_wire=3 if args.q8 else 2 if args.q4 else 0,
                  kv_row_bytes=HKV*((D//32)*(34 if args.q8 else 18) if args.q4 or args.q8 else D*2), checks=checks,
                  timings=client.timings, legacy_v1=True)
    result = json.dumps(report, indent=2)
    print(result)
    if args.output:
        with open(args.output, 'w') as stream:
            stream.write(result+'\n')


if __name__ == '__main__':
    main()
