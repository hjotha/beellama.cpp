#!/usr/bin/env python3
"""Native bulk vs serial op16/op14 GPU parity; --execute requires console lease.

Same immutable native upload/query bytes in two sessions, covering maximum
256-query envelopes, seals, tail wrap, resident rollback/reappend, a rejected
last segment followed by retry, and both native tail policies. Kernels stay
identical: every OUT/LSE byte must match. Offline default opens no connection.
"""
import argparse
import json
import os
import struct
import time


# Keep the independent NumPy oracle within the single-worker test budget.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"


def run(host, port):
    import numpy as np
    import rkva_kvarn_serial_prefix_probe as p
    import rkva_kvarn_serial_prefix_model as sp
    from rkva_range_probe import Client
    p.CAPACITY = 2048
    client = Client(host, port)
    results = []
    try:
        flags = p.PROTO_F16 | p.PROTO_F32 | p.PROTO_NATIVE_REWIND_V2 | p.PROTO_SERIAL_PREFIX
        ack = struct.unpack('<IIIIQQQ', client.call(1, p.hello_payload(1, flags)))
        assert ack[0] & (1 << 22) and ack[5] & (1 << 8), 'native bulk not negotiated'
        for tail in (0, 128):
            rng = np.random.default_rng(270 + tail)
            layer = p.Layer(np, sp, rng, tail)
            generation = {}
            for session in (0, 1):
                client.call(2, struct.pack('<4I', session, session, p.CAPACITY, 1))
                body = client.call(15, struct.pack('<18I', *p.descriptor(tail)), session=session)
                generation[session] = struct.unpack('<Q', body)[0]
            client.call(20, struct.pack('<IIQ', 0, 2, 0))  # qualified fine/mq2 kernel

            def batch(n, label, negative=False):
                chunks = []
                first = layer.live
                layer.canon.append(n)
                limit = n if n <= 9 else 8
                for offset in range(0, n, limit):
                    count = min(limit, n - offset)
                    live = first + offset + count
                    upload = layer.upload(live, live == count)
                    positions = list(range(p.ORIGIN + live - count, p.ORIGIN + live))
                    q = rng.normal(size=(count, p.H, p.D)).astype('<f4')
                    query = struct.pack('<4I', 1, 0, 0, 0) + struct.pack('<%di' % count, *positions) + q.tobytes()
                    chunks.append((upload, query, count))
                bulk = struct.pack('<4I', 1, len(chunks), n, 0) + b''.join(
                    struct.pack('<2I', len(upload), count) + upload + query
                    for upload, query, count in chunks)
                expected_o, expected_l = [], []
                client.call(20, struct.pack('<IIQ', 3, 2, 0))
                start = time.monotonic()
                for upload, query, count in chunks:
                    client.call(16, upload, session=0)
                    generation[0] += 1
                    reply = bytes(client.call(14, query, session=0, nq=count))
                    size = count * p.H * p.D * 4
                    expected_o.append(reply[:size]); expected_l.append(reply[size:])
                serial_ms = (time.monotonic() - start) * 1000
                profile = json.loads(bytes(client.call(20, struct.pack('<IIQ', 8, 0, 0))))
                serial_counters = profile.get('engine', profile)['counters']
                assert serial_counters['submits'] == 2 * len(chunks), serial_counters
                if negative:
                    # Corrupt the last query position: admission must reject
                    # before the first segment mutates the second session.
                    corrupted = bytearray(bulk)
                    last = chunks[-1]
                    pos_offset = len(bulk) - len(last[1]) + 16
                    struct.pack_into('<i', corrupted, pos_offset, -1)
                    client.call(22, corrupted, session=1, nq=n, success=False)
                client.call(20, struct.pack('<IIQ', 3, 2, 0))
                start = time.monotonic()
                got = bytes(client.call(22, bulk, session=1, nq=n))
                bulk_ms = (time.monotonic() - start) * 1000
                profile = json.loads(bytes(client.call(20, struct.pack('<IIQ', 8, 0, 0))))
                bulk_counters = profile.get('engine', profile)['counters']
                assert bulk_counters['submits'] == 1, bulk_counters
                assert bulk_counters['fence_blocking_waits'] + bulk_counters['fence_already_complete'] == 1
                generation[1] += len(chunks)
                expected = b''.join(expected_o) + b''.join(expected_l)
                assert got == expected, '%s tail%d OUT/LSE byte mismatch' % (label, tail)
                results.append(dict(label=label, tail=tail, nq=n, live=layer.live,
                                    exact=True, serial_ms=serial_ms, bulk_ms=bulk_ms,
                                    serial_submits=serial_counters['submits'], bulk_submits=bulk_counters['submits'],
                                    bulk_fence_blocking_waits=bulk_counters['fence_blocking_waits'],
                                    bulk_fence_already_complete=bulk_counters['fence_already_complete'],
                                    serial_transactions=2 * len(chunks), bulk_transactions=1))

            batch(256, 'initial maximum envelope, two seals and ring wrap')
            batch(9, 'verification nine')
            batch(64, 'late malformed segment rejected then valid retry', True)
            batch(256, 'append maximum envelope')
            batch(55, 'reach seal640 before rollback')
            for back in (1, 8, 9):
                keep = layer.live - back
                block = layer.rewind_layer(keep)
                for session in (0, 1):
                    head = struct.pack('<QIIII', generation[session], p.ORIGIN + keep, 1, tail, 0)
                    client.call(21, head + block, session=session)
                    generation[session] += 1
                layer.canon.rewind(keep)
                batch(back, 'rollback%d and new-content reappend' % back)
            for session in (0, 1):
                client.call(3, struct.pack('<4I', session, 0, 0, 0), session=session)
    finally:
        client.socket.close()
    return dict(ok=True, server_version=ack[3], checks=results)


def run_cuda_fixtures(host, port, directory):
    import numpy as np
    import rkva_kvarn_serial_prefix_probe as p
    from rkva_kvarn_gpu_fixture import load_fixture
    from rkva_range_probe import Client
    results = []
    oracle_indices = {0, 14, 15, 16, 30, 31, 32, 46, 47, 48, 62, 63, 64, 78, 79}
    for tail in (0, 128):
        fixture = load_fixture(directory, tail)
        p.ORIGIN, p.CAPACITY = fixture.origin, fixture.capacity
        client = Client(host, port)
        try:
            flags = p.PROTO_F16 | p.PROTO_F32 | p.PROTO_NATIVE_REWIND_V2 | p.PROTO_SERIAL_PREFIX
            ack = struct.unpack('<IIIIQQQ', client.call(1, p.hello_payload(1, flags)))
            assert ack[0] & (1 << 22) and ack[5] & (1 << 8), 'native bulk not negotiated'
            for session in (0, 1):
                client.call(2, struct.pack('<4I', session, session, fixture.capacity, 1))
                client.call(15, struct.pack('<18I', *p.descriptor(tail)), session=session)
            for first in range(0, len(fixture.segments), 32):
                segments = fixture.segments[first:first + 32]
                n = 8 * len(segments)
                chunks = []
                for segment in segments:
                    query = struct.pack('<4I', 1, 0, 0, 0) + segment['positions'].tobytes() + segment['q'].tobytes()
                    chunks.append((segment['upload'], query))
                client.call(20, struct.pack('<IIQ', 3, 2, 0))
                expected_o, expected_l = [], []
                start = time.monotonic()
                for upload, query in chunks:
                    client.call(16, upload, session=0)
                    reply = bytes(client.call(14, query, session=0, nq=8))
                    expected_o.append(reply[:8 * 24 * 256 * 4])
                    expected_l.append(reply[8 * 24 * 256 * 4:])
                serial_ms = (time.monotonic() - start) * 1000
                profile = json.loads(bytes(client.call(20, struct.pack('<IIQ', 8, 0, 0))))
                serial_counters = profile.get('engine', profile)['counters']
                assert serial_counters['submits'] == 2 * len(chunks)
                bulk = struct.pack('<4I', 1, len(chunks), n, 0) + b''.join(
                    struct.pack('<2I', len(upload), 8) + upload + query for upload, query in chunks)
                client.call(20, struct.pack('<IIQ', 3, 2, 0))
                start = time.monotonic()
                reply = bytes(client.call(22, bulk, session=1, nq=n))
                bulk_ms = (time.monotonic() - start) * 1000
                profile = json.loads(bytes(client.call(20, struct.pack('<IIQ', 8, 0, 0))))
                bulk_counters = profile.get('engine', profile)['counters']
                assert bulk_counters['submits'] == 1
                assert bulk_counters['fence_blocking_waits'] + bulk_counters['fence_already_complete'] == 1
                assert reply == b''.join(expected_o) + b''.join(expected_l), 'CUDA fixture serial/bulk byte mismatch'
                out = np.frombuffer(reply, dtype='<f4', count=n * 24 * 256).reshape(n, 24, 256)
                lse = np.frombuffer(reply, dtype='<f4', offset=n * 24 * 256 * 4).reshape(n, 24)
                checks = []
                for i in range(len(segments)):
                    if first + i not in oracle_indices: continue
                    check = p.compare(np, out[i * 8:(i + 1) * 8], lse[i * 8:(i + 1) * 8],
                                      *fixture.expected(first + i))
                    assert check['ok'], (tail, first + i, check)
                    checks.append(dict(index=first + i, vs_fp64=check))
                results.append(dict(tail=tail, first_segment=first, queries=n, exact=True,
                                    serial_ms=serial_ms, bulk_ms=bulk_ms,
                                    serial_submits=serial_counters['submits'], bulk_submits=bulk_counters['submits'],
                                    bulk_fence_blocking_waits=bulk_counters['fence_blocking_waits'],
                                    bulk_fence_already_complete=bulk_counters['fence_already_complete'],
                                    checks=checks))
        finally:
            client.socket.close()
    return dict(ok=True, cuda_fixtures=str(directory), checks=results)


def run_f16(host, port):
    import numpy as np
    import rkva_kvarn_serial_prefix_probe as p
    import rkva_kvarn_serial_prefix_model as sp
    from rkva_range_probe import Client
    p.CAPACITY = 2048
    client = Client(host, port)
    results = []
    try:
        flags = p.PROTO_F16 | p.PROTO_F32 | p.PROTO_NATIVE_REWIND_V2 | p.PROTO_SERIAL_PREFIX
        ack = struct.unpack('<IIIIQQQ', client.call(1, p.hello_payload(1, flags)))
        assert ack[0] & (1 << 22) and ack[5] & (1 << 8), 'native bulk not negotiated'
        assert ack[5] & (1 << 9), 'native bulk f16 (RKVA_CAP_KVARN_BULK_F16) not negotiated'
        for tail in (0, 128):
            rng = np.random.default_rng(370 + tail)
            layer = p.Layer(np, sp, rng, tail)
            generation = {}
            for session in (0, 1):
                client.call(2, struct.pack('<4I', session, session, p.CAPACITY, 1))
                body = client.call(15, struct.pack('<18I', *p.descriptor(tail)), session=session)
                generation[session] = struct.unpack('<Q', body)[0]
            client.call(20, struct.pack('<IIQ', 0, 2, 0))
            for n, label, negative in (
                (1, 'decode single query f16 wire', False),
                (9, 'speculative verification nine f16 wire', True),
                (64, 'chunked prefill 64 f16 wire', False),
                (256, 'maximum envelope 256 f16 wire', False),
            ):
                chunks_f32, chunks_f16 = [], []
                first = layer.live
                layer.canon.append(n)
                limit = n if n <= 9 else 8
                for offset in range(0, n, limit):
                    count = min(limit, n - offset)
                    live = first + offset + count
                    upload = layer.upload(live, live == count)
                    positions = list(range(p.ORIGIN + live - count, p.ORIGIN + live))
                    q_f32 = rng.normal(size=(count, p.H, p.D)).astype('<f4')
                    q_f16 = q_f32.astype('<f2')
                    q_f32_quant = q_f16.astype('<f4')
                    pos_bytes = struct.pack('<%di' % count, *positions)
                    query_f32 = struct.pack('<4I', 1, 0, 0, 0) + pos_bytes + q_f32_quant.tobytes()
                    query_f16 = struct.pack('<4I', 0, 0, 0, 0) + pos_bytes + q_f16.tobytes()
                    chunks_f32.append((upload, query_f32, count))
                    chunks_f16.append((upload, query_f16, count))
                bulk_f32 = struct.pack('<4I', 1, len(chunks_f32), n, 0) + b''.join(
                    struct.pack('<2I', len(u), c) + u + q for u, q, c in chunks_f32)
                bulk_f16 = struct.pack('<4I', 1, len(chunks_f16), n, 0) + b''.join(
                    struct.pack('<2I', len(u), c) + u + q for u, q, c in chunks_f16)
                if negative:
                    corrupted = bytearray(bulk_f16)
                    last_q_offset = len(bulk_f16) - (chunks_f16[-1][2] * p.H * p.D * 2)
                    struct.pack_into('<H', corrupted, last_q_offset, 0x7E00)  # FP16 qNaN
                    client.call(22, corrupted, session=1, nq=n, success=False)
                ref_reply = bytes(client.call(22, bulk_f32, session=0, nq=n))
                generation[0] += len(chunks_f32)
                f16_reply = bytes(client.call(22, bulk_f16, session=1, nq=n))
                generation[1] += len(chunks_f16)
                out_elems = n * p.H * p.D
                lse_elems = n * p.H
                assert len(f16_reply) == out_elems * 2 + lse_elems * 4
                ref_out = np.frombuffer(ref_reply, dtype='<f4', count=out_elems)
                ref_lse = np.frombuffer(ref_reply, dtype='<f4', offset=out_elems * 4, count=lse_elems)
                got_out = np.frombuffer(f16_reply, dtype='<f2', count=out_elems).astype(np.float32)
                got_lse = np.frombuffer(f16_reply, dtype='<f4', offset=out_elems * 2, count=lse_elems)
                max_abs_out = float(np.max(np.abs(got_out - ref_out)))
                max_abs_lse = float(np.max(np.abs(got_lse - ref_lse)))
                assert max_abs_out <= 2e-2 and max_abs_lse == 0.0, (label, tail, max_abs_out, max_abs_lse)
                results.append(dict(label=label, tail=tail, nq=n,
                                    req_bytes_f32=len(bulk_f32), req_bytes_f16=len(bulk_f16),
                                    rsp_bytes_f32=len(ref_reply), rsp_bytes_f16=len(f16_reply),
                                    max_abs_out=max_abs_out, max_abs_lse=max_abs_lse))
            for session in (0, 1):
                client.call(3, struct.pack('<4I', session, 0, 0, 0), session=session)
    finally:
        client.socket.close()
    return dict(ok=True, wire='f16', server_version=ack[3], checks=results)


def offline_self_test():
    import numpy as np
    rng = np.random.default_rng(20261019)
    checks = []
    for n in (1, 9, 64, 256):
        q_f32 = rng.normal(scale=0.5, size=(n, 24, 256)).astype(np.float32)
        q_f16 = q_f32.astype(np.float16)
        k_f32 = rng.normal(scale=0.5, size=(128, 4, 256)).astype(np.float32)
        v_f32 = rng.normal(scale=0.5, size=(128, 4, 256)).astype(np.float32)
        k_rep = np.repeat(k_f32, 6, axis=1)
        v_rep = np.repeat(v_f32, 6, axis=1)
        scale = np.float32(1.0 / np.sqrt(256.0))
        scores_f32 = np.einsum('qhd,khd->qhk', q_f32, k_rep) * scale
        scores_f16q = np.einsum('qhd,khd->qhk', q_f16.astype(np.float32), k_rep) * scale
        m_f32 = np.max(scores_f32, axis=-1, keepdims=True)
        m_f16q = np.max(scores_f16q, axis=-1, keepdims=True)
        w_f32 = np.exp(scores_f32 - m_f32)
        w_f16q = np.exp(scores_f16q - m_f16q)
        lse_f32 = (np.squeeze(m_f32, axis=-1) + np.log(np.sum(w_f32, axis=-1))).astype(np.float32)
        lse_f16q = (np.squeeze(m_f16q, axis=-1) + np.log(np.sum(w_f16q, axis=-1))).astype(np.float32)
        out_f32 = (np.einsum('qhk,khd->qhd', w_f32, v_rep) / np.sum(w_f32, axis=-1, keepdims=True)).astype(np.float32)
        out_f16_wire = (np.einsum('qhk,khd->qhd', w_f16q, v_rep) / np.sum(w_f16q, axis=-1, keepdims=True)).astype(np.float16).astype(np.float32)
        max_abs_out = float(np.max(np.abs(out_f16_wire - out_f32)))
        mean_abs_out = float(np.mean(np.abs(out_f16_wire - out_f32)))
        max_abs_lse = float(np.max(np.abs(lse_f16q - lse_f32)))
        cos = float(np.dot(out_f16_wire.ravel(), out_f32.ravel()) /
                    (np.linalg.norm(out_f16_wire.ravel()) * np.linalg.norm(out_f32.ravel())))
        assert max_abs_out < 1e-3 and max_abs_lse < 2e-3 and cos > 0.99999
        q_bytes_f32 = n * 24 * 256 * 4
        q_bytes_f16 = n * 24 * 256 * 2
        rsp_bytes_f32 = n * 24 * 256 * 4 + n * 24 * 4
        rsp_bytes_f16 = n * 24 * 256 * 2 + n * 24 * 4
        checks.append(dict(
            nq=n,
            q_bytes_f32=q_bytes_f32,
            q_bytes_f16=q_bytes_f16,
            rsp_bytes_f32=rsp_bytes_f32,
            rsp_bytes_f16=rsp_bytes_f16,
            saved_bytes_per_layer=(q_bytes_f32 - q_bytes_f16) + (rsp_bytes_f32 - rsp_bytes_f16),
            max_abs_out=max_abs_out,
            mean_abs_out=mean_abs_out,
            max_abs_lse=max_abs_lse,
            cosine_similarity=cos,
        ))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='192.168.1.26')
    parser.add_argument('--port', type=int, default=50054)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--wire', choices=('f32', 'f16', 'both'), default='f32')
    parser.add_argument('--cuda-fixtures', help='real GPU op16 packet fixture directory')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.execute:
        if args.cuda_fixtures:
            result = run_cuda_fixtures(args.host, args.port, args.cuda_fixtures)
        elif args.wire == 'f16':
            result = run_f16(args.host, args.port)
        elif args.wire == 'both':
            result = dict(ok=True, f32=run(args.host, args.port), f16=run_f16(args.host, args.port))
        else:
            result = run(args.host, args.port)
    else:
        result = dict(
            execute=False,
            plan='2 tail policies; exact bulk/serial F32 bytes and negotiated FP16 Q/OUT bulk gate; seal/wrap/rollback/retry gates',
            offline_f16_checks=offline_self_test(),
        )
    text = json.dumps(result, indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
