#!/usr/bin/env python3
"""Gated live A/B probe for the RKVA Q4_0 range kernels (mode 0 vs optimized 1-3).

Offline by default: without --execute it prints the plan and opens no socket.
With --execute it talks to an already running dedicated RKVA listener (no model,
no credentials). The console must be leased to the caller; this script never
deploys, restarts, or changes anything outside its own RKVA session.

Per checkpoint (live overflow rows), every mode is run on the same nq=1 decode
queries and compared with a float64 oracle on the exact dequantized Q4_0 data
and with mode 0. Then a decode loop (append one row + attend, like one layer of
a decoded token) runs per mode with op 20 profiling enabled and reset, and the
engine report is fetched with report_only. Shape: H24/HKV4/D256, origin 102400.
"""
import argparse
import json
import struct
import sys

sys.path.insert(0, __import__('os').path.dirname(__file__))

RKVA_RANGE_PROFILE = 20
PROFILE_ENABLE, PROFILE_RESET, PROFILE_REPORT_ONLY = 1, 2, 8
MODES = (0, 1, 2, 3)
ROW_BYTES = 4 * (256 // 32) * 18  # one plane (K or V) of one row, Q4_0


def profile(client, flags, mode):
    body = client.call(RKVA_RANGE_PROFILE, struct.pack('<IIQ', flags, mode, 0))
    return json.loads(bytes(body).decode('utf-8'))


def make_rows(rng, rows, np):
    planes = []
    for _ in range(2):
        blocks = rows * 4 * 8
        scale = rng.uniform(0.01, 0.06, blocks).astype(np.float16)
        quant = rng.integers(0, 256, (blocks, 16), dtype=np.uint8)
        raw = np.zeros((blocks, 18), np.uint8)
        raw[:, :2] = scale.view(np.uint8).reshape(blocks, 2)
        raw[:, 2:] = quant
        low = (quant & 15).astype(np.float64) - 8
        high = (quant >> 4).astype(np.float64) - 8
        values = np.concatenate([low, high], axis=1) * scale.astype(np.float64)[:, None]
        planes.append((raw.tobytes(), values.reshape(rows, 4, 256)))
    return planes


def oracle(np, q, keys, values, limit, scale):
    out = np.zeros((24, 256))
    lse = np.full(24, -np.inf)
    if limit <= 0:
        return out, lse
    for head in range(24):
        kv = head // 6
        scores = keys[:limit, kv, :] @ q[head] * scale
        peak = scores.max()
        weights = np.exp(scores - peak)
        total = weights.sum()
        out[head] = weights @ values[:limit, kv, :] / total
        lse[head] = peak + np.log(total)
    return out, lse


def compare(np, got_out, got_lse, want_out, want_lse):
    finite = np.isfinite(want_lse)
    if not np.array_equal(np.isfinite(got_lse), finite):
        return dict(ok=False, reason='LSE finiteness mismatch')
    out_err = float(np.max(np.abs(got_out - want_out))) if got_out.size else 0.
    lse_err = float(np.max(np.abs(got_lse[finite] - want_lse[finite]))) if finite.any() else 0.
    ok = bool(np.all(np.abs(got_out - want_out) <= 3e-4 + 3e-4 * np.abs(want_out)) and
              np.all(np.abs(got_lse[finite] - want_lse[finite]) <= 3e-4 + 3e-5 * np.abs(want_lse[finite])))
    return dict(ok=ok, out_max_abs=out_err, lse_max_abs=lse_err)


def warm_bucket(report, mode):
    kernel = 'reference' if mode == 0 else 'optimized-%d' % mode
    engine = report.get('engine', {})
    return {key: value for key, value in engine.get('buckets', {}).items()
            if key.startswith('attention/q4_0/%s/nq1/' % kernel) and key.endswith('/warm')}


def execute(args):
    import numpy as np
    from rkva_range_probe import Client, ORIGIN
    rng = np.random.default_rng(args.seed)
    capacity = args.capacity
    checkpoints = sorted({c for c in args.checkpoints if 0 < c <= capacity - args.decode_steps})
    (k_raw, k_val), (v_raw, v_val) = make_rows(rng, capacity, np)
    scale = 1 / 16.
    client = Client(args.host, args.port)
    report = dict(success=False, host=args.host, port=args.port, capacity=capacity,
                  checkpoints=checkpoints, decode_steps=args.decode_steps, parity=[], decode=[])
    try:
        client.hello(capacity, layers=1)
        if not client.capabilities & (1 << 5):
            raise RuntimeError('listener lacks RKVA_CAP_RANGE_PROFILE')
        client.call(2, struct.pack('<4I', 0, 0, capacity, 1))
        client.configure(capacity, q4=True)
        live = 0

        def upload_to(target):
            nonlocal live
            while live < target:
                rows = min(512, target - live)
                client.upload_packed(k_raw[live*ROW_BYTES:(live+rows)*ROW_BYTES],
                                     v_raw[live*ROW_BYTES:(live+rows)*ROW_BYTES], start=live)
                live += rows

        profile(client, PROFILE_RESET, 0)
        for checkpoint in checkpoints:
            upload_to(checkpoint)
            positions = [ORIGIN + checkpoint - 1, ORIGIN + checkpoint // 2, ORIGIN - 1]
            for position in positions:
                q = rng.standard_normal((24, 256))
                limit = max(0, min(live, position - ORIGIN + 1))
                want = oracle(np, q, k_val, v_val, limit, scale)
                baseline = None
                for mode in MODES:
                    profile(client, PROFILE_RESET, mode)
                    out, lse = client.attention(q.astype(np.float32).ravel().tolist(), [position])
                    got = (np.array(out, np.float64).reshape(24, 256), np.array(lse, np.float64))
                    entry = dict(rows=checkpoint, position=position, causal_rows=limit, mode=mode,
                                 vs_oracle=compare(np, *got, *want))
                    if baseline is None:
                        baseline = got
                    else:
                        entry['vs_mode0'] = compare(np, *got, *baseline)
                    report['parity'].append(entry)
        upload_to(capacity - args.decode_steps)
        base_live = live
        for mode in MODES:
            # Uploads are append-only (start_row 0 replaces). Rebuild the same
            # base window so every mode appends the same rows.
            if live != base_live:
                live = 0
                upload_to(base_live)
            profile(client, PROFILE_ENABLE | PROFILE_RESET, mode)
            for _ in range(args.warmup + args.decode_steps):
                if live < capacity and _ >= args.warmup:
                    upload_to(live + 1)
                q = rng.standard_normal(24 * 256).astype(np.float32).tolist()
                client.attention(q, [ORIGIN + live - 1])
            full = profile(client, PROFILE_REPORT_ONLY, 0)
            report['decode'].append(dict(mode=mode, warm_nq1=warm_bucket(full, mode), report=full))
        profile(client, PROFILE_RESET, 0)
        client.call(3, session=0)
        report['success'] = all(e['vs_oracle']['ok'] and e.get('vs_mode0', {'ok': True})['ok']
                                for e in report['parity'])
    finally:
        try:
            profile(client, 0, 0)
        except Exception:  # best effort: leave the listener in reference, profiling off
            pass
        client.socket.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--execute', action='store_true',
                        help='actually connect; requires the caller to hold the console lease')
    parser.add_argument('--host')
    parser.add_argument('--port', type=int, default=50054)
    parser.add_argument('--capacity', type=int, default=28672)
    parser.add_argument('--checkpoints', type=int, nargs='*',
                        default=[1, 64, 257, 4096, 16384, 28000])
    parser.add_argument('--decode-steps', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=16)
    parser.add_argument('--seed', type=int, default=261)
    parser.add_argument('--output')
    args = parser.parse_args()
    if not args.execute:
        plan = dict(execute=False, note='offline plan only; pass --execute --host H to run',
                    modes=list(MODES), capacity=args.capacity, checkpoints=args.checkpoints,
                    decode_steps=args.decode_steps, warmup=args.warmup,
                    ops=['hello', 'session(2)', 'range_config(12,q4_0)', 'range_upload(13,wire2)',
                         'range_attention(14,nq1)', 'range_profile(20)', 'close(3)'])
        print(json.dumps(plan, indent=2))
        return 0
    if not args.host:
        parser.error('--execute needs --host')
    report = execute(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        with open(args.output, 'w') as stream:
            stream.write(text + '\n')
    return 0 if report['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
