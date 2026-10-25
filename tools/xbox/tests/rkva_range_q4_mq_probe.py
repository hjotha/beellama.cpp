#!/usr/bin/env python3
"""Gated live probe for the RKVA Q4_0 multi-query (MTP verify) range kernels.

Offline by default: without --execute it prints the plan and opens no socket.
--self-test runs the offline oracle/filter checks only (no socket either).
With --execute it talks to an already running dedicated RKVA listener (no model,
no credentials). The console must be leased to the caller; this script never
deploys, restarts, or changes anything outside its own RKVA session.

Kernel selection under the legacy q4_kernel_mode field (op 20), Q4_0 only:
  mode 0  reference (default)                       every nq
  mode 1  nq1 per-query q4_gqa_balanced; nq2..9 q4_mq4_balanced (4-query tile)
  mode 2  nq1 per-query q4_gqa_fine;     nq2..9 q4_mq2_fine     (2-query tile)
  mode 3  per-query q4_gqa_coarse for every nq (in-build A/B baseline)
A mode-1/2 call whose tile PSO is missing runs the per-query kernel instead;
the report's q4_multi_query object and bucket names show what actually ran.

Parity: per checkpoint (live overflow rows) and query set (contiguous MTP
verify tails nq 2/3/5/9 plus a masked/unsorted nq5 set) every mode is compared
per query against a float64 oracle on the exact dequantized Q4_0 rows (signed
fp16 block scales) and against mode 0. Decode: per mode and nq, append nq rows
then attend nq contiguous causal queries ending at the newest row (one layer of
an MTP verify step), with profiling enabled+reset, then fetch report_only.
Shape: H24/HKV4/D256, origin 72704, capacity 58368 (C131072). One-layer kernel
metrics only: they are not model tokens/s.
"""
import argparse
import json
import struct
import sys

sys.path.insert(0, __import__('os').path.dirname(__file__))

RKVA_RANGE_PROFILE = 20
PROFILE_ENABLE, PROFILE_RESET, PROFILE_REPORT_ONLY = 1, 2, 8
MODES = (0, 1, 2, 3)
H, HKV, D = 24, 4, 256
ORIGIN = 72704
CAPACITY = 58368
ROW_BYTES = HKV * (D // 32) * 18  # one plane (K or V) of one row, Q4_0
NQ_SETS = (2, 3, 5, 9)
# Expected bucket kernel names (rkva-range-profile.h bucket_key) for nq 2..9.
MQ_KERNEL = {0: 'reference', 1: 'optimized-q4_mq4_balanced', 2: 'optimized-q4_mq2_fine',
             3: 'optimized-q4_gqa_coarse'}
NQ1_KERNEL = {0: 'reference', 1: 'optimized-q4_gqa_balanced', 2: 'optimized-q4_gqa_fine',
              3: 'optimized-q4_gqa_coarse'}


def profile(client, flags, mode):
    body = client.call(RKVA_RANGE_PROFILE, struct.pack('<IIQ', flags, mode, 0))
    return json.loads(bytes(body).decode('utf-8'))


def make_rows(rng, rows, np):
    """Q4_0 planes with signed, varied fp16 scales; values kept as exact float32."""
    planes = []
    for _ in range(2):
        blocks = rows * HKV * (D // 32)
        magnitude = rng.uniform(0.004, 0.08, blocks)
        sign = np.where(rng.random(blocks) < 0.5, -1.0, 1.0)
        scale = (magnitude * sign).astype(np.float16)
        quant = rng.integers(0, 256, (blocks, 16), dtype=np.uint8)
        raw = np.zeros((blocks, 18), np.uint8)
        raw[:, :2] = scale.view(np.uint8).reshape(blocks, 2)
        raw[:, 2:] = quant
        low = (quant & 15).astype(np.float32) - 8
        high = (quant >> 4).astype(np.float32) - 8
        values = np.concatenate([low, high], axis=1) * scale.astype(np.float32)[:, None]
        planes.append((raw.tobytes(), values.reshape(rows, HKV, D)))
    return planes


def dequant_row_independent(np, raw, row):
    """Second, byte-level Q4_0 decoder (struct-based) used by --self-test."""
    out = np.zeros((HKV, D))
    base = row * ROW_BYTES
    for kv in range(HKV):
        for block in range(D // 32):
            at = base + (kv * (D // 32) + block) * 18
            (scale,) = struct.unpack_from('<e', raw, at)
            for i in range(16):
                byte = raw[at + 2 + i]
                out[kv, block * 32 + i] = ((byte & 15) - 8) * scale
                out[kv, block * 32 + 16 + i] = ((byte >> 4) - 8) * scale
    return out


def oracle(np, q, keys, values, positions, live, scale):
    """Independent per-query float64 attention: rows [0, min(live, pos-origin+1))."""
    nq = len(positions)
    out = np.zeros((nq, H, D))
    lse = np.full((nq, H), -np.inf)
    for qi, position in enumerate(positions):
        limit = max(0, min(live, position - ORIGIN + 1))
        if limit <= 0:
            continue
        for head in range(H):
            kv = head // (H // HKV)
            k = keys[:limit, kv, :].astype(np.float64)
            v = values[:limit, kv, :].astype(np.float64)
            scores = k @ q[qi, head] * scale
            peak = scores.max()
            weights = np.exp(scores - peak)
            total = weights.sum()
            out[qi, head] = weights @ v / total
            lse[qi, head] = peak + np.log(total)
    return out, lse


def compare(np, got_out, got_lse, want_out, want_lse):
    """Per-query verdicts at the qualified probe tolerance (3e-4 + 3e-4|x|, LSE 3e-5 rel)."""
    per_query = []
    for qi in range(want_out.shape[0]):
        finite = np.isfinite(want_lse[qi])
        if not np.array_equal(np.isfinite(got_lse[qi]), finite):
            per_query.append(dict(ok=False, reason='LSE finiteness mismatch'))
            continue
        if not np.all(np.isfinite(got_out[qi])):
            per_query.append(dict(ok=False, reason='non-finite OUT'))
            continue
        d_out = np.abs(got_out[qi] - want_out[qi])
        d_lse = np.abs(got_lse[qi][finite] - want_lse[qi][finite])
        ok = bool(np.all(d_out <= 3e-4 + 3e-4 * np.abs(want_out[qi])) and
                  np.all(d_lse <= 3e-4 + 3e-5 * np.abs(want_lse[qi][finite])))
        per_query.append(dict(ok=ok, out_max_abs=float(d_out.max()),
                              lse_max_abs=float(d_lse.max()) if d_lse.size else 0.))
    return dict(ok=all(e['ok'] for e in per_query), per_query=per_query)


def warm_buckets(report, mode, nq):
    """Warm attention buckets of the kernel that should have run (actual names)."""
    kernel = (NQ1_KERNEL if nq == 1 else MQ_KERNEL)[mode]
    band = 'nq1' if nq == 1 else 'nq2-16'
    prefix = 'attention/q4_0/%s/%s/' % (kernel, band)
    engine = report.get('engine', report)
    return {key: value for key, value in engine.get('buckets', {}).items()
            if key.startswith(prefix) and key.endswith('/warm')}


def query_sets(live):
    sets = []
    for nq in NQ_SETS:
        sets.append(('tail_nq%d' % nq, [ORIGIN + live - nq + i for i in range(nq)]))
    # Masked/unsorted: before origin, mid, newest, future (clamped to live), newest-1.
    sets.append(('mixed_nq5', [ORIGIN - 1, ORIGIN + live // 2, ORIGIN + live - 1,
                               ORIGIN + live + 5, ORIGIN + live - 2]))
    return sets


def self_test(args):
    import numpy as np
    rng = np.random.default_rng(args.seed)
    rows = 70
    (k_raw, k_val), (v_raw, v_val) = make_rows(rng, rows, np)
    for row in (0, 1, 63, 64, 69):
        assert np.array_equal(dequant_row_independent(np, k_raw, row), k_val[row].astype(np.float64))
        assert np.array_equal(dequant_row_independent(np, v_raw, row), v_val[row].astype(np.float64))
    scale = 1 / 16.
    positions = [ORIGIN - 1, ORIGIN, ORIGIN + 40, ORIGIN + rows - 1, ORIGIN + rows + 9]
    q = rng.standard_normal((len(positions), H, D))
    q[4] = q[3]
    out, lse = oracle(np, q, k_val, v_val, positions, rows, scale)
    assert not np.isfinite(lse[0]).any() and not out[0].any()           # before origin
    assert np.allclose(out[1], np.repeat(v_val[0], H // HKV, axis=0))   # single row
    assert np.array_equal(out[3], out[4]) and np.array_equal(lse[3], lse[4])  # future clamps
    assert compare(np, out, lse, out, lse)['ok']
    bad = out.copy()
    bad[2, 5, 7] += 1e-2
    verdict = compare(np, bad, lse, out, lse)
    assert not verdict['ok'] and verdict['per_query'][2]['ok'] is False
    assert verdict['per_query'][1]['ok'] and verdict['per_query'][3]['ok']
    lse_bad = lse.copy()
    lse_bad[0, 0] = 0.
    assert not compare(np, out, lse_bad, out, lse)['ok']
    fake = dict(engine=dict(buckets={
        'attention/q4_0/optimized-q4_mq4_balanced/nq2-16/r32k+/warm': 1,
        'attention/q4_0/optimized-q4_mq4_balanced/nq2-16/r32k+/cold': 2,
        'attention/q4_0/optimized-q4_gqa_balanced/nq1/r32k+/warm': 3,
        'attention/q4_0/optimized-q4_gqa_coarse/nq2-16/r32k+/warm': 4}))
    assert list(warm_buckets(fake, 1, 3).values()) == [1]
    assert list(warm_buckets(fake, 1, 1).values()) == [3]
    assert list(warm_buckets(fake, 3, 5).values()) == [4]
    assert warm_buckets(fake, 2, 3) == {}
    for name, positions in query_sets(CAPACITY):
        assert all(p < ORIGIN + CAPACITY + 6 for p in positions), name
    print(json.dumps(dict(self_test='ok')))
    return 0


def execute(args):
    import numpy as np
    from rkva_range_probe import Client
    rng = np.random.default_rng(args.seed)
    capacity = args.capacity
    steps_rows = max(args.decode_nq) * args.decode_steps
    checkpoints = sorted({c for c in args.checkpoints if 0 <= c <= capacity})
    (k_raw, k_val), (v_raw, v_val) = make_rows(rng, capacity, np)
    scale = 1 / 16.
    client = Client(args.host, args.port)
    report = dict(success=False, host=args.host, port=args.port, origin=ORIGIN, capacity=capacity,
                  checkpoints=checkpoints, decode_steps=args.decode_steps,
                  decode_nq=args.decode_nq, metric_scope='one-layer kernel, not model TPS',
                  parity=[], decode=[])
    try:
        client.hello(capacity, layers=1)
        if not client.capabilities & (1 << 5):
            raise RuntimeError('listener lacks RKVA_CAP_RANGE_PROFILE')
        client.call(2, struct.pack('<4I', 0, 0, capacity, 1))
        client.configure(capacity, origin=ORIGIN, q4=True)
        live = 0

        def upload_to(target):
            nonlocal live
            while live < target:
                rows = min(512, target - live)
                client.upload_packed(k_raw[live*ROW_BYTES:(live+rows)*ROW_BYTES],
                                     v_raw[live*ROW_BYTES:(live+rows)*ROW_BYTES], start=live)
                live += rows

        def attend(q, positions):
            out, lse = client.attention(q.astype(np.float32).ravel().tolist(), positions)
            nq = len(positions)
            return (np.array(out, np.float64).reshape(nq, H, D),
                    np.array(lse, np.float64).reshape(nq, H))

        report['initial_profile'] = profile(client, PROFILE_RESET, 0)
        for checkpoint in checkpoints:
            upload_to(checkpoint)
            for name, positions in query_sets(live):
                q = rng.standard_normal((len(positions), H, D))
                want = oracle(np, q, k_val, v_val, positions, live, scale)
                baseline = None
                for mode in MODES:
                    profile(client, PROFILE_ENABLE | PROFILE_RESET, mode)
                    got = attend(q, positions)
                    used = profile(client, PROFILE_REPORT_ONLY, 0)
                    keys = sorted(k for k in used.get('engine', used).get('buckets', {})
                                  if k.startswith('attention/'))
                    entry = dict(rows=live, set=name, positions=positions, mode=mode,
                                 kernel_buckets=keys, vs_oracle=compare(np, *got, *want))
                    if baseline is None:
                        baseline = got
                    else:
                        entry['vs_mode0'] = compare(np, *got, *baseline)
                    report['parity'].append(entry)
        base_live = capacity - steps_rows
        for nq in args.decode_nq:
            for mode in MODES:
                # Uploads are append-only (start_row 0 replaces). Rebuild the same
                # base window so every mode appends the same rows.
                if live != base_live:
                    live = 0
                    upload_to(base_live)
                profile(client, PROFILE_ENABLE | PROFILE_RESET, mode)
                for step in range(args.warmup + args.decode_steps):
                    if step >= args.warmup:
                        upload_to(min(capacity, live + nq))
                    q = rng.standard_normal((nq, H, D))
                    attend(q, [ORIGIN + live - nq + i for i in range(nq)])
                full = profile(client, PROFILE_REPORT_ONLY, 0)
                report['decode'].append(dict(mode=mode, nq=nq, warm=warm_buckets(full, mode, nq),
                                             q4_multi_query=full.get('engine', full)
                                             .get('q4_multi_query'), report=full))
        client.call(3, session=0)
        report['success'] = all(e['vs_oracle']['ok'] and e.get('vs_mode0', {'ok': True})['ok']
                                for e in report['parity'])
    finally:
        try:
            profile(client, PROFILE_RESET, 0)  # leave reference, profiling off
        except Exception:  # best effort
            pass
        client.socket.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--execute', action='store_true',
                        help='actually connect; requires the caller to hold the console lease')
    parser.add_argument('--self-test', action='store_true', help='offline oracle/filter checks')
    parser.add_argument('--host')
    parser.add_argument('--port', type=int, default=50054)
    parser.add_argument('--capacity', type=int, default=CAPACITY)
    parser.add_argument('--checkpoints', type=int, nargs='*',
                        default=[1, 2, 63, 64, 65, 129, 4099, 58128, CAPACITY])
    parser.add_argument('--decode-nq', type=int, nargs='*', default=[1, 3, 5])
    parser.add_argument('--decode-steps', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--seed', type=int, default=266)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.self_test:
        return self_test(args)
    if any(nq < 1 or nq > 9 for nq in args.decode_nq):
        parser.error('--decode-nq values must be 1..9')
    if not args.execute:
        plan = dict(execute=False, note='offline plan only; pass --execute --host H to run',
                    modes=list(MODES), mq_kernel=MQ_KERNEL, nq1_kernel=NQ1_KERNEL,
                    origin=ORIGIN, capacity=args.capacity, checkpoints=args.checkpoints,
                    query_sets=[name for name, _ in query_sets(args.capacity)],
                    decode_nq=args.decode_nq, decode_steps=args.decode_steps,
                    warmup=args.warmup, metric_scope='one-layer kernel, not model TPS',
                    ops=['hello', 'session(2)', 'range_config(12,q4_0)', 'range_upload(13,wire2)',
                         'range_attention(14,nq1..9)', 'range_profile(20)', 'close(3)'])
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
