#!/usr/bin/env python3
"""Gated live A/B probe for the RKVA range kernels of every storage (future 263).

Offline by default: without --execute it prints the plan and opens no socket.
With --execute it talks to an already running dedicated RKVA listener (no model,
no credentials). The console must be leased to the caller; this script never
deploys, restarts, or changes anything outside its own RKVA session.

Formats: f16, q8_0, q4_0 and native Bee kvarn4 (descriptor v1, genuine Bee
records from --bee-fixtures mixed with synthetic signed records, rotated F16
stage, optional original-domain F16 precision tail). Per format and checkpoint
every mode 0..3 (op 20 field q4_kernel_mode: legacy name, storage-qualified
optimized kernel per the engine report's optimized_kernels) runs the same query-indexed verification batches (nq1..9, nq10 fallback); results are compared with a float64 oracle on the exact stored data
(KVarN4 body rows mapped to the original domain with the explicit H256/16
matrix) and with mode 0. Then a one-layer decode loop (append one row +
attend) runs per mode with profiling enabled+reset and the engine report is
fetched with report_only. Shape: H24/HKV4/D256, origin 102400.

The warm decode summary uses the real bucket names
attention/<storage>/{reference|optimized-<kernel>}/nq1/<rows>/warm, e.g.
optimized-q4_gqa_balanced (the 261 Q4 probe filtered on 'optimized-<mode>' and
missed them). --summarize RAW.json recomputes that summary offline from a raw
report of this probe or of tests/rkva_range_q4_kernel_probe.py without
modifying the raw file.

These are synthetic one-layer kernel timings, not model tokens/s.
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RKVA_RANGE_PROFILE = 20
PROFILE_ENABLE, PROFILE_RESET, PROFILE_REPORT_ONLY = 1, 2, 8
MODES = (0, 1, 2, 3)
STORAGE = {'f16': 0, 'q4_0': 1, 'q8_0': 2, 'kvarn4': 3}
CAPABILITY = {'f16': 0, 'q4_0': 1 << 0, 'q8_0': 1 << 1, 'kvarn4': 1 << 2}
MODE_SUFFIX = {1: 'balanced', 2: 'fine', 3: 'coarse'}
KERNEL_PREFIX = {'f16': 'f16', 'q4_0': 'q4', 'q8_0': 'q8', 'kvarn4': 'kvarn4'}
H, HKV, D = 24, 4, 256
ORIGIN = 102400
RECORD = 8960
PHYS = HKV * 2
DEFAULT_BEE = Path(__file__).resolve().parents[3] / "build" / "xllama-kvarn4-range260-fixtures"


def kernel_name(fmt, mode):
    """Mirror of rkva_rprof::kernel_name (uwp/rkva/rkva-range-profile.h)."""
    return 'reference' if mode == 0 else '%s_gqa_%s' % (KERNEL_PREFIX[fmt], MODE_SUFFIX[mode])


def warm_buckets(engine, fmt, mode, optimized_expected=True):
    kernel = 'reference' if mode == 0 or not optimized_expected else \
        'optimized-' + kernel_name(fmt, mode)
    prefix = 'attention/%s/%s/nq1/' % (fmt, kernel)
    return {key: value for key, value in engine.get('buckets', {}).items()
            if key.startswith(prefix) and key.endswith('/warm')}


def profile(client, flags, mode):
    body = client.call(RKVA_RANGE_PROFILE, struct.pack('<IIQ', flags, mode, 0))
    return json.loads(bytes(body).decode('utf-8'))


def engine_of(report):
    return report.get('engine', report)


# ---------------------------------------------------------------- formats ---


class Blocked:
    """q4_0 (wire 2) / q8_0 (wire 3) rows, ggml block layout, signed scales."""

    def __init__(self, np, rng, capacity, q8):
        self.np, self.q8, self.capacity = np, q8, capacity
        self.wire = 3 if q8 else 2
        self.block = 34 if q8 else 18
        self.row_bytes = HKV * (D // 32) * self.block
        self.planes = [self._plane(rng) for _ in range(2)]
        self.live = 0

    def _plane(self, rng):
        np = self.np
        blocks = self.capacity * HKV * (D // 32)
        scale = (rng.uniform(0.004, 0.06, blocks) * rng.choice([-1, 1], blocks)).astype(np.float16)
        raw = np.zeros((blocks, self.block), np.uint8)
        raw[:, :2] = scale.view(np.uint8).reshape(blocks, 2)
        if self.q8:
            quant = rng.integers(-128, 128, (blocks, 32), dtype=np.int16).astype(np.int8)
            raw[:, 2:] = quant.view(np.uint8)
            values = quant.astype(np.float64)
        else:
            quant = rng.integers(0, 256, (blocks, 16), dtype=np.uint8)
            raw[:, 2:] = quant
            values = np.concatenate([(quant & 15).astype(np.float64) - 8,
                                     (quant >> 4).astype(np.float64) - 8], axis=1)
        values = values * scale.astype(np.float64)[:, None]
        return raw.tobytes(), values.reshape(self.capacity, HKV, D)

    def configure(self, client):
        client.configure(self.capacity, q4=not self.q8, q8=self.q8)
        self.live = 0

    def upload_to(self, client, target):
        rb = self.row_bytes
        while self.live < target:
            rows = min(512, target - self.live)
            a, b = self.live * rb, (self.live + rows) * rb
            client.upload_packed(self.planes[0][0][a:b], self.planes[1][0][a:b],
                                 start=self.live, wire=self.wire)
            self.live += rows

    def rebuild(self, client, target):
        self.live = 0
        self.upload_to(client, target)

    def kv(self, limit):
        return self.planes[0][1][:limit], self.planes[1][1][:limit]


class F16:
    def __init__(self, np, rng, capacity):
        self.np, self.capacity = np, capacity
        self.planes = [rng.uniform(-2, 2, (capacity, HKV, D)).astype('<f2') for _ in range(2)]
        self.live = 0

    def configure(self, client):
        client.configure(self.capacity)
        self.live = 0

    def upload_to(self, client, target):
        while self.live < target:
            rows = min(512, target - self.live)
            sl = slice(self.live, self.live + rows)
            payload = struct.pack('<4I', self.live, rows, 0, 0)
            payload += self.planes[0][sl].tobytes() + self.planes[1][sl].tobytes()
            client.call(13, payload, session=0)
            self.live += rows

    def rebuild(self, client, target):
        self.live = 0
        self.upload_to(client, target)

    def kv(self, limit):
        return tuple(plane[:limit].astype(self.np.float64) for plane in self.planes)


class Kvarn4:
    """Native descriptor-v1 KVarN4 session fed by op 15/16 (records, stage, tail)."""

    def __init__(self, np, rng, capacity, ptt, bee_dir):
        import rkva_range_kvarn4_opt_model as km
        self.np, self.km, self.capacity, self.ptt = np, km, capacity, ptt
        self.groups = capacity // 128
        self.bee = None
        if bee_dir and (bee_dir / 'bee-kvarn4-records.bin').exists():
            self.bee = km.check_bee(bee_dir)
        # records[kind][group] = bytes of PHYS records; genuine Bee records on every 3rd head.
        self.records = [[b''.join(self.bee[kind] if (self.bee and ph % 3 == 0)
                                  else km.random_record(rng) for ph in range(PHYS))
                         for _ in range(self.groups)] for kind in range(2)]
        self.rot = [rng.normal(0, 0.5, (capacity, HKV, D)).astype('<f2') for _ in range(2)]
        self.orig = [rng.normal(0, 0.5, (capacity, HKV, D)).astype('<f2') for _ in range(2)]
        self.decoded = {}
        self.live = 0
        self.groups_uploaded = 0

    def descriptor(self):
        return [ORIGIN, self.capacity, 4, 1, 128, 128, 256, 2, 4, 4, RECORD, 3, 2, 128, self.ptt,
                1, 1, 1]

    def configure(self, client):
        client.call(15, struct.pack('<18I', *self.descriptor()), session=0)
        self.live = 0
        self.groups_uploaded = 0

    def _stage_rows(self, rows, kind):
        return self.rot[kind][rows].tobytes() if len(rows) else b''

    def _tail_rows(self, rows, kind):
        return self.orig[kind][rows].tobytes() if len(rows) and self.ptt else b''

    def _upload(self, client, live, reset):
        np = self.np
        previous = 0 if reset else self.live
        first = 0 if reset else self.groups_uploaded
        groups = live // 128 - first
        header = struct.pack('<8I', live, first, groups, int(reset), 0 if reset else previous,
                             0 if reset else live - previous, 0, 0)
        records = b''.join(b''.join(self.records[kind][first:first + groups]) for kind in range(2))
        if reset:
            stage = []
            for kind in range(2):
                image = np.zeros((384, HKV, D), '<f2')
                for token in range(live):
                    gg = (ORIGIN + token) // 128
                    slot = 0 if gg == 0 else 1 + (gg - 1) % 2
                    image[slot * 128 + token % 128] = self.rot[kind][token]
                stage.append(image.tobytes())
            tail_rows = np.arange(max(0, live - 128), live)
        else:
            rows = np.arange(previous, live)
            stage = [self._stage_rows(rows, 0), self._stage_rows(rows, 1)]
            tail_rows = rows
        tail = [self._tail_rows(tail_rows, 0), self._tail_rows(tail_rows, 1)]
        client.call(16, header + records + stage[0] + stage[1] + tail[0] + tail[1], session=0)
        self.live = live
        self.groups_uploaded = live // 128

    def upload_to(self, client, target):
        if self.live == 0 and target:
            self._upload(client, min(target, 128), reset=True)
        while self.live < target:
            # Append chunks never cross a 128-row group (one stage slot per copy).
            step = min(target - self.live, 128 - self.live % 128)
            self._upload(client, self.live + step, reset=False)

    def rebuild(self, client, target):
        self.live = 0
        self.groups_uploaded = 0
        self.upload_to(client, target)

    def _record(self, kind, group, ph):
        key = (kind, group, ph)
        if key not in self.decoded:
            blob = self.records[kind][group][ph * RECORD:(ph + 1) * RECORD]
            self.decoded[key] = self.km.decode_record(blob, value=kind == 1)
        return self.decoded[key]

    def kv(self, limit):
        """Original-domain float64 K/V of rows < limit for the current live count.

        Independent of the engine: completed overflow groups decode the uploaded
        records, the live (incomplete) group reads the rotated F16 stage rows, and
        both are mapped back with the explicit H256/16 matrix; precision-tail rows
        are the uploaded original-domain F16 rows.
        """
        np, km = self.np, self.km
        live = self.live
        tail_begin = live - min(live, self.ptt)
        out = []
        for kind in range(2):
            x = self.rot[kind][:limit].astype(np.float64)
            for group in range(min(live // 128, (limit + 127) // 128)):
                if (ORIGIN // 128) + group == 0:
                    continue  # global group 0 stays staged (never for origin 102400)
                rows = slice(group * 128, min(limit, group * 128 + 128))
                count = rows.stop - rows.start
                for kv in range(HKV):
                    for s in range(2):
                        x[rows, kv, s * 128:(s + 1) * 128] = \
                            self._record(kind, group, kv * 2 + s)[:count]
            x = x @ km.H256
            if tail_begin < limit:
                x[tail_begin:] = self.orig[kind][tail_begin:limit].astype(np.float64)
            out.append(x)
        return out[0], out[1]


# ---------------------------------------------------------------- oracle ---


def oracle(np, q, keys, values, limit, scale):
    out = np.zeros((H, D))
    lse = np.full(H, -np.inf)
    if limit <= 0:
        return out, lse
    for head in range(H):
        kv = head // (H // HKV)
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
    if not np.all(np.isfinite(got_out)):
        return dict(ok=False, reason='non-finite output')
    out_err = float(np.max(np.abs(got_out - want_out))) if got_out.size else 0.
    lse_err = float(np.max(np.abs(got_lse[finite] - want_lse[finite]))) if finite.any() else 0.
    ok = bool(np.all(np.abs(got_out - want_out) <= 3e-4 + 3e-4 * np.abs(want_out)) and
              np.all(np.abs(got_lse[finite] - want_lse[finite]) <=
                     3e-4 + 3e-5 * np.abs(want_lse[finite])))
    return dict(ok=ok, out_max_abs=out_err, lse_max_abs=lse_err)


# --------------------------------------------------------------- execute ---


def make_format(np, rng, fmt, args):
    if fmt == 'f16':
        return F16(np, rng, args.capacity)
    if fmt in ('q4_0', 'q8_0'):
        return Blocked(np, rng, args.capacity, q8=fmt == 'q8_0')
    return Kvarn4(np, rng, args.capacity, args.precision_tail, args.bee_fixtures)


def run_format(np, client, fmt, data, rng, args, report):
    scale = 1 / 16.
    client.call(2, struct.pack('<4I', 0, 0, args.capacity, 1))
    data.configure(client)
    available = profile(client, PROFILE_REPORT_ONLY, 0)
    kernels = engine_of(available).get('optimized_kernels', {})
    opt_ok = bool(kernels.get(fmt, {}).get('available'))
    entry = dict(format=fmt, optimized_available=opt_ok,
                 optimized_status=kernels.get(fmt, {}).get('status'), parity=[], decode=[])
    profile(client, PROFILE_RESET, 0)
    checkpoints = sorted({c for c in args.checkpoints if 0 < c <= args.capacity - args.decode_steps})
    for checkpoint in checkpoints:
        data.upload_to(client, checkpoint)
        # Distinct positions in one batch exercise query-specific causal masks,
        # including a fully masked query and future poisoned/random live rows.
        # nq10 deliberately exceeds the bounded optimized tier (N=8 => nq9).
        for nq in args.queries:
            positions = [ORIGIN + checkpoint - 1 - i for i in range(nq)]
            if nq > 1:
                positions[-1] = ORIGIN - 1
            q = rng.standard_normal((nq, H, D)).astype(np.float32)
            wants, limits = [], []
            for qi, position in enumerate(positions):
                limit = max(0, min(data.live, position - ORIGIN + 1))
                keys, values = data.kv(limit)
                wants.append(oracle(np, q[qi].astype(np.float64), keys, values, limit, scale))
                limits.append(limit)
            want = (np.stack([w[0] for w in wants]), np.stack([w[1] for w in wants]))
            baseline = None
            for mode in MODES:
                profile(client, PROFILE_ENABLE | PROFILE_RESET, mode)
                out, lse = client.attention(q.ravel().tolist(), positions)
                got = (np.array(out, np.float64).reshape(nq, H, D),
                       np.array(lse, np.float64).reshape(nq, H))
                item = dict(rows=checkpoint, nq=nq, positions=positions,
                            causal_rows=limits, mode=mode,
                            vs_oracle=compare(np, *got, *want))
                engine = engine_of(profile(client, PROFILE_REPORT_ONLY, 0))
                counts = engine.get('optimized_kernels', {}).get(fmt, {})
                item['optimized_calls'] = counts.get('optimized_calls')
                item['reference_calls'] = counts.get('reference_calls')
                # A legacy report without fast_nq_max promises only nq1.
                fast_nq_max = int(counts.get('fast_nq_max', 1 if opt_ok else 0))
                item['fast_nq_max'] = fast_nq_max
                want_opt = 1 if (mode and opt_ok and nq <= fast_nq_max) else 0
                item['kernel_selection_ok'] = counts.get('optimized_calls') == want_opt and \
                    counts.get('reference_calls') == 1 - want_opt
                if baseline is None:
                    baseline = got
                else:
                    item['vs_mode0'] = compare(np, *got, *baseline)
                entry['parity'].append(item)
    data.upload_to(client, args.capacity - args.decode_steps)
    base_live = data.live
    for mode in MODES:
        if data.live != base_live:
            data.rebuild(client, base_live)
        profile(client, PROFILE_ENABLE | PROFILE_RESET, mode)
        for step in range(args.warmup + args.decode_steps):
            if data.live < args.capacity and step >= args.warmup:
                data.upload_to(client, data.live + 1)
            q = rng.standard_normal(H * D).astype(np.float32).tolist()
            client.attention(q, [ORIGIN + data.live - 1])
        full = profile(client, PROFILE_REPORT_ONLY, 0)
        entry['decode'].append(dict(mode=mode, report=full,
                                    warm_nq1=warm_buckets(engine_of(full), fmt, mode, opt_ok)))
        profile(client, PROFILE_RESET, 0)
    client.call(3, session=0)
    entry['success'] = all(p['vs_oracle']['ok'] and p.get('vs_mode0', {'ok': True})['ok'] and
                           p['kernel_selection_ok'] for p in entry['parity'])
    report['formats'].append(entry)


def execute(args):
    import numpy as np
    from rkva_range_probe import Client
    rng = np.random.default_rng(args.seed)
    report = dict(success=False, host=args.host, port=args.port, capacity=args.capacity,
                  checkpoints=args.checkpoints, queries=args.queries, decode_steps=args.decode_steps,
                  precision_tail=args.precision_tail, formats=[],
                  note='synthetic one-layer kernel timing, not model TPS')
    client = Client(args.host, args.port)
    try:
        client.hello(args.capacity, layers=1)
        if not client.capabilities & (1 << 5):
            raise RuntimeError('listener lacks RKVA_CAP_RANGE_PROFILE')
        for fmt in args.formats:
            if CAPABILITY[fmt] and not client.capabilities & CAPABILITY[fmt]:
                report['formats'].append(dict(format=fmt, success=False,
                                              reason='listener lacks the storage capability'))
                continue
            run_format(np, client, fmt, make_format(np, rng, fmt, args), rng, args, report)
        report['success'] = bool(report['formats']) and all(f['success'] for f in report['formats'])
    finally:
        try:
            profile(client, 0, 0)  # leave the listener in reference, profiling off
        except Exception:
            pass
        client.socket.close()
    report['summary'] = summarize(report)
    return report


# ------------------------------------------------------------- self-test ---


class FakeKvarnServer:
    """Offline transliteration of rkva_kvarn_range.h valid_upload + stage/tail
    chunking and RemoteAttnEngine::upload_kvarn_range copy placement. Replays
    the probe's real op 15/16 payloads into a NaN-poisoned plane image."""

    def __init__(self, km, capacity, ptt):
        self.km, self.lay = km, km.Layout(ORIGIN, capacity, ptt)
        self.planes = [bytearray(b'\xff' * self.lay.half) for _ in range(2)]
        self.live = self.groups = 0
        self.ready = False
        self.calls = 0

    def call(self, op, payload=b'', session=None, **_):
        self.calls += 1
        if op == 15:
            assert list(struct.unpack('<18I', payload))[13:15] == [128, self.lay.ptt]
            self.ready, self.live, self.groups = False, 0, 0
            return b''
        assert op == 16, op
        live, begin, groups, flags, append_begin, append_rows, r0, r1 = \
            struct.unpack_from('<8I', payload)
        reset = flags == 1
        assert 0 < live <= self.lay.capacity and flags in (0, 1) and not (r0 or r1)
        if reset:
            assert begin == 0 and append_begin == 0 and append_rows == 0
        else:
            assert self.ready and live >= self.live and begin <= self.groups
            assert append_begin == self.live and append_rows == live - self.live
        assert begin + groups <= live // 128
        covered = groups if reset else max(self.groups, begin + groups)
        assert covered >= live // 128
        row_bytes = HKV * 256 * 2
        rec = groups * PHYS * RECORD
        stage_rows = 384 if reset else append_rows
        tail_rows = (min(live, 128) if reset else append_rows) if self.lay.ptt else 0
        body = payload[32:]
        assert len(body) == 2 * (rec + (stage_rows + tail_rows) * row_bytes), 'payload size'
        for kind in range(2):
            plane = self.planes[kind]
            off = begin * PHYS * RECORD
            plane[off:off + rec] = body[kind * rec:(kind + 1) * rec]
            src = 2 * rec + kind * stage_rows * row_bytes
            dst = self.lay.records_bytes
            if reset:
                plane[dst:dst + 384 * row_bytes] = body[src:src + 384 * row_bytes]
            else:
                for row in range(append_rows):
                    token = ORIGIN + append_begin + row
                    gg = token // 128
                    slot = 0 if gg == 0 else 1 + (gg - 1) % 2
                    at = dst + (slot * 128 + token % 128) * row_bytes
                    plane[at:at + row_bytes] = body[src + row * row_bytes:src + (row + 1) * row_bytes]
            if tail_rows:
                src = 2 * rec + 2 * stage_rows * row_bytes + kind * tail_rows * row_bytes
                first = (0 if reset else append_begin) + (append_rows if not reset else live) - tail_rows
                for i in range(tail_rows):
                    token = ORIGIN + first + i
                    at = dst + 384 * row_bytes + (token % 128) * row_bytes
                    plane[at:at + row_bytes] = body[src + i * row_bytes:src + (i + 1) * row_bytes]
        self.live, self.ready = live, True
        self.groups = covered
        return b''

    def original_domain(self, limit):
        km = self.km
        out = []
        for kind in range(2):
            src, exact = km.oracle_sources(self.lay, bytes(self.planes[kind]), self.live, kind == 1)
            x = src[:limit].astype('float64')
            body = ~exact[:limit]
            x[body] = x[body] @ km.H256
            out.append(x)
        return out


def self_test(args):
    """Offline: probe KVarN4 uploads + oracle vs the independent model oracle."""
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rkva_range_kvarn4_opt_model as km
    rng = np.random.default_rng(args.seed)
    results = []
    for ptt in (128, 0):
        capacity = 1152
        data = Kvarn4(np, rng, capacity, ptt, args.bee_fixtures)
        fake = FakeKvarnServer(km, capacity, ptt)
        data.configure(fake)
        steps = [1, 64, 127, 128, 129, 255, 256, 257, 300, 513, 900, 1000]
        for target in steps + ['rebuild-640', 641, 642, 769, 1100]:
            if isinstance(target, str):
                data.rebuild(fake, int(target.split('-')[1]))
            else:
                data.upload_to(fake, target)
            assert fake.live == data.live
            for limit in sorted({1, data.live // 2 or 1, data.live}):
                mine = data.kv(limit)
                ref = fake.original_domain(limit)
                err = max(float(np.max(np.abs(mine[i] - ref[i]))) for i in range(2))
                assert err < 1e-9, (ptt, target, limit, err)
                results.append(dict(ptt=ptt, live=data.live, limit=limit, max_abs=err))
    for fmt in ('q4_0', 'q8_0'):
        data = Blocked(np, rng, 64, q8=fmt == 'q8_0')
        raw, val = data.planes[0]
        block = data.block
        for b in range(0, 64 * HKV * 8, 37):
            blob = raw[b * block:(b + 1) * block]
            sc = float(np.frombuffer(blob[:2], '<f2')[0])
            if fmt == 'q8_0':
                q = np.frombuffer(blob[2:], np.int8).astype(np.float64)
            else:
                qs = np.frombuffer(blob[2:], np.uint8)
                q = np.concatenate([(qs & 15) - 8., (qs >> 4) - 8.])
            assert np.array_equal(q * sc, val.reshape(-1, 32)[b]), (fmt, b)
    print(json.dumps(dict(self_test='ok', kvarn4_checks=len(results),
                          worst_abs=max(r['max_abs'] for r in results),
                          fake_server_calls=fake.calls, blocked_formats=['q4_0', 'q8_0']),
                     indent=2))
    return 0


# ------------------------------------------------------------- summarize ---


def _p50(stats, name):
    value = stats.get(name, {})
    return value.get('p50') if isinstance(value, dict) else None


def summarize(raw):
    """Warm nq1 decode summary from a raw report (this probe or the 261 Q4 probe)."""
    rows = []
    entries = raw.get('formats')
    if entries is None:  # tests/rkva_range_q4_kernel_probe.py layout
        entries = [dict(format='q4_0', decode=raw.get('decode', []))]
    for entry in entries:
        fmt = entry.get('format')
        for run in entry.get('decode', []):
            engine = engine_of(run.get('report', {}))
            mode = run.get('mode')
            available = engine.get('optimized_kernels', {}).get(fmt, {}).get('available')
            if available is None:  # 261/262 reports: only the Q4 field exists
                available = engine.get('q4_optimized_available', True) if fmt == 'q4_0' else False
            for key, bucket in sorted(warm_buckets(engine, fmt, mode, bool(available)).items()):
                gpu, cpu = bucket.get('gpu_us', {}), bucket.get('cpu_us', {})
                rows.append(dict(format=fmt, mode=mode, bucket=key,
                                 samples=bucket.get('samples'),
                                 split_k_p50=_p50(bucket, 'split_k'),
                                 gpu_pass1_us_p50=_p50(gpu, 'pass1'),
                                 gpu_reduce_us_p50=_p50(gpu, 'reduce'),
                                 gpu_total_us_p50=_p50(gpu, 'total'),
                                 cpu_engine_total_us_p50=_p50(cpu, 'engine_total'),
                                 cpu_fence_wait_us_p50=_p50(cpu, 'fence_wait')))
    return dict(note='synthetic one-layer kernel timing, not model TPS', warm_nq1=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--execute', action='store_true',
                        help='actually connect; requires the caller to hold the console lease')
    parser.add_argument('--host')
    parser.add_argument('--port', type=int, default=50054)
    parser.add_argument('--formats', nargs='+', choices=sorted(STORAGE),
                        default=['f16', 'q8_0', 'q4_0', 'kvarn4'])
    parser.add_argument('--capacity', type=int, default=28672)
    parser.add_argument('--checkpoints', type=int, nargs='*',
                        default=[1, 64, 127, 128, 129, 257, 1000, 4133, 16384, 28000])
    parser.add_argument('--queries', type=int, nargs='+', default=[1, 2, 3, 4, 7, 8, 9, 10],
                        help='query counts; 9 covers draft N=8, 10 checks reference fallback')
    parser.add_argument('--precision-tail', type=int, choices=(0, 128), default=128,
                        help='KVarN4 precision_tail_tokens')
    parser.add_argument('--bee-fixtures', type=Path, default=DEFAULT_BEE)
    parser.add_argument('--decode-steps', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=16)
    parser.add_argument('--seed', type=int, default=263)
    parser.add_argument('--output')
    parser.add_argument('--summarize', metavar='RAW.json',
                        help='offline: summarize a saved raw report (never modified)')
    parser.add_argument('--self-test', action='store_true',
                        help='offline: replay KVarN4 payloads into a fake engine image + oracles')
    args = parser.parse_args()
    if any(nq < 1 or nq > 256 for nq in args.queries):
        parser.error('--queries must be in [1,256]')
    if args.self_test:
        return self_test(args)
    if args.summarize:
        raw_path = Path(args.summarize).resolve()
        if args.output and Path(args.output).resolve() == raw_path:
            parser.error('--output must not overwrite the raw report')
        summary = summarize(json.loads(raw_path.read_text()))
        summary['source'] = str(raw_path)
        text = json.dumps(summary, indent=2)
        print(text)
        if args.output:
            Path(args.output).write_text(text + '\n')
        return 0
    if not args.execute:
        plan = dict(execute=False, note='offline plan only; pass --execute --host H to run',
                    formats=args.formats, modes=list(MODES), capacity=args.capacity,
                    checkpoints=args.checkpoints, queries=args.queries, decode_steps=args.decode_steps,
                    warmup=args.warmup, precision_tail=args.precision_tail,
                    kernels={f: [kernel_name(f, m) for m in MODES] for f in args.formats},
                    ops=['hello', 'session(2)', 'range_config(12 f16/q4_0/q8_0 | 15 kvarn4)',
                         'range_upload(13 wire0/2/3 | 16 kvarn4)', 'range_attention(14,query-indexed batch)',
                         'range_profile(20)', 'close(3)'])
        print(json.dumps(plan, indent=2))
        return 0
    if not args.host:
        parser.error('--execute needs --host')
    if args.output and Path(args.output).exists():
        parser.error('--output exists; raw reports are never overwritten')
    report = execute(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + '\n')
    return 0 if report['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
