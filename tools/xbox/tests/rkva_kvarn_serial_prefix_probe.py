#!/usr/bin/env python3
"""Gated GPU probe for native Bee KVarN4 descriptor revision 3 (serial prefix).

Offline by default: without --execute it prints the plan and opens no socket.
--self-test replays the exact op 15/16/21/14 payloads into an offline
transliteration of the engine (byte placement, coverage gate, revision-3
reference shader emulation) and checks the probe's own oracle; no socket.
With --execute it talks to an already running dedicated RKVA listener (no
model, no credentials) that the caller holds a console lease for. It never
deploys, restarts or touches anything outside its own RKVA session.

Wire assumptions (root-owned, proposed; not edited by the kernel side):
* HELLO proto_flags bit16 (native rewind v2) | bit17 (serial prefix);
* ACK capability bit7 RKVA_CAP_KVARN_SERIAL_PREFIX; without it the probe
  stops (no fallback to revision 1/2 semantics);
* op 15 with layout_version 3 returns the 8-byte generation seed like
  revision 2, and is refused for a HELLO without bit17.

Data: a canonical store of nonzero ORIGINAL float32 K/V (signed, per-row
scaled, outlier dims) per layer; records are encoded from it at seal time,
stage rows are F16(WHT256 row), tail rows F16(original). Every query of a
batch is checked against its OWN serial-prefix FP64 image (records sealed by
q_count, partial-group stage of q_count, tail [q_count-128, q_count)), built
from the canonical store only: tests/rkva_kvarn_serial_prefix_model.py. The
legacy resident-image oracle (rkva_range_kvarn4_opt_model.oracle_sources) is
a different contract and is not used here.

Cases (origin 72704, capacity 58368, H24/HKV4/D256, tail 0 and 128): batches
ending at 127/128/129/255/256/257 rows for nq 1/2/3/5/9, decode history with
append chunks <= 9, rollback 1..9 rows (rewind op 21) and re-append of NEW
content across the 256 seal and the 256-row ring wrap, every mode 0..3 (op
20 q4_kernel_mode, legacy name), plus negative cases: deep RESET then nq3
(missing tail coverage, must fail without poisoning the session), rollback
128 past a stage slot (tail 0), non-contiguous positions, last position not
the newest row, and layout 3 without bit17. Tolerances: the qualified
format-probe bounds (|dOUT| <= 3e-4 + 3e-4|OUT|, |dLSE| <= 3e-4 + 3e-5|LSE|).
These are correctness checks, not performance measurements.
"""
import argparse
import json
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ORIGIN, CAPACITY = 72704, 58368
H, HKV, D = 24, 4, 256
RECORD, PHYS = 8960, 8
ROW_BYTES = HKV * D * 2
RKVA_CAP_RANGE_KVARN4 = 1 << 2
RKVA_CAP_RANGE_PROFILE = 1 << 5
RKVA_CAP_KVARN_REWIND = 1 << 6
RKVA_CAP_KVARN_SERIAL_PREFIX = 1 << 7  # proposed, root-owned
PROTO_F16, PROTO_F32 = 1 << 0, 1 << 1
PROTO_NATIVE_REWIND_V2 = 1 << 16
PROTO_SERIAL_PREFIX = 1 << 17  # proposed, root-owned
OP_HELLO, OP_CREATE, OP_DESTROY, OP_ATTN = 1, 2, 3, 14
OP_KVARN_CONFIG, OP_KVARN_UPLOAD, OP_PROFILE, OP_KVARN_REWIND = 15, 16, 20, 21
PROFILE_ENABLE, PROFILE_RESET, PROFILE_REPORT_ONLY = 1, 2, 8
SCALE = 1.0 / 16.0


def descriptor(ptt, layout=3):
    return [ORIGIN, CAPACITY, 4, layout, 128, 128, 256, 2, 4, 4, RECORD, 3, 2, 128, ptt, 1, 1, 1]


def hello_payload(layers, flags):
    return struct.pack('<16IfQQ', flags, layers, H, HKV, D, CAPACITY, 0, 0, 128, 0, 0, 2, 0, 0,
                       0, 0, 1 / math.sqrt(D), 0, 0)


# ------------------------------------------------------------ host session ---


class Layer:
    """Canonical store + payload builder for one remote layer."""

    def __init__(self, np, sp, rng, ptt):
        self.np, self.sp, self.ptt = np, sp, ptt
        self.canon = sp.Canon(rng, CAPACITY)
        self.lay = sp.km.Layout(ORIGIN, CAPACITY, ptt, ring=sp.RING)
        self.live = 0
        self.groups = 0

    def _rows(self, rows, kind, rotated):
        np = self.np
        if not len(rows):
            return b''
        if rotated:
            return self.canon.rotated(list(rows), kind == 1).astype('<f2').tobytes()
        src = self.canon.v if kind else self.canon.k
        return src[list(rows)].astype('<f2').tobytes()

    def upload(self, live, reset):
        """op 16 payload moving the server image to `live` (canon must hold it)."""
        np = self.np
        previous = 0 if reset else self.live
        first = 0 if reset else self.groups
        groups = live // 128 - first
        header = struct.pack('<8I', live, first, groups, int(reset), 0 if reset else previous,
                             0 if reset else live - previous, 0, 0)
        records = b''.join(b''.join(self.canon.record(g, kind == 1))
                           for kind in range(2) for g in range(first, first + groups))
        stage = []
        if reset:
            lg = (live - 1) // 128
            for kind in range(2):
                image = np.zeros((384, HKV, D), '<f2')
                for t in range(live):
                    gg = (ORIGIN + t) // 128
                    if gg == 0 or t // 128 + 1 >= lg:  # sink + both ping-pong groups
                        slot = 0 if gg == 0 else 1 + (gg - 1) % 2
                        image[slot * 128 + (ORIGIN + t) % 128] = \
                            self.canon.rotated([t], kind == 1)[0].astype('<f2')
                stage.append(image.tobytes())
            tail_rows = range(live - min(live, self.ptt), live) if self.ptt else range(0)
        else:
            rows = range(previous, live)
            stage = [self._rows(rows, 0, True), self._rows(rows, 1, True)]
            tail_rows = rows if self.ptt else range(0)
        tail = [self._rows(tail_rows, 0, False), self._rows(tail_rows, 1, False)]
        self.live, self.groups = live, live // 128
        return header + records + stage[0] + stage[1] + tail[0] + tail[1]

    def rewind_layer(self, keep):
        """Per-layer op 21 block: header + ORIGINAL-F16 repair rows (K then V)."""
        begin, rows = self.sp.rewind_tail(self.live, keep, self.ptt)
        block = struct.pack('<II', self.live, keep)
        block += self._rows(range(begin, begin + rows), 0, False)
        block += self._rows(range(begin, begin + rows), 1, False)
        self.live, self.groups = keep, keep // 128
        return block

    def expected(self, q, positions):
        """Per-query serial-prefix FP64 OUT/LSE from the canonical store."""
        np, sp = self.np, self.sp
        outs, lses = [], []
        for i, pos in enumerate(positions):
            count = max(0, min(self.live, pos - ORIGIN + 1))
            ks, ex = sp.serial_sources(self.canon, self.lay, count, False)
            vs, _ = sp.serial_sources(self.canon, self.lay, count, True)
            o, l = sp.km.oracle_attention(q[i], ks, vs, ex, count, SCALE)
            outs.append(o)
            lses.append(l)
        return np.stack(outs), np.stack(lses)


def compare(np, got_out, got_lse, want_out, want_lse):
    finite = np.isfinite(want_lse)
    if not np.array_equal(np.isfinite(got_lse), finite):
        return dict(ok=False, reason='LSE finiteness mismatch')
    if not np.all(np.isfinite(got_out)):
        return dict(ok=False, reason='non-finite output')
    ok = bool(np.all(np.abs(got_out - want_out) <= 3e-4 + 3e-4 * np.abs(want_out)) and
              np.all(np.abs(got_lse[finite] - want_lse[finite]) <=
                     3e-4 + 3e-5 * np.abs(want_lse[finite])))
    return dict(ok=ok, out_max_abs=float(np.max(np.abs(got_out - want_out))),
                lse_max_abs=float(np.max(np.abs(got_lse[finite] - want_lse[finite])))
                if finite.any() else 0.0)


# -------------------------------------------------------------- scenario ---


class Probe:
    def __init__(self, np, client, rng, ptt, modes, full):
        import rkva_kvarn_serial_prefix_model as sp
        self.np, self.sp, self.client, self.rng = np, sp, client, rng
        self.ptt, self.modes, self.full = ptt, modes, full
        self.results = []
        self.generation = None

    def profile(self, flags, mode):
        body = self.client.call(OP_PROFILE, struct.pack('<IIQ', flags, mode, 0))
        return json.loads(bytes(body).decode('utf-8'))

    def configure(self):
        self.client.call(OP_CREATE, struct.pack('<4I', 0, 0, CAPACITY, 1))
        ack = self.client.call(OP_KVARN_CONFIG, struct.pack('<18I', *descriptor(self.ptt)),
                               session=0)
        if len(ack) != 8:
            raise RuntimeError('revision-3 configure returned no generation ACK')
        self.generation = struct.unpack('<Q', bytes(ack))[0]
        self.layer = Layer(self.np, self.sp, self.rng, self.ptt)

    def append(self, rows, reset=False):
        layer = self.layer
        target = rows if reset else layer.live + rows
        if layer.canon.live < target:
            layer.canon.append(target - layer.canon.live)
        payload = layer.upload(target, reset)
        self.client.call(OP_KVARN_UPLOAD, payload, session=0)
        self.generation += 1

    def rewind(self, back):
        layer = self.layer
        keep = layer.live - back
        block = layer.rewind_layer(keep)
        head = struct.pack('<QIIII', self.generation, ORIGIN + keep, 1, self.ptt, 0)
        self.client.call(OP_KVARN_REWIND, head + block, session=0)
        self.generation += 1
        layer.canon.rewind(keep)

    def batch(self, n, label):
        np = self.np
        live = self.layer.live
        positions = [ORIGIN + live - n + i for i in range(n)]
        q = self.rng.normal(0, 1, (n, H, D)).astype(np.float32)
        want = self.layer.expected(q, positions)
        baseline = None
        for mode in self.modes:
            self.profile(PROFILE_ENABLE | PROFILE_RESET, mode)
            out, lse = self.client.attention(q.ravel().tolist(), positions)
            got = (np.array(out, np.float64).reshape(n, H, D),
                   np.array(lse, np.float64).reshape(n, H))
            engine = self.profile(PROFILE_REPORT_ONLY, 0)
            engine = engine.get('engine', engine)
            buckets = [k for k in engine.get('buckets', {}) if k.startswith('attention/')]
            item = dict(label=label, ptt=self.ptt, live=live, nq=n, mode=mode,
                        vs_oracle=compare(np, *got, *want),
                        prefix_bucket=all(k.startswith('attention/kvarn4-prefix/')
                                          for k in buckets) and bool(buckets),
                        optimized_used=any('/optimized-' in k for k in buckets),
                        buckets=buckets,
                        serial_prefix=engine.get('kvarn4_serial_prefix'))
            if baseline is None:
                baseline = got
            else:
                item['vs_mode0'] = compare(np, *got, *baseline)
            self.results.append(item)

    def expect_failure(self, label, payload_positions=None, n=3):
        np = self.np
        live = self.layer.live
        positions = payload_positions or [ORIGIN + live - n + i for i in range(n)]
        q = self.rng.normal(0, 1, (len(positions), H, D)).astype(np.float32)
        body = struct.pack('<4I', 1, 0, 0, 0) + struct.pack('<%di' % len(positions), *positions)
        body += q.astype('<f4').tobytes()
        self.client.call(OP_ATTN, body, session=0, nq=len(positions), success=False)
        self.results.append(dict(label=label, ptt=self.ptt, live=live, nq=len(positions),
                                 rejected=True, vs_oracle=dict(ok=True)))
        # The refusal must not poison the session: nq1 at the newest row still works.
        self.batch(1, label + ' then nq1')

    def run(self):
        self.configure()
        for live in (127, 128, 129, 255, 256, 257):
            for n in (1, 2, 3, 5, 9):
                self.configure_fresh()
                self.append(live - n, reset=True)
                self.append(n)
                self.batch(n, 'boundary %d' % live)
        # Decode history across the 256 seal / ring wrap with rollback+reappend.
        self.configure_fresh()
        self.append(230, reset=True)
        steps = [(9, 0), (9, 4), (9, 2), (5, 0), (9, 9), (3, 1), (9, 0), (9, 7), (2, 0),
                 (9, 3), (9, 0), (9, 8), (9, 0), (5, 2), (9, 0), (9, 0), (9, 4), (9, 0)]
        for chunk, back in steps:
            self.append(chunk)
            self.batch(chunk, 'decode append')
            if back:
                self.rewind(back)
                self.append(back)
                self.batch(min(9, back + 1), 'rollback %d+reappend' % back)
        live = self.layer.live
        self.expect_failure('gap positions', [ORIGIN + live - 3, ORIGIN + live - 1,
                                              ORIGIN + live - 1])
        self.expect_failure('not newest', [ORIGIN + live - 3, ORIGIN + live - 2])
        if self.ptt:
            self.configure_fresh()
            self.append(257, reset=True)
            self.batch(1, 'deep reset nq1')
            self.expect_failure('deep reset nq3 (tail coverage)')
        else:
            self.configure_fresh()
            self.append(1, reset=True)
            self.append(256)
            self.rewind(128)
            self.expect_failure('rollback 128 (stage slot)')
        # Leading queries below the origin are masked (empty), not uncovered.
        self.configure_fresh()
        self.append(2, reset=True)
        self.batch(5, 'masked below origin')
        if self.full:
            self.configure_fresh()
            self.append(CAPACITY - 2, reset=True)
            self.append(2)
            self.batch(2, 'capacity edge')
        self.client.call(OP_DESTROY, session=0)

    def configure_fresh(self):
        self.client.call(OP_DESTROY, session=0)
        self.configure()


# ------------------------------------------------------------- self-test ---


class FakeServer:
    """Offline engine transliteration fed with the probe's real payloads:
    rkva_kvarn_range valid_upload/placement (stage_chunks, 256-row ring), the
    revision-3 rewind repair, valid_prefix_queries and the RKVA_SERIAL_PREFIX
    reference shader emulation. CPU only."""

    def __init__(self, np, sp, serial_cap=True):
        self.np, self.sp, self.km = np, sp, sp.km
        self.serial_cap = serial_cap
        self.flags = 0
        self.mode = 0
        self.lay = None
        self.capabilities = 0

    def hello(self, flags):
        self.flags = flags
        caps = RKVA_CAP_RANGE_KVARN4 | RKVA_CAP_RANGE_PROFILE
        if flags & PROTO_NATIVE_REWIND_V2:
            caps |= RKVA_CAP_KVARN_REWIND
        if self.serial_cap and flags & PROTO_SERIAL_PREFIX:
            caps |= RKVA_CAP_KVARN_SERIAL_PREFIX
        self.capabilities = caps
        return caps

    def call(self, op, payload=b'', session=None, nq=0, success=True, **_):
        try:
            body = self._call(op, bytes(payload), nq)
            ok = True
        except (AssertionError, ValueError):
            body, ok = b'', False
        assert ok == success, (op, ok)
        return body

    def _call(self, op, payload, nq):
        np, sp = self.np, self.sp
        if op in (OP_CREATE, OP_DESTROY):
            self.lay = None
            return b''
        if op == OP_PROFILE:
            flags, mode, _ = struct.unpack('<IIQ', payload)
            if not flags & PROFILE_REPORT_ONLY:
                self.mode = mode
            return json.dumps(dict(buckets={'attention/kvarn4-prefix/reference/x': {}},
                                   kvarn4_serial_prefix={})).encode()
        if op == OP_KVARN_CONFIG:
            c = struct.unpack('<18I', payload)
            assert c[3] in (1, 2, 3) and c[0] == ORIGIN and c[1] == CAPACITY
            assert c[3] != 3 or self.flags & PROTO_SERIAL_PREFIX
            assert c[3] != 2 or self.flags & PROTO_NATIVE_REWIND_V2
            self.ptt = c[14]
            self.lay = self.km.Layout(ORIGIN, CAPACITY, self.ptt, ring=256 if c[3] == 3 else 128)
            self.planes = [bytearray(b'\xff' * self.lay.half) for _ in range(2)]
            self.live, self.groups, self.ready = 0, 0, False
            self.cov = sp.Coverage()
            self.generation = 7 << 32
            return struct.pack('<Q', self.generation)
        if op == OP_KVARN_UPLOAD:
            return self._upload(payload)
        if op == OP_KVARN_REWIND:
            gen, pos0, n_layers, tail, reserved = struct.unpack_from('<QIIII', payload)
            assert gen == self.generation and n_layers == 1 and tail == self.ptt and not reserved
            expected_live, new_live = struct.unpack_from('<II', payload, 24)
            keep = max(0, min(pos0 - ORIGIN, CAPACITY))
            assert expected_live == self.live and new_live == min(expected_live, keep)
            assert expected_live - new_live <= 128
            begin, rows = sp.rewind_tail(self.live, new_live, self.ptt)
            data = payload[32:]
            assert len(data) == 2 * rows * ROW_BYTES
            for kind in range(2):
                for r in range(rows):
                    src = data[(kind * rows + r) * ROW_BYTES:(kind * rows + r + 1) * ROW_BYTES]
                    off = self.lay.records_bytes + self.km.STAGE_BYTES + \
                        ((begin + r) % self.lay.ring) * ROW_BYTES
                    self.planes[kind][off:off + ROW_BYTES] = src
            self.generation += 1
            self.cov.rewind(new_live, begin, rows)
            self.live, self.groups, self.ready = new_live, new_live // 128, new_live != 0
            return b''
        if op == OP_ATTN:
            return self._attention(payload, nq)
        raise ValueError(op)

    def _upload(self, payload):
        np, km = self.np, self.km
        live, rb, rg, flags, ab, ar, r0, r1 = struct.unpack_from('<8I', payload)
        reset = flags == 1
        assert 0 < live <= CAPACITY and flags in (0, 1) and not (r0 or r1)
        if reset:
            assert rb == 0 and not ab and not ar and rg == live // 128
        else:
            assert self.ready and live >= self.live and rb <= self.groups
            assert ab == self.live and ar == live - self.live
            assert max(self.groups, rb + rg) >= live // 128 and rb + rg <= live // 128
        stage_rows = 384 if reset else ar
        tail_rows = (min(live, self.ptt) if reset else ar) if self.ptt else 0
        rec = rg * PHYS * RECORD
        assert len(payload) == 32 + 2 * (rec + (stage_rows + tail_rows) * ROW_BYTES)
        cursor = 32
        for kind in range(2):
            self.planes[kind][rb * PHYS * RECORD:rb * PHYS * RECORD + rec] = \
                payload[cursor:cursor + rec]
            cursor += rec
        for kind in range(2):
            base = self.lay.records_bytes
            if reset:
                self.planes[kind][base:base + 384 * ROW_BYTES] = \
                    payload[cursor:cursor + 384 * ROW_BYTES]
            else:
                for i in range(ar):
                    t = ORIGIN + ab + i
                    gg = t // 128
                    slot = 0 if gg == 0 else 1 + (gg - 1) % 2
                    off = base + (slot * 128 + t % 128) * ROW_BYTES
                    self.planes[kind][off:off + ROW_BYTES] = \
                        payload[cursor + i * ROW_BYTES:cursor + (i + 1) * ROW_BYTES]
            cursor += stage_rows * ROW_BYTES
        for kind in range(2):
            begin = live - tail_rows if reset else ab
            for i in range(max(0, tail_rows - self.lay.ring), tail_rows):
                off = self.lay.records_bytes + km.STAGE_BYTES + \
                    ((begin + i) % self.lay.ring) * ROW_BYTES
                self.planes[kind][off:off + ROW_BYTES] = \
                    payload[cursor + i * ROW_BYTES:cursor + (i + 1) * ROW_BYTES]
            cursor += tail_rows * ROW_BYTES
        self.groups = rg if reset else max(self.groups, rb + rg)
        self.generation += 1
        if reset:
            self.cov.reset(live, self.ptt)
        else:
            self.cov.append(live, self.ptt)
        self.live, self.ready = live, True
        return b''

    def _attention(self, payload, nq):
        np, km, sp = self.np, self.km, self.sp
        assert struct.unpack_from('<4I', payload) == (1, 0, 0, 0)
        positions = list(struct.unpack_from('<%di' % nq, payload, 16))
        q = np.frombuffer(payload[16 + 4 * nq:], '<f4').reshape(nq, H, D)
        assert self.ready
        assert sp.valid_prefix_queries(positions, ORIGIN, self.live, self.cov, self.ptt)
        Kp = km.Plane(bytes(self.planes[0]), self.lay, self.live)
        Vp = km.Plane(bytes(self.planes[1]), self.lay, self.live)
        outs, lses = [], []
        for i, pos in enumerate(positions):
            count = km.causal_limit(self.lay, self.live, pos)
            # Shader emulation reads only the buffer; give it the buffer's own
            # resident decode of the prefix window for its bitwise load checks.
            k_src = self._window(count, False)
            v_src = self._window(count, True)
            if self.mode and nq <= 9:
                _, _, o, l = km.run_optimized(Kp, Vp, q[i], self.live, pos, SCALE, self.mode,
                                              64, k_src, v_src, prefix=True)
            else:
                _, o, l = km.run_reference(Kp, Vp, q[i], self.live, pos, SCALE, k_src, v_src,
                                           prefix=True)
            outs.append(np.asarray(o, np.float32))
            lses.append(np.asarray(l, np.float32))
        return np.concatenate([np.stack(outs).ravel(), np.stack(lses).ravel()]).astype(
            '<f4').tobytes()

    def _window(self, count, value):
        km = self.km
        lay = self.lay
        plane = bytes(self.planes[1 if value else 0])
        out = self.np.zeros((count, HKV, D), self.np.float32)
        tail_begin = count - min(count, lay.ptt)
        lg = (count - 1) // 128 if count else 0
        stage = self.np.frombuffer(plane[lay.records_bytes:lay.records_bytes + km.STAGE_BYTES],
                                   '<f2').astype(self.np.float32).reshape(384, HKV, D)
        t0 = lay.records_bytes + km.STAGE_BYTES
        tail = self.np.frombuffer(plane[t0:t0 + lay.tail_rows * ROW_BYTES], '<f2').astype(
            self.np.float32).reshape(lay.tail_rows, HKV, D) if lay.ptt else None
        for row in range(count):
            g, p = divmod(row, 128)
            gg = (ORIGIN + row) // 128
            if row >= tail_begin:
                out[row] = tail[row % lay.ring]
            elif gg == 0 or (g == lg and count % 128):
                slot = 0 if gg == 0 else 1 + (gg - 1) % 2
                out[row] = stage[slot * 128 + p]
            else:
                if p == 0:
                    dec = [km.decode_record(plane[(g * PHYS + ph) * RECORD:
                                                  (g * PHYS + ph + 1) * RECORD], value)
                           for ph in range(PHYS)]
                for ph in range(PHYS):
                    kv, s = divmod(ph, 2)
                    out[row, kv, s * 128:(s + 1) * 128] = dec[ph][p]
        return out


class FakeClient:
    def __init__(self, server):
        self.server = server

    def call(self, op, payload=b'', session=None, nq=0, success=True, **kw):
        return self.server.call(op, payload, session=session, nq=nq, success=success)

    def attention(self, q, positions):
        import numpy as np
        body = struct.pack('<4I', 1, 0, 0, 0) + struct.pack('<%di' % len(positions), *positions)
        body += np.asarray(q, '<f4').tobytes()
        raw = np.frombuffer(self.call(OP_ATTN, body, nq=len(positions)), '<f4')
        n = len(positions) * H * D
        return raw[:n].tolist(), raw[n:].tolist()


def self_test(args):
    import numpy as np
    import rkva_kvarn_serial_prefix_model as sp
    rng = np.random.default_rng(args.seed)
    # Capability negotiation: no bit17 -> no cap, layout 3 refused.
    server = FakeServer(np, sp)
    assert not server.hello(PROTO_F16 | PROTO_NATIVE_REWIND_V2) & RKVA_CAP_KVARN_SERIAL_PREFIX
    server.call(OP_CREATE)
    server.call(OP_KVARN_CONFIG, struct.pack('<18I', *descriptor(128)), success=False)
    legacy = FakeServer(np, sp, serial_cap=False)
    assert not legacy.hello(PROTO_F16 | PROTO_NATIVE_REWIND_V2 | PROTO_SERIAL_PREFIX) & \
        RKVA_CAP_KVARN_SERIAL_PREFIX
    summary = []
    for ptt in (128, 0):
        server = FakeServer(np, sp)
        caps = server.hello(PROTO_F16 | PROTO_NATIVE_REWIND_V2 | PROTO_SERIAL_PREFIX)
        assert caps & RKVA_CAP_KVARN_SERIAL_PREFIX
        probe = Probe(np, FakeClient(server), rng, ptt, (0, 1), full=False)
        # Shortened scenario (the full one is --execute only): boundaries across
        # the 128 seal and a rollback across the 256 seal and ring wrap.
        probe.configure()
        for live, n in ((128, 2), (129, 3), (257, 5)):
            probe.configure_fresh()
            probe.append(live - n, reset=True)
            probe.append(n)
            probe.batch(n, 'boundary %d' % live)
        probe.configure_fresh()
        probe.append(250, reset=True)
        probe.append(9)
        probe.batch(9, 'decode append')
        probe.rewind(4)
        probe.append(4)
        probe.batch(5, 'rollback 4+reappend')
        live = probe.layer.live
        probe.expect_failure('gap positions', [ORIGIN + live - 3, ORIGIN + live - 1,
                                               ORIGIN + live - 1])
        if ptt:
            probe.configure_fresh()
            probe.append(257, reset=True)
            probe.expect_failure('deep reset nq3 (tail coverage)')
        else:
            probe.configure_fresh()
            probe.append(1, reset=True)
            probe.append(256)
            probe.rewind(128)
            probe.expect_failure('rollback 128 (stage slot)')
        probe.configure_fresh()
        probe.append(2, reset=True)
        probe.batch(5, 'masked below origin')
        bad = [r for r in probe.results
               if not r['vs_oracle']['ok'] or not r.get('vs_mode0', {'ok': True})['ok']]
        assert not bad, bad[:2]
        summary.append(dict(ptt=ptt, checks=len(probe.results)))
    print(json.dumps(dict(self_test='ok', note='offline fake server; no GPU', runs=summary)))
    return 0


# ----------------------------------------------------------------- main ---


def execute(args):
    import numpy as np
    from rkva_range_probe import Client
    rng = np.random.default_rng(args.seed)
    report = dict(success=False, host=args.host, port=args.port, origin=ORIGIN,
                  capacity=CAPACITY, modes=args.modes, runs=[],
                  note='serial-prefix correctness probe; not a performance measurement')
    # Negative negotiation first: layout 3 without bit17 must be refused.
    plain = Client(args.host, args.port)
    try:
        plain.call(OP_HELLO, hello_payload(1, PROTO_F16 | PROTO_F32 | PROTO_NATIVE_REWIND_V2))
        plain.call(OP_CREATE, struct.pack('<4I', 0, 0, CAPACITY, 1))
        plain.call(OP_KVARN_CONFIG, struct.pack('<18I', *descriptor(128)), session=0,
                   success=False)
        plain.call(OP_DESTROY, session=0)
        report['layout3_without_bit17_refused'] = True
    finally:
        plain.socket.close()
    client = Client(args.host, args.port)
    try:
        ack = struct.unpack('<IIIIQQQ', client.call(
            OP_HELLO, hello_payload(1, PROTO_F16 | PROTO_F32 | PROTO_NATIVE_REWIND_V2 |
                                    PROTO_SERIAL_PREFIX)))
        caps = ack[5]
        report['server_version'] = ack[3]
        report['capabilities'] = caps
        need = RKVA_CAP_RANGE_KVARN4 | RKVA_CAP_RANGE_PROFILE | RKVA_CAP_KVARN_REWIND | \
            RKVA_CAP_KVARN_SERIAL_PREFIX
        if caps & need != need:
            report['reason'] = 'listener lacks kvarn4/profile/rewind/serial-prefix capability'
            return report
        for ptt in args.precision_tails:
            probe = Probe(np, client, rng, ptt, args.modes, full=args.full)
            probe.run()
            ok = all(r['vs_oracle']['ok'] and r.get('vs_mode0', {'ok': True})['ok'] and
                     r.get('prefix_bucket', True) for r in probe.results)
            report['runs'].append(dict(ptt=ptt, success=ok, results=probe.results))
        report['success'] = bool(report['runs']) and all(r['success'] for r in report['runs'])
    finally:
        try:
            client.call(OP_PROFILE, struct.pack('<IIQ', 0, 0, 0))  # reference, profiling off
        except Exception:
            pass
        client.socket.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--execute', action='store_true',
                        help='talk to a live RKVA listener (requires a console lease)')
    parser.add_argument('--self-test', action='store_true',
                        help='offline fake-server replay of the real payloads (CPU only)')
    parser.add_argument('--host')
    parser.add_argument('--port', type=int, default=50054)
    parser.add_argument('--modes', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--precision-tails', type=int, nargs='+', default=[128, 0])
    parser.add_argument('--full', action='store_true', help='add the capacity-edge case')
    parser.add_argument('--seed', type=int, default=20261266)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.self_test:
        return self_test(args)
    if not args.execute:
        print(json.dumps(dict(
            execute=False, note='offline plan only; pass --execute --host H (console lease)',
            origin=ORIGIN, capacity=CAPACITY, descriptor_layout=3,
            hello_flags=['F16', 'F32', 'bit16 native rewind v2', 'bit17 serial prefix'],
            required_caps=['bit2 kvarn4', 'bit5 profile', 'bit6 rewind', 'bit7 serial prefix'],
            boundaries=[127, 128, 129, 255, 256, 257], nq=[1, 2, 3, 5, 9],
            precision_tails=args.precision_tails, modes=args.modes), indent=2))
        return 0
    if not args.host:
        parser.error('--execute needs --host')
    if any(m not in (0, 1, 2, 3) for m in args.modes) or \
            any(t not in (0, 128) for t in args.precision_tails):
        parser.error('modes 0..3, precision tails 0|128')
    report = execute(args)
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, 'w') as stream:
            stream.write(text + '\n')
    print(json.dumps(dict(success=report['success'], reason=report.get('reason'),
                          runs=[dict(ptt=r['ptt'], success=r['success'],
                                     checks=len(r['results'])) for r in report['runs']])))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
