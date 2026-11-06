#!/usr/bin/env python3
"""Serial-prefix (descriptor revision 3) model of native Bee KVarN4 attention.

MTP batched verification sends nq contiguous queries ending at the newest row.
Query i must see exactly the image a sequential decoder would have had after
q_count_i = position_i - origin + 1 rows: records sealed by then, the staged
partial group of q_count_i and the ORIGINAL-F16 tail window
[q_count_i - 128, q_count_i). Revisions 1/2 (legacy resident image) instead
read the image of active_count for every query; that remains a separate
oracle (rkva_range_kvarn4_opt_model.oracle_sources) and is shown here to
differ for nq > 1 across a 128-row seal.

Independence:
* canonical store: nonzero ORIGINAL float32 K/V rows (signed, per-row scaled,
  outlier dims), versioned through rewinds. Records are encoded from it
  (Bee-like signed scale/zero/other KVarN4 records of the WHT256-rotated
  group), stage rows are F16(rotated row), tail rows are F16(original row).
* serial image per query: built from the canonical store at that prefix only,
  never from the resident GPU buffer, then mapped to float64 (body through
  H256) for the FP64 oracle.
* resident buffer: an engine-layout byte image (records, 384-row stage slots,
  256-row ORIGINAL-F16 ring) updated only by RESET / append / REWIND
  operations exactly as remote-attn-engine.cpp issues the copies. Unwritten
  bytes are 0xFF (F16 NaN).
* kernels: float32 emulations of rkva_range_attn_kvarn4.hlsl and
  rkva_range_attn_kvarn4_opt.hlsl compiled with RKVA_SERIAL_PREFIX=1
  (rkva_range_kvarn4_opt_model run_reference/run_optimized with prefix=True,
  waves 32/64/128, modes 1..3) read the resident buffer; every decoded K/V must
  equal the per-query serial image bit for bit, and OUT/LSE are compared with
  the FP64 oracle using the qualified tolerances of the legacy model.
* gate: a Python mirror of rkva_kvarn_range::valid_prefix_queries and the
  coverage helpers; every accepted batch must pass the bitwise check, and the
  designed negative batches (deep RESET, rollback past a stage slot) must be
  rejected and really be non-resident.

CPU only: no GPU, no hardware, no performance claim. Run:
    python3 tests/rkva_kvarn_serial_prefix_model.py [--quick]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rkva_range_kvarn4_opt_model as km  # noqa: E402
import rkva_range_q4_opt_model as m4  # noqa: E402

N_HEAD_KV, PHYS, RECORD, ROW_BYTES = km.N_HEAD_KV, km.PHYS, km.RECORD, km.ROW_BYTES
STAGE_BYTES = km.STAGE_BYTES
ORIGIN, CAPACITY = 72704, 58368
RING = 256
SCALE = 1.0 / 16.0
F32 = np.float32


# ------------------------------------------------------- canonical CPU store ---


class Canon:
    """ORIGINAL float32 K/V per row; rows past a rewind are regenerated."""

    def __init__(self, rng, capacity):
        self.rng = rng
        self.k = np.zeros((capacity, N_HEAD_KV, 256), F32)
        self.v = np.zeros((capacity, N_HEAD_KV, 256), F32)
        self.live = 0
        self.records = {}

    def _rows(self, n):
        rng = self.rng
        x = rng.normal(0, 1, (n, N_HEAD_KV, 256))
        x *= 10.0 ** rng.uniform(-1.5, 0.7, (n, 1, 1))  # varied per-row scale
        x *= rng.choice([-1.0, 1.0], (n, N_HEAD_KV, 1))  # signed
        x[:, :, rng.integers(0, 256, 3)] *= 6.0  # outlier dims
        x = np.where(np.abs(x) < 1e-3, np.copysign(1e-3, x + 1e-12), x)
        return x.astype(F32)

    def append(self, n):
        a, b = self.live, self.live + n
        self.k[a:b], self.v[a:b] = self._rows(n), self._rows(n)
        for g in range(a // 128, (b + 127) // 128):
            self.records.pop(g, None)
        self.live = b

    def rewind(self, keep):
        for g in range(keep // 128, (self.live + 127) // 128):
            self.records.pop(g, None)
        self.k[keep : self.live] = np.nan
        self.v[keep : self.live] = np.nan
        self.live = keep

    def rotated(self, rows, value):
        x = (self.v if value else self.k)[rows].astype(np.float64)
        return (x @ km.H256).astype(F32)

    def record(self, group, value):
        """[PHYS] KVarN4 record blobs of a complete group, encoded from the store."""
        key = group
        if key not in self.records:
            self.records[key] = (self._encode(group, False), self._encode(group, True))
        return self.records[key][1 if value else 0]

    def _encode(self, group, value):
        rows = np.arange(group * 128, group * 128 + 128)
        assert rows[-1] < self.live, "record of an incomplete group"
        rot = self.rotated(rows, value).astype(np.float64)  # [tok, kv, 256]
        out = []
        for ph in range(PHYS):
            kv, s = divmod(ph, 2)
            block = rot[:, kv, s * 128 : (s + 1) * 128]  # [tok, dim]
            m = block if value else block.T  # V: row=tok col=dim; K: row=dim col=tok
            out.append(encode_record(m, (group * PHYS + ph) & 1))
        return out


def encode_record(m, flip):
    """Affine 4-bit rows with signed F16 scale/zero/other (decode_record form)."""
    m = np.asarray(m, np.float64)
    other = np.abs(m).max(axis=0)
    other = np.where(other > 0, other, 1.0) * np.where(np.arange(128) % 3 == 0, -1.0, 1.0)
    other = other.astype(np.float16)
    z = m / other.astype(np.float64)[None, :]
    lo, hi = z.min(axis=1), z.max(axis=1)
    signed = (np.arange(128) % 2 == flip)  # half the rows use a negative scale
    span = np.where(hi > lo, hi - lo, 1.0) / 15.0
    scale = np.where(signed, -span, span).astype(np.float16)
    zero = np.where(signed, hi, lo).astype(np.float16)
    q = np.rint((z - zero.astype(np.float64)[:, None]) / scale.astype(np.float64)[:, None])
    q = np.clip(q, 0, 15).astype(np.uint8).reshape(-1)
    nib = (q[0::2] | (q[1::2] << 4)).astype(np.uint8).tobytes()
    blob = nib + scale.astype("<f2").tobytes() + zero.astype("<f2").tobytes() + \
        other.astype("<f2").tobytes()
    assert len(blob) == RECORD
    return blob


def serial_sources(canon, lay, count, value):
    """Per-query serial image [count, kv, 256] in storage domain + exact mask."""
    out = np.zeros((count, N_HEAD_KV, 256), F32)
    live_group = (count - 1) // 128 if count else 0
    tail_begin = count - min(count, lay.ptt)
    src = canon.v if value else canon.k
    for row in range(count):
        group, pos = divmod(row, 128)
        gg = lay.origin // 128 + group
        if row >= tail_begin:
            out[row] = src[row].astype(np.float16).astype(F32)
        elif gg == 0 or (group == live_group and count % 128):
            out[row] = canon.rotated([row], value)[0].astype(np.float16).astype(F32)
        else:
            if pos == 0:  # body rows start every record group at pos 0
                blobs = canon.record(group, value)
                dec = [km.decode_record(b, value) for b in blobs]
            for ph in range(PHYS):
                kv, s = divmod(ph, 2)
                out[row, kv, s * 128 : (s + 1) * 128] = dec[ph][pos]
    return out, np.arange(count) >= tail_begin


# ------------------------------------------------- resident engine image ---


class Coverage:
    """Python mirror of rkva_kvarn_range::prefix_coverage and helpers."""

    def __init__(self):
        self.tail_begin = 0
        self.stage_begin = 0

    def reset(self, live, ptt):
        lg = (live - 1) // 128 if live else 0
        self.tail_begin = live - min(live, ptt)
        self.stage_begin = (lg - 1) * 128 if lg else 0

    def append(self, live, ptt):
        lg = (live - 1) // 128 if live else 0
        if ptt:
            self.tail_begin = max(self.tail_begin, live - RING if live > RING else 0)
        self.stage_begin = max(self.stage_begin, (lg - 1) * 128 if lg else 0)

    def rewind(self, keep, rb, rr):
        self.tail_begin = min(self.tail_begin, keep)
        if rr and rb < self.tail_begin <= rb + rr:
            self.tail_begin = rb


def valid_prefix_queries(positions, origin, live, cov, ptt):
    n = len(positions)
    if not n or n > 128 or not live:
        return False
    if any(positions[i] != positions[i - 1] + 1 for i in range(1, n)):
        return False
    if positions[-1] != origin + live - 1:
        return False
    for i in range(n):
        count = live - (n - 1 - i)
        if count <= 0:
            continue
        if ptt:
            return cov.tail_begin <= max(0, count - ptt)
        if count % 128 and cov.stage_begin > count // 128 * 128:
            return False
    return True


def rewind_tail(before, keep, tail):
    begin = keep - tail if keep > tail else 0
    end = min(keep, before - tail if before > tail else 0)
    return begin, (end - begin if tail and end > begin else 0)


class Resident:
    """K/V planes laid out exactly as the revision-3 engine session buffer."""

    def __init__(self, canon, lay):
        self.canon, self.lay = canon, lay
        self.planes = [bytearray(b"\xff" * lay.half), bytearray(b"\xff" * lay.half)]
        self.live = 0
        self.cov = Coverage()

    def _record(self, group):
        for value in (0, 1):
            for ph, blob in enumerate(self.canon.record(group, bool(value))):
                off = (group * PHYS + ph) * RECORD
                self.planes[value][off : off + RECORD] = blob

    def _stage(self, row):
        gg = (self.lay.origin + row) // 128
        slot = 0 if gg == 0 else 1 + (gg - 1) % 2
        off = self.lay.records_bytes + (slot * 128 + (self.lay.origin + row) % 128) * ROW_BYTES
        for value in (0, 1):
            data = self.canon.rotated([row], bool(value))[0].astype("<f2").tobytes()
            self.planes[value][off : off + ROW_BYTES] = data

    def _tail(self, row):
        off = self.lay.records_bytes + STAGE_BYTES + (row % self.lay.ring) * ROW_BYTES
        for value, src in ((0, self.canon.k), (1, self.canon.v)):
            self.planes[value][off : off + ROW_BYTES] = src[row].astype("<f2").tobytes()

    def reset(self, live):
        """Upload flags=1: every complete record, the producer's 384-row stage
        image (sink slot + the two newest ping-pong groups) and min(live, ptt)
        tail rows."""
        assert self.canon.live >= live
        self.planes = [bytearray(b"\xff" * self.lay.half), bytearray(b"\xff" * self.lay.half)]
        for g in range(live // 128):
            self._record(g)
        lg = (live - 1) // 128
        for t in range(live):
            if (self.lay.origin + t) // 128 == 0 or t // 128 + 1 >= lg:
                self._stage(t)
        if self.lay.ptt:
            for t in range(live - min(live, self.lay.ptt), live):
                self._tail(t)
        self.live = live
        self.cov.reset(live, self.lay.ptt)

    def append(self, live):
        prev = self.live
        assert self.canon.live >= live > prev
        for g in range(prev // 128, live // 128):
            self._record(g)
        for t in range(prev, live):
            self._stage(t)
            if self.lay.ptt:
                self._tail(t)
        self.live = live
        self.cov.append(live, self.lay.ptt)

    def rewind(self, keep):
        before = self.live
        assert before - keep <= 128
        rb, rr = rewind_tail(before, keep, self.lay.ptt)
        for t in range(rb, rb + rr):
            self._tail(t)
        self.live = keep
        self.cov.rewind(keep, rb, rr)


# ------------------------------------------------------------------ checks ---


class Stats:
    def __init__(self):
        self.queries = self.opt_runs = self.batches = self.rejected = 0
        self.worst_ref = self.worst_opt = 0.0
        self.legacy_worst = 0.0


def planes(res):
    return (km.Plane(bytes(res.planes[0]), res.lay, res.live),
            km.Plane(bytes(res.planes[1]), res.lay, res.live))


def check_query(canon, res, KV, q, pos, modes, waves, st):
    lay, live = res.lay, res.live
    Kp, Vp = KV
    count = km.causal_limit(lay, live, pos)
    k_src, exact = serial_sources(canon, lay, count, False)
    v_src, _ = serial_sources(canon, lay, count, True)
    assert np.all(np.isfinite(k_src)) and np.all(np.isfinite(v_src))
    o64, l64 = km.oracle_attention(q, k_src, v_src, exact, count, SCALE)
    _, r_out, r_lse = km.run_reference(Kp, Vp, q, live, pos, SCALE, k_src, v_src, prefix=True)
    re_o, re_l = m4.compare("reference", r_out, r_lse, o64, l64)
    mag = max(1.0, float(np.max(np.abs(o64))))
    lmag = max(1.0, float(np.nanmax(np.abs(np.where(np.isinf(l64), 0, l64)))))
    assert re_o <= 1e-4 * mag and re_l <= 1e-4 * lmag, ("reference vs oracle", pos, re_o, re_l)
    st.worst_ref = max(st.worst_ref, re_o / mag)
    st.queries += 1
    for mode in modes:
        for wave in waves:
            _, _, out, lse = km.run_optimized(Kp, Vp, q, live, pos, SCALE, mode, wave, k_src,
                                              v_src, prefix=True)
            e_o, e_l = m4.compare("opt", out, lse, o64, l64)
            assert e_o <= max(2e-5 * mag, 4 * re_o) and e_l <= max(2e-5, 4 * re_l), (
                "optimized vs oracle", pos, mode, wave, e_o, e_l)
            st.worst_opt = max(st.worst_opt, e_o / mag)
            st.opt_runs += 1
    return o64, l64


def check_batch(canon, res, n, rng, st, opt_all=False, label=""):
    positions = list(range(ORIGIN + res.live - n, ORIGIN + res.live))
    assert valid_prefix_queries(positions, ORIGIN, res.live, res.cov, res.lay.ptt), (
        "engine gate refused a covered batch", label, n, res.live)
    q = rng.normal(0, 1, (n, km.N_HEAD, km.D)).astype(F32)
    KV = planes(res)
    for i, pos in enumerate(positions):
        # The engine runs nq <= 9 through the optimized prefix PSO (modes 1..3)
        # and larger batches through the reference prefix PSO only.
        if n <= 9 and (opt_all or i in (0, n - 1)):
            modes, waves = ((1, 2, 3), (32, 64, 128)) if opt_all else ((1,), (64,))
        else:
            modes, waves = (), ()
        check_query(canon, res, KV, q[i], pos, modes, waves, st)
    st.batches += 1
    print(f"  {label:<28} ptt={res.lay.ptt:3d} live={res.live:5d} nq={n:3d} "
          f"tail_begin={res.cov.tail_begin} stage_begin={res.cov.stage_begin} ok")


def expect_rejected(canon, res, n, rng, st, label):
    positions = list(range(ORIGIN + res.live - n, ORIGIN + res.live))
    assert not valid_prefix_queries(positions, ORIGIN, res.live, res.cov, res.lay.ptt), label
    # ... and the prefix image really is not resident: the first non-empty query
    # trips the bitwise load check or the NaN poison.
    q = rng.normal(0, 1, (km.N_HEAD, km.D)).astype(F32)
    try:
        check_query(canon, res, planes(res), q, positions[0], (), (), Stats())
    except AssertionError:
        pass
    else:
        raise AssertionError(f"{label}: rejected batch was resident after all")
    st.rejected += 1
    print(f"  {label:<28} ptt={res.lay.ptt:3d} live={res.live:5d} nq={n:3d} rejected "
          "(and non-resident)")


def legacy_gap(canon, lay_prefix, ops, n, rng, st):
    """Legacy resident-image semantics (ring 128, window = active_count) on the
    same history versus the serial-prefix FP64 oracle."""
    lay = km.Layout(lay_prefix.origin, lay_prefix.capacity, lay_prefix.ptt, ring=128)
    res = Resident(canon, lay)
    for op, arg in ops:
        getattr(res, op)(arg)
    live = res.live
    k_res, ex = km.oracle_sources(lay, bytes(res.planes[0]), live, value=False)
    v_res, _ = km.oracle_sources(lay, bytes(res.planes[1]), live, value=True)
    Kp = km.Plane(bytes(res.planes[0]), lay, live)
    Vp = km.Plane(bytes(res.planes[1]), lay, live)
    worst = 0.0
    q = rng.normal(0, 1, (n, km.N_HEAD, km.D)).astype(F32)
    for i in range(n):
        pos = ORIGIN + live - n + i
        count = live - n + i + 1
        _, out, _ = km.run_reference(Kp, Vp, q[i], live, pos, SCALE, k_res, v_res)
        ks, exs = serial_sources(canon, lay_prefix, count, False)
        vs, _ = serial_sources(canon, lay_prefix, count, True)
        o64, _ = km.oracle_attention(q[i], ks, vs, exs, count, SCALE)
        mag = max(1.0, float(np.max(np.abs(o64))))
        worst = max(worst, float(np.max(np.abs(out - o64))) / mag)
    del ex
    st.legacy_worst = max(st.legacy_worst, worst)
    return worst


def run_boundaries(rng, ptt, lives, nqs, st, full):
    lay = km.Layout(ORIGIN, CAPACITY, ptt, ring=RING)
    for live in lives:
        for n in nqs:
            canon = Canon(rng, CAPACITY)
            canon.append(live)
            res = Resident(canon, lay)
            res.reset(live - n)  # history, then the verification chunk
            res.append(live)
            check_batch(canon, res, n, rng, st, opt_all=full and n in (2, 9),
                        label=f"boundary {live}")
            if n > 1 and live in (128, 129, 256, 257):  # batch spans a seal
                gap = legacy_gap(canon, lay, [("reset", live - n), ("append", live)], n, rng, st)
                print(f"    legacy resident-image gap vs serial oracle: {gap:.2e}")


def run_decode_history(rng, ptt, st, full):
    """Incremental decode across seals and ring wraps with MTP rollback/reappend."""
    lay = km.Layout(ORIGIN, CAPACITY, ptt, ring=RING)
    canon = Canon(rng, CAPACITY)
    canon.append(230)
    res = Resident(canon, lay)
    res.reset(230)  # appends then cross the 256 seal, wrap the 256-row ring
    steps = [(9, 0), (9, 4), (9, 2), (5, 0), (9, 9), (3, 1), (9, 0), (9, 7), (2, 0),
             (9, 3), (9, 0), (9, 8), (9, 0), (5, 2), (9, 0), (9, 0), (9, 4), (9, 0)]
    if full:
        steps += [(9, b % 5) for b in range(24)]
    for chunk, back in steps:
        canon.append(chunk)
        res.append(canon.live)
        check_batch(canon, res, chunk, rng, st, opt_all=full and back == 0 and chunk == 9,
                    label="decode append")
        if back:
            canon.rewind(canon.live - back)
            res.rewind(canon.live)
            canon.append(back)  # different content for the re-verified rows
            res.append(canon.live)
            check_batch(canon, res, min(9, back + 1), rng, st, label=f"rollback {back}+reappend")


def run_negative(rng, st):
    # Deep RESET provides only the newest 128 ORIGINAL-F16 rows.
    lay = km.Layout(ORIGIN, CAPACITY, 128, ring=RING)
    canon = Canon(rng, CAPACITY)
    canon.append(257)
    res = Resident(canon, lay)
    res.reset(257)
    check_batch(canon, res, 1, rng, st, label="deep reset nq1")
    expect_rejected(canon, res, 3, rng, st, "deep reset nq3")
    # Stage-only session: rollback 128 past the seal at 256 leaves group 0's
    # ping-pong slot overwritten by group 2.
    lay0 = km.Layout(ORIGIN, CAPACITY, 0, ring=RING)
    canon = Canon(rng, CAPACITY)
    canon.append(1)
    res = Resident(canon, lay0)
    res.reset(1)
    canon.append(256)
    res.append(257)
    canon.rewind(129)
    res.rewind(129)
    expect_rejected(canon, res, 3, rng, st, "rollback 128 stage slot")
    check_batch(canon, res, 2, rng, st, label="rollback 128 nq2")
    # Masked leading queries below the origin are empty, not uncovered.
    canon = Canon(rng, CAPACITY)
    canon.append(2)
    res = Resident(canon, lay)
    res.reset(2)
    check_batch(canon, res, 5, rng, st, opt_all=True, label="masked below origin")


def run_cap_edge(rng, st):
    lay = km.Layout(ORIGIN, CAPACITY, 128, ring=RING)
    canon = Canon(rng, CAPACITY)
    canon.append(CAPACITY)
    res = Resident(canon, lay)
    res.reset(CAPACITY - 2)
    res.append(CAPACITY)
    check_batch(canon, res, 2, rng, st, label="capacity edge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fewer optimized mode/wave sweeps, no capacity-edge case")
    ap.add_argument("--seed", type=int, default=20261266)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    full = not args.quick
    st = Stats()
    lives = (127, 128, 129, 255, 256, 257)
    nqs = (1, 2, 3, 5, 9)
    for ptt in (0, 128):
        print(f"boundaries ptt={ptt}")
        run_boundaries(rng, ptt, lives, nqs, st, full)
        print(f"decode history ptt={ptt}")
        run_decode_history(rng, ptt, st, full)
    print("negative / masked")
    run_negative(rng, st)
    if full:
        print("capacity edge (origin 72704, capacity 58368)")
        run_cap_edge(rng, st)
    assert st.legacy_worst > 1e-3, "legacy resident image unexpectedly matched serial prefix"
    print(f"rkva_kvarn_serial_prefix_model: ok batches={st.batches} queries={st.queries} "
          f"optimized_runs={st.opt_runs} rejected={st.rejected} "
          f"worst |O-oracle|/mag reference={st.worst_ref:.2e} optimized={st.worst_opt:.2e} "
          f"legacy-image gap={st.legacy_worst:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
