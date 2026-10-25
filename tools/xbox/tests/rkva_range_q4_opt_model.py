#!/usr/bin/env python3
"""Host model of the optimized RKVA Q4_0 decode kernel.

Emulates shaders/rkva/rkva_range_attn_q4_opt.hlsl + rkva_range_reduce_opt.hlsl
with the real Qwen35 shape (24 Q heads, 4 KV heads, D=256, original-domain
Q4_0 rows of 144 bytes per KV head) and checks it against a float64 oracle and
an emulation of the live259 reference kernel. It verifies:

* the quarter-unit lane mapping covers every dimension exactly once for 32-,
  64- and 128-lane waves;
* the dword-assembled Q4_0 loads equal the reference per-element dequant and
  never read outside the (row, KV head) record;
* split coverage, the causal limit, empty splits/waves, LSE=-inf for empty
  ranges, direct output for split_k == 1 and the split-K reducer.

It does not run on a GPU and makes no performance claim. Run:
    python3 tests/rkva_range_q4_opt_model.py [--quick]
"""
import argparse
import struct
import sys

import numpy as np

N_HEAD, N_HEAD_KV, D = 24, 4, 256
GQA = N_HEAD // N_HEAD_KV
ROW_HEAD = 144
THREADS = 128
QUARTERS = 64
FLT_MAX = np.float32(3.402823466e38)
MODES = {1: (128, 128), 2: (32, 256), 3: (512, 64)}  # target rows, max splits


def optimized_split_k(active, mode):
    target, max_splits = MODES[mode]
    return max(1, min(max_splits, 256, (active + target - 1) // target))


def reference_split_k(active):
    return max(1, min(32, (active + 255) // 256))


def quarter_dim(u):
    return (u >> 3) * 32 + (u & 1) * 16 + ((u >> 1) & 3) * 4


def make_q4_rows(rng, rows):
    """Original-domain ggml Q4_0: per (row, kv head) 8 blocks of f16 d + 16 bytes."""
    data = bytearray()
    for _ in range(rows * N_HEAD_KV * 8):
        scale = np.float16(rng.uniform(0.002, 0.05) * rng.choice([-1, 1]))
        data += scale.tobytes()
        data += bytes(rng.integers(0, 256, 16, dtype=np.uint8).tolist())
    return bytes(data)


def ref_dequant(buf, base_element):
    """rkva_range_attn.hlsl load_cache for RKVA_RANGE_STORAGE=1, whole head row."""
    out = np.empty(D, np.float32)
    for e in range(D):
        element = base_element + e
        block = (element // 32) * 18
        scale = np.frombuffer(buf[block : block + 2], np.float16)[0].astype(np.float32)
        byte = buf[block + 2 + (element & 15)]
        q = byte & 15 if (element & 31) < 16 else byte >> 4
        out[e] = np.float32(q - 8) * scale
    return out


class Q4Reader:
    """Opt shader load_q4 with explicit dword loads and bounds tracking."""

    def __init__(self, buf):
        self.buf = buf
        self.words = np.frombuffer(buf + b"\0" * ((-len(buf)) % 4), "<u4")

    def load(self, addr, lo, hi):
        assert addr % 4 == 0 and lo <= addr and addr + 4 <= hi, (addr, lo, hi)
        return int(self.words[addr // 4])

    def load_q4(self, base, u):
        lo, hi = base, base + ROW_HEAD
        block = base + (u >> 3) * 18
        packed = self.load(block & ~3, lo, hi)
        bits = (packed >> ((block & 3) * 8)) & 0xFFFF
        scale = np.float32(np.frombuffer(struct.pack("<H", bits), np.float16)[0])
        quant = block + 2 + ((u >> 1) & 3) * 4
        aligned = quant & ~3
        shift = (quant & 3) * 8
        w_lo = self.load(aligned, lo, hi)
        w_hi = self.load(aligned + 4 if shift else aligned, lo, hi)
        word = ((w_lo >> shift) | (w_hi << (32 - shift))) & 0xFFFFFFFF if shift else w_lo
        word >>= (u & 1) * 4
        nib = np.array([(word >> (8 * i)) & 15 for i in range(4)], np.float32)
        return (nib - np.float32(8)) * scale


def dequant_table(buf, rows):
    """[rows, kv_head, D] float32 via the opt load path, checked against the reference."""
    reader = Q4Reader(buf)
    table = np.zeros((rows, N_HEAD_KV, D), np.float32)
    for r in range(rows):
        for h in range(N_HEAD_KV):
            base = (r * N_HEAD_KV + h) * ROW_HEAD
            for u in range(QUARTERS):
                d = quarter_dim(u)
                table[r, h, d : d + 4] = reader.load_q4(base, u)
    return table


def check_lane_mapping():
    dims = sorted(d + i for u in range(QUARTERS) for i in range(4) for d in [quarter_dim(u)])
    assert dims == list(range(D)), "quarter mapping must cover each dim once"
    for wave in (32, 64, 128):
        covered = []
        for lane in range(wave):
            if lane < QUARTERS:
                covered.append(lane)
            if wave < QUARTERS and lane + wave < QUARTERS:
                covered.append(lane + wave)
        assert sorted(covered) == list(range(QUARTERS)), wave


def oracle(q, k, v, limit, scale):
    """float64 softmax attention over rows [0, limit) for 24 heads."""
    out = np.zeros((N_HEAD, D))
    lse = np.full(N_HEAD, -np.inf)
    if limit == 0:
        return out, lse
    for h in range(N_HEAD):
        kv = h // GQA
        s = (k[:limit, kv].astype(np.float64) @ q[h].astype(np.float64)) * scale
        m = s.max()
        w = np.exp(s - m)
        out[h] = (w @ v[:limit, kv].astype(np.float64)) / w.sum()
        lse[h] = m + np.log(w.sum())
    return out, lse


def online_update(state, scores, vrows):
    """Float32 online-softmax step shared by both emulations (one or two rows)."""
    mx, den, acc = state
    next_max = np.maximum(mx, np.max(scores, axis=0))
    old = np.where(den == 0, np.float32(0), np.exp(mx - next_max)).astype(np.float32)
    acc = acc * old[:, None]
    den = den * old
    for s, vrow in zip(scores, vrows):
        w = np.exp(s - next_max).astype(np.float32)
        acc = acc + w[:, None] * vrow[None, :]
        den = den + w
    return next_max.astype(np.float32), den.astype(np.float32), acc.astype(np.float32)


def reduce_splits(partials):
    """rkva_range_reduce(_opt).hlsl: partials list of (max[H], den[H], acc[H,D])."""
    mx = np.full(N_HEAD, -FLT_MAX, np.float32)
    for m, den, _ in partials:
        mx = np.where(den > 0, np.maximum(mx, m), mx)
    total = np.zeros(N_HEAD, np.float32)
    out = np.zeros((N_HEAD, D), np.float32)
    for m, den, acc in partials:
        w = np.where(den > 0, np.exp(m - mx), np.float32(0)).astype(np.float32)
        total = total + np.where(den > 0, den * w, np.float32(0))
        out = out + np.where((den > 0)[:, None], acc * w[:, None], np.float32(0))
    inv = np.where(total > 0, np.float32(1) / np.where(total > 0, total, 1), np.float32(0))
    with np.errstate(divide="ignore"):
        lse = np.where(total > 0, mx + np.log(np.where(total > 0, total, 1)), -np.inf)
    return (out * inv[:, None]).astype(np.float32), lse.astype(np.float32)


def run_reference_exact(q, k, v, active, limit, scale):
    split_k = reference_split_k(active)
    rows = (active + split_k - 1) // split_k
    kh = np.repeat(np.arange(N_HEAD_KV), GQA)
    partials = []
    for split in range(split_k):
        b, e = min(limit, split * rows), min(limit, (split + 1) * rows)
        mx = np.full(N_HEAD, -FLT_MAX, np.float32)
        den = np.zeros(N_HEAD, np.float32)
        acc = np.zeros((N_HEAD, D), np.float32)
        for r in range(b, e):
            score = (np.einsum("hd,hd->h", q, k[r, kh]) * np.float32(scale)).astype(np.float32)
            nm = np.maximum(mx, score)
            old = np.where(den == 0, np.float32(0), np.exp(mx - nm)).astype(np.float32)
            w = np.exp(score - nm).astype(np.float32)
            acc = (acc * old[:, None] + w[:, None] * v[r, kh]).astype(np.float32)
            den = (den * old + w).astype(np.float32)
            mx = nm
        partials.append((mx, den, acc))
    return reduce_splits(partials)


def run_optimized(q, k, v, active, limit, scale, mode, wave):
    split_k = optimized_split_k(active, mode)
    rows = (active + split_k - 1) // split_k
    n_waves = THREADS // wave
    partials = []
    for split in range(split_k):
        b, e = min(limit, split * rows), min(limit, (split + 1) * rows)
        mx = np.full(N_HEAD, -FLT_MAX, np.float32)
        den = np.zeros(N_HEAD, np.float32)
        acc = np.zeros((N_HEAD, D), np.float32)
        for kv in range(N_HEAD_KV):
            hs = slice(kv * GQA, (kv + 1) * GQA)
            wave_states = []
            for w in range(n_waves):
                st = (np.full(GQA, -FLT_MAX, np.float32), np.zeros(GQA, np.float32),
                      np.zeros((GQA, D), np.float32))
                row = b + w
                while row + n_waves < e:
                    ra, rb = row, row + n_waves
                    sa = (q[hs] @ k[ra, kv] * np.float32(scale)).astype(np.float32)
                    sb = (q[hs] @ k[rb, kv] * np.float32(scale)).astype(np.float32)
                    st = online_update(st, [sa, sb], [v[ra, kv], v[rb, kv]])
                    row += 2 * n_waves
                if row < e:
                    sa = (q[hs] @ k[row, kv] * np.float32(scale)).astype(np.float32)
                    st = online_update(st, [sa], [v[row, kv]])
                wave_states.append(st)
            # Cross-wave merge in groupshared.
            mm = np.full(GQA, -FLT_MAX, np.float32)
            for m, d_, _ in wave_states:
                mm = np.where(d_ > 0, np.maximum(mm, m), mm)
            md = np.zeros(GQA, np.float32)
            ma = np.zeros((GQA, D), np.float32)
            for m, d_, a in wave_states:
                wt = np.where(d_ > 0, np.exp(m - mm), np.float32(0)).astype(np.float32)
                md = md + np.where(d_ > 0, d_ * wt, np.float32(0))
                ma = ma + np.where((d_ > 0)[:, None], a * wt[:, None], np.float32(0))
            mx[hs], den[hs], acc[hs] = mm, md.astype(np.float32), ma.astype(np.float32)
        partials.append((mx, den, acc))
    if split_k == 1:  # direct output
        mx, den, acc = partials[0]
        inv = np.where(den > 0, np.float32(1) / np.where(den > 0, den, 1), np.float32(0))
        with np.errstate(divide="ignore"):
            lse = np.where(den > 0, mx + np.log(np.where(den > 0, den, 1)), -np.inf)
        return split_k, (acc * inv[:, None]).astype(np.float32), lse.astype(np.float32)
    out, lse = reduce_splits(partials)
    return split_k, out, lse


def compare(name, out, lse, ref_out, ref_lse):
    assert np.array_equal(np.isneginf(lse), np.isneginf(ref_lse)), name
    finite = ~np.isneginf(ref_lse)
    err_o = float(np.max(np.abs(out - ref_out))) if out.size else 0.0
    err_l = float(np.max(np.abs(lse[finite] - ref_lse[finite]))) if finite.any() else 0.0
    return err_o, err_l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the 28672-row case")
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    check_lane_mapping()

    # Dequant parity on a few random records.
    buf = make_q4_rows(rng, 3)
    table = dequant_table(buf, 3)
    for r in range(3):
        for h in range(N_HEAD_KV):
            assert np.array_equal(table[r, h], ref_dequant(buf, (r * N_HEAD_KV + h) * D))
    print("lane mapping + q4 dword loads: exact vs reference dequant")

    scale = 1.0 / 16.0
    origin = 102400
    cases = [
        # (active, query position relative to origin, modes, waves)
        (0, 0, (1,), (32,)),
        (1, 0, (1, 2, 3), (32, 64, 128)),
        (64, 63, (1, 2), (32, 64)),
        (257, 256, (1, 2, 3), (32, 64, 128)),
        (257, 99, (1, 2), (32, 64)),  # causal limit inside the range: empty splits
        (300, -5, (1,), (32,)),  # query before origin: everything masked
        (4096, 4095, (1, 2, 3), (32, 64)),
    ]
    if not args.quick:
        cases.append((28672, 28671, (1,), (32,)))
    worst = 0.0
    for active, rel_pos, modes, waves in cases:
        rows = max(active, 1)
        kq = rng.integers(-8, 8, (rows, N_HEAD_KV, D)).astype(np.float32)
        vq = rng.integers(-8, 8, (rows, N_HEAD_KV, D)).astype(np.float32)
        ks = rng.uniform(0.002, 0.05, (rows, N_HEAD_KV, D // 32, 1)).astype(np.float16)
        vs = rng.uniform(0.002, 0.05, (rows, N_HEAD_KV, D // 32, 1)).astype(np.float16)
        k = (kq.reshape(rows, N_HEAD_KV, D // 32, 32) * ks.astype(np.float32)).reshape(
            rows, N_HEAD_KV, D)
        v = (vq.reshape(rows, N_HEAD_KV, D // 32, 32) * vs.astype(np.float32)).reshape(
            rows, N_HEAD_KV, D)
        q = rng.normal(0, 4, (N_HEAD, D)).astype(np.float32)
        pos = origin + rel_pos
        causal = 0 if pos < origin else pos - origin + 1
        limit = min(active, causal)
        o_ref64, l_ref64 = oracle(q, k, v, limit, scale)
        r_out, r_lse = run_reference_exact(q, k, v, active, limit, scale)
        re_o, re_l = compare("reference", r_out, r_lse, o_ref64, l_ref64)
        for mode in modes:
            for wave in waves:
                split_k, out, lse = run_optimized(q, k, v, active, limit, scale, mode, wave)
                e_o, e_l = compare(f"opt{mode}", out, lse, o_ref64, l_ref64)
                d_o, d_l = compare(f"opt{mode}-vs-ref", out, lse, r_out, r_lse)
                tol_o = max(1e-5, 4 * re_o)
                tol_l = max(1e-4, 4 * re_l)
                ok = e_o <= tol_o and e_l <= tol_l
                worst = max(worst, e_o)
                print(f"active={active:5d} limit={limit:5d} mode={mode} wave={wave:3d} "
                      f"split_k={split_k:3d} direct={int(split_k == 1)} "
                      f"|O-oracle|={e_o:.2e} (ref {re_o:.2e}) |LSE-oracle|={e_l:.2e} "
                      f"(ref {re_l:.2e}) |O-ref|={d_o:.2e} {'OK' if ok else 'FAIL'}")
                if not ok:
                    return 1
    print(f"rkva_range_q4_opt_model: ok (worst |O-oracle| {worst:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
