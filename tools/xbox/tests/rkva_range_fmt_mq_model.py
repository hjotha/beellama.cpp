#!/usr/bin/env python3
"""Host model of the RKVA F16 / Q8_0 multi-query tile kernel (267, MTP verification).

Emulates shaders/rkva/rkva_range_attn_fmt_mq.hlsl (RKVA_OPT_STORAGE = 0 / 2,
RKVA_MQ_QT = 2 / 4) plus rkva_range_reduce_opt.hlsl with the real Qwen35 shape
(24 Q heads, 4 KV heads, D=256; F16 rows of 512 bytes and Q8_0 rows of 272
bytes per KV head) and checks every query's OUT/LSE against an independent
float64 oracle over the actual stored bytes with that query's own causal limit.
It verifies:

* byte level: the F16 four-Load4 block reads and the Q8_0 68-byte block-pair
  dword assembly (unaligned f16 scales at byte 0 / 34, quant words shifted by
  16 bits) of the score phase, and the value-phase load_unit of thread u, equal
  the reference per-element decode (f16tof32 / signed int8 x signed f16 scale)
  and never read outside the (row, KV head) record;
* thread -> row (scores) and thread -> 4 linear dims (values) ownership for
  32-, 64- and 128-lane waves (64-thread groups);
* the 64-row split policy shared with Q4_0 multi-query, the query-tile
  dispatch, partials layout [split][query*n_head + head][D + 2] bounds and
  uniqueness against the bounded workspace, direct output for split_k == 1;
* per-query causal masking inside one tile: contiguous MTP batches, queries
  before the origin, positions beyond active_count, unsorted positions, partial
  query tiles (nq 3/5/9), partial 64-row tiles, live 127/128/129/255/256/257,
  58128 and the capacity boundary 58368 at the real origin 72704;
* signed scales, signed F16 values, int8 -128..127 and outlier rows.

`--self-test` injects defects (union mask, off-by-one mask, missing rescale,
missing denominator rescale, swapped Q8_0 pair scales, swapped F16 halves,
1-byte Q8_0 quant misalignment) and requires each one to be detected.

FP32 summation order is emulated per tile but not bit-exactly per GPU lane;
the gate is the GPU probe tolerance against the float64 oracle. It does not
run on a GPU and makes no performance claim. Run:
    python3 tests/rkva_range_fmt_mq_model.py [--quick] [--self-test]
"""
import argparse
import sys

import numpy as np

N_HEAD, N_HEAD_KV, D = 24, 4, 256
GQA = N_HEAD // N_HEAD_KV
ROW_HEAD = {0: 512, 2: 272}
STORAGE_NAME = {0: "f16", 2: "q8_0"}
TILE = 64
THREADS = 64
OPT_MAX_SPLITS = 256
OPT_MAX_QUERIES = 9
FLT_MAX = np.float32(3.402823466e38)
SPLIT_MODES = {1: (128, 128), 2: (32, 256), 3: (512, 64)}  # target rows, max splits
MQ_TILE_BY_MODE = {1: 4, 2: 2}  # mode 3 keeps the per-query optimized kernel
# GPU probe gate (tests/rkva_range_q4_kernel_probe.py compare()).
TOL_ABS, TOL_REL, TOL_LSE_REL = 3e-4, 3e-4, 3e-5


def optimized_split_k(active, mode):
    target, max_splits = SPLIT_MODES[mode]
    return max(1, min(max_splits, OPT_MAX_SPLITS, (active + target - 1) // target))


def mq_split(active, mode):
    """rkva-range-profile.h q4_mq_split (shared by F16/Q8_0 multi-query)."""
    k0 = optimized_split_k(active, mode)
    rows = (active + k0 - 1) // k0 if k0 else 0
    rows = (rows + TILE - 1) // TILE * TILE
    k = (active + rows - 1) // rows if rows else 1
    return max(1, k), rows


# ------------------------------------------------------------------ data
def make_f16(rng, rows, outlier_rows=0.01):
    """Raw F16 bytes [rows, kv, 256 halves]: signed normals plus outlier rows."""
    x = rng.normal(0, 1.0, (rows, N_HEAD_KV, D))
    hot = rng.random(rows) < outlier_rows
    x[hot] *= 8.0
    x[:, :, ::37] *= -1.5  # sign/magnitude structure across dims
    return x.astype(np.float16).tobytes()


def make_q8(rng, rows, outlier_rows=0.01):
    """Raw ggml Q8_0 bytes [rows, kv, 8 blocks x 34] with signed f16 scales."""
    blocks = rows * N_HEAD_KV * 8
    mag = rng.uniform(0.002, 0.03, blocks)
    hot = np.repeat(rng.random(rows) < outlier_rows, N_HEAD_KV * 8)
    mag[hot] *= 8.0
    sign = rng.choice([-1.0, 1.0], blocks)
    scale = (mag * sign).astype(np.float16)
    quant = rng.integers(-128, 128, (blocks, 32), dtype=np.int16).astype(np.int8)
    quant[::97, 0] = -128  # extreme int8 values
    quant[::89, 31] = 127
    raw = np.zeros((blocks, 34), np.uint8)
    raw[:, :2] = scale.view(np.uint8).reshape(blocks, 2)
    raw[:, 2:] = quant.view(np.uint8)
    return raw.reshape(-1).tobytes()


def make(storage, rng, rows):
    return make_f16(rng, rows) if storage == 0 else make_q8(rng, rows)


def dense(storage, buf, rows):
    """Exact decoded element values [rows, kv, 8, 32] and per-block scales [rows, kv, 8].

    F16: values are the halves, scales 1. Q8_0: values are the int8 quants."""
    if storage == 0:
        v = np.frombuffer(buf, np.float16).reshape(rows, N_HEAD_KV, 8, 32).astype(np.float32)
        return v, np.ones((rows, N_HEAD_KV, 8), np.float32)
    raw = np.frombuffer(buf, np.uint8).reshape(rows, N_HEAD_KV, 8, 34)
    scale = raw[..., :2].copy().view(np.float16)[..., 0].astype(np.float32)
    quant = raw[..., 2:].copy().view(np.int8).astype(np.float32)
    return quant, scale


def ref_decode_record(storage, buf, base):
    """Reference per-element decode (rkva_range_attn.hlsl STORAGE=0/2) of one record."""
    if storage == 0:
        return np.frombuffer(buf[base : base + 512], np.float16).astype(np.float32)
    out = np.empty(D, np.float32)
    for e in range(D):
        block = base + (e // 32) * 34
        scale = np.frombuffer(buf[block : block + 2], np.float16)[0].astype(np.float32)
        q = np.frombuffer(buf[block + 2 + (e & 31) : block + 3 + (e & 31)], np.int8)[0]
        out[e] = np.float32(q) * scale
    return out


# --------------------------------------------------------- byte-level loads
class Words:
    def __init__(self, buf):
        self.words = np.frombuffer(buf + b"\0" * ((-len(buf)) % 4), "<u4")

    def load(self, addr, lo, hi):
        assert addr % 4 == 0 and lo <= addr and addr + 4 <= hi, (addr, lo, hi)
        return int(self.words[addr // 4])


def f16(bits):
    return np.float32(np.array([bits & 0xFFFF], np.uint16).view(np.float16)[0])


def int8x4(w):
    return [np.float32(((w >> (8 * i)) & 0xFF) - (256 if (w >> (8 * i)) & 0x80 else 0))
            for i in range(4)]


def halfx4(w0, w1):
    return [f16(w0), f16(w0 >> 16), f16(w1), f16(w1 >> 16)]


def score_record_loads(storage, words, base, bug=None):
    """Score-phase decode of one record: (elements[256] before scale, scale per block)."""
    lo, hi = base, base + ROW_HEAD[storage]
    elems = np.empty(D, np.float32)
    scales = np.ones(8, np.float32)
    m = 0xFFFFFFFF
    if storage == 0:
        for blk in range(8):
            addr = base + blk * 64
            w = [words.load(addr + 4 * i, lo, hi) for i in range(16)]
            vals = []
            for k in range(8):
                a, b = w[2 * k], w[2 * k + 1]
                if bug == "f16_half_swap":
                    a, b = ((a >> 16) | (a << 16)) & m, ((b >> 16) | (b << 16)) & m
                vals += halfx4(a, b)
            elems[blk * 32 : blk * 32 + 32] = vals
        return elems, scales
    for pair in range(4):
        addr = base + pair * 68
        w = [words.load(addr + 4 * i, lo, hi) for i in range(9)]  # Load4, Load4, Load
        w += [words.load(addr + 36 + 4 * i, lo, hi) for i in range(8)]  # Load4, Load4
        shift = 8 if bug == "q8_misalign" else 16
        q0 = [((w[j] >> shift) | (w[j + 1] << (32 - shift))) & m for j in range(8)]
        q1 = w[9:17]
        s0, s1 = f16(w[0]), f16(w[8] >> 16)
        if bug == "q8_scale_swap":
            s0, s1 = s1, s0
        e = []
        for qw in q0 + q1:
            e += int8x4(qw)
        elems[pair * 64 : pair * 64 + 64] = e
        scales[2 * pair], scales[2 * pair + 1] = s0, s1
    return elems, scales


def load_unit(storage, words, base, u):
    """Value phase: rkva_range_attn_fmt_mq.hlsl / gqa_opt load_unit (dims 4u..4u+3)."""
    lo, hi = base, base + ROW_HEAD[storage]
    if storage == 0:
        a = words.load(base + u * 8, lo, hi)
        b = words.load(base + u * 8 + 4, lo, hi)
        return np.array(halfx4(a, b), np.float32)
    block = base + (u >> 3) * 34
    packed = words.load(block & ~3, lo, hi)
    scale = f16(packed >> ((block & 3) * 8))
    quant = block + 2 + (u & 7) * 4
    aligned = quant & ~3
    shift = (quant & 3) * 8
    w_lo = words.load(aligned, lo, hi)
    w_hi = words.load(aligned + 4 if shift else aligned, lo, hi)
    word = ((w_lo >> shift) | (w_hi << (32 - shift))) & 0xFFFFFFFF if shift else w_lo
    return np.array(int8x4(word), np.float32) * scale


def check_loads(storage, rng, bug=None):
    """True when every score/value load of 6 rows equals the reference decode."""
    buf = make(storage, rng, 6)
    words = Words(buf)
    for r in range(6):
        for h in range(N_HEAD_KV):
            base = (r * N_HEAD_KV + h) * ROW_HEAD[storage]
            want = ref_decode_record(storage, buf, base)
            elems, scales = score_record_loads(storage, words, base, bug)
            got = (elems.reshape(8, 32) * scales[:, None]).reshape(D)
            if not np.array_equal(got, want):
                return False
            got = np.empty(D, np.float32)
            for u in range(THREADS):
                got[4 * u : 4 * u + 4] = load_unit(storage, words, base, u)
            if not np.array_equal(got, want):
                return False
    return True


def check_ownership():
    dims = sorted(4 * t + i for t in range(THREADS) for i in range(4))
    assert dims == list(range(D)), "values: each dim owned once"
    for wave in (32, 64, 128):
        n_waves = (THREADS + wave - 1) // wave
        lanes = [t for w in range(n_waves) for t in range(w * wave, min(THREADS, (w + 1) * wave))]
        assert sorted(lanes) == list(range(THREADS)), wave
        assert n_waves <= 2, "gs_red holds RKVA_MAX_WAVES = 2 waves"


def check_partials_layout(n_query, split_k, qt):
    total = n_query * N_HEAD
    seen = set()
    limit = OPT_MAX_SPLITS * OPT_MAX_QUERIES * N_HEAD * (D + 2) * 4
    for split in range(split_k):
        for tile in range((n_query + qt - 1) // qt):
            for qq in range(min(qt, n_query - tile * qt)):
                for kv in range(N_HEAD_KV):
                    for g in range(GQA):
                        qh = (tile * qt + qq) * N_HEAD + kv * GQA + g
                        dst = (split * total + qh) * (D + 2) * 4
                        assert dst + (D + 2) * 4 <= limit
                        assert dst not in seen
                        seen.add(dst)
    assert len(seen) == split_k * total


# ------------------------------------------------------------------ oracle
def oracle(q64, kq, ks, vq, vs, limit, kq_scale):
    """float64 attention for one query over rows [0, limit) (independent of tiling)."""
    out = np.zeros((N_HEAD, D))
    lse = np.full(N_HEAD, -np.inf)
    if limit <= 0:
        return out, lse
    for kv in range(N_HEAD_KV):
        k = (kq[:limit, kv].astype(np.float64) * ks[:limit, kv, :, None]).reshape(limit, D)
        v = (vq[:limit, kv].astype(np.float64) * vs[:limit, kv, :, None]).reshape(limit, D)
        for g in range(GQA):
            h = kv * GQA + g
            s = k @ q64[h] * kq_scale
            mx = s.max()
            w = np.exp(s - mx)
            out[h] = (w @ v) / w.sum()
            lse[h] = mx + np.log(w.sum())
    return out, lse


def query_limit(pos, origin, active):
    causal = 0 if pos < origin else pos - origin + 1
    return min(active, causal)


# ------------------------------------------------------------- kernel model
BUGS = ("union_limit", "off_by_one", "no_rescale", "den_no_rescale", "q8_scale_swap",
        "f16_half_swap")


def run_mq(storage, q, kq, ks, vq, vs, positions, active, origin, kq_scale, mode, wave, bug=None):
    """FP32 emulation of CSRangeAttnFmtMq + reducer. Returns OUT[nq,24,256], LSE[nq,24]."""
    qt = MQ_TILE_BY_MODE[mode]
    nq = len(positions)
    split_k, split_rows = mq_split(active, mode)
    check_partials_layout(nq, split_k, qt)
    n_waves = (THREADS + wave - 1) // wave
    f32 = np.float32
    scale32 = f32(kq_scale)
    p_acc = np.zeros((split_k, nq, N_HEAD, D), f32)
    p_max = np.full((split_k, nq, N_HEAD), -FLT_MAX, f32)
    p_den = np.zeros((split_k, nq, N_HEAD), f32)
    vdeq = (vq * vs[..., None]).reshape(vq.shape[0], N_HEAD_KV, D)  # exact per element
    k_scales = ks
    if bug == "q8_scale_swap" and storage == 2:
        k_scales = ks[..., [1, 0, 3, 2, 5, 4, 7, 6]]
    k_elems = kq
    if bug == "f16_half_swap" and storage == 0:
        k_elems = kq.reshape(*kq.shape[:2], 128, 2)[..., ::-1].reshape(kq.shape)
    for split in range(split_k):
        begin = split * split_rows
        split_end = begin + split_rows
        for tile_q in range((nq + qt - 1) // qt):
            q_first = tile_q * qt
            nq_tile = min(qt, nq - q_first)
            qend = [min(query_limit(positions[q_first + i], origin, active), split_end)
                    for i in range(nq_tile)]
            if bug == "off_by_one":
                qend = [min(e + 1, active, split_end) if e > 0 else e for e in qend]
            end_max = max(qend)
            for kv in range(N_HEAD_KV):
                hq = nq_tile * GQA
                qs = np.stack([q[q_first + i, kv * GQA + g] for i in range(nq_tile)
                               for g in range(GQA)])
                qblk = qs.reshape(hq, 8, 32)
                ends = np.repeat(np.array(qend), GQA)
                if bug == "union_limit":
                    ends[:] = end_max
                mx = np.full(hq, -FLT_MAX, f32)
                dpart = np.zeros((hq, THREADS), f32)
                acc = np.zeros((hq, D), f32)
                for tile in range(begin, end_max, TILE):
                    rows = tile + np.arange(THREADS)
                    clamped = np.minimum(rows, end_max - 1)
                    blk = np.einsum("hbj,rbj->hrb", qblk, k_elems[clamped, kv]).astype(f32)
                    if storage == 2:  # Q8_0: block dot * scale; F16 sums the block dots
                        blk = (blk * k_scales[clamped, kv][None]).astype(f32)
                    s = blk.sum(axis=2, dtype=f32)
                    valid = rows[None, :] < ends[:, None]
                    s = np.where(valid, s * scale32, -FLT_MAX).astype(f32)
                    tile_max = s.max(axis=1)
                    nxt = np.maximum(mx, tile_max)
                    alpha = np.exp(mx - nxt).astype(f32)
                    if bug == "no_rescale":
                        alpha = np.ones_like(alpha)
                    with np.errstate(over="ignore"):
                        p = np.where(valid, np.exp(s - nxt[:, None]), f32(0)).astype(f32)
                    d_alpha = np.ones_like(alpha) if bug == "den_no_rescale" else alpha
                    dpart = (dpart * d_alpha[:, None] + p).astype(f32)
                    acc = (acc * alpha[:, None]).astype(f32)
                    n_rows = min(TILE, end_max - tile)
                    for j in range(0, n_rows, 4):
                        idx = np.minimum(tile + j + np.arange(4), end_max - 1)
                        acc = (acc + (p[:, j : j + 4] @ vdeq[idx, kv]).astype(f32)).astype(f32)
                    mx = nxt
                den = np.zeros(hq, f32)
                for w in range(n_waves):
                    lanes = slice(w * wave, min(THREADS, (w + 1) * wave))
                    den = (den + dpart[:, lanes].sum(axis=1, dtype=f32)).astype(f32)
                for i in range(nq_tile):
                    hs = slice(kv * GQA, (kv + 1) * GQA)
                    sl = slice(i * GQA, (i + 1) * GQA)
                    p_acc[split, q_first + i, hs] = acc[sl]
                    p_max[split, q_first + i, hs] = np.where(den[sl] > 0, mx[sl], -FLT_MAX)
                    p_den[split, q_first + i, hs] = den[sl]
    if split_k == 1:  # direct output
        den, mx, acc = p_den[0], p_max[0], p_acc[0]
        inv = np.where(den > 0, f32(1) / np.where(den > 0, den, 1), f32(0)).astype(f32)
        with np.errstate(divide="ignore"):
            lse = np.where(den > 0, mx + np.log(np.where(den > 0, den, 1)), -np.inf)
        return split_k, split_rows, (acc * inv[..., None]).astype(f32), lse.astype(f32)
    mx = np.where(p_den > 0, p_max, -FLT_MAX).max(axis=0)
    w = np.where(p_den > 0, np.exp(p_max - mx[None]), f32(0)).astype(f32)
    total = (p_den * w).sum(axis=0, dtype=f32)
    out = (p_acc * w[..., None]).sum(axis=0, dtype=f32)
    inv = np.where(total > 0, f32(1) / np.where(total > 0, total, 1), f32(0))
    with np.errstate(divide="ignore"):
        lse = np.where(total > 0, mx + np.log(np.where(total > 0, total, 1)), -np.inf)
    return split_k, split_rows, (out * inv[..., None]).astype(f32), lse.astype(f32)


def compare(out, lse, ref_out, ref_lse):
    if not np.array_equal(np.isneginf(lse), np.isneginf(ref_lse)):
        return False, float("inf"), float("inf")
    if not np.all(np.isfinite(out)):
        return False, float("inf"), float("inf")
    finite = ~np.isneginf(ref_lse)
    err_o = float(np.max(np.abs(out - ref_out))) if out.size else 0.0
    err_l = float(np.max(np.abs(lse[finite] - ref_lse[finite]))) if finite.any() else 0.0
    ok = bool(np.all(np.abs(out - ref_out) <= TOL_ABS + TOL_REL * np.abs(ref_out)) and
              np.all(np.abs(lse[finite] - ref_lse[finite]) <=
                     TOL_ABS + TOL_LSE_REL * np.abs(ref_lse[finite])))
    return ok, err_o, err_l


def make_queries(rng, nq):
    q = rng.normal(0, 1.0, (nq, N_HEAD, D)).astype(np.float32)
    q[:, ::2] *= 3  # heads with very different score ranges
    return q


def self_test(rng):
    """Every injected defect must be detected, clean runs must pass."""
    failed = False
    for storage in (0, 2):
        clean = check_loads(storage, rng)
        bad = [b for b in ("q8_misalign", "q8_scale_swap") if storage == 2] + \
              (["f16_half_swap"] if storage == 0 else [])
        caught = [not check_loads(storage, rng, b) for b in bad]
        print(f"self-test loads {STORAGE_NAME[storage]}: clean={'pass' if clean else 'FAIL'} "
              f"defects {dict(zip(bad, ['detected' if c else 'MISSED' for c in caught]))}")
        failed |= not clean or not all(caught)
    origin, kq_scale = 72704, 1.0 / 16.0
    # 17000 rows: every mode-2 split spans more than one 64-row tile. Query +99
    # sits inside a split, so an off-by-one mask leaks row 100 into it.
    active, rel = 17000, [-5, 99, 16999, 30000, 8000]
    q = make_queries(rng, len(rel))
    positions = [origin + r for r in rel]
    for storage in (0, 2):
        kq, ks = dense(storage, make(storage, rng, active), active)
        vq, vs = dense(storage, make(storage, rng, active), active)
        oracles = [oracle(q[i].astype(np.float64), kq, ks, vq, vs,
                          query_limit(positions[i], origin, active), kq_scale)
                   for i in range(len(rel))]
        bugs = [b for b in BUGS if not (b == "q8_scale_swap" and storage == 0) and
                not (b == "f16_half_swap" and storage == 2)]
        for bug in [None] + bugs:
            for mode in (1, 2):
                _, _, out, lse = run_mq(storage, q, kq, ks, vq, vs, positions, active, origin,
                                        kq_scale, mode, 64, bug)
                ok = all(compare(out[i], lse[i], *oracles[i])[0] for i in range(len(rel)))
                print(f"self-test {STORAGE_NAME[storage]:4s} bug={bug or 'none':14s} "
                      f"mode={mode}: {'pass' if ok else 'detected'}")
                failed |= ok != (bug is None)
    if failed:
        print("rkva_range_fmt_mq_model self-test: FAIL")
        return 1
    print("rkva_range_fmt_mq_model self-test: ok (clean passes, every defect detected)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the 58128/58368-row cases")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--storage", choices=("f16", "q8_0", "both"), default="both")
    ap.add_argument("--self-test", action="store_true",
                    help="only run the negative controls (each injected defect must fail)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.self_test:
        return self_test(rng)
    storages = {"f16": (0,), "q8_0": (2,), "both": (0, 2)}[args.storage]
    check_ownership()
    for storage in storages:
        assert check_loads(storage, rng), STORAGE_NAME[storage]
    print("fmt mq: score loads + value load_unit exact vs reference decode for "
          f"{[STORAGE_NAME[s] for s in storages]}; ownership ok for 32/64/128-lane waves")

    for active in (0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 4096, 58127, 58128, 58368):
        for mode in (1, 2):
            k, rows = mq_split(active, mode)
            assert 1 <= k <= OPT_MAX_SPLITS and rows % TILE == 0, (active, mode)
            assert k * rows >= active and (k - 1) * rows < max(active, 1), (active, mode)
            assert k <= optimized_split_k(active, mode)

    origin = 72704
    kq_scale = 1.0 / 16.0
    # (active, positions relative to origin, modes, waves)
    cases = [
        (0, [0, 1], (1, 2), (64,)),
        (1, [0, 1, 2], (1, 2), (32, 64, 128)),
        (64, [62, 63, 64], (1, 2), (32, 64)),
        (127, [124, 125, 126], (1, 2), (64,)),
        (128, [126, 127], (1, 2), (32, 64, 128)),
        (129, [124, 125, 126, 127, 128], (1, 2), (64,)),
        (255, [252, 253, 254], (1, 2), (64,)),
        (256, [254, 255], (1, 2), (64, 128)),
        (257, [248, 249, 250, 251, 252, 253, 254, 255, 256], (1, 2), (32, 64)),
        # independent masking: before origin, mid-range, beyond active, unsorted
        (300, [-5, 99, 299, 1000, 150], (1, 2), (64,)),
        (300, [-3, -2], (1,), (64,)),  # every query masked: LSE -inf, O 0
        (300, [299, 150, 7], (2,), (32,)),  # descending positions
        (4099, [4094, 4095, 4096, 4097, 4098], (1, 2), (64,)),
    ]
    if not args.quick:
        cap = 58368  # body capacity at origin 72704, C131072
        cases += [
            (cap, [cap - 3, cap - 2, cap - 1], (1, 2), (64,)),
            (cap, list(range(cap - 9, cap)), (1, 2), (32, 64)),
            (58128, [58125, 58126, 58127], (1, 2), (64,)),
        ]
    worst_o = worst_l = 0.0
    failures = 0
    for storage in storages:
        for active, rel, modes, waves in cases:
            rows = max(active, 1)
            kq, ks = dense(storage, make(storage, rng, rows), rows)
            vq, vs = dense(storage, make(storage, rng, rows), rows)
            nq = len(rel)
            q = make_queries(rng, nq)
            positions = [origin + r for r in rel]
            oracles = [oracle(q[i].astype(np.float64), kq, ks, vq, vs,
                              query_limit(positions[i], origin, active), kq_scale)
                       for i in range(nq)]
            for mode in modes:
                for wave in waves:
                    split_k, split_rows, out, lse = run_mq(storage, q, kq, ks, vq, vs, positions,
                                                           active, origin, kq_scale, mode, wave)
                    for i in range(nq):
                        ok, e_o, e_l = compare(out[i], lse[i], *oracles[i])
                        worst_o, worst_l = max(worst_o, e_o), max(worst_l, e_l)
                        failures += 0 if ok else 1
                        if not ok:
                            print(f"FAIL {STORAGE_NAME[storage]} active={active} q{i} "
                                  f"pos=+{rel[i]} mode={mode} wave={wave} "
                                  f"|O|={e_o:.2e} |LSE|={e_l:.2e}")
                    print(f"{STORAGE_NAME[storage]:4s} active={active:5d} nq={nq} "
                          f"rel={rel if nq <= 5 else str(rel[:2]) + '..'} mode={mode} "
                          f"qt={MQ_TILE_BY_MODE[mode]} wave={wave:3d} split_k={split_k:3d} "
                          f"split_rows={split_rows:4d} direct={int(split_k == 1)}")
    print(f"worst |O-oracle64| {worst_o:.3e}, worst |LSE-oracle64| {worst_l:.3e}, "
          f"tolerance {TOL_ABS}+{TOL_REL}|x|")
    if failures:
        print(f"rkva_range_fmt_mq_model: FAIL ({failures} query checks)")
        return 1
    print("rkva_range_fmt_mq_model: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
