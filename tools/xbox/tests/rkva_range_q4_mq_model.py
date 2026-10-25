#!/usr/bin/env python3
"""Host model of the RKVA Q4_0 multi-query tile kernel (MTP verification).

Emulates shaders/rkva/rkva_range_attn_q4_mq.hlsl (RKVA_MQ_QT = 2 and 4) plus
rkva_range_reduce_opt.hlsl with the real Qwen35 shape (24 Q heads, 4 KV heads,
D=256, original-domain Q4_0 rows of 144 bytes per KV head) and checks every
query's OUT/LSE against an independent float64 oracle that uses that query's
own causal limit. It verifies:

* the 36-byte block-pair dword assembly of the score phase and the quarter-unit
  V loads equal the reference per-element Q4_0 dequant (signed f16 scales) and
  never read outside the (row, KV head) record;
* thread -> row (scores) and thread -> 4 dims (values) ownership for 32-, 64-
  and 128-lane waves (64-thread groups: 2, 1 and one half-filled wave);
* the engine split policy (64-row aligned), the query-tile dispatch, the
  partials layout [split][query*n_head + head][D + 2] bounds/uniqueness against
  the engine's bounded workspace, direct output for split_k == 1;
* per-query causal masking inside one tile: contiguous MTP batches, queries
  before the origin, positions beyond active_count, unsorted positions, partial
  tiles (nq 3/5/9), partial 64-row tiles and the capacity boundary 58368 at the
  real origin 72704.

FP32 summation order is emulated per tile but not bit-exactly per GPU lane;
the gate is the GPU probe tolerance against the float64 oracle. It does not
run on a GPU and makes no performance claim. Run:
    python3 tests/rkva_range_q4_mq_model.py [--quick]
"""
import argparse
import struct
import sys

import numpy as np

N_HEAD, N_HEAD_KV, D = 24, 4, 256
GQA = N_HEAD // N_HEAD_KV
ROW_HEAD = 144
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
    """rkva-range-profile.h q4_mq_split: mode split rows rounded up to 64-row tiles."""
    k0 = optimized_split_k(active, mode)
    rows = (active + k0 - 1) // k0 if k0 else 0
    rows = (rows + TILE - 1) // TILE * TILE
    k = (active + rows - 1) // rows if rows else 1
    return max(1, k), rows


def quarter_dim(u):
    return (u >> 3) * 32 + (u & 1) * 16 + ((u >> 1) & 3) * 4


# ---------------------------------------------------------------- Q4_0 data
def make_q4(rng, rows, scale_lo=0.002, scale_hi=0.05):
    """Raw ggml Q4_0 bytes [rows, kv, 8 blocks x 18] with signed f16 scales."""
    blocks = rows * N_HEAD_KV * 8
    mag = rng.uniform(scale_lo, scale_hi, blocks)
    sign = rng.choice([-1.0, 1.0], blocks)
    scale = (mag * sign).astype(np.float16)
    quant = rng.integers(0, 256, (blocks, 16), dtype=np.uint8)
    raw = np.zeros((blocks, 18), np.uint8)
    raw[:, :2] = scale.view(np.uint8).reshape(blocks, 2)
    raw[:, 2:] = quant
    return raw.reshape(-1).tobytes()


def ref_dequant_record(buf, base):
    """Reference per-element dequant (rkva_range_attn.hlsl STORAGE=1) of one record."""
    out = np.empty(D, np.float32)
    for e in range(D):
        block = base + (e // 32) * 18
        scale = np.frombuffer(buf[block : block + 2], np.float16)[0].astype(np.float32)
        byte = buf[block + 2 + (e & 15)]
        q = byte & 15 if (e & 31) < 16 else byte >> 4
        out[e] = np.float32(q - 8) * scale
    return out


class Words:
    def __init__(self, buf):
        self.words = np.frombuffer(buf + b"\0" * ((-len(buf)) % 4), "<u4")

    def load(self, addr, lo, hi):
        assert addr % 4 == 0 and lo <= addr and addr + 4 <= hi, (addr, lo, hi)
        return int(self.words[addr // 4])


def f16(bits):
    return np.float32(np.frombuffer(struct.pack("<H", bits & 0xFFFF), np.float16)[0])


def mq_score_pair_loads(words, base, pair):
    """Score phase: Load4(addr), Load4(addr+16), Load(addr+32) -> 2 blocks."""
    lo, hi = base, base + ROW_HEAD
    addr = base + pair * 36
    a = [words.load(addr + 4 * i, lo, hi) for i in range(4)]
    b = [words.load(addr + 16 + 4 * i, lo, hi) for i in range(4)]
    w8 = words.load(addr + 32, lo, hi)
    m = 0xFFFFFFFF
    qw0 = [((a[0] >> 16) | (a[1] << 16)) & m, ((a[1] >> 16) | (a[2] << 16)) & m,
           ((a[2] >> 16) | (a[3] << 16)) & m, ((a[3] >> 16) | (b[0] << 16)) & m]
    qw1 = [b[1], b[2], b[3], w8]
    return [(qw0, f16(a[0])), (qw1, f16(b[0] >> 16))]


def nibble_block(qw):
    """32 (q - 8) values in element order from the 4 quant words of one block."""
    out = np.empty(32, np.float32)
    for k, w in enumerate(qw):
        for c in range(4):
            out[4 * k + c] = ((w >> (8 * c)) & 15) - 8
            out[16 + 4 * k + c] = ((w >> (8 * c + 4)) & 15) - 8
    return out


def load_q4_quarter(words, base, u):
    """V phase / per-query kernel load_q4."""
    lo, hi = base, base + ROW_HEAD
    block = base + (u >> 3) * 18
    packed = words.load(block & ~3, lo, hi)
    scale = f16(packed >> ((block & 3) * 8))
    quant = block + 2 + ((u >> 1) & 3) * 4
    aligned = quant & ~3
    shift = (quant & 3) * 8
    w_lo = words.load(aligned, lo, hi)
    w_hi = words.load(aligned + 4 if shift else aligned, lo, hi)
    word = ((w_lo >> shift) | (w_hi << (32 - shift))) & 0xFFFFFFFF if shift else w_lo
    word >>= (u & 1) * 4
    nib = np.array([(word >> (8 * i)) & 15 for i in range(4)], np.float32)
    return (nib - np.float32(8)) * scale


def check_loads(rng):
    buf = make_q4(rng, 6)
    words = Words(buf)
    for r in range(6):
        for h in range(N_HEAD_KV):
            base = (r * N_HEAD_KV + h) * ROW_HEAD
            want = ref_dequant_record(buf, base)
            got = np.empty(D, np.float32)
            for pair in range(4):
                for i, (qw, scale) in enumerate(mq_score_pair_loads(words, base, pair)):
                    blk = pair * 64 + i * 32
                    got[blk : blk + 32] = nibble_block(qw) * scale
            assert np.array_equal(got, want), ("score loads", r, h)
            got[:] = 0
            for u in range(THREADS):
                d = quarter_dim(u)
                got[d : d + 4] = load_q4_quarter(words, base, u)
            assert np.array_equal(got, want), ("value loads", r, h)


def check_ownership():
    dims = sorted(quarter_dim(t) + i for t in range(THREADS) for i in range(4))
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


# ------------------------------------------------------------- dense views
def dense(buf, rows):
    """(q - 8) float32 [rows, kv, 8, 32] and f32 block scales [rows, kv, 8]."""
    raw = np.frombuffer(buf, np.uint8).reshape(rows, N_HEAD_KV, 8, 18)
    scale = raw[..., :2].copy().view(np.float16)[..., 0].astype(np.float32)
    quant = raw[..., 2:]
    q = np.concatenate([quant & 15, quant >> 4], axis=-1).astype(np.float32) - 8
    return q, scale


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
            m = s.max()
            w = np.exp(s - m)
            out[h] = (w @ v) / w.sum()
            lse[h] = m + np.log(w.sum())
    return out, lse


def query_limit(pos, origin, active):
    causal = 0 if pos < origin else pos - origin + 1
    return min(active, causal)


# ------------------------------------------------------------- kernel model
BUGS = ("union_limit", "no_rescale", "nibble_swap", "den_no_rescale")


def run_mq(q, kq, ks, vq, vs, positions, active, origin, kq_scale, mode, wave, bug=None):
    """FP32 emulation of CSRangeAttnQ4Mq + reducer. Returns OUT[nq,24,256], LSE[nq,24].

    `bug` injects a deliberate defect for the negative controls (--self-test)."""
    qt = MQ_TILE_BY_MODE[mode]
    nq = len(positions)
    split_k, split_rows = mq_split(active, mode)
    check_partials_layout(nq, split_k, qt)
    n_waves = (THREADS + wave - 1) // wave
    f32 = np.float32
    scale32 = f32(kq_scale)
    # partials[split][query][head] = (acc[D], max, den)
    p_acc = np.zeros((split_k, nq, N_HEAD, D), f32)
    p_max = np.full((split_k, nq, N_HEAD), -FLT_MAX, f32)
    p_den = np.zeros((split_k, nq, N_HEAD), f32)
    vdeq = (vq * vs[..., None]).reshape(vq.shape[0], N_HEAD_KV, D)  # exact per element
    for split in range(split_k):
        begin = split * split_rows
        split_end = begin + split_rows
        for tile_q in range((nq + qt - 1) // qt):
            q_first = tile_q * qt
            nq_tile = min(qt, nq - q_first)
            qend = [min(query_limit(positions[q_first + i], origin, active), split_end)
                    for i in range(nq_tile)]
            end_max = max(qend)
            for kv in range(N_HEAD_KV):
                hq = nq_tile * GQA
                qs = np.stack([q[q_first + i, kv * GQA + g] for i in range(nq_tile)
                               for g in range(GQA)])  # [hq, D]
                qblk = qs.reshape(hq, 8, 32)
                ends = np.repeat(np.array(qend), GQA)
                if bug == "union_limit":  # one limit for the whole query tile
                    ends[:] = end_max
                mx = np.full(hq, -FLT_MAX, f32)
                dpart = np.zeros((hq, THREADS), f32)
                acc = np.zeros((hq, D), f32)
                for tile in range(begin, end_max, TILE):
                    rows = tile + np.arange(THREADS)
                    clamped = np.minimum(rows, end_max - 1)
                    # scores: per block sum_j q_j (n_j - 8), then * scale, summed over blocks
                    kblk = kq[clamped, kv]
                    if bug == "nibble_swap":  # high/low nibble halves exchanged
                        kblk = np.concatenate([kblk[..., 16:], kblk[..., :16]], axis=-1)
                    blk = np.einsum("hbj,rbj->hrb", qblk, kblk).astype(f32)
                    s = (blk * ks[clamped, kv][None]).astype(f32)
                    s = s.sum(axis=2, dtype=f32)
                    valid = rows[None, :] < ends[:, None]
                    s = np.where(valid, s * scale32, -FLT_MAX).astype(f32)
                    tile_max = s.max(axis=1)  # wave max then cross-wave max: exact
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
                        v4 = vdeq[idx, kv]  # [4, D]
                        acc = (acc + (p[:, j : j + 4] @ v4).astype(f32)).astype(f32)
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
    # rkva_range_reduce_opt.hlsl over splits, per (query, head)
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


def self_test(rng):
    """Each injected defect must be caught by the float64 per-query oracle."""
    origin, kq_scale = 72704, 1.0 / 16.0
    # 17000 rows: every mode-2 split spans more than one 64-row tile.
    active, rel = 17000, [-5, 99, 16999, 30000, 8000]
    kq, ks = dense(make_q4(rng, active), active)
    vq, vs = dense(make_q4(rng, active), active)
    q = rng.normal(0, 3, (len(rel), N_HEAD, D)).astype(np.float32)
    q[:, ::2] *= 4  # heads with very different score ranges
    positions = [origin + r for r in rel]
    oracles = [oracle(q[i].astype(np.float64), kq, ks, vq, vs,
                      query_limit(positions[i], origin, active), kq_scale)
               for i in range(len(rel))]
    for bug in (None,) + BUGS:
        for mode in (1, 2):
            _, _, out, lse = run_mq(q, kq, ks, vq, vs, positions, active, origin, kq_scale,
                                    mode, 64, bug)
            ok = all(compare(out[i], lse[i], *oracles[i])[0] for i in range(len(rel)))
            print(f"self-test bug={bug or 'none':12s} mode={mode}: "
                  f"{'pass' if ok else 'detected'}")
            if ok != (bug is None):
                print("rkva_range_q4_mq_model self-test: FAIL")
                return 1
    print("rkva_range_q4_mq_model self-test: ok (clean passes, every defect detected)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the 58368-row capacity cases")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--self-test", action="store_true",
                    help="only run the negative controls (each injected defect must fail)")
    args = ap.parse_args()
    if args.self_test:
        return self_test(np.random.default_rng(args.seed))
    rng = np.random.default_rng(args.seed)
    check_ownership()
    check_loads(rng)
    print("q4 mq: block-pair score loads + quarter value loads exact vs reference dequant; "
          "ownership ok for 32/64/128-lane waves")

    # Split policy invariants (engine mirrors these in rkva_range_profile_cpu.cpp).
    for active in (0, 1, 63, 64, 65, 455, 4096, 58127, 58128, 58368):
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
        (63, [61, 62], (1, 2), (64,)),
        (64, [62, 63, 64], (1, 2), (32, 64)),
        (65, [62, 63, 64, 65, 66], (1, 2), (64, 128)),
        (129, [128], (1, 2), (64,)),  # kernel capability only: the engine keeps nq1 on its own path
        (129, [126, 127, 128], (1, 2), (32, 64, 128)),
        (257, [254, 255, 256, 257, 258, 259, 260, 261, 262], (1, 2), (32, 64)),
        # independent masking: before origin, mid-range, beyond active, unsorted
        (300, [-5, 99, 299, 1000, 150], (1, 2), (64,)),
        (300, [-3, -2], (1,), (64,)),  # every query masked: LSE -inf, O 0
        (1024, [1021, 1022, 1023], (1, 2), (32, 64)),
        (4099, [4094, 4095, 4096, 4097, 4098], (1, 2), (64,)),
    ]
    if not args.quick:
        cap = 58368  # Q4 body capacity at origin 72704, C131072
        cases += [
            (cap, [cap - 3, cap - 2, cap - 1], (1, 2), (64,)),
            (cap, list(range(cap - 9, cap)), (1, 2), (32, 64)),
            (58128, [58125, 58126, 58127], (1, 2), (64,)),
        ]
    worst_o = worst_l = 0.0
    failures = 0
    for active, rel, modes, waves in cases:
        rows = max(active, 1)
        kq, ks = dense(make_q4(rng, rows), rows)
        vq, vs = dense(make_q4(rng, rows), rows)
        nq = len(rel)
        q = rng.normal(0, 3, (nq, N_HEAD, D)).astype(np.float32)
        positions = [origin + r for r in rel]
        oracles = []
        for i in range(nq):
            limit = query_limit(positions[i], origin, active)
            oracles.append(oracle(q[i].astype(np.float64), kq, ks, vq, vs, limit, kq_scale))
        for mode in modes:
            for wave in waves:
                split_k, split_rows, out, lse = run_mq(q, kq, ks, vq, vs, positions, active,
                                                       origin, kq_scale, mode, wave)
                for i in range(nq):
                    ok, e_o, e_l = compare(out[i], lse[i], *oracles[i])
                    worst_o, worst_l = max(worst_o, e_o), max(worst_l, e_l)
                    failures += 0 if ok else 1
                    if not ok:
                        print(f"FAIL active={active} q{i} pos=+{rel[i]} mode={mode} wave={wave} "
                              f"|O|={e_o:.2e} |LSE|={e_l:.2e}")
                print(f"active={active:5d} nq={nq} rel={rel if nq <= 5 else str(rel[:2]) + '..'} "
                      f"mode={mode} qt={MQ_TILE_BY_MODE[mode]} wave={wave:3d} split_k={split_k:3d} "
                      f"split_rows={split_rows:4d} direct={int(split_k == 1)}")
    print(f"worst |O-oracle64| {worst_o:.3e}, worst |LSE-oracle64| {worst_l:.3e}, "
          f"tolerance {TOL_ABS}+{TOL_REL}|x|")
    if failures:
        print(f"rkva_range_q4_mq_model: FAIL ({failures} query checks)")
        return 1
    print("rkva_range_q4_mq_model: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
