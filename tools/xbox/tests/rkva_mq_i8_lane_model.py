#!/usr/bin/env python3
"""Lane-by-lane NumPy execution of the shipped `attn_kvarn4_mq2_i8` HLSL block.

`rkva_mq_i8_quality.py` validated the *algebra*: it evaluates every score as one
128-dim slice sum, with the per-slice constant added once. The GPU runs a
different program, and this module executes that program instead:

  * 128 threads as two waves of 64 lanes (`lane`, `wave = tid / 64`);
  * the 16-dim / 16-lane mapping (`slot = lane & 15`, `row = lane / 16`);
  * the per-group int8 query quantizer with wave-wide slice maxima;
  * `dot4add_i8packed` as signed-int8 x signed-int8, four bytes, int32 wrap;
  * inclusive F32 prefixes and constant endpoints 15/31/47/63, whose
    differences hand the four row scores to every lane;
  * the per-lane application of `q8scale`/`q8const`/`other`, which is where the
    shipped kernel and the fix differ.

Constant modes (the only difference between them is how the per-slice constant
`C_s = Z_s + 8*sum(Q') - 8*sq_s*sum(Q8)` reaches the score sum):

  per_lane     FIX. Every lane contributes its own partial
                     zs + 8*(qsum - sq*sum(Q8))
               so summing the 16 lanes yields `C_s` exactly once per slice.
  slice        SHIPPED, WRONG. `zs`, `qsum` and `sum(Q8)` are first reduced over
               the 8 lanes of the slice, so every lane holds the whole slice
               total and the 16-lane sum yields `8*C_0 + 8*C_1`: seven extra
               constants per slice.
  slice_split  the shipped form with the reduced constant divided by 8; must
               agree with `per_lane` up to float rounding, which cross-checks
               the diagnosis.

Run: python3 tools/xbox/tests/rkva_mq_i8_lane_model.py [--json out.json] [--quick]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rkva_mq_i8_quality as Q  # noqa: E402

I8_DIMS = 16
I8_LANES = 16
I8_SLICE_LANES = 8
I8_MAXV = 126.0
WAVE = 64
N_WAVES = 2
KQ = Q.KQ_SCALE
F32 = np.float32
INT32 = 2 ** 31


# ------------------------------------------------------- wave/lane primitives ---


def int32(x):
    """Wrap to signed int32, as every int add/dp4a in the kernel does."""
    return ((np.asarray(x, np.int64) + INT32) % (2 * INT32) - INT32)


def dot4add_i8packed(a, b, c):
    """Signed int8 x signed int8 over four bytes, accumulated into int32 `c`."""
    total = np.asarray(c, np.int64).copy()
    for k in range(4):
        av = (a >> (8 * k)) & 0xFF
        av = np.where(av >= 128, av - 256, av)
        bv = (b >> (8 * k)) & 0xFF
        bv = np.where(bv >= 128, bv - 256, bv)
        total = int32(total + av * bv)
    return total


def reduce_add(v, masks):
    """Grouped reference sum for the slice-constant negative control."""
    width = max(masks) * 2
    groups = np.asarray(v, np.float64).reshape(-1, width)
    return np.repeat(groups.sum(axis=1), width)


def reduce_add_int(v, masks):
    return int32(reduce_add(v, masks))


def pack4(values):
    """Four int8 values into one dword, low byte first (two's complement)."""
    word = 0
    for i, value in enumerate(values):
        word |= (int(value) & 0xFF) << (8 * i)
    return word & 0xFFFFFFFF


# ------------------------------------------------------------------- kernel ---


def fast_bounds(live):
    """fast_lo/fast_hi for one mq2 segment, as the host and the shader compute them."""
    tail_begin = live - min(live, Q.TAIL)
    shared_hi = min(live & ~127, tail_begin)
    return 0, (shared_hi // 8) * 8


def shader_scores(store, qrot, live, const_mode):
    """Execute the shipped fast shared record tile.

    `qrot` is [n_heads, 256] and holds `wht256x6`'s rotated query for the heads
    being modelled, exactly the values the host stages at `q_stride`. Returns
    {head: [score per row]}, NaN outside the fast range.
    """
    fast_lo, fast_hi = fast_bounds(live)
    n_heads = qrot.shape[0]
    out = {g: np.full(fast_hi, np.nan) for g in range(n_heads)}
    # Structural guard: the group-outer tile walk must visit every fast row
    # exactly once across the two waves (a double visit would silently double a
    # row's softmax contribution).
    hits = np.zeros(fast_hi, np.int64)
    if fast_hi <= fast_lo:
        return out, (fast_lo, fast_hi)

    lane = np.arange(WAVE)
    slot = lane & (I8_LANES - 1)
    row = lane // I8_LANES
    dim = slot * I8_DIMS
    sl = dim >> 7
    dloc = dim & 127
    ar = np.arange(I8_DIMS)

    for wave in range(N_WAVES):
        for grp in range(fast_lo >> 7, ((fast_hi - 1) >> 7) + 1):
            planes = store.kplanes[grp]
            # Absolute-dim views of both record slices (dims 0..255).
            sc256 = np.concatenate([planes[0][1], planes[1][1]])
            ze256 = np.concatenate([planes[0][2], planes[1][2]])
            nib = np.stack([planes[0][0], planes[1][0]])       # [slice, dim, pos]
            oth = np.stack([planes[0][3], planes[1][3]])       # [slice, pos]
            ksv = sc256[dim[:, None] + ar[None, :]]
            kzv = ze256[dim[:, None] + ar[None, :]]

            # ---- per-group int8 query quantizer ---------------------------
            q8 = np.zeros((n_heads, 4, WAVE), np.uint32)
            sq_lane = np.zeros((n_heads, WAVE), np.float64)
            c_lane = np.zeros((n_heads, WAVE), np.float64)
            for g in range(n_heads):
                qo = qrot[g][dim[:, None] + ar[None, :]]
                qp = qo * ksv
                partial_max = np.abs(qp).max(axis=1)
                mx0 = np.max(np.where(sl == 0, partial_max, 0.0))
                mx1 = np.max(np.where(sl == 1, partial_max, 0.0))
                mx = np.where(sl == 0, mx0, mx1)
                zs = (qo * kzv).sum(axis=1)
                qsum = qp.sum(axis=1)
                inv = np.where(mx > 0, I8_MAXV / np.where(mx > 0, mx, 1.0), 0.0)
                sq = np.where(mx > 0, mx * (1.0 / I8_MAXV), 0.0)
                qint = np.clip(np.rint(qp * inv[:, None]), -127, 127).astype(np.int64)
                s8 = qint.sum(axis=1)
                if const_mode == 'slice':
                    zs = reduce_add(zs, [4, 2, 1])
                    qsum = reduce_add(qsum, [4, 2, 1])
                    s8 = reduce_add_int(s8, [4, 2, 1]).astype(np.float64)
                elif const_mode == 'slice_split':
                    # same as the shipped form, but each lane carries C_s / 8
                    zs = reduce_add(zs, [4, 2, 1]) * 0.125
                    qsum = reduce_add(qsum, [4, 2, 1]) * 0.125
                    s8 = reduce_add_int(s8, [4, 2, 1]).astype(np.float64) * 0.125
                elif const_mode != 'per_lane':
                    raise SystemExit('unknown const_mode: ' + const_mode)
                sq_lane[g] = sq
                c_lane[g] = zs + 8.0 * (qsum - sq * s8)
                for c in range(4):
                    for ln in range(WAVE):
                        q8[g, c, ln] = pack4(qint[ln, 4 * c:4 * c + 4])

            # ---- 8-row tile loop ------------------------------------------
            tile_lo = max(fast_lo, grp << 7) + wave * 8
            tile_hi = min(fast_hi, (grp + 1) << 7)
            for t0 in range(tile_lo, tile_hi, N_WAVES * 8):
                pos0 = t0 & 127
                # One dword per lane per dim: 8 nibbles of dim (dloc+i) at
                # positions pos0..pos0+7, from the lane's own slice -- exactly
                # `keys.Load(nib_base + i * 64)` with a per-lane address.
                pos_idx = pos0 + np.arange(8)
                shifts4 = (4 * np.arange(8)).astype(np.int64)
                kd = []
                for i in range(I8_DIMS):
                    n8 = nib[sl, dloc + i][:, pos_idx].astype(np.int64)
                    kd.append(((n8 << shifts4[None, :]).sum(axis=1) & 0xFFFFFFFF))
                for h in range(2):
                    tt = h * 4 + row                       # per-lane row of the half
                    sh = 4 * tt
                    k8 = np.zeros((4, WAVE), np.uint32)
                    for c in range(4):
                        w0 = (kd[4 * c + 0] >> sh) & 0xF
                        w1 = (kd[4 * c + 1] >> sh) & 0xF
                        w2 = (kd[4 * c + 2] >> sh) & 0xF
                        w3 = (kd[4 * c + 3] >> sh) & 0xF
                        k8[c] = w0 | (w1 << 8) | (w2 << 16) | (w3 << 24)
                    sll = KQ * oth[sl, pos0 + tt]
                    for g in range(n_heads):
                        acc = np.zeros(WAVE, np.int64)
                        for c in range(4):
                            acc = dot4add_i8packed(k8[c], q8[g, c], acc)
                        val = (acc.astype(np.float64) * sq_lane[g] + c_lane[g]) * sll
                        # One legal F32 prefix order; DXIL VM separately checks
                        # the emitted program. Hardware may use a different tree.
                        prefix = np.cumsum(val.astype(F32), dtype=F32)
                        endpoints = prefix[[15, 31, 47, 63]]
                        scores = np.diff(np.concatenate([np.zeros(1, F32), endpoints]))
                        for t in range(4):
                            s_t = float(scores[t])
                            r = t0 + h * 4 + t
                            if r < fast_hi:
                                out[g][r] = s_t
                                if g == 0:
                                    hits[r] += 1
    if fast_hi > fast_lo:
        assert hits[fast_lo:].min() == 1 and hits[fast_lo:].max() == 1, \
            'fast rows not visited exactly once: %s' % np.unique(hits[fast_lo:])
    return out, (fast_lo, fast_hi)


# ------------------------------------------------------------------ pipelines ---


def store_image(store):
    """Rotated-domain K/V of every row, plus the original-F16 tail, float64."""
    n = store.tail_begin
    k = np.zeros((store.live, Q.D), np.float64)
    v = np.zeros((store.live, Q.D), np.float64)
    for g in range(store.groups):
        for s in (0, 1):
            qq, ss, zz, oo = Q.record_planes(store.blobs[g][0][s])
            k[g * 128:(g + 1) * 128, s * 128:(s + 1) * 128] = \
                Q.decode_plane(qq, ss, zz, oo).T.astype(np.float64)
            qq, ss, zz, oo = Q.record_planes(store.blobs[g][1][s])
            v[g * 128:(g + 1) * 128, s * 128:(s + 1) * 128] = \
                Q.decode_plane(qq, ss, zz, oo).astype(np.float64)
    k[n:] = Q.butterfly_wht(Q.f16(store.k[n:])).astype(np.float64)
    v[n:] = Q.butterfly_wht(Q.f16(store.v[n:])).astype(np.float64)
    return k, v


def oracle_attention(k_img, v_img, qrot, live):
    """Float64 attention over the storage image; the probe's `want`."""
    s = (k_img[:live] @ qrot.astype(np.float64)) * KQ
    mx = s.max()
    w = np.exp(s - mx)
    body = (w @ v_img[:live]) / w.sum()
    return (Q.butterfly_wht(body.astype(F32)).astype(np.float64),
            float(mx + np.log(w.sum())))


def run_case(live, style, seed, const_modes, n_queries=1):
    rng = np.random.default_rng(seed)
    store = Q.Store(rng, live, style)
    k_img, v_img = store_image(store)
    krec, vrec = store.sources()
    k_tail = Q.f16(store.k[store.tail_begin:])
    v_tail = Q.f16(store.v[store.tail_begin:])
    fast_lo, fast_hi = fast_bounds(live)
    results = {m: [] for m in const_modes}
    bias = {m: [] for m in const_modes}
    n_heads = 2
    for _ in range(n_queries):
        q_orig = [rng.normal(size=Q.D).astype(F32) for _ in range(n_heads)]
        qrot = np.stack([Q.butterfly_wht(q) for q in q_orig])
        exact = [Q.ref_scores(qrot[g], krec) for g in range(n_heads)]
        want = [oracle_attention(k_img, v_img, qrot[g], live) for g in range(n_heads)]
        for mode in const_modes:
            if mode == 'algebra':
                # validated 128-dim slice algebra (rkva_mq_i8_quality.i8_scores)
                scores = [Q.i8_scores(qrot[g], store, True) for g in range(n_heads)]
            else:
                lane_scores, _ = shader_scores(store, qrot, live, mode)
                scores = []
                for g in range(n_heads):
                    rows = lane_scores[g]
                    covered = ~np.isnan(rows)
                    assert covered[:fast_lo].sum() == 0
                    assert covered[fast_lo:].all(), 'lane model left a fast row uncovered'
                    s = np.array(exact[g], F32)
                    s[fast_lo:len(rows)] = rows[fast_lo:]
                    scores.append(s)
            worst_out = worst_lse = 0.0
            bias_sum = 0.0
            for g in range(n_heads):
                s = scores[g]
                o, lse = Q.kernel_attention(s, q_orig[g], k_tail, v_tail, vrec)
                worst_out = max(worst_out, float(np.abs(o - want[g][0]).max()))
                worst_lse = max(worst_lse, abs(float(lse) - want[g][1]))
                bias_sum += float(np.mean(s[fast_lo:len(s)] - exact[g][fast_lo:len(s)]))
            results[mode].append({'out_max_abs': worst_out, 'lse_max_abs': worst_lse})
            bias[mode].append(bias_sum / n_heads)
    return {'live': int(live), 'style': style, 'fast': [fast_lo, fast_hi],
            'queries': n_queries,
            'results': {m: {'out_max_abs': max(r['out_max_abs'] for r in v),
                            'lse_max_abs': max(r['lse_max_abs'] for r in v)}
                        for m, v in results.items()},
            'score_bias': {m: float(np.mean(bias[m])) for m in const_modes}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seed', type=int, default=4242)
    args = ap.parse_args()
    modes = ['algebra', 'slice', 'per_lane', 'slice_split']
    lives = [256, 384] if args.quick else [256, 384, 640, 1920]
    cases = []
    for live in lives:
        for style in ('tight', 'model'):
            cases.append(run_case(live, style, args.seed + live, modes, n_queries=2))
    ok = True
    for c in cases:
        alg = c['results']['algebra']
        broken = c['results']['slice']
        fixed = c['results']['per_lane']
        split = c['results']['slice_split']
        # The lane model must equal the validated algebra path, and the shipped
        # form must be orders of magnitude worse than it on the same store.
        reproduced = (broken['out_max_abs'] > 50.0 * max(alg['out_max_abs'], 1e-6) and
                      broken['lse_max_abs'] > 50.0 * max(alg['lse_max_abs'], 1e-6))
        matches = (fixed['out_max_abs'] <= 1.2 * alg['out_max_abs'] + 1e-5 and
                   fixed['lse_max_abs'] <= 1.2 * alg['lse_max_abs'] + 1e-5)
        agrees = (abs(split['out_max_abs'] - fixed['out_max_abs']) <=
                  1e-3 * max(fixed['out_max_abs'], 1e-6) + 1e-6)
        ok = ok and reproduced and matches and agrees
        print('%-5s live %5d fast=[%d,%d)  algebra out %8.2e lse %8.2e'
              % (c['style'], c['live'], c['fast'][0], c['fast'][1],
                 alg['out_max_abs'], alg['lse_max_abs']))
        print('    shipped  "slice"        out %8.3f lse %8.2f bias %+9.2e   (%6.0fx algebra)'
              % (broken['out_max_abs'], broken['lse_max_abs'],
                 c['score_bias']['slice'],
                 broken['out_max_abs'] / max(alg['out_max_abs'], 1e-30)))
        print('    fix      per_lane       out %8.2e lse %8.2e bias %+9.2e   (%6.2fx algebra)'
              % (fixed['out_max_abs'], fixed['lse_max_abs'],
                 c['score_bias']['per_lane'],
                 fixed['out_max_abs'] / max(alg['out_max_abs'], 1e-30)))
        print('    fix      slice_split    out %8.2e lse %8.2e'
              % (split['out_max_abs'], split['lse_max_abs']))
        flags = []
        if not reproduced:
            flags.append('shipped form did NOT reproduce the console failure')
        if not matches:
            flags.append('per_lane does not match the validated algebra path')
        if not agrees:
            flags.append('slice_split disagrees with per_lane')
        if flags:
            print('        !!', '; '.join(flags))
    doc = {'schema': 'rkva-mq-i8-lane-model-v1', 'kq_scale': KQ, 'cases': cases, 'ok': ok}
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(doc, f, indent=2)
    print('rkva_mq_i8_lane_model: %d cases, %s' % (len(cases), 'PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
