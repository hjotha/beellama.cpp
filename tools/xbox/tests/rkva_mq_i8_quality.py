#!/usr/bin/env python3
"""Quality gate for the opt-in INT8 `dot4add_i8packed` QK variant of the RKVA
KVarN4 multi-query prefill tile (`attn_kvarn4_mq2_i8`).

CPU only; no GPU, no console lease. Three attention paths over the same store:

  GOLD  float64 attention over the ORIGINAL F16-rounded K/V rows, i.e. what an
        exact F16 KV cache returns. Carries no 4-bit and no INT8 loss.
  REF   the shipped `attn_kvarn4_mq2` arithmetic: `precise` F32 decode of the
        KVarN4 record (`float(q)*scale + zero` then `*other`), F32 dot in the
        WHT-rotated domain with the 4-dim/lane 64-lane butterfly, F32 online
        softmax in 4-row halves, body merged through `wht256x6`, ORIGINAL-F16
        128-row tail in the original domain.
  I8    the same kernel with the record QK replaced by the INT8 factorization.

Algebra (see docs/agent-skills note in the report; `K` plane is [dim][pos],
`V` plane is [pos][dim], both 128-dim slices share one 8960-byte record block):

    K(d,p) = (nibble(d,p) * scale_k[d] + zero_k[d]) * other_k[p]

`scale_k`/`zero_k` are per dim, `other_k` per position and **per 128-dim
slice** (the store kernel derives it from `best_col` of its own slice), so the
score splits into two slice terms that cannot share one factored `other`:

    score = kq * sum_s other_s[p] * ( sum_{d in s} Qr[d]*nibble(d,p)*scale_k[d]
                                    + sum_{d in s} Qr[d]*zero_k[d] )
          = kq * sum_s other_s[p] * ( sq_s * dot_s(Q8) + Z_s )

with the column scale folded into the query once per (query, head, group):

    Q'  = WHT(Q) * scale_k[d]                    (F32, per dim)
    sq  = max_{d in slice} |Q'| / 126            (per query, head, group, slice)
    Q8  = clamp(rint(Q' / sq), -127, 127)        (int8)
    Z   = sum_{d in slice} WHT(Q)[d] * zero_k[d] (F32, unquantized Q)

and `dot_s(Q8)` the exact int32 sum of `Q8*(nibble - 8)` over the 128 dims of
the slice (the `+8*sum(Q8)` constant is folded into `Z`). Nibbles are 0..15, so
they are already valid non-negative int8 lanes; `dot4add_i8packed` therefore
needs no offset term. `--check-dp4a` proves the 4-lane int8 chain equals the
int64 sum and that the widest lane group stays inside int32.

Stage rows, the ORIGINAL-F16 tail, the softmax, the WHT and the partial layout
are untouched: they keep the exact F32 path in REF and in I8, so they cannot
mask a difference. Only the record-row score changes.

Run: python3 tools/xbox/tests/rkva_mq_i8_quality.py [--json out.json] [--quick]
"""
import argparse
import json

import numpy as np

GQA = 6
D = 256
RECORD = 8960
TAIL = 128
F32 = np.float32
KQ_SCALE = 1.0 / 16.0   # 1/sqrt(head_dim)


def hadamard256():
    h = np.array([[1.0]])
    for _ in range(8):
        h = np.block([[h, h], [h, -h]])
    return h / 16.0


H256 = hadamard256()


def butterfly_wht(x):
    """F32 lane-order mirror of wht256x6 (one rounding per statement)."""
    src = np.asarray(x, F32)
    t = src.reshape(-1, 256).copy()
    lane = np.arange(128)
    for stride in (1, 2, 4, 8, 16, 32, 64):
        own0, other0 = t[:, lane].copy(), t[:, lane ^ stride].copy()
        own1, other1 = t[:, lane + 128].copy(), t[:, (lane ^ stride) + 128].copy()
        upper = (lane & stride) != 0
        t[:, lane] = np.where(upper, other0 - own0, own0 + other0)
        t[:, lane + 128] = np.where(upper, other1 - own1, own1 + other1)
    x0 = t[:, :128] * F32(0.08838834764831845)
    y0 = t[:, 128:] * F32(0.08838834764831845)
    out = np.concatenate([(x0 + y0) * F32(0.7071067811865475),
                          (x0 - y0) * F32(0.7071067811865475)], axis=1)
    return out.astype(F32).reshape(src.shape)


def f16(x):
    return np.asarray(x, F32).astype(np.float16).astype(F32)


# --------------------------------------------------- record encode / decode ---


def encode_record(m, flip=0):
    """One 128x128 KVarN4 plane of `m` ([row, col]): nibbles + 3 F16 axes.

    Row-major nibble packing, per-row scale/zero, per-column signed `other`,
    and the shipped half-the-rows negative-scale pattern selected by `flip`
    (rkva_store_kvarn4.hlsl writes `best_row*scale`, `best_row*lo`,
    `best_col`; the sign split mirrors the Sinkhorn balancing it produces).
    """
    m = np.asarray(m, np.float64)
    other = np.abs(m).max(axis=0)
    other = np.where(other > 0, other, 1.0) * np.where(np.arange(128) % 3 == 0, -1.0, 1.0)
    other = f16(other)
    z = m / other.astype(np.float64)[None, :]
    lo, hi = z.min(axis=1), z.max(axis=1)
    signed = np.arange(128) % 2 == flip
    span = np.where(hi > lo, hi - lo, 1.0) / 15.0
    scale = f16(np.where(signed, -span, span))
    zero = f16(np.where(signed, hi, lo))
    q = np.rint((z - zero.astype(np.float64)[:, None]) /
                scale.astype(np.float64)[:, None])
    q = np.clip(q, 0, 15).astype(np.uint8).reshape(-1)
    nib = (q[0::2] | (q[1::2] << 4)).astype(np.uint8).tobytes()
    blob = (nib + scale.astype('<f2').tobytes() + zero.astype('<f2').tobytes() +
            other.astype('<f2').tobytes())
    assert len(blob) == RECORD
    return blob


def record_planes(blob):
    """(nibble[128,128] uint8, scale[128], zero[128], other[128]) of one plane."""
    q = np.frombuffer(blob[:8192], np.uint8)
    quants = np.empty((128, 128), np.uint8)
    quants[:, 0::2] = q.reshape(128, 64) & 15
    quants[:, 1::2] = q.reshape(128, 64) >> 4
    axes = np.frombuffer(blob[8192:], '<f2').astype(F32).reshape(3, 128)
    return quants, axes[0], axes[1], axes[2]


def decode_plane(quants, scale, zero, other):
    """Shipped decode: (q*scale[row] + zero[row]) * other[col], one rounding each."""
    with np.errstate(over='ignore', invalid='ignore'):
        affine = (quants.astype(F32) * scale[:, None]).astype(F32) + zero[:, None]
        return (affine.astype(F32) * other[None, :]).astype(F32)


# ------------------------------------------------------------------- store ---


def make_rows(rng, n, style):
    x = rng.normal(0, 1, (n, D))
    if style == 'model':
        # the repo's own oracle distribution (rkva_kvarn_serial_prefix_model)
        x *= 10.0 ** rng.uniform(-1.5, 0.7, (n, 1))
        x *= rng.choice([-1.0, 1.0], (n, 1))
        x[:, rng.integers(0, D, 3)] *= 6.0
        x = np.where(np.abs(x) < 1e-3, np.copysign(1e-3, x + 1e-12), x)
    elif style == 'tight':
        x *= 10.0 ** rng.uniform(-0.4, 0.4, (n, 1))
    return x.astype(F32)


class Store:
    """Sealed records for complete groups + original-F16 ring for the last 128."""

    def __init__(self, rng, live, style):
        self.live = live
        self.tail_begin = live - min(live, TAIL)
        self.groups = self.tail_begin // 128
        self.k = make_rows(rng, live, style)
        self.v = make_rows(rng, live, style)
        self.blobs = []       # per group: (K blobs[2], V blobs[2])
        self.kplanes = []     # per group: list of 2 (nibbles[D/2,128], scale, zero, other)
        for g in range(self.groups):
            krot = butterfly_wht(self.k[g * 128:(g + 1) * 128])
            vrot = butterfly_wht(self.v[g * 128:(g + 1) * 128])
            kb = [encode_record(krot[:, s * 128:(s + 1) * 128].T, flip=s) for s in (0, 1)]
            vb = [encode_record(vrot[:, s * 128:(s + 1) * 128], flip=s) for s in (0, 1)]
            self.blobs.append((kb, vb))
            self.kplanes.append([record_planes(kb[s]) for s in (0, 1)])

    def sources(self):
        """[tail_begin, 256] storage-domain K and V of the record rows."""
        n = self.tail_begin
        k = np.zeros((n, D), F32)
        v = np.zeros((n, D), F32)
        for g in range(self.groups):
            for s in (0, 1):
                qq, ss, zz, oo = record_planes(self.blobs[g][0][s])
                k[g * 128:(g + 1) * 128, s * 128:(s + 1) * 128] = decode_plane(qq, ss, zz, oo).T
                qq, ss, zz, oo = record_planes(self.blobs[g][1][s])
                v[g * 128:(g + 1) * 128, s * 128:(s + 1) * 128] = decode_plane(qq, ss, zz, oo)
        return k, v


# ---------------------------------------------------------------- score paths --


def wave_sum_f32(part):
    """64-lane WaveActiveSum butterfly over the last axis in F32."""
    a = part.astype(F32)
    lane = np.arange(64)
    for stride in (32, 16, 8, 4, 2, 1):
        a = (a + a[..., lane ^ stride]).astype(F32)
    return a[..., 0]


def ref_scores(qrot, ksrc):
    """Shipped record score: 4 dims per lane, 64-lane butterfly, then *kq_scale."""
    part = (qrot.reshape(64, 4).astype(np.float64)[None, :, :] *
            ksrc.reshape(-1, 64, 4).astype(np.float64)).sum(axis=2)
    return (wave_sum_f32(part) * F32(KQ_SCALE)).astype(F32)


def slice_reduce_max(v):
    """Max over the 8 lanes that share one 128-dim slice (xor 4,2,1)."""
    a = v.astype(F32)
    lane = np.arange(16)
    for stride in (4, 2, 1):
        a = np.maximum(a, a[..., lane ^ stride])
    return a


def i8_scores(qrot, store, dc_exact=False):
    """INT8 record score, exact int32 dot per slice, slice-local Q8 scale.

    `dc_exact` uses the nibble-centred operand (n - 8) plus the exact
    `8*sum(Q')` constant, which cancels the DC part of the Q8 rounding error at
    no per-position cost; without it the operand is the raw nibble (0..15).
    """
    n = store.tail_begin
    out = np.empty(n, F32)
    for g in range(store.groups):
        acc = np.zeros((128,), np.float64)
        for s in (0, 1):
            qn, sc, ze, oo = store.kplanes[g][s]
            d0 = s * 128
            qv = qrot[d0:d0 + 128]
            qp = (qv.astype(F32) * sc.astype(F32)).astype(F32)
            mx = F32(np.abs(qp).max())
            sq = F32(mx / 126.0) if mx > 0 else F32(0.0)
            if mx > 0:
                q8 = np.clip(np.rint(qp.astype(np.float64) / float(sq)),
                             -127.0, 127.0).astype(np.int32)
            else:
                q8 = np.zeros(128, np.int32)
            z = F32(np.dot(qv.astype(np.float64), ze.astype(np.float64)))
            if dc_exact:
                qsum = F32(np.sum(qp.astype(np.float64)))
                dot = (qn.astype(np.int64) - 8).T @ q8.astype(np.int64)
                const = F32(z + F32(8.0) * qsum)
            else:
                dot = qn.astype(np.int64).T @ q8.astype(np.int64)
                const = z
            acc += (dot.astype(F32) * sq + const).astype(np.float64) * oo.astype(np.float64)
        out[g * 128:(g + 1) * 128] = (acc * F32(KQ_SCALE)).astype(F32)
    return out


# ----------------------------------------------------------------- attention ---


def kernel_attention(s_rec, q_orig, k_tail, v_tail, v_rec):
    """4-row half online softmax, original-domain tail, WHT(body), F32."""
    body = np.zeros(D, F32)
    tbody = np.zeros(D, F32)
    maximum = F32(-3.402823466e38)
    den = F32(0)
    for h0 in range(0, len(s_rec), 4):
        half = s_rec[h0:h0 + 4]
        nmax = F32(max(maximum, F32(half.max())))
        ow = F32(0) if den == 0 else F32(np.exp(F32(maximum - nmax)))
        w = np.exp((half - nmax).astype(F32)).astype(F32)
        body = (body * ow + (w[:, None] * v_rec[h0:h0 + 4]).sum(axis=0)).astype(F32)
        den = F32(den * ow + w.sum())
        maximum = nmax
    for row in range(len(k_tail)):
        s = F32(np.dot(q_orig.astype(np.float64),
                       k_tail[row].astype(np.float64)) * KQ_SCALE)
        nmax = F32(max(maximum, s))
        ow = F32(0) if den == 0 else F32(np.exp(F32(maximum - nmax)))
        w = F32(np.exp(F32(s - nmax)))
        body = (body * ow).astype(F32)
        tbody = (tbody * ow + w * v_tail[row]).astype(F32)
        den = F32(den * ow + w)
        maximum = nmax
    o = (butterfly_wht(body) + tbody).astype(F32)
    scale = float(den) if den != 0 else 1.0
    lse = F32(maximum + F32(np.log(den))) if den != 0 else F32(-np.inf)
    return o.astype(np.float64) / scale, lse


def run_case(rng, live, style, dc_exact=True, n_queries=8):
    """Aggregate GOLD/REF/I8 metrics over `n_queries` independent queries."""
    store = Store(rng, live, style)
    krec, vrec = store.sources()
    k_tail = f16(store.k[store.tail_begin:])
    v_tail = f16(store.v[store.tail_begin:])

    # GOLD: exact F16 rows, float64. The kernel's rotated-domain score
    # WHT(k).WHT(q) equals k.q, so the reference query is the original one.
    k_all = np.concatenate([f16(store.k[:store.tail_begin]).astype(np.float64),
                            k_tail.astype(np.float64)], axis=0)
    v_all = np.concatenate([f16(store.v[:store.tail_begin]).astype(np.float64),
                            v_tail.astype(np.float64)], axis=0)

    acc = dict(rel_o_rms=[], rel_o_max=[], one_minus_cos=[], lse_abs_max=[],
               wl1_ref=[], wl1_i8=[], gap=[], rms_i8=[], added=[])
    for _ in range(n_queries):
        q_orig = rng.normal(size=D).astype(F32)
        qrot = butterfly_wht(q_orig)
        q64 = q_orig.astype(np.float64)
        s_gold_rec = f16(store.k[:store.tail_begin]).astype(np.float64) @ q64 * KQ_SCALE
        s_g = k_all @ q64 * KQ_SCALE
        mx = s_g.max()
        w_g = np.exp(s_g - mx)
        o_gold = (w_g @ v_all) / w_g.sum()
        lse_gold = mx + np.log(w_g.sum())

        s_ref = ref_scores(qrot, krec)
        s_i8 = i8_scores(qrot, store, dc_exact)
        o_ref, lse_ref = kernel_attention(s_ref, q_orig, k_tail, v_tail, vrec)
        o_i8, lse_i8 = kernel_attention(s_i8, q_orig, k_tail, v_tail, vrec)

        def stats(o, lse):
            return (float(np.sqrt(((o - o_gold) ** 2).mean()) /
                          (np.sqrt((o_gold ** 2).mean()) + 1e-30)),
                    float(np.abs(o - o_gold).max() / max(np.abs(o_gold).max(), 1e-30)),
                    1.0 - float(np.dot(o, o_gold) /
                                (np.linalg.norm(o) * np.linalg.norm(o_gold) + 1e-30)),
                    float(np.abs(lse - lse_gold).max()))

        r, i = stats(o_ref, lse_ref), stats(o_i8, lse_i8)
        acc['rel_o_rms'].append((r[0], i[0]))
        acc['rel_o_max'].append((r[1], i[1]))
        acc['one_minus_cos'].append((r[2], i[2]))
        acc['lse_abs_max'].append((r[3], i[3]))
        w_ref = np.exp(s_ref.astype(np.float64) - float(s_ref.max()))
        w_i8 = np.exp(s_i8.astype(np.float64) - float(s_i8.max()))
        w_g_rec = w_g[:store.tail_begin] / w_g.sum()
        acc['wl1_ref'].append(float(np.abs(w_ref / w_ref.sum() - w_g_rec).sum()))
        acc['wl1_i8'].append(float(np.abs(w_i8 / w_i8.sum() - w_g_rec).sum()))
        acc['gap'].append(float(np.sqrt(((s_ref - s_gold_rec) ** 2).mean())))
        acc['rms_i8'].append(float(np.sqrt(((s_i8 - s_gold_rec) ** 2).mean())))
        acc['added'].append(float(np.sqrt(((s_i8 - s_ref) ** 2).mean())))

    def mean(xs):
        return float(np.mean(xs))

    gap = mean(acc['gap'])
    added = mean(acc['added'])
    out = dict(
        live=int(live), groups=int(store.groups), style=style, dc_exact=bool(dc_exact),
        queries=int(n_queries), score_std=float(np.mean(acc['gap'])) and None,
        metrics={}, score_rms={})
    for key in ('rel_o_rms', 'rel_o_max', 'one_minus_cos', 'lse_abs_max'):
        out['metrics'][key] = {'ref': mean([a for a, _ in acc[key]]),
                               'i8': mean([b for _, b in acc[key]])}
    out['metrics']['weight_l1'] = {'ref': mean(acc['wl1_ref']), 'i8': mean(acc['wl1_i8'])}
    out['score_rms'] = {'ref': gap, 'i8': mean(acc['rms_i8']),
                        'ratio': mean(acc['rms_i8']) / gap if gap else 0.0,
                        'added_over_ref': added / gap if gap else 0.0}
    out.pop('score_std', None)
    return out


# --------------------------------------------------------------- dp4a proof ---


def check_dp4a(seed=7, trials=64):
    """Exact int32 dp4a chain == int64 sum; widest score stays inside int32."""
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        nib = rng.integers(0, 16, D)
        q8 = rng.integers(-127, 128, D)
        acc = 0
        for c in range(D // 4):
            word = sum(int(nib[4 * c + b]) << (8 * b) for b in range(4))
            qword = sum((int(q8[4 * c + b]) & 0xFF) << (8 * b) for b in range(4))
            for b in range(4):
                av = (word >> (8 * b)) & 0xFF
                av = av - 256 if av >= 128 else av
                bv = (qword >> (8 * b)) & 0xFF
                bv = bv - 256 if bv >= 128 else bv
                acc += av * bv
        exact = int(np.dot(nib.astype(np.int64), q8.astype(np.int64)))
        assert acc == exact, (acc, exact)
    worst = D * 15 * 127          # widest full 256-dim score
    assert worst < 2 ** 31 - 1
    return {'trials': trials, 'worst_full_score': worst, 'int32_limit': 2 ** 31 - 1}


# ---------------------------------------------------------------- acceptance ---


def verdict(case, tol_added=0.35, tol_ratio=1.15):
    """Accept only if the variant stays inside the error the product already ships.

    Primary criterion: the Q8 rounding error must stay a small fraction of the
    4-bit KVarN4 score error production already accepts (`added_over_ref` is the
    variant-vs-shipped gap over the shipped-vs-F16 gap). Secondary: the
    attention output, LSE and softmax weights must not grow.
    """
    reasons = []
    if case['score_rms']['added_over_ref'] > tol_added:
        reasons.append('Q8 rounding adds %.2fx of the shipped 4-bit score error (> %.2f)'
                       % (case['score_rms']['added_over_ref'], tol_added))
    if case['score_rms']['ratio'] > tol_ratio:
        reasons.append('total score RMS grew past %.2fx the shipped error' % tol_ratio)
    for key in ('rel_o_rms', 'rel_o_max', 'one_minus_cos', 'lse_abs_max'):
        ref, i8 = case['metrics'][key]['ref'], case['metrics'][key]['i8']
        if i8 > tol_ratio * ref:
            reasons.append('%s grew past %.2fx the shipped error' % (key, tol_ratio))
    ref, i8 = case['metrics']['weight_l1']['ref'], case['metrics']['weight_l1']['i8']
    if i8 > tol_ratio * ref:
        reasons.append('softmax weight L1 grew past %.2fx the shipped error' % tol_ratio)
    return (not reasons), reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--queries', type=int, default=8)
    ap.add_argument('--seed', type=int, default=11)
    args = ap.parse_args()
    dp4a = check_dp4a()
    lives = [256, 640] if args.quick else [256, 384, 640, 1920, 4096]
    results = []
    for live in lives:
        for style in ('tight', 'model'):
            for dc in (True, False):
                rng = np.random.default_rng(args.seed + live)
                case = run_case(rng, live, style, dc, args.queries)
                ok, why = verdict(case)
                results.append(dict(case=case, ok=ok, reasons=why))
    gated = [r for r in results if r['case']['dc_exact']]
    all_ok = all(r['ok'] for r in gated)
    report = {'schema': 'rkva-mq-i8-quality-v1', 'kq_scale': KQ_SCALE,
              'tail_tokens': TAIL, 'dp4a': dp4a, 'cases': results,
              'gate': 'dc_exact cases only', 'ok': all_ok}
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=2)
    for r in results:
        c = r['case']
        m = c['metrics']
        print('%-5s live %5d dc=%-5s add/ref %5.3f total/ref %5.3f | '
              'rel_o_rms %8.3e->%8.3e  1-cos %.3e->%.3e  lse %8.3e->%8.3e  '
              'wL1 %.3e->%.3e  %s'
              % (c['style'], c['live'], c['dc_exact'],
                 c['score_rms']['added_over_ref'], c['score_rms']['ratio'],
                 m['rel_o_rms']['ref'], m['rel_o_rms']['i8'],
                 m['one_minus_cos']['ref'], m['one_minus_cos']['i8'],
                 m['lse_abs_max']['ref'], m['lse_abs_max']['i8'],
                 m['weight_l1']['ref'], m['weight_l1']['i8'],
                 ('PASS' if r['ok'] else 'FAIL') if c['dc_exact'] else 'info'))
        for why in r['reasons']:
            print('        -', why)
    print('rkva_mq_i8_quality: %d gated cases (dc exact, %d queries each), %s' %
          (len(gated), args.queries, 'PASS' if all_ok else 'FAIL'))
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
