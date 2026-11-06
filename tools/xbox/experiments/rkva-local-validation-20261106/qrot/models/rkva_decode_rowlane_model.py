#!/usr/bin/env python3
"""CPU FP32 transliteration of RKVA_DECODE_ROWLANE (not a GPU/ISA emulator).

Reads the resident byte image through bounds-checked shader loads. Follows the
128-row group / two wave64 mapping, split and phase order, per-slice Q*scale and
zero folding, sequential fused QK and optionally contracted PV, tree wave sums, online
softmax, separate body/tail merge, and both output contracts. The independent
canonical serial-prefix FP64 oracle and tolerances are those of the GPU probe.
Use the bundled run.sh for the supported local CPU test entry point.
"""
import numpy as np
import rkva_kvarn_serial_prefix_model as sp


km = sp.km
F = np.float32
CONTRACT = False


def mad(a, b, c):
    # Exact product of binary32 inputs fits binary64; round once like DXIL FMad.
    return (np.asarray(a, np.float64) * np.asarray(b, np.float64) + c).astype(F)


def muladd(a, b, c):
    # DXIL fast fmul/fadd may contract in the console JIT. Test both choices.
    return mad(a, b, c) if CONTRACT else (np.asarray(a, F)*b+c).astype(F)


def wave_sum(x):
    x = np.asarray(x, F).copy()
    assert x.shape[-1] == 64
    lanes = np.arange(64)
    for stride in (1, 2, 4, 8, 16, 32):
        x = (x + x[..., lanes ^ stride]).astype(F)
    return x[..., 0]


def dot4(a, b):
    p = (a * b).astype(F).reshape(*a.shape[:-1], 64, 4)
    return ((p[..., 0] + p[..., 1]) + p[..., 2]) + p[..., 3]


def record_scores(P, qr, group, kv, wave):
    """One row/lane, six heads: byte addresses exactly as the HLSL."""
    pos = wave * 64 + np.arange(64, dtype=np.uint64)
    score = np.zeros((6, 64), F)
    for sl in range(2):
        ph = kv * 2 + sl
        lo, hi = P.region('rec', group, ph)
        dims = np.arange(128)
        sc = P.half_at(lo + 8192 + dims * 2, lo, hi)
        ze = P.half_at(lo + 8448 + dims * 2, lo, hi)
        qs = (qr[:, sl*128:(sl+1)*128] * sc).astype(F)
        zero = np.zeros(256, F)
        zero[sl*128:(sl+1)*128] = ze
        z = wave_sum(dot4(qr, zero))
        acc = np.zeros((6, 64), F)
        for d in range(128):
            word = P.load(lo + d * 64 + (pos >> 3) * 4, lo, hi)
            nib = ((word >> ((pos & 7) * 4)) & 15).astype(F)
            acc = mad(qs[:, d, None], nib, acc)
        other = P.half_at(lo + 8704 + pos * 2, lo, hi)
        score = muladd(acc + z[:, None], other, score)
    return (score * F(sp.SCALE)).astype(F)


def run(K, V, q, active, pos, mode=2, forced=None, qrot=None, bad_tail=False):
    limit = km.causal_limit(K.lay, active, pos)
    K.window = V.window = limit
    split_k, rows = forced or km.kvarn4_decode_split(active, mode)
    tb = limit - min(limit, K.lay.ptt)
    partials = []
    for split in range(split_k):
        begin, end = min(limit, split*rows), min(limit, (split+1)*rows)
        flo = (max(begin, 128 if K.lay.origin == 0 else 0) + 127) & ~127
        fhi = min(end, limit & ~127, tb) & ~127
        if fhi <= flo:
            flo = fhi = end
        pm, pd, pa = np.empty(24, F), np.empty(24, F), np.empty((24, 256), F)
        for kv in range(4):
            hs = slice(kv*6, (kv+1)*6)
            qo = q[hs]
            qr = km.butterfly_wht(qo) if qrot is None else qrot[hs]
            if bad_tail:
                qo = qr
            states, covered = [], []
            for wave in range(2):
                mx = np.full(6, -km.FLT_MAX, F)
                den = np.zeros(6, F)
                body, tail = np.zeros((6, 256), F), np.zeros((6, 256), F)
                for t0 in range(flo, fhi, 128):
                    score = record_scores(K, qr, t0//128, kv, wave)
                    nm = np.maximum(mx, score.max(axis=1))
                    ow = np.where(den == 0, F(0), np.exp(mx-nm)).astype(F)
                    weights = np.exp(score-nm[:, None]).astype(F)
                    body *= ow[:, None]
                    den = muladd(den, ow, wave_sum(weights))
                    mx = nm
                    for t in range(64):
                        row = t0 + wave*64 + t
                        vv = km.opt_body4(V, True, row, kv, limit)
                        body = muladd(weights[:, t, None], vv, body)
                        covered.append(row)
                # Existing residual-body and exact-tail phases, in that order.
                for sb, se, exact in ((begin, min(flo, tb), False),
                                      (fhi, min(end, tb), False),
                                      (max(begin, tb), end, True)):
                    for row in range(sb+wave, se, 2):
                        if exact:
                            kk, vv, qa = km.opt_tail4(K, row, kv), km.opt_tail4(V, row, kv), qo
                        else:
                            kk = km.opt_body4(K, False, row, kv, limit)
                            vv = km.opt_body4(V, True, row, kv, limit)
                            qa = qr
                        score = (wave_sum(dot4(qa, kk))*F(sp.SCALE)).astype(F)
                        nm = np.maximum(mx, score)
                        ow = np.where(den == 0, F(0), np.exp(mx-nm)).astype(F)
                        wt = np.exp(score-nm).astype(F)
                        if exact:
                            body *= ow[:, None]
                            tail = muladd(wt[:, None], vv, tail*ow[:, None])
                        else:
                            body = muladd(wt[:, None], vv, body*ow[:, None])
                        den = muladd(den, ow, wt)
                        mx = nm
                        covered.append(row)
                states.append((mx, den, np.concatenate((body, tail), axis=1)))
            assert sorted(covered) == list(range(begin, end)), 'row coverage'
            mm, md, both = km.merge_partial_states(states)
            bm, tm = both[:, :256], both[:, 256:]
            pm[hs], pd[hs] = mm, md
            pa[hs] = (km.butterfly_wht(bm)+tm if split_k == 1 else
                      bm+km.butterfly_wht(tm))
        partials.append((pm, pd, pa))
    return km.reduce_partials(partials, rotated=split_k != 1)
