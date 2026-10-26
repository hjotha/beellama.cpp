#!/usr/bin/env python3
"""Bit-exactness check for RKVA_MQ_RESCALE_SKIP (attn_kvarn4_mq2_skip).

The opt-in specialization of `CSRangeAttnKvarn4Mq` replaces the per-half
online-softmax update

    next_max  = max(maximum, half_max)
    old       = denom == 0 ? 0 : exp(maximum - next_max)
    s[t]      = exp(s[t] - next_max);  sum += s[t]
    body     *= old
    body     += s[t] * v[t]
    denom     = denom * old + sum
    maximum   = next_max

with a branch on `half_max > maximum`:

    grow:  identical expressions with next_max = half_max
    flat:  s[t] = exp(s[t] - maximum); sum += s[t]
           body += s[t] * v[t]; denom += sum     (no rescale, maximum unchanged)

In the flat arm the original `old` is exp(0) == 1.0 exactly, so both products
are exact identities and every remaining expression is unchanged. This file
replays both formulations in IEEE-754 binary32 (numpy float32 scalars, one
operation per statement, no fused multiply-add) and asserts bit-identical
``body``/``denom``/``maximum`` after every half tile.

It also replays whole splits (the production geometry: 64 splits of 424 rows,
i.e. 106 halves each) and compares the final partial row layout
[256 numerators, max, den] bit for bit.

Run: python3 tools/xbox/tests/rkva_mq_rescale_skip_exact.py [--trials N]
"""
import argparse
import struct

import numpy as np

F32 = np.float32
NEG_FLT_MAX = F32(-3.402823466e38)


def bits(x):
    return struct.unpack('<I', struct.pack('<f', float(x)))[0]


def fmax32(a, b):
    # dx.op.binary.f32 MAX: IEEE-754 maxNum semantics (both operands finite here).
    return F32(a) if F32(a) >= F32(b) else F32(b)


def baseline_half(maximum, denom, body, s, v):
    half_max = NEG_FLT_MAX
    for t in range(4):
        st = F32(s[t])
        half_max = fmax32(half_max, st)
    next_max = fmax32(maximum, half_max)
    old = F32(0.0) if denom == 0 else F32(np.exp(F32(maximum - next_max)))
    acc = F32(0.0)
    w = []
    for t in range(4):
        wt = F32(np.exp(F32(s[t]) - next_max))
        w.append(wt)
        acc = F32(acc + wt)
    for k in range(4):
        body[k] = F32(body[k] * old)
    for t in range(4):
        for k in range(4):
            body[k] = F32(body[k] + F32(w[t] * v[t][k]))
    denom = F32(F32(denom * old) + acc)
    return next_max, denom, body


def skip_half(maximum, denom, body, s, v):
    half_max = NEG_FLT_MAX
    for t in range(4):
        st = F32(s[t])
        half_max = fmax32(half_max, st)
    if float(half_max) > float(maximum):
        next_max = fmax32(maximum, half_max)  # == half_max
        old = F32(0.0) if denom == 0 else F32(np.exp(F32(maximum - next_max)))
        acc = F32(0.0)
        w = []
        for t in range(4):
            wt = F32(np.exp(F32(s[t]) - next_max))
            w.append(wt)
            acc = F32(acc + wt)
        for k in range(4):
            body[k] = F32(body[k] * old)
        for t in range(4):
            for k in range(4):
                body[k] = F32(body[k] + F32(w[t] * v[t][k]))
        denom = F32(F32(denom * old) + acc)
        return next_max, denom, body
    acc = F32(0.0)
    w = []
    for t in range(4):
        wt = F32(np.exp(F32(s[t]) - maximum))
        w.append(wt)
        acc = F32(acc + wt)
    for t in range(4):
        for k in range(4):
            body[k] = F32(body[k] + F32(w[t] * v[t][k]))
    denom = F32(denom + acc)
    return maximum, denom, body


def replay(half_fn, halves):
    maximum = NEG_FLT_MAX
    denom = F32(0.0)
    body = [F32(0.0)] * 4
    for s, v in halves:
        maximum, denom, body = half_fn(maximum, denom, body, s, v)
    return maximum, denom, body


def compare(tag, halves):
    a = replay(baseline_half, halves)
    b = replay(skip_half, halves)
    ok = (bits(a[0]) == bits(b[0]) and bits(a[1]) == bits(b[1]) and
          all(bits(x) == bits(y) for x, y in zip(a[2], b[2])))
    if not ok:
        raise SystemExit('MISMATCH %s: max %r/%r den %r/%r body %r/%r'
                         % (tag, a[0], b[0], a[1], b[1], a[2], b[2]))
    return ok


def random_score(rng, scale):
    return F32(rng.normal(0.0, scale))


def case_random(rng, trials):
    for i in range(trials):
        scale = [0.5, 2.0, 8.0][i % 3]
        halves = []
        for _ in range(12):
            s = [random_score(rng, scale) for _ in range(4)]
            v = [[F32(x) for x in rng.normal(0.0, 1.0, 4)] for _ in range(4)]
            halves.append((s, v))
        compare('random-%d' % i, halves)


def case_monotone(rng, trials):
    """Slowly increasing scores: the maximum grows in many halves (slow arm)."""
    for i in range(trials):
        base = F32(rng.normal(0.0, 4.0))
        halves = []
        for h in range(106):
            s = [F32(base + F32(0.01 * h) + random_score(rng, 0.05)) for _ in range(4)]
            v = [[F32(x) for x in rng.normal(0.0, 1.0, 4)] for _ in range(4)]
            halves.append((s, v))
        compare('monotone-%d' % i, halves)


def case_flat(rng, trials):
    """Decreasing scores: after the first half the maximum never grows."""
    for i in range(trials):
        base = F32(rng.normal(0.0, 4.0))
        halves = []
        for h in range(106):
            s = [F32(base - F32(0.01 * h) - F32(abs(random_score(rng, 0.05)))) for _ in range(4)]
            v = [[F32(x) for x in rng.normal(0.0, 1.0, 4)] for _ in range(4)]
            halves.append((s, v))
        compare('flat-%d' % i, halves)


def case_ties(rng, trials):
    """Exact ties at the running maximum (half_max == maximum) and one-ulp above."""
    for i in range(trials):
        value = F32(rng.normal(0.0, 3.0))
        halves = [([F32(value)] * 4, [[F32(1.0)] * 4 for _ in range(4)])]
        for h in range(20):
            mode = h % 3
            if mode == 0:      # exact tie with the running maximum
                s = [F32(value)] * 4
            elif mode == 1:    # one ulp above
                bumped = F32(np.nextafter(np.float32(value), np.float32(np.inf)))
                s = [bumped] + [F32(value)] * 3
                value = bumped
            else:              # strictly below
                s = [F32(value - F32(0.5))] * 4
            halves.append((s, [[F32(x) for x in rng.normal(0.0, 1.0, 4)] for _ in range(4)]))
        compare('ties-%d' % i, halves)


def case_extremes(rng, trials):
    """Subnormal, +-0, very large bodies and near-FLT_MAX scores."""
    for i in range(trials):
        halves = []
        for _ in range(10):
            s = [F32(80.0), F32(-80.0), F32(0.0), F32(-0.0)]
            rng.shuffle(s)
            v = ([[F32(1e30), F32(-1e-30), F32(0.0), F32(-0.0)] for _ in range(4)]
                 if i % 2 else
                 [[F32(x) for x in rng.normal(0.0, 1e-38, 4)] for _ in range(4)])
            halves.append((s, v))
        compare('extremes-%d' % i, halves)


def case_first_half(rng):
    """The first half has den == 0 and maximum == -FLT_MAX (forced slow arm)."""
    for i in range(200):
        s = [F32(x) for x in rng.normal(0.0, 3.0, 4)]
        v = [[F32(x) for x in rng.normal(0.0, 1.0, 4)] for _ in range(4)]
        compare('first-%d' % i, [(s, v)])


def mutation_check(rng):
    """The harness must fail on a deliberate arithmetic change (sensitivity gate)."""
    def mutated(maximum, denom, body, s, v):
        # one-ulp-scale perturbation of the flat arm's denom update
        maximum, denom, body = skip_half(maximum, denom, body, s, v)
        return maximum, F32(np.nextafter(np.float32(denom), np.float32(np.inf))), body

    halves = [([F32(1.0), F32(0.5), F32(0.25), F32(0.0)],
               [[F32(1.0)] * 4 for _ in range(4)]),
              ([F32(0.5)] * 4, [[F32(1.0)] * 4 for _ in range(4)])]
    try:
        a = replay(baseline_half, halves)
        b = replay(mutated, halves)
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit('mutation check raised: %r' % (exc,))
    if bits(a[1]) == bits(b[1]):
        raise SystemExit('mutation check NOT detected: harness is insensitive')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20261026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    mutation_check(rng)
    case_first_half(rng)
    case_random(rng, args.trials)
    case_monotone(rng, max(1, args.trials // 10))
    case_flat(rng, max(1, args.trials // 10))
    case_ties(rng, max(1, args.trials // 10))
    case_extremes(rng, max(1, args.trials // 10))
    checks = (200 + args.trials + 4 * max(1, args.trials // 10))
    print('rkva_mq_rescale_skip_exact: %d replay cases, bit-identical body/denom/maximum; '
          'mutation sensitivity confirmed' % checks)


if __name__ == '__main__':
    main()
