# RKVA prefill rescale-elision kernel (`attn_kvarn4_mq2_skip`)

Opt-in, bit-exact specialization of the native Bee KVarN4 two-query prefill
tile that runs pass 1 of the RKVA range path. It is the largest single kernel in
XL/XXL/XXXL prefill: in the measured XXXL continuation it accounts for
~97% of each 8-query attention segment (`gpu_pass1_us` 2725.7 µs of
`gpu_total_us` 2786.5 µs).

## What changes

`rkva_range_attn_kvarn4_mq.hlsl` keeps the production update

```
next_max = max(maximum[i], half_max);
old      = denom[i] == 0 ? 0 : exp(maximum[i] - next_max);
s[t]     = exp(s[t] - next_max);  sum += s[t];
body[i] *= old;  body[i] += s[t] * v_half[t];
denom[i] = denom[i] * old + sum;
maximum[i] = next_max;
```

`RKVA_MQ_RESCALE_SKIP=1` (`rkva_range_attn_kvarn4_mq_skip.hlsl`, PSO target
`attn_kvarn4_mq2_skip`) splits that block on `half_max > maximum[i]`:

* **grow** – the original expressions with `next_max == half_max`;
* **flat** – `next_max == maximum[i]`, so the original `old` is `exp(0) == 1.0`
  and both products are exact identities. The flat arm keeps
  `s[t] = exp(s[t] - maximum[i])`, the same accumulation order, and updates
  `denom[i] += sum`.

No other line changes: row classification, split ranges, K/V decode, query
rotation, exponentials, output layout and LSE are untouched. `RKVA_MQ_RESCALE_SKIP`
defaults to 0, and the qualified `attn_kvarn4_mq2` / `attn_kvarn4_mq4` DXIL
payloads stay byte-identical (verified by SHA-256).

## Why it is bit-exact

Multiplying by exactly 1.0 is exact for every IEEE-754 binary32 value
(`x * 1.0f == x`, including `±0`, subnormals and infinities), and
`fma(denom, 1.0f, sum)` rounds to `denom + sum`. The elided `exp(0)` is the only
removed transcendental. `tools/xbox/tests/rkva_mq_rescale_skip_exact.py` replays
both formulations in binary32 over randomized, monotone, flat, tie,
one-ulp-growth, subnormal and `denom == 0` cases (760 replay cases by default,
plus 106-half production splits) and asserts bit-identical
`body`/`denom`/`maximum` after every half tile. The same file carries a mutation
sensitivity gate: a one-ulp change to `denom` must be detected.

## Expected effect (estimate, not measured on hardware)

DXIL instruction accounting of the fast shared-tile loop (8 KV rows,
2 queries × 6 heads, `tools/xbox/.oc-artifacts` helpers in the A2 branch):

| per KV row per lane | production | fast arm | delta |
| --- | --- | --- | --- |
| total | 512.0 | 471.5 | −7.9% |
| FMA class | 237.0 | 216.0 | −8.9% |
| reduction (adds + shuffles) | 144.0 | 144.0 | 0 |
| SFU (exp) | 15.0 | 12.0 | −20% |
| int/misc | 89.1 | 72.6 | −18.5% |

The fast arm applies whenever the half-tile maximum does not grow, which for
`424`-row splits happens in ~95% of half tiles (harmonic estimate `H(106)/106`
growth events per query-group split). With that share the average instruction
count per row is ≈ 474 (−7.4%), so pass 1 is expected to drop by roughly 5–8%
and an XXXL envelope from 65.3 ms to ~61 ms (+5–7% prefill t/s) **if** pass 1
stays issue-bound. Those numbers are derived, not measured.

## Enabling and measuring

The app selects the PSO from the LocalState flag file
`rkva-mq-rescale-skip.flag` (read at HELLO, default absent = off). No RKVA
protocol field, wire format or session negotiation changes; the kernel computes
the same values, so an old client cannot observe the difference.

`tools/xbox/scripts/run-rkva-prefill-kernel-ab.py` runs the frozen-prefix A/B:
flag off → benchmark, flag on → benchmark, restore off, compare generated token
IDs byte for byte, and read the engine profile buckets
`…/optimized-kvarn4_mq2_fine/…` vs `…/optimized-kvarn4_mq2_skip/…` plus the
`kvarn4_multi_query.mq2_rescale_skip.calls` counter. It is inert without
`--execute`, installs nothing, and refuses to run while a Codex benchmark or
build is active.

## Related

* `docs/rkva-2xn256-status.md` – range qualification and measured throughput.
* `docs/rkva-range-profiling.md` – profile schema and bucket names.
* `docs/xbox-position-overflow.md` – kernel-selection modes and split policies.
