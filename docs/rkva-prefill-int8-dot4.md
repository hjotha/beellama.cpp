# RKVA prefill INT8 QK kernel (`attn_kvarn4_mq2_i8`)

Opt-in, **not bit-exact** specialization of the native Bee KVarN4 2-query prefill
tile used by the Xbox frontend for positional overflow attention (`pass1` of
prefill). It computes the fast shared record tile's QK with
`dot4add_i8packed` (SM 6.4) instead of a 4-dim/lane F32 dot.

* Shader: `tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq_i8.hlsl` (wrapper,
  sets `RKVA_MQ_I8 1`) over `rkva_range_attn_kvarn4_mq.hlsl`.
* DXIL: `tools/xbox/shaders/generated/rkva_range_attn_kvarn4_mq2_i8_dxil.h`
  (`cs_6_4`, 140 356 B).
* Build: `tools/xbox/scripts/compile-rkva-range-shaders.sh attn_kvarn4_mq2_i8`.
* Flag: LocalState file `rkva-mq-i8.flag` (default off), read at HELLO.
  Selection: `remote-attn.cpp` → `RemoteAttnEngine::set_mq_i8()` →
  `attend_range()`.
* Bucket: `attention/kvarn4-prefix/optimized-kvarn4_mq2_i8/<nq>/<rows>/warm`;
  counters `kvarn4_multi_query.mq2_i8.{available,status,calls,exact:false}`.
* Quality gate: `tools/xbox/tests/rkva_mq_i8_quality.py`.
* Hardware A/B: `tools/xbox/scripts/run-rkva-prefill-int8-ab.py` (inert).

## Why the reduction, not the dot, is the target

The shipped `attn_kvarn4_mq2` tile gives every lane 4 dims of a 256-dim head and
needs a full 64-lane `WaveActiveSum` per score. In the static wave64 cost model,
cross-lane reduction accounts for 144 of 512 units/KV row (28%), while the QK
dot itself is one `dot4` plus one multiply. Each dot intrinsic expresses four
MACs per lane; its instruction count alone does not establish GPU throughput. The win comes from int8's
density letting a lane hold **16 dims** instead of 4, which shrinks the
reduction from 64 to 16 lanes per row and lets four KV rows
reduce through one wave prefix and constant endpoint broadcasts.

## Exact algebra

The record decodes as `K(d,p) = (nibble(d,p) * scale_k[d] + zero_k[d]) * other_k[p]`
with `scale_k`/`zero_k` per dim and `other_k` per position **and per 128-dim
slice** (`rkva_store_kvarn4.hlsl` writes `best_row*scale`, `best_row*lo`,
`best_col` of its own slice). A score therefore splits into two slice terms:

```
score = kq * sum_s other_s[p] * ( sum_{d in s} Qr[d]*nibble(d,p)*scale_k[d]
                                + sum_{d in s} Qr[d]*zero_k[d] ),   Qr = WHT(Q)
```

The column scale is folded into the rotated query once per (query, head, group):

```
Q'   = Qr * scale_k                       (F32, per dim)
sq_s = max_{d in s} |Q'| / 126            (per query, head, group, slice)
Q8   = clamp(round(Q' / sq_s), -127, 127) (int8; |Q'| <= 126*sq_s so no clamp fires)
Z_s  = sum_{d in s} Qr[d] * zero_k[d]     (F32, unquantized Q)
Qsum_s = sum_{d in s} Q'
```

and each lane's contribution becomes one dp4a chain plus one fma:

```
dot  = sum_{d in lane} Q8[d] * nibble(d,p)            (exact int32)
C_s  = Z_s + 8*Qsum_s - 8*sq_s*sum_d Q8               (per query,head,group,slice)
C_lane = Z_lane + 8*Qsum_lane - 8*sq_s*sum_lane(Q8)
lane_value = (float(dot) * sq_s + C_lane) * other_s[p]
score      = kq * sum over the 16 lanes of the group
```

`nibble` is 0..15, i.e. already a valid non-negative int8 lane, so the dp4a
operand needs no offset. `C_s` is the important term: it carries the *exact* DC
part of the Q8 rounding error, so the residual is `-sum_d (Q'-sq*Q8)*(nibble-8)`
instead of `-sum_d (Q'-sq*Q8)*nibble`. That single constant cuts the added score
error ~2.5x at no per-position cost.

`C_s` must be **distributed over the lanes of the slice, not reduced into them**.
`zs`, `qsum` and `sum(Q8)` are each lane's own partial; the row score sum turns
them into the slice totals. Only `sq_s` is slice-wide. The lane model retains a
negative control that multiplies the slice constant to detect this regression.
The widest possible integer dot is `256 * 15 * 126 = 483 840`, inside int32.

## What stays exact

Stage rows, the ORIGINAL-F16 tail ring, the non-tail residual rows, the online
softmax, the WHT, the split ranges and the partial layout are untouched — they
keep the F32 path and the wire contract. Only the sealed-record rows of the fast
shared tile change, and the host sees the same root signature, the same 12 cbuffer
constants, the same dispatch and the same partials.

## Prefix reduction and validation

Every lane contributes its own partial dot and constant. `WaveActiveMax` over
copies of each slice supplies the quantizer scale. `WavePrefixSum(val) + val`
forms inclusive prefixes; broadcasts at lanes 15/31/47/63 and differences of
consecutive endpoints produce the four row scores. This is the only INT8 path.
F32 cancellation between prefixes is additional to Q8 rounding.

The operator measured pass1 −17.1%, prefill throughput +7.96%, 256/256 matching
tokens and Q8 error about 1%. See [the acceptance policy](rkva-final-optimizations.md)
for attribution and limits. Historical defects and rejected experiments are
preserved in [the numerical audit](agent-reports/astra-int8-review.md) and
[the earlier investigation](agent-reports/codex-int8-fix.md).

`rkva_mq_i8_quality.py` checks the slice algebra. `rkva_mq_i8_lane_model.py`
checks lane mapping, slice maxima, prefix row sums and the constant-multiplication
negative control. `rkva_mq_i8_dxil_model.py` interprets the compiled fast QK path
and requires uniform broadcast indices when `--require-uniform-reads` is passed.
These CPU checks do not emulate the GPU driver or qualify model/logit quality.

### Quality gate

`tools/xbox/tests/rkva_mq_i8_quality.py` compares three float64/float32 paths
over the same store: GOLD (original F16 K/V), REF (the shipped 4-bit F32 kernel
arithmetic) and I8 (this variant). Metric of record: `added/ref`, the Q8
rounding error as a fraction of the score error the 4-bit cache already
introduces. Historical v1 result (the L1 domain was corrected in v2 below). Across 10 cases (prefixes 256..4096, two store distributions, 8
queries each):

```
add/ref 0.062 .. 0.076     total/ref 0.999 .. 1.012
rel_o_rms, max|O|, 1-cos, max|dLSE| and softmax-weight L1 all within 1.15x of REF
rkva_mq_i8_quality: 10 gated cases (dc exact, 8 queries each), PASS
```

Without the `C_s` DC term the same harness reports `add/ref 0.12 .. 0.24`, i.e.
the constant is worth ~2.5x.

### Metric contract v2

The JSON schema is now `rkva-mq-i8-quality-v2`: `weight_l1` compares
normalized probabilities on the **full records + tail** domain in all three
paths. V1 normalized REF/I8 over records alone but compared with GOLD record
probabilities whose denominator included the tail. Even identical uniform
attention then appeared to have a nonzero error equal to tail mass, potentially
masking relative degradation. Do not compare v1/v2 L1 series as equivalent.

Existing `case.metrics.*.ref/i8` keys remain **means over independent synthetic
queries**, including the historical names `rel_o_max` and `lse_abs_max`.
`per_query_metrics` retains each [REF, I8] pair; `metric_distribution` adds
actual maximum, zero-based max-query index and p95 per case.
`gated_query_maxima` reports true global maxima across DC-exact cases with
case/query provenance. These distributions are observational, not new gates.

The acceptance thresholds are unchanged: added score RMS / REF-vs-GOLD RMS
<=0.35, total score RMS ratio <=1.15, and query-mean attention metrics <=1.15x
REF. Only L1's domain is corrected; passing a mean gate does **not** guarantee
each query stays within 1.15x. `added_over_ref` is mean RMS(I8−REF) divided by
mean RMS(REF−GOLD), not percent model-quality loss. No production flag, shader,
or hardware qualification is changed. Independent queries are not autoregressive
generation steps.

Run accounting regressions with:
```sh
OPENBLAS_NUM_THREADS=1 python3 tools/xbox/tests/test_rkva_mq_i8_quality_metrics.py
```
Readers expecting an exact v1 schema must explicitly migrate; numeric legacy
keys remain readable, but old L1 semantics are deliberately not preserved.
Repository search found no parser consuming this quality-report schema.


## Caveats

* Not bit-exact. Never a default; the flag plus this gate are the contract.
* SM 6.4 is required for `dot4add_i8packed`. The PSO is created on the console
  for the prefix artifact accepted by the operator.
* Algebra-only checks do not validate lane mapping or driver lowering; use
  the lane model, DXIL interpreter and console oracle as separate checks.
* The INT8 path needs exactly 64 lanes per wave (one 16-lane group per KV row of
  a half tile); unqualified calls must use the existing fallback.
* VGPR pressure rises (int8 Q8 is 48 VGPRs for 12 query-head pairs, offset by
  dropping the resident F32 rotated Q). Occupancy has not been measured; this is
  the main hardware risk.

## Hardware A/B

`tools/xbox/scripts/run-rkva-prefill-int8-ab.py` is inert without `--execute`.
It toggles `rkva-mq-i8.flag` through operator-supplied commands, runs
`rkva_kvarn_serial_prefix_probe.py` in both arms (OUT/LSE error against its
NumPy oracle — the numeric gate, tolerance 1.15x of the OFF arm) and
`bench-rkva-range.py` in both arms (`pass1_us` from the engine buckets, prefill
t/s, the 256 generated ids), then always restores the flag OFF.
