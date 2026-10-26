# RKVA prefill INT8 QK kernel (`attn_kvarn4_mq2_i8`)

Opt-in, **not bit-exact** specialization of the native Bee KVarN4 2-query prefill
tile used by the Xbox frontend for positional overflow attention (`pass1` of
prefill). It computes the fast shared record tile's QK with
`dot4add_i8packed` (SM 6.4) instead of a 4-dim/lane F32 dot.

* Shader: `tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq_i8.hlsl` (wrapper,
  sets `RKVA_MQ_I8 1`) over `rkva_range_attn_kvarn4_mq.hlsl`.
* DXIL: `tools/xbox/shaders/generated/rkva_range_attn_kvarn4_mq2_i8_dxil.h`
  (`cs_6_4`, 143 620 B).
* Build: `tools/xbox/scripts/compile-rkva-range-shaders.sh attn_kvarn4_mq2_i8`.
* Flag: LocalState file `rkva-mq-i8.flag` (default off), read at HELLO next to
  `rkva-mq-rescale-skip.flag`. Selection: `remote-attn.cpp` →
  `RemoteAttnEngine::set_mq_i8()` → `attend_range()`.
* Bucket: `attention/kvarn4-prefix/optimized-kvarn4_mq2_i8/<nq>/<rows>/warm`;
  counters `kvarn4_multi_query.mq2_i8.{available,status,calls,exact:false}`.
* Quality gate: `tools/xbox/tests/rkva_mq_i8_quality.py`.
* Hardware A/B: `tools/xbox/scripts/run-rkva-prefill-int8-ab.py` (inert).

## Why the reduction, not the dot, is the target

The shipped `attn_kvarn4_mq2` tile gives every lane 4 dims of a 256-dim head and
needs a full 64-lane `WaveActiveSum` per score: 12 instructions of a 512
instruction/KV-row budget (28%) are cross-lane reduction, while the QK dot
itself is one `dot4` plus one multiply. `dot4add_i8packed` does not change that
arithmetic: both intrinsics retire 4 MACs per lane per instruction, so the same
12 instructions per lane cover the same 96 QK scores. The win comes from int8's
density letting a lane hold **16 dims** instead of 4, which shrinks the
reduction disc from 64 to 16 lanes (4 butterfly steps) and lets four KV rows
reduce in the same instruction stream.

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
lane_value = (float(dot) * sq_s + C_s) * other_s[p]
score      = kq * sum over the 16 lanes of the group
```

`nibble` is 0..15, i.e. already a valid non-negative int8 lane, so the dp4a
operand needs no offset. `C_s` is the important term: it carries the *exact* DC
part of the Q8 rounding error, so the residual is `-sum_d (Q'-sq*Q8)*(nibble-8)`
instead of `-sum_d (Q'-sq*Q8)*nibble`. That single constant cuts the added score
error ~2.5x at no per-position cost.

`sum(Q8)` per slice is reduced at quantization time. The widest possible score
is `256 * 15 * 126 = 483 840`, far inside int32.

## What stays exact

Stage rows, the ORIGINAL-F16 tail ring, the non-tail residual rows, the online
softmax, the WHT, the split ranges and the partial layout are untouched — they
keep the F32 path and the wire contract. Only the sealed-record rows of the fast
shared tile change, and the host sees the same root signature, the same 12 cbuffer
constants, the same dispatch and the same partials.

## Measured instruction count

`tools/xbox/tests/rkva_dxil_loop_cost.py` (DXIL class model, wave64 issue slots,
one 8-KV-row tile, 2 queries x 6 heads), `cs_6_0` baseline vs `cs_6_4` variant:

| build | instr/KV row | fma_class | reduce | sfu | int |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attn_kvarn4_mq2` (shipped) | 512.0 | 237.0 | 144.0 | 15.0 | 89.1 |
| `attn_kvarn4_mq2_i8` | **331.9** | 198.2 | 24.0 | 15.0 | 81.6 |

`-35.2%` instructions per KV row. The cross-lane reduction drops from 144 to 24
(-120), the F32 class from 237.0 to 198.2 (int8 dot replaces the K dequant),
and the per-group quantizer sits outside the tile loop (replacing it with
constants changes the measured loop by 37 of 4 019 instructions), amortized over
128 rows. `tools/xbox/tests/rkva_dxil_loop_cost.py` gained `dot4AddPacked` and
`waveReadLaneAt` accounting plus a proper natural-loop (reverse-reachability)
bound for this measurement.

## Quality gate

`tools/xbox/tests/rkva_mq_i8_quality.py` compares three float64/float32 paths
over the same store: GOLD (original F16 K/V), REF (the shipped 4-bit F32 kernel
arithmetic) and I8 (this variant). Metric of record: `added/ref`, the Q8
rounding error as a fraction of the score error the 4-bit cache already
introduces. Across 10 cases (prefixes 256..4096, two store distributions, 8
queries each):

```
add/ref 0.062 .. 0.076     total/ref 0.999 .. 1.012
rel_o_rms, max|O|, 1-cos, max|dLSE| and softmax-weight L1 all within 1.15x of REF
rkva_mq_i8_quality: 10 gated cases (dc exact, 8 queries each), PASS
```

Without the `C_s` DC term the same harness reports `add/ref 0.12 .. 0.24`, i.e.
the constant is worth ~2.5x.

## Caveats

* Not bit-exact. Never a default; the flag plus this gate are the contract.
* SM 6.4 is required for `dot4add_i8packed`; the console PSO probe accepted a
  `dot4_i8` compute PSO, but this specific PSO has not been created on hardware
  yet.
* The INT8 path needs exactly 64 lanes per wave (one 16-lane group per KV row of
  a half tile); other wave sizes poison the outputs like the existing shape gate.
* VGPR pressure rises (int8 Q8 is 48 VGPRs for 12 query-head pairs, offset by
  dropping the resident F32 rotated Q). Occupancy has not been measured; this is
  the main hardware risk.
* The exact rescale elision of `attn_kvarn4_mq2_skip` is deliberately not
  combined: with both softmax arms present the static tile loop grows from 331.9
  to 502.4 instructions per KV row, more than the elision can save.

## Hardware A/B

`tools/xbox/scripts/run-rkva-prefill-int8-ab.py` is inert without `--execute`.
It toggles `rkva-mq-i8.flag` through operator-supplied commands, runs
`rkva_kvarn_serial_prefix_probe.py` in both arms (OUT/LSE error against its
NumPy oracle — the numeric gate, tolerance 1.15x of the OFF arm) and
`bench-rkva-range.py` in both arms (`pass1_us` from the engine buckets, prefill
t/s, the 256 generated ids), then always restores the flag OFF.
