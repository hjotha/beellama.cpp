# KV calibration findings

Status: investigation in progress on GOKAYA (`192.168.1.57`). Updated after
the compute-aware probe, sentinel fix, and live `8091` tests.

All measurements below use CUDA0, `q4_0` K/V, `parallel=1`, unless stated
otherwise. A context value is the total per-slot budget: prompt plus output.

## Goal

Make `--ctx-size 0` report a usable context size for the four combinations:

1. normal/traditional KV without MTP;
2. normal/traditional KV with MTP;
3. paged KV without MTP;
4. paged KV with MTP.

The usable value must account for the actual `batch-size`/`ubatch-size`, model
compute buffers, and MTP draft resources, not only the KV byte estimate.

## Earlier matrix: Qwen3.8-27B-UD-IQ3_XXS, batch/ubatch 64, fit-target 643

| KV path | MTP | Reported candidate | Runtime result |
|---|---:|---:|---|
| traditional | no | 79,360 | model loaded/listening |
| traditional | yes | 53,504 | OOM allocating a 225.80 MiB compute buffer |
| paged | no | 77,824 | model loaded/listening |
| paged | yes | 43,008 | model loaded/listening |

The MTP traditional candidate was not usable despite the successful arithmetic.

## ISTA-DASLab GGQ-RCO-IQ3_XXS-mtp matrix, batch/ubatch 64, fit-target 643

Model:
`Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`

| KV path | MTP | Reported candidate | Runtime result |
|---|---:|---:|---|
| traditional | no | 105,472 | startup passed; maximum request later OOMed |
| traditional | yes | 78,080 | OOM during context creation |
| paged | no | 104,448 | pool froze, then final initialization OOMed |
| paged | yes | 67,584 | startup and fixed-pool calibration passed |

The high reported limits are therefore not automatically usable request
limits. The 67,584 MTP-paged case also became impractically slow in the
reference paged-attention path during a very large prefill request.

## Compute-aware matrix: GSQ-RCO MTP model, batch/ubatch 64, fit-target 643

The calibration now probes a small real target context and, when enabled, the
MTP draft context. It measures the fixed context/compute overhead before
extrapolating the per-token or per-block KV cost.

| KV path | MTP | Probe overhead | New candidate | Startup result |
|---|---:|---:|---:|---|
| traditional | no | 165.2 MiB | 96,000 | model loaded/listening |
| traditional | yes | 486.0 MiB | 51,968 | model loaded/listening |
| paged | no | 164.9 MiB | 94,208 | model loaded/listening |
| paged | yes | 484.2 MiB | 41,984 | model loaded/listening |

The probe logs include `batch=64` and `ubatch=64`. The `fit-target` is
subtracted separately (`643 MiB` here); MTP adds its explicit workspace
reserve. These are safer startup candidates than the prior KV-only values,
but a maximum request should still be used as the final gate.

## Final automatic calibration matrix: ISTA GGUFs, batch/ubatch 64, fit-target 643

The automatic path now accounts for the actual GGUF loaded in each process.
The model size, free VRAM, KV geometry, and probe result are not reused
between the MTP and non-MTP files. The probe treats the `--ctx-size 0` fit
sentinel as a 4096-token floor.

| GGUF | KV path | MTP | Free VRAM | Probe overhead | Effective overhead | Candidate context |
|---|---|---:|---:|---:|---:|---:|
| `...-mtp.gguf` | traditional | yes | 2,165.6 MiB | 502.4 MiB | 502.4 MiB | 54,272 |
| `...-mtp.gguf` | paged | yes | 2,165.6 MiB | 480.9 MiB | 400.7 MiB | 60,064 |
| `...XXS.gguf` | traditional | no | 2,499.6 MiB | 176.4 MiB | 144.4 MiB | 97,280 |
| `...XXS.gguf` | paged | no | 2,499.6 MiB | not used | 0 MiB | 105,584 |

The automatic candidates now match the manually measured functional limits
except for the MTP-paged candidate, which is intentionally 96 tokens below
the absolute 60,160-token boundary. Both automatic paged candidates reached
`model loaded` and completed their maximum request with `initial/growth=64`;
the older fixed-pool freeze path was removed because it severely reduced
throughput. The service on `8090` was intentionally left stopped while the
tests ran; no production unit was changed.

The automatic MTP-paged request at `ctx=60,064` used a 55,968-token prompt,
returned HTTP 200 with 4,094 generated tokens, and measured 474.22 prompt
tok/s, 30.41 decode tok/s, and 252.604 s total time. The automatic non-MTP
paged request at `ctx=105,584` used a 101,488-token prompt, returned HTTP 200
with 4,096 generated tokens, and measured 441.97 prompt tok/s, 17.38 decode
tok/s, and 465.297 s total time.

## Direct request checks

For the previous UD model, traditional KV without MTP was tested with a
56,320-token context, 52,223 prompt tokens, and 4,096 output tokens. It
completed 4,096 output tokens without truncation.

For the GSQ-RCO model, a traditional no-MTP request at the reported 105,472
candidate failed with CUDA OOM. An MTP-traditional request at 78,080 could not
start because context creation failed first. These checks show that the
KV-only candidate is not sufficient as a production threshold.

The GSQ-RCO MTP-paged path was also tested at an explicit 44,032-token context
with 39,936 prompt tokens and a 4,096-token output limit. It completed 4,094
output tokens and reached:

```text
prefill: 496.71 tokens/s (80.401 s / 39,936 tokens)
decode:   36.97 tokens/s (110.706 s / 4,094 tokens)
total:   191.108 s / 44,030 tokens
```

The server reported `truncated=1` at the exact boundary, so 44,032 is a
working threshold but not yet a clean full-4,096-token boundary. The new
compute-aware candidate of 41,984 leaves additional headroom.

The same MTP-paged path was tested at the new 41,984-token candidate with a
37,888-token prompt and a 4,096-token output limit. It completed 4,094 output
tokens without OOM, at approximately 502.34 prompt tokens/s and 38.25 decode
tokens/s; total logged time was 182.435 s for 41,982 processed tokens. The
server again marked the exact boundary as `truncated=1`, so client-side output
reserve should leave a small additional margin when strict completion of all
4,096 tokens is required.

The MTP-traditional candidate was then tested at the exact `50,944`-token
context boundary with a `46,848`-token prompt (`50,944 - 4,096`) and
`max_tokens=4096`. The request returned HTTP 200 without OOM, with
`prompt_tokens=46848`, `completion_tokens=4094`, `total_tokens=50942`, and
`finish_reason=length`. The server metrics recorded 566.24 prompt tokens/s,
48.03 decode tokens/s, and 167.948 s total time. The two-token difference is
the server's boundary/protocol behavior, so the candidate is operationally
valid but does not yield a literal 4,096 generated tokens at the exact
context edge.

## Absolute MTP-traditional request boundary

To separate the startup limit from the usable request limit, the same server
configuration was tested with `prompt = ctx-size - 4,096` and
`max_tokens=4096`. The search narrowed the transition to one token:

| `ctx-size` | Prompt | Result |
|---:|---:|---|
| 53,120 | 49,024 | HTTP 200, 4,096 generated |
| 54,208 | 50,112 | HTTP 200, 4,096 generated |
| 54,240 | 50,144 | HTTP 200, 4,096 generated |
| 54,256 | 50,160 | HTTP 200, 4,096 generated |
| 54,264 | 50,168 | HTTP 200, 4,096 generated |
| 54,272 | 50,176 | HTTP 200, 4,094 generated; boundary truncation |
| 54,273 | 50,177 | CUDA OOM during request |

`54,272` is therefore the highest context that accepted this maximum-size
request in two independent runs; `54,273` was the first failing integer. If
the requirement is literally all 4,096 generated tokens, `54,264` is the
highest value validated. This is an operational absolute for the tested
model, GPU state, MTP settings, q4_0 KV, and batch/ubatch 64—not a hardware
invariant; changing any of those can move the boundary.

## Paged MTP memory versus throughput boundary

The paged search used dynamic growth with a 16-token block and enough
`--n-gpu-blocks`/`--kv-paged-admission-blocks` for the requested context. This
is distinct from the calibrated fixed pool of 41,984 tokens; omitting those
capacity values causes an artificial `Context size has been exceeded` error.

| `ctx-size` | Prompt | Result |
|---:|---:|---|
| 41,984 | 37,888 | HTTP 200, 4,094 generated |
| 44,032 | 39,936 | HTTP 200, 4,094 generated |
| 54,272 | 50,176 | HTTP 200, 4,094 generated |
| 59,392 | 55,296 | HTTP 200, 4,094 generated |
| 60,416 | 56,320 | CUDA OOM during prefill |
| 63,488 | 59,392 | CUDA OOM during prefill |
| 67,584 | 63,488 | CUDA OOM during prefill |

The maximum paged context that passed the full request test was therefore
59,392 tokens, but it is not a usable replacement target: its measured
throughput fell to 475.18 prompt tok/s and 30.45 decode tok/s, versus
496.71 prompt tok/s and 36.97 decode tok/s at 44,032. Since the reference
paged-attention implementation shows a clear throughput collapse as the
context grows, `44,032` is the current practical ceiling; `41,984` remains the
safer calibrated value. The memory ceiling and the performance ceiling must
not be reported as the same number.

With the explicit collapse rule of decode throughput below `10 tok/s`, the
last stable paged-MTP point was refined to `60,160` tokens. Its request used a
56,064-token prompt and completed with `30.11` decode tok/s. The next paged
allocation step, `60,176` tokens, required one additional 16-token block and
repeatedly failed to allocate approximately 63 MiB on CUDA before decode
could start. Thus `60,160` is the last validated non-collapsed value under
this exact configuration; the failure is an OOM boundary rather than a
measured sub-10 tok/s decode.

## Traditional versus paged MTP performance

At the same `ctx-size=54,272`, with the same 50,176-token prompt and
`max_tokens=4096`, the measured comparison was:

| Metric | Traditional KV | Paged KV | Paged delta |
|---|---:|---:|---:|
| Prefill | 549.13 tok/s | 485.01 tok/s | 11.7% slower |
| Decode | 46.66 tok/s | 32.26 tok/s | 30.9% slower |
| Total time | 179.10 s | 230.32 s | +51.2 s |

At each path's maximum validated context, traditional KV reached 54,272
tokens at 549.13 prompt tok/s and 46.66 decode tok/s, while paged KV reached
60,160 tokens at 471.00 prompt tok/s and 30.11 decode tok/s. Paged KV gained
5,888 context tokens, but lost approximately 78 prompt tok/s and 16.5 decode
tok/s. These values are for the current build, where the log reports that
paged attention is using the CPU reference implementation for correctness
validation; it is explicitly not optimized. The measured speed penalty is
therefore not an inherent limit of paged KV and may change with an optimized
backend.

No completed paged request reached the defined collapse threshold of decode
below 10 tok/s; the higher-context failures occurred first as CUDA OOM.

## ISTA GGUF without MTP: absolute request limits

The non-MTP ISTA file is
`Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf`. The same maximum-request test was used in
both KV modes: `prompt = ctx-size - 4,096`, `max_tokens=4096`, batch/ubatch
64, q4_0 KV, CUDA0, and no `--spec-*` flags.

For traditional KV, the search ended at:

| `ctx-size` | Prompt | Result |
|---:|---:|---|
| 95,488 | 91,392 | HTTP 200, 4,096 generated |
| 96,736 | 92,640 | HTTP 200, 4,096 generated |
| 97,048 | 92,952 | HTTP 200, 4,096 generated |
| 97,204 | 93,108 | HTTP 200, 4,096 generated |
| 97,243 | 93,147 | HTTP 200, 4,096 generated |
| 97,272 | 93,176 | HTTP 200, 4,096 generated |
| 97,277 | 93,181 | HTTP 200, 4,096 generated |
| 97,280 | 93,184 | HTTP 200, 4,096 generated |
| 97,281 | 93,185 | CUDA OOM during request |

`97,280` is the highest traditional-KV context validated, with `456.42`
prompt tok/s, `21.19` decode tok/s, and `397.444 s` total time.

For paged KV, dynamic growth was enabled and the physical/admission pool was
set to the number of 16-token blocks needed by each candidate:

| `ctx-size` | Prompt | GPU blocks | Result |
|---:|---:|---:|---|
| 94,208 | 90,112 | 5,888 | HTTP 200, 4,096 generated |
| 99,328 | 95,232 | 6,208 | HTTP 200, 4,096 generated |
| 101,888 | 97,792 | 6,368 | HTTP 200, 4,096 generated |
| 104,448 | 100,352 | 6,528 | HTTP 200, 4,096 generated |
| 105,472 | 101,376 | 6,592 | HTTP 200, 4,096 generated |
| 105,536 | 101,440 | 6,596 | HTTP 200, 4,096 generated |
| 105,568 | 101,472 | 6,598 | HTTP 200, 4,096 generated |
| 105,584 | 101,488 | 6,599 | HTTP 200, 4,096 generated |
| 105,600 | 101,504 | 6,600 | CUDA OOM during prefill |

`105,584` is the highest paged-KV context validated, with `439.02` prompt
tok/s, `17.27` decode tok/s, and `468.267 s` total time. It adds 8,304
tokens over traditional KV, while decode is approximately 3.92 tok/s slower.
The paged-attention path still uses the unoptimized CPU reference
implementation noted above, so this speed difference is implementation-

## Unsloth UD GGUF: automatic MTP calibration

The local Unsloth model is
`Qwen3.8-27B-UD-IQ3_XXS.gguf`. It is a different, larger GGUF from the two
ISTA files above: with the same settings it left `1707.6 MiB` free on CUDA0,
whereas the ISTA MTP file left `2165.6 MiB`. The automatic calibration must
therefore measure this file independently.

With `ctx-size=0`, `fit-target=643`, batch/ubatch 64, q4_0 KV, and MTP enabled,
the automatic candidates and maximum-request checks were:

| KV path | Candidate | Prompt | Result | Prefill | Decode |
|---|---:|---:|---|---:|---:|
| traditional | 29,696 | 25,600 | HTTP 200, 4,094 generated | 660.04 tok/s | 57.16 tok/s |
| paged | 35,536 | 31,440 | HTTP 200, 4,094 generated | 503.37 tok/s | 43.72 tok/s |

Both requests completed without OOM. The paged candidate uses 2,221 GPU
blocks with `initial/growth=64`; its lower context than the ISTA MTP result is
expected from the larger Unsloth GGUF and its lower free-VRAM budget.

## Calibration implementation observations

- The normal KV estimator now counts only regular attention layers for hybrid
  target models and adds appended `nextn` layers when MTP is enabled.
- `fit-target` is a VRAM safety margin. With `--fit-target 643`, normal MTP
  fitting adds a measured 5 MiB boundary margin; the old 64 MiB duplicate
  margin and 224 MiB paged speculative reserve are no longer charged.
- `batch-size` and `ubatch-size` do not appear directly in the KV byte formula,
  but they change compute-graph memory. The new probe measures that fixed
  overhead using the configured values before calculating the usable
  threshold.
- There is no separate `block unlock` option in this source. Paged automatic
  fitting reserves the candidate capacity without running the attention kernel
  during startup, then keeps `initial/growth=64` and sets the admission cap.
  This avoids both startup OOM aborts and the throughput collapse caused by
  freezing the full pool as the initial allocation.

## Threshold validation status

The first conservative functional threshold for the GSQ-RCO MTP-paged model
was around 44k tokens, based on the previous model's successful 43,008-token
MTP-paged calibration. Requests at 44,032 and 41,984 tokens completed without
OOM, although the server stopped at 4,094 generated tokens at the exact
boundary. The MTP-traditional candidate at 50,944 shows the same two-token
boundary behavior. A small client-side reserve below the advertised context
limit is therefore still advisable when exactly 4,096 generated tokens are
required.

## Unsloth UD GGUF: fit-off absolute boundary

The local Unsloth/UD file was then tested with automatic fitting disabled,
batch/ubatch 64, q4_0 KV, CUDA0, MTP enabled, and a request shaped as
`prompt = ctx-size - 4,096`, `max_tokens=4096`, and `ignore_eos=true`.

For traditional KV, the last operational context was:

| `ctx-size` | Prompt | Result |
|---:|---:|---|
| 37,116 | 33,020 | HTTP 200, 4,096 generated |
| 37,120 | 33,024 | HTTP 200, 4,094 generated; boundary truncation |
| 37,121 | 33,025 | CUDA OOM during request |

At `ctx-size=37,120`, prefill was `621.17` tok/s, decode was `53.71`
tok/s, and total time was `129.376 s` for `37,118` processed tokens. If a
literal four-thousand-and-ninety-six-token completion is required, `37,116`
is the highest value validated in this run.

For paged KV, the production-sized pool was tested with 2,560 blocks of 16
tokens each:

| `ctx-size` | Prompt | GPU blocks | Result |
|---:|---:|---:|---|
| 40,960 | 36,864 | 2,560 | HTTP 200, 4,094 generated; boundary truncation |
| 40,976 | 36,880 | 2,561 | request dropped during prefill |
| 41,984 | 37,888 | 2,624 | CUDA OOM during request |
| 45,056 | 40,960 | 2,816 | CUDA OOM during request |

At `ctx-size=40,960`, the request measured `500.92` prompt tok/s,
`39.58` decode tok/s, and `176.998 s` total time for `40,958` processed
tokens. Decode remained well above the `10 tok/s` collapse threshold; the
next paged allocation step failed before a usable response. Therefore
`40,960` is the last validated fit-off paged context for this Unsloth file
and current GPU/configuration.

## Why production used 40,960 with max_tokens 4,094

The production unit used the following relevant limits:

```text
--ctx-size 40960 --batch-size 128 --ubatch-size 128
--kv-paged --kv-paged-dynamic --kv-block-size 16
--n-gpu-blocks 2560 --n-gpu-blocks-initial 64 --n-gpu-blocks-growth 64
--kv-paged-admission-blocks 2560 --fit off
```

The primary reason it worked was the explicit paged capacity: `2,560 * 16
= 40,960` tokens. With a 36,864-token prompt, the MTP slot's
`n_ctx - prompt - 2` guard leaves at most `4,094` generated tokens. The
server source documents that the two-token reserve is needed for the sampled
token and a possible context-shift boundary. This is why a request asking for
4,096 can finish with 4,094 at the exact edge; it is not evidence that the
client's `max_tokens` value was ignored.

Batch and ubatch 128 were relevant to compute workspace size and throughput,
and can move an OOM boundary, especially for traditional KV. They did not
define the 40,960 context window: the context and block/admission limits did.
In particular, `--fit off` meant that production was using a fixed, explicit
budget rather than relying on automatic calibration to choose the limit.

## What is and is not model-independent

There is no exact universal closed-form context formula for these models. The
KV bytes per token/block are calculable, but the remaining budget includes
model weights, MTP target/draft state, CUDA graph/workspace allocations,
backend fragmentation, and batch/ubatch-dependent temporary buffers. Those
effects are both model- and configuration-dependent.

The result is also discrete: normal KV is padded to backend graph
granularities, paged KV allocates 16-token blocks, and dynamic growth uses
64-block steps in the tested configuration. A robust calibrator therefore
needs to measure the loaded model and configured batch, then round down to a
safe allocation unit and validate the request shape. It should not select a
constant based on the filename or assume that the Unsloth and ISTA files have
the same free-VRAM budget.

The current implementation uses per-model free-VRAM and runtime probe data,
not a filename-specific Unsloth/ISTA branch. Latest automatic checks still
kept the ISTA candidates at 54,272/60,064 for MTP traditional/paged and
97,280/105,584 for no-MTP traditional/paged, while the Unsloth fit-on
candidates were 29,696/35,536. Those numbers differ because the loaded files
and runtime budgets differ. The small correction terms in the current patch
remain backend calibration heuristics, so the implementation should be
treated as conservative and configuration-specific until a fully safe
allocation probe replaces them.

## ISTA MTP traditional production validation on port 8090

The production unit was switched to
`Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` with traditional KV, `ctx-size=54264`,
batch/ubatch 64, q4_0 K/V, `fit off`, and the same MTP settings
(`n-max=2`, `p-min=0.80`). The runtime padded the slot to `n_ctx_slot=54272`.

Three sequential requests were sent to `http://127.0.0.1:8090/v1/completions`
with `temperature=0`, `ignore_eos=true`, and `max_tokens=4096`:

| Case | Prompt tokens | Completion | Total | HTTP | Decode | Result |
|---|---:|---:|---:|---:|---:|---|
| small | 128 | 4,096 | 4,224 | 200 | 69.63 tok/s | pass |
| medium | 8,192 | 4,096 | 12,288 | 200 | 64.72 tok/s | pass |
| large | 50,168 | 4,096 | 54,264 | 200 | 46.98 tok/s | pass |

The large request used the validated literal-4,096 boundary and completed
without truncation or CUDA OOM. Its measured prefill was `510.54` tok/s and
total request time was `161.359 s` (the shell wall time was `162 s`). The
medium and large requests reused prompt-cache tokens from the preceding
requests, so their prompt timing reflects cached-prefix reuse.

After validation, `llama-server.service` remained active on port 8090 with
the ISTA MTP traditional configuration. No temporary test server remained on
the GPU.

## Stock upstream comparison

The upstream stock binary was fetched from `ggml-org/llama.cpp` at commit
`67a17c17c` and built with CUDA for `sm_89`. The fork binary was the existing
CUDA build used by the production unit, with the same model and runtime
parameters. The production unit was stopped only while the stock process used
the GPU on port 8091; it was started again after the comparison.

Both binaries used the ISTA MTP traditional configuration with `ctx-size`
54264, batch/ubatch 64, q4_0 K/V, `parallel=1`, Flash Attention, and
`n-max=2`, `p-min=0.80`. The same three sequential prompts and
`max_tokens=4096`, `temperature=0`, `ignore_eos=true` were used:

| Case | Fork prefill | Stock prefill | Fork decode | Stock decode | Stock decode delta | Result |
|---|---:|---:|---:|---:|---:|---|
| small, 128 prompt | 217.94 | 323.45 | 69.63 | 69.67 | +0.053% | both pass |
| medium, 8,192 prompt | 706.39 | 710.70 | 64.72 | 64.79 | +0.104% | both pass |
| large, 50,168 prompt | 510.54 | 510.11 | 46.98 | 47.23 | +0.532% | both pass |

All six requests returned HTTP 200 and exactly 4,096 completion tokens. The
stock total times were 0.375%, 0.145%, and 0.247% lower for small, medium,
and large respectively. The small-prompt prefill difference is dominated by
fixed-request timing; at the large boundary the prefill difference was only
`-0.085%`. The stock and fork had identical MTP draft counts and acceptance
for these runs, so there was no material TPS regression from the fork changes.
