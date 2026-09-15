# OOM reproduction trace

Status: the shape-stability fix remains valid, but it did not remove a large
IQ1_M cuBLAS dequantization workspace competing with CUDA executable graph
reservations. A bounded IQ1_M MMVQ fallback now passes numerical and live
regressions without reducing context or disabling graphs; the September 5
follow-up below records the evidence and full replay validation. This record
applies to the RTX 4070 production model and settings, not a general CUDA limit.

## Configuration

- Model: `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`
- GPU: CUDA0, GeForce RTX 4070, 12,282 MiB reported by NVML, 11,902.12 MiB allocatable in CUDA
- KV: traditional, K/V `q4_0`, `--kv-paged` absent
- MTP: `draft-mtp`, `n-max=2`, `p-min=0.80`
- Context: `--ctx-size 54272`, one slot
- Compute: `--batch-size 64`, `--ubatch-size 64`, Flash Attention
- `--fit off`
- Power governor: prefill 200 W, decode 165 W
- Prompt cache: tested with both `--cache-ram 4096` and `--cache-ram 0`

All prompt sizes in this document were measured by the running server's
`POST /tokenize` endpoint. Chat sizes were measured after rendering through
`POST /apply-template`.

## Original failure

The earlier OpenCode request reached 38,309 prompt tokens and aborted during
prefill:

```text
ggml_cuda_compute_forward: MUL_MAT failed
CUDA error: out of memory
current device: 0, in function launch_mul_mat_q
```

The process then produced a core dump and systemd restarted it. A later
controlled reproduction reached the same failure in a chat request at 50,090
processed tokens, with the stack including:

```text
ggml_cuda_graph_evaluate_and_capture
llama_context::graph_compute
llama_context::process_ubatch
llama_context::decode
server_context_impl::update_slots
```

The client observed `curl: (52) Empty reply from server`; systemd recorded a
core dump and incremented the restart counter.

## Sequence that reproduces it

After a clean server start, the following completion requests all succeeded,
in this order. Each request used `cache_prompt=true` and the output budget
shown below:

| Step | Endpoint | Prompt tokens | `max_tokens` | Result |
|---:|---|---:|---:|---|
| 1 | completion | 36,864 | 64 | pass |
| 2 | completion | 38,308 | 4,096 | pass |
| 3 | completion | 50,176 | 4,096 | pass; 4,094 at boundary |
| 4 | completion | 43,008 | 256 | pass |
| 5 | completion | 48,128 | 1,024 | pass |
| 6 | completion | 24,576 | 256 | pass |
| 7 | completion | 40,960 | 512 | pass |
| 8 | completion | 32,768 | 1,024 | pass |
| 9 | chat | 38,359 | 4,096 | pass; model stopped at 151 |
| 10 | chat | 50,175 | 4,096 | OOM, empty reply, server restart |

The same final chat request passed as the first request after a clean start,
and also passed after only `chat 38,359 -> chat 50,175`. The failure therefore
depends on accumulated allocations/graph shapes rather than the request size
alone.

## Cache A/B

The complete sequence above was run with `--cache-ram 4096` and again with
`--cache-ram 0`.

- With `4096`, the sequence failed at the final chat request. Prompt-cache
  entries of approximately 1,032--1,372 MiB were created and evicted, and
  process RSS reached several GiB.
- With `0`, the same sequence still failed at the final chat request with the
  same CUDA OOM/core-dump signature. RSS stayed around 1.34--1.70 GiB and
  device memory remained around 11,901 MiB before the failure.
- A separate `4096` test confirmed real cache reuse (`cache_n=50172`) without
  OOM.

Thus prompt cache can add host-memory pressure and state-copy work, but it is
not the necessary trigger for this failure.

## CUDA graph A/B

The same eight completions followed by the two chats were run with
`--cache-ram 4096` and the environment variable:

```text
GGML_CUDA_DISABLE_GRAPHS=1
```

All ten requests passed. The final chat processed 50,175 prompt tokens and
stopped after 140 generated tokens. The process remained alive with
`NRestarts=0`; device memory stayed at approximately 11,881--11,883 MiB.

This is the strongest current evidence that the failure is in accumulated
CUDA graph capture/replay or device-allocation fragmentation after variable
graph shapes. It is not proof of a single leaked allocation: the CUDA
allocator may retain graph/workspace blocks by design, or the fork's graph
inputs may prevent correct reuse/eviction.

Before the backend fix, disabling context checkpoints was also tested
separately. With CUDA graphs enabled, `--cache-ram 4096`, and
`--ctx-checkpoints 0`, the process crashed during the third request of the
sequence. The log contained zero context-checkpoint events. Checkpoints can
add memory pressure, but they were not the root cause for this model and
reproduction.

### Post-fix validation with `--ctx-checkpoints 0`

On 2026-09-05, the current binary (`acf5ea5b0`) was tested from a clean
server start with CUDA graphs enabled, `--cache-ram 4096`, the same RTX 4070
configuration, and `--ctx-checkpoints 0`. The complete sequence passed:

| Step | Endpoint | Prompt tokens | Generated | Result |
|---:|---|---:|---:|---|
| 1 | completion | 36,864 | 64 | HTTP 200 |
| 2 | completion | 38,308 | 4,096 | HTTP 200 |
| 3 | completion | 50,176 | 4,094 | HTTP 200 |
| 4 | completion | 43,008 | 256 | HTTP 200 |
| 5 | completion | 48,128 | 1,024 | HTTP 200; 36,927 cached |
| 6 | completion | 24,576 | 256 | HTTP 200 |
| 7 | completion | 40,960 | 512 | HTTP 200 |
| 8 | completion | 32,768 | 1,024 | HTTP 200 |
| 9 | chat | 38,359 | 386 | HTTP 200; stop |
| 10 | chat | 50,175 | 149 | HTTP 200; stop |

The server remained alive for all ten requests, with no CUDA OOM, core dump,
empty reply, or restart. The test log contained no persistent context
checkpoint events, confirming that the result exercises the `0` setting. The
previous `ctx-checkpoints 0` failure is therefore fixed; the setting is not
itself an invalid or crashing mode. These passes validate capture/update stabilization for that sequence; they
do not establish allocation fragmentation as the root cause or prove
permanent VRAM headroom.

### Follow-up test with `--ctx-checkpoints 2`

The same current binary was then tested with `--ctx-checkpoints 2`. The first
eight completion cases and the 38,359-token chat completed successfully. The
final temporary chat run was interrupted before its result could be recorded.
After promotion to production, a fresh long chat request with approximately
50,000 prompt tokens caused a real failure: the server processed 49,945 tokens
and aborted in `cudaGraphInstantiate` with CUDA OOM. systemd recorded a core
dump and restarted the service (`NRestarts=1`).

The production configuration was immediately reverted to
`--ctx-checkpoints 1`, which remains the validated setting. This does not establish a direct VRAM cost for the second checkpoint.
Checkpoint payloads are serialized host-memory byte vectors; changing their
count also changes prefill segmentation and the timing of stable graph shapes. `--ctx-checkpoints 2` is therefore not promoted or recommended for
this deployment, even though the original graph-capture bug is fixed.

## Clean upstream comparison

The intact upstream binaries passed a first-request chat at 50,175 tokens:

- `llama2` build 10807, commit `163a40796`: pass
- upstream build 10805, commit `86b351fd6`: pass

The complete ten-request sequence was then run on the compiled `llama2`
upstream binary. It reproduced the same failure in the final chat at 50,090
processed tokens, with `MUL_MAT failed`, CUDA OOM, and a core dump. This rules
out the power governor and the fork's paged-KV additions as the source. The
fork and upstream builds used the same CUDA architecture (`89`), Flash
Attention, CUDA graphs enabled, and MMQ settings.

## Root cause and backend fix

Instrumentation was added behind `GGML_CUDA_GRAPH_DEBUG=1`. A short request
showed two graph-cache keys and that the first executable graph instantiation
reduced free device memory from 21.62 MiB to 5.62 MiB. This proves a 16 MiB executable-instantiation cost in that short run. It
does not distinguish driver reservations, workspace retention, fragmentation,
or cache growth across the full failing sequence, which lacked this telemetry.

Traditional KV rounds its active size in blocks while a long prompt is split
into 64-token ubatches. The same graph shape therefore appears briefly,
becomes eligible after one matching warmup call, is captured, and then changes
again as KV grows. Repeated captures can consume the already small device-memory margin.
The later investigation below identifies the large retained workspace that
was missing from the original explanation; no contiguous-block fragmentation
measurement was collected.

The backend fix makes three changes:

1. Require four structurally identical calls before enabling a CUDA graph.
   The short-lived prefill shapes never reach that threshold, while stable
   decode starts using graphs after a few tokens.
2. Avoid calling `cudaGraphExecUpdate()` immediately after first
   instantiation of the same captured graph.
3. Synchronize the graph stream before destroying and reinstantiating an
   executable graph after an incompatible update.

Optional diagnostics report graph key, cache-entry count, capture/update
counters, stable-call count, and CUDA free memory:

```text
GGML_CUDA_GRAPH_DEBUG=1
```

The corrected binary passed all ten requests with CUDA graphs enabled,
`--cache-ram 4096`, and `--ctx-checkpoints 1`. The same corrected binary also
passed the complete sequence with `--ctx-checkpoints 0`; the final 50,175-token
chat returned HTTP 200 in both runs, the process remained alive, and no
restart occurred.
Across the eight completion requests, the previous graph-enabled behavior
averaged 586.990 prompt tok/s and 49.113 decode tok/s; the fix averaged 586.261
and 49.107 tok/s respectively (approximately -0.12% and -0.01%). Production
was therefore restored with CUDA graphs enabled; `GGML_CUDA_DISABLE_GRAPHS`
is no longer set.

## External correlation

The upstream reports [#20315](https://github.com/ggml-org/llama.cpp/issues/20315),
[#23181](https://github.com/ggml-org/llama.cpp/issues/23181), and
[#25835](https://github.com/ggml-org/llama.cpp/issues/25835) describe the same
class of failure: varied requests eventually fail in CUDA graph capture or
instantiation, while `GGML_CUDA_DISABLE_GRAPHS=1` prevents the crash. The
current upstream cache uses the first graph-node pointer as a key and evicts
entries only by a time-to-live sweep. The
[NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
documents that graph executable resources and references can remain associated
with graph lifetime and that destroying a graph executable does not itself
synchronize prior work.

## Reproduction artifacts

The original runs recorded the following temporary paths. They are no longer
present after the September 5 reboot; recover their recorded outputs from the
Codex transcripts listed in the follow-up below:

- `/tmp/llama-random-suite.OdRmG3/cases.txt`
- `/tmp/llama-cache0-sequence.9Cwb0b/report.txt`
- `/tmp/llama-graphs-disabled-sequence2.kYFBxb/report.txt`
- `/tmp/llama-checkpoints0-sequence.8G2nA5/report.txt`
- `/tmp/llama2-upstream-sequence.l8ZaNH/report.txt`
- `/tmp/llama-graph-fix-sequence.7LfhAi/report.txt`
- `/tmp/llama-graph-debug-short.log`


## September 5 follow-up: IQ1_M workspace and CUDA executable reservation

The previous shape-stability fix (`acf5ea5b0`) is still present: four matching
calls before capture, no redundant update immediately after instantiate, and
stream synchronization before incompatible executable replacement. The later
OOM fallback prevents a fatal `cudaGraphInstantiate` OOM by disabling graphs
for that CUDA context. Neither change removes the prefill workspace pressure.

Instrumentation on the unchanged production settings (`ctx=54272`, batch and
ubatch `64`, cache RAM `3072`, checkpoints `1`, MTP `2`) found:

- The prefill leaves a **174 MiB VMM scratch pool** mapped, with zero logical
  bytes in use when capture begins. Only **10.06 MiB** CUDA memory remains.
- Four model matrices are `IQ1_M`, each with `5120 * 17408` elements. This type
  has MMVQ support for at most eight columns, but no MMQ support. At ubatch64,
  the old dispatch falls through to cuBLAS, which expands the entire matrix
  to FP16: about **170 MiB**, plus small temporaries and allocation rounding.
- A successful model graph instantiation consumes another **16 MiB**.
  Destroying cached graphs and scratch pools enabled a short decode, but the
  next long prefill then failed in `cuMemCreate` from `ggml_cuda_mul_mat_cublas`.
  Those experimental cleanup patches were discarded.
- A standalone 1000-kernel graph test consumed 8 MiB at first instantiate.
  `cudaGraphExecDestroy`, `cudaGraphDestroy`, synchronization, and
  `cudaDeviceGraphMemTrim` did not return it. Graph memory-node used/reserved
  attributes stayed zero. This is executable/driver memory, distinct from
  graph allocation-node pools. The behavior also has a
  [NVIDIA report](https://forums.developer.nvidia.com/t/cuda-graph-memory-reservations/233369).

The old instrumented short probe used about 2048 tokens and had 21.62 MiB
before instantiate (5.62 MiB after). The newer failing probe uses 3700 tokens
and has only 10.06 MiB. Prompt/shape history therefore matters; ten requests
are not a necessary precondition for OOM. The exact allocation breakdown of
the old 11.56 MiB advantage was not recorded and should not be invented.
Also, server `graphs reused` counts GGML graph reuse (`llama_perf_context`),
not CUDA Graph launches. The old full-suite counter alone cannot establish
that CUDA executable graphs were exercised throughout that suite.

### Bounded IQ1_M fallback

For ordinary contiguous 2D IQ1_M matrices that would otherwise use cuBLAS,
`ggml_cuda_mul_mat` now splits activation columns into batches of at most
eight and calls the existing IQ1_M MMVQ kernel. It does not introduce new
quantization math. Existing type, compute-view/padding, normal MMVQ, and MMQ
gates are preserved; other types and batched/broadcast/noncontiguous cases
keep their previous dispatch.

This keeps the configured context, cache RAM, and overall ubatch unchanged.
The defensive CUDA Graph OOM fallback remains available for other memory
pressure, but the validated IQ1_M repro no longer triggers it. Opt-in graph
diagnostics now expose VMM mapped/used bytes and actual launches after
capture, rather than relying on the GGML reuse counter.

Validation so far: build succeeded; 22 IQ1_M MUL_MAT comparisons against CPU
passed, including nine new boundary cases (odd row count33, k5120, columns
9/15/16/17/31/32/63/64/65). Five live prompts in short->long and long->short
order passed with real CUDA captures/launches, zero OOM/fallback, a 6 MiB
scratch pool and 164.06 MiB CUDA free after instantiate. The long prefill ran
at 593-600 tokens/s; the reference run at context54016 was602 tokens/s.
The eight historical completion sizes and two matching chat sizes all passed,
followed by an extra short request. The maximum completion produced 4094
tokens after a50176-token input (the4096-token budget hits the existing
context-boundary behavior). The two canary chats returned six tokens each,
so an additional50175-token chat with varied long output was run:3077 tokens
generated, then another short request passed. The additional chat's MTP
acceptance varied (1749/1785), unlike the uniform repeated-token completions.

During this production run, PID1284199 remained unchanged with NRestarts=0,
including other production traffic interleaved with the controlled requests.
The journal contains **322 captures and actual launches after capture**, zero
instantiate OOM, zero fatal CUDA errors, zero graph-disable fallbacks, and a
minimum **160.06 MiB** CUDA free at the sampled graph diagnostic points.

The temporary debug drop-in was then removed and production restarted after
an idle-slot check. PID1343816 started on2026-09-05 at23:00:48 CEST, passed a
3700+64 warmup and OpenAI canary `QWEN_IQ1M_CUDA_GRAPHS_OK`, and was confirmed
active/running with NRestarts=0. Its loaded CUDA-library inode matches the
validated file, SHA256
`476651d845a561cc82cdd5c3482fed97d905b9a6a213607b4713e055e75c9e66`.
Effective settings remain ctx54272/cache-ram3072/batch64/ubatch64/checkpoints1/
parallel1, with no diagnostic or graph-disable environment flags.

These results cover this model/settings and the exercised shapes, not every
possible IQ1_M layout. Batched/noncontiguous IQ1_M still uses its prior path.
The changes remain uncommitted in the existing authorized direct checkout on
`feat/phase-aware-nvidia-power-governor-20260904`; repository policy requires
explicit approval for commit/push.


### Preserved evidence

Remote evidence: `/home/hjotha/.data/cuda-graph-rootcause-20260905/`.
Local investigation report: `/home/hjotha/.data/qwen-cuda-rootcause-20260905/`.

Codex transcripts:

- optiplex: `/home/hjotha/.codex/sessions/2026/09/05/rollout-2026-09-05T20-52-42-01a072ea-70d6-7ab3-b315-0406a6a0cb1d.jsonl`
- GOKAYA: `/home/hjotha/.codex/sessions/2026/09/04/rollout-2026-09-04T16-38-14-01a06cdb-1c4f-7b01-a062-c420144ea05f.jsonl`
- GOKAYA cache change: `/home/hjotha/.codex/sessions/2026/09/05/rollout-2026-09-05T19-39-27-01a072a7-6291-74a1-a8c3-8a0e04d51ebf.jsonl`
