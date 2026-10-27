# RKVA KV-format study (KVarN4 / native Q8_0 / synchronous F16)

Status: **prepared and CPU-tested offline; no hardware result exists yet.**
Root must run the live campaign after the pending QA INT8 work, one GPU child
at a time. Astra should review this report and the three files before any
integration. Nothing here is a claim that a format is faster, that a GPU run
passed, or that KV format alone explains a latency/TPS delta.

## Scope and ownership

- Branch: `perf/rkva-kv-format-study-20261026`
- Base: `4bf192bb2241aabe595960bfd44d17cde59ebf7d` (`main`)
- New files only:
  - `tools/xbox/scripts/rkva-kv-format-study.py`
  - `tools/xbox/tests/test_rkva_kv_format_study.py`
  - `docs/agent-reports/rkva-kv-format-study.md`
- Protected paths were not edited: the root checkout `/home/hjotha/beellama.cpp`,
  `bench-rkva-range.py`, INI/tier files, models, slots, configuration, other
  clones, the Xbox app/package, and services.
- No GPU/CUDA runtime, Xbox, LAN, SSH, service, deployment, or full build was
  used for this preparation. Unit tests are pure CPU and run with `nice -19`.

## Helper provenance and overrides

The wrapper imports `tools/xbox/scripts/bench-rkva-range.py` by repository
path (or `--repo`) and reuses its lifecycle, snapshot hashing, slot operations,
timing checks, F5 handling, and provenance freezing. It does **not** fork that
file.

- Helper SHA256: `967a2a2af8784d9706b41859c18657834da687c7c93844caf3800c7cafb61dea`
- The wrapper verifies source anchors and the helper `ROOT` before use. A
  changed helper fails closed instead of applying stale overrides.
- Small in-memory overrides, recorded in every study manifest:
  `server_command`, `pinned_environment`, `prefix_contract`, `classify_logs`,
  `validate_startup`, `request_body`, and memoized `fingerprint`.
- The helper's file is not written. The study manifest records the helper
  hash, anchors, and override names.

The helper hardcodes KVarN4 in its command, negotiation, snapshot contract and
environment. The wrapper changes only those inputs per selected format; it
does not rewrite the helper's 900-line lifecycle or checks.

## Format matrix

| Route | Server-visible KV / transport | Prefix contract | Required actual proof |
|---|---|---|---|
| `kvar4` (descriptor 3) | Global `kvarn4` K/V; per-layer remote `q4_0/q4_0`; `--remote-attn-format kvarn4`; `--remote-attn-wire-format fp8`; B1024/UB512; 2xN256 wavefront; TCP prefetch; async overlap; tail 128; CUDA native producer | Existing local KVarN4 contract; may use proven `--frozen-prefix-batch 512` | `descriptor=3`, `async_overlap=1`, `wire=fp8_e5m2`, stats `kvarn4/projected_f32/tail=128`, real calls/upload bytes, TCP completion, CUDA FP8 pack/unpack, `gpu_profile.engine.optimized_kernels.kvarn4`, bucket `attention/kvarn4-prefix/optimized-...`, `engine.kvarn4_multi_query.mq2_i8.calls=0`, `engine.bulk.fp8_cpu_fallback_calls=0`, transport op22 |
| `q8_0` (descriptor 5) | Global `kvarn4` preserved; per-layer remote `q8_0/q8_0`; `--remote-attn-format q8_0`; `--remote-attn-wire-format fp8`; host CPU Q8_0 producer; op16/op22; no KVarN tail | New format-owned contract; never the KVarN4 contract | `native Q8_0 bulk descriptor=5 ... wire=fp8_e5m2`, stats `q8_0/projected_f32/tail=0`, real upload/RPC, `engine.optimized_kernels.q8_0`, bucket `attention/q8_0/optimized-...`, `engine.kvarn4_multi_query.mq2_i8.calls=0`, no KVarN4 multi-query/rowlane, transport op22, no synchronous fallback |
| `f16` | Per-layer remote `f16/f16`; `--remote-attn-format f16`; `--remote-attn-wire-format f16`; no native bulk; no async overlap; no TCP prefetch; CUDA FP8 off; B1024/UB256; wavefront off; serial decode | New format-owned contract; never the KVarN4 contract | `range source=f16 storage=f16; direct original K/V half bytes, no quantized intermediate`, stats `f16/f16/tail=0`, wire f16/f32, TCP counters zero, `engine.optimized_kernels.f16`, bucket `attention/f16/optimized-...`, `engine.format_multi_query.f16`, `engine.bulk.fp8_gpu_calls=0`, `engine.kvarn4_multi_query.mq2_i8.calls=0`, transport ops 13/14 (no op22) |

Notes from the audited source:

- Native Q8_0 is descriptor layout 5 in
  `ggml/src/ggml-remote-attn/ggml-remote-attn.cpp:4434-4513`. Its KV payload is
  ordinary Q8_0 blocks with a host CPU producer; Q/OUT stay on the FP8 wire.
  The server reports `range_source_cache_type=projected_f32` for every native
  owner, so `range_cache_type=q8_0` plus descriptor 5 is the discriminator.
- F16 takes the non-native range path. Explicit async overlap without a bulk
  format is rejected (`ggml-remote-attn.cpp:4568-4577`); `native_bulk` stays
  false (`:4652`) and the direct-original-F16 line is required (`:4671`).
- Per-layer remote formats require the global KVarN cache, TCP position split,
  Qwen35 target, one slot, remote prefill and Flash Attention
  (`common/common.cpp:3428-3439`, `src/llama-context.cpp:612-636`).
- F16 KV with FP32 math is the supported UWP path. FP16 PSO native math is not
  claimed here and must not be conflated with KV F16.

### GPU-profile schema gate

The real `remote_attn gpu_profile` payload is nested as
`{"engine": {...}, "transport": {...}}`. Engine counters, `optimized_kernels`,
`kvarn4_multi_query`, `kvarn4_decode_rowlane`, `format_multi_query`, `bulk` and
`buckets` live under `engine`; per-op transport counters live under
`transport.ops`. The proof validates the schema, object/list/bool/int/string
types, required counters and bucket names, and never substitutes zero for a
missing field.

- INT8 is read only from `engine.kvarn4_multi_query.mq2_i8.calls`. It must be
  present, a nonnegative integer, and exactly zero. The old nonexistent
  `kvarn4_mq_i8` root field is not accepted.
- KVarN4 revision-3 serial-prefix buckets are named
  `attention/kvarn4-prefix/optimized-...`; Q8_0/F16 use
  `attention/q8_0/optimized-...` and `attention/f16/optimized-...`.
- `engine.bulk.fp8_gpu_calls` must be positive for native KVarN4/Q8_0 and zero
  for F16; `engine.bulk.fp8_cpu_fallback_calls` must be exactly zero.
- Native KVarN4/Q8_0 require transport op22 with real request/response byte
  counters. F16 requires synchronous op13/14/19 and must not show bulk op22.
- Missing engine/transport, a missing counter, malformed types, a reference
  fallback, or a missing bucket is `FAIL`/`UNSUPPORTED`, never `PASS`.

The KVarN4 fixture is a compact subset of the archived real profile
`/home/hjotha/.codex/artifacts/rkva-priorities-20261026/overflow-refusal-r2/reference/server.log`.
The same artifact was parsed directly during CPU validation and passed the real
proof. Q8_0/F16 test fixtures follow the current C++ reporter schema; they are
**not** hardware evidence and are not claimed as hardware PASS.

## Snapshot and campaign rules

- A prefix is reusable only when its stored contract matches the current
  model, token stream, P, context, batch, physical prefix UB256, and
  format-specific KV contract. KVarN4 uses the helper's original contract
  unchanged, so the existing C131072/B512 snapshot remains reusable with
  `--frozen-prefix-batch 512`.
- `--frozen-prefix-batch 512` is accepted with any `--formats` selection and is
  applied only to the KVarN4 arm. Q8_0/F16 keep their own contracts and
  B1024-derived prefix paths; no global rejection is needed.
- Q8_0 and F16 get their own contract (`format`, `remote_kv`, local prefix
  still KVarN4). A KVarN4 metadata contract is rejected for Q8_0/F16; there is
  no silent conversion and the tool never edits an old `.meta.json`.
- A missing snapshot is a preflight error unless `--allow-new-prefix` is
  explicit. A new snapshot is created only by that arm after its cold P
  request; the study checks that the creation warm had `cache_n=0` and
  `prompt_n=P`.
- A restored prefix must prove the helper's bounded replay: at most 128 prompt
  tokens replayed to P. Measured points must expose `cache_n=P` and
  `prompt_n=point-P` (for 130560: 28160), or the arm fails.
- Order defaults to `ABBA`; `interleaved` repeats each format twice as well.
  `--execute` refuses `sequential` because same-format determinism would be
  unproven. One child runs at a time; no parallel GPU arm is started.
- Output must be a new directory. Prefix and output roots must be disjoint
  dedicated directories. Model, server binary, libraries, prompt, source
  HEAD/diff and snapshot files are hashed before and checked after the run.
- Requests are greedy (`temperature=0`, `top_k=0`, `top_p=1`, `min_p=0`,
  seed 1234) with `return_tokens`. F4/F5/per-token arrays, nprobs and full
  token dumps are scrubbed from study publications; raw per-arm server/response
  files remain for audit.
- `GGML_CUDA_MASK_PACK` stays at DEFAULT/unset for all three formats; the
  wrapper no longer forces it off. Every proof audits the frozen environment:
  the key must be absent, the 128 MiB graph guard and 4096-token KVarN window
  must remain `128`/`4096`, and the binary is unchanged. F16 still declares its
  serial (no wavefront/async) alternative.
- Same-format repeats must produce identical token IDs and identical
  cache/prompt/predicted counts. Token IDs/quality across formats are
  diagnostics only because quantization and cadence can change them.
- The study publishes `FAIL` or `UNSUPPORTED` explicitly. It never hides a
  cold replay, a synchronous Q8_0 fallback, or a reference-PSO fallback as a
  qualified native route.

## Device flags

The study does not call `deploy.sh`, the Xbox frontend, or any service. It therefore
does not mutate INT8 prefill or rowlane flags; the original device state is
preserved by non-interference. The proof records the actual state from the GPU
profile: KVarN4 must show `engine.kvarn4_multi_query.mq2_i8.calls=0` (INT8 off)
and records the actual `engine.kvarn4_decode_rowlane` state/calls. Q8_0/F16 must not use the
KVarN4-only rowlane or INT8 paths. If Root needs a non-default device state, it
must be changed outside this harness and recorded with the hardware result.

## Commands

Preview is inert and is the default. Replace placeholders with the real 27B
main-profile model, token file, console endpoint and thread count. Do not change
tiers/INI/guards or P to make a point fit; the tool rejects context above
160000 instead.

```bash
nice -19 python3 tools/xbox/scripts/rkva-kv-format-study.py \
  --repo /home/hjotha/beellama.cpp \
  --server-binary /home/hjotha/beellama.cpp/build-optimized/bin/llama-server \
  --model <27B-main.gguf> \
  --prompt-tokens <tokens-covering-130560.json> \
  --xbox-host <CONSOLE-IP>:50054 \
  --threads <THREADS> \
  --p 102400 --context 131072 --points 130560 --n-predict 256 \
  --prefix-dir /home/hjotha/rkva-kv-study-prefixes \
  --output /home/hjotha/rkva-kv-study-out-YYYYmmdd-HHMM
```

Full matrix, reusing an existing KVarN4 B512 prefix and creating fresh Q8_0/F16
prefixes from cold P (only after QA INT8 and with Root's explicit approval):

```bash
nice -19 python3 tools/xbox/scripts/rkva-kv-format-study.py \
  --repo /home/hjotha/beellama.cpp \
  --server-binary /home/hjotha/beellama.cpp/build-optimized/bin/llama-server \
  --model <27B-main.gguf> --prompt-tokens <tokens-covering-130560.json> \
  --xbox-host <CONSOLE-IP>:50054 --threads <THREADS> \
  --p 102400 --context 131072 --points 130560 --n-predict 256 \
  --formats kvar4,q8_0,f16 \
  --kvar4-prefix-snapshot <existing/C131072/B512/snapshot.slot> \
  --frozen-prefix-batch 512 \
  --prefix-dir /home/hjotha/rkva-kv-study-prefixes \
  --output /home/hjotha/rkva-kv-study-out-YYYYmmdd-HHMM \
  --allow-new-prefix --execute
```

Q8_0 alternative if the matched 2xN256/TCP profile is not qualified for
descriptor 5 (this is a declared serial profile, not a silent fallback):

```bash
--q8-profile serial
```

Select subsets with `--formats kvar4`, `--formats q8_0`, `--formats f16`, or a
comma-separated list. `--order ABBA` is the default; `--order interleaved` is
the other repeated control.

## Outputs

- `study-manifest.json`: plan, helper hash/anchors/overrides, freeze, protected
  flags, device-flag policy, per-arm manifests and summary.
- `study-results.json`: per-format statuses, actual proof, pooled
  `n_predict-1=255` decode metrics, determinism, accounting congruence,
  diagnostic token divergence, and upload-byte deltas.
- `study-arms.json`: per-arm status/proof without raw token arrays.
- `<NN>-<format>/`: the helper's own request/response/server logs, manifests,
  and results. These remain the audit source; study publications omit F4/F5,
  per-token arrays and full dumps.

GPU and host RAM are **not instrumented** by the helper. The result marks them
`unmeasured`; attach external measurements rather than inventing numbers.

## Astra blocker fixes

- `gpu_profile` is unwrapped as `engine` + `transport`; all counters, buckets,
  kernels and INT8/rowlane fields are read under `engine`, with mandatory type
  and presence checks. Missing fields are never treated as zero.
- INT8 is `engine.kvarn4_multi_query.mq2_i8.calls`; it must exist and equal
  zero. KVarN4 uses `attention/kvarn4-prefix/optimized-...` buckets.
- `--frozen-prefix-batch 512` is accepted with all three formats and applied
  only by the KVarN4 arm; Q8_0/F16 contracts and B1024 prefixes are unchanged.
- `GGML_CUDA_MASK_PACK` is no longer forced off; DEFAULT/unset is preserved and
  audited, while guard 128, window 4096 and the binary stay frozen.
- Missing route/counter/schema/command/environment is `FAIL`/`UNSUPPORTED`,
  never `PASS`.
- Q8_0 keeps the native bulk/op22/FP8 checks, but KVarN4-exclusive rowlane and
  `kvarn4_multi_query` call checks use `name != 'kvar4'`, so Q8_0 cannot skip
  them. PSO `available` is not treated as a call counter.

## CPU validation performed

```bash
python3 - <<'PY'
import ast
from pathlib import Path
for p in ['tools/xbox/scripts/rkva-kv-format-study.py',
          'tools/xbox/tests/test_rkva_kv_format_study.py']:
    ast.parse(Path(p).read_text())
PY
```

```bash
nice -19 python3 -m unittest discover -s tools/xbox/tests \
  -p test_rkva_kv_format_study.py -v
```

Result: **37 tests, OK** (0 failures). The suite covers: helper hash/anchors and
no removed-integration dependency; inert default preview; exact command args;
global KVarN4 preservation; F16 serial/FP8-off profile; Q8_0 serial alternative;
160000 context cap; unchanged P/points; format-owned prefix paths and contracts;
KVarN4 B512 reuse plus Q8_0/F16 contract rejection and full three-format
documented preview; missing snapshot requiring `--allow-new-prefix`;
`--frozen-prefix-batch` accepted generally but applied only to KVarN4;
repeated-order guard; greedy temperature 0; nested real-engine `gpu_profile`
subset; old root-level profile rejected; missing engine/transport/counter/bucket
rejected; nested `mq2_i8.calls` missing/positive rejected; `kvarn4-prefix`
bucket name enforced; command/source exactness; mask-pack DEFAULT/unset and
guard/window audit; transport op22/sync op checks; Q8_0 bulk/op22 PASS with zero
KVarN calls; Q8_0 rowlane/mq2/mq4 calls rejected while PSO availability alone
is not accepted; Q8_0 sync/reference fallback
and F16 expansion rejection; same-format source checks; F16 CUDA-FP8/TCP-off
checks; pooled 255-step decode; cross-format accounting and same-format
determinism; unsupported publication; and a fake-client full six-arm campaign.

Regression check of the protected helper:

```bash
nice -19 python3 -m unittest discover -s tools/xbox/tests \
  -p test_bench_rkva_range.py
```

Result: **48 tests, OK**. The helper is unchanged (`git diff` shows only the
three new files).

The documented three-format command shape was exercised with tiny CPU fixtures,
including `--frozen-prefix-batch 512`, an existing dummy KVarN4 slot and
`--allow-new-prefix`. Result: exit 0, preview-only, `ABBA` sequence, no output
or prefix directory created, KVarN4 path kept at the existing snapshot, Q8_0/F16
paths derived at B1024, and `GGML_CUDA_MASK_PACK` absent from every environment.

The archived real `gpu_profile` line was parsed directly from
`/home/hjotha/.codex/artifacts/rkva-priorities-20261026/overflow-refusal-r2/reference/server.log`.
Result: `PASS`, bucket
`attention/kvarn4-prefix/optimized-kvarn4_mq2_fine/nq2-16/r1-256/warm`,
`engine.kvarn4_multi_query.mq2_i8.calls=0`, rowlane enabled with zero calls,
and transport ops `[22]`. The same artifact is embedded as a compact fixture in
the unit tests; it is a past reference log, not a new hardware run.

`git diff --check` was run after staging the three files; no whitespace errors.

## Hardware handoff to Root

1. Wait for QA INT8 to finish. Keep the protected helper untouched.
2. Prepare a token file covering 130560 IDs; do not alter the 27B main profile,
   tiers, INI, guards or P.
3. Run the inert preview on the Root host and confirm the printed commands,
   prefix intents, contracts and output path.
4. Run one GPU child at a time, serially, after the INT8 QA gate. Use the
   existing KVarN4 B512 prefix for the KVarN4 arm and create Q8_0/F16 prefixes
   only with `--allow-new-prefix`.
5. If Q8_0 reports `UNSUPPORTED` for the matched wavefront/TCP profile, rerun
   the declared `--q8-profile serial` arm; do not treat a synchronous or
   reference fallback as native Q8_0.
6. If F16 reports `FAIL`, inspect the recorded config/proof; do not call it an
   application error without proving that the config was valid. F16 KV with
   FP32 math is supported; FP16 PSO math is a different question.
7. Send the three files, this report, the study manifests/results, and the
   exact commands/hardware/model hashes to Astra for pre-integration review.

## Known limitations

- No hardware benchmark, GPU-memory measurement, host-RAM measurement, Xbox
  deployment, or console runtime was performed.
- The comparison is not a KV-format-only isolation: wire, async overlap,
  producer, query cadence, engine and kernel differ. The study states this in
  every plan/result and only requires determinism within a format.
- Q8_0 matched transport relies on the current server/console qualification for
  descriptor 5. If that is absent, the explicit serial alternative is the only
  honest Q8_0 path in this tool.
- Only the archived KVarN4 profile is a real artifact subset. Q8_0/F16 test
  fixtures follow the current C++ reporter schema but are not hardware
  evidence; no Q8_0/F16 hardware PASS is claimed.
- The wrapper validates the proof it can see in logs/stats/profile. External
  GPU/host memory and any non-default device flag state still require Root's
  recorded evidence.
