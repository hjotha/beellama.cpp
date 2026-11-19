# MTP performance model — CPU-tested prototype, revision 2

No inference, GPU benchmark, CUDA build, new GPU telemetry, or model hashing was
performed in revision 2. No 4070/T4 numerical prediction or calibrated profile is
provided. The earlier baseline is historical evidence, not a fresh measurement.
All numerical fixtures in tests describe fictional hardware.

## Optimization objective and deferred tier tuning

Owner clarification (Henrique, relayed 2026-11-17): maximize effective decode
TPS for each dynamic context tier, choosing MTP depth N, batch B and microbatch
UB per tier. Maximizing VRAM occupancy is NOT the objective. Available VRAM,
transient allocation peaks and safe transitions constrain the feasible set.
Existing serving configuration already supports p_min per tier; tuning it is a
later phase, not a missing feature to implement. Initial p_min=.65 and fixed
B/UB are experimental controls, not the intended final tier configuration.

The current prototype predicts one qualified regime at a time. Its strict
identity intentionally prevents pooling incompatible B/UB/p_min regimes into
a single local fit. Future tier search must compare separately qualified
candidate regimes at matched context/workload, including N0, then confirm
crossovers, transition cost and stability before activation. It does not yet
implement this joint N/B/UB optimizer, dynamic tier selection or p_min tuning.
Predictive bands should prevent declaring a winner when candidates overlap;
VRAM alone must not determine a performance tier. Physical workload/kernel
profiles must represent the candidate B/UB configuration rather than treating
B/UB changes as universally portable coefficients.

The approximately 100 MB headroom mentioned for a Colab 16 GB scenario is
contextual and approximate. It is neither a validated safe reserve nor an
instruction for production. Retain the conservative runner guards until
separate measurements establish a suitable margin for the actual regime.

## Components

- bench.py: single isolated point runner, explicit future execution gate,
  exact prompt token slices, JSON/CSV, raw cycle traces, CPU-only plan/analyze.
- identity.py: mandatory regime identity and canonical digest.
- model.py: explicitly LOCAL empirical baseline and holdout validation.
- physical.py: hardware-dependent family roofline + host/launch/communication;
  prior-transfer and target-calibrated modes, bounded sensitivity output.
- acceptance.py: per-position empirical survival, observed bonus, actual emitted
  tokens per cycle, and cumulative MTPSTATS parser.
- test_model.py / test_revision.py: CPU-only tests.
- server-context.cpp: existing opt-in MTPCYCLE extension (7 lines), still NOT
  compiled or GPU-qualified.

## Safe planning and future execution

```sh
python3 -m unittest discover -s tools/mtp-perf -p 'test_*.py' -v
python3 tools/mtp-perf/bench.py plan --out /path/plan
```

Default design: N0/1/2/4/6/8 x prompt contexts 4096,8192,16384,24576,32768.
plan validates independent point count and matrix rank for the full fit, every
leave-one-context fold, and every leave-one-N fold (N0 cannot be an absent class).
Repetitions cannot replace independent contexts. Plan validation is CPU-only and
does NOT qualify memory. Three contexts are rejected before any GPU call.

Execution is intentionally one point per invocation. Even after an authorized
window exists, start with 4k N0/N2/N6; inspect memory, kernel routing and graphs
before authorizing each larger context. No automatic expansion to 64k/100k.
The operator must supply --window-authorized and --qualified-point N,context,
--identity identity.json, --build-manifest build.json, and binary/model/prompt.
These switches record an external authorization; they do not create permission.
Do NOT invoke run while production owns the GPU or while this task is CPU-only.
The occupancy guard and sampled VRAM reserve are retained. They do not prove
zero-OOM; allocations can race sampling. Do not adopt the approximate Colab
100 MB observation as a validated reserve or a target for production.

Current runner explicitly disables CUDA graphs using GGML_CUDA_DISABLE_GRAPHS=1
(presence, not value, is what the checked source consumes). This creates a known
graph-off regime, not production-equivalent performance. Graph-on/recapture
qualification and a separate measured regime remain future work.

Each point writes rows.json even when insufficient for a full local fit.
Combine qualified points without losing identity:
```sh
python3 tools/mtp-perf/bench.py analyze --out /path/combined \
 --inputs /path/point1/rows.json /path/point2/rows.json
```

## Identity and reproducibility

Every measured row carries data_kind=measured, identity, regime_id, token-slice
hash, prompt context, measured cycle context and coverage. JSON objects are
canonical JSON strings in CSV cells. check_rows rejects missing fields, null
identities, mixed regimes, digest mismatches and differing token slices at the
same prompt length. synthetic=True is an explicit test-only path requiring
data_kind=synthetic on EVERY row.

Required identity fields (see identity.FIELDS):
hardware_id, gpu_uuid, driver, model_sha256, binary_sha256, build_id, backend,
kernel_profile, kv_target, kv_draft, graphs, sampling, batch, ubatch, p_min,
corpus_sha256, generation, parallel.
hardware_id currently equals hostname + ':' + uname -m; GPU UUID and driver are
separate. Kernel profile is an operator-qualified identifier: supply a versioned
route/shape qualification, not 'unknown'. Dynamic route changes require a new
regime. CPU clocks/power still need observational metadata for performance drift.

Future run verifies model/binary/corpus hashes, actual GPU UUID/driver, controls
and build manifest BEFORE loading its server. A manifest has
{"artifacts": {"/absolute/binary/path": "sha256", "/absolute/linked/library": "sha256"}};
include all actual linked GGML/CUDA libraries. build_id is SHA256 of manifest
bytes, and binary path must be present. Completeness of the library inventory and
kernel qualification remain operator responsibilities. Hashes are computed only
inside future authorized run after its occupancy guard, never in plan or tests.

Fixed controls: B/UB=2048/256, p_min=.65, parallel=1, generation=256,
temperature1/top_k20/top_p.95/min_p.05/seed1234/ignore_eos.
Identity varies if any control changes. B is llama.cpp token batching, NOT
concurrent requests. Corpus file hash is global; exact token slice hash is
compared only among rows with the same prompt context. Model/quantizer/tokenizer
changes therefore cannot silently mix.

## Accounting, context and acceptance

Primary steady-state TPS = sum(committed)*1e6 / sum(MTPCYCLE wall_us).
The first sampled/bootstrap token is outside cycle timing. Coverage requires
exactly response predicted_n = sum(committed)+1, a single request/slot, unique
cycle IDs, monotone anchors, finite positive wall and cycle wall <= request wall.
A missing BEGIN/end, overlapping markers or incomplete generation fails closed.
This strict bootstrap convention must be checked on the first real instrumented
run; the code has not been GPU-qualified.

End-to-end TPS = response predicted_n / HTTP start-to-final-SSE elapsed seconds.
It includes prefill, network/client handling and bootstrap. Server predicted TPS,
prefill_ms and observed TTFT are separate fields. TTFT is first nonempty content,
not a precise first-device-token timestamp.
Rows retain prompt context separately from measured_context_mean/min/max.
Raw cycles retain exact anchors. Local fitting uses mean measured cycle context,
not just initial prompt length.

draft_us, synchronized target_us (includes attention/KV), catchup_us,
accept_us, sample_us (SUBSET of accept_us), and residual_us remain separate.
Residual is unclassified wall time, NOT a GPU counter. KV-only GPU time needs
Nsight. Actual phase timing is diagnostic; local empirical fit uses total wall.

Acceptance uses actual committed tokens per cycle, including retries/stops, for
local fit. Eligible single-pass, non-stopped cycles additionally expose:
S_i=P(accepted drafts >=i) and G_i=P(proposed >=i), each divided by eligible
cycles, with S_i<=G_i and nonincreasing sequences. Bonus is measured as
committed-accepted_drafts; E=bonus+sum(S_i) within the proposal cap.
Excluded cycles and denominator are explicit; survival estimated on eligible
cycles is not substituted for full-request throughput.
Replay/stop cases preserve empirical emission instead of fabricating positions.
N0 is separately modeled with one token per cycle.

MTPSTATS already reports n_acc_tokens_per_pos/n_gen_tokens_per_pos.
Parser saves these cumulative implementation counters after server shutdown.
They include warmup and possibly multiple calls; no false request-level
attribution. from_counters requires explicit step denominator, bonus and source.
A fitted constant geometric survival is only a labeled fallback for an unobserved
N in the LOCAL model. Draft confidence p_min is never acceptance probability.

## Two prediction paths

LOCAL: model.fit/predict_detail fit nonnegative a+bN+cL+dNL (N0 a0+c0L).
Observed effective token counts are interpolated by measured context for each N.
No local coefficient transfers to another hardware identity. New N may use the
explicitly labeled geometric fallback. Default predict rejects outside N/context
envelope; allow_extrapolation must be explicit. Holdout edge folds deliberately
allow it and report the number of extrapolated predictions and acceptance source.
Validation reports median APE or a blocked fold. No real error is available.

PHYSICAL: physical.predict_physical(profile,workload,acceptance,n,context).
For each named kernel family:
device_us=max((weight_bytes+KV_bytes)/effective_BW, FLOPs/effective_compute).
Effective rate = hardware rate * family efficiency. Work terms contain
constant/per_n/per_context/per_n_context with explicit units and source.
Families carry kernel signatures; workload contains domain and signature.
Weight traffic, attention/KV traffic, FLOPs and launch count are supplied from
GGUF/graph/kernel inspection or measured traces; none are guessed from filename.
SM/wave/shape effects are encoded in family/domain-specific efficiency profiles;
there is no universal attention or KVarN efficiency.

The total separates draft, verify, sample, catchup, host, launch and communication.
Communication = message_count*latency + bytes/bandwidth. This implementation sums
serialized critical-path costs; overlap requires an appropriate qualified model,
not a double-counted total. See test_revision.inputs for the executable schema,
with fictional values only.

Hardware profile mode:
- prior_transfer requires explicit transfer assumptions and parameter provenance.
  It is an uncalibrated hypothesis, even if source-GPU efficiencies are measured.
- target_calibrated requires matching target hardware/workload/probe IDs and
  calibration domain. calibrate_target derives efficiencies from >=3 explicitly
  memory-bound AND >=3 compute-bound device-only probes per family, plus measured
  host/launch times. It refuses underidentified total-latency-only probes.
  Communication needs separate target probe evidence. Probe kind must be measured;
  synthetic=True is explicit and propagates data_kind=synthetic into predictions.

Parameters have low/value/high/source. Output includes central TPS, phase/kernel
breakdown, effective tokens, provenance, domain/extrapolation and a sensitivity
envelope from parameter corners and explicit effective-token bounds.
This envelope is NOT a statistical confidence interval. It omits unmodeled
correlation and physical-work/model error; widen assumptions or reject the domain.
No bundled profile supplies numerical 4070/T4 predictions.

CLI for an externally prepared, qualified physical input:
```sh
python3 tools/mtp-perf/physical.py input.json --out prediction.json
```
This is CPU arithmetic only; it never measures hardware.

## Evidence and remaining qualification

See evidence/REPORT.md, revision-tests.txt and plan-revision/plan.json.
Historical baseline/guard evidence is retained under its original filenames.
No real calibration, target holdout error, thresholds N/B/UB, automatic pruning,
bootstrap CI, live KV tier transition, profiler counters or zero-OOM validation.
No production changes, commit, push or merge.

Methodological context supplied by reviewer:
[KernelSight-LM](https://arxiv.org/html/2606.28565v2) motivates per-family
efficiencies and distinguishes transfer from serving-matched target measurement.
[Speculative Decoding: Performance or Illusion?](https://arxiv.org/html/2601.11580v2)
motivates verification cost and request/position-dependent acceptance.
Neither establishes accuracy for these GGML IQ3/KVarN4/MTP kernels. This code is
an experimental implementation, not a reproduction of either paper's results.
