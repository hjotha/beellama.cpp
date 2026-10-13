# F5 occupancy evidence harness

Offline CPU-only tooling that prepares, validates and summarises the F5
(position-split 27B campaign) evidence described in
`docs/occupancy-placement-plan-20261002.md` (§3.7, §3.8) and
`docs/occupancy-progress-20261003.md` (F5 row).

**This tool never launches a server, never compiles, never touches a GPU and
never performs network access.** It only reads/writes JSON evidence files and
the raw logs referenced by that evidence. Server and benchmark commands appear
only as documentation templates and are marked `NOT RUN`.

## Acceptance vs schema admissibility

`summarize` reports an `acceptance` object on every run:

```json
"acceptance": {
  "schema_admissibility": true,
  "gate_arithmetic_only": true,
  "genuine_f5_acceptance": false,
  "reason": "genuine F5 acceptance additionally requires the campaign owner ..."
}
```

* `schema_admissibility` — the bundle satisfies the schema and the hash bindings.
* `gate_arithmetic_only` — the gate formulas were evaluated on the supplied
  numbers. **Passing gates here is never F5 acceptance.**
* `genuine_f5_acceptance` is always `false` from this CPU-only tool. It can only
  come from the campaign owner's GPU/server runs with F4/F7 green.

`evidence_scope` tells you what kind of bundle you are looking at:
`synthetic_fixture`, `fixture_or_placeholder_provenance`,
`placeholder_provenance` or `campaign_measured`. Fixture bundles built by the
tests are reported as `fixture_or_placeholder_provenance` even when their
`evidence_kind` field says `measured`.

## Layout

```
tools/occupancy-f5/
  f5_harness.py                 the harness (schema, plan, validate, summarize)
  fixtures/make_fixtures.py     deterministic synthetic fixture builder
  fixtures/expected-context.json
  fixtures/*.json               evidence fixtures (one defect per file)
  tests/test_f5_harness.py      CPU unit tests
  README.md                     this file
```

## Quickstart

```bash
# machine-readable schema (what must be in an evidence object)
python3 tools/occupancy-f5/f5_harness.py schema

# warmup-per-arm + alternating 5x A/B schedule for the prescribed lengths
python3 tools/occupancy-f5/f5_harness.py plan --context <context.json>

# server command templates (documentation only, status = NOT RUN)
python3 tools/occupancy-f5/f5_harness.py template --context <context.json>

# scaffold for a collector to fill in; always rejected until it is real
python3 tools/occupancy-f5/f5_harness.py init --context <context.json> --out evidence.json

# admissibility check of an evidence object
python3 tools/occupancy-f5/f5_harness.py validate evidence.json \
  --context <context.json> --artifact-root <dir-with-raw-logs>

# gates over admissible evidence only
python3 tools/occupancy-f5/f5_harness.py summarize evidence.json \
  --context <context.json> --artifact-root <dir-with-raw-logs> --require-gates-pass
```

Exit codes: `0` accept / admissible and (for `summarize --require-gates-pass`)
all gates pass, `1` reject / inadmissible or a failing or incomplete gate,
`2` usage, I/O or invalid context error.

Every command also accepts `--out <file>` to write the JSON it prints.

## Campaign context (`f5-context-v1`)

The context is validated in full before anything else. `bad_context:*`
reasons (and CLI exit 2) cover:

* every `expected` key present: `commit`, `branch`, `model_path`,
  `model_sha256`, `corpus_path`, `corpus_sha256`, `p`, `c`, `seed`,
  `batch_size`, `ubatch_size`;
* `commit` is a git object id (40–64 hex), paths and branch are non-empty and
  not placeholder text (`<...>`, `{...}`, `REPLACE_BEFORE_MEASUREMENT`);
* `model_sha256`/`corpus_sha256` are 64 hex chars and **all-zero placeholder
  hashes are rejected**;
* `p` is a positive multiple of 256, `p < c`, `p <= c - 512`, and `c >= 130560`
  so the prescribed cases fit;
* every `protocol` key is present with `warmup_runs >= 1`,
  `measured_runs_per_arm >= 5`, `alternating` true, a valid `first_arm` and
  `min_token_samples >= 32`.

## Evidence contract (`f5-evidence-v2`)

Required top-level keys: `schema_version`, `evidence_kind`, `provenance`,
`protocol`, `cases`, `runs`, `quality`, `capacity`, `restore`, `cycles`,
`bindings`.

* `evidence_kind` is `measured` or `synthetic_fixture`. Synthetic evidence is
  always rejected.
* `provenance.evidence_origin` is `campaign_measured` or `synthetic_fixture`.
  `evidence_kind=measured` requires `campaign_measured`, so relabelling a
  fixture cannot make it acceptable. Placeholder strings and all-zero hashes in
  provenance are rejected (`placeholder_value:*`, `zero_hash:*`).
* `provenance` must match the campaign context exactly: `commit`, `branch`,
  `model_sha256`, `corpus_sha256`, `p`, `c`, `seed`, `batch_size`,
  `ubatch_size`, plus verbatim `command` and the hardware/driver/build/clock/
  governor/cache-state/concurrency facts.
* `protocol` must match the context.
* `cases` are prescribed from `p`: `below_p_*` (baseline arm `xxl`) and
  `above_p_*` at 105000, 115000, 130560 (baseline arm `xxxl`), plus
  `above_p_<p+256>` whenever `p != 102400`. Extra or missing cases are rejected.
* Per case: **one discarded warmup run per arm** (both `xxl`/`xxxl` and
  `split` are warmed), all warmups before the first measurement, then 5
  measured runs per arm, strictly alternating starting on the baseline arm,
  each with `prefill_ms`, `ttft_ms` and >= 32 finite `token_latencies_ms`
  samples. `sequence_index` must be unique, monotonic in the run list, and
  warmups must precede measurements.
* `telemetry` per measured run, all four stages (`load`, `prefill`, `decode`,
  `restore`; `boundary_cross` additionally above P) under `cuda`, `vulkan` and
  `host`, plus `vulkan.gtt_peak_bytes` / `vulkan.gtt_available_min_bytes`,
  `host.mem_available_min_bytes`, `host.swap_in_bytes` and
  `host.swap_out_bytes`.
* **Vulkan attention copy evidence**: every Vulkan stage carries
  `attention_copy_bytes`. It must be `0` in every stage below P (no attention
  copy, together with `routes.vulkan_attention_executed=false`) and must be
  present above P (`routes.vulkan_attention_executed=true`).
* `artifacts` entries must resolve under `--artifact-root` (relative paths, no
  `..`), hash to the recorded SHA-256, and their **first line must be**
  `f5-log-v1 run_id=<run_id> context_sha256=<64hex> provenance_sha256=<64hex>`
  matching the run, the campaign context and the provenance. An arbitrary
  relabelled log does not pass.
* `bindings` holds `context_sha256` (hash of the campaign context) and
  `*_sha256` for `provenance`, `runs` (telemetry included), `quality`,
  `capacity`, `restore` and `cycles`; every value is recomputed at validate
  time and a mismatch rejects.
* `run_ids` coverage: each `quality` entry must name exactly the measured runs
  of its case/arm; each `capacity`, `restore` and cycle entry must name
  existing measured runs (warmup ids do not count), and capacity/restore must
  name runs of the matching case where such a case exists.
* `restore` entries carry `scenario` (`exact_import` or
  `cache_miss_fallback`), `tokens_reused`, `prefix_reprefill_tokens`,
  `next_logits_max_abs_diff`, `ok` and `run_ids`, with
  `tokens_reused + prefix_reprefill_tokens == length_tokens`.
* `cycles` must contain exactly 10 clean restore/divergence cycles.

## Gates

| Gate | Limit |
| --- | --- |
| `below_p_*` prefill / TTFT / decode_p95 regression | `cand/base <= 1.05` |
| `above_p_*` prefill / TTFT improvement | `cand/base <= 0.90` |
| `above_p_*` decode_p95 regression | `cand/base <= 1.05` |
| `*.quality_kl_mean` | `cand - base <= 0.001` nat/token |
| `*.quality_perplexity` | `cand/base <= 1.01` |
| `*.route_vulkan_attention` | `false` below P, `true` above P |
| `memory.no_oom` | `oom_count == 0` in every warmup, measured run and cycle |
| `capacity.*` | occupy `c`, reject `c+1` cleanly, generate to `c` |
| `restore.<length>` | `scenario=exact_import`, `tokens_reused == length`, `prefix_reprefill_tokens == 0`, `ok`, `next_logits_max_abs_diff == 0` |
| `cycles.all_cycles_clean` | no OOM, driver error or corruption |
| `cycles.retained_growth` | `cycle10 - cycle1 <= 64 MiB` per backend |

`cache_miss_fallback` restore records never satisfy `restore.<length>`: partial
reuse (`tokens_reused < length`) and any re-prefill of the imported prefix
(`prefix_reprefill_tokens > 0`) are explicit gate failures with their own
reason codes (`partial_prefix_reuse:*`, `reprefill_required:*`).

`summarize` computes nothing and prints `measured: null` with an empty gate
list unless the evidence passed `validate`.

## Fixtures

`python3 tools/occupancy-f5/fixtures/make_fixtures.py` regenerates the set
deterministically. Each file proves one rejection path:

| File | Expected result |
| --- | --- |
| `expected-context.json` | valid fixture context (synthetic paths/hashes) |
| `complete-synthetic.json` | structural pass, rejected: `evidence_kind:synthetic_fixture` |
| `malformed.json` | `malformed_json:...` |
| `missing-evidence.json` | `missing_field:runs[...].telemetry`, `missing_evidence:quality`, `missing_field:cycles` |
| `nonfinite.json` | `non_finite:runs[...].prefill_ms`, `non_finite:quality[...].kl_mean_nat_per_token` |
| `wrong-provenance.json` | `provenance_mismatch:model_sha256:...` |
| `insufficient-repetitions.json` | `insufficient_repetitions:...:got=3:need=5` |

## Tests

```bash
python3 -m unittest discover -s tools/occupancy-f5/tests -v
```

55 tests, all passing on CPU (Python 3.14.7, ~0.5 s). Coverage: the no
`os`/`subprocess` guard; full context validation (empty `expected`, zero
hashes, P/C alignment and range, placeholder paths via CLI exit 2);
`evidence_origin` refusal; case prescription incl. P+256; schedule with warmup
for **both** arms; every defect fixture; artifact root / hash / missing-file /
header-binding rejection; route enforcement (kernel execution *and* attention
copy bytes below P); warmup-per-arm, warmup-before-measurement and alternation
enforcement; `bindings` presence and mismatch; `run_ids` coverage for
quality/capacity/restore/cycles; gate arithmetic for regression, improvement,
decode p95, quality, capacity, restore (exact vs fallback vs partial reuse),
cycles, swap-in and swap-out; and the `acceptance` block that always reports
`genuine_f5_acceptance: false`.

## NOT RUN — GPU / server commands

Nothing below has been executed by this tool or by this workstream. These are
documentation templates for the campaign owner; every `{...}` placeholder must
be filled from a command recorded before the run.

```bash
# candidate (position split, P fixed per context)
LLAMA_KV_POSITION_SPLIT_P={P} build/bin/llama-server \
  -m {MODEL_PATH} -c {CTX} -b {BATCH} -ub {UBATCH} \
  -fa on --spec-type none --spec-draft-n-max 0 --port {PORT} \
  {KV_AND_OVERFLOW_FLAGS_FROM_F3_F4_PROFILE} {SAMPLING_FLAGS}

# baselines: same flags, arm profile recorded by the campaign owner
build/bin/llama-server -m {MODEL_PATH} -c {CTX} -b {BATCH} -ub {UBATCH} \
  -fa on --spec-type none --spec-draft-n-max 0 --port {PORT} \
  {XXL_CUDA_ONLY_FLAGS_RECORDED_BY_CAMPAIGN_OWNER}
build/bin/llama-server -m {MODEL_PATH} -c {CTX} -b {BATCH} -ub {UBATCH} \
  -fa on --spec-type none --spec-draft-n-max 0 --port {PORT} \
  {XXXL_PER_LAYER_MIXED_FLAGS_RECORDED_BY_CAMPAIGN_OWNER}
```

No `llama-server`, `llama-bench`, `llama-perplexity`, `cmake` or `ctest` command
has been run for F5 here, and no measured latency, quality, memory, restore or
cycle number exists yet.

Memory policy revision (2026-10-10): VRAM headroom, host MemAvailable and
swap remain required observations; they do not determine acceptance. Evidence
v2 requires an explicit nonnegative `oom_count` on every run, including warmup.
Missing OOM evidence is rejected, and any OOM fails the memory gate. Historical
v1 reports retain their original policy and cannot be relabelled as v2.

Full-vocabulary structural comparisons use `compare_logits.py`. It checks
consecutive teacher-forced positions, identical prefix token ids and model/core
provenance, and computes mean and p99 of FP64 KL **per token**. The input
`--teacher-tokens` JSON includes the ground-truth next token after every captured
prefix. At least 32 positions are required; a missing checkpoint rejects the
comparison. `--metrics-only` collects F16-reference quality measurements without
claiming a structural gate. The comparator avoids probability underflow by
using log-softmax differences directly.
