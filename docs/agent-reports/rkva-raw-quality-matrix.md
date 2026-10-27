# RAW full-vocabulary OFF/ON matrix controller — CPU delivery (2026-10-27)

Branch: `perf/rkva-raw-matrix-20261027` (clone
`/home/hjotha/beellama-agents/quality-gate`, base `0922ae350`). No worktree,
no merge, no push. The delivery commit is the branch HEAD reported with this
file.

**This delivery did not execute hardware.** No `llama-server`, CUDA runtime,
`llama_decode`, model load, capture child, Xbox LocalState call, deploy
endpoint, LAN/SSH, `systemctl`, UWP/MSVC build, or CUDA build was run. Only the
CPU tests and an inert preview were executed. The root owner and Astra remain
responsible for the real run and review.

## 1. Files delivered

| File | Contents |
| --- | --- |
| `tools/xbox/scripts/rkva-raw-quality-matrix.py` | Preview-by-default RAW matrix controller; `--execute` explicit |
| `tools/xbox/tests/test_rkva_raw_quality_matrix.py` | 46 CPU-only tests; fakes for deploy/flags/capture |
| `docs/agent-reports/rkva-raw-quality-matrix.md` | This report |

No existing file, root checkout, model, prefix snapshot, `.codex` artifact,
Xbox package, or service was modified.

The controller is 1,903 lines and the test file is 1,057 lines. The previous
CPU delivery was already 1,633 lines; this fix adds the event-driven library
sample, full-contract prefix delegation, model/common-args guards, shifted NLL,
fatal child cleanup and their tests. The ~800-line target was not met; the
excess is validation and fail-closed assertions, not a copied 3,000-line HTTP
harness. The approved gate/helper are imported by `importlib`.

## 2. Astra blockers fixed

### 2.1 Loaded libraries are sampled on a real startup event

The previous code read `/proc/<pid>/maps` immediately after `Popen`; that can
legitimately be empty or incomplete. It now:

- tees child stdout/stderr line by line into `capture.log`;
- watches the real gold-log marker
  `llama_context: backend_ptrs.size()` (found in
  `raw-gold-http-off-r1.log`) and samples `/proc/<pid>/maps` exactly when that
  line arrives;
- permits the documented `CAPTURE step=0` fallback when the startup marker was
  not observed, but records `startup_fallback=true`;
- performs no RPC and no polling loop;
- fails closed if the startup event is missing, the map is empty/None, the
  mapped set is incomplete, an unexpected llama/ggml path appears, or a mapped
  library stat/hash differs from the frozen binding;
- requires the complete active `ldd` llama/ggml set, not just any nonempty map.

Tests cover an empty map, a missing startup marker, the step-0 fallback, the
real gold-log marker identity, a missing expected library, and a changed
loaded-library file.

### 2.2 Prefix loading reuses the full approved contract validator

The custom partial contract checks were removed. `load_prefix` now calls
`bench-rkva-range.py::load_prefix_snapshot` through the imported helper, with
the frozen batch-512 namespace (`P=102400`, `context=131072`, `batch=1024`,
`frozen_prefix_batch=512`, mapped FP8, fine range kernel, WF 2xN256,
`n256_lookahead=false`, TCP prefetch on, local reuse off, mask pack default).

That validator checks the complete metadata contract (schema, codec, model,
P, context, slot, batch, physical ubatch, flash attention, n_rs_seq, native
Kvarn bits, tail, window, full environment), the actual prefix token/hash,
the saved/sidecar files and their SHA-256 values. The controller additionally
compares:

- metadata SHA-256 with the original HTTP preflight/binding;
- actual prefix tokens/hash and the exact file list/details with the original
  HTTP binding.

Tests tamper environment (`GGML_KVARN_WINDOW_CHUNK=999`), slot, schema,
requested prefix hash, context, metadata SHA, and file list. A re-hashed forged
main file with a matching new metadata file is still refused by the original
HTTP file-list binding.

### 2.3 Model and common-args identity are frozen per arm

- The model path, SHA-256, size, mtime/ctime/inode are frozen in the binding and
  rechecked before and after every arm. The model SHA is recomputed at each
  guard check.
- `common-args.json` is read once as exact bytes, parsed for validation, and
  copied into `output/inputs/common-args.json`. The child receives only that
  owned frozen copy; it never receives the mutable original path.
- Before/after every arm the guard checks the original file stat and SHA, and
  the frozen copy stat and SHA. Mutating either between arms fails the run.
- The binding records source HEAD/tracked diff, binary SHA/stat, model
  SHA/stat, original/frozen common-args SHA/stat, reference binding, prefix
  metadata/files, environment, gates, cases, and teacher plan.

Tests mutate the model stat, model SHA, original common args, and frozen common
args; all fail. A separate test confirms the child command uses the owned
frozen path rather than the original.

### 2.4 Stable shifted NLL and KL

NLL/KL no longer use `maximum + log(total)` followed by `logit - lse`, which
cancels at high offsets. `raw_stats` keeps `maximum` and `log_total` separate;
both NLL and full-vocabulary KL use:

```text
log p = (logit - maximum) - log(total)
```

Regression tests:

- `[1e20, 1e20]` gives `NLL = log(2)`;
- unequal high-offset logits keep a nonzero representable NLL;
- `[0, -1000]` gives `NLL = 1000`, self-KL `0`;
- `-inf` and NaN logits fail closed.

### 2.5 Child cleanup failures are fatal and flags are not restored over a live child

`_terminate_child` no longer swallows timeout/reap failures. It returns a
record with:

- `pid`, `alive`, `terminated`, `killed`, `reaped`;
- `terminate_error`, `kill_error`, `reap_error`.

On timeout or signal it performs a bounded SIGTERM wait, then SIGKILL and a
bounded wait. Any terminate/kill/reap error, unreaped child, or child still
alive becomes `ChildCleanupError`; `cleanup-failure.json` is written in the arm
directory and the failure is recorded in the final result.

If the child may still be alive, `flags.arm()` is not used for that arm:
`LocalStateFlags.apply()` is called, and `restore()` is skipped until a later
guard has a dead child. The outer `execute` finally also skips restoration,
records `flags_restored=false`, `model_alive_pid`, and the cleanup details, and
returns FAIL. A dead but uncleanly terminated child may be restored, but the
run still fails.

Tests cover SIGTERM timeout, SIGKILL timeout, terminate error, wait/reap error,
and an `execute` path where the owned child is reported alive and flag
restoration is skipped.

### 2.6 Semantic scope is explicit

The catalog consumes semantic prompt tokens only. `reference_answer_json`,
`reference_nll_steps`, fixed reference NLL, and free-greedy semantic accuracy
are **not** implemented or consumed. Preview and final output report:

```json
{
  "reference_answer_json_consumed": false,
  "fixed_reference_nll": false,
  "free_greedy_semantic_accuracy": false
}
```

The result label remains `RAW_NUMERICAL_STRESS_ONLY`; it is not broad semantic
quality and not INT8 default approval.

## 3. CLI

```text
rkva-raw-quality-matrix.py --help
rkva-raw-quality-matrix.py [preview by default] \
  --capture-binary BIN \
  --common-args-json FILE \
  --cases-dir DIR \
  --reference-campaign DIR \
  --output NEW_DIR \
  [--teacher-steps 128] \
  [--xbox-host 192.168.1.200:50054] \
  [--repo CHECKOUT] \
  [--teacher-mode greedy|http] \
  [--determinism none|off|on|both] \
  [--determinism-case NAME]

rkva-raw-quality-matrix.py --execute ...   # only the root hardware owner
```

- Preview starts no child, touches no flags, contacts no deploy endpoint,
  writes no output, and does not create the output directory.
- `--output` must not exist. Every arm/case directory is created new; artifact
  writes never overwrite.
- `--teacher-mode greedy` (default) uses the OFF argmax sequence as the ON
  teacher. `--teacher-mode http` uses the frozen HTTP teacher ids/hashes.
- `--determinism off|on|both` re-runs the selected case and requires bitwise
  logits and teacher equality.

## 4. Inputs

### Cases directory

Accepted layouts:

```text
DIR/catalog.json                  # {"cases": [...]}
DIR/<case>/case.json              # optional prompt_json/tokens
DIR/<case>/prompt.json            # full token array
DIR/<case>/tokens.json            # existing gate artifact (tail or full)
DIR/<case>.json                   # one case per file
```

Existing gate artifacts with `base_length == 102400` and a tail `tokens` array
are combined with the exact reference base token stream; no tokenizer and no
generation are used. Every case must have:

- `length % 512 == 0`;
- `length >= P + 512`, `length + teacher_steps <= 131072`;
- nonnegative int32 ids;
- first 102400 ids hashing as little-endian uint32 to the reference prefix;
- optional JSON token digests and raw `tokens_sha256` verified when present.

Synthetic/default token data is labelled numerical stress only.

### Reference campaign

The existing HTTP campaign provides:

- `binding.json` or `preflight.json.binding`: model, prefix path, source,
  libraries, and frozen gates;
- `tokenizer/manifest.json` (or a case arm manifest): exact environment and
  `helper_args.prompt_tokens`;
- `cases/<name>/off/teacher.json` for `--teacher-mode http`.

Reference gates must equal `rkva-quality-gate.py::GATE_DEFAULTS`; a different
threshold set is refused, not relaxed.

### Common arguments

The frozen HTTP argv must keep `-m` on the reference model, `C131072`,
`B1024/UB512`, `parallel 1`, `CUDA0`, `gpu-layers 99`, `flash-attn on`,
`fit off`, `kvarn4/kvarn4`, `--kv-position-split-p 102400`, `full` remote
layers, `q4_0/q4_0` remote cache, `kvarn4`/`fp8` remote format,
`--remote-attn` equal to `--xbox-host`, `none` speculation,
`--spec-draft-n-max 0`, `--no-context-shift`, and `--no-warmup`.

## 5. Lifecycle

`--execute` performs, in order:

1. Import the approved gate/helper modules from `--repo` with `importlib`.
2. Load/validate the reference campaign, cases, common args, gates.
3. Reuse the full prefix validator and bind metadata/filelist to HTTP.
4. Freeze model SHA/stat and common-args bytes; check disk budget.
5. Verify GPU quiet with `nvidia-smi --query-compute-apps` and the existing
   foreign-process matcher, ignoring only this process chain.
6. Create the new output directory, write the frozen common args, binding and
   preflight.
7. Create `DeployClient`/`LocalStateFlags`, capture the original flags.
8. For each case: OFF capture, teacher, ON capture with the same teacher,
   manifest validation, activation proof, metrics.
9. Optional fresh-process OFF/ON determinism controls.
10. Restore and verify flags in `finally` unless a child may still be alive;
    then verify source, prefix, model, common args, and library bindings.

The capture child receives a frozen environment with per-arm isolated
`TMPDIR`, `CUDA_CACHE_PATH`, and `XDG_CACHE_HOME`. The controller never starts
an HTTP server, PING/HELLO, or extra remote call during capture.

## 6. Metrics and windows

Each step reads exactly `4 * 248320` bytes. At most two rows plus temporary
float64 exponent arrays are live; rows are discarded after the step. Raw
artifacts are preserved.

The metrics are:

- `nll_off`, `nll_on`, `nll_delta`;
- `kl` over all 248320 ids;
- top-1 ids/probabilities, margins, agreement;
- high-confidence divergence (`margin >= 0.5` on both arms).

Independent windows:

| Window | Meaning |
| --- | --- |
| `first_1` | one-step numerical check |
| `first_8` | first eight steps, matching the root GOLD8 comparison |
| `all` | every `--teacher-steps` step (128 in the matrix) |

Every window applies the unchanged thresholds from
`rkva-quality-gate.py::GATE_DEFAULTS`. A case is PASS only when activation is
proved and all selected windows pass. High-confidence divergence in a fully
passing matrix produces `NEEDS_REVIEW`, never PASS.

Final status:

- PASS requires every case complete, activation proved, numeric finite, every
  window passing, controls passing, no lifecycle/cleanup error, flags restored,
  and binding unchanged. Label: `RAW_NUMERICAL_STRESS_ONLY`.
- Missing/incomplete case, missing activation, invalid/non-finite metric,
  changed model/common args/prefix/binary/library, failed control, unreaped
  child, or skipped flag restore is FAIL.
- `int8_default_authorized` is always `false`.

## 7. Root GOLD8 reference (read-only)

Existing root artifacts under
`/home/hjotha/.codex/artifacts/rkva-priorities-20261026/` were inspected, not
modified:

- `raw-http-gold-comparison-r1.json` reports PASS for the 8-step raw-vs-HTTP
  comparison, with maximum absolute probability
  `1.4234307094129761e-08` and per-step KL around `1e-14` (step 4 KL
  `5.405876799854657e-15`).
- `raw-gold-http-off-r1/` contains the 8 raw rows, teacher, token history, and
  manifest; its log contains the startup marker used for map sampling.

The strict manifest validator accepted that real 8-step output:
`validated [737, 65361, 737, 65361, 737, 65361, 737, 65361] 8`. A self-replay
through the shifted metric path produced `max_abs_kl 0.0` and
`max_abs_nll_delta 0.0`.

These are input-contract and arithmetic checks only. GOLD8 PASS is not a
whole-QA approval and does not authorize INT8 as a default.

## 8. CPU validation

```text
$ python3 -m py_compile tools/xbox/scripts/rkva-raw-quality-matrix.py \
    tools/xbox/tests/test_rkva_raw_quality_matrix.py
exit 0

$ PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
    tools/xbox/tests/test_rkva_raw_quality_matrix.py
Ran 46 tests in 0.700s
OK
```

Tests include all prior coverage plus:

- real startup-marker identity from the GOLD log;
- empty maps, missing startup marker, step-0 fallback;
- missing/changed loaded llama/ggml libraries;
- full prefix contract tamper: environment, slot, schema, context, hash,
  metadata SHA, file list, re-hashed forgery;
- model stat/SHA mutation, original/frozen common-args mutation, and the
  child using the frozen common-args path;
- high-offset stable NLL and shifted KL regressions;
- terminate/kill/reap failures, alive-child flag-restore skip, and final FAIL.

Inert preview against the existing 18-case campaign:

```text
$ python3 tools/xbox/scripts/rkva-raw-quality-matrix.py \
    --capture-binary .../rkva_quality_capture_0922 \
    --common-args-json .../raw-capture-gold-inputs/common-args.json \
    --cases-dir .../quality-wide-r1/cases \
    --reference-campaign .../quality-wide-r1 \
    --output /tmp/.../raw-matrix-preview
PREVIEW_ONLY 18 (prefix SHA verified=true)
```

The full prefix validator hashed the real 2.2 GB snapshot plus sidecars during
preview (CPU/disk only, no GPU). It also verified the original HTTP metadata
SHA and exact file list.

## 9. Pending real-hardware tests

The following were not run and must be done by the root owner before use:

1. Build `rkva_quality_capture` against the root checkout and run
   `--help`/`--self-test`.
2. Re-run the real GOLD8 OFF capture and compare its raw rows with the existing
   HTTP OFF distributions; confirm the root's max-abs/KL acceptance before
   adopting the transport.
3. Run one full case OFF/ON (128 steps) with `--teacher-mode greedy`, verify
   the startup marker/map set, activation counters, first_1/first_8/all
   windows, and flags restoration.
4. Run one case with `--teacher-mode http` against the frozen HTTP teacher.
5. Run `--determinism off|on|both` for the selected case.
6. Run the full 18-case × 2-arm matrix only after steps 1–5 pass; verify the
   ~4.6 GB raw budget and the narrow label.
7. Astra reviews artifacts, activation proof, bindings, cleanup records, and
   the absence of INT8 default authorization.

## 10. Remaining limitations

- No real hardware or capture child was run; the full lifecycle remains to be
  validated on the root.
- No fixed reference NLL and no free-greedy semantic accuracy are implemented;
  the catalog consumes semantic prompt tokens only, not `reference_answer_json`
  or `reference_nll_steps`.
- The default data is synthetic/numerical stress only, not broad semantic or
  external-corpus quality.
- A live child that survives SIGKILL intentionally leaves flags unrestored and
  fails the run; the root must resolve the process manually before touching the
  Xbox state again.
- No merge/push was performed. Astra review is still required.

Root hardware smoke R1 captured a valid OFF RAW arm but refused its manifest
before continuing: `GGML_RKVA_UPLOAD_FORMAT` was absent from the launch environment
and present after parsing `--remote-attn-format kvarn4`. This is the documented
`setenv(..., overwrite=1)` side effect in `common/arg.cpp`, not an unknown override.
The validator now derives exactly that runtime value from the already-validated
CLI declaration; launch provenance stays frozen and unrelated extras still fail.
Tests reject an incorrect runtime value and verify that the input environment
is not mutated. R1 remains a failed, preserved campaign; it is not relabeled.

Root R2 completed OFF/ON and both fresh-process determinism controls for8 steps.
All captures, loaded-library/slot bindings, activations and finite metrics passed;
both controls reproduced bit-identical logits, and flags were restored. INT8
failed the predeclared first-step PPL/KL/NLL gates, so the candidate remains FAIL.
A reporting correction keeps numeric case failures in the case/window checks,
not `lifecycle_errors`; a non-PASS case still prevents global PASS independently
of its window dictionary. R2's original report is preserved, with a separate
derived analysis documenting that classification. No quality limit was relaxed.
