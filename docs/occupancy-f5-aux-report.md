# F5 auxiliary report — occupancy position-split evidence harness

Date: 2026-10-05 (round 2, after independent CPU review)
Workstream: isolated OpenCode auxiliary (`opencode/mimo-v2.6-flash-free`), F5
preparation only. Snapshot: `aux-snapshot-manifest.json`, head
`d9db0b88155ec4350942306696378c57ac0485c8`, branch `feat/occupancy-position-split`,
tree dirty, captured 2026-10-05T01:16:33Z.

## 0. Round 2 — reviewer probes, red first

The independent reviewer reproduced acceptance holes. Each probe was turned
into a failing test first, then fixed. Captured red state before the fixes:

```
$ python3 -m unittest discover -s tools/occupancy-f5/tests
Ran 55 tests in 0.194s
FAILED (failures=3, errors=36)
```

| Probe | Red test | Fix | Behaviour now |
| --- | --- | --- | --- |
| `context.expected = {}` still accepted | `test_empty_expected_context_is_rejected`, `test_missing_expected_field_is_rejected` | `context_problems()` runs first in `validate` and in `load_context` (CLI exit 2) | reject with `bad_context:expected.*:missing` |
| zero placeholder hashes accepted | `test_zero_placeholder_hashes_in_context_rejected`, `test_placeholder_and_zero_hash_values_in_evidence_rejected` | 64-hex + non-zero required for `model_sha256`/`corpus_sha256` in context and for provenance hashes; placeholder strings rejected | `bad_context:expected.model_sha256:zero_placeholder_hash_not_allowed`, `zero_hash:provenance.binary_sha256`, `placeholder_value:provenance.driver:'<fill-me>'` |
| P/C alignment/range unchecked | `test_context_p_c_alignment_and_range`, `test_context_placeholder_paths_and_zero_hashes_rejected_by_cli` | `p % 256 == 0`, `0 < p`, `p < c`, `p <= c-512`, `c >= 130560`, non-placeholder paths, git id `commit` | reject for p=100000 / p==c / c=130000 / p=0 / p=c-256 |
| relabelled synthetic accepted as measured | `test_synthetic_origin_cannot_be_claimed_as_measured` | required `provenance.evidence_origin`; `measured` demands `campaign_measured` | `evidence_origin:synthetic_fixture:measured_evidence_requires_campaign_measured` |
| restore `tokens_reused=1`, `reprefill=length-1` passed | `test_restore_partial_reuse_fails_gate` | gate requires `tokens_reused == length` **and** `prefix_reprefill_tokens == 0` | `restore.130560` → fail, `partial_prefix_reuse:reused=1:need=130560`, exit 1 |
| fallback restore indistinguishable | `test_restore_fallback_scenario_cannot_pass_gate` | required `scenario` ∈ {`exact_import`,`cache_miss_fallback`}; fallback never passes the exact gate | `restore.105000` → fail, `scenario:cache_miss_fallback:exact_import_only`, exit 1 |
| only baseline arm warmed | `test_schedule_warms_both_arms_per_case`, `test_candidate_warmup_is_required` | warmup is per arm; warmups must precede measurements | schedule = 48 runs (2 warmups + 10 measured per case); missing candidate warmup → `missing_evidence:warmup:<case>:split:got=0:need=1` |
| arbitrary log could be relabelled | `test_log_header_must_match_run_and_context`, `test_binding_mismatch_detected`, `test_bindings_required` | `bindings` (`context_sha256` + 6 block hashes, recomputed) and a mandatory first-line header `f5-log-v1 run_id=… context_sha256=… provenance_sha256=…` | `missing_field:bindings`, `binding_mismatch:quality_sha256:…`, `artifact_header_mismatch:…` |
| quality/capacity/restore/cycles not tied to runs | `test_quality_run_ids_must_reference_case_arms_measured_runs`, `test_capacity_and_cycles_run_ids_required`, `test_restore_scenario_and_run_ids_required` | required `run_ids`: quality must equal the exact measured runs of its case/arm; capacity/restore/cycles must name existing measured runs (warmups excluded) | reject on missing/unknown/mismatched ids |
| missing Vulkan attention **copy** evidence below P | `test_vulkan_attention_copy_bytes_required`, `test_vulkan_attention_copy_must_be_zero_below_p` | every Vulkan stage requires `attention_copy_bytes`; must be 0 below P alongside `vulkan_attention_executed=false` | `missing_field:….attention_copy_bytes`, `route_mismatch:…:attention_copy_bytes=4096:expected=0` |
| swap only checked in one direction | `test_swap_out_bytes_required`, `test_swap_out_activity_fails_gate` | required `host.swap_out_bytes` + gate | `memory.swap_out_bytes` → fail, exit 1 |
| summary could read as real acceptance | `test_summary_never_claims_genuine_f5_acceptance` | `acceptance` block + `evidence_scope` | `genuine_f5_acceptance: false` always; fixture path reported as `fixture_or_placeholder_provenance` |

Green state after the fixes (exact output):

```
$ python3 -m unittest discover -s tools/occupancy-f5/tests
Ran 55 tests in 0.389s

OK
$ python3 -m py_compile tools/occupancy-f5/f5_harness.py \
    tools/occupancy-f5/fixtures/make_fixtures.py \
    tools/occupancy-f5/tests/test_f5_harness.py
py_compile OK
```

## 1. Scope and limits observed

Deliverables: `tools/occupancy-f5/` (harness, fixtures, CPU tests) and this
report. Observed limits, unchanged from round 1:

- No GPU, no build/compile, no server, no service, no network, no cache/model
  download, no permission change, no commit/merge/push.
- No edit outside `tools/occupancy-f5/` and `docs/occupancy-f5-aux-report.md`.
  Existing tests, `tests/CMakeLists.txt`, core sources, F4 tooling, F7 files,
  `docs/occupancy-placement-plan-20261002.md` and
  `docs/occupancy-progress-20261003.md` were read only.
- Harness imports neither `os` nor `subprocess` (asserted by test); it can only
  read/write JSON evidence and the raw logs named by that evidence.
- Runtime: Python 3.14.7, CPU only, entire suite runs in <0.6 s.
- No GPU collector or execution path was added: all new code stays inside
  `tools/occupancy-f5/`.

## 2. Artifacts produced

| Path | Lines | SHA-256 |
| --- | ---: | --- |
| `tools/occupancy-f5/f5_harness.py` | 2548 | `ae1599c78cc7668be7fc9c03ff1fc32c0d7f56dec02555df2d323bd3730d95f0` |
| `tools/occupancy-f5/fixtures/make_fixtures.py` | 437 | `c0c35a9ee2bd128d202f0dff6672b0c990bfad176ece2c1225b1776b40e01e79` |
| `tools/occupancy-f5/tests/test_f5_harness.py` | 795 | `8914ccdfab07f5cf4d04b7b17fbeb04a31abf333fa763b30716cd60d30baf170` |
| `tools/occupancy-f5/README.md` | — | evidence contract, gate table, acceptance vs schema admissibility, NOT RUN commands |
| `tools/occupancy-f5/fixtures/expected-context.json` | — | `8fa5615fdbf266299013fdcdaf52620cdaf05dc0ce7453983ccbf24d44dea1f9` |
| `tools/occupancy-f5/fixtures/complete-synthetic.json` | — | `5a203f0b096599ecd86eb52cd8522e3ef6470a30ce4fc0844cd9995e560079eb` |
| `tools/occupancy-f5/fixtures/malformed.json` | — | `f3f30e4b5d97b3536963044f7bf47223ba31d2b3305f2176fc249bce72f88239` |
| `tools/occupancy-f5/fixtures/missing-evidence.json` | — | `e991face8f8a0d036f121f4ae6d867552417939af141d3c4127ca6c0f68f8abc` |
| `tools/occupancy-f5/fixtures/nonfinite.json` | — | `d5951be1f1ae33128c1059bc1e76653b7f340130390110f75a0fa292e884bdac` |
| `tools/occupancy-f5/fixtures/wrong-provenance.json` | — | `1bd9cd519a0f6162d1d92beb531d6c4ba87d86c6e9115a2db74154ee04eea55a` |
| `tools/occupancy-f5/fixtures/insufficient-repetitions.json` | — | `4565bb01c06ff43a6a4ce44706f0bff513b694e02472bd1ccd3e04dbe907b558` |

Schemas: evidence `f5-evidence-v1`, context `f5-context-v1`, report
`f5-report-v1`, summary `f5-summary-v1`, schedule `f5-schedule-v1`, templates
`f5-templates-v1`.

## 3. Commands actually executed (real output)

Fixture regeneration (deterministic, 7 files) — `wrote …/complete-synthetic.json`
through `…/wrong-provenance.json`, exit 0.

Validation of each fixture (`validate <file> --context expected-context.json`),
exit code 1 for all six evidence files:

```
complete-synthetic         pass  reject evidence_kind:synthetic_fixture      []
malformed                  fail  reject evidence_kind:missing_or_unknown     ['malformed_json:… Unterminated string … (char 54)']
missing-evidence           fail  reject evidence_kind:synthetic_fixture      ['binding_mismatch:runs_sha256:…', 'binding_mismatch:quality_sha256:…']
nonfinite                  fail  reject evidence_kind:synthetic_fixture      ['binding_mismatch:runs_sha256:…', 'binding_mismatch:quality_sha256:…']
wrong-provenance           fail  reject evidence_kind:synthetic_fixture      ["provenance_mismatch:model_sha256:…", 'binding_mismatch:provenance_sha256:…']
insufficient-repetitions   fail  reject evidence_kind:synthetic_fixture      ['binding_mismatch:runs_sha256:…', 'insufficient_repetitions:below_p_96000:xxl:got=3:need=5']
```

CLI smoke: `plan` → 48 entries, first four for `below_p_96000` are
`[('xxl', True), ('split', True), ('xxl', False), ('split', False)]` (both arms
warmed, then alternating); `schema`, `template`, `init` exit 0;
`validate complete-synthetic` exit 1.

Admissible-path exercise (fixture relabelled to `measured` with `origin`
switched to `campaign_measured`, raw logs written under a temp artifact root):

```
healthy: exit 0  gates 38 pass / 0 fail / 0 incomplete
         evidence_scope = fixture_or_placeholder_provenance
         acceptance = {schema_admissibility: true, gate_arithmetic_only: true,
                       genuine_f5_acceptance: false, reason: "..."}
PROBE2 partial reuse: exit 1  restore.130560 = fail  partial_prefix_reuse:reused=1:need=130560
PROBE2 fallback:      exit 1  restore.105000 = fail  scenario:cache_miss_fallback:exact_import_only
swap_out activity:    exit 1  memory.swap_out_bytes = fail (value 1)
gate ids: above_p_105000, above_p_115000, above_p_130560, below_p_96000,
          capacity, cycles, memory, restore — total 38
```

**Reading of that block:** 38 is the number of gate *computations* the harness
performs on invented fixture values. It proves the formulas, the plumbing
(medians, nearest-rank p95, ratio gates, guard limits, hash bindings) execute
end to end. It is **not** 37 (now 38) real F5 gates passed, not a measurement,
and the summary itself labels it `gate_arithmetic_only` with
`genuine_f5_acceptance: false`. No F5 latency, quality, memory, capacity,
restore or cycle number exists in this snapshot.

## 4. Coverage against the F5 requirement

| Requirement (plan / F5 row) | Implementation | Status |
| --- | --- | --- |
| Warmup discarded + alternating 5x A/B per arm | `plan` + `_validate_case_runs` (warmup **per arm**, warmups before measurements, ≥5 measured/arm) | ready, CPU-tested |
| Lengths 96000 (baseline XXL), 105000/115000/130560 (baseline XXXL), P+256 if P≠102400 | `prescribed_cases(p)`; unexpected/missing case = reject | ready, CPU-tested |
| Per-run medians and p95 token latency | `_arm_aggregate` (median of per-run medians; nearest-rank p95) | ready, CPU-tested |
| Below-P prefill/TTFT/decode-p95 ≤ 1.05x | `*.prefill_regression`, `*.ttft_regression`, `*.decode_p95_regression` | limit encoded; **measurement NOT RUN** |
| Above-P prefill/TTFT ≤ 0.90x, decode p95 ≤ 1.05x | `*.prefill_improvement`, `*.ttft_improvement`, `*.decode_p95_regression` | limit encoded; **measurement NOT RUN** |
| Quality KL ≤ baseline + 0.001 nat/token, ppl ≤ 1.01x | `*.quality_kl_mean`, `*.quality_perplexity` | limit encoded; **KL/ppl NOT RUN** |
| No Vulkan attention below P | `routes.vulkan_attention_executed=false` below P **and** `stages[*].attention_copy_bytes=0` (missing key = reject) | ready, CPU-tested |
| Capacity 131072 / reject 131073 / generate 130560+512 | `capacity.*` + 3 gates + `run_ids` | ready, CPU-tested |
| 10 restore/divergence cycles, growth ≤ 64 MiB/backend, no OOM/driver/corruption | `cycles` (10, `run_ids`) + `cycles.all_cycles_clean`, `cycles.retained_growth` | ready, CPU-tested |
| Exact restore/import: full reuse, no re-prefill | `scenario=exact_import`, `tokens_reused==length`, `prefix_reprefill_tokens==0`; fallback/partial = explicit fail | ready, CPU-tested |
| Telemetry peaks per backend/stage, GTT, MemAvailable, swap | `_validate_telemetry` requires `cuda/vulkan/host` × stages + GTT/available/swap in **and out** | schema ready; **data NOT RUN** |
| CUDA guard `max(512 MiB, 5% VRAM)`, host ≥ 2 GiB, no swap | `memory.cuda_guard_margin`, `memory.host_mem_available_min`, `memory.swap_in_bytes`, `memory.swap_out_bytes` | ready, CPU-tested |
| Reproducibility (hashes, command, commit, hardware) | full `context_problems` validation + exact provenance match + placeholder/zero-hash rejection | ready, CPU-tested |
| Evidence integrity (no relabelled data) | `bindings` (context + 6 block hashes), artifact header binding, `run_ids` coverage | ready, CPU-tested |
| Distinguish schema admissibility from acceptance | `acceptance` block + `evidence_scope` in every summary | ready, CPU-tested |

## 5. Open items this workstream could not close

1. **There is still no measured F5 result.** No latency, quality, memory,
   capacity, restore or cycle data exists in this snapshot; nothing here may be
   quoted as an F5 number.
2. `evidence_origin=campaign_measured` is an *attestation field*, not a proof.
   The tool can reject placeholders, mismatches and broken hash bindings, but it
   cannot verify authorship; that is why `genuine_f5_acceptance` is hard-wired
   to `false`.
3. Command placeholders must be filled from recorded F3/F4 profile commands:
   `{KV_AND_OVERFLOW_FLAGS_FROM_F3_F4_PROFILE}`,
   `{XXL_CUDA_ONLY_FLAGS_RECORDED_BY_CAMPAIGN_OWNER}`,
   `{XXXL_PER_LAYER_MIXED_FLAGS_RECORDED_BY_CAMPAIGN_OWNER}`. Not guessed.
4. Fixture context carries synthetic (non-zero, non-placeholder) paths and
   hashes on purpose: real campaign values must replace them, otherwise
   `provenance_mismatch` fires.
5. P itself is not fixed here: plan §3.8 requires choosing P from a complete
   peak measurement before the campaign context can be finalised.
6. `--spec-type none --spec-draft-n-max 0` follows plan §3.7 (MTP out of scope).
   Any deviation must be reflected in `provenance.command`.

## 6. Remaining owner actions (concrete, in order)

1. After F4 is green, measure P per §3.8 on the real hardware; write the
   campaign context with real `model_sha256`/`corpus_sha256` and the verbatim
   commands. `validate` must not report any `bad_context:*` reason.
2. Run the `plan` schedule (48 entries per 4 cases: warmup `xxl`+`warmup split`,
   then 5 alternating measured pairs). Emit one raw log per measured run with
   the `f5-log-v1 run_id=… context_sha256=… provenance_sha256=…` first line.
3. Collect per run: prefill, TTFT, ≥32 token latencies, telemetry peaks per
   backend/stage, Vulkan `attention_copy_bytes`, GTT and host headroom,
   `swap_in_bytes`/`swap_out_bytes`, and the attention route flag.
4. Collectors for quality (KL mean, ppl), capacity, restore (`scenario`,
   full reuse, zero re-prefill, `run_ids`) and the 10 cycles; then recompute
   `bindings` (`f5_harness.py` does this for a bundle only via
   `apply_bindings`, used by fixtures — the campaign collector must compute the
   same canonical hashes).
5. `validate … --artifact-root <root>` must exit 0 before anyone reads a
   number; then `summarize … --require-gates-pass` must exit 0 **and** the
   owner must still confirm acceptance out of band — the tool will keep
   reporting `genuine_f5_acceptance: false`.
6. F7 canary/rollback and the final acceptance decision remain with the
   main worker; this workstream closes nothing of that.

## 7. Explicit non-claims

No `llama-server`, `llama-bench`, `llama-perplexity`, `cmake`, `ctest` or
`git` command was run for F5. No GPU, Vulkan or CUDA device was touched. No
service was started, stopped or reconfigured. Nothing was committed, merged or
pushed. The 55 passing tests are CPU unit tests over synthetic fixtures; the 38
gate computations exercised by them run on invented values labelled
`fixture_or_placeholder_provenance`. None of it is evidence for or against the
position-split feature, and none of it constitutes F5 acceptance.
