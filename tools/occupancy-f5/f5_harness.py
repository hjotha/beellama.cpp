#!/usr/bin/env python3
"""Offline F5 evidence harness for the occupancy position-split campaign.

CPU-only and offline. It validates, schedules and summarises F5 campaign
evidence bundles described by `docs/occupancy-placement-plan-20261002.md`
(section 4, phase 5, and the protocol/limits of section 4.1).

This tool never launches a server, never compiles anything, never touches a
GPU and never performs network access. It reads JSON files and writes JSON
reports.

Subcommands:
  schema     print the machine-readable evidence schema
  plan       emit the reproducible warmup + alternating A/B schedule
  template   emit server/GPU command templates (documentation only, NOT RUN)
  init       scaffold an evidence bundle (always marked synthetic)
  validate   admissibility: structure, completeness, finiteness, provenance,
             repetitions, alternation, artifacts
  summarize  compute gates from admissible evidence; never invents numbers

Exit codes: 0 accept/admissible, 1 reject/inadmissible, 2 usage or I/O error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA = "f5-evidence-v2"
CONTEXT_SCHEMA = "f5-context-v1"
REPORT_SCHEMA = "f5-report-v1"
SUMMARY_SCHEMA = "f5-summary-v2"
SCHEDULE_SCHEMA = "f5-schedule-v1"
TEMPLATE_SCHEMA = "f5-templates-v1"
SCHEMA_DOC_SCHEMA = "f5-schema-doc-v1"

MIB = 1024 * 1024
GIB = 1024 * MIB

PLAN_DEFAULT_P = 102400
CAPACITY_C = 131072
PLAN_BELOW_P_LENGTH = 96000
PLAN_ABOVE_P_LENGTHS = (105000, 115000, 130560)

MIN_WARMUP_RUNS = 1
MIN_MEASURED_RUNS_PER_ARM = 5
DEFAULT_MIN_TOKEN_SAMPLES = 32

MIN_CUDA_GUARD_MIB = 512
MIN_CUDA_GUARD_FRACTION = 0.05
MIN_MEM_AVAILABLE_BYTES = 2 * GIB
RETAINED_GROWTH_LIMIT_BYTES = 64 * MIB
RESTORE_CYCLES = 10

CANDIDATE_ARM = "split"
XXL_ARM = "xxl"
XXXL_ARM = "xxxl"
FIRST_ARM_BASELINE = "baseline"
FIRST_ARM_CANDIDATE = "candidate"

EVIDENCE_KINDS = ("measured", "synthetic_fixture")
CACHE_STATES = ("cold", "warm")
QUALITY_REFERENCE = "f16"

TELEMETRY_STAGES = ("load", "prefill", "decode", "restore")
BOUNDARY_STAGE = "boundary_cross"
TELEMETRY_BACKENDS = ("cuda", "vulkan", "host")

LATENCY_GATE_LIMIT_REGRESSION = 1.05
PREFILL_GATE_LIMIT_IMPROVEMENT = 0.90
QUALITY_KL_DELTA_LIMIT = 0.001
QUALITY_PPL_RATIO_LIMIT = 1.01

MISSING = object()


class HarnessError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def load_json_file(path: str | Path) -> Any:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise HarnessError("io_error", f"{p}: {exc}") from exc
    try:
        return json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise HarnessError("malformed_json", f"{p}: {exc}") from exc


def dump_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in value)


def percentile_nearest_rank(values: list[float], quantile: float = 0.95) -> float:
    if not values:
        raise ValueError("percentile of empty sample")
    ordered = sorted(float(v) for v in values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def min_cuda_guard_bytes(vram_total_bytes: int) -> int:
    return max(
        MIN_CUDA_GUARD_MIB * MIB,
        int(math.ceil(MIN_CUDA_GUARD_FRACTION * vram_total_bytes)),
    )


EVIDENCE_ORIGINS = ("campaign_measured", "synthetic_fixture")
RESTORE_SCENARIOS = ("exact_import", "cache_miss_fallback")
BINDING_BLOCKS = ("provenance", "runs", "quality", "capacity", "restore", "cycles")

REQUIRED_EXPECTED_KEYS = (
    "commit",
    "branch",
    "model_path",
    "model_sha256",
    "corpus_path",
    "corpus_sha256",
    "p",
    "c",
    "seed",
    "batch_size",
    "ubatch_size",
)
REQUIRED_PROTOCOL_KEYS = (
    "warmup_runs",
    "measured_runs_per_arm",
    "alternating",
    "first_arm",
    "min_token_samples",
)

LOG_HEADER_PREFIX = "f5-log-v1"


def canonical_sha256(value: Any) -> str:
    blob = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True
    ).encode("utf-8")
    return sha256_of(blob)


def context_binding(context: dict) -> str:
    return canonical_sha256(
        {"expected": context.get("expected", {}), "protocol": context.get("protocol", {})}
    )


def recompute_bindings(bundle: dict, context: dict) -> dict:
    bindings = {"context_sha256": context_binding(context)}
    for block in BINDING_BLOCKS:
        if block in bundle:
            bindings[f"{block}_sha256"] = canonical_sha256(bundle[block])
    return bindings


def apply_bindings(bundle: dict, context: dict) -> dict:
    bundle["bindings"] = recompute_bindings(bundle, context)
    return bundle["bindings"]


def is_zero_hash(value: Any) -> bool:
    return isinstance(value, str) and value.strip("0") == "" and len(value) >= 8


def is_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return True
    if text == PLACEHOLDER or "REPLACE_BEFORE_MEASUREMENT" in text:
        return True
    if "{PLACEHOLDER}" in text or "PLACEHOLDER" in text:
        return True
    if text.startswith("<"):
        return True
    if "<" in text and ">" in text:
        return True
    if "{MODEL_PATH}" in text or "{PORT}" in text:
        return True
    return False


PLACEHOLDER = "REPLACE_BEFORE_MEASUREMENT"


def _bad(field: str, detail: str) -> str:
    return f"bad_context:{field}:{detail}"


def context_problems(context: dict) -> list[str]:
    problems: list[str] = []
    if not isinstance(context, dict):
        return ["bad_context:root:not_an_object"]
    if context.get("schema_version") != CONTEXT_SCHEMA:
        problems.append(
            _bad("schema_version", f"expected={CONTEXT_SCHEMA!r}:got={context.get('schema_version')!r}")
        )
    expected = context.get("expected")
    protocol = context.get("protocol")
    if not isinstance(expected, dict):
        problems.append(_bad("expected", "missing_object"))
        expected = {}
    if not isinstance(protocol, dict):
        problems.append(_bad("protocol", "missing_object"))
        protocol = {}

    for key in REQUIRED_EXPECTED_KEYS:
        if key not in expected:
            problems.append(_bad(f"expected.{key}", "missing"))

    commit = expected.get("commit")
    if "commit" in expected:
        if not isinstance(commit, str) or not all(
            c in "0123456789abcdefABCDEF" for c in commit
        ) or not (40 <= len(commit) <= 64):
            problems.append(_bad("expected.commit", f"not_a_git_object_id:{commit!r}"))
        elif is_placeholder(commit):
            problems.append(_bad("expected.commit", f"placeholder:{commit!r}"))
    for key in ("branch", "model_path", "corpus_path"):
        if key in expected:
            value = expected[key]
            if not isinstance(value, str) or not value:
                problems.append(_bad(f"expected.{key}", f"not_a_nonempty_string:{value!r}"))
            elif is_placeholder(value):
                problems.append(_bad(f"expected.{key}", f"placeholder:{value!r}"))
    for key in ("model_sha256", "corpus_sha256"):
        if key in expected:
            value = expected[key]
            if not is_sha256(value):
                problems.append(_bad(f"expected.{key}", f"not_sha256_hex:{value!r}"))
            elif is_zero_hash(value):
                problems.append(_bad(f"expected.{key}", "zero_placeholder_hash_not_allowed"))
            elif is_placeholder(value):
                problems.append(_bad(f"expected.{key}", f"placeholder:{value!r}"))

    p = expected.get("p")
    c = expected.get("c")
    if "p" in expected:
        if not is_int(p):
            problems.append(_bad("expected.p", f"not_an_int:{p!r}"))
        elif p % 256 != 0:
            problems.append(_bad("expected.p", f"not_a_multiple_of_256:{p}"))
        elif p <= 0:
            problems.append(_bad("expected.p", f"must_be_positive:{p}"))
    if "c" in expected:
        if not is_int(c):
            problems.append(_bad("expected.c", f"not_an_int:{c!r}"))
        elif c < PLAN_ABOVE_P_LENGTHS[-1]:
            problems.append(
                _bad("expected.c", f"too_small_for_prescribed_cases:{c}<{PLAN_ABOVE_P_LENGTHS[-1]}")
            )
    if is_int(p) and is_int(c):
        if p >= c:
            problems.append(_bad("expected.p", f"must_be_less_than_c:{p}>={c}"))
        elif p > c - 512:
            problems.append(_bad("expected.p", f"insufficient_headroom:{p}>c-512={c - 512}"))

    for key in ("seed", "batch_size", "ubatch_size"):
        if key in expected:
            value = expected[key]
            if not is_int(value):
                problems.append(_bad(f"expected.{key}", f"not_an_int:{value!r}"))
            elif value < 0 or (key != "seed" and value < 1):
                problems.append(_bad(f"expected.{key}", f"out_of_range:{value}"))

    for key in REQUIRED_PROTOCOL_KEYS:
        if key not in protocol:
            problems.append(_bad(f"protocol.{key}", "missing"))
    if "warmup_runs" in protocol:
        value = protocol["warmup_runs"]
        if not is_int(value) or value < MIN_WARMUP_RUNS:
            problems.append(
                _bad("protocol.warmup_runs", f"must_be_int_ge_{MIN_WARMUP_RUNS}:{value!r}")
            )
    if "measured_runs_per_arm" in protocol:
        value = protocol["measured_runs_per_arm"]
        if not is_int(value) or value < MIN_MEASURED_RUNS_PER_ARM:
            problems.append(
                _bad(
                    "protocol.measured_runs_per_arm",
                    f"must_be_int_ge_{MIN_MEASURED_RUNS_PER_ARM}:{value!r}",
                )
            )
    if protocol.get("alternating") is not True:
        problems.append(_bad("protocol.alternating", f"must_be_true:{protocol.get('alternating')!r}"))
    if protocol.get("first_arm") not in (FIRST_ARM_BASELINE, FIRST_ARM_CANDIDATE):
        problems.append(_bad("protocol.first_arm", f"invalid:{protocol.get('first_arm')!r}"))
    if "min_token_samples" in protocol:
        value = protocol["min_token_samples"]
        if not is_int(value) or value < DEFAULT_MIN_TOKEN_SAMPLES:
            problems.append(
                _bad(
                    "protocol.min_token_samples",
                    f"must_be_int_ge_{DEFAULT_MIN_TOKEN_SAMPLES}:{value!r}",
                )
            )
    return problems


def load_context(path: str | Path) -> dict:
    raw = load_json_file(path)
    problems = context_problems(raw)
    if problems:
        raise HarnessError(
            "bad_context", f"{path}: " + "; ".join(problems[:8])
        )
    return raw


def below_p_length(p: int) -> int:
    if p > PLAN_BELOW_P_LENGTH + 4096:
        return PLAN_BELOW_P_LENGTH
    return max(4096, ((p // 2) // 256) * 256)


def prescribed_cases(p: int) -> list[dict]:
    cases = [
        {
            "case_id": f"below_p_{below_p_length(p)}",
            "kind": "below_p",
            "length_tokens": below_p_length(p),
            "baseline_arm": XXL_ARM,
        }
    ]
    for length in PLAN_ABOVE_P_LENGTHS:
        cases.append(
            {
                "case_id": f"above_p_{length}",
                "kind": "above_p",
                "length_tokens": length,
                "baseline_arm": XXXL_ARM,
            }
        )
    if p != PLAN_DEFAULT_P:
        cases.append(
            {
                "case_id": f"above_p_{p + 256}",
                "kind": "above_p",
                "length_tokens": p + 256,
                "baseline_arm": XXXL_ARM,
            }
        )
    return cases


RESTORE_REQUIRED_LENGTHS_EXTRA = (CAPACITY_C,)


def required_restore_lengths(cases: list[dict]) -> list[int]:
    lengths = {case["length_tokens"] for case in cases}
    lengths.update(RESTORE_REQUIRED_LENGTHS_EXTRA)
    return sorted(lengths)


class _Report:
    def __init__(self) -> None:
        self.checks: dict[str, dict[str, Any]] = {}
        self.reasons: list[str] = []
        self.kind: str | None = None

    def fail(self, check_id: str, reason: str) -> None:
        entry = self.checks.setdefault(check_id, {"status": "pass", "reasons": []})
        entry["status"] = "fail"
        if reason not in entry["reasons"]:
            entry["reasons"].append(reason)
        if reason not in self.reasons:
            self.reasons.append(reason)

    def ok(self, check_id: str) -> None:
        self.checks.setdefault(check_id, {"status": "pass", "reasons": []})

    @property
    def structural_status(self) -> str:
        return "fail" if self.reasons else "pass"

    @property
    def kind_reason(self) -> str | None:
        if self.kind == "measured":
            return None
        if self.kind is None:
            return "evidence_kind:missing_or_unknown"
        return f"evidence_kind:{self.kind}"

    @property
    def verdict(self) -> str:
        if self.reasons:
            return "reject"
        return "accept" if self.kind == "measured" else "reject"

    def as_dict(self) -> dict:
        return {
            "schema_version": REPORT_SCHEMA,
            "verdict": self.verdict,
            "structural_status": self.structural_status,
            "evidence_kind": self.kind,
            "reasons": list(self.reasons),
            "kind_reason": self.kind_reason,
            "checks": {
                k: {"status": v["status"], "reasons": list(v["reasons"])}
                for k, v in self.checks.items()
            },
        }


def _check_str(
    rep: _Report, check: str, mapping: dict, key: str, base: str, nonempty: bool = True
) -> str | None:
    path = f"{base}.{key}"
    if key not in mapping:
        rep.fail(check, f"missing_field:{path}")
        return None
    value = mapping[key]
    if not isinstance(value, str):
        rep.fail(check, f"invalid_type:{path}")
        return None
    if nonempty and not value.strip():
        rep.fail(check, f"invalid_value:{path}:empty")
        return None
    return value


def _check_bool(rep: _Report, check: str, mapping: dict, key: str, base: str) -> bool | None:
    path = f"{base}.{key}"
    if key not in mapping:
        rep.fail(check, f"missing_field:{path}")
        return None
    value = mapping[key]
    if not isinstance(value, bool):
        rep.fail(check, f"invalid_type:{path}")
        return None
    return value


def _check_int(
    rep: _Report, check: str, mapping: dict, key: str, base: str, minimum: int = 0
) -> int | None:
    path = f"{base}.{key}"
    if key not in mapping:
        rep.fail(check, f"missing_field:{path}")
        return None
    value = mapping[key]
    if not is_int(value):
        rep.fail(check, f"invalid_type:{path}")
        return None
    if value < minimum:
        rep.fail(check, f"invalid_value:{path}:{value}")
        return None
    return value


def _check_number(
    rep: _Report, check: str, mapping: dict, key: str, base: str, positive: bool = True
) -> float | None:
    path = f"{base}.{key}"
    if key not in mapping:
        rep.fail(check, f"missing_field:{path}")
        return None
    value = mapping[key]
    if not is_number(value):
        rep.fail(check, f"invalid_type:{path}")
        return None
    if not math.isfinite(float(value)):
        rep.fail(check, f"non_finite:{path}")
        return None
    if positive and float(value) <= 0.0:
        rep.fail(check, f"invalid_value:{path}:{value}")
        return None
    return float(value)


def _check_sha(rep: _Report, check: str, mapping: dict, key: str, base: str) -> str | None:
    path = f"{base}.{key}"
    if key not in mapping:
        rep.fail(check, f"missing_field:{path}")
        return None
    value = mapping[key]
    if not is_sha256(value):
        rep.fail(check, f"invalid_value:{path}:expected_sha256")
        return None
    return str(value).lower()


PROVENANCE_REQUIRED_STRINGS = (
    "commit",
    "branch",
    "model_path",
    "model_sha256",
    "corpus_path",
    "corpus_sha256",
    "prompt_sha256",
    "binary_path",
    "binary_sha256",
    "driver",
    "build",
    "governor",
    "concurrency",
    "collected_at_utc",
    "collector",
    "command",
)
PROVENANCE_MATCH_KEYS = (
    "commit",
    "branch",
    "model_sha256",
    "corpus_sha256",
    "p",
    "c",
    "seed",
    "batch_size",
    "ubatch_size",
)
PROTOCOL_KEYS = (
    "warmup_runs",
    "measured_runs_per_arm",
    "alternating",
    "first_arm",
    "min_token_samples",
)
CAPACITY_KEYS = (
    "occupied_positions",
    "over_capacity_tokens",
    "over_capacity_rejected",
    "over_capacity_state_intact",
    "gen_start_tokens",
    "gen_count",
    "gen_end_tokens",
    "gen_ok",
    "run_ids",
)
CYCLE_KEYS = (
    "index",
    "restore_ok",
    "divergent_suffix_ok",
    "generation_ok",
    "retained_bytes",
    "oom_count",
    "driver_error_count",
    "corruption_count",
    "run_ids",
)
RESTORE_KEYS = (
    "length_tokens",
    "tokens_reused",
    "prefix_reprefill_tokens",
    "next_logits_max_abs_diff",
    "ok",
    "scenario",
    "run_ids",
)
QUALITY_KEYS = (
    "case_id",
    "arm",
    "kl_mean_nat_per_token",
    "perplexity",
    "reference",
    "run_ids",
)
CASE_KEYS = (
    "case_id",
    "kind",
    "length_tokens",
    "baseline_arm",
)
RUN_KEYS_ALWAYS = (
    "run_id",
    "case_id",
    "arm",
    "warmup",
    "sequence_index",
    "prefill_ms",
    "ttft_ms",
    "token_latencies_ms",
    "oom_count",
)


def _resolve_artifact(root: Path | None, rel: Any) -> Path | None:
    if root is None:
        return None
    if not isinstance(rel, str) or not rel or rel.startswith("/") or "\\" in rel:
        return None
    parts = Path(rel).parts
    if any(part in ("..", "") for part in parts):
        return None
    resolved_root = root.resolve()
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        return None
    return candidate


def _validate_telemetry(rep: _Report, run: dict, base: str, above_p: bool) -> None:
    telemetry = run.get("telemetry")
    if not isinstance(telemetry, dict):
        rep.fail("telemetry", f"missing_field:{base}.telemetry")
        return
    for backend in telemetry:
        if backend not in TELEMETRY_BACKENDS:
            rep.fail("telemetry", f"invalid_value:{base}.telemetry.{backend}:unknown_backend")
    for backend in TELEMETRY_BACKENDS:
        bpath = f"{base}.telemetry.{backend}"
        block = telemetry.get(backend)
        if not isinstance(block, dict):
            rep.fail("telemetry", f"missing_field:{bpath}")
            continue
        stages = block.get("stages")
        if not isinstance(stages, dict):
            rep.fail("telemetry", f"missing_field:{bpath}.stages")
            continue
        required_stages = list(TELEMETRY_STAGES)
        if above_p:
            required_stages.append(BOUNDARY_STAGE)
        for stage in required_stages:
            spath = f"{bpath}.stages.{stage}"
            record = stages.get(stage)
            if not isinstance(record, dict):
                rep.fail("telemetry", f"missing_field:{spath}")
                continue
            if "peak_bytes" not in record:
                rep.fail("telemetry", f"missing_field:{spath}.peak_bytes")
                continue
            peak = record["peak_bytes"]
            if not is_int(peak) or peak < 0:
                rep.fail("telemetry", f"invalid_value:{spath}.peak_bytes")
            if backend == "vulkan":
                if "attention_copy_bytes" not in record:
                    rep.fail(
                        "telemetry", f"missing_field:{spath}.attention_copy_bytes"
                    )
                    continue
                copied = record["attention_copy_bytes"]
                if not is_int(copied) or copied < 0:
                    rep.fail(
                        "telemetry", f"invalid_value:{spath}.attention_copy_bytes"
                    )
                elif not above_p and copied != 0:
                    rep.fail(
                        "routes",
                        f"route_mismatch:{spath}:attention_copy_bytes={copied}:expected=0",
                    )
        if backend == "vulkan":
            for key in ("gtt_peak_bytes", "gtt_available_min_bytes"):
                _check_int(rep, "telemetry", block, key, bpath, minimum=0)
        if backend == "host":
            for key in (
                "mem_available_min_bytes",
                "swap_in_bytes",
                "swap_out_bytes",
            ):
                _check_int(rep, "telemetry", block, key, bpath, minimum=0)


def _check_log_header(
    target: Path, run_id: Any, context_hash: str, provenance_hash: str
) -> tuple[bool, str]:
    try:
        with target.open("rb") as handle:
            first_line = handle.readline(4096)
    except OSError as exc:
        return False, f"unreadable:{exc}"
    try:
        text = first_line.decode("utf-8").strip()
    except UnicodeDecodeError:
        return False, "not_utf8"
    if not text.startswith(f"{LOG_HEADER_PREFIX} "):
        return False, f"missing_prefix:{text[:40]!r}"
    tokens: dict[str, str] = {}
    for field in text[len(LOG_HEADER_PREFIX) + 1 :].split():
        if "=" in field:
            key, value = field.split("=", 1)
            tokens[key] = value
    if tokens.get("run_id") != str(run_id):
        return False, f"run_id:expected={run_id!r}:got={tokens.get('run_id')!r}"
    if tokens.get("context_sha256") != context_hash:
        return False, (
            f"context_sha256:expected={context_hash}:got={tokens.get('context_sha256')!r}"
        )
    if tokens.get("provenance_sha256") != provenance_hash:
        return False, (
            "provenance_sha256:expected="
            f"{provenance_hash}:got={tokens.get('provenance_sha256')!r}"
        )
    return True, ""


def _validate_artifacts(
    rep: _Report,
    run: dict,
    base: str,
    root: Path | None,
    context_hash: str | None,
    provenance_hash: str | None,
) -> None:
    artifacts = run.get("artifacts")
    if artifacts is MISSING:
        rep.fail("artifacts", f"missing_field:{base}.artifacts")
        return
    if not isinstance(artifacts, list):
        rep.fail("artifacts", f"invalid_type:{base}.artifacts")
        return
    if not artifacts:
        rep.fail("artifacts", f"missing_evidence:{base}.artifacts")
        return
    for index, entry in enumerate(artifacts):
        ebase = f"{base}.artifacts[{index}]"
        if not isinstance(entry, dict):
            rep.fail("artifacts", f"invalid_type:{ebase}")
            continue
        rel = _check_str(rep, "artifacts", entry, "path", ebase)
        digest = _check_sha(rep, "artifacts", entry, "sha256", ebase)
        if rel is None or digest is None:
            continue
        if rel.startswith("/") or ".." in Path(rel).parts:
            rep.fail("artifacts", f"invalid_value:{ebase}.path:unsafe_relative_path")
            continue
        if root is None:
            continue
        target = _resolve_artifact(root, rel)
        if target is None or not target.is_file():
            rep.fail("artifacts", f"artifact_missing:{ebase}.path:{rel}")
            continue
        if sha256_file(target) != digest.lower():
            rep.fail("artifacts", f"artifact_hash_mismatch:{ebase}.path:{rel}")
            continue
        if context_hash is None or provenance_hash is None:
            continue
        ok, detail = _check_log_header(
            target, run.get("run_id"), context_hash, provenance_hash
        )
        if not ok:
            rep.fail("artifacts", f"artifact_header_mismatch:{ebase}:path={rel}:{detail}")


def _validate_run_metrics(
    rep: _Report, run: dict, base: str, min_samples: int
) -> None:
    _check_number(rep, "run_metrics", run, "prefill_ms", base)
    _check_number(rep, "run_metrics", run, "ttft_ms", base)
    path = f"{base}.token_latencies_ms"
    if "token_latencies_ms" not in run:
        rep.fail("run_metrics", f"missing_field:{path}")
        return
    samples = run["token_latencies_ms"]
    if not isinstance(samples, list):
        rep.fail("run_metrics", f"invalid_type:{path}")
        return
    if len(samples) < min_samples:
        rep.fail(
            "token_samples",
            f"missing_evidence:{path}:got={len(samples)}:need={min_samples}",
        )
    for index, value in enumerate(samples):
        element = f"{path}[{index}]"
        if not is_number(value):
            rep.fail("run_metrics", f"invalid_type:{element}")
            continue
        if not math.isfinite(float(value)):
            rep.fail("run_metrics", f"non_finite:{element}")
            continue
        if float(value) <= 0.0:
            rep.fail("run_metrics", f"invalid_value:{element}:{value}")


def _validate_case_runs(
    rep: _Report,
    case: dict,
    case_runs: list[dict],
    protocol: dict,
) -> None:
    case_id = case["case_id"]
    warmup_runs = protocol.get("warmup_runs", MIN_WARMUP_RUNS)
    per_arm = protocol.get("measured_runs_per_arm", MIN_MEASURED_RUNS_PER_ARM)
    first_arm = protocol.get("first_arm", FIRST_ARM_BASELINE)
    expected_first = case["baseline_arm"] if first_arm == FIRST_ARM_BASELINE else CANDIDATE_ARM

    warmups = [r for r in case_runs if r.get("warmup") is True]
    measured = [r for r in case_runs if r.get("warmup") is False]
    for arm in (case["baseline_arm"], CANDIDATE_ARM):
        arm_warmups = sum(1 for r in warmups if r.get("arm") == arm)
        if arm_warmups < warmup_runs:
            rep.fail(
                "warmup",
                f"missing_evidence:warmup:{case_id}:{arm}:"
                f"got={arm_warmups}:need={warmup_runs}",
            )

    for arm in (case["baseline_arm"], CANDIDATE_ARM):
        count = sum(1 for r in measured if r.get("arm") == arm)
        if count < per_arm:
            rep.fail(
                "repetitions",
                f"insufficient_repetitions:{case_id}:{arm}:got={count}:need={per_arm}",
            )

    indices = [r.get("sequence_index") for r in case_runs]
    if not all(is_int(i) for i in indices) or len(set(indices)) != len(indices):
        rep.fail("run_order", f"invalid_value:{case_id}:sequence_index_not_unique")
        return
    list_sequence = [r.get("sequence_index") for r in case_runs]
    if list_sequence != sorted(list_sequence):
        rep.fail(
            "run_order",
            f"invalid_value:{case_id}:sequence_index_order_not_monotonic_in_run_list",
        )
    ordered = sorted(case_runs, key=lambda r: r["sequence_index"])
    if warmups and measured:
        if max(r["sequence_index"] for r in warmups) >= min(
            r["sequence_index"] for r in measured
        ):
            rep.fail(
                "run_order",
                f"warmup_after_measured:{case_id}:sequence_index",
            )
    seen_measured = False
    previous_arm = None
    for run in ordered:
        warmup = run.get("warmup")
        if warmup is False:
            seen_measured = True
        elif warmup is True and seen_measured:
            rep.fail("run_order", f"warmup_after_measured:{case_id}:{run.get('run_id')}")
            previous_arm = None
            continue
        if seen_measured:
            arm = run.get("arm")
            if previous_arm is None:
                if arm != expected_first:
                    rep.fail(
                        "alternation",
                        f"alternation_violation:{case_id}:first_measured_arm={arm!r}",
                    )
            elif arm == previous_arm:
                rep.fail(
                    "alternation",
                    f"alternation_violation:{case_id}:repeated_arm={arm!r}:run={run.get('run_id')}",
                )
            previous_arm = arm


def _check_run_ids(
    rep: _Report,
    check: str,
    entry: dict,
    base: str,
    measured_run_ids: set[str],
    measured_ids_by_case_arm: dict[tuple[str, str], list[str]],
    exact_case_arm: tuple[str, str] | None = None,
    required_case_id: str | None = None,
    case_of_run: dict[str, str] | None = None,
) -> None:
    path = f"{base}.run_ids"
    if "run_ids" not in entry:
        rep.fail(check, f"missing_field:{path}")
        return
    value = entry["run_ids"]
    if not isinstance(value, list) or not value:
        rep.fail(check, f"invalid_value:{path}:expected_nonempty_list")
        return
    if any(not isinstance(item, str) for item in value):
        rep.fail(check, f"invalid_value:{path}:entries_must_be_run_ids")
        return
    if len(set(value)) != len(value):
        rep.fail(check, f"invalid_value:{path}:duplicate_run_id")
    unknown = [item for item in value if item not in measured_run_ids]
    if unknown:
        rep.fail(
            check,
            f"invalid_value:{path}:not_a_measured_run_id:{sorted(set(unknown))}",
        )
    if exact_case_arm is not None:
        expected = set(measured_ids_by_case_arm.get(exact_case_arm, []))
        if set(value) != expected:
            rep.fail(
                check,
                f"invalid_value:{path}:expected_all_measured_runs_of_"
                f"{exact_case_arm[0]}:{exact_case_arm[1]}:got={sorted(set(value))}:"
                f"need={sorted(expected)}",
            )
    if required_case_id is not None and case_of_run is not None:
        wrong = [
            item
            for item in value
            if case_of_run.get(item, required_case_id) != required_case_id
        ]
        if wrong:
            rep.fail(
                check,
                f"invalid_value:{path}:wrong_case:{sorted(set(wrong))}:"
                f"need={required_case_id}",
            )


def validate(bundle: Any, context: dict, artifact_root: str | Path | None = None) -> dict:
    rep = _Report()
    root = Path(artifact_root) if artifact_root is not None else None

    for problem in context_problems(context):
        rep.fail("context", problem)

    if not isinstance(bundle, dict):
        rep.fail("evidence_object", "not_an_object")
        return rep.as_dict()

    schema_value = bundle.get("schema_version", MISSING)
    if schema_value is MISSING:
        rep.fail("schema_version", "missing_field:schema_version")
    elif schema_value != EVIDENCE_SCHEMA:
        rep.fail("schema_version", f"schema_version:{schema_value!r}")
    else:
        rep.ok("schema_version")

    kind_value = bundle.get("evidence_kind", MISSING)
    if kind_value is MISSING:
        rep.fail("evidence_kind", "missing_field:evidence_kind")
        rep.kind = None
    elif kind_value not in EVIDENCE_KINDS:
        rep.fail("evidence_kind", f"evidence_kind:unknown:{kind_value!r}")
        rep.kind = None
    else:
        rep.kind = str(kind_value)
        rep.ok("evidence_kind")
    is_measured = rep.kind == "measured"

    expected = context.get("expected", {})
    expected_p = int(expected.get("p", PLAN_DEFAULT_P))
    expected_c = int(expected.get("c", CAPACITY_C))
    cases = prescribed_cases(expected_p)

    provenance = bundle.get("provenance", MISSING)
    if not isinstance(provenance, dict):
        rep.fail("provenance_fields", "missing_field:provenance")
    else:
        for key in PROVENANCE_REQUIRED_STRINGS:
            _check_str(rep, "provenance_fields", provenance, key, "provenance")
        for key in ("p", "c", "seed", "batch_size", "ubatch_size"):
            _check_int(rep, "provenance_fields", provenance, key, "provenance", minimum=0)
        for key in ("model_sha256", "corpus_sha256", "prompt_sha256", "binary_sha256"):
            if key in provenance:
                _check_sha(rep, "provenance_fields", provenance, key, "provenance")
        _check_bool(rep, "provenance_fields", provenance, "dirty", "provenance")
        cache_types = provenance.get("cache_types")
        if not isinstance(cache_types, dict):
            rep.fail("provenance_fields", "missing_field:provenance.cache_types")
        else:
            for key in ("local", "overflow"):
                _check_str(
                    rep, "provenance_fields", cache_types, key, "provenance.cache_types"
                )
        hardware = provenance.get("hardware")
        if not isinstance(hardware, dict):
            rep.fail("provenance_fields", "missing_field:provenance.hardware")
        else:
            _check_str(rep, "provenance_fields", hardware, "gpu", "provenance.hardware")
            _check_int(
                rep,
                "provenance_fields",
                hardware,
                "vram_total_bytes",
                "provenance.hardware",
                minimum=1,
            )
        for key in ("sampling", "clocks"):
            if not isinstance(provenance.get(key), dict):
                rep.fail("provenance_fields", f"missing_field:provenance.{key}")
        cache_state = provenance.get("cache_state", MISSING)
        if cache_state is MISSING:
            rep.fail("provenance_fields", "missing_field:provenance.cache_state")
        elif cache_state not in CACHE_STATES:
            rep.fail(
                "provenance_fields", f"invalid_value:provenance.cache_state:{cache_state!r}"
            )
        origin = provenance.get("evidence_origin", MISSING)
        if origin is MISSING:
            rep.fail("provenance_fields", "missing_field:provenance.evidence_origin")
        elif origin not in EVIDENCE_ORIGINS:
            rep.fail(
                "provenance_fields",
                f"invalid_value:provenance.evidence_origin:{origin!r}",
            )
        elif is_measured and origin != "campaign_measured":
            rep.fail(
                "evidence_origin",
                f"evidence_origin:{origin}:measured_evidence_requires_campaign_measured",
            )
        else:
            rep.ok("evidence_origin")
        for key in PROVENANCE_REQUIRED_STRINGS:
            value = provenance.get(key)
            if isinstance(value, str) and is_placeholder(value):
                rep.fail(
                    "provenance_fields", f"placeholder_value:provenance.{key}:{value!r}"
                )
        for key in ("model_sha256", "corpus_sha256", "prompt_sha256", "binary_sha256"):
            value = provenance.get(key)
            if isinstance(value, str) and is_zero_hash(value):
                rep.fail("provenance_fields", f"zero_hash:provenance.{key}")
        rep.ok("provenance_fields")

        for key in PROVENANCE_MATCH_KEYS:
            if key not in expected:
                continue
            if key not in provenance:
                rep.fail("provenance_match", f"missing_field:provenance.{key}")
            elif provenance[key] != expected[key]:
                rep.fail(
                    "provenance_match",
                    f"provenance_mismatch:{key}:expected={expected[key]!r}:"
                    f"got={provenance[key]!r}",
                )
        rep.ok("provenance_match")

    context_hash_for_artifacts = context_binding(context)
    provenance_hash_for_artifacts = (
        canonical_sha256(provenance) if isinstance(provenance, dict) else None
    )
    evidence_bindings = bundle.get("bindings", MISSING)
    if not isinstance(evidence_bindings, dict):
        rep.fail("bindings", "missing_field:bindings")
    else:
        got_context = evidence_bindings.get("context_sha256", MISSING)
        expected_context_hash = context_hash_for_artifacts
        if got_context is MISSING:
            rep.fail("bindings", "missing_field:bindings.context_sha256")
        elif not is_sha256(got_context):
            rep.fail("bindings", f"invalid_value:bindings.context_sha256:{got_context!r}")
        elif got_context != expected_context_hash:
            rep.fail(
                "bindings",
                f"binding_mismatch:context_sha256:expected={expected_context_hash}:"
                f"got={got_context}",
            )
        for block in BINDING_BLOCKS:
            key = f"{block}_sha256"
            if key not in evidence_bindings:
                rep.fail("bindings", f"missing_field:bindings.{key}")
                continue
            if block not in bundle:
                continue
            recomputed = canonical_sha256(bundle[block])
            recorded = evidence_bindings[key]
            if recorded != recomputed:
                rep.fail(
                    "bindings",
                    f"binding_mismatch:{key}:expected={recomputed}:got={recorded}",
                )
        rep.ok("bindings")

    protocol = bundle.get("protocol", MISSING)
    context_protocol = context.get("protocol", {})
    if not isinstance(protocol, dict):
        rep.fail("protocol", "missing_field:protocol")
        protocol = {}
    else:
        for key in PROTOCOL_KEYS:
            path = f"protocol.{key}"
            if key not in protocol:
                rep.fail("protocol", f"missing_field:{path}")
            elif key in context_protocol and protocol[key] != context_protocol[key]:
                rep.fail(
                    "protocol",
                    f"protocol_mismatch:{key}:expected={context_protocol[key]!r}:"
                    f"got={protocol[key]!r}",
                )
        warmup = protocol.get("warmup_runs")
        if not is_int(warmup) or warmup < MIN_WARMUP_RUNS:
            rep.fail("protocol", f"invalid_value:protocol.warmup_runs:{warmup!r}")
        measured = protocol.get("measured_runs_per_arm")
        if not is_int(measured) or measured < MIN_MEASURED_RUNS_PER_ARM:
            rep.fail(
                "protocol", f"invalid_value:protocol.measured_runs_per_arm:{measured!r}"
            )
        if protocol.get("alternating") is not True:
            rep.fail(
                "protocol",
                f"invalid_value:protocol.alternating:{protocol.get('alternating')!r}",
            )
        first_arm = protocol.get("first_arm")
        if first_arm not in (FIRST_ARM_BASELINE, FIRST_ARM_CANDIDATE):
            rep.fail("protocol", f"invalid_value:protocol.first_arm:{first_arm!r}")
        min_samples = protocol.get("min_token_samples")
        if not is_int(min_samples) or min_samples < 1:
            rep.fail(
                "protocol", f"invalid_value:protocol.min_token_samples:{min_samples!r}"
            )
        rep.ok("protocol")
    min_samples = protocol.get("min_token_samples")
    if not is_int(min_samples) or min_samples < 1:
        min_samples = DEFAULT_MIN_TOKEN_SAMPLES

    bundle_cases = bundle.get("cases", MISSING)
    cases_by_id: dict[str, dict] = {}
    if not isinstance(bundle_cases, list):
        rep.fail("cases", "missing_field:cases")
    else:
        if not bundle_cases:
            rep.fail("cases", "missing_evidence:cases")
        for index, case in enumerate(bundle_cases):
            base = f"cases[{index}]"
            if not isinstance(case, dict):
                rep.fail("cases", f"invalid_type:{base}")
                continue
            for key in CASE_KEYS:
                if key not in case:
                    rep.fail("cases", f"missing_field:{base}.{key}")
            case_id = case.get("case_id")
            if isinstance(case_id, str):
                if case_id in cases_by_id:
                    rep.fail("cases", f"invalid_value:{base}.case_id:duplicate:{case_id}")
                cases_by_id[case_id] = case
        expected_by_id = {c["case_id"]: c for c in cases}
        for case_id, expected_case in expected_by_id.items():
            if case_id not in cases_by_id:
                rep.fail("cases", f"missing_evidence:case:{case_id}")
                continue
            got = cases_by_id[case_id]
            for key in CASE_KEYS:
                if key in got and got[key] != expected_case[key]:
                    rep.fail(
                        "cases",
                        f"case_mismatch:{case_id}.{key}:"
                        f"expected={expected_case[key]!r}:got={got[key]!r}",
                    )
        for case_id in cases_by_id:
            if case_id not in expected_by_id:
                rep.fail("cases", f"case_mismatch:{case_id}:unexpected_case")
        rep.ok("cases")

    runs = bundle.get("runs", MISSING)
    runs_by_case: dict[str, list[dict]] = {}
    seen_run_ids: set[str] = set()
    measured_run_ids: set[str] = set()
    measured_ids_by_case_arm: dict[tuple[str, str], list[str]] = {}
    case_of_run: dict[str, str] = {}
    if not isinstance(runs, list):
        rep.fail("runs_structure", "missing_field:runs")
    elif not runs:
        rep.fail("runs_structure", "missing_evidence:runs")
    else:
        known_case_ids = {c["case_id"] for c in cases}
        arms_by_case = {c["case_id"]: {c["baseline_arm"], CANDIDATE_ARM} for c in cases}
        for index, run in enumerate(runs):
            base = f"runs[{index}]"
            if not isinstance(run, dict):
                rep.fail("runs_structure", f"invalid_type:{base}")
                continue
            for key in RUN_KEYS_ALWAYS:
                if key not in run:
                    rep.fail("runs_structure", f"missing_field:{base}.{key}")
            run_id = run.get("run_id")
            if isinstance(run_id, str):
                if run_id in seen_run_ids:
                    rep.fail(
                        "runs_structure", f"invalid_value:{base}.run_id:duplicate:{run_id}"
                    )
                seen_run_ids.add(run_id)
            case_id = run.get("case_id")
            if not isinstance(case_id, str):
                rep.fail("runs_structure", f"invalid_type:{base}.case_id")
                continue
            if case_id not in known_case_ids:
                rep.fail("runs_structure", f"invalid_value:{base}.case_id:unknown:{case_id}")
                continue
            if run.get("arm") not in arms_by_case[case_id]:
                rep.fail("runs_structure", f"invalid_value:{base}.arm:{run.get('arm')!r}")
            warmup = run.get("warmup", MISSING)
            if warmup is MISSING:
                rep.fail("runs_structure", f"missing_field:{base}.warmup")
            elif not isinstance(warmup, bool):
                rep.fail("runs_structure", f"invalid_type:{base}.warmup")
            seq = run.get("sequence_index", MISSING)
            if seq is MISSING:
                rep.fail("runs_structure", f"missing_field:{base}.sequence_index")
            elif not is_int(seq) or seq < 0:
                rep.fail("runs_structure", f"invalid_value:{base}.sequence_index:{seq!r}")
            runs_by_case.setdefault(case_id, []).append(run)
            if isinstance(run_id, str):
                case_of_run[run_id] = case_id
                if warmup is False:
                    measured_run_ids.add(run_id)
                    measured_ids_by_case_arm.setdefault(
                        (case_id, run.get("arm")), []
                    ).append(run_id)
            _check_int(rep, "runs_structure", run, "oom_count", base, minimum=0)
            _validate_run_metrics(rep, run, base, min_samples)
            if warmup is False:
                _validate_telemetry(
                    rep, run, base, cases_by_id.get(case_id, {}).get("kind") == "above_p"
                )
                routes = run.get("routes", MISSING)
                if not isinstance(routes, dict):
                    rep.fail("routes", f"missing_field:{base}.routes")
                else:
                    executed = _check_bool(
                        rep, "routes", routes, "vulkan_attention_executed", f"{base}.routes"
                    )
                    expected_route = cases_by_id.get(case_id, {}).get("kind") == "above_p"
                    if executed is not None and executed != expected_route:
                        rep.fail(
                            "routes",
                            f"route_mismatch:{base}:vulkan_attention_executed="
                            f"{executed}:expected={expected_route}",
                        )
                if is_measured:
                    _validate_artifacts(
                        rep,
                        run,
                        base,
                        root,
                        context_hash_for_artifacts,
                        provenance_hash_for_artifacts,
                    )
        rep.ok("runs_structure")

    if is_measured and root is None:
        rep.fail("artifacts", "artifact_root_required")

    for case in cases:
        case_runs = runs_by_case.get(case["case_id"], [])
        if not case_runs:
            rep.fail("repetitions", f"missing_evidence:runs:{case['case_id']}")
            continue
        _validate_case_runs(rep, case, case_runs, protocol)
    rep.ok("repetitions")
    rep.ok("warmup")
    rep.ok("alternation")
    rep.ok("run_order")

    quality = bundle.get("quality", MISSING)
    if not isinstance(quality, list):
        rep.fail("quality", "missing_field:quality")
    elif not quality:
        rep.fail("quality", "missing_evidence:quality")
    else:
        seen_quality: set[tuple] = set()
        for index, entry in enumerate(quality):
            base = f"quality[{index}]"
            if not isinstance(entry, dict):
                rep.fail("quality", f"invalid_type:{base}")
                continue
            for key in QUALITY_KEYS:
                if key not in entry:
                    rep.fail("quality", f"missing_field:{base}.{key}")
            _check_number(rep, "quality", entry, "kl_mean_nat_per_token", base)
            _check_number(rep, "quality", entry, "perplexity", base)
            reference = entry.get("reference")
            if reference is not None and reference != QUALITY_REFERENCE:
                rep.fail(
                    "quality",
                    f"invalid_value:{base}.reference:{reference!r}:expected={QUALITY_REFERENCE!r}",
                )
            key = (entry.get("case_id"), entry.get("arm"))
            if key in seen_quality:
                rep.fail("quality", f"invalid_value:{base}:duplicate:{key!r}")
            seen_quality.add(key)
            if isinstance(key[0], str) and isinstance(key[1], str):
                _check_run_ids(
                    rep,
                    "quality",
                    entry,
                    base,
                    measured_run_ids,
                    measured_ids_by_case_arm,
                    exact_case_arm=key,
                )
        for case in cases:
            for arm in (case["baseline_arm"], CANDIDATE_ARM):
                if (case["case_id"], arm) not in seen_quality:
                    rep.fail(
                        "quality",
                        f"missing_evidence:quality:{case['case_id']}:{arm}",
                    )
        rep.ok("quality")

    capacity = bundle.get("capacity", MISSING)
    if not isinstance(capacity, dict):
        rep.fail("capacity", "missing_field:capacity")
    else:
        for key in CAPACITY_KEYS:
            if key not in capacity:
                rep.fail("capacity", f"missing_field:capacity.{key}")
        _check_int(rep, "capacity", capacity, "occupied_positions", "capacity")
        _check_int(rep, "capacity", capacity, "over_capacity_tokens", "capacity")
        _check_int(rep, "capacity", capacity, "gen_start_tokens", "capacity")
        _check_int(rep, "capacity", capacity, "gen_count", "capacity")
        _check_int(rep, "capacity", capacity, "gen_end_tokens", "capacity")
        _check_bool(rep, "capacity", capacity, "over_capacity_rejected", "capacity")
        _check_bool(rep, "capacity", capacity, "over_capacity_state_intact", "capacity")
        _check_bool(rep, "capacity", capacity, "gen_ok", "capacity")
        if capacity.get("occupied_positions") not in (MISSING, None, expected_c):
            rep.fail(
                "capacity",
                f"case_mismatch:capacity.occupied_positions:expected={expected_c}:"
                f"got={capacity.get('occupied_positions')}",
            )
        if capacity.get("over_capacity_tokens") not in (MISSING, None, expected_c + 1):
            rep.fail(
                "capacity",
                f"case_mismatch:capacity.over_capacity_tokens:expected={expected_c + 1}",
            )
        if capacity.get("gen_end_tokens") not in (MISSING, None, expected_c):
            rep.fail(
                "capacity", f"case_mismatch:capacity.gen_end_tokens:expected={expected_c}"
            )
        if capacity.get("gen_start_tokens") not in (MISSING, None, expected_c - 512):
            rep.fail(
                "capacity",
                f"case_mismatch:capacity.gen_start_tokens:expected={expected_c - 512}",
            )
        if capacity.get("gen_count") not in (MISSING, None, 512):
            rep.fail("capacity", f"case_mismatch:capacity.gen_count:expected=512")
        capacity_case_id = next(
            (
                c["case_id"]
                for c in cases
                if c["length_tokens"] == expected_c - 512
            ),
            None,
        )
        _check_run_ids(
            rep,
            "capacity",
            capacity,
            "capacity",
            measured_run_ids,
            measured_ids_by_case_arm,
            required_case_id=capacity_case_id,
            case_of_run=case_of_run,
        )
        rep.ok("capacity")

    restore = bundle.get("restore", MISSING)
    restore_by_length: dict[int, dict] = {}
    if not isinstance(restore, list):
        rep.fail("restore", "missing_field:restore")
    elif not restore:
        rep.fail("restore", "missing_evidence:restore")
    else:
        for index, entry in enumerate(restore):
            base = f"restore[{index}]"
            if not isinstance(entry, dict):
                rep.fail("restore", f"invalid_type:{base}")
                continue
            for key in RESTORE_KEYS:
                if key not in entry:
                    rep.fail("restore", f"missing_field:{base}.{key}")
            length = _check_int(rep, "restore", entry, "length_tokens", base, minimum=1)
            reused = _check_int(rep, "restore", entry, "tokens_reused", base, minimum=0)
            reprefill = _check_int(
                rep, "restore", entry, "prefix_reprefill_tokens", base, minimum=0
            )
            diff = _check_number(
                rep, "restore", entry, "next_logits_max_abs_diff", base, positive=False
            )
            if diff is not None and diff < 0.0:
                rep.fail(
                    "restore",
                    f"invalid_value:{base}.next_logits_max_abs_diff:{diff}",
                )
            _check_bool(rep, "restore", entry, "ok", base)
            scenario = entry.get("scenario", MISSING)
            if scenario is MISSING:
                pass
            elif scenario not in RESTORE_SCENARIOS:
                rep.fail(
                    "restore",
                    f"invalid_value:{base}.scenario:{scenario!r}:"
                    f"allowed={list(RESTORE_SCENARIOS)!r}",
                )
            restore_case_id = next(
                (
                    c["case_id"]
                    for c in cases
                    if is_int(length) and c["length_tokens"] == length
                ),
                None,
            )
            _check_run_ids(
                rep,
                "restore",
                entry,
                base,
                measured_run_ids,
                measured_ids_by_case_arm,
                required_case_id=restore_case_id,
                case_of_run=case_of_run,
            )
            if length is not None:
                if length in restore_by_length:
                    rep.fail("restore", f"invalid_value:{base}:duplicate_length:{length}")
                restore_by_length[length] = entry
                if reused is not None and reprefill is not None:
                    if reused + reprefill != length:
                        rep.fail(
                            "restore",
                            f"invalid_value:{base}:tokens_reused+prefix_reprefill_tokens"
                            f"={reused + reprefill}:expected={length}",
                        )
        for length in required_restore_lengths(cases):
            if length not in restore_by_length:
                rep.fail("restore", f"missing_evidence:restore:{length}")
        rep.ok("restore")

    cycles = bundle.get("cycles", MISSING)
    if not isinstance(cycles, list):
        rep.fail("cycles", "missing_field:cycles")
    elif len(cycles) != RESTORE_CYCLES:
        rep.fail(
            "cycles",
            f"missing_evidence:cycles:got={len(cycles) if isinstance(cycles, list) else 'n/a'}:"
            f"need={RESTORE_CYCLES}",
        )
    else:
        for index, entry in enumerate(cycles):
            base = f"cycles[{index}]"
            if not isinstance(entry, dict):
                rep.fail("cycles", f"invalid_type:{base}")
                continue
            for key in CYCLE_KEYS:
                if key not in entry:
                    rep.fail("cycles", f"missing_field:{base}.{key}")
            idx = _check_int(rep, "cycles", entry, "index", base, minimum=1)
            if idx is not None and idx != index + 1:
                rep.fail("cycles", f"invalid_value:{base}.index:{idx}:expected={index + 1}")
            for key in ("restore_ok", "divergent_suffix_ok", "generation_ok"):
                _check_bool(rep, "cycles", entry, key, base)
            for key in ("oom_count", "driver_error_count", "corruption_count"):
                _check_int(rep, "cycles", entry, key, base, minimum=0)
            retained = entry.get("retained_bytes")
            if not isinstance(retained, dict):
                rep.fail("cycles", f"missing_field:{base}.retained_bytes")
            else:
                for backend in ("cuda", "vulkan"):
                    _check_int(
                        rep, "cycles", retained, backend, f"{base}.retained_bytes", minimum=0
                    )
            _check_run_ids(
                rep,
                "cycles",
                entry,
                base,
                measured_run_ids,
                measured_ids_by_case_arm,
            )
        rep.ok("cycles")

    for check_id in (
        "run_metrics",
        "token_samples",
        "routes",
        "telemetry",
        "artifacts",
        "evidence_object",
    ):
        rep.ok(check_id)

    return rep.as_dict()


def _gate(
    gate_id: str,
    status: str,
    value: Any = None,
    limit: Any = None,
    relation: str = "le",
    unit: str = "",
    group: str = "",
    case_id: str | None = None,
    reason: str | None = None,
) -> dict:
    return {
        "id": gate_id,
        "group": group,
        "case_id": case_id,
        "status": status,
        "value": value,
        "limit": limit,
        "relation": relation,
        "unit": unit,
        "reason": reason,
    }


def _ratio_gate(
    gate_id: str,
    group: str,
    case_id: str,
    candidate: float | None,
    baseline: float | None,
    limit: float,
    unit: str = "ratio",
) -> dict:
    if candidate is None or baseline is None:
        return _gate(
            gate_id,
            "incomplete",
            limit=limit,
            unit=unit,
            group=group,
            case_id=case_id,
            reason="missing_evidence",
        )
    if baseline <= 0.0:
        return _gate(
            gate_id,
            "incomplete",
            limit=limit,
            unit=unit,
            group=group,
            case_id=case_id,
            reason="invalid_reference",
        )
    value = candidate / baseline
    status = "pass" if value <= limit else "fail"
    return _gate(
        gate_id,
        status,
        value=value,
        limit=limit,
        unit=unit,
        group=group,
        case_id=case_id,
    )


def _arm_aggregate(runs: list[dict]) -> dict | None:
    if not runs:
        return None
    prefill = [r["prefill_ms"] for r in runs]
    ttft = [r["ttft_ms"] for r in runs]
    medians = [statistics.median(r["token_latencies_ms"]) for r in runs]
    p95s = [percentile_nearest_rank(r["token_latencies_ms"]) for r in runs]
    return {
        "runs": len(runs),
        "prefill_ms": statistics.median(prefill),
        "ttft_ms": statistics.median(ttft),
        "decode_median_ms": statistics.median(medians),
        "decode_p95_ms": statistics.median(p95s),
    }


GENUINE_ACCEPTANCE_REASON = (
    "genuine F5 acceptance additionally requires the campaign owner to run the "
    "GPU/server measurements on the recorded hardware with F4/F7 gates green "
    "and to confirm provenance.command; this CPU-only tool attests schema "
    "admissibility and gate arithmetic, never campaign results"
)

FIXTURE_MARKERS = ("synthetic", "fixture", "replace_before_measurement")


def evidence_scope(bundle: Any, kind: str | None) -> str:
    if kind != "measured":
        return "synthetic_fixture"
    provenance = bundle.get("provenance") if isinstance(bundle, dict) else None
    if not isinstance(provenance, dict):
        return "unknown_provenance"
    origin = provenance.get("evidence_origin")
    if origin != "campaign_measured":
        return str(origin) if isinstance(origin, str) else "unknown_provenance"
    for key in ("collector", "command", "driver", "build"):
        value = provenance.get(key)
        if isinstance(value, str) and any(marker in value.lower() for marker in FIXTURE_MARKERS):
            return "fixture_or_placeholder_provenance"
    hardware = provenance.get("hardware")
    if isinstance(hardware, dict):
        gpu = hardware.get("gpu")
        if isinstance(gpu, str) and any(marker in gpu.lower() for marker in FIXTURE_MARKERS):
            return "fixture_or_placeholder_provenance"
    for key, value in provenance.items():
        if is_placeholder(value) or (isinstance(value, str) and is_zero_hash(value)):
            return "placeholder_provenance"
    return "campaign_measured"


def acceptance_block(schema_admissible: bool, gate_arithmetic_only: bool) -> dict:
    return {
        "schema_admissibility": schema_admissible,
        "gate_arithmetic_only": gate_arithmetic_only,
        "genuine_f5_acceptance": False,
        "reason": GENUINE_ACCEPTANCE_REASON,
    }


def summarize(
    bundle: Any,
    context: dict,
    artifact_root: str | Path | None = None,
    require_gates_pass: bool = False,
) -> tuple[dict, int]:
    report = validate(bundle, context, artifact_root)
    if report["verdict"] != "accept":
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "admissible": False,
            "evidence_kind": report["evidence_kind"],
            "evidence_scope": evidence_scope(bundle, report["evidence_kind"]),
            "acceptance": acceptance_block(False, False),
            "reasons": list(report["reasons"]),
            "kind_reason": report["kind_reason"],
            "measured": None,
            "gates": [],
            "gates_pass": 0,
            "gates_fail": 0,
            "gates_incomplete": 0,
            "notes": [
                "evidence not admissible; no measurement was computed and no number "
                "is reported"
            ],
        }
        return summary, 1

    expected = context.get("expected", {})
    expected_p = int(expected.get("p", PLAN_DEFAULT_P))
    expected_c = int(expected.get("c", CAPACITY_C))
    cases = prescribed_cases(expected_p)
    gates: list[dict] = []

    runs = bundle["runs"]
    measured_runs = [r for r in runs if r.get("warmup") is False]
    by_case_arm: dict[tuple[str, str], list[dict]] = {}
    for run in measured_runs:
        by_case_arm.setdefault((run["case_id"], run["arm"]), []).append(run)

    measured: dict[str, Any] = {"cases": {}}
    for case in cases:
        case_id = case["case_id"]
        baseline_arm = case["baseline_arm"]
        candidate_runs = by_case_arm.get((case_id, CANDIDATE_ARM), [])
        baseline_runs = by_case_arm.get((case_id, baseline_arm), [])
        candidate = _arm_aggregate(candidate_runs)
        baseline = _arm_aggregate(baseline_runs)
        measured["cases"][case_id] = {
            "kind": case["kind"],
            "length_tokens": case["length_tokens"],
            "baseline_arm": baseline_arm,
            "arms": {CANDIDATE_ARM: candidate, baseline_arm: baseline},
        }

        if case["kind"] == "below_p":
            prefill_limit = LATENCY_GATE_LIMIT_REGRESSION
            ttft_limit = LATENCY_GATE_LIMIT_REGRESSION
            gate_prefix = "regression"
        else:
            prefill_limit = PREFILL_GATE_LIMIT_IMPROVEMENT
            ttft_limit = PREFILL_GATE_LIMIT_IMPROVEMENT
            gate_prefix = "improvement"

        cand_prefill = candidate["prefill_ms"] if candidate else None
        base_prefill = baseline["prefill_ms"] if baseline else None
        cand_ttft = candidate["ttft_ms"] if candidate else None
        base_ttft = baseline["ttft_ms"] if baseline else None
        cand_p95 = candidate["decode_p95_ms"] if candidate else None
        base_p95 = baseline["decode_p95_ms"] if baseline else None

        gates.append(
            _ratio_gate(
                f"{case_id}.prefill_{gate_prefix}",
                "latency",
                case_id,
                cand_prefill,
                base_prefill,
                prefill_limit,
            )
        )
        gates.append(
            _ratio_gate(
                f"{case_id}.ttft_{gate_prefix}",
                "latency",
                case_id,
                cand_ttft,
                base_ttft,
                ttft_limit,
            )
        )
        gates.append(
            _ratio_gate(
                f"{case_id}.decode_p95_regression",
                "latency",
                case_id,
                cand_p95,
                base_p95,
                LATENCY_GATE_LIMIT_REGRESSION,
            )
        )

        quality_entries = {
            (e.get("case_id"), e.get("arm")): e for e in bundle.get("quality", [])
        }
        cand_q = quality_entries.get((case_id, CANDIDATE_ARM))
        base_q = quality_entries.get((case_id, baseline_arm))
        if cand_q is None or base_q is None:
            gates.append(
                _gate(
                    f"{case_id}.quality_kl_mean",
                    "incomplete",
                    limit=QUALITY_KL_DELTA_LIMIT,
                    unit="nat/token",
                    group="quality",
                    case_id=case_id,
                    reason="missing_evidence",
                )
            )
            gates.append(
                _gate(
                    f"{case_id}.quality_perplexity",
                    "incomplete",
                    limit=QUALITY_PPL_RATIO_LIMIT,
                    unit="ratio",
                    group="quality",
                    case_id=case_id,
                    reason="missing_evidence",
                )
            )
        else:
            kl_delta = (
                cand_q["kl_mean_nat_per_token"] - base_q["kl_mean_nat_per_token"]
            )
            gates.append(
                _gate(
                    f"{case_id}.quality_kl_mean",
                    "pass" if kl_delta <= QUALITY_KL_DELTA_LIMIT else "fail",
                    value=kl_delta,
                    limit=QUALITY_KL_DELTA_LIMIT,
                    unit="nat/token",
                    group="quality",
                    case_id=case_id,
                )
            )
            ppl_ratio = cand_q["perplexity"] / base_q["perplexity"]
            gates.append(
                _gate(
                    f"{case_id}.quality_perplexity",
                    "pass" if ppl_ratio <= QUALITY_PPL_RATIO_LIMIT else "fail",
                    value=ppl_ratio,
                    limit=QUALITY_PPL_RATIO_LIMIT,
                    unit="ratio",
                    group="quality",
                    case_id=case_id,
                )
            )

        expected_route = case["kind"] == "above_p"
        case_runs = candidate_runs + baseline_runs
        route_ok = bool(case_runs) and all(
            r.get("routes", {}).get("vulkan_attention_executed") == expected_route
            for r in case_runs
        )
        gates.append(
            _gate(
                f"{case_id}.route_vulkan_attention",
                "pass" if route_ok else "fail",
                value=all(
                    r.get("routes", {}).get("vulkan_attention_executed")
                    for r in case_runs
                )
                if case_runs
                else None,
                limit=expected_route,
                relation="eq",
                unit="bool",
                group="routes",
                case_id=case_id,
                reason=None if case_runs else "missing_evidence",
            )
        )

    provenance = bundle["provenance"]
    vram_total = int(provenance["hardware"]["vram_total_bytes"])
    guard_limit = min_cuda_guard_bytes(vram_total)
    guards: list[int] = []
    host_avail: list[int] = []
    swap_in: list[int] = []
    swap_out: list[int] = []
    gtt_avail: list[int] = []
    gtt_peak: list[int] = []
    for run in measured_runs:
        cuda_stages = run["telemetry"]["cuda"]["stages"]
        peak = max(rec["peak_bytes"] for rec in cuda_stages.values())
        guards.append(vram_total - peak)
        host = run["telemetry"]["host"]
        host_avail.append(host["mem_available_min_bytes"])
        swap_in.append(host["swap_in_bytes"])
        swap_out.append(host["swap_out_bytes"])
        vulkan = run["telemetry"]["vulkan"]
        gtt_avail.append(vulkan["gtt_available_min_bytes"])
        gtt_peak.append(vulkan["gtt_peak_bytes"])

    # Revision 2026-10-10: memory headroom and swap are observations. Only
    # an OOM during any warmup, measured run or stability cycle fails memory.
    min_guard = min(guards) if guards else None
    min_host = min(host_avail) if host_avail else None
    max_swap = max(swap_in) if swap_in else None
    max_swap_out = max(swap_out) if swap_out else None
    oom_total = sum(run["oom_count"] for run in bundle["runs"]) + sum(
        cycle["oom_count"] for cycle in bundle["cycles"]
    )
    gates.append(_gate(
        "memory.no_oom", "pass" if oom_total == 0 else "fail",
        value=oom_total, limit=0, relation="eq", unit="count", group="memory",
    ))

    capacity = bundle["capacity"]
    gates.append(
        _gate(
            "capacity.occupancy_full",
            "pass" if capacity["occupied_positions"] == expected_c else "fail",
            value=capacity["occupied_positions"],
            limit=expected_c,
            relation="eq",
            unit="tokens",
            group="capacity",
        )
    )
    reject_ok = (
        capacity["over_capacity_rejected"] is True
        and capacity["over_capacity_state_intact"] is True
    )
    gates.append(
        _gate(
            "capacity.reject_over_capacity",
            "pass" if reject_ok else "fail",
            value=reject_ok,
            limit=True,
            relation="eq",
            unit="bool",
            group="capacity",
        )
    )
    gen_ok = capacity["gen_ok"] is True and capacity["gen_end_tokens"] == expected_c
    gates.append(
        _gate(
            "capacity.generate_to_capacity",
            "pass" if gen_ok else "fail",
            value=gen_ok,
            limit=expected_c,
            relation="eq",
            unit="tokens",
            group="capacity",
        )
    )

    restore_records = {e["length_tokens"]: e for e in bundle["restore"]}
    restore_measured: dict[str, Any] = {}
    for length in required_restore_lengths(cases):
        entry = restore_records.get(length)
        if entry is None:
            gates.append(
                _gate(
                    f"restore.{length}",
                    "incomplete",
                    limit=0,
                    unit="tokens",
                    group="restore",
                    reason="missing_evidence",
                )
            )
            continue
        restore_measured[str(length)] = {
            "scenario": entry.get("scenario"),
            "tokens_reused": entry["tokens_reused"],
            "prefix_reprefill_tokens": entry["prefix_reprefill_tokens"],
            "next_logits_max_abs_diff": entry["next_logits_max_abs_diff"],
            "ok": entry["ok"],
        }
        scenario = entry.get("scenario")
        status = "pass"
        reason = None
        value: Any = entry["next_logits_max_abs_diff"]
        limit: Any = 0.0
        relation = "le"
        unit = "abs_logit_delta"
        if scenario != "exact_import":
            status = "fail"
            reason = f"scenario:{scenario}:exact_import_only"
        elif entry["tokens_reused"] != length:
            status = "fail"
            reason = (
                f"partial_prefix_reuse:reused={entry['tokens_reused']}:need={length}"
            )
            value = entry["tokens_reused"]
            limit = length
            relation = "eq"
            unit = "tokens"
        elif entry["prefix_reprefill_tokens"] != 0:
            status = "fail"
            reason = (
                f"reprefill_required:reprefill={entry['prefix_reprefill_tokens']}:need=0"
            )
            value = entry["prefix_reprefill_tokens"]
            limit = 0
            relation = "eq"
            unit = "tokens"
        elif entry["ok"] is not True or entry["next_logits_max_abs_diff"] > 0.0:
            status = "fail"
            reason = "restore_not_clean"
        gates.append(
            _gate(
                f"restore.{length}",
                status,
                value=value,
                limit=limit,
                relation=relation,
                unit=unit,
                group="restore",
                reason=reason,
            )
        )

    cycles = bundle["cycles"]
    cycles_ok = all(
        c["restore_ok"] is True
        and c["divergent_suffix_ok"] is True
        and c["generation_ok"] is True
        and c["oom_count"] == 0
        and c["driver_error_count"] == 0
        and c["corruption_count"] == 0
        for c in cycles
    )
    gates.append(
        _gate(
            "cycles.all_cycles_clean",
            "pass" if cycles_ok else "fail",
            value=cycles_ok,
            limit=True,
            relation="eq",
            unit="bool",
            group="cycles",
        )
    )
    first, last = cycles[0], cycles[-1]
    growth = {
        backend: last["retained_bytes"][backend] - first["retained_bytes"][backend]
        for backend in ("cuda", "vulkan")
    }
    worst_growth = max(growth.values())
    gates.append(
        _gate(
            "cycles.retained_growth",
            "pass" if worst_growth <= RETAINED_GROWTH_LIMIT_BYTES else "fail",
            value=worst_growth,
            limit=RETAINED_GROWTH_LIMIT_BYTES,
            unit="bytes",
            group="cycles",
        )
    )

    measured["memory"] = {
        "policy": "no_oom_2026-10-10",
        "oom_total": oom_total,
        "vram_total_bytes": vram_total,
        "cuda_guard_margin_min_bytes": min_guard,
        "cuda_guard_limit_bytes": guard_limit,
        "host_mem_available_min_bytes": min_host,
        "swap_in_bytes_max": max_swap,
        "swap_out_bytes_max": max_swap_out,
        "vulkan_gtt_available_min_bytes": min(gtt_avail) if gtt_avail else None,
        "vulkan_gtt_peak_bytes_max": max(gtt_peak) if gtt_peak else None,
    }
    measured["restore"] = restore_measured
    measured["cycles"] = {
        "count": len(cycles),
        "retained_growth_bytes": growth,
        "limit_bytes": RETAINED_GROWTH_LIMIT_BYTES,
        "oom_total": sum(c["oom_count"] for c in cycles),
        "driver_errors_total": sum(c["driver_error_count"] for c in cycles),
        "corruption_total": sum(c["corruption_count"] for c in cycles),
    }

    counts = {"pass": 0, "fail": 0, "incomplete": 0}
    for gate in gates:
        counts[gate["status"]] = counts.get(gate["status"], 0) + 1

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "admissible": True,
        "evidence_kind": report["evidence_kind"],
        "evidence_scope": evidence_scope(bundle, report["evidence_kind"]),
        "acceptance": acceptance_block(True, True),
        "reasons": [],
        "kind_reason": None,
        "measured": measured,
        "gates": gates,
        "gates_pass": counts["pass"],
        "gates_fail": counts["fail"],
        "gates_incomplete": counts["incomplete"],
        "notes": [
            "values are computed only from admissible evidence; no value is "
            "invented, extrapolated or defaulted",
            "gate results here are schema-admissibility and gate-arithmetic "
            "coverage only; they are not F5 acceptance (see acceptance)",
        ],
    }
    exit_code = 0
    if require_gates_pass and counts["fail"] > 0:
        exit_code = 1
    if counts["incomplete"] > 0:
        exit_code = 1
    return summary, exit_code


def build_schedule(context: dict) -> dict:
    protocol = context.get("protocol", {})
    expected = context.get("expected", {})
    cases = prescribed_cases(int(expected.get("p", PLAN_DEFAULT_P)))
    warmup_runs = int(protocol.get("warmup_runs", MIN_WARMUP_RUNS))
    per_arm = int(protocol.get("measured_runs_per_arm", MIN_MEASURED_RUNS_PER_ARM))
    first_kind = protocol.get("first_arm", FIRST_ARM_BASELINE)
    entries: list[dict] = []
    for case in cases:
        case_id = case["case_id"]
        if first_kind == FIRST_ARM_CANDIDATE:
            first_arm, second_arm = CANDIDATE_ARM, case["baseline_arm"]
        else:
            first_arm, second_arm = case["baseline_arm"], CANDIDATE_ARM
        case_sequence = 0
        for warmup_arm in (first_arm, second_arm):
            for warmup_index in range(warmup_runs):
                entries.append(
                    {
                        "run_id": f"{case_id}-warm-{warmup_arm}-{warmup_index}",
                        "case_id": case_id,
                        "arm": warmup_arm,
                        "warmup": True,
                        "sequence_index": case_sequence,
                        "schedule_step": len(entries),
                        "discarded": True,
                    }
                )
                case_sequence += 1
        for pair in range(per_arm):
            for arm in (first_arm, second_arm):
                entries.append(
                    {
                        "run_id": f"{case_id}-m{pair * 2 + (arm == second_arm)}",
                        "case_id": case_id,
                        "arm": arm,
                        "warmup": False,
                        "sequence_index": case_sequence,
                        "schedule_step": len(entries),
                        "discarded": False,
                    }
                )
                case_sequence += 1
    return {
        "schema_version": SCHEDULE_SCHEMA,
        "protocol": {
            "warmup_runs_per_case": warmup_runs,
            "measured_runs_per_arm": per_arm,
            "alternating": True,
            "first_arm": first_kind,
            "min_token_samples": int(protocol.get("min_token_samples", DEFAULT_MIN_TOKEN_SAMPLES)),
        },
        "cases": cases,
        "sequence": entries,
        "total_runs": len(entries),
        "notes": [
            "warmup runs are discarded and never counted",
            "measured runs alternate strictly between the case baseline arm and "
            f"{CANDIDATE_ARM}",
            "this schedule is emitted as documentation of the protocol; the harness "
            "does not execute runs",
        ],
    }


VERIFIED_FLAGS = (
    {"flag": "-c / --ctx-size", "source": "common/arg.cpp:2267"},
    {"flag": "-b / --batch-size", "source": "common/arg.cpp:2784"},
    {"flag": "-ub / --ubatch-size", "source": "common/arg.cpp:2791"},
    {"flag": "-fa / --flash-attn", "source": "common/arg.cpp:3034"},
    {"flag": "-ctk / --cache-type-k", "source": "common/arg.cpp:3725"},
    {"flag": "-ctv / --cache-type-v", "source": "common/arg.cpp:3738"},
    {"flag": "--spec-type (value: none)", "source": "common/arg.cpp:6076, common/speculative.cpp:5074"},
    {"flag": "--spec-draft-n-max", "source": "common/arg.cpp:5900"},
    {"flag": "--port", "source": "common/arg.cpp:4800"},
    {
        "flag": "env LLAMA_KV_POSITION_SPLIT_P (multiple of 256, 0 < P < n_ctx)",
        "source": "src/llama-context.cpp:450-455",
    },
)


def build_templates(context: dict) -> dict:
    expected = context.get("expected", {})
    p = int(expected.get("p", PLAN_DEFAULT_P))
    c = int(expected.get("c", CAPACITY_C))
    batch = int(expected.get("batch_size", 512))
    ubatch = int(expected.get("ubatch_size", 256))
    commands = {
        "candidate_server": (
            f"LLAMA_KV_POSITION_SPLIT_P={p} build/bin/llama-server "
            "-m {MODEL_PATH} "
            f"-c {c} -b {batch} -ub {ubatch} "
            "-fa on --spec-type none --spec-draft-n-max 0 --port {PORT} "
            "{KV_AND_OVERFLOW_FLAGS_FROM_F3_F4_PROFILE} {SAMPLING_FLAGS}"
        ),
        "baseline_xxl_server": (
            "-m {MODEL_PATH} -c {c} -b {batch} -ub {ubatch} "
            "-fa on --spec-type none --spec-draft-n-max 0 --port {PORT} "
            "{XXL_CUDA_ONLY_FLAGS_RECORDED_BY_CAMPAIGN_OWNER}"
        ),
        "baseline_xxxl_server": (
            "-m {MODEL_PATH} -c {c} -b {batch} -ub {ubatch} "
            "-fa on --spec-type none --spec-draft-n-max 0 --port {PORT} "
            "{XXXL_PER_LAYER_MIXED_FLAGS_RECORDED_BY_CAMPAIGN_OWNER}"
        ),
        "schedule": "python3 tools/occupancy-f5/f5_harness.py plan --context {CONTEXT}",
        "validate": (
            "python3 tools/occupancy-f5/f5_harness.py validate {EVIDENCE} "
            "--context {CONTEXT} --artifact-root {ARTIFACT_ROOT}"
        ),
        "summarize": (
            "python3 tools/occupancy-f5/f5_harness.py summarize {EVIDENCE} "
            "--context {CONTEXT} --artifact-root {ARTIFACT_ROOT} "
            "--require-gates-pass"
        ),
    }
    return {
        "schema_version": TEMPLATE_SCHEMA,
        "status": "NOT RUN",
        "warning": (
            "Documentation templates only. This harness never launches a server, "
            "never compiles, never touches a GPU and never performs network access. "
            "Nothing in this block has been executed by this tool."
        ),
        "placeholders": {
            "{MODEL_PATH}": "target/baseline GGUF path; record sha256 in provenance",
            "{PORT}": "free local port chosen by the campaign owner",
            "{KV_AND_OVERFLOW_FLAGS_FROM_F3_F4_PROFILE}": (
                "local KVarN4 + overflow Q4 flags: copy verbatim from the F3/F4 "
                "profile command recorded by the main worker; do not guess"
            ),
            "{XXL_CUDA_ONLY_FLAGS_RECORDED_BY_CAMPAIGN_OWNER}": (
                "exact baseline command recorded before the campaign"
            ),
            "{XXXL_PER_LAYER_MIXED_FLAGS_RECORDED_BY_CAMPAIGN_OWNER}": (
                "exact baseline command recorded before the campaign"
            ),
            "{SAMPLING_FLAGS}": "seed/temperature/top-p fixed identically for both arms",
        },
        "verified_flags": list(VERIFIED_FLAGS),
        "commands": commands,
        "notes": [
            "plan section 3.7 keeps MTP out of F5: the profile runs with "
            "spec-draft-n-max = 0 as XXL and XXXL do today, so these templates "
            "use --spec-type none --spec-draft-n-max 0",
            "every command used for measured evidence must be recorded verbatim in "
            "provenance.command before the run, never reconstructed afterwards",
            "fill every {PLACEHOLDER} from a recorded command; a template with an "
            "unfilled placeholder is not runnable evidence",
        ],
    }


def build_schema_doc() -> dict:
    return {
        "schema_version": SCHEMA_DOC_SCHEMA,
        "evidence_schema": EVIDENCE_SCHEMA,
        "context_schema": CONTEXT_SCHEMA,
        "required_top_level": [
            "schema_version",
            "evidence_kind",
            "provenance",
            "protocol",
            "cases",
            "runs",
            "quality",
            "capacity",
            "restore",
            "cycles",
            "bindings",
        ],
        "context_requirements": {
            "required_expected_keys": list(REQUIRED_EXPECTED_KEYS),
            "required_protocol_keys": list(REQUIRED_PROTOCOL_KEYS),
            "p": "positive multiple of 256, p < c and p <= c - 512",
            "c": f">= {PLAN_ABOVE_P_LENGTHS[-1]} (prescribed cases must fit)",
            "sha256_fields": "64 hex chars, all-zero placeholder hashes rejected",
            "placeholders": "placeholder path/command strings rejected (bad_context)",
        },
        "evidence_kind": {
            "allowed": list(EVIDENCE_KINDS),
            "rule": "only 'measured' can ever be accepted; 'synthetic_fixture' is "
            "always rejected for F5 acceptance",
        },
        "evidence_origin": {
            "allowed": list(EVIDENCE_ORIGINS),
            "rule": "evidence_kind=measured requires provenance.evidence_origin="
            "'campaign_measured'; a flipped synthetic fixture is rejected",
        },
        "bindings": {
            "keys": ["context_sha256"] + [f"{b}_sha256" for b in BINDING_BLOCKS],
            "rule": "sha256 of canonical JSON (sort_keys, compact separators) of the "
            "campaign context (expected+protocol) and of each evidence block; "
            "recomputed at validate time, mismatch rejects",
            "purpose": "binds quality/capacity/restore/cycles and telemetry runs to "
            "one specific context and provenance so relabelled data cannot pass",
        },
        "artifact_header": {
            "first_line": f"{LOG_HEADER_PREFIX} run_id=<run_id> "
            "context_sha256=<64hex> provenance_sha256=<64hex>",
            "rule": "every artifact must embed this header; content must hash to "
            "artifacts[].sha256 and the header must match run, context and "
            "provenance",
        },
        "run_ids": {
            "rule": "quality entries must name exactly the measured runs of their "
            "case/arm; capacity, restore and cycle entries must name measured runs "
            "that exist (warmup ids do not count)",
        },
        "provenance_required": list(PROVENANCE_REQUIRED_STRINGS)
        + ["evidence_origin", "dirty", "p", "c", "seed", "batch_size", "ubatch_size",
           "cache_types", "sampling", "hardware", "clocks", "cache_state"],
        "protocol": {
            "warmup_runs": f">= {MIN_WARMUP_RUNS} discarded warmup run per arm per case",
            "measured_runs_per_arm": f">= {MIN_MEASURED_RUNS_PER_ARM} per arm per case",
            "alternating": True,
            "first_arm": [FIRST_ARM_BASELINE, FIRST_ARM_CANDIDATE],
            "min_token_samples": DEFAULT_MIN_TOKEN_SAMPLES,
        },
        "prescribed_cases": {
            "below_p": [PLAN_BELOW_P_LENGTH],
            "above_p": list(PLAN_ABOVE_P_LENGTHS),
            "conditional": f"plus P+256 when P != {PLAN_DEFAULT_P}",
            "capacity": [CAPACITY_C, CAPACITY_C + 1, CAPACITY_C - 512],
        },
        "telemetry": {
            "backends": list(TELEMETRY_BACKENDS),
            "stages": list(TELEMETRY_STAGES) + [BOUNDARY_STAGE + " (above_p only)"],
            "per_backend_required": {
                "cuda": "stages[*].peak_bytes",
                "vulkan": "stages[*].peak_bytes, stages[*].attention_copy_bytes "
                "(must be 0 below P), gtt_peak_bytes, gtt_available_min_bytes",
                "host": "stages[*].peak_bytes, mem_available_min_bytes, "
                "swap_in_bytes, swap_out_bytes",
            },
        },
        "restore": {
            "scenarios": list(RESTORE_SCENARIOS),
            "exact_import_gate": "scenario=exact_import, tokens_reused == length, "
            "prefix_reprefill_tokens == 0, ok=true, next_logits_max_abs_diff == 0",
            "cache_miss_fallback": "never satisfies the exact import gate",
        },
        "gates": {
            "below_p_prefill_ttft_decode_p95_ratio": LATENCY_GATE_LIMIT_REGRESSION,
            "above_p_prefill_ttft_ratio": PREFILL_GATE_LIMIT_IMPROVEMENT,
            "above_p_decode_p95_ratio": LATENCY_GATE_LIMIT_REGRESSION,
            "quality_kl_mean_delta_nat_per_token": QUALITY_KL_DELTA_LIMIT,
            "quality_perplexity_ratio": QUALITY_PPL_RATIO_LIMIT,
            "memory_oom_count": 0,
            "memory_observations": ["cuda_guard_margin", "host_mem_available", "swap_in", "swap_out"],
            "route_below_p": "vulkan_attention_executed=false and "
            "stages[*].attention_copy_bytes=0",
            "route_above_p": "vulkan_attention_executed=true",
            "restore_cycles": RESTORE_CYCLES,
            "retained_growth_bytes_per_backend": RETAINED_GROWTH_LIMIT_BYTES,
            "restore_next_logits_max_abs_diff": 0.0,
        },
        "exit_codes": {"accept_or_admissible": 0, "reject_or_inadmissible": 1, "usage": 2},
    }


PLACEHOLDER = "REPLACE_BEFORE_MEASUREMENT"


def build_skeleton(context: dict) -> dict:
    expected = context.get("expected", {})
    p = int(expected.get("p", PLAN_DEFAULT_P))
    c = int(expected.get("c", CAPACITY_C))
    provenance = {
        "commit": expected.get("commit", PLACEHOLDER),
        "branch": expected.get("branch", PLACEHOLDER),
        "dirty": False,
        "model_path": expected.get("model_path", PLACEHOLDER),
        "model_sha256": expected.get("model_sha256", "0" * 64),
        "corpus_path": expected.get("corpus_path", PLACEHOLDER),
        "corpus_sha256": expected.get("corpus_sha256", "0" * 64),
        "prompt_sha256": "0" * 64,
        "binary_path": PLACEHOLDER,
        "binary_sha256": "0" * 64,
        "p": p,
        "c": c,
        "cache_types": {"local": PLACEHOLDER, "overflow": PLACEHOLDER},
        "seed": expected.get("seed", 0),
        "sampling": {},
        "batch_size": expected.get("batch_size", 0),
        "ubatch_size": expected.get("ubatch_size", 0),
        "hardware": {"gpu": PLACEHOLDER, "vram_total_bytes": 1},
        "driver": PLACEHOLDER,
        "build": PLACEHOLDER,
        "clocks": {},
        "governor": PLACEHOLDER,
        "cache_state": "cold",
        "concurrency": "none",
        "collected_at_utc": PLACEHOLDER,
        "collector": PLACEHOLDER,
        "command": PLACEHOLDER,
        "evidence_origin": "synthetic_fixture",
    }
    bundle = {
        "schema_version": EVIDENCE_SCHEMA,
        "evidence_kind": "synthetic_fixture",
        "warning": (
            "scaffold written by f5_harness.py init; synthetic until a collector "
            "fills every field and flips evidence_kind to 'measured'"
        ),
        "provenance": provenance,
        "protocol": dict(context.get("protocol", {})),
        "cases": prescribed_cases(p),
        "runs": [],
        "quality": [],
        "capacity": {
            "occupied_positions": 0,
            "over_capacity_tokens": 0,
            "over_capacity_rejected": False,
            "over_capacity_state_intact": False,
            "gen_start_tokens": 0,
            "gen_count": 0,
            "gen_end_tokens": 0,
            "gen_ok": False,
        },
        "restore": [],
        "cycles": [],
    }
    apply_bindings(bundle, context)
    return bundle


def _reject_report(reason: str) -> dict:
    return {
        "schema_version": REPORT_SCHEMA,
        "verdict": "reject",
        "structural_status": "fail",
        "evidence_kind": None,
        "reasons": [reason],
        "kind_reason": "evidence_kind:missing_or_unknown",
        "checks": {"evidence_object": {"status": "fail", "reasons": [reason]}},
    }


def _write_out(ns: argparse.Namespace, payload: str) -> None:
    out = getattr(ns, "out", None)
    if out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="f5_harness.py",
        description=(
            "Offline F5 evidence harness: schedule, validate and summarise "
            "occupancy position-split campaign evidence. CPU-only; never runs "
            "servers, builds or GPU workloads."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    schema = sub.add_parser("schema", help="print the machine-readable schema")
    schema.add_argument("--out")

    plan = sub.add_parser("plan", help="emit the warmup + alternating A/B schedule")
    plan.add_argument("--context", required=True)
    plan.add_argument("--out")

    template = sub.add_parser(
        "template", help="emit server command templates (documentation only, NOT RUN)"
    )
    template.add_argument("--context", required=True)
    template.add_argument("--out")

    init = sub.add_parser("init", help="write a synthetic evidence scaffold")
    init.add_argument("--context", required=True)
    init.add_argument("--out", required=True)

    validate_cmd = sub.add_parser("validate", help="admissibility check of evidence")
    validate_cmd.add_argument("evidence")
    validate_cmd.add_argument("--context", required=True)
    validate_cmd.add_argument("--artifact-root")
    validate_cmd.add_argument("--out")

    summarize_cmd = sub.add_parser(
        "summarize", help="compute gates from admissible evidence"
    )
    summarize_cmd.add_argument("evidence")
    summarize_cmd.add_argument("--context", required=True)
    summarize_cmd.add_argument("--artifact-root")
    summarize_cmd.add_argument("--require-gates-pass", action="store_true")
    summarize_cmd.add_argument("--out")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_cli()
    ns = parser.parse_args(argv)
    try:
        if ns.command == "schema":
            payload = dump_json(build_schema_doc())
            print(payload, end="")
            _write_out(ns, payload)
            return 0

        context = load_context(ns.context)

        if ns.command == "plan":
            payload = dump_json(build_schedule(context))
            print(payload, end="")
            _write_out(ns, payload)
            return 0

        if ns.command == "template":
            payload = dump_json(build_templates(context))
            print(payload, end="")
            _write_out(ns, payload)
            return 0

        if ns.command == "init":
            payload = dump_json(build_skeleton(context))
            _write_out(ns, payload)
            print(payload, end="")
            return 0

        try:
            bundle = load_json_file(ns.evidence)
        except HarnessError as exc:
            if exc.code == "malformed_json":
                report = _reject_report(f"malformed_json:{exc.message}")
                print(dump_json(report), end="")
                _write_out(ns, dump_json(report))
                return 1
            raise

        if ns.command == "validate":
            report = validate(bundle, context, ns.artifact_root)
            payload = dump_json(report)
            print(payload, end="")
            _write_out(ns, payload)
            return 0 if report["verdict"] == "accept" else 1

        summary, exit_code = summarize(
            bundle, context, ns.artifact_root, ns.require_gates_pass
        )
        payload = dump_json(summary)
        print(payload, end="")
        _write_out(ns, payload)
        if not summary["admissible"]:
            return 1
        return exit_code

    except HarnessError as exc:
        print(
            dump_json({"error": exc.code, "message": exc.message}),
            file=sys.stderr,
            end="",
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
