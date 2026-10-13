#!/usr/bin/env python3
"""Calibrate split-candidate KL limits against paired fixed-KV baselines.

This tool compares teacher-forced logits only. A passing result is not evidence
for production O/LSE parity, performance, capacity, or serving acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import compare_logits

SCHEMA = "occupancy-kl-calibrated-v1"
PAIRED_SCHEMA = "occupancy-native-paired-teacher-v1"
MINIMUM_POSITIONS = 32
MEAN_ALLOWANCE = 1e-4
P99_ALLOWANCE = 1e-3
NEGATIVE_KL_TOLERANCE = -1e-10

_CONFIG_REQUIRED = ("model_sha256", "n_ctx", "n_batch", "n_ubatch", "n_seq", "n_params", "p",
                    "kv_k", "kv_v", "overflow_k", "overflow_v", "remote_layers")
_FORMAT_KEYS = {"kv_k", "kv_v", "cache_k", "cache_v", "cache_type_k", "cache_type_v",
                "overflow_k", "overflow_v", "overflow_type_k", "overflow_type_v",
                "ovf_k", "ovf_v", "ovf_type_k", "ovf_type_v"}
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


class InvalidEvidence(ValueError):
    """Input evidence is incomplete, malformed, or not bound to this comparison."""


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidEvidence(f"cannot read JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise InvalidEvidence(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidEvidence(message)


def _config_core(report: dict[str, Any], label: str) -> dict[str, str]:
    configuration = report.get("configuration")
    _require(isinstance(configuration, dict), f"{label}: missing configuration object")
    text = configuration.get("config_core")
    _require(isinstance(text, str) and bool(text), f"{label}: missing config_core")
    fields: dict[str, str] = {}
    for item in text.split(";"):
        if "=" not in item:
            raise InvalidEvidence(f"{label}: malformed config_core field {item!r}")
        key, value = item.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or (not value and key != "remote_layers") or key in fields:
            raise InvalidEvidence(f"{label}: empty or duplicate config_core field {key!r}")
        fields[key] = value
    missing = [key for key in _CONFIG_REQUIRED if key not in fields]
    _require(not missing, f"{label}: config_core missing {', '.join(missing)}")
    for key in ("n_ctx", "n_batch", "n_ubatch", "n_seq", "n_params", "p"):
        try:
            number = int(fields[key], 10)
        except ValueError as exc:
            raise InvalidEvidence(f"{label}: config_core.{key} is not an integer") from exc
        _require(number >= 0, f"{label}: config_core.{key} is negative")
    config = report["configuration"]
    model_sha = config.get("model_sha256_operator_verified")
    build_sha = config.get("build_bundle_sha256_operator_verified")
    _require(isinstance(model_sha, str) and _HEX64.fullmatch(model_sha),
             f"{label}: missing or invalid operator-verified model SHA-256")
    _require(isinstance(build_sha, str) and _HEX64.fullmatch(build_sha),
             f"{label}: missing or invalid operator-verified build-bundle SHA-256")
    _require(fields["model_sha256"].lower() == model_sha.lower(),
             f"{label}: config_core model SHA differs from operator-verified model SHA")
    return fields


def _report(root: Path, label: str) -> tuple[dict[str, Any], Path, dict[str, str]]:
    _require(root.is_dir(), f"{label}: output root is not a directory: {root}")
    path = root / "paired-result.json"
    report = _json(path)
    _require(isinstance(report, dict) and report.get("schema") == PAIRED_SCHEMA,
             f"{label}: paired-result.json schema must be {PAIRED_SCHEMA}")
    _require(report.get("complete") is True, f"{label}: paired report is incomplete")
    _require(report.get("fixed_current_kv") is True,
             f"{label}: fixed_current_kv must be true; natural/old protocol is diagnostic only")
    _require(report.get("fixed_memory_reference_complete") is True,
             f"{label}: fixed-memory reference is incomplete")
    _require(report.get("reference_arm") == "cpu" and
             report.get("checkpoint_method") == "full",
             f"{label}: report must use the CPU reference and full checkpoints")
    _require(isinstance(report.get("configuration"), dict) and
             report["configuration"].get("reference_arm") == "cpu" and
             report["configuration"].get("checkpoint_method") == "full" and
             report["configuration"].get("fixed_current_kv") is True and
             report["configuration"].get("full_prefix_verification") is True,
             f"{label}: configuration does not bind fixed-current-KV full-prefix protocol")
    if label == "candidate":
        _require(report.get("calibration_candidate_eligible") is True,
                 "candidate calibration_candidate_eligible must be true")
    config = _config_core(report, label)
    ranges = report.get("ranges")
    _require(isinstance(ranges, list) and ranges, f"{label}: missing ranges")
    seen_starts: set[int] = set()
    for index, item in enumerate(ranges):
        _require(isinstance(item, dict), f"{label}: range {index} is not an object")
        start, count = item.get("start"), item.get("count")
        _require(type(start) is int and start > 0 and type(count) is int and count >= MINIMUM_POSITIONS,
                 f"{label}: range {index} needs positive start and count >= {MINIMUM_POSITIONS}")
        _require(start not in seen_starts, f"{label}: duplicate range start {start}")
        seen_starts.add(start)
        _require(item.get("complete") is True,
                 f"{label}: range {start} is incomplete")
        _require(item.get("fixed_memory_reference_complete") is True,
                 f"{label}: range {start} fixed-memory reference is incomplete")
        rows = item.get("rows")
        _require(isinstance(rows, list) and len(rows) == count,
                 f"{label}: range {start} row count differs from declared count")
        expected_positions = list(range(start, start + count))
        positions = [row.get("n") if isinstance(row, dict) else None for row in rows]
        _require(positions == expected_positions,
                 f"{label}: range {start} rows are not the full consecutive checkpoint set")
        for row in rows:
            n = row["n"]
            _require(row.get("state") == "complete",
                     f"{label}: checkpoint {n} row is incomplete")
            _require(row.get("checkpoint_method") == "full",
                     f"{label}: checkpoint {n} is not a full checkpoint")
            prefix = row.get("prefix_proof")
            _require(isinstance(prefix, dict) and prefix.get("verified") is True and
                     prefix.get("equal") is True,
                     f"{label}: checkpoint {n} row-prefix proof is missing or unequal")
            proof = row.get("current_kv_proof")
            _require(isinstance(proof, dict) and proof.get("complete") is True and
                     proof.get("all_byte_equal") is True,
                     f"{label}: checkpoint {n} current-KV byte proof is incomplete or unequal")
            required = proof.get("required_count")
            _require(type(required) is int and required > 0 and
                     type(proof.get("captured_count")) is int and proof["captured_count"] == required and
                     type(proof.get("injected_count")) is int and proof["injected_count"] == required,
                     f"{label}: checkpoint {n} fixed-KV proof counts do not match")
            _require(isinstance(proof.get("tensor_manifest"), list) and
                     len(proof["tensor_manifest"]) == required,
                     f"{label}: checkpoint {n} fixed-KV tensor manifest is incomplete")
            gpu = row.get("post_gpu_restore_proof")
            _require(isinstance(gpu, dict) and gpu.get("equal") is True,
                     f"{label}: checkpoint {n} GPU post-restore byte proof is missing or unequal")
    return report, path, config


def _baseline_classification(report: dict[str, Any], config: dict[str, str]) -> str:
    _require(int(config["p"]) == 0, "baseline config_core must have P=0")
    if report.get("pure_cuda_baseline") is True and report.get("layered_baseline") is False:
        _require(report.get("calibration_baseline_eligible") is True,
                 "baseline pure-CUDA report is not calibration eligible")
        _require(config["remote_layers"].lower() == "auto",
                 "pure-CUDA baseline remote_layers must be auto (remote placement disabled)")
        return "pure_cuda_xxl"
    if report.get("layered_baseline") is True and report.get("pure_cuda_baseline") is False:
        _require(report.get("calibration_baseline_eligible") is True,
                 "baseline layered report is not calibration eligible")
        try:
            layer_count = int(config["remote_layers"], 10)
        except ValueError as exc:
            raise InvalidEvidence("layered baseline remote_layers must be a positive integer") from exc
        _require(layer_count > 0,
                 "layered baseline remote_layers must be a positive integer")
        return "layered_xxxl_diagnostic"
    raise InvalidEvidence("baseline lacks explicit pure-CUDA-XXL or layered-XXXL-diagnostic classification")


def _full_layer_count(report: dict[str, Any], label: str) -> int:
    counts = {
        row["current_kv_proof"]["required_count"]
        for range_report in report["ranges"]
        for row in range_report["rows"]
    }
    _require(len(counts) == 1, f"{label}: required current-KV tensor count changes across rows")
    tensor_count = next(iter(counts))
    _require(tensor_count % 2 == 0,
             f"{label}: current-KV tensor count is not K/V pairs")
    return tensor_count // 2


def _validate_remote_placement(report: dict[str, Any], config: dict[str, str],
                               label: str, full_layers: int) -> str:
    value = config["remote_layers"].lower()
    if label == "candidate":
        if value == "full":
            return value
        try:
            layer_count = int(value, 10)
        except ValueError as exc:
            raise InvalidEvidence(
                "candidate remote_layers must be full or equal the full KV layer count") from exc
        _require(layer_count == full_layers,
                 f"candidate remote_layers={layer_count} must cover all {full_layers} KV layers")
        return value
    if report.get("pure_cuda_baseline") is True:
        _require(value == "auto",
                 "pure-CUDA baseline remote_layers must be auto (remote placement disabled)")
        return value
    try:
        layer_count = int(value, 10)
    except ValueError as exc:
        raise InvalidEvidence("layered baseline remote_layers must be a positive integer") from exc
    _require(0 < layer_count <= full_layers,
             f"layered baseline remote_layers must be in [1,{full_layers}]")
    return value


def _candidate_classification(report: dict[str, Any], config: dict[str, str],
                              full_layers: int) -> str:
    _require(int(config["p"]) > 0, "candidate config_core must have P>0")
    _require(report.get("pure_cuda_baseline") is False and
             report.get("layered_baseline") is False,
             "candidate report is mislabeled as a P=0 baseline")
    _validate_remote_placement(report, config, "candidate", full_layers)
    return "position_split_candidate"


def _cross_configuration(baseline: dict[str, str], candidate: dict[str, str]) -> dict[str, Any]:
    ignored = {"p", "n_ctx", "remote_layers"} | _FORMAT_KEYS
    baseline_keys, candidate_keys = set(baseline), set(candidate)
    _require(baseline_keys == candidate_keys,
             "config_core field sets differ between baseline and candidate")
    mismatched = [key for key in sorted(baseline_keys - ignored)
                  if baseline[key] != candidate[key]]
    _require(not mismatched,
             "disallowed config_core differences: " + ", ".join(mismatched))
    shared = ("model_sha256", "n_batch", "n_ubatch", "n_seq", "n_params")
    return {
        "shared_config_core": {key: baseline[key] for key in shared},
        "baseline_n_ctx": int(baseline["n_ctx"]),
        "candidate_n_ctx": int(candidate["n_ctx"]),
        "baseline_p": int(baseline["p"]),
        "candidate_p": int(candidate["p"]),
        "remote_layers": {"baseline": baseline["remote_layers"],
                          "candidate": candidate["remote_layers"]},
        "allowed_config_core_differences": {
            key: {"baseline": baseline[key], "candidate": candidate[key]}
            for key in sorted(baseline_keys & ignored) if baseline[key] != candidate[key]
        },
    }


def _teacher_tokens(root: Path, override: Path | None, label: str) -> tuple[list[int], Path]:
    path = override
    if path is not None and path.is_dir():
        path = path / "teacher-tokens.json"
    if path is None:
        path = root / "teacher-tokens.json"
    _require(path.is_file(), f"{label}: missing teacher-tokens.json (or explicit teacher JSON)")
    value = _json(path)
    if isinstance(value, dict):
        value = value.get("prompt")
    _require(isinstance(value, list) and all(type(token) is int for token in value),
             f"{label}: teacher tokens must be an integer array")
    return value, path


def _capture_meta(directory: Path, position: int) -> tuple[dict[str, Any], Path]:
    path = directory / f"cap.{directory.name}.n{position}.json"
    # Capture phase names are cpu/gpu, independent of report role.
    if not path.is_file():
        candidates = list(directory.glob(f"*.n{position}.json"))
        _require(len(candidates) == 1,
                 f"capture metadata missing or ambiguous at {directory}/n{position}")
        path = candidates[0]
    metadata = _json(path)
    _require(isinstance(metadata, dict), f"capture metadata is not an object: {path}")
    return metadata, path


def _capture_files(directory: Path) -> list[dict[str, str]]:
    captures = sorted(p for p in directory.glob("*.n*")
                      if p.suffix in {".tokens", ".logits", ".json"})
    _require(bool(captures), f"no capture artifacts found in {directory}")
    return [{"path": str(path.resolve()), "sha256": _sha256(path)} for path in captures]


def _validate_capture_pair(root: Path, report: dict[str, Any], label: str) -> dict[int, dict[str, Path]]:
    result: dict[int, dict[str, Path]] = {}
    for run in report["ranges"]:
        start, count = run["start"], run["count"]
        range_dir = root / f"range-{start}"
        phases: dict[str, Path] = {}
        for phase in ("cpu", "gpu"):
            directory = range_dir / phase
            _require(directory.is_dir(), f"{label}: missing {phase} captures for range {start}")
            indexed = compare_logits.capture_index(directory)
            expected = set(range(start, start + count))
            _require(set(indexed) == expected,
                     f"{label}: {phase} capture set differs from paired checkpoint rows at range {start}")
            phases[phase] = directory
            for n in sorted(expected):
                metadata, metadata_path = _capture_meta(directory, n)
                _require(metadata.get("checkpoint_method") == "full",
                         f"{label}: {phase} capture {n} is not a full checkpoint")
                _require(metadata.get("split_p") == int(_config_core(report, label)["p"]),
                         f"{label}: {phase} capture {n} P differs from paired config")
                prefix = metadata.get("prefix_proof")
                _require(isinstance(prefix, dict) and prefix.get("verified") is True and
                         prefix.get("equal") is True,
                         f"{label}: {phase} capture {n} prefix proof is incomplete")
                cfg = _config_core(report, label)
                _require(metadata.get("model_sha256") == cfg["model_sha256"] and
                         metadata.get("n_params") == int(cfg["n_params"]),
                         f"{label}: {phase} capture {n} model provenance differs")
                _require(metadata.get("model_fingerprint") == "sha256:" + cfg["model_sha256"],
                         f"{label}: {phase} capture {n} model fingerprint differs from model SHA")
                _require(metadata.get("build_id") == report["configuration"][
                    "build_bundle_sha256_operator_verified"],
                    f"{label}: {phase} capture {n} build bundle differs from paired report")
                _require(metadata.get("config_core") == report["configuration"]["config_core"] and
                         metadata.get("n_ctx") == int(cfg["n_ctx"]),
                         f"{label}: {phase} capture {n} config_core/n_ctx differs from paired report")
                _require(metadata.get("tokens_n") == n and metadata.get("input_pos") == n - 1 and
                         metadata.get("output_pos") == n - 1,
                         f"{label}: {phase} capture {n} token position differs")
                _require(metadata.get("finite") is True and
                         type(metadata.get("n_vocab")) is int and metadata["n_vocab"] > 0,
                         f"{label}: {phase} capture {n} has invalid/nonfinite logits metadata")
        for n in sorted(expected):
            cpu_meta, _ = _capture_meta(phases["cpu"], n)
            gpu_meta, _ = _capture_meta(phases["gpu"], n)
            for key in ("config_core", "n_ctx", "model_fingerprint", "model_sha256",
                        "n_params", "n_vocab", "split_p"):
                _require(cpu_meta.get(key) == gpu_meta.get(key),
                         f"{label}: CPU/GPU {key} differs at checkpoint {n}")
        # CPU and GPU are a strict same-configuration pair within each root.
        result[start] = phases
    return result


def _compare_arm(cpu_dir: Path, gpu_dir: Path, teacher: list[int],
                 range_report: dict[str, Any], minimum: int,
                 label: str) -> dict[str, Any]:
    """Recompute each arm's CPU-reference/GPU-candidate logits in FP64."""
    try:
        metrics = compare_logits.compare(cpu_dir, gpu_dir, teacher, minimum=minimum)
    except (ValueError, OSError, OverflowError) as exc:
        raise InvalidEvidence(f"{label} logit comparison failed: {exc}") from exc
    expected_rows = range_report["rows"]
    _require(len(metrics["rows"]) == len(expected_rows),
             f"{label}: recomputed row count differs from paired report")
    for expected, actual in zip(expected_rows, metrics["rows"]):
        value = expected.get("metrics")
        _require(isinstance(value, dict) and type(value.get("kl_nat")) in (int, float) and
                 math.isfinite(value["kl_nat"]),
                 f"{label}: missing/nonfinite C++ row KL at checkpoint {actual['position']}")
        _require(value["kl_nat"] >= NEGATIVE_KL_TOLERANCE,
                 f"{label}: negative C++ KL below tolerance at checkpoint {actual['position']}")
        _require(abs(float(value["kl_nat"]) - actual["kl"]) <= 1e-10,
                 f"{label}: C++/Python row KL differs by more than 1e-10 at checkpoint {actual['position']}")
        _require(expected.get("n") == actual["position"],
                 f"{label}: row position differs at checkpoint {actual['position']}")
        proof = expected.get("prefix_proof")
        _require(isinstance(proof, dict) and proof.get("verified") is True and
                 proof.get("equal") is True,
                 f"{label}: row-prefix proof is missing or unequal at checkpoint {actual['position']}")
    for key, metric_key in (("mean_kl_nat_per_token", "mean"),
                            ("p99_kl_nat_per_token", "p99")):
        reported = range_report.get(key)
        actual = metrics[key]
        _require(type(reported) in (int, float) and math.isfinite(reported) and
                 abs(float(reported) - actual) <= 1e-10,
                 f"{label}: C++/Python {metric_key} KL differs by more than 1e-10")
        _require(reported >= NEGATIVE_KL_TOLERANCE,
                 f"{label}: negative C++ aggregate {metric_key} KL below tolerance")
    for row in metrics["rows"]:
        _require(math.isfinite(row["kl"]), f"nonfinite KL at checkpoint {row['position']}")
        _require(row["kl"] >= NEGATIVE_KL_TOLERANCE,
                 f"negative KL below tolerance at checkpoint {row['position']}: {row['kl']}")
    _require(math.isfinite(metrics["mean_kl_nat_per_token"]) and
             math.isfinite(metrics["p99_kl_nat_per_token"]), "nonfinite aggregate KL")
    _require(metrics["mean_kl_nat_per_token"] >= NEGATIVE_KL_TOLERANCE and
             metrics["p99_kl_nat_per_token"] >= NEGATIVE_KL_TOLERANCE,
             "negative aggregate KL below tolerance")
    return metrics


def _validate_fixed_kv_manifest(root: Path, row: dict[str, Any], label: str) -> list[dict[str, Any]]:
    proof = row["current_kv_proof"]
    manifest = proof["tensor_manifest"]
    result = []
    for entry in manifest:
        _require(isinstance(entry, dict), f"{label}: malformed current-KV manifest entry")
        path_text = entry.get("path")
        _require(isinstance(path_text, str) and bool(path_text),
                 f"{label}: current-KV manifest entry has no path")
        byte_count = entry.get("bytes")
        _require(type(byte_count) is int and byte_count > 0,
                 f"{label}: invalid current-KV tensor byte count")
        _require(isinstance(entry.get("xxh64"), str) and
                 re.fullmatch(r"[0-9a-fA-F]{16}", entry["xxh64"]) is not None,
                 f"{label}: invalid current-KV tensor xxh64")
        _require(isinstance(entry.get("name"), str) and entry.get("type") == "F32" and
                 entry.get("contiguous") is True and type(entry.get("nq")) is int and
                 entry.get("nq") == 1,
                 f"{label}: invalid current-KV tensor layout metadata")
        shape = entry.get("shape")
        _require(isinstance(shape, list) and shape and
                 all(type(dim) is int and dim > 0 for dim in shape),
                 f"{label}: invalid current-KV tensor shape")
        result.append({"path": path_text, "bytes": byte_count,
                       "xxh64": entry["xxh64"], "shape": shape,
                       "name": entry["name"], "type": entry["type"]})
    return result


def calibrate(candidate_root: Path, baseline_root: Path,
              candidate_teacher_path: Path | None = None,
              baseline_teacher_path: Path | None = None,
              minimum: int = MINIMUM_POSITIONS) -> tuple[dict[str, Any], bool]:
    if minimum < MINIMUM_POSITIONS:
        raise InvalidEvidence(f"minimum positions must be at least {MINIMUM_POSITIONS}")
    candidate, candidate_report_path, candidate_config = _report(candidate_root, "candidate")
    baseline, baseline_report_path, baseline_config = _report(baseline_root, "baseline")
    baseline_class = _baseline_classification(baseline, baseline_config)
    baseline_layers = _full_layer_count(baseline, "baseline")
    candidate_layers = _full_layer_count(candidate, "candidate")
    _require(baseline_layers == candidate_layers,
             "baseline and candidate fixed-KV layer counts differ")
    _validate_remote_placement(baseline, baseline_config, "baseline", baseline_layers)
    candidate_class = _candidate_classification(candidate, candidate_config, candidate_layers)
    _require(baseline["configuration"]["build_bundle_sha256_operator_verified"].lower() ==
             candidate["configuration"]["build_bundle_sha256_operator_verified"].lower(),
             "baseline and candidate build_bundle SHA-256 differ")
    _require(baseline["configuration"]["model_sha256_operator_verified"].lower() ==
             candidate["configuration"]["model_sha256_operator_verified"].lower(),
             "baseline and candidate model SHA-256 differ")
    base_teacher_source = baseline["configuration"].get("teacher_file")
    candidate_teacher_source = candidate["configuration"].get("teacher_file")
    _require(isinstance(base_teacher_source, dict) and isinstance(candidate_teacher_source, dict),
             "baseline/candidate teacher-file binding is missing")
    for field in ("bytes", "xxh64"):
        _require(base_teacher_source.get(field) == candidate_teacher_source.get(field),
                 f"baseline and candidate teacher-file {field} binding differs")
    _require(type(base_teacher_source.get("bytes")) is int and base_teacher_source["bytes"] > 0 and
             isinstance(base_teacher_source.get("xxh64"), str) and
             re.fullmatch(r"[0-9a-fA-F]{16}", base_teacher_source["xxh64"]) is not None,
             "teacher-file binding is invalid")
    shared_config = _cross_configuration(baseline_config, candidate_config)
    baseline_teacher, baseline_teacher_file = _teacher_tokens(
        baseline_root, baseline_teacher_path, "baseline")
    candidate_teacher, candidate_teacher_file = _teacher_tokens(
        candidate_root, candidate_teacher_path, "candidate")
    _require(baseline_teacher == candidate_teacher,
             "baseline and candidate teacher-token sets differ")
    _require(len(baseline_teacher) >= minimum + 1,
             f"teacher token file needs at least {minimum + 1} tokens")
    b_phases = _validate_capture_pair(baseline_root, baseline, "baseline")
    c_phases = _validate_capture_pair(candidate_root, candidate, "candidate")
    _require(set(b_phases) == set(c_phases), "baseline and candidate checkpoint ranges differ")

    range_results = []
    passed = True
    baseline_ranges = {item["start"]: item for item in baseline["ranges"]}
    candidate_ranges = {item["start"]: item for item in candidate["ranges"]}
    _require(set(baseline_ranges) == set(candidate_ranges),
             "baseline and candidate range definitions differ")
    fixed_manifests: dict[str, dict[str, list[Any]]] = {"baseline": {}, "candidate": {}}
    for start in sorted(baseline_ranges):
        base_range, cand_range = baseline_ranges[start], candidate_ranges[start]
        start = base_range["start"]
        _require(start == cand_range["start"] and base_range["count"] == cand_range["count"],
                 f"baseline and candidate range definitions differ at {start}")
        count = base_range["count"]
        _require(count >= minimum, f"range {start} has fewer than {minimum} positions")
        base_rows, cand_rows = base_range["rows"], cand_range["rows"]
        _require([r["n"] for r in base_rows] == [r["n"] for r in cand_rows],
                 f"baseline and candidate row positions differ at range {start}")
        base_cpu_gpu = _compare_arm(b_phases[start]["cpu"], b_phases[start]["gpu"],
                                    baseline_teacher, base_range, minimum,
                                    f"baseline range {start}")
        cand_cpu_gpu = _compare_arm(c_phases[start]["cpu"], c_phases[start]["gpu"],
                                    candidate_teacher, cand_range, minimum,
                                    f"candidate range {start}")
        _require(len(base_cpu_gpu["rows"]) == len(cand_cpu_gpu["rows"]),
                 f"baseline and candidate recomputed row counts differ at range {start}")
        for b_row, c_row in zip(base_cpu_gpu["rows"], cand_cpu_gpu["rows"]):
            _require(b_row["position"] == c_row["position"] and
                     b_row["teacher_token"] == c_row["teacher_token"] and
                     b_row["prefix_sha256"] == c_row["prefix_sha256"],
                     f"baseline/candidate checkpoint inputs differ at N={b_row['position']}")
        for n in range(start, start + count):
            bmeta, _ = _capture_meta(b_phases[start]["cpu"], n)
            cmeta, _ = _capture_meta(c_phases[start]["cpu"], n)
            for key in ("model_fingerprint", "model_sha256", "n_params", "n_vocab"):
                _require(bmeta.get(key) == cmeta.get(key),
                         f"baseline/candidate {key} differs at checkpoint {n}")
        base_kls = [row["kl"] for row in base_cpu_gpu["rows"]]
        cand_kls = [row["kl"] for row in cand_cpu_gpu["rows"]]
        for phase, root, rows in (("baseline", baseline_root, base_rows),
                                  ("candidate", candidate_root, cand_rows)):
            fixed_manifests[phase][str(start)] = []
            for row in rows:
                fixed_manifests[phase][str(start)].append({
                    "n": row["n"],
                    "checkpoint_method": row["checkpoint_method"],
                    "prefix_proof": row["prefix_proof"],
                    "current_kv_proof": {
                        "required_count": row["current_kv_proof"]["required_count"],
                        "captured_count": row["current_kv_proof"]["captured_count"],
                        "injected_count": row["current_kv_proof"]["injected_count"],
                        "all_byte_equal": row["current_kv_proof"]["all_byte_equal"],
                        "tensors": _validate_fixed_kv_manifest(
                            root, row, f"{phase} range {start} N={row['n']}")},
                    "post_gpu_restore_proof": row["post_gpu_restore_proof"]})
        base_mean = math.fsum(base_kls) / len(base_kls)
        cand_mean = math.fsum(cand_kls) / len(cand_kls)
        base_p99 = compare_logits.percentile99(base_kls)
        cand_p99 = compare_logits.percentile99(cand_kls)
        delta_mean = cand_mean - base_mean
        delta_p99 = cand_p99 - base_p99
        group_pass = delta_mean <= MEAN_ALLOWANCE and delta_p99 <= P99_ALLOWANCE
        passed = passed and group_pass
        groups = {
            "cpu_reference_vs_native_gpu": {
                "baseline_comparison": base_cpu_gpu,
                "candidate_comparison": cand_cpu_gpu,
                "baseline_mean_kl_nat_per_token": base_mean,
                "candidate_mean_kl_nat_per_token": cand_mean,
                "delta_mean_kl_nat_per_token": delta_mean,
                "baseline_p99_kl_nat_per_token": base_p99,
                "candidate_p99_kl_nat_per_token": cand_p99,
                "delta_p99_kl_nat_per_token": delta_p99,
                "limits": {"delta_mean_max": MEAN_ALLOWANCE, "delta_p99_max": P99_ALLOWANCE},
                "pass": group_pass,
            },
        }
        range_results.append({"start": start, "count": count, "groups": groups,
                              "pass": group_pass})

    provenance = {
        "baseline_root": str(baseline_root.resolve()),
        "candidate_root": str(candidate_root.resolve()),
        "baseline_paired_report": {"path": str(baseline_report_path.resolve()),
                                    "sha256": _sha256(baseline_report_path)},
        "candidate_paired_report": {"path": str(candidate_report_path.resolve()),
                                    "sha256": _sha256(candidate_report_path)},
        "baseline_teacher_tokens": {"path": str(baseline_teacher_file.resolve()),
                                    "sha256": _sha256(baseline_teacher_file)},
        "candidate_teacher_tokens": {"path": str(candidate_teacher_file.resolve()),
                                     "sha256": _sha256(candidate_teacher_file)},
        "model_sha256": baseline["configuration"]["model_sha256_operator_verified"],
        "build_bundle_sha256": baseline["configuration"]["build_bundle_sha256_operator_verified"],
        "baseline_teacher_file": base_teacher_source,
        "candidate_teacher_file": candidate_teacher_source,
        "baseline_classification": baseline_class,
        "candidate_classification": candidate_class,
        "configuration": shared_config,
        "fixed_current_kv_manifests": fixed_manifests,
        "capture_artifacts": {
            "baseline": {str(start): {phase: _capture_files(path)
                                       for phase, path in phases.items()}
                         for start, phases in b_phases.items()},
            "candidate": {str(start): {phase: _capture_files(path)
                                        for phase, path in phases.items()}
                          for start, phases in c_phases.items()},
        },
    }
    return ({"schema": SCHEMA,
             "verdict": "pass" if passed else "fail",
             "scope": "teacher_forced_logits_only",
             "acceptance_claim": "This result does not establish production O/LSE/F16 parity or whole-serving acceptance.",
             "minimum_positions": minimum,
             "calibrated_limits": {"candidate_mean_delta_max": MEAN_ALLOWANCE,
                                   "candidate_p99_delta_max": P99_ALLOWANCE},
             "ranges": range_results,
             "provenance": provenance}, passed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("--candidate-teacher-json", type=Path)
    parser.add_argument("--baseline-teacher-json", type=Path)
    parser.add_argument("--teacher-json", type=Path,
                        help="use this teacher-token JSON (or directory containing it) for both roots")
    parser.add_argument("--minimum", type=int, default=MINIMUM_POSITIONS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.teacher_json is not None and (
                args.candidate_teacher_json is not None or args.baseline_teacher_json is not None):
            raise InvalidEvidence("--teacher-json cannot be combined with per-root teacher JSON options")
        candidate_teacher = args.candidate_teacher_json or args.teacher_json
        baseline_teacher = args.baseline_teacher_json or args.teacher_json
        result, passed = calibrate(args.candidate_root, args.baseline_root,
                                   candidate_teacher, baseline_teacher, args.minimum)
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if passed else 1
    except (InvalidEvidence, OSError, ValueError, OverflowError) as exc:
        print(json.dumps({"schema": SCHEMA, "verdict": "invalid", "reason": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
