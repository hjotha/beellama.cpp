#!/usr/bin/env python3
"""Deterministic offline fixtures for the F5 evidence harness.

Every evidence fixture produced here is declared `synthetic_fixture` with
`provenance.evidence_origin = "synthetic_fixture"`. The harness rejects that
combination for F5 acceptance even if `evidence_kind` is relabelled to
`measured`, and it rejects fixtures whose provenance still carries placeholder
markers. These files only exercise the validator and summariser on a CPU with
no GPU, server, model or network involved.

Usage:
    python3 tools/occupancy-f5/fixtures/make_fixtures.py
    python3 tools/occupancy-f5/fixtures/make_fixtures.py --out-dir /tmp/fixtures
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import f5_harness as h  # noqa: E402

GIB = h.GIB
MIB = h.MIB

CONTEXT_FILENAME = "expected-context.json"
FIXTURE_FILENAMES = (
    "complete-synthetic.json",
    "malformed.json",
    "missing-evidence.json",
    "nonfinite.json",
    "wrong-provenance.json",
    "insufficient-repetitions.json",
)

METRICS = {
    "below_p_96000": {
        h.XXL_ARM: {"prefill": 30000.0, "ttft": 30200.0, "lat": 40.0},
        h.CANDIDATE_ARM: {"prefill": 30500.0, "ttft": 30700.0, "lat": 40.5},
    },
    "above_p_105000": {
        h.XXXL_ARM: {"prefill": 40000.0, "ttft": 40200.0, "lat": 45.0},
        h.CANDIDATE_ARM: {"prefill": 34000.0, "ttft": 34200.0, "lat": 46.0},
    },
    "above_p_115000": {
        h.XXXL_ARM: {"prefill": 45000.0, "ttft": 45200.0, "lat": 48.0},
        h.CANDIDATE_ARM: {"prefill": 38000.0, "ttft": 38200.0, "lat": 49.0},
    },
    "above_p_130560": {
        h.XXXL_ARM: {"prefill": 52000.0, "ttft": 52200.0, "lat": 52.0},
        h.CANDIDATE_ARM: {"prefill": 43000.0, "ttft": 43200.0, "lat": 53.0},
    },
}

VRAM_TOTAL = 16 * GIB
CUDA_PEAKS = {
    "load": 10 * GIB,
    "prefill": 14 * GIB,
    "decode": 13 * GIB,
    "restore": 12 * GIB,
    "boundary_cross": 14 * GIB,
}
VULKAN_PEAKS = {
    "load": 96 * MIB,
    "prefill": 512 * MIB,
    "decode": 512 * MIB,
    "restore": 256 * MIB,
    "boundary_cross": 608 * MIB,
}
VULKAN_COPY_BYTES = {
    "load": 0,
    "prefill": 0,
    "decode": 0,
    "restore": 0,
    "boundary_cross": 4 * MIB,
}
HOST_PEAKS = {
    "load": 6 * GIB,
    "prefill": 8 * GIB,
    "decode": 8 * GIB,
    "restore": 7 * GIB,
    "boundary_cross": 8 * GIB,
}

MALFORMED_TEXT = '{"schema_version": "f5-evidence-v1", "evidence_kind": "syn'


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def context_dict() -> dict:
    return {
        "schema_version": h.CONTEXT_SCHEMA,
        "notes": (
            "Fixture context: synthetic paths and content hashes, never a real "
            "campaign. Replace model/corpus hashes with the real campaign values "
            "before any measured run."
        ),
        "expected": {
            "commit": "d9db0b88155ec4350942306696378c57ac0485c8",
            "branch": "feat/occupancy-position-split",
            "model_path": "fixtures/synthetic-target-model.gguf",
            "model_sha256": _digest("f5-fixture-model-gguf"),
            "corpus_path": "fixtures/synthetic-corpus.txt",
            "corpus_sha256": _digest("f5-fixture-corpus"),
            "p": h.PLAN_DEFAULT_P,
            "c": h.CAPACITY_C,
            "seed": 42,
            "batch_size": 512,
            "ubatch_size": 256,
        },
        "protocol": {
            "warmup_runs": 1,
            "measured_runs_per_arm": 5,
            "alternating": True,
            "first_arm": "baseline",
            "min_token_samples": h.DEFAULT_MIN_TOKEN_SAMPLES,
        },
    }


def raw_payload(run_id: str, context_sha256: str, provenance_sha256: str) -> bytes:
    header = (
        f"{h.LOG_HEADER_PREFIX} run_id={run_id} "
        f"context_sha256={context_sha256} provenance_sha256={provenance_sha256}\n"
    )
    body = f"synthetic raw log for {run_id} (invented values, not a measurement)\n"
    return header.encode("utf-8") + body.encode("utf-8")


def _telemetry(above_p: bool) -> dict:
    stage_names = list(h.TELEMETRY_STAGES)
    if above_p:
        stage_names.append(h.BOUNDARY_STAGE)
    return {
        "cuda": {
            "stages": {st: {"peak_bytes": CUDA_PEAKS[st]} for st in stage_names}
        },
        "vulkan": {
            "stages": {
                st: {
                    "peak_bytes": VULKAN_PEAKS[st],
                    "attention_copy_bytes": VULKAN_COPY_BYTES[st],
                }
                for st in stage_names
            },
            "gtt_peak_bytes": 608 * MIB,
            "gtt_available_min_bytes": 2 * GIB,
        },
        "host": {
            "stages": {st: {"peak_bytes": HOST_PEAKS[st]} for st in stage_names},
            "mem_available_min_bytes": 8 * GIB,
            "swap_in_bytes": 0,
            "swap_out_bytes": 0,
        },
    }


def _latencies(base: float) -> list[float]:
    return [
        round(base + 0.4 * ((index * 5) % 7), 3)
        for index in range(h.DEFAULT_MIN_TOKEN_SAMPLES)
    ]


def _provenance(context: dict) -> dict:
    expected = context["expected"]
    return {
        "evidence_origin": "synthetic_fixture",
        "commit": expected["commit"],
        "branch": expected["branch"],
        "dirty": True,
        "model_path": expected["model_path"],
        "model_sha256": expected["model_sha256"],
        "corpus_path": expected["corpus_path"],
        "corpus_sha256": expected["corpus_sha256"],
        "prompt_sha256": _digest("f5-fixture-prompt"),
        "binary_path": "build/bin/llama-server",
        "binary_sha256": _digest("f5-fixture-binary"),
        "p": expected["p"],
        "c": expected["c"],
        "cache_types": {"local": "kvarn4", "overflow": "q4_0"},
        "seed": expected["seed"],
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "greedy": True},
        "batch_size": expected["batch_size"],
        "ubatch_size": expected["ubatch_size"],
        "hardware": {
            "gpu": "SYNTHETIC-FIXTURE-GPU",
            "vram_total_bytes": VRAM_TOTAL,
        },
        "driver": "SYNTHETIC-FIXTURE-DRIVER",
        "build": "SYNTHETIC-FIXTURE-BUILD",
        "clocks": {"sm_mhz": 0, "mem_mhz": 0},
        "governor": "auto",
        "cache_state": "cold",
        "concurrency": "none",
        "collected_at_utc": "2026-10-05T00:00:00+00:00",
        "collector": "make_fixtures.py (synthetic, never a measurement)",
        "command": (
            "build/bin/llama-server -m fixtures/synthetic-target-model.gguf "
            "-c 131072 -b 512 -ub 256 -fa on --spec-type none "
            "--spec-draft-n-max 0"
        ),
    }


def build_bundle(context: dict) -> dict:
    cases = h.prescribed_cases(int(context["expected"]["p"]))
    protocol = dict(context["protocol"])
    provenance = _provenance(context)
    context_hash = h.context_binding(context)
    provenance_hash = h.canonical_sha256(provenance)

    runs: list[dict] = []
    measured_ids_by_key: dict[tuple[str, str], list[str]] = {}
    for case in cases:
        case_id = case["case_id"]
        above_p = case["kind"] == "above_p"
        arms = (case["baseline_arm"], h.CANDIDATE_ARM)
        schedule: list[tuple[bool, str]] = []
        for warmup_arm in arms:
            for _ in range(protocol["warmup_runs"]):
                schedule.append((True, warmup_arm))
        for _ in range(protocol["measured_runs_per_arm"]):
            schedule.extend([(False, arms[0]), (False, arms[1])])
        case_sequence = 0
        for warmup, arm in schedule:
            spec = METRICS[case_id][arm]
            if warmup:
                run_id = f"{case_id}-warm-{arm}"
            else:
                run_id = f"{case_id}-m{len(runs):03d}"
            run: dict = {
                "run_id": run_id,
                "case_id": case_id,
                "arm": arm,
                "warmup": warmup,
                "oom_count": 0,
                "sequence_index": case_sequence,
                "prefill_ms": spec["prefill"],
                "ttft_ms": spec["ttft"],
                "token_latencies_ms": _latencies(spec["lat"]),
            }
            case_sequence += 1
            if not warmup:
                run["telemetry"] = _telemetry(above_p)
                run["routes"] = {"vulkan_attention_executed": above_p}
                payload = raw_payload(run_id, context_hash, provenance_hash)
                run["artifacts"] = [
                    {"path": f"raw/{run_id}.log", "sha256": h.sha256_of(payload)}
                ]
                measured_ids_by_key.setdefault((case_id, arm), []).append(run_id)
            runs.append(run)

    quality: list[dict] = []
    for index, case in enumerate(cases):
        base_ppl = 9.0 + index * 0.5
        for arm in (case["baseline_arm"], h.CANDIDATE_ARM):
            is_candidate = arm == h.CANDIDATE_ARM
            quality.append(
                {
                    "case_id": case["case_id"],
                    "arm": arm,
                    "kl_mean_nat_per_token": 0.0024 if is_candidate else 0.0020,
                    "perplexity": (
                        round(base_ppl * 1.005, 6) if is_candidate else base_ppl
                    ),
                    "reference": h.QUALITY_REFERENCE,
                    "run_ids": list(measured_ids_by_key[(case["case_id"], arm)]),
                }
            )

    capacity_case = next(
        (case for case in cases if case["length_tokens"] == h.CAPACITY_C - 512),
        cases[-1],
    )
    capacity_ids = list(measured_ids_by_key[(capacity_case["case_id"], capacity_case["baseline_arm"])])

    restore: list[dict] = []
    for length in h.required_restore_lengths(cases):
        matching = [case for case in cases if case["length_tokens"] == length]
        if matching:
            entry_case = matching[0]
            run_ids = list(
                measured_ids_by_key[(entry_case["case_id"], entry_case["baseline_arm"])]
            )
        else:
            run_ids = list(capacity_ids)
        restore.append(
            {
                "length_tokens": length,
                "scenario": "exact_import",
                "tokens_reused": length,
                "prefix_reprefill_tokens": 0,
                "next_logits_max_abs_diff": 0.0,
                "ok": True,
                "run_ids": run_ids,
            }
        )

    all_measured_ids = [rid for ids in measured_ids_by_key.values() for rid in ids]
    cycles: list[dict] = []
    for index in range(h.RESTORE_CYCLES):
        cycles.append(
            {
                "index": index + 1,
                "restore_ok": True,
                "divergent_suffix_ok": True,
                "generation_ok": True,
                "retained_bytes": {
                    "cuda": 12 * GIB + index * 4 * MIB,
                    "vulkan": 3 * GIB + index * 2 * MIB,
                },
                "oom_count": 0,
                "driver_error_count": 0,
                "corruption_count": 0,
                "run_ids": [all_measured_ids[index % len(all_measured_ids)]],
            }
        )

    bundle = {
        "schema_version": h.EVIDENCE_SCHEMA,
        "evidence_kind": "synthetic_fixture",
        "provenance": provenance,
        "protocol": protocol,
        "cases": copy.deepcopy(cases),
        "runs": runs,
        "quality": quality,
        "capacity": {
            "occupied_positions": h.CAPACITY_C,
            "over_capacity_tokens": h.CAPACITY_C + 1,
            "over_capacity_rejected": True,
            "over_capacity_state_intact": True,
            "gen_start_tokens": h.CAPACITY_C - 512,
            "gen_count": 512,
            "gen_end_tokens": h.CAPACITY_C,
            "gen_ok": True,
            "run_ids": capacity_ids,
        },
        "restore": restore,
        "cycles": cycles,
        "notes": [
            "Synthetic fixture: every number here is invented for parser/validator "
            "coverage. It is not a measurement and must never be quoted as an F5 "
            "result. evidence_origin stays 'synthetic_fixture' so relabelling "
            "evidence_kind to 'measured' cannot make it acceptable.",
        ],
    }
    h.apply_bindings(bundle, context)
    return bundle


def _variant_missing_evidence(bundle: dict) -> dict:
    out = copy.deepcopy(bundle)
    out.pop("cycles")
    out["quality"] = []
    measured = [i for i, r in enumerate(out["runs"]) if r["warmup"] is False]
    del out["runs"][measured[2]]["telemetry"]
    return out


def _variant_nonfinite(bundle: dict) -> dict:
    out = copy.deepcopy(bundle)
    measured = [i for i, r in enumerate(out["runs"]) if r["warmup"] is False]
    out["runs"][measured[0]]["prefill_ms"] = float("inf")
    out["quality"][0]["kl_mean_nat_per_token"] = float("nan")
    return out


def _variant_wrong_provenance(bundle: dict) -> dict:
    out = copy.deepcopy(bundle)
    out["provenance"]["model_sha256"] = "1" * 64
    return out


def _variant_insufficient(bundle: dict) -> dict:
    out = copy.deepcopy(bundle)
    keep: list[dict] = []
    counted: dict[tuple, int] = {}
    for run in out["runs"]:
        if run["warmup"]:
            keep.append(run)
            continue
        key = (run["case_id"], run["arm"])
        if counted.get(key, 0) < 3:
            counted[key] = counted.get(key, 0) + 1
            keep.append(run)
    out["runs"] = keep
    return out


def build_all(out_dir: Path) -> dict[str, bytes]:
    context = context_dict()
    complete = build_bundle(context)
    variants = {
        "complete-synthetic.json": complete,
        "missing-evidence.json": _variant_missing_evidence(complete),
        "nonfinite.json": _variant_nonfinite(complete),
        "wrong-provenance.json": _variant_wrong_provenance(complete),
        "insufficient-repetitions.json": _variant_insufficient(complete),
    }
    files: dict[str, bytes] = {
        CONTEXT_FILENAME: (
            json.dumps(context, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "malformed.json": MALFORMED_TEXT.encode("utf-8"),
    }
    for name, bundle in variants.items():
        files[name] = (
            json.dumps(bundle, indent=2, sort_keys=True, allow_nan=True) + "\n"
        ).encode("utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (out_dir / name).write_bytes(payload)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).resolve().parent
    )
    ns = parser.parse_args(argv)
    files = build_all(ns.out_dir)
    for name in sorted(files):
        print(f"wrote {ns.out_dir / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
