#!/usr/bin/env python3
"""Focused MTP acceptance harness, reusing tools/test-mixed-kv-transition.py.

M1: static mixed target (KVarN4 local + one Vulkan Qx layer) with draft-mtp
    ACTIVE. The draft is an independent CUDA KVarN4 context
    (--spec-draft-type-k/v kvarn4). Cold prefixes at KVarN/checkpoint
    boundaries (127/128/129/385), each followed by an isolated determinism
    pair (erase + rebuild identical prefix, then the same extension twice:
    cadence AND tokens/text equality are REQUIRED) and a reuse-only repeat
    (cache_n>0 required; cadence is recorded honestly and may legitimately
    differ because the repeat starts from a later checkpoint).
M2: adaptive MTP 512 -> LONG 1024 with MTP REMAINING ACTIVE
    (--spec-draft-n-max 2 local tier, --spec-draft-n-max-l 2 long tier).
    Correct profile mapping --ctx-size 1024 --ctx-size-mtp 512, KVarN4
    explicit for target global/medium/long and draft medium/long tiers,
    --cache-ram 256 MiB and --ctx-checkpoints 1 (B-style). The gate prompt
    must preserve >0 <=300 tokens (validate_handoff_counts). After the
    handoff the draft must keep generating. Configurable handoff evidence
    markers; without them the run is pending_evidence (exit 1), never falsely
    complete.

Draft evidence contract (checked in source):
  - PRIMARY per-request evidence: the completion response timings expose
    draft_n and draft_n_accepted ONLY when drafts were generated
    (server-common.cpp:161-164). Absent fields mean zero drafts.
  - cache_reprocessed_n in timings exposes reprocessing instead of reuse;
    continuation legs require 0.
  - SECONDARY: the per-task "draft acceptance = ..." log line
    (server-context.cpp:1795-1808) may flush asynchronously after the HTTP
    response; it is recorded and reconciled, never a false fallback.
  - MTP K/V stays on CUDA under the mixed-target contract: the draft params
    clear remote attention and set layers=0 (speculative.cpp:5403-5418),
    logged as "MTP K/V placement: local_attention=off, layers=0".
  - slot save/restore with MTP is NOT exercised (native route is target-only
    + MTP re-bootstrap); documented instead of forced.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shlex
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "test_mixed_kv_transition", _TOOLS_DIR / "test-mixed-kv-transition.py")
assert _spec and _spec.loader
r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r)

DEFAULT_MODEL = Path("/home/hjotha/models/Qwen3.5-4B-MTP-Q4_K_M.gguf")
DEFAULT_BINARY = Path("build-optimized/bin/llama-server")
REMOTE_TYPES = ("q4_0", "q5_0", "q6_0", "q8_0")
DRAFT_ACCEPTANCE_RE = re.compile(
    r"draft acceptance = (\S+) \(\s*(\d+) accepted / \s*(\d+) generated\), mean len =\s*(\S+)")
GRAPHS_REUSED_RE = re.compile(r"graphs reused =\s*(\d+)")
MTP_PLACEMENT_RE = re.compile(
    r"MTP K/V placement: local_attention=([^,\s]+), layers=([^,\s]+)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18093)
    parser.add_argument("--scenarios", default="M1,M2",
                        help="comma-separated: M1,M2")
    parser.add_argument("--remote-kv-types", default="q4_0",
                        help="remote Qx types for the target mixed layer: q4_0,q5_0,q6_0,q8_0")
    parser.add_argument("--ctx-size", type=int, default=8192,
                        help="M1 context (--ctx-size)")
    parser.add_argument("--prompt-tokens", type=int, default=4096,
                        help="frozen corpus size for POST /tokenize (>= max length + suffix)")
    parser.add_argument("--prompt-lengths", default="127,128,129,385",
                        help="M1 cold prefix token counts (KVarN/checkpoint boundaries)")
    parser.add_argument("--mandatory-reuse-lengths", default="385",
                        help="lengths whose FIRST suffix requires a positive cache hit; "
                             "385 is never waived. Other lengths are optional boundary "
                             "probes (reuse recorded, baseline may re-process)")
    parser.add_argument("--suffix-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--n-probs", type=int, default=3)
    parser.add_argument("--spec-draft-n-max", type=int, default=4,
                        help="M1/M2 local tier draft N (--spec-draft-n-max)")
    parser.add_argument("--spec-draft-n-max-l", type=int, default=2,
                        help="M2 long tier draft N (--spec-draft-n-max-l)")
    parser.add_argument("--local-ctx", type=int, default=512,
                        help="M2 MTP profile (--ctx-size-mtp)")
    parser.add_argument("--high-ctx", type=int, default=1024,
                        help="M2 LONG profile (--ctx-size); must exceed --local-ctx")
    parser.add_argument("--mtp-max-tokens", type=int, default=512)
    parser.add_argument("--gate-ctx", type=int, default=513)
    parser.add_argument("--prompt-short", type=int, default=300)
    parser.add_argument("--prompt-long", type=int, default=700)
    parser.add_argument("--n-predict", type=int, default=16)
    parser.add_argument("--handoff-evidence-markers", default="",
                        help="log markers proving explicit KVarN->Qx handoff success; "
                             "empty = acceptance stays pending_evidence")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--cuda-device", default="CUDA0")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--shutdown-timeout", type=float, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--telemetry-device", default="/sys/class/drm/card1/device")
    # Power controls: passthrough only, never sudo/hardware writes here.
    parser.add_argument("--apu-tdp", type=int, default=20)
    parser.add_argument("--amd-sclk-prefill", type=int, default=2700)
    parser.add_argument("--amd-sclk-decode", type=int, default=2700)
    parser.add_argument("--skip-model-hash", action="store_true")
    parser.add_argument("--skip-binary-hash", action="store_true")
    return parser


# ---------------------------------------------------------------------------
# Draft evidence: PRIMARY = response timings, SECONDARY = task log line
# ---------------------------------------------------------------------------

def validate_response_draft(timings: dict[str, Any], label: str) -> list[str]:
    """Per-request draft evidence from the response timings. The server only
    emits draft_n/draft_n_accepted when drafts were generated, so absent
    fields mean zero drafts for this request (fallback or fallback-like)."""
    failures: list[str] = []
    draft_n = timings.get("draft_n")
    if draft_n is None:
        return [f"{label}: timings lack draft_n (no drafts generated for this request)"]
    if not isinstance(draft_n, int) or draft_n <= 0:
        failures.append(f"{label}: draft_n invalid: {draft_n!r}")
    accepted = timings.get("draft_n_accepted")
    if accepted is None or not isinstance(accepted, int):
        failures.append(f"{label}: draft_n_accepted missing or non-integer: {accepted!r}")
    elif isinstance(draft_n, int) and (accepted < 0 or accepted > draft_n):
        failures.append(f"{label}: draft_n_accepted {accepted} outside [0, draft_n={draft_n}]")
    return failures


def validate_reprocessed(timings: dict[str, Any], label: str) -> dict[str, Any]:
    """cache_reprocessed_n is the number of prompt tokens actually processed
    (new suffix tokens plus unavoidable checkpoint rounding); it equals
    prompt_n on every observed baseline phase and is NOT a fallback signal.
    Recorded for review only."""
    return {"cache_reprocessed_n": timings.get("cache_reprocessed_n"),
            "cache_lcp_n": timings.get("cache_lcp_n"),
            "cache_source": timings.get("cache_source"),
            "cache_reason": timings.get("cache_reason")}


def validate_leg(timings: dict[str, Any], truncated: Any, expected_prompt: int,
                 expected_decode: int, label: str, cold: bool,
                 reuse_required: bool) -> tuple[list[str], bool]:
    """Exact count contract (cache_n + prompt_n == submitted, predicted_n ==
    requested, truncated false). Complete recomputation is rejected ONLY via
    cache_n == 0 where reuse is required; optional-reuse boundary legs record
    reuse_proven = (cache_n > 0) without failing (the working baseline
    re-processes the first suffix at 127/128/129 with cache_n=0)."""
    failures: list[str] = []
    cache_n = timings.get("cache_n")
    prompt_n = timings.get("prompt_n")
    predicted_n = timings.get("predicted_n")
    if not (isinstance(cache_n, int) and isinstance(prompt_n, int) and
            isinstance(predicted_n, int)):
        return [f"{label}: timings missing integer cache_n/prompt_n/predicted_n"], False
    if cache_n + prompt_n != expected_prompt:
        failures.append(f"{label}: cache_n({cache_n}) + prompt_n({prompt_n}) != "
                        f"submitted prompt ({expected_prompt})")
    if cold and cache_n != 0:
        failures.append(f"{label}: cold request must have cache_n=0, got {cache_n}")
    if not cold and reuse_required and cache_n <= 0:
        failures.append(f"{label}: reuse required but cache_n={cache_n} "
                        f"(complete recomputation passed as reuse)")
    if predicted_n != expected_decode:
        failures.append(f"{label}: predicted_n({predicted_n}) != requested decode "
                        f"({expected_decode})")
    if truncated is not False:
        failures.append(f"{label}: completion is missing truncated=false")
    return failures, cache_n > 0


def extract_draft_evidence(delta: str) -> dict[str, Any] | None:
    for line in delta.splitlines():
        match = DRAFT_ACCEPTANCE_RE.search(line)
        if match:
            return {
                "ratio": float(match.group(1)),
                "n_accepted": int(match.group(2)),
                "n_generated": int(match.group(3)),
                "mean_len": float(match.group(4)),
            }
    return None


def extract_graphs_reused(delta: str) -> int | None:
    for line in delta.splitlines():
        match = GRAPHS_REUSED_RE.search(line)
        if match:
            return int(match.group(1))
    return None


def reconcile_log_draft(delta: str, response_draft_n: Any) -> dict[str, Any]:
    """The task log line may flush asynchronously after the HTTP response;
    its absence in the current delta is recorded as pending, never a false
    fallback. When present it is reconciled against the response counters."""
    log_draft = extract_draft_evidence(delta)
    reconciled: dict[str, Any] = {
        "log_line_present": log_draft is not None,
        "log_generated": log_draft["n_generated"] if log_draft else None,
        "log_accepted": log_draft["n_accepted"] if log_draft else None,
        "matches_response": bool(log_draft and isinstance(response_draft_n, int) and
                                 log_draft["n_generated"] == response_draft_n),
    }
    if log_draft is None:
        reconciled["note"] = "log line may flush asynchronously; response timings are primary"
    return reconciled


def cadence_of(result: dict[str, Any]) -> tuple[Any, Any, Any]:
    timings = result.get("json", {}).get("timings") if isinstance(
        result.get("json"), dict) else {}
    return timings.get("cache_n"), timings.get("prompt_n"), timings.get("predicted_n")


# ---------------------------------------------------------------------------
# Acceptance decisions (scenario-level, unit-testable)
# ---------------------------------------------------------------------------

def m1_acceptance(failures: list[str], layout: list[dict[str, Any]],
                  expected_type: str, expected_layers: int = 1) -> tuple[str, str]:
    """Layout failures are ALWAYS merged into the final acceptance decision."""
    merged = list(failures)
    merged += r.validate_mixed_layout(layout, expected_layers, expected_type)
    return ("complete", "") if not merged else ("failed", "; ".join(merged))


def mtp_placement_failures(full_log: str) -> list[str]:
    """The mixed-target contract forces the independent MTP cache local CUDA:
    draft params clear remote attention and set layers=0
    (speculative.cpp:5403-5418), logged as local_attention=off, layers=0."""
    failures: list[str] = []
    found = False
    for line in full_log.splitlines():
        match = MTP_PLACEMENT_RE.search(line)
        if match:
            found = True
            if match.group(1) != "off":
                failures.append(f"MTP K/V placement local_attention is {match.group(1)}, "
                                f"expected off (MTP must stay on CUDA)")
            if match.group(2) != "0":
                failures.append(f"MTP K/V placement layers is {match.group(2)}, expected 0")
    if not found:
        failures.append("log lacks the 'MTP K/V placement:' line")
    return failures


def power_evidence_failures(args: argparse.Namespace, full_log: str) -> list[str]:
    """Exact markers from bench-radeon-q4.py: APU TDP active limit, and a
    single clock write when prefill == decode (the governor keeps the lock,
    logging only the first hardware write)."""
    failures: list[str] = []
    if args.apu_tdp is not None:
        expected = f"APU TDP: active limit {args.apu_tdp} W"
        if not any(expected in line for line in full_log.splitlines()):
            failures.append(f"log lacks '{expected}'")
    if args.amd_sclk_prefill is not None and args.amd_sclk_decode is not None:
        if args.amd_sclk_prefill == args.amd_sclk_decode:
            expected = f"locked {args.amd_sclk_prefill} MHz"
            if not any(expected in line for line in full_log.splitlines()):
                failures.append(f"log lacks '{expected}' (single write for equal clocks)")
        else:
            expected_prefill = f"-> prefill, locked {args.amd_sclk_prefill} MHz"
            expected_decode = f"-> decode, locked {args.amd_sclk_decode} MHz"
            if not any(expected_prefill in line for line in full_log.splitlines()):
                failures.append(f"log lacks '{expected_prefill}'")
            if not any(expected_decode in line for line in full_log.splitlines()):
                failures.append(f"log lacks '{expected_decode}'")
    elif args.amd_sclk_prefill is not None:
        expected = f"locked {args.amd_sclk_prefill} MHz"
        if not any(expected in line for line in full_log.splitlines()):
            failures.append(f"log lacks '{expected}'")
    elif args.amd_sclk_decode is not None:
        expected = f"-> decode, locked {args.amd_sclk_decode} MHz"
        if not any(expected in line for line in full_log.splitlines()):
            failures.append(f"log lacks '{expected}'")
    return failures


# ---------------------------------------------------------------------------
# Scenario M1: static mixed target with draft-mtp active
# ---------------------------------------------------------------------------

def m1_extra_args(args: argparse.Namespace, qx: str) -> list[str]:
    return [
        "--spec-type", "draft-mtp",
        "--spec-draft-type-k", "kvarn4", "--spec-draft-type-v", "kvarn4",
        "--spec-draft-n-max", str(args.spec_draft_n_max),
        "--remote-attn", "vulkan:0", "--remote-attn-layers", "1",
        "--remote-attn-cache-type-k", qx, "--remote-attn-cache-type-v", qx,
        "--cache-ram", "0", "--ctx-checkpoints", "1",
    ]


def scenario_m1(args: argparse.Namespace, output_dir: Path, run_meta: dict[str, Any],
                frozen: dict[str, Any] | None) -> dict[str, Any]:
    base_url = f"http://{args.host}:{args.port}"
    records_path = output_dir / "requests.jsonl"
    results_path = output_dir / "results.jsonl"
    lengths = sorted({int(item) for item in args.prompt_lengths.split(",") if item.strip()})
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("--prompt-lengths must be comma-separated positive integers")
    # Reuse policy: first-suffix hits are MANDATORY at the lengths in the
    # mandatory set (385 is never waived); any other length is an optional
    # boundary probe whose reuse is recorded, never required (the working
    # baseline re-processes the first suffix at 127/128/129 with cache_n=0).
    mandatory = {int(item) for item in args.mandatory_reuse_lengths.split(",") if item.strip()}
    mandatory |= ({385} if 385 in lengths else set())
    mandatory &= set(lengths)
    summary: dict[str, Any] = {"scenario": "M1", "runs": []}
    for qx in args.remote_kv_types:
        case_id = f"M1-{qx}"
        instance = r.ServerInstance(args, output_dir, case_id, m1_extra_args(args, qx))
        case_record: dict[str, Any] = {
            "case_id": case_id, "scenario": "M1", "state": "starting",
            "remote_kv_type": qx, "command_argv": instance.command,
            "command_shell": shlex.join(instance.command),
            "server_log": instance.log_path.name,
            "slot_save_path": instance.slot_save_path.name,
            "started_utc": r.utc_now(), "requests": [],
        }
        r.append_jsonl(output_dir / "cases.jsonl", case_record)
        print(f"[{case_id}] starting: {shlex.join(instance.command)}", flush=True)
        failures: list[str] = []
        try:
            instance.start()
            case_record["state"] = "ready"
            case_record["ready_utc"] = r.utc_now()
            r.append_jsonl(output_dir / "cases.jsonl", case_record)
            frozen = r.create_frozen_prompts(
                base_url, args, records_path, output_dir, case_id, frozen)
            run_meta["frozen_prompts"] = {"file": "prompts.json",
                                          "prefix_tokens": len(frozen["prefix_token_ids"]),
                                          "suffix_tokens": len(frozen["suffix_token_ids"])}
            r.write_json(output_dir / "run.json", run_meta)

            def completion(ids: list[int], phase: str) -> tuple[dict[str, Any], str, list[str]]:
                payload = {
                    "prompt": ids, "n_predict": args.decode_tokens, "id_slot": 0,
                    "cache_prompt": True, "return_tokens": True,
                    "temperature": 0, "seed": args.seed, "ignore_eos": True,
                    "n_probs": args.n_probs,
                }
                result = r.http_request(base_url, "POST", "/completion", payload,
                                        args.request_timeout, records_path,
                                        args.telemetry_device, case_id, phase,
                                        request_body_record={**payload, "prompt_sha256":
                                                             r.sha256_bytes(r.json_bytes(ids))})
                phase_failures: list[str] = []
                # Reject an HTTP 200 error body before touching timings.
                failed, detail = r.response_is_error(result)
                if failed:
                    return result, "", [f"{phase}: request failed: {detail}"]
                timings = result["json"].get("timings") if isinstance(
                    result["json"], dict) else {}
                phase_failures += validate_response_draft(timings, phase)
                return result, instance.read_log_delta(), phase_failures

            def erase(phase: str) -> None:
                r.http_request(base_url, "POST", "/slots/0?action=erase", None,
                               args.request_timeout, records_path, args.telemetry_device,
                               case_id, phase)

            for length in lengths:
                prefix = frozen["prefix_token_ids"][:length]
                extension = prefix + frozen["suffix_token_ids"]
                prefix_hash = r.sha256_bytes(r.json_bytes(prefix))
                extension_hash = r.sha256_bytes(r.json_bytes(extension))
                reuse_required = length in mandatory

                erase(f"M1-erase-before-{length}")
                cold, cold_delta, cold_extra = completion(prefix, f"M1-cold-{length}")
                cold_failures, _ = validate_leg(
                    cold["json"].get("timings") or {}, cold["json"].get("truncated"),
                    len(prefix), args.decode_tokens, f"M1-cold-{length}",
                    cold=True, reuse_required=False)
                cold_timings = cold["json"].get("timings") or {}
                cold_failures += cold_extra
                failures += cold_failures
                r.append_jsonl(results_path, {
                    "scenario": "M1", "case_id": case_id, "phase": f"M1-cold-{length}",
                    "prompt_tokens": len(prefix), "prompt_sha256": prefix_hash,
                    "timings": cold_timings, "content": cold["json"].get("content"),
                    "generated_tokens": cold["json"].get("tokens"),
                    "completion_probabilities": cold["json"].get("completion_probabilities"),
                    "cache_meta": validate_reprocessed(cold_timings, f"M1-cold-{length}"),
                    "draft_log": reconcile_log_draft(cold_delta, cold_timings.get("draft_n")),
                    "graphs_reused": extract_graphs_reused(cold_delta),
                    "validation_failures": cold_failures,
                })

                # Determinism pair: both extensions start from a freshly
                # rebuilt identical prefix (erase + cold rebuild). Cadence,
                # tokens and text must match EXACTLY.
                ext_results: list[dict[str, Any]] = []
                for leg in ("A", "B"):
                    erase(f"M1-erase-before-ext-{leg}-{length}")
                    rebuild, rebuild_delta, rebuild_extra = completion(
                        prefix, f"M1-rebuild-{leg}-{length}")
                    rebuild_failures, _ = validate_leg(
                        rebuild["json"].get("timings") or {},
                        rebuild["json"].get("truncated"),
                        len(prefix), args.decode_tokens, f"M1-rebuild-{leg}-{length}",
                        cold=True, reuse_required=False)
                    rebuild_failures += rebuild_extra
                    failures += rebuild_failures
                    r.append_jsonl(results_path, {
                        "scenario": "M1", "case_id": case_id,
                        "phase": f"M1-rebuild-{leg}-{length}",
                        "prompt_tokens": len(prefix), "prompt_sha256": prefix_hash,
                        "timings": rebuild["json"].get("timings") or {},
                        "content": rebuild["json"].get("content"),
                        "generated_tokens": rebuild["json"].get("tokens"),
                        "completion_probabilities": rebuild["json"].get("completion_probabilities"),
                        "cache_meta": validate_reprocessed(
                            rebuild["json"].get("timings") or {}, f"M1-rebuild-{leg}-{length}"),
                        "draft_log": reconcile_log_draft(
                            rebuild_delta, (rebuild["json"].get("timings") or {}).get("draft_n")),
                        "graphs_reused": extract_graphs_reused(rebuild_delta),
                        "validation_failures": rebuild_failures,
                    })
                    ext, ext_delta, ext_extra = completion(extension, f"M1-ext-{leg}-{length}")
                    ext_timings = ext["json"].get("timings") or {}
                    ext_failures, ext_reuse_proven = validate_leg(
                        ext_timings, ext["json"].get("truncated"),
                        len(extension), args.decode_tokens, f"M1-ext-{leg}-{length}",
                        cold=False, reuse_required=reuse_required)
                    ext_failures += ext_extra
                    failures += ext_failures
                    r.append_jsonl(results_path, {
                        "scenario": "M1", "case_id": case_id,
                        "phase": f"M1-ext-{leg}-{length}",
                        "prompt_tokens": len(extension), "prompt_sha256": extension_hash,
                        "timings": ext_timings, "content": ext["json"].get("content"),
                        "generated_tokens": ext["json"].get("tokens"),
                        "completion_probabilities": ext["json"].get("completion_probabilities"),
                        "cache_meta": validate_reprocessed(ext_timings, f"M1-ext-{leg}-{length}"),
                        "reuse_expected": reuse_required,
                        "reuse_proven": ext_reuse_proven,
                        "draft_log": reconcile_log_draft(ext_delta, ext_timings.get("draft_n")),
                        "graphs_reused": extract_graphs_reused(ext_delta),
                        "validation_failures": ext_failures,
                    })
                    ext_results.append(ext)
                    case_record["requests"].append({
                        "phase": f"M1-ext-{leg}-{length}",
                        "cache_n": ext_timings.get("cache_n"),
                        "prompt_n": ext_timings.get("prompt_n"),
                        "reuse_expected": reuse_required,
                        "reuse_proven": ext_reuse_proven,
                        "draft_n": ext_timings.get("draft_n"),
                        "draft_n_accepted": ext_timings.get("draft_n_accepted"),
                    })

                # Mandatory determinism: both legs started from the identical
                # rebuilt prefix, so cadence AND tokens AND text must match.
                determinism_failures: list[str] = []
                if cadence_of(ext_results[0]) != cadence_of(ext_results[1]):
                    determinism_failures.append(
                        f"M1-ext-{length}: cadence differs between identical rebuilt starts "
                        f"{cadence_of(ext_results[0])} vs {cadence_of(ext_results[1])}")
                if ext_results[0]["json"].get("tokens") != ext_results[1]["json"].get("tokens"):
                    determinism_failures.append(f"M1-ext-{length}: generated token ids differ")
                if ext_results[0]["json"].get("content") != ext_results[1]["json"].get("content"):
                    determinism_failures.append(f"M1-ext-{length}: generated text differs")
                failures += determinism_failures
                r.append_jsonl(results_path, {
                    "scenario": "M1", "case_id": case_id, "phase": f"M1-determinism-{length}",
                    "cadence_matches": cadence_of(ext_results[0]) == cadence_of(ext_results[1]),
                    "tokens_equal": ext_results[0]["json"].get("tokens") ==
                                    ext_results[1]["json"].get("tokens"),
                    "text_equal": ext_results[0]["json"].get("content") ==
                                  ext_results[1]["json"].get("content"),
                    "validation_failures": determinism_failures,
                })

                # Reuse-only repeat: a positive cache hit is REQUIRED for every
                # length (baseline 127->128, 128->128, 129->128, 385->384).
                # The cadence may legitimately differ from the previous leg
                # (later checkpoint); it is recorded, never an equality test.
                ext2, ext2_delta, ext2_extra = completion(extension, f"M1-repeat-{length}")
                ext2_timings = ext2["json"].get("timings") or {}
                ext2_failures, ext2_reuse_proven = validate_leg(
                    ext2_timings, ext2["json"].get("truncated"),
                    len(extension), args.decode_tokens, f"M1-repeat-{length}",
                    cold=False, reuse_required=True)
                ext2_failures += ext2_extra
                failures += ext2_failures
                r.append_jsonl(results_path, {
                    "scenario": "M1", "case_id": case_id, "phase": f"M1-repeat-{length}",
                    "prompt_tokens": len(extension), "prompt_sha256": extension_hash,
                    "timings": ext2_timings, "content": ext2["json"].get("content"),
                    "generated_tokens": ext2["json"].get("tokens"),
                    "completion_probabilities": ext2["json"].get("completion_probabilities"),
                    "cache_meta": validate_reprocessed(ext2_timings, f"M1-repeat-{length}"),
                    "reuse_expected": True,
                    "reuse_proven": ext2_reuse_proven,
                    "draft_log": reconcile_log_draft(ext2_delta, ext2_timings.get("draft_n")),
                    "graphs_reused": extract_graphs_reused(ext2_delta),
                    "cadence_vs_previous": cadence_of(ext2),
                    "reuse_only": True,
                    "validation_failures": ext2_failures,
                })
                case_record["requests"].append({
                    "phase": f"M1-repeat-{length}",
                    "cache_n": ext2_timings.get("cache_n"),
                    "prompt_n": ext2_timings.get("prompt_n"),
                    "reuse_expected": True,
                    "reuse_proven": ext2_reuse_proven,
                    "draft_n": ext2_timings.get("draft_n"),
                    "draft_n_accepted": ext2_timings.get("draft_n_accepted"),
                })
        except Exception as exc:
            case_record["state"] = "failed"
            case_record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{case_id}] failed: {case_record['error']}", flush=True)
        finally:
            case_record["server_exit_code"] = instance.stop()
            full_log = instance.log_path.read_text(encoding="utf-8", errors="replace")
            layout = r.parse_mixed_layout(full_log)
            failures += mtp_placement_failures(full_log)
            failures += power_evidence_failures(args, full_log)
            case_record["state"], case_record["error"] = m1_acceptance(failures, layout, qx)
            case_record["mixed_format_log_evidence"] = layout
            case_record["validation_failures"] = failures
            case_record["finished_utc"] = r.utc_now()
            r.append_jsonl(output_dir / "cases.jsonl", case_record)
            summary["runs"].append(case_record)
            print(f"[{case_id}] state={case_record['state']}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Scenario M2: adaptive MTP 512 -> LONG 1024 with MTP remaining active
# ---------------------------------------------------------------------------

def m2_extra_args(args: argparse.Namespace, qx: str) -> list[str]:
    # The base ServerInstance command already carries --ctx-size from the
    # namespace (set to high_ctx for M2); never duplicate it here.
    return [
        "--spec-type", "draft-mtp",
        "--ctx-size-mtp", str(args.local_ctx),
        "--mtp-max-tokens", str(args.mtp_max_tokens),
        "--cache-type-k-m", "kvarn4", "--cache-type-v-m", "kvarn4",
        "--cache-type-k-l", "kvarn4", "--cache-type-v-l", "kvarn4",
        "--spec-draft-type-k-m", "kvarn4", "--spec-draft-type-v-m", "kvarn4",
        "--spec-draft-type-k-l", "kvarn4", "--spec-draft-type-v-l", "kvarn4",
        "--spec-draft-n-max", str(args.spec_draft_n_max),
        "--spec-draft-n-max-l", str(args.spec_draft_n_max_l),
        "--remote-attn", "vulkan:0", "--remote-attn-layers", "1",
        "--remote-attn-min-context", str(args.gate_ctx),
        "--remote-attn-cache-type-k", qx, "--remote-attn-cache-type-v", qx,
        "--cache-ram", "256", "--ctx-checkpoints", "1",
    ]


def scenario_m2(args: argparse.Namespace, output_dir: Path, run_meta: dict[str, Any],
                frozen: dict[str, Any] | None) -> dict[str, Any]:
    base_url = f"http://{args.host}:{args.port}"
    records_path = output_dir / "requests.jsonl"
    results_path = output_dir / "results.jsonl"
    qx = args.remote_kv_types[0]
    case_id = "M2-adaptive-mtp"
    m2_args = argparse.Namespace(**{**vars(args), "ctx_size": args.high_ctx})
    instance = r.ServerInstance(m2_args, output_dir, case_id, m2_extra_args(args, qx))
    case_record: dict[str, Any] = {
        "case_id": case_id, "scenario": "M2", "state": "starting",
        "remote_kv_type": qx, "local_ctx": args.local_ctx, "high_ctx": args.high_ctx,
        "mtp_max_tokens": args.mtp_max_tokens, "gate_ctx": args.gate_ctx,
        "spec_draft_n_max": args.spec_draft_n_max, "spec_draft_n_max_l": args.spec_draft_n_max_l,
        "handoff_evidence_markers": args.handoff_evidence_markers,
        "command_argv": instance.command, "command_shell": shlex.join(instance.command),
        "server_log": instance.log_path.name,
        "slot_save_path": instance.slot_save_path.name,
        "started_utc": r.utc_now(), "requests": [],
    }
    r.append_jsonl(output_dir / "cases.jsonl", case_record)
    print(f"[{case_id}] starting: {shlex.join(instance.command)}", flush=True)
    failures: list[str] = []
    try:
        instance.start()
        case_record["state"] = "ready"
        case_record["ready_utc"] = r.utc_now()
        r.append_jsonl(output_dir / "cases.jsonl", case_record)
        frozen = r.create_frozen_prompts(
            base_url, args, records_path, output_dir, case_id, frozen)
        r.write_json(output_dir / "run.json", run_meta)

        short_ids = frozen["prefix_token_ids"][:args.prompt_short]
        long_ids = frozen["prefix_token_ids"][:args.prompt_long]
        short_hash = r.sha256_bytes(r.json_bytes(short_ids))
        long_hash = r.sha256_bytes(r.json_bytes(long_ids))

        def completion(ids: list[int], ids_hash: str, phase: str) -> tuple[dict[str, Any], str]:
            payload = {
                "prompt": ids, "n_predict": args.n_predict, "id_slot": 0,
                "cache_prompt": True, "return_tokens": True,
                "temperature": 0, "seed": args.seed, "ignore_eos": True,
                "n_probs": args.n_probs,
            }
            result = r.http_request(base_url, "POST", "/completion", payload,
                                    args.request_timeout, records_path, args.telemetry_device,
                                    case_id, phase,
                                    request_body_record={**payload, "prompt_sha256": ids_hash})
            return result, instance.read_log_delta()

        # Pre-gate: MTP profile (512 < gate 513), remote off, MTP active.
        short, short_delta = completion(short_ids, short_hash, "M2-short-cold")
        short_failures: list[str] = []
        failed, detail = r.response_is_error(short)
        if failed:
            short_failures.append(f"M2-short-cold: request failed: {detail}")
        else:
            short_timings = short["json"].get("timings") or {}
            short_leg_failures, _ = validate_leg(
                short_timings, short["json"].get("truncated"),
                len(short_ids), args.n_predict, "M2-short-cold",
                cold=True, reuse_required=False)
            short_failures += short_leg_failures
            short_failures += validate_response_draft(short_timings, "M2-short-cold")
        failures += short_failures
        r.append_jsonl(results_path, {
            "scenario": "M2", "case_id": case_id, "phase": "M2-short-cold",
            "prompt_tokens": len(short_ids), "prompt_sha256": short_hash,
            "timings": short["json"].get("timings"), "content": short["json"].get("content"),
            "generated_tokens": short["json"].get("tokens"),
            "completion_probabilities": short["json"].get("completion_probabilities"),
            "cache_meta": validate_reprocessed(short["json"].get("timings") or {},
                                               "M2-short-cold"),
            "draft_log": reconcile_log_draft(short_delta,
                                             (short["json"].get("timings") or {}).get("draft_n")),
            "validation_failures": short_failures,
        })
        pre_gate_log = short_delta + instance.read_log_delta()

        # Handoff: LONG profile (1024 >= gate 513), MTP still active. A
        # positive prefix hit at the gate is ALWAYS mandatory.
        handoff, handoff_delta = completion(long_ids, long_hash, "M2-handoff")
        handoff_failures: list[str] = []
        failed, detail = r.response_is_error(handoff)
        if failed:
            handoff_failures.append(f"M2-handoff: request failed: {detail}")
        else:
            handoff_timings = handoff["json"].get("timings") or {}
            handoff_leg_failures, _ = validate_leg(
                handoff_timings, handoff["json"].get("truncated"),
                len(long_ids), args.n_predict, "M2-handoff",
                cold=False, reuse_required=True)
            handoff_failures += handoff_leg_failures
            handoff_failures += r.validate_handoff_counts(
                handoff["json"], args.prompt_short, len(long_ids),
                expected_decode=args.n_predict)
        failures += handoff_failures
        r.append_jsonl(results_path, {
            "scenario": "M2", "case_id": case_id, "phase": "M2-handoff",
            "prompt_tokens": len(long_ids), "prompt_sha256": long_hash,
            "common_prefix": args.prompt_short, "timings": handoff["json"].get("timings"),
            "content": handoff["json"].get("content"),
            "generated_tokens": handoff["json"].get("tokens"),
            "completion_probabilities": handoff["json"].get("completion_probabilities"),
            "cache_meta": validate_reprocessed(handoff["json"].get("timings") or {},
                                               "M2-handoff"),
            "draft_log": reconcile_log_draft(handoff_delta,
                                             (handoff["json"].get("timings") or {}).get("draft_n")),
            "validation_failures": handoff_failures,
        })
        gate_log = handoff_delta + instance.read_log_delta()

        # Post-gate: draft generation must keep flowing after the handoff.
        long2, long2_delta = completion(long_ids, long_hash, "M2-long-continuation")
        long2_failures: list[str] = []
        failed, detail = r.response_is_error(long2)
        if failed:
            long2_failures.append(f"M2-long-continuation: request failed: {detail}")
        else:
            long2_timings = long2["json"].get("timings") or {}
            long2_leg_failures, _ = validate_leg(
                long2_timings, long2["json"].get("truncated"),
                len(long_ids), args.n_predict, "M2-long-continuation",
                cold=False, reuse_required=True)
            long2_failures += long2_leg_failures
            long2_failures += validate_response_draft(long2_timings, "M2-long-continuation")
        failures += long2_failures
        r.append_jsonl(results_path, {
            "scenario": "M2", "case_id": case_id, "phase": "M2-long-continuation",
            "prompt_tokens": len(long_ids), "prompt_sha256": long_hash,
            "timings": long2["json"].get("timings"), "content": long2["json"].get("content"),
            "generated_tokens": long2["json"].get("tokens"),
            "completion_probabilities": long2["json"].get("completion_probabilities"),
            "cache_meta": validate_reprocessed(long2["json"].get("timings") or {},
                                               "M2-long-continuation"),
            "draft_log": reconcile_log_draft(long2_delta,
                                             (long2["json"].get("timings") or {}).get("draft_n")),
            "validation_failures": long2_failures,
        })

        pre_gate_evidence = instance.log_evidence(pre_gate_log)
        gate_evidence = instance.log_evidence(gate_log)
        # Layout evidence scoped to the gate delta: no mixed allocation may
        # exist in the pre-gate delta, and the gate delta must carry the exact
        # Qx layout (the full log alone could hide the wrong phase).
        gate_layout = r.parse_mixed_layout(gate_log)
        pre_gate_has_qx = any("mixed kv layer=" in line.lower() for line in pre_gate_evidence)
        pre_gate_backend_off = any(
            "adaptive remote attention:" in line.lower() and "backend=off" in line.lower()
            for line in pre_gate_evidence)
        gate_backend_on = any(
            "adaptive remote attention:" in line.lower() and "backend=off" not in line.lower()
            for line in gate_evidence)
        gate_transition = any("adaptive context transition complete" in line.lower()
                              for line in gate_evidence)
        layout_failures = r.validate_mixed_layout(gate_layout, 1, qx)
        if not pre_gate_backend_off:
            failures.append("pre-gate log lacks 'adaptive remote attention: ... backend=off'")
        if pre_gate_has_qx:
            failures.append("pre-gate log contains a Qx 'mixed KV layer' allocation")
        if not gate_backend_on:
            failures.append("gate log lacks an enabled 'adaptive remote attention' line")
        if not gate_transition:
            failures.append("gate log lacks 'adaptive context transition complete'")
        failures += layout_failures

        markers = [marker.strip() for marker in args.handoff_evidence_markers.split(",")
                   if marker.strip()]
        handoff_evidence: dict[str, Any] = {"configured": bool(markers), "found": False,
                                            "matched_lines": []}
        if markers:
            handoff_evidence["matched_lines"] = [
                line.strip() for line in gate_log.splitlines()
                if any(marker.lower() in line.lower() for marker in markers)]
            handoff_evidence["found"] = bool(handoff_evidence["matched_lines"])
            if not handoff_evidence["found"]:
                failures.append(f"gate log lacks the configured handoff evidence markers: {markers}")
        case_record["handoff_evidence"] = handoff_evidence
        case_record["bootstrap_markers"] = [
            line.strip() for line in instance.log_path.read_text(encoding="utf-8",
                                                                 errors="replace").splitlines()
            if "mtp bootstrap" in line.lower()]

        case_record["requests"] = [
            {"phase": "M2-short-cold",
             "cache_n": (short["json"].get("timings") or {}).get("cache_n"),
             "prompt_n": (short["json"].get("timings") or {}).get("prompt_n"),
             "draft_n": (short["json"].get("timings") or {}).get("draft_n"),
             "draft_n_accepted": (short["json"].get("timings") or {}).get("draft_n_accepted")},
            {"phase": "M2-handoff",
             "cache_n": (handoff["json"].get("timings") or {}).get("cache_n"),
             "prompt_n": (handoff["json"].get("timings") or {}).get("prompt_n")},
            {"phase": "M2-long-continuation",
             "cache_n": (long2["json"].get("timings") or {}).get("cache_n"),
             "prompt_n": (long2["json"].get("timings") or {}).get("prompt_n"),
             "draft_n": (long2["json"].get("timings") or {}).get("draft_n"),
             "draft_n_accepted": (long2["json"].get("timings") or {}).get("draft_n_accepted")},
        ]
        case_record["pre_gate_evidence"] = pre_gate_evidence
        case_record["gate_evidence"] = gate_evidence
        case_record["gate_mixed_format_log_evidence"] = gate_layout
        case_record["validation_failures"] = failures
        case_record["state"] = r.acceptance_state(failures, handoff_evidence)
        if case_record["state"] == "failed":
            case_record["error"] = "; ".join(failures)
        elif case_record["state"] == "pending_evidence":
            case_record["error"] = ("acceptance pending: handoff conversion evidence markers "
                                    "were not configured (--handoff-evidence-markers)")
    except Exception as exc:
        case_record["state"] = "failed"
        case_record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[{case_id}] failed: {case_record['error']}", flush=True)
    finally:
        case_record["server_exit_code"] = instance.stop()
        full_log = instance.log_path.read_text(encoding="utf-8", errors="replace")
        failures += mtp_placement_failures(full_log)
        failures += power_evidence_failures(args, full_log)
        case_record["validation_failures"] = failures
        evidence = case_record.get("handoff_evidence")
        if case_record.get("server_exit_code") not in (None, 0):
            case_record["state"] = "failed"
            case_record["error"] = (case_record.get("error", "") + " " + "; ".join(failures) +
                                    f" (server exit={case_record['server_exit_code']})").strip()
        elif evidence is not None:
            case_record["state"] = r.acceptance_state(failures, evidence)
            if case_record["state"] == "failed":
                case_record["error"] = "; ".join(failures)
            elif case_record["state"] == "pending_evidence":
                case_record["error"] = ("acceptance pending: handoff conversion evidence markers "
                                        "were not configured (--handoff-evidence-markers)")
        else:
            case_record["state"] = "failed" if failures else "complete"
            if failures:
                case_record["error"] = "; ".join(failures)
        case_record["finished_utc"] = r.utc_now()
        r.append_jsonl(output_dir / "cases.jsonl", case_record)
        print(f"[{case_id}] state={case_record['state']}", flush=True)
    return {"scenario": "M2", "runs": [case_record]}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.scenarios = r.parse_names(args.scenarios, "--scenarios", {"M1", "M2"})
        args.remote_kv_types = r.parse_names(args.remote_kv_types, "--remote-kv-types",
                                             set(REMOTE_TYPES))
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be in 1..65535")
    if args.local_ctx >= args.high_ctx:
        parser.error("--local-ctx (MTP profile, --ctx-size-mtp) must be smaller than "
                     "--high-ctx (LONG profile, --ctx-size)")
    if not (args.local_ctx < args.gate_ctx <= args.high_ctx):
        parser.error("--gate-ctx must satisfy --local-ctx < gate <= --high-ctx")
    if args.prompt_short + args.n_predict > args.local_ctx:
        parser.error("--prompt-short + --n-predict must fit the MTP tier")
    if args.prompt_long + args.n_predict > args.high_ctx:
        parser.error("--prompt-long + --n-predict must fit the LONG tier")
    if args.spec_draft_n_max < 1 or args.spec_draft_n_max_l < 1:
        parser.error("draft N values must be >= 1")

    script_path = Path(__file__).resolve()
    repo = script_path.parents[1]
    binary = args.server_binary if args.server_binary.is_absolute() else repo / args.server_binary
    binary = binary.resolve()
    model = args.model.resolve()
    if not binary.is_file():
        parser.error(f"server binary not found: {binary}")
    if not model.is_file():
        parser.error(f"model not found: {model}")
    args.server_binary = binary
    args.model = model

    output_dir = args.output_dir or Path(
        "bench-results") / f"mixed-mtp-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    def port_free() -> None:
        try:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((args.host, args.port))
        except OSError as exc:
            raise RuntimeError(
                f"cannot bind isolated server port {args.host}:{args.port}: {exc}") from exc

    port_free()
    manifest = {
        "created_utc": r.utc_now(), "machine_hostname": platform.node(),
        "platform": platform.platform(), "cwd": str(Path.cwd()), "repo": str(repo),
        "git_commit": r.git_value(repo, "rev-parse", "HEAD"),
        "git_status_porcelain": r.git_value(repo, "status", "--short"),
        "server_binary": str(binary), "server_binary_size": binary.stat().st_size,
        "server_binary_sha256": None if args.skip_binary_hash else r.sha256_file(binary),
        "model": str(model), "model_size": model.stat().st_size,
        "model_sha256": None if args.skip_model_hash else r.sha256_file(model),
        "harness": str(script_path), "harness_sha256": r.sha256_file(script_path),
        "listen_host": args.host, "port": args.port,
        "configuration": r.configuration_from_args(args),
        "reuses": "tools/test-mixed-kv-transition.py helpers (frozen)",
        "power_controls": {
            "apu_tdp_w": args.apu_tdp, "amd_sclk_prefill_mhz": args.amd_sclk_prefill,
            "amd_sclk_decode_mhz": args.amd_sclk_decode,
            "applied_by_harness": False,
            "note": "passed through to server argv only; root-level controls are the parent's job",
        },
        "artifacts": ["run.json", "cases.jsonl", "results.jsonl", "requests.jsonl",
                      "prompts.json", "prompt-corpus.txt", "continuation-suffix.txt",
                      "server-<case>.log"],
    }
    r.write_json(output_dir / "run.json", manifest)
    print(f"Output: {output_dir}\nScenarios: {','.join(args.scenarios)}; "
          f"remote KV: {','.join(args.remote_kv_types)}", flush=True)

    frozen: dict[str, Any] | None = None
    summaries: list[dict[str, Any]] = []
    for scenario in args.scenarios:
        port_free()
        if scenario == "M1":
            summary = scenario_m1(args, output_dir, manifest, frozen)
        else:
            summary = scenario_m2(args, output_dir, manifest, frozen)
        summaries.append(summary)
        prompts_path = output_dir / "prompts.json"
        if prompts_path.is_file() and frozen is None:
            frozen = json.loads(prompts_path.read_text(encoding="utf-8"))

    runs = [run for summary in summaries for run in summary.get("runs", [])]
    failed = sum(1 for run in runs if run["state"] == "failed")
    pending = sum(1 for run in runs if run["state"] == "pending_evidence")
    r.write_json(output_dir / "summary.json", {
        "finished_utc": r.utc_now(), "scenario_count": len(args.scenarios),
        "failed_runs": failed, "pending_evidence_runs": pending,
        "note": "pending_evidence runs are NOT fully accepted: handoff evidence markers "
                "were not configured",
        "output_dir": str(output_dir),
    })
    print(f"Finished: {len(args.scenarios)} scenarios, {failed} failed, "
          f"{pending} pending_evidence; results={output_dir}", flush=True)
    return 1 if failed or pending else 0


if __name__ == "__main__":
    sys.exit(main())