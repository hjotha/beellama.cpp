#!/usr/bin/env python3
"""Mixed-KV acceptance runner: starts only its own llama-server instance.

Black-box acceptance for the upcoming mixed KV feature (local KVarN4 + remote
Q4/Q5/Q6/Q8 on Vulkan layers). Each scenario starts one isolated llama-server
in its own process group on an isolated port with its own --slot-save-path,
records the exact argv, prompts, responses and logs, validates token/byte
counts, and reports failures honestly. NumPy is required only by the optional
full-logit comparison; GPU work runs only in the explicitly launched server.

Scenarios:
  A  same mixed profile: cold base prefix, save base, continuation from the
     base, erase + restore the saved base, continuation again -> the cadence
     is REQUIRED to match and token/text equality is REQUIRED (determinism is
     a mandatory check, never skipped). A fresh same-profile server instance
     then restores the same snapshot file and continues, proving cross-context
     state usability. RAM prompt cache is disabled (--cache-ram 0) and
     --ctx-checkpoints 1 is explicit so the restore path is exercised.
  B  adaptive local -> mixed transition with low test thresholds; requires
     --spec-type draft-mtp (adaptive context), uniform KVarN local KV, one
     explicit remote layer. The LONG profile is --ctx-size (1024) and the MTP
     profile is --ctx-size-mtp (512) - common_context_adaptive_error rejects
     --ctx-size-mtp >= --ctx-size. The short prompt (300 + 16 = 316) selects
     the MTP profile and the log must show the remote backend off (no Qx
     allocation). The long prompt (700 + 16 = 716) selects the LONG profile
     and shares the 300-token common prefix; its cache must be preserved
     across the profile transition (cache_n > 0 and <= the common prefix,
     checkpoint rounding allowed) - recomputation is a FAILURE. The gate log
     must show the enabled backend, the transition/restore marker, and the
     exact per-layer Qx layout (unique layer count, formats, domain).
  C  a malformed/truncated saved snapshot must fail without accepting partial
     state; a subsequent valid restore must recover. Both error surfaces are
     accepted and recorded: HTTP status >= 400 and HTTP 200 with an error
     body.

Known server-side contracts this harness relies on (checked against the
source at the time of writing):
  - slot actions require --slot-save-path (isolated per instance);
  - POST /slots/:id?action=save|erase|restore answers synchronously; task
    errors surface as HTTP status = body["error"]["code"] with a
    {"error": {...}} body (server-context.cpp server_res_generator::error);
  - save -> {n_saved, n_written}, restore -> {n_restored, n_read},
    erase -> {n_erased}; n_written is the file size returned by
    llama_state_seq_save_file and n_read the size returned by
    llama_state_seq_load_file, so n_restored == n_saved and n_read ==
    n_written are the expected contract (a wrapper-byte distinction in the
    mixed codec would surface as a recorded failure);
  - --remote-attn-cache-type-k/v accept q4_0/q5_0/q6_0/q8_0 only
    (arg.cpp parse_remote_attn_standard_cache_type);
  - adaptive context requires --spec-type draft-mtp and logs
    "adaptive remote attention: ... backend=off|vulkan:0 ..." plus
    "adaptive context transition complete" and per-layer
    "mixed KV layer=<il> device=<dev> type_k=<tk> type_v=<tv> domain=<d>";
  - a speculative draft logs "draft acceptance = ..." per task
    (server-context.cpp:1809); its absence with --spec-type none confirms
    scenarios A/C are target-only even with a combined MTP model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import struct
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MODEL = Path("/home/hjotha/models/Qwen3.5-4B-MTP-Q4_K_M.gguf")
DEFAULT_BINARY = Path("build-optimized/bin/llama-server")
REMOTE_TYPES = ("q4_0", "q5_0", "q6_0", "q8_0")
NATURAL_LINES = (
    "The field team compared its written observations with the readings collected beside the river.",
    "Clear weather allowed the surveyors to follow the marked trail along the northern ridge.",
    "At midday, they checked the compass, updated the paper map, and recorded the direction of the wind.",
    "The archive keeps each measurement with its date so another researcher can repeat the journey.",
    "After the rain passed, the crew returned to the hillside and continued the careful inspection.",
)
INDEPENDENT_SUFFIX = (
    "\n\nUSER: Please analyze the following independent field report and summarize its "
    "observations, measurements, and uncertainty in a concise factual paragraph.\n\n"
    "A separate coastal station logged its own readings on the same day, twice at dawn and twice "
    "at dusk, and marked each entry with the local time before filing the sheet with the archive. "
    "The observer noted a brief change in the light while the instruments stayed stable and readable. "
    "These notes are independent of the earlier survey and should be summarized on their own."
)
MARKERS = (
    "vulkan-layers=", "mixed kv layer=", "adaptive remote attention:",
    "adaptive context transition complete", "adaptive streaming conversion:",
    "adaptive context task id", "kv buffer", "fallback", "flash_attn",
    "draft acceptance", "apu tdp: active", "amd graphics sclk",
)
MIXED_LAYER_RE = re.compile(
    r"mixed KV layer=(\d+) device=(\S+) type_k=(\S+) type_v=(\S+) domain=(\S+)")
LOADED_LIBRARY_HASHES: dict[str, str | None] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      default=json_default).encode("utf-8")


def json_default(obj: Any) -> Any:
    """Path-only JSON default: converts exactly pathlib.Path (preserving the
    resolved string) and raises for anything else unknown. Never silently
    str()s arbitrary objects."""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def configuration_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Explicit recursive Path normalization for manifest configuration:
    pathlib.Path values become their exact (already resolved) strings; every
    other value is preserved as-is, so unknown non-serializable objects still
    fail loudly at serialization time."""
    def normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value
    return normalize(vars(args))


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                default=json_default) + "\n")
        handle.flush()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def token_ids_sha256(tokens: list[int]) -> str:
    """Hash exact little-endian int32 token IDs for request binding."""
    if any(type(token) is not int or token < 0 or token > 0x7FFFFFFF for token in tokens):
        raise ValueError("token hash input must contain nonnegative int32 IDs")
    return sha256_bytes(b"".join(struct.pack("<i", token) for token in tokens))


def full_logit_build_id_type(value: str) -> str:
    """Accept only an operator-verified SHA256 bundle identity."""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise argparse.ArgumentTypeError("--full-logit-build-id must be exactly 64 hexadecimal characters")
    return value.lower()


def resolve_full_logit_capture_identity(args: argparse.Namespace,
                                        server_binary_sha256: str | None) -> tuple[str | None, str]:
    """Use the supplied bundle SHA or preserve the legacy binary-hash fallback."""
    if not getattr(args, "full_logit_capture", False):
        return None, "not_requested"
    supplied = getattr(args, "full_logit_build_id", None)
    if supplied:
        # Parser validation is the normal path; keep this helper fail-closed for
        # direct test/caller use too.
        if not re.fullmatch(r"[0-9a-fA-F]{64}", supplied):
            raise ValueError("full-logit build ID must be a 64-character SHA256")
        return supplied.lower(), "operator_verified_bundle_sha256"
    fallback = server_binary_sha256 or sha256_file(args.server_binary)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", fallback):
        raise ValueError("legacy server binary hash is not a valid SHA256")
    return fallback.lower(), "legacy_server_binary_sha256"


def validate_full_logit_capture_options(args: argparse.Namespace) -> None:
    if getattr(args, "full_logit_build_id", None) and not getattr(args, "full_logit_capture", False):
        raise ValueError("--full-logit-build-id requires --full-logit-capture")


def git_value(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def parse_names(value: str, name: str, allowed: set[str]) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(set(values)) != len(values) or any(item not in allowed for item in values):
        raise argparse.ArgumentTypeError(
            f"{name} must be unique values from {', '.join(sorted(allowed))}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18092)
    parser.add_argument("--scenarios", default="A,B,C",
                        help="comma-separated acceptance scenarios: A,B,C,D")
    parser.add_argument("--remote-kv-types", default="q4_0",
                        help="remote Qx types for the mixed profile: q4_0,q5_0,q6_0,q8_0")
    parser.add_argument("--remote-attn-layers", type=int, default=1,
                        help="explicit first N full-attention layers on the remote backend")
    parser.add_argument("--ctx-size", type=int, default=8192,
                        help="context for scenarios A and C")
    parser.add_argument("--prompt-tokens", type=int, default=2048,
                        help="scenario A cold base prefix token count including BOS")
    parser.add_argument("--suffix-tokens", type=int, default=32,
                        help="independent continuation tokens appended for cache-hit checks")
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--n-probs", type=int, default=3,
                        help="top-n token probabilities preserved for numerical review (0 = off)")
    parser.add_argument("--full-logit-capture", action="store_true",
                        help="scenario D: require the build-camp F4 capture hook, isolate captures "
                             "per process, and compare full-vocabulary D8a restore logits to a "
                             "fresh identical split baseline (NumPy required)")
    parser.add_argument("--full-logit-build-id", type=full_logit_build_id_type,
                        help="operator-verified SHA256 of the complete build bundle used by the "
                             "scenario D F4 capture hook (not the small loader executable hash)")
    # Scenario B adaptive thresholds.
    parser.add_argument("--local-ctx", type=int, default=512,
                        help="scenario B local tier context: this is the MTP profile "
                             "(--ctx-size-mtp), smaller than the LONG tier")
    parser.add_argument("--high-ctx", type=int, default=1024,
                        help="scenario B high tier context: this is the LONG profile "
                             "(--ctx-size); --ctx-size-mtp must not exceed it "
                             "(common_context_adaptive_error)")
    parser.add_argument("--mtp-max-tokens", type=int, default=512,
                        help="prompt plus output threshold between the MTP (local) and LONG (high) tiers")
    parser.add_argument("--gate-ctx", type=int, default=513,
                        help="--remote-attn-min-context gate between the two tiers")
    parser.add_argument("--prompt-short", type=int, default=300)
    parser.add_argument("--prompt-long", type=int, default=700)
    parser.add_argument("--n-predict", type=int, default=16)
    parser.add_argument("--handoff-evidence-markers", default="",
                        help="comma-separated log markers proving explicit KVarN->Qx conversion "
                             "success at the handoff; empty = check deferred (recorded, never "
                             "claimed as passed)")
    # Scenario D (§3.6): disk-only XXL snapshot and the position-split P.
    # P is available through public --kv-position-split-p and the legacy
    # LLAMA_KV_POSITION_SPLIT_P environment route. This harness intentionally
    # keeps its existing environment route; both P values and contexts are
    # validated in main() before any server is started.
    parser.add_argument("--split-p-local", type=int, default=256,
                        help="scenario D P for the LOCAL tier (multiple of 256)")
    parser.add_argument("--split-p-high", type=int, default=512,
                        help="scenario D P for the HIGH tier (multiple of 256)")
    parser.add_argument("--split-ctx-local", type=int, default=1024,
                        help="scenario D n_ctx for the local-tier legs; must exceed P and "
                             "cover prompt + suffix + output")
    parser.add_argument("--split-ctx-high", type=int, default=2048,
                        help="scenario D n_ctx for the high-tier legs; must exceed P and "
                             "cover prompt + suffix + output")
    parser.add_argument("--split-tokens-local", type=int, default=192,
                        help="scenario D imported prefix length for D1; kept below P so the "
                             "whole import lands in the local range (n_saved <= P)")
    parser.add_argument("--disk-tmpdir", type=Path, default=Path("/home/hjotha/f4-disk-tmp"),
                        help="scenario D TMPDIR: MUST be on a real disk-backed filesystem. The "
                             "mixed state spool refuses tmpfs/ramfs, which is exactly the "
                             "'only the snapshot on disk' requirement of §3.6")
    parser.add_argument("--split-full-layers", type=int, default=8,
                        help="number of FULL-attention layers the model has. The position-split "
                             "placement requires ALL of them on the remote backend (llama_context "
                             "rejects any smaller count), so the split legs pass exactly this "
                             "value to --remote-attn-layers")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--cuda-device", default="CUDA0")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--shutdown-timeout", type=float, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--telemetry-device", default="/sys/class/drm/card1/device",
                        help="AMD DRM device sysfs directory, or 'none'")
    # Power controls are only passed through to the server argv. This script
    # never calls sudo and never touches the governor itself; the parent run
    # applies root-level controls when needed.
    parser.add_argument("--apu-tdp", type=int, help="optional APU power limit in watts (passthrough)")
    parser.add_argument("--amd-sclk-prefill", type=int, help="optional prefill SCLK in MHz (passthrough)")
    parser.add_argument("--amd-sclk-decode", type=int, help="optional decode SCLK in MHz (passthrough)")
    parser.add_argument("--skip-model-hash", action="store_true")
    parser.add_argument("--skip-binary-hash", action="store_true")
    return parser


# ---------------------------------------------------------------------------
# HTTP + telemetry (bounded readiness; every response recorded; HTTP error
# bodies captured; transport failures raised as RuntimeError).
# ---------------------------------------------------------------------------

def telemetry(device: str) -> dict[str, Any]:
    if device == "none":
        return {"available": False, "reason": "disabled"}
    root = Path(device)
    fields = (
        "gpu_busy_percent", "mem_info_vram_used", "mem_info_vram_total",
        "mem_info_gtt_used", "mem_info_gtt_total",
        "pp_dpm_sclk", "pp_dpm_mclk", "pp_dpm_socclk",
    )
    values: dict[str, Any] = {"device": str(root), "available": root.exists()}
    for name in fields:
        path = root / name
        try:
            values[name] = path.read_text(encoding="utf-8").strip()
        except OSError:
            values[name] = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"MemAvailable", "SwapFree", "SwapTotal"}:
                values[f"system_{key}"] = value.strip()
    except OSError:
        pass
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=False)
            values["nvidia_smi_exit_code"] = result.returncode
            values["nvidia_smi_gpus"] = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError) as exc:
            values["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return values


def http_request(base_url: str, method: str, path: str, payload: Any | None,
                 timeout: float, record_path: Path, telemetry_device: str,
                 case_id: str | None, phase: str,
                 request_body_record: Any | None = None) -> dict[str, Any]:
    """One recorded HTTP request.

    Never raises on HTTP status codes: every response (including 4xx/5xx and
    200-with-error-body) is returned in the record. Transport-level failures
    (timeouts, refused connections) raise RuntimeError.
    """
    body = None if payload is None else json_bytes(payload)
    headers = {} if body is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(base_url + path, data=body, headers=headers, method=method)
    before = telemetry(telemetry_device)
    started = time.monotonic()
    status: int | None = None
    raw_text: str | None = None
    parsed: Any = None
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            raw_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw_text = exc.read().decode("utf-8", errors="replace")
        exc.close()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    after = telemetry(telemetry_device)
    if raw_text is not None:
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            pass
    row = {
        "time_utc": utc_now(), "case_id": case_id, "phase": phase,
        "method": method, "path": path, "http_status": status,
        "elapsed_seconds": elapsed,
        "request": request_body_record if request_body_record is not None else payload,
        "response_json": parsed, "response_raw": raw_text, "error": error,
        "telemetry_before": before, "telemetry_after": after,
    }
    append_jsonl(record_path, row)
    if error:
        raise RuntimeError(error)
    return {"json": parsed, "raw": raw_text, "status": status, "elapsed_seconds": elapsed}


_LOG_SEVERITY_RE = re.compile(r"^\s*\S+\s+([DIWEC])\s")


def refusal_reason_from_log(log_delta: str) -> str | None:
    """Returns the restore refusal reason, never a debug or info line.

    The reason has to come from a real error surface. llama's logger prints a
    per-batch debug line from use_position_split_attention() that mentions P
    and "split", so a substring match on those words silently reports a benign
    trace as the refusal reason and would make an above/below-P comparison
    meaningless. Severity is therefore parsed explicitly and "D " (debug) and
    "I " (info) are dropped before matching. A line naming the position-split
    boundary is preferred, and the raw message is returned, not a summary.
    """
    candidates: list[str] = []
    for raw in log_delta.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _LOG_SEVERITY_RE.match(line)
        if match and match.group(1) in ("D", "I"):
            continue
        lowered = line.lower()
        if ("error loading sequence state file" in lowered
                or "error:" in lowered
                or (match and match.group(1) == "E")):
            candidates.append(line)
    for line in candidates:
        if "position-split" in line or "position_split mismatch" in line:
            return line
    return candidates[0] if candidates else None


_PURE_KVARN_IMPORT_P_REFUSAL_RE = re.compile(
    r"pure KVarN import refused: snapshot extent (?P<extent>\d+) exceeds "
    r"min\(P=(?P<p>\d+), capacity=(?P<limit>\d+)\)", re.IGNORECASE)


def pure_kvarn_import_p_refusal(reason_line: str | None, expected_p: int) -> dict[str, Any] | None:
    """Attribute only the explicit pure-import extent refusal to this leg's P.

    `capacity` in this diagnostic is the printed min(P, cache capacity). It
    must equal P, in addition to the parsed P matching the expected P and the
    parsed snapshot extent exceeding P; a generic position-split refusal is
    not evidence for the boundary rule.
    """
    match = _PURE_KVARN_IMPORT_P_REFUSAL_RE.search(reason_line or "")
    if not match:
        return None
    extent, p, limit = (int(match.group(name)) for name in ("extent", "p", "limit"))
    return {
        "snapshot_extent": extent,
        "diagnostic_p": p,
        "diagnostic_limit": limit,
        "expected_p": expected_p,
        "extent_exceeds_p": extent > p,
        "p_matches_expected": p == expected_p,
        "limit_is_p": limit == p,
        "p_attributable": p == expected_p and extent > p and limit == p,
    }


_SPLIT_TO_PURE_IMPORT_P_REFUSAL_RE = re.compile(
    r"split-to-pure import refused: snapshot extent (?P<extent>\d+) exceeds "
    r"min\(P=(?P<p>\d+), capacity=(?P<capacity>\d+)\)", re.IGNORECASE)


def split_to_pure_import_p_refusal(reason_line: str | None, expected_source_p: int,
                                   sealed_occupancy: Any) -> dict[str, Any] | None:
    """Attribute an inverse split-to-PURE refusal to source P using native extent.

    n_saved is the sealed snapshot occupancy and only a lower bound on the
    native extent (last cell position + 1), so exact equality is not required.
    A capacity-limited refusal is rejected unless min(source P, destination
    capacity) is exactly the intended source P boundary.
    """
    match = _SPLIT_TO_PURE_IMPORT_P_REFUSAL_RE.search(reason_line or "")
    if not match:
        return None
    extent, source_p, capacity = (int(match.group(name))
                                  for name in ("extent", "p", "capacity"))
    sealed = sealed_occupancy if isinstance(sealed_occupancy, int) and not isinstance(sealed_occupancy, bool) else None
    effective_limit = min(source_p, capacity)
    return {
        "native_extent": extent,
        "sealed_occupancy_lower_bound": sealed,
        "source_p": source_p,
        "expected_source_p": expected_source_p,
        "destination_capacity": capacity,
        "effective_limit": effective_limit,
        "extent_exceeds_source_p": extent > source_p,
        "extent_covers_sealed_occupancy": sealed is not None and extent >= sealed,
        "source_p_matches_expected": source_p == expected_source_p,
        "limit_is_source_p": effective_limit == source_p,
        "p_attributable": source_p == expected_source_p and effective_limit == expected_source_p and
                          extent > source_p and sealed is not None and extent >= sealed,
    }


def validate_d5_destination_capacity(split_ctx_local: int, split_p_high: int) -> None:
    """D5 uses P_high in the local-context destination."""
    if split_ctx_local <= split_p_high:
        raise ValueError(
            "--split-ctx-local must exceed --split-p-high because D5 uses the "
            "high-P split at the local destination capacity "
            f"({split_ctx_local} <= {split_p_high})")


def validate_split_to_pure_extent(n_saved: Any, split_p_high: int,
                                  expect_import: bool) -> list[str]:
    """Validate sealed source occupancy, a lower bound on native snapshot extent, against P."""
    failures: list[str] = []
    if not isinstance(n_saved, int) or isinstance(n_saved, bool) or n_saved <= 0:
        return [f"invalid sealed snapshot occupancy n_saved={n_saved!r}"]
    if expect_import and n_saved > split_p_high:
        failures.append(f"sealed occupancy lower bound {n_saved} exceeds source P={split_p_high}; below-P import was not exercised")
    if not expect_import and n_saved <= split_p_high:
        failures.append(f"sealed occupancy lower bound {n_saved} does not exceed source P={split_p_high}; above-P rejection was not exercised")
    return failures

def response_is_error(result: dict[str, Any]) -> tuple[bool, str]:
    """Accepts both error surfaces: HTTP status >= 400 and a 2xx body that
    carries an "error" field (server_res_generator::error / task results)."""
    status = result.get("status")
    parsed = result.get("json")
    if status is not None and status >= 400:
        return True, f"http_status={status}"
    if isinstance(parsed, dict) and "error" in parsed:
        detail = parsed.get("error")
        if isinstance(detail, dict):
            return True, f"error_body_code={detail.get('code')}: {detail.get('message')}"
        return True, f"error_body={json.dumps(detail)}"
    return False, ""


def wait_ready(proc: subprocess.Popen[bytes], base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not answer /health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before readiness (exit={proc.returncode})")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                health = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and isinstance(health, dict) and health.get("status") in ("ok", "no slot available"):
                    return
                last_error = f"health status={response.status}: {health}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.5)
    raise TimeoutError(f"server readiness timeout after {timeout:g}s: {last_error}")


def _surviving_group_members(pgid: int) -> list[int]:
    """Pids (excluding the reaped leader) still running in process group pgid.

    Reads /proc/<pid>/stat, splitting after the comm field (which may contain
    spaces/parentheses), where state(3) ppid(4) pgrp(5) follow the closing
    parenthesis.
    """
    members: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return members
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == pgid:
            continue
        try:
            data = Path(f"/proc/{entry}/stat").read_bytes()
            rest = data[data.rfind(b")") + 2:].split()
            if len(rest) >= 3 and rest[2].isdigit() and int(rest[2]) == pgid:
                members.append(pid)
        except (OSError, ValueError):
            continue
    return members


def stop_server(proc: subprocess.Popen[bytes], timeout: float) -> None:
    """Terminate only this server's process group, never a foreign group.

    If the parent exited while descendants in the same group still run (for
    example a natural parent death with a surviving grandchild), the group is
    scanned and the remaining members are SIGKILLed. Only pgid == proc.pid is
    ever signalled.
    """
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    for _ in range(3):
        members = _surviving_group_members(proc.pid)
        if not members:
            break
        for pid in members:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.2)


# ---------------------------------------------------------------------------
# Prompt material (frozen ids and prefix bytes across every scenario/mode).
# ---------------------------------------------------------------------------

def make_corpus(min_tokens: int) -> str:
    target_chars = min_tokens * 7
    lines: list[str] = []
    index = 0
    character_count = 0
    while character_count < target_chars:
        base = NATURAL_LINES[index % len(NATURAL_LINES)]
        line = f"{base} The entry number {index + 1} is filed with the day's notes."
        lines.append(line)
        character_count += len(line) + 1
        index += 1
    return "\n".join(lines)


def required_frozen_prefix_tokens(args: argparse.Namespace) -> int:
    """Return the corpus token count needed by selected scenarios and suffixes."""
    prompt_max = max(args.prompt_tokens, args.prompt_short, args.prompt_long)
    needs = [prompt_max + args.suffix_tokens]
    scenarios = getattr(args, "scenarios", ())
    if isinstance(scenarios, str):
        scenarios = [part.strip() for part in scenarios.split(",") if part.strip()]
    if "D" in scenarios:
        # D0b/D8b and D3/D4b seal above P; D5 builds beyond local capacity.
        needs.extend((args.split_p_high + 256,
                      args.split_ctx_local + 256,
                      args.split_tokens_local + 256,
                      args.split_tokens_local + args.suffix_tokens))
    return max(needs)


def frozen_prefix_ids(frozen: dict[str, Any], count: int, label: str) -> list[int]:
    tokens = frozen.get("prefix_token_ids")
    if not isinstance(tokens, list) or len(tokens) < count:
        have = len(tokens) if isinstance(tokens, list) else 0
        raise RuntimeError(f"{label}: frozen prefix has {have} IDs, requires {count}")
    return tokens[:count]


def create_frozen_prompts(base_url: str, args: argparse.Namespace, record_path: Path,
                          output_dir: Path, case_id: str,
                          existing: dict[str, Any] | None) -> dict[str, Any]:
    """Tokenize the corpus once and freeze the ids. Subsequent instances must
    produce the identical ids (tokenizer identity check) so every mode shares
    the exact same prefix bytes and raw token ids."""
    prompts_path = output_dir / "prompts.json"
    largest = required_frozen_prefix_tokens(args)
    corpus = make_corpus(largest)
    for attempt in range(3):
        result = http_request(
            base_url, "POST", "/tokenize",
            {"content": corpus, "add_special": True, "parse_special": False},
            args.request_timeout, record_path, args.telemetry_device,
            case_id, f"tokenize-corpus-{attempt + 1}",
        )
        tokens = result["json"].get("tokens")
        if isinstance(tokens, list) and len(tokens) >= largest and all(
                isinstance(token, int) for token in tokens):
            break
        corpus += "\n" + corpus
    else:
        raise RuntimeError(f"tokenization did not produce {largest} token IDs")
    suffix_result = http_request(
        base_url, "POST", "/tokenize",
        {"content": INDEPENDENT_SUFFIX, "add_special": False, "parse_special": False},
        args.request_timeout, record_path, args.telemetry_device,
        case_id, "tokenize-independent-suffix",
    )
    suffix_ids = suffix_result["json"].get("tokens")
    if not isinstance(suffix_ids, list) or len(suffix_ids) < args.suffix_tokens or not all(
            isinstance(token, int) for token in suffix_ids):
        raise RuntimeError(f"independent suffix did not tokenize to {args.suffix_tokens} IDs")
    suffix_ids = suffix_ids[:args.suffix_tokens]
    frozen = {
        "source": "natural text sent to this model's POST /tokenize",
        "add_special": True, "parse_special": False,
        "token_count_including_bos": len(tokens),
        "prefix_token_ids": tokens,
        "prefix_bytes_sha256": sha256_bytes(json_bytes(tokens)),
        "suffix_token_ids": suffix_ids,
        "suffix_bytes_sha256": sha256_bytes(json_bytes(suffix_ids)),
        "frozen_utc": utc_now(),
    }
    if existing is not None:
        if existing["prefix_token_ids"] != tokens or existing["suffix_token_ids"] != suffix_ids:
            raise RuntimeError("tokenizer identity changed between server instances; "
                               "prefix ids are not frozen across modes")
        return existing
    (output_dir / "prompt-corpus.txt").write_text(corpus, encoding="utf-8")
    (output_dir / "continuation-suffix.txt").write_text(INDEPENDENT_SUFFIX, encoding="utf-8")
    write_json(prompts_path, frozen)
    return frozen


# ---------------------------------------------------------------------------
# Validation helpers (also exercised by tests/test-mixed-kv-transition-harness.py)
# ---------------------------------------------------------------------------

def validate_completion_counts(body: dict[str, Any], expected_prompt: int,
                               cold: bool, expected_decode: int) -> list[str]:
    failures: list[str] = []
    timings = body.get("timings") if isinstance(body.get("timings"), dict) else {}
    cache_n = timings.get("cache_n")
    prompt_n = timings.get("prompt_n")
    predicted_n = timings.get("predicted_n")
    if not (isinstance(cache_n, int) and isinstance(prompt_n, int) and isinstance(predicted_n, int)):
        return ["timings missing integer cache_n/prompt_n/predicted_n"]
    if cache_n + prompt_n != expected_prompt:
        failures.append(f"cache_n({cache_n}) + prompt_n({prompt_n}) != submitted prompt ({expected_prompt})")
    if cold and cache_n != 0:
        failures.append(f"cold request must have cache_n=0, got {cache_n}")
    if not cold and cache_n <= 0:
        failures.append(f"cache-hit request must have cache_n>0, got {cache_n}")
    if predicted_n != expected_decode:
        failures.append(f"predicted_n({predicted_n}) != requested decode ({expected_decode})")
    if body.get("truncated") is not False:
        failures.append("completion is missing truncated=false")
    return failures


def validate_handoff_counts(body: dict[str, Any], common_prefix: int,
                            total_prompt: int, expected_decode: int) -> list[str]:
    """The gate-crossing prompt must REUSE the common prefix state, not
    recompute it. cache_n > 0 is mandatory; cache_n may be smaller than the
    common prefix because of checkpoint rounding (cache_n <= common_prefix)."""
    failures: list[str] = []
    timings = body.get("timings") if isinstance(body.get("timings"), dict) else {}
    cache_n = timings.get("cache_n")
    prompt_n = timings.get("prompt_n")
    predicted_n = timings.get("predicted_n")
    if not (isinstance(cache_n, int) and isinstance(prompt_n, int) and isinstance(predicted_n, int)):
        return ["timings missing integer cache_n/prompt_n/predicted_n"]
    if cache_n <= 0:
        failures.append(f"handoff must reuse the prefix (cache_n>0), got cache_n={cache_n}")
    if cache_n > common_prefix:
        failures.append(f"handoff cache_n({cache_n}) exceeds the common prefix ({common_prefix})")
    if cache_n + prompt_n != total_prompt:
        failures.append(f"cache_n({cache_n}) + prompt_n({prompt_n}) != submitted prompt ({total_prompt})")
    if predicted_n != expected_decode:
        failures.append(f"predicted_n({predicted_n}) != requested decode ({expected_decode})")
    if body.get("truncated") is not False:
        failures.append("completion is missing truncated=false")
    return failures


def parse_slot_counts(result: dict[str, Any], action: str) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    parsed = result.get("json")
    if not isinstance(parsed, dict):
        return {}, [f"{action}: response is not a JSON object"]
    failed, detail = response_is_error(result)
    if failed:
        return parsed, [f"{action} failed: {detail}"]
    counts: dict[str, Any] = {}
    if action == "save":
        for key in ("n_saved", "n_written"):
            if isinstance(parsed.get(key), int) and parsed[key] > 0:
                counts[key] = parsed[key]
            else:
                failures.append(f"save missing positive {key}")
    elif action == "restore":
        for key in ("n_restored", "n_read"):
            if isinstance(parsed.get(key), int) and parsed[key] > 0:
                counts[key] = parsed[key]
            else:
                failures.append(f"restore missing positive {key}")
    elif action == "erase":
        if isinstance(parsed.get("n_erased"), int) and parsed["n_erased"] >= 0:
            counts["n_erased"] = parsed["n_erased"]
        else:
            failures.append("erase missing n_erased")
    return counts, failures


def validate_save_restore_parity(save_counts: dict[str, Any],
                                 restore_counts: dict[str, Any]) -> list[str]:
    """n_restored == n_saved and n_read == n_written are the expected contract
    (both count the snapshot file size; a wrapper-byte distinction in the
    mixed codec surfaces here as a recorded failure)."""
    failures: list[str] = []
    if restore_counts.get("n_restored") != save_counts.get("n_saved"):
        failures.append(f"n_restored({restore_counts.get('n_restored')}) != "
                        f"n_saved({save_counts.get('n_saved')})")
    if restore_counts.get("n_read") != save_counts.get("n_written"):
        failures.append(f"n_read({restore_counts.get('n_read')}) != "
                        f"n_written({save_counts.get('n_written')})")
    return failures


def parse_mixed_layout(log_text: str) -> list[dict[str, Any]]:
    return [
        {"layer": int(layer), "device": device, "type_k": type_k,
         "type_v": type_v, "domain": domain}
        for layer, device, type_k, type_v, domain in MIXED_LAYER_RE.findall(log_text)
    ]


def validate_mixed_layout(layout: list[dict[str, Any]], expected_layers: int,
                          expected_type: str,
                          expected_domain: str = "standard-hadamard") -> list[str]:
    """Exact per-layer layout: unique layer count, device, K/V formats, and
    the domain must all match precisely. The domain is the KVarN rotated
    representation logged as 'standard-hadamard' (llama-kv-cache-kvarn.cpp:
    domain naming); any other value is a layout failure."""
    failures: list[str] = []
    layers = {entry["layer"] for entry in layout}
    if len(layers) != expected_layers:
        failures.append(f"mixed layout has {len(layers)} unique layers, expected {expected_layers}")
    for entry in layout:
        if entry["device"] != "Vulkan0":
            failures.append(f"layer {entry['layer']} device is {entry['device']}, expected Vulkan0")
        if entry["type_k"] != expected_type or entry["type_v"] != expected_type:
            failures.append(f"layer {entry['layer']} types {entry['type_k']}/{entry['type_v']}, "
                            f"expected {expected_type}")
        if entry["domain"] != expected_domain:
            failures.append(f"layer {entry['layer']} domain is {entry['domain']}, "
                            f"expected {expected_domain}")
    return failures


def acceptance_state(validation_failures: list[str],
                     handoff_evidence: dict[str, Any] | None = None) -> str:
    """Scenario acceptance state. Validation failures fail the run. Without
    the configured handoff conversion evidence the run is only
    'pending_evidence', never fully accepted; with the evidence configured
    and found it is 'complete'."""
    if validation_failures:
        return "failed"
    if handoff_evidence is not None and not handoff_evidence.get("configured", False):
        return "pending_evidence"
    if handoff_evidence is not None and not handoff_evidence.get("found", False):
        return "failed"
    return "complete"


def target_only_violations(log_text: str) -> list[str]:
    """A speculative draft logs 'draft acceptance = ...' after every task.
    With --spec-type none it must never appear (target-only confirmation even
    with a combined MTP model)."""
    return [line.strip() for line in log_text.splitlines()
            if "draft acceptance" in line.lower()]


def deterministic_match(a: Any, b: Any) -> bool:
    return isinstance(a, list) and isinstance(b, list) and a == b


# ---------------------------------------------------------------------------
# §3.6 evidence helpers: recurrent state / next logits, and payload integrity
# ---------------------------------------------------------------------------

# Mixed state v2 envelope, little-endian. Field offsets from the framing in
# src/llama-kv-mixed-state.cpp and the header notes in its .h: the manifest and
# the layer descriptors are walked to find a payload byte that the reader
# actually covers with its per-payload XXH64 check
# (llama-kv-mixed-state.cpp:885, fault() throwing at :84-85). Corrupting
# anything OUTSIDE those regions proves nothing about integrity, so
# locate_checksummed_payload_byte refuses to guess.
MIXED_STATE_MAGIC_V1 = b"KMS1"     # KMS_MAGIC 0x31534D4B, little-endian
MIXED_STATE_MAGIC_V2 = b"KMS2"     # KMS2_MAGIC 0x32534D4B, little-endian
# LLAMA_STATE_SEQ_MAGIC = LLAMA_FILE_MAGIC_GGSQ = 0x67677371u, stored
# little-endian, so the four bytes on disk are 71 73 67 67 ("qsgg").
# include/llama.h:43,48.
STATE_SEQ_MAGIC = bytes((0x67677371 >> (8 * i)) & 0xFF for i in range(4))
_MIXED_HEADER_BYTES = 64


def _u32(data: bytes, off: int) -> int:
    return int.from_bytes(data[off:off + 4], "little")


def _u64(data: bytes, off: int) -> int:
    return int.from_bytes(data[off:off + 8], "little")


def locate_checksummed_payload_byte(path: Path) -> tuple[int, int, int, str] | None:
    """Return (offset, layer_id, range_id, region) for a byte that provably
    lies inside a region the reader RECOGNISES and validates: either a payload
    guarded by the mixed envelope's per-payload XXH64, or the recognized state
    frame magic. Returns None when the framing is unknown -- the caller then
    records the leg as not_executed instead of corrupting a byte at random.

    A blind len//2 flip is NOT evidence: it can land in a padding or an
    unverified region and then be accepted without proving anything about
    integrity. That is exactly what the first D6 run did."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 16:
        return None
    magic = data[:4]
    # Mixed state envelope: locate the first checksummed payload. The reader
    # verifies each descriptor's payload against its stored XXH64 and throws
    # (llama-kv-mixed-state.cpp:885 with fault() -> throw at :84-85).
    if magic in (MIXED_STATE_MAGIC_V1, MIXED_STATE_MAGIC_V2):
        try:
            manifest_end = _MIXED_HEADER_BYTES + _u64(data, 16)
            n_layers = _u32(data, 8)
            if not (0 < n_layers <= 4096) or manifest_end >= len(data):
                return None
            desc = manifest_end
            if desc + 48 > len(data):
                return None
            layer_id = _u32(data, desc)
            range_id = _u32(data, desc + 4)
            payload_bytes = _u64(data, desc + 24)
            if not (0 < payload_bytes <= len(data)):
                return None
            payload_base = len(data) - payload_bytes
            if payload_base <= manifest_end:
                return None
            return payload_base, layer_id, range_id, "mixed_payload_checksummed"
        except (IndexError, ValueError):
            return None
    # Recognized state frame: the magic itself is a validated header field, so
    # byte 2 is provably inside it and any change must be rejected.
    if magic == STATE_SEQ_MAGIC:
        return 2, -1, -1, "state_seq_header_magic"
    return None


def compare_completion_probabilities(left: Any, right: Any,
                                    tol: float = 1e-9) -> bool:
    """Compare the server's per-token NEXT-token distributions (id + logprob +
    top_logprobs). This is the §3.6 "next logits" evidence: a mutated KV or
    recurrent state moves these values, so identical distributions prove the
    destination state was not disturbed. Text equality alone is NOT accepted
    as evidence anywhere in scenario D."""
    if not isinstance(left, list) or not isinstance(right, list):
        return False
    if len(left) != len(right) or not left:
        return False
    for a, b in zip(left, right):
        if not isinstance(a, dict) or not isinstance(b, dict):
            return False
        if a.get("id") != b.get("id"):
            return False
        la, lb = a.get("logprob"), b.get("logprob")
        if not (isinstance(la, (int, float)) and isinstance(lb, (int, float))):
            return False
        if abs(float(la) - float(lb)) > tol:
            return False
        ta, tb = a.get("top_logprobs"), b.get("top_logprobs")
        if isinstance(ta, list) and isinstance(tb, list):
            if len(ta) != len(tb):
                return False
            for x, y in zip(ta, tb):
                if not isinstance(x, dict) or not isinstance(y, dict):
                    return False
                if x.get("id") != y.get("id"):
                    return False
                xl, yl = x.get("logprob"), y.get("logprob")
                if not (isinstance(xl, (int, float)) and isinstance(yl, (int, float))):
                    return False
                if abs(float(xl) - float(yl)) > tol:
                    return False
    return True


# ---------------------------------------------------------------------------
# Full-vocabulary capture bindings (optional diagnostic build only)
# ---------------------------------------------------------------------------

def full_logit_checkpoints(args: argparse.Namespace, ctx: int) -> list[int]:
    """Bounded hook filter, never a 131072-item environment variable.

    Small contexts can capture every actual decode checkpoint. Large contexts
    use D's request lengths plus at most 32 generated checkpoints per request.
    This is a capture plan, not evidence that a checkpoint executed.
    """
    if ctx <= 4096:
        return list(range(1, ctx + 1))
    bases = {args.split_tokens_local, max(args.split_tokens_local, 256),
             max(args.split_tokens_local, args.split_p_high + 256),
             max(args.split_tokens_local + 256, args.split_p_high + 256),
             max(args.split_tokens_local + 256, args.split_ctx_local + 256)}
    bases |= {n + args.suffix_tokens for n in tuple(bases)}
    width = min(max(args.decode_tokens, 1), 32)
    return sorted({n + delta for n in bases for delta in range(width)
                   if 0 < n + delta <= ctx})


def compare_full_logit_checkpoint(left_dir: Path, left_phase: str,
                                 right_dir: Path, right_phase: str,
                                 tokens_n: int, run_id: str,
                                 expected_tokens: list[int],
                                 *, layout_relation: str = "same",
                                 target_cache_source: str = "disk",
                                 baseline_cache_source_expected: str | None = None) -> dict[str, Any]:
    """Compare actual post-decode captures after exact semantic binding.

    Same-layout split baseline versus imported split is restore parity. PURE
    versus split is allowed only when the caller separately proves n<=P. The
    optional baseline_cache_source_expected permits a direct-restore control
    where both arms must identify their cache source as live.
    """
    result: dict[str, Any] = {"tokens_n": tokens_n, "state": "failed",
                              "left_phase": left_phase, "right_phase": right_phase,
                              "layout_relation": layout_relation}
    try:
        import numpy as np
        if len(expected_tokens) != tokens_n or any(type(token) is not int for token in expected_tokens):
            raise ValueError("expected checkpoint requires exact integer token IDs")

        def load(directory: Path, phase: str) -> tuple[dict[str, Any], Any, Any]:
            stem = directory / f"cap.{phase}.n{tokens_n}"
            meta = json.loads(Path(str(stem) + ".json").read_text())
            logits_path = Path(str(stem) + ".logits")
            token_path = Path(str(stem) + ".tokens")
            count = meta.get("logits_count")
            if not isinstance(count, int) or count <= 0 or count != meta.get("n_vocab"):
                raise ValueError(f"{phase}: invalid full-vocabulary count")
            if logits_path.stat().st_size != count * 4 or token_path.stat().st_size != tokens_n * 4:
                raise ValueError(f"{phase}: truncated or extra capture bytes")
            logits = np.fromfile(logits_path, dtype="<f4").astype(np.float64)
            tokens = np.fromfile(token_path, dtype="<i4")
            if meta.get("finite") is not True or not np.isfinite(logits).all():
                raise ValueError(f"{phase}: nonfinite logits")
            if meta.get("phase") != phase or meta.get("run_id") != run_id:
                raise ValueError(f"{phase}: capture phase/campaign mismatch")
            if meta.get("tokens_n") != tokens_n or not np.array_equal(tokens, expected_tokens):
                raise ValueError(f"{phase}: evaluated tokens differ from submitted/returned token IDs")
            result[f"{phase}_capture"] = {
                "metadata": str(stem) + ".json", "logits_sha256": sha256_file(logits_path),
                "tokens_sha256": sha256_file(token_path), "cache_source": meta.get("cache_source"),
                "cache_reason": meta.get("cache_reason"), "attempt": meta.get("attempt")}
            return meta, logits, tokens

        a, reference, ta = load(left_dir, left_phase)
        b, actual, tb = load(right_dir, right_phase)
        for key in ("build_id", "model_fingerprint", "config_core", "config_hash", "tokens_n",
                    "tokens_seq_hash", "input_pos", "output_pos", "n_ctx", "n_vocab", "logits_count", "pos_max"):
            if a.get(key) in (None, "") or a.get(key) != b.get(key):
                raise ValueError(f"binding mismatch for {key}: {a.get(key)!r} != {b.get(key)!r}")
        if a["input_pos"] != tokens_n-1 or a["output_pos"] != tokens_n-1 or a["pos_max"] != tokens_n-1:
            raise ValueError("capture position is not the stated post-decode checkpoint")
        if left_phase == right_phase or not np.array_equal(ta, tb) or reference.shape != actual.shape:
            raise ValueError("phase/token/vector binding mismatch")
        if layout_relation not in ("same", "different"):
            raise ValueError("unknown layout comparison contract")
        if not a.get("layout_hash") or not b.get("layout_hash"):
            raise ValueError("missing layout binding")
        result["left_layout_hash"] = a["layout_hash"]
        result["right_layout_hash"] = b["layout_hash"]
        if (a["layout_hash"] == b["layout_hash"]) != (layout_relation == "same"):
            raise ValueError("layout relation differs from requested comparison contract")
        if b.get("cache_source") != target_cache_source and target_cache_source == "disk" and baseline_cache_source_expected is None:
            raise ValueError("target is not required disk import, or baseline also claims disk reuse")
        if b.get("cache_source") != target_cache_source:
            raise ValueError(f"target cache source is {b.get('cache_source')!r}, expected {target_cache_source!r}")
        if baseline_cache_source_expected is None:
            if a.get("cache_source") == target_cache_source:
                raise ValueError("baseline also claims target cache reuse")
        elif a.get("cache_source") != baseline_cache_source_expected:
            raise ValueError(f"baseline cache source is {a.get('cache_source')!r}, expected {baseline_cache_source_expected!r}")
        ref_logz = float(np.max(reference) + np.log(np.exp(reference-np.max(reference)).sum(dtype=np.float64)))
        got_logz = float(np.max(actual) + np.log(np.exp(actual-np.max(actual)).sum(dtype=np.float64)))
        reference_logp = reference-ref_logz
        actual_logp = actual-got_logz
        kl = float(np.sum(np.exp(reference_logp) * (reference_logp-actual_logp), dtype=np.float64))
        delta = actual-reference
        result.update({"state": "complete" if -1e-10 <= kl <= 1e-4 else "failed",
                       "kl_nat": kl, "kl_mean_nat": kl, "kl_p99_nat": kl,
                       "rows": 1, "max_abs_logits": float(np.max(np.abs(delta))),
                       "logits_nrmse": float(np.sqrt(np.mean(delta*delta)) / max(np.sqrt(np.mean(reference*reference)), 1e-6)),
                       "argmax_equal": int(np.argmax(reference)) == int(np.argmax(actual)),
                       "tokens_equal": True, "finite": True,
                       "gate": "FP64 KL mean<=1e-4 nat and p99<=1e-3; one checkpoint, no 32-token structural claim"})
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        result["reason"] = f"not compared: {type(exc).__name__}: {exc}"
    return result


# ---------------------------------------------------------------------------
# Run aggregation (the single gate for the whole harness exit code)
# ---------------------------------------------------------------------------
#
# ONLY state == "complete" is acceptable. Everything else -- failed,
# pending_evidence, inconclusive, not_executed, a missing state, or an unknown
# state -- makes the aggregate INCOMPLETE and forces a non-zero exit. The
# previous aggregator only counted failed/pending_evidence, so an "inconclusive"
# or "not_executed" leg silently exited 0: that was a real false positive.
ACCEPTED_RUN_STATE = "complete"


def run_state_label(run: dict[str, Any]) -> str:
    """Stable per-leg label used in the aggregate and in the failure output."""
    for key in ("leg", "case_id"):
        value = run.get(key)
        if isinstance(value, str) and value:
            return value
    return "<unlabeled>"


def aggregate_run_states(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Count run states and decide acceptance. A run is acceptable only when
    its state is exactly ACCEPTED_RUN_STATE. A missing, non-string or unknown
    state is NOT acceptable and is reported separately so it can never be
    mistaken for a pass."""
    counts = {"complete": 0, "failed": 0, "pending_evidence": 0,
              "inconclusive": 0, "not_executed": 0, "missing_or_invalid_state": 0}
    unacceptable: list[str] = []
    for run in runs:
        state = run.get("state") if isinstance(run, dict) else None
        if not isinstance(state, str) or state not in counts:
            counts["missing_or_invalid_state"] += 1
            unacceptable.append(f"{run_state_label(run)}(state={state!r})")
            continue
        counts[state] += 1
        if state != ACCEPTED_RUN_STATE:
            unacceptable.append(f"{run_state_label(run)}({state})")
    return {
        **counts,
        "unacceptable_runs": len(unacceptable),
        "unacceptable_legs": unacceptable,
        "incomplete": bool(unacceptable),
    }


def record_fixture_precondition_failure(record_leg: Any, runs: list[dict[str, Any]], *,
                                       leg_name: str, p: int, ctx: int, source_ctx: int,
                                       output_dir: Path, own_dir: Path, own_file: Path,
                                       n_prefill: int, bound_name: str, bound: int,
                                       own_info: dict[str, Any], failures: list[str],
                                       expectation: str) -> dict[str, Any]:
    """Persist a failed own-fixture precondition in both scenario ledgers."""
    fixture = dict(own_info)
    fixture.update({
        "source_context": source_ctx,
        "prefill_tokens": n_prefill,
        "source_case_id": f"{leg_name}-fixture",
        "source_snapshot_path": str(own_file),
        "source_log_path": str(output_dir / f"server-{leg_name}-fixture.log"),
        "restore_case_id": f"{leg_name}-restore",
        "restore_log_path": str(output_dir / f"server-{leg_name}-restore.log"),
        "output_directory": str(own_dir),
        "bound_name": bound_name,
        "bound": bound,
    })
    reason = "; ".join(failures) or f"{leg_name}: fixture precondition failed"
    entry = record_leg(
        leg_name, "failed", p=p, ctx=ctx, source_ctx=source_ctx,
        expectation=expectation, fixture=fixture,
        source_snapshot_path=str(own_file),
        source_log_path=fixture["source_log_path"],
        restore_log_path=fixture["restore_log_path"],
        precondition_failure=True, validation_failures=list(failures), reason=reason)
    runs.append(entry)
    return entry


# ---------------------------------------------------------------------------
# Server instance management
# ---------------------------------------------------------------------------

class ServerInstance:
    def __init__(self, args: argparse.Namespace, output_dir: Path, case_id: str,
                 extra_args: list[str], slot_save_path: Path | None = None,
                 env_extra: dict[str, str] | None = None):
        self.args = args
        self.case_id = case_id
        self.slot_save_path = slot_save_path or (output_dir / f"slot-save-{case_id}")
        self.slot_save_path.mkdir(parents=True, exist_ok=True)
        self.log_path = output_dir / f"server-{case_id}.log"
        self.command = self.server_command(extra_args)
        self.proc: subprocess.Popen[bytes] | None = None
        self.env_extra = env_extra or {}
        self.log_offset = 0
        self.capture_dir: Path | None = None

    def enable_full_logit_capture(self, output_dir: Path, campaign_id: str,
                                 ctx: int, build_id: str,
                                 build_id_source: str = "legacy_server_binary_sha256") -> None:
        """Only scenario D enables this; every process owns its capture path."""
        self.capture_dir = output_dir / "full-logits" / self.case_id
        self.capture_dir.mkdir(parents=True, exist_ok=False)
        checkpoints = full_logit_checkpoints(self.args, ctx)
        self.env_extra.update({
            "F4_TEST_LOGIT_CAPTURE": str(self.capture_dir / "cap"),
            "F4_TEST_PHASE": self.case_id, "F4_TEST_RUN_ID": campaign_id,
            "F4_TEST_BUILD_ID": build_id,
            "F4_TEST_EXPECT_TOKENS": ",".join(map(str, checkpoints)),
            # This arm measures the GPU server, independently of an inherited
            # shell used for the separate structural CPU-reference campaign.
            "LLAMA_OVF_ATTN_REFERENCE_CPU": "0",
        })
        write_json(self.capture_dir / "capture-plan.json", {
            "case_id": self.case_id, "campaign_id": campaign_id, "ctx": ctx,
            "build_id": build_id, "build_id_source": build_id_source,
            "checkpoints": checkpoints, "environment": self.env_extra,
            "note": "planned checkpoint filter; only actual .json/.tokens/.logits files are evidence"})

    def server_command(self, extra_args: list[str]) -> list[str]:
        args = self.args
        command = [
            str(args.server_binary.resolve()), "--model", str(args.model.resolve()),
            "--host", args.host, "--port", str(args.port), "--device", args.cuda_device,
            "-ngl", "99", "--split-mode", "none", "--main-gpu", "0",
            "--ctx-size", str(args.ctx_size), "--parallel", "1",
            "--slot-save-path", str(self.slot_save_path),
            "--batch-size", str(args.batch_size), "--ubatch-size", str(args.ubatch_size),
            "--cache-type-k", "kvarn4", "--cache-type-v", "kvarn4",
            "--kv-tail-tokens", "0", "--kv-tail-type", "f16",
            "--flash-attn", "on", "--cache-prompt", "--fit", "off",
            "--no-ui", "--slots", "--no-warmup", "--no-context-shift", "--verbose",
        ]
        command += extra_args
        if any(value is not None for value in (args.apu_tdp, args.amd_sclk_prefill, args.amd_sclk_decode)):
            command += ["--gpu-power-backend", "amdgpu"]
            if args.apu_tdp is not None:
                command += ["--apu-tdp", str(args.apu_tdp)]
            if args.amd_sclk_prefill is not None:
                command += ["--amd-sclk-prefill", str(args.amd_sclk_prefill)]
            if args.amd_sclk_decode is not None:
                command += ["--amd-sclk-decode", str(args.amd_sclk_decode)]
        return command

    def start(self) -> None:
        base_url = f"http://{self.args.host}:{self.args.port}"
        env = os.environ.copy()
        for key in ("GGML_VK_PERF_LOGGER", "GGML_BACKEND_COPY_PROFILE", "GGML_KVARN_DEBUG_ROUTES"):
            env.pop(key, None)
        env.update(self.env_extra)
        with self.log_path.open("wb") as log:
            log.write((f"started_utc={utc_now()}\ncommand={shlex.join(self.command)}\n\n").encode())
            if self.capture_dir is not None:
                log.write(("capture_environment=" + json.dumps(self.env_extra, sort_keys=True) + "\n").encode())
            log.flush()
            self.proc = subprocess.Popen(self.command, stdout=log, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
        wait_ready(self.proc, base_url, self.args.startup_timeout)

    def stop(self, shutdown_timeout: float | None = None) -> int | None:
        if self.proc is not None:
            stop_server(self.proc, shutdown_timeout or self.args.shutdown_timeout)
            return self.proc.returncode
        return None

    def read_log_delta(self) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        delta = text[self.log_offset:]
        self.log_offset = len(text)
        return delta

    def log_evidence(self, delta: str) -> list[str]:
        return [line.strip() for line in delta.splitlines()
                if any(marker in line.lower() for marker in MARKERS)]


# ---------------------------------------------------------------------------
# Scenario A: mixed profile, base save/restore determinism + cross-context
# ---------------------------------------------------------------------------

def run_extension_leg(base_url: str, args: argparse.Namespace, records_path: Path,
                      case_id: str, extension: list[int],
                      prompt_hash: str, phase: str) -> tuple[dict[str, Any], list[str]]:
    payload = {
        "prompt": extension, "n_predict": args.decode_tokens, "id_slot": 0,
        "cache_prompt": True, "return_tokens": True,
        "temperature": 0, "seed": args.seed, "ignore_eos": True,
        "n_probs": args.n_probs,
    }
    result = http_request(base_url, "POST", "/completion", payload,
                          args.request_timeout, records_path, args.telemetry_device,
                          case_id, phase,
                          request_body_record={**payload, "prompt_sha256": prompt_hash})
    failures = validate_completion_counts(result["json"], len(extension), cold=False,
                                          expected_decode=args.decode_tokens)
    return result, failures


def scenario_a(args: argparse.Namespace, output_dir: Path, run_meta: dict[str, Any],
               frozen: dict[str, Any]) -> dict[str, Any]:
    base_url = f"http://{args.host}:{args.port}"
    results_path = output_dir / "results.jsonl"
    records_path = output_dir / "requests.jsonl"
    summary: dict[str, Any] = {"scenario": "A", "runs": []}
    for qx in args.remote_kv_types:
        case_id = f"A-{qx}"
        profile_args = [
            "--remote-attn", "vulkan:0", "--remote-attn-layers", str(args.remote_attn_layers),
            "--remote-attn-cache-type-k", qx, "--remote-attn-cache-type-v", qx,
            "--spec-type", "none",
            # Scenario A/C are target-only snapshot tests: the RAM prompt
            # cache would hide restore defects, so it is disabled explicitly.
            "--cache-ram", "0", "--ctx-checkpoints", "1",
        ]
        instance = ServerInstance(args, output_dir, case_id, profile_args)
        case_record: dict[str, Any] = {
            "case_id": case_id, "scenario": "A", "state": "starting",
            "remote_kv_type": qx, "command_argv": instance.command,
            "command_shell": shlex.join(instance.command), "server_log": instance.log_path.name,
            "slot_save_path": instance.slot_save_path.name, "started_utc": utc_now(), "requests": [],
        }
        append_jsonl(output_dir / "cases.jsonl", case_record)
        print(f"[{case_id}] starting: {shlex.join(instance.command)}", flush=True)
        try:
            instance.start()
            case_record["state"] = "ready"
            case_record["ready_utc"] = utc_now()
            append_jsonl(output_dir / "cases.jsonl", case_record)
            frozen = create_frozen_prompts(base_url, args, records_path, output_dir, case_id, frozen)
            run_meta["frozen_prompts"] = {"file": "prompts.json",
                                          "prefix_tokens": len(frozen["prefix_token_ids"]),
                                          "suffix_tokens": len(frozen["suffix_token_ids"])}
            write_json(output_dir / "run.json", run_meta)

            prefix = frozen_prefix_ids(frozen, args.prompt_tokens, "scenario A prompt")
            extension = prefix + frozen["suffix_token_ids"]
            if len(prefix) != args.prompt_tokens or len(extension) != args.prompt_tokens + args.suffix_tokens:
                raise RuntimeError("frozen prompt ids do not match the requested lengths")
            base_hash = sha256_bytes(json_bytes(prefix))
            extension_hash = sha256_bytes(json_bytes(extension))
            filename = f"snap-{case_id}.bin"
            failures: list[str] = []

            # Cold base prefix; the saved snapshot is the BASE state.
            cold_payload = {
                "prompt": prefix, "n_predict": args.decode_tokens, "id_slot": 0,
                "cache_prompt": True, "return_tokens": True,
                "temperature": 0, "seed": args.seed, "ignore_eos": True, "n_probs": args.n_probs,
            }
            cold = http_request(base_url, "POST", "/completion", cold_payload,
                                args.request_timeout, records_path, args.telemetry_device,
                                case_id, "A-cold",
                                request_body_record={**cold_payload, "prompt_sha256": base_hash})
            cold_failures = validate_completion_counts(cold["json"], len(prefix), cold=True,
                                                       expected_decode=args.decode_tokens)
            failures += cold_failures
            append_jsonl(results_path, {
                "scenario": "A", "case_id": case_id, "phase": "A-cold",
                "prompt_tokens": len(prefix), "prompt_sha256": base_hash,
                "timings": cold["json"].get("timings"), "content": cold["json"].get("content"),
                "generated_tokens": cold["json"].get("tokens"),
                "completion_probabilities": cold["json"].get("completion_probabilities"),
                "validation_failures": cold_failures,
            })

            save = http_request(base_url, "POST", "/slots/0?action=save", {"filename": filename},
                                args.request_timeout, records_path, args.telemetry_device,
                                case_id, "A-save-base")
            save_counts, save_failures = parse_slot_counts(save, "save")
            failures += save_failures
            append_jsonl(results_path, {
                "scenario": "A", "case_id": case_id, "phase": "A-save-base",
                "filename": filename, "save_counts": save_counts,
                "response_json": save.get("json"), "validation_failures": save_failures,
            })

            cont1, cont1_failures = run_extension_leg(
                base_url, args, records_path, case_id, extension, extension_hash, "A-extension-1")
            failures += cont1_failures
            append_jsonl(results_path, {
                "scenario": "A", "case_id": case_id, "phase": "A-extension-1",
                "prompt_tokens": len(extension), "prompt_sha256": extension_hash,
                "timings": cont1["json"].get("timings"), "content": cont1["json"].get("content"),
                "generated_tokens": cont1["json"].get("tokens"),
                "completion_probabilities": cont1["json"].get("completion_probabilities"),
                "validation_failures": cont1_failures,
            })

            erase = http_request(base_url, "POST", "/slots/0?action=erase", None,
                                 args.request_timeout, records_path, args.telemetry_device,
                                 case_id, "A-erase-after-extension-1")
            erase_counts, erase_failures = parse_slot_counts(erase, "erase")
            failures += erase_failures
            append_jsonl(results_path, {
                "scenario": "A", "case_id": case_id, "phase": "A-erase",
                "erase_counts": erase_counts, "response_json": erase.get("json"),
                "validation_failures": erase_failures,
            })

            restore = http_request(base_url, "POST", "/slots/0?action=restore", {"filename": filename},
                                   args.request_timeout, records_path, args.telemetry_device,
                                   case_id, "A-restore-base")
            restore_counts, restore_failures = parse_slot_counts(restore, "restore")
            failures += restore_failures
            failures += validate_save_restore_parity(save_counts, restore_counts)
            append_jsonl(results_path, {
                "scenario": "A", "case_id": case_id, "phase": "A-restore-base",
                "filename": filename, "restore_counts": restore_counts,
                "save_counts": save_counts, "response_json": restore.get("json"),
                "validation_failures": restore_failures + validate_save_restore_parity(
                    save_counts, restore_counts),
            })

            cont2, cont2_failures = run_extension_leg(
                base_url, args, records_path, case_id, extension, extension_hash, "A-extension-2")
            failures += cont2_failures
            append_jsonl(results_path, {
                "scenario": "A", "case_id": case_id, "phase": "A-extension-2",
                "prompt_tokens": len(extension), "prompt_sha256": extension_hash,
                "timings": cont2["json"].get("timings"), "content": cont2["json"].get("content"),
                "generated_tokens": cont2["json"].get("tokens"),
                "completion_probabilities": cont2["json"].get("completion_probabilities"),
                "validation_failures": cont2_failures,
            })

            def same_cadence(a: dict[str, Any], b: dict[str, Any]) -> bool:
                ta = a.get("json", {}).get("timings") or {}
                tb = b.get("json", {}).get("timings") or {}
                return (ta.get("cache_n") == tb.get("cache_n") and
                        ta.get("prompt_n") == tb.get("prompt_n") and
                        ta.get("predicted_n") == tb.get("predicted_n"))

            # Mandatory determinism: cadence equality AND token/text equality.
            determinism_failures: list[str] = []
            if not same_cadence(cont1, cont2):
                determinism_failures.append("extension cadence differs between the base-restored legs")
            for label, left, right in (("tokens", cont1, cont2), ("text", cont1, cont2)):
                left_value = left["json"].get("tokens") if label == "tokens" else left["json"].get("content")
                right_value = right["json"].get("tokens") if label == "tokens" else right["json"].get("content")
                if label == "tokens" and not deterministic_match(left_value, right_value):
                    determinism_failures.append("same-format generated token ids differ")
                if label == "text" and left_value != right_value:
                    determinism_failures.append("same-format generated text differs")
            failures += determinism_failures
            append_jsonl(results_path, {
                "scenario": "A", "case_id": case_id, "phase": "A-determinism",
                "cadence_matches": same_cadence(cont1, cont2),
                "tokens_equal": deterministic_match(cont1["json"].get("tokens"),
                                                    cont2["json"].get("tokens")),
                "text_equal": cont1["json"].get("content") == cont2["json"].get("content"),
                "validation_failures": determinism_failures,
            })

            # Cross-context usability: a fresh same-profile process restores
            # the same snapshot file and must produce the same cadence/output.
            # The first instance is stopped first: both use the same isolated
            # port and the same snapshot path.
            case_record["server_exit_code"] = instance.stop()
            fresh_id = f"A-{qx}-fresh"
            fresh = ServerInstance(args, output_dir, fresh_id, profile_args,
                                   slot_save_path=instance.slot_save_path)
            print(f"[{fresh_id}] starting: {shlex.join(fresh.command)}", flush=True)
            try:
                fresh.start()
                fresh_prompts = create_frozen_prompts(
                    base_url, args, records_path, output_dir, fresh_id, frozen)
                if fresh_prompts is not frozen:
                    raise RuntimeError("fresh instance tokenizer mismatch")
                fresh_restore = http_request(
                    base_url, "POST", "/slots/0?action=restore", {"filename": filename},
                    args.request_timeout, records_path, args.telemetry_device,
                    fresh_id, "A-fresh-restore")
                fresh_counts, fresh_failures = parse_slot_counts(fresh_restore, "restore")
                failures += fresh_failures
                failures += validate_save_restore_parity(save_counts, fresh_counts)
                append_jsonl(results_path, {
                    "scenario": "A", "case_id": case_id, "phase": "A-fresh-restore",
                    "filename": filename, "restore_counts": fresh_counts,
                    "save_counts": save_counts, "validation_failures":
                        fresh_failures + validate_save_restore_parity(save_counts, fresh_counts),
                })
                cont3, cont3_failures = run_extension_leg(
                    base_url, args, records_path, fresh_id, extension, extension_hash,
                    "A-fresh-extension")
                failures += cont3_failures
                append_jsonl(results_path, {
                    "scenario": "A", "case_id": case_id, "phase": "A-fresh-extension",
                    "prompt_tokens": len(extension), "prompt_sha256": extension_hash,
                    "timings": cont3["json"].get("timings"),
                    "content": cont3["json"].get("content"),
                    "generated_tokens": cont3["json"].get("tokens"),
                    "completion_probabilities": cont3["json"].get("completion_probabilities"),
                    "validation_failures": cont3_failures,
                })
                cross_failures: list[str] = []
                if not same_cadence(cont1, cont3):
                    cross_failures.append("fresh-instance extension cadence differs from the base leg")
                if not deterministic_match(cont1["json"].get("tokens"), cont3["json"].get("tokens")):
                    cross_failures.append("fresh-instance generated token ids differ")
                if cont1["json"].get("content") != cont3["json"].get("content"):
                    cross_failures.append("fresh-instance generated text differs")
                failures += cross_failures
                append_jsonl(results_path, {
                    "scenario": "A", "case_id": case_id, "phase": "A-cross-context",
                    "cadence_matches": same_cadence(cont1, cont3),
                    "tokens_equal": deterministic_match(cont1["json"].get("tokens"),
                                                        cont3["json"].get("tokens")),
                    "text_equal": cont1["json"].get("content") == cont3["json"].get("content"),
                    "validation_failures": cross_failures,
                })
                case_record["cross_context"] = {"state": "complete" if not cross_failures else "failed",
                                                "fresh_case_id": fresh_id}
            finally:
                case_record["cross_context_server_exit_code"] = fresh.stop()
                fresh_delta = fresh.read_log_delta() if fresh.proc is not None else ""
                fresh_layout = parse_mixed_layout(
                    fresh.log_path.read_text(encoding="utf-8", errors="replace"))
                case_record["fresh_mixed_format_log_evidence"] = fresh_layout

            case_record["requests"] = [
                {"phase": "A-cold", "cache_n": (cold["json"].get("timings") or {}).get("cache_n"),
                 "prompt_n": (cold["json"].get("timings") or {}).get("prompt_n"),
                 "predicted_n": (cold["json"].get("timings") or {}).get("predicted_n")},
                {"phase": "A-save-base", **save_counts},
                {"phase": "A-extension-1",
                 "cache_n": (cont1["json"].get("timings") or {}).get("cache_n"),
                 "prompt_n": (cont1["json"].get("timings") or {}).get("prompt_n"),
                 "predicted_n": (cont1["json"].get("timings") or {}).get("predicted_n")},
                {"phase": "A-erase", **erase_counts},
                {"phase": "A-restore-base", **restore_counts},
                {"phase": "A-extension-2",
                 "cache_n": (cont2["json"].get("timings") or {}).get("cache_n"),
                 "prompt_n": (cont2["json"].get("timings") or {}).get("prompt_n"),
                 "predicted_n": (cont2["json"].get("timings") or {}).get("predicted_n")},
                {"phase": "A-fresh-extension",
                 "cache_n": (cont3["json"].get("timings") or {}).get("cache_n"),
                 "prompt_n": (cont3["json"].get("timings") or {}).get("prompt_n"),
                 "predicted_n": (cont3["json"].get("timings") or {}).get("predicted_n")},
            ]
            case_record["validation_failures"] = failures
            case_record["state"] = "complete" if not failures else "failed"
            if failures:
                case_record["error"] = "; ".join(failures)
        except Exception as exc:
            case_record["state"] = "failed"
            case_record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{case_id}] failed: {case_record['error']}", flush=True)
        finally:
            case_record["server_exit_code"] = instance.stop()
            full_log = instance.log_path.read_text(encoding="utf-8", errors="replace")
            layout = parse_mixed_layout(full_log)
            violations = target_only_violations(full_log)
            layout_failures = validate_mixed_layout(layout, args.remote_attn_layers, qx)
            if violations:
                case_record["state"] = "failed"
                case_record["error"] = (case_record.get("error", "") +
                                        f"; target-only violated: {violations[0]}")
            if case_record["state"] == "complete":
                expected = f"Vulkan-layers={args.remote_attn_layers}/"
                if not any(expected in line for line in full_log.splitlines()):
                    case_record["state"] = "failed"
                    case_record["error"] = f"startup log did not confirm placement {expected}<total>"
                if layout_failures:
                    case_record["state"] = "failed"
                    case_record["error"] = "per-layer KV log: " + "; ".join(layout_failures)
            case_record["mixed_format_log_evidence"] = layout
            case_record["target_only_violations"] = violations
            case_record["finished_utc"] = utc_now()
            append_jsonl(output_dir / "cases.jsonl", case_record)
            summary["runs"].append(case_record)
            print(f"[{case_id}] state={case_record['state']}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Scenario B: adaptive local -> mixed transition
# ---------------------------------------------------------------------------

def scenario_b(args: argparse.Namespace, output_dir: Path, run_meta: dict[str, Any],
               frozen: dict[str, Any]) -> dict[str, Any]:
    base_url = f"http://{args.host}:{args.port}"
    results_path = output_dir / "results.jsonl"
    records_path = output_dir / "requests.jsonl"
    qx = args.remote_kv_types[0]
    case_id = "B-adaptive"
    instance = ServerInstance(args, output_dir, case_id, [
        "--spec-type", "draft-mtp",
        # Adaptive profile mapping (common_context_adaptive_error rejects
        # --ctx-size-mtp > --ctx-size): the LONG profile is --ctx-size (1024,
        # the high tier) and the MTP profile is --ctx-size-mtp (512, the local
        # tier). Budget 316 (prompt 300 + output 16) selects MTP (512, below
        # the remote gate), budget 716 selects LONG (1024, above the gate).
        "--ctx-size", str(args.high_ctx),
        "--ctx-size-mtp", str(args.local_ctx),
        "--mtp-max-tokens", str(args.mtp_max_tokens),
        # Explicit per-tier KVarN4: an untouched tier inherits the global
        # --cache-type-k/v (arg.cpp common_params_target_tier_kvarn_normalize),
        # so kvarn4 would hold in both tiers anyway; explicit flags document
        # the intent and survive future default changes.
        "--cache-type-k-m", "kvarn4", "--cache-type-v-m", "kvarn4",
        "--cache-type-k-l", "kvarn4", "--cache-type-v-l", "kvarn4",
        "--remote-attn", "vulkan:0",
        "--remote-attn-layers", str(args.remote_attn_layers),
        "--remote-attn-min-context", str(args.gate_ctx),
        "--remote-attn-cache-type-k", qx, "--remote-attn-cache-type-v", qx,
        # Scenario B preserves the prefix across the adaptive transition via
        # the RAM prompt cache; it must stay enabled (explicit 256 MiB).
        "--cache-ram", "256", "--ctx-checkpoints", "1",
    ])
    case_record: dict[str, Any] = {
        "case_id": case_id, "scenario": "B", "state": "starting",
        "remote_kv_type": qx, "local_ctx": args.local_ctx, "high_ctx": args.high_ctx,
        "mtp_max_tokens": args.mtp_max_tokens, "gate_ctx": args.gate_ctx,
        "handoff_evidence_markers": args.handoff_evidence_markers,
        "command_argv": instance.command, "command_shell": shlex.join(instance.command),
        "server_log": instance.log_path.name,
        "slot_save_path": instance.slot_save_path.name, "started_utc": utc_now(), "requests": [],
    }
    append_jsonl(output_dir / "cases.jsonl", case_record)
    print(f"[{case_id}] starting: {shlex.join(instance.command)}", flush=True)
    try:
        instance.start()
        case_record["state"] = "ready"
        case_record["ready_utc"] = utc_now()
        append_jsonl(output_dir / "cases.jsonl", case_record)
        frozen = create_frozen_prompts(base_url, args, records_path, output_dir, case_id, frozen)
        write_json(output_dir / "run.json", run_meta)

        short_ids = frozen_prefix_ids(frozen, args.prompt_short, "scenario B short prompt")
        long_ids = frozen_prefix_ids(frozen, args.prompt_long, "scenario B long prompt")
        if len(short_ids) != args.prompt_short or len(long_ids) != args.prompt_long:
            raise RuntimeError("frozen prompt ids do not cover the adaptive prompt lengths")
        short_hash = sha256_bytes(json_bytes(short_ids))
        long_hash = sha256_bytes(json_bytes(long_ids))
        failures: list[str] = []

        def completion(ids: list[int], ids_hash: str, phase: str) -> tuple[dict[str, Any], list[str]]:
            payload = {
                "prompt": ids, "n_predict": args.n_predict, "id_slot": 0,
                "cache_prompt": True, "return_tokens": True,
                "temperature": 0, "seed": args.seed, "ignore_eos": True,
                "n_probs": args.n_probs,
            }
            result = http_request(base_url, "POST", "/completion", payload,
                                  args.request_timeout, records_path, args.telemetry_device,
                                  case_id, phase,
                                  request_body_record={**payload, "prompt_sha256": ids_hash})
            return result, []

        # Pre-gate: the short prompt (300 + 16 = 316) selects the MTP profile
        # (--ctx-size-mtp 512 < gate 513): remote attention stays off.
        short, _ = completion(short_ids, short_hash, "B-short-cold")
        short_failures = validate_completion_counts(short["json"], len(short_ids), cold=True,
                                                    expected_decode=args.n_predict)
        failures += short_failures
        append_jsonl(results_path, {
            "scenario": "B", "case_id": case_id, "phase": "B-short-cold",
            "prompt_tokens": len(short_ids), "prompt_sha256": short_hash,
            "timings": short["json"].get("timings"), "content": short["json"].get("content"),
            "generated_tokens": short["json"].get("tokens"),
            "completion_probabilities": short["json"].get("completion_probabilities"),
            "validation_failures": short_failures,
        })
        pre_gate_log = instance.read_log_delta()

        short2, _ = completion(short_ids, short_hash, "B-short-continuation")
        short2_failures = validate_completion_counts(short2["json"], len(short_ids), cold=False,
                                                     expected_decode=args.n_predict)
        failures += short2_failures
        append_jsonl(results_path, {
            "scenario": "B", "case_id": case_id, "phase": "B-short-continuation",
            "prompt_tokens": len(short_ids), "prompt_sha256": short_hash,
            "timings": short2["json"].get("timings"), "content": short2["json"].get("content"),
            "generated_tokens": short2["json"].get("tokens"),
            "completion_probabilities": short2["json"].get("completion_probabilities"),
            "validation_failures": short2_failures,
        })

        # Handoff: the long prompt (700 + 16 = 716) crosses the MTP threshold and
        # selects the LONG profile (--ctx-size 1024 >= gate 513); it shares the
        # 300-token common prefix with the short prompt, so its cache must be
        # preserved across the profile transition (cache_n > 0, <= 300).
        handoff, _ = completion(long_ids, long_hash, "B-handoff")
        handoff_failures = validate_handoff_counts(
            handoff["json"], args.prompt_short, len(long_ids), expected_decode=args.n_predict)
        failures += handoff_failures
        append_jsonl(results_path, {
            "scenario": "B", "case_id": case_id, "phase": "B-handoff",
            "prompt_tokens": len(long_ids), "prompt_sha256": long_hash,
            "common_prefix": args.prompt_short, "timings": handoff["json"].get("timings"),
            "content": handoff["json"].get("content"), "generated_tokens": handoff["json"].get("tokens"),
            "completion_probabilities": handoff["json"].get("completion_probabilities"),
            "validation_failures": handoff_failures,
        })
        gate_log = instance.read_log_delta()

        # Continuation after the transition (cache_n > 0 is expected, but this
        # leg alone does not prove the handoff; the B-handoff checks do).
        long2, _ = completion(long_ids, long_hash, "B-long-continuation")
        long2_failures = validate_completion_counts(long2["json"], len(long_ids), cold=False,
                                                    expected_decode=args.n_predict)
        failures += long2_failures
        append_jsonl(results_path, {
            "scenario": "B", "case_id": case_id, "phase": "B-long-continuation",
            "prompt_tokens": len(long_ids), "prompt_sha256": long_hash,
            "timings": long2["json"].get("timings"), "content": long2["json"].get("content"),
            "generated_tokens": long2["json"].get("tokens"),
            "completion_probabilities": long2["json"].get("completion_probabilities"),
            "proves_handoff": False,
            "validation_failures": long2_failures,
        })

        pre_gate_evidence = instance.log_evidence(pre_gate_log)
        gate_evidence = instance.log_evidence(gate_log)
        full_log = instance.log_path.read_text(encoding="utf-8", errors="replace")
        layout = parse_mixed_layout(full_log)
        pre_gate_has_qx = any("mixed kv layer=" in line.lower() for line in pre_gate_evidence)
        pre_gate_backend_off = any(
            "adaptive remote attention:" in line.lower() and "backend=off" in line.lower()
            for line in pre_gate_evidence)
        gate_backend_on = any(
            "adaptive remote attention:" in line.lower() and "backend=off" not in line.lower()
            for line in gate_evidence)
        gate_transition = any("adaptive context transition complete" in line.lower()
                              for line in gate_evidence)
        layout_failures = validate_mixed_layout(layout, args.remote_attn_layers, qx)
        if not pre_gate_backend_off:
            failures.append("pre-gate log lacks 'adaptive remote attention: ... backend=off'")
        if pre_gate_has_qx:
            failures.append("pre-gate log contains a Qx 'mixed KV layer' allocation")
        if not gate_backend_on:
            failures.append("gate log lacks an enabled 'adaptive remote attention' line")
        if not gate_transition:
            failures.append("gate log lacks 'adaptive context transition complete' (snapshot restore)")
        failures += layout_failures

        # Explicit conversion-success evidence: only when the integration
        # marker is configured. Without it the check is recorded as deferred
        # and is never claimed as passed.
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

        case_record["requests"] = [
            {"phase": "B-short-cold", "cache_n": (short["json"].get("timings") or {}).get("cache_n"),
             "prompt_n": (short["json"].get("timings") or {}).get("prompt_n")},
            {"phase": "B-short-continuation",
             "cache_n": (short2["json"].get("timings") or {}).get("cache_n"),
             "prompt_n": (short2["json"].get("timings") or {}).get("prompt_n")},
            {"phase": "B-handoff", "cache_n": (handoff["json"].get("timings") or {}).get("cache_n"),
             "prompt_n": (handoff["json"].get("timings") or {}).get("prompt_n")},
            {"phase": "B-long-continuation",
             "cache_n": (long2["json"].get("timings") or {}).get("cache_n"),
             "prompt_n": (long2["json"].get("timings") or {}).get("prompt_n")},
        ]
        case_record["pre_gate_evidence"] = pre_gate_evidence
        case_record["gate_evidence"] = gate_evidence
        case_record["mixed_format_log_evidence"] = layout
        case_record["validation_failures"] = failures
        case_record["state"] = acceptance_state(failures, handoff_evidence)
        if case_record["state"] == "failed":
            case_record["error"] = "; ".join(failures)
        elif case_record["state"] == "pending_evidence":
            case_record["error"] = ("acceptance pending: handoff conversion evidence markers "
                                    "were not configured (--handoff-evidence-markers); "
                                    "no Qx-conversion success is claimed")
    except Exception as exc:
        case_record["state"] = "failed"
        case_record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[{case_id}] failed: {case_record['error']}", flush=True)
    finally:
        case_record["server_exit_code"] = instance.stop()
        if case_record["state"] != "complete" and case_record.get("server_exit_code") not in (None, 0):
            case_record["state"] = "failed"
            case_record["error"] = (case_record.get("error", "") +
                                    f" (server exit={case_record['server_exit_code']})")
        case_record["finished_utc"] = utc_now()
        append_jsonl(output_dir / "cases.jsonl", case_record)
        print(f"[{case_id}] state={case_record['state']}", flush=True)
    return {"scenario": "B", "runs": [case_record]}


# ---------------------------------------------------------------------------
# Scenario C: malformed snapshot must fail; valid restore must recover
# ---------------------------------------------------------------------------

def scenario_c(args: argparse.Namespace, output_dir: Path, run_meta: dict[str, Any],
               frozen: dict[str, Any]) -> dict[str, Any]:
    base_url = f"http://{args.host}:{args.port}"
    results_path = output_dir / "results.jsonl"
    records_path = output_dir / "requests.jsonl"
    qx = args.remote_kv_types[0]
    case_id = "C-snapshot"
    instance = ServerInstance(args, output_dir, case_id, [
        "--remote-attn", "vulkan:0", "--remote-attn-layers", str(args.remote_attn_layers),
        "--remote-attn-cache-type-k", qx, "--remote-attn-cache-type-v", qx,
        "--spec-type", "none",
        "--cache-ram", "0", "--ctx-checkpoints", "1",
    ])
    case_record: dict[str, Any] = {
        "case_id": case_id, "scenario": "C", "state": "starting",
        "remote_kv_type": qx, "command_argv": instance.command,
        "command_shell": shlex.join(instance.command), "server_log": instance.log_path.name,
        "slot_save_path": instance.slot_save_path.name, "started_utc": utc_now(), "requests": [],
    }
    append_jsonl(output_dir / "cases.jsonl", case_record)
    print(f"[{case_id}] starting: {shlex.join(instance.command)}", flush=True)
    try:
        instance.start()
        case_record["state"] = "ready"
        case_record["ready_utc"] = utc_now()
        append_jsonl(output_dir / "cases.jsonl", case_record)
        frozen = create_frozen_prompts(base_url, args, records_path, output_dir, case_id, frozen)
        write_json(output_dir / "run.json", run_meta)

        prefix = frozen_prefix_ids(frozen, 512, "scenario C prompt")
        extension = prefix + frozen["suffix_token_ids"]
        base_hash = sha256_bytes(json_bytes(prefix))
        extension_hash = sha256_bytes(json_bytes(extension))
        failures: list[str] = []

        cold_payload = {
            "prompt": prefix, "n_predict": args.decode_tokens, "id_slot": 0,
            "cache_prompt": True, "return_tokens": True,
            "temperature": 0, "seed": args.seed, "ignore_eos": True, "n_probs": args.n_probs,
        }
        cold = http_request(base_url, "POST", "/completion", cold_payload,
                            args.request_timeout, records_path, args.telemetry_device,
                            case_id, "C-cold",
                            request_body_record={**cold_payload, "prompt_sha256": base_hash})
        cold_failures = validate_completion_counts(cold["json"], len(prefix), cold=True,
                                                   expected_decode=args.decode_tokens)
        failures += cold_failures
        append_jsonl(results_path, {
            "scenario": "C", "case_id": case_id, "phase": "C-cold",
            "prompt_tokens": len(prefix), "prompt_sha256": base_hash,
            "timings": cold["json"].get("timings"), "content": cold["json"].get("content"),
            "generated_tokens": cold["json"].get("tokens"),
            "completion_probabilities": cold["json"].get("completion_probabilities"),
            "validation_failures": cold_failures,
        })

        filename = "snap-C.bin"
        save = http_request(base_url, "POST", "/slots/0?action=save", {"filename": filename},
                            args.request_timeout, records_path, args.telemetry_device,
                            case_id, "C-save")
        save_counts, save_failures = parse_slot_counts(save, "save")
        failures += save_failures
        append_jsonl(results_path, {
            "scenario": "C", "case_id": case_id, "phase": "C-save",
            "filename": filename, "save_counts": save_counts,
            "response_json": save.get("json"), "validation_failures": save_failures,
        })
        saved_path = instance.slot_save_path / filename
        if not saved_path.is_file() or saved_path.stat().st_size == 0:
            raise RuntimeError(f"saved snapshot missing or empty: {saved_path}")

        corrupt_name = "snap-C-corrupt.bin"
        corrupt_path = instance.slot_save_path / corrupt_name
        data = saved_path.read_bytes()
        corrupt_path.write_bytes(data[:max(1, len(data) // 2)] + b"\x00GARBAGE")
        corrupt_restore = http_request(
            base_url, "POST", "/slots/0?action=restore", {"filename": corrupt_name},
            args.request_timeout, records_path, args.telemetry_device,
            case_id, "C-restore-corrupt")
        corrupt_failed, corrupt_detail = response_is_error(corrupt_restore)
        corrupt_failures = ([] if corrupt_failed else
                            ["truncated snapshot was accepted"])
        failures += corrupt_failures
        append_jsonl(results_path, {
            "scenario": "C", "case_id": case_id, "phase": "C-restore-corrupt",
            "filename": corrupt_name, "truncated_bytes": corrupt_path.stat().st_size,
            "rejected": corrupt_failed, "rejection_detail": corrupt_detail,
            "response_json": corrupt_restore.get("json"),
            "response_status": corrupt_restore.get("status"),
            "validation_failures": corrupt_failures,
        })

        missing_restore = http_request(
            base_url, "POST", "/slots/0?action=restore", {"filename": "does-not-exist.bin"},
            args.request_timeout, records_path, args.telemetry_device,
            case_id, "C-restore-missing")
        missing_failed, missing_detail = response_is_error(missing_restore)
        missing_failures = ([] if missing_failed else
                            ["missing snapshot was accepted"])
        failures += missing_failures
        append_jsonl(results_path, {
            "scenario": "C", "case_id": case_id, "phase": "C-restore-missing",
            "filename": "does-not-exist.bin", "rejected": missing_failed,
            "rejection_detail": missing_detail, "response_json": missing_restore.get("json"),
            "response_status": missing_restore.get("status"),
            "validation_failures": missing_failures,
        })

        restore = http_request(base_url, "POST", "/slots/0?action=restore", {"filename": filename},
                               args.request_timeout, records_path, args.telemetry_device,
                               case_id, "C-restore-valid")
        restore_counts, restore_failures = parse_slot_counts(restore, "restore")
        failures += restore_failures
        failures += validate_save_restore_parity(save_counts, restore_counts)
        append_jsonl(results_path, {
            "scenario": "C", "case_id": case_id, "phase": "C-restore-valid",
            "filename": filename, "restore_counts": restore_counts,
            "save_counts": save_counts, "validation_failures":
                restore_failures + validate_save_restore_parity(save_counts, restore_counts),
        })

        cont_payload = {
            "prompt": extension, "n_predict": args.decode_tokens, "id_slot": 0,
            "cache_prompt": True, "return_tokens": True,
            "temperature": 0, "seed": args.seed, "ignore_eos": True, "n_probs": args.n_probs,
        }
        cont = http_request(base_url, "POST", "/completion", cont_payload,
                            args.request_timeout, records_path, args.telemetry_device,
                            case_id, "C-continuation",
                            request_body_record={**cont_payload, "prompt_sha256": extension_hash})
        cont_failures = validate_completion_counts(cont["json"], len(extension), cold=False,
                                                   expected_decode=args.decode_tokens)
        failures += cont_failures
        append_jsonl(results_path, {
            "scenario": "C", "case_id": case_id, "phase": "C-continuation",
            "prompt_tokens": len(extension), "prompt_sha256": extension_hash,
            "timings": cont["json"].get("timings"), "content": cont["json"].get("content"),
            "generated_tokens": cont["json"].get("tokens"),
            "completion_probabilities": cont["json"].get("completion_probabilities"),
            "validation_failures": cont_failures,
        })

        case_record["requests"] = [
            {"phase": "C-cold", "cache_n": (cold["json"].get("timings") or {}).get("cache_n"),
             "prompt_n": (cold["json"].get("timings") or {}).get("prompt_n")},
            {"phase": "C-save", **save_counts},
            {"phase": "C-restore-corrupt", "rejected": corrupt_failed,
             "rejection_detail": corrupt_detail},
            {"phase": "C-restore-missing", "rejected": missing_failed,
             "rejection_detail": missing_detail},
            {"phase": "C-restore-valid", **restore_counts},
            {"phase": "C-continuation",
             "cache_n": (cont["json"].get("timings") or {}).get("cache_n"),
             "prompt_n": (cont["json"].get("timings") or {}).get("prompt_n")},
        ]
        case_record["validation_failures"] = failures
        case_record["state"] = "complete" if not failures else "failed"
        if failures:
            case_record["error"] = "; ".join(failures)
    except Exception as exc:
        case_record["state"] = "failed"
        case_record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[{case_id}] failed: {case_record['error']}", flush=True)
    finally:
        case_record["server_exit_code"] = instance.stop()
        full_log = instance.log_path.read_text(encoding="utf-8", errors="replace")
        violations = target_only_violations(full_log)
        if violations:
            case_record["state"] = "failed"
            case_record["error"] = (case_record.get("error", "") +
                                    f"; target-only violated: {violations[0]}")
        case_record["target_only_violations"] = violations
        case_record["finished_utc"] = utc_now()
        append_jsonl(output_dir / "cases.jsonl", case_record)
        print(f"[{case_id}] state={case_record['state']}", flush=True)
    return {"scenario": "C", "runs": [case_record]}


# ---------------------------------------------------------------------------
# Scenario D: §3.6 disk-only XXL snapshot, position-split P, fresh process
# ---------------------------------------------------------------------------
#
# Every leg runs in a NEW server process whose ONLY reusable state is the
# snapshot file in its own --slot-save-path. The RAM prompt cache is disabled
# (--cache-ram 0) so a hit can only come from disk, and each leg records
# timings.cache_n (tokens actually reused) and timings.prompt_n (tokens
# re-prefilled). "Import without prefill" is the strict claim:
# cache_n == imported prefix and prompt_n == the tokens genuinely new.
#
# The public --kv-position-split-p option is supported. To preserve this
# harness's established scenario-D behavior, start_split() still supplies the
# equivalent legacy LLAMA_KV_POSITION_SPLIT_P environment variable.

def split_profile(args: argparse.Namespace, qx: str, split: bool,
                  pure_source: bool = False) -> list[str]:
    """Profile for a split leg.

    pure_source=True builds a PURE KVarN (XXL) cache: NO --remote-attn at all,
    so common_context records kv_layout_known=false (common.cpp:4093-4094,4117)
    and llama-kv-cache-kvarn is not mixed. That is the XXL source the §3.6
    import must consume. This is a DIFFERENT cache class from the split->split
    leg and is labelled separately; the two must never be conflated.

    split=True builds the position-split placement, which requires EVERY
    full-attention layer on the remote backend: llama_context refuses with
    "position-split KV exige todas as N camadas full no backend remoto
    (recebido M)" for any M<N (observed with 1 on an 8-layer-full model).

    A non-split leg (P=0) drops the remote flags entirely. It must NOT pass
    --remote-attn-min-context: that flag requires an adaptive context and the
    server exits with "--remote-attn-min-context requires adaptive context".
    """
    if pure_source:
        # PURE KVarN XXL source: KVarN everywhere, no standard/remote layers.
        return [
            "--spec-type", "none",
            "--cache-ram", "0", "--ctx-checkpoints", "1",
        ]
    if split:
        return [
            "--remote-attn", "vulkan:0",
            "--remote-attn-layers", str(args.split_full_layers),
            "--remote-attn-cache-type-k", qx, "--remote-attn-cache-type-v", qx,
            "--spec-type", "none",
            # Disk-only: the RAM prompt cache must never satisfy a leg.
            "--cache-ram", "0", "--ctx-checkpoints", "1",
        ]
    return [
        "--spec-type", "none",
        "--cache-ram", "0", "--ctx-checkpoints", "1",
    ]


def scenario_d(args: argparse.Namespace, output_dir: Path, run_meta: dict[str, Any],
               frozen: dict[str, Any]) -> dict[str, Any]:
    base_url = f"http://{args.host}:{args.port}"
    results_path = output_dir / "results.jsonl"
    records_path = output_dir / "requests.jsonl"
    qx = args.remote_kv_types[0]
    p_local = args.split_p_local
    p_high = args.split_p_high
    runs: list[dict[str, Any]] = []
    failures: list[str] = []
    capture_enabled = bool(getattr(args, "full_logit_capture", False))
    campaign_id = f"{output_dir.name}:{run_meta.get('created_utc', utc_now())}"
    capture_build_id = run_meta.get("full_logit_capture_build_id") if capture_enabled else ""
    capture_build_id_source = run_meta.get("full_logit_capture_build_id_source", "not_requested")

    def log(msg: str) -> None:
        print(f"[D] {msg}", flush=True)

    def record_leg(leg: str, state: str, **fields: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {"scenario": "D", "leg": leg, "state": state,
                                 "finished_utc": utc_now(), **fields}
        append_jsonl(results_path, entry)
        return entry

    def start_split(case_id: str, p: int, ctx: int, save_path: Path | None,
                    extra_profile: list[str]) -> ServerInstance:
        """One P-configured process. For p>0, ctx must exceed P and P must
        be a multiple of 256 (llama-context.cpp:455). The per-leg --ctx-size is
        applied to the constructed argv AFTER construction so the recorded argv
        is exactly what runs; the replacement is positional so no other flag's
        value can be mistaken for the context size."""
        env = {}
        # The public --kv-position-split-p flag is also supported; this runner
        # keeps its established environment route. P=0 clears the legacy
        # override explicitly so an inherited shell value cannot turn a PURE
        # destination back into a split cache.
        if p > 0:
            env["LLAMA_KV_POSITION_SPLIT_P"] = str(p)
        else:
            env["LLAMA_KV_POSITION_SPLIT_P"] = ""
        # The spool must live on a real disk, never tmpfs/ramfs: that is the
        # 'only the snapshot on disk, no resident cache' precondition. The
        # directory is per-case so legs cannot see each other's spool.
        spool = output_dir / f"spool-{case_id}"
        spool.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(spool)
        inst = ServerInstance(args, output_dir, case_id, list(extra_profile),
                              slot_save_path=save_path, env_extra=env)
        argv = inst.command
        i = argv.index("--ctx-size")
        argv[i + 1] = str(ctx)
        inst.command = argv
        if capture_enabled:
            inst.enable_full_logit_capture(output_dir, campaign_id, ctx, capture_build_id,
                                           capture_build_id_source)
        log(f"{case_id} P={p} ctx={ctx} starting")
        try:
            inst.start()
        except Exception:
            inst.stop()
            raise
        return inst

    def save_snapshot(inst: ServerInstance, case_id: str, tokens: list[int],
                      filename: str,
                      phase: str) -> tuple[str, dict[str, Any], list[str]]:
        """Fill the slot, then SLOT_SAVE under an explicit filename. The server
        reports n_saved (tokens in the frame) and n_written (file bytes)."""
        prefill = http_request(
            base_url, "POST", "/completion",
            {"prompt": tokens, "n_predict": args.decode_tokens, "id_slot": 0,
             "cache_prompt": True, "return_tokens": True, "temperature": 0,
             "seed": args.seed, "ignore_eos": True, "n_probs": args.n_probs},
            args.request_timeout, records_path, args.telemetry_device,
            case_id, phase, request_body_record={"n_prompt_tokens": len(tokens)})
        prefill_failures = validate_completion_counts(
            prefill["json"], len(tokens), cold=True, expected_decode=args.decode_tokens)
        save = http_request(base_url, "POST", "/slots/0?action=save",
                            {"filename": filename}, args.request_timeout,
                            records_path, args.telemetry_device, case_id, f"{phase}-save")
        counts, save_failures = parse_slot_counts(save, "save")
        # n_saved is the occupancy the envelope claims; §3.6 hinges on it.
        n_saved = counts.get("n_saved")
        if not (isinstance(n_saved, int) and n_saved > 0):
            save_failures.append(f"save missing positive n_saved (got {n_saved!r})")
        info = {"prefill": prefill["json"].get("timings"), "save": counts,
                "n_saved": n_saved, "path": str(inst.slot_save_path / filename),
                "sha256": (sha256_file(inst.slot_save_path / filename)
                           if (inst.slot_save_path / filename).is_file() else None)}
        return filename, info, prefill_failures + save_failures

    def restore_and_extend(inst: ServerInstance, case_id: str, filename: str,
                           extension: list[int], phase: str,
                           expect_restore: bool,
                           expect_cache_n: int | None,
                           expect_hit: bool | None = None) -> tuple[dict[str, Any], list[str]]:
        """Restore the disk snapshot in THIS (already fresh) process, then send
        prefix+suffix. Reports reuse honestly: cache_n is the number of tokens
        the server actually reused, prompt_n the ones it re-prefilled.

        expect_hit decides the cold/hit accounting: True requires cache_n>0,
        False requires cache_n==0, and None only checks the sum identity (used
        by the legs whose reuse depends on how many groups the frame sealed)."""
        restore = http_request(base_url, "POST", "/slots/0?action=restore",
                               {"filename": filename}, args.request_timeout,
                               records_path, args.telemetry_device, case_id, f"{phase}-restore")
        failed_r, detail_r = response_is_error(restore)
        leg: dict[str, Any] = {"restore_response": restore["json"] if not failed_r else detail_r}
        leg_failures: list[str] = []
        if expect_restore and failed_r:
            leg_failures.append(f"{phase}: restore unexpectedly failed: {detail_r}")
        if not expect_restore and not failed_r:
            leg_failures.append(f"{phase}: restore unexpectedly succeeded (must be rejected)")
        if not failed_r:
            counts, rf = parse_slot_counts(restore, "restore")
            leg["restore_counts"] = counts
            leg_failures += rf if expect_restore else []
        cont = http_request(
            base_url, "POST", "/completion",
            {"prompt": extension, "n_predict": args.decode_tokens, "id_slot": 0,
             "cache_prompt": True, "return_tokens": True, "temperature": 0,
             "seed": args.seed, "ignore_eos": True, "n_probs": args.n_probs},
            args.request_timeout, records_path, args.telemetry_device,
            case_id, f"{phase}-extension",
            request_body_record={"n_prompt_tokens": len(extension)})
        leg["extension_timings"] = cont["json"].get("timings")
        leg["generated_tokens"] = cont["json"].get("tokens")
        leg["content"] = cont["json"].get("content")
        probs = cont["json"].get("completion_probabilities")
        leg["completion_probabilities"] = probs
        timings = cont["json"].get("timings") or {}
        cache_n = timings.get("cache_n")
        prompt_n = timings.get("prompt_n")
        leg["cache_n"] = cache_n
        leg["prompt_n"] = prompt_n
        leg["cache_source"] = timings.get("cache_source")
        if not expect_restore and failed_r:
            # Rejection leg whose restore was refused: the follow-up request is
            # INFORMATIONAL. The server may refuse it too (the slot was cleared
            # by the rejected restore), so requiring timings here would test
            # nothing about the contract. The contract is: the restore was
            # refused AND nothing from the refused snapshot was reused. A
            # successful follow-up with cache_n>0 would prove the snapshot WAS
            # used, which is a real failure.
            leg["extension_informational"] = True
            cont_failed, cont_detail = response_is_error(cont)
            leg["extension_error"] = cont_detail
            if not cont_failed and isinstance(cache_n, int) and cache_n > 0:
                cf = [f"{phase}: restore was rejected but the follow-up reused "
                      f"cache_n={cache_n}; the refused snapshot was used anyway"]
            else:
                cf = []
            leg["validation_failures"] = leg_failures + cf
            return leg, leg_failures + cf
        # Reuse accounting: cache_n + prompt_n must equal the submitted prompt,
        # and the cold/hit expectation is only asserted when the leg states it.
        if expect_hit is None:
            cf = [f for f in validate_completion_counts(
                cont["json"], len(extension), cold=True,
                expected_decode=args.decode_tokens)
                if "cache_n" not in f]
        else:
            cf = validate_completion_counts(cont["json"], len(extension),
                                            cold=not expect_hit,
                                            expected_decode=args.decode_tokens)
        if expect_cache_n is not None and isinstance(cache_n, int) and cache_n != expect_cache_n:
            cf.append(f"{phase}: cache_n={cache_n}, expected {expect_cache_n}")
        if isinstance(cache_n, int) and isinstance(prompt_n, int) and \
                cache_n + prompt_n != len(extension):
            cf.append(f"{phase}: cache_n({cache_n}) + prompt_n({prompt_n}) != "
                      f"submitted prompt ({len(extension)})")
        leg_failures += cf
        leg["validation_failures"] = leg_failures
        return leg, leg_failures

    # ---------------------------------------------------------------------
    # Leg D1: XXL snapshot saved with P_high, restored in a fresh process
    # with the SAME P/C, n_saved <= P. Must import with no prefill of the
    # imported prefix (cache_n == prefix, prompt_n == suffix only).
    # ---------------------------------------------------------------------
    # HONEST LABEL. This leg is SPLIT -> SPLIT below P, in a fresh process, disk
    # only. It is NOT the XXL PURE -> SPLIT import of §3.6: both the source and
    # the destination here carry the position-split placement, so this proves
    # the split envelope restore contract, not the XXL import. The XXL path is
    # leg D0 below. The two must never be reported as the same evidence.
    leg1_id = f"D1-p{p_high}"
    leg1_kind = "split_to_split_below_p"
    inst = None
    case_record: dict[str, Any] = {
        "case_id": leg1_id, "scenario": "D", "state": "starting",
        "kind": leg1_kind, "p": p_high, "ctx": args.split_ctx_high, "cache_ram": 0,
        "started_utc": utc_now(),
    }
    try:
        inst = start_split(leg1_id, p_high, args.split_ctx_high,
                           None, split_profile(args, qx, split=True))
        case_record.update({
            "command_argv": inst.command, "command_shell": shlex.join(inst.command),
            "env": {"LLAMA_KV_POSITION_SPLIT_P": str(p_high)},
            "server_log": inst.log_path.name,
            "slot_save_path": inst.slot_save_path.name,
        })
        append_jsonl(output_dir / "cases.jsonl", case_record)
        case_record["state"] = "ready"
        frozen = create_frozen_prompts(base_url, args, records_path, output_dir,
                                       leg1_id, frozen)
        write_json(output_dir / "run.json", run_meta)
        # Prefix shorter than P so the whole import lands in the LOCAL range.
        n_local = args.split_tokens_local
        prefix = frozen_prefix_ids(frozen, n_local, "scenario D1 local prompt")
        suffix = frozen["suffix_token_ids"]
        if len(prefix) != n_local:
            raise RuntimeError(f"prefix must hold {n_local} tokens, got {len(prefix)}")
        case_record["prefix_tokens"] = n_local
        case_record["suffix_tokens"] = len(suffix)
        case_record["n_saved_expected_le_p"] = True
        d1_filename = f"snap-{leg1_id}.bin"
        filename, save_info, sf = save_snapshot(inst, leg1_id, prefix, d1_filename,
                                                "D1-save")
        failures += sf
        case_record["save"] = save_info
        case_record["snapshot_filename"] = filename
        append_jsonl(output_dir / "cases.jsonl", case_record)
        if filename:
            # Fresh process, same P, ONLY the snapshot file on disk.
            case_record["state"] = "source_saved"
            case_record["source_exit_code"] = inst.stop()
            inst = None
            fresh_id = f"{leg1_id}-fresh"
            inst = start_split(fresh_id, p_high, args.split_ctx_high,
                               output_dir / f"slot-save-{leg1_id}",
                               split_profile(args, qx, split=True))
            case_record["fresh_case_id"] = fresh_id
            case_record["fresh_command_argv"] = inst.command
            case_record["fresh_p"] = p_high
            # The §3.6 claim. n_saved is the occupancy the SERVER wrote; it is
            # <= the prefill length because the on-disk frame keeps sealed
            # record groups (stage_groups/tail_groups), so the tail still in
            # stage is not part of the snapshot. The import therefore covers
            # n_saved tokens, and the §3.6 requirement is that NONE of those
            # n_saved tokens is re-prefilled:
            #     cache_n == n_saved            (all snapshot tokens reused)
            #     prompt_n == (prefix - n_saved) + suffix   (only the unsealed
            #         remainder plus the new suffix is computed)
            n_saved = save_info.get("n_saved")
            case_record["n_saved"] = n_saved
            case_record["p"] = p_high
            case_record["n_saved_le_p"] = isinstance(n_saved, int) and n_saved <= p_high
            if not case_record["n_saved_le_p"]:
                failures.append(f"D1: n_saved={n_saved} is not <= P={p_high}")
            leg, lf = restore_and_extend(inst, fresh_id, filename, prefix + suffix,
                                         "D1-restore", expect_restore=True,
                                         expect_cache_n=None)
            failures += lf
            case_record["restore"] = leg
            n_saved = save_info.get("n_saved")
            cn = leg.get("cache_n")
            pn = leg.get("prompt_n")
            unsealed = len(prefix) - n_saved if isinstance(n_saved, int) else None
            expected_pn = unsealed + len(suffix) if unsealed is not None else None
            if isinstance(cn, int) and isinstance(n_saved, int) and cn != n_saved:
                failures.append(f"D1: cache_n({cn}) != n_saved({n_saved}); the imported "
                                "snapshot tokens were not fully reused")
            if isinstance(pn, int) and expected_pn is not None and pn != expected_pn:
                failures.append(f"D1: prompt_n({pn}) != unsealed({unsealed})+suffix({len(suffix)})"
                                f"={expected_pn}; the imported prefix was re-prefilled")
            case_record["import_reuse"] = {
                "cache_n": cn, "prompt_n": pn,
                "n_saved": n_saved, "n_restored": (leg.get("restore_counts") or {}).get("n_restored"),
                "p": p_high, "prefix_len": len(prefix), "suffix_len": len(suffix),
                "unsealed_not_in_snapshot": unsealed,
                "expected_cache_n": n_saved, "expected_prompt_n": expected_pn,
                "snapshot_tokens_reused_entirely":
                    isinstance(cn, int) and isinstance(n_saved, int) and cn == n_saved,
                "no_prefill_of_imported_prefix":
                    isinstance(pn, int) and expected_pn is not None and pn == expected_pn,
            }
    except Exception as exc:  # noqa: BLE001
        failures.append(f"D1: {exc}")
        case_record["state"] = "failed"
        case_record["error"] = str(exc)
    finally:
        if inst is not None:
            case_record["server_exit_code"] = inst.stop()
            delta = inst.read_log_delta()
            case_record["log_evidence"] = inst.log_evidence(delta)
            case_record["log_envelope_errors"] = [
                line.strip() for line in delta.splitlines()
                if "position_split" in line or "mixed state" in line.lower()]
        case_record.setdefault("state", "unknown")
        case_record["state"] = "failed" if failures else "complete"
        append_jsonl(output_dir / "cases.jsonl", case_record)
        runs.append(case_record)
        record_leg("D1", case_record["state"], case_id=leg1_id,
                   p=p_high, failures=failures)
    runs_state = case_record["state"]
    log(f"D1 state={runs_state}")

    # ---------------------------------------------------------------------
    # D7: controlled reference for the §3.6 "compare recurrent state and NEXT
    # LOGITS, never approve on text/timings alone" requirement.
    #
    # In a fresh process with an EMPTY slot (no snapshot touched, no restore),
    # the SAME prompt is sent twice with the same seed and the second run must
    # reproduce the first run's per-token next-token distributions exactly.
    # This establishes the reference against which any later leg compares, and
    # it fails loudly if the server is not deterministic, so a later
    # logits-equality pass can never be a coincidence of text matching.
    # ---------------------------------------------------------------------
    d7_id = f"D7-logits-reference-p{p_high}"
    d7_failures: list[str] = []
    d7_inst = None
    d7_ref: dict[str, Any] = {}
    d7_rep: dict[str, Any] = {}
    try:
        d7_inst = start_split(d7_id, p_high, args.split_ctx_high, None,
                              split_profile(args, qx, split=True))
        ref_tokens = frozen_prefix_ids(frozen, args.split_tokens_local, "scenario D7 reference")
        payload = {"prompt": ref_tokens, "n_predict": args.decode_tokens,
                   "id_slot": 0, "cache_prompt": True, "return_tokens": True,
                   "temperature": 0, "seed": args.seed, "ignore_eos": True,
                   "n_probs": args.n_probs}
        first = http_request(base_url, "POST", "/completion", payload,
                             args.request_timeout, records_path,
                             args.telemetry_device, d7_id, "D7-reference")
        # Erase so the second request recomputes from scratch, not from a
        # warm prefix: this is what makes it a real reproducibility check.
        http_request(base_url, "POST", "/slots/0?action=erase", {},
                     args.request_timeout, records_path, args.telemetry_device,
                     d7_id, "D7-erase")
        second = http_request(base_url, "POST", "/completion", payload,
                              args.request_timeout, records_path,
                              args.telemetry_device, d7_id, "D7-repeat")
        fb, sb = first["json"], second["json"]
        d7_ref = {"tokens": fb.get("tokens"), "probs": fb.get("completion_probabilities")}
        d7_rep = {"tokens": sb.get("tokens"), "probs": sb.get("completion_probabilities")}
        if not deterministic_match(fb.get("tokens"), sb.get("tokens")):
            d7_failures.append("D7: generated token IDs differ between identical "
                               "reference runs; logits comparison would be meaningless")
        if not compare_completion_probabilities(fb.get("completion_probabilities"),
                                                sb.get("completion_probabilities")):
            d7_failures.append("D7: next-token probabilities are not reproducible; "
                               "the reference is unusable for state comparison")
        if not isinstance(fb.get("completion_probabilities"), list) or \
                not fb.get("completion_probabilities"):
            d7_failures.append("D7: no completion_probabilities returned; "
                               "--n-probs evidence is required")
    except Exception as exc:  # noqa: BLE001
        d7_failures.append(f"D7: {exc}")
    finally:
        d7_exit = d7_inst.stop() if d7_inst is not None else None
    d7_entry = record_leg(d7_id, "failed" if d7_failures else "complete",
                          kind="controlled_logits_reference",
                          expectation="same prompt + seed reproduces identical tokens "
                                      "AND identical next-token logprobs in a fresh "
                                      "process, so later state comparisons are meaningful",
                          p=p_high, ctx=args.split_ctx_high,
                          reference_tokens=d7_ref.get("tokens"),
                          repeat_tokens=d7_rep.get("tokens"),
                          logits_reproducible=compare_completion_probabilities(
                              d7_ref.get("probs"), d7_rep.get("probs")),
                          server_exit_code=d7_exit,
                          validation_failures=d7_failures)
    runs.append(d7_entry)
    log(f"{d7_id} state={d7_entry['state']} logits_reproducible="
        f"{d7_entry['logits_reproducible']}")

    # ---------------------------------------------------------------------
    # D0: the §3.6 XXL path. PURE KVarN (XXL) snapshot built in its OWN
    # process with NO position split and NO remote layers (kv_layout_known=
    # false), then RESTORED in a brand-new process that DOES carry the split
    # placement -- so there is no live PURE context anywhere, which is exactly
    # the "fresh process, only the XXL snapshot on disk" requirement.
    #
    # Two sub-legs:
    #   D0a n_saved <= P : §3.6 requires an EXPLICIT import that fills the
    #                     local range and leaves the overflow empty.
    #   D0b n_saved >  P : §3.6 requires REJECTION of the direct import with
    #                     the destination intact and an explicit fallback.
    # Each asserts its own occupancy precondition BEFORE asserting anything
    # else, so a fixture that does not actually cross P can never be reported
    # as passing the above/below-P contract.
    # ---------------------------------------------------------------------
    for sub, want_above, n_prefill in (
            ("D0a-pure-xxl-below-p", False, args.split_tokens_local),
            ("D0b-pure-xxl-above-p", True, p_high + 256)):
        d0_id = f"{sub}-p{p_high}"
        d0_failures: list[str] = []
        d0_inst = None
        d0_dir = output_dir / f"slot-save-{d0_id}"
        d0_dir.mkdir(parents=True, exist_ok=True)
        d0_file = d0_dir / f"snap-{d0_id}.bin"
        d0_info: dict[str, Any] = {}
        d0_leg: dict[str, Any] = {}
        n_pre = max(n_prefill, args.split_tokens_local)
        if n_pre + args.suffix_tokens + args.decode_tokens >= args.split_ctx_high:
            d0_failures.append(
                f"{d0_id}: prefill {n_pre} does not fit ctx={args.split_ctx_high}")
        else:
            try:
                # SOURCE process: PURE XXL, no split, no remote layers.
                src = start_split(f"{d0_id}-pure", 0, args.split_ctx_high, d0_dir,
                                  split_profile(args, qx, split=False,
                                                pure_source=True))
                d0_info["source_command"] = shlex.join(src.command)
                d0_info["source_env"] = "LLAMA_KV_POSITION_SPLIT_P explicitly cleared (PURE XXL)"
                pre_tokens = frozen_prefix_ids(frozen, n_pre, d0_id)
                _, d0_info, sf = save_snapshot(src, f"{d0_id}-pure", pre_tokens,
                                               f"snap-{d0_id}.bin", f"{sub}-save")
                d0_failures += sf
                d0_info["prefill_tokens"] = n_pre
                d0_info["p"] = p_high
                src_exit = src.stop()
                d0_info["source_exit_code"] = src_exit
                n_occ = d0_info.get("n_saved")
                d0_info["occupancy_gt_p"] = isinstance(n_occ, int) and n_occ > p_high
                d0_info["occupancy_le_p"] = isinstance(n_occ, int) and n_occ <= p_high
                # Precondition first: the leg must really be above/below P.
                if want_above and not d0_info["occupancy_gt_p"]:
                    d0_failures.append(
                        f"{d0_id}: fixture occupancy n_saved={n_occ} is not > P={p_high}; "
                        "the above-P contract would be untested")
                if not want_above and not d0_info["occupancy_le_p"]:
                    d0_failures.append(
                        f"{d0_id}: fixture occupancy n_saved={n_occ} is not <= P={p_high}; "
                        "the below-P import contract would be untested")
                if not d0_file.is_file():
                    d0_failures.append(f"{d0_id}: XXL snapshot {d0_file} was not written")
            except Exception as exc:  # noqa: BLE001
                d0_failures.append(f"{d0_id}: PURE source build failed: {exc}")

        # §3.6: below P must import; above P must REJECT the direct import.
        expect_import = not want_above
        if not d0_failures:
            # DESTINATION process: fresh, split placement, NO live PURE ctx.
            try:
                d0_inst = start_split(f"{d0_id}-split", p_high, args.split_ctx_high,
                                      d0_dir, split_profile(args, qx, split=True))
                d0_info["dest_command"] = shlex.join(d0_inst.command)
                d0_leg, lf = restore_and_extend(
                    d0_inst, f"{d0_id}-split", d0_file.name,
                    pre_tokens + suffix, d0_id,
                    expect_restore=expect_import, expect_cache_n=None,
                    expect_hit=True if expect_import else False)
                d0_failures += lf
                n_occ = d0_info.get("n_saved")
                cn = d0_leg.get("cache_n")
                if expect_import:
                    # Import with no prefill of the imported prefix, and the
                    # overflow range left EMPTY because n_saved <= P.
                    unsealed = n_pre - n_occ if isinstance(n_occ, int) else None
                    expected_pn = unsealed + len(suffix) if unsealed is not None else None
                    d0_info["import"] = {
                        "cache_n": cn, "prompt_n": d0_leg.get("prompt_n"),
                        "expected_cache_n": n_occ, "expected_prompt_n": expected_pn,
                        "overflow_left_empty": isinstance(n_occ, int) and n_occ <= p_high,
                        "no_prefill_of_imported_prefix":
                            isinstance(d0_leg.get("prompt_n"), int) and
                            expected_pn is not None and
                            d0_leg.get("prompt_n") == expected_pn,
                    }
                    if not d0_info["import"]["no_prefill_of_imported_prefix"]:
                        d0_failures.append(
                            f"{d0_id}: imported prefix was re-prefilled "
                            f"(prompt_n={d0_leg.get('prompt_n')} != {expected_pn})")
                else:
                    # Above P: direct import must be refused and nothing reused.
                    # The refusal must be ATTRIBUTABLE to n_saved>P. If the same
                    # generic envelope-incompatibility reason also rejects the
                    # below-P case, then above/below P cannot be distinguished
                    # and this leg proves nothing about the P boundary: it is
                    # recorded inconclusive, never as a pass.
                    log_delta = d0_inst.read_log_delta()
                    reason_line = refusal_reason_from_log(log_delta)
                    p_refusal = pure_kvarn_import_p_refusal(reason_line, p_high)
                    p_specific = bool(p_refusal and p_refusal["p_attributable"])
                    d0_info["rejection"] = {
                        "cache_n": cn, "cache_n_is_zero": cn == 0,
                        "restore_rejected":
                            "http_status" in str(d0_leg.get("restore_response")),
                        "destination_intact": cn == 0,
                        "reject_reason": reason_line,
                        "p_refusal_parse": p_refusal,
                        "p_attributable": p_specific,
                    }
                    if cn != 0:
                        d0_failures.append(
                            f"{d0_id}: n_saved>P but cache_n={cn}; the direct import "
                            "was NOT rejected")
                    if not p_specific:
                        # NOT a pass: the refusal is generic, so it does not
                        # demonstrate the n_saved>P rule.
                        d0_failures.append(
                            f"{d0_id}: refusal reason is not attributable to P "
                            f"(got: {reason_line!r}); above/below P is "
                            "indistinguishable, so the P boundary is untested")
            except Exception as exc:  # noqa: BLE001
                d0_failures.append(f"{d0_id}: split restore failed: {exc}")
        exit_code = d0_inst.stop() if d0_inst is not None else None
        delta = d0_inst.read_log_delta() if d0_inst is not None else ""
        if d0_file.is_file():
            d0_info["snapshot_sha256"] = sha256_file(d0_file)
        entry = record_leg(d0_id, "failed" if d0_failures else "complete",
                           kind="pure_xxl_to_split",
                           expectation=("below P: explicit import, overflow empty, "
                                        "no prefill of the imported prefix" if expect_import
                                        else "above P: direct import rejected, "
                                             "destination intact, fallback explicit"),
                           p=p_high, ctx=args.split_ctx_high,
                           restore_response=str(d0_leg.get("restore_response"))[:300],
                           cache_n=d0_leg.get("cache_n"),
                           prompt_n=d0_leg.get("prompt_n"),
                           n_restored=(d0_leg.get("restore_counts") or {}).get("n_restored"),
                           detail=d0_info, server_exit_code=exit_code,
                           log_evidence=[line.strip() for line in delta.splitlines()
                                         if "position_split" in line or
                                         "mixed state" in line.lower() or
                                         "kv_layout" in line.lower()][:40],
                           validation_failures=d0_failures)
        runs.append(entry)
        log(f"{d0_id} state={entry['state']} n_saved={d0_info.get('n_saved')} P={p_high}")

    # ---------------------------------------------------------------------
    # D8: REAL DISCOVERY on the index. §3.6 asks for the XXL snapshot to be
    # discovered by the destination process, with no manual restore endpoint
    # and no live PURE context anywhere.
    #
    #   D8a below P : the PURE source auto-publishes a unified snapshot into a
    #                 shared --slot-save-path; the source process is stopped;
    #                 a BRAND-NEW split process is started on the SAME path and
    #                 a plain /completion must be served by discovery +
    #                 the transactional pure import. No /slots/:id?action=
    #                 restore is ever issued.
    #   D8b above P : identical, but the source occupancy is > P, so discovery
    #                 may select the file but the import gate must refuse and
    #                 the destination must fall back with the destination
    #                 intact (no partial materialization).
    #   D8c mismatch : the pure source uses a different P/ctx, so the stored
    #                 layout no longer matches the destination and the file
    #                 must be ignored at index time (safe miss, no restore).
    #   D8d invalid : the unified snapshot is corrupted in place; the index
    #                 must skip it and serve a normal prefill.
    # ---------------------------------------------------------------------
    for sub, want_above, n_prefill, expect_restored in (
            ("D8a-pure-xxl-discovery-below-p", False, max(args.split_tokens_local, 256), True),
            ("D8b-pure-xxl-discovery-above-p", True, p_high + 256, False),
            ("D8c-pure-xxl-discovery-mismatch", False, max(args.split_tokens_local, 256), False),
            ("D8d-pure-xxl-discovery-invalid", False, max(args.split_tokens_local, 256), False)):
        d8_id = f"{sub}-p{p_high}"
        d8_failures: list[str] = []
        d8_inst = None
        d8_dir = output_dir / f"slot-save-{d8_id}"
        d8_dir.mkdir(parents=True, exist_ok=True)
        d8_info: dict[str, Any] = {}
        n_pre = max(n_prefill, args.split_tokens_local)
        pure_prof = split_profile(args, qx, split=False, pure_source=True)
        if n_pre + args.suffix_tokens + args.decode_tokens >= args.split_ctx_high:
            d8_failures.append(
                f"{d8_id}: prefill {n_pre} does not fit ctx={args.split_ctx_high}")
        else:
            try:
                # SOURCE process: PURE XXL, unified snapshot AUTO-published.
                # No explicit save endpoint: the index must find the file that
                # the destination publishes on its own.
                src = start_split(f"{d8_id}-pure", 0, args.split_ctx_high, d8_dir,
                                  pure_prof + ["--slot-save-auto"])
                d8_inst = src
                d8_info["source_command"] = shlex.join(src.command)
                d8_info["source_env"] = ("LLAMA_KV_POSITION_SPLIT_P omitted (PURE XXL); "
                                         "unified snapshot auto-published")
                pre_tokens = frozen_prefix_ids(frozen, n_pre, d8_id)
                saved = http_request(
                    base_url, "POST", "/completion",
                    {"prompt": pre_tokens, "n_predict": args.decode_tokens,
                     "id_slot": 0, "cache_prompt": True, "return_tokens": True,
                     "temperature": 0, "seed": args.seed, "ignore_eos": True},
                    # No n_probs: the saved-logits fast path that lets the
                    # destination emit the first token without re-prefilling is
                    # only armed when the request asks for no next-token
                    # probabilities. With n_probs set the server correctly falls
                    # back to clearing the restored state and re-prefilling.
                    args.request_timeout, records_path, args.telemetry_device,
                    d8_id, f"{sub}-autosave")
                d8_failures += validate_completion_counts(
                    saved["json"], len(pre_tokens), cold=True,
                    expected_decode=args.decode_tokens)
                # The source must have retired the slot so the auto-save runs on
                # release; a /slots/0?action=erase is a local reset, not a
                # restore, so it does not reintroduce the manual restore path.
                http_request(base_url, "POST", "/slots/0?action=erase", {},
                             args.request_timeout, records_path,
                             args.telemetry_device, d8_id, f"{sub}-source-release")
                published = sorted(q for q in d8_dir.glob("*.bin") if q.is_file())
                d8_info["published_unified"] = [q.name for q in published]
                src_exit = src.stop()
                d8_inst = None
                d8_info["source_exit_code"] = src_exit
                if not published:
                    d8_failures.append(f"{d8_id}: PURE source published no unified "
                                       f"snapshot into {d8_dir}")
                snap = published[0] if published else None
                if snap is not None:
                    if sub.startswith("D8d"):
                        # Corrupt EVERY published payload in place, keeping the
                        # names/index keys intact so the INDEX path is what has to
                        # reject them. Corrupting only one file would leave the
                        # other selectable and prove nothing.
                        for target in published:
                            with open(target, "r+b") as fh:
                                fh.seek(target.stat().st_size // 2)
                                fh.write(b"\x00" * 4096)
                        d8_info["corrupted_files"] = len(published)
                        d8_info["corrupted_bytes_each"] = 4096
                    if sub.startswith("D8c"):
                        # Keep the pure layout but publish it under a snapshot
                        # whose stored layout can never match the destination.
                        pass

                # DESTINATION process: BRAND NEW, split placement, same path,
                # no live PURE context and NO manual restore endpoint use.
                # The destination MUST also have auto-cache on, otherwise
                # auto_cache_enabled() is false and no index scan happens at all.
                dst_prof = split_profile(args, qx, split=True) + ["--slot-save-auto"]
                if sub.startswith("D8c"):
                    # Genuine index-time layout mismatch: a different RoPE
                    # base is a layout field that the PURE->split predicate
                    # does NOT normalize away, so the stored file must be
                    # ignored while scanning instead of being restored.
                    dst_prof += ["--rope-freq-base", "1500000"]
                dst = start_split(f"{d8_id}-split", p_high, args.split_ctx_high,
                                  d8_dir, dst_prof)
                d8_inst = dst
                d8_info["destination_command"] = shlex.join(dst.command)
                d8_info["manual_restore_endpoint_used"] = False
                served = http_request(
                    base_url, "POST", "/completion",
                    {"prompt": pre_tokens, "n_predict": args.decode_tokens,
                     "id_slot": 0, "cache_prompt": True, "return_tokens": True,
                     "temperature": 0, "seed": args.seed, "ignore_eos": True},
                    # No n_probs: the saved-logits fast path that lets the
                    # destination emit the first token without re-prefilling is
                    # only armed when the request asks for no next-token
                    # probabilities. With n_probs set the server correctly falls
                    # back to clearing the restored state and re-prefilling.
                    args.request_timeout, records_path, args.telemetry_device,
                    d8_id, f"{sub}-discovery")
                # The slot stats are reset once the task completes, so cache_n /
                # prompt_n come from the response timings and the selection /
                # cache_source evidence comes from the destination server log,
                # which is the authoritative record of the disk restore.
                st_timings = served["json"].get("timings")
                st_timings = st_timings if isinstance(st_timings, dict) else {}
                d8_failures += validate_completion_counts(
                    served["json"], len(pre_tokens), cold=not expect_restored,
                    expected_decode=args.decode_tokens)
                log_text = ""
                if dst.log_path.is_file():
                    log_text = dst.log_path.read_text(errors="replace")
                reused = [ln.strip() for ln in log_text.splitlines()
                          if "auto-restore: reused" in ln or
                             "auto-restore: converted cached snapshot" in ln]
                imported = [ln.strip() for ln in log_text.splitlines()
                            if "importing PURE KVarN snapshot" in ln]
                d8_info["n_prompt_tokens"] = len(pre_tokens)
                d8_info["cache_n"] = st_timings.get("cache_n")
                d8_info["prompt_n"] = st_timings.get("prompt_n")
                d8_info["predicted_n"] = st_timings.get("predicted_n")
                d8_info["cache_source"] = "disk" if reused else None
                d8_info["cache_reason"] = ("unified_snapshot_restore" if reused
                                           else None)
                d8_info["restore_log_evidence"] = reused[:4]
                d8_info["pure_import_log_evidence"] = imported[:4]
                # Discovery+import must reuse the whole imported prefix, so the
                # destination prefills NOTHING of it.
                d8_info["no_prefill_of_imported_prefix"] = (
                    st_timings.get("prompt_n") == 0 and
                    st_timings.get("cache_n") == len(pre_tokens))
                restored = bool(reused) and d8_info["no_prefill_of_imported_prefix"]
                if expect_restored and not restored:
                    d8_failures.append(
                        f"{d8_id}: discovery did not restore the PURE snapshot "
                        f"(cache_source={d8_info['cache_source']!r}, "
                        f"cache_n={d8_info['cache_n']}, prompt_n={d8_info['prompt_n']}, "
                        f"import_log={len(imported)}, reuse_log={len(reused)})")
                if not expect_restored and restored:
                    d8_failures.append(
                        f"{d8_id}: a snapshot that must be rejected was restored "
                        f"(cache_source={d8_info['cache_source']!r}, "
                        f"cache_reason={d8_info['cache_reason']!r})")
                if expect_restored and not d8_info["no_prefill_of_imported_prefix"]:
                    d8_failures.append(
                        f"{d8_id}: restore re-prefilled the imported prefix "
                        f"(cache_n={d8_info['cache_n']}, "
                        f"prompt_n={d8_info['prompt_n']})")
                if want_above and restored:
                    d8_failures.append(
                        f"{d8_id}: occupancy above P={p_high} was restored; the "
                        f"import gate must refuse without truncation")
                d8_info["restore_refused_above_p"] = bool(want_above and not restored)
                d8_exit = dst.stop()
                d8_inst = None
                d8_info["destination_exit_code"] = d8_exit
                if capture_enabled and expect_restored:
                    # A cold split baseline has the exact destination layout,
                    # including any generated token that crosses P. Its private
                    # directory contains no PURE snapshot or resident cache.
                    baseline = start_split(f"{d8_id}-baseline", p_high,
                                           args.split_ctx_high, None,
                                           split_profile(args, qx, split=True))
                    d8_inst = baseline
                    d8_info["full_logit_baseline_command"] = shlex.join(baseline.command)
                    control = http_request(
                        base_url, "POST", "/completion",
                        {"prompt": pre_tokens, "n_predict": args.decode_tokens,
                         "id_slot": 0, "cache_prompt": True, "return_tokens": True,
                         "temperature": 0, "seed": args.seed, "ignore_eos": True},
                        args.request_timeout, records_path, args.telemetry_device,
                        f"{d8_id}-baseline", f"{sub}-cold-full-logit-baseline")
                    d8_failures += validate_completion_counts(
                        control["json"], len(pre_tokens), cold=True,
                        expected_decode=args.decode_tokens)
                    d8_info["full_logit_baseline_timings"] = control["json"].get("timings")
                    d8_info["full_logit_baseline_exit_code"] = baseline.stop()
                    d8_inst = None
                    generated = served["json"].get("tokens")
                    if not isinstance(generated, list) or not generated or \
                            not isinstance(generated[0], int):
                        raise RuntimeError("D8a full-logit proof needs actual generated token IDs")
                    checkpoint = len(pre_tokens) + 1
                    evaluated = pre_tokens + [generated[0]]
                    restore_parity = compare_full_logit_checkpoint(
                        baseline.capture_dir, baseline.case_id,
                        dst.capture_dir, dst.case_id, checkpoint, campaign_id,
                        evaluated, layout_relation="same")
                    d8_info["full_logit_restore_parity"] = restore_parity
                    if restore_parity["state"] != "complete":
                        d8_failures.append(f"{d8_id}: full-vocabulary restore parity failed: "
                                           f"{restore_parity}")
                    if checkpoint <= p_high:
                        source_parity = compare_full_logit_checkpoint(
                            src.capture_dir, src.case_id, dst.capture_dir, dst.case_id,
                            checkpoint, campaign_id, evaluated, layout_relation="different")
                        d8_info["full_logit_pure_source_parity"] = source_parity
                        if source_parity["state"] != "complete":
                            d8_failures.append(f"{d8_id}: PURE/below-P full-vocabulary parity failed: "
                                               f"{source_parity}")
                    else:
                        d8_info["full_logit_pure_source_parity"] = {
                            "state": "not_applicable", "tokens_n": checkpoint,
                            "reason": "checkpoint exceeds P; PURE and split use different KV formats, so only identical split baseline proves restore parity"}
            except Exception as exc:  # noqa: BLE001
                d8_failures.append(f"{d8_id}: {exc}")
                if d8_inst is not None:
                    d8_inst.stop()
        entry = record_leg(d8_id, "failed" if d8_failures else "complete",
                           kind="index_discovery_pure_to_split",
                           expectation="a fresh split process with ONLY the PURE "
                                       "unified snapshot on disk is served by "
                                       "discovery + transactional import, with no "
                                       "manual restore endpoint and no live PURE "
                                       "context; above-P/mismatch/invalid are "
                                       "refused with the destination intact",
                           p=p_high, ctx=args.split_ctx_high,
                           manual_restore_endpoint_used=False,
                           cache_source=d8_info.get("cache_source"),
                           cache_reason=d8_info.get("cache_reason"),
                           restore_log_evidence=d8_info.get("restore_log_evidence"),
                           pure_import_log_evidence=d8_info.get(
                               "pure_import_log_evidence"),
                           corrupted_files=d8_info.get("corrupted_files"),
                           cache_n=d8_info.get("cache_n"),
                           prompt_n=d8_info.get("prompt_n"),
                           predicted_n=d8_info.get("predicted_n"),
                           no_prefill_of_imported_prefix=d8_info.get(
                               "no_prefill_of_imported_prefix"),
                           n_prompt_tokens=d8_info.get("n_prompt_tokens"),
                           published_unified=d8_info.get("published_unified"),
                           restore_refused_above_p=d8_info.get("restore_refused_above_p"),
                           source_command=d8_info.get("source_command"),
                           destination_command=d8_info.get("destination_command"),
                           full_logit_restore_parity=d8_info.get("full_logit_restore_parity"),
                           full_logit_pure_source_parity=d8_info.get("full_logit_pure_source_parity"),
                           full_logit_baseline_command=d8_info.get("full_logit_baseline_command"),
                           full_logit_baseline_timings=d8_info.get("full_logit_baseline_timings"),
                           full_logit_capture_scope=("D8a: one decoded checkpoint versus cold split baseline; other D processes captured for separate qualified comparisons"
                                                     if capture_enabled else None),
                           validation_failures=d8_failures)
        runs.append(entry)
        log(f"{d8_id} state={entry['state']} cache_source={d8_info.get('cache_source')}")

    # ---------------------------------------------------------------------
    # D9 (destination intactness after a refused import) was REMOVED from this
    # gate: it issued two separate HTTP requests, and the slot is request-scoped,
    # so the server legitimately clears the slot when a divergent prompt arrives
    # ("adaptive cache common prefix=0"). The destination was therefore never
    # populated at refusal time and the leg could not measure what it claimed -
    # cache_n=0 on an already-empty slot proves nothing.
    #
    # It is now covered, in the correct cycle and against the real production
    # restore path, by tests/test-kv-mixed-destination-intact.cpp, which populates
    # the destination inside the test, calls server_prompt_cache::restore, and
    # measures bytes/positions/tokens/checkpoints and the COMPLETE next-token
    # logits IN PLACE (no clear, no re-prefill) against an untouched control.
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # D4a is the positive inverse below-P case above. Remaining negative/reject
    # legs each use their own fixture when occupancy or source layout requires it.
    # and its OWN fresh process. They only run when D1 produced a usable
    # snapshot; otherwise they are recorded as NOT EXECUTED with the reason
    # (never as passed).
    # ---------------------------------------------------------------------
    source_dir = output_dir / f"slot-save-{leg1_id}"
    expected_snapshot = source_dir / f"snap-{leg1_id}.bin"
    snapshot_files = sorted(p for p in source_dir.glob("*") if p.is_file()) \
        if source_dir.is_dir() else []
    snapshot = expected_snapshot if expected_snapshot.is_file() else (snapshot_files[0] if snapshot_files else None)

    # leg, P, ctx, expect_restore, expect_hit, own_fixture, precondition, reason
    # own_fixture=None -> reuse D1's snapshot. own_fixture=(min_occupancy,
    # bound_name, bound) -> build an OWN snapshot whose persisted occupancy must
    # be strictly greater than `bound`, asserted BEFORE the restore is tested.
    # n_ctx is NOT occupancy: a destination with a smaller C does not by itself
    # make the snapshot's occupancy exceed the destination, so D5 needs a
    # fixture whose OWN occupancy really is larger than the destination.
    negative_legs: list[tuple[str, int, int, bool, bool | None,
                              tuple[Any, ...] | None, str]] = [
        ("D2-different-p", args.split_p_local, args.split_ctx_high, False, False,
         None,
         "same snapshot restored with a different P must be rejected (cache miss)"),
        # Occupancy strictly ABOVE P, proven with an independent snapshot whose
        # sealed occupancy really exceeds P. D1's n_saved=128 cannot prove this.
        ("D3-above-p", p_high, args.split_ctx_high, True, None,
         ("above-p", "P", p_high),
         "same P/C with persisted occupancy above P must restore with the "
         "overflow range occupied"),
        ("D4b-occupied-overflow-to-pure", 0, args.split_ctx_high, False, False,
         ("above-pure", "P", p_high, p_high),
         "split snapshot occupancy above P must be rejected by a pure destination"),
        # Occupancy strictly GREATER than the destination capacity. n_ctx is
        # NOT occupancy: the source context is built LARGE (so the fixture can
        # hold more tokens than the destination ever could) and only the
        # DESTINATION ctx is smaller. The assertion is on the persisted
        # occupancy versus the destination capacity, which is the real §3.6
        # condition ("ocupação maior que a capacidade destino").
        ("D5-insufficient-capacity", p_high, args.split_ctx_local, False, False,
         ("above-dest", "dest_ctx", args.split_ctx_local),
         "snapshot occupancy larger than the destination capacity must be rejected"),
    ]

    if runs_state != "complete" or snapshot is None:
        reason = (f"source snapshot unavailable (D1 state={runs_state}, "
                  f"file={'yes' if snapshot else 'no'})")
        record_leg("D4a-split-local-to-pure", "not_executed", reason=reason)
        runs.append({"leg": "D4a-split-local-to-pure", "state": "not_executed", "reason": reason})
        for leg_name, _p, _c, _exp, _hit, _pre, why in negative_legs:
            record_leg(leg_name, "not_executed", reason=reason, expectation=why)
        log(f"negative legs NOT EXECUTED: {reason}")
        return {"scenario": "D", "runs": runs}

    # D4a: the actual below-P inverse import, split-local source to PURE target.
    # Reuse D1's real snapshot and compare it against a fresh same-P/C split
    # restore from the exact same bytes when full-vocabulary capture is enabled.
    d4a_id = "D4a-split-local-to-pure"
    d4a_failures: list[str] = []
    d4a_info: dict[str, Any] = {"source_case_id": leg1_id, "original_snapshot": str(snapshot)}
    d4a_info["source_p"] = p_high
    d4a_info["target_p"] = 0
    d4a_info["decode_tokens_requested"] = args.decode_tokens
    d4a_dir = output_dir / f"slot-save-{d4a_id}"
    d4a_dir.mkdir(parents=True, exist_ok=True)
    d4a_file = d4a_dir / snapshot.name
    d4a_target: ServerInstance | None = None
    d4a_control: ServerInstance | None = None
    original_sha = sha256_file(snapshot)
    d1_case = next((item for item in runs if item.get("case_id") == leg1_id), {})
    n_saved_d1 = next((item.get("save", {}).get("n_saved") for item in runs
                       if item.get("case_id") == leg1_id), None)
    d4a_info["source_command_argv"] = d1_case.get("command_argv")
    d4a_info["source_restore_command_argv"] = d1_case.get("fresh_command_argv")
    d4a_failures += validate_split_to_pure_extent(n_saved_d1, p_high, expect_import=True)
    try:
        shutil.copy2(snapshot, d4a_file)
        copy_sha = sha256_file(d4a_file)
        d4a_info.update({"source_snapshot_sha256": original_sha,
                         "target_snapshot_sha256_before": copy_sha,
                         "n_saved": n_saved_d1, "p_high": p_high})
        if copy_sha != original_sha:
            d4a_failures.append(f"{d4a_id}: target snapshot copy is not byte-identical to D1 source")
        prefix_d4a = frozen_prefix_ids(frozen, args.split_tokens_local, d4a_id)
        suffix_d4a = frozen["suffix_token_ids"]
        extension_d4a = prefix_d4a + suffix_d4a
        d4a_info["prefix_tokens"] = len(prefix_d4a)
        d4a_info["suffix_tokens"] = len(suffix_d4a)
        d4a_info["extension_tokens_sha256"] = token_ids_sha256(extension_d4a)
        target_id = f"{d4a_id}-pure-target"
        d4a_target = start_split(target_id, 0, args.split_ctx_high, d4a_dir,
                                 split_profile(args, qx, split=False))
        d4a_info["target_command_argv"] = d4a_target.command
        d4a_info["target_command_shell"] = shlex.join(d4a_target.command)
        target_leg, target_failures = restore_and_extend(
            d4a_target, target_id, d4a_file.name, extension_d4a,
            d4a_id, expect_restore=True, expect_cache_n=n_saved_d1, expect_hit=True)
        d4a_failures += target_failures
        d4a_info["target"] = target_leg
        n_restored = (target_leg.get("restore_counts") or {}).get("n_restored")
        if n_restored != n_saved_d1:
            d4a_failures.append(f"{d4a_id}: n_restored={n_restored} != n_saved={n_saved_d1}")
        expected_prompt_n = (len(prefix_d4a) - n_saved_d1 + len(suffix_d4a)
                             if isinstance(n_saved_d1, int) else None)
        d4a_info["expected_prompt_n"] = expected_prompt_n
        if target_leg.get("prompt_n") != expected_prompt_n:
            d4a_failures.append(f"{d4a_id}: prompt_n={target_leg.get('prompt_n')} != expected unsealed+suffix={expected_prompt_n}")
        if target_leg.get("cache_source") != "live":
            d4a_failures.append(f"{d4a_id}: target cache_source={target_leg.get('cache_source')!r}, expected 'live'")
        d4a_info["target_capture_dir"] = str(d4a_target.capture_dir) if d4a_target.capture_dir else None
        d4a_info["target_exit_code"] = d4a_target.stop()
        d4a_info["target_log_evidence"] = d4a_target.log_evidence(d4a_target.read_log_delta())
        target_capture_dir = d4a_target.capture_dir
        d4a_target = None
        target_sha_after = sha256_file(d4a_file)
        d4a_info["target_snapshot_sha256_after"] = target_sha_after
        if target_sha_after != copy_sha:
            d4a_failures.append(f"{d4a_id}: target mutated the copied source snapshot")

        if capture_enabled and not d4a_failures:
            control_dir = output_dir / f"slot-save-{d4a_id}-split-control"
            control_dir.mkdir(parents=True, exist_ok=True)
            control_file = control_dir / snapshot.name
            shutil.copy2(snapshot, control_file)
            control_sha_before = sha256_file(control_file)
            d4a_info["source_snapshot_sha256_for_control"] = original_sha
            if control_sha_before != original_sha:
                d4a_failures.append(f"{d4a_id}: control snapshot copy is not byte-identical to D1 source")
            control_id = f"{d4a_id}-split-control"
            d4a_control = start_split(control_id, p_high, args.split_ctx_high,
                                      control_dir, split_profile(args, qx, split=True))
            d4a_info["control_p"] = p_high
            d4a_info["control_command_argv"] = d4a_control.command
            d4a_info["control_command_shell"] = shlex.join(d4a_control.command)
            d4a_info["control_snapshot_sha256_before"] = control_sha_before
            control_leg, control_failures = restore_and_extend(
                d4a_control, control_id, control_file.name, extension_d4a,
                f"{d4a_id}-control", expect_restore=True,
                expect_cache_n=n_saved_d1, expect_hit=True)
            d4a_failures += control_failures
            d4a_info["control"] = control_leg
            if control_leg.get("cache_source") != "live":
                d4a_failures.append(f"{d4a_id}: control cache_source={control_leg.get('cache_source')!r}, expected 'live'")
            target_tokens = target_leg.get("generated_tokens")
            control_tokens = control_leg.get("generated_tokens")
            d4a_info["generated_tokens_equal"] = isinstance(target_tokens, list) and target_tokens == control_tokens
            if not d4a_info["generated_tokens_equal"] or not target_tokens:
                d4a_failures.append(f"{d4a_id}: target/control generated token IDs differ or are missing")
            d4a_info["control_capture_dir"] = str(d4a_control.capture_dir) if d4a_control.capture_dir else None
            d4a_info["control_exit_code"] = d4a_control.stop()
            d4a_info["control_log_evidence"] = d4a_control.log_evidence(d4a_control.read_log_delta())
            control_capture_dir = d4a_control.capture_dir
            d4a_control = None
            d4a_info["control_snapshot_sha256_after"] = sha256_file(control_file)
            if d4a_info["control_snapshot_sha256_after"] != control_sha_before:
                d4a_failures.append(f"{d4a_id}: control mutated the copied source snapshot")
            if target_tokens:
                checkpoint = len(prefix_d4a) + len(suffix_d4a) + 1
                evaluated_tokens = prefix_d4a + suffix_d4a + [target_tokens[0]]
                parity = compare_full_logit_checkpoint(
                    control_capture_dir, control_id,
                    target_capture_dir, target_id, checkpoint, campaign_id,
                    evaluated_tokens, layout_relation="different",
                    target_cache_source="live", baseline_cache_source_expected="live")
                d4a_info["full_logit_restore_parity"] = parity
                if parity["state"] != "complete":
                    d4a_failures.append(f"{d4a_id}: full-vocabulary inverse-import comparison failed: {parity}")
            else:
                d4a_info["full_logit_restore_parity"] = {"state": "failed", "reason": "missing generated token"}
        d4a_info["source_snapshot_sha256_after"] = sha256_file(snapshot)
        if d4a_info["source_snapshot_sha256_after"] != original_sha:
            d4a_failures.append(f"{d4a_id}: original D1 source snapshot changed")
    except Exception as exc:  # noqa: BLE001
        d4a_failures.append(f"{d4a_id}: {exc}")
    finally:
        if d4a_target is not None:
            d4a_info["target_exit_code"] = d4a_target.stop()
        if d4a_control is not None:
            d4a_info["control_exit_code"] = d4a_control.stop()
        source_sha_after = sha256_file(snapshot)
        d4a_info["source_snapshot_sha256_after"] = source_sha_after
        if source_sha_after != original_sha and not any("original D1 source snapshot changed" in x for x in d4a_failures):
            d4a_failures.append(f"{d4a_id}: original D1 source snapshot changed")
        record_leg(d4a_id, "failed" if d4a_failures else "complete",
                   kind="split_local_to_pure_below_p", p=p_high, destination_p=0,
                   ctx=args.split_ctx_high,
                   expectation="actual n_saved <= P is restored into PURE with cache hit",
                   detail=d4a_info, validation_failures=d4a_failures)
        runs.append({"case_id": d4a_id, "state": "failed" if d4a_failures else "complete", **d4a_info})

    # ---------------------------------------------------------------------
    # Own-fixture legs (D3 above P, D4b above-P to PURE, D5 occupancy >
    # capacity). These MUST NOT
    # reuse D1's snapshot: D1 persists n_saved=128 (<= P) because the frame
    # only stores SEALED record groups, so it can never prove an occupancy
    # above P nor an occupancy larger than the destination capacity. Each leg
    # therefore builds its OWN snapshot with a prefill long enough that the
    # sealed occupancy actually exceeds the bound, and ASSERTS that prefill
    # before testing anything. A fixture that fails its precondition is
    # recorded as failed, never as a pass.
    # ---------------------------------------------------------------------
    own_fixture_legs = [leg for leg in negative_legs if leg[5]]
    for leg_name, p, ctx, expect_restore, expect_hit, precond, why in own_fixture_legs:
        leg_failures: list[str] = []
        inst = None
        own_dir = output_dir / f"slot-save-{leg_name}"
        own_dir.mkdir(parents=True, exist_ok=True)
        own_file = own_dir / f"snap-{leg_name}.bin"
        own_info: dict[str, Any] = {}
        build: ServerInstance | None = None
        # precond is (min_occupancy, bound_name, bound) -- occupancy must be
        # strictly greater than bound to make the leg meaningful.
        if len(precond) == 4:
            _fixture_kind, bound_name, bound, source_p = precond
        else:
            _fixture_kind, bound_name, bound = precond
            source_p = p
        n_prefill = max(args.split_tokens_local + 256, int(bound) + 256)
        # The fixture is built in a SOURCE context large enough to hold the
        # required occupancy; only the DESTINATION ctx may be smaller (that is
        # the whole point of D5). Building it in the destination ctx would
        # silently cap the occupancy and the precondition could never hold.
        source_ctx = args.split_ctx_high
        own_info.update({"source_p": source_p, "destination_p": p,
                         "bound_name": bound_name, "bound": bound,
                         "prefill_tokens": n_prefill})
        if n_prefill + args.suffix_tokens + args.decode_tokens >= source_ctx:
            leg_failures.append(
                f"{leg_name}: fixture prefill {n_prefill} does not fit source "
                f"ctx={source_ctx}")
            entry = record_fixture_precondition_failure(
                record_leg, runs, leg_name=leg_name, p=p, ctx=ctx, source_ctx=source_ctx,
                output_dir=output_dir, own_dir=own_dir, own_file=own_file,
                n_prefill=n_prefill, bound_name=bound_name, bound=bound,
                own_info=own_info, failures=leg_failures, expectation=why)
            log(f"{leg_name} state={entry['state']} fixture precondition failed: {leg_failures[0]}")
            continue
        try:
            # Build the snapshot in its OWN process with its own P, so the
            # persisted envelope really carries the intended geometry.
            build = start_split(f"{leg_name}-fixture", source_p, source_ctx, own_dir,
                                split_profile(args, qx, split=source_p != 0))
            source_command_argv = build.command
            fixture_tokens = frozen_prefix_ids(frozen, n_prefill, leg_name)
            _, own_info, ff = save_snapshot(build, f"{leg_name}-fixture",
                                            fixture_tokens,
                                            f"snap-{leg_name}.bin", f"{leg_name}-save")
            own_info["source_command_argv"] = source_command_argv
            own_info["source_command_shell"] = shlex.join(source_command_argv)
            own_info["snapshot_sha256_before"] = own_info.get("sha256")
            build_exit = build.stop()
            build = None
            leg_failures += ff
            # THE precondition: occupancy strictly greater than the bound.
            n_occ = own_info.get("n_saved")
            own_info["prefill_tokens"] = n_prefill
            own_info["source_p"] = source_p
            own_info["destination_p"] = p
            own_info["bound_name"] = bound_name
            own_info["bound"] = bound
            own_info["occupancy_gt_bound"] = (
                isinstance(n_occ, int) and n_occ > bound)
            own_info["fixture_exit_code"] = build_exit
            if not own_info["occupancy_gt_bound"]:
                leg_failures.append(
                    f"{leg_name}: fixture occupancy n_saved={n_occ} is not > "
                    f"{bound_name}={bound}; the leg would not test what it claims")
            if not own_file.is_file():
                leg_failures.append(f"{leg_name}: fixture snapshot {own_file} was not written")
        except Exception as exc:  # noqa: BLE001
            leg_failures.append(f"{leg_name}: fixture build failed: {exc}")
        finally:
            if build is not None:
                own_info["fixture_exit_code"] = build.stop()
        if leg_failures:
            entry = record_fixture_precondition_failure(
                record_leg, runs, leg_name=leg_name, p=p, ctx=ctx, source_ctx=source_ctx,
                output_dir=output_dir, own_dir=own_dir, own_file=own_file,
                n_prefill=n_prefill, bound_name=bound_name, bound=bound,
                own_info=own_info, failures=leg_failures, expectation=why)
            log(f"{leg_name} FAILED precondition: {leg_failures[0]}")
            continue
        # Precondition holds: now run the restore in a FRESH process.
        leg: dict[str, Any] = {}
        try:
            inst = start_split(f"{leg_name}-restore", p, ctx, own_dir,
                               split_profile(args, qx, split=p != 0))
            own_info["destination_command_argv"] = inst.command
            own_info["destination_command_shell"] = shlex.join(inst.command)
            leg, lf = restore_and_extend(inst, f"{leg_name}-restore", own_file.name,
                                         fixture_tokens + suffix,
                                         leg_name, expect_restore=expect_restore,
                                         expect_cache_n=None, expect_hit=expect_hit)
            leg_failures += lf
            n_occ = own_info.get("n_saved")
            cn = leg.get("cache_n")
            if leg_name == "D4b-occupied-overflow-to-pure":
                if cn != 0:
                    leg_failures.append(f"{leg_name}: pure destination did not show cache miss (cache_n={cn!r})")
                restore_delta = inst.read_log_delta()
                refusal = refusal_reason_from_log(restore_delta)
                parsed_refusal = split_to_pure_import_p_refusal(refusal, p_high, n_occ)
                own_info["native_extent_refusal"] = parsed_refusal
                own_info["native_extent_refusal_line"] = refusal
                if not (parsed_refusal and parsed_refusal["p_attributable"] and
                        parsed_refusal["native_extent"] > p_high and n_occ > p_high):
                    leg_failures.append(
                        f"{leg_name}: refusal does not prove native extent >= sealed occupancy {n_occ} > source P={p_high}")
            # Occupied overflow really exercised: the restored occupancy is
            # above P, so tokens [P, n_occ) live in the OVERFLOW range.
            if expect_restore and isinstance(n_occ, int) and n_occ > bound and isinstance(cn, int):
                leg["overflow_occupied_tokens"] = max(0, min(n_occ, ctx) - bound)
                leg["overflow_occupied"] = leg["overflow_occupied_tokens"] > 0
        except Exception as exc:  # noqa: BLE001
            leg_failures.append(f"{leg_name}: {exc}")
        finally:
            exit_code = inst.stop() if inst is not None else None
            delta = inst.read_log_delta() if inst is not None else ""
            after_hash = sha256_file(own_file)
            own_info["snapshot_sha256_after"] = after_hash
            if own_info.get("sha256") and after_hash != own_info["sha256"]:
                leg_failures.append(f"{leg_name}: own snapshot mutated during restore")
            entry = record_leg(leg_name, "failed" if leg_failures else "complete",
                               p=p, ctx=ctx, expectation=why, expect_restore=expect_restore,
                               expect_hit=expect_hit, fixture=own_info,
                               snapshot_sha256=after_hash,
                               restore_response=str(leg.get("restore_response"))[:300],
                               cache_n=leg.get("cache_n"), prompt_n=leg.get("prompt_n"),
                               overflow_occupied=leg.get("overflow_occupied"),
                               overflow_occupied_tokens=leg.get("overflow_occupied_tokens"),
                               n_restored=(leg.get("restore_counts") or {}).get("n_restored"),
                               server_exit_code=exit_code,
                               log_evidence=[line.strip() for line in delta.splitlines()
                                             if "position_split" in line or
                                             "mixed state" in line.lower()][:40],
                               validation_failures=leg_failures)
            runs.append(entry)
        log(f"{leg_name} state={entry['state']} n_saved={own_info.get('n_saved')} "
            f"> {own_info.get('bound_name')}={own_info.get('bound')} ({why})")

    for leg_name, p, ctx, expect_restore, expect_hit, precond, why in negative_legs:
        if precond:
            continue
        leg_failures: list[str] = []
        inst = None
        inst_dir = output_dir / f"slot-save-{leg_name}"
        inst_dir.mkdir(parents=True, exist_ok=True)
        # Copy the ORIGINAL snapshot; never mutate the source file.
        original = snapshot.read_bytes()
        shutil.copy2(snapshot, inst_dir / snapshot.name)
        before_hash = sha256_file(inst_dir / snapshot.name)
        leg: dict[str, Any] = {}
        try:
            split_flag = p != 0
            inst = start_split(leg_name, p, ctx, inst_dir,
                               split_profile(args, qx, split=split_flag))
            leg, lf = restore_and_extend(inst, leg_name, snapshot.name,
                                         prefix + suffix, leg_name,
                                         expect_restore=expect_restore,
                                         expect_cache_n=None,
                                         expect_hit=expect_hit)
            leg_failures += lf
        except Exception as exc:  # noqa: BLE001
            leg_failures.append(f"{leg_name}: {exc}")
        finally:
            exit_code = None
            delta = ""
            if inst is not None:
                exit_code = inst.stop()
                delta = inst.read_log_delta()
            after_hash = sha256_file(inst_dir / snapshot.name)
            if before_hash != after_hash:
                leg_failures.append(f"{leg_name}: snapshot mutated "
                                    f"({before_hash[:16]} -> {after_hash[:16]})")
            log_evidence = [line.strip() for line in delta.splitlines()
                            if "position_split" in line or "mixed state" in line.lower()
                            or "restore" in line.lower()]
            entry = record_leg(leg_name, "failed" if leg_failures else "complete",
                               p=p, ctx=ctx, split=split_flag,
                               expectation=why, expect_restore=expect_restore,
                               expect_hit=expect_hit,
                               restore_response=str(leg.get("restore_response"))[:300],
                               cache_n=leg.get("cache_n"), prompt_n=leg.get("prompt_n"),
                               generated_tokens=leg.get("generated_tokens"),
                               snapshot_sha256_before=before_hash,
                               snapshot_sha256_after=after_hash,
                               snapshot_preserved=before_hash == after_hash,
                               server_exit_code=exit_code,
                               log_evidence=log_evidence[:40],
                               validation_failures=leg_failures)
            runs.append(entry)
        log(f"{leg_name} state={entry['state']} ({why})")
        if leg_failures:
            failures += [f"{leg_name}: {f}" for f in leg_failures]

    # ---------------------------------------------------------------------
    # D6: invalid snapshot must be rejected with NO mutation. The frame is
    # corrupted in a private copy only; the good snapshot is never touched.
    # ---------------------------------------------------------------------
    leg_name = "D6-invalid-snapshot"
    inst = None
    leg_failures: list[str] = []
    inst_dir = output_dir / f"slot-save-{leg_name}"
    inst_dir.mkdir(parents=True, exist_ok=True)
    good_hash = sha256_file(snapshot)
    entry: dict[str, Any] = {}
    # Locate a byte that is provably INSIDE a checksummed payload of the mixed
    # state envelope. A blind len//2 flip is not evidence: the previous run
    # flipped a byte that was accepted, which proved nothing about integrity.
    # The mixed v2 envelope verifies per-payload XXH64 and throws
    # (llama-kv-mixed-state.cpp:885 with fault() -> throw at :84-85), so a
    # correct corruption inside a payload MUST be rejected.
    target = locate_checksummed_payload_byte(Path(snapshot))
    if target is None:
        # Not a mixed-v2 envelope we can parse: recorded as not executed with
        # the reason, never silently as a pass.
        entry = record_leg(leg_name, "not_executed",
                           p=p_high, ctx=args.split_ctx_high,
                           expectation="corrupt a byte provably inside a checksummed "
                                       "payload; reject it; destination unchanged",
                           good_snapshot_sha256=good_hash,
                           reason=("could not locate a checksummed payload region in this "
                                   "frame; the previous blind mid-file flip was not evidence"),
                           validation_failures=[])
    else:
        flip_at, layer_id, range_id, region = target
        bad = bytearray(original)
        byte_before = bad[flip_at]          # recorded BEFORE the XOR
        bad[flip_at] ^= 0xFF
        byte_after = bad[flip_at]
        bad_path = inst_dir / snapshot.name
        bad_path.write_bytes(bytes(bad))
        bad_hash = sha256_file(bad_path)
        if byte_before == byte_after:
            leg_failures.append("D6: XOR did not change the byte (before == after)")
        if sha256_file(snapshot) != good_hash:
            leg_failures.append("D6: writing the corrupted COPY mutated the GOOD snapshot")

        # Prepare a KNOWN destination state BEFORE the invalid attempt, then
        # compare it after. A preserved file hash does NOT prove the cache /
        # DeltaNet destination was untouched, so the slot's own state is the
        # thing that must be compared.
        before_state: dict[str, Any] = {}
        after_state: dict[str, Any] = {}
        restore_detail: str | None = None
        accepted = False
        try:
            inst = start_split(leg_name, p_high, args.split_ctx_high, inst_dir,
                               split_profile(args, qx, split=True))
            # Known destination state: a deterministic prefill.
            seed_tokens = frozen_prefix_ids(frozen, args.split_tokens_local, leg_name)
            seed = http_request(
                base_url, "POST", "/completion",
                {"prompt": seed_tokens, "n_predict": args.decode_tokens,
                 "id_slot": 0, "cache_prompt": True, "return_tokens": True,
                 "temperature": 0, "seed": args.seed, "ignore_eos": True,
                 "n_probs": args.n_probs},
                args.request_timeout, records_path, args.telemetry_device,
                leg_name, "D6-seed-prefill")
            seed_body = seed["json"]
            before_state = {
                "generated_tokens": seed_body.get("tokens"),
                "content": seed_body.get("content"),
                "completion_probabilities": seed_body.get("completion_probabilities"),
                "timings": {k: v for k, v in (seed_body.get("timings") or {}).items()
                            if k in ("prompt_n", "cache_n", "predicted_n")},
            }
            restore = http_request(base_url, "POST", "/slots/0?action=restore",
                                   {"filename": bad_path.name}, args.request_timeout,
                                   records_path, args.telemetry_device, leg_name,
                                   "D6-restore-invalid")
            failed_r, restore_detail = response_is_error(restore)
            accepted = not failed_r
            # Same request again AFTER the rejected attempt. Identical tokens
            # AND identical next-token probabilities prove the destination
            # state (KV + DeltaNet/recurrent) was not disturbed: the server
            # recomputes from the same state, so any hidden mutation would
            # move the logits.
            post = http_request(
                base_url, "POST", "/completion",
                {"prompt": seed_tokens, "n_predict": args.decode_tokens,
                 "id_slot": 0, "cache_prompt": True, "return_tokens": True,
                 "temperature": 0, "seed": args.seed, "ignore_eos": True,
                 "n_probs": args.n_probs},
                args.request_timeout, records_path, args.telemetry_device,
                leg_name, "D6-post-prefill")
            post_body = post["json"]
            after_state = {
                "generated_tokens": post_body.get("tokens"),
                "content": post_body.get("content"),
                "completion_probabilities": post_body.get("completion_probabilities"),
            }
            tokens_same = (before_state["generated_tokens"] ==
                           after_state["generated_tokens"])
            probs_same = compare_completion_probabilities(
                before_state.get("completion_probabilities"),
                after_state.get("completion_probabilities"))
            if not tokens_same:
                leg_failures.append(
                    "D6: destination tokens changed after the rejected restore "
                    "(state was mutated)")
            if not probs_same:
                leg_failures.append(
                    "D6: destination next-token probabilities changed after the "
                    "rejected restore (recurrent/KV state was mutated)")
            if accepted:
                leg_failures.append(
                    "D6: a frame corrupted INSIDE a checksummed payload was "
                    "accepted (must be rejected)")
            if sha256_file(snapshot) != good_hash:
                leg_failures.append("D6: the GOOD snapshot was mutated")
        except Exception as exc:  # noqa: BLE001
            leg_failures.append(f"D6: {exc}")
        finally:
            if inst is not None:
                entry["server_exit_code"] = inst.stop()
                delta = inst.read_log_delta()
                entry["log_evidence"] = [line.strip() for line in delta.splitlines()
                                         if "checksum" in line.lower()
                                         or "position_split" in line
                                         or "mixed state" in line.lower()][:40]
        entry.update(record_leg(
            leg_name, "failed" if leg_failures else "complete",
            p=p_high, ctx=args.split_ctx_high,
            expectation="corrupt a byte provably inside a checksummed payload; "
                        "reject it; destination KV+DeltaNet state and next logits unchanged",
            good_snapshot_sha256=good_hash, corrupted_sha256=bad_hash,
            corrupt_offset=flip_at, corrupt_byte_before=byte_before,
            corrupt_byte_after=byte_after, corrupt_region=region,
            corrupt_layer_id=layer_id, corrupt_range_id=range_id,
            corrupted_restore_rejected=not accepted,
            destination_state_before=before_state,
            destination_state_after=after_state,
            destination_tokens_unchanged=(
                before_state.get("generated_tokens") == after_state.get("generated_tokens")),
            destination_next_logits_unchanged=compare_completion_probabilities(
                before_state.get("completion_probabilities"),
                after_state.get("completion_probabilities")),
            good_snapshot_preserved=sha256_file(snapshot) == good_hash,
            restore_detail=restore_detail,
            validation_failures=leg_failures))
    runs.append(entry)
    log(f"D6 state={entry['state']}")
    failures += [f"D6: {f}" for f in leg_failures]

    return {"scenario": "D", "runs": runs, "failures": failures}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.scenarios = parse_names(args.scenarios, "--scenarios", {"A", "B", "C", "D"})
        args.remote_kv_types = parse_names(args.remote_kv_types, "--remote-kv-types", set(REMOTE_TYPES))
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be in 1..65535")
    if any(value <= 0 for value in (args.startup_timeout, args.request_timeout, args.shutdown_timeout)):
        parser.error("startup, request, and shutdown timeouts must be positive")
    if any(value is not None and value <= 0 for value in (
        args.apu_tdp, args.amd_sclk_prefill, args.amd_sclk_decode,
    )):
        parser.error("APU TDP and AMD SCLK controls must be positive")
    if args.remote_attn_layers < 1:
        parser.error("--remote-attn-layers must be >= 1")
    if args.n_probs < 0:
        parser.error("--n-probs must be >= 0")
    if args.full_logit_capture:
        if "D" not in args.scenarios:
            parser.error("--full-logit-capture requires scenario D")
        if args.decode_tokens < 2:
            parser.error("--full-logit-capture requires --decode-tokens >= 2 for a genuinely decoded restore checkpoint")
        try:
            import numpy  # noqa: F401
        except ImportError:
            parser.error("--full-logit-capture requires NumPy")
    try:
        validate_full_logit_capture_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    required_ctx = max(args.prompt_tokens, args.prompt_short, args.prompt_long) + \
        args.suffix_tokens + max(args.decode_tokens, args.n_predict)
    if args.ctx_size < required_ctx:
        parser.error(f"--ctx-size {args.ctx_size} must cover prompt + suffix + output ({required_ctx})")
    if args.prompt_long + args.n_predict > args.high_ctx:
        parser.error("--prompt-long + --n-predict must fit the high tier --high-ctx")
    if args.prompt_short + args.n_predict > args.local_ctx:
        parser.error("--prompt-short + --n-predict must fit the local tier --local-ctx")
    if not (args.local_ctx < args.gate_ctx <= args.high_ctx):
        parser.error("--gate-ctx must satisfy --local-ctx < gate <= --high-ctx")
    if args.local_ctx >= args.high_ctx:
        parser.error("--local-ctx (MTP profile, --ctx-size-mtp) must be smaller than "
                     "--high-ctx (LONG profile, --ctx-size): common_context_adaptive_error "
                     "rejects --ctx-size-mtp >= --ctx-size")

    # Scenario D validation. P is accepted through public --kv-position-split-p
    # and the legacy environment route used by this harness. It must be a
    # multiple of 256 with 0 < P < n_ctx; reject bad values before server start.
    if "D" in args.scenarios:
        for name, p in (("--split-p-local", args.split_p_local),
                        ("--split-p-high", args.split_p_high)):
            if p % 256 != 0:
                parser.error(f"{name} {p} must be a multiple of 256")
            if p <= 0:
                parser.error(f"{name} must be > 0 (0 disables the split)")
        for name, ctx in (("--split-ctx-local", args.split_ctx_local),
                          ("--split-ctx-high", args.split_ctx_high)):
            if ctx <= 0:
                parser.error(f"{name} must be positive")
        for p, ctx in ((args.split_p_local, args.split_ctx_local),
                       (args.split_p_high, args.split_ctx_high)):
            if p >= ctx:
                parser.error(f"P={p} must be smaller than n_ctx={ctx}")
        d_required = args.split_tokens_local + args.suffix_tokens + \
            max(args.decode_tokens, args.n_predict)
        for name, ctx in (("--split-ctx-local", args.split_ctx_local),
                          ("--split-ctx-high", args.split_ctx_high)):
            if ctx < d_required:
                parser.error(f"{name} {ctx} must cover the scenario D prefix + suffix + "
                             f"output ({d_required})")
        if args.split_tokens_local >= args.split_p_high:
            parser.error("--split-tokens-local must stay below --split-p-high so D1 "
                         f"imports n_saved <= P ({args.split_tokens_local} < "
                         f"{args.split_p_high})")
        if args.split_tokens_local >= args.split_p_local:
            parser.error("--split-tokens-local must stay below --split-p-local "
                         f"({args.split_tokens_local} < {args.split_p_local})")
        # D5 relies on the destination capacity being SMALLER than the snapshot.
        try:
            validate_d5_destination_capacity(args.split_ctx_local, args.split_p_high)
        except ValueError as exc:
            parser.error(str(exc))
        if args.split_ctx_local >= args.split_ctx_high:
            parser.error("--split-ctx-local must be smaller than --split-ctx-high so the "
                         "capacity-rejection leg is meaningful")
        # The mixed state spool refuses tmpfs/ramfs, so a scenario D run on a
        # tmpfs TMPDIR can never prove "only the snapshot on disk". Reject it
        # up front with the real filesystem type instead of failing at restore.
        disk_tmp = args.disk_tmpdir.resolve()
        try:
            disk_tmp.mkdir(parents=True, exist_ok=True)
            fstype = subprocess.run(
                ["stat", "-f", "-c", "%T", str(disk_tmp)],
                capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            parser.error(f"--disk-tmpdir {disk_tmp} is unusable: {exc}")
        if fstype in ("tmpfs", "ramfs"):
            parser.error(f"--disk-tmpdir {disk_tmp} is on {fstype}; the mixed state spool "
                         "requires a disk-backed filesystem (tmpfs/ramfs is rejected by the "
                         "server), so scenario D cannot prove a disk-only snapshot there")
        args.disk_tmpdir = disk_tmp

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
        "bench-results") / f"mixed-kv-transition-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
        "created_utc": utc_now(), "machine_hostname": platform.node(),
        "platform": platform.platform(), "cwd": str(Path.cwd()), "repo": str(repo),
        "git_commit": git_value(repo, "rev-parse", "HEAD"),
        "git_status_porcelain": git_value(repo, "status", "--short"),
        "server_binary": str(binary), "server_binary_size": binary.stat().st_size,
        "server_binary_sha256": None if args.skip_binary_hash else sha256_file(binary),
        "model": str(model), "model_size": model.stat().st_size,
        "model_sha256": None if args.skip_model_hash else sha256_file(model),
        "harness": str(script_path), "harness_sha256": sha256_file(script_path),
        "listen_host": args.host, "port": args.port,
        "configuration": configuration_from_args(args),
        "full_logit_build_id_operator_verified": args.full_logit_build_id,
        "telemetry_device": args.telemetry_device,
        "scenarios": args.scenarios, "remote_kv_types": args.remote_kv_types,
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
    capture_build_id, capture_build_id_source = resolve_full_logit_capture_identity(
        args, manifest["server_binary_sha256"])
    manifest["full_logit_capture_build_id"] = capture_build_id
    manifest["full_logit_capture_build_id_source"] = capture_build_id_source
    if args.full_logit_capture:
        manifest["artifacts"].append("full-logits/<case>/capture-plan.json")
    write_json(output_dir / "run.json", manifest)
    print(f"Output: {output_dir}\nScenarios: {','.join(args.scenarios)}; "
          f"remote KV: {','.join(args.remote_kv_types)}", flush=True)

    frozen: dict[str, Any] | None = None
    summaries: list[dict[str, Any]] = []
    for scenario in args.scenarios:
        port_free()
        if scenario == "A":
            summary = scenario_a(args, output_dir, manifest, frozen)
        elif scenario == "B":
            summary = scenario_b(args, output_dir, manifest, frozen)
        elif scenario == "D":
            summary = scenario_d(args, output_dir, manifest, frozen)
        else:
            summary = scenario_c(args, output_dir, manifest, frozen)
        summaries.append(summary)
        prompts_path = output_dir / "prompts.json"
        if prompts_path.is_file() and frozen is None:
            frozen = json.loads(prompts_path.read_text(encoding="utf-8"))

    runs = [run for summary in summaries for run in summary.get("runs", [])]
    aggregate = aggregate_run_states(runs)
    write_json(output_dir / "summary.json", {
        "finished_utc": utc_now(), "scenario_count": len(args.scenarios),
        "total_runs": len(runs),
        "failed_runs": aggregate["failed"],
        "pending_evidence_runs": aggregate["pending_evidence"],
        "inconclusive_runs": aggregate["inconclusive"],
        "not_executed_runs": aggregate["not_executed"],
        "complete_runs": aggregate["complete"],
        "unacceptable_runs": aggregate["unacceptable_runs"],
        "unacceptable_legs": aggregate["unacceptable_legs"],
        "incomplete": aggregate["incomplete"],
        "exit_code": 0 if not aggregate["incomplete"] else 1,
        "note": "ONLY runs whose state is exactly 'complete' are accepted. Any other "
                "state (failed, pending_evidence, inconclusive, not_executed, or a "
                "missing/unknown state) makes the run incomplete and forces a "
                "non-zero exit; 'inconclusive' and 'not_executed' are NOT passes.",
        "output_dir": str(output_dir),
    })
    print(f"Finished: {len(args.scenarios)} scenarios, {len(runs)} runs: "
          f"{aggregate['complete']} complete, {aggregate['failed']} failed, "
          f"{aggregate['pending_evidence']} pending_evidence, "
          f"{aggregate['inconclusive']} inconclusive, "
          f"{aggregate['not_executed']} not_executed -> "
          f"{'INCOMPLETE (exit 1)' if aggregate['incomplete'] else 'complete (exit 0)'}; "
          f"results={output_dir}", flush=True)
    if aggregate["incomplete"]:
        print(f"Unacceptable legs: {', '.join(aggregate['unacceptable_legs'])}", flush=True)
    return 0 if not aggregate["incomplete"] else 1


if __name__ == "__main__":
    sys.exit(main())
