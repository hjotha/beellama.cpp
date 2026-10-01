#!/usr/bin/env python3
"""Mixed-KV acceptance runner: starts only its own llama-server instance.

Black-box acceptance for the upcoming mixed KV feature (local KVarN4 + remote
Q4/Q5/Q6/Q8 on Vulkan layers). Each scenario starts one isolated llama-server
in its own process group on an isolated port with its own --slot-save-path,
records the exact argv, prompts, responses and logs, validates token/byte
counts, and reports failures honestly. stdlib only; no GPU work is performed
by this script itself.

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
                        help="comma-separated acceptance scenarios: A,B,C")
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


def create_frozen_prompts(base_url: str, args: argparse.Namespace, record_path: Path,
                          output_dir: Path, case_id: str,
                          existing: dict[str, Any] | None) -> dict[str, Any]:
    """Tokenize the corpus once and freeze the ids. Subsequent instances must
    produce the identical ids (tokenizer identity check) so every mode shares
    the exact same prefix bytes and raw token ids."""
    prompts_path = output_dir / "prompts.json"
    largest = max(args.prompt_tokens, args.prompt_short, args.prompt_long) + args.suffix_tokens
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

            prefix = frozen["prefix_token_ids"][:args.prompt_tokens]
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

        short_ids = frozen["prefix_token_ids"][:args.prompt_short]
        long_ids = frozen["prefix_token_ids"][:args.prompt_long]
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

        prefix = frozen["prefix_token_ids"][:512]
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
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.scenarios = parse_names(args.scenarios, "--scenarios", {"A", "B", "C"})
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
        else:
            summary = scenario_c(args, output_dir, manifest, frozen)
        summaries.append(summary)
        prompts_path = output_dir / "prompts.json"
        if prompts_path.is_file() and frozen is None:
            frozen = json.loads(prompts_path.read_text(encoding="utf-8"))

    runs = [run for summary in summaries for run in summary.get("runs", [])]
    failed = sum(1 for run in runs if run["state"] == "failed")
    pending = sum(1 for run in runs if run["state"] == "pending_evidence")
    write_json(output_dir / "summary.json", {
        "finished_utc": utc_now(), "scenario_count": len(args.scenarios),
        "failed_runs": failed, "pending_evidence_runs": pending,
        "note": "pending_evidence runs are NOT fully accepted: the handoff "
                "conversion evidence markers were not configured",
        "output_dir": str(output_dir),
    })
    print(f"Finished: {len(args.scenarios)} scenarios, {failed} failed, "
          f"{pending} pending_evidence; results={output_dir}", flush=True)
    return 1 if failed or pending else 0


if __name__ == "__main__":
    sys.exit(main())