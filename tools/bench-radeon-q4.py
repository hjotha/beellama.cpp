#!/usr/bin/env python3
"""Controlled llama-server Q4/KVarN comparison; starts only its own server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import signal
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path(
    "/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
)
DEFAULT_BINARY = Path("build-optimized/bin/llama-server")
NATURAL_LINES = (
    "The field team compared its written observations with the readings collected beside the river.",
    "Clear weather allowed the surveyors to follow the marked trail along the northern ridge.",
    "At midday, they checked the compass, updated the paper map, and recorded the direction of the wind.",
    "The archive keeps each measurement with its date so another researcher can repeat the journey.",
    "After the rain passed, the crew returned to the hillside and continued the careful inspection.",
    "A small change in the clouds altered the light, but the instruments remained steady and readable.",
    "The notebook describes the landscape in plain language and lists the tools used at every stop.",
    "By evening, the observers had matched the river marks to the map and checked their notes twice.",
)
LOADED_LIBRARY_HASHES: dict[str, str | None] = {}
PROFILE_ENV_KEYS = ("GGML_VK_PERF_LOGGER", "GGML_BACKEND_COPY_PROFILE")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def linked_runtime_libraries(binary: Path) -> tuple[str | None, list[dict[str, Any]]]:
    try:
        result = subprocess.run(["ldd", str(binary)], check=False, capture_output=True,
                                text=True, timeout=15)
        ldd_output = result.stdout + result.stderr
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{type(exc).__name__}: {exc}", []
    libraries: dict[str, dict[str, Any]] = {}
    for line in ldd_output.splitlines():
        if "libllama" not in line and "libggml" not in line:
            continue
        match = re.search(r"=>\s+(/\S+)", line)
        if not match:
            continue
        path = Path(match.group(1))
        if path.is_file():
            libraries[str(path.resolve())] = {
                "path": str(path.resolve()), "size": path.stat().st_size,
                "sha256": sha256_file(path), "ldd_line": line.strip(),
            }
    return ldd_output.strip(), list(libraries.values())


def loaded_runtime_libraries(pid: int) -> list[dict[str, Any]]:
    """Hash libraries mapped by the live server, including dlopen backends."""
    try:
        lines = Path(f"/proc/{pid}/maps").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return [{"maps_error": f"{type(exc).__name__}: {exc}"}]
    paths: dict[str, bool] = {}
    for line in lines:
        fields = line.split(None, 5)
        if len(fields) < 6:
            continue
        name = fields[5]
        if not name.startswith("/") or ("libggml" not in name and "libllama" not in name):
            continue
        deleted = name.endswith(" (deleted)")
        paths[name.removesuffix(" (deleted)")] = deleted
    records = []
    for path_text, deleted in sorted(paths.items()):
        path = Path(path_text)
        if path_text not in LOADED_LIBRARY_HASHES:
            try:
                LOADED_LIBRARY_HASHES[path_text] = sha256_file(path) if path.is_file() and not deleted else None
            except OSError:
                LOADED_LIBRARY_HASHES[path_text] = None
        try:
            size = path.stat().st_size if not deleted else None
        except OSError:
            size = None
        records.append({"path": path_text, "size": size,
                        "sha256": LOADED_LIBRARY_HASHES[path_text], "deleted": deleted})
    return records


def parse_ints(value: str, name: str, minimum: int = 1) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not values or any(item < minimum for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} must contain unique integers >= {minimum}")
    return values


def parse_names(value: str, name: str, allowed: set[str]) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(set(values)) != len(values) or any(item not in allowed for item in values):
        raise argparse.ArgumentTypeError(f"{name} must be unique values from {', '.join(sorted(allowed))}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18091)
    parser.add_argument("--modes", default="cuda-only,remote-vulkan",
                        help="comma-separated: cuda-only,remote-vulkan")
    parser.add_argument("--cache-types", default="kvarn4,q4_0",
                        help="global K and V types: kvarn4,q4_0,q5_0,q6_0,q8_0")
    parser.add_argument("--remote-cache-types",
                        help="optional Qx overrides on Vulkan layers; requires --cache-types kvarn4")
    parser.add_argument("--remote-layers", default="1,16",
                        help="first N full-attention layers for remote-vulkan mode")
    parser.add_argument("--prompt-lengths", default="8192,16384",
                        help="exact token counts including BOS")
    parser.add_argument("--decode-tokens", type=int, default=48)
    parser.add_argument("--suffix-decode-tokens", type=int, default=32)
    parser.add_argument("--suffix-tokens", type=int, default=32,
                        help="new prompt tokens appended for the cache-hit continuation")
    parser.add_argument("--ctx-size", type=int,
                        help="fixed context; default is max prompt + suffix + decode + 256")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--tail-tokens", type=int, default=0,
                        help="configured exact standard KV tail; KVarN retains its intrinsic 128")
    parser.add_argument("--tail-type", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--cuda-device", default="CUDA0")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--shutdown-timeout", type=float, default=15)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--telemetry-device", default="/sys/class/drm/card1/device",
                        help="AMD DRM device sysfs directory, or 'none'")
    parser.add_argument("--debug-kvarn-routes", action="store_true",
                        help="enable GGML_KVARN_DEBUG_ROUTES; use only for diagnostic runs")
    parser.add_argument("--apu-tdp", type=int, help="optional APU power limit in watts")
    parser.add_argument("--amd-sclk-prefill", type=int, help="optional AMD prefill SCLK in MHz")
    parser.add_argument("--amd-sclk-decode", type=int, help="optional AMD decode SCLK in MHz")
    parser.add_argument("--gpu-power-amd-device", type=int,
                        help="AMD GPU governor device index (default 0 when controls are enabled)")
    parser.add_argument("--gpu-power-backend", choices=("auto", "nvml", "amdgpu", "dual"))
    parser.add_argument("--skip-model-hash", action="store_true")
    parser.add_argument("--skip-binary-hash", action="store_true")
    return parser


def telemetry(device: str) -> dict[str, Any]:
    if device == "none":
        return {"available": False, "reason": "disabled"}
    root = Path(device)
    fields = (
        "gpu_busy_percent", "mem_info_vram_used", "mem_info_vram_total",
        "mem_info_gtt_used", "mem_info_gtt_total",
        "pp_dpm_sclk", "pp_dpm_mclk", "pp_dpm_socclk", "power_dpm_force_performance_level",
    )
    values: dict[str, Any] = {"device": str(root), "available": root.exists()}
    for name in fields:
        path = root / name
        try:
            values[name] = path.read_text(encoding="utf-8").strip()
        except OSError:
            values[name] = None
    for path in sorted(root.glob("hwmon/hwmon*/temp*_input")):
        try:
            values[path.name] = path.read_text(encoding="utf-8").strip()
        except OSError:
            values[path.name] = None
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
                [nvidia_smi, "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=False,
            )
            values["nvidia_smi_exit_code"] = result.returncode
            values["nvidia_smi_gpus"] = result.stdout.strip() if result.returncode == 0 else None
            if result.returncode != 0:
                values["nvidia_smi_error"] = result.stderr.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            values["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return values


def http_request(base_url: str, method: str, path: str, payload: Any | None,
                 timeout: float, record_path: Path, telemetry_device: str,
                 case_id: str | None, phase: str, request_body_record: Any | None = None) -> dict[str, Any]:
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
        error = f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:  # Preserve timeout/socket errors in the run record.
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
        "elapsed_seconds": elapsed, "request": request_body_record if request_body_record is not None else payload,
        "response_json": parsed, "response_raw": raw_text, "error": error,
        "telemetry_before": before, "telemetry_after": after,
    }
    append_jsonl(record_path, row)
    if error:
        raise RuntimeError(error)
    if status is None or status >= 400:
        raise RuntimeError(f"HTTP request failed with status {status}")
    if not isinstance(parsed, dict):
        raise RuntimeError(f"expected a JSON object from {path}; raw response recorded")
    return {"json": parsed, "raw": raw_text, "elapsed_seconds": elapsed}


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


def stop_server(proc: subprocess.Popen[bytes], timeout: float) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


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


def create_prompts(base_url: str, args: argparse.Namespace, record_path: Path,
                   output_dir: Path, first_case_id: str) -> tuple[list[int], list[int]]:
    largest = max(args.prompt_lengths) + args.suffix_tokens
    corpus = make_corpus(largest)
    for attempt in range(3):
        result = http_request(
            base_url, "POST", "/tokenize",
            {"content": corpus, "add_special": True, "parse_special": False},
            args.request_timeout, record_path, args.telemetry_device,
            first_case_id, f"tokenize-corpus-{attempt + 1}",
        )
        tokens = result["json"].get("tokens")
        if isinstance(tokens, list) and all(isinstance(token, int) for token in tokens):
            if len(tokens) >= largest:
                suffix_notes = (
                    "The independent report records a second set of observations from the coastal station.",
                    "Its gauge was checked twice, and the observer marked each reading with the local time.",
                    "A passing cloud changed the light briefly, while the instrument readings remained stable.",
                    "The team noted the distance between markers and listed possible sources of measurement error.",
                    "These notes are separate from the earlier survey and should be summarized on their own.",
                )
                suffix_text = (
                    "\n\nUSER: Please analyze the following independent field report and summarize its "
                    "observations, measurements, and uncertainty in a concise factual paragraph.\n\n"
                    + " ".join(suffix_notes * (args.suffix_tokens + 1))
                )
                suffix_result = http_request(
                    base_url, "POST", "/tokenize",
                    {"content": suffix_text, "add_special": False, "parse_special": False},
                    args.request_timeout, record_path, args.telemetry_device,
                    first_case_id, "tokenize-independent-suffix",
                )
                suffix_ids = suffix_result["json"].get("tokens")
                if not isinstance(suffix_ids, list) or len(suffix_ids) < args.suffix_tokens or not all(
                    isinstance(token, int) for token in suffix_ids
                ):
                    raise RuntimeError(f"independent suffix did not tokenize to {args.suffix_tokens} IDs")
                suffix_ids = suffix_ids[:args.suffix_tokens]
                (output_dir / "prompt-corpus.txt").write_text(corpus, encoding="utf-8")
                (output_dir / "continuation-suffix.txt").write_text(suffix_text, encoding="utf-8")
                write_json(output_dir / "prompts.json", {
                    "source": "natural text sent to this model's POST /tokenize",
                    "add_special": True, "parse_special": False,
                    "token_count_including_bos": len(tokens),
                    "prompt_lengths": args.prompt_lengths,
                    "prompt_token_ids": tokens,
                    "continuation_suffix": {
                        "source_file": "continuation-suffix.txt",
                        "add_special": False, "requested_tokens": args.suffix_tokens,
                        "token_ids": suffix_ids,
                    },
                })
                return tokens, suffix_ids
        corpus += "\n" + corpus
    raise RuntimeError(f"tokenization did not produce {largest} token IDs")


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    modes = args.modes
    cases: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        cache_types = args.cache_types if repeat % 2 else list(reversed(args.cache_types))
        for mode in modes:
            layers: list[int | None] = [None] if mode == "cuda-only" else args.remote_layers
            for layer_count in layers:
                for cache_type in cache_types:
                    remote_types = (args.remote_cache_types or [None]) if mode == "remote-vulkan" else [None]
                    if repeat % 2 == 0:
                        remote_types = list(reversed(remote_types))
                    for remote_type in remote_types:
                        suffix = f"-remote-{remote_type}" if remote_type else ""
                        case_id = f"r{repeat}-{mode}-layers{layer_count or 0}-{cache_type}{suffix}-tail{args.tail_tokens}"
                        cases.append({
                            "case_id": case_id, "repeat": repeat, "mode": mode,
                            "remote_layers": layer_count, "cache_type": cache_type,
                            "remote_cache_type": remote_type,
                        })
    return cases


def server_command(args: argparse.Namespace, case: dict[str, Any], ctx_size: int,
                   slot_save_path: Path) -> list[str]:
    command = [
        str(args.server_binary.resolve()), "--model", str(args.model.resolve()),
        "--host", args.host, "--port", str(args.port), "--device", args.cuda_device,
        "-ngl", "99", "--split-mode", "none", "--main-gpu", "0",
        "--ctx-size", str(ctx_size), "--parallel", "1",
        "--slot-save-path", str(slot_save_path),
        "--batch-size", str(args.batch_size), "--ubatch-size", str(args.ubatch_size),
        "--cache-type-k", case["cache_type"], "--cache-type-v", case["cache_type"],
        "--kv-tail-tokens", str(args.tail_tokens), "--kv-tail-type", args.tail_type,
        "--flash-attn", "on", "--spec-type", "none", "--cache-prompt", "--fit", "off",
        "--no-ui", "--slots", "--no-warmup", "--no-context-shift", "--verbose",
    ]
    if case["mode"] == "remote-vulkan":
        command += ["--remote-attn", "vulkan:0", "--remote-attn-layers", str(case["remote_layers"])]
        if case.get("remote_cache_type"):
            command += ["--remote-attn-cache-type-k", case["remote_cache_type"],
                        "--remote-attn-cache-type-v", case["remote_cache_type"]]
    controls_requested = any(value is not None for value in (
        args.apu_tdp, args.amd_sclk_prefill, args.amd_sclk_decode,
    ))
    if controls_requested or args.gpu_power_backend is not None or args.gpu_power_amd_device is not None:
        command += ["--gpu-power-backend", args.gpu_power_backend or "amdgpu"]
        if args.apu_tdp is not None:
            command += ["--apu-tdp", str(args.apu_tdp)]
        if args.amd_sclk_prefill is not None:
            command += ["--amd-sclk-prefill", str(args.amd_sclk_prefill)]
        if args.amd_sclk_decode is not None:
            command += ["--amd-sclk-decode", str(args.amd_sclk_decode)]
        if args.gpu_power_amd_device is not None:
            command += ["--gpu-power-amd-device", str(args.gpu_power_amd_device)]
    return command


def required_power_log_fragments(args: argparse.Namespace) -> list[str]:
    expected = []
    if args.apu_tdp is not None:
        expected.append(f"APU TDP: active limit {args.apu_tdp} W")
    if args.amd_sclk_prefill is not None and args.amd_sclk_prefill == args.amd_sclk_decode:
        # The governor keeps an unchanged lock across prefill -> decode; it
        # correctly logs only the first hardware write in that case.
        expected.append(f"locked {args.amd_sclk_prefill} MHz")
    else:
        if args.amd_sclk_prefill is not None:
            expected.append(f"-> prefill, locked {args.amd_sclk_prefill} MHz")
        if args.amd_sclk_decode is not None:
            expected.append(f"-> decode, locked {args.amd_sclk_decode} MHz")
    return expected


def run_case(args: argparse.Namespace, case: dict[str, Any], ctx_size: int,
             output_dir: Path, run_meta: dict[str, Any],
             prompts: tuple[list[int], list[int]] | None) -> tuple[dict[str, Any], tuple[list[int], list[int]] | None]:
    case_id = case["case_id"]
    base_url = f"http://{args.host}:{args.port}"
    slot_save_path = output_dir / f"slot-save-{case_id}"
    slot_save_path.mkdir(parents=True, exist_ok=True)
    command = server_command(args, case, ctx_size, slot_save_path)
    log_path = output_dir / f"server-{case_id}.log"
    case_record: dict[str, Any] = {
        **case, "state": "starting", "command_argv": command,
        "command_shell": shlex.join(command), "server_log": log_path.name,
        "slot_save_path": slot_save_path.name, "started_utc": utc_now(), "requests": [],
    }
    append_jsonl(output_dir / "cases.jsonl", case_record)
    print(f"[{case_id}] starting server: {shlex.join(command)}", flush=True)
    env = os.environ.copy()
    case_record["profiling_environment_inherited"] = {
        key: os.environ.get(key) for key in PROFILE_ENV_KEYS + ("GGML_KVARN_DEBUG_ROUTES",)
    }
    for key in PROFILE_ENV_KEYS:
        env.pop(key, None)
    if args.debug_kvarn_routes:
        env["GGML_KVARN_DEBUG_ROUTES"] = "1"
    else:
        env.pop("GGML_KVARN_DEBUG_ROUTES", None)
    with log_path.open("wb") as log:
        log.write((f"started_utc={utc_now()}\ncommand={shlex.join(command)}\n\n").encode())
        log.flush()
        proc: subprocess.Popen[bytes] | None = None
        try:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
            wait_ready(proc, base_url, args.startup_timeout)
            case_record["loaded_runtime_libraries"] = loaded_runtime_libraries(proc.pid)
            run_meta.setdefault("loaded_runtime_libraries_by_case", {})[case_id] = case_record["loaded_runtime_libraries"]
            write_json(output_dir / "run.json", run_meta)
            case_record["state"] = "ready"
            case_record["ready_utc"] = utc_now()
            append_jsonl(output_dir / "cases.jsonl", case_record)
            print(f"[{case_id}] server ready; context={ctx_size}; tail={args.tail_tokens}", flush=True)
            if prompts is None:
                prompts = create_prompts(base_url, args, output_dir / "requests.jsonl",
                                         output_dir, case_id)
                run_meta["prompt_token_ids"] = {
                    "file": "prompts.json", "source_file": "prompt-corpus.txt",
                    "count": len(prompts[0]), "suffix_source_file": "continuation-suffix.txt",
                    "suffix_count": len(prompts[1]),
                }
                write_json(output_dir / "run.json", run_meta)
            prompt_tokens, suffix_tokens = prompts
            for length in args.prompt_lengths:
                base_ids = prompt_tokens[:length]
                extension_ids = base_ids + suffix_tokens
                if len(base_ids) != length or len(extension_ids) != length + args.suffix_tokens:
                    raise RuntimeError("generated prompt IDs do not match requested length")
                for phase, ids, decode_count in (
                    (f"cold-{length}", base_ids, args.decode_tokens),
                    (f"continuation-{length}", extension_ids, args.suffix_decode_tokens),
                ):
                    if phase.startswith("cold-"):
                        http_request(base_url, "POST", "/slots/0?action=erase", None,
                                     args.request_timeout, output_dir / "requests.jsonl",
                                     args.telemetry_device, case_id, f"erase-before-{phase}")
                    prompt_hash = hashlib.sha256(json_bytes(ids)).hexdigest()
                    payload = {
                        "prompt": ids, "n_predict": decode_count,
                        "id_slot": 0, "cache_prompt": True, "return_tokens": True,
                        "temperature": 0, "seed": args.seed, "ignore_eos": True,
                    }
                    print(f"[{case_id}] {phase}: prompt_tokens={len(ids)} decode={decode_count}", flush=True)
                    response = http_request(
                        base_url, "POST", "/completion", payload, args.request_timeout,
                        output_dir / "requests.jsonl", args.telemetry_device, case_id,
                        phase, request_body_record={**payload, "prompt_sha256": prompt_hash},
                    )
                    body = response["json"]
                    timings = body.get("timings") if isinstance(body.get("timings"), dict) else {}
                    cache_n = timings.get("cache_n")
                    prompt_n = timings.get("prompt_n")
                    predicted_n = timings.get("predicted_n")
                    valid_counts = isinstance(cache_n, int) and isinstance(prompt_n, int) and cache_n + prompt_n == len(ids)
                    cache_expectation = (cache_n == 0) if phase.startswith("cold-") else (isinstance(cache_n, int) and cache_n > 0)
                    row = {
                        "time_utc": utc_now(), "case_id": case_id, "phase": phase,
                        "prompt_tokens": len(ids), "prompt_sha256": prompt_hash,
                        "expected_cached_base_tokens": length,
                        "cache_n_delta_from_base": cache_n - length if isinstance(cache_n, int) else None,
                        "decode_requested": decode_count, "wall_seconds": response["elapsed_seconds"],
                        "timings": timings, "cache_count_sum_matches_prompt": valid_counts,
                        "cache_expectation_met": cache_expectation,
                        "predicted_count_matches_request": predicted_n == decode_count,
                        "content": body.get("content"), "generated_tokens": body.get("tokens"),
                        "truncated": body.get("truncated"),
                    }
                    validation_failures = []
                    if not valid_counts:
                        validation_failures.append("cache_n + prompt_n != submitted prompt token count")
                    if not cache_expectation:
                        validation_failures.append("cold/cache-hit expectation was not met")
                    if predicted_n != decode_count:
                        validation_failures.append("predicted_n != requested decode count")
                    if body.get("truncated") is not False:
                        validation_failures.append("completion is missing truncated=false")
                    row["validation_failures"] = validation_failures
                    append_jsonl(output_dir / "results.jsonl", row)
                    case_record["requests"].append({"phase": phase, "prompt_tokens": len(ids),
                                                     "cache_n": cache_n, "prompt_n": prompt_n,
                                                     "predicted_n": predicted_n})
                    print(f"[{case_id}] {phase}: cache_n={cache_n} prompt_n={prompt_n} "
                          f"predicted_n={predicted_n} wall={response['elapsed_seconds']:.2f}s", flush=True)
                    if validation_failures:
                        raise RuntimeError(f"{phase} validation failed: {'; '.join(validation_failures)}")
            case_record["state"] = "complete"
        except Exception as exc:
            case_record["state"] = "failed"
            case_record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{case_id}] failed: {case_record['error']}", flush=True)
        finally:
            if proc is not None:
                stop_server(proc, args.shutdown_timeout)
                case_record["server_exit_code"] = proc.returncode
            log.flush()
            try:
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                log_text = ""
                case_record["log_read_error"] = str(exc)
            case_record["route_log_evidence"] = [
                line.strip() for line in log_text.splitlines()
                if any(marker in line.lower() for marker in (
                    "vulkan-layers=", "local-split attention enabled", "kv buffer",
                    "fallback", "flash_attn", "cuda0", "q4_0", "q5_0", "q6_0", "q8_0",
                    "gpu governor enabled", "amd graphics sclk", "apu tdp: active",
                ))
            ]
            if case_record["state"] == "complete" and case["mode"] == "remote-vulkan":
                expected = f"Vulkan-layers={case['remote_layers']}/"
                if not any(expected in line for line in case_record["route_log_evidence"]):
                    case_record["state"] = "failed"
                    case_record["error"] = f"startup log did not confirm expected placement {expected}<total>"
            if case.get("remote_cache_type"):
                matches = re.findall(
                    r"mixed KV layer=(\d+) device=(\S+) type_k=(\S+) type_v=(\S+) domain=(\S+)",
                    log_text,
                )
                case_record["mixed_format_log_evidence"] = [
                    {"layer": int(layer), "device": device, "type_k": type_k,
                     "type_v": type_v, "domain": domain}
                    for layer, device, type_k, type_v, domain in matches
                ]
                requested = case["remote_cache_type"]
                valid_layout = (
                    len({entry[0] for entry in matches}) == case["remote_layers"]
                    and all(device == "Vulkan0" and type_k == type_v == requested
                            for _, device, type_k, type_v, _ in matches)
                )
                if case_record["state"] == "complete" and not valid_layout:
                    case_record["state"] = "failed"
                    case_record["error"] = "allocated per-layer KV log does not match requested mixed layout"
            controls_requested = any(value is not None for value in (
                args.apu_tdp, args.amd_sclk_prefill, args.amd_sclk_decode,
            ))
            case_record["power_control_log_evidence"] = [
                line for line in case_record["route_log_evidence"]
                if "amd graphics sclk" in line.lower() or "apu tdp: active" in line.lower()
            ]
            expected_controls = required_power_log_fragments(args)
            if case_record["state"] == "complete" and any(
                not any(expected in line for line in case_record["power_control_log_evidence"])
                for expected in expected_controls
            ):
                case_record["state"] = "failed"
                case_record["error"] = "server log did not confirm every requested AMD power control"
            case_record["requested_power_controls"] = {
                "apu_tdp_w": args.apu_tdp, "amd_sclk_prefill_mhz": args.amd_sclk_prefill,
                "amd_sclk_decode_mhz": args.amd_sclk_decode,
                "backend": args.gpu_power_backend or ("amdgpu" if controls_requested else None),
                "amd_device": args.gpu_power_amd_device if args.gpu_power_amd_device is not None
                    else (0 if controls_requested else None),
            }
            case_record["finished_utc"] = utc_now()
            append_jsonl(output_dir / "cases.jsonl", case_record)
    return case_record, prompts


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.modes = parse_names(args.modes, "--modes", {"cuda-only", "remote-vulkan"})
        args.cache_types = parse_names(
            args.cache_types, "--cache-types", {"kvarn4", "q4_0", "q5_0", "q6_0", "q8_0"}
        )
        args.remote_cache_types = parse_names(
            args.remote_cache_types, "--remote-cache-types", {"q4_0", "q5_0", "q6_0", "q8_0"}
        ) if args.remote_cache_types else []
        args.remote_layers = parse_ints(args.remote_layers, "--remote-layers")
        args.prompt_lengths = parse_ints(args.prompt_lengths, "--prompt-lengths")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.remote_cache_types and ("remote-vulkan" not in args.modes or args.cache_types != ["kvarn4"]):
        parser.error("--remote-cache-types requires remote-vulkan mode and --cache-types kvarn4")
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be in 1..65535")
    if any(value <= 0 for value in (args.startup_timeout, args.request_timeout, args.shutdown_timeout)):
        parser.error("startup, request, and shutdown timeouts must be positive")
    if args.repeats < 1 or args.decode_tokens < 1 or args.suffix_decode_tokens < 1 or args.suffix_tokens < 1:
        parser.error("repeats, decode counts, and suffix tokens must be positive")
    if args.tail_tokens < 0 or args.batch_size < 1 or args.ubatch_size < 1:
        parser.error("tail tokens must be >=0 and batch sizes must be positive")
    if any(value is not None and value <= 0 for value in (
        args.apu_tdp, args.amd_sclk_prefill, args.amd_sclk_decode,
    )):
        parser.error("APU TDP and AMD SCLK controls must be positive")
    if args.gpu_power_amd_device is not None and args.gpu_power_amd_device < 0:
        parser.error("--gpu-power-amd-device must be non-negative")
    required_ctx = max(args.prompt_lengths) + args.suffix_tokens + max(args.decode_tokens, args.suffix_decode_tokens)
    ctx_size = args.ctx_size or (((required_ctx + 255) // 256) * 256 + 256)
    if ctx_size < required_ctx:
        parser.error(f"--ctx-size {ctx_size} must cover prompt + suffix + output ({required_ctx})")
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
    output_dir = args.output_dir or Path("bench-results") / f"radeon-q4-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((args.host, args.port))
    except OSError as exc:
        parser.error(f"cannot bind isolated server port {args.host}:{args.port}: {exc}")

    print("Hashing benchmark inputs once...", flush=True)
    binary_hash = None if args.skip_binary_hash else sha256_file(binary)
    model_hash = None if args.skip_model_hash else sha256_file(model)
    ldd_output, runtime_libraries = linked_runtime_libraries(binary)
    script_hash = sha256_file(script_path)
    cases = build_cases(args)
    manifest = {
        "created_utc": utc_now(), "machine_hostname": platform.node(), "platform": platform.platform(),
        "cwd": str(Path.cwd()), "repo": str(repo), "git_commit": git_value(repo, "rev-parse", "HEAD"),
        "git_status_porcelain": git_value(repo, "status", "--short"),
        "server_binary": str(binary), "server_binary_size": binary.stat().st_size,
        "server_binary_sha256": binary_hash, "server_ldd": ldd_output,
        "linked_runtime_libraries": runtime_libraries,
        "harness": str(script_path), "harness_sha256": script_hash,
        "model": str(model), "model_size": model.stat().st_size,
        "model_sha256": model_hash, "listen_host": args.host, "port": args.port,
        "profiling_environment_inherited": {
            key: os.environ.get(key) for key in PROFILE_ENV_KEYS + ("GGML_KVARN_DEBUG_ROUTES",)
        },
        "profiling_environment_child": {
            "GGML_VK_PERF_LOGGER": None, "GGML_BACKEND_COPY_PROFILE": None,
            "GGML_KVARN_DEBUG_ROUTES": "1" if args.debug_kvarn_routes else None,
        },
        "configuration": {
            "modes": args.modes, "cache_types_global_k_and_v": args.cache_types,
            "remote_cache_types_k_and_v": args.remote_cache_types,
            "remote_layers": args.remote_layers, "prompt_lengths_including_bos": args.prompt_lengths,
            "decode_tokens": args.decode_tokens, "suffix_decode_tokens": args.suffix_decode_tokens,
            "suffix_tokens": args.suffix_tokens, "ctx_size": ctx_size,
            "batch_size": args.batch_size, "ubatch_size": args.ubatch_size,
            "tail_tokens": args.tail_tokens, "tail_type": args.tail_type,
            "spec_type": "none", "cache_prompt": True, "temperature": 0,
            "seed": args.seed, "ignore_eos": True, "debug_kvarn_routes": args.debug_kvarn_routes,
            "gpu_power": {
                "backend": args.gpu_power_backend or ("amdgpu" if any(value is not None for value in (
                    args.apu_tdp, args.amd_sclk_prefill, args.amd_sclk_decode,
                )) else None),
                "apu_tdp_w": args.apu_tdp, "amd_sclk_prefill_mhz": args.amd_sclk_prefill,
                "amd_sclk_decode_mhz": args.amd_sclk_decode,
                "amd_device": args.gpu_power_amd_device if args.gpu_power_amd_device is not None
                    else (0 if any(value is not None for value in (
                        args.apu_tdp, args.amd_sclk_prefill, args.amd_sclk_decode,
                    )) else None),
            },
        },
        "telemetry_device": args.telemetry_device,
        "cases": cases,
        "artifacts": ["run.json", "cases.jsonl", "results.jsonl", "requests.jsonl",
                      "prompts.json", "prompt-corpus.txt", "continuation-suffix.txt",
                      "server-<case>.log"],
    }
    write_json(output_dir / "run.json", manifest)
    print(f"Output: {output_dir}\nCases: {len(cases)}; context={ctx_size}; modes={','.join(args.modes)}; "
          f"KV={','.join(args.cache_types)}; tail={args.tail_tokens}", flush=True)

    prompts: tuple[list[int], list[int]] | None = None
    failures = 0
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}]", flush=True)
        result, prompts = run_case(args, case, ctx_size, output_dir, manifest, prompts)
        if result["state"] != "complete":
            failures += 1
    summary = {
        "finished_utc": utc_now(), "case_count": len(cases), "failed_cases": failures,
        "result_rows": sum(1 for _ in (output_dir / "results.jsonl").open(encoding="utf-8"))
                       if (output_dir / "results.jsonl").exists() else 0,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Finished: {summary['case_count']} cases, {failures} failed; results={output_dir}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
