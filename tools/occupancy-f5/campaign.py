#!/usr/bin/env python3
"""Sequential, manifest-driven F5 campaign runner.

The runner never supplies model, sampling, or server flags. A manifest provides
the exact argv and request bytes. Each invocation owns its server process group
and tears down only that group. All observations are saved verbatim; absent
telemetry stays absent and is called out in the report. A timestamped memory
sampler runs alongside each server/request and journals procfs, nvidia-smi,
and DRM sysfs observations when those sources are available. OOM count zero is
reported only after complete log/response/exit evidence; otherwise it is null.

Manifest format and examples are printed by ``--help``. CPU-only tests exercise
the scheduling and parsers without starting a server or making HTTP requests.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

SCHEMA = "f5-campaign-manifest-v1"
REPORT_SCHEMA = "f5-campaign-run-v1"
REPS = 5
OOM_PATTERNS = (
    re.compile(r"out of memory", re.I),
    re.compile(r"oom[-_ ]?(?:kill|killed|killer)", re.I),
    re.compile(r"cuda_error_out_of_memory", re.I),
    re.compile(r"vk_error_out_of_(?:device_)?memory", re.I),
    re.compile(r"std::bad_alloc", re.I),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def schedule(arms: tuple[str, str] = ("baseline", "candidate")) -> list[dict[str, Any]]:
    """Warm both arms, then execute five strictly alternating A/B pairs."""
    a, b = arms
    result = [{"arm": a, "warmup": True, "pair": None},
              {"arm": b, "warmup": True, "pair": None}]
    for pair in range(REPS):
        result.extend({"arm": arm, "warmup": False, "pair": pair} for arm in (a, b))
    return result


def _finite_positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def p95_nearest_rank(values: list[float]) -> float:
    if not values or any(not _finite_positive(v) for v in values):
        raise ValueError("server token timing samples must be a nonempty finite nonnegative array")
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def parse_completion(body: bytes, content_type: str = "application/json") -> dict[str, Any]:
    """Parse ordinary JSON or OpenAI-style SSE and retain the final response.

    SSE delivery cadence is deliberately ignored. F5 decode p95 comes only
    from the server's ``f5_per_token.token_ms`` array.
    """
    text = body.decode("utf-8")
    if "text/event-stream" not in content_type.lower() and not text.lstrip().startswith("data:"):
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("completion response JSON must be an object")
        return value
    candidates: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    # The native chat stream emits a final event carrying timings. Some
    # compatible servers instead emit one final JSON object after streaming.
    for value in reversed(candidates):
        if "f5_per_token" in value or "timings" in value or "usage" in value:
            return value
    if candidates:
        return candidates[-1]
    raise ValueError("SSE response contained no JSON data events")


def extract_timing(response: dict[str, Any]) -> dict[str, Any]:
    timings = response.get("timings")
    if not isinstance(timings, dict):
        raise ValueError("completion response is missing timings object")
    required = ("prompt_ms", "predicted_ms", "prompt_n", "predicted_n")
    missing = [k for k in required if k not in timings]
    if missing:
        raise ValueError("completion timings missing: " + ", ".join(missing))
    for k in ("prompt_ms", "predicted_ms"):
        if not _finite_positive(timings[k]):
            raise ValueError(f"timings.{k} must be finite and nonnegative")
    for k in ("prompt_n", "predicted_n"):
        if type(timings[k]) is not int or timings[k] < 0:
            raise ValueError(f"timings.{k} must be a nonnegative integer")
    result: dict[str, Any] = {
        "prompt_ms": timings["prompt_ms"], "predicted_ms": timings["predicted_ms"],
        "prompt_n": timings["prompt_n"], "predicted_n": timings["predicted_n"],
        "timings": timings,
    }
    # F5 serializers have emitted this object both as a response sibling and
    # inside the timings object. Prefer the nested location when present.
    native = timings.get("f5_per_token", response.get("f5_per_token"))
    result["prefill_ms"] = timings["prompt_ms"]
    ttft = native.get("ttft_ms") if isinstance(native, dict) else None
    if _finite_positive(ttft):
        result["ttft_ms"] = ttft
    else:
        result["ttft_ms"] = None
        result["ttft_metrics_missing"] = "server f5_per_token.ttft_ms absent or invalid"
    if isinstance(native, dict):
        samples = native.get("token_ms")
        if isinstance(samples, list) and samples and all(_finite_positive(x) for x in samples):
            result["server_token_ms"] = samples
            result["server_token_sample_count"] = len(samples)
            result["server_token_p95_ms_nearest_rank"] = p95_nearest_rank(samples)
            result["f5_per_token"] = native
        else:
            result["server_token_ms"] = None
            result["server_token_sample_count"] = 0
            result["server_token_metrics_missing"] = "f5_per_token.token_ms absent, empty, or invalid"
    else:
        result["server_token_ms"] = None
        result["server_token_sample_count"] = 0
        result["server_token_metrics_missing"] = "response has no f5_per_token object"
    return result


def enforce_measured_samples(timing: dict[str, Any], warmup: bool,
                             minimum: int = 32) -> dict[str, Any]:
    """Annotate short measured samples; warmups are allowed to be short."""
    count = timing.get("server_token_sample_count", 0)
    if not warmup and count < minimum:
        timing["server_token_metrics_missing"] = (
            f"measured run requires at least {minimum} non-TTFT token samples; got {count}"
        )
    return timing


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_manifest(manifest: Any) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError(f"manifest schema must be {SCHEMA}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest must contain a nonempty cases array")
    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str) or not case["case_id"]:
            raise ValueError("each case needs a nonempty case_id")
        if case["case_id"] in ids:
            raise ValueError(f"duplicate case_id: {case['case_id']}")
        ids.add(case["case_id"])
        req = case.get("request")
        if not isinstance(req, str) or not req:
            raise ValueError(f"case {case['case_id']} needs request path")
        arms = case.get("arms")
        if not isinstance(arms, dict) or set(arms) != {"baseline", "candidate"}:
            raise ValueError(f"case {case['case_id']} must define baseline and candidate arms")
        for arm, cfg in arms.items():
            if not isinstance(cfg, dict) or not isinstance(cfg.get("server_argv"), list) or not cfg["server_argv"]:
                raise ValueError(f"{case['case_id']}/{arm} needs explicit nonempty server_argv")
            if any(not isinstance(x, str) or not x for x in cfg["server_argv"]):
                raise ValueError(f"{case['case_id']}/{arm} server_argv must be strings")
            if not isinstance(cfg.get("url"), str) or not cfg["url"].startswith(("http://", "https://")):
                raise ValueError(f"{case['case_id']}/{arm} needs explicit HTTP url")
            if not isinstance(cfg.get("ready_url"), str) or not cfg["ready_url"].startswith(("http://", "https://")):
                raise ValueError(f"{case['case_id']}/{arm} needs explicit ready_url")
            timeout = cfg.get("timeout_seconds")
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError(f"{case['case_id']}/{arm} needs positive timeout_seconds")
            collectors = cfg.get("collectors", [])
            if not isinstance(collectors, list):
                raise ValueError(f"{case['case_id']}/{arm} collectors must be an array")
            for collector in collectors:
                if not isinstance(collector, dict) or not isinstance(collector.get("name"), str) or not isinstance(collector.get("argv"), list) or not collector["argv"]:
                    raise ValueError(f"{case['case_id']}/{arm} collectors need name and explicit argv")
            telemetry = cfg.get("telemetry", {})
            if not isinstance(telemetry, dict):
                raise ValueError(f"{case['case_id']}/{arm} telemetry must be an object")
            interval = telemetry.get("interval_seconds", 1.0)
            if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
                raise ValueError(f"{case['case_id']}/{arm} telemetry interval_seconds must be positive")
            nvidia_argv = telemetry.get("nvidia_smi_argv", ["nvidia-smi"])
            if not isinstance(nvidia_argv, list) or not nvidia_argv or any(not isinstance(x, str) for x in nvidia_argv):
                raise ValueError(f"{case['case_id']}/{arm} telemetry nvidia_smi_argv must be a nonempty string array")
            devices = telemetry.get("amd_sysfs_devices", [])
            if not isinstance(devices, list) or any(not isinstance(path, str) for path in devices):
                raise ValueError(f"{case['case_id']}/{arm} telemetry amd_sysfs_devices must be a string array")
        if case.get("request_method", "POST") != "POST":
            raise ValueError("only request_method POST is supported")


def _replace_tokens(argv: list[str], replacements: dict[str, str]) -> list[str]:
    return [arg.format_map(replacements) for arg in argv]


def _read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            try:
                values[fields[0].rstrip(":")] = int(fields[1])
            except ValueError:
                continue
    return values


def _meminfo_bytes(meminfo: dict[str, int], key: str) -> int | None:
    # Linux meminfo reports kB (1024 bytes), not decimal kilobytes.
    return meminfo[key] * 1024 if key in meminfo else None


def _process_rss_bytes(pid: int, proc_root: Path) -> int | None:
    status = (proc_root / str(pid) / "status").read_text(encoding="ascii")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1]) * 1024
    return None


def _host_sample(pid: int, proc_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"method": "linux /proc/{pid}/status + meminfo + vmstat"}
    errors: dict[str, str] = {}
    try:
        result["process_rss_bytes"] = _process_rss_bytes(pid, proc_root)
    except (OSError, ValueError) as exc:
        result["process_rss_bytes"] = None
        errors["process_rss_bytes"] = f"{type(exc).__name__}: {exc}"
    try:
        meminfo = _read_key_values(proc_root / "meminfo")
        result["mem_available_bytes"] = _meminfo_bytes(meminfo, "MemAvailable")
        result["swap_total_bytes"] = _meminfo_bytes(meminfo, "SwapTotal")
        result["swap_free_bytes"] = _meminfo_bytes(meminfo, "SwapFree")
        if result["swap_total_bytes"] is not None and result["swap_free_bytes"] is not None:
            result["swap_used_bytes"] = result["swap_total_bytes"] - result["swap_free_bytes"]
        else:
            result["swap_used_bytes"] = None
    except (OSError, ValueError) as exc:
        for key in ("mem_available_bytes", "swap_total_bytes", "swap_free_bytes", "swap_used_bytes"):
            result[key] = None
        errors["meminfo"] = f"{type(exc).__name__}: {exc}"
    try:
        vmstat = _read_key_values(proc_root / "vmstat")
        result["swap_in_pages"] = vmstat.get("pswpin")
        result["swap_out_pages"] = vmstat.get("pswpout")
    except (OSError, ValueError) as exc:
        result["swap_in_pages"] = result["swap_out_pages"] = None
        errors["vmstat"] = f"{type(exc).__name__}: {exc}"
    result["available"] = all(result.get(key) is not None for key in (
        "process_rss_bytes", "mem_available_bytes", "swap_total_bytes", "swap_free_bytes",
        "swap_in_pages", "swap_out_pages"))
    if errors:
        result["errors"] = errors
    return result


def _mib_bytes(value: str) -> int | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    try:
        return round(float(value) * 1024 * 1024)
    except ValueError:
        return None


def _nvidia_sample(pid: int, command: list[str], command_runner: Callable[..., Any]) -> dict[str, Any]:
    method = "nvidia-smi query-gpu memory.used,memory.free and query-compute-apps used_memory"
    base: dict[str, Any] = {"method": method, "command": command}
    try:
        gpu = command_runner(command + ["--query-gpu=timestamp,memory.used,memory.free",
                                        "--format=csv,noheader,nounits"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=3, check=False)
        if gpu.returncode != 0:
            raise RuntimeError(gpu.stderr.strip() or f"nvidia-smi exit {gpu.returncode}")
        rows = []
        for row in csv.reader(io.StringIO(gpu.stdout)):
            if len(row) >= 3:
                rows.append({"timestamp": row[0].strip(), "used_bytes": _mib_bytes(row[1]),
                             "free_bytes": _mib_bytes(row[2])})
        process_rows = []
        app_error = None
        try:
            apps = command_runner(command + ["--query-compute-apps=pid,used_memory",
                                             "--format=csv,noheader,nounits"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, timeout=3, check=False)
            if apps.returncode != 0:
                raise RuntimeError(apps.stderr.strip() or f"nvidia-smi app query exit {apps.returncode}")
            for row in csv.reader(io.StringIO(apps.stdout)):
                if len(row) >= 2:
                    try:
                        app_pid = int(row[0].strip())
                    except ValueError:
                        continue
                    process_rows.append({"pid": app_pid, "used_bytes": _mib_bytes(row[1])})
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
            app_error = f"{type(exc).__name__}: {exc}"
        matching = [row["used_bytes"] for row in process_rows if row["pid"] == pid and row["used_bytes"] is not None]
        base.update({"available": any(row["used_bytes"] is not None or row["free_bytes"] is not None for row in rows), "devices": rows,
                     "process_apps": process_rows,
                     "server_process_vram_used_bytes": sum(matching) if matching else None,
                     "server_process_visible": bool(matching)})
        if app_error:
            base["process_query_error"] = app_error
        if not base["available"]:
            base["error"] = "nvidia-smi returned no numeric GPU memory values"
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        base.update({"available": False, "devices": [], "process_apps": [],
                     "server_process_vram_used_bytes": None,
                     "server_process_visible": False,
                     "error": f"{type(exc).__name__}: {exc}"})
    return base


def _sysfs_int(device: Path, name: str) -> int | None:
    try:
        value = (device / name).read_text(encoding="ascii").strip()
        return int(value) if value else None
    except (OSError, ValueError):
        return None


def _amd_sysfs_sample(devices: list[str], drm_root: Path) -> dict[str, Any]:
    method = "DRM sysfs mem_info_vram_* and mem_info_gtt_* counters"
    paths = [Path(path) for path in devices] if devices else sorted(drm_root.glob("card*/device"))
    rows = []
    for path in paths:
        item = {"sysfs_device": str(path)}
        for field, filename in (("vram_used_bytes", "mem_info_vram_used"),
                                 ("vram_total_bytes", "mem_info_vram_total"),
                                 ("gtt_used_bytes", "mem_info_gtt_used"),
                                 ("gtt_total_bytes", "mem_info_gtt_total")):
            item[field] = _sysfs_int(path, filename)
        item["available"] = any(item[key] is not None for key in
                                ("vram_used_bytes", "gtt_used_bytes"))
        if item["available"]:
            rows.append(item)
    return {"method": method, "available": bool(rows), "devices": rows,
            "error": None if rows else "no readable DRM VRAM/GTT sysfs counters"}


def _drm_memory_bytes(value: str) -> int | None:
    match = re.fullmatch(r"\s*(\d+)\s*(KiB|MiB)?\s*", value, re.I)
    if not match:
        return None
    amount = int(match.group(1))
    unit = (match.group(2) or "B").lower()
    return amount * {"b": 1, "kib": 1024, "mib": 1024 * 1024}[unit]


def parse_drm_fdinfo(text: str) -> dict[str, Any]:
    """Parse standard DRM client identity and memory fields from fdinfo text.

    drm-memory/drm-total values default to bytes; optional KiB/MiB suffixes
    follow the Linux DRM usage-stats ABI.
    """
    result: dict[str, Any] = {"memory_bytes": {}, "raw_memory_fields": {}}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if key == "drm-client-id":
            result["drm_client_id"] = value
        elif key == "drm-pdev":
            result["drm_pdev"] = value
        elif key == "drm-driver":
            result["drm_driver"] = value
        elif key.startswith("drm-memory-") or key.startswith("drm-total-"):
            parsed = _drm_memory_bytes(value)
            result["memory_bytes"][key] = parsed
            result["raw_memory_fields"][key] = value
    return result


def sample_process_drm(pid: int, proc_root: Path = Path("/proc"), *,
                       fd_reader: Callable[[Path], str] | None = None) -> dict[str, Any]:
    """Read DRM fdinfo for one PID and aggregate unique drm-client-id values.

    Duplicate descriptors for one DRM client describe the same client memory.
    They are retained verbatim in ``fd_records`` but counted once, using the
    maximum observed value per field if duplicate reads raced an update.
    """
    fd_dir = proc_root / str(pid) / "fdinfo"
    read_fd = fd_reader or (lambda path: path.read_text(encoding="utf-8", errors="replace"))
    records: list[dict[str, Any]] = []
    read_errors: list[dict[str, str]] = []
    try:
        entries = sorted(fd_dir.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        return {
            "method": "Linux /proc/<pid>/fdinfo; deduplicate on drm-pdev + drm-client-id",
            "pid": pid, "available": False, "complete": False, "fd_count": 0,
            "unique_client_count": 0, "memory_bytes_by_key": {}, "fd_records": [],
            "read_errors": [{"path": str(fd_dir), "error": f"{type(exc).__name__}: {exc}"}],
            "missing": ["/proc fdinfo directory unreadable"],
        }
    for entry in entries:
        try:
            raw = read_fd(entry)
        except OSError as exc:
            read_errors.append({"path": str(entry), "error": f"{type(exc).__name__}: {exc}"})
            continue
        parsed = parse_drm_fdinfo(raw)
        if not parsed.get("drm_driver") and not parsed["memory_bytes"] and not parsed.get("drm_client_id"):
            continue
        records.append({"fd": entry.name, "raw": raw, **parsed})

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    unkeyed = []
    for record in records:
        client_id = record.get("drm_client_id")
        if client_id is None:
            unkeyed.append(record["fd"])
            continue
        # The kernel ABI requires drm-pdev when client IDs are only unique per
        # device. Older drivers may omit it; in that case preserve driver scope.
        scope = record.get("drm_pdev") or record.get("drm_driver") or "unknown-device"
        key = (scope, client_id)
        group = grouped.setdefault(key, {"drm_pdev_or_driver": scope,
                                         "drm_client_id": client_id,
                                         "fds": [], "memory_bytes": {}})
        group["fds"].append(record["fd"])
        for metric, value in record["memory_bytes"].items():
            if value is not None:
                group["memory_bytes"][metric] = max(group["memory_bytes"].get(metric, 0), value)

    clients = list(grouped.values())
    fields = sorted({field for client in clients for field in client["memory_bytes"]})
    missing = []
    if not records:
        missing.append("no DRM fdinfo records for process")
    if unkeyed:
        missing.append("drm-client-id absent; fd memory excluded from deduplicated totals")
    if read_errors:
        missing.append("one or more DRM fdinfo entries unreadable; totals may be incomplete")
    for field in ("drm-memory-vram", "drm-memory-gtt", "drm-total-vram", "drm-total-gtt"):
        if not clients or any(field not in client["memory_bytes"] for client in clients):
            missing.append(f"{field} absent for one or more DRM clients")
    totals: dict[str, int | None] = {}
    for field in fields:
        if read_errors or unkeyed or any(field not in client["memory_bytes"] for client in clients):
            totals[field] = None
        else:
            totals[field] = sum(client["memory_bytes"][field] for client in clients)
    return {
        "method": "Linux /proc/<pid>/fdinfo; deduplicate on drm-pdev + drm-client-id",
        "pid": pid, "available": bool(clients and fields),
        "complete": bool(clients and fields and not missing and not read_errors and not unkeyed),
        "fd_count": len(records), "unique_client_count": len(clients),
        "duplicate_fd_count": sum(max(0, len(client["fds"]) - 1) for client in clients),
        "unkeyed_fd_count": len(unkeyed), "memory_bytes_by_key": totals,
        "clients": clients, "fd_records": records, "read_errors": read_errors,
        "missing": missing,
    }


def sample_telemetry(pid: int, config: dict[str, Any] | None = None, *,
                     proc_root: Path = Path("/proc"), drm_root: Path = Path("/sys/class/drm"),
                     command_runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """One timestamped host/NVIDIA/AMD memory sample; injectable for CPU fixtures."""
    config = config or {}
    nvidia_command = config.get("nvidia_smi_argv", ["nvidia-smi"])
    sample = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "host": _host_sample(pid, proc_root),
        "cuda": _nvidia_sample(pid, nvidia_command, command_runner),
        "vulkan": _amd_sysfs_sample(config.get("amd_sysfs_devices", []), drm_root),
    }
    sample["vulkan"]["process"] = sample_process_drm(pid, proc_root)
    sample["vulkan"]["available"] = (sample["vulkan"].get("available", False) or
                                      sample["vulkan"]["process"].get("available", False))
    return sample


class TelemetryMonitor:
    """Append timestamped memory samples while the owned server is alive."""

    def __init__(self, output: Path, pid: int, config: dict[str, Any], interval_seconds: float,
                 sampler: Callable[..., dict[str, Any]] = sample_telemetry):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("telemetry interval_seconds must be positive and finite")
        self.output = output
        self.pid = pid
        self.config = config
        self.interval_seconds = interval_seconds
        self.sampler = sampler
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"f5-memory-{pid}", daemon=True)
        self.output_lock = threading.Lock()
        self.sample_count = 0
        self.errors: list[str] = []
        self.started_at = None
        self.ended_at = None
        self.availability = {"host": 0, "cuda": 0, "vulkan": 0}
        self.process_visible_samples = 0
        self.vulkan_process_complete_samples = 0

    def start(self) -> "TelemetryMonitor":
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.thread.start()
        return self

    def _run(self) -> None:
        while True:
            try:
                self._record(self.sampler(self.pid, self.config))
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            if self.stop_event.wait(self.interval_seconds):
                break

    def _record(self, sample: dict[str, Any]) -> None:
        with self.output_lock:
            with self.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(sample, allow_nan=False) + "\n")
                stream.flush()
            self.sample_count += 1
            for backend in self.availability:
                if sample.get(backend, {}).get("available"):
                    self.availability[backend] += 1
            if sample.get("cuda", {}).get("server_process_visible"):
                self.process_visible_samples += 1
            if sample.get("vulkan", {}).get("process", {}).get("complete"):
                self.vulkan_process_complete_samples += 1

    def mark(self, label: str) -> dict[str, Any]:
        """Write an immediate labelled sample, e.g. retained_cycle_1/10."""
        if not label or "\n" in label:
            raise ValueError("phase marker must be a nonempty single-line string")
        sample = self.sampler(self.pid, self.config)
        sample["phase_marker"] = label
        self._record(sample)
        return sample

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join()
        self.ended_at = datetime.now(timezone.utc).isoformat()
        result = {"method": "sample_telemetry: linux procfs + nvidia-smi + DRM sysfs",
                  "interval_seconds": self.interval_seconds,
                  "started_at_utc": self.started_at, "ended_at_utc": self.ended_at,
                  "sample_count": self.sample_count, "sample_path": self.output.name,
                  "available_samples": self.availability,
                  "server_process_vram_visible_samples": self.process_visible_samples,
                  "vulkan_process_complete_samples": self.vulkan_process_complete_samples,
                  "errors": self.errors}
        result["sources_missing"] = [backend for backend, count in self.availability.items()
                                     if count == 0]
        result["complete"] = self.sample_count > 0 and not self.errors
        return result


def summarize_telemetry(path: Path) -> dict[str, Any]:
    """Summarize observed peaks/minima from the raw sample journal."""
    samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def extrema(values: list[int | float | None], operation: Callable[[list[Any]], Any]) -> Any:
        valid = [value for value in values if type(value) in (int, float) and math.isfinite(value)]
        return operation(valid) if valid else None

    host = [sample.get("host", {}) for sample in samples]
    nvidia = [sample.get("cuda", {}) for sample in samples]
    amd = [sample.get("vulkan", {}) for sample in samples]
    amd_devices = [device for item in amd for device in item.get("devices", [])]
    drm_process = [item.get("process", {}) for item in amd]
    process_vram = [item.get("server_process_vram_used_bytes") for item in nvidia]
    swap_in = [item.get("swap_in_pages") for item in host]
    swap_out = [item.get("swap_out_pages") for item in host]
    return {
        "sample_count": len(samples),
        "timestamp_start_utc": samples[0].get("timestamp_utc") if samples else None,
        "timestamp_end_utc": samples[-1].get("timestamp_utc") if samples else None,
        "host": {
            "process_rss_peak_bytes": extrema([item.get("process_rss_bytes") for item in host], max),
            "mem_available_min_bytes": extrema([item.get("mem_available_bytes") for item in host], min),
            "swap_used_peak_bytes": extrema([item.get("swap_used_bytes") for item in host], max),
            "swap_in_pages_delta": (swap_in[-1] - swap_in[0] if len(swap_in) > 1 and
                                    all(type(x) is int for x in (swap_in[0], swap_in[-1])) else None),
            "swap_out_pages_delta": (swap_out[-1] - swap_out[0] if len(swap_out) > 1 and
                                     all(type(x) is int for x in (swap_out[0], swap_out[-1])) else None),
        },
        "cuda": {
            "vram_used_peak_bytes": extrema([device.get("used_bytes") for item in nvidia
                                               for device in item.get("devices", [])], max),
            "vram_free_min_bytes": extrema([device.get("free_bytes") for item in nvidia
                                              for device in item.get("devices", [])], min),
            "server_process_vram_used_peak_bytes": extrema(process_vram, max),
        },
        "vulkan": {
            "gtt_used_peak_bytes": extrema([device.get("gtt_used_bytes") for device in amd_devices], max),
            "gtt_free_min_bytes": extrema([
                device["gtt_total_bytes"] - device["gtt_used_bytes"]
                for device in amd_devices if type(device.get("gtt_total_bytes")) is int and
                type(device.get("gtt_used_bytes")) is int], min),
            "vram_used_peak_bytes": extrema([device.get("vram_used_bytes") for device in amd_devices], max),
            "process": {
                "drm_memory_gtt_peak_bytes": extrema([
                    item.get("memory_bytes_by_key", {}).get("drm-memory-gtt") for item in drm_process], max),
                "drm_total_gtt_peak_bytes": extrema([
                    item.get("memory_bytes_by_key", {}).get("drm-total-gtt") for item in drm_process], max),
                "drm_memory_vram_peak_bytes": extrema([
                    item.get("memory_bytes_by_key", {}).get("drm-memory-vram") for item in drm_process], max),
                "drm_total_vram_peak_bytes": extrema([
                    item.get("memory_bytes_by_key", {}).get("drm-total-vram") for item in drm_process], max),
                "unique_clients_peak": extrema([item.get("unique_client_count") for item in drm_process], max),
                "duplicate_fds_peak": extrema([item.get("duplicate_fd_count") for item in drm_process], max),
                "complete_sample_count": sum(1 for item in drm_process if item.get("complete")),
                "missing": sorted({reason for item in drm_process for reason in item.get("missing", [])}),
            },
        },
    }


def retained_growth_observation(path: Path, limit_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    """Compare per-process allocation samples marked after retained cycle 1/10.

    This is an observation for the existing stability analysis, not a memory
    gate. Global sysfs totals are deliberately not substituted for PID usage.
    """
    samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    markers = {sample.get("phase_marker"): sample for sample in samples if sample.get("phase_marker")}
    first, last = markers.get("retained_cycle_1"), markers.get("retained_cycle_10")
    result: dict[str, Any] = {
        "comparison": "retained_cycle_10 minus retained_cycle_1",
        "limit_bytes": limit_bytes, "observation_only": True,
    }
    if first is None or last is None:
        result.update({"status": "incomplete", "missing_markers": [
            marker for marker, sample in (("retained_cycle_1", first), ("retained_cycle_10", last))
            if sample is None]})
        return result

    def delta(left: Any, right: Any) -> int | None:
        return right - left if type(left) is int and type(right) is int else None

    cuda_first = first.get("cuda", {}).get("server_process_vram_used_bytes")
    cuda_last = last.get("cuda", {}).get("server_process_vram_used_bytes")
    vulkan_first = first.get("vulkan", {}).get("process", {}).get("memory_bytes_by_key", {}).get("drm-memory-gtt")
    vulkan_last = last.get("vulkan", {}).get("process", {}).get("memory_bytes_by_key", {}).get("drm-memory-gtt")
    result["backends"] = {
        "cuda_process_vram": {"cycle_1_bytes": cuda_first, "cycle_10_bytes": cuda_last,
                              "growth_bytes": delta(cuda_first, cuda_last)},
        "vulkan_process_gtt": {"cycle_1_bytes": vulkan_first, "cycle_10_bytes": vulkan_last,
                               "growth_bytes": delta(vulkan_first, vulkan_last)},
    }
    missing = [name for name, row in result["backends"].items() if row["growth_bytes"] is None]
    result["status"] = "incomplete" if missing else "observed"
    if missing:
        result["missing_backend_observations"] = missing
    else:
        result["over_limit_backends"] = [name for name, row in result["backends"].items()
                                          if row["growth_bytes"] > limit_bytes]
    return result


def assess_oom_evidence(server_log: Path, response_path: Path, *, http_status: int | None,
                        exit_code: int | None, runner_error: str | None = None,
                        expected_stop_codes: tuple[int, ...] = (0, -signal.SIGTERM)) -> dict[str, Any]:
    """Count explicit OOM markers; only complete successful evidence can say zero."""
    sources: list[dict[str, Any]] = []
    errors = []
    for label, path in (("server_log", server_log), ("http_response", response_path)):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            matches = [line.strip() for line in lines if any(pattern.search(line) for pattern in OOM_PATTERNS)]
            sources.append({"source": label, "complete": True, "line_count": len(lines), "matches": matches})
        except OSError as exc:
            sources.append({"source": label, "complete": False, "error": f"{type(exc).__name__}: {exc}"})
            errors.append(f"{label}:{type(exc).__name__}")
    if runner_error:
        lines = runner_error.splitlines()
        matches = [line.strip() for line in lines if any(pattern.search(line) for pattern in OOM_PATTERNS)]
        sources.append({"source": "runner_error", "complete": True,
                        "line_count": len(lines), "matches": matches})
    count = sum(len(source.get("matches", [])) for source in sources)
    complete = (not errors and http_status is not None and 200 <= http_status < 300
                and exit_code in expected_stop_codes and runner_error is None)
    result = {"oom_count": count if count > 0 or complete else None,
              "complete": complete, "method": "case-insensitive explicit OOM marker scan of server.log and response.raw",
              "sources": sources}
    if not complete:
        result["incomplete_reason"] = ("missing readable logs/response, successful HTTP completion, "
                                       "or controlled server exit; zero OOM is not asserted")
    return result


def _request(url: str, payload: bytes, timeout: float) -> tuple[int | None, bytes, str | None, float]:
    request = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
    }, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), response.headers.get("Content-Type", ""), (time.monotonic() - started) * 1000
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", ""), (time.monotonic() - started) * 1000


def _health(url: str, process: subprocess.Popen[bytes], deadline: float) -> None:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before ready (exit={process.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    raise TimeoutError(f"server readiness timeout: {url}")


def _stop_owned(proc: subprocess.Popen[bytes], grace: float = 5.0) -> dict[str, Any]:
    """Stop only this subprocess's new process group; never process-name match."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    return {"exit_code": proc.returncode, "cleanup": "owned_process_group_stopped"}


def _run_collector(cfg: dict[str, Any], *, pid: int, pgid: int, arm: str,
                   run_id: str, phase: str, outdir: Path, timeout: float) -> dict[str, Any]:
    argv = _replace_tokens(cfg["argv"], {
        "pid": str(pid), "pgid": str(pgid), "arm": arm,
        "run_id": run_id, "phase": phase, "output_dir": str(outdir),
    })
    cp = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        timeout=timeout, check=False, text=True)
    record: dict[str, Any] = {
        "name": cfg["name"], "argv": argv, "exit_code": cp.returncode,
        "stdout": cp.stdout, "stderr": cp.stderr,
    }
    try:
        record["json"] = json.loads(cp.stdout)
    except json.JSONDecodeError:
        record["parse_error"] = "collector stdout is not JSON"
    return record


def _run_one(case: dict[str, Any], arm: str, phase: str, sequence: dict[str, Any],
             request_bytes: bytes, case_dir: Path, manifest_dir: Path,
             collector_timeout: float) -> dict[str, Any]:
    cfg = case["arms"][arm]
    rid = f"{case['case_id']}-{arm}-{'warmup' if sequence['warmup'] else 'rep'+str(sequence['pair'])}-{uuid.uuid4().hex[:10]}"
    run_dir = case_dir / "runs" / rid
    run_dir.mkdir(parents=True)
    (run_dir / "request.json").write_bytes(request_bytes)
    record: dict[str, Any] = {
        "run_id": rid, "case_id": case["case_id"], "arm": arm,
        "warmup": sequence["warmup"], "sequence_index": sequence["sequence_index"],
        "pair": sequence["pair"], "request_sha256": sha256(request_bytes),
        "request_bytes": len(request_bytes), "request_url": cfg["url"],
        "server_argv": cfg["server_argv"], "server_env": cfg.get("env", {}),
        "started_at_unix": time.time(),
    }
    (run_dir / "command.json").write_text(json.dumps({"argv": cfg["server_argv"], "env": cfg.get("env", {})}, indent=2) + "\n")
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in cfg.get("env", {}).items()})
    server_log = (run_dir / "server.log").open("wb")
    proc: subprocess.Popen[bytes] | None = None
    response_bytes = b""
    status: int | None = None
    content_type = ""
    req_ms: float | None = None
    error: str | None = None
    monitor: TelemetryMonitor | None = None
    try:
        proc = subprocess.Popen(cfg["server_argv"], cwd=str(manifest_dir), env=env,
                                stdout=server_log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        record["server_pid"] = proc.pid
        record["server_pgid"] = proc.pid
        telemetry_config = dict(cfg.get("telemetry", {}))
        telemetry_interval = float(telemetry_config.get("interval_seconds", 1.0))
        monitor = TelemetryMonitor(run_dir / "telemetry.jsonl", proc.pid,
                                   telemetry_config, telemetry_interval).start()
        _health(cfg["ready_url"], proc, time.monotonic() + float(cfg["timeout_seconds"]))
        status, response_bytes, content_type, req_ms = _request(
            cfg["url"], request_bytes, float(cfg["timeout_seconds"]))
        (run_dir / "response.raw").write_bytes(response_bytes)
        record.update({"http_status": status, "response_content_type": content_type,
                       "client_wall_ms": req_ms})
        if status is not None and 200 <= status < 300:
            try:
                parsed = parse_completion(response_bytes, content_type)
                record["response"] = parsed
                try:
                    record["timing"] = enforce_measured_samples(
                        extract_timing(parsed), sequence["warmup"])
                except ValueError as exc:
                    record["timing_missing"] = str(exc)
            except (ValueError, UnicodeDecodeError) as exc:
                record["response_parse_error"] = str(exc)
        else:
            record["response_parse_error"] = "HTTP response was not successful"
    except Exception as exc:  # save failure evidence before stopping our process
        error = f"{type(exc).__name__}: {exc}"
        record["runner_error"] = error
    finally:
        if proc is not None:
            if monitor is not None:
                monitor_result = monitor.stop()
                try:
                    sample_path = run_dir / "telemetry.jsonl"
                    monitor_result["summary"] = summarize_telemetry(sample_path)
                    monitor_result["summary"]["retained_growth_observation"] = retained_growth_observation(sample_path)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    monitor_result["summary_error"] = f"{type(exc).__name__}: {exc}"
                (run_dir / "telemetry.meta.json").write_text(
                    json.dumps(monitor_result, indent=2, allow_nan=False) + "\n")
                record["telemetry_observations"] = monitor_result
                record["telemetry"] = {
                    backend: {
                        "method": monitor_result["method"],
                        "sample_path": monitor_result["sample_path"],
                        "sample_count": monitor_result["sample_count"],
                        "available_samples": monitor_result["available_samples"][backend],
                        "available": monitor_result["available_samples"][backend] > 0,
                        "server_process_vram_visible_samples": monitor_result[
                            "server_process_vram_visible_samples"] if backend == "cuda" else None,
                        "process_complete_samples": monitor_result[
                            "vulkan_process_complete_samples"] if backend == "vulkan" else None,
                    }
                    for backend in ("cuda", "vulkan", "host")
                }
            for collector in cfg.get("collectors", []):
                try:
                    record.setdefault("collectors", []).append(_run_collector(
                        collector, pid=proc.pid, pgid=proc.pid, arm=arm, run_id=rid,
                        phase=phase, outdir=run_dir, timeout=collector_timeout))
                except Exception as exc:
                    record.setdefault("collectors", []).append({"name": collector.get("name"),
                        "runner_error": f"{type(exc).__name__}: {exc}"})
            record.update(_stop_owned(proc))
        else:
            record["exit_code"] = None
        server_log.close()
    record["runner_error"] = error if error else record.get("runner_error")
    oom_evidence = assess_oom_evidence(run_dir / "server.log", run_dir / "response.raw",
                                       http_status=record.get("http_status"),
                                       exit_code=record.get("exit_code"),
                                       runner_error=record.get("runner_error"))
    record["oom_evidence"] = oom_evidence
    record["oom_count"] = oom_evidence["oom_count"]
    if record["oom_count"] is None:
        record["oom_count_missing"] = oom_evidence.get("incomplete_reason")
    for collected in record.get("collectors", []):
        value = collected.get("json")
        if isinstance(value, dict) and isinstance(value.get("telemetry"), dict):
            record.setdefault("collector_telemetry", {}).update(value["telemetry"])
    if "timing" not in record:
        record["timing_missing"] = record.get("timing_missing", "no valid HTTP completion timing")
    record["metrics_missing"] = []
    token_metric_issue = record.get("timing", {}).get("server_token_metrics_missing")
    if token_metric_issue:
        record["metrics_missing"].append(token_metric_issue)
    elif record.get("timing", {}).get("server_token_ms") is None:
        record["metrics_missing"].append("server-side per-token decode timing array")
    if record.get("timing", {}).get("ttft_ms") is None:
        record["metrics_missing"].append(record["timing"].get(
            "ttft_metrics_missing", "server-side TTFT unavailable"))
    if record["oom_count"] is None:
        record["metrics_missing"].append(record.get(
            "oom_count_missing", "explicit run.oom_count unavailable"))
    if not isinstance(record.get("telemetry"), dict) or not record["telemetry"]:
        record["metrics_missing"].append("per-process CUDA/Vulkan/host memory observations")
    else:
        for backend in ("cuda", "vulkan", "host"):
            observation = record["telemetry"].get(backend)
            if not isinstance(observation, dict) or not observation.get("available"):
                record["metrics_missing"].append(f"{backend} memory observation unavailable in live sampler")
        cuda_observation = record["telemetry"].get("cuda", {})
        if cuda_observation.get("available") and not cuda_observation.get("server_process_vram_visible_samples"):
            record["metrics_missing"].append("NVIDIA server-process VRAM usage unavailable")
        vulkan_observation = record["telemetry"].get("vulkan", {})
        monitor_meta = record.get("telemetry_observations", {})
        if (vulkan_observation.get("available") and
                monitor_meta.get("vulkan_process_complete_samples") != monitor_meta.get("sample_count")):
            record["metrics_missing"].append("per-PID Vulkan DRM fdinfo incomplete or missing in one or more samples")
    if not record.get("telemetry_observations", {}).get("complete", False):
        record["metrics_missing"].append("live telemetry sampler incomplete or produced no samples")
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def run_campaign(manifest_path: Path, outdir: Path, collector_timeout: float = 30.0) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    _validate_manifest(manifest)
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "manifest.json").write_bytes(manifest_path.read_bytes())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA, "campaign_id": manifest.get("campaign_id"),
        "manifest_sha256": sha256(manifest_path.read_bytes()),
        "manifest_path": str(manifest_path), "runs": [],
        "missing_evidence": [],
        "acceptance_scope": "runner evidence only; no quality, capacity, restore, cycle, or full F5 acceptance claim",
    }
    for case in manifest["cases"]:
        case_dir = outdir / case["case_id"]
        case_dir.mkdir()
        request_path = (manifest_path.parent / case["request"]).resolve()
        request_bytes = request_path.read_bytes()
        try:
            request_obj = json.loads(request_bytes)
            if not isinstance(request_obj, dict):
                raise ValueError("request JSON must be object")
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid request JSON for {case['case_id']}: {exc}") from exc
        (case_dir / "request.source.json").write_bytes(request_bytes)
        case_record = {"case_id": case["case_id"], "request_path": str(request_path),
                       "request_sha256": sha256(request_bytes), "request_bytes": len(request_bytes)}
        report.setdefault("cases", []).append(case_record)
        for index, entry in enumerate(schedule()):
            entry = dict(entry, sequence_index=index)
            run = _run_one(case, entry["arm"], "warmup" if entry["warmup"] else "measured",
                           entry, request_bytes, case_dir, manifest_path.parent, collector_timeout)
            report["runs"].append(run)
            for missing in run["metrics_missing"]:
                report["missing_evidence"].append({"run_id": run["run_id"], "item": missing})
            # Fail closed on request/command problems and HTTP or server failures,
            # but persist the partial report for diagnosis.
            if run.get("runner_error") or run.get("exit_code") not in (0, -signal.SIGTERM, None) or not (run.get("http_status") and 200 <= run["http_status"] < 300):
                break
    report["complete_schedule"] = all(sum(1 for r in report["runs"] if r["case_id"] == c["case_id"]) == 12
                                       for c in manifest["cases"])
    report["full_f5_acceptance"] = False
    (outdir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=["run"], help="run sequential campaign")
    parser.add_argument("manifest", type=Path, help="explicit f5-campaign-manifest-v1 JSON")
    parser.add_argument("--out", required=True, type=Path, help="new output directory; must not already exist")
    parser.add_argument("--collector-timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        result = run_campaign(args.manifest, args.out, args.collector_timeout)
        print(json.dumps({"report": str(args.out / "report.json"),
                          "runs": len(result["runs"]), "complete_schedule": result["complete_schedule"],
                          "missing_evidence_count": len(result["missing_evidence"]),
                          "full_f5_acceptance": False}, indent=2))
        return 0 if result["complete_schedule"] and not result["missing_evidence"] else 1
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
