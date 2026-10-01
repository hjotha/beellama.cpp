#!/usr/bin/env python3
"""Stage 3: actual 27B mixed-KV validation driver (planned; GPU ownership is
the parent's explicit grant).

Scope: STATIC mixed runtime and remote formats (Q4/Q5/Q6/Q8). This driver
does NOT perform or claim a KVarN->Qx conversion: it starts fresh servers with
the mixed profile, so there is no handoff transfer inside S1-S5. Actual
conversion handoff acceptance is a separate stage H (adaptive local->mixed,
same semantics as the shared runner scenario B) that requires the configured
handoff evidence marker; without it H stays pending_evidence (exit 1), never
falsely complete.

Per case it assesses:
  - actual occupied context (cache_n + prompt_n + predicted_n vs capacity),
  - memory: incremental remote KV estimate (KV only, never total RAM) plus
    before/after snapshots (labeled; not peak) and an optional threaded
    in-request sampler for real peak (MemAvailable, GTT, CUDA, server VmRSS/
    VmHWM); shared iGPU RAM counted once, CUDA VRAM separate;
  - prefill/decode throughput and prefix reuse (cache_n>0; complete
    recomputation rejected via cache_n==0; cache_reprocessed_n/cache_lcp_n/
    cache_reason recorded, never guards),
  - first-predicted-token top-k distribution comparison vs the all-KVarN
    baseline at the identical frozen prompt and identical cadence: text-keyed
    top-k overlap, retained mass and mean prob delta. This is NOT a KL, NOT a
    full KLD, and NOT conversion accuracy (later generated prefixes diverge
    and their conditional distributions are incompatible; only position 0 is
    compared).

Reuses the frozen test-mixed-kv-transition.py helpers (own-PGID lifecycle,
frozen prompts, telemetry, group cleanup) and the MTP power log verification.
Never writes production configs; power values are argv passthrough only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import re
import shlex
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
_transition = importlib.util.spec_from_file_location(
    "test_mixed_kv_transition", _TOOLS_DIR / "test-mixed-kv-transition.py")
assert _transition and _transition.loader
r = importlib.util.module_from_spec(_transition)
_transition.loader.exec_module(r)

_mtp = importlib.util.spec_from_file_location("test_mixed_kv_mtp",
                                              _TOOLS_DIR / "test-mixed-kv-mtp.py")
assert _mtp and _mtp.loader
m = importlib.util.module_from_spec(_mtp)
_mtp.loader.exec_module(m)

DEFAULT_MODEL = Path("/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf")
DEFAULT_BINARY = Path("build-optimized/bin/llama-server")
REMOTE_TYPES = ("q4_0", "q5_0", "q6_0", "q8_0")
# Incremental per-token per-layer remote Qx bytes (Qwen D256 hkv4, 1024-wide):
REMOTE_BYTES_PER_TOKEN = {"q4_0": 1152, "q5_0": 1408, "q6_0": 1664, "q8_0": 2176}
# KVarN4 local per layer: 1120 B/token records + ~2.1 MiB fixed stage+tail.
KVARN_RECORDS_PER_TOKEN = 1120
KVARN_FIXED_PER_LAYER_MIB = 2.1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18094)
    parser.add_argument("--stages", default="S1,S2,S3",
                        help="comma-separated: S1 baseline, S2 mixed 8K, S3 mixed 16K, "
                             "S4 optional 32K/64K, S5 200K-capacity probe (opt-in), "
                             "H handoff acceptance (requires evidence marker)")
    parser.add_argument("--remote-kv-types", default="q4_0",
                        help="remote Qx types: q4_0,q5_0,q6_0,q8_0")
    parser.add_argument("--remote-attn-layers", default="1",
                        help="explicit layer count, or 'auto'")
    parser.add_argument("--capacity", type=int, default=16384,
                        help="--ctx-size capacity for the server (S2/S3 use this; "
                             "S1 uses prompt + margin)")
    parser.add_argument("--ctx-size", type=int, default=16384,
                        help="fallback --ctx-size (overridden per case by capacity)")
    parser.add_argument("--prompt-tokens", type=int, default=32768,
                        help="frozen corpus size for POST /tokenize; must cover the "
                             "largest prompt length + suffix")
    parser.add_argument("--prompt-short", type=int, default=300,
                        help="stage H pre-gate prompt tokens")
    parser.add_argument("--prompt-long", type=int, default=700,
                        help="stage H gate prompt tokens")
    parser.add_argument("--prompt-lengths", default="8192,16384",
                        help="occupied prompt token counts (including BOS)")
    parser.add_argument("--long-prompt-lengths", default="32768,65536,98304,200000",
                        help="S4 occupied prompt lengths; capacities are aligned >= "
                             "prompt+suffix+decode with reserve margin")
    parser.add_argument("--memory-hard-floor-mib", type=float, default=1024.0,
                        help="runtime UMA guard: abort the case when MemAvailable drops "
                             "below this floor before a request")
    parser.add_argument("--suffix-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--n-probs", type=int, default=10,
                        help="top-n probabilities returned per generated token")
    parser.add_argument("--spec-type", choices=("draft-mtp", "none"), default="draft-mtp")
    parser.add_argument("--memory-cap-fraction", type=float, default=0.5,
                        help="max fraction of MemAvailable projected for incremental "
                             "remote KV (0..1); the S5 opt-in probe does NOT bypass it")
    parser.add_argument("--time-budget-seconds", type=float, default=1800,
                        help="per-case wall budget; request timeouts are capped to it")
    parser.add_argument("--sample-memory-interval", type=float, default=0.0,
                        help=">0: threaded in-request memory sampler interval (s) for "
                             "real peak (MemAvailable/GTT/CUDA/server VmRSS+VmHWM)")
    parser.add_argument("--handoff-cache-ram-mib", type=int, default=2048,
                        help="stage H --cache-ram in MiB (27B snapshot working set)")
    parser.add_argument("--handoff-evidence-markers", default="",
                        help="stage H required conversion marker; empty -> pending_evidence")
    parser.add_argument("--allow-capacity-200k", action="store_true",
                        help="S5: 200K-capacity allocation probe (estimate-labeled, "
                             "NOT full occupancy; occupies --s5-occupied tokens only)")
    parser.add_argument("--s5-occupied", type=int, default=8192,
                        help="S5 occupied prompt tokens (never full 200K claim)")
    parser.add_argument("--mock-endpoint", action="store_true",
                        help="test-only: run the real CLI/request logic against a fake "
                             "HTTP endpoint with a recording ServerInstance stub; no GPU")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--cuda-device", default="CUDA0")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--shutdown-timeout", type=float, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--telemetry-device", default="auto",
                        help="AMD DRM sysfs dir, or 'auto' (vendor discovery), or 'none'")
    parser.add_argument("--apu-tdp", type=int, default=20)
    parser.add_argument("--amd-sclk-prefill", type=int, default=2700)
    parser.add_argument("--amd-sclk-decode", type=int, default=2700)
    parser.add_argument("--skip-model-hash", action="store_true")
    parser.add_argument("--skip-binary-hash", action="store_true")
    parser.add_argument("--self-check", action="store_true",
                        help="run built-in unit checks (no server/GPU) and exit")
    return parser


# ---------------------------------------------------------------------------
# Incremental projections (estimates, labeled) and memory ramping
# ---------------------------------------------------------------------------

def project_remote_mib(qx: str, capacity: int, layers: int) -> float:
    """ESTIMATE: incremental remote Qx capacity bytes (KV only, never total RAM)."""
    return REMOTE_BYTES_PER_TOKEN[qx] * capacity * layers / (1024.0 * 1024.0)


def project_local_mib(capacity: int, layers: int) -> float:
    """ESTIMATE: incremental KVarN4 capacity bytes (records + fixed stage/tail)."""
    return (KVARN_RECORDS_PER_TOKEN * capacity / (1024.0 * 1024.0) +
            KVARN_FIXED_PER_LAYER_MIB) * layers


def parse_ints(value: str, name: str, minimum: int = 1) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not values or any(item < minimum for item in values):
        raise argparse.ArgumentTypeError(f"{name} must contain integers >= {minimum}")
    return values


def long_capacity(prompt_len: int, suffix_tokens: int, decode_tokens: int,
                  align: int = 256) -> int:
    """S4 capacity: 256-aligned and >= prompt+suffix+decode with a 512-token
    reserve margin; lengths near 98K/200K snap to the normal desired
    102400/204800 capacities so allocation behavior is comparable."""
    need = prompt_len + suffix_tokens + decode_tokens
    aligned = ((need + align - 1) // align) * align
    margin = ((need + 512 + align - 1) // align) * align
    if prompt_len > 90000:
        for capacity in (102400, 204800):
            if capacity >= margin:
                return capacity
    return max(margin, aligned)


def required_corpus_tokens(args: argparse.Namespace) -> int:
    """Smallest frozen corpus that covers every chosen occupied length + 1.
    A literal --prompt-tokens smaller than this must ERROR, never silently
    truncate a prefix."""
    def as_ints(value: Any, name: str) -> list[int]:
        return value if isinstance(value, list) else parse_ints(value, name)
    lengths: list[int] = []
    stages = getattr(args, "stages", None) or []
    if "S1" in stages:
        lengths += as_ints(args.prompt_lengths, "--prompt-lengths")
    if "S4" in stages:
        lengths += as_ints(args.long_prompt_lengths, "--long-prompt-lengths")
    if "S5" in stages:
        lengths.append(args.s5_occupied)
    if "H" in stages:
        lengths += [args.prompt_short, args.prompt_long]
    if not lengths:
        lengths = as_ints(args.prompt_lengths, "--prompt-lengths")
    return max(lengths) + 1


def mem_available_mib() -> float:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key == "MemAvailable":
                return int(value.strip().split()[0]) / 1024.0
    except OSError:
        pass
    return math.inf


def ramp_guard(args: argparse.Namespace, qx: str, capacity: int, remote_layers: int,
               case_id: str) -> tuple[list[str], float]:
    """Conservative ramp over the INCREMENTAL remote KV projection (counted
    once against shared RAM). Applies to every case including the S5 opt-in
    probe; baseline (remote_layers <= 0) is never gated."""
    warnings: list[str] = []
    if remote_layers <= 0:
        return warnings, 0.0
    projected = project_remote_mib(qx, capacity, remote_layers)
    available = mem_available_mib()
    ceiling = available * args.memory_cap_fraction
    if projected > ceiling:
        warnings.append(
            f"{case_id}: incremental remote KV projection {projected:.1f} MiB (ESTIMATE) "
            f"exceeds memory ceiling {ceiling:.1f} MiB = {args.memory_cap_fraction:.0%} of "
            f"MemAvailable {available:.1f} MiB; case skipped")
    return warnings, projected


def discover_amdgpu_drm() -> str:
    """Vendor discovery for the AMD DRM sysfs device (vendor 0x1002). Returns
    'none' when no AMD card is found; the caller records what was chosen."""
    try:
        for card in sorted(Path("/sys/class/drm").glob("card*")):
            vendor_path = card / "device" / "vendor"
            if vendor_path.is_file():
                vendor = vendor_path.read_text(encoding="utf-8").strip()
                if vendor == "0x1002":
                    return str(card / "device")
    except OSError:
        pass
    return "none"


# ---------------------------------------------------------------------------
# Quality: first-predicted top-k distribution comparison (limited, labeled)
# ---------------------------------------------------------------------------

def cadence_equal(ta: dict[str, Any], tb: dict[str, Any]) -> bool:
    """True only when every count field is an equal non-None integer. Missing
    fields (None == None) never pass."""
    for key in ("cache_n", "prompt_n", "predicted_n"):
        va = ta.get(key)
        vb = tb.get(key)
        if not (isinstance(va, int) and isinstance(vb, int) and va == vb):
            return False
    return True


def first_distribution_compare(baseline_probs: list[dict[str, Any]],
                               mixed_probs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Compares ONLY the first predicted token's top-k distribution, at the
    identical frozen prompt and identical cadence (caller must verify). The
    real server schema is per-token {"id": int, "token": str, "bytes": [...],
    "logprob": float, "top_logprobs": [{"id", "token", "bytes", "logprob"}]};
    the comparison is keyed by the INTEGER token id (no text-key collapse) and
    uses exp(logprob), finite-checked. Missing top-k returns None and is
    reported, never fabricated. This is NOT a KL, NOT a full KLD, and NOT
    conversion accuracy."""
    def row0(probs: list[dict[str, Any]]) -> dict[int, float] | None:
        if not isinstance(probs, list) or not probs:
            return None
        first = probs[0]
        if not isinstance(first, dict):
            return None
        top = first.get("top_logprobs")
        if not isinstance(top, list) or not top:
            return None
        dist: dict[int, float] = {}
        for item in top:
            if not isinstance(item, dict):
                continue
            token_id = item.get("id")
            logprob = item.get("logprob")
            if not isinstance(token_id, int) or not isinstance(logprob, (int, float)):
                continue
            logprob_f = float(logprob)
            if not math.isfinite(logprob_f):
                continue
            dist[token_id] = math.exp(logprob_f)
        return dist or None

    da = row0(baseline_probs)
    db = row0(mixed_probs)
    if da is None or db is None:
        return None
    union_ids = list(dict.fromkeys(list(da) + list(db)))
    overlap = sum(1 for token_id in da if token_id in db)
    retained_mass = sum(da[token_id] for token_id in da if token_id in db)
    mean_delta = sum(abs(da.get(token_id, 0.0) - db.get(token_id, 0.0))
                     for token_id in union_ids) / len(union_ids)
    return {
        "baseline_topk_size": len(da), "mixed_topk_size": len(db),
        "overlap_token_ids": overlap, "union_token_ids": len(union_ids),
        "retained_baseline_mass": retained_mass,
        "mean_abs_prob_delta": mean_delta,
        "label": ("first-predicted top-k comparison keyed by INTEGER token id, "
                  "exp(logprob) finite-checked, identical frozen prompt and "
                  "identical cadence only; NOT KL, NOT full KLD, NOT conversion "
                  "accuracy"),
    }


# ---------------------------------------------------------------------------
# Optional threaded in-request memory sampler (real peak)
# ---------------------------------------------------------------------------

class MemorySampler:
    """Threaded in-request memory sampler. Sysfs values are converted to ints
    (never lexicographic min/max over strings). CUDA memory is sampled via
    nvidia-smi with a bounded timeout. The sampled peak is the sampled peak,
    not a guaranteed absolute peak, and is labeled as such. Sampler errors are
    recorded, never silently dropped."""

    def __init__(self, args: argparse.Namespace, interval: float, server_pid: int | None):
        self.telemetry_device = args.telemetry_device
        # nvidia-smi runs per sample; keep the interval at >= 0.5 s.
        self.interval = max(interval, 0.5) if interval > 0 else 0.0
        self.server_pid = server_pid
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.interval <= 0:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import shutil
        nvidia_smi = shutil.which("nvidia-smi")
        while not self._stop.is_set():
            sample: dict[str, Any] = {"t": time.monotonic()}
            try:
                for line in Path("/proc/meminfo").read_text(
                        encoding="utf-8", errors="replace").splitlines():
                    key, _, value = line.partition(":")
                    if key in {"MemAvailable", "MemFree"}:
                        sample[f"system_{key}_kib"] = int(value.strip().split()[0])
                if self.telemetry_device not in ("none", "auto"):
                    for field in ("mem_info_gtt_used", "mem_info_vram_used",
                                  "gpu_busy_percent"):
                        path = Path(self.telemetry_device) / field
                        try:
                            text = path.read_text(encoding="utf-8").strip()
                            sample[field] = int(text.split()[0])
                        except (OSError, ValueError, IndexError):
                            pass
                if self.server_pid is not None:
                    try:
                        status = Path(f"/proc/{self.server_pid}/status").read_text(
                            encoding="utf-8", errors="replace")
                        for line in status.splitlines():
                            key, _, value = line.partition(":")
                            if key in {"VmRSS", "VmHWM"}:
                                sample[f"server_{key}_kib"] = int(value.strip().split()[0])
                    except OSError:
                        pass
                if nvidia_smi:
                    try:
                        import subprocess
                        result = subprocess.run(
                            [nvidia_smi, "--query-gpu=memory.used,memory.total",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=2, check=False)
                        if result.returncode == 0 and result.stdout.strip():
                            parts = result.stdout.strip().split(",")
                            sample["cuda_memory_used_mib"] = int(parts[0].strip())
                            sample["cuda_memory_total_mib"] = int(parts[1].strip())
                    except (OSError, subprocess.SubprocessError, ValueError) as exc:
                        self.errors.append(f"nvidia-smi: {type(exc).__name__}: {exc}")
            except Exception as exc:  # never die silently
                self.errors.append(f"sampler: {type(exc).__name__}: {exc}")
            self.samples.append(sample)
            self._stop.wait(self.interval)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"enabled": False}
        keys = ("system_MemAvailable_kib", "system_MemFree_kib",
                "mem_info_gtt_used", "mem_info_vram_used", "gpu_busy_percent",
                "server_VmRSS_kib", "server_VmHWM_kib",
                "cuda_memory_used_mib", "cuda_memory_total_mib")
        out: dict[str, Any] = {"enabled": True, "sample_count": len(self.samples),
                               "interval_seconds": self.interval,
                               "label": "sampled peak over requests only; the startup "
                                        "allocation phase is a snapshot, not sampled",
                               "errors": self.errors}
        for key in keys:
            values = [sample[key] for sample in self.samples if key in sample]
            if values:
                out[f"{key}_min"] = min(values)
                out[f"{key}_max"] = max(values)
                out[f"{key}_median"] = sorted(values)[len(values) // 2]
        return out

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Case execution (own PGID via the shared ServerInstance)
# ---------------------------------------------------------------------------

def case_extra_args(args: argparse.Namespace, qx: str | None,
                    remote_layers: str) -> list[str]:
    extra = [
        "--spec-type", args.spec_type,
        "--cache-ram", "0", "--ctx-checkpoints", "1",
    ]
    if args.spec_type == "draft-mtp":
        extra += ["--spec-draft-type-k", "kvarn4", "--spec-draft-type-v", "kvarn4",
                  "--spec-draft-n-max", "4"]
    if qx is not None:
        extra += ["--remote-attn", "vulkan:0",
                  "--remote-attn-layers", remote_layers,
                  "--remote-attn-cache-type-k", qx,
                  "--remote-attn-cache-type-v", qx]
    return extra


def mem_snapshot_once(args: argparse.Namespace) -> dict[str, Any]:
    """One-shot memory snapshot (startup allocation phase); recorded as a
    snapshot, never labeled as a peak."""
    snapshot: dict[str, Any] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition(":")
            if key in {"MemAvailable", "MemFree"}:
                snapshot[f"system_{key}_kib"] = int(value.strip().split()[0])
    except OSError:
        pass
    if args.telemetry_device not in ("none", "auto"):
        for field in ("mem_info_gtt_used", "mem_info_vram_used", "gpu_busy_percent"):
            path = Path(args.telemetry_device) / field
            try:
                snapshot[field] = int(path.read_text(encoding="utf-8").strip().split()[0])
            except (OSError, ValueError, IndexError):
                pass
    return snapshot


def runtime_uma_guard(args: argparse.Namespace) -> None:
    """Capacity-safety guard checked before every request: abort the case when
    MemAvailable drops below the hard floor. No retry loop is ever started."""
    if mem_available_mib() < args.memory_hard_floor_mib:
        raise RuntimeError(
            f"UMA guard: MemAvailable {mem_available_mib():.0f} MiB below hard floor "
            f"{args.memory_hard_floor_mib:.0f} MiB; case aborted (no retry)")


PROMPT_EVAL_RE = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")


def speculation_placement_failures(spec_type: str, full_log: str) -> list[str]:
    if spec_type == "draft-mtp":
        return m.mtp_placement_failures(full_log)
    return r.target_only_violations(full_log)


def partial_prefill_from_log(full_log: str) -> dict[str, Any] | None:
    """Last real progress/timing event, including unfinished prefills."""
    progress_re = re.compile(
        r"prompt processing, n_tokens =\s*(\d+), progress =\s*([\d.]+), "
        r"t =\s*([\d.]+) s /\s*([\d.]+) tokens per second")
    for line in reversed(full_log.splitlines()):
        progress = progress_re.search(line)
        if progress:
            return {"processed_tokens": int(progress.group(1)),
                    "ms": float(progress.group(3)) * 1000,
                    "progress_fraction_logged": float(progress.group(2)),
                    "cumulative_tps_logged": float(progress.group(4)),
                    "source": "server progress line (rounded timing as logged)",
                    "log_line": line, "outcome": "request_timeout"}
        match = PROMPT_EVAL_RE.search(line)
        if match:
            return {"processed_tokens": int(match.group(2)),
                    "ms": float(match.group(1)),
                    "source": "server log prompt eval line",
                    "log_line": line, "outcome": "request_timeout"}
    return None


def run_case(args: argparse.Namespace, output_dir: Path, run_meta: dict[str, Any],
             case_id: str, qx: str | None, remote_layers: str,
             capacity: int, prompt_len: int,
             baseline_quality: dict[str, Any] | None,
             frozen: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    base_url = f"http://{args.host}:{args.port}"
    records_path = output_dir / "requests.jsonl"
    results_path = output_dir / "results.jsonl"
    n_remote_estimate = int(remote_layers) if remote_layers.isdigit() else 16
    case_record: dict[str, Any] = {
        "case_id": case_id, "state": "starting", "remote_kv_type": qx,
        "remote_attn_layers": remote_layers, "capacity": capacity,
        "prompt_len": prompt_len, "started_utc": r.utc_now(), "requests": [],
    }
    r.append_jsonl(output_dir / "cases.jsonl", case_record)
    warnings, projected = ramp_guard(args, qx or "q4_0", capacity,
                                     n_remote_estimate, case_id)
    case_record["incremental_remote_kv_estimate_mib"] = projected
    case_record["estimate_label"] = ("incremental remote KV capacity only; resident model "
                                     "weights, CPU caches and OS are NOT included")
    case_record["ramp_warnings"] = warnings
    if warnings:
        case_record["state"] = "skipped_ramp"
        case_record["error"] = "; ".join(warnings)
        r.append_jsonl(output_dir / "cases.jsonl", case_record)
        return case_record, None
    instance = r.ServerInstance(
        argparse.Namespace(**{**vars(args), "ctx_size": capacity}),
        output_dir, case_id, case_extra_args(args, qx, remote_layers))
    case_record["command_argv"] = instance.command
    case_record["command_shell"] = shlex.join(instance.command)
    r.append_jsonl(output_dir / "cases.jsonl", case_record)
    print(f"[{case_id}] starting: {shlex.join(instance.command)}", flush=True)
    failures: list[str] = []
    quality: dict[str, Any] | None = None
    sampler = MemorySampler(args, args.sample_memory_interval, None)
    try:
        # Startup allocation snapshot BEFORE the server starts; the sampled
        # peak below covers requests only and is labeled as such.
        case_record["startup_memory_snapshot"] = mem_snapshot_once(args)
        instance.start()
        sampler = MemorySampler(args, args.sample_memory_interval,
                                instance.proc.pid if instance.proc else None)
        sampler.start()
        case_record["state"] = "ready"
        case_record["ready_utc"] = r.utc_now()
        r.append_jsonl(output_dir / "cases.jsonl", case_record)
        frozen = r.create_frozen_prompts(base_url, args, records_path, output_dir,
                                         case_id, frozen)
        run_meta["frozen_prompts"] = {"file": "prompts.json",
                                      "prefix_tokens": len(frozen["prefix_token_ids"]),
                                      "suffix_tokens": len(frozen["suffix_token_ids"])}
        r.write_json(output_dir / "run.json", run_meta)

        prefix = frozen["prefix_token_ids"][:prompt_len]
        extension = prefix + frozen["suffix_token_ids"]
        # Never silently truncate: the exact lengths must exist before HTTP.
        if len(prefix) != prompt_len:
            raise RuntimeError(f"{case_id}: frozen corpus has only {len(prefix)} ids for "
                               f"prompt_len={prompt_len}; raise --prompt-tokens")
        if len(extension) != prompt_len + args.suffix_tokens:
            raise RuntimeError(f"{case_id}: extension length {len(extension)} != "
                               f"{prompt_len + args.suffix_tokens}")
        base_hash = r.sha256_bytes(r.json_bytes(prefix))
        ext_hash = r.sha256_bytes(r.json_bytes(extension))
        deadline = time.monotonic() + args.time_budget_seconds

        def completion(ids: list[int], ids_hash: str, phase: str,
                       cold: bool, reuse_required: bool) -> tuple[dict[str, Any], list[str]]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"{case_id}: time budget {args.time_budget_seconds:g}s exceeded")
            runtime_uma_guard(args)
            # Cap the actual request timeout to the remaining budget.
            timeout = min(args.request_timeout, max(0.1, remaining))
            payload = {
                "prompt": ids, "n_predict": args.decode_tokens, "id_slot": 0,
                "cache_prompt": True, "return_tokens": True,
                "temperature": 0, "seed": args.seed, "ignore_eos": True,
                "n_probs": args.n_probs,
            }
            result = r.http_request(base_url, "POST", "/completion", payload,
                                    timeout, records_path, args.telemetry_device,
                                    case_id, phase,
                                    request_body_record={**payload, "prompt_sha256": ids_hash})
            if time.monotonic() > deadline:
                raise RuntimeError(f"{case_id}: time budget exceeded after {phase}")
            phase_failures: list[str] = []
            failed, detail = r.response_is_error(result)
            if failed:
                return result, [f"{phase}: request failed: {detail}"]
            timings = result["json"].get("timings") if isinstance(result["json"], dict) else {}
            leg_failures, _ = m.validate_leg(timings, result["json"].get("truncated"),
                                             len(ids), args.decode_tokens, phase,
                                             cold=cold, reuse_required=reuse_required)
            phase_failures += leg_failures
            if args.spec_type == "draft-mtp":
                phase_failures += m.validate_response_draft(timings, phase)
            return result, phase_failures

        r.http_request(base_url, "POST", "/slots/0?action=erase", None,
                       args.request_timeout, records_path, args.telemetry_device,
                       case_id, f"{case_id}-erase-before")
        cold, cold_failures = completion(prefix, base_hash, f"{case_id}-cold",
                                         cold=True, reuse_required=False)
        failures += cold_failures
        cold_delta = instance.read_log_delta()
        cold_timings = cold["json"].get("timings") or {}

        ext1, ext1_failures = completion(extension, ext_hash, f"{case_id}-suffix",
                                         cold=False, reuse_required=prompt_len >= 8192)
        failures += ext1_failures
        ext1_delta = instance.read_log_delta()
        ext1_timings = ext1["json"].get("timings") or {}

        ext2, ext2_failures = completion(extension, ext_hash, f"{case_id}-repeat",
                                         cold=False, reuse_required=True)
        failures += ext2_failures
        ext2_delta = instance.read_log_delta()
        ext2_timings = ext2["json"].get("timings") or {}

        occupied = (ext2_timings.get("cache_n", 0) + ext2_timings.get("prompt_n", 0) +
                    ext2_timings.get("predicted_n", 0))
        row: dict[str, Any] = {
            "case_id": case_id, "remote_kv_type": qx, "capacity": capacity,
            "prompt_tokens": prompt_len, "occupied_tokens": occupied,
            "occupancy_fraction": occupied / capacity if capacity else None,
            "cold_timings": cold_timings, "ext1_timings": ext1_timings,
            "ext2_timings": ext2_timings,
            "cold_probs": cold["json"].get("completion_probabilities"),
            "ext1_probs": ext1["json"].get("completion_probabilities"),
            "ext2_probs": ext2["json"].get("completion_probabilities"),
            "draft_log": m.reconcile_log_draft(ext2_delta, ext2_timings.get("draft_n")),
            "graphs_reused": m.extract_graphs_reused(ext2_delta),
            "memory_snapshots_label": ("telemetry before/after each request; NOT peak "
                                       "(enable --sample-memory-interval for sampled peak)"),
            "validation_failures": cold_failures + ext1_failures + ext2_failures,
        }
        if baseline_quality is not None:
            base_timings = baseline_quality["ext2_timings"]
            if cadence_equal(base_timings, ext2_timings):
                compare = first_distribution_compare(baseline_quality["ext2_probs"],
                                                     row["ext2_probs"])
                if compare is None:
                    failures.append(f"{case_id}: baseline and mixed first-predicted top-k "
                                    "distributions unavailable for comparison")
                row["quality"] = compare
            else:
                row["quality_skipped"] = (
                    f"cadence differs from baseline ({base_timings.get('cache_n')} vs "
                    f"{ext2_timings.get('cache_n')}); only identical-cadence legs are "
                    f"comparable; not reported")
        r.append_jsonl(results_path, row)

        case_record["requests"] = [
            {"phase": f"{case_id}-cold", "cache_n": cold_timings.get("cache_n"),
             "prompt_n": cold_timings.get("prompt_n"),
             "predicted_n": cold_timings.get("predicted_n"),
             "draft_n": cold_timings.get("draft_n")},
            {"phase": f"{case_id}-suffix", "cache_n": ext1_timings.get("cache_n"),
             "prompt_n": ext1_timings.get("prompt_n"),
             "predicted_n": ext1_timings.get("predicted_n"),
             "draft_n": ext1_timings.get("draft_n")},
            {"phase": f"{case_id}-repeat", "cache_n": ext2_timings.get("cache_n"),
             "prompt_n": ext2_timings.get("prompt_n"),
             "predicted_n": ext2_timings.get("predicted_n"),
             "draft_n": ext2_timings.get("draft_n")},
        ]
        case_record["validation_failures"] = failures
        case_record["state"] = "complete" if not failures else "failed"
        if failures:
            case_record["error"] = "; ".join(failures)
        quality = {"ext2_timings": ext2_timings,
                   "ext2_probs": row["ext2_probs"]}
    except Exception as exc:
        case_record["state"] = "failed"
        case_record["error"] = f"{type(exc).__name__}: {exc}"
        # On a prefill timeout, report the actual processed count/time from
        # the raw server log; never invent TPS/decode numbers.
        if any(word in case_record["error"].lower() for word in ("time budget", "timeout", "timed out")):
            case_record["partial_prefill"] = partial_prefill_from_log(
                instance.log_path.read_text(encoding="utf-8", errors="replace")) \
                if instance.log_path.is_file() else None
        print(f"[{case_id}] failed: {case_record['error']}", flush=True)
    finally:
        sampler.stop()
        case_record["memory_sampler"] = sampler.summary()
        case_record["server_exit_code"] = instance.stop()
        full_log = instance.log_path.read_text(encoding="utf-8", errors="replace")
        if any(word in case_record.get("error", "").lower() for word in ("time budget", "timeout", "timed out")):
            case_record["partial_prefill"] = partial_prefill_from_log(full_log)
        if qx is not None:
            layout = r.parse_mixed_layout(full_log)
            if remote_layers.isdigit():
                layout_failures = r.validate_mixed_layout(layout, int(remote_layers), qx)
            elif layout:
                layout_failures = r.validate_mixed_layout(layout, len(layout), qx)
            else:
                # auto placement with zero remote layers is a VALID planner
                # outcome; record the actual count instead of failing.
                layout_failures = []
            case_record["auto_placed_remote_layers"] = len(layout)
            case_record["kv_tail_tokens"] = 0
            case_record["mtp_disabled"] = args.spec_type == "none"
            if layout_failures:
                case_record["state"] = "failed"
                case_record["error"] = (case_record.get("error", "") +
                                        "; " + "; ".join(layout_failures)).strip("; ")
            case_record["mixed_format_log_evidence"] = layout
            mtp_failures = speculation_placement_failures(args.spec_type, full_log)
            case_record["mtp_placement_failures"] = mtp_failures
            if mtp_failures:
                case_record["state"] = "failed"
                case_record["error"] = (case_record.get("error", "") +
                                        "; MTP placement: " + "; ".join(mtp_failures)).strip("; ")
        power_failures = m.power_evidence_failures(args, full_log)
        case_record["power_evidence_failures"] = power_failures
        if power_failures:
            case_record["state"] = "failed"
            case_record["error"] = (case_record.get("error", "") +
                                    "; " + "; ".join(power_failures)).strip("; ")
        case_record["finished_utc"] = r.utc_now()
        r.append_jsonl(output_dir / "cases.jsonl", case_record)
        print(f"[{case_id}] state={case_record['state']}", flush=True)
    return case_record, quality


# ---------------------------------------------------------------------------
# Stage H: actual handoff acceptance (static S1-S5 never claims conversion)
# ---------------------------------------------------------------------------

def stage_h(args: argparse.Namespace, output_dir: Path, run_meta: dict[str, Any],
            frozen: dict[str, Any] | None) -> dict[str, Any]:
    """Adaptive local->mixed handoff with the 27B model: MTP profile 512 ->
    LONG 1024 (--ctx-size 1024 --ctx-size-mtp 512), gate 513, prompt 300 ->
    700. cache_n > 0 and <= 300 at the gate is mandatory; the configured
    handoff evidence marker must appear in the gate log, otherwise the stage
    stays pending_evidence (exit 1) and never claims conversion."""
    base_url = f"http://{args.host}:{args.port}"
    records_path = output_dir / "requests.jsonl"
    results_path = output_dir / "results.jsonl"
    qx = args.remote_kv_types[0]
    case_id = "H-handoff"
    h_args = argparse.Namespace(**{**vars(args), "ctx_size": 1024})
    instance = r.ServerInstance(h_args, output_dir, case_id, stage_h_extra_args(args))
    case_record: dict[str, Any] = {
        "case_id": case_id, "scenario": "H", "state": "starting",
        "command_argv": instance.command, "command_shell": shlex.join(instance.command),
        "server_log": instance.log_path.name,
        "slot_save_path": instance.slot_save_path.name,
        "handoff_cache_ram_mib": args.handoff_cache_ram_mib,
        "handoff_evidence_markers": args.handoff_evidence_markers,
        "started_utc": r.utc_now(), "requests": [],
    }
    r.append_jsonl(output_dir / "cases.jsonl", case_record)
    print(f"[{case_id}] starting: {shlex.join(instance.command)}", flush=True)
    failures: list[str] = []
    sampler = MemorySampler(args, args.sample_memory_interval, None)
    try:
        instance.start()
        sampler = MemorySampler(args, args.sample_memory_interval,
                                instance.proc.pid if instance.proc else None)
        sampler.start()
        case_record["state"] = "ready"
        case_record["ready_utc"] = r.utc_now()
        r.append_jsonl(output_dir / "cases.jsonl", case_record)
        frozen = r.create_frozen_prompts(base_url, args, records_path, output_dir,
                                         case_id, frozen)
        r.write_json(output_dir / "run.json", run_meta)
        short_ids = frozen["prefix_token_ids"][:300]
        long_ids = frozen["prefix_token_ids"][:700]
        short_hash = r.sha256_bytes(r.json_bytes(short_ids))
        long_hash = r.sha256_bytes(r.json_bytes(long_ids))
        deadline = time.monotonic() + args.time_budget_seconds

        def completion(ids: list[int], ids_hash: str, phase: str) -> dict[str, Any]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"{case_id}: time budget {args.time_budget_seconds:g}s exceeded")
            timeout = min(args.request_timeout, max(0.1, remaining))
            payload = {
                "prompt": ids, "n_predict": args.decode_tokens, "id_slot": 0,
                "cache_prompt": True, "return_tokens": True,
                "temperature": 0, "seed": args.seed, "ignore_eos": True,
                "n_probs": args.n_probs,
            }
            result = r.http_request(base_url, "POST", "/completion", payload,
                                    timeout, records_path,
                                    args.telemetry_device, case_id, phase,
                                    request_body_record={**payload, "prompt_sha256": ids_hash})
            if time.monotonic() > deadline:
                raise RuntimeError(f"{case_id}: time budget exceeded after {phase}")
            return result

        short = completion(short_ids, short_hash, "H-short-cold")
        short_failures = m.validate_leg(short["json"].get("timings") or {},
                                        short["json"].get("truncated"),
                                        300, args.decode_tokens, "H-short-cold",
                                        cold=True, reuse_required=False)[0]
        short_failures += m.validate_response_draft(short["json"].get("timings") or {},
                                                    "H-short-cold")
        failures += short_failures
        r.append_jsonl(results_path, {
            "scenario": "H", "case_id": case_id, "phase": "H-short-cold",
            "timings": short["json"].get("timings"),
            "validation_failures": short_failures,
        })
        pre_gate_log = instance.read_log_delta()

        handoff = completion(long_ids, long_hash, "H-handoff")
        handoff_failures = m.validate_leg(handoff["json"].get("timings") or {},
                                          handoff["json"].get("truncated"),
                                          700, args.decode_tokens, "H-handoff",
                                          cold=False, reuse_required=True)[0]
        handoff_failures += r.validate_handoff_counts(
            handoff["json"], 300, 700, expected_decode=args.decode_tokens)
        failures += handoff_failures
        r.append_jsonl(results_path, {
            "scenario": "H", "case_id": case_id, "phase": "H-handoff",
            "timings": handoff["json"].get("timings"),
            "validation_failures": handoff_failures,
        })
        gate_log = instance.read_log_delta()

        long2 = completion(long_ids, long_hash, "H-long-continuation")
        long2_failures = m.validate_leg(long2["json"].get("timings") or {},
                                        long2["json"].get("truncated"),
                                        700, args.decode_tokens, "H-long-continuation",
                                        cold=False, reuse_required=True)[0]
        long2_failures += m.validate_response_draft(long2["json"].get("timings") or {},
                                                    "H-long-continuation")
        failures += long2_failures
        r.append_jsonl(results_path, {
            "scenario": "H", "case_id": case_id, "phase": "H-long-continuation",
            "timings": long2["json"].get("timings"),
            "validation_failures": long2_failures,
        })

        pre_gate_evidence = instance.log_evidence(pre_gate_log)
        gate_evidence = instance.log_evidence(gate_log)
        gate_layout = r.parse_mixed_layout(gate_log)
        if not any("adaptive remote attention:" in line.lower() and "backend=off" in line.lower()
                   for line in pre_gate_evidence):
            failures.append("pre-gate log lacks 'adaptive remote attention: ... backend=off'")
        if any("mixed kv layer=" in line.lower() for line in pre_gate_evidence):
            failures.append("pre-gate log contains a Qx 'mixed KV layer' allocation")
        if not any("adaptive remote attention:" in line.lower() and "backend=off" not in line.lower()
                   for line in gate_evidence):
            failures.append("gate log lacks an enabled 'adaptive remote attention' line")
        if not any("adaptive context transition complete" in line.lower()
                   for line in gate_evidence):
            failures.append("gate log lacks 'adaptive context transition complete'")
        failures += r.validate_mixed_layout(gate_layout, 1, qx)

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
        case_record["gate_mixed_format_log_evidence"] = gate_layout
        case_record["validation_failures"] = failures
        case_record["state"] = r.acceptance_state(failures, handoff_evidence)
        if case_record["state"] == "failed":
            case_record["error"] = "; ".join(failures)
        elif case_record["state"] == "pending_evidence":
            case_record["error"] = ("acceptance pending: handoff conversion evidence markers "
                                    "were not configured (--handoff-evidence-markers); "
                                    "static S1-S5 never claim conversion")
    except Exception as exc:
        case_record["state"] = "failed"
        case_record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[{case_id}] failed: {case_record['error']}", flush=True)
    finally:
        sampler.stop()
        case_record["memory_sampler"] = sampler.summary()
        case_record["server_exit_code"] = instance.stop()
        if case_record.get("server_exit_code") not in (None, 0) and \
                case_record["state"] != "pending_evidence":
            case_record["state"] = "failed"
            case_record["error"] = (case_record.get("error", "") +
                                    f" (server exit={case_record['server_exit_code']})")
        power_failures = m.power_evidence_failures(args, instance.log_path.read_text(
            encoding="utf-8", errors="replace"))
        case_record["power_evidence_failures"] = power_failures
        if power_failures:
            case_record["state"] = "failed"
            case_record["error"] = (case_record.get("error", "") +
                                    "; " + "; ".join(power_failures)).strip("; ")
        case_record["finished_utc"] = r.utc_now()
        r.append_jsonl(output_dir / "cases.jsonl", case_record)
        print(f"[{case_id}] state={case_record['state']}", flush=True)
    return {"scenario": "H", "runs": [case_record]}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_check:
        return self_check(args)
    try:
        args.stages = r.parse_names(args.stages, "--stages",
                                    {"S1", "S2", "S3", "S4", "S5", "H"})
        args.remote_kv_types = r.parse_names(args.remote_kv_types, "--remote-kv-types",
                                             set(REMOTE_TYPES))
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be in 1..65535")
    if not 0.0 < args.memory_cap_fraction <= 1.0:
        parser.error("--memory-cap-fraction must be in (0, 1]")
    if args.time_budget_seconds <= 0:
        parser.error("--time-budget-seconds must be positive")
    if "S5" in args.stages and not args.allow_capacity_200k:
        parser.error("S5 (200K-capacity probe) requires --allow-capacity-200k; "
                     "it is an allocation-size probe, not full occupancy")
    args.long_prompt_lengths = parse_ints(args.long_prompt_lengths, "--long-prompt-lengths")
    if args.prompt_tokens < required_corpus_tokens(args):
        parser.error(f"--prompt-tokens {args.prompt_tokens} must cover the largest chosen "
                     f"occupied prompt length + 1 ({required_corpus_tokens(args)}); the "
                     f"literal value never truncates a prefix silently")

    script_path = Path(__file__).resolve()
    repo = script_path.parents[1]
    binary = args.server_binary if args.server_binary.is_absolute() else repo / args.server_binary
    binary = binary.resolve()
    model = args.model.resolve()
    if not binary.is_file():
        parser.error(f"server binary not found: {binary} (expected build-optimized/bin)")
    if not model.is_file():
        parser.error(f"model not found: {model}")
    args.server_binary = binary
    args.model = model
    if args.telemetry_device == "auto":
        args.telemetry_device = discover_amdgpu_drm()
        print(f"DRM telemetry device (vendor discovery): {args.telemetry_device}", flush=True)

    output_dir = args.output_dir or Path(
        "bench-results") / f"mixed-kv-27b-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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

    if args.mock_endpoint:
        # Test-only: a fake HTTP endpoint owns the port; no server is spawned.
        # Synthetic logs from the recording stub must never appear in a real
        # run, so mock mode is refused unless both files are /bin/true
        # (compared after resolve()).
        if binary != Path("/bin/true").resolve() or model != Path("/bin/true").resolve():
            parser.error("--mock-endpoint is test-only: pass --server-binary /bin/true "
                         "and --model /bin/true; synthetic logs are never used in "
                         "actual runs")
        fake = _FakeEndpoint(args.host, args.port)
        fake.start()
        original_instance = r.ServerInstance
        r.ServerInstance = _RecordingServer
    else:
        fake = None
        original_instance = None
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
        "telemetry_device": args.telemetry_device,
        "estimates": {
            "label": ("all numbers are ESTIMATES of INCREMENTAL KV capacity; resident "
                      "model weights, CPU caches and OS usage are NOT included; "
                      "snapshots are before/after, not peak (unless the sampler ran)"),
            "remote_qx_mib_per_layer_per_16k": {qx: round(
                REMOTE_BYTES_PER_TOKEN[qx] * 16384 / 1048576.0, 2) for qx in REMOTE_TYPES},
            "kvarn4_local_mib_per_layer_per_16k": round(
                KVARN_RECORDS_PER_TOKEN * 16384 / 1048576.0 + KVARN_FIXED_PER_LAYER_MIB, 2),
            "observed_2026_10_01": {
                "radeon_gtt_budget_mib": 8091, "vulkan_workspace_mib": 126.27,
                "gtt_after_q8_all16_200k_capacity_bytes": 7146627072,
                "system_mem_available_kib_after": 6036404,
                "cuda_4070_used_mib_after": 9689,
                "per_16_layers_200k_capacity_mib": {"q4_0": 3600, "q5_0": 4400,
                                                    "q6_0": 5200, "q8_0": 6800},
            },
        },
        "artifacts": ["run.json", "cases.jsonl", "results.jsonl", "requests.jsonl",
                      "prompts.json", "prompt-corpus.txt", "continuation-suffix.txt",
                      "server-<case>.log"],
    }
    r.write_json(output_dir / "run.json", manifest)
    print(f"Output: {output_dir}\nStages: {','.join(args.stages)}; "
          f"remote KV: {','.join(args.remote_kv_types)}", flush=True)

    try:
        frozen: dict[str, Any] | None = None
        summaries: list[dict[str, Any]] = []
        baseline_by_length: dict[int, dict[str, Any]] = {}
        for stage in args.stages:
            if not args.mock_endpoint:
                port_free()
            if stage == "S1":
                for length in parse_ints(args.prompt_lengths, "--prompt-lengths"):
                    case_record, quality = run_case(
                        args, output_dir, manifest, f"S1-baseline-{length}",
                        None, "0", length + 512, length, None, frozen)
                    summaries.append(case_record)
                    if case_record["state"] == "complete" and quality is not None:
                        baseline_by_length[length] = quality
                    prompts_path = output_dir / "prompts.json"
                    if prompts_path.is_file() and frozen is None:
                        frozen = json.loads(prompts_path.read_text(encoding="utf-8"))
            elif stage == "S2" or stage == "S3":
                length = 8192 if stage == "S2" else 16384
                capacity = args.capacity if stage == "S2" else max(args.capacity, 32768)
                for qx in args.remote_kv_types:
                    case_record, quality = run_case(
                        args, output_dir, manifest, f"{stage}-{qx}-{length}",
                        qx, args.remote_attn_layers, capacity, length,
                        baseline_by_length.get(length), frozen)
                    summaries.append(case_record)
                    prompts_path = output_dir / "prompts.json"
                    if prompts_path.is_file() and frozen is None:
                        frozen = json.loads(prompts_path.read_text(encoding="utf-8"))
            elif stage == "S4":
                # Long occupied-context validation: auto minimal-required N,
                # no MTP in long profiles (comparable to production XXL with
                # draft n_max 0), no extra tail, power 20W/2700. Capacities
                # are 256-aligned with reserve margin (see long_capacity).
                # No quality claims are made from static long cases.
                s4_args = argparse.Namespace(**{**vars(args), "spec_type": "none"})
                for length in args.long_prompt_lengths:
                    capacity = long_capacity(length, args.suffix_tokens,
                                             args.decode_tokens)
                    for qx in args.remote_kv_types:
                        case_record, quality = run_case(
                            s4_args, output_dir, manifest, f"S4-{qx}-{length}",
                            qx, "auto", capacity, length, None, frozen)
                        case_record["mtp_disabled"] = True
                        case_record["kv_tail_tokens"] = 0
                        case_record["quality_claims"] = "none for static long cases"
                        summaries.append(case_record)
            elif stage == "S5":
                for qx in args.remote_kv_types:
                    case_record, quality = run_case(
                        args, output_dir, manifest, f"S5-{qx}-cap200k",
                        qx, "auto", 204800, args.s5_occupied, None, frozen)
                    summaries.append(case_record)
            elif stage == "H":
                h_result = stage_h(args, output_dir, manifest, frozen)
                summaries.append(h_result["runs"][0])
    finally:
        if fake is not None:
            fake.stop()
        if original_instance is not None:
            r.ServerInstance = original_instance

    runs = [run for run in summaries]
    failed = sum(1 for run in runs if run["state"] == "failed")
    skipped = sum(1 for run in runs if run["state"] == "skipped_ramp")
    pending = sum(1 for run in runs if run["state"] == "pending_evidence")
    r.write_json(output_dir / "summary.json", {
        "finished_utc": r.utc_now(), "stage_count": len(args.stages),
        "failed_runs": failed, "skipped_ramp_runs": skipped,
        "pending_evidence_runs": pending,
        "output_dir": str(output_dir),
    })
    print(f"Finished: {len(args.stages)} stages, {failed} failed, "
          f"{skipped} skipped by ramp guard, {pending} pending evidence; "
          f"results={output_dir}", flush=True)
    return 1 if (failed or pending) else 0


# ---------------------------------------------------------------------------
# Test-only fake endpoint + recording server stub (no GPU)
# ---------------------------------------------------------------------------

class _RecordingServer(r.ServerInstance):
    started: list = []

    def start(self) -> None:
        _RecordingServer.started.append(self.command)
        qx = "q4_0"
        for index, item in enumerate(self.command):
            if item == "--remote-attn-cache-type-k" and index + 1 < len(self.command):
                qx = self.command[index + 1]
        self.log_path.write_text(
            f"MTP K/V placement: local_attention=off, layers=0, prefill=remote, "
            f"cuda_reserve=650.00 MiB\n"
            f"mixed KV layer=0 device=Vulkan0 type_k={qx} type_v={qx} "
            f"domain=standard-hadamard\n"
            f"APU TDP: active limit 20 W\n"
            f"AMD graphics SCLK -> prefill, locked 2700 MHz\n",
            encoding="utf-8")


class _FakeEndpoint:
    """Canned responses exercising the real CLI/request/validation logic:
    cold -> cache_n=0; extension -> cache_n=len-32, prompt_n=32; drafts 8/4;
    first-predicted top-k distribution with three text-keyed entries."""

    def __init__(self, host: str, port: int):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import urllib.parse
        self.urllib_parse = urllib.parse
        self.threading_http = ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                if self.path == "/health":
                    self._json(200, {"status": "ok"})
                else:
                    self._json(404, {"error": {"code": 404, "message": "nf"}})

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/tokenize":
                    if body.get("add_special") is False:
                        self._json(200, {"tokens": list(range(500000, 500080))})
                    else:
                        self._json(200, {"tokens": list(range(210000))})
                elif parsed.path == "/completion":
                    ids = body.get("prompt", [])
                    # Extensions carry the independent suffix ids (>= 500000,
                    # outside any corpus range).
                    extension = any(isinstance(token, int) and token >= 500000
                                    for token in ids)
                    if extension:
                        cache_n = len(ids) - 32
                        prompt_n = 32
                    else:
                        cache_n = 0
                        prompt_n = len(ids)
                    timings = {
                        "cache_n": cache_n, "cache_lcp_n": len(ids), "cache_planned_n": cache_n,
                        "cache_reprocessed_n": prompt_n, "cache_source": "live",
                        "cache_reason": "committed" if extension else "none",
                        "prompt_n": prompt_n, "predicted_n": 16,
                        "draft_n": 8, "draft_n_accepted": 4,
                    }
                    probs = [{"id": 561, "token": " The", "bytes": [32, 84, 104, 101],
                              "logprob": -0.117,
                              "top_logprobs": [
                                  {"id": 561, "token": " The", "bytes": [32, 84, 104, 101],
                                   "logprob": -0.117},
                                  {"id": 198, "token": "\n", "bytes": [10],
                                   "logprob": -2.74},
                                  {"id": 271, "token": "\n\n", "bytes": [10, 10],
                                   "logprob": -3.98}]}]
                    self._json(200, {"content": "mock", "tokens": [1, 2],
                                     "timings": timings, "truncated": False,
                                     "completion_probabilities": probs})
                elif parsed.path.startswith("/slots/"):
                    self._json(200, {"n_erased": 0})
                else:
                    self._json(404, {"error": {"code": 404, "message": "nf"}})

        self.httpd = self.threading_http((host, port), Handler)

    def start(self) -> None:
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------------------------------------------------------------------------
# self-check (no server/GPU)
# ---------------------------------------------------------------------------

def self_check(args: argparse.Namespace) -> int:
    failures: list[str] = []
    checks: list[tuple[str, bool]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok))
        if not ok:
            failures.append(f"{name}: {detail}")

    check("project_q4_16k_1layer",
          abs(project_remote_mib("q4_0", 16384, 1) - 18.0) < 0.1,
          f"got {project_remote_mib('q4_0', 16384, 1):.2f}")
    check("project_q8_200k_16layer",
          abs(project_remote_mib("q8_0", 204800, 16) - 6800.0) < 5.0,
          f"got {project_remote_mib('q8_0', 204800, 16):.2f}")
    check("ramp_guard_blocks_huge",
          bool(ramp_guard(argparse.Namespace(memory_cap_fraction=0.01),
                          "q4_0", 204800, 16, "t")[0]))
    check("ramp_guard_allows_small",
          not ramp_guard(argparse.Namespace(memory_cap_fraction=0.99),
                         "q4_0", 16384, 1, "t")[0])
    check("ramp_guard_baseline_never_gated",
          ramp_guard(argparse.Namespace(memory_cap_fraction=0.001),
                     "q4_0", 204800, 0, "t") == ([], 0.0))
    check("ramp_guard_s5_probe_not_bypassed",
          bool(ramp_guard(argparse.Namespace(memory_cap_fraction=0.01),
                          "q8_0", 204800, 16, "t")[0]),
          "the opt-in probe does not disable the memory safety guard")

    check("cadence_equal_true",
          cadence_equal({"cache_n": 8192, "prompt_n": 32, "predicted_n": 16},
                        {"cache_n": 8192, "prompt_n": 32, "predicted_n": 16}))
    check("cadence_divergent_false",
          not cadence_equal({"cache_n": 8192, "prompt_n": 32, "predicted_n": 16},
                            {"cache_n": 7936, "prompt_n": 288, "predicted_n": 16}))
    check("cadence_missing_fields_never_equal",
          not cadence_equal({}, {}) and not cadence_equal({}, {"cache_n": 1}))

    # REAL server schema fixture (from stage5-B-q4_0-r4/run/results.jsonl):
    # {id, token, bytes, logprob, top_logprobs:[{id, token, bytes, logprob}]}.
    base_probs = [{"id": 561, "token": " The", "bytes": [32, 84, 104, 101],
                   "logprob": -0.117,
                   "top_logprobs": [
                       {"id": 561, "token": " The", "bytes": [32, 84, 104, 101],
                        "logprob": -0.117},
                       {"id": 198, "token": "\n", "bytes": [10], "logprob": -2.74},
                       {"id": 271, "token": "\n\n", "bytes": [10, 10], "logprob": -3.98}]}]
    shifted_probs = [{"id": 561, "token": " The", "bytes": [32, 84, 104, 101],
                      "logprob": -0.200,
                      "top_logprobs": [
                          {"id": 271, "token": "\n\n", "bytes": [10, 10],
                           "logprob": -1.50},
                          {"id": 561, "token": " The", "bytes": [32, 84, 104, 101],
                           "logprob": -2.10},
                          {"id": 9, "token": "!", "bytes": [33], "logprob": -3.30}]}]
    cmp_self = first_distribution_compare(base_probs, base_probs)
    check("compare_identical_overlap_full", cmp_self is not None and
          cmp_self["overlap_token_ids"] == 3 and cmp_self["union_token_ids"] == 3)
    # Top-k logprobs do not sum to 1 (mass remains outside the top-k); a
    # self-comparison retains exactly the baseline's own top-k total mass.
    expected_mass = math.exp(-0.117) + math.exp(-2.74) + math.exp(-3.98)
    check("compare_identical_retained_mass", cmp_self is not None and
          abs(cmp_self["retained_baseline_mass"] - expected_mass) < 1e-9)
    cmp_shifted = first_distribution_compare(base_probs, shifted_probs)
    check("compare_shifted_retained_mass_below_one", cmp_shifted is not None and
          cmp_shifted["retained_baseline_mass"] < 1.0 and
          cmp_shifted["mean_abs_prob_delta"] > 0.0)
    check("compare_ids_int_keyed", cmp_shifted is not None and
          "overlap_token_ids" in cmp_shifted and "union_token_ids" in cmp_shifted)
    check("compare_missing_row_none",
          first_distribution_compare([], base_probs) is None and
          first_distribution_compare(None, base_probs) is None)
    check("compare_nonfinite_logprob_skipped",
          first_distribution_compare(
              [{"id": 1, "top_logprobs": [{"id": 1, "logprob": float("nan")}]}],
              base_probs) is None)

    base_argv = " ".join(case_extra_args(
        argparse.Namespace(spec_type="draft-mtp", remote_attn_layers="1"), None, "0"))
    check("baseline_argv_no_remote", "--remote-attn" not in base_argv and
          "--spec-draft-type-k kvarn4" in base_argv, base_argv)
    mixed_argv = " ".join(case_extra_args(
        argparse.Namespace(spec_type="draft-mtp", remote_attn_layers="1"), "q8_0", "1"))
    check("mixed_argv_remote", "--remote-attn-cache-type-k q8_0" in mixed_argv and
          "--remote-attn-layers 1" in mixed_argv, mixed_argv)
    h_argv = " ".join(stage_h_argv_for_check())
    check("handoff_argv", "--ctx-size-mtp 512" in h_argv and
          "--remote-attn-min-context 513" in h_argv and
          "--cache-ram 2048" in h_argv, h_argv)

    # S4 capacity alignment (256-aligned, >= need, 512-token reserve margin;
    # 98K/200K snap to 102400/204800).
    check("long_capacity_32k",
          long_capacity(32768, 32, 16) >= 32768 + 32 + 16 + 512 and
          long_capacity(32768, 32, 16) % 256 == 0,
          str(long_capacity(32768, 32, 16)))
    check("long_capacity_64k",
          long_capacity(65536, 32, 16) % 256 == 0 and
          long_capacity(65536, 32, 16) >= 65536 + 32 + 16 + 512,
          str(long_capacity(65536, 32, 16)))
    check("long_capacity_98k_snaps_102400", long_capacity(98304, 32, 16) == 102400)
    check("long_capacity_200k_snaps_204800", long_capacity(200000, 32, 16) == 204800)
    check("long_capacity_small", long_capacity(32768, 32, 16) == 33536)

    # Corpus requirement: automatic coverage of the largest chosen length + 1.
    ns = argparse.Namespace(stages=["S1", "S4", "S5"], prompt_lengths="8192,16384",
                            long_prompt_lengths=[200000], s5_occupied=8192,
                            prompt_short=300, prompt_long=700)
    check("required_corpus_covers_s4", required_corpus_tokens(ns) == 200001)
    ns2 = argparse.Namespace(stages=["H"], prompt_lengths="8192",
                             long_prompt_lengths=[], s5_occupied=8192,
                             prompt_short=300, prompt_long=700)
    check("required_corpus_h", required_corpus_tokens(ns2) == 701)

    # Partial-prefill parsing from the raw log (timeout evidence, not invented).
    log_line = "slot 0: prompt eval time = 12345.67 ms / 24178 tokens (0.51 ms per token, 1959.12 tokens per second)\n"
    partial = partial_prefill_from_log(log_line)
    check("partial_prefill_parsed", partial is not None and
          partial["processed_tokens"] == 24178 and partial["outcome"] == "request_timeout",
          str(partial))
    check("partial_prefill_absent", partial_prefill_from_log("no eval line") is None)
    check("spec_none_needs_no_mtp_log", speculation_placement_failures("none", "") == [])
    check("spec_none_rejects_drafting", bool(speculation_placement_failures("none", "draft acceptance = 1")))
    check("spec_mtp_still_requires_placement", bool(speculation_placement_failures("draft-mtp", "")))

    progress_line = "prompt processing, n_tokens = 7424, progress = 0.91, t = 8.75 s / 848.83 tokens per second"
    progress = partial_prefill_from_log(log_line + progress_line)
    check("partial_prefill_latest_progress", progress is not None and
          progress["processed_tokens"] == 7424 and progress["ms"] == 8750)
    check("long_capacity_boundary_has_margin", long_capacity(102400, 32, 16) >= 102400 + 560)


    # Runtime UMA guard aborts below the floor.
    guarded = argparse.Namespace(memory_hard_floor_mib=10.0 ** 12)
    try:
        runtime_uma_guard(guarded)
        check("uma_guard_aborts", False, "guard did not raise")
    except RuntimeError:
        check("uma_guard_aborts", True)

    print(f"self-check: {len(checks)} checks, {len(failures)} failures")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    return 1 if failures else 0


def stage_h_argv_for_check() -> list[str]:
    args = argparse.Namespace(handoff_cache_ram_mib=2048, remote_kv_types=["q4_0"])
    return stage_h_extra_args(args)


def stage_h_extra_args(args: argparse.Namespace) -> list[str]:
    qx = args.remote_kv_types[0]
    return [
        "--spec-type", "draft-mtp",
        "--ctx-size-mtp", "512",
        "--mtp-max-tokens", "512",
        "--cache-type-k-m", "kvarn4", "--cache-type-v-m", "kvarn4",
        "--cache-type-k-l", "kvarn4", "--cache-type-v-l", "kvarn4",
        "--spec-draft-type-k-m", "kvarn4", "--spec-draft-type-v-m", "kvarn4",
        "--spec-draft-type-k-l", "kvarn4", "--spec-draft-type-v-l", "kvarn4",
        "--spec-draft-n-max", "2", "--spec-draft-n-max-l", "2",
        "--remote-attn", "vulkan:0", "--remote-attn-layers", "1",
        "--remote-attn-min-context", "513",
        "--remote-attn-cache-type-k", qx, "--remote-attn-cache-type-v", qx,
        "--cache-ram", str(args.handoff_cache_ram_mib), "--ctx-checkpoints", "1",
    ]


if __name__ == "__main__":
    sys.exit(main())