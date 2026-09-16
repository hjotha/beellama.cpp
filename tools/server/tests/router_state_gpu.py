#!/usr/bin/env python3
"""Exclusive GPU window against the promoted KVarN router release.

The wrapper validates the live release, model ID and router idle/queue state
before stopping anything. It clones the frozen production INI into the private artifact
directory and changes only the snapshot store path. The matrix always runs on
8091 and every exit path, including SIGINT/SIGTERM and harness failure, starts
the exact original unit and verifies health, release path and unit/drop-in
hashes. It must be run explicitly as root.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request


DEFAULT_RELEASE = Path("/home/hjotha/releases/beellama-router-kvarn-convert-20260917-r2")
DEFAULT_PRESET = DEFAULT_RELEASE / "qwen-3.8-27b-q4-mixed-20260916.ini"
DEFAULT_MODEL = "/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
PUBLIC_CONTEXT = 114688


def command(*argv):
    return subprocess.check_output(argv, text=True).strip()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_path(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def forwarded_value(values, option):
    for index, value in enumerate(values):
        if value == option and index + 1 < len(values):
            return values[index + 1]
    return None


def remove_option(values, option, count=1):
    result = []
    index = 0
    while index < len(values):
        if values[index] == option:
            index += 1 + count
        else:
            result.append(values[index])
            index += 1
    return result


def metric_values(text):
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$", line)
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


def process_tree(root_pid):
    try:
        rows = command("ps", "-eo", "pid=,ppid=").splitlines()
    except subprocess.CalledProcessError:
        return {root_pid}
    children = {}
    for row in rows:
        fields = row.split()
        if len(fields) != 2:
            continue
        try:
            pid, ppid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    result = {root_pid}
    queue = [root_pid]
    while queue:
        parent = queue.pop()
        for child in children.get(parent, []):
            if child not in result:
                result.add(child)
                queue.append(child)
    return result


def gpu_apps():
    return command("nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                   "--format=csv,noheader,nounits")


def app_pids(text):
    result = set()
    for line in text.splitlines():
        match = re.match(r"\s*(\d+)\s*,", line)
        if match:
            result.add(int(match.group(1)))
    return result


def start_gpu_sampler(root):
    output_path = root / "gpu-memory.csv"
    output = output_path.open("w", encoding="utf-8")
    process = subprocess.Popen([
        "nvidia-smi", "--query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits", "--loop-ms=500",
    ], stdout=output, stderr=subprocess.STDOUT, text=True)
    return process, output, output_path


def scan_matrix_logs(root):
    patterns = {
        "oom_or_cuda_alloc_failure": re.compile(
            r"(?i)(?:cuda|gpu).*(?:out of memory|allocation failed|failed to allocate|cuda error).*|"
            r"(?:out of memory|allocation failed|failed to allocate).*(?:cuda|gpu)"),
        "recovery": re.compile(r"(?i)\b(?:recovery|recovered)\b"),
        "cuda_graph_preventive": re.compile(r"(?i)cuda\s+graph.*(?:disable|prevent|headroom)|(?:disable|prevent).*cuda\s+graph"),
        "batch_or_slot_fallback": re.compile(r"(?i)failed to find a memory slot|retrying with smaller batch size"),
    }
    matches = {name: [] for name in patterns}
    files = []
    matrix = root / "matrix"
    if matrix.exists():
        files = sorted(matrix.rglob("*.log"))
    for path in files:
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            for name, pattern in patterns.items():
                if pattern.search(line):
                    matches[name].append({"path": str(path), "line": number, "text": line})
    result = {"files": [str(path) for path in files],
              "counts": {name: len(lines) for name, lines in matches.items()},
              "matches": matches,
              "oom_is_not_hidden": True}
    (root / "log-scan.json").write_text(json.dumps(result, indent=2))
    return result


def prepare_private_preset(root, source, model):
    matrix = root / "matrix"
    store = matrix / "auto-store"
    text = source.read_text()
    if not re.search(r"(?m)^\s*slot-save-path\s*=", text):
        raise RuntimeError("frozen production INI has no slot-save-path to isolate")
    text = re.sub(r"(?m)^\s*slot-save-path\s*=.*$",
                  f"slot-save-path = {store}/", text, count=1)
    if "slot-save-auto = true" not in text:
        raise RuntimeError("frozen production INI does not enable slot-save-auto")
    if f"model = {model}" not in text:
        raise RuntimeError("frozen production INI model does not match the requested public model")
    if "/home/hjotha/llama-slot-cache" in text:
        raise RuntimeError("private GPU candidate still references the production cache")
    # Keep the candidate beside (not inside) the harness work root: the GPU
    # harness requires its --work-dir to be newly created by its own main.
    candidate = root / ("candidate-" + source.name)
    candidate.write_text(text)
    return candidate, store


def clone_forwarded(args, candidate, model_id):
    values = list(args.harness_args)
    if values[:1] == ["--"]:
        values = values[1:]
    values = remove_option(values, "--")
    values = remove_option(values, "--port")
    values = remove_option(values, "--work-dir")
    values = remove_option(values, "--preset")
    values = remove_option(values, "--gpu", 0)
    values = remove_option(values, "--model-id")
    values = remove_option(values, "--contexts", 4)
    values = remove_option(values, "--output-tokens")
    if "--auto-cache" not in values:
        values.append("--auto-cache")
    values.extend(["--model-id", model_id, "--preset", str(candidate), "--output-tokens", "4096"])
    return values


def validate_baseline(service, release, model_id, record):
    state = command("systemctl", "is-active", service)
    if state != "active":
        raise RuntimeError(f"baseline service is not active: {state}")
    unit_text = command("systemctl", "cat", service)
    expected_binary = str(release / "bin" / "llama-server")
    if expected_binary not in unit_text or f"WorkingDirectory={release}" not in unit_text:
        raise RuntimeError("active unit is not the promoted production release")
    record["before_state"] = state
    record["before_unit_sha256"] = sha256_bytes(unit_text.encode())
    record["before_exec_release"] = expected_binary
    record["before_working_directory"] = str(release)
    dropin = Path(f"/etc/systemd/system/{service}.d/10-gpu-ready.conf")
    if not dropin.exists():
        raise RuntimeError(f"expected GPU drop-in is missing: {dropin}")
    record["dropin_path"] = str(dropin)
    record["before_dropin_sha256"] = sha256_path(dropin)
    record["before_main_pid"] = int(command("systemctl", "show", "--value", "-p", "MainPID", service))

    models = get_json("http://127.0.0.1:8090/v1/models")
    data = models.get("data", [])
    listed = models.get("models", [])
    if len(data) != 1 or len(listed) != 1 or data[0].get("id") != model_id or listed[0].get("name") != model_id:
        raise RuntimeError(f"production model identity is not the expected single public ID: {models}")
    if listed[0].get("context_window") != PUBLIC_CONTEXT:
        raise RuntimeError(f"production context window is not {PUBLIC_CONTEXT}: {listed[0]}")
    record["before_models"] = models

    query = urllib.parse.quote(model_id, safe="")
    metrics_url = f"http://127.0.0.1:8090/metrics?model={query}"
    with urllib.request.urlopen(metrics_url, timeout=10) as response:
        metrics_text = response.read().decode()
    metrics = metric_values(metrics_text)
    required = ("llamacpp:requests_processing", "llamacpp:requests_deferred")
    if any(key not in metrics for key in required):
        raise RuntimeError(f"router metrics did not expose processing/deferred queue: {metrics}")
    if metrics[required[0]] != 0 or metrics[required[1]] != 0:
        raise RuntimeError(f"production is busy or has deferred work: {metrics}")
    slots = get_json(f"http://127.0.0.1:8090/slots?model={query}")
    if not slots or any(slot.get("is_processing") for slot in slots):
        raise RuntimeError(f"production slots are not idle: {slots}")
    record["before_metrics"] = {key: metrics[key] for key in required}
    record["before_slots"] = slots

    apps = gpu_apps()
    service_pids = process_tree(record["before_main_pid"])
    external = sorted(app_pids(apps) - service_pids)
    if external:
        raise RuntimeError(f"GPU has compute processes outside {service}: {external}")
    if not app_pids(apps):
        raise RuntimeError("production service has no visible GPU compute child")
    record["before_gpu"] = apps
    record["before_service_pids"] = sorted(service_pids)


def validate_restored_baseline(service, release, model_id, record):
    state = command("systemctl", "is-active", service)
    unit_text = command("systemctl", "cat", service)
    expected_binary = str(release / "bin" / "llama-server")
    record["after_state"] = state
    record["after_unit_sha256"] = sha256_bytes(unit_text.encode())
    record["after_exec_release"] = expected_binary in unit_text
    record["after_working_directory"] = f"WorkingDirectory={release}" in unit_text
    record["unit_unchanged"] = record["before_unit_sha256"] == record["after_unit_sha256"]
    record["after_dropin_sha256"] = sha256_path(Path(record["dropin_path"]))
    record["dropin_unchanged"] = record["before_dropin_sha256"] == record["after_dropin_sha256"]
    query = urllib.parse.quote(model_id, safe="")
    health = get_json("http://127.0.0.1:8090/health", timeout=10)
    models = get_json("http://127.0.0.1:8090/v1/models", timeout=10)
    slots = get_json(f"http://127.0.0.1:8090/slots?model={query}", timeout=10)
    with urllib.request.urlopen(f"http://127.0.0.1:8090/metrics?model={query}", timeout=10) as response:
        metrics = metric_values(response.read().decode())
    record["after_health"] = health
    record["after_models"] = models
    record["after_slots"] = slots
    record["after_metrics"] = {key: metrics.get(key) for key in
                                ("llamacpp:requests_processing", "llamacpp:requests_deferred")}
    record["after_gpu"] = gpu_apps()
    after_service_pids = process_tree(int(command("systemctl", "show", "--value", "-p", "MainPID", service)))
    record["after_service_pids"] = sorted(after_service_pids)
    record["after_external_gpu_pids"] = sorted(app_pids(record["after_gpu"]) - after_service_pids)
    loaded = (len(models.get("data", [])) == 1 and
              models["data"][0].get("id") == model_id and
              models["data"][0].get("status", {}).get("value") == "loaded")
    listed = models.get("models", [])
    context_ok = len(listed) == 1 and listed[0].get("context_window") == PUBLIC_CONTEXT
    slots_ready = bool(slots) and not any(slot.get("is_processing") for slot in slots)
    if (state != "active" or not record["after_exec_release"] or
            not record["after_working_directory"] or not record["unit_unchanged"] or
            not record["dropin_unchanged"] or health.get("status") != "ok" or
            not loaded or not slots_ready or record["after_external_gpu_pids"] or
            not context_ok or
            metrics.get("llamacpp:requests_processing") != 0 or
            metrics.get("llamacpp:requests_deferred") != 0):
        raise RuntimeError("original production baseline did not restore exactly: child/model/slots/metrics validation failed")
    record["restored_exactly"] = True


def wait_for_restored_baseline(service, release, model_id, record, timeout=180):
    deadline = time.monotonic() + timeout
    attempts = 0
    last_error = None
    while time.monotonic() < deadline:
        attempts += 1
        try:
            validate_restored_baseline(service, release, model_id, record)
            record["restore_readiness_attempts"] = attempts
            return
        except BaseException as error:
            last_error = error
            time.sleep(1)
    record["restore_readiness_attempts"] = attempts
    raise RuntimeError(f"production child readiness did not complete within {timeout}s: {last_error!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--release", default=str(DEFAULT_RELEASE))
    parser.add_argument("--preset-source", default=str(DEFAULT_PRESET))
    parser.add_argument("--harness", default="router_state_unified_gpu.py",
                        help="harness script under tools/server/tests to run inside the window")
    parser.add_argument("--harness-raw", action="store_true",
                        help="pass harness args verbatim (only --preset is injected) instead of "
                             "cloning the unified-matrix defaults")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate baseline and prepare private INI without stopping production")
    parser.add_argument("--pause-after-stop", type=float, default=0.0,
                        help="test-only signal window after stop; production is restored in finally")
    parser.add_argument("harness_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise RuntimeError("root is required for service control and GPU governors")

    root = Path(args.artifacts).resolve()
    root.mkdir(parents=True, exist_ok=False)
    release = Path(args.release).resolve()
    source = Path(args.preset_source).resolve()
    if not release.is_dir() or not source.is_file():
        raise RuntimeError("production release or frozen preset is missing")
    model_id = forwarded_value(args.harness_args, "--model-id") or forwarded_value(args.harness_args, "--model") or DEFAULT_MODEL
    record = {
        "service": args.service,
        "release": str(release),
        "preset_source": str(source),
        "model_id": model_id,
        "started": time.time(),
        "dry_run": args.dry_run,
    }
    stopped = False
    gpu_sampler = None
    signal_state = {"pending": None, "stop_in_progress": False, "restore_in_progress": False}

    def handle_signal(signum, frame):
        del frame
        record["signal_received"] = signum
        if signal_state["stop_in_progress"] or signal_state["restore_in_progress"]:
            signal_state["pending"] = signum
            return
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        validate_baseline(args.service, release, model_id, record)
        candidate, private_store = prepare_private_preset(root, source, model_id)
        if args.harness_raw:
            forwarded = list(args.harness_args)
            if forwarded[:1] == ["--"]:
                forwarded = forwarded[1:]
            forwarded = remove_option(forwarded, "--")
            if "--preset" not in forwarded:
                forwarded.extend(["--preset", str(candidate)])
        else:
            forwarded = clone_forwarded(args, candidate, model_id)
        record["candidate_preset"] = str(candidate)
        record["candidate_preset_sha256"] = sha256_path(candidate)
        record["private_store"] = str(private_store)
        record["candidate_model_id"] = model_id
        record["command"] = [args.harness, "--work-dir", str(root / "matrix")] + forwarded
        (root / "control.json").write_text(json.dumps(record, indent=2))
        if args.dry_run:
            record["validated"] = True
            record["finished"] = time.time()
            (root / "control.json").write_text(json.dumps(record, indent=2))
            print(json.dumps({"dry_run": True, "baseline": "production", "candidate": str(candidate)}), flush=True)
            return

        print("GPU_WINDOW_BEGIN: production idle, stopping llama-server-root.service", flush=True)
        record["window_begin"] = time.time()
        stopped = True
        signal_state["stop_in_progress"] = True
        try:
            subprocess.run(["systemctl", "stop", args.service], check=True, timeout=120)
            record["stopped"] = time.time()
            for _ in range(30):
                record["exclusive_gpu"] = gpu_apps()
                if not record["exclusive_gpu"]:
                    break
                time.sleep(1)
            if record.get("exclusive_gpu"):
                raise RuntimeError(f"GPU remained busy after stopping baseline: {record['exclusive_gpu']}")
            record["stop_completed"] = True
            if signal_state["pending"] is not None:
                raise KeyboardInterrupt(f"signal {signal_state['pending']}")
        finally:
            signal_state["stop_in_progress"] = False
        if args.pause_after_stop > 0:
            record["pause_after_stop"] = args.pause_after_stop
            (root / "control.json").write_text(json.dumps(record, indent=2))
            time.sleep(args.pause_after_stop)
        gpu_sampler = start_gpu_sampler(root)
        record["gpu_sampler_path"] = str(gpu_sampler[2])
        (root / "control.json").write_text(json.dumps(record, indent=2))

        harness = Path(__file__).with_name(args.harness)
        sys.argv = [str(harness), "--work-dir", str(root / "matrix")] + forwarded
        runpy.run_path(str(harness), run_name="__main__")
        record["log_scan"] = scan_matrix_logs(root)
        if record["log_scan"]["counts"]["oom_or_cuda_alloc_failure"]:
            raise RuntimeError("GPU matrix logged a CUDA allocation/OOM failure")
        record["matrix_passed"] = True
    except BaseException as error:
        record["failure"] = repr(error)
        raise
    finally:
        if gpu_sampler is not None:
            sampler_process, sampler_output, _ = gpu_sampler
            sampler_process.terminate()
            try:
                sampler_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sampler_process.kill()
                sampler_process.wait()
            sampler_output.close()
            if stopped and "log_scan" not in record:
                try:
                    record["log_scan"] = scan_matrix_logs(root)
                except BaseException as error:
                    record["log_scan_failure"] = repr(error)
        if stopped:
            print("GPU_WINDOW_END: restoring original production service", flush=True)
            restore_error = None
            signal_state["restore_in_progress"] = True
            try:
                record["pre_restore_gpu"] = gpu_apps()
                if record["pre_restore_gpu"]:
                    record["harness_gpu_residual"] = record["pre_restore_gpu"]
            except BaseException as error:
                restore_error = error
                record["restore_failure"] = "pre-restore GPU cleanup check failed: " + repr(error)
            try:
                subprocess.run(["systemctl", "start", args.service], check=True, timeout=120)
                subprocess.run(["curl", "-fsS", "--max-time", "10", "--retry", "90",
                                "--retry-all-errors", "--retry-delay", "1",
                                "http://127.0.0.1:8090/health"], check=True, timeout=120,
                               stdout=subprocess.PIPE, text=True)
                wait_for_restored_baseline(args.service, release, model_id, record)
                if record.get("harness_gpu_residual"):
                    raise RuntimeError("harness GPU process remained before production restore")
            except BaseException as error:
                if restore_error is None:
                    restore_error = error
                record["restore_failure"] = repr(error)
            signal_state["restore_in_progress"] = False
            if restore_error is None and signal_state["pending"] is not None:
                restore_error = KeyboardInterrupt(f"signal {signal_state['pending']}")
                record["restore_failure"] = repr(restore_error)
            if restore_error is None and record.get("log_scan_failure"):
                restore_error = RuntimeError("GPU log scan failed; matrix evidence is incomplete")
                record["restore_failure"] = repr(restore_error)
            record["window_end"] = time.time()
            record["finished"] = record["window_end"]
            (root / "control.json").write_text(json.dumps(record, indent=2))
            if restore_error is None:
                print(json.dumps({"after_state": record.get("after_state"),
                                  "after_health": record.get("after_health"),
                                  "unit_unchanged": record.get("unit_unchanged"),
                                  "dropin_unchanged": record.get("dropin_unchanged"),
                                  "restored_exactly": record.get("restored_exactly")}), flush=True)
            else:
                raise restore_error
        else:
            record["finished"] = time.time()
            (root / "control.json").write_text(json.dumps(record, indent=2))

        uid = int(os.environ.get("SUDO_UID", "0"))
        gid = int(os.environ.get("SUDO_GID", "0"))
        for directory, _, files in os.walk(root):
            os.chown(directory, uid, gid)
            for filename in files:
                os.chown(Path(directory) / filename, uid, gid)


if __name__ == "__main__":
    main()
