#!/usr/bin/env python3
"""Exercise repeated adaptive short/long transitions on the real Qwen3.8 GGUF."""
import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(os.environ.get("ADAPTIVE_CYCLES_OUT", ROOT / "gauntlet/stage-7-cycles"))
OUT.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("ADAPTIVE_CYCLES_PORT", "19402"))
MODEL = os.environ.get("ADAPTIVE_MODEL", "/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf")
SERVER = os.environ.get("ADAPTIVE_SERVER", str(ROOT / "build-adaptive-cuda/bin/llama-server"))
N_PREDICT = int(os.environ.get("ADAPTIVE_CYCLES_N_PREDICT", "1"))
MTP_MAX_TOKENS = int(os.environ.get("ADAPTIVE_CYCLES_MTP_MAX", "8"))
DECODE_MODE = N_PREDICT > 1

cmd = [
    SERVER, "--model", MODEL, "--host", "127.0.0.1", "--port", str(PORT),
    "--ctx-size", "97536", "--ctx-size-mtp", "56320", "--mtp-max-tokens", str(MTP_MAX_TOKENS),
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-p-min", "0.80",
    "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0", "--gpu-layers", "99",
    "--device", "CUDA0", "--flash-attn", "on", "--fit", "off", "--parallel", "1",
    "--batch-size", "256", "--ubatch-size", "256", "--cache-type-k", "q4_0",
    "--cache-type-v", "q4_0", "--cache-ram", "2048", "--ctx-checkpoints", "1",
    "--load-mode", "none", "--no-warmup", "--no-context-shift", "--slots", "--n-predict", "0",
]
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = str(ROOT / "build-adaptive-cuda/bin") + ":/opt/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")


def gpu_sample():
    try:
        line = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
        used, free, util = [int(x.strip()) for x in line.split(",", 2)]
        return {"memory_used_mib": used, "memory_free_mib": free, "utilization_gpu": util}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def process_memory(process):
    """Read resident/private memory without requiring a profiler or ptrace."""
    try:
        status = {}
        for line in pathlib.Path(f"/proc/{process.pid}/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM", "VmPeak", "VmSwap"}:
                status[key.lower()] = int(value.strip().split()[0])
        rollup = {}
        for line in pathlib.Path(f"/proc/{process.pid}/smaps_rollup").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"Rss", "Pss", "Private_Clean", "Private_Dirty", "Swap"}:
                rollup[key.lower()] = int(value.strip().split()[0])
        return {**status, **{f"smaps_{key}": value for key, value in rollup.items()}}
    except (OSError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def call(method, path, payload=None, timeout=900):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read(), exc.code
    elapsed = time.monotonic() - started
    try:
        body = json.loads(raw)
    except Exception:
        body = {"raw": raw.decode(errors="replace")}
    return status, body, elapsed


def wait_ready(process):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before readiness with rc={process.returncode}")
        try:
            status, _, _ = call("GET", "/health", timeout=10)
            if status == 200:
                return
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    raise RuntimeError("cycles server readiness timeout")


# Distinct widths keep the transition path honest while staying cheap after the max-prefill gate.
long_prompts = [
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty" + (" token") * (180 if DECODE_MODE else 0),
    "adaptive context cycle width sixty four " + ("token ") * (96 if DECODE_MODE else 64),
    "adaptive context cycle width one twenty eight " + ("token ") * 160,
    "adaptive context cycle width two fifty six " + ("token ") * 288,
]
short_prompt = "hi"
results = {"port": PORT, "model": MODEL, "n_predict": N_PREDICT,
           "mtp_max_tokens": MTP_MAX_TOKENS, "cycles": [], "errors": []}
log_path = OUT / "server.log"
process = None
with log_path.open("w") as log:
    try:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        wait_ready(process)
        status, body, elapsed = call("GET", "/props", timeout=30)
        results["initial"] = {"status": status, "elapsed_s": elapsed,
                               "adaptive_context": body.get("adaptive_context"), "gpu": gpu_sample(),
                               "memory": process_memory(process)}
        if status != 200:
            raise RuntimeError(f"initial props failed: {status} {body}")
        # Warmup establishes the short/MTP allocation before counting round trips.
        status, body, elapsed = call("POST", "/completion", {
            "prompt": short_prompt, "n_predict": N_PREDICT, "temperature": 0,
            "cache_prompt": False, "ignore_eos": DECODE_MODE,
        }, timeout=900)
        results["warmup"] = {"status": status, "elapsed_s": elapsed,
                              "timings": body.get("timings"), "gpu": gpu_sample(),
                              "memory": process_memory(process)}
        if status != 200:
            raise RuntimeError(f"warmup failed: {status} {body}")
        for index in range(1, 21):
            prompt = long_prompts[(index - 1) % len(long_prompts)]
            status, body, elapsed = call("POST", "/completion", {
                "prompt": prompt, "n_predict": N_PREDICT, "temperature": 0,
                "cache_prompt": False, "ignore_eos": DECODE_MODE,
            }, timeout=900)
            long_item = {"cycle": index, "profile": "long", "status": status,
                         "elapsed_s": elapsed, "timings": body.get("timings"),
                         "tokens_evaluated": body.get("tokens_evaluated"),
                         "tokens_predicted": body.get("tokens_predicted"), "gpu": gpu_sample(),
                         "memory": process_memory(process)}
            pstatus, props, pelapsed = call("GET", "/props", timeout=30)
            long_item["props_status"] = pstatus
            long_item["props_elapsed_s"] = pelapsed
            long_item["adaptive_context"] = props.get("adaptive_context")
            results["cycles"].append(long_item)
            if status != 200 or pstatus != 200 or (DECODE_MODE and body.get("tokens_predicted") != N_PREDICT):
                raise RuntimeError(f"cycle {index} long failed: completion={status} props={pstatus}")
            adaptive = props.get("adaptive_context", {})
            if adaptive.get("profile") != "long" or adaptive.get("mtp_weights_resident") is not False:
                raise RuntimeError(f"cycle {index} long profile mismatch: {adaptive}")

            status, body, elapsed = call("POST", "/completion", {
                "prompt": short_prompt, "n_predict": N_PREDICT, "temperature": 0,
                "cache_prompt": False, "ignore_eos": DECODE_MODE,
            }, timeout=900)
            short_item = {"cycle": index, "profile": "mtp", "status": status,
                          "elapsed_s": elapsed, "timings": body.get("timings"),
                          "tokens_evaluated": body.get("tokens_evaluated"),
                          "tokens_predicted": body.get("tokens_predicted"), "gpu": gpu_sample(),
                         "memory": process_memory(process)}
            pstatus, props, pelapsed = call("GET", "/props", timeout=30)
            short_item["props_status"] = pstatus
            short_item["props_elapsed_s"] = pelapsed
            short_item["adaptive_context"] = props.get("adaptive_context")
            results["cycles"].append(short_item)
            if status != 200 or pstatus != 200 or (DECODE_MODE and body.get("tokens_predicted") != N_PREDICT):
                raise RuntimeError(f"cycle {index} short failed: completion={status} props={pstatus}")
            adaptive = props.get("adaptive_context", {})
            if adaptive.get("profile") != "mtp" or adaptive.get("mtp_weights_resident") is not True:
                raise RuntimeError(f"cycle {index} short profile mismatch: {adaptive}")
        results["success"] = True
    except Exception as exc:
        results["success"] = False
        results["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=60)
        results["server_rc"] = None if process is None else process.returncode
        if results.get("success") and results["server_rc"] != 0:
            results["success"] = False
            results["errors"].append(f"server exited with rc={results['server_rc']}")
        (OUT / "result.json").write_text(json.dumps(results, indent=2) + "\n")

print(json.dumps({
    "success": results.get("success"), "cycles_recorded": len(results.get("cycles", [])),
    "errors": results.get("errors"), "server_rc": results.get("server_rc"),
}, sort_keys=True))
if not results.get("success"):
    raise SystemExit(1)
