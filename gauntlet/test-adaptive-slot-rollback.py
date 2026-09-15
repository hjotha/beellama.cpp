#!/usr/bin/env python3
"""Prove that a failed cross-profile slot restore preserves the previous slot."""

import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER = pathlib.Path(os.environ.get("ADAPTIVE_SERVER", ROOT / "build/bin/llama-server"))
MODEL = pathlib.Path(os.environ.get("ADAPTIVE_MODEL", ROOT / "gauntlet/Qwen3.5-4B-MTP-Q4_K_M.gguf"))
PORT = int(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)


def call(method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw, status = response.read(), response.status
    except urllib.error.HTTPError as error:
        raw, status = error.read(), error.code
    return status, json.loads(raw)


def wait_ready():
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            status, _ = call("GET", "/health")
            if status == 200:
                return
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(0.25)
    raise RuntimeError("rollback test server did not become ready")


cmd = [
    str(SERVER), "--model", str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
    "--ctx-size", "512", "--ctx-size-mtp", "256", "--mtp-max-tokens", "256",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-p-min", "0.80",
    "--n-gpu-layers", os.environ.get("ADAPTIVE_GPU_LAYERS", "0"),
    "--device", os.environ.get("ADAPTIVE_DEVICE", "none"), "--fit", "off",
    "--no-repack", "--parallel", "1", "--slot-save-path", str(OUT),
    "--cache-ram", os.environ.get("ADAPTIVE_CACHE_RAM", "0"),
    "--threads", "8", "--threads-batch", "8", "--batch-size", "64", "--ubatch-size", "64",
    "--load-mode", "none", "--temp", "0", "--no-warmup",
]
env = os.environ.copy()
env["LLAMA_TEST_ADAPTIVE_TRANSITION_FAIL"] = "mtp"
log_path = OUT / "server.log"
with log_path.open("w") as log:
    process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)

results = []
try:
    wait_ready()
    status, props = call("GET", "/props")
    assert status == 200 and props["adaptive_context"]["profile"] == "mtp", props

    short_prompt = "Explain adaptive cache in one sentence."
    status, short = call("POST", "/completion", {
        "prompt": short_prompt, "id_slot": 0, "cache_prompt": True,
        "n_predict": 2, "temperature": 0,
    })
    assert status == 200, short
    status, saved_short = call("POST", "/slots/0?action=save", {"filename": "short.bin"})
    assert status == 200, saved_short

    long_prompt = ("adaptive context transition preserves the evaluated prompt state and reuses it safely. " * 25).strip()
    status, long = call("POST", "/completion", {
        "prompt": long_prompt, "id_slot": 0, "cache_prompt": True,
        "n_predict": 1, "temperature": 0,
    })
    assert status == 200, long
    status, saved_long = call("POST", "/slots/0?action=save", {"filename": "long.bin"})
    assert status == 200, saved_long

    if int(os.environ.get("ADAPTIVE_CACHE_RAM", "0")) > 0:
        for index in range(8):
            status, filled = call("POST", "/completion", {
                "prompt": f"adaptive cache fill {index}: " + long_prompt,
                "id_slot": 0, "cache_prompt": True,
                "n_predict": 1, "temperature": 0,
            })
            assert status == 200, filled
        status, restored_long = call("POST", "/slots/0?action=restore", {"filename": "long.bin"})
        assert status == 200, restored_long

    failed_restore_status, failed_restore = call("POST", "/slots/0?action=restore", {"filename": "short.bin"})
    assert failed_restore_status >= 400, failed_restore
    status, after = call("GET", "/props")
    assert status == 200 and after["adaptive_context"]["profile"] == "long" and after["adaptive_context"]["state"] == "ready", after

    status, reused = call("POST", "/completion", {
        "prompt": long_prompt, "id_slot": 0, "cache_prompt": True,
        "n_predict": 1, "temperature": 0,
    })
    assert status == 200, reused
    assert reused["timings"]["cache_n"] > 0, reused
    results = {"failed_restore_status": failed_restore_status, "profile_after": after["adaptive_context"],
               "cache_n_after": reused["timings"]["cache_n"],
               "cache_ram_mib": int(os.environ.get("ADAPTIVE_CACHE_RAM", "0")),
               "saved_short": saved_short,
               "saved_long": saved_long}
finally:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)

# A disk snapshot must remain usable after the original model instance exits.
restart_env = os.environ.copy()
restart_env.pop("LLAMA_TEST_ADAPTIVE_TRANSITION_FAIL", None)
restart_log_path = OUT / "server-restart.log"
with restart_log_path.open("w") as restart_log:
    restarted = subprocess.Popen(cmd, cwd=ROOT, env=restart_env,
                                 stdout=restart_log, stderr=subprocess.STDOUT)
try:
    wait_ready()
    status, restored = call("POST", "/slots/0?action=restore", {"filename": "long.bin"})
    assert status == 200, restored
    status, restart_props = call("GET", "/props")
    assert status == 200 and restart_props["adaptive_context"]["profile"] == "long", restart_props
    results["restart_restore_status"] = status
    results["restart_profile"] = restart_props["adaptive_context"]
finally:
    restarted.terminate()
    try:
        restarted.wait(timeout=15)
    except subprocess.TimeoutExpired:
        restarted.kill()
        restarted.wait(timeout=15)

(OUT / "result-rollback.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps({"failed_restore_status": results["failed_restore_status"],
                  "profile_after": results["profile_after"]["profile"],
                  "cache_n_after": results["cache_n_after"],
                  "restart_restore_status": results["restart_restore_status"]}))
