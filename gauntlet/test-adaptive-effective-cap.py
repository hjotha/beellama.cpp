#!/usr/bin/env python3
"""Verify adaptive capacity uses the model's effective training-context cap."""

import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SERVER = Path(os.environ.get("ADAPTIVE_CAP_SERVER", ROOT / "build/bin/llama-server"))
MODEL = Path(os.environ.get("ADAPTIVE_CAP_MODEL", ROOT / "gauntlet/fixtures-stage2/qwen35-mtp.gguf"))
OUT = Path(os.environ.get("ADAPTIVE_CAP_OUT", ROOT / "gauntlet/stage-16-effective-cap"))
PORT = int(os.environ.get("ADAPTIVE_CAP_PORT", "19460"))


def call(path, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def start_server(log_path, ctx_size, ctx_size_mtp):
    command = [
        str(SERVER), "--model", str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
        "--ctx-size", str(ctx_size), "--ctx-size-mtp", str(ctx_size_mtp),
        "--mtp-max-tokens", str(ctx_size_mtp), "--spec-type", "draft-mtp",
        "--parallel", "1", "--gpu-layers", "0", "--device", "none", "--fit", "off",
        "--load-mode", "none", "--no-warmup", "--slots", "--batch-size", "32", "--ubatch-size", "32",
    ]
    log = log_path.open("w")
    return subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log


def stop_server(process, log):
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)
    log.close()


def wait_ready(process):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before readiness rc={process.returncode}")
        try:
            if call("/health")[0] == 200:
                return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise RuntimeError("server readiness timeout")


def assert_status(body, expected_context):
    status = body["adaptive_context"]
    assert status["enabled"] is True, status
    assert status["context_size"] == expected_context, status
    assert status["context_size_long"] == expected_context, status
    assert status["profile"] == "mtp", status
    assert status["state"] == "ready", status


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"success": False, "model": str(MODEL), "checks": []}

    invalid_log = OUT / "invalid-start.log"
    process, log = start_server(invalid_log, 512, 128)
    try:
        rc = process.wait(timeout=120)
    finally:
        log.close()
    invalid_text = invalid_log.read_text()
    assert rc != 0, rc
    assert "exceeds model training context" in invalid_text, invalid_text[-2000:]
    assert "adaptive context transition" not in invalid_text, invalid_text[-2000:]
    result["checks"].append({"case": "long-above-training", "server_rc": rc})

    valid_log = OUT / "valid-budget.log"
    process, log = start_server(valid_log, 256, 128)
    try:
        wait_ready(process)
        status, props = call("/props")
        assert status == 200, props
        assert_status(props, 256)
        status, models = call("/models")
        assert status == 200, models
        assert_status(models["data"][0], 256)
        status, slots = call("/slots")
        assert status == 200 and slots, slots
        assert_status(slots[0], 256)

        status, body = call(
            "/completion", "POST", {"prompt": [1] * 256, "n_predict": 1, "temperature": 0, "cache_prompt": False}
        )
        completion_status = status
        assert completion_status == 400, (completion_status, body)
        assert body["error"]["code"] == 400, body
        assert body["error"]["type"] == "exceed_context_size_error", body
        status, after = call("/props")
        assert status == 200, after
        assert_status(after, 256)
        result["checks"].append({
            "case": "budget-before-transition", "prompt_tokens": 256,
            "completion_status": completion_status, "error": body["error"],
            "props": after["adaptive_context"],
        })
        result["success"] = True
    finally:
        stop_server(process, log)

    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
