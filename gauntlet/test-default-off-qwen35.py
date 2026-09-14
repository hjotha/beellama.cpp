#!/usr/bin/env python3
"""Check the legacy server contract with adaptive context disabled."""
import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(os.environ.get("DEFAULT_OFF_OUT", ROOT / "gauntlet/stage-11-default-off-qwen35"))
OUT.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("DEFAULT_OFF_PORT", "19410"))
MODEL = os.environ.get("DEFAULT_OFF_MODEL", "/home/hjotha/models/Qwen3.5-4B-Q4_K_M.gguf")
SERVER = os.environ.get("DEFAULT_OFF_SERVER", str(ROOT / "build-adaptive-cuda/bin/llama-server"))
CMD = [
    SERVER, "--model", MODEL, "--host", "127.0.0.1", "--port", str(PORT),
    "--ctx-size", "512", "--parallel", "1", "--gpu-layers", "99", "--device", "CUDA0",
    "--fit", "off", "--flash-attn", "on", "--slots", "--batch-size", "256",
    "--ubatch-size", "256", "--load-mode", "none", "--no-warmup", "--no-context-shift",
    "--n-predict", "2", "--no-jinja",
]
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = str(ROOT / "build-adaptive-cuda/bin") + ":/opt/cuda/lib64:" + ENV.get("LD_LIBRARY_PATH", "")


def call(method, path, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw, status = response.read(), response.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read(), exc.code
    try:
        body = json.loads(raw)
    except Exception:
        body = {"raw": raw.decode(errors="replace")}
    return status, body


result = {"port": PORT, "model": MODEL, "checks": [], "success": False}
process = None
with (OUT / "server.log").open("w") as log:
    try:
        process = subprocess.Popen(CMD, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited before readiness rc={process.returncode}")
            try:
                if call("GET", "/health", timeout=10)[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("readiness timeout")

        status, props = call("GET", "/props")
        assert status == 200 and "adaptive_context" not in props, props
        assert ".gguf" in props.get("model_path", ""), props
        result["checks"].append({"endpoint": "/props", "status": status,
                                  "model_path": props.get("model_path"),
                                  "adaptive_present": "adaptive_context" in props})

        status, models = call("GET", "/models")
        assert status == 200 and len(models.get("data", [])) == 1, models
        assert "adaptive_context" not in models and "adaptive_context" not in models["data"][0], models
        assert models["models"][0]["context_window"] == 512, models
        assert models["models"][0]["max_context_window"] == 512, models
        result["checks"].append({"endpoint": "/models", "status": status,
                                  "context_window": models["models"][0]["context_window"],
                                  "max_context_window": models["models"][0]["max_context_window"],
                                  "adaptive_present": False})

        status, slots = call("GET", "/slots")
        assert status == 200 and isinstance(slots, list) and slots, slots
        assert all("adaptive_context" not in slot for slot in slots), slots
        result["checks"].append({"endpoint": "/slots", "status": status, "slots": len(slots)})

        status, completion = call("POST", "/completion", {
            "prompt": "Say hello in one word.", "n_predict": 1, "temperature": 0,
        }, timeout=120)
        assert status == 200 and completion.get("content") is not None, completion
        assert completion.get("tokens_predicted") == 1, completion
        result["checks"].append({"endpoint": "/completion", "status": status,
                                  "tokens_predicted": completion.get("tokens_predicted")})
        result["success"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=60)
        result["server_rc"] = None if process is None else process.returncode
        if result.get("success") and result["server_rc"] != 0:
            result["success"] = False
            result["error"] = f"server exited with rc={result['server_rc']}"
        (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")

print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if result["success"] else 1)
