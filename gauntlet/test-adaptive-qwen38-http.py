#!/usr/bin/env python3
"""Native HTTP contract coverage for the real Qwen3.8 adaptive server."""
import json
import os
import pathlib
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(os.environ.get("ADAPTIVE_HTTP_OUT", ROOT / "gauntlet/stage-8-http-qwen38"))
OUT.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("ADAPTIVE_HTTP_PORT", "19403"))
MODEL = os.environ.get("ADAPTIVE_MODEL", "/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf")
SERVER = os.environ.get("ADAPTIVE_SERVER", str(ROOT / "build-adaptive-cuda/bin/llama-server"))

cmd = [
    SERVER, "--model", MODEL, "--host", "127.0.0.1", "--port", str(PORT),
    "--ctx-size", "97536", "--ctx-size-mtp", "56320", "--mtp-max-tokens", "8",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-p-min", "0.80",
    "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0", "--gpu-layers", "99",
    "--device", "CUDA0", "--flash-attn", "on", "--fit", "off", "--parallel", "1",
    "--batch-size", "256", "--ubatch-size", "256", "--cache-type-k", "q4_0",
    "--cache-type-v", "q4_0", "--cache-ram", "2048", "--ctx-checkpoints", "1",
    "--load-mode", "none", "--no-warmup", "--no-context-shift", "--slots", "--n-predict", "0",
    "--jinja", "--slot-save-path", str(OUT),
]
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = str(ROOT / "build-adaptive-cuda/bin") + ":/opt/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")


def gpu_sample():
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
        used, free, util = [int(v.strip()) for v in raw.split(",", 2)]
        return {"memory_used_mib": used, "memory_free_mib": free, "utilization_gpu": util}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def call(method, path, payload=None, headers=None, timeout=900, parse_json=True):
    data = None if payload is None else json.dumps(payload).encode()
    request_headers = {"Content-Type": "application/json"} if data else {}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=data, method=method, headers=request_headers,
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw, status = response.read(), response.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read(), exc.code
    elapsed = time.monotonic() - started
    if parse_json:
        try:
            body = json.loads(raw)
        except Exception:
            body = {"raw": raw.decode(errors="replace")}
    else:
        body = raw.decode(errors="replace")
    return status, body, elapsed


def wait_ready(process):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before readiness rc={process.returncode}")
        try:
            status, _, _ = call("GET", "/health", timeout=10)
            if status == 200:
                return
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    raise RuntimeError("HTTP harness readiness timeout")


def adaptive(body):
    return body.get("adaptive_context", {}) if isinstance(body, dict) else {}


def compact(body):
    if not isinstance(body, dict):
        return {"type": type(body).__name__}
    result = {}
    for key in ("n_saved", "n_restored", "tokens_evaluated", "tokens_predicted", "truncated", "stop", "stop_type", "object", "id"):
        if key in body:
            result[key] = body[key]
    if "timings" in body:
        result["timings"] = body["timings"]
    if "usage" in body:
        result["usage"] = body["usage"]
    if "choices" in body:
        result["choices_n"] = len(body["choices"])
    if "output" in body:
        result["output_n"] = len(body["output"])
    if "error" in body:
        result["error"] = body["error"]
    return result

results = {"port": PORT, "model": MODEL, "steps": [], "errors": []}
process = None
log_path = OUT / "server.log"
with log_path.open("w") as log:
    try:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        wait_ready(process)

        status, body, elapsed = call("GET", "/props", timeout=30)
        results["steps"].append({"name": "initial-props", "status": status, "elapsed_s": elapsed,
                                 "adaptive_context": adaptive(body), "gpu": gpu_sample()})
        if status != 200 or adaptive(body).get("profile") != "mtp":
            raise RuntimeError(f"initial props failed: {status} {body}")

        for path in ("/models", "/slots"):
            status, body, elapsed = call("GET", path, timeout=30)
            results["steps"].append({"name": path, "status": status, "elapsed_s": elapsed,
                                     "adaptive_context": adaptive(body["data"][0] if path == "/models" else body[0]) if status == 200 else None,
                                     "gpu": gpu_sample()})
            if status != 200:
                raise RuntimeError(f"{path} failed: {status} {body}")

        # Keep the saved snapshot in the MTP tier when the probe threshold is 8,
        # while leaving at least one token for the cache_n reuse assertion.
        short_prompt = "one two three four five"
        status, body, elapsed = call("POST", "/completion", {
            "prompt": short_prompt, "id_slot": 0, "cache_prompt": True,
            "n_predict": 1, "temperature": 0,
        })
        results["steps"].append({"name": "short", "status": status, "elapsed_s": elapsed,
                                 "response": compact(body), "gpu": gpu_sample()})
        if status != 200:
            raise RuntimeError(f"short failed: {status} {body}")

        status, body, elapsed = call("POST", "/slots/0?action=save", {"filename": "short.bin"})
        results["steps"].append({"name": "save-short", "status": status, "elapsed_s": elapsed,
                                 "response": compact(body), "gpu": gpu_sample()})
        if status != 200 or int(body.get("n_saved", 0)) <= 0:
            raise RuntimeError(f"save short failed: {status} {body}")

        long_prompt = ("adaptive transition preserves the complete formatted prompt state. " * 8).strip()
        status, body, elapsed = call("POST", "/completion", {
            "prompt": long_prompt, "id_slot": 0, "cache_prompt": True,
            "n_predict": 1, "temperature": 0,
        })
        results["steps"].append({"name": "long", "status": status, "elapsed_s": elapsed,
                                 "response": compact(body), "gpu": gpu_sample()})
        if status != 200:
            raise RuntimeError(f"long failed: {status} {body}")
        status, props, elapsed = call("GET", "/props", timeout=30)
        results["steps"].append({"name": "long-props", "status": status, "elapsed_s": elapsed,
                                 "adaptive_context": adaptive(props), "gpu": gpu_sample()})
        if status != 200 or adaptive(props).get("profile") != "long":
            raise RuntimeError(f"long props mismatch: {status} {props}")

        conv_id = "adaptive-qwen38::http"
        status, stream, elapsed = call("POST", "/completion", {
            "prompt": "stream after long profile", "id_slot": 0, "cache_prompt": True,
            "n_predict": 1, "temperature": 0, "stream": True,
        }, headers={"X-Conversation-Id": conv_id}, timeout=900, parse_json=False)
        results["steps"].append({"name": "completion-stream", "status": status, "elapsed_s": elapsed,
                                 "events": stream.count("data:"), "has_done": "[DONE]" in stream,
                                 "has_data": "data: " in stream, "gpu": gpu_sample()})
        if status != 200 or "data: " not in stream:
            raise RuntimeError(f"stream failed: {status} {stream[:400]}")

        query = urllib.parse.urlencode({"conv_id": conv_id, "from": "0"})
        status, replay, elapsed = call("GET", f"/v1/stream?{query}", timeout=30, parse_json=False)
        results["steps"].append({"name": "stream-replay", "status": status, "elapsed_s": elapsed,
                                 "has_data": "data: " in replay, "bytes": len(replay), "gpu": gpu_sample()})
        if status != 200 or "data: " not in replay:
            raise RuntimeError(f"replay failed: {status} {replay[:400]}")

        status, chat, elapsed = call("POST", "/v1/chat/completions", {
            "model": "qwen38-adaptive", "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 1, "temperature": 0,
        })
        results["steps"].append({"name": "chat", "status": status, "elapsed_s": elapsed,
                                 "response": compact(chat), "gpu": gpu_sample()})
        if status != 200 or not chat.get("choices"):
            raise RuntimeError(f"chat failed: {status} {chat}")

        status, response, elapsed = call("POST", "/v1/responses", {
            "model": "qwen38-adaptive", "input": "Say hello.",
            "max_output_tokens": 1, "temperature": 0,
        })
        results["steps"].append({"name": "responses", "status": status, "elapsed_s": elapsed,
                                 "response": compact(response), "gpu": gpu_sample()})
        if status != 200 or not response.get("output"):
            raise RuntimeError(f"responses failed: {status} {response}")

        tool = {"type": "function", "function": {
            "name": "noop", "description": "Return no result.",
            "parameters": {"type": "object", "properties": {}},
        }}
        status, tool_body, elapsed = call("POST", "/v1/chat/completions", {
            "model": "qwen38-adaptive", "messages": [{"role": "user", "content": "Say hello."}],
            "tools": [tool], "tool_choice": "none", "max_tokens": 1, "temperature": 0,
        })
        results["steps"].append({"name": "tools", "status": status, "elapsed_s": elapsed,
                                 "response": compact(tool_body), "gpu": gpu_sample()})
        if status != 200:
            raise RuntimeError(f"tools failed: {status} {tool_body}")

        status, restored, elapsed = call("POST", "/slots/0?action=restore", {"filename": "short.bin"})
        results["steps"].append({"name": "restore-short", "status": status, "elapsed_s": elapsed,
                                 "response": compact(restored), "gpu": gpu_sample()})
        if status != 200 or int(restored.get("n_restored", 0)) <= 0:
            raise RuntimeError(f"restore short failed: {status} {restored}")
        status, props, elapsed = call("GET", "/props", timeout=30)
        results["steps"].append({"name": "restored-props", "status": status, "elapsed_s": elapsed,
                                 "adaptive_context": adaptive(props), "gpu": gpu_sample()})
        if status != 200 or adaptive(props).get("profile") != "mtp":
            raise RuntimeError(f"restored props mismatch: {status} {props}")

        status, reused, elapsed = call("POST", "/completion", {
            "prompt": short_prompt, "id_slot": 0, "cache_prompt": True,
            "n_predict": 1, "temperature": 0,
        })
        results["steps"].append({"name": "reuse-short", "status": status, "elapsed_s": elapsed,
                                 "response": compact(reused), "gpu": gpu_sample()})
        if status != 200 or int((reused.get("timings") or {}).get("cache_n", 0)) <= 0:
            raise RuntimeError(f"cache reuse failed: {status} {reused}")
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
        (OUT / "result.json").write_text(json.dumps(results, indent=2) + "\n")

log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
results["log_summary"] = {
    "load_model_lines": len(re.findall(r"load_model: loading model", log_text)),
    "transition_lines": len(re.findall(r"adaptive context transition complete", log_text)),
    "mtp_gpu_zero": len(re.findall(r"MTP_GPU=0", log_text)),
    "mtp_gpu_resident": len(re.findall(r"MTP_GPU=348469248", log_text)),
    "fatal_markers": {p: len(re.findall(re.escape(p), log_text, re.I)) for p in ("out of memory", "Xid", "cuda error", "GGML_ASSERT", "abort", "segmentation fault")},
}
# Rewrite with log summary after process cleanup.
(OUT / "result.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps({"success": results.get("success"), "steps": len(results.get("steps", [])),
                  "errors": results.get("errors"), "server_rc": results.get("server_rc"),
                  "log_summary": results.get("log_summary")}, sort_keys=True))
if not results.get("success"):
    raise SystemExit(1)
