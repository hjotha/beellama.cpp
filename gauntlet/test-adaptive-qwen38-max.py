import json, os, pathlib, subprocess, time, urllib.error, urllib.request
root = pathlib.Path("/home/hjotha/worktrees/llama-adaptive-context-impl")
out = root / os.environ.get("ADAPTIVE_MAX_OUT", "gauntlet/stage-13-max-useful")
port = int(os.environ.get("ADAPTIVE_MAX_PORT", "19416"))
model = "/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
cmd = [str(root / "build-adaptive-cuda/bin/llama-server"), "--model", model,
       "--host", "127.0.0.1", "--port", str(port), "--ctx-size", "97536",
       "--ctx-size-mtp", "56320", "--mtp-max-tokens", "8", "--spec-type", "draft-mtp",
       "--spec-draft-n-max", "2", "--spec-draft-p-min", "0.80", "--spec-draft-type-k", "q4_0",
       "--spec-draft-type-v", "q4_0", "--gpu-layers", "99", "--device", "CUDA0",
       "--flash-attn", "on", "--fit", "off", "--parallel", "1", "--batch-size", "256",
       "--ubatch-size", "256", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
       "--cache-ram", "2048", "--ctx-checkpoints", "1", "--load-mode", "none",
       "--no-warmup", "--no-context-shift", "--slots", "--n-predict", "0"]
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = "/home/hjotha/worktrees/llama-adaptive-context-impl/build-adaptive-cuda/bin:/opt/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
log_path = out / "server.log"
results = []

def gpu():
    try:
        line = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader"], text=True).strip()
        return line
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

with log_path.open("w") as log:
    proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    def call(method, path, payload=None, timeout=1800):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return res.status, json.loads(res.read())
        except urllib.error.HTTPError as exc:
            try: body = json.loads(exc.read())
            except Exception: body = {"raw": exc.read().decode(errors="replace")}
            return exc.code, body
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                status, _ = call("GET", "/health", timeout=10)
                if status == 200: break
            except (OSError, ValueError, json.JSONDecodeError): pass
            time.sleep(0.5)
        else: raise RuntimeError("max prefill server readiness timeout")
        status, props = call("GET", "/props")
        results.append({"name":"initial-props", "status":status, "props":props, "gpu":gpu()})
        status, body = call("POST", "/completion", {"prompt":"hi", "n_predict":0, "temperature":0, "cache_prompt":False}, timeout=900)
        results.append({"name":"short", "status":status, "body":body, "gpu":gpu()})
        if status != 200: raise RuntimeError(f"short completion failed: {status} {body}")
        # Find the largest repeated prompt whose /tokenize count is <= the declared long ceiling.
        target = 97534
        lo, hi = 1, 110000
        while lo < hi:
            mid = (lo + hi + 1) // 2
            text = "x " * mid
            status, body = call("POST", "/tokenize", {"content": text, "add_special": True}, timeout=180)
            if status != 200: raise RuntimeError(f"tokenize failed: {status} {body}")
            count = len(body.get("tokens", []))
            if count <= target: lo = mid
            else: hi = mid - 1
        prompt = "x " * lo
        status, token_body = call("POST", "/tokenize", {"content": prompt, "add_special": True}, timeout=180)
        token_count = len(token_body.get("tokens", []))
        results.append({"name":"token-count", "status":status, "units":lo, "token_count":token_count, "target":target, "gpu":gpu()})
        if token_count != target:
            raise RuntimeError(f"could not construct exact max prompt: {token_count} != {target}")
        started = time.monotonic()
        status, body = call("POST", "/completion", {"prompt":prompt, "n_predict":2, "temperature":0, "ignore_eos":True, "cache_prompt":False}, timeout=1800)
        results.append({"name":"max-prefill", "status":status, "elapsed_s":round(time.monotonic()-started,2), "body":body, "gpu":gpu()})
        if status != 200: raise RuntimeError(f"max prefill failed: {status} {body}")
        if body.get("tokens_evaluated") != target or body.get("tokens_predicted") != 2:
            raise RuntimeError(f"max useful output mismatch: {body.get('tokens_evaluated')} evaluated, {body.get('tokens_predicted')} predicted")
        status, body = call("GET", "/props")
        results.append({"name":"long-props", "status":status, "props":body, "gpu":gpu()})
        status, body = call("POST", "/completion", {"prompt":"hi", "n_predict":0, "temperature":0, "cache_prompt":False}, timeout=900)
        short_return_status = status
        results.append({"name":"short-return", "status":short_return_status, "body":body, "gpu":gpu()})
        if short_return_status != 200: raise RuntimeError(f"short return failed: {short_return_status} {body}")
        status, body = call("GET", "/props")
        results.append({"name":"return-props", "status":status, "props":body, "gpu":gpu()})
        (out / "result.json").write_text(json.dumps({"results":results}, indent=2) + "\n")
        print(json.dumps({"token_count":token_count, "max_status":results[-4]["status"], "short_return_status":short_return_status}, sort_keys=True), flush=True)
    finally:
        proc.terminate()
        try: proc.wait(timeout=60)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=60)
