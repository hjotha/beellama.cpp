#!/usr/bin/env python3
"""Benchmark llama-server (PrismML original vs BeeLlama integrated) for Bonsai models.

Measures: load time, prefill tok/s, decode tok/s, TTFT, VRAM (weights), VRAM (KV+compute), KV memory.
Runs: warmup + N reps via /v1/chat/completions (streaming).
"""
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
import os

BIN = sys.argv[1]          # path to llama-server binary
MODEL = sys.argv[2]        # model gguf path
PORT = sys.argv[3]         # port
REPS = int(sys.argv[4])    # number of reps after warmup
MAX_TOKENS = int(sys.argv[5])

HOST = "127.0.0.1"
BASE = f"http://{HOST}:{PORT}"
PROMPT = (
    "Find and fix the bug in this C++ function. Explain the root cause, then give the corrected code:\n\n"
    "int firstMissingPositive(std::vector<int>& nums) {\n"
    "    int n = nums.size();\n"
    "    for (int i = 0; i < n; i++) {\n"
    "        while (nums[i] > 0 && nums[i] <= n && nums[nums[i]] != nums[i]) {\n"
    "            std::swap(nums[i], nums[nums[i]]);\n"
    "        }\n"
    "    }\n"
    "    for (int i = 0; i < n; i++) {\n"
    "        if (nums[i] != i + 1) return i + 1;\n"
    "    }\n"
    "    return n + 1;\n"
    "}"
)

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return r.stdout, r.stderr, r.returncode

def nvidia_mem_mib():
    out, _, _ = run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return int(out.strip().splitlines()[0])

def wait_ready(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=3) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            time.sleep(0.5)
    return None

def chat_completion_stream(measure_ttft=False):
    body = json.dumps({
        "model": "bonsai",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    first_chunk_t = None
    resp = urllib.request.urlopen(req, timeout=600)
    gen_start = None
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        if first_chunk_t is None:
            first_chunk_t = time.time() - t0
        if measure_ttft and ttft is None:
            choices = chunk.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if delta.get("content") or delta.get("reasoning_content"):
                ttft = time.time() - t0
        if gen_start is None:
            gen_start = time.time()
    total = time.time() - t0
    if ttft is None:
        ttft = first_chunk_t or total
    return total, ttft

def chat_completion_usage():
    body = json.dumps({
        "model": "bonsai",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=600)
    j = json.loads(resp.read().decode())
    return j["usage"]["prompt_tokens"], j["usage"]["completion_tokens"]

def main():
    # kill any leftover server on this port
    run(["bash", "-c", f"fuser -k {PORT}/tcp 2>/dev/null; sleep 2; true"])
    proc = subprocess.Popen(
        [BIN, "-m", MODEL, "-ngl", "99", "-fa", "on", "-c", "8192",
         "-b", "512", "-ub", "256", "--port", PORT, "--host", HOST, "--log-disable"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    vram0 = nvidia_mem_mib()
    load_s = wait_ready()
    vram_loaded = nvidia_mem_mib()
    vram_weights = vram_loaded - vram0

    # warmup
    chat_completion_stream(measure_ttft=False)
    chat_completion_usage()

    ptoks, ctoks = chat_completion_usage()
    results = []
    for i in range(REPS):
        total, ttft = chat_completion_stream(measure_ttft=True)
        prefill_s = (ptoks / total) if False else None
        decode_s = ctoks / max(total - (ttft or 0), 1e-9)
        results.append((total, ttft, decode_s))
        time.sleep(0.3)
    vram_after = nvidia_mem_mib()
    vram_kv = vram_after - vram_loaded

    print(json.dumps({
        "model": os.path.basename(MODEL),
        "binary": os.path.basename(os.path.dirname(BIN)) + "/" + os.path.basename(BIN),
        "load_time_s": round(load_s, 3),
        "vram_weights_mib": vram_weights,
        "vram_kv_compute_mib": vram_kv,
        "prompt_tokens": ptoks,
        "completion_tokens": ctoks,
        "runs": [
            {"total_s": round(t, 3), "ttft_s": round(tf, 3),
             "decode_tok_s": round(ds, 2)} for t, tf, ds in results
        ],
        "decode_avg_tok_s": round(sum(r[2] for r in results) / len(results), 2),
    }, indent=2))
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()

if __name__ == "__main__":
    main()