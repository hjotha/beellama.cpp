#!/usr/bin/env python3
"""Capture fixed teacher-forced prefixes with a diagnostic server we own.

Both arms receive identical token IDs. This runner does not assert numerical
acceptance: compare_logits.py evaluates the completed captures separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request


def request(url, payload=None, timeout=3600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--port", type=int, default=8280)
    parser.add_argument("--arm", choices=("gpu", "cpu"), required=True)
    parser.add_argument("--reference-suffix", action="store_true",
                        help="build a common GPU prefix, then reference only the fixed suffix")
    parser.add_argument("--optimized-overflow", action="store_true",
                        help="qualify packed Q4 with a real K dependency and local CUDA tail WHT")
    args = parser.parse_args()
    tokens = json.loads(Path(args.tokens).read_text())
    if isinstance(tokens, dict):
        tokens = tokens["prompt"]
    end = args.start + args.count
    if args.count <= 0 or args.start <= 0 or end > len(tokens) or end >= args.context:
        parser.error("invalid checkpoint interval or missing teacher token")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "teacher-tokens.json").write_text(json.dumps(tokens))
    env = dict(os.environ)
    env.update(F4_TEST_LOGIT_CAPTURE=str(out / "cap"), F4_TEST_PHASE=args.arm,
               F4_TEST_EXPECT_TOKENS=",".join(str(n) for n in range(args.start, end)),
               F4_TEST_RUN_ID=out.name, GGML_KVARN_WINDOW_CHUNK="4096",
               LLAMA_GRAPH_REUSE_DISABLE="1")
    if args.optimized_overflow:
        env.update(LLAMA_OVF_PACKED_VIEW="1", LLAMA_OVF_PACKED_Q4_FA="1",
                   F5_OVF_CONT_K_ONLY="1", LLAMA_OVF_TAIL_LOCAL_CUDA="1")
    if args.arm == "cpu":
        env.update(LLAMA_OVF_ATTN_REFERENCE_CPU="1", GGML_CPU_KVARN_ALLOW_MIXED_DOMAIN="1")
        if args.reference_suffix:
            env["LLAMA_OVF_ATTN_REFERENCE_CPU_MIN_POS"] = str(args.start-1)
    else:
        env.pop("LLAMA_OVF_ATTN_REFERENCE_CPU", None)
    argv = [str(Path(args.binary).resolve()), "-m", str(Path(args.model).resolve()),
            "-c", str(args.context), "-b", "512", "-ub", "256", "-np", "1",
            "-ngl", "99", "--device", "CUDA0", "--flash-attn", "on",
            "-ctk", "kvarn4", "-ctv", "kvarn4", "--fit", "off",
            "--no-context-shift", "--cache-ram", "0", "--ctx-checkpoints", "1",
            "--remote-attn", "vulkan:0", "--remote-attn-layers", "full",
            "--remote-attn-cache-type-k", "q4_0", "--remote-attn-cache-type-v", "q4_0",
            "--kv-position-split-p", str(args.p), "--host", "127.0.0.1",
            "--port", str(args.port), "--timeout", "3600", "--ignore-eos"]
    (out / "launch.json").write_text(json.dumps({"argv": argv,
        "env": {k: env[k] for k in env if k.startswith(("F4_", "F5_", "LLAMA_OVF", "LLAMA_GRAPH_", "GGML_"))},
        "tokens_sha256": hashlib.sha256(Path(args.tokens).read_bytes()).hexdigest()}, indent=2))
    report = {"schema": "occupancy-teacher-capture-v1", "arm": args.arm,
              "start": args.start, "count": args.count, "complete": False, "requests": []}
    url = f"http://127.0.0.1:{args.port}"
    with (out / "server.log").open("wb") as log:
        process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        def reap_aborted_helpers():
            process.wait()
            # Backtrace helpers can inherit HTTP sockets after a failed server.
            # Close our whole group so a dead leader cannot leave the client
            # blocked until the long-context HTTP timeout.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        threading.Thread(target=reap_aborted_helpers, daemon=True).start()
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited during startup: {process.returncode}")
                try:
                    request(url + "/health", timeout=2)
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("server readiness timeout")
                    time.sleep(0.5)
            if args.reference_suffix:
                payload = {"prompt": tokens[:args.start-1], "n_predict": 1,
                           "temperature": 0, "seed": 42, "cache_prompt": True,
                           "ignore_eos": True}
                (out / "common-prefix-request.json").write_text(json.dumps(payload))
                response = request(url + "/completion", payload)
                (out / "common-prefix-response.json").write_text(json.dumps(response, indent=2))
                if "error" in response:
                    raise RuntimeError("common prefix failed")
            for n in range(args.start, end):
                payload = {"prompt": tokens[:n], "n_predict": 1, "temperature": 0,
                           "seed": 42, "cache_prompt": True, "ignore_eos": True}
                (out / f"request-{n}.json").write_text(json.dumps(payload))
                started = time.monotonic()
                response = request(url + "/completion", payload)
                (out / f"response-{n}.json").write_text(json.dumps(response, indent=2))
                report["requests"].append({"n": n, "wall_seconds": time.monotonic()-started,
                                           "timings": response.get("timings")})
                if "error" in response or not (out / f"cap.{args.arm}.n{n}.logits").exists():
                    raise RuntimeError(f"missing capture or error at {n}")
                (out / "report.json").write_text(json.dumps(report, indent=2))
            report["complete"] = True
        except Exception as error:
            report["error"] = str(error)
            raise
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
            report["server_exit"] = process.returncode
            (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
