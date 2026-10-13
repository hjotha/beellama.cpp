#!/usr/bin/env python3
"""F4 step 1: reproduce the position-split abort and capture the exact assertion.

Runs the 4B fixture at a chosen (P, C, n_tokens) and records, for the failing
case, the FULL server output (stdout+stderr merged straight into the log file so
the ggml_abort() assertion text is not lost) plus the dual-FA dispatch lines that
describe which backend handled Q/K/V/O/LSE.

usage: repro-abort.py <P> <C> <n_tokens> <port> <outdir> [decode]
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

SERVER = "/home/hjotha/beellama.cpp/build-optimized/bin/llama-server"
MODEL = "/home/hjotha/models/Qwen3.5-4B-Q4_K_M.gguf"


def http(port, path, payload=None, timeout=600):
    url = "http://127.0.0.1:%d%s" % (port, path)
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        b = e.read().decode()
        try:
            b = json.loads(b)
        except Exception:
            pass
        return e.code, b
    except Exception as e:  # noqa: BLE001
        return 0, {"exception": repr(e)}


def main():
    P, C, n_tok = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    port = int(sys.argv[4])
    outdir = sys.argv[5]
    decode = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    os.makedirs(outdir, exist_ok=True)
    tag = "P%d-C%d-n%d" % (P, C, n_tok)
    log_path = os.path.join(outdir, tag + ".log")

    env = dict(os.environ)
    env["LLAMA_KV_POSITION_SPLIT_P"] = str(P)
    env["TMPDIR"] = "/home/hjotha/occupancy-f4-spool"
    # the KVarN dispatch prints the exact Q/K/V shapes when a route is refused
    if os.environ.get("F4_KVARN_DEBUG", "1") == "1":
        env["GGML_KVARN_DEBUG_ROUTES"] = "1"
    if os.environ.get("F4_VIEW_TRACE", "1") == "1":
        env["GGML_KVARN_VIEW_TRACE"] = "1"
    if os.environ.get("F4_RANGE_TRACE", "1") == "1":
        env["LLAMA_KVARN_RANGE_TRACE"] = "1"
    if os.environ.get("F4_STORE_TRACE", "1") == "1":
        env["LLAMA_KVARN_STORE_TRACE"] = "1"
    if os.environ.get("F4_SLOT_TRACE", "1") == "1":
        env["LLAMA_KVARN_SLOT_TRACE"] = "1"
    # merge stderr into the same file so ggml_abort's message is captured
    log = open(log_path, "wb")
    cmd = [SERVER, "--model", MODEL, "--host", "127.0.0.1", "--port", str(port),
           "--device", "CUDA0", "-ngl", "99", "--split-mode", "none", "--main-gpu", "0",
           "--ctx-size", str(C), "--parallel", "1", "--batch-size", "256",
           "--ubatch-size", "256", "--cache-type-k", "kvarn4", "--cache-type-v", "kvarn4",
           "--kv-tail-tokens", "0", "--kv-tail-type", "f16", "--flash-attn", "on",
           "--cache-prompt", "--fit", "off", "--no-ui", "--slots", "--no-warmup",
           "--no-context-shift", "--remote-attn", "vulkan:0", "--remote-attn-layers", "8",
           "--remote-attn-cache-type-k", "q4_0", "--remote-attn-cache-type-v", "q4_0",
           "--spec-type", "none", "--verbose"]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    rc = None
    try:
        for _ in range(300):
            if http(port, "/health", timeout=2)[0] == 200:
                break
            if proc.poll() is not None:
                print("%s: server exited during startup rc=%s" % (tag, proc.returncode))
                return 1
            time.sleep(1)
        st, tok = http(port, "/tokenize",
                       {"content": " ".join("w%d" % i for i in range(n_tok * 2))})
        if st != 200:
            print("%s: tokenize failed %s" % (tag, tok))
            return 1
        ids = tok["tokens"][:n_tok]
        st, body = http(port, "/completion",
                        {"prompt": ids, "n_predict": decode, "temperature": 0.0,
                         "cache_prompt": True, "seed": 42, "n_probs": 5})
        if st != 200:
            print("%s: completion FAILED http=%s %s" % (tag, st, body))
            rc = 1
        else:
            print("%s: OK n_predict=%d content=%r" % (
                tag, decode, (body.get("content") or "")[:60]))
            t = body.get("timings", {})
            print("%s: prompt_n=%s prompt_ms=%.1f tps=%.1f" % (
                tag, t.get("prompt_n"), t.get("prompt_ms") or 0.0,
                t.get("prompt_per_second") or 0.0))
    finally:
        # exact-PID termination only: never a broad pattern kill
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=25)
        except Exception:  # noqa: BLE001
            pass
        log.close()

    # extract the evidence lines
    try:
        with open(log_path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        lines = []
    asserts = [l.strip() for l in lines
               if "GGML_ASSERT" in l or "assertion" in l.lower()
               or ".cu:" in l or "unsupported" in l.lower()]
    shapes = [l.strip() for l in lines if "kvarn-route" in l or "q-shape" in l]
    views = [l.strip() for l in lines if "ggml_kvarn_view:" in l]
    rng = [l.strip() for l in lines if "kv-range-trace" in l]
    stores = [l.strip() for l in lines if "kv-store-trace" in l]
    resets = [l.strip() for l in lines if "kv-slot-reset" in l]
    aborts = [l.strip() for l in lines if "ggml_abort" in l or "abort" in l.lower()]
    fa = [l.strip() for l in lines if "position-split dual FA" in l
          or "use_position_split_attention" in l]
    print("\n--- assertion lines ---")
    for l in asserts[:8]:
        print("  " + l[:200])
    print("--- kv-slot-reset (%d) ---" % len(resets))
    for l in resets[:4]:
        print("  " + l[:160])
    print("--- kv-store-trace (last 4 of %d) ---" % len(stores))
    for l in stores[-4:]:
        print("  " + l[:300])
    print("--- kv-range-trace (last 4 of %d) ---" % len(rng))
    for l in rng[-4:]:
        print("  " + l[:340])
    print("--- kvarn_view geometry (last 6 of %d) ---" % len(views))
    for l in views[-6:]:
        print("  " + l[:300])
    print("--- kvarn route shapes ---")
    for l in shapes[:6]:
        print("  " + l[:300])
    print("--- abort/backtrace ---")
    for l in aborts[:4]:
        print("  " + l[:200])
    print("--- split dispatch (first 6 of %d) ---" % len(fa))
    for l in fa[:6]:
        print("  " + l[:200])
    print("log: %s" % log_path)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())