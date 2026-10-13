#!/usr/bin/env python3
"""F3: FP64 KL between the position-split candidate and the P=0 baseline.

Same model, same token ids, same seed, same sampler, same context size. The only
difference is LLAMA_KV_POSITION_SPLIT_P (candidate) vs unset (baseline).

The next-token distribution is read from /completion for the SAME position and
the KL is accumulated in Python floats (IEEE-754 binary64) over the union of
the returned tokens, renormalizing both sides.

SCOPE LIMIT (explicit, and why this does NOT close the F3 gate):
  /completion returns only a truncated top-N candidate set, so this is a KL over
  that truncated support, NOT over the full vocabulary as the plan requires
  (KL <= 1e-4 mean / 1e-3 p99 over all logits, teacher-forced, per-token). A
  truncated-support KL cannot bound the full-vocabulary KL: mass that falls
  outside the returned top-N is invisible here. This number is reported as
  diagnostic evidence only.

usage: f4-kl.py <P> <C> <n_tokens> <n_probs> <port>
"""
import json
import math
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


def spawn(port, P, C, tag, work, n_probs):
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ)
    if P:
        env["LLAMA_KV_POSITION_SPLIT_P"] = str(P)
    else:
        env.pop("LLAMA_KV_POSITION_SPLIT_P", None)
    env["TMPDIR"] = "/home/hjotha/occupancy-f4-spool"
    log = open("%s/%s.log" % (work, tag), "wb")
    proc = subprocess.Popen(
        [SERVER, "--model", MODEL, "--host", "127.0.0.1", "--port", str(port),
         "--device", "CUDA0", "-ngl", "99", "--split-mode", "none", "--main-gpu", "0",
         "--ctx-size", str(C), "--parallel", "1", "--batch-size", "256",
         "--ubatch-size", "256", "--cache-type-k", "kvarn4", "--cache-type-v", "kvarn4",
         "--kv-tail-tokens", "0", "--kv-tail-type", "f16", "--flash-attn", "on",
         "--cache-prompt", "--fit", "off", "--no-ui", "--slots", "--no-warmup",
         "--no-context-shift", "--remote-attn", "vulkan:0", "--remote-attn-layers", "8",
         "--remote-attn-cache-type-k", "q4_0", "--remote-attn-cache-type-v", "q4_0",
         "--spec-type", "none"],
        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    for _ in range(300):
        if http(port, "/health", timeout=2)[0] == 200:
            return proc, log
        if proc.poll() is not None:
            raise RuntimeError("%s exited at startup (see %s/%s.log)" % (tag, work, tag))
        time.sleep(1)
    raise RuntimeError("%s never healthy" % tag)


def kill(proc, log):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:  # noqa: BLE001
        proc.terminate()
    try:
        proc.wait(timeout=25)
    except Exception:  # noqa: BLE001
        pass
    log.close()


def topn(port, ids, n_probs):
    """Return {token_id: prob} plus how many candidates the server returned."""
    st, body = http(port, "/completion",
                    {"prompt": ids, "n_predict": 1, "temperature": 0.0,
                     "cache_prompt": True, "seed": 42, "n_probs": n_probs})
    if st != 200:
        raise RuntimeError("completion failed: %s" % body)
    # /completion reports the sampled token with "logprob" and the top-N
    # candidates under "top_logprobs" (server-task.cpp probs_vector_to_json).
    out = {}
    total = 0
    for d in body.get("completion_probabilities") or []:
        total += 1
        entries = []
        if d.get("top_logprobs"):
            entries = d["top_logprobs"]
        elif d.get("logprob") is not None:
            entries = [d]
        for e in entries:
            lp = e.get("logprob")
            if lp is not None:
                out[e["id"]] = math.exp(lp)   # FP64
    return out, total


def kl_and_top1(base, cand):
    keys = set(base) | set(cand)
    zb = sum(base.values())
    zc = sum(cand.values())
    if zb <= 0.0 or zc <= 0.0:
        return None, None, None
    kl = 0.0
    for t in keys:
        pb = base.get(t, 0.0) / zb
        pc = cand.get(t, 0.0) / zc
        if pb > 0.0 and pc > 0.0:
            kl += pb * math.log(pb / pc)
    tb = max(base, key=base.get)
    tc = max(cand, key=cand.get)
    return kl, (tb, tc), len(keys)


def main():
    P, C, n_tok, n_probs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    port = int(sys.argv[5]) if len(sys.argv) > 5 else 18501
    work = "/tmp/f4-kl-%d-%d" % (P, n_tok)
    subprocess.run(["rm", "-rf", work], check=False)

    proc, log = spawn(port, P, C, "candidate", work, n_probs)
    try:
        st, tok = http(port, "/tokenize",
                       {"content": " ".join("w%d" % i for i in range(n_tok * 2))})
        ids = tok["tokens"][:n_tok]
        # force a real crossing of P by continuing past the checkpoint
        cand, ncand = topn(port, ids, n_probs)
    finally:
        kill(proc, log)

    proc, log = spawn(port, 0, C, "baseline", work, n_probs)
    try:
        base, nbase = topn(port, ids, n_probs)
    finally:
        kill(proc, log)

    print("candidate P=%d C=%d: %d candidates returned" % (P, C, ncand))
    print("baseline  P=0      : %d candidates returned" % nbase)
    if len(base) < 2 or len(cand) < 2:
        print("KL NOT COMPUTED: server returned <2 candidates; increase n_probs")
        print("F3_KL_GATE=OPEN (no usable distribution)")
        return 2
    kl, tops, support = kl_and_top1(base, cand)
    print("support=%d  KL(base||cand) = %.6e nats" % (support, kl))
    print("argmax base=%d cand=%d  |dp|=%.3e" % (
        tops[0], tops[1], abs(base.get(tops[0], 0.0) - cand.get(tops[0], 0.0))))
    print("SCOPE: truncated top-%d support, renormalized. NOT the plan's "
          "full-vocabulary FP64 KL. F3_KL_GATE=OPEN." % max(nbase, ncand))
    return 0


if __name__ == "__main__":
    sys.exit(main())