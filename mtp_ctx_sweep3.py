#!/usr/bin/env python3
"""v3: same measurement as mtp_ctx_sweep.py, with the host-OOM contamination fixed.

Run 2 produced false ceilings: the box has 15 GiB of RAM and the model is a 10.4 GiB
mmap, so a fresh load on top of a still-warm page cache gets SIGKILLed by the kernel
OOM killer (rc=-9) instead of failing on VRAM. Fixes here:
  - drop_caches + wait for MemAvailable before every start
  - rc=-9 with no CUDA OOM in the log is classified as host_oom and RETRIED, never
    counted as a ceiling
Phase 1 refines only the ceilings that run 2 could not trust.
Phase 2 runs one identical request on all four configs at a common ctx, which is the
actual b/ub 64 vs 512 comparison (wall time / prefill tps / decode tps).
"""
import json, os, subprocess, time, urllib.error, urllib.request

MODEL = "/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
BIN   = "/home/hjotha/src/llama.cpp/build/bin/llama-server"
PORT  = 8095
URL   = "http://127.0.0.1:%d" % PORT
LOG   = "/home/hjotha/mtp-ctx-sweep-server.log"
OUT   = "/home/hjotha/mtp-ctx-sweep3-results.json"
PAD   = 256
SUDO  = "amdk6x"
ENV   = dict(os.environ, LD_LIBRARY_PATH="/home/hjotha/src/llama.cpp/build/bin")
RAM_TARGET = 11500   # MiB of MemAvailable required before spawning a child

# phase 1: only what run 2 could not measure cleanly.
# gate_lo/gate_hi are known-PASS / known-FAIL ctx values, so the bisect resumes.
CEILINGS = [
    dict(name="mtp_off_b64",  mtp=False, b=64,  ub=64,
         startup=105728, gate_lo=75520, gate_hi=105728),
    dict(name="mtp_on_b64",   mtp=True,  b=64,  ub=64,
         load_lo=59392, load_hi=65536, gate_lo=59392),
    dict(name="mtp_off_b512", mtp=False, b=512, ub=512,
         load_lo=25600, load_hi=105728),
]

# phase 2: identical request on every config at a ctx that all four can load
CMP_CTX     = 40960
CMP_PROMPT  = 32768
CMP_PREDICT = 1024
COMPARE = [("mtp_off_b64",  False,  64,  64),
           ("mtp_on_b64",   True,   64,  64),
           ("mtp_off_b512", False, 512, 512),
           ("mtp_on_b512",  True,  512, 512)]


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def http(path, data=None, timeout=1800):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(URL + path, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode(errors="replace"))
        except Exception:
            return e.code, {"raw": str(e)[:300]}
    except Exception as e:
        return 0, {"raw": "%s: %s" % (type(e).__name__, str(e)[:200])}


def sh(cmd):
    return subprocess.run(["bash", "-c", "echo %s | sudo -S -p '' %s" % (SUDO, cmd)],
                          capture_output=True, text=True)


def vram_used():
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
    except Exception:
        return 99999


def mem_avail():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def wait_for(fn, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if fn():
            return True
        time.sleep(3)
    return fn()


def settle(ram_target=RAM_TARGET):
    """free the GPU and, crucially, the page cache the previous child left behind"""
    subprocess.run(["pkill", "-9", "-f", "llama-server.*%d" % PORT], capture_output=True)
    time.sleep(2)
    wait_for(lambda: vram_used() <= 600, 180)
    sh("sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'")
    ok = wait_for(lambda: mem_avail() >= ram_target, 60)
    if not ok:
        log("   warn: MemAvailable only %d MiB" % mem_avail())
    return ok


def args_for(mtp, b, ub, ctx):
    a = [BIN, "--model", MODEL, "--alias", "sweep", "--host", "127.0.0.1", "--port", str(PORT),
         "--ctx-size", str(ctx), "--batch-size", str(b), "--ubatch-size", str(ub),
         "--parallel", "1", "--device", "CUDA0", "--flash-attn", "on",
         "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
         # cache-ram is a host-side prompt cache: irrelevant to the VRAM ceiling, and 3 GiB
         # of it on a 15 GiB box is what fed the OOM killer. 0 while measuring.
         "--fit", "off", "--cache-ram", "0", "--ctx-checkpoints", "1",
         "--gpu-power-prefill", "200", "--gpu-power-decode", "165", "--gpu-power-device", "0",
         "--gpu-mem-clock-decode", "11001",
         "--metrics", "--no-webui", "--log-file", LOG]
    if mtp:
        a += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-p-min", "0.80",
              "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0"]
    return a


def cuda_oom(tail):
    return ("out of memory" in tail) or ("cudaMalloc" in tail) or ("failed to allocate" in tail)


def start_once(mtp, b, ub, ctx, timeout=420):
    settle()
    open(LOG, "w").close()
    lf = open(LOG, "a")
    proc = subprocess.Popen(args_for(mtp, b, ub, ctx), stdout=lf, stderr=lf, env=ENV)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            tail = open(LOG).read()[-4000:]
            if cuda_oom(tail):
                return None, "cuda_oom"
            if proc.returncode == -9:
                return None, "host_oom"
            return None, "died rc=%s" % proc.returncode
        if http("/health", timeout=2)[1].get("status") == "ok":
            return proc, "ok"
        time.sleep(2)
    proc.kill()
    return None, "timeout"


def start(mtp, b, ub, ctx, tries=4):
    """retries host_oom, so only VRAM decides the answer"""
    for i in range(tries):
        proc, why = start_once(mtp, b, ub, ctx)
        if proc or why != "host_oom":
            return proc, why
        log("   host_oom on load ctx=%d, retry %d/%d (avail=%d MiB)"
            % (ctx, i + 1, tries - 1, mem_avail()))
        time.sleep(15)
    return None, "host_oom_persistent"


def stop(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=45)
        except Exception:
            proc.kill()


def exact_prompt(n):
    phrase = ("A calibracao de contexto exige medir o teto real de KV e nao apenas o candidato "
              "de startup, porque o limite util fica abaixo do limite anunciado. ")
    ids = http("/tokenize", {"content": phrase}, timeout=120)[1]["tokens"]
    return (ids * (n // len(ids) + 2))[:n]


def request_at(n_prompt, n_predict):
    t0 = time.time()
    status, r = http("/completion", {"prompt": exact_prompt(n_prompt), "n_predict": n_predict,
                                     "temperature": 0, "ignore_eos": True}, timeout=3600)
    wall = time.time() - t0
    t = r.get("timings", {}) if isinstance(r, dict) else {}
    return {"http": status, "wall_s": round(wall, 1), "prompt_tokens": n_prompt,
            "tokens_evaluated": r.get("tokens_evaluated") if isinstance(r, dict) else None,
            "tokens_predicted": r.get("tokens_predicted") if isinstance(r, dict) else None,
            "prefill_tps": round(t.get("prompt_per_second", 0), 2),
            "decode_tps": round(t.get("predicted_per_second", 0), 2),
            "prefill_s": round(t.get("prompt_ms", 0) / 1000.0, 1),
            "decode_s": round(t.get("predicted_ms", 0) / 1000.0, 1),
            "vram_mib": vram_used(),
            "error": None if status == 200 else str(r)[:300]}


def try_load(mtp, b, ub, ctx):
    proc, why = start(mtp, b, ub, ctx)
    stop(proc)
    settle()
    return proc is not None, why


def gate(mtp, b, ub, ctx, n_predict=1, tries=3):
    """full-length prefill at ctx. host_oom during the request is retried too."""
    for i in range(tries):
        proc, why = start(mtp, b, ub, ctx)
        if proc is None:
            if why in ("cuda_oom",):
                return False, {"stage": "load", "reason": why}
            return False, {"stage": "load", "reason": why, "inconclusive": True}
        info = request_at(ctx - 4096, n_predict)
        tail = open(LOG).read()
        info["server_oom"] = cuda_oom(tail)
        rc = proc.poll()
        info["server_died"] = rc is not None
        info["rc"] = rc
        stop(proc)
        settle()
        if rc == -9 and not info["server_oom"]:
            log("   host_oom during request at ctx=%d, retry %d/%d" % (ctx, i + 1, tries - 1))
            continue
        info["passed"] = info["http"] == 200 and not info["server_oom"] and not info["server_died"]
        return info["passed"], info
    return False, {"stage": "request", "reason": "host_oom_persistent", "inconclusive": True}


def ceilings(cfg):
    name, mtp, b, ub = cfg["name"], cfg["mtp"], cfg["b"], cfg["ub"]
    log("=== %s (mtp=%s b=%d ub=%d)" % (name, mtp, b, ub))
    res = {"name": name, "mtp": mtp, "batch": b, "ubatch": ub, "loads": {}, "gates": {}}

    startup = cfg.get("startup")
    if startup:
        log("  startup ceiling given = %d" % startup)
    else:
        lo, hi = cfg["load_lo"] // PAD, cfg["load_hi"] // PAD
        tried = {}

        def loads(blk):
            if blk not in tried:
                ok, why = try_load(mtp, b, ub, blk * PAD)
                tried[blk] = ok
                res["loads"][blk * PAD] = why
                log("  load ctx=%d -> %s (%s)" % (blk * PAD, "OK" if ok else "FAIL", why))
            return tried[blk]

        if not loads(lo):
            res["error"] = "floor %d does not load" % (lo * PAD)
            log("  !! " + res["error"])
            return res
        if loads(hi):
            lo = hi
        else:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if loads(mid):
                    lo = mid
                else:
                    hi = mid
        startup = lo * PAD
        log("  startup ceiling = %d" % startup)
    res["startup_ceiling"] = startup

    # usable ceiling: bisect between a known-good and a known-bad full-length prefill
    hi = min(startup, cfg.get("gate_hi", startup)) // PAD
    lo = cfg.get("gate_lo")
    if lo is None:
        ok, info = gate(mtp, b, ub, hi * PAD)
        res["gates"][hi * PAD] = info
        log("  gate ctx=%d -> %s" % (hi * PAD, "PASS" if ok else "FAIL"))
        if ok:
            res["usable_ceiling"] = hi * PAD
            log("  USABLE CEILING = %d (startup %d)" % (hi * PAD, startup))
            return res
        lo = max(30720, hi * PAD - 40960)
        for _ in range(3):
            ok, info = gate(mtp, b, ub, lo)
            res["gates"][lo] = info
            log("  gate floor ctx=%d -> %s" % (lo, "PASS" if ok else "FAIL"))
            if ok:
                break
            lo = max(20480, lo - 10240)
        else:
            res["error"] = "no gate floor"
            return res
    lo //= PAD
    if cfg.get("gate_hi") is None and hi == lo:
        res["usable_ceiling"] = lo * PAD
        log("  USABLE CEILING = %d (startup %d)" % (lo * PAD, startup))
        return res
    n = 0
    while hi - lo > 1 and n < 10:
        mid = (lo + hi) // 2
        ok, info = gate(mtp, b, ub, mid * PAD)
        res["gates"][mid * PAD] = info
        log("  gate ctx=%d -> %s (%s)" % (mid * PAD, "PASS" if ok else "FAIL",
                                          info.get("reason") or info.get("error") or "ok"))
        if ok:
            lo = mid
        else:
            hi = mid
        n += 1
    res["usable_ceiling"] = lo * PAD
    res["gate_resolution"] = (hi - lo) * PAD
    log("  USABLE CEILING = %d (startup %d, resolution %d)"
        % (lo * PAD, startup, (hi - lo) * PAD))

    # confirm with the real thing: full prompt + 4096 forced output
    proc, why = start(mtp, b, ub, lo * PAD)
    if proc:
        log("  confirm: %d + 4096 at ctx=%d" % (lo * PAD - 4096, lo * PAD))
        res["max_request"] = request_at(lo * PAD - 4096, 4096)
        res["max_request"]["server_oom"] = cuda_oom(open(LOG).read())
        log("  -> http=%s wall=%ss prefill=%s t/s decode=%s t/s"
            % (res["max_request"]["http"], res["max_request"]["wall_s"],
               res["max_request"]["prefill_tps"], res["max_request"]["decode_tps"]))
        stop(proc)
    else:
        res["confirm_error"] = why
    settle()
    return res


def compare(name, mtp, b, ub):
    log("=== compare %s at ctx=%d (%d + %d)" % (name, CMP_CTX, CMP_PROMPT, CMP_PREDICT))
    proc, why = start(mtp, b, ub, CMP_CTX)
    if not proc:
        log("  load failed: %s" % why)
        return {"name": name, "mtp": mtp, "batch": b, "ubatch": ub, "error": why}
    t_load = vram_used()
    r = request_at(CMP_PROMPT, CMP_PREDICT)
    r.update({"name": name, "mtp": mtp, "batch": b, "ubatch": ub, "ctx": CMP_CTX,
              "vram_after_load_mib": t_load})
    log("  -> http=%s wall=%ss prefill=%s t/s (%ss) decode=%s t/s (%ss) vram=%s MiB"
        % (r["http"], r["wall_s"], r["prefill_tps"], r["prefill_s"],
           r["decode_tps"], r["decode_s"], r["vram_mib"]))
    stop(proc)
    settle()
    return r


def prod(action):
    r = sh("systemctl %s llama-server-root" % action)
    log("prod %s rc=%s %s" % (action, r.returncode, r.stderr.strip()[:200]))


def main():
    out = {"started": time.strftime("%F %T"), "model": MODEL, "kv": "traditional q4_0",
           "fit": "off", "host_ram_mib": mem_avail(), "ceilings": [], "compare": []}
    prod("stop")
    settle()
    log("after prod stop: vram=%d MiB avail=%d MiB" % (vram_used(), mem_avail()))
    try:
        for cfg in CEILINGS:
            out["ceilings"].append(ceilings(cfg))
            json.dump(out, open(OUT, "w"), indent=2)
        for c in COMPARE:
            out["compare"].append(compare(*c))
            json.dump(out, open(OUT, "w"), indent=2)
    finally:
        settle()
        prod("start")
        ok = wait_for(lambda: http("/health", timeout=2)[0] == 200 or
                              subprocess.run(["curl", "-sf", "-m", "3",
                                              "http://127.0.0.1:8090/health"],
                                             capture_output=True).returncode == 0, 300)
        out["prod_restored"] = ok
        out["finished"] = time.strftime("%F %T")
        json.dump(out, open(OUT, "w"), indent=2)
        log("prod restored: %s" % ok)
    log("=== DONE ===")


if __name__ == "__main__":
    main()
