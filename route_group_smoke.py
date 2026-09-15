#!/usr/bin/env python3
"""Router route-group smoke test. Two profiles, both meant to run on the GPU box:

  tiny  4B model, absurdly small caps -> exercises every routing *decision* cheaply
  real  the actual deployment: one -mtp GGUF, qwen3-fast (60,928, MTP) and qwen3-long
        (105,472), public name qwen-3.8-27b capped at 56,000 -> exercises the contract

Assertions read the router's own "route group ..." log lines, which name the child that was
picked, plus the response bodies (the "model" field must never leak the tier).

usage: route_group_smoke.py [tiny|real]
"""
import json, os, re, subprocess, sys, time, urllib.error, urllib.request

HOME  = os.path.expanduser("~")
BIN   = HOME + "/src/llama-mtp-ctx/build/bin/llama-server"
PORT  = 8096
URL   = "http://127.0.0.1:%d" % PORT
INI   = "/tmp/route-group-smoke.ini"
LOG   = "/tmp/route-group-smoke.log"
RESERVE    = 4096   # router's default output reserve when a request omits max_tokens
MAX_TOKENS = 32     # every request asks for a tiny answer; only the *budget* is under test

COMMON_INI = """version = 1

[*]
model = %(model)s
parallel = 1
device = CUDA0
flash-attn = on
cache-type-k = q4_0
cache-type-v = q4_0
fit = off
ctx-checkpoints = 1
no-webui = true
metrics = true
"""

PROFILES = {
    "tiny": dict(
        group="testmodel", fast="tiny-fast", long="tiny-long",
        cap=300, fast_ctx=2048, long_ctx=8192, load_timeout=120, client_ctx=8192,
        # prompt sizes in tokens. under/over straddle the cap once MAX_TOKENS is added; big does
        # not fit the fast tier's context at all
        small=32, big=2000, under=200, over=290,
        ini=(COMMON_INI + """batch-size = 512
ubatch-size = 512
cache-ram = 0

[tiny-fast]
ctx-size = 2048
route-group = testmodel
route-max-tokens = 300

[tiny-long]
ctx-size = 8192
route-group = testmodel
""") % dict(model=HOME + "/models/Qwen3.5-4B-Q4_K_M.gguf"),
    ),
    "real": dict(
        group="qwen-3.8-27b", fast="qwen3-fast", long="qwen3-long",
        cap=56000, fast_ctx=60928, long_ctx=105472, load_timeout=900, client_ctx=80000,
        # under/over straddle the 56,000 cap with ~2k of slack on each side: the tokenizer lands
        # within ~0.5% of the aim, and both still fit the fast tier's 60,928 context, so what is
        # under test is the cap, not the ceiling
        small=512, big=70000, under=50000, over=57500,
        ini=(COMMON_INI + """batch-size = 64
ubatch-size = 64
cache-ram = 0
gpu-power-prefill = 200
gpu-power-decode = 165
gpu-power-device = 0
gpu-mem-clock-decode = 11001

[qwen3-fast]
ctx-size = 60928
spec-type = draft-mtp
spec-draft-n-max = 2
spec-draft-p-min = 0.80
spec-draft-type-k = q4_0
spec-draft-type-v = q4_0
route-group = qwen-3.8-27b
route-max-tokens = 56000

[qwen3-long]
ctx-size = 105472
route-group = qwen-3.8-27b
""") % dict(model=HOME + "/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"),
    ),
}


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def http(path, data=None, timeout=3600, headers=None):
    body = None if data is None else json.dumps(data).encode()
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(URL + path, body, h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode(errors="replace"))
        except Exception:
            return e.code, {"raw": str(e)[:200]}
    except Exception as e:
        return 0, {"raw": "%s: %s" % (type(e).__name__, str(e)[:200])}


def log_routes():
    """every routing decision the router logged so far, as (prompt, output, target)"""
    out = []
    for line in open(LOG, errors="replace"):
        m = re.search(r"route group '\S+': (\d+) prompt \+ (\d+) output = \d+ tokens -> (\S+)", line)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    return out


class Runner:
    def __init__(self, prof):
        self.p = prof
        self.words_per_token = 34   # rough, until calibrate() measures it on the real tokenizer

    def calibrate(self):
        """tokens per repetition of the filler phrase, measured on the child itself. measured over
        100 repetitions, not one: BPE merges across the repetition boundary, so a single phrase
        overestimates by ~3% and a 56k prompt would land 1,700 tokens short of where it was aimed"""
        reps = 100
        st, out = http("/tokenize", {"model": self.p["group"], "content": PHRASE * reps},
                       timeout=self.p["load_timeout"])
        if st != 200:
            sys.exit("tokenize failed: %s %s" % (st, out))
        self.words_per_token = len(out["tokens"]) / reps
        log("filler phrase = %.2f tokens" % self.words_per_token)

    def prompt_of(self, n_tokens):
        return PHRASE * max(1, round(n_tokens / self.words_per_token))

    def worst(self):
        """the largest request a client can produce with the advertised contract: ctx 80,000 minus
        the 4,096 output it also asks for, decoded to the last token (ignore_eos). this is the one
        that has to survive the router's arg rendering, not just load"""
        n = self.p["client_ctx"] - RESERVE
        body = {"model": self.p["group"], "temperature": 0, "max_tokens": RESERVE,
                "ignore_eos": True, "messages": [{"role": "user", "content": self.prompt_of(n)}]}
        t0 = time.time()
        st, out = http("/v1/chat/completions", body)
        p, o, tgt = log_routes()[-1]
        t = out.get("timings", {}) if st == 200 else {}
        log("  worst case: %d + %d -> %s http=%s in %.0fs (prefill %.1f t/s, decode %.1f t/s)"
            % (p, o, tgt, st, time.time() - t0, t.get("prompt_per_second", 0),
               t.get("predicted_per_second", 0)))
        return st, out, (p, o, tgt)

    def chat(self, n_tokens, max_tokens=MAX_TOKENS, conv=None, path="/v1/chat/completions"):
        before = len(log_routes())
        # "." stops the generation almost immediately: the routing decision is made before a
        # single token is produced, so no test needs to wait for a real 4096-token answer
        body = {"model": self.p["group"], "temperature": 0, "stop": ["."],
                "messages": [{"role": "user", "content": self.prompt_of(n_tokens)}]}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        headers = {"X-Conversation-Id": conv} if conv else None
        t0 = time.time()
        st, out = http(path, body, headers=headers)
        routes = log_routes()
        assert len(routes) > before, "router logged no routing decision"
        p, o, tgt = routes[-1]
        log("  ask ~%d tok, max_tokens=%s -> %s (counted %d + %d) http=%s in %.0fs"
            % (n_tokens, max_tokens, tgt, p, o, st, time.time() - t0))
        return st, out, (p, o, tgt)


PHRASE = ("A calibracao de contexto exige medir o teto real de KV e nao apenas o candidato "
          "de startup, porque o limite util fica abaixo do limite anunciado. ")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    prof = PROFILES[name]
    r = Runner(prof)
    open(INI, "w").write(prof["ini"])
    subprocess.run(["pkill", "-9", "-f", "llama-server.*%d" % PORT], capture_output=True)
    time.sleep(1)
    open(LOG, "w").close()
    lf = open(LOG, "a")
    proc = subprocess.Popen([BIN, "--models-preset", INI, "--models-max", "1",
                             "--host", "127.0.0.1", "--port", str(PORT), "--no-webui"],
                            stdout=lf, stderr=lf,
                            env=dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(BIN)))
    fails = []

    def check(label, cond, detail=""):
        log("%-52s %s %s" % (label, "PASS" if cond else "FAIL", detail))
        if not cond:
            fails.append(label)

    try:
        for _ in range(60):
            if http("/health", timeout=3)[0] == 200:
                break
            if proc.poll() is not None:
                print(open(LOG).read()[-3000:])
                sys.exit("router died rc=%s" % proc.returncode)
            time.sleep(1)
        else:
            sys.exit("router never became healthy")
        log("router up, profile=%s" % name)

        # "worst" mode: skip the decision matrix, spend the whole run on the one request the
        # advertised contract allows a client to send. run it after the matrix passes
        if len(sys.argv) > 2 and sys.argv[2] == "worst":
            r.calibrate()
            st, out, (p, o, tgt) = r.worst()
            check("worst-case request survives the router", st == 200, (st, out.get("error")))
            check("... on the long tier, answering as the public name",
                  tgt == prof["long"] and out.get("model") == prof["group"],
                  (tgt, out.get("model")))
            # the full 4,096 unless the tier's own context runs out first (which the tiny profile
            # hits by design: its long ctx is exactly prompt + output)
            want = min(RESERVE, prof["long_ctx"] - p)
            check("... and decoded the full output",
                  o == RESERVE and out.get("usage", {}).get("completion_tokens") == want,
                  (want, out.get("usage", {}).get("completion_tokens")))
            log("%s (%d checks failed)" % ("FAILED" if fails else "ALL PASS", len(fails)))
            return 1 if fails else 0

        # 1. the group is advertised, its members are not, n_ctx is the widest member's ctx-size
        _, models = http("/v1/models")
        ids = [m["id"] for m in models["data"]]
        grp = next((m for m in models["data"] if m["id"] == prof["group"]), None)
        check("group advertised in /v1/models", grp is not None, ids)
        check("members hidden", prof["fast"] not in ids and prof["long"] not in ids, ids)
        if grp:
            check("group n_ctx = widest member ctx-size",
                  grp["meta"].get("n_ctx") == prof["long_ctx"], grp["meta"].get("n_ctx"))
            members = [(m["name"], m["max_tokens"]) for m in grp["meta"]["route_members"]]
            check("route_members ordered, capped tier first",
                  members == [(prof["fast"], prof["cap"]), (prof["long"], -1)], members)

        # 2. small request -> capped tier (cold: byte heuristic, so only the target is asserted)
        st, out, (p, o, tgt) = r.chat(prof["small"])
        check("small request -> fast tier", st == 200 and tgt == prof["fast"], (st, tgt))
        check("response does not leak the tier", out.get("model") == prof["group"], out.get("model"))
        r.calibrate()

        # 3. exact counting through the loaded child
        st, out, (p, o, tgt) = r.chat(prof["under"])
        check("prompt counted exactly (not bytes/3.5)",
              abs(p - prof["under"]) < max(64, prof["under"] * 0.05), (p, prof["under"]))
        check("under the cap stays on the fast tier",
              st == 200 and tgt == prof["fast"] and p + o <= prof["cap"], (st, p, o, tgt))

        # 4. just over the cap -> the uncapped tier
        st, out, (p, o, tgt) = r.chat(prof["over"])
        check("over the cap -> long tier",
              st == 200 and tgt == prof["long"] and p + o > prof["cap"], (st, p, o, tgt))

        # 5. the contract itself: a request larger than the fast tier's whole context
        st, out, (p, o, tgt) = r.chat(prof["big"])
        check("full-size request served by the long tier",
              st == 200 and tgt == prof["long"], (st, tgt))
        check("... and answers as the public name", out.get("model") == prof["group"],
              out.get("model"))
        if st == 200:
            t = out.get("timings", {})
            log("  long tier: prefill %.1f t/s, decode %.1f t/s"
                % (t.get("prompt_per_second", 0), t.get("predicted_per_second", 0)))

        # 6. omitted max_tokens is budgeted, not treated as 0
        st, out, (p, o, tgt) = r.chat(prof["small"], max_tokens=None)
        check("omitted max_tokens reserves the default", o == RESERVE, o)

        # 7. no-demotion: a conversation pinned to the wide tier stays there
        st, out, (_, _, up) = r.chat(prof["big"], conv="conv-A")
        st, out, (_, _, dn) = r.chat(prof["small"], conv="conv-A")
        check("no demotion after a wide turn", up == prof["long"] and dn == prof["long"], (up, dn))
        st, out, (_, _, other) = r.chat(prof["small"], conv="conv-B")
        check("other conversation still routes by size", other == prof["fast"], other)

        # 8. metadata routes must not pick a tier
        before = len(log_routes())
        st, out = http("/tokenize", {"model": prof["group"], "content": PHRASE * 3000})
        check("large /tokenize answered", st == 200 and len(out.get("tokens", [])) > 3000,
              (st, len(out.get("tokens", []))))
        check("/tokenize logged no size-based routing", len(log_routes()) == before)

        log("%s (%d checks failed)" % ("FAILED: " + ", ".join(fails) if fails else "ALL PASS",
                                       len(fails)))
        return 1 if fails else 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except Exception:
            proc.kill()
        subprocess.run(["pkill", "-9", "-f", "llama-server.*%d" % PORT], capture_output=True)


if __name__ == "__main__":
    sys.exit(main())
