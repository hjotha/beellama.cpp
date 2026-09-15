#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

MODEL = os.environ.get("MODEL", "/home/hjotha/models/Qwen3.8-27B-UD-IQ3_XXS.gguf")
BIN = os.environ["LLAMA_SERVER"]
CONTINUOUS_BIN = os.environ.get("CONTINUOUS_SERVER", BIN)
PORT = int(os.environ.get("PORT", "8092"))
OUT = os.environ.get("OUT", "/home/hjotha/snapkv_episode_results.json")
LOG = os.environ.get("LOG", "/home/hjotha/snapkv_episode.log")
URL = f"http://127.0.0.1:{PORT}"


def request(path, data=None, timeout=3600):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(URL + path, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def wait_health():
    for _ in range(90):
        try:
            if request("/health", timeout=2).get("status") == "ok":
                return
        except Exception:
            time.sleep(2)
    raise RuntimeError("server did not become healthy")


def vram():
    return int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())


def text_to_tokens(text):
    return len(request("/tokenize", {"content": text}, timeout=30)["tokens"])


def filler(tokens):
    unit = "O arquivo de referencia permanece consistente e cada paragrafo confirma o contexto anterior. "
    text = unit * max(1, tokens * 5 // len(unit))
    while text_to_tokens(text) < tokens:
        text += unit
    return text


def prompt(tokens, task):
    facts = [
        "FATO ALFA: a chave do arquivo antigo e ORION-741.",
        "FATO BETA: a funcao normalize_ledger usa modulo 97.",
        "FATO GAMA: o apelido do servico e delta-coral.",
    ]
    pieces = [filler(tokens // 6), facts[0], filler(tokens // 3), facts[1], filler(tokens // 3), facts[2], filler(tokens // 6)]
    return "\n".join(pieces) + "\n\n" + task, facts


def start(ctx, blocks, snap=None, binary=BIN):
    subprocess.run(["pkill", "-9", "-f", f"--port {PORT}"], capture_output=True)
    subprocess.run("echo amdk6x | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'",
                   shell=True, capture_output=True)
    time.sleep(4)
    cmd = [binary, "--model", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "--ctx-size", str(ctx),
           "--batch-size", "128", "--ubatch-size", "128", "--parallel", "1", "--n-gpu-layers", "999",
           "--device", "CUDA0", "--flash-attn", "on", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
           "--kv-paged", "--kv-paged-dynamic", "--n-gpu-blocks", str(blocks), "--n-gpu-blocks-initial", "64",
           "--n-cpu-blocks", "16384", "--kv-block-size", "16", "--fit", "off", "--cache-ram", "0", "--metrics"]
    if snap:
        cmd += ["--snapkv", str(snap["window"]), "--snapkv-recent", "8192", "--snapkv-pinned", "1024",
                "--snapkv-retention", str(snap["retention"]), "--snapkv-budget-blocks", str(blocks)]
    log = open(LOG, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    wait_health()
    return proc, log


def metrics():
    text = open(LOG).read()
    def values(name): return [float(v) for v in re.findall(rf"SNAPKVDBG {name}=([0-9.]+)", text)]
    return {
        "score_sync_ms": values("score_sync_ms"),
        "select_evict_ms": values("select_evict_ms"),
        "progressive_evict_ms": values("progressive_evict_ms"),
        "mass_pct": [float(v) for v in re.findall(r"SNAPKVDBG mass [0-9.]+ pct=([0-9.]+)", text)],
        "evicted": [int(v) for v in re.findall(r"SNAPKVDBG evicted (\d+)", text)],
    }


def run_case(name, ctx, blocks, snap, binary=BIN):
    proc, log = start(ctx, blocks, snap, binary)
    try:
        source, facts = prompt(int(ctx * .80), "Responda somente com os tres fatos ALFA, BETA e GAMA.")
        payload = {"messages": [{"role": "system", "content": "Responda exatamente ao pedido do usuario."},
                                {"role": "user", "content": source}], "temperature": 0, "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}}
        began = time.time()
        response = request("/v1/chat/completions", payload)
        wall = time.time() - began
        answer = response["choices"][0]["message"]["content"]
        timing = response.get("timings", {})
        return {"name": name, "ctx": ctx, "blocks": blocks, "snap": snap, "prefill_tps": timing.get("prompt_per_second"),
                "decode_tps": timing.get("predicted_per_second"), "ttft_ms": timing.get("prompt_ms"), "wall_s": wall,
                "vram_mib": vram(), "quality": all(k in answer for k in ("ORION-741", "97", "delta-coral")), "answer": answer[:500], **metrics()}
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        log.close()


def main():
    budget = 3072
    rows = []
    def safe(name, fn):
        try:
            rows.append(fn())
        except Exception as e:
            rows.append({"name": name, "error": repr(e)[:200]})
            print(f"CASE_FAILED {name}: {e!r}", flush=True)

    for ctx in (32768, 65536, 98304, 131072):
        safe(f"paged-{ctx // 1024}k", lambda ctx=ctx: run_case(f"paged-{ctx // 1024}k", ctx, ctx // 16, None))
        safe(f"continuous-{ctx // 1024}k", lambda ctx=ctx: run_case(f"continuous-{ctx // 1024}k", ctx, budget,
                             {"window": 16384, "retention": .75}, CONTINUOUS_BIN))
        for retention in (1.0, .75, .5):
            safe(f"episode-{ctx // 1024}k-r{retention}", lambda ctx=ctx, retention=retention: run_case(
                f"episode-{ctx // 1024}k-r{retention}", ctx, budget, {"window": 1024, "retention": retention}))
        with open(OUT, "w") as result:
            json.dump(rows, result, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
