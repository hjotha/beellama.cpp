#!/usr/bin/env python3
"""Run the SnapKV strict/streaming matrix with an exact forced-output gate."""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


MODEL = os.environ.get("MODEL", "/home/hjotha/models/Qwen3.8-27B-UD-IQ3_XXS.gguf")
BIN = os.environ["LLAMA_SERVER"]
PORT = int(os.environ.get("PORT", "8090"))
POOL_BLOCKS = int(os.environ.get("POOL_BLOCKS", "2560"))
PAGE_SIZE = int(os.environ.get("PAGE_SIZE", "16"))
SNAPKV_BUDGET_BLOCKS = int(os.environ.get("SNAPKV_BUDGET_BLOCKS", "1536"))
OBSERVATION = int(os.environ.get("SNAPKV_OBSERVATION", "1024"))
MTP_NMAX = int(os.environ.get("MTP_NMAX", "2"))
OUTPUT_MAX = int(os.environ.get("OUTPUT_MAX", "4096"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
OUT = os.environ.get("OUT", "/home/hjotha/snapkv-40k-mtp-results.json")
LOG_DIR = os.environ.get("LOG_DIR", "/home/hjotha/snapkv-40k-mtp-logs")
CONTEXTS = [int(value) for value in os.environ.get(
    "CONTEXTS", "32768,36864,40960,49152,57344,65536").split(",") if value]

FACTS = [
    "FATO ALFA: a chave do arquivo antigo e ORION-741.",
    "FATO BETA: a funcao normalize_ledger usa modulo 97.",
    "FATO GAMA: o apelido do servico e delta-coral.",
]
TASK = "\n\nTAREFA: responda somente com os tres fatos ALFA, BETA e GAMA."
FILLER_UNIT = "O arquivo de referencia permanece consistente e cada paragrafo confirma o contexto anterior. "


def request(url, data=None, timeout=30):
    body = None if data is None else json.dumps(data).encode()
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, body, headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def api(path, data=None, timeout=30):
    return request(f"http://127.0.0.1:{PORT}{path}", data, timeout)


def tokenize(content):
    return api("/tokenize", {"content": content, "add_special": False}, 60)["tokens"]


def detokenize(tokens):
    return api("/detokenize", {"tokens": tokens}, 60)["content"]


def repeat_tokens(tokens, count):
    if count <= 0:
        return []
    return [tokens[index % len(tokens)] for index in range(count)]


def exact_prompt(token_count):
    filler = tokenize(FILLER_UNIT * 256)
    fact_tokens = [tokenize(fact) for fact in FACTS]
    task_tokens = tokenize(TASK)
    fixed = sum(len(tokens) for tokens in fact_tokens) + len(task_tokens)
    if token_count <= fixed + 32:
        raise ValueError(f"context {token_count} is too small for benchmark anchors ({fixed} fixed tokens)")

    filler_count = token_count - fixed
    for _ in range(8):
        first = filler_count // 6
        second = filler_count // 3
        third = filler_count // 3
        fourth = filler_count - first - second - third
        ids = (
            repeat_tokens(filler, first) + fact_tokens[0] +
            repeat_tokens(filler, second) + fact_tokens[1] +
            repeat_tokens(filler, third) + fact_tokens[2] +
            repeat_tokens(filler, fourth) + task_tokens
        )
        content = detokenize(ids)
        actual = len(tokenize(content))
        if actual == token_count:
            return content
        filler_count += token_count - actual
        if filler_count <= 0:
            break
    raise RuntimeError(f"exact prompt round-trip failed: requested={token_count} actual={actual}")


def wait_health():
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            if api("/health", timeout=3).get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("server did not become healthy")


def gpu_memory():
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip().splitlines()[0]
        return int(value)
    except Exception:
        return None


def parse_log(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    def numbers(name):
        return [float(value) for value in re.findall(rf"SNAPKVDBG {name}=([0-9.]+)", text)]
    mode = re.findall(r"SNAPKVDBG mode=(strict|streaming) expected_prefill_end=([0-9-]+) physical_pool_blocks=(\d+)", text)
    ended = re.findall(
        r"SNAPKVDBG prefill_ended seq=(-?\d+) ctx_len=(\d+) pages_before=(\d+) pages_after=(\d+)", text
    )
    mass = re.findall(
        r"SNAPKVDBG mass ([0-9.]+) pct=([0-9.]+) pages_kept=(\d+) total_pages=(\d+) mode=(\w+) target=(\d+) budget_old=(\d+)",
        text,
    )
    return {
        "mode": mode[-1][0] if mode else (mass[-1][4] if mass else None),
        "expected_prefill_end": int(mode[-1][1]) if mode else None,
        "physical_pool_blocks": int(mode[-1][2]) if mode else None,
        "prefill_ended": ended[-1] if ended else None,
        "mass": mass[-1] if mass else None,
        "score_sync_ms": numbers("score_sync_ms"),
        "pool_ms": numbers("pool_ms"),
        "select_evict_ms": numbers("select_evict_ms"),
        "progressive_evict_ms": numbers("progressive_evict_ms"),
        "evicted": [int(value) for value in re.findall(r"SNAPKVDBG evicted (\d+)", text)],
        "progressive_evictions": [int(value) for value in re.findall(r"SNAPKVDBG progressive_evicted (\d+)", text)],
        "stream_switches": re.findall(r"switched strict -> streaming at (\d+) logical blocks \(pool=(\d+)\)", text),
    }


def start_server(ctx, log_path):
    server_ctx = ctx + (2 if MTP_NMAX > 0 else 0)
    command = [
        BIN, "--model", MODEL, "--host", "127.0.0.1", "--port", str(PORT),
        "--ctx-size", str(server_ctx), "--batch-size", str(BATCH_SIZE), "--ubatch-size", str(BATCH_SIZE), "--parallel", "1",
        "--n-gpu-layers", "999", "--device", "CUDA0", "--flash-attn", "on",
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--kv-paged", "--kv-paged-dynamic",
        "--n-gpu-blocks", str(POOL_BLOCKS), "--n-gpu-blocks-initial", "64",
        "--n-gpu-blocks-growth", "64", "--n-cpu-blocks", "16384", "--kv-block-size", str(PAGE_SIZE),
        "--fit", "off", "--fit-target", "643", "--cache-ram", "0", "--metrics", "--no-warmup",
        "--snapkv", str(OBSERVATION), "--snapkv-recent", str(OBSERVATION), "--snapkv-pinned", "1024",
        "--snapkv-retention", "1.0", "--snapkv-budget-blocks", str(SNAPKV_BUDGET_BLOCKS),
    ]
    if MTP_NMAX > 0:
        command += [
            "--spec-type", "draft-mtp", "--spec-draft-n-max", str(MTP_NMAX), "--spec-draft-p-min", "0.80",
            "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0",
        ]
    log = open(log_path, "w", encoding="utf-8")
    env = dict(os.environ)
    env["LLAMA_SNAPKV_DEBUG"] = "1"
    process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True, env=env)
    return process, log


def stop_server(process, log):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    log.close()


def run_case(ctx):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"ctx-{ctx}-mtp-{MTP_NMAX}.log")
    process, log = start_server(ctx, log_path)
    try:
        wait_health()
        prompt_tokens = ctx - OUTPUT_MAX
        source = exact_prompt(prompt_tokens)
        began = time.time()
        response = api("/completion", {
            "prompt": source,
            "n_predict": OUTPUT_MAX,
            "temperature": 0,
            "seed": 1,
            "ignore_eos": True,
            "cache_prompt": False,
            "stream": False,
        }, timeout=max(3600, OUTPUT_MAX * 10))
        wall = time.time() - began
        answer = response.get("content", "")
        timing = response.get("timings", {})
        predicted = response.get("tokens_predicted", timing.get("predicted_n"))
        result = {
            "context": ctx,
            "server_context": ctx + (2 if MTP_NMAX > 0 else 0),
            "prompt_requested": prompt_tokens,
            "output_requested": OUTPUT_MAX,
            "output_predicted": predicted,
            "tokens_evaluated": response.get("tokens_evaluated", timing.get("prompt_n")),
            "stopped_eos": response.get("stop_type") == "eos",
            "truncated": response.get("truncated"),
            "stop_type": response.get("stop_type"),
            "prefill_tps": timing.get("prompt_per_second"),
            "decode_tps": timing.get("predicted_per_second"),
            "ttft_ms": timing.get("prompt_ms"),
            "wall_s": wall,
            "vram_mib": gpu_memory(),
            "quality": all(marker in answer for marker in ("ORION-741", "97", "delta-coral")),
            "answer_preview": answer[:500],
            "snapkv": parse_log(log_path),
        }
        result["gate_pass"] = (
            result["output_predicted"] == OUTPUT_MAX and
            result["tokens_evaluated"] == prompt_tokens and
            result["quality"]
        )
        return result
    except Exception as error:
        return {"context": ctx, "error": repr(error), "snapkv": parse_log(log_path)}
    finally:
        stop_server(process, log)


def main():
    rows = []
    for context in CONTEXTS:
        print(f"CASE_START context={context} prompt={context - OUTPUT_MAX} output={OUTPUT_MAX}", flush=True)
        row = run_case(context)
        rows.append(row)
        with open(OUT, "w", encoding="utf-8") as result:
            json.dump(rows, result, indent=2, ensure_ascii=False)
        print(json.dumps({
            key: row.get(key) for key in (
                "context", "server_context", "prompt_requested", "output_predicted", "tokens_evaluated", "truncated", "stop_type",
                "prefill_tps", "decode_tps", "wall_s", "quality", "gate_pass", "snapkv",
            )
        }, ensure_ascii=False), flush=True)
    print(f"RESULT_FILE {OUT}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
