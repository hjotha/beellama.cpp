#!/usr/bin/env python3
"""Benchmark tool comparing local CPU/CUDA vs Xbox Series X RKVA remote attention."""
import argparse
import json
import os
import re
import subprocess
import sys
import time

PROMPT_DEFAULT = (
    "A inteligência artificial transformou profundamente diversos setores da sociedade moderna, "
    "desde o diagnóstico médico assistido até a automação industrial e os veículos autônomos. "
    "Explique detalhadamente como funcionam os modelos de linguagem autorregressivos baseados na arquitetura "
    "Transformer, focando especialmente no mecanismo de atenção de produto escalar com escala (scaled dot-product attention), "
    "no agrupamento de consultas (grouped-query attention ou GQA), nas matrizes de projeção Q, K e V, e nas funções "
    "de ativação dos blocos de feed-forward como SwiGLU. Compare o custo computacional de prefill e decode em termos de complexidade "
    "de tempo e memória, destacando o impacto do crescimento do cache de chaves e valores (KV cache) à medida que o contexto aumenta "
    "para milhares ou dezenas de milhares de tokens."
)

def run_trial(llama_cli, model_path, name, extra_args, prompt, n_predict=16, ctx=2048, threads=4):
    env = dict(os.environ)
    cli_dir = os.path.dirname(os.path.abspath(llama_cli))
    env["LD_LIBRARY_PATH"] = cli_dir + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        llama_cli,
        "-m", model_path,
        "-c", str(ctx),
        "-p", prompt,
        "-n", str(n_predict),
        "--threads", str(threads),
        "--simple-io",
        "--single-turn",
        "--no-warmup",
        "-v"
    ] + extra_args

    print(f"\n==========================================")
    print(f"Running {name}...")
    print(f"Command: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace", stdin=subprocess.DEVNULL)
    elapsed = time.time() - t0
    print(f"Finished in {elapsed:.2f}s (exit code {res.returncode})")

    out = res.stdout + "\n" + res.stderr

    speed_match = re.search(r"\[\s*Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s\s*\]", out)
    prompt_ts = float(speed_match.group(1)) if speed_match else None
    gen_ts = float(speed_match.group(2)) if speed_match else None

    prompt_perf = re.search(r"prompt eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d\.]+)\s*ms per token,\s*([\d\.]+)\s*tokens per second\)", out)
    eval_perf = re.search(r"eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)\s*\(\s*([\d\.]+)\s*ms per token,\s*([\d\.]+)\s*tokens per second\)", out)

    # Use the last remote_attn stats occurrence (after probe/warmup)
    remote_stats_all = re.findall(r"remote_attn stats (\{.*?\})", out)
    stats_json = json.loads(remote_stats_all[-1]) if remote_stats_all else None

    result = {
        "name": name,
        "exit_code": res.returncode,
        "elapsed_sec": round(elapsed, 2),
        "prompt_ts": prompt_ts,
        "gen_ts": gen_ts,
        "stats": stats_json
    }
    if prompt_perf:
        result["prompt_ms"] = float(prompt_perf.group(1))
        result["prompt_tokens"] = int(prompt_perf.group(2))
        result["prompt_tokens_per_sec"] = float(prompt_perf.group(4))
    if eval_perf:
        result["eval_ms"] = float(eval_perf.group(1))
        result["eval_tokens"] = int(eval_perf.group(2))
        result["eval_tokens_per_sec"] = float(eval_perf.group(4))

    print("Result:", json.dumps(result, indent=2))
    return result

def main():
    parser = argparse.ArgumentParser(description="Benchmark Xbox RKVA remote attention")
    parser.add_argument("--cli", default="/home/hjotha/beellama.cpp/build-optiplex/bin/llama-cli", help="Path to llama-cli")
    parser.add_argument("--model", default="/home/hjotha/artifacts/models/Qwen3.5-4B-Q4_K_M.gguf", help="Path to GGUF model")
    parser.add_argument("--xbox", default="192.168.1.26:50054", help="Xbox host:port")
    parser.add_argument("--predict", type=int, default=16, help="Tokens to generate")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--ctx", type=int, default=2048, help="Context size")
    parser.add_argument("--output", default="bench_xbox_rkva_results.json", help="Output JSON results")
    args = parser.parse_args()

    results = []

    # 1. Baseline
    results.append(run_trial(args.cli, args.model, "Baseline Local (0 remote layers)", [], PROMPT_DEFAULT, args.predict, args.ctx, args.threads))

    # 2. 8 Layers
    results.append(run_trial(args.cli, args.model, "Remote Xbox RKVA (8 layers)", [
        "--remote-attn", args.xbox,
        "--remote-attn-layers", "8",
        "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--remote-attn-stats"
    ], PROMPT_DEFAULT, args.predict, args.ctx, args.threads))

    # 3. 16 Layers
    results.append(run_trial(args.cli, args.model, "Remote Xbox RKVA (16 layers)", [
        "--remote-attn", args.xbox,
        "--remote-attn-layers", "16",
        "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--remote-attn-stats"
    ], PROMPT_DEFAULT, args.predict, args.ctx, args.threads))

    # 4. 32 Layers
    results.append(run_trial(args.cli, args.model, "Remote Xbox RKVA (32 layers)", [
        "--remote-attn", args.xbox,
        "--remote-attn-layers", "32",
        "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--remote-attn-stats"
    ], PROMPT_DEFAULT, args.predict, args.ctx, args.threads))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll benchmark trials finished. Results saved to {args.output}")

if __name__ == "__main__":
    main()
