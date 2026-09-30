#!/usr/bin/env python3
"""
bench-local-split-baseline.py — Fase 0: Profiling Baseline (RTX 4070 pura, IQ3_XXS)
Mede prefill tok/s, decode tok/s, VRAM pico e uso de memória na escada de contextos.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

def get_nvidia_vram_mb() -> float:
    """Retorna a VRAM ocupada na NVIDIA RTX 4070 em MiB."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return float(res.stdout.strip().split("\n")[0])
    except Exception:
        return 0.0

def run_llama_bench(
    bin_path: str,
    model_path: str,
    n_prompt: int,
    n_gen: int,
    n_gpu_layers: int = 99,
    cache_type: str = "kvarn4",
    threads: int = 8
) -> Dict[str, float]:
    """Executa o llama-bench para um determinado tamanho de prompt e geração."""
    vram_before = get_nvidia_vram_mb()
    
    cmd = [
        bin_path,
        "-m", model_path,
        "-ngl", str(n_gpu_layers),
        "-ctk", cache_type,
        "-ctv", cache_type,
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-t", str(threads),
        "-o", "json"
    ]
    
    print(f"[*] Executando: {' '.join(cmd)}")
    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        t_total = time.time() - t0
        vram_after = get_nvidia_vram_mb()
        
        # Parse output
        output_str = res.stdout
        try:
            data = json.loads(output_str)
            # data is a list of benchmark runs
            pp_tps = 0.0
            tg_tps = 0.0
            for item in data:
                if item.get("n_prompt", 0) > 0 and item.get("n_gen", 0) == 0:
                    pp_tps = item.get("avg_ts", 0.0)
                elif item.get("n_gen", 0) > 0:
                    tg_tps = item.get("avg_ts", 0.0)
            return {
                "n_prompt": n_prompt,
                "n_gen": n_gen,
                "pp_tps": pp_tps,
                "tg_tps": tg_tps,
                "vram_mb": vram_after,
                "vram_delta_mb": vram_after - vram_before,
                "elapsed_s": t_total,
                "status": "OK"
            }
        except Exception:
            # Fallback parsing se não for JSON puro
            return {
                "n_prompt": n_prompt,
                "n_gen": n_gen,
                "raw_output": output_str,
                "vram_mb": vram_after,
                "elapsed_s": t_total,
                "status": "RAW"
            }
    except subprocess.CalledProcessError as e:
        print(f"[!] Erro ao rodar llama-bench: {e.stderr}", file=sys.stderr)
        return {
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "error": e.stderr,
            "status": "FAIL"
        }

def main():
    parser = argparse.ArgumentParser(description="Fase 0 Profiling Baseline")
    parser.add_argument("--bin", default="/home/hjotha/beellama.cpp/build-optimized/bin/llama-bench", help="Caminho do binário llama-bench")
    parser.add_argument("--model", default="/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf", help="Caminho do modelo GGUF")
    parser.add_argument("--cache-type", default="kvarn4", help="Tipo de KV cache (kvarn4, f16, q4_0, etc.)")
    # Tiers de produção calibrados (router-production.ini):
    # S=24576, M=38912, L=56320, XL=72704, XXL=102400 (Teto 4070 pura), Split=131072, 204800
    parser.add_argument("--contexts", default="24576,38912,56320,72704,102400,131072", help="Lista de contextos separados por vírgula")
    parser.add_argument("--n-gen", type=int, default=128, help="Número de tokens gerados no decode")
    args = parser.parse_args()

    if not os.path.exists(args.bin):
        print(f"[!] Binário {args.bin} não encontrado. Aguarde o término da compilação.")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"[!] Modelo {args.model} não encontrado.")
        sys.exit(1)

    contexts = [int(c.strip()) for c in args.contexts.split(",") if c.strip()]
    results = []

    print("=" * 70)
    print("  BeeLLaMA — Fase 0: Profiling Baseline (4070 Pura)")
    print(f"  Modelo:     {args.model}")
    print(f"  KV Cache:   {args.cache_type}")
    print(f"  Contextos:  {contexts}")
    print(f"  Decode gen: {args.n_gen} tokens")
    print("=" * 70)

    for ctx in contexts:
        print(f"\n---> Testando Contexto {ctx} tokens...")
        r = run_llama_bench(
            bin_path=args.bin,
            model_path=args.model,
            n_prompt=ctx,
            n_gen=args.n_gen,
            cache_type=args.cache_type
        )
        results.append(r)
        time.sleep(2)  # cooldown térmico

    print("\n\n" + "=" * 70)
    print("  RESULTADOS — TABELA BASELINE (FASE 0)")
    print("=" * 70)
    print("| Contexto (tokens) | Prefill (tok/s) | Decode (tok/s) | VRAM Peak (MiB) | Status |")
    print("|---|---|---|---|---|")
    for r in results:
        ctx = r.get("n_prompt", 0)
        pp = r.get("pp_tps", 0.0)
        tg = r.get("tg_tps", 0.0)
        vram = r.get("vram_mb", 0.0)
        st = r.get("status", "UNKNOWN")
        print(f"| {ctx:<17} | {pp:<15.2f} | {tg:<14.2f} | {vram:<15.1f} | {st:<6} |")

if __name__ == "__main__":
    main()
