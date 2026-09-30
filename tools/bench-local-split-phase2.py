#!/usr/bin/env python3
"""
bench-local-split-phase2.py — Fase 2: 1–2 Remote Layers Benchmark & Scaling
Mede TPS de prefill, TPS de decode, VRAM e métricas de latência por boundary
(avg_boundary_us, avg_cuda_to_host_us, avg_vulkan_attn_us, avg_host_to_cuda_us).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, Any, List

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

def run_local_split_test(
    cli_path: str,
    model_path: str,
    n_remote_layers: int,
    n_ctx: int = 4096,
    n_gen: int = 64,
    prompt: str = "Explain how the human eye perceives colors in two concise sentences."
) -> Dict[str, Any]:
    """Executa o llama-cli com a configuração especificada e extrai TPS e métricas."""
    vram_before = get_nvidia_vram_mb()

    cmd = [
        cli_path,
        "-m", model_path,
        "-ngl", "99",
        "-c", str(n_ctx),
        "-b", "256",
        "-ub", "256",
        "--cache-type-k", "kvarn4",
        "--cache-type-v", "kvarn4",
        "-p", prompt,
        "-n", str(n_gen),
        "--no-warmup"
    ]

    if n_remote_layers > 0:
        cmd.extend([
            "--remote-attn", "vulkan:0",
            "--remote-attn-layers", str(n_remote_layers),
            "--remote-attn-stats"
        ])

    print(f"\n[*] Executando teste ({n_remote_layers} remote layers): {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Envia /exit após receber a resposta para não prender a CLI interativa
        stdout, stderr = proc.communicate(input="/exit\n", timeout=90)
        t_total = time.time() - t0
        vram_after = get_nvidia_vram_mb()

        full_output = stdout + "\n" + stderr

        # Extrai Prompt t/s e Generation t/s
        # Formato: [ Prompt: 178,2 t/s | Generation: 32,7 t/s ] ou [ Prompt: 178.2 t/s | Generation: 32.7 t/s ]
        m_tps = re.search(r"\[\s*Prompt:\s*([\d,\.]+)\s*t/s\s*\|\s*Generation:\s*([\d,\.]+)\s*t/s\s*\]", full_output)
        pp_tps = 0.0
        gen_tps = 0.0
        if m_tps:
            pp_tps = float(m_tps.group(1).replace(",", "."))
            gen_tps = float(m_tps.group(2).replace(",", "."))

        # Extrai remote_attn stats JSON se houver
        stats_json = {}
        m_stats = re.search(r"remote_attn stats (\{.*?\})", full_output)
        if m_stats:
            try:
                stats_json = json.loads(m_stats.group(1))
            except Exception:
                pass

        return {
            "remote_layers": n_remote_layers,
            "pp_tps": pp_tps,
            "gen_tps": gen_tps,
            "stats": stats_json,
            "vram_mb": vram_after,
            "vram_delta_mb": vram_after - vram_before,
            "elapsed_s": t_total,
            "status": "OK"
        }
    except Exception as e:
        print(f"[!] Erro durante execução: {e}", file=sys.stderr)
        return {
            "remote_layers": n_remote_layers,
            "error": str(e),
            "status": "FAIL"
        }

def main():
    parser = argparse.ArgumentParser(description="Fase 2 Local-Split Benchmark")
    parser.add_argument("--cli", default="/home/hjotha/beellama.cpp/build-optimized/bin/llama-cli")
    parser.add_argument("--model", default="/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--gen", type=int, default=64)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--output", default="phase2_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("  BeeLLaMA — Fase 2: Local-Split Attention Scaling Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Context: {args.ctx} | Gen: {args.gen} tokens")
    print(f"  Layers to test: {args.layers}")
    print("=" * 70)

    results = []
    baseline_gen_tps = None

    for n_layers in args.layers:
        res = run_local_split_test(
            cli_path=args.cli,
            model_path=args.model,
            n_remote_layers=n_layers,
            n_ctx=args.ctx,
            n_gen=args.gen
        )
        if res.get("status") == "OK":
            if n_layers == 0:
                baseline_gen_tps = res["gen_tps"]
            
            gen_tps = res["gen_tps"]
            pp_tps = res["pp_tps"]
            stats = res.get("stats", {})
            avg_boundary = stats.get("avg_boundary_us", 0.0)
            avg_c2h = stats.get("avg_cuda_to_host_us", 0.0)
            avg_vk = stats.get("avg_vulkan_attn_us", 0.0)
            avg_h2c = stats.get("avg_host_to_cuda_us", 0.0)

            overhead_pct = 0.0
            if baseline_gen_tps and baseline_gen_tps > 0:
                overhead_pct = ((baseline_gen_tps - gen_tps) / baseline_gen_tps) * 100.0

            print(f"--> [Layers={n_layers:2d}] Decode: {gen_tps:5.2f} tok/s (diff: {overhead_pct:+5.1f}%) | "
                  f"Prefill: {pp_tps:6.1f} tok/s | Boundary: {avg_boundary:6.1f} µs "
                  f"(C2H: {avg_c2h:.1f}µs, VK: {avg_vk:.1f}µs, H2C: {avg_h2c:.1f}µs) | "
                  f"VRAM: {res['vram_mb']:.0f} MiB")
        results.append(res)
        time.sleep(1)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Resultados salvos em: {args.output}")

if __name__ == "__main__":
    main()
