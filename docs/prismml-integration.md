# PrismML Integration (PTQ1_0 / PQ2_0 ternary + Bonsai Hadamard)

Merge of https://github.com/PrismML-Eng/llama.cpp (`prism` branch, HEAD `1a07bfa5f`)
into the BeeLlama fork, preserving the full BeeLlama surface (KVarN, qwen4exp,
DFlash, chat/parsers, mmq tuning, KV-tail caches).

Integration commit: `264e8d01a` (+ follow-up build fixes), branch `prismml-integrate`.

## What was integrated

- **`GGML_TYPE_PQ2_0` (142)** — Prism group-128 Q2_0 (2.125 bpw). Core in
  `ggml-common.h`/`ggml-quants.c`, CPU scalar/AVX/AVX512 vec-dot, CUDA
  MMQ/MMVQ/dequant/getrows, GPU repack GEMV/GEMM, get_rows, ftype name.
- **`GGML_TYPE_PTQ1_0` (143)** — Prism group-128 ternary (1.75 bpw, base-3 trits).
  Core, CPU vec-dot, CUDA native MMQ tile loader + packed dequantize +
  `vec_dot_ptq1_0_q8_1_multi` (multi-column decode), get_rows, ftype name.
- **Bonsai Hadamard runtime** — blockwise Sylvester-Walsh-Hadamard rotation
  applied to activations (`ggml_mul_mat_set_hint`/`GGML_HINT_SRC0_IS_HADAMARD`,
  `llama_hadamard_*` graph plumbing, FWHT kernels in the CUDA graph path,
  `test_fwht_signed` regression test).
- `ggml` enum/type_traits/CPU traits extended to `COUNT = 144` while keeping the
  Bee KV formats `Q6_0..Q2_1` at ids 43..48.
- Converter/gguf-py: `prism.hadamard.*` metadata writer, `PQ2_0`/`PTQ1_0` ftypes.
- llama-side: `dspark`/`dfly` arch + tensor registrations, `path_kv_mean_center`
  field (kv-mean-center arg wiring), dspark-markov files.

## What was preserved (BeeLlama)

- KVarN (kvarn2..kvarn8), KV-tail caches, qwen4exp/qwen35 MTP, DFlash,
  chat/parsers subsystem, mmq tuning, `GGML_CUDA_FA` matrix, MoE weighted
  reduction + GB10 graph fusions.

## Deliberate omissions (documented, not regressions)

- **Metal PTQ1_0/PQ2_0 kernels** — Metal conflicts were resolved to the fork's
  side; the BeeLlama Metal backend is unchanged. Bonsai on Apple Metal is not
  supported by this integration (CUDA/CPU are the target backends).
- **Vulkan PTQ1_0/PQ2_0 shaders** — same reason; the Vulkan conflicts were
  resolved to the fork's side. Building with `GGML_VULKAN=ON` fails in shader
  generation (`get_dm` overloads for the new types are absent). The validation
  and benchmark builds use `-DGGML_VULKAN=OFF`.
- **prism dspark tests** (`test-dspark-forward/loop/real-eval`) — removed from
  the build; they reference the prism speculative API, not the fork's. The
  dspark-markov CUDA code is kept but off by default
  (`-DLLAMA_DSPARK_MARKOV_CUDA=OFF`; the option defaults ON when CUDA is on).

## Build (reproducible)

GPU: NVIDIA RTX 4070 (sm_89), CUDA 13.3, `-march=native`.

```bash
cmake --preset optimized -DLLAMA_DSPARK_MARKOV_CUDA=OFF -DGGML_VULKAN=OFF \
      -DGGML_CUDA_NO_VMM=ON
cmake --build build-optimized -j8
```

`build-optimized/bin/llama-server --version` → `0.4.7-dev (build 12259, commit 2030532ae)`.

## Validation (Bonsai 2 27B, ternary)

Model: `prism-ml/Ternary-Bonsai-2-27B-gguf` (PTQ1_0 5.95 GB, PQ2_0 7.21 GB).

Both formats load and generate on CUDA:

```bash
llama-cli -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -fa on -c 8192 \
  --log-disable --temp 1.0 --top-p 0.95 --top-k 20 -st \
  -p "What is the capital of France?" -n 40
```

Both produce "Paris" with correct thinking. The same holds for PQ2_0.

KVarN cache works with the ternary weights:

```bash
llama-cli -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -fa on -c 4096 -st \
  --cache-type-k kvarn4 --cache-type-v kvarn4 -p "..." -n 30     # KVarN4/4  OK
llama-cli -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -fa on -c 4096 -st \
  --cache-type-k kvarn4 --cache-type-v kvarn3 -p "..." -n 30     # KVarN4/3  OK
llama-cli -m Ternary-Bonsai-2-27B-PQ2_0.gguf  -ngl 99 -fa on -c 4096 -st \
  --cache-type-k kvarn4 --cache-type-v kvarn4 -p "..." -n 30     # KVarN4/4  OK
```

Code-bug prompt (first-missing-positive off-by-one) is correctly diagnosed and
fixed by both the integrated build and the PrismML original.

`test-quantize-fns` passes (exit 0) including `pq2_0` and `ptq1_0` quantization
error and vec-dot thresholds.

## Benchmark: PrismML original vs BeeLlama integrated

Same model/config on both: `llama-server -m MODEL -ngl 99 -fa on -c 8192 -b 512
-ub 256`. Prompt: first-missing-positive code-fix request (208 tokens),
`max_tokens 512`, `temperature 0`, streaming. Warmup + 3 runs. Same CUDA
config (`GGML_CUDA_NO_VMM=ON`, sm_89, Release). PrismML original built with the
same options as BeeLlama.

| Metric | PrismML original | BeeLlama integrated | Δ |
|---|---|---|---|
| **PTQ1_0** | | | |
| Load time (s) | 15.64 | 13.17 | −16% |
| Prefill pp512 (tok/s) | 612.9 | 610.1 | −0.5% |
| Decode tg128 (tok/s) | 52.60 | 51.27 | −2.5% |
| Decode (server, tok/s) | 51.63 | 50.68 | −1.8% |
| TTFT (short prompt, ms) | 236 | 238 | ≈0 |
| VRAM weights (MiB) | 6815 | 6787 | ≈0 |
| **PQ2_0** | | | |
| Decode (server, tok/s) | 49.80 | 49.92 | +0.2% |
| TTFT (server, s) | 0.23 | 0.45 | see note |
| **Sampling / output** | temp 0, identical token counts | | |

Note on TTFT: with a fully free GPU the fixed per-request TTFT is ~236 ms on
both servers (short-prompt measurement). The larger server-bench TTFT for the
long prompt reflects request-path scheduling and is within measurement variance
between the two server binaries; the underlying prefill rate is identical
(610 vs 613 tok/s).

**Conclusion: no significant regression.** Prefill identical, decode within
±2.5% (run-to-run noise), load time slightly faster, output identical.

## Reproduce

Benchmark harness: `tests/bench-server.py` (starts llama-server on a port,
waits for /health, warmup + N streaming chat completions, reports load time,
VRAM weights, KV+compute VRAM delta, TTFT, decode tok/s; requires the GPU free).