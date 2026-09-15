# Anbeeld's BeeLlama.cpp

![BeeLlama.cpp logo](beellama.jpg)

BeeLlama.cpp (or just Bee) is a performance-focused llama.cpp fork for squeezing more speed and context out of local GGUF inference. It adds variance-normalized KV-cache quantization (KVarN), KV cache precision tail for recent tokens, low-bit cache types, adaptive draft control for speculative decoding, reasoning-loop protection, and more.

> Not quite a pegasus, but close enough.

[![Support my work!](https://anbeeld.com/images/support.jpg)](https://anbeeld.com/support)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

## Fork Features

- **Variance-normalized KV-cache quantization (KVarN)**: provides higher precision at similar memory costs. Independent K and V bit widths at `kvarn2`, `kvarn3`, `kvarn4`, `kvarn5`, `kvarn6`, and `kvarn8`, set with `--cache-type-k` and `--cache-type-v`.
- **KV cache precision tail**: keep most of the KV cache quantized while storing recent tokens in F16/BF16, enabled with `--kv-tail-tokens`. A single global softmax merges the quantized body and the precision tail under FlashAttention, without materializing the whole cache.
- **Standard low-bit KV cache types**: `q2_0`, `q2_1`, `q3_0`, `q3_1`, `q6_0`, and `q6_1`, usable for either target or draft caches alongside the upstream `q4`/`q5`/`q8` types.
- **Adaptive draft-max for DFlash**: adjusts the active DFlash draft horizon at runtime instead of using a fixed `--spec-draft-n-max`, comparing speculative throughput against a no-spec baseline.
- **Reasoning-loop protection**: the server detects repeated hidden reasoning output and intervenes.

For the full feature and public-repo comparison, read [docs/beellama-features.md](docs/beellama-features.md). For the complete argument reference, read [docs/beellama-args.md](docs/beellama-args.md).

## About this fork

This fork of `ggml-org/llama.cpp` adds adaptive MTP context switching with resident model weights, context-based model routing, cache migration, automatic disk prompt/KV persistence, CUDA memory recovery, NVIDIA GPU governors, and experimental paged KV/SnapKV serving. Standard upstream capabilities, including the CLI, web UI, model conversion, quantization and OpenAI-compatible server, remain available.

- [Current GOKAYA deployment](#current-gokaya-deployment-2026-09-14)
- [Adaptive context and cache reuse](#adaptive-context-with-resident-weights-tri-profile-architecture)
- [Automatic disk prompt cache](#automatic-disk-prompt-cache)
- [Router mode](#context-based-router-mode)
- [CUDA and GPU controls](#cuda-memory-recovery-and-gpu-governors)
- [Paged KV, SnapKV and speculation](#paged-kv-snapkv-and-other-speculation)
- [Validation and limitations](#validation-and-operational-history)

### Current GOKAYA deployment (2026-09-14)

The development checkout is `/home/hjotha/llama` on GOKAYA (`192.168.1.57`), branch `master`; `origin` is `hjotha/llama.cpp` and `upstream` is the official repository. The source includes upstream `093a2f86c` and the later upstream fix `661643e43`. The slot-save release includes the reviewed integration of upstream PRs 24003/24004 with the fork's tri-profile and MTP paths. The deployed binary reports `0.4.0-dev`, build `11088`, commit `65db067dc`.

The system unit `llama-server-root.service` runs `/home/hjotha/releases/llama-slot-save-20260914/build-cuda-vulkan/bin/llama-server` on port `8090`. It loads `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` directly into a single adaptive process. The old `/home/hjotha/prod-two-tier.ini` still exists but is not used by the active unit.

| Setting | Active configuration |
| --- | --- |
| GPU | RTX 4070, `--device CUDA0 --gpu-layers 99` |
| Short profile (`mtp-short`) | 32,768 context tokens, MTP enabled with N=4 draft tokens |
| Medium profile (`mtp`) | 56,320 context tokens, MTP enabled with N=2 draft tokens |
| Long profile (`long`) | 97,536 context tokens, MTP disabled (N=0), MTP weights offloaded to host RAM |
| Profile selection thresholds | Prompt + output reserve <= 32,768 -> mtp-short; <= 56,320 -> mtp; else -> long |
| Batch / ubatch / slots | 256 / 256 / 1 |
| KV cache | Traditional KV, target and draft K/V types `q4_0`, FlashAttention on |
| RAM prompt cache / checkpoints | 2,048 MiB / one checkpoint per slot |
| Speculation | `draft-mtp`, dynamic draft N (4 in short, 2 in medium, 0 in long), draft cutoff `0.80` |
| Loading / fitting | `--load-mode none --fit off --no-context-shift` |
| Slot persistence | `--slot-save-path /home/hjotha/llama-slot-cache/ --slot-save-auto`; 64 snapshots / 32 GiB by default |
| NVIDIA power | 200 W prefill, 170 W decode |
| Memory-clock target | 11,001 MHz during decode; prefill/idle use automatic clocks |
| Backend isolation | CUDA+Vulkan build, `GGML_DISABLE_VULKAN=1` for this CUDA process |
| CUDA graph recovery margin | `GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB=18` |
| Monitoring | `--metrics`, `/health`, `/props`, `/slots`, `/v1/models` |

After startup, read-only HTTP checks returned `/health: {"status":"ok"}`; `/props` and `/slots` reported `profile=mtp-short`, `state=ready`, `context_size=32768`, `context_size_long=97536` and `mtp_weights_resident=true`. The model catalog advertises the long context while the slot reports the active profile context. These checks establish current status, not a new maximum-context benchmark.

The active unit has no `--alias`: its model ID is the full GGUF path reported by `/v1/models`. The former router name `qwen-3.8-27b` is not the catalog ID of this direct deployment. Use the returned ID, or configure `--alias` when starting a separate server.

### Adaptive context with resident weights (Tri-Profile Architecture)

Enable this mode with `--ctx-size-mtp N`; the default `0` preserves ordinary server behavior. `--ctx-size` sets the long capacity. `--mtp-max-tokens` sets the medium-profile budget threshold (defaults to `--ctx-size-mtp`). `--ctx-size-mtp-short` sets the short-profile context size, `--mtp-short-max-tokens` sets its budget threshold, and `--spec-draft-n-max-short` sets the draft N for the short profile (default: 4).

For example, the core GOKAYA configuration is:

```sh
GGML_DISABLE_VULKAN=1 GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB=18 \
./build-cuda-vulkan/bin/llama-server \
  --model /home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf \
  --ctx-size 97536 --ctx-size-mtp 56320 --mtp-max-tokens 56320 \
  --ctx-size-mtp-short 32768 --mtp-short-max-tokens 32768 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-max-short 4 \
  --spec-draft-p-min 0.80 --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 \
  --device CUDA0 --gpu-layers 99 --parallel 1 --fit off \
  --batch-size 256 --ubatch-size 256 --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --cache-ram 2048 --ctx-checkpoints 1 --load-mode none \
  --slot-save-path /home/hjotha/llama-slot-cache/ --slot-save-auto \
  --no-context-shift --metrics --host 127.0.0.1 --port 8090
```

Run this from a matching build directory's parent with the port free; it is a launch example, not a command to run alongside the production service. GPU governors are optional and described below.

**Selection and lifecycle**

- Selection counts the complete formatted prompt, including cached tokens, plus the normalized output reserve. With no finite request/server output limit, 4,096 tokens are reserved for selection only; this does not impose a new generation limit.
- If total budget <= 32,768 tokens, the server selects the short profile (`mtp-short`, N=4).
- If total budget is between 32,768 and 56,320 tokens, the server selects the medium profile (`mtp`, N=2).
- If total budget exceeds 56,320 tokens, the server selects the long profile (`long`, N=0, up to 97,536 tokens).
- An explicit budget above the effective long capacity is rejected before switching.
- The server boots in the short profile (`mtp-short`) and switches between requests after active work is drained.
- Transitions preserve model weights in GPU memory. Only the MTP head weights are moved between VRAM and host RAM when entering or exiting the long profile.
- Transition failures attempt rollback to the previous profile. If rollback also fails, the server publishes `state=unavailable` and rejects inference with HTTP 503.

**RAM cache and explicit slot files across profiles**

The adaptive RAM prompt cache carries target state, draft/speculative state and checkpoints across profile switches:
- Layout and attention semantics: `common_prompt_cache_layout()` checks tensor layout semantics rather than buffer capacity, so cached prefixes remain compatible across profile transitions.
- Automatic cache preservation: when a profile switch occurs, `slot.prompt_save(*prompt_cache)` serializes valid prefixes and checkpoints into the global RAM cache (`--cache-ram`) before tearing down the old context. The new profile context rehydrates compatible prefixes via `slot.prompt_restore`, avoiding repeated prefill.
- Explicit slot persistence: `POST /slots/{id_slot}?action=save` and `?action=restore` record profile identifiers (`0` = medium MTP, `1` = long, `2` = short MTP). When restoring a snapshot from a different profile, the server automatically triggers a dynamic transition (`switch_adaptive_context(snapshot.profile)`), reloads MTP weights to GPU if needed, and applies the saved KV state.

`adaptive_context` in `/props`, `/models` and `/slots` exposes `enabled`, `profile`, `state`, `context_size`, `context_size_long` and `mtp_weights_resident`.

**Supported scope:** dense Qwen35-family MTP models, including this Qwen3.8 GGUF, with one MTP head, one CUDA device, one slot, traditional KV and `fit=off`. CPU lifecycle checks are supported with `--gpu-layers 0`. Adaptive mode rejects external draft models, other speculative backends, paged KV, multimodal input, LoRA/control vectors and automatic sleep. These restrictions apply to adaptive mode, not to all upstream server features.

See the [server usage guide](tools/server/README.md), [lifecycle and selection code](common/common.cpp), [server transitions and slot persistence](tools/server/server-context.cpp), [RAM cache implementation](tools/server/server-task.cpp) and [model identity implementation](tools/server/server-model-identity.cpp).

### Automatic disk prompt cache

This fork integrates the closed upstream [PR 24003](https://github.com/ggml-org/llama.cpp/pull/24003) and [PR 24004](https://github.com/ggml-org/llama.cpp/pull/24004), adapted to its token serialization, request statistics, adaptive contexts and MTP bootstrap. Automatic persistence is opt-in:

```sh
mkdir -p /home/hjotha/llama-slot-cache
# Add these options to the server launch command:
# --slot-save-path /home/hjotha/llama-slot-cache/ --slot-save-auto
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--slot-save-auto` | Disabled | Save completed generative requests and restore useful disk prefixes before prefill. Requires `--slot-save-path`. |
| `--slot-save-block N` | 256 | Token block size for prefix lookup; disk reuse must improve the memory match by at least one block. |
| `--slot-save-max-count N` | 64 | Maximum snapshot count; `0` disables this limit. |
| `--slot-save-max-mb N` | 32768 | Store size limit in MiB; `0` disables this limit. A snapshot exceeding this limit is rejected without evicting the older snapshots. |

The corresponding environment variables are `LLAMA_ARG_SLOT_SAVE_AUTO`, `LLAMA_ARG_SLOT_SAVE_BLOCK`, `LLAMA_ARG_SLOT_SAVE_MAX_COUNT` and `LLAMA_ARG_SLOT_SAVE_MAX_MB`. Both size/count limits apply to manual saves as well, including adaptive snapshots, even when automatic persistence is disabled.

**Use a dedicated directory.** Bounded eviction treats regular files in `--slot-save-path` as owned by the server; never use `/tmp/` itself or mix unrelated files into this directory. State files and their `.meta`/`.logits` sidecars are evicted together by modification time; a successful automatic restore refreshes that time. `.tmp` is reserved for temporary files. Interrupted writes can leave temporary files that require operator cleanup after all writers stop.

Completed requests save from the first turn, including with `--no-cache-idle-slots` or `--cache-ram 0`. Requests with `cache_prompt=false`, embeddings/reranking, and prompts containing media are excluded. Adapter/control-vector configurations disable automatic disk caching with a startup warning because their content identity is not captured. Saves and restores use synchronous disk I/O on the server loop, so large snapshots can delay queued requests.

A cold process indexes compatible metadata and verifies the actual token prefix before accepting a native state. Identity uses the fork's existing model-content digest (including GGUF shards and metadata overrides), effective cache layout/RoPE/state versions, KV types and active context size. The fingerprint and index are refreshed after every adaptive profile rebind. Automatic files restore into the selected request profile; explicit adaptive `/slots` snapshots retain their separate format and profile-switching behavior. The stronger fingerprint deliberately invalidates automatic files from the initial integration; they remain eligible for normal LRU eviction.

For target-only restores with MTP, the next real target suffix decode rebuilds the draft carry before speculation resumes. FULL/recurrent states require the whole saved token sequence to match; they cannot safely rewind arbitrary prefixes. An exact restored prompt can sample its first token from a validated `.logits` sidecar with `prompt_n=0`. Missing/invalid sidecars, speculative requests requiring unsaved draft state, and pre-sampling probability requests use a normal prefill instead. A matching hash alone never authorizes reuse. Snapshots from different branches, including tails shorter than a block, remain separate candidates.

The store is a best-effort cache, not a durable conversation archive. Missing, incompatible or invalid snapshots fall back to prefill. Keep the directory restricted to trusted local writers and use the same build/configuration for cooperating processes; concurrent writer failure and hostile native-state payloads are not a hardened storage protocol. The model-content identity helper currently requires POSIX file identity support; automatic caching is not validated on Windows.

Recurrent save/restore regressions live in [the existing slot tests](tools/server/tests/unit/test_slot_save.py). Set `SLOT_SAVE_HTTP_MODEL` to a local FULL/recurrent GGUF, `N_GPU_LAYERS` as appropriate and run `pytest tools/server/tests/unit/test_slot_save.py -k recurrent_slot_disk_cache -v -s` with `SLOW_TESTS=1` and `LLAMA_SERVER_BIN_PATH` pointing at the tested build. The tests cover manual and automatic cold restores, exact-prompt logits, suffix reuse, prediction metrics, corrupt/oversized sidecars and disagreement between metadata and native token payloads.

Validation on 2026-09-14: Qwen3.5-4B FULL/recurrent tests regenerated a cold 601-token snapshot with `prompt_n=0`, evaluated a three-token suffix with `prompt_n=3`, and safely reprefilled corrupt snapshots. A separate Qwen3.5-4B-MTP GPU run restored 301/1301/2401-token snapshots in short/medium/long profiles (1024/2048/4096 context), processing three new tokens in each case; count/size limits and `cache_prompt=false` also passed. Legacy slot save/restore/erase, model identity and RAM/MTP cache checks passed.

The deployed Qwen3.8-27B MTP model processed 6653 prompt tokens, answered `AZUL`, and saved 6655 tokens. After a service restart, its continuation logged `auto-restore: reused 6655 tokens` and `target-only MTP bootstrap accepted: decoded_suffix=24`; it answered `4` with `prompt_n=28` in 0.748 s. The initial request took 6.991 s. These are different requests demonstrating disk reuse, not a controlled speedup benchmark. Production health returned `{"status":"ok"}` and the active profile remained `mtp-short` (32768 context). The previous release binaries are retained in `/home/hjotha/releases/llama-slot-save-20260914/pre-review-65db067dc-bin/` for rollback.

### Context-based router mode

The earlier multi-process design remains available as an alternative to resident switching:

- Preset entries can share a `route-group` public name and use `route-max-tokens` thresholds. The router chooses the smallest fitting tier for prompt plus requested output, with a byte estimate on cold start until tokenization is available.
- Selection is recalculated for each request. Conversations can return to a smaller tier; the old README's claim that conversation pinning prevents demotion is obsolete.
- `--models-max 1` limits loaded children. Swaps wait for active work, and cold-load request reservations protect concurrently arriving requests.
- `X-Conversation-Id` supports best-effort slot-state migration on promotion when `--slot-save-path` is set. Failed destination loads discard the saved snapshot. This does not guarantee cache transfer on every downward swap or incompatible prefix.
- Chat/Responses compatibility and both OpenAI/Codex model-list payloads remain available in router mode.

Router swaps unload/load child processes; adaptive mode instead rebuilds contexts inside one model process. The historical two-tier preset is retained for that router mode. See [router implementation](tools/server/server-models.cpp), [server options](tools/server/README.md) and the [historical router design and measurements](docs/mtp-router-split-plan.md).

### CUDA memory recovery and GPU governors

| Feature | Behavior and controls |
| --- | --- |
| CUDA Graph stability | Handles changing prompt shapes, graph capture/instantiation failure and cleanup of stale graph resources. Recoverable memory pressure can fall back to ordinary kernel execution. |
| Preventive graph margin and recovery | Checks free VRAM before capture and can re-enable graphs after memory recovers. `GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB` controls the margin: current source default 32 MiB, GOKAYA override 18 MiB. This margin does not reserve VRAM or make every OOM recoverable. |
| CUDA allocator recovery | Restores recoverable VMM-pool OOM handling and limits retry/allocation behavior. |
| IQ1_M workspace and kernels | Bounds the full-dequantization workspace via the existing MMVQ path; adds IQ1_M MMQ/tensor-core support and initializes only the selected MMQ kernel variant. |
| Backend isolation | `GGML_DISABLE_VULKAN` prevents Vulkan instance creation in both static registration and plugin entry paths, avoiding unused Vulkan allocations in a CUDA-only process. Presence is enough: even `GGML_DISABLE_VULKAN=0` disables it. Set it before startup; omit it for Vulkan workloads. |
| Power governor | `--gpu-power-prefill W` and `--gpu-power-decode W` select phase-specific NVML power limits; `--gpu-power-device N` selects the NVML device. Idle retains the last power limit; shutdown/sleep restore the original limit. |
| Memory-clock governor | `--gpu-mem-clock-decode MHz` and optional `--gpu-mem-clock-prefill MHz` set phase-specific clocks. Idle and unconfigured phases reset memory clocks. Above-stock targets use supported locks plus a bounded offset when the driver supports it. |

Governors run inside the server through NVML, deduplicate unchanged settings and require driver permission to change GPU controls. For the active RTX 4070 configuration, startup logs resolve the 11,001 MHz decode target to a 10,501 MHz lock plus a +1,500 MHz offset. This mapping is device/driver-specific. The historical statement that the memory governor is disabled is no longer current.

**GPU core and memory-clock parameters**

| Server parameter | Environment variable | Effect |
| --- | --- | --- |
| `--gpu-mem-clock-decode MHz` | `LLAMA_ARG_GPU_MEM_CLOCK_DECODE` | Memory-clock target during token generation; GOKAYA uses `11001`. |
| `--gpu-mem-clock-prefill MHz` | `LLAMA_ARG_GPU_MEM_CLOCK_PREFILL` | Memory-clock target during prompt processing; omitted on GOKAYA. |
| `--gpu-power-prefill W` | `LLAMA_ARG_GPU_POWER_PREFILL` | Prefill power limit; GOKAYA uses `200`. |
| `--gpu-power-decode W` | `LLAMA_ARG_GPU_POWER_DECODE` | Decode power limit; GOKAYA uses `170`. |
| `--gpu-power-device N` | `LLAMA_ARG_GPU_POWER_DEVICE` | NVML GPU index shared by both governors; default `0`. This is not the CUDA layer-offload selector. |

Clock and power values must be positive. Omit both memory-clock options to disable the memory governor; with it enabled, idle and any phase without a configured target release its memory lock and restore any offset it applied. The two power options must be supplied together; the memory governor can run without them. Power limits influence available GPU boost but do not set a core frequency in MHz.

The server has no GPU core/SM clock flag. NVIDIA's external controls are:

| Command template | Effect |
| --- | --- |
| `sudo nvidia-smi -i N --lock-gpu-clocks=MIN,MAX` (`-lgc`) | Request a GPU core clock range in MHz; use equal bounds for one target. |
| `sudo nvidia-smi -i N --reset-gpu-clocks` (`-rgc`) | Restore default GPU core clock control. |
| `sudo nvidia-smi -i N --lock-memory-clocks=MIN,MAX` (`-lmc`) | Request a memory clock range in MHz, independently of server phases. |
| `sudo nvidia-smi -i N --reset-memory-clocks` (`-rmc`) | Restore default memory clock control. |

Replace `N` with the NVIDIA device index and `MIN,MAX` with supported MHz values. These controls require GPU/driver support and permission; external locks apply beyond a request and have no automatic server-idle reset. Use a single owner for memory clocks: manual locks can be overwritten by the server governor's phase transitions. A plain memory lock does not reproduce its above-stock offset logic.

Read the observed clocks with `nvidia-smi -i 0 --query-gpu=clocks.current.sm,clocks.current.memory,power.limit --format=csv`.

Relevant sources are [CUDA graph/allocator handling](ggml/src/ggml-cuda/common.cuh), [CUDA kernels](ggml/src/ggml-cuda), [Vulkan registration](ggml/src/ggml-vulkan/ggml-vulkan.cpp) and the [GPU governor](tools/server/server-gpu-power.cpp). The [power](docs/phase-aware-nvidia-gpu-power-governor.md) and [memory-clock](docs/phase-aware-nvidia-gpu-memory-clock-governor.md) design documents retain earlier measurements and configurations.

### Paged KV, SnapKV and other speculation

These paths remain in the fork but are not enabled by the current traditional-KV adaptive deployment.

| Capability | Options / implementation |
| --- | --- |
| Shared physical KV pool | `--kv-paged` shares blocks across sequences/slots. `--kv-block-size`, `--n-gpu-blocks` and `--n-cpu-blocks` control block size and physical capacity. |
| Elastic growth and migration | `--kv-paged-dynamic`, `--n-gpu-blocks-initial` and `--n-gpu-blocks-growth` separate initial allocation from growth, with attention-layer migration across registered GPU backends and optional CPU spill. |
| Automatic pool fitting | `--kv-paged-prealloc-max` calibrates capacity after model load, accounting for model, compute, hybrid/recurrent and MTP requirements, and caps dynamic growth. |
| Request admission | `--kv-paged-admission-blocks` and `--kv-paged-watermark` gate physical demand instead of treating logical context size as unlimited memory. Over-budget requests are rejected cleanly. |
| Multi-device placement | `--paged-attn-cuda` pins full-attention layers to the first offload device while tensor splitting can place recurrent-only layers elsewhere. Paged paths include shared multi-sequence batches and meta-buffer handling. |
| SnapKV selective retention | `--snapkv OW`, `--snapkv-retention`, `--snapkv-recent`, `--snapkv-pinned` and `--snapkv-budget-blocks` score and retain selected old pages while protecting configured leading/trailing windows. Streaming, episodic, per-head and per-sequence scoring paths and eviction timing are present. |
| Paged backends | CUDA/CPU paths plus Vulkan quantized paged attention, parallel prefill and Vulkan 1.1 Android compatibility work. Backend support does not imply equal speed or stability on every device. |
| Traditional KV fitting | Model-specific, compute-aware context fitting accounts for hybrid/MTP state. Production selects explicit contexts with `--fit off`. |
| DFlash2 | Separate draft-model speculation path with `p_min` and stochastic verification support; no DFlash draft model is configured in production. Adaptive mode supports only `draft-mtp`. |

SnapKV retention is lossy and experimental: reducing physical KV can change output quality. Paged KV is not automatically faster than traditional KV; older Vulkan iGPU experiments include slowdown and failure cases. Context, pool sizes and throughput need validation for the selected model, backend and workload.

Use the [paged example and controls](examples/paged/README.md), [SnapKV episode benchmark](tools/snapkv-episode-bench.py), [40K/MTP benchmark](tools/snapkv-40k-mtp-bench.py), [argument definitions](common/arg.cpp), [speculation implementation](common/speculative.cpp), [KV calibration findings](docs/kv-calibration-findings.md) and [Vulkan iGPU notes](docs/notes-paged-kv-vulkan-igpu.md).

### Validation and operational history

The [adaptive implementation plan and dated evidence](plans/001-adaptive-context-resident-weights.md) records CPU lifecycle/cache tests, sanitizer runs, real CUDA short/long/short transitions, maximum-context requests, repeated cycles, fault/rollback checks and subsequent production fixes. Recorded 20-pair CUDA runs kept one main-model instance/load through 40 completions; those serial results are workload-specific, not a general concurrency or latency guarantee.

Reproducible coverage lives in [adaptive HTTP tests](tools/server/tests/unit/test_adaptive_context.py), [state/lifecycle tests](tests/test-save-load-state.cpp), [prompt-cache tests](tests/test-server-prompt-cache.cpp), [model-identity tests](tests/test-server-model-identity.cpp) and the [Gauntlet harnesses](gauntlet). The plan still records open work on detailed physical profiling, quantitative comparison with the router, broader concurrency, additional sanitizer coverage, persistence and rollback/soak validation. Implemented features and a successful deployment do not close all those gates.

The [previous production capture](gauntlet/production/production-final-20260914.txt) and [unit snapshot](gauntlet/production/llama-server-root.service.final) describe the earlier `llama-adaptive-fix-20260914` release; the live optimized release path/build at the top of this README supersedes them. The [router report](docs/mtp-router-split-plan.md), [calibration notes](docs/kv-calibration-findings.md) and [OOM trace](docs/oom-reproduction-trace.md) preserve earlier results. Their 60,416-token MTP profile, no-demotion rule, disabled memory-clock setting and old free-VRAM figures are historical, not the current deployment.

## Quick start

## KV Cache Quantization

K and V cache types are set independently with `--cache-type-k` and `--cache-type-v`. The research is covered in articles: [KVarN KV Cache: Implementation and Benchmarks](https://anbeeld.com/articles/kvarn-kv-cache-implementation-and-benchmarks), [KV Cache Precision Tail: Implementation and Benchmarks](https://anbeeld.com/articles/kv-cache-precision-tail-implementation-and-benchmarks), and [KV Cache Quantization Benchmarks: KVarN, Precision Tail](https://anbeeld.com/articles/kv-cache-quantization-benchmarks-kvarn-precision-tail).

### KV Cache Recommenation Ladder

The measurements come from Qwen 3.6 27B Q5_K_S at 64K context on Wikitext-2 raw with `-b 2048 -ub 512` on an RTX 3090. Median KLD is the primary quality metric; lower is better.

| K / V | Tail | Size | Size vs bf16 | Median KLD | 99.9% KLD | What it is for |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bf16 / bf16` | 0 | 4096 MiB | 100.0% | 0 | 0.000050 | Reference |
| `q8_0 / q8_0` | 1024 | 2272 MiB | 55.5% | 0.000897 | 0.087699 | Standard fidelity with a precision tail |
| `kvarn8 / kvarn8` | 1024 | 2256 MiB | 55.1% | 0.000871 | 0.087639 | Best measured quality below BF16 |
| `q8_0 / q8_0` | 0 | 2176 MiB | 53.1% | 0.000909 | 0.093029 | Standard fidelity |
| `q8_0 / q6_0` | 1024 | 2016 MiB | 49.2% | 0.000894 | 0.091098 | q8_0 quality within noise, 256 MiB less |
| `kvarn6 / kvarn6` | 1024 | 1744 MiB | 42.6% | 0.000879 | 0.084629 | High-end value pick |
| `kvarn6 / kvarn5` | 1024 | 1616 MiB | 39.5% | 0.000886 | 0.092778 | Much cheaper, almost as good |
| `kvarn5 / kvarn5` | 1024 | 1488 MiB | 36.3% | 0.000897 | 0.087666 | Highest value in the mid-range |
| `q5_0 / q4_1` | 1024 | 1440 MiB | 35.2% | 0.000966 | 0.089128 | Standard option when VRAM-constrained |
| `kvarn5 / kvarn4` | 1024 | 1360 MiB | 33.2% | 0.000936 | 0.089469 | Balanced default |
| `q4_0 / q4_0` | 1024 | 1248 MiB | 30.5% | 0.001057 | 0.104486 | Compact standard cache |
| `kvarn4 / kvarn4` | 1024 | 1232 MiB | 30.1% | 0.000994 | 0.090391 | Cleaner than q4_0 for less memory |
| `kvarn4 / kvarn3` | 1024 | 1104 MiB | 27.0% | 0.001112 | 0.113968 | Smallest recommended tier |
| `kvarn3 / kvarn3` | 1024 | 976 MiB | 23.8% | 0.001316 | 0.139558 | When the context must fit |
| `kvarn3 / kvarn2` | 1024 | 848 MiB | 20.7% | 0.002424 | 0.238780 | Emergency compression |
| `kvarn2 / kvarn2` | 1024 | 720 MiB | 17.6% | 0.003811 | 0.450496 | Last resort |

A 1024-token tail is a useful starting point for low-bit Qwen caches. Prefer more body precision when old and recent tokens matter equally; consider 2048 only when the newest two thousand tokens are genuinely the privileged working set. Standard caches generally have slightly faster prefill.

Gemma 4 needs a separate decision. Its 1024-token sliding window makes a 1024 tail exact across most layers, sharply changing memory and throughput. Standard `q8_0 / q8_0` without a tail is the safer Gemma default when throughput and older-context coverage matter, and the benchmark results recommend avoiding quantized KV cache when Gemma quality is non-negotiable.

### Standard-Only KV Cache Ladder

Use this generic fallback ladder when KVarN or precision tails are unavailable. It is based on the same Qwen 3.6 27B benchmark, with every standard cache row using tail 0.

| K / V | Size | Size vs bf16 | Median KLD | 99.9% KLD | What it is for |
| --- | ---: | ---: | ---: | ---: | --- |
| `bf16 / bf16` | 4096 MiB | 100.0% | 0 | 0.000050 | Reference |
| `q8_0 / q8_0` | 2176 MiB | 53.1% | 0.000909 | 0.093029 | Compression with minimal losses |
| `q8_0 / q6_0` | 1920 MiB | 46.9% | 0.000937 | 0.093575 | 256 MiB below q8_0 |
| `q6_0 / q6_0` | 1664 MiB | 40.6% | 0.000960 | 0.091134 | High-end value pick |
| `q6_0 / q5_0` | 1536 MiB | 37.5% | 0.001054 | 0.094670 | Balanced default |
| `q5_0 / q5_0` | 1408 MiB | 34.4% | 0.001154 | 0.097070 | Last tier before the cliff |
| `q5_0 / q4_1` | 1344 MiB | 32.8% | 0.001433 | 0.122096 | Default when VRAM-constrained |
| `q5_0 / q4_0` | 1280 MiB | 31.3% | 0.001516 | 0.121068 | 64 MiB cheaper, worse median |
| `q4_0 / q4_0` | 1152 MiB | 28.1% | 0.001846 | 0.154408 | Smallest recommended tier |
| `q4_0 / q3_0` | 1024 MiB | 25.0% | 0.003313 | 0.218912 | When the context must fit |
| `q3_0 / q3_0` | 896 MiB | 21.9% | 0.004696 | 0.304186 | Emergency compression |
| `q2_0 / q2_0` | 640 MiB | 15.6% | 0.019374 | 1.198902 | Last resort |

### KV Cache Type Reference

<details>
<summary><strong>All KV cache types available in BeeLlama</strong></summary>

| Type | Origin | bpv | Size vs bf16 |
| --- | --- | ---: | ---: |
| `q8_0` | upstream | 8.5 | 53.1% |
| `kvarn8` | Huawei / fork | 8.375 | 52.3% |
| `q6_1` | fork | 7.0 | 43.8% |
| `q6_0` | fork | 6.5 | 40.6% |
| `kvarn6` | Huawei / fork | 6.375 | 39.8% |
| `q5_1` | upstream | 6.0 | 37.5% |
| `q5_0` | upstream | 5.5 | 34.4% |
| `kvarn5` | Huawei / fork | 5.375 | 33.6% |
| `q4_1` | upstream | 5.0 | 31.3% |
| `q4_0` | upstream | 4.5 | 28.1% |
| `iq4_nl` | upstream | 4.5 | 28.1% |
| `kvarn4` | Huawei / fork | 4.375 | 27.3% |
| `q3_1` | fork | 4.0 | 25.0% |
| `q3_0` | fork | 3.5 | 21.9% |
| `kvarn3` | Huawei / fork | 3.375 | 21.1% |
| `q2_1` | fork | 3.0 | 18.8% |
| `q2_0` | fork | 2.5 | 15.6% |
| `kvarn2` | Huawei / fork | 2.375 | 14.8% |

Standard ratios come directly from each block format. KVarN ratios describe the compressed record body, including scale and zero-point metadata; the permanent exact sink, exact suffix, staging, and alignment add overhead.

</details>

## Installation

### Prebuilt

Current release binaries are on the [releases page](https://github.com/Anbeeld/beellama.cpp/releases).

| Platform | Backend | Asset suffix |
| --- | --- | --- |
| macOS arm64 | Metal | `bin-macos-arm64.tar.gz` |
| Ubuntu x64 | CPU | `bin-ubuntu-x64.tar.gz` |
| Ubuntu arm64 | CPU | `bin-ubuntu-arm64.tar.gz` |
| Ubuntu x64 | CUDA 12.4 | `bin-ubuntu-cuda-12.4-x64.tar.gz` |
| Ubuntu x64 | CUDA 13.1 | `bin-ubuntu-cuda-13.1-x64.tar.gz` |
| Ubuntu x64 | Vulkan | `bin-ubuntu-vulkan-x64.tar.gz` |
| Ubuntu x64 | ROCm 7.2 | `bin-ubuntu-rocm-7.2-x64.tar.gz` |
| Ubuntu x64 | SYCL | `bin-ubuntu-sycl-x64.tar.gz` |
| Windows x64 | CPU | `bin-win-cpu-x64.zip` |
| Windows x64 | Vulkan | `bin-win-vulkan-x64.zip` |
| Windows x64 | SYCL | `bin-win-sycl-x64.zip` |
| Windows x64 | CUDA 12.4 | `bin-win-cuda-12.4-x64.zip` |
| Windows x64 | CUDA 13.1 | `bin-win-cuda-13.1-x64.zip` |
| Windows x64 | HIP/Radeon | `bin-win-hip-radeon-x64.zip` |

Windows CUDA archives contain a `ggml-cuda.dll` backend; download the matching `beellama-<version>-cudart-win-cuda-*-x64.zip` runtime archive and extract it into the same folder. Windows SYCL and HIP archives ship as standalone packages with all required runtime DLLs bundled.

Docker images are published to `ghcr.io/anbeeld/beellama.cpp`:

| Image | Acceleration | Platforms |
| --- | --- | --- |
| `server`, `server-cpu` | CPU | linux/amd64, linux/arm64 |
| `server-cuda`, `server-cuda12` | CUDA 12.4 | linux/amd64 |
| `server-cuda13` | CUDA 13.1 | linux/amd64 |
| `server-rocm` | ROCm | linux/amd64 |
| `server-vulkan` | Vulkan | linux/amd64 |
| `server-sycl` | SYCL | linux/amd64 |

Building from source with `-DGGML_NATIVE=ON` *may* result in a *tiny* bit better performance, so it might still be a good idea to do that if/when you decide to use this fork long-term.

### CUDA Build

```bash
# Linux (GCC + CUDA)
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Windows (MSVC + CUDA)
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON ^
  -DGGML_CUDA_FA=ON ^
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

# macOS (Metal)
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The default CUDA FlashAttention build covers 50 standard cache pairs and 15 KVarN fast-decode pairs, including the homogeneous F16 and BF16 pairs needed by precision tails. Add `-DGGML_CUDA_FA_ALL_QUANTS=ON` to compile all 169 standard and 36 KVarN pairs, or `-DGGML_CUDA_KVARN=OFF` to build without KVarN kernels.

### Other Backends

Bee inherits llama.cpp backend support, including Metal, HIP, Vulkan, SYCL, BLAS, CANN, MUSA, OpenVINO, OpenCL, and RPC. Use the upstream-style build docs in [docs/build.md](docs/build.md) and backend-specific pages under [docs/backend](docs/backend).

## Common Commands

### Local CLI

```sh
llama-cli -m model.gguf
llama-cli -m model.gguf -cnv --chat-template chatml
llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p "Request: schedule a call at 8pm; Command:"
```

### OpenAI-Compatible Server

```sh
llama-server -m model.gguf --port 8080
llama-server -m model.gguf -c 16384 -np 4
llama-server -m model.gguf -md draft.gguf
```

### DFlash Speculative Decoding

```sh
llama-server -m target.gguf --spec-type draft-dflash \
  --spec-draft-model drafter.gguf \
  --spec-draft-ngl all \
  --spec-dm-controller profit \
  --flash-attn on --cache-type-k q5_0 --cache-type-v q4_1
```

DFlash draft contexts remain on standard cache types. KVarN draft caches are
supported only for the draft-owned MTP route described below.

### KVarN Target Cache

```sh
# Balanced general starting point from the benchmark ladder
llama-server -m model.gguf --flash-attn on \
  --cache-type-k kvarn5 --cache-type-v kvarn4 \
  --kv-tail-tokens 1024
```

### KVarN MTP Draft Cache

Qwen3.5/Qwen3.6 dense and MoE MTP, and standalone Qwen3.8/Qwen4Exp MTP
sidecars, can select an independent draft-owned KVarN cache:

```sh
llama-server -m target.gguf --spec-type draft-mtp \
  --spec-draft-model mtp.gguf \
  --spec-draft-type-k kvarn4 --spec-draft-type-v kvarn2
```

The target and draft cache types are independent. A one-sided draft KVarN
selection promotes the other side to the same width with a warning. There is no
draft precision-tail option: the explicit draft tail request stays zero and
KVarN retains its intrinsic exact suffix of up to 128 tokens. Gemma 4 MTP shares
the target cache, so configure target `--cache-type-k/v kvarn*` instead of a
draft cache type. DFlash, DSpark, Eagle3, draft-simple, and unclassified MTP
architectures reject explicit draft KVarN requests.

### Router Mode With Presets

```sh
llama-server --models-dir /path/to/models
llama-server --models-preset presets.ini
```

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

- [BeeLlama features and public repo diff](docs/beellama-features.md)
- [BeeLlama args reference](docs/beellama-args.md)
- [Build docs](docs/build.md)
- [Server docs](tools/server/README.md)
- [Docker docs](docs/docker.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

Keep PRs small and scoped. Run the narrowest relevant tests or benchmarks before opening a PR, and include the exact commands. For fork-specific changes, update the corresponding docs when behavior or args change.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for inherited llama.cpp contribution conventions and this fork's AI usage policy.

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - HTTP client/server library used by `llama-server` - MIT
- [stb-image](https://github.com/nothings/stb) - single-header image decoder used by multimodal code - public domain
- [nlohmann/json](https://github.com/nlohmann/json) - single-header JSON library - MIT
- [miniaudio.h](https://github.com/mackron/miniaudio) - single-header audio decoder - public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - process launching helper - public domain
- [Intel OpenVINO](https://github.com/openvinotoolkit/openvino) - frontend header used in OpenVINO backend (`ggml/src/ggml-openvino/openvino/frontend.h`) - Apache-2.0
- Intel SYCL/oneAPI - SYCL backend (`ggml/src/ggml-sycl/`) - Apache-2.0 WITH LLVM-exception

See the `licenses/` directory for full license texts.

[![Support my work!](https://anbeeld.com/images/support.jpg)](https://anbeeld.com/support)
