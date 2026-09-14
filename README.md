# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

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

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

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

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
