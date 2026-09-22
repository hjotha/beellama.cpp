# hjotha's BeeLlama.cpp

![BeeLlama.cpp logo](beellama.jpg)

BeeLlama.cpp (or just Bee) is a performance-focused llama.cpp fork for squeezing more speed and context out of local GGUF inference. It adds variance-normalized KV-cache quantization (KVarN), KV cache precision tail for recent tokens, low-bit cache types, adaptive draft control for speculative decoding, reasoning-loop protection, and more.

This repository combines [Anbeeld's BeeLlama](https://github.com/Anbeeld/beellama.cpp), the serving and memory-management extensions from [hjotha/llama.cpp](https://github.com/hjotha/llama.cpp), and PrismML/Bonsai weight-format support. The inventory below describes the source at `712c8f9ea` (2026-09-20); enabled features depend on the model, build backend and launch options.

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
- **Reasoning-loop protection and realtime control**: detect repeated hidden reasoning, stop or close it, and let opted-in clients end reasoning through a live control request.
- **INI model presets and KLD tooling**: configure reusable model profiles and compare logits, perplexity and cache quality at the actual serving batch/ubatch.
- **PrismML/Bonsai weights**: PQ2_0/PTQ1_0, folded Hadamard/FWHT execution and compatible MTP/DFlash integration.
- **Serving extensions from this fork**: adaptive resident contexts, unified disk snapshots, cross-tier q4_0-to-KVarN conversion, CUDA recovery, paged KV/SnapKV and AMD/NVIDIA power controls, detailed below.

For the full feature and public-repo comparison, read [docs/beellama-features.md](docs/beellama-features.md). For the complete argument reference, read [docs/beellama-args.md](docs/beellama-args.md).

## About this fork

The combined fork adds resident adaptive MTP profiles, context-based routing, shared prompt snapshots and KV conversion, automatic disk prompt/KV persistence, CUDA memory recovery, NVIDIA and AMDGPU governors, Ryzen APU TDP control, PrismML/Bonsai quantization and experimental paged KV/SnapKV serving. These are selectable capabilities, not a claim that every combination is supported.

Standard llama.cpp tools remain available: local CLI/chat, GGUF conversion and quantization, benchmarking, perplexity/KLD evaluation, and a web UI. The [HTTP server](tools/server/README.md) provides streaming chat/completions, Responses and Anthropic Messages compatibility, embeddings, reranking, infill, tokenization, templates, model routing, slots and Prometheus metrics. Multimodal inputs, tool calling, structured output/grammars and LoRA depend on the selected model/template and serving mode. See [model implementations](src/models), [conversion tools](convert_hf_to_gguf.py), [quantization](tools/quantize/README.md) and [multimodal tools](tools/mtmd/README.md).

- [PrismML, Bonsai and calibration](#prismml-bonsai-and-calibration)
- [Bee cache and generation controls](#bee-cache-and-generation-controls)
- [Verified Qwen3.5 deployment](#verified-qwen35-deployment-2026-09-20)
- [Adaptive context and cache reuse](#adaptive-context-with-resident-weights)
- [Automatic disk prompt cache](#automatic-disk-prompt-cache)
- [Router mode](#context-based-router-mode)
- [CUDA, NVIDIA, AMDGPU and Ryzen controls](#cuda-memory-recovery-and-gpu-governors)
- [Paged KV, SnapKV and speculation](#paged-kv-snapkv-and-other-speculation)
- [Validation and limitations](#validation-and-operational-history)

### PrismML, Bonsai and calibration

| Capability | Implemented behavior |
| --- | --- |
| Prism quantized weights | `PQ2_0` uses 128-weight blocks at 2.125 bits/weight; `PTQ1_0` packs ternary weights in base 3 at 1.75 bits/weight. These are model-weight formats, distinct from Bee's low-bit KV cache types. |
| CPU and CUDA paths | Quantization/dequantization, row access and matrix/vector kernels are integrated with GGUF model loading. Availability and speed depend on hardware and the built kernels. |
| Hadamard/FWHT | Reads `prism.hadamard.*` metadata and applies the corresponding activation transforms for folded weights, including wide FWHT kernels and shared transforms. |
| Bonsai speculation | MTP embedding transforms account for the target's Hadamard rotation. DFlash graph construction handles rotated target embeddings/output heads and encoder input width. A compatible drafter is still required. |
| Vulkan integration | Includes PTQ1_0 dequantization, row access/matvec and FWHT pipeline support. Commit `2089183b7` fixes the missing PTQ1 `get_dm` helper and FWHT pipeline state. This does not claim PQ2_0 Vulkan parity or a full Bonsai Vulkan validation. |
| Other backends | Metal contains Prism quantization/FWHT implementations, but this integration was not validated on Apple hardware in this README update. |
| Q4 K-cache calibration | `--kv-mean-center bias.gguf` subtracts a precomputed per-head/channel K bias before q4_0 cache quantization. Generate it with `llama-kv-mean-center`, using matching model/cache rotation settings. |

The mean-centering feature supports standard-attention caches, including supported SWA/hybrid attention layouts; it requires `--cache-type-k q4_0` and does not cover recurrent-only or MLA/DSA memory. The bias file records the calibration basis and incompatible bases are rejected. See [mean-centering](docs/kv-mean-center.md) and its [calibration tool](tools/kv-mean-center/README.md).

Sources: [quantization types](ggml/include/ggml.h), [Qwen3.5 graph](src/models/qwen35.cpp), [DFlash graph](src/models/dflash.cpp), [Vulkan backend](ggml/src/ggml-vulkan/ggml-vulkan.cpp) and the [Prism integration record](docs/prismml-integration.md). The record's original Vulkan build failure predates `2089183b7`; its claim that Vulkan support is absent is historical. Its backend and benchmark notes are not a fresh cross-platform qualification.

### Bee cache and generation controls

- **KVarN:** independent K/V widths `kvarn2/3/4/5/6/8`, group transforms/normalization, exact sink/suffix and native CPU/CUDA/HIP/Vulkan attention where supported. Vulkan direct attention requires shader Int64 and buffer-device-address support; 64-dimensional K/V heads are qualified on CPU/CUDA, not Vulkan. SWA precision can be overridden with paired `--cache-type-k-swa` / `--cache-type-v-swa`.
- **Exact precision tail:** `--kv-tail-tokens` accepts a count, `auto`, group lists or role mappings such as `full=1024,swa=1024`; `--kv-tail-type` chooses `f16` or `bf16`. Standard caches default to no tail and BF16 when enabled. KVarN retains its intrinsic exact suffix of up to 128 tokens even at `0`, defaults to F16, and rounds positive lengths to 128-token groups. Target tail settings are not inherited by draft contexts.
- **Draft-owned KVarN:** independent `--spec-draft-type-k` / `--spec-draft-type-v` are runtime-qualified on CUDA for draft-simple, EAGLE3, allowed MTP models, DFlash1/2 and non-MLA DSpark. Non-causal DFlash keeps compressed storage but uses materialized attention. Shared Gemma 4 MTP uses the target cache; unsupported MTP/MLA layouts and n-gram modes reject independent draft KVarN. HIP/Vulkan DFlash-family draft KVarN remains unqualified.
- **CUDA prefill workspace:** `GGML_KVARN_WINDOW_CHUNK` requests a transient materialization window (default 65,536 tokens), capped by active KV length and available VRAM. It controls temporary workspace, not persistent context capacity.
- **Shared KV capacity:** `--kv-unified-per-slot N` sets a per-slot context limit; without an explicit `--ctx-size`, the shared pool is sized to `parallel * N`. This is separate from paged KV and is rejected by adaptive resident mode.
- **DFlash profit controller:** `--spec-dm-controller profit|off` adjusts draft depth using observed speculative throughput against a no-spec baseline. Use `--spec-type draft-dflash` with a compatible `--spec-draft-model`; the bare `dflash` alias is not supported.
- **Reasoning loop guard:** `--reasoning-loop-guard force-close|stop|off` defaults to `force-close`; token/window/coverage/period/intervention settings tune it. Force-close needs a chat template with a usable reasoning-end sequence.
- **Realtime reasoning control:** opt in with request JSON `"reasoning_control": true`, then send `{"id":"chatcmpl-...","action":"reasoning_end"}` to `POST /v1/chat/completions/control` while that completion is running. It is not a general edit endpoint.
- **Presets and quality measurement:** INI model profiles support startup/loading policies and `POST /models/reload`. `llama-perplexity --save-all-logits FILE` and `--kl-divergence-base FILE --kl-divergence` compare against a saved baseline; use the same corpus, context, batch and ubatch.
- **Observability:** cache/tail statistics distinguish prefix matches, planned/reprocessed tokens, cache source/reason and tail allocation. Server metrics and slot endpoints complement benchmark/KLD tools.

See the [argument reference](docs/beellama-args.md), [feature guide](docs/beellama-features.md), [presets](docs/preset.md) and [speculation implementation](common/speculative.cpp). Some inherited feature-guide text still describes draft KVarN as MTP-only; the current route qualifications above reflect the newer implementation. Removed Bee systems such as TurboQuant/TCQ, DDTree and CopySpec are not part of this inventory.

### Verified Qwen3.5 deployment (2026-09-20)

The development checkout is `/home/hjotha/beellama.cpp` on GOKAYA; its branch is `main` and `origin` is `hjotha/beellama.cpp`. The verified Qwen3.5 instance runs from `/home/hjotha/beellama-vulkan-explain-build/bin/llama-server`, reporting `0.4.7-dev`, build `12275`, commit `712c8f9ea`. The user service is `qwen35-4b-mtp-8092.service`.

| Setting | Observed configuration |
| --- | --- |
| Model / public alias | `Qwen3.5-4B-MTP-Q4_K_M.gguf` / `qwen-3.5-4b-mtp-vulkan` |
| Backend / port | Radeon APU, `--device Vulkan0`, port `8092` |
| Context / slots | `--ctx-size 2048 --parallel 2`; HTTP slots report **1,024 tokens per slot** |
| Batch / ubatch / CPU threads | `256 / 256 / 8` |
| Target and draft KV | `q4_0 / q4_0`, FlashAttention enabled |
| Speculation | `draft-mtp`, draft maximum `2`, acceptance cutoff `0.80` |
| RAM prompt cache / checkpoints | `--cache-ram 0 --ctx-checkpoints 1` |
| AMD governor | `--gpu-power-backend amdgpu --apu-tdp 20` |
| Graphics/fabric clocks | Automatic; no clock or fabric-state flags in the active command |
| Adaptive resident switching / paged KV | Not enabled for this Vulkan instance |

Read-only verification returned `/health: {"status":"ok"}`, two slots with speculation enabled, and the model path above. This is a dated configuration snapshot, not a throughput benchmark or a new validation of every feature. The older Qwen3.8 CUDA/adaptive configuration under `/home/hjotha/llama` belongs to the imported fork's operational history; it is not this Qwen3.5 launch.

### Adaptive context with resident weights

Enable this mode with `--ctx-size-mtp N`; the default `0` preserves ordinary server behavior. `--ctx-size` sets the long capacity. `--mtp-max-tokens` sets the medium-profile budget threshold (defaults to `--ctx-size-mtp`). `--ctx-size-mtp-short` sets the short-profile context size, `--mtp-short-max-tokens` sets its budget threshold, and `--spec-draft-n-max-short` sets the draft N for the short profile (default: 4).

The following is the historical three-profile CUDA example, with local paths replaced by placeholders; size it for the selected model and available VRAM:

```sh
GGML_DISABLE_VULKAN=1 GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB=18 \
./build/bin/llama-server \
  --model /path/to/qwen35-family-mtp.gguf \
  --ctx-size 97536 --ctx-size-mtp 56320 --mtp-max-tokens 56320 \
  --ctx-size-mtp-short 32768 --mtp-short-max-tokens 32768 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-max-short 4 \
  --spec-draft-p-min 0.80 --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 \
  --device CUDA0 --gpu-layers 99 --parallel 1 --fit off \
  --batch-size 256 --ubatch-size 256 --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --cache-ram 2048 --ctx-checkpoints 1 --load-mode none \
  --slot-save-path /path/to/dedicated-slot-cache/ --slot-save-auto \
  --no-context-shift --metrics --host 127.0.0.1 --port 8090
```

Create the dedicated cache directory first and use a free port. GPU governors are optional and described below.

The source also supports optional fourth/fifth profiles, `xlong` and `xxlong`, through `--ctx-size-xlong` / `--ctx-size-xxlong` and `--xlong-max-tokens` / `--xxlong-max-tokens`. Their context sizes and thresholds must follow the preceding tiers. LONG and XL may keep MTP enabled with `--spec-draft-n-max-long N` / `--spec-draft-n-max-xlong N`; XXL remains target-only (`--spec-draft-n-max-xxlong` must be zero). Each has independent batch/ubatch controls, such as `--batch-size-xlong` and `--ubatch-size-xlong`, to reduce workspace at larger contexts. Short aliases `--ctx-size-s`, `--ctx-size-m`, `--ctx-size-l`, `--ctx-size-xl` and `--ctx-size-xxl` are available. Long/XL/XXL also expose per-profile K/V type controls, such as `--cache-type-k-xlong` and `--cache-type-v-xlong`; representation changes remain subject to restore compatibility. These options do not establish a hardware-independent context ceiling.

**Selection and lifecycle (three-profile example above)**

- Selection counts the complete formatted prompt, including cached tokens, plus the normalized output reserve. With no finite request/server output limit, 4,096 tokens are reserved for selection only; this does not impose a new generation limit.
- If total budget <= 32,768 tokens, the server selects the short profile (`mtp-short`, N=4).
- If total budget is between 32,768 and 56,320 tokens, the server selects the medium profile (`mtp`, N=2).
- If total budget exceeds 56,320 tokens, the server selects the long profile (`long`, N=`--spec-draft-n-max-long`, or target-only when zero, up to 97,536 tokens).
- An explicit budget above the largest enabled capacity is rejected before switching.
- The server boots in the short profile (`mtp-short`) and switches between requests after active work is drained.
- Transitions preserve model weights in GPU memory. The MTP head weights are moved between VRAM and host RAM only when crossing between an MTP profile and a target-only profile.
- Transition failures attempt rollback to the previous profile. If rollback also fails, the server publishes `state=unavailable` and rejects inference with HTTP 503.

**RAM cache and explicit slot files across profiles**

The adaptive RAM prompt cache carries target state, draft/speculative state and checkpoints across profile switches:
- Layout and attention semantics: `common_prompt_cache_layout()` checks tensor layout semantics rather than buffer capacity, so cached prefixes remain compatible across profile transitions.
- Automatic cache preservation: when a profile switch occurs, `slot.prompt_save(*prompt_cache)` serializes valid prefixes and checkpoints into the global RAM cache (`--cache-ram`) before tearing down the old context. The new profile context rehydrates compatible prefixes via `slot.prompt_restore`, avoiding repeated prefill.
- Explicit slot persistence: `POST /slots/{id_slot}?action=save` and `?action=restore` record profile identifiers (`0` = medium MTP, `1` = long, `2` = short MTP). When restoring a snapshot from a different profile, the server automatically triggers a dynamic transition (`switch_adaptive_context(snapshot.profile)`), reloads MTP weights to GPU if needed, and applies the saved KV state.

`adaptive_context` in `/props`, `/models` and `/slots` exposes `enabled`, `profile`, `state`, `context_size`, `context_size_long` and `mtp_weights_resident`.

**Supported scope:** dense Qwen35-family MTP models, with one MTP head, one CUDA device, one slot, traditional KV and `fit=off`. CPU lifecycle checks are supported with `--gpu-layers 0`. Adaptive mode rejects external draft models, other speculative backends, paged KV, `--kv-unified-per-slot`, multimodal input, LoRA/control vectors and automatic sleep. These restrictions apply to adaptive mode, not to all upstream server features.

See the [server usage guide](tools/server/README.md), [lifecycle and selection code](common/common.cpp), [server transitions and slot persistence](tools/server/server-context.cpp), [RAM cache implementation](tools/server/server-task.cpp) and [model identity implementation](tools/server/server-model-identity.cpp).

### Automatic disk prompt cache

Opt-in `--slot-save-auto` saves evaluated prompt/target KV state in a dedicated `--slot-save-path`, allowing compatible prefixes to be reused across requests and process restarts.

| Option | Default | Meaning |
| --- | --- | --- |
| `--slot-save-auto` | Disabled | Enables automatic save/restore; requires a slot-save directory. |
| `--slot-save-block N` | 256 | Token-ID block size used for prefix indexing. |
| `--slot-save-max-count N` | 64 | Maximum retained snapshots; `0` removes this limit. |
| `--slot-save-max-mb N` | 32768 | Store budget in MiB; `0` removes this limit. Oversized publications are rejected. |

Environment equivalents are `LLAMA_ARG_SLOT_SAVE_AUTO`, `LLAMA_ARG_SLOT_SAVE_BLOCK`, `LLAMA_ARG_SLOT_SAVE_MAX_COUNT` and `LLAMA_ARG_SLOT_SAVE_MAX_MB`. Retention limits also apply to manual saves. `--cache-ram` controls the RAM prompt cache, not total process/system memory.

**Unified file format.** Automatic caching and the streaming upward router handoff share a canonical native target-state file with a bounded, checksummed manifest and optional embedded regeneration logits. New writes do **not** create separate `.meta` / `.logits` sidecars. Publication uses a private temporary file, sync and atomic rename; failed preflight/publication leaves a previously valid destination intact. Explicit adaptive slot files retain their separate format with profile, draft/carry and checkpoints.

**Identity and restore.** SHA-256 model identity covers actual GGUF shards and effective metadata overrides, with a file-stamp-keyed sidecar caching the model digest. Snapshot lookup checks tokens, model identity and effective attention/cache/RoPE layout; batch and capacity may differ when compatible. Payload validation uses the selected file's checksum. The streaming loader requires an empty destination, publishes state only after successful validation and clears it on a late failure. An 8 MiB transfer buffer bounds payload staging; metadata, token vectors, context storage and reservations still consume memory proportional to context.

**Recurrent and MTP semantics.** Saved prefixes contain evaluated tokens, so the last sampled token may need evaluation on continuation. FULL/recurrent state cannot rewind arbitrary prefixes. A managed MTP prompt branch point can save the N-1 target prefix, then restore it and decode a real suffix to rebuild draft/carry before speculation resumes. This special bootstrap is gated on managed weights; ordinary unmanaged MTP does not use that reconstruction path. Exact no-MTP prompt hits can use validated embedded logits without prompt evaluation. Missing/incompatible state or unavailable speculative reconstruction falls back to normal prefill.

**Store boundaries.** Use a dedicated directory with trusted local writers: retention owns the snapshots, and checksums detect corruption rather than authenticate writers. Requests disabling prompt caching, embeddings/reranking and media prompts are excluded from automatic reuse; adapters/control vectors disable automatic persistence when identity cannot safely cover them. Disk I/O can delay queued work. This is a best-effort cache, not a durable conversation archive or a guarantee of cache hits.

The implementation and current format are documented in [unified KV snapshots](tools/server/README-router-state-streaming.md), [server context](tools/server/server-context.cpp) and [model identity](tools/server/server-model-identity.cpp). Existing [slot tests](tools/server/tests/unit/test_slot_save.py), [streaming tests](tests/test-state-file-stream.cpp) and [restart harness](tools/server/tests/router_state_auto_restart.py) exercise restore/fallback behavior. Older records using separate sidecars describe earlier formats.

### Context-based router mode

The earlier multi-process design remains available as an alternative to resident switching:

- Preset entries can share a `route-group` public name and use `route-max-tokens` thresholds. The router chooses the smallest fitting tier for prompt plus requested output, with a byte estimate on cold start until tokenization is available.
- Selection is recalculated for each request. Conversations can return to a smaller tier; the old README's claim that conversation pinning prevents demotion is obsolete.
- `--models-max 1` limits loaded children. Swaps wait for active work, and cold-load request reservations protect concurrently arriving requests.
- `X-Conversation-Id` associates upward state handoff between child processes. With a persistent store, canonical snapshots survive child/router shutdown and remain subject to the shared retention policy; route cleanup drops transient references/files. Reverse interchild migration is not implemented.
- A fixed no-MTP tier can stream a compatible target snapshot without materializing a second full payload in RAM. Invalid identity/layout/checksum/capacity falls back to prefill.
- An explicit compact tier can convert `q4_0/q4_0` target snapshots to KVarN through `llama_state_seq_convert_file` / `llama_state_seq_convert_data`. Conversion is CPU-side with bounded groups/workers, preserves hybrid recurrent/conv bytes, and records provenance in the persisted result. Converted snapshots can be reused after restart without converting again.
- Conversion accepts one stream and one contiguous sequence at positions `0..n-1`, with untransposed q4_0 K/V and matching row/layout metadata. It is **lossy** and is not equivalent to a native KVarN prefill; arbitrary cache-format conversion is not supported.
- Chat/Responses compatibility and both OpenAI/Codex model-list payloads remain available in router mode.

Router swaps unload/load child processes; adaptive mode instead rebuilds contexts inside one model process. See [snapshot/handoff/conversion contract](tools/server/README-router-state-streaming.md), [conversion tests](tests/test-state-convert-q4-kvarn.cpp), [router implementation](tools/server/server-models.cpp), [server options](tools/server/README.md) and the [historical router design and measurements](docs/mtp-router-split-plan.md).

### CUDA memory recovery and GPU governors

| Feature | Behavior and controls |
| --- | --- |
| CUDA Graph stability | Handles changing prompt shapes, graph capture/instantiation failure and stale graph cleanup. Recoverable pressure can fall back to ordinary kernels and re-enable graphs after memory recovers. |
| Preventive graph margin | `GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB` controls the free-VRAM check; **source default: 18 MiB**. It does not reserve VRAM or make every OOM recoverable. |
| CUDA allocator recovery | Restores recoverable VMM-pool OOM handling and bounds retries/allocation behavior. |
| IQ1_M workspace and kernels | Bounds full-dequantization workspace via MMVQ, adds MMQ/tensor-core support and initializes only the selected MMQ variant. |
| KVarN prefill workspace | CUDA materialization windows are chunked according to free VRAM, avoiding an oversized temporary buffer during prompt processing. |
| Backend isolation | Presence of `GGML_DISABLE_VULKAN` disables Vulkan registration/initialization, even when its value is `0`. Omit it for Vulkan workloads. |

The phase-aware governor runs inside the server, deduplicates unchanged settings, and restores controls on supported idle/shutdown paths. Backend selection is explicit: **`auto` currently selects NVML; it does not discover AMD automatically.**

| Parameter | Environment variable | Meaning |
| --- | --- | --- |
| `--gpu-power-backend auto\|nvml\|amdgpu` | `LLAMA_ARG_GPU_POWER_BACKEND` | NVML for NVIDIA; select `amdgpu` for Linux sysfs/Ryzen controls. |
| `--gpu-power-device N` | `LLAMA_ARG_GPU_POWER_DEVICE` | NVML index, or index in the sorted list of AMD DRM cards; default `0`. Independent of the inference `--device` selector. |
| `--gpu-power-prefill W` | `LLAMA_ARG_GPU_POWER_PREFILL` | NVIDIA power limit during prompt processing; requires the decode option. |
| `--gpu-power-decode W` | `LLAMA_ARG_GPU_POWER_DECODE` | NVIDIA power limit during generation; requires the prefill option. |
| `--gpu-mem-clock-prefill MHz` | `LLAMA_ARG_GPU_MEM_CLOCK_PREFILL` | NVIDIA **memory** clock or AMDGPU **graphics SCLK** during prompt processing. |
| `--gpu-mem-clock-decode MHz` | `LLAMA_ARG_GPU_MEM_CLOCK_DECODE` | NVIDIA **memory** clock or AMDGPU **graphics SCLK** during generation. |
| `--gpu-fabric-state N` | `LLAMA_ARG_GPU_FABRIC_STATE` | AMD raw fabric DPM state index `0..31` during prefill/decode; omit to leave automatic. |
| `--apu-tdp W` | `LLAMA_ARG_APU_TDP` | Ryzen APU STAPM, fast and slow limits, each set to W while active; restores each original value at idle. |

**NVIDIA:** power options must be positive and supplied together. Idle retains the last power limit; shutdown/sleep restore the original. Memory-clock options can operate independently; idle and phases without a target release the applied lock/offset. Above-stock targets use supported locks plus a bounded offset where the driver allows it. Core/SM clocks are not controlled by these NVIDIA server flags; an external `nvidia-smi --lock-gpu-clocks` is a separate driver control. GPU/driver permissions and supported ranges still apply.

**AMD graphics and fabric:** despite their historical names, the `--gpu-mem-clock-*` options write SCLK through `pp_od_clk_voltage`, not a standalone VRAM/MCLK target. The backend checks OD ranges, commits writes and restores the saved SCLK range/performance level. `--gpu-fabric-state` writes `pp_dpm_fclk`, which controls coupled fabric/memory states; raw kernel indices can differ from the displayed list. Fabric control requires an original non-manual performance level so it can be restored. Write failures trigger rollback attempts; restoration failures are logged. Generic GPU power caps and NVIDIA-style clock offsets are unsupported by the AMD backend.

**Ryzen APU TDP:** requires Linux, `libryzenadj.so`, Ryzen SMU access and explicit `--gpu-power-backend amdgpu`. It cannot be combined with `--gpu-power-prefill` or `--gpu-power-decode`. It can be used without graphics/fabric locks, as in the verified Qwen3.5 deployment:

```sh
# Append to a compatible Vulkan Qwen3.5 server command:
# --gpu-power-backend amdgpu --apu-tdp 20
# Optional graphics targets, only after checking the device's OD range:
# --gpu-mem-clock-prefill 2700 --gpu-mem-clock-decode 2700
# Optional coupled fabric/memory state, after checking raw kernel indices:
# --gpu-fabric-state 0
```

Programmed power limits are firmware settings, not a guarantee that instantaneous measured package power stays below the same number. A fixed clock is not a demonstrated speedup for every workload. Normal idle/shutdown restoration cannot run after an uncatchable kill or power loss.

See [GPU governor implementation](tools/server/server-gpu-power.cpp), [governor tests](tests/test-server-gpu-power.cpp), [server usage](tools/server/README.md), [CUDA allocator/graph handling](ggml/src/ggml-cuda/common.cuh), and the historical NVIDIA [power](docs/phase-aware-nvidia-gpu-power-governor.md) / [memory-clock](docs/phase-aware-nvidia-gpu-memory-clock-governor.md) design documents.

### Paged KV, SnapKV and other speculation

These optional paths remain in the fork; they are not enabled in the Qwen3.5 deployment above.

| Capability | Options / implementation |
| --- | --- |
| Shared physical KV pool | `--kv-paged` shares blocks across sequences/slots. `--kv-block-size`, `--n-gpu-blocks` and `--n-cpu-blocks` control block size and physical capacity. |
| Elastic growth and migration | `--kv-paged-dynamic`, `--n-gpu-blocks-initial` and `--n-gpu-blocks-growth` separate initial allocation from growth, with attention-layer migration across registered GPU backends and optional CPU spill. |
| Automatic pool fitting | `--kv-paged-prealloc-max` calibrates capacity after model load, accounting for model, compute, hybrid/recurrent and MTP requirements, and caps dynamic growth. |
| Request admission | `--kv-paged-admission-blocks` and `--kv-paged-watermark` gate physical demand instead of treating logical context size as unlimited memory. Over-budget requests are rejected cleanly. |
| Multi-device placement | `--paged-attn-cuda` pins full-attention layers to the first offload device while tensor splitting can place recurrent-only layers elsewhere. Paged paths include shared multi-sequence batches and meta-buffer handling. |
| SnapKV selective retention | `--snapkv OW`, `--snapkv-retention`, `--snapkv-recent`, `--snapkv-pinned` and `--snapkv-budget-blocks` score and retain selected old pages while protecting configured leading/trailing windows. Streaming, episodic, per-head and per-sequence scoring paths and eviction timing are present. |
| Paged backends | CUDA/CPU paths plus Vulkan quantized paged attention, parallel prefill and Vulkan 1.1 Android compatibility work. Backend support does not imply equal speed or stability on every device. |
| Traditional KV fitting | Model-specific, compute-aware context fitting accounts for hybrid/MTP state. Use `--fit off` when supplying explicitly calibrated contexts. |
| DFlash2 | Separate draft-model speculation path with `p_min` and stochastic verification support; requires a matching drafter and model/backend validation. Adaptive resident mode supports only `draft-mtp`. |

SnapKV retention is lossy and experimental: reducing physical KV can change output quality. Paged KV is not automatically faster than traditional KV; older Vulkan iGPU experiments include slowdown and failure cases. Context, pool sizes and throughput need validation for the selected model, backend and workload.

Use the [paged example and controls](examples/paged/README.md), [SnapKV episode benchmark](tools/snapkv-episode-bench.py), [40K/MTP benchmark](tools/snapkv-40k-mtp-bench.py), [argument definitions](common/arg.cpp), [speculation implementation](common/speculative.cpp), [KV calibration findings](docs/kv-calibration-findings.md) and [Vulkan iGPU notes](docs/notes-paged-kv-vulkan-igpu.md).

### Validation and operational history

The [adaptive implementation plan and dated evidence](plans/001-adaptive-context-resident-weights.md) records CPU lifecycle/cache tests, sanitizer runs, real CUDA short/long/short transitions, maximum-context requests, repeated cycles, fault/rollback checks and subsequent production fixes. Recorded 20-pair CUDA runs kept one main-model instance/load through 40 completions; those serial results are workload-specific, not a general concurrency or latency guarantee.

Reproducible coverage lives in [adaptive HTTP tests](tools/server/tests/unit/test_adaptive_context.py), [state/lifecycle tests](tests/test-save-load-state.cpp), [prompt-cache tests](tests/test-server-prompt-cache.cpp), [model-identity tests](tests/test-server-model-identity.cpp) and the [Gauntlet harnesses](gauntlet). The plan still records open work on detailed physical profiling, quantitative comparison with the router, broader concurrency, additional sanitizer coverage, persistence and rollback/soak validation. Implemented features and a successful deployment do not close all those gates.

The [previous production capture](gauntlet/production/production-final-20260914.txt), [unit snapshot](gauntlet/production/llama-server-root.service.final), [router report](docs/mtp-router-split-plan.md), [calibration notes](docs/kv-calibration-findings.md) and [OOM trace](docs/oom-reproduction-trace.md) are dated evidence from earlier releases. Their release paths, context ceilings, clock settings and free-VRAM figures are not current deployment facts. Historical passing tests also do not prove every newly combined backend/model feature.

## KV Cache Quantization

K and V cache types are set independently with `--cache-type-k` and `--cache-type-v`. The research is covered in articles: [KVarN KV Cache: Implementation and Benchmarks](https://anbeeld.com/articles/kvarn-kv-cache-implementation-and-benchmarks), [KV Cache Precision Tail: Implementation and Benchmarks](https://anbeeld.com/articles/kv-cache-precision-tail-implementation-and-benchmarks), and [KV Cache Quantization Benchmarks: KVarN, Precision Tail](https://anbeeld.com/articles/kv-cache-quantization-benchmarks-kvarn-precision-tail).

### KV Cache Recommendation Ladder

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

The following binaries and containers are distributed by **Anbeeld's BeeLlama**, on its [releases page](https://github.com/Anbeeld/beellama.cpp/releases). They must not be assumed to contain this fork's additional commits. Build this checkout from source to use the combined feature set.

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

Anbeeld's Docker images are published to `ghcr.io/anbeeld/beellama.cpp`:

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

### Vulkan Build

```bash
cmake -S . -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --parallel 2
```

Use a Vulkan SDK/development environment as described in the build guide. Shader compilation can consume substantial RAM; limit parallel jobs on shared-memory APUs. This builds the source without changing any running service.

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

Standard draft cache types remain a portable starting point. The qualified CUDA
DFlash-family route also accepts draft KVarN, with materialized attention; see
[Bee cache and generation controls](#bee-cache-and-generation-controls).

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
draft cache type. CUDA-qualified draft-simple, EAGLE3, DFlash1/2 and non-MLA DSpark can also use
independent draft KVarN. Unsupported MTP/MLA layouts and modes without an owned
KV context reject it; see the qualifications above.

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
- [XCFramework build script](build-xcframework.sh)
- [Completions](tools/completion/README.md)
- [Model implementations](src/models)
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
