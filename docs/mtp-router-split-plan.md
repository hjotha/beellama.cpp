# Plan: one public model name, two context profiles from the same GGUF

## Current deployment - CUDA Graph headroom and slot cleanup, 2026-09-08

GOKAYA `8090` runs `/home/hjotha/llama-releases/graph-headroom-20260908/build/bin` under `llama-server-root.service`. It uses the `de57d0269` launcher, CUDA/model/GGML libraries and a rebuilt `libllama-server-impl.so` containing the failed-load snapshot cleanup. The release manifest proves that only this server library changed. Its SHA-256 is `fd389edd1c96045b722609c5855113248f2ce4e861c54ecd8264fb780d37c20a`.

| Profile | Context | Batch / ubatch | Validated input + output | Stable free VRAM |
| --- | ---: | ---: | ---: | ---: |
| MTP | 60,416 | 64 | 56,320 + 4,095 | 8 MiB |
| No MTP | 97,536 | 512 | 93,440 + 4,096 | 4 MiB |

Both contexts and the MTP route threshold were reduced by 768 tokens. These are the largest clean settings tested on the 256-token grid below, with the same GPU ownership and production settings. This is a workload-bounded validation, not an absolute hardware ceiling. The small remaining VRAM margin is measured after graph and kernel warmup, not just after loading weights.

The preset remains `/home/hjotha/prod-two-tier.ini`: one child, one slot, traditional q4_0 KV, flash attention, fit off, load-mode none, cache-ram 2048 MiB, one context checkpoint, and unchanged MTP/batch/power settings. The memory-clock override was subsequently removed after the separate GSP incident below. The route threshold is 60,416 prompt plus requested output tokens; larger budgets use the 97,536-token no-MTP tier, subject to existing conversation pinning. Slot transfer remains enabled under `/dev/shm`.

### What caused the failure

The first graph failed in a fresh standalone process with no router, conversation ID or slot restore. At the old MTP limit, CUDA reported only 5.62 MiB free before instantiation. Reducing MTP by 256 and 512 tokens still failed at 13.62 and 17.62 MiB free. A successful MTP graph instantiation consumed 18 MiB; the corresponding no-MTP graph consumed 14 MiB.

Graph memory is not the entire budget. At no-MTP context 97,792, the first graph succeeded, but later widths exhausted VRAM. The existing attribute tracer reproduced successful first-time `cudaFuncSetAttribute` calls consuming another 2 MiB each, then error 2 with 1.62 MiB free at `launch_mul_mat_q`. Graph update-begin/update-complete pairs had unchanged free memory, and the temporary VMM pool remained mapped at 14 MiB in this trace. The exact internal CUDA allocation mechanism was not fully traced; the evidence identifies first-time function setup costs, not progressive graph-update leakage.

The identical 97,792-token sequence passed with `GGML_CUDA_DISABLE_GRAPHS=1` (21 completions). Keeping the graph's extra allocation at that limit removes the headroom needed by later shapes. The chosen limits keep graphs enabled; the recoverable OOM fallback remains compiled and available.

| Profile | Context tested | Result |
| --- | ---: | --- |
| MTP | 61,184 / 60,928 / 60,672 | First-request graph instantiate OOM; direct fallback completed |
| MTP | 60,416 / 60,160 | Varied shapes passed with graphs active |
| No MTP | 98,304 | First-request graph instantiate OOM; direct fallback completed |
| No MTP | 98,048 | Graph entered; fatal CUDA OOM on request 2 |
| No MTP | 97,792 | Graph entered; fatal CUDA OOM on request 14; reproduced with tracer |
| No MTP | 97,536 / 97,280 | Varied shapes passed with graphs active |
| No MTP, graphs disabled | 97,792 | Same 21-request sequence passed |

### Leak and save/restore checks

Each selected process completed 43 requests: 21 varied shapes, one maximum-input/4096-output request, one follow-up, and 20 identical uncached 4096-input/64-output requests. There were zero graph fallbacks and zero fatal CUDA errors. The maximum request evaluated every input token with context shifting disabled. `truncated=true` at the context edge reflects the generation boundary; measured input counts matched the complete request. MTP produced 4095 of the 4096 requested tokens.

During the 20 identical repetitions, free VRAM stayed at 8 MiB for MTP and 4 MiB for no MTP. Each model instantiated one graph and completed 75/59 graph updates respectively. Child RSS varied by 776 KiB within each repeat series after cache warmup. GPU usage returned to the same 13 MiB baseline when each process exited. No progressive VRAM leak was observed in this sample; this does not establish the absence of every possible leak.

The router A/B ran four cycles without conversation transfer and four with real transfer, using the same prefix/output. Both modes completed 16 graph instantiations with no OOM or fallback. All four transfers logged save and restore, reused 4164 cached tokens and left zero test snapshot files afterward. The no-MTP child ended each cycle with 10 MiB free without transfer and 22 MiB with transfer: restoring the prefix avoided prefill work and its extra temporary allocations. Restore was not the cause of the reproduced GPU failure.

A separate RAM/tmpfs leak was found and fixed: if target loading threw after a migration snapshot was saved, the snapshot and route-state entry survived until router cleanup. A CPU regression reproduced a retained 13284-byte snapshot after an injected missing-LoRA load failure returned HTTP 500. The fix discards saved state when `ensure_model_ready()` throws, then preserves the original error. The same regression and 17 other local router tests passed with the new server library and the original release dependencies. Three external/download tests were not selected.

### Separate GSP firmware incident and final mitigation

After the first successful deployment canaries, NVIDIA driver `610.57.04` reported GSP task exception / load access page fault (Xid 120), followed by recovery-required Xid 154, at 15:25:34 CEST. This occurred during the memory-clock governor's decode-to-idle reset operation. The next real request only surfaced the already-failed CUDA context at 15:27:18 as `unspecified launch failure` while synchronizing for `prompt_save`. It was not a CUDA OOM, and the cache-save stack was downstream of the earlier firmware failure. The internal GSP/driver root cause was not isolated.

`nvidia-smi --gpu-reset` returned Not Supported. A PCIe FLR of the RTX 4070 at `0000:65:00.0`, with NVIDIA unbind/rebind, recovered the device. The idle 8092 Vulkan compactor briefly had to stop because its backend enumeration held NVIDIA device handles; it was restarted and passed a real completion afterward. Restoring the previous `flr bus` reset-method list returned EINVAL, so the device remains selected for the supported `flr` method.

The final preset removes `gpu-mem-clock-decode=11001`, leaving memory clocks under driver control as a mitigation of this reset path. Power limits remain 200 W prefill / 165 W decode. CUDA Graphs, MTP, save/restore, the validated contexts and all other model settings remain enabled as described above. The 86-request capacity measurements and the first router A/B used the old clock setting; they are not a throughput benchmark of the final automatic-clock configuration.

After recovery, the complete production hash/context/cache/Chat/Responses/decode canary set passed again. Three additional requests, alternating MTP / no MTP / MTP after 12-second idle intervals, each generated 129 tokens; a final idle check and the 8092 completion passed. From recovery start, driver and service logs contained no new Xid, CUDA error, reset-required event or graph fallback. Evidence: `post-canary-kernel.log`, `post-canary-transition.log`, `gpu-reset.log`, `gpu-flr-recovery.json`, `reset-method-restoration.json`, `recovery-verification.json`, `recovery-kernel.log`, and `recovery-service.log`.

The harnesses now use automatic memory clocks by default. `LLAMA_GRAPH_MEMORY_CLOCK=11001` explicitly reproduces the earlier clock setting; it is not the current production default.

### Production verification and artifacts

Production validation checked actual loaded-library paths/hashes, effective slot contexts/MTP status, exact route budgets 60,416 and 60,417, model catalog context 97,536, real cache migration, Chat, Responses and sustained constrained decoding in both profiles. It completed with zero fallback and zero fatal CUDA errors. An earlier deployment attempt was automatically rolled back because a constrained-output canary exhausted its 128-token test budget; increasing only that canary budget to 256 allowed the complete validation to pass. A busy-service guard also deferred the retry until the active request finished. Those events were not treated as successful deployments.

Evidence directory: `/home/hjotha/graph-headroom-20260908/`.

- `probe-*.json`, `refine-*.json`, matching logs: context sweep and 256-token refinement.
- `trace-nomtp-97792.log`: first-time CUDA function allocation trace and fatal reproduction.
- `control-no-graphs-nomtp-97792.json`: same-context graph-off control.
- `confirm-mtp-60416.json`, `confirm-nomtp-97536.json`: all 86 final direct requests, token counts, timing and memory samples.
- `route-transfer-off.json`, `route-transfer-on.json`: repeated cold loads and actual slot migration.
- `slot-cleanup/`: fail-before log, pass-after logs, 18-test JUnit result, build, dependency and hash evidence.
- `production-verification.json`, `deployment-result.json`: final live deployment proof.
- `deployment-attempt1.json`, `deployment-attempt1-journal.log`: preserved rollback evidence.
- `preset.before.ini`, `candidate.ini`, `release-dropin.before.conf`: configuration before/after and rollback inputs.

Reproducible host-specific harnesses are `scripts/bench-cuda-graph-headroom.py` and `scripts/bench-route-graph-memory.py`. They run on GOKAYA as root, stop the production service while owning port 8095/GPU, and restore the service in `finally`. Set `LLAMA_GRAPH_TEST_BIN` to the release under test. `--sweep` probes the initial limits; `--confirm` reads `selected.json`, validates the full workload and lowers a failing candidate in 256-token steps. The route harness reads the same selected limits. Raw experiment failures remain evidence; only the final selected settings and corrected code passed the deployment gates.

## Historical deployment - upstream sync, 2026-09-08

GOKAYA `8090` now runs fork `de57d0269` with official upstream through `64e9bceb2`.
The release and shared libraries are in `/home/hjotha/llama-releases/de57d0269/build/bin`.
Cold-autoload reservations fix both integration races; 9 router tests passed,
including slot-state transfer. Real Qwen tier migration preserved cached tokens.

| Profile | Context | Batch / ubatch | Validated input + output |
| --- | ---: | ---: | ---: |
| MTP | 61,184 | 64 | 57,088 + 4,094 |
| No MTP | 98,304 | 512 | 94,208 + 4,096 |

This release passed 7 CPU/parser/governor tests, 130 CUDA comparisons, and
56 sequential Qwen requests at the contexts above. Production routing, cache,
Chat, Responses and loaded-library checks passed. The preset remains
`/home/hjotha/prod-two-tier.ini`, with `fit=off`, `cache-ram=2048`, one slot,
and the 61184-token MTP route threshold. `load-mode=none` avoids the host-RAM OOM seen during rapid mmap reloads; both maximum requests were repeated through the router with this setting. Slot transfer is active in this release.

Evidence directory: `/home/hjotha/upstream-sync-20260908/`.

## Corrected operational validation - CUDA MMQ, 2026-09-08

This section supersedes the earlier single-pattern calibration below. The original 61,440 MTP setting reproduced fatal CUDA OOM on the second request (9,954 + 265 tokens passed; 10,952 + 32 failed). Disabling CUDA Graphs reproduced the same failure with 1.625 MiB free. Successful first-time cudaFuncSetAttribute calls consumed memory in 2 MiB increments; this is evidence of late kernel allocation under capacity pressure, not demonstrated fragmentation. The NVIDIA driver also reserves 380 MiB outside the usable CUDA memory.

The one-line MMQ fix (fork commit `3955ac793`) configures only mul_mat_q<type, J, fallback>, instead of loading/configuring both fallback variants. It passed 139 numerical MUL_MAT checks for IQ1/IQ2/IQ3 against CPU. The patched 61,440 MTP and 98,560 no-MTP settings passed the original second request but failed later widths; the code saving alone does not validate that old ceiling.

| Profile | batch/ubatch | Context | Input tokens | Output tokens | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| MTP | 64 | 61,184 | 57,088 | 4,095 | 520.71 | 42.85 |
| No MTP | 512 | 98,304 | 94,208 | 4,096 | 686.54 | 19.76 |

Each selected process completed 28 sequential requests on port 8095 as root, including the incident-sized calls, 16 uncached prompts varying the batch remainder, a ctx-minus-4096 request with 4096 requested output, and another completion afterward. The maximum request runs after kernel warmup. Exact prompt IDs, cache accounting, argv, timing and errors are in the JSON files. An output of 4094-4096 is accepted at the context edge; truncated=true there does not imply lost input. CUDA Graph allocation warnings with successful direct-execution fallback remain recorded separately from fatal OOM.

Production keeps fit=off, cache-ram=2048, one slot, q4_0 KV, and the same batch/ubatch and MTP/power options. The MTP route cap is 61184; larger total prompt-plus-requested-output budgets select the 98304-token no-MTP profile, subject to existing conversation pinning. These are tested operational values for these workloads, not a guarantee for arbitrary GPU co-tenants or all allocation histories.

Evidence: /home/hjotha/cuda-oom-20260908/. Selected results: patched-mtp-long-ctx61184.json, patched-nomtp-long-ctx98304.json. Reproducible checks: run.py and numerical.log. Production canaries and loaded-library hashes are recorded separately in production-verification.json.

## Historical design and calibration notes

The status statements and ceilings below describe earlier experiments and
are superseded by the current deployment and operational validation above.

Status: **implemented, tested and in production on GOKAYA** since 2026-09-07 (sections 6, 13, 14;
the code is in `tools/server/server-models.{h,cpp}`, documented in `tools/server/README.md`).
Section 11, the KV-state transfer across a swap, is implemented in this branch and passed a
two-process local integration smoke; it is not yet promoted to GOKAYA. Written after tracing
`tools/server/server-models.{h,cpp}`, `common/arg.cpp` (preset-only keys), `common/preset.h`,
`common/common.cpp` (`common_fit_normal_kv_context`) and re-deriving the VRAM arithmetic from
`docs/kv-calibration-findings.md` (GOKAYA, `192.168.1.57`).

**Both ceilings are now measured, not derived** — see 1.1. The measurement killed the third
tier: the `-mtp` GGUF with MTP *off* reaches 105,472 tokens, so one file serves both tiers and
the second GGUF is not needed at all.

**This plan supersedes `docs/mtp-adaptive-context-plan.md`**, whose goal is arithmetically
impossible on this model/GPU — see section 1.

## 0. The client contract (the requirement that drives everything)

Clients are **not** changing. Each one is configured with:

- model name: `qwen-3.8-27b` — one name, always
- context: 80,000
- output: 4,096

Whether a request is served with or without MTP must be invisible to them. That has three
hard consequences:

1. **The tier choice is server-side and automatic.** Letting the client name the tier
   (`qwen3-fast` / `qwen3-long`) is off the table as a deliverable — it survives only as a
   staging step to validate the children before the routing code exists (section 5).
2. **The single-process plan is dead, not just suboptimal.** Its ceiling is ~57,900 tokens
   (section 1); the clients are configured for 80,000. It cannot satisfy the contract at all.
3. **Under-estimating a request's size becomes a client-visible bug**, not a performance
   detail. If a 60k request is routed to the 60,928 child, the client gets an error or a
   truncation on a request its own config says is legal. So the exact `/tokenize` estimator
   (section 6) is mandatory, not an optimization.

Ceiling check: 80,000 (client) + 4,096 output = 84,096 fits inside the **measured** 105,472 of
the `-mtp` GGUF with MTP off (1.1), with a real 101,376 + 4,096 request behind that number. The
contract is servable, with ~21k of headroom.

### 0.1 Both tiers run off one file

| Member | GGUF | MTP | `ctx-size` | b/ub | Serves | Why it exists |
|---|---|---|---:|---:|---|---|
| `qwen3-fast` | `-mtp.gguf` | on | 60,928 | 64 | ≤ 56,000 budget | the speedup, ~44 decode tok/s |
| `qwen3-long` | `-mtp.gguf` | off | 105,472 | 64 | > 56,000, uncapped fallback | **same weights** → no style shift across the swap, KV state transfer legal (section 11), covers the whole 84,096 contract |

One GGUF, two children, two `ctx-size` values. Consequences, all of them good:

- **No second model file** to load, cache or keep in sync; a swap in either direction reads the
  same warm page cache.
- **No style shift** mid-thread — the weights are byte-identical, only the MTP head is idle in
  the long child.
- **Section 11's KV transfer is legal on the only hop there is**, since the structural checks
  (`src/llama-kv-cache.cpp:2535,2563,2572`) compare a state file written by the same weights.
- The old `qwen3-mid` / `qwen3-long` split and the second `...XXS.gguf` are **dropped**. They
  existed only because the `-mtp` file was believed to cap out near 77k.

## 1. Why one process cannot do it

Both calibration runs share one model: `available = free_vram − margin − overhead`, then
`candidate = available / bytes_per_token`. From the two rows of the final automatic matrix
(`kv-calibration-findings.md:73-78`, both `fit-target 643` MiB, batch/ubatch 64, q4_0 KV,
`parallel=1`):

- `...XXS.gguf` (no MTP head): `2499.6 − 643 − 144.4 = 1712.2` MiB / 97,280 tok
  → **18.0 KiB/token**
- `...-mtp.gguf` (MTP on): `2165.6 − 643 − 502.4 = 1020.2` MiB / 54,272 tok
  → **19.3 KiB/token**

Same architecture and quant, so the target-side KV cost is the same in both and the delta is
the nextn layer's own KV:

| Term | Value | Nature |
|---|---:|---|
| Target attention KV | 18.0 KiB/token | scales with `--ctx-size` |
| MTP nextn KV | **1.2 KiB/token** | scales with the draft budget |
| MTP context compute/workspace | **358 MiB** (502.4 − 144.4) | **fixed**, whenever the MTP context exists |
| Extra weights in the `-mtp` GGUF | **334 MiB** (2499.6 − 2165.6 free VRAM) | **fixed**, whenever that file is loaded |

The MTP draft KV is only **6.8%** of the per-token cost, so capping it buys almost nothing:

| `--mtp-ctx-size` | resulting `--ctx-size` ceiling | MTP covers |
|---:|---:|---|
| 54,264 (today) | 54,272 | everything |
| 32,768 | ~55,700 | prompts < 32k |
| 8,192 | ~57,400 | prompts < 8k |
| 0 (draft KV gone) | **~57,900** | nothing |

**~57,900 is the structural ceiling of the single-process approach** — below the clients'
80,000 — and it is only reached by shrinking the draft budget to zero, i.e. by disabling the
speedup the whole exercise exists to keep. The missing ~39,300 tokens are the two *fixed*
costs: 358 + 334 ≈ 692 MiB ≈ 39,300 tokens at 18.0 KiB/token. Cross-check:
`57,900 + 39,300 = 97,200` vs the measured 97,280 (the fit aligns to 256) — the model closes
to within 0.1%.

Those 692 MiB are recoverable only by not having the MTP head and its context in the process
at all. That is a process boundary. No per-request gating inside one process reaches it.

Confidence: two datapoints, one linear model, but it reproduces both measured candidates and
the implied 6.8% nextn share is sensible for one extra layer.

### 1.1 Measured: the `-mtp` GGUF reaches 60,928 with MTP on and 105,472 with MTP off

Measured on GOKAYA 2026-09-07 with `mtp_ctx_sweep3.py`, one GGUF
(`Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`), traditional KV `q4_0`, `--flash-attn on`,
`--fit off`, `--parallel 1`, the production power/clock flags
(`--gpu-power-prefill 200 --gpu-power-decode 165 --gpu-mem-clock-decode 11001`). Two ceilings per
config, because they are not the same number:

- **startup ceiling** — the largest `--ctx-size` that reaches `/health`.
- **usable ceiling** — the largest one whose *full-length* prefill also survives. This is the
  number that matters; `kv-calibration-findings.md:36-46` warned about it and it is real:
  `ggml_cuda_pool_vmm::alloc` grabs VRAM inside `launch_mul_mat_q` at prefill time, outside
  everything reserved at load.

| MTP | b/ub | startup ceiling | usable ceiling | gap |
|---|---:|---:|---:|---:|
| off | 64 | 105,728 | **105,472** | 256 |
| off | 512 | 98,560 | **98,048** | 512 |
| on | 64 | 61,440 | **60,928** | 512 |
| on | 512 | 52,736 | **51,712** | 1,024 |

All boundaries are sharp and land on multiples of 256 (the FA KV padding): 105,984 fails to load,
105,728 loads but OOMs on a 101,632-token prefill, 105,472 serves 101,376 + 4,096 cleanly.

Three findings that change the plan:

1. **The 77,200 in the previous version of this section was wrong by ~28k**, and wrong by method:
   it applied fit-*probe* accounting (which subtracts a 643 MiB `fit-params-target` margin plus
   probe overhead) to a `--fit off` configuration, which subtracts neither. The 96,000 row in
   `kv-calibration-findings.md:54-59` does belong to the non-MTP file — that part was right — but
   the correct comparison for prod is `--fit off` against `--fit off`, and there the `-mtp` file
   with MTP off gives **105,472**. Never carry a probe candidate across into a `--fit off` budget.
2. **MTP costs 44,544 tokens of context on this GPU** (105,472 → 60,928 at b/ub 64), i.e. the
   speedup is bought at 42% of the available context. That is the whole reason two children exist.
3. **The current production `--ctx-size 54272` for the MTP child leaves 6,656 tokens unused** —
   60,928 works at b/ub 64. The old 54,272 was a fit-probe candidate, not a measured limit.

**Batch size is a context/throughput knob, not a free choice.** b/ub 512 costs context and buys
prefill speed; b/ub 64 does the opposite:

| MTP | b/ub 64 usable | b/ub 512 usable | context cost of 512 | prefill tok/s (64 → 512) |
|---|---:|---:|---:|---|
| on | 60,928 | 51,712 | **−9,216** | 598.6 → 855.1 (+43%) |
| off | 105,472 | 98,048 | **−7,424** | 632.4 → 916.6 (+45%) |

(prefill measured on the same 32,768-token prompt at a common `ctx-size` of 40,960 — see 12.3;
the ceiling runs are not comparable to each other, their prompts differ by design)

For `qwen3-fast` that trade is worth taking in favour of context: b/ub 64 gives 9,216 more tokens
before a swap to the long child is needed, and a swap costs far more than the prefill difference.
Production runs b/ub 512 today, which is why its MTP child was stuck below 52k.

**Methodological trap, recorded so the next run does not lose two hours to it.** GOKAYA has
15 GiB of RAM, its swap is **zram** (compressed RAM, so there is no real swap headroom), other
services hold ~5 GiB, and the model is a 10.4 GiB mmap. A load burst on top of a warm page cache
plus the production `--cache-ram 3072` gets the child **SIGKILLed by the kernel OOM killer**
(`rc=-9`, `dmesg`: `Out of memory: Killed process llama-server ... file-rss:9.7GB`) *before* VRAM
is ever the constraint. An earlier run read those kills as VRAM ceilings and reported a false
usable ceiling of 75,520 for the MTP-off/b64 config — 30k too low. Any ceiling harness must:
run with `--cache-ram 0` (host-side prompt cache, irrelevant to a VRAM ceiling), `drop_caches`
and wait for `MemAvailable` between children, and treat `rc=-9` with no CUDA OOM in the log as
**retry**, never as a ceiling.

### 1.2 A zero-code win available today

Production currently serves the *non-MTP* `...XXS.gguf` at `--ctx-size 65536`, b/ub 512, no MTP.
The same box, same `--fit off`, running the `-mtp` file with MTP off at b/ub 64 is measured at
**105,472** — a 61% larger context, from a single unit-file edit and no router at all. The price is
prefill throughput (516 vs 776 tok/s) and giving up nothing else, since that child is not using MTP
either way. If the router work slips, this is the interim configuration to run.

## 2. Why the swap cost is acceptable

The requests that need the long tier are the ones where a model load is noise:

- 101,488-token prompt: 441.97 prompt tok/s → **230 s of prefill**, 465 s total (`:88-93`)
- 50,176-token prompt: 566.24 prompt tok/s → **89 s of prefill** (`:128-136`)
- so an 80,000-token prompt ≈ **180 s** before the first token

A swap is one model load per direction — sleeping is a full `destroy()`
(`server-context.cpp:964-1005`), so waking is a full load; there is no cheap warm swap. At the
measured 10 s (13.1) that is ~20 s against ~180 s, **≈ 11%**. And for these requests the honest
baseline is not "10 s slower" but **"completes" versus "CUDA OOM"**. Both children name the same
file, so every swap in either direction reads a warm page cache — that is what the 10 s figure
assumes; cold from disk it is worse.

**But the swap path has a host-RAM constraint, not just a VRAM one.** 15 GiB of RAM, zram swap
(no real headroom), ~5 GiB held by other services, a 10.4 GiB mmap — the calibration run had
children SIGKILLed by the OOM killer while cycling loads (1.1). The router will do exactly that
cycling. So: keep `cache-ram` modest (≤ 2048, not the 3072 prod uses today), and make sure the
evicted child is fully gone before the next one loads — `unload_lru()` already destroys before
loading, which is what keeps this safe; do not raise `--models-max` above 1 on this box.

## 3. What the router already does (no code)

- **`--models-max 1`** (`common/arg.cpp:3895`) caps residency; the second child evicts the
  first via `unload_lru()` (`server-models.cpp:938-960`).
- **A busy child is never evicted**: candidates are filtered on `req_count != 0`
  (`server-models.cpp:92`, `:203`) and a request with no free slot is **queued**
  (`:113`, `server_lru_sched`). A long request arriving mid-generation waits, then swaps.
- **`ensure_model_ready(name)`** loads on demand and blocks until the child is listening.
- **Per-child args** via `--models-preset file.ini` (`tools/server/README.md:1770-1800`),
  with a `[*]` section for shared defaults.
- **Preset-only keys** already exist as a pattern: `set_preset_only()` + a
  `COMMON_ARG_PRESET_*` env name, read back in `load_models()` with `preset.get_option()`
  (`common/arg.cpp:4985-4999`, consumed at `server-models.cpp:558-589`). The two new routing
  keys in section 4 follow it exactly — no new mechanism.
- **`hidden`** already exists on `server_model_meta` ("hidden from GET /models, but still
  accept if requested"), today only set by the cache-dedup logic.

What is missing is exactly one thing: **one public name resolving to several children by request
size.** Aliases cannot express it — `get_meta()` resolves an alias to a single canonical model
(`server-models.cpp:1811-1812`), so several children cannot share one alias.

## 4. Configuration

`my-models.ini`:

```ini
version = 1

[*]
model = /home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf   ; one file, both children
batch-size = 64                ; 64 not 512: buys 7,680 tokens of ctx on the MTP child (1.1)
ubatch-size = 64
cache-type-k = q4_0
cache-type-v = q4_0
parallel = 1
flash-attn = on
fit = off                      ; the measured ceilings below are --fit off numbers
cache-ram = 2048               ; not 3072: host RAM is the tight resource on this box (2)
ctx-checkpoints = 1
gpu-power-prefill = 200
gpu-power-decode = 165
gpu-power-device = 0
gpu-mem-clock-decode = 11001
sleep-idle-seconds = 300

[qwen3-fast]
ctx-size = 60928               ; measured usable ceiling, b/ub 64, MTP on (1.1)
spec-type = draft-mtp
spec-draft-n-max = 2
spec-draft-p-min = 0.80
spec-draft-type-k = q4_0
spec-draft-type-v = q4_0
route-group = qwen-3.8-27b     ; new preset-only key
route-max-tokens = 56000       ; new preset-only key: serve only requests up to this budget

[qwen3-long]
ctx-size = 105472              ; measured usable ceiling, b/ub 64, MTP off (1.1)
route-group = qwen-3.8-27b
; no route-max-tokens = the group's uncapped fallback; 105,472 covers the whole 84,096 contract
```

```sh
llama-server --models-preset ./my-models.ini --models-max 1
```

`route-max-tokens = 56000` against a 60,928 ceiling leaves 4,928 tokens of margin — the 4,096-token
output the clients ask for plus 832 of slack. That margin is not optional: the boundary is a cliff
(60,928 serves a full-length request, 61,184 loads and then dies on the prefill, 61,696 does not
load at all). Never set a cap to the ceiling itself.

`qwen3-long` needs no cap: 105,472 − 4,096 = 101,376 usable prompt, comfortably above the 80,000
the clients are configured for, and a 101,376 + 4,096 request is measured working.

## 5. Staging step (config only, no code)

Before the routing code exists, run the same INI **without** the `route-*` keys and have a
test client name `qwen3-fast` / `qwen3-long` directly. Nothing to code-review, and
it produces every number the rest of the plan assumes:

1. ~~The ceilings~~ — **done**, measured directly (1.1): 60,928 for the MTP child and 105,472 for
   the MTP-off child, both at b/ub 64, both confirmed with a full-length prompt + 4,096 output.
   Both configs also loaded and served **when spawned by the router** (13.1), so its arg rendering
   costs no extra VRAM. Still untested through the router: a *full-length* 101,376-token prompt on
   the long child — 13.1 only went to 70,019, and the cliff is 256 tokens wide.
2. ~~The real load time~~ — **done** (13.1): 7.5 s of child startup, ~10 s for a swap, page cache
   warm. A cold-file number is still missing.
3. **Eviction and VRAM reclaim** across the pair, via `nvidia-smi`. Also watch host RAM: the
   swap cycle is exactly the pattern that hit the OOM killer during calibration (1.1, 2).
4. **The b/ub choice**, if prefill latency turns out to matter more than context: b/ub 512 costs
   the MTP child 7,680 tokens of ceiling and buys +49% prefill throughput (1.1).

## 6. The routing code

### 6.1 Group resolution — deliberately outside `mapping`

Do **not** insert a synthetic group entry into `server_models::mapping`. Every existing
invariant (LRU counting, `load()`, `req_count`, status transitions) assumes a mapping entry is
a loadable child. Instead keep a separate, read-mostly map built in `load_models()`:

```
route_groups : public_name -> [ {member_name, max_tokens}, ... ]   // ordered, ascending cap
```

One helper resolves a public name to a concrete child, and every existing route stays
untouched:

```
std::string resolve_route_target(name, /* optional */ body)
```

- name is not a group → return it unchanged (all current behavior preserved)
- name is a group, `body` present → estimate the budget (6.2) and return the first member
  whose `max_tokens` covers it, else the uncapped member
- name is a group, no `body` (GET `/props`, `/tokenize`, …) → return the currently loaded
  member, else the group's **capped** member (`qwen3-fast`) — a metadata request should not get to
  pick which child boots

Call it in `proxy_post` right after `json::parse(req.body)`, before
`router_validate_model` (`server-models.cpp:1914-1920`). Everything downstream —
validation, `conv_models.remember`, `ensure_model_ready` (load/queue/evict) and
`proxy_request` — already works with whatever `name` holds, so the whole feature is that one
substitution. Also call it in `proxy_get` for the non-body routes.

### 6.2 The estimator

The router parent has **no tokenizer** — it never loads a model, it spawns children and
proxies HTTP. Given consequence 3 of the contract, undershoot is a client-visible failure, so:

- **Exact path (required):** `POST /tokenize` to whichever member is *already loaded*.
  Tokenization needs no KV cache and no context space, so the 54k child tokenizes an 80k
  prompt fine. One localhost round-trip, a few ms.
- **Cold start (no member loaded):** body bytes ÷ 3.5, and **bias up the ladder** on ties or
  doubt. Loading a model just to count tokens would cost more than routing one tier too high and
  being wrong.
- **Always add the output reserve:** the budget is `prompt + n_predict`. When the request omits
  `max_tokens`/`n_predict`, assume the router's `-n` default (4,096 here, matching the client
  config) rather than 0 — otherwise a 57k prompt with an implicit 4k output routes to the 60,928
  child and dies at the boundary.

### 6.3 No demotion within a conversation

`conv_model_tracker` (`server-models.h:145-197`) already maps `X-Conversation-Id` → serving
child, and `proxy_post` re-`remember()`s it on every POST *after* name resolution
(`server-models.cpp:1928`) — so a migrating conversation re-pins itself with no extra code.

Add one rule: **a conversation never moves down the ladder.** If it is pinned to `qwen3-long`, a
turn that estimates under 56k still goes to `qwen3-long`. Conversations grow monotonically, so this
makes each one migrate at most **once** instead of flapping around a threshold, and it costs a
single map lookup. Without it, a conversation hovering near 56k would swap models on nearly every
turn — and each swap also throws away the prompt cache (see 7).

### 6.4 `/v1/models` (same phase, cheap)

Mark all members `hidden` (extend the existing flag with a preset-only key, or derive it from
`route-group` membership) and append one synthetic entry for the group in the listing loop
(`server-models.cpp:1995-2035`). Advertise 80,000 (the contract) so a client that inspects the
list sees that rather than a 60,928 ceiling. This is cosmetic for the
current clients — they already know the name — but any OpenAI-compatible library that
validates the model list would otherwise break on a name that answers requests yet is absent
from `/v1/models`.

## 7. What clients WILL notice (transparency is not free)

Worth agreeing on these before implementing, because they are the price of the contract:

- **TTFT spikes.** A small request that follows a big one pays the swap: ~10 s of first-token
  latency, measured (13.1), on a request that would otherwise answer in ~2 s. Same model name, wildly
  variable TTFT. Unavoidable at `--models-max 1`; the mitigation is 6.3 plus, if measurement
  justifies it, a hysteresis timer keeping the long child resident for N seconds. Every swap here
  is the cheap kind — one file, warm page cache.
- **Prompt cache loss on migration.** A swap discards the child's cached prefix. A 60k
  conversation crossing the threshold re-prefills from scratch, ~110 s. 6.3 limits this to **once**
  per conversation; interleaved conversations of mixed sizes are the bad case. This is the hop that
  section 11 can optimise away, since both children share weights.
- **No style shift.** Both children load the same GGUF, so crossing 56k changes nothing about the
  weights — only whether the MTP head is used, and MTP is distribution-preserving anyway (every
  draft token is verified against the target). This was the main risk of the earlier two-GGUF
  design and the measurement in 1.1 removed it.
- **Throughput asymmetry stays visible**, and it is large: **42.9 decode tok/s** on the fast child
  against **19.15** on the long one, each at its own full length (both measured, section 12). A
  4,096-token answer takes 95 s on the fast tier and 214 s on the long one. Nothing to do about
  that — it is the point of the exercise.

## 8. Files touched

Staging step (section 5): none — one INI file outside the repo.

Routing:

| File | Change |
|---|---|
| `common/arg.cpp` | two `set_preset_only()` keys: `route-group`, `route-max-tokens` |
| `tools/server/server-models.h` | `route_groups` map + `resolve_route_target()` declaration |
| `tools/server/server-models.cpp` | build `route_groups` in `load_models()`; `resolve_route_target()` (estimator + no-demotion rule); call it in `proxy_post` (`:1917`) and `proxy_get`; hide members and emit the group entry in the `/v1/models` loop |
| `tools/server/README.md` | document the routing group, the estimator fallback and the TTFT/prompt-cache caveats |

`route_groups` holds an ordered list of members, so adding or removing a tier later is INI-only and
touches none of these rows.

Nothing in `common/speculative.cpp`, nothing in `server-context.cpp`, no change to
`can_speculate()`, no new inference-side flag.

## 9. Validation

Status after the 2026-09-07 session (13): **1, 2, 5, 7, 8 done** — including the largest request the
contract allows (80,036 tokens) through the router, on the long child, at 11,887 MiB of 12,282.
**3 partly**: the swap was timed (~10 s) but VRAM reclaim was not watched *across* the eviction.
**4, 6, 9 open** (13.3).

1. **Staging, both tiers named explicitly.** `qwen3-fast` with 56,832 prompt + 4,096 output →
   HTTP 200, ~44 decode tok/s. `qwen3-long` with 101,376 + 4,096 → HTTP 200. Both are already
   measured directly (1.1); this repeats them through the router's own child spawn.
2. **The ceilings survive the router.** The children must reach 60,928 and 105,472 when the router
   renders their args, not only when launched by hand. A few extra MiB of VRAM would push a
   full-length request over the cliff, and the cliff is 256 tokens wide.
3. **Swap cost and VRAM reclaim.** Time the second request's queue-to-first-token delta and
   watch `nvidia-smi` across the eviction: VRAM must return to the idle baseline between
   children. Also watch host `MemAvailable` — the swap cycle is the OOM-killer pattern from 1.1.
   The measured load time replaces the assumed 8 s everywhere in this doc.
4. **A busy child is never evicted.** Start a long generation on the fast tier, then fire a
   long-tier request: the first completes normally, the second queues and only then swaps. The
   whole design leans on this (`server-models.cpp:92`, `:203`, `:113`).
5. **Transparency, the actual acceptance test.** With only `qwen-3.8-27b` configured, send
   8k / 40k / 55k / 60k / 80k-token prompts each with `max_tokens=4096`. All must return HTTP
   200; log which child served each. The boundary to get right: 55k + 4k > the 56,000 cap, so that
   one must land on `qwen3-long`, while 40k + 4k stays on `qwen3-fast`.
6. **Estimator accuracy**, Portuguese and English, `/tokenize` truth vs bytes ÷ 3.5, at the same
   sizes. Record the worst-case undershoot; it must stay inside `qwen3-fast`'s margin
   (60,928 − 56,000 − 4,096 = 832 tokens). 832 is not much — this is why the exact `/tokenize` path
   is mandatory and the byte heuristic is a cold-start fallback only.
7. **Omitted `max_tokens`.** A 55k prompt with no `max_tokens` must still route to `qwen3-long`
   (implicit 4k reserve applied).
8. **No demotion.** A conversation growing 40k → 55k → 60k must migrate exactly once, to
   `qwen3-long`, and never come back; count the model loads (exactly one).
9. **Thrash cost.** Alternate small/large 5× and record total wall clock, to decide whether a
   hysteresis timer is worth building.

## 10. Sequencing

1. Section 5 staging config — today, no code. The ceilings are already measured (1.1); staging now
   only has to confirm them through the router and produce the real load times.
2. Sections 6.1–6.3 — the routing itself. This is the deliverable the clients need.
3. Section 6.4 — `/v1/models` cosmetics, same phase if cheap.
4. Hysteresis only if test 9 says so.
5. Section 11 KV state transfer between the two children. It is legal (same weights) and now
   implemented; promote it only with a new CUDA build and a controlled production restart.

## 11. Carrying the KV state across the swap

The question this answers: can the RAM cache make the context load faster when the router
switches tiers? Not the RAM cache — but a file-backed path already in the tree can, and now that
both children run the same GGUF it applies to the only hop there is.

### 11.1 `--cache-ram` cannot cross a swap

`server_prompt_cache` (`server-task.h:597-635`, implementation `server-task.cpp:1689-1793`) is a
`std::list` **in the child process's heap**, instantiated at `server-context.cpp:1419` and sized
by `-cram`/`--cache-ram` (`arg.cpp:1726`). Sleeping is a full `destroy()`
(`server-context.cpp:964-1005`) and eviction kills the subprocess, so the cache dies with the
child. It accelerates turns **within** a tier and does nothing for the crossing.

### 11.2 What can cross: slot state in a file

`--slot-save-path` (`arg.cpp:3853`) plus `SERVER_TASK_TYPE_SLOT_SAVE`/`_RESTORE`
(`server-context.cpp:2688`/`:2738`, HTTP `action=save|restore` at `:4974-4977`) over
`llama_state_seq_save_file`/`load_file` (`src/llama-context.cpp:4348`/`:4359`) moves a slot's KV
between processes through a file. Sizing at 18.0 KiB/token: ~1.05 GiB for a full 60,928-token fast
slot. **Not tmpfs on this box** — tmpfs is RAM and RAM is the scarce resource here (1.1, 2); write
it to disk, or cap it well below `MemAvailable`. Either way it is a few seconds of write plus read
against **~110 s of prefill** for a 60k prompt. That, not the model load, is where the latency is.

Flow: `save` on the outgoing child before eviction → swap → `restore` on the incoming child → the
next request finds its prefix in the slot and prefills only the delta.

What `save` captures, precisely (`server-context.cpp:2688-2737`): the **live KV of one slot**, via
`slot->prompt.tokens.serialize()` (`:2710`) plus
`llama_state_seq_save_file(ctx_tgt, …, slot->id, …)` (`:2717`). It does **not** dump
`server_prompt_cache`. So:

- **One slot per call.** At `parallel = 1` that is the single resident conversation. Prefixes
  parked in `-cram` (idle slots pushed there by `--cache-idle-slots`) are not captured — such an
  entry only becomes real again by passing through a slot.
- **Target only.** `ctx_tgt`; the `ctx_dft` (nextn) state never enters the file. Harmless: the
  receiving child runs with MTP off and has no `ctx_dft`.
- **Ordering is free.** A save on a processing slot is *deferred*, not failed (`:2696-2701`), so a
  "save before evict" hook waits for the generation to finish on its own, which matches the
  never-evict-a-busy-child rule.

### 11.3 Where it is legal, and where it is not

**Identical weights are mandatory.** Restore validation is purely structural — KV layer count
(`src/llama-kv-cache.cpp:2535`), KV type (`:2563`), row size (`:2572`) — with **no weight hash**.
A state file from a different GGUF either fails those checks or, worse, is accepted and serves KV
computed by other weights, silently. So:

- **`qwen3-fast` → `qwen3-long`: legal.** Same file, so same hparams and the same target KV
  geometry; MTP on/off only decides whether a separate `ctx_dft` exists, which is not in the file.
  Since the measurement collapsed the ladder onto one GGUF (1.1), this is the only hop and it is
  the legal one.
- **Any second GGUF: forbidden.** If a differently-quantized file is ever added as a third tier,
  no restore path may reach it — the checks would pass structurally and serve KV computed by other
  weights, silently.
- **Only upward.** 61k of cells fit in a 105k cache, never the reverse — and the no-demotion rule
  (6.3) already forbids the other direction.

### 11.4 Implementation status

The router now injects `--slots` and a private state directory into every group child. For an
upward migration carrying `X-Conversation-Id`, it saves slot 0 before `ensure_model_ready()`,
restores it after the target child is loaded, checks the saved/restored token count, and removes
the file. The transfer is best-effort: a failed action falls back to the normal full prefill.
The router compares model and mmproj identity before saving, so a different GGUF cannot receive
the state file. The directory is below the router's `--slot-save-path` parent when provided, or
the system temporary directory otherwise; a RAM filesystem such as `/dev/shm` is an explicit
operator choice and must fit the host's available memory.

The local smoke used the same GGUF in two children with `--models-max 1`: 20 tokens were saved and
restored, and the second request processed 289 of 309 prompt tokens (`309 - 20`). The production
deployment still needs a new CUDA build and a controlled restart before this path is live.

Keep `-cram` at 2048, not 3072: on this box the host RAM is what breaks first (1.1).

Unrelated config win, cheap and independent: **do not pass `--ctx-size 0` to the children in
production.** With 0 the fitting probe builds and frees full target + MTP contexts on every load,
inflating swap time. Use it once during staging to *find* the numbers, then pin them (section 4).

Keep `docs/mtp-adaptive-context-plan.md` for its code-path analysis — the always-on
`common_speculative_process()` shadow decode and the five static-capability uses of
`can_speculate()` are correct findings worth having on record — but do not implement it.

## 12. Test session: how these numbers were produced

Run on GOKAYA (`192.168.1.57`, RTX 4070 12 GiB, 15 GiB RAM) on 2026-09-07, with the production
`llama-server-root` unit stopped for the duration. Harness: `mtp_ctx_sweep3.py` at the repo root
(an earlier version produced the contaminated numbers described in 12.4 and is not kept).

### 12.1 What the harness does

Per config (MTP on/off × b/ub 64/512), against one GGUF and one set of flags where only those two
axes vary:

1. **Phase A — startup ceiling.** Binary search over multiples of 256 (the FA KV pad) for the
   largest `--ctx-size` that reaches `GET /health`.
2. **Phase B — usable ceiling.** From that ceiling downward, binary search for the largest
   `--ctx-size` whose **full-length prefill** (`ctx − 4096` tokens, `n_predict=1`) returns 200
   without the child dying. A pass/fail here is a real request, not an allocation guess.
3. **Confirm.** At the usable ceiling, one maximum-size request: `ctx − 4096` prompt +
   4,096 forced tokens (`ignore_eos`), recording wall time, prefill tok/s and decode tok/s.
4. **Comparison.** The same request (32,768 + 1,024) on every config at a common `ctx-size`
   of 40,960, which isolates the b/ub effect on throughput from the ceiling effect.

Prompts are built by tokenizing one phrase via `/tokenize` and repeating the ids to an exact
token count, so "50,176 tokens" means exactly that and not an estimate. The unit is stopped and
restarted by the harness itself in a `finally`, so an abort still returns the box to production.

### 12.2 Flags held constant

```
--model Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf --device CUDA0 --parallel 1
--flash-attn on --cache-type-k q4_0 --cache-type-v q4_0
--fit off --ctx-checkpoints 1 --cache-ram 0
--gpu-power-prefill 200 --gpu-power-decode 165 --gpu-power-device 0 --gpu-mem-clock-decode 11001
[MTP only] --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.80
           --spec-draft-type-k q4_0 --spec-draft-type-v q4_0
```

Same as the production unit except `--cache-ram` (0 instead of 3072, see 12.4) and the two axes
under test. `--fit off` matters: these are hard ceilings, not probe candidates.

### 12.3 Results

Ceilings — see the table in 1.1. Maximum-size requests at each usable ceiling:

| Config | ctx | prompt + output | wall | prefill tok/s | decode tok/s |
|---|---:|---|---:|---:|---:|
| MTP off, b/ub 64 | 105,472 | 101,376 + 4,096 | 449.9 s | 432.98 | **19.15** |
| MTP on, b/ub 64 | 60,928 | 56,832 + 4,096 | 205.6 s | 516.10 | **42.90** |
| MTP on, b/ub 512 | 51,712 | 47,616 + 4,096 | 152.2 s | 776.46 | 45.08 |
| MTP off, b/ub 512 | 98,048 | 93,952 + 4,096 | 345.8 s | 674.32 | 19.84 |

Each row runs at its own ceiling, so the prompt lengths differ and the throughputs are **not**
comparable across rows. That is what the last phase is for — the same 32,768 + 1,024 request on all
four configs at a common `--ctx-size 40960`:

| Config | prefill tok/s | decode tok/s | wall | VRAM |
|---|---:|---:|---:|---:|
| MTP off, b/ub 64 | 632.41 | 28.46 | 87.8 s | 10,519 MiB |
| MTP off, b/ub 512 | **916.61** | 28.88 | 71.2 s | 10,633 MiB |
| MTP on, b/ub 64 | 598.63 | 51.09 | 74.8 s | 11,385 MiB |
| MTP on, b/ub 512 | 855.09 | **51.82** | 58.1 s | 11,613 MiB |

Reading it, with everything but one axis held fixed:

- **MTP buys 1.80× on decode** (51.09 vs 28.46 at b/ub 64) and costs 44,544 tokens of ceiling.
- **b/ub 512 buys +45% on prefill** (916.61 vs 632.41) and costs 7,424–9,216 tokens of ceiling,
  while leaving decode alone (28.46 → 28.88). It is a pure prefill/context trade.
- MTP also costs ~5% of prefill (632.41 → 598.63): the draft head is evaluated during prefill too.

Caveat on the decode figures: the prompt is a repeated phrase, so the continuation is unusually
predictable and MTP's acceptance rate is flattered. The *relative* comparison across configs holds
(same prompt everywhere), the absolute 1.80× is an upper bound; real conversational text will be
lower.

### 12.4 What went wrong twice, and what to do about it

- **Missing `LD_LIBRARY_PATH`** (the production unit sets it) — every child exited `rc=127`.
- **Host OOM masquerading as a VRAM ceiling.** Detailed in 1.1. Symptom: `rc=-9` with no CUDA
  message in the log; the same config that had loaded at 79,360 minutes earlier "failed" at
  78,848. It produced a usable ceiling of 75,520 that was 30k too low, and it is the reason
  `--cache-ram 0` plus `drop_caches` plus retry-on-`rc=-9` are in the harness now. **A ceiling
  measurement on this box is only trustworthy if the failure at the boundary is a CUDA OOM in the
  log.**

### 12.5 Reproducing

```sh
scp mtp_ctx_sweep3.py 192.168.1.57:/home/hjotha/
ssh 192.168.1.57 'cd /home/hjotha && nohup python3 -u mtp_ctx_sweep3.py > sweep3.out 2>&1 &'
```

Results land in `/home/hjotha/mtp-ctx-sweep3-results.json`; progress in `sweep3.out`. Budget
~2 h of production downtime for the full four-config matrix — each usable-ceiling probe is a real
100k-token prefill.

## 13. Test session: the routing code itself

Harness: `route_group_smoke.py` at the repo root. It writes a `--models-preset` INI, starts the
router with `--models-max 1`, fires real requests at the **public name only**, and asserts against
the router's own `route group '...': N prompt + M output = ... -> child` log lines plus the response
bodies. Two profiles:

- `tiny` — a 4B GGUF, `tiny-fast` (ctx 2,048, cap 300) + `tiny-long` (ctx 8,192). Same decisions,
  seconds per request instead of minutes. Use this one while changing the code.
- `real` — the deployment: one `-mtp` GGUF, `qwen3-fast` (60,928, MTP) + `qwen3-long` (105,472),
  public name `qwen-3.8-27b`, cap 56,000. This is the acceptance test.

```sh
scp route_group_smoke.py 192.168.1.57:/home/hjotha/
ssh 192.168.1.57 'echo $PASS | sudo -S systemctl stop llama-server-root'
ssh 192.168.1.57 'cd /home/hjotha && python3 -u route_group_smoke.py tiny'   # ~15 s
ssh 192.168.1.57 'cd /home/hjotha && python3 -u route_group_smoke.py real'   # ~10 min
ssh 192.168.1.57 'cd /home/hjotha && python3 -u route_group_smoke.py real worst'   # ~6 min
ssh 192.168.1.57 'echo $PASS | sudo -S systemctl start llama-server-root'
```

### 13.1 Results — 2026-09-07, both profiles 15/15

The `real` run, one line per assertion (prompt/output are the router's counts, not the aim):

| Check | Result |
|---|---|
| `/v1/models` lists `qwen-3.8-27b` and nothing else | PASS |
| group `meta.n_ctx` = widest member's ctx-size | PASS 105,472 |
| `meta.route_members` ordered, capped tier first | PASS `[(qwen3-fast, 56000), (qwen3-long, -1)]` |
| 512-token ask → fast tier | PASS (660 + 32) |
| response `model` field | PASS `qwen-3.8-27b` |
| exact `/tokenize` count, not bytes ÷ 3.5 | PASS aimed 50,000, counted 50,024 |
| 50,024 + 32 (under the cap) stays on fast | PASS |
| 57,526 + 32 (over the cap) → long | PASS, swapped mid-request |
| 70,019-token ask → long | PASS, 502.2 prefill / 22.6 decode tok/s |
| omitted `max_tokens` reserves 4,096 | PASS |
| conversation pinned wide is never demoted | PASS long → long |
| a *different* conversation still routes by size | PASS → fast |
| 93,001-token `/tokenize` answered | PASS |
| …and logged no routing decision | PASS |

Plus `route_group_smoke.py real worst`, run separately because it costs ~6 minutes: **the largest
request the advertised contract allows** — ctx 80,000 minus the 4,096 output the client also asks
for, decoded to the last token with `ignore_eos`.

| | |
|---|---|
| counted | 75,940 prompt + 4,096 output = 80,036 → `qwen3-long` |
| result | HTTP 200 in 354 s, prefill 489.5 tok/s, decode 21.5 tok/s |
| output | exactly 4,096 completion tokens, no truncation |
| model field | `qwen-3.8-27b` |
| VRAM | 11,887 MiB with the long child resident (of 12,282 available) |

Two numbers worth keeping:

- **Swap cost is ~10 s, not the 8 s assumed throughout this doc.** Child startup is 7.5 s
  (`load_model` → `model loaded`) with the GGUF warm in page cache, and a 512-token request that
  forced a tier swap took 10 s end to end. Cold from disk it will be worse — the 10.4 GiB file is
  read twice per alternation.
- **Transparency is real, verified two ways.** The child's rendered args carry
  `--alias qwen-3.8-27b` (not `qwen3-fast`), and every response body echoed `qwen-3.8-27b`. Without
  the forced alias a client that replays `response["model"]` would address a tier directly and
  bypass routing entirely.

### 13.2 What the harness got wrong, and the fix

Both failures were in the test, not the router — worth recording because both look like routing
bugs at first glance:

- **`%` binds tighter than `+`.** `COMMON_INI + """...""" % dict(...)` substituted only the second
  string, so the children were launched with a literal `%(model)s` path and died with
  `failed to open GGUF file '%(model)s'`. Surfaced as HTTP 500 `failed to load`, which reads exactly
  like a VRAM problem.
- **BPE merges across a repetition boundary.** Calibrating on one copy of the filler phrase (32
  tokens) overestimated by ~3%: a prompt aimed at 56,200 tokens arrived as 54,488, landed *under*
  the 56,000 cap, and the correct routing decision was scored as a failure. Calibrating on 100
  repetitions (31.01 tokens each) brought the error to 0.05%. The lesson transfers to the router's
  own estimator: any tokens-per-unit-of-text constant measured on a short sample is optimistic.

### 13.3 Not covered by this harness

- **Test 4 of section 9** (a busy child is never evicted) — needs two concurrent clients; the
  harness is sequential.
- **Test 9** (thrash cost over an alternating series) — the swap cost above is a single measurement,
  enough to defer hysteresis, not enough to price it.
- **A full-length request through the router.** `worst` covers what clients can actually send
  (80,036 tokens); it does not reach the long child's own ceiling of 101,376 + 4,096. The cliff is
  256 tokens wide, so if the advertised context is ever raised above 80,000, re-measure — do not
  assume the headroom is there.
- **Estimator accuracy on natural text.** Every prompt here is one repeated phrase, so the
  cold-start bytes ÷ 3.5 fallback was only exercised incidentally (it read 660 tokens for a
  579-token prompt: a 14% *over*estimate, which is the safe direction).

## 14. Production deployment (GOKAYA, promoted 2026-09-07)

The `llama-server-root` unit no longer runs a single model. It runs the router:

```
ExecStart=/home/hjotha/src/llama-mtp-ctx/build/bin/llama-server \
  --models-preset /home/hjotha/prod-two-tier.ini --models-max 1 \
  --host 0.0.0.0 --port 8090 --metrics \
  --log-file /home/hjotha/router-two-tier-prod-20260907.log
WorkingDirectory=/home/hjotha/src/llama-mtp-ctx
Environment=LD_LIBRARY_PATH=/home/hjotha/src/llama-mtp-ctx/build/bin
```

`/home/hjotha/prod-two-tier.ini` is section 4's INI with two deliberate differences:

- **The tiers carry the names the old unit advertised as aliases** — `qwen-3.8-27b-ista-mtp` (fast)
  and `qwen-3.8-27b-ista-nomtp` (long) — so a client that used to name one of them directly still
  reaches a working model instead of a 404. `qwen-3.8-27b` is the group.
- **No `sleep-idle-seconds`.** The old unit never slept; adding it would make every request after
  an idle gap pay a reload. Add it later if the GPU is wanted for something else.

What changed for clients, all of it measured:

| | before | after |
|---|---|---|
| GGUF | `...IQ3_XXS.gguf`, no MTP | `...IQ3_XXS-mtp.gguf`, both tiers |
| context | 65,536 | 60,928 (≤ 56,000 budget) / 105,472 |
| advertised `context_window` | 65,536 | 105,472 |
| batch / ubatch | 512 | 64 |
| decode | ~28 tok/s | ~51 tok/s on the fast tier, ~20 on the long one |
| prefill | ~917 tok/s | ~600 tok/s fast, ~500 long |
| `/metrics`, `/props` | no parameter | need `?model=<name>`, the group name works |
| web UI at `/` | served | still served |

The trade is deliberate: b/ub 64 costs ~35% of prefill throughput and buys 9,216 tokens of MTP
ceiling; MTP doubles decode. For interactive use decode dominates, and the long tier is what makes
the 80,000-token contract possible at all.

Verified on the live unit after the switch: a 264-token request routed to the MTP tier, a
60,053-token one to the long tier (`60037 prompt + 16 output`), both answering as `qwen-3.8-27b`;
the mem clock locked at 11,001 MHz and the decode power cap at 165 W, as before; real client traffic
(`8836 prompt + 4096 output`) routed to the fast tier on its own. Routing decisions are logged by
the router, so `journalctl -u llama-server-root | grep "route group"` is the way to see which tier
served what.

**Rollback** — the old unit is at `/home/hjotha/llama-server-root.service.bak-20260907` and the old
build tree `/home/hjotha/src/llama.cpp` is untouched:

```sh
sudo cp /home/hjotha/llama-server-root.service.bak-20260907 \
        /etc/systemd/system/llama-server-root.service
sudo systemctl daemon-reload && sudo systemctl restart llama-server-root
```

One thing to keep an eye on: the advertised `context_window` is now the long tier's full 105,472,
so a client that fills it will send a prefill larger than anything tested through the router (13.3).
If that becomes a real pattern rather than a theoretical one, either measure 101,376 through the
router or advertise less.
