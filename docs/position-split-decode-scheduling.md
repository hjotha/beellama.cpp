# Position-split decode scheduling (2026-10-16)

The decode change groups overflow stores, gathers/packed-view copies, mask
packing, FA2/LSE2 and merge intermediates into one Vulkan cluster. It does
**not** achieve one Vulkan cluster for the entire rotated-query layer: a
separate Vulkan Q WHT must precede local CUDA attention to preserve its bits.

`build_attn_mha()` calls `ggml_build_forward_expand()` itself. Tensor creation
order alone therefore does not control scheduler order. Previously the graph
expanded overflow stores, local FA1, overflow FA2, local LSE1, then overflow
LSE2/merge. Expanding LSE1 immediately after FA1 removes the CUDA interruption
between FA2 and LSE2. Deferring overflow stores and diagnostic probe expansion
until after local FA/LSE keeps the overflow operations together, before their
consumers. The stores write the disjoint Q4 range, not the local KVarN history.

The remaining dependency chain is:

```
Vulkan Q WHT -> CUDA local FA/LSE -> Vulkan overflow FA/LSE + merge -> CUDA outputs
```

The CUDA/KVarN + Vulkan regression measures **7 -> 5 total scheduler splits**
and **3 -> 2 Vulkan clusters** in its isolated layer graph. This is not a
production token-rate measurement or a measurement of the full model's split
count. No server was started.

## Numerical constraints

- An attempted CUDA relocation of Q's F32 WHT failed bitwise Q/O/LSE equality
  at D=256 on RTX 4070 / RADV PHOENIX. That relocation was removed. Matching
  tensor dtypes does not make backend WHT implementations bit-identical.
- Merge intermediates already followed LSE2 onto Vulkan under scheduler
  inference. Their placement is now explicit; EXP/LOG/DIV keep the same
  backend implementations and dtypes. CUDA `expf`/`logf` and Vulkan `exp`/`log`
  are not interchangeable under a bitwise contract.
- The final merged-LSE ADD and owning output CONT stay CUDA-pinned. No final
  copy or mask materialization was removed, and the formula is unchanged.
- Reordering is restricted to single-query decode with no local tail or with
  `LLAMA_OVF_TAIL_LOCAL_CUDA=1`. Prefill and unanchored F16-tail placement retain
  their legacy order. Unsupported Vulkan Q-WHT widths retain backend fallback.
- Host index/mask pins and all packed-view, raw-Q4, gather-type, CONT and F5
  diagnostic gates remain in place. Diagnostic roots are deferred with the
  overflow work, not removed.

## Validation

`test-position-split-merge-graph` compares old/new expansion order at D=128 and
D=256, with raw Q4, contiguous K only, contiguous K/V, F16 gathers and F32
then F16 gathers. It includes a capacity-sized host mask cropped on Vulkan,
and two executions of each allocated graph with different Q. All comparisons
require bit-identical Q/O/LSE, with no tolerance relaxation. Fixture leaves
are retained across executions to model persistent KV and input buffers.

Required commands:

```
cmake --build build-optimized --target test-position-split-lse-sched test-position-split-merge-graph test-position-split-lse-cu llama-server -j4
ctest --test-dir build-optimized -R "position-split" --output-on-failure
```

Both commands passed on the working tree based on `a2be53dae`, branch
`feat/position-split-prod`: **10/10 CTest tests passed**, 67.83 seconds total.
The merge-graph test reported bitwise Q/O/LSE equality in all ten scheduling
variants, on both evaluations. Logs are in
`build-optimized/position-split-build.log`,
`build-optimized/position-split-ctest.log`, and
`build-optimized/Testing/Temporary/LastTest.log`.

# Phase-3 findings: the decode floor is the dGPU, not the cross-device fence

Measured 2026-10-17 on the 27B (kvarn4 P=96256 + Q4 overflow, 120k ctx, test-server 8095):

- `LLAMA_OVF_FUSED_SPAN=1` (phase 2, synchronous executor): decode 4.658 t/s, spans 18->2 on a
  4B smoke, identical greedy output. Hmm: structurally clean, no speedup.
- `LLAMA_OVF_FUSED_SPAN=1` (phase 3, stage-node + pinned async transfers + single host fence per
  layer): decode 4.233-4.456 vs OFF 4.044 same-session (profiles: ON 2 spans/token total; OFF 30
  spans/token: 15 CUDA + 14 Vulkan + 1 CPU), i.e. +5-10%, well short of the 8 t/s target.
- SPLITSPAN profiling explains why: in the LEGACY path the Vulkan (iGPU) queue is never fenced
  per layer -- the 14 layers' overflow work pipelines continuously and only end-of-graph events
  synchronize. The fused executor submits + reads back O2/LSE2 per layer, replacing that
  pipeline with 14 host round-trips; the fa1||fa2 overlap gain is cancelled.
- The decode ms/token is dominated by the dGPU backlog itself: ~15 CUDA spans x 8-12.9 ms ~
  150-190 ms of 4070 work per token (the MoE graph, 3688 nodes per split). That is the floor:
  ~4-6 t/s at 120k with MTP disabled.
- Consequence: further decode TPS must come from amortizing the per-step dGPU cost, i.e. the
  MTP/speculative path over the split (piece d), not from cross-device scheduling tricks.
- The phase-3 executor code was left uncommitted and reverted; phase 2 (`cd91cea7f`) stays as an
  env-gated, default-off scaffold with its reviewed correctness gates.
