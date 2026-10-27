# RKVA KVarN4 MQ two-local-head compile experiment

Status: **prepared and CPU-tested offline; no DXC, Windows, GPU or device result
exists yet.** This report does not claim a speedup, a TPS delta or bit-exact
output. It scopes a candidate PSO for the root to compile and inspect later.

## Scope and ownership

- Clone: `/home/hjotha/beellama-agents/latency-audit` (all existing branches
  preserved; no worktree).
- Branch: `perf/rkva-mq2-gqa2-compile-20261027`.
- Base: `8bad8e84e67e9fcd43d12f637962b87685e0fecc` (`main` of
  `/home/hjotha/beellama.cpp`, fetched read-only into `FETCH_HEAD`; the root
  checkout was not modified).
- Changed files:
  - `tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq.hlsl`
  - `tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq_gqa2.hlsl` (new)
  - `tools/xbox/tests/test_rkva_mq_gqa2_layout.py` (new)
  - `docs/agent-reports/rkva-mq-gqa2-compile.md` (this file)
- Not touched: engine, headers, generated DXIL, build scripts, activation,
  models, config, artifacts. The wrapper is not wired into
  `compile-rkva-range-shaders.sh`, the engine, or the activation flag.

## Design

The shipped tile keeps a group on six local Q heads (24 query heads over 4 KV
heads, `RKVA_GQA == 6`). The candidate keeps the **global** ratio at 6 and adds
`RKVA_MQ_HEADS_PER_GROUP` (default 6, accepts 2 or 6). At 2, the local arrays,
loops, WHT and accumulators use two heads while every query/KV/global-head
stride still uses `n_head == 24` and `RKVA_GQA == 6`.

Proposed dispatch (host change is out of scope):

```
DispatchX = 3 * n_head_kv
kv_head   = group.x / 3
head_base = 2 * (group.x % 3)
global    = kv_head * 6 + head_base + g   (g = 0, 1)
          == 2 * group.x + g
```

`group.y` (query tile) and `group.z` (split) stay unchanged, as do the reducer,
`[numthreads(128,1,1)]`, resources, cbuffer, entry point and the
`[split][query * n_head + head][258]` partial layout. Each global head is
written once. The candidate reloads and dequantizes each K/V row for three
groups instead of one (and triples the group count); whether that trade is
profitable is **UNKNOWN**.

Preprocessor contract:

- `RKVA_MQ_HEADS_PER_GROUP` defaults to 6; values other than 2 or 6 `#error`.
- `RKVA_MQ_HEADS_PER_GROUP == 2` requires `RKVA_MQ_QT == 2` and
  `RKVA_MQ_I8 == 0`; both violations `#error` before HLSL parsing.
- Constants are branch-local literals so the shipped defaults keep their old
  tokens: `RKVA_ELEMS` 12 or 4, `RKVA_WHT_HALF` 1536 or 512.
- `RKVA_LH` is `RKVA_GQA` at 6 (expands to 6) and `2` at 2. The global head
  offset uses `RKVA_MQ_HEAD_OFF`/`RKVA_MQ_HEAD0_OFFSET`, which are empty/identity
  at 6.
- The wrapper defines `RKVA_MQ_HEADS_PER_GROUP 2`, `RKVA_MQ_QT 2`,
  `RKVA_MQ_I8 0` and includes the original kernel.

The per-head F32 expression order, `precise` affine decode, wave reductions,
online softmax/PV, WHT butterflies, causal record/stage/tail classification and
NaN/inactive-query/partial-write paths are unchanged. Only local indices and
loop bounds changed. Bit-exactness is therefore **possible but not claimed**
until a real DXC build and replay.

## CPU validation (this host)

Command:

```bash
python3 tools/xbox/tests/test_rkva_mq_gqa2_layout.py
```

Result: `Ran 6 tests ... OK`. The same suite also passes with `CPP=clang` and
`CPP=gcc`.

What the tests prove, using `cpp`/`clang` preprocessing (not D3D12 codegen):

1. **Default token parity.** For QT2 F32, QT4 F32 and QT2 INT8, the modified
   default source preprocesses to the exact same normalized token stream as
   `git show <base>:tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq.hlsl`:
   - QT2 F32: 5,609 tokens, equal;
   - QT4 F32: 5,910 tokens, equal;
   - QT2 INT8: 6,924 tokens, equal.
2. **Entry contract.** The wrapper keeps the base cbuffer/resource declaration
   block, `CSRangeAttnKvarn4Mq` signature and `[numthreads(128,1,1)]`.
3. **Real groupshared sizes.** The declarations in the preprocessed source
   evaluate to `gs_max`/`gs_den` 48 B + `gs_vec` 12,288 B = **12,384 B** at six
   local heads, and 16 B + 4,096 B = **4,128 B** at two.
4. **Dispatch/partial oracle.** For KV4 (24 global heads), the two-head mapping
   covers heads 0..23 exactly once for `nq` in {2,3,5,8,9} and `split` in
   {1,2,17,64}; every per-(query, head) partial offset equals the shipped
   six-head offset, and the closed form `global == 2 * group.x + g` holds.
5. **Local bounds.** `RKVA_ELEMS == 2 * LH`, the WHT half is `LH * 256`, the
   ping-pong destination stays below `2 * RKVA_WHT_HALF`, and `q_lo`,
   `tail_pair`, `body`, `merged_body` and the wave slots stay in range at both
   two and six heads.
6. **Bad configurations.** `HEADS == 1`, `HEADS == 2 && QT == 4` and
   `HEADS == 2 && I8 == 1` fail preprocessing with the expected `#error`.

The base shader SHA256 is
`e52f25fb5fd50e1d35f05a4ca5923b7011f16ffa7aa77bc15c87af6e074540f2`, matching
the hash recorded by the existing generated `attn_kvarn4_mq2_i8` header. The
modified file necessarily has a new file hash; the shipped defaults still
preprocess identically, but existing generated headers keep recording the old
hash until the root regenerates. Regeneration is expected to change only the
recorded hash if DXC output is deterministic, which the root must verify rather
than assume. No generated header was touched in this phase.

## Pending (root / Windows)

1. Compile the shipped `attn_kvarn4_mq2` (baseline) and the new wrapper with the
   real DXC from Windows SDK 26100, same target profile (`cs_6_0`, entry
   `CSRangeAttnKvarn4Mq`). Keep the wrapper out of the generated headers and the
   activation path until reviewed.
2. Inspect DXIL IR/liveness for the candidate: local head state, LDS traffic,
   K/V reload sites, barriers and register pressure. Do **not** treat declared
   or physical VGPR counts as a performance result.
3. Astra decides whether the compile evidence justifies a hardware C++ variant
   and a replay. No TPS estimate or device/Windows result is implied here.

## Non-claims

- No DXC/D3D12 compilation, no Windows/MSVC/UWP build, no Xbox, GPU, LAN, SSH,
  `systemctl`, CUDA or model run was performed.
- No merge, push or activation change. The branch is committed for review only.
