> Historical QSA Q8 implementation report. The F16 alloc-size mapping described here was corrected after the review; see docs/PLAN-strata-parity.md.
>
> Historical agent report (2026-11-21). Current status and numbers: docs/PLAN-strata-parity.md.

# Strata parity — Fase 1: QSA sparse attention with a Q8_0 KV cache

Checkout: `/tmp/opencode/impl-qsa` (isolated copy of `beellama.cpp` at
`49a560ca3`). No commit, no push, no GPU execution. The main checkout was only
read.

## Goal

Make the Qwen3.8-Flash-Next QSA selection (`n_kv_max` ≈ 2048) drive the sparse
FlashAttention MMA path when the local KV cache is **Q8_0/Q8_0**, instead of
only when it is F16/F16. The selected rows must be dequantized on the fly, not
by converting the whole cache to F16, so that the 12 QSA layers do not scan the
full KV at long context.

## What changed and why

### 1. `src/llama-graph.cpp` (lines 5117–5126)

`build_attn()` previously propagated `sparse_kv_bound` only for physical
`F16/F16` K/V. It now also accepts `Q8_0/Q8_0`:

```cpp
const bool sparse_kv_type =
        (k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16) ||
        (k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q8_0);
const int64_t sparse_kv_bound = cparams.flash_attn && !use_kvarn &&
        tail_route == LLAMA_KV_TAIL_ROUTE_NONE && !k_tail && !v_tail &&
        sparse_kv_type ? n_kv_max : 0;
```

All existing conditions are preserved: no KVarN, no tail, local body, Flash
Attention enabled. `n_kv_max` is still the bound produced by `build_qsa_sel()`
(512 pools × 4 cells + 3-cell tail = 2051 for this model). With `n_kv_max == 0`
nothing changes; mixed K/V types and every other shape keep the previous dense
path.

### 2. `ggml/src/ggml-cuda/fattn-mma-f16.cuh`

- `flash_attn_ext_f16_load_tile` (line 392) gained a compile-time
  `ggml_type type_KV` and an `elem_offset` argument. The original F16 path is
  byte-for-byte the same addressing, just expressed as
  `base + elem_offset/2 + k*h2_per_chunk`.
- For `GGML_TYPE_Q8_0`, the loader gathers the selected row and calls the
  existing `dequantize_V_q8_0<half, 8>()` from `fattn-common.cuh` for each
  8-value chunk (16 bytes of shared `half2` tile). Every 8-value chunk lies
  inside one 32-value Q8_0 block, so the existing block/scale decoder is used
  unchanged. `stride_KV` is in bytes for Q8_0 and in `half2` for F16.
- Q8_0 loads are synchronous (no `cp_async`) because they dequantize instead of
  copying; the call sites now select `use_cp_async` only for F16
  (`nstages == 1 && type_K == GGML_TYPE_F16` / `type_V == ...`).
- The kernel (line 2168) sets `stride_K`/`stride_V` to the raw byte row strides
  when `type_K/type_V == Q8_0`, and asserts that the direct Q8_0 loader is only
  used with Q8_0/Q8_0 and without K/V aliasing.
- `ggml_cuda_flash_attn_ext_mma_f16_case` (line 2319) has a constexpr
  `can_sparse_q8` for `D=256, ncols2=8, ncols1=1|8`. When sparse is selected
  and `K/V` are Q8_0, it instantiates
  `flash_attn_ext_f16<..., GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true>` and calls
  `launch_fattn(..., need_f16_K=false, need_f16_V=false, ...)`, so the packed
  cache is consumed directly.

No new template-instance file was necessary. The q8 kernel is instantiated
inside the already-generated cases `256/256/1/8` and `256/256/8/8`:

- `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_8.cu`
  (`DECL_FATTN_MMA_F16_CASE(256, 256, 1, 8)`)
- `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_8.cu`
  (`DECL_FATTN_MMA_F16_CASE(256, 256, 8, 8)`)

Proof from the compiled sm_80 objects (`cuobjdump -symbols`):

```
_Z18flash_attn_ext_f16ILi256ELi256ELi1ELi8ELb0ELb0EL9ggml_type8ELS0_8ELb1EE...
_Z18flash_attn_ext_f16ILi256ELi256ELi8ELi8ELb0ELb0EL9ggml_type8ELS0_8ELb1EE...
```

`ggml_type8` is `GGML_TYPE_Q8_0` and the trailing `Lb1E` is `use_sparse=true`.

### 3. `ggml/src/ggml-cuda/fattn.cu`

- `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse` (lines 143–189) now
  recognises a direct Q8_0 case. It is restricted to NVIDIA Ampere+,
  `D=256/D=256`, `ncols2=8` and an instantiated `ncols1`. F16 keeps its original
  `2 * n_gather` threshold; Q8_0 uses `K->ne[1] > n_gather` because each
  gathered row is 34 bytes per 32 values instead of 64. Both keep the existing
  `K->ne[1] >= 4096` floor and mask/max_bias/logit_softcap checks. Q8_0
  geometries outside the direct kernel keep the historical
  F16-conversion sparse path.
- `ggml_cuda_flash_attn_ext_uses_sparse_q8` (line 198) mirrors
  `switch_ncols1` for allocation/selection: GQA-optimised path
  (`mask && max_bias == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0`), ratio > 4,
  `ncols1=1` for batches ≤ 4 and only the wide `ncols1=8` candidate for larger
  batches (matching the exact `has_wide_sparse` skip in `switch_ncols1`).
- `ggml_cuda_get_best_fattn_kernel` (lines 1145–1158) no longer routes an
  applicable sparse Q8_0 decode to the vector kernel; it falls through to
  `BEST_FATTN_KERNEL_MMA_F16`. The batch-invariant override still wins.
- `ggml_cuda_flash_attn_ext_get_alloc_size` (lines 1289–1298) omits the dense
  F16 conversion buffers exactly when the Q8_0 sparse kernel will run. This is
  essential: otherwise the allocator would reserve a full F16 copy of K/V and
  `launch_fattn` would convert the cache even though the kernel does not need
  it.

### 4. `tests/test-backend-ops.cpp` (lines 13760–13786)

Added 24 Q8_0 QSA `FLASH_ATTN_EXT` cases:

- full matrix `n_kv = 4096, 16384, 32768` × `nq = 1, 5, 512` ×
  `n_kv_max = 0, 2048` (18 cases);
- no mask (`mask=false`) with `n_kv_max=2048` for `nq=1,5,512` (3 cases);
- KV tail `n_tail = 1,2,3` with `n_kv_max=2048` (3 cases; tails are F16 because
  `ggml_kv_tail_attention_merge` accepts F16/BF16/F32 tail bodies only).

All use `hsk=hsv=256`, `nh=2`, `nr23={12,1}` (GQA 12) and `kv_view=true`, which
is the QSA geometry.

## Validation performed here (CPU only)

### Build

```bash
cmake -B build-cpu -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-cpu --target test-backend-ops llama-cli -j4
```

Result: configure succeeded; both targets built (`100% Built target llama-cli`).
`llama-cli --version`:

```
version: 0.5.0 (build 13854, commit 49a560ca3-dirty)
built with GNU 16.2.1 for Linux x86_64
ggml op count: 113
```

### New Q8_0 QSA test matrix

```bash
./build-cpu/bin/test-backend-ops -b CPU -j 4 -o FLASH_ATTN_EXT \
  -p 'hsk=256,hsv=256,nh=2,nr23=\[12,1\].*type_K=q8_0'
```

Output:

```
  24/24 tests passed
selected_cases=24
  Backend CPU: OK
1/1 backends passed
OK
```

The largest case (`kv=32768, nq=512, n_kv_max=2048`) is included and passed.
The CPU backend ignores the `n_kv_max` hint (as before), so this validates test
construction, mask/tail composition and dense numerical agreement on CPU, not
the CUDA sparse kernel itself.

### Regression checks for the pre-existing paths

```bash
./build-cpu/bin/test-backend-ops -b CPU -j 4 -o FLASH_ATTN_EXT \
  -p 'hsk=256,hsv=256,nh=2,nr23=\[12,1\].*type_K=f16'
# 4/4 tests passed

./build-cpu/bin/test-backend-ops -b CPU -j 4 -o FLASH_ATTN_EXT -p 'n_kv_max=257'
# 13/13 tests passed

./build-cpu/bin/test-backend-ops -b CPU -j 2 -o FLASH_ATTN_EXT \
  -p 'hsk=256,hsv=256,nh=2,nr23=\[12,1\],kv=32768,nb=1,.*type_K=q4_0'
# 1/1 tests passed
```

### CUDA compilation without a GPU run

`nvcc` 13.4.92 was used to compile the CUDA translation units for `sm_80`
(A100) with the same `-DGGML_CUDA_FA_*`, `-DGGML_CUDA_KVARN`,
`-DGGML_USE_EXL3` and include flags as the repository's existing CUDA
`compile_commands.json`, changing only the source path and
`--generate-code=arch=compute_80,code=[compute_80,sm_80]`:

- `ggml/src/ggml-cuda/fattn.cu` → rc 0, 0 errors;
- `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_8.cu`
  → rc 0, 0 errors;
- `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_8.cu`
  → rc 0, 0 errors.

The presence of both Q8_0 sparse kernels in the objects was verified with
`cuobjdump -symbols` (symbols above). No CUDA kernel was executed: this machine
has a GPU, but the task explicitly forbids GPU execution and requires CPU-only
validation.

## Not validated here

- **Numerical/correctness execution of the CUDA Q8_0 sparse kernel**: no GPU
  run. The CPU test above cannot exercise it because the CPU backend ignores
  `n_kv_max`.
- **A100 throughput/memory measurements**: not run.
- **End-to-end `llama-cli`/server with Qwen3.8-Flash-Next and `-ctk q8_0
  -ctv q8_0`**: not run (requires GPU and the model).
- **Other CUDA architectures**: only `sm_80` was compiled. The direct Q8_0
  path is explicitly disabled below Ampere and on non-NVIDIA GPUs.
- **Other backends**: the graph now also exposes the Q8_0 hint to Metal,
  Vulkan, etc. Metal already implements Q8_0 sparse (`ggml-metal-ops.cpp`
  accepts Q8_0 and `n_kv_max <= 4096`); Vulkan only enables sparse for
  quantized KV above a larger ratio. Neither backend was built or run here.

## Risks

1. **Heuristic risk**: Q8_0 uses `K->ne[1] > n_gather` (1×) instead of the F16
   `2×` bound. At 28k with `n_kv_max=2051`, `n_gather` for the wide kernel is
   `8*2051 = 16408`, so sparse is selected. At 16k it is not selected
   (`16384 < 16409`), and at 4k wide prompt is dense. If the union of the
   selected rows is close to the full KV, dequantization overhead can make the
   gather slower than dense; the A/B benchmark must confirm. A conservative
   fallback is to raise the Q8_0 bound to `2*n_gather` or require a larger
   `n_kv_max`.
2. **Dequant granularity**: the kernel dequantizes 8 values per 16-byte shared
   chunk. This reuses the existing block decoder and keeps every call inside
   one 32-value Q8_0 block, but it is more instruction-heavy than a bulk F16
   conversion. It is only profitable because the gather reads ~2k rows instead
   of the full cache.
3. **Allocation/dispatch coupling**: `get_alloc_size` and kernel selection share
   `ggml_cuda_flash_attn_ext_uses_sparse_q8`; any future change to
   `switch_ncols1` must update that helper in lockstep, or the F16 conversion
   buffer size and the launched kernel will disagree.
4. **Architecture coverage**: Q8_0 direct sparse is limited to NVIDIA
   Ampere+ (sm80+), D=256, ncols2=8. Turing sm75 and other head sizes fall back
   to the previous dense/vector path, so the feature is not universal.
5. **Graph-level effect on non-CUDA backends**: Q8_0 QSA now passes `n_kv_max`
   everywhere. Backends that implement sparse Q8_0 (Metal) may change path;
   backends that ignore the hint (CPU) are unchanged. No non-CUDA backend was
   tested.

## How to validate on the A100 (sm_80)

Build the patched tree and a baseline worktree, both with the same flags:

```bash
cmake -B build-a100 -S . -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-a100 --target test-backend-ops llama-bench llama-cli -j8
```

### 1. Correctness (must pass)

```bash
./build-a100/bin/test-backend-ops -b CUDA0 -j 4 -o FLASH_ATTN_EXT \
  -p 'hsk=256,hsv=256,nh=2,nr23=\[12,1\].*type_K=q8_0'
```

Criterion: **24/24 OK**, each case below the test's `max_nmse_err = 5e-4`.
This covers `nq=1/5/512`, `n_kv=4096/16384/32768`, `n_kv_max=0/2048`, no-mask
and tail 1–3. `n_kv_max=0` and the no-mask/tail cases prove the dense fallback
is unchanged.

### 2. Confirm the Q8_0 sparse kernel is launched

```bash
ncu --target-processes all --kernel-name-base demangled \
  --kernel-name 'regex:flash_attn_ext_f16.*Q8_0' --launch-count 12 \
  ./build-a100/bin/llama-cli -m <model.gguf> -ngl 99 -fa \
  -ctk q8_0 -ctv q8_0 -c 32768 -n 1 -p "<short prompt>"
```

Criterion: the Q8_0 `flash_attn_ext_f16` kernel appears for the 12 QSA layers
at long context. `cuobjdump -symbols build-a100/bin/libggml-cuda.so | grep
'ggml_type8'` must also show the two symbols above.

### 3. Prefill A/B

Run the same command on baseline `49a560ca3` and on this branch:

```bash
./build-a100/bin/llama-bench -m <model.gguf> -ngl 99 -fa 1 -c 32768 \
  -ctk q8_0 -ctv q8_0 -p 4096  -n 0 -r 5
./build-a100/bin/llama-bench -m <model.gguf> -ngl 99 -fa 1 -c 32768 \
  -ctk q8_0 -ctv q8_0 -p 28672 -n 0 -r 5
```

Baseline from the planning run: **1,456 t/s @4k** and **1,175 t/s @~28.6k**.
Acceptance:

- `p=28672`: **≥ 1,469 t/s** (+25% over 1,175);
- `p=4096`: **≥ 1,413 t/s** (no more than 3% below 1,456);
- if the 4k number moves more than 3%, keep the Q8_0 bound at `2*n_gather`
  (or require `K->ne[1] >= 2*n_gather`) and re-measure.

### 4. Decode after ~28k

Use a real ~28k-token prompt and measure generation after the prefill:

```bash
./build-a100/bin/llama-cli -m <model.gguf> -ngl 99 -fa \
  -ctk q8_0 -ctv q8_0 -c 32768 -n 64 -f prompt_28k.txt --no-warmup \
  2>&1 | grep -E 'eval time|tokens per second'
```

Criterion: **decode ≥ 55 t/s** after the 28k prefill. Also repeat a short
prompt and require no more than 3% regression versus the baseline build.

If correctness passes but `p=28672` is below the target, the first knob is the
Q8_0 `min_kv` ratio in `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse`;
the second is the 8-value dequant chunk (widening it to 16 values would halve
the per-chunk instruction count at the cost of a larger shared-memory write).

## Notes on the isolated checkout

The initial `/tmp/opencode/impl-qsa` copy was missing 478 tracked files
(including `src/models/`). They were restored from
`/home/hjotha/beellama.cpp` at `HEAD 49a560ca3`; `cmp` over every tracked file
that was already present showed **0 differences**, so the baseline is intact.
The `.git` metadata was copied read-only from the same checkout to make
`git status` meaningful; no commit or push was made. `IMPL_REPORT.md` is
excluded through `.git/info/exclude` so that:

```bash
git -C /tmp/opencode/impl-qsa status --short
```

lists only the four scope files:

```
 M ggml/src/ggml-cuda/fattn-mma-f16.cuh
 M ggml/src/ggml-cuda/fattn.cu
 M src/llama-graph.cpp
 M tests/test-backend-ops.cpp
```
