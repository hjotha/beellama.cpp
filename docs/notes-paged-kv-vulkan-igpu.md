# Paged KV on Vulkan / iGPU: crash and slowdown notes

Status: documented for a future fix. Not yet optimized.

## Hardware context

- Host: GOKAYA (192.168.1.57), APU AMD Ryzen Z1 Extreme (Phoenix), iGPU Radeon 780M
  (gfx1103, RADV PHOENIX, ~8 GiB shared GTT budget).
- MiniStral-3-3B-Instruct-2512 Q4_K_M served on the iGPU via Vulkan0.
- Qwen3.8-27B served on the RTX 4070 via CUDA0 with paged KV (fast path).

## Symptom 1: ABRT at startup with `--kv-paged-prealloc-max` on Vulkan

```
common_fit_paged_kv_blocks: free_vram=17592186044416.0 MiB ... n_gpu_blocks=807599201
radv/amdgpu: The CS has been cancelled because the context is lost.
ggml_vulkan: device lost on Vulkan0
terminate called after throwing an instance of 'vk::DeviceLostError'
```

Root cause chain:

1. `ggml_backend_vk_get_device_memory` (ggml/src/ggml-vulkan/ggml-vulkan.cpp)
   sums ALL memory heaps for integrated GPUs (`is_integrated_gpu || DEVICE_LOCAL`).
2. RADV reports an unbounded system heap with `heap.size == SIZE_MAX` (2^64).
   The sum overflows to 2^44 MiB.
3. `common_fit_paged_kv_blocks` (common/common.cpp) then computes ~807M blocks
   and `--kv-paged-prealloc-max` tries to preallocate them. The RADV driver
   cancels the command stream (device lost) and the uncaught
   `vk::DeviceLostError` aborts the process.

Fix (commit 907c7594c):

- Vulkan: skip heaps with `size == SIZE_MAX || size > 1<<60` in the memory sum.
- common: reject budgets above a 1 PiB sanity bound; if the selected device
  reports 0/absurd, walk the other selected devices and fall back to the
  reported total before giving up.

With the fix the calibration reports the real iGPU budget
(free_vram=5638.0 MiB, blocks=8128, ctx=130048).

## Symptom 2: paged KV on Vulkan is ~4-5x slower than traditional KV

Measured on MiniStral-3B Q4_K_M, Vulkan0, batch 1024 / ubatch 256,
2 concurrent requests (300 tokens each), same master build:

| Config | ctx/slot | t/s per slot (2 concurrent) |
|---|---|---|
| Traditional KV (no paged) | 71168 | 19.9 / 20.4 |
| `--kv-paged-prealloc-max` | 130048 | 4.4 - 5.3 |

Single request without paged: ~28 t/s (71168 ctx).

Conclusion: the Vulkan backend has no optimized paged-attention path; the
generic path used for paged KV decodes is several times slower than the
traditional KV cache. CUDA keeps its fast paged path, so production uses:

- Qwen (CUDA0): `--kv-paged-prealloc-max` (fast, working).
- MiniStral (Vulkan0): traditional KV, `--ctx-size 142336 --parallel 2`
  (2 slots x 71168, sized to hold the Qwen context 66560 + 4096 output).

## ROCm/HIP on the 780M (gfx1103): not usable

- `GGML_BACKEND_DL=ON` is required for the HIP backend to be loadable
  (`libggml-hip.so` needs the `ggml_backend_init` symbol; with
  `GGML_BACKEND_DL=OFF` it is a no-op and ROCm0 never appears).
- With DL=ON the backend enumerates ROCm0, but the first decode ABRTs inside
  libhsa-runtime/libamdhip64 (gfx1103). Known upstream issue:
  ggml-org/llama.cpp#20839 - rocBLAS TensileLibrary has no gfx1103 kernels;
  `HSA_OVERRIDE_GFX_VERSION=11.0.0` (spoof gfx1100) fails later with
  `Assertion 'err == hipSuccess' failed` in hip_code_object.cpp, and the
  amdgpu MES bug on kernels <= 6.13 can hang the machine.
- Decision: keep Vulkan (RADV) for the 780M.

## Future work

- Optimize paged attention for Vulkan (or add a hybrid: paged pool for the
  compactor slot only when it is strictly needed).
- Revisit ROCm when gfx1103 is officially supported by TensileLibrary and the
  MES bug is fixed in the shipped kernel.