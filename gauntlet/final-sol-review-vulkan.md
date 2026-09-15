# Final Gauntlet review — CUDA Graph and Vulkan runtime disable

- Reviewer: Codex Sol persistente (`gpt-5.6-sol`, high), sessão reutilizada.
- Decision: `ACCEPT_WITH_NOTES`.
- Diff: `/tmp/final-sol-review-vulkan-diff.patch` (SHA-256 `1c7245ca6b0dadcd7d897efb3c2ea39f54b5f6a0686dce34daeee3cd9c8b041d`).

No concrete blocker remains for the promoted CUDA profile. The Vulkan registration guard runs before `ggml_vk_instance_init()`, preventing `VkInstance` creation and device enumeration when `GGML_DISABLE_VULKAN` is present. The device-count guard covers the defensive window after registration. CUDA graph cleanup remains fail-closed and preserves handles on destroy failure.

Evidence reviewed: Release CUDA+Vulkan `-j8` build exit 0, SHA-256 `980f6b62e808a4518a89a6908fb6a4000de492b1327c47f6a7ebc1ed39790db3`; parser with env absent found two Vulkan devices and parser with env present found no Vulkan devices; A/B 19446 was 5/5 HTTP 200 with tokens `1024/1024/1024/1024/1`, graph capture/instantiate, cleanup `1 executable + 1 capture`, headroom `17.56 < 18 MiB`, and no OOM/Xid/abort; Vulkan0 smoke 19447 returned health/completion 200; production 8090 is active with PID `1497943`, `NRestarts=0`, health/props/completion 200 and no Vulkan/OOM/CUDA/Xid/abort markers.

Notes: `GGML_BACKEND_DL=ON` was not tested specifically; in that mode dlopen may still occur but backend init returns null before Vulkan initialization. `GGML_DISABLE_VULKAN` uses presence semantics (including value `0`) and must be fixed before registry construction. The validated profile is serial `parallel=1`, headroom 18 MiB; broad concurrency, low-threshold overrides, and long soak remain outside this promotion gate.
