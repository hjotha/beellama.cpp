# Sol review — Vulkan runtime disable

- Reviewer: Codex Sol persistente (`gpt-5.6-sol`, high), sessão reutilizada.
- Decision: `ACCEPT_WITH_NOTES`.
- Scope: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` guard in `ggml_backend_vk_reg()` and `ggml_backend_vk_reg_get_device_count()`; production CUDA unit environment; parser and GOKAYA A/B evidence.

## Decision

No blocker for commit or promotion of the CUDA profile when `GGML_DISABLE_VULKAN` is fixed at process startup. The guard executes before `ggml_vk_instance_init()`, so it prevents `VkInstance` creation and device enumeration. The device-count guard covers the defensive window after a registration handle is obtained.

The A/B on port 19446 proves 12 MiB recovered at baseline (50 MiB free with the variable versus 39 MiB with Vulkan enabled), real CUDA graph capture/instantiate followed by the 17.56 MiB < 18 MiB preflight and release of 1 executable plus 1 capture, 5/5 HTTP 200, and `server_rc=0`. The parser without the variable still found two Vulkan devices.

## Notes

- `GGML_BACKEND_DL=ON` was not tested specifically. In that mode `dlopen` may still occur, but the backend init returns `nullptr` before Vulkan initialization and the loader discards the handle.
- The variable has presence semantics and must be fixed at startup. Calling `setenv` or `unsetenv` after registry construction can leave the registry/API inconsistent and does not release an instance already created. This does not affect the validated systemd process.
- HIP/ROCm is outside this CUDA+Vulkan promotion by explicit user decision.
