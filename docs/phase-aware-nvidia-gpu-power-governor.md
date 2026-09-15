# Phase-aware NVIDIA GPU power governor

## Revised implementation plan

This document records the implementation contract for the first version of the
phase-aware NVIDIA GPU power governor. It was revised against fork `master`
`9b6ab8994` on 2026-09-04.

## Goal and scope

When explicitly enabled, `llama-server` will select one global GPU power
profile from the real server slot state:

- `PREFILL`: configured high-performance power limit.
- `DECODE`: configured efficient power limit.
- `IDLE`: no power-limit write; the NVIDIA driver controls idle power.

The feature is opt-in. With no GPU-power arguments, the server must not load
NVML, write a power limit, or change the upstream behavior.

The first version does not include auto-calibration, PID control, clock or
voltage control, VRAM-clock control, model-specific profiles, a daemon, a
proxy, metrics polling, utilization heuristics, or `nvidia-smi` subprocesses.

## Actual server integration points

The current slot state enum is in `tools/server/server-context.cpp` near line
100. The relevant transitions are:

- `launch_slot_with_task()` sets `SLOT_STATE_STARTED` near line 1935.
- `update_slots()` runs the global scheduling pass near line 2899.
- `pre_decode()` changes `STARTED` to `PROCESSING_PROMPT` near line 3251 and
  fills text and multimodal prompt batches in the same pass.
- Prompt completion changes the slot to `DONE_PROMPT` near line 3710.
- `post_decode()` changes `DONE_PROMPT` to `GENERATING` near line 3961, or
  releases embedding/reranking slots after their prompt is evaluated.
- `server_slot::release()` changes a finished slot to `IDLE` near line 559.
- Sleep callbacks are registered in `init()` near line 1447 and model unload
  happens in `handle_sleeping_state()` near line 963.

The governor phase is updated at the beginning of `update_slots()`, after new
tasks have been assigned and before `pre_decode()` can submit heavy work. This
means `STARTED` already selects `PREFILL` before the first prompt batch is
executed.

## Global phase arbitration

The arbitration is global because the NVIDIA power limit is global to the
selected GPU:

| Slot state | Governor contribution |
| --- | --- |
| `IDLE` | none |
| `WAIT_OTHER` | none |
| `STARTED` | `PREFILL` |
| `PROCESSING_PROMPT` | `PREFILL` |
| `DONE_PROMPT` | `PREFILL` to avoid a transient drop |
| `GENERATING` | `DECODE` |

The priority is `PREFILL > DECODE > IDLE`. A generating slot plus a slot in
prompt processing therefore selects `PREFILL`. `WAIT_OTHER` by itself does
not select a GPU profile.

The phase arbitrator is independent of NVML and is covered by a fake-backend
unit test. The production server observes all slots once per scheduling
iteration. It does not maintain independent per-slot power state.

This first version assumes one governor owner per physical GPU. Router mode or
other independently running server processes must not enable governors that
target the same GPU, because NVML power limits are global to that device.

## Governor state and lifecycle

`tools/server/server-gpu-power.{h,cpp}` will contain the isolated governor,
phase arbitrator, backend interface, and runtime NVML backend.

The governor stores:

- the original device power limit;
- the validated minimum and maximum limits;
- the configured prefill and decode limits;
- the last applied limit;
- the last global phase;
- whether the original limit was changed and the number of phase transitions.

All governor calls are made by the `server_context` inference thread:

1. Initialize after the model/context are loaded and before the first loop;
   reinitialize and revalidate the device after a sleep resume.
2. Update once at the beginning of each `update_slots()` call.
3. Restore the original limit on entry to server sleep, without forcing an
   idle limit. During ordinary idle, the last active limit remains in place;
   no NVML write is made and the driver still controls instantaneous idle
   consumption.
4. Restore and unload NVML after the inference loop terminates, before the
   surrounding server cleanup frees the llama backend.
5. Keep cleanup idempotent so failed startup and destructor paths are safe.

The update path only compares phase and target limit. It does not lock, poll,
query telemetry, or call NVML unless the configured target limit actually
changes. A repeated `PREFILL` phase produces no additional NVML write.

If a runtime power-limit write fails, the governor logs the error once and
disables further automatic writes so inference can continue. Startup errors
(missing NVML after explicit opt-in, invalid device, missing paired profile,
or a value outside the device range) fail clearly before serving requests.

## CLI and environment interface

Server-only arguments:

```text
--gpu-power-prefill W
--gpu-power-decode W
--gpu-power-device N
```

Environment equivalents:

```text
LLAMA_ARG_GPU_POWER_PREFILL
LLAMA_ARG_GPU_POWER_DECODE
LLAMA_ARG_GPU_POWER_DEVICE
```

The two profile values are required together when the feature is enabled.
`--gpu-power-device` alone does not enable the governor. Values are accepted
in watts at the CLI and converted to milliwatts only at the NVML boundary.
The backend validates both values against the device's reported constraints;
it never clamps an out-of-range value silently.

## NVML integration

The backend uses runtime loading of `libnvidia-ml.so.1`/`libnvidia-ml.so` on
POSIX systems and `nvml.dll` on Windows. It resolves only the required NVML
functions for initialization, device lookup, name/constraint/current-limit
queries, power-limit writes, error strings, and shutdown.

No NVML header or link-time NVIDIA library is required. The small dynamic
loader is isolated from `server-context.cpp`; only the platform dynamic-loader
library is linked where the platform requires it. CPU-only, AMD, Vulkan-only,
macOS, and other non-NVIDIA builds therefore remain buildable. An explicitly
requested governor on a machine without a usable NVML device returns a clear
startup error.

Startup logs include the device name, NVML index, original limit, allowed
range, and configured profiles. Runtime logs are limited to phase transitions
and errors, for example:

```text
GPU power: idle -> prefill, limit 200 W
GPU power: prefill -> decode, limit 165 W
GPU power: decode -> idle
```

## Coverage of server workloads

The integration is attached to the common slot state and batch scheduler, not
to an HTTP route. It therefore covers text completions, OpenAI-compatible chat
completions, prompt-cache reuse followed by prompt evaluation, embeddings,
reranking, multimodal prompt chunks, and speculative/MTP execution.

Speculative/MTP work performed while a slot is `GENERATING` uses `DECODE`.
If any other slot is in `STARTED`, `PROCESSING_PROMPT`, or `DONE_PROMPT`, the
global priority selects `PREFILL`.

## Tests

The new unit test uses a fake backend and no NVIDIA device. It covers:

- all idle -> `IDLE`;
- one generating -> `DECODE`;
- one prompt-processing slot -> `PREFILL`;
- generating plus prompt processing -> `PREFILL`;
- prompt completion while another slot generates -> `DECODE`;
- all requests complete -> `IDLE`;
- repeated same-phase updates -> no repeated NVML writes;
- paired profile validation and out-of-range rejection;
- opt-in disabled -> no backend initialization.

The existing argument parser test will also verify the three server arguments
and their `LLAMA_ARG_*` names.

## Documentation and validation deliverables

The generated server help in `tools/server/README.md` was refreshed from the
argument definitions with `llama-gen-docs`. The server README includes
the usage example:

```bash
llama-server \
  -m model.gguf \
  --gpu-power-prefill 200 \
  --gpu-power-decode 165 \
  --gpu-power-device 0
```

Validation must include the focused governor test, argument parser test,
`llama-server` build, relevant server unit tests, no-flag help/build checks,
and a final review of hot-path overhead, multi-slot arbitration, speculative
decoding, sleep/shutdown restoration, optional NVML loading, and diff scope.

## Implementation status

The implementation is on branch
`feat/phase-aware-nvidia-power-governor-20260904`, based on fork `master`
`9b6ab8994`. The changed source files are:

- `tools/server/server-gpu-power.{h,cpp}`;
- `tools/server/server-context.cpp`;
- `common/common.h` and `common/arg.cpp`;
- `tools/server/CMakeLists.txt`;
- `tests/test-server-gpu-power.cpp` and the existing argument parser test;
- generated and explanatory server documentation.

The CUDA/Vulkan build and a CPU-only build both compiled `llama-server` and
the focused tests. The focused CTest cases passed in both builds, and a real
CPU-only server smoke test loaded NVML, reported the RTX 4070 constraints,
processed a request, and restored cleanly on shutdown. A second request reached
the real power-limit write but the host returned `NVML code 4: Insufficient
Permissions`; the governor logged the error and disabled further writes, as
designed. The fake backend test covers successful writes and restoration.

The pytest server suite was not run because `pytest` is not installed in this
environment. After explicit deployment, the system unit
`/etc/systemd/system/llama-server-root.service` runs the committed
`build/bin/llama-server` as root and enables the `200/165/0` profiles. The
previous user unit remains at
`~/.config/systemd/user/llama-server.service`, disabled for rollback. Real
production requests returned successfully and the service stayed healthy. The
root process produced the transitions `idle -> prefill -> decode -> idle`, and
`nvidia-smi` confirmed the active decode limit at 165 W. A controlled stop
restored the original 200 W limit before the service was started again. Router
deployments with multiple independent processes targeting the same physical GPU
remain unsupported because the power limit is device-global.

During the first production observation, the `54272` context configuration
showed repeated CUDA OOM/ABRT crashes when OpenCode sessions approached the
available VRAM. The stack was in `MUL_MAT` and was independent of NVML. The
context was temporarily reduced to `40960`, but a deterministic sequence later
reproduced the same failure in the intact upstream `llama2` binary with both
`--cache-ram 0` and `4096`. Disabling CUDA graphs eliminated it.

Instrumentation showed that short-lived prefill shapes were repeatedly becoming
eligible for CUDA graph capture while only a few MiB of VRAM remained. The CUDA
backend now waits for four structurally stable calls before capture, avoids a
redundant instantiate/update cycle, and synchronizes before incompatible graph
replacement. The full failing sequence passed with CUDA graphs re-enabled,
`--ctx-size 54272`, `--cache-ram 4096`, and no server restart. See
[OOM reproduction trace](oom-reproduction-trace.md) for the exact requests and
A/B evidence.
