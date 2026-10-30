# RKVA HELLO startup telemetry

The HELLO response's existing `xbox_prepare_us` field now reports the elapsed
time in its handler. The Xbox event log also records `phase=hello_init`,
`cycle`, `ok`, and the following CPU wall durations in microseconds:

- `evict_us`: lock and eviction of any local Xbox session.
- `engine_us`: whole `RemoteAttnEngine::initialize` call.
- `device_us`: device factory/adapter selection and device creation.
- `queue_us`: command queue, allocator, list, fence and event creation.
- `roots_us`: mandatory root signature serialization and creation.
- `required_psos_us`: mandatory pipeline creation, including the optional
  serial-prefix reference PSO currently built in the same group.
- `q4_psos_us` and `format_psos_us`: optional groups, including their roots.
- `options_us`: post-initialization option setters, including the Q8 paired
  reducer PSO when requested.
- `hello_before_log_us`: full handler elapsed before emitting the event.
- `format_init_us` and `format_init_attempts`: individual format PSO calls
  made inside `format_psos_us`. An attempt count of zero distinguishes a
  deferred or unavailable PSO from a measured sub-microsecond call.
- `format_late_us` and `format_late_attempts`: PSO calls made after initialization
  by control-path setters. When made during HELLO they are inside `options_us`,
  not `format_psos_us`; these nested spans must not be added to their parents.

`engine_us` contains the device/queue/root/PSO spans and must not be added to
them. `hello_before_log_us` contains eviction, engine and options. The wire
field is set after the log write and includes its cost. A failed initialization
can leave unfinished subspan fields at zero; `ok=false` distinguishes that
case from a completed zero-microsecond span. The HELLO response is still sent
only after initialization. No shader source compilation is inferred: PSO
creation consumes the DXIL embedded in the executable.

These are Xbox `steady_clock` durations; do not subtract absolute Xbox and host
timestamps. Compare the host's HELLO `rpc_total` and `rpc_header_wait` with
the remote handler duration as nested measurements, and measure wall time again
with diagnostic capture disabled before making a performance claim.

## Optional lazy INT8 initialization

With `rkva-lazy-int8-init.flag` in Xbox LocalState, the KVarN4 MQ2 INT8 and
Q8 MQ2 INT8 PSOs are created only when their respective INT8 option is enabled.
Without that flag, both retain the original eager initialization. INT8 itself
remains controlled by its existing flags, is approximate and is not made
qualified by this change. An enabled option with missing DXIL, unsupported
wave/root or failed creation keeps the existing fallback. A failed factory is
not retried by dispatch or later toggles in the same engine. Disabling an option
does not release its PSO; resources remain owned until engine retirement.

The flag affects only the current engine and is read before initialization.
Changing or removing it takes effect on the next HELLO. It does not cache
objects across connections or change other PSOs, kernels or split policies.

## Proposed opt-in cache: not implemented

Reusing device/root signatures/PSOs across connections would need an immutable,
strongly owned bundle separate from the engine and its per-connection queue,
allocator, command list, fence, sessions, descriptors, buffers and counters.
The proposed process-local key must cover the actual adapter/device identity
and feature capabilities, root-signature bytes, all included DXIL bytes and
pipeline descriptors, geometry-dependent eligibility and relevant options.
Only a complete, successful bundle could be published under a mutex. A disabled
flag, changed/incomplete key, removed device, creation error or uncertain GPU
retirement must fail closed or invalidate the bundle before reuse. New engines
would still own fresh mutable execution state and preserve reset/owner epochs.

The current source creates several geometry- and wave-dependent optional PSOs,
creates the Q8 paired reducer after initialization, and lazily creates bulk
PSOs later. It also preserves an engine on unconfirmed GPU retirement. A
complete invalidation contract spanning those paths and D3D12 object lifetimes
has not been proven by the CPU-only checks. Therefore this change deliberately
does not implement `rkva-pipeline-cache.flag` or retain D3D12 resources across
connections. Measure the breakdown first, then validate any cache candidate on
Windows/Xbox with device-removal, reconnect, reset, failure and flag-change
tests before considering it for use.
