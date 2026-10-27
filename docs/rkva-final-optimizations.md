# RKVA optimization policy — 2026-10-26

This consolidates the user's hardware verdicts for integration base `8ac3e9d10`.
These measurements were supplied by the operator; this consolidation reruns
CPU/Python checks and DXC only. It does not independently reproduce the hardware
campaign. The exact model filenames, prompt, sampling settings, commands and
run commit IDs were not supplied with the verdict table; the percentages below
are acceptance evidence for this integration, not a reproducible benchmark recipe.

| Change | Supplied hardware result | Shipping policy |
| --- | --- | --- |
| CUDA VRAM guard cache | Decode +2.2%; prefill neutral; identical tokens | Retained unchanged, including 64 MiB slack and `GGML_CUDA_GRAPH_GUARD_SAMPLE_MS` |
| Rowlane decode | Kernel time −35%; decode throughput +6.2%; oracle error about 1e-5; 254/256 matching tokens | Enabled by default for qualified calls |
| INT8 prefix prefill | Pass1 time −17.1%; prefill throughput +7.96%; 256/256 matching tokens; Q8 error about 1% | Opt-in only |
| Console fence spin | +0.29%, within noise | Removed; blocking event wait retained |
| Rescale-skip | Slower | Remains removed |
| GPU ceiling probe | Console accepts shader models through SM 6.4 | Retained, including SM 6.5/6.6 capability probes |

Flags are files under the app's **LocalState**, sampled at each RKVA HELLO:

- `rkva-decode-rowlane-off.flag`: presence disables rowlane. Absent means enabled.
- `rkva-mq-i8.flag`: presence enables INT8 prefill. Absent means disabled.
- `rkva-mq-gqa2.flag`: presence arms the unqualified GQA2 two-local-head
  prefill candidate when its PSO exists. Absent means disabled; a simultaneous
  `rkva-mq-i8.flag` keeps GQA2 off (fail-closed). This candidate is not part of
  the accepted optimization set and has no hardware result yet.

Rowlane still requires native KVarN4 descriptor revision 3, nq=1, the optimized
kernel selection, fixed wave64, the qualified dense 256-dimensional shape and
an available rowlane PSO. Calls that cannot select it retain the existing kernel
fallback. The opt-out restores the previous decode kernel without recompilation.
INT8 still requires the existing MQ2 shape/tail/wave/PSO gates. It approximates
record-row QK; it is not bit-exact. The sole INT8 reduction uses `WavePrefixSum`
and constant lane endpoints 15/31/47/63. Unsupported calls retain the F32 path.
Matching generated token counts do not establish general model-quality parity.

Profiles preserve rowlane and INT8 kernel identities for both ordinary and bulk
submissions, including counters and `BulkSegmentProfile` fields. The ceiling
probe remains a tool; successful DXC compilation of SM 6.5/6.6 does not imply
console PSO support.

See [INT8 arithmetic](rkva-prefill-int8-dot4.md) and the
[consolidation report](agent-reports/astra-final-consolidation.md) for payload
hashes, validation commands and the pending Windows/console handoff.
