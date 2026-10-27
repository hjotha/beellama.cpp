# CUDA graph scratch lifetime with the legacy pool

CUDA graphs capture temporary pool addresses as kernel arguments. Those addresses
are not represented by the graph tensor-property cache. With
`GGML_CUDA_NO_VMM=ON`, an allocation-OOM retry clears cached legacy-pool buffers;
a full 256-entry cache also physically frees returned buffers. Replaying a graph
that captured one of those buffers can access freed device memory.

The production S-tier reproduction used Qwen3.8 27B, KVarN4, MTP6, context24576,
batch/ubatch256, prompt22528 and the production graph headroom18MiB. Capture and
first launch succeeded. A subsequent allocation-OOM retry freed242899968 bytes
of scratch immediately before an MTP graph replay failed. Disabling graphs
completed the control; disabling the newer Q2 verification routing did not fix it.
A diagnostic that bypassed the same-UID tensor-property shortcut also failed.

Each context now tracks a scratch-allocation epoch. Physical legacy-pool releases
advance it. Before replay, an executable captured under an older epoch is retired
and normal warmup/capture resumes. This includes all device/stream pools owned by
the context, without changing the attention arithmetic or production guard.

A cache-full return during capture is deferred through the first graph launch.
Only devices with pending releases are synchronized; buffers are then released,
the epoch advances, and the context device is restored. Existing graph
instantiation/update OOM fallbacks release deferred scratch before retrying the
direct evaluator. A genuine scratch allocation OOM during capture retains the
existing fatal allocation-error contract, before attempting an illegal capture
synchronization/cache flush. No new exception unwinding is introduced.

## Hardware regression

The white-box fixture compiles the actual CUDA implementation with the build's
exact flags, avoiding a public backend test API. It uses tiny kernels and no
model, Xbox, or server. It checks exact numeric output after:

- physical scratch release following capture, invalidation and recapture;
- full-cache eviction during capture, deferred first-launch lifetime, draining,
  invalidation and recapture.

```sh
python3 tests/run-cuda-graph-pool-regression.py \
  --build-dir build-optimized \
  --output-dir /var/tmp/cuda-graph-pool-regression \
  --execute --memcheck
```

The build must enable CUDA graphs, disable VMM and export compile commands.
GPU execution is opt-in. A build-only invocation omits `--execute`.

The fixture and tracked runner passed both numeric cases; Compute Sanitizer
reported zero errors. An adapted negative control using the pre-fix CUDA
implementation detected a surviving stale executable and exited before unsafe
replay. The real-model S reproduction completed 512 outputs with graphs enabled
and produced the same 512 token IDs as the graphs-disabled control. Its log
showed one scratch invalidation followed by new captures.

All six production tiers completed a cold prefill near their configured context
limits and 512 outputs with the same final executable and libraries. The XXXL
run used batch 1024, ubatch 512, two N256 wavefront tiles, FP8 bulk async and
TCP prefetch. Its counters recorded 1,680 frames sent and completed, including
1,553 sent ahead, with at most two frames in flight. An independent XXXL-to-S
transition restored MTP and reused a live checkpoint with bounded suffix replay.
The wavefront pipeline and TCP prefetch regression tests passed.
