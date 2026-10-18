# Xbox positional overflow

RKVA range attention divides each full-attention layer by token position.
With P=102400 and C=131072, CUDA owns `[0,102400)` and Xbox owns
`[102400,131072)`. Projections, RoPE, recurrent state, FFN, and the final
attention merge stay local. This differs from the original RKVA operation
that delegates complete layers.

## Configuration

```sh
GGML_RKVA_RANGE_STORAGE=auto build-optimized/bin/llama-server \
  -m model.gguf --device CUDA0 -ngl 99 -c 131072 -b 256 -ub 256 \
  --parallel 1 --flash-attn on --fit off --load-mode none \
  --cache-type-k kvarn4 --cache-type-v kvarn4 \
  --spec-draft-n-max 0 --no-context-shift \
  --kv-position-split-p 102400 \
  --remote-attn 192.168.1.26:50054 --remote-attn-layers full \
  --remote-attn-cache-type-k q4_0 --remote-attn-cache-type-v q4_0 \
  --remote-attn-stats
```

P must be a positive multiple of 256 and smaller than C. In an adaptive
server, retain the pure CUDA tier at C=102400 and enable remote attention only
in the larger tier using `--remote-attn-min-context 131072`. Requests whose
budget selects the pure tier do not construct an RPC context. Within a split
context, only queries at positions P or later execute range attention RPCs.
Creating a split context can negotiate a connection and reserve remote KV
before those positions are reached.

The initial range path supports one contiguous, unshifted, causal text
sequence with Qwen35 dense full-attention layers, a CUDA KVarN prefix and a
Q4_0 or Q8_0 companion (matching K and V types). It rejects holes or changed
absolute positions rather than silently reading a different history. SWA, attention sinks, ALiBi and softcap
are outside this path. The initial serving and comparison workload disables
MTP in the split profile. Only one active RPC context per server process is
qualified; the existing remote-backend selector is process-global.

## Storage and transport

The host retains a compact CPU companion for checkpointing and
resynchronization. It sends only newly appended overflow rows during ordinary
inference; it does not transmit the full history for every token. The prefix
never enters the RPC cache. A restored or rewound range is uploaded again
before it is read remotely. New overflow K/V is quantized directly from the
model's projected K/V into the selected companion type; existing CUDA KVarN
records are neither converted nor uploaded. Q8_0 offers finer overflow quantization
than Q4_0, but remains lossy and cannot guarantee identical logits.

`GGML_RKVA_RANGE_STORAGE` selects the remote representation:

- `auto` negotiates native storage matching the host companion, otherwise F16.
- `q4_0` or `q8_0` requires the matching host cache and server capability and
  transmits the packed records directly. To use Q8_0, set both
  `--remote-attn-cache-type-k q8_0` and `--remote-attn-cache-type-v q8_0`.
- `f16` expands the selected host Q4_0 or Q8_0 records to F16 for the remote
  cache. It does not recover precision lost during source quantization.

Unknown values and missing required capabilities fail closed. With 16 layers,
4 KV heads and head width 256, the 28672-position overflow requires
528482304 bytes of Q4_0, 998244352 bytes of Q8_0 or 1879048192 bytes of F16 payload, excluding
workspace, metadata and application memory. These are layout calculations,
not a model-capacity or performance guarantee. The CPU checkpoint mirror also
consumes host RAM. CPU and Vulkan Q4_0 stores can differ in rounding; matching
the format alone does not establish numerical parity. RPC statistics distinguish
`range_source_cache_type` (host quantization) from `range_cache_type` (Xbox storage).

## Protocol and correctness

The v2 range extension negotiates the position origin and capacity, uploads KV
rows, and returns both OUT and natural-log softmax normalization (LSE) in F32.
The host combines local and remote OUT using their LSE weights; adding or
averaging normalized attention outputs would be incorrect. Global positions
remain unchanged for RoPE and causal masking, while the remote physical row
is `position-P`.

The Xbox implementation retains legacy v1 whole-layer commands. A client
requiring ranges rejects a server without the range capabilities; it does not
silently substitute Vulkan. Snapshot layout validation remains authoritative:
native snapshots from different device layouts must not be force-restored.
Full restore invalidates the remote upload epoch, and the CPU companion is
the source for resynchronization. Session clear and rewind cannot leave stale
future keys visible.

`test-remote-attn-range` exercises the host against a loopback oracle without
a model or GPU. It covers protocol rejection, Q4/Q8/F16 transport, OUT/LSE
packing, incremental uploads, rewind and invalidation. Hardware qualification
also requires GPU attention tests, full-vocabulary model comparisons at the
boundary, snapshot replay, zero attention RPCs below P, and filled-context
benchmarks. Allocating C alone does not demonstrate throughput at C tokens.

## RPC batching and profiling candidate

The candidate protocol adds capability-negotiated operation 19, which sends the
last KV upload chunk and its queries in one request. Earlier chunks of a large
resynchronization still use operation 13. OUT and LSE, quantization, absolute
positions and checkpoint invalidation retain their existing semantics.

`GGML_RKVA_RANGE_FUSE=auto` (default) uses this operation when advertised; an older
server uses separate upload and attention requests. `0` forces separate requests
for comparison, and `1` requires fusion and rejects missing support before range
configuration. For a normal appended decode token this changes 16 uploads plus
16 attention requests into 16 combined requests. The initial server implementation
still submits upload and attention separately to the GPU; fewer network round
trips alone do not establish a TPS gain.

Operation 20 controls optional profiling and the candidate Q4 kernel. These
settings require a server advertising the profiling capability:

- `GGML_RKVA_PROFILE=1` enables recording; the default is `0`. Collected GPU and
  CPU spans are printed as a separate `remote_attn gpu_profile` JSON log when
  remote statistics are requested. GPU timestamps can be unavailable; inspect
  the report rather than treating CPU wall time as GPU time.
- `GGML_RKVA_Q4_KERNEL=reference` (default, or `0`) selects the existing kernel.
  `optimized`/`balanced`/`1`, `fine`/`2`, and `coarse`/`3` select candidate split
  policies. These require native Q4 transport. The optimized shader is limited
  to qualified single-query Q4 shapes; other shapes retain the reference path.

Profiling and candidate kernels are disabled by default. Per-request disk tracing
is also disabled; the diagnostic protocol trace flag is separate from aggregate
profiling. `range_fused_calls` counts combined wire operations, while
`range_upload_calls` and `range_attn_calls` count their semantic work. Do not add
the latter counters to infer the number of wire requests.

Console version 1.6.0.262 combines the native KVarN4 and disk-snapshot source
with batching, profiling and the candidate Q4 kernels. The range protocol,
combined Q4/Q8/F16 operation, and Q4 GPU attention have passed independent
numeric tests on the console, including a full 28672-row Q4 overflow. Disk
snapshot replay still requires separate tests. Package263 subsequently passed
108 independent GPU OUT/LSE and kernel-selection cases per format for Q4, Q8,
F16 and native KVarN4; the KVarN4 stage path also passed with precision tail0.

An exploratory 130816-input/64-output model test measured 8.55 tokens/s for
reference Q4 with separate requests and 12.38 tokens/s with the balanced kernel
and combined requests. These rates exclude the initial history upload; including
that upload in the short decode gives 5.15 and 6.29 tokens/s, respectively. There
was one run per configuration, MTP was disabled, and the model quality gate
remains unqualified. See the recorded test artifacts for the exact model,
commands, source/binary hashes, prompt and power settings; these numbers do not
establish production readiness.

## Native snapshots and RAM handoff

Pure KVarN caches with an exact precision tail require partial state containing
both recurrent state and the original stage/tail overlay. A recurrent-only
checkpoint cannot reconstruct the overlay after the live tail has advanced.
The partial-state reader rejects that older representation when an exact tail
is required. RAM handoff also verifies that every required tail row is present
before mutating the candidate GPU context.

A private CPU restore can acquire its checkpoint lineage only after validating
the originating full snapshot and matching its captured owner epoch. Native
partial-state ownership and geometry checks remain authoritative. A controlled
Radeon handoff regression now reproduces the cold target's greedy output before
and after warm checkpoint reuse; this is separate from Xbox model quality.

## Future native KVarN4 overflow

A separate Xbox candidate, version 1.6.0.260, implements native Bee KVarN4
record decoding with explicit stage and precision-tail uploads. Its implementation is included in 1.6.0.262 and has CPU
oracle and shader-compilation coverage; native KVarN4 GPU qualification remains
pending. The current Q4/RPC benchmark uses 1.6.0.262. Further Q8, F16 and KVarN4
kernel optimization is isolated in a future candidate so it cannot alter the
current Q4 measurements.

Bee can now produce and upload native overflow KVarN4 through the explicit
experimental path described below. General snapshot/replay support remains
pending; it is not a drop-in replacement for the production cache.
Using KVarN4 in both ranges requires that client work, GPU correctness tests,
checkpoint replay, and model quality/performance measurements. Converting the
existing Q4/Q8 companion into KVarN4 would not establish equivalence to native
KVarN4 generated directly from the model's projected K/V.

## Storage-qualified kernel selection

`GGML_RKVA_RANGE_KERNEL=reference|balanced|fine|coarse` opts into the
storage-qualified selector for Q4_0, Q8_0, F16 or native KVarN4. Optimized modes
require the server control reply to advertise that storage's kernel as available;
missing/unavailable reports fail closed. The historical `GGML_RKVA_Q4_KERNEL`
retains its Q4-only meaning. If both variables are set, their modes must agree.
The default remains reference. Profiling can be enabled separately.

Native KVarN4 now has an experimental append-only host producer consuming the
original projected F32 K/V, before the separate standard bookkeeping cache.
It preserves genuine CPU-produced native records, rotated F16 stage and the
original-domain F16 precision tail. It cannot resume a Q4/Q8 overflow snapshot
and has no native full-state snapshot codec yet. Automatic full saves are
skipped once native overflow exists; explicit full saves/restores and real
rewinds fail closed. See [the native experiment](xbox-native-kvarn4-experimental.md)
for the opt-in flags and memory cost. This is separate from general model
quality qualification and from the standard Q4/Q8 snapshot path.
