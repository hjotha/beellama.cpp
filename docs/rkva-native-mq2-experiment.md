# Native low4/SAW/NC shared prefill MQ2 experiment

Status: experimental, **disabled by default**. Host numerical/scheduling tests
and DXC/DXV validation are not Windows compilation, GPU execution, model quality,
or performance qualification. No claim of speedup or CPU-free remote execution.

## Algorithm and format contract

`tools/xbox/shaders/rkva/rkva_range_attn_native_mq2.hlsl` specializes Q4_1,
IQ4_NL, SAW-INT4 and TurboQuant NC into separate SM 6.0 PSOs. It requires
D=256, GQA=6 and a device reporting **fixed Wave64**. Each 128-thread group
owns one KV head, two queries and one split. Wave 0 loads/dequantizes one K/V
row into 2 KiB of LDS; both waves consume those same values. This is actual
shared K/V work, not two independent rowlane dispatches.

Every row has a publication and consumption group barrier. The row loop bounds
are group-uniform and depend on active rows/split, not query position. An odd
query tail still participates in all barriers and wave operations, while its
query, position and output memory accesses are suppressed. Each real query uses
its own causal limit. Fully masked queries produce zero OUT and negative-infinity
LSE. Split records use the existing 258-float layout and reducer. No new
workspace allocation is needed.

Q4_1/IQ4_NL use the existing native scalar loader and quarter permutation.
SAW loads its actual scale/zero/split-half-nibble format, applies BDR16 to Q,
accumulates in the rotated domain and applies BDR16 to each split result.
NC loads its adjacent-nibble centroid K, original FP16 norm and centroid-norm
correction; it applies normalized WHT256 to Q and leaves V/OUT in the original
domain. NC does not become SAW or Q4_0. Different reduction order can produce
small floating-point differences; byte-identical attention results are not
promised.

This first candidate processes rows serially within each group. Its purpose is
a bounded shared-prefill implementation, not an assumed improvement over the
existing rowlane shader. Barrier cost, occupancy and reduction cost must be
measured before enabling it outside an experiment.

## Gates, fallback and telemetry

Compile the Xbox engine with `RKVA_EXPERIMENTAL_NATIVE_MQ=1` to opt in; the source
default is `0`. Q4_1/IQ4_NL still independently require
`RKVA_EXPERIMENTAL_LOW4=1` and their baseline native pipelines. No defaults or
negotiated storage identities change.

Each MQ PSO is optional, requires its generated header, successful PSO creation,
and fixed Wave64. Selection requires an existing baseline format PSO, at least
two queries, supported geometry, and a non-reference Q4 kernel mode. If these
conditions are not met, the existing per-query GPU attention remains selected.
This selection fallback is not a retry after GPU execution failure. Successful
MQ selection reports `q_tile=2`, query group count `ceil(n_queries/2)`, and clears
`decode_opt`/`decode_rowlane`; the old path retains its previous telemetry.

## Reproduce local checks

From repository root, with a CPU build of this same source containing shared
GGML libraries in `$BUILD/bin`:

```bash
python tests/test-rkva-native-mq2.py "$BUILD"
ASAN_OPTIONS=detect_leaks=0 python tests/test-rkva-native-mq2.py "$BUILD" --sanitize
DXC=/path/to/dxc bash tools/xbox/scripts/compile-rkva-range-shaders.sh \
  attn_q4_1_mq2 attn_iq4_nl_mq2 attn_saw_int4_mq2 attn_turboquant_nc_mq2
for format in 0 1 2 3; do
  /path/to/dxc -T cs_6_0 -E CSRangeAttnNativeMq2 -D NATIVE_MQ_FORMAT=$format \
    tools/xbox/shaders/rkva/rkva_range_attn_native_mq2.hlsl -Fo /tmp/mq-$format.dxil
  /path/to/dxv /tmp/mq-$format.dxil
done
```

Leak detection is disabled in the sanitizer command because this dot environment
reports that LeakSanitizer cannot run under ptrace. AddressSanitizer and UBSan
still run for the oracle executable; the linked release GGML library is not
instrumented by that command. This is not a full-library sanitizer qualification.

The test extracts the actual HLSL scalar loaders/codebook into bounded C++
buffer shims and compares them with actual GGML encoders/dequantizers. It then
emulates MQ2 scheduling and split softmax, using a dense double-precision
original-domain reference. It exercises two KV heads/GQA6, all four formats,
queries 1/2/3/9/17/128/255/256, rows 1/3/65/129, splits 1/2/5, odd tails,
nonmonotone positions, fully masked queries and positions up to INT32_MAX.
The test does **not** execute DXIL, emulate GPU barriers, or establish the
hardware reduction order.

## Required real-device qualification before adoption

1. Build the existing Windows/UWP solution normally and with
   `RKVA_EXPERIMENTAL_NATIVE_MQ=1` (also LOW4's gate when testing LOW4). Keep
   device, model, geometry and all other settings identical. Do not enable a
   default or replace a deployed package solely because DXC passes.
2. Run each format with MQ gate off/on; prove `q_tile=2` and expected group counts
   when enabled, and q_tile=0 for nq1, reference mode and disabled gate. Test
   omitted MQ headers/failed optional PSO creation without claiming MQ support.
3. Compare captured OUT/LSE against the same packed K/V and query reference for
   nq2/3/9/128/255/256, tail/partial splits, masked and nonmonotone positions,
   nonzero origins, and supported multi-head geometries. Test both direct and
   split-reduce outputs; use debug layer/device validation for bounds/barriers.
4. Compare model logits/KLD at matching batch/ubatch and context, covering decode,
   prefill and speculative verification/rollback. This must include NC norm
   correction and SAW transform numerics, not only LOW4 loader byte parity.
5. Benchmark GPU timestamps, total RPC latency, memory and throughput for the
   exact hardware/model/commit/command. Expect possible regressions from the
   serial row loop; retain the default-off gate until results justify a change.

The remote path still includes CPU validation/orchestration, staging/readback,
network I/O and other host work even when this attention PSO and FP8 codecs run
on GPU.
