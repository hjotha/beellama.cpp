# GPU ceiling suite (Series X, UWP sandbox)

The RKVA overflow tier spends most of its prefill time in one attention kernel
(`pass1`). This suite measures the device that runs it: memory bandwidth, ALU
peak, the fixed cost of one D3D12 dispatch, and the CU count the sandbox
actually exposes. It exists so an optimization claim can be placed against a
**measured** roofline instead of the published nominal specification.

Everything here is measurement infrastructure. It changes no production path:
the probe is a separate headless mode, and the pass1 sweep only drives the
existing RKVA diagnostic ops (`op 20` profiling, `op 22` bulk attention,
`op 14` decode attention).

## What runs where

| Piece | Path | Runs on |
| --- | --- | --- |
| Probe driver | `src/bridge/gpupeak.cpp`, `include/xllama/gpupeak.h` | Xbox (D3D12), Linux stub reports unavailable |
| Headless mode | `uwp/App.cpp` `gpupeak.flag` -> `bridge::run_gpupeak()` | Xbox |
| Shaders | `shaders/gpupeak_stream.hlsl`, `gpupeak_alu.hlsl`, `gpupeak_empty.hlsl` | compiled to DXIL by DXC |
| Embedded DXIL | `shaders/generated/gpupeak_*_dxil.h` | committed, no runtime compiler |
| DXIL manifest | `shaders/generated/gpupeak-shaders.json` | hashes, DXC version, exact flags |
| Regenerator | `scripts/compile-gpupeak-shaders.py` | local `dxc` or `--ssh-host` |
| pass1 sweep | `tests/rkva_pass1_sweep.py` | host, talks to the RKVA listener |
| Roofline model | `tests/rkva_ceiling_roofline.py` | host (pure Python, unit-tested) |
| Orchestrator | `scripts/run-xbox-ceiling.py` | host, inert without `--execute` |

## Probe plan

All rates come from a GPU timestamp pair recorded in the same command list as
the work; the resolve and the `Map` happen in later submits so they never land
inside a measured window. The timestamp frequency is read from
`ID3D12CommandQueue::GetTimestampFrequency` and written to the JSON; without it
the probe refuses to report a rate at all. On AMD this counter is a fixed
tick source (the existing 0.18 ms decode call reports 18,024 ticks, i.e. a
~100 MHz counter), **not** the shader clock, so it is reported as a tick rate
and never as the GPU frequency.

1. **stream** — 256 MiB by default (`stream_bytes=`, `run_big_stream=1` adds a
   1 GiB pass). Three kernels: pure read (`uint4` loads, XOR reduced), pure
   write, and copy. The write kernel stamps a deterministic pattern; a separate
   untimed read pass XOR-folds one accumulator per flat thread on the host and
   compares with `gpupeak_pattern_xor()`, so a read/copy rate is only reported
   when the whole buffer provably flowed through the kernel. Copy reports
   `2 * bytes` of traffic.
2. **alu** — `fma_f32` (8 independent FP32 FMA chains), `fma_f16`
   (`-enable-16bit-types`, packed `half2`), `fma_f16_min16` (relaxed
   `min16float2`, `cs_6_0`, a driver-acceptance data point), `dot2_f16`
   (`dot2add`, SM 6.4), `dot4_i8` (`dot4add_i8packed`, SM 6.4). Each is swept
   over 52/104/208/416/832/1664 groups of 64 threads, so the peak *and* the
   occupancy knee are visible. A dispatch that cannot create its PSO (older
   shader model support) is reported in `pso_errors`, never faked.
3. **latency** — a near-empty dispatch (`empty`): GPU time per dispatch
   (min/p50/p95), CPU submit+fence wall, readback `Map` cost, and the amortized
   per-dispatch GPU cost of 64 dispatches in one list. This is the per-call
   floor the 16 remote calls of a decoded token pay.
4. **occupancy** — LDS-limited group sweeps at 32 KiB and 16 KiB per group.
   D3D12 caps one workgroup at 32 KiB, so a 64 KiB CU LDS admits 2 or 4 groups
   per CU; the step positions of the total time versus group count give two
   independent CU estimates. The scheduler fit is done on the host
   (`rkva_ceiling_roofline.fit_concurrency`) and the two estimates must agree
   within 15%, otherwise the CU count is withheld.

The dependent-FMA chain (`latency_fma`) is a latency probe only. No clock is
inferred from it: the old 4-clock VALU assumption does not hold for wave64.
The roofline derives a clock from the *measured* FP32 peak divided by
52 dual-CUs × 128 FLOP/cycle and labels it derived, never measured.

A failed stream checksum no longer aborts the roofline. The stream rates become
observations only (`stream_observed_gbs`), `bandwidth_gbs` is forced to zero,
and the result carries `"partial": true` with the warnings while the ALU,
latency and occupancy sections remain valid.

## Plan options (`LocalState\gpupeak-opts.txt`)

```
# key=value, '#' comments; unknown keys ignored
stream_bytes=268435456
run_big_stream=0
big_stream_bytes=1073741824
iterations=5
run_stream=1
run_alu=1
run_latency=1
run_occupancy=1
```

`gpupeak.flag` triggers one headless run; the app writes
`LocalState\gpupeak-result.json` and `gpupeak-result.json.done`. The flag is
consumed like every other headless operator flag.

## Regenerating the DXIL

```
# on a Windows host with dxc on PATH
tools/xbox/scripts/compile-gpupeak-shaders.py

# from this Linux checkout against an isolated Windows scratch dir
tools/xbox/scripts/compile-gpupeak-shaders.py --ssh-host hjotha@192.168.1.193 \
    --remote-root 'C:/Users/hjotha/build/oc-probe-fixes-20261026'

# refuse a committed header whose embedded source hash is stale
tools/xbox/scripts/compile-gpupeak-shaders.py --check
```

The script rewrites `shaders/generated/gpupeak_*_dxil.h` and the manifest. It
never builds the app and never writes outside its scratch directory. `--check`
exists because the first Series X run used a stream_read DXIL that predated the
`mode == 1` validation store; the header and the source were in the same commit
but the header had not been regenerated.

## pass1 sweep (production shapes)

One session per KV row count, native KVarN4 descriptor revision 3, FP8 wire,
`q4_kernel_mode=2` (the qualified "fine" policy):

* `nq == 1`: `op 14` — the decode path, `kvarn4_decode_fine` on wave64.
* `nq >= 8`: `op 22` envelopes of `ceil(nq/8)` segments with 8 queries each,
  appending exactly the queries' rows, `kvarn4_mq2_fine` per segment — the
  production prefill shape. Every segment appends, because the revision-3 bulk
  validator requires `append_rows == n_queries` for non-reset segments.

Per run: `op 20` `ENABLE|RESET` (mode 2), one attention call, then
`op 20` `REPORT_ONLY`. The point is rejected (fail closed) if the profile shows
no GPU timestamps, a different kernel than the qualified one, missing
segments, or any FP8 CPU fallback (`fp8_cpu_fallback_calls != 0`).

Grid: `nq ∈ {1, 8, 64, 256}` × `rows ∈ {4096, 8192, 16384, 28672}`,
origin 102400, tail 128.

## Roofline formulas

One attention pass1 for one layer:

```
MAC   = 2 * nq * n_head * rows * head_dim        (QK scores + PV output)
FLOP  = 2 * MAC                                  (one multiply + one add)
      = 4 * nq * 24 * rows * 256
```

The WHT of Q and the inverse WHT of OUT add
`2 * nq * n_head * 2 * (8 * 256)` FLOP — below 0.1% of the matmuls at 28k rows
and reported separately.

KV bytes (unique, compulsory):

```
records = floor(rows/128) * n_head_kv * 2 slices * 8960 B * 2 planes
stage   = min(256, partial_group + 128) rows * n_head_kv * head_dim * 2 B * 2
tail    = min(rows, 128) rows * n_head_kv * head_dim * 2 B * 2
```

At 28,672 rows this is 30.6 MiB of records + ~1 MiB stage/tail ≈ 31.2 MiB,
the figure the kernel analysis used. The JSON reports two bandwidth columns:

* `gbs_unique` — unique KV bytes over measured time (lower bound on traffic).
* `gbs_issued_upper` — `unique_kv * nq` over measured time (upper bound; no KV
  reuse at all).

The true issued traffic for a split-K kernel lies between them; the report does
not invent a single value for it. Because both bounds are tiny at 28k rows
(< 1 GB/s even for the upper bound), pass1 sits on the compute side of the
roofline by a wide margin, and `bound` in the JSON says so explicitly.

## Nominal versus measured

| Quantity | Published nominal (Series X) | Measured by this suite |
| --- | --- | --- |
| FP32 | 12.15 TFLOP/s (52 CU × 64 lanes × 2 × 1.825 GHz) | `fma_f32` sweep, best TFLOP/s |
| FP16 packed | 24.30 TFLOP/s | `fma_f16` and `dot2_f16` best TFLOP/s |
| INT8 | 48.6 TOPS | `dot4_i8` best GOPS |
| Bandwidth | 560 GB/s (10 GB pool, 320-bit) / 336 GB/s (6 GB pool) | stream read / write / copy best GB/s |
| Compute units | 52 dual-CU | occupancy step fit, two LDS sizes must agree |
| Shader clock | 1.825 GHz | no UWP API exposes it: the timestamp counter frequency is recorded (a fixed counter, not the shader clock) and a derived clock comes from the measured FP32 peak / (52 CU × 128 FLOP/cycle) |
| GPU budget | ~4.35 GB reported for this app | `DXGI_QUERY_VIDEO_MEMORY_INFO` budget |

The nominal column is the denominator only. A measured value below it is a
result, not an error: the sandbox may cap clocks, memory or occupancy.

## Running the window

```
export XBOX_IP=... XBOX_USER=... XBOX_PASS=...   # or source ~/.config/xllama/xbox-env
tools/xbox/scripts/run-xbox-ceiling.py                     # plan only, no connection
tools/xbox/scripts/run-xbox-ceiling.py --execute \
    --expect-pfn 'GianlucaMazza.xllama_1.6.0.296_x64__<publisher-hash>' \
    --output .oc-artifacts/xbox-ceiling-result.json

# B2: probe only (no listener/sweep), with per-row checksums and the partial flag
tools/xbox/scripts/run-gpupeak-probe.py                    # plan only
tools/xbox/scripts/run-gpupeak-probe.py --execute --host "$XBOX_IP" \
    --output .oc-artifacts/gpupeak-probe-result.json
```

The package (a `beellama-xbox-*-llamacpp.msix` that contains the `gpupeak`
headless mode) must already be installed; building and deploying it is the
reviewer's step. The script refuses to run while a host benchmark process is
active, deletes stale result files, and records every command in the output
JSON.
