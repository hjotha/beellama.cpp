# RKVA full-GPU qualification, 2026-11-08

Candidate: `53e4f7881f43bd12deceb2c56b37deaf71ff5b7a`, parent
`15cf144a94d721298704a9fdcefa1d275346cd56`. Tests used the candidate plus
local fixes described below, on an RTX 4070 and Xbox Series X. The model was
Swift-1.5 Qwen3.8 27B, SHA-256
`e23eb251493b8a89f7d0217590b593e696ac9bacaeac8605541e477533dc5838`.
The Xbox test packages were installed in place and rolled back; the running
production server and its settings were not promoted to this candidate.

## Corrections

* The generic optimized-kernel negotiation now advertises Q4_1 and IQ4_NL only
  when the corresponding LOW4 prerequisites and MQ2 pipeline are available.
  Native MQ2 profiles identify each format and count it separately; LOW4 upload
  telemetry counts both K and V copies.
* TurboQuant NC MQ2 failed on Xbox with both E4 and E5: the E4 case had 457 OUT
  interval failures and maximum LSE error 0.005278, while the same package and
  GPU codec using the per-query kernel passed. The corrected shader uses a
  synchronized group-shared WHT instead of lane-varying wave reads. The same
  failing case and the broader matrix passed afterward. The exact compiler or
  hardware cause of the previous lane-read failure is not independently proved:
  the corrected shaders were rebuilt with Windows SDK DXC 1.8.2502.
* The native-row profile test now verifies `cycle_id` at each of the ten call
  sites. The MQ2 host test checks the shared-memory WHT model for both waves.

## Correctness and failures

The optimized CUDA build used sm_89, KVarN and CUDA graphs enabled, with the
producer translation unit compiled with `--ftz=false --prec-div=true
--prec-sqrt=true --fmad=false` after the target fast-math option. Its PTX had no
F32 FMA or FTZ instructions. The explicit real-GPU producer test passed all 16
format/count combinations, host and device inputs, exact same-format packed
bytes, invalid input, chunking and rewind; it also passed with an AVX2 host
build and under compute-sanitizer memcheck with zero errors.

The final CPU CTest run had 16 passes and three skips: the GPU test was run as
above, and both Python/NumPy tests passed separately using the system Python.
The MQ2 host oracle passed 528 cases and 29,736,960 output elements under
ASan/UBSan; the native-row profile suite passed 18 tests. The host model does
not replace a GPU/barrier test.

Windows/UWP builds used MSVC 19.44 and SDK 10.0.26100.0. A control package had
LOW4 enabled with the new codec and MQ2 gates disabled; the candidate enabled
all three gates for testing. MSVC emitted an APPX2101 `cl.exe` warning in both
builds, but compilation and packaging completed without errors. The signed
candidate package was 1.6.0.1304, SHA-256
`887edac8164f7fa409e8858e92d2f8052b61ecef97a1f16cff25a8868624a5d7`.
All four rebuilt MQ2 DXIL blobs were found byte-for-byte in its executable.
The original payload was restored as 1.6.0.1313, SHA-256
`a2460041ebecd09752c46064b6d9d1d7e3b2493cf0461b29358d34f9cec08539`.

On the Xbox, the four native formats each passed E4 and E5 decoded-KV OUT/LSE
oracles, 42 cases per format/wire, including unsplit query counts 1, 2, 3, 9,
128, 255 and 256, three seed lengths and empty prefixes. The profile confirmed
MQ2 and its geometry in individual cases; aggregate matrix telemetry does not
prove the geometry of every individual dispatch. Q4_0 E4 passed 24 additional
cases; KVarN4 with F16 tail 128 passed 35 cases per wire. Native masks out of
order, before origin and in the future, plus rewind/reappend, passed for all
four formats and both wires. Two-layer sync versus async/prefetch matched OUT/LSE
exactly for each format/wire and actually sent a frame ahead.

GPU codec tests covered every finite E4 and E5 input code on the affected
formats, every finite output code for Q4_1, 758 direct-F32 E4 midpoint and
adjacent cases including saturation, and 1,498 F16-representable E4/E5 boundary
cases. NaN/Inf in a later segment was rejected before mutation, and a valid
retry passed. A finite SAW payload producing non-finite GPU output returned an
error with no output and invalidated the session. Separate test-only packages
injected an E4 or E5 PSO creation failure and confirmed correct CPU fallback;
an injected tiny nonzero result for an empty prefix and a non-finite LSE were
rejected without output. Those injected builds are not part of the candidate.
FIFO, reset, disconnect and malformed-envelope checks passed on Q4_1, IQ4_NL,
SAW, NC, Q4_0 and KVarN. A host proxy test rejected an altered response cycle
and a truncated response rather than publishing results.

## KLD and request measurements

Each model comparison below used context 2048, split 1024, a 1280-token prompt,
16 teacher-forced full-vocabulary steps, E4, 16 remote layers and MTP disabled.
Same-format CPU versus CUDA producers had mean/p99 KLD zero and 100% top-1
agreement for each format; Nsight traces confirmed 64 CUDA producer launches
per request and none in each CPU trace. These short checks are not a long-context
quality qualification.

| KV format | Per-query vs MQ2 mean KLD | p99 KLD | Top-1 agreement |
| --- | ---: | ---: | ---: |
| Q4_1 | 0.000419 | 0.002762 | 100% |
| IQ4_NL | 0.000215 | 0.001075 | 100% |
| SAW | 0.000319 | 0.001398 | 100% |
| TurboQuant NC | 0.000297 | 0.001430 | 100% |

All four met the project's metric gates, although free-running output sequences
in the request benchmark did differ between per-query and MQ2. Against the
15cf baseline, the unchanged KVarN/E4 path had mean/p99 KLD zero over the same
16 positions.

The following producer comparison uses the same model, E4, prompt, split,
16 layers and producer input, with 128 output tokens, one warmup and three cold
requests per arm. Values are medians: prefill TPS, decode TPS, total seconds.
The generated tokens matched between the CPU and CUDA producer arms.

| KV format | CPU producer | CUDA producer |
| --- | --- | --- |
| Q4_1 | 558.05 / 23.16 / 7.82 | 554.18 / 22.18 / 8.08 |
| IQ4_NL | 246.98 / 19.12 / 11.88 | 509.92 / 13.43 / 12.05 |
| SAW | 550.12 / 23.74 / 7.72 | 561.45 / 22.82 / 7.89 |
| TurboQuant NC | 533.43 / 23.49 / 7.85 | 557.87 / 22.17 / 8.07 |

Per-query versus MQ2 total medians were respectively 7.876/7.845, 11.941/11.923,
7.724/7.731 and 7.885/7.845 seconds for the same four formats. These are grouped,
not interleaved, samples with profiling on both sides. They establish neither
significance nor long-context performance, and do not justify enabling defaults.

CUDA resources on sm_89: 255 registers/thread, 1,472-byte stack, and ptxas
reported 540 bytes of spill stores and 540 bytes of spill loads. The occupancy
API returned a maximum of eight 32-thread blocks per SM. On the measured
128-token/four-head producer workload, Nsight Compute measured 2.08% active-SM
warp occupancy and about 0.71–0.72% across elapsed cycles, with a 16-block grid
on 46 SMs. IQ4_NL produced approximately 4.24–4.33 million local-load sectors
and 4.46 million local-store sectors per measured key/value launch. Local-memory
transactions include stack accesses and are not a count of spill instructions.
Counters were collected using the existing sudo privilege without changing
permissions or driver settings.

## Operational state and limits

The final production unit was active, `/health` and a short inference passed,
and it still used the pre-campaign release and configuration. Xbox 1.6.0.1313
was running the pre-campaign payload. Snapshot inventory remained 782 entries
with the same names/types/sizes hash; file contents were not hashed. No CI
approval is claimed. E4/E5 model KLD above is limited to the stated short E4
case; KVarN tails other than 128, long-context model quality and long-context
performance were not qualified. Experimental defaults and production settings
remain unchanged.

Detailed local logs, commands, package hashes, profiles, traces and machine-
readable comparisons are under `.codex/artifacts/rkva-fullgpu-20261107` in the
operator's home directory.
