# Remote 4-bit GPU completion candidates

This change extends the implementation, not the hardware qualification. It was
prepared against `15cf144a94d721298704a9fdcefa1d275346cd56` on a Linux CPU-only
host. No NVIDIA device, D3D12 runtime, Windows SDK or Xbox was
available. An official NVIDIA CUDA 13.1.1 compiler toolchain was subsequently
provisioned in an isolated workspace for compilation without a GPU. The candidate was subsequently published on a separate review branch. Nothing
was deployed or run on the serving host/console. Existing experimental gates remain in force.

## Scope and contracts

| Native KV | New remote CUDA KV producer | New Xbox Q/OUT codec | New shared prefill |
| --- | --- | --- | --- |
| Q4_0 | Existing, unchanged | Direct F32 E4M3FN | Existing, unchanged |
| KVarN4 | Existing, unchanged | Existing, unchanged | Existing, unchanged |
| Q4_1 | Candidate | E4M3FN and E5M2 | Experimental MQ2 |
| IQ4_NL | Candidate | E4M3FN and E5M2 | Experimental MQ2 |
| SAW-INT4 | Candidate | Direct F32 E4M3FN | Experimental MQ2 |
| TurboQuant NC | Candidate | Direct F32 E4M3FN | Experimental MQ2 |

Native identities, metadata, K/V layout and negotiated wire capabilities are
not interchangeable. In particular, IQ4_NL is not Q4_0 despite equal record
size; NC has different key/value records; SAW retains BDR-16. IQ4_NL uses the
GGML chunk quantizer's search, not the cheaper reference-row quantizer.

The new CUDA producers require format-specific opt-in:
`GGML_RKVA_LOW4_PRODUCER=cuda` (Q4_1/IQ4_NL),
`GGML_RKVA_SAW_PRODUCER=cuda` or `GGML_RKVA_TQNC_PRODUCER=cuda`.
The old generic `GGML_RKVA_NATIVE_PRODUCER=cuda` is insufficient to activate
these candidates. LOW4 retains its rejection of that unsupported generic
selection. SAW/NC retain their historical CPU route in this case and emit an
explicit warning identifying the required format-specific CUDA setting. This
preserves old captures without falsely claiming GPU quantization. A specific
`cpu` selection overrides the generic setting; a specific `cuda` selection must
fail clearly if unavailable and must never silently fall back to CPU packing.
Old SAW/NC captures containing only the generic setting do not prove GPU use.

Native E4 rounds F32 directly to E4M3FN, ties to even, saturating finite values
at 448. KVarN's existing F16-intermediate E4 codec is intentionally unchanged.
Invalid queries and metadata are rejected before GPU mutation. Invalid OUT or
LSE must not be published; an empty prefix requires exact-zero OUT and negative
infinite LSE. GPU execution failure is not permission for a silent CPU retry.
The new E4 and LOW4 E5 routes additionally require
`RKVA_EXPERIMENTAL_NATIVE_FP8_GPU=1` (default zero), preserving existing default
codec routing. An unavailable codec PSO or disabled codec flag may retain the
existing CPU codec path before GPU execution is chosen. Existing Q4/SAW/NC E5
GPU routing does not depend on this new gate.

MQ2 is a separate, conservative candidate enabled only by
`RKVA_EXPERIMENTAL_NATIVE_MQ=1`. It requires fixed Wave64, D256/GQA6, successful
PSO creation and the native format's existing admission gate. Two query waves
share one dequantized K/V row through 2 KiB LDS, with independent causal masks
and uniform producer/consumer barriers, including an odd final query. A missing
MQ PSO retains the existing per-query GPU attention. There is no performance
claim, and the new MQ gate defaults to zero.

The conservative scalar CUDA kernel is not a throughput optimization. Resource
inspection of the final sm_86 root-CMake object reported 255 registers and
1472 bytes of stack per thread, with zero shared memory. This is a reason to
measure actual launch behavior, occupancy and latency on hardware, not to
assume acceleration from moving quantization to a GPU. For sm_86/sm_89, the
published limits are 255 registers/thread, 65,536 registers/block and 1,024
threads/block. This launch uses 32 threads, or 8,160 raw registers/block before
allocation granularity, below the published block bound. That arithmetic is
not a runtime launch test or occupancy measurement. The cubins expose no
explicit MAXNTID/REQNTID; actual function limits and stack behavior still need
device queries and execution. See NVIDIA's [technical specification table](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability).

Even when quantization and attention run on GPUs, this is not a CPU-free
pipeline. Validation, staging, readback, network transport and orchestration
remain host work. No removal of redundant staging/readback is claimed here.

## Local validation summary (2026-11-07)

- CPU CTest: 18 executed tests passed; one new GPU test explicitly skipped.
- Host reference: 16,384 packed comparisons per scalar/AVX2 arithmetic mode;
  raw E4: 1,001,025 F32 cases plus 254 finite decodes; MQ2: 528 modeled cases.
  Selected harnesses also passed ASan/UBSan. These are not GPU execution.
- Six new shader variants compiled with DXC 1.9.2609.5 and passed DXV.
- NVCC 13.1.115 compiled the new producer TU for sm_86 and sm_89. Manual and
  root-CMake object PTX had no FTZ/FMA instructions.
- The full root-CMake `ggml-cuda` target compiled/linked at 18:10:39 UTC with
  Release, GNU 14.2, sm_86, KVarN ON, FlashAttention OFF and native CPU flags OFF.
  The new helper symbol and registration resolved. KVarN OFF was checked only
  for the registration translation unit, not as a full backend build.
- No CUDA kernel execution, Windows/UWP build, D3D12/Xbox runtime, model/KLD
  measurement or performance qualification occurred.

## Reproduce the CPU checks from this repository

Use Linux, Bash, Python 3, Ninja, CMake and GCC/G++ with ASan/UBSan support.
Run from the repository root; no private workspace, external evidence bundle,
CUDA toolkit or GPU is required for this CPU suite. Choose a fresh build folder.
The GNU linker flag below reproduces the tested dependency-order workaround;
it is not a Windows/MSVC flag. These Python oracle harnesses use GCC-compatible
compiler flags and Linux shared-library layout; run them on Linux/WSL, not as
native MSVC tests.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Required only when LeakSanitizer cannot run (e.g. a ptrace-constrained runner).
# This disables leak checks, not ASan memory errors or UBSan checks.
export ASAN_OPTIONS=detect_leaks=0
B=build-rkva-fullgpu-cpu
cmake -S . -B "$B" -G Ninja -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXE_LINKER_FLAGS=-Wl,--no-as-needed
cmake --build "$B" --target test-rkva-low4-native-reference test-rkva-low4-gpu-producer test-rkva-low4 test-rkva-kvarn-producer test-rkva-turboquant-nc-cpu test-turboquant-nc test-saw-int4-codec test-remote-attn-range test-kv-mixed-state test-rkva-native-e4-cpu test-rkva-e4m3-cpu test-rkva-e4m3-policy-cpu test-rkva-kvarn-bulk-cpu -j6
ctest --test-dir "$B" --output-on-failure -R '(test-rkva-(low4|kvarn-producer|turboquant-nc|native-rows-e4|e4m3|native-e4|kvarn-bulk)|test-turboquant-nc|test-saw-int4-codec|test-remote-attn-range)'
python3 tests/test-rkva-low4-shader.py "$B"
python3 tests/test-rkva-native-mq2.py "$B" --sanitize
python3 tools/xbox/tests/rkva_native_e4_raw_verify.py
python3 tests/test-rkva-low4-cuda-lifecycle.py
```

Expected frozen-candidate CTest result: 18 executed passes and one explicit
GPU skip (77). Legacy GPU tests reporting exit 0 after printing SKIP are not
hardware evidence. The Python numerical tests are CPU models, not shader
execution; linked GGML libraries in this recipe are not fully sanitizer-built.
For real CUDA parity use the commands in
[`rkva-low4-experiment.md`](rkva-low4-experiment.md#experimental-remote-cuda-producer-qualification)
after a CUDA-enabled build. Set `GGML_RKVA_GPU_TEST=1` only with exclusive GPU
ownership; an explicitly requested test must fail if the backend is absent.
For all six new DXC/DXV shader builds, see
[`rkva-native-mq2-experiment.md`](rkva-native-mq2-experiment.md) and
[`rkva-native-raw-fp8-gpu.md`](rkva-native-raw-fp8-gpu.md).

## Enabling a qualification build (not deployment)

On the authorized Windows build host, a Developer PowerShell can append the
three experimental definitions to MSVC's `_CL_` environment variable for this
build only. Choose the installed SDK and a valid new package revision following
`tools/xbox/README.md`; do not reuse a deployed package revision blindly.

```powershell
$previousCL = $env:_CL_
try {
    $env:_CL_ = "$previousCL /DRKVA_EXPERIMENTAL_LOW4=1 /DRKVA_EXPERIMENTAL_NATIVE_FP8_GPU=1 /DRKVA_EXPERIMENTAL_NATIVE_MQ=1"
    # Run the normal authorized build command from tools/xbox/README.md here.
    # Inspect the actual compiler invocation to confirm all three definitions.
} finally {
    $env:_CL_ = $previousCL
}
```

The host still needs `GGML_RKVA_EXPERIMENTAL_LOW4=1` for Q4_1/IQ4_NL and the
matching explicit native format/cache types. Setting an Xbox compile definition
does not set host environment variables or qualify a device. The existing
`rkva-q4-fp8-gpu-off.flag` can still disable the native codecs. Reference attention
mode, missing PSOs or unsupported geometry do not qualify MQ2.

## Qualification still required on real hardware

1. Build the exact patched tree using the supported CUDA toolkit and target
   architecture, then run the new native producer parity test with explicit
   GPU-test opt-in and exclusive GPU ownership. Compare host and device inputs,
   every packed byte, sliced appends, rewind, invalid inputs, error handling and
   recovery. Record compiler flags: fusion, fast math and denormal behavior can
   change quantizer results even when a host version matches a CPU reference.
2. Build the Windows/UWP project with the existing LOW4 gate,
   `RKVA_EXPERIMENTAL_NATIVE_FP8_GPU=1` for the new codecs and
   `RKVA_EXPERIMENTAL_NATIVE_MQ=1` for MQ2. Confirm D3D12 root-signature and PSO creation, default-off behavior,
   absent-header/PSO behavior and fixed Wave64 admission. A successful DXC/DXV
   check does not establish those facts.
3. On the console, compare E4/E5 CPU-codec and GPU-codec Q/OUT paths for each
   format. Include all finite FP8 codes, signed zero, midpoint neighborhoods,
   saturation, NaN/Inf, malformed later segments, fully masked queries, exact
   zero versus tiny nonzero OUT and invalid LSE. Require rejection without
   publication or silent retry on GPU errors.
4. Compare MQ2 against both per-query GPU attention and an independent oracle
   over the actual decoded native K/V. Cover nq 1, 2, 3, 9, 128, 255, 256;
   origins other than zero; positions before origin, unsorted and beyond the
   live prefix; zero/partial/full splits; first and last K/V heads; and all four
   native formats. Measure OUT and LSE separately, specifying tolerances before
   evaluating results.
5. Exercise sync/async op22, prefetch, rewind and fault paths end to end. Then
   run representative model continuation and KLD with exact model hash,
   command, prompt, sampling, `-b`/`-ub`, hardware and patched tree hash recorded.
   Benchmark latency/throughput only after correctness and quality pass.

Do not promote defaults, advertise missing PSOs as support, or call the full
matrix hardware-qualified based on host tests, generated DXIL or simulated
attention alone.
