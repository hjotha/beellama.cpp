# Native Q4_1 and IQ4_NL remote overflow (experimental)

This implementation connects the pre-existing Q4_1 and IQ4_NL Wave64 shaders
to the TCP native-row pipeline. Review found that their shared shader template
broadcast Q from inactive lanes on partial/causal tiles. The template now
executes broadcasts uniformly while guarding K/V loads; the Q4_1, IQ4_NL and
SAW-INT4 DXIL headers were regenerated with official Microsoft DXC 1.9.2609.5
using the existing cs_6_0 profile. This does not establish Windows compilation,
D3D12 pipeline creation, GPU numerical parity, model quality or production
readiness.

## Representation and negotiation

- Q4_1: layout 9, storage bit 6, capability bit 31. A head contains eight
  20-byte GGML blocks: FP16 scale, FP16 minimum, 16 packed nibble bytes.
- IQ4_NL: layout 10, storage bit 7, capability bit 32. A head contains eight
  18-byte GGML blocks: FP16 scale and 16 packed nibble bytes indexing the
  nonlinear IQ4_NL table. Equal byte size does **not** mean Q4_0 compatibility.
- Existing v2/v3 framing and op15/op16/op22 structures are unchanged. New
  descriptors are additive; old peers without the exact capability are refused.
- K/V are matching types, in original domain, token-major/head-major, K plane
  then V plane. No rotation, KVarN records, stage ring or precision tail.
- Logical position-split identities are 6 and 7. Mixed state stores the real
  GGML IDs (3 and 20), never the Q4_0 ID. Native Xbox disk snapshots are not
  enabled for these formats. The local companion checkpoint codec preserves
  its type and bytes.

## Gates and limits

The host requires `GGML_RKVA_EXPERIMENTAL_LOW4=1` and an explicit
`--remote-attn-format q4_1` or `iq4_nl`, together with matching
`--remote-attn-cache-type-k` and `--remote-attn-cache-type-v`. The existing
Qwen3.5 TCP position-split requirements remain: one non-unified slot, offload,
Flash Attention, no paged cache and the existing placement constraints.
Vulkan is not enabled by these new TCP types.

The Xbox engine requires compilation with `RKVA_EXPERIMENTAL_LOW4=1`, both
corrected shader headers, the corresponding successfully created PSO,
required root signatures/reducer, head dimension 256, GQA ratio 6 and fixed
Wave64. The default build leaves the gate off and advertises neither format.
Missing capabilities or incompatible options fail explicitly. There is no
silent Q4_0 or F16 representation fallback.

The host producer defaults to the real GGML CPU quantizer. Experimental CUDA
KV packing is selected by `GGML_RKVA_LOW4_PRODUCER=cuda` explicitly. It requires the CUDA backend with
`GGML_CUDA_KVARN=ON`; a missing helper or failed prepare rejects the operation,
without a CPU retry. The scalar CUDA kernels preserve reference arithmetic
order and IQ4_NL uses the same 15-search quantizer as `ggml_quantize_chunk`.
Host oracle comparisons do not qualify hardware parity.
Inputs must
be finite and within ±65504; packed FP16 metadata is checked again. Server
metadata, contiguous upload state, query bytes/positions and CPU OUT/LSE are
validated before mutation or publication. Legacy op13/op14/op19 routes are
refused for these native-only sessions.

Activation transfer supports E5M2 and separately negotiated E4M3FN Q/OUT.
The Xbox GPU codec candidate covers both wires for LOW4 only with
`RKVA_EXPERIMENTAL_NATIVE_FP8_GPU=1` (default 0), its existing LOW4 experimental
gate and codec PSOs available. The existing Q4 codec-off flag
or unavailable PSOs retains CPU codecs; GPU execution failure never retries on
CPU. See [native raw FP8 codec](rkva-native-raw-fp8-gpu.md) and
[local GPU qualification](rkva-fullgpu-local-qualification.md) for tested scope
and outstanding Windows/Xbox qualification. E4 uses v3 and per-layout capability bits 33/34,
in addition to the native storage capabilities; old peers cannot qualify it.
Q and OUT round directly from F32 to E4 with ties to even, finite saturation,
and no intermediate F16 rounding. Invalid queries, metadata, mixed-wire
segments, OUT/LSE and incomplete envelopes fail closed. Native K/V remain
packed on the server GPU. Async op22 and TCP prefetch use the existing ordered
pipeline and its original admission limits, including prefetch nq >= 10.
Batches are at most 256 queries, 32 segments. Rewind trims native rows with
op8. The host and Xbox experimental gates apply equally to E5 and E4;
requesting E4 does not enable a default-off server pipeline.

## Local validation

With a CPU CMake build:

```
cmake --build build --target test-rkva-low4 test-remote-attn-range \
  test-arg-parser test-position-split-ranges test-kv-mixed-state
ctest --test-dir build -R '^(test-rkva-low4|test-rkva-low4-loopback|test-rkva-low4-state|test-position-split-ranges|test-arg-parser)$' --output-on-failure
python tests/test-rkva-low4-shader.py build
python tests/test-rkva-rowlane-uniform.py
```

The producer suite checks exact GGML bytes, descriptor identity, negative
metadata/query cases, rollback and publication on failure. The scalar loader
script extracts the actual HLSL loader functions and executes their aligned
reads/table lookups through a C++ shim, comparing every value with GGML. This
is a packing/loader check, **not** HLSL compilation or GPU execution.

The loopback suite runs the real host backend over TCP in sync/async modes,
nq 1/9/128/256, both formats, E5M2/E4M3FN bytes, prefetch admission counters, rewind
and capability/type/gate refusal. The loopback replies are synthetic and do
not prove attention numerics. Compact companion state roundtrips check that
even equal-size IQ4_NL/Q4_0 rows cannot be restored under the wrong identity.

Required before qualification: Windows build with the explicit gate, D3D12
PSO creation on the target, OUT/LSE parity against decoded K/V, multi-layer
async/prefetch fault testing, and model/KLD regressions with exact model,
command, sampling, hardware and commit recorded. No benchmark claim follows
from CPU tests.

## Rebuilding the corrected shaders

Use Microsoft's official DirectXShaderCompiler distribution and run:

```
DXC=/path/to/dxc bash tools/xbox/scripts/compile-rkva-range-shaders.sh \
  attn_q4_1_decode_rowlane attn_iq4_nl_decode_rowlane \
  attn_saw_int4_decode_rowlane
python tests/test-rkva-rowlane-uniform.py
```

Only these three generated headers are changed. Regeneration is required when
the shared template or SAW source changes; source/include hashes embedded in
the headers are checked by the regression. The test also verifies uniform Q
broadcasts and guarded row loads and rejects the old divergent source. Shader
compilation and provenance checks do not replace partial-tile numerical tests
on the target GPU.

## Experimental remote CUDA producer qualification

The four native producers now expose real CUDA kernels through
`ggml_backend_cuda_low4_native_prepare_v1`. Selection is opt-in and remains
independent of the Xbox gates. Per-format variables are
`GGML_RKVA_LOW4_PRODUCER`, `GGML_RKVA_SAW_PRODUCER`, and
`GGML_RKVA_TQNC_PRODUCER`; each accepts `cpu` or `cuda`. CUDA requires the
format-specific variable explicitly. Generic `GGML_RKVA_NATIVE_PRODUCER=cuda`
alone still fails for LOW4. For SAW/NC it preserves historical CPU behavior
and logs the required specific opt-in: historical captures could contain that
variable while those producers ignored it. It does not incidentally activate
new kernels or claim CUDA. A specific `cuda` with no helper fails, without retry. A format-specific `cpu` overrides the generic setting.
Missing backend/helper, invalid selection or an
unsuccessful CUDA prepare fails closed. CUDA permits 1..32 heads and up to
512 tokens per prepare, with 256 elements per head. HIP/MUSA do not register
this experimental ABI.

The kernels are deliberately scalar per head to preserve sequential CPU
reduction order. This is a correctness candidate, not a performance claim.
SAW selects the scalar or AVX2 arithmetic contract matching the caller's
compiled CPU codec through an explicit ABI flag. That includes deferred WHT
scaling, reciprocal multiplication and signed-zero min/max tie behavior.
NC retains distinct K/V records and two zero padding bytes in each K record.
No format is reinterpreted as Q4_0. Q4_1 rejects a nonzero scale whose
reciprocal overflows; the CPU reference integer conversion is undefined for
that extreme-subnormal case. Ordinary tiny/subnormal cases are tested.

`native-low4-producer.cu` has source-specific NVIDIA flags `--ftz=false`,
`--prec-div=true`, `--prec-sqrt=true`, `--fmad=false`, placed after target
options by CMake. Other CUDA translation units retain their existing flags.
The standalone translation unit was compiled locally with official NVIDIA
NVCC 13.1.115 for sm_86 and sm_89, including a minimal CMake target reproducing
the target/source option order. Generated PTX contained no `.ftz` or FMA
instructions, and verbose compiler flags confirmed gradual underflow and
precise division/sqrt. The full root-CMake `ggml-cuda` target also compiled and
linked on 2026-11-07 with Release, NVCC 13.1.115/GNU 14.2, sm_86,
`GGML_CUDA_KVARN=ON`, `GGML_CUDA_FA=OFF`, and `GGML_NATIVE=OFF`. The new helper
symbol is defined/resolved and its registration string is present. The registry
translation unit also compiled with KVarN OFF; that was not a full OFF build.
These are compile/link checks, not GPU runtime qualification. GPU numerical,
subnormal and device-pointer cases remain unexecuted. Host tests execute the
shared scalar routines, not CUDA kernels.

Reproducible hardware qualification after a CUDA build (exclusive GPU use):

```sh
cmake --build build --target test-rkva-low4-gpu-producer
GGML_RKVA_GPU_TEST=1 ctest --test-dir build -R '^test-rkva-low4-gpu-producer$' --output-on-failure
```

Without `GGML_RKVA_GPU_TEST=1`, this test exits 77 and CTest reports skipped.
An explicitly requested run without CUDA/helper is a failure, not a skip.
It compares actual CUDA host/device-input bytes with CPU owner payloads for
all four formats, counts 1/3/128/512, zeros, tiny/subnormal inputs, slice
publication, rewind invalidation, failed prepare, nonfinite input and invalid
flags/counts. This is still not Xbox runtime, model quality, throughput, or
complete host-free pipeline qualification.

CUDA fault paths discard destroyed scratch handles before attempting to restore
the caller's device, so a failed `cudaSetDevice` restoration cannot make a later
call reuse freed allocations/streams. Cleanup or restore errors reject rather
than proceeding, and scratch growth similarly discards uncertain handles and
rejects a failed free before allocating again. Each K/V kernel launch checks its own
launch error before proceeding, synchronizes and rejects on failure. These
paths received source review, compile checks and host-only fault injection with
fake CUDA APIs. Device-switch failure, launch failure and recovery still require
fault injection on a real CUDA runtime.

Host lifecycle regression (no CUDA compiler or device required):

```sh
python3 tests/test-rkva-low4-cuda-lifecycle.py
# Optional: retain extracted C++ and logs for review.
python3 tests/test-rkva-low4-cuda-lifecycle.py --output-dir /tmp/rkva-lifecycle
```

The test extracts the actual producer host orchestration, substitutes fake CUDA
APIs/kernel launches, and compiles it with a GCC-compatible C++17 compiler
(`CXX`, default `g++`). ASan/UBSan are enabled by default; `--no-sanitize` is an
explicitly weaker run. LeakSanitizer is disabled for the retained process-lifetime
scratch and ptrace-constrained test environments. Artifacts use a temporary
directory unless `--output-dir` is supplied.

It covers failed device restoration and retry, failed old-device selection,
free/stream-destroy errors, scratch growth failure/recovery, and separate K/V
launch errors without downloading results. A checked negative mutation removes
the pre-restoration scratch reset and must fail specifically on destroyed-stream
reuse. This validates host control flow, not CUDA execution, asynchronous driver
cleanup semantics, numerical GPU parity or hardware memory/race safety.
