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

The host producer uses the real GGML CPU quantizer. CUDA KV packing is not
implemented, and an explicit non-CPU producer request is rejected. Inputs must
be finite and within ±65504; packed FP16 metadata is checked again. Server
metadata, contiguous upload state, query bytes/positions and CPU OUT/LSE are
validated before mutation or publication. Legacy op13/op14/op19 routes are
refused for these native-only sessions.

Activation transfer supports E5M2 and separately negotiated E4M3FN Q/OUT with
CPU codecs on the Xbox. E4 uses v3 and per-layout capability bits 33/34,
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
