# TurboQuant 4-bit NC experiment

This is an experimental KV-cache implementation. It is not qualified for
production. The CPU codec, CUDA helpers and Xbox shader use a 256-element head
and implement the `turboquant_4bit_nc` preset from TheTom's vLLM fork at
`36fc048255d0bbdab05811d667182a965fe05936`:

- <https://github.com/TheTom/vllm-turboquant/tree/36fc048255d0bbdab05811d667182a965fe05936>
- <https://github.com/TheTom/llama-cpp-turboquant/tree/643956891dfdf51aecdbd05ca508dd21e21e8195>

The vLLM-derived files carry Apache-2.0 notices; the corresponding license is
in `licenses/LICENSE-vllm-turboquant`.

## Format and arithmetic

The key and value formats are deliberately different.

- **K:** Compute its F32 L2 norm, divide by `norm + 1e-8`, apply the normalized
  Sylvester WHT over all 256 elements, and select one of the D=256 Lloyd-Max
  centroids for each element. Store 128 bytes of adjacent 4-bit pairs (even
  element in the low nibble) followed by the original norm as FP16: 130 bytes.
  On decode, multiply the centroid vector by
  `fp16(norm) / sqrt(sum(centroid²) + 1e-16)` before the inverse WHT. This is
  the NC step; the stored norm is **not** already corrected.
- **V:** Do not rotate. Compute `scale = max((max - min)/15, 1e-8)` and
  `q = clamp(floor((v - min)/scale + 0.5), 0, 15)`. Store 128 bytes of
  adjacent nibble pairs followed by FP16 scale and FP16 minimum: 132 bytes.
  Decode as `q*fp16(scale) + fp16(minimum)`.

Norm correction does not guarantee small error for every input. For example,
the deterministic `0.1 + 2*cos(i)` test vector concentrates energy outside
the key codebook after WHT and produces about 0.519 per-element RMSE. The
independent codec oracle covers this case; model quality must be measured on
representative data rather than inferred from it.

The scalar implementation rejects non-finite inputs and metadata. It also
rejects a value block when rounding its scale and minimum to FP16 would make
the maximum decoded value overflow FP16. That fail-closed check is an explicit
difference from the upstream implementation, whose stored FP16 metadata may
produce an infinity for extreme ranges.

This is not the historical llama-cpp `turbo4` representation. The latter uses
128-element rotation groups, different signs and centroids, stores a corrected
norm, and rotates/centroid-quantizes values. The old BeeLlama type ID 43 now
belongs to Q6_0 and must not be reused. The experimental integration reserves
IDs 145 and 146 for NC K and V, respectively.

The theoretical KV payload for one D=256 head is 262 bytes for NC K+V versus
288 bytes for Q4_0 K+V. This excludes temporary F16 attention buffers,
alignment, weights, recurrent state and all transport overhead. The initial
CUDA attention path materializes both K and V to F16, so actual peak VRAM and
throughput must be measured rather than inferred from payload size.

## CUDA and Xbox paths

The CUDA helper uses 32-lane groups for WHT, quantization and decode. The
integration stages each complete `SET_ROWS` operation, reads a device
error flag and publishes rows only after validation succeeds. Conversion to
F16 also reports invalid metadata before attention can run. Because these
checks synchronize the stream, NC graphs are excluded from CUDA graph capture.
Destination indices that resolve to the same physical block are rejected
before publication; identical row numbers in distinct streams are allowed.
Negative row indices are ignored.
The earlier prototype's device `trap` was removed: it could corrupt the CUDA
context and require a process restart, so it was a blocker for promotion.

CPU backend graph execution rejects unsupported NC operations before planning
or execution; contiguous same-type raw copies remain available. The lower-level
`ggml_graph_compute` and NC row-codec wrappers bypass that backend admission and
can still abort on invalid data. Shared CUDA attention and launch paths also use
`CUDA_CHECK`/`GGML_ABORT` for some failures, so NC-specific validation does not
guarantee recovery from every CUDA error.

The first attention route is a correctness-oriented F16 materialization, not a
native packed NC FlashAttention kernel. RTX 4070 tests cover the codec,
transactional failure/reuse, attention against FP64, and model continuation
after snapshot, truncation and rollback. These tests do not qualify the format
for all models, speculative modes or production configurations.

An optional F16 exact tail of 128 or 256 tokens is available for the local
CUDA Qwen35 GQA24/4 target path with one non-unified sequence. This experimental
route rejects position splitting and speculative rollback. Legacy local state
round trips are tested; precision-tail state is not supported by the bounded
streaming restore path.

The experimental Xbox shader `CSRangeAttnTurboQuantNC` uses D=256, GQA=6 and
Wave64. It rotates Q with a threadgroup-shared WHT. K stays in the rotated
domain; V and the output stay in the original domain. To make every
ByteAddressBuffer load aligned, the RKVA wire pads each 130-byte K block with
two zero bytes; V remains 132 bytes. Thus each wire K or V plane uses
`132 * n_head_kv` bytes per token. The native protocol uses descriptor layout
8, storage bit 5 and capability bit 24. Query/output FP8 wire quantization,
when enabled, is a separate source of error that must be included in the
end-to-end oracle.

DXC 10.0.26100.0 compiled the shader as `cs_6_0`. On Xbox Series X, a
qualification build passed ten op22 FP8-wire comparisons against an independent
CPU attention oracle: 1, 63, 64, 65 and 257 rows, each with one and eight
queries. The checks include the returned output and F32 LSE. Model-level KLD
and request measurements have also been run on the matching local/Xbox NC
path. These measurements do not establish general model quality, snapshot or
speculative-mode support, or production readiness.

## Evaluation

For the existing Swift 27B Qwen35 baseline, the local BF16-KV reference is
`/home/hjotha/.codex/artifacts/saw-int4-ab-20261104/local-kld-20k/bf16.kld`.
It was recorded on WikiText-2 with `-c 20480 -b 128 -ub 32`, one sequence and
one chunk. Reuse it only with the same model weights, corpus, tokenization,
batching and evaluation settings. Measure mean and tail KLD, prefill TPS,
decode TPS, wall time and peak memory for NC, Q4_0 and BF16. Keep the local
test and an Xbox-overflow test separate and confirm the actual path and format
from runtime evidence.

The upstream vLLM prefill and optional boundary-layer policies may differ from
BeeLlama's serving path. Matching the codec definition alone does not establish
identical model-level quality or performance.
