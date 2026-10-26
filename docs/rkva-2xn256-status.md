# RKVA 2xN256 development status

The Xbox app lives in `tools/xbox/` and builds against this repository's core.
The UWP build does not need a second xllama/llama checkout. Host RKVA and the
console's listener remain separate processes, built from the same repository.

## Reused implementation

A content audit of 8,647 refs found two relevant implementations:

- `74b3cdc7e`: two original N256 ubatches, CPU schedulers, Qwen35 with 32 layers.
  Its GPU/RKVA memory lifecycle is not implemented. It remains on its original
  branch; an attempted import and its diagnostics were preserved outside the
  checkout with the experimental CPU integration disabled.
- `e8842f1a9`: serving wavefront CUDA/RKVA, originally restricted to tiles 64/128.
  Its existing partitioner already accepts 256. This work reuses its planner,
  recurrent-state dependencies, async transport and two-slot CUDA lease.

The new experimental path uses logical batch 512 and requested parent ubatch 512,
partitioned into exactly two complete N256 tiles. The serial control uses
logical batch 512 and physical ubatch 256. This is not a claim that the public
`n_ubatch` setting is identical between arms.

Only a contiguous, single-sequence, fully overflow 512-token parent can activate.
Prefix, boundary crossings, small tails and ineligible batches keep physical
ubatch 256. Every final FFN and final normalization also runs per N256 tile before
concatenation. Recurrent rollback slots, SWA, draft contexts, captures and masked
nextn outputs remain outside this experiment. RPC batching is 1; cohort4 is
unavailable and cannot reuse the Q4/Q8 capability bits 14/15.

## Qualification

`tests/test-rkva-wavefront-2x256-live.cpp` is a manual executable, never an
implicit CTest hardware operation. With `--execute` it loads one CUDA model,
runs serial and wavefront contexts sequentially, and compares complete raw F32
logits and finite F32 convolution/GDN state from the recurrent owner with no
tolerance. Public PARTIAL snapshots can include KV metadata; those are retained
only as diagnostics. It captures P,
P+512, P+1024 and one identical follow-up input token. Activation must also be
verified in the arm-scoped runtime logs; matching outputs alone can be a serial
fallback.

Default P=512/C=2048 is a development smoke test. It does not qualify the
103k–131k regime. The input used here is the existing development benchmark token
stream, not the historical V2 holdout.

`tools/xbox/scripts/bench-rkva-range.py` previews by default. Its WF256 campaign
requires `--batch-size 512 --ubatch 512 --wavefront-tile 256`; compare against
`--batch-size 512 --ubatch 256 --wavefront-tile 0`. It requires both the prefix
N256 marker and overflow parent512/tile256/final-FFN256 marker before qualifying
TPS. CPU prototype or WF128 markers cannot qualify WF256.

## Real failure found during qualification

The serial control passed its local prefix, then crashed on the first remote
segment. The core dump proved an op-parameter collision introduced by the
upstream integration: KVarN domain slot 9 overwrote the high 32 bits of the LSE
pointer stored in slots8/9. CUDA's force-vector hint also overlapped LSE flag 7.
The corrected layout reserves domain 12 and force-vector 13; sparse 4, tail 5/6,
LSE 7/8/9 and window/non-causal 10/11 retain separate storage.

The CPU regression performs 1,112 checks and reproduces pointer corruption under
the old layout. The full CUDA rebuild passed. Seven targeted CTests passed, including the
parameter-layout regression, CUDA LSE and the two-slot transport. A separate
merge omission in Vulkan CM1 was also corrected: its LSE partial-output guard
now matches the host reduction contract. Three previously failing real-pair
cases pass with unchanged oracles.

The Xbox package 291 was built, signed, installed in place and passed chat and
twelve F16 RKVA checks, with LocalState preserved. After console availability
was restored, the installed package was started through Device Portal; its HTTP
health and RKVA listener responded.

The real 27B development smoke at P=512/C=2048 passed with FP8 strict wire,
native KVarN4, tail128 and async overlap. All 248,320 finite F32 logits and the
156,894,356-byte recurrent serialization were byte-identical at prefix, two
consecutive overflow parents and the identical follow-up token. ON-arm logs
confirmed physical prefix ubatch256, two tiles256, parent512, final FFN256 and
two tiled parents. Evidence: `wf2x256-dev-p512-r3/manifest.json` and its companion
log under the session artifact directory. This smoke does not qualify long
context or claim a performance gain.

Real parity at P=102400/C=131072 failed before WF activation in the first
long-context runs. Correcting the unnecessary initial serial-N512 reservation
restored the 1024-token diagnostic, but did not fix the complete prefix.
The revised automatic reservation uses physical256 while keeping parent512
node capacity and conservative no-allocation estimates. Matching serial A/A
and serial/WF prefix-only diagnostics pass through8192 tokens.

The full r2 trace contains68897 events with zero drops. At the first route
mismatch (ubatch157, end40448, attention layer3), both arms requested window40448
and2097152 bytes of pool growth. OFF had free143654912/budget7340032 bytes;
ON had free137363456/budget1048576 bytes and selected40192 instead. This changes
single-window execution to multiple windows plus FP32 merging. All16 attention
layers show this first mismatch at the same position. The source of the6MiB
budget difference is not yet attributed; it is not proof of a capture leak.

The configured production service uses `GGML_KVARN_WINDOW_CHUNK=4096`, while the
harness previously sanitized that variable and used the automatic65536 default.
The harness now pins4096 in both arms, records it, and has a regression covering
inherited/unset environments. Requested4096 remains a ceiling: effective route
choices must also match. The third full qualification passed without F4 callbacks: all logits and pure
recurrent state were byte-identical at P102400, two512-token overflow parents
and one fixed follow-up. Both arms used effective4096 windows. This limited
fixture does not qualify the complete28160-token overflow continuation.

The live fixture creates fresh contexts and performs no tier migration. The
production preset's XXL/XXXL tiers use B/UB256; XL uses128. The causal experiment
uses logicalB512 with physicalN256 in both arms. Do not enable WF256 globally
for existing tiers: its constructor requires requestedUB512/B>=512. Pure-to-mixed
state import and actual XXL-to-XXXL migration need their own qualification;
these fresh-context checks do not establish that migration is byte-exact.

Package293 built successfully on the Windows host with parallelism10 in583.35s,
was verified against its SHA256 and existing certificate, and was upgraded in
place. All four CRT DLLs are x64 and byte-identical to292. The five LocalState
names were preserved; health returned ok and the installed local model replied
`OK.` through the chat API. Native F32 bulk16 checks and FP8 codec10 checks pass.
The earlier F16 bulk exact-LSE oracle failure is unchanged and is not reclassified
by these FP8 development results.

## Late continuation investigation

The server pair at point130560 generated256 outputs with no truncation. Runtime
confirmed FP8 E5M2 bulk transport and async overlap in both arms,5856 FP8 GPU
calls each and no CPU fallback. Serial prefill measured121.15035757919541t/s
versus149.2111180107712t/s for WF256; decode measured17.35 versus17.24t/s.
These timings are **not quality-qualified**: the first206 generated IDs matched,
then output207 (zero-based206) differed.

Review found an additional server cadence mismatch. The experimental control
B512/UB256 split its final512 as128/256/128; WF B512/UB512 used256/128/128.
Actual production B256/UB256 uses256/128/128. A checkpoint-only helper now uses
physical compute size for KVarN checkpoint offsets and near-end decisions,
preserving production cadence and leaving ordinary alignment1 behavior intact.
A new public getter reports nominal physical ubatch without changing the logical
ubatch API. Eight targeted CTests pass, including canonical boundary regressions
at102400 and130560. The corrected frozen-prefix continuation now passes: all248320 logits and
156894356 recurrent bytes match exactly at the end of28160 overflow tokens
and at fixed follow-up steps1/128/206/256. Both restored recurrent states match
and all54 eligible parents actually use2xN256. The trace contains1017864 events
with zero drops; all320 observed window choices are4096. CUDA graph replay means
these observations are not one record per executed tile.

The benchmark can now save a local prefix with `--prefix-snapshot PATH` and
reuse it in a fresh child. `--prefix-only` creates or validates that cache without
claiming performance. Hashes, token IDs, model, cache geometry and relevant
settings are checked before reuse. Source/build provenance is retained; cross-build
reuse represents a frozen experimental input, not a new cold-prefix proof.
The real102400-token snapshot saved2230564412 bytes in2896.117ms and restored
in3857.374ms with zero reprocessed prompt tokens. It resides outside Git under
`/home/hjotha/llama-slot-cache/rkva-2xn256-dev/a7e8962f16a8/`.
Native remote-overflow snapshots remain unsupported, so this cache is taken
before the first remote token.

The extended manual fixture restores that same prefix, uses production final
cuts256/128/128, and feeds the serial run's fixed256 follow-up IDs. Selected raw
logit/GDN checkpoints isolate prefill and late decode differences without sampling
feedback. Its first attempt failed before overflow: server slot files contain
packed prompt words (102404 for102400 IDs), while the manual adapter assumed
raw IDs. The adapter now queries the packed size, strictly decodes the server v1
text-only envelope, and validates the actual IDs and KV position. Raw legacy
CAPI token lists remain supported; malformed marked envelopes cannot fall back
to raw IDs. The target rebuild and malformed-input parser checks pass.

The new paired server run from the same frozen prefix passes all256 sampled
IDs at temperature1/seed1234. No truncation, runtime error or CPU FP8 fallback
was detected. Each arm records5856 GPU FP8 bulk calls with async overlap.
Serial prefill is119.679580702589t/s versus149.42178095300713t/s for WF256
(+24.851524442025983%). Decode is17.32683375487368 versus17.23988611939696t/s
(-0.5018091401278735%); this run establishes no decode gain. Both arms restored
102400 tokens with zero prefix replay. This is one development pair, not a
cold-prefix or tier-migration qualification.

The corrected serial output also matches all256 IDs of the previous WF run;
it differs from the previous serial at zero-based206. This supports the checkpoint
cadence diagnosis. Canonical frozen prefix and measured-end cuts changed together,
so their individual contributions were not isolated.

Evidence under the session artifact directory:
`wf2x256-cached-prod-seam-late-r2/manifest.json`,
`wf2x256-cached-prod-seam-late-r2-astra-verification.json`,
`checkpoint-cadence-generation-evidence.json`, and
`wf256-tps-comparison-r4-cached.json`.

The legacy F16 bulk exact-LSE gate initially failed
(3.814697265625e-06 versus exact0); see the verified correction below.
WF256 remains opt-in experimental; actual tier migration is not qualified. The older WF128 V2 failure and REMOTE_ATTN graph-trim
NO-GO remain unchanged.


## 2026-10-26: F16 correction and further optimization

The F16 failure came from DXC reassociating the query WHT's two normalizations,
removing a CPU-visible intermediate rounding. Only `bulk_unpack_q_f16` now uses
`-Gis`; its generated header was rebuilt with the existing DXC toolchain. FP8
unpack and both output pack headers remain byte-identical. The offline
`rkva_bulk_f16_wht_verify.py` checker accepts the new DXIL and rejects the old
one. The numerical oracle was not modified.

Package294 built with parallelism10 in77.53s and installed in place with the same
certificate and four identical x64 CRT DLLs. On the console the original F16
probe passes8/8 cases with exact-zero LSE differences, native F32 bulk passes
16 checks and FP8 codec passes10 checks. Eight targeted host CTests pass5.98s;
the benchmark harness has38 hardware-free tests.

A fresh WF128 campaign uses B512/parentUB256 with two128 tiles, the same local
frozen prefix and canonical checkpoint cadence. Runtime activation was confirmed;
all256 sampled IDs match the corrected serial. Prefill measured
146.64873351242636t/s and decode17.30342125959407t/s. Previous WF256 values were
149.42178095300713 and17.23988611939696t/s. This single comparison offers no
material decode gain and favors WF256 prefill; it does not supersede the older
WF128 V2 holdout rejection on a different model/corpus.

The harness separately accepts `--cuda-fp8 mapped`, preserving the original
`prefill` default. Mapped small-query helpers already existed in the core; they
are a new measurement candidate, not a newly imported implementation. These
modes differ only for remote nq<=9, so the local prefix snapshot is reusable;
the actual codec mode is retained in the run's environment. An explicit CAPI
A/B on the same frozen prefix,1024 overflow tokens and16 fixed continuation
inputs passes complete logits/GDN comparisons at four selected checkpoints.
Late-context mapped measurements now pass all256 sampled IDs: prefill
149.57967473984715t/s, decode17.476955434860233t/s. All4080 small-query
pack/unpack calls used CUDA. Decode is1.375% above the earlier WF256 control
in this single comparison; repetition is needed to establish a small gain.

A read-only content audit of the old193 worktrees confirmed that FP8 bulk,
async transport, writev/TCP_QUICKACK/pooling, D2D/SIMD and fast KVarN decode are
already integrated. No old missing runtime optimization was established. A new
MQ WHT ping-pong candidate with fewer barriers is prepared outside the checkout;
its CPU/DXIL checks do not yet qualify it on the console.

Evidence: `/home/hjotha/.codex/artifacts/beellama-xbox-rkva-20261026/`, including
`wf128-comparison-r1.json`, `mapped-codec-parity-r1/manifest.json`,
`new294-native-bulk-f16.json`, `new294-native-bulk-f32.json`,
`new294-fp8-pack.json`, `package294-verification.json`, and
`final-phase1-ctest.log`. Historical frozen-prefix limitations above still apply.


## MQ ping-pong validation and next wavefront experiment

The new MQ WHT uses the existing two halves of LDS scratch and reduces its
synchronizations from17 to9. MQ4's additional query vectors are dead before
this scratch is reused. The source arithmetic and ordered compiled floating
operations/flags/constants are unchanged; strict compilation remains specific
to F16 query unpack. DXIL static barriers change80→48 forMQ2 and162→98 forMQ4.

Package295 built and installed in place. All696 upload/query/reply records from
both MQ modes and both tail policies match package294 exactly. The unchanged
serial-prefix oracle passes through remote capacity58368, including its edge,
with reference/MQ4/MQ2. F16 passes8 cases with LSE exactly equal; FP8 codec
passes10 cases. Four complete-vocabulary and pure recurrent checkpoints per
codec arm give16 files byte-identical to the prior model fixture.

The long WF256/mapped/fine run passes all256 generated IDs with no runtime
errors or CPU FP8 fallback: prefill150.1545776323961t/s and
 decode17.482578610414723t/s. Against package294's149.57967473984715 and
17.476955434860233, these are+0.384% and+0.032%. This is no established TPS gain.
The same4112 late-context MQ2 GPU samples average2785.531us before and2786.489us
afterward, essentially unchanged. The synchronization reduction is numerically
qualified and shows no measured regression; do not present static barrier
counts as demonstrated throughput gains.

The balanced kernel experiment is not the selected profile:98.581502153633t/s
prefill and17.574478053508702t/s decode, with a sampled-ID difference at206.
It loses34.09% prefill for a0.56% decode timing increase. The harness retains
it as an explicit diagnostic variant; fine remains its default.

Evidence: `new295-mq-replay.json`, `new295-serial-prefix-full.json`,
`new295-model-raw-comparison.json`, `mq295-comparison-r1.json`,
`mq295-gpu-bucket-comparison.json`, `mapped-comparison-r1.json`, and
`balanced-comparison-r1.json` in the20261026 artifact directory. The new
`rkva_kvarn_bulk_replay.py` probe records or compares package replies only with
explicit `--execute`; its original bulk/serial numerical oracle is unchanged.

For a larger wavefront, first distinguish parent512 (the existing2xN256) from
compute tile512. A tile512 campaign would change GEMM/FFN shapes and workspace;
the current native bulk path also rejects query ranges above256. It needs
transport segmentation and a separate numerical qualification.

A parent1024 with4xN256 is the smaller next experiment. RetainN256 FFN/norm,
physical prefix/tailN256, production checkpoint cadence, RPCbatch1 and at most
two outstanding requests. The current scheduler starts tile2 only after tile0's
last group: four tiles mostly behave as two pairs, with a possible boundary
and executor overhead saving, not four-way concurrency. Increasing in-flight
requests or using cohort4 is a separate transport/lifetime change.

Do not merely increase B: the present coalescing probe/exact-parent guard can
make the2tile control fall back when given1024 tokens. A causal comparison
needs explicit512/1024 parent selection with the same logical batch, expanded
plan/mask guards, runtime4tile markers, failure draining and ordered conv/GDN
commits. Reuse the frozen prefix, compare long-context logits/GDN and256 IDs,
check effective window/memory behavior, then repeat trace-free timing pairs.
The current harness's B512 snapshot contract must not simply be removed.

None of these larger-tile paths provides parallel autoregressiveN1 decode.
Use the measured codec/kernel/latency path for decode optimization. Actual
production tier migration and the older WF128 holdout remain separate gates.
