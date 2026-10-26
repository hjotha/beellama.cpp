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


## Controlled four-tile and TCP prefetch experiments, 2026-10-26

The implementation now accepts requested UB512/two or UB1024/four N256 tiles,
with logical B at least the requested UB. Eligibility is checked per internal
parent, so B1024/UB512 remains a real two-tile control. Unsupported parents,
local prefix, crossings and short tails are refined to physical N256 before
attention/recurrent preparation. Final FFN/norm remain N256; RPCbatch1 and the
two-outstanding limit are unchanged. Four tiles do not mean four GPU requests
running concurrently. The subset helper also preserves original batch indices.

On the same frozen P102400/C131072 prefix, B1024 in both arms, serving overflow
28160 tokens and256 fixed continuation inputs, two versus four tiles pass five
complete-vocabulary logits and serialized recurrent checkpoints bit-exactly.
Restored prefix states match, both runtime geometries activate, all576 observed
window choices remain4096 and the trace loses zero events. CUDA compute buffers
are262.5060 versus269.1144MiB; host buffers141.2618 versus281.6330MiB.
A separate trace-free timing pair preserves all256 generated IDs:

| Parent geometry | Overflow prefill t/s | Decode t/s |
|---|---:|---:|
| B1024/UB512, two N256 | 151.179863542242 | 17.492394581227 |
| B1024/UB1024, four N256 | 151.336101078586 | 17.419140520480 |

The +0.103% prefill difference is not an established improvement. Two N256 tiles
remain the performance reference. The opt-in four-tile implementation is useful
for controlled experiments, not a demonstrated throughput upgrade.

Astra's producer change emits each exact F16 tail block with one vector resize
rather than a two-byte insert per scalar. Conversion remains the same scalar
F32-to-F16 RNE operation. Seventy-two complete payloads/17686528bytes match,
including reset/rewind and boundary cases. Both timing arms above include this
change, so the table does not independently attribute a producer speedup.

### Bounded host TCP prefetch

`GGML_RKVA_TCP_PREFETCH=1` enables an experimental POSIX host transport window.
Default is OFF. Only FP8 native op22, async capacity2 and nq>9 use it. Decode N1,
verification nq<=9, other codecs and cohorts keep their existing path. There is
no Xbox packet/shader change; the installed package295 serves these tests.

The existing I/O worker polls one socket and a wake pipe. It can send the next
prepared frame while the previous reply is outstanding. At most two existing
slots own all buffers through send, receive, validation, unpack and ordered ACK.
Replies remain FIFO and validate magic/version/op/cycle/length before decode.
Controls drain; cancellation/EOF/malformed replies poison the stream, interrupt
I/O and retain owners until retirement. Partial-TX HUP fails immediately; after
TX completion recv can drain a final buffered reply. No autoregressive future
query is fabricated. Frame RTT intervals overlap and must not be summed as
elapsed wall time.

The causal live test holds B1024/UB512/two N256 and mapped/fine fixed, switching
only prefetch OFF/ON. Five whole-logit/recurrent checkpoints match bit-exactly
through130816 tokens, including identical restored prefix state. ON completes
1776/1776 frames with1626 transmitted ahead and maximum window2. Both arms use
5872 GPU FP8 calls and zero CPU fallback. Identical request/response byte counts
accompany Xbox payload-receive blocking44.372885s ->18.692473s; handler time is
94.769826s ->94.785190s. The reduction hides network waiting; it does not imply
that the physical link became faster.

A trace-free A/B/B/A campaign on the final host binary/source has these results:

| Prefetch | Repetition | Overflow prefill t/s | Decode t/s |
|---|---:|---:|---:|
| OFF | 1 | 150.114506823389 | 17.481713270972 |
| ON | 1 | 203.151580794520 | 17.438669737240 |
| ON | 2 | 204.658022404065 | 17.489472023703 |
| OFF | 2 | 149.572404312077 | 17.533690470542 |

Pooled prefill is149.842965264150 ->203.902019214688t/s (+36.077%). Pooled decode
is17.507663293031 ->17.464033935015t/s (-0.249%); no material decode improvement
is established. Every run has256 identical generated IDs, also equal to the
previous reference, with no errors or memory-policy warnings. Both ON runs
complete1776 frames,1627 ahead, maximum2. The timed API reports256 outputs but
255 timed decode intervals; pooling retains that convention.

The benchmark uses explicit `--tcp-prefetch`, pins OFF otherwise, and requires
final counters proving completed bounded overlap. An earlier successful counter
record cannot qualify an incomplete final record. Errors during teardown also
invalidate the run. CAPI per-arm `--off-tcp-prefetch 0 --on-tcp-prefetch 1`
requires `--control-only-pair --parent-tokens 1024` and identical codec modes.
`--frozen-prefix-batch 512` explicitly permits this existing B512 snapshot as a
frozen input for B1024 continuation; default batch metadata remains strict.
Remote-only prefetch does not alter the local-prefix codec contract. These
measurements qualify the102400->130560 continuation, not a new cold prefix,
production tier migration, every corpus, or the historical WF128 holdout.

### Remaining bottlenecks and validation

The current mapped trace attributes roughly39.24ms/decode token to local CUDA,
including13.45ms in FA-labelled subgraphs and25.79ms elsewhere. Exposed remote
wait is6.93ms/token. These intervals overlap. WAIT producer synchronization
mostly waits for the same local FA work already counted; removing that wait
would not remove an additional12.87ms. Memory guards cost about3.09ms/token but
were preserved. Measuring kernels inside FA and matvec/GDN blocks is the next
justified decode investigation. This is not an architectural TPS ceiling.

Late Xbox prefill measurements put pass1 at65.30ms of67.55ms GPU time per
envelope. Those late-ring samples are not a full-run average. The console is
confirmed Series X; no conclusion about available CU occupancy follows from
its memory budget. Network overlap, host CUDA and remote attention all matter.

Linux build uses-j10. Nine selected CTests pass in7.24s, including the maintained
TCP concurrency/failure suite;45 benchmark unit tests pass. MSVC AppContainer
builds the entire GGML/core/model library with parallelism10:668.13s initially,
16.60s for the final TCP delta. The optional standalone Windows desktop CTest
setup remains unqualified: shared-DLL link and NOMINMAX/EHsc configuration
problems are recorded; it was stopped without source changes. This does not
constitute a desktop Windows test pass. UWP compilation and Linux live runtime
validation are distinct gates; no new console package was installed for the
host-only transport change.

On193, cleanup reclaimed33795993600bytes (33.80GB/31.47GiB), removing67 old build
directories and one clean linked worktree. Its unique branch and verified
bundle/source backup remain. Fourteen source trees with unique/dirty/submodule
state were preserved, along with183 model/key/certificate paths and five
junctions. The current Bee checkout, build and shared caches were protected.

Evidence in the20261026 artifact directory: `wf4-controlled-comparison-r1.json`,
`wf4-cached-p102400-long-r1/manifest.json`,
`wf4-long-runtime-verification.json`,
`tcp-prefetch-cached-p102400-long-r1/manifest.json`,
`tcp-prefetch-causal-transport-profile.json`, `tcp-prefetch-abba-comparison.json`,
`tcp-host-ctest-r2.log`, `windows-4xn256-final-validation.json`,
`decode-mapped-current-trace-analysis.txt` and `cleanup193-summary.json`.

### Exact local split-input reuse (experimental)

`GGML_RKVA_LOCAL_INPUT_REUSE=1` enables reuse of the local I64 prefix indices
and F16 attention masks in position-split RKVA. Default is OFF. The benchmark
requires explicit `--local-input-reuse` and proves activation using final
copied/skipped-byte counters. The frozen prefix, FP8 mapped bulk async, fine
range kernel, TCP prefetch, CUDA graph policy and numerical operations remain
the same in the comparisons below.

Inputs carry explicit cache intent. Only contiguous CPU-host INPUT roots and
explicitly marked views are eligible; computed roots and host weights are
excluded. The scheduler retains the destination against allocator reuse and
compares the complete final source bytes after the input writers finish. A hit
skips the copy while preserving ownership fences. A miss completes the copy
before taking a private snapshot. Snapshots are bounded to256MiB/128entries
and invalidated on planning/allocation/reset/failure and ordinary fallback
writes. The API requires inputs and aliases to remain immutable during compute.

Four-tile prefill masks retain ordinary copies: pinning all four would defeat
workspace aliasing and can cross the128MiB graph guard on a12GiB GPU. Their I64
indices and N1 masks remain eligible. Serial and two-tile masks are eligible.
This fallback was exercised by the live two-versus-four-tile comparison.

Validation passes10/10 Linux CTests, including the new fake-device scheduler
regressions, and47 benchmark unit tests. Regression coverage includes allocator
clobbering, changed bytes, views, graph/arena reuse, copy slots, allocation and
compute failure, unmark/normal-copy/re-mark, and stateful host-weight callbacks.
MSVC AppContainer compiles the complete GGML/core/model library at parallelism10:
374.75s for the full corrected candidate and13.06s for the final backend delta;
all ten source hashes match Linux. No console package change is needed.

Live short comparisons pass for serial N128, serial N256, two N256 OFF/ON and
two-versus-four N256 with reuse ON. Selected complete logits and pure recurrent
state are byte-identical. The long causal OFF/ON run restores the same102400
prefix and continues28160 overflow tokens plus256 teacher inputs. All five
248320-logit checkpoints and full serialized convolution/GDN states match
bit-exactly, including the restored prefix state. The ON arm records669hits,
11misses and5861990400skipped bytes. Both arms retain262.5060MiB CUDA compute
allocation; all416 observed window choices are4096, all71090 graph guards
pass, and trace drops/pressure states are zero. The explicit guard is128MiB.

A trace-free same-binary OFF/ON/ON/OFF campaign, with TCP prefetch ON throughout:

| Local reuse | Repetition | Overflow prefill t/s | Decode t/s |
|---|---:|---:|---:|
| OFF | 1 | 202.295097 | 17.376769 |
| ON | 1 | 204.740716 | 17.178116 |
| ON | 2 | 207.042458 | 17.333898 |
| OFF | 2 | 203.179099 | 17.391206 |

Pooled prefill is202.736134 ->205.885154t/s (+1.553%). Pooled decode is
17.383984 ->17.255656t/s (-0.738%); both ON measurements are below both OFF
measurements. There is no demonstrated decode benefit or general latency win.
These are descriptive results from two repetitions per condition, without a
confidence bound. The cause of the decode decrease has not been established.
All four runs generate the same256 IDs as the previous reference, with zero
errors or memory-policy warnings. Source, binary and library hashes are fixed.
Decode pooling uses255 timed intervals for the API's256 generated outputs.

CUPTI confirms the intended H2D removal. Prefill copies of52428800bytes fall
from109 to3 and819200-byte copies from57 to4; the two26214400-byte seam copies
remain. Decode819200-byte and204800-byte copies each fall from255 to1. Their
total reduction is5860966400bytes, exactly matching the benchmark cache
counter. No other H2D/D2H/D2D size-count changes occur within those stage
windows. Tensor attribution uses exact dimensions and source inspection;
CUPTI does not name tensors. The historical main064 baseline is used only for
copy traffic; TPS comes from the same-new-binary ABBA. Both trace and CUPTI
report zero drops; generated IDs match and windows/graph policy are preserved.

The qualified scope is frozen-prefix continuation in this development profile,
not cold prefix, production tier migration, all tiers/corpora or the historical
WF128 holdout. Keep the feature experimental and OFF by default. Prefill-only
reuse is a separate future A/B; it has not been implemented or qualified here.

Evidence under `20261026/local-input-reuse/`: `short-matrix-summary.json`,
`long-runtime-verification.json`, `abba-comparison.json`,
`cupti-copy-comparison.json`, `host-ctest-r1.log` and
`windows/final-validation.json`. The long raw manifest is
`20261026/reuse-cached-p102400-long-r1/manifest.json`.

### Lossless binary-mask transport for local and RKVA prefill

`GGML_CUDA_MASK_PACK` selects a lossless CUDA transport for explicitly marked,
contiguous F16 prefill masks. Unset defaults to ON; `1` enables it and `0`
(or any other value) retains the classic copy. This is separate from
`GGML_RKVA_LOCAL_INPUT_REUSE`, which remains experimental and OFF by default.

The CPU validates every raw half word before packing +0/-infinity into one
bit. A CUDA integer-only kernel reads mapped pinned host staging and restores
the exact half bits in the existing device allocation. No mask is retained
across graph plans and allocator lifetimes are unchanged. The payload is
1/16 of F16, with last-word padding. Staging is bounded to8MiB per CUDA buffer;
eligible masks are128KiB through128MiB, with more than9 query rows. Nonbinary
values, unsupported placement/shape, partial writes, odd byte alignment and
unavailable mapped staging take the classic path before any device enqueue.
Unexpected CUDA errors after enqueue remain fatal rather than permitting a
concurrent fallback writer. HIP/MUSA and Xbox shaders are unchanged.

This specifically covers the normal Qwen35 local-tier masks as well as the
marked position-split and wavefront masks. It is not a claim that every
architecture or mask builder is covered. Decode shapes with at most9 queries
are excluded. The first mapped-kernel use measured2MiB of driver/device
footprint; no device-data scratch is allocated. Performance comparisons must
include CPU packing, staging initialization, mapped reads and device unpack.

Validation: Linux13/13 selected CTests,48 benchmark unit tests; SSE2, scalar
and sanitizer codec checks; real CUDA raw-bit round trips including50MiB
RKVA masks and128MiB maximum raw masks. Local target-only Qwen35 quality
checks compare complete248320-word logits and pure convolution/GDN state at
four checkpoints. MSVC AppContainer/UWP core build passes at parallelism10;
this host-only change requires no new console package.

All TPS comparisons use fresh OFF/default/default/OFF server processes, one
GPU job at a time, the same binary within each ABBA, and an immutable saved prefix. DEFAULT is
the actual unset environment value. The timers include CPU validation/packing,
first staging allocation, mapped transfer and integer unpack; startup warmup is
disabled. Rates pool tokens/time across two repetitions, with 255 timed decode
intervals for 256 generated outputs. This is one descriptive ABBA per profile,
without confidence bounds. Every arm must retain identical cached/prefill work,
all 256 greedy output IDs, normal process exit, zero unavailable staging and
positive DEFAULT/zero OFF packing counters.

| Profile | Prefill OFF -> default (t/s) | Change | Decode OFF -> default (t/s) | Change |
|---|---:|---:|---:|---:|
| S, C24576, B/UB256, MTP6 | 564.5481 -> 574.9609 | +1.8445% | 37.8112 -> 37.8865 | +0.1992% |
| M, C38912, B/UB256, MTP4 | 498.8604 -> 509.0656 | +2.0457% | 43.9089 -> 44.3143 | +0.9234% |
| L, C56320, B/UB256, MTP2 | 430.1231 -> 442.6921 | +2.9222% | 34.5508 -> 34.7385 | +0.5431% |
| XL, C72704, B/UB128, MTP2 | 350.5574 -> 370.3801 | +5.6546% | 33.8180 -> 34.2023 | +1.1363% |
| Fixed XXL, C102400, B/UB256, no MTP | 347.7347 -> 354.1730 | +1.8515% | 24.8004 -> 24.6599 | -0.5666% |
| XXXL RKVA, C131072, B1024/UB512, 2xN256 | 215.8965 -> 217.7212 | +0.8452% | 17.9837 -> 17.9369 | -0.2603% |

S/M/L/XL use the actual production tier geometry and MTP depths. The target and
draft KV remain KVarN4, Flash Attention is on, GPU layers99 and parallelism1.
The local tests retain production repeat1.1/frequency0.3 penalties, but use
greedy sampling and private children: cache RAM16384MiB rather than4096MiB,
CUDA graph guard128MiB rather than the live service's18MiB, window4096, verbosity4,
no automatic save and no remote attention. Local prompts end at C-2048 and the
measured suffix contains4081 tokens after the warm16 saved prefix. No production
service/configuration was restarted or rewritten. XL's two OFF prefill rates
were340.2851 and361.4691t/s, versus370.5215 and370.2388 default; its +5.65% is
especially sensitive to control variation and is not a guaranteed gain.
Decode changes are small and mixed across profiles; no consistent decode
benefit is demonstrated, and the cause of the fixed XXL decrease is not
established.

Fixed XXL measures the same main64 target layers without MTP, but its attention
tail has rollback reserve1 versus adaptive XXL's6 (320KiB less storage; both
retain exact F16 tail128 and execution rows round to256). Thus it qualifies
fixed target rendering, not exact production cache layout, adaptive admission
or migration. It uses the existing bounded native routing save/restore path,
which serializes full target KV and recurrent state with flags0. The immutable
1922513548-byte snapshot contains96256 evaluated tokens; all arms restore it and
process4096 suffix tokens. Native checksum/model/layout/token/position validation
and real save->erase->restore readback pass. The adaptive five-copy RAM guard
is unchanged. Exact initial-load warnings for15 unused block64 MTP tensors are
audited separately; changed names/sizes/layers, duplicates and late occurrences
fail the external rig. They do not mask target-layer or inference failures.

RKVA uses the real Series X at192.168.1.200:50054 with mapped FP8 bulk async,
fine range kernels, two N256 tiles and TCP prefetch enabled. It is the development
B1024/UB512 profile, not the production XXXL B/UB256 geometry. All four arms
restore the frozen102400-token local prefix, warm with no prompt replay, then
process28160 overflow tokens to130560 and generate256 outputs. Repeat penalty
is1.0 in this existing helper, unlike the local MTP campaigns. DEFAULT packs111
masks per arm,5767168000 raw bytes ->360448000 payload bytes; OFF packs0. Runtime
proof confirms GPU FP8 bulk with zero CPU fallback, two-tile activation and
completed two-frame TCP overlap. Window4096 and graph guard128MiB are configured;
these uninstrumented timing logs do not directly record each selected N256
staging window. The original campaign was interrupted by orchestration during
arm2 and is preserved separately; only the complete restarted ABBA is pooled.

Model: `Huihui-Qwen3.8-27B-abliterated-Swift-1.5-GSQ-RCO-IQ3_XXS-mtp.gguf`,
SHA256 `a7e8962f16a84945b2277663da41824b936d40ae90d55852718ac389431c9166`.
Raw IDs come from `/home/hjotha/beellama-gate/20261017/tokens-prompt120k-t2.json`;
the RKVA helper repeats that stream to reach130560. Local MTP requests retain
the15 actually evaluated warm-generated prefix IDs before the original suffix;
the complete request vectors and hashes are preserved in each manifest. Host is the RTX4070 eGPU
machine; Xbox is Series X. Measurements bind base
`40f22aa6b47ce4eef22402fa180f8417a743d6a3` plus these uncommitted transport/server
changes using source diff, executable and every shared-library SHA in each
manifest. They are not measurements of bare40f. The CUDA/core/llama libraries
are unchanged across local campaigns; S/XL/XXL/RKVA use the final server fixes,
while M/L retain their recorded pre-save-hook server build provenance. The
optional explicit-save hook is outside those timed paths.

The production-profile validation also exposed three existing server defects.
A complete native MTP slot restore lost its ready carry before the first new
suffix batch; exact full-prefix continuations now retain it only when the
managed MTP implementation proves matching target/draft/carry positions.
Target-only bootstrap and other existing fallback paths are unchanged.
Linux slot budgeting now uses strictly parsed MemAvailable, with the existing
25% reserve, configured cap and five-copy allowance preserved; invalid/missing
measurements retain the old fallback. Finally, every recreated adaptive
context reinitializes the context-dependent batch descriptor, including
equal/smaller batches and successful rollback. Previously S->M with B256
retained a pointer into freed S memory and crashed on its first decode.

A real disk regression uses three sequential server lifetimes: save, restore A,
restore B. Both continuations reuse exactly1039 cached tokens, process the
same512-token suffix, engage MTP and emit identical32-token outputs. Native
snapshot/logits hashes remain unchanged and all children exit normally.
The budget parser/cap test passes36 checks. A separate low-threshold live
cycle covers equal/shrinking/growing batch sizes and actual decoding after an
injected transition failure successfully rolls back. Its small contexts and
XXL B512 are test-only and are not performance measurements.

Cold full-prefill is not a bit-exact oracle for the MTP disk regression: it
rebatches autoregressively generated prefix tokens into prefill kernels. The
initial attempted comparison differed at index24 (25th output); the artifact is retained.
Cadence is a plausible explanation, not a proven sole cause. The regression
qualifies reproducible restored continuation, not cold-prefill equivalence.

Likewise, a two-arm long CAPI test shares one CUDA process. Both OFF/default
and default/default reproduced exactly42MiB less free VRAM in the second arm
and a32768->30976 N256 window choice. This42MiB/window discrepancy also occurs without changing the toggle, so it
does not isolate a mask-pack effect; its earlier attribution to pinning was
not established. The
causal comparison uses first arms from separate fresh processes, includes
actual initialization cost and leaves window selection natural. All five
whole-vocabulary checkpoints, five complete recurrent snapshots and restored
prefix match bit-for-bit. Both internally mismatching pair manifests remain
FAIL evidence; they are not relabeled successful.


`LLAMA_ADAPTIVE_SLOT_SAVE_CHECKPOINTS=0` is an explicit-save-only option for
omitting auxiliary rollback snapshots. Default/unset retains them. Full target,
draft and speculative carry are captured before that optional section; runtime
`ctx-checkpoints=1`, rollback capture and all file/RAM guards remain unchanged.
XL uses this option to fit the native full snapshot on the current host. The
three-process disk regression also passes with this option: native auxiliary
count0, positive full target/draft/carry payloads, identical restored outputs and
unchanged snapshot/logits hashes. Fresh or erased readback avoids the separate
RAM expense of capturing an already dirty destination for rollback; rollback
safety itself is not weakened.

Successful adaptive restores can discard old auxiliary history after an epoch
mismatch, then reimport full target/draft/carry and publish complete_mtp=1,
bootstrap=0. External audits allow only the exact native-header/log-matched
sequence within a successful restore. A few canceled non-inference request IDs
have unknown origins: the rig bounds them to64 unique IDs inside an HTTP/log
verified idle native interval, before successful publication, and excludes every
ID launched for inference anywhere in the complete log. All other diagnostic
occurrences remain fatal. These are benchmark audit exceptions, not runtime
suppression or a claim to have fixed their origin.

Evidence under `20261026/local-input-reuse-default/mask-pack/`:
`prod-tier-S-abba-r4/`, `prod-tier-M-abba-r3/`, `prod-tier-L-abba-r2/`,
`prod-tier-XL-abba-r3/`, `prod-tier-XXL-stream-abba/`, `rkva-mask-abba-r2/`,
`tier-comparison-progress.json`, `mask-serial-cached-p102400-long-r1/fresh-process-off-vs-default-control-a.json`,
`mtp-restore-regression-small-r2/`, `mtp-restore-regression-no-aux-r1/`,
`adaptive-batch-rebind-qual-r1/`, `raw-cuda-large-r2.log` and `windows/candidate2-validation.json`.
Per-run requests, command vectors, effective INIs, environments, SHA bindings,
full native state proof, raw responses and failure manifests remain preserved.
The two long same-process FAIL manifests and the initial cold-prefill MTP
comparison remain diagnostic evidence. No new Xbox package is required for this
host CUDA/server change; the compiled UWP inputs remain byte-identical.
