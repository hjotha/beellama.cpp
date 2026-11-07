# 4-bit KV native TCP activation-wire matrix

This matrix describes implemented native TCP position-split paths, not model
quality or production qualification. Local/normal KV support is a different
matrix. None of these changes changes a KV quantizer, block layout, defaults,
SAW BDR16, NC WHT256, or the legacy TurboQuant aliases.

| KV representation | Native layout | Bytes per 256-dim head/plane | E5M2 bulk/async/prefetch | E4M3FN bulk/async/prefetch | E4 capability bit |
|---|---:|---:|---|---|---:|
| KVarN4 | 3 | Descriptor-defined groups, stage and tail | Existing negotiated path | Existing negotiated path | 25 |
| Q4_0 | 4 | 144 | Existing negotiated path | Added native-row path | 28 |
| SAW-INT4 | 6 | 136 | Existing experimental path | Added experimental native-row path | 29 |
| TurboQuant NC | 8 | 132, including K padding | Existing experimental path | Added experimental native-row path | 30 |
| Q4_1 | 9 | 160 | Added, explicit experimental gates | Added, same gates | 33 |
| IQ4_NL | 10 | 144 | Added, explicit experimental gates | Added, same gates | 34 |

“Bulk/async/prefetch” means protocol and host scheduling eligibility after all
geometry, type, device and capability checks. Prefetch is opt-in through
`GGML_RKVA_TCP_PREFETCH=1`, admits at most two frames and requires nq >= 10.
Native-row envelopes remain bounded to 256 queries and 32 segments. Async
local/remote overlap is controlled by `GGML_RKVA_ASYNC_LOCAL_REMOTE`. No row
format silently falls back to another quantizer to satisfy a requested wire.

E4 requires explicit `--remote-attn-wire-format fp8_e4m3fn`, an explicit
supported `--remote-attn-format`, TCP position split and an RKVA v3 HELLO
advertising the exact E4 capability plus the corresponding native storage and
bulk capabilities. E5 support, or the old KVarN-only E4 bit, is not enough.
Changing E4 selection after HELLO fails instead of renegotiating silently.

Q4_1/IQ4_NL require `GGML_RKVA_EXPERIMENTAL_LOW4=1` on the host and a server
compiled with `RKVA_EXPERIMENTAL_LOW4=1`; the server additionally checks PSOs,
shader availability, D256, GQA6 and Wave64. These server gates remain off by
default. Their cache-type K/V must match the selected format. See
[rkva-low4-experiment.md](rkva-low4-experiment.md) for limits and state identity.
SAW/NC and KVarN retain their existing experimental and geometry restrictions.

The five native-row E4 paths use the Xbox CPU activation codec, then the
existing format-specific GPU attention path. They do not reinterpret E4 as
E5 or use the E5-only raw GPU codec. Q and OUT are encoded directly from F32,
round-to-nearest-even with finite saturation at ±448. KVarN E4 keeps its
existing F16-intermediate OUT codec. All keep F32 LSE. Empty remote prefixes
require zero OUT and negative-infinite LSE; nonempty LSE must be finite.
The KVarN server enforces this contract for both E4 and E5. The current native
host admission rejects queries before the remote origin, so empty-prefix
coverage exercises the server through the protocol directly.

## Validation and remaining qualification

CPU validation exercises the real host producers, GGML block bytes, E4/E5
codecs, TCP transport, FIFO/control barriers, failure poisoning, descriptor
negotiation and the actual server-side pure preflight/output validators.
The five-format two-frame N128 matrix covers 150 control/failure cases, and
the separate Q4_1/IQ4_NL loopback covers 32 wire/mode/nq combinations at
nq 1/9/128/256 with append and trim. A scalar C++ shim extracts and checks
the pre-existing Q4_1/IQ4_NL HLSL loader functions against GGML dequantization.
Review identified an inactive-lane query broadcast in the pre-existing
Q4_1/IQ4_NL rowlane template and SAW rowlane shader. For partial or causal
tiles, `WaveReadLaneAt` ran inside `if (row_valid)` and could read inactive
source lanes. The source now broadcasts with every lane active and guards
only the K/V memory loads. The three affected DXIL headers were regenerated
with official Microsoft DXC 1.9.2609.5, targeting the existing `cs_6_0`.
A source/provenance regression fails on the old code and checks partial-wave
cases; it is not a GPU execution test. Other generated shader blobs are
unchanged, including the unused Q4_0 rowlane specialization and legacy Q8/F16
rowlane paths, which are outside this change.

Synthetic loopback replies are not GPU attention numerics. The Linux test
machine compiled these three DXIL shaders but did not compile Windows/UWP or
execute CUDA or D3D12 kernels.
Before hardware qualification, build the Windows/UWP sources with the intended
gates, check PSO creation, compare OUT/LSE to decoded KV references, exercise
multi-layer async/prefetch and faults on the target, and run model/KLD tests.
Record exact model, commands, sampling, hardware and source revision for any
quality or performance claim. This work does not deploy to Xbox or change a
serving configuration.
