# Experimental native KVarN4 Xbox overflow

This opt-in path keeps CUDA attention for positions `[0,P)` and consumes genuine
native KVarN4 records on the Xbox for `[P,C)`. It supports negotiated bounded resident rollback, but does not yet
have a native FULL checkpoint codec. Existing Q4/Q8 overflow snapshots cannot initialize
it. Start a fresh server, restore a pure local prefix ending at or before P, and
prefill the overflow from original projected values.

The explicit environment controls are:

```bash
GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL=1
GGML_RKVA_RANGE_STORAGE=kvarn4
GGML_RKVA_RANGE_KERNEL=balanced
GGML_RKVA_RANGE_FUSE=0
```

The ordinary remote companion flags remain Q4_0 or Q8_0 for metadata/bookkeeping.
Its quantized values are not the native producer's input. Two additional graph
operands carry original projected F32 K/V. The CPU backend's canonical KVarN store
performs WHT, F16 staging, eager record quantization and rounding; the RPC layer
does not duplicate those algorithms. Native records, canonical rotated stage rows
and original-domain F16 precision-tail rows are sent through RKVA operations 15/16.
Operation 14 returns original-domain OUT and natural-log LSE for the existing CUDA
merge. There are no attention or upload RPCs below P.

Revision 1 requires 256-dimensional heads, 128-token groups, two physical slices,
three canonical stage slots, eager K4/V4 records and an F16 precision tail of at
most 128 requested tokens (the native descriptor uses a 128-token tail). Query
batches are limited to 256. The CPU producer retains full global record addressing
to match the canonical store. With C=131072 and four KV heads this adds about
2.35 GB of host record storage across 16 attention layers, in addition to the
existing bookkeeping companion. Only actual overflow groups cross the network.

Each layer must begin with an original-data append exactly at P. Bootstrap from
occupied standard-cache overflow fails closed. FULL state serialization is rejected
once native overflow exists; automatic disk save skips that transient owner.
Native disk snapshots and fused standard-cache upload/attention remain disabled.
Without the negotiated rewind extension, real rewinds and live PARTIAL restores
are refused; trimming an unoccupied suffix remains a no-op.

The `test-rkva-kvarn-producer` regression replays native upload packets into an
independent console-layout image and compares completed records and stage bytes
with a canonical whole-batch CPU store over the original F32 input. It covers
stage/tail wrap across a 256-token append, chronological original-domain tails,
non-finite input refusal and replay refusal. This source regression does not
establish model quality or Xbox throughput; those require actual device/model
runs and their exact provenance.

Native MTP requires capability bit 6, HELLO bit 16 and operation 21. The
resident rewind transaction checks the configuration generation and every layer's
frontier before changing state. The host retains 256 original-F32 input rows,
rebuilds the affected incomplete group and repairs original-F16 exact-tail rows.
A rewind can remove at most 128 rows. Draft depth is therefore capped at 127,
including room for the verification anchor. Older servers refuse active MTP
before the first speculative write, including when overflow is still empty.

A PARTIAL restore additionally requires the same hybrid owner, dirty-prefix
lineage and retained native coverage. These checks precede metadata/recurrent
mutation; an uncertain transport ACK poisons the remote owner. This resident
contract cannot import a FULL native snapshot or initialize native KV from Q4/Q8
bookkeeping. Standard Q4/Q8/F16 overflow uses its existing rewind contract.

Descriptor revision 3 additionally negotiates HELLO bit 17 and capability bit 7.
Each query reads its own serial-equivalent stage, sealed-record boundary and exact
tail, even when an append batch crosses a 128-token group seal. Up to nine queries
are sent together; larger prefill batches use eight-query chunks. Revisions 1/2
retain interleaved one-row append/query calls to preserve those semantics.
The logical 128-row exact tail uses a 256-row physical original-F16 ring in
revision 3. `GGML_RKVA_NATIVE_TAIL_TOKENS=0` omits that ring and tail payloads;
it leaves record/stage storage and original-F32 rollback retention intact.

`GGML_MTP_PROF` also enables `MTPCYCLE` records for generation (including the
non-speculative control). Each record reports request/slot/cycle, initial anchor,
local/overflow tier, proposed and accepted drafts, logical target query rows and
passes including checkpoint replays, generated token count and complete wall
microseconds through token processing. Generated counts follow server `n_gen`,
including a stop/EOS token; they are not visible text-token counts. This cycle
span includes queue, checkpoint, draft, target, catch-up, restore/replay and token
processing costs. Existing phase and RKVA profile logs provide the breakdown.
A cycle interrupted by cancellation or an error has no completed record.

Standard Q4/Q8/F16 overflow can retain an unchanged prefix after successful
suffix removal or a validated owner-bound PARTIAL restore. The context sends
one session-wide REWIND, waits for its ACK, then clamps every cached layer's
frontier. The next verification uploads only its new/replacement suffix. A
no-op trim sends no RPC. Unsupported older servers retain full-upload fallback;
an ACK failure makes the remote owner unusable. FULL restores, arbitrary range
removals and unannounced backward/same-count rewrites retain full resynchronization.
Overwriting an existing target position also advances the mixed owner generation,
so a PARTIAL checkpoint cannot validate merely through matching token/position
IDs after the prefix bytes changed. These rules use existing memory/restore
transactions and the existing standard RKVA REWIND operation; native overflow
continues to require a fresh context for unsupported rollback/state operations.

Resident prefix tickets also record dirty suffix generations. Removing positions
at or beyond r invalidates any saved prefix that included those positions, even
if the same cell positions and token IDs are later reused with different bytes.
A checkpoint ending strictly before r remains valid, including repeated restore
and replay. Dominated suffix edits are coalesced; FULL restore and clear start a
new lineage. This prevents token/position identity from substituting for byte
ownership while retaining the accepted-prefix fast path.

Dirty suffix history is not part of durable FULL snapshots. Position-split
checkpoint sidecars therefore adopt the FULL save-time epoch and keep only
tickets captured in that epoch; older/missing-epoch tickets fall back to
reprocessing instead of being revived through epoch adoption.

For positional standard companions, hybrid PARTIAL state includes the attention
ownership envelope followed by recurrent state. Historical RS-only PARTIAL
blobs are explicitly refused before commit; they cannot authorize remote prefix
retention. FULL and pure FULL imports retain their existing formats.
