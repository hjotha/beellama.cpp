# Experimental native KVarN4 Xbox overflow

This opt-in path keeps CUDA attention for positions `[0,P)` and consumes genuine
native KVarN4 records on the Xbox for `[P,C)`. It is append-only and does not yet
have a native checkpoint codec. Existing Q4/Q8 overflow snapshots cannot initialize
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

Each layer must begin with an original-data append exactly at P. Epoch changes,
real rewinds, partial restores after native rows exist and bootstrap from occupied
standard-cache overflow fail closed. Trimming an unoccupied suffix is a no-op.
FULL state serialization is rejected once native overflow exists; automatic disk
save skips that transient owner. Partial RS checkpoints may be captured, but cannot
be restored over a live native owner. Use a new context for another independent
run. Native disk snapshots and fused standard-cache upload/attention are disabled.

The `test-rkva-kvarn-producer` regression replays native upload packets into an
independent console-layout image and compares completed records and stage bytes
with a canonical whole-batch CPU store over the original F32 input. It covers
stage/tail wrap across a 256-token append, chronological original-domain tails,
non-finite input refusal and replay refusal. This source regression does not
establish model quality or Xbox throughput; those require actual device/model
runs and their exact provenance.
