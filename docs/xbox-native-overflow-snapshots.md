# Persistent native KVarN4 overflow snapshots

Status: implementation branch `feat/rkva-native-overflow-snapshots-20261027`.
This document describes the on-disk contract; it is not a hardware qualification
report. MSVC/D3D12 compilation, a real Xbox fresh-process roundtrip and logits
comparison remain mandatory before this path is enabled.

## Scope

The supported envelope is deliberately narrow and matches the current
production configuration:

- one sequence (`n_seq_max == 1`, `n_rs_seq == 0`, `seq_id == 0`);
- native KVarN4 overflow, descriptor revision 3 (`layout_version == 3`),
  128-token groups, 256-dimensional heads, two physical slices, 384 stage rows,
  exact-tail 0 or 128, and the 256-row original-F16 physical tail ring when the
  exact tail is enabled;
- `GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL=1`, `GGML_RKVA_RANGE_STORAGE=kvarn4`
  and `GGML_RKVA_DISK_SNAPSHOTS=1`;
- a server that advertises `RKVA_CAP_NATIVE_KVARN_SNAPSHOT` and supports
  snapshot operations 17/18 with request revision 2.

Anything else keeps the pre-existing failure guards: native Q4_0/Q8_0 row
layouts, descriptor revisions 1/2, multiple sequences, recurrent rollback,
partial native checkpoints and plain `llama_state_seq_set_data()` restore are
not silently reinterpreted. MTP target KV is covered by the native owner; the
draft/carry scheduler state is transient and must be rebuilt by the existing
server bootstrap. A caller that cannot bootstrap MTP must treat the
combination as unsupported. Model/layout identity is enforced by the server
slot manifest before streaming restore; direct `llama_state_seq_load_file`
callers must apply the same model/layout contract.

## Authoritative data split

The host owns the original projected inputs and the native producer state:

- normalized complete KVarN4 records relative to the remote origin group
  (`origin / 128`);
- the 384-row rotated F16 stage planes;
- the logical 128-row original-F16 precision tail (or no tail);
- the 256-row original-F32 rollback ring and its monotonic positions;
- rows/counts, geometry, descriptor parameters and the mutation generation.

The Xbox owns the GPU-resident revision-3 representation:

- the same complete records (stored relative to the range origin);
- both 384-row stage planes;
- the 256-row physical original-F16 tail ring;
- per-layer prefix coverage (`tail_begin`, `stage_begin`), record-group
  coverage, live rows and the resident mutation generation.

Neither side serializes pointers, C++ objects, D3D12 handles or transient
preparation caches. All integers are canonical little-endian; F16/F32 values
are raw IEEE bit patterns in little-endian order.

## Host state trailer

`llama_state_seq_save_file()` appends a `RKVK` set to the sequence state when
native overflow is resident and the snapshot envelope is supported:

```
u32 magic 'RKVK'      u16 version=1      u16 header_bytes=80
u32 layers            u32 flags
u32 origin            u32 capacity
u32 heads             u32 iterations
u32 precision_tail    u32 rows
u32 record_base_group u32 reserved
u64 saved_generation  u64 payload_bytes
u64 layer_bytes       u64 reserved
payload[layer][records K, records V, stage K, stage V,
               tail K, tail V, original K, original V, positions]
SHA-256(header || payload)
```

`record_base_group` records whether the exporting host used the global CPU
layout (`0`) or the compact CUDA layout (`origin / 128`). The payload itself is
always normalized to the origin group, so import is independent of the target
layout. The SHA-256 footer covers every field and payload byte. A reader
validates the complete header, exact derived sizes, position-ring invariants
and footer before any producer is mutated. Import builds shadow producers and
swaps them only after the complete set validates.

The trailer exists only for a non-empty native overflow. A pure local prefix
(`evaluated_tokens <= P`) keeps the legacy state format byte-for-byte.

## Xbox disk snapshot

The existing LocalState store is reused with a new namespace
`rkva-snapshots-v2/<host-prefix-sha256>/`. A native snapshot is one immutable
directory:

- `manifest-v2.bin`: canonical manifest (`RKVS` magic, version 1, geometry,
  descriptor, live rows, generation, per-layer coverage, chunk descriptors and
  a SHA-256 footer);
- `n-<index>.kv`: at most 8 MiB per chunk, each checked by SHA-256;
- publication is `tmp-<key>` directory, flushed files, then one rename. An
  existing published directory is never overwritten. Native retention never
  evicts a published pair during save: a full 4-snapshot/8 GiB store fails
  explicitly, so the previous host file can never point at deleted Xbox data.
  Incomplete `tmp-*` directories are removed. The legacy `rkva-snapshots-v1`
  format keeps its historical eviction policy.

The snapshot is bound to the SHA-256 of the complete host prefix through the
end of the RKVK frame (server header, token IDs, local KV/recurrent state and
the native trailer), the exact geometry/descriptor, the live row count and the
host-recorded generation. Server SCKP checkpoint bytes appended after the RKVK
frame are deliberately excluded, but they are not interpreted as native data.
Restore validates the manifest, every extent and every chunk digest before the
first GPU copy, rechecks each chunk on the copy pass, records all copies, and
publishes the new coverage/counts only after the final fence. Any failure marks
the session invalid; it never publishes partial coverage. Restore advances the
resident generation and returns it in the ACK so the host adopts the new
generation before a later rewind.

Legacy storage 0/1/2 and descriptor revisions 1/2 keep their original
`rkva-snapshots-v1` manifest and codec. Revision 2 requests are never sent to a
server that did not advertise the native capability, and an old server rejects
revision 2 before touching state.

## Capability negotiation

The host adds `RKVA_PROTO_NATIVE_SNAPSHOT_V4` (`1 << 18`) to HELLO
`proto_flags`. The server only advertises
`RKVA_CAP_NATIVE_KVARN_SNAPSHOT` (`1 << 18`) when that bit, native rewind and
serial-prefix negotiation succeeded and the revision-3 pipeline exists. Host
snapshot use requires both the capability and snapshot ops 17/18. Unnegotiated
peers keep the old fail-closed guards.

`rkva_snapshot_req.revision == 2` appends:

```
u64 expected_generation  // session generation the engine currently has
u64 saved_generation     // generation recorded in the host trailer/manifest
u32 format_version=1
u32 flags=0
```

and the reply appends:

```
u64 generation           // new resident generation after restore
u64 payload_checksum     // manifest checksum (diagnostic/binding)
u32 layers
u32 reserved
```

## Save/restore flow

Save:

1. `llama_state_seq_save_file` writes the token/state payload including the
   native owner trailer.
2. It hashes the complete host prefix through the RKVK frame and asks the Xbox
   for snapshot op17 with revision 2. Failure removes the temporary host file;
   the previous slot is untouched.
3. `server_route_state_save` publishes the host file only after the Xbox
   snapshot succeeds (native path). The Xbox directory is content-addressed and
   immutable.

Restore in a fresh host process / fresh HELLO / empty Xbox session:

1. The host verifies the state-file checksum, parses target memory into the
   normal staging transaction and parses the native trailer into shadow
   producers.
2. It parses the trailer into shadow state, then hashes the complete host
   prefix through the RKVK frame and asks the Xbox for op18. A same-size local
   mutation changes the key and fails before the destination commits.
3. The Xbox validates everything, uploads into the empty session, fences, and
   publishes coverage; it returns the new generation.
4. Only then does the host commit memory and producer state. A later append,
   rewind (<=128 rows), RESET, or 129-row rejection follows the existing
   revision-3 semantics.

No path falls back to cold prefill, Q4/Q8 bookkeeping or a half-restored owner.

## Review corrections

The preliminary review found and this branch fixes:

1. Native SAVE now validates resident counts/ready/record groups; native
   RESTORE does not consult the empty destination counters and validates the
   manifest into the empty session before the first copy.
2. A native FULL restore keeps the owner epoch installed with the restored
   snapshot instead of invalidating the range in the later FULL commit callback.
   Standard FULL restores still invalidate and resync.
3. Aggregate state limits use the actual layer count (the 16-layer production
   fixture is 544,227,328 bytes; a fixed 256-layer product was wrong).
4. Reusing an existing Xbox directory now re-verifies every chunk SHA-256, so a
   same-size bit flip cannot be acknowledged as a valid snapshot.
5. The host staging file is closed before rename/publication, and Windows
   publication uses `MoveFileExW` with a UTF-8 to UTF-16 conversion.
6. Native quota admission never selects or deletes a victim before the new pair
   commits; a full store fails explicitly.
7. Direct native save/load binds the Xbox snapshot to the complete host prefix
   through the RKVK frame, not only the trailer. A same-size mutation in the
   local state changes the key; the Xbox restore is refused before the local
   destination commits. Server SCKP bytes remain outside the key.

## CPU validation

The following was run in `/tmp/opencode/native-overflow-snapshots-cpu` with
`GGML_CUDA=OFF`, `GGML_VULKAN=OFF`, `LLAMA_CURL=OFF`, `GGML_NATIVE=ON`,
`CMAKE_BUILD_TYPE=Release` and `-j4`:

```bash
ctest --test-dir /tmp/opencode/native-overflow-snapshots-cpu \
  -R 'test-rkva-native-state|test-rkva-native-snapshot|test-native-save-file|test-rkva-kvarn-producer|test-state-file-stream|test-remote-attn-range|test-rkva-tcp-prefetch' \
  --output-on-failure
```

The suite covers producer/owner-state bitwise roundtrips (including the F32
rollback ring and tail 0/128), nonzero origins, global/compact record layouts,
multi-layer sets, wraps, rewind 128/re-append, 129 rejection, corruption and
short I/O, manifest geometry/coverage/chunk validation, same-size chunk bit
flips, quota admission, UTF-8 file publication/replacement, and the real TCP
revision-2 save/reset/restore/continue/rewind path against the loopback server.
It also appends a real SCKP appendix to the server sequence file and verifies
that a non-native FULL load stops at the state end and preserves the appendix;
the native TCP path appends SCKP after the RKVK frame, verifies the owner parser
leaves it unconsumed, and confirms a same-size mutation in the simulated local
prefix changes the key, fails the Xbox restore, and leaves the fresh destination
empty; a trailer mutation also fails before transport.

## Source map

- Host owner codec: `ggml/include/rkva-native-state.h`.
- Native producer export/import: `ggml/src/ggml-remote-attn/rkva-kvarn-producer.h`.
- Host snapshot APIs: `ggml/src/ggml-remote-attn/ggml-remote-attn.cpp`
  (`ggml_backend_remote_attn_native_kvarn_*`).
- Host state trailer integration: `src/llama-context.cpp`
  (`state_seq_write_data`, `state_seq_read_data`, `state_seq_save_file`,
  `state_seq_load_file_streaming`) and `src/llama-io*.h` remaining extents.
- Atomic host publication: `src/llama-native-save-file.h`.
- Wire contract: `ggml/src/ggml-remote-attn/remote-attn-protocol.h`.
- Xbox manifest codec: `ggml/include/rkva-native-snapshot.h` (shared through
  `tools/xbox/uwp/rkva/rkva-native-snapshot.h`).
- Xbox request path: `tools/xbox/uwp/rkva/remote-attn.cpp`.
- Xbox disk/publication/quota/writeback: `tools/xbox/uwp/rkva/rkva-disk-snapshot.inc`.
- Server publication gate: `tools/server/server-route-state.cpp` and
  `tools/server/server-context.cpp`.

## Pending qualification

The CPU suite exercises the real codecs, producer export/import and manifest
validation with synthetic buffers and storage. It does not compile UWP/D3D12.
The following remain pending for the root review/hardware owner:

- MSVC compile of `rkva-disk-snapshot.inc` and the UWP project;
- real Xbox save -> process restart -> fresh HELLO -> restore -> continuation;
- rewind 128 and re-append after restore;
- 129-row rejection after restore;
- logits comparison against the pre-save continuation;
- disk-full/interrupted-write and missing/corrupt snapshot tests on the console.
