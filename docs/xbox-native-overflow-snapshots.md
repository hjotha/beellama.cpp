# Persistent native KVarN4 overflow snapshots

Status: experimental, opt-in implementation. The Xbox build
and real fresh-session roundtrips described below passed, but lifecycle and
failure limitations remain. Do not treat this as fully qualified for production.

## Scope

The supported envelope is deliberately narrow and matches the current
production configuration:

- one sequence (`n_seq_max == 1`, `n_rs_seq == 0`, `seq_id == 0`);
- native KVarN4 overflow, descriptor revision 3 (`layout_version == 3`),
  128-token groups, 256-dimensional heads, two physical slices, 384 stage rows,
  exact-tail 0 or 128, and the 256-row original-F16 physical tail ring when the
  exact tail is enabled;
- native KVarN4 explicitly enabled, KVarN4 selected as the remote upload
  format, and `GGML_RKVA_DISK_SNAPSHOTS=1`;
- a server that advertises `RKVA_CAP_NATIVE_KVARN_SNAPSHOT` and supports
  snapshot operations 17/18 with request revision 2.

Anything else keeps the pre-existing failure guards: native Q4_0/Q8_0 row
layouts, descriptor revisions 1/2, multiple sequences, recurrent rollback,
partial native checkpoints and plain `llama_state_seq_set_data()` restore are
not silently reinterpreted. The explicit slot API requires a target-only
context with no active draft or speculative slot; it does not persist or
restore MTP draft/carry state. The current production XXXL profile sets draft
`n_max=0`, and its transition removes the active draft context; enabling MTP in
that tier would require separate support and qualification.

The server slot manifest enforces model/layout identity and installs a RAM
reservation hook before native streaming restore. Generic
`llama_state_seq_load_file` and generic streaming readers reject the RKVK
trailer. Direct callers must explicitly use `llama_state_seq_load_file_native`
or `llama_state_seq_load_file_streaming_native` and provide the appropriate
identity checks and memory admission themselves.

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
  explicitly rather than selecting an existing Xbox snapshot as a victim.
  Incomplete `tmp-*` directories are removed. The legacy `rkva-snapshots-v1`
  format keeps its historical eviction policy.

Host deletion and retention do not currently delete or reconcile the Xbox
directory. Consequently, even successful saves can fill the remote store after
up to four distinct snapshots (or earlier at the byte limit), and removing a
host slot alone does not reclaim that quota. The host runs disk-budget and
retention preflight before issuing op17, but failure or a crash after the Xbox
save and before host publication can still leave an orphan. Existing keys can
be shared or reused, so blindly deleting a remote directory on failure is not
safe. A reference-aware lifecycle/reconciliation mechanism remains necessary.
The file flushes and rename are not proof of durable cross-device atomicity
after power loss.

The snapshot is bound to the SHA-256 of the complete host prefix through the
end of the RKVK frame (server header, token IDs, local KV/recurrent state and
the native trailer), the exact geometry/descriptor, the live row count and the
host-recorded generation. Server SCKP checkpoint bytes appended after the RKVK
frame are deliberately excluded, but they are not interpreted as native data.
Restore validates the manifest, every extent and every chunk digest before the
first GPU copy, rechecks each chunk on the copy pass, records all copies, and
publishes the new coverage/counts only after the final fence. Failures after
GPU mutation invalidate the session; validation failures before mutation do
not publish partial coverage. Restore advances the
resident generation and returns it in the ACK so the host adopts the new
generation before a later rewind.

Legacy storage 0/1/2 and descriptor revisions 1/2 keep their original
`rkva-snapshots-v1` manifest and codec. Revision 2 requests are never sent to a
server that did not advertise the native capability, and an old server rejects
revision 2 before touching state.

## Capability negotiation

The host adds `RKVA_PROTO_NATIVE_SNAPSHOT_V4` (`1 << 18`) to HELLO
`proto_flags`. The server advertises `RKVA_CAP_NATIVE_KVARN_SNAPSHOT`
(`1 << 18`) for the native-snapshot client in the supported F16/domain HELLO.
Actual host admission also requires native serial-prefix capability, the
revision-3 descriptor, the snapshot capability and ops 17/18. Unnegotiated
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
2. The server checks the staged host size and host retention budget before the
   remote SAVE. The host then hashes the complete prefix through RKVK and asks
   the Xbox for snapshot op17 with revision 2. A failed SAVE removes the
   temporary host file; the previous slot is untouched.
3. `server_route_state_save` publishes the host file only after the Xbox
   snapshot succeeds (native path). The Xbox directory is content-addressed and
   immutable. An error after op17 can leave an unreferenced remote directory.

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
   rewind (<=128 rows), RESET, or 129-row rejection is expected to follow the
   existing revision-3 semantics; those boundaries still require Xbox testing
   after a native disk restore.

No path falls back to cold prefill, Q4/Q8 bookkeeping or a half-restored owner.

## Explicit adaptive streaming slot API

The normal adaptive `LLAMASLT` path still uses `llama_state_seq_get_data` /
`set_data` in RAM. It is unchanged and does not activate the native Xbox pair.
Native streaming slots are explicit opt-in:

```http
POST /slots/0?action=save
{"filename":"native-xxxl.slot","native_stream":true}

POST /slots/0?action=restore
{"filename":"native-xxxl.slot","native_stream":true}
```

The option is rejected unless all of the following hold:

- adaptive server, active profile `xxxl`;
- target-only context: no draft/MTP context and no speculative slot;
- one sequence;
- a qualified native revision-3 peer and geometry;
- SAVE: native overflow resident and a complete evaluated text prefix;
- RESTORE: empty target slot and no occupied sequence state;
- no `route_state_transfer`, `route_target_no_mtp` or `route_state_reuse`;
- text-only prompt, verified model identity, no LoRA/control vectors and
  `--slot-save-max-mb` bounds.

Without `native_stream`, saves with live native overflow are rejected; normal
legacy/adaptive saves remain available otherwise. `ADAPTIVE_SLOT_WORKING_COPIES`
remains five for `LLAMASLT`. Native overflow auto-save is disabled, and a native
file discovered by automatic or router lookup is rejected by the generic reader.
SAVE uses the unified checksummed route-state envelope:
its manifest records the model fingerprint and complete prompt-cache layout, and
the host/Xbox pair is published only after the Xbox save succeeds. RESTORE uses
the bounded streaming reader, validates model/layout/adapters before mutation,
and reserves the estimated staged-RKVK + owner-allocation peak against the
prompt-cache RAM budget before allocating the trailer; insufficient reservation
fails with the destination unchanged.

Admission is cumulative and uses `adaptive_slot_ram_budget`: the smaller of
the configured `cache-ram` limit and measured `MemAvailable` after the server's
existing 25% reserve. The bounded streaming workspace is reserved first; the RKVK staged/owner peak is then admitted
against the remaining hard limit. SAVE reserves its own `16 MiB + C*16`
workspace under the same cumulative cap before synchronization, size probing,
token copies or route-state publication.
`prompt_cache.reserve_transient` still enforces its own configured cap. A
rejected reservation prevents native trailer allocation and publication; the
workspace already reserved is released when the operation ends.

The RKVK header is peeked before `read_set` allocates. The estimator adds:

- staged `RKVK` bytes (`header + payload + footer`);
- one full-capacity owner allocation per layer (the destination is empty, and
  owners are created before the staged vectors are swapped in);
- a bounded allocator margin.

For the real 16-layer fixture (`origin=102400`, `capacity=26624`, `heads=4`,
payload `544,227,328` bytes) the owner bound is `148,702,208` bytes/layer and
the restore peak is about `2.95 GiB` including margin. It fits a 4096 MiB
prompt-cache budget but is rejected in full by a 2 GiB budget. The explicit
slot handler installs the reservation hook; the generic router reader rejects
native trailers even if it installs the hook. A fixed workspace is no substitute
for the owner peak.

### Selecting xxxl without a large prefill

1. Start the adaptive server with the intended `--slot-save-path`, the xxxl
   profile configured, `--cache-ram 4096`, and slot auto-save disabled.
2. Send one very small completion request with a large output reserve so the
   computed `prompt + n_predict` budget exceeds the xxlong limit. The profile
   switch runs before scheduling; only the tiny prompt is prefilled. Wait for
   the request to finish and all slots to become idle.
3. Erase the target slot: `POST /slots/{id}?action=erase`.
4. Use the `native_stream` requests above. RESTORE requires the same xxxl
   profile, an empty target slot and no resident native owner rows. If the
   snapshot was saved in the same resident xxxl context, re-enter the profile
   (or restart/recreate the context) so the Xbox owner is cleared; emptying the
   prompt alone does not erase native owner rows.

Do not claim the default adaptive slot path covers native snapshots: the
`native_stream` marker is mandatory.

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
8. The explicit adaptive slot uses the checksummed route-state manifest, so the
   complete model fingerprint and prompt-cache layout are validated before the
   memory transaction. LoRA, control vectors, multimodal contexts and a missing
   model identity are rejected.
9. Explicit RESTORE uses the bounded streaming reader, not the direct
   `llama_io_read_file` materialization. Its workspace plus the measured native
   staged/owner peak is reserved before allocation.
10. Preflight rejection no longer clears an occupied slot; only a tracked
    mutation triggers rollback/clear.

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

## Xbox validation on 2026-10-29

The UWP package `1.6.0.411` was built with MSVC/D3D12. The qualification used
the staged worktree over `724613730`, a separate host binary, the current
Qwen3.8-27B model and one sequence, with `P=110592`, `ctx=162816`,
`batch=512`, `ubatch=512`, KVarN4 native overflow, XXXL draft disabled, TCP
prefetch and `GGML_RKVA_DISK_SNAPSHOTS=1`. The isolated server temporarily
enabled auto-cache with a 262144-token block to select the local-prefix
streaming route; the production preset had auto-save disabled.

- At 110848 evaluated tokens, native save, a fresh host/Xbox app session,
  restore and a 256-token continuation passed; the continuation token IDs
  matched before and after restore. Evidence:
  `/home/hjotha/.codex/artifacts/native-snapshot-20261029/qualification-20261029-r2/`.
- At 160000 evaluated tokens, the same roundtrip and 256-token continuation
  comparison passed. The 119700-token input corpus was repeated from its start
  to form the 160000-token prompt; the effective token stream was hashed and
  the saved 110592-token local prefix was reused. Evidence:
  `/home/hjotha/.codex/artifacts/native-snapshot-20261029/qualificacao-160k-r1/`.
- The 256-token continuation result files record
  `probability_comparison.available=false`. A separate 16-step teacher-forced
  comparison at 160000 subsequently compared the complete vocabulary of
  pre-sampling log probabilities between the pre-save baseline and restored
  execution: all serialized values were identical, KL was zero, and no values
  were saturated. These are log probabilities derived from float softmax, not
  raw logits. Evidence:
  `/home/hjotha/.codex/artifacts/native-snapshot-20261029/logprobs-160k-r1/comparison.json`.
- A host v2 build restored the earlier 160000-token snapshot with the Xbox
  `1.6.0.411` app and compared 16 teacher-forced complete-vocabulary
  log-probability vectors against the earlier host baseline: all serialized
  values were identical, KL was zero and there was no saturation. This is a
  cross-version restore test, not a raw-logit comparison. Evidence:
  `/home/hjotha/.codex/artifacts/native-snapshot-20261029/cross-v2-160k-r1/comparison.json`.
- With snapshot support disabled, the host v2 build passed an isolated local
  MTP and XXXL KVarN4 save/restore, corruption-clear, erase and continuation
  smoke. A separate Q8 native overflow smoke passed at a reduced, fixed 2048
  context and split 256, with MTP disabled; it is not a full-production-tier
  Q8 benchmark. Evidence:
  `/home/hjotha/.codex/artifacts/native-snapshot-20261029/snapshot-off-v2-r1/result.json`
  and `/home/hjotha/.codex/artifacts/native-snapshot-20261029/q8-v2-r2/result.json`.
- The UWP package `1.6.0.412` was built but not installed; the host v2 hardware
  tests used the installed `1.6.0.411` app.

Separate negative tests passed for a corrupted host-file copy, rejection under
a 512 MiB host RAM limit, host retention preflight denial with SAVE op17 blocked
at a tracing proxy, and a proxy-injected nonexistent Xbox snapshot key. The
host-corruption and missing-key cases subsequently restored the good snapshot
and reproduced its continuation. Evidence is under
`/home/hjotha/.codex/artifacts/native-snapshot-20261029/failure-tests-20261029/`.
These tests check visible slot emptiness and remote directory names/types; they
do not prove the physical owner is empty, compare Xbox chunk bytes, measure
actual peak RAM, exercise Xbox quota exhaustion or corrupt a console chunk.

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

## Pending qualification and lifecycle work

- Add reference-aware remote cleanup/reconciliation and recovery for the
  post-op17 failure window; host retention alone cannot reclaim Xbox quota.
- Verify rewind 128 and re-append, and rejection of rewind 129, after a real
  Xbox disk restore.
- Compare full-vocabulary logits against the pre-save continuation.
- Exercise console quota exhaustion, missing/corrupt console chunks, failure
  after peer mutation, disk-full/interrupted writes and power-loss recovery.
- Qualify any draft/MTP-enabled XXXL configuration separately; the current
  implementation supports only a target-only snapshot.
