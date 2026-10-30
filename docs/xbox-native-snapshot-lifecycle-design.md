# Native overflow snapshot lifecycle: safe reconciliation design

Status: **design only**, not implemented or qualified. The existing revision-2
native SAVE/RESTORE protocol and its conservative 4-object / 8-GiB admission
remain unchanged. The proposal does not authorize deleting existing snapshots.

## Current behavior and limits

The host publishes a local state file after an Xbox SAVE keyed by the SHA-256 of
the host state prefix. The Xbox stores immutable objects in
`rkva-snapshots-v2/<key>`. A byte-identical host prefix can be referenced by
multiple files, multiple processes, different machines, or an offline copy.
The host quota preflight reduces the window but a crash, I/O error, lost ACK or
retention failure after the remote SAVE can still leave an unreferenced object.
The remote store currently cannot tell that object apart from one referenced by
an unavailable host or an offline copy. Neither file age nor a scan of one host
is evidence that a key is unreferenced.

The native quota admission path rejects a fifth object or an 8-GiB excess
instead of evicting a published object. This protects existing references but a
store full of orphans can block all new saves. The current code also removes
`tmp-<key>` directories during admission; the existing listener processes one
connection at a time, but no cross-process durable ownership is recorded. New GC
logic must not assume an unknown temporary directory is safe to remove during a
concurrent writer or external management operation.

A host file that has been copied or exported outside a managed registry cannot
be tracked from that registry. No automatic scheme can safely infer that its
Xbox state is unused. The legacy v2 namespace therefore remains protected by
default and is never imported into a collectable registry by scanning one host.
Migration requires an explicit opt-in inventory of all intended copies and an
ownership policy; unresolved or unavailable copies remain pinned indefinitely.

## Proposed managed protocol and authority

Introduce separately negotiated operations and a new managed namespace. The
Xbox owns a durable, checksummed registry keyed by `(owner_id, reference_id)`;
records bind a host identity, a stable random reference ID, a transaction ID,
the complete host-prefix SHA-256, geometry, generation, and state. The host owns
a durable journal for the same transactions, protected by its snapshot store
lock. IDs are generated randomly and never derived solely from a path or PID.
Each managed file contains its reference identity and an explicit statement
that copies require separate registration. The reference ID belongs in a
checksummed manifest outside the key's hashed data prefix: placing it in that
prefix would turn otherwise identical content into different keys. Export must
register the new reference before releasing the old one. An out-of-band file
copy is invisible to the registry: lifetime safety for such a copy is impossible
unless the original reference stays pinned indefinitely. Automatic GC can
therefore be enabled only for an explicitly managed store whose users accept
that scope; the default protects unmanaged and externally copied snapshots.
Existing v2/unmanaged files have no registered reference and are never
automatically deleted.

The Xbox serializes all managed operations across sessions and processes using
a store-wide authority whose durability and exclusivity are established on the
actual UWP filesystem. A transactional log or atomic database must survive
power loss; file flush, rename and parent-directory durability must be verified
on the platform rather than inferred from `ofstream::flush`. Idempotent request
IDs and monotonically increasing registry versions reject stale updates. The
protocol uses a new capability and revision, bounded length fields and explicit
negative replies; older peers retain the current fail-closed path. Registry
ownership and release authorization require an authenticated peer identity or
an explicit single-owner trust boundary; unauthenticated callers must not be
able to impersonate another owner and release its pins.

## Publication and reconciliation

1. Host serializes to a private temporary file and validates size, checksums,
   model identity and host quota. It durably journals `PREPARE(tx, ref, key)`
   **before** requesting the peer. It does not remove an existing slot.
2. Peer admits temporary bytes against quota, writes and verifies the object in
   a unique transaction staging directory and durably records a **pending pin**.
   SAVE is idempotent by transaction and key. Existing deduplicated bytes are
   validated and get an additional pending pin, never deleted on failure.
3. Host atomically publishes and durably syncs its file and containing
   directory, then journals `HOST_PUBLISHED` with the exact path and identity.
   A reader observing this file before peer COMMIT must treat it as pending and
   either wait for transaction recovery or fail closed.
4. Host sends `COMMIT(tx, ref, key)`. Peer atomically records the durable
   reference, releases the pending pin and acknowledges the committed version.
   A lost ACK is retried using the same transaction ID. Only then may the host
   publish success and consider replacing/releasing the old reference.
5. To release a managed reference, host durably records the slot transition or
   deletion first, proves the reference belongs to its exclusive managed store,
   then sends an idempotent `RELEASE(owner, ref, expected_version)`. A peer may
   collect a key only under the same store authority when **no committed
   reference, pending transaction, active reader pin, unknown legacy pin or
   quarantine hold** can reference it. Logical tombstone and physical deletion
   must be crash-recoverable and must exclude new pins/restores until completed.

Recovery replays the host journal under its lock and asks peer transaction
status. After host publication it retries COMMIT; when the host can prove the
file was not published it may ABORT its own pending transaction. If identity,
source path or host durability cannot be proven, the pending pin is retained;
timeouts alone never authorize deletion. A permanently unavailable owner may
require explicit human-assisted reconciliation. The host must never release a
reference while still publishing or restoring that reference in another thread.

At quota, pending and quarantined bytes count towards admission. If the new
snapshot cannot coexist with the old one during a replacement, reject safely;
no implementation may delete the old blob to make room before the new pair is
committed. Reserved capacity or a larger explicitly configured limit can make
replacement practical, but is not a substitute for power-loss-safe accounting.

## Required tests before implementation can be enabled

All tests below must run against the real implementation, not only a mock
reference model; tests in CPU/temporary directories precede UWP qualification.

- Two managed host files and two hosts reference the same key: release either
  reference and verify bytes/restore for every remaining reference; release the
  final one and verify only then that GC may collect.
- A copied, unregistered file and every v2 object remain pinned by default;
  incomplete or ambiguous imports fail closed. Lost/offline owners never expire.
- Inject process crash and lost ACK before/after each journal write, peer
  staging, rename, directory sync, host publish, COMMIT, RELEASE, tombstone,
  delete and ACK; replay must be idempotent and preserve each published pair.
- Concurrent SAVE of the same key, distinct references, duplicate/replayed
  request IDs, RESTORE versus GC, racing RELEASE and COMMIT, writer death and
  store restart: no referenced blob is deleted, no stale epoch succeeds and no
  writer removes another writer's staging directory.
- Disk full, count quota full, corrupt/truncated registry, manifest or chunks,
  symlinks, interrupted directory sync and permissions failure all fail closed;
  quota and temporary bytes remain accurately charged after restart.
- Replacement under count/byte pressure leaves old host and peer snapshots
  restorable until the replacement pair is durable and committed. Snapshot
  eviction on host cannot bypass registered remote-reference release.
- UWP tests exercise actual Windows path, lock, flush and recovery semantics;
  real Xbox tests validate restart/power interruption, peer session recovery and
  post-restore logits/continuation. Missing instrumentation cannot prove that
  the GPU data or remote registry survived.

The existing `test-rkva-native-snapshot` only checks the pure admission helper
and manifest/chunk codecs. It does **not** prove remote directory preservation,
registry durability, transaction recovery, or multi-host GC safety.
