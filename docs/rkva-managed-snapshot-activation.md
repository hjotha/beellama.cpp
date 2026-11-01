# Managed snapshot activation composition (default off, CPU only)

This worktree adds the activation-phase pieces that Phase 4 listed as
irreducible blockers 1, 4, 5 and 6:

1. one RAII store authority across the host journal, the orchestrator, the
   anchored SQLite store and the v3 object/quota locks (blocker 6);
2. the authenticated successful COMMIT/STATUS reply that publishes
   `commit_ack` from a verified receipt, object proof and successor anchor
   (blocker 1);
3. a listener-minted session authority test seam bound to the accepted
   connection, challenge and state mutex (blocker 4);
4. receipt-based lost-ACK classification and STATUS recovery (blocker 5).

It is CPU-only library, harness and test code. There is no listener, RPC,
capability, HELLO, dispatch, GC, GPU, Xbox, UWP or production path, no
advertisement of any managed op, and no existing behaviour changes while the
feature is off. **This is not production activation.** Items 2 (authenticated
pin/metadata proof), 3 (qualified durable sync/rename) and 7 (UWP/console
qualification) still require the real Xbox/UWP hardware and are unchanged.

## Activation

Both gates must be open; both default to off.

- Compile time: CMake option `RKVA_MANAGED_SNAPSHOT_ACTIVATION` (default
  `OFF`). It is only accepted together with
  `-DRKVA_MANAGED_SNAPSHOT_PHASE4=ON` and otherwise fails configuration. It
  builds the static library `rkva-managed-activation` and the activation tests,
  and defines `RKVA_MANAGED_SNAPSHOT_ACTIVATION` for them.
- Run time: `rkva_managed_activation::enabled()` in
  `tools/server/server-rkva-managed-activation.h` returns true only when the
  build is compiled in **and** `RKVA_MANAGED_SNAPSHOT_ACTIVATION=1` is set.
  Unset, empty, `0`, `true`, `01`, ` 1`, `1 ` and any other value return false.
  `rkva_session_authority::authority::available()` requires the same gate.

There is deliberately no call site. `HELLO` is unchanged, `supported_ops` does
not contain ops 24–31, and no default build compiles the activation library.
The successful-reply guard also requires an independent proof object that does
not exist in this tree, so even an enabled build answers `unavailable`.

## Item 6: one store authority

`tools/server/server-rkva-managed-authority.{h,cpp}` acquires the three
filesystem locks exactly once, in this documented order, and keeps them until
the authority is destroyed:

| order | lock | covers |
| --- | --- | --- |
| 1 | `<parent>/.rkva-managed-store.lock` (`rkva_managed_store::lease`) | anchored SQLite store, v3 object writer/verifier, v3 quota scan |
| 2 | `<parent>/host-journal.lock` | host journal |
| 3 | `<parent>/.managed-orchestrator.lock` | orchestrator |

Rules enforced by the code:

- **No nesting.** `rkva_host_journal::journal` and
  `rkva_managed_orchestrator::orchestrator` accept the authority through the
  new `rkva_managed_store::delegation` interface and run in *delegated* mode:
  they enter the authority for the whole operation and never open or lock
  `host-journal.lock` / `.managed-orchestrator.lock` themselves. The legacy
  constructors keep their own locks unchanged.
- **Same token.** `store::concrete()` returns the one `lease` that the anchored
  SQLite backend and the v3 object writer/verifier require; the v3 quota scan
  is called inside `store::enter()`.
- **Non-blocking, fail-closed.** Acquisition uses `O_NOFOLLOW`, `LOCK_NB` and
  inode/owner/mode revalidation. Contention (`EWOULDBLOCK`), a symlink, a
  missing or non-private parent, a replaced lock pathname, a stray hardlink, a
  forked child or a lost lease all fail closed and release everything already
  acquired.
- **Publication revalidation.** A delegated journal revalidates the authority
  immediately before the staging `fsync` and again after the rename, so a lock
  replaced mid-operation is refused before publication (or reported as an
  uncertain outcome after the rename).
- **Lifetime.** A delegated journal/orchestrator holds a `shared_ptr` to the
  authority, so the lock descriptors cannot be dropped while a component still
  uses them.
- **Legacy locks kept.** The standalone journal and orchestrator still take
  their own locks, so existing users and tests are unchanged. The authority
  also holds those same files, which means a legacy writer blocks instead of
  interleaving; the lease excludes legacy SQLite/object writers.

What it does **not** do: it does not make the three files one atomic
transaction, it does not turn `fsync` + `rename` into a power-loss proof
(item 3), and it does not authenticate a pin, an object or metadata (item 2).

`tests/test-rkva-managed-authority.cpp` covers: documented order, one holder
per store in-process and across processes, legacy lease/journal/orchestrator
exclusion, thread serialization, delegated journal and orchestrator use
(initialize, prepare, publish, release), SQLite provisioning and inspection on
the same token, quota scan under the section, object-v3 token acceptance,
symlinked/missing/non-private parents, a lock pathname replaced after
acquisition, a forked child (inherited and fresh), process death (kernel lock
release) and an end-to-end orchestrator publication + release.

## Item 1: authenticated successful COMMIT/STATUS reply

`tools/server/server-rkva-host-v3-success.{h,cpp}` implements the concrete
success `reply_ack_guard` on the Phase 4 v3 framing. It verifies, in order:

1. the reply guard is called exactly once and only for an authenticated
   `RKVA_OK` reply to op 25 (COMMIT) or 28 (STATUS);
2. the listener session authority is held and bound to this exact session
   (owner, ledger epoch, auth epoch, transcript, current peer challenge);
3. the acknowledgement is a `committed` record carrying a receipt whose
   canonical host MAC verifies (`RKVA-V3-COMMIT-RECEIPT-V1`, HMAC-SHA256 over
   the canonical receipt bytes without `mac`);
4. the receipt's `previous_version`/`previous_digest` equal the durable journal
   anchor exactly;
5. the successor advances the anchor by exactly one version, its digest equals
   the replied state digest, and the journal ledger still has the unspent v3
   nonce for this operation, transaction, reference, key, transcript,
   challenge and complete-frame request hash;
6. an independent `v3_success_proof` accepts this exact receipt under the same
   concrete lease;
7. only then `journal::commit_v3_success()` performs one atomic whole-file
   publication: consume the nonce with the authenticated response hash, advance
   the anchor and set `commit_ack`.

Receipt MAC, anchor and proof failures return `unavailable`/`replay` with the
precise reason in `error`; the journal is not modified and the session is
invalidated. `commit_v3_success()` is idempotent only for an identical,
already-completed publication; any other mismatch fails closed.

There is **no production `v3_success_proof`**. The header says so, the guard
takes a pointer, and a null or failing proof fails closed before any durable
change with one of:

- `single store authority unavailable (item 6)`;
- `concrete authenticated pin and metadata proof unavailable (item 2)`;
- whatever precise refusal the proof returns (for example qualified durable
  sync/rename, item 3).

The tests use a test-only proof double. It is a fixture, not evidence, and it
is never linked into a shipping target (the activation library is default off
and has no call site).

`tests/test-rkva-host-v3-success.cpp` covers: missing authority, missing proof,
durability refusal, tampered receipt MAC with a recomputed outer tag, forged
previous anchor, successor skipping a version, object-proof mismatch, wrong
nonce, wrong session transcript, valid success, replay, and ledger-side binding
(wrong request hash, wrong transaction) plus a crash at both persist stages.

## Item 5: lost-ACK classification and recovery

A crash between the peer commit and the host ACK leaves the transaction in
`host_published` with an unconsumed v3 nonce. The activation layer:

- refuses to issue a second COMMIT for that transaction
  (`v3 lost-ACK ambiguity: an unconsumed COMMIT nonce exists; use
  authenticated STATUS recovery`) so a fresh nonce can never advance the
  anchor twice for one peer commit;
- `rkva_host_journal::lost_ack()` classifies the transaction as
  `ambiguous`, reports the pending operation and a nonce fingerprint, and
  leaves it pinned, charged and quarantined. Nothing is deleted or repaired;
- `issue_status_recovery()` builds and durably issues the authenticated STATUS
  request. A challenge is single-use, so recovery happens on a fresh
  authenticated session (new HELLO transcript and challenge) bound to the same
  accepted connection authority;
- `accept_v3_success()` accepts the STATUS reply only through the same receipt,
  anchor, MAC and proof checks, publishes `commit_ack` once and reports
  `recovered = true`. There is no automatic anchor replacement and no
  timeout-driven cleanup.

Tests cover the ambiguous state, the refused COMMIT retry, a no-proof recovery,
a forged recovery receipt (anchor unchanged, still ambiguous), a valid recovery,
replay of the recovery reply, and a crash between peer commit and host ACK
followed by a successful recovery. The crash-in-publication test also reopens
the journal after `before_rename`/`after_rename` death and proves the state is
either the complete old record or the complete new one, then retries safely.

## Item 4: listener session authority (test seam)

`tools/server/server-rkva-session-authority.{h,cpp}` mints one authority per
accepted managed connection and binds it to:

- the concrete accepted socket (device/inode plus a listener connection
  cookie); a closed, replaced or reused descriptor fails;
- the durable pre-HELLO challenge issued for that connection;
- the exact recursive state mutex; a contended mutex fails closed instead of
  waiting, and the mutex stays held until release/invalidate/destruction so
  the publication cannot race another writer;
- the accepted HELLO ACK session (owner, ledger epoch, auth epoch, transcript
  and current peer challenge).

It is a test seam: there is no listener call site, no capability/operation
advertisement, and `available()` is false unless the activation gate is open.
`tests/test-rkva-session-authority.cpp` covers: default off, non-socket and
disconnected descriptors, zero cookie/challenge and missing mutex, unbound and
mismatched sessions, mutex contention from another thread, failed checks not
leaving the mutex held, descriptor close and reuse, fork, invalidation, and
that no managed negotiation or advertisement exists.

## Files

New in this worktree:

- `tools/server/server-rkva-managed-authority.{h,cpp}`
- `tools/server/server-rkva-host-v3-success.{h,cpp}`
- `tools/server/server-rkva-session-authority.{h,cpp}`
- `tools/server/server-rkva-managed-activation.h`
- `tests/test-rkva-managed-authority.cpp`
- `tests/test-rkva-host-v3-success.cpp`
- `tests/test-rkva-session-authority.cpp`
- `tests/test-rkva-managed-activation-gate.cpp`

Changed:

- `ggml/include/rkva-managed-store-authority.h`: adds the pure
  `rkva_managed_store::delegation` interface. `lease` is unchanged.
- `tools/server/server-rkva-host-journal.{h,cpp}`: delegated constructor,
  `delegated()`, persist-time authority revalidation, the lost-ACK COMMIT
  refusal and `commit_v3_success()`. The standalone path is unchanged.
- `tools/server/server-rkva-managed-orchestrator.{h,cpp}`: optional delegated
  authority; the standalone path is unchanged.
- root `CMakeLists.txt` and `tests/CMakeLists.txt`: the new default-off option,
  library and tests.

## Validation (CPU only)

```sh
cmake -S . -B /tmp/opencode/beellama-activation-all -G Ninja \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
  -DRKVA_MANAGED_SNAPSHOT_PHASE2=ON -DRKVA_MANAGED_SNAPSHOT_PHASE3=ON \
  -DRKVA_MANAGED_SNAPSHOT_PHASE4=ON -DRKVA_MANAGED_SNAPSHOT_ACTIVATION=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/beellama-activation-all -j 10
ctest --test-dir /tmp/opencode/beellama-activation-all -R rkva --output-on-failure
```

Result: 31/31 tests matching `rkva` pass: the 17 Phase 2/3/4 lifecycle tests
(unchanged), the 10 pre-existing RKVA tests and the four new activation tests:

- `test-rkva-managed-authority` (item 6);
- `test-rkva-host-v3-success` (items 1 and 5);
- `test-rkva-session-authority` (item 4);
- `test-rkva-managed-activation-gate` (default-off gate and no advertisement).

Every new translation unit also compiles standalone with
`g++ -std=c++17 -Wall -Wextra -Werror -Wpedantic -Wcast-qual
-Wmissing-declarations -Wextra-semi`.

Default-off check:

```sh
cmake -S . -B /tmp/opencode/beellama-activation-off -G Ninja \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/beellama-activation-off --target llama-server -j 10
```

Result: `llama-server` configures, compiles and links with all four RKVA options
`OFF`; the activation library and its tests are not in the build graph and no
managed capability, op or HELLO field changes.

Not run, by instruction: real Xbox/UWP deployment, power-loss or failed-sync
injection on hardware, CUDA/Vulkan, production service reload, or any live
transport.

## Remaining blockers before activation

These require the real Xbox/UWP hardware or an independently qualified
platform backend. Nothing below can be produced by more CPU tests.

### Item 2: authenticated pin/metadata proof

- No implementation of `rkva_host_journal::v3_success_proof` exists. The real
  one must authenticate the complete canonical host identity, geometry,
  descriptor, rows, generation and layer coverage against an authenticated
  complete HELLO and a verified host RKVK; persist owner, transaction,
  reference, key and metadata digest with a real quota reservation, the nonce
  and a durable pending pin; and hand that exact pin to the v3 object
  writer/verifier.
- The anchored SQLite `apply()` still refuses every managed transition with
  `concrete authenticated pin and metadata proof unavailable`, and
  `rkva_prepare_v3::concrete_guard` (the friend hook for `prepare_pending`) is
  still only a forward declaration. A real PREPARE must create the pending pin
  before a COMMIT can transition it.
- Needed: the engine capture/UWP object writer path
  (`tools/xbox/uwp/rkva/remote-attn-engine.{h,cpp}`,
  `rkva-managed-object-v3-windows.{h,cpp}`), a Windows/AppContainer store
  authority with an exclusive LocalState/ACL contract, and a frozen GPU source
  bound to the exact pin.

### Item 3: qualified durable sync/rename

- No platform backend in this tree proves directory-entry and data durability
  across power loss. The journal checksum, `fsync` ordering, SQLite
  `synchronous=EXTRA` and Linux `flock` are not that proof.
- Needed: per-platform qualified sync/rename (Windows file and directory
  flush/rename semantics, AppContainer paths) plus a real console
  power-interruption test that shows the previous committed pair remains usable,
  pending/quarantined bytes stay charged and no partial object or anchor is
  visible.

### Item 7: peer identity, UWP and console qualification

- Windows/AppContainer provisioning, certificate/package pin and the
  independently authenticated WDP identity (`tools/rkva-peer-identity/` is
  Windows-only and still unbuilt here).
- MSVC/UWP compile of the new code, actual console restart and power
  interruption, package identity/update behavior.
- Independent cryptographic review of the v3 HMAC framing, the receipt MAC
  domain and key provisioning.

### Still not wired

- The real listener transport: nothing mints a session authority outside tests,
  and no dispatch path calls `negotiated()` or `accept_v3_success()`.
- Capability advertisement, GC, release/abort for anchored journals (still
  fail-closed), and the ATTEST success path (the ATTEST codec exists and is
  tested offline; there is no host guard that publishes its evidence).
- A production delivery channel for the shared `v3::secret`; the tests use
  fixed keys.
