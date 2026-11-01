# Managed snapshot Phase 3 PREPARE/COMMIT/receipt core (default off)

This worktree integrates the PREPARE/COMMIT/receipt lifecycle core developed on
the 2026-10-30 research branches into main `bced4c15d` (after the Phase 2 merge).
It is CPU-only library and test code. No server, RPC, capability, dispatch, GC,
GPU, Xbox, UWP or production path is wired, and no existing behaviour changes
while the feature is off.

## Activation

Both gates must be open; both default to off.

- Compile time: CMake option `RKVA_MANAGED_SNAPSHOT_PHASE3` (default `OFF`,
  declared in the root `CMakeLists.txt`). It builds the static library
  `rkva-managed-phase3` and the Phase 3 tests, and defines
  `RKVA_MANAGED_SNAPSHOT_PHASE3` for them.
- Run time: `rkva_managed_phase3::enabled()` in
  `tools/server/server-rkva-managed-phase3.h` returns true only when the build
  is compiled in **and** `RKVA_MANAGED_SNAPSHOT_PHASE3=1` is set. Unset, empty,
  `0`, `true` and any other value return false.

There is deliberately no call site and nothing is advertised: `HELLO` is
unchanged, `supported_ops` does not contain the reserved managed operations and
the reserved protocol constants remain absent from `remote-attn-protocol.h`.
Phase 4 must add the authenticated framing/ATTEST negotiation and the guarded
handler together.

## Contract choice (one publication contract, not a merge)

Phase 2 chose the **registry/journal/orchestrator flat-file contract** as the
only lifecycle contract and kept the object v3 manifest/chunk component as an
isolated, unconditionally fail-closed candidate. Phase 3 keeps that decision
and adds the **anchored SQLite pinned-operation contract** underneath it:

1. **Host journal** (`tools/server/server-rkva-host-journal.*`): phases
   `prepared` → `host_published` → `commit_ack`, release intent/ack and aborted.
   Phase 3 adds a one-time provisioned authenticated anchor, durable command
   nonces and a single atomic verified-COMMIT publication that consumes the
   nonce, advances the anchor and records `commit_ack` in one whole-file
   publication. Once an anchor is provisioned the legacy unauthenticated
   `commit_ack` and all release/abort transitions fail closed.
2. **Anchored SQLite store** (`ggml/include/rkva-anchored-sqlite.h`,
   `ggml/src/ggml-remote-attn/rkva-anchored-sqlite.cpp`): a single-use
   challenge/anchor admission, canonical metadata + identity digest
   validation, conservative quota reservation, durable nonce admission, the
   pending pin, and the MACed receipt/STATUS experiment. It is a storage
   primitive: `prepare_pending` is private to the future authenticated guard.
3. **Managed auth** (`ggml/src/ggml-remote-attn/rkva-managed-auth.h`) and the
   offline reconcile gate (`tools/server/server-rkva-reconcile-gate.h`):
   HMAC-SHA256 command binding plus a read-only host/peer evidence classifier.

The SQLite store and the journal are two durable records of the same
`prepare → host_published → commit` sequence, not two competing object layouts.
Neither one publishes or deletes a v3 object. The object v3 writer/verifier is
**not** called, not linked into the Phase 3 library and remains
`concrete authenticated pin and metadata proof unavailable`.

Known composition gap (unchanged from Phase 2): the orchestrator guards its flat
host/peer stores with `.managed-orchestrator.lock`; the SQLite backend uses the
concrete `rkva_managed_store::lease`; the journal uses its own directory lock.
Phase 3 does not claim these three locks compose. A production integration needs
one authoritative store contract that spans them.

## Ported files

From `feat/rkva-host-anchor-atomic-20261030` `b82c84a9e`:

- `tools/server/server-rkva-host-journal.{h,cpp}` (anchor, nonce and atomic
  verified commit added to the Phase 2 journal)

From `feat/rkva-managed-auth-20261030` / `feat/rkva-registry-reconcile-20261030`
`dc0c40876`:

- `ggml/src/ggml-remote-attn/rkva-managed-auth.h`
- `ggml/src/ggml-remote-attn/rkva-managed-registry-wire.h` (adapted, see below)
- `tools/server/server-rkva-reconcile-gate.h`
- `tests/test-rkva-managed-auth.cpp`
- `tests/test-rkva-reconcile-gate.cpp`

From `research/rkva-commit-receipt-20261030` `dc31b8389`:

- `ggml/include/rkva-anchored-sqlite.h`
- `ggml/src/ggml-remote-attn/rkva-anchored-sqlite.cpp`

New in this worktree:

- `tools/server/server-rkva-managed-phase3.h`: the two-gate activation header.
- `tests/test-rkva-host-journal.cpp`: the preserved Phase 2-era journal test
  (replay, fault injection, process death, corruption, concurrency) extended
  with anchor provisioning, nonce issuance/replay, atomic verified commit,
  crash injection and fail-closed authenticated release/abort.
- `tests/test-rkva-anchored-sqlite-receipt.cpp`: direct PREPARE/COMMIT/receipt
  exercise without the Phase 4 attestation frame.
- `tests/test-rkva-managed-phase3-gate.cpp`: default-off gate semantics.

## Deliberate adaptations

- **Reserved operation numbers stay local.** The Phase 2 report deferred ops
  24–28, `RKVA_CAP_MANAGED_REGISTRY_V1`, `RKVA_CAP_MANAGED_AUTH_V2`,
  `RKVA_PROTO_MANAGED_REGISTRY_V1` and `RKVA_PROTO_MANAGED_AUTH_V2` to Phase 4
  framing. `rkva-managed-registry-wire.h` therefore declares the same numbers in
  a local `rkva_registry_wire::op` namespace and drops `negotiated()`,
  `request_header()` and `response_header()`. `remote-attn-protocol.h` is
  byte-identical to main. Phase 4 must move the constants into the protocol
  header in one change with the negotiation code and delete the local block.
- **Test-only entry points are explicit.** `RKVA_HOST_JOURNAL_TESTING` adds
  `journal::test_commit_verified()` and the persist hook;
  `RKVA_ANCHORED_SQLITE_TESTING` adds `test_prepare`/`test_commit`/
  `test_status_read` and the commit hook. The Phase 3 library is test-only and
  default off; a shipping target must not define either macro.
- **`std::filesystem::u8path` replacement.** `rkva-anchored-sqlite.cpp` keeps
  the same narrow/UTF-8 contract through a small `path_from_utf8()` helper
  because newer standard libraries deprecate `u8path`.
- The older `b097b6a76` object v3 engine, the Windows quota scanner, the v3
  framing/nonce ledger and the authenticated COMMIT reply guard are **not**
  ported.

## Validation (CPU only)

```sh
cmake -S . -B /tmp/opencode/beellama-phase3-on -G Ninja \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
  -DRKVA_MANAGED_SNAPSHOT_PHASE3=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/beellama-phase3-on -j 12
ctest --test-dir /tmp/opencode/beellama-phase3-on \
  -R "test-rkva-(host-journal|managed-auth|reconcile-gate|anchored-sqlite-receipt|managed-phase3-gate)" \
  --output-on-failure
```

Results and the default-off check are recorded in the commit message; every new
translation unit also compiles standalone with
`g++ -std=c++17 -Wall -Wextra -Werror`.

Not run, by instruction: real Xbox/UWP deployment, power-loss/failed-sync
injection on hardware, CUDA/Vulkan, production service reload.

## Still deferred

### Phase 4: framing / ATTEST / identity

- `ggml/src/ggml-remote-attn/rkva-managed-framing-v3.h`
  (`<beellama-host-nonce-durable-20261030>`)
- `ggml/src/ggml-remote-attn/rkva-managed-auth-e2e.h`
- `ggml/src/ggml-remote-attn/rkva-managed-transactional-peer.h`
- `tools/xbox/uwp/rkva/rkva-prepare-v3.h`
- `tools/xbox/uwp/rkva/rkva-resident-attestation.h`
- `tools/server/server-rkva-host-auth.{h,cpp}` (authenticated COMMIT reply
  guard; needs the framing above)
- `tools/server/server-rkva-host-v3-error.{h,cpp}` and the v3 nonce ledger in
  the `beellama-host-nonce-durable-20261030` journal
- The reserved protocol additions (`remote-attn-protocol.h` ops 24–29,
  `RKVA_CAP_MANAGED_REGISTRY_V1`, `RKVA_CAP_MANAGED_AUTH_V2`,
  `RKVA_PROTO_MANAGED_REGISTRY_V1`, `RKVA_PROTO_MANAGED_AUTH_V2`), the
  capability gate and registry request/response framing with authenticated
  receipts
- `tests/test-rkva-prepare-v3.cpp`, `tests/test-rkva-host-auth-atomic.cpp`,
  `tests/test-rkva-host-v3-nonce.cpp`, `tests/test-rkva-transactional-peer.cpp`,
  `tests/test-rkva-managed-store-authority.cpp` (lease + v3 refusal composition;
  its lease and v3 refusal halves are already covered by the Phase 2 tests)
- `docs/rkva-prepare-v3-frame.md`, `docs/rkva-prepare-sqlite-blocker.md`,
  `docs/rkva-receipt-status-experiment.md`, `docs/rkva-windows-pinned-lease.md`,
  `docs/rkva-v3-quota-experiment.md`, `docs/rkva-v3-windows-quota.md`,
  `tools/rkva-managed-store-authority/` (standalone CMake/README)

### Windows / AppContainer only (deferred)

- `tools/xbox/uwp/rkva/rkva-managed-object-v3-windows.{h,cpp}`,
  `ggml/src/ggml-remote-attn/rkva-v3-quota-windows.h`, the UWP vcxproj
  fixtures, `tests/uwp-shared-lease/` and the Windows quota admission path.

## Irreducible blockers before any of this may be enabled

1. **Authenticated pin/metadata proof.** Nothing in the tree proves that a
   pending pin corresponds to a live, authenticated peer session and real
   resident coverage. The SQLite store validates shape and a caller-supplied
   identity digest only; `apply()` still refuses every managed transition with
   `concrete authenticated pin and metadata proof unavailable`.
2. **Qualified durable sync/rename.** No platform backend in this tree proves
   directory-entry and data durability across power loss; the journal checksum
   and SQLite `synchronous=EXTRA` are not that proof, and Linux `flock` covers
   only cooperating writers in a trusted private store.
3. **Nonce/anchor admission from a real transport.** `issue_commit_nonce` and
   the SQLite nonce table are storage; the authenticated frame, live session and
   anchor recovery that must feed them do not exist (Phase 4).
4. **Lost-ACK reconciliation.** A crash between peer commit and host
   `commit_ack` is detected as ambiguous and stays pinned/quarantined. There is
   no automatic recovery, no safe anchor replacement and no STATUS receipt that
   a host may persist yet.
5. **One authority across journal, registry and SQLite.** The three locks are
   independent; a real integration must hold one store authority across the
   snapshot, the gate and the mutation.
6. **UWP and console qualification.** Individual component compilation and
   synthetic filesystem tests do not qualify the target runtime, AppContainer
   identity or Xbox restart/power interruption.

The v3 `need(false, ...)` close must not be removed and the capability/GC must
stay disabled until 1–6 exist.
