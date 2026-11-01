# Managed snapshot Phase 2 lifecycle core (default off)

This worktree integrates the isolated managed snapshot lifecycle core developed
on the 2026-10-30 research branches into main `51dd994ef`. It is CPU-only
library and test code. No server, RPC, capability, GC, GPU, Xbox, UWP or
production path is wired, and no existing behaviour changes while the feature is
off.

## Activation

Both gates must be open; both default to off.

- Compile time: CMake option `RKVA_MANAGED_SNAPSHOT_PHASE2` (default `OFF`,
  declared in the root `CMakeLists.txt`). It builds the static library
  `rkva-managed-phase2` and the Phase 2 tests, and defines
  `RKVA_MANAGED_SNAPSHOT_PHASE2` for them.
- Run time: `rkva_managed_phase2::enabled()` in
  `tools/server/server-rkva-managed-phase2.h` returns true only when the build
  is compiled in **and** `RKVA_MANAGED_SNAPSHOT_PHASE2=1` is set. Unset, empty,
  `0`, `true` and any other value return false.

There is deliberately no call site: Phase 2 ships the state machine, not a
request handler. Phase 3 must add the guarded PREPARE/COMMIT/receipt path and
advertise a capability; nothing is advertised here.

## One lifecycle contract (not a merge)

Two object layouts existed in the source branches:

1. **Registry/journal/orchestrator contract** (`66e71838b`):
   lower-case 64-hex owner/transaction/reference/key identifiers, object key =
   SHA-256 of the complete host file, flat host file `<tx>.managed`, peer
   object `<key>.blob`, peer registry phases pending/committed/aborted/released,
   host journal phases prepared/host_published/commit_ack/release_intent/
   release_ack/aborted.
2. **Object v3 contract** (`b097b6a76` / `a4c3ddc8a`):
   `rkva-snapshots-v3/<host-state-sha256>/` directories with a canonical
   `manifest-v2.bin` and `n-*.kv` chunks, publication state
   `published_pending` that is explicitly **not** COMMIT, and an external
   authority/pin/quota/durable-filesystem contract.

Phase 2 chooses **contract 1 as the only lifecycle contract**. The orchestrator
is the composition point; the v3 component is not called by it and cannot
publish into its state machine.

The v3 component is ported from `a4c3ddc8a` because that revision is
lease-bound and **unconditionally fail-closed**: after validating the concrete
`rkva_managed_store::lease`, root binding and pin shape, both `write()` and
`verify()` refuse with "concrete authenticated pin and metadata proof
unavailable". The older `b097b6a76` variant, whose abstract `authority`
callback let a test fixture authorize real staging/publication, is **not**
ported. No fixture can authorize v3 publication in this tree.

Merging the two layouts would require a concrete authenticated pending-pin and
metadata proof that the integration blockers document
(`beellama-managed-dispatch-audit-20261030/docs/rkva-v3-proof-integration-blockers.md`)
shows does not exist. Phase 2 therefore keeps the registry/journal lifecycle and
the v3 object candidate separate instead of inventing that proof.

Known composition gap: the orchestrator still guards its flat host/peer stores
with its own `.managed-orchestrator.lock`; it does not take the concrete
`rkva_managed_store::lease`, and the lease currently guards only the v3/quota
stack. Unifying them requires one authoritative store contract and a concrete
pin proof, which is Phase 3 work. Do not assume the two locks compose.

## Ported files

From `feat/rkva-native-key-inspect-20261030` `66e71838b`:

- `ggml/include/rkva-managed-registry.h`
- `ggml/src/ggml-remote-attn/rkva-managed-registry.cpp`
- `tools/server/server-rkva-host-journal.{h,cpp}`
- `tools/server/server-rkva-managed-orchestrator.{h,cpp}`
- `tests/test-rkva-native-orchestrator-preflight.cpp` (built, not registered:
  it consumes two externally qualified snapshot files)

From `feat/rkva-v3-windows-quota-20261030` `a4c3ddc8a`:

- `ggml/include/rkva-managed-store-authority.h`
- `ggml/src/ggml-remote-attn/rkva-managed-store-authority.cpp`
- `tools/xbox/uwp/rkva/rkva-managed-object-v3.{h,cpp}` (lease-bound,
  fail-closed)

New in this worktree:

- `ggml/src/ggml-remote-attn/rkva-v3-quota.h`: POSIX extraction of the
  conservative v3 accounting scan previously embedded in
  `rkva-anchored-sqlite.cpp` (`scan_v3_bytes`). Same charging, name, manifest,
  chunk, symlink, hardlink and overflow rules; no SQLite transaction or
  admission policy. The Windows `FileStandardInfo` scanner is not extracted.
- `tools/server/server-rkva-managed-phase2.h`: the two-gate activation header.
- `tests/test-rkva-managed-orchestrator.cpp`: synthetic end-to-end lifecycle
  (publish, idempotent replay, missing peer object, source digest mismatch,
  release, tombstone, legacy isolation).
- `tests/test-rkva-managed-store-lease.cpp`: Linux lease privacy, process
  exclusion, ownership identity, mutex serialization, lock replacement and
  reacquisition.
- `tests/test-rkva-managed-object-v3-phase2.cpp`: lease/root binding and
  unconditional v3 refusal.
- `tests/test-rkva-v3-quota.cpp`: charging, orphan stage preservation,
  unknown-name/symlink/corrupt-chunk rejection and legacy isolation.
- `tests/test-rkva-managed-phase2-gate.cpp`: default-off gate semantics.

## Classification of the source-branch files

| Class | Files | Phase 2 action |
| --- | --- | --- |
| Authority / lease | `rkva-managed-store-authority.{h,cpp}` | ported |
| Object v3 / quota | `rkva-managed-object-v3.{h,cpp}`, POSIX quota scan | ported; v3 fail-closed |
| Host journal / orchestrator | `server-rkva-host-journal.{h,cpp}`, `server-rkva-managed-orchestrator.{h,cpp}` | ported |
| Managed registry | `rkva-managed-registry.{h,cpp}` | ported |
| Test-only / fixture-only | `tests/test-rkva-managed-*`, `tests/uwp-shared-lease/*`, `artifacts/*` | tests ported/rewritten; Windows fixture and logs left on the source branches |
| Windows / AppContainer only | `rkva-managed-object-v3-windows.{h,cpp}`, `rkva-v3-quota-windows.h`, `tools/rkva-managed-object-v3/windows-*.vcxproj` | deferred |

## Validation (CPU only)

```sh
cmake -S . -B /tmp/opencode/beellama-phase2-on -G Ninja \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
  -DRKVA_MANAGED_SNAPSHOT_PHASE2=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/beellama-phase2-on \
  --target test-rkva-managed-orchestrator test-rkva-managed-store-lease \
  test-rkva-managed-object-v3-phase2 test-rkva-v3-quota \
  test-rkva-managed-phase2-gate test-rkva-native-orchestrator-preflight -j 8
ctest --test-dir /tmp/opencode/beellama-phase2-on \
  -R "test-rkva-(managed|v3-quota)" --output-on-failure
```

Result: 5/5 passed (orchestrator, lease, object-v3 refusal, quota, gate).
`test-rkva-native-orchestrator-preflight` builds and links but is not registered
because it requires two externally qualified Qwen35 snapshots that are not in
the tree.

Every new translation unit was also compiled standalone with
`g++ -std=c++17 -Wall -Wextra -Werror` (all passed).

Default-off check: a second configure without
`-DRKVA_MANAGED_SNAPSHOT_PHASE2` leaves
`RKVA_MANAGED_SNAPSHOT_PHASE2:BOOL=OFF`, defines no Phase 2/ninja targets and
builds `llama-server` successfully. No server, RPC, scheduler, protocol or
tool source was modified; the only tracked edits are the root CMake option and
the appended tests block.

Not run, by instruction: real Xbox/UWP deployment, power-loss/failed-sync
injection on hardware, CUDA/Vulkan, production service reload.

## Still deferred

### Phase 3: PREPARE / COMMIT / receipt / durable pin

Superseded by `docs/rkva-managed-snapshot-phase3.md` (branch
`feat/snapshot-phase3-20261101`), which ports the host journal
anchor/nonce/verified-commit, the anchored SQLite pending-pin/receipt store,
the managed auth contract and the offline reconcile gate behind
`RKVA_MANAGED_SNAPSHOT_PHASE3` (default off). The list below is kept as the
original scoping record.

- `ggml/include/rkva-anchored-sqlite.h`
- `ggml/src/ggml-remote-attn/rkva-anchored-sqlite.cpp` (SQLite pin/quota/
  metadata schema, anchor MAC, nonce admission, `prepare_pending`)
- `tools/xbox/uwp/rkva/rkva-prepare-v3.h` (PREPARE metadata frame and guard)
- `ggml/src/ggml-remote-attn/rkva-managed-auth.h`
- `ggml/src/ggml-remote-attn/rkva-managed-auth-e2e.h`
- `tests/test-rkva-prepare-v3.cpp`
- `tests/test-rkva-managed-store-authority.cpp` (source version needs SQLite)
- `docs/rkva-prepare-v3-frame.md`
- `docs/rkva-prepare-sqlite-blocker.md`
- `docs/rkva-windows-pinned-lease.md`
- `docs/rkva-v3-quota-experiment.md`
- `docs/rkva-v3-windows-quota.md`
- `tools/rkva-managed-store-authority/` (standalone CMake/README and
  `SHARED-WINDOWS-LEASE.md`)
- Windows quota admission (`rkva-v3-quota-windows.h`) and its SQLite coupling

Superseded and not ported: the older `b097b6a76` object v3 engine plus
`tests/test-rkva-managed-object-v3*.cpp`,
`tools/rkva-managed-object-v3/{README.md,CMakeLists.txt,windows-*.vcxproj}`
and `windows-uwp-link-fixture.cpp`. Keeping both v3 authority models would
create two conflicting publication contracts; this tree keeps only the
lease-bound fail-closed one.

Required before enabling: a concrete authenticated pin/metadata proof, a
qualified platform durable-sync/rename backend, PREPARE nonce/anchor admission,
COMMIT that returns the record/version/digest from the same transaction, and
lost-ACK reconciliation. The v3 `need(false, ...)` close must not be removed
until those exist.

### Phase 4: framing / ATTEST

- `ggml/src/ggml-remote-attn/rkva-managed-registry-wire.h`
- `tools/xbox/uwp/rkva/rkva-resident-attestation.h`
- The reserved protocol additions in
  `ggml/src/ggml-remote-attn/remote-attn-protocol.h`: ops 24-29,
  `RKVA_CAP_MANAGED_REGISTRY_V1`, `RKVA_CAP_MANAGED_AUTH_V2`,
  `RKVA_PROTO_MANAGED_REGISTRY_V1`, `RKVA_PROTO_MANAGED_AUTH_V2`
- Registry request/response framing, response authentication, status receipts
  and the negotiated capability gate

### Windows / AppContainer only (deferred)

- `tools/xbox/uwp/rkva/rkva-managed-object-v3-windows.{h,cpp}`
- `ggml/src/ggml-remote-attn/rkva-v3-quota-windows.h`
- `tools/rkva-managed-object-v3/windows-uwp-compile.vcxproj`,
  `windows-uwp-link.vcxproj`, `windows-uwp-link-fixture.cpp`
- `tests/uwp-shared-lease/` (desktop/AppContainer fixture and `build.cmd`)
- `artifacts/*` validation logs (ignored, preserved on the source branches)
