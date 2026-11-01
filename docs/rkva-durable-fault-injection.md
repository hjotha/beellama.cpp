# Durable sync/rename/staging fault injection and backend skeleton

Status: **test-only CPU work.** No production target, UWP app, server, service
or console path is changed or exercised by this document. The v3 object writer
and SQLite managed transition remain unconditionally closed. The Windows/UWP
backend is a guarded skeleton that is not executed on this host.

Base: main `3a02e90a2`. Work: branch `feat/snapshot-durable-20261101`.

## 1. Existing durability code on main

| Component | File | Durable operation | Test coverage today |
| --- | --- | --- | --- |
| Host journal | `tools/server/server-rkva-host-journal.cpp:175` (`sync_dir`), `:231` (`fsync` staging file), `:237` rename, `:245` directory sync | staging file `fsync` → rename/link → directory `fsync` | `tests/test-rkva-host-journal.cpp:75-113` injects a thrown failure at every hook stage including `after_rename` and `before_directory_sync`; it does not inject an actual `fsync` `EIO` |
| Managed orchestrator | `tools/server/server-rkva-managed-orchestrator.cpp:50`, `:127`, `:167` | host/peer staging `fsync` + directory sync | phase 2 orchestrator test |
| v3 object writer | `tools/xbox/uwp/rkva/rkva-managed-object-v3.cpp` | staging dir sync, per-chunk `sync_file`, manifest `sync_file` + `sync_directory`, validated no-replace rename, post-publish verify | production `write()`/`verify()` still refuse before any backend call; the test-only staging body is now covered by `test-rkva-managed-object-v3-staging` |
| SQLite | `ggml/src/ggml-remote-attn/rkva-anchored-sqlite.cpp` | `synchronous=EXTRA`, transactions | receipt tests; not a power-loss proof |

The journal and orchestrator are the only reachable durable publication paths.
The object writer's staging/rename body is dead code in every production build,
including production-facing test builds, because there is no concrete
`pin_proof`.

## 2. Implemented test seam

### 2.1 Files

- `tests/rkva-durable-fault.h` — compile-guarded by
  `RKVA_DURABLE_FAULT_TESTING`; refuses to compile without it. Implements
  `rkva_managed_object_v3::durable_filesystem` over a real POSIX directory with
  per-call fault queues for `sync_file`, `sync_directory` and
  `rename_noreplace_and_sync`.
- `tools/xbox/uwp/rkva/rkva-managed-object-v3-internal.h` — the canonical
  manifest/path/lease helpers, shared by the production fail-closed writer and
  the test seam so the two cannot drift.
- `tools/xbox/uwp/rkva/rkva-managed-object-v3-write.inc` — test-only staging
  body `write_prepared(...)`. It revalidates the concrete lease, root, namespace
  and pin, then executes the same staging/manifest/rename sequence as `write()`
  minus the authenticated-pin refusal. It is compiled **only** by
  `tests/test-rkva-managed-object-v3-staging.cpp`; it is not part of
  `rkva-managed-phase2`, `llama-server`, the UWP app or any Phase 2/3/4
  library.
- `tests/test-rkva-managed-object-v3-staging.cpp` — CPU tests for every fault
  path through `write_prepared`.
- `tests/test-rkva-durable-fault.cpp` — injector-only queue/symlink semantics.
- `tools/xbox/uwp/rkva/rkva-managed-object-v3-windows.{h,cpp}` — guarded
  Windows/UWP backend skeleton (see §3).
- `tests/test-rkva-managed-object-v3-platform.cpp` — Linux fail-closed checks
  for the Windows skeleton, static source checks, and a POSIX reference backend
  with the same conservative contract.

CMake defines `RKVA_DURABLE_FAULT_TESTING` and
`RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST` only for those test targets inside
the default-off `RKVA_MANAGED_SNAPSHOT_PHASE2` block. No production target sees
either macro or the `.inc`.

### 2.2 Failure modes the seam can produce (and the tests assert)

| Injected mode | Exact behavior | Required caller-visible result |
| --- | --- | --- |
| `eio` in `sync_file` | file is written and readable, `fsync` reports `EIO` | `sync_file=false`; writer state `staging_preserved`; no rename |
| `eio` in `sync_directory` | real `fsync(dir)` skipped, returns `EIO` | `sync_directory=false`; writer state `staging_preserved` when pre-rename |
| `refuse_existing` | destination exists or a competing writer is modelled | `no_change`; source intact; destination unchanged |
| `os_error` | rename fails with a benign error, nothing moved | `no_change`; source intact |
| `rename_eio` | rename reports `EIO`; applied state is unknown | `uncertain`; never `durable`; stage preserved in the deterministic model |
| `ack_lost` | real no-replace rename applied; parent sync skipped | `uncertain`; destination bytes exist; **never** `durable`/`published` |
| `crash_after_apply` | same as `ack_lost` plus a crash marker | `uncertain`; crash marker set; destination bytes exist |
| parent sync failure after apply | real rename applied, directory `fsync` fails | `uncertain` |

The staging tests also assert:

- the previously committed pair is byte-identical before and after every fault;
- `rkva-snapshots-v1`/`v2` sentinels are untouched;
- staging bytes remain present for pre-rename faults and the applied bytes
  remain present for uncertain renames;
- a retry with the same transaction refuses the stale stage, including after
  the lease is released and reacquired;
- a destination/source symlink is never followed;
- fault queues are one-shot and independent per operation kind;
- an unsupported fault kind is refused, not ignored;
- `write()` still returns `rejected` with
  `concrete authenticated pin and metadata proof unavailable` and never calls
  the proof/quota/source/backend callbacks.

The seam's happy path intentionally reaches `published_pending` **only in the
test binary**, to prove the body's result mapping. That state is not a managed
COMMIT and grants no registry or GC authority. The production `write()` remains
fail-closed.

### 2.3 What this does **not** prove

- It does not execute the production `write()`/`verify()` success path.
- It does not prove power-loss durability, filesystem metadata ordering on the
  Xbox, UWP/AppContainer behavior, SQLite rollback behavior, journal/SQLite
  composition, stale-lock behavior on real hardware, or that staging bytes stay
  charged under a real quota.
- It does not replace the console rows D3–D7 in
  `docs/rkva-snapshot-console-qualification-matrix.md`.
- The injector and the POSIX reference backend are not durability backends and
  must never be shipped.

## 3. Windows/UWP durable backend skeleton

`tools/xbox/uwp/rkva/rkva-managed-object-v3-windows.{h,cpp}` is compiled only in
`test-rkva-managed-object-v3-platform` with
`RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST`. The Win32 body is additionally
inside `#ifdef _WIN32`; a Linux build compiles the fail-closed stub.

What it does when compiled for Windows:

- opens the already-provisioned absolute `rkva-snapshots-v3` root with
  `CreateFile2`, `FILE_FLAG_OPEN_REPARSE_POINT` and `FILE_FLAG_BACKUP_SEMANTICS`;
- rejects legacy v1/v2 path components and reparse-point roots;
- `sync_file`: accepts only `tmp-<64 hex>/manifest-v2.bin` or
  `tmp-<64 hex>/n-<index>.kv`, verifies a regular single-link file, then calls
  `FlushFileBuffers`;
- `sync_directory`: flushes the root handle or an exact staged directory handle;
- `rename_noreplace_and_sync`: verifies the staged directory, calls
  `MoveFileExW` with no replace/copy flags, maps `ERROR_ALREADY_EXISTS`/
  `ERROR_FILE_EXISTS` to `no_change`, then flushes the root handle.

Exact semantics and limits:

- A successful file flush does not flush the parent directory entry.
- A successful root-handle flush after `MoveFileExW` is the strongest
  observation available here, but Microsoft documents `FlushFileBuffers` for
  files and does not establish that a UWP/AppContainer directory-rename +
  flush survives power loss. `MoveFileExW` also documents copy-and-delete
  behavior, not a same-volume rename durability guarantee.
- Therefore `rename_noreplace_and_sync` returns `uncertain` even after an
  observed move and parent flush, and `qualified()` is **false in every build,
  including the experimental one**. In production `write()` fails before the
  backend; the backend cannot enable publication by itself.
- Not proven: AppContainer execution, Xbox deployment, actual power-loss
  ordering, ACL/ownership, filesystem-specific rename semantics, stale-lock
  recovery, quota charging and authenticated pin binding.

The Linux test executes the same contract against a POSIX reference backend
that returns `uncertain` after a successful rename + parent `fsync` for the same
reason. It also statically checks that the Windows source still contains
`FlushFileBuffers`, `MoveFileExW`, `FILE_FLAG_OPEN_REPARSE_POINT`, no
`MOVEFILE_REPLACE_EXISTING`/`MOVEFILE_COPY_ALLOWED`, and the conservative
`uncertain` result. This is semantic parity plus compile/static evidence, not a
Windows build or console run.

## 4. Console probe contract for D3/D5/D7

The controller is `tools/xbox/scripts/xbox-snapshot-console-qual.py`. It now
validates `rkva-console-durability-v1` documents and drives the D3/D4/D5 fault
cases, the D6 power-loss timeline and the D7 stale-lock flow. The app-side
action is still absent, so an execute run records `BLOCKED`, never `PASS`.

### 4.1 Request

`LocalState\rkva-qual\request-<probe-id>.json`:

```json
{
  "schema": "rkva-console-durability-v1",
  "probe_id": "<32 lowercase hex>",
  "case": "chunk_sync",
  "phase": "run",
  "fault": {"kind": "chunk_sync", "injected": true, "at": 2},
  "test_only": true,
  "store": {
    "root": "rkva-qual\\durable\\rkva-snapshots-v3",
    "namespace": "rkva-snapshots-v3",
    "production_namespace": false
  },
  "phase_flags": {"phase3": false, "phase4": false}
}
```

`case`/`fault.kind` are one of `chunk_sync`, `manifest_sync`, `dir_sync`,
`rename_existing`, `rename_eio`, `ack_lost`, `crash_after_apply`. D7 uses
`case: "stale_lock"` with `phase: "hold"` and then `phase: "verify"` plus the
previous probe id in `predecessor`. The store root must be under
`rkva-qual\durable\`; the probe must never open `LocalState\rkva-snapshots-v3`.
The probe must not enable Phase 3/4 publication gates.

### 4.2 Response

`LocalState\rkva-qual\response-<probe-id>.json`, plus the required artifacts
`durability-fault-<case>.json`, `staging-inventory.json`, and for D7
`stale-lock.json`:

```json
{
  "schema": "rkva-console-durability-v1",
  "probe_id": "<same id>",
  "case": "chunk_sync",
  "build": {
    "commit": "<7-40 hex>",
    "test_only_probe": true,
    "durable_fault_seam": true,
    "production_publication_enabled": false,
    "windows_backend_qualified": false,
    "phase3_enabled": false,
    "phase4_enabled": false
  },
  "store": {
    "root": "rkva-qual\\durable\\rkva-snapshots-v3",
    "namespace": "rkva-snapshots-v3",
    "production_namespace": false
  },
  "fault": {"kind": "chunk_sync", "injected": true, "at": 2},
  "writer": {
    "state": "staging_preserved",
    "object_present": false,
    "error": "v3 chunk durable sync failed: injected file sync failure"
  },
  "before": {"committed_sha256": "<64 hex>", "inventory": ["manifest-v2.bin", "n-0.kv"]},
  "after":  {"committed_sha256": "<same 64 hex>", "inventory": ["manifest-v2.bin", "n-0.kv"]},
  "legacy": {"touched": false, "v1_unchanged": true, "v2_unchanged": true},
  "staging": {
    "present": true, "charged": true, "adoptable": false,
    "automatic_cleanup": false
  },
  "pin": {"consumed": true, "reusable": false},
  "claims": {"durable": false, "published": false, "committed": false, "gc_authorized": false},
  "cleanup": {"automatic": false},
  "lock": {
    "held_by_predecessor": true,
    "owner_evidence_required": true,
    "reclaimable_with_owner_evidence": true,
    "pid_reuse_accepted": false,
    "second_writer_entered": false
  }
}
```

`lock` is required for D7. The hold response uses `"held": true`; the verify
artifact uses `"held_by_predecessor": true` (or `held_by_probe`). The validator
requires one of those. `staging.present` is false for `ack_lost` and
`crash_after_apply` because the applied rename moved the directory, while
`writer.object_present` must then be true. The validator refuses any
`published_pending`/durable/committed claim, a changed committed digest, a
legacy-namespace change, an uncharged stage, an adoptable stage, a reusable
nonce, or a qualified Windows backend claim.

`staging-inventory.json`:

```json
{
  "schema": "rkva-console-staging-inventory-v1",
  "probe_id": "<same id>",
  "root": "rkva-qual\\durable\\rkva-snapshots-v3",
  "entries": [{"name": "tmp-<64 hex>", "bytes": 4096}]
}
```

The controller fetches and validates the response, the per-case artifact and
the inventory. A missing artifact is `INCOMPLETE`, an unknown probe action is
`BLOCKED`, and only a fully validated D3–D7 set can become `PASS`.

## 5. Validation (CPU only)

```sh
cmake -S . -B /tmp/opencode/beellama-snapshot-durable -G Ninja \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DRKVA_MANAGED_SNAPSHOT_PHASE2=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/beellama-snapshot-durable -j 8 --target \
  test-rkva-managed-object-v3-staging test-rkva-managed-object-v3-platform \
  test-rkva-durable-fault test-rkva-managed-object-v3-phase2 \
  test-rkva-v3-quota test-rkva-managed-orchestrator \
  test-rkva-managed-store-lease test-rkva-managed-phase2-gate
ctest --test-dir /tmp/opencode/beellama-snapshot-durable \
  -R 'test-rkva-(managed-object-v3|durable-fault|v3-quota|managed-orchestrator|managed-store-lease|managed-phase2-gate)|test-xbox-snapshot-console-qual' \
  --output-on-failure
```

No hardware, Xbox, production service, UWP build, CUDA/Vulkan run or real
power loss is involved.

## 6. Non-claims

- No Windows/UWP build or execution was performed here.
- No hardware power-loss test was run.
- No console qualification row is PASS.
- The injector, POSIX reference backend and Windows skeleton are not qualified
  durability backends and must never be shipped as production implementations.
- The v3 `need(false, ...)` close is unchanged and must stay until all blockers
  are closed.
