# Durable sync/rename/staging fault injection: what exists, what is missing

Status: **test-only CPU work.** No production target, UWP app, server, service
or console path is changed or exercised by this document. The v3 object writer
and SQLite managed transition remain unconditionally closed.

## 1. Existing durability code on main `4443ba6ee`

| Component | File | Durable operation | Test coverage today |
| --- | --- | --- | --- |
| Host journal | `tools/server/server-rkva-host-journal.cpp:175` (`sync_dir`), `:231` (`fsync` staging file), `:237` rename, `:245` directory sync | staging file `fsync` → rename/link → directory `fsync` | `tests/test-rkva-host-journal.cpp:75-113` injects a thrown failure at every hook stage including `after_rename` and `before_directory_sync`; it does not inject an actual `fsync` `EIO` |
| Managed orchestrator | `tools/server/server-rkva-managed-orchestrator.cpp:50`, `:127`, `:167` | host/peer staging `fsync` + directory sync | phase 2 orchestrator test |
| v3 object writer | `tools/xbox/uwp/rkva/rkva-managed-object-v3.cpp:129-179` | staging dir sync, per-chunk `sync_file`, manifest `sync_file` + `sync_directory`, validated no-replace rename, post-publish verify | **none reachable**: `:92` and `:115` return “concrete authenticated pin and metadata proof unavailable” before the backend is called |
| SQLite | `ggml/src/ggml-remote-attn/rkva-anchored-sqlite.cpp` | `synchronous=EXTRA`, transactions | receipt tests; not a power-loss proof |

The journal and orchestrator are the only reachable durable publication paths.
The object writer's staging/rename body is dead code in every build, including
the test builds, because there is no concrete `pin_proof`.

## 2. New test-only injector (implemented here)

- `tests/rkva-durable-fault.h` — compile-guarded by
  `RKVA_DURABLE_FAULT_TESTING`; refuses to compile without it. Implements
  `rkva_managed_object_v3::durable_filesystem` over a real POSIX directory with
  per-call fault queues for `sync_file`, `sync_directory` and
  `rename_noreplace_and_sync`.
- `tests/test-rkva-durable-fault.cpp` — CPU tests for the injector semantics.
- CMake: the macro is defined only for `test-rkva-durable-fault` inside the
  default-off `RKVA_MANAGED_SNAPSHOT_PHASE2` block. No production target sees
  the header.

Failure modes the injector can produce (and the tests assert):

| Injected mode | Exact behavior | Required caller-visible result |
| --- | --- | --- |
| `eio` in `sync_file` | file is written and readable, `fsync` returns `EIO` | `sync_file=false`; no durability claim |
| `eio` in `sync_directory` | real `fsync(dir)` skipped, returns `EIO` | `sync_directory=false`; caller must not rename |
| `refuse_existing` | destination exists or a competing writer is modelled | `no_change`; source intact; destination unchanged |
| `os_error` | rename fails with a benign error, nothing moved | `no_change`; source intact |
| `ack_lost` | real no-replace rename applied; parent sync skipped | `uncertain`; destination bytes exist; **never** `durable` |
| `crash_after_apply` | same as `ack_lost` plus a crash marker | `uncertain`; crash marker set; destination exists |
| parent sync failure after apply | real rename applied, directory `fsync` fails | `uncertain` |

The tests also cover queue one-shot semantics, call counters, symlink refusal,
unsupported-fault refusal and lease-free operation.

### What this does **not** prove

- It does not execute `rkva_managed_object_v3::write()` or `verify()`.
- It does not prove power-loss durability, filesystem metadata ordering on the
  Xbox, SQLite rollback behavior, journal/SQLite composition, stale-lock
  behavior, or that staging bytes stay charged.
- It does not replace the console rows D3–D7 in
  `docs/rkva-snapshot-console-qualification-matrix.md`.

## 3. Precise test-seam proposal (not applied)

Applying this requires review and is deliberately out of scope for a
qualification-harness change, because it moves the fail-closed gate.

### 3.1 Make the v3 staging body testable without weakening production

Current shape in `tools/xbox/uwp/rkva/rkva-managed-object-v3.cpp`:

- `:102` `result r; bool move_started=false;`
- `:107-113` concrete lease/root/store checks
- `:114` `validate_root` and pin shape
- `:115` `need(false, "concrete authenticated pin and metadata proof unavailable")`
- `:116-179` unreachable pin proof, quota, staging, per-chunk sync, manifest
  sync, rename and post-publish verification
- `:180-181` catch blocks that turn a started move into
  `publication_uncertain`

Proposal (Option A, preferred):

1. Move lines `116-179` into a new file
   `tools/xbox/uwp/rkva/rkva-managed-object-v3-write.inc` defining
   `result write_prepared(...)` with the same arguments and preconditions
   already validated by the caller.
2. Keep `write()` exactly as it is up to and including the `need(false)`.
   `write_prepared` is **not** compiled into `rkva-managed-phase2` or any
   production target.
3. Add `tests/test-rkva-managed-object-v3-staging.cpp` which includes the
   `.inc`, defines `RKVA_MANAGED_OBJECT_V3_TESTING`, supplies a test-only
   `pin_proof`/`quota_lease`/`source`/`durable_filesystem` and exercises every
   fault row in §4. The test target links the copied `.inc` translation unit,
   not the production `write()`.
4. Add a static test that asserts
   `rkva-managed-object-v3-write.inc` is referenced by the test CMake target
   only and that `rkva-managed-object-v3.cpp` still contains the
   `need(false, "concrete authenticated pin and metadata proof unavailable")`
   line in `write()` and `verify()`.

Option B (rejected): add a test-only branch inside `write()` under
`RKVA_MANAGED_OBJECT_V3_TESTING`. It is smaller, but it puts reachable
publication code behind a macro in the production translation unit and makes
an accidental define a publication bypass. Do not do this.

### 3.2 Journal true-`fsync` injection (optional, smaller)

`tests/test-rkva-host-journal.cpp` already simulates failures by throwing from
the `RKVA_HOST_JOURNAL_TESTING` hooks. It cannot prove the code handles a real
`fsync`/`rename` error return. A minimal addition under the same test macro:

- `server-rkva-host-journal.cpp`: add
  `int (*test_io_hook)(const char *, int, int) = nullptr;` beside
  `test_persist_hook`, and call it from `sync_dir` (`:175`) and the staging
  `fsync` (`:231`) and `renameat` (`:237`). The hook receives the stage name,
  the real result and `errno`, and may replace the result.
- `tests/test-rkva-host-journal.cpp`: add cases that force `EIO` at
  `before_file_sync`, `before_rename` and `before_directory_sync`, then assert:
  before rename → old journal byte-identical, temp unlinked; after rename →
  reopen reports the uncertain publication and the new revision is visible;
  after directory sync → success is still not claimed when the hook forces
  failure at the final `sync_dir`. Verify `RKHJNL01/02/03` decode and no record
  is lost.
- Keep the default builds free of the symbol; the existing test target already
  defines `RKVA_HOST_JOURNAL_TESTING`.

### 3.3 Staging/durability console seam (needed for D3–D7)

The app-side probe must expose a test-only staging run over
`rkva_managed_object_v3::write_prepared` with a fault selector from
`LocalState\rkva-qual\request-<hex>.json`:

- `fault`: one of `none`, `chunk_sync`, `manifest_sync`, `dir_sync`,
  `rename_eexist`, `rename_eio`, `crash_before_rename`,
  `crash_after_rename`, `stale_lock`.
- Evidence: exact staging/object inventory before and after, pending pin and
  charge state (from the host store, not a boolean), nonce consumption state,
  and the writer's `result.state` (`rejected`, `staging_preserved`,
  `publication_uncertain`, `published_pending`).
- A `published_pending` result is not a managed COMMIT and grants no GC.

Until this exists, the console matrix rows D3–D7 stay `BLOCKED`; the header
and its tests in this change are CPU contract evidence only.

## 4. Required assertions for the staging seam (when implemented)

For every fault, the test must prove all of:

1. the previously committed pair is byte-identical and still readable;
2. no v2/v1 object was imported, rewritten or removed;
3. staging/pending bytes remain present and charged;
4. the consumed nonce/attempt record is never reused;
5. `uncertain` is never reported as `durable`/`published`;
6. a second writer cannot enter while the lease is held and cannot adopt the
   stale stage after a crash;
7. recovery is explicit: no automatic cleanup, GC or anchor replacement.

## 5. Non-claims

- No hardware power-loss test was run.
- No Windows/UWP build was run.
- The injector is not a durability backend and must never be shipped.
- The v3 `need(false, ...)` close is unchanged in this change and must stay
  until blockers 1–7 are closed.
