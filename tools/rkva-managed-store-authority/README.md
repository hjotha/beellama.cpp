# Managed store authority: Linux-only isolated base

This worktree refactors copies of the SQLite and v3-object prototypes. It is not
wired into a runtime, capability, Xbox, UWP, production store or GC. There is
no completed authenticated native pin, object publication or managed COMMIT.

`rkva_managed_store::lease::try_acquire(parent, error)` returns a concrete RAII
shared token only for an existing private Linux directory owned by the current
UID. It holds a nonblocking exclusive `flock` on a single non-symlink,
single-link `.rkva-managed-store.lock`. It records the owning PID and the
parent/lock inode identities. Each operation holds the token by value, acquires
its recursive in-process operation mutex, and revalidates the directory and
lock path/inodes, owner and permissions. A forked child cannot reuse a parent's
token; `O_CLOEXEC` prevents inheritance across exec. No public manual unlock is
exposed, so an in-progress operation retains the lock until its own token copy
is released. A separately constructed token for another directory is rejected.
The token returns unavailable outside Linux, including Windows/UWP.

The copied SQLite API now requires the concrete token for `provision`,
`challenge`, `inspect` and `apply`; there is no overload without one. A
challenge is bound to the same token instance and cannot be used after release
and reacquisition. The copied object-v3 `write` and `verify` APIs also require
the concrete token and require the v3 root to be a direct child of the same
parent. The callback for pin identity is separate and is not allowed to stand
in for the concrete lock. Callbacks must not reenter the same SQLite connection
while its transaction is active; a future concrete transaction-bound pin proof
must avoid opening a second writer while the first holds SQLite's mutex.

The scope is cooperative Linux processes with trusted private ancestors. A
program that bypasses these APIs and opens SQLite or modifies files directly
can ignore advisory `flock`; permissions do not distinguish processes of the
same UID. Replacing a lock pathname is detected on the next validation, but a
hostile concurrent writer between checks is outside this prototype. Every
real writer/reader must adopt this authority and OS/application isolation
before it can be relied on; do not treat this as authentication or UWP proof.

**Fail-closed pin gate:** authenticated geometry/descriptor/rows/generation/
coverage metadata is not yet durably bound to the SQLite pin, and no concrete
frozen-source/durable object proof has been supplied. Therefore SQLite denies
PREPARE/COMMIT/STATUS/ABORT/RELEASE and object-v3 `write`/`verify` refuse even
with a valid token. A test callback that always returns true is not called and
cannot turn this into a successful pin or object operation. Only the existing
`hello_claim` diagnostic transition is permitted under the concrete token.
There is no STATUS recovery, object COMMIT, RELEASE or GC implementation here.

```sh
cmake -S tools/rkva-managed-store-authority -B /tmp/rkva-managed-store-authority
cmake --build /tmp/rkva-managed-store-authority
ctest --test-dir /tmp/rkva-managed-store-authority --output-on-failure
```

The local test exercises missing/wrong/released tokens, a challenge after
release/reacquire, an inherited token after fork, two independent processes
competing for the same flock, threads sharing a token, lock-path replacement,
SQLite rollback/stale anchor and sentinel preservation. It checks that forged
true-returning proof fixtures are not invoked or accepted. It does not exercise
a successful v3 object, authenticated PREPARE/COMMIT, network protocol,
Windows/UWP, GPU source, power loss or production interoperability.

Phase 4 note: this standalone copy builds only the concrete lease + SQLite +
fail-closed v3 object composition (`test-rkva-managed-store-authority`). The
earlier inline PREPARE-SQLite experiment is **not** built here: its
`RKVA_SQLITE_COMPOSITION_TEST` block targets an `rkva_anchored_sqlite::pending_request`
with a `native_header` member that the Phase 3 publication does not have. Porting
it would require republishing the anchored SQLite contract, which stays
fail-closed until authenticated pin/metadata proof exists. The normal `apply`
path and object writer/verify remain closed.
