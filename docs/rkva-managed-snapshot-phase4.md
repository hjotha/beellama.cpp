# Managed snapshot Phase 4 framing/ATTEST/identity layer (default off)

This worktree integrates the authenticated framing, ATTEST, resident-attestation
and durable v3 host nonce ledger developed on the 2026-10-30/31 research
branches into main `543241c2f` (after the Phase 3 merge). It is CPU-only
library and test code. No listener, RPC, capability, HELLO, dispatch, GC, GPU,
Xbox, UWP or production path is wired, and no existing behaviour changes while
the feature is off.

## Activation

Both gates must be open; both default to off.

- Compile time: CMake option `RKVA_MANAGED_SNAPSHOT_PHASE4` (default `OFF`,
  declared in the root `CMakeLists.txt`). It builds the static library
  `rkva-managed-phase4` and the Phase 4 tests, and defines
  `RKVA_MANAGED_SNAPSHOT_PHASE4` for them.
- Run time: `rkva_managed_phase4::enabled()` in
  `tools/server/server-rkva-managed-phase4.h` returns true only when the build
  is compiled in **and** `RKVA_MANAGED_SNAPSHOT_PHASE4=1` is set. Unset, empty,
  `0`, `true`, `01` and any other value return false.
  `accepts_negotiated()` additionally requires the negotiated capability.

There is deliberately no call site and nothing is advertised. The default
`llama-server`/`server-context` source lists are unchanged, `HELLO` is
unchanged, `supported_ops` does not contain ops 24–31 and no default build
compiles the Phase 4 library. The reserved constants added to
`remote-attn-protocol.h` are inert enum/bit values; `rkva_managed_auth_e2e` and
`rkva_managed_framing_v3` are only reachable from the Phase 4 target.

## Contract choice (one framing contract, not a merge)

Two incompatible authenticated framing shapes exist on the source branches for
the same reserved capability bits and op numbers:

1. **E2E v2 two-frame** (`feat/rkva-managed-auth-e2e-20261030`,
   `rkva-managed-auth-e2e.h`): a 132-byte HELLO offer, a 196-byte HELLO reply,
   then separate 240-byte commands and 184-byte replies. It signs flags,
   capabilities and server version but **not** an anchor, the complete host
   RKVK header or canonical metadata. `rkva-managed-transactional-peer.h`
   wraps the SQLite state outside `payload_bytes`.
2. **v3 single frame** (`research/rkva-live-session-guard-20261030`,
   `rkva-managed-framing-v3.h`): one 384-byte-prefixed, HMAC-sealed payload per
   complete RKVA header, a pre-HELLO challenge (op 30), an offer that carries
   the durable host anchor, a HELLO ACK that carries the committed anchor
   version/digest, commands bound to the anchor, and ATTEST (op 31).

Both allocate the same caps (`REGISTRY_V1|AUTH_V2`) and use op 29 for HELLO
with different payloads, so they cannot both be negotiated. **Phase 4 chooses
contract 2 (v3 single frame) as the only negotiated managed framing.** It
supersedes contract 1 for the reasons the source itself records
(`docs/rkva-v3-proof-integration-blockers.md` on
`research/rkva-managed-dispatch-audit-20261030`): the E2E v2 HELLO does not
bind the anchor, RKVK header or canonical metadata, and the transactional state
wrapper sits outside `payload_bytes`. Merging them would let two peers agree on
`AUTH_V2` and then disagree about op 29's body.

Consequences, deliberately accepted:

- The E2E v2 codec header is still ported because the v3 code and the resident
  attestation encoder share its vocabulary (`secret`, `nonce`, `digest`,
  `session`, `purpose`, `nonce_guard`, header serializers, `mac`) and its
  offline `authenticated_hello::valid_hello` check. Its offer/command/reply
  functions are **offline-only**: nothing in this tree negotiates a connection
  through them, and `rkva_managed_auth_e2e::negotiated()` is not the Phase 4
  gate. The gate is `rkva_registry_wire::negotiated()`, which requires
  `PROTO/CAP_REGISTRY_V1 | AUTH_V2 | FRAMING_V3` **and** every op 24–31.
- `rkva-managed-transactional-peer.h`, `server-rkva-host-auth.{h,cpp}` (the
  authenticated successful-COMMIT guard for the E2E v2 shape) and their tests
  are **not ported**; they are frozen on their source branches. Besides the
  contract conflict, the transactional adapter is uncompilable against the
  published `rkva_anchored_sqlite::apply` contract (7-argument call without the
  concrete lease and with the pre-Phase-2 `object_proof` signature).

## Ported files

From `research/rkva-live-session-guard-20261030` `73cd53ed0`:

- `ggml/src/ggml-remote-attn/rkva-managed-auth-e2e.h` (shared vocabulary;
  offline-only handshake)
- `ggml/src/ggml-remote-attn/rkva-managed-framing-v3.h`
- `ggml/src/ggml-remote-attn/rkva-managed-attest-v3.h`
- `tools/xbox/uwp/rkva/rkva-prepare-v3.h`
- `tools/xbox/uwp/rkva/rkva-resident-attestation.h`
- `tests/test-rkva-managed-framing-v3.cpp`
- `tests/test-rkva-prepare-v3.cpp` (non-SQLite half only, see below)
- `tests/test-rkva-resident-attestation.cpp`
- `docs/rkva-prepare-v3-frame.md`
- `tools/rkva-managed-framing-v3/`

From `research/rkva-host-nonce-durable-20261030` `297585a69`:

- the durable v3 request ledger in `tools/server/server-rkva-host-journal.{h,cpp}`
  (format `RKHJNL03`, `configure_v3_epoch`, `issue_v3_command`,
  `consume_v3_error`, `v3_connection_authority`, `issued_v3_nonce`)
- `tools/server/server-rkva-host-v3-error.{h,cpp}` (adapted to the v3
  `reply_ack_guard`)

From `research/rkva-peer-identity-20261030` `e2526b34d` (protocol allocation
only; the store itself is deferred):

- `RKVA_REGISTRY_*` ops 24–28, `RKVA_MANAGED_AUTH_HELLO` 29,
  `RKVA_MANAGED_AUTH_CHALLENGE` 30, `RKVA_MANAGED_ATTEST` 31,
  `RKVA_CAP_MANAGED_{REGISTRY_V1,AUTH_V2,FRAMING_V3}` and the matching
  `RKVA_PROTO_*` bits in `remote-attn-protocol.h`

From `research/rkva-managed-framing-v3-20261030` `3ab2bc653` and
`research/rkva-resident-attestation-20261030` `1eb06ad60`:

- no additional files; their framing/resident-attestation content is an earlier
  revision of the live-session-guard files above (the live-session-guard copy is
  the superset and is the one ported)

From `research/rkva-commit-receipt-20261030` `dc31b8389`:

- nothing new; Phase 3 already ported the anchored SQLite receipt core

New in this worktree:

- `tools/server/server-rkva-managed-phase4.h`: the two-gate activation header.
- `ggml/src/ggml-remote-attn/rkva-managed-registry-wire.h`: the local op block is
  deleted, ops alias the protocol constants and `negotiated()` becomes the single
  capability gate.
- `tests/test-rkva-managed-phase4-gate.cpp`: default-off gate semantics.
- `tests/test-rkva-managed-phase4-negotiation.cpp`: exact bit/op requirements
  and no-fallback negotiation.
- `tools/rkva-managed-store-authority/`: standalone harness for the concrete
  lease + SQLite + fail-closed v3 object composition.

## Deliberate adaptations

- **v3 error guard moved to `reply_ack_guard`.** The source branch implemented
  `nonce_guard::consume()` against the earlier framing (which called
  `accept_reply(..., receipt_guard*)`). The selected v3 framing hands the guard
  the authenticated acknowledgement and the exact request/reply frames through
  `reply_ack_guard::commit(...)`, so the guard now rejects success (`ok`) and
  commits the negative-ACK nonce consumption through `consume_v3_error`. This
  is an interface adaptation, not a semantic change: only authenticated
  negative ACKs are supported.
- **One negotiation predicate.** `rkva_managed_framing_v3::negotiated()` delegates
  to `rkva_registry_wire::negotiated()`. The required set is exactly
  `REGISTRY_V1|AUTH_V2|FRAMING_V3` plus ops 24–28, 29, 30 and 31; any missing bit
  or op fails closed, and there is no unauthenticated fallback.
- **Journal ledger is additive.** Phase 3's anchored journal and the source
  branch's v3 journal were otherwise byte-identical; the port adds the v3
  records and refuses the v2 commit-nonce path once `configure_v3_epoch()`
  succeeds. Format magic is `RKHJNL03` only when v3 is configured, so existing
  `RKHJNL01/02` journals still decode. The same translation unit is compiled by
  the Phase 3 and Phase 4 libraries; without `configure_v3_epoch()` the v3
  fields and methods stay inert and the Phase 3 file format is unchanged.
- **PREPARE v3 has two encodings and only one is reachable.**
  `rkva-prepare-v3.h` is ported because the v3 framing uses its HELLO codec
  (`put_hello`/`get_hello`/`valid_hello`). The standalone PREPARE frame in the
  same header is the earlier E2E-v2 experiment: its `accept()` requires an
  `authenticated_hello` built from the offline E2E v2 exchange, so it has no
  call site. The negotiated PREPARE body is `framing_v3::canonical_prepare`
  (wire request + RKVK80 + HELLO84 + fences + attestation/coverage MACs +
  canonical metadata). Its test is registered as an offline unit test.
- **SQLite composition half stays out.** `test-rkva-prepare-v3.cpp`'s
  `RKVA_SQLITE_COMPOSITION_TEST` block needs
  `rkva_anchored_sqlite::pending_request::native_header`, which the Phase 3
  publication does not have. Building it would republish the anchored SQLite
  contract, so the block is not enabled and the standalone harness omits it.
- **Peer identity is not built.** `tools/rkva-peer-identity/` is a Windows /
  AppContainer implementation: every real operation is inside `#ifdef _WIN32`,
  and on Linux `provision_local_state`/`load_local_state` return false with
  "AppContainer LocalState unavailable". There is nothing CPU-testable to port
  without inventing a Linux backend, so only its protocol allocation is taken.

## Validation (CPU only)

```sh
cmake -S . -B /tmp/opencode/beellama-phase4-on -G Ninja \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
  -DRKVA_MANAGED_SNAPSHOT_PHASE4=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/beellama-phase4-on -j 6
ctest --test-dir /tmp/opencode/beellama-phase4-on -R "rkva" --output-on-failure
```

Result: 27/27 `rkva`-matching tests passed: 10 Phase 2/3 targets (including
`test-rkva-host-journal` and `test-rkva-anchored-sqlite-receipt` unchanged), 7
Phase 4 targets and the 10 pre-existing RKVA tests.

Phase 4 CTests:

- `test-rkva-managed-framing-v3`: challenge/offer/HELLO ACK/ATTEST/command/
  reply byte tampering, header mutation, nested-binding forgery with a
  recomputed outer tag, no-downgrade negotiation, replay, lost ACK + reconnect,
  truncated/concatenated parser input, socketpair and TCP loopback streams.
- `test-rkva-prepare-v3`: PREPARE v3 metadata frame, exact retries, duplicate
  nonce concurrency, reconnect with a stable identity, forged nested proof.
- `test-rkva-resident-attestation`: metadata MAC over geometry/descriptor/
  coverage/fences/generation/session and refusal of invalid coverage.
- `test-rkva-host-v3-nonce`: durable v3 issue under a connection authority,
  CSPRNG nonce, per-byte frame-hash binding, signed-error consumption, crash
  before/after rename, two-process issuance race, timeout keeps the nonce
  pending, corrupt/missing journal refusal, cross-version nonce reuse.
- `test-rkva-managed-phase4-gate`: compiled-in, off by default, opt-in only `1`.
- `test-rkva-managed-phase4-negotiation`: exact managed bit/op set, single-bit
  removal fails, no fallback to `wire::operations`.
- `test-rkva-managed-store-authority`: concrete lease + SQLite + fail-closed v3
  object, stale anchor, lock replacement, fork/exec exclusion, sentinel
  preservation (Linux only).

Default-off check:

```sh
cmake -S . -B /tmp/opencode/beellama-phase4-off -G Ninja \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/beellama-phase4-off --target llama-server -j 6
```

Result: `llama-server` configured, compiled and linked with every RKVA option
off; the Phase 4 library and tests are not in the build graph.

Every new translation unit also compiles standalone with
`g++ -std=c++17 -Wall -Wextra -Werror`; the standalone framing and store
harnesses pass their own CTest.

Not run, by instruction: real Xbox/UWP deployment, power-loss or failed-sync
injection on hardware, CUDA/Vulkan, production service reload, or any live
transport.

## Deferred (precise list)

### E2E v2 / transactional peer / host success guard (contract 1)

- `ggml/src/ggml-remote-attn/rkva-managed-transactional-peer.h` and
  `tests/test-rkva-transactional-peer.cpp` (only on
  `research/rkva-managed-dispatch-audit-20261030`).
- `tools/server/server-rkva-host-auth.{h,cpp}` and
  `tests/test-rkva-host-auth-atomic.cpp` (source: `297585a69`).
- `tests/test-rkva-managed-auth-e2e.cpp`, `tests/test-rkva-managed-auth.cpp`
  (E2E v2 codec tests) — Phase 3 already ports `rkva-managed-auth` and its test.
- Reason: conflicting HELLO/reply wire shape, and the adapter no longer
  compiles against `rkva_anchored_sqlite::apply` (concrete lease + current
  `object_proof`).
- The framing-v3 `hello_ack_guard`, `command_guard`, `reply_ack_guard` and
  ATTEST `peer_guard`/`host_guard` remain abstract; only the negative-ACK host
  guard has a concrete durable implementation.

### Peer identity

- `tools/rkva-peer-identity/rkva-peer-identity.{h,cpp}` (Windows/AppContainer
  only; needs an MSVC UWP build plus an exclusive LocalState/ACL contract and an
  independently authenticated WDP certificate/package pin).
- `tools/rkva-managed-bootstrap/tofu.py` and its local TLS test
  (explicit-TOFU host prototype; not wired, not run here).

### Real transport / backend

- `rkva-managed-host-v3.{h,cpp}`, `rkva-managed-host-transport.{h,cpp}`,
  `rkva-managed-host-backend.h`, the `ggml-remote-attn.cpp` backend adapter
  changes and `tests/test-rkva-managed-backend-*.cpp`
  (`feat/rkva-host-v3-durable-guard-20261030` `9b6cf301b`). Production
  admission is disabled by a test macro in that branch and the journal it uses
  is a test seam; porting it would touch the live RKVA backend.
- `tools/xbox/uwp/rkva/remote-attn-engine.{h,cpp}`,
  `remote-attn.cpp` and `rkva-managed-object-v3-windows.{h,cpp}` changes
  (engine capture / UWP object writer) — no host build can validate them.
- `tools/rkva-managed-host-transport/`, `tests/uwp-live-guard/`, MSVC vcxproj
  fixtures.

### Storage composition

- `test-rkva-prepare-v3.cpp`'s `RKVA_SQLITE_COMPOSITION_TEST` block and the
  newer anchored-sqlite `pending_request`/`test_prepare` revision it needs.
- `docs/rkva-prepare-sqlite-blocker.md`, `docs/rkva-receipt-status-experiment.md`,
  `docs/rkva-windows-pinned-lease.md`, `docs/rkva-v3-quota-experiment.md`,
  `docs/rkva-v3-windows-quota.md`.

## Irreducible blockers before any of this may be enabled

These are unchanged by Phase 4 unless noted.

1. **Authenticated successful COMMIT/ATTEST reply.** Phase 4 adds the signed
   negative-ACK path and the durable request ledger, but there is still no
   concrete `reply_ack_guard` that verifies a COMMIT receipt, the object proof
   and the successor anchor and then atomically publishes `commit_ack`. A
   successful reply remains `unavailable`.
2. **Authenticated pin/metadata proof.** Nothing proves that a pending pin
   corresponds to a live authenticated peer session and real resident coverage.
   The SQLite store validates shape and a caller-supplied identity digest only;
   the v3 object writer/verifier stays fail-closed.
3. **Qualified durable sync/rename.** No platform backend in this tree proves
   directory-entry and data durability across power loss. The journal checksum,
   `fsync` ordering and SQLite `synchronous=EXTRA` are not that proof.
4. **Real transport and live session authority.** No listener owns a session
   minted from an accepted connection, challenge source and accepted HELLO.
   `v3_connection_authority` has no production implementation; the source
   transport branch is test-only and would modify the live RKVA backend.
5. **Lost-ACK reconciliation.** A crash between peer commit and host ACK stays
   ambiguous: the peer challenge advances and the host nonce remains issued.
   There is no authenticated receipt/recovery or safe anchor replacement.
6. **One authority across journal, registry, SQLite and the v3 object.** The
   four locks are independent; the composition test shows they do not compose
   into one transaction.
7. **Peer identity and UWP/console qualification.** No Windows/AppContainer
   provisioning, certificate pin or console restart/power-interruption test.

The v3 `need(false, ...)` close must not be removed and the capability/GC must
stay disabled until 1–7 exist.

## What activation would require

1. A concrete listener-side session authority bound to the accepted connection,
   the durable challenge source and the exact state mutex; no handle reuse.
2. Concrete v3 guards: `hello_guard`/`hello_ack_guard` over the anchored SQLite
   anchor and journal; `command_guard` over the SQLite pending pin and quota;
   `peer_guard`/`host_guard` over a real engine capture and fence; and a
   `reply_ack_guard` that publishes COMMIT receipts and the successor anchor in
   one durable journal/SQLite transaction.
3. One store authority spanning registry, journal, SQLite and the v3 object,
   with qualified durable sync/rename and explicit lost-ACK reconciliation.
4. Advertise the three managed caps and ops 24–31 only after the above, then
   enable the run-time gate; until then `negotiated()` must never be called
   from a dispatch path.
5. MSVC/UWP compile and real console restart/power-loss qualification, plus
   independent cryptographic review of the HMAC framing and key provisioning.
