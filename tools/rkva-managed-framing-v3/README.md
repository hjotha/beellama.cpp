# RKVA managed v3: single authenticated framing prototype

This is an **isolated codec and bounded stream-parser prototype**, not a
listener, host transport, production peer, object store, registry, COMMIT or
GC implementation. No runtime advertises or dispatches the reserved V3 bit.
The API is `ggml/src/ggml-remote-attn/rkva-managed-framing-v3.h` and depends
on the isolated auth/registry wire helpers and the native PREPARE metadata
helpers. Do not integrate the earlier experimental form that appended an
anchor, binding or state wrapper outside `payload_bytes`.

Phase 4 status: this repository copy is integrated behind the default-off
`RKVA_MANAGED_SNAPSHOT_PHASE4` gate; see `docs/rkva-managed-snapshot-phase4.md`
for the contract choice, the pre-HELLO challenge (`op 30`), ATTEST (`op 31`),
the durable v3 nonce ledger and the remaining blockers. Nothing here is
advertised by the default listener, and the successful COMMIT receipt guard is
still abstract/unimplemented.

Every request or response consists of its complete existing RKVA header
(request 32 bytes, response 56 bytes), followed by **one** V3 payload. The
header's `payload_bytes` equals the exact payload size including the final
32-byte HMAC. The HMAC covers the canonical little-endian outer header and
all payload bytes except the tag itself. Requests and responses use distinct
domain strings. No sidecar/suffix is sent beyond that advertised size.
Unknown revision, type, reserved bits, operation, status or length fail closed.
The V3 capability, V2 auth capability, registry capability and all required
ops must all be present; missing bits never trigger an unauthenticated fallback.

All payloads start with this 384-byte prefix, followed by a bounded type-specific
payload and a 32-byte HMAC:

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 4 | revision = 3 |
| 4 | 4 | kind: offer 1, hello ACK 2, command 3, reply 4, challenge 5, attest 6, attest reply 7 |
| 8 | 4 | reserved = 0 |
| 12 | 4 | type-specific payload length |
| 16 | 32 | owner |
| 48 | 8 | authentication key epoch, nonzero |
| 56 | 32 | SQLite/ledger epoch, nonzero |
| 88 | 32 | authenticated session transcript, zero for HELLO |
| 120 | 32 | fresh request nonce/challenge; reply echoes command nonce |
| 152 | 32 | SHA-256 binding to the complete request frame, zero in requests |
| 184 | 168 | host anchor in commands; zero in HELLO/replies |
| 352 | 32 | command binding HMAC; zero for other frame types |

Authentication epoch u64 and ledger epoch 32 bytes are **distinct**; no
implicit padding or conversion establishes their relationship. Provisioning
and the peer guard must explicitly authenticate the allowed owner/epoch pair.
The anchor layout is owner32, ledger_epoch32, peer challenge32, accepted
version u64, accepted digest32 and host anchor MAC32. The challenge must equal
the current session challenge. The command binding HMAC covers the outer
header and full command prefix+payload with the binding field zeroed; the
outer HMAC covers the resulting binding too. Neither MAC alone proves that the
host anchor was independently persisted or that the peer owns the claimed GPU
state; the external `command_guard::admit` must prove those facts, authorize,
consume the challenge/nonce and perform its durable transaction atomically.

The pre-HELLO challenge (`op 30`, cycle 0) is a prefix-only response frame with
no type-specific payload. It must originate from the peer's durable challenge
source and binds owner, authentication epoch and ledger epoch. HELLO offer and
ACK start with a 100-byte payload: the full canonical 84-byte RKVA HELLO,
selected capability bits u64, operation mask u32 and server version u32; the
ACK appends the committed anchor version u64 and digest32 (140 bytes total). Owner, both epochs and fresh challenge are in the prefix. The signed ACK
binds a SHA-256 of the complete offer including its header and tag. The session
transcript hashes both complete headers and frames. The client must use a
cryptographically secure challenge and durable nonce guard; the server must
supply its own fresh unpredictable challenge and replay guard. `make_hello_ack`
is only valid after `accept_offer` was admitted; caller enforces ordering.

ATTEST (`op 31`) carries a 48-byte request (session id u32, rows u32,
generation u64, key32) and a 204-byte-plus-metadata response (request echo,
submitted/completed fences, attestation MAC32, coverage digest32, state
version u64, state digest32, next challenge32, metadata length u32 and
canonical resident metadata). It is authenticated by the same outer frame and
must be admitted by a peer guard that captures the real engine and advances
the anchored state exactly once.

Every command payload begins with the 144-byte canonical registry request.
PREPARE appends an 80-byte host RKVK header, canonical HELLO84, peer session ID
u32, submitted/completed fence u64 each, attestation MAC32, coverage digest32,
metadata length u32 and canonical manifest metadata (at most
`rkva_native_snapshot::kMaxManifestBytes`). The fixed prefix is 396 bytes.
The codec validates RKVK framing and its actual origin/capacity/heads/rows/
generation/tail/iterations against the canonical metadata and HELLO. **RKVK
contains no resident coverage**: the peer must independently capture and
validate coverage and fence completion under the real engine/store lock.
The semantic identity digest excludes volatile flags, session, fences,
nonce, transcript and attestation MAC; it includes request identity, RKVK,
semantic HELLO and canonical metadata. This is not a proof of GPU bytes.

Successful ACK payload is canonical registry response152 + state digest32 +
receipt length u32 + next peer challenge32 + optional receipt384. The
registry response supplies the state version; committed/released states
require a receipt. The receipt is owner32, ledger epoch32, tx/ref/key32 each,
semantic identity digest32, **object/manifest digest32**, commit nonce32,
predecessor version u64 and digest32, successor version u64 and digest32,
commit version u64, result u32, reserved u32 zero and receipt MAC32. Object
digest is independently verified under the object lease and must not be
assumed equal to the host-prefix key. A mandatory `receipt_guard::validate`
checks external durable anchor/receipt MAC, version succession, object proof
and authorization before a successful ACK is exposed. The client nonce guard
then atomically consumes the previously issued request nonce. On accepted ACK,
peer and host session advance to the authenticated next challenge. A signed
negative response contains a zero fixed ACK body, returns `peer_error` and
invalidates the session; it is **not evidence that a commit is absent**. If an
ACK is lost, reconnect with a fresh HELLO/challenge and query/retry the same
transaction with a fresh nonce and external durable reconciliation.

`stream_decoder` accepts fragmented input and bounds allocation using the
outer header before reading its payload. It reports exactly how many bytes it
consumed so concatenated frames remain distinct; EOF before `finish()` is an
error. It parses framing only: callers must invoke the corresponding HMAC and
guard validation before exposing or acting on any fields. `request()` and
`response()` decode header fields explicitly little-endian.

## Blocking production dependencies

There is no real CSPRNG/key provisioning/rotation, durable nonce guard,
authenticated epoch mapping, SQLite transaction/anchor recovery, real resident
engine/fence attestation, GPU/object proof, actual runtime transport, UWP
qualification or power-loss qualification in this worktree. Guards used by
unit tests are in-memory fixtures and cannot authorize production. HMAC provides
authentication but not confidentiality; key storage, authenticated peer
identity, authorization, replay ledger anti-rollback and independent crypto
review remain required. A valid local frame, receipt or manifest is not a peer
lease or permission to delete any object. Keep capability disabled.

Isolated CPU tests:

```sh
cmake -S tools/rkva-managed-framing-v3 -B /tmp/rkva-managed-framing-v3
cmake --build /tmp/rkva-managed-framing-v3 -j 4
ctest --test-dir /tmp/rkva-managed-framing-v3 --output-on-failure
```

They use synthetic metadata/guard fixtures and exercise exact header lengths,
per-byte tampering, outer-header mutation, nested-binding forgery even with a
recomputed outer tag, receipt-proof refusal, no downgrade, replay, authenticated
errors, lost ACK and reconnect, truncated/concatenated parser input, and
fragmented socketpair and TCP loopback transfers. Process-local tests do not
prove durable replay, production peer correctness or Xbox behavior.
