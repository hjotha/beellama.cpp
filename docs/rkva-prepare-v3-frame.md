# Experimental PREPARE v3 metadata frame

> Phase 4 status: this standalone frame is **offline-only**. The negotiated
> PREPARE body is `rkva_managed_framing_v3::canonical_prepare` (see
> `docs/rkva-managed-snapshot-phase4.md`). The header is ported because the v3
> framing reuses its HELLO codec; `make()`/`accept()` below require an
> `authenticated_hello` built from the E2E v2 exchange, which is not negotiable
> in this tree and has no call site.

This is an isolated codec, not a negotiated protocol, runtime handler, SQLite
integration, Xbox object proof or GC authorization. It does not modify or
import v1/v2 objects and no capability is advertised. The proposed body is
variable-length but bounded; the caller must set the **actual complete body
length** in the outer RKVA header. There is no compatibility fallback to the
older 144-byte registry request. Final transport framing, capability allocation,
response authentication, durable host journal and lost-ACK recovery are separate
work and have not been implemented here.

The body starts with revision 3 and canonical metadata length, then owner,
transaction, reference and host-prefix key (four 32-byte fields), the complete
84-byte canonical RKVA HELLO, resident session ID and recorded submitted/
completed fence values, the resident attestation MAC, coverage-attestation
digest, E2E key epoch/transcript and fresh 32-byte command nonce. The canonical
attestation metadata follows, limited to `rkva_native_snapshot::kMaxManifestBytes`,
then a 32-byte HMAC-SHA256. The HMAC covers a domain, the complete canonical
outer RKVA request header (including actual length and cycle), and every body
byte before the HMAC. HELLO is encoded field by field little-endian, including
float bits and reserved fields; nonfinite/invalid geometry and nonzero reserved
fields are refused. The attestation metadata includes canonical geometry,
descriptor, rows, generation and per-layer coverage. Its chunk digest fields
are zero metadata placeholders, not GPU-content hashes.

`make()` accepts only a valid E2E HELLO transcript and a matching resident
attestation; it cross-checks the HELLO fields that correspond to the attested
geometry. `accept()` checks exact frame length, expected full HELLO, owner,
epoch/session transcript, nonce, outer HMAC, canonical manifest and nested
attestation MAC before invoking the mandatory `prepare_guard::persist`. Its
`metadata` argument exposes identity, HELLO, geometry/descriptor/rows/
generation, coverage and `identity_digest` for a future SQLite transaction.
The guard must verify the external durable host anchor, independently validate
actual live peer state under a real engine/store lease, validate the available
host RKVK fields, reserve quota, and atomically consume the nonce and persist
that exact metadata and pending pin. A null, failed or unavailable guard never
returns success. No production guard or such SQLite transaction is supplied.

The host and peer share HMAC material in this prototype. A holder of the key
can synthesize both frame and nested attestation; the MAC alone cannot prove
that any peer session or GPU resource really had that coverage. The future
peer must obtain and revalidate the attestation from its live engine under the
same real authority, and must not accept a caller-supplied fixture as proof.
The authenticated session's durable nonce admission and liveness must also be
established by the transport. `accept()` cannot enforce these external facts.

RKVK does **not** contain per-layer coverage; no code here claims otherwise or
compares a host RKVK to coverage. The actual fields RKVK does contain still need
independent validation against the authenticated HELLO and peer state. The
frame does not prove GPU contents, immutable v3 chunks, publication durability,
pin ownership or safe release. No object write, COMMIT, RELEASE or GC is
implemented. If an ACK is lost, the existing anchor/replay protocol still lacks
an integrated recovery path. `identity_digest` covers owner/transaction/reference/key, semantic canonical
HELLO fields and canonical metadata including coverage. It excludes HELLO
negotiation flags, nonce, session, fences and attestation MAC. Thus a reconnect can preserve the exact semantic
identity while the full frame HMAC independently authenticates the complete
HELLO and its new volatile session/fence/proof fields. A changed semantic HELLO
field or canonical metadata changes that identity and must not silently replace a pin.

The isolated test uses synthetic E2E sessions and synthetic resident metadata;
its `prepare_guard` is an in-memory test fixture, not a durable replay guard,
quota reservation or SQLite pin. Tests mutate every frame byte, headers and
sessions, forge an outer MAC over inconsistent geometry/reserved/nested proof,
test bounds, missing/failing guard, exact retry and simultaneous duplicate
nonces. These checks do not qualify a real peer, Xbox, UWP runtime, durable
replay, GPU state or power loss.
