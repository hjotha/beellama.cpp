# Peer identity / AppContainer / UWP inventory (blocker 7)

Status: **store and pin implemented as a default-off library; inventory and
qualification plan still open.** The portable store, pin/TOFU/rotate/revoke
API, Windows/AppContainer backend source, pinned WDP helper and CPU/static
tests now exist in this tree behind `RKVA_PEER_IDENTITY_STORE` (default `OFF`)
and the `RKVA_PEER_IDENTITY_STORE=1` run-time gate. **No Windows/MSVC/UWP build
was run, no identity was provisioned, no package was deployed and no console
state changed.** The Windows path is uncompiled and the AppContainer/ACL,
restart and power-loss rows remain BLOCKED. See
`docs/rkva-peer-identity-store.md` for the implemented contract and the exact
remaining work.

## 1. What exists, and where

| Piece | Location | Reachable on main? | Platform |
| --- | --- | --- | --- |
| Identity store and pin (`verify`, `enroll_tofu`, `provision`, `rotate`, `revoke`, `wipe`) | `tools/rkva-peer-identity/` in this change, default `RKVA_PEER_IDENTITY_STORE=OFF` | **No production call site** | Portable core + Windows/AppContainer backend; LocalState needs `RKVA_PEER_IDENTITY_UWP` + C++/WinRT `Windows::Storage::ApplicationData` |
| Identity bundle format | `rkva-peer-identity-core.cpp`: canonical JSON, 16 top-level fields, HMAC-SHA256 with a DPAPI-protected key | Default-off library only | Portable parser, Windows I/O |
| Pinned WDP evidence helper | `tools/xbox/scripts/rkva-peer-identity-wdp.py` | Yes (read-only script) | Linux host; pin compared before credentials |
| Protocol allocation (ops 24–31, cap bits) | `ggml/src/ggml-remote-attn/remote-attn-protocol.h` | Yes (inert constants) | Portable |
| Managed auth HMAC vocabulary | `ggml/src/ggml-remote-attn/rkva-managed-auth.h` | Yes (Phase 3/4 library only) | Portable |
| v3 HELLO/ATTEST framing | `ggml/src/ggml-remote-attn/rkva-managed-framing-v3.h`, `rkva-managed-attest-v3.h` | Yes (Phase 4, default off, no call site) | Portable |
| Resident attestation encoder | `tools/xbox/uwp/rkva/rkva-resident-attestation.h` | Yes (Phase 4 library only) | Portable |
| Explicit TOFU prototype | `tools/rkva-managed-bootstrap/tofu.py` + `test_tofu.py` on `research/rkva-live-session-guard-20261030` | **No** | Linux prototype, not wired |
| WDP helpers | `tools/xbox/scripts/deploy.sh`, `DeployClient` in `tools/xbox/scripts/rkva-quality-gate.py` | Yes | Linux host; `curl -k`, no certificate pin |
| App autopilot | `tools/xbox/uwp/MainPage.cpp` (`ApParseScript`/`ApRun`), `include/xllama/autopilot.h` | Yes | UWP, Dev Mode; fixed op allowlist, no RKVA probe op |

## 2. Missing pieces (each is a hard prerequisite, not a TODO)

### 2.1 Identity store

- The store and pin API now exist behind a default-off CMake option and are
  covered by CPU/static tests, but **no CMake/MSBuild UWP target includes
  them**. There is no Windows test binary and no UWP compilation evidence.
- `open_local_state()` returns `"LocalState requires the UWP AppContainer build
  (RKVA_PEER_IDENTITY_UWP)"` outside a UWP build. The POSIX test backend only
  mirrors the write sequence; it does not qualify AppContainer behavior.
- The record no longer carries `auth_secret` or any plaintext MAC key. Its
  integrity tag is HMAC-SHA256 keyed by a DPAPI-protected random key stored in
  the same record. That detects corruption and off-user/off-machine copies, not
  a same-user author. `provision()` still requires an authorized WDP session
  label and an out-of-band certificate digest; the MAC is not authenticity.
- The peer claim is caller-supplied by the future accepted connection. No
  listener in this tree produces it from a live authenticated peer yet.

### 2.2 Certificate pin

- `tools/xbox/scripts/rkva-peer-identity-wdp.py` compares the leaf SHA-256
  against the out-of-band pin immediately after the TLS handshake and before
  any HTTP request or Basic credential. It is read-only and records package,
  certificate and namespace evidence.
- `deploy.sh` still authenticates with HTTP Basic over
  `https://${XBOX_IP}:11443` and `curl -k`: TLS verification is disabled and no
  leaf certificate digest is pinned. `DeployClient` inherits this.
- The app does not yet compare the live WDP leaf with the store record at
  connection time; the pinned helper compares the exported public fields only,
  and the store compares the peer claim supplied by the future transport.
- `curl` supports `--pinnedpubkey sha256//<base64>`; no `deploy.sh` subcommand
  uses it.
- The controller's `--expected-cert-sha256` compares the live leaf digest with
  an operator-supplied pin, but the pin cannot be assumed trustworthy until it
  is obtained out of band (console UI / physical label), not from the
  connection it is validating.

### 2.3 TOFU boundary

- The store now implements first-use TOFU behind an explicit operator
  confirmation and an out-of-band digest, and refuses TOFU once any record
  exists. Rotation and revocation are explicit and revocation is monotonic.
  The console operator flow and the app-side provisioning action are still
  absent.
- `tofu.py` pins the first presented certificate when
  `explicit_tofu=True`. It is a prototype with a local `identity.json`, no
  upload, no WDP and no managed capability.
- First-use pinning is acceptable only as an explicit, operator-confirmed
  bootstrap for a **new** host identity, and never as managed trust after an
  epoch change, rollback or certificate replacement.
- The store now has the out-of-band digest requirement, operator confirmation,
  durable host-side record and refusal policy. The app-side action that invokes
  it on the console, the operator workflow and the restart/power-loss evidence
  are still absent.
- `tofu.py` is not on main and must not be copied into a production path as-is.

### 2.4 AppContainer / UWP

- Windows source now refuses reparse points, takes an exclusive `LockFileEx`,
  writes a pending marker then a staging file, renames, flushes the directory
  and applies a protected DACL with only the user/package SIDs. This code is
  **uncompiled and unqualified**; there is no `icacls` evidence.
- LocalState access relies on the package identity and AppContainer ACL; the
  ACL is set by the new backend but has not been read back on a console. `read()`
  requires exactly `identity.lock` + `identity.json`, no reparse points, and an
  exclusive `LockFileEx` on `identity.lock`.
- `write()` creates `identity.pending`, writes and flushes `identity.new`,
  renames it over `identity.json`, flushes the directory, deletes the marker and
  re-inspects the namespace. The crash window is not tested on Windows; a
  surviving `identity.pending`/`identity.new` makes the store ambiguous and it
  must be refused, never adopted.
- There is no UWP `vcxproj` entry, no AppContainer compile, and no console
  runtime test. A desktop MSVC compile would not qualify the AppContainer
  path. The exact integration and commands are in
  `docs/rkva-peer-identity-store.md` §7.1.
- ACL evidence is not exposed by WDP. It needs a test-only app action that
  reads the security descriptor (or a Windows host test on a copied tree), and
  `icacls` output recorded as evidence.

### 2.5 Package identity

- The PFN contains a publisher hash derived from the signing certificate.
  `deploy.sh pfn` returns the highest registered version and only warns when
  several are live; that is an implicit selection and is forbidden for
  destructive runs.
- The pinned WDP helper checks the exact PFN, package family and optional
  publisher, and compares an exported record's certificate digest with the
  pinned leaf. It does not validate the signing certificate chain, and no code
  compares a live managed session with the store yet.
- Updating must preserve the existing `GianlucaMazza.xllama` package identity;
  uninstalling destroys LocalState (models and sessions).

### 2.6 Console runtime

- No console restart/power-interruption test with an identity store exists.
- No test proves that an old LocalState identity can be restored (rollback)
  and is rejected rather than silently accepted.
- No test proves secrets do not reach `xllama.log`, crash dumps or WDP
  listings.

## 3. Concrete build/test plan

### Step 1 — Windows desktop compile and unit tests (no console)

The portable tests exist as `test-rkva-peer-identity` (Linux CPU) and the
Windows backend has a test entry point, but no MSVC test has been built. The
intended Windows-only coverage is:

```text
Windows test target (not built yet)
  - provision/load round-trip through open_test_directory(temp_dir)
  - wrong PFN/publisher/certificate/host/port → refused
  - tampered JSON / non-canonical JSON / extra key / truncated → refused
  - identity.pending or identity.new present → refused
  - hardlink/symlink/reparse-point substitutes → refused
  - crash hook after pending/staging/rename → refuses until explicit wipe
  - ACL read-back and DPAPI-sealed key behavior on the test tree
```

The CMake test now covers the portable equivalents on Linux. A Windows build
must still run the same cases plus ACL/DPAPI tests with MSVC in an x64 developer
prompt; record the compiler version and the exact `git rev-parse HEAD`.

### Step 2 — UWP AppContainer compile

- Not done. Add `rkva-peer-identity-core.cpp` and
  `rkva-peer-identity-win32.cpp` to `tools/xbox/uwp/BeeLlama.props` (or a
  dedicated test-only vcxproj) with `RKVA_PEER_IDENTITY_STORE`,
  `RKVA_PEER_IDENTITY_UWP`, C++/WinRT and the `advapi32/crypt32/bcrypt`
  libraries.
- Build the existing package pipeline:

```powershell
pwsh -File scripts/build-xbox.ps1 -Configuration Release -Backend llamacpp `
  -WindowsSdkVersion 10.0.26100.0 -BuildRevision <n> -MaxParallel 10
```

- A compile is not runtime qualification. Record `dumpbin /dependents` and the
  packaged manifest; do not deploy unless an explicit deployment session is
  authorized.

### Step 3 — Certificate pin and TOFU boundary

- The pinned WDP helper is implemented; it has not been run against a console.
  Obtain the WDP leaf SHA-256 out of band from the console UI. Record it in
  `identity-inventory.json` and compare against `--expected-cert-sha256`.
- Negative test: run `rkva-peer-identity-wdp.py` with a deliberately wrong pin
  and require connection refusal before any credential is sent.
- Positive test: the same helper succeeds only with the out-of-band pin and
  records package/certificate/namespace evidence.
- Decide and document the TOFU boundary: explicit first-use confirmation is
  allowed only for a new host identity; after an accepted identity, a changed
  certificate is a hard refusal until the operator provisions a new bundle
  and records the new pin.

### Step 4 — Provisioning on the console

- Still blocked: there is no app-side test action that calls
  `open_local_state()`/`provision()`. Add one behind the existing
  `XLLAMA_QUAL_PROBE` pattern, then deploy the test build (explicit session,
  new revision).
- Use a test-only autopilot action `rkva_qual_identity` to:
  1. write the identity bundle into `LocalState\rkva-peer-identity-v3` under
     the AppContainer;
  2. return the namespace listing, the security-descriptor summary (owner,
     DACL entries, no inherited write for `Users`/`Everyone`), and the
     provision result;
  3. never return the protected key, MAC, certificate bytes or credentials —
     only presence flags and one-way public digests.
- Cross-check the provisioned `cert_sha256` with the out-of-band pin and the
  package publisher; a mismatch is a failure.
- Restart the app, reload the identity, and record that the namespace is
  exactly `identity.lock` + `identity.json`.
- Restore an older LocalState copy and require a hard refusal (rollback test).
- Power-interrupt during provisioning; on restart, a surviving
  `identity.pending` or an incomplete namespace must refuse, not repair
  silently.

### Step 5 — Console evidence

The controller's `identity` scenario collects I1–I3 today (certificate digest,
PFN/publisher, namespace inventory). I4–I6 require the Windows/UWP work above
and must be recorded as `BLOCKED` until then.

## 4. Acceptance criteria (do not weaken)

- A certificate pin is accepted only from an out-of-band source; a pin
  observed over the same connection is not evidence.
- TOFU is never used for managed trust after the first provisioning; a
  certificate change is a refusal and requires explicit re-provisioning.
- The identity record is accepted only from an authorized provisioning
  session; its protected-key MAC is corruption detection, not authenticity.
- The AppContainer namespace is exact, exclusive and free of reparse points;
  a pending marker or partial namespace is ambiguous and refused.
- No secret appears in logs, evidence JSON or WDP listings.
- A compile, a desktop test or a local filesystem test does not qualify the
  console runtime or power-loss behavior.

## 5. Explicit non-claims

- Peer identity is implemented as a **default-off library** and is not enabled
  or wired into any production path.
- No Windows or UWP build was produced by this change.
- No AppContainer path, ACL, DPAPI behavior or console runtime was exercised.
- No certificate pin, TOFU bootstrap, provisioning, rotation or revocation was
  executed against a console.
- No restart or power-loss test exists.
- Blocker 7 stays open until Steps 1–5 produce real artifacts and the console
  rows I4–I6 pass.
