# Peer identity / AppContainer / UWP inventory (blocker 7)

Status: **inventory and plan only.** No Windows build was run, no identity was
provisioned, no package was deployed and no console state changed. The files
described below are **not on main `4443ba6ee`** and nothing in this tree claims
that peer identity works.

## 1. What exists, and where

| Piece | Location | Reachable on main? | Platform |
| --- | --- | --- | --- |
| Identity store (`provision_local_state`, `load_local_state`, `provision_test_path`, `load_test_path`) | `tools/rkva-peer-identity/rkva-peer-identity.{h,cpp}` on `research/rkva-peer-identity-20261030` (`e2526b34d`) | **No** | Windows only; every real path is under `#ifdef _WIN32`, and LocalState needs `RKVA_PEER_IDENTITY_UWP` + C++/WinRT `Windows::Storage::ApplicationData` |
| Identity bundle format | same file: canonical JSON, 11 keys, self-HMAC with `auth_secret` | No | Portable parser, Windows I/O |
| Protocol allocation (ops 24–31, cap bits) | `ggml/src/ggml-remote-attn/remote-attn-protocol.h` | Yes (inert constants) | Portable |
| Managed auth HMAC vocabulary | `ggml/src/ggml-remote-attn/rkva-managed-auth.h` | Yes (Phase 3/4 library only) | Portable |
| v3 HELLO/ATTEST framing | `ggml/src/ggml-remote-attn/rkva-managed-framing-v3.h`, `rkva-managed-attest-v3.h` | Yes (Phase 4, default off, no call site) | Portable |
| Resident attestation encoder | `tools/xbox/uwp/rkva/rkva-resident-attestation.h` | Yes (Phase 4 library only) | Portable |
| Explicit TOFU prototype | `tools/rkva-managed-bootstrap/tofu.py` + `test_tofu.py` on `research/rkva-live-session-guard-20261030` | **No** | Linux prototype, not wired |
| WDP helpers | `tools/xbox/scripts/deploy.sh`, `DeployClient` in `tools/xbox/scripts/rkva-quality-gate.py` | Yes | Linux host; `curl -k`, no certificate pin |
| App autopilot | `tools/xbox/uwp/MainPage.cpp` (`ApParseScript`/`ApRun`), `include/xllama/autopilot.h` | Yes | UWP, Dev Mode; fixed op allowlist, no RKVA probe op |

## 2. Missing pieces (each is a hard prerequisite, not a TODO)

### 2.1 Identity store on main

- `tools/rkva-peer-identity/` is not built by any CMake/MSBuild target on main;
  there is no Windows test and no UWP compilation evidence.
- `provision_local_state()` returns
  `"AppContainer LocalState unavailable"` on every non-UWP build. There is no
  Linux backend, and adding one would not qualify AppContainer behavior.
- The store's integrity tag is an HMAC keyed by `auth_secret` carried **inside
  the same bundle**. It detects corruption, not an untrusted author. The API
  comment says the bundle must come from an explicitly authorized WDP
  provisioning session; no code in the tree establishes that provenance.
- The bundle expectation (`owner`, `ledger_epoch`, `cert_sha256`,
  `auth_epoch`, `package`) is caller-supplied. Nothing on main produces those
  values from a live authenticated peer.

### 2.2 Certificate pin

- `deploy.sh` authenticates with HTTP Basic over
  `https://${XBOX_IP}:11443` and `curl -k`: TLS verification is disabled and no
  leaf certificate digest is pinned. `DeployClient` inherits this.
- The console WDP certificate is not compared with the `cert_sha256` inside the
  identity bundle anywhere.
- `curl` supports `--pinnedpubkey sha256//<base64>`; no wrapper uses it.
- The controller's `--expected-cert-sha256` compares the live leaf digest with
  an operator-supplied pin, but the pin cannot be assumed trustworthy until it
  is obtained out of band (console UI / physical label), not from the
  connection it is validating.

### 2.3 TOFU boundary

- `tofu.py` pins the first presented certificate when
  `explicit_tofu=True`. It is a prototype with a local `identity.json`, no
  upload, no WDP and no managed capability.
- First-use pinning is acceptable only as an explicit, operator-confirmed
  bootstrap for a **new** host identity, and never as managed trust after an
  epoch change, rollback or certificate replacement.
- The managed boundary needs: an out-of-band digest, an operator confirmation
  that names host and package, a durable host-side record of the accepted
  digest, and a refusal policy when the console presents a different
  certificate. None of that is implemented.
- `tofu.py` is not on main and must not be copied into a production path as-is.

### 2.4 AppContainer / UWP

- LocalState access relies on the package identity and AppContainer ACL; the
  code does not inspect the ACL. `load_path` requires exactly
  `identity.lock` + `identity.json`, no reparse points, and an exclusive
  `LockFileEx` on `identity.lock`.
- `provision_path` writes `identity.pending`, then `identity.json` with
  `FlushFileBuffers`, flushes the directory handle, deletes the marker and
  re-inspects the namespace. The crash window between file flush and marker
  deletion is not tested on Windows; a surviving marker must make the store
  ambiguous, not adopted.
- There is no UWP `vcxproj` entry, no AppContainer compile, and no console
  runtime test. A desktop MSVC compile would not qualify the AppContainer
  path.
- ACL evidence is not exposed by WDP. It needs a test-only app action that
  reads the security descriptor (or a Windows host test on a copied tree), and
  `icacls` output recorded as evidence.

### 2.5 Package identity

- The PFN contains a publisher hash derived from the signing certificate.
  `deploy.sh pfn` returns the highest registered version and only warns when
  several are live; that is an implicit selection and is forbidden for
  destructive runs.
- No code checks that the installed package's publisher and certificate chain
  match the pinned WDP certificate or the identity bundle's `cert_sha256`.
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

Add a Windows-only test target (not present yet):

```text
tests/test-rkva-peer-identity-windows.cpp
  - provision_test_path(temp_dir, bundle, expectation) → true
  - load_test_path(temp_dir, expectation, identity) → exact round-trip
  - wrong owner/ledger_epoch/cert_sha256/auth_epoch/package → refused
  - tampered JSON / non-canonical JSON / extra key → refused
  - identity.pending present → refused
  - hardlink/symlink/reparse-point substitutes → refused
  - crash hook after file flush → marker survives → load refuses
  - auth_secret/anchor_secret wiped on move/destruction (instrumented buffer)
```

Register it only when `WIN32` and `RKVA_MANAGED_SNAPSHOT_PHASE4` are on, link
`rkva-managed-auth.h` (header-only) and the vendored nlohmann JSON. Build with
MSVC in an x64 developer prompt; record the compiler version and the exact
`git rev-parse HEAD`.

### Step 2 — UWP AppContainer compile

- Add `tools/rkva-peer-identity/rkva-peer-identity.cpp` to
  `tools/xbox/uwp/BeeLlama.props` (or a dedicated test-only vcxproj) with
  `RKVA_PEER_IDENTITY_UWP` and C++/WinRT.
- Build the existing package pipeline:

```powershell
pwsh -File scripts/build-xbox.ps1 -Configuration Release -Backend llamacpp `
  -WindowsSdkVersion 10.0.26100.0 -BuildRevision <n> -MaxParallel 10
```

- A compile is not runtime qualification. Record `dumpbin /dependents` and the
  packaged manifest; do not deploy unless an explicit deployment session is
  authorized.

### Step 3 — Certificate pin and TOFU boundary

- Obtain the WDP leaf SHA-256 out of band from the console UI. Record it in
  `identity-inventory.json` and compare against `--expected-cert-sha256`.
- Negative test: run `deploy.sh`/`curl` against the console with a deliberately
  wrong pin and require connection failure before any credential is sent.
- Positive test: `curl --pinnedpubkey sha256//<base64>` succeeds only with the
  out-of-band pin.
- Decide and document the TOFU boundary: explicit first-use confirmation is
  allowed only for a new host identity; after an accepted identity, a changed
  certificate is a hard refusal until the operator provisions a new bundle
  and records the new pin.

### Step 4 — Provisioning on the console

- Deploy the test build (explicit session, new revision).
- Use a test-only autopilot action `rkva_qual_identity` to:
  1. write the identity bundle into `LocalState\rkva-peer-identity-v3` under
     the AppContainer;
  2. return the namespace listing, the security-descriptor summary (owner,
     DACL entries, no inherited write for `Users`/`Everyone`), and the
     provision result;
  3. never return `auth_secret` or `anchor_secret` — only their presence and
     a one-way digest.
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
- The identity bundle is accepted only from an authorized provisioning
  session; its self-MAC is corruption detection, not authenticity.
- The AppContainer namespace is exact, exclusive and free of reparse points;
  a pending marker or partial namespace is ambiguous and refused.
- No secret appears in logs, evidence JSON or WDP listings.
- A compile, a desktop test or a local filesystem test does not qualify the
  console runtime or power-loss behavior.

## 5. Explicit non-claims

- Peer identity is **not implemented on main** and is not enabled.
- No Windows or UWP build was produced by this change.
- No certificate pin, TOFU bootstrap or identity provisioning was executed.
- Blocker 7 stays open until Steps 1–5 produce real artifacts and the console
  rows I4–I6 pass.
