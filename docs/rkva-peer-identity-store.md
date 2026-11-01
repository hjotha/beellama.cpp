# Peer identity store and certificate pin (blocker 7, default off)

Status: **implemented library and CPU tests only.** The store and pin API are
built only with `-DRKVA_PEER_IDENTITY_STORE=ON` and every entry point also
requires `RKVA_PEER_IDENTITY_STORE=1` at run time. No listener, HELLO, dispatch,
capability, UWP project or credential path calls it. No Windows/MSVC/UWP build,
console deployment, AppContainer runtime check, restart or power-loss test has
been run. Item 7 remains **BLOCKED** for those rows.

This document replaces the "store is Windows-only and untested" deferral in
`docs/rkva-managed-snapshot-phase4.md`. It does not close blockers 1-6 and does
not authorize enabling the managed capability.

## 1. Files

| File | Purpose |
| --- | --- |
| `tools/rkva-peer-identity/rkva-peer-identity.h` | Public types, record/schema constants, `store` interface, pin/enroll/rotate/revoke/provision/wipe/inspect API |
| `tools/rkva-peer-identity/rkva-peer-identity-core.cpp` | Portable canonical JSON codec, HMAC integrity, validation and pin state machine |
| `tools/rkva-peer-identity/rkva-peer-identity-win32.cpp` | Windows LocalState backend: reparse-point refusal, exclusive lock, atomic replace, protected AppContainer DACL, DPAPI key sealing, BCrypt randomness |
| `tools/rkva-peer-identity/rkva-peer-identity-test-store.cpp` | Test-only in-memory backend and POSIX file backend that mirrors the Windows write sequence |
| `tools/xbox/scripts/rkva-peer-identity-wdp.py` | Pinned read-only WDP inventory/evidence helper |
| `tests/test-rkva-peer-identity.cpp` | CPU tests for schema, MAC, pin match/mismatch, TOFU, rotation, revocation, wipe, corruption, atomic windows |
| `tests/test-rkva-peer-identity-wdp.py` | CPU tests for pin-before-auth ordering, exact PFN selection, store comparison, evidence redaction |
| `tests/test-rkva-peer-identity-static.py` | Static guards for CMake default-off, ACL API intent, no logging, pin ordering and non-claims |

CMake: `RKVA_PEER_IDENTITY_STORE` (default `OFF`) in the root
`CMakeLists.txt`; library and test targets in `tests/CMakeLists.txt`. The test
library compiles the same core with `RKVA_PEER_IDENTITY_TESTING` and is never
linked into a production target.

## 2. Record and trust model

Storage is an atomic file, not SQLite: the record is one bounded JSON object
(well under the 8 KiB limit), so a transaction database would add a second
durability and locking contract without a query need. The store is one canonical
JSON record in `<LocalState>/rkva-peer-identity-v3/identity.json` plus an
exclusive `identity.lock`. At rest the namespace must be exactly those two names. During
a write the transient names `identity.pending` and `identity.new` may exist; a
load that sees either refuses the store as ambiguous. There is no silent repair
and no adoption of a partial write: recovery is an explicit, operator-confirmed
`wipe()` followed by re-enrollment or re-provisioning.

Schema `rkva-peer-identity-v1`, version `1`, with these fields:

- `pfn`, `package_family`, `publisher`, `publisher_id`;
- `cert_sha256` (DER leaf digest) and `cert_thumbprint` (SHA-1, Windows
  convention);
- `peer_host`, `peer_port`;
- `first_seen` (Unix seconds);
- `pin_state` (`pinned` or `revoked`);
- `tofu` (`boundary`, `enrolled`, `enrolled_at`, `operator_label`);
- `rotation` (`generation`, `rotated_at`, `previous_cert_sha256`,
  `operator_label`);
- `revocation` (`revoked`, `reason`, `revoked_at`, `operator_label`);
- `integrity` (`algorithm: hmac-sha256`, `key_id`, `key_protected`, `mac`).

The MAC covers the canonical body with the `integrity` object removed. The key
is 32 random bytes from `BCryptGenRandom`; `key_protected` is the DPAPI-protected
key and `key_id` is the first 8 bytes of its SHA-256. The key never appears in
plaintext, in an error string or in the inspection view.

Limits that must not be misread:

- The MAC detects corruption, truncation, non-canonical text and copies to a
  different Windows user/machine. The protected key lives in the same record, so
  a same-user writer who can call DPAPI can replace the whole record and re-MAC
  it. The MAC is **not** proof of who authored a bundle.
- A bundle is accepted only from an explicitly authorized provisioning session,
  and `provision()` still requires the same out-of-band certificate digest as
  enrollment. The self-MAC is not authenticity.
- DPAPI under an AppContainer has **not** been compiled or qualified. If the
  UWP build cannot use `CryptProtectData`, the backend needs a
  `DataProtectionProvider("LOCAL=user")` implementation and a fresh review; do
  not silently fall back to a plaintext key.

## 3. Pin / verify API

`verify(store&, const peer&, error)` is the connection hook. It must run after
the leaf certificate is observed and before any authenticated frame,
credential or managed payload is accepted. It is read-only and never enrolls.
It refuses, in order, when:

- the compile/run-time gate is closed (`disabled`);
- the peer claim is malformed (`invalid_argument`);
- the record is absent (`unknown`), corrupt/ambiguous/bad MAC (`corrupt`) or
  locked/unreadable (`unavailable`);
- the record is revoked (`revoked`);
- PFN, package family, publisher, host, port, certificate SHA-256 or thumbprint
  differ (`mismatch`).

`enroll_tofu` is first-use only. It refuses when any record exists
(`already_pinned`), when `explicit_operator_confirmation` is false or the label
is invalid (`unauthorized`/`invalid_argument`), or when the caller's
out-of-band digest differs from the presented leaf (`unauthorized`).
`provision` is the authorized-session path and never acts as a TOFU shortcut:
`authorized_wdp_session=true`, a bounded operator label and an out-of-band
digest are all required.

`rotate` requires explicit confirmation, a label and the out-of-band digest of
the new certificate. It refuses a revoked record, the current certificate, a
certificate already in the bounded rotation history (rollback to a retired pin)
and an exhausted generation counter. It records the retired digest and the
rotation time and updates the bound host/package fields.

`revoke` requires explicit confirmation, a label and a reason, is monotonic, and
makes every later `verify` and `rotate` fail closed. `wipe` is the explicit
recovery path for a corrupt/ambiguous store; it also requires confirmation.

The API returns only status codes and static error descriptions. Error strings
name the mismatched field and never contain a key, MAC, certificate byte,
credential or protected-key blob. The library contains no logging call.

## 4. Windows LocalState and AppContainer behavior (not qualified)

`open_local_state()` uses
`Windows::Storage::ApplicationData::Current().LocalFolder()` only in a build
with `RKVA_PEER_IDENTITY_UWP`; a desktop build refuses rather than guessing a
path. `open_test_directory()` exists only for tests.

The Windows backend:

1. rejects reparse points on the namespace, lock, record, pending and staging
   entries (`FILE_FLAG_OPEN_REPARSE_POINT`, attribute checks) and requires a
   single-link regular file for the record;
2. takes `LockFileEx(LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY)` on
   `identity.lock`;
3. creates `identity.pending`, flushes it, creates `identity.new`, writes and
   flushes it, renames it over `identity.json` with
   `MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)`, flushes the
   directory, deletes `identity.pending` and flushes again;
4. re-inspects the namespace and refuses if it is not exactly the lock and
   record;
5. applies a protected DACL (`PROTECTED_DACL_SECURITY_INFORMATION`) containing
   only the current user SID and, when the token is an AppContainer, its package
   SID (`TokenAppContainerSid`). It does not grant Everyone, Users or inherited
   ACEs.

None of this is AppContainer runtime evidence. The exact ACL must be read back
with `icacls` on the console (row I6), and the directory flush behavior must be
tested on NTFS with the real package. The POSIX test backend mirrors the
sequence but makes no Windows/ACL claim.

## 5. WDP helper

```sh
python3 tools/xbox/scripts/rkva-peer-identity-wdp.py \
  --xbox-host 192.168.1.200 \
  --pfn GianlucaMazza.xllama_1.6.0.436_x64__pj67f1fcj4n14 \
  --expected-cert-sha256 <out-of-band leaf SHA-256> \
  --out /path/to/identity-inventory.json
```

Optional: `--expected-publisher`, `--store-json <exported identity.json>`,
`--skip-inventory`, `--xbox-env <credential file>`.

The helper connects with TLS verification disabled **only** because the leaf
digest is compared against the out-of-band pin immediately after the handshake
and before any HTTP request. On mismatch it closes the connection and sends no
Basic credential. It then fetches the exact installed PFN from
`/api/app/packagemanager/packages`, records publisher/version/family, optionally
compares the public fields of an exported record (the MAC/DPAPI key stay opaque
to the host), and records the `rkva-peer-identity-v3` namespace names without
downloading `identity.json`.

It deliberately does not call `deploy.sh`: that script uses `curl -k`, disables
TLS verification and authenticates before any certificate check. The helper
reuses the same WDP endpoints and credential-file format.

Evidence never contains `XBOX_PASS`, an Authorization header or Basic token.

## 6. CPU validation

```sh
cmake -S . -B /tmp/opencode/beellama-peer-identity -G Ninja \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
  -DRKVA_PEER_IDENTITY_STORE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/beellama-peer-identity -j 6
ctest --test-dir /tmp/opencode/beellama-peer-identity \
  -R 'rkva-peer-identity' --output-on-failure
```

The default-off check configures without `-DRKVA_PEER_IDENTITY_STORE` and
requires that no `rkva-peer-identity` target exists. The static test also
asserts the CMake option is `OFF`.

Covered by CPU tests: canonical schema round-trip; version/extra-key/non-
canonical/truncated refusal; MAC tamper; pin match and every field mismatch;
unknown store; explicit TOFU only; repeated enrollment refusal; authorized
provisioning requirements; rotation generation/history, same-cert and retired-
cert refusal; revocation and post-revocation refusal; explicit wipe; runtime
gate values; POSIX namespace exactness, 0600 files/0700 directory, symlink,
pending/staging/rename crash windows, concurrent lock, missing LocalState;
error strings with no key material. Not covered: Windows, ACL, DPAPI, UWP,
console, WDP network, restart or power loss.

## 7. Exact remaining item-7 work (do not mark PASS before these)

### 7.1 MSVC/UWP compile (row I4, still BLOCKED)

Not run. The source is not in `tools/xbox/uwp/BeeLlama.props` or any vcxproj.
The required integration is:

- add `rkva-peer-identity-core.cpp` and `rkva-peer-identity-win32.cpp` to the
  UWP project (or a dedicated test-only vcxproj) with
  `RKVA_PEER_IDENTITY_STORE` and `RKVA_PEER_IDENTITY_UWP`, C++/WinRT and C++17;
- link `advapi32.lib`, `crypt32.lib` and `bcrypt.lib`;
- build from an x64 MSVC developer prompt and the existing UWP pipeline
  (`pwsh -File scripts/build-uwp.ps1 -Configuration Release -Platform x64
  -Backend llamacpp -BuildRevision <n>` or
  `scripts/build-xbox.ps1 ... -WindowsSdkVersion 10.0.26100.0`);
- record the compiler version, exact `git rev-parse HEAD`, `dumpbin
  /dependents` and the packaged manifest;
- run the C++ tests on Windows (memory backend plus
  `open_test_directory` over a copied LocalState tree).

A desktop MSVC compile is not AppContainer qualification. A compile is not
console runtime qualification.

### 7.2 AppContainer ACL and namespace (row I6, still BLOCKED)

Run on the console with the test build: `icacls` the namespace, record that only
the package SID (and the expected user SID) can read/write, that inherited
Users/Everyone entries are absent, and that no reparse point exists. Also record
the exact namespace at rest (`identity.lock` + `identity.json`), then the
transient pending/new names during a controlled write.

### 7.3 Console provisioning and restart (row I5, still BLOCKED; DESTRUCTIVE)

No app-side provisioning action exists yet. A test-only autopilot action (behind
the existing `XLLAMA_QUAL_PROBE` pattern, not a production capability) must call
`open_local_state()` and `provision()` with the operator-supplied out-of-band
digest and label, return only non-secret evidence, and never return the key,
MAC or certificate bytes. Then, with an explicit deployment session:

1. obtain the WDP leaf SHA-256 out of band from the console UI/physical label;
2. deploy the test build with a new revision, preserving package identity;
3. provision, then confirm the namespace and `icacls` evidence;
4. restart the app cleanly and require `verify()` to succeed with the same pin;
5. present a wrong certificate/publisher/PFN and require refusal;
6. copy an older LocalState identity back and require a hard refusal (rollback),
   not silent adoption;
7. interrupt power during provisioning; on restart a surviving
   `identity.pending`/`identity.new` must refuse, not repair itself;
8. revoke and rotate through the explicit operator paths and require the old
   pin to fail.

Power loss can lose in-flight console data. Use a disposable test build and a
snapshot set that can be lost; an operator must be physically present.

### 7.4 Production wiring

There is still no listener/session authority to call `verify()` at accept time.
The connection hook must be bound to the accepted connection, the challenge
source and the exact session state before the managed capability is enabled
(blockers 1-6). Until then the store must not be called from a dispatch path and
the runtime gate must stay closed.

### 7.5 Cryptographic and platform review

DPAPI/ACL behavior under AppContainer, constant-time comparison codegen on the
qualified MSVC build, key provisioning, MAC domain separation, rotation/revocation
operator procedure and secret handling still need independent review.

## 8. Explicit non-claims

- No Windows or UWP build was produced by this change.
- No AppContainer path, ACL, DPAPI behavior, package identity or console
  runtime was exercised.
- No certificate pin, TOFU enrollment, provisioning, rotation or revocation was
  executed against a console.
- No restart or power-loss test exists.
- Blocker 7 and rows I1-I6 remain open until the artifacts above exist and the
  controller records PASS for the specific rows.
