# Managed snapshot console qualification matrix (blockers 2, 3, 7)

Status: **planning and harness only.** Nothing in this document has been run
against the Xbox. No managed capability is advertised, no object is published
and no GC exists. The Phase 2/3/4 code on main `4443ba6ee` remains default-off
and reachable only from CPU tests. Every row below is either:

- **READY (host)** — executable now on a Linux CPU host without a console;
- **HARDWARE** — needs the console and the exact state named in the row;
- **BLOCKED** — needs code that is not in this tree; the harness records
  `BLOCKED`, never `PASS`.

The controller is
`tools/xbox/scripts/xbox-snapshot-console-qual.py`. It is inert without
`--execute`; see `--help` and the preview plan. It does not install, uninstall,
delete snapshots or change package identity.

## 0. Facts the harness relies on

- Production router: `llama-server-root.service`, preset
  `/home/hjotha/router-production.ini`, root-owned `llama-server` child on a
  private port.
- Console app: `GianlucaMazza.xllama_1.6.0.436_x64__pj67f1fcj4n14`, reached over
  WDP at `https://<xbox>:11443` through
  `tools/xbox/scripts/deploy.sh` and `DeployClient` from
  `tools/xbox/scripts/rkva-quality-gate.py`.
- LocalState flag/probe root: `LocalState\rkva-qual\`. Existing autopilot
  rendez-vous files (`autopilot.flag`, `autopilot.json`,
  `autopilot-done.txt`) are used only when absent; the harness refuses to
  overwrite them.
- Managed snapshot namespace: `rkva-snapshots-v3` (never `v1`/`v2`).
- Peer identity namespace: `rkva-peer-identity-v3` (absent today).
- Phase 3/4 runtime gates: `RKVA_MANAGED_SNAPSHOT_PHASE3=1` and
  `RKVA_MANAGED_SNAPSHOT_PHASE4=1`, both only meaningful in a build that
  defines the matching compile-time option.

There is **no** call site, capability, HELLO change, dispatch, listener or app
probe for any of this in the deployed app. The controller's probe action
(`rkva_qual_probe`) does not exist yet, so today an execute run ends with
`BLOCKED` and the app's exact `autopilot-done.txt` error, not `PASS`.

## 1. Required app-side probe contract (not implemented)

The matrix rows that say PROBE require a **test-only** UWP build that adds
exactly one autopilot action and no production capability:

- action: `{"op": "rkva_qual_probe", "name": "<case>", "label": "<128-bit hex>"}`;
- input: `LocalState\rkva-qual\request-<hex>.json`, canonical JSON;
- output: `LocalState\rkva-qual\response-<hex>.json`;
- completion: the existing `autopilot-done.txt` (`ok` or `error: ...`);
- the action must be compiled only under a test-only macro and must not be
  advertised in HELLO or `supported_ops`;
- the UWP app cannot receive process environment variables from WDP, so the
  test action itself may set `RKVA_MANAGED_SNAPSHOT_PHASE3=1` /
  `RKVA_MANAGED_SNAPSHOT_PHASE4=1` **only** after validating the explicit
  `phase_flags` object in the request, and must unset them before returning.
  A production build must not compile this action or read these variables.

Attestation response (schema `rkva-console-attest-v1`):

```json
{
  "schema": "rkva-console-attest-v1",
  "probe_id": "<hex>",
  "build": {
    "commit": "4443ba6",
    "phase3_compiled": true, "phase3_enabled": true,
    "phase4_compiled": true, "phase4_enabled": true,
    "caps": ["REGISTRY_V1", "AUTH_V2", "FRAMING_V3"]
  },
  "attestation": {
    "session_id": 0, "live_rows": 0, "generation": 0,
    "submitted_fence": 0, "completed_fence": 0,
    "host_key_sha256": "<64 hex>", "prefix_sha256": "<64 hex>",
    "metadata_sha256": "<64 hex>", "coverage_mac": "<64 hex>",
    "state_digest": "<64 hex>"
  },
  "negative": {
    "kinds": ["generation_mismatch", "key_mismatch", "session_mismatch",
              "prefix_mismatch", "fence_regression"],
    "all_refused": true, "accepted_any": false,
    "coverage_mac_emitted": false
  }
}
```

The positive document must be produced by
`rkva_resident_attestation::make()` and
`rkva_managed_attest_v3::make_reply/accept_reply` over a live session; the
negative document must show the same code refusing tampered requests. A
hand-written JSON file is not acceptable evidence: the controller cannot
distinguish it from a probe that never ran, and both must remain non-PASS.

The host expectation file (`--expect`) is produced by the host side, not by
this controller:

```json
{
  "session_id": 1,
  "rows": 4096,
  "generation": 1,
  "host_key_sha256": "<64 hex>",
  "prefix_sha256": "<64 hex>"
}
```

Today no host-side producer exists either: there is no listener, session
authority, challenge source or authenticated COMMIT/ATTEST reply (blockers 1
and 4). Therefore blocker 2 is blocked on **both** sides of the wire.

## 2. Blocker 2 — authenticated pin/metadata proof

| ID | Action | Artifacts | PASS evidence | FAIL evidence | Hardware state |
| --- | --- | --- | --- | --- | --- |
| A0 | Host: run the Phase 2/3/4 CPU suites and record the exact build. | `ctest-rkva.log`, `git rev-parse HEAD` | 29/29 `rkva|console-qual` tests pass (27 prior + `test-rkva-durable-fault` + `test-xbox-snapshot-console-qual`), Phase 4 gate tests included | any failure, or Phase 3/4 options not actually ON | Linux host, CPU only |
| A1 | Console: positive probe with the host expectation. | `probe-request-positive.json`, `probe-response-positive.json`, `autopilot-done.txt`, `xllama-*.tail.log` | session/rows/generation/key/prefix equal the host expectation; `submitted<=completed`; `completed>0`; nonzero coverage MAC distinct from metadata/key digests; build reports both phases compiled **and** enabled and all three caps | any mismatch, sentinel MAC, missing build evidence, or `done != ok` | Test-only app build with the probe; Phase 4 compiled in; `--allow-phase-flags` |
| A2 | Console: repeat the positive probe with a fresh probe id/session. | `probe-response-repeat.json` | identical prefix digest; coverage MAC changes with the session id | identical MAC across sessions or changed prefix | Same app instance; no reboot required |
| A3 | Console: negative probe with tampered generation, key, session, prefix and a regressed fence. | `probe-response-negative.json` | every case refused, `accepted_any=false`, no coverage MAC | any acceptance | Same app instance |
| A4 | Host: bind the accepted pin to the journal/SQLite attempt (owner, transaction, reference, key, metadata digest, reservation). | `host-pin-record.json` | the exact pin is the same record the pending attempt holds; no caller-supplied identity digest alone | shape-only check or a fabricated identity | Linux host; needs the anchored-SQLite `apply` pin proof that main does **not** have (`ggml/src/ggml-remote-attn/rkva-managed-object-v3.cpp:115` still closes) |

A4 is **BLOCKED**: `rkva_anchored_sqlite::apply` refuses every managed
transition with “concrete authenticated pin and metadata proof unavailable”,
and `rkva_managed_object_v3::write()`/`verify()` return that same refusal
before any backend call (`tools/xbox/uwp/rkva/rkva-managed-object-v3.cpp:92`
and `:115`). The console probe can therefore prove the *frames* (A1–A3) at
best; it cannot prove that a pending pin corresponds to the live session and
real resident coverage.

## 3. Blocker 3 — durable sync / rename / power loss / stale lock / staging

These rows are the acceptance criteria from
`docs/rkva-v3-proof-integration-blockers.md` and
`docs/rkva-managed-snapshot-phase4.md`. D1/D2 are prerequisites only; D3–D7
need the app-side staging probe and fault selector, which do not exist.

| ID | Action | Artifacts | PASS evidence | FAIL evidence | Hardware state |
| --- | --- | --- | --- | --- | --- |
| D1 | Write/fetch/delete one self-created 4 KiB file in `LocalState\rkva-qual`. | `durability-probe.json` | fetched bytes equal the uploaded bytes; file absent afterwards | digest mismatch or silent write | App may be running; console storage writable |
| D2 | Clean app stop/start; re-inventory `rkva-snapshots-v3`. | `app-restart.json`, `xbox-inventory-after-restart.json` | identical inventory; no new/removed staging name | namespace drift | App started by the controller |
| D3 | Stage a v3 object and fail `sync_file` on chunk *k* (injected). | `durability-fault-chunk-sync.json`, `staging-inventory.json` | previous committed pair byte-identical; staging preserved and charged; no object published; retry refuses to reuse the stale stage | any deletion, adoption, overwrite or “durable” claim | Test build with fault selector; app stopped except the probe |
| D4 | Fail manifest sync and staging directory sync. | `durability-fault-manifest-sync.json`, `durability-fault-dir-sync.json` | same as D3; no rename attempted | object appears | Same |
| D5 | Rename no-replace with a pre-existing destination; then inject rename `EIO`. | `durability-fault-rename*.json` | destination unchanged; source either intact or explicitly `uncertain`; never `durable` | overwrite or `durable` on `EIO` | Same |
| D6 | Power loss between rename and parent-directory sync. | `power-loss-timeline.json`, `post-power-loss-inventory.json` | after restart the pair is classified `uncertain`/pending, charged and pinned; old pair still usable | object reported committed, charge lost, nonce replayed, or silent cleanup | **DESTRUCTIVE**: operator cuts power; `--destructive --power-loss-method manual` |
| D7 | `SIGKILL`/force-stop the app while it holds the store lease, then restart. | `stale-lock.json` | stale lock is reclaimable only with owner evidence; PID reuse cannot impersonate the owner; stage remains charged | second writer enters, or the stage is adopted/deleted | Test build; app started by the controller |
| D8 | Lost ACK: host COMMIT durable, console reply lost; restart both. | `lost-ack.json`, host journal inspect | ambiguous pair stays pinned; no automatic anchor replacement or GC | automatic recovery or nonce reuse | Host + console; requires blocker 1/4/5 code that is absent → **BLOCKED** |

D3–D7 are blocked until a test-only staging/fault probe exists. The
test-only injector in `tests/rkva-durable-fault.h` exercises the *contract
semantics* on CPU (no-replace, sync failure, ack loss, crash-after-apply) but
does not make the closed writer reachable; see
`docs/rkva-durable-fault-injection.md` for the exact seam proposal.

## 4. Blocker 7 — peer identity / AppContainer / UWP

| ID | Action | Artifacts | PASS evidence | FAIL evidence | Hardware state |
| --- | --- | --- | --- | --- | --- |
| I1 | Inventory the WDP leaf certificate SHA-256 out of band and compare it with `--expected-cert-sha256`. | `identity-inventory.json`, `wdp-certificate.json` | exact digest match; operator obtained the digest from the console UI, not from the connection | missing pin, mismatch, or a TOFU-derived pin | Console reachable; WDP credentials |
| I2 | Inventory the installed package: exact PFN, publisher, version. | `identity-inventory.json` | PFN equals the operator-supplied value; publisher matches the signing certificate pin | implicit `deploy.sh pfn` selection or publisher mismatch | Same |
| I3 | Inventory `LocalState\rkva-peer-identity-v3` and its entries. | `identity-inventory.json` | either absent (expected today) or exactly `identity.lock` + `identity.json` | unexpected names or a partial identity | Same |
| I4 | MSVC UWP compile of the identity store with `RKVA_PEER_IDENTITY_UWP`; provision/load/wipe tests on Windows. | `msvc-uwp-build.log`, `peer-identity-tests.log` | compile passes; provisioning writes under AppContainer LocalState; secrets wiped; mismatched expectations refused | compile failure, non-AppContainer path, plaintext secret in logs | Windows build host with MSVC + UWP SDK; **compile is not console runtime** |
| I5 | Deploy the test build, provision through the authorized WDP session, restart, then power-interrupt. | `identity-provision.json`, `identity-reload.json`, `restart-timeline.json` | identity survives a clean restart; mismatched cert pin/PFN refused; rollback attempts rejected | bundle accepted without an authenticated source, TOFU used for managed trust, or rollback accepted | **DESTRUCTIVE**: test package deployment and console restart; explicit deployment session required |
| I6 | AppContainer ACL and namespace check on the console. | `icacls-identity.txt` | only the package SID can read/write; no inherited Users/Everyone | broad ACL or reparse point | Console shell; `icacls` output pasted into the evidence dir |

I4–I6 are **BLOCKED** today: `tools/rkva-peer-identity/` exists only on
`research/rkva-peer-identity-20261030` (`e2526b34d`), is Windows-only, and is
not built or tested by any target in this tree. The missing pieces and the
build/test plan are in `docs/rkva-peer-identity-uwp-inventory.md`.

## 5. Destructive-test warnings

- Power loss (D6) can lose in-flight console data. Use only a disposable test
  build and a snapshot set you can afford to lose. Do not run it while any
  other LAN host can connect to the console.
- `--destructive` does not make a run safe by itself; it only unlocks D6/D7.
  An operator must be physically present for D6.
- Never uninstall the app: it removes LocalState, including models and saved
  sessions. `deploy.sh` is used only for `start-app`, `stop-app`,
  `upload-file`, `fetch-file`, `delete-file` (probe files only) and
  `list-localstate`.
- Never run `deploy.sh pfn` as an implicit package selector in a destructive
  run; the PFN is a required argument.
- Do not enable `RKVA_MANAGED_SNAPSHOT_PHASE3/4` on the production service or
  in the deployed package without blocker 1–7 closure.
- The controller stops production only after two idle checks and refuses to
  start if a GPU compute process or an established `xbox:50054` socket exists.
  This is not an atomic lease: another LAN client can still connect, so an
  exclusive-use agreement is mandatory.
- A failed/blocked run must not be retried by reusing a previous result
  directory. Each run uses a new `--label`.

## 6. Rollback

Per run, in the controller's `finally` (in this order):

1. stop the app only if this controller started it (a pre-existing app
   is never stopped by an identity-only run);
2. delete only the probe files listed in `probe_files_created`, then
   `autopilot.flag/json/done` if they still exist; never touch snapshot files;
3. re-read flags and the `rkva-snapshots-v3` namespace and record drift as a
   `cleanup_error`;
4. re-verify the exact PFN is the sole running app, then `systemctl start
   llama-server-root.service` and require `ActiveState=active`;
5. write `result.json`.

If the controller aborts before production is paused, no service action is
taken. If production was paused, it is always restarted even when the console
changed; the drift is recorded as `FAILED_CLEANUP` rather than hidden.

Manual fallback:

```sh
sudo systemctl status llama-server-root.service
sudo systemctl start llama-server-root.service
systemctl is-active llama-server-root.service
bash /home/hjotha/beellama.cpp/tools/xbox/scripts/deploy.sh stop-app 'GianlucaMazza.xllama_1.6.0.436_x64__pj67f1fcj4n14'
```

The manual fallback for a stuck probe is to delete only
`LocalState\rkva-qual\*` and the three autopilot files; never delete
`rkva-snapshots-v3`, `rkva-snapshots-v2` or any `n-*.kv` / `manifest*.bin`.
The empty `rkva-qual` directory itself is left in place: `deploy.sh` has no
rmdir subcommand and the harness does not add one.

## 7. Exact commands when the Xbox is available

Preview (safe, no console access):

```sh
python3 /home/hjotha/worktrees/beellama-snapshot-console-qual-20261101/tools/xbox/scripts/xbox-snapshot-console-qual.py \
  --label preview --scenario attest --scenario durability --scenario identity
```

Host CPU evidence:

```sh
cmake -S /home/hjotha/worktrees/beellama-snapshot-console-qual-20261101 -B /tmp/opencode/beellama-snapshot-qual \
  -G Ninja -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
  -DRKVA_MANAGED_SNAPSHOT_PHASE2=ON -DRKVA_MANAGED_SNAPSHOT_PHASE3=ON \
  -DRKVA_MANAGED_SNAPSHOT_PHASE4=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/beellama-snapshot-qual -j 6
ctest --test-dir /tmp/opencode/beellama-snapshot-qual -R 'rkva|console-qual' --output-on-failure
```

Attestation run (only after the test build exists and a host expectation has
been produced; it will be `BLOCKED` until then):

```sh
python3 tools/xbox/scripts/xbox-snapshot-console-qual.py \
  --execute --label attest-r1 --scenario attest \
  --xbox-host 192.168.1.200 \
  --pfn GianlucaMazza.xllama_1.6.0.436_x64__pj67f1fcj4n14 \
  --expect /path/to/host-expect.json \
  --allow-app-run --allow-phase-flags
```

Durability prerequisite run (no power loss, no destructive flag):

```sh
python3 tools/xbox/scripts/xbox-snapshot-console-qual.py \
  --execute --label durability-prereq-r1 --scenario durability \
  --xbox-host 192.168.1.200 \
  --pfn GianlucaMazza.xllama_1.6.0.436_x64__pj67f1fcj4n14 \
  --allow-app-run
```

Power-loss run (operator present; expected `BLOCKED` for blocker 3 until the
staging probe exists):

```sh
python3 tools/xbox/scripts/xbox-snapshot-console-qual.py \
  --execute --label durability-power-r1 --scenario durability \
  --xbox-host 192.168.1.200 \
  --pfn GianlucaMazza.xllama_1.6.0.436_x64__pj67f1fcj4n14 \
  --allow-app-run --destructive --power-loss-method manual
```

Identity inventory (read-only):

```sh
python3 tools/xbox/scripts/xbox-snapshot-console-qual.py \
  --execute --label identity-inventory-r1 --scenario identity \
  --xbox-host 192.168.1.200 \
  --pfn GianlucaMazza.xllama_1.6.0.436_x64__pj67f1fcj4n14 \
  --expected-cert-sha256 <out-of-band leaf SHA-256>
```

What the controller actually implements today: D1, D2, the D4 power-loss
timeline and I1-I3. It does **not** implement D3, D5, D7 or I4-I6; those need
the test-only app probe/fault selector and the Windows/UWP work described in
`docs/rkva-peer-identity-uwp-inventory.md`. It refuses `--destructive` for the
identity scenario rather than pretending to deploy.

## 8. Acceptance rule

A row is PASS only when its artifact exists, the controller's validator
accepted it, and `result.json` says `PASS` for that scenario. Everything else
is `BLOCKED`, `PREREQ_ONLY`, `INCOMPLETE` or `FAILED`. No row in this document
may be marked PASS from a host unit test, a synthetic filesystem, a WDP
round-trip or a clean app restart.
