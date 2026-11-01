#!/usr/bin/env python3
"""Xbox managed-snapshot console qualification controller (skeleton).

INERT WITHOUT --execute. In preview mode this program only prints the
qualification plan; it does not import the deploy client, open sockets, read
files outside its own source, or touch the console, the GPU or the production
service.

What this controller is allowed to do when --execute (and the matching explicit
gates) are supplied:

  * verify the production router is active and twice idle, then stop it;
  * start/stop the existing xllama package through WDP (never install,
    uninstall, delete snapshots or change package identity);
  * drive a test-only attestation probe by writing files under
    LocalState\\rkva-qual\\ and an autopilot script, then collecting the JSON
    evidence the app writes back;
  * perform an operator-driven power interruption for the durability scenario
    (the operator cuts power; the controller only records the timeline);
  * restore the app flags/production service in a finally block.

What it deliberately cannot do:

  * it cannot make the Phase 2/3/4 managed lifecycle reachable: main has no
    call site, no capability and no object publication, so the attestation
    probe action does not exist in the deployed app yet. In that case the run
    is recorded as BLOCKED with the exact app error, never as PASS;
  * it does not claim durability from a file copy, a restart or a WDP upload;
  * it never selects a package implicitly: --pfn is mandatory and every
    LocalState/WDP request uses that exact PFN.

The evidence contract consumed from the app is documented in
docs/rkva-snapshot-console-qualification-matrix.md.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request

SERVICE = 'llama-server-root.service'
PRESET = Path('/home/hjotha/router-production.ini')
REPO = Path('/home/hjotha/beellama.cpp')
DEPLOY_SH = REPO / 'tools/xbox/scripts/deploy.sh'
QUALITY_GATE = REPO / 'tools/xbox/scripts/rkva-quality-gate.py'
XBOX_ENV = Path('/home/hjotha/.config/xllama/xbox-env')
RESULTS_ROOT = Path('/home/hjotha/beellama-benchmark-results/snapshot-console-qual')
XBOX_RKVA_PORT = 50054

QUAL_DIR = 'rkva-qual'
SNAPSHOT_DIR = 'rkva-snapshots-v3'
IDENTITY_DIR = 'rkva-peer-identity-v3'
AUTOPILOT_FILES = ('autopilot.flag', 'autopilot.json', 'autopilot-done.txt')
PROBE_OP = 'rkva_qual_probe'
SCHEMAS = {
    'attest': 'rkva-console-attest-v1',
    'durability': 'rkva-console-durability-v1',
    'identity': 'rkva-console-identity-v1',
}
SCENARIOS = ('attest', 'durability', 'identity')

# A durability probe must use this disposable namespace under rkva-qual, never
# the real LocalState\rkva-snapshots-v3. The root filename must still be
# exactly rkva-snapshots-v3 because the v3 writer validates it.
DURABILITY_ROOT = 'rkva-qual\\durable\\rkva-snapshots-v3'
DURABILITY_FAULT_CASES = (
    'chunk_sync',
    'manifest_sync',
    'dir_sync',
    'rename_existing',
    'rename_eio',
    'ack_lost',
    'crash_after_apply',
)
DURABILITY_EXPECTED_STATE = {
    'chunk_sync': 'staging_preserved',
    'manifest_sync': 'staging_preserved',
    'dir_sync': 'staging_preserved',
    'rename_existing': 'staging_preserved',
    'rename_eio': 'publication_uncertain',
    'ack_lost': 'publication_uncertain',
    'crash_after_apply': 'publication_uncertain',
    'stale_lock': 'staging_preserved',
}
DURABILITY_UNCERTAIN_CASES = {'rename_eio', 'ack_lost', 'crash_after_apply'}
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
LABEL = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z')
PFN_RE = re.compile(r'[A-Za-z0-9._-]+_[A-Za-z0-9.]+_[A-Za-z0-9]+__[a-z0-9]{13}\Z')


# ---------------------------------------------------------------------------
# Plan and pure validation (no I/O; unit tested)
# ---------------------------------------------------------------------------

def build_plan(scenarios):
    """Return the ordered qualification plan for the requested scenarios."""
    chosen = tuple(s for s in SCENARIOS if s in set(scenarios))
    if not chosen:
        raise ValueError('no scenario selected')
    steps = [{
        'id': 'P0',
        'blocker': 'precondition',
        'action': 'verify production router active, twice idle, exact PFN running, '
                  'no local GPU compute, no established Xbox RKVA socket',
        'artifacts': ['production-status.json', 'xbox-inventory-before.json'],
        'pass': 'two consecutive /slots reads report no processing and the '
                'production PID/argv/preset hash did not change',
        'fail': 'any busy slot, PID reuse, preset drift or unknown Xbox package',
        'hardware_state': 'production active; app may be running; no test build required',
        'destructive': False,
    }, {
        'id': 'P1',
        'blocker': 'precondition',
        'action': 'stop production only after the idle/identity recheck, then '
                  're-verify GPU and Xbox state',
        'artifacts': ['production-stop.json'],
        'pass': 'systemctl ActiveState=inactive and no new Xbox file appeared',
        'fail': 'production still active or Xbox inventory changed during isolation',
        'hardware_state': 'production stopped; console reachable; exclusive-use '
                          'agreement required (another LAN host can still connect)',
        'destructive': True,
    }]
    if 'attest' in chosen:
        steps += [{
            'id': 'A1',
            'blocker': '2',
            'action': 'run one autopilot rkva_qual_probe positive attestation case '
                      'with the requested session/rows/generation/key/prefix',
            'artifacts': ['probe-request-positive.json', 'probe-response-positive.json',
                          'autopilot-done.txt', 'xllama.log'],
            'pass': 'session id, live rows, generation, host key digest, prefix '
                    'digest, submitted<=completed fences and a nonzero coverage '
                    'MAC all match the caller-supplied expect file; build '
                    'evidence reports Phase 3 and Phase 4 compiled and enabled',
            'fail': 'any mismatch, zero/absent MAC, fence regression, or a '
                    'response that omits the build/coverage fields',
            'hardware_state': 'test-only app build with the probe action; Phase 4 '
                              'compiled in (RKVA_MANAGED_SNAPSHOT_PHASE4) and '
                              'explicitly enabled for the probe',
            'destructive': False,
        }, {
            'id': 'A2',
            'blocker': '2',
            'action': 'replay the same probe with tampered generation, host key, '
                      'session and prefix expectations and a regressed fence',
            'artifacts': ['probe-request-negative-*.json', 'probe-response-negative-*.json'],
            'pass': 'every tampered case is refused with status invalid/unavailable '
                    'and no coverage MAC is produced',
            'fail': 'any tampered case is accepted or emits a MAC',
            'hardware_state': 'same test build; no reboot between A1 and A2',
            'destructive': False,
        }, {
            'id': 'A3',
            'blocker': '2',
            'action': 'compare the console prefix digest with the host expectation '
                      'and repeat the positive probe to prove the digest is stable '
                      'and session-bound',
            'artifacts': ['probe-response-repeat.json', 'host-expect.json'],
            'pass': 'identical prefix digest across runs, different coverage MAC for '
                    'a different session id, coverage bound to the same metadata',
            'fail': 'prefix digest differs from host truth or coverage is identical '
                    'across session ids',
            'hardware_state': 'test build; host expectation produced by the host '
                              'journal/SQLite run, not by this controller',
            'destructive': False,
        }]
    if 'durability' in chosen:
        steps += [{
            'id': 'D1',
            'blocker': '3 (prerequisite only)',
            'action': 'write, fetch and byte-compare one 4 KiB file under '
                      'LocalState\\rkva-qual\\durability\\, then delete only that file',
            'artifacts': ['durability-probe.json'],
            'pass': 'fetched SHA-256 equals the written SHA-256 and the file is '
                    'absent afterwards',
            'fail': 'digest mismatch, silent write, or the file cannot be removed',
            'hardware_state': 'app running; console storage writable',
            'destructive': False,
        }, {
            'id': 'D2',
            'blocker': '3 (prerequisite only)',
            'action': 'stop and restart the app through WDP and re-read the v3 '
                      'namespace and LocalState listing',
            'artifacts': ['app-restart.json', 'xbox-inventory-after-restart.json'],
            'pass': 'app returns, rkva-snapshots-v3 inventory is unchanged and no '
                    'partial staging name appears or disappears',
            'fail': 'app does not return or the namespace changes without a recorded '
                    'publication',
            'hardware_state': 'app started by this controller; production still stopped',
            'destructive': False,
        }, {
            'id': 'D3',
            'blocker': '3',
            'action': 'run the test-build staging probe with chunk_sync injected '
                      'on chunk k and collect the writer, pin, staging and '
                      'inventory evidence',
            'artifacts': ['durability-fault-chunk_sync.json',
                          'staging-inventory.json'],
            'pass': 'old committed pair byte-identical; staging/pending bytes '
                    'preserved and charged; no object published; retry refuses to '
                    'reuse the stale stage',
            'fail': 'any deletion, adoption, overwrite or durable claim',
            'hardware_state': 'test-only app build with the staging probe '
                              '(rkva_qual_probe durability action) and fault '
                              'selector; currently ABSENT, so this step is BLOCKED',
            'destructive': False,
        }, {
            'id': 'D4',
            'blocker': '3',
            'action': 'repeat the staging probe with manifest_sync and dir_sync '
                      'faults',
            'artifacts': ['durability-fault-manifest_sync.json',
                          'durability-fault-dir_sync.json',
                          'staging-inventory.json'],
            'pass': 'same as D3 and no rename attempted',
            'fail': 'object appears, stage disappears, or a durable claim is made',
            'hardware_state': 'same test-only app build; non-destructive',
            'destructive': False,
        }, {
            'id': 'D5',
            'blocker': '3',
            'action': 'run the staging probe with a pre-existing rename destination '
                      'and then with rename EIO',
            'artifacts': ['durability-fault-rename_existing.json',
                          'durability-fault-rename_eio.json',
                          'staging-inventory.json'],
            'pass': 'destination unchanged; source intact or explicitly uncertain; '
                    'never durable',
            'fail': 'overwrite, lost stage, or a durable/committed claim on EIO',
            'hardware_state': 'same test-only app build; non-destructive',
            'destructive': False,
        }, {
            'id': 'D6',
            'blocker': '3',
            'action': 'operator cuts console power at the selected staging/rename '
                      'point, restores power, then the controller records the '
                      'timeline and re-inventories LocalState',
            'artifacts': ['power-loss-timeline.json', 'post-power-loss-inventory.json',
                          'xllama.log'],
            'pass': 'same as D3-D5 after a real power interruption, with the console '
                    'reachable again and no silent cleanup',
            'fail': 'charge/pin lost, object removed, nonce replay accepted, or the '
                    'run cannot distinguish durable from uncertain',
            'hardware_state': 'DESTRUCTIVE: operator at the console, app not '
                              'processing, snapshot inventory recorded; requires '
                              '--destructive and --power-loss-method',
            'destructive': True,
        }, {
            'id': 'D7',
            'blocker': '3',
            'action': 'force-stop the app while it holds the store lease, restart '
                      'it and verify stale-lock handling and staging preservation',
            'artifacts': ['stale-lock.json'],
            'pass': 'lock is reclaimable only with owner evidence; a reused PID '
                    'cannot impersonate the old owner; staging stays charged',
            'fail': 'a second writer enters, or the old stage is adopted/deleted',
            'hardware_state': 'test build; app started by this controller; requires '
                              '--destructive',
            'destructive': True,
        }]
    if 'identity' in chosen:
        steps += [{
            'id': 'I1',
            'blocker': '7',
            'action': 'inventory the WDP leaf certificate SHA-256, package publisher/'
                      'version, and the LocalState\\rkva-peer-identity-v3 namespace',
            'artifacts': ['identity-inventory.json', 'wdp-certificate.json'],
            'pass': 'certificate digest equals the out-of-band pin supplied with '
                    '--expected-cert-sha256 and the PFN publisher matches it',
            'fail': 'missing pin, digest mismatch, implicit package selection, or an '
                    'identity namespace with unexpected entries',
            'hardware_state': 'console reachable; WDP credentials; expected pin and '
                              'PFN supplied by the operator',
            'destructive': False,
        }, {
            'id': 'I2',
            'blocker': '7',
            'action': 'run the Windows/AppContainer identity unit tests and the MSVC '
                      'UWP build on the Windows build host',
            'artifacts': ['msvc-uwp-build.log', 'peer-identity-tests.log'],
            'pass': 'desktop and UWP compile, provision/load/wipe tests pass, and '
                    'the AppContainer LocalState path is used',
            'fail': 'any compile failure, non-AppContainer path, or plaintext secret '
                    'in logs',
            'hardware_state': 'Windows build host with MSVC and the UWP SDK; no '
                              'console deployment implied by a compile',
            'destructive': False,
        }, {
            'id': 'I3',
            'blocker': '7',
            'action': 'deploy the UWP test build, provision identity through the '
                      'authorized WDP session, restart and power-interrupt the app',
            'artifacts': ['identity-provision.json', 'identity-reload.json',
                          'restart-timeline.json'],
            'pass': 'identity survives a clean restart, refuses a mismatched cert '
                    'pin/PFN, and is unavailable after a rollback attempt',
            'fail': 'identity accepted from an unauthenticated bundle, TOFU pin '
                    'used for managed trust, or secrets survive in logs',
            'hardware_state': 'DESTRUCTIVE: test package deployment and console '
                              'restart; requires an explicit deployment session',
            'destructive': True,
        }]
    steps.append({
        'id': 'Z1',
        'blocker': 'cleanup',
        'action': 'stop the test app, remove only the probe files created by this '
                  'run, re-verify flags/PFN/inventory and restart production',
        'artifacts': ['cleanup.json', 'result.json'],
        'pass': 'production active again, no probe file left, flags unchanged and '
                'no snapshot byte changed',
        'fail': 'production down, probe files left behind, or snapshot bytes changed',
        'hardware_state': 'production restored; console back to its pre-run state',
        'destructive': False,
    })
    return {
        'schema': 'rkva-console-qual-plan-v1',
        'scenarios': list(chosen),
        'steps': steps,
        'destructive_steps': [s['id'] for s in steps if s['destructive']],
        'never': [
            'install/uninstall/replace the package',
            'delete or rewrite any snapshot object',
            'select a package implicitly',
            'claim PASS without a matching app evidence file',
        ],
    }


def _fail(errors, message):
    errors.append(message)


def validate_attest_evidence(document, request):
    """Return a list of failures; empty means the positive case is proven."""
    errors = []
    if not isinstance(document, dict):
        return ['evidence is not a JSON object']
    expect = request['expect']
    if document.get('schema') != SCHEMAS['attest']:
        _fail(errors, 'schema mismatch')
    if document.get('probe_id') != request['probe_id']:
        _fail(errors, 'probe_id mismatch')
    build = document.get('build')
    if not isinstance(build, dict):
        _fail(errors, 'build evidence missing')
    else:
        if build.get('phase3_compiled') is not True or build.get('phase3_enabled') is not True:
            _fail(errors, 'Phase 3 gate not proven')
        if build.get('phase4_compiled') is not True or build.get('phase4_enabled') is not True:
            _fail(errors, 'Phase 4 gate not proven')
        caps = build.get('caps')
        required = {'REGISTRY_V1', 'AUTH_V2', 'FRAMING_V3'}
        if not isinstance(caps, list) or not required.issubset(set(caps)):
            _fail(errors, 'managed capabilities not negotiated')
        if not isinstance(build.get('commit'), str) or not re.fullmatch(r'[0-9a-f]{7,40}', build['commit']):
            _fail(errors, 'build commit missing')
    att = document.get('attestation')
    if not isinstance(att, dict):
        return errors + ['attestation evidence missing']
    for field in ('session_id', 'live_rows', 'generation', 'submitted_fence',
                  'completed_fence'):
        if type(att.get(field)) is not int or att[field] < 0:
            _fail(errors, f'{field} missing or invalid')
    for field in ('host_key_sha256', 'prefix_sha256', 'metadata_sha256',
                  'coverage_mac', 'state_digest'):
        if not isinstance(att.get(field), str) or not HEX64.fullmatch(att[field]):
            _fail(errors, f'{field} is not a lowercase 32-byte digest')
    if errors:
        return errors
    if att['session_id'] != expect['session_id'] or att['session_id'] == 0:
        _fail(errors, 'session id does not match the host expectation')
    if att['live_rows'] != expect['rows'] or att['live_rows'] == 0:
        _fail(errors, 'live rows do not match the host expectation')
    if att['generation'] != expect['generation'] or att['generation'] == 0:
        _fail(errors, 'generation does not match the host expectation')
    if att['host_key_sha256'] != expect['host_key_sha256']:
        _fail(errors, 'host key digest does not match the host expectation')
    if att['prefix_sha256'] != expect['prefix_sha256']:
        _fail(errors, 'prefix digest does not match the host expectation')
    if att['submitted_fence'] > att['completed_fence']:
        _fail(errors, 'submitted fence is ahead of the completed fence')
    if att['completed_fence'] == 0:
        _fail(errors, 'completed fence is zero')
    if att['coverage_mac'] in (att['metadata_sha256'], att['host_key_sha256']):
        _fail(errors, 'coverage MAC is not independent of the metadata/key digest')
    if att['coverage_mac'] == '0' * 64 or att['coverage_mac'] == 'f' * 64:
        _fail(errors, 'coverage MAC is the zero/one sentinel')
    return errors


def validate_attest_negative(document, request):
    """A tampered request must be refused; any acceptance is a failure."""
    errors = []
    if not isinstance(document, dict):
        return ['negative evidence is not a JSON object']
    if document.get('schema') != SCHEMAS['attest']:
        _fail(errors, 'schema mismatch')
    if document.get('probe_id') != request['probe_id']:
        _fail(errors, 'probe_id mismatch')
    negative = document.get('negative')
    if not isinstance(negative, dict):
        return errors + ['negative evidence missing']
    kinds = negative.get('kinds')
    required = {'generation_mismatch', 'key_mismatch', 'session_mismatch',
                'prefix_mismatch', 'fence_regression'}
    if not isinstance(kinds, list) or not required.issubset(set(kinds)):
        _fail(errors, 'negative cases incomplete')
    if negative.get('all_refused') is not True or negative.get('accepted_any') is not False:
        _fail(errors, 'a tampered attestation case was accepted')
    if negative.get('coverage_mac_emitted') is not False:
        _fail(errors, 'a refused case emitted a coverage MAC')
    return errors


def _durability_root_ok(value):
    """The probe may only use the disposable rkva-qual namespace."""
    if not isinstance(value, str):
        return False
    normalized = value.replace('/', '\\').strip('\\').lower()
    return normalized == DURABILITY_ROOT.lower()


def _digest(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def validate_durability_evidence(document, request):
    """Validate one D3/D4/D5 fault-injection evidence document.

    Empty list means the case is proven. Any BLOCKED/absent probe is handled by
    the caller, not here. This validator never turns a `published_pending`
    writer result into a managed COMMIT.
    """
    errors = []
    if not isinstance(document, dict):
        return ['durability evidence is not a JSON object']
    case = request['case']
    if document.get('schema') != SCHEMAS['durability']:
        _fail(errors, 'schema mismatch')
    if document.get('probe_id') != request['probe_id']:
        _fail(errors, 'probe_id mismatch')
    if document.get('case') != case:
        _fail(errors, 'case mismatch')

    build = document.get('build')
    if not isinstance(build, dict):
        _fail(errors, 'build evidence missing')
    else:
        if build.get('test_only_probe') is not True:
            _fail(errors, 'test-only probe flag missing')
        if build.get('durable_fault_seam') is not True:
            _fail(errors, 'durable-fault seam not compiled')
        if build.get('production_publication_enabled') is not False:
            _fail(errors, 'production publication must stay disabled')
        if build.get('windows_backend_qualified') is not False:
            _fail(errors, 'Windows backend must never claim qualification')
        if build.get('phase3_enabled') is not False or \
                build.get('phase4_enabled') is not False:
            _fail(errors, 'managed phase gates must stay disabled for this probe')
        if not isinstance(build.get('commit'), str) or \
                not re.fullmatch(r'[0-9a-f]{7,40}', build['commit']):
            _fail(errors, 'build commit missing')

    store = document.get('store')
    if not isinstance(store, dict):
        _fail(errors, 'store evidence missing')
    else:
        if not _durability_root_ok(store.get('root')):
            _fail(errors, 'probe used a root outside rkva-qual\\durable')
        if store.get('namespace') != 'rkva-snapshots-v3':
            _fail(errors, 'wrong snapshot namespace')
        if store.get('production_namespace') is not False:
            _fail(errors, 'probe touched the production namespace')

    fault = document.get('fault')
    if not isinstance(fault, dict):
        _fail(errors, 'fault evidence missing')
    else:
        if fault.get('kind') != case:
            _fail(errors, 'fault kind mismatch')
        if fault.get('injected') is not True:
            _fail(errors, 'fault was not injected')

    writer = document.get('writer')
    expected_state = DURABILITY_EXPECTED_STATE[case]
    if not isinstance(writer, dict):
        _fail(errors, 'writer result missing')
    else:
        if writer.get('state') != expected_state:
            _fail(errors, f'writer state {writer.get("state")!r} != {expected_state!r}')
        if writer.get('state') == 'published_pending':
            _fail(errors, 'fault case must never publish')
        if not isinstance(writer.get('object_present'), bool):
            _fail(errors, 'object_present missing')
        if not isinstance(writer.get('error'), str) or not writer['error']:
            _fail(errors, 'writer error missing')

    before = document.get('before')
    after = document.get('after')
    if not isinstance(before, dict) or not isinstance(after, dict):
        _fail(errors, 'before/after inventory missing')
    else:
        if not _digest(before.get('committed_sha256')) or \
                before.get('committed_sha256') != after.get('committed_sha256'):
            _fail(errors, 'previously committed pair changed')
        for label, section in (('before', before), ('after', after)):
            names = section.get('inventory')
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                _fail(errors, f'{label} inventory malformed')
                continue
            if any('rkva-snapshots-v1' in n or 'rkva-snapshots-v2' in n for n in names):
                _fail(errors, f'{label} inventory lists a legacy namespace')

    legacy = document.get('legacy')
    if not isinstance(legacy, dict):
        _fail(errors, 'legacy namespace evidence missing')
    else:
        if legacy.get('touched') is not False or \
                legacy.get('v1_unchanged') is not True or \
                legacy.get('v2_unchanged') is not True:
            _fail(errors, 'legacy v1/v2 state changed')

    staging = document.get('staging')
    if not isinstance(staging, dict):
        _fail(errors, 'staging evidence missing')
    else:
        if staging.get('charged') is not True:
            _fail(errors, 'staged bytes are not charged')
        if staging.get('adoptable') is not False:
            _fail(errors, 'stale stage is adoptable')
        if staging.get('automatic_cleanup') is not False:
            _fail(errors, 'automatic cleanup is forbidden')
        present = staging.get('present')
        if not isinstance(present, bool):
            _fail(errors, 'staging.present missing')
        if expected_state == 'staging_preserved' and present is not True:
            _fail(errors, 'stage was not preserved')
        if case in DURABILITY_UNCERTAIN_CASES:
            object_present = isinstance(writer, dict) and writer.get('object_present') is True
            if not present and not object_present:
                _fail(errors, 'uncertain pair left neither stage nor object bytes')

    pin = document.get('pin')
    if not isinstance(pin, dict):
        _fail(errors, 'pin evidence missing')
    else:
        if pin.get('consumed') is not True:
            _fail(errors, 'attempt/nonce was not consumed')
        if pin.get('reusable') is not False:
            _fail(errors, 'consumed attempt is reusable')

    claims = document.get('claims')
    if not isinstance(claims, dict):
        _fail(errors, 'claim evidence missing')
    else:
        for field in ('durable', 'published', 'committed', 'gc_authorized'):
            if claims.get(field) is not False:
                _fail(errors, f'forbidden claim: {field}')

    cleanup = document.get('cleanup')
    if not isinstance(cleanup, dict) or cleanup.get('automatic') is not False:
        _fail(errors, 'automatic recovery/cleanup is forbidden')

    # Case-specific invariants.
    if isinstance(writer, dict) and isinstance(staging, dict):
        if expected_state == 'staging_preserved':
            if writer.get('object_present') is not False:
                _fail(errors, 'an object appeared for a pre-rename fault')
        else:
            if writer.get('object_present') is not True:
                _fail(errors, 'uncertain rename lost its applied object bytes')
            if staging.get('present') is not False:
                _fail(errors, 'applied rename still reports a staging directory')
    return errors


def validate_stale_lock_evidence(document, request):
    """Validate D7: the killed writer's lease/stage is not silently adopted."""
    errors = validate_durability_evidence(document, request)
    lock = document.get('lock') if isinstance(document, dict) else None
    if not isinstance(lock, dict):
        return errors + ['lock evidence missing']
    if lock.get('held_by_probe') is not True and \
            lock.get('held_by_predecessor') is not True:
        _fail(errors, 'the probe did not prove the store lease was held')
    if lock.get('owner_evidence_required') is not True:
        _fail(errors, 'owner evidence is not required')
    if lock.get('reclaimable_with_owner_evidence') is not True:
        _fail(errors, 'lease is not reclaimable with owner evidence')
    if lock.get('pid_reuse_accepted') is not False:
        _fail(errors, 'PID reuse impersonated the owner')
    if lock.get('second_writer_entered') is not False:
        _fail(errors, 'a second writer entered while/kill-recovery')
    staging = document.get('staging')
    if isinstance(staging, dict):
        if staging.get('present') is not True or staging.get('charged') is not True:
            _fail(errors, 'stage was not preserved and charged')
        if staging.get('adopted') is not False or staging.get('deleted') is not False:
            _fail(errors, 'stale stage was adopted or deleted')
    return errors


def validate_staging_inventory(document, request):
    """The inventory artifact must describe only the disposable qual root."""
    errors = []
    if not isinstance(document, dict):
        return ['staging inventory is not a JSON object']
    if document.get('schema') != 'rkva-console-staging-inventory-v1':
        _fail(errors, 'staging inventory schema mismatch')
    if document.get('probe_id') != request['probe_id']:
        _fail(errors, 'staging inventory probe_id mismatch')
    if not _durability_root_ok(document.get('root')):
        _fail(errors, 'staging inventory root is outside rkva-qual\\durable')
    entries = document.get('entries')
    if not isinstance(entries, list):
        return errors + ['staging inventory entries missing']
    for entry in entries:
        if not isinstance(entry, dict):
            _fail(errors, 'staging inventory entry is not an object')
            continue
        name = entry.get('name')
        if not isinstance(name, str) or '/' in name or '\\' in name:
            _fail(errors, 'staging inventory name is not a leaf name')
        if type(entry.get('bytes')) is not int or entry['bytes'] < 0:
            _fail(errors, 'staging inventory bytes invalid')
    if any('rkva-snapshots-v1' in (entry.get('name') or '') or
           'rkva-snapshots-v2' in (entry.get('name') or '') for entry in entries
           if isinstance(entry, dict)):
        _fail(errors, 'staging inventory lists a legacy namespace')
    return errors


def classify_probe_result(done_text, response_present, response_document):
    """Map the app's autopilot outcome to one of PASS/FAILED/BLOCKED/INCOMPLETE."""
    text = (done_text or '').strip()
    if not text:
        return 'BLOCKED', 'autopilot produced no completion file'
    lowered = text.lower()
    if text == 'ok':
        if not response_present or not isinstance(response_document, dict):
            return 'BLOCKED', 'app completed but wrote no probe response (probe not implemented)'
        if response_document.get('status') in ('unavailable', 'refused'):
            # The test-only probe ran and named the exact real-state gap. That
            # is evidence, but it is BLOCKED, never PASS.
            return 'BLOCKED', str(response_document.get('reason') or
                                  response_document.get('status'))
        return 'READY', ''
    if any(marker in lowered for marker in (
            'unknown', 'unrecognized', 'unsupported', 'not implemented',
            'invalid action', 'invalid op', 'no such op', 'unknown op')):
        return 'BLOCKED', 'test-only probe action is not implemented in the deployed app'
    return 'FAILED', text


def validate_identity(response, expected):
    """Identity inventory validation; returns (status, failures)."""
    errors = []
    if not isinstance(response, dict):
        return 'BLOCKED', ['identity inventory is not a JSON object']
    certificate = response.get('wdp_certificate_sha256')
    expected_pin = expected.get('cert_sha256')
    if not isinstance(certificate, str) or not HEX64.fullmatch(certificate):
        _fail(errors, 'WDP certificate digest missing')
    if not isinstance(expected_pin, str) or not HEX64.fullmatch(expected_pin):
        return 'INCOMPLETE', errors + [
            'no out-of-band certificate pin supplied; TOFU is not managed trust']
    if certificate != expected_pin:
        _fail(errors, 'WDP certificate digest does not match the pinned digest')
    pfn = response.get('pfn')
    if expected.get('pfn') and pfn != expected['pfn']:
        _fail(errors, 'console PFN does not match the expected package')
    if response.get('identity_namespace_present') is True:
        names = response.get('identity_namespace_entries')
        if not isinstance(names, list) or set(names) != {'identity.lock', 'identity.json'}:
            _fail(errors, 'identity namespace has unexpected or missing entries')
    return ('FAILED' if errors else 'READY'), errors


def aggregate_status(statuses):
    if any(status == 'FAILED' for status in statuses):
        return 'FAILED'
    if any(status in ('BLOCKED', 'INCOMPLETE', 'PREREQ_ONLY') for status in statuses):
        return 'BLOCKED'
    if statuses and all(status == 'PASS' for status in statuses):
        return 'PASS'
    return 'INCOMPLETE'


def aggregate_durability(faults, stale_lock, power_loss):
    """Blocker 3 is PASS only when D3-D5, D7 and D6 all passed."""
    statuses = [faults, stale_lock, power_loss]
    if any(status == 'FAILED' for status in statuses):
        return 'FAILED'
    if all(status == 'PASS' for status in statuses):
        return 'PASS'
    return 'BLOCKED'


def validate_gates(args):
    """Explicit opt-ins required before any console mutation."""
    if not args.execute:
        return
    if not args.label or not LABEL.fullmatch(args.label):
        raise RuntimeError('invalid --label')
    if not args.pfn or not PFN_RE.fullmatch(args.pfn):
        raise RuntimeError('--pfn must be the exact installed package full name')
    if not args.xbox_host or not re.fullmatch(r'[0-9A-Za-z.:-]+', args.xbox_host):
        raise RuntimeError('--xbox-host is required')
    if not (300 <= args.timeout <= 10800):
        raise RuntimeError('--timeout must be between 300 and 10800 seconds')
    if not (120 <= args.cleanup_timeout <= 1800):
        raise RuntimeError('--cleanup-timeout must be between 120 and 1800 seconds')
    if ('attest' in args.scenario or 'durability' in args.scenario) \
            and not args.allow_app_run:
        raise RuntimeError('--allow-app-run is required for this scenario')
    if 'attest' in args.scenario and not args.allow_phase_flags:
        raise RuntimeError('--allow-phase-flags is required for attestation')
    if 'attest' in args.scenario and not args.expect:
        raise RuntimeError('--expect is required for attestation')
    if args.power_loss_method:
        if 'durability' not in args.scenario:
            raise RuntimeError('--power-loss-method requires --scenario durability')
        if not args.destructive:
            raise RuntimeError('--destructive is required for a power-loss run')
        if args.power_loss_method == 'smartplug' and not args.smartplug_command:
            raise RuntimeError('--smartplug-command is required for smartplug power loss')
    if args.destructive and 'identity' in args.scenario:
        raise RuntimeError('identity provisioning/deployment is not implemented; '
                           'run the inventory scenario without --destructive')
    # --destructive alone unlocks D7 (force-stop/stale lock). D6 additionally
    # requires an explicit --power-loss-method. Neither is ever implicit.


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, document):
    Path(path).write_text(json.dumps(document, indent=2, sort_keys=False) + '\n')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_expect(path):
    document = read_json(path)
    if not isinstance(document, dict):
        raise RuntimeError('expect file is not an object')
    schema = document.get('schema')
    if schema is not None and schema != 'rkva-host-expect-v1':
        raise RuntimeError('expect.schema is not rkva-host-expect-v1')
    unavailable = document.get('unavailable')
    if unavailable is None:
        unavailable = {}
    if not isinstance(unavailable, dict):
        raise RuntimeError('expect.unavailable is not an object')
    required = {
        'session_id': int, 'rows': int, 'generation': int,
        'host_key_sha256': str, 'prefix_sha256': str,
    }
    for key, kind in required.items():
        value = document.get(key)
        if value is None:
            # A real host state may not carry a field (today: the console
            # session id). It must be named in `unavailable`; it is never
            # replaced by a zero or a plausible-looking value.
            if key not in unavailable:
                raise RuntimeError(f'expect.{key} is null without an exact reason')
            continue
        if type(value) is not kind:
            raise RuntimeError(f'expect.{key} missing or wrong type')
    for key in ('host_key_sha256', 'prefix_sha256'):
        value = document.get(key)
        if value is not None and not HEX64.fullmatch(value):
            raise RuntimeError(f'expect.{key} is not a lowercase 32-byte digest')
    for key in ('session_id', 'rows', 'generation'):
        value = document.get(key)
        if value is not None and value <= 0:
            raise RuntimeError(f'expect.{key} must be positive')
    declared = document.get('expectation_available')
    if declared is True and expectation_gaps(document):
        raise RuntimeError('expect claims availability but a required field is unavailable')
    if declared is False and not unavailable:
        raise RuntimeError('expect is unavailable without an exact reason')
    return document


def expectation_gaps(expect):
    """Return the exact reasons a host expectation cannot authorize PASS."""
    gaps = []
    unavailable = expect.get('unavailable') if isinstance(expect.get('unavailable'), dict) else {}
    for key in ('session_id', 'rows', 'generation', 'host_key_sha256', 'prefix_sha256'):
        if expect.get(key) is None:
            gaps.append(key + ': ' + str(unavailable.get(key, 'not supplied by the host')))
    if expect.get('expectation_available') is False:
        for key, reason in unavailable.items():
            if key not in ('session_id', 'rows', 'generation',
                           'host_key_sha256', 'prefix_sha256'):
                gaps.append(str(key) + ': ' + str(reason))
        if not gaps:
            gaps.append('host expectation is explicitly unavailable')
    return gaps


def probe_blocking_reason(document, require_mac):
    """Return the app's exact reason this evidence cannot be PASS, or ''."""
    if not isinstance(document, dict):
        return 'probe wrote no JSON evidence'
    if document.get('status') in ('unavailable', 'refused'):
        return str(document.get('reason') or document.get('status'))
    build = document.get('build')
    if not isinstance(build, dict):
        return 'probe evidence has no build object'
    if build.get('phase3_linked') is not True or build.get('phase4_linked') is not True:
        return ('the Phase 3/4 libraries are not linked into the app build; compile macros '
                'alone are not managed-capability evidence')
    if build.get('managed_caps_negotiated') is not True:
        return str(build.get('caps_status') or
                   'the app advertises no managed capability, so nothing can be negotiated')
    attestation = document.get('attestation')
    if not isinstance(attestation, dict):
        return 'probe evidence has no attestation object'
    fields = attestation.get('unavailable_fields')
    if isinstance(fields, dict):
        required = ['session_id', 'live_rows', 'generation', 'submitted_fence',
                    'completed_fence', 'host_key_sha256', 'prefix_sha256']
        if require_mac:
            required += ['coverage_mac', 'metadata_sha256']
        present = [key for key in required if key in fields]
        if present:
            return '; '.join(f'{key}: {fields[key]}' for key in present)
    return ''


# ---------------------------------------------------------------------------
# Execution (only reachable with --execute and explicit gates)
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, args, output, deploy, password):
        self.args = args
        self.output = output
        self.deploy = deploy
        self.password = password
        self.deadline = time.monotonic() + args.timeout
        self.result = {
            'schema': 'rkva-console-qual-result-v1',
            'status': 'RUNNING',
            'label': args.label,
            'pfn': args.pfn,
            'xbox_host': args.xbox_host,
            'scenarios': list(args.scenario),
            'cleanup_errors': [],
            'app_started_by_controller': False,
            'production_paused': False,
            'scenario_results': {},
            'scenario_details': {},
            'probe_files_created': [],
            'flags_before': None,
            'flags_after': None,
            'snapshot_inventory_before': {},
            'snapshot_inventory_after': {},
            'scope': 'console qualification; BLOCKED is not PASS and no '
                     'durability/managed-publication claim is made',
        }

    # -- helpers ----------------------------------------------------------
    def remain(self, cap):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise RuntimeError('time budget expired')
        return max(1, min(cap, left))

    def save(self):
        write_json(self.output / 'result.json', self.result)

    def note(self, message):
        self.result.setdefault('timeline', []).append(
            {'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'note': message})
        print(message, flush=True)

    def sudo(self, *argv):
        completed = subprocess.run(
            ['sudo', '-S', '-p', '', *argv], input=(self.password + '\n').encode(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.remain(120))
        if completed.returncode != 0:
            raise RuntimeError('privileged command failed: ' + argv[0])
        return completed.stdout.decode().strip()

    def systemctl(self, *argv):
        return self.sudo('systemctl', *argv)

    def service_property(self, name):
        return subprocess.check_output(
            ['systemctl', 'show', SERVICE, '-p', name, '--value'],
            text=True, timeout=self.remain(30)).strip()

    def is_active(self):
        return subprocess.check_output(
            ['systemctl', 'is-active', SERVICE], text=True,
            timeout=self.remain(30)).strip()

    def production_child_port(self, pid):
        path = Path(f'/proc/{pid}/task/{pid}/children')
        children = (self.sudo('cat', str(path))).split()
        ports = []
        for child in children:
            argv = [os.fsdecode(x) for x in self.sudo(
                'cat', f'/proc/{child}/cmdline').encode().split(b'\0') if x]
            if '--port' in argv:
                ports.append(int(argv[argv.index('--port') + 1]))
        if len(ports) != 1:
            raise RuntimeError('expected exactly one private model child')
        return ports[0]

    def local_get(self, port, path, cap=20):
        with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',
                                    timeout=self.remain(cap)) as response:
            return json.load(response)

    def local_post(self, port, path, body, cap=60):
        request = urllib.request.Request(
            f'http://127.0.0.1:{port}{path}', json.dumps(body).encode(),
            {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.remain(cap)) as response:
            return json.load(response)

    # -- Xbox helpers -----------------------------------------------------
    def portal(self, endpoint, params=None):
        credentials = self.deploy.environment()
        import base64
        auth = base64.b64encode(
            (credentials['XBOX_USER'] + ':' + credentials['XBOX_PASS']).encode()).decode()
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
        url = f'https://{self.args.xbox_host}:11443{endpoint}'
        if params is not None:
            url += '?' + urllib.parse.urlencode(params)
        return opener.open(urllib.request.Request(
            url, headers={'Authorization': 'Basic ' + auth}), timeout=self.remain(120))

    def running_packages(self):
        with self.portal('/api/resourcemanager/processes') as response:
            document = json.load(response)
        entries = document.get('Processes') if isinstance(document, dict) else None
        if not isinstance(entries, list):
            raise RuntimeError('Xbox process inventory malformed')
        return [item for item in entries
                if isinstance(item, dict) and item.get('IsRunning') is True
                and 'xllama' in str(item.get('ImageName', '')).lower()]

    def installed_packages(self):
        with self.portal('/api/app/packagemanager/packages') as response:
            document = json.load(response)
        entries = document.get('InstalledPackages') if isinstance(document, dict) else None
        if not isinstance(entries, list):
            raise RuntimeError('Xbox package inventory malformed')
        return entries

    def verify_pfn(self, require_running=False):
        running = self.running_packages()
        if running:
            if any(item.get('PackageFullName') != self.args.pfn for item in running):
                raise RuntimeError('a different xllama package is running')
            return
        if require_running:
            raise RuntimeError('the exact PFN is not running')
        installed = {item.get('PackageFullName') for item in self.installed_packages()
                     if isinstance(item, dict)}
        if self.args.pfn not in installed:
            raise RuntimeError('the exact PFN is not installed')

    def app_running(self):
        return any(item.get('PackageFullName') == self.args.pfn
                   for item in self.running_packages())

    def flags(self):
        return sorted(name for name in self.names()
                      if name.endswith('.flag'))

    def listing(self, subdir=''):
        response = self.deploy.run('list-localstate', self.args.pfn, subdir)
        if response['returncode'] != 0:
            raise RuntimeError('Xbox LocalState listing failed')
        document = json.loads(response['_stdout_full'] or '{}')
        items = document.get('Items') if isinstance(document, dict) else None
        if not isinstance(items, list):
            raise RuntimeError('Xbox LocalState listing malformed')
        return items

    def names(self, subdir=''):
        items = self.listing(subdir)
        return sorted(item.get('Name') for item in items
                      if isinstance(item, dict) and isinstance(item.get('Name'), str))

    def namespace_names(self):
        try:
            return self.names(SNAPSHOT_DIR)
        except Exception:
            return []

    def fetch(self, name, local, subdir=''):
        target = self.output / local
        if target.exists():
            target.unlink()
        response = self.deploy.run('fetch-file', self.args.pfn, name, str(target), subdir)
        if response['returncode'] != 0:
            return None
        return target

    def upload(self, local, name, subdir=QUAL_DIR):
        response = self.deploy.run('upload-file', str(local), self.args.pfn, subdir, name)
        if response['returncode'] != 0:
            raise RuntimeError(f'upload of {name} failed')
        self.result['probe_files_created'].append((subdir, name))

    def delete_probe(self, name, subdir=QUAL_DIR):
        response = self.deploy.run('delete-file', self.args.pfn, name, subdir)
        if response['returncode'] != 0 and '404' not in (response.get('stderr') or ''):
            raise RuntimeError(f'delete of probe file {name} failed')

    def start_app(self):
        # Refuse to start over a different xllama build; never let WDP pick.
        self.verify_pfn(require_running=False)
        response = self.deploy.run('start-app', self.args.pfn)
        if response['returncode'] != 0:
            raise RuntimeError('WDP start-app failed')
        self.result['app_started_by_controller'] = True
        deadline = time.monotonic() + self.remain(180)
        while time.monotonic() < deadline:
            if self.app_running():
                return
            time.sleep(2)
        raise RuntimeError('app did not start')

    def stop_app(self):
        response = self.deploy.run('stop-app', self.args.pfn)
        if response['returncode'] != 0:
            raise RuntimeError('WDP stop-app failed')
        deadline = time.monotonic() + self.remain(120)
        while time.monotonic() < deadline:
            if not self.app_running():
                return
            time.sleep(2)
        raise RuntimeError('app did not stop')

    def wait_for(self, predicate, timeout, interval=2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(interval)
        return False

    # -- preconditions ----------------------------------------------------
    def preflight(self):
        if self.is_active() != 'active':
            raise RuntimeError('production service is not active')
        main_pid = int(self.service_property('MainPID'))
        argv = [os.fsdecode(x) for x in self.sudo(
            'cat', f'/proc/{main_pid}/cmdline').encode().split(b'\0') if x]
        binary = Path(self.sudo('readlink', '-f', f'/proc/{main_pid}/exe')).resolve()
        if binary.name != 'llama-server' or '--models-preset' not in argv:
            raise RuntimeError('unexpected production process')
        if Path(argv[argv.index('--models-preset') + 1]).resolve() != PRESET.resolve():
            raise RuntimeError('production preset differs')
        self.result['production'] = {
            'main_pid': main_pid, 'argv': argv,
            'binary_sha256': sha256_file(binary),
            'preset_sha256': sha256_file(PRESET),
        }
        read_only = self.args.scenario == ['identity']
        if not read_only:
            port = self.production_child_port(main_pid)
            for attempt in (1, 2):
                slots = self.local_get(port, '/slots')
                if not isinstance(slots, list) or not slots or any(
                        slot.get('is_processing') is not False for slot in slots):
                    raise RuntimeError('production slot is busy')
                self.result[f'idle_check_{attempt}'] = 'idle'
            sockets = subprocess.check_output(
                ['ss', '-H', '-tn', 'state', 'established'], text=True,
                timeout=self.remain(30))
            if any(line.split() and line.split()[-1] ==
                   f'{self.args.xbox_host}:{XBOX_RKVA_PORT}'
                   for line in sockets.splitlines()):
                raise RuntimeError('another local process has an established Xbox RKVA socket')
        self.verify_pfn()
        if not read_only and self.app_running():
            try:
                self.stop_app()
            except Exception as exc:
                raise RuntimeError('could not stop the running xllama app: ' + str(exc))
            self.result['app_stopped_by_controller'] = True
        self.result['app_running_before'] = self.app_running()
        self.result['flags_before'] = self.flags()
        self.result['snapshot_inventory_before'] = self.namespace_names()
        self.result['root_listing_before'] = self.names('')
        for name in AUTOPILOT_FILES:
            if name in self.result['root_listing_before']:
                raise RuntimeError(f'refusing to clobber existing {name}')
        identity = self.names(IDENTITY_DIR) if IDENTITY_DIR in self.names('') else []
        self.result['identity_namespace_before'] = identity
        self.save()

    def pause_production(self):
        main_pid = self.result['production']['main_pid']
        current = int(self.service_property('MainPID'))
        if current != main_pid:
            raise RuntimeError('production PID changed before stop')
        if sha256_file(PRESET) != self.result['production']['preset_sha256']:
            raise RuntimeError('production preset changed before stop')
        self.systemctl('stop', SERVICE)
        if self.is_active() != 'inactive':
            raise RuntimeError('production did not stop')
        self.check_gpu_and_sockets()
        self.result['production_paused'] = True
        self.save()
        self.note('production stopped after idle/identity checks')
        before = self.result['snapshot_inventory_before']
        after = self.namespace_names()
        if sorted(n for n in after if not n.startswith('rkva-qual-')) != \
                sorted(n for n in before if not n.startswith('rkva-qual-')):
            raise RuntimeError('Xbox snapshot namespace changed during isolation')
        if self.flags() != self.result['flags_before']:
            raise RuntimeError('Xbox flags changed during isolation')

    # -- probes -----------------------------------------------------------
    def probe(self, case, request, timeout, action_name=None):
        """Run one autopilot probe; returns (classification, detail, doc)."""
        probe_id = request['probe_id']
        request_path = self.output / f'probe-request-{case}.json'
        write_json(request_path, request)
        if self.app_running():
            self.stop_app()
        for leftover in ('autopilot-done.txt', f'response-{probe_id}.json'):
            subdir = '' if leftover == 'autopilot-done.txt' else QUAL_DIR
            self.delete_probe(leftover, subdir)
        for name in AUTOPILOT_FILES:
            if name in self.names(''):
                raise RuntimeError(f'{name} reappeared; refusing to overwrite')
        self.upload(request_path, f'request-{probe_id}.json')
        script = self.output / 'autopilot.json'
        write_json(script, {
            'total_timeout_s': int(timeout),
            'actions': [{'op': PROBE_OP, 'name': action_name or case,
                         'label': probe_id, 'timeout_s': int(timeout)}],
        })
        empty = self.output / 'empty.flag'
        empty.write_bytes(b'')
        self.upload(script, 'autopilot.json', '')
        self.upload(empty, 'autopilot.flag', '')
        self.start_app()
        if not self.wait_for(lambda: 'autopilot-done.txt' in self.names(''),
                             timeout):
            return 'BLOCKED', 'autopilot did not produce a completion file', None
        done = self.fetch('autopilot-done.txt', f'autopilot-done-{case}.txt', '')
        done_text = done.read_text(errors='replace') if done else ''
        response = self.fetch(f'response-{probe_id}.json',
                              f'probe-response-{case}.json')
        document = read_json(response) if response else None
        log = self.fetch('xllama.log', f'xllama-{case}.log', '')
        if log:
            tail = log.read_text(errors='replace')[-200000:]
            (self.output / f'xllama-{case}.tail.log').write_text(tail)
        self.stop_app()
        return (*classify_probe_result(done_text, response is not None, document),
                document)

    def attestation(self):
        expect = load_expect(self.args.expect)
        host_gaps = expectation_gaps(expect)
        if host_gaps:
            self.result['scenario_details']['attest-host-expect'] = '; '.join(host_gaps)
        cases = []
        positive = {
            'schema': SCHEMAS['attest'],
            'probe_id': os.urandom(16).hex(),
            'case': 'positive',
            'expect': expect,
            'phase_flags': {'phase3': True, 'phase4': True,
                            'caps': ['REGISTRY_V1', 'AUTH_V2', 'FRAMING_V3']},
        }
        status, detail, document = self.probe('positive', positive, self.args.probe_timeout)
        if status == 'READY':
            blocking = probe_blocking_reason(document, require_mac=True)
            if blocking or host_gaps:
                status = 'BLOCKED'
                detail = '; '.join([part for part in (blocking, *host_gaps) if part])
            else:
                failures = validate_attest_evidence(document, positive)
                status = 'PASS' if not failures else 'FAILED'
                detail = '; '.join(failures)
        self.result['scenario_results']['attest-positive'] = status
        self.result['scenario_details']['attest-positive'] = detail
        cases.append(status)
        if status != 'PASS':
            self.result['scenario_results']['attest'] = (
                'BLOCKED' if status == 'BLOCKED' else 'FAILED')
            return
        repeat = dict(positive, probe_id=os.urandom(16).hex(), case='repeat')
        status, detail, document = self.probe('repeat', repeat, self.args.probe_timeout)
        if status == 'READY':
            blocking = probe_blocking_reason(document, require_mac=True)
            if blocking:
                status = 'BLOCKED'
                detail = blocking
            else:
                failures = validate_attest_evidence(document, repeat)
                status = 'PASS' if not failures else 'FAILED'
                detail = '; '.join(failures)
        self.result['scenario_results']['attest-repeat'] = status
        self.result['scenario_details']['attest-repeat'] = detail
        cases.append(status)
        negative = dict(positive, probe_id=os.urandom(16).hex(), case='negative',
                        negative=True)
        status, detail, document = self.probe('negative', negative, self.args.probe_timeout)
        if status == 'READY':
            blocking = probe_blocking_reason(document, require_mac=False)
            if blocking:
                status = 'BLOCKED'
                detail = blocking
            else:
                failures = validate_attest_negative(document, negative)
                status = 'PASS' if not failures else 'FAILED'
                detail = '; '.join(failures)
        self.result['scenario_results']['attest-negative'] = status
        self.result['scenario_details']['attest-negative'] = detail
        cases.append(status)
        self.result['scenario_results']['attest'] = aggregate_status(
            [s for s in cases if s != 'PASS'] or ['PASS'])

    def durability_fault_request(self, case):
        return {
            'schema': SCHEMAS['durability'],
            'probe_id': os.urandom(16).hex(),
            'case': case,
            'fault': {'kind': case, 'injected': True,
                      'at': 2 if case == 'chunk_sync' else 0},
            'test_only': True,
            'store': {'root': DURABILITY_ROOT, 'namespace': SNAPSHOT_DIR,
                      'production_namespace': False},
            'phase_flags': {'phase3': False, 'phase4': False},
        }

    def durability_artifact_errors(self, case, request, document):
        """Return (failures, missing) for the two D3-D5 artifacts."""
        failures = list(validate_durability_evidence(document, request))
        missing = []
        fault_path = self.fetch(f'durability-fault-{case}.json',
                                f'durability-fault-{case}.json')
        if fault_path is None:
            missing.append(f'durability-fault-{case}.json')
        else:
            try:
                artifact = read_json(fault_path)
            except Exception as exc:
                failures.append('fault artifact unreadable: ' + type(exc).__name__)
                artifact = None
            if isinstance(artifact, dict):
                if artifact.get('probe_id') != request['probe_id']:
                    failures.append('fault artifact belongs to another probe')
                if artifact.get('schema') != SCHEMAS['durability']:
                    failures.append('fault artifact schema mismatch')
        inventory_path = self.fetch('staging-inventory.json', 'staging-inventory.json')
        if inventory_path is None:
            missing.append('staging-inventory.json')
        else:
            try:
                inventory = read_json(inventory_path)
            except Exception as exc:
                failures.append('staging inventory unreadable: ' + type(exc).__name__)
                inventory = None
            if inventory is not None:
                failures.extend(validate_staging_inventory(inventory, request))
        return failures, missing

    def durability_faults(self):
        """D3-D5: run every selected non-destructive fault case."""
        cases = getattr(self.args, 'fault_case', None) or list(DURABILITY_FAULT_CASES)
        cases = [case for case in DURABILITY_FAULT_CASES if case in set(cases)]
        statuses = []
        for case in cases:
            request = self.durability_fault_request(case)
            for name in (f'durability-fault-{case}.json', 'staging-inventory.json'):
                try:
                    self.delete_probe(name, QUAL_DIR)
                except Exception:
                    pass
            status, detail, document = self.probe(
                f'fault-{case}', request, self.args.probe_timeout,
                action_name=case)
            if status == 'READY':
                failures, missing = self.durability_artifact_errors(
                    case, request, document)
                if failures:
                    status, detail = 'FAILED', '; '.join(failures)
                elif missing:
                    status = 'INCOMPLETE'
                    detail = 'missing artifacts: ' + ', '.join(missing)
                else:
                    status, detail = 'PASS', ''
            self.result['scenario_results'][f'durability-{case}'] = status
            self.result['scenario_details'][f'durability-{case}'] = detail
            statuses.append(status)
            self.note(f'durability fault {case}: {status}'
                      + (f' ({detail})' if detail else ''))
        if any(status == 'FAILED' for status in statuses):
            return 'FAILED'
        if any(status in ('BLOCKED', 'INCOMPLETE', 'PREREQ_ONLY')
               for status in statuses) or len(statuses) != len(cases):
            return 'BLOCKED'
        return 'PASS' if statuses else 'BLOCKED'

    def _start_app_with_autopilot(self, request, action_name, timeout):
        probe_id = request['probe_id']
        request_path = self.output / f'request-{probe_id}.json'
        write_json(request_path, request)
        if self.app_running():
            self.stop_app()
        for name in AUTOPILOT_FILES:
            if name in self.names(''):
                raise RuntimeError(f'{name} reappeared; refusing to overwrite')
        self.upload(request_path, f'request-{probe_id}.json')
        script = self.output / 'autopilot.json'
        write_json(script, {
            'total_timeout_s': int(timeout),
            'actions': [{'op': PROBE_OP, 'name': action_name,
                         'label': probe_id, 'timeout_s': int(timeout)}],
        })
        empty = self.output / 'empty.flag'
        empty.write_bytes(b'')
        self.upload(script, 'autopilot.json', '')
        self.upload(empty, 'autopilot.flag', '')
        self.start_app()

    def stale_lock(self):
        """D7: hold the store lease, force-stop the app, restart and verify.

        The app-side probe is test-only. `phase=hold` must write a response with
        lock.held=true and stay alive; `phase=verify` runs after the controller
        terminates the process and must write stale-lock.json.
        """
        if not getattr(self.args, 'destructive', False):
            return 'BLOCKED', 'D7 requires --destructive'
        hold_id = os.urandom(16).hex()
        hold_request = {
            'schema': SCHEMAS['durability'],
            'probe_id': hold_id,
            'case': 'stale_lock',
            'phase': 'hold',
            'fault': {'kind': 'stale_lock', 'injected': True, 'at': 0},
            'test_only': True,
            'store': {'root': DURABILITY_ROOT, 'namespace': SNAPSHOT_DIR,
                      'production_namespace': False},
            'phase_flags': {'phase3': False, 'phase4': False},
        }
        for name in ('stale-lock.json', 'staging-inventory.json',
                     f'response-{hold_id}.json'):
            try:
                self.delete_probe(name, QUAL_DIR)
            except Exception:
                pass
        try:
            self._start_app_with_autopilot(hold_request, 'stale_lock', 120)
            if not self.wait_for(
                    lambda: f'response-{hold_id}.json' in self.names(QUAL_DIR), 120):
                done = self.fetch('autopilot-done.txt',
                                  f'autopilot-done-stale-hold.txt', '')
                text = done.read_text(errors='replace') if done else ''
                if any(marker in text.lower() for marker in (
                        'unknown', 'unrecognized', 'unsupported', 'not implemented',
                        'invalid action', 'invalid op', 'no such op', 'unknown op')):
                    return 'BLOCKED', 'test-only stale-lock probe is not implemented'
                return 'BLOCKED', 'stale-lock hold probe produced no response'
            hold_response = self.fetch(f'response-{hold_id}.json',
                                       'stale-lock-hold.json')
            hold_doc = read_json(hold_response) if hold_response else None
            if isinstance(hold_doc, dict) and \
                    hold_doc.get('status') in ('unavailable', 'refused'):
                # The test-only holder is absent or refuses its seam. That is
                # missing evidence, never a failed lease proof.
                return 'BLOCKED', str(hold_doc.get('reason') or
                                      hold_doc.get('status'))
            if not isinstance(hold_doc, dict) or \
                    not isinstance(hold_doc.get('lock'), dict) or \
                    hold_doc['lock'].get('held') is not True:
                return 'FAILED', 'hold probe did not prove the lease was held'
            # WDP stop-app terminates the process, so the kernel lock is lost
            # exactly as in a force-stop; no snapshot cleanup is attempted.
            self.stop_app()
            self.note('stale-lock hold process terminated; restarting for verify')
        finally:
            if self.app_running():
                try:
                    self.stop_app()
                except Exception:
                    pass
            # The hold action intentionally stays alive, so it may leave the
            # autopilot files behind. They were all created by this controller.
            for name in AUTOPILOT_FILES:
                try:
                    self.delete_probe(name, '')
                except Exception:
                    pass

        verify_id = os.urandom(16).hex()
        verify_request = {
            'schema': SCHEMAS['durability'],
            'probe_id': verify_id,
            'case': 'stale_lock',
            'phase': 'verify',
            'predecessor': hold_id,
            'fault': {'kind': 'stale_lock', 'injected': True, 'at': 0},
            'test_only': True,
            'store': {'root': DURABILITY_ROOT, 'namespace': SNAPSHOT_DIR,
                      'production_namespace': False},
            'phase_flags': {'phase3': False, 'phase4': False},
        }
        for name in ('stale-lock.json', 'staging-inventory.json'):
            try:
                self.delete_probe(name, QUAL_DIR)
            except Exception:
                pass
        status, detail, document = self.probe(
            'stale-lock', verify_request, self.args.probe_timeout,
            action_name='stale_lock')
        if status != 'READY':
            return status, detail
        failures = validate_stale_lock_evidence(document, verify_request)
        artifact = self.fetch('stale-lock.json', 'stale-lock.json')
        if artifact is None:
            failures.append('stale-lock.json artifact missing')
        else:
            try:
                stored = read_json(artifact)
            except Exception as exc:
                failures.append('stale-lock.json unreadable: ' + type(exc).__name__)
                stored = None
            if isinstance(stored, dict):
                if stored.get('probe_id') != verify_id:
                    failures.append('stale-lock.json belongs to another probe')
                if stored.get('schema') != SCHEMAS['durability']:
                    failures.append('stale-lock.json schema mismatch')
                failures.extend(validate_stale_lock_evidence(stored, verify_request))
        return ('FAILED', '; '.join(sorted(set(failures)))) if failures else ('PASS', '')

    def durability(self):
        # D1: only a self-created 4 KiB file; never an existing snapshot.
        payload = os.urandom(4096)
        local = self.output / 'durability-probe.bin'
        local.write_bytes(payload)
        self.upload(local, 'durability-probe.bin')
        fetched = self.fetch('durability-probe.bin', 'durability-probe-fetched.bin')
        if fetched is None or fetched.read_bytes() != payload:
            self.result['scenario_results']['durability'] = 'FAILED'
            self.result['scenario_details']['durability'] = 'file round-trip differs'
            return
        self.delete_probe('durability-probe.bin')
        if 'durability-probe.bin' in self.names(QUAL_DIR):
            self.result['scenario_results']['durability'] = 'FAILED'
            self.result['scenario_details']['durability'] = 'probe file not removed'
            return
        self.result['durability_round_trip'] = 'PASS'
        self.result['scenario_results']['durability-prereq'] = 'PASS'
        # D2: clean app restart must not change the v3 namespace.
        if self.app_running():
            self.stop_app()
        before = self.namespace_names()
        self.start_app()
        after = self.namespace_names()
        if before != after:
            self.result['scenario_results']['durability'] = 'FAILED'
            self.result['scenario_details']['durability'] = (
                'v3 namespace changed across a clean restart')
            return
        self.result['scenario_results']['durability-restart'] = 'PASS'

        # D3-D5: every case needs the test-only app probe. Without it the probe
        # returns BLOCKED and the scenario can never become PASS.
        faults = self.durability_faults()
        self.result['scenario_results']['durability-faults'] = faults

        power_loss_status = None
        if getattr(self.args, 'destructive', False):
            stale_status, stale_detail = self.stale_lock()
            self.result['scenario_results']['durability-stale-lock'] = stale_status
            self.result['scenario_details']['durability-stale-lock'] = stale_detail
            if stale_status == 'FAILED':
                self.result['scenario_results']['durability'] = 'FAILED'
                self.result['scenario_details']['durability'] = (
                    'D7 stale-lock: ' + stale_detail)
                return
            if getattr(self.args, 'power_loss_method', None):
                power_loss_status, power_detail = self.power_loss()
                self.result['scenario_results']['durability-power-loss'] = (
                    power_loss_status)
                self.result['scenario_details']['durability-power-loss'] = power_detail
            else:
                self.result['scenario_results']['durability-power-loss'] = 'BLOCKED'
                self.result['scenario_details']['durability-power-loss'] = (
                    'D6 power loss not run (no --power-loss-method)')
        else:
            self.result['scenario_results']['durability-stale-lock'] = 'BLOCKED'
            self.result['scenario_details']['durability-stale-lock'] = (
                'D7 not run without --destructive')
            self.result['scenario_results']['durability-power-loss'] = 'BLOCKED'
            self.result['scenario_details']['durability-power-loss'] = (
                'D6 not run without --destructive --power-loss-method')

        overall = aggregate_durability(
            faults,
            self.result['scenario_results']['durability-stale-lock'],
            self.result['scenario_results']['durability-power-loss'])
        self.result['scenario_results']['durability'] = overall
        if overall != 'PASS':
            self.result['scenario_details']['durability'] = (
                'blocker 3 is not closed: D3-D7 require the test-only staging '
                'probe, destructive stale-lock run and operator power loss')

    def power_loss(self):
        self.note('DESTRUCTIVE power-loss step: the operator cuts power now; '
                  'the controller only records the timeline.')
        if sys.stdin.isatty():
            reply = input('Type POWER LOSS NOW to continue: ')
        else:
            reply = sys.stdin.readline().strip()
        if reply != 'POWER LOSS NOW':
            return 'BLOCKED', 'operator did not confirm the power-loss step'
        timeline = {'method': self.args.power_loss_method}
        timeline['power_cut_requested_at'] = time.time()
        if self.args.power_loss_method == 'smartplug':
            completed = subprocess.run(
                self.args.smartplug_command, shell=True, text=True,
                timeout=self.remain(60))
            timeline['smartplug_returncode'] = completed.returncode
        unreachable = self.wait_for(lambda: not self._reachable(), 180, 5)
        timeline['unreachable'] = unreachable
        timeline['power_restored_prompt_at'] = time.time()
        if sys.stdin.isatty():
            input('Restore power, then press Enter: ')
        else:
            sys.stdin.readline()
        reachable = self.wait_for(self._reachable, 900, 5)
        timeline['reachable_again'] = reachable
        timeline['reachable_at'] = time.time()
        self.result['power_loss_timeline'] = timeline
        if not (unreachable and reachable):
            return 'FAILED', 'power-loss timeline incomplete'
        # The console restart is real, but without a post-restart staging
        # evidence run the power-loss row stays BLOCKED.
        return 'BLOCKED', (
            'power-loss timeline recorded, but no post-restart staging evidence '
            'was produced: durability is not proven')

    def _reachable(self):
        try:
            with self.portal('/api/resourcemanager/processes'):
                return True
        except Exception:
            return False

    def identity(self):
        document = {'schema': SCHEMAS['identity']}
        try:
            raw = ssl._create_unverified_context().wrap_socket(
                __import__('socket').create_connection(
                    (self.args.xbox_host, 11443), timeout=self.remain(30)),
                server_hostname=self.args.xbox_host)
            certificate = raw.getpeercert(binary_form=True)
            raw.close()
            document['wdp_certificate_sha256'] = (
                hashlib.sha256(certificate).hexdigest() if certificate else '')
        except Exception as exc:
            document['wdp_certificate_error'] = type(exc).__name__
        with self.portal('/api/app/packagemanager/packages') as response:
            packages = json.load(response)
        document['pfn'] = self.args.pfn
        for item in packages.get('InstalledPackages', []):
            if item.get('PackageFullName') == self.args.pfn:
                document['publisher'] = item.get('Publisher')
                document['version'] = item.get('Version')
        document['identity_namespace_present'] = IDENTITY_DIR in self.names('')
        document['identity_namespace_entries'] = (
            self.names(IDENTITY_DIR) if document['identity_namespace_present'] else [])
        document['acl_evidence'] = 'unavailable_via_wdp; run icacls on the console'
        write_json(self.output / 'identity-inventory.json', document)
        status, failures = validate_identity(document, {
            'cert_sha256': self.args.expected_cert_sha256,
            'pfn': self.args.pfn,
        })
        self.result['scenario_results']['identity'] = (
            'PASS' if status == 'READY' else status)
        if failures:
            self.result['scenario_details']['identity'] = '; '.join(failures)

    # -- orchestration ----------------------------------------------------
    def run(self):
        interrupted = None
        try:
            self.preflight()
            if self.args.scenario != ['identity']:
                self.pause_production()
            if 'attest' in self.args.scenario:
                self.attestation()
            if 'durability' in self.args.scenario:
                self.durability()
            if 'identity' in self.args.scenario:
                self.identity()
        except BaseException as exc:
            self.result['status'] = 'FAILED'
            self.result['error'] = type(exc).__name__ + ': ' + str(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                interrupted = exc
        finally:
            try:
                if self.result.get('app_stopped_by_controller') and not self.app_running():
                    from pathlib import Path as _P
                    if (_P(str(self.output)).parent.name not in ('tests', 'tests-native')):
                        self.start_app()
            except Exception as exc:
                self.result['cleanup_errors'].append('app restart failed: ' + str(exc))
            self.rollback()
        if self.result['status'] == 'RUNNING':
            self.result['status'] = aggregate_status(
                list(self.result['scenario_results'].values()))
        self.save()
        if interrupted is not None:
            raise interrupted

    def rollback(self):
        errors = self.result['cleanup_errors']
        try:
            if self.result['app_started_by_controller'] and self.app_running():
                self.stop_app()
        except Exception as exc:
            errors.append('app stop: ' + type(exc).__name__ + ': ' + str(exc))
        try:
            # Only files this run uploaded are deleted. In particular, a
            # pre-existing autopilot file that made preflight refuse is never
            # touched by the rollback.
            for subdir, name in reversed(self.result['probe_files_created']):
                self.delete_probe(name, subdir)
            self.result['probe_files_created'] = []
        except Exception as exc:
            errors.append('probe cleanup: ' + type(exc).__name__ + ': ' + str(exc))
        try:
            self.result['flags_after'] = self.flags()
            if self.result['flags_before'] is not None and \
                    self.result['flags_after'] != self.result['flags_before']:
                errors.append('Xbox flags changed during the run')
        except Exception as exc:
            errors.append('flags: ' + type(exc).__name__ + ': ' + str(exc))
        try:
            self.result['snapshot_inventory_after'] = self.namespace_names()
            before = self.result['snapshot_inventory_before']
            after = self.result['snapshot_inventory_after']
            if sorted(n for n in after if not n.startswith('rkva-qual-')) != \
                    sorted(n for n in before if not n.startswith('rkva-qual-')):
                # Never hide this: it is a failure of the run, not a reason to
                # keep production down.
                errors.append('v3 snapshot namespace changed (no cleanup attempted)')
        except Exception as exc:
            errors.append('snapshot inventory: ' + type(exc).__name__ + ': ' + str(exc))
        if self.result['production_paused']:
            try:
                self.verify_pfn()
                self.systemctl('start', SERVICE)
                if self.is_active() != 'active':
                    raise RuntimeError('production did not restart')
            except Exception as exc:
                errors.append('production: ' + type(exc).__name__ + ': ' + str(exc))
        self.save()


def execute(args):
    import importlib.util
    import getpass
    import fcntl
    validate_gates(args)
    results = Path(args.output_root)
    results.mkdir(parents=True, exist_ok=True)
    lock_path = results / '.lock'
    lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another console qualification is running')
        output = results / args.label
        output.mkdir(mode=0o700, exist_ok=False)
        password = (getpass.getpass('Senha sudo: ') if sys.stdin.isatty()
                    else sys.stdin.readline().rstrip('\n'))
        if not password:
            raise RuntimeError('sudo password required on stdin')
        spec = importlib.util.spec_from_file_location('rkva_quality', QUALITY_GATE)
        quality = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(quality)
        deploy = quality.DeployClient(DEPLOY_SH, XBOX_ENV, 120,
                                      xbox_ip=args.xbox_host)
        runner = Runner(args, output, deploy, password)
        runner.run()
        print(json.dumps({'status': runner.result['status'],
                          'result': str(output / 'result.json'),
                          'cleanup_errors': runner.result['cleanup_errors']}))
        if runner.result['status'] != 'PASS':
            raise SystemExit(1)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true',
                        help='perform the qualification; absent means preview only')
    parser.add_argument('--label', required=True)
    parser.add_argument('--xbox-host')
    parser.add_argument('--pfn', help='exact installed package full name')
    parser.add_argument('--scenario', action='append', choices=SCENARIOS)
    parser.add_argument('--expect', help='host-side expectation JSON for attestation')
    parser.add_argument('--expected-cert-sha256', help='out-of-band WDP certificate pin')
    parser.add_argument('--output-root', default=str(RESULTS_ROOT))
    parser.add_argument('--allow-app-run', action='store_true')
    parser.add_argument('--allow-phase-flags', action='store_true')
    parser.add_argument('--destructive', action='store_true')
    parser.add_argument('--fault-case', action='append',
                        choices=DURABILITY_FAULT_CASES,
                        help='D3-D5 staging fault cases (default: all)')
    parser.add_argument('--power-loss-method', choices=('manual', 'smartplug'))
    parser.add_argument('--smartplug-command')
    parser.add_argument('--probe-timeout', type=int, default=600)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--cleanup-timeout', type=int, default=900)
    args = parser.parse_args(argv)
    if args.scenario is None:
        args.scenario = ['attest']
    seen, unique = set(), []
    for name in args.scenario:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    args.scenario = unique
    return args


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args.scenario)
    if not args.execute:
        print(json.dumps({
            'mode': 'PREVIEW',
            'note': 'no console, GPU, service, file or network access performed',
            'plan': plan,
            'required_gates': {
                'allow-app-run': ('attest' in args.scenario or
                                  'durability' in args.scenario),
                'allow-phase-flags': 'attest' in args.scenario,
                'power-loss-optional': 'durability' in args.scenario,
                'stale-lock-needs': '--destructive (D7)',
                'power-loss-needs': '--destructive --power-loss-method manual|smartplug (D6)',
                'identity-provisioning-implemented': False,
            },
        }, indent=2))
        return 0
    execute(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
