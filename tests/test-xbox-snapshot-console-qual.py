#!/usr/bin/env python3
"""CPU-only checks of the console qualification controller skeleton.

Run: python3 tests/test-xbox-snapshot-console-qual.py
Never touches a console, service, GPU, model or network.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / 'tools/xbox/scripts/xbox-snapshot-console-qual.py'
spec = importlib.util.spec_from_file_location('console_qual', PATH)
console = importlib.util.module_from_spec(spec)
spec.loader.exec_module(console)

EXPECT = {
    'session_id': 7,
    'rows': 4096,
    'generation': 3,
    'host_key_sha256': '1' * 64,
    'prefix_sha256': '2' * 64,
}


def request(probe_id='a' * 32, case='positive'):
    return {
        'schema': console.SCHEMAS['attest'],
        'probe_id': probe_id,
        'case': case,
        'expect': EXPECT,
        'phase_flags': {'phase3': True, 'phase4': True,
                        'caps': ['REGISTRY_V1', 'AUTH_V2', 'FRAMING_V3']},
    }


def durability_request(case, probe_id='d' * 32):
    return {
        'schema': console.SCHEMAS['durability'],
        'probe_id': probe_id,
        'case': case,
        'fault': {'kind': case, 'injected': True, 'at': 0},
        'test_only': True,
        'store': {'root': console.DURABILITY_ROOT,
                  'namespace': 'rkva-snapshots-v3',
                  'production_namespace': False},
        'phase_flags': {'phase3': False, 'phase4': False},
    }


def durability_evidence(case, probe_id='d' * 32):
    state = console.DURABILITY_EXPECTED_STATE[case]
    uncertain = case in console.DURABILITY_UNCERTAIN_CASES
    document = {
        'schema': console.SCHEMAS['durability'],
        'probe_id': probe_id,
        'case': case,
        'build': {
            'commit': '3a02e90',
            'test_only_probe': True,
            'durable_fault_seam': True,
            'production_publication_enabled': False,
            'windows_backend_qualified': False,
            'phase3_enabled': False,
            'phase4_enabled': False,
        },
        'store': {'root': console.DURABILITY_ROOT,
                  'namespace': 'rkva-snapshots-v3',
                  'production_namespace': False},
        'fault': {'kind': case, 'injected': True, 'at': 0},
        'writer': {'state': state, 'object_present': uncertain,
                   'error': 'injected fault'},
        'before': {'committed_sha256': 'a' * 64,
                   'inventory': ['manifest-v2.bin', 'n-0.kv']},
        'after': {'committed_sha256': 'a' * 64,
                  'inventory': ['manifest-v2.bin', 'n-0.kv']},
        'legacy': {'touched': False, 'v1_unchanged': True, 'v2_unchanged': True},
        'staging': {'present': not uncertain, 'charged': True, 'adoptable': False,
                    'automatic_cleanup': False, 'adopted': False, 'deleted': False},
        'pin': {'consumed': True, 'reusable': False},
        'claims': {'durable': False, 'published': False, 'committed': False,
                   'gc_authorized': False},
        'cleanup': {'automatic': False},
    }
    if case == 'stale_lock':
        document['lock'] = {
            'held_by_probe': True,
            'owner_evidence_required': True,
            'reclaimable_with_owner_evidence': True,
            'pid_reuse_accepted': False,
            'second_writer_entered': False,
        }
    return document


def staging_inventory(probe_id='d' * 32):
    return {
        'schema': 'rkva-console-staging-inventory-v1',
        'probe_id': probe_id,
        'root': console.DURABILITY_ROOT,
        'entries': [
            {'name': 'tmp-' + '1' * 64, 'bytes': 4096},
            {'name': 'manifest-v2.bin', 'bytes': 512},
        ],
    }


def evidence(probe_id='a' * 32):
    return {
        'schema': console.SCHEMAS['attest'],
        'probe_id': probe_id,
        'build': {
            'commit': '4443ba6',
            'phase3_compiled': True,
            'phase3_enabled': True,
            'phase4_compiled': True,
            'phase4_enabled': True,
            'caps': ['REGISTRY_V1', 'AUTH_V2', 'FRAMING_V3'],
        },
        'attestation': {
            'session_id': EXPECT['session_id'],
            'live_rows': EXPECT['rows'],
            'generation': EXPECT['generation'],
            'submitted_fence': 8,
            'completed_fence': 8,
            'host_key_sha256': EXPECT['host_key_sha256'],
            'prefix_sha256': EXPECT['prefix_sha256'],
            'metadata_sha256': '3' * 64,
            'coverage_mac': '4' * 64,
            'state_digest': '5' * 64,
        },
    }


class PreviewTests(unittest.TestCase):
    def test_preview_performs_no_io(self):
        argv = [str(PATH), '--label', 'preview', '--scenario', 'attest',
                '--scenario', 'durability', '--scenario', 'identity']
        with mock.patch.object(sys, 'argv', argv), \
                mock.patch.object(console.os, 'open', side_effect=AssertionError('os.open')), \
                mock.patch.object(console.subprocess, 'Popen', side_effect=AssertionError('popen')), \
                mock.patch.object(console.subprocess, 'run', side_effect=AssertionError('run')), \
                mock.patch.object(console.urllib.request, 'urlopen', side_effect=AssertionError('urlopen')):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(console.main(), 0)
        document = json.loads(out.getvalue())
        self.assertEqual(document['mode'], 'PREVIEW')
        self.assertIn('plan', document)

    def test_plan_covers_all_three_blockers(self):
        plan = console.build_plan(['attest', 'durability', 'identity'])
        blockers = {step['blocker'] for step in plan['steps']}
        self.assertIn('2', blockers)
        self.assertIn('3', blockers)
        self.assertIn('3 (prerequisite only)', blockers)
        self.assertIn('7', blockers)
        self.assertIn('D6', plan['destructive_steps'])
        self.assertIn('D7', plan['destructive_steps'])
        self.assertIn('I3', plan['destructive_steps'])
        ids = {step['id'] for step in plan['steps']}
        self.assertTrue({'D3', 'D4', 'D5', 'D6', 'D7'} <= ids)
        for step in plan['steps']:
            for key in ('id', 'action', 'artifacts', 'pass', 'fail',
                        'hardware_state', 'destructive'):
                self.assertIn(key, step)
        self.assertTrue(any('staging probe' in step['hardware_state'].lower()
                            for step in plan['steps']))


class GateTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(
            execute=True, label='run', pfn='GianlucaMazza.xllama_1.6.0.436_x64__pj67f1fcj4n14',
            xbox_host='192.168.1.200', scenario=['attest'], expect='/tmp/expect.json',
            allow_app_run=True, allow_phase_flags=True, destructive=False,
            power_loss_method=None, smartplug_command=None, timeout=3600,
            cleanup_timeout=900)
        values.update(overrides)
        return mock.Mock(**values)

    def test_preview_skips_gates(self):
        console.validate_gates(self.args(execute=False, pfn=None))

    def test_requires_exact_pfn(self):
        with self.assertRaises(RuntimeError):
            console.validate_gates(self.args(pfn=None))
        with self.assertRaises(RuntimeError):
            console.validate_gates(self.args(pfn='xllama'))

    def test_requires_app_run_and_phase_flags(self):
        with self.assertRaises(RuntimeError):
            console.validate_gates(self.args(allow_app_run=False))
        with self.assertRaises(RuntimeError):
            console.validate_gates(self.args(allow_phase_flags=False))
        with self.assertRaises(RuntimeError):
            console.validate_gates(self.args(expect=None))

    def test_power_loss_requires_destructive_and_method(self):
        with self.assertRaises(RuntimeError):
            console.validate_gates(self.args(scenario=['durability'], allow_app_run=True,
                                             power_loss_method='manual', destructive=False))
        # --destructive alone is D7 only; D6 additionally needs the method.
        console.validate_gates(self.args(scenario=['durability'], allow_app_run=True,
                                         power_loss_method=None, destructive=True))
        console.validate_gates(self.args(scenario=['durability'], allow_app_run=True,
                                         power_loss_method='manual', destructive=True))

    def test_durability_without_destructive_is_allowed_as_prerequisite_only(self):
        console.validate_gates(self.args(scenario=['durability'], allow_app_run=True,
                                         destructive=False, power_loss_method=None))

    def test_identity_destructive_is_refused_as_unimplemented(self):
        with self.assertRaises(RuntimeError):
            console.validate_gates(self.args(scenario=['identity'], allow_app_run=False,
                                             destructive=True, power_loss_method=None))


class AttestationTests(unittest.TestCase):
    def test_positive_evidence_passes(self):
        self.assertEqual(console.validate_attest_evidence(evidence(), request()), [])

    def test_each_field_mismatch_fails(self):
        for mutate in (
                lambda d: d['attestation'].__setitem__('generation', 4),
                lambda d: d['attestation'].__setitem__('live_rows', 1),
                lambda d: d['attestation'].__setitem__('session_id', 0),
                lambda d: d['attestation'].__setitem__('prefix_sha256', '9' * 64),
                lambda d: d['attestation'].__setitem__('host_key_sha256', '9' * 64),
                lambda d: d['attestation'].__setitem__('submitted_fence', 9),
                lambda d: d['attestation'].__setitem__('coverage_mac', '0' * 64),
                lambda d: d['attestation'].__setitem__('coverage_mac', d['attestation']['metadata_sha256']),
                lambda d: d['build'].__setitem__('phase4_enabled', False),
                lambda d: d['build'].__setitem__('caps', ['REGISTRY_V1']),
                lambda d: d.__setitem__('probe_id', 'b' * 32),
        ):
            document = json.loads(json.dumps(evidence()))
            mutate(document)
            self.assertTrue(console.validate_attest_evidence(document, request()),
                            'mutation was accepted')

    def test_negative_evidence(self):
        document = {
            'schema': console.SCHEMAS['attest'],
            'probe_id': 'c' * 32,
            'negative': {
                'kinds': ['generation_mismatch', 'key_mismatch', 'session_mismatch',
                          'prefix_mismatch', 'fence_regression'],
                'all_refused': True,
                'accepted_any': False,
                'coverage_mac_emitted': False,
            },
        }
        self.assertEqual(console.validate_attest_negative(document, request('c' * 32)), [])
        document['negative']['accepted_any'] = True
        self.assertTrue(console.validate_attest_negative(document, request('c' * 32)))


class DurabilityEvidenceTests(unittest.TestCase):
    def test_every_fault_case_passes_without_a_publication_claim(self):
        for case in console.DURABILITY_FAULT_CASES:
            self.assertEqual(
                console.validate_durability_evidence(
                    durability_evidence(case), durability_request(case)), [],
                case)

    def test_stale_lock_evidence(self):
        case = 'stale_lock'
        document = durability_evidence(case)
        self.assertEqual(
            console.validate_stale_lock_evidence(document, durability_request(case)), [])
        predecessor = json.loads(json.dumps(document))
        del predecessor['lock']['held_by_probe']
        predecessor['lock']['held_by_predecessor'] = True
        self.assertEqual(
            console.validate_stale_lock_evidence(
                predecessor, durability_request(case)), [])
        for mutate in (
                lambda d: d['lock'].__setitem__('held_by_probe', False),
                lambda d: d['lock'].__setitem__('pid_reuse_accepted', True),
                lambda d: d['lock'].__setitem__('second_writer_entered', True),
                lambda d: d['staging'].__setitem__('adopted', True),
                lambda d: d['staging'].__setitem__('deleted', True),
        ):
            broken = json.loads(json.dumps(document))
            mutate(broken)
            self.assertTrue(
                console.validate_stale_lock_evidence(broken, durability_request(case)))

    def test_forbidden_claims_and_scope_are_refused(self):
        for case in ('chunk_sync', 'rename_eio'):
            request = durability_request(case)
            mutations = (
                lambda d: d['writer'].__setitem__('state', 'published_pending'),
                lambda d: d['claims'].__setitem__('durable', True),
                lambda d: d['claims'].__setitem__('committed', True),
                lambda d: d['store'].__setitem__(
                    'root', 'rkva-snapshots-v3'),
                lambda d: d['store'].__setitem__('production_namespace', True),
                lambda d: d['after'].__setitem__('committed_sha256', 'b' * 64),
                lambda d: d['legacy'].__setitem__('touched', True),
                lambda d: d['staging'].__setitem__('charged', False),
                lambda d: d['staging'].__setitem__('adoptable', True),
                lambda d: d['pin'].__setitem__('reusable', True),
                lambda d: d['build'].__setitem__('windows_backend_qualified', True),
                lambda d: d['build'].__setitem__('phase4_enabled', True),
                lambda d: d['build'].__setitem__('durable_fault_seam', False),
                lambda d: d['fault'].__setitem__('injected', False),
            )
            for mutate in mutations:
                document = durability_evidence(case)
                mutate(document)
                self.assertTrue(
                    console.validate_durability_evidence(document, request),
                    f'{case} mutation was accepted')

    def test_uncertain_case_requires_object_and_no_stage(self):
        case = 'ack_lost'
        request = durability_request(case)
        document = durability_evidence(case)
        self.assertEqual(console.validate_durability_evidence(document, request), [])
        document['writer']['object_present'] = False
        self.assertTrue(console.validate_durability_evidence(document, request))
        document = durability_evidence(case)
        document['staging']['present'] = True
        self.assertTrue(console.validate_durability_evidence(document, request))

    def test_staging_failure_requires_preserved_stage(self):
        case = 'chunk_sync'
        request = durability_request(case)
        document = durability_evidence(case)
        document['staging']['present'] = False
        self.assertTrue(console.validate_durability_evidence(document, request))

    def test_staging_inventory_scope(self):
        request = durability_request('chunk_sync')
        self.assertEqual(
            console.validate_staging_inventory(staging_inventory(), request), [])
        document = staging_inventory()
        document['root'] = 'rkva-snapshots-v3'
        self.assertTrue(console.validate_staging_inventory(document, request))
        document = staging_inventory()
        document['entries'].append({'name': '../escape', 'bytes': 1})
        self.assertTrue(console.validate_staging_inventory(document, request))
        document = staging_inventory()
        document['entries'].append({'name': 'rkva-snapshots-v2', 'bytes': 1})
        self.assertTrue(console.validate_staging_inventory(document, request))

    def test_durability_needs_every_row_for_pass(self):
        self.assertEqual(console.aggregate_durability('PASS', 'PASS', 'PASS'), 'PASS')
        self.assertEqual(console.aggregate_durability('PASS', 'PASS', 'BLOCKED'),
                         'BLOCKED')
        self.assertEqual(console.aggregate_durability('PASS', 'FAILED', 'PASS'),
                         'FAILED')
        self.assertEqual(console.aggregate_durability('BLOCKED', 'PASS', 'PASS'),
                         'BLOCKED')


class ProbeClassificationTests(unittest.TestCase):
    def test_unknown_action_is_blocked_not_pass(self):
        status, detail = console.classify_probe_result(
            'error: unknown autopilot op rkva_qual_probe', False, None)
        self.assertEqual(status, 'BLOCKED')
        self.assertIn('not implemented', detail)

    def test_ok_without_response_is_blocked(self):
        status, _ = console.classify_probe_result('ok', False, None)
        self.assertEqual(status, 'BLOCKED')

    def test_ok_with_response_is_ready(self):
        status, _ = console.classify_probe_result('ok', True, evidence())
        self.assertEqual(status, 'READY')

    def test_product_error_is_failed(self):
        status, _ = console.classify_probe_result('error: disk full', True, evidence())
        self.assertEqual(status, 'FAILED')

    def test_missing_done_is_blocked(self):
        status, _ = console.classify_probe_result('', True, evidence())
        self.assertEqual(status, 'BLOCKED')

    def test_unavailable_probe_is_blocked_with_exact_reason(self):
        status, detail = console.classify_probe_result(
            'ok', True, {'status': 'unavailable',
                         'reason': 'no live RKVA engine: no accepted HELLO'})
        self.assertEqual(status, 'BLOCKED')
        self.assertIn('no live RKVA engine', detail)
        status, detail = console.classify_probe_result(
            'ok', True, {'status': 'refused',
                         'reason': 'staging fault unavailable: writer closed'})
        self.assertEqual(status, 'BLOCKED')
        self.assertIn('writer closed', detail)

    def test_unavailable_field_blocks_pass(self):
        document = evidence()
        document['build'].update({'phase3_linked': True, 'phase4_linked': True,
                                  'managed_caps_negotiated': True})
        document['attestation']['unavailable_fields'] = {
            'coverage_mac': 'no authenticated managed session'}
        reason = console.probe_blocking_reason(document, require_mac=True)
        self.assertIn('coverage_mac', reason)

    def test_unlinked_phase_libraries_block_pass(self):
        document = evidence()
        document['build'].update({'phase3_linked': False, 'phase4_linked': False,
                                  'managed_caps_negotiated': False})
        reason = console.probe_blocking_reason(document, require_mac=True)
        self.assertIn('not linked', reason)


class IdentityTests(unittest.TestCase):
    def test_pin_must_match_out_of_band_value(self):
        response = {'wdp_certificate_sha256': 'a' * 64,
                    'pfn': 'GianlucaMazza.xllama_1.6.0.436_x64__pj67f1fcj4n14',
                    'identity_namespace_present': False}
        status, errors = console.validate_identity(
            response, {'cert_sha256': 'a' * 64, 'pfn': response['pfn']})
        self.assertEqual((status, errors), ('READY', []))
        status, errors = console.validate_identity(
            response, {'cert_sha256': 'b' * 64, 'pfn': response['pfn']})
        self.assertEqual(status, 'FAILED')
        status, errors = console.validate_identity(response, {'cert_sha256': None})
        self.assertEqual(status, 'INCOMPLETE')
        self.assertIn('TOFU', errors[0])

    def test_identity_namespace_must_be_exact(self):
        response = {'wdp_certificate_sha256': 'a' * 64,
                    'pfn': 'p', 'identity_namespace_present': True,
                    'identity_namespace_entries': ['identity.json']}
        status, _ = console.validate_identity(
            response, {'cert_sha256': 'a' * 64, 'pfn': 'p'})
        self.assertEqual(status, 'FAILED')


class AggregateTests(unittest.TestCase):
    def test_blocked_dominates_pass_but_failed_dominates_all(self):
        self.assertEqual(console.aggregate_status(['PASS', 'PASS']), 'PASS')
        self.assertEqual(console.aggregate_status(['PASS', 'BLOCKED']), 'BLOCKED')
        self.assertEqual(console.aggregate_status(['PASS', 'PREREQ_ONLY']), 'BLOCKED')
        self.assertEqual(console.aggregate_status(['PASS', 'FAILED', 'BLOCKED']), 'FAILED')
        self.assertEqual(console.aggregate_status([]), 'INCOMPLETE')


class RunnerAggregationTests(unittest.TestCase):
    def fake_args(self, scenarios):
        return mock.Mock(label='fake', pfn='p', xbox_host='h',
                         scenario=scenarios, timeout=600)

    def test_detail_strings_do_not_change_the_status(self):
        class Fake(console.Runner):
            def preflight(self):
                self.result['flags_before'] = []

            def pause_production(self):
                self.result['production_paused'] = False

            def identity(self):
                self.result['scenario_results']['identity'] = 'PASS'
                self.result['scenario_details']['identity'] = 'acl evidence pending'

            def rollback(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            runner = Fake(self.fake_args(['identity']), Path(directory), None, 'x')
            runner.run()
        self.assertEqual(runner.result['status'], 'PASS')

    def test_identity_only_does_not_pause_production(self):
        calls = []

        class Fake(console.Runner):
            def preflight(self):
                self.result['flags_before'] = []

            def pause_production(self):
                calls.append('pause')

            def identity(self):
                self.result['scenario_results']['identity'] = 'PASS'

            def rollback(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            runner = Fake(self.fake_args(['identity']), Path(directory), None, 'x')
            runner.run()
        self.assertEqual(calls, [])
        self.assertEqual(runner.result['status'], 'PASS')

    def test_probe_scenario_pauses_production(self):
        calls = []

        class Fake(console.Runner):
            def preflight(self):
                self.result['flags_before'] = []

            def pause_production(self):
                calls.append('pause')
                self.result['production_paused'] = False

            def attestation(self):
                self.result['scenario_results']['attest'] = 'PASS'

            def rollback(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            runner = Fake(self.fake_args(['attest']), Path(directory), None, 'x')
            runner.run()
        self.assertEqual(calls, ['pause'])
        self.assertEqual(runner.result['status'], 'PASS')

    def test_blocked_and_failed_never_become_pass(self):
        class Fake(console.Runner):
            def preflight(self):
                self.result['flags_before'] = []

            def pause_production(self):
                self.result['production_paused'] = False

            def durability(self):
                self.result['scenario_results']['durability'] = 'PREREQ_ONLY'
                self.result['scenario_details']['durability'] = 'staging probe absent'

            def rollback(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            runner = Fake(self.fake_args(['durability']), Path(directory), None, 'x')
            runner.run()
        self.assertEqual(runner.result['status'], 'BLOCKED')
        self.assertEqual(console.aggregate_status(['PREREQ_ONLY']), 'BLOCKED')

    def test_durability_fault_refusal_is_blocked(self):
        # The app probe accepts the D3-D5 request but refuses the test-only
        # seam. The controller must record BLOCKED, not FAILED and not PASS.
        args = self.fake_args(['durability'])
        args.fault_case = None
        with tempfile.TemporaryDirectory() as directory:
            class Fake(console.Runner):
                def delete_probe(self, name, subdir=console.QUAL_DIR):
                    pass

                def probe(self, case, request, timeout, action_name=None):
                    return ('BLOCKED',
                            'staging fault refused: seam is CPU-test-only',
                            {'status': 'refused'})

                def note(self, message):
                    pass

            runner = Fake(args, Path(directory), None, 'x')
            status = runner.durability_faults()
        self.assertEqual(status, 'BLOCKED')
        self.assertEqual(runner.result['scenario_results']['durability-chunk_sync'],
                         'BLOCKED')

    def test_stale_lock_hold_refusal_is_blocked(self):
        # The D7 hold probe is absent/refusing; that is missing evidence, not a
        # failed lease proof.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = self.fake_args(['durability'])
            args.destructive = True

            class Fake(console.Runner):
                def _start_app_with_autopilot(self, request, action_name, timeout):
                    pass

                def wait_for(self, predicate, timeout, interval=2):
                    return True

                def names(self, subdir=''):
                    return ['response-hold.json']

                def fetch(self, name, local, subdir=''):
                    target = output / local
                    target.write_text(json.dumps(
                        {'status': 'refused',
                         'reason': 'no cross-process lease backend in the app'}))
                    return target

                def app_running(self):
                    return False

                def stop_app(self):
                    pass

                def delete_probe(self, name, subdir=console.QUAL_DIR):
                    pass

                def note(self, message):
                    pass

            runner = Fake(args, output, None, 'x')
            status, detail = runner.stale_lock()
        self.assertEqual(status, 'BLOCKED')
        self.assertIn('no cross-process lease backend', detail)

    def test_attestation_with_partial_host_expect_is_blocked(self):
        # A complete-looking probe response still cannot PASS when the host
        # expectation lacks the console session id or the build does not link
        # the managed libraries.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            expect_path = output / 'expect.json'
            expect_path.write_text(json.dumps({
                'schema': 'rkva-host-expect-v1', 'session_id': None,
                'rows': 513, 'generation': 3,
                'host_key_sha256': '1' * 64, 'prefix_sha256': '1' * 64,
                'expectation_available': False,
                'unavailable': {'session_id': 'not stored by the host attempt'},
            }))
            args = self.fake_args(['attest'])
            args.expect = str(expect_path)
            args.probe_timeout = 10
            document = evidence()
            document['build'].update({
                'phase3_linked': False, 'phase4_linked': False,
                'managed_caps_negotiated': False})

            class Fake(console.Runner):
                def probe(self, case, request, timeout):
                    return 'READY', '', document

            runner = Fake(args, output, None, 'x')
            runner.attestation()
        self.assertEqual(runner.result['scenario_results']['attest-positive'],
                         'BLOCKED')
        self.assertEqual(runner.result['scenario_results']['attest'], 'BLOCKED')


class ExpectFileTests(unittest.TestCase):
    def test_expect_file_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'expect.json'
            path.write_text(json.dumps(EXPECT))
            self.assertEqual(console.load_expect(path), EXPECT)
            for key, value in (('session_id', 0), ('rows', 'x'),
                               ('host_key_sha256', 'A' * 64)):
                document = dict(EXPECT, **{key: value})
                path.write_text(json.dumps(document))
                with self.assertRaises(RuntimeError):
                    console.load_expect(path)

    def test_partial_host_expect_is_accepted_but_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'expect.json'
            document = dict(EXPECT, schema='rkva-host-expect-v1', session_id=None,
                            expectation_available=False,
                            unavailable={'session_id': 'not stored by the host attempt'})
            path.write_text(json.dumps(document))
            loaded = console.load_expect(path)
            gaps = console.expectation_gaps(loaded)
            self.assertTrue(gaps)
            self.assertIn('session_id', gaps[0])

    def test_null_without_reason_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'expect.json'
            path.write_text(json.dumps(dict(EXPECT, session_id=None)))
            with self.assertRaises(RuntimeError):
                console.load_expect(path)


class StaticSafetyTests(unittest.TestCase):
    def setUp(self):
        self.source = PATH.read_text()

    def test_no_package_or_snapshot_destruction(self):
        import re
        commands = set(re.findall(r"\.deploy\.run\('([a-z-]+)'", self.source))
        self.assertEqual(commands, {'list-localstate', 'fetch-file', 'upload-file',
                                    'delete-file', 'start-app', 'stop-app'})
        for forbidden in ('install-cert', 'action=save', 'rm -rf',
                          'shutil.rmtree', 'os.remove('):
            self.assertNotIn(forbidden, self.source)
        self.assertNotIn('systemctl restart', self.source)
        self.assertIn("self.systemctl('stop', SERVICE)", self.source)
        self.assertIn("self.systemctl('start', SERVICE)", self.source)

    def test_execute_is_gated_and_preview_returns_before_imports(self):
        self.assertIn("if not args.execute:", self.source)
        main = self.source[self.source.index('def main(argv=None):'):]
        preview = main.index('if not args.execute:')
        execute = main.index('execute(args)')
        self.assertLess(preview, execute)
        self.assertNotIn('execute(args)', main[:preview])
        self.assertNotIn('import importlib', self.source[:self.source.index('def execute(args):')])

    def test_only_self_created_probe_files_are_deleted(self):
        self.assertIn("self.result['probe_files_created'].append", self.source)
        self.assertIn('for subdir, name in reversed', self.source)
        rollback = self.source[self.source.index('    def rollback(self):'):
                               self.source.index('\ndef execute(args):')]
        self.assertNotIn('AUTOPILOT_FILES', rollback)
        self.assertNotIn("self.delete_probe('autopilot", rollback)


if __name__ == '__main__':
    unittest.main()
