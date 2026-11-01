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
        self.assertIn('D4', plan['destructive_steps'])
        self.assertIn('I3', plan['destructive_steps'])
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
        with self.assertRaises(RuntimeError):
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
