#!/usr/bin/env python3
"""Offline negative controls for native_companion_acceptance."""
import json
import argparse
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from native_companion_acceptance import FORMATS, NATIVE, assess
from native_companion_hardware import server_command, server_environment


def log_for(fmt, companion=None, **updates):
    stats = {'range_cache_type': fmt, 'range_origin': 4096,
             'range_upload_calls': 1, 'range_upload_rows': 1,
             'range_upload_bytes': 1, 'range_attn_calls': 1,
             'range_query_max': 4096, 'range_source_cache_type': 'projected_f32'}
    stats.update(updates)
    log = 'remote_attn stats ' + json.dumps(stats) + '\n'
    if companion is not None:
        log += f'mixed KV layer=0 device=CPU type_k={companion} type_v={companion} domain=standard-original\n'
    return log


class Contract(unittest.TestCase):
    def test_negotiated_native_owner_must_match_and_have_no_q4_mirror(self):
        marker = 'native companion negotiated format=kvarn4 standard_cache=absent\n'
        result = assess('kvarn4', log_for('kvarn4') + marker, 4096)
        self.assertNotIn('actual companion allocation and format', result['unverified'])
        self.assertEqual(result['observed_native_owners'], ['kvarn4'])
        self.assertEqual(assess('saw_int4', log_for('saw_int4') + marker, 4096)['status'], 'FAIL')
        self.assertEqual(assess('kvarn4', log_for('kvarn4', 'q4_0') + marker, 4096)['status'], 'FAIL')

    def test_every_format_requires_real_companion_and_never_promotes_external_log(self):
        for fmt in FORMATS:
            with self.subTest(fmt=fmt):
                result = assess(fmt, log_for(fmt), 4096)
                self.assertEqual(result['status'], 'UNVERIFIED')
                self.assertIn('actual companion allocation and format', result['unverified'])
                self.assertFalse(result['errors'])

    def test_wrong_q4_companion_is_a_failure_for_each_other_format(self):
        for fmt in FORMATS:
            if fmt != 'q4_0':
                with self.subTest(fmt=fmt):
                    self.assertEqual(assess(fmt, log_for(fmt, 'q4_0'), 4096)['status'], 'FAIL')

    def test_matching_standard_companion_does_not_prove_process_or_snapshots(self):
        for fmt in ('q4_0', 'q8_0', 'f16'):
            with self.subTest(fmt=fmt):
                result = assess(fmt, log_for(fmt, fmt), 4096)
                self.assertEqual(result['status'], 'UNVERIFIED')
                self.assertNotIn('actual companion allocation and format', result['unverified'])
                self.assertIn('companion state/restore/rewind qualification', result['unverified'])

    def test_no_remote_activity_or_wrong_geometry_fails(self):
        for updates in ({'range_origin': 8192}, {'range_query_max': 4095},
                        {'range_upload_bytes': 0}, {'range_attn_calls': 0},
                        {'range_cache_type': 'q4_0'}, {'range_source_cache_type': 'q4_0'}):
            with self.subTest(updates=updates):
                self.assertEqual(assess('kvarn4', log_for('kvarn4', 'kvarn4', **updates), 4096)['status'], 'FAIL')

    def test_invalid_json_fails_and_missing_stats_is_unverified(self):
        self.assertEqual(assess('q4_0', 'remote_attn stats {garbled}\n', 4096)['status'], 'FAIL')
        self.assertEqual(assess('q4_0', '', 4096)['status'], 'UNVERIFIED')

    def test_unverified_log_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.log'
            path.write_text(log_for('q4_0', 'q4_0'))
            check = subprocess.run([sys.executable,
                                    str(Path(__file__).with_name('native_companion_acceptance.py')),
                                    '--format', 'q4_0', '--origin', '4096', '--log', str(path)],
                                   capture_output=True, text=True, check=False)
            self.assertEqual(check.returncode, 3)
            self.assertEqual(json.loads(check.stdout)['status'], 'UNVERIFIED')

    def test_each_hardware_command_requests_matching_formats_without_snapshots(self):
        for fmt in FORMATS:
            with self.subTest(fmt=fmt):
                args = argparse.Namespace(server_binary='/tmp/server', model='/tmp/model',
                                          context=2048, origin=1024, xbox='192.168.1.200:50054',
                                          threads=4, format=fmt)
                cmd = server_command(args, 9999)
                self.assertEqual(cmd[cmd.index('--remote-attn-format') + 1], fmt)
                for flag in ('--remote-attn-cache-type-k', '--remote-attn-cache-type-v'):
                    self.assertEqual(cmd[cmd.index(flag) + 1], fmt)
                self.assertFalse(any('snapshot' in flag or 'slot-save' in flag for flag in cmd))
                env = server_environment(fmt, '/candidate')
                self.assertEqual(env.get('GGML_RKVA_NATIVE_COMPANION'), '1' if fmt in NATIVE else None)
                self.assertEqual(env['GGML_RKVA_DISK_SNAPSHOTS'], '0')
                self.assertEqual(env['GGML_RKVA_RANGE_STORAGE'], fmt)
                self.assertEqual(env['LD_LIBRARY_PATH'].split(':')[0], '/candidate')

    def test_explicit_tail_reaches_server(self):
        args = argparse.Namespace(server_binary='/tmp/server', model='/tmp/model',
                                  context=2048, origin=1024, xbox='192.168.1.200:50054',
                                  threads=4, format='kvarn4', tail_tokens=256)
        cmd = server_command(args, 9999)
        self.assertEqual(cmd[cmd.index('--kv-tail-tokens') + 1], '256')

    def test_hardware_preview_never_executes_server(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tokens.json'
            path.write_text(json.dumps([1] * 1280))
            output = Path(directory) / 'not_created'
            probe = subprocess.run([sys.executable,
                                    str(Path(__file__).with_name('native_companion_hardware.py')),
                                    '--server-binary', '/does/not/exist', '--model', '/does/not/exist',
                                    '--library-dir', '/does/not/exist',
                                    '--prompt-tokens', str(path), '--output', str(output),
                                    '--xbox', '192.168.1.200:50054', '--format', 'kvarn4'],
                                   capture_output=True, text=True, check=False)
            self.assertEqual(probe.returncode, 0, probe.stderr)
            self.assertFalse(json.loads(probe.stdout)['starts_server'])
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
