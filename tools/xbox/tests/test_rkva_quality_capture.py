#!/usr/bin/env python3
"""Offline CPU tests for the raw-vocabulary RKVA capture driver.

The driver is compiled with the host g++ against the root build-optimized
llama/ggml libraries. No test initializes CUDA, loads a model, starts a server,
touches the network, or enables execute mode.

Set RKVA_CAPTURE_DRIVER to an already-built binary to skip compilation.
Set RKVA_LLAMA_ROOT to the source/build root (default /home/hjotha/beellama.cpp).
Set RKVA_CAPTURE_REQUIRE_DRIVER=1 to turn a compilation failure into a test
failure instead of a skip.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'rkva_quality_capture.cpp'
ROOT = Path(os.environ.get('RKVA_LLAMA_ROOT', '/home/hjotha/beellama.cpp'))
DRIVER: Path | None = None
BUILD_DIR: Path | None = None


def _build_driver() -> None:
    global DRIVER, BUILD_DIR
    configured = os.environ.get('RKVA_CAPTURE_DRIVER')
    if configured:
        DRIVER = Path(configured)
        if not DRIVER.is_file():
            raise RuntimeError(f'RKVA_CAPTURE_DRIVER is not a file: {DRIVER}')
        return
    lib = ROOT / 'build-optimized/bin/libllama-common.so'
    if not SOURCE.is_file() or not lib.is_file():
        if os.environ.get('RKVA_CAPTURE_REQUIRE_DRIVER') == '1':
            raise RuntimeError(f'missing driver source or root library: {SOURCE} {lib}')
        return
    BUILD_DIR = Path(tempfile.mkdtemp(prefix='rkva-capture-build-'))
    out = BUILD_DIR / 'rkva_quality_capture'
    command = [
        'g++', '-std=c++17', '-O2', '-Wall', '-Wextra',
        f'-I{ROOT / "include"}', f'-I{ROOT / "common"}', f'-I{ROOT / "ggml/include"}',
        f'-I{ROOT / "ggml/src"}', f'-I{ROOT / "vendor"}', f'-I{ROOT / "src"}',
        str(SOURCE), '-o', str(out),
        f'-L{ROOT / "build-optimized/bin"}', f'-Wl,-rpath,{ROOT / "build-optimized/bin"}',
        '-lllama-common', '-lllama', '-lggml', '-lggml-base', '-pthread', '-ldl', '-fopenmp',
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, 'stderr', '') or str(error)
        if os.environ.get('RKVA_CAPTURE_REQUIRE_DRIVER') == '1':
            raise RuntimeError(f'driver compilation failed:\n{detail}') from error
        return
    DRIVER = out


def setUpModule() -> None:
    _build_driver()


def tearDownModule() -> None:
    if BUILD_DIR is not None:
        shutil.rmtree(BUILD_DIR, ignore_errors=True)


class DriverCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix='rkva-capture-test-'))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def require_driver(self) -> Path:
        if DRIVER is None:
            self.skipTest('driver was not built; set RKVA_CAPTURE_REQUIRE_DRIVER=1 to fail')
        return DRIVER

    def run_driver(self, *args: str, env: dict[str, str] | None = None,
                   cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        driver = self.require_driver()
        return subprocess.run([str(driver), *args], capture_output=True, text=True, timeout=120,
                              env=env, cwd=str(cwd or self.tmp))

    def common_args(self, model: Path | None = None, **overrides: str) -> list[str]:
        model = model or (self.tmp / 'model.gguf')
        command = [
            '/home/hjotha/beellama.cpp/build-optimized/bin/llama-server',
            '-m', str(model), '--host', '127.0.0.1', '--port', '12345',
            '--ctx-size', '131072', '--batch-size', '1024', '--ubatch-size', '512',
            '--threads', '8', '--threads-batch', '8', '--parallel', '1',
            '--device', 'CUDA0', '--gpu-layers', '99', '--flash-attn', 'on',
            '--fit', 'off', '--cache-type-k', 'kvarn4', '--cache-type-v', 'kvarn4',
            '--remote-attn', '192.168.1.200:50054', '--kv-position-split-p', '102400',
            '--remote-attn-layers', 'full', '--remote-attn-cache-type-k', 'q4_0',
            '--remote-attn-cache-type-v', 'q4_0', '--remote-attn-format', 'kvarn4',
            '--remote-attn-wire-format', 'fp8', '--remote-attn-stats',
            '--spec-type', 'none', '--spec-draft-n-max', '0', '--cache-ram', '4096',
            '--ctx-checkpoints', '1', '--no-context-shift', '--no-webui',
            '--metrics', '--verbosity', '4', '--slot-save-path', str(self.tmp) + '/',
            '--no-warmup',
        ]
        if overrides:
            for key, value in overrides.items():
                flag = '--' + key.replace('_', '-')
                if value is None:
                    command = [item for item in command if item != flag]
                elif flag in command:
                    command[command.index(flag) + 1] = value
                else:
                    command += [flag, value]
        return command

    def fixtures(self, *, length: int = 106496, teacher_steps: int = 128,
                 teacher: list[int] | None = None, common: list[str] | None = None) -> dict[str, object]:
        (self.tmp / 'model.gguf').write_bytes(b'')
        (self.tmp / 'prefix.bin').write_bytes(b'X' * 4096)
        prompt = {'name': 'portuguese', 'tokens': [0] * length}
        (self.tmp / 'prompt.json').write_text(json.dumps(prompt))
        if teacher is None:
            teacher = [0] * teacher_steps
        (self.tmp / 'teacher.json').write_text(json.dumps({'teacher_ids': teacher}))
        (self.tmp / 'common.json').write_text(json.dumps(common or self.common_args()))
        return {
            'common': str(self.tmp / 'common.json'),
            'prompt': str(self.tmp / 'prompt.json'),
            'prefix': str(self.tmp / 'prefix.bin'),
            'teacher': str(self.tmp / 'teacher.json'),
            'out': self.tmp / 'out',
        }

    def preview_args(self, fixtures: dict[str, object], **overrides: str) -> list[str]:
        values = {
            '--common-args-json': str(fixtures['common']),
            '--prompt-json': str(fixtures['prompt']),
            '--prefix-snapshot': str(fixtures['prefix']),
            '--p': '102400',
            '--context': '131072',
            '--teacher-steps': '128',
            '--teacher-json': str(fixtures['teacher']),
            '--output': str(fixtures['out']),
        }
        values.update(overrides)
        args: list[str] = []
        for key, value in values.items():
            if value is not None:
                args.extend([key, value])
        return args

    def self_test_report(self) -> dict[str, object]:
        report = self.tmp / 'selftest.json'
        result = self.run_driver('--self-test', '--self-test-json', str(report))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('SELF-TEST PASS', result.stdout)
        return json.loads(report.read_text())


class TestSourceContract(unittest.TestCase):
    def test_driver_uses_capi_and_explicit_scope(self) -> None:
        text = SOURCE.read_text()
        for token in ('common_params_parse', 'LLAMA_EXAMPLE_SERVER', 'common_init_from_params',
                      'llama_state_seq_load_file', 'llama_get_logits_ith', 'llama_decode',
                      'load_mtp = false', 'RAW_CAPTURE_ONLY', 'seam-tail-128',
                      'prefix_packed_count_ok', 'P-128', 'failed.json', 'manifest.json',
                      'common_speculative_get_output_limits', 'n_outputs_max = std::max<int32_t>(1, limits.total)',
                      'backend_sampling = false', 'llama_get_sampled_logits_count_ith',
                      'one_device_with_terminator'):
            self.assertIn(token, text)
        for forbidden in ('httplib', 'nvidia-smi', 'systemctl', 'curl ', 'llama_sampler_init('):
            self.assertNotIn(forbidden, text)

    def test_driver_does_not_edit_the_approved_harness(self) -> None:
        self.assertTrue(SOURCE.is_file())
        self.assertTrue((HERE / 'test_rkva_quality_capture.py').is_file())
        self.assertTrue((SOURCE.parents[3] / 'docs/agent-reports/rkva-quality-raw-capture.md').is_file())


class TestCliInert(DriverCase):
    def test_help_is_inert(self) -> None:
        result = self.run_driver('--help')
        self.assertEqual(result.returncode, 0)
        self.assertIn('RAW_CAPTURE_ONLY', result.stdout)
        self.assertIn('No backend/model/network', result.stdout)

    def test_unknown_argument_is_rejected(self) -> None:
        result = self.run_driver('--nonsense')
        self.assertEqual(result.returncode, 2)
        self.assertIn('unknown argument', result.stderr)

    def test_self_test_alone(self) -> None:
        result = self.run_driver('--self-test')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), 'SELF-TEST PASS')
        report = self.self_test_report()
        self.assertEqual(report['scope'], {'backend_initialized': False, 'model_loaded': False,
                                           'network_started': False, 'writes_output': False})


class TestPreview(DriverCase):
    def test_preview_is_inert_and_complete(self) -> None:
        fixtures = self.fixtures()
        result = self.run_driver(*self.preview_args(fixtures))
        self.assertEqual(result.returncode, 0, result.stderr)
        for line in ('mode: preview', 'execute: false', 'starts_child: false',
                     'writes_output: false', 'prompt_tokens: 106496',
                     'tail_tokens: 4096', 'teacher_steps: 128',
                     'teacher_source: fixed-file', 'cadence_api_calls: 7',
                     'cadence_ubatch_512: 7', 'cadence_seam_256: 1', 'cadence_seam_128: 2'):
            self.assertIn(line, result.stdout)
        self.assertFalse(fixtures['out'].exists())

    def test_preview_without_teacher_uses_argmax(self) -> None:
        fixtures = self.fixtures()
        result = self.run_driver(*self.preview_args(fixtures, **{'--teacher-json': None}))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('teacher_source: argmax', result.stdout)
        self.assertFalse(fixtures['out'].exists())

    def test_hard_cap_and_exact_context(self) -> None:
        fixtures = self.fixtures()
        for context in ('160001', '160000', '1'):
            result = self.run_driver(*self.preview_args(fixtures, **{'--context': context}))
            self.assertEqual(result.returncode, 2, context)
            self.assertFalse(fixtures['out'].exists())

    def test_exact_p_and_tail_bounds(self) -> None:
        cases = [(102399, 106496, '--p'), (102400, 102911, '--prompt-json'),
                 (102400, 131072, '--prompt-json')]
        for p, length, key in cases:
            fixtures = self.fixtures(length=length)
            overrides = {'--p': str(p)}
            result = self.run_driver(*self.preview_args(fixtures, **overrides))
            self.assertEqual(result.returncode, 2, (p, length, key))
            self.assertFalse(fixtures['out'].exists())

    def test_minimum_valid_tail(self) -> None:
        fixtures = self.fixtures(length=102912)
        result = self.run_driver(*self.preview_args(fixtures))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('tail_tokens: 512', result.stdout)
        self.assertFalse(fixtures['out'].exists())

    def test_existing_output_is_refused(self) -> None:
        fixtures = self.fixtures()
        fixtures['out'].mkdir()
        result = self.run_driver(*self.preview_args(fixtures))
        self.assertEqual(result.returncode, 2)
        self.assertIn('overwrite', result.stderr)

    def test_symlink_output_is_refused(self) -> None:
        fixtures = self.fixtures()
        target = self.tmp / 'target'
        target.mkdir()
        fixtures['out'].symlink_to(target)
        result = self.run_driver(*self.preview_args(fixtures))
        self.assertEqual(result.returncode, 2)
        self.assertIn('symlink', result.stderr)

    def test_common_args_must_pin_the_config(self) -> None:
        variants = [
            {'no_warmup': None},
            {'no_context_shift': None},
            {'cache_type_k': 'q4_0'},
            {'gpu_layers': '98'},
            {'parallel': '2'},
            {'device': 'CPU'},
            {'fit': 'on'},
            {'remote_attn_wire_format': 'f16'},
            {'ctx_size': '131000'},
            {'kv_position_split_p': '102399'},
        ]
        for overrides in variants:
            fixtures = self.fixtures(common=self.common_args(**overrides))
            result = self.run_driver(*self.preview_args(fixtures))
            self.assertEqual(result.returncode, 2, overrides)
            self.assertFalse(fixtures['out'].exists(), overrides)

    def test_help_in_common_args_is_refused(self) -> None:
        args = self.common_args() + ['--help']
        fixtures = self.fixtures(common=args)
        result = self.run_driver(*self.preview_args(fixtures))
        self.assertEqual(result.returncode, 2)
        self.assertIn('must not request help', result.stderr)

    def test_teacher_must_be_complete_and_integer(self) -> None:
        fixtures = self.fixtures(teacher=[0] * 127)
        result = self.run_driver(*self.preview_args(fixtures))
        self.assertEqual(result.returncode, 2)
        fixtures = self.fixtures(teacher=[0] * 127 + [-1])
        result = self.run_driver(*self.preview_args(fixtures))
        self.assertEqual(result.returncode, 2)
        fixtures = self.fixtures(teacher=[0] * 127 + ['x'])  # type: ignore[list-item]
        result = self.run_driver(*self.preview_args(fixtures))
        self.assertEqual(result.returncode, 2)


class TestSelfTestReport(DriverCase):
    def test_cadence_matches_independent_reference(self) -> None:
        report = self.self_test_report()
        for case in report['cadence']:
            expected = reference_cadence(case['p'], case['length'])
            actual = [(step['start'], step['count'], step['want_logits'], step['kind'])
                      for step in case['plan']]
            self.assertEqual(actual, expected, (case['p'], case['length']))
            self.assertEqual(case['ubatch_512'], expected_ubatch(expected))
            self.assertEqual(case['seam_256'], sum(1 for step in expected if step[1] == 256))
            self.assertEqual(case['seam_128'], sum(1 for step in expected if step[1] == 128))

    def test_four_k_has_seven_ub512_plus_seam(self) -> None:
        report = self.self_test_report()
        four_k = [case for case in report['cadence'] if case['tail'] == 4096 and case['p'] == 1024][0]
        self.assertEqual(four_k['ubatch_512'], 7)
        self.assertEqual((four_k['seam_256'], four_k['seam_128']), (1, 2))
        self.assertEqual(four_k['plan'][-1]['count'], 128)

    def test_exact_tail_cadence_regressions(self) -> None:
        report = self.self_test_report()
        expected = {4096: 7, 16384: 31, 28160: 54, 57344: 111}
        seen = set()
        for case in report['cadence']:
            tail = case['tail']
            self.assertEqual(case['ubatch_512'], expected[tail], tail)
            covered = case['ubatch_512'] * 512 + case['seam_256'] * 256 + case['seam_128'] * 128
            self.assertEqual(covered, tail, tail)
            self.assertEqual(case['api_calls'], len(case['plan']), tail)
            seen.add(tail)
        self.assertEqual(seen, set(expected))
        root_cap = [case for case in report['cadence'] if case['length'] == 159744]
        self.assertEqual(len(root_cap), 1)
        self.assertEqual(root_cap[0]['tail'], 57344)
        self.assertTrue(root_cap[0]['within_root_cap'])

    def test_device_list_contract_with_opaque_pointers(self) -> None:
        report = self.self_test_report()
        cases = {case['name']: case for case in report['device_slots']}
        self.assertTrue(cases['one_gpu_terminated']['valid'])
        self.assertEqual(cases['one_gpu_terminated']['non_null'], 1)
        self.assertTrue(cases['one_gpu_terminated']['terminated'])
        for name in ('none', 'two_gpus', 'leading_null', 'hole_before_gpu', 'missing_terminator'):
            self.assertFalse(cases[name]['valid'], name)
        self.assertEqual(cases['two_gpus']['non_null'], 2)
        self.assertTrue(cases['leading_null']['leading_null'])
        self.assertTrue(cases['hole_before_gpu']['holes'])
        self.assertFalse(cases['missing_terminator']['terminated'])

    def test_output_limits_match_the_server_contract(self) -> None:
        report = self.self_test_report()
        limits = report['output_limits']
        self.assertEqual((limits['total'], limits['per_seq']), (1, 1))
        self.assertEqual(limits['expected_total'], 1)
        self.assertEqual(limits['expected_per_seq'], 1)

    def test_envelope_v1_never_falls_back(self) -> None:
        report = self.self_test_report()
        cases = report['envelope']
        self.assertEqual([case['ok'] for case in cases], [True, True, False, False, False, False, False, False])
        self.assertEqual(cases[0]['ids'], [11, 22, 33])
        self.assertTrue(cases[0]['server_v1'])
        self.assertEqual(cases[1]['ids'], [11, 22, 33])
        self.assertFalse(cases[1]['server_v1'])
        for case in cases[2:]:
            self.assertIsNotNone(case['error'])

    def test_prefix_p_minus_128_is_refused(self) -> None:
        report = self.self_test_report()
        guard = report['prefix_guard']
        self.assertTrue(guard['packed_p_ok'])
        self.assertTrue(guard['packed_p_plus_4_ok'])
        self.assertFalse(guard['packed_p_minus_128_ok'])
        self.assertFalse(guard['packed_p_plus_1_ok'])

    def test_writer_is_exact_and_hash_stable(self) -> None:
        report = self.self_test_report()
        writer = report['writer']
        row = [0.5, -1.0, 2.5, 0.0, 1.0, -0.25, 0.75]
        self.assertEqual(writer['bytes'], 7 * 4)
        self.assertEqual(writer['sha256'], hashlib.sha256(struct.pack('<7f', *row)).hexdigest())
        names = [item['filename'] for item in writer['files']]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names, [f'{step:04d}.logits.f32.bin' for step in range(4)])
        self.assertTrue(all(item['exists'] for item in writer['files']))

    def test_writer_rejects_unsafe_rows(self) -> None:
        report = self.self_test_report()
        writer = report['writer']
        self.assertTrue(writer['overwrite_refused'])
        self.assertTrue(writer['nonfinite_rejected'])
        self.assertTrue(writer['short_rejected'])

    def test_prompt_hash_matches_the_harness_format(self) -> None:
        report = self.self_test_report()
        hashes = report['prompt_hash']
        self.assertEqual(hashes['base'],
                         hashlib.sha256(struct.pack('<3I', 1, 2, 3)).hexdigest())
        self.assertEqual(hashes['after_token'],
                         hashlib.sha256(struct.pack('<3I', 1, 2, 3) + struct.pack('<I', 4)).hexdigest())

    def test_fixed_teacher_cannot_be_replaced(self) -> None:
        report = self.self_test_report()
        teacher = report['teacher']
        self.assertEqual(teacher['fixed_choice'], [3, 4])
        self.assertEqual(teacher['argmax_choice'], [0, 0])
        self.assertEqual(teacher['tie_argmax'], 0)


class TestDependencies(unittest.TestCase):
    def test_ldd_is_read_only_and_does_not_use_nvidia_smi(self) -> None:
        if DRIVER is None:
            self.skipTest('driver was not built')
        result = subprocess.run(['ldd', str(DRIVER)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = result.stdout.lower()
        self.assertNotIn('nvidia-smi', text)
        self.assertNotIn('libnvidia-ml', text)


def reference_cadence(p: int, length: int) -> list[tuple[int, int, bool, str]]:
    """Independent copy of the server B1024/UB512 + final 256/128/128 cadence."""
    assert length > p and (length - p) % 512 == 0
    plan: list[tuple[int, int, bool, str]] = []
    start = p
    while start < length:
        remaining = length - start
        count = min(1024, remaining)
        if remaining > 512 and count == remaining:
            count -= 512
        final = start + count == length
        if final and count == 512:
            plan += [(start, 256, False, 'seam-head-256'),
                     (start + 256, 128, False, 'seam-middle-128'),
                     (start + 384, 128, True, 'seam-tail-128')]
        elif final and count == 256:
            plan += [(start, 128, False, 'seam-head-128'),
                     (start + 128, 128, True, 'seam-tail-128')]
        else:
            plan.append((start, count, final, 'parent-final' if final else 'parent'))
        start += count
    assert sum(step[1] for step in plan) == length - p
    assert plan[-1][1] == 128 and plan[-1][2]
    return plan


def expected_ubatch(plan: list[tuple[int, int, bool, str]]) -> int:
    return sum(count // 512 for _, count, _, _ in plan if count % 512 == 0 and count >= 512)


if __name__ == '__main__':
    unittest.main(verbosity=2)
