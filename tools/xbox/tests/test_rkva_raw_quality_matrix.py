#!/usr/bin/env python3
"""CPU-only tests for the RAW logits RKVA OFF/ON matrix controller.

Every test is offline: no llama model, CUDA backend, Xbox LocalState call,
deploy endpoint, capture child, or network.  Synthetic logits/manifests exercise
the strict validators and the float64 metric path; execute lifecycle tests use
fake deploy/flags/capture objects.
"""
from __future__ import annotations

import array
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'rkva-raw-quality-matrix.py'
spec = importlib.util.spec_from_file_location('rkva_raw_quality_matrix', SCRIPT)
rq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rq)

REPO = SCRIPT.resolve().parents[3]
GATE, HELPER = rq.load_runtime(REPO)
VOCAB = 8
P = 1024
CONTEXT = 2048


def write_logits(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as target:
        array.array('f', values).tofile(target)


def prompt_fixture(length, seed=0):
    return [seed + (index % 7) for index in range(length)]


@contextlib.contextmanager
def fake_row(path, values):
    write_logits(path, values)
    yield path


def build_manifest(root, prompt, teacher, steps, *, vocab=VOCAB, p=P,
                   context=CONTEXT, fixed=True, model='model.gguf',
                   environment=None, greedy_values=None):
    root = Path(root)
    (root / 'raw').mkdir(parents=True, exist_ok=True)
    length = len(prompt)
    teacher = list(teacher)
    expected_hashes = [rq.token_hash(prompt + teacher[:step]) for step in range(steps)]
    step_meta = []
    for step in range(steps):
        filename = f'{step:04d}.logits.f32.bin'
        values = [0.0] * vocab
        if greedy_values is not None:
            values = greedy_values[step]
        elif teacher:
            values[teacher[step] % vocab] = 8.0
        write_logits(root / 'raw' / filename, values)
        step_meta.append({
            'filename': filename, 'bytes': 4 * vocab, 'finite': True,
            'position': length - 1 + step,
            'prompt_sha256': expected_hashes[step],
            'teacher_id': teacher[step] if fixed else int(
                max(range(vocab), key=lambda i: values[i])),
            'sha256': rq.sha256_file(root / 'raw' / filename),
            'cadence': ({'api_calls': 1, 'ubatch_512': 0, 'seam_256': 0, 'seam_128': 0}
                        if step else
                        {'api_calls': 1, 'ubatch_512': (length - p) // 512,
                         'seam_256': 0, 'seam_128': 0}),
        })
    history = list(prompt) + teacher[:steps - 1]
    with open(root / 'raw' / 'tokens.u32.bin', 'wb') as target:
        array.array('I', history).tofile(target)
    cadence = {'p': p, 'length': length,
               'plan': [{'count': length - p, 'kind': 'parent', 'start': p,
                         'want_logits': True}],
               'ubatch_512': (length - p) // 512, 'seam_256': 0, 'seam_128': 0}
    manifest = {
        'schema': 'rkva-quality-raw-capture-v1', 'label': 'RAW_CAPTURE_ONLY',
        'mode': 'execute', 'result': 'PASS',
        'finalized': {'context_destroyed': True, 'profile_flushed': True,
                      'backend_freed': True, 'manifest_after_context_destroy': True},
        'capture': {'vocab': vocab, 'steps': steps, 'prefix_tokens': p,
                    'prompt_tokens': length, 'tail_tokens': length - p,
                    'step_logits_bytes': 4 * vocab, 'all_logits': True,
                    'finite': True, 'source': 'capi_llama_decode',
                    'logits_contract': {'sampled_logits_active': False,
                                        'sampled_logits_count': vocab,
                                        'sampled_candidates_count': 0,
                                        'outputs_requested': 'last-row-only'}},
        'geometry': {'n_ctx': context, 'n_batch': 1024, 'n_ubatch': 512,
                     'n_ubatch_physical': 256, 'n_outputs_max': 1,
                     'n_outputs_max_per_seq': 1, 'n_rs_seq': 0, 'n_seq_max': 1},
        'config': {'path': str((root / model).resolve()), 'n_vocab': vocab},
        'declared': {'cache_k': 'kvarn4', 'cache_v': 'kvarn4', 'device': 'CUDA0',
                     'context': context, 'p': p, 'parallel': 1, 'batch': 1024,
                     'ubatch': 512, 'spec_types': 'none', 'load_mtp': False,
                     'warmup': False, 'n_rs_seq': 0,
                     'remote': {'host': '192.168.1.200', 'format': 'kvarn4',
                                'wire_format': 'fp8'}},
        'environment': {**dict(environment or {}), 'GGML_RKVA_UPLOAD_FORMAT': 'kvarn4'},
        'prefix_restore': {'file': str(root / 'prefix.bin'), 'file_size': 1,
                           'n_loaded': p, 'pos_max': p - 1, 'token_ids_match': True},
        'teacher': {'source': 'fixed-file' if fixed else 'argmax',
                    'steps': steps, 'ids': teacher if fixed else [],
                    'prompt_hashes': expected_hashes},
        'cadence': cadence, 'prompt_sha256': rq.token_hash(prompt),
        'token_history': {'file': 'tokens.u32.bin', 'count': len(history),
                          'sha256': rq.sha256_file(root / 'raw' / 'tokens.u32.bin')},
        'steps': step_meta,
    }
    (root / 'raw' / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def case_fixture(root, name='case', length=1536, p=P, seed=0):
    prompt = prompt_fixture(length, seed)
    return {'name': name, 'category': 'synthetic', 'length': length,
            'tail_length': length - p, 'prompt': prompt,
            'prompt_sha256': rq.token_hash(prompt),
            'prefix_sha256': rq.token_hash(prompt[:p]),
            'tokens_sha256': rq.token_hash(prompt),
            'source': 'test', 'reference': name, 'answer_tokens': None}


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='rkva-raw-matrix-test-'))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def prefix(self):
        path = self.tmp / 'prefix.bin'
        path.write_bytes(b'x')
        return {'path': str(path), 'main_size': 1, 'meta_sha256': 'meta'}


class MetricsTests(TempCase):
    def test_known_softmax_and_mass(self):
        values = [0.0, math.log(3.0), -1.0, 2.0]
        write_logits(self.tmp / 'row.bin', values)
        stats = rq.raw_stats(self.tmp / 'row.bin', 4, 1e-9)
        values = [float(array.array('f', [value])[0]) for value in values]
        expected = [math.exp(value) for value in values]
        total = math.fsum(expected)
        self.assertEqual(stats['top1'], 3)
        self.assertAlmostEqual(stats['mass'], 1.0, places=12)
        self.assertAlmostEqual(stats['prob_top1'], expected[3] / total, places=12)
        self.assertAlmostEqual(stats['prob_top2'], expected[1] / total, places=12)
        self.assertEqual(stats['lse'], math.log(total))

    def test_nonfinite_and_size_are_rejected(self):
        write_logits(self.tmp / 'nan.bin', [0.0, math.nan, 1.0, 2.0])
        with self.assertRaisesRegex(rq.MatrixError, 'non-finite'):
            rq.raw_stats(self.tmp / 'nan.bin', 4)
        (self.tmp / 'short.bin').write_bytes(b'\0' * 4)
        with self.assertRaisesRegex(rq.MatrixError, 'expected exactly'):
            rq.raw_stats(self.tmp / 'short.bin', 4)
        write_logits(self.tmp / 'extra.bin', [0.0] * 5)
        with self.assertRaisesRegex(rq.MatrixError, 'expected exactly'):
            rq.raw_stats(self.tmp / 'extra.bin', 4)

    def test_teacher_out_of_vocab_is_rejected(self):
        write_logits(self.tmp / 'off.bin', [0.0] * 4)
        write_logits(self.tmp / 'on.bin', [0.0] * 4)
        off = rq.raw_stats(self.tmp / 'off.bin', 4)
        on = rq.raw_stats(self.tmp / 'on.bin', 4)
        with self.assertRaisesRegex(rq.MatrixError, 'outside vocabulary'):
            rq.step_metrics_from_stats(off, on, 4, 0.5)

    def test_full_vocabulary_kl_matches_direct_probabilities(self):
        off_values = [0.0, 1.0, -2.0, 3.0, 0.5, -1.0, 2.0, 0.25]
        on_values = [0.1, 0.8, -1.5, 3.2, 0.2, -1.2, 1.8, 0.5]
        off_values = [float(array.array('f', [value])[0]) for value in off_values]
        on_values = [float(array.array('f', [value])[0]) for value in on_values]
        write_logits(self.tmp / 'off.bin', off_values)
        write_logits(self.tmp / 'on.bin', on_values)
        off = rq.raw_stats(self.tmp / 'off.bin', 8)
        on = rq.raw_stats(self.tmp / 'on.bin', 8)
        p_off = [math.exp(value - off['lse']) for value in off_values]
        p_on = [math.exp(value - on['lse']) for value in on_values]
        expected = math.fsum(
            p_off[i] * (math.log(p_off[i]) - math.log(p_on[i])) for i in range(8))
        metrics = rq.step_metrics_from_stats(off, on, 2, 0.5)
        self.assertAlmostEqual(metrics['kl'], expected, places=12)
        self.assertFalse(metrics['top1_agree'] is None)

    def test_window_gate_uses_frozen_thresholds(self):
        steps = [{'nll_off': 1.0, 'nll_on': 1.005, 'kl': 0.0005,
                  'top1_agree': True, 'margin_off': 0.1, 'margin_on': 0.1,
                  'high_confidence_divergence': False}]
        window = rq.evaluate_window('first_1', steps, dict(GATE.GATE_DEFAULTS), GATE)
        self.assertTrue(window['pass'])
        steps[0]['kl'] = 0.02
        window = rq.evaluate_window('first_1', steps, dict(GATE.GATE_DEFAULTS), GATE)
        self.assertFalse(window['pass'])
        self.assertIn('kl_p99', window['failures'])


class ManifestTests(TempCase):
    def setUp(self):
        super().setUp()
        self.case = case_fixture(self.tmp)
        self.prefix = self.prefix()
        self.model = self.tmp / 'model.gguf'
        self.model.write_bytes(b'model')
        self.common = {'model_path': str(self.model)}

    def manifest(self, *, fixed=True, teacher=(1, 2)):
        return build_manifest(self.tmp, self.case['prompt'], teacher, 2,
                              fixed=fixed, model=self.model.name,
                              environment={'GGML_RKVA_PROFILE': '1'})

    def test_accepts_fixed_and_greedy_manifests(self):
        self.manifest(fixed=True)
        first = rq.validate_capture_manifest(
            self.tmp / 'raw', self.case, [1, 2], 2, self.prefix,
            {'GGML_RKVA_PROFILE': '1'}, self.common, '192.168.1.200:50054',
            GATE, VOCAB, P, CONTEXT)
        self.assertEqual(first['teacher_ids'], [1, 2])
        shutil.rmtree(self.tmp / 'raw')
        self.manifest(fixed=False, teacher=(1, 2))
        second = rq.validate_capture_manifest(
            self.tmp / 'raw', self.case, None, 2, self.prefix,
            {'GGML_RKVA_PROFILE': '1'}, self.common, '192.168.1.200:50054',
            GATE, VOCAB, P, CONTEXT)
        self.assertEqual(second['teacher_ids'], [1, 2])

    def test_nonmatching_teacher_is_rejected(self):
        manifest = self.manifest()
        manifest['steps'][1]['teacher_id'] = 7
        (self.tmp / 'raw' / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(rq.MatrixError, 'step teacher ids differ'):
            rq.validate_capture_manifest(
                self.tmp / 'raw', self.case, [1, 2], 2, self.prefix,
                {'GGML_RKVA_PROFILE': '1'}, self.common, '192.168.1.200:50054',
                GATE, VOCAB, P, CONTEXT)

    def test_cli_resolved_upload_format_overrides_launch_environment(self):
        self.manifest()
        launch = {'GGML_RKVA_PROFILE': '1', 'GGML_RKVA_UPLOAD_FORMAT': 'f16'}
        result = rq.validate_capture_manifest(
            self.tmp / 'raw', self.case, [1, 2], 2, self.prefix,
            launch, self.common, '192.168.1.200:50054', GATE, VOCAB, P, CONTEXT)
        self.assertEqual(result['teacher_ids'], [1, 2])
        self.assertEqual(launch['GGML_RKVA_UPLOAD_FORMAT'], 'f16')

    def test_incorrect_runtime_upload_format_is_rejected(self):
        manifest = self.manifest()
        manifest['environment']['GGML_RKVA_UPLOAD_FORMAT'] = 'f16'
        (self.tmp / 'raw' / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(rq.MatrixError, 'frozen environment differs'):
            rq.validate_capture_manifest(
                self.tmp / 'raw', self.case, [1, 2], 2, self.prefix,
                {'GGML_RKVA_PROFILE': '1'}, self.common, '192.168.1.200:50054',
                GATE, VOCAB, P, CONTEXT)

    def test_prompt_hash_mismatch_is_rejected(self):
        manifest = self.manifest()
        manifest['steps'][0]['prompt_sha256'] = 'bad'
        (self.tmp / 'raw' / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(rq.MatrixError, 'prompt hash'):
            rq.validate_capture_manifest(
                self.tmp / 'raw', self.case, [1, 2], 2, self.prefix,
                {'GGML_RKVA_PROFILE': '1'}, self.common, '192.168.1.200:50054',
                GATE, VOCAB, P, CONTEXT)

    def test_incomplete_and_inconsistent_manifests_fail(self):
        manifest = self.manifest()
        manifest['steps'] = manifest['steps'][:1]
        (self.tmp / 'raw' / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(rq.MatrixError, 'step list is incomplete'):
            rq.validate_capture_manifest(
                self.tmp / 'raw', self.case, [1, 2], 2, self.prefix,
                {'GGML_RKVA_PROFILE': '1'}, self.common, '192.168.1.200:50054',
                GATE, VOCAB, P, CONTEXT)
        manifest = self.manifest()
        manifest['result'] = 'FAILED'
        (self.tmp / 'raw' / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(rq.MatrixError, 'not PASS'):
            rq.validate_capture_manifest(
                self.tmp / 'raw', self.case, [1, 2], 2, self.prefix,
                {'GGML_RKVA_PROFILE': '1'}, self.common, '192.168.1.200:50054',
                GATE, VOCAB, P, CONTEXT)
        manifest = self.manifest()
        manifest['steps'][0]['sha256'] = '0' * 64
        (self.tmp / 'raw' / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(rq.MatrixError, 'sha mismatch'):
            rq.validate_capture_manifest(
                self.tmp / 'raw', self.case, [1, 2], 2, self.prefix,
                {'GGML_RKVA_PROFILE': '1'}, self.common, '192.168.1.200:50054',
                GATE, VOCAB, P, CONTEXT)

    def test_environment_and_bytes_mismatch_fail(self):
        manifest = self.manifest()
        manifest['environment']['GGML_EXTRA'] = '1'
        (self.tmp / 'raw' / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(rq.MatrixError, 'frozen environment differs'):
            rq.validate_capture_manifest(
                self.tmp / 'raw', self.case, [1, 2], 2, self.prefix,
                {'GGML_RKVA_PROFILE': '1'}, self.common, '192.168.1.200:50054',
                GATE, VOCAB, P, CONTEXT)
        manifest = self.manifest()
        manifest['steps'][0]['bytes'] = 4
        (self.tmp / 'raw' / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(rq.MatrixError, 'byte count'):
            rq.validate_capture_manifest(
                self.tmp / 'raw', self.case, [1, 2], 2, self.prefix,
                {'GGML_RKVA_PROFILE': '1'}, self.common, '192.168.1.200:50054',
                GATE, VOCAB, P, CONTEXT)

    def test_missing_activation_profile_is_not_a_pass(self):
        activation = GATE.activation_checks(None, None, [], [], HELPER)
        self.assertFalse(activation['pass'])
        result = {'activation': activation, 'windows': {
            'first_1': {'pass': True}, 'first_8': {'pass': True},
            'all': {'pass': True, 'metrics': {'finite': True,
                                              'high_confidence_divergences': 0}}}}
        checks = rq._final_checks([result], [self.case], [], [])
        self.assertFalse(checks['all_activation_proved']['pass'])


class CaseAndPrefixTests(TempCase):
    def test_explicit_and_tail_cases(self):
        prefix = prompt_fixture(P, 0)
        full = prompt_fixture(1536, 0)
        prefix_sha = rq.token_hash(full[:P])
        cases_dir = self.tmp / 'cases'
        cases_dir.mkdir()
        (cases_dir / 'full.json').write_text(json.dumps({
            'name': 'full', 'length': len(full), 'tokens': full,
            'prompt_sha256': rq.sha256_text(json.dumps(full, separators=(',', ':')))}))
        (cases_dir / 'tail.json').write_text(json.dumps({
            'name': 'tail', 'length': 1536, 'base_length': P,
            'tokens': full[P:], 'tokens_sha256': rq.token_hash(full[P:])}))
        cases = rq.load_cases(cases_dir, 2, prefix_sha, prefix, P, CONTEXT)
        self.assertEqual([case['name'] for case in cases], ['full', 'tail'])
        self.assertEqual(cases[0]['prompt'], cases[1]['prompt'])
        bad = list(full)
        bad[0] = 99
        (cases_dir / 'bad.json').write_text(json.dumps({
            'name': 'bad', 'length': len(bad), 'tokens': bad}))
        with self.assertRaisesRegex(rq.RefuseError, 'first P tokens hash'):
            rq.load_cases(cases_dir, 2, prefix_sha, prefix, P, CONTEXT)

    def test_length_and_context_limits(self):
        prefix = prompt_fixture(P, 0)
        prefix_sha = rq.token_hash(prefix)
        cases_dir = self.tmp / 'cases'
        cases_dir.mkdir()
        (cases_dir / 'bad.json').write_text(json.dumps({
            'name': 'bad', 'tokens': prefix + [0] * 511}))
        with self.assertRaisesRegex(ValueError, 'multiple of 512'):
            rq.load_cases(cases_dir, 2, prefix_sha, prefix, P, CONTEXT)
        (cases_dir / 'bad.json').write_text(json.dumps({
            'name': 'bad', 'tokens': prefix + [0] * (CONTEXT - P)}))
        with self.assertRaisesRegex(ValueError, 'exceeds context'):
            rq.load_cases(cases_dir, 2, prefix_sha, prefix, P, CONTEXT)

    def prefix_fixture(self, *, mutate=None, reference_meta=None,
                       reference_files=None):
        base = [0] * 102400
        main = self.tmp / 'prefix.bin'
        main.write_bytes(b'x')
        namespace = SimpleNamespace(
            prefix_snapshot=str(main), p=102400, context=131072, batch=1024,
            ubatch=512, frozen_prefix_batch=512, cuda_fp8='mapped',
            range_kernel='fine', wavefront_tile=256, n256_lookahead=False,
            tcp_prefetch=True, local_input_reuse=False, mask_pack='default')
        contract = HELPER.prefix_contract(namespace, base, {'sha256': 'm'})
        if mutate is not None:
            mutate(contract)
        files = {main.name: {'size': main.stat().st_size,
                             'sha256': rq.sha256_file(main)}}
        metadata = {
            'contract': contract, 'actual_prefix_tokens': 102400,
            'actual_prefix_sha256': rq.token_hash(base),
            'save': {'n_saved': 102400, 'n_written': main.stat().st_size},
            'files': dict(files),
        }
        (self.tmp / 'prefix.bin.meta.json').write_text(json.dumps(metadata))
        meta_sha = rq.sha256_file(self.tmp / 'prefix.bin.meta.json')
        binding_prefix = {
            'P': 102400, 'context': 131072, 'batch': 512,
            'actual_prefix_tokens': 102400,
            'actual_prefix_sha256': rq.token_hash(base),
            'files': reference_files if reference_files is not None else dict(files),
            'meta_sha256': reference_meta or meta_sha,
        }
        reference = {
            'prefix_path': main,
            'binding': {'prefix': binding_prefix},
            'model': {'sha256': 'm'},
            'preflight': {'prefix_contract': {
                'meta_sha256': reference_meta or meta_sha}},
        }
        return reference, base, metadata, main

    def test_prefix_contract_tamper_fields_all_refuse(self):
        for field, value in (('environment', {'GGML_KVARN_WINDOW_CHUNK': '999'}),
                             ('slot', 1), ('schema', 2)):
            reference, base, _metadata, _main = self.prefix_fixture(
                mutate=lambda contract, field=field, value=value:
                    contract[field].update(value) if field == 'environment'
                    else contract.__setitem__(field, value))
            with self.assertRaisesRegex(rq.RefuseError, 'frozen contract'):
                rq.load_prefix(reference, base, GATE, HELPER)

    def test_prefix_context_conversion_is_refused(self):
        reference, base, _metadata, _main = self.prefix_fixture(
            mutate=lambda contract: contract.__setitem__('context', 160000))
        with self.assertRaisesRegex(rq.RefuseError, 'frozen contract'):
            rq.load_prefix(reference, base, GATE, HELPER)

    def test_prefix_hash_mismatch_is_refused(self):
        reference, base, _metadata, _main = self.prefix_fixture(
            mutate=lambda contract: contract.__setitem__(
                'requested_prefix_sha256', 'bad'))
        with self.assertRaisesRegex(rq.RefuseError, 'frozen contract'):
            rq.load_prefix(reference, base, GATE, HELPER)

    def test_prefix_meta_sha_tamper_is_refused(self):
        reference, base, _metadata, _main = self.prefix_fixture(
            reference_meta='0' * 64)
        with self.assertRaisesRegex(rq.RefuseError, 'metadata SHA'):
            rq.load_prefix(reference, base, GATE, HELPER)

    def test_prefix_filelist_tamper_is_refused(self):
        reference, base, _metadata, main = self.prefix_fixture()
        reference['binding']['prefix']['files'] = {
            main.name: {'size': 1, 'sha256': '0' * 64}}
        with self.assertRaisesRegex(rq.RefuseError, 'file list/details'):
            rq.load_prefix(reference, base, GATE, HELPER)

    def test_prefix_rehashed_forgery_is_refused(self):
        reference, base, metadata, main = self.prefix_fixture()
        original_files = dict(reference['binding']['prefix']['files'])
        main.write_bytes(b'forged')
        metadata['files'][main.name] = {
            'size': main.stat().st_size, 'sha256': rq.sha256_file(main)}
        metadata['save']['n_written'] = main.stat().st_size
        (self.tmp / 'prefix.bin.meta.json').write_text(json.dumps(metadata))
        reference['binding']['prefix']['files'] = original_files
        reference['preflight']['prefix_contract']['meta_sha256'] = rq.sha256_file(
            self.tmp / 'prefix.bin.meta.json')
        reference['binding']['prefix']['meta_sha256'] = rq.sha256_file(
            self.tmp / 'prefix.bin.meta.json')
        with self.assertRaisesRegex(rq.RefuseError, 'file list/details'):
            rq.load_prefix(reference, base, GATE, HELPER)


class LifecycleTests(TempCase):
    def fake_args(self, *, execute=True, determinism='none'):
        capture = self.tmp / 'capture'
        capture.write_bytes(b'binary')
        common = self.tmp / 'common.json'
        common.write_text('[]')
        cases = self.tmp / 'cases'
        cases.mkdir()
        reference = self.tmp / 'reference'
        reference.mkdir()
        deploy = self.tmp / 'deploy.sh'
        deploy.write_bytes(b'deploy')
        argv = ['--capture-binary', str(capture), '--common-args-json', str(common),
                '--cases-dir', str(cases), '--reference-campaign', str(reference),
                '--output', str(self.tmp / 'out'), '--teacher-steps', '2',
                '--deploy', str(deploy)]
        if determinism != 'none':
            argv += ['--determinism', determinism]
        args = rq.validate_args(rq.parser().parse_args(argv))
        args.execute = execute
        return args

    def fake_reference(self, args):
        model = self.tmp / 'model.gguf'
        model.write_bytes(b'model')
        return {'environment': {}, 'binding': {'prefix': {'actual_prefix_sha256': 'p'}},
                'model': {'path': str(model), 'size': model.stat().st_size,
                          'sha256': rq.sha256_file(model)},
                'prefix_path': self.tmp / 'prefix.bin', 'binding_sha256': 'ref',
                'source': {}, 'libraries': {'files': []}, 'root': self.tmp,
                'environment_path': 'fake'}

    def fake_case(self):
        return {'name': 'case', 'category': 'synthetic', 'length': 1536,
                'tail_length': 512, 'prompt': [0] * 1536,
                'prompt_sha256': 'p', 'prefix_sha256': 'x', 'tokens_sha256': 't',
                'source': 'test', 'reference': 'case', 'answer_tokens': None}

    def fake_result(self, status='PASS'):
        return {'case': 'case', 'status': status, 'teacher_ids': [1, 2],
                'activation': {'pass': True},
                'windows': {'all': {'pass': True,
                                    'metrics': {'finite': True,
                                                'high_confidence_divergences': 0}}},
                'steps': [{'logits_sha256': 'a', 'candidate_logits_sha256': 'b'}],
                'failures': [] if status == 'PASS' else ['forced failure']}

    def run_with_fakes(self, args, *, capture_side_effect=None, result=None,
                       cleanup_alive=False):
        reference = self.fake_reference(args)
        case = self.fake_case()
        common_bytes = args.common_args_json.read_bytes()
        common_info = {
            'path': str(args.common_args_json), 'sha256': rq.sha256_bytes(common_bytes),
            'bytes': common_bytes, 'stat': rq.stat_identity(args.common_args_json),
            'model_path': str(self.tmp / 'model.gguf'),
            'frozen_path': str(args.output / 'inputs' / 'common-args.json'),
            'frozen_sha256': rq.sha256_bytes(common_bytes),
        }
        flag_state = {'original': False, 'restore': 0, 'apply': []}
        marker = self.tmp / 'restored.txt'

        class FakeDeploy:
            def __init__(self, *arguments, **keywords):
                pass

            def pfn(self):
                return 'GianlucaMazza.xllama_test'

        class FakeFlags:
            def __init__(self, deploy, pfn, output):
                pass

            def capture_original(self):
                flag_state['original'] = True

            def apply(self, int8_on):
                flag_state['apply'].append(int8_on)

            @contextlib.contextmanager
            def arm(self, int8_on):
                self.apply(int8_on)
                try:
                    yield
                finally:
                    flag_state['restore'] += 1

            def restore(self):
                flag_state['restore'] += 1
                marker.write_text('x')

        def fake_arm_capture(ctx, case, arm, teacher, arm_dir, label):
            if cleanup_alive:
                ctx['flags'].apply(True)
                ctx['model_alive_pid'] = 4242
                raise rq.ChildCleanupError('cleanup failed', 4242, True,
                                           {'pid': 4242, 'alive': True})
            if capture_side_effect is not None:
                raise capture_side_effect
            with ctx['flags'].arm(arm == 'on'):
                pass
            return {'teacher_ids': [1, 2], 'prompt_hashes': ['p'] * 2,
                    'step_sha256': ['a'] * 2, 'raw': str(self.tmp / 'raw'),
                    'directory': str(self.tmp / 'arm')}

        patches = [
            mock.patch.object(rq, 'load_reference', return_value=reference),
            mock.patch.object(rq, 'verify_common_args', return_value=common_info),
            mock.patch.object(rq, '_plan_cases', return_value=([case], [0] * 1024)),
            mock.patch.object(rq, 'load_prefix',
                              return_value={'path': str(self.tmp / 'prefix.bin'),
                                            'meta_sha256': 'm', 'contract': {},
                                            'files': {}, 'main_size': 1,
                                            'metadata': {'actual_prefix_sha256': 'p'}}),
            mock.patch.object(rq, 'disk_preflight', return_value={'free_bytes': 10}),
            mock.patch.object(rq, 'frozen_environment', return_value={}),
            mock.patch.object(rq, 'active_libraries_for', return_value=[]),
            mock.patch.object(rq._GATE, 'build_library_binding',
                              return_value={'files': [], 'sha256': 'l'}),
            mock.patch.object(rq, '_verify_libraries_against_reference'),
            mock.patch.object(rq, 'build_binding',
                              return_value={'binding_sha256': 'b',
                                            'capture_binary': {},
                                            'common_args': {
                                                'path': common_info['path'],
                                                'sha256': common_info['sha256'],
                                                'stat': common_info['stat'],
                                                'frozen_path': common_info['frozen_path'],
                                                'frozen_sha256': common_info['frozen_sha256']},
                                            'model': dict(reference['model'],
                                                          stat=rq.stat_identity(
                                                              reference['model']['path'])),
                                            'environment': {'sha256': 'e', 'keys': []}}),
            mock.patch.object(rq, 'BindingGuard', return_value=mock.Mock()),
            mock.patch.object(rq, 'preview', return_value={'schema': 'preview'}),
            mock.patch.object(rq, 'gpu_quiet'),
            mock.patch.object(rq._GATE, 'DeployClient', FakeDeploy),
            mock.patch.object(rq._GATE, 'LocalStateFlags', FakeFlags),
            mock.patch.object(rq, 'arm_capture', side_effect=fake_arm_capture),
            mock.patch.object(rq, 'compute_case_result',
                              return_value=result or self.fake_result()),
            mock.patch.object(rq, 'verify_prefix_unchanged'),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        with contextlib.redirect_stdout(io.StringIO()):
            return rq.execute(args), flag_state

    def test_flags_are_restored_when_capture_fails(self):
        args = self.fake_args()
        result, state = self.run_with_fakes(
            args, capture_side_effect=rq.MatrixError('boom'))
        self.assertEqual(result, 3)
        self.assertTrue(state['original'])
        self.assertGreaterEqual(state['restore'], 1)
        final = json.loads((args.output / 'final.json').read_text())
        self.assertEqual(final['status'], 'FAIL')
        self.assertNotEqual(final['qualification_label'], 'RAW_NUMERICAL_STRESS_ONLY')
        self.assertTrue((args.output / 'campaign-abort.json').is_file())

    def test_signal_restores_flags_and_is_not_a_pass(self):
        args = self.fake_args()
        with self.assertRaises(KeyboardInterrupt):
            self.run_with_fakes(args, capture_side_effect=KeyboardInterrupt('sigterm'))
        self.assertTrue((args.output / 'campaign-abort.json').is_file())
        self.assertTrue((self.tmp / 'restored.txt').is_file())

    def test_completed_matrix_is_narrow_stress_only(self):
        args = self.fake_args()
        result, state = self.run_with_fakes(args)
        self.assertEqual(result, 0)
        final = json.loads((args.output / 'final.json').read_text())
        self.assertEqual(final['status'], 'PASS')
        self.assertEqual(final['qualification_label'], 'RAW_NUMERICAL_STRESS_ONLY')
        self.assertFalse(final['int8_default_authorized'])
        self.assertEqual(state['apply'], [False, True])
        self.assertTrue(final['flags_restored'])
        self.assertFalse(final['semantic_reference']['reference_answer_json_consumed'])
        self.assertFalse(final['semantic_reference']['fixed_reference_nll'])
        self.assertFalse(final['semantic_reference']['free_greedy_semantic_accuracy'])
        self.assertIn('sha256', final['model_identity'])
        self.assertIn('frozen_sha256', final['common_args'])

    def test_failed_case_never_produces_global_pass(self):
        args = self.fake_args()
        result, _state = self.run_with_fakes(args, result=self.fake_result('FAIL'))
        self.assertEqual(result, 3)
        final = json.loads((args.output / 'final.json').read_text())
        self.assertEqual(final['status'], 'FAIL')
        self.assertEqual(final['lifecycle_errors'], [])
        self.assertTrue(final['checks']['no_lifecycle_or_protocol_errors']['pass'])
        self.assertFalse(final['checks']['all_window_gates']['pass'])

    def test_owned_child_alive_skips_flag_restore_and_fails(self):
        args = self.fake_args()
        result, state = self.run_with_fakes(args, cleanup_alive=True)
        self.assertEqual(result, 3)
        self.assertEqual(state['restore'], 0)
        final = json.loads((args.output / 'final.json').read_text())
        self.assertEqual(final['status'], 'FAIL')
        self.assertFalse(final['flags_restored'])
        self.assertEqual(final['model_alive_pid'], 4242)
        self.assertTrue(final['cleanup_failures'])

    def test_arm_uses_owned_frozen_common_args(self):
        args = self.fake_args()
        capture = args.capture_binary
        frozen = self.tmp / 'frozen-common.json'
        frozen.write_bytes(b'[]')
        case = self.fake_case()
        flag_events = []

        class Flags:
            def apply(self, value):
                flag_events.append(('apply', value))

            def restore(self):
                flag_events.append(('restore', None))

        ctx = {
            'gate': GATE,
            'helper': SimpleNamespace(classify_logs=lambda lines: ([], [])),
            'args': SimpleNamespace(capture_binary=capture, teacher_steps=2,
                                    child_timeout=10, xbox_host='192.168.1.200:50054'),
            'reference': {'environment': {}},
            'common': {'frozen_path': str(frozen), 'path': str(args.common_args_json)},
            'prefix': {'path': str(self.tmp / 'prefix.bin')},
            'binding': {'binding_sha256': 'b'},
            'binding_guard': mock.Mock(),
            'flags': Flags(),
            'popen_factory': None,
        }
        calls = []
        fake_result = {
            'returncode': 0, 'loaded_startup': ['/x/libllama.so'],
            'loaded_error': None, 'startup_event': 'marker',
            'startup_fallback': False, 'command': [],
        }
        with mock.patch.object(rq, 'frozen_environment', return_value={}), \
                mock.patch.object(rq, 'gpu_quiet'), \
                mock.patch.object(rq, 'run_capture_child',
                                  side_effect=lambda command, *a, **k:
                                      (calls.append(command), fake_result)[1]), \
                mock.patch.object(rq, 'validate_capture_manifest',
                                  return_value={'manifest': {}, 'teacher_ids': [1, 2],
                                                'step_sha256': ['a', 'b'],
                                                'prompt_hashes': ['p', 'q']}), \
                mock.patch.object(rq, 'read_log_lines',
                                  return_value=['CAPTURE PASS steps=2']):
            result = rq.arm_capture(ctx, case, 'off', None,
                                    self.tmp / 'arm', 'off')
        command = calls[0]
        self.assertEqual(command[command.index('--common-args-json') + 1], str(frozen))
        self.assertNotEqual(command[command.index('--common-args-json') + 1],
                            str(args.common_args_json))
        self.assertEqual(flag_events, [('apply', False), ('restore', None)])
        self.assertEqual(result['startup_event'], 'marker')

    def test_arm_alive_child_skips_restore(self):
        args = self.fake_args()
        frozen = self.tmp / 'frozen-common.json'
        frozen.write_bytes(b'[]')
        events = []

        class Flags:
            def apply(self, value):
                events.append(('apply', value))

            def restore(self):
                events.append(('restore', None))

        ctx = {
            'gate': GATE,
            'helper': SimpleNamespace(classify_logs=lambda lines: ([], [])),
            'args': SimpleNamespace(capture_binary=args.capture_binary,
                                    teacher_steps=2, child_timeout=10,
                                    xbox_host='192.168.1.200:50054'),
            'reference': {'environment': {}},
            'common': {'frozen_path': str(frozen)},
            'prefix': {'path': str(self.tmp / 'prefix.bin')},
            'binding': {'binding_sha256': 'b'},
            'binding_guard': mock.Mock(),
            'flags': Flags(), 'popen_factory': None,
            'model_alive_pid': None, 'cleanup_failures': [],
            'restore_errors': [], 'flags_restore_failed': None,
        }
        with mock.patch.object(rq, 'frozen_environment', return_value={}), \
                mock.patch.object(rq, 'gpu_quiet'), \
                mock.patch.object(rq, 'run_capture_child',
                                  side_effect=rq.ChildCleanupError(
                                      'cleanup failed', 4242, True,
                                      {'pid': 4242, 'alive': True})):
            with self.assertRaises(rq.ChildCleanupError):
                rq.arm_capture(ctx, self.fake_case(), 'on', [1, 2],
                               self.tmp / 'arm-alive', 'on')
        self.assertEqual(ctx['model_alive_pid'], 4242)
        self.assertEqual(events, [('apply', True)])
        self.assertTrue(ctx['cleanup_failures'])

    def test_preview_is_inert(self):
        args = self.fake_args(execute=False)
        with mock.patch.object(rq, 'load_reference',
                               return_value=self.fake_reference(args)), \
                mock.patch.object(rq, 'verify_common_args',
                                  return_value={'path': 'c', 'sha256': 'c',
                                                'model_path': 'm'}), \
                mock.patch.object(rq, '_plan_cases',
                                  return_value=([self.fake_case()], [0] * 1024)), \
                mock.patch.object(rq, 'load_prefix',
                                  return_value={'path': 'p', 'meta_sha256': 'm',
                                                'contract': {}, 'files': {},
                                                'verify_sha': False}), \
                mock.patch.object(rq, 'disk_preflight', return_value={}), \
                mock.patch.object(rq, 'build_binding', return_value={'binding_sha256': 'b'}), \
                mock.patch.object(rq._GATE, 'DeployClient',
                                  side_effect=AssertionError('deploy in preview')), \
                mock.patch.object(rq, 'run_capture_child',
                                  side_effect=AssertionError('child in preview')):
            with contextlib.redirect_stdout(io.StringIO()):
                code = rq.main(['--capture-binary', str(args.capture_binary),
                            '--common-args-json', str(args.common_args_json),
                            '--cases-dir', str(args.cases_dir),
                            '--reference-campaign', str(args.reference_campaign),
                                '--output', str(args.output)])
        self.assertEqual(code, 0)
        self.assertFalse(args.output.exists())


class BindingGuardTests(TempCase):
    def test_binary_change_is_detected_before_and_after_arm(self):
        binary = self.tmp / 'capture'
        binary.write_bytes(b'original')
        binding = {
            'capture_binary': {'path': str(binary), 'size': binary.stat().st_size,
                               'sha256': rq.sha256_file(binary)},
            'source': HELPER.git_identity(),
            'prefix': {'path': None, 'files': {}},
        }
        libraries = {'files': [], 'sha256': rq.sha256_text(rq.canonical([]))}
        with mock.patch.object(rq, 'active_libraries_for', return_value=[]):
            guard = rq.BindingGuard(binding, binary, {}, libraries, HELPER,
                                    path_provider=lambda path: [])
            guard.verify('pre-arm')
            binary.write_bytes(b'changed')
            with self.assertRaisesRegex(rq.MatrixError, 'capture binary'):
                guard.verify('post-arm')


class StableNllTests(TempCase):
    def _stats(self, name, values):
        write_logits(self.tmp / name, values)
        return rq.raw_stats(self.tmp / name, len(values), 1e-9)

    def test_high_offset_equal_logits_keeps_log2(self):
        stats = self._stats('high.bin', [1e20, 1e20])
        metrics = rq.step_metrics_from_stats(stats, stats, 0, 0.5)
        self.assertAlmostEqual(metrics['nll_off'], math.log(2.0), places=12)
        self.assertEqual(metrics['kl'], 0.0)

    def test_high_offset_unequal_does_not_cancel(self):
        values = [1e20, float(array.array('f', [1e20 + 1e13])[0])]
        stats = self._stats('unequal.bin', values)
        expected = -((stats['values'][0] - stats['maximum']) - stats['log_total'])
        metrics = rq.step_metrics_from_stats(stats, stats, 0, 0.5)
        self.assertAlmostEqual(metrics['nll_off'], expected, places=10)
        self.assertNotEqual(metrics['nll_off'], 0.0)

    def test_zero_minus_1000_keeps_nll_and_zero_kl(self):
        stats = self._stats('range.bin', [0.0, -1000.0])
        metrics = rq.step_metrics_from_stats(stats, stats, 1, 0.5)
        self.assertAlmostEqual(metrics['nll_off'], 1000.0, places=10)
        self.assertEqual(metrics['kl'], 0.0)

    def test_negative_infinity_fails_closed(self):
        write_logits(self.tmp / 'inf.bin', [0.0, -math.inf])
        with self.assertRaisesRegex(rq.MatrixError, 'non-finite'):
            rq.raw_stats(self.tmp / 'inf.bin', 2)


class BindingGuardIdentityTests(TempCase):
    def make_guard(self, *, libraries=None, loaded_paths=None):
        binary = self.tmp / 'capture'
        binary.write_bytes(b'binary')
        model = self.tmp / 'model.gguf'
        model.write_bytes(b'model')
        common = self.tmp / 'common.json'
        common.write_bytes(b'[]')
        frozen = self.tmp / 'frozen-common.json'
        frozen.write_bytes(b'[]')
        libraries = libraries or {'files': [],
                                  'sha256': rq.sha256_text(rq.canonical([]))}
        binding = {
            'capture_binary': {'path': str(binary), 'size': binary.stat().st_size,
                               'sha256': rq.sha256_file(binary)},
            'source': HELPER.git_identity(),
            'prefix': {'path': None, 'files': {}},
            'model': {'path': str(model), 'sha256': rq.sha256_file(model),
                      'size': model.stat().st_size, 'stat': rq.stat_identity(model)},
            'common_args': {'path': str(common), 'sha256': rq.sha256_file(common),
                            'stat': rq.stat_identity(common),
                            'frozen_path': str(frozen),
                            'frozen_sha256': rq.sha256_file(frozen)},
        }
        provider = (lambda path: list(loaded_paths)) if loaded_paths is not None else (lambda path: [])
        guard = rq.BindingGuard(binding, binary, {}, libraries, HELPER,
                                path_provider=provider)
        return guard, model, common, frozen, binary

    def test_model_and_common_changes_fail(self):
        guard, model, common, frozen, _binary = self.make_guard()
        guard.verify('pre')
        model.write_bytes(b'changed')
        with self.assertRaisesRegex(rq.MatrixError, 'model'):
            guard.verify('post')
        guard, _model, common, frozen, _binary = self.make_guard()
        common.write_bytes(b'[1]')
        with self.assertRaisesRegex(rq.MatrixError, 'common args'):
            guard.verify('post')
        guard, _model, _common, frozen, _binary = self.make_guard()
        frozen.write_bytes(b'[2]')
        with self.assertRaisesRegex(rq.MatrixError, 'frozen common'):
            guard.verify('post')
        guard, _model, _common, _frozen, _binary = self.make_guard()
        guard.binding['model']['sha256'] = '0' * 64
        with self.assertRaisesRegex(rq.MatrixError, 'model content SHA changed'):
            guard.verify('post')

    def test_empty_or_missing_loaded_libraries_fail(self):
        guard, _model, _common, _frozen, _binary = self.make_guard()
        with self.assertRaisesRegex(rq.MatrixError, 'loaded library map is empty'):
            guard.verify('post', loaded=[])
        first = self.tmp / 'libllama.so'
        second = self.tmp / 'libggml.so'
        first.write_bytes(b'one')
        second.write_bytes(b'two')
        files = [{'path': str(first), 'size': 3, 'sha256': rq.sha256_file(first)},
                 {'path': str(second), 'size': 3, 'sha256': rq.sha256_file(second)}]
        libraries = {'files': files, 'sha256': rq.sha256_text(rq.canonical(files))}
        guard, _model, _common, _frozen, _binary = self.make_guard(
            libraries=libraries, loaded_paths=[first, second])
        with self.assertRaisesRegex(rq.MatrixError, 'expected loaded libraries missing'):
            guard.verify('post', loaded=[first])
        first.write_bytes(b'xxx')
        with self.assertRaisesRegex(rq.MatrixError, 'content changed'):
            guard.verify('post', loaded=[first, second])


class CaptureChildTests(TempCase):
    class FakeProcess:
        def __init__(self, lines, returncode=0):
            self.stdout = io.StringIO(''.join(lines))
            self.pid = 4242
            self.returncode = returncode
            self.alive = True
            self.terminated = False
            self.killed = False

        def wait(self, timeout=None):
            self.alive = False
            return self.returncode

        def poll(self):
            return None if self.alive else self.returncode

        def terminate(self):
            self.terminated = True
            self.alive = False

        def kill(self):
            self.killed = True
            self.alive = False

    def run_child(self, lines, loaded=('/x/libllama.so',)):
        process = self.FakeProcess(lines)
        with mock.patch.object(rq._GATE, 'loaded_library_paths',
                               return_value=[Path(item) for item in loaded]):
            result = rq.run_capture_child(
                ['child'], {}, self.tmp / 'capture.log', 10, self.tmp,
                popen_factory=lambda *args, **kwargs: process)
        return result

    def test_startup_marker_samples_actual_libraries(self):
        result = self.run_child([
            'noise\n', 'llama_context: backend_ptrs.size() = 2\n',
            'CAPTURE step=0 position=1 teacher=1 argmax=1 finite=1\n'])
        self.assertEqual(result['startup_event'],
                         'llama_context: backend_ptrs.size() = 2')
        self.assertEqual(result['loaded_startup'], [Path('/x/libllama.so')])
        self.assertFalse(result['startup_fallback'])

    def test_empty_map_is_recorded_not_accepted(self):
        result = self.run_child([
            'llama_context: backend_ptrs.size() = 2\n'], loaded=())
        self.assertIsNone(result['loaded_startup'])
        self.assertIn('empty', result['loaded_error'])

    def test_missing_startup_marker_is_fail_closed(self):
        result = self.run_child(['CAPTURE PASS steps=2 vocab=8\n'])
        self.assertIsNone(result['startup_event'])

    def test_step0_is_explicit_fallback(self):
        result = self.run_child([
            'CAPTURE step=0 position=1 teacher=1 argmax=1 finite=1\n'])
        self.assertTrue(result['startup_fallback'])
        self.assertIn('CAPTURE step=0', result['startup_event'])

    def test_startup_marker_matches_real_gold_log(self):
        log = Path('/home/hjotha/.codex/artifacts/rkva-priorities-20261026/'
                   'raw-gold-http-off-r1.log')
        if not log.is_file():
            self.skipTest('real gold log is not present')
        lines = log.read_text(errors='replace').splitlines()
        matches = [line for line in lines if rq.STARTUP_MARKER in line]
        self.assertTrue(matches)
        self.assertIn('backend_ptrs.size()', matches[0])

    def test_cleanup_reap_failure_is_explicit(self):
        class ReapFailure:
            pid = 99
            def __init__(self):
                self.calls = 0
            def poll(self):
                return None
            def terminate(self):
                pass
            def wait(self, timeout=None):
                self.calls += 1
                raise subprocess.TimeoutExpired('child', timeout)
            def kill(self):
                pass
        details = rq._terminate_child(ReapFailure(), term_timeout=0.1, kill_timeout=0.1)
        self.assertTrue(details['alive'])
        self.assertFalse(details['reaped'])
        fatal = rq._cleanup_failure(details, 'test')
        self.assertIsInstance(fatal, rq.ChildCleanupError)
        self.assertTrue(fatal.alive)

    def test_terminate_error_is_explicit_but_dead_can_restore(self):
        class TerminateFailure:
            pid = 100
            def __init__(self):
                self.alive = True
            def poll(self):
                return None if self.alive else 0
            def terminate(self):
                self.alive = False
                raise OSError('terminate failed')
            def wait(self, timeout=None):
                return 0
        details = rq._terminate_child(TerminateFailure(), term_timeout=0.1, kill_timeout=0.1)
        self.assertFalse(details['alive'])
        self.assertEqual(details['terminate_error'], 'terminate failed')
        fatal = rq._cleanup_failure(details, 'test')
        self.assertIsInstance(fatal, rq.ChildCleanupError)
        self.assertFalse(fatal.alive)

    def test_reap_error_is_not_silenced(self):
        class WaitFailure:
            pid = 101
            def __init__(self):
                self.alive = True
            def poll(self):
                return None if self.alive else 0
            def terminate(self):
                pass
            def wait(self, timeout=None):
                if self.alive:
                    self.alive = False
                    raise OSError('wait failed')
                return 0
            def kill(self):
                pass
        details = rq._terminate_child(WaitFailure(), term_timeout=0.1, kill_timeout=0.1)
        self.assertIsNotNone(details['reap_error'])
        fatal = rq._cleanup_failure(details, 'test')
        self.assertIsInstance(fatal, rq.ChildCleanupError)
        self.assertFalse(fatal.alive)


class EnvironmentAndDiskTests(TempCase):
    def test_frozen_environment_isolated_and_mask_pack_absent(self):
        arm = self.tmp / 'arm'
        environment = rq.frozen_environment(
            {'GGML_RKVA_PROFILE': '1', 'GGML_RKVA_TEST': 'x',
             'CUDA_CACHE_PATH': '/old', 'TMPDIR': '/old'}, arm)
        self.assertEqual(environment['GGML_RKVA_TEST'], 'x')
        self.assertEqual(environment['TMPDIR'], str(arm / 'tmp'))
        self.assertEqual(environment['CUDA_CACHE_PATH'], str(arm / 'cuda-cache'))
        self.assertNotIn('GGML_CUDA_MASK_PACK', environment)
        with self.assertRaisesRegex(rq.RefuseError, 'PROFILE'):
            rq.frozen_environment({}, arm)

    def test_disk_preflight_budget(self):
        output = self.tmp / 'out'
        with mock.patch.object(rq.shutil, 'disk_usage',
                               return_value=SimpleNamespace(free=10)):
            with self.assertRaisesRegex(rq.RefuseError, 'insufficient disk'):
                rq.disk_preflight(output, [{}], 2, VOCAB, 1.0)
        with mock.patch.object(rq.shutil, 'disk_usage',
                               return_value=SimpleNamespace(free=10 ** 12)):
            estimate = rq.disk_preflight(output, [{}], 2, VOCAB, 1.0)
        self.assertGreater(estimate['raw_bytes'], 0)

    def test_output_must_be_new(self):
        output = self.tmp / 'out'
        output.mkdir()
        with self.assertRaisesRegex(rq.RefuseError, 'already exists'):
            rq.disk_preflight(output, [{}], 2, VOCAB)

    def test_compute_apps_parser(self):
        pids = rq._compute_app_pids('123, python\n456, llama\n\n')
        self.assertEqual(pids, {123, 456})


if __name__ == '__main__':
    unittest.main()
