"""Hardware-free regressions for the INT8 RKVA teacher-forced quality gate.

Run:
  nice -n 19 env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
      discover -s tools/xbox/tests -p test_rkva_quality_gate.py -v

No test starts llama-server, contacts the Xbox, touches LocalState or a model.
Every process/deploy/HTTP interaction is a fake.
"""
import array
import contextlib
import gzip
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import tempfile
import tracemalloc
from types import SimpleNamespace
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'rkva-quality-gate.py'
spec = importlib.util.spec_from_file_location('rkva_quality_gate', SCRIPT)
qg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qg)


def response_from(probabilities, *, cache_n=0, prompt_n=0, predicted_n=1,
                  temperature=0):
    """Build a full-vocab response; zero probabilities use the server sentinel."""
    entries = []
    for token, probability in enumerate(probabilities):
        logprob = (qg.SENTINEL_THRESHOLD - 1) if probability == 0 else math.log(probability)
        entries.append({'id': token, 'logprob': logprob})
    return {
        'completion_probabilities': [{'top_logprobs': entries}],
        'timings': {'predicted_n': predicted_n, 'cache_n': cache_n, 'prompt_n': prompt_n},
        'generation_settings': {'temperature': temperature},
    }


def parsed(probabilities, tolerance=1e-3):
    return qg.parse_distribution(response_from(probabilities), len(probabilities), tolerance)


def fake_tokenizer(vocab=4096):
    def tokenize(text):
        words = text.split()
        if not words:
            return [1]
        return [int.from_bytes(__import__('hashlib').sha256(word.encode()).digest()[:4],
                               'little') % vocab for word in words]
    return tokenize


def gate_args(**overrides):
    arguments = ['--model', 'model.gguf', '--output', 'out', '--p', '1024',
                 '--context', '131072', '--lengths', '5120',
                 '--cases', 'portuguese,retrieval', '--teacher-steps', '8']
    for key, value in overrides.items():
        arguments += ['--' + key.replace('_', '-'), str(value)]
    return qg.validate(qg.parser().parse_args(arguments))


def step(nll_off=1.0, nll_on=1.0, kl=0.0, agree=True, high_confidence=False):
    return {'nll_off': nll_off, 'nll_on': nll_on,
            'nll_delta': nll_on - nll_off, 'kl': kl,
            'top1_agree': agree, 'high_confidence_divergence': high_confidence}


def case_result(name='portuguese-5120', category='portuguese', length=5120,
                steps=None, answer=None, status='PASS'):
    steps = steps if steps is not None else [step()]
    return {'case': name, 'category': category, 'length': length,
            'status': status, 'steps': steps, 'summary': qg.summarize_steps(steps),
            'answer': answer}


class DistributionTests(unittest.TestCase):
    def test_arbitrary_id_order_is_aligned_by_id(self):
        response = response_from([0.5, 0.0, 0.5])
        response['completion_probabilities'][0]['top_logprobs'] = [
            {'id': 2, 'logprob': math.log(0.5)},
            {'id': 0, 'logprob': math.log(0.5)},
            {'id': 1, 'logprob': qg.SENTINEL_THRESHOLD - 1},
        ]
        result = qg.parse_distribution(response, 3, 1e-3)
        self.assertEqual(list(result['probs']), [0.5, 0.0, 0.5])
        self.assertEqual(result['top1'], 0)
        self.assertEqual(result['top2'], 2)
        self.assertAlmostEqual(result['margin'], 0.0)

    def test_sentinel_is_zero_not_a_huge_negative_probability(self):
        result = qg.parse_distribution(response_from([0.7, 0.0, 0.3]), 3, 1e-3)
        self.assertEqual(result['probs'][1], 0.0)
        self.assertAlmostEqual(result['mass'], 1.0)

    def test_missing_duplicate_nan_and_out_of_range_ids_are_fatal(self):
        base = response_from([0.5, 0.5])
        variants = []
        missing = json.loads(json.dumps(base)); missing['completion_probabilities'][0]['top_logprobs'].pop()
        variants.append(missing)
        duplicate = json.loads(json.dumps(base)); duplicate['completion_probabilities'][0]['top_logprobs'][1]['id'] = 0
        variants.append(duplicate)
        nan = json.loads(json.dumps(base)); nan['completion_probabilities'][0]['top_logprobs'][0]['logprob'] = float('nan')
        variants.append(nan)
        null = json.loads(json.dumps(base)); null['completion_probabilities'][0]['top_logprobs'][0]['logprob'] = None
        variants.append(null)
        boolean = json.loads(json.dumps(base)); boolean['completion_probabilities'][0]['top_logprobs'][0]['id'] = True
        variants.append(boolean)
        outside = json.loads(json.dumps(base)); outside['completion_probabilities'][0]['top_logprobs'][0]['id'] = 2
        variants.append(outside)
        positive = json.loads(json.dumps(base)); positive['completion_probabilities'][0]['top_logprobs'][0]['logprob'] = 0.1
        variants.append(positive)
        for variant in variants:
            with self.subTest(variant=variant):
                with self.assertRaises(qg.GateError):
                    qg.parse_distribution(variant, 2, 1e-3)

    def test_mass_temperature_error_and_truncation_are_fatal(self):
        wrong_mass = response_from([0.5, 0.4])
        with self.assertRaises(qg.GateError):
            qg.parse_distribution(wrong_mass, 2, 1e-3)
        hot = response_from([0.5, 0.5], temperature=1)
        with self.assertRaises(qg.GateError):
            qg.parse_distribution(hot, 2, 1e-3)
        truncated = response_from([0.5, 0.5]); truncated['truncated'] = True
        with self.assertRaises(qg.GateError):
            qg.parse_distribution(truncated, 2, 1e-3)
        error = response_from([0.5, 0.5]); error['error'] = 'boom'
        with self.assertRaises(qg.GateError):
            qg.parse_distribution(error, 2, 1e-3)
        short = response_from([0.5, 0.5]); short['completion_probabilities'][0]['top_logprobs'].pop()
        with self.assertRaises(qg.GateError):
            qg.parse_distribution(short, 2, 1e-3)


class MetricTests(unittest.TestCase):
    def test_golden_kl_and_nll(self):
        off = parsed([0.75, 0.25])
        on = parsed([0.5, 0.5])
        result = qg.step_metrics(off, on, 1, 0.5)
        self.assertAlmostEqual(result['nll_off'], -math.log(0.25))
        self.assertAlmostEqual(result['nll_on'], -math.log(0.5))
        self.assertAlmostEqual(result['nll_delta'], -math.log(2))
        self.assertAlmostEqual(result['kl'], 0.75 * math.log(1.5) + 0.25 * math.log(0.5))
        self.assertTrue(result['top1_agree'])

    def test_equal_distributions_are_zero(self):
        value = parsed([0.4, 0.35, 0.25])
        result = qg.step_metrics(value, value, 0, 0.5)
        self.assertAlmostEqual(result['kl'], 0.0)
        self.assertAlmostEqual(result['nll_delta'], 0.0)
        self.assertFalse(result['high_confidence_divergence'])

    def test_summary_uses_ratio_mean_and_nearest_rank_p99(self):
        steps = [step(nll_off=1.0, nll_on=1.0 + math.log(1.01), kl=0.001),
                 step(nll_off=1.0, nll_on=1.0 + math.log(1.01), kl=0.02)]
        summary = qg.summarize_steps(steps)
        self.assertAlmostEqual(summary['ppl_ratio'], 1.01, places=9)
        self.assertAlmostEqual(summary['kl_mean'], 0.0105)
        self.assertAlmostEqual(summary['kl_p99'], 0.02)
        self.assertEqual(summary['top1_agreement'], 1.0)

    def test_nonfinite_nll_or_kl_is_not_a_summary_metric(self):
        summary = qg.summarize_steps([step(nll_off=float('inf'), nll_on=1.0, kl=0.0)])
        self.assertFalse(summary['finite'])
        self.assertIsNone(summary['ppl_ratio'])
        self.assertIsNone(summary['kl_mean'])

    def test_nonfinite_step_is_still_writable_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'x.json'
            qg.write_json(path, {'value': float('inf'), 'list': [float('nan')]})
            self.assertEqual(json.loads(path.read_text()), {'value': None, 'list': [None]})

    def test_high_confidence_divergence_is_flagged(self):
        off = parsed([0.9, 0.1])
        on = parsed([0.1, 0.9])
        result = qg.step_metrics(off, on, 0, 0.5)
        self.assertFalse(result['top1_agree'])
        self.assertTrue(result['high_confidence_divergence'])


class CaseGeneratorTests(unittest.TestCase):
    def spec(self, category='portuguese', length=5120, p=1024):
        case = qg.build_case_spec(category, category, length, 17)
        if category == 'retrieval':
            case['query'] = ('Pergunta: qual é o valor associado à chave M4X9? '
                             'Responda apenas com o valor. Resposta:')
            case['answer'] = 'L3S8'
        if category == 'factual':
            case['query'] = ('Qual é a capital da França? '
                             'Responda apenas com a resposta. Resposta:')
            case['answer'] = 'Paris'
        return case

    def test_generator_is_deterministic_and_hits_exact_suffix_length(self):
        tokenize = fake_tokenizer()
        first = qg.fit_case_tokens(self.spec(), 1024, tokenize, 4096)
        second = qg.fit_case_tokens(self.spec(), 1024, tokenize, 4096)
        self.assertEqual(first['tokens'], second['tokens'])
        self.assertEqual(first['tokens_sha256'], second['tokens_sha256'])
        self.assertEqual(first['tail_length'], 4096)
        self.assertEqual(len(first['tokens']), 4096)
        self.assertTrue(all(0 <= token < 4096 for token in first['tokens']))

    def test_retrieval_and_factual_keep_the_end_key_exactly_128_tokens_away(self):
        tokenize = fake_tokenizer()
        for category in ('retrieval', 'factual'):
            result = qg.fit_case_tokens(self.spec(category), 1024, tokenize, 4096)
            self.assertEqual(result['metadata']['end_key_distance_tokens'], 128)
            self.assertTrue(result['answer_tokens'])
            self.assertEqual(len(result['tokens']), 4096)

    def test_shared_prefix_is_not_part_of_the_generated_suffix(self):
        tokenize = fake_tokenizer()
        base = list(range(1024))
        result = qg.fit_case_tokens(self.spec(), 1024, tokenize, 4096)
        self.assertEqual(len(result['tokens']), 4096)
        self.assertNotEqual(result['tokens'][:10], base[:10])
        self.assertEqual(result['base_length'], 1024)

    def test_external_case_requires_the_declared_length(self):
        tokenize = fake_tokenizer()
        text = ' '.join(f'w{index}' for index in range(64))
        exact = {'name': 'external', 'category': 'portuguese', 'length': 1088,
                 'seed': 1, 'text': text, 'keys': [], 'values': [],
                 'query': None, 'answer': None, 'text_sha256': qg.sha256_text(text)}
        result = qg.fit_case_tokens(exact, 1024, tokenize, 4096)
        self.assertEqual(len(result['tokens']), 64)
        wrong = dict(exact, length=1089)
        with self.assertRaises(ValueError):
            qg.fit_case_tokens(wrong, 1024, tokenize, 4096)

    def test_external_retrieval_case_needs_an_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'cases.json'
            path.write_text(json.dumps({'cases': [{
                'name': 'x', 'category': 'retrieval', 'tail_text': 'a b c',
            }]}))
            args = SimpleNamespace(cases_json=path, seed=1, lengths_list=[5120], p=1024)
            with self.assertRaises(ValueError):
                qg.load_external_cases(args)

    def test_external_case_ids_and_answers_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'cases.json'
            path.write_text(json.dumps({'cases': [{
                'name': 'target', 'category': 'retrieval',
                'tail_text': 'alpha beta gamma delta',
                'lengths': [1028], 'answer_ids': [1, 2],
            }]}))
            args = SimpleNamespace(cases_json=path, seed=1, lengths_list=[1028], p=1024)
            specs = qg.load_external_cases(args)
            self.assertEqual(specs[0]['answer_ids'], [1, 2])
            result = qg.fit_case_tokens(specs[0], 1024, fake_tokenizer(), 4096)
            self.assertEqual(result['tokens'], fake_tokenizer()('alpha beta gamma delta'))
            self.assertEqual(result['answer_tokens'], [1, 2])

    def test_smoke_is_one_case_and_two_steps(self):
        args = qg.validate(qg.parser().parse_args([
            '--model', 'm', '--output', 'out', '--p', '1024',
            '--context', '131072', '--smoke']))
        self.assertEqual(args.cases, 'portuguese')
        self.assertEqual(args.lengths_list, [5120])
        self.assertEqual(args.teacher_steps, 2)

    def test_selected_case_plan_is_unique_and_hashed(self):
        args = gate_args()
        plan, specs = qg.case_plan(args)
        self.assertEqual(len(plan), 2)
        self.assertEqual(len({item['name'] for item in plan}), 2)
        self.assertEqual(qg.sha256_text(qg.canonical(plan)),
                         qg.sha256_text(qg.canonical(plan)))

    def test_smoke_with_the_frozen_profile_is_106496_and_two_steps(self):
        args = qg.validate(qg.parser().parse_args([
            '--model', 'm', '--output', 'out', '--smoke']))
        self.assertEqual(args.lengths_list, [106496])
        self.assertEqual(args.cases, 'portuguese')
        self.assertEqual(args.teacher_steps, 2)
        self.assertEqual(args.p, 102400)

    def test_default_teacher_steps_is_128(self):
        args = qg.validate(qg.parser().parse_args([
            '--model', 'm', '--output', 'out', '--p', '256',
            '--context', '131072', '--lengths', '512']))
        self.assertEqual(args.teacher_steps, 128)


class ActivationTests(unittest.TestCase):
    def helper(self):
        return SimpleNamespace(
            wavefront_activation=lambda lines, tile, tiles: {'requested_geometry_confirmed': True},
            tcp_prefetch_activation=lambda lines: {'two_frame_overlap_confirmed': True})

    def engine(self):
        return {
            'kvarn4_multi_query': {
                'mq2': {'calls': 9},
                'mq2_i8': {'calls': 0, 'available': True, 'exact': False},
            },
            'kvarn4_decode': {'calls': 12},
            'kvarn4_decode_rowlane': {'calls': 12, 'enabled': True},
            'bulk': {'fp8_gpu_calls': 4, 'fp8_cpu_fallback_calls': 0,
                     'bulk_segments': 8, 'bulk_queries': 72},
            'buckets': {
                'attention/kvarn4-prefix/optimized-kvarn4_mq2_fine/nq2-16/r16k-32k/warm':
                    {'samples': 4},
                'attention/kvarn4-prefix/optimized-kvarn4_mq2_i8/nq2-16/r16k-32k/warm':
                    {'samples': 4},
            },
        }

    def test_positive_on_and_zero_off_activation_is_mandatory(self):
        off = self.engine()
        on = self.engine()
        on['kvarn4_multi_query']['mq2_i8']['calls'] = 11
        result = qg.activation_checks(off, on, [], [], self.helper())
        self.assertTrue(result['pass'], result['failed'])
        on['kvarn4_multi_query']['mq2_i8']['calls'] = 0
        self.assertFalse(qg.activation_checks(off, on, [], [], self.helper())['pass'])

    def test_off_must_not_use_the_int8_kernel(self):
        off = self.engine()
        off['kvarn4_multi_query']['mq2_i8']['calls'] = 5
        on = self.engine(); on['kvarn4_multi_query']['mq2_i8']['calls'] = 5
        result = qg.activation_checks(off, on, [], [], self.helper())
        self.assertIn('off_int8_zero', result['failed'])

    def test_rowlane_fallback_wavefront_and_tcp_are_gated(self):
        off = self.engine(); on = self.engine()
        on['kvarn4_multi_query']['mq2_i8']['calls'] = 2
        on['kvarn4_decode_rowlane']['calls'] = 0
        result = qg.activation_checks(off, on, [], [], self.helper())
        self.assertIn('on_rowlane_positive', result['failed'])
        on = self.engine(); on['kvarn4_multi_query']['mq2_i8']['calls'] = 2
        on['bulk']['fp8_cpu_fallback_calls'] = 1
        result = qg.activation_checks(off, on, [], [], self.helper())
        self.assertIn('on_fp8_cpu_fallback_zero', result['failed'])
        no_wf = SimpleNamespace(
            wavefront_activation=lambda lines, tile, tiles: {'requested_geometry_confirmed': False},
            tcp_prefetch_activation=lambda lines: {'two_frame_overlap_confirmed': False})
        result = qg.activation_checks(off, on, [], [], no_wf)
        self.assertIn('off_wavefront_confirmed', result['failed'])
        self.assertIn('on_tcp_overlap_confirmed', result['failed'])


class GateTests(unittest.TestCase):
    def evaluate(self, results, activation=True, lifecycle=(), control=None, **overrides):
        args = gate_args(**overrides)
        return qg.evaluate_gates(results, args.gates, activation, list(lifecycle), control, args)

    def test_complete_matrix_passes_but_never_authorizes_default(self):
        answer = {'baseline_correct': True, 'candidate_correct': True, 'regression': False}
        results = [case_result(), case_result('retrieval-5120', 'retrieval', 5120,
                                              answer=answer)]
        final = self.evaluate(results, control={'pass': True})
        self.assertEqual(final['status'], 'PASS')
        self.assertEqual(final['qualification_label'], 'QUALIFIED_SCOPE_ONLY')
        self.assertFalse(final['int8_default_authorized'])

    def test_nonfinite_step_fails_the_distribution_gate(self):
        results = [case_result(steps=[step(nll_off=float('inf'))])]
        final = self.evaluate(results, control={'pass': True})
        self.assertEqual(final['status'], 'FAIL')
        self.assertFalse(final['checks']['distributions_finite']['pass'])

    def test_missing_metrics_never_pass(self):
        broken = case_result(steps=[])
        final = self.evaluate([broken], control={'pass': True})
        self.assertEqual(final['status'], 'FAIL')
        self.assertFalse(final['checks']['distributions_finite']['pass'])
        self.assertFalse(final['checks']['aggregate_ppl_ratio']['pass'])

    def test_each_case_nll_delta_is_gated(self):
        bad = case_result(steps=[step(nll_off=0.0, nll_on=0.03)])
        final = self.evaluate([bad], control={'pass': True})
        self.assertEqual(final['status'], 'FAIL')
        self.assertFalse(final['checks']['case_portuguese-5120_nll_delta']['pass'])

    def test_ppl_and_kl_gates_are_independent(self):
        ratio = case_result(steps=[step(nll_off=0.0, nll_on=0.02)])
        final = self.evaluate([ratio], control={'pass': True})
        self.assertFalse(final['checks']['aggregate_ppl_ratio']['pass'])
        kl = case_result(steps=[step(kl=0.002)])
        final = self.evaluate([kl], control={'pass': True})
        self.assertFalse(final['checks']['aggregate_kl_mean']['pass'])
        kl99 = case_result(steps=[step(kl=0.02)])
        final = self.evaluate([kl99], control={'pass': True})
        self.assertFalse(final['checks']['aggregate_kl_p99']['pass'])

    def test_retrieval_baseline_must_be_informative_and_must_not_regress(self):
        uninformed = case_result('retrieval-x', 'retrieval', 5120, answer=None)
        final = self.evaluate([uninformed], control={'pass': True})
        self.assertFalse(final['checks']['retrieval_baseline_informative']['pass'])
        regression = {'baseline_correct': True, 'candidate_correct': False,
                      'regression': True}
        failed = case_result('retrieval-x', 'retrieval', 5120, answer=regression)
        final = self.evaluate([failed], control={'pass': True})
        self.assertFalse(final['checks']['no_retrieval_regression']['pass'])
        self.assertEqual(final['status'], 'FAIL')

    def test_high_confidence_divergence_is_needs_review_not_pass(self):
        results = [case_result(steps=[step(agree=False, high_confidence=True)]),
                   case_result('retrieval-x', 'retrieval', 5120,
                               answer={'baseline_correct': True,
                                       'candidate_correct': True,
                                       'regression': False})]
        final = self.evaluate(results, control={'pass': True})
        self.assertEqual(final['status'], 'NEEDS_REVIEW')
        self.assertEqual(final['qualification_label'], 'NEEDS_REVIEW')

    def test_failed_activation_or_lifecycle_is_fail(self):
        final = self.evaluate([case_result()], activation=False, control={'pass': True})
        self.assertEqual(final['status'], 'FAIL')
        final = self.evaluate([case_result()], lifecycle=['server exited 9'],
                              control={'pass': True})
        self.assertEqual(final['status'], 'FAIL')

    def test_missing_determinism_control_is_not_a_silent_pass(self):
        final = self.evaluate([case_result('portuguese-130560', length=130560)],
                              control=None)
        self.assertEqual(final['status'], 'FAIL')
        self.assertFalse(final['checks']['determinism_control']['pass'])

    def test_smoke_is_labeled_smoke_only_and_does_not_require_control(self):
        args = qg.validate(qg.parser().parse_args([
            '--model', 'm', '--output', 'out', '--p', '1024',
            '--context', '131072', '--smoke']))
        results = [case_result('portuguese-5120', 'portuguese', 5120)]
        final = qg.evaluate_gates(results, args.gates, True, [], None, args)
        self.assertEqual(final['qualification_label'], 'SMOKE_ONLY')


class FlagTests(unittest.TestCase):
    class FakeDeploy:
        def __init__(self, files=(), fail_delete=False, fail_upload=False):
            self.files = set(files)
            self.fail_delete = fail_delete
            self.fail_upload = fail_upload
            self.calls = []

        def __call__(self, argv, env, timeout):
            self.calls.append(argv)
            command = argv[1]
            if command == 'list-localstate':
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({'Items': [{'Name': name} for name in sorted(self.files)]}),
                    stderr='')
            if command == 'upload-file':
                if self.fail_upload:
                    return SimpleNamespace(returncode=1, stdout='', stderr='upload failed')
                self.files.add(argv[5])
                return SimpleNamespace(returncode=0, stdout='ok', stderr='')
            if command == 'delete-file':
                name = argv[3]
                if name in self.files:
                    if self.fail_delete:
                        return SimpleNamespace(returncode=1, stdout='', stderr='delete failed')
                    self.files.remove(name)
                    return SimpleNamespace(returncode=0, stdout='', stderr='')
                return SimpleNamespace(returncode=22, stdout='', stderr='curl: (22) 404')
            if command == 'pfn':
                return SimpleNamespace(
                    returncode=0,
                    stdout='GianlucaMazza.xllama_1.6.0.305_x64__pj67f1fcj4n14\n',
                    stderr='')
            return SimpleNamespace(returncode=1, stdout='', stderr='unexpected')

    def deploy(self, directory, fake):
        env = Path(directory) / 'env'
        env.write_text('XBOX_IP=192.0.2.10\nXBOX_USER=operator\nXBOX_PASS=secret-value\n')
        return qg.DeployClient(Path(directory) / 'deploy.sh', env, 10, runner=fake)

    def test_original_flags_are_restored_after_both_arms(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake = self.FakeDeploy(files={qg.INT8_FLAG, qg.ROWLAND_OPT_OUT_FLAG})
            client = self.deploy(temporary, fake)
            pfn = client.pfn()
            flags = qg.LocalStateFlags(client, pfn, temporary)
            original = flags.capture_original()
            self.assertEqual(original, {qg.INT8_FLAG: True, qg.ROWLAND_OPT_OUT_FLAG: True})
            with flags.arm(False):
                self.assertEqual(fake.files, set())
            self.assertEqual(fake.files, {qg.INT8_FLAG, qg.ROWLAND_OPT_OUT_FLAG})
            with flags.arm(True):
                self.assertEqual(fake.files, {qg.INT8_FLAG})
            self.assertEqual(fake.files, {qg.INT8_FLAG, qg.ROWLAND_OPT_OUT_FLAG})
            self.assertTrue((Path(temporary) / 'flags-restored.json').is_file())

    def test_absent_flag_delete_404_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake = self.FakeDeploy()
            client = self.deploy(temporary, fake)
            flags = qg.LocalStateFlags(client, client.pfn(), temporary)
            flags.capture_original()
            flags.set(qg.INT8_FLAG, False)
            self.assertNotIn(qg.INT8_FLAG, fake.files)

    def test_real_delete_or_upload_failure_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake = self.FakeDeploy(files={qg.INT8_FLAG}, fail_delete=True)
            client = self.deploy(temporary, fake)
            flags = qg.LocalStateFlags(client, client.pfn(), temporary)
            flags.capture_original()
            with self.assertRaises(qg.GateError):
                flags.set(qg.INT8_FLAG, False)
        with tempfile.TemporaryDirectory() as temporary:
            fake = self.FakeDeploy(fail_upload=True)
            client = self.deploy(temporary, fake)
            flags = qg.LocalStateFlags(client, client.pfn(), temporary)
            flags.capture_original()
            with self.assertRaises(qg.GateError):
                flags.set(qg.INT8_FLAG, True)

    def test_large_localstate_json_is_parsed_from_full_stdout(self):
        names = [f'file-{index:04d}.dat' for index in range(800)]
        names[0] = qg.INT8_FLAG
        names[len(names) // 2] = qg.ROWLAND_OPT_OUT_FLAG
        names[-1] = 'tail.flag'
        names.append('secret-value')
        payload = json.dumps({'Items': [{'Name': name} for name in names]})
        self.assertGreater(len(payload), 20000)

        class LargeDeploy(self.FakeDeploy):
            def __call__(self, argv, env, timeout):
                if argv[1] == 'list-localstate':
                    return SimpleNamespace(
                        returncode=0,
                        stdout=payload,
                        stderr='password secret-value')
                if argv[1] == 'pfn':
                    return SimpleNamespace(
                        returncode=0,
                        stdout='GianlucaMazza.xllama_1.6.0.305_x64__pj67f1fcj4n14\n',
                        stderr='')
                return super().__call__(argv, env, timeout)

        with tempfile.TemporaryDirectory() as temporary:
            fake = LargeDeploy()
            client = self.deploy(temporary, fake)
            pfn = client.pfn()
            result = client.run('list-localstate', pfn)
            self.assertEqual(client.list_names(pfn), set(names))
            self.assertGreater(len(result['_stdout_full']), 4000)
            self.assertLessEqual(len(result['stdout']), 4000)
            self.assertNotIn('secret-value', result['stdout'])
            self.assertNotIn('secret-value', result['stderr'])
            self.assertEqual(result['_stdout_full'], payload)

    def test_pfn_identity_and_secret_redaction(self):
        with self.assertRaises(qg.RefuseError):
            qg.validate_pfn('VenereLabs.xllama_1.0.0.0_x64__abc')
        with self.assertRaises(qg.RefuseError):
            qg.validate_pfn('GianlucaMazza.xllama_1.0.0.0_x64__abc',
                            'Warning: 2 xllama packages registered')
        self.assertEqual(qg.validate_pfn('GianlucaMazza.xllama_1.0.0.0_x64__abc'),
                         'GianlucaMazza.xllama_1.0.0.0_x64__abc')
        self.assertEqual(qg.redact('token=secret-value', ['secret-value']),
                         'token=***')

    def test_missing_credentials_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'env'
            path.write_text('XBOX_IP=192.0.2.1\n')
            with self.assertRaises(qg.RefuseError):
                qg.load_xbox_env(path)


class FakeDrain:
    def __init__(self, lines=None, listening=True):
        self.lines = lines or []
        self.listening = listening
        self.ready = SimpleNamespace(wait=lambda timeout: listening)
        self.thread = SimpleNamespace(join=lambda timeout=None: None,
                                      is_alive=lambda: False)


class FakeProcess:
    def __init__(self, exit_code=0, exit_before_stop=None):
        self.pid = 4242
        self.returncode = exit_code if exit_before_stop is None else exit_before_stop
        self.exit_before_stop = exit_before_stop
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.exit_before_stop

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.exit_before_stop is None:
            return self.returncode
        return self.exit_before_stop


class FakeHelper:
    PINNED_ENV = {'GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB': '128'}

    def server_command(self, namespace, port):
        return ['llama-server', '--port', str(port)]

    def environment(self, namespace, output):
        return {}, {}

    def validate_startup(self, namespace, lines):
        return []

    def loaded_libraries(self, pid):
        return []

    def classify_logs(self, lines):
        return [], []


class SessionTests(unittest.TestCase):
    def args(self):
        return qg.validate(qg.parser().parse_args([
            '--model', 'm', '--output', 'out', '--p', '256',
            '--context', '131072', '--lengths', '512', '--cases', 'portuguese']))

    def make(self, directory, process, http=None, drain=None):
        return qg.ServerSession(
            FakeHelper(), qg.helper_args(self.args()), self.args(), directory,
            {'sample_ms': 10, 'slack_mb': 64}, 'arm',
            popen=lambda *a, **k: process,
            http=http or (lambda port, endpoint, body=None, timeout=30: (
                {'data': [{'meta': {'n_vocab': 7}}]} if endpoint == '/v1/models' else {})),
            drain_factory=lambda *a, **k: drain or FakeDrain(),
            loaded_libraries=lambda pid: [])

    def test_startup_timeout_terminates_the_owned_child(self):
        process = FakeProcess()
        with tempfile.TemporaryDirectory() as temporary:
            session = self.make(Path(temporary) / 'arm', process,
                                drain=FakeDrain(listening=False))
            with self.assertRaises(qg.GateError):
                session.__enter__()
            self.assertTrue(process.terminated)
            self.assertTrue((Path(temporary) / 'arm' / 'manifest.json').is_file())

    def test_nonzero_exit_is_a_failure_and_graceful_exit_zero_is_not(self):
        process = FakeProcess(exit_before_stop=1)
        with tempfile.TemporaryDirectory() as temporary:
            session = self.make(Path(temporary) / 'arm', process)
            with self.assertRaises(qg.GateError) as error:
                with session:
                    pass
            self.assertIn('did not exit 0', str(error.exception))
        process = FakeProcess()
        with tempfile.TemporaryDirectory() as temporary:
            session = self.make(Path(temporary) / 'arm', process)
            with session:
                self.assertEqual(session.n_vocab(), 7)
            self.assertTrue(process.terminated)

    def test_request_timeout_propagates_and_still_stops(self):
        process = FakeProcess()

        def http(port, endpoint, body=None, timeout=30):
            if endpoint == '/v1/models':
                return {'data': [{'meta': {'n_vocab': 7}}]}
            if endpoint == '/completion':
                raise TimeoutError('request timed out')
            return {}

        with tempfile.TemporaryDirectory() as temporary:
            session = self.make(Path(temporary) / 'arm', process, http=http)
            with self.assertRaises(TimeoutError):
                with session:
                    session.request('/completion', {})
            self.assertTrue(process.terminated)


class FakeSession:
    def __init__(self, vocab, responses):
        self.vocab = vocab
        self.responses = responses
        self.requests = []

    def request(self, endpoint, body=None, timeout=None):
        self.requests.append((endpoint, body))
        return self.responses(body)

    def n_vocab(self):
        return self.vocab


class TeacherProtocolTests(unittest.TestCase):
    def args(self):
        return qg.validate(qg.parser().parse_args([
            '--model', 'm', '--output', 'out', '--p', '256',
            '--context', '131072', '--lengths', '512', '--cases', 'portuguese',
            '--teacher-steps', '2']))

    def test_prompt_hashes_are_deterministic_and_misalignment_is_fatal(self):
        base = [0] * 256
        suffix = [1, 2]
        hashes = qg.teacher_prompt_hashes(base, suffix, [3], 2)
        self.assertEqual(hashes, qg.teacher_prompt_hashes(base, suffix, [3], 2))
        qg.verify_prompt_alignment(hashes, hashes)
        changed = list(hashes); changed[1] = '0' * 64
        with self.assertRaises(qg.GateError):
            qg.verify_prompt_alignment(hashes, changed)

    def test_off_teacher_sequence_is_the_only_sequence_given_to_on(self):
        args = self.args()
        base = [0] * 256
        case = {'name': 'portuguese-512', 'category': 'portuguese', 'length': 512,
                'tail_length': 256, 'tokens': [1] * 256,
                'tokens_sha256': qg.token_hash([1] * 256), 'answer_tokens': None,
                'metadata': {}}
        distribution = [0.6, 0.3, 0.1]
        calls = {'count': 0}

        def responses(body):
            calls['count'] += 1
            step_index = (calls['count'] - 1) % args.teacher_steps
            if step_index == 0:
                return response_from(distribution, cache_n=256, prompt_n=256)
            return response_from(distribution, cache_n=len(body['prompt']) - 1, prompt_n=1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args.output = root
            off_dir = qg.arm_directory(args, 'portuguese-512', 'off'); off_dir.mkdir(parents=True)
            on_dir = qg.arm_directory(args, 'portuguese-512', 'on'); on_dir.mkdir(parents=True)
            session = FakeSession(3, responses)
            progress = {'requests': [], 'model': {'sha256': 'm'},
                        'binary': {'sha256': 'b'}}
            with mock.patch.object(qg, 'restore_snapshot', return_value={}):
                qg.run_arm(session, None, None, {'metadata': {'x': 1}}, base, case,
                           'off', None, 3, args, off_dir, progress)
            teacher = qg.load_teacher(
                SimpleNamespace(output=root, p=256, teacher_steps=2), 'portuguese-512')
            self.assertEqual(teacher['teacher_ids'], [0, 0])
            on_session = FakeSession(3, responses)
            on_progress = {'requests': [], 'model': {'sha256': 'm'},
                           'binary': {'sha256': 'b'}}
            with mock.patch.object(qg, 'restore_snapshot', return_value={}):
                qg.run_arm(on_session, None, None, {'metadata': {'x': 1}}, base, case,
                           'on', teacher, 3, args, on_dir, on_progress)
            off_account = qg.read_json(off_dir / 'accounting.json')
            on_account = qg.read_json(on_dir / 'accounting.json')
            self.assertEqual(off_account['teacher_ids'], on_account['teacher_ids'])
            self.assertEqual(off_account['prompt_hashes'], on_account['prompt_hashes'])
            self.assertEqual(off_account['accounting'], on_account['accounting'])
            self.assertEqual(off_account['vocab'], 3)

            result = qg.compute_case_result(args, case, off_dir, on_dir, 3)
            self.assertEqual(result['answer'], None)
            self.assertEqual(result['summary']['steps'], 2)
            self.assertAlmostEqual(result['summary']['kl_mean'], 0.0)

    def test_cache_accounting_is_bounded_and_exact(self):
        self.assertEqual(qg.validate_accounting({'cache_n': 256, 'prompt_n': 2}, 258, 256, 0),
                         {'cache_n': 256, 'prompt_n': 2})
        self.assertEqual(qg.validate_accounting({'cache_n': 512, 'prompt_n': 1}, 513, 256, 1,
                                                512),
                         {'cache_n': 512, 'prompt_n': 1})
        for timings, prompt, p, step, previous in [
                ({'cache_n': 0, 'prompt_n': 258}, 258, 256, 0, None),
                ({'cache_n': 256, 'prompt_n': 1}, 258, 256, 0, None),
                ({'cache_n': 300, 'prompt_n': 0}, 300, 256, 0, None),
                ({'cache_n': 258, 'prompt_n': 129}, 387, 256, 1, 258),
                ({'cache_n': 257, 'prompt_n': 2}, 259, 256, 1, 258),
                ({'cache_n': 258, 'prompt_n': 1}, 259, 256, 1, 257),
                ({'cache_n': True, 'prompt_n': 1}, 257, 256, 1, 256)]:
            with self.subTest(timings=timings):
                with self.assertRaises(qg.GateError):
                    qg.validate_accounting(timings, prompt, p, step, previous)


class BindingTests(unittest.TestCase):
    def args(self, context='131072'):
        return qg.validate(qg.parser().parse_args([
            '--model', 'm', '--output', 'out', '--p', '256', '--context', context,
            '--lengths', '512', '--cases', 'portuguese', '--teacher-steps', '2']))

    def source(self, head='h', diff='d'):
        return {'head': head, 'tracked_diff_sha256': diff, 'branch': 'branch'}

    def snapshot(self, prefix_sha='prefix', context=131072, p=256):
        return {'meta_sha256': 'meta', 'metadata': {
            'actual_prefix_tokens': p, 'actual_prefix_sha256': prefix_sha,
            'contract': {'requested_prefix_sha256': prefix_sha, 'model_sha256': 'model',
                         'P': p, 'context': context, 'batch': 512},
            'files': {'snap.bin': {'size': 1, 'sha256': 'file'}}}}

    def binding(self, args, source=None, snapshot=None, gates=None, libraries=None):
        plan, specs = qg.case_plan(args)
        return qg.compute_binding(
            args, plan, source or self.source(),
            {'path': 'server', 'size': 1, 'sha256': 'binary'},
            {'path': 'model', 'size': 1, 'sha256': 'model'},
            {'path': 'snap.bin'}, snapshot or self.snapshot(),
            {'sample_ms': 10, 'slack_mb': 64, 'source_sha256': 'guard'},
            gates or args.gates, libraries)

    def test_each_frozen_dimension_is_part_of_the_resume_binding(self):
        args = self.args()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'out'
            first = self.binding(args)
            original = args.output
            args.output = output
            self.assertEqual(qg.prepare_output(args, first)['mode'], 'new')
            for changed in (self.binding(args, source=self.source('other')),
                            self.binding(args, snapshot=self.snapshot('other')),
                            self.binding(args, gates=dict(args.gates, max_kl_mean=0.5))):
                with self.assertRaises(qg.RefuseError):
                    qg.prepare_output(args, changed)
            args.output = original

    def test_new_prefix_run_can_resume_after_the_snapshot_exists(self):
        args = self.args()
        create = {'metadata': None, 'meta_sha256': None, 'files': None,
                  'path': Path('snap.bin'), 'intent': 'create'}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'out'
            args.output = output
            first = self.binding(args, snapshot=create)
            qg.prepare_output(args, first)
            restored = self.binding(args, snapshot=self.snapshot())
            self.assertEqual(qg.prepare_output(args, restored)['mode'], 'resume')

    def test_seed_is_frozen_at_1234(self):
        with self.assertRaises(ValueError):
            qg.validate(qg.parser().parse_args([
                '--model', 'm', '--output', 'out', '--p', '256',
                '--context', '131072', '--lengths', '512', '--seed', '42']))

    def test_context_is_capped_at_160000(self):
        self.assertEqual(self.args('160000').context, 160000)
        with self.assertRaises(ValueError):
            self.args('160001')

    def test_no_tier_or_router_production_file_can_be_touched(self):
        source = SCRIPT.read_text()
        self.assertNotIn('router-production.ini', source)
        self.assertNotIn('router-production', source)

    def test_source_prefix_and_model_mismatch_are_refused_by_the_helper_contract(self):
        # The helper owns the real contract; this test pins the gate's own
        # preflight refusal for a changed model/prefix identity.
        args = self.args()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'out'
            args.output = output
            first = self.binding(args)
            qg.prepare_output(args, first)
            changed = self.binding(
                args, snapshot=self.snapshot('changed-prefix'))
            with self.assertRaises(qg.RefuseError):
                qg.prepare_output(args, changed)

    def test_missing_snapshot_requires_explicit_allow_new_prefix(self):
        class Helper:
            @staticmethod
            def load_prefix_snapshot(namespace, tokens, model):
                return None
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args()
            args.prefix_snapshot = Path(temporary) / 'missing.bin'
            args.allow_new_prefix = False
            with self.assertRaises(qg.RefuseError):
                qg.load_snapshot(Helper(), SimpleNamespace(), [0] * args.p, {}, args)
            args.allow_new_prefix = True
            snapshot = qg.load_snapshot(Helper(), SimpleNamespace(), [0] * args.p, {}, args)
            self.assertEqual(snapshot['intent'], 'create')
            self.assertIsNone(snapshot['metadata'])

    def test_snapshot_contract_mismatch_is_refused(self):
        class Helper:
            @staticmethod
            def load_prefix_snapshot(namespace, tokens, model):
                raise ValueError('prefix snapshot semantic/model/token contract mismatch')
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args()
            args.prefix_snapshot = Path(temporary) / 'snap.bin'
            with self.assertRaises(qg.RefuseError):
                qg.load_snapshot(Helper(), SimpleNamespace(), [0] * args.p, {}, args)


class PersistenceTests(unittest.TestCase):
    def test_distribution_round_trip_and_step_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'd.bin.gz'
            values = [[0.5, 0.25, 0.25], [0.1, 0.2, 0.7]]
            qg.save_distributions(path, values, 3)
            loaded = qg.load_distributions(path, 2, 3)
            for expected, actual in zip(values, loaded):
                for left, right in zip(expected, actual):
                    self.assertAlmostEqual(left, right, places=6)
            self.assertEqual(qg.step_hashes(values)[0],
                             qg.step_digest(values[0]))
            with self.assertRaises((ValueError, qg.GateError)):
                qg.load_distributions(path, 3, 3)

    def test_complete_marker_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'a.json').write_text('{}')
            qg.write_complete(directory, 'PASS', ['a.json'])
            self.assertEqual(qg.complete_marker(directory)['status'], 'PASS')
            (directory / 'a.json').write_text('{"changed":true}')
            with self.assertRaises(qg.RefuseError):
                qg.complete_marker(directory)

    def test_failed_case_is_persisted_and_not_retried_without_the_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = gate_args()
            args.output = Path(temporary)
            directory = args.output / 'cases' / 'portuguese-5120'
            directory.mkdir(parents=True)
            (directory / 'tokens.json').write_text('{}')
            (directory / 'case-result.json').write_text(json.dumps({
                'case': 'portuguese-5120', 'status': 'FAIL', 'steps': [], 'summary': {}}))
            qg.write_complete(directory, 'FAIL', ['tokens.json', 'case-result.json'])
            result = qg.load_case_result(args, 'portuguese-5120')
            self.assertEqual(result['status'], 'FAIL')
            with mock.patch.object(qg, 'run_case_arm', side_effect=AssertionError('must not run')):
                returned = qg.run_case(args, None, None, None, None, None,
                                       {'name': 'portuguese-5120'}, None, 7, {})
            self.assertEqual(returned['status'], 'FAIL')

    def test_stored_pass_without_complete_arms_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = gate_args()
            args.output = Path(temporary)
            directory = args.output / 'cases' / 'portuguese-5120'
            directory.mkdir(parents=True)
            (directory / 'tokens.json').write_text('{}')
            (directory / 'case-result.json').write_text(json.dumps({
                'case': 'portuguese-5120', 'status': 'PASS',
                'steps': [], 'summary': {}}))
            qg.write_complete(directory, 'PASS', ['tokens.json', 'case-result.json'])
            with self.assertRaises(qg.RefuseError):
                qg.run_case(args, None, None, None, None, None,
                            {'name': 'portuguese-5120'}, None, 4096, {})

    def test_preview_never_starts_a_child_and_writes_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / 'model.gguf'; model.write_bytes(b'model')
            tokens = root / 'tokens.json'; tokens.write_text(json.dumps(list(range(512))))
            output = root / 'out'
            argv = ['--model', str(model), '--prompt-tokens', str(tokens),
                    '--prefix-snapshot', str(root / 'missing.bin'),
                    '--output', str(output), '--p', '256', '--context', '131072',
                    '--lengths', '512', '--cases', 'portuguese', '--teacher-steps', '2']
            with mock.patch.object(qg.subprocess, 'Popen',
                                   side_effect=AssertionError('child started')), \
                    mock.patch.object(qg, 'DeployClient',
                                      side_effect=AssertionError('deploy touched')), \
                    mock.patch.object(qg, 'foreign_process_rows', return_value=[]), \
                    contextlib.redirect_stdout(io.StringIO()) as captured:
                self.assertEqual(qg.main(argv), 0)
            self.assertFalse(output.exists())
            preview = json.loads(captured.getvalue())
            self.assertEqual(preview['mode'], 'preview')
            self.assertFalse(preview['starts_child'])
            self.assertFalse(preview['writes_output'])
            self.assertEqual(preview['status'], 'PREVIEW_ONLY')
            self.assertEqual(preview['pinned_environment_preview']['GGML_RKVA_CUDA_FP8'],
                             'mapped')
            self.assertEqual(preview['pinned_environment_preview']
                             ['GGML_CUDA_GRAPH_GUARD_SAMPLE_MS'], '10')
            self.assertEqual(preview['pinned_environment_preview']
                             ['GGML_CUDA_GRAPH_GUARD_SLACK_MB'], '64')
            self.assertEqual(preview['pinned_environment_preview']
                             ['GGML_RKVA_LOCAL_INPUT_REUSE'], '0')

    def test_execute_without_xbox_host_is_a_configuration_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / 'model.gguf'; model.write_bytes(b'model')
            tokens = root / 'tokens.json'; tokens.write_text(json.dumps(list(range(512))))
            argv = ['--model', str(model), '--prompt-tokens', str(tokens),
                    '--output', str(root / 'out'), '--p', '256', '--context', '131072',
                    '--lengths', '512', '--cases', 'portuguese', '--teacher-steps', '2',
                    '--execute']
            with mock.patch.object(qg.subprocess, 'Popen',
                                   side_effect=AssertionError('child started')), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(qg.main(argv), 2)


class TokenPreparationTests(unittest.TestCase):
    class Session:
        def __init__(self, vocab=4096):
            self.vocab = vocab
        def n_vocab(self):
            return self.vocab
        def request(self, endpoint, body=None, timeout=None):
            self.assert_endpoint = endpoint
            words = body['content'].split()
            return {'tokens': [int.from_bytes(
                __import__('hashlib').sha256(word.encode()).digest()[:4], 'little') % self.vocab
                for word in words]}

    def test_case_tokens_are_persisted_verified_and_reused(self):
        args = gate_args()
        plan, specs = qg.case_plan(args)
        with tempfile.TemporaryDirectory() as temporary:
            args.output = Path(temporary)
            session = self.Session()
            first = qg.ensure_case_tokens(session, specs, args)
            self.assertEqual(set(first), {item['name'] for item in specs})
            for name, document in first.items():
                self.assertEqual(document['length'], 5120)
                self.assertEqual(document['tail_length'], 4096)
                self.assertEqual(document['vocab'], 4096)
                path = args.output / 'cases' / name / 'tokens.json'
                self.assertTrue(path.is_file())
                self.assertEqual(qg.read_json(path)['tokens_sha256'], document['tokens_sha256'])
            second = qg.ensure_case_tokens(session, specs, args)
            for name in first:
                self.assertEqual(first[name]['tokens'], second[name]['tokens'])
            vocab = qg.prepare_all_tokens(args, None, None, None, specs, None, {})
            self.assertEqual(vocab, 4096)

    def test_changed_persisted_token_hash_is_refused(self):
        args = gate_args()
        plan, specs = qg.case_plan(args)
        with tempfile.TemporaryDirectory() as temporary:
            args.output = Path(temporary)
            qg.ensure_case_tokens(self.Session(), specs, args)
            path = args.output / 'cases' / specs[0]['name'] / 'tokens.json'
            document = qg.read_json(path)
            document['tokens'][0] ^= 1
            qg.write_json(path, document)
            with self.assertRaises(qg.RefuseError):
                qg.ensure_case_tokens(self.Session(), specs, args)

class ArmFinalizationTests(unittest.TestCase):
    def test_profile_is_flushed_before_the_complete_marker(self):
        class FakeSession:
            def __init__(self, *a, **k):
                self.manifest = {'profile': None}
            def __enter__(self):
                return self
            def __exit__(self, *a):
                self.manifest['profile'] = {'flushed': True}
                return False
            def profile_lines(self):
                return []

        class FakeFlags:
            @contextlib.contextmanager
            def arm(self, on):
                yield

        args = qg.validate(qg.parser().parse_args([
            '--model', 'm', '--output', 'out', '--p', '256',
            '--context', '131072', '--lengths', '512', '--cases', 'portuguese',
            '--teacher-steps', '1']))
        case = {'name': 'c', 'category': 'portuguese', 'length': 512,
                'tail_length': 256, 'tokens': [1] * 256,
                'tokens_sha256': qg.token_hash([1] * 256), 'answer_tokens': None,
                'metadata': {}}
        observed = {}
        with tempfile.TemporaryDirectory() as temporary:
            args.output = Path(temporary)
            arm = args.output / 'cases' / 'c' / 'off'
            arm.mkdir(parents=True)
            with mock.patch.object(qg, 'load_or_prepare_arm',
                                   return_value={'mode': 'new', 'directory': arm}), \
                    mock.patch.object(qg, 'assert_quiet', return_value=None), \
                    mock.patch.object(qg, 'ServerSession', FakeSession), \
                    mock.patch.object(qg, 'run_arm', return_value=None), \
                    mock.patch.object(qg, 'restore_snapshot', return_value={}), \
                    mock.patch.object(qg, 'finalize_arm_dir',
                                      side_effect=lambda directory, session, files:
                                      observed.update(profile=session.manifest.get('profile'))), \
                    mock.patch.object(qg, 'load_arm', return_value=({}, [])):
                qg.run_case_arm(args, None, None, None, {'metadata': {'x': 1}},
                                [0] * 256, case, 'off', FakeFlags(), None, 3,
                                {'model': {'sha256': 'm'}, 'binary': {'sha256': 'b'}})
        self.assertEqual(observed['profile'], {'flushed': True})

class CaseExecutionIntegrationTests(unittest.TestCase):
    def test_case_runner_uses_teacher_then_writes_a_passed_case(self):
        case_name = 'portuguese-512'
        args = qg.validate(qg.parser().parse_args([
            '--model', 'm', '--output', 'out', '--p', '256',
            '--context', '131072', '--lengths', '512',
            '--cases', 'portuguese', '--teacher-steps', '2']))

        def engine(i8_calls):
            return {
                'kvarn4_multi_query': {
                    'mq2': {'calls': 4},
                    'mq2_i8': {'calls': i8_calls, 'available': True, 'exact': False},
                },
                'kvarn4_decode': {'calls': 4},
                'kvarn4_decode_rowlane': {'calls': 4, 'enabled': True},
                'bulk': {'fp8_gpu_calls': 3, 'fp8_cpu_fallback_calls': 0,
                         'bulk_segments': 4, 'bulk_queries': 32},
                'buckets': {
                    'attention/kvarn4-prefix/optimized-kvarn4_mq2_fine/nq2-16/x':
                        {'samples': 3},
                    'attention/kvarn4-prefix/optimized-kvarn4_mq2_i8/nq2-16/x':
                        {'samples': 3},
                },
            }

        class FakeSession:
            def __init__(self, helper, namespace, namespace_args, output, guard, name, **kwargs):
                self.output = Path(output)
                self.name = name
                self.manifest = {'profile': engine(1 if name.endswith('-on') else 0),
                                 'child_exit_code': 0}
                self.lines = [
                    'RKVA wavefront graph: tile=256 tiles=2 parent_tokens=512 '
                    'final_ffn_rows=256 max_outstanding=2',
                    '{"tcp_prefetch_admitted":2,"tcp_prefetch_sent":2,'
                    '"tcp_prefetch_completed":2,"tcp_prefetch_ahead_sent":1,'
                    '"tcp_prefetch_max_frames":2}',
                    'remote_attn gpu_profile ' + json.dumps(
                        {'engine': engine(1 if name.endswith('-on') else 0)}),
                ]
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                qg.write_json(self.output / 'profile-lines.json', self.lines)
                return False
            def profile_lines(self):
                return self.lines
            def n_vocab(self):
                return 3
            def request(self, endpoint, body=None, timeout=None):
                prompt = len(body['prompt'])
                if prompt == 512:
                    cache, processed = 256, 256
                else:
                    cache, processed = prompt - 1, 1
                probabilities = [0.6, 0.3, 0.1]
                return response_from(probabilities, cache_n=cache, prompt_n=processed)

        class FakeFlags:
            @contextlib.contextmanager
            def arm(self, on):
                yield

        def tokenize(text):
            return fake_tokenizer(3)(text)

        spec = qg.build_case_spec(case_name, 'portuguese', 512, 7)
        with tempfile.TemporaryDirectory() as temporary:
            args.output = Path(temporary)
            case = qg.fit_case_tokens(spec, 256, tokenize, 3)
            case_dir = args.output / 'cases' / case_name
            case_dir.mkdir(parents=True)
            qg.write_json(case_dir / 'tokens.json', case)
            with mock.patch.object(qg, 'assert_quiet', return_value=None), \
                    mock.patch.object(qg, 'ServerSession', FakeSession), \
                    mock.patch.object(qg, 'restore_snapshot', return_value={}), \
                    mock.patch.object(qg, 'finalize_arm_dir', wraps=qg.finalize_arm_dir):
                result = qg.run_case(
                    args, None, None, None, {'metadata': {'x': 1}}, [0] * 256,
                    spec, FakeFlags(), 3,
                    {'model': {'sha256': 'm'}, 'binary': {'sha256': 'b'}})
            self.assertEqual(result['status'], 'PASS', result.get('status_detail'))
            self.assertTrue(result['activation']['pass'])
            self.assertEqual(result['summary']['nll_delta'], 0.0)
            self.assertEqual(qg.load_case_result(args, case_name)['status'], 'PASS')
            self.assertTrue((case_dir / 'complete.json').is_file())

class PreflightSmokeTests(unittest.TestCase):
    def args(self, output):
        return qg.validate(qg.parser().parse_args([
            '--model', 'm', '--output', str(output), '--p', '1024',
            '--context', '131072', '--lengths', '5120',
            '--cases', 'portuguese', '--teacher-steps', '2']))

    def test_separate_preflight_runs_before_matrix_and_persists_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            result = {
                'case': 'preflight-portuguese-5120', 'length': 5120,
                'status': 'PASS', 'activation': {'pass': True},
                'summary': {'steps': 2}, 'off': {}, 'on': {},
            }
            with mock.patch.object(qg, 'prepare_all_tokens', return_value=7), \
                    mock.patch.object(qg, 'run_case', return_value=result):
                document = qg.run_preflight_smoke(
                    args, None, None, None, {}, [0] * 1024, None,
                    {'model': {'sha256': 'm'}, 'binary': {'sha256': 'b'}})
            self.assertEqual(document['label'], 'PREFLIGHT_SMOKE_ONLY')
            self.assertEqual(document['teacher_steps'], 2)
            self.assertTrue((args.output / 'preflight-smoke.json').is_file())

    def test_failed_preflight_blocks_the_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            result = {'case': 'x', 'length': 5120, 'status': 'FAIL',
                      'activation': {'pass': False}, 'summary': {}}
            with mock.patch.object(qg, 'prepare_all_tokens', return_value=7), \
                    mock.patch.object(qg, 'run_case', return_value=result):
                with self.assertRaises(qg.GateError):
                    qg.run_preflight_smoke(args, None, None, None, {}, [0] * 1024,
                                           None, {'model': {'sha256': 'm'},
                                                  'binary': {'sha256': 'b'}})

class DeterminismControlTests(unittest.TestCase):
    def test_control_uses_a_fresh_off_process_and_real_arm_signatures(self):
        args = qg.validate(qg.parser().parse_args([
            '--model', 'm', '--output', 'out', '--p', '256',
            '--context', '131072', '--lengths', '512',
            '--cases', 'portuguese', '--teacher-steps', '2']))
        case = {'name': 'c', 'category': 'portuguese', 'length': 512,
                'tail_length': 256, 'tokens': [1] * 256,
                'tokens_sha256': qg.token_hash([1] * 256), 'answer_tokens': None,
                'metadata': {}}
        base = [0] * 256
        probabilities = [0.6, 0.3, 0.1]

        class FakeSession:
            def __init__(self, helper, namespace, namespace_args, output, guard, name, **kwargs):
                self.output = Path(output)
                self.name = name
                self.manifest = {'profile': {'bulk': {'fp8_gpu_calls': 1}},
                                 'child_exit_code': 0}
                self.lines = ['remote_attn gpu_profile ' + json.dumps(
                    {'engine': {'bulk': {'fp8_gpu_calls': 1}}})]
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                qg.write_json(self.output / 'profile-lines.json', self.lines)
                return False
            def profile_lines(self):
                return self.lines
            def request(self, endpoint, body=None, timeout=None):
                prompt = len(body['prompt'])
                return response_from(probabilities, cache_n=256 if prompt == 512 else 512,
                                     prompt_n=256 if prompt == 512 else 1)

        class FakeFlags:
            @contextlib.contextmanager
            def arm(self, on):
                yield

        with tempfile.TemporaryDirectory() as temporary:
            args.output = Path(temporary)
            main_dir = args.output / 'cases' / 'c' / 'off'
            main_dir.mkdir(parents=True)
            progress = {'requests': [], 'model': {'sha256': 'm'}, 'binary': {'sha256': 'b'}}
            session = FakeSession(None, None, args, main_dir, None, 'c-off')
            with mock.patch.object(qg, 'restore_snapshot', return_value={}):
                qg.run_arm(session, None, None, {'metadata': {'x': 1}}, base, case,
                           'off', None, 3, args, main_dir, progress)
            with mock.patch.object(qg, 'ServerSession', FakeSession), \
                    mock.patch.object(qg, 'assert_quiet', return_value=None), \
                    mock.patch.object(qg, 'restore_snapshot', return_value={}):
                control = qg.run_determinism_control(
                    args, None, None, None, {'metadata': {'x': 1}}, base, case,
                    FakeFlags(), 3,
                    {'model': {'sha256': 'm'}, 'binary': {'sha256': 'b'}}, main_dir)
            self.assertTrue(control['pass'], control)
            self.assertEqual(control['top1_agreement'], 1.0)
            self.assertTrue((args.output / 'control' / 'c-off' / 'complete.json').is_file())

class MissingProfileTests(unittest.TestCase):
    def test_missing_profile_never_writes_a_pass_marker(self):
        class FakeSession:
            def __init__(self, *a, **k):
                self.manifest = {'profile': None, 'child_exit_code': 0}
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def profile_lines(self):
                return []

        class FakeFlags:
            @contextlib.contextmanager
            def arm(self, on):
                yield

        args = qg.validate(qg.parser().parse_args([
            '--model', 'm', '--output', 'out', '--p', '256',
            '--context', '131072', '--lengths', '512', '--cases', 'portuguese',
            '--teacher-steps', '1']))
        case = {'name': 'c', 'category': 'portuguese', 'length': 512,
                'tail_length': 256, 'tokens': [1] * 256,
                'tokens_sha256': qg.token_hash([1] * 256), 'answer_tokens': None,
                'metadata': {}}
        with tempfile.TemporaryDirectory() as temporary:
            args.output = Path(temporary)
            arm = args.output / 'cases' / 'c' / 'off'
            arm.mkdir(parents=True)
            with mock.patch.object(qg, 'load_or_prepare_arm',
                                   return_value={'mode': 'new', 'directory': arm}), \
                    mock.patch.object(qg, 'assert_quiet', return_value=None), \
                    mock.patch.object(qg, 'ServerSession', FakeSession), \
                    mock.patch.object(qg, 'run_arm', return_value=None), \
                    mock.patch.object(qg, 'restore_snapshot', return_value={}):
                with self.assertRaises(qg.CampaignAbort):
                    qg.run_case_arm(args, None, None, None, {'metadata': {'x': 1}},
                                    [0] * 256, case, 'off', FakeFlags(), None, 3,
                                    {'model': {'sha256': 'm'}, 'binary': {'sha256': 'b'}})
            self.assertFalse((arm / 'complete.json').exists())
            self.assertTrue((arm / 'failure.json').exists())

class ExecuteOrchestrationTests(unittest.TestCase):
    def test_execute_writes_final_artifacts_and_restores_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / 'model.gguf'; model.write_bytes(b'model')
            tokens = root / 'tokens.json'; tokens.write_text(json.dumps(list(range(2048))))
            prefix = root / 'prefix.bin'; prefix.write_bytes(b'prefix')
            args = qg.validate(qg.parser().parse_args([
                '--model', str(model), '--prompt-tokens', str(tokens),
                '--prefix-snapshot', str(prefix), '--output', str(root / 'out'),
                '--p', '1024', '--context', '131072', '--smoke',
                '--xbox-host', '192.0.2.1:50054', '--execute']))
            source = {'head': 'h', 'tracked_diff_sha256': 'd', 'branch': 'b'}

            class Helper:
                PINNED_ENV = {}
                def git_identity(self):
                    return source
                def memory_policy(self):
                    return {'effective_mib': 128}
                def effective_tokens(self, path, length, repeat=False):
                    return [0] * length, {}
                def fingerprint(self, path):
                    return {'path': str(path), 'size': 1, 'sha256': 'b'}
                def server_command(self, namespace, port):
                    return ['llama-server', '--port', str(port)]
                def pinned_environment(self, namespace):
                    return {}
                def environment(self, namespace, output):
                    return {}, {}

            class Flags:
                def __init__(self, *a, **k):
                    self.original = {'rkva-mq-i8.flag': False,
                                     'rkva-decode-rowlane-off.flag': False}
                    self.restored = 0
                def capture_original(self):
                    return self.original
                def read(self):
                    return set()
                def restore(self):
                    self.restored += 1
                @contextlib.contextmanager
                def arm(self, on):
                    yield

            flags = Flags()

            def prepare(namespace, helper, helper_ns, guard, specs, flag_manager, hooks):
                for spec in specs:
                    directory = namespace.output / 'cases' / spec['name']
                    directory.mkdir(parents=True, exist_ok=True)
                    qg.write_json(directory / 'tokens.json', {
                        'name': spec['name'], 'category': spec['category'],
                        'length': spec['length'], 'tail_length': spec['length'] - namespace.p,
                        'tokens': [1] * (spec['length'] - namespace.p),
                        'tokens_sha256': qg.token_hash([1] * (spec['length'] - namespace.p)),
                        'answer_tokens': None, 'metadata': {}})
                return 5

            result = {
                'case': 'portuguese-5120', 'category': 'portuguese', 'length': 5120,
                'status': 'PASS', 'steps': [step()],
                'summary': qg.summarize_steps([step()]), 'answer': None,
                'activation': {'pass': True, 'reason': 'proved'},
            }
            with mock.patch.object(qg, 'load_helper', return_value=Helper()), \
                    mock.patch.object(qg, 'capture_guard_policy',
                                      return_value={'sample_ms': 10, 'slack_mb': 64,
                                                    'source_sha256': 'g'}), \
                    mock.patch.object(qg, 'source_identity', return_value=source), \
                    mock.patch.object(qg, 'load_snapshot',
                                      return_value={'metadata': None, 'meta_sha256': None,
                                                    'files': None, 'path': prefix,
                                                    'intent': 'create'}), \
                    mock.patch.object(qg, 'preview', return_value={'mode': 'preview'}), \
                    mock.patch.object(qg, 'DeployClient', return_value=SimpleNamespace(
                        pfn=lambda: 'GianlucaMazza.xllama_1.0.0.0_x64__abc')), \
                    mock.patch.object(qg, 'LocalStateFlags', return_value=flags), \
                    mock.patch.object(qg, 'prepare_all_tokens', side_effect=prepare), \
                    mock.patch.object(qg, 'build_library_binding',
                                      return_value={'files': [], 'stats': {},
                                                    'sha256': qg.sha256_text('[]')}), \
                    mock.patch.object(qg, 'active_library_paths', return_value=[]), \
                    mock.patch.object(qg, 'run_case', return_value=result), \
                    mock.patch.object(qg, 'foreign_process_rows', return_value=[]), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = qg.execute(args)
            self.assertIn(code, (0, 4))
            self.assertTrue((root / 'out' / 'final.json').is_file())
            self.assertTrue((root / 'out' / 'preflight.json').is_file())
            self.assertTrue((root / 'out' / 'case-tokens.json').is_file())
            self.assertGreaterEqual(flags.restored, 1)
            final = json.loads((root / 'out' / 'final.json').read_text())
            self.assertFalse(final['int8_default_authorized'])
            self.assertIn('libraries', final)

class DisabledOverflowOptionTests(unittest.TestCase):
    def test_disabled_option_fails_validation_without_any_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / 'model.gguf'; model.write_bytes(b'model')
            tokens = root / 'tokens.json'; tokens.write_text(json.dumps(list(range(512))))
            output = root / 'out'
            base = ['--model', str(model), '--prompt-tokens', str(tokens),
                    '--prefix-snapshot', str(root / 'missing.bin'),
                    '--output', str(output), '--p', '256', '--context', '131072',
                    '--lengths', '512', '--cases', 'portuguese', '--teacher-steps', '2',
                    '--verify-overflow-save-refusal']
            with self.assertRaises(ValueError):
                qg.validate(qg.parser().parse_args(base))
            with mock.patch.object(qg.subprocess, 'Popen',
                                   side_effect=AssertionError('child started')), \
                    mock.patch.object(qg, 'DeployClient',
                                      side_effect=AssertionError('deploy touched')), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(qg.main(base), 2)
                self.assertFalse(output.exists())
                self.assertEqual(qg.main(base + ['--xbox-host', '192.0.2.1:50054',
                                                 '--execute']), 2)
                self.assertFalse(output.exists())

    def test_no_placeholder_overflow_feature_api_remains(self):
        self.assertFalse(hasattr(qg, 'verify_overflow_save_refusal'))
        source = SCRIPT.read_text()
        self.assertNotIn('UNSUPPORTED_WITH_SAFE_REFUSAL', source)
        self.assertNotIn('route_target_no_mtp', source)

class LibraryBindingTests(unittest.TestCase):
    def library_files(self, directory):
        root = Path(directory)
        first = root / 'libllama.so.0.26.0'
        second = root / 'libggml-base.so.0.26.0'
        first.write_bytes(b'llama-v1')
        second.write_bytes(b'ggml-v1')
        return [first, second]

    def test_stub_sha_same_but_library_content_changed_changes_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = self.library_files(temporary)
            binding_a = qg.build_library_binding('stub', [first, second])
            first.write_bytes(b'llama-v2')
            binding_b = qg.build_library_binding('stub', [first, second])
            self.assertNotEqual(binding_a['sha256'], binding_b['sha256'])
            args = gate_args()
            with tempfile.TemporaryDirectory() as output:
                args.output = Path(output) / 'out'
                first_binding = qg.compute_binding(
                    args, qg.case_plan(args)[0], {'head': 'h', 'tracked_diff_sha256': 'd'},
                    {'path': 'stub', 'size': 1, 'sha256': 'same-stub'},
                    {'path': 'model', 'size': 1, 'sha256': 'model'},
                    {'path': 'snap.bin'},
                    {'metadata': None, 'meta_sha256': None, 'files': None,
                     'path': Path('snap.bin'), 'intent': 'create'},
                    {'sample_ms': 10, 'slack_mb': 64, 'source_sha256': 'guard'},
                    args.gates, binding_a)
                qg.prepare_output(args, first_binding)
                changed = qg.compute_binding(
                    args, qg.case_plan(args)[0], {'head': 'h', 'tracked_diff_sha256': 'd'},
                    {'path': 'stub', 'size': 1, 'sha256': 'same-stub'},
                    {'path': 'model', 'size': 1, 'sha256': 'model'},
                    {'path': 'snap.bin'},
                    {'metadata': None, 'meta_sha256': None, 'files': None,
                     'path': Path('snap.bin'), 'intent': 'create'},
                    {'sample_ms': 10, 'slack_mb': 64, 'source_sha256': 'guard'},
                    args.gates, binding_b)
                with self.assertRaises(qg.RefuseError):
                    qg.prepare_output(args, changed)

    def test_guard_accepts_bound_paths_and_rejects_unknown_or_mutated(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.library_files(temporary)
            binding = qg.build_library_binding('stub', paths)
            guard = qg.LibraryGuard('stub', binding, path_provider=lambda _: paths)
            guard.verify_local('ok')
            guard.verify_loaded([str(paths[0]), str(paths[1])], 'ok')
            with self.assertRaises(qg.GateError):
                guard.verify_loaded([str(paths[0]) + '.unknown'], 'bad')
            paths[0].write_bytes(b'llama-v2')
            with self.assertRaises(qg.GateError):
                guard.verify_local('mutated')

    def test_library_mutation_during_control_avoids_a_complete_marker(self):
        class FakeSession:
            def __init__(self, *a, **k):
                self.manifest = {'profile': {'bulk': {'fp8_gpu_calls': 1}},
                                 'child_exit_code': 0,
                                 'loaded_libraries': [str(paths[0]), str(paths[1])],
                                 'loaded_libraries_final': [str(paths[0]), str(paths[1])]}
            def __enter__(self):
                return self
            def __exit__(self, *a):
                paths[1].write_bytes(b'ggml-v2')
                return False
            def profile_lines(self):
                return []
            def request(self, endpoint, body=None, timeout=None):
                prompt = len(body['prompt'])
                return response_from([0.6, 0.3, 0.1],
                                     cache_n=256 if prompt == 512 else 512,
                                     prompt_n=256 if prompt == 512 else 1)

        class FakeFlags:
            @contextlib.contextmanager
            def arm(self, on):
                yield

        args = qg.validate(qg.parser().parse_args([
            '--model', 'm', '--output', 'out', '--p', '256',
            '--context', '131072', '--lengths', '512',
            '--cases', 'portuguese', '--teacher-steps', '2']))
        case = {'name': 'c', 'category': 'portuguese', 'length': 512,
                'tail_length': 256, 'tokens': [1] * 256,
                'tokens_sha256': qg.token_hash([1] * 256), 'answer_tokens': None,
                'metadata': {}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args.output = root / 'out'
            paths = self.library_files(root)
            binding = qg.build_library_binding('stub', paths)
            guard = qg.LibraryGuard('stub', binding, path_provider=lambda _: paths)
            main_dir = args.output / 'cases' / 'c' / 'off'
            main_dir.mkdir(parents=True)
            with qg.DistributionWriter(main_dir / 'distributions.bin.gz', 2, 3) as writer:
                hashes = [writer.append([0.6, 0.3, 0.1]) for _ in range(2)]
            qg.write_json(main_dir / 'accounting.json', {
                'arm': 'off', 'steps': 2, 'p': 256, 'vocab': 3,
                'case_tokens_sha256': qg.token_hash([1] * 256),
                'base_tokens_sha256': qg.token_hash([0] * 256),
                'model_sha256': 'm', 'binary_sha256': 'b',
                'teacher_ids': [0, 0], 'prompt_hashes': ['p0', 'p1'],
                'step_distribution_sha256': hashes,
                'distributions_sha256': qg.sha256_file(
                    main_dir / 'distributions.bin.gz'),
                'accounting': [{'cache_n': 256, 'prompt_n': 256},
                               {'cache_n': 512, 'prompt_n': 1}],
                'tail': {'cache_n': 256, 'prompt_n': 256}, 'restore': {},
            })
            with mock.patch.object(qg, 'ServerSession', FakeSession), \
                    mock.patch.object(qg, 'assert_quiet', return_value=None), \
                    mock.patch.object(qg, 'restore_snapshot', return_value={}):
                with self.assertRaises(qg.GateError):
                    qg.run_determinism_control(
                        args, None, None, None, {'metadata': {'x': 1}}, [0] * 256,
                        case, FakeFlags(), 3,
                        {'model': {'sha256': 'm'}, 'binary': {'sha256': 'b'},
                         'library_guard': guard}, main_dir)
            self.assertFalse((args.output / 'control' / 'c-off' / 'complete.json').exists())

    def test_library_mutation_during_arm_avoids_a_pass_marker(self):
        class MutatingSession:
            def __init__(self, *a, **k):
                self.manifest = {'profile': {'bulk': {'fp8_gpu_calls': 1}},
                                 'child_exit_code': 0,
                                 'loaded_libraries': [str(paths[0]), str(paths[1])],
                                 'loaded_libraries_final': [str(paths[0]), str(paths[1])]}
            def __enter__(self):
                return self
            def __exit__(self, *a):
                paths[0].write_bytes(b'llama-v2')
                return False
            def profile_lines(self):
                return []

        class FakeFlags:
            @contextlib.contextmanager
            def arm(self, on):
                yield

        args = qg.validate(qg.parser().parse_args([
            '--model', 'm', '--output', 'out', '--p', '256',
            '--context', '131072', '--lengths', '512', '--cases', 'portuguese',
            '--teacher-steps', '1']))
        case = {'name': 'c', 'category': 'portuguese', 'length': 512,
                'tail_length': 256, 'tokens': [1] * 256,
                'tokens_sha256': qg.token_hash([1] * 256), 'answer_tokens': None,
                'metadata': {}}
        with tempfile.TemporaryDirectory() as temporary:
            args.output = Path(temporary) / 'out'
            paths = self.library_files(temporary)
            binding = qg.build_library_binding('stub', paths)
            guard = qg.LibraryGuard('stub', binding, path_provider=lambda _: paths)
            arm = args.output / 'cases' / 'c' / 'off'
            arm.mkdir(parents=True)
            with mock.patch.object(qg, 'load_or_prepare_arm',
                                   return_value={'mode': 'new', 'directory': arm}), \
                    mock.patch.object(qg, 'assert_quiet', return_value=None), \
                    mock.patch.object(qg, 'ServerSession', MutatingSession), \
                    mock.patch.object(qg, 'run_arm', return_value=None):
                with self.assertRaises(qg.CampaignAbort):
                    qg.run_case_arm(args, None, None, None, {'metadata': {'x': 1}},
                                    [0] * 256, case, 'off', FakeFlags(), None, 3,
                                    {'model': {'sha256': 'm'},
                                     'binary': {'sha256': 'b'},
                                     'library_guard': guard})
            self.assertFalse((arm / 'complete.json').exists())
            self.assertTrue((arm / 'failure.json').exists())

class DistributionStreamingTests(unittest.TestCase):
    def write_arm(self, path, steps, vocab, value):
        path.mkdir(parents=True, exist_ok=True)
        with qg.DistributionWriter(path / 'distributions.bin.gz', steps, vocab) as writer:
            for _ in range(steps):
                writer.append(array.array('f', [value]) * vocab)

    def accounting(self, steps, vocab, name, path):
        teacher = [0] * steps
        hashes = [f'{name}-prompt-{index}' for index in range(steps)]
        step_hashes = []
        with qg.DistributionReader(path / 'distributions.bin.gz', steps, vocab) as reader:
            for step in reader:
                step_hashes.append(qg.step_digest(step))
        return {
            'arm': 'off', 'steps': steps, 'p': 256, 'vocab': vocab,
            'case_tokens_sha256': qg.token_hash([1] * 256),
            'base_tokens_sha256': qg.token_hash([0] * 256),
            'teacher_ids': teacher, 'prompt_hashes': hashes,
            'step_distribution_sha256': step_hashes,
            'distributions_sha256': qg.sha256_file(path / 'distributions.bin.gz'),
            'accounting': [{'cache_n': 256 if index == 0 else 256 + index,
                            'prompt_n': 256 if index == 0 else 1}
                           for index in range(steps)],
            'tail': {'cache_n': 256, 'prompt_n': 256},
            'restore': {},
        }

    def test_reader_is_bounded_with_128_large_vocab_steps(self):
        steps, vocab = 128, 20000
        with tempfile.TemporaryDirectory() as temporary:
            self.write_arm(Path(temporary), steps, vocab, 1.0 / vocab)
            path = Path(temporary) / 'distributions.bin.gz'
            tracemalloc.start()
            with qg.DistributionReader(path, steps, vocab) as reader:
                count = 0
                for step in reader:
                    count += 1
                    self.assertEqual(len(step), vocab)
                peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            self.assertEqual(count, steps)
            self.assertEqual(reader.reuses, steps)
            self.assertEqual(reader.max_buffers, 1)
            # The full matrix would be steps*vocab*4 ~= 10.2 MiB.
            self.assertLess(peak, 4 << 20)

    def test_compute_streams_pairs_without_retaining_matrices(self):
        steps, vocab = 128, 20000
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            off_dir = root / 'off'; on_dir = root / 'on'
            self.write_arm(off_dir, steps, vocab, 1.0 / vocab)
            self.write_arm(on_dir, steps, vocab, 1.0 / vocab)
            qg.write_json(off_dir / 'accounting.json',
                          self.accounting(steps, vocab, 'off', off_dir))
            on_accounting = self.accounting(steps, vocab, 'on', on_dir)
            on_accounting['arm'] = 'on'
            on_accounting['prompt_hashes'] = [
                f'off-prompt-{index}' for index in range(steps)]
            qg.write_json(on_dir / 'accounting.json', on_accounting)
            args = qg.validate(qg.parser().parse_args([
                '--model', 'm', '--output', str(root), '--p', '256',
                '--context', '131072', '--lengths', '512', '--cases', 'portuguese',
                '--teacher-steps', str(steps)]))
            case = {'name': 'c', 'category': 'portuguese', 'length': 512,
                    'tail_length': 256, 'tokens': [1] * 256,
                    'tokens_sha256': qg.token_hash([1] * 256), 'answer_tokens': None}
            tracemalloc.start()
            result = qg.compute_case_result(args, case, off_dir, on_dir, vocab)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            self.assertEqual(result['summary']['steps'], steps)
            self.assertAlmostEqual(result['summary']['kl_mean'], 0.0)
            # Two matrices would be ~20.5 MiB; the pair loop is one step each.
            self.assertLess(peak, 12 << 20)
            self.assertNotIn('distributions', result)

    def test_reader_rejects_truncated_wrong_and_extra_data(self):
        steps, vocab = 2, 4
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_arm(root, steps, vocab, 0.25)
            path = root / 'distributions.bin.gz'
            with self.assertRaises(qg.GateError):
                qg.DistributionReader(path, 3, vocab)
            with self.assertRaises(qg.GateError):
                qg.DistributionReader(path, steps, vocab + 1)
            raw = path.read_bytes()
            (root / 'truncated.bin.gz').write_bytes(raw[:len(raw) // 2])
            with self.assertRaises(qg.GateError):
                with qg.DistributionReader(root / 'truncated.bin.gz', steps, vocab) as reader:
                    for _ in reader:
                        pass
            (root / 'oversize.bin.gz').write_bytes(
                b'x' * (len(qg.DIST_MAGIC) + qg.DIST_HEADER.size
                        + steps * vocab * 4 + (1 << 20) + 1))
            with self.assertRaises(qg.GateError):
                qg.DistributionReader(root / 'oversize.bin.gz', steps, vocab)
            extra = root / 'extra.bin.gz'
            with gzip.GzipFile(str(extra), 'wb', mtime=0) as handle:
                handle.write(qg.DIST_MAGIC + qg.DIST_HEADER.pack(1, vocab))
                handle.write(array.array('f', [0.25] * vocab).tobytes())
                handle.write(array.array('f', [0.25] * vocab).tobytes())
            with self.assertRaises(qg.GateError):
                with qg.DistributionReader(extra, 1, vocab) as reader:
                    for _ in reader:
                        pass

class GuardPolicyTests(unittest.TestCase):
    SOURCE = (
        'const char * env = getenv("GGML_CUDA_GRAPH_GUARD_SAMPLE_MS");\n'
        'const int64_t ms = env != nullptr ? atoll(env) : 10;\n'
        'const char * env = getenv("GGML_CUDA_GRAPH_GUARD_SLACK_MB");\n'
        'const long long mb = env != nullptr ? atoll(env) : 64;\n')

    def test_current_defaults_are_parsed_and_pinned(self):
        self.assertEqual(qg.parse_guard_policy(self.SOURCE),
                         {'sample_ms_default': 10, 'slack_mb_default': 64})
        with self.assertRaises(qg.GateError):
            qg.parse_guard_policy('unrecognized')

    def test_stricter_inherited_guard_is_never_weakened(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            common = repo / 'ggml/src/ggml-cuda/common.cuh'
            common.parent.mkdir(parents=True)
            common.write_text(self.SOURCE)
            with mock.patch.dict(os.environ, {}, clear=True):
                policy = qg.capture_guard_policy(repo)
                self.assertEqual(policy['sample_ms'], 10)
                self.assertEqual(policy['slack_mb'], 64)
            with mock.patch.dict(os.environ, {'GGML_CUDA_GRAPH_GUARD_SAMPLE_MS': '0'}, clear=True):
                with self.assertRaises(qg.GateError):
                    qg.capture_guard_policy(repo)
            with mock.patch.dict(os.environ, {'GGML_CUDA_GRAPH_GUARD_SLACK_MB': '128'}, clear=True):
                with self.assertRaises(qg.GateError):
                    qg.capture_guard_policy(repo)


class GuardProcessTests(unittest.TestCase):
    def test_benchmark_processes_are_detected_but_agent_text_is_not(self):
        rows = [
            {'pid': 10, 'comm': 'llama-server', 'cmdline': 'llama-server --port 1'},
            {'pid': 11, 'comm': 'ninja', 'cmdline': 'ninja -C build'},
            {'pid': 12, 'comm': 'opencode', 'cmdline': 'opencode run llama-server bench-rkva'},
            {'pid': 13, 'comm': 'python3', 'cmdline': 'python3 tools/xbox/scripts/bench-rkva-range.py'},
        ]
        with mock.patch.object(qg, 'self_pid_chain', return_value={os.getpid()}):
            matches = qg.foreign_process_matches(rows)
        self.assertEqual({row['pid'] for row in matches}, {10, 11, 13})

    def test_source_has_no_forbidden_hardware_side_effect_calls(self):
        source = SCRIPT.read_text()
        for forbidden in ('systemctl', 'start-app', 'stop-app', 'CUDA_VISIBLE_DEVICES='):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


class AstTests(unittest.TestCase):
    def test_tool_and_test_compile(self):
        compile(SCRIPT.read_text(), str(SCRIPT), 'exec')
        compile(Path(__file__).read_text(), str(Path(__file__)), 'exec')


if __name__ == '__main__':
    unittest.main()
