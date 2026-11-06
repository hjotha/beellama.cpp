import copy
import json
from pathlib import Path
import unittest
from qualify_abba import analyze, METRICS, GATES


def fixture(blocks=8):
    h = 'a'*64
    workload = dict(model_sha256=h, requests_sha256=h, tokens_sha256=h,
                    hardware='SYNTHETIC hardware, not Xbox measurements',
                    configuration={'batch': 512, 'ubatch': 256, 'context': 129024, 'codec': 'F16'},
                    command=['synthetic-fixture'], sampling={'seed': 42}, thermal_clock_policy='synthetic constant')
    variants = {v: dict(source_sha256=v.lower()*64, binary_sha256=v.lower()*64,
                       pso_sha256=v.lower()*64, commit='synthetic-'+v, compiler='synthetic',
                       compiler_flags=['-O3'], feature_flags={'candidate': v=='B'}) for v in 'AB'}
    data = dict(schema=1, data_kind='synthetic', workload=workload, variants=variants,
                correctness={g: dict(status='pass', evidence_sha256=h) for g in GATES}, runs=[])
    for i, v in enumerate('ABBA'*blocks, 1):
        data['runs'].append(dict(run_id=f'synthetic-{i}', sequence=i, variant=v, warmup=False,
                                workload=copy.deepcopy(workload), artifact=copy.deepcopy(variants[v]),
                                raw_log_sha256=h, timings={m: 100 if v=='A' else 90 for m in METRICS}))
    return data


class QualificationTests(unittest.TestCase):
    def test_synthetic_never_qualifies(self):
        result = analyze(fixture())
        self.assertEqual(result['decision'], 'not_qualified')
        self.assertEqual(result['reasons'], ['synthetic_data_not_hardware_evidence'])

    def test_kernel_only_not_total_gain(self):
        data = fixture()
        for r in data['runs']:
            for m in ('decode_ms', 'total_ms'): r['timings'][m] = 100
        result = analyze(data)
        self.assertEqual(result['metrics']['kernel_ms']['screen'], 'improvement_signal')
        self.assertIn('total_ms_gain_not_established', result['reasons'])

    def test_insufficient_replicates(self):
        result = analyze(fixture(5))
        self.assertEqual(result['metrics']['total_ms']['screen'], 'inconclusive')

    def test_noise_and_tiny_gain(self):
        for mode in ('tiny', 'drift'):
            data = fixture()
            for i, r in enumerate(data['runs']):
                r['timings'] = {m: (100 if r['variant']=='A' else 99.92) if mode=='tiny'
                               else [90, 95, 95, 110][i%4] for m in METRICS}
            self.assertEqual(analyze(data)['metrics']['total_ms']['screen'], 'inconclusive')

    def test_replica_dispersion(self):
        data = fixture()
        for i, r in enumerate(data['runs']):
            if r['variant']=='B': r['timings']['total_ms'] = [80, 120][(i//4)%2]
        self.assertEqual(analyze(data)['metrics']['total_ms']['screen'], 'inconclusive')

    def test_regression(self):
        data = fixture()
        for r in data['runs']:
            if r['variant']=='B': r['timings']['prefill_ms'] = 120
        self.assertIn('prefill_ms_regression', analyze(data)['reasons'])

    def test_quality_cannot_be_overridden_by_speed(self):
        for state in ('fail', 'not_run'):
            data = fixture()
            data['correctness']['long_context_quality']['status'] = state
            self.assertIn('correctness_long_context_quality_'+state, analyze(data)['reasons'])

    def test_measured_path_only_review_not_promotion(self):
        # Exercise logic only: relabeling a fixture is NOT real evidence.
        data = fixture(); data['data_kind'] = 'measured'
        self.assertEqual(analyze(data)['decision'], 'eligible_for_human_review_only')

    def test_invalid_evidence(self):
        mutations = [
            lambda d: d['runs'][0]['workload'].update(tokens_sha256='b'*64),
            lambda d: d['runs'][0]['artifact'].update(pso_sha256='c'*64),
            lambda d: d['runs'][0].update(variant='B'),
            lambda d: d['runs'][1].update(sequence=1),
            lambda d: d['runs'][1].update(run_id='synthetic-1'),
            lambda d: d['runs'][1].update(warmup=True),
            lambda d: d['runs'].pop(),
            lambda d: d['runs'][0]['timings'].update(total_ms=float('nan')),
            lambda d: d['runs'][0]['timings'].update(total_ms=float('inf')),
            lambda d: d['runs'][0]['timings'].update(total_ms=0),
            lambda d: d['runs'][0]['timings'].update(total_ms=True),
            lambda d: d['runs'][0]['timings'].pop('kernel_ms'),
            lambda d: d['variants']['B'].update(compiler_flags=['-Ofast']),
            lambda d: d['correctness']['serial'].update(evidence_sha256='missing'),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                data = fixture(); mutate(data)
                with self.assertRaises(ValueError): analyze(data)

    def test_warmup_excluded(self):
        data = fixture()
        warmup = copy.deepcopy(data['runs'][0]); warmup.update(run_id='warmup', sequence=1, warmup=True)
        for r in data['runs']: r['sequence'] += 1
        data['runs'].insert(0, warmup)
        self.assertEqual(analyze(data)['complete_blocks'], 8)


if __name__ == '__main__':
    unittest.main()
