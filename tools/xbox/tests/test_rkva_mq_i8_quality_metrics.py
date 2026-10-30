#!/usr/bin/env python3
"""Regressions for CPU INT8 qualification accounting, not GPU validation."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rkva_mq_i8_quality as quality


class MetricTests(unittest.TestCase):
    def test_uniform_full_domain_has_zero_error(self):
        weights = quality.full_softmax(np.zeros(128), np.zeros(128))
        gold = np.full(256, 1 / 256)
        self.assertEqual(float(np.abs(weights - gold).sum()), 0)
        # The former comparison has a 0.5 artificial tail-mass error.
        self.assertEqual(float(np.abs(np.full(128, 1 / 128) - gold[:128]).sum()), 0.5)

    def test_empty_partition_and_stability(self):
        for records, tail in (([], [1000., 1000.]), ([1000., 1000.], [])):
            np.testing.assert_array_equal(quality.full_softmax(records, tail), [.5, .5])
        for records, tail in (([], []), ([np.nan], []), ([np.inf], [])):
            with self.assertRaises(ValueError):
                quality.full_softmax(records, tail)

    def test_tail_affects_full_distribution(self):
        weights = quality.full_softmax([0.], [np.log(3.)])
        np.testing.assert_allclose(weights, [.25, .75])

    def test_means_are_not_maxima(self):
        means, distribution = quality.summarize_metrics({'rel_o_max': [(1., 2.), (9., 12.)]})
        self.assertEqual(means['rel_o_max'], {'ref': 5., 'i8': 7.})
        self.assertEqual(distribution['rel_o_max']['i8']['max'], 12.)
        self.assertEqual(distribution['rel_o_max']['i8']['max_query_index'], 1)
        self.assertAlmostEqual(distribution['rel_o_max']['i8']['p95'], 11.5)

    def test_invalid_metric_inputs(self):
        for rows in ([], [(1.,)], [(float('nan'), 1.)]):
            with self.assertRaises(ValueError):
                quality.summarize_metrics({'x': rows})
        with self.assertRaises(ValueError):
            quality.run_case(np.random.default_rng(1), 256, 'tight', n_queries=0)

    def test_gate_uses_means_and_unchanged_thresholds(self):
        case = {'score_rms': {'added_over_ref': .35, 'ratio': 1.15},
                'metrics': {key: {'ref': 1., 'i8': 1.15} for key in
                            ('rel_o_rms', 'rel_o_max', 'one_minus_cos', 'lse_abs_max', 'weight_l1')},
                'metric_distribution': {'rel_o_max': {'i8': {'max': 100.}}}}
        self.assertTrue(quality.verdict(case)[0])
        case['metrics']['weight_l1']['i8'] = 1.15001
        self.assertFalse(quality.verdict(case)[0])

    def test_cli_rejects_empty_query_set(self):
        result = subprocess.run([sys.executable, quality.__file__, '--queries', '0'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--queries must be positive', result.stderr)

    def test_cli_json_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'quality.json'
            result = subprocess.run([sys.executable, quality.__file__, '--quick', '--queries', '2',
                                     '--json', str(output)], capture_output=True, text=True)
            self.assertIn(result.returncode, (0, 1))  # Both valid gate outcomes serialize.
            report = json.loads(output.read_text())
        self.assertEqual(report['schema'], 'rkva-mq-i8-quality-v2')
        self.assertEqual(report['metric_contract']['weight_l1_domain'], 'full records+tail')
        self.assertIn('query MEANS', result.stdout)
        self.assertIn('query maxima', result.stdout)
        for key, labels in report['gated_query_maxima'].items():
            for label, maximum in labels.items():
                source = report['cases'][maximum['case_index']]['case']
                self.assertTrue(source['dc_exact'])
                column = ('ref', 'i8').index(label)
                self.assertEqual(maximum['value'],
                                 source['per_query_metrics'][key][maximum['query_index']][column])
                self.assertEqual(maximum['value'], max(
                    row['case']['metric_distribution'][key][label]['max']
                    for row in report['cases'] if row['case']['dc_exact']))
        for row in report['cases']:
            case = row['case']
            self.assertEqual(set(case['metrics']), set(case['per_query_metrics']))
            for key, pairs in case['per_query_metrics'].items():
                self.assertEqual(len(pairs), 2)
                for index, label in enumerate(('ref', 'i8')):
                    self.assertAlmostEqual(case['metrics'][key][label],
                                           np.mean([pair[index] for pair in pairs]))
                    self.assertEqual(case['metric_distribution'][key][label]['max'],
                                     max(pair[index] for pair in pairs))
            self.assertEqual((row['ok'], row['reasons']), quality.verdict(case))


if __name__ == '__main__':
    unittest.main()
