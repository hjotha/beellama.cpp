import unittest

from analyze import evaluate_job, match_suffix, parse_log, put_token


class SuffixAnalysis(unittest.TestCase):
    def test_missing_is_censored(self):
        self.assertEqual(match_suffix([11, 12], {}, 101), (0, False))
        self.assertEqual(match_suffix([11, 12], {101: 11}, 101), (1, False))
        self.assertEqual(match_suffix([11, 12], {101: 99}, 101), (0, True))
        self.assertEqual(match_suffix([11, 12], {101: 11, 102: 12}, 101), (2, True))

    def test_compares_next_round_including_bonus(self):
        job = dict(job=1, epoch=2, request=3, pos0=100, anchor=10,
                   confirmed=[11, 12, 13, 14], proposed=[11, 12, 13, 14, 15, 16, 17],
                   c=4, d=7, usable_tokens=3, ready=1, position_ok=1, prefix_ok=1)
        drafts = {104: dict(anchor=14, proposed=[15, 99], start_us=100, done_us=120)}
        commits = {104: dict(anchor=14, confirmed=[15, 16], done_us=170)}
        truth = dict(enumerate([10, 11, 12, 13, 14, 15, 16, 17, 18], start=100))
        row = evaluate_job(job, truth, drafts, commits)
        self.assertEqual(row['origin_c'], 4)
        self.assertEqual(row['next_mtp_c'], 2)
        self.assertEqual(row['delta_tokens'], 2)  # (3 + bonus) - NEXT 2
        self.assertTrue(row['comparison_complete'])
        del truth[108]  # no emitted bonus after a fully matching suffix
        self.assertFalse(evaluate_job(job, truth, drafts, commits)['comparison_complete'])

    def test_request_scoping_and_conflicts(self):
        traces, jobs = parse_log('logger prefix SHADOWTRACE {"request":1,"slot":0,"event":"token","pos":100,"token":10}\n'
                                'SHADOWTRACE {"request":2,"slot":0,"event":"token","pos":100,"token":20}\n')
        maps = {}
        for event in traces:
            put_token(maps.setdefault((event['request'], event['slot']), {}), event['pos'], event['token'])
        self.assertEqual(maps[(1, 0)][100], 10)
        self.assertEqual(maps[(2, 0)][100], 20)
        self.assertFalse(jobs)
        with self.assertRaises(ValueError):
            put_token(maps[(1, 0)], 100, 20)


if __name__ == '__main__':
    unittest.main()
