"""Hardware-free regressions for the RKVA decode-path A/B runner.

Run: python3 -m unittest discover -s tools/xbox/tests -p test_run_rkva_decode_host_ab.py

The runner only executes on --execute and only on a quiet host, so these tests
pin the parts that can be checked without GPU, Xbox or RPC: the arm plan, the
Codex process guard (which must not match agent command text or itself), the
token parity comparison and the profile/guard log parsers.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run-rkva-decode-host-ab.py'
spec = importlib.util.spec_from_file_location('run_rkva_decode_host_ab', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ArmPlanTest(unittest.TestCase):
    def test_default_plan_is_abba(self):
        self.assertEqual(runner.arm_plan('A,B,B,A'), ['A', 'B', 'B', 'A'])

    def test_unknown_arm_is_rejected(self):
        with self.assertRaises(ValueError):
            runner.arm_plan('A,C')

    def test_arms_only_flip_the_guard_variable(self):
        self.assertEqual(runner.ARM_ENVIRONMENT['A'], {'GGML_CUDA_GRAPH_GUARD_SAMPLE_MS': '0'})
        self.assertEqual(runner.ARM_ENVIRONMENT['B'], {'GGML_CUDA_GRAPH_GUARD_SAMPLE_MS': '10'})


class CodexGuardTest(unittest.TestCase):
    def rows(self, *items):
        return [tuple(item) for item in items]

    def test_benchmark_executables_are_detected(self):
        rows = self.rows(
            (10, 1, 'llama-server', '/home/x/build/bin/llama-server --port 8080'),
            (11, 1, 'ninja', 'ninja -C build'),
            (12, 1, 'cmake', 'cmake --build build -j1'),
            (13, 1, 'python3', 'python3 tools/xbox/scripts/bench-rkva-range.py --execute'),
            (14, 1, 'test-rkva-transport', 'test-rkva-transport'),
        )
        matched = {entry['pid'] for entry in runner.codex_matches(rows)}
        self.assertEqual(matched, {10, 11, 12, 13, 14})

    def test_agent_command_text_is_not_a_match(self):
        # Agent prompts mention the benchmark names; only the executable counts.
        rows = self.rows(
            (20, 1, 'opencode', 'opencode run --auto llama-server bench-rkva test-rkva ninja'),
            (21, 1, 'bash', 'bash -c "pgrep -af llama-server|bench-rkva|ninja"'),
            (22, 1, 'cmake', 'cmake -B build -DGGML_CUDA=ON'),
            (23, 1, 'python3', 'python3 tools/xbox/scripts/analyze-rkva-decode-trace.py trace.jsonl'),
        )
        self.assertEqual(runner.codex_matches(rows), [])

    def test_self_and_ancestors_are_ignored(self):
        rows = self.rows(
            (30, 1, 'python3', 'python3 tools/xbox/scripts/run-rkva-decode-host-ab.py --dry-run'),
            (31, 30, 'bash', 'bash -c run-rkva-decode-host-ab.py'),
        )
        self.assertEqual(runner.codex_matches(rows, {30, 31}), [])

    def test_codex_session_is_marked_session_only(self):
        rows = self.rows((40, 1, 'codex', 'codex resume 01a1'))
        matches = runner.codex_matches(rows)
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0]['session_only'])

    def test_self_pid_chain_walks_ancestors(self):
        rows = self.rows((5, 4, 'a', 'a'), (4, 3, 'b', 'b'), (3, 0, 'init', 'init'))
        self.assertEqual(runner.self_pid_chain(rows, 5), {5, 4, 3})

    def test_self_pid_chain_includes_current_process(self):
        import os
        rows = self.rows((os.getpid(), 1, 'python3', 'python3 x.py'))
        self.assertIn(os.getpid(), runner.self_pid_chain(rows))


class ParityTest(unittest.TestCase):
    def test_identical_streams_match(self):
        passes = [{'name': 'p1', 'arm': 'A', 'tokens': [1, 2, 3]},
                  {'name': 'p2', 'arm': 'B', 'tokens': [1, 2, 3]}]
        report = runner.parity_report(passes)
        self.assertTrue(report['matched'])
        self.assertEqual(report['lengths'], {'p1': 3, 'p2': 3})

    def test_divergent_stream_is_rejected(self):
        passes = [{'name': 'p1', 'arm': 'A', 'tokens': [1, 2, 3]},
                  {'name': 'p2', 'arm': 'B', 'tokens': [1, 2, 4]}]
        report = runner.parity_report(passes)
        self.assertFalse(report['matched'])
        self.assertEqual(report['mismatches'][0]['index'], 2)

    def test_length_change_is_rejected(self):
        passes = [{'name': 'p1', 'arm': 'A', 'tokens': [1, 2, 3]},
                  {'name': 'p2', 'arm': 'B', 'tokens': [1, 2]}]
        self.assertFalse(runner.parity_report(passes)['matched'])

    def test_missing_tokens_is_rejected(self):
        passes = [{'name': 'p1', 'arm': 'A', 'tokens': [1]}, {'name': 'p2', 'arm': 'B'}]
        report = runner.parity_report(passes)
        self.assertFalse(report['matched'])
        self.assertEqual(report['mismatches'][0]['error'], 'no generated tokens')


class ResponseParsingTest(unittest.TestCase):
    def write(self, payload):
        handle = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
        with handle:
            json.dump(payload, handle)
        return Path(handle.name)

    def test_tokens_round_trip(self):
        path = self.write({'tokens': [5, 6, 7], 'content': 'abc'})
        self.assertEqual(runner.tokens_from_response(path), [5, 6, 7])
        path.unlink()

    def test_missing_tokens_fails(self):
        path = self.write({'content': 'abc'})
        with self.assertRaises(ValueError):
            runner.tokens_from_response(path)
        path.unlink()

    def test_non_integer_tokens_fail(self):
        path = self.write({'tokens': [5, '6']})
        with self.assertRaises(ValueError):
            runner.tokens_from_response(path)
        path.unlink()


class LogParsingTest(unittest.TestCase):
    def test_profile_line_takes_the_last_payload(self):
        lines = [
            'remote_attn gpu_profile {"counters":{"submits":1}}',
            'other line',
            'remote_attn gpu_profile {"counters":{"submits":7},"memory":{"bytes":6}}',
        ]
        profile = runner.parse_profile_line(lines)
        self.assertEqual(profile['counters']['submits'], 7)
        self.assertEqual(profile['memory']['bytes'], 6)

    def test_profile_line_absent(self):
        self.assertIsNone(runner.parse_profile_line(['no profile here']))

    def test_guard_line_parses_activation(self):
        lines = ['llama_backend_init: headroom guard submissions fastpath=1200 queried=7 sample_age_ms=10']
        guard = runner.parse_guard_line(lines)
        self.assertEqual(guard['fastpath'], 1200)
        self.assertEqual(guard['queried'], 7)
        self.assertEqual(guard['sample_age_ms'], 10)

    def test_guard_line_absent(self):
        self.assertIsNone(runner.parse_guard_line(['nothing']))

    def test_arm_summary_uses_medians(self):
        passes = [
            {'arm': 'A', 'decode_tps': [10.0]},
            {'arm': 'A', 'decode_tps': [12.0]},
            {'arm': 'B', 'decode_tps': [20.0]},
        ]
        summary = runner.summarize_arm(passes, 'A')
        self.assertEqual(summary['n'], 2)
        self.assertEqual(summary['p50_tps'], 11.0)
        self.assertAlmostEqual(summary['ms_per_token_p50'], 1000.0 / 11.0)


if __name__ == '__main__':
    unittest.main()
