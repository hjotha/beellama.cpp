#!/usr/bin/env python3
"""Offline unit tests for native-restore-logits-rewind-qual.py (no hardware)."""
import importlib.util
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).with_name('native-restore-logits-rewind-qual.py')
SPEC = importlib.util.spec_from_file_location('native_restore_qual', MODULE)
QUAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUAL)

MODULE_468 = Path(__file__).with_name('native-restore-logits-rewind-qual-468.py')
SPEC_468 = importlib.util.spec_from_file_location('native_restore_qual_468', MODULE_468)
QUAL_468 = importlib.util.module_from_spec(SPEC_468)
SPEC_468.loader.exec_module(QUAL_468)


def synthetic_prompt(length=QUAL.PROMPT_N):
    return [index % 5000 for index in range(length)]


def rkvk_file(directory, prefix=b'P' * 4096, payload=b'K' * 1024, corrupt=False):
    import struct as _struct
    header = _struct.pack('<4sHHIIIIIIIIIIQQQQ', b'RKVK', 1, 80,
                          16, 0, 1024, 128, 4, 2, 128, 8, 0, 0, 7, len(payload), 0, 0)
    footer = hashlib.sha256(header + payload).digest()
    if corrupt:
        footer = bytes([footer[0] ^ 0xFF]) + footer[1:]
    path = Path(directory) / 'fresh-112000-test.bin'
    path.write_bytes(prefix + header + payload + footer + b'SCKP-appendix')
    return path, len(prefix) + 80 + len(payload) + 32


class TestPrompt(unittest.TestCase):
    def test_build_prompt(self):
        corpus = [index % 97 for index in range(200)]
        code = {'tokens': [1000 + index for index in range(64)]}
        prompt = QUAL.build_prompt(corpus, code, total=160, split=96, suffix=32)
        self.assertEqual(len(prompt), 160)
        self.assertEqual(prompt[:128], corpus[:128])
        self.assertEqual(prompt[128:], code['tokens'][-32:])
        self.assertTrue(all(type(token) is int and token >= 0 for token in prompt))

    def test_build_prompt_rejects_short_suffix(self):
        with self.assertRaises(ValueError):
            QUAL.build_prompt([1, 2, 3], {'tokens': [4]}, total=8, split=4, suffix=4)

    def test_frozen_geometry(self):
        self.assertEqual(QUAL.PROMPT_N - QUAL.PREFIX, 1)
        self.assertEqual(QUAL.CUT + 1, 111872)
        self.assertEqual(QUAL.PROMPT_N - QUAL.CUT, 129)


class TestRewindArithmetic(unittest.TestCase):
    def test_probe_construction(self):
        prompt = synthetic_prompt()
        q128 = QUAL.divergent_probe(prompt, QUAL.PROMPT_N - QUAL.REWIND, 7777)
        q129 = QUAL.divergent_probe(q128, QUAL.PROMPT_N - QUAL.REWIND - 1, 8888)
        self.assertEqual(len(q128), QUAL.PROMPT_N)
        self.assertEqual(len(q129), QUAL.PROMPT_N)
        self.assertEqual(QUAL.lcp(prompt, q128), QUAL.PROMPT_N - QUAL.REWIND)
        self.assertEqual(QUAL.lcp(q128, q129), QUAL.PROMPT_N - QUAL.REWIND - 1)
        self.assertEqual(q128[QUAL.PROMPT_N - QUAL.REWIND], 7777)
        self.assertEqual(q129[QUAL.PROMPT_N - QUAL.REWIND - 1], 8888)
        self.assertEqual(q129[QUAL.PROMPT_N - QUAL.REWIND], 7777)

    def test_128_plan(self):
        prompt = synthetic_prompt()
        q128 = QUAL.divergent_probe(prompt, QUAL.PROMPT_N - QUAL.REWIND, 7777)
        plan = QUAL.rewind_plan(prompt + [123], QUAL.PROMPT_N, q128)
        self.assertEqual(plan, {'common_prefix': QUAL.PROMPT_N - QUAL.REWIND,
                                'n_past': QUAL.PROMPT_N - QUAL.REWIND, 'removed': 128})

    def test_129_plan_after_128(self):
        prompt = synthetic_prompt()
        q128 = QUAL.divergent_probe(prompt, QUAL.PROMPT_N - QUAL.REWIND, 7777)
        q129 = QUAL.divergent_probe(q128, QUAL.PROMPT_N - QUAL.REWIND - 1, 8888)
        plan = QUAL.rewind_plan(q128 + [124], QUAL.PROMPT_N, q129)
        self.assertEqual(plan['common_prefix'], QUAL.PROMPT_N - QUAL.REWIND - 1)
        self.assertEqual(plan['n_past'], QUAL.PROMPT_N - QUAL.REWIND - 1)
        self.assertEqual(plan['removed'], 129)

    def test_repeat_plan_decrements_for_logits(self):
        q128 = synthetic_prompt()
        generated = [1, 2, 3] + list(range(13))
        plan = QUAL.rewind_plan(q128 + generated, QUAL.PROMPT_N + 15, q128)
        self.assertEqual(plan['common_prefix'], QUAL.PROMPT_N)
        self.assertEqual(plan['n_past'], QUAL.PROMPT_N - 1)
        self.assertEqual(plan['removed'], 16)

    def test_probe_rejects_bad_arguments(self):
        prompt = synthetic_prompt(32)
        with self.assertRaises(ValueError):
            QUAL.divergent_probe(prompt, 0, 1)
        with self.assertRaises(ValueError):
            QUAL.divergent_probe(prompt, len(prompt), 1)
        with self.assertRaises(ValueError):
            QUAL.divergent_probe(prompt, 4, prompt[4])
        with self.assertRaises(ValueError):
            QUAL.rewind_plan([1, 2, 3], 1, [1, 2, 3])

    def test_pick_token(self):
        self.assertEqual(QUAL.pick_token([5, 6, 7], {5, 6}), 7)
        with self.assertRaises(ValueError):
            QUAL.pick_token([5], {5})


class TestDistribution(unittest.TestCase):
    @staticmethod
    def entries(probabilities):
        return [{'id': index, 'logprob': math.log(value) if value > 0 else -3.4e38}
                for index, value in enumerate(probabilities)]

    def test_roundtrip_and_identical_metrics(self):
        probabilities = [0.5, 0.3, 0.15, 0.05]
        probs = QUAL.distribution_from_top_logprobs(self.entries(probabilities), 4)
        for got, want in zip(probs, probabilities):
            self.assertAlmostEqual(got, want, places=6)
        metrics = QUAL.distribution_metrics(probs, probs)
        self.assertEqual(metrics['kl_symmetric'], 0.0)
        self.assertEqual(metrics['total_variation'], 0.0)
        self.assertAlmostEqual(metrics['mass_left'], 1.0)
        self.assertEqual(metrics['top1_left'], metrics['top1_right'])
        self.assertTrue(all(ok for _, ok, _ in QUAL.gate_distributions(metrics)))

    def test_small_noise_passes_and_shift_fails(self):
        left = [0.5, 0.3, 0.15, 0.05]
        noisy = [0.5000000001, 0.3, 0.15, 0.0499999999]
        shifted = [0.1, 0.1, 0.1, 0.7]
        self.assertTrue(all(ok for _, ok, _ in QUAL.gate_distributions(
            QUAL.distribution_metrics(left, noisy))))
        self.assertFalse(all(ok for _, ok, _ in QUAL.gate_distributions(
            QUAL.distribution_metrics(left, shifted))))

    def test_support_loss_marks_kl_unavailable(self):
        metrics = QUAL.distribution_metrics([0.6, 0.4], [1.0, 0.0])
        self.assertEqual(metrics['support_loss'], 0.4)
        self.assertFalse(math.isfinite(metrics['kl_symmetric']))
        gates = {name: ok for name, ok, _ in QUAL.gate_distributions(metrics)}
        self.assertFalse(gates['kl'])
        self.assertFalse(gates['support'])

    def test_rejects_wrong_count_duplicate_and_bad_id(self):
        good = self.entries([0.5, 0.5])
        with self.assertRaises(ValueError):
            QUAL.distribution_from_top_logprobs(good[:1], 2)
        with self.assertRaises(ValueError):
            QUAL.distribution_from_top_logprobs([good[0], good[0]], 2)
        with self.assertRaises(ValueError):
            QUAL.distribution_from_top_logprobs([good[0], {'id': 5, 'logprob': 0.0}], 2)
        with self.assertRaises(ValueError):
            QUAL.distribution_from_top_logprobs([{'id': 0}], 1)

    def test_underflow_floor(self):
        probs = QUAL.distribution_from_top_logprobs(
            [{'id': 0, 'logprob': math.log(0.9)}, {'id': 1, 'logprob': -3.4e38}], 2)
        self.assertEqual(probs[0], 0.9)
        self.assertEqual(probs[1], 0.0)

    def test_gate_thresholds(self):
        self.assertEqual(QUAL.TOL['kl'], 1e-4)
        self.assertLess(QUAL.TOL['tv'], math.sqrt(QUAL.TOL['kl'] / 2.0))


class TestRewriteAcceptance468(unittest.TestCase):
    """The 468 variant must accept the hybrid model's safe full reprocess."""

    def test_full_reprocess_r2_evidence(self):
        timings = {'cache_n': 0, 'prompt_n': 112000, 'cache_reprocessed_n': 112000,
                   'cache_lcp_n': 111872, 'cache_reason': 'candidate', 'cache_source': 'live'}
        signature = QUAL_468.classify_rewind(timings, 111872, 112000)
        self.assertEqual(signature['mode'], 'full_reprocess')
        self.assertFalse(signature['rewind_reuse'])
        self.assertEqual(signature['cache_n'], 0)
        self.assertEqual(signature['cache_reprocessed_n'], 112000)

    def test_reuse_is_accepted(self):
        timings = {'cache_n': 111872, 'prompt_n': 128, 'cache_reprocessed_n': 128,
                   'cache_lcp_n': 111872}
        signature = QUAL_468.classify_rewind(timings, 111872, 112000)
        self.assertEqual(signature['mode'], 'reuse')
        self.assertTrue(signature['rewind_reuse'])

    def test_partial_or_short_reprocess_is_rejected(self):
        partial = {'cache_n': 111871, 'prompt_n': 129, 'cache_reprocessed_n': 129,
                   'cache_lcp_n': 111871}
        self.assertEqual(QUAL_468.classify_rewind(partial, 111872, 112000)['mode'], 'partial')
        short = {'cache_n': 0, 'prompt_n': 112000, 'cache_reprocessed_n': 111999,
                 'cache_lcp_n': 111872}
        self.assertNotIn(QUAL_468.classify_rewind(short, 111872, 112000)['mode'],
                         ('reuse', 'full_reprocess'))
        low_lcp = {'cache_n': 0, 'prompt_n': 112000, 'cache_reprocessed_n': 112000,
                   'cache_lcp_n': 111871}
        self.assertNotIn(QUAL_468.classify_rewind(low_lcp, 111872, 112000)['mode'],
                         ('reuse', 'full_reprocess'))

    def test_missing_accounting(self):
        signature = QUAL_468.classify_rewind({'cache_n': None, 'prompt_n': 1}, 111872, 112000)
        self.assertEqual(signature['mode'], 'missing')
        self.assertIsNone(signature['rewind_reuse'])

    def test_q128b_plan_after_sixteen_generations(self):
        q128 = synthetic_prompt()
        generated = [7] * 16
        plan = QUAL_468.rewind_plan(q128 + generated, QUAL_468.PROMPT_N + 15, q128)
        self.assertEqual(plan['removed'], 16)
        self.assertEqual(plan['n_past'], 111999)
        signature = QUAL_468.classify_rewind(
            {'cache_n': 111999, 'prompt_n': 1, 'cache_reprocessed_n': 1, 'cache_lcp_n': 112000},
            plan['n_past'], len(q128))
        self.assertEqual(signature['mode'], 'reuse')
        signature = QUAL_468.classify_rewind(
            {'cache_n': 0, 'prompt_n': 112000, 'cache_reprocessed_n': 112000,
             'cache_lcp_n': 112000}, plan['n_past'], len(q128))
        self.assertEqual(signature['mode'], 'full_reprocess')

    def test_q129_plan_from_post_q128b_state(self):
        q128 = synthetic_prompt()
        cached = q128 + [11] * 16
        rows = QUAL_468.PROMPT_N + 15
        keep = rows - 129
        self.assertEqual(keep, 111886)
        probe = QUAL_468.divergent_probe(cached[:rows], keep, 424242)
        plan = QUAL_468.rewind_plan(cached, rows, probe)
        self.assertEqual(plan['removed'], 129)
        # A 200 response that reports reuse here would be a forbidden 129-row trim.
        reuse = QUAL_468.classify_rewind(
            {'cache_n': keep, 'prompt_n': 129, 'cache_reprocessed_n': 129, 'cache_lcp_n': keep},
            keep, len(probe))
        self.assertEqual(reuse['mode'], 'reuse')
        full = QUAL_468.classify_rewind(
            {'cache_n': 0, 'prompt_n': len(probe), 'cache_reprocessed_n': len(probe),
             'cache_lcp_n': keep}, keep, len(probe))
        self.assertEqual(full['mode'], 'full_reprocess')


class TestFreshSnapshotMode468(unittest.TestCase):
    """Same-build fresh-snapshot mode: 112000 saved prefix, 112001 reference."""

    def test_reference_prompt_arithmetic(self):
        prompt = synthetic_prompt()
        next_token = 424242
        reference = QUAL_468.fresh_reference_prompt(prompt, next_token)
        self.assertEqual(len(prompt), QUAL_468.PROMPT_N)
        self.assertEqual(len(reference), QUAL_468.PROMPT_N + 1)
        self.assertEqual(reference[:QUAL_468.PROMPT_N], prompt)
        self.assertEqual(reference[-1], next_token)
        self.assertEqual(QUAL_468.lcp(prompt, reference), QUAL_468.PROMPT_N)
        self.assertEqual(len(reference) - len(prompt), 1)

    def test_reference_prompt_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            QUAL_468.fresh_reference_prompt([], 1)
        with self.assertRaises(ValueError):
            QUAL_468.fresh_reference_prompt([1, 2], -1)
        with self.assertRaises(ValueError):
            QUAL_468.fresh_reference_prompt([1, 2], True)

    def test_restore_and_reference_accounting(self):
        saved, request = 112000, 112001
        self.assertEqual(saved, QUAL_468.PROMPT_N)
        restored = QUAL_468.classify_rewind(
            {'cache_n': saved, 'prompt_n': 1, 'cache_reprocessed_n': 1, 'cache_lcp_n': saved},
            saved, request)
        self.assertEqual(restored['mode'], 'reuse')
        live = QUAL_468.classify_rewind(
            {'cache_n': saved, 'prompt_n': 1, 'cache_reprocessed_n': 1, 'cache_lcp_n': saved},
            saved, request)
        self.assertEqual(live['cache_n'], saved)

    def test_inventory_split_six_plus_new(self):
        originals = {f'{index:064x}' for index in range(6)}
        extra = 'e' * 64
        new = 'f' * 64
        before = originals | {extra}
        split = QUAL_468.split_inventory(before, originals)
        self.assertEqual(split['missing'], [])
        self.assertEqual(split['extras'], [extra])
        self.assertEqual(len(split['present']), 6)
        after = before | {new}
        self.assertEqual(QUAL_468.new_object_keys(before, after), [new])
        self.assertEqual(QUAL_468.new_object_keys(after, after), [])
        missing = QUAL_468.split_inventory(originals - {next(iter(originals))}, originals)
        self.assertEqual(len(missing['missing']), 1)
        self.assertEqual(missing['extras'], [])

    def test_native_trailer_and_key(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            path, frame_end = rkvk_file(directory)
            trailer = QUAL_468.native_trailer(path)
            self.assertEqual(trailer['offset'], 4096)
            self.assertEqual(trailer['frame_end'], frame_end)
            self.assertEqual(trailer['origin'], 1024)
            self.assertEqual(trailer['layers'], 16)
            full, key = QUAL_468.file_hashes(path, trailer['frame_end'])
            data = path.read_bytes()
            self.assertEqual(full, hashlib.sha256(data).hexdigest())
            self.assertEqual(key, hashlib.sha256(data[:frame_end]).hexdigest())
            self.assertNotEqual(key, full)

    def test_native_trailer_rejects_corruption(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            path, _ = rkvk_file(directory, corrupt=True)
            with self.assertRaises(ValueError):
                QUAL_468.native_trailer(path)

    def test_host_footer_metadata(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            path = Path(directory) / 'envelope.bin'
            metadata = {'n_tokens': QUAL_468.PROMPT_N, 'position': QUAL_468.PROMPT_N - 1,
                        'native_remote': True, 'native_split_p': 102400, 'source_ctx': 162816,
                        'index_block': 256}
            body = b'payload'
            blob = json.dumps(metadata).encode()
            path.write_bytes(b'qsgg' + struct.pack('<II', 4, QUAL_468.PROMPT_N) + body +
                             blob + struct.pack('<QQ8s', len(blob), 1234, b'LLROUTE1'))
            footer = QUAL_468.inspect_host_footer(path)
            self.assertEqual(footer['n_tokens'], QUAL_468.PROMPT_N)
            self.assertEqual(footer['metadata']['position'], QUAL_468.PROMPT_N - 1)
            self.assertTrue(footer['metadata']['native_remote'])


class TestPreview(unittest.TestCase):
    def test_preview_is_inert(self):
        label = f'unit-preview-{__import__("os").getpid()}'
        out = QUAL.ROOT / label
        completed = subprocess.run([sys.executable, str(MODULE), '--label', label,
                                    '--binary', sys.executable],
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"status": "PREVIEW"', completed.stdout)
        self.assertFalse(out.exists(), 'preview created its output directory')

    def test_preview_468_is_inert(self):
        label = f'unit-preview-468-{__import__("os").getpid()}'
        out = QUAL_468.ROOT / label
        completed = subprocess.run([sys.executable, str(MODULE_468), '--label', label,
                                    '--binary', sys.executable],
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"status": "PREVIEW"', completed.stdout)
        self.assertIn(QUAL_468.PFN, completed.stdout)
        self.assertFalse(out.exists(), 'preview created its output directory')


if __name__ == '__main__':
    unittest.main(verbosity=2)
