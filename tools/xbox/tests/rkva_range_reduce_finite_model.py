#!/usr/bin/env python3
"""CPU numerical contract tests for poisoned wave/split states and GPU Q layout.

This is a float32 model, not D3D12 execution or a throughput measurement.
"""
import math
import unittest

import numpy as np

import rkva_range_kvarn4_opt_model as model


class FiniteReduction(unittest.TestCase):
    def state(self, maximum=0.0, denominator=1.0, numerator=2.0):
        return (np.array([maximum], np.float32), np.array([denominator], np.float32),
                np.full((1, 256), numerator, np.float32))

    def assert_poison(self, states):
        for rotated in (False, True):
            out, lse = model.reduce_partials(states, rotated=rotated)
            self.assertTrue(np.all(np.isnan(out)))
            self.assertTrue(np.all(np.isnan(lse)))

    def test_empty_and_partially_empty(self):
        # Empty numerator memory is not participating; its poison is ignored.
        empty = self.state(maximum=-math.inf, denominator=0, numerator=math.nan)
        for splits in (1, 2, 32, 64, 128, 256):
            out, lse = model.reduce_partials([empty] * splits, rotated=True)
            np.testing.assert_array_equal(out, 0)
            np.testing.assert_array_equal(lse, -math.inf)
            out, lse = model.reduce_partials([empty] * (splits - 1) + [self.state()])
            np.testing.assert_array_equal(out, 2)
            np.testing.assert_array_equal(lse, 0)

    def test_bad_denominator_survives_valid_neighbors(self):
        for denominator in (math.nan, math.inf, -math.inf, -1.0):
            for reverse in (False, True):
                states = [self.state(), self.state(denominator=denominator)]
                self.assert_poison(states[::-1] if reverse else states)

    def test_bad_participating_maximum(self):
        for maximum in (math.nan, math.inf, -math.inf):
            self.assert_poison([self.state(), self.state(maximum=maximum)])

    def test_bad_numerator_cannot_hide_under_zero_weight(self):
        for value in (math.nan, math.inf, -math.inf):
            bad = self.state(maximum=-1000.0)
            bad[2][0, 173] = value
            self.assert_poison([self.state(), bad])
            # The producer must poison before body/tail wave weights suppress it.
            merged = model.merge_partial_states([self.state(), bad])
            self.assertTrue(np.isnan(merged[1][0]))
            self.assertTrue(np.all(np.isnan(merged[2])))

    def test_sum_overflow_is_poisoned(self):
        huge = np.finfo(np.float32).max
        self.assert_poison([self.state(denominator=huge)] * 2)
        self.assert_poison([self.state(numerator=huge)] * 2)

    def test_post_reduce_wht_matches_float64_original_domain(self):
        rng = np.random.default_rng(20261020)
        for splits in (1, 2, 3, 32, 64, 128, 256):
            states, original = [], []
            for _ in range(splits):
                body = rng.normal(0, 0.25, (1, 256)).astype(np.float32)
                tail = rng.normal(0, 0.25, (1, 256)).astype(np.float32)
                maximum, den = float(rng.uniform(-10, 10)), float(rng.uniform(1, 8))
                rotated = (body + model.butterfly_wht(tail)).astype(np.float32)
                states.append((np.array([maximum], np.float32), np.array([den], np.float32),
                               rotated))
                original.append((maximum, den, model.butterfly_wht(body).astype(np.float64) + tail))
            out, lse = model.reduce_partials(states, rotated=True)
            maximum = max(s[0] for s in original)
            den = sum(s[1] * math.exp(s[0] - maximum) for s in original)
            expected = sum(s[2] * math.exp(s[0] - maximum) for s in original) / den
            np.testing.assert_allclose(out, expected, rtol=5e-5, atol=8e-7)
            np.testing.assert_allclose(lse, maximum + math.log(den), rtol=0, atol=2e-6)


class BulkGpuLayout(unittest.TestCase):
    def unpack_model(self, half, positions):
        # Mirror the 256-thread, two-head LDS address mapping, independently
        # checking against the existing per-head butterfly oracle below.
        pairs = half.size // 2
        output = np.full(half.size * 2 + positions.size, 0xFFFFFFFF, np.uint32)
        floats = output.view(np.float32)
        for group in range((pairs + 255) // 256):
            begin, end = group * 512, min(half.size, (group + 1) * 512)
            shared = np.zeros(512, np.float32)
            shared[:end - begin] = half[begin:end].astype(np.float32)
            floats[begin:end] = shared[:end - begin]
            indices = np.arange(512)
            for stride in (1, 2, 4, 8, 16, 32, 64):
                own, other = shared.copy(), shared[indices ^ stride].copy()
                shared = np.where(indices & stride, other - own, own + other).astype(np.float32)
            own = shared * np.float32(0.08838834764831845)
            other = shared[indices ^ 128] * np.float32(0.08838834764831845)
            rotated = np.where(indices & 128, other - own, own + other)
            rotated *= np.float32(0.7071067811865475)
            floats[half.size + begin:half.size + end] = rotated[:end - begin]
        output[half.size * 2:] = positions.view(np.uint32)
        return output

    def test_raw_rotated_positions_even_and_odd_heads(self):
        rng = np.random.default_rng(20261020)
        for nq in (1, 2, 3, 8, 9):
            for heads in (1, 2, 3, 24):
                half = rng.normal(0, 8, nq * heads * 256).astype(np.float16)
                positions = np.arange(102400, 102400 + nq, dtype=np.int32)
                output = self.unpack_model(half, positions)
                raw = half.astype(np.float32)
                expected_rot = model.butterfly_wht(raw.reshape(-1, 256)).reshape(-1)
                np.testing.assert_array_equal(output[:half.size], raw.view(np.uint32))
                np.testing.assert_array_equal(output[half.size:half.size * 2],
                                              expected_rot.view(np.uint32))
                np.testing.assert_array_equal(output[half.size * 2:], positions.view(np.uint32))


if __name__ == '__main__':
    unittest.main(verbosity=2)
