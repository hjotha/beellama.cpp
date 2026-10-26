#!/usr/bin/env python3
"""Host tests for the CPU model of the gpupeak STREAM kernels.

Run: python3 -m unittest tools/xbox/tests/test_gpupeak_stream_model.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpupeak_stream_model as model  # noqa: E402


class StreamModelTest(unittest.TestCase):
    def test_analytic_helper_matches_a_direct_vector_walk(self):
        for shape in ((1, 1, 1), (255, 1, 1), (512, 2, 1), (1000, 2, 1),
                      (4097, 3, 1), (16384, 4, 2)):
            n_vec, gx, gy = shape
            self.assertEqual(model.analytic_pattern_xor(n_vec, gx, gy, 0),
                             model.naive_vector_xor(n_vec, gx, gy, 0), shape)
        # mode is part of the pattern; an odd count does not cancel it.
        self.assertNotEqual(model.analytic_pattern_xor(4097, 2, 1, 0),
                            model.analytic_pattern_xor(4097, 2, 1, 1))

    def test_per_thread_fold_reproduces_the_expected_checksum(self):
        for shape in ((512, 2, 1), (1000, 2, 1), (4097, 2, 1),
                      (65536, 2048, 1), (268435456 // 16, 2048, 1)):
            n_vec, gx, gy = shape
            expected = model.analytic_pattern_xor(n_vec, gx, gy, 0)
            self.assertEqual(model.fixed_read_checksum(n_vec, gx, gy, 0), expected, shape)

    def test_per_thread_model_matches_direct_walk_on_small_shapes(self):
        for shape in ((512, 2, 1), (1000, 2, 1), (4097, 2, 1)):
            n_vec, gx, gy = shape
            sink = model.per_thread_sink(n_vec, gx, gy, 0)
            direct = 0
            for acc in sink:
                direct ^= acc
            self.assertEqual(direct, model.naive_vector_xor(n_vec, gx, gy, 0), shape)

    def test_group_slot_race_is_not_a_group_fold(self):
        # One vector per thread: every thread stores a different accumulator in
        # the shared group slot. Any single winner misses the other 255
        # accumulators, so the group XOR cannot equal the expected checksum.
        n_vec, gx, gy = 512, 2, 1
        expected = model.analytic_pattern_xor(n_vec, gx, gy, 0)
        raced = model.group_slot_models(n_vec, gx, gy, 0)
        self.assertNotEqual(raced["first_lane"], expected)
        self.assertNotEqual(raced["last_lane"], expected)

    def test_shipped_256mib_plan_is_an_even_period(self):
        # 256 MiB / 16 B = 16,777,216 vectors; 2048 groups * 256 threads is
        # 524,288 threads, so every thread owns 32 vectors (even). This is why
        # the old race coincidentally read 0 for this one shape, and why the
        # console mismatch had to come from the undefined UAV->COPY_SOURCE
        # readback as well.
        shape = model.model_shape(268435456 // 16, 2048, 1)
        self.assertEqual(shape["vectors_per_thread"], 32)
        self.assertEqual(shape["expected"], 0)
        self.assertTrue(shape["fixed_matches"])


if __name__ == "__main__":
    unittest.main()
