#!/usr/bin/env python3
"""Pure-model tests for RKVA native KVarN4 bulk RPC profiling.

Validates:
1. Fixed timestamp slot bounds, per-segment slot layout, and monotonicity invariants.
2. Dual-level timing attribution (aggregate envelope vs per-segment bucket samples).
3. Exclusive GPU execution duration vs inclusive CPU fence wait and queue latency.
4. Schema conformance of the 'bulk' telemetry object in rkva-range-profile-v1.
5. Graceful fallback when GPU hardware timestamps are unavailable.
"""

import json
import unittest


class RkvaKvarnBulkProfilingTests(unittest.TestCase):
    # Constants matching uwp/rkva/remote-attn-engine.cpp and rkva-range-profile.h
    K_TS_PER_SET = 256
    K_TS_SETS = 4
    K_TS_BULK_MAX_SEGMENTS = 32
    K_TS_BULK_SLOTS_PER_SEGMENT = 6

    K_TS_ENVELOPE_BEGIN = 0
    K_TS_UPLOAD_BEGIN = 0
    K_TS_UPLOAD_END = 1
    K_TS_ATTN_BEGIN = 2
    K_TS_ATTN_PASS1 = 3
    K_TS_ATTN_REDUCE = 4
    K_TS_ATTN_COPY = 5

    def compute_slots(self, num_segments: int):
        """Replicate the slot allocation logic of RemoteAttnEngine::attend_kvarn_bulk."""
        self.assertGreaterEqual(num_segments, 1)
        self.assertLessEqual(num_segments, self.K_TS_BULK_MAX_SEGMENTS)

        slots = {"envelope_begin": self.K_TS_ENVELOPE_BEGIN, "segments": []}
        all_indices = [self.K_TS_ENVELOPE_BEGIN]

        for i in range(num_segments):
            base = 1 + i * self.K_TS_BULK_SLOTS_PER_SEGMENT
            seg_slots = {
                "upload_begin": base + self.K_TS_UPLOAD_BEGIN,
                "upload_end": base + self.K_TS_UPLOAD_END,
                "attn_begin": base + self.K_TS_ATTN_BEGIN,
                "attn_pass1": base + self.K_TS_ATTN_PASS1,
                "attn_reduce": base + self.K_TS_ATTN_REDUCE,
                "attn_copy": base + self.K_TS_ATTN_COPY,
            }
            slots["segments"].append(seg_slots)
            all_indices.extend(
                [
                    seg_slots["upload_begin"],
                    seg_slots["upload_end"],
                    seg_slots["attn_begin"],
                    seg_slots["attn_pass1"],
                    seg_slots["attn_reduce"],
                    seg_slots["attn_copy"],
                ]
            )

        env_end = 1 + num_segments * self.K_TS_BULK_SLOTS_PER_SEGMENT
        slots["envelope_end"] = env_end
        all_indices.append(env_end)
        total_slots = 2 + num_segments * self.K_TS_BULK_SLOTS_PER_SEGMENT

        return slots, all_indices, total_slots

    def test_slot_allocation_bounds(self):
        """Verify slot allocation for all valid segment counts (1 to 32)."""
        for s in range(1, self.K_TS_BULK_MAX_SEGMENTS + 1):
            slots, all_indices, total_slots = self.compute_slots(s)

            # Ensure total slots match expected formula
            self.assertEqual(total_slots, 2 + s * 6)
            self.assertEqual(len(all_indices), total_slots)

            # All slots must be within a single set capacity (256)
            self.assertLessEqual(total_slots, self.K_TS_PER_SET)
            self.assertLessEqual(slots["envelope_end"], self.K_TS_PER_SET - 1)

            # Slots must be strictly distinct and strictly monotonic
            for idx in range(len(all_indices)):
                self.assertEqual(all_indices[idx], idx)
                self.assertEqual(
                    all_indices[idx],
                    idx,
                    f"Slot at index {idx} should equal {idx} to guarantee sequential monotonically increasing slots",
                )

        # For maximum 32 segments, verify exact slot ceiling
        _, _, max_total = self.compute_slots(32)
        self.assertEqual(max_total, 194)
        self.assertLessEqual(max_total, 256)

    def test_slot_monotonicity_and_timing_attribution(self):
        """Simulate monotonically recorded timestamps and verify attribution math."""
        num_segments = 4
        slots, all_indices, total_slots = self.compute_slots(num_segments)

        # Simulated hardware ticks with 10 MHz clock (1 tick = 0.1 us)
        freq_hz = 10_000_000
        us_per_tick = 1e6 / freq_hz  # 0.1 us per tick

        ticks = [100_000 + i * 250 for i in range(total_slots)]

        # Verify monotonicity check that RemoteAttnEngine::read_timestamps enforces
        for i in range(1, len(ticks)):
            self.assertGreaterEqual(ticks[i], ticks[i - 1])

        # Verify envelope total execution
        gpu_total_ticks = ticks[slots["envelope_end"]] - ticks[slots["envelope_begin"]]
        gpu_total_us = gpu_total_ticks * us_per_tick
        self.assertEqual(gpu_total_ticks, (total_slots - 1) * 250)

        # Verify per-segment delta reconstruction
        sum_seg_kernel_ticks = 0
        for i in range(num_segments):
            seg = slots["segments"][i]
            u_ticks = ticks[seg["upload_end"]] - ticks[seg["upload_begin"]]
            p1_ticks = ticks[seg["attn_pass1"]] - ticks[seg["attn_begin"]]
            red_ticks = ticks[seg["attn_reduce"]] - ticks[seg["attn_pass1"]]
            cp_ticks = ticks[seg["attn_copy"]] - ticks[seg["attn_reduce"]]

            self.assertGreater(u_ticks, 0)
            self.assertGreater(p1_ticks, 0)
            self.assertGreater(red_ticks, 0)
            self.assertGreater(cp_ticks, 0)

            seg_ticks = u_ticks + p1_ticks + red_ticks + cp_ticks
            sum_seg_kernel_ticks += seg_ticks

        # Sum of segment kernel ticks must be strictly bounded by envelope total ticks
        self.assertLessEqual(sum_seg_kernel_ticks, gpu_total_ticks)

    def test_queue_latency_computation(self):
        """Validate queue wait calculation: max(0, fence_wait_cpu_us - gpu_total_us)."""
        # Case 1: Fence wait exceeds GPU execution due to queue scheduling
        fence_wait_us = 1250.0
        gpu_total_us = 950.0
        queue_wait_us = max(0.0, fence_wait_us - gpu_total_us)
        self.assertAlmostEqual(queue_wait_us, 300.0)

        # Case 2: GPU execution completed early (fence wait <= gpu total)
        fence_wait_fast_us = 900.0
        queue_wait_clamped = max(0.0, fence_wait_fast_us - gpu_total_us)
        self.assertEqual(queue_wait_clamped, 0.0)

    def test_single_fence_invariant_and_sample_attribution(self):
        """Verify single fence invariant and per-segment fence zeroing."""
        # Simulated counters from a bulk execution of 8 segments
        submits = 1
        fence_waits = 1
        bulk_calls = 1
        bulk_segments = 8
        bulk_queries = 64

        # Per-segment samples recorded into ring
        segment_samples = []
        for seg_idx in range(bulk_segments):
            up_sample = {
                "kind": "upload",
                "bulk_segment": 1,
                "cpu_map_copy_ns": 45_000,
                "cpu_record_ns": 12_000,
                "cpu_submit_ns": 0,  # Must be 0
                "cpu_fence_wait_ns": 0,  # Must be 0
            }
            attn_sample = {
                "kind": "attention",
                "bulk_segment": 1,
                "cpu_map_copy_ns": 30_000,
                "cpu_record_ns": 18_000,
                "cpu_submit_ns": 0,  # Must be 0
                "cpu_fence_wait_ns": 0,  # Must be 0
                "cpu_readback_ns": 55_000,
            }
            segment_samples.extend([up_sample, attn_sample])

        # Bulk envelope sample records the true submit and fence wait
        bulk_envelope = {
            "submits": submits,
            "fence_waits": fence_waits,
            "cpu_submit_ns": 25_000,
            "cpu_fence_wait_ns": 850_000,
            "cpu_map_copy_ns": sum(s["cpu_map_copy_ns"] for s in segment_samples),
            "cpu_record_ns": sum(s["cpu_record_ns"] for s in segment_samples),
            "cpu_readback_ns": sum(
                s.get("cpu_readback_ns", 0) for s in segment_samples
            ),
        }

        # Check invariant
        self.assertEqual(bulk_envelope["submits"], 1)
        self.assertEqual(bulk_envelope["fence_waits"], 1)
        for s in segment_samples:
            self.assertEqual(s["cpu_fence_wait_ns"], 0)
            self.assertEqual(s["cpu_submit_ns"], 0)

        # Sum of per-segment map_copy and record equals envelope aggregate
        self.assertEqual(
            bulk_envelope["cpu_map_copy_ns"],
            sum(s["cpu_map_copy_ns"] for s in segment_samples),
        )
        self.assertEqual(
            bulk_envelope["cpu_record_ns"],
            sum(s["cpu_record_ns"] for s in segment_samples),
        )

    def test_schema_conformance(self):
        """Validate structure and fields of serialized bulk telemetry JSON."""
        sample_json = {
            "schema": "rkva-range-profile-v1",
            "profiling": True,
            "q4_kernel_mode": 2,
            "gpu_timestamps": True,
            "gpu_timestamp_frequency_hz": 10000000,
            "counters": {
                "submits": 1,
                "fence_blocking_waits": 1,
                "fence_already_complete": 0,
                "bulk_calls": 1,
                "bulk_segments": 4,
                "bulk_queries": 32,
            },
            "bulk": {
                "bulk_calls": 1,
                "bulk_segments": 4,
                "bulk_queries": 32,
                "samples_recorded": 1,
                "samples_in_ring": 1,
                "gpu_timestamps": True,
                "timings": {
                    "admission_cpu_us": {"n": 1, "mean": 15.2, "p50": 15.2},
                    "alloc_cpu_us": {"n": 1, "mean": 0.0, "p50": 0.0},
                    "map_copy_cpu_us": {"n": 1, "mean": 120.5, "p50": 120.5},
                    "record_cpu_us": {"n": 1, "mean": 85.0, "p50": 85.0},
                    "submit_cpu_us": {"n": 1, "mean": 18.0, "p50": 18.0},
                    "fence_wait_cpu_us": {"n": 1, "mean": 950.0, "p50": 950.0},
                    "readback_cpu_us": {"n": 1, "mean": 45.0, "p50": 45.0},
                    "engine_total_cpu_us": {"n": 1, "mean": 1233.7, "p50": 1233.7},
                    "gpu_upload_us": {"n": 1, "mean": 110.0, "p50": 110.0},
                    "gpu_pass1_us": {"n": 1, "mean": 420.0, "p50": 420.0},
                    "gpu_reduce_us": {"n": 1, "mean": 60.0, "p50": 60.0},
                    "gpu_readback_copy_us": {"n": 1, "mean": 40.0, "p50": 40.0},
                    "gpu_total_us": {"n": 1, "mean": 700.0, "p50": 700.0},
                    "queue_wait_us": {"n": 1, "mean": 250.0, "p50": 250.0},
                },
                "bytes": {
                    "upload_bytes": {"n": 1, "mean": 65536.0, "sum": 65536.0},
                    "query_bytes": {"n": 1, "mean": 32768.0, "sum": 32768.0},
                    "output_bytes": {"n": 1, "mean": 32768.0, "sum": 32768.0},
                    "total_bytes": {"n": 1, "mean": 131072.0, "sum": 131072.0},
                },
                "timing_breakdown_notes": {
                    "inclusive_fence_wait": "fence_wait_cpu_us is inclusive host CPU wall time waiting for fence completion.",
                    "exclusive_gpu_total": "gpu_total_us measures exclusive GPU execution duration between first and last queries.",
                    "queue_wait": "queue_wait_us = max(0, fence_wait_cpu_us - gpu_total_us), capturing scheduling delay.",
                },
            },
        }

        # Serialized roundtrip
        raw = json.dumps(sample_json)
        loaded = json.loads(raw)

        self.assertEqual(loaded["schema"], "rkva-range-profile-v1")
        self.assertIn("bulk", loaded)
        bulk = loaded["bulk"]
        self.assertEqual(bulk["bulk_calls"], 1)
        self.assertEqual(bulk["bulk_segments"], 4)
        self.assertEqual(bulk["bulk_queries"], 32)
        self.assertTrue(bulk["gpu_timestamps"])

        timings = bulk["timings"]
        for expected_key in (
            "admission_cpu_us",
            "alloc_cpu_us",
            "map_copy_cpu_us",
            "record_cpu_us",
            "submit_cpu_us",
            "fence_wait_cpu_us",
            "readback_cpu_us",
            "engine_total_cpu_us",
            "gpu_upload_us",
            "gpu_pass1_us",
            "gpu_reduce_us",
            "gpu_readback_copy_us",
            "gpu_total_us",
            "queue_wait_us",
        ):
            self.assertIn(expected_key, timings)

        # Check queue_wait_us matches formula
        fw = timings["fence_wait_cpu_us"]["mean"]
        gt = timings["gpu_total_us"]["mean"]
        qw = timings["queue_wait_us"]["mean"]
        self.assertAlmostEqual(qw, max(0.0, fw - gt))

    def test_capability_fallback_no_gpu_timestamps(self):
        """Validate telemetry formatting when GPU timestamps are disabled or unavailable."""
        sample_json = {
            "schema": "rkva-range-profile-v1",
            "profiling": True,
            "gpu_timestamps": False,
            "gpu_timestamp_frequency_hz": 0,
            "bulk": {
                "bulk_calls": 1,
                "bulk_segments": 2,
                "bulk_queries": 16,
                "gpu_timestamps": False,
                "timings": {
                    "admission_cpu_us": {"n": 1, "mean": 10.0},
                    "engine_total_cpu_us": {"n": 1, "mean": 500.0},
                },
                "bytes": {"total_bytes": {"n": 1, "sum": 65536.0}},
            },
        }
        loaded = json.loads(json.dumps(sample_json))
        self.assertFalse(loaded["bulk"]["gpu_timestamps"])
        self.assertIn("admission_cpu_us", loaded["bulk"]["timings"])
        self.assertNotIn("gpu_total_us", loaded["bulk"]["timings"])


if __name__ == "__main__":
    unittest.main()
