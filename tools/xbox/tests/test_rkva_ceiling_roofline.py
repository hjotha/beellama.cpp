#!/usr/bin/env python3
"""Host tests for the ceiling suite's roofline model. No socket, no GPU.

Run: python3 -m unittest tools/xbox/tests/test_rkva_ceiling_roofline.py
"""

import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rkva_ceiling_roofline as roof  # noqa: E402


class Pass1ModelTest(unittest.TestCase):
    def test_flops_match_the_production_envelope(self):
        # The published envelope: 256 queries, 1 layer, ~28k KV rows -> ~177 GFLOP.
        flops = roof.pass1_flops(256, 28672)
        self.assertAlmostEqual(flops / 1e9, 180.4, delta=0.5)
        # The task's derived figure was ~177 GFLOP at ~28k rows; 27,904 rows is
        # the exact row count that lands there, which pins the formula.
        self.assertAlmostEqual(roof.pass1_flops(256, 27904) / 1e9, 175.6, delta=0.2)

    def test_flops_scale_with_queries_and_rows(self):
        base = roof.pass1_flops(1, 4096)
        self.assertEqual(roof.pass1_flops(8, 4096), 8 * base)
        self.assertEqual(roof.pass1_flops(1, 8192), 2 * base)
        self.assertEqual(roof.pass1_flops(0, 4096), 0)
        self.assertEqual(roof.pass1_flops(4, 0), 0)

    def test_kvarn_record_bytes_match_the_documented_31mb_layer(self):
        # 28k rows * ~1120 B/row(K+V) + stage/tail ~= 31 MB, the figure the
        # kernel analysis used.
        total = roof.kvarn4_kv_bytes(28672)
        self.assertAlmostEqual(total / (1 << 20), 31.2, delta=0.6)
        records = roof.kvarn4_record_bytes(28672)
        self.assertEqual(records, 224 * 4 * 2 * 8960 * 2)
        self.assertGreater(roof.kvarn4_kv_bytes(28672), records)

    def test_kvarn_partial_group_has_no_record(self):
        self.assertEqual(roof.kvarn4_record_bytes(127), 0)
        self.assertEqual(roof.kvarn4_record_bytes(128), 4 * 2 * 8960 * 2)
        self.assertGreater(roof.kvarn4_stage_tail_bytes(127), 0)
        self.assertEqual(roof.kvarn4_kv_bytes(0), 0)

    def test_rates_and_fractions(self):
        # 180.4 GFLOP in 65.3 ms is the published derived 2.76 TFLOP/s.
        r = roof.pass1_rates(256, 28672, 65.3e-3, bandwidth_gbs=400.0, fp32_tflops=6.0)
        self.assertAlmostEqual(r["tflops"], 2.763, delta=0.01)
        self.assertAlmostEqual(r["compute_fraction_pct"], 46.0, delta=0.3)
        self.assertGreater(r["bandwidth_fraction_pct"], 0.0)
        self.assertIn(r["bound"], ("compute", "bandwidth"))
        # 31 MB unique over 65 ms is ~0.5 GB/s, so the point is compute bound.
        self.assertEqual(r["bound"], "compute")
        self.assertLess(r["gbs_unique"], 1.0)
        # The issued-upper bound assumes no reuse at all (nq x unique).
        self.assertAlmostEqual(r["issued_bytes_upper_bound"],
                               256 * (r["unique_bytes"] - roof.qo_bytes(256)) +
                               roof.qo_bytes(256), delta=1.0)

    def test_zero_seconds_never_divides(self):
        r = roof.pass1_rates(1, 1, 0.0, 0.0, 0.0)
        self.assertEqual(r["tflops"], 0.0)
        self.assertEqual(r["gbs_unique"], 0.0)
        self.assertEqual(r["speedup_cap_vs_measured_ceiling"], 0.0)


class ClockAndOccupancyTest(unittest.TestCase):
    def test_implied_clock(self):
        # 4 clocks per dependent FMA at 0.7 ns per FMA -> 5.71 GHz implied.
        self.assertAlmostEqual(roof.implied_clock_mhz(0.7e-9), 5714.2857, places=2)
        self.assertEqual(roof.implied_clock_mhz(0.0), 0.0)

    def test_fit_concurrency_recovers_the_plateau(self):
        # 104 concurrent groups, 0.5 ms per group: the ceil() staircase.
        groups = [13, 26, 52, 78, 104, 130, 156, 208, 260]
        seconds = [math.ceil(n / 104) * 0.5e-3 for n in groups]
        fit = roof.fit_concurrency(groups, seconds)
        self.assertEqual(fit["concurrency"], 104.0)
        self.assertAlmostEqual(fit["group_seconds"], 0.5e-3, places=9)
        self.assertEqual(fit["confident"], 1.0)

    def test_fit_concurrency_requires_enough_points(self):
        fit = roof.fit_concurrency([52, 104], [1e-3, 1e-3])
        self.assertEqual(fit["concurrency"], 0.0)
        self.assertEqual(fit["confident"], 0.0)

    def test_cu_estimate_needs_two_agreeing_plateaus(self):
        est = roof.cu_estimate(104.0, 208.0)
        self.assertEqual(est["dual_cu_from_32kib"], 52.0)
        self.assertEqual(est["dual_cu_from_16kib"], 52.0)
        self.assertEqual(est["dual_cu"], 52.0)
        self.assertEqual(est["agree"], 1.0)
        # Disagreement withholds the CU count instead of averaging nonsense.
        bad = roof.cu_estimate(104.0, 100.0)
        self.assertEqual(bad["agree"], 0.0)
        self.assertEqual(bad["dual_cu"], 0.0)


def probe_document(checksum_ok=True, run=True):
    def row(probe, **kw):
        base = {"suite": "stream", "probe": probe, "variant": "268435456", "bytes": 268435456,
                "flops": 0, "gpu_seconds": 0.001, "gpu_seconds_median": 0.0011,
                "cpu_seconds": 0.002, "readback_seconds": 0.0, "gbs": 0.0, "tflops": 0.0,
                "gops": 0.0, "groups": 2048, "threads": 256, "runs": 5, "iters": 0,
                "gpu_valid": True, "checksum_checked": False, "checksum_ok": False,
                "error": ""}
        base.update(kw)
        return base

    rows = [
        row("read", gpu_seconds=0.0010, checksum_checked=True, checksum_ok=checksum_ok),
        row("write", gpu_seconds=0.0014),
        row("copy", gpu_seconds=0.0022, checksum_checked=True, checksum_ok=checksum_ok),
        row("fma_f32", suite="alu", gpu_seconds=0.0008, flops=4.0e9, tflops=5.0),
        row("fma_f16", suite="alu", gpu_seconds=0.0008, flops=8.0e9, tflops=10.0),
        row("dot2_f16", suite="alu", gpu_seconds=0.0008, flops=8.0e9, tflops=10.0),
        row("dot4_i8", suite="alu", gpu_seconds=0.0008, flops=16.0e9, gops=20.0),
        row("empty_dispatch_gpu", suite="latency", variant="p50", gpu_seconds=4e-6),
        row("empty_dispatch_batch", suite="latency", gpu_seconds=3e-6),
        row("empty_submit_fence_cpu", suite="latency", variant="p50", cpu_seconds=9e-5),
        row("empty_readback_cpu", suite="latency", variant="p50", cpu_seconds=2e-5),
    ]
    for probe, groups in (("cu32", [13, 26, 52, 78, 104, 130, 156, 208]),
                          ("cu16", [13, 26, 52, 78, 104, 130, 156, 208])):
        for n in groups:
            conc = 104 if probe == "cu32" else 208
            rows.append(row(probe, suite="occupancy", groups=n, threads=64,
                            gpu_seconds=math.ceil(n / conc) * 0.5e-3))
    return {
        "schema": "xllama-gpu-ceiling-v1",
        "host": "xbox-series-x",
        "recorded_utc": "2026-10-26T00:00:00Z",
        "d3d12_ran": run,
        "error": "" if run else "d3d12 unavailable on this platform",
        "options": {"summary": "stream_bytes=268435456", "stream_bytes": 268435456,
                    "run_big_stream": False, "big_stream_bytes": 268435456,
                    "iterations": 5},
        "nominal": dict(roof.NOMINAL),
        "device": {"adapter": "Xbox Series X", "local_budget_bytes": 4348654388,
                   "timestamp_frequency": 10000000, "timestamp_us_per_tick": 0.1,
                   "wave_lane_min": 64, "wave_lane_max": 64, "max_lds_bytes": 32768},
        "derived": {"implied_clock_mhz": 1400.0},
        "pso_errors": [],
        "rows": rows,
    }


class ProbeParsingTest(unittest.TestCase):
    def test_valid_document_indexes_every_probe(self):
        parsed = roof.parse_gpupeak_json(probe_document())
        self.assertIn("read", parsed["by_probe"])
        self.assertIn("cu16", parsed["by_probe"])
        self.assertEqual(parsed["device"]["timestamp_frequency"], 10000000)

    def test_missing_checksum_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            roof.parse_gpupeak_json(probe_document(checksum_ok=False))

    def test_unavailable_probe_is_rejected(self):
        with self.assertRaises(ValueError):
            roof.parse_gpupeak_json(probe_document(run=False))

    def test_unknown_schema_is_rejected(self):
        doc = probe_document()
        doc["schema"] = "other"
        with self.assertRaises(ValueError):
            roof.parse_gpupeak_json(doc)

    def test_stream_rate_uses_the_traffic_multiplier(self):
        rows = probe_document()["rows"]
        read = roof.best(rows, "read")
        copy = roof.best(rows, "copy")
        self.assertAlmostEqual(roof.stream_rate_gbs(read), 268435456 / 1e9 / 0.0010)
        self.assertAlmostEqual(roof.stream_rate_gbs(copy, 2.0), 2 * 268435456 / 1e9 / 0.0022)


class ResultAssemblyTest(unittest.TestCase):
    def test_result_carries_measured_ceiling_and_roofline(self):
        sweep = {"points": [
            {"n_queries": 1, "rows": 28672, "gpu_seconds": 0.000144},
            {"n_queries": 8, "rows": 28672, "gpu_seconds": 0.002725},
            {"n_queries": 256, "rows": 28672, "gpu_seconds": 0.0653},
        ]}
        result = roof.build_result(probe_document(), sweep)
        self.assertEqual(result["schema"], "xbox-ceiling-result-v1")
        ceiling = result["ceiling_measured"]
        self.assertAlmostEqual(ceiling["bandwidth_gbs"], 268.435456, places=3)
        self.assertAlmostEqual(ceiling["fp32_tflops"], 5.0, places=6)
        # Occupancy plateaus 104 / 208 -> 52 dual-CUs, the nominal count.
        self.assertEqual(result["compute_units"]["dual_cu"], 52.0)
        self.assertEqual(result["compute_units"]["agree"], 1.0)
        # Each pass1 point is placed against the *measured* ceilings.
        self.assertEqual(len(result["pass1_points"]), 3)
        point = result["pass1_points"][2]
        self.assertAlmostEqual(point["tflops"], 2.763, delta=0.01)
        self.assertGreater(point["compute_fraction_pct"], 50.0)
        self.assertEqual(point["bound"], "compute")
        self.assertAlmostEqual(result["dispatch_latency"]["empty_dispatch_gpu_us"], 4.0)
        self.assertAlmostEqual(result["dispatch_latency"]["empty_submit_fence_cpu_us"], 90.0)
        json.dumps(result)  # must be serialisable as-is

    def test_result_without_occupancy_withholds_the_cu_count(self):
        doc = probe_document()
        doc["rows"] = [r for r in doc["rows"] if not str(r["probe"]).startswith("cu")]
        result = roof.build_result(doc, {"points": []})
        self.assertEqual(result["compute_units"]["dual_cu"], 0.0)
        self.assertEqual(result["compute_units"]["agree"], 0.0)


if __name__ == "__main__":
    unittest.main()
