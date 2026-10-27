#!/usr/bin/env python3
"""Host tests for tools/xbox/scripts/rkva-latency-audit.py.

CPU only: no socket, no GPU, no subprocess, no LAN. Socket and process probing
are exercised through injected fakes so the suite can never contact hardware.

Run: python3 -m unittest tools/xbox/tests/test_rkva_latency_audit.py
"""

import ast
import contextlib
import importlib.util
import io
import json
import math
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "scripts", "rkva-latency-audit.py")

spec = importlib.util.spec_from_file_location("rkva_latency_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def event(phase, begin, end, layer, cycle, graph_run=7, first_pos=130560,
          nq=1, gpu_us=0.0, session=1, model_layer=3):
    return {
        "schema": "rkva-host-trace-v1",
        "phase": phase,
        "node": "",
        "begin_us": begin,
        "end_us": end,
        "request": 0,
        "graph_run": graph_run,
        "graph_uid": graph_run * 100 + layer,
        "cycle": cycle,
        "ticket": 0,
        "epoch": 0,
        "session": session,
        "layer": layer,
        "nq": nq,
        "first_pos": first_pos,
        "last_pos": first_pos,
        "model_layer": model_layer,
        "bytes": 0,
        "value": 0,
        "flags": 0,
        "gpu_us": gpu_us,
    }


def golden_events(tokens=2, layers=2, unique_cycles=True):
    """Overlapping parent/child spans with a measurable inter-layer gap."""
    events = []
    base = 1_000_000
    cycle = 1
    for token in range(tokens):
        token_base = base + token * 20_000
        for layer in range(layers):
            start = token_base + layer * 6_000
            run = 7 + token
            token_cycle = cycle if unique_cycles else layer + 1
            events += [
                event("job_finish", start, start + 4_000, layer, token_cycle, run),
                event("rpc_total", start + 10, start + 1_490, layer, token_cycle, run,
                      gpu_us=800.0),
                event("rpc_send", start + 10, start + 50, layer, token_cycle, run),
                event("rpc_header_wait", start + 50, start + 1_000, layer, token_cycle, run),
                event("rpc_body_recv", start + 1_000, start + 1_100, layer, token_cycle, run),
                event("out_decode", start + 1_520, start + 2_900, layer, token_cycle, run),
                event("cuda_small_unpack_kernel_gpu", start + 1_600, start + 2_000, layer,
                      token_cycle, run, gpu_us=380.0),
            ]
            cycle += 1
    return events


def write_trace(path, events):
    with open(path, "w", encoding="utf-8") as handle:
        for item in events:
            handle.write(json.dumps(item) + "\n")


class TraceCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.trace = os.path.join(self.tmp.name, "trace.jsonl")
        write_trace(self.trace, golden_events())

    def analyze(self, **kwargs):
        options = dict(min_position=130560, max_position=160000)
        options.update(kwargs)
        report = audit.analyze_trace(self.trace, **options)
        report.pop("_analysis", None)
        return report


class TraceAttributionTest(TraceCase):
    def test_dropped_trace_cannot_report_complete_window(self):
        write_trace(self.trace, golden_events() + [
            {"schema": "rkva-host-trace-status-v1", "events_written": 100, "dropped": 7},
            {"schema": "rkva-host-trace-status-v1", "events_written": 101, "dropped": 7}])
        report = self.analyze()
        self.assertEqual(report["scope"], "partial")
        self.assertEqual(report["capture"]["dropped_events"], 7)  # cumulative, not 14
        self.assertEqual(report["counts"]["tokens"], 2)

    def test_metadata_after_truncated_prefill_is_not_a_decode_token(self):
        status = event("cuda_local_timing_status", 100, 100, 0, 0)
        status["value"] = 0
        write_trace(self.trace, [status,
            {"schema": "rkva-host-trace-status-v1", "events_written": 262145,
             "dropped": 201335}])
        report = self.analyze()
        self.assertEqual(report["scope"], "unknown")
        self.assertEqual(report["counts"]["tokens"], 0)
        self.assertEqual(report["counts"]["decode_events"], 0)
        self.assertNotIn("cuda_local_timing_status", report["phases"])

    def test_complete_footer_and_zero_local_status_preserve_spans(self):
        metadata = event("cuda_local_timing_status", 100, 100, 0, 0)
        metadata["value"] = 0
        write_trace(self.trace, golden_events() + [metadata,
            {"schema": "rkva-host-trace-status-v1", "events_written": 100, "dropped": 0}])
        report = self.analyze()
        self.assertEqual(report["scope"], "ok")
        self.assertEqual(report["counts"]["tokens"], 2)
        self.assertEqual(report["capture"]["dropped_events"], 0)

    def test_invalid_footer_is_refused(self):
        for value in (-1, True, 1.5, "1", None):
            with self.subTest(value=value):
                write_trace(self.trace, golden_events() + [
                    {"schema": "rkva-host-trace-status-v1", "events_written": 100,
                     "dropped": value}])
                with self.assertRaises(audit.TraceFormatError):
                    self.analyze()

    def test_cuda_timing_loss_is_metadata_and_partial(self):
        for phase in ("cuda_local_timing_status", "cuda_local_timing_unresolved",
                      "cuda_local_profile_error"):
            with self.subTest(phase=phase):
                status = event(phase, 100, 100, 0, 0)
                status["value"] = 1
                write_trace(self.trace, golden_events() + [status])
                report = self.analyze()
                self.assertEqual(report["scope"], "partial")
                self.assertEqual(report["counts"]["tokens"], 2)

    def test_counts_tokens_layers_and_phases(self):
        report = self.analyze()
        self.assertEqual(report["scope"], "ok")
        self.assertEqual(report["counts"]["tokens"], 2)
        self.assertEqual(report["counts"]["layer_groups"], 4)
        self.assertEqual(report["counts"]["malformed_lines"], 0)
        self.assertEqual(report["phases"]["job_finish"]["count"], 4)
        self.assertEqual(report["phases"]["rpc_total"]["count"], 4)
        self.assertEqual(report["per_token"]["layers_per_token"], {"min": 2, "max": 2})

    def test_parent_child_overlap_is_reported_not_summed(self):
        report = self.analyze()
        overlap = report["parent_child"]["job_finish"]
        self.assertIn("rpc_total", overlap)
        self.assertIn("out_decode", overlap)
        self.assertGreater(overlap["rpc_total"]["overlap_us"], 0.0)
        self.assertGreater(overlap["rpc_total"]["overlap_ratio"], 0.0)
        self.assertLessEqual(overlap["rpc_total"]["overlap_ratio"], 1.0)
        self.assertTrue(report["phases"]["job_finish"]["container"])
        self.assertTrue(report["phases"]["rpc_total"]["container"])
        # A container's own duration is not added to its children's.
        token = report["per_token"]
        self.assertLess(token["attributed_ms"]["p50"], token["envelope_ms"]["p50"] + 1e-9)
        self.assertGreater(token["rpc_total_ms_container"]["p50"], 0.0)
        rr = report["phases"]["rpc_total"]
        # A container is not a wait: it publishes a residual, never "exposed".
        self.assertIsNone(rr["exposed_total_us"])
        self.assertIsNotNone(rr["residual_total_us"])
        self.assertAlmostEqual(rr["overlapped_total_us"],
                               rr["total"] - rr["residual_total_us"])
        self.assertGreater(rr["overlapped_total_us"], 0.0)
        self.assertGreater(rr["residual_total_us"], 0.0)
        self.assertNotIn("rpc_total", report["exposed_wait"]["wait_phases"])

    def test_parent_child_overlap_is_per_child_phase(self):
        # Exact reproduction: parent 0..100 us, send 10..20 us, wait 30..60 us.
        # Each pair must report its own clipped union (10 and 30), never the
        # sibling union (40) for both.
        events = [
            event("job_finish", 0, 100, 0, 1),
            event("rpc_send", 10, 20, 0, 1),
            event("rpc_header_wait", 30, 60, 0, 1),
        ]
        write_trace(self.trace, events)
        report = self.analyze()
        pairs = report["parent_child"]["job_finish"]
        self.assertEqual(pairs["rpc_send"]["overlap_us"], 10.0)
        self.assertEqual(pairs["rpc_header_wait"]["overlap_us"], 30.0)
        self.assertAlmostEqual(pairs["rpc_send"]["overlap_ratio"], 0.10)
        self.assertAlmostEqual(pairs["rpc_header_wait"]["overlap_ratio"], 0.30)
        # The parent residual is the union of all direct children: 100 - 40.
        self.assertEqual(report["phases"]["job_finish"]["residual_total_us"], 60.0)

    def test_parent_child_same_phase_children_are_unioned(self):
        events = [
            event("job_finish", 0, 100, 0, 1),
            event("rpc_send", 10, 20, 0, 1),
            event("rpc_send", 15, 25, 0, 1),
        ]
        write_trace(self.trace, events)
        report = self.analyze()
        pair = report["parent_child"]["job_finish"]["rpc_send"]
        self.assertEqual(pair["count"], 2)
        self.assertEqual(pair["overlap_us"], 15.0)  # union 10..25
        self.assertEqual(report["phases"]["job_finish"]["residual_total_us"], 85.0)

    def test_origin_metadata_separates_same_position_tokens(self):
        # Exact reproduction: processes A/B, same session 0, graph_run 1,
        # layer 0, position 130560, same cycle; A at 0..100, B at 1000..1100.
        # Without origin in the key this merges into one token with a 900 us gap.
        first = event("job_finish", 0, 100, 0, 7, graph_run=1)
        second = event("job_finish", 1000, 1100, 0, 7, graph_run=1)
        first["pid"] = 111
        second["pid"] = 222
        write_trace(self.trace, [first, second])
        report = self.analyze()
        self.assertEqual(report["counts"]["tokens"], 2)
        self.assertEqual(report["counts"]["layer_groups"], 2)
        self.assertEqual(report["per_token"]["origin"]["distinct"], 2)
        self.assertEqual(report["per_token"]["origin"]["positions_with_multiple_origins"], 1)
        self.assertEqual(report["per_token"]["unattributed_gap_ms"]["total"], 0.0)
        self.assertEqual(report["per_token"]["envelope_ms"]["p50"], 0.1)
        self.assertNotEqual(report["correlation"]["scope"], "failed")

    def test_missing_origin_metadata_is_labelled_unknown(self):
        first = event("job_finish", 0, 100, 0, 7, graph_run=1)
        second = event("job_finish", 1000, 1100, 0, 7, graph_run=1)
        for item in (first, second):
            item.pop("session", None)
        write_trace(self.trace, [first, second])
        report = self.analyze()
        self.assertFalse(report["per_token"]["origin"]["qualified"])
        self.assertEqual(report["correlation"]["scope"], "unknown")

    def test_exposed_wait_uses_span_ranges_not_rpc_total(self):
        report = self.analyze()
        wait = report["exposed_wait"]
        self.assertIn("rpc_header_wait", wait["wait_phases"])
        self.assertNotIn("rpc_total", wait["wait_phases"])
        header = wait["wait_phases"]["rpc_header_wait"]
        self.assertTrue(header["wait_span"])
        self.assertAlmostEqual(header["exposed_total_us"], header["total"])
        self.assertGreaterEqual(wait["unattributed_gap_ms_per_token"]["p50"], 0.0)

    def test_gap_is_the_uncovered_wall_time(self):
        report = self.analyze()
        token = report["per_token"]
        for key in ("attributed_ms", "unattributed_gap_ms"):
            self.assertEqual(token[key]["count"], 2)
        self.assertAlmostEqual(token["envelope_ms"]["total"],
                               token["attributed_ms"]["total"]
                               + token["unattributed_gap_ms"]["total"])

    def test_window_filter_and_context_cap(self):
        filtered = self.analyze(min_position=130561)
        self.assertEqual(filtered["counts"]["tokens"], 0)
        self.assertEqual(filtered["scope"], "unknown")
        with self.assertRaises(audit.TraceFormatError):
            self.analyze(max_position=160001)
        with self.assertRaises(audit.TraceFormatError):
            self.analyze(context_cap=160001)
        with self.assertRaises(audit.TraceFormatError):
            self.analyze(min_position=200, max_position=100)
        self.assertEqual(audit.CONTEXT_CAP, 160000)

    def test_duplicate_cycle_ids_in_independent_windows_are_unknown(self):
        write_trace(self.trace, golden_events(unique_cycles=False))
        report = self.analyze()
        correlation = report["correlation"]
        # The fixture tokens are 20 ms apart: reuse is not a transfer failure.
        self.assertEqual(correlation["scope"], "unknown")
        self.assertGreater(correlation["duplicate_ids"], 0)
        self.assertGreater(correlation["reused_in_independent_windows"], 0)
        self.assertEqual(correlation["ambiguous_groups"], 0)
        self.assertTrue(any("independent windows" in line
                            for line in correlation["reasons"]))
        self.assertEqual(report["counts"]["duplicate_cycle_events"], 0)

    def test_duplicate_cycle_ids_in_overlapping_windows_fail(self):
        events = golden_events(tokens=1, layers=1)
        # One extra group reusing the same session/cycle while time overlaps.
        events += [
            event("job_finish", 1_000_500, 1_004_000, 0, 1, graph_run=99),
            event("rpc_total", 1_000_510, 1_001_990, 0, 1, graph_run=99),
        ]
        write_trace(self.trace, events)
        report = self.analyze()
        correlation = report["correlation"]
        self.assertEqual(correlation["scope"], "failed")
        self.assertGreater(correlation["ambiguous_groups"], 0)
        self.assertTrue(any("overlapping windows" in line
                            for line in correlation["reasons"]))

    def test_distinct_origins_with_the_same_cycle_are_ok(self):
        events = golden_events(tokens=2, layers=1, unique_cycles=False)
        for item in events:
            item["session"] = 1 if item["graph_run"] == 7 else 2
        write_trace(self.trace, events)
        report = self.analyze()
        correlation = report["correlation"]
        self.assertEqual(correlation["scope"], "ok")
        self.assertEqual(correlation["duplicate_ids"], 0)
        self.assertTrue(correlation["origin"]["available"])

    def test_explicit_origin_metadata_is_part_of_the_composite_key(self):
        events = golden_events(tokens=2, layers=1, unique_cycles=False)
        for item in events:
            item["origin"] = "arm-a" if item["graph_run"] == 7 else "arm-b"
        write_trace(self.trace, events)
        report = self.analyze()
        correlation = report["correlation"]
        self.assertEqual(correlation["scope"], "ok")
        self.assertIn("origin", correlation["origin"]["fields_seen"])
        self.assertEqual(correlation["duplicate_ids"], 0)

    def test_origin_conflict_splits_groups_instead_of_merging(self):
        events = golden_events(tokens=1, layers=1)
        events[0]["origin"] = "arm-a"  # only one span of this layer is labelled
        write_trace(self.trace, events)
        report = self.analyze()
        # The labelled span forms its own qualified group; nothing is merged.
        self.assertEqual(report["counts"]["layer_groups"], 2)
        self.assertEqual(report["counts"]["tokens"], 2)
        self.assertEqual(report["per_token"]["origin"]["distinct"], 2)

    def test_missing_origin_context_is_unknown_not_ok(self):
        events = golden_events(tokens=1, layers=1)
        for item in events:
            item.pop("session", None)
        write_trace(self.trace, events)
        report = self.analyze()
        correlation = report["correlation"]
        self.assertEqual(correlation["scope"], "unknown")
        self.assertFalse(correlation["origin"]["available"])
        self.assertTrue(any("no process/session origin" in line
                            for line in correlation["reasons"]))

    def test_absent_cycle_ids_are_unknown(self):
        events = golden_events()
        for item in events:
            item["cycle"] = 0
        write_trace(self.trace, events)
        report = self.analyze()
        correlation = report["correlation"]
        self.assertEqual(correlation["scope"], "unknown")
        self.assertEqual(correlation["groups_without_correlation"], 4)
        self.assertEqual(correlation["groups_with_correlation"], 0)

    def test_valid_cycle_ids_are_matched_one_to_one(self):
        report = self.analyze()
        correlation = report["correlation"]
        self.assertEqual(correlation["scope"], "ok")
        self.assertEqual(correlation["groups_with_correlation"], 4)
        self.assertEqual(correlation["duplicate_ids"], 0)
        self.assertEqual(correlation["reasons"], [])

    def test_identity_conflict_is_partial_not_averaged(self):
        events = golden_events(tokens=1, layers=1)
        events.append(event("rpc_send", 2_000_000, 2_000_050, 0, 99, 7,
                            first_pos=130561, nq=1))
        write_trace(self.trace, events)
        report = self.analyze()
        self.assertEqual(report["counts"]["identity_conflicts"], 1)
        self.assertEqual(report["scope"], "partial")
        self.assertGreater(report["unknown"]["events_without_identity"], 0)

    def test_negative_or_non_finite_timings_fail(self):
        events = golden_events(tokens=1, layers=1)
        events.append(event("rpc_send", 5_000, 4_000, 0, 50))
        write_trace(self.trace, events)
        with self.assertRaises(audit.TraceFormatError):
            self.analyze()
        events = golden_events(tokens=1, layers=1)
        events[0]["gpu_us"] = float("nan")
        write_trace(self.trace, events)
        with self.assertRaises(audit.TraceFormatError):
            self.analyze()

    def test_non_decode_events_are_ignored(self):
        events = golden_events(tokens=1, layers=1)
        events.append(event("rpc_send", 1, 2, 0, 60, nq=8))
        write_trace(self.trace, events)
        report = self.analyze()
        self.assertEqual(report["counts"]["decode_events"], 7)
        self.assertEqual(report["counts"]["tokens"], 1)


class SampleStatsTest(unittest.TestCase):
    def test_percentiles_use_nearest_rank(self):
        stats = audit.summarize(range(1, 101))
        self.assertEqual(stats["count"], 100)
        self.assertEqual(stats["p50"], 50.0)
        self.assertEqual(stats["p95"], 95.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 100.0)
        self.assertEqual(stats["total"], 5050.0)

    def test_single_and_empty_samples(self):
        one = audit.summarize([5.0])
        self.assertEqual((one["p50"], one["p95"]), (5.0, 5.0))
        empty = audit.summarize([])
        self.assertEqual(empty["count"], 0)
        self.assertIsNone(empty["p50"])
        with self.assertRaises(ValueError):
            audit.summarize([1.0, math.inf])

    def test_interval_union_does_not_double_count(self):
        merged = audit.merge_intervals([(0, 10), (5, 15), (20, 30)])
        self.assertEqual(merged, [(0.0, 15.0), (20.0, 30.0)])
        self.assertEqual(audit.union_us([(0, 10), (5, 15)]), 15.0)
        self.assertEqual(audit.union_within((0, 10), [(5, 20)]), 5.0)


class ComparisonGateTest(unittest.TestCase):
    def samples(self, values, window=(130560, 160000), op="rpc_total"):
        return {"source": "x", "op": op,
                "window": {"min_position": window[0], "max_position": window[1]},
                "samples_us": values}

    def test_missing_window_or_op_is_not_proven_equality(self):
        no_window = {"source": "x", "op": "rpc_total", "samples_us": [10.0]}
        no_op = {"source": "x", "samples_us": [10.0],
                 "window": {"min_position": 130560, "max_position": 160000}}
        both_missing = {"source": "x", "samples_us": [10.0]}
        for left, right in ((no_window, no_window), (no_op, no_op),
                            (both_missing, both_missing),
                            (no_window, self.samples([10.0])),
                            (no_op, self.samples([10.0]))):
            result = audit.compare_samples(left, right)
            self.assertFalse(result["comparable"], (left, right))
            self.assertEqual(result["scope"], "unknown")
            self.assertIn("lacks an explicit", result["reason"])

        invalid_window = {"source": "x", "op": "rpc_total", "samples_us": [10.0],
                          "window": {"min_position": 160000, "max_position": 130560}}
        result = audit.compare_samples(invalid_window, self.samples([10.0]))
        self.assertFalse(result["comparable"])
        self.assertEqual(result["scope"], "failed")
        self.assertIn("invalid", result["reason"])

        invalid_op = {"source": "x", "op": "  ", "samples_us": [10.0],
                      "window": {"min_position": 130560, "max_position": 160000}}
        result = audit.compare_samples(invalid_op, self.samples([10.0]))
        self.assertFalse(result["comparable"])
        self.assertEqual(result["scope"], "unknown")

    def test_congruent_samples_compare(self):
        result = audit.compare_samples(self.samples([10.0, 12.0, 14.0]),
                                       self.samples([8.0, 9.0, 11.0]))
        self.assertTrue(result["comparable"])
        self.assertEqual(result["scope"], "ok")
        self.assertEqual(result["delta_us"]["count"], 3)

    def test_sample_count_mismatch_is_rejected(self):
        result = audit.compare_samples(self.samples([10.0, 12.0]),
                                       self.samples([8.0]))
        self.assertFalse(result["comparable"])
        self.assertIn("sample count mismatch", result["reason"])
        self.assertEqual(result["scope"], "failed")

    def test_window_mismatch_is_rejected(self):
        result = audit.compare_samples(self.samples([10.0], (130560, 160000)),
                                       self.samples([9.0], (130560, 131072)))
        self.assertFalse(result["comparable"])
        self.assertIn("window mismatch", result["reason"])
        self.assertEqual(result["scope"], "failed")

    def test_op_mismatch_and_missing_samples_are_rejected(self):
        result = audit.compare_samples(self.samples([10.0], op="rpc_total"),
                                       self.samples([9.0], op="out_decode"))
        self.assertFalse(result["comparable"])
        self.assertIn("op mismatch", result["reason"])
        self.assertEqual(result["scope"], "failed")
        result = audit.compare_samples(self.samples([]), self.samples([9.0]))
        self.assertFalse(result["comparable"])
        self.assertIn("no samples", result["reason"])
        self.assertEqual(result["scope"], "unknown")

    def test_non_finite_comparison_input_is_rejected(self):
        result = audit.compare_samples(self.samples([10.0]),
                                       self.samples([float("nan")]))
        self.assertFalse(result["comparable"])
        self.assertEqual(result["scope"], "failed")


class ProcessGuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def add_proc(self, pid, argv, comm):
        directory = os.path.join(self.root, str(pid))
        os.makedirs(directory)
        with open(os.path.join(directory, "cmdline"), "wb") as handle:
            handle.write(b"\0".join(arg.encode() for arg in argv) + b"\0")
        with open(os.path.join(directory, "comm"), "w", encoding="utf-8") as handle:
            handle.write(comm + "\n")

    def test_busy_workloads_are_detected_and_own_cli_is_not(self):
        self.add_proc(111, ["llama-server", "--port", "8080"], "llama-server")
        self.add_proc(222, ["python3", os.path.abspath(SCRIPT), "--execute",
                            "--host", "192.168.1.200:50054", "--empty-rtt"], "python3")
        self.add_proc(333, ["python3", "tools/xbox/scripts/analyze-rkva-decode-trace.py",
                            "--trace", "x.jsonl"], "python3")
        self.add_proc(444, ["bash", "-c", "llama-server --port 8080"], "bash")
        self.add_proc(555, ["python3", "tools/xbox/scripts/bench-rkva-range.py"], "python3")
        self.add_proc(666, ["llama-bench", "-m", "model.gguf"], "llama-bench")
        busy = audit.find_busy_processes(proc_root=self.root, own_pid=222)
        names = sorted(entry["name"] for entry in busy)
        pids = sorted(entry["pid"] for entry in busy)
        self.assertEqual(pids, [111, 555, 666])
        self.assertNotIn("rkva-latency-audit.py", names)
        self.assertNotIn("analyze-rkva-decode-trace.py", names)

    def test_own_pid_and_ancestors_are_excluded(self):
        self.add_proc(700, ["llama-server"], "llama-server")
        busy = audit.find_busy_processes(proc_root=self.root, own_pid=700)
        self.assertEqual(busy, [])

    def test_name_matching_is_exact_not_substring(self):
        self.assertFalse(audit.busy_process_name("rkva-latency-audit.py"))
        self.assertFalse(audit.busy_process_name("analyze-rkva-decode-trace.py"))
        self.assertFalse(audit.busy_process_name("python3"))
        self.assertTrue(audit.busy_process_name("llama-server"))
        self.assertTrue(audit.busy_process_name("llama-server.exe"))
        self.assertTrue(audit.busy_process_name("run-rkva-decode-host-ab.py"))
        self.assertTrue(audit.busy_process_name("rkva_pass1_sweep.py"))


class FakeSocket:
    """Deterministic socket double; records every frame and closes exactly once."""

    def __init__(self, timeout_at=None, mutate=None, payload=b"", echo_cycle=None):
        self.sent = []
        self.closed = 0
        self.timeout_at = timeout_at
        self.mutate = mutate or (lambda frame: frame)
        self.payload = payload
        self.echo_cycle = echo_cycle
        self.recv_calls = 0
        self._pending = b""
        self._samples = 0

    def settimeout(self, _value):
        pass

    def sendall(self, data):
        self.sent.append(bytes(data))
        self._samples += 1
        if self.timeout_at is not None and self._samples == self.timeout_at:
            raise socket.timeout("simulated timeout")
        fields = audit.struct.unpack(audit.REQUEST_FMT, data)
        cycle = fields[3] if self.echo_cycle is None else self.echo_cycle
        body = self.payload if fields[2] == audit.RKVA_GET_STATS else b""
        header = audit.struct.pack(
            audit.RESPONSE_FMT, audit.RKVA_MAGIC, audit.RKVA_VERSION, fields[2],
            cycle, 0, len(body), 100, 200, 300, 400)
        self._pending = self.mutate(header) + body

    def recv(self, size):
        self.recv_calls += 1
        chunk, self._pending = self._pending[:size], self._pending[size:]
        return chunk

    def close(self):
        self.closed += 1


class ProbeTest(unittest.TestCase):
    def probe(self, sock, **kwargs):
        options = dict(reps=3, warmup=1, timeout=0.5, connector=lambda *_: sock)
        options.update(kwargs)
        with mock.patch.object(audit, "find_busy_processes", return_value=[]):
            return audit.probe_empty_rtt("192.0.2.10", 50054, **options)

    def test_ping_only_and_bounded_samples(self):
        sock = FakeSocket()
        report = self.probe(sock)
        self.assertEqual(report["mode"], "rtt")
        self.assertEqual(report["samples_us"]["count"], 3)
        self.assertEqual(report["op"]["id"], audit.RKVA_PING)
        self.assertFalse(report["op"]["gpu_touched"])
        self.assertFalse(report["op"]["hello"])
        self.assertEqual(len(sock.sent), 4)  # 1 warmup + 3 measured
        for frame in sock.sent:
            fields = audit.struct.unpack(audit.REQUEST_FMT, frame)
            self.assertEqual(fields[2], audit.RKVA_PING)
            self.assertEqual(fields[4], audit.RKVA_SESSION_NONE)
            self.assertEqual(fields[7], 0)
        self.assertEqual(sock.closed, 1)
        self.assertEqual(report["bytes"]["request"], audit.REQUEST_BYTES * 4)
        self.assertEqual(report["bytes"]["response"], audit.RESPONSE_BYTES * 4)
        self.assertGreater(report["samples_us"]["p50"], 0.0)
        self.assertTrue(report["response_echo"]["cycle_ids_unique"])
        self.assertEqual(report["response_echo"]["cycle_id_count"], 4)

    def test_identical_durations_with_unique_ids_are_valid(self):
        sock = FakeSocket()
        report = self.probe(sock, clock=lambda: 42.0)
        self.assertEqual(report["samples_us"]["count"], 3)
        self.assertEqual(len(set(report["samples"])), 1)
        self.assertTrue(report["response_echo"]["cycle_ids_unique"])
        self.assertEqual(report["scope"], "ok")

    def test_repeated_cycle_ids_fail_even_with_distinct_durations(self):
        sock = FakeSocket(echo_cycle=7)
        with self.assertRaises(audit.ProbeError) as ctx:
            self.probe(sock, reps=3, warmup=0)
        self.assertIn("cycle", str(ctx.exception))
        self.assertEqual(sock.closed, 1)

    def test_cycle_sequence_requires_exact_send_order(self):
        self.assertTrue(audit.verify_cycle_sequence([1, 2, 3], [1, 2, 3]))
        with self.assertRaises(audit.ProbeError):
            audit.verify_cycle_sequence([1, 2, 3], [1, 3, 2])
        with self.assertRaises(audit.ProbeError):
            audit.verify_cycle_sequence([1, 1], [1, 1])
        with self.assertRaises(audit.ProbeError):
            audit.verify_cycle_sequence([1, 2], [1])

    def test_header_payload_bytes_is_checked_before_any_body_read(self):
        def huge_payload(frame):
            fields = list(audit.struct.unpack(audit.RESPONSE_FMT, frame))
            fields[5] = 0xFFFFFFFF  # UINT32_MAX declared, no body follows
            return audit.struct.pack(audit.RESPONSE_FMT, *fields)

        sock = FakeSocket(mutate=huge_payload)
        with self.assertRaises(audit.ProbeError) as ctx:
            self.probe(sock, reps=1, warmup=0)
        self.assertIn("payload", str(ctx.exception))
        self.assertEqual(sock.recv_calls, 1)  # header only, no body recv
        self.assertEqual(sock.closed, 1)

    def test_header_mutations_fail_closed(self):
        mutations = {
            "magic": (0, audit.RKVA_MAGIC ^ 1),
            "version": (1, 99),
            "op": (2, audit.RKVA_GET_STATS),
            "cycle": (3, 123456),
            "status": (4, -5),
            "payload": (5, 8),
        }
        for name, (index, value) in mutations.items():
            def mutate(frame, index=index, value=value):
                fields = list(audit.struct.unpack(audit.RESPONSE_FMT, frame))
                fields[index] = value
                return audit.struct.pack(audit.RESPONSE_FMT, *fields)

            sock = FakeSocket(mutate=mutate)
            with self.assertRaises(audit.ProbeError, msg=name):
                self.probe(sock, reps=1, warmup=0)
            self.assertEqual(sock.closed, 1, name)

    def test_request_session_is_validated_before_send(self):
        audit._validate_control_request(audit.RKVA_PING, audit.RKVA_SESSION_NONE)
        audit._validate_control_request(audit.RKVA_GET_STATS, audit.RKVA_SESSION_NONE,
                                        allowed=(audit.RKVA_GET_STATS,))
        for op, session in ((4, audit.RKVA_SESSION_NONE),
                            (audit.RKVA_PING, 3),
                            (audit.RKVA_GET_STATS, 3)):
            with self.assertRaises(audit.ProbeError):
                audit._validate_control_request(op, session)
        with self.assertRaises(audit.ProbeError):
            audit._validate_control_request(audit.RKVA_GET_STATS,
                                            audit.RKVA_SESSION_NONE,
                                            allowed=(audit.RKVA_PING,))

    def test_ping_header_rejects_any_payload(self):
        header = audit.parse_response(audit.struct.pack(
            audit.RESPONSE_FMT, audit.RKVA_MAGIC, audit.RKVA_VERSION,
            audit.RKVA_PING, 1, 0, 4, 0, 0, 0, 0))
        with self.assertRaises(audit.ProbeError):
            audit._validate_ping(header, 1)
        header = audit.parse_response(audit.struct.pack(
            audit.RESPONSE_FMT, audit.RKVA_MAGIC, audit.RKVA_VERSION,
            audit.RKVA_PING, 1, 0, 0, 0, 0, 0, 0))
        audit._validate_ping(header, 1)

    def test_timeout_is_reported_and_socket_is_closed(self):
        sock = FakeSocket(timeout_at=1)
        with self.assertRaises(socket.timeout):
            self.probe(sock)
        self.assertEqual(sock.closed, 1)

    def test_eof_before_header_fails_and_closes(self):
        class ClosedSocket(FakeSocket):
            def recv(self, size):
                self.recv_calls += 1
                return b""

        sock = ClosedSocket()
        with self.assertRaises(audit.ProbeError):
            self.probe(sock, reps=1, warmup=0)
        self.assertEqual(sock.closed, 1)

    def test_bad_echo_fails_closed(self):
        def corrupt(frame):
            fields = list(audit.struct.unpack(audit.RESPONSE_FMT, frame))
            fields[3] += 1  # wrong cycle echo
            return audit.struct.pack(audit.RESPONSE_FMT, *fields)

        sock = FakeSocket(mutate=corrupt)
        with self.assertRaises(audit.ProbeError):
            self.probe(sock)
        self.assertEqual(sock.closed, 1)

    def test_busy_guard_blocks_and_allow_busy_is_opt_in(self):
        sock = FakeSocket()
        with mock.patch.object(audit, "find_busy_processes",
                               return_value=[{"pid": 1, "name": "llama-server"}]):
            with self.assertRaises(audit.ProbeError):
                audit.probe_empty_rtt("192.0.2.10", 50054, reps=1, warmup=0,
                                      connector=lambda *_: sock)
            self.assertEqual(sock.sent, [])
            report = audit.probe_empty_rtt("192.0.2.10", 50054, reps=1, warmup=0,
                                           allow_busy=True, connector=lambda *_: sock)
            self.assertTrue(report["guard"]["allow_busy"])
            self.assertEqual(report["samples_us"]["count"], 1)

    def test_bounds_are_enforced(self):
        for kwargs in ({"reps": 0}, {"reps": 1001}, {"warmup": 21},
                       {"timeout": 0.0}, {"timeout": 3.1}):
            with self.assertRaises(ValueError):
                audit.probe_empty_rtt("192.0.2.10", 50054, connector=lambda *_: None,
                                      **kwargs)
        self.assertEqual(audit.MAX_RTT_REPS, 1000)
        self.assertEqual(audit.MAX_RTT_TIMEOUT_S, 3.0)

    def test_get_stats_oversized_payload_fails_before_body(self):
        sock = FakeSocket(payload=b"x" * (audit.MAX_CONTROL_PAYLOAD + 1))
        with self.assertRaises(audit.ProbeError) as ctx:
            self.probe(sock, get_stats=True, reps=1, warmup=0)
        self.assertIn("cap", str(ctx.exception))
        self.assertEqual(sock.recv_calls, 1)
        self.assertEqual(sock.closed, 1)

    def test_get_stats_body_is_capped_and_read_after_header(self):
        sock = FakeSocket(payload=b'{"service":"rkva"}')
        report = self.probe(sock, get_stats=True, reps=1, warmup=0)
        ops = [audit.struct.unpack(audit.REQUEST_FMT, frame)[2] for frame in sock.sent]
        self.assertEqual(ops, [audit.RKVA_GET_STATS, audit.RKVA_PING])
        self.assertEqual(report["server_stats"], {"service": "rkva"})
        self.assertEqual(audit.MAX_CONTROL_PAYLOAD, 1 << 20)


class NetworkHelperTest(unittest.TestCase):
    def test_allowlist_accepts_only_read_only_queries(self):
        accepted = [
            ["ethtool", "eth0"],
            ["ethtool", "-i", "eth0"],
            ["ethtool", "-k", "eth0"],
            ["ethtool", "-c", "eth0"],
            ["ethtool", "--version"],
            ["ip", "-json", "route", "get", "192.168.1.200"],
            ["ip", "-json", "link", "show", "dev", "eth0"],
        ]
        rejected = [
            ["ethtool", "-s", "eth0", "speed", "1000"],
            ["ethtool", "-K", "eth0", "rx", "off"],
            ["ethtool", "-C", "eth0", "rx-usecs", "10"],
            ["ip", "route", "add", "10.0.0.0/8", "dev", "eth0"],
            ["ip", "link", "set", "eth0", "mtu", "9000"],
            ["sysctl", "-w", "net.core.rmem_max=1"],
            ["sh", "-c", "ethtool -s eth0 speed 1000"],
        ]
        for argv in accepted:
            self.assertTrue(audit.is_readonly_network_argv(argv), argv)
        for argv in rejected:
            self.assertFalse(audit.is_readonly_network_argv(argv), argv)

    def test_network_info_reports_missing_privilege_without_fake_zero(self):
        calls = []

        def runner(argv):
            calls.append(list(argv))
            if argv[:2] == ["ip", "-json"]:
                return 0, json.dumps([{"dev": "enp1s0", "gateway": "192.168.1.1"}])
            return 1, "Operation not permitted"

        report = audit.network_info("192.168.1.200", runner=runner)
        self.assertEqual(report["route"]["interface"], "enp1s0")
        self.assertIsNone(report["offloads"])
        self.assertIsNone(report["coalescing"])
        self.assertNotEqual(report["scope"], "ok")
        self.assertTrue(report["unavailable"])
        for argv in calls:
            self.assertTrue(audit.is_readonly_network_argv(argv), argv)
        self.assertFalse(any(flag in {"-s", "-K", "-C", "-G"} for argv in calls
                             for flag in argv[1:]))
        self.assertNotIn("speed_mbps", json.dumps(report["unavailable"]))

    def test_network_info_never_opens_a_socket(self):
        with mock.patch.object(audit.socket, "socket",
                               side_effect=AssertionError("socket opened")):
            report = audit.network_info(None, runner=lambda argv: (1, ""))
        self.assertEqual(report["mode"], "network")


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.trace = os.path.join(self.tmp.name, "trace.jsonl")
        write_trace(self.trace, golden_events(tokens=1, layers=1))

    def run_main(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = audit.main(argv)
        self.last_output = stdout.getvalue() + stderr.getvalue()
        return code

    def test_defaults_are_offline_and_capped(self):
        args = audit.parse_args(["--trace", self.trace])
        self.assertFalse(args.execute)
        self.assertFalse(args.empty_rtt)
        self.assertFalse(args.network_info)
        self.assertFalse(args.allow_busy)
        self.assertIsNone(args.out)
        self.assertEqual(args.min_position, 130560)
        self.assertEqual(args.max_position, 160000)
        self.assertEqual(args.context_cap, 160000)
        self.assertEqual(audit.parse_args(["--trace", self.trace]).max_position,
                         audit.CONTEXT_CAP)

    def test_execute_requires_host_and_safe_op(self):
        args = audit.parse_args(["--execute", "--empty-rtt", "--host",
                                 "192.168.1.200:50054"])
        self.assertEqual(audit.parse_endpoint(args.host), ("192.168.1.200", 50054))
        with self.assertRaises(ValueError):
            audit.parse_endpoint("192.168.1.200")
        with mock.patch.object(audit.socket, "create_connection",
                               side_effect=AssertionError("dialled")):
            self.assertEqual(self.run_main(["--execute", "--host", "192.168.1.200:50054"]), 1)
            self.assertEqual(self.run_main(["--execute", "--empty-rtt",
                                         "--host", "192.168.1.200"]), 1)

    def test_trace_mode_creates_only_the_requested_file_and_never_a_socket(self):
        out = os.path.join(self.tmp.name, "report.json")
        before = set(os.listdir(self.tmp.name))
        with mock.patch.object(audit.socket, "create_connection",
                               side_effect=AssertionError("socket")), \
                mock.patch.object(audit, "socket", autospec=False,
                                  new=mock.MagicMock(side_effect=AssertionError("socket"))), \
                mock.patch.object(audit.subprocess, "run",
                                  side_effect=AssertionError("subprocess")):
            code = self.run_main(["--trace", self.trace, "--out", out])
        self.assertEqual(code, 0)
        self.assertEqual(set(os.listdir(self.tmp.name)) - before, {"report.json"})
        report = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], audit.SCHEMA)
        self.assertEqual(report["mode"], "trace")
        self.assertEqual(report["counts"]["tokens"], 1)

    def test_trace_mode_without_out_writes_nothing(self):
        before = set(os.listdir(self.tmp.name))
        code = self.run_main(["--trace", self.trace])
        self.assertEqual(code, 0)
        self.assertEqual(set(os.listdir(self.tmp.name)), before)

    def test_independent_window_reuse_is_not_a_default_failure(self):
        write_trace(self.trace, golden_events(unique_cycles=False))
        self.assertEqual(self.run_main(["--trace", self.trace]), 0)
        self.assertEqual(self.run_main(["--trace", self.trace, "--strict"]), 2)
        out = os.path.join(self.tmp.name, "report.json")
        code = self.run_main(["--trace", self.trace, "--out", out])
        self.assertEqual(code, 0)
        report = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertEqual(report["correlation"]["scope"], "unknown")
        self.assertGreater(report["correlation"]["reused_in_independent_windows"], 0)

    def test_strict_returns_nonzero_for_unknown_correlation(self):
        events = golden_events()
        for item in events:
            item["cycle"] = 0
        write_trace(self.trace, events)
        self.assertEqual(self.run_main(["--trace", self.trace]), 0)
        self.assertEqual(self.run_main(["--trace", self.trace, "--strict"]), 2)

    def test_comparison_gate_is_wired_into_the_report(self):
        cpu = os.path.join(self.tmp.name, "cpu.json")
        gpu = os.path.join(self.tmp.name, "gpu.json")
        with open(cpu, "w", encoding="utf-8") as handle:
            json.dump({"op": "rpc_total", "samples_us": [10.0],
                       "window": {"min_position": 130560, "max_position": 130560}}, handle)
        with open(gpu, "w", encoding="utf-8") as handle:
            json.dump({"op": "rpc_total", "samples_us": [9.0, 12.0],
                       "window": {"min_position": 130560, "max_position": 130560}}, handle)
        out = os.path.join(self.tmp.name, "report.json")
        code = self.run_main(["--trace", self.trace, "--cpu-samples", cpu,
                           "--gpu-samples", gpu, "--out", out])
        self.assertEqual(code, 2)
        report = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertFalse(report["comparison"]["comparable"])
        self.assertEqual(report["comparison"]["scope"], "failed")
        self.assertEqual(report["unknown"]["comparison_scope"], "failed")

    def test_strict_fails_when_comparison_has_no_samples(self):
        cpu = os.path.join(self.tmp.name, "cpu.json")
        gpu = os.path.join(self.tmp.name, "gpu.json")
        with open(cpu, "w", encoding="utf-8") as handle:
            json.dump({"op": "rpc_total", "samples_us": [],
                       "window": {"min_position": 130560, "max_position": 160000}}, handle)
        with open(gpu, "w", encoding="utf-8") as handle:
            json.dump({"op": "rpc_total", "samples_us": [9.0],
                       "window": {"min_position": 130560, "max_position": 160000}}, handle)
        argv = ["--trace", self.trace, "--cpu-samples", cpu, "--gpu-samples", gpu]
        self.assertEqual(self.run_main(argv), 0)
        self.assertEqual(self.run_main(argv + ["--strict"]), 2)

    def test_source_parses_and_help_is_clear(self):
        source = Path(SCRIPT).read_text(encoding="utf-8")
        ast.parse(source)
        parser = audit.build_parser()
        text = parser.format_help()
        for token in ("--trace", "--execute", "--empty-rtt", "--network-info",
                      "--out", "offline", "safety"):
            self.assertIn(token, text)
        self.assertNotIn("import requests", source)
        for unsafe in ("os.system(", "shell=True", "shutil.rmtree"):
            self.assertNotIn(unsafe, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
