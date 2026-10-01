#!/usr/bin/env python3
"""Unit tests for tools/test-mixed-kv-mtp.py: draft-counter parsing, phase
validators, profile/draft flag composition, and error surfaces against a fake
HTTP endpoint. No real server or GPU. Run with:

    python3 tests/test-mixed-kv-mtp-harness.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import unittest.mock
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
_spec = importlib.util.spec_from_file_location("test_mixed_kv_mtp", _TOOLS / "test-mixed-kv-mtp.py")
assert _spec and _spec.loader
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

DRAFT_LOG = (
    "slot 0: prompt eval time = 100.00 ms / 300 tokens\n"
    "slot 0: draft acceptance = 0.62500 ( 10 accepted /  16 generated), mean len =  2.25\n"
    "slot 0: graphs reused = 42\n"
)
ZERO_DRAFT_LOG = "slot 0: draft acceptance = 0.00000 (  0 accepted /   0 generated), mean len =  1.00\n"
NO_DRAFT_LOG = "slot 0: eval time = 10.00 ms / 16 tokens\n"


def minimal_args(**overrides) -> argparse.Namespace:
    values = {
        "server_binary": Path("/bin/true"), "model": Path("/bin/true"),
        "host": "127.0.0.1", "port": 1, "ctx_size": 8192,
        "prompt_tokens": 4096, "suffix_tokens": 32, "decode_tokens": 16,
        "prompt_short": 300, "prompt_long": 700, "n_predict": 16,
        "request_timeout": 30.0, "telemetry_device": "none", "seed": 42,
        "n_probs": 3, "batch_size": 256, "ubatch_size": 256,
        "cuda_device": "CUDA0", "remote_kv_types": ["q4_0"],
        "spec_draft_n_max": 4, "spec_draft_n_max_l": 2,
        "local_ctx": 512, "high_ctx": 1024, "mtp_max_tokens": 512,
        "gate_ctx": 513, "prompt_lengths": "127,128,129,385",
        "handoff_evidence_markers": "", "apu_tdp": None,
        "amd_sclk_prefill": None, "amd_sclk_decode": None,
        "mandatory_reuse_lengths": "385",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestResponseDraft(unittest.TestCase):
    """PRIMARY per-request draft evidence: timings draft_n/draft_n_accepted."""

    def test_present_counts_pass(self):
        timings = {"cache_n": 256, "prompt_n": 44, "predicted_n": 16,
                   "draft_n": 8, "draft_n_accepted": 4}
        self.assertEqual(m.validate_response_draft(timings, "t"), [])

    def test_missing_draft_fields_fail(self):
        failures = m.validate_response_draft({"cache_n": 256, "prompt_n": 44}, "t")
        self.assertTrue(any("lack draft_n" in f for f in failures))

    def test_zero_drafts_fail(self):
        failures = m.validate_response_draft({"draft_n": 0, "draft_n_accepted": 0}, "t")
        self.assertTrue(any("draft_n invalid" in f for f in failures))

    def test_accepted_missing_fails(self):
        failures = m.validate_response_draft({"draft_n": 8}, "t")
        self.assertTrue(any("draft_n_accepted missing" in f for f in failures))

    def test_accepted_out_of_range_fails(self):
        failures = m.validate_response_draft({"draft_n": 8, "draft_n_accepted": 9}, "t")
        self.assertTrue(any("outside [0, draft_n=8]" in f for f in failures))

    def test_non_integer_fails(self):
        failures = m.validate_response_draft({"draft_n": "8", "draft_n_accepted": 2}, "t")
        self.assertTrue(any("draft_n invalid" in f for f in failures))


class TestBaselineRegression(unittest.TestCase):
    """Real observed responses from the immutable baseline
    (baseline-mtp-boundaries/results.json): target/draft KVarN4, MTP N4,
    ctx 8192, b/ub 256, cache-ram 0, checkpoints 1, 20W/2700.
    Observed: 127/128/129 cold->suffix cache_n=0 (existing baseline behavior);
    suffix->repeat cache_n=128 with prompt_n 31/32/33; 385 cold->suffix
    cache_n=256/prompt_n 161; 385 repeat cache_n=384/prompt_n 33. All phases
    drafted with positive accepted counts. cache_reprocessed_n equals
    prompt_n (tokens actually processed), never a fallback signal."""

    def _timings(self, cache_n, prompt_n, reprocessed, draft_n, accepted,
                 predicted_n=16, reason="committed"):
        return {"cache_n": cache_n, "cache_lcp_n": 0, "cache_planned_n": cache_n,
                "cache_reprocessed_n": reprocessed, "cache_source": "live",
                "cache_reason": reason, "prompt_n": prompt_n, "predicted_n": predicted_n,
                "draft_n": draft_n, "draft_n_accepted": accepted}

    def test_boundary_suffix_cache0_is_valid_baseline(self):
        # 129 prefix + 32 suffix: baseline cache_n=0, prompt_n=161, drafts 20/9.
        timings = self._timings(0, 161, 161, 20, 9, reason="candidate")
        failures, reuse_proven = m.validate_leg(timings, False, 161, 16,
                                                "129-suffix", cold=False,
                                                reuse_required=False)
        self.assertEqual(failures, [])
        self.assertFalse(reuse_proven)

    def test_boundary_repeat_positive_hit(self):
        # 129 repeat: cache_n=128, prompt_n=33, drafts 20/9.
        timings = self._timings(128, 33, 33, 20, 9)
        failures, reuse_proven = m.validate_leg(timings, False, 161, 16,
                                                "129-repeat", cold=False,
                                                reuse_required=True)
        self.assertEqual(failures, [])
        self.assertTrue(reuse_proven)

    def test_385_suffix_positive_hit(self):
        # 385 suffix: cache_n=256, prompt_n=161, drafts 18/9.
        timings = self._timings(256, 161, 161, 18, 9)
        failures, reuse_proven = m.validate_leg(timings, False, 417, 16,
                                                "385-suffix", cold=False,
                                                reuse_required=True)
        self.assertEqual(failures, [])
        self.assertTrue(reuse_proven)

    def test_385_repeat_positive_hit(self):
        # 385 repeat: cache_n=384, prompt_n=33.
        timings = self._timings(384, 33, 33, 18, 9)
        failures, reuse_proven = m.validate_leg(timings, False, 417, 16,
                                                "385-repeat", cold=False,
                                                reuse_required=True)
        self.assertEqual(failures, [])
        self.assertTrue(reuse_proven)

    def test_full_recompute_rejected_at_385(self):
        # Fabricated full recomputation: cache_n=0 where reuse is mandatory.
        timings = self._timings(0, 417, 417, 18, 9)
        failures, reuse_proven = m.validate_leg(timings, False, 417, 16,
                                                "385-suffix", cold=False,
                                                reuse_required=True)
        self.assertTrue(any("reuse required but cache_n=0" in f for f in failures))
        self.assertFalse(reuse_proven)

    def test_cold_counts_pass(self):
        timings = self._timings(0, 385, 385, 15, 11, reason="no_common_prefix")
        failures, _ = m.validate_leg(timings, False, 385, 16, "385-cold",
                                     cold=True, reuse_required=False)
        self.assertEqual(failures, [])

    def test_baseline_drafts_all_positive(self):
        # Every baseline phase drafted with positive accepted counts.
        for timings in (
                self._timings(0, 127, 127, 16, 10, reason="no_common_prefix"),
                self._timings(0, 159, 159, 20, 9, reason="candidate"),
                self._timings(128, 31, 31, 20, 9),
                self._timings(256, 161, 161, 18, 9),
                self._timings(384, 33, 33, 18, 9)):
            self.assertEqual(m.validate_response_draft(timings, "t"), [])

    def test_reprocessed_meta_recorded_not_guarded(self):
        # cache_reprocessed_n == prompt_n on valid baseline legs; recorded
        # for review and never treated as a fallback.
        timings = self._timings(128, 33, 33, 20, 9)
        meta = m.validate_reprocessed(timings, "t")
        self.assertEqual(meta["cache_reprocessed_n"], 33)
        self.assertEqual(meta["cache_reason"], "committed")
        failures, _ = m.validate_leg(timings, False, 161, 16, "t", cold=False,
                                     reuse_required=True)
        self.assertEqual(failures, [])


class TestM1Acceptance(unittest.TestCase):
    LAYOUT_OK = [
        {"layer": 0, "device": "Vulkan0", "type_k": "q4_0", "type_v": "q4_0",
         "domain": "standard-hadamard"},
    ]

    def test_missing_layout_fails_scenario(self):
        state, error = m.m1_acceptance([], [], "q4_0")
        self.assertEqual(state, "failed")
        self.assertIn("unique layers", error)

    def test_wrong_type_fails_scenario(self):
        state, error = m.m1_acceptance([], self.LAYOUT_OK, "q8_0")
        self.assertEqual(state, "failed")
        self.assertIn("expected q8_0", error)

    def test_correct_layout_allowed_scenario(self):
        state, error = m.m1_acceptance([], self.LAYOUT_OK, "q4_0")
        self.assertEqual(state, "complete")
        self.assertEqual(error, "")

    def test_failures_always_merged(self):
        state, error = m.m1_acceptance(["phase failed"], [], "q4_0")
        self.assertEqual(state, "failed")
        self.assertIn("phase failed", error)
        self.assertIn("unique layers", error)


class TestMtpPlacement(unittest.TestCase):
    def test_local_attention_off_layers_zero(self):
        log = "MTP K/V placement: local_attention=off, layers=0, prefill=remote, cuda_reserve=650.00 MiB\n"
        self.assertEqual(m.mtp_placement_failures(log), [])

    def test_local_attention_on_fails(self):
        log = "MTP K/V placement: local_attention=vulkan:0, layers=1, prefill=remote, cuda_reserve=650.00 MiB\n"
        failures = m.mtp_placement_failures(log)
        self.assertTrue(any("local_attention is vulkan:0" in f for f in failures))

    def test_layers_nonzero_fails(self):
        log = "MTP K/V placement: local_attention=off, layers=auto, prefill=remote, cuda_reserve=650.00 MiB\n"
        failures = m.mtp_placement_failures(log)
        self.assertTrue(any("layers is auto" in f for f in failures))

    def test_missing_line_fails(self):
        failures = m.mtp_placement_failures("Vulkan-layers=1/16\n")
        self.assertTrue(any("lacks the 'MTP K/V placement:'" in f for f in failures))


class TestPowerEvidence(unittest.TestCase):
    def test_missing_markers_fail(self):
        args = minimal_args(apu_tdp=20, amd_sclk_prefill=2700, amd_sclk_decode=2700)
        failures = m.power_evidence_failures(args, "Vulkan-layers=1/16\n")
        self.assertTrue(any("APU TDP: active limit 20 W" in f for f in failures))
        self.assertTrue(any("locked 2700 MHz" in f for f in failures))

    def test_equal_clocks_single_write_pass(self):
        args = minimal_args(apu_tdp=20, amd_sclk_prefill=2700, amd_sclk_decode=2700)
        log = ("APU TDP: active limit 20 W\n"
               "AMD graphics SCLK -> prefill, locked 2700 MHz\n")
        # One hardware write for equal clocks (the prefill lock is the write;
        # the decode keeps it). Both the generic locked marker and the
        # directional line satisfy the check.
        self.assertEqual(m.power_evidence_failures(args, log), [])

    def test_directional_write_pass(self):
        args = minimal_args(apu_tdp=20, amd_sclk_prefill=2400, amd_sclk_decode=2700)
        log = ("APU TDP: active limit 20 W\n"
               "AMD graphics SCLK -> prefill, locked 2400 MHz\n"
               "AMD graphics SCLK -> decode, locked 2700 MHz\n")
        self.assertEqual(m.power_evidence_failures(args, log), [])

    def test_missing_decode_write_fails(self):
        args = minimal_args(apu_tdp=None, amd_sclk_prefill=None, amd_sclk_decode=2700)
        failures = m.power_evidence_failures(args, "")
        self.assertTrue(any("-> decode, locked 2700 MHz" in f for f in failures))


class TestDraftEvidence(unittest.TestCase):
    """SECONDARY log-line evidence: parsed, recorded and reconciled; never a
    false fallback (async flush) and never a stale previous-phase line."""

    def test_present_counters(self):
        evidence = m.extract_draft_evidence(DRAFT_LOG)
        self.assertEqual(evidence["n_accepted"], 10)
        self.assertEqual(evidence["n_generated"], 16)
        self.assertAlmostEqual(evidence["ratio"], 0.625)

    def test_absent_line_means_pending_not_failure(self):
        self.assertIsNone(m.extract_draft_evidence(NO_DRAFT_LOG))
        reconciled = m.reconcile_log_draft(NO_DRAFT_LOG, 16)
        self.assertFalse(reconciled["log_line_present"])
        self.assertIn("asynchronously", reconciled["note"])

    def test_zero_generated_parsed(self):
        self.assertIsNotNone(m.extract_draft_evidence(ZERO_DRAFT_LOG))

    def test_reconcile_matching(self):
        reconciled = m.reconcile_log_draft(DRAFT_LOG, 16)
        self.assertTrue(reconciled["log_line_present"])
        self.assertTrue(reconciled["matches_response"])

    def test_reconcile_mismatch_recorded(self):
        reconciled = m.reconcile_log_draft(DRAFT_LOG, 8)
        self.assertFalse(reconciled["matches_response"])

    def test_stale_line_not_reused(self):
        # A previous phase's line in an earlier delta must not satisfy the
        # current phase: reconcile only sees the current delta.
        self.assertIsNone(m.extract_draft_evidence(""))

    def test_graphs_reused(self):
        self.assertEqual(m.extract_graphs_reused(DRAFT_LOG), 42)
        self.assertIsNone(m.extract_graphs_reused(NO_DRAFT_LOG))


class TestCadence(unittest.TestCase):
    def test_cadence_equal(self):
        a = {"json": {"timings": {"cache_n": 300, "prompt_n": 32, "predicted_n": 16}}}
        b = {"json": {"timings": {"cache_n": 300, "prompt_n": 32, "predicted_n": 16}}}
        self.assertEqual(m.cadence_of(a), m.cadence_of(b))

    def test_cadence_differs(self):
        a = {"json": {"timings": {"cache_n": 300, "prompt_n": 32, "predicted_n": 16}}}
        b = {"json": {"timings": {"cache_n": 256, "prompt_n": 76, "predicted_n": 16}}}
        self.assertNotEqual(m.cadence_of(a), m.cadence_of(b))


class TestFlags(unittest.TestCase):
    def test_m1_argv(self):
        args = minimal_args()
        command = " ".join(m.m1_extra_args(args, "q4_0"))
        self.assertIn("--spec-type draft-mtp", command)
        self.assertIn("--spec-draft-type-k kvarn4", command)
        self.assertIn("--spec-draft-type-v kvarn4", command)
        self.assertIn("--spec-draft-n-max 4", command)
        self.assertIn("--remote-attn-cache-type-k q4_0", command)
        self.assertIn("--cache-ram 0", command)
        self.assertIn("--ctx-checkpoints 1", command)

    def test_m2_argv_profile_and_draft_flags(self):
        args = minimal_args()
        command = " ".join(m.m2_extra_args(args, "q4_0"))
        # The base ServerInstance command already carries --ctx-size (set to
        # high_ctx for M2); the extra args must NOT duplicate it.
        self.assertNotIn("--ctx-size ", command)
        self.assertIn("--ctx-size-mtp 512", command)
        self.assertIn("--mtp-max-tokens 512", command)
        # Target per-tier KVarN4 and draft per-tier KVarN4, both tiers.
        for flag in ("--cache-type-k-m kvarn4", "--cache-type-v-m kvarn4",
                     "--cache-type-k-l kvarn4", "--cache-type-v-l kvarn4",
                     "--spec-draft-type-k-m kvarn4", "--spec-draft-type-v-m kvarn4",
                     "--spec-draft-type-k-l kvarn4", "--spec-draft-type-v-l kvarn4"):
            self.assertIn(flag, command)
        self.assertIn("--spec-draft-n-max 4", command)
        self.assertIn("--spec-draft-n-max-l 2", command)
        self.assertIn("--remote-attn-min-context 513", command)
        self.assertIn("--cache-ram 256", command)
        self.assertIn("--ctx-checkpoints 1", command)

    def test_m2_full_argv_exactly_one_ctx_size(self):
        args = minimal_args()
        m2_args = argparse.Namespace(**{**vars(args), "ctx_size": args.high_ctx})
        instance = m.r.ServerInstance(m2_args, Path(tempfile.mkdtemp()), "M2",
                                      m.m2_extra_args(args, "q4_0"))
        command = " ".join(instance.command)
        self.assertEqual(command.count("--ctx-size "), 1)
        self.assertEqual(command.count("--ctx-size-mtp"), 1)
        self.assertIn("--ctx-size 1024", command)


class TestReusedValidators(unittest.TestCase):
    """Validators inherited from the frozen transition runner."""

    def test_gate_recomputation_fails(self):
        body = {"timings": {"cache_n": 0, "prompt_n": 700, "predicted_n": 16}, "truncated": False}
        failures = m.r.validate_handoff_counts(body, 300, 700, 16)
        self.assertTrue(any("cache_n>0" in f for f in failures))

    def test_gate_preserved_prefix_passes(self):
        body = {"timings": {"cache_n": 256, "prompt_n": 444, "predicted_n": 16}, "truncated": False}
        self.assertEqual(m.r.validate_handoff_counts(body, 300, 700, 16), [])

    def test_cache_hit_required(self):
        body = {"timings": {"cache_n": 0, "prompt_n": 48, "predicted_n": 16}, "truncated": False}
        failures = m.r.validate_completion_counts(body, 48, cold=False, expected_decode=16)
        self.assertTrue(any("cache_n>0" in f for f in failures))

    def test_missing_handoff_evidence_pending(self):
        evidence = {"configured": False, "found": False, "matched_lines": []}
        self.assertEqual(m.r.acceptance_state([], evidence), "pending_evidence")
        self.assertNotEqual(m.r.acceptance_state([], evidence), "complete")

    def test_handoff_evidence_found_complete(self):
        evidence = {"configured": True, "found": True, "matched_lines": ["x"]}
        self.assertEqual(m.r.acceptance_state([], evidence), "complete")

    def test_error_surfaces(self):
        self.assertTrue(m.r.response_is_error({"status": 500, "json": {}})[0])
        self.assertTrue(m.r.response_is_error(
            {"status": 200, "json": {"error": {"code": 200, "message": "boom"}}})[0])
        self.assertFalse(m.r.response_is_error({"status": 200, "json": {"ok": 1}})[0])


class FakeConfig:
    token_ids = list(range(4000))
    suffix_ids = list(range(8000, 8040))
    completion = {
        "content": "fake", "tokens": [1, 2, 3],
        "timings": {"cache_n": 0, "prompt_n": 64, "predicted_n": 16},
        "truncated": False,
    }
    completion_status = 200
    completion_mode = "ok"  # ok | error_200 | error_500


class FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"code": 404, "message": "not found"}})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/tokenize":
            ids = FakeConfig.suffix_ids if body.get("add_special") is False else FakeConfig.token_ids
            self._json(200, {"tokens": ids})
        elif parsed.path == "/completion":
            if FakeConfig.completion_mode == "error_200":
                self._json(200, {"error": {"code": 200, "message": "canned 200"}})
            elif FakeConfig.completion_mode == "error_500":
                self._json(500, {"error": {"code": 500, "message": "canned 500"}})
            else:
                self._json(FakeConfig.completion_status, FakeConfig.completion)
        elif parsed.path.startswith("/slots/"):
            self._json(200, {"n_erased": 0})
        else:
            self._json(404, {"error": {"code": 404, "message": "not found"}})


class FakeServer:
    def __init__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.httpd.server_port}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class TestManifestSerialization(unittest.TestCase):
    """The shared runner's entrypoint crashed on PosixPath in the manifest
    configuration before any server launch; the MTP manifest uses the shared
    configuration_from_args helper and must serialize the same way."""

    def _assert_no_paths(self, value, where):
        if isinstance(value, Path):
            self.fail(f"{where}: pathlib.Path leaked into the manifest: {value}")
        if isinstance(value, dict):
            for key, item in value.items():
                self._assert_no_paths(item, f"{where}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self._assert_no_paths(item, f"{where}[{index}]")

    def test_real_default_namespace_serializes(self):
        args = m.build_parser().parse_args([])
        with tempfile.TemporaryDirectory() as tmp:
            config = m.r.configuration_from_args(args)
            target = Path(tmp) / "run.json"
            m.r.write_json(target, {"configuration": config, "probe": True})
            parsed = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(parsed["probe"], True)
        self.assertIsInstance(parsed["configuration"]["server_binary"], str)
        self.assertIsInstance(parsed["configuration"]["model"], str)
        self.assertIsNone(parsed["configuration"]["output_dir"])
        self._assert_no_paths(parsed, "manifest")

    def test_unknown_object_still_fails_loudly(self):
        with self.assertRaises(TypeError):
            m.r.write_json(Path(tempfile.mkdtemp()) / "x.json", {"bad": {1, 2}})


class _FakeServerInstance(m.r.ServerInstance):
    started: list = []

    def start(self) -> None:
        _FakeServerInstance.started.append(self.command)
        self.log_path.write_text("", encoding="utf-8")


class TestEntrypointFakeEndToEnd(unittest.TestCase):
    """MTP main() up to server launch with a fresh empty output dir, stub
    ServerInstance, no real server or GPU."""

    def test_main_reaches_launch_without_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "results"
            out_dir.mkdir()
            port = 20000 + (os.getpid() % 2000)
            argv = ["test-mixed-kv-mtp.py",
                    "--model", "/bin/true", "--server-binary", "/bin/true",
                    "--output-dir", str(out_dir), "--scenarios", "M1",
                    "--remote-kv-types", "q4_0", "--port", str(port),
                    "--telemetry-device", "none"]
            with unittest.mock.patch.object(sys, "argv", argv), \
                    unittest.mock.patch.object(m.r, "ServerInstance",
                                               _FakeServerInstance):
                _FakeServerInstance.started.clear()
                exit_code = m.main()
            self.assertEqual(exit_code, 1)  # scenario fails on dead HTTP; no crash
            run_json = out_dir / "run.json"
            self.assertTrue(run_json.is_file())
            manifest = json.loads(run_json.read_text(encoding="utf-8"))
            self.assertIsInstance(manifest["configuration"]["server_binary"], str)
            self.assertTrue(_FakeServerInstance.started,
                            "server launch path was not reached")
            self.assertTrue((out_dir / "summary.json").is_file())


class _FakeM2ServerInstance:
    gate_lines: list[str] = []
    active: "_FakeM2ServerInstance | None" = None

    def __init__(self, args, output_dir, case_id, extra_args):
        self.args = args
        self.case_id = case_id
        self.command = ["fake-llama-server", *extra_args]
        self.slot_save_path = output_dir / f"slot-save-{case_id}"
        self.slot_save_path.mkdir(parents=True, exist_ok=True)
        self.log_path = output_dir / f"server-{case_id}.log"
        self.log_offset = 0
        _FakeM2ServerInstance.active = self

    def start(self):
        startup = (
            "MTP K/V placement: local_attention=off, layers=0, prefill=remote, cuda_reserve=256.00 MiB\n"
            "adaptive remote attention: threshold=513 backend=off\n")
        self.log_path.write_text(startup, encoding="utf-8")

    def append_log(self, lines):
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write("".join(line + "\n" for line in lines))

    def read_log_delta(self):
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        delta = text[self.log_offset:]
        self.log_offset = len(text)
        return delta

    def log_evidence(self, delta):
        markers = m.r.MARKERS
        return [line.strip() for line in delta.splitlines()
                if any(marker in line.lower() for marker in markers)]

    def stop(self):
        return 0


class TestM2LogDeltaConsumption(unittest.TestCase):
    """Exercise scenario_m2's returned completion deltas, including evidence
    that may arrive immediately before the completion helper reads the log."""

    HANDOFF_MARKER = "adaptive streaming conversion: KVarN-to-Qx completed"
    GATE_LINES = [
        "adaptive remote attention: threshold=513 backend=vulkan:0",
        "mixed KV layer=7 device=Vulkan0 type_k=q4_0 type_v=q4_0 domain=standard-hadamard",
        "adaptive context transition complete",
        HANDOFF_MARKER,
    ]

    def _run_m2(self, gate_lines):
        args = minimal_args(
            remote_kv_types=["q4_0"],
            handoff_evidence_markers=self.HANDOFF_MARKER,
        )
        response_by_phase = {
            "M2-short-cold": {"cache_n": 0, "prompt_n": 300, "predicted_n": 16,
                               "draft_n": 16, "draft_n_accepted": 10},
            "M2-handoff": {"cache_n": 256, "prompt_n": 444, "predicted_n": 16,
                            "draft_n": 16, "draft_n_accepted": 10},
            "M2-long-continuation": {"cache_n": 384, "prompt_n": 316, "predicted_n": 16,
                                      "draft_n": 16, "draft_n_accepted": 10},
        }

        def fake_http_request(_base_url, _method, path, _payload, _timeout,
                              _record_path, _telemetry_device, _case_id, phase,
                              request_body_record=None):
            del request_body_record
            self.assertEqual(path, "/completion")
            server = _FakeM2ServerInstance.active
            self.assertIsNotNone(server)
            if phase == "M2-handoff":
                server.append_log(gate_lines)
            return {"status": 200, "json": {
                "timings": response_by_phase[phase],
                "truncated": False,
                "content": "fake continuation",
                "tokens": [1, 2],
                "completion_probabilities": [],
            }}

        frozen = {"prefix_token_ids": list(range(1000)), "suffix_token_ids": []}
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with unittest.mock.patch.object(m.r, "ServerInstance", _FakeM2ServerInstance), \
                    unittest.mock.patch.object(m.r, "http_request", fake_http_request), \
                    unittest.mock.patch.object(m.r, "create_frozen_prompts", return_value=frozen):
                result = m.scenario_m2(args, output_dir, {}, None)
        return result["runs"][0]

    def test_m2_uses_completion_returned_deltas_for_gate_evidence(self):
        _FakeM2ServerInstance.gate_lines = self.GATE_LINES
        case = self._run_m2(self.GATE_LINES)
        self.assertEqual(case["state"], "complete", case.get("error"))
        self.assertTrue(any("backend=off" in line for line in case["pre_gate_evidence"]))
        self.assertTrue(any("adaptive context transition complete" in line
                            for line in case["gate_evidence"]))
        self.assertEqual(len(case["gate_mixed_format_log_evidence"]), 1)
        self.assertTrue(case["handoff_evidence"]["found"])

    def test_m2_still_fails_when_gate_markers_are_genuinely_missing(self):
        incomplete = [
            "adaptive remote attention: threshold=513 backend=vulkan:0",
            "mixed KV layer=7 device=Vulkan0 type_k=q4_0 type_v=q4_0 domain=standard-hadamard",
        ]
        case = self._run_m2(incomplete)
        self.assertEqual(case["state"], "failed")
        self.assertIn("gate log lacks 'adaptive context transition complete'",
                      case["validation_failures"])
        self.assertTrue(any("lacks the configured handoff evidence markers" in failure
                            for failure in case["validation_failures"]))


class TestFakeEndpointIntegration(unittest.TestCase):
    def setUp(self):
        self.fake = FakeServer()
        FakeConfig.completion_mode = "ok"
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.fake.close()
        self.tmp.cleanup()

    def _records(self):
        return Path(self.tmp.name) / "requests.jsonl"

    def test_completion_ok_and_recorded(self):
        result = m.r.http_request(self.fake.base_url, "POST", "/completion",
                                  {"prompt": [1]}, 10, self._records(), "none", "t", "phase")
        self.assertEqual(result["status"], 200)
        rows = [json.loads(line) for line in self._records().read_text().splitlines()]
        self.assertEqual(rows[0]["phase"], "phase")
        self.assertIn("telemetry_before", rows[0])

    def test_error_200_body_captured(self):
        FakeConfig.completion_mode = "error_200"
        result = m.r.http_request(self.fake.base_url, "POST", "/completion",
                                  {"prompt": [1]}, 10, self._records(), "none", "t", "phase")
        self.assertEqual(result["status"], 200)
        self.assertTrue(m.r.response_is_error(result)[0])

    def test_error_500_captured(self):
        FakeConfig.completion_mode = "error_500"
        result = m.r.http_request(self.fake.base_url, "POST", "/completion",
                                  {"prompt": [1]}, 10, self._records(), "none", "t", "phase")
        self.assertEqual(result["status"], 500)
        self.assertTrue(m.r.response_is_error(result)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
