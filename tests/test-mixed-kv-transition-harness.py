#!/usr/bin/env python3
"""Unit tests for tools/test-mixed-kv-transition.py request/assertion logic.

Exercises the acceptance runner against a fake HTTP endpoint (stdlib
http.server) and verifies process cleanup only touches its own process group.
No llama-server and no GPU are involved. Run with:

    python3 tests/test-mixed-kv-transition-harness.py
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "test_mixed_kv_transition", TOOLS_DIR / "test-mixed-kv-transition.py")
assert _spec and _spec.loader
runner = importlib.util.module_from_spec(_spec)
assert _spec.loader
_spec.loader.exec_module(runner)


class FakeConfig:
    """Mutable canned responses for the fake server."""

    token_ids = list(range(1000))
    suffix_ids = list(range(5000, 5040))
    completion = {
        "content": "fake", "tokens": [1, 2, 3],
        "timings": {"cache_n": 0, "prompt_n": 64, "predicted_n": 16},
        "truncated": False,
        "completion_probabilities": [{"content": "a", "probs": [{"content": "a", "prob": 0.9}]}],
    }
    completion_status = 200
    completion_mode = "ok"  # ok | error_200 | error_500
    slot_counts = {
        "save": {"id_slot": 0, "filename": "x.bin", "n_saved": 64, "n_written": 1234},
        "restore": {"id_slot": 0, "filename": "x.bin", "n_restored": 64, "n_read": 1234},
        "erase": {"id_slot": 0, "n_erased": 64},
    }
    slot_mode = "ok"  # ok | error_200 | error_500


class FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"code": 404, "type": "not_found_error",
                                       "message": "not found"}})

    def do_POST(self):  # noqa: N802
        body = self._read_body()
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/tokenize":
            ids = FakeConfig.suffix_ids if body.get("add_special") is False else FakeConfig.token_ids
            self._json(200, {"tokens": ids})
            return
        if parsed.path == "/completion":
            if FakeConfig.completion_mode == "error_200":
                self._json(200, {"error": {"code": 200, "type": "server_error",
                                           "message": "canned 200 error"}})
            elif FakeConfig.completion_mode == "error_500":
                self._json(500, {"error": {"code": 500, "type": "server_error",
                                           "message": "canned 500 error"}})
            else:
                self._json(FakeConfig.completion_status, FakeConfig.completion)
            return
        if parsed.path.startswith("/slots/"):
            action = urllib.parse.parse_qs(parsed.query).get("action", [""])[0]
            if FakeConfig.slot_mode == "error_200":
                self._json(200, {"error": {"code": 200, "type": "server_error",
                                           "message": "canned slot 200 error"}})
            elif FakeConfig.slot_mode == "error_500":
                self._json(500, {"error": {"code": 500, "type": "server_error",
                                           "message": "canned slot 500 error"}})
            else:
                self._json(200, FakeConfig.slot_counts.get(action, {}))
            return
        self._json(404, {"error": {"code": 404, "type": "not_found_error",
                                   "message": "not found"}})


class FakeServer:
    def __init__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def minimal_args(**overrides) -> argparse.Namespace:
    values = {
        "server_binary": Path("/bin/true"), "model": Path("/bin/true"),
        "host": "127.0.0.1", "port": 1, "ctx_size": 8192,
        "prompt_tokens": 64, "suffix_tokens": 8, "decode_tokens": 16,
        "prompt_short": 32, "prompt_long": 48, "n_predict": 16,
        "request_timeout": 30.0, "telemetry_device": "none", "seed": 42,
        "n_probs": 3, "batch_size": 256, "ubatch_size": 256,
        "cuda_device": "CUDA0", "remote_attn_layers": 1,
        "apu_tdp": None, "amd_sclk_prefill": None, "amd_sclk_decode": None,
        "remote_kv_types": ["q4_0"],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestResponseErrorDetection(unittest.TestCase):
    def test_error_200_body_detected(self):
        failed, detail = runner.response_is_error(
            {"status": 200, "json": {"error": {"code": 200, "type": "server_error",
                                               "message": "boom"}}})
        self.assertTrue(failed)
        self.assertIn("boom", detail)

    def test_error_500_status_detected(self):
        failed, detail = runner.response_is_error(
            {"status": 500, "json": {"error": {"code": 500, "type": "server_error",
                                               "message": "boom"}}})
        self.assertTrue(failed)
        self.assertIn("http_status=500", detail)

    def test_ok_response_not_error(self):
        failed, _ = runner.response_is_error(
            {"status": 200, "json": {"n_erased": 1}})
        self.assertFalse(failed)

    def test_non_json_body_not_error(self):
        failed, _ = runner.response_is_error({"status": 200, "json": None})
        self.assertFalse(failed)


class TestCountValidation(unittest.TestCase):
    def test_cold_counts_pass(self):
        body = {"timings": {"cache_n": 0, "prompt_n": 64, "predicted_n": 16}, "truncated": False}
        self.assertEqual(runner.validate_completion_counts(body, 64, cold=True, expected_decode=16), [])

    def test_continuation_counts_pass(self):
        body = {"timings": {"cache_n": 32, "prompt_n": 32, "predicted_n": 16}, "truncated": False}
        self.assertEqual(runner.validate_completion_counts(body, 64, cold=False, expected_decode=16), [])

    def test_cold_with_cache_fails(self):
        body = {"timings": {"cache_n": 1, "prompt_n": 63, "predicted_n": 16}, "truncated": False}
        failures = runner.validate_completion_counts(body, 64, cold=True, expected_decode=16)
        self.assertTrue(any("cache_n=0" in failure for failure in failures))

    def test_sum_mismatch_fails(self):
        body = {"timings": {"cache_n": 0, "prompt_n": 40, "predicted_n": 16}, "truncated": False}
        failures = runner.validate_completion_counts(body, 64, cold=True, expected_decode=16)
        self.assertTrue(any("cache_n(0) + prompt_n(40)" in failure for failure in failures))

    def test_missing_counts_fail(self):
        failures = runner.validate_completion_counts({"timings": {}}, 64, cold=True, expected_decode=16)
        self.assertEqual(len(failures), 1)

    def test_predicted_mismatch_fails(self):
        body = {"timings": {"cache_n": 0, "prompt_n": 64, "predicted_n": 5}, "truncated": False}
        failures = runner.validate_completion_counts(body, 64, cold=True, expected_decode=16)
        self.assertTrue(any("predicted_n(5)" in failure for failure in failures))


class TestSlotCounts(unittest.TestCase):
    def test_save_counts(self):
        counts, failures = runner.parse_slot_counts(
            {"status": 200, "json": {"n_saved": 64, "n_written": 1234}}, "save")
        self.assertEqual(failures, [])
        self.assertEqual(counts, {"n_saved": 64, "n_written": 1234})

    def test_save_missing_counts(self):
        counts, failures = runner.parse_slot_counts(
            {"status": 200, "json": {"n_saved": 0}}, "save")
        self.assertTrue(any("n_written" in failure for failure in failures))

    def test_restore_error_body(self):
        counts, failures = runner.parse_slot_counts(
            {"status": 200, "json": {"error": {"code": 500, "message": "truncated"}}}, "restore")
        self.assertTrue(any("restore failed" in failure for failure in failures))

    def test_erase_counts(self):
        counts, failures = runner.parse_slot_counts(
            {"status": 200, "json": {"n_erased": 5}}, "erase")
        self.assertEqual(failures, [])
        self.assertEqual(counts, {"n_erased": 5})


class TestFakeEndpointIntegration(unittest.TestCase):
    """http_request + response_is_error end to end against the fake server."""

    def setUp(self):
        self.fake = FakeServer()
        FakeConfig.completion_mode = "ok"
        FakeConfig.slot_mode = "ok"
        self.tmp = tempfile.TemporaryDirectory()
        self.records = Path(self.tmp.name) / "requests.jsonl"

    def tearDown(self):
        self.fake.close()
        self.tmp.cleanup()

    def test_health_ok(self):
        result = runner.http_request(self.fake.base_url, "GET", "/health", None,
                                     10, self.records, "none", "t", "health")
        self.assertEqual(result["status"], 200)
        self.assertEqual(result["json"], {"status": "ok"})

    def test_completion_ok(self):
        result = runner.http_request(self.fake.base_url, "POST", "/completion",
                                     {"prompt": [1, 2]}, 10, self.records, "none", "t", "c")
        self.assertEqual(result["status"], 200)
        self.assertEqual(runner.validate_completion_counts(
            result["json"], 64, cold=True, expected_decode=16), [])

    def test_completion_error_200_body(self):
        FakeConfig.completion_mode = "error_200"
        result = runner.http_request(self.fake.base_url, "POST", "/completion",
                                     {"prompt": [1]}, 10, self.records, "none", "t", "c")
        self.assertEqual(result["status"], 200)
        self.assertTrue(runner.response_is_error(result)[0])

    def test_completion_error_500_body(self):
        FakeConfig.completion_mode = "error_500"
        result = runner.http_request(self.fake.base_url, "POST", "/completion",
                                     {"prompt": [1]}, 10, self.records, "none", "t", "c")
        self.assertEqual(result["status"], 500)
        self.assertTrue(runner.response_is_error(result)[0])

    def test_slot_restore_error_200_body(self):
        FakeConfig.slot_mode = "error_200"
        result = runner.http_request(self.fake.base_url, "POST", "/slots/0?action=restore",
                                     {"filename": "x.bin"}, 10, self.records, "none", "t", "r")
        self.assertEqual(result["status"], 200)
        failed, detail = runner.response_is_error(result)
        self.assertTrue(failed)
        self.assertIn("canned slot 200 error", detail)
        _, failures = runner.parse_slot_counts(result, "restore")
        self.assertTrue(any("restore failed" in failure for failure in failures))

    def test_recorded_on_disk(self):
        runner.http_request(self.fake.base_url, "POST", "/completion",
                            {"prompt": [1]}, 10, self.records, "none", "t", "c")
        rows = [json.loads(line) for line in self.records.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["phase"], "c")
        self.assertIn("telemetry_before", rows[0])


class TestFrozenPrompts(unittest.TestCase):
    def test_freeze_and_reuse(self):
        self.fake = FakeServer()
        self.tmp = tempfile.TemporaryDirectory()
        try:
            output = Path(self.tmp.name) / "out"
            output.mkdir()
            records = output / "requests.jsonl"
            args = minimal_args(prompt_tokens=64, suffix_tokens=8)
            frozen = runner.create_frozen_prompts(
                self.fake.base_url, args, records, output, "case", None)
            self.assertEqual(len(frozen["prefix_token_ids"]), 1000)
            self.assertEqual(len(frozen["suffix_token_ids"]), 8)
            self.assertTrue((output / "prompts.json").is_file())
            reused = runner.create_frozen_prompts(
                self.fake.base_url, args, records, output, "case", frozen)
            self.assertIs(reused, frozen)

            FakeConfig.token_ids = list(range(1000, 2000))  # tokenizer changed
            with self.assertRaisesRegex(RuntimeError, "not frozen across modes"):
                runner.create_frozen_prompts(
                    self.fake.base_url, args, records, output, "case", frozen)
        finally:
            self.fake.close()
            self.tmp.cleanup()


class TestLogDelta(unittest.TestCase):
    def test_delta_evidence_and_offsets(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            output = Path(tmp.name)
            args = minimal_args()
            instance = runner.ServerInstance(args, output, "case", [])
            instance.log_path.write_text(
                "started\nadaptive remote attention: profile=long, ctx=512, backend=off\n"
                "mixed KV layer=0 device=Vulkan0 type_k=q4_0 type_v=q4_0 domain=rotated\n",
                encoding="utf-8")
            instance.log_offset = 0
            first = instance.read_log_delta()
            evidence = instance.log_evidence(first)
            self.assertTrue(any("backend=off" in line for line in evidence))
            self.assertTrue(any("mixed kv layer=" in line.lower() for line in evidence))
            with instance.log_path.open("a", encoding="utf-8") as handle:
                handle.write("adaptive context transition complete: mtp, n_ctx=1024\n")
            second = instance.read_log_delta()
            self.assertNotIn("mixed kv layer", second)
            self.assertTrue(any("transition complete" in line for line in instance.log_evidence(second)))
        finally:
            tmp.cleanup()


class TestHandoffCounts(unittest.TestCase):
    def test_handoff_reuses_prefix(self):
        body = {"timings": {"cache_n": 256, "prompt_n": 444, "predicted_n": 16}, "truncated": False}
        self.assertEqual(runner.validate_handoff_counts(body, 300, 700, 16), [])

    def test_handoff_full_prefix_reuse(self):
        body = {"timings": {"cache_n": 300, "prompt_n": 400, "predicted_n": 16}, "truncated": False}
        self.assertEqual(runner.validate_handoff_counts(body, 300, 700, 16), [])

    def test_handoff_recomputation_fails(self):
        body = {"timings": {"cache_n": 0, "prompt_n": 700, "predicted_n": 16}, "truncated": False}
        failures = runner.validate_handoff_counts(body, 300, 700, 16)
        self.assertTrue(any("cache_n>0" in failure for failure in failures))

    def test_handoff_cache_beyond_prefix_fails(self):
        body = {"timings": {"cache_n": 512, "prompt_n": 188, "predicted_n": 16}, "truncated": False}
        failures = runner.validate_handoff_counts(body, 300, 700, 16)
        self.assertTrue(any("exceeds the common prefix" in failure for failure in failures))

    def test_handoff_sum_mismatch_fails(self):
        body = {"timings": {"cache_n": 256, "prompt_n": 400, "predicted_n": 16}, "truncated": False}
        failures = runner.validate_handoff_counts(body, 300, 700, 16)
        self.assertTrue(any("!= submitted prompt" in failure for failure in failures))


class TestSaveRestoreParity(unittest.TestCase):
    def test_parity_ok(self):
        self.assertEqual(runner.validate_save_restore_parity(
            {"n_saved": 64, "n_written": 1234}, {"n_restored": 64, "n_read": 1234}), [])

    def test_token_mismatch(self):
        failures = runner.validate_save_restore_parity(
            {"n_saved": 64, "n_written": 1234}, {"n_restored": 63, "n_read": 1234})
        self.assertTrue(any("n_restored(63)" in failure for failure in failures))

    def test_byte_mismatch(self):
        failures = runner.validate_save_restore_parity(
            {"n_saved": 64, "n_written": 1234}, {"n_restored": 64, "n_read": 1200})
        self.assertTrue(any("n_read(1200)" in failure for failure in failures))


class TestMixedLayout(unittest.TestCase):
    LOG_OK = ("mixed KV layer=0 device=Vulkan0 type_k=q4_0 type_v=q4_0 domain=standard-hadamard\n"
              "mixed KV layer=1 device=Vulkan0 type_k=q4_0 type_v=q4_0 domain=standard-hadamard\n")

    def test_parse_and_validate_ok(self):
        layout = runner.parse_mixed_layout(self.LOG_OK)
        self.assertEqual(len(layout), 2)
        self.assertEqual(runner.validate_mixed_layout(layout, 2, "q4_0"), [])

    def test_wrong_layer_count(self):
        layout = runner.parse_mixed_layout("mixed KV layer=0 device=Vulkan0 type_k=q4_0 "
                                           "type_v=q4_0 domain=standard-hadamard\n")
        failures = runner.validate_mixed_layout(layout, 2, "q4_0")
        self.assertTrue(any("1 unique layers" in failure for failure in failures))

    def test_wrong_format(self):
        layout = runner.parse_mixed_layout("mixed KV layer=0 device=Vulkan0 type_k=q8_0 "
                                           "type_v=q4_0 domain=standard-hadamard\n")
        failures = runner.validate_mixed_layout(layout, 1, "q4_0")
        self.assertTrue(any("types q8_0/q4_0" in failure for failure in failures))

    def test_wrong_device(self):
        layout = runner.parse_mixed_layout("mixed KV layer=0 device=CUDA0 type_k=q4_0 "
                                           "type_v=q4_0 domain=standard-hadamard\n")
        failures = runner.validate_mixed_layout(layout, 1, "q4_0")
        self.assertTrue(any("expected Vulkan0" in failure for failure in failures))

    def test_wrong_domain(self):
        layout = runner.parse_mixed_layout("mixed KV layer=0 device=Vulkan0 type_k=q4_0 "
                                           "type_v=q4_0 domain=original\n")
        failures = runner.validate_mixed_layout(layout, 1, "q4_0")
        self.assertTrue(any("domain is original" in failure for failure in failures))

    def test_mixed_domains_fail(self):
        layout = runner.parse_mixed_layout(
            "mixed KV layer=0 device=Vulkan0 type_k=q4_0 type_v=q4_0 domain=standard-hadamard\n"
            "mixed KV layer=1 device=Vulkan0 type_k=q4_0 type_v=q4_0 domain=standard-original\n")
        failures = runner.validate_mixed_layout(layout, 2, "q4_0")
        self.assertTrue(any("domain is standard-original" in failure for failure in failures))


class TestAcceptanceState(unittest.TestCase):
    def test_complete_with_evidence(self):
        evidence = {"configured": True, "found": True, "matched_lines": ["x"]}
        self.assertEqual(runner.acceptance_state([], evidence), "complete")

    def test_pending_without_evidence(self):
        evidence = {"configured": False, "found": False, "matched_lines": []}
        self.assertEqual(runner.acceptance_state([], evidence), "pending_evidence")
        self.assertNotEqual(runner.acceptance_state([], evidence), "complete")

    def test_pending_when_no_evidence_object(self):
        self.assertEqual(runner.acceptance_state([]), "complete")  # non-B scenarios

    def test_failed_when_evidence_configured_but_missing(self):
        evidence = {"configured": True, "found": False, "matched_lines": []}
        self.assertEqual(runner.acceptance_state([], evidence), "failed")

    def test_failed_with_validation_failures(self):
        evidence = {"configured": True, "found": True, "matched_lines": ["x"]}
        self.assertEqual(runner.acceptance_state(["boom"], evidence), "failed")


class TestTargetOnly(unittest.TestCase):
    def test_draft_acceptance_detected(self):
        violations = runner.target_only_violations(
            "draft acceptance = 0.50000 (4 accepted / 8 generated), mean len = 2.00\n")
        self.assertEqual(len(violations), 1)
        self.assertIn("draft acceptance", violations[0])

    def test_no_violations(self):
        self.assertEqual(runner.target_only_violations(
            "Vulkan-layers=1/16\nmixed KV layer=0 device=Vulkan0 type_k=q4_0 type_v=q4_0\n"), [])


class TestHandoffFakeIntegration(unittest.TestCase):
    def test_handoff_preserved_prefix_pass(self):
        FakeConfig.completion = {
            "content": "x", "tokens": [1],
            "timings": {"cache_n": 256, "prompt_n": 444, "predicted_n": 16},
            "truncated": False,
        }
        fake = FakeServer()
        tmp = tempfile.TemporaryDirectory()
        try:
            records = Path(tmp.name) / "requests.jsonl"
            result = runner.http_request(fake.base_url, "POST", "/completion", {"prompt": [1]},
                                         10, records, "none", "t", "handoff")
            self.assertEqual(runner.validate_handoff_counts(result["json"], 300, 700, 16), [])
        finally:
            fake.close()
            tmp.cleanup()

    def test_handoff_recomputation_fail(self):
        FakeConfig.completion = {
            "content": "x", "tokens": [1],
            "timings": {"cache_n": 0, "prompt_n": 700, "predicted_n": 16},
            "truncated": False,
        }
        fake = FakeServer()
        tmp = tempfile.TemporaryDirectory()
        try:
            records = Path(tmp.name) / "requests.jsonl"
            result = runner.http_request(fake.base_url, "POST", "/completion", {"prompt": [1]},
                                         10, records, "none", "t", "handoff")
            failures = runner.validate_handoff_counts(result["json"], 300, 700, 16)
            self.assertTrue(any("cache_n>0" in failure for failure in failures))
        finally:
            fake.close()
            tmp.cleanup()


class TestManifestSerialization(unittest.TestCase):
    """Regression for the parent-reported entrypoint failure: run.json
    manifest serialization crashed on PosixPath before any server launch
    (TypeError: Object of type PosixPath is not JSON serializable)."""

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
        args = runner.build_parser().parse_args([])
        with tempfile.TemporaryDirectory() as tmp:
            config = runner.configuration_from_args(args)
            target = Path(tmp) / "run.json"
            runner.write_json(target, {"configuration": config, "probe": True})
            parsed = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(parsed["probe"], True)
        self.assertIsInstance(parsed["configuration"]["server_binary"], str)
        self.assertIsInstance(parsed["configuration"]["model"], str)
        self.assertIsNone(parsed["configuration"]["output_dir"])
        self._assert_no_paths(parsed, "manifest")

    def test_resolved_paths_preserved(self):
        binary = Path("/bin/true").resolve()
        args = runner.build_parser().parse_args(
            ["--server-binary", str(binary), "--model", str(binary)])
        config = runner.configuration_from_args(args)
        self.assertEqual(config["server_binary"], str(binary))
        self.assertEqual(config["model"], str(binary))

    def test_unknown_object_still_fails_loudly(self):
        with self.assertRaises(TypeError):
            runner.write_json(Path(tempfile.mkdtemp()) / "x.json", {"bad": {1, 2}})

    def test_path_only_default(self):
        with self.assertRaises(TypeError):
            runner.json_default({"not": "a path"})


class _FakeServerInstance(runner.ServerInstance):
    """Recording stub: constructor is harmless (no spawn); start() records the
    launch argv and writes the empty log the scenarios read back."""

    started: list = []

    def start(self) -> None:
        _FakeServerInstance.started.append(self.command)
        self.log_path.write_text("", encoding="utf-8")


class TestEntrypointFakeEndToEnd(unittest.TestCase):
    """main() up to server launch with a fresh empty output dir: the manifest
    must serialize and the launch path must be reached without a real server
    or GPU (ServerInstance.start is stubbed)."""

    def test_main_reaches_launch_without_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "results"
            out_dir.mkdir()
            port = 20000 + (os.getpid() % 2000)
            argv = ["test-mixed-kv-transition.py",
                    "--model", "/bin/true", "--server-binary", "/bin/true",
                    "--output-dir", str(out_dir), "--scenarios", "A",
                    "--remote-kv-types", "q4_0", "--port", str(port),
                    "--telemetry-device", "none"]
            with unittest.mock.patch.object(sys, "argv", argv), \
                    unittest.mock.patch.object(runner, "ServerInstance",
                                               _FakeServerInstance):
                _FakeServerInstance.started.clear()
                exit_code = runner.main()
            self.assertEqual(exit_code, 1)  # scenario fails on dead HTTP; no crash
            run_json = out_dir / "run.json"
            self.assertTrue(run_json.is_file())
            manifest = json.loads(run_json.read_text(encoding="utf-8"))
            self.assertIsInstance(manifest["configuration"]["server_binary"], str)
            self.assertTrue(_FakeServerInstance.started,
                            "server launch path was not reached")
            self.assertTrue((out_dir / "summary.json").is_file())
            self.assertTrue((out_dir / "cases.jsonl").is_file())


class TestProcessCleanup(unittest.TestCase):
    @staticmethod
    def _proc_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def test_stop_server_kills_only_its_group(self):
        marker = subprocess.Popen(["sleep", "30"])
        try:
            python = ("import subprocess,time,sys;"
                      "p=subprocess.Popen(['sleep','30']);"
                      "print(p.pid, flush=True);"
                      "time.sleep(30)")
            server = subprocess.Popen([sys.executable, "-c", python],
                                      stdout=subprocess.PIPE, text=True,
                                      start_new_session=True)
            child_pid = int(server.stdout.readline().strip())
            server.stdout.close()
            self.assertTrue(self._proc_alive(child_pid))
            self.assertTrue(self._proc_alive(marker.pid))
            runner.stop_server(server, timeout=5)
            self.assertIsNotNone(server.returncode)
            deadline = time.monotonic() + 5
            while self._proc_alive(child_pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(self._proc_alive(child_pid),
                             "nested child survived the group kill")
            self.assertTrue(self._proc_alive(marker.pid),
                            "unrelated process was killed by stop_server")
        finally:
            marker.kill()
            marker.wait()

    def test_stop_server_already_exited(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        runner.stop_server(proc, timeout=2)  # must not raise
        self.assertEqual(proc.returncode, 0)

    def test_stop_server_natural_parent_death_with_grandchild(self):
        """The parent dies on its own while a same-group grandchild survives;
        stop_server must still clean the group and never touch others."""
        marker = subprocess.Popen(["sleep", "30"])
        try:
            python = ("import subprocess;"
                      "p=subprocess.Popen(['sleep','30']);"
                      "print(p.pid, flush=True)")
            server = subprocess.Popen([sys.executable, "-c", python],
                                      stdout=subprocess.PIPE, text=True,
                                      start_new_session=True)
            child_pid = int(server.stdout.readline().strip())
            server.stdout.close()
            self.assertEqual(server.wait(timeout=5), 0)  # natural parent death
            self.assertTrue(self._proc_alive(child_pid),
                            "grandchild should outlive its parent")
            runner.stop_server(server, timeout=2)
            deadline = time.monotonic() + 5
            while self._proc_alive(child_pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(self._proc_alive(child_pid),
                             "surviving grandchild was not cleaned up")
            self.assertTrue(self._proc_alive(marker.pid),
                            "unrelated process was killed by stop_server")
        finally:
            marker.kill()
            marker.wait()

    def test_server_command_remote_flags(self):
        args = minimal_args(remote_kv_types=["q8_0"])
        instance = runner.ServerInstance(args, Path(tempfile.mkdtemp()), "x", [
            "--remote-attn", "vulkan:0", "--remote-attn-layers", "1",
            "--remote-attn-cache-type-k", "q8_0",
            "--remote-attn-cache-type-v", "q8_0", "--spec-type", "none",
            "--cache-ram", "0", "--ctx-checkpoints", "1",
        ])
        command = " ".join(instance.command)
        self.assertIn("--cache-type-k kvarn4", command)
        self.assertIn("--remote-attn-cache-type-k q8_0", command)
        self.assertIn("--slot-save-path", command)
        self.assertIn("--cache-ram 0", command)
        self.assertIn("--ctx-checkpoints 1", command)
        self.assertNotIn("--kv-tail-tokens 1024", command)

    def test_server_command_b_keeps_prompt_cache(self):
        """Scenario B relies on the RAM prompt cache for the adaptive prefix
        capture; the global command must not force --cache-ram 0 on it. The
        profile mapping must be --ctx-size = LONG (high) and --ctx-size-mtp =
        MTP (local), with explicit per-tier KVarN4."""
        args = minimal_args(remote_kv_types=["q4_0"])
        instance = runner.ServerInstance(args, Path(tempfile.mkdtemp()), "b", [
            "--spec-type", "draft-mtp",
            "--ctx-size", "1024", "--ctx-size-mtp", "512",
            "--mtp-max-tokens", "512",
            "--cache-type-k-m", "kvarn4", "--cache-type-v-m", "kvarn4",
            "--cache-type-k-l", "kvarn4", "--cache-type-v-l", "kvarn4",
            "--remote-attn", "vulkan:0",
            "--remote-attn-layers", "1", "--remote-attn-min-context", "513",
            "--remote-attn-cache-type-k", "q4_0",
            "--remote-attn-cache-type-v", "q4_0",
            "--cache-ram", "256", "--ctx-checkpoints", "1",
        ])
        command = " ".join(instance.command)
        self.assertIn("--cache-ram 256", command)
        self.assertNotIn("--cache-ram 0", command)
        self.assertIn("--ctx-checkpoints 1", command)
        self.assertIn("--ctx-size 1024", command)
        self.assertIn("--ctx-size-mtp 512", command)
        self.assertIn("--mtp-max-tokens 512", command)
        self.assertIn("--cache-type-k-m kvarn4", command)
        self.assertIn("--cache-type-v-m kvarn4", command)
        self.assertIn("--cache-type-k-l kvarn4", command)
        self.assertIn("--cache-type-v-l kvarn4", command)
        self.assertNotIn("--ctx-size 512", command)
        self.assertNotIn("--ctx-size-mtp 1024", command)

    def test_server_command_a_disables_prompt_cache(self):
        args = minimal_args(remote_kv_types=["q4_0"])
        instance = runner.ServerInstance(args, Path(tempfile.mkdtemp()), "a", [
            "--spec-type", "none", "--cache-ram", "0", "--ctx-checkpoints", "1",
        ])
        command = " ".join(instance.command)
        self.assertIn("--cache-ram 0", command)
        self.assertNotIn("--cache-ram 256", command)


if __name__ == "__main__":
    unittest.main(verbosity=2)