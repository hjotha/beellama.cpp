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
import struct
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
        # Scenario D defaults, mirrored from build_parser() so the split
        # profile tests exercise the real values.
        "split_p_local": 256, "split_p_high": 512,
        "split_ctx_local": 1024, "split_ctx_high": 2048,
        "split_tokens_local": 192, "split_full_layers": 8,
        "gate_ctx": 513, "disk_tmpdir": Path("/home/hjotha/f4-disk-tmp"),
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


class TestFrozenPromptRequirements(unittest.TestCase):
    def test_d_default_freeze_covers_d0b_d3_d4b_d5(self):
        args = minimal_args(scenarios=["A", "B", "C", "D"])
        required = runner.required_frozen_prefix_tokens(args)
        self.assertGreaterEqual(required, args.split_p_high + 256)
        self.assertGreaterEqual(required, args.split_ctx_local + 256)
        self.assertGreaterEqual(required, args.split_tokens_local + args.suffix_tokens)
        self.assertGreaterEqual(required, args.split_tokens_local + 256)
        self.assertEqual(required, 1280)

    def test_large_d5_local_context_drives_freeze_size(self):
        args = minimal_args(scenarios=["D"], split_p_high=10000,
                            split_ctx_local=12000, split_tokens_local=9000)
        self.assertEqual(runner.required_frozen_prefix_tokens(args), 12256)

    def test_non_d_freeze_length_is_unchanged(self):
        args = minimal_args(scenarios=["A", "B", "C"], prompt_tokens=64,
                            prompt_short=32, prompt_long=48, suffix_tokens=8,
                            split_p_high=10000, split_ctx_local=12000)
        self.assertEqual(runner.required_frozen_prefix_tokens(args), 72)

    def test_exact_frozen_prefix_guard_rejects_undersized_source(self):
        with self.assertRaisesRegex(RuntimeError, "has 2 IDs, requires 3"):
            runner.frozen_prefix_ids({"prefix_token_ids": [1, 2]}, 3, "D5")

    def test_create_frozen_prompts_freezes_d_sized_source(self):
        self.fake = FakeServer()
        self.tmp = tempfile.TemporaryDirectory()
        old_ids = FakeConfig.token_ids
        try:
            FakeConfig.token_ids = list(range(2000))
            output = Path(self.tmp.name) / "out"
            output.mkdir()
            args = minimal_args(scenarios=["D"], split_p_high=1024,
                                split_ctx_local=1024, split_tokens_local=192)
            frozen = runner.create_frozen_prompts(
                self.fake.base_url, args, output / "requests.jsonl", output, "D", None)
            self.assertGreaterEqual(len(frozen["prefix_token_ids"]),
                                    runner.required_frozen_prefix_tokens(args))
        finally:
            FakeConfig.token_ids = old_ids
            self.fake.close()
            self.tmp.cleanup()


class TestFrozenPromptRequirements(unittest.TestCase):
    def test_d_default_freeze_covers_d0b_d3_d4b_and_d5(self):
        args = minimal_args(scenarios=["A", "B", "C", "D"])
        required = runner.required_frozen_prefix_tokens(args)
        self.assertGreaterEqual(required, args.split_p_high + 256)
        self.assertGreaterEqual(required, args.split_ctx_local + 256)
        self.assertGreaterEqual(required, args.split_tokens_local + args.suffix_tokens)
        self.assertGreaterEqual(required, args.split_tokens_local + 256)
        self.assertEqual(required, 1280)

    def test_large_d5_size_controls_freeze(self):
        args = minimal_args(scenarios=["D"], split_p_high=10000,
                            split_ctx_local=12000, split_tokens_local=9000)
        self.assertEqual(runner.required_frozen_prefix_tokens(args), 12256)

    def test_non_d_freeze_requirement_is_unchanged(self):
        args = minimal_args(scenarios=["A", "B", "C"], prompt_tokens=64,
                            prompt_short=32, prompt_long=48, suffix_tokens=8,
                            split_p_high=10000, split_ctx_local=12000)
        self.assertEqual(runner.required_frozen_prefix_tokens(args), 72)

    def test_exact_prefix_guard_rejects_undersized_frozen_ids(self):
        with self.assertRaisesRegex(RuntimeError, "has 2 IDs, requires 3"):
            runner.frozen_prefix_ids({"prefix_token_ids": [1, 2]}, 3, "D5")

    def test_frozen_prompt_creation_meets_d_requirement(self):
        self.fake = FakeServer()
        self.tmp = tempfile.TemporaryDirectory()
        old_ids = FakeConfig.token_ids
        try:
            FakeConfig.token_ids = list(range(2000))
            output = Path(self.tmp.name) / "out"
            output.mkdir()
            args = minimal_args(scenarios=["D"], split_p_high=1024,
                                split_ctx_local=1024, split_tokens_local=192)
            frozen = runner.create_frozen_prompts(
                self.fake.base_url, args, output / "requests.jsonl", output, "D", None)
            self.assertGreaterEqual(len(frozen["prefix_token_ids"]),
                                    runner.required_frozen_prefix_tokens(args))
        finally:
            FakeConfig.token_ids = old_ids
            self.fake.close()
            self.tmp.cleanup()


class TestTokenIDHash(unittest.TestCase):
    def test_extension_prefix_hash_uses_little_endian_int32(self):
        tokens = [11, 12, 0x123456]
        expected = runner.sha256_bytes(struct.pack("<3i", *tokens))
        self.assertEqual(runner.token_ids_sha256(tokens), expected)

    def test_extension_prefix_hash_rejects_invalid_ids(self):
        for tokens in ([True], [-1], [0x80000000]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                runner.token_ids_sha256(tokens)


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


class TestFixturePreconditionEmission(unittest.TestCase):
    def test_failed_fixture_precondition_is_persisted_and_counted(self):
        recorded = []
        runs = []

        def record_leg(leg, state, **fields):
            entry = {"leg": leg, "state": state, **fields}
            recorded.append(entry)
            return entry

        output = Path("/campaign/out")
        own_dir = output / "slot-save-D5"
        own_file = own_dir / "snap-D5.bin"
        failures = ["D5: fixture prefill does not fit source context"]
        entry = runner.record_fixture_precondition_failure(
            record_leg, runs, leg_name="D5-insufficient-capacity", p=1024,
            ctx=1024, source_ctx=2048, output_dir=output, own_dir=own_dir,
            own_file=own_file, n_prefill=1280, bound_name="dest_ctx", bound=1024,
            own_info={"source_p": 1024}, failures=failures,
            expectation="fixture occupancy exceeds destination capacity")
        self.assertEqual(runs, [entry])
        self.assertEqual(recorded, [entry])
        self.assertEqual(entry["state"], "failed")
        self.assertEqual(entry["validation_failures"], failures)
        self.assertEqual(entry["source_snapshot_path"], str(own_file))
        self.assertEqual(entry["source_log_path"], str(output / "server-D5-insufficient-capacity-fixture.log"))
        self.assertEqual(entry["restore_log_path"], str(output / "server-D5-insufficient-capacity-restore.log"))
        self.assertTrue(entry["precondition_failure"])

    def test_both_early_fixture_failure_branches_use_counted_result(self):
        source = (TOOLS_DIR / "test-mixed-kv-transition.py").read_text()
        self.assertGreaterEqual(source.count("record_fixture_precondition_failure("), 3)
        helper = source.split("def record_fixture_precondition_failure", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("runs.append(entry)", helper)


class TestRunAggregation(unittest.TestCase):
    """The exit code gate. Regression for a real false positive: the previous
    aggregator counted only failed/pending_evidence, so 'inconclusive' and
    'not_executed' legs exited 0. Every state other than 'complete' must make
    the aggregate incomplete."""

    def test_all_complete_is_acceptable(self):
        agg = runner.aggregate_run_states([
            {"leg": "D1", "state": "complete"},
            {"leg": "D2", "state": "complete"},
        ])
        self.assertFalse(agg["incomplete"])
        self.assertEqual(agg["complete"], 2)
        self.assertEqual(agg["unacceptable_runs"], 0)

    def test_failed_is_unacceptable(self):
        agg = runner.aggregate_run_states([{"leg": "D1", "state": "failed"}])
        self.assertTrue(agg["incomplete"])
        self.assertEqual(agg["failed"], 1)
        self.assertEqual(agg["unacceptable_runs"], 1)

    def test_pending_evidence_is_unacceptable(self):
        agg = runner.aggregate_run_states([{"leg": "B1", "state": "pending_evidence"}])
        self.assertTrue(agg["incomplete"])
        self.assertEqual(agg["pending_evidence"], 1)

    def test_inconclusive_is_unacceptable(self):
        """D6 was inconclusive and the old aggregator let it exit 0."""
        agg = runner.aggregate_run_states([{"leg": "D6", "state": "inconclusive"}])
        self.assertTrue(agg["incomplete"])
        self.assertEqual(agg["inconclusive"], 1)
        self.assertIn("D6(inconclusive)", agg["unacceptable_legs"])

    def test_not_executed_is_unacceptable(self):
        """Every negative leg recorded as not_executed must force non-zero."""
        agg = runner.aggregate_run_states([
            {"leg": "D2-different-p", "state": "not_executed"},
            {"leg": "D3-above-p", "state": "not_executed"},
            {"leg": "D4a-split-local-to-pure", "state": "not_executed"},
            {"leg": "D4b-occupied-overflow-to-pure", "state": "not_executed"},
            {"leg": "D5-insufficient-capacity", "state": "not_executed"},
        ])
        self.assertTrue(agg["incomplete"])
        self.assertEqual(agg["not_executed"], 5)
        self.assertEqual(agg["unacceptable_runs"], 5)

    def test_one_non_complete_among_completes_is_incomplete(self):
        agg = runner.aggregate_run_states([
            {"leg": "D1", "state": "complete"},
            {"leg": "D3", "state": "complete"},
            {"leg": "D6", "state": "inconclusive"},
        ])
        self.assertTrue(agg["incomplete"])
        self.assertEqual(agg["complete"], 2)
        self.assertEqual(agg["inconclusive"], 1)

    def test_missing_state_is_unacceptable(self):
        agg = runner.aggregate_run_states([{"leg": "D1"}])
        self.assertTrue(agg["incomplete"])
        self.assertEqual(agg["missing_or_invalid_state"], 1)

    def test_unknown_state_is_unacceptable(self):
        agg = runner.aggregate_run_states([{"leg": "D1", "state": "passed"}])
        self.assertTrue(agg["incomplete"])
        self.assertEqual(agg["missing_or_invalid_state"], 1)
        self.assertNotEqual(agg["complete"], 1)

    def test_empty_run_list_is_complete_but_zero(self):
        agg = runner.aggregate_run_states([])
        self.assertFalse(agg["incomplete"])
        self.assertEqual(agg["complete"], 0)

    def test_run_state_label_prefers_leg_then_case_id(self):
        self.assertEqual(runner.run_state_label({"leg": "D1", "case_id": "x"}), "D1")
        self.assertEqual(runner.run_state_label({"case_id": "A-q4_0"}), "A-q4_0")
        self.assertEqual(runner.run_state_label({}), "<unlabeled>")


class TestScenarioDArgumentGuards(unittest.TestCase):
    """The P guards must reject a run that could not prove anything, BEFORE any
    server is started."""

    def _parser(self):
        return runner.build_parser()

    def test_split_options_exist_with_safe_defaults(self):
        args = self._parser().parse_args([])
        self.assertEqual(args.split_p_local % 256, 0)
        self.assertEqual(args.split_p_high % 256, 0)
        self.assertGreater(args.split_p_local, 0)
        self.assertGreater(args.split_p_high, 0)
        self.assertLess(args.split_p_local, args.split_ctx_local)
        self.assertLess(args.split_p_high, args.split_ctx_high)
        self.assertLess(args.split_tokens_local, args.split_p_local)
        self.assertLess(args.split_tokens_local, args.split_p_high)
        self.assertLess(args.split_ctx_local, args.split_ctx_high)

    def test_d5_local_destination_must_exceed_high_p(self):
        runner.validate_d5_destination_capacity(2048, 1024)
        with self.assertRaisesRegex(ValueError, "D5 uses the high-P split"):
            runner.validate_d5_destination_capacity(1024, 1024)
        with self.assertRaisesRegex(ValueError, "D5 uses the high-P split"):
            runner.validate_d5_destination_capacity(768, 1024)

    def test_disk_tmpdir_default_is_not_tmpfs(self):
        args = self._parser().parse_args([])
        self.assertNotIn(str(args.disk_tmpdir), ("/tmp", "/var/tmp", "/run"))

    def test_scenarios_accepts_d(self):
        names = runner.parse_names("D", "--scenarios", {"A", "B", "C", "D"})
        self.assertEqual(names, ["D"])

    def test_split_profile_requires_all_full_layers_remote(self):
        """The split placement refuses a partial remote layer count, so the
        split profile must pass exactly the full-layer count."""
        args = minimal_args(split_full_layers=8)
        split = runner.split_profile(args, "q4_0", split=True)
        command = " ".join(split)
        self.assertIn("--remote-attn-layers 8", command)
        self.assertIn("--cache-ram 0", command)

    def test_non_split_profile_omits_split_only_flags(self):
        """--remote-attn-min-context requires an adaptive context and kills the
        server, so a non-split leg must not pass it."""
        args = minimal_args()
        profile = runner.split_profile(args, "q4_0", split=False)
        command = " ".join(profile)
        self.assertNotIn("--remote-attn-min-context", command)
        self.assertNotIn("--remote-attn", command)
        self.assertIn("--cache-ram 0", command)

    def test_pure_destination_clears_inherited_legacy_split_override(self):
        import inspect
        source = inspect.getsource(runner.scenario_d)
        self.assertIn('env["LLAMA_KV_POSITION_SPLIT_P"] = ""', source)


class TestLocateChecksummedPayloadByte(unittest.TestCase):
    """A corruption must land provably inside a RECOGNIZED region. The first D6
    run flipped a blind len//2 byte, landed outside any verified region, and the
    frame was accepted -- proving nothing. The locator must refuse unknown
    framings instead of guessing."""

    def test_unknown_framing_returns_none(self):
        path = Path(tempfile.mkdtemp()) / "junk.bin"
        path.write_bytes(b"NOTAFRAME" + bytes(64))
        self.assertIsNone(runner.locate_checksummed_payload_byte(path))

    def test_short_file_returns_none(self):
        path = Path(tempfile.mkdtemp()) / "tiny.bin"
        path.write_bytes(runner.STATE_SEQ_MAGIC)
        self.assertIsNone(runner.locate_checksummed_payload_byte(path))

    def test_missing_file_returns_none(self):
        path = Path(tempfile.mkdtemp()) / "absent.bin"
        self.assertIsNone(runner.locate_checksummed_payload_byte(path))

    def test_state_seq_magic_locates_header_byte(self):
        path = Path(tempfile.mkdtemp()) / "state.bin"
        path.write_bytes(runner.STATE_SEQ_MAGIC + bytes(64))
        target = runner.locate_checksummed_payload_byte(path)
        self.assertIsNotNone(target)
        offset, layer_id, range_id, region = target
        # The offset must be INSIDE the 4-byte magic field, i.e. a header byte
        # the reader validates.
        self.assertTrue(0 <= offset < 4)
        self.assertEqual(region, "state_seq_header_magic")
        self.assertEqual(layer_id, -1)
        self.assertEqual(range_id, -1)

    def test_state_seq_magic_matches_llama_h_definition(self):
        """include/llama.h:43,48 -> GGSQ 0x67677371 little-endian on disk."""
        self.assertEqual(runner.STATE_SEQ_MAGIC, bytes((0x71, 0x73, 0x67, 0x67)))
        self.assertEqual(runner.STATE_SEQ_MAGIC, b"qsgg")

    def test_mixed_v2_magic_is_kms2(self):
        self.assertEqual(runner.MIXED_STATE_MAGIC_V2, b"KMS2")
        self.assertEqual(runner.MIXED_STATE_MAGIC_V1, b"KMS1")

    def test_mixed_v2_garbage_after_magic_returns_none(self):
        """A KMS2 header whose shape does not validate must return None, never
        a guessed payload offset."""
        path = Path(tempfile.mkdtemp()) / "mixed.bin"
        path.write_bytes(runner.MIXED_STATE_MAGIC_V2 + bytes(4096))
        self.assertIsNone(runner.locate_checksummed_payload_byte(path))


class TestCompareCompletionProbabilities(unittest.TestCase):
    """§3.6: compare next logits, never approve on text alone."""

    def _probs(self, logprob):
        return [{"id": 7, "token": "x", "logprob": logprob,
                 "top_logprobs": [{"id": 7, "token": "x", "logprob": logprob},
                                  {"id": 8, "token": "y", "logprob": logprob - 1.0}]}]

    def test_identical_is_equal(self):
        self.assertTrue(runner.compare_completion_probabilities(
            self._probs(-1.5), self._probs(-1.5)))

    def test_within_tolerance_is_equal(self):
        self.assertTrue(runner.compare_completion_probabilities(
            self._probs(-1.5), self._probs(-1.5 + 1e-12)))

    def test_beyond_tolerance_is_not_equal(self):
        self.assertFalse(runner.compare_completion_probabilities(
            self._probs(-1.5), self._probs(-1.4)))

    def test_different_token_id_is_not_equal(self):
        left = self._probs(-1.5)
        right = self._probs(-1.5)
        right[0]["id"] = 9
        self.assertFalse(runner.compare_completion_probabilities(left, right))

    def test_none_inputs_are_not_equal(self):
        self.assertFalse(runner.compare_completion_probabilities(None, None))

    def test_empty_is_not_equal(self):
        """Empty evidence must never be read as 'unchanged'."""
        self.assertFalse(runner.compare_completion_probabilities([], []))

    def test_length_mismatch_is_not_equal(self):
        left = self._probs(-1.5)
        right = left + self._probs(-2.0)
        self.assertFalse(runner.compare_completion_probabilities(left, right))

    def test_top_logprobs_mismatch_is_not_equal(self):
        left = self._probs(-1.5)
        right = self._probs(-1.5)
        right[0]["top_logprobs"][1]["logprob"] = -99.0
        self.assertFalse(runner.compare_completion_probabilities(left, right))


class TestScenarioDNegativeLegTable(unittest.TestCase):
    """The negative leg tuple arity is a real trap: the table has SEVEN
    fields, and a loop that unpacks eight raises ValueError mid-run instead of
    recording the remaining legs. Guard every unpack site against the table
    arity rather than hard-coding one expected string."""

    # Every unpack of negative_legs must bind exactly seven names.
    EXPECTED_UNPACKS = (
        "for leg_name, _p, _c, _exp, _hit, _pre, why in negative_legs:",
        "for leg_name, p, ctx, expect_restore, expect_hit, precond, why in own_fixture_legs:",
        "for leg_name, p, ctx, expect_restore, expect_hit, precond, why in negative_legs:",
    )

    def _source(self):
        import inspect
        return inspect.getsource(runner.scenario_d)

    def test_every_negative_leg_unpack_matches_table_arity(self):
        source = self._source()
        for line in source.splitlines():
            if "negative_legs:" not in line or not line.strip().startswith("for "):
                continue
            self.assertIn(
                line.strip(), self.EXPECTED_UNPACKS,
                f"unpack site disagrees with the 7-field negative_leg table: {line.strip()}")

    def test_table_arity_is_seven(self):
        source = self._source()
        # The table annotation is the single source of truth for arity.
        self.assertIn(
            'negative_legs: list[tuple[str, int, int, bool, bool | None,\n'
            '                              tuple[Any, ...] | None, str]] = [',
            source)

    def test_source_unavailable_branch_packs_all_fields(self):
        """Regression: this branch used to unpack five of six and raise
        UnboundLocalError instead of recording not_executed."""
        source = self._source()
        branch = source.split("source snapshot unavailable")[1][:600]
        self.assertIn("for leg_name, _p, _c, _exp, _hit, _pre, why in negative_legs:",
                      branch)
        self.assertIn('record_leg(leg_name, "not_executed"', branch)

    def test_own_fixture_legs_declare_preconditions(self):
        """D3 and D5 must carry their own fixture and a real occupancy
        precondition; n_ctx is NOT occupancy."""
        source = self._source()
        self.assertIn('("above-p", "P", p_high)', source)
        self.assertIn('("above-dest", "dest_ctx", args.split_ctx_local)', source)
        self.assertIn('("above-pure", "P", p_high, p_high)', source)

    def test_inverse_boundary_legs_and_pure_profiles_are_explicit(self):
        source = self._source()
        self.assertIn('d4a_id = "D4a-split-local-to-pure"', source)
        self.assertIn('("D4b-occupied-overflow-to-pure", 0, args.split_ctx_high', source)
        self.assertIn('split_profile(args, qx, split=source_p != 0)', source)
        self.assertIn('split_profile(args, qx, split=p != 0)', source)
        self.assertIn('split_to_pure_import_p_refusal(refusal, p_high, n_occ)', source)

    def test_d1_is_labelled_split_to_split(self):
        source = self._source()
        self.assertIn('leg1_kind = "split_to_split_below_p"', source)
        self.assertIn('kind="pure_xxl_to_split"', source)


class TestPureKvarnImportPRefusal(unittest.TestCase):
    def test_named_extent_refusal_for_expected_p_is_attributable(self):
        reason = ("error loading sequence state file: pure KVarN import refused: "
                  "snapshot extent 1283 exceeds min(P=1024, capacity=1024)")
        parsed = runner.pure_kvarn_import_p_refusal(reason, 1024)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["snapshot_extent"], 1283)
        self.assertTrue(parsed["extent_exceeds_p"])
        self.assertTrue(parsed["p_matches_expected"])
        self.assertTrue(parsed["p_attributable"])

    def test_pure_destination_above_source_p_requires_actual_extent(self):
        failures = runner.validate_split_to_pure_extent(128, 1024, expect_import=True)
        self.assertEqual(failures, [])
        failures = runner.validate_split_to_pure_extent(1283, 1024, expect_import=True)
        self.assertTrue(any("exceeds source P" in item for item in failures))
        failures = runner.validate_split_to_pure_extent(1283, 1024, expect_import=False)
        self.assertEqual(failures, [])
        parsed = runner.pure_kvarn_import_p_refusal(
            "pure KVarN import refused: snapshot extent 1283 exceeds min(P=0, capacity=0)", 0)
        self.assertTrue(parsed and parsed["p_attributable"])
        self.assertEqual(parsed["snapshot_extent"], 1283)


class TestSplitToPureInverseRefusal(unittest.TestCase):
    def test_native_extent_may_exceed_sealed_occupancy(self):
        reason = ("split-to-pure import refused: snapshot extent 1283 exceeds "
                  "min(P=1024, capacity=4096)")
        parsed = runner.split_to_pure_import_p_refusal(reason, 1024, 1152)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["native_extent"], 1283)
        self.assertEqual(parsed["sealed_occupancy_lower_bound"], 1152)
        self.assertTrue(parsed["extent_covers_sealed_occupancy"])
        self.assertTrue(parsed["p_attributable"])

    def test_wrong_source_p_is_not_attributable(self):
        reason = "split-to-pure import refused: snapshot extent 1283 exceeds min(P=2048, capacity=4096)"
        parsed = runner.split_to_pure_import_p_refusal(reason, 1024, 1152)
        self.assertFalse(parsed["p_attributable"])
        self.assertFalse(parsed["source_p_matches_expected"])

    def test_extent_at_or_below_p_is_not_attributable(self):
        reason = "split-to-pure import refused: snapshot extent 1024 exceeds min(P=1024, capacity=4096)"
        parsed = runner.split_to_pure_import_p_refusal(reason, 1024, 1000)
        self.assertFalse(parsed["extent_exceeds_source_p"])
        self.assertFalse(parsed["p_attributable"])

    def test_capacity_limited_boundary_is_not_attributable(self):
        reason = "split-to-pure import refused: snapshot extent 1283 exceeds min(P=1024, capacity=768)"
        parsed = runner.split_to_pure_import_p_refusal(reason, 1024, 1152)
        self.assertFalse(parsed["limit_is_source_p"])
        self.assertFalse(parsed["p_attributable"])

    def test_extent_below_sealed_occupancy_is_not_attributable(self):
        reason = "split-to-pure import refused: snapshot extent 1100 exceeds min(P=1024, capacity=4096)"
        parsed = runner.split_to_pure_import_p_refusal(reason, 1024, 1152)
        self.assertFalse(parsed["extent_covers_sealed_occupancy"])
        self.assertFalse(parsed["p_attributable"])

    def test_generic_refusal_is_not_attributable(self):
        parsed = runner.split_to_pure_import_p_refusal(
            "split-to-pure import refused: incompatible layer layout", 1024, 1152)
        self.assertIsNone(parsed)

    def test_named_extent_refusal_with_wrong_p_is_not_attributable(self):
        reason = "pure KVarN import refused: snapshot extent 1283 exceeds min(P=1024, capacity=1024)"
        parsed = runner.pure_kvarn_import_p_refusal(reason, 2048)
        self.assertIsNotNone(parsed)
        self.assertFalse(parsed["p_matches_expected"])
        self.assertFalse(parsed["p_attributable"])

    def test_named_refusal_without_extent_above_p_is_not_attributable(self):
        reason = "pure KVarN import refused: snapshot extent 900 exceeds min(P=1024, capacity=1024)"
        parsed = runner.pure_kvarn_import_p_refusal(reason, 1024)
        self.assertIsNotNone(parsed)
        self.assertFalse(parsed["extent_exceeds_p"])
        self.assertFalse(parsed["p_attributable"])

    def test_capacity_limited_or_generic_refusals_are_not_p_evidence(self):
        capacity_limited = runner.pure_kvarn_import_p_refusal(
            "pure KVarN import refused: snapshot extent 1283 exceeds min(P=1024, capacity=768)", 1024)
        self.assertFalse(capacity_limited["p_attributable"])
        generic = runner.pure_kvarn_import_p_refusal(
            "position-split mismatch: snapshot P does not match destination", 1024)
        self.assertIsNone(generic)


class TestSplitProfileVariants(unittest.TestCase):
    def test_pure_source_profile_has_no_remote_attn(self):
        args = minimal_args()
        profile = " ".join(runner.split_profile(args, "q4_0", split=False,
                                                pure_source=True))
        self.assertNotIn("--remote-attn", profile)
        self.assertNotIn("--remote-attn-layers", profile)
        self.assertIn("--cache-ram 0", profile)


class TestFullLogitCapturePlan(unittest.TestCase):
    def test_optional_flag_defaults_off(self):
        self.assertFalse(runner.build_parser().parse_args([]).full_logit_capture)
        self.assertTrue(runner.build_parser().parse_args(["--full-logit-capture"]).full_logit_capture)

    def test_full_logit_bundle_id_is_optional_and_sha256_validated(self):
        parser = runner.build_parser()
        self.assertIsNone(parser.parse_args([]).full_logit_build_id)
        valid = "A" * 64
        args = parser.parse_args(["--full-logit-capture", "--full-logit-build-id", valid])
        self.assertEqual(args.full_logit_build_id, "a" * 64)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--full-logit-capture", "--full-logit-build-id", "binary-hash"])

    def test_supplied_bundle_id_wins_over_legacy_binary_hash(self):
        args = minimal_args(full_logit_capture=True, full_logit_build_id="b" * 64)
        build_id, source = runner.resolve_full_logit_capture_identity(args, "a" * 64)
        self.assertEqual(build_id, "b" * 64)
        self.assertEqual(source, "operator_verified_bundle_sha256")

    def test_legacy_binary_hash_remains_fallback(self):
        args = minimal_args(full_logit_capture=True, full_logit_build_id=None)
        build_id, source = runner.resolve_full_logit_capture_identity(args, "c" * 64)
        self.assertEqual(build_id, "c" * 64)
        self.assertEqual(source, "legacy_server_binary_sha256")

    def test_build_id_requires_capture_mode_at_cli_validation(self):
        with self.assertRaisesRegex(ValueError, "requires --full-logit-capture"):
            runner.validate_full_logit_capture_options(
                minimal_args(full_logit_build_id="b" * 64, full_logit_capture=False))
        runner.validate_full_logit_capture_options(
            minimal_args(full_logit_build_id="b" * 64, full_logit_capture=True))

    def test_public_split_flag_is_documented_and_legacy_route_stays(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn("public --kv-position-split-p", source)
        self.assertIn('env["LLAMA_KV_POSITION_SPLIT_P"] = str(p)', source)

    def test_small_context_filter_is_bounded(self):
        self.assertEqual(runner.full_logit_checkpoints(minimal_args(), 16), list(range(1, 17)))

    def test_long_context_uses_request_points_not_every_position(self):
        args = minimal_args(split_p_high=102400, split_ctx_local=102400,
                            split_ctx_high=131072, split_tokens_local=96000)
        checkpoints = runner.full_logit_checkpoints(args, 131072)
        self.assertIn(96001, checkpoints)
        self.assertIn(102657, checkpoints)
        self.assertLess(len(checkpoints), 512)
        self.assertLess(len(",".join(map(str, checkpoints))), 8192)
        self.assertLessEqual(max(checkpoints), 131072)

    def test_every_server_instance_owns_its_capture_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            left = runner.ServerInstance(minimal_args(), out, "D8a-pure", [])
            right = runner.ServerInstance(minimal_args(), out, "D8a-split", [])
            left.enable_full_logit_capture(out, "campaign", 2048, "b" * 64,
                                           "operator_verified_bundle_sha256")
            right.enable_full_logit_capture(out, "campaign", 2048, "binary-hash")
            self.assertNotEqual(left.capture_dir, right.capture_dir)
            self.assertNotEqual(left.env_extra["F4_TEST_LOGIT_CAPTURE"], right.env_extra["F4_TEST_LOGIT_CAPTURE"])
            self.assertEqual(left.env_extra["F4_TEST_PHASE"], left.case_id)
            self.assertEqual(right.env_extra["F4_TEST_PHASE"], right.case_id)
            self.assertEqual(left.env_extra["F4_TEST_RUN_ID"], right.env_extra["F4_TEST_RUN_ID"])
            self.assertEqual(left.env_extra["LLAMA_OVF_ATTN_REFERENCE_CPU"], "0")
            capture_plan = json.loads((left.capture_dir / "capture-plan.json").read_text())
            self.assertEqual(capture_plan["build_id"], "b" * 64)
            self.assertEqual(capture_plan["build_id_source"], "operator_verified_bundle_sha256")

    def test_scenario_d_uses_recorded_capture_bundle_identity(self):
        import inspect
        source = inspect.getsource(runner.scenario_d)
        self.assertIn('run_meta.get("full_logit_capture_build_id")', source)
        self.assertIn('run_meta.get("full_logit_capture_build_id_source",', source)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "optional full-logit comparator requires NumPy")
class TestFullLogitCheckpointComparison(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.left = self.directory / "source"
        self.right = self.directory / "target"
        self.left.mkdir()
        self.right.mkdir()
        self.write_capture(self.left, "baseline", "cold")
        self.write_capture(self.right, "restored", "disk")

    def write_capture(self, directory, phase, source, *, logits=(1.0, 0.0, -2.0),
                      tokens=(11, 12, 13), **changes):
        stem = directory / f"cap.{phase}.n3"
        meta = {"run_id": "campaign", "phase": phase, "build_id": "build",
                "model_fingerprint": "model", "config_core": "same core", "config_hash": "core-hash",
                "tokens_n": 3, "tokens_seq_hash": "same-token-hash", "input_pos": 2,
                "output_pos": 2, "pos_max": 2, "n_ctx": 2048, "n_vocab": 3,
                "logits_count": 3, "finite": True, "layout_hash": "same-layout",
                "cache_source": source, "cache_reason": "reason", "attempt": 1}
        meta.update(changes)
        Path(str(stem) + ".json").write_text(json.dumps(meta))
        Path(str(stem) + ".logits").write_bytes(struct.pack("<3f", *logits))
        Path(str(stem) + ".tokens").write_bytes(struct.pack("<3i", *tokens))

    def compare(self, **kwargs):
        return runner.compare_full_logit_checkpoint(
            self.left, "baseline", self.right, "restored", 3, "campaign", [11, 12, 13], **kwargs)

    def test_exact_same_layout_restore_has_zero_fp64_kl(self):
        result = self.compare()
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["kl_nat"], 0.0)
        self.assertEqual(result["rows"], 1)

    def test_full_vocabulary_difference_fails_even_when_argmax_matches(self):
        self.write_capture(self.right, "restored", "disk", logits=(1.0, 0.0, 0.9))
        result = self.compare()
        self.assertTrue(result["argmax_equal"])
        self.assertGreater(result["kl_nat"], 1e-4)
        self.assertEqual(result["state"], "failed")

    def test_missing_capture_is_failure(self):
        (self.right / "cap.restored.n3.logits").unlink()
        self.assertEqual(self.compare()["state"], "failed")

    def test_nonfinite_payload_is_failure_even_when_metadata_claims_finite(self):
        self.write_capture(self.right, "restored", "disk", logits=(1.0, float("nan"), -2.0))
        result = self.compare()
        self.assertEqual(result["state"], "failed")
        self.assertIn("nonfinite", result["reason"])

    def test_trailing_partial_float_is_rejected(self):
        with (self.right / "cap.restored.n3.logits").open("ab") as handle:
            handle.write(b"x")
        self.assertIn("capture bytes", self.compare()["reason"])

    def test_model_or_numeric_configuration_mismatch_is_not_compared(self):
        for field in ("model_fingerprint", "build_id", "config_core", "config_hash", "pos_max"):
            self.write_capture(self.right, "restored", "disk", **{field: "different"})
            result = self.compare()
            self.assertEqual(result["state"], "failed")
            self.assertNotIn("kl_nat", result)
            self.assertIn(field, result["reason"])

    def test_actual_tokens_must_match_even_when_metadata_hashes_match(self):
        self.write_capture(self.right, "restored", "disk", tokens=(11, 12, 14))
        result = self.compare()
        self.assertEqual(result["state"], "failed")
        self.assertIn("evaluated tokens", result["reason"])

    def test_cold_target_cannot_claim_disk_restore_parity(self):
        self.write_capture(self.right, "restored", "cold")
        self.assertIn("disk import", self.compare()["reason"])

    def test_pure_to_split_requires_explicit_different_layout_contract(self):
        self.write_capture(self.left, "baseline", "cold", layout_hash="pure-layout")
        self.assertEqual(self.compare()["state"], "failed")
        self.assertEqual(self.compare(layout_relation="different")["state"], "complete")

    def test_direct_inverse_restore_can_require_live_cache_source_on_both_arms(self):
        self.write_capture(self.left, "baseline", "live", layout_hash="split-layout")
        self.write_capture(self.right, "restored", "live", layout_hash="pure-layout")
        result = self.compare(layout_relation="different", target_cache_source="live",
                              baseline_cache_source_expected="live")
        self.assertEqual(result["state"], "complete")
        self.assertNotEqual(result["left_layout_hash"], result["right_layout_hash"])

    def test_direct_inverse_restore_rejects_non_live_baseline_source(self):
        self.write_capture(self.left, "baseline", "disk", layout_hash="split-layout")
        self.write_capture(self.right, "restored", "live", layout_hash="pure-layout")
        result = self.compare(layout_relation="different", target_cache_source="live",
                              baseline_cache_source_expected="live")
        self.assertEqual(result["state"], "failed")
        self.assertIn("baseline cache source", result["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
