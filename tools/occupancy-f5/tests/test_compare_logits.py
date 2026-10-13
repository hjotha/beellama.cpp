import importlib.util
import contextlib
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("compare_logits", Path(__file__).parents[1]/"compare_logits.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class CompareLogitsTest(unittest.TestCase):
    def test_underflow_does_not_destroy_kl(self):
        result = m.distribution_metrics([0.0, -1000000.0], [-1000000.0, 0.0], 0)
        self.assertEqual(result["kl"], 1000000.0)

    def test_p99_is_over_tokens(self):
        self.assertEqual(m.percentile99([0.0]*99 + [1.0]), 0.0)

    def test_nonfinite_is_rejected(self):
        with self.assertRaises(ValueError):
            m.distribution_metrics([0.0, math.nan], [0.0, 1.0], 0)

    def captures(self, directory, positions=(1, 2), mismatch=False, n_ctx=8, config_core=None):
        directory.mkdir()
        if config_core is None:
            config_core = (f"desc=model;n_ctx={n_ctx};n_seq=1;n_batch=512;"
                           "n_ubatch=256;n_embd=64;n_layer=4;n_params=1")
        for n in positions:
            stem = directory/f"cap.test.n{n}"
            prefix = [0]*n
            if mismatch: prefix[0] = 1
            Path(str(stem)+".tokens").write_bytes(struct.pack("<"+"i"*n, *prefix))
            Path(str(stem)+".logits").write_bytes(struct.pack("<2f", 1.0, 0.0))
            Path(str(stem)+".json").write_text(json.dumps({"model_fingerprint":"model", "config_core":config_core, "n_vocab":2, "n_params":1, "n_ctx":n_ctx}))

    def test_identical_teacher_forced_captures(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root)/"a", Path(root)/"b"
            self.captures(a); self.captures(b)
            result = m.compare(a, b, [0,0,0], minimum=2)
            self.assertTrue(result["structural_gate_pass"])
            self.assertEqual(result["mean_kl_nat_per_token"], 0.0)
            self.assertEqual(result["p99_kl_nat_per_token"], 0.0)

    def test_prefix_mismatch_is_rejected_even_when_logits_match(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root)/"a", Path(root)/"b"
            self.captures(a); self.captures(b, mismatch=True)
            with self.assertRaisesRegex(ValueError, "prefix mismatch"):
                m.compare(a, b, [0,0,0], minimum=2)

    def test_incomplete_checkpoint_set_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root)/"a", Path(root)/"b"
            self.captures(a); self.captures(b, positions=(1,))
            with self.assertRaisesRegex(ValueError, "sets differ"):
                m.compare(a, b, [0,0,0], minimum=2)

    def test_context_capacity_difference_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root)/"a", Path(root)/"b"
            self.captures(a, n_ctx=102400)
            self.captures(b, n_ctx=131072)
            with self.assertRaisesRegex(ValueError, "provenance mismatch: config_core"):
                m.compare(a, b, [0,0,0], minimum=2)

    def test_context_capacity_difference_allowed_only_for_metrics_and_not_structural(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root)/"a", Path(root)/"b"
            self.captures(a, n_ctx=102400)
            self.captures(b, n_ctx=131072)
            with self.assertRaisesRegex(ValueError, "requires --metrics-only"):
                m.compare(a, b, [0,0,0], minimum=2, allow_context_capacity_difference=True)
            result = m.compare(a, b, [0,0,0], minimum=2,
                               allow_context_capacity_difference=True, metrics_only=True)
            self.assertEqual(result["context_capacities"],
                             {"reference": 102400, "candidate": 131072})
            self.assertTrue(result["context_capacity_difference_allowed"])
            self.assertEqual(result["quality_comparison_scope"], "metrics_only_context_capacity_difference")
            self.assertIsNone(result["structural_gate_pass"])
            self.assertFalse(result["structural_gate_evaluated"])

    def test_context_override_still_rejects_other_config_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root)/"a", Path(root)/"b"
            self.captures(a, n_ctx=102400)
            self.captures(b, n_ctx=131072,
                          config_core="desc=model;n_ctx=131072;n_seq=1;n_batch=512;"
                                      "n_ubatch=128;n_embd=64;n_layer=4;n_params=1")
            with self.assertRaisesRegex(ValueError, "config_core excluding n_ctx"):
                m.compare(a, b, [0,0,0], minimum=2,
                          allow_context_capacity_difference=True, metrics_only=True)

    def test_cli_rejects_context_override_without_metrics_only(self):
        with tempfile.TemporaryDirectory() as root:
            tokens = Path(root)/"tokens.json"
            tokens.write_text("[0,0,0]")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = m.main(["ref", "cand", "--teacher-tokens", str(tokens),
                                 "--allow-context-capacity-difference"])
            self.assertEqual(status, 2)
            self.assertIn("requires --metrics-only", output.getvalue())

    @staticmethod
    def cache_layout_core(*, n_ctx=8, p=0, kv_k=0, kv_v=0, overflow_k=1,
                          overflow_v=1, type_k=1, type_v=1, remote_layers="auto",
                          n_batch=512, n_ubatch=256, extra=""):
        fields = (f"desc=model;n_ctx={n_ctx};n_seq=1;n_batch={n_batch};n_ubatch={n_ubatch};"
                  f"n_embd=64;n_layer=4;n_params=1;model_sha256=abc;p={p};"
                  f"kv_k={kv_k};kv_v={kv_v};overflow_k={overflow_k};overflow_v={overflow_v};"
                  f"cache_type_k={type_k};cache_type_v={type_v};remote_layers={remote_layers}")
        return fields + (f";{extra}" if extra else "")

    def test_cache_layout_difference_is_metrics_only_and_records_normalization(self):
        with tempfile.TemporaryDirectory() as root:
            reference, candidate = Path(root)/"f16", Path(root)/"kvarn4"
            f16 = self.cache_layout_core()
            k4 = self.cache_layout_core(p=1024, kv_k=4, kv_v=4, overflow_k=14,
                                        overflow_v=14, type_k=0, type_v=0,
                                        remote_layers="full")
            self.captures(reference, config_core=f16)
            self.captures(candidate, config_core=k4)
            with self.assertRaisesRegex(ValueError, "requires --metrics-only"):
                m.compare(reference, candidate, [0, 0, 0], minimum=2,
                          allow_cache_layout_difference=True)
            result = m.compare(reference, candidate, [0, 0, 0], minimum=2,
                               metrics_only=True, allow_cache_layout_difference=True)
            self.assertEqual(result["quality_comparison_scope"],
                             "metrics_only_cache_layout_difference")
            self.assertEqual(result["cache_layout_differences"]["cache_type_k"],
                             {"reference": "1", "candidate": "0"})
            self.assertEqual(result["cache_layout_differences"]["kv_k"],
                             {"reference": "0", "candidate": "4"})
            self.assertEqual(result["cache_layout_differences"]["remote_layers"],
                             {"reference": "auto", "candidate": "full"})
            self.assertIsNone(result["structural_gate_pass"])
            self.assertFalse(result["structural_gate_evaluated"])
            self.assertEqual(set(result["cache_layout_allowed_fields"]), m._CACHE_LAYOUT_FIELDS)

    def test_cache_layout_flag_keeps_n_ctx_strict_unless_context_flag_is_also_set(self):
        with tempfile.TemporaryDirectory() as root:
            reference, candidate = Path(root)/"ref", Path(root)/"cand"
            self.captures(reference, n_ctx=8,
                          config_core=self.cache_layout_core(n_ctx=8))
            self.captures(candidate, n_ctx=16,
                          config_core=self.cache_layout_core(n_ctx=16, p=512, kv_k=4,
                                                             kv_v=4, type_k=0, type_v=0,
                                                             remote_layers="full"))
            with self.assertRaisesRegex(ValueError, "n_ctx"):
                m.compare(reference, candidate, [0, 0, 0], minimum=2,
                          metrics_only=True, allow_cache_layout_difference=True)
            result = m.compare(reference, candidate, [0, 0, 0], minimum=2,
                               metrics_only=True, allow_cache_layout_difference=True,
                               allow_context_capacity_difference=True)
            self.assertTrue(result["context_capacity_difference_allowed"])
            self.assertEqual(result["quality_comparison_scope"],
                             "metrics_only_cache_layout_and_context_difference")
            self.assertIsNone(result["structural_gate_pass"])

    def test_cache_layout_mode_rejects_other_config_differences(self):
        for field in ("n_batch", "n_ubatch", "model_sha256", "custom_setting"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                reference, candidate = Path(root)/"ref", Path(root)/"cand"
                base = self.cache_layout_core()
                changed_fields = {
                    "n_batch": self.cache_layout_core(n_batch=128),
                    "n_ubatch": self.cache_layout_core(n_ubatch=128),
                    "model_sha256": base.replace("model_sha256=abc", "model_sha256=def"),
                    "custom_setting": self.cache_layout_core(extra="custom_setting=one"),
                }
                changed = changed_fields[field]
                if field == "custom_setting":
                    base += ";custom_setting=two"
                self.captures(reference, config_core=base)
                self.captures(candidate, config_core=changed)
                with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                    m.compare(reference, candidate, [0, 0, 0], minimum=2,
                              metrics_only=True, allow_cache_layout_difference=True)

    def test_cache_layout_mode_rejects_missing_duplicate_and_malformed_fields(self):
        valid = self.cache_layout_core()
        with self.assertRaisesRegex(ValueError, "missing cache-layout fields"):
            m._parse_config_core(valid.replace(";remote_layers=auto", ""))
        with self.assertRaisesRegex(ValueError, "duplicate config_core field: p"):
            m._parse_config_core(valid + ";p=1")
        with self.assertRaisesRegex(ValueError, "malformed config_core field"):
            m._parse_config_core(valid + ";broken")

    def test_cli_cache_layout_flag_requires_metrics_only_and_minimum_32(self):
        with tempfile.TemporaryDirectory() as root:
            tokens = Path(root)/"tokens.json"
            tokens.write_text("[0,0,0]")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = m.main(["ref", "cand", "--teacher-tokens", str(tokens),
                                 "--allow-cache-layout-difference"])
            self.assertEqual(status, 2)
            self.assertIn("requires --metrics-only", output.getvalue())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = m.main(["ref", "cand", "--teacher-tokens", str(tokens),
                                 "--metrics-only", "--allow-cache-layout-difference",
                                 "--minimum", "31"])
            self.assertEqual(status, 2)
            self.assertIn("at least 32", output.getvalue())
