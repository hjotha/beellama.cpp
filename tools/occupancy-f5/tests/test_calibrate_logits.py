"""Synthetic, CPU-only tests for paired fixed-current-KV KL calibration."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import calibrate_logits as calibration  # noqa: E402
import compare_logits  # noqa: E402


class CalibrateLogitsTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="kl-calibration-test-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.teacher = list(range(32)) + [0]
        self.base = self.make_root("baseline", p=0, n_ctx=64, gpu_peak=0.04,
                                   classification="pure")
        self.candidate = self.make_root("candidate", p=256, n_ctx=96, gpu_peak=0.045,
                                        classification="candidate")

    @staticmethod
    def core(*, n_ctx: int, p: int, kv_k: int = 4, kv_v: int = 4,
             overflow_k: int = 14, overflow_v: int = 14,
             remote_layers: str = "full") -> str:
        return (f"model_sha256={'a' * 64};n_ctx={n_ctx};n_batch=512;n_ubatch=256;"
                f"n_seq=1;n_params=1000;p={p};kv_k={kv_k};kv_v={kv_v};"
                f"overflow_k={overflow_k};overflow_v={overflow_v};remote_layers={remote_layers}")

    def make_root(self, name: str, *, p: int, n_ctx: int, gpu_peak: float,
                  classification: str) -> Path:
        root = self.root / name
        root.mkdir()
        (root / "teacher-tokens.json").write_text(json.dumps(self.teacher), encoding="utf-8")
        remote_layers = "auto" if classification == "pure" else "5" if classification == "layered" else "full"
        core = self.core(n_ctx=n_ctx, p=p, remote_layers=remote_layers)
        ranges = []
        for n in range(1, 33):
            range_dir = root / "range-1"
            for phase in ("cpu", "gpu"):
                (range_dir / phase).mkdir(parents=True, exist_ok=True)
                stem = range_dir / phase / f"cap.{phase}.n{n}"
                prefix = self.teacher[:n]
                (Path(str(stem) + ".tokens")).write_bytes(struct.pack("<" + "i" * n, *prefix))
                logits = [0.0] * 32
                if phase == "gpu":
                    logits[0] = gpu_peak
                (Path(str(stem) + ".logits")).write_bytes(struct.pack("<32f", *logits))
                metadata = {
                    "build_id": "b" * 64,
                    "model_fingerprint": "sha256:" + "a" * 64,
                    "model_sha256": "a" * 64,
                    "config_core": core,
                    "n_ctx": n_ctx,
                    "n_params": 1000,
                    "n_vocab": 32,
                    "split_p": p,
                    "tokens_n": n,
                    "input_pos": n - 1,
                    "output_pos": n - 1,
                    "checkpoint_method": "full",
                    "finite": True,
                    "prefix_proof": {"verified": True, "equal": True},
                }
                Path(str(stem) + ".json").write_text(json.dumps(metadata), encoding="utf-8")

            rows = []
        # A paired-result row describes the CPU reference vs native GPU logits.
        data = compare_logits.compare(range_dir / "cpu", range_dir / "gpu", self.teacher)
        for result_row in data["rows"]:
            n = result_row["position"]
            rows.append({
                "n": n,
                "state": "complete",
                "checkpoint_method": "full",
                "prefix_proof": {"verified": True, "equal": True},
                "current_kv_proof": {
                    "complete": True,
                    "required_count": 32,
                    "captured_count": 32,
                    "injected_count": 32,
                    "all_byte_equal": True,
                    "tensor_manifest": [
                        {"name": f"layer{layer}-{part}", "path": f"current-kv/layer{layer}-{part}.f32",
                         "bytes": 128, "xxh64": "0123456789abcdef", "shape": [32],
                         "type": "F32", "contiguous": True, "nq": 1}
                        for layer in range(16) for part in ("k", "v")
                    ],
                },
                "post_gpu_restore_proof": {"equal": True},
                "metrics": {"kl_nat": result_row["kl"]},
            })
        range_result = {
            "start": 1,
            "count": 32,
            "complete": True,
            "fixed_memory_reference_complete": True,
            "rows": rows,
            "mean_kl_nat_per_token": data["mean_kl_nat_per_token"],
            "p99_kl_nat_per_token": data["p99_kl_nat_per_token"],
        }
        report = {
            "schema": calibration.PAIRED_SCHEMA,
            "complete": True,
            "reference_arm": "cpu",
            "checkpoint_method": "full",
            "fixed_current_kv": True,
            "fixed_memory_reference_complete": True,
            "protocol_acceptance_eligible": classification == "candidate",
            "calibration_candidate_eligible": classification == "candidate",
            "calibration_baseline_eligible": classification != "candidate",
            "pure_cuda_baseline": classification == "pure",
            "layered_baseline": classification == "layered",
            "configuration": {
                "config_core": core,
                "reference_arm": "cpu",
                "checkpoint_method": "full",
                "fixed_current_kv": True,
                "full_prefix_verification": True,
                "model_sha256_operator_verified": "a" * 64,
                "build_bundle_sha256_operator_verified": "b" * 64,
                "teacher_file": {"path": "/synthetic/teacher.bin", "bytes": 132,
                                 "xxh64": "0123456789abcdef"},
            },
            "ranges": [range_result],
        }
        (root / "paired-result.json").write_text(json.dumps(report), encoding="utf-8")
        return root

    def invoke(self, candidate: Path | None = None, baseline: Path | None = None,
               minimum: int = 32) -> tuple[int, dict]:
        out = io.StringIO()
        argv = [str(candidate or self.candidate), str(baseline or self.base),
                "--minimum", str(minimum)]
        with contextlib.redirect_stdout(out):
            code = calibration.main(argv)
        return code, json.loads(out.getvalue())

    def test_passes_with_different_context_split_and_cache_formats(self) -> None:
        # Candidate K/V and overflow formats are allowed to differ; CPU/GPU
        # config remains exact within each paired root.
        cfg = json.loads((self.candidate / "paired-result.json").read_text())
        cfg["configuration"]["config_core"] = self.core(
            n_ctx=96, p=256, kv_k=5, kv_v=6, overflow_k=15, overflow_v=16)
        (self.candidate / "paired-result.json").write_text(json.dumps(cfg))
        for path in (self.candidate / "range-1").glob("*/*.json"):
            meta = json.loads(path.read_text())
            meta["config_core"] = cfg["configuration"]["config_core"]
            meta["n_ctx"] = 96
            path.write_text(json.dumps(meta))
        code, result = self.invoke()
        self.assertEqual(code, 0, result)
        self.assertEqual(result["schema"], "occupancy-kl-calibrated-v1")
        self.assertEqual(result["scope"], "teacher_forced_logits_only")
        self.assertTrue(result["ranges"][0]["pass"])
        metrics = result["ranges"][0]["groups"]["cpu_reference_vs_native_gpu"]
        self.assertLessEqual(metrics["delta_mean_kl_nat_per_token"], 1e-4)
        self.assertLessEqual(metrics["delta_p99_kl_nat_per_token"], 1e-3)
        self.assertIn("baseline_comparison", metrics)
        self.assertIn("candidate_comparison", metrics)

    def test_numeric_delta_failure_returns_one(self) -> None:
        self.candidate = self.make_root("too-different", p=256, n_ctx=96,
                                        gpu_peak=0.25, classification="candidate")
        code, result = self.invoke()
        self.assertEqual(code, 1, result)
        self.assertEqual(result["verdict"], "fail")

    def test_natural_or_old_protocol_is_invalid(self) -> None:
        report_path = self.candidate / "paired-result.json"
        report = json.loads(report_path.read_text())
        report["fixed_current_kv"] = False
        report_path.write_text(json.dumps(report))
        code, result = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("fixed_current_kv", result["reason"])

    def test_incomplete_fixed_kv_proof_is_invalid(self) -> None:
        report_path = self.candidate / "paired-result.json"
        report = json.loads(report_path.read_text())
        report["ranges"][0]["rows"][0]["current_kv_proof"]["injected_count"] = 1
        report_path.write_text(json.dumps(report))
        code, result = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("proof counts", result["reason"])

    def test_missing_gpu_post_restore_proof_is_invalid(self) -> None:
        report_path = self.candidate / "paired-result.json"
        report = json.loads(report_path.read_text())
        report["ranges"][0]["rows"][0].pop("post_gpu_restore_proof")
        report_path.write_text(json.dumps(report))
        code, result = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("GPU post-restore", result["reason"])

    def test_materially_negative_reported_kl_is_invalid(self) -> None:
        report_path = self.candidate / "paired-result.json"
        report = json.loads(report_path.read_text())
        report["ranges"][0]["rows"][0]["metrics"]["kl_nat"] = -1e-9
        report_path.write_text(json.dumps(report))
        code, result = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("negative C++ KL", result["reason"])

    def test_different_teacher_tokens_are_invalid(self) -> None:
        (self.candidate / "teacher-tokens.json").write_text(json.dumps([0] * 33))
        code, result = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("teacher-token sets differ", result["reason"])

    def test_disallowed_shared_configuration_difference_is_invalid(self) -> None:
        report_path = self.candidate / "paired-result.json"
        report = json.loads(report_path.read_text())
        report["configuration"]["config_core"] = report["configuration"]["config_core"].replace(
            "n_ubatch=256", "n_ubatch=128")
        report_path.write_text(json.dumps(report))
        code, result = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("disallowed config_core differences", result["reason"])

    def test_minimum_below_32_is_invalid(self) -> None:
        code, result = self.invoke(minimum=31)
        self.assertEqual(code, 2)
        self.assertIn("at least 32", result["reason"])

    def test_layered_xxxl_diagnostic_baseline_is_explicit(self) -> None:
        self.base = self.make_root("layered", p=0, n_ctx=64, gpu_peak=0.04,
                                   classification="layered")
        code, result = self.invoke()
        self.assertEqual(code, 0, result)
        self.assertEqual(result["provenance"]["baseline_classification"],
                         "layered_xxxl_diagnostic")
        self.assertEqual(result["provenance"]["configuration"]["remote_layers"],
                         {"baseline": "5", "candidate": "full"})

    def test_layered_zero_or_auto_placement_is_invalid(self) -> None:
        for remote_layers in ("0", "auto"):
            with self.subTest(remote_layers=remote_layers):
                base = self.make_root(f"layered-{remote_layers}", p=0, n_ctx=64,
                                      gpu_peak=0.04, classification="layered")
                report_path = base / "paired-result.json"
                report = json.loads(report_path.read_text())
                report["configuration"]["config_core"] = self.core(
                    n_ctx=64, p=0, remote_layers=remote_layers)
                report_path.write_text(json.dumps(report))
                code, result = self.invoke(baseline=base)
                self.assertEqual(code, 2)
                self.assertIn("remote_layers", result["reason"])

    def test_candidate_remote_placement_must_cover_all_kv_layers(self) -> None:
        report_path = self.candidate / "paired-result.json"
        report = json.loads(report_path.read_text())
        report["configuration"]["config_core"] = self.core(
            n_ctx=96, p=256, remote_layers="5")
        report_path.write_text(json.dumps(report))
        for path in (self.candidate / "range-1").glob("*/*.json"):
            meta = json.loads(path.read_text())
            meta["config_core"] = report["configuration"]["config_core"]
            path.write_text(json.dumps(meta))
        code, result = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("cover all 16 KV layers", result["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
