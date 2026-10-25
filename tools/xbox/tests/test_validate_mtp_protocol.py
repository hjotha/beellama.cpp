#!/usr/bin/env python3
"""Comprehensive unit and adversarial test suite for Plan 003 Paired MTP Offline Protocol Validator.

Test Categories:
1. Synthetic Positive Fixtures (SYNTHETIC_PASS, fixture-only label).
2. Non-Synthetic Product Certification Inability (DIAGNOSTIC_ONLY_NOT_CERTIFIED, product PASS impossible).
3. Historical October 5 Negative Regression Fixture (FAIL with exact token divergence positions).
4. Exact Mathematical Binomial Coverage for Median CI (N=5, 6, 7, 10).
5. Adversarial Block Pairing & Order Evidence Tests.
6. Adversarial Provenance & Hash Tests (model, package, config, prompts SHA-256).
7. Adversarial Mandatory Gate Reports Tests (assertion-only, scope mismatch, incomplete assertions, EOG).
8. Adversarial Ingestion & Schema Integrity Tests (missing files, bad types, duplicate keys, non-d3d12, bad sampling).
9. Adversarial Grid Matrix Tests (reduced/expanded/duplicate prompts/lengths/arms, missing/extra cells).
10. Adversarial CSV & Token Sidecar Integrity Tests (RFC-4180 quotes, non-finite values, bad hashes, determinism).
11. Statistical Math & Sequence Parity Comparison Tests (Hodges-Lehmann, Student-t, Walsh averages, sequence parity).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import scripts.validate_mtp_protocol as vmp


class SyntheticPositiveTests(unittest.TestCase):
    """Verify that a clean synthetic positive dataset produces SYNTHETIC_PASS."""

    def setUp(self) -> None:
        self.fixture_dir = Path(__file__).parent / "fixtures" / "mtp_protocol" / "synthetic_positive"
        self.manifest_path = self.fixture_dir / "manifest.json"

    def test_clean_synthetic_fixture_returns_synthetic_pass(self) -> None:
        validator = vmp.MtpProtocolValidator(self.manifest_path)
        report = validator.validate()

        self.assertEqual(report.overall_verdict, "SYNTHETIC_PASS")
        self.assertTrue(report.is_synthetic)
        self.assertTrue(report.all_cells_parity_pass)
        self.assertTrue(report.all_cells_kernel_parity_pass)
        self.assertTrue(report.all_cells_mtp_speedup_pass)
        self.assertTrue(report.all_cells_block_pairing_pass)
        self.assertTrue(report.all_mandatory_gates_pass)
        self.assertTrue(report.all_sample_sizes_sufficient)
        self.assertEqual(len(report.cells), 6)

    def test_non_synthetic_cannot_yield_product_pass_returns_diagnostic_only(self) -> None:
        """Verify that in this diagnostic draft, product PASS is impossible for non-synthetic data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["synthetic"] = False
            data["provenance"]["model"]["file_sha256"] = "a" * 64
            data["provenance"]["package"]["digest_sha256"] = "b" * 64
            data["provenance"]["config"]["config_hash"] = "c" * 64
            for p_info in data["provenance"]["prompts"].values():
                p_info["prompt_hash"] = "d" * 64

            # Update gate reports scope to match
            for g in data["mandatory_gates"].values():
                g["scope"]["package_identity"] = data["provenance"]["package"]["declared_identity"]
                g["scope"]["model_file_sha256"] = "a" * 64
                g["scope"]["config_hash"] = "c" * 64

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            report = validator.validate()
            # Product PASS is impossible: returns DIAGNOSTIC_ONLY_NOT_CERTIFIED
            self.assertEqual(report.overall_verdict, "DIAGNOSTIC_ONLY_NOT_CERTIFIED")
            self.assertNotEqual(report.overall_verdict, "PASS")
            self.assertFalse(report.is_synthetic)
            self.assertTrue(any("DIAGNOSTIC NOTICE" in msg for msg in report.summary_messages))


class HistoricalOctober5FixtureTests(unittest.TestCase):
    """Verify that historical October 5 console artifacts are strictly evaluated as negative regression."""

    def setUp(self) -> None:
        self.manifest_path = Path(__file__).resolve().parent.parent / "bench" / "results" / "protocol-20261005-provenance.json"

    def test_historical_manifest_evaluates_to_fail(self) -> None:
        if not self.manifest_path.is_file():
            self.skipTest(f"Historical manifest not found: {self.manifest_path}")

        validator = vmp.MtpProtocolValidator(self.manifest_path)
        report = validator.validate()

        self.assertEqual(report.overall_verdict, "FAIL")
        self.assertFalse(report.is_synthetic)
        self.assertFalse(report.all_cells_parity_pass)
        self.assertFalse(report.all_cells_block_pairing_pass)
        self.assertFalse(report.all_mandatory_gates_pass)
        self.assertFalse(report.all_sample_sizes_sufficient)

    def test_historical_spec_chat_divergence_position(self) -> None:
        if not self.manifest_path.is_file():
            self.skipTest(f"Historical manifest not found: {self.manifest_path}")

        validator = vmp.MtpProtocolValidator(self.manifest_path)
        report = validator.validate()

        chat_64 = next(c for c in report.cells if c.cell_key == "spec-chat-open_64")
        self.assertFalse(chat_64.greedy_parity_passed)
        self.assertEqual(chat_64.mtp_auto_vs_baseline_parity.earliest_mismatch_pos_0idx, 33)
        self.assertEqual(chat_64.mtp_auto_vs_baseline_parity.earliest_mismatch_pos_1idx, 34)
        self.assertEqual(chat_64.mtp_auto_vs_baseline_parity.left_token, 279)
        self.assertEqual(chat_64.mtp_auto_vs_baseline_parity.right_token, 1204)

    def test_historical_standard_512_divergence_position(self) -> None:
        if not self.manifest_path.is_file():
            self.skipTest(f"Historical manifest not found: {self.manifest_path}")

        validator = vmp.MtpProtocolValidator(self.manifest_path)
        report = validator.validate()

        std_64 = next(c for c in report.cells if c.cell_key == "standard-512_64")
        self.assertFalse(std_64.greedy_parity_passed)
        self.assertEqual(std_64.mtp_auto_vs_baseline_parity.earliest_mismatch_pos_0idx, 39)
        self.assertEqual(std_64.mtp_auto_vs_baseline_parity.earliest_mismatch_pos_1idx, 40)
        self.assertEqual(std_64.mtp_auto_vs_baseline_parity.left_token, 16)
        self.assertEqual(std_64.mtp_auto_vs_baseline_parity.right_token, 9)


class GenericBinomialMedianCoverageTests(unittest.TestCase):
    """Verify exact mathematical binomial order statistics coverage for N=5, 6, 7, 10."""

    def test_n5_binomial_coverage(self) -> None:
        cov_r1 = vmp.compute_binomial_median_coverage(5, 1)
        cov_r2 = vmp.compute_binomial_median_coverage(5, 2)
        self.assertAlmostEqual(cov_r1, 0.9375, places=4)
        self.assertAlmostEqual(cov_r2, 0.6250, places=4)

        res = vmp.compute_nonparametric_median_ci([10, 20, 30, 40, 50], target_confidence=0.95)
        self.assertFalse(res["sufficient_for_target"])
        self.assertEqual(res["achieved_confidence_pct"], 93.75)
        self.assertEqual(res["ci"], (10, 50))

    def test_n6_binomial_coverage(self) -> None:
        cov_r1 = vmp.compute_binomial_median_coverage(6, 1)
        cov_r2 = vmp.compute_binomial_median_coverage(6, 2)
        self.assertAlmostEqual(cov_r1, 0.96875, places=4)
        self.assertAlmostEqual(cov_r2, 0.78125, places=4)

        res = vmp.compute_nonparametric_median_ci([10, 20, 30, 40, 50, 60], target_confidence=0.95)
        self.assertTrue(res["sufficient_for_target"])
        self.assertEqual(res["r"], 1)
        self.assertAlmostEqual(res["achieved_confidence_pct"], 96.88, places=2)
        self.assertEqual(res["ci"], (10, 60))

    def test_n7_binomial_coverage_narrowest_interval(self) -> None:
        cov_r1 = vmp.compute_binomial_median_coverage(7, 1)
        cov_r2 = vmp.compute_binomial_median_coverage(7, 2)
        self.assertAlmostEqual(cov_r1, 0.984375, places=4)
        self.assertAlmostEqual(cov_r2, 0.875000, places=4)

        res = vmp.compute_nonparametric_median_ci([10, 20, 30, 40, 50, 60, 70], target_confidence=0.95)
        self.assertTrue(res["sufficient_for_target"])
        self.assertEqual(res["r"], 1)
        self.assertAlmostEqual(res["achieved_confidence_pct"], 98.44, places=2)
        self.assertEqual(res["ci"], (10, 70))

    def test_n10_binomial_coverage_narrowest_interval(self) -> None:
        cov_r1 = vmp.compute_binomial_median_coverage(10, 1)
        cov_r2 = vmp.compute_binomial_median_coverage(10, 2)
        cov_r3 = vmp.compute_binomial_median_coverage(10, 3)
        self.assertAlmostEqual(cov_r1, 0.998046875, places=4)
        self.assertAlmostEqual(cov_r2, 0.978515625, places=4)
        self.assertAlmostEqual(cov_r3, 0.890625000, places=4)

        data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        res = vmp.compute_nonparametric_median_ci(data, target_confidence=0.95)
        self.assertTrue(res["sufficient_for_target"])
        self.assertEqual(res["r"], 2)
        self.assertAlmostEqual(res["achieved_confidence_pct"], 97.85, places=2)
        self.assertEqual(res["ci"], (20, 90))


class AdversarialBlockPairingTests(unittest.TestCase):
    """Adversarial tests for block pairing, order evidence, and randomization."""

    def setUp(self) -> None:
        self.fixture_dir = Path(__file__).parent / "fixtures" / "mtp_protocol" / "synthetic_positive"

    def test_absent_blocks_in_product_mode_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["synthetic"] = False
            data["provenance"]["model"]["file_sha256"] = "0" * 64
            data["provenance"]["package"]["digest_sha256"] = "0" * 64
            data["provenance"]["config"]["config_hash"] = "0" * 64
            for p_info in data["provenance"]["prompts"].values():
                p_info["prompt_hash"] = "0" * 64

            for cell in data["cells"].values():
                del cell["blocks"]

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            report = validator.validate()

            self.assertEqual(report.overall_verdict, "INCOMPLETE_NOT_PROVEN")
            self.assertFalse(report.all_cells_block_pairing_pass)
            for c in report.cells:
                self.assertFalse(c.block_pairing_passed)
                self.assertIn("UNVERIFIED", c.block_pairing_status)

    def test_malformed_block_missing_arm_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            first_cell = next(iter(data["cells"].values()))
            del first_cell["blocks"][0]["runs"]["auto"]

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.BlockPairingError):
                validator.validate()

    def test_block_duplicate_run_index_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            first_cell = next(iter(data["cells"].values()))
            first_cell["blocks"][1]["runs"]["off"] = 2

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.BlockPairingError):
                validator.validate()


class AdversarialProvenanceAndHashTests(unittest.TestCase):
    """Adversarial tests for model/package/prompt/config SHA-256 hashes and identities."""

    def setUp(self) -> None:
        self.fixture_dir = Path(__file__).parent / "fixtures" / "mtp_protocol" / "synthetic_positive"

    def test_malformed_model_sha256_rejected_in_product_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["synthetic"] = False
            data["provenance"]["model"]["file_sha256"] = "not_a_valid_64_hex_hash"
            data["provenance"]["package"]["digest_sha256"] = "0" * 64
            data["provenance"]["config"]["config_hash"] = "0" * 64

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.ManifestError) as ctx:
                validator.validate()
            self.assertIn("model.file_sha256", str(ctx.exception))

    def test_malformed_package_sha256_rejected_in_product_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["synthetic"] = False
            data["provenance"]["model"]["file_sha256"] = "0" * 64
            data["provenance"]["package"]["digest_sha256"] = "short_hash"
            data["provenance"]["config"]["config_hash"] = "0" * 64

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.ManifestError) as ctx:
                validator.validate()
            self.assertIn("package.digest_sha256", str(ctx.exception))

    def test_missing_config_hash_rejected_in_product_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["synthetic"] = False
            data["provenance"]["model"]["file_sha256"] = "0" * 64
            data["provenance"]["package"]["digest_sha256"] = "0" * 64
            data["provenance"]["config"]["config_hash"] = ""

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.ManifestError) as ctx:
                validator.validate()
            self.assertIn("config.config_hash", str(ctx.exception))

    def test_missing_prompt_hash_rejected_in_product_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["synthetic"] = False
            data["provenance"]["model"]["file_sha256"] = "0" * 64
            data["provenance"]["package"]["digest_sha256"] = "0" * 64
            data["provenance"]["config"]["config_hash"] = "0" * 64
            data["provenance"]["prompts"]["spec-chat-open"]["prompt_hash"] = "invalid"

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.ManifestError) as ctx:
                validator.validate()
            self.assertIn("prompt 'spec-chat-open'", str(ctx.exception))


class AdversarialMandatoryGatesTests(unittest.TestCase):
    """Adversarial tests for structured gate reports and assertion-only evidence."""

    def setUp(self) -> None:
        self.fixture_dir = Path(__file__).parent / "fixtures" / "mtp_protocol" / "synthetic_positive"

    def test_bare_assertion_gate_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["mandatory_gates"]["active_mtp_engagement"] = {
                "status": "PASS",
                "evidence_type": "assertion_only",
                "evidence": "I assert this passed"
            }

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            report = validator.validate()

            self.assertEqual(report.overall_verdict, "INCOMPLETE_NOT_PROVEN")
            gate = next(g for g in report.mandatory_gates if g.name == "active_mtp_engagement")
            self.assertFalse(gate.is_pass)
            self.assertEqual(gate.status, "NOT_PROVEN")

    def test_incomplete_assertions_count_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["mandatory_gates"]["rollback_under_mismatch"]["verification"]["assertions_passed"] = 8
            data["mandatory_gates"]["rollback_under_mismatch"]["verification"]["assertions_total"] = 10

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            report = validator.validate()

            self.assertEqual(report.overall_verdict, "INCOMPLETE_NOT_PROVEN")
            gate = next(g for g in report.mandatory_gates if g.name == "rollback_under_mismatch")
            self.assertFalse(gate.is_pass)

    def test_eog_unobserved_eos_is_partially_proven(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["mandatory_gates"]["eog_handling"]["details"] = "No EOS observed on fixtures; length saturation only"

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            report = validator.validate()

            self.assertEqual(report.overall_verdict, "INCOMPLETE_NOT_PROVEN")
            gate = next(g for g in report.mandatory_gates if g.name == "eog_handling")
            self.assertFalse(gate.is_pass)
            self.assertEqual(gate.status, "PARTIALLY_PROVEN")


class AdversarialIngestionAndSchemaTests(unittest.TestCase):
    """Adversarial tests for manifest schema, top-level keys, and backend/sampling parameters."""

    def setUp(self) -> None:
        self.fixture_dir = Path(__file__).parent / "fixtures" / "mtp_protocol" / "synthetic_positive"

    def test_missing_manifest_file(self) -> None:
        validator = vmp.MtpProtocolValidator("/nonexistent/manifest.json")
        with self.assertRaises(vmp.ManifestError):
            validator.validate()

    def test_unsupported_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["schema_version"] = 999
            m_path.write_text(json.dumps(data), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.ManifestError):
                validator.validate()

    def test_unsupported_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["protocol"] = "wrong-protocol"
            m_path.write_text(json.dumps(data), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.ManifestError):
                validator.validate()

    def test_non_d3d12_backend_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["provenance"]["backend"] = "vulkan"
            m_path.write_text(json.dumps(data), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.ManifestError):
                validator.validate()

    def test_invalid_sampling_non_greedy_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["provenance"]["sampling"]["greedy"] = 0
            m_path.write_text(json.dumps(data), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.ManifestError):
                validator.validate()

    def test_invalid_sampling_zero_seed_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["provenance"]["sampling"]["seed"] = 0
            m_path.write_text(json.dumps(data), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.ManifestError):
                validator.validate()

    def test_synthetic_flag_typing_rejected_if_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["synthetic"] = "true"
            m_path.write_text(json.dumps(data), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.ManifestError):
                validator.validate()


class AdversarialGridTests(unittest.TestCase):
    """Adversarial tests for grid dimensions, prompts, lengths, and arms."""

    def setUp(self) -> None:
        self.fixture_dir = Path(__file__).parent / "fixtures" / "mtp_protocol" / "synthetic_positive"

    def test_grid_reduced_prompt_count_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["grid"]["prompts"] = ["spec-chat-open", "spec-code-edit"]

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.GridIncompleteError):
                validator.validate()

    def test_grid_expanded_prompt_count_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["grid"]["prompts"] = ["spec-chat-open", "spec-code-edit", "standard-512", "extra-prompt"]

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.GridIncompleteError):
                validator.validate()

    def test_grid_duplicate_prompts_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["grid"]["prompts"] = ["spec-chat-open", "spec-chat-open", "standard-512"]

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.GridIncompleteError):
                validator.validate()

    def test_grid_invalid_output_lengths_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["grid"]["output_lengths"] = [64, 128]

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.GridIncompleteError):
                validator.validate()

    def test_extra_cell_matrix_key_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["cells"]["unapproved_extra_cell"] = copy.deepcopy(data["cells"]["spec-chat-open_64"])

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.GridIncompleteError):
                validator.validate()

    def test_missing_cell_matrix_key_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            del data["cells"]["spec-chat-open_64"]

            m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.GridIncompleteError):
                validator.validate()


class AdversarialCsvAndTokenSidecarTests(unittest.TestCase):
    """Adversarial tests for CSV format, duplicate JSON keys, and token integrity."""

    def setUp(self) -> None:
        self.fixture_dir = Path(__file__).parent / "fixtures" / "mtp_protocol" / "synthetic_positive"

    def test_duplicate_json_keys_rejected(self) -> None:
        raw_json = '{"schema_version": 1, "protocol": "plan003-paired-mtp", "synthetic": true, "synthetic": false}'
        with self.assertRaises(vmp.ManifestError):
            vmp.strict_json_loads(raw_json)

    def test_rfc4180_quotes_in_unquoted_field_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "bad.csv"
            header = ",".join(vmp.REQUIRED_CSV_CORE)
            bad_row = '"qwen35"extra,quant,backend,1024,4,10.0,15.0,100.0,50.0,200.0,1000.0,512,64,64,host,2026-10-05,1'
            csv_path.write_text(f"{header}\n{bad_row}\n", encoding="utf-8")

            prov = vmp.ProvenanceConfig(
                model_declared_name="qwen35",
                model_file_sha256="0"*64,
                package_declared_identity="pkg",
                package_digest_sha256="0"*64,
                backend="d3d12",
                gpu_layers=28,
                config_hash="0"*64,
                effective_params={},
                prompts={},
                sampling={"greedy": 1, "seed": 1},
                device_host="host",
                warmup_runs=1,
                recorded_runs=3,
                synthetic=True
            )

            with self.assertRaises(vmp.CsvFormatError):
                vmp.parse_and_validate_csv(csv_path, prov, 64, "off")

    def test_negative_token_id_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tok_path = Path(tmp_dir) / "tokens.txt"
            tok_path.write_text("10\n20\n-5\n30\n", encoding="utf-8")
            with self.assertRaises(vmp.TokenFormatError):
                vmp.parse_token_sidecar(tok_path, 4, 1)

    def test_truncated_token_sequence_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tok_path = Path(tmp_dir) / "tokens.txt"
            tok_path.write_text("10\n20\n30\n", encoding="utf-8")
            with self.assertRaises(vmp.TokenFormatError) as ctx:
                vmp.parse_token_sidecar(tok_path, 64, 1)
            self.assertIn("truncated", str(ctx.exception))

    def test_excess_token_sequence_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tok_path = Path(tmp_dir) / "tokens.txt"
            tok_path.write_text("10\n20\n30\n40\n50\n", encoding="utf-8")
            with self.assertRaises(vmp.TokenFormatError) as ctx:
                vmp.parse_token_sidecar(tok_path, 3, 1)
            self.assertIn("excess", str(ctx.exception))

    def test_malformed_fnv1a_footer_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tok_path = Path(tmp_dir) / "tokens.txt"
            tok_path.write_text("10\n20\nfnv1a=not_16_hex\n", encoding="utf-8")
            with self.assertRaises(vmp.TokenFormatError):
                vmp.parse_token_sidecar(tok_path, 2, 1)

    def test_csv_sha256_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["cells"]["spec-chat-open_64"]["arms"]["off"]["csv_sha256"] = "f" * 64
            m_path.write_text(json.dumps(data), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.IntegrityError):
                validator.validate()

    def test_token_sha256_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["cells"]["spec-chat-open_64"]["arms"]["off"]["token_sidecars"]["1"]["sha256"] = "e" * 64
            m_path.write_text(json.dumps(data), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.IntegrityError):
                validator.validate()

    def test_intra_arm_non_determinism_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copytree(self.fixture_dir, tmp_path / "fixture")
            m_path = tmp_path / "fixture" / "manifest.json"

            # Corrupt run 3 tokens for auto arm in spec-chat-open_64
            t_path = tmp_path / "fixture" / "spec-chat-open-64-auto.run3.tokens"
            lines = t_path.read_text(encoding="utf-8").splitlines()
            lines[5] = "99999"
            t_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            # Update hash in manifest to match the edited file
            new_sha = hashlib.sha256(t_path.read_bytes()).hexdigest()
            data = json.loads(m_path.read_text(encoding="utf-8"))
            data["cells"]["spec-chat-open_64"]["arms"]["auto"]["token_sidecars"]["3"]["sha256"] = new_sha
            m_path.write_text(json.dumps(data), encoding="utf-8")

            validator = vmp.MtpProtocolValidator(m_path)
            with self.assertRaises(vmp.DeterminismError):
                validator.validate()


class StatisticalMathAndComparisonTests(unittest.TestCase):
    """Unit tests for statistical math and sequence parity helpers."""

    def test_student_t_critical_values(self) -> None:
        self.assertEqual(vmp.get_t_critical_95(1), 12.706)
        self.assertEqual(vmp.get_t_critical_95(2), 4.303)
        self.assertEqual(vmp.get_t_critical_95(3), 3.182)
        self.assertEqual(vmp.get_t_critical_95(5), 2.571)

    def test_median_mean_std_calculations(self) -> None:
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(vmp.calculate_median(data), 30.0)
        self.assertEqual(vmp.calculate_mean(data), 30.0)
        self.assertAlmostEqual(vmp.calculate_std(data), 15.811388, places=5)

    def test_hodges_lehmann_median(self) -> None:
        data = [1.0, 2.0, 10.0]
        hl = vmp.calculate_hodges_lehmann(data)
        self.assertEqual(hl, 3.75)

    def test_compare_token_sequences_matching(self) -> None:
        seq = [101, 102, 103, 104]
        res = vmp.compare_token_sequences(seq, seq)
        self.assertTrue(res.is_match)
        self.assertEqual(res.matched_prefix_len, 4)

    def test_compare_token_sequences_divergence(self) -> None:
        seq1 = [101, 102, 103, 104]
        seq2 = [101, 102, 999, 104]
        res = vmp.compare_token_sequences(seq1, seq2, "off", "mtp")
        self.assertFalse(res.is_match)
        self.assertEqual(res.earliest_mismatch_pos_0idx, 2)
        self.assertEqual(res.earliest_mismatch_pos_1idx, 3)
        self.assertEqual(res.left_token, 103)
        self.assertEqual(res.right_token, 999)

    def test_compare_token_sequences_length_divergence(self) -> None:
        seq1 = [101, 102, 103]
        seq2 = [101, 102, 103, 104]
        res = vmp.compare_token_sequences(seq1, seq2, "off", "mtp")
        self.assertFalse(res.is_match)
        self.assertEqual(res.earliest_mismatch_pos_0idx, 3)
        self.assertEqual(res.matched_prefix_len, 3)


if __name__ == "__main__":
    unittest.main()
