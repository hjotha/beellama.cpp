#!/usr/bin/env python3
"""CPU-only unit tests for the F5 evidence harness.

Run from the repository root:

    python3 -m unittest discover -s tools/occupancy-f5/tests -v

These tests never launch a server, a build, a GPU workload or a network
request. All evidence used here is the synthetic fixture set produced by
``tools/occupancy-f5/fixtures/make_fixtures.py``.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FIXTURES))

import f5_harness as h  # noqa: E402
import make_fixtures as mf  # noqa: E402

MIB = h.MIB


def gate_by_id(summary: dict, gate_id: str) -> dict:
    for gate in summary["gates"]:
        if gate["id"] == gate_id:
            return gate
    raise AssertionError(f"gate {gate_id} not present in summary")


def run_cli(args: list[str]) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = h.main(args)
        except SystemExit as exc:  # argparse usage errors
            code = int(exc.code) if isinstance(exc.code, int) else 2
    return code, out.getvalue()


class F5HarnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = json.loads((FIXTURES / "expected-context.json").read_text())
        cls.source = (ROOT / "f5_harness.py").read_text(encoding="utf-8")

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="f5-test-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def load(self, name: str) -> dict:
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))

    def measured(self, mutate=None, rebind=True) -> dict:
        bundle = copy.deepcopy(self.load("complete-synthetic.json"))
        bundle["evidence_kind"] = "measured"
        bundle["provenance"]["evidence_origin"] = "campaign_measured"
        context_hash = h.context_binding(self.context)
        provenance_hash = h.canonical_sha256(bundle["provenance"])
        for run in bundle["runs"]:
            if run["warmup"]:
                continue
            for artifact in run["artifacts"]:
                path = self.root / artifact["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = mf.raw_payload(run["run_id"], context_hash, provenance_hash)
                path.write_bytes(payload)
                artifact["sha256"] = h.sha256_of(payload)
        if mutate is not None:
            mutate(bundle)
        if rebind:
            h.apply_bindings(bundle, self.context)
        return bundle

    @staticmethod
    def select_runs(bundle: dict, case_id: str, arm: str, measured_only=True):
        return [
            r
            for r in bundle["runs"]
            if r["case_id"] == case_id
            and r["arm"] == arm
            and (not measured_only or r["warmup"] is False)
        ]

    # ------------------------------------------------------------------ #
    # 1. guards on the harness itself
    # ------------------------------------------------------------------ #
    def test_harness_imports_neither_os_nor_subprocess(self) -> None:
        for forbidden in ("import os", "import subprocess", "os.system", "Popen("):
            self.assertNotIn(forbidden, self.source)
        self.assertFalse(hasattr(h, "os"))
        self.assertFalse(hasattr(h, "subprocess"))

    def test_schema_doc_is_machine_readable(self) -> None:
        doc = h.build_schema_doc()
        self.assertEqual(doc["evidence_schema"], "f5-evidence-v2")
        self.assertEqual(doc["exit_codes"]["accept_or_admissible"], 0)
        self.assertEqual(doc["exit_codes"]["reject_or_inadmissible"], 1)
        self.assertEqual(doc["exit_codes"]["usage"], 2)
        self.assertIn("telemetry", doc)
        self.assertIn("measured", doc["evidence_kind"]["allowed"])
        self.assertIn("synthetic_fixture", doc["evidence_kind"]["allowed"])

    # ------------------------------------------------------------------ #
    # 2. protocol / schedule
    # ------------------------------------------------------------------ #
    def test_prescribed_cases_follow_plan_lengths(self) -> None:
        cases = h.prescribed_cases(102400)
        ids = [c["case_id"] for c in cases]
        self.assertEqual(
            ids, ["below_p_96000", "above_p_105000", "above_p_115000", "above_p_130560"]
        )
        for case in cases:
            self.assertIn(case["baseline_arm"], (h.XXL_ARM, h.XXXL_ARM))
            self.assertEqual(case["baseline_arm"], h.XXXL_ARM if case["kind"] == "above_p" else h.XXL_ARM)

    def test_p_plus_256_case_is_prescribed_when_p_differs(self) -> None:
        cases = h.prescribed_cases(99840)
        ids = [c["case_id"] for c in cases]
        self.assertIn("above_p_100096", ids)
        self.assertNotIn("above_p_100096", [c["case_id"] for c in h.prescribed_cases(102400)])
        other = copy.deepcopy(self.context)
        other["expected"]["p"] = 99840
        report = h.validate(self.load("complete-synthetic.json"), other, None)
        self.assertEqual(report["structural_status"], "fail")
        self.assertIn("missing_evidence:case:above_p_100096", report["reasons"])

    def test_schedule_alternates_five_runs_per_arm_after_warmup(self) -> None:
        schedule = h.build_schedule(self.context)
        self.assertEqual(schedule["schema_version"], "f5-schedule-v1")
        self.assertEqual(schedule["total_runs"], 48)
        for case in h.prescribed_cases(102400):
            case_id = case["case_id"]
            entries = [e for e in schedule["sequence"] if e["case_id"] == case_id]
            self.assertTrue(entries[0]["warmup"], f"{case_id} must start with a warmup")
            warmups = [e for e in entries if e["warmup"]]
            self.assertEqual(
                {w["arm"] for w in warmups},
                {case["baseline_arm"], "split"},
                f"{case_id} must warm both arms",
            )
            measured = [e for e in entries if not e["warmup"]]
            self.assertEqual(len(measured), 10)
            first_measured = next(i for i, e in enumerate(entries) if not e["warmup"])
            self.assertTrue(all(e["warmup"] for e in entries[:first_measured]))
            self.assertEqual(len(warmups), 2)
            self.assertEqual([e["arm"] for e in measured[:4]], [
                case["baseline_arm"], "split", case["baseline_arm"], "split",
            ])
            for arm in (case["baseline_arm"], "split"):
                count = sum(1 for e in measured if e["arm"] == arm)
                self.assertEqual(count, 5)
            indices = [e["sequence_index"] for e in entries]
            self.assertEqual(indices, sorted(indices))
            self.assertEqual(len(set(indices)), len(indices))
        for entry in schedule["sequence"]:
            self.assertIn("schedule_step", entry)

    # ------------------------------------------------------------------ #
    # 3. validator: reason codes for every defect fixture
    # ------------------------------------------------------------------ #
    def test_complete_fixture_is_structurally_sound_but_synthetic(self) -> None:
        report = h.validate(self.load("complete-synthetic.json"), self.context, None)
        self.assertEqual(report["structural_status"], "pass")
        self.assertEqual(report["verdict"], "reject")
        self.assertEqual(report["evidence_kind"], "synthetic_fixture")
        self.assertEqual(report["reasons"], [])
        self.assertTrue(report["kind_reason"].startswith("evidence_kind:"))

    def test_malformed_fixture(self) -> None:
        code, payload = run_cli(
            ["validate", str(FIXTURES / "malformed.json"), "--context", str(FIXTURES / "expected-context.json")]
        )
        self.assertEqual(code, 1)
        report = json.loads(payload)
        self.assertEqual(report["structural_status"], "fail")
        self.assertTrue(report["reasons"][0].startswith("malformed_json:"))

    def test_missing_evidence_fixture(self) -> None:
        report = h.validate(self.load("missing-evidence.json"), self.context, None)
        self.assertEqual(report["structural_status"], "fail")
        reasons = report["reasons"]
        self.assertTrue(any(r.startswith("missing_field:runs[") for r in reasons))
        self.assertIn("missing_evidence:quality", reasons)
        self.assertIn("missing_field:cycles", reasons)

    def test_nonfinite_fixture(self) -> None:
        report = h.validate(self.load("nonfinite.json"), self.context, None)
        self.assertEqual(report["structural_status"], "fail")
        self.assertTrue(any(r.startswith("non_finite:runs[") for r in report["reasons"]))
        self.assertTrue(any(r.startswith("non_finite:quality[") for r in report["reasons"]))

    def test_wrong_provenance_fixture(self) -> None:
        report = h.validate(self.load("wrong-provenance.json"), self.context, None)
        self.assertEqual(report["structural_status"], "fail")
        self.assertTrue(
            any(r.startswith("provenance_mismatch:model_sha256:") for r in report["reasons"])
        )

    def test_insufficient_repetitions_fixture(self) -> None:
        report = h.validate(self.load("insufficient-repetitions.json"), self.context, None)
        self.assertEqual(report["structural_status"], "fail")
        self.assertTrue(
            any(r.startswith("insufficient_repetitions:") and "got=3:need=5" in r
                for r in report["reasons"])
        )

    def test_protocol_mismatch_when_repetitions_differ(self) -> None:
        context = copy.deepcopy(self.context)
        context["protocol"]["measured_runs_per_arm"] = 3
        report = h.validate(self.load("complete-synthetic.json"), context, None)
        self.assertTrue(
            any(r.startswith("protocol_mismatch:measured_runs_per_arm:") for r in report["reasons"])
        )

    # ------------------------------------------------------------------ #
    # 4. admissible (measured) evidence: acceptance and rejection paths
    # ------------------------------------------------------------------ #
    def test_measured_evidence_with_verified_artifacts_is_accepted(self) -> None:
        report = h.validate(self.measured(), self.context, self.root)
        self.assertEqual(report["verdict"], "accept")
        self.assertEqual(report["evidence_kind"], "measured")
        self.assertEqual(report["structural_status"], "pass")
        self.assertEqual(report["reasons"], [])

    def test_measured_evidence_without_artifact_root_is_rejected(self) -> None:
        report = h.validate(self.measured(), self.context, None)
        self.assertEqual(report["verdict"], "reject")
        self.assertIn("artifact_root_required", report["reasons"])

    def test_artifact_hash_mismatch_is_rejected(self) -> None:
        def tamper(bundle: dict) -> None:
            run = next(r for r in bundle["runs"] if not r["warmup"])
            (self.root / run["artifacts"][0]["path"]).write_bytes(b"tampered\n")

        report = h.validate(self.measured(tamper), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(
            any(r.startswith("artifact_hash_mismatch:") for r in report["reasons"])
        )

    def test_missing_artifact_file_is_rejected(self) -> None:
        def drop(bundle: dict) -> None:
            run = next(r for r in bundle["runs"] if not r["warmup"])
            (self.root / run["artifacts"][0]["path"]).unlink()

        report = h.validate(self.measured(drop), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(any(r.startswith("artifact_missing:") for r in report["reasons"]))

    def test_vulkan_attention_below_p_is_a_route_mismatch(self) -> None:
        def flip(bundle: dict) -> None:
            run = next(
                r for r in bundle["runs"]
                if not r["warmup"] and r["case_id"] == "below_p_96000"
            )
            run["routes"]["vulkan_attention_executed"] = True

        report = h.validate(self.measured(flip), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(any(r.startswith("route_mismatch:") for r in report["reasons"]))

    def test_warmup_is_required_per_case(self) -> None:
        def drop_warmup(bundle: dict) -> None:
            bundle["runs"] = [
                r for r in bundle["runs"]
                if not (r["warmup"] and r["case_id"] == "above_p_115000")
            ]

        report = h.validate(self.measured(drop_warmup), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(
            any(
                r.startswith("missing_evidence:warmup:above_p_115000:")
                and "got=0:need=1" in r
                for r in report["reasons"]
            ),
            report["reasons"],
        )
        self.assertTrue(
            any(
                r.startswith("missing_evidence:warmup:above_p_115000:")
                for r in report["reasons"]
            )
        )

    def test_alternating_order_is_enforced(self) -> None:
        def swap(bundle: dict) -> None:
            measured = [r for r in bundle["runs"] if not r["warmup"] and r["case_id"] == "below_p_96000"]
            i = bundle["runs"].index(measured[1])
            j = bundle["runs"].index(measured[2])
            bundle["runs"][i]["arm"], bundle["runs"][j]["arm"] = (
                bundle["runs"][j]["arm"],
                bundle["runs"][i]["arm"],
            )

        report = h.validate(self.measured(swap), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(any("alternation_violation:" in r for r in report["reasons"]))

    # ------------------------------------------------------------------ #
    # 5. summariser: refuses inadmissible evidence, computes gates
    # ------------------------------------------------------------------ #
    def test_summarize_reports_no_numbers_for_synthetic_evidence(self) -> None:
        summary, code = h.summarize(self.load("complete-synthetic.json"), self.context, None)
        self.assertEqual(code, 1)
        self.assertFalse(summary["admissible"])
        self.assertIsNone(summary["measured"])
        self.assertEqual(summary["gates"], [])
        self.assertEqual(summary["gates_pass"] + summary["gates_fail"] + summary["gates_incomplete"], 0)
        self.assertTrue(any("no number" in n for n in summary["notes"]))

    def test_summarize_passes_every_gate_on_complete_measured_evidence(self) -> None:
        summary, code = h.summarize(self.measured(), self.context, self.root, True)
        self.assertTrue(summary["admissible"])
        self.assertEqual(summary["gates_fail"], 0)
        self.assertEqual(summary["gates_incomplete"], 0)
        self.assertEqual(code, 0)
        self.assertGreater(summary["gates_pass"], 0)
        for gate in summary["gates"]:
            self.assertEqual(gate["status"], "pass", f"{gate['id']} -> {gate}")

        prefill = gate_by_id(summary, "below_p_96000.prefill_regression")
        self.assertAlmostEqual(prefill["value"], 30500.0 / 30000.0, places=6)
        self.assertEqual(prefill["limit"], 1.05)
        above = gate_by_id(summary, "above_p_105000.prefill_improvement")
        self.assertLessEqual(above["value"], 0.90)
        self.assertEqual(above["limit"], 0.90)
        self.assertEqual(gate_by_id(summary, "below_p_96000.route_vulkan_attention")["limit"], False)
        self.assertEqual(gate_by_id(summary, "above_p_130560.route_vulkan_attention")["limit"], True)
        memory = summary["measured"]["memory"]
        self.assertGreaterEqual(
            memory["cuda_guard_margin_min_bytes"], memory["cuda_guard_limit_bytes"]
        )
        self.assertEqual(memory["swap_in_bytes_max"], 0)
        self.assertEqual(summary["measured"]["cycles"]["count"], 10)
        self.assertLessEqual(
            max(summary["measured"]["cycles"]["retained_growth_bytes"].values()),
            summary["measured"]["cycles"]["limit_bytes"],
        )

    def test_below_p_regression_gate_fails_over_5_percent(self) -> None:
        def slower(bundle: dict) -> None:
            for run in self.select_runs(bundle, "below_p_96000", "split"):
                run["prefill_ms"] = 30000.0 * 1.10
                run["ttft_ms"] = 30200.0 * 1.10

        summary, _ = h.summarize(self.measured(slower), self.context, self.root, True)
        gate = gate_by_id(summary, "below_p_96000.prefill_regression")
        self.assertEqual(gate["status"], "fail")
        self.assertGreater(gate["value"], 1.05)
        self.assertEqual(summary["gates_fail"], 2)
        self.assertEqual(_, 1)

    def test_above_p_improvement_gate_fails_when_not_0_90x(self) -> None:
        def marginal(bundle: dict) -> None:
            for run in self.select_runs(bundle, "above_p_105000", "split"):
                run["prefill_ms"] = 40000.0 * 0.95
                run["ttft_ms"] = 40200.0 * 0.95

        summary, _ = h.summarize(self.measured(marginal), self.context, self.root, True)
        gate = gate_by_id(summary, "above_p_105000.prefill_improvement")
        self.assertEqual(gate["status"], "fail")
        self.assertGreater(gate["value"], 0.90)

    def test_decode_p95_gate_fails_over_1_05x(self) -> None:
        def slower_decode(bundle: dict) -> None:
            for run in self.select_runs(bundle, "above_p_115000", "split"):
                run["token_latencies_ms"] = [v * 1.20 for v in run["token_latencies_ms"]]

        summary, _ = h.summarize(self.measured(slower_decode), self.context, self.root, True)
        gate = gate_by_id(summary, "above_p_115000.decode_p95_regression")
        self.assertEqual(gate["status"], "fail")
        self.assertGreater(gate["value"], 1.05)

    def test_quality_gate_fails_when_kl_delta_exceeds_limit(self) -> None:
        def noisy(bundle: dict) -> None:
            for entry in bundle["quality"]:
                if entry["case_id"] == "below_p_96000" and entry["arm"] == "split":
                    entry["kl_mean_nat_per_token"] = 0.0020 + h.QUALITY_KL_DELTA_LIMIT + 0.001

        summary, _ = h.summarize(self.measured(noisy), self.context, self.root, True)
        gate = gate_by_id(summary, "below_p_96000.quality_kl_mean")
        self.assertEqual(gate["status"], "fail")

    def test_capacity_and_cycle_gates_fail_when_broken(self) -> None:
        def broken(bundle: dict) -> None:
            bundle["capacity"]["over_capacity_rejected"] = False
            bundle["cycles"][-1]["retained_bytes"]["cuda"] += 100 * MIB

        summary, _ = h.summarize(self.measured(broken), self.context, self.root, True)
        self.assertEqual(gate_by_id(summary, "capacity.reject_over_capacity")["status"], "fail")
        growth = gate_by_id(summary, "cycles.retained_growth")
        self.assertEqual(growth["status"], "fail")
        self.assertGreater(growth["value"], growth["limit"])

    def test_restore_gate_fails_when_restore_is_not_clean(self) -> None:
        def dirty(bundle: dict) -> None:
            for entry in bundle["restore"]:
                if entry["length_tokens"] == 130560:
                    entry["ok"] = False

        summary, code = h.summarize(self.measured(dirty), self.context, self.root, True)
        self.assertEqual(gate_by_id(summary, "restore.130560")["status"], "fail")
        self.assertEqual(summary["gates_fail"], 1)
        self.assertEqual(code, 1)

    # ------------------------------------------------------------------ #
    # 6. CLI exit codes and subcommands
    # ------------------------------------------------------------------ #
    def test_cli_validate_exit_codes(self) -> None:
        ctx = str(FIXTURES / "expected-context.json")
        code, _ = run_cli(["validate", str(FIXTURES / "complete-synthetic.json"), "--context", ctx])
        self.assertEqual(code, 1)
        code, _ = run_cli(["schema"])
        self.assertEqual(code, 0)
        with self.assertRaises(SystemExit) as ctx_exit:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                h.main(["validate"])
        self.assertEqual(ctx_exit.exception.code, 2)

    def test_cli_summarize_requires_gates_for_exit_zero(self) -> None:
        measured_path = self.root / "measured.json"
        measured_path.write_text(json.dumps(self.measured()), encoding="utf-8")
        args = [
            "summarize",
            str(measured_path),
            "--context",
            str(FIXTURES / "expected-context.json"),
            "--artifact-root",
            str(self.root),
            "--require-gates-pass",
        ]
        code, payload = run_cli(args)
        self.assertEqual(code, 0)
        summary = json.loads(payload)
        self.assertTrue(summary["admissible"])
        self.assertEqual(summary["gates_fail"], 0)

    def test_cli_init_writes_scaffold_that_is_always_rejected(self) -> None:
        out = self.root / "init.json"
        code, _ = run_cli(
            [
                "init",
                "--context",
                str(FIXTURES / "expected-context.json"),
                "--out",
                str(out),
            ]
        )
        self.assertEqual(code, 0)
        bundle = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(bundle["evidence_kind"], "synthetic_fixture")
        report = h.validate(bundle, self.context, None)
        self.assertEqual(report["verdict"], "reject")

    def test_cli_rejects_missing_context_file_with_usage_exit(self) -> None:
        code, _ = run_cli(
            ["validate", str(FIXTURES / "complete-synthetic.json"), "--context", str(self.root / "nope.json")]
        )
        self.assertEqual(code, 2)

    def test_cli_plan_and_template_write_expected_output(self) -> None:
        plan_path = self.root / "plan.json"
        code, _ = run_cli(
            ["plan", "--context", str(FIXTURES / "expected-context.json"), "--out", str(plan_path)]
        )
        self.assertEqual(code, 0)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(plan["schema_version"], "f5-schedule-v1")

        template_path = self.root / "templates.json"
        code, _ = run_cli(
            [
                "template",
                "--context",
                str(FIXTURES / "expected-context.json"),
                "--out",
                str(template_path),
            ]
        )
        self.assertEqual(code, 0)
        templates = json.loads(template_path.read_text(encoding="utf-8"))
        blob = json.dumps(templates)
        self.assertIn("--spec-type none", blob)
        self.assertIn("--spec-draft-n-max 0", blob)
        self.assertIn("LLAMA_KV_POSITION_SPLIT_P", blob)
        self.assertIn("NOT RUN", blob)
        self.assertIn("{PLACEHOLDER}", blob)
        self.assertIn("{KV_AND_OVERFLOW_FLAGS_FROM_F3_F4_PROFILE}", blob)
        self.assertEqual(templates["status"], "NOT RUN")


    # ------------------------------------------------------------------ #
    # 7. reviewer probes: context binding, exact restore, coverage hashes
    # ------------------------------------------------------------------ #
    def test_fixture_context_is_fully_valid(self) -> None:
        self.assertEqual(h.context_problems(self.context), [])

    def test_empty_expected_context_is_rejected(self) -> None:
        context = copy.deepcopy(self.context)
        context["expected"] = {}
        report = h.validate(self.measured(), context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(
            any(r.startswith("bad_context:") for r in report["reasons"]),
            report["reasons"],
        )

    def test_missing_expected_field_is_rejected(self) -> None:
        context = copy.deepcopy(self.context)
        context["expected"].pop("model_sha256")
        report = h.validate(self.measured(), context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(
            any("bad_context:" in r and "model_sha256" in r for r in report["reasons"]),
            report["reasons"],
        )

    def test_zero_placeholder_hashes_in_context_rejected(self) -> None:
        context = copy.deepcopy(self.context)
        context["expected"]["model_sha256"] = "0" * 64
        context["expected"]["corpus_sha256"] = "0" * 64
        report = h.validate(self.measured(), context, self.root)
        self.assertEqual(report["verdict"], "reject")
        reasons = report["reasons"]
        self.assertTrue(any("model_sha256" in r and r.startswith("bad_context:") for r in reasons), reasons)
        self.assertTrue(any("corpus_sha256" in r and r.startswith("bad_context:") for r in reasons), reasons)

    def test_context_p_c_alignment_and_range(self) -> None:
        for p, c in [(100000, 131072), (102400, 102400), (102400, 130000), (0, 131072), (102400, 102656)]:
            context = copy.deepcopy(self.context)
            context["expected"]["p"] = p
            context["expected"]["c"] = c
            report = h.validate(self.measured(), context, self.root)
            self.assertEqual(report["verdict"], "reject", f"p={p} c={c}")
            self.assertTrue(
                any(r.startswith("bad_context:") for r in report["reasons"]),
                (p, c, report["reasons"]),
            )

    def test_context_placeholder_paths_and_zero_hashes_rejected_by_cli(self) -> None:
        context = copy.deepcopy(self.context)
        context["expected"]["model_path"] = "<target-model>.gguf"
        bad_context = self.root / "bad-context.json"
        bad_context.write_text(json.dumps(context), encoding="utf-8")
        code, _ = run_cli(
            [
                "validate",
                str(FIXTURES / "complete-synthetic.json"),
                "--context",
                str(bad_context),
            ]
        )
        self.assertEqual(code, 2)

    def test_placeholder_and_zero_hash_values_in_evidence_rejected(self) -> None:
        def placeholders(bundle: dict) -> None:
            bundle["provenance"]["driver"] = "<fill-me>"
            bundle["provenance"]["binary_sha256"] = "0" * 64

        report = h.validate(self.measured(placeholders), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        reasons = report["reasons"]
        self.assertTrue(any("placeholder_value:provenance.driver" in r for r in reasons), reasons)
        self.assertTrue(any("zero_hash:provenance.binary_sha256" in r for r in reasons), reasons)

    def test_synthetic_origin_cannot_be_claimed_as_measured(self) -> None:
        bundle = self.load("complete-synthetic.json")
        bundle["evidence_kind"] = "measured"
        report = h.validate(bundle, self.context, None)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(
            any(r.startswith("evidence_origin:") for r in report["reasons"]),
            report["reasons"],
        )

    def test_restore_partial_reuse_fails_gate(self) -> None:
        def partial(bundle: dict) -> None:
            for entry in bundle["restore"]:
                if entry["length_tokens"] == 130560:
                    entry["tokens_reused"] = 1
                    entry["prefix_reprefill_tokens"] = entry["length_tokens"] - 1

        summary, code = h.summarize(self.measured(partial), self.context, self.root, True)
        self.assertTrue(summary["admissible"], summary["reasons"])
        gate = gate_by_id(summary, "restore.130560")
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(code, 1)

    def test_restore_fallback_scenario_cannot_pass_gate(self) -> None:
        def fallback(bundle: dict) -> None:
            for entry in bundle["restore"]:
                if entry["length_tokens"] == 105000:
                    entry["scenario"] = "cache_miss_fallback"

        summary, code = h.summarize(self.measured(fallback), self.context, self.root, True)
        self.assertTrue(summary["admissible"], summary["reasons"])
        gate = gate_by_id(summary, "restore.105000")
        self.assertEqual(gate["status"], "fail")
        self.assertIn("cache_miss_fallback", gate["reason"] or "")
        self.assertEqual(code, 1)

    def test_restore_scenario_and_run_ids_required(self) -> None:
        def strip(bundle: dict) -> None:
            for entry in bundle["restore"]:
                if entry["length_tokens"] == 96000:
                    entry.pop("scenario")
                    entry.pop("run_ids")

        report = h.validate(self.measured(strip), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        reasons = report["reasons"]
        self.assertTrue(any("scenario" in r and "restore" in r for r in reasons), reasons)
        self.assertTrue(any("run_ids" in r and "restore" in r for r in reasons), reasons)

    def test_candidate_warmup_is_required(self) -> None:
        def drop(bundle: dict) -> None:
            bundle["runs"] = [
                r for r in bundle["runs"] if not (r["warmup"] and r["arm"] == "split")
            ]

        report = h.validate(self.measured(drop), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(
            any("missing_evidence:warmup:below_p_96000:split" in r for r in report["reasons"]),
            report["reasons"],
        )

    def test_warmups_must_precede_measurements(self) -> None:
        def reorder(bundle: dict) -> None:
            warm = next(
                r
                for r in bundle["runs"]
                if r["case_id"] == "below_p_96000" and r["warmup"] and r["arm"] == "split"
            )
            bundle["runs"].remove(warm)
            first_measured = next(
                i
                for i, r in enumerate(bundle["runs"])
                if r["case_id"] == "below_p_96000" and not r["warmup"]
            )
            bundle["runs"].insert(first_measured + 1, warm)

        report = h.validate(self.measured(reorder), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(
            any("sequence_index" in r and "below_p_96000" in r for r in report["reasons"]),
            report["reasons"],
        )

    def test_bindings_required(self) -> None:
        def unbind(bundle: dict) -> None:
            bundle.pop("bindings")

        report = h.validate(self.measured(unbind, rebind=False), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertIn("missing_field:bindings", report["reasons"])

    def test_binding_mismatch_detected(self) -> None:
        def tamper(bundle: dict) -> None:
            bundle["quality"][0]["perplexity"] += 1.0

        report = h.validate(self.measured(tamper, rebind=False), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(
            any(r.startswith("binding_mismatch:quality_sha256") for r in report["reasons"]),
            report["reasons"],
        )

    def test_log_header_must_match_run_and_context(self) -> None:
        bundle = self.measured()
        run = next(r for r in bundle["runs"] if not r["warmup"])
        artifact = run["artifacts"][0]
        path = self.root / artifact["path"]
        payload = path.read_bytes()
        _, sep, rest = payload.partition(b"\n")
        header = (
            f"f5-log-v1 run_id=not-the-right-run "
            f"context_sha256={'0' * 64} provenance_sha256={'0' * 64}"
        ).encode("utf-8")
        path.write_bytes(header + sep + rest)
        artifact["sha256"] = h.sha256_of(path.read_bytes())
        report = h.validate(bundle, self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(
            any("artifact_header_mismatch" in r for r in report["reasons"]),
            report["reasons"],
        )

    def test_quality_run_ids_must_reference_case_arms_measured_runs(self) -> None:
        def break_ids(bundle: dict) -> None:
            bundle["quality"][0]["run_ids"] = ["does-not-exist"]

        report = h.validate(self.measured(break_ids), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(any("run_ids" in r for r in report["reasons"]), report["reasons"])

    def test_capacity_and_cycles_run_ids_required(self) -> None:
        def strip(bundle: dict) -> None:
            bundle["capacity"].pop("run_ids")
            bundle["cycles"][0].pop("run_ids")

        report = h.validate(self.measured(strip), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        reasons = report["reasons"]
        self.assertIn("missing_field:capacity.run_ids", reasons)
        self.assertTrue(any(r.startswith("missing_field:cycles[0].run_ids") for r in reasons), reasons)

    def test_vulkan_attention_copy_bytes_required(self) -> None:
        def drop(bundle: dict) -> None:
            run = next(
                r
                for r in bundle["runs"]
                if not r["warmup"] and r["case_id"] == "below_p_96000"
            )
            del run["telemetry"]["vulkan"]["stages"]["prefill"]["attention_copy_bytes"]

        report = h.validate(self.measured(drop), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(
            any("attention_copy_bytes" in r for r in report["reasons"]),
            report["reasons"],
        )

    def test_vulkan_attention_copy_must_be_zero_below_p(self) -> None:
        def nonzero(bundle: dict) -> None:
            run = next(
                r
                for r in bundle["runs"]
                if not r["warmup"] and r["case_id"] == "below_p_96000"
            )
            run["telemetry"]["vulkan"]["stages"]["prefill"]["attention_copy_bytes"] = 4096

        report = h.validate(self.measured(nonzero), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(
            any(r.startswith("route_mismatch:") and "attention_copy_bytes" in r for r in report["reasons"]),
            report["reasons"],
        )

    def test_swap_out_bytes_required(self) -> None:
        def drop(bundle: dict) -> None:
            for run in bundle["runs"]:
                if not run["warmup"]:
                    run["telemetry"]["host"].pop("swap_out_bytes")

        report = h.validate(self.measured(drop), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(any("swap_out_bytes" in r for r in report["reasons"]), report["reasons"])

    def test_swap_out_activity_is_observational(self) -> None:
        def swapping(bundle: dict) -> None:
            for run in bundle["runs"]:
                if not run["warmup"]:
                    run["telemetry"]["host"]["swap_out_bytes"] = 1

        summary, code = h.summarize(self.measured(swapping), self.context, self.root, True)
        self.assertTrue(summary["admissible"], summary["reasons"])
        gate = gate_by_id(summary, "memory.no_oom")
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(summary["measured"]["memory"]["swap_out_bytes_max"], 1)
        self.assertEqual(code, 0)

    def test_low_headroom_and_swap_do_not_fail_memory(self) -> None:
        def low_memory(bundle: dict) -> None:
            total = bundle["provenance"]["hardware"]["vram_total_bytes"]
            for run in bundle["runs"]:
                if not run["warmup"]:
                    for stage in run["telemetry"]["cuda"]["stages"].values():
                        stage["peak_bytes"] = total - 10*h.MIB
                    run["telemetry"]["host"]["mem_available_min_bytes"] = 100*h.MIB
                    run["telemetry"]["host"]["swap_in_bytes"] = 48*h.MIB
                    run["telemetry"]["host"]["swap_out_bytes"] = 48*h.MIB
        summary, code = h.summarize(self.measured(low_memory), self.context, self.root, True)
        self.assertTrue(summary["admissible"], summary["reasons"])
        self.assertEqual(code, 0)
        self.assertEqual(gate_by_id(summary, "memory.no_oom")["status"], "pass")

    def test_oom_during_warmup_fails_memory(self) -> None:
        def oom(bundle: dict) -> None:
            next(run for run in bundle["runs"] if run["warmup"])["oom_count"] = 1
        summary, code = h.summarize(self.measured(oom), self.context, self.root, True)
        self.assertTrue(summary["admissible"], summary["reasons"])
        self.assertEqual(code, 1)
        self.assertEqual(gate_by_id(summary, "memory.no_oom")["status"], "fail")

    def test_missing_oom_evidence_is_rejected(self) -> None:
        def missing(bundle: dict) -> None:
            bundle["runs"][0].pop("oom_count")
        report = h.validate(self.measured(missing), self.context, self.root)
        self.assertEqual(report["verdict"], "reject")
        self.assertTrue(any("oom_count" in r for r in report["reasons"]))

    def test_summary_never_claims_genuine_f5_acceptance(self) -> None:
        summary, code = h.summarize(self.measured(), self.context, self.root, True)
        self.assertEqual(code, 0)
        acceptance = summary["acceptance"]
        self.assertTrue(acceptance["schema_admissibility"])
        self.assertTrue(acceptance["gate_arithmetic_only"])
        self.assertFalse(acceptance["genuine_f5_acceptance"])
        self.assertIn("F5 acceptance", acceptance["reason"])

        blocked, blocked_code = h.summarize(
            self.load("complete-synthetic.json"), self.context, None
        )
        self.assertEqual(blocked_code, 1)
        self.assertFalse(blocked["acceptance"]["schema_admissibility"])
        self.assertFalse(blocked["acceptance"]["gate_arithmetic_only"])
        self.assertFalse(blocked["acceptance"]["genuine_f5_acceptance"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
