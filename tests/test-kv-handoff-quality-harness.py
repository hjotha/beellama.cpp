#!/usr/bin/env python3
"""CPU self-check harness for the handoff-quality driver: pure validation and
metrics PLUS an end-to-end mock of main() with a stubbed own server and fake
HTTP (no real server, no GPU, no engine).

Run:
  python3 tests/test-kv-handoff-quality-harness.py
Exit code 0 only when every case passes.
"""

import importlib.util
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
_spec = importlib.util.spec_from_file_location(
    "test_kv_handoff_quality", _TOOLS_DIR / "test-kv-handoff-quality.py")
assert _spec and _spec.loader
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)

LOGPROB_ZERO = q.LOGPROB_ZERO
HANDOFF_MARKER = q.HANDOFF_MARKER
extract_cache_cadence = q.extract_cache_cadence
metrics = q.metrics
parse_top_logprobs = q.parse_top_logprobs
read_n_vocab_from_log = q.read_n_vocab_from_log
run_selfcheck = q.run_selfcheck
selfcheck_cases = q.selfcheck_cases
validate_distribution = q.validate_distribution
distribution_from_entries = q.distribution_from_entries

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILURES.append(f"{name}: {detail}")


# ---------------------------------------------------------------------------
# pure-function unit checks
# ---------------------------------------------------------------------------


def test_parse_sentinel():
    entries = [
        {"id": 0, "token": "a", "bytes": [0], "logprob": float(LOGPROB_ZERO)},
        {"id": 1, "token": "b", "bytes": [0], "logprob": -30.0},
    ]
    probs = parse_top_logprobs(entries)
    check("parse_sentinel_zero", probs[0] == 0.0, str(probs[0]))
    check("parse_sentinel_finite", abs(probs[1] - math.exp(-30.0)) < 1e-12, str(probs[1]))
    try:
        parse_top_logprobs([{"id": 2, "token": "c", "bytes": [0], "logprob": float("-inf")}])
        check("parse_minus_inf_rejected", False, "minus-inf accepted as zero")
    except ValueError:
        check("parse_minus_inf_rejected", True)


def test_metrics_known():
    n = 4
    p_pure = np.array([0.7, 0.2, 0.05, 0.05])
    p_mixed = np.array([0.6, 0.25, 0.1, 0.05])
    m = metrics(p_pure, p_mixed, ref_id=1)
    expected_tv = 0.5 * (abs(0.7 - 0.6) + abs(0.2 - 0.25) + abs(0.05 - 0.1) + 0.0)
    check("metrics_tv", abs(m["tv"] - expected_tv) < 1e-12, f"{m['tv']} vs {expected_tv}")
    check("metrics_max_delta", abs(m["max_prob_delta"] - 0.1) < 1e-12, str(m["max_prob_delta"]))
    check("metrics_greedy", m["greedy_pure"] == 0 and m["greedy_mixed"] == 0)
    check("metrics_ref_surprisal", abs(m["ref_surprisal"] - (-math.log(0.25))) < 1e-12,
          str(m["ref_surprisal"]))
    check("metrics_kl_not_null", m["kl"] is not None and not m["kl_inf"])
    check("metrics_kl_json_safe", json.dumps(m["kl"]).lower() not in ("infinity", "nan"),
          json.dumps(m["kl"]))


def test_kl_inf_explicit():
    p_pure = np.array([0.5, 0.5])
    p_mixed = np.array([1.0, 0.0])
    m = metrics(p_pure, p_mixed)
    check("kl_inf_flag", m["kl_inf"], str(m))
    check("kl_inf_null", m["kl"] is None, str(m["kl"]))
    check("kl_inf_json_safe", "infinity" not in json.dumps(m).lower(), json.dumps(m))
    check("kl_inf_tv", abs(m["tv"] - 0.5) < 1e-12, str(m["tv"]))


def test_cadence_gate():
    good = {"timings": {"cache_n": 3968, "prompt_n": 1032, "predicted_n": 1}}
    c = extract_cache_cadence(good)
    check("cadence_extract", c == (3968, 1032, 1), str(c))
    try:
        extract_cache_cadence({"timings": {}})
        check("cadence_missing", False, "missing timings accepted")
    except ValueError:
        check("cadence_missing", True)


def test_n_vocab_log():
    text = "print_info: n_vocab               = 248320\nprint_info: n_ctx_train = 32768\n"
    v = read_n_vocab_from_log(text)
    check("n_vocab_log", v == 248320, str(v))
    check("n_vocab_log_none", read_n_vocab_from_log("no vocab here") is None)


# ---------------------------------------------------------------------------
# end-to-end CPU mock: stub own server + fake HTTP driving q.main()
# ---------------------------------------------------------------------------

MOCK_N_VOCAB = 1024
# corpus ids must be valid token ids (< n_vocab) so the next-token reference
# stays in range; length >= max probe (6000) + 1
MOCK_IDS = [i % MOCK_N_VOCAB for i in range(7001)]
MOCK_CURRENT_SERVER = {"instance": None}  # registry for the fake HTTP -> server


class FakeServerInstance:
    """Stands in for the shared ServerInstance: records the extra argv and
    serves n_vocab in the log; markers are injected for qx cases."""

    def __init__(self, args, output_dir, case_id, extra_args):
        self.args = args
        self.output_dir = Path(output_dir)
        self.case_id = case_id
        self.extra_args = list(extra_args)
        self.log_path = self.output_dir / f"server-{case_id}.log"
        self.started = False
        self.marker_after_first_probe = "qx-" in case_id
        self.probes_seen = 0
        self.log_text = f"print_info: n_vocab = {MOCK_N_VOCAB}\n"
        self.log_offset = 0  # the whole log is new (real ServerInstance starts at 0)
        self.slot_save_path = self.output_dir / f"slot-save-{case_id}"
        self.slot_save_path.mkdir(parents=True, exist_ok=True)

    def server_command(self, extra_args):
        return [str(self.args.server_binary), "--model", str(self.args.model),
                "--host", self.args.host, "--port", str(self.args.port),
                "--device", self.args.cuda_device, "--ctx-size", str(self.args.ctx_size),
                "--batch-size", str(self.args.batch_size),
                "--ubatch-size", str(self.args.ubatch_size)] + list(extra_args)

    def start(self):
        self.started = True

    def stop(self):
        return None

    def read_log_delta(self):
        text = self.log_text
        delta = text[self.log_offset:]
        self.log_offset = len(text)
        return delta

    def log_evidence(self, delta):
        return [line.strip() for line in delta.splitlines()
                if HANDOFF_MARKER.lower() in line.lower()]

    def note_probe(self):
        self.probes_seen += 1
        if self.marker_after_first_probe and self.probes_seen == 1:
            self.log_text += f"debug: {HANDOFF_MARKER}\n"

    def inject_marker(self):
        if HANDOFF_MARKER not in self.log_text:
            self.log_text += f"debug: {HANDOFF_MARKER}\n"


class FakeHTTP:
    """Fake http_request: correct return shape {json, raw, status,
    elapsed_seconds}. Enforces the native n_probs key and cadence control."""

    def __init__(self, n_vocab: int, diverge_qx_cadence: bool = False,
                 fail_on_missing_n_probs: bool = True):
        self.n_vocab = n_vocab
        self.diverge_qx_cadence = diverge_qx_cadence
        self.fail_on_missing_n_probs = fail_on_missing_n_probs
        self.last_probe_payload: dict | None = None
        self.probe_payloads: list[dict] = []
        self.prompt_hashes: list[str] = []
        self.qx_probes = 0

    def _make_entries(self, seed: int) -> list[dict]:
        rng = np.random.default_rng(seed)
        logits = rng.standard_normal(self.n_vocab)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = np.argsort(-logits)
        entries = []
        for i in order:
            entries.append({"id": int(i), "token": str(i), "bytes": [0],
                            "logprob": float(math.log(float(p[i])))})
        return entries

    def _cadence(self, probe_n: int, qx: bool) -> tuple[int, int, int]:
        # Both routes must use the same saved checkpoint after warm output.
        if probe_n == 5000:
            cache, prompt = (3968, 1032)
        elif probe_n == 5500:
            cache, prompt = (5000, 500)
        else:
            cache, prompt = (5500, 500)
        if qx and self.diverge_qx_cadence:
            cache += 128  # a full group difference must NEVER be waived
        return cache, prompt, 1

    def __call__(self, base_url, method, path, payload, timeout, record_path,
                 telemetry_device, case_id, phase, request_body_record=None):
        if path == "/tokenize":
            return {"json": {"tokens": list(MOCK_IDS)}, "raw": "{}", "status": 200,
                    "elapsed_seconds": 0.0}
        if path != "/completion":
            return {"json": {}, "raw": "{}", "status": 404, "elapsed_seconds": 0.0}
        if phase.startswith("source"):
            if payload.get("n_predict") != 16:
                raise RuntimeError("source must generate a suffix to force checkpoint reuse")
            return {"json": {"timings": {"cache_n": 0, "prompt_n": payload["prompt"].__len__(),
                                         "predicted_n": 16}},
                    "raw": "{}", "status": 200, "elapsed_seconds": 0.0}
        # probe
        self.probe_payloads.append(payload)
        if self.fail_on_missing_n_probs and "n_probs" not in payload:
            raise RuntimeError("driver probe payload missing native n_probs key")
        if "top_logprobs" in payload or payload.get("logprobs") is not None:
            raise RuntimeError("driver probe payload used the OAI-only key")
        if payload.get("n_predict") != 1:
            raise RuntimeError("driver probe must request n_predict 1")
        if not payload.get("cache_prompt"):
            raise RuntimeError("driver probe must set cache_prompt")
        if request_body_record:
            self.prompt_hashes.append(request_body_record.get("prompt_sha256", ""))
        probe_n = len(payload["prompt"])
        qx = "qx-" in (case_id or "")
        if qx:
            self.qx_probes += 1
            if self.qx_probes == 1:
                server = MOCK_CURRENT_SERVER["instance"]
                if server is not None:
                    server.inject_marker()
        cache, prompt, predicted = self._cadence(probe_n, qx)
        body = {
            "timings": {"cache_n": cache, "prompt_n": prompt, "predicted_n": predicted},
            "completion_probabilities": [{"top_logprobs": self._make_entries(seed=probe_n)}],
        }
        return {"json": body, "raw": json.dumps(body), "status": 200, "elapsed_seconds": 0.0}


def run_mock_case(name: str, output_dir: Path, fake: FakeHTTP,
                  server_extra_checks: object | None = None) -> int:
    """Run q.main() with patched sys.argv, ServerInstance and http_request."""
    old_argv = sys.argv
    old_server = q.ServerInstance
    old_http = q.http_request
    old_frozen = q.create_frozen_prompts

    captured = {"extra_args": None, "case_ids": []}

    class RecordingFake(FakeServerInstance):
        def __init__(self, args, out_dir, case_id, extra_args):
            super().__init__(args, out_dir, case_id, extra_args)
            captured["extra_args"] = list(extra_args)
            captured["case_ids"].append(case_id)
            MOCK_CURRENT_SERVER["instance"] = self

    def fake_frozen(base_url, args, record_path, out_dir, case_id, existing=None):
        # mirrors the real helper's side effect: the baseline run persists
        # prompts.json, which the driver reloads for the qx cases
        frozen = {"prefix_token_ids": list(MOCK_IDS),
                  "prefix_bytes_sha256": "mock",
                  "suffix_token_ids": [],
                  "suffix_bytes_sha256": "mock"}
        if existing is not None:
            if existing["prefix_token_ids"] != frozen["prefix_token_ids"]:
                raise RuntimeError("mock tokenizer identity changed")
            return existing
        q.write_json(Path(out_dir) / "prompts.json", frozen)
        return frozen

    fixture_model = output_dir.parent / "mock-model.gguf"
    fixture_binary = output_dir.parent / "mock-server"
    fixture_model.write_bytes(b"mock model fixture; never loaded by GGML")
    fixture_binary.write_bytes(b"mock executable fixture; never launched")
    sys.argv = ["test-kv-handoff-quality",
                "--model", str(fixture_model),
                "--server-binary", str(fixture_binary),
                "--output-dir", str(output_dir),
                "--remote-kv-types", "q4_0",
                "--port", "18131"]
    q.ServerInstance = RecordingFake
    q.http_request = fake
    q.create_frozen_prompts = fake_frozen
    try:
        rc = q.main()
        if server_extra_checks is not None:
            server_extra_checks(captured)
        return rc
    finally:
        sys.argv = old_argv
        q.ServerInstance = old_server
        q.http_request = old_http
        q.create_frozen_prompts = old_frozen


def check_extra_args(captured):
    extra = captured["extra_args"] or []
    # qx flags truth (stage5-B): explicit layer 1, gate 4609, no prefill flag
    check("mock_extra_layers", "--remote-attn-layers" in extra and
          extra[extra.index("--remote-attn-layers") + 1] == "1", str(extra))
    check("mock_extra_gate", "--remote-attn-min-context" in extra and
          extra[extra.index("--remote-attn-min-context") + 1] == "4609", str(extra))
    check("mock_extra_ctx_mtp", "--ctx-size-mtp" in extra and
          extra[extra.index("--ctx-size-mtp") + 1] == "4608", str(extra))
    check("mock_extra_mtp_max", "--mtp-max-tokens" in extra and
          extra[extra.index("--mtp-max-tokens") + 1] == "4608", str(extra))
    check("mock_no_prefill", "--remote-attn-prefill" not in extra, str(extra))
    check("mock_no_ctx_size_m_alias", "--ctx-size-m" not in extra, str(extra))


def test_end_to_end_happy():
    with tempfile.TemporaryDirectory(prefix="hq-e2e-") as tmp:
        out = Path(tmp) / "out"
        fake = FakeHTTP(MOCK_N_VOCAB)
        rc = run_mock_case("happy", out, fake, check_extra_args)
        check("e2e_happy_rc", rc == 0, f"rc={rc}")
        rows = [json.loads(l) for l in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        results = rows[-1]
        check("e2e_happy_last_ok", results.get("status") == "OK", str(results))
        qx_rows = [r for r in rows if r["case_id"] == "qx-q4_0"]
        first_qx = qx_rows[0]
        check("e2e_marker_seen", first_qx.get("handoff_marker_seen") is True, str(first_qx))
        check("e2e_metrics_present", "metrics" in first_qx and first_qx["metrics"]["greedy_agree"] is True,
              str(first_qx.get("metrics")))
        cache = first_qx["cadence"][0]
        check("e2e_first_cache_range", 128 < cache <= 4096, str(cache))
        check("e2e_n_probs_key", all("n_probs" in p for p in fake.probe_payloads))
        check("e2e_no_oai_key", all("top_logprobs" not in p for p in fake.probe_payloads))
        check("e2e_real_prompt_hash",
              all(h and len(h) == 64 and h != "frozen-corpus" for h in fake.prompt_hashes),
              str(fake.prompt_hashes[:2]))
        check("e2e_artifacts_exist", len(list((out / "distributions").glob("*.npz"))) >= 6)
        check("e2e_frozen_len", len(MOCK_IDS) >= 6001, str(len(MOCK_IDS)))
        check("e2e_runjson", (out / "run.json").exists() and
              "server_binary_sha256" in json.loads((out / "run.json").read_text()))


def test_end_to_end_missing_marker():
    with tempfile.TemporaryDirectory(prefix="hq-marker-") as tmp:
        out = Path(tmp) / "out"
        fake = FakeHTTP(MOCK_N_VOCAB)
        # the stub injects the marker; disable it for this scenario
        old_inject = FakeServerInstance.inject_marker
        FakeServerInstance.inject_marker = lambda self: None
        try:
            rc = run_mock_case("missing-marker", out, fake)
        finally:
            FakeServerInstance.inject_marker = old_inject
        check("e2e_marker_rc", rc != 0, f"rc={rc}")
        rows = [json.loads(l) for l in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        check("e2e_marker_status", any(r.get("status") == "MISSING-MARKER" for r in rows), str(rows))


def test_end_to_end_divergent_cadence():
    with tempfile.TemporaryDirectory(prefix="hq-diverge-") as tmp:
        out = Path(tmp) / "out"
        fake = FakeHTTP(MOCK_N_VOCAB, diverge_qx_cadence=True)
        rc = run_mock_case("divergent", out, fake)
        check("e2e_diverge_rc", rc != 0, f"rc={rc}")
        rows = [json.loads(l) for l in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        check("e2e_diverge_status", any(r.get("status") == "CADENCE-DIVERGENT" for r in rows), str(rows))


def test_end_to_end_nonempty_output_dir_rejected():
    with tempfile.TemporaryDirectory(prefix="hq-nonempty-") as tmp:
        out = Path(tmp) / "out"
        out.mkdir()
        (out / "stale.txt").write_text("stale", encoding="utf-8")
        old_argv = sys.argv
        sys.argv = ["test-kv-handoff-quality", "--self-check", "x"]  # placeholder
        sys.argv = ["test-kv-handoff-quality", "--output-dir", str(out)]
        try:
            try:
                q.main()
                check("e2e_nonempty_rejected", False, "nonempty output dir accepted")
            except SystemExit as exc:
                check("e2e_nonempty_rejected", exc.code != 0, f"code={exc.code}")
        finally:
            sys.argv = old_argv


def main() -> int:
    test_parse_sentinel()
    test_metrics_known()
    test_kl_inf_explicit()
    test_cadence_gate()
    test_n_vocab_log()
    for name, fn in selfcheck_cases():
        try:
            fn()
            print(f"HARNESS {name}: PASS")
        except Exception as exc:
            FAILURES.append(f"selfcheck {name}: {exc}")
            print(f"HARNESS {name}: FAIL {exc}")
    test_end_to_end_happy()
    test_end_to_end_missing_marker()
    test_end_to_end_divergent_cadence()
    test_end_to_end_nonempty_output_dir_rejected()
    if FAILURES:
        for f in FAILURES:
            print(f"HARNESS FAIL: {f}")
        print(f"HARNESS: {len(FAILURES)} failures")
        return 1
    print("HARNESS: PASS (pure + end-to-end mock)")
    return 0


if __name__ == "__main__":
    sys.exit(main())