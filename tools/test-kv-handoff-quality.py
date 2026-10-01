#!/usr/bin/env python3
"""Full-vocab conditional-distribution QUALITY driver for the KVarN->Qx
handoff acceptance (conversion quality, not throughput).

Compares the REAL KVarN->Qx handoff against a pure-KVarN control over the
SAME frozen input and cadence, using the ENTIRE vocabulary conditional
distribution at each probe (not a truncated top-k KL):

  baseline  : adaptive MTP profile, pure KVarN, native prefix restore
  qx case   : identical profile + KVarN->Qx handoff with the per-layer remote
              format Q4_0/Q5_0/Q6_0/Q8_0

Server argv follows the PROVEN stage5-B-large-q4 case (run.json/cases.jsonl
under /home/hjotha/beellama-mixed-kv-20261001-130910/stage5-B-large-q4):
  --ctx-size 8192 --ctx-size-mtp 4608 --mtp-max-tokens 4608
  --cache-type-k-m/v-m kvarn4 --cache-type-k-l/v-l kvarn4
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-max-l 2 (M2)
  --cache-ram 2048 --ctx-checkpoints 1
  qx adds: --remote-attn vulkan:0 --remote-attn-layers 1
           --remote-attn-min-context 4609
           --remote-attn-cache-type-k/v <qx>
  NO --remote-attn-prefill flag (default 'remote'; 'migrate' is explicitly
  rejected by the mixed contract). The gate 4609 is strictly above the local
  MTP tier 4608 and <= LONG 8192; the 4096 source warm-up happens below the
  gate and is pure.

Protocol per case (each starts its own server sequentially):
  source    : warm at 4096 tokens (checkpoint established, gate armed)
  probe 5000/5500/6000 : n_predict=1, n_probs=n_vocab (native request key),
              temperature 0, top_k 0, top_p 1, min_p 0, no penalties
              (full-vocab retention). The cache cadence
              (timings.cache_n/prompt_n/predicted_n) is recorded and REQUIRED
              to equal the baseline BEFORE metrics; the first probe must show
              marker + cache_n in (128, 4096]. A divergent cadence fails that
              probe honestly.

The distribution source is the server's PRE-SAMPLE logits path:
need_pre_sample_logits = n_probs > 0 && !post_sampling_probs
(server-context.cpp:6151), so get_token_probabilities softmaxes the RAW
logits (independent of temperature; backend sampling disabled on that path).
Zero probability is serialized as logprob -3.4028235e38
(completion_token_output::logarithm(0.0f)); -inf is NOT a valid sentinel and
fails validation. KL(P_pure || P_mixed) is reported as null + kl_inf=true
(explicit) when p_pure>0 and p_mixed==0.

Every probe's raw HTTP body is preserved gzipped (the shared requests.jsonl
already carries the parsed JSON, so no duplicate plain raw file), and the
probability vectors are saved gzipped (.npz). Nothing is printed to the
console at all-vocab scale.

Usage (GPU run, gated by the parent; not executed this round):
  python3 tools/test-kv-handoff-quality.py --model <27B Swift MTP gguf>
CPU self-check (runs now):
  python3 tools/test-kv-handoff-quality.py --self-check
End-to-end CPU mock (runs now, no server):
  python3 tests/test-kv-handoff-quality-harness.py
"""

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "test_mixed_kv_transition", _TOOLS_DIR / "test-mixed-kv-transition.py")
assert _spec and _spec.loader
t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t)

# shared ServerInstance/http/corpus helpers from the mixed-KV transition tool
ServerInstance = t.ServerInstance
create_frozen_prompts = t.create_frozen_prompts
http_request = t.http_request
response_is_error = t.response_is_error
wait_ready = t.wait_ready
stop_server = t.stop_server
make_corpus = t.make_corpus
append_jsonl = t.append_jsonl
write_json = t.write_json
utc_now = t.utc_now
parse_names = t.parse_names

DEFAULT_MODEL = "/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
DEFAULT_SERVER = "/home/hjotha/beellama.cpp/build-optimized/bin/llama-server"
HANDOFF_MARKER = "mixed KV handoff: completed"

# server zero-policy sentinel: completion_token_output::logarithm(0.0f) ==
# std::numeric_limits<float>::lowest()
LOGPROB_ZERO = -3.4028235e38

N_VOCAB_LOG_RE = re.compile(r"\bn_vocab\s*=\s*(\d+)")

# ---------------------------------------------------------------------------
# pure validation / metrics (importable by tests/test-kv-handoff-quality-harness.py)
# ---------------------------------------------------------------------------


def parse_top_logprobs(entries: list[dict]) -> dict[int, float]:
    """Map id -> probability. Only the exact -FLT_MAX sentinel maps to 0.0;
    any other non-finite logprob (e.g. -inf) raises instead of being treated
    as zero."""
    out: dict[int, float] = {}
    for e in entries:
        tid = e.get("id")
        lp = e.get("logprob")
        if not isinstance(lp, (int, float)) or not math.isfinite(lp):
            raise ValueError(f"non-finite logprob {lp!r} for id {tid!r} (not the zero policy)")
        p = 0.0 if lp <= LOGPROB_ZERO / 2.0 else math.exp(lp)
        out[int(tid)] = float(p)
    return out


def validate_distribution(entries: list[dict], n_vocab: int, mass_tol: float = 1e-3) -> list[str]:
    """Full-vocab structural validation. Returns failure strings (empty = ok).
    -inf is NOT the zero policy (the sentinel is exactly -FLT_MAX)."""
    failures: list[str] = []
    if len(entries) != n_vocab:
        failures.append(f"top_logprobs length {len(entries)} != n_vocab {n_vocab}")
    ids = [e.get("id") for e in entries]
    if any(not isinstance(i, int) or isinstance(i, bool) for i in ids):
        failures.append("non-integer token id present")
    uniq = set(ids)
    if len(uniq) != len(ids):
        failures.append(f"duplicate token ids ({len(ids) - len(uniq)} dup)")
    if uniq != set(range(n_vocab)):
        failures.append("ids do not cover exactly 0..n_vocab-1")
    total = 0.0
    for e in entries:
        lp = e.get("logprob")
        if not isinstance(lp, (int, float)) or not math.isfinite(lp):
            failures.append("non-finite logprob (inf/-inf is not the zero policy)")
            continue
        if lp > 0:
            failures.append("positive logprob is not a valid probability")
            continue
        if lp <= LOGPROB_ZERO / 2.0:
            continue  # documented zero policy (exact -FLT_MAX sentinel)
        total += math.exp(lp)
    if abs(total - 1.0) > mass_tol:
        failures.append(f"probability mass {total:.6f} outside FP32 tolerance {mass_tol}")
    return failures


def distribution_from_entries(entries: list[dict], n_vocab: int) -> np.ndarray:
    """Build the dense vector AFTER validate_distribution passed."""
    probs = parse_top_logprobs(entries)
    dist = np.zeros(n_vocab, dtype=np.float64)
    for tid, p in probs.items():
        dist[tid] = p
    return dist


def metrics(p_pure: np.ndarray, p_mixed: np.ndarray, ref_id: int | None = None) -> dict:
    """KL(P_pure || P_mixed) with explicit inf for p_pure>0,p_mixed==0; TV;
    max prob delta; greedy agreement; reference-token surprisal under
    P_mixed. inf is reported as None + a boolean flag for JSON-safety."""
    assert p_pure.shape == p_mixed.shape
    kl = 0.0
    kl_inf = False
    for a, b in zip(p_pure.tolist(), p_mixed.tolist()):
        if a > 0.0:
            if b == 0.0:
                kl_inf = True
            else:
                kl += a * math.log(a / b)
    tv = float(np.sum(np.abs(p_pure - p_mixed))) / 2.0
    max_delta = float(np.max(np.abs(p_pure - p_mixed)))
    greedy_pure = int(np.argmax(p_pure))
    greedy_mixed = int(np.argmax(p_mixed))
    surprisal = None
    surprisal_inf = False
    if ref_id is not None:
        q = float(p_mixed[ref_id])
        if q == 0.0:
            surprisal_inf = True
        else:
            surprisal = -math.log(q)
    return {
        "kl": None if kl_inf else kl,
        "kl_inf": kl_inf,
        "tv": tv,
        "max_prob_delta": max_delta,
        "greedy_pure": greedy_pure,
        "greedy_mixed": greedy_mixed,
        "greedy_agree": greedy_pure == greedy_mixed,
        "ref_surprisal": None if surprisal_inf else surprisal,
        "ref_surprisal_inf": surprisal_inf,
        "ref_id": ref_id,
    }


def extract_cache_cadence(body: dict) -> tuple[int, int, int]:
    timings = body.get("timings") or {}
    cache_n = timings.get("cache_n")
    prompt_n = timings.get("prompt_n")
    predicted_n = timings.get("predicted_n")
    if not all(isinstance(v, int) for v in (cache_n, prompt_n, predicted_n)):
        raise ValueError(f"timings missing integer cache_n/prompt_n/predicted_n: {timings}")
    return cache_n, prompt_n, predicted_n


def read_n_vocab_from_log(log_text: str) -> int | None:
    for match in N_VOCAB_LOG_RE.finditer(log_text):
        return int(match.group(1))
    return None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def available_mem_bytes() -> int:
    """Linux MemAvailable in bytes; 0 when unavailable (guard disabled)."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key == "MemAvailable":
                return int(value.strip().split()[0]) * 1024
    except OSError:
        return 0
    return 0


# ---------------------------------------------------------------------------
# self-check cases (CPU, no server): synthetic distributions
# ---------------------------------------------------------------------------


def selfcheck_cases():
    """Yield (name, fn) pairs exercising validation and metrics."""
    n_vocab = 1024

    def make_dist(seed: int, zero_fraction: float = 0.0) -> np.ndarray:
        r = np.random.default_rng(seed)
        logits = r.standard_normal(n_vocab)
        if zero_fraction > 0.0:
            logits[: int(n_vocab * zero_fraction)] = -1e9
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return p

    def run_finite_probs():
        p_pure = make_dist(1)
        p_mixed = make_dist(2)
        entries = [
            {"id": int(i), "token": str(i), "bytes": [0], "logprob": float(math.log(p_mixed[i]))}
            for i in range(n_vocab)
        ]
        fails = validate_distribution(entries, n_vocab)
        assert not fails, f"finite case failed validation: {fails}"
        dist = distribution_from_entries(entries, n_vocab)
        assert np.allclose(dist, p_mixed, atol=1e-12)
        m = metrics(p_pure, dist, ref_id=17)
        assert m["kl"] is not None and m["kl"] >= 0.0 and not m["kl_inf"]
        assert 0.0 <= m["tv"] <= 1.0
        assert isinstance(m["greedy_agree"], bool)
        return {"n_vocab": n_vocab, "kl": m["kl"], "tv": m["tv"]}

    def run_missing_ids():
        entries = [
            {"id": int(i), "token": str(i), "bytes": [0], "logprob": -1.0}
            for i in range(n_vocab - 3)
        ]
        fails = validate_distribution(entries, n_vocab)
        assert fails and any("length" in f for f in fails)
        assert any("cover exactly" in f for f in fails)
        return {"missing_detected": True}

    def run_duplicate_ids():
        entries = [
            {"id": int(i), "token": str(i), "bytes": [0], "logprob": -1.0}
            for i in range(n_vocab)
        ]
        entries[10] = dict(entries[11])
        fails = validate_distribution(entries, n_vocab)
        assert fails and any("duplicate" in f for f in fails)
        return {"duplicate_detected": True}

    def run_negative_id():
        entries = [
            {"id": int(i), "token": str(i), "bytes": [0], "logprob": -1.0}
            for i in range(n_vocab)
        ]
        entries[0] = {"id": -5, "token": "neg", "bytes": [0], "logprob": -1.0}
        fails = validate_distribution(entries, n_vocab)
        assert fails and any("cover exactly" in f for f in fails)
        return {"negative_id_detected": True}

    def run_zero_probs():
        p_pure = make_dist(3)
        p_mixed = make_dist(4, zero_fraction=0.05)
        entries = [
            {"id": int(i), "token": str(i), "bytes": [0],
             "logprob": float(LOGPROB_ZERO) if p_mixed[i] == 0.0 else float(math.log(p_mixed[i]))}
            for i in range(n_vocab)
        ]
        fails = validate_distribution(entries, n_vocab)
        assert not fails, f"zero-policy case failed validation: {fails}"
        nonzero = [i for i in range(n_vocab) if p_pure[i] > 0.0 and p_mixed[i] == 0.0]
        m = metrics(p_pure, p_mixed, ref_id=nonzero[0])
        assert m["kl_inf"] and m["kl"] is None, "KL must be explicit null+flag when P>0 and Q==0"
        assert m["ref_surprisal_inf"]
        return {"zero_policy_ok": True, "kl_inf_expected": True}

    def run_minus_inf_rejected():
        entries = [
            {"id": int(i), "token": str(i), "bytes": [0], "logprob": -1.0}
            for i in range(n_vocab)
        ]
        entries[3] = {"id": 3, "token": "x", "bytes": [0], "logprob": float("-inf")}
        fails = validate_distribution(entries, n_vocab)
        assert fails and any("non-finite" in f for f in fails), "minus-inf must not be a zero"
        return {"minus_inf_rejected": True}

    def run_mass_floor():
        entries = [
            {"id": int(i), "token": str(i), "bytes": [0],
             "logprob": float(LOGPROB_ZERO) if i < 10 else math.log(1.0 / (n_vocab - 10))}
            for i in range(n_vocab)
        ]
        probs = parse_top_logprobs(entries)
        assert all(probs[i] == 0.0 for i in range(10))
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        return {"mass_floor_ok": True}

    def run_divergent_cadence():
        baseline_cache = (4096, 904, 1)
        qx_cache = (3968, 1032, 1)
        ok = baseline_cache == qx_cache
        assert not ok, "divergent cadence must NOT compare"
        return {"cadence_equal": ok, "honest_skip": not ok}

    return [
        ("finite_probs", run_finite_probs),
        ("missing_ids", run_missing_ids),
        ("duplicate_ids", run_duplicate_ids),
        ("negative_id", run_negative_id),
        ("zero_probs", run_zero_probs),
        ("minus_inf_rejected", run_minus_inf_rejected),
        ("mass_floor", run_mass_floor),
        ("divergent_cadence", run_divergent_cadence),
    ]


def run_selfcheck() -> int:
    failed = 0
    total = 0
    for name, fn in selfcheck_cases():
        total += 1
        try:
            fn()
            print(f"SELFCHECK {name}: PASS")
        except Exception as exc:
            failed += 1
            print(f"SELFCHECK {name}: FAIL {exc}")
    print(f"SELFCHECK done: {total - failed}/{total} passed")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# GPU driver (written now; executed only after the parent grants)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--server-binary", type=Path, default=Path(DEFAULT_SERVER))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18131)
    parser.add_argument("--cuda-device", default="CUDA0")
    parser.add_argument("--remote-kv-types", default="q4_0,q5_0,q6_0,q8_0")
    parser.add_argument("--ctx-size", type=int, default=8192, help="LONG tier (--ctx-size)")
    parser.add_argument("--local-ctx", type=int, default=4608, help="MTP tier (--ctx-size-mtp)")
    parser.add_argument("--mtp-max-tokens", type=int, default=4608, help="adaptive gate == local")
    parser.add_argument("--gate-ctx", type=int, default=4609, help="--remote-attn-min-context")
    parser.add_argument("--cache-ram", type=int, default=2048, help="--cache-ram MiB")
    parser.add_argument("--ctx-checkpoints", type=int, default=1)
    parser.add_argument("--prompts", default="5000,5500,6000")
    parser.add_argument("--source-tokens", type=int, default=4096)
    parser.add_argument("--spec-draft-n-max", type=int, default=2, help="MTP tier draft N (M2)")
    parser.add_argument("--spec-draft-n-max-l", type=int, default=2, help="LONG tier draft N (M2)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--shutdown-timeout", type=float, default=60)
    parser.add_argument("--apu-tdp", type=int, default=20)
    parser.add_argument("--amd-sclk-prefill", type=int, default=2700)
    parser.add_argument("--amd-sclk-decode", type=int, default=2700)
    parser.add_argument("--telemetry-device", default="none")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/home/hjotha/beellama-mixed-kv-20261001-130910/handoff-quality"))
    parser.add_argument("--min-available-mem", type=int, default=4096,
                        help="abort a case when host MemAvailable is below this many MiB "
                             "(avoid OOM on a low-RAM host)")
    parser.add_argument("--self-check", action="store_true",
                        help="CPU self-check only; no server, no GPU")
    return parser


def validate_config(args: argparse.Namespace) -> None:
    if not (args.local_ctx < args.gate_ctx <= args.ctx_size):
        raise SystemExit(f"--gate-ctx {args.gate_ctx} must satisfy local {args.local_ctx} < gate "
                         f"<= LONG {args.ctx_size}")
    if args.mtp_max_tokens != args.local_ctx:
        raise SystemExit("--mtp-max-tokens must equal --local-ctx (proven stage5-B cadence)")
    if args.source_tokens >= args.gate_ctx:
        raise SystemExit("--source-tokens must be below --gate-ctx (pure warm-up)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"--output-dir exists and is not empty: {args.output_dir}")


def base_case_args(args: argparse.Namespace, qx: str | None) -> list[str]:
    """stage5-B-large-q4 proven argv (truth), with M2 long draft 2. The
    ServerInstance base already adds --ctx-size/--batch-size/--ubatch-size/
    --cache-type-k/v kvarn4/--kv-tail-tokens 0/--fit off/powers/--device."""
    common = [
        "--spec-type", "draft-mtp",
        "--ctx-size-mtp", str(args.local_ctx),
        "--mtp-max-tokens", str(args.mtp_max_tokens),
        "--cache-type-k-m", "kvarn4", "--cache-type-v-m", "kvarn4",
        "--cache-type-k-l", "kvarn4", "--cache-type-v-l", "kvarn4",
        "--spec-draft-type-k-m", "kvarn4", "--spec-draft-type-v-m", "kvarn4",
        "--spec-draft-type-k-l", "kvarn4", "--spec-draft-type-v-l", "kvarn4",
        "--spec-draft-n-max", str(args.spec_draft_n_max),
        "--spec-draft-n-max-l", str(args.spec_draft_n_max_l),
        "--cache-ram", str(args.cache_ram),
        "--ctx-checkpoints", str(args.ctx_checkpoints),
    ]
    if qx is None:
        return common  # baseline: pure KVarN, no remote allocation
    # Mixed contract: NO --remote-attn-prefill (default 'remote'; 'migrate'
    # is explicitly rejected), explicit single remote layer (auto at 8192
    # would choose 0), gate strictly above the local MTP tier.
    return common + [
        "--remote-attn", "vulkan:0",
        "--remote-attn-layers", "1",
        "--remote-attn-min-context", str(args.gate_ctx),
        "--remote-attn-cache-type-k", qx,
        "--remote-attn-cache-type-v", qx,
    ]


def record_env(args: argparse.Namespace, output_dir: Path) -> dict:
    """Exact argv/model/binary/current library hashes for the run."""
    env = {
        "argv": sys.argv,
        "created_utc": utc_now(),
    }
    def file_hash(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    for label, path in (("model", args.model), ("server_binary", args.server_binary)):
        try:
            env[f"{label}_sha256"] = file_hash(path)
            env[f"{label}_size"] = path.stat().st_size
        except OSError as exc:
            env[f"{label}_error"] = str(exc)
    for lib in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cuda.so", "libggml-vulkan.so", "libllama-server-impl.so"):
        p = args.server_binary.resolve().parent / lib
        try:
            env[f"lib_{lib}_path"] = str(p.resolve())
            env[f"lib_{lib}_sha256"] = file_hash(p)
        except OSError as exc:
            env[f"lib_{lib}_error"] = str(exc)
    try:
        out = subprocess.run(["git", "-C", "/home/hjotha/beellama.cpp", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=False)
        env["git_head"] = out.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        env["git_head_error"] = str(exc)
    write_json(output_dir / "run.json", env)
    return env


def run_probe(args: argparse.Namespace, base_url: str, records_path: Path,
              case_id: str, prompt_ids: list[int], n_vocab: int,
              probe_n: int) -> dict:
    """One full-vocab conditional probe. Native request key is n_probs."""
    payload = {
        "prompt": prompt_ids,
        "n_predict": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "return_tokens": True,
        "temperature": 0,
        "top_k": 0,               # retain the FULL vocab in the sampler chain
        "top_p": 1.0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": args.seed,
        "ignore_eos": True,
        "post_sampling_probs": False,
        "n_probs": n_vocab,       # native key (schema: n_probs, alias logprobs)
    }
    result = http_request(base_url, "POST", "/completion", payload,
                          args.request_timeout, records_path, args.telemetry_device,
                          case_id, f"probe{probe_n}",
                          request_body_record={
                              "prompt_sha256": sha256_hex(json.dumps(prompt_ids).encode()),
                              "probe_tokens": len(prompt_ids),
                          })
    return result


def save_artifacts(out_dir: Path, case_id: str, probe_n: int, raw_text: str,
                   dist: np.ndarray, ref_id: int | None) -> dict:
    """Raw response preserved gzipped only (the shared requests.jsonl already
    carries the parsed JSON); probability vectors gzipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gz_path = out_dir / f"{case_id}-probe{probe_n}-raw.json.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        f.write(raw_text)
    npz_path = out_dir / f"{case_id}-probe{probe_n}-probs.npz"
    # np.savez needs a seekable sink (zipfile); gzip files are not seekable,
    # so serialize to memory first and gzip the bytes
    buf = io.BytesIO()
    np.savez(buf, probs=dist.astype(np.float32), ref_id=np.int32(ref_id if ref_id is not None else -1))
    with gzip.open(npz_path, "wb") as f:
        f.write(buf.getvalue())
    return {"raw_gz": str(gz_path), "npz": str(npz_path)}


def run_quality_case(args: argparse.Namespace, output_dir: Path, records_path: Path,
                     results_path: Path, cases_path: Path, case_id: str,
                     qx: str | None, frozen: dict, baseline_dists: dict[int, np.ndarray],
                     baseline_cadence: dict[int, tuple[int, int, int]],
                     baseline_n_vocab: int) -> dict:
    """One server case. Baseline (qx=None) records distributions; qx cases
    gate on cadence equality with the baseline, require the handoff marker on
    the first probe, and compute metrics."""
    case_record = {"case_id": case_id, "qx": qx, "state": "starting",
                   "started_utc": utc_now(), "probes": []}
    append_jsonl(cases_path, case_record)
    instance = ServerInstance(args, output_dir, case_id, base_case_args(args, qx))
    primary_error: BaseException | None = None
    try:
        # host memory guard: abort the OWN process instead of OOMing others
        avail = available_mem_bytes()
        if avail and avail < args.min_available_mem * (1 << 20):
            raise RuntimeError(f"host memory too low for a new server case: "
                               f"MemAvailable {avail >> 20} MiB < "
                               f"{args.min_available_mem} MiB; aborting")
        instance.start()
        base_url = f"http://{args.host}:{args.port}"
        log_text = instance.read_log_delta()
        n_vocab = read_n_vocab_from_log(log_text)
        if n_vocab is None:
            log_text = instance.log_path.read_text(encoding="utf-8", errors="replace")
            n_vocab = read_n_vocab_from_log(log_text)
        if n_vocab is None:
            raise RuntimeError("server log did not expose n_vocab")
        if baseline_n_vocab and n_vocab != baseline_n_vocab:
            raise RuntimeError(f"n_vocab mismatch: baseline {baseline_n_vocab}, case {n_vocab}")
        case_record["n_vocab"] = n_vocab

        frozen = create_frozen_prompts(base_url, args, records_path, output_dir,
                                       case_id, frozen)
        prefix = frozen["prefix_token_ids"]
        if len(prefix) < max(args.prompts) + 1:
            raise RuntimeError("frozen prompts must cover the next-token reference "
                               "(len >= max probe + 1)")

        # Generate a suffix in the pure source so the requested natural
        # continuation diverges from the saved live sequence. Both the native
        # control and mixed handoff must then restore the SAME checkpoint,
        # rather than native FULL reuse versus mixed PARTIAL reuse.
        warm = http_request(base_url, "POST", "/completion",
                            {"prompt": prefix[: args.source_tokens], "n_predict": 16,
                             "id_slot": 0, "cache_prompt": True, "temperature": 0,
                             "seed": args.seed, "ignore_eos": True,
                             "logit_bias": [[0 if prefix[args.source_tokens] != 0 else 1, 1000.0]]},
                            args.request_timeout, records_path, args.telemetry_device,
                            case_id, f"source{args.source_tokens}",
                            request_body_record={"probe_tokens": args.source_tokens,
                                                 "prompt_sha256": sha256_hex(json.dumps(prefix[:args.source_tokens]).encode()),
                                                 "n_predict": 16,
                                                 "logit_bias": [[0 if prefix[args.source_tokens] != 0 else 1, 1000.0]]})
        warm_error, warm_msg = response_is_error(warm)
        if warm_error:
            raise RuntimeError(f"source warm-up failed: {warm_msg}")
        warm_cadence = extract_cache_cadence(warm["json"])
        if warm_cadence != (0, args.source_tokens, 16):
            raise RuntimeError(f"unexpected source warm-up cadence: {warm_cadence}")

        case_probes: dict[int, dict] = {}
        for probe_i, probe_n in enumerate(args.prompts):
            result = run_probe(args, base_url, records_path, case_id,
                               prefix[: probe_n], n_vocab, probe_n)
            probe_error, probe_msg = response_is_error(result)
            if probe_error:
                raise RuntimeError(f"probe {probe_n} failed: {probe_msg}")
            body = result["json"]
            cadence = extract_cache_cadence(body)
            entries = body.get("completion_probabilities", [{}])[0].get("top_logprobs", [])
            validation = validate_distribution(entries, n_vocab)
            ref_id = int(prefix[probe_n])
            row = {"case_id": case_id, "probe": probe_n, "n_vocab": n_vocab,
                   "cadence": cadence, "validation_failures": validation}

            if qx is None:
                # baseline: first probe must reuse the warm source; an invalid
                # distribution must FAIL the case, never be stored for
                # comparison
                if probe_i == 0 and cadence[0] <= 0:
                    raise RuntimeError(f"baseline first probe did not reuse the cache: {cadence}")
                if validation:
                    raise RuntimeError(f"baseline probe {probe_n} distribution invalid: {validation}")
                dist = distribution_from_entries(entries, n_vocab)
                baseline_dists[probe_n] = dist
                baseline_cadence[probe_n] = cadence
                artifacts = save_artifacts(output_dir / "distributions", case_id,
                                           probe_n, result["raw"], dist, ref_id)
                row.update({"status": "BASELINE", **artifacts})
            else:
                # The COMPLETE replay history must have the same cadence.
                # Once any probe differs, later recurrent states remain
                # confounded even if their per-call counts match again.
                bl = baseline_cadence.get(probe_n)
                if cadence != bl:
                    row.update({"status": "CADENCE-DIVERGENT", "baseline_cadence": bl})
                    append_jsonl(results_path, row)
                    raise RuntimeError("cadence diverged; stopping this history before any further quality comparison")
                if probe_i == 0:
                    if not (128 < cadence[0] <= args.source_tokens):
                        raise RuntimeError("first probe did not reuse a sealed checkpoint prefix")
                    marker_lines = instance.read_log_delta().splitlines()
                    seen = [ln for ln in marker_lines if HANDOFF_MARKER.lower() in ln.lower()]
                    if not seen:
                        row.update({"status": "MISSING-MARKER"})
                        append_jsonl(results_path, row)
                        raise RuntimeError("first probe lacks a completed handoff marker")
                    row["handoff_marker_seen"] = True
                if validation:
                    row.update({"status": "INVALID-DIST"})
                else:
                    dist = distribution_from_entries(entries, n_vocab)
                    m = metrics(baseline_dists[probe_n], dist, ref_id=ref_id)
                    artifacts = save_artifacts(output_dir / "distributions", case_id,
                                               probe_n, result["raw"], dist, ref_id)
                    row.update({"status": "OK", "metrics": m, **artifacts})
            append_jsonl(results_path, row)
            case_probes[probe_n] = row
        case_record["state"] = "complete" if all(
                row.get("status") in ("BASELINE", "OK") for row in case_probes.values()) else "failed"
        case_record["probes"] = list(case_probes.values())
        return case_record
    except BaseException as exc:
        primary_error = exc
        case_record["state"] = "failed"
        case_record["error"] = str(exc)
        raise
    finally:
        try:
            instance.stop()
        except BaseException as stop_exc:
            if primary_error is None:
                raise
            case_record["stop_error"] = str(stop_exc)
        append_jsonl(cases_path, case_record)


def main() -> int:
    args = build_parser().parse_args()
    if args.self_check:
        return run_selfcheck()

    args.remote_kv_types = parse_names(args.remote_kv_types, "--remote-kv-types",
                                       {"q4_0", "q5_0", "q6_0", "q8_0"})
    args.prompts = [int(p) for p in args.prompts.split(",")]
    validate_config(args)

    # the shared frozen-prompt helper reads these from the namespace
    args.prompt_tokens = max(args.prompts)
    args.prompt_short = args.prompts[0]
    args.prompt_long = args.prompts[1] if len(args.prompts) > 1 else args.prompts[0]
    args.suffix_tokens = 1  # next-token reference needs prefix len >= probe + 1
    args.ctx_size = args.ctx_size  # LONG tier used by ServerInstance

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "requests.jsonl"
    results_path = output_dir / "results.jsonl"
    cases_path = output_dir / "cases.jsonl"
    write_json(output_dir / "config.json", {
        "argv": sys.argv,
        "config": {k: v for k, v in vars(args).items()},
    })
    record_env(args, output_dir)

    baseline_dists: dict[int, np.ndarray] = {}
    baseline_cadence: dict[int, tuple[int, int, int]] = {}
    baseline_n_vocab = 0
    frozen = None

    failed = 0
    try:
        baseline_record = run_quality_case(args, output_dir, records_path, results_path,
                                           cases_path, "baseline", None, frozen,
                                           baseline_dists, baseline_cadence, baseline_n_vocab)
        baseline_n_vocab = baseline_record.get("n_vocab", 0)
        frozen = json.loads((output_dir / "prompts.json").read_text(encoding="utf-8"))
    except BaseException as exc:
        failed = 1
        print(f"BASELINE FAILED: {exc}", flush=True)

    summary: dict = {"cases": []}
    if not failed and baseline_record:
        summary["cases"].append(baseline_record)
        for qx in args.remote_kv_types:
            try:
                record = run_quality_case(args, output_dir, records_path, results_path,
                                          cases_path, f"qx-{qx}", qx, frozen,
                                          baseline_dists, baseline_cadence, baseline_n_vocab)
                summary["cases"].append(record)
            except BaseException as exc:
                failed = 1
                print(f"CASE qx-{qx} FAILED: {exc}", flush=True)

    # a case that produced CADENCE-DIVERGENT / INVALID-DIST / MISSING-MARKER
    # rows is a failure even without an exception
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") not in ("BASELINE", "OK"):
                failed = 1
    write_json(output_dir / "summary.json", summary)
    print(f"handoff-quality done: cases={len(summary.get('cases', []))} failed={failed} "
          f"output={output_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())