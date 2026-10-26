#!/usr/bin/env python3
"""A/B/B/A decode-path campaign for the RKVA nq=1 critical path.

The host CUDA-graph headroom guard sampling cache is measured on the same
frozen local prefix (GGML_CUDA_GRAPH_GUARD_SAMPLE_MS: 0 = per-submission
cudaMemGetInfo; 10 = cached sample with a 10 ms age bound).

Everything is inert unless --execute is passed. Before launching anything the
runner refuses to continue when the Codex benchmark session is active, because
that session owns the RTX 4070, the Xbox and the 20 W host budget.

Outputs, next to the four pass directories:
  rkva-decode-host-ab.json   full report: plan, per-pass evidence, parity,
                             per-arm medians and the derived deltas
  decode-attribution.json    per-layer trace attribution of the traced passes

The script never edits the bench script; the arm environment is injected
through the module-level PINNED_ENV mapping the bench script already owns.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "tools/xbox/scripts/bench-rkva-range.py"
ANALYZER = ROOT / "tools/xbox/scripts/analyze-rkva-decode-trace.py"

GUARD_ENV = "GGML_CUDA_GRAPH_GUARD_SAMPLE_MS"
# Executables owned by the Codex benchmark session. Matching is on the process
# executable name (comm), never on free-form command text: agent shells carry
# these words in their prompts.
CODEX_COMMS = {"llama-server", "llama-bench", "llama-perplexity", "ninja", "ctest"}
CODEX_COMM_PREFIXES = ("test-rkva", "bench-rkva")
CODEX_SCRIPT_SUFFIXES = ("bench-rkva-range.py",)
# A live Codex CLI may launch a benchmark at any moment, so it also blocks a
# campaign unless the operator explicitly declares it idle.
CODEX_SESSION_COMMS = {"codex"}
# arm -> environment overrides injected into the bench script's PINNED_ENV
ARM_ENVIRONMENT = {
    "A": {GUARD_ENV: "0"},   # baseline: exact per-submission query
    "B": {GUARD_ENV: "10"},  # cached sample, 10 ms age bound (source default)
}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", required=True)
    parser.add_argument("--xbox-host", required=True, help="HOST:PORT of the installed RKVA listener")
    parser.add_argument("--output", required=True)
    parser.add_argument("--xbox-build-id", required=True,
                        help="installed MSIX/package identity; both arms must share it")
    parser.add_argument("--passes", default="A,B,B,A", help="arm order (default A,B,B,A)")
    parser.add_argument("--points", default="130560")
    parser.add_argument("--p", type=int, default=102400)
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--threads-batch", type=int)
    parser.add_argument("--prefix-snapshot", help="frozen local-prefix slot file (same for every arm)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trace", action="store_true", default=True,
                        help="collect the RKVA host trace for the decode attribution (default on)")
    parser.add_argument("--no-trace", dest="trace", action="store_false")
    parser.add_argument("--execute", action="store_true", help="actually run the campaign")
    parser.add_argument("--dry-run", action="store_true", help="preview only (default)")
    parser.add_argument("--allow-codex-idle", action="store_true",
                        help="start the campaign while only the Codex CLI (no benchmark process) is running")
    return parser.parse_args(argv)


def arm_plan(order):
    arms = [item.strip().upper() for item in order.split(",") if item.strip()]
    if not arms or any(arm not in ARM_ENVIRONMENT for arm in arms):
        raise ValueError("passes must be a comma-separated list of A/B arms")
    return arms


def codex_matches(process_rows, self_pids=()):
    """Return the process rows owned by the Codex benchmark session.

    process_rows items are (pid, ppid, comm, args). Command text is not
    trusted: only the executable name, a build-tool argv, or an explicit
    benchmark script path counts, and this runner plus its ancestors are
    excluded so a dry-run can never refuse itself.
    """
    blocked = {int(pid) for pid in self_pids}
    matches = []
    for row in process_rows:
        pid, ppid, comm, args = row
        if int(pid) in blocked:
            continue
        name = (comm or "").strip().rsplit("/", 1)[-1]
        argv = (args or "").split()
        hit = name in CODEX_SESSION_COMMS
        if not hit:
            if name in CODEX_COMMS:
                hit = True
            elif name.startswith(CODEX_COMM_PREFIXES):
                hit = True
            elif name in ("cmake", "cmake3"):
                hit = any(part in ("--build",) or part.startswith("--build=") for part in argv)
            elif name in ("python", "python3", "python3.11", "python3.12"):
                hit = any(part.endswith(CODEX_SCRIPT_SUFFIXES) for part in argv[1:])
        if hit:
            matches.append({"pid": int(pid), "ppid": int(ppid), "comm": name, "args": (args or "").strip(),
                            "session_only": name in CODEX_SESSION_COMMS})
    return matches


def read_process_table():
    try:
        completed = subprocess.run(["ps", "-eo", "pid=,ppid=,comm=,args="],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"cannot inspect the process table: {error}") from error
    rows = []
    for line in completed.stdout.splitlines():
        fields = line.split(None, 3)
        if len(fields) < 4:
            continue
        pid, ppid, comm, args = fields
        if not pid.isdigit() or not ppid.isdigit():
            continue
        rows.append((int(pid), int(ppid), comm, args))
    return rows


def self_pid_chain(rows, pid=None):
    """This process and its ancestors: a dry-run must not refuse itself."""
    parents = {row[0]: row[1] for row in rows}
    chain = set()
    current = os.getpid() if pid is None else int(pid)
    while current and current not in chain:
        chain.add(current)
        current = parents.get(current, 0)
    return chain


def load_bench():
    spec = importlib.util.spec_from_file_location("bench_rkva_range", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "PINNED_ENV"):
        raise RuntimeError("bench-rkva-range.py no longer exposes PINNED_ENV; arm injection needs review")
    return module


def pass_arguments(args, output):
    argv = [
        "--server-binary", args.server_binary,
        "--model", args.model,
        "--prompt-tokens", args.prompt_tokens,
        "--output", str(output),
        "--xbox-host", args.xbox_host,
        "--points", args.points,
        "--p", str(args.p),
        "--context", str(args.context),
        "--n-predict", str(args.n_predict),
        "--threads", str(args.threads),
        "--batch-size", str(args.batch_size),
        "--ubatch", str(args.ubatch),
        "--seed", str(args.seed),
        "--cache-ram", "4096",
        "--execute",
    ]
    if args.threads_batch is not None:
        argv += ["--threads-batch", str(args.threads_batch)]
    if args.prefix_snapshot:
        argv += ["--prefix-snapshot", args.prefix_snapshot]
    if args.trace:
        argv += ["--trace"]
    return argv


def tokens_from_response(path):
    """Generated token IDs of a /completion response (return_tokens=True)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"{path}: response has no generated token list")
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError(f"{path}: malformed token IDs")
    return tokens


def parity_report(passes):
    """Compare the generated token IDs of every pass against the first one."""
    reference_name = passes[0]["name"]
    reference = passes[0].get("tokens")
    report = {"reference": reference_name, "matched": True, "lengths": {}, "mismatches": []}
    if not reference:
        report.update(matched=False, error="reference pass produced no tokens")
        return report
    for entry in passes:
        tokens = entry.get("tokens")
        report["lengths"][entry["name"]] = len(tokens) if tokens else 0
        if not tokens:
            report["mismatches"].append({"pass": entry["name"], "error": "no generated tokens"})
            continue
        if len(tokens) != len(reference):
            report["mismatches"].append({"pass": entry["name"], "error": "length differs",
                                         "expected": len(reference), "actual": len(tokens)})
            continue
        for index, (expected, actual) in enumerate(zip(reference, tokens)):
            if expected != actual:
                report["mismatches"].append({"pass": entry["name"], "index": index,
                                             "expected": expected, "actual": actual})
                break
    report["matched"] = not report["mismatches"]
    return report


def parse_profile_line(lines):
    """Last `remote_attn gpu_profile {json}` line of a server log."""
    payload = None
    for line in lines:
        marker = "remote_attn gpu_profile "
        if marker not in line:
            continue
        try:
            payload = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError:
            continue
    return payload


def parse_guard_line(lines):
    """Activation line of the host headroom guard (destructor summary)."""
    found = None
    for line in lines:
        if "headroom guard submissions" not in line:
            continue
        fields = {"raw": line.strip()}
        for token in line.split():
            key, separator, value = token.partition("=")
            if separator and key in ("fastpath", "queried", "sample_age_ms") and value.isdigit():
                fields[key] = int(value)
        found = fields
    return found


def decode_tps(results):
    values = []
    for point in results.get("points", []):
        tps = point.get("authoritative_decode_tps")
        if isinstance(tps, (int, float)) and tps > 0:
            values.append(float(tps))
    return values


def summarize_arm(passes, arm):
    values = []
    for entry in passes:
        if entry["arm"] == arm:
            values.extend(entry.get("decode_tps") or [])
    if not values:
        return {"n": 0}
    return {"n": len(values), "p50_tps": statistics.median(values), "min_tps": min(values), "max_tps": max(values),
            "ms_per_token_p50": 1000.0 / statistics.median(values)}


def main(argv=None):
    args = parse_args(argv)
    try:
        arms = arm_plan(args.passes)
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    if args.execute and args.dry_run:
        print("Configuration error: choose --execute or --dry-run", file=sys.stderr)
        return 2

    matches = []
    try:
        rows = read_process_table()
        matches = codex_matches(rows, self_pid_chain(rows))
    except RuntimeError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    blocking = [entry for entry in matches
                if not (entry["session_only"] and args.allow_codex_idle)]
    plan = {
        "schema": "rkva-decode-host-ab-v1",
        "arms": arms,
        "arm_environment": ARM_ENVIRONMENT,
        "xbox_build_id": args.xbox_build_id,
        "points": args.points,
        "n_predict": args.n_predict,
        "prefix_snapshot": args.prefix_snapshot,
        "trace": args.trace,
        "codex_processes": matches,
        "codex_blocking": [entry["pid"] for entry in blocking],
        "allow_codex_idle": args.allow_codex_idle,
        "bench": str(BENCH),
        "analyzer": str(ANALYZER),
        "caveats": [
            "Both arms must run the same installed Xbox package; the fence spin is compile-time there.",
            "A/B compares only the host headroom-guard sampling until a second package is installed.",
            "Token-ID parity is required before any throughput number is reported as a win.",
        ],
    }
    if blocking:
        plan["refused"] = "Codex benchmark session is active"
        print(json.dumps({"mode": "refused", "reason": plan["refused"],
                          "processes": blocking}, indent=2))
        return 3
    if not args.execute:
        plan["mode"] = "dry-run"
        plan["bench_arguments"] = pass_arguments(args, Path(args.output) / "pass-1")
        print(json.dumps(plan, indent=2))
        return 0

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bench = load_bench()
    analyzer = None
    if args.trace:
        spec = importlib.util.spec_from_file_location("analyze_rkva_decode_trace", ANALYZER)
        analyzer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analyzer)

    passes = []
    for index, arm in enumerate(arms, start=1):
        name = f"pass-{index}-{arm}"
        pass_dir = output / name
        for key, value in ARM_ENVIRONMENT[arm].items():
            bench.PINNED_ENV[key] = value
        print(f"[{name}] arm {arm} environment {ARM_ENVIRONMENT[arm]}", flush=True)
        rc = bench.main(pass_arguments(args, pass_dir))
        entry = {"name": name, "arm": arm, "exit_code": rc, "environment": dict(ARM_ENVIRONMENT[arm])}
        log_path = pass_dir / "server.log"
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.is_file() else []
        entry["guard"] = parse_guard_line(lines)
        profile = parse_profile_line(lines)
        if profile:
            counters = profile.get("counters", {})
            entry["engine"] = {
                "submits": counters.get("submits"),
                "fence_blocking_waits": counters.get("fence_blocking_waits"),
                "transport_decode": profile.get("transport_decode_phases"),
            }
        responses = sorted(pass_dir.glob("point-*.response.json"))
        if responses:
            try:
                entry["tokens"] = tokens_from_response(responses[-1])
                entry["tokens_source"] = responses[-1].name
            except (OSError, ValueError, json.JSONDecodeError) as error:
                entry["tokens_error"] = str(error)
        results_path = pass_dir / "results.json"
        if results_path.is_file():
            results = json.loads(results_path.read_text(encoding="utf-8"))
            entry["status"] = results.get("status")
            entry["decode_tps"] = decode_tps(results)
        if analyzer is not None:
            traces = sorted(pass_dir.glob("rkva-trace*"))
            if traces:
                attribution = pass_dir / "decode-attribution.json"
                analyzer.main([str(traces[0]), "--json", str(attribution)])
                entry["decode_attribution"] = str(attribution)
        passes.append(entry)
        if rc != 0:
            break

    report = {
        "schema": "rkva-decode-host-ab-v1",
        "plan": plan,
        "passes": passes,
        "parity": parity_report(passes),
        "summary": {arm: summarize_arm(passes, arm) for arm in sorted(set(arms))},
        "verdict": "incomplete",
    }
    if len(passes) == len(arms) and all(entry["exit_code"] == 0 for entry in passes):
        baseline = report["summary"].get("A", {})
        candidate = report["summary"].get("B", {})
        if baseline.get("n") and candidate.get("n") and report["parity"]["matched"]:
            gain = 100.0 * (candidate["p50_tps"] / baseline["p50_tps"] - 1.0)
            report["verdict"] = "measured"
            report["delta_percent_p50_tps"] = gain
            guard_active = [entry["guard"] for entry in passes if entry["arm"] == "B" and entry.get("guard")]
            report["guard_fastpath_proven"] = any(entry.get("fastpath", 0) > 0 for entry in guard_active)
        elif not report["parity"]["matched"]:
            report["verdict"] = "parity-failed"
        else:
            report["verdict"] = "incomplete"
    (output / "rkva-decode-host-ab.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("verdict", "parity", "summary")}, indent=2))
    return 0 if report["verdict"] == "measured" else 1


if __name__ == "__main__":
    sys.exit(main())
