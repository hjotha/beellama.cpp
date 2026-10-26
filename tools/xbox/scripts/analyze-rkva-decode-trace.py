#!/usr/bin/env python3
"""Per-layer decode attribution from a recorded RKVA host trace.

The RKVA host trace (GGML_RKVA_TRACE_FILE) interleaves every backend scope, so
raw sums overlap. This tool keeps only decode events (nq == 1 with
first_pos >= --min-pos), prints per-phase medians/p95/sums and the per-token
attribution, and never touches GPU, RPC or the Xbox.

Usage:
  analyze-rkva-decode-trace.py TRACE.jsonl [--min-pos 130560] [--json OUT]

Read-only. No network access.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict

# Per-token totals for phases that fire once per decode layer.
PER_LAYER_PHASES = (
    "rpc_send", "rpc_header_wait", "rpc_body_recv", "rpc_total", "range_wait",
    "job_prepare", "job_finish", "kv_producer", "q_pack", "k_readback",
    "v_readback", "positions_readback", "out_decode", "envelope",
)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--min-pos", type=int, default=130560,
                        help="first_pos of the decode window (default: 130560)")
    parser.add_argument("--json", help="write the aggregated report as JSON")
    return parser.parse_args(argv)


def summarize(samples):
    ordered = sorted(samples)
    count = len(ordered)
    return {
        "n": count,
        "p50_us": statistics.median(ordered),
        "p95_us": ordered[min(count - 1, int(round(0.95 * (count - 1))))],
        "sum_us": sum(ordered),
    }


def main(argv=None):
    args = parse_args(argv)
    phases = defaultdict(list)
    gpu = defaultdict(list)
    tokens = set()
    scanned = kept = 0
    with open(args.trace, "r", encoding="utf-8", errors="replace") as trace:
        for line in trace:
            scanned += 1
            if '"nq":1' not in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("nq") != 1 or int(event.get("first_pos") or 0) < args.min_pos:
                continue
            kept += 1
            phase = event.get("phase") or ""
            phases[phase].append(int(event.get("end_us", 0)) - int(event.get("begin_us", 0)))
            gpu[phase].append(float(event.get("gpu_us") or 0.0))
            if phase == "rpc_total":
                tokens.add(int(event.get("graph_run") or 0))

    report = {"trace": args.trace, "min_pos": args.min_pos, "scanned": scanned,
              "decode_events": kept, "decode_tokens": len(tokens),
              "phases": {}, "per_token_ms": {}, "note":
              "CUDA spans overlap by design; only per-token sums of once-per-layer phases are additive."}
    for phase in sorted(phases, key=lambda name: -sum(phases[name])):
        entry = summarize(phases[phase])
        entry["gpu_p50_us"] = statistics.median(gpu[phase])
        report["phases"][phase] = entry

    print(f"scanned={scanned} decode_events={kept} decode_tokens={len(tokens)}")
    print(f"{'phase':30s} {'n':>6s} {'p50_us':>10s} {'p95_us':>10s} {'sum_ms':>10s} {'gpu_p50_us':>11s}")
    for phase, entry in report["phases"].items():
        print(f"{phase:30s} {entry['n']:6d} {entry['p50_us']:10.0f} {entry['p95_us']:10.0f} "
              f"{entry['sum_us']/1000.0:10.3f} {entry['gpu_p50_us']:11.3f}")

    token_count = len(tokens) or 1
    print("\nper-token milliseconds (sum of once-per-layer phases / decode tokens):")
    for phase in PER_LAYER_PHASES:
        entry = report["phases"].get(phase)
        if not entry:
            continue
        per_token_ms = entry["sum_us"] / 1000.0 / token_count
        report["per_token_ms"][phase] = per_token_ms
        print(f"  {phase:30s} {per_token_ms:8.3f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump(report, out, indent=2, sort_keys=True)
            out.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
