#!/usr/bin/env python3
"""Summarize Bee prefill logs, RKVA JSONL, or an Nsight Systems SQLite export."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sqlite3


def server_log(path):
    totals = defaultdict(Counter)
    for line in Path(path).read_text().splitlines():
        match = re.search(r"\[PREFILL (PREP|SCHED|CLASS|SYNC|NEXTN)\] (.*)", line)
        if not match:
            continue
        section, fields = match.groups()
        fields = dict(field.split("=", 1) for field in fields.split())
        key = f"{section}/{fields.get('class', fields.get('ctx', 'all'))}"
        totals[key]["samples"] += 1
        for field, value in fields.items():
            if field not in {"class", "ctx", "start_us"}:
                totals[key][field] += int(value)
    if not totals:
        raise ValueError(f"{path}: no prefill data")
    return totals


def trace(paths):
    totals = defaultdict(Counter)
    latencies = defaultdict(list)
    for path in paths:
        events = [json.loads(line) for line in Path(path).read_text().splitlines()]
        widths = {e["graph_run"]: (e["nq"], e["flags"]) for e in events if e.get("phase") == "strata_verify_ubatch"}
        for e in events:
            phase = e.get("phase", "")
            if e.get("dropped") or phase in {"cuda_local_timing_drop", "cuda_local_timing_unresolved", "cuda_local_profile_error"}:
                totals["trace_loss"][phase or "dropped"] += e.get("value", 0) or e.get("dropped", 0) or 1
            if phase.startswith("cuda_graph_") or phase == "cuda_local_span_gpu":
                if e["graph_run"] not in widths or phase not in {
                    "cuda_graph_decision", "cuda_graph_invalidated", "cuda_graph_fallback", "cuda_local_span_gpu"
                }:
                    continue
                nq, ctx = widths[e["graph_run"]]
                key = f"{phase}/ctx={ctx}/nq={nq}/value={e['value']}"
            elif phase == "strata_verify_ubatch":
                key = f"{phase}/ctx={e['flags']}/nq={e['nq']}"
            elif phase.startswith("strata_"):
                key = f"{phase}/flags={e['flags']}"
            else:
                continue
            row = totals[key]
            row["samples"] += 1
            row["host_us"] += e["end_us"] - e["begin_us"]
            row["bytes"] += e.get("bytes", 0)
            row["gpu_us"] += e.get("gpu_us", 0)
            if phase == "cuda_local_span_gpu" and not e["flags"] & 1:
                latencies[key, "gpu_us"].append(e["gpu_us"])
            elif phase == "strata_verify_ubatch":
                latencies[key, "host_us"].append(e["end_us"] - e["begin_us"])
            elif phase.startswith("strata_"):
                row["value"] += e.get("value", 0)
    for (key, unit), samples in latencies.items():
        samples.sort()
        for percentile in (50, 95):
            totals[key][f"p{percentile}_{unit}"] = samples[math.ceil(len(samples)*percentile/100) - 1]
    return totals


def kernel_class(name):
    name = name.lower()
    if re.search(r"gated_delta|gdn_|ssm_conv|ssm_scan|gated_delta_net_cache", name):
        return "gdn"
    if re.search(r"lightning_indexer|flash_attn|fattn|qsa|sparse_(mask|union)", name):
        return "qsa_fa"
    if re.search(r"(^|[^a-z])(mmq|mmid)|mul_mat_id|mul_mat_q|mul_mat_vec_q|expert|moe_", name):
        return "mmq_moe_candidates"
    return "other"


def union_ns(ranges):
    end = total = 0
    for first, last in sorted(ranges):
        total += max(0, last - max(end, first))
        end = max(end, last)
    return total


def nsys(path, begin, end):
    totals = defaultdict(Counter)
    kernels, copies = defaultdict(list), defaultdict(list)
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    with db:
        for start, stop, dev, name in db.execute(
            "SELECT k.start,k.end,k.deviceId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.demangledName=s.id WHERE k.start >= ? AND k.end <= ?", (begin, end)
        ):
            kernels[dev].append((start, stop))
            row = totals[f"kernel/{dev}/{kernel_class(name)}"]
            row["samples"] += 1
            row["summed_ns"] += stop - start
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            for start, stop, dev, stream, kind, size in db.execute(
                "SELECT start,end,deviceId,streamId,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE start >= ? AND end <= ?", (begin, end)
            ):
                copies[dev].append((start, stop))
                direction = {1: "h2d", 2: "d2h", 8: "d2d", 10: "p2p"}.get(kind, str(kind))
                row = totals[f"copy/{dev}/{stream}/{direction}"]
                row["samples"] += 1
                row["bytes"] += size
                row["summed_ns"] += stop - start
        if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
            for start, stop, name in db.execute(
                "SELECT r.start,r.end,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
                "WHERE r.start >= ? AND r.end <= ?", (begin, end)
            ):
                if any(part in name for part in ("Synchronize", "WaitEvent", "EventRecord", "Memcpy")):
                    row = totals[f"runtime/{name}"]
                    row["samples"] += 1
                    row["host_ns"] += stop - start
        for dev in kernels.keys() | copies.keys():
            compute = union_ns(kernels[dev])
            busy = union_ns(kernels[dev] + copies[dev])
            totals[f"timeline/{dev}"].update(
                kernel_busy_ns=compute, copy_outside_kernel_ns=busy - compute, idle_ns=end - begin - busy
            )
    db.close()
    if not totals:
        raise ValueError("no CUDA kernels or copies entirely inside the selected interval")
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--log")
    inputs.add_argument("--trace", nargs="+")
    inputs.add_argument("--sqlite")
    parser.add_argument("--start-ns", type=int)
    parser.add_argument("--end-ns", type=int)
    args = parser.parse_args()
    if args.sqlite:
        if args.start_ns is None or args.end_ns is None or not 0 <= args.start_ns < args.end_ns:
            parser.error("--sqlite requires a prefill window selected in Nsight with --start-ns and --end-ns")
        totals = nsys(args.sqlite, args.start_ns, args.end_ns)
    else:
        totals = server_log(args.log) if args.log else trace(args.trace)
    print(json.dumps(totals, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
