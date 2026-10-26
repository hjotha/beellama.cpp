#!/usr/bin/env python3
"""RKVA pass1 sweep: the production attention kernel against a measured roofline.

This drives the *existing* diagnostic surface of the RKVA listener, without
changing the production protocol:

  op 20 PROFILE ENABLE|RESET (mode 2, the qualified "fine" policy)
  op 22 KVARN_RANGE_BULK_ATTN    one envelope = ceil(nq/8) segments of 8 queries
                                 (native KVarN4 descriptor revision 3, FP8 wire)
  op 14 RANGE_ATTN               nq == 1, the decode path (no append)
  op 20 PROFILE REPORT_ONLY      JSON with GPU timestamps per segment

It reports the pass1 GPU time per envelope for
nq in {1, 8, 64, 256} x rows in {4096, 8192, 16384, 28672}, plus the kernel name,
split_k, active rows, and the FP8 pipeline counters. No rate is computed here:
the FLOP/byte model lives in rkva_ceiling_roofline.py and the final roofline is
assembled by scripts/run-xbox-ceiling.py.

Offline by default: without --execute it prints the plan and the model table and
opens no socket. --self-test additionally feeds a synthetic profile report
through the aggregation code (no socket, no numpy).

The console must be leased by the caller. This script never deploys, restarts,
or changes anything outside its own RKVA session; it refuses to run while a
host benchmark process is active unless --allow-busy is given.
"""

import argparse
import json
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rkva_ceiling_roofline as roof  # noqa: E402

OP_HELLO, OP_CREATE, OP_ATTN, OP_KVARN_CONFIG = 1, 2, 14, 15
OP_KVARN_UPLOAD, OP_PROFILE, OP_BULK = 16, 20, 22
PROFILE_ENABLE, PROFILE_RESET, PROFILE_REPORT_ONLY = 1, 2, 8
CAP_KVARN4, CAP_PROFILE, CAP_PREFIX, CAP_BULK = 1 << 2, 1 << 5, 1 << 7, 1 << 8
CAP_BULK_F16, CAP_BULK_FP8, CAP_FP8_RNE = 1 << 9, 1 << 10, 1 << 12
PROTO_F16, PROTO_F32 = 1 << 0, 1 << 1
PROTO_NATIVE_REWIND_V2, PROTO_SERIAL_PREFIX = 1 << 16, 1 << 17
WIRE_FP8 = 4
DEFAULT_ROWS = "28672,16384,8192,4096"
DEFAULT_NQ = "1,8,64,256"
SEGMENT_QUERIES = 8


# --------------------------------------------------------------- aggregation ---


def attention_buckets(report):
    buckets = (report.get("engine") or {}).get("buckets") or {}
    return {key: value for key, value in buckets.items() if key.startswith("attention/")}


def select_point(report, expected_kernel, segments, nq):
    """Extract one measured pass1 point from a single-call profile report.

    Fails closed: a fallback kernel, a missing timestamp, or an FP8 CPU
    fallback invalidates the point instead of producing a plausible number.
    """
    engine = report.get("engine") or {}
    if not engine:
        raise ValueError("profile report has no engine section")
    if not engine.get("gpu_timestamps") or not engine.get("gpu_timestamp_frequency_hz"):
        raise ValueError("GPU timestamps unavailable; refusing a CPU-timed rate")
    buckets = attention_buckets(report)
    if len(buckets) != 1:
        raise ValueError("expected exactly one attention bucket, got %d (%s)" %
                         (len(buckets), sorted(buckets)))
    key, bucket = next(iter(buckets.items()))
    if expected_kernel not in key:
        raise ValueError("kernel fallback: bucket %s does not contain %s" % (key, expected_kernel))
    if int(bucket.get("samples", 0)) != segments:
        raise ValueError("bucket has %s samples, expected %d segments" %
                         (bucket.get("samples"), segments))
    if not int(bucket.get("gpu_samples", 0)):
        raise ValueError("bucket has no GPU-timestamped samples")

    bulk = (engine.get("bulk") or {})
    if int(bulk.get("fp8_cpu_fallback_calls", 0)):
        raise ValueError("FP8 CPU fallback ran; the GPU pack pipeline is not qualified")
    counters = engine.get("counters") or {}
    if not int(counters.get("bulk_calls", 0)) and nq != 1:
        raise ValueError("bulk path did not run")

    rows = float((bucket.get("active_rows") or {}).get("mean", 0.0))
    per_segment_us = float((bucket.get("gpu_us") or {}).get("pass1", {}).get("mean", 0.0))
    if per_segment_us <= 0.0:
        raise ValueError("pass1 GPU time missing from bucket %s" % key)
    if nq == 1:
        envelope_us = per_segment_us
        reduce_us = float((bucket.get("gpu_us") or {}).get("reduce", {}).get("mean", 0.0))
        copy_us = float((bucket.get("gpu_us") or {}).get("readback_copy", {}).get("mean", 0.0))
    else:
        envelope_us = float((bulk.get("timings") or {}).get("gpu_pass1_us", {}).get("mean", 0.0))
        reduce_us = float((bulk.get("timings") or {}).get("gpu_reduce_us", {}).get("mean", 0.0))
        copy_us = float((bulk.get("timings") or {}).get("gpu_readback_copy_us", {}).get("mean", 0.0))
        if envelope_us <= 0.0:
            envelope_us = per_segment_us * segments
    return {
        "bucket": key,
        "kernel": expected_kernel,
        "segments": segments,
        "nq": nq,
        "rows": rows,
        "pass1_us": envelope_us,
        "pass1_per_segment_us": per_segment_us,
        "reduce_us": reduce_us,
        "readback_copy_us": copy_us,
        "gpu_seconds": envelope_us / 1e6,
        "split_k": float((bucket.get("split_k") or {}).get("mean", 0.0)),
        "pass1_groups": float((bucket.get("pass1_groups") or {}).get("mean", 0.0)),
        "wave_lane_min": int(engine.get("wave_lane_min", 0)),
        "timestamp_frequency_hz": int(engine.get("gpu_timestamp_frequency_hz", 0)),
        "memory": engine.get("memory") or {},
    }


def expected_kernel_for(nq, wave_lane_min=64, tail=128):
    """Kernel the qualified mode-2 policy must select for this call shape."""
    if nq == 1:
        return "kvarn4_decode_fine" if wave_lane_min >= 64 else "kvarn4_gqa_fine"
    # The shared-record tile needs the 128-row exact tail; without it the
    # per-query optimized kernel is the qualified path.
    return "kvarn4_mq2_fine" if tail == 128 else "kvarn4_gqa_fine"


def aggregate_runs(points):
    """One point per (rows, nq): best and median envelope pass1 over the runs."""
    grouped = {}
    for point in points:
        grouped.setdefault((point["nq"], round(point["rows"])), []).append(point)
    out = []
    for (nq, rows), runs in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        times = sorted(run["gpu_seconds"] for run in runs)
        best = runs[min(range(len(runs)), key=lambda i: runs[i]["gpu_seconds"])]
        median = times[len(times) // 2] if len(times) % 2 else \
            0.5 * (times[len(times) // 2 - 1] + times[len(times) // 2])
        out.append({
            "n_queries": nq,
            "rows": rows,
            "gpu_seconds": best["gpu_seconds"],
            "gpu_seconds_median": median,
            "runs": len(runs),
            "kernel": best["kernel"],
            "bucket": best["bucket"],
            "segments": best["segments"],
            "pass1_per_segment_us": best["pass1_per_segment_us"],
            "reduce_us": best["reduce_us"],
            "readback_copy_us": best["readback_copy_us"],
            "split_k": best["split_k"],
            "pass1_groups": best["pass1_groups"],
            "wave_lane_min": best["wave_lane_min"],
            "timestamp_frequency_hz": best["timestamp_frequency_hz"],
            "gpu_memory": best["memory"],
        })
    return out


# ------------------------------------------------------------------ hardware ---


BUSY_EXECUTABLES = ("llama-server", "llama-bench", "llama-cli", "bench-rkva",
                    "test-rkva", "ninja", "cmake")
# Interpreter-like argv[0] values may carry a whole prompt on their command
# line (this agent's own session does), so only a real executable basename can
# make the guard fire.
BUSY_PREFIXES = ("bench-rkva", "test-rkva")


def busy_processes():
    """Host benchmark/toolchain processes, matched by executable name only.

    `pgrep -f` would match the task text carried by an agent or shell command
    line; this walks /proc and compares the basename of argv[0] instead, which
    is what the Codex benchmark and a build actually run as.
    """
    found = []
    proc_root = "/proc"
    if not os.path.isdir(proc_root):
        return found
    mine = os.getpid()
    for entry in os.listdir(proc_root):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == mine:
            continue
        try:
            with open(os.path.join(proc_root, entry, "cmdline"), "rb") as handle:
                raw = handle.read()
        except OSError:
            continue
        if not raw:
            continue
        argv0 = raw.split(b"\0", 1)[0].decode("utf-8", "replace")
        name = os.path.basename(argv0)
        if name in BUSY_EXECUTABLES or name.startswith(BUSY_PREFIXES):
            try:
                with open(os.path.join(proc_root, entry, "cmdline"), "rb") as handle:
                    full = handle.read().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                full = name
            found.append("pid=%d %s" % (pid, full.strip()[:200]))
    return found


def encode_fp8_rne(np, values):
    """F32 -> F16 -> FP8 E5M2 with round-to-nearest-even (the deployed pack)."""
    half = np.asarray(values, dtype="<f4").astype("<f2").view("<u2").astype(np.uint32)
    mag = np.minimum(half & 0x7FFF, 0x7B00)
    return (((half >> 8) & 0x80) | ((mag + 0x7F + ((mag >> 8) & 1)) >> 8)).astype(np.uint8)


def execute(args):
    import numpy as np
    import rkva_kvarn_serial_prefix_probe as p
    import rkva_kvarn_serial_prefix_model as sp
    from rkva_range_probe import Client

    busy = busy_processes()
    if busy and not args.allow_busy:
        raise SystemExit("refusing --execute: benchmark processes are active:\n  " +
                         "\n  ".join(busy[:8]))

    rows_targets = sorted({int(v) for v in args.rows.split(",") if v.strip()}, reverse=True)
    nq_targets = [int(v) for v in args.nq.split(",") if v.strip()]
    if not rows_targets or not nq_targets:
        raise SystemExit("empty rows/nq plan")
    for rows in rows_targets:
        if rows % 128 or rows < 1024:
            raise SystemExit("rows must be a multiple of 128 and >= 1024: %d" % rows)
    slack = 128 + max(nq_targets) * (args.runs + args.warmup) + 512

    rng = np.random.default_rng(args.seed)
    client = Client(args.host, args.port)
    points = []
    sessions = []
    meta = {"hello": {}, "rows": rows_targets, "nq": nq_targets, "seed": args.seed,
            "wire": "fp8", "mode": 2, "tail": args.tail, "segment_queries": SEGMENT_QUERIES}
    try:
        flags = PROTO_F16 | PROTO_F32 | PROTO_NATIVE_REWIND_V2 | PROTO_SERIAL_PREFIX
        capacity_probe = max(rows_targets) + slack
        p.CAPACITY = capacity_probe
        p.ORIGIN = args.origin
        ack = struct.unpack("<IIIIQQQ", client.call(OP_HELLO, p.hello_payload(1, flags)))
        caps = ack[5]
        meta["hello"] = {"supported_ops": ack[0], "server_version": ack[3],
                         "mem_budget_bytes": ack[4], "capabilities": caps}
        needed = CAP_KVARN4 | CAP_PROFILE | CAP_PREFIX | CAP_BULK | CAP_BULK_FP8 | CAP_FP8_RNE
        missing = [name for name, bit in (("kvarn4", CAP_KVARN4), ("profile", CAP_PROFILE),
                                          ("serial_prefix", CAP_PREFIX), ("bulk", CAP_BULK),
                                          ("bulk_fp8", CAP_BULK_FP8), ("fp8_rne", CAP_FP8_RNE))
                   if not caps & bit]
        if missing:
            raise SystemExit("listener lacks required capabilities: " + ", ".join(missing))
        if not ack[0] & (1 << OP_BULK):
            raise SystemExit("listener does not advertise op 22")

        for rows in rows_targets:
            capacity = rows + slack
            p.CAPACITY = capacity
            client.call(OP_CREATE, struct.pack("<4I", 0, 0, capacity, 1))
            generation = struct.unpack("<Q", bytes(client.call(
                OP_KVARN_CONFIG, struct.pack("<18I", *p.descriptor(args.tail)), session=0)))[0]
            layer = p.Layer(np, sp, rng, args.tail)
            sessions.append({"rows": rows, "capacity": capacity})
            while layer.live < rows:
                chunk_begin = layer.live
                chunk = min(args.upload_chunk, rows - chunk_begin)
                layer.canon.append(chunk)
                payload = layer.upload(chunk_begin + chunk, reset=(chunk_begin == 0))
                client.call(OP_KVARN_UPLOAD, payload, session=0)
                generation += 1
            uploaded = layer.live

            for nq in nq_targets:
                for run in range(args.warmup + args.runs):
                    client.call(OP_PROFILE, struct.pack("<IIQ", PROFILE_ENABLE | PROFILE_RESET, 2, 0))
                    if nq == 1:
                        q = rng.standard_normal((roof.N_HEAD, roof.HEAD_DIM)).astype("<f4")
                        positions = [args.origin + layer.live - 1]
                        body = struct.pack("<4I", 1, 0, 0, 0)
                        body += struct.pack("<i", *positions) + q.tobytes()
                        client.call(OP_ATTN, body, session=0, nq=1)
                        segments = 1
                    else:
                        chunks = []
                        remaining = nq
                        while remaining:
                            count = min(SEGMENT_QUERIES, remaining)
                            begin = layer.live
                            layer.canon.append(count)
                            upload = layer.upload(begin + count, reset=False)
                            positions = list(range(args.origin + begin, args.origin + begin + count))
                            q = rng.standard_normal((count, roof.N_HEAD, roof.HEAD_DIM)).astype("<f4")
                            q_bytes = encode_fp8_rne(np, q).tobytes()
                            query = struct.pack("<4I", WIRE_FP8, 0, 0, 0)
                            query += struct.pack("<%di" % count, *positions) + q_bytes
                            chunks.append((upload, query, count))
                            remaining -= count
                        segments = len(chunks)
                        bulk = struct.pack("<4I", 1, segments, nq, 0) + b"".join(
                            struct.pack("<2I", len(upload), count) + upload + query
                            for upload, query, count in chunks)
                        client.call(OP_BULK, bulk, session=0, nq=nq)
                    report = json.loads(bytes(client.call(
                        OP_PROFILE, struct.pack("<IIQ", PROFILE_REPORT_ONLY, 0, 0))).decode("utf-8"))
                    if run < args.warmup:
                        continue
                    point = select_point(
                        report,
                        expected_kernel_for(
                            nq, int(((report.get("engine") or {}).get("wave_lane_min")) or 64),
                            args.tail),
                        segments, nq)
                    points.append(point)
                    print("rows=%d nq=%d run=%d pass1=%.1fus kernel=%s" %
                          (rows, nq, run, point["pass1_us"], point["kernel"]), flush=True)
    finally:
        try:
            client.socket.close()
        except OSError:
            pass

    result = {
        "schema": "rkva-pass1-sweep-v1",
        "execute": True,
        "host": args.host,
        "port": args.port,
        "plan": meta,
        "sessions": sessions,
        "client_timings": client.timings,
        "points": aggregate_runs(points),
    }
    text = json.dumps(result, indent=1)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print("wrote " + args.output)
    else:
        print(text)
    return 0


# ------------------------------------------------------------------- offline ---


def model_table(rows_targets, nq_targets):
    table = []
    for rows in rows_targets:
        for nq in nq_targets:
            flops = roof.pass1_flops(nq, rows)
            kv = roof.kvarn4_kv_bytes(rows)
            table.append({
                "rows": rows,
                "n_queries": nq,
                "segments": 1 if nq == 1 else math.ceil(nq / SEGMENT_QUERIES),
                "gflop": flops / 1e9,
                "kv_mib": kv / (1 << 20),
                "qo_mib": roof.qo_bytes(nq, wire_bytes=4) / (1 << 20),
            })
    return table


def synthetic_report(nq, segments):
    """Minimal REPORT_ONLY shape for the aggregation self-test."""
    per_segment_us = 2000.0 + 10.0 * nq
    total_us = per_segment_us * segments
    bucket = {
        "samples": segments,
        "gpu_samples": segments,
        "n_tokens": {"n": segments, "mean": 8.0 if nq > 1 else 1.0},
        "active_rows": {"n": segments, "mean": 28672.0},
        "split_k": {"n": segments, "mean": 64.0},
        "pass1_groups": {"n": segments, "mean": 32400.0},
        "gpu_us": {"pass1": {"n": segments, "mean": per_segment_us},
                   "reduce": {"n": segments, "mean": 60.0},
                   "readback_copy": {"n": segments, "mean": 1.2},
                   "total": {"n": segments, "mean": per_segment_us + 61.2}},
    }
    report = {
        "engine": {
            "profiling": True,
            "gpu_timestamps": True,
            "gpu_timestamp_frequency_hz": 10000000,
            "wave_lane_min": 64,
            "wave_lane_max": 64,
            "counters": {"bulk_calls": 1 if nq > 1 else 0},
            "memory": {"gpu_local_budget_bytes": 4348654388},
            "bulk": {
                "fp8_cpu_fallback_calls": 0,
                "timings": {"gpu_pass1_us": {"n": 1, "mean": total_us},
                            "gpu_reduce_us": {"n": 1, "mean": 60.0 * segments},
                            "gpu_readback_copy_us": {"n": 1, "mean": 1.2 * segments}},
            },
            "buckets": {("attention/kvarn4-prefix/optimized-%s/nq2-16/r16k-32k/warm" %
                         expected_kernel_for(nq)): bucket},
        }
    }
    return report


def self_test():
    for nq in (1, 8, 64, 256):
        segments = 1 if nq == 1 else math.ceil(nq / SEGMENT_QUERIES)
        report = synthetic_report(nq, segments)
        point = select_point(report, expected_kernel_for(nq), segments, nq)
        expected_us = (2000.0 + 10.0 * nq) * segments
        assert abs(point["pass1_us"] - expected_us) < 1e-6, (nq, point["pass1_us"], expected_us)
        assert point["rows"] == 28672.0
    # A fallback kernel must be rejected, not silently averaged in.
    bad = synthetic_report(8, 1)
    try:
        select_point(bad, "kvarn4_mq4_balanced", 1, 8)
    except ValueError as exc:
        assert "fallback" in str(exc), exc
    else:
        raise AssertionError("kernel fallback was accepted")
    # Missing GPU timestamps must be rejected.
    bad = synthetic_report(8, 1)
    bad["engine"]["gpu_timestamps"] = False
    try:
        select_point(bad, expected_kernel_for(8), 1, 8)
    except ValueError as exc:
        assert "timestamp" in str(exc), exc
    else:
        raise AssertionError("missing timestamps were accepted")
    # FP8 CPU fallback must be rejected.
    bad = synthetic_report(8, 1)
    bad["engine"]["bulk"]["fp8_cpu_fallback_calls"] = 3
    try:
        select_point(bad, expected_kernel_for(8), 1, 8)
    except ValueError as exc:
        assert "fallback" in str(exc), exc
    else:
        raise AssertionError("FP8 CPU fallback was accepted")
    # Aggregation: median and best over runs.
    runs = [{"nq": 8, "rows": 4096.0, "gpu_seconds": s, "kernel": "k", "bucket": "b",
             "segments": 1, "pass1_per_segment_us": 0.0, "reduce_us": 0.0,
             "readback_copy_us": 0.0, "split_k": 0.0, "pass1_groups": 0.0,
             "wave_lane_min": 64, "timestamp_frequency_hz": 1, "memory": {}}
            for s in (0.003, 0.002, 0.004)]
    agg = aggregate_runs(runs)
    assert len(agg) == 1 and agg[0]["gpu_seconds"] == 0.002 and agg[0]["runs"] == 3
    assert abs(agg[0]["gpu_seconds_median"] - 0.003) < 1e-12
    print("self-test: OK (4 shapes, 3 fail-closed paths, aggregation)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true",
                        help="talk to a live RKVA listener (default: offline plan only)")
    parser.add_argument("--self-test", action="store_true",
                        help="validate the aggregation and fail-closed paths offline")
    parser.add_argument("--host", default="192.168.1.200")
    parser.add_argument("--port", type=int, default=50054)
    parser.add_argument("--rows", default=DEFAULT_ROWS)
    parser.add_argument("--nq", default=DEFAULT_NQ)
    parser.add_argument("--origin", type=int, default=102400)
    parser.add_argument("--tail", type=int, default=128, choices=(0, 128))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--upload-chunk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--allow-busy", action="store_true",
                        help="skip the host benchmark-process guard")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        if not args.execute:
            return 0
    if args.execute:
        return execute(args)

    rows_targets = sorted({int(v) for v in args.rows.split(",") if v.strip()}, reverse=True)
    nq_targets = [int(v) for v in args.nq.split(",") if v.strip()]
    table = model_table(rows_targets, nq_targets)
    print(json.dumps({
        "execute": False,
        "plan": {"host": args.host, "port": args.port, "origin": args.origin,
                 "tail": args.tail, "runs": args.runs, "warmup": args.warmup,
                 "segment_queries": SEGMENT_QUERIES, "rows": rows_targets, "nq": nq_targets},
        "model_table": table,
        "note": "FLOP/bytes are the model, not a measurement; --execute records GPU timestamps",
        "busy": busy_processes()[:8],
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
