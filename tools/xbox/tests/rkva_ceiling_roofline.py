#!/usr/bin/env python3
"""Roofline model and derivation helpers for the Xbox ceiling suite.

Pure Python (no numpy, no socket): every function here is unit-tested by
tests/test_rkva_ceiling_roofline.py and used by both the pass1 sweep driver and
the hardware orchestration script, so the FLOP/byte accounting exists in one
place.

Measured versus derived is explicit: this module never produces a "measured"
value. `pass1_flops`, `kvarn4_kv_bytes`, and the nominal constants are the
model; a measured timestamp enters only as `gpu_seconds`.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

# ---------------------------------------------------------------------------
# Geometry of the deployed overflow tier (Qwen3.8-27B on the Xbox Series X).
# These are the production shapes, not tunables of this probe.
# ---------------------------------------------------------------------------
N_HEAD = 24
N_HEAD_KV = 4
HEAD_DIM = 256
KVARN_GROUP = 128
KVARN_RECORD_BYTES = 8960  # 8192 payload + 3 * 128 * F16 affine axes
KVARN_SLICES = 2  # each KV head is stored as two 128-dim slices
KVARN_PRECISION_TAIL = 128

# Published Series X figures. Nominal, not measured; the JSON labels them.
NOMINAL = {
    "fp32_tflops": 12.15,   # 52 dual-CUs * 64 lanes * 2 FLOP * 1.825 GHz
    "fp16_tflops": 24.30,   # packed half2 FMA at twice the FP32 rate
    "int8_tops": 48.6,      # 4 int8 MAC per lane per clock
    "fast_gbs": 560.0,      # 10 GB GDDR6 pool, 320-bit, 14 Gbps
    "slow_gbs": 336.0,      # 6 GB GDDR6 pool, 192-bit
    "dual_cu": 52,
    "note": "published Series X specification; not measured by this suite",
}

# RDNA2 FP32 throughput per dual-CU per clock: 64 lanes * 2 FLOP (one FMA).
# Only used to derive a clock from the *measured* FP32 peak; never presented as
# a measured clock.
FLOP_PER_DUAL_CU_CLOCK = 128.0
NOMINAL_CLOCK_MHZ = 1825.0


def pass1_macs(n_queries: int, rows: int, n_head: int = N_HEAD,
               head_dim: int = HEAD_DIM) -> int:
    """Multiply-accumulates of one attention pass1 for a single layer.

    Two matmuls per (query, head): scores = Q . K over `rows` (head_dim MAC
    each) and out = softmax(scores) . V (also head_dim MAC per row).
    """
    if n_queries <= 0 or rows <= 0:
        return 0
    return 2 * n_queries * n_head * rows * head_dim


def pass1_flops(n_queries: int, rows: int, n_head: int = N_HEAD,
                head_dim: int = HEAD_DIM) -> int:
    """FLOP = 2 * MAC (one multiply + one add)."""
    return 2 * pass1_macs(n_queries, rows, n_head, head_dim)


def kvarn4_record_bytes(rows: int, kv_heads: int = N_HEAD_KV) -> int:
    """Unique KVarN4 record bytes for a complete 128-row group prefix.

    Records exist only for complete groups; an incomplete trailing group lives
    in the stage plane and (with the qualified policy) the exact F16 tail.
    """
    if rows <= 0:
        return 0
    groups = rows // KVARN_GROUP
    return groups * kv_heads * KVARN_SLICES * KVARN_RECORD_BYTES * 2  # K and V planes


def kvarn4_stage_tail_bytes(rows: int, kv_heads: int = N_HEAD_KV,
                            head_dim: int = HEAD_DIM,
                            tail_tokens: int = KVARN_PRECISION_TAIL) -> int:
    """Stage plane plus exact F16 tail, both planes, current live prefix.

    The stage keeps up to three 128-row slots; only the slots the current live
    prefix can read are counted (partial group + the exact-tail ring is one
    group for the qualified 128-token tail), which is the smallest image any
    query actually touches.
    """
    if rows <= 0:
        return 0
    partial = KVARN_GROUP if rows % KVARN_GROUP else 0
    stage_rows = min(KVARN_GROUP * 2, partial + KVARN_GROUP)
    tail_rows = min(rows, tail_tokens)
    return 2 * (stage_rows + tail_rows) * kv_heads * head_dim * 2  # F16, K+V


def kvarn4_kv_bytes(rows: int, kv_heads: int = N_HEAD_KV, head_dim: int = HEAD_DIM,
                    tail_tokens: int = KVARN_PRECISION_TAIL) -> int:
    """Compulsory KV bytes for one layer: records + stage + exact tail."""
    return (kvarn4_record_bytes(rows, kv_heads) +
            kvarn4_stage_tail_bytes(rows, kv_heads, head_dim, tail_tokens))


def qo_bytes(n_queries: int, n_head: int = N_HEAD, head_dim: int = HEAD_DIM,
             wire_bytes: int = 4) -> int:
    """Query and output planes (F32 wire on op14; F16 on the bulk path)."""
    return 2 * max(0, n_queries) * n_head * head_dim * wire_bytes


def wht_flops(n_queries: int, n_head: int = N_HEAD) -> int:
    """Rotated-domain WHT of Q and inverse WHT of OUT (2 * 8 * 256 ops per head)."""
    return 2 * max(0, n_queries) * n_head * 2 * (8 * HEAD_DIM)


def pass1_rates(n_queries: int, rows: int, gpu_seconds: float,
                bandwidth_gbs: float = NOMINAL["fast_gbs"],
                fp32_tflops: float = NOMINAL["fp32_tflops"]) -> Dict[str, float]:
    """Measured pass1 point placed against the *measured* ceiling.

    bandwidth_gbs / fp32_tflops must be the values the ceiling suite measured;
    the nominal table is only used by the caller to compute fractions of spec.
    """
    flops = pass1_flops(n_queries, rows)
    unique = kvarn4_kv_bytes(rows) + qo_bytes(n_queries)
    issued_upper = kvarn4_kv_bytes(rows) * max(1, n_queries) + qo_bytes(n_queries)
    out: Dict[str, float] = {
        "n_queries": float(n_queries),
        "rows": float(rows),
        "gpu_seconds": float(gpu_seconds),
        "flops": float(flops),
        "unique_bytes": float(unique),
        "issued_bytes_upper_bound": float(issued_upper),
    }
    if gpu_seconds > 0:
        out["tflops"] = flops / 1e12 / gpu_seconds
        out["gbs_unique"] = unique / 1e9 / gpu_seconds
        out["gbs_issued_upper"] = issued_upper / 1e9 / gpu_seconds
    else:
        out["tflops"] = 0.0
        out["gbs_unique"] = 0.0
        out["gbs_issued_upper"] = 0.0
    if bandwidth_gbs > 0:
        out["bandwidth_fraction_pct"] = 100.0 * out["gbs_unique"] / bandwidth_gbs
        out["bandwidth_fraction_issued_pct"] = 100.0 * out["gbs_issued_upper"] / bandwidth_gbs
    else:
        out["bandwidth_fraction_pct"] = 0.0
        out["bandwidth_fraction_issued_pct"] = 0.0
    if fp32_tflops > 0:
        out["compute_fraction_pct"] = 100.0 * out["tflops"] / fp32_tflops
    else:
        out["compute_fraction_pct"] = 0.0
    # The measured roofline bound: whichever ceiling the point is closer to.
    out["bound"] = ("bandwidth" if out["bandwidth_fraction_issued_pct"] >
                    out["compute_fraction_pct"] else "compute")
    # Headroom to the ceiling the point is NOT bound by, as a multiplier.
    if out["tflops"] > 0 and bandwidth_gbs > 0:
        out["speedup_cap_vs_measured_ceiling"] = min(
            fp32_tflops / out["tflops"] if out["tflops"] else 0.0,
            bandwidth_gbs / out["gbs_unique"] if out["gbs_unique"] else 0.0)
    else:
        out["speedup_cap_vs_measured_ceiling"] = 0.0
    return out


def derived_clock_mhz(fp32_tflops: float, dual_cu: float = NOMINAL["dual_cu"],
                      flops_per_cu_clock: float = FLOP_PER_DUAL_CU_CLOCK) -> float:
    """Shader clock derived from the *measured* FP32 peak.

    clock = TFLOP/s * 1e12 / (dual_cu * 128 FLOP/cycle) / 1e6

    This assumes the FP32 probe is issue-bound at 64 lanes per CU and that the
    measured peak is the sustained peak at that clock. It is a derived number,
    not a hardware clock measurement; the JSON labels it as such.
    """
    if fp32_tflops <= 0 or dual_cu <= 0 or flops_per_cu_clock <= 0:
        return 0.0
    return fp32_tflops * 1e12 / (dual_cu * flops_per_cu_clock) / 1e6


def fit_concurrency(groups: Sequence[int], seconds: Sequence[float],
                    max_concurrency: int = 1024) -> Dict[str, float]:
    """Infer how many equal-cost workgroups run concurrently.

    Model: one dispatch of N groups takes ceil(N / P) waves of one group each,
    so seconds_i = t * ceil(N_i / P). For every candidate P the implied group
    time is t_i = seconds_i / ceil(N_i / P); the correct P is the one whose
    t_i agree best (smallest relative spread). Returns the fit and its quality.
    Nothing is asserted about CUs here: the caller divides by the LDS-limited
    groups-per-CU of the probe.
    """
    pairs = [(int(n), float(s)) for n, s in zip(groups, seconds) if n > 0 and s > 0]
    if len(pairs) < 3:
        return {"concurrency": 0.0, "group_seconds": 0.0, "relative_spread": 0.0,
                "samples": float(len(pairs)), "confident": 0.0}
    best = None
    for p in range(1, max_concurrency + 1):
        times = [s / math.ceil(n / p) for n, s in pairs]
        lo, hi = min(times), max(times)
        spread = (hi - lo) / hi if hi > 0 else 1.0
        if best is None or spread < best[1]:
            best = (p, spread, sorted(times)[len(times) // 2])
    assert best is not None
    return {
        "concurrency": float(best[0]),
        "group_seconds": best[2],
        "relative_spread": best[1],
        "samples": float(len(pairs)),
        # Below 15% spread the ceil() staircase is unambiguous with our sweep.
        "confident": 1.0 if best[1] <= 0.15 else 0.0,
    }


def cu_estimate(concurrency_32kib: float, concurrency_16kib: float,
                groups_per_cu_32kib: float = 2.0,
                groups_per_cu_16kib: float = 4.0) -> Dict[str, float]:
    """Two independent CU estimates from the LDS-limited plateaus.

    A 64 KiB CU LDS admits 2 groups of 32 KiB or 4 groups of 16 KiB. Agreement
    between the two estimates is the evidence; disagreement means another
    limit (VGPR/wave slots) bound occupancy first and the number is withheld.
    """
    est32 = concurrency_32kib / groups_per_cu_32kib if groups_per_cu_32kib else 0.0
    est16 = concurrency_16kib / groups_per_cu_16kib if groups_per_cu_16kib else 0.0
    agree = bool(est32 and est16 and abs(est32 - est16) <= 0.15 * max(est32, est16))
    return {
        "dual_cu_from_32kib": est32,
        "dual_cu_from_16kib": est16,
        "agree": 1.0 if agree else 0.0,
        "dual_cu": round((est32 + est16) / 2.0) if agree else 0.0,
    }


def parse_gpupeak_json(document: Dict) -> Dict[str, object]:
    """Validate and index the console probe document.

    Hard failures (wrong schema, D3D12 did not run, no timestamp frequency)
    still raise: without them no GPU-timed value exists. A failed or missing
    stream checksum is *not* fatal: it is recorded as a warning so the ALU,
    latency and occupancy sections can still produce a roofline, explicitly
    marked partial. The caller must not use an unverified bandwidth ceiling.
    """
    if document.get("schema") != "xllama-gpu-ceiling-v1":
        raise ValueError("not an xllama-gpu-ceiling-v1 document")
    device = document.get("device") or {}
    if not document.get("d3d12_ran"):
        raise ValueError("probe did not run: %s" % document.get("error", "unknown"))
    if not device.get("timestamp_frequency"):
        raise ValueError("no timestamp frequency: GPU-timed rates are unavailable")
    rows = document.get("rows") or []
    by_probe: Dict[str, List[Dict]] = {}
    for row in rows:
        by_probe.setdefault(str(row.get("probe")), []).append(row)

    warnings: List[str] = []
    missing = []
    for probe in ("read", "write", "copy"):
        timed = [r for r in by_probe.get(probe, []) if r.get("gpu_valid") and r.get("runs")]
        if not timed:
            missing.append(probe)
    if missing:
        warnings.append("missing stream rows: " + ", ".join(missing))
    evidence = [r for r in by_probe.get("read", []) + by_probe.get("copy", [])
                if r.get("checksum_checked")]
    failed = [r for r in evidence if not r.get("checksum_ok")]
    if not evidence:
        warnings.append("stream checksum evidence missing")
    elif failed:
        details = []
        for row in failed:
            name = "%s/%s" % (row.get("probe"), row.get("variant"))
            got = row.get("checksum")
            want = row.get("expected_checksum")
            if got is not None and want is not None:
                name += " got=0x%08x want=0x%08x" % (int(got), int(want))
            if row.get("error"):
                name += " (" + str(row["error"]) + ")"
            details.append(name)
        warnings.append("stream checksum failed: " + "; ".join(details))

    stream_verified = not missing and bool(evidence) and not failed
    return {
        "device": device,
        "rows": rows,
        "by_probe": by_probe,
        "warnings": warnings,
        "stream_verified": stream_verified,
        "stream_checksum_ok": bool(evidence) and not failed,
    }


def best(rows: Iterable[Dict], probe: str) -> Dict:
    """Fastest valid row of a fixed-size probe (stream or latency).

    For an ALU group sweep the shortest dispatch does less work; use
    best_throughput() there.
    """
    candidates = [r for r in rows if r.get("probe") == probe and r.get("gpu_valid")
                  and r.get("gpu_seconds", 0) > 0]
    if not candidates:
        return {}
    return min(candidates, key=lambda r: r["gpu_seconds"])


def throughput_per_second(row: Dict) -> float:
    """Attributed FLOP/OPS per GPU second for one row."""
    if not row or not row.get("gpu_seconds"):
        return 0.0
    return row.get("flops", 0.0) / row["gpu_seconds"]


def best_throughput(rows: Iterable[Dict], probe: str) -> Dict:
    """Highest-throughput row of an ALU probe.

    `best()` minimises time, which is right for a fixed-size stream or a
    latency sample but wrong for a group sweep: the 52-group dispatch is always
    shorter than the 1664-group one while doing less work. The ALU ceiling must
    come from the largest FLOP/s, not the shortest dispatch.
    """
    candidates = [r for r in rows if r.get("probe") == probe and r.get("gpu_valid")
                  and r.get("gpu_seconds", 0) > 0]
    if not candidates:
        return {}
    return max(candidates, key=throughput_per_second)


def stream_rate_gbs(row: Dict, traffic_multiplier: float = 1.0) -> float:
    if not row or not row.get("gpu_seconds"):
        return 0.0
    return row["bytes"] * traffic_multiplier / 1e9 / row["gpu_seconds"]


def build_result(gpupeak: Dict, sweep: Dict) -> Dict:
    """Assemble xbox-ceiling-result.json: measured ceilings + pass1 roofline.

    `gpupeak` is the raw probe document; `sweep` is the pass1 driver output
    ({"points": [{"n_queries", "rows", "gpu_seconds", ...}]}).

    A failed stream checksum is a partial failure, not an abort: the ALU,
    latency and occupancy sections still produce their roofline entries, the
    untrusted stream rates are reported as observations only, and the document
    carries `"partial": true` plus the warnings.
    """
    parsed = parse_gpupeak_json(gpupeak)
    rows = parsed["rows"]
    device = parsed["device"]
    warnings = list(parsed["warnings"])
    stream_verified = bool(parsed["stream_verified"])
    observed_read_gbs = stream_rate_gbs(best(rows, "read"))
    observed_write_gbs = stream_rate_gbs(best(rows, "write"))
    observed_copy_gbs = stream_rate_gbs(best(rows, "copy"), 2.0)
    read_gbs = observed_read_gbs if stream_verified else 0.0
    write_gbs = observed_write_gbs if stream_verified else 0.0
    copy_gbs = observed_copy_gbs if stream_verified else 0.0
    measured_bw = max(read_gbs, copy_gbs)
    fp32 = best_throughput(rows, "fma_f32")
    fp16 = best_throughput(rows, "fma_f16")
    fp16_min16 = best_throughput(rows, "fma_f16_min16")
    dot2 = best_throughput(rows, "dot2_f16")
    dot4 = best_throughput(rows, "dot4_i8")
    fp32_tflops = throughput_per_second(fp32) / 1e12
    fp16_tflops = throughput_per_second(fp16) / 1e12
    fp16_min16_tflops = throughput_per_second(fp16_min16) / 1e12
    dot2_tflops = throughput_per_second(dot2) / 1e12
    dot4_tops = throughput_per_second(dot4) / 1e12
    if not fp32:
        warnings.append("missing fma_f32 rows: compute ceiling unavailable")

    occupancy = {"cu32": [], "cu16": []}
    for row in rows:
        for key in occupancy:
            if row.get("probe") == key and row.get("gpu_valid"):
                occupancy[key].append(row)
    cu = {"dual_cu": 0.0, "agree": 0.0, "dual_cu_from_32kib": 0.0,
          "dual_cu_from_16kib": 0.0, "concurrency_32kib": 0.0, "concurrency_16kib": 0.0}
    if occupancy["cu32"] and occupancy["cu16"]:
        fit32 = fit_concurrency([r["groups"] for r in occupancy["cu32"]],
                                [r["gpu_seconds"] for r in occupancy["cu32"]])
        fit16 = fit_concurrency([r["groups"] for r in occupancy["cu16"]],
                                [r["gpu_seconds"] for r in occupancy["cu16"]])
        est = cu_estimate(fit32["concurrency"], fit16["concurrency"])
        cu.update(est)
        cu.update({"concurrency_32kib": fit32["concurrency"],
                   "concurrency_16kib": fit16["concurrency"],
                   "fit_32kib": fit32, "fit_16kib": fit16})

    latency = {name: best(rows, name) for name in
               ("empty_dispatch_gpu", "empty_dispatch_batch")}
    derived_clock = derived_clock_mhz(fp32_tflops)

    ceiling = {
        # Zero when the stream checksum failed: an unverified rate must not be
        # promoted to the bandwidth ceiling. The observations are kept below.
        "bandwidth_gbs": measured_bw,
        "bandwidth_verified": stream_verified,
        "stream_observed_gbs": {
            "read": observed_read_gbs,
            "write": observed_write_gbs,
            "copy_read_plus_write": observed_copy_gbs,
        },
        "read_gbs": read_gbs,
        "write_gbs": write_gbs,
        "copy_gbs": copy_gbs,
        "fp32_tflops": fp32_tflops,
        "fp16_tflops": fp16_tflops,
        "fp16_min16_tflops": fp16_min16_tflops,
        "fp16_min16_note": "min16float cs_6_0 without -enable-16bit-types; the DXIL "
                           "may promote to FP32, so this only reports whether the "
                           "driver accepted the relaxed half type and at what rate",
        "dot2_f16_tflops": dot2_tflops,
        "dot4_i8_tops": dot4_tops,
        # Pass1 is FP32 FMA dominated, so the FP32 peak is the compute ceiling.
        "pass1_compute_ceiling_tflops": fp32_tflops,
        "nominal": NOMINAL,
        "fraction_of_nominal_pct": {
            "bandwidth": 100.0 * measured_bw / NOMINAL["fast_gbs"] if measured_bw else 0.0,
            "fp32": 100.0 * fp32_tflops / NOMINAL["fp32_tflops"] if fp32_tflops else 0.0,
            "fp16": 100.0 * fp16_tflops / NOMINAL["fp16_tflops"] if fp16_tflops else 0.0,
            "int8": 100.0 * dot4_tops / NOMINAL["int8_tops"] if dot4_tops else 0.0,
        },
    }
    points = []
    for point in sweep.get("points", []):
        entry = dict(point)
        entry.update(pass1_rates(point["n_queries"], point["rows"],
                                 point["gpu_seconds"], measured_bw or 0.0,
                                 fp32_tflops or 0.0))
        entry["bandwidth_reference_gbs"] = measured_bw
        entry["bandwidth_verified"] = stream_verified
        points.append(entry)
    return {
        "schema": "xbox-ceiling-result-v1",
        "recorded_utc": gpupeak.get("recorded_utc", ""),
        "partial": bool(warnings),
        "warnings": warnings,
        "device": device,
        "ceiling_measured": ceiling,
        "stream_checksum": {
            "verified": stream_verified,
            "warnings": list(parsed["warnings"]),
        },
        "dispatch_latency": {
            "empty_dispatch_gpu_us": (latency["empty_dispatch_gpu"] or {}).get(
                "gpu_seconds", 0.0) * 1e6,
            "empty_dispatch_batch_us": (latency["empty_dispatch_batch"] or {}).get(
                "gpu_seconds", 0.0) * 1e6,
            "empty_submit_fence_cpu_us": next(
                (r.get("cpu_seconds", 0.0) * 1e6 for r in rows
                 if r.get("probe") == "empty_submit_fence_cpu" and r.get("variant") == "p50"),
                0.0),
            "empty_readback_cpu_us": next(
                (r.get("cpu_seconds", 0.0) * 1e6 for r in rows
                 if r.get("probe") == "empty_readback_cpu" and r.get("variant") == "p50"),
                0.0),
        },
        "clocks": {
            "timestamp_frequency": device.get("timestamp_frequency", 0),
            "timestamp_us_per_tick": device.get("timestamp_us_per_tick", 0.0),
            "derived_shader_clock_mhz": derived_clock,
            "derived_from_fp32_tflops": fp32_tflops,
            "assumed_dual_cu": NOMINAL["dual_cu"],
            "assumed_flops_per_dual_cu_clock": FLOP_PER_DUAL_CU_CLOCK,
            "nominal_clock_mhz": NOMINAL_CLOCK_MHZ,
            "derived_shader_clock_note": "derived from the measured FP32 peak divided by "
                                         "52 dual-CUs x 128 FLOP/cycle; assumes the FP32 "
                                         "probe is issue-bound at 64 lanes per CU. Not a "
                                         "measured clock.",
        },
        "compute_units": cu,
        "pass1_points": points,
        "pass1_model": {
            "flops_per_point": "2 * 2 * nq * n_head * rows * head_dim",
            "kv_bytes": "records(rows//128 groups) + stage + 128-row exact tail",
            "issued_bytes_upper_bound": "kv_bytes * nq (no KV reuse across queries)",
            "note": "unique KV bytes are a lower bound; the issued-upper column assumes "
                    "each query re-reads every row. The truth lies between them and is "
                    "reported as the measured gbs range, never as a single invented number.",
        },
        "sweep": sweep,
    }
