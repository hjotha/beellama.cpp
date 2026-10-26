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

# RDNA2 VALU FMA latency in clocks, used only for the *implied* clock.
RDNA2_FMA_LATENCY_CLOCKS = 4.0


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


def implied_clock_mhz(seconds_per_fma: float,
                      latency_clocks: float = RDNA2_FMA_LATENCY_CLOCKS) -> float:
    """Clock implied by a dependent-FMA chain; derived, never measured."""
    if seconds_per_fma <= 0:
        return 0.0
    return latency_clocks / seconds_per_fma / 1e6


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

    Raises ValueError on anything that would silently produce a wrong
    roofline: missing schema, no device timestamp frequency, or a stream rate
    without its checksum evidence.
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
    for probe in ("read", "write", "copy"):
        timed = [r for r in by_probe.get(probe, []) if r.get("gpu_valid") and r.get("runs")]
        if not timed:
            raise ValueError("missing %s stream rows" % probe)
    evidence = [r for r in by_probe.get("read", []) + by_probe.get("copy", [])
                if r.get("checksum_checked")]
    if not evidence or not all(r.get("checksum_ok") for r in evidence):
        raise ValueError("stream checksum evidence missing or failed")
    return {"device": device, "rows": rows, "by_probe": by_probe}


def best(rows: Iterable[Dict], probe: str) -> Dict:
    """Fastest valid row of a probe (largest rate = smallest gpu_seconds)."""
    candidates = [r for r in rows if r.get("probe") == probe and r.get("gpu_valid")
                  and r.get("gpu_seconds", 0) > 0]
    if not candidates:
        return {}
    return min(candidates, key=lambda r: r["gpu_seconds"])


def stream_rate_gbs(row: Dict, traffic_multiplier: float = 1.0) -> float:
    if not row or not row.get("gpu_seconds"):
        return 0.0
    return row["bytes"] * traffic_multiplier / 1e9 / row["gpu_seconds"]


def build_result(gpupeak: Dict, sweep: Dict) -> Dict:
    """Assemble xbox-ceiling-result.json: measured ceilings + pass1 roofline.

    `gpupeak` is the raw probe document; `sweep` is the pass1 driver output
    ({"points": [{"n_queries", "rows", "gpu_seconds", ...}]}).
    """
    parsed = parse_gpupeak_json(gpupeak)
    rows = parsed["rows"]
    device = parsed["device"]
    read_gbs = stream_rate_gbs(best(rows, "read"))
    write_gbs = stream_rate_gbs(best(rows, "write"))
    copy_gbs = stream_rate_gbs(best(rows, "copy"), 2.0)
    measured_bw = max(read_gbs, copy_gbs)
    fp32 = best(rows, "fma_f32")
    fp16 = best(rows, "fma_f16")
    dot2 = best(rows, "dot2_f16")
    dot4 = best(rows, "dot4_i8")
    fp32_tflops = (fp32.get("tflops", 0.0) if fp32 else 0.0)
    fp16_tflops = (fp16.get("tflops", 0.0) if fp16 else 0.0)
    dot2_tflops = (dot2.get("tflops", 0.0) if dot2 else 0.0)
    dot4_tops = (dot4.get("gops", 0.0) if dot4 else 0.0)

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
    implied_clock = (gpupeak.get("derived") or {}).get("implied_clock_mhz", 0.0)

    ceiling = {
        "bandwidth_gbs": measured_bw,
        "read_gbs": read_gbs,
        "write_gbs": write_gbs,
        "copy_gbs": copy_gbs,
        "fp32_tflops": fp32_tflops,
        "fp16_tflops": fp16_tflops,
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
        points.append(entry)
    return {
        "schema": "xbox-ceiling-result-v1",
        "recorded_utc": gpupeak.get("recorded_utc", ""),
        "device": device,
        "ceiling_measured": ceiling,
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
            "implied_shader_clock_mhz": implied_clock,
            "implied_shader_clock_note": "derived from a dependent FP32 FMA chain "
                                         "assuming the documented RDNA2 4-clock VALU "
                                         "latency; not a measured clock",
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
