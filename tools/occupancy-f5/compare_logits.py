#!/usr/bin/env python3
"""Compare full-vocabulary, teacher-forced captures using FP64 KL per token."""

import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import re
import sys

_CACHE_LAYOUT_FIELDS = frozenset({
    "p", "kv_k", "kv_v", "overflow_k", "overflow_v",
    "cache_type_k", "cache_type_v", "remote_layers",
})


def percentile99(values):
    return sorted(values)[max(0, math.ceil(0.99 * len(values)) - 1)]


def distribution_metrics(reference, candidate, target):
    if not reference or len(reference) != len(candidate):
        raise ValueError("vocabulary size mismatch")
    if not all(math.isfinite(x) for x in reference + candidate):
        raise ValueError("nonfinite logits")
    def log_normalizer(logits):
        maximum = max(logits)
        return maximum + math.log(math.fsum(math.exp(x - maximum) for x in logits))
    ref_lse, cand_lse = log_normalizer(reference), log_normalizer(candidate)
    # Differences of log-softmax values avoid log(0) when probabilities underflow.
    kl = math.fsum(math.exp(x - ref_lse) * ((x - ref_lse) - (y - cand_lse))
                   for x, y in zip(reference, candidate))
    return {"kl": kl, "reference_nll": ref_lse - reference[target],
            "candidate_nll": cand_lse - candidate[target],
            "max_logp_difference": max(abs((x-ref_lse) - (y-cand_lse))
                                       for x, y in zip(reference, candidate))}


def capture_index(directory):
    found = {}
    for path in Path(directory).glob("*.logits"):
        match = re.search(r"\.n(\d+)\.logits$", path.name)
        if match:
            position = int(match.group(1))
            if position in found:
                raise ValueError(f"duplicate capture at {position}")
            found[position] = path.with_suffix("")
    if not found:
        raise ValueError("no full-vocabulary captures")
    return found


def read_array(path, kind):
    result = array.array(kind)
    raw = path.read_bytes()
    if len(raw) % result.itemsize:
        raise ValueError(f"truncated capture: {path}")
    result.frombytes(raw)
    if sys.byteorder != "little":
        result.byteswap()
    return list(result)


def _config_core_context(config_core):
    if not isinstance(config_core, str):
        raise ValueError("config_core must be a string to allow context-capacity normalization")
    pattern = re.compile(r"(?P<prefix>(?:^|;)\s*n_ctx\s*=\s*)(?P<value>\d+)(?=\s*(?:;|$))")
    matches = list(pattern.finditer(config_core))
    if len(matches) != 1:
        raise ValueError("config_core must contain exactly one n_ctx field for capacity normalization")
    return int(matches[0].group("value")), pattern.sub(r"\g<prefix><N_CTX>", config_core)


def _parse_config_core(config_core):
    if not isinstance(config_core, str) or not config_core:
        raise ValueError("config_core must be a nonempty semicolon-delimited string")
    fields = {}
    for entry in config_core.split(";"):
        if "=" not in entry:
            raise ValueError(f"malformed config_core field: {entry!r}")
        key, value = entry.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or not value:
            raise ValueError(f"malformed config_core field: {entry!r}")
        if key in fields:
            raise ValueError(f"duplicate config_core field: {key}")
        fields[key] = value
    missing = sorted(_CACHE_LAYOUT_FIELDS - fields.keys())
    if missing:
        raise ValueError("config_core missing cache-layout fields: " + ", ".join(missing))
    if "n_ctx" not in fields:
        raise ValueError("config_core must contain n_ctx")
    return fields


def _cache_layout_config_difference(reference, candidate, reference_n_ctx,
                                    candidate_n_ctx, allow_context_capacity_difference):
    ref_fields = _parse_config_core(reference)
    cand_fields = _parse_config_core(candidate)
    if ref_fields.keys() != cand_fields.keys():
        raise ValueError("provenance mismatch: config_core field set")
    try:
        ref_ctx = int(ref_fields["n_ctx"], 10)
        cand_ctx = int(cand_fields["n_ctx"], 10)
    except ValueError as exc:
        raise ValueError("config_core n_ctx must be an integer") from exc
    if ref_ctx != reference_n_ctx or cand_ctx != candidate_n_ctx:
        raise ValueError("config_core n_ctx disagrees with capture provenance")
    if not allow_context_capacity_difference and (
            ref_ctx != cand_ctx or reference_n_ctx != candidate_n_ctx):
        raise ValueError("provenance mismatch: n_ctx")
    ignored = set(_CACHE_LAYOUT_FIELDS)
    if allow_context_capacity_difference:
        ignored.add("n_ctx")
    mismatched = [key for key in sorted(ref_fields.keys() - ignored)
                  if ref_fields[key] != cand_fields[key]]
    if mismatched:
        raise ValueError("provenance mismatch: config_core excluding allowed cache-layout fields: " +
                         ", ".join(mismatched))
    differences = {
        key: {"reference": ref_fields[key], "candidate": cand_fields[key]}
        for key in sorted(_CACHE_LAYOUT_FIELDS)
        if ref_fields[key] != cand_fields[key]
    }
    return differences


def compare(reference_dir, candidate_dir, teacher_tokens, minimum=32,
            allow_context_capacity_difference=False, metrics_only=False,
            allow_cache_layout_difference=False):
    if allow_context_capacity_difference and not metrics_only:
        raise ValueError("--allow-context-capacity-difference requires --metrics-only")
    if allow_cache_layout_difference and not metrics_only:
        raise ValueError("--allow-cache-layout-difference requires --metrics-only")
    reference, candidate = capture_index(reference_dir), capture_index(candidate_dir)
    if set(reference) != set(candidate):
        raise ValueError("checkpoint sets differ; incomplete capture cannot be discarded")
    positions = sorted(reference)
    if len(positions) < minimum:
        raise ValueError(f"need {minimum} teacher-forced positions, got {len(positions)}")
    if any(b != a + 1 for a, b in zip(positions, positions[1:])):
        raise ValueError("teacher-forced positions must be consecutive")
    rows = []
    context_capacities = None
    capacity_difference = False
    cache_layout_differences = {}
    for position in positions:
        ra, ca = reference[position], candidate[position]
        rt = read_array(ra.with_suffix(ra.suffix + ".tokens"), "i")
        ct = read_array(ca.with_suffix(ca.suffix + ".tokens"), "i")
        if position >= len(teacher_tokens) or rt != ct or rt != teacher_tokens[:position]:
            raise ValueError(f"teacher-forced prefix mismatch at {position}")
        rm = json.loads(ra.with_suffix(ra.suffix + ".json").read_text())
        cm = json.loads(ca.with_suffix(ca.suffix + ".json").read_text())
        for key in ("model_fingerprint", "n_vocab", "n_params"):
            if not rm.get(key) or rm.get(key) != cm.get(key):
                raise ValueError(f"provenance mismatch: {key} at {position}")
        if not rm.get("config_core") or not cm.get("config_core"):
            raise ValueError(f"provenance mismatch: config_core at {position}")
        if type(rm.get("n_ctx")) is not int or type(cm.get("n_ctx")) is not int:
            raise ValueError(f"invalid context capacity provenance at {position}")
        if context_capacities is None:
            context_capacities = {"reference": rm["n_ctx"], "candidate": cm["n_ctx"]}
        elif context_capacities != {"reference": rm["n_ctx"], "candidate": cm["n_ctx"]}:
            raise ValueError(f"context capacity changed within capture set at {position}")
        if allow_cache_layout_difference:
            differences = _cache_layout_config_difference(
                rm["config_core"], cm["config_core"], rm["n_ctx"], cm["n_ctx"],
                allow_context_capacity_difference)
            if position == positions[0]:
                cache_layout_differences = differences
            elif differences != cache_layout_differences:
                raise ValueError(f"cache-layout config changed within capture set at {position}")
            capacity_difference = capacity_difference or (rm["n_ctx"] != cm["n_ctx"])
        elif allow_context_capacity_difference:
            ref_config_ctx, ref_config_normalized = _config_core_context(rm["config_core"])
            cand_config_ctx, cand_config_normalized = _config_core_context(cm["config_core"])
            if ref_config_ctx != rm["n_ctx"] or cand_config_ctx != cm["n_ctx"]:
                raise ValueError(f"config_core n_ctx disagrees with capture provenance at {position}")
            if ref_config_normalized != cand_config_normalized:
                raise ValueError(f"provenance mismatch: config_core excluding n_ctx at {position}")
            capacity_difference = capacity_difference or (rm["n_ctx"] != cm["n_ctx"])
        elif rm["config_core"] != cm["config_core"] or rm["n_ctx"] != cm["n_ctx"]:
            key = "config_core" if rm["config_core"] != cm["config_core"] else "n_ctx"
            raise ValueError(f"provenance mismatch: {key} at {position}")
        rv = read_array(ra.with_suffix(ra.suffix + ".logits"), "f")
        cv = read_array(ca.with_suffix(ca.suffix + ".logits"), "f")
        if len(rv) != rm["n_vocab"] or len(rt) != position:
            raise ValueError(f"capture dimensions disagree with provenance at {position}")
        target = teacher_tokens[position]
        if not 0 <= target < len(rv):
            raise ValueError("teacher token outside vocabulary")
        rows.append({"position": position, "teacher_token": target,
                     "prefix_sha256": hashlib.sha256(ra.with_suffix(ra.suffix + ".tokens").read_bytes()).hexdigest(),
                     **distribution_metrics(rv, cv, target)})
    kls = [r["kl"] for r in rows]
    ref_nll = math.fsum(r["reference_nll"] for r in rows) / len(rows)
    cand_nll = math.fsum(r["candidate_nll"] for r in rows) / len(rows)
    mean_kl, p99_kl = math.fsum(kls)/len(kls), percentile99(kls)
    structural_pass = mean_kl <= 1e-4 and p99_kl <= 1e-3
    result = {"schema": "occupancy-kl-per-token-v1", "positions": len(rows),
            "mean_kl_nat_per_token": mean_kl, "p99_kl_nat_per_token": p99_kl,
            "reference_mean_nll": ref_nll, "candidate_mean_nll": cand_nll,
            "reference_perplexity": math.exp(ref_nll), "candidate_perplexity": math.exp(cand_nll),
            "context_capacities": context_capacities,
            "context_capacity_difference_allowed": capacity_difference,
            "quality_comparison_scope": (
                "metrics_only_cache_layout_and_context_difference"
                if allow_cache_layout_difference and allow_context_capacity_difference else
                "metrics_only_cache_layout_difference" if allow_cache_layout_difference else
                "metrics_only_context_capacity_difference" if capacity_difference else
                "structural_equivalence"),
            "structural_gate_pass": None if capacity_difference or allow_cache_layout_difference else structural_pass,
            "structural_gate_evaluated": False if capacity_difference or allow_cache_layout_difference else None,
            "rows": rows}
    if capacity_difference:
        result["structural_gate_suppressed_reason"] = "context capacities differ; quality metrics are non-structural"
    if allow_cache_layout_difference:
        result["cache_layout_difference_allowed"] = True
        result["cache_layout_allowed_fields"] = sorted(_CACHE_LAYOUT_FIELDS)
        result["cache_layout_differences"] = cache_layout_differences
        result["structural_gate_pass"] = None
        result["structural_gate_evaluated"] = False
        result["structural_gate_suppressed_reason"] = (
            "cache-layout normalization is quality-only; structural equivalence was not evaluated")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--teacher-tokens", required=True, help="JSON array of ground-truth token ids")
    parser.add_argument("--minimum", type=int, default=32)
    parser.add_argument("--metrics-only", action="store_true", help="report quality measurements without applying structural limits")
    parser.add_argument("--allow-context-capacity-difference", action="store_true",
                        help="metrics-only quality comparison; normalize only config_core.n_ctx")
    parser.add_argument("--allow-cache-layout-difference", action="store_true",
                        help="metrics-only quality comparison; normalize only declared KV placement/format fields")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.allow_context_capacity_difference and not args.metrics_only:
            raise ValueError("--allow-context-capacity-difference requires --metrics-only")
        if args.allow_cache_layout_difference and not args.metrics_only:
            raise ValueError("--allow-cache-layout-difference requires --metrics-only")
        tokens = json.loads(Path(args.teacher_tokens).read_text())
        if not isinstance(tokens, list) or any(type(t) is not int for t in tokens):
            raise ValueError("teacher tokens must be integer ids")
        if args.minimum < 32:
            raise ValueError("acceptance requires at least 32 teacher-forced positions")
        result = compare(args.reference, args.candidate, tokens, args.minimum,
                         args.allow_context_capacity_difference, args.metrics_only,
                         args.allow_cache_layout_difference)
        if result["structural_gate_pass"] is not None:
            result["structural_gate_evaluated"] = not args.metrics_only
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            Path(args.output).write_text(payload)
        print(payload, end="")
        return 0 if args.metrics_only or result["structural_gate_pass"] else 1
    except (ValueError, OSError, OverflowError) as error:
        print(json.dumps({"verdict": "reject", "reason": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
