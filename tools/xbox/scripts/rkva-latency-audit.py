#!/usr/bin/env python3
"""RKVA latency audit: trace attribution, empty-RTT probe, local link facts.

This tool only collects evidence. It does not optimise anything, does not pick a
fix, and never touches the GPU, the Xbox, the app flags or the network settings.

Three independent modes (all offline unless --execute is passed):

  1. Trace attribution (default, read-only file access)
       rkva-latency-audit.py --trace TRACE.jsonl \\
           --min-position 130560 --max-position 160000 --out report.json

     Attributes every decode event (nq == 1) inside the position window to its
     token/layer/correlation when the trace carries those fields, reports
     count/min/p50/p95/total, per-token totals, parent/child overlap and the
     unaccounted wall time. Parent and child spans are never summed together;
     rpc_total and job_finish are containers, not additive work. Missing or
     duplicated correlation ids become an explicit "unknown" scope instead of a
     guessed join. CPU engine numbers are compared with GPU numbers only when
     both sample counts and both windows are congruent.

  2. Empty round trip (opt-in, needs an idle console)
       rkva-latency-audit.py --execute --host 192.168.1.200:50054 --empty-rtt

     Opens one TCP connection and sends RKVA_PING only (op 9, empty payload,
     sessionless, no HELLO, no session, no GPU engine). Timings are wall clock
     around write + read of the fixed 32-byte request / 56-byte response.
     The server frees its engine when a new client connects, so the probe
     refuses to run while any llama/RKVA workload is alive (see the process
     guard) and never retries on its own.

  3. Local link facts (read-only)
       rkva-latency-audit.py --network-info [--host 192.168.1.200:50054]

     Reads the route/interface, sysfs link facts and ethtool show commands.
     No sudo, no writes, no IRQ/MTU/offload/sysctl change, and it never
     contacts the target. Missing privilege is reported as "unavailable",
     never as a fake zero. These are local link facts only: they do not prove
     causality for RKVA latency.

Safety rules enforced here:
  * No socket is opened unless --execute --empty-rtt is passed explicitly.
  * No subprocess runs unless --network-info is requested, and only for
    allowlisted read-only ethtool/ip queries.
  * No file is created unless --out is given.
  * Busy-process detection matches the executable/script name, never arbitrary
    argv strings, so this CLI's own arguments cannot trip it.

Read-only evidence only. Any future engine/overhead change is a separate step
that depends on the three KV formats and QA, reviewed after this audit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import struct
import subprocess
import sys
import time
from collections import defaultdict

SCHEMA = "rkva-latency-audit-v1"
CONTEXT_CAP = 160000
CONTEXT_CAP_LIMIT = 160000

# RKVA wire protocol (ggml/src/ggml-remote-attn/remote-attn-protocol.h).
RKVA_MAGIC = 0x41564B52
RKVA_VERSION = 2
RKVA_VERSION_LEGACY = 1
RKVA_PING = 9
RKVA_GET_STATS = 10
RKVA_SESSION_NONE = 0xFFFFFFFF
REQUEST_FMT = "<IHHQIIII"
RESPONSE_FMT = "<IHHQiIQQQQ"
REQUEST_BYTES = struct.calcsize(REQUEST_FMT)   # 32
RESPONSE_BYTES = struct.calcsize(RESPONSE_FMT)  # 56
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024

MAX_RTT_TIMEOUT_S = 3.0
MAX_RTT_REPS = 1000
# GET_STATS is the only control op with a body; cap it before reading.
MAX_CONTROL_PAYLOAD = 1 << 20

BUSY_EXACT_NAMES = frozenset({
    "llama-server", "llama-cli", "llama-bench", "llama-perplexity",
    "llama-speculative", "rkva-tester", "rkva-bench", "rkva-probe",
})
BUSY_NAME_TOKENS = ("bench", "test", "probe", "sweep")
BUSY_NAME_PREFIXES = ("bench-rkva", "run-rkva")

# Phases that are containers: their duration is reported, never added up.
CONTAINER_PHASES = frozenset({"job_finish", "rpc_total", "out_decode", "envelope"})
# Wait-like spans are reported on their own wall range, never inferred from a
# container such as rpc_total.
WAIT_PHASES = frozenset({"rpc_header_wait", "range_wait", "job_prepare", "rpc_queue"})
# Optional trace metadata that identifies the origin of a cycle-id sequence.
# Cycle ids are only unique inside one origin; a composite key keeps independent
# processes/arms from looking like a duplicate.
ORIGIN_FIELDS = ("pid", "process", "origin", "arm", "run_id", "trace_id",
                 "instance", "host")

READONLY_ETHTOOL_FLAGS = frozenset({"-i", "-k", "-c", "-S", "-g", "-a", "-T", "--version"})
READONLY_IP_ACTIONS = frozenset({"route", "link", "addr", "address", "neigh"})
IP_MUTATING_WORDS = frozenset({
    "add", "del", "delete", "set", "change", "replace", "flush", "up", "down",
})
NET_CMDS = frozenset({"ethtool", "ip"})


class TraceFormatError(ValueError):
    """Input trace violates the invariants this audit relies on."""


class ProbeError(RuntimeError):
    """The empty-RTT probe could not produce trustworthy samples."""


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #

def percentile(ordered, q):
    """Nearest-rank percentile over an already sorted sequence."""
    if not ordered:
        return None
    if q <= 0.0:
        return float(ordered[0])
    if q >= 1.0:
        return float(ordered[-1])
    idx = math.ceil(q * len(ordered)) - 1
    idx = max(0, min(len(ordered) - 1, idx))
    return float(ordered[idx])


def summarize(values):
    """count/min/p50/p95/max/total/mean over finite numbers."""
    vals = [float(v) for v in values]
    for v in vals:
        if not math.isfinite(v):
            raise ValueError("non-finite sample")
    if not vals:
        return {"count": 0, "min": None, "p50": None, "p95": None,
                "max": None, "total": None, "mean": None}
    ordered = sorted(vals)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
        "total": math.fsum(ordered),
        "mean": math.fsum(ordered) / len(ordered),
    }


def merge_intervals(intervals):
    """Merge [begin, end) half-open intervals into a sorted disjoint list."""
    ordered = sorted((float(b), float(e)) for b, e in intervals if e > b)
    merged = []
    for begin, end in ordered:
        if merged and begin <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((begin, end))
    return merged


def union_us(intervals):
    return math.fsum(e - b for b, e in merge_intervals(intervals))


def union_within(parent, intervals):
    """Union of child intervals clipped to the parent span."""
    begin, end = parent
    clipped = [(max(begin, b), min(end, e)) for b, e in intervals if e > b]
    return union_us(clipped)


# --------------------------------------------------------------------------- #
# Trace parsing and attribution
# --------------------------------------------------------------------------- #

def _as_int(value, field, line_no):
    if isinstance(value, bool) or value is None:
        raise TraceFormatError(f"line {line_no}: {field} is not an integer")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise TraceFormatError(f"line {line_no}: {field} is empty")
        try:
            return int(text, 10)
        except ValueError as exc:
            raise TraceFormatError(f"line {line_no}: {field}={value!r} is not an integer") from exc
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TraceFormatError(f"line {line_no}: {field} is not finite")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TraceFormatError(f"line {line_no}: {field}={value!r} is not an integer") from exc


def _as_duration(value, field, line_no):
    if isinstance(value, bool) or value is None:
        raise TraceFormatError(f"line {line_no}: {field} is missing")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TraceFormatError(f"line {line_no}: {field}={value!r} is not numeric") from exc
    if not math.isfinite(number):
        raise TraceFormatError(f"line {line_no}: {field} is not finite")
    return number


class Span:
    __slots__ = ("phase", "begin", "end", "cycle", "gpu_us", "depth", "parent", "self_us")

    def __init__(self, phase, begin, end, cycle, gpu_us):
        self.phase = phase
        self.begin = begin
        self.end = end
        self.cycle = cycle
        self.gpu_us = gpu_us
        self.depth = 0
        self.parent = None
        self.self_us = float(end - begin)

    @property
    def duration(self):
        return float(self.end - self.begin)


def _contains(outer, inner):
    """Strict wall-clock containment; identical spans stay siblings."""
    return (outer.begin <= inner.begin and outer.end >= inner.end
            and (outer.begin < inner.begin or outer.end > inner.end))


def _nest(spans):
    """Assign depth and immediate parent by strict containment."""
    for span in spans:
        span.depth = 0
        span.parent = None
        span.self_us = span.duration
    for span in spans:
        containers = [other for other in spans if other is not span and _contains(other, span)]
        span.depth = len(containers)
        if containers:
            span.parent = min(containers, key=lambda other: (other.duration, other.begin))
    for span in spans:
        children = [other for other in spans if other.parent is span]
        span.self_us = span.duration - union_within(
            (span.begin, span.end), [(c.begin, c.end) for c in children])


class TraceAnalysis:
    def __init__(self, path, min_position, max_position, context_cap=CONTEXT_CAP):
        if min_position < 0:
            raise TraceFormatError("--min-position must be >= 0")
        if max_position < min_position:
            raise TraceFormatError("--max-position must be >= --min-position")
        if context_cap > CONTEXT_CAP_LIMIT:
            raise TraceFormatError(
                f"--context-cap {context_cap} exceeds the audited cap {CONTEXT_CAP_LIMIT}")
        if max_position > context_cap:
            raise TraceFormatError(
                f"--max-position {max_position} exceeds --context-cap {context_cap}; "
                "the audited tier stops at 160000")
        self.path = path
        self.min_position = min_position
        self.max_position = max_position
        self.context_cap = context_cap
        self.groups = defaultdict(list)          # (session, graph_run, layer) -> spans
        self.group_positions = {}                # group key -> (first_pos, last_pos, model_layer)
        self.tokens = defaultdict(list)          # (session, graph_run, first_pos) -> spans
        self.token_layers = defaultdict(set)     # token key -> layer ids
        self.group_origins = {}                  # group key -> origin tuple
        self.group_windows = {}                  # group key -> (min begin, max end)
        self.group_session = {}                  # group key -> session or None
        self.group_origin_mixed = set()
        self.origin_fields_seen = set()
        self.session_field_seen = False
        self.conflicted_groups = set()
        self.unattributed_spans = []             # spans without usable identity
        self.counts = {
            "lines": 0,
            "malformed_lines": 0,
            "decode_events": 0,
            "events_outside_window": 0,
            "events_without_position": 0,
            "events_without_identity": 0,
            "identity_conflicts": 0,
            "events_unknown_phase": 0,
        }
        self.phase_values = defaultdict(list)
        self.phase_self = defaultdict(list)
        self.phase_gpu = defaultdict(list)

    def load(self):
        with open(self.path, "r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                self.counts["lines"] = line_no
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    self.counts["malformed_lines"] += 1
                    continue
                if not isinstance(event, dict):
                    self.counts["malformed_lines"] += 1
                    continue
                if _as_int(event.get("nq", 0), "nq", line_no) != 1:
                    continue
                self._ingest(event, line_no)
        self._tokenize()
        return self

    def _ingest(self, event, line_no):
        phase = str(event.get("phase") or "")
        if not phase:
            self.counts["events_unknown_phase"] += 1
        if event.get("first_pos") is None:
            self.counts["events_without_position"] += 1
            return
        first_pos = _as_int(event["first_pos"], "first_pos", line_no)
        last_pos = _as_int(event.get("last_pos", first_pos), "last_pos", line_no)
        if first_pos < self.min_position or first_pos > self.max_position:
            self.counts["events_outside_window"] += 1
            return
        begin = _as_int(event.get("begin_us"), "begin_us", line_no)
        end = _as_int(event.get("end_us"), "end_us", line_no)
        if end < begin:
            raise TraceFormatError(
                f"line {line_no}: end_us {end} precedes begin_us {begin} for phase {phase!r}")
        cycle = _as_int(event.get("cycle", 0), "cycle", line_no)
        gpu_us = 0.0
        if event.get("gpu_us") is not None:
            gpu_us = _as_duration(event["gpu_us"], "gpu_us", line_no)
            if gpu_us < 0:
                raise TraceFormatError(f"line {line_no}: negative gpu_us for phase {phase!r}")
        graph_run = _as_int(event.get("graph_run", 0), "graph_run", line_no)
        layer = _as_int(event.get("layer", -1), "layer", line_no)
        session = _as_int(event.get("session", 0), "session", line_no)
        session_present = "session" in event
        model_layer = _as_int(event.get("model_layer", -1), "model_layer", line_no)
        origin_parts = []
        for name in ORIGIN_FIELDS:
            value = event.get(name)
            if value in (None, "", 0):
                continue
            origin_parts.append((name, str(value)))
            self.origin_fields_seen.add(name)
        origin = tuple(sorted(origin_parts))
        if session_present:
            self.session_field_seen = True

        self.counts["decode_events"] += 1
        span = Span(phase, begin, end, cycle, gpu_us)
        self.phase_values[phase].append(span.duration)
        if gpu_us:
            self.phase_gpu[phase].append(gpu_us)
        if graph_run <= 0 or layer < 0:
            self.counts["events_without_identity"] += 1
            self.unattributed_spans.append(span)
            return
        key = (session, graph_run, layer)
        known = self.group_positions.get(key)
        if key in self.conflicted_groups or (known and
                                             (known[0] != first_pos or known[1] != last_pos)):
            # A graph_run/layer pair may not describe two different positions.
            # Refuse to merge two model states instead of averaging them.
            if key not in self.conflicted_groups:
                self.conflicted_groups.add(key)
                self.counts["identity_conflicts"] += 1
                for stale in self.groups.pop(key, []):
                    self.unattributed_spans.append(stale)
                self.group_positions.pop(key, None)
            self.counts["events_without_identity"] += 1
            self.unattributed_spans.append(span)
            return
        self.group_positions[key] = (first_pos, last_pos, model_layer)
        if key in self.group_origins and self.group_origins[key] != origin:
            # Conflicting origin metadata inside one layer group is not a
            # correlation we may guess around.
            self.group_origin_mixed.add(key)
        else:
            self.group_origins[key] = origin
        current_session = session if session_present else None
        if key not in self.group_session:
            self.group_session[key] = current_session
        else:
            previous = self.group_session[key]
            if previous is None and current_session is not None:
                self.group_session[key] = current_session
            elif (previous is not None and current_session is not None
                  and previous != current_session):
                self.group_origin_mixed.add(key)
        window = self.group_windows.get(key)
        self.group_windows[key] = ((min(window[0], begin), max(window[1], end))
                                   if window else (begin, end))
        self.groups[key].append(span)
        self.token_layers[(session, graph_run, first_pos)].add(layer)
        self.tokens[(session, graph_run, first_pos)].append(span)

    def _tokenize(self):
        for spans in self.groups.values():
            _nest(spans)
            for span in spans:
                self.phase_self[span.phase].append(span.self_us)
        self._nest_token_totals()

    def _nest_token_totals(self):
        self.token_totals = {}
        for key, spans in self.tokens.items():
            envelope_begin = min(span.begin for span in spans)
            envelope_end = max(span.end for span in spans)
            envelope = float(envelope_end - envelope_begin)
            # depth is per layer group, so depth-0 spans are the frontier of
            # every layer and their union is the additive per-token work.
            root_spans = [span for span in spans if span.depth == 0]
            attributed = union_us([(s.begin, s.end) for s in root_spans])
            rpc_total = math.fsum(s.duration for s in spans if s.phase == "rpc_total")
            peer_gpu = math.fsum(s.gpu_us for s in spans if s.phase == "rpc_total")
            self.token_totals[key] = {
                "envelope_us": envelope,
                "attributed_us": attributed,
                "gap_us": envelope - attributed,
                "rpc_total_us": rpc_total,
                "peer_gpu_us": peer_gpu,
                "layers": len(self.token_layers.get(key, ())),
                "root_spans": len(root_spans),
            }

    # -- derived reports ---------------------------------------------------- #

    def phase_report(self):
        phases = {}
        for phase in sorted(self.phase_values):
            stats = summarize(self.phase_values[phase])
            self_values = self.phase_self.get(phase, [])
            residual_us = math.fsum(self_values)
            stats["container"] = phase in CONTAINER_PHASES
            stats["wait_span"] = phase in WAIT_PHASES
            if stats["container"]:
                # A container (job_finish/rpc_total/out_decode) is not a wait:
                # its uncovered remainder is a residual, never "exposed" work.
                stats["residual_total_us"] = residual_us
                stats["exposed_total_us"] = None
            else:
                stats["residual_total_us"] = None
                stats["exposed_total_us"] = residual_us
            stats["overlapped_total_us"] = stats["total"] - residual_us
            gpu = self.phase_gpu.get(phase, [])
            stats["gpu_us"] = summarize(gpu) if gpu else None
            stats["per_token_us"] = None
            phases[phase] = stats
        tokens = max(1, len(self.token_totals))
        for stats in phases.values():
            stats["per_token_us"] = stats["total"] / tokens
        return phases

    def parent_child_report(self):
        pairs = defaultdict(lambda: {"count": 0, "overlap_us": 0.0, "parent_total_us": 0.0})
        for spans in self.groups.values():
            for span in spans:
                if span.parent is None:
                    continue
                key = (span.parent.phase, span.phase)
                children = [other for other in spans if other.parent is span.parent]
                clipped = union_within((span.parent.begin, span.parent.end),
                                       [(c.begin, c.end) for c in children])
                entry = pairs[key]
                entry["count"] += 1
                entry["overlap_us"] += clipped
                entry["parent_total_us"] += span.parent.duration
        report = {}
        for (parent, child), entry in sorted(pairs.items()):
            ratio = entry["overlap_us"] / entry["parent_total_us"] if entry["parent_total_us"] else None
            report.setdefault(parent, {})[child] = {
                "count": entry["count"],
                "overlap_us": entry["overlap_us"],
                "overlap_ratio": ratio,
            }
        return report

    def containment_report(self):
        parents = defaultdict(set)
        children = defaultdict(set)
        for spans in self.groups.values():
            for span in spans:
                if span.parent is not None:
                    parents[span.phase].add(span.parent.phase)
                    children[span.parent.phase].add(span.phase)
        return {
            "parents": {phase: sorted(value) for phase, value in sorted(parents.items())},
            "children": {phase: sorted(value) for phase, value in sorted(children.items())},
        }

    def token_report(self):
        totals = list(self.token_totals.values())
        layers = [entry["layers"] for entry in totals]
        return {
            "tokens": len(totals),
            "layer_groups": len(self.groups),
            "layers_per_token": {
                "min": min(layers) if layers else 0,
                "max": max(layers) if layers else 0,
            },
            "envelope_ms": self._ms(entry["envelope_us"] for entry in totals),
            "attributed_ms": self._ms(entry["attributed_us"] for entry in totals),
            "unattributed_gap_ms": self._ms(entry["gap_us"] for entry in totals),
            "rpc_total_ms_container": self._ms(entry["rpc_total_us"] for entry in totals),
            "peer_gpu_ms": self._ms(entry["peer_gpu_us"] for entry in totals),
        }

    @staticmethod
    def _ms(values):
        stats = summarize(values)
        for key in ("min", "p50", "p95", "max", "total", "mean"):
            if stats[key] is not None:
                stats[key] = stats[key] / 1000.0
        return stats

    def _windows_overlap(self, left, right):
        return left[0] < right[1] and right[0] < left[1]

    def correlation_report(self):
        """Resolve cycle ids only inside a proven origin.

        Cycle ids are local to one origin (process/session/arm). The composite
        key includes the session and any origin metadata the trace carries.
        Reuse in independent windows is reported as unknown, not as a failure;
        only an ambiguous overlap or mixed ids inside one group fails.
        """
        composite_groups = defaultdict(set)
        groups_with = groups_without = mixed = 0
        origin_mixed = 0
        for key, spans in self.groups.items():
            cycles = {span.cycle for span in spans if span.cycle > 0}
            if not cycles:
                groups_without += 1
                continue
            if len(cycles) > 1:
                mixed += 1
                continue
            groups_with += 1
            if key in self.group_origin_mixed:
                origin_mixed += 1
                continue
            composite = (self.group_session.get(key), self.group_origins.get(key),
                         next(iter(cycles)))
            composite_groups[composite].add(key)

        duplicated = {composite: keys for composite, keys in composite_groups.items()
                      if len(keys) > 1}
        reused_groups = 0
        ambiguous_groups = 0
        for keys in duplicated.values():
            windows = [self.group_windows[key] for key in keys]
            overlapping = any(self._windows_overlap(windows[i], windows[j])
                              for i in range(len(windows))
                              for j in range(i + 1, len(windows)))
            if overlapping:
                ambiguous_groups += len(keys)
            else:
                reused_groups += len(keys)

        origin_available = self.session_field_seen or bool(self.origin_fields_seen)
        failed = bool(mixed or origin_mixed or ambiguous_groups)
        reasons = []
        if groups_without:
            reasons.append(f"{groups_without} layer groups carry no cycle id")
        if mixed:
            reasons.append(f"{mixed} layer groups mix several cycle ids")
        if origin_mixed:
            reasons.append(f"{origin_mixed} layer groups carry conflicting origin metadata")
        if ambiguous_groups:
            reasons.append(f"{ambiguous_groups} groups share one composite "
                           "(origin, cycle) key inside overlapping windows")
        if reused_groups:
            reasons.append(f"{reused_groups} groups reuse a cycle id in independent "
                           "windows; treated as unknown, not as a failure")
        if not origin_available:
            reasons.append("the trace carries no process/session origin; cycle ids "
                           "cannot be resolved across runs")
        if failed:
            scope = "failed"
        elif groups_without or reused_groups or not origin_available:
            scope = "unknown"
        else:
            scope = "ok"
        return {
            "scope": scope,
            "origin": {
                "available": origin_available,
                "session_field": self.session_field_seen,
                "fields_seen": sorted(self.origin_fields_seen),
                "composite_key": ["session", "origin", "cycle"],
            },
            "groups_with_correlation": groups_with,
            "groups_without_correlation": groups_without,
            "groups_with_mixed_correlation": mixed + origin_mixed,
            "duplicate_ids": len(duplicated),
            "ambiguous_groups": ambiguous_groups,
            "reused_in_independent_windows": reused_groups,
            "reasons": reasons,
            "note": ("A cycle id resolves only when its composite (session, origin, "
                     "cycle) key identifies exactly one layer group. Reuse across "
                     "independent windows is unknown; overlapping reuse is a failure; "
                     "ids are never invented, merged or joined across runs."),
        }

    def report(self):
        phases = self.phase_report()
        tokens = self.token_report()
        correlation = self.correlation_report()
        conflict = self.counts["identity_conflicts"]
        if conflict and not self.groups:
            scope = "partial"
            reasons = [f"{conflict} graph_run/layer pairs described more than one "
                       "position and were rejected"]
        elif not self.groups:
            scope = "unknown"
            reasons = ["no decode event with a usable token/layer identity in the window"]
        elif self.counts["events_without_identity"] or conflict:
            scope = "partial"
            reasons = []
            if self.counts["events_without_identity"]:
                reasons.append(f"{self.counts['events_without_identity']} decode events "
                               "have no graph_run/layer identity")
            if conflict:
                reasons.append(f"{conflict} events reuse a graph_run at another position")
        else:
            scope = "ok"
            reasons = []
        exposed = {
            "definition": ("wall-clock ranges inside each token envelope that no recorded "
                           "span covers; rpc_total is a container and is never used as a wait"),
            "unattributed_gap_ms_per_token": tokens["unattributed_gap_ms"],
            "wait_phases": {
                phase: dict(phases[phase],
                            exposed_per_token_us=phases[phase]["exposed_total_us"] / max(1, len(self.token_totals)))
                for phase in sorted(WAIT_PHASES) if phase in phases
            },
        }
        return {
            "schema": SCHEMA,
            "mode": "trace",
            "scope": scope,
            "scope_reasons": reasons,
            "trace": self.path,
            "window": {
                "min_position": self.min_position,
                "max_position": self.max_position,
                "context_cap": self.context_cap,
                "filter": "decode events with nq == 1 and first_pos inside the window",
            },
            "counts": dict(self.counts, layer_groups=len(self.groups),
                           tokens=len(self.token_totals),
                           duplicate_cycle_events=correlation["ambiguous_groups"]),
            "phases": phases,
            "per_token": tokens,
            "parent_child": self.parent_child_report(),
            "containment": self.containment_report(),
            "exposed_wait": exposed,
            "correlation": correlation,
            "unknown": {
                "events_without_identity": self.counts["events_without_identity"],
                "events_without_position": self.counts["events_without_position"],
                "events_unknown_phase": self.counts["events_unknown_phase"],
                "malformed_lines": self.counts["malformed_lines"],
                "correlation_scope": correlation["scope"],
                "comparison_scope": "not_requested",
            },
            "evidence": [
                {"metric": "phase durations", "class": "measured",
                 "source": "trace begin_us/end_us per event"},
                {"metric": "device GPU compute", "class": "measured",
                 "source": "peer-reported gpu_us, only where the peer emitted it"},
                {"metric": "per-token attributed/overlap", "class": "derived",
                 "source": "union of root spans and strict containment"},
                {"metric": "CPU engine vs GPU", "class": "unknown",
                 "source": "requires congruent sample counts and windows (see compare)"},
                {"metric": "~823 us local FA, ~1100 us RTT, ~387/300/420 us split",
                 "class": "hypothesis",
                 "source": "prior attribution notes; not measured by this trace"},
            ],
            "limitations": [
                "Parent and child spans overlap by design; only root/frontier spans are additive.",
                "A container (rpc_total, job_finish, out_decode) is reported, never summed with children.",
                "Peer gpu_us is comparable only when the peer counts and window match this trace.",
                "Duplicate cycle ids are not joined: per-correlation attribution is unknown.",
            ],
        }


def analyze_trace(path, min_position=130560, max_position=CONTEXT_CAP,
                  context_cap=CONTEXT_CAP):
    analysis = TraceAnalysis(path, min_position, max_position, context_cap).load()
    report = analysis.report()
    report["_analysis"] = analysis
    return report


# --------------------------------------------------------------------------- #
# CPU engine vs GPU congruence gate
# --------------------------------------------------------------------------- #

def load_samples(path):
    """Load {"window": {...}, "op": ..., "samples_us": [...]} for comparison."""
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise TraceFormatError(f"{path}: expected a JSON object")
    samples = data.get("samples_us")
    if not isinstance(samples, list):
        raise TraceFormatError(f"{path}: samples_us must be a list")
    clean = []
    for index, value in enumerate(samples):
        number = _as_duration(value, f"samples_us[{index}]", 0)
        if number < 0:
            raise TraceFormatError(f"{path}: negative sample at index {index}")
        clean.append(number)
    window = data.get("window") or {}
    return {
        "source": path,
        "op": data.get("op"),
        "window": {
            "min_position": window.get("min_position"),
            "max_position": window.get("max_position"),
        },
        "samples_us": clean,
    }


def compare_samples(cpu, gpu):
    """Compare CPU engine and GPU samples only when both are congruent."""
    result = {
        "comparable": False,
        "reason": None,
        "cpu": None,
        "gpu": None,
        "scope": "unknown",
    }
    for side, data in (("cpu", cpu), ("gpu", gpu)):
        samples = data.get("samples_us") or []
        if not samples:
            result["reason"] = f"{side} has no samples"
            return result
        for value in samples:
            if not math.isfinite(value) or value < 0:
                result["reason"] = f"{side} has a non-finite or negative sample"
                result["scope"] = "failed"
                return result
    result["cpu"] = summarize(cpu["samples_us"])
    result["gpu"] = summarize(gpu["samples_us"])
    if len(cpu["samples_us"]) != len(gpu["samples_us"]):
        result["reason"] = (f"sample count mismatch: cpu={len(cpu['samples_us'])} "
                            f"gpu={len(gpu['samples_us'])}")
        result["scope"] = "failed"
        return result
    if cpu.get("window") != gpu.get("window"):
        result["reason"] = (f"window mismatch: cpu={cpu.get('window')} "
                            f"gpu={gpu.get('window')}")
        result["scope"] = "failed"
        return result
    if (cpu.get("op") or "rpc_total") != (gpu.get("op") or "rpc_total"):
        result["reason"] = f"op mismatch: cpu={cpu.get('op')} gpu={gpu.get('op')}"
        result["scope"] = "failed"
        return result
    delta = [c - g for c, g in zip(cpu["samples_us"], gpu["samples_us"])]
    result.update({
        "comparable": True,
        "scope": "ok",
        "delta_us": summarize(delta),
        "delta_p50_ratio": (result["cpu"]["p50"] / result["gpu"]["p50"]
                            if result["gpu"]["p50"] else None),
    })
    return result


# --------------------------------------------------------------------------- #
# Busy-process guard
# --------------------------------------------------------------------------- #

def _proc_pids(proc_root):
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return []
    return sorted(int(name) for name in entries if name.isdigit())


def _read_proc_bytes(path):
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return b""


def _ancestor_pids(proc_root, pid):
    ancestors = set()
    current = pid
    while current and current not in ancestors:
        ancestors.add(current)
        stat = _read_proc_bytes(os.path.join(proc_root, str(current), "stat")).decode(
            "utf-8", "replace")
        close = stat.rfind(")")
        if close < 0:
            break
        fields = stat[close + 2:].split()
        if len(fields) < 2:
            break
        current = int(fields[1]) if fields[1].isdigit() else 0
    return ancestors


def busy_process_name(name):
    """True only for executables/scripts that own the console or the host GPU."""
    lowered = name.lower()
    if lowered in BUSY_EXACT_NAMES:
        return True
    if lowered.endswith(".exe") and lowered[:-4] in BUSY_EXACT_NAMES:
        return True
    if lowered.startswith(BUSY_NAME_PREFIXES):
        return True
    if lowered.startswith(("rkva-", "rkva_")) and any(token in lowered
                                                      for token in BUSY_NAME_TOKENS):
        return True
    return False


def find_busy_processes(proc_root="/proc", own_pid=None):
    """Scan /proc for live llama/RKVA workloads.

    Matching uses the executable basename (and the running script when the
    interpreter is python/bash), never arbitrary argument text, so a CLI that
    merely mentions these names in its own arguments does not trip it.
    """
    own_pid = os.getpid() if own_pid is None else own_pid
    skip = _ancestor_pids(proc_root, own_pid)
    skip.add(own_pid)
    busy = []
    for pid in _proc_pids(proc_root):
        if pid in skip:
            continue
        raw = _read_proc_bytes(os.path.join(proc_root, str(pid), "cmdline"))
        argv = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        names = set()
        if argv:
            exe = os.path.basename(argv[0])
            names.add(exe)
            interpreter = re.match(r"^(python|bash|sh|env)", exe.lower()) is not None
            if interpreter:
                for token in argv[1:3]:
                    if token.startswith("-") or any(ch.isspace() for ch in token):
                        continue
                    names.add(os.path.basename(token))
                    break
        comm = _read_proc_bytes(os.path.join(proc_root, str(pid), "comm")).decode(
            "utf-8", "replace").strip()
        if comm:
            names.add(comm)
        matched = sorted(name for name in names if busy_process_name(name))
        if matched:
            busy.append({"pid": pid, "name": matched[0],
                         "argv0": os.path.basename(argv[0]) if argv else ""})
    return busy


# --------------------------------------------------------------------------- #
# Empty-RTT probe (opt-in)
# --------------------------------------------------------------------------- #

def build_ping_request(cycle_id):
    return struct.pack(REQUEST_FMT, RKVA_MAGIC, RKVA_VERSION, RKVA_PING,
                       cycle_id & 0xFFFFFFFFFFFFFFFF, RKVA_SESSION_NONE,
                       0, 0, 0)


def parse_response(buffer):
    if len(buffer) != RESPONSE_BYTES:
        raise ProbeError(f"response is {len(buffer)} bytes, expected {RESPONSE_BYTES}")
    fields = struct.unpack(RESPONSE_FMT, buffer)
    return {
        "magic": fields[0],
        "version": fields[1],
        "op": fields[2],
        "cycle_id": fields[3],
        "status": fields[4],
        "payload_bytes": fields[5],
        "xbox_receive_us": fields[6],
        "xbox_prepare_us": fields[7],
        "xbox_compute_us": fields[8],
        "xbox_response_us": fields[9],
    }


def _recv_exact(sock, size):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ProbeError(f"connection closed with {remaining} of {size} bytes pending")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _validate_control_request(op, session, allowed=(RKVA_PING,)):
    """Validate the request we are about to send, before it leaves the process."""
    if op not in allowed:
        raise ProbeError(f"refusing to send op {op}: only {allowed} are safe here")
    if session != RKVA_SESSION_NONE:
        raise ProbeError("refusing to send a session-bound request in --empty-rtt mode")


def _read_control_response(sock, expected_op, expected_cycle, allow_payload=False,
                           max_payload=0):
    """Validate the whole header before reading any body.

    PING is a zero-payload op: a non-zero payload_bytes is a protocol fault and
    is refused without allocating or reading a body. No response field is
    trusted before magic/version/op/cycle/status/payload_bytes are checked.
    """
    header = parse_response(_recv_exact(sock, RESPONSE_BYTES))
    if header["magic"] != RKVA_MAGIC:
        raise ProbeError("bad magic in response")
    if header["version"] not in (RKVA_VERSION, RKVA_VERSION_LEGACY):
        raise ProbeError(f"unexpected protocol version {header['version']}")
    if header["op"] != expected_op:
        raise ProbeError(f"response op {header['op']} does not echo the request")
    if header["cycle_id"] != (expected_cycle & 0xFFFFFFFFFFFFFFFF):
        raise ProbeError("response cycle id does not echo the request")
    if header["status"] != 0:
        raise ProbeError(f"server returned status {header['status']}")
    if not allow_payload and header["payload_bytes"] != 0:
        raise ProbeError(f"{expected_op} response declares "
                         f"{header['payload_bytes']} payload bytes; PING must be empty, "
                         "refusing to read a body")
    if header["payload_bytes"] > max_payload:
        raise ProbeError(f"response payload {header['payload_bytes']} exceeds the "
                         f"{max_payload}-byte cap")
    payload = b""
    if header["payload_bytes"]:
        payload = _recv_exact(sock, header["payload_bytes"])
    return header, payload


def _validate_ping(response, cycle_id):
    """Header-only PING validation (kept as the single-op entry point)."""
    if response["magic"] != RKVA_MAGIC:
        raise ProbeError("bad magic in response")
    if response["version"] not in (RKVA_VERSION, RKVA_VERSION_LEGACY):
        raise ProbeError(f"unexpected protocol version {response['version']}")
    if response["op"] != RKVA_PING:
        raise ProbeError(f"response op {response['op']} does not echo PING")
    if response["cycle_id"] != (cycle_id & 0xFFFFFFFFFFFFFFFF):
        raise ProbeError("response cycle id does not echo the request")
    if response["status"] != 0:
        raise ProbeError(f"server returned status {response['status']}")
    if response["payload_bytes"] != 0:
        raise ProbeError("PING response must not carry a payload")


def verify_cycle_sequence(sent, received):
    """Prove the received cycle ids are exactly the sent ones, in order.

    Duration equality is irrelevant: two samples with identical wall times but
    distinct ids are valid. A repeated id in the received sequence is a real
    correlation fault and fails.
    """
    if len(sent) != len(received):
        raise ProbeError(f"cycle sequence length mismatch: sent {len(sent)}, "
                         f"received {len(received)}")
    for index, (want, got) in enumerate(zip(sent, received)):
        if want != got:
            raise ProbeError(f"cycle id mismatch at sample {index}: sent {want}, got {got}")
    if len(set(received)) != len(received):
        raise ProbeError("duplicate cycle ids in the received sequence")
    return True


def probe_empty_rtt(host, port, reps=50, warmup=5, timeout=1.0,
                    allow_busy=False, connector=None, clock=time.perf_counter,
                    get_stats=False):
    """One TCP connection, bounded PING samples, no HELLO and no session."""
    if not 1 <= reps <= MAX_RTT_REPS:
        raise ValueError(f"--reps must be between 1 and {MAX_RTT_REPS}")
    if warmup < 0 or warmup > 20:
        raise ValueError("--warmup must be between 0 and 20")
    if not 0 < timeout <= MAX_RTT_TIMEOUT_S:
        raise ValueError(f"--timeout must be in (0, {MAX_RTT_TIMEOUT_S}] s")
    busy = find_busy_processes()
    if busy and not allow_busy:
        names = ", ".join(f"pid {entry['pid']} {entry['name']}" for entry in busy[:5])
        raise ProbeError(
            "refusing --execute: a live llama/RKVA workload is running "
            f"({names}). A new TCP connection makes the server free its engine; "
            "wait until the console is idle or pass --allow-busy explicitly.")
    if connector is None:
        connector = socket.create_connection
    sock = None
    samples = []
    sent_cycles = []
    received_cycles = []
    response_echo = {"status_ok": True, "cycle_ids_unique": True,
                     "payload_bytes": 0, "versions": [],
                     "session_checked_on_request": True,
                     "session_on_wire": False}
    stats_payload = None
    try:
        sock = connector((host, port), timeout)
        try:
            sock.settimeout(timeout)
        except OSError:
            pass
        cycle = 1
        if get_stats:
            _validate_control_request(RKVA_GET_STATS, RKVA_SESSION_NONE,
                                      allowed=(RKVA_GET_STATS,))
            sock.sendall(struct.pack(REQUEST_FMT, RKVA_MAGIC, RKVA_VERSION,
                                     RKVA_GET_STATS, cycle, RKVA_SESSION_NONE,
                                     0, 0, 0))
            header, payload = _read_control_response(
                sock, RKVA_GET_STATS, cycle, allow_payload=True,
                max_payload=MAX_CONTROL_PAYLOAD)
            try:
                stats_payload = json.loads(payload.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                stats_payload = {"raw_bytes": len(payload), "parse": "failed"}
            cycle += 1
        for index in range(warmup + reps):
            _validate_control_request(RKVA_PING, RKVA_SESSION_NONE)
            request = build_ping_request(cycle)
            started = clock()
            sock.sendall(request)
            # Header is fully validated, including payload_bytes == 0, before
            # any body read; PING never has a body.
            header, _payload = _read_control_response(sock, RKVA_PING, cycle)
            elapsed = (clock() - started) * 1e6
            sent_cycles.append(cycle)
            received_cycles.append(header["cycle_id"])
            _validate_ping(header, cycle)
            response_echo["versions"].append(header["version"])
            if index >= warmup:
                samples.append(elapsed)
            cycle += 1
        verify_cycle_sequence(sent_cycles, received_cycles)
        # Uniqueness is about the actual protocol ids, never about wall times:
        # two samples with identical durations are valid.
        response_echo["cycle_ids_unique"] = len(set(received_cycles)) == len(received_cycles)
        response_echo["cycle_id_count"] = len(received_cycles)
        if not samples:
            raise ProbeError("no sample was collected")
        stats = summarize(samples)
        return {
            "schema": SCHEMA,
            "mode": "rtt",
            "scope": "ok",
            "target": f"{host}:{port}",
            "op": {
                "name": "RKVA_PING",
                "id": RKVA_PING,
                "payload_bytes": 0,
                "sessionless": True,
                "hello": False,
                "gpu_touched": False,
                "note": ("PING is served before any engine is created; HELLO/CREATE_SESSION "
                         "and RANGE_PROFILE were not sent"),
            },
            "protocol": {
                "magic": "RKVA",
                "version_tx": RKVA_VERSION,
                "version_rx": sorted(set(response_echo["versions"])),
                "request_bytes": REQUEST_BYTES,
                "response_bytes": RESPONSE_BYTES,
            },
            "method": ("single TCP connection, one PING per sample, wall clock around "
                       "write + full response read; warmup samples excluded"),
            "guard": {
                "busy_processes": busy,
                "allow_busy": bool(allow_busy),
                "checked": "executable/script basename of every /proc entry, own pid excluded",
            },
            "warmup": warmup,
            "reps": reps,
            "timeout_s": timeout,
            "samples_us": stats,
            "samples": samples,
            "wall_ms": stats["total"] / 1000.0,
            "bytes": {
                "request": REQUEST_BYTES * (warmup + reps),
                "response": RESPONSE_BYTES * (warmup + reps),
            },
            "response_echo": response_echo,
            "server_stats": stats_payload,
            "notes": [
                "Wall time includes both hosts' scheduling; it is not GPU time.",
                "Cycle ids are validated as an exact send/receive sequence; equal "
                "wall times are valid and never used as an id check.",
                "A new connection resets the server's engine/session state by design.",
                "No deploy, no flag change, no restart was performed.",
            ],
            "evidence": [
                {"metric": "empty RTT", "class": "measured",
                 "source": "wall clock around PING write + read"},
                {"metric": "GPU / CPU / network split", "class": "hypothesis",
                 "source": "requires aligned peer counters; not derived here"},
            ],
        }
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Local network facts (read-only)
# --------------------------------------------------------------------------- #

def is_readonly_network_argv(argv):
    """Allowlist for the only subprocesses this tool may run."""
    if not argv or argv[0] not in NET_CMDS:
        return False
    tool = argv[0]
    rest = list(argv[1:])
    if not rest:
        return False
    if tool == "ethtool":
        if len(rest) == 1 and not rest[0].startswith("-"):
            return True
        if rest[0] == "--version":
            return len(rest) == 1
        if rest[0] not in READONLY_ETHTOOL_FLAGS:
            return False
        return len(rest) == 2 and not rest[1].startswith("-")
    # ip: read-only show subcommands only
    if rest[0] == "-json":
        if len(rest) < 2 or rest[1] not in READONLY_IP_ACTIONS:
            return False
        words = rest[2:]
    else:
        if rest[0] not in READONLY_IP_ACTIONS:
            return False
        words = rest[1:]
    return not any(word in IP_MUTATING_WORDS for word in words)


def _default_runner(argv, timeout=5.0):
    try:
        completed = subprocess.run(argv, capture_output=True, text=True,
                                   timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return completed.returncode, completed.stdout


def _safe_run(argv, runner, commands):
    if not is_readonly_network_argv(argv):
        raise ValueError(f"refusing non-read-only command: {argv}")
    commands.append(list(argv))
    return runner(argv)


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _read_int(path):
    text = _read_text(path)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _ethtool_sections(iface, runner, commands, unavailable):
    sections = {}
    for flag, key in (("-i", "driver_info"), ("-k", "features"), ("-c", "coalesce")):
        code, output = _safe_run(["ethtool", flag, iface], runner, commands)
        if code != 0:
            unavailable.append({"item": key, "reason": (output or "").strip()[:200] or
                                f"ethtool {flag} exit {code}"})
            sections[key] = None
            continue
        parsed = {}
        for line in output.splitlines():
            if ":" not in line:
                continue
            name, _, value = line.partition(":")
            parsed[name.strip()] = value.strip()
        sections[key] = parsed
    return sections


def _route_target(target):
    if not target:
        try:
            with open("/proc/net/route", "r", encoding="utf-8") as handle:
                for line in handle.readlines()[1:]:
                    fields = line.split()
                    if len(fields) > 3 and fields[1] == "00000000":
                        return fields[0], "default_route"
        except OSError:
            return None, "unavailable"
        return None, "no_default_route"
    return target, "target"


def network_info(target=None, runner=_default_runner):
    """Read-only interface/driver/link facts; never contacts the target."""
    commands = []
    unavailable = []
    iface, route_source = None, "unavailable"
    if target:
        code, output = _safe_run(["ip", "-json", "route", "get", target], runner, commands)
        if code == 0:
            try:
                routes = json.loads(output)
                if isinstance(routes, list) and routes:
                    iface = routes[0].get("dev")
                    route_source = "ip -json route get"
            except json.JSONDecodeError:
                unavailable.append({"item": "route", "reason": "ip returned non-JSON"})
        else:
            unavailable.append({"item": "route", "reason": (output or "").strip()[:200] or
                                f"ip exit {code}"})
    if not iface:
        iface, route_source = _route_target(None)

    report = {
        "schema": SCHEMA,
        "mode": "network",
        "scope": "unknown",
        "target": target,
        "route": {"interface": iface, "source": route_source},
        "link": {},
        "offloads": None,
        "coalescing": None,
        "driver": None,
        "bus": None,
        "id": None,
        "unavailable": unavailable,
        "commands": commands,
        "notes": [
            "Read-only: no sudo, no ethtool -s/-K/-C/-G, no ip set, no sysctl write.",
            "Local addresses, MACs and IPv6 privacy addresses are not reported.",
            "The remote host is never contacted by --network-info.",
            "These facts do not establish causality for RKVA latency; only an "
            "aligned trace/probe can attribute the delay.",
        ],
    }
    if not iface:
        report["unavailable"].append({"item": "interface", "reason": "not resolved"})
        return report

    base = f"/sys/class/net/{iface}"
    report["link"] = {
        "mtu": _read_int(f"{base}/mtu"),
        "speed_mbps": _read_int(f"{base}/speed"),
        "duplex": _read_text(f"{base}/duplex"),
        "operstate": _read_text(f"{base}/operstate"),
        "carrier": _read_int(f"{base}/carrier"),
        "wireless": os.path.isdir(f"{base}/wireless"),
    }
    if report["link"]["speed_mbps"] is not None and report["link"]["speed_mbps"] <= 0:
        report["link"]["speed_mbps"] = None
        report["unavailable"].append({"item": "speed", "reason": "driver reports no rate"})
    driver_link = os.path.join(base, "device", "driver")
    try:
        report["driver"] = os.path.basename(os.readlink(driver_link))
    except OSError:
        report["unavailable"].append({"item": "driver", "reason": "sysfs has no driver link"})
    try:
        device_path = os.path.realpath(os.path.join(base, "device"))
    except OSError:
        device_path = ""
    if "/usb" in device_path:
        report["bus"] = "usb"
        vendor = product = None
        current = os.path.join(base, "device")
        for _ in range(3):
            vendor = vendor or _read_text(f"{current}/idVendor")
            product = product or _read_text(f"{current}/idProduct")
            parent = os.path.dirname(os.path.realpath(current))
            if parent in ("/", current) or not parent:
                break
            current = parent
        report["id"] = {"vendor": vendor, "product": product}
    elif device_path.startswith("/sys/devices"):
        report["bus"] = "pci"
        uevent = _read_text(f"{base}/device/uevent") or ""
        for line in uevent.splitlines():
            if line.startswith("PCI_ID="):
                report["id"] = {"pci": line.split("=", 1)[1]}
                break

    sections = _ethtool_sections(iface, runner, commands, unavailable)
    features = sections.get("features") or {}
    report["offloads"] = {
        key: features.get(key) for key in (
            "rx-checksumming", "tx-checksumming", "tcp-segmentation-offload",
            "generic-segmentation-offload", "generic-receive-offload",
            "large-receive-offload",
        )
    } if features else None
    coalesce = sections.get("coalesce") or {}
    report["coalescing"] = {
        key: coalesce.get(key) for key in (
            "adaptive-rx", "adaptive-tx", "rx-usecs", "tx-usecs",
            "rx-frames", "tx-frames",
        )
    } if coalesce else None
    report["driver_info"] = sections.get("driver_info")

    missing = []
    if not report["offloads"]:
        missing.append("offloads")
    if not report["coalescing"]:
        missing.append("coalescing")
    report["scope"] = "partial" if (missing or unavailable) else "ok"
    if missing:
        report["unavailable"].append({"item": ",".join(missing),
                                      "reason": "ethtool did not return usable output"})
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    parser = argparse.ArgumentParser(
        prog="rkva-latency-audit.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "RKVA latency evidence tool. Offline by default: it reads a trace file "
            "only. --execute opens one TCP connection and sends PING (no HELLO, no "
            "session, no GPU). --network-info reads local link facts with read-only "
            "ethtool/ip queries. No file is written without --out."),
        epilog=(
            "examples:\n"
            "  rkva-latency-audit.py --trace t.jsonl --min-position 130560 \\\n"
            "      --max-position 160000 --out audit.json\n"
            "  rkva-latency-audit.py --execute --host 192.168.1.200:50054 \\\n"
            "      --empty-rtt --reps 50 --timeout 1.0 --out rtt.json\n"
            "  rkva-latency-audit.py --network-info --host 192.168.1.200:50054\n"
            "\n"
            "safety: no GPU/Xbox/LAN traffic unless --execute is explicit; the "
            "process guard refuses to run while llama/RKVA workloads are alive."),
    )
    parser.add_argument("--trace", metavar="FILE",
                        help="RKVA host trace JSONL (offline, read-only)")
    parser.add_argument("--min-position", "--min-pos", dest="min_position", type=int,
                        default=130560,
                        help="first decode position to attribute (default: 130560)")
    parser.add_argument("--max-position", "--max-pos", dest="max_position", type=int,
                        default=CONTEXT_CAP,
                        help=f"last decode position to attribute (default: {CONTEXT_CAP})")
    parser.add_argument("--context-cap", type=int, default=CONTEXT_CAP,
                        help=f"audited context cap; values above {CONTEXT_CAP_LIMIT} are refused")
    parser.add_argument("--cpu-samples", metavar="FILE",
                        help="CPU engine samples JSON for the congruence gate")
    parser.add_argument("--gpu-samples", metavar="FILE",
                        help="GPU samples JSON for the congruence gate")
    parser.add_argument("--execute", action="store_true",
                        help="allow the opt-in network probe (requires --empty-rtt)")
    parser.add_argument("--empty-rtt", action="store_true",
                        help="measure PING round trips on one control connection")
    parser.add_argument("--host", metavar="HOST:PORT",
                        help="RKVA server endpoint, e.g. 192.168.1.200:50054")
    parser.add_argument("--reps", type=int, default=50,
                        help=f"measured samples, 1..{MAX_RTT_REPS} (default: 50)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="discarded warmup samples, 0..20 (default: 5)")
    parser.add_argument("--timeout", type=float, default=1.0,
                        help=f"per-sample timeout in seconds, max {MAX_RTT_TIMEOUT_S} (default: 1.0)")
    parser.add_argument("--include-stats", action="store_true",
                        help="send one GET_STATS control read before the samples")
    parser.add_argument("--allow-busy", action="store_true",
                        help="bypass the busy-process guard (loudly reported; off by default)")
    parser.add_argument("--network-info", action="store_true",
                        help="report local route/interface/link facts (read-only)")
    parser.add_argument("--out", "--json", dest="out", metavar="FILE",
                        help="write the JSON report to FILE (no file without this)")
    parser.add_argument("--strict", action="store_true",
                        help="exit 2 when any requested scope stays unknown")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def _print_trace_summary(report):
    counts = report["counts"]
    print(f"trace={report['trace']}")
    print(f"scope={report['scope']} window={report['window']['min_position']}.."
          f"{report['window']['max_position']} events={counts['decode_events']} "
          f"tokens={counts['tokens']} layer_groups={counts['layer_groups']}")
    print(f"{'phase':28s} {'n':>6s} {'min_us':>9s} {'p50_us':>9s} {'p95_us':>9s} "
          f"{'total_ms':>9s} {'per_tok_ms':>10s} {'resid/exp_ms':>13s} class")
    for phase, stats in sorted(report["phases"].items(),
                               key=lambda item: -item[1]["total"]):
        kind = "container" if stats["container"] else ("wait" if stats["wait_span"] else "span")
        remainder = (stats["residual_total_us"] if stats["container"]
                     else stats["exposed_total_us"])
        print(f"{phase:28s} {stats['count']:6d} {stats['min']:9.0f} {stats['p50']:9.0f} "
              f"{stats['p95']:9.0f} {stats['total'] / 1000.0:9.3f} "
              f"{stats['per_token_us'] / 1000.0:10.3f} {remainder / 1000.0:13.3f} "
              f"{kind}")
    token = report["per_token"]
    print(f"\nper token ({token['tokens']} tokens, "
          f"layers/token {token['layers_per_token']['min']}"
          f"..{token['layers_per_token']['max']}):")
    for key in ("envelope_ms", "attributed_ms", "unattributed_gap_ms",
                "rpc_total_ms_container", "peer_gpu_ms"):
        stats = token[key]
        if stats["count"]:
            print(f"  {key:24s} p50={stats['p50']:8.3f} p95={stats['p95']:8.3f} "
                  f"total={stats['total']:9.3f}")
    corr = report["correlation"]
    print(f"correlation: scope={corr['scope']} valid={corr['groups_with_correlation']} "
          f"absent={corr['groups_without_correlation']} "
          f"duplicate_ids={corr['duplicate_ids']} mixed={corr['groups_with_mixed_correlation']} "
          f"reused_windows={corr['reused_in_independent_windows']} "
          f"ambiguous={corr['ambiguous_groups']}")
    for reason in corr["reasons"]:
        print(f"  unknown: {reason}")


def _print_rtt(report):
    stats = report["samples_us"]
    print(f"target={report['target']} op={report['op']['name']} reps={report['reps']} "
          f"warmup={report['warmup']} timeout={report['timeout_s']}s")
    print(f"empty RTT: n={stats['count']} min={stats['min']:.1f}us p50={stats['p50']:.1f}us "
          f"p95={stats['p95']:.1f}us max={stats['max']:.1f}us wall={report['wall_ms']:.1f}ms")
    print(f"bytes: request={report['bytes']['request']} response={report['bytes']['response']} "
          f"method={report['method']}")


def _print_network(report):
    link = report["link"]
    print(f"target={report['target']} interface={report['route']['interface']} "
          f"route_source={report['route']['source']}")
    print(f"driver={report['driver']} bus={report['bus']} id={report['id']} "
          f"mtu={link.get('mtu')} speed_mbps={link.get('speed_mbps')} "
          f"duplex={link.get('duplex')} operstate={link.get('operstate')}")
    if report.get("offloads"):
        print("offloads: " + ", ".join(f"{k}={v}" for k, v in report["offloads"].items()))
    else:
        print("offloads: unavailable")
    if report.get("coalescing"):
        print("coalescing: " + ", ".join(f"{k}={v}" for k, v in report["coalescing"].items()))
    else:
        print("coalescing: unavailable")
    for item in report["unavailable"]:
        print(f"unavailable: {item['item']} ({item['reason']})")


def parse_endpoint(text):
    if not text:
        raise ValueError("--execute requires --host HOST:PORT")
    host, sep, port = text.rpartition(":")
    if not sep or not host or not port.isdigit():
        raise ValueError(f"--host must be HOST:PORT, got {text!r}")
    value = int(port)
    if not 0 < value < 65536:
        raise ValueError(f"invalid port in {text!r}")
    return host, value


def write_report(path, report):
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        raise OSError(f"output directory does not exist: {parent}")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    report = {}
    exit_code = 0
    try:
        if args.trace:
            trace = analyze_trace(args.trace, args.min_position, args.max_position,
                                  args.context_cap)
            trace.pop("_analysis", None)
            if args.cpu_samples or args.gpu_samples:
                if not (args.cpu_samples and args.gpu_samples):
                    raise TraceFormatError(
                        "the CPU/GPU comparison needs both --cpu-samples and --gpu-samples")
                cpu = load_samples(args.cpu_samples)
                gpu = load_samples(args.gpu_samples)
                comparison = compare_samples(cpu, gpu)
                trace["comparison"] = comparison
                trace["unknown"]["comparison_scope"] = comparison["scope"]
                if comparison["scope"] == "failed":
                    trace["scope"] = "partial" if trace["scope"] == "ok" else trace["scope"]
                    trace["scope_reasons"].append(
                        f"CPU/GPU comparison failed: {comparison['reason']}")
            if trace.get("comparison", {}).get("scope") == "failed":
                print("error: CPU/GPU comparison failed: "
                      f"{trace['comparison']['reason']}", file=sys.stderr)
                exit_code = 2
            report = trace
            if args.network_info:
                report["network"] = network_info(args.host.split(":")[0] if args.host else None)
            _print_trace_summary(report)
            if report.get("comparison"):
                comparison = report["comparison"]
                state = "comparable" if comparison["comparable"] else "NOT comparable"
                print(f"cpu/gpu: {state}"
                      + ("" if comparison["comparable"] else f" ({comparison['reason']})"))
            if report.get("network"):
                _print_network(report["network"])
            if report["correlation"]["scope"] == "failed":
                print("error: correlation attribution failed: duplicate or mixed "
                      "cycle ids (see report.correlation.reasons)", file=sys.stderr)
                exit_code = 2
            if args.strict and (report["scope"] != "ok"
                                or report["correlation"]["scope"] != "ok"
                                or report["unknown"]["comparison_scope"] == "unknown"):
                exit_code = 2
        elif args.execute:
            if not args.empty_rtt:
                raise ValueError(
                    "--execute without a safe op: only --empty-rtt is implemented "
                    "(RANGE_PROFILE needs HELLO and would create the GPU engine)")
            host, port = parse_endpoint(args.host)
            report = probe_empty_rtt(host, port, reps=args.reps, warmup=args.warmup,
                                     timeout=args.timeout, allow_busy=args.allow_busy,
                                     get_stats=args.include_stats)
            if args.network_info:
                report["network"] = network_info(host)
            _print_rtt(report)
            if report.get("network"):
                _print_network(report["network"])
        elif args.network_info:
            report = network_info(args.host.split(":")[0] if args.host else None)
            _print_network(report)
        else:
            parser.error("nothing to do: pass --trace FILE, --execute --empty-rtt, "
                         "or --network-info")
        if args.strict:
            if report.get("mode") == "network" and report.get("scope") != "ok":
                exit_code = 2
            if report.get("network") and report["network"].get("scope") != "ok":
                exit_code = 2
    except (TraceFormatError, ProbeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.out:
        try:
            write_report(args.out, report)
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {args.out}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
