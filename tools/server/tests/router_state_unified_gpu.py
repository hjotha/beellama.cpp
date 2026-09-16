#!/usr/bin/env python3
"""Real-context GPU matrix for the unified persistent KV snapshot store.

The wrapper owns the exclusive service window. This harness only uses the
private work directory and the frozen r1 preset clone supplied with
``--preset``. It deliberately separates candidate restore events from
effective ``cache_n`` reuse and never treats a restore that is discarded by a
checkpoint/carry guard as a hit.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import re
import struct
import time
import urllib.parse

from router_state_streaming import Server
from router_state_unified_cpu import (
    assert_profile,
    auto_state_name,
    memory_peak,
    stream_completion,
    token_pattern,
)


CONTEXTS = [32768, 56320, 97536, 104448]
PROFILES = ["mtp-short", "mtp", "long", "wide"]
MAGIC = b"LLROUTE1"
FOOTER_SIZE = 24


def event_records(lines, start=0):
    """Collect auto-cache and router-transfer events in source order.

    The bracketed prefix in a server log is an actor/port namespace, not an
    operating-system PID.  The caller records the real PID from ``Server``
    and verifies it with the all-TID sampler.  Router transfer events do not
    currently include a path or timing, but they are still disk evidence for
    effective-origin classification when they are followed by consumed
    ``cache_n``.
    """
    saves = []
    restores = []
    save_re = re.compile(
        r"auto-save: persisted unified snapshot reason=(\S+) tokens=(\d+) bytes=(\d+) "
        r"save_ms=([0-9.]+) workspace=(\d+) path=(\S+)")
    restore_re = re.compile(
        r"auto-restore: reused (\d+) tokens from disk .*?bytes=(\d+) restore_ms=([0-9.]+) "
        r"workspace=(\d+), file=(\S+)")
    route_save_re = re.compile(
        r"router streaming save: tokens=(\d+) bytes=(\d+).*?workspace=(\d+)")
    route_restore_re = re.compile(
        r"router streaming restore: tokens=(\d+) bytes=(\d+).*?workspace=(\d+)")
    for line_number, line in enumerate(lines[start:], start=start + 1):
        actor = re.search(r"\[(\d+)\]", line)
        log_actor_prefix = actor.group(1) if actor else None
        match = save_re.search(line)
        if match:
            saves.append({"kind": "auto", "line": line_number,
                          "reason": match.group(1), "tokens": int(match.group(2)),
                          "bytes": int(match.group(3)), "save_ms": float(match.group(4)),
                          "workspace": int(match.group(5)), "path": match.group(6),
                          "log_actor_prefix": log_actor_prefix})
            continue
        match = route_save_re.search(line)
        if match:
            saves.append({"kind": "router", "line": line_number,
                          "reason": "router-transfer", "tokens": int(match.group(1)),
                          "bytes": int(match.group(2)), "save_ms": None,
                          "workspace": int(match.group(3)), "path": None,
                          "log_actor_prefix": log_actor_prefix})
            continue
        match = restore_re.search(line)
        if match:
            restores.append({"kind": "auto", "line": line_number,
                             "tokens": int(match.group(1)), "bytes": int(match.group(2)),
                             "restore_ms": float(match.group(3)),
                             "workspace": int(match.group(4)), "path": match.group(5),
                             "log_actor_prefix": log_actor_prefix})
            continue
        match = route_restore_re.search(line)
        if match:
            restores.append({"kind": "router", "line": line_number,
                             "tokens": int(match.group(1)), "bytes": int(match.group(2)),
                             "restore_ms": None, "workspace": int(match.group(3)),
                             "path": None, "log_actor_prefix": log_actor_prefix})
    return saves, restores


def bounded_manifest(path):
    with path.open("rb") as source:
        size = path.stat().st_size
        if size < FOOTER_SIZE:
            raise ValueError("snapshot is shorter than footer")
        source.seek(-FOOTER_SIZE, 2)
        length, footer_hash, magic = struct.unpack("<QQ8s", source.read(FOOTER_SIZE))
        if magic != MAGIC or length <= 0 or length > 64 * 1024 or length > size - FOOTER_SIZE:
            raise ValueError("invalid snapshot footer")
        source.seek(-FOOTER_SIZE - length, 2)
        metadata = source.read(length)
    if len(metadata) != length:
        raise ValueError("short snapshot manifest")
    manifest = json.loads(metadata)
    return manifest, footer_hash, size


def canonical_states(store):
    result = []
    if not store.is_dir():
        return result
    for path in sorted(store.glob("auto-*.bin")):
        try:
            manifest, _, _ = bounded_manifest(path)
            if manifest.get("version") == 1 and int(manifest.get("index_block", 0)) > 0:
                result.append(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError, struct.error):
            continue
    return result


def identity(path):
    manifest, _, size = bounded_manifest(path)
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "inode": path.stat().st_ino,
        "bytes": size,
        "state_checksum": str(manifest["state_checksum"]),
        "n_tokens": int(manifest["n_tokens"]),
        "position": int(manifest["position"]),
        "index_block": int(manifest["index_block"]),
    }


def find_prompt_snapshot(store, tokens):
    states = canonical_states(store)
    if not states:
        raise AssertionError(f"no canonical snapshots in {store}")
    salts = {int(path.name.split("-")[1], 16) for path in states}
    for salt in salts:
        expected = store / auto_state_name(salt, tokens)
        if expected.exists():
            manifest, _, _ = bounded_manifest(expected)
            if int(manifest.get("n_tokens", -1)) == len(tokens):
                return expected
    candidates = []
    for path in states:
        manifest, _, _ = bounded_manifest(path)
        if int(manifest.get("n_tokens", -1)) == len(tokens):
            candidates.append(path)
    if len(candidates) != 1:
        raise AssertionError(f"ambiguous prompt snapshot n={len(tokens)}: {candidates}")
    return candidates[0]


def annotate(row, process_root, source_pid=None):
    sample = memory_peak(process_root)
    row["os_pid"] = row["pid"]
    row["observed_os_pids"] = sample["observed_os_pids"]
    row["pid_verified_by_sampler"] = row["pid"] in sample["observed_os_pids"]
    row["memory"] = sample
    if source_pid is not None:
        row["source_pid"] = source_pid
    if not row["pid_verified_by_sampler"]:
        raise AssertionError(f"{row['label']}: router PID absent from all-TID sampler")


def scenario_for_label(label):
    if "restart-original" in label:
        return "restart-original"
    if "restart-continue" in label:
        return "restart-continue"
    if label.startswith("disk-only"):
        return "disk-only-chain"
    if label.startswith("D-"):
        return "normal-D"
    if "B-after-A" in label:
        return "A-B-active"
    if "A-after-B" in label:
        return "A-B-restart"
    if "A-generate" in label:
        return "A-generate"
    return "unclassified"


def request_row(server, public, label, prompt, n_predict, origin, conversation=None,
                cache_prompt=True, expected_disk_path=None, require_disk=False,
                allow_context_clip=False, require_no_cache=False, scenario=None):
    line_start = len(server.lines)
    started = time.perf_counter()
    body = stream_completion(server, {
        "model": public,
        "prompt": prompt,
        "n_predict": n_predict,
        "id_slot": 0,
        "cache_prompt": cache_prompt,
        "return_tokens": True,
        "temperature": 0,
        "seed": 1234,
        "ignore_eos": True,
    }, conversation=conversation)
    elapsed = (time.perf_counter() - started) * 1000.0
    time.sleep(0.2)
    tokens = body.get("tokens", [])
    clipped = len(tokens) != n_predict
    if clipped:
        if not allow_context_clip or len(tokens) >= n_predict or not body.get("truncated") or \
                body.get("stop_type") != "limit" or body.get("stop_detail") != "context_limit":
            raise AssertionError(f"{label}: unexpected output length/body: {body}")
    if body.get("model") != public:
        raise AssertionError(f"{label}: model ID changed: {body.get('model')} != {public}")
    timings = body.get("timings", {})
    if "cache_n" not in timings or "prompt_n" not in timings:
        raise AssertionError(f"{label}: timing evidence is insufficient: {body}")
    cache_n = int(timings["cache_n"])
    prompt_n = int(timings["prompt_n"])
    if cache_n < 0 or prompt_n < 0 or cache_n + prompt_n != len(prompt):
        raise AssertionError(f"{label}: cache_n+prompt_n != input ({cache_n}+{prompt_n}!={len(prompt)})")
    saves, restores = event_records(server.lines, line_start)
    matching = [event for event in restores
                if expected_disk_path is None or
                (event.get("path") is not None and
                 Path(event["path"]).resolve() == expected_disk_path.resolve())]
    restore = (matching[-1] if expected_disk_path is not None and matching else
               None if expected_disk_path is not None else
               restores[-1] if restores else None)
    if require_no_cache and (cache_n != 0 or restores):
        raise AssertionError(f"{label}: distinct prompt unexpectedly reused cache: "
                             f"cache_n={cache_n} restores={restores}")
    if require_disk:
        if restore is None:
            raise AssertionError(f"{label}: no explicit disk restore event: {restores}")
        if cache_n <= 0 or prompt_n >= len(prompt):
            discarded = restore is not None and cache_n == 0
            reason = next((line.strip() for line in server.lines[line_start:]
                           if "restore discarded" in line or "cold fallback" in line), None)
            raise AssertionError(f"{label}: restored candidate not consumed; discarded={discarded}; "
                                 f"cache_n={cache_n} prompt_n={prompt_n} reason={reason!r}")
    if origin == "RAM" and cache_n <= 0:
        raise AssertionError(f"{label}: RAM hit did not produce cache_n>0")
    effective_origin = ("NONE" if cache_n == 0 else
                        "DISK" if restore is not None else "RAM")
    fallback_lines = [line.strip() for line in server.lines[line_start:]
                      if re.search(r"(?i)(?:restore discarded|cold fallback|fallback reason|"
                                   r"incompatible|checksum|truncated|layout differs)", line)]
    fallback_reason = fallback_lines[-1] if fallback_lines else None
    save = next((event for event in reversed(saves)
                 if event["reason"] in ("prompt", "prompt-prefix")), None)
    row = {
        "label": label,
        "origin": origin,
        "expected_origin": origin,
        "cache_origin": effective_origin,
        "scenario": scenario or scenario_for_label(label),
        "pid": server.proc.pid,
        "input_tokens": len(prompt),
        "requested_output": n_predict,
        "generated_tokens": len(tokens),
        "context_limit_clip": clipped,
        "n_saved": save["tokens"] if save else 0,
        "n_restored": restore["tokens"] if restore else 0,
        "n_reused": cache_n,
        "n_recomputed": prompt_n,
        "snapshot_bytes": restore["bytes"] if restore else (save["bytes"] if save else None),
        "save_ms": save["save_ms"] if save else None,
        "restore_ms": restore["restore_ms"] if restore else None,
        "ttft_ms": body.get("ttft_ms"),
        "total_ms": body.get("total_ms", elapsed),
        "transient_workspace_bytes": restore["workspace"] if restore else (save["workspace"] if save else None),
        "timings": timings,
        "tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "disk_restore_event": restore is not None,
        "disk_restore_consumed": restore is not None and cache_n > 0,
        "restored_then_discarded": restore is not None and cache_n == 0,
        "disk_restore_path": restore["path"] if restore else None,
        "disk_restore_kind": restore["kind"] if restore else None,
        "restore_events": restores,
        "save_event_path": save["path"] if save else None,
        "save_events": saves,
        "fallback_reason": fallback_reason,
        "fallback_reason_status": ("logged" if fallback_reason else
                                    "not_logged" if restore is not None and cache_n == 0 else
                                    "not_applicable"),
    }
    return tokens, body, row


def server_args(args, store):
    return SimpleNamespace(server=args.server, gpu=True, port=0, timeout=args.timeout,
                           slot_save_path=str(store), env_overrides={})


def fill_prompt(history, length, salt):
    prompt = list(history)
    while len(prompt) < length:
        prompt.extend(token_pattern(min(length - len(prompt), 64), salt))
    return prompt[:length]


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def common_prefix_length(left, right):
    length = min(len(left), len(right))
    for index in range(length):
        if left[index] != right[index]:
            return index
    return length


def run_d_matrix(base, args, public, preset):
    root = base / "normal-d-matrix"
    store = root / "store"
    store.mkdir(parents=True)
    server = None
    rows = []
    profiles = []
    history = []
    try:
        (root / "router").mkdir(parents=True, exist_ok=False)
        server = Server(server_args(args, store), root / "router", preset)
        tokenized = server.request("/tokenize", {
            "model": public,
            "content": "The river is blue. The sky is clear.",
            "add_special": True,
        })["tokens"]
        if not tokenized:
            raise AssertionError("GPU tokenizer returned no tokens")
        for index, capacity in enumerate(CONTEXTS):
            prompt = fill_prompt(history, capacity - 4096, 101 + index)
            exact, _, row = request_row(server, public, f"D-{capacity}-exact",
                                        prompt, 4096, "COLD", conversation="gpu-d",
                                        allow_context_clip=True)
            row["profile"] = PROFILES[index]
            rows.append(row)
            slot = server.request("/slots?model=" + urllib.parse.quote(public, safe=""))[0]
            assert_profile(slot, capacity, index)
            profiles.append({"profile": PROFILES[index], "capacity": capacity,
                             "slot": slot, "pid": server.proc.pid})
            repeat, _, repeat_row = request_row(server, public, f"D-{capacity}-repeat-RAM",
                                                prompt, 16, "RAM", conversation="gpu-d")
            repeat_row["profile"] = PROFILES[index]
            rows.append(repeat_row)
            cold, _, cold_row = request_row(server, public, f"D-{capacity}-cold-control",
                                            prompt, 16, "COLD_CONTROLLED", conversation="gpu-cold",
                                            cache_prompt=False)
            cold_row["profile"] = PROFILES[index]
            rows.append(cold_row)
            if repeat != cold:
                raise AssertionError(f"D {capacity}: cached and cold deterministic outputs differ")
            if index < 2:
                margin_prompt = prompt[:-1]
                margin, _, margin_row = request_row(server, public, f"D-{capacity}-mtp-margin",
                                                    margin_prompt, 4096, "COLD_MARGIN",
                                                    conversation="gpu-margin", cache_prompt=False)
                margin_row["profile"] = PROFILES[index]
                rows.append(margin_row)
                if len(margin) != 4096:
                    raise AssertionError(f"D {capacity}: MTP margin did not produce 4096 tokens")
            history = prompt + exact
    finally:
        if server:
            server.close()
    for row in rows:
        annotate(row, root / "router")
    log_text = (root / "router" / "server.log").read_text(errors="replace")
    route_saves = re.findall(r"router streaming save: tokens=(\d+) bytes=(\d+)", log_text)
    route_restores = re.findall(r"router streaming restore: tokens=(\d+) bytes=(\d+)", log_text)
    return {"kind": "normal-d", "rows": rows, "profiles": profiles,
            "route_saves": route_saves, "route_restores": route_restores,
            "store": str(store), "memory": memory_peak(root / "router")}


def run_restart_profile(base, args, public, preset, index):
    capacity = CONTEXTS[index]
    profile = PROFILES[index]
    root = base / f"restart-{capacity}"
    store = root / "store"
    store.mkdir(parents=True)
    prompt = fill_prompt([], capacity - 4096, 211 + index)
    prompt_b = fill_prompt([], capacity - 4096, 212 + index)
    if prompt == prompt_b or prompt[0] == prompt_b[0]:
        raise AssertionError(f"{profile}: A/B fixture prompts are not clearly divergent")
    rows = []
    first_pid = None
    generated = None
    first = None
    try:
        (root / "first").mkdir(parents=True, exist_ok=False)
        first = Server(server_args(args, store), root / "first", preset)
        first_pid = first.proc.pid
        generated, _, row = request_row(first, public, f"{profile}-{capacity}-A-generate",
                                         prompt, 16, "COLD", conversation="gpu-profile")
        row["profile"] = profile
        rows.append(row)
        prompt_event_path = row.get("save_event_path")
        repeated, _, row = request_row(first, public, f"{profile}-{capacity}-A-repeat-RAM",
                                       prompt, 8, "RAM", conversation="gpu-profile")
        if repeated != generated[:8]:
            raise AssertionError(f"{profile}: active repeat differs")
        row["profile"] = profile
        rows.append(row)
        _, _, row = request_row(first, public, f"{profile}-{capacity}-B-after-A",
                                prompt_b, 16, "COLD", conversation="gpu-profile")
        row["profile"] = profile
        rows.append(row)
        slot = first.request("/slots?model=" + urllib.parse.quote(public, safe=""))[0]
        assert_profile(slot, capacity, index)
    finally:
        if first:
            first.close()
    if generated is None:
        raise AssertionError(f"{profile}: generation produced no IDs")
    expected_tokens = prompt[:-1] if index < 2 else prompt
    if not prompt_event_path:
        raise AssertionError(f"{profile}: prompt checkpoint publication was not logged")
    prompt_path = Path(prompt_event_path)
    if not prompt_path.exists() or identity(prompt_path)["n_tokens"] != len(expected_tokens):
        raise AssertionError(f"{profile}: logged prompt checkpoint is not the expected {len(expected_tokens)}-token object")
    prompt_id = identity(prompt_path)
    before_third = set(canonical_states(store))

    second = None
    try:
        (root / "restart-original").mkdir(parents=True, exist_ok=False)
        second = Server(server_args(args, store), root / "restart-original", preset)
        original, _, row = request_row(second, public, f"{profile}-{capacity}-restart-original-DISK",
                                       prompt, 8, "DISK", conversation=None,
                                       expected_disk_path=prompt_path, require_disk=True)
        row["profile"] = profile
        row["source_pid"] = first_pid
        rows.append(row)
        slot = second.request("/slots?model=" + urllib.parse.quote(public, safe=""))[0]
        assert_profile(slot, capacity, index)
        if index < 2 and "target-only MTP bootstrap accepted" not in "".join(second.lines):
            raise AssertionError(f"{profile}: original restart lacked MTP bootstrap")
        cold, _, cold_row = request_row(second, public, f"{profile}-{capacity}-restart-original-cold",
                                        prompt, 8, "COLD_CONTROLLED", conversation="gpu-cold",
                                        cache_prompt=False)
        cold_row["profile"] = profile
        rows.append(cold_row)
        if original != cold:
            raise AssertionError(f"{profile}: original restart differs from cold")
    finally:
        if second:
            second.close()

    third = None
    try:
        (root / "restart-continuation").mkdir(parents=True, exist_ok=False)
        third = Server(server_args(args, store), root / "restart-continuation", preset)
        resumed_prompt = prompt + generated + [7, 11, 19, 23]
        resumed, _, row = request_row(third, public, f"{profile}-{capacity}-restart-continue-all-IDs",
                                       resumed_prompt, 8, "DISK", conversation=None,
                                       require_disk=True)
        row["profile"] = profile
        row["source_pid"] = first_pid
        rows.append(row)
        if Path(row["disk_restore_path"]).resolve() not in before_third:
            raise AssertionError(f"{profile}: continuation restore path was not preexisting")
        slot = third.request("/slots?model=" + urllib.parse.quote(public, safe=""))[0]
        assert_profile(slot, capacity, index)
        if index < 2 and "target-only MTP bootstrap accepted" not in "".join(third.lines):
            raise AssertionError(f"{profile}: continuation lacked MTP bootstrap")
        cold, _, cold_row = request_row(third, public, f"{profile}-{capacity}-restart-continue-cold",
                                        resumed_prompt, 8, "COLD_CONTROLLED", conversation="gpu-cold",
                                        cache_prompt=False)
        cold_row["profile"] = profile
        rows.append(cold_row)
        if resumed != cold:
            raise AssertionError(f"{profile}: continuation differs from cold")
    finally:
        if third:
            third.close()
    after_id = identity(prompt_path)
    if after_id["inode"] != prompt_id["inode"] or after_id["state_checksum"] != prompt_id["state_checksum"]:
        raise AssertionError(f"{profile}: dedicated prompt snapshot identity changed")
    process_roots = {"first": root / "first", "restart-original": root / "restart-original",
                     "restart-continuation": root / "restart-continuation"}
    for row in rows:
        which = ("restart-original" if "restart-original" in row["label"] else
                 "restart-continuation" if "restart-continue" in row["label"] else "first")
        annotate(row, process_roots[which], first_pid if row["origin"] == "DISK" else None)
    return {"profile": profile, "capacity": capacity, "first_pid": first_pid,
            "generated_ids": len(generated), "rows": rows,
            "dedicated_prompt_snapshot_before": prompt_id,
            "dedicated_prompt_snapshot_after": after_id,
            "identity_path_checksum_inode_preserved": True,
            "store": str(store),
            "memory": {name: memory_peak(path) for name, path in process_roots.items()}}


def run_ab_distinct_supplement(base, args, public, preset):
    """Exercise a genuinely different B prompt before a fresh-process A hit.

    This is intentionally a small supplement to the full GPU matrix.  It does
    not rerun D or the existing cold controls: the A response is compared to
    its deterministic first-process response, while the fresh process must
    consume the original A snapshot after B has published a different object.
    """
    profiles = []
    for index, capacity in enumerate(CONTEXTS):
        profile = PROFILES[index]
        root = base / f"ab-distinct-{capacity}"
        store = root / "store"
        store.mkdir(parents=True)
        prompt_a = fill_prompt([], capacity - 4096, 211 + index)
        prompt_b = fill_prompt([], capacity - 4096, 212 + index)
        lcp = common_prefix_length(prompt_a, prompt_b)
        prompt_a_hash = token_hash(prompt_a)
        prompt_b_hash = token_hash(prompt_b)
        if prompt_a == prompt_b or lcp != 0 or prompt_a[0] == prompt_b[0]:
            raise AssertionError(f"{profile}: A/B supplement is not clearly divergent: "
                                 f"lcp={lcp} first={prompt_a[0]}/{prompt_b[0]}")

        rows = []
        process_roots = {}
        first = None
        first_pid = None
        generated_a = None
        prompt_path = None
        try:
            first_root = root / "first"
            first_root.mkdir(parents=True, exist_ok=False)
            process_roots["first"] = first_root
            first = Server(server_args(args, store), first_root, preset)
            first_pid = first.proc.pid
            generated_a, _, row = request_row(
                first, public, f"{profile}-{capacity}-A-generate", prompt_a, 16,
                "COLD", conversation="gpu-ab-distinct", scenario="A-generate")
            row.update({"profile": profile, "prompt_role": "A",
                        "prompt_sha256": prompt_a_hash})
            rows.append(row)
            prompt_path = row.get("save_event_path")
            if not prompt_path:
                raise AssertionError(f"{profile}: A prompt snapshot was not published")

            repeated, _, row = request_row(
                first, public, f"{profile}-{capacity}-A-repeat-RAM", prompt_a, 8,
                "RAM", conversation="gpu-ab-distinct", scenario="A-repeat-RAM")
            if repeated != generated_a[:8]:
                raise AssertionError(f"{profile}: active A repeat differs")
            row.update({"profile": profile, "prompt_role": "A",
                        "prompt_sha256": prompt_a_hash})
            rows.append(row)

            generated_b, _, row = request_row(
                first, public, f"{profile}-{capacity}-B-after-A", prompt_b, 16,
                "NONE", conversation="gpu-ab-distinct", require_no_cache=True,
                scenario="B-after-A")
            row.update({"profile": profile, "prompt_role": "B",
                        "prompt_sha256": prompt_b_hash})
            rows.append(row)
            if row["cache_origin"] != "NONE":
                raise AssertionError(f"{profile}: B was not a cold/no-cache request")
            slot = first.request("/slots?model=" + urllib.parse.quote(public, safe=""))[0]
            assert_profile(slot, capacity, index)
        finally:
            if first:
                first.close()

        if generated_a is None or not prompt_path:
            raise AssertionError(f"{profile}: supplement A setup did not complete")
        prompt_path = Path(prompt_path)
        if not prompt_path.exists():
            raise AssertionError(f"{profile}: A snapshot disappeared before restart: {prompt_path}")
        before_restart = set(canonical_states(store))
        if prompt_path.resolve() not in {path.resolve() for path in before_restart}:
            raise AssertionError(f"{profile}: A snapshot was not retained after distinct B")
        prompt_before_b = identity(prompt_path)

        second = None
        try:
            second_root = root / "restart-a"
            second_root.mkdir(parents=True, exist_ok=False)
            process_roots["restart-a"] = second_root
            second = Server(server_args(args, store), second_root, preset)
            restarted_a, _, row = request_row(
                second, public, f"{profile}-{capacity}-A-after-B-restart-DISK", prompt_a, 8,
                "DISK", conversation=None, expected_disk_path=prompt_path,
                require_disk=True, scenario="A-B-restart")
            row.update({"profile": profile, "prompt_role": "A",
                        "prompt_sha256": prompt_a_hash, "source_pid": first_pid})
            rows.append(row)
            if restarted_a != generated_a[:8]:
                raise AssertionError(f"{profile}: restart A after distinct B differs from A")
            if index < 2 and "target-only MTP bootstrap accepted" not in "".join(second.lines):
                raise AssertionError(f"{profile}: restart A after B lacked MTP bootstrap")
            slot = second.request("/slots?model=" + urllib.parse.quote(public, safe=""))[0]
            assert_profile(slot, capacity, index)
        finally:
            if second:
                second.close()

        prompt_after_b = identity(prompt_path)
        if (prompt_after_b["inode"] != prompt_before_b["inode"] or
                prompt_after_b["state_checksum"] != prompt_before_b["state_checksum"]):
            raise AssertionError(f"{profile}: A snapshot identity changed across B/restart")
        for row in rows:
            annotate(row, process_roots["restart-a"] if "A-after-B" in row["label"] else
                     process_roots["first"], first_pid if row["expected_origin"] == "DISK" else None)
        profiles.append({
            "profile": profile,
            "capacity": capacity,
            "expected_mtp": index < 2,
            "first_pid": first_pid,
            "generated_ids": len(generated_a),
            "prompt_a_sha256": prompt_a_hash,
            "prompt_b_sha256": prompt_b_hash,
            "prompt_lcp": lcp,
            "prompt_first_tokens": [prompt_a[0], prompt_b[0]],
            "a_snapshot_before_b": prompt_before_b,
            "a_snapshot_after_b": prompt_after_b,
            "identity_path_checksum_inode_preserved": True,
            "rows": rows,
            "store": str(store),
            "memory": {name: memory_peak(path) for name, path in process_roots.items()},
        })
    return {"kind": "distinct-a-b-restart", "profiles": profiles,
            "contexts": CONTEXTS,
            "notes": [
                "A and B are distinct token fixtures with LCP=0 and distinct first tokens.",
                "B requires cache_n=0 and no restore event; it is not a cache hit.",
                "Fresh-process A must consume the retained A snapshot after B.",
                "Existing D and cold-control results are referenced, not rerun.",
            ]}


def run_disk_chain(base, args, public, preset):
    root = base / "disk-only-chain"
    store = root / "store"
    store.mkdir(parents=True)
    history = []
    rows = []
    prior_pid = None
    previous_states = []
    for index, capacity in enumerate(CONTEXTS):
        profile = PROFILES[index]
        prompt = fill_prompt(history, capacity - 4096, 411 + index)
        before = set(canonical_states(store))
        server = None
        current_pid = None
        try:
            (root / f"pid-{capacity}").mkdir(parents=True, exist_ok=False)
            server = Server(server_args(args, store), root / f"pid-{capacity}", preset)
            current_pid = server.proc.pid
            generated, _, row = request_row(server, public, f"disk-only-{profile}-{capacity}",
                                             prompt, 16, "DISK" if index else "COLD",
                                             conversation=None, require_disk=bool(index))
            row["profile"] = profile
            row["source_pid"] = prior_pid if index else None
            if index:
                if Path(row["disk_restore_path"]).resolve() not in before:
                    raise AssertionError(f"disk-only {profile}: restore path was not preexisting")
                if row["n_reused"] <= 0:
                    raise AssertionError(f"disk-only {profile}: no effective cache reuse")
            slot = server.request("/slots?model=" + urllib.parse.quote(public, safe=""))[0]
            assert_profile(slot, capacity, index)
            if index == 1 and "target-only MTP bootstrap accepted" not in "".join(server.lines):
                raise AssertionError("disk-only MTP transition lacked bootstrap")
            rows.append(row)
            cached, _, cached_row = request_row(server, public, f"disk-only-{profile}-repeat",
                                                prompt, 8, "RAM", conversation="gpu-chain")
            cold, _, cold_row = request_row(server, public, f"disk-only-{profile}-cold",
                                            prompt, 8, "COLD_CONTROLLED", conversation="gpu-cold",
                                            cache_prompt=False)
            cached_row["profile"] = profile
            cold_row["profile"] = profile
            rows.extend([cached_row, cold_row])
            if cached != cold:
                raise AssertionError(f"disk-only {profile}: cached/cold mismatch")
        finally:
            if server:
                server.close()
        for row in rows:
            if row["pid"] == current_pid and row.get("profile") == profile:
                annotate(row, root / f"pid-{capacity}")
        states = canonical_states(store)
        if not states:
            raise AssertionError(f"disk-only {profile}: store empty after restart")
        if index and not before.intersection(set(states)):
            raise AssertionError(f"disk-only {profile}: previous snapshots disappeared")
        previous_states = states
        prior_pid = current_pid
        history = prompt + generated
    return {"kind": "disk-only", "rows": rows,
            "profiles": PROFILES, "contexts": CONTEXTS,
            "store": str(store),
            "prompt_snapshots": [str(path.resolve()) for path in previous_states],
            "memory": {str(capacity): memory_peak(root / f"pid-{capacity}") for capacity in CONTEXTS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output-tokens", type=int, default=4096)
    parser.add_argument("--auto-cache", action="store_true")
    parser.add_argument("--ab-supplement", action="store_true",
                        help="run only the distinct A->B->fresh-process-A supplement")
    args = parser.parse_args()
    if not args.auto_cache:
        raise ValueError("GPU unified matrix requires --auto-cache")
    if args.output_tokens != 4096:
        raise ValueError("GPU unified matrix evidence fixes output at 4096 tokens")
    if Path(args.model).resolve().as_posix() != args.model_id:
        raise ValueError("GPU model path and public model ID must be the same literal path")
    base = Path(args.work_dir).resolve()
    base.mkdir(parents=True, exist_ok=False)
    preset = Path(args.preset).resolve()
    if not preset.is_file():
        raise ValueError(f"preset missing: {preset}")
    public = args.model_id
    result = {"schema": "unified-kv-gpu-a-d-v1", "model_id": public,
              "contexts": CONTEXTS, "profiles": PROFILES,
              "preset": str(preset), "rows": [],
              "notes": [
                  "Real production-sized contexts; no CPU training-context override.",
                  "DISK requires a fresh PID, explicit restore event, cache_n>0 and prompt_n<input.",
                  "MTP exact boundary clipping and one-token margin are recorded separately.",
                  "TTFT is first nonempty SSE emission; workspace is separate from RSS/VRAM.",
              ]}
    if args.ab_supplement:
        result["ab_supplement"] = run_ab_distinct_supplement(base, args, public, preset)
        (base / "results.json").write_text(json.dumps(result, indent=2))
        print("PASS: GPU distinct A->B->fresh-process-A supplement in all four profiles",
              flush=True)
        return
    result["normal_d"] = run_d_matrix(base, args, public, preset)
    (base / "results.json").write_text(json.dumps(result, indent=2))
    result["restart_profiles"] = []
    for index in range(4):
        result["restart_profiles"].append(run_restart_profile(base, args, public, preset, index))
        (base / "results.json").write_text(json.dumps(result, indent=2))
    result["disk_only_chain"] = run_disk_chain(base, args, public, preset)
    (base / "results.json").write_text(json.dumps(result, indent=2))
    print("PASS: GPU unified D real contexts, MTP margins/bootstrap, fresh-PID A/B restart cases, "
          "normal and disk-only chains, cold controls and bounded snapshot evidence", flush=True)


if __name__ == "__main__":
    main()
