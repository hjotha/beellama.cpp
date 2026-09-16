#!/usr/bin/env python3
"""CPU A--C matrix for the unified persistent KV snapshot store.

This deliberately uses scaled fixtures (256/512/768/1024 tokens) rather than
the production GPU capacities.  Each disk case starts a fresh router process,
does not send ``X-Conversation-Id``, and requires an explicit auto-restore log;
``cache_n`` by itself is never accepted as disk evidence.

Example:
  TMPDIR=/home/hjotha/router-kv-snapshots-unificados-20260916/cpu-matrix-unified \
    python3 tools/server/tests/router_state_unified_cpu.py \
      --server build-bench/bin/llama-server \
      --model build-bench/tests/test-models/qwen35-mtp.gguf \
      --work-dir /home/hjotha/router-kv-snapshots-unificados-20260916/cpu-matrix-unified
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from types import SimpleNamespace
import struct
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

from router_state_streaming import Server


MAGIC = b"LLROUTE1"
FOOTER_SIZE = 24
DEFAULT_CONTEXTS = [256, 512, 768, 1024]
PROFILE_NAMES = ["mtp-short", "mtp", "long", "wide"]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path):
    raw = path.read_bytes()
    if len(raw) < FOOTER_SIZE:
        raise AssertionError(f"snapshot too short: {path}")
    length, checksum, magic = struct.unpack("<QQ8s", raw[-FOOTER_SIZE:])
    if magic != MAGIC or length == 0 or length > 64 * 1024 or length > len(raw) - FOOTER_SIZE:
        raise AssertionError(f"snapshot footer invalid: {path}")
    start = len(raw) - FOOTER_SIZE - length
    manifest = json.loads(raw[start:start + length])
    return manifest, raw, checksum


def snapshot_identity(path):
    manifest, _, _ = read_manifest(path)
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "inode": path.stat().st_ino,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "state_checksum": str(manifest["state_checksum"]),
        "n_tokens": int(manifest["n_tokens"]),
        "model": manifest.get("model"),
        "layout": manifest.get("layout"),
        "index_block": int(manifest.get("index_block", 0)),
    }


def canonical_states(store):
    states = []
    try:
        candidates = sorted(store.glob("auto-*.bin"))
    except OSError:
        return states
    for path in candidates:
        try:
            manifest, _, _ = read_manifest(path)
            if manifest.get("version") == 1 and int(manifest.get("index_block", 0)) > 0:
                states.append(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError, struct.error):
            pass
    return states


def make_preset(path, model, contexts, public, max_count=64, max_mb=16384):
    path.write_text(f"""version = 1
[*]
model = {model}
override-kv = qwen35.context_length=int:{contexts[-1]}
load-mode = none
parallel = 1
device = none
gpu-layers = 0
flash-attn = on
fit = off
cache-ram = 8192
ctx-checkpoints = 2
checkpoint-min-step = 1
metrics = true
cache-prompt = true
context-shift = false
threads = 2
threads-batch = 2
cache-type-k = q4_0
cache-type-v = q4_0
slot-save-auto = true
slot-save-block = 1
slot-save-max-count = {max_count}
slot-save-max-mb = {max_mb}
[tri]
load-on-startup = true
route-group = {public}
route-max-tokens = {contexts[2]}
ctx-size = {contexts[2]}
ctx-size-mtp = {contexts[1]}
mtp-max-tokens = {contexts[1]}
ctx-size-mtp-short = {contexts[0]}
mtp-short-max-tokens = {contexts[0]}
spec-type = draft-mtp
spec-draft-n-max = 2
spec-draft-n-max-short = 4
spec-draft-p-min = 0.70
spec-draft-type-k = q4_0
spec-draft-type-v = q4_0
batch-size = 256
ubatch-size = 256
[wide]
route-group = {public}
ctx-size = {contexts[3]}
spec-type = none
batch-size = 64
ubatch-size = 64
""")


def start_server(args, root, preset, store, env_overrides=None):
    root.mkdir(parents=True, exist_ok=False)
    server_args = SimpleNamespace(server=args.server, gpu=False, port=0,
                                   timeout=args.timeout, slot_save_path=str(store),
                                   env_overrides=env_overrides or {})
    return Server(server_args, root, preset)


def stream_completion(server, payload, conversation=None):
    headers = {"Content-Type": "application/json"}
    if conversation is not None:
        headers["X-Conversation-Id"] = conversation
    request = urllib.request.Request(server.url + "/completion",
                                     data=json.dumps({**payload, "stream": True}).encode(),
                                     headers=headers)
    started = time.perf_counter()
    first = None
    final = {}
    tokens = []
    try:
        with urllib.request.urlopen(request, timeout=server.timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk is None:
                    continue
                final.update({key: value for key, value in chunk.items() if value is not None})
                if not chunk.get("stop"):
                    chunk_tokens = chunk.get("tokens") or []
                    tokens.extend(chunk_tokens)
                    if first is None and (chunk_tokens or chunk.get("content")):
                        first = time.perf_counter()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"/completion: HTTP {error.code}: {error.read().decode()}") from error
    ended = time.perf_counter()
    final["tokens"] = tokens
    final.setdefault("model", payload.get("model"))
    final["ttft_ms"] = None if first is None else (first - started) * 1000.0
    final["total_ms"] = (ended - started) * 1000.0
    return final


def event_records(lines, start=0):
    saves = []
    restores = []
    save_re = re.compile(
        r"auto-save: persisted unified snapshot reason=(\S+) tokens=(\d+) bytes=(\d+) "
        r"save_ms=([0-9.]+) workspace=(\d+) path=(\S+)")
    restore_re = re.compile(
        r"auto-restore: reused (\d+) tokens from disk .*?bytes=(\d+) restore_ms=([0-9.]+) "
        r"workspace=(\d+), file=(\S+)")
    for line in lines[start:]:
        pid_match = re.search(r"\[(\d+)\]", line)
        # The bracketed prefix is the server's log actor/port namespace, not
        # an operating-system PID. Keep it only as a diagnostic label; callers
        # use Server.proc.pid and the all-TID sampler for real PID evidence.
        log_actor_prefix = pid_match.group(1) if pid_match else None
        match = save_re.search(line)
        if match:
            saves.append({"reason": match.group(1), "tokens": int(match.group(2)),
                          "bytes": int(match.group(3)), "save_ms": float(match.group(4)),
                          "workspace": int(match.group(5)), "path": match.group(6),
                          "log_actor_prefix": log_actor_prefix})
        match = restore_re.search(line)
        if match:
            restores.append({"tokens": int(match.group(1)), "bytes": int(match.group(2)),
                             "restore_ms": float(match.group(3)), "workspace": int(match.group(4)),
                             "path": match.group(5), "log_actor_prefix": log_actor_prefix})
    return saves, restores


def settle_log(server):
    # The completion response is synchronous, while the line-reader is a
    # sibling thread. One short drain interval makes the event evidence stable
    # without a polling loop and is negligible beside model startup.
    time.sleep(0.05)


def memory_peak(root):
    path = root / "memory.jsonl"
    peak_rss = None
    peak_anon = None
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            for process in sample.get("processes_kib", []):
                peak_rss = max(peak_rss or 0, int(process.get("VmRSS", 0)))
                peak_anon = max(peak_anon or 0, int(process.get("RssAnon", 0)))
    observed = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            observed.update(int(process["pid"]) for process in sample.get("processes_kib", [])
                            if "pid" in process)
    return {"rss_max_kib": peak_rss, "anon_max_kib": peak_anon,
            "observed_os_pids": sorted(observed),
            "sampler": "all-TID descendant tree; process RSS is not the 8 MiB stream buffer"}


def status(server, public):
    query = urllib.parse.quote(public, safe="")
    slots = server.request("/slots?model=" + query)
    if not slots:
        raise AssertionError("no slot status")
    return slots[0]


def assert_profile(slot, capacity, index):
    if int(slot.get("n_ctx", -1)) != capacity:
        raise AssertionError(f"expected n_ctx={capacity}, got {slot}")
    adaptive = slot.get("adaptive_context")
    if index < 3:
        if not adaptive or adaptive.get("profile") != PROFILE_NAMES[index]:
            raise AssertionError(f"expected adaptive profile {PROFILE_NAMES[index]}: {slot}")
        if index < 2 and not adaptive.get("mtp_weights_resident"):
            raise AssertionError(f"MTP was not resident for {PROFILE_NAMES[index]}: {slot}")
        if index < 2 and not slot.get("speculative"):
            raise AssertionError(f"slot did not advertise speculative MTP: {slot}")
    elif adaptive is not None:
        raise AssertionError(f"wide no-MTP child unexpectedly adaptive: {slot}")


def request_row(server, public, label, prompt, n_predict, origin, conversation=None,
                expected_disk_path=None, require_disk=False, cache_prompt=True):
    line_start = len(server.lines)
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
    settle_log(server)
    tokens = body.get("tokens", [])
    if len(tokens) != n_predict:
        raise AssertionError(f"{label}: expected {n_predict} generated IDs, got {len(tokens)}; {body}")
    if body.get("model") != public:
        raise AssertionError(f"{label}: model ID changed: {body.get('model')} != {public}")
    saves, restores = event_records(server.lines, line_start)
    matching = [event for event in restores
                if expected_disk_path is None or Path(event["path"]).resolve() == expected_disk_path.resolve()]
    if require_disk and not matching:
        raise AssertionError(f"{label}: no explicit disk restore event; restores={restores}")
    timing = body.get("timings", {})
    if "cache_n" not in timing or "prompt_n" not in timing:
        raise AssertionError(f"{label}: timing evidence insufficient; cache_n/prompt_n absent: {body}")
    restore = matching[-1] if matching else (restores[-1] if restores else None)
    cache_n = int(timing["cache_n"])
    prompt_n = int(timing["prompt_n"])
    if cache_n < 0 or prompt_n < 0 or cache_n + prompt_n != len(prompt):
        raise AssertionError(f"{label}: cache_n + prompt_n does not cover exactly the request: "
                             f"cache_n={cache_n} prompt_n={prompt_n} input={len(prompt)}")
    if require_disk and (restore is None or restore["tokens"] <= 0):
        raise AssertionError(f"{label}: disk restore event did not report reused tokens: {restores}")
    if require_disk and (cache_n <= 0 or prompt_n >= len(prompt)):
        discarded = restore is not None and cache_n == 0
        reason = next((line.strip() for line in server.lines[line_start:]
                       if "target-only snapshot restore discarded" in line or
                          "cold fallback" in line), None)
        raise AssertionError(f"{label}: disk candidate was not consumed; restored_then_discarded={discarded}; "
                             f"cache_n={cache_n} prompt_n={prompt_n} reason={reason!r}")
    if origin == "RAM" and cache_n <= 0:
        raise AssertionError(f"{label}: RAM cache reuse was not consumed: {timing}")
    save = next((event for event in reversed(saves)
                 if event["reason"] in ("prompt", "prompt-prefix")), None)
    reused = cache_n
    restored_then_discarded = restore is not None and reused == 0
    row = {
        "label": label,
        "origin": origin,
        "pid": server.proc.pid,
        "log_actor_prefix": (restore["log_actor_prefix"] if restore else
                              save["log_actor_prefix"] if save else None),
        "input_tokens": len(prompt),
        "generated_tokens": len(tokens),
        "n_saved": save["tokens"] if save else 0,
        "n_restored": restore["tokens"] if restore else 0,
        "n_reused": reused,
        "n_recomputed": int(timing.get("prompt_n", 0)),
        "snapshot_bytes": (restore["bytes"] if restore and "bytes" in restore else
                            save["bytes"] if save else None),
        "save_ms": save["save_ms"] if save else None,
        "restore_ms": restore["restore_ms"] if restore else None,
        "save_event_path": save["path"] if save else None,
        "save_events": saves,
        "ttft_ms": body.get("ttft_ms"),
        "total_ms": body.get("total_ms"),
        "transient_workspace_bytes": (restore["workspace"] if restore else
                                       save["workspace"] if save else None),
        "timings": timing,
        "tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "disk_restore_event": restore is not None,
        "disk_restore_consumed": restore is not None and reused > 0,
        "restored_then_discarded": restored_then_discarded,
        "disk_restore_path": restore["path"] if restore else None,
    }
    if require_disk and restore is None:
        row["fallback_reason"] = "required explicit restore event was absent"
    elif restored_then_discarded:
        row["fallback_reason"] = next((line.strip() for line in server.lines[line_start:]
            if "target-only snapshot restore discarded" in line or "cold fallback" in line),
            "restore event present but no logged discard reason")
    elif origin == "COLD" and not restore:
        row["fallback_reason"] = "no compatible resident/persistent prefix"
    return tokens, body, row


def remove_other_states(store, keep):
    keep = keep.resolve()
    for path in canonical_states(store):
        if path.resolve() == keep:
            continue
        path.unlink()
        for suffix in (".logits", ".meta"):
            sidecar = Path(str(path) + suffix)
            if sidecar.exists():
                sidecar.unlink()


def prompt_snapshot(store, token_count, expected_tokens=None):
    candidates = []
    for path in canonical_states(store):
        try:
            manifest, _, _ = read_manifest(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError, struct.error):
            continue
        if int(manifest.get("n_tokens", -1)) == token_count:
            candidates.append(path)
    if expected_tokens is not None and candidates:
        salts = {int(path.name.split("-")[1], 16) for path in candidates}
        for salt in salts:
            expected = store / auto_state_name(salt, expected_tokens)
            if expected in candidates:
                return expected
    if len(candidates) != 1:
        raise AssertionError(f"expected one dedicated prompt snapshot for {token_count}: {candidates}")
    return candidates[0]


def token_pattern(length, salt):
    return [3 + ((salt * 19 + index * 17) % 50) for index in range(length)]


def auto_hash_mix(value, token):
    mask = (1 << 64) - 1
    value ^= token & 0xFFFFFFFF
    value = (value * 0x100000001B3) & mask
    value ^= value >> 29
    value = (value * 0xBF58476D1CE4E5B9) & mask
    value ^= value >> 32
    return value & mask


def auto_state_name(salt, tokens):
    value = 0xCBF29CE484222325 ^ salt
    for token in tokens:
        value = auto_hash_mix(value, token)
    return f"auto-{salt:016x}-{value:016x}-{len(tokens)}.bin"


def run_profile_case(base, args, fixture, public, contexts, index):
    capacity = contexts[index]
    root = base / f"profile-{capacity}"
    store = root / "store"
    store.mkdir(parents=True)
    preset = root / "models.ini"
    make_preset(preset, fixture, contexts, public)
    prompt = token_pattern(capacity - 32, 11 + index)
    prompt_b = token_pattern(capacity - 32, 71 + index)
    rows = []
    first_pid = None
    generated = None
    server = None
    try:
        first_root = root / "first"
        server = start_server(args, first_root, preset, store)
        first_pid = server.proc.pid
        generated, _, row = request_row(server, public, "profile-%d-A-generate" % capacity,
                                         prompt, 16, "COLD", conversation="profile")
        rows.append(row)
        slot = status(server, public)
        assert_profile(slot, capacity, index)
        # The active repeat exercises RAM prompt reuse in the same PID.
        repeated, _, row = request_row(server, public, "profile-%d-A-repeat-RAM" % capacity,
                                       prompt, 8, "RAM", conversation="profile")
        if repeated != generated[:8]:
            raise AssertionError(f"profile {capacity}: RAM repeat differs from first output")
        if row["n_reused"] <= 0:
            raise AssertionError(f"profile {capacity}: RAM repeat did not report cache_n")
        rows.append(row)
        b_tokens, _, row = request_row(server, public, "profile-%d-B-after-A" % capacity,
                                       prompt_b, 16, "COLD", conversation="profile")
        rows.append(row)
        assert_profile(status(server, public), capacity, index)
    finally:
        if server:
            server.close()
    if generated is None:
        raise AssertionError("profile generation produced no IDs")

    snapshot_tokens = prompt[:-1] if index < 2 else prompt
    prompt_path = prompt_snapshot(store, len(snapshot_tokens), snapshot_tokens)
    prompt_id = snapshot_identity(prompt_path)
    # Isolate the dedicated prompt branch point for the two fresh-PID checks.
    # This is test fixture preparation in an exclusive store; production LRU
    # never deletes a valid object merely because a process stopped.
    remove_other_states(store, prompt_path)

    second = None
    try:
        second = start_server(args, root / "restart-original", preset, store)
        original, _, row = request_row(second, public, "profile-%d-restart-original-DISK" % capacity,
                                       prompt, 8, "DISK", conversation=None,
                                       expected_disk_path=prompt_path, require_disk=True)
        if row["n_reused"] <= 0:
            raise AssertionError(f"profile {capacity}: original restart had no reused tokens")
        if index < 2 and "target-only MTP bootstrap accepted" not in "".join(second.lines):
            raise AssertionError(f"profile {capacity}: restart did not bootstrap MTP before generation")
        assert_profile(status(second, public), capacity, index)
        rows.append(row)
        original_cold, _, original_cold_row = request_row(
            second, public, "profile-%d-restart-original-cold-control" % capacity,
            prompt, 8, "COLD_CONTROLLED", conversation="cold-original", cache_prompt=False)
        if original != original_cold:
            raise AssertionError(f"profile {capacity}: original-prompt disk output differs from cold control")
        if original_cold_row["n_reused"] != 0:
            raise AssertionError(f"profile {capacity}: original-prompt cold control reused cache")
        rows.append(original_cold_row)
    finally:
        if second:
            second.close()
    remove_other_states(store, prompt_path)

    third = None
    try:
        third = start_server(args, root / "restart-continuation", preset, store)
        # Keep every generated ID from the first request. The four suffix IDs
        # are an extension, not an artificial truncation of the generated run.
        resumed_prompt = prompt + generated + [7, 11, 19, 23]
        resumed, _, row = request_row(third, public,
                                      "profile-%d-restart-continue-all-IDs-DISK" % capacity,
                                      resumed_prompt, 8, "DISK", conversation=None,
                                      expected_disk_path=prompt_path, require_disk=True)
        if row["n_reused"] <= 0:
            raise AssertionError(f"profile {capacity}: continuation had no disk reuse")
        if index < 2 and "target-only MTP bootstrap accepted" not in "".join(third.lines):
            raise AssertionError(f"profile {capacity}: continuation did not bootstrap MTP")
        assert_profile(status(third, public), capacity, index)
        rows.append(row)
        cold, _, cold_row = request_row(third, public, "profile-%d-restart-continue-cold-control" % capacity,
                                        resumed_prompt, 8, "COLD_CONTROLLED", conversation="cold-control",
                                        cache_prompt=False)
        if resumed != cold:
            raise AssertionError(f"profile {capacity}: cached continuation differs from cold control")
        if cold_row["n_reused"] != 0:
            raise AssertionError(f"profile {capacity}: cold control unexpectedly reused RAM cache")
        rows.append(cold_row)
    finally:
        if third:
            third.close()
    after_id = snapshot_identity(prompt_path)
    if after_id["inode"] != prompt_id["inode"] or after_id["sha256"] != prompt_id["sha256"]:
        raise AssertionError(f"profile {capacity}: prompt snapshot identity changed across restart")
    process_roots = {}
    for label in ("first", "restart-original", "restart-continuation"):
        process_roots[label] = memory_peak(root / label)
    for row in rows:
        process_root = ("restart-original" if "restart-original" in row["label"] else
                        "restart-continuation" if "restart-continue" in row["label"] else "first")
        sample = process_roots[process_root]
        row["os_pid"] = row["pid"]
        row["pid_verified_by_sampler"] = row["pid"] in sample["observed_os_pids"]
        row["observed_os_pids"] = sample["observed_os_pids"]
        if not row["pid_verified_by_sampler"]:
            raise AssertionError(f"{row['label']}: real router PID was absent from its sampler")
        if row["origin"] == "DISK":
            row["source_pid"] = first_pid

    return {
        "profile": PROFILE_NAMES[index],
        "capacity": capacity,
        "expected_mtp": index < 2,
        "first_pid": first_pid,
        "generated_ids": len(generated),
        "rows": rows,
        "dedicated_prompt_snapshot_before": prompt_id,
        "dedicated_prompt_snapshot_after": after_id,
        "identity_preserved_path_checksum_inode": True,
        "store": str(store),
        "memory": {
            "first": memory_peak(root / "first"),
            "restart_original": memory_peak(root / "restart-original"),
            "restart_continuation": memory_peak(root / "restart-continuation"),
        },
    }


def run_disk_chain(base, args, fixture, public, contexts):
    root = base / "disk-only-chain"
    store = root / "store"
    store.mkdir(parents=True)
    preset = root / "models.ini"
    make_preset(preset, fixture, contexts, public)
    history = []
    rows = []
    prior_paths = []
    prior_pid = None
    for index, capacity in enumerate(contexts):
        target_len = capacity - 32
        prompt = list(history)
        prompt.extend(token_pattern(target_len - len(prompt), 131 + index))
        if len(prompt) != target_len:
            raise AssertionError(f"chain prompt length {len(prompt)} != {target_len}")
        before = set(canonical_states(store))
        server = None
        current_pid = None
        try:
            server = start_server(args, root / f"pid-{capacity}", preset, store)
            current_pid = server.proc.pid
            generated, _, row = request_row(server, public,
                                             f"disk-chain-{capacity}-DISK" if index else
                                             f"disk-chain-{capacity}-COLD",
                                             prompt, 16, "DISK" if index else "COLD",
                                             conversation=None,
                                             expected_disk_path=None, require_disk=bool(index))
            if index:
                row["source_pid"] = prior_pid
            if index and row["n_reused"] <= 0:
                raise AssertionError(f"disk chain {capacity}: no cache_n on disk case")
            slot = status(server, public)
            assert_profile(slot, capacity, index)
            if index == 1 and "target-only MTP bootstrap accepted" not in "".join(server.lines):
                raise AssertionError("disk chain MTP transition lacked bootstrap evidence")
            rows.append(row)
            cold, _, cold_row = request_row(server, public, f"disk-chain-{capacity}-cold-control",
                                            prompt, 8, "COLD_CONTROLLED", conversation="cold-control",
                                            cache_prompt=False)
            # Compare the same deterministic continuation path; the cold
            # control is deliberately not used to advance the chain history.
            cached_again, _, _ = request_row(server, public, f"disk-chain-{capacity}-repeat-control",
                                             prompt, 8, "RAM", conversation="chain-repeat")
            if cold != cached_again:
                raise AssertionError(f"disk chain {capacity}: cached/cold output mismatch")
            rows.append(cold_row)
        finally:
            if server:
                server.close()
        states = canonical_states(store)
        if not states:
            raise AssertionError(f"disk chain {capacity}: store empty after shutdown")
        after = set(states)
        if index and not before.intersection(after):
            raise AssertionError(f"disk chain {capacity}: prior snapshot disappeared unexpectedly")
        snapshot_tokens = prompt[:-1] if index < 2 else prompt
        prompt_state = prompt_snapshot(store, len(snapshot_tokens), snapshot_tokens)
        prior_paths.append(str(prompt_state.resolve()))
        sample = memory_peak(root / f"pid-{capacity}")
        for candidate in rows:
            if candidate["pid"] == current_pid and "disk-chain-%d" % capacity in candidate["label"]:
                candidate["os_pid"] = current_pid
                candidate["pid_verified_by_sampler"] = current_pid in sample["observed_os_pids"]
                candidate["observed_os_pids"] = sample["observed_os_pids"]
            elif candidate["pid"] == current_pid and candidate["label"] == f"disk-chain-{capacity}-cold-control":
                candidate["os_pid"] = current_pid
                candidate["pid_verified_by_sampler"] = current_pid in sample["observed_os_pids"]
                candidate["observed_os_pids"] = sample["observed_os_pids"]
        prior_pid = current_pid
        history = prompt + generated
    return {
        "kind": "disk-only",
        "profiles": [PROFILE_NAMES[i] for i in range(len(contexts))],
        "contexts": contexts,
        "rows": rows,
        "processes": sorted({row["pid"] for row in rows}),
        "prompt_snapshots": prior_paths,
        "same_store_available_after_each_restart": True,
        "store": str(store),
        "memory": {str(capacity): memory_peak(root / f"pid-{capacity}") for capacity in contexts},
    }


def xxh64(server):
    library = ctypes.CDLL(str(Path(server).resolve().parent / "libllama-server-impl.so"))
    fn = library.XXH64
    fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
    fn.restype = ctypes.c_uint64
    return fn


def rewrite_manifest(path, changes, checksum):
    manifest, raw, _ = read_manifest(path)
    length = struct.unpack("<Q", raw[-FOOTER_SIZE:-FOOTER_SIZE + 8])[0]
    payload = raw[:len(raw) - FOOTER_SIZE - length]
    updated = dict(manifest)
    updated.update(changes)
    metadata = json.dumps(updated, separators=(",", ":")).encode()
    footer = struct.pack("<QQ8s", len(metadata), checksum(metadata, len(metadata), 0), MAGIC)
    path.write_bytes(payload + metadata + footer)


def run_failure_cases(base, args, fixture, public, contexts, source_path, prompt):
    checksum = xxh64(args.server)
    rows = []
    for case in ("payload", "truncated", "model", "layout"):
        root = base / f"failure-{case}"
        store = root / "store"
        store.mkdir(parents=True)
        invalid = store / "auto-invalid.bin"
        shutil.copyfile(source_path, invalid)
        if case == "payload":
            raw = bytearray(invalid.read_bytes())
            manifest, _, _ = read_manifest(invalid)
            state_bytes = int(manifest["state_bytes"])
            raw[state_bytes - 1] ^= 1
            invalid.write_bytes(raw)
        elif case == "truncated":
            invalid.write_bytes(invalid.read_bytes()[:-1])
        elif case == "model":
            rewrite_manifest(invalid, {"model": "incompatible-model"}, checksum)
        else:
            rewrite_manifest(invalid, {"layout": "incompatible-layout"}, checksum)
        preset = root / "models.ini"
        make_preset(preset, fixture, contexts, public)
        server = None
        try:
            server = start_server(args, root / "router", preset, store)
            cached, _, row = request_row(server, public, f"failure-{case}-cold-fallback",
                                         prompt, 8, "COLD", conversation=None)
            if row["disk_restore_event"] or row["n_reused"] != 0:
                raise AssertionError(f"{case}: invalid snapshot produced a false hit: {row}")
            cold, _, cold_row = request_row(server, public, f"failure-{case}-cold-control",
                                            prompt, 8, "COLD_CONTROLLED", conversation="cold-control",
                                            cache_prompt=False)
            if cached != cold:
                raise AssertionError(f"{case}: fallback output differs from cold control")
            reason = next((line.strip() for line in server.lines
                if ("unified snapshot ignored" in line and "auto-invalid.bin" in line) or
                   ("candidate" in line and "failed native streaming validation" in line and
                    "auto-invalid.bin" in line)), None)
            if reason is None:
                raise AssertionError(f"{case}: invalid snapshot was ignored without a logged reason")
            row["fallback_reason"] = reason
            sample = memory_peak(root / "router")
            row["os_pid"] = row["pid"]
            row["pid_verified_by_sampler"] = row["pid"] in sample["observed_os_pids"]
            row["observed_os_pids"] = sample["observed_os_pids"]
            if not row["pid_verified_by_sampler"]:
                raise AssertionError(f"{case}: real router PID was absent from its sampler")
            rows.append({"case": case, "pass": True, "row": row,
                         "invalid_path": str(invalid), "cold_control": cold_row})
        finally:
            if server:
                server.close()

    root = base / "failure-write"
    store = root / "store"
    store.mkdir(parents=True)
    preset = root / "models.ini"
    make_preset(preset, fixture, contexts, public)
    server = None
    try:
        server = start_server(args, root / "router", preset, store,
                              {"LLAMA_TEST_ROUTE_STATE_PUBLISH_FAIL": "1"})
        written, _, row = request_row(server, public, "failure-write-no-publication",
                                      prompt, 8, "COLD", conversation=None)
        if row["disk_restore_event"] or row["n_reused"] != 0:
            raise AssertionError("write-failure case unexpectedly restored")
        cold, _, cold_row = request_row(server, public, "failure-write-cold-control",
                                        prompt, 8, "COLD_CONTROLLED", conversation="cold-control",
                                        cache_prompt=False)
        if written != cold:
            raise AssertionError("write-failure fallback differs from cold control")
        if canonical_states(store):
            raise AssertionError("write-failure left a visible canonical snapshot")
        if "failed safely" not in "".join(server.lines):
            raise AssertionError("write-failure did not log safe publication failure")
        sample = memory_peak(root / "router")
        row["os_pid"] = row["pid"]
        row["pid_verified_by_sampler"] = row["pid"] in sample["observed_os_pids"]
        row["observed_os_pids"] = sample["observed_os_pids"]
        if not row["pid_verified_by_sampler"]:
            raise AssertionError("write: real router PID was absent from its sampler")
        rows.append({"case": "write", "pass": True, "row": row,
                     "cold_control": cold_row, "visible_canonical_states": 0,
                     "fallback_reason": "injected publication failure"})
    finally:
        if server:
            server.close()
    return rows


def run_store_inspection_failure(base, args, fixture, public, contexts, prompt):
    root = base / "failure-store-inspection"
    store = root / "store-file"
    root.mkdir(parents=True)
    store.write_text("this is intentionally not a directory\n")
    preset = root / "models.ini"
    make_preset(preset, fixture, contexts, public)
    server = None
    try:
        try:
            server = start_server(args, root / "router", preset, store)
        except RuntimeError:
            log_path = root / "router" / "server.log"
            log = log_path.read_text() if log_path.exists() else ""
            if "not a directory" not in log:
                raise AssertionError("non-directory store failed without a closed configuration error")
            return {"case": "store-inspection", "pass": True,
                    "startup_rejected": True, "store": str(store),
                    "fallback_reason": next(line.strip() for line in log.splitlines()
                                             if "not a directory" in line)}
        output, _, row = request_row(server, public, "failure-store-inspection-cold-fallback",
                                     prompt, 8, "COLD", conversation=None)
        if row["disk_restore_event"] or row["n_reused"] != 0:
            raise AssertionError("non-directory store produced a cache hit")
        cold, _, cold_row = request_row(server, public, "failure-store-inspection-cold-control",
                                        prompt, 8, "COLD_CONTROLLED", conversation="cold-control",
                                        cache_prompt=False)
        if output != cold or cold_row["n_reused"] != 0:
            raise AssertionError("non-directory store fallback differs from cold control")
        reason = next((line.strip() for line in server.lines
                       if "unified snapshot index scan failed" in line or
                          "unified snapshot store is unavailable" in line), None)
        if reason is None:
            raise AssertionError("non-directory store failed closed without a logged reason")
        if canonical_states(store):
            raise AssertionError("non-directory store exposed a canonical snapshot")
        sample = memory_peak(root / "router")
        row["fallback_reason"] = reason
        row["os_pid"] = row["pid"]
        row["pid_verified_by_sampler"] = row["pid"] in sample["observed_os_pids"]
        row["observed_os_pids"] = sample["observed_os_pids"]
        if not row["pid_verified_by_sampler"]:
            raise AssertionError("non-directory store PID was absent from its sampler")
        return {"case": "store-inspection", "pass": True, "row": row,
                "store": str(store), "cold_control": cold_row}
    finally:
        if server:
            server.close()


def artifact_ref(path):
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "available": False}
    return {"path": str(path), "available": True, "sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--contexts", type=int, nargs=4, default=DEFAULT_CONTEXTS)
    args = parser.parse_args()
    contexts = args.contexts
    if contexts != DEFAULT_CONTEXTS:
        raise ValueError("this evidence harness requires the four scaled contexts 256/512/768/1024")
    base = Path(args.work_dir).resolve()
    base.mkdir(parents=True, exist_ok=False)
    fixture = base / "qwen35-http.gguf"
    subprocess.run([str(Path(args.server).resolve().parent / "test-state-file-stream"),
                    "--make-http-fixture", str(Path(args.model).resolve()), str(fixture)], check=True)
    public = "router-state-unified"
    results = {
        "schema": "unified-kv-cpu-a-c-v1",
        "model_id": public,
        "fixture": str(fixture),
        "contexts": contexts,
        "profiles": [],
        "disk_only_chain": None,
        "failure_cases": None,
        "reused_prior_artifacts": [
            artifact_ref("/home/hjotha/router-kv-snapshots-unificados-20260916/route-reuse-unified-3/results.json"),
            artifact_ref("/home/hjotha/router-kv-snapshots-unificados-20260916/auto-restart-unified-9/results.json"),
        ],
        "notes": [
            "CPU scaled fixtures; not GPU D evidence.",
            "DISK rows require a fresh PID and an explicit auto-restore event.",
            "TTFT is first nonempty SSE emission, not non-stream total time.",
            "transient_workspace_bytes is separate from sampled process RSS/anon.",
            "No production store or port 8090 was used.",
        ],
    }

    for index in range(4):
        results["profiles"].append(run_profile_case(base, args, fixture, public, contexts, index))
        (base / "results.json").write_text(json.dumps(results, indent=2))

    results["disk_only_chain"] = run_disk_chain(base, args, fixture, public, contexts)
    (base / "results.json").write_text(json.dumps(results, indent=2))

    source_store = Path(results["profiles"][0]["store"])
    source_prompt = token_pattern(contexts[0] - 32, 11)
    source_tokens = source_prompt[:-1]
    source_path = prompt_snapshot(source_store, len(source_tokens), source_tokens)
    results["failure_cases"] = run_failure_cases(base, args, fixture, public, contexts,
                                                  source_path, source_prompt)
    results["failure_cases"].append(run_store_inspection_failure(
        base, args, fixture, public, contexts, source_prompt))
    (base / "results.json").write_text(json.dumps(results, indent=2))
    print("PASS: CPU unified A-C profiles 256/512/768/1024, fresh-PID original/continuation disk restores, "
          "disk-only chain, deterministic cold controls, and corruption/truncation/layout/write-failure fallbacks",
          flush=True)


if __name__ == "__main__":
    main()
