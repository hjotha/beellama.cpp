#!/usr/bin/env python3
"""CPU smoke proving slot-save-auto survives a process restart and hits disk."""

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
from types import SimpleNamespace
from contextlib import contextmanager

from router_state_streaming import Server


def make_preset(path, model, max_count=16, max_mb=16384):
    path.write_text(f"""version = 1
[*]
model = {model}
override-kv = qwen35.context_length=int:1024
load-mode = none
parallel = 1
device = none
gpu-layers = 0
flash-attn = on
fit = off
cache-prompt = true
context-shift = false
metrics = true
cache-type-k = q4_0
cache-type-v = q4_0
slot-save-auto = true
slot-save-block = 1
slot-save-max-count = {max_count}
slot-save-max-mb = {max_mb}
[fixed]
load-on-startup = true
ctx-size = 1024
spec-type = none
batch-size = 256
ubatch-size = 256
""")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def store_read_lock(store):
    path = store / ".llama-kv-snapshot-store.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def canonical_states(store):
    # Hold the store read lock while listing and reading manifests; this keeps
    # the integration evidence from racing a publisher's atomic commit.
    with store_read_lock(store):
        states = sorted(store.glob("auto-*.bin"))
        for state in states:
            with state.open("rb") as source:
                source.seek(-24, 2)
                length, _, magic = struct.unpack("<QQ8s", source.read(24))
                if magic != b"LLROUTE1" or length <= 0 or length > 64 * 1024:
                    raise AssertionError(f"non-canonical auto snapshot: {state}")
                source.seek(-24 - length, 2)
                manifest = json.loads(source.read(length))
                if manifest.get("version") != 1 or manifest.get("index_block", 0) <= 0:
                    raise AssertionError(f"invalid canonical manifest: {state}")
        return states


def xxh64_library(server):
    library = ctypes.CDLL(str(Path(server).resolve().parent / "libllama-server-impl.so"))
    checksum = library.XXH64
    checksum.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
    checksum.restype = ctypes.c_uint64
    return checksum


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


def snapshot_stripe(path, checksum):
    key = str(Path(path).resolve()).encode()
    return int(checksum(key, len(key), 0) % 64)


def hold_reference_stripes(store, paths, checksum):
    fds = []
    stripes = sorted({snapshot_stripe(path, checksum) for path in paths})
    try:
        for stripe in stripes:
            lock_path = store / f".llama-kv-snapshot-ref-{stripe}.ref"
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
            fd = os.open(lock_path, flags, 0o600)
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fds.append(fd)
        return fds
    except Exception:
        for fd in fds:
            os.close(fd)
        raise


def release_reference_stripes(fds):
    for fd in fds:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def completion_request(server, prompt, cache=True, n_predict=8, conversation="matrix"):
    return server.request("/completion", {
        "model": "fixed", "prompt": prompt, "n_predict": n_predict,
        "cache_prompt": cache, "return_tokens": True,
        "temperature": 0, "seed": 1234, "ignore_eos": True}, conversation=conversation)


def choose_unprotected_prompt(server, store, checksum, salt, protected_stripes, start):
    # Cold probes do not save because cache_prompt=false. They reveal the
    # deterministic response so both the prompt checkpoint and completion
    # snapshot can be selected away from protected lock stripes.
    for offset in range(64):
        prompt = [3 + ((start + offset) * 13 + i * 17) % 50 for i in range(400)]
        response = completion_request(server, prompt, cache=False)
        generated = response.get("tokens", [])
        candidates = [prompt, prompt + generated]
        if all(snapshot_stripe(store / auto_state_name(salt, tokens), checksum) not in protected_stripes
               for tokens in candidates):
            return prompt, generated
    raise AssertionError("could not choose a prompt outside protected lock stripes")


def assert_budget_rejection(server, store, checksum, salt, max_count, max_bytes, label, seed):
    before_paths = canonical_states(store)
    if not before_paths:
        raise AssertionError(f"{label}: no canonical snapshots before protection")
    before = {path: file_sha256(path) for path in before_paths}
    fds = hold_reference_stripes(store, before_paths, checksum)
    try:
        protected = {snapshot_stripe(path, checksum) for path in before_paths}
        prompt, generated = choose_unprotected_prompt(server, store, checksum, salt, protected, seed)
        response = completion_request(server, prompt, cache=True)
        if len(response.get("tokens", [])) != 8:
            raise AssertionError(f"{label}: protected-budget request failed: {response}")
        after_paths = canonical_states(store)
        after = {path: file_sha256(path) for path in after_paths}
        if set(after) != set(before) or after != before:
            raise AssertionError(f"{label}: protected snapshots changed or new residue remained; "
                                 f"before={sorted(path.name for path in before_paths)} "
                                 f"after={sorted(path.name for path in after_paths)} "
                                 f"protected_stripes={sorted(protected)}")
        if any(path.name.endswith(".meta") for path in store.iterdir()):
            raise AssertionError(f"{label}: legacy metadata residue remained")
        if max_count and len(after_paths) > max_count:
            raise AssertionError(f"{label}: count budget exceeded while protected: {len(after_paths)}")
        if max_bytes and sum(path.stat().st_size for path in after_paths) > max_bytes:
            raise AssertionError(f"{label}: byte budget exceeded while protected")
        log = "".join(server.lines)
        if "router snapshot store budget rejected publication" not in log:
            raise AssertionError(f"{label}: no explicit protected-budget rejection in log")
    finally:
        release_reference_stripes(fds)

    # Once readers release their kernel locks, the real LRU must be able to
    # remove an old entry on the next publication.
    follow_prompt = [11 + (i * 23) % 50 for i in range(400)]
    follow = completion_request(server, follow_prompt, cache=True)
    if len(follow.get("tokens", [])) != 8:
        raise AssertionError(f"{label}: post-release request failed: {follow}")
    released_paths = canonical_states(store)
    total = sum(path.stat().st_size for path in released_paths)
    if max_count and len(released_paths) > max_count:
        raise AssertionError(f"{label}: LRU count budget failed after release: {len(released_paths)}")
    if max_bytes and total > max_bytes:
        raise AssertionError(f"{label}: LRU byte budget failed after release: {total}")
    if set(released_paths) == set(before_paths):
        raise AssertionError(f"{label}: no protected candidate was evicted after release")
    return {"label": label, "before": len(before_paths), "after_rejection": len(after_paths),
            "after_release": len(released_paths), "bytes_after_release": total,
            "response_tokens": len(response.get("tokens", [])),
            "post_release_tokens": len(follow.get("tokens", [])),
            "prompt_probe_tokens": len(generated)}


def run_budget_case(base, args, checksum, label, max_count, max_mb, seed):
    root = base / label
    root.mkdir()
    store = root / "store"
    store.mkdir()
    first_root = root / "first"
    first_root.mkdir()
    model = str(Path(args.model).resolve())
    fixture = first_root / "qwen35-http.gguf"
    subprocess.run([str(Path(args.server).resolve().parent / "test-state-file-stream"),
                    "--make-http-fixture", model, str(fixture)], check=True)
    preset = first_root / "models.ini"
    make_preset(preset, fixture, max_count=max_count, max_mb=max_mb)
    server_args = SimpleNamespace(server=args.server, gpu=False, port=0,
                                  timeout=args.timeout, slot_save_path=str(store))
    server = Server(server_args, root, preset)
    try:
        if max_count:
            response = completion_request(server, [3 + (i * 17) % 50 for i in range(400)], n_predict=8)
            if len(response.get("tokens", [])) != 8:
                raise AssertionError(f"{label}: seed request failed: {response}")
        else:
            for index in range(6):
                prompt = [3 + ((index + 5) * 11 + i * 17) % 50 for i in range(400)]
                response = completion_request(server, prompt, n_predict=8)
                if len(response.get("tokens", [])) != 8:
                    raise AssertionError(f"{label}: seed request {index} failed: {response}")
        states = canonical_states(store)
        salt = int(states[0].name.split("-")[1], 16)
        return assert_budget_rejection(server, store, checksum, salt, max_count,
                max_mb * 1024 * 1024 if max_mb else 0, label, seed)
    finally:
        server.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    base = Path(args.work_dir).resolve()
    base.mkdir(parents=True, exist_ok=False)
    store = base / "persistent-store"
    store.mkdir()
    first_root = base / "first"
    second_root = base / "second"
    third_root = base / "third"
    model = str(Path(args.model).resolve())
    fixture = first_root / "qwen35-http.gguf"
    first_root.mkdir()
    subprocess.run([str(Path(args.server).resolve().parent / "test-state-file-stream"),
                    "--make-http-fixture", model, str(fixture)], check=True)

    prompt = [3 + (i * 17) % 50 for i in range(400)]
    server_args = SimpleNamespace(server=args.server, gpu=False, port=0,
                                  timeout=args.timeout, slot_save_path=str(store))
    first_preset = first_root / "models.ini"
    make_preset(first_preset, fixture)
    first = Server(server_args, first_root, first_preset)
    try:
        response = first.request("/completion", {
            "model": "fixed", "prompt": prompt, "n_predict": 32,
            "cache_prompt": True, "return_tokens": True,
            "temperature": 0, "seed": 1234, "ignore_eos": True})
        generated = response.get("tokens", [])
        if len(generated) != 32:
            raise AssertionError(f"first generation length: {len(generated)}")
    finally:
        first.close()

    states = canonical_states(store)
    metadata = sorted(store.glob("auto-*.bin.meta"))
    if not states or metadata:
        raise AssertionError(f"first process did not publish auto cache: {states} {metadata}")

    # Keep only the dedicated prompt checkpoint before starting the new process;
    # this proves the original prompt hit does not depend on a snapshot that also
    # contains the first process's generated response.
    salt = int(states[0].name.split("-")[1], 16)
    prompt_state = store / auto_state_name(salt, prompt)
    if not prompt_state.exists():
        raise AssertionError(f"dedicated prompt checkpoint was not published: {prompt_state}")
    prompt_inode = prompt_state.stat().st_ino
    for state in states:
        if state != prompt_state:
            state.unlink()
            sidecar = Path(str(state) + ".logits")
            if sidecar.exists():
                sidecar.unlink()

    second_root.mkdir()
    second_preset = second_root / "models.ini"
    make_preset(second_preset, fixture)
    second = Server(server_args, second_root, second_preset)
    try:
        original = completion_request(second, prompt, n_predict=8, conversation=None)
        if len(original.get("tokens", [])) != 8 or original.get("model") != "fixed":
            raise AssertionError(f"original-prompt restart response: {original}")
        if original.get("timings", {}).get("cache_n", 0) <= 0:
            raise AssertionError(f"original prompt did not report a disk cache hit: {original}")
        original_log = "".join(second.lines)
        if "auto-restore: reused" not in original_log or prompt_state.name not in original_log:
            raise AssertionError("restart did not restore the dedicated original-prompt snapshot")
    finally:
        second.close()

    # A separate PID must prove continuation after restart. Remove snapshots
    # created by the repeat request so this case is forced to consume the
    # dedicated original-prompt checkpoint, then append every generated ID from
    # the first process (no artificial last-token truncation).
    for state in canonical_states(store):
        if state != prompt_state:
            state.unlink()
            sidecar = Path(str(state) + ".logits")
            if sidecar.exists():
                sidecar.unlink()
    third_root.mkdir()
    third_preset = third_root / "models.ini"
    make_preset(third_preset, fixture)
    third = Server(server_args, third_root, third_preset)
    try:
        resumed_prompt = prompt + generated + [7, 11, 19, 23] * 5
        if resumed_prompt[len(prompt):len(prompt) + len(generated)] != generated:
            raise AssertionError("continuation fixture did not preserve all generated IDs")
        resumed = completion_request(third, resumed_prompt, n_predict=8, conversation=None)
        if len(resumed.get("tokens", [])) != 8 or resumed.get("model") != "fixed":
            raise AssertionError(f"restart continuation response: {resumed}")
        if resumed.get("timings", {}).get("cache_n", 0) <= 0:
            raise AssertionError(f"restart continuation did not report a disk cache hit: {resumed}")
        log = "".join(third.lines)
        if "auto-restore: reused" not in log or prompt_state.name not in log:
            raise AssertionError("separate restart continuation has no dedicated disk restore event")
        if prompt_state.stat().st_ino != prompt_inode:
            raise AssertionError("restart replaced the reused prompt snapshot inode")
        result = {"auto_states": len(states), "auto_meta": len(metadata),
                  "dedicated_prompt_snapshot": prompt_state.name,
                  "original_restart_cache_n": original["timings"]["cache_n"],
                  "original_restart_prompt_n": original["timings"]["prompt_n"],
                  "restart_continuation_cache_n": resumed["timings"]["cache_n"],
                  "restart_continuation_prompt_n": resumed["timings"]["prompt_n"],
                  "restart_continuation_generated_ids": len(generated),
                  "conversation_header_for_auto_cases": False,
                  "prompt_snapshot_inode": prompt_inode,
                  "store": str(store)}
    finally:
        third.close()

    (base / "results.json").write_text(json.dumps(result, indent=2))
    print("PASS: slot-save-auto reload hit after process restart")

    checksum = xxh64_library(args.server)
    result["budget_count"] = run_budget_case(base, args, checksum, "budget-count", 1, 16384, 900)
    result["budget_bytes"] = run_budget_case(base, args, checksum, "budget-bytes", 0, 1, 1200)
    (base / "results.json").write_text(json.dumps(result, indent=2))
    print("PASS: unified LRU rejects protected count/byte publications and evicts after release")


if __name__ == "__main__":
    main()
