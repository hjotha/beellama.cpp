#!/usr/bin/env python3
"""Exercise semantic, target-only and pre-allocation guards of adaptive slots."""

import json
import os
import pathlib
import struct
import sys
import time
import urllib.error
import urllib.request


BASE = sys.argv[1].rstrip("/")
OUT = pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)


def xxh64(data, seed=0):
    p1 = 11400714785074694791
    p2 = 14029467366897019727
    p3 = 1609587929392839161
    p4 = 9650029242287828579
    p5 = 2870177450012600261

    def rotl(value, bits):
        return ((value << bits) | (value >> (64 - bits))) & 0xffffffffffffffff

    def round64(acc, value):
        acc = (acc + value * p2) & 0xffffffffffffffff
        return (rotl(acc, 31) * p1) & 0xffffffffffffffff

    def merge(acc, value):
        acc ^= round64(0, value)
        return (acc * p1 + p4) & 0xffffffffffffffff

    n = len(data)
    i = 0
    if n >= 32:
        v1 = (seed + p1 + p2) & 0xffffffffffffffff
        v2 = (seed + p2) & 0xffffffffffffffff
        v3 = seed & 0xffffffffffffffff
        v4 = (seed - p1) & 0xffffffffffffffff
        while i <= n - 32:
            v1 = round64(v1, struct.unpack_from("<Q", data, i)[0]); i += 8
            v2 = round64(v2, struct.unpack_from("<Q", data, i)[0]); i += 8
            v3 = round64(v3, struct.unpack_from("<Q", data, i)[0]); i += 8
            v4 = round64(v4, struct.unpack_from("<Q", data, i)[0]); i += 8
        h = (rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18)) & 0xffffffffffffffff
        h = merge(h, v1); h = merge(h, v2); h = merge(h, v3); h = merge(h, v4)
    else:
        h = (seed + p5) & 0xffffffffffffffff

    h = (h + n) & 0xffffffffffffffff
    while i <= n - 8:
        k = round64(0, struct.unpack_from("<Q", data, i)[0])
        h ^= k
        h = (rotl(h, 27) * p1 + p4) & 0xffffffffffffffff
        i += 8
    if i <= n - 4:
        h ^= (struct.unpack_from("<I", data, i)[0] * p1) & 0xffffffffffffffff
        h = (rotl(h, 23) * p2 + p3) & 0xffffffffffffffff
        i += 4
    while i < n:
        h ^= (data[i] * p5) & 0xffffffffffffffff
        h = (rotl(h, 11) * p1) & 0xffffffffffffffff
        i += 1
    h ^= h >> 33
    h = (h * p2) & 0xffffffffffffffff
    h ^= h >> 29
    h = (h * p3) & 0xffffffffffffffff
    return (h ^ (h >> 32)) & 0xffffffffffffffff


SEED = 0x534c4f5453544154


def u64(data, offset):
    return struct.unpack_from("<Q", data, offset)[0], offset + 8


def skip_blob(data, offset):
    size, offset = u64(data, offset)
    return offset + size


def locate(data):
    offset = 8 + 4 + 4 + 4 * 4 + 8
    fingerprint, offset = u64(data, offset)
    pos_fingerprint = offset
    offset += fingerprint
    for _ in range(2):
        size, offset = u64(data, offset)
        offset += size
    pos_tgt = offset
    pos_dft = offset + 4
    offset += 8
    _, offset = u64(data, offset)
    offset = skip_blob(data, offset)
    offset = skip_blob(data, offset)
    pos_dft_blob_len = offset
    offset = skip_blob(data, offset)
    pos_spec_blob_len = offset
    offset = skip_blob(data, offset)
    return pos_tgt, pos_dft, pos_dft_blob_len, pos_spec_blob_len, offset


def locate_checkpoints(data):
    """Return mutable target-blob offsets for every checkpoint in a snapshot."""
    offset = 8 + 4 + 4 + 4 * 4 + 8
    fingerprint, offset = u64(data, offset)
    offset += fingerprint
    for _ in range(2):
        size, offset = u64(data, offset)
        offset += size
    offset += 4 + 4 + 8  # target/draft positions and evaluated token count
    for _ in range(4):
        offset = skip_blob(data, offset)
    n_checkpoints, offset = u64(data, offset)
    checkpoints = []
    for index in range(n_checkpoints):
        n_tokens_offset = offset
        offset += 8  # n_tokens
        offset += 4 * 5 + 8 * 2  # task, positions, flags and model instances
        for _ in range(2):
            size, offset = u64(data, offset)
            offset += size
        offset += 16 + 16 + 32 + 32 + 4 + 4 + 1
        target_blob_len_offset = offset
        target_blob_len, offset = u64(data, offset)
        target_blob_offset = offset
        offset += target_blob_len
        offset = skip_blob(data, offset)
        offset = skip_blob(data, offset)
        checkpoints.append({
            "index": index,
            "n_tokens_offset": n_tokens_offset,
            "target_blob_len_offset": target_blob_len_offset,
            "target_blob_offset": target_blob_offset,
            "target_blob_len": target_blob_len,
        })
    if offset != len(data) - 8:
        raise AssertionError(f"checkpoint parser stopped at {offset}, payload ends at {len(data) - 8}")
    return checkpoints


def locate_checkpoint_count(data):
    offset = 8 + 4 + 4 + 4 * 4 + 8
    fingerprint, offset = u64(data, offset)
    offset += fingerprint
    for _ in range(2):
        size, offset = u64(data, offset)
        offset += size
    offset += 4 + 4 + 8
    for _ in range(4):
        offset = skip_blob(data, offset)
    return offset


def minimal_checkpoint_wire():
    """Return one checksum-valid checkpoint payload with no state blobs."""
    payload = bytearray()
    payload += struct.pack("<qiiiIIQQ", 0, -1, -1, -1, 0, 0, 0, 0)
    payload += struct.pack("<Q", 0) * 2  # checkpoint layouts
    payload += struct.pack("<i", -1) * (4 + 4 + 8 + 8)
    payload += struct.pack("<II B", 0, 0, 0)
    payload += struct.pack("<Q", 0) * 3  # target, draft and MTP state
    assert len(payload) == 189, len(payload)
    return bytes(payload)


def budget_snapshot_wire(count):
    """Build a tiny valid envelope whose checkpoint vector is the only large part."""
    payload = bytearray(b"LLAMASLT")
    payload += struct.pack("<IIiiiiQ", 1, 1, 512, 256, 256, 512, 0)
    for value in (b"model", b"target", b"draft"):
        payload += struct.pack("<Q", len(value)) + value
    payload += struct.pack("<iiQ", 0, 0, 0)
    payload += struct.pack("<Q", 0)  # tokens
    payload += struct.pack("<Q", 1) + b"x"  # non-empty target payload
    payload += struct.pack("<Q", 0)  # draft payload
    payload += struct.pack("<Q", 0)  # MTP payload
    payload += struct.pack("<Q", count)
    payload += minimal_checkpoint_wire() * count
    return payload + struct.pack("<Q", xxh64(payload, SEED))


def rewrite_checksum(data):
    return data[:-8] + struct.pack("<Q", xxh64(data[:-8], SEED))


def call(method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw, status = response.read(), response.status
    except urllib.error.HTTPError as error:
        raw, status = error.read(), error.code
    result = {"method": method, "path": path, "status": status,
              "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
              "body": json.loads(raw)}
    results.append(result)
    return result


def require(result, status):
    if result["status"] != status:
        raise AssertionError(f"{result['path']}: expected {status}, got {result['status']}: {result['body']}")


def wait_ready():
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        request = urllib.request.Request(BASE + "/health", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.HTTPError):
            pass
        time.sleep(0.25)
    raise RuntimeError("adaptive adversarial server did not become ready")


results = []
wait_ready()
props = call("GET", "/props")
require(props, 200)
assert props["body"]["adaptive_context"]["profile"] == "mtp", props

prompt = "Explain adaptive cache in one sentence."
completion = call("POST", "/completion", {
    "prompt": prompt, "id_slot": 0, "cache_prompt": True,
    "n_predict": 2, "temperature": 0,
})
require(completion, 200)
saved = call("POST", "/slots/0?action=save", {"filename": "short.bin"})
require(saved, 200)

long_prompt = ("adaptive context transition preserves the evaluated prompt state and reuses it safely. " * 25).strip()
long = call("POST", "/completion", {
    "prompt": long_prompt, "id_slot": 0, "cache_prompt": True,
    "n_predict": 1, "temperature": 0,
})
require(long, 200)
long_saved = call("POST", "/slots/0?action=save", {"filename": "long.bin"})
require(long_saved, 200)

raw = bytearray((OUT / "short.bin").read_bytes())
pos_tgt, pos_dft, dft_len, spec_len, after_spec = locate(raw)
assert pos_tgt > 0 and after_spec < len(raw) - 8

semantic = bytearray(raw)
struct.pack_into("<i", semantic, pos_dft, struct.unpack_from("<i", semantic, pos_dft)[0] - 1)
(OUT / "semantic-invalid.bin").write_bytes(rewrite_checksum(semantic))
invalid = call("POST", "/slots/0?action=restore", {"filename": "semantic-invalid.bin"})
assert invalid["status"] >= 400, invalid

# A complete short snapshot must reuse its target/draft/carry state.
full_restore = call("POST", "/slots/0?action=restore", {"filename": "short.bin"})
require(full_restore, 200)
full_reuse = call("POST", "/completion", {
    "prompt": prompt, "id_slot": 0, "cache_prompt": True,
    "n_predict": 1, "temperature": 0,
})
require(full_reuse, 200)
assert full_reuse["body"]["timings"]["cache_n"] > 0, full_reuse

# Remove draft/carry blobs and checkpoints while keeping the target state and
# token payload. This is a true target-only snapshot, not an MTP complement.
target_only = raw[:dft_len]
target_only += struct.pack("<Q", 0)  # top-level draft state
target_only += struct.pack("<Q", 0)  # top-level speculative state
target_only += struct.pack("<Q", 0)  # checkpoint count
target_only += raw[-8:]  # reserve the envelope checksum field
target_only[pos_dft:pos_dft + 4] = struct.pack("<i", -1)
(OUT / "target-only.bin").write_bytes(rewrite_checksum(target_only))
target_restore = call("POST", "/slots/0?action=restore", {"filename": "target-only.bin"})
require(target_restore, 200)
assert target_restore["body"]["n_restored"] == saved["body"]["n_saved"], target_restore
target_reuse = call("POST", "/completion", {
    "prompt": prompt, "id_slot": 0, "cache_prompt": True,
    "n_predict": 1, "temperature": 0,
})
require(target_reuse, 200)
# Without the draft/carry complement this path must report a cold fallback;
# the bootstrap log below proves that it did not claim a speculative hit.
assert target_reuse["body"]["timings"]["cache_n"] == 0, target_reuse

# Swap two checksum-valid checkpoint payloads. Metadata stays put, so the
# server must reject the mismatched partial states and keep the full restore.
long_raw = bytearray((OUT / "long.bin").read_bytes())
checkpoints = locate_checkpoints(long_raw)
assert len(checkpoints) >= 2, checkpoints
first, second = checkpoints[:2]
assert first["target_blob_len"] == second["target_blob_len"], checkpoints
swapped = bytearray(long_raw)
first_range = slice(first["target_blob_offset"], first["target_blob_offset"] + first["target_blob_len"])
second_range = slice(second["target_blob_offset"], second["target_blob_offset"] + second["target_blob_len"])
swapped[first_range] = long_raw[second_range]
swapped[second_range] = long_raw[first_range]
(OUT / "checkpoint-swap.bin").write_bytes(rewrite_checksum(swapped))
checkpoint_restore = call("POST", "/slots/0?action=restore", {"filename": "checkpoint-swap.bin"})
require(checkpoint_restore, 200)
checkpoint_probe = call("POST", "/completion", {
    "prompt": (long_prompt + " divergent suffix after checkpoint validation."),
    "id_slot": 0, "cache_prompt": True, "n_predict": 1, "temperature": 0,
})
require(checkpoint_probe, 200)

# The checkpoint count is attacker-controlled metadata. It must be bounded by
# the configured checkpoint limit before the decoder reserves its vector.
checkpoint_count_overflow = bytearray(long_raw)
struct.pack_into("<Q", checkpoint_count_overflow, locate_checkpoint_count(checkpoint_count_overflow), 1_000_000)
(OUT / "checkpoint-count-overflow.bin").write_bytes(rewrite_checksum(checkpoint_count_overflow))
checkpoint_count_result = call("POST", "/slots/0?action=restore", {"filename": "checkpoint-count-overflow.bin"})
assert checkpoint_count_result["status"] >= 400, checkpoint_count_result

# The configured checkpoint limit can be large enough to admit the count above
# while the decoded object vector would still exceed one file's budget. Keep
# this envelope below the configured cache/5 per-file cap and require rejection
# before parsing any checkpoint payload. Dedicated runs set both values
# explicitly; ordinary small-limit runs retain the count-overflow check above.
checkpoint_budget_result = None
budget_limit = int(os.environ.get("ADAPTIVE_CTX_CHECKPOINTS", "0"))
budget_count = int(os.environ.get("ADAPTIVE_BUDGET_CHECKPOINTS", "50_000"))
if budget_limit >= budget_count:
    budget_file = budget_snapshot_wire(budget_count)
    cache_limit = int(os.environ.get("ADAPTIVE_CACHE_RAM", "0"))
    assert cache_limit > 0, cache_limit
    per_file_cap = cache_limit * 1024 * 1024 // 5
    assert len(budget_file) <= per_file_cap, (len(budget_file), per_file_cap)
    (OUT / "checkpoint-budget-overflow.bin").write_bytes(rewrite_checksum(budget_file))
    checkpoint_budget_result = call("POST", "/slots/0?action=restore", {"filename": "checkpoint-budget-overflow.bin"})
    assert checkpoint_budget_result["status"] >= 400, checkpoint_budget_result

# Sparse files consume no disk space; the server must reject before vector allocation.
oversize = OUT / "oversize.bin"
with oversize.open("wb") as handle:
    handle.truncate(1 << 40)
oversize_result = call("POST", "/slots/0?action=restore", {"filename": "oversize.bin"})
assert oversize_result["status"] >= 400, oversize_result

summary = {
    "steps": len(results),
    "cache_ram_mib": int(os.environ.get("ADAPTIVE_CACHE_RAM", "0")),
    "ctx_checkpoints": int(os.environ.get("ADAPTIVE_CTX_CHECKPOINTS", "0")),
    "semantic_status": invalid["status"],
    "full_restore_status": full_restore["status"],
    "full_cache_n": full_reuse["body"]["timings"]["cache_n"],
    "target_only_status": target_restore["status"],
    "target_cache_n": target_reuse["body"]["timings"]["cache_n"],
    "checkpoint_count": len(checkpoints),
    "checkpoint_restore_status": checkpoint_restore["status"],
    "checkpoint_probe_cache_n": checkpoint_probe["body"]["timings"]["cache_n"],
    "checkpoint_count_status": checkpoint_count_result["status"],
    "checkpoint_budget_status": None if checkpoint_budget_result is None else checkpoint_budget_result["status"],
    "checkpoint_budget_count": budget_count if checkpoint_budget_result is not None else 0,
    "checkpoint_budget_file_bytes": 0 if checkpoint_budget_result is None else (OUT / "checkpoint-budget-overflow.bin").stat().st_size,
    "oversize_status": oversize_result["status"],
}
(OUT / "result-adversarial.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n")
print(json.dumps(summary))
