#!/usr/bin/env python3
"""Exercise adaptive slot save/restore while the automatic RAM cache is near its cap."""

import json
import pathlib
import sys
import time
import urllib.error
import urllib.request


BASE = sys.argv[1].rstrip("/")
OUT = pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)


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


def wait_ready():
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            if call("GET", "/health")["status"] == 200:
                return
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(0.25)
    raise RuntimeError("adaptive budget server did not become ready")


def require(result, status):
    if result["status"] != status:
        raise AssertionError(f"{result['path']}: expected {status}, got {result['status']}: {result['body']}")


results = []
wait_ready()
props = call("GET", "/props")
require(props, 200)
assert props["body"]["adaptive_context"]["profile"] == "mtp", props

last_prompt = ""
for index in range(8):
    last_prompt = (
        f"cache fill entry {index}: preserve independent automatic cache state before adaptive slot save. "
        "The checkpoint rich request must remain valid across profile transitions. " * 18
    ).strip()
    completion = call("POST", "/completion", {
        "prompt": last_prompt, "id_slot": 0, "cache_prompt": True,
        "n_predict": 1, "temperature": 0,
    })
    require(completion, 200)

filled = call("GET", "/props")
require(filled, 200)
assert filled["body"]["adaptive_context"]["state"] == "ready", filled

save = call("POST", "/slots/0?action=save", {"filename": "near-limit.bin"})
if save["status"] != 200 and save["status"] < 400:
    raise AssertionError(f"unexpected save status: {save}")
if save["status"] == 200:
    restore = call("POST", "/slots/0?action=restore", {"filename": "near-limit.bin"})
    require(restore, 200)
else:
    restore = {"status": None, "body": {}}

after = call("GET", "/props")
require(after, 200)
assert after["body"]["adaptive_context"]["state"] == "ready", after
probe = call("POST", "/completion", {
    "prompt": last_prompt, "id_slot": 0, "cache_prompt": True,
    "n_predict": 1, "temperature": 0,
})
require(probe, 200)

(OUT / "result-budget.json").write_text(json.dumps({"results": results}, indent=2) + "\n")
print(json.dumps({
    "fill_steps": 8,
    "save_status": save["status"],
    "restore_status": restore["status"],
    "probe_status": probe["status"],
    "cache_eviction_expected": True,
}))
