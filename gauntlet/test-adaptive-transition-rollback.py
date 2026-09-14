#!/usr/bin/env python3
"""Exercise server-side adaptive transition rollback with deterministic test faults."""

import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SERVER = Path(os.environ.get("ADAPTIVE_SERVER", ROOT / "build/bin/llama-server"))
MODEL = Path(os.environ.get("ADAPTIVE_MODEL", ROOT / "gauntlet/Qwen3.5-4B-MTP-Q4_K_M.gguf"))
OUT = ROOT / "gauntlet"
SHORT_CTX = 256
LONG_CTX = 512
# llama_context rounds each per-sequence context up to a 256-token boundary.
# Keep this fault harness fast: the threshold is deliberately one token; the
# separate 4B transition evidence covers the normal 256-token threshold.
MTP_LIMIT = 1
LONG_PROMPT = "hi"
SHORT_PROMPT = ""
GPU_ARGS = ["--gpu-layers", "99", "--device", "CUDA0"] if os.environ.get("ADAPTIVE_USE_GPU") == "1" else [
    "--gpu-layers", "0", "--device", "none",
]


def http(port, method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def wait_ready(port):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            status, _ = http(port, "GET", "/health")
            if status == 200:
                return
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(0.25)
    raise RuntimeError(f"server on port {port} did not become ready")


def completion(port, prompt, n_predict=1):
    return http(port, "POST", "/completion", {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": False,
    })


def assert_status(body, expected_profile, expected_state, expected_ctx, expected_resident=None, require_enabled=True):
    status = body["adaptive_context"]
    if require_enabled:
        assert status["enabled"] is True, status
    assert status["profile"] == expected_profile, status
    assert status["state"] == expected_state, status
    assert status["context_size"] == expected_ctx, status
    assert status["context_size_long"] == LONG_CTX, status
    if expected_resident is not None:
        assert status["mtp_weights_resident"] is expected_resident, status


def run_case(name, fault, steps, markers=()):
    port = 19200 + run_case.index
    run_case.index += 1
    log_path = OUT / f"adaptive-transition-rollback-http-{name}.log"
    env = os.environ.copy()
    if fault:
        env["LLAMA_TEST_ADAPTIVE_TRANSITION_FAIL"] = fault
    else:
        env.pop("LLAMA_TEST_ADAPTIVE_TRANSITION_FAIL", None)
    cmd = [
        str(SERVER), "-m", str(MODEL),
        "--host", "127.0.0.1", "--port", str(port),
        "-c", str(LONG_CTX), "--ctx-size-mtp", str(SHORT_CTX), "--mtp-max-tokens", str(MTP_LIMIT),
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-p-min", "0",
        "--n-predict", "1", *GPU_ARGS, "--fit", "off",
        "--load-mode", "none", "-b", "16", "-ub", "16", "--parallel", "1",
        "--cache-ram", "0", "--no-warmup", "--slots",
    ]
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(port)
        result = {"fault": fault, "port": port, "steps": []}
        for step in steps:
            kind = step["kind"]
            if kind == "props":
                status, body = http(port, "GET", "/props")
            elif kind == "models":
                status, body = http(port, "GET", "/models")
            elif kind == "slots":
                status, body = http(port, "GET", "/slots")
            elif kind == "long":
                status, body = completion(port, LONG_PROMPT, step.get("n_predict", 1))
            elif kind == "short":
                status, body = completion(port, SHORT_PROMPT, step.get("n_predict", 0))
            else:
                raise AssertionError(kind)
            result["steps"].append({"kind": kind, "status": status, "body": body})
            if "status" in step:
                assert status == step["status"], (name, kind, status, body)
            if kind == "models" and step.get("adaptive", False):
                assert_status(body["data"][0], step["profile"], step["state"], step["ctx"], step.get("resident"))
            if kind == "slots" and step.get("adaptive", False):
                assert_status(body[0], step["profile"], step["state"], step["ctx"], step.get("resident"))
            if kind not in {"models", "slots"} and "profile" in step:
                assert_status(body, step["profile"], step["state"], step["ctx"], step.get("resident"))
        if markers:
            log_text = log_path.read_text()
            for marker in markers:
                assert marker in log_text, (name, marker, log_path)
        return result
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


run_case.index = 0


cases = [
    ("long-rollback", "long", [
        {"kind": "props", "profile": "mtp", "state": "ready", "ctx": SHORT_CTX, "resident": True},
        {"kind": "models", "adaptive": True, "profile": "mtp", "state": "ready", "ctx": SHORT_CTX, "resident": True},
        {"kind": "slots", "adaptive": True, "profile": "mtp", "state": "ready", "ctx": SHORT_CTX, "resident": True},
        {"kind": "long", "status": 500},
        {"kind": "props", "profile": "mtp", "state": "ready", "ctx": SHORT_CTX, "resident": True},
        {"kind": "short", "status": 200},
    ]),
    ("mtp-rollback", "mtp", [
        {"kind": "long", "status": 200},
        {"kind": "props", "profile": "long", "state": "ready", "ctx": LONG_CTX, "resident": False},
        {"kind": "short", "status": 500},
        {"kind": "props", "profile": "long", "state": "ready", "ctx": LONG_CTX, "resident": False},
        {"kind": "long", "status": 200},
    ]),
    ("long-unavailable", "long+rollback", [
        {"kind": "long", "status": 500},
        {"kind": "props", "profile": "none", "state": "unavailable", "ctx": 0, "resident": True},
        {"kind": "models", "adaptive": True, "profile": "none", "state": "unavailable", "ctx": 0, "resident": True},
        {"kind": "slots", "adaptive": True, "profile": "none", "state": "unavailable", "ctx": 0, "resident": True},
        {"kind": "short", "status": 503},
    ]),
    ("long-unavailable-context", "long+rollback-context", [
        {"kind": "long", "status": 500},
        {"kind": "props", "profile": "none", "state": "unavailable", "ctx": 0, "resident": True},
        {"kind": "models", "adaptive": True, "profile": "none", "state": "unavailable", "ctx": 0, "resident": True},
        {"kind": "slots", "adaptive": True, "profile": "none", "state": "unavailable", "ctx": 0, "resident": True},
        {"kind": "short", "status": 503},
    ], ("rollback after context recreation",)),
    ("mtp-unavailable", "mtp+rollback", [
        {"kind": "long", "status": 200},
        {"kind": "short", "status": 500},
        {"kind": "props", "profile": "none", "state": "unavailable", "ctx": 0, "resident": False},
        {"kind": "models", "adaptive": True, "profile": "none", "state": "unavailable", "ctx": 0, "resident": False},
        {"kind": "short", "status": 503},
    ]),
]


all_results = []
for case in cases:
    all_results.append(run_case(*case))

(OUT / "adaptive-transition-rollback-http.json").write_text(
    json.dumps({"cases": all_results}, indent=2) + "\n"
)
print(json.dumps({"cases": [
    {"fault": item["fault"], "steps": [(s["kind"], s["status"]) for s in item["steps"]]}
    for item in all_results
]}, indent=2))
