#!/usr/bin/env python3
"""Focused probe: restore one saved KVarN state file into a fresh child.

Temporary diagnostic harness used while validating the conversion tier; runs a
single compact child (no router) with the private route-state directory set and
posts a streaming route restore for an existing candidate snapshot.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--preset", help="accepted for exclusive-window compatibility; unused")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--ctx-size", type=int, default=16384)
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    store = root / "store"
    store.mkdir(exist_ok=True)
    candidate = store / Path(args.candidate).name
    if not candidate.exists():
        shutil.copyfile(args.candidate, candidate)

    env = os.environ.copy()
    env.update(TMPDIR=str(root), LLAMA_CACHE=str(root / "cache"))
    env.update(GGML_DISABLE_VULKAN="1")
    env.update(LLAMA_SERVER_ROUTER_STATE="1", LLAMA_SERVER_ROUTER_STATE_DIR=str(store))
    command = [str(Path(args.server).resolve()),
               "--model", args.model, "--device", "CUDA0", "--n-gpu-layers", "99",
               "--ctx-size", str(args.ctx_size), "--cache-type-k", "kvarn4", "--cache-type-v", "kvarn4",
               "--batch-size", "64", "--ubatch-size", "64", "--parallel", "1",
               "--flash-attn", "on", "--fit", "off", "--cache-ram", "8192",
               "--gpu-mem-clock-decode", "10501", "--gpu-power-prefill", "200",
               "--gpu-power-decode", "170", "--gpu-power-device", "0",
               "--host", "127.0.0.1", "--port", str(args.port), "--log-verbosity", "3",
               "--slots", "--slot-save-path", str(store)]
    (root / "command.json").write_text(json.dumps(command, indent=2))
    log = root / "server.log"
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            text=True, bufsize=1, start_new_session=True)
    ready = threading.Event()

    def consume():
        with log.open("w") as output:
            for line in proc.stdout:
                output.write(line)
                output.flush()
                if "listening on http://" in line:
                    ready.set()
        ready.set()

    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    result = {}
    try:
        if not ready.wait(args.timeout) or proc.poll() is not None:
            raise RuntimeError(f"child failed to start: {log}")
        url = f"http://127.0.0.1:{args.port}"
        with urllib.request.urlopen(url + "/health", timeout=60) as response:
            result["health"] = json.load(response)
        body = json.dumps({"filename": candidate.name, "route_state_transfer": True,
                           "route_target_no_mtp": True}).encode()
        request = urllib.request.Request(url + "/slots/0?action=restore", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                result["restore"] = json.load(response)
        except urllib.error.HTTPError as error:
            result["restore_error"] = {"code": error.code, "body": error.read().decode()}
        with urllib.request.urlopen(url + "/slots", timeout=60) as response:
            result["slots"] = json.load(response)
    finally:
        result["log_tail"] = log.read_text().splitlines()[-40:] if log.exists() else []
        (root / "probe.json").write_text(json.dumps(result, indent=2))
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        reader.join(timeout=10)
    print(json.dumps({k: v for k, v in result.items() if k != "log_tail"}, indent=2))
    for line in result.get("log_tail", [])[-25:]:
        print(line)
    return 0 if "restore" in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
