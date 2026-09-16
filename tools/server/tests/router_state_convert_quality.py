#!/usr/bin/env python3
"""Real-model quality probe: q4 prefix vs converted KVarN prefix vs native KVarN.

Runs inside the exclusive GPU window (router_state_gpu.py --harness-raw). It
builds a two-member route group (wide q4 capped, kvarn4 uncapped) and compares
the next-token probability distribution after the same 8192-token prefix:

  1. direct cold q4 prefill on the wide child
  2. router hand-off: a small pinned request, then the long prompt converts the
     wide q4_0 prefix into the KVarN child
  3. cold native KVarN prefill on the kvarn child after erasing the slot

Reports KL divergences and maximum probability deltas. This is a lossy
conversion; no equality with the native path is expected or required.
"""

import argparse
import json
import math
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


class Child:
    def __init__(self, args, root, preset):
        self.root = root
        self.log = root / "server.log"
        self.lines = []
        self.ready = threading.Event()
        env = dict(__import__("os").environ)
        env.update(TMPDIR=str(root), LLAMA_CACHE=str(root / "cache"), GGML_DISABLE_VULKAN="1")
        self.port = args.port
        if not self.port:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                self.port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        command = [str(Path(args.server).resolve()), "--models-preset", str(preset), "--models-max", "1",
                   "--host", "127.0.0.1", "--port", str(self.port), "--log-verbosity", "3"]
        (root / "command.json").write_text(json.dumps(command, indent=2))
        self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                     text=True, bufsize=1, start_new_session=True)

        def consume():
            with self.log.open("w") as output:
                for line in self.proc.stdout:
                    output.write(line)
                    output.flush()
                    self.lines.append(line)
                    if "listening on http://" in line:
                        self.ready.set()
            self.ready.set()

        self.reader = threading.Thread(target=consume, daemon=True)
        self.reader.start()
        if not self.ready.wait(args.timeout) or self.proc.poll() is not None:
            self.close()
            raise RuntimeError(f"router failed to start: {self.log}")

    def request(self, path, body=None, conversation=None, timeout=None):
        headers = {"Content-Type": "application/json"}
        if conversation is not None:
            headers["X-Conversation-Id"] = conversation
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout or 1800) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{path}: HTTP {error.code}: {error.read().decode()}") from error

    def child_port(self, name):
        for line in reversed(self.lines):
            match = re.search(rf"spawning server instance with name={re.escape(name)} on port (\d+)", line)
            if match:
                return int(match.group(1))
        raise AssertionError(f"child port for {name} was not logged")

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                import os
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.reader.join(timeout=10)


def first_divergence(a, b):
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--pin-tokens", type=int, default=1024)
    parser.add_argument("--probe-tokens", type=int, default=64)
    args = parser.parse_args()

    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    public = args.model
    results = {}

    router = None
    try:
        router = Child(args, root, Path(args.preset))
        router.request("/health", timeout=60)
        tokenized = router.request("/tokenize", {"model": public,
            "content": "The river is blue. The sky is clear. The mountain is tall. The road is long.",
            "add_special": True})["tokens"]

        def ids(count):
            return [tokenized[i % len(tokenized)] for i in range(count)]

        prefix = ids(args.prefix_tokens)
        pin = ids(args.pin_tokens)
        body = {"prompt": prefix, "n_predict": args.probe_tokens, "temperature": 0, "seed": 1234,
                "ignore_eos": True, "return_tokens": True}

        # 1) direct cold q4 prefill on the wide child.
        wide_port = router.child_port("wide")
        wide_url = f"http://127.0.0.1:{wide_port}"
        started = time.monotonic()
        with urllib.request.urlopen(urllib.request.Request(wide_url + "/completion",
                data=json.dumps({"model": public, **body}).encode(),
                headers={"Content-Type": "application/json"}), timeout=args.timeout) as response:
            q4 = json.load(response)
        results["q4_prefill_and_probe_seconds"] = time.monotonic() - started
        q4_tokens = q4.get("tokens") or []

        # 2) router conversion: pin the wide tier, then cross the cap.
        router.request("/completion", {"model": public, "prompt": pin, "n_predict": 1,
            "temperature": 0, "seed": 1234, "ignore_eos": True, "return_tokens": True},
            conversation="quality")
        started = time.monotonic()
        converted = router.request("/completion", {"model": public, **body}, conversation="quality")
        results["converted_request_seconds"] = time.monotonic() - started
        converted_tokens = converted.get("tokens") or []
        results["converted_cache_n"] = converted.get("timings", {}).get("cache_n")
        results["converted_prompt_n"] = converted.get("timings", {}).get("prompt_n")

        # 3) cold native KVarN prefill on the same child after erasing the slot.
        kvarn_port = router.child_port("kvarn")
        kvarn_url = f"http://127.0.0.1:{kvarn_port}"
        with urllib.request.urlopen(urllib.request.Request(
                kvarn_url + "/slots/0?action=erase",
                data=json.dumps({"filename": "quality-erase"}).encode(),
                headers={"Content-Type": "application/json"}), timeout=args.timeout) as response:
            response.read()
        started = time.monotonic()
        with urllib.request.urlopen(urllib.request.Request(kvarn_url + "/completion",
                data=json.dumps({"model": public, **body}).encode(),
                headers={"Content-Type": "application/json"}), timeout=args.timeout) as response:
            native = json.load(response)
        results["native_prefill_and_probe_seconds"] = time.monotonic() - started
        native_tokens = native.get("tokens") or []

        results["probe_tokens"] = args.probe_tokens
        results["q4_tokens"] = q4_tokens
        results["converted_tokens"] = converted_tokens
        results["native_tokens"] = native_tokens
        results["first_divergence_q4_converted"] = first_divergence(q4_tokens, converted_tokens)
        results["first_divergence_q4_native"] = first_divergence(q4_tokens, native_tokens)
        results["first_divergence_native_converted"] = first_divergence(native_tokens, converted_tokens)
        results["tokens_equal_all"] = q4_tokens == converted_tokens == native_tokens
        results["prefix_tokens"] = args.prefix_tokens
        (root / "quality.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        return 0
    finally:
        if router is not None:
            router.close()


if __name__ == "__main__":
    raise SystemExit(main())
