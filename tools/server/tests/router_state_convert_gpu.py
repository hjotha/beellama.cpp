#!/usr/bin/env python3
"""GPU router test for the q4_0 -> KVarN conversion tier on the real model.

Runs the updated multi-tier preset (tri q4/MTP -> wide q4 capped -> KVarN4
uncapped) on port 8091 and drives one conversation up the ladder:

  1. 97520 tokens + small output on the tri tier (cold prefill)
  2. 100352 tokens + 4096 on the wide tier (RAM hand-off, bounded suffix)
  3. 110592 tokens + up to 4096 on the KVarN tier: the wide q4_0 prefix is
     converted instead of re-prefilled (this is the target of the test)
  4. repeat of (3): converted/auto snapshot reuse without conversion
  5. cold 110592-token prefill on the KVarN tier for the latency comparison
  6. router restart: the persisted converted snapshot restores from disk

Must be launched through router_state_gpu.py --harness-raw so production is
stopped first and restored in finally. Writes results.json, rows and memory
samples under --work-dir.
"""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


class Server:
    def __init__(self, args, root, preset, store):
        self.root = root
        self.log = root / "server.log"
        self.lines = []
        self.ready = threading.Event()
        env = os.environ.copy()
        env.update(TMPDIR=str(root), LLAMA_CACHE=str(root / "cache"))
        env.update(GGML_DISABLE_VULKAN="1")
        self.port = args.port
        if not self.port:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                self.port = sock.getsockname()[1]
        if self.port == 8090:
            raise ValueError("8090 is reserved for production")
        self.url = f"http://127.0.0.1:{self.port}"
        command = [str(Path(args.server).resolve()), "--models-preset", str(preset), "--models-max", "1",
                   "--host", "127.0.0.1", "--port", str(self.port), "--log-verbosity", "3"]
        (root / "command.json").write_text(json.dumps(command, indent=2))
        self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                     text=True, bufsize=1, start_new_session=True)

        def consume():
            with self.log.open("w") as log:
                for line in self.proc.stdout:
                    log.write(line)
                    log.flush()
                    self.lines.append(line)
                    if "listening on http://" in line or "server is listening on" in line:
                        self.ready.set()
            self.ready.set()

        self.reader = threading.Thread(target=consume, daemon=True)
        self.reader.start()
        self.sampling_done = threading.Event()

        def sample_memory():
            with (root / "memory.jsonl").open("w") as output:
                while not self.sampling_done.is_set():
                    queue = [self.proc.pid]
                    seen = set()
                    processes = []
                    while queue:
                        pid = queue.pop()
                        if pid in seen:
                            continue
                        seen.add(pid)
                        try:
                            status = Path(f"/proc/{pid}/status").read_text()
                            values = {key: int(re.search(rf"^{key}:\s+(\d+)", status, re.M).group(1))
                                      for key in ("VmRSS", "VmHWM", "RssAnon", "RssFile")}
                            processes.append(dict(pid=pid, **values))
                        except (OSError, AttributeError):
                            pass
                        try:
                            for tid in Path(f"/proc/{pid}/task").iterdir():
                                try:
                                    queue.extend(int(value) for value in (tid / "children").read_text().split())
                                except (OSError, ValueError):
                                    pass
                        except OSError:
                            pass
                    output.write(json.dumps({"time": time.time(), "processes_kib": processes}) + "\n")
                    output.flush()
                    self.sampling_done.wait(0.5)

        self.memory_reader = threading.Thread(target=sample_memory, daemon=True)
        self.memory_reader.start()
        if not self.ready.wait(args.timeout) or self.proc.poll() is not None:
            self.close()
            raise RuntimeError(f"router failed to start: {self.log}")
        self.timeout = args.timeout

    def request(self, path, body=None, conversation=None, timeout=None):
        headers = {"Content-Type": "application/json"}
        if conversation is not None:
            headers["X-Conversation-Id"] = conversation
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{path}: HTTP {error.code}: {error.read().decode()}") from error

    def log_has(self, needle):
        return any(needle in line for line in self.lines)

    def log_count(self, needle):
        return sum(1 for line in self.lines if needle in line)

    def close(self):
        self.sampling_done.set()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.reader.join(timeout=10)
        self.memory_reader.join(timeout=5)


def read_manifest(path):
    with open(path, "rb") as file:
        file.seek(-24, 2)
        length, _, magic = struct.unpack("<QQ8s", file.read(24))
        assert magic == b"LLROUTE1", magic
        file.seek(-24 - length, 2)
        return json.loads(file.read(length))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--preset", required=True, help="candidate preset with the private store")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--output-tokens", type=int, default=4096)
    parser.add_argument("--contexts", type=int, nargs=3, default=[97536, 104448, 114688])
    parser.add_argument("--skip-cold", action="store_true")
    args = parser.parse_args()

    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    public = args.model
    tri_ctx, wide_ctx, kvarn_ctx = args.contexts
    out_tokens = args.output_tokens
    rows = []

    def persist():
        (root / "results.json").write_text(json.dumps(
            {"contexts": args.contexts, "output_tokens": out_tokens, "rows": rows}, indent=2))

    server = None
    try:
        server = Server(args, root, Path(args.preset), None)
        server.request("/health", timeout=60)
        models = server.request("/v1/models")
        listed = models.get("models", [])
        assert len(listed) == 1 and listed[0]["name"] == public, listed

        def completion(label, prompt, n, conversation, timeout=None, want_hit=None):
            started = time.monotonic()
            body = server.request("/completion", {"model": public, "prompt": prompt, "n_predict": n,
                "id_slot": 0, "cache_prompt": True, "return_tokens": True, "temperature": 0,
                "seed": 1234, "ignore_eos": True}, conversation=conversation, timeout=timeout)
            tokens = body.get("tokens", [])
            timings = body.get("timings", {})
            row = {"label": label, "input": len(prompt), "requested_output": n, "output": len(tokens),
                   "model": body.get("model"), "cache_n": timings.get("cache_n"),
                   "prompt_n": timings.get("prompt_n"), "prompt_ms": timings.get("prompt_ms"),
                   "predicted_n": timings.get("predicted_n"), "predicted_ms": timings.get("predicted_ms"),
                   "stop_type": body.get("stop_type"), "stop_detail": body.get("stop_detail"),
                   "wall_seconds": time.monotonic() - started}
            rows.append(row)
            persist()
            print(json.dumps(row), flush=True)
            assert body.get("model") == public, row
            if want_hit is True:
                assert timings.get("cache_n", 0) > 0, row
            if want_hit is False:
                assert timings.get("cache_n", 0) == 0, row
            return tokens

        # Real tokenizer output avoids reasoning-preserve parse failures that
        # arbitrary token IDs can trigger on this chat template.
        tokenized = server.request("/tokenize", {"model": public,
            "content": "The river is blue. The sky is clear. The mountain is tall. The road is long.",
            "add_special": True})["tokens"]
        assert tokenized, "tokenizer returned no tokens"
        cold_base = server.request("/tokenize", {"model": public,
            "content": "A quiet morning follows a long night. The harbor is calm and the gulls are loud.",
            "add_special": True})["tokens"]

        def ids(count, base=None):
            source = base if base is not None else tokenized
            return [source[i % len(source)] for i in range(count)]

        # 1) tri tier, near its cap: 97520 + 8.
        tri_prompt = ids(tri_ctx - 16)
        generated = completion("tri-cold", tri_prompt, 8, "convert-gpu", want_hit=False)
        assert server.log_has("-> qwen-3.8-27b-q4-tri"), "tri tier was not selected"
        chain = tri_prompt + generated

        # 2) wide tier: extend to cap - output tokens, then generate fully.
        wide_prompt = chain + ids(wide_ctx - out_tokens - len(chain))
        assert len(wide_prompt) + out_tokens == wide_ctx, (len(wide_prompt), out_tokens)
        generated = completion("wide-handoff", wide_prompt, out_tokens, "convert-gpu", want_hit=True)
        assert server.log_has("-> qwen-3.8-27b-q4-long-b64"), "wide tier was not selected"
        chain = wide_prompt + generated

        # 3) KVarN tier: extend past the wide cap; the wide prefix converts.
        kvarn_prompt = chain + ids(kvarn_ctx - out_tokens - len(chain))
        assert len(kvarn_prompt) + out_tokens == kvarn_ctx, (len(kvarn_prompt), out_tokens)
        before = server.log_count("route snapshot converted:")
        converted_tokens = completion("kvarn-converted", kvarn_prompt, out_tokens, "convert-gpu",
                                     want_hit=True)
        assert server.log_has("-> qwen-3.8-27b-kvarn4-long-b64"), "kvarn tier was not selected"
        after = server.log_count("route snapshot converted:")
        assert after == before + 1, (before, after)
        rows[-1]["conversion_events"] = after - before
        persist()
        chain = kvarn_prompt + converted_tokens

        store = root / "auto-store"
        converted = []
        for path in sorted(store.glob("auto-*.bin")):
            if "convert-native" in path.name:
                continue
            try:
                manifest = read_manifest(path)
            except Exception:
                continue
            if "converted" in manifest:
                converted.append((path, manifest))
        assert converted, sorted(path.name for path in store.iterdir())
        path, manifest = converted[0]
        rows.append({"label": "converted-manifest", "path": str(path),
                     "n_tokens": manifest["n_tokens"], "state_bytes": manifest["state_bytes"],
                     "converted": manifest["converted"]})
        persist()

        # 4) repeat: reuse without another conversion.
        before = server.log_count("route snapshot converted:")
        completion("kvarn-reuse", kvarn_prompt, out_tokens, "convert-gpu", want_hit=True)
        after = server.log_count("route snapshot converted:")
        assert after == before, (before, after)

        # 5) cold comparison: a fresh conversation prefills the same prompt on
        # the KVarN tier without any transfer.
        if not args.skip_cold:
            cold_prompt = ids(kvarn_ctx - out_tokens, base=cold_base)
            assert len(cold_prompt) == len(kvarn_prompt)
            completion("kvarn-cold", cold_prompt, out_tokens, "convert-gpu-cold", want_hit=False)

        # 6) restart: the converted snapshot must restore from disk.
        server.close()
        restart_root = root / "restart"
        restart_root.mkdir(parents=True, exist_ok=True)
        server = Server(args, restart_root, Path(args.preset), None)
        server.request("/health", timeout=60)
        completion("kvarn-restart", kvarn_prompt, out_tokens, "convert-gpu", want_hit=True)
        assert server.log_has("auto-restore") or server.log_has("unified_snapshot_restore"), \
            "restart did not restore a snapshot from disk"

        persist()
        print("PASS: ladder, conversion, reuse, cold comparison, restart")
        return 0
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    raise SystemExit(main())
