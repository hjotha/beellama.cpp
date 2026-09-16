#!/usr/bin/env python3
"""CPU router test for the explicit q4_0 -> KVarN conversion tier.

Runs a private three-member route group on the generated qwen35 fixture:
tri (q4/MTP) -> wide (q4, capped) -> kvarn (KVarN4, uncapped). Verifies the
route ladder boundaries, the wide -> kvarn hand-off converting a saved q4_0
prefix instead of repeating the prefill, the persisted converted snapshot
(provenance + reuse + restart reuse), and that the earlier profiles keep
working. Standard library only; CPU by default, pass --gpu for a real run.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


class Server:
    def __init__(self, args, root, preset):
        self.root = root
        self.log = root / "server.log"
        self.lines = []
        self.ready = threading.Event()
        env = os.environ.copy()
        env.update(TMPDIR=str(root), LLAMA_CACHE=str(root / "cache"))
        env.update(GGML_DISABLE_VULKAN="1")
        if not args.gpu:
            env.update(CUDA_VISIBLE_DEVICES="", VK_ICD_FILENAMES="/nonexistent", GGML_VK_VISIBLE_DEVICES="")
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
        if not self.ready.wait(args.timeout) or self.proc.poll() is not None:
            self.close()
            raise RuntimeError(f"router failed to start: {self.log}")
        self.timeout = args.timeout

    def request(self, path, body=None, conversation=None):
        headers = {"Content-Type": "application/json"}
        if conversation is not None:
            headers["X-Conversation-Id"] = conversation
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{path}: HTTP {error.code}: {error.read().decode()}") from error

    def log_has(self, needle):
        return any(needle in line for line in self.lines)

    def log_count(self, needle):
        return sum(1 for line in self.lines if needle in line)

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.reader.join(timeout=10)


def server_layout(server):
    """The private layout of the converting child, recorded in its log."""
    for line in reversed(server.lines):
        match = re.search(r"layout=(\{.*\})$", line)
        if match:
            return match.group(1)
    raise AssertionError("child layout was not logged")


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
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    store = root / "auto-store"
    store.mkdir()
    model = str(Path(args.model).resolve())
    if not args.gpu:
        fixture = root / "qwen35-http.gguf"
        subprocess.run([str(Path(args.server).resolve().parent / "test-state-file-stream"),
                        "--make-http-fixture", model, str(fixture)], check=True)
        model = str(fixture)
    public = model
    contexts = [256, 320, 384] if not args.gpu else [32768, 56320, 114688]
    caps = [192, 256] if not args.gpu else [97536, 104448]

    preset = root / "models.ini"
    preset.write_text(f"""version = 1
[*]
model = {model}
override-kv = qwen35.context_length=int:{contexts[-1]}
load-mode = none
parallel = 1
device = {'CUDA0' if args.gpu else 'none'}
gpu-layers = {99 if args.gpu else 0}
flash-attn = on
fit = off
cache-ram = 8192
ctx-checkpoints = {1 if args.gpu else 2}
{'checkpoint-min-step = 1' if not args.gpu else ''}
metrics = true
cache-prompt = true
context-shift = false
threads = {8 if args.gpu else 2}
threads-batch = {8 if args.gpu else 2}
slot-save-path = {store}/
slot-save-auto = true
slot-save-block = 1
slot-save-max-count = 16
slot-save-max-mb = 16384
{'''gpu-mem-clock-decode = 10501
gpu-power-prefill = 200
gpu-power-decode = 170
gpu-power-device = 0''' if args.gpu else ''}
cache-type-k = q4_0
cache-type-v = q4_0
[tri]
load-on-startup = true
route-group = {public}
route-max-tokens = {caps[0]}
ctx-size = {contexts[0]}
ctx-size-mtp = {contexts[0]}
mtp-max-tokens = {contexts[0]}
ctx-size-mtp-short = {contexts[0]}
mtp-short-max-tokens = {contexts[0]}
spec-type = draft-mtp
spec-draft-n-max = 2
spec-draft-n-max-short = 4
spec-draft-p-min = 0.70
spec-draft-type-k = q4_0
spec-draft-type-v = q4_0
batch-size = 64
ubatch-size = 64
[wide]
route-group = {public}
route-max-tokens = {caps[1]}
ctx-size = {contexts[1]}
spec-type = none
batch-size = 64
ubatch-size = 64
[kvarn]
route-group = {public}
ctx-size = {contexts[2]}
spec-type = none
batch-size = 64
ubatch-size = 64
cache-type-k = kvarn4
cache-type-v = kvarn4
""")

    rows = []

    def persist():
        (root / "results.json").write_text(json.dumps(
            {"contexts": contexts, "caps": caps, "rows": rows}, indent=2))

    server = None
    try:
        server = Server(args, root, preset)
        def _terminate(signum, frame):
            del frame
            if server is not None:
                server.close()
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, _terminate)
        signal.signal(signal.SIGINT, _terminate)
        models = server.request("/v1/models")
        listed = models.get("models", [])
        assert len(listed) == 1 and listed[0]["name"] == public, listed

        def completion(label, prompt, n, want_hit=None):
            started = time.monotonic()
            body = server.request("/completion", {"model": public, "prompt": prompt, "n_predict": n,
                "id_slot": 0, "cache_prompt": True, "return_tokens": True, "temperature": 0,
                "seed": 1234, "ignore_eos": True}, conversation="convert-cpu")
            tokens = body.get("tokens", [])
            timings = body.get("timings", {})
            row = {"label": label, "input": len(prompt), "output": len(tokens),
                   "model": body.get("model"), "cache_n": timings.get("cache_n"),
                   "prompt_n": timings.get("prompt_n"), "wall_seconds": time.monotonic() - started}
            rows.append(row)
            persist()
            print(json.dumps(row), flush=True)
            assert body.get("model") == public, row
            if want_hit is True:
                assert timings.get("cache_n", 0) > 0, row
            if want_hit is False:
                assert timings.get("cache_n", 0) == 0, row
            return tokens

        vocab = 128
        base = [3 + (i * 17) % (vocab - 3) for i in range(100)]

        # 1) short conversation on the tri tier.
        generated = completion("short-cold", base, args.output_tokens, want_hit=False)
        assert server.log_has("-> tri") or server.log_has("'tri'"), "tri tier was not selected"
        chain = base + generated

        # 2) grow past the tri cap: wide tier, RAM route hand-off.
        prompt = chain + [3 + (i * 29) % (vocab - 3) for i in range(120)]
        assert len(prompt) + args.output_tokens > caps[0]
        generated = completion("wide-handoff", prompt, args.output_tokens, want_hit=True)
        assert server.log_has("-> wide"), "wide tier was not selected"
        chain = prompt + generated

        # 3) grow past the wide cap: KVarN tier with explicit conversion.
        prompt = chain + [3 + (i * 31) % (vocab - 3) for i in range(60)]
        assert len(prompt) + args.output_tokens > caps[1]
        generated = completion("kvarn-converted", prompt, args.output_tokens, want_hit=True)
        assert server.log_has("-> kvarn"), "kvarn tier was not selected"
        assert server.log_has("route snapshot converted"), "conversion was not logged"
        chain = prompt + generated

        converted = []
        for path in store.glob("auto-*.bin"):
            if "convert-native" in path.name:
                continue
            try:
                candidate = read_manifest(path)
            except (AssertionError, ValueError, OSError):
                continue
            if "converted" in candidate:
                converted.append(path)
        assert converted, sorted(path.name for path in store.iterdir())
        manifest = read_manifest(converted[0])
        assert manifest["layout"] == server_layout(server), (manifest["layout"], server_layout(server))
        assert "converted" in manifest and manifest["converted"]["converter"] == 1, manifest
        assert manifest["converted"]["source_format"] == "q4_0/q4_0", manifest["converted"]
        assert manifest["converted"]["target_format"] == "kvarn_k4v4_g128", manifest["converted"]
        rows.append({"label": "converted-manifest", "path": str(converted[0]),
                     "converted": manifest["converted"], "tokens": manifest["n_tokens"]})
        persist()

        # 4) repeat the same request: the converted snapshot is reused.
        before = server.log_count("route snapshot converted:")
        completion("kvarn-reuse", prompt, args.output_tokens, want_hit=True)
        after = server.log_count("route snapshot converted:")
        assert after == before, (before, after)
        assert server.log_has("router snapshot reused") or server.log_has("converted route snapshot reused"), \
            "no snapshot was reused for the repeat request"

        # 5) restart: the converted snapshot must restore from disk.
        server.close()
        server = Server(args, root, preset)
        completion("kvarn-restart", prompt, args.output_tokens, want_hit=True)
        assert server.log_has("auto-restore") or server.log_has("unified_snapshot_restore"), \
            "restart did not restore the converted snapshot from disk"

        # 6) the earlier profiles still work.
        completion("tri-again", base, args.output_tokens)
        completion("wide-again", prompt[:caps[1] - args.output_tokens], args.output_tokens, want_hit=True)

        (root / "results.json").write_text(json.dumps(
            {"contexts": contexts, "caps": caps, "rows": rows}, indent=2))
        print("PASS: route ladder, q4->KVarN conversion, provenance, reuse, restart, earlier profiles")
        return 0
    except BaseException as error:
        if server is not None:
            try:
                server.close()
            except BaseException:
                pass
        (root / "results.json").write_text(json.dumps(
            {"contexts": contexts, "caps": caps, "rows": rows, "error": repr(error)}, indent=2))
        raise
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    sys.exit(main())
