#!/usr/bin/env python3
"""Exercise snapshot preservation through a real isolated server and HTTP API."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server")
    parser.add_argument("model")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="rkva-retention-") as tmp:
        root = Path(tmp)
        pinned = root / "unclassified.snapshot"
        data = b"preserve-unknown-host-snapshot"
        pinned.write_bytes(data)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GGML_DISABLE_VULKAN="1")
        command = [args.server, "-m", args.model, "--host", "127.0.0.1", "--port", str(port),
                   "-c", "512", "-np", "1", "-t", "2", "--slot-save-path", tmp,
                   "--slot-save-max-count", "1", "--slot-save-max-mb", "32"]
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, env=env)
        ready = threading.Event()
        logs = []

        def drain():
            for line in child.stdout:
                logs.append(line)
                if "listening on http://" in line:
                    ready.set()

        threading.Thread(target=drain, daemon=True).start()

        def request(path, data):
            payload = json.dumps(data).encode()
            req = Request(f"http://127.0.0.1:{port}{path}", data=payload,
                          headers={"Content-Type": "application/json"})
            try:
                with urlopen(req, timeout=10) as response:
                    return response.status, json.load(response)
            except HTTPError as error:
                return error.code, json.load(error)

        try:
            if not ready.wait(20):
                raise RuntimeError("server failed to start: " + "".join(logs[-15:]))
            status, body = request("/completion", {"prompt": "The quick brown fox", "n_predict": 4,
                                                    "cache_prompt": True, "id_slot": 0})
            assert status == 200, (status, body)
            status, body = request("/slots/0?action=save", {"filename": pinned.name})
            assert status == 400, (status, body)
            assert pinned.read_bytes() == data
            status, body = request("/slots/0?action=save", {"filename": "new.slot"})
            assert status == 400, (status, body)
            assert pinned.read_bytes() == data
            assert not (root / "new.slot").exists(), "rejected publication left a new file behind"
            pinned.unlink()  # only this isolated fixture owns the test file
            staged = root / "retention.tmp-interrupted"
            staged.mkdir()
            (staged / "victim.slot").write_bytes(b"interrupted-backup")
            status, body = request("/slots/0?action=save", {"filename": "after-crash.slot"})
            assert status == 400, (status, body)
            assert (staged / "victim.slot").read_bytes() == b"interrupted-backup"
            assert not (root / "after-crash.slot").exists()
            print("PASS: HTTP save preserves pinned files and blocks unresolved retention staging")
        finally:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
