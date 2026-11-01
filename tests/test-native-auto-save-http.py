#!/usr/bin/env python3
"""HTTP contract for the opt-in native overflow auto-save.

Runs only on a CPU/server fixture without native overflow, which is exactly the
ineligible case: the feature must stay silent by default and must skip cleanly
(with a log line and no file) when explicitly enabled, never turning a
completed request into an error.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

AUTO_SAVE_MARKER = "auto-save native overflow"


def request(port, path, data):
    payload = json.dumps(data).encode()
    req = Request(f"http://127.0.0.1:{port}{path}", data=payload,
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=30) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


class Server:
    def __init__(self, binary, model, root, extra_env, max_count="64"):
        self.root = root
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GGML_DISABLE_VULKAN="1")
        env.pop("LLAMA_NATIVE_SNAPSHOT_AUTO_SAVE", None)
        env.update(extra_env)
        command = [binary, "-m", model, "--host", "127.0.0.1", "--port", str(self.port),
                   "-c", "512", "-np", "1", "-t", "2", "--slot-save-path", str(root),
                   "--slot-save-max-count", max_count]
        self.child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1, env=env)
        self.ready = threading.Event()
        self.logs = []
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for line in self.child.stdout:
            self.logs.append(line)
            if "listening on http://" in line:
                self.ready.set()

    def text(self):
        return "".join(self.logs)

    def wait_for_log(self, needle, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if needle in self.text():
                return True
            time.sleep(0.05)
        return needle in self.text()

    def complete(self):
        return request(self.port, "/completion", {"prompt": "The quick brown fox", "n_predict": 4,
                                                 "cache_prompt": True, "id_slot": 0})

    def stop(self):
        self.child.terminate()
        try:
            self.child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.child.kill()
            self.child.wait()


def auto_saved_files(root):
    return sorted(path.name for path in Path(root).glob("native-auto-*.bin"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server")
    parser.add_argument("model")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="native-auto-save-") as tmp:
        root = Path(tmp)

        # Default: no opt-in env, so the completion hook is never entered.
        default_dir = root / "default"
        default_dir.mkdir()
        server = Server(args.server, args.model, default_dir, {})
        try:
            assert server.ready.wait(60), "server failed to start: " + "".join(server.logs[-15:])
            status, body = server.complete()
            assert status == 200, (status, body)
            assert body["content"], body
            assert not auto_saved_files(default_dir), "auto-save ran without opt-in"
            assert AUTO_SAVE_MARKER not in server.text(), "unexpected auto-save log without opt-in"
            # The shared publisher must still reject an explicit native save on
            # this unqualified CPU peer with the historical error status.
            status, body = server.complete()
            assert status == 200, (status, body)
            status, body = request(server.port, "/slots/0?action=save",
                                   {"filename": "explicit.slot", "native_stream": True})
            assert status == 400, (status, body)
            assert "Unable to save slot" in body.get("error", {}).get("message", ""), body
            assert not (default_dir / "explicit.slot").exists(), \
                "rejected explicit save left a file behind"
        finally:
            server.stop()

        # Opt-in, but this CPU fixture has no native overflow: one clean skip.
        enabled_dir = root / "enabled"
        enabled_dir.mkdir()
        server = Server(args.server, args.model, enabled_dir,
                        {"LLAMA_NATIVE_SNAPSHOT_AUTO_SAVE": "1"})
        try:
            assert server.ready.wait(60), "server failed to start: " + "".join(server.logs[-15:])
            status, body = server.complete()
            assert status == 200, (status, body)
            assert body["content"], body
            assert server.wait_for_log(AUTO_SAVE_MARKER), "no auto-save decision was logged"
            assert "no native overflow is resident" in server.text(), \
                "unexpected skip reason: " + server.text()[-2000:]
            assert not auto_saved_files(enabled_dir), "ineligible auto-save wrote a snapshot"
        finally:
            server.stop()

        # A reference journal is store metadata, not a snapshot: retention must
        # not count, evict or rewrite it even with a one-snapshot quota.
        journal_dir = root / "journal"
        journal_dir.mkdir()
        journal = journal_dir / ".rkva-native-refs.jsonl"
        pinned = {
            "schema": "rkva-native-refs", "version": 1, "state": "pinned",
            "xbox_key": "fff30e532caeac47750f17351664f3b392ffdafc0801514d09395c5326bee09a",
            "host_file": "", "n_tokens": 0, "split_p": 0, "native": True,
            "timestamp": "2026-11-01T00:00:00Z", "source": "migrated",
        }
        journal.write_text(json.dumps(pinned, separators=(",", ":")) + "\n")
        before = journal.read_bytes()
        server = Server(args.server, args.model, journal_dir, {}, max_count="1")
        try:
            assert server.ready.wait(60), "server failed to start: " + "".join(server.logs[-15:])
            status, body = server.complete()
            assert status == 200, (status, body)
            status, body = request(server.port, "/slots/0?action=save", {"filename": "legacy.slot"})
            assert status == 200, (status, body)
            assert (journal_dir / "legacy.slot").exists(), "the snapshot was not published"
            assert journal.read_bytes() == before, "retention modified the reference journal"
        finally:
            server.stop()

    print("PASS: native auto-save is silent by default, skips cleanly with no native overflow, "
          "and the reference journal survives retention")


if __name__ == "__main__":
    main()
