#!/usr/bin/env python3
"""Real local router/cache matrix. Standard library only, CPU by default.

CPU example (the generated fixture needs no downloads):
  TMPDIR=/path/on/disk python3 tools/server/tests/router_state_streaming.py \
    --server build-bench/bin/llama-server --model build-bench/tests/test-models/qwen35-mtp.gguf \
    --work-dir /path/on/disk/run

GPU must be explicitly selected; this script never stops or modifies services.
The caller must make the GPU available before selecting --gpu. All child
processes spawned here are stopped in finally, including failed assertions.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# This file is also safe to collect beside the existing pytest integration suite.
NO_PRELOAD_SERVER_PRESETS = True


class Server:
    def __init__(self, args, root, preset):
        self.root = root
        self.log = root / "server.log"
        self.lines = []
        self.ready = threading.Event()
        env = os.environ.copy()
        env.update(TMPDIR=str(root), LLAMA_CACHE=str(root / "cache"))
        env.update(GGML_DISABLE_VULKAN="1", GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB="18")
        env.pop("LLAMA_SERVER_DEBUG_FAKE_TIMING", None)
        env.pop("LLAMA_TEST_STATE_FILE_COMMIT_FAIL_AFTER", None)
        for key, value in getattr(args, "env_overrides", {}).items():
            env[key] = str(value)
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
        slot_save_path = getattr(args, "slot_save_path", None)
        if slot_save_path is None:
            slot_save_path = str(root)
        command = [str(Path(args.server).resolve()), "--models-preset", str(preset), "--models-max", "1",
                   "--host", "127.0.0.1", "--port", str(self.port), "--log-verbosity", "3"]
        if slot_save_path:
            command.extend(["--slot-save-path", str(slot_save_path)])
        (root / "command.json").write_text(json.dumps(command, indent=2))
        self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                     text=True, bufsize=1, start_new_session=True)

        def consume():
            with self.log.open("w") as log:
                for line in self.proc.stdout:
                    log.write(line)
                    log.flush()
                    self.lines.append(line)
                    if "server is listening on" in line or "listening on http://" in line:
                        self.ready.set()
            self.ready.set()

        self.reader = threading.Thread(target=consume, daemon=True)
        self.reader.start()
        self.sampling_done = threading.Event()

        def sample_memory():
            # Follow only this router's process tree. No system-wide process scan.
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
                        # The router can spawn a replacement child from a
                        # worker TID. /proc/<pid>/task/<pid>/children sees only
                        # the main thread's children, so enumerate every TID
                        # before descending through this private process tree.
                        try:
                            for tid in Path(f"/proc/{pid}/task").iterdir():
                                try:
                                    queue.extend(int(value) for value in (tid / "children").read_text().split())
                                except (OSError, ValueError):
                                    pass
                        except OSError:
                            pass
                    available = re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())
                    output.write(json.dumps({"time": time.time(), "processes_kib": processes,
                                             "available_kib": int(available.group(1))}) + "\n")
                    output.flush()
                    self.sampling_done.wait(0.5)

        self.memory_reader = threading.Thread(target=sample_memory, daemon=True)
        self.memory_reader.start()
        if not self.ready.wait(args.timeout) or self.proc.poll() is not None:
            self.close()
            raise RuntimeError(f"router failed to start: {self.log}")
        self.timeout = args.timeout

    def request(self, path, body=None, conversation="matrix"):
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

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                # Only this harness's private process group, never a service.
                import signal
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.reader.join(timeout=10)
        self.sampling_done.set()
        self.memory_reader.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", help="Public model ID; GPU default is the literal model path")
    parser.add_argument("--preset", help="Use an existing preset verbatim (GPU release clone); CPU generates a fixture preset")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--contexts", type=int, nargs=4)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--probe-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--cache-ram", type=int, default=8192)
    parser.add_argument("--auto-cache", action="store_true",
                        help="enable slot-save-auto and verify its persistent store survives route cleanup")
    parser.add_argument("--handoff-smoke", action="store_true",
                        help="GPU-only large handoff smoke: 97535+1 then 100352+16")
    args = parser.parse_args()
    contexts = args.contexts or ([32768, 56320, 97536, 104448] if args.gpu else [256, 512, 768, 1024])
    output = args.output_tokens or (4096 if args.gpu else 64)
    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    persistent_store = root
    if args.auto_cache:
        persistent_store = root / "auto-store"
        persistent_store.mkdir()
        args.slot_save_path = str(persistent_store)
    model = str(Path(args.model).resolve())
    if not args.gpu:
        # Generated tensor fixtures omit a tokenizer. Native HTTP generation
        # still detokenizes output IDs, so give our private copy real pieces.
        fixture = root / "qwen35-http.gguf"
        subprocess.run([str(Path(args.server).resolve().parent / "test-state-file-stream"),
                        "--make-http-fixture", model, str(fixture)], check=True)
        model = str(fixture)
    public = args.model_id or (model if args.gpu else "router-state-test")
    preset = Path(args.preset).resolve() if args.preset else root / "models.ini"
    auto_options = "slot-save-auto = true\nslot-save-block = 1" if args.auto_cache else ""
    if not args.preset:
        preset.write_text(f"""version = 1
[*]
model = {model}
{'' if args.gpu else 'override-kv = qwen35.context_length=int:' + str(contexts[-1])}
load-mode = none
parallel = 1
device = {'CUDA0' if args.gpu else 'none'}
gpu-layers = {99 if args.gpu else 0}
flash-attn = on
fit = off
cache-ram = {args.cache_ram}
ctx-checkpoints = {1 if args.gpu else 2}
{'' if args.gpu else 'checkpoint-min-step = 1'}
metrics = true
cache-prompt = true
context-shift = false
threads = {8 if args.gpu else 2}
threads-batch = {8 if args.gpu else 2}
{'''gpu-mem-clock-decode = 10501
gpu-power-prefill = 200
gpu-power-decode = 170
gpu-power-device = 0''' if args.gpu else ''}
cache-type-k = q4_0
cache-type-v = q4_0
slot-save-max-count = 16
slot-save-max-mb = 16384
{auto_options}
[tri]
load-on-startup = true
route-group = {public}
route-max-tokens = {contexts[2]}
ctx-size = {contexts[2]}
ctx-size-mtp = {contexts[1]}
mtp-max-tokens = {contexts[1]}
ctx-size-mtp-short = {contexts[0]}
mtp-short-max-tokens = {contexts[0]}
spec-type = draft-mtp
spec-draft-n-max = 2
spec-draft-n-max-short = 4
spec-draft-p-min = 0.70
spec-draft-type-k = q4_0
spec-draft-type-v = q4_0
batch-size = 256
ubatch-size = 256
[wide]
route-group = {public}
ctx-size = {contexts[3]}
spec-type = none
batch-size = 64
ubatch-size = 64
""")
    rows = []
    server = None

    def persist():
        (root / "results.json").write_text(json.dumps({"contexts": contexts, "output": output, "rows": rows}, indent=2))

    def completion(label, prompt, n, cache=True, want_hit=False, classify_mtp_boundary=False):
        started = time.monotonic()
        body = server.request("/completion", {"model": public, "prompt": prompt, "n_predict": n,
            "id_slot": 0, "cache_prompt": cache, "return_tokens": True, "temperature": 0,
            "seed": 1234, "ignore_eos": True})
        tokens = body.get("tokens", [])
        row = {"label": label, "input": len(prompt), "requested_output": n, "output": len(tokens),
               "model": body.get("model"), "timings": body.get("timings"),
               "stop_type": body.get("stop_type"), "stop_detail": body.get("stop_detail"),
               "truncated": body.get("truncated"), "timestamp": time.time(),
               "wall_seconds": time.monotonic()-started,
               "tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()}
        rows.append(row)
        persist()
        print(json.dumps(row), flush=True)
        if len(tokens) != n:
            # Explicit negative control for the existing MTP context guard.
            # Every other request, including the one-token-margin retry below,
            # still requires the exact requested output length.
            assert classify_mtp_boundary and 0 < n-len(tokens) <= 4, row
            assert body.get("truncated") is True and body.get("stop_type") == "limit", row
            assert body.get("stop_detail") == "context_limit", row
            row["exact_limit_failed"] = True
            persist()
        assert body.get("model") == public, row
        if want_hit:
            assert body["timings"]["cache_n"] > 0, row
        return tokens

    def rejected_route_files(action):
        # Re-sign the small manifest using the exact vendored hash already in
        # the server library. This distinguishes semantic identity/layout
        # rejection from a mere checksum error, without Python dependencies.
        library = ctypes.CDLL(str(Path(args.server).resolve().parent / "libllama-server-impl.so"))
        checksum = library.XXH64
        checksum.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
        checksum.restype = ctypes.c_uint64
        files = list(persistent_store.glob("llama-router-state-*/" + action["filename"]))
        if not files:
            # Unified route snapshots share the configured persistent store;
            # the private directory is now only a legacy/fallback layout.
            direct = persistent_store / action["filename"]
            if direct.exists():
                files = [direct]
        assert len(files) == 1, files
        original = files[0]
        with original.open("rb") as file:
            file.seek(-24, 2)
            length, _, magic = struct.unpack("<QQ8s", file.read(24))
            file.seek(-24-length, 2)
            meta = json.loads(file.read(length))
        cases = {
            "model": "model identity differs",
            "layout": "layout differs and no conversion is available",
            "payload": ("checksum", "native streaming restore failed"),
            "truncated": "manifest",
        }
        for case, message in cases.items():
            damaged = original.with_name("invalid-" + case + ".bin")
            shutil.copyfile(original, damaged)
            with damaged.open("r+b") as file:
                if case in ("model", "layout"):
                    changed = dict(meta)
                    changed[case] = "wrong-" + changed[case]
                    data = json.dumps(changed).encode()
                    file.seek(meta["state_bytes"])
                    file.write(data)
                    file.write(struct.pack("<QQ8s", len(data), checksum(data, len(data), 0), magic))
                    file.truncate()
                elif case == "payload":
                    file.seek(meta["state_bytes"]-1)
                    byte = file.read(1)[0] ^ 1
                    file.seek(meta["state_bytes"]-1)
                    file.write(bytes([byte]))
                else:
                    file.truncate(meta["state_bytes"]-1)
            try:
                server.request("/slots/0?action=restore", dict(action, filename=damaged.name))
                raise AssertionError("invalid route file accepted: " + case)
            except RuntimeError as error:
                expected = message if isinstance(message, tuple) else (message,)
                assert any(part in str(error) for part in expected), error
            slots = server.request("/slots?model=" + urllib.parse.quote(public))
            assert slots[0].get("n_prompt_tokens", 0) == 0, slots
            rows.append({"label": "native-rejected-" + case, "empty_destination": True})
            persist()

    try:
        server = Server(args, root, preset)
        # The warm child provides tokenizer metadata. Use real IDs, avoiding
        # tokenization drift when generated output becomes the next input.
        server.request("/props?model=" + urllib.parse.quote(public))
        tokenized = (server.request("/tokenize", {"model": public, "content": "The river is blue. The sky is clear.", "add_special": True})["tokens"]
                     if args.gpu else [3, 7, 11, 19, 23, 31, 47, 53])
        assert tokenized
        if args.handoff_smoke:
            if not args.gpu or contexts[2:] != [97536, 104448]:
                raise ValueError("--handoff-smoke requires the default GPU context pair 97536/104448")
            source_prompt = (tokenized * ((97535 + len(tokenized) - 1) // len(tokenized)))[:97535]
            generated = completion("handoff-smoke-source", source_prompt, 1)
            wide_prompt = source_prompt + generated
            wide_prompt.extend((tokenized * ((100352 - len(wide_prompt) + len(tokenized) - 1) // len(tokenized)))[:100352-len(wide_prompt)])
            completion("handoff-smoke-wide", wide_prompt, 16)
            slots = server.request("/slots?model=" + urllib.parse.quote(public))
            assert slots[0]["n_ctx"] == 104448 and "adaptive_context" not in slots[0], slots
            text = "".join(server.lines)
            saves = re.findall(r"router streaming save: tokens=(\d+) bytes=(\d+)", text)
            restores = re.findall(r"router streaming restore: tokens=(\d+) bytes=(\d+)", text)
            assert saves and saves == restores and saves[0][0] == "97535", (saves, restores)
            assert "MTP_GPU=0" in text, "LONG MTP release not observed"
            rows.append({"label": "handoff-smoke", "saves": saves, "restores": restores, "pass": True})
            persist()
            print("PASS: GPU large handoff smoke 97535+1 -> 100352+16", flush=True)
            return
        if args.gpu:
            small = tokenized * 8
            warm = completion("gpu-small-warm", small, 16)
            extended = small + warm + tokenized[:4]
            cached = completion("gpu-small-continue", extended, 16, want_hit=True)
            cold = completion("gpu-small-cold-oracle", extended, 16, cache=False)
            assert cached == cold, "GPU small cached continuation differs from cold"
        history = []
        for index, capacity in enumerate(contexts):
            prompt_size = capacity-output
            prompt = history[:]
            assert len(prompt) < prompt_size
            while len(prompt) < prompt_size:
                prompt.extend(tokenized[:prompt_size-len(prompt)])
            prefix = f"tier-{capacity}"
            generated = completion(prefix+"-transition-limit", prompt, output, want_hit=index > 0,
                                   classify_mtp_boundary=args.gpu and index < 2)
            if len(generated) != output:
                prompt = prompt[:-1]
                generated = completion(prefix+"-one-token-margin", prompt, output, want_hit=True)
            probe = generated[:args.probe_tokens]
            slots = server.request("/slots?model=" + urllib.parse.quote(public))
            expected_profile = ["mtp-short", "mtp", "long", None][index]
            row = {"label": prefix+"-profile", "slots": slots}
            rows.append(row); persist()
            if expected_profile:
                assert slots[0]["adaptive_context"]["profile"] == expected_profile, slots
            else:
                assert "adaptive_context" not in slots[0], slots
            assert slots[0]["n_ctx"] == capacity, slots
            repeat = completion(prefix+"-repeat", prompt, args.probe_tokens, want_hit=True)
            assert repeat == probe, "cached repetition differs"
            extension = prompt + probe + tokenized[:4]
            continued = completion(prefix+"-continue", extension, args.probe_tokens, want_hit=True)
            if not args.gpu:
                # Compare cached continuation against a cold forward pass.
                cold = completion(prefix+"-cold-oracle", extension, args.probe_tokens, cache=False)
                assert cold == continued, "cached continuation differs from cold"
                # Explicit adaptive files preserve their existing full snapshot
                # contract; the interchild file is tested by the last hop.
                if index < 3:
                    filename = f"explicit-{capacity}.bin"
                    saved = server.request("/slots/0?action=save", {"model": public, "filename": filename})
                    server.request("/slots/0?action=erase", {"model": public})
                    restored = server.request("/slots/0?action=restore", {"model": public, "filename": filename})
                    assert saved["n_saved"] == restored["n_restored"] > 0
                    rows.append({"label": prefix+"-explicit", "saved": saved, "restored": restored}); persist()
                    completion(prefix+"-explicit-continue", extension+cold+tokenized[:4], 2, want_hit=True)
                else:
                    handoff_identity_before = {
                        path.resolve(): path.stat().st_ino for path in persistent_store.glob("auto-*.bin")
                    } if args.auto_cache else {}
                    action = {"model": public, "filename": "native-route.bin",
                              "route_state_transfer": True, "route_target_no_mtp": True}
                    saved = server.request("/slots/0?action=save", action)
                    try:
                        server.request("/slots/0?action=restore", action)
                        raise AssertionError("nonempty endpoint restore was accepted")
                    except RuntimeError as error:
                        assert "not empty" in str(error), error
                    completion(prefix+"-nonempty-preserved", extension+cold+tokenized[:4], 2, want_hit=True)
                    server.request("/slots/0?action=erase", {"model": public})
                    rejected_route_files(action)
                    restored = server.request("/slots/0?action=restore", action)
                    assert saved["n_saved"] == restored["n_restored"] > 0
                    completion(prefix+"-native-continue", extension+cold+tokenized[:4], 2, want_hit=True)
            history = prompt + generated
            # Probes deliberately branch and consume checkpoint history. Put
            # the exact generated chain back in the live slot before migrating.
            # The last output is sampled, so the native state must cover all
            # preceding output tokens, not claim an unevaluated final token.
            tail = completion(prefix+"-reprime-chain", history[:-1], 1)
            assert tail == history[-1:], "teacher-forced chain continuation differs"
        # A distinct short conversation can return to tri; reverse KV migration
        # is deliberately outside the upward handoff contract.
        completion("return-short", tokenized, 8)
        slots = server.request("/slots?model=" + urllib.parse.quote(public))
        assert slots[0]["adaptive_context"]["profile"] == "mtp-short", slots
        completion("return-short-cache", tokenized, 8, want_hit=True)
        text = "".join(server.lines)
        saves = re.findall(r"router streaming save: tokens=(\d+) bytes=(\d+)", text)
        restores = re.findall(r"router streaming restore: tokens=(\d+) bytes=(\d+)", text)
        assert saves and saves == restores, (saves, restores)
        assert "MTP_GPU=0" in text, "LONG MTP release not observed"
        reused = re.findall(r"router snapshot reused: path=(\S+) checksum=(\d+)", text)
        if args.auto_cache:
            assert reused, "route handoff did not reference an existing unified auto snapshot"
            assert all(Path(path).name.startswith("auto-") for path, _ in reused), reused
            for path, checksum in reused:
                snapshot = Path(path)
                assert snapshot.exists(), snapshot
                if args.auto_cache:
                    assert snapshot.resolve() in handoff_identity_before, (snapshot, handoff_identity_before)
                    assert snapshot.stat().st_ino == handoff_identity_before[snapshot.resolve()], snapshot
                with snapshot.open("rb") as file:
                    file.seek(-24, 2)
                    length, _, magic = struct.unpack("<QQ8s", file.read(24))
                    file.seek(-24 - length, 2)
                    manifest = json.loads(file.read(length))
                assert str(manifest["state_checksum"]) == checksum, (snapshot, manifest)
        rows.append({"label": "handoff", "saves": saves, "restores": restores,
                     "reused_snapshots": reused,
                     "identity_path_checksum_inode_preserved": bool(reused) if args.auto_cache else None,
                     "pass": True})
        persist()
        boundaries = [row["label"] for row in rows if row.get("exact_limit_failed")]
        print("PASS: chain, normal repeat/continuation, return-short and disk handoff; "
              f"documented MTP exact-limit failures ({output}-output margin retries passed): {boundaries}", flush=True)
    finally:
        if server:
            server.close()
        if args.auto_cache:
            auto_states = sorted(persistent_store.glob("auto-*.bin"))
            auto_meta = sorted(persistent_store.glob("auto-*.bin.meta"))
            route_dirs = sorted(persistent_store.glob("llama-router-state-*"))
            assert auto_states and not auto_meta, (auto_states, auto_meta)
            # Auto-cache entries now use the exact canonical route envelope:
            # native state + footer manifest, not the former .meta index format.
            for state in auto_states:
                with state.open("rb") as file:
                    file.seek(-24, 2)
                    length, _, magic = struct.unpack("<QQ8s", file.read(24))
                    assert magic == b"LLROUTE1" and 0 < length <= 64 * 1024, state
                    file.seek(-24 - length, 2)
                    manifest = json.loads(file.read(length))
                    assert manifest["version"] == 1 and manifest["index_block"] > 0, state
            assert not route_dirs, route_dirs
            rows.append({"label": "auto-cache-persistence", "auto_states": len(auto_states),
                         "auto_meta": len(auto_meta), "canonical_envelope": True,
                         "route_dirs_after_close": len(route_dirs),
                         "store": str(persistent_store)})
        persist()


if __name__ == "__main__":
    main()
