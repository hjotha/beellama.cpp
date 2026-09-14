#!/usr/bin/env python3
"""Exercise adaptive queueing, concurrent requests and client cancellation."""
import concurrent.futures
import http.client
import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(os.environ.get("ADAPTIVE_CONCURRENCY_OUT", ROOT / "gauntlet/stage-12-concurrency-cancel"))
OUT.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("ADAPTIVE_CONCURRENCY_PORT", "19413"))
MODEL = os.environ.get("ADAPTIVE_MODEL", "/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf")
SERVER = os.environ.get("ADAPTIVE_SERVER", str(ROOT / "build-adaptive-cuda/bin/llama-server"))
CMD = [
    SERVER, "--model", MODEL, "--host", "127.0.0.1", "--port", str(PORT),
    "--ctx-size", "1024", "--ctx-size-mtp", "512", "--mtp-max-tokens", "32",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-p-min", "0.80",
    "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0", "--gpu-layers", "99",
    "--device", "CUDA0", "--flash-attn", "on", "--fit", "off", "--parallel", "1",
    "--batch-size", "256", "--ubatch-size", "256", "--cache-type-k", "q4_0",
    "--cache-type-v", "q4_0", "--cache-ram", "2048", "--ctx-checkpoints", "1",
    "--load-mode", "none", "--no-warmup", "--no-context-shift", "--slots", "--n-predict", "0",
]
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = str(ROOT / "build-adaptive-cuda/bin") + ":/opt/cuda/lib64:" + ENV.get("LD_LIBRARY_PATH", "")


def call(method, path, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw, status = response.read(), response.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read(), exc.code
    try:
        body = json.loads(raw)
    except Exception:
        body = {"raw": raw.decode(errors="replace")}
    return status, body


def wait_ready(process):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before readiness rc={process.returncode}")
        try:
            if call("GET", "/health", timeout=10)[0] == 200:
                return
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError("readiness timeout")


def completion(prompt, n_predict):
    return call("POST", "/completion", {
        "prompt": prompt, "n_predict": n_predict, "temperature": 0,
        "cache_prompt": False,
    }, timeout=300)


def close_client_request(prompt, n_predict):
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "temperature": 0,
        "cache_prompt": False,
    }).encode()
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
    conn.connect()
    conn.putrequest("POST", "/completion")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(len(body)))
    conn.endheaders()
    conn.send(body)
    time.sleep(0.08)
    conn.close()


def explicit_stream_cancel(prompt, n_predict, conversation_id):
    """Cancel a live stream through the server's user-stop route."""
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "temperature": 0,
        "cache_prompt": False, "stream": True, "ignore_eos": True,
    }).encode()
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    conn.connect()
    conn.putrequest("POST", "/completion")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("X-Conversation-Id", conversation_id)
    conn.putheader("Content-Length", str(len(body)))
    conn.endheaders()
    conn.send(body)

    session_seen = False
    slot_busy = False
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status, sessions = call("POST", "/v1/streams/lookup", {"conversation_ids": [conversation_id]}, timeout=10)
        if status == 200 and isinstance(sessions, list) and any(not item.get("is_done", True) for item in sessions):
            session_seen = True
            slot_status, slots = call("GET", "/slots", timeout=10)
            slot_busy = slot_status == 200 and any(slot.get("is_processing") for slot in slots)
            if slot_busy:
                break
        time.sleep(0.02)

    delete_status, delete_body = call(
        "DELETE", "/v1/stream?conv_id=" + urllib.parse.quote(conversation_id, safe=""), timeout=30
    )
    stream_status = None
    stream_bytes = b""
    stream_error = None
    try:
        response = conn.getresponse()
        stream_status = response.status
        stream_bytes = response.read(65536)
    except Exception as exc:
        stream_error = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()

    slot_idle = False
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        slot_status, slots = call("GET", "/slots", timeout=10)
        if slot_status == 200 and slots and not any(slot.get("is_processing") for slot in slots):
            slot_idle = True
            break
        time.sleep(0.05)

    replay_status, replay_body = call(
        "GET", "/v1/stream?conv_id=" + urllib.parse.quote(conversation_id, safe="") + "&from=0", timeout=30
    )
    return {
        "session_seen": session_seen,
        "slot_busy": slot_busy,
        "delete_status": delete_status,
        "delete_body": delete_body,
        "stream_status": stream_status,
        "stream_bytes": len(stream_bytes),
        "stream_error": stream_error,
        "slot_idle": slot_idle,
        "replay_status": replay_status,
        "replay_body": replay_body,
    }


result = {"port": PORT, "model": MODEL, "success": False, "errors": []}
process = None
with (OUT / "server.log").open("w") as log:
    try:
        process = subprocess.Popen(CMD, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        wait_ready(process)
        status, props = call("GET", "/props")
        assert status == 200 and props["adaptive_context"]["profile"] == "mtp", props
        result["initial"] = props["adaptive_context"]

        long_prompt = "concurrent long request token " * 170
        short_prompt = "short queued request"
        requests = [(long_prompt, 8), (short_prompt, 8), (short_prompt + " again", 8)]
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(completion, prompt, n_predict) for prompt, n_predict in requests]
            responses = [future.result(timeout=300) for future in futures]
        result["concurrent_elapsed_s"] = time.monotonic() - started
        result["concurrent"] = [{"status": status, "tokens_predicted": body.get("tokens_predicted"),
                                  "content_len": len(body.get("content", ""))}
                                 for status, body in responses]
        assert all(status == 200 and body.get("content") is not None for status, body in responses), responses

        status, props = call("GET", "/props")
        assert status == 200 and props["adaptive_context"]["state"] == "ready", props
        result["after_concurrent"] = props["adaptive_context"]

        close_client_request("cancel this long generation token " * 20, 128)
        saw_busy = False
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status, slots = call("GET", "/slots", timeout=10)
            if status == 200 and any(slot.get("is_processing") for slot in slots):
                saw_busy = True
            if saw_busy and status == 200 and slots and not any(slot.get("is_processing") for slot in slots):
                break
            time.sleep(0.1)
        assert saw_busy, "cancelled request never occupied a slot"
        assert status == 200 and not any(slot.get("is_processing") for slot in slots), slots
        result["cancel"] = {"saw_busy": saw_busy, "slots_after": slots}

        explicit = explicit_stream_cancel(
            "explicit stop must cancel before generation " * 40,
            128,
            "adaptive-concurrency-explicit-stop",
        )
        result["explicit_cancel"] = explicit
        assert explicit["session_seen"], explicit
        assert explicit["slot_busy"], explicit
        assert explicit["delete_status"] == 204, explicit
        assert explicit["stream_status"] == 200 and explicit["stream_error"] is None, explicit
        assert explicit["slot_idle"], explicit
        assert explicit["replay_status"] == 404, explicit

        status, body = completion("after cancellation", 4)
        assert status == 200 and body.get("content") is not None, body
        result["after_cancel"] = {"status": status, "tokens_predicted": body.get("tokens_predicted")}
        status, props = call("GET", "/props")
        assert status == 200 and props["adaptive_context"]["state"] == "ready" \
            and props["adaptive_context"]["profile"] == "mtp" \
            and props["adaptive_context"]["mtp_weights_resident"] is True, props
        result["final"] = props["adaptive_context"]
        result["success"] = True
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=60)
        result["server_rc"] = None if process is None else process.returncode
        if result.get("success") and result["server_rc"] != 0:
            result["success"] = False
            result["errors"].append(f"server exited with rc={result['server_rc']}")
        (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")

print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if result["success"] else 1)
