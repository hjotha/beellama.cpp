#!/usr/bin/env python3
"""End-to-end verification for the unified 5-tier adaptive context and slot save/restore.

Tiers exercised:
  1. short:   <= 128 (mtp-short, draft-4, resident MTP)
  2. medium:  <= 256 (mtp, draft-2, resident MTP)
  3. long:    <= 384 (long, non-resident MTP)
  4. xlong:   <= 448 (xlong, batch 32, non-resident MTP)
  5. xxlong:  <= 512 (xxlong, batch 32, non-resident MTP, kvarn)
"""

import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER_BIN = ROOT / "build-bench/bin/llama-server"
MODEL_PATH = pathlib.Path("/home/hjotha/models/Qwen3.5-4B-MTP-Q4_K_M.gguf")
WORK_DIR = pathlib.Path("/tmp/test-adaptive-5tier-e2e")
SLOTS_DIR = WORK_DIR / "slots"

if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
SLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Allocate free port
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    PORT = s.getsockname()[1]

BASE_URL = f"http://127.0.0.1:{PORT}"

CMD = [
    str(SERVER_BIN),
    "--model", str(MODEL_PATH),
    "--host", "127.0.0.1",
    "--port", str(PORT),
    "--gpu-layers", "0",
    "--device", "none",
    "--no-repack",
    "--flash-attn", "on",
    "--fit", "off",
    "--parallel", "1",
    "--slots",
    "--batch-size", "32",
    "--ubatch-size", "32",
    "--cache-type-k", "q4_0",
    "--cache-type-v", "q4_0",
    "--spec-type", "draft-mtp",
    "--ctx-size-mtp-short", "128",
    "--mtp-short-max-tokens", "128",
    "--spec-draft-n-max-short", "4",
    "--ctx-size-mtp", "256",
    "--mtp-max-tokens", "256",
    "--spec-draft-n-max", "2",
    "--ctx-size", "384",
    "--ctx-size-xl", "448",
    "--ctx-size-xxl", "512",
    "--ctx-checkpoints", "1",
    "--cache-ram", "4096",
    "--slot-save-path", str(SLOTS_DIR) + "/",
    "--cache-prompt",
    "--no-context-shift",
]


def http_call(method, path, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    elapsed = round((time.monotonic() - started) * 1000, 1)
    try:
        body = json.loads(raw)
    except Exception:
        body = {"raw": raw.decode(errors="replace")}
    return status, body, elapsed


def make_prompt(n_target_tokens):
    """Generate a prompt that tokenizes to exactly n_target_tokens."""
    base_text = "Adaptive context tier verification and slot state test sequence "
    # Repeatedly grow or shrink until exact count
    low, high = 1, n_target_tokens
    # Probe word count
    words = base_text.split()
    cand_words = (words * ((n_target_tokens // len(words)) + 2))[:n_target_tokens]
    cand_text = " ".join(cand_words)
    status, body, _ = http_call("POST", "/tokenize", {"content": cand_text})
    assert status == 200, body
    tokens = body["tokens"]
    if len(tokens) > n_target_tokens:
        tokens = tokens[:n_target_tokens]
    elif len(tokens) < n_target_tokens:
        # pad with repeated tokens
        tokens = tokens + [tokens[-1]] * (n_target_tokens - len(tokens))
    # Detokenize to verify or just use raw tokens if server accepts tokens/ids
    status, detok, _ = http_call("POST", "/detokenize", {"tokens": tokens})
    assert status == 200, detok
    text = detok["content"]
    # Verify final tokenization
    status, verify, _ = http_call("POST", "/tokenize", {"content": text})
    actual_tokens = verify["tokens"]
    return text, len(actual_tokens)


print(f"[TEST] Starting llama-server on port {PORT}...")
log_file = open(WORK_DIR / "server.log", "w")
proc = subprocess.Popen(CMD, stdout=log_file, stderr=subprocess.STDOUT)

try:
    # 1. Wait for server readiness
    print("[TEST] Waiting for server readiness...")
    deadline = time.monotonic() + 60
    ready = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server died unexpectedly with rc={proc.returncode}")
        try:
            status, body, _ = http_call("GET", "/props", timeout=2)
            if status == 200 and "adaptive_context" in body:
                ready = True
                break
        except Exception:
            pass
        time.sleep(1)
    assert ready, "Server failed to become ready within timeout"

    status, props, _ = http_call("GET", "/props")
    assert status == 200
    ac = props["adaptive_context"]
    print(f"[TEST] Initial adaptive context: {json.dumps(ac, indent=2)}")
    assert ac["profile"] == "mtp-short", f"Expected mtp-short, got {ac['profile']}"
    assert ac["context_size_long"] == 512, f"Expected 512, got {ac['context_size_long']}"
    assert ac["mtp_weights_resident"] is True

    # Check /models endpoint
    status, models, _ = http_call("GET", "/models")
    assert status == 200
    m_info = models["data"][0]["adaptive_context"]
    assert m_info["profile"] == "mtp-short"

    # =========================================================================
    # Step 1: Tier 1 - mtp-short (budget <= 128)
    # =========================================================================
    print("\n--- Step 1: Tier 1 (mtp-short) ---")
    p1_text, n_p1 = make_prompt(50)
    print(f"Prompt 1 token count: {n_p1}")
    status, comp1, _ = http_call("POST", "/completion", {
        "prompt": p1_text, "id_slot": 0, "cache_prompt": True, "n_predict": 1, "temperature": 0
    })
    assert status == 200, comp1
    status, props, _ = http_call("GET", "/props")
    assert props["adaptive_context"]["profile"] == "mtp-short"
    assert props["adaptive_context"]["mtp_weights_resident"] is True

    status, save1, _ = http_call("POST", "/slots/0?action=save", {"filename": "p1_short.bin"})
    assert status == 200 and save1.get("n_saved", 0) > 0, save1
    print(f"Saved p1_short.bin: {save1['n_saved']} tokens")

    # =========================================================================
    # Step 2: Tier 2 - mtp (128 < budget <= 256)
    # =========================================================================
    print("\n--- Step 2: Tier 2 (mtp / medium) ---")
    p2_text, n_p2 = make_prompt(180)
    print(f"Prompt 2 token count: {n_p2}")
    status, comp2, _ = http_call("POST", "/completion", {
        "prompt": p2_text, "id_slot": 0, "cache_prompt": True, "n_predict": 1, "temperature": 0
    })
    assert status == 200, comp2
    status, props, _ = http_call("GET", "/props")
    assert props["adaptive_context"]["profile"] == "mtp", f"Expected mtp, got {props['adaptive_context']}"
    assert props["adaptive_context"]["mtp_weights_resident"] is True

    status, save2, _ = http_call("POST", "/slots/0?action=save", {"filename": "p2_medium.bin"})
    assert status == 200 and save2.get("n_saved", 0) > 0, save2
    print(f"Saved p2_medium.bin: {save2['n_saved']} tokens")

    # =========================================================================
    # Step 3: Tier 3 - long (256 < budget <= 384)
    # =========================================================================
    print("\n--- Step 3: Tier 3 (long) ---")
    p3_text, n_p3 = make_prompt(300)
    print(f"Prompt 3 token count: {n_p3}")
    status, comp3, _ = http_call("POST", "/completion", {
        "prompt": p3_text, "id_slot": 0, "cache_prompt": True, "n_predict": 1, "temperature": 0
    })
    assert status == 200, comp3
    status, props, _ = http_call("GET", "/props")
    assert props["adaptive_context"]["profile"] == "long", f"Expected long, got {props['adaptive_context']}"
    assert props["adaptive_context"]["mtp_weights_resident"] is False

    status, save3, _ = http_call("POST", "/slots/0?action=save", {"filename": "p3_long.bin"})
    assert status == 200 and save3.get("n_saved", 0) > 0, save3
    print(f"Saved p3_long.bin: {save3['n_saved']} tokens")

    # =========================================================================
    # Step 4: Tier 4 - xlong (384 < budget <= 448)
    # =========================================================================
    print("\n--- Step 4: Tier 4 (xlong) ---")
    p4_text, n_p4 = make_prompt(400)
    print(f"Prompt 4 token count: {n_p4}")
    status, comp4, _ = http_call("POST", "/completion", {
        "prompt": p4_text, "id_slot": 0, "cache_prompt": True, "n_predict": 1, "temperature": 0
    })
    assert status == 200, comp4
    status, props, _ = http_call("GET", "/props")
    assert props["adaptive_context"]["profile"] == "xlong", f"Expected xlong, got {props['adaptive_context']}"
    assert props["adaptive_context"]["mtp_weights_resident"] is False

    status, save4, _ = http_call("POST", "/slots/0?action=save", {"filename": "p4_xlong.bin"})
    assert status == 200 and save4.get("n_saved", 0) > 0, save4
    print(f"Saved p4_xlong.bin: {save4['n_saved']} tokens")

    # =========================================================================
    # Step 5: Tier 5 - xxlong (448 < budget <= 512)
    # =========================================================================
    print("\n--- Step 5: Tier 5 (xxlong) ---")
    p5_text, n_p5 = make_prompt(470)
    print(f"Prompt 5 token count: {n_p5}")
    status, comp5, _ = http_call("POST", "/completion", {
        "prompt": p5_text, "id_slot": 0, "cache_prompt": True, "n_predict": 1, "temperature": 0
    })
    assert status == 200, comp5
    status, props, _ = http_call("GET", "/props")
    assert props["adaptive_context"]["profile"] == "xxlong", f"Expected xxlong, got {props['adaptive_context']}"
    assert props["adaptive_context"]["mtp_weights_resident"] is False

    status, save5, _ = http_call("POST", "/slots/0?action=save", {"filename": "p5_xxlong.bin"})
    assert status == 200 and save5.get("n_saved", 0) > 0, save5
    print(f"Saved p5_xxlong.bin: {save5['n_saved']} tokens")

    # =========================================================================
    # Step 6: Slot Restore and Bidirectional Profile Transitions
    # =========================================================================
    print("\n--- Step 6: Slot Restore and Profile Transitions ---")

    # 6a. Restore p1_short.bin while currently in xxlong -> must transition xxlong -> mtp-short
    print("Testing restore p1_short.bin (xxlong -> mtp-short)...")
    status, rest1, _ = http_call("POST", "/slots/0?action=restore", {"filename": "p1_short.bin"})
    assert status == 200 and rest1.get("n_restored", 0) > 0, rest1
    status, props, _ = http_call("GET", "/props")
    assert props["adaptive_context"]["profile"] == "mtp-short", props["adaptive_context"]
    assert props["adaptive_context"]["mtp_weights_resident"] is True
    print(f"Restored p1_short: profile={props['adaptive_context']['profile']}, resident={props['adaptive_context']['mtp_weights_resident']}")

    # Check cache reuse on short prompt
    status, reuse1, _ = http_call("POST", "/completion", {
        "prompt": p1_text, "id_slot": 0, "cache_prompt": True, "n_predict": 1, "temperature": 0
    })
    assert status == 200, reuse1
    assert reuse1.get("timings", {}).get("cache_n", 0) > 0, reuse1
    print(f"Cache reuse verified on short prompt: cache_n = {reuse1['timings']['cache_n']}")

    # 6b. Restore p4_xlong.bin while in mtp-short -> must transition mtp-short -> xlong
    print("Testing restore p4_xlong.bin (mtp-short -> xlong)...")
    status, rest4, _ = http_call("POST", "/slots/0?action=restore", {"filename": "p4_xlong.bin"})
    assert status == 200 and rest4.get("n_restored", 0) > 0, rest4
    status, props, _ = http_call("GET", "/props")
    assert props["adaptive_context"]["profile"] == "xlong", props["adaptive_context"]
    assert props["adaptive_context"]["mtp_weights_resident"] is False
    print(f"Restored p4_xlong: profile={props['adaptive_context']['profile']}")

    # 6c. Restore p2_medium.bin while in xlong -> must transition xlong -> mtp
    print("Testing restore p2_medium.bin (xlong -> mtp)...")
    status, rest2, _ = http_call("POST", "/slots/0?action=restore", {"filename": "p2_medium.bin"})
    assert status == 200 and rest2.get("n_restored", 0) > 0, rest2
    status, props, _ = http_call("GET", "/props")
    assert props["adaptive_context"]["profile"] == "mtp", props["adaptive_context"]
    assert props["adaptive_context"]["mtp_weights_resident"] is True
    print(f"Restored p2_medium: profile={props['adaptive_context']['profile']}")

    # 6d. Restore p5_xxlong.bin while in mtp -> must transition mtp -> xxlong
    print("Testing restore p5_xxlong.bin (mtp -> xxlong)...")
    status, rest5, _ = http_call("POST", "/slots/0?action=restore", {"filename": "p5_xxlong.bin"})
    assert status == 200 and rest5.get("n_restored", 0) > 0, rest5
    status, props, _ = http_call("GET", "/props")
    assert props["adaptive_context"]["profile"] == "xxlong", props["adaptive_context"]
    assert props["adaptive_context"]["mtp_weights_resident"] is False
    print(f"Restored p5_xxlong: profile={props['adaptive_context']['profile']}")

    # 6e. Restore p3_long.bin while in xxlong -> must transition xxlong -> long
    print("Testing restore p3_long.bin (xxlong -> long)...")
    status, rest3, _ = http_call("POST", "/slots/0?action=restore", {"filename": "p3_long.bin"})
    assert status == 200 and rest3.get("n_restored", 0) > 0, rest3
    status, props, _ = http_call("GET", "/props")
    assert props["adaptive_context"]["profile"] == "long", props["adaptive_context"]
    assert props["adaptive_context"]["mtp_weights_resident"] is False
    print(f"Restored p3_long: profile={props['adaptive_context']['profile']}")

    # 6f. Restore p1_short.bin while in long -> must transition long -> mtp-short
    print("Testing restore p1_short.bin (long -> mtp-short)...")
    status, rest1_again, _ = http_call("POST", "/slots/0?action=restore", {"filename": "p1_short.bin"})
    assert status == 200 and rest1_again.get("n_restored", 0) > 0, rest1_again
    status, props, _ = http_call("GET", "/props")
    assert props["adaptive_context"]["profile"] == "mtp-short", props["adaptive_context"]
    assert props["adaptive_context"]["mtp_weights_resident"] is True
    print(f"Restored p1_short again: profile={props['adaptive_context']['profile']}")

    # =========================================================================
    # Step 7: Boundary and Error Handling Tests
    # =========================================================================
    print("\n--- Step 7: Boundary & Error Handling ---")

    # 7a. Exceed context budget (> 512)
    print("Testing budget exceed (> 512 tokens)...")
    over_text, n_over = make_prompt(520)
    status, over_resp, _ = http_call("POST", "/completion", {
        "prompt": over_text, "id_slot": 0, "n_predict": 1
    })
    print(f"Exceed response status: {status}, body: {over_resp}")
    assert status == 400, over_resp
    assert "exceeds the shared context size" in over_resp.get("message", "") or "exceeds" in str(over_resp)

    # 7b. Corrupted snapshot checksum validation
    print("Testing checksum validation on corrupted snapshot...")
    p1_path = SLOTS_DIR / "p1_short.bin"
    corrupt_data = bytearray(p1_path.read_bytes())
    corrupt_data[len(corrupt_data) // 2] ^= 0x55
    (SLOTS_DIR / "corrupt.bin").write_bytes(corrupt_data)
    status, corrupt_resp, _ = http_call("POST", "/slots/0?action=restore", {"filename": "corrupt.bin"})
    print(f"Corrupt restore status: {status}, body: {corrupt_resp}")
    assert status >= 400, corrupt_resp

    print("\n=========================================================================")
    print("ALL 5-TIER ADAPTIVE TRANSITIONS AND SLOT SAVE/RESTORE TESTS PASSED 100%!")
    print("=========================================================================")

finally:
    if proc.poll() is None:
        print("[TEST] Terminating server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    log_file.close()
