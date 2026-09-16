#!/usr/bin/env python3
"""Small CPU regression for legacy route-state paths.

The source and speculative destination use different explicit slot stores. A
second case omits --slot-save-path entirely and checks the private fallback.
Both cases require the route transfer marker, but deliberately omit the
streaming target flag so the destination uses the legacy state API. The route
snapshot is retained in the shared store; cleanup must not erase it.
"""

import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from router_state_streaming import Server


def make_preset(path, model, source_store, target_store, public):
    source_option = f"slot-save-path = {source_store}\n" if source_store else ""
    target_option = f"slot-save-path = {target_store}\n" if target_store else ""
    path.write_text(f"""version = 1
[*]
model = {model}
override-kv = qwen35.context_length=int:1024
load-mode = none
parallel = 1
device = none
gpu-layers = 0
flash-attn = on
fit = off
cache-prompt = true
context-shift = false
metrics = true
cache-type-k = q4_0
cache-type-v = q4_0
[source]
load-on-startup = true
route-group = {public}
route-max-tokens = 512
ctx-size = 512
spec-type = none
batch-size = 256
ubatch-size = 256
{source_option}
[target]
route-group = {public}
ctx-size = 1024
spec-type = draft-mtp
spec-draft-n-max = 2
spec-draft-type-k = q4_0
spec-draft-type-v = q4_0
batch-size = 64
ubatch-size = 64
{target_option}""")


def run_case(args, root, explicit):
    root.mkdir(parents=True, exist_ok=False)
    model = str(Path(args.model).resolve())
    fixture = root / "qwen35-http.gguf"
    subprocess.run([str(Path(args.server).resolve().parent / "test-state-file-stream"),
                    "--make-http-fixture", model, str(fixture)], check=True)
    source_store = root / "source-store" if explicit else None
    target_store = root / "target-store" if explicit else None
    if source_store:
        source_store.mkdir()
        target_store.mkdir()
    public = "legacy-route-test"
    preset = root / "models.ini"
    make_preset(preset, fixture, source_store, target_store, public)
    server_args = SimpleNamespace(server=args.server, gpu=False, port=0, timeout=args.timeout,
                                  slot_save_path="")
    server = Server(server_args, root, preset)
    prompt = [3 + (i * 17) % 50 for i in range(400)]
    try:
        first = server.request("/completion", {
            "model": public, "prompt": prompt, "n_predict": 32,
            "cache_prompt": True, "return_tokens": True,
            "temperature": 0, "seed": 1234, "ignore_eos": True},
            conversation="legacy")
        generated = first.get("tokens", [])
        if len(generated) != 32:
            raise AssertionError(f"source generation length: {len(generated)}")
        extended = (prompt + generated + [7, 11, 19, 23] * 17)[:500]
        second = server.request("/completion", {
            "model": public, "prompt": extended, "n_predict": 32,
            "cache_prompt": True, "return_tokens": True,
            "temperature": 0, "seed": 1234, "ignore_eos": True},
            conversation="legacy")
        if len(second.get("tokens", [])) != 32 or second.get("model") != public:
            raise AssertionError(f"target generation response: {second}")
        slots = server.request("/slots?model=" + public)
        slot = slots[0]
        if not slot.get("speculative") or "draft-mtp" not in slot.get("params", {}).get("speculative.types", ""):
            raise AssertionError(f"speculative destination was not loaded: {slots}")
        text = "".join(server.lines)
        if "saved route state for conversation legacy" not in text or \
                "restored route state for conversation legacy" not in text:
            raise AssertionError("legacy route save/restore was not published")
        if "router streaming save" in text or "router streaming restore" in text:
            raise AssertionError("legacy destination unexpectedly used streaming route API")
        return {"case": "explicit-stores" if explicit else "no-explicit-store",
                "source_store": str(source_store) if source_store else None,
                "target_store": str(target_store) if target_store else None,
                "target_speculative": True}
    finally:
        server.close()
        if explicit:
            source_files = list(source_store.glob("slot-*.bin"))
            target_files = list(target_store.glob("slot-*.bin"))
            if not source_files or target_files:
                raise AssertionError(f"unified route store placement changed: source={source_files} target={target_files}")
        else:
            fallback_dirs = list(root.glob("llama-router-state-*"))
            if not fallback_dirs or not any(list(directory.glob("slot-*.bin")) for directory in fallback_dirs):
                raise AssertionError("fallback unified route snapshot was deleted on shutdown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    base = Path(args.work_dir).resolve()
    base.mkdir(parents=True, exist_ok=False)
    results = [
        run_case(args, base / "explicit", True),
        run_case(args, base / "fallback", False),
    ]
    (base / "results.json").write_text(json.dumps(results, indent=2))
    print("PASS: legacy route transfer with different stores and without explicit slot-save-path")


if __name__ == "__main__":
    main()
