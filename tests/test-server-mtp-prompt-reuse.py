#!/usr/bin/env python3
"""Integration regression for MTP prompt reuse with recurrent target state.

Run against a locally started llama-server with draft-mtp and a model with
recurrent layers. The caller controls the model, device, and server options.
Alternatively, --disk-save-restore starts three sequential server processes using
the supplied adaptive draft-mtp command and raw token fixture. It tests a full
native disk snapshot across restart, followed by a nonempty B256 prompt suffix.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request


def completion(url, prompt, n_predict=1, cache_prompt=True):
    request = urllib.request.Request(
        url.rstrip("/") + "/completion",
        json.dumps({
            "prompt": prompt,
            "n_predict": n_predict,
            "n_probs": 256 if n_predict == 1 else 10,
            "post_sampling_probs": False,
            "cache_prompt": cache_prompt,
            "temperature": 1.0 if n_predict == 1 else 0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "repeat_last_n": 0,
            "seed": 42,
        }).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    assert "error" not in result, result
    return result


def probabilities(result):
    return {item["id"]: item["logprob"] for item in
            result["completion_probabilities"][0]["top_logprobs"]}


def assert_same_scores(actual, expected):
    actual, expected = probabilities(actual), probabilities(expected)
    assert actual.keys() == expected.keys(), (actual, expected)
    for token, value in expected.items():
        assert abs(actual[token] - value) < 1e-4, (token, actual[token], value)


def live_reuse(url):
    base = ("Classify this support ticket: I was charged twice for my subscription. "
            "Please refund the duplicate payment. Answer one category: ")
    prompt = base + "billing, shipping, access, technical. Category:"
    alternate = base + "technical, access, shipping, billing. Category:"
    first = completion(url, prompt)
    repeat = completion(url, prompt)
    assert repeat["timings"]["prompt_n"] == 1, (first["timings"], repeat["timings"])
    assert_same_scores(repeat, first)
    changed_tail = base + "billing, shipping, access, technical. Category?"
    changed_cached = completion(url, changed_tail)
    changed_fresh = completion(url, changed_tail, cache_prompt=False)
    assert_same_scores(changed_cached, changed_fresh)
    completion(url, alternate)
    again = completion(url, prompt)
    assert_same_scores(again, first)

    # Exercise actual MTP drafting after prompt reuse, not just prefill logits.
    fresh_generation = completion(url, prompt, n_predict=24, cache_prompt=False)
    cached_generation = completion(url, prompt, n_predict=24)
    assert cached_generation["content"] == fresh_generation["content"], (
        cached_generation["content"], fresh_generation["content"])
    assert_same_scores(cached_generation, fresh_generation)
    print("MTP cached repeated and edited prompts match initial probabilities and generation")
    print("prompt latency: cold %.2f ms, cached %.2f ms; generated tokens: cold %.2f ms, cached %.2f ms" % (
        first["timings"]["prompt_ms"], repeat["timings"]["prompt_ms"],
        fresh_generation["timings"]["predicted_ms"], cached_generation["timings"]["predicted_ms"]))


def request_json(url, endpoint, data=None, timeout=180):
    request = urllib.request.Request(url + endpoint, None if data is None else json.dumps(data).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise AssertionError((endpoint, error.code, error.read().decode(errors="replace"))) from error
    assert "error" not in result, result
    return result


@contextlib.contextmanager
def fresh_server(command, output, name, timeout, omit_auxiliary=False):
    # The test owns all lifetimes: an in-process erase/restore cannot reproduce
    # the restored slot with no pre-existing in-memory checkpoint or carry.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    argv = command + ["--host", "127.0.0.1", "--port", str(port), "--parallel", "1",
                      "--slots", "--slot-save-path", str(output / "slots") + "/",
                      "--batch-size", "256", "--ubatch-size", "256"]
    for profile in ("short", "medium", "long", "xlong", "xxlong", "xxxlong"):
        argv += ["--batch-size-" + profile, "256", "--ubatch-size-" + profile, "256"]
    log = output / (name + ".log")
    (output / (name + ".command.json")).write_text(json.dumps(argv, indent=2) + "\n")
    with log.open("w") as stream:
        env = dict(os.environ)
        if omit_auxiliary:
            env["LLAMA_ADAPTIVE_SLOT_SAVE_CHECKPOINTS"] = "0"
        (output / (name + ".save-policy.json")).write_text(json.dumps({
            "LLAMA_ADAPTIVE_SLOT_SAVE_CHECKPOINTS": env.get("LLAMA_ADAPTIVE_SLOT_SAVE_CHECKPOINTS"),
        }) + "\n")
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=env)
        try:
            url = "http://127.0.0.1:" + str(port)
            deadline = time.monotonic() + timeout
            while True:
                assert process.poll() is None, ("server exited during startup", log)
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                assert time.monotonic() < deadline, ("server startup timed out", log)
                time.sleep(0.1)
            yield url, log, process.pid
        finally:
            killed = False
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    killed = True
                    process.kill()
                    process.wait()
            status = {"pid": process.pid, "returncode": process.returncode, "killed": killed}
            (output / (name + ".exit.json")).write_text(json.dumps(status, indent=2) + "\n")
            assert not killed and process.returncode == 0, ("server did not exit normally", status, log)


def saved_mtp_prefix(path, expected_checkpoints=None):
    """Read bounded v2 metadata; native restore verifies the payload checksum."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        def number(fmt):
            raw = stream.read(struct.calcsize(fmt))
            assert len(raw) == struct.calcsize(fmt), "truncated slot header"
            return struct.unpack(fmt, raw)[0]

        def blob(read=False):
            length = number("<Q")
            assert length <= size - stream.tell(), "slot blob exceeds file"
            if read:
                assert length <= 4 * 1024 * 1024, "slot metadata exceeds bound"
                return stream.read(length)
            stream.seek(length, 1)
            return length

        assert stream.read(8) == b"LLAMASLT" and number("<I") == 2, "full adaptive v2 slot required"
        for _ in range(5):
            number("<I")  # profile and context geometry
        for _ in range(5):
            number("<Q")  # model instance and target/draft ownership
        for _ in range(3):
            blob()  # model identity and target/draft layouts
        target_pos, draft_pos, count = number("<i"), number("<i"), number("<Q")
        packed = blob(True)
        assert len(packed) >= 16 and len(packed) % 4 == 0, "invalid text-token envelope"
        words = struct.unpack("<" + "i" * (len(packed) // 4), packed)
        assert words[:2] == (-1, 1) and words[2] == count == len(words) - 4 and words[-1] == 0
        assert target_pos == draft_pos == count - 1, (target_pos, draft_pos, count)
        state_bytes = [blob() for _ in range(3)]
        assert all(n > 0 for n in state_bytes), "target, draft and MTP carry must all be saved"
        if expected_checkpoints is not None:
            assert number("<Q") == expected_checkpoints, "saved auxiliary checkpoint count differs"
        return list(words[3:-1])


def disk_save_restore(args):
    command = json.loads(args.server_command_json.read_text())
    assert isinstance(command, list) and command and all(isinstance(x, str) for x in command)
    raw = json.loads(args.tokens_json.read_text())
    tokens = raw if isinstance(raw, list) else raw["tokens"]
    assert all(type(x) is int and x >= 0 for x in tokens), "nonnegative token IDs required"
    assert len(tokens) >= args.prefix_tokens + 16 + args.suffix_tokens, "token source too short"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "slots").mkdir()

    def record(name, value):
        (output / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")

    def generate(url, prompt, count, name):
        result = request_json(url, "/completion", {
            "prompt": prompt, "n_predict": count, "cache_prompt": True,
            "id_slot": 0, "temperature": 0, "seed": 42, "ignore_eos": True,
            "return_tokens": True, "repeat_penalty": 1.0,
        }, timeout=args.timeout)
        record(name, result)
        assert not result.get("truncated"), result
        assert len(result["tokens"]) == result["tokens_predicted"] == count, result
        return result

    prefix = tokens[:args.prefix_tokens]
    filename = "mtp-prefix.bin"

    def snapshot_hashes():
        hashes = {}
        for name in (filename, filename + ".logits"):
            path = output / "slots" / name
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                hashes[name] = digest.hexdigest()
        assert filename in hashes, "saved snapshot disappeared"
        return hashes

    with fresh_server(command, output, "save", args.timeout, args.omit_aux_checkpoints) as (url, _, save_pid):
        warm = generate(url, prefix, 16, "warm.response")
        saved = request_json(url, "/slots/0?action=save", {"filename": filename}, args.timeout)
        record("save.response", saved)
        frozen = saved_mtp_prefix(output / "slots" / filename, 0 if args.omit_aux_checkpoints else None)
        assert saved["n_saved"] == len(frozen) > 0, saved
        assert frozen == (prefix + warm["tokens"])[:len(frozen)], "saved token trajectory changed"
        assert args.prefix_tokens <= len(frozen) <= args.prefix_tokens + 16, len(frozen)
    frozen_hashes = snapshot_hashes()
    record("save.snapshot.sha256", frozen_hashes)
    prompt = frozen + tokens[len(frozen):len(frozen) + args.suffix_tokens]
    continuations = []
    # Use identical decode cadence and independent fresh processes for both
    # continuations. A cold prefill groups the generated prefix differently on
    # recurrent models, so it is not an exact numerical oracle for this test.
    for name in ("restore-a", "restore-b"):
        with fresh_server(command, output, name, args.timeout, args.omit_aux_checkpoints) as (url, log, pid):
            restored = request_json(url, "/slots/0?action=restore", {"filename": filename}, args.timeout)
            record(name + ".restore.response", restored)
            assert restored["n_restored"] == len(frozen) and restored["n_read"] > 0, restored
            assert re.search(r"adaptive slot restore published[^\n]*complete_mtp=1 bootstrap=0",
                             log.read_text(errors="replace")), "restore did not publish complete resident MTP carry"
            # Before the fix this first nonempty suffix erased the restored carry
            # and failed with 'resident MTP carry position -1 does not precede batch'.
            generation = generate(url, prompt, 32, name + ".completion.response")
            timing = generation["timings"]
            assert timing["cache_n"] == len(frozen) and timing["prompt_n"] == args.suffix_tokens, timing
            assert timing["draft_n"] > 0, "MTP was bypassed after restore"
        hashes = snapshot_hashes()
        record(name + ".snapshot.sha256", hashes)
        assert hashes == frozen_hashes, ("restore modified frozen snapshot or logits", hashes, frozen_hashes)
        continuations.append({"pid": pid, "restore": restored, "generation": generation, "snapshot_sha256": hashes})
    actual, reference = [item["generation"] for item in continuations]
    report = {"save_pid": save_pid, "save": saved, "snapshot_sha256": frozen_hashes,
              "continuations": continuations}
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    assert actual["tokens"] == reference["tokens"], (actual["tokens"], reference["tokens"])
    assert actual["content"] == reference["content"], (actual["content"], reference["content"])
    print("PASS: full MTP disk save -> two independent fresh restores -> identical nonempty B256 suffix; "
          "snapshot unchanged, cache reused, MTP drafted, restored continuation token IDs reproducible")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", help="URL of a running draft-mtp llama-server")
    parser.add_argument("--disk-save-restore", action="store_true",
                        help="own three server lifetimes to test full adaptive MTP disk restore reproducibility")
    parser.add_argument("--omit-aux-checkpoints", action="store_true",
                        help="save full target/draft/MTP state without auxiliary checkpoints; runtime cadence is unchanged")
    parser.add_argument("--server-command-json", type=Path,
                        help="JSON argv array for an adaptive draft-mtp server; model/preset options supplied by caller")
    parser.add_argument("--tokens-json", type=Path, help="raw token ID array or {tokens: [...]} source")
    parser.add_argument("--prefix-tokens", type=int, default=1024)
    parser.add_argument("--suffix-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, help="new directory for isolated slot files, commands, logs and result")
    parser.add_argument("--timeout", type=float, default=180, help="startup/request timeout in seconds")
    args = parser.parse_args()
    if args.disk_save_restore:
        if args.url or not all((args.server_command_json, args.tokens_json, args.output)):
            parser.error("disk mode requires --server-command-json, --tokens-json and --output, without URL")
        if min(args.prefix_tokens, args.suffix_tokens, args.timeout) <= 0:
            parser.error("prefix, suffix and timeout must be positive")
        disk_save_restore(args)
    elif args.url:
        live_reuse(args.url)
    else:
        parser.error("provide URL or --disk-save-restore")


if __name__ == "__main__":
    main()
