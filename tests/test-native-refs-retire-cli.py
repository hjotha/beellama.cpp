#!/usr/bin/env python3
"""CPU regression for `llama-native-refs retire` (no Xbox, no network).

The binary is passed by CTest. Host files here are raw native frames (no
route-state manifest), so the CLI's key check hashes the whole file; that is the
same staged-prefix rule the save path journals. Published canonical envelopes
are exercised by test-server-native-refs.cpp and the production path.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BIN = Path(sys.argv[1])

KEY_A = "3c1ac4014f51a0ff550c3e1346463623140e22f8c9a21bba80ff66d6b09fc582"
KEY_PIN = "4d5532221ee86514043d1cabb790afb0b09e757ff11b824931b0de3bb0b24fc3"
KEY_UNKNOWN = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def record(state: str, key: str, host_file: str = "", source: str = "explicit") -> dict:
    return {
        "schema": "rkva-native-refs",
        "version": 1,
        "state": state,
        "xbox_key": key,
        "host_file": host_file,
        "n_tokens": 0 if state == "retired" else 128,
        "split_p": 0 if state == "retired" else 96,
        "native": True,
        "timestamp": "2026-11-01T00:00:00Z",
        "source": source,
    }


def write_journal(store: Path, records) -> Path:
    path = store / ".rkva-native-refs.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    return path


def read_lines(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(BIN), *args], capture_output=True, text=True, check=False)


def make_store(root: Path, name: str):
    store = root / name
    store.mkdir()
    return store


def raw_host(store: Path, name: str, payload: bytes):
    path = store / name
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_dry_run_default(root: Path) -> None:
    store = make_store(root, "dry-run")
    host, key = raw_host(store, "a.bin", b"raw-native-frame-dry-run")
    journal = write_journal(store, [record("committed", key, "a.bin")])
    before = journal.read_bytes()

    result = run("retire", "--store", str(store), "--key", key)
    assert result.returncode == 0, result.stderr
    assert host.exists(), "dry-run must not remove the host file"
    assert journal.read_bytes() == before, "dry-run must not append"
    assert "would delete host file" in result.stdout
    assert "dry-run" in result.stdout
    assert "Xbox object is not touched" in result.stdout
    assert key in result.stdout

    # Explicit --dry-run behaves the same as the default.
    result = run("retire", "--store", str(store), "--key", key, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert host.exists() and journal.read_bytes() == before


def test_execute_and_idempotent_rerun(root: Path) -> None:
    store = make_store(root, "execute")
    host, key = raw_host(store, "a.bin", b"raw-native-frame-execute")
    journal = write_journal(store, [record("committed", key, "a.bin")])

    result = run("retire", "--store", str(store), "--key", key, "--execute")
    assert result.returncode == 0, result.stderr
    assert not host.exists(), "the verified host file must be removed"
    lines = read_lines(journal)
    assert len(lines) == 2, lines
    retired = lines[-1]
    assert retired["state"] == "retired"
    assert retired["source"] == "retired"
    assert retired["xbox_key"] == key
    assert retired["host_file"] == "a.bin"
    assert retired["n_tokens"] == 0 and retired["split_p"] == 0
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", retired["timestamp"])

    # The audit reports the key separately from committed/pinned/orphan lists.
    result = run("audit", "--store", str(store), "--json")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["retired"] == [key], report
    assert key not in report["referenced"]
    assert key not in report["pinned"]
    assert key not in report["pending_only"]
    assert key not in report["orphan_candidates"]

    # Re-running is a no-op: no second retired record and no touching another
    # journal line.
    before = journal.read_bytes()
    result = run("retire", "--store", str(store), "--key", key, "--execute")
    assert result.returncode == 0, result.stderr
    assert "already retired" in result.stdout
    assert journal.read_bytes() == before


def test_pending_refused(root: Path) -> None:
    store = make_store(root, "pending")
    host, key = raw_host(store, "a.bin", b"raw-native-frame-pending")
    journal = write_journal(store, [record("pending", key, "a.bin")])
    before = journal.read_bytes()
    result = run("retire", "--store", str(store), "--key", key, "--execute")
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "pending" in result.stderr
    assert host.exists() and journal.read_bytes() == before


def test_unknown_key_refused(root: Path) -> None:
    store = make_store(root, "unknown")
    journal = write_journal(store, [record("committed", KEY_A, "a.bin")])
    before = journal.read_bytes()
    result = run("retire", "--store", str(store), "--key", KEY_UNKNOWN, "--execute")
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "no record" in result.stderr
    assert journal.read_bytes() == before


def test_pinned_requires_force(root: Path) -> None:
    store = make_store(root, "pinned")
    host, key = raw_host(store, "p.bin", b"raw-native-frame-pinned")
    # The key is derived from the host content, so the pin is not a baseline
    # key; the pin refusal itself is what this exercises.
    journal = write_journal(store, [record("pinned", key, "", "migrated")])
    before = journal.read_bytes()
    result = run("retire", "--store", str(store), "--key", key, "--execute",
                 "--host-file", str(host))
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "pinned" in result.stderr
    assert host.exists() and journal.read_bytes() == before

    result = run("retire", "--store", str(store), "--key", key, "--execute",
                 "--host-file", str(host), "--force-pinned")
    assert result.returncode == 0, result.stderr
    assert not host.exists()
    lines = read_lines(journal)
    assert lines[-1]["state"] == "retired"
    assert lines[-1]["host_file"] == "p.bin"


def test_pinned_without_host_file(root: Path) -> None:
    store = make_store(root, "pinned-no-host")
    journal = write_journal(store, [record("pinned", KEY_PIN, "", "migrated")])
    # A known campaign key with no host file: --force-pinned is mandatory and
    # the retired record carries no host name.
    refused = run("retire", "--store", str(store), "--key", KEY_PIN, "--execute")
    assert refused.returncode == 2, (refused.returncode, refused.stderr)
    assert "pinned" in refused.stderr
    result = run("retire", "--store", str(store), "--key", KEY_PIN, "--execute",
                 "--force-pinned")
    assert result.returncode == 0, result.stderr
    lines = read_lines(journal)
    assert lines[-1]["state"] == "retired"
    assert lines[-1]["source"] == "retired"
    assert lines[-1]["host_file"] == ""
    report = json.loads(run("audit", "--store", str(store), "--json").stdout)
    assert report["retired"] == [KEY_PIN]
    assert report["pinned"] == []


def test_key_mismatch_refused(root: Path) -> None:
    store = make_store(root, "mismatch")
    journal_key = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    host, other_key = raw_host(store, "a.bin", b"raw-native-frame-mismatch")
    assert other_key != journal_key
    journal = write_journal(store, [record("committed", journal_key, "a.bin")])
    before = journal.read_bytes()
    result = run("retire", "--store", str(store), "--key", journal_key, "--execute")
    assert result.returncode != 0
    assert "does not match" in result.stderr
    assert host.exists() and journal.read_bytes() == before


def test_missing_host_file_recovers(root: Path) -> None:
    store = make_store(root, "missing-host")
    _, key = raw_host(store, "a.bin", b"raw-native-frame-missing")
    (store / "a.bin").unlink()
    journal = write_journal(store, [record("committed", key, "a.bin")])
    result = run("retire", "--store", str(store), "--key", key, "--execute")
    assert result.returncode == 0, result.stderr
    assert "already absent" in result.stdout
    assert read_lines(journal)[-1]["state"] == "retired"


def test_host_file_must_be_inside_store(root: Path) -> None:
    store = make_store(root, "outside")
    host, key = raw_host(store, "a.bin", b"raw-native-frame-outside")
    outside = root / "outside.bin"
    outside.write_bytes(b"raw-native-frame-outside")
    journal = write_journal(store, [record("committed", key, "a.bin")])
    before = journal.read_bytes()
    result = run("retire", "--store", str(store), "--key", key,
                 "--host-file", str(outside))
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "direct child" in result.stderr
    assert host.exists() and outside.exists() and journal.read_bytes() == before


def test_symlink_refused(root: Path) -> None:
    store = make_store(root, "symlink")
    payload = b"raw-native-frame-symlink"
    target = store / "target.bin"
    target.write_bytes(payload)
    link = store / "a.bin"
    try:
        link.symlink_to(target.name)
    except (OSError, NotImplementedError):
        return  # platform without symlink support
    key = hashlib.sha256(payload).hexdigest()
    journal = write_journal(store, [record("committed", key, "a.bin")])
    before = journal.read_bytes()
    result = run("retire", "--store", str(store), "--key", key, "--execute")
    assert result.returncode != 0
    assert link.exists() and target.exists()
    assert journal.read_bytes() == before


def test_uppercase_key(root: Path) -> None:
    store = make_store(root, "uppercase")
    host, key = raw_host(store, "a.bin", b"raw-native-frame-uppercase")
    journal = write_journal(store, [record("committed", key, "a.bin")])
    result = run("retire", "--store", str(store), "--key", key.upper(), "--execute")
    assert result.returncode == 0, result.stderr
    assert not host.exists()
    assert read_lines(journal)[-1]["xbox_key"] == key


def main() -> int:
    if not BIN.is_file():
        raise SystemExit(f"missing llama-native-refs binary: {BIN}")
    tests = [
        test_dry_run_default,
        test_execute_and_idempotent_rerun,
        test_pending_refused,
        test_unknown_key_refused,
        test_pinned_requires_force,
        test_pinned_without_host_file,
        test_key_mismatch_refused,
        test_missing_host_file_recovers,
        test_host_file_must_be_inside_store,
        test_symlink_refused,
        test_uppercase_key,
    ]
    with tempfile.TemporaryDirectory(prefix="native-refs-retire-") as temp:
        root = Path(temp)
        for test in tests:
            test(root)
            print(f"PASS: {test.__name__}")
    print("PASS: llama-native-refs retire CPU regression")
    return 0


if __name__ == "__main__":
    sys.exit(main())
