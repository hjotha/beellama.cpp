#!/usr/bin/env python3
"""Retire one native snapshot on both sides of the pair.

This is the operator wrapper around two independent actions:

1. delete exactly one `LocalState\\rkva-snapshots-v2\\<key>` folder through the
   Xbox Device Portal (WDP), and
2. run `llama-native-refs retire --execute` on the host store so the durable
   journal records the retirement and the host file is removed.

The safe default is inert: without `--execute` the tool only reads the local
journal and prints the decision and the exact WDP URL. It never contacts the
Xbox and never starts, stops or deletes anything in dry-run.

Execution requirements (deliberately conservative):

* the xllama app must already be stopped; this tool does not stop it. Stopping
  a live app could abort an active request, and the WDP delete plus the host
  journal append are not atomic across the two machines;
* `--key` is mandatory and must be 64 lowercase hex characters. There is no
  wildcard, prefix or directory mode, so no other object is touched;
* the key must already exist in the host journal; a pending record refuses the
  retirement, and a pinned/key-known campaign key additionally requires
  `--force-pinned`;
* the WDP delete happens first, then the host CLI. A crash between them leaves
  the remote object gone while the journal still says committed/pinned; rerun
  the host CLI (`llama-native-refs retire --execute`) to converge. The tool
  prints that recovery command if the CLI step fails.

The durable record of a deleted key is the `retired` line appended to
`<slot-save-path>/.rkva-native-refs.jsonl` by the CLI. This tool prints the same
key as a JSON receipt but creates no extra files or Xbox objects.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import ssl
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

JOURNAL_NAME = ".rkva-native-refs.jsonl"
SNAPSHOT_ROOT = "\\LocalState\\rkva-snapshots-v2"
KEY_RE = re.compile(r"^[0-9a-f]{64}$")
STATES = ("pending", "committed", "pinned", "retired")

# Six published campaign keys plus the same-build key recorded on 2026-11-01.
# These are pinned by the migration CLI; deleting one is allowed only with the
# explicit --force-pinned override (and the key is still passed one at a time).
KNOWN_KEYS = frozenset({
    "3c1ac4014f51a0ff550c3e1346463623140e22f8c9a21bba80ff66d6b09fc582",
    "3c7218bdc40af2312e5dba397b7d2a846d3f4beb717f760353cdcc7b6f627ab1",
    "4d5532221ee86514043d1cabb790afb0b09e757ff11b824931b0de3bb0b24fc3",
    "674016a7971750acd47f65809f25c08733712390e717792ec41685224bb2378b",
    "68b109a4d14bbd26d7937cc9f8268a6bdb706584239531c440762857036d82c5",
    "ef1c05ae42b5d79406ba3cc48b15f8107049a5c07e76339b23ad9fd96b972dd7",
    "fff30e532caeac47750f17351664f3b392ffdafc0801514d09395c5326bee09a",
})


class RetireError(RuntimeError):
    """A refused or failed retirement; the message is operator-facing."""


def normalize_key(key: str) -> str:
    return key.strip().lower()


def valid_key(key: str) -> bool:
    return bool(KEY_RE.fullmatch(key))


def journal_path(store: str) -> Path:
    return Path(store) / JOURNAL_NAME


def load_journal(store: str) -> List[Dict[str, object]]:
    """Read the append-only journal. A missing journal has no records."""
    path = journal_path(store)
    if not path.exists():
        return []
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RetireError(f"journal line {number} is not JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise RetireError(f"journal line {number} is not an object")
            if record.get("schema") != "rkva-native-refs" or record.get("version") != 1:
                raise RetireError(f"journal line {number} has an unsupported schema/version")
            if record.get("state") not in STATES:
                raise RetireError(f"journal line {number} has an unknown state")
            key = record.get("xbox_key")
            if not isinstance(key, str) or not valid_key(normalize_key(key)):
                raise RetireError(f"journal line {number} has an invalid xbox_key")
            record["xbox_key"] = normalize_key(key)
            records.append(record)
    return records


def fold_records(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Mirror the C++ fold: retired is key-wide, later records supersede it."""
    folded: List[Dict[str, object]] = []
    for record in records:
        key = record["xbox_key"]
        if record["state"] == "retired":
            folded = [cur for cur in folded if cur["xbox_key"] != key]
            folded.append(record)
            continue
        if any(cur["xbox_key"] == key and cur["state"] == "retired" for cur in folded):
            folded = [cur for cur in folded if cur["xbox_key"] != key]
        match = next((cur for cur in folded
                      if cur["xbox_key"] == key and cur.get("host_file") == record.get("host_file")),
                     None)
        if match is None:
            folded.append(record)
        else:
            folded[folded.index(match)] = record
    return folded


@dataclass(frozen=True)
class RetireDecision:
    allowed: bool
    reason: str
    key: str
    state: str
    host_files: Tuple[str, ...]
    pinned: bool = False
    known: bool = False
    already_retired: bool = False


def decide_retire(key: str, records: Sequence[Dict[str, object]],
                  force_pinned: bool = False) -> RetireDecision:
    """Pure host-side decision, matching the CLI's refusals."""
    normalized = normalize_key(key)
    if not valid_key(normalized):
        return RetireDecision(False, "key must be 64 lowercase hex characters",
                              normalized, "invalid", ())
    folded = fold_records(records)
    entries = [record for record in folded if record["xbox_key"] == normalized]
    if not entries:
        return RetireDecision(False, "key has no record in the journal",
                              normalized, "unknown", ())
    states = {str(record["state"]) for record in entries}
    host_files = tuple(sorted({str(record["host_file"]) for record in entries
                               if record.get("host_file")}))
    known = normalized in KNOWN_KEYS
    pinned = "pinned" in states
    state = "retired" if "retired" in states else (
        "pinned" if pinned else ("committed" if "committed" in states else "pending"))
    if "retired" in states:
        return RetireDecision(True, "already retired on the host; remote folder is not expected",
                              normalized, state, host_files, pinned, known, True)
    if "pending" in states:
        return RetireDecision(False, "a pending record exists for this key; finish the save first",
                              normalized, state, host_files, pinned, known)
    if (pinned or known) and not force_pinned:
        return RetireDecision(
            False,
            "key is pinned/known; pass --force-pinned to retire it explicitly",
            normalized, state, host_files, pinned, known)
    return RetireDecision(True, "eligible", normalized, state, host_files, pinned, known)


def snapshot_folder_url(base_url: str, pfn: str, key: str) -> str:
    """The exact WDP folder-delete URL for one key (no wildcard, no prefix)."""
    normalized = normalize_key(key)
    if not valid_key(normalized):
        raise RetireError("refusing to build a URL for a non-key filename")
    base = base_url.rstrip("/")
    query = (
        "knownfolderid=LocalAppData"
        f"&packagefullname={urllib.parse.quote(pfn, safe='')}"
        f"&path={urllib.parse.quote(SNAPSHOT_ROOT, safe='')}"
        f"&filename={urllib.parse.quote(normalized, safe='')}"
    )
    return f"{base}/api/filesystem/apps/folder?{query}"


def base_url_from_env(env: Optional[Dict[str, str]] = None) -> str:
    env = os.environ if env is None else env
    ip = env.get("XBOX_IP", "").strip()
    if not ip:
        raise RetireError("XBOX_IP is not set")
    return f"https://{ip}:11443"


def auth_header(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def xllama_processes(payload: str) -> List[Dict[str, object]]:
    """Return xllama entries from /api/resourcemanager/processes."""
    data = json.loads(payload)
    matches: List[Dict[str, object]] = []
    for process in data.get("Processes", []):
        image = str(process.get("ImageName", "")).lower()
        package = str(process.get("PackageFullName", ""))
        if "xllama" in image or "GianlucaMazza.xllama" in package or "VenereLabs.xllama" in package:
            matches.append(process)
    return matches


def http_request(url: str, method: str, auth: str, csrf: Optional[str] = None,
                 timeout: float = 30.0) -> Tuple[int, bytes, object]:
    request = urllib.request.Request(url, method=method)
    request.add_header("Authorization", auth)
    if csrf:
        request.add_header("X-CSRF-Token", csrf)
    if method in ("POST", "DELETE"):
        request.add_header("Content-Length", "0")
    context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
            return response.status, response.read(), response.headers
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers
    except OSError as exc:
        reason = getattr(exc, "reason", exc)
        raise RetireError(f"cannot reach {url}: {reason}") from exc


def fetch_csrf(base_url: str, auth: str, timeout: float) -> str:
    status, _, headers = http_request(base_url + "/", "GET", auth, timeout=timeout)
    if status != 200:
        raise RetireError(f"Device Portal root returned HTTP {status}")
    cookies = headers.get_all("Set-Cookie") if hasattr(headers, "get_all") else []
    for cookie in cookies:
        match = re.search(r"CSRF-Token=([^;,\s]+)", cookie)
        if match:
            return match.group(1)
    raise RetireError("Device Portal did not return a CSRF token")


def resolve_pfn(deploy: Path, env: Dict[str, str]) -> str:
    result = subprocess.run([str(deploy), "pfn"], capture_output=True, text=True,
                            env=env, check=False)
    pfn = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if result.returncode != 0 or not pfn:
        raise RetireError(f"cannot resolve the xllama package full name: {result.stderr.strip()}")
    return pfn


def require_app_stopped(base_url: str, auth: str, timeout: float) -> None:
    status, body, _ = http_request(
        f"{base_url}/api/resourcemanager/processes", "GET", auth, timeout=timeout)
    if status != 200:
        raise RetireError(f"process list returned HTTP {status}")
    try:
        running = xllama_processes(body.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise RetireError(f"process list is not valid JSON: {exc}") from exc
    if running:
        raise RetireError(
            "xllama is still running; stop it first (the host journal and the WDP delete "
            "are not atomic across machines): tools/xbox/scripts/deploy.sh stop-app <pfn>")

def delete_snapshot_folder(base_url: str, pfn: str, key: str, csrf: str, auth: str,
                           timeout: float) -> str:
    url = snapshot_folder_url(base_url, pfn, key)
    status, body, _ = http_request(url, "DELETE", auth, csrf=csrf, timeout=timeout)
    text = body.decode("utf-8", "replace").strip()
    if status in (200, 204):
        if "ErrorMessage" in text:
            raise RetireError(f"WDP folder delete reported an error: {text}")
        return "deleted"
    if status == 404:
        # The key is content-addressed and exact; a missing folder means the
        # host retired record can still be written safely.
        return "absent"
    raise RetireError(f"WDP folder delete returned HTTP {status}: {text}")


def default_cli() -> str:
    override = os.environ.get("LLAMA_NATIVE_REFS", "").strip()
    if override:
        return override
    root = Path(__file__).resolve().parents[3]
    return str(root / "build" / "bin" / "llama-native-refs")


def run_host_cli(cli: str, store: str, key: str, host_file: Optional[str],
                 force_pinned: bool) -> None:
    if not Path(cli).is_file():
        raise RetireError(f"host CLI not found: {cli} (pass --cli or set LLAMA_NATIVE_REFS)")
    command = [cli, "retire", "--store", store, "--key", key, "--execute"]
    if host_file:
        command += ["--host-file", host_file]
    if force_pinned:
        command.append("--force-pinned")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        if result.stderr:
            sys.stderr.write(result.stderr)
        raise RetireError(
            "remote folder is gone but the host journal was not retired; rerun: "
            + " ".join(command))
    print(f"host journal retired: {key}")


def validate_host_file(store: str, host_file: str, decision: RetireDecision) -> str:
    """Mirror the CLI's --host-file guards before any remote delete."""
    store_path = Path(store).resolve()
    host_path = Path(host_file)
    if host_path.is_symlink():
        raise RetireError("--host-file is a symbolic link; refusing")
    resolved = host_path.resolve()
    if resolved.parent != store_path:
        raise RetireError("--host-file must be a direct child of --store")
    if not resolved.is_file():
        raise RetireError(f"--host-file is not a regular file: {host_file}")
    name = resolved.name
    if decision.host_files and name not in decision.host_files:
        raise RetireError(
            "--host-file does not match the host file named by the journal for this key")
    return name


def plan_lines(store: str, decision: RetireDecision, url: str, host_file: Optional[str]) -> List[str]:
    lines = [
        f"store: {store}",
        f"journal: {journal_path(store)}",
        f"key: {decision.key}",
        f"decision: {decision.reason}",
        f"journal state: {decision.state}",
        f"known campaign key: {'yes' if decision.known else 'no'}",
        f"pinned: {'yes' if decision.pinned else 'no'}",
        f"host files named by the journal: {', '.join(decision.host_files) or '(none)'}",
    ]
    if host_file:
        lines.append(f"host file override: {host_file}")
    lines.append(f"WDP delete: DELETE {url}")
    lines.append("Xbox object delete and host journal retire are separate steps; the app must be stopped.")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retire exactly one native snapshot key (Xbox folder + host journal).")
    parser.add_argument("--store", required=True, help="slot-save-path containing the journal")
    parser.add_argument("--key", required=True, help="64 hex snapshot key")
    parser.add_argument("--pfn", help="xllama package full name (defaults to deploy.sh pfn)")
    parser.add_argument("--host-file", help="explicit host file to remove (passed to the CLI)")
    parser.add_argument("--force-pinned", action="store_true",
                        help="required to retire a pinned or known campaign key")
    parser.add_argument("--cli", default=default_cli(), help="llama-native-refs binary")
    parser.add_argument("--execute", action="store_true", help="perform the WDP delete and CLI retire")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    try:
        records = load_journal(args.store)
        decision = decide_retire(args.key, records, args.force_pinned)
        if not decision.allowed:
            raise RetireError(decision.reason)
        if decision.already_retired and not args.execute:
            print(f"note: {decision.key} is already retired in the journal")
        if args.host_file:
            if not Path(args.host_file).exists():
                raise RetireError(f"--host-file does not exist: {args.host_file}")
            validate_host_file(args.store, args.host_file, decision)
        pfn = args.pfn or "<pfn>"
        url = snapshot_folder_url("https://<XBOX_IP>:11443" if not os.environ.get("XBOX_IP")
                                  else base_url_from_env(), pfn, decision.key)
        for line in plan_lines(args.store, decision, url, args.host_file):
            print(line)
        if not args.execute:
            print("dry-run only: re-run with --execute to delete the folder and retire the host key")
            return 0

        env = dict(os.environ)
        base_url = base_url_from_env(env)
        user = env.get("XBOX_USER", "")
        password = env.get("XBOX_PASS", "")
        if not user or not password:
            raise RetireError("XBOX_USER and XBOX_PASS must be set")
        auth = auth_header(user, password)
        deploy = Path(__file__).resolve().with_name("deploy.sh")
        pfn = args.pfn or resolve_pfn(deploy, env)
        csrf = fetch_csrf(base_url, auth, args.timeout)
        # Point-in-time only: this is not a lease against a later app start.
        # The app must stay stopped until the host CLI finishes.
        require_app_stopped(base_url, auth, args.timeout)

        # Re-read immediately before the remote delete: a concurrent save may
        # have added a pending/committed record since the first decision.
        decision = decide_retire(args.key, load_journal(args.store), args.force_pinned)
        if not decision.allowed:
            raise RetireError(f"journal changed before delete: {decision.reason}")
        remote = delete_snapshot_folder(base_url, pfn, decision.key, csrf, auth, args.timeout)
        print(f"remote folder: {remote}")
        run_host_cli(args.cli, args.store, decision.key, args.host_file, args.force_pinned)
        print(json.dumps({
            "retired_key": decision.key,
            "remote": remote,
            "host_journal": "retired",
            "pfn": pfn,
        }, sort_keys=True))
        return 0
    except RetireError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
