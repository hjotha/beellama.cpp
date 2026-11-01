#!/usr/bin/env python3
"""Unit tests for the native-retire WDP URL and host-journal decision.

No network, no Xbox, no service. The module is imported from
tools/xbox/scripts/native-retire.py and only pure functions plus the dry-run
entry point are exercised.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "xbox" / "scripts" / "native-retire.py"
HEADER = ROOT / "tools" / "server" / "server-native-refs.h"

KEY_A = "3c1ac4014f51a0ff550c3e1346463623140e22f8c9a21bba80ff66d6b09fc582"
KEY_B = "fff30e532caeac47750f17351664f3b392ffdafc0801514d09395c5326bee09a"
KEY_UNKNOWN = "2222222222222222222222222222222222222222222222222222222222222222"
KEY_OTHER = "3333333333333333333333333333333333333333333333333333333333333333"
PFN = "GianlucaMazza.xllama_1.4.0.0_x64__abcde"


def load_tool():
    spec = importlib.util.spec_from_file_location("native_retire", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


retire = load_tool()


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


class KnownKeysTest(unittest.TestCase):
    def test_constants_match_header(self):
        source = HEADER.read_text(encoding="utf-8")
        block = source[source.index("kBaselineKeys"):source.index("}}", source.index("kBaselineKeys"))]
        keys = set(re.findall(r'"([0-9a-f]{64})"', block))
        self.assertEqual(keys, set(retire.KNOWN_KEYS))
        self.assertEqual(len(keys), 7)


class UrlTest(unittest.TestCase):
    def test_exact_single_key_url(self):
        url = retire.snapshot_folder_url("https://192.168.1.70:11443", PFN, KEY_A)
        self.assertEqual(
            url,
            "https://192.168.1.70:11443/api/filesystem/apps/folder"
            "?knownfolderid=LocalAppData"
            f"&packagefullname={PFN}"
            "&path=%5CLocalState%5Crkva-snapshots-v2"
            f"&filename={KEY_A}",
        )
        # Exactly one filename, no wildcard/prefix and no other key.
        self.assertEqual(url.count("filename="), 1)
        self.assertNotIn(KEY_B, url)
        self.assertNotIn("*", url)
        self.assertNotIn("tmp-", url)

    def test_uppercase_key_normalizes(self):
        url = retire.snapshot_folder_url("https://x:11443", PFN, KEY_A.upper())
        self.assertTrue(url.endswith(f"filename={KEY_A}"))

    def test_refuses_non_key(self):
        for bad in ("", "..", "../" + KEY_A, "*", KEY_A[:-1], KEY_A + "0", "/" + KEY_A,
                    "tmp-" + KEY_A, "rkva-snapshots-v2"):
            with self.subTest(bad=bad):
                with self.assertRaises(retire.RetireError):
                    retire.snapshot_folder_url("https://x:11443", PFN, bad)


class DecisionTest(unittest.TestCase):
    def test_committed_allowed(self):
        decision = retire.decide_retire(KEY_UNKNOWN, [record("committed", KEY_UNKNOWN, "a.bin")])
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.state, "committed")
        self.assertEqual(decision.host_files, ("a.bin",))
        self.assertFalse(decision.already_retired)

    def test_pending_refused(self):
        decision = retire.decide_retire(KEY_UNKNOWN, [record("pending", KEY_UNKNOWN, "a.bin")])
        self.assertFalse(decision.allowed)
        self.assertIn("pending", decision.reason)

    def test_unknown_refused(self):
        decision = retire.decide_retire(KEY_UNKNOWN, [record("committed", KEY_A, "a.bin")])
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.state, "unknown")

    def test_invalid_refused(self):
        decision = retire.decide_retire("not-a-key", [])
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.state, "invalid")

    def test_pinned_requires_force(self):
        journal = [record("pinned", KEY_A, "", "migrated")]
        self.assertFalse(retire.decide_retire(KEY_A, journal).allowed)
        forced = retire.decide_retire(KEY_A, journal, force_pinned=True)
        self.assertTrue(forced.allowed)
        self.assertTrue(forced.pinned)
        self.assertTrue(forced.known)

    def test_known_key_requires_force_even_when_committed(self):
        journal = [record("committed", KEY_A, "a.bin")]
        self.assertFalse(retire.decide_retire(KEY_A, journal).allowed)
        self.assertTrue(retire.decide_retire(KEY_A, journal, force_pinned=True).allowed)

    def test_retired_is_idempotent(self):
        decision = retire.decide_retire(KEY_UNKNOWN, [record("retired", KEY_UNKNOWN, "a.bin", "retired")])
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.already_retired)
        self.assertEqual(decision.state, "retired")

    def test_fold_retired_then_resave(self):
        journal = [
            record("committed", KEY_UNKNOWN, "a.bin"),
            record("retired", KEY_UNKNOWN, "a.bin", "retired"),
        ]
        self.assertTrue(retire.decide_retire(KEY_UNKNOWN, journal).already_retired)
        journal.append(record("committed", KEY_UNKNOWN, "a.bin"))
        decision = retire.decide_retire(KEY_UNKNOWN, journal)
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.already_retired)
        self.assertEqual(decision.state, "committed")

    def test_retired_does_not_touch_other_keys(self):
        journal = [
            record("retired", KEY_A, "a.bin", "retired"),
            record("committed", KEY_OTHER, "other.bin"),
        ]
        self.assertEqual(retire.decide_retire(KEY_OTHER, journal).host_files, ("other.bin",))
        self.assertTrue(retire.decide_retire(KEY_OTHER, journal).allowed)


class DeleteDecisionTest(unittest.TestCase):
    def call(self, status, body):
        with mock.patch.object(retire, "http_request", return_value=(status, body, {})):
            return retire.delete_snapshot_folder(
                "https://x:11443", PFN, KEY_A, "csrf", "Basic x", 1.0)

    def test_success(self):
        self.assertEqual(self.call(200, b""), "deleted")
        self.assertEqual(self.call(204, b""), "deleted")

    def test_absent(self):
        self.assertEqual(self.call(404, b""), "absent")

    def test_error_body_on_200_refused(self):
        with self.assertRaises(retire.RetireError):
            self.call(200, b'{"ErrorMessage":"Failed to delete the folder."}')

    def test_http_error_refused(self):
        with self.assertRaises(retire.RetireError):
            self.call(403, b"denied")


class ProcessTest(unittest.TestCase):
    def test_xllama_detection(self):
        payload = json.dumps({"Processes": [
            {"ImageName": "xllama.exe", "PackageFullName": PFN, "IsRunning": True},
            {"ImageName": "explorer.exe", "PackageFullName": "Microsoft.Windows"},
        ]})
        matches = retire.xllama_processes(payload)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["ImageName"], "xllama.exe")

    def test_no_xllama(self):
        payload = json.dumps({"Processes": [
            {"ImageName": "System", "PackageFullName": ""},
        ]})
        self.assertEqual(retire.xllama_processes(payload), [])


class JournalAndDryRunTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_journal(self, records):
        path = self.store / retire.JOURNAL_NAME
        path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
        return path

    def test_missing_journal_is_empty(self):
        self.assertEqual(retire.load_journal(str(self.store)), [])

    def test_bad_schema_refused(self):
        path = self.store / retire.JOURNAL_NAME
        path.write_text(json.dumps({"schema": "other", "version": 1}) + "\n", encoding="utf-8")
        with self.assertRaises(retire.RetireError):
            retire.load_journal(str(self.store))

    def test_host_file_validation(self):
        host = self.store / "a.bin"
        host.write_bytes(b"host-state")
        (self.store / "b.bin").write_bytes(b"host-state")
        decision = retire.decide_retire(KEY_UNKNOWN, [record("committed", KEY_UNKNOWN, "a.bin")])
        self.assertEqual(retire.validate_host_file(str(self.store), str(host), decision), "a.bin")

        outside = self.store.parent / (self.store.name + "-outside.bin")
        outside.write_bytes(b"host-state")
        try:
            with self.assertRaises(retire.RetireError):
                retire.validate_host_file(str(self.store), str(outside), decision)
        finally:
            outside.unlink()
        with self.assertRaises(retire.RetireError):
            retire.validate_host_file(str(self.store), str(self.store / "b.bin"), decision)
        missing = self.store / "missing.bin"
        with self.assertRaises(retire.RetireError):
            retire.validate_host_file(str(self.store), str(missing), decision)

    def test_dry_run_is_inert(self):
        host = self.store / "a.bin"
        host.write_bytes(b"host-state")
        journal = self.write_journal([record("committed", KEY_UNKNOWN, "a.bin")])
        before = journal.read_bytes()
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = retire.main(["--store", str(self.store), "--key", KEY_UNKNOWN])
        self.assertEqual(code, 0, err.getvalue())
        self.assertTrue(host.exists())
        self.assertEqual(journal.read_bytes(), before)
        self.assertIn("dry-run only", out.getvalue())
        self.assertIn("rkva-snapshots-v2", out.getvalue())
        self.assertIn(KEY_UNKNOWN, out.getvalue())

    def test_dry_run_refuses_pending(self):
        self.write_journal([record("pending", KEY_UNKNOWN, "a.bin")])
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = retire.main(["--store", str(self.store), "--key", KEY_UNKNOWN])
        self.assertEqual(code, 2)
        self.assertIn("pending", err.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
