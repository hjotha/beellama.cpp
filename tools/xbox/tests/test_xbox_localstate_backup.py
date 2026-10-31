import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error

import xbox_localstate_backup as backup


def item(name, kind, size=None):
    value = {'Name': name, 'Type': kind, 'DateCreated': 7}
    if kind == 32:
        value['FileSize'] = size
    return value


class FakePortal:
    def __init__(self):
        self.list_calls = 0
        self.read_calls = 0
        self.changed_content = False
        self.changed_listing = False
        self.short = False

    def listing(self, path):
        self.list_calls += 1
        if path == ():
            entries = [item('dir', 16), item('api.flag', 32, 0)]
        else:
            entries = [item('data.bin', 32, 3)]
        if self.changed_listing and self.list_calls > 2 and path == ():
            entries.append(item('new.flag', 32, 0))
        return backup.parse_listing({'Items': entries})

    def file(self, directory, name):
        self.read_calls += 1
        if name == 'api.flag':
            return io.BytesIO(b'')
        if self.short:
            return io.BytesIO(b'ab')
        if self.changed_content and self.read_calls == 4:
            return io.BytesIO(b'xyz')
        return io.BytesIO(b'abc')


class ScopedPortal:
    def __init__(self):
        self.root_reads = 0
        self.files_read = []
        self.change_flag = False
        self.missing_snapshots = False

    def listing(self, path):
        if path == ():
            self.root_reads += 1
            entries = [item('api.flag', 32, 0), item('xllama.log', 32, self.root_reads)]
            if not self.missing_snapshots:
                entries.append(item('rkva-snapshots-v2', 16))
            if self.change_flag and self.root_reads > 1:
                entries.append(item('new.flag', 32, 0))
        elif path == ('rkva-snapshots-v2',):
            entries = [item('key', 16)]
        else:
            entries = [item('manifest-v2.bin', 32, 3)]
        return backup.parse_listing({'Items': entries})

    def file(self, directory, name):
        self.files_read.append((directory, name))
        return io.BytesIO(b'' if name == 'api.flag' else b'abc')


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'backup'
        self.old_umask = os.umask(0o077)
        self.addCleanup(os.umask, self.old_umask)

    def state(self):
        return json.loads((self.path / 'state.json').read_text())

    def test_complete_recursive_backup_and_private_permissions(self):
        portal = FakePortal()
        state = backup.backup(portal, self.path, 'pfn')
        self.assertEqual(state['status'], 'COMPLETE')
        self.assertEqual(state['total_bytes'], 3)
        self.assertEqual(portal.read_calls, 4)
        self.assertEqual(portal.list_calls, 4)
        self.assertEqual((self.path / 'LocalState/dir/data.bin').read_bytes(), b'abc')
        row = next(row for row in state['files'] if row['path'] == 'dir/data.bin')
        self.assertEqual(row['sha256'], hashlib.sha256(b'abc').hexdigest())
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.path / 'LocalState/dir/data.bin').stat().st_mode & 0o777, 0o600)
        self.assertIn('No independent remote hash', state['limitations'][0])
        with self.assertRaises(FileExistsError):
            backup.backup(portal, self.path, 'pfn')

    def test_rejects_unsafe_or_ambiguous_entries(self):
        for entries in ([item('../escape', 32, 0)], [item('a', 32, -1)],
                        [item('Foo', 32, 0), item('foo', 32, 0)],
                        [item('a', 99)], [item('a', 32, True)]):
            with self.subTest(entries=entries), self.assertRaises(backup.BackupError):
                backup.parse_listing({'Items': entries})
        with self.assertRaises(backup.BackupError):
            backup.parse_listing({'Success': False})

    def test_detects_same_size_changed_content(self):
        portal = FakePortal()
        portal.changed_content = True
        with self.assertRaisesRegex(backup.BackupError, 'repeat content'):
            backup.backup(portal, self.path, 'pfn')
        self.assertEqual(self.state()['status'], 'PARTIAL')

    def test_detects_changed_tree_and_short_download(self):
        portal = FakePortal()
        portal.changed_listing = True
        with self.assertRaisesRegex(backup.BackupError, 'listing changed'):
            backup.backup(portal, self.path, 'pfn')
        self.assertEqual(self.state()['status'], 'PARTIAL')
        self.path = Path(self.temp.name) / 'short'
        portal = FakePortal()
        portal.short = True
        with self.assertRaisesRegex(backup.BackupError, 'size differs'):
            backup.backup(portal, self.path, 'pfn')
        self.assertEqual(self.state()['status'], 'PARTIAL')

    def test_http_errors_are_explicit_and_credentials_are_not_in_error(self):
        portal = backup.Portal('192.0.2.1', 'test-user', 'secret-value', 'safe_pfn')
        class ErrorOpener:
            def open(self, request, timeout):
                raise urllib.error.HTTPError(request.full_url, 403, 'Forbidden', None, None)
        portal.opener = ErrorOpener()
        with self.assertRaises(backup.BackupError) as caught:
            portal.listing(())
        self.assertEqual(str(caught.exception), 'Device Portal HTTP 403')
        self.assertNotIn('secret-value', str(caught.exception))
        with self.assertRaises(backup.BackupError):
            backup.Portal('example.com/path', 'u', 'p', 'safe_pfn')

    def test_snapshot_scope_ignores_changing_log_but_reads_selected_twice(self):
        portal = ScopedPortal()
        state = backup.backup(portal, self.path, 'pfn', 'snapshots-v2-flags')
        self.assertEqual(state['scope'], 'snapshots-v2-flags')
        self.assertEqual(state['status'], 'COMPLETE')
        self.assertEqual({row['path'] for row in state['files']},
                         {'api.flag', 'rkva-snapshots-v2/key/manifest-v2.bin'})
        self.assertEqual(len(portal.files_read), 4)
        self.assertFalse(any(name == 'xllama.log' for _, name in portal.files_read))
        self.assertIn('not a full LocalState backup', state['limitations'][-1])

    def test_snapshot_scope_fails_on_changed_flag_or_missing_snapshot_dir(self):
        portal = ScopedPortal()
        portal.change_flag = True
        with self.assertRaisesRegex(backup.BackupError, 'listing changed'):
            backup.backup(portal, self.path, 'pfn', 'snapshots-v2-flags')
        self.assertEqual(self.state()['status'], 'PARTIAL')
        self.path = Path(self.temp.name) / 'missing'
        portal = ScopedPortal()
        portal.missing_snapshots = True
        with self.assertRaisesRegex(backup.BackupError, 'directory is missing'):
            backup.backup(portal, self.path, 'pfn', 'snapshots-v2-flags')
        self.assertEqual(self.state()['status'], 'PARTIAL')


if __name__ == '__main__':
    unittest.main()
