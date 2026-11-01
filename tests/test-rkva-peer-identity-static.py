#!/usr/bin/env python3
"""Static/source checks for the default-off peer identity store (blocker 7).

These checks do not compile Windows code and do not qualify AppContainer,
console runtime or power-loss behavior. They only prove the source keeps the
Windows-only path, the ACL intent, the no-logging rule and the pin-before-auth
order that the CPU tests cannot exercise.
"""
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'tools/rkva-peer-identity'
WDP = ROOT / 'tools/xbox/scripts/rkva-peer-identity-wdp.py'
DOC = ROOT / 'docs/rkva-peer-identity-store.md'
INVENTORY = ROOT / 'docs/rkva-peer-identity-uwp-inventory.md'
MATRIX = ROOT / 'docs/rkva-snapshot-console-qualification-matrix.md'


def read(path):
    return path.read_text(encoding='utf-8')


class LayoutTests(unittest.TestCase):
    def test_cmake_option_is_default_off(self):
        text = read(ROOT / 'CMakeLists.txt')
        self.assertRegex(
            text,
            r'option\(RKVA_PEER_IDENTITY_STORE\s+"[^"]*"\s+OFF\)')

    def test_cmake_targets_are_guarded(self):
        text = read(ROOT / 'tests/CMakeLists.txt')
        self.assertIn('if (RKVA_PEER_IDENTITY_STORE)', text)
        self.assertIn('add_library(rkva-peer-identity STATIC', text)
        self.assertIn('add_library(rkva-peer-identity-testing STATIC', text)
        self.assertIn('add_executable(test-rkva-peer-identity', text)
        self.assertIn(
            'RKVA_PEER_IDENTITY_TESTING', text,
            'the test backend macro must reach the test source')
        # No production target includes the store outside the option block.
        guarded = text.split('if (RKVA_PEER_IDENTITY_STORE)', 1)
        self.assertGreaterEqual(len(guarded), 2)
        for path in ROOT.rglob('CMakeLists.txt'):
            if path == ROOT / 'tests/CMakeLists.txt':
                continue
            self.assertNotIn('rkva-peer-identity-core.cpp', read(path), str(path))

    def test_required_files_exist(self):
        for name in ('rkva-peer-identity.h', 'rkva-peer-identity-core.cpp',
                     'rkva-peer-identity-win32.cpp', 'rkva-peer-identity-test-store.cpp'):
            self.assertTrue((STORE / name).is_file(), name)
        self.assertTrue(WDP.is_file())
        self.assertTrue(DOC.is_file())


class WindowsStaticTests(unittest.TestCase):
    def setUp(self):
        self.source = read(STORE / 'rkva-peer-identity-win32.cpp')

    def test_appcontainer_localstate_entry(self):
        self.assertIn('RKVA_PEER_IDENTITY_UWP', self.source)
        self.assertIn('ApplicationData::Current().LocalFolder()', self.source)
        self.assertIn('rkva-peer-identity-v3', self.source)

    def test_atomic_exclusive_file_sequence(self):
        for token in ('CreateFile2', 'FILE_FLAG_OPEN_REPARSE_POINT', 'LockFileEx',
                      'FlushFileBuffers', 'MoveFileExW', 'identity.pending',
                      'identity.new', 'identity.json', 'identity.lock'):
            self.assertIn(token, self.source, token)

    def test_protected_appcontainer_acl(self):
        for token in ('PROTECTED_DACL_SECURITY_INFORMATION', 'TokenAppContainerSid',
                      'TokenIsAppContainer', 'SetSecurityInfo', 'FILE_ALL_ACCESS'):
            self.assertIn(token, self.source, token)
        for forbidden in ('WinWorldSid', 'WinBuiltinUsersSid', 'ACCESS_ALLOWED_ACE'):
            self.assertNotIn(forbidden, self.source, forbidden)

    def test_integrity_key_is_protected(self):
        self.assertIn('CryptProtectData', self.source)
        self.assertIn('CryptUnprotectData', self.source)
        self.assertIn('BCryptGenRandom', self.source)
        self.assertIn('CRYPTPROTECT_UI_FORBIDDEN', self.source)
        self.assertNotIn('auth_secret', self.source)

    def test_no_logging_in_store(self):
        for path in STORE.glob('*'):
            if path.suffix not in ('.h', '.cpp'):
                continue
            text = read(path)
            for token in ('printf(', 'fprintf(', 'std::cout', 'std::cerr',
                          'OutputDebugString', 'spdlog', 'LLAMA_LOG'):
                self.assertNotIn(token, text, f'{path.name}: {token}')


class CoreStaticTests(unittest.TestCase):
    def test_schema_and_pin_fields(self):
        header = read(STORE / 'rkva-peer-identity.h')
        text = read(STORE / 'rkva-peer-identity-core.cpp')
        self.assertIn('rkva-peer-identity-v1', header)
        for token in ('schema_version', 'pfn', 'package_family', 'publisher',
                      'cert_sha256', 'cert_thumbprint', 'peer_host', 'first_seen',
                      'pin_state', 'tofu', 'rotation', 'revocation', 'hmac-sha256',
                      'key_protected', 'integrity'):
            self.assertIn(token, text, token)

    def test_verify_never_writes_or_enrolls(self):
        text = read(STORE / 'rkva-peer-identity-core.cpp')
        start = text.index('status verify(store & backend')
        end = text.index('status enroll_tofu(store & backend')
        body = text[start:end]
        for forbidden in ('save_record', 'backend.write', 'new_key', 'seal_key',
                          'enroll_tofu', 'provision('):
            self.assertNotIn(forbidden, body, forbidden)

    def test_gate_is_compile_and_run_time_off(self):
        text = read(STORE / 'rkva-peer-identity-core.cpp')
        self.assertIn('#ifdef RKVA_PEER_IDENTITY_STORE', text)
        self.assertIn('std::getenv("RKVA_PEER_IDENTITY_STORE")', text)
        self.assertIn('std::strcmp(value, "1") == 0', text)

    def test_tofu_and_rotation_are_explicit(self):
        text = read(STORE / 'rkva-peer-identity-core.cpp')
        for token in ('explicit_operator_confirmation', 'out-of-band certificate digest',
                      'already_pinned', 'previously retired certificate',
                      'peer identity is revoked and cannot rotate'):
            self.assertIn(token, text, token)

    def test_cpu_test_covers_fail_closed_cases(self):
        text = read(ROOT / 'tests/test-rkva-peer-identity.cpp')
        for token in ('status::corrupt', 'status::mismatch', 'status::revoked',
                      'status::already_pinned', 'status::unauthorized',
                      'explicit-operator-first-use', 'after_staging', 'after_rename',
                      'pending_name', 'record_name', 'symlink', 'LOCK_EX', '0600'):
            self.assertIn(token, text, token)
        header = read(STORE / 'rkva-peer-identity.h')
        for token in ('"identity.lock"', '"identity.json"', '"identity.pending"'):
            self.assertIn(token, header, token)


class WdpStaticTests(unittest.TestCase):
    def setUp(self):
        self.source = read(WDP)

    def test_pin_precedes_request_and_credentials(self):
        body = self.source[self.source.index('    def get_json'):]
        peer_cert = body.index('getpeercert')
        compare = body.index('hmac.compare_digest')
        request = body.index('connection.request(')
        self.assertLess(peer_cert, compare)
        self.assertLess(compare, request)

    def test_no_unpinned_deploy_or_curl_path(self):
        for token in ('subprocess', 'os.system', 'Popen(', '--insecure',
                      'verify_mode = ssl.CERT_REQUIRED'):
            self.assertNotIn(token, self.source, token)
        self.assertIn('verify_mode = ssl.CERT_NONE', self.source)
        self.assertIn('out-of-band', self.source)

    def test_no_credentials_in_evidence(self):
        body = self.source[self.source.index('def build_evidence'):]
        body = body[:body.index('def write_evidence')]
        for token in ('XBOX_PASS', 'self._password', 'Authorization', 'Basic '):
            self.assertNotIn(token, body, token)

    def test_explicit_pfn_and_namespace(self):
        self.assertIn("required=True", self.source)
        self.assertIn("'exact PackageFullName", self.source)
        self.assertIn("identity.json", self.source)
        self.assertIn("identity.lock", self.source)


class DocumentationTests(unittest.TestCase):
    def test_remaining_blockers_are_documented(self):
        text = read(DOC)
        for token in ('MSVC', 'UWP', 'AppContainer', 'restart', 'power',
                      'BLOCKED', 'not qualified', 'TOFU'):
            self.assertIn(token, text, token)

    def test_inventory_and_matrix_do_not_claim_runtime(self):
        inventory = read(INVENTORY)
        self.assertIn('default-off', inventory)
        matrix = read(MATRIX)
        self.assertIn('I4', matrix)
        self.assertIn('I5', matrix)
        self.assertIn('I6', matrix)
        for text in (inventory, matrix):
            self.assertNotIn('AppContainer runtime is qualified', text)


if __name__ == '__main__':
    unittest.main()
