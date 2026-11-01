#!/usr/bin/env python3
"""CPU-only checks for the pinned WDP peer-identity evidence helper.

Run: python3 tests/test-rkva-peer-identity-wdp.py
No console, network, credential, GPU or service is touched.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / 'tools/xbox/scripts/rkva-peer-identity-wdp.py'
spec = importlib.util.spec_from_file_location('peer_wdp', PATH)
wdp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wdp)

PFN = 'GianlucaMazza.xllama_1.6.0.436_x64__pj67f1fcj4n14'
FAMILY = 'GianlucaMazza.xllama_pj67f1fcj4n14'
PUBLISHER = 'CN=Gianluca Mazza, O=Gianluca Mazza, C=IT'
DER = b'leaf-certificate-der'
CERT = hashlib.sha256(DER).hexdigest()
THUMB = hashlib.sha1(DER).hexdigest()
SECRET = 'do-not-print-this-password'


def package():
    return {
        'PackageFullName': PFN,
        'PackageFamilyName': FAMILY,
        'Publisher': PUBLISHER,
        'Version': '1.6.0.436',
        'PackageRelativeId': 'GianlucaMazza.xllama',
    }


def store_record():
    return {
        'schema': 'rkva-peer-identity-v1',
        'version': 1,
        'pfn': PFN,
        'package_family': FAMILY,
        'publisher': PUBLISHER,
        'publisher_id': 'pj67f1fcj4n14',
        'cert_sha256': CERT,
        'cert_thumbprint': THUMB,
        'peer_host': '192.168.1.200',
        'peer_port': 11443,
        'first_seen': 1700000000,
        'pin_state': 'pinned',
        'tofu': {},
        'rotation': {},
        'revocation': {},
        'integrity': {},
    }


class FakeSocket:
    def __init__(self, der):
        self.der = der

    def getpeercert(self, binary_form=False):
        return self.der if binary_form else {}


class FakeResponse:
    def __init__(self, status, document):
        self.status = status
        self._body = json.dumps(document).encode('utf-8')

    def read(self, count=-1):
        return self._body


class FakeConnection:
    opened = []
    der = DER
    status = 200
    document = None

    def __init__(self, host, port, timeout=None, context=None):
        self.host = host
        self.port = port
        self.context = context
        self.sock = None
        self.requested = None
        self.closed = False
        FakeConnection.opened.append(self)

    def connect(self):
        self.sock = FakeSocket(self.der)

    def request(self, method, path, headers=None):
        self.requested = (method, path, dict(headers or {}))

    def getresponse(self):
        return FakeResponse(self.status, self.document or {'InstalledPackages': [package()]})

    def close(self):
        self.closed = True


class PortalTests(unittest.TestCase):
    def tearDown(self):
        FakeConnection.opened.clear()
        FakeConnection.der = DER
        FakeConnection.status = 200
        FakeConnection.document = None

    def test_pin_checked_before_request_and_credentials(self):
        portal = wdp.PinnedPortal('192.168.1.200', CERT, 'user', SECRET,
                                  connection_factory=FakeConnection)
        document = portal.get_json('/api/app/packagemanager/packages')
        self.assertIn('InstalledPackages', document)
        connection = FakeConnection.opened[-1]
        self.assertIsNotNone(connection.requested)
        self.assertEqual(connection.requested[0], 'GET')
        self.assertIn('Authorization', connection.requested[2])
        self.assertEqual(portal.last_certificate_sha256, CERT)
        self.assertEqual(portal.last_thumbprint, THUMB)
        self.assertTrue(connection.closed)

    def test_pin_mismatch_sends_nothing(self):
        portal = wdp.PinnedPortal('192.168.1.200', '0' * 64, 'user', SECRET,
                                  connection_factory=FakeConnection)
        with self.assertRaises(wdp.PinError):
            portal.get_json('/api/app/packagemanager/packages')
        connection = FakeConnection.opened[-1]
        self.assertIsNone(connection.requested)
        self.assertEqual(portal.last_certificate_sha256, CERT)
        self.assertTrue(connection.closed)

    def test_missing_leaf_certificate_refused(self):
        FakeConnection.der = b''
        portal = wdp.PinnedPortal('192.168.1.200', CERT, 'user', SECRET,
                                  connection_factory=FakeConnection)
        with self.assertRaises(wdp.PinError):
            portal.get_json('/api/app/packagemanager/packages')
        self.assertIsNone(FakeConnection.opened[-1].requested)

    def test_http_error_refused(self):
        FakeConnection.status = 401
        portal = wdp.PinnedPortal('192.168.1.200', CERT, 'user', SECRET,
                                  connection_factory=FakeConnection)
        with self.assertRaises(wdp.RefuseError):
            portal.get_json('/api/app/packagemanager/packages')


class PureFunctionTests(unittest.TestCase):
    def test_parse_pfn(self):
        parsed = wdp.parse_pfn(PFN)
        self.assertEqual(parsed['package_family'], FAMILY)
        self.assertEqual(parsed['publisher_id'], 'pj67f1fcj4n14')
        for bad in ('', 'GianlucaMazza.xllama', PFN + '_x', PFN.replace('_x64__', '_x64_'),
                    PFN.replace('pj67f1fcj4n14', 'SHORT')):
            with self.assertRaises(wdp.RefuseError):
                wdp.parse_pfn(bad)

    def test_select_package_exact_only(self):
        listing = {'InstalledPackages': [package()]}
        selected, others = wdp.select_package(listing, PFN)
        self.assertEqual(selected['PackageFullName'], PFN)
        self.assertEqual(others, [])
        with self.assertRaises(wdp.RefuseError):
            wdp.select_package({'InstalledPackages': []}, PFN)
        with self.assertRaises(wdp.RefuseError):
            wdp.select_package({'InstalledPackages': [package(), package()]}, PFN)

    def test_compare_store_public_fields(self):
        checks, errors = wdp.compare_store(store_record(), package(), CERT, '192.168.1.200')
        self.assertEqual(errors, [])
        self.assertTrue(checks['pfn'])
        self.assertTrue(checks['certificate_sha256'])
        self.assertFalse(checks['integrity_verified'])

        cases = {
            'pfn': ('pfn', 'GianlucaMazza.xllama_1.6.0.437_x64__pj67f1fcj4n14'),
            'package_family': ('package_family', 'GianlucaMazza.xllama_other'),
            'publisher': ('publisher', 'CN=Someone Else'),
            'certificate_sha256': ('cert_sha256', 'f' * 64),
            'peer_host': ('peer_host', '192.168.1.201'),
            'peer_port': ('peer_port', 11444),
            'pin_state': ('pin_state', 'revoked'),
        }
        for name, (field, value) in cases.items():
            record = store_record()
            record[field] = value
            checks, errors = wdp.compare_store(record, package(), CERT, '192.168.1.200')
            self.assertFalse(checks[name], name)
            self.assertTrue(errors, name)

    def test_read_store_record_rejects_bad_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'identity.json'
            path.write_text('not-json')
            with self.assertRaises(wdp.RefuseError):
                wdp.read_store_record(path)
            path.write_text(json.dumps({k: v for k, v in store_record().items()
                                        if k != 'integrity'}))
            with self.assertRaises(wdp.RefuseError):
                wdp.read_store_record(path)
            path.write_text(json.dumps(store_record()))
            self.assertEqual(wdp.read_store_record(path)['pfn'], PFN)

    def test_canonical_pin(self):
        self.assertEqual(wdp.canonical_pin(CERT.upper()), CERT)
        for bad in ('', 'a' * 63, 'g' * 64, None):
            with self.assertRaises(wdp.RefuseError):
                wdp.canonical_pin(bad)


class EvidenceTests(unittest.TestCase):
    def credentials(self, directory):
        path = Path(directory) / 'xbox-env'
        path.write_text(f'XBOX_USER=user\nXBOX_PASS={SECRET}\n')
        return path

    def arguments(self, directory, **overrides):
        values = {
            'xbox_host': '192.168.1.200',
            'pfn': PFN,
            'expected_cert_sha256': CERT,
            'expected_publisher': None,
            'store_json': None,
            'xbox_env': str(self.credentials(directory)),
            'port': 11443,
            'timeout': 5,
            'skip_inventory': False,
            'out': str(Path(directory) / 'evidence.json'),
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_run_records_read_only_evidence_without_secrets(self):
        class StubPortal:
            last_certificate_sha256 = CERT
            last_thumbprint = THUMB

            def __init__(self, *args, **kwargs):
                pass

            def get_json(self, path, allow_missing=False):
                if 'packagemanager' in path:
                    return {'InstalledPackages': [package()]}
                return {'Items': [{'Name': 'identity.json'}, {'Name': 'identity.lock'}]}

        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            with mock.patch.object(wdp, 'PinnedPortal', StubPortal):
                status = wdp.run(args)
            self.assertEqual(status, 0)
            text = (Path(directory) / 'evidence.json').read_text()
            document = json.loads(text)
            self.assertEqual(document['result'], 'pinned')
            self.assertEqual(document['package']['PackageFullName'], PFN)
            self.assertEqual(document['namespace'],
                             {'present': True, 'entries': ['identity.json', 'identity.lock']})
            self.assertNotIn(SECRET, text)
            self.assertNotIn('Authorization', text)
            self.assertNotIn('Basic ', text)

    def test_run_refuses_pin_mismatch_and_writes_evidence(self):
        class MismatchPortal:
            last_certificate_sha256 = 'f' * 64
            last_thumbprint = THUMB

            def __init__(self, *args, **kwargs):
                pass

            def get_json(self, path, allow_missing=False):
                raise wdp.PinError('pinned digest mismatch', actual='f' * 64)

        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            with mock.patch.object(wdp, 'PinnedPortal', MismatchPortal):
                status = wdp.run(args)
            self.assertEqual(status, 2)
            document = json.loads((Path(directory) / 'evidence.json').read_text())
            self.assertEqual(document['result'], 'refused')
            self.assertFalse(document['checks']['certificate_pin'])
            self.assertIn('certificate pin was not established', document['errors'])
            self.assertNotIn(SECRET, json.dumps(document))

    def test_run_refuses_unknown_package(self):
        class EmptyPortal:
            last_certificate_sha256 = CERT
            last_thumbprint = THUMB

            def __init__(self, *args, **kwargs):
                pass

            def get_json(self, path, allow_missing=False):
                return {'InstalledPackages': [dict(package(), PackageFullName=PFN + '.x')]}

        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            with mock.patch.object(wdp, 'PinnedPortal', EmptyPortal):
                status = wdp.run(args)
            self.assertEqual(status, 2)
            document = json.loads((Path(directory) / 'evidence.json').read_text())
            self.assertEqual(document['result'], 'refused')
            self.assertIn('the exact pinned package is not installed', document['errors'])

    def test_expected_publisher_mismatch_refused(self):
        class StubPortal:
            last_certificate_sha256 = CERT
            last_thumbprint = THUMB

            def __init__(self, *args, **kwargs):
                pass

            def get_json(self, path, allow_missing=False):
                if 'packagemanager' in path:
                    return {'InstalledPackages': [package()]}
                return {'Items': [{'Name': 'identity.json'}, {'Name': 'identity.lock'}]}

        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory, expected_publisher='CN=Other')
            with mock.patch.object(wdp, 'PinnedPortal', StubPortal):
                status = wdp.run(args)
            self.assertEqual(status, 2)
            document = json.loads((Path(directory) / 'evidence.json').read_text())
            self.assertFalse(document['checks']['publisher_matches_expected'])


if __name__ == '__main__':
    unittest.main()
