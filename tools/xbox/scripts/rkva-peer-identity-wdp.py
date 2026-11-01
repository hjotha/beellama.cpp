#!/usr/bin/env python3
"""Pinned, read-only WDP evidence fetch for the RKVA peer identity (blocker 7).

The leaf certificate digest is supplied out of band (console UI or physical
label) and compared BEFORE any HTTP request or Basic credential is sent. This
script never deploys, installs, deletes, restarts or modifies a console. It
records the installed package identity, the certificate digests, the optional
LocalState namespace inventory and an optional comparison with an exported
identity.json record.

It deliberately does not call deploy.sh: deploy.sh uses `curl -k`, disables TLS
verification and authenticates before any certificate check. This helper uses
the same WDP endpoints and the same credential file format, but pins the leaf
first.

Run: python3 tools/xbox/scripts/rkva-peer-identity-wdp.py --help
No credentials or certificate bytes are printed or written to the evidence.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import http.client
import json
import os
from pathlib import Path
import re
import ssl
import sys
import time
from urllib.parse import urlencode

SCHEMA = 'rkva-peer-identity-wdp-v1'
NAMESPACE = 'rkva-peer-identity-v3'
DEFAULT_PORT = 11443
MAX_RESPONSE = 8 << 20
HEX64 = re.compile(r'^[0-9a-f]{64}$')
PFN = re.compile(r'^[A-Za-z0-9.-]+_[0-9]+(?:\.[0-9]+){0,3}_[A-Za-z0-9]+__[a-z0-9]{13}$')
STORE_KEYS = {
    'schema', 'version', 'pfn', 'package_family', 'publisher', 'publisher_id',
    'cert_sha256', 'cert_thumbprint', 'peer_host', 'peer_port', 'first_seen',
    'pin_state', 'tofu', 'rotation', 'revocation', 'integrity',
}


class RefuseError(RuntimeError):
    pass


class PinError(RefuseError):
    def __init__(self, message, actual=None):
        super().__init__(message)
        self.actual = actual


def now_iso():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def parse_pfn(pfn):
    value = (pfn or '').strip()
    if not PFN.fullmatch(value):
        raise RefuseError('package full name is not the expected five-part PFN form')
    name, version, arch, empty, publisher_id = value.split('_')
    if empty or not publisher_id:
        raise RefuseError('package full name has an empty field')
    return {
        'pfn': value,
        'name': name,
        'version': version,
        'arch': arch,
        'publisher_id': publisher_id,
        'package_family': f'{name}_{publisher_id}',
    }


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def thumbprint_sha1(der):
    return hashlib.sha1(der).hexdigest()


def canonical_pin(value):
    pin = (value or '').strip().lower()
    if not HEX64.fullmatch(pin):
        raise RefuseError('--expected-cert-sha256 must be 64 hexadecimal characters')
    return pin


def load_credentials(path=None):
    """Load the existing XBOX_USER/XBOX_PASS file format without logging values."""
    if path is None:
        path = Path.home() / '.config/xllama/xbox-env'
    path = Path(path).expanduser()
    if not path.is_file():
        raise RefuseError(f'Xbox credential file not found: {path}')
    values = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].strip()
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    if not values.get('XBOX_USER') or not values.get('XBOX_PASS'):
        raise RefuseError('Xbox credential file is missing XBOX_USER or XBOX_PASS')
    return values['XBOX_USER'], values['XBOX_PASS']


def installed_packages(document):
    if not isinstance(document, dict) or not isinstance(document.get('InstalledPackages'), list):
        raise RefuseError('WDP package listing has an unexpected shape')
    return document['InstalledPackages']


def select_package(document, pfn):
    packages = installed_packages(document)
    matches = [item for item in packages
               if isinstance(item, dict) and item.get('PackageFullName') == pfn]
    if not matches:
        raise RefuseError('the exact pinned package is not installed')
    if len(matches) != 1:
        raise RefuseError('the exact pinned package is listed more than once')
    others = sorted(item.get('PackageFullName', '') for item in packages
                    if isinstance(item, dict) and
                    item.get('PackageFullName') != pfn and
                    'xllama' in item.get('PackageRelativeId', ''))
    package = matches[0]
    family = package.get('PackageFamilyName')
    if family and family != parse_pfn(pfn)['package_family']:
        raise RefuseError('installed package family does not match the pinned PFN')
    return package, others


def read_store_record(path):
    record_path = Path(path).expanduser()
    try:
        document = json.loads(record_path.read_text())
    except (OSError, ValueError) as error:
        raise RefuseError(f'identity record could not be read: {record_path}') from error
    if not isinstance(document, dict) or not STORE_KEYS.issubset(document):
        raise RefuseError('identity record is missing required fields')
    if document.get('schema') != 'rkva-peer-identity-v1' or document.get('version') != 1:
        raise RefuseError('identity record has an unexpected schema or version')
    return document


def compare_store(record, package, certificate_sha256, host):
    """Compare public fields only; the MAC/DPAPI key stay opaque to WDP."""
    checks = {}
    errors = []
    record_pfn = record.get('pfn')
    checks['pfn'] = record_pfn == package.get('PackageFullName')
    if not checks['pfn']:
        errors.append('local identity record PFN differs from the installed package')
    checks['package_family'] = record.get('package_family') == package.get('PackageFamilyName')
    if not checks['package_family']:
        errors.append('local identity record package family differs from the installed package')
    checks['publisher'] = record.get('publisher') == package.get('Publisher')
    if not checks['publisher']:
        errors.append('local identity record publisher differs from the installed package')
    checks['certificate_sha256'] = record.get('cert_sha256') == certificate_sha256
    if not checks['certificate_sha256']:
        errors.append('local identity record certificate sha256 differs from the pinned leaf')
    checks['peer_host'] = str(record.get('peer_host', '')).lower() == host.lower()
    if not checks['peer_host']:
        errors.append('local identity record peer host differs from the pinned console')
    checks['peer_port'] = record.get('peer_port') == DEFAULT_PORT
    if not checks['peer_port']:
        errors.append('local identity record peer port differs from the pinned WDP port')
    checks['pin_state'] = record.get('pin_state') == 'pinned'
    if not checks['pin_state']:
        errors.append('local identity record is not in the pinned state')
    checks['integrity_verified'] = False
    return checks, errors


class PinnedPortal:
    """HTTPS client that refuses before sending credentials on a pin mismatch."""

    def __init__(self, host, expected_sha256, user, password, port=DEFAULT_PORT,
                 timeout=30, connection_factory=None):
        self.host = host
        self.port = int(port)
        self.expected_sha256 = canonical_pin(expected_sha256)
        self._user = user
        self._password = password
        self.timeout = timeout
        self.connection_factory = connection_factory or http.client.HTTPSConnection
        self.last_certificate_sha256 = None
        self.last_thumbprint = None

    def _connect(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        # The certificate is authenticated by an out-of-band digest, not by the
        # host name or the OS trust store. This is a pin, not TOFU: the digest
        # is compared before any request is sent.
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return self.connection_factory(
            self.host, self.port, timeout=self.timeout, context=context)

    def _authorized_headers(self):
        token = base64.b64encode(
            f'{self._user}:{self._password}'.encode('utf-8')).decode('ascii')
        return {
            'Authorization': f'Basic {token}',
            'Accept': 'application/json',
            'Connection': 'close',
        }

    def get_json(self, path, allow_missing=False):
        connection = self._connect()
        try:
            connection.connect()
            der = connection.sock.getpeercert(binary_form=True)
            if not der:
                raise PinError('the console presented no leaf certificate')
            actual = sha256_hex(der)
            self.last_certificate_sha256 = actual
            self.last_thumbprint = thumbprint_sha1(der)
            if not hmac.compare_digest(actual, self.expected_sha256):
                raise PinError('WDP leaf certificate does not match the out-of-band pin',
                               actual=actual)
            connection.request('GET', path, headers=self._authorized_headers())
            response = connection.getresponse()
            if allow_missing and response.status in (400, 404):
                return None
            if response.status != 200:
                raise RefuseError(f'WDP request failed with HTTP {response.status}')
            body = response.read(MAX_RESPONSE + 1)
            if len(body) > MAX_RESPONSE:
                raise RefuseError('WDP response exceeds the size limit')
            try:
                return json.loads(body)
            except ValueError as error:
                raise RefuseError('WDP response is not JSON') from error
        finally:
            connection.close()


def inventory(portal, pfn):
    query = urlencode({
        'knownfolderid': 'LocalAppData',
        'packagefullname': pfn,
        'path': '\\LocalState\\' + NAMESPACE,
    })
    document = portal.get_json('/api/filesystem/apps/files?' + query, allow_missing=True)
    if document is None:
        return False, []
    items = document.get('Items') if isinstance(document, dict) else document
    if not isinstance(items, list):
        raise RefuseError('LocalState inventory has an unexpected shape')
    names = sorted(item['Name'] for item in items
                   if isinstance(item, dict) and isinstance(item.get('Name'), str))
    return True, names


def build_evidence(host, port, pin, pfn, package, certificate_sha256,
                   certificate_thumbprint, errors, checks, store=None, namespace=None,
                   others=None, inventory_error=None):
    return {
        'schema': SCHEMA,
        'timestamp': now_iso(),
        'host': host,
        'port': port,
        'pfn_expected': pfn,
        'pin_source': 'out-of-band-operator',
        'certificate_sha256': certificate_sha256,
        'certificate_thumbprint_sha1': certificate_thumbprint,
        'expected_certificate_sha256': pin,
        'package': None if package is None else {
            'PackageFullName': package.get('PackageFullName'),
            'PackageFamilyName': package.get('PackageFamilyName'),
            'Publisher': package.get('Publisher'),
            'Version': package.get('Version'),
        },
        'other_xllama_packages': others or [],
        'checks': checks,
        'store_comparison': store,
        'namespace': namespace,
        'inventory_error': inventory_error,
        'errors': errors,
        'result': 'pinned' if not errors else 'refused',
    }


def write_evidence(path, document):
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as output:
        json.dump(document, output, indent=2, sort_keys=True)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, destination)


def run(args):
    pin = canonical_pin(args.expected_cert_sha256)
    parsed = parse_pfn(args.pfn)
    user, password = load_credentials(args.xbox_env)
    portal = PinnedPortal(args.xbox_host, pin, user, password, port=args.port,
                          timeout=args.timeout)
    errors = []
    checks = {}
    package = None
    others = []
    store_comparison = None
    namespace = None
    inventory_error = None
    try:
        listing = portal.get_json('/api/app/packagemanager/packages')
        package, others = select_package(listing, args.pfn)
        checks['certificate_pin'] = hmac.compare_digest(
            portal.last_certificate_sha256 or '', pin)
        checks['pfn_exact'] = package.get('PackageFullName') == args.pfn
        checks['package_family'] = (
            package.get('PackageFamilyName') == parsed['package_family'])
        checks['publisher_id'] = (package.get('PackageFamilyName', '').endswith(
            '_' + parsed['publisher_id']))
        if args.expected_publisher:
            checks['publisher_matches_expected'] = (
                package.get('Publisher') == args.expected_publisher)
            if not checks['publisher_matches_expected']:
                errors.append('installed publisher differs from the expected publisher')
        if args.store_json:
            record = read_store_record(args.store_json)
            store_comparison, store_errors = compare_store(
                record, package, portal.last_certificate_sha256, args.xbox_host)
            errors.extend(store_errors)
        if not args.skip_inventory:
            try:
                present, names = inventory(portal, args.pfn)
                namespace = {'present': present, 'entries': names}
                if present and names != ['identity.json', 'identity.lock']:
                    errors.append('identity namespace has unexpected or missing entries')
            except RefuseError as error:
                inventory_error = str(error)
    except PinError as error:
        errors.append(str(error))
        checks['certificate_pin'] = False
    except RefuseError as error:
        errors.append(str(error))

    checks.setdefault('certificate_pin', bool(
        portal.last_certificate_sha256 and
        hmac.compare_digest(portal.last_certificate_sha256, pin)))
    if not checks['certificate_pin']:
        errors.append('certificate pin was not established')
    evidence = build_evidence(
        args.xbox_host, args.port, pin, args.pfn, package,
        portal.last_certificate_sha256, portal.last_thumbprint,
        sorted(set(errors)), checks, store_comparison, namespace, others,
        inventory_error)
    write_evidence(args.out, evidence)
    return 0 if evidence['result'] == 'pinned' else 2


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xbox-host', required=True,
                        help='console address; the pin is checked before auth')
    parser.add_argument('--pfn', required=True,
                        help='exact PackageFullName; no implicit highest-version selection')
    parser.add_argument('--expected-cert-sha256', required=True,
                        help='out-of-band leaf certificate SHA-256 (64 hex)')
    parser.add_argument('--expected-publisher', default=None,
                        help='optional exact WDP Publisher string')
    parser.add_argument('--store-json', default=None,
                        help='optional exported identity.json for public-field comparison')
    parser.add_argument('--xbox-env', default=str(Path.home() / '.config/xllama/xbox-env'),
                        help='credential file with XBOX_USER and XBOX_PASS')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--skip-inventory', action='store_true',
                        help='do not query the LocalState namespace')
    parser.add_argument('--out', required=True, help='evidence JSON path')
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except RefuseError as error:
        print(f'refused: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
