#!/usr/bin/env python3
"""Back up Xbox LocalState via read-only Device Portal GETs.

Requires XBOX_IP, XBOX_USER, XBOX_PASS and an explicit PFN and new output path.
The app must be quiescent: relisting and a second content read detect changes,
but cannot establish an atomic snapshot or provide an independent remote hash.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request


class BackupError(Exception):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Portal:
    def __init__(self, host, user, password, pfn):
        if not re.fullmatch(r'[A-Za-z0-9.-]+', host):
            raise BackupError('invalid XBOX_IP')
        if not re.fullmatch(r'[A-Za-z0-9._-]+', pfn):
            raise BackupError('invalid package full name')
        self.base = 'https://' + host + ':11443/api/filesystem/apps/'
        self.auth = 'Basic ' + base64.b64encode((user + ':' + password).encode()).decode()
        self.pfn = pfn
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect(),
            urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))

    def get(self, endpoint, directory, filename=None):
        params = {'knownfolderid': 'LocalAppData', 'packagefullname': self.pfn,
                  'path': '\\LocalState' + ('\\' + '\\'.join(directory) if directory else '')}
        if filename is not None:
            params['filename'] = filename
        request = urllib.request.Request(self.base + endpoint + '?' + urllib.parse.urlencode(params),
                                         headers={'Authorization': self.auth}, method='GET')
        try:
            response = self.opener.open(request, timeout=120)
        except urllib.error.HTTPError as exc:
            raise BackupError('Device Portal HTTP ' + str(exc.code)) from None
        except urllib.error.URLError:
            raise BackupError('Device Portal connection failed') from None
        if response.status != 200:
            response.close()
            raise BackupError('Device Portal returned non-200 response')
        return response

    def listing(self, directory):
        with self.get('files', directory) as response:
            try:
                data = json.load(response)
            except (ValueError, UnicodeError):
                raise BackupError('invalid Device Portal listing') from None
        return parse_listing(data)

    def file(self, directory, name):
        return self.get('file', directory, name)


def parse_listing(data):
    if not isinstance(data, dict) or not isinstance(data.get('Items'), list):
        raise BackupError('listing has no Items array')
    result, seen = [], set()
    for item in data['Items']:
        if not isinstance(item, dict):
            raise BackupError('invalid listing item')
        name, kind = item.get('Name'), item.get('Type')
        if (not isinstance(name, str) or not name or name in ('.', '..') or
                name[-1] in '. ' or any(ord(c) < 32 or c in '\\/:*?"<>|' for c in name)):
            raise BackupError('unsafe filename in listing')
        key = name.casefold()
        if key in seen:
            raise BackupError('duplicate filename in listing')
        seen.add(key)
        if type(kind) is not int or kind not in (16, 32):
            raise BackupError('unknown listing item type')
        size = item.get('FileSize') if kind == 32 else None
        if kind == 32 and (type(size) is not int or size < 0):
            raise BackupError('invalid file size in listing')
        created = item.get('DateCreated')
        if created is not None and (type(created) is not int or created < 0):
            raise BackupError('invalid creation time in listing')
        result.append({'name': name, 'type': kind, 'size': size, 'created': created})
    return sorted(result, key=lambda item: item['name'])


def inventory(portal, scope='all'):
    pending, listings = [()], {}
    while pending:
        directory = pending.pop()
        items = portal.listing(directory)
        if scope == 'snapshots-v2-flags' and not directory:
            snapshots = [item for item in items if item['name'] == 'rkva-snapshots-v2']
            if len(snapshots) != 1 or snapshots[0]['type'] != 16:
                raise BackupError('rkva-snapshots-v2 directory is missing')
            items = [item for item in items if item in snapshots or
                     (item['type'] == 32 and item['name'].endswith('.flag'))]
        listings[directory] = items
        pending.extend(directory + (item['name'],) for item in items if item['type'] == 16)
    return listings


def read_file(portal, directory, item, dest=None):
    digest = hashlib.sha256()
    count = 0
    with portal.file(directory, item['name']) as response:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            if count > item['size']:
                raise BackupError('download exceeds listed size')
            digest.update(chunk)
            if dest is not None:
                dest.write(chunk)
    if count != item['size']:
        raise BackupError('download size differs from listing')
    return digest.hexdigest()


def write_state(root, state):
    temp = root / 'state.json.tmp'
    temp.write_text(json.dumps(state, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    os.replace(temp, root / 'state.json')


def backup(portal, output, pfn, scope='all'):
    if scope not in ('all', 'snapshots-v2-flags'):
        raise BackupError('invalid backup scope')
    output = Path(output)
    output.mkdir(mode=0o700)  # Exclusive: existing destinations are never reused.
    os.chmod(output, 0o700)
    state = {'status': 'PARTIAL', 'pfn': pfn, 'scope': scope, 'files': [], 'directories': [],
             'limitations': ['No independent remote hash; two reads are compared.',
                             'Relisting and repeat reads do not make a live tree atomic.',
                             'Device Portal TLS certificate is not verified.']}
    if scope == 'snapshots-v2-flags':
        state['limitations'].append('Only rkva-snapshots-v2 and root .flag files are included; this is not a full LocalState backup.')
    try:
        write_state(output, state)
        data = output / 'LocalState'
        data.mkdir(mode=0o700)
        before = inventory(portal, scope)
        for directory, items in sorted(before.items()):
            local = data.joinpath(*directory)
            local.mkdir(mode=0o700, parents=True, exist_ok=True)
            state['directories'].append('/'.join(directory))
            for item in items:
                if item['type'] != 32:
                    continue
                path = local / item['name']
                with path.open('xb') as dest:
                    first = read_file(portal, directory, item, dest)
                second = read_file(portal, directory, item)
                if first != second:
                    raise BackupError('repeat content read differs')
                state['files'].append({'path': '/'.join(directory + (item['name'],)),
                                       'size': item['size'], 'sha256': first})
        if inventory(portal, scope) != before:
            raise BackupError('LocalState listing changed during backup')
        state['status'] = 'COMPLETE'
        state['total_bytes'] = sum(item['size'] for item in state['files'])
        write_state(output, state)
        return state
    except Exception as exc:
        state['status'] = 'PARTIAL'
        state['error'] = str(exc) if isinstance(exc, BackupError) else type(exc).__name__
        write_state(output, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pfn', required=True, help='Exact installed package full name')
    parser.add_argument('--output', type=Path, required=True, help='New private directory')
    parser.add_argument('--scope', choices=('all', 'snapshots-v2-flags'), default='all',
                        help='Select full LocalState or only snapshots v2 and root flags')
    args = parser.parse_args()
    os.umask(0o077)
    try:
        host, user, password = (os.environ[name] for name in ('XBOX_IP', 'XBOX_USER', 'XBOX_PASS'))
        state = backup(Portal(host, user, password, args.pfn), args.output, args.pfn, args.scope)
        print('COMPLETE scope=' + state['scope'] + ':', len(state['files']), 'files,', state['total_bytes'], 'bytes')
        if args.scope != 'all':
            print('Not a full LocalState backup.')
        print('No independent remote hash or atomic snapshot; inspect state.json.')
    except KeyError:
        print('ERROR: XBOX_IP, XBOX_USER and XBOX_PASS are required', file=sys.stderr)
        return 1
    except Exception as exc:
        message = str(exc) if isinstance(exc, (BackupError, FileExistsError)) else type(exc).__name__
        print('ERROR:', message, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
