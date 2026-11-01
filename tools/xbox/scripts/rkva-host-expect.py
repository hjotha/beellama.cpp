#!/usr/bin/env python3
"""Emit a host expectation for the test-only console qualification probe.

Reads REAL host state only: the anchored SQLite managed store
(<store>/rkva-managed-v3/peer.db), an optional host journal
(host-journal.bin) cross-check, and an optional published object manifest
(<root>/<prefix-key>/manifest-v2.bin). It never contacts a console, a service
or a GPU and never writes into the store.

The output is consumed by
tools/xbox/scripts/xbox-snapshot-console-qual.py --expect. Fields that the host
state cannot prove (the console session id and the GPU fence pair) are emitted
as null with an exact reason and expectation_available=false. They are never
invented.

Canonical digests (geometry_sha256, descriptor_sha256, coverage_sha256,
state_digest) mirror rkva_qual_probe::* in
tools/xbox/uwp/rkva/rkva-qual-probe.h byte for byte.

Examples:
  rkva-host-expect.py --store /srv/rkva-managed --out host-expect.json
  rkva-host-expect.py --store /srv/rkva-managed --tx <64 hex> \
      --journal /srv/rkva-journal --owner <64 hex> \
      --object-root /srv/rkva-snapshots-v3 --out host-expect.json
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
import sqlite3
import struct
import sys

HEX64 = re.compile(r'[0-9a-f]{64}\Z')
MANIFEST_MAGIC = 0x53564B52  # "RKVS"
MANIFEST_VERSION = 1
MANIFEST_HEADER = 176
MANIFEST_MAX_BYTES = 1 << 20
MANIFEST_MAX_LAYERS = 128
MANIFEST_MAX_CHUNKS = 4096
KVARN4_STORAGE = 4
EXPECT_SCHEMA = 'rkva-host-expect-v1'


class Refusal(RuntimeError):
    """The host state cannot produce this field; refuse, never synthesize."""


def _u16(data, offset):
    return struct.unpack_from('<H', data, offset)[0]


def _u32(data, offset):
    return struct.unpack_from('<I', data, offset)[0]


def _u64(data, offset):
    return struct.unpack_from('<Q', data, offset)[0]


def _pack_u32(value):
    return struct.pack('<I', value)


def _pack_u64(value):
    return struct.pack('<Q', value)


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Canonical digests shared with rkva-qual-probe.h
# ---------------------------------------------------------------------------

def geometry_digest(geometry):
    body = b'RKVA-QUAL-GEOMETRY-V1'
    for key in ('n_head', 'n_head_kv', 'head_dim', 'head_slices', 'max_ctx', 'n_layers'):
        body += _pack_u32(geometry[key])
    return sha256_hex(body)


def descriptor_digest(descriptor):
    body = b'RKVA-QUAL-DESCRIPTOR-V1'
    order = ('origin', 'capacity', 'storage', 'layout_version', 'token_group', 'record_dim',
             'head_dim', 'head_slices', 'key_bits', 'value_bits', 'record_bytes',
             'stage_groups', 'tail_groups', 'sink_tokens', 'precision_tail_tokens',
             'wht_version', 'domain', 'record_policy')
    for key in order:
        body += _pack_u32(descriptor[key])
    return sha256_hex(body)


def coverage_digest(live_rows, generation, layers):
    body = b'RKVA-QUAL-COVERAGE-V1'
    body += _pack_u32(live_rows) + _pack_u64(generation) + _pack_u32(len(layers))
    for index, layer in enumerate(layers):
        body += _pack_u32(index)
        body += _pack_u32(layer['tail_begin'])
        body += _pack_u32(layer['stage_begin'])
        # One checkpoint: the manifest requires every layer at live_rows.
        body += _pack_u32(live_rows)
    return sha256_hex(body)


def state_digest(geometry, descriptor, live_rows, generation, layers):
    body = b'RKVA-QUAL-STATE-V1'
    body += bytes.fromhex(geometry_digest(geometry))
    body += bytes.fromhex(descriptor_digest(descriptor))
    body += bytes.fromhex(coverage_digest(live_rows, generation, layers))
    return sha256_hex(body)


# ---------------------------------------------------------------------------
# Native manifest decoder (mirrors ggml/include/rkva-native-snapshot.h)
# ---------------------------------------------------------------------------

def decode_manifest(data):
    if len(data) < MANIFEST_HEADER + 32 or len(data) > MANIFEST_MAX_BYTES:
        raise Refusal('manifest: invalid size')
    if _u32(data, 0) != MANIFEST_MAGIC or _u16(data, 4) != MANIFEST_VERSION or \
            _u16(data, 6) != MANIFEST_HEADER or _u32(data, 20) != 0:
        raise Refusal('manifest: unsupported version')
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise Refusal('manifest: checksum mismatch')
    live_rows = _u32(data, 8)
    n_layers = _u32(data, 12)
    chunk_count = _u32(data, 16)
    generation = _u64(data, 24)
    host_key = data[32:64]
    expected = MANIFEST_HEADER + 16 * n_layers + 56 * chunk_count + 32
    if expected != len(data):
        raise Refusal('manifest: extent mismatch')
    if not chunk_count or chunk_count > MANIFEST_MAX_CHUNKS or not n_layers or \
            n_layers > MANIFEST_MAX_LAYERS:
        raise Refusal('manifest: invalid layer/chunk count')
    geometry = {
        'n_head': _u32(data, 64), 'n_head_kv': _u32(data, 68),
        'head_dim': _u32(data, 72), 'head_slices': _u32(data, 76),
        'max_ctx': _u32(data, 80), 'n_layers': _u32(data, 84),
    }
    if geometry['n_layers'] != n_layers:
        raise Refusal('manifest: layer count differs from geometry')
    fields = [_u32(data, 88 + 4 * i) for i in range(18)]
    names = ('origin', 'capacity', 'storage', 'layout_version', 'token_group', 'record_dim',
             'head_dim', 'head_slices', 'key_bits', 'value_bits', 'record_bytes',
             'stage_groups', 'tail_groups', 'sink_tokens', 'precision_tail_tokens',
             'wht_version', 'domain', 'record_policy')
    descriptor = dict(zip(names, fields))
    payload_bytes = _u64(data, 160)
    if _u64(data, 168) != 0 or not live_rows or not generation:
        raise Refusal('manifest: invalid live rows/generation/reserved')
    if not all(host_key):
        raise Refusal('manifest: host prefix key is zero')
    _validate_descriptor(geometry, descriptor)
    _validate_coverage(descriptor, live_rows, None)
    layers = []
    offset = MANIFEST_HEADER
    for _ in range(n_layers):
        layers.append({
            'tail_begin': _u32(data, offset), 'stage_begin': _u32(data, offset + 4),
            'record_groups': _u32(data, offset + 8), 'ready': _u32(data, offset + 12),
        })
        offset += 16
    chunks = []
    for _ in range(chunk_count):
        chunk = {
            'layer': _u32(data, offset), 'region': _u32(data, offset + 4),
            'offset': _u64(data, offset + 8), 'bytes': _u32(data, offset + 16),
            'reserved': _u32(data, offset + 20),
            'digest': data[offset + 24:offset + 56].hex(),
        }
        if chunk['reserved'] != 0 or not chunk['bytes'] or chunk['bytes'] > (8 << 20):
            raise Refusal('manifest: non-canonical chunk')
        chunks.append(chunk)
        offset += 56
    for layer in layers:
        _validate_coverage(descriptor, live_rows, layer)
    return {
        'live_rows': live_rows, 'generation': generation,
        'host_state_sha256': host_key.hex(), 'geometry': geometry,
        'descriptor': descriptor, 'layers': layers, 'chunks': chunks,
        'payload_bytes': payload_bytes, 'manifest_sha256': sha256_hex(data),
        'manifest_bytes': len(data),
    }


def _validate_descriptor(geometry, descriptor):
    ok = (geometry['n_head'] and geometry['n_head_kv'] and
          geometry['n_head'] % geometry['n_head_kv'] == 0 and
          geometry['head_dim'] == 256 and geometry['head_slices'] == 2 and
          geometry['max_ctx'] and descriptor['storage'] == KVARN4_STORAGE and
          descriptor['layout_version'] == 3 and descriptor['origin'] % 128 == 0 and
          descriptor['capacity'] > 0 and descriptor['token_group'] == 128 and
          descriptor['record_dim'] == 128 and descriptor['head_dim'] == 256 and
          descriptor['head_slices'] == 2 and descriptor['key_bits'] == 4 and
          descriptor['value_bits'] == 4 and descriptor['record_bytes'] == 8960 and
          descriptor['stage_groups'] == 3 and descriptor['tail_groups'] == 2 and
          descriptor['sink_tokens'] == 128 and
          descriptor['precision_tail_tokens'] in (0, 128) and
          descriptor['wht_version'] == 1 and descriptor['domain'] == 1 and
          descriptor['record_policy'] == 1)
    if not ok:
        raise Refusal('manifest: descriptor does not match the revision-3 native contract')


def _validate_coverage(descriptor, live_rows, layer):
    if live_rows > descriptor['capacity']:
        raise Refusal('manifest: live rows exceed descriptor capacity')
    if layer is None:
        return
    stage_limit = ((live_rows - 1) // 128 * 128) if live_rows else 0
    if layer['ready'] != 1 or layer['record_groups'] != live_rows // 128 or \
            layer['stage_begin'] % 128 or layer['stage_begin'] > stage_limit or \
            layer['tail_begin'] > live_rows:
        raise Refusal('manifest: invalid per-layer coverage')
    if descriptor['precision_tail_tokens'] and layer['tail_begin'] > \
            (live_rows - descriptor['precision_tail_tokens']
             if live_rows > descriptor['precision_tail_tokens'] else 0):
        raise Refusal('manifest: tail coverage does not reach the live prefix')


# ---------------------------------------------------------------------------
# Host journal (transactions + v3 ledger) cross-check
# ---------------------------------------------------------------------------

def decode_journal(path, owner):
    data = Path(path).read_bytes()
    if len(data) < 8 + 64 + 8 + 4 + 32 or len(data) > 16 * 1024 * 1024:
        raise Refusal('journal: invalid size')
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise Refusal('journal: checksum mismatch')
    magic = data[:8].decode('ascii', 'strict')
    if magic not in ('RKHJNL01', 'RKHJNL02', 'RKHJNL03'):
        raise Refusal('journal: unsupported format')
    if data[8:72].decode('ascii', 'strict') != owner:
        raise Refusal('journal: owner mismatch')
    offset = 72
    revision = _u64(data, offset)
    offset += 8
    count = _u32(data, offset)
    offset += 4
    if count > 8192:
        raise Refusal('journal: transaction count exceeded')
    transactions = []
    seen_tx, seen_ref = set(), set()
    for _ in range(count):
        tx = data[offset:offset + 64].decode('ascii')
        ref = data[offset + 64:offset + 128].decode('ascii')
        key = data[offset + 128:offset + 192].decode('ascii')
        state = data[offset + 192]
        host_bytes = _u64(data, offset + 193)
        peer_version = _u64(data, offset + 201)
        host_sha = data[offset + 209:offset + 273].decode('ascii')
        path_len = _u32(data, offset + 273)
        offset += 277
        if path_len > 4096:
            raise Refusal('journal: path too long')
        host_path = data[offset:offset + path_len].decode('utf-8', 'strict')
        offset += path_len
        for value in (tx, ref, key):
            if not HEX64.fullmatch(value) or value == '0' * 64:
                raise Refusal('journal: invalid identity')
        if state < 1 or state > 6 or tx in seen_tx or ref in seen_ref:
            raise Refusal('journal: invalid phase/duplicate identity')
        seen_tx.add(tx)
        seen_ref.add(ref)
        transactions.append({
            'tx': tx, 'reference': ref, 'key': key, 'phase': state,
            'host_bytes': host_bytes, 'peer_commit_version': peer_version,
            'host_sha256': '' if host_sha == '0' * 64 else host_sha,
            'host_path': host_path,
        })
    v3_nonces = []
    if magic in ('RKHJNL02', 'RKHJNL03'):
        offset += 8 + 8 + 32  # key_epoch, version, digest
        nonce_count = _u32(data, offset)
        offset += 4
        if nonce_count > 50000:
            raise Refusal('journal: nonce count exceeded')
        offset += nonce_count * 97
        if magic == 'RKHJNL03':
            offset += 32
            v3_count = _u32(data, offset)
            offset += 4
            if v3_count > 50000:
                raise Refusal('journal: v3 nonce count exceeded')
            for _ in range(v3_count):
                tx = data[offset:offset + 64].decode('ascii')
                ref = data[offset + 64:offset + 128].decode('ascii')
                key = data[offset + 128:offset + 192].decode('ascii')
                op = _u32(data, offset + 192)
                nonce = data[offset + 196:offset + 228].hex()
                consumed = data[offset + 396] != 0
                offset += 397
                v3_nonces.append({'tx': tx, 'reference': ref, 'key': key, 'op': op,
                                  'nonce': nonce, 'consumed': consumed})
    return {'magic': magic, 'revision': revision,
            'transactions': transactions, 'v3_nonces': v3_nonces}


def find_journal(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'host-journal.bin'
    if not path.is_file():
        raise Refusal('journal: host-journal.bin not found')
    return path


# ---------------------------------------------------------------------------
# SQLite managed store
# ---------------------------------------------------------------------------

def open_store(path):
    path = Path(path)
    if path.is_dir():
        candidate = path / 'peer.db'
        if candidate.is_file():
            db_path = candidate
        else:
            nested = path / 'rkva-managed-v3' / 'peer.db'
            db_path = nested if nested.is_file() else candidate
    else:
        db_path = path
    if not db_path.is_file():
        raise Refusal('store: peer.db not found')
    uri = 'file:' + str(db_path.resolve()) + '?mode=ro'
    connection = sqlite3.connect(uri, uri=True)
    connection.execute('PRAGMA query_only=ON')
    tables = {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    needed = {'meta', 'refs', 'pins', 'receipts'}
    if not needed.issubset(tables):
        connection.close()
        raise Refusal('store: managed schema is missing ' + ','.join(sorted(needed - tables)))
    return connection, db_path


def hex_blob(value):
    return bytes(value).hex()


def read_store(db, tx):
    meta = db.execute('SELECT owner,epoch,version,quota FROM meta').fetchall()
    if len(meta) != 1:
        raise Refusal('store: meta must contain exactly one row')
    owner, epoch, version, quota = meta[0]
    pins = db.execute(
        'SELECT tx,identity,hello,metadata,reservation FROM pins ORDER BY tx').fetchall()
    if not pins:
        raise Refusal('store: no pending pin; there is no real attempt to bind')
    if tx:
        pins = [row for row in pins if hex_blob(row[0]) == tx]
        if not pins:
            raise Refusal('store: tx is not a pending pin')
    elif len(pins) != 1:
        raise Refusal('store: %d pending pins; pass --tx to select one' % len(pins))
    pin_tx, identity, hello, metadata, reservation = pins[0]
    if len(metadata) > MANIFEST_MAX_BYTES:
        raise Refusal('store: pin metadata exceeds the manifest bound')
    refs = db.execute(
        'SELECT tx,ref,obj,phase,commit_version FROM refs WHERE tx=?', (pin_tx,)).fetchall()
    receipts = db.execute(
        'SELECT tx,ref,obj,obj_digest,nonce,prev_version,prev_digest,next_version,next_digest,mac '
        'FROM receipts WHERE tx=?', (pin_tx,)).fetchall()
    return {
        'owner': hex_blob(owner), 'epoch': hex_blob(epoch),
        'version': int(version), 'quota': int(quota),
        'pin': {'tx': hex_blob(pin_tx), 'identity': hex_blob(identity),
                'hello_bytes': len(hello), 'reservation': int(reservation)},
        'refs': [{'tx': hex_blob(row[0]), 'reference': hex_blob(row[1]),
                  'obj_id': hex_blob(row[2]), 'phase': int(row[3]),
                  'commit_version': int(row[4])} for row in refs],
        'receipts': [{'tx': hex_blob(row[0]), 'reference': hex_blob(row[1]),
                      'obj_id': hex_blob(row[2]), 'obj_digest': hex_blob(row[3]),
                      'nonce': hex_blob(row[4]), 'prev_version': int(row[5]),
                      'next_version': int(row[7]), 'next_digest': hex_blob(row[8]),
                      'mac': hex_blob(row[9])} for row in receipts],
        'metadata': bytes(metadata),
    }


# ---------------------------------------------------------------------------
# Expectation document
# ---------------------------------------------------------------------------

def build_expectation(args):
    document = {
        'schema': EXPECT_SCHEMA,
        'expectation_available': False,
        'unavailable': {},
        'source': {},
        'object': None,
        'journal': None,
    }
    manifest_bytes = None
    store_state = None
    db = None
    try:
        if args.store:
            db, db_path = open_store(args.store)
            store_state = read_store(db, args.tx)
            manifest_bytes = store_state['metadata']
            document['source']['store'] = str(db_path)
            document['store'] = {
                'owner': store_state['owner'], 'epoch': store_state['epoch'],
                'version': store_state['version'], 'quota': store_state['quota'],
                'pin': store_state['pin'],
                'refs': store_state['refs'],
                'receipts': store_state['receipts'],
            }
            if store_state['refs']:
                ref = store_state['refs'][0]
                document['object'] = {
                    'tx': ref['tx'], 'reference': ref['reference'],
                    'object_id': ref['obj_id'], 'phase': ref['phase'],
                    'commit_version': ref['commit_version'],
                }
            else:
                document['unavailable']['object_key'] = (
                    'the selected pin has no refs row; no object identity was published')
        if args.manifest:
            raw = Path(args.manifest).read_bytes()
            document['source']['manifest'] = str(Path(args.manifest))
            if manifest_bytes is not None and raw != manifest_bytes:
                raise Refusal('manifest file differs from the SQLite pin metadata')
            manifest_bytes = raw
        if manifest_bytes is None:
            raise Refusal('no real source: pass --store or --manifest')
        manifest = decode_manifest(manifest_bytes)
    finally:
        if db is not None:
            db.close()

    # Repeated rows are composed from real manifest values only.
    document['rows'] = manifest['live_rows']
    document['generation'] = manifest['generation']
    document['host_key_sha256'] = manifest['host_state_sha256']
    document['prefix_sha256'] = manifest['host_state_sha256']
    document['geometry'] = manifest['geometry']
    document['descriptor'] = manifest['descriptor']
    document['layers'] = manifest['layers']
    document['chunk_count'] = len(manifest['chunks'])
    document['payload_bytes'] = manifest['payload_bytes']
    document['manifest_sha256'] = manifest['manifest_sha256']
    document['geometry_sha256'] = geometry_digest(manifest['geometry'])
    document['descriptor_sha256'] = descriptor_digest(manifest['descriptor'])
    document['coverage_sha256'] = coverage_digest(
        manifest['live_rows'], manifest['generation'], manifest['layers'])
    document['state_digest'] = state_digest(
        manifest['geometry'], manifest['descriptor'], manifest['live_rows'],
        manifest['generation'], manifest['layers'])

    # Session id and fence pair are not stored by the host attempt or the
    # native manifest. They exist only in the live HELLO/session and the GPU.
    document['session_id'] = None
    document['unavailable']['session_id'] = (
        'the SQLite pin and native manifest bind the host prefix key, geometry, rows, '
        'generation and coverage, but not the RKVA session id; only a live accepted HELLO '
        'carries it and no host transport exists in this tree')
    document['unavailable']['submitted_fence'] = (
        'the host journal records a commit version, not the console GPU fence pair')
    document['unavailable']['completed_fence'] = (
        'the host journal records a commit version, not the console GPU fence pair')
    document['unavailable']['coverage_mac'] = (
        'the host has no authenticated ATTEST reply; the resident coverage MAC comes from '
        'the live managed session, not from the journal/SQLite attempt')

    if args.object_root and document['object']:
        root = Path(args.object_root)
        object_dir = root / manifest['host_state_sha256']
        object_manifest = object_dir / 'manifest-v2.bin'
        if not object_manifest.is_file():
            raise Refusal('object: %s does not exist' % object_manifest)
        published = object_manifest.read_bytes()
        if published != manifest_bytes:
            raise Refusal('object: published manifest differs from the pinned metadata')
        document['object']['manifest_path'] = str(object_manifest)
        document['object']['manifest_sha256'] = sha256_hex(published)
        document['object']['present'] = True

    if args.journal:
        if not args.owner:
            raise Refusal('--journal requires --owner (the journal owner is 64 hex)')
        if not HEX64.fullmatch(args.owner) or args.owner == '0' * 64:
            raise Refusal('--owner must be 64 lowercase hex')
        journal_path = find_journal(args.journal)
        journal = decode_journal(journal_path, args.owner)
        document['source']['journal'] = str(journal_path)
        selected = document['store']['pin']['tx'] if 'store' in document else None
        if not selected:
            raise Refusal('journal cross-check requires the SQLite pin')
        matches = [item for item in journal['transactions'] if item['tx'] == selected]
        if len(matches) != 1:
            raise Refusal('journal: the selected tx is not a journal transaction')
        tx = matches[0]
        if document['object'] and tx['reference'] != document['object']['reference']:
            raise Refusal('journal: reference differs from the SQLite refs row')
        if tx['key'] != manifest['host_state_sha256']:
            raise Refusal('journal: key differs from the pinned manifest host key')
        v3 = [item for item in journal['v3_nonces'] if item['tx'] == selected]
        document['journal'] = {
            'magic': journal['magic'], 'revision': journal['revision'],
            'phase': tx['phase'], 'host_bytes': tx['host_bytes'],
            'host_sha256': tx['host_sha256'],
            'peer_commit_version': tx['peer_commit_version'],
            'v3_nonces': v3,
        }

    # The controller compares five fields. Four are real; session_id is not,
    # so this expectation can never authorize PASS by itself.
    document['expectation_available'] = (
        document['session_id'] is not None and
        not any(key in document['unavailable'] for key in
                ('session_id', 'rows', 'generation', 'host_key_sha256', 'prefix_sha256')))
    if not args.store:
        document['unavailable']['object_key'] = (
            'no --store supplied; the manifest file alone cannot bind a transaction/object')
    return document


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', help='managed store dir or peer.db path (read-only)')
    parser.add_argument('--tx', help='select a pending pin by 64-hex transaction id')
    parser.add_argument('--manifest', help='published manifest-v2.bin to use or cross-check')
    parser.add_argument('--journal', help='host journal dir or host-journal.bin')
    parser.add_argument('--owner', help='journal owner, 64 lowercase hex')
    parser.add_argument('--object-root', help='published rkva-snapshots-v3 root to verify')
    parser.add_argument('--out', help='write here instead of stdout')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        document = build_expectation(args)
    except Refusal as exc:
        print(json.dumps({'schema': EXPECT_SCHEMA, 'status': 'refused',
                          'reason': str(exc)}, sort_keys=False), file=sys.stderr)
        return 2
    text = json.dumps(document, indent=2, sort_keys=False) + '\n'
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
