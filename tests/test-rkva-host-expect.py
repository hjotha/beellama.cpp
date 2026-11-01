#!/usr/bin/env python3
"""CPU tests for the host expectation producer and its controller schema.

Builds a real SQLite managed store (exact schema used by
rkva-anchored-sqlite.cpp), a canonical native manifest with the real chunk
layout, and a real RKHJNL03 journal. Then runs
tools/xbox/scripts/rkva-host-expect.py and checks the emitted expectation.
No console, network, GPU or production service is touched.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT / 'tools/xbox/scripts/rkva-host-expect.py'
CONTROLLER = ROOT / 'tools/xbox/scripts/xbox-snapshot-console-qual.py'

spec = importlib.util.spec_from_file_location('host_expect', PRODUCER)
expect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expect)

spec = importlib.util.spec_from_file_location('console_qual', CONTROLLER)
console = importlib.util.module_from_spec(spec)
spec.loader.exec_module(console)

GEOMETRY = {'n_head': 8, 'n_head_kv': 2, 'head_dim': 256, 'head_slices': 2,
            'max_ctx': 4096, 'n_layers': 2}
DESCRIPTOR = {'origin': 0, 'capacity': 4096, 'storage': 4, 'layout_version': 3,
              'token_group': 128, 'record_dim': 128, 'head_dim': 256, 'head_slices': 2,
              'key_bits': 4, 'value_bits': 4, 'record_bytes': 8960, 'stage_groups': 3,
              'tail_groups': 2, 'sink_tokens': 128, 'precision_tail_tokens': 128,
              'wht_version': 1, 'domain': 1, 'record_policy': 1}
LAYERS = [{'tail_begin': 385, 'stage_begin': 384, 'record_groups': 4, 'ready': 1},
          {'tail_begin': 385, 'stage_begin': 384, 'record_groups': 4, 'ready': 1}]
KEY = bytes(range(1, 33))


def _p32(value):
    return struct.pack('<I', value)


def _p64(value):
    return struct.pack('<Q', value)


def canonical_chunks(live_rows=513, geometry=GEOMETRY, descriptor=DESCRIPTOR):
    groups = live_rows // 128
    record = geometry['n_head_kv'] * 2 * descriptor['record_bytes']
    stage = 384 * geometry['n_head_kv'] * 256 * 2
    tail = 256 * geometry['n_head_kv'] * 256 * 2
    half = groups * record + stage + tail
    chunks = []
    total = 0
    for layer in range(geometry['n_layers']):
        for kind in range(2):
            for region, base, size in (
                    (1 if kind == 0 else 2, kind * half, groups * record),
                    (3 if kind == 0 else 4, kind * half + groups * record, stage),
                    (5 if kind == 0 else 6,
                     kind * half + groups * record + stage, tail)):
                offset = 0
                while offset < size:
                    n = min(size - offset, 8 << 20)
                    chunks.append({'layer': layer, 'region': region,
                                   'offset': base + offset, 'bytes': n,
                                   'reserved': 0, 'digest': b'\0' * 32})
                    total += n
                    offset += n
    return chunks, total


def encode_manifest(live_rows=513, generation=0x700000001, key=KEY,
                    geometry=GEOMETRY, descriptor=DESCRIPTOR, layers=LAYERS):
    chunks, payload = canonical_chunks(live_rows, geometry, descriptor)
    out = bytearray(176)
    struct.pack_into('<I', out, 0, 0x53564B52)
    struct.pack_into('<H', out, 4, 1)
    struct.pack_into('<H', out, 6, 176)
    struct.pack_into('<I', out, 8, live_rows)
    struct.pack_into('<I', out, 12, geometry['n_layers'])
    struct.pack_into('<I', out, 16, len(chunks))
    struct.pack_into('<I', out, 20, 0)
    struct.pack_into('<Q', out, 24, generation)
    out[32:64] = key
    for index, name in enumerate(('n_head', 'n_head_kv', 'head_dim', 'head_slices',
                                  'max_ctx', 'n_layers')):
        struct.pack_into('<I', out, 64 + 4 * index, geometry[name])
    order = ('origin', 'capacity', 'storage', 'layout_version', 'token_group', 'record_dim',
             'head_dim', 'head_slices', 'key_bits', 'value_bits', 'record_bytes',
             'stage_groups', 'tail_groups', 'sink_tokens', 'precision_tail_tokens',
             'wht_version', 'domain', 'record_policy')
    for index, name in enumerate(order):
        struct.pack_into('<I', out, 88 + 4 * index, descriptor[name])
    struct.pack_into('<Q', out, 160, payload)
    for layer in layers:
        out += _p32(layer['tail_begin']) + _p32(layer['stage_begin'])
        out += _p32(layer['record_groups']) + _p32(layer['ready'])
    for chunk in chunks:
        out += _p32(chunk['layer']) + _p32(chunk['region'])
        out += _p64(chunk['offset']) + _p32(chunk['bytes']) + _p32(0)
        out += chunk['digest']
    out += hashlib.sha256(bytes(out)).digest()
    return bytes(out)


def make_store(directory, manifest, tx=None, ref=None, obj=None, pins=1):
    tx = tx or bytes.fromhex('11' * 32)
    ref = ref or bytes.fromhex('22' * 32)
    obj = obj or hashlib.sha256(b'object-key').digest()
    root = Path(directory) / 'rkva-managed-v3'
    root.mkdir(parents=True)
    db = sqlite3.connect(root / 'peer.db')
    db.execute('CREATE TABLE meta(owner BLOB NOT NULL,epoch BLOB NOT NULL,'
               'version INTEGER NOT NULL,quota INTEGER NOT NULL)')
    db.execute('CREATE TABLE refs(tx BLOB PRIMARY KEY,ref BLOB NOT NULL UNIQUE,'
               'obj BLOB NOT NULL,phase INTEGER NOT NULL,commit_version INTEGER NOT NULL)')
    db.execute('CREATE TABLE nonces(nonce BLOB PRIMARY KEY)')
    db.execute('CREATE TABLE pins(tx BLOB PRIMARY KEY,identity BLOB NOT NULL,'
               'hello BLOB NOT NULL,metadata BLOB NOT NULL,reservation INTEGER NOT NULL)')
    db.execute('CREATE TABLE receipts(tx BLOB PRIMARY KEY,ref BLOB NOT NULL,'
               'obj BLOB NOT NULL,obj_digest BLOB NOT NULL,nonce BLOB NOT NULL,'
               'prev_version INTEGER NOT NULL,prev_digest BLOB NOT NULL,'
               'next_version INTEGER NOT NULL,next_digest BLOB NOT NULL,mac BLOB NOT NULL)')
    db.execute('INSERT INTO meta VALUES(?,?,?,?)',
               (bytes.fromhex('33' * 32), bytes.fromhex('44' * 32), 9, 1 << 40))
    for index in range(pins):
        pin_tx = tx if index == 0 else bytes([0x55] * 31 + [index])
        pin_ref = ref if index == 0 else bytes([0x66] * 31 + [index])
        db.execute('INSERT INTO pins VALUES(?,?,?,?,?)',
                   (pin_tx, bytes.fromhex('77' * 32), bytes(80), manifest, 4096))
        db.execute('INSERT INTO refs VALUES(?,?,?,?,?)',
                   (pin_tx, pin_ref, obj, 2, 0))
    db.commit()
    db.close()
    return root / 'peer.db', tx, ref, obj


def encode_journal(tx, ref, key, owner, path=b'/tmp/host-snapshot.bin',
                   v3_key=None, state=2):
    body = bytearray(b'RKHJNL03')
    body += owner.encode()
    body += _p64(1)
    body += _p32(1)
    body += tx.hex().encode() + ref.hex().encode() + key.hex().encode()
    body += bytes([state])
    body += _p64(1234)
    body += _p64(0)
    body += hashlib.sha256(b'host-file').hexdigest().encode()
    body += _p32(len(path)) + path
    body += _p64(5) + _p64(3) + bytes.fromhex('88' * 32)
    body += _p32(0)  # v2 nonces
    body += bytes.fromhex('99' * 32)
    body += _p32(1)
    body += tx.hex().encode() + ref.hex().encode() + (v3_key or key).hex().encode()
    body += _p32(31)
    body += bytes.fromhex('aa' * 32)  # nonce
    body += bytes.fromhex('bb' * 32)  # request hash
    body += bytes.fromhex('cc' * 32)  # transcript
    body += bytes.fromhex('dd' * 32)  # challenge
    body += bytes.fromhex('ee' * 32)  # anchor digest
    body += bytes(32)                 # response hash (not consumed)
    body += _p64(4) + bytes([0])
    body += hashlib.sha256(bytes(body)).digest()
    return bytes(body)


def run_producer(*args):
    return subprocess.run([sys.executable, str(PRODUCER), *args],
                          text=True, capture_output=True, check=False)


class HostExpectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='rkva-host-expect-'))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_single_pin_emits_real_fields(self):
        manifest = encode_manifest()
        db, tx, ref, obj = make_store(self.tmp, manifest)
        result = run_producer('--store', str(db.parent), '--out', str(self.tmp / 'e.json'))
        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads((self.tmp / 'e.json').read_text())
        self.assertEqual(document['schema'], 'rkva-host-expect-v1')
        self.assertEqual(document['rows'], 513)
        self.assertEqual(document['generation'], 0x700000001)
        self.assertEqual(document['host_key_sha256'], KEY.hex())
        self.assertEqual(document['prefix_sha256'], KEY.hex())
        self.assertEqual(document['geometry'], GEOMETRY)
        self.assertEqual(document['descriptor'], DESCRIPTOR)
        self.assertEqual([{k: layer[k] for k in ('tail_begin', 'stage_begin',
                                                 'record_groups', 'ready')}
                          for layer in document['layers']], LAYERS)
        self.assertEqual(document['object']['tx'], tx.hex())
        self.assertEqual(document['object']['reference'], ref.hex())
        self.assertEqual(document['object']['object_id'], obj.hex())
        self.assertIsNone(document['session_id'])
        self.assertIn('session_id', document['unavailable'])
        self.assertFalse(document['expectation_available'])
        self.assertEqual(document['source']['store'], str(db))
        self.assertEqual(document['geometry_sha256'],
                         expect.geometry_digest(GEOMETRY))
        self.assertEqual(document['descriptor_sha256'],
                         expect.descriptor_digest(DESCRIPTOR))
        self.assertEqual(document['coverage_sha256'],
                         expect.coverage_digest(513, 0x700000001, LAYERS))
        self.assertEqual(document['state_digest'],
                         expect.state_digest(GEOMETRY, DESCRIPTOR, 513,
                                             0x700000001, LAYERS))
        # Cross-language vectors: rkva_qual_probe_cpu.cpp asserts the same
        # values for the same real inputs.
        self.assertEqual(document['geometry_sha256'],
                         'f872fd04fe0c09da8ee163b191aa45164a0715f1c6d95338be4933f072ea8e0d')
        self.assertEqual(document['descriptor_sha256'],
                         '403df8d102e91bc73a2abe2fd3bc24351e24b5da20a2444b307b20a48a0c98c6')
        self.assertEqual(document['coverage_sha256'],
                         'e51bfd43789fdbf1e34129895feda0b69077ebb2f2304fd9b9365ab2b3704a1e')
        self.assertEqual(document['state_digest'],
                         '2fbd10c8fd7b21728b3935ca4abf73cc74003c09da3660f549fe213b5dad0af8')

    def test_values_follow_the_real_attempt(self):
        first = encode_manifest()
        db, _, _, _ = make_store(self.tmp, first)
        out1 = run_producer('--store', str(db.parent))
        self.assertEqual(out1.returncode, 0, out1.stderr)
        doc1 = json.loads(out1.stdout)
        # A different real attempt must produce different expected values.
        shutil.rmtree(self.tmp)
        self.tmp.mkdir()
        second = encode_manifest(live_rows=641, generation=0x700000009,
                                 key=bytes(range(33, 65)),
                                 layers=[{'tail_begin': 513, 'stage_begin': 512,
                                          'record_groups': 5, 'ready': 1},
                                         {'tail_begin': 513, 'stage_begin': 512,
                                          'record_groups': 5, 'ready': 1}])
        db2, _, _, _ = make_store(self.tmp, second)
        out2 = run_producer('--store', str(db2.parent))
        self.assertEqual(out2.returncode, 0, out2.stderr)
        doc2 = json.loads(out2.stdout)
        self.assertEqual(doc2['rows'], 641)
        self.assertEqual(doc2['generation'], 0x700000009)
        self.assertEqual(doc2['host_key_sha256'], bytes(range(33, 65)).hex())
        self.assertNotEqual(doc1['coverage_sha256'], doc2['coverage_sha256'])
        self.assertNotEqual(doc1['descriptor_sha256'] is None, True)

    def test_ambiguous_pins_refused(self):
        manifest = encode_manifest()
        db, _, _, _ = make_store(self.tmp, manifest, pins=2)
        result = run_producer('--store', str(db.parent))
        self.assertEqual(result.returncode, 2)
        self.assertIn('2 pending pins', result.stderr)

    def test_tampered_manifest_refused(self):
        manifest = bytearray(encode_manifest())
        manifest[40] ^= 0xFF
        db, _, _, _ = make_store(self.tmp, bytes(manifest))
        result = run_producer('--store', str(db.parent))
        self.assertEqual(result.returncode, 2)
        self.assertIn('checksum mismatch', result.stderr)

    def test_journal_cross_check(self):
        manifest = encode_manifest()
        db, tx, ref, _ = make_store(self.tmp, manifest)
        owner = 'aa' * 32
        journal_dir = self.tmp / 'journal'
        journal_dir.mkdir()
        (journal_dir / 'host-journal.bin').write_bytes(
            encode_journal(tx, ref, KEY, owner))
        result = run_producer('--store', str(db.parent), '--journal', str(journal_dir),
                              '--owner', owner)
        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(result.stdout)
        self.assertEqual(document['journal']['magic'], 'RKHJNL03')
        self.assertEqual(document['journal']['phase'], 2)
        self.assertEqual(document['journal']['peer_commit_version'], 0)
        self.assertEqual(len(document['journal']['v3_nonces']), 1)
        self.assertEqual(document['journal']['v3_nonces'][0]['op'], 31)
        self.assertFalse(document['journal']['v3_nonces'][0]['consumed'])
        # A journal whose key differs from the pinned manifest must refuse.
        (journal_dir / 'host-journal.bin').write_bytes(
            encode_journal(tx, ref, bytes(range(33, 65)), owner))
        mismatch = run_producer('--store', str(db.parent), '--journal', str(journal_dir),
                                '--owner', owner)
        self.assertEqual(mismatch.returncode, 2)
        self.assertIn('key differs', mismatch.stderr)

    def test_object_manifest_cross_check(self):
        manifest = encode_manifest()
        db, _, _, _ = make_store(self.tmp, manifest)
        object_root = self.tmp / 'rkva-snapshots-v3'
        object_dir = object_root / KEY.hex()
        object_dir.mkdir(parents=True)
        (object_dir / 'manifest-v2.bin').write_bytes(manifest)
        ok = run_producer('--store', str(db.parent), '--object-root', str(object_root))
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertTrue(json.loads(ok.stdout)['object']['present'])
        (object_dir / 'manifest-v2.bin').write_bytes(encode_manifest(live_rows=641))
        bad = run_producer('--store', str(db.parent), '--object-root', str(object_root))
        self.assertEqual(bad.returncode, 2)
        self.assertIn('published manifest differs', bad.stderr)

    def test_controller_schema_accepts_partial_expect(self):
        manifest = encode_manifest()
        db, _, _, _ = make_store(self.tmp, manifest)
        result = run_producer('--store', str(db.parent), '--out', str(self.tmp / 'e.json'))
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded = console.load_expect(self.tmp / 'e.json')
        gaps = console.expectation_gaps(loaded)
        self.assertTrue(gaps)
        self.assertIn('session_id', gaps[0])
        # A probe that honestly reports the same gap is BLOCKED, never PASS.
        probe = {'status': 'unavailable',
                 'reason': 'no live RKVA engine: no accepted HELLO in this app process',
                 'attestation': {
                     'unavailable_fields': {
                         'session_id': 'no live engine',
                         'coverage_mac': 'no authenticated managed session'}}}
        status, detail = console.classify_probe_result('ok', True, probe)
        self.assertEqual(status, 'BLOCKED')
        self.assertIn('no live RKVA engine', detail)
        self.assertIn('no live RKVA engine',
                      console.probe_blocking_reason(probe, require_mac=True))
        # Field-level reasons are reported when the document claims availability
        # but a required field is null.
        field_only = {'status': 'available',
                      'build': {'phase3_linked': True, 'phase4_linked': True,
                                'managed_caps_negotiated': True},
                      'attestation': {
                          'unavailable_fields': {
                              'session_id': 'no live engine',
                              'coverage_mac': 'no authenticated managed session'}}}
        blocking = console.probe_blocking_reason(field_only, require_mac=True)
        self.assertIn('session_id', blocking)
        self.assertIn('coverage_mac', blocking)

    def test_controller_rejects_null_without_reason(self):
        document = {'rows': 513, 'generation': 1, 'session_id': None,
                    'host_key_sha256': KEY.hex(), 'prefix_sha256': KEY.hex()}
        (self.tmp / 'bad.json').write_text(json.dumps(document))
        with self.assertRaises(RuntimeError):
            console.load_expect(self.tmp / 'bad.json')


if __name__ == '__main__':
    unittest.main()
