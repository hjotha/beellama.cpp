import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

import rkva_mq_capture_compare as capture


def fixture(path, kernel=1, direct=False, mutate=None, splits=2):
    nq, nh, nkv, dim, capacity, tail = 2, 6, 1, 256, 128, 128
    split = 1 if direct else splits
    records = 17920
    stage = 384 * nkv * dim * 2
    half = records + stage + 256 * nkv * dim * 2
    q = nq * nh * dim * 4
    query, kv = 2 * q + 4 * nq, 2 * half
    partial = 0 if direct else split * nq * nh * (dim + 2) * 4
    output = q + nq * nh * 4
    params = (nq, 128, nh, nkv, dim, 0, 0, split, 64, int(direct), records, tail)
    words = (0x5043514d, 1, 344, 0, 0, 0, 0, kernel,
             nkv * (3 if kernel == 2 else 1), (nq + 1) // 2, split,
             64, 64, *params, capacity, 1, 0, 0)
    geometry = (nh, nkv, dim) + (0,) * 12
    descriptor = (0, capacity, 4, 3, 128, 128, dim, 2, 4, 4, 8960, 3, 2, 128,
                  tail, 1, 1, 1)
    header = struct.pack('<29I8Q32s15I18I', *words, 1, records, stage, half,
                         query, kv, partial, output, bytes(32), *geometry, *descriptor)
    assert len(header) == 344
    payload = bytearray(query + kv + partial + output)
    if mutate is not None:
        payload[mutate] ^= 1
    body = header + payload
    path.write_bytes(body + hashlib.sha256(body).digest())


class CaptureCompare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.first = Path(self.tmp.name) / 'a.bin'
        self.second = Path(self.tmp.name) / 'b.bin'
        fixture(self.first)
        fixture(self.second, kernel=2)

    def tearDown(self):
        self.tmp.cleanup()

    def check_pair(self):
        a, b = capture.Capture(self.first), capture.Capture(self.second)
        try:
            return capture.compare(a, b)
        finally:
            a.close()
            b.close()

    def test_equal_and_physical_kv_inconclusive(self):
        result = self.check_pair()
        self.assertEqual(result['kv_consumed_status'], 'identical_physical_planes')
        self.assertTrue(result['all_physical_inputs_identical'])
        self.assertTrue(result['spans']['query']['equal'])
        q = 2 * 6 * 256 * 4
        fixture(self.second, kernel=2, mutate=2 * q + 8 + 4)
        result = self.check_pair()
        self.assertEqual(result['kv_consumed_status'], 'inconclusive_unproven_consumed_spans')
        self.assertFalse(result['all_physical_inputs_identical'])
        self.assertFalse(result['spans']['k_records']['equal'])
        self.assertTrue(result['spans']['query']['equal'])

    def test_independent_query_partial_and_lse(self):
        q = 2 * 6 * 256 * 4
        query = 2 * q + 8
        half = 17920 + 384 * 256 * 2 + 256 * 256 * 2
        partial = 2 * 2 * 6 * 258 * 4
        for name, offset in (('qrot', q), ('positions', 2 * q),
                             ('partial', query + 2 * half),
                             ('lse', query + 2 * half + partial + q)):
            with self.subTest(name=name):
                fixture(self.second, kernel=2, mutate=offset)
                self.assertFalse(self.check_pair()['spans'][name]['equal'])

    def test_direct_output_and_corruption(self):
        fixture(self.first, direct=True)
        fixture(self.second, kernel=2, direct=True)
        self.assertEqual(self.check_pair()['spans']['partial']['bytes_a'], 0)
        data = bytearray(self.second.read_bytes())
        data[344] ^= 1
        self.second.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            capture.Capture(self.second)

    def test_bad_header_and_truncation(self):
        data = bytearray(self.second.read_bytes())
        struct.pack_into('<I', data, 8, 343)
        data[-32:] = hashlib.sha256(data[:-32]).digest()
        self.second.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'header'):
            capture.Capture(self.second)
        self.second.write_bytes(b'short')
        with self.assertRaisesRegex(ValueError, 'size'):
            capture.Capture(self.second)

    def test_invalid_extent_with_valid_digest(self):
        data = bytearray(self.second.read_bytes())
        struct.pack_into('<Q', data, 116 + 4 * 8, capture.LIMIT + 1)
        data[-32:] = hashlib.sha256(data[:-32]).digest()
        self.second.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'extents'):
            capture.Capture(self.second)

    def test_split_64_extent(self):
        fixture(self.first, splits=64)
        fixture(self.second, kernel=2, splits=64)
        result = self.check_pair()
        self.assertEqual(result['spans']['partial']['bytes_a'], 64 * 2 * 6 * 258 * 4)
        self.assertTrue(result['spans']['partial']['equal'])

    def test_split_limit(self):
        fixture(self.first, splits=256)
        a = capture.Capture(self.first)
        a.close()
        fixture(self.first, splits=257)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            capture.Capture(self.first)


if __name__ == '__main__':
    unittest.main()
