#!/usr/bin/env python3
"""Validate and compare two RKVA MQ captures without printing captured values."""
import argparse
import hashlib
import json
import mmap
import struct

HEADER = 344
LIMIT = 512 << 20


class Capture:
    def __init__(self, path):
        self.file = open(path, 'rb')
        self.data = None
        try:
            size = self.file.seek(0, 2)
            if not HEADER + 32 < size <= HEADER + LIMIT + 32:
                raise ValueError('file size outside limits')
            self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            words = struct.unpack_from('<29I', self.data)
            if words[:3] != (0x5043514d, 1, HEADER):
                raise ValueError('invalid header')
            self.layer, self.kernel = words[4], words[7]
            self.params = words[13:25]
            self.descriptor = struct.unpack_from('<18I', self.data, 272)
            self.geometry = self.data[212:272]
            nq, active, nh, nkv, dim, _, origin, split, _, direct, rec_const, tail = self.params
            generation, records, stage, half, query, kv, partial, output = struct.unpack_from(
                '<8Q', self.data, 116)
            del generation, active
            if not (2 <= nq <= 9 and 1 <= nkv and nh <= 256 and nh == 6 * nkv and dim == 256 and
                    1 <= split <= 256 and direct == (split == 1) and self.kernel in (1, 2)):
                raise ValueError('invalid dispatch geometry')
            if (words[8:11] != (nkv * (3 if self.kernel == 2 else 1), (nq + 1) // 2, split) or
                    (self.kernel == 2 and words[11:13] != (64, 64))):
                raise ValueError('invalid dispatch dimensions')
            if struct.unpack_from('<3I', self.geometry) != (nh, nkv, dim):
                raise ValueError('geometry mismatch')
            desc = self.descriptor
            if (desc[0] != origin or desc[2] != 4 or desc[3] != 3 or desc[6] != dim or
                    desc[14] != tail or rec_const != records or words[25] != desc[1]):
                raise ValueError('descriptor mismatch')
            if (not desc[1] or tail not in (0, 128) or
                    records != ((desc[1] + 127) // 128) * nkv * 2 * 8960 or
                    stage != 384 * nkv * 256 * 2 or
                    half != records + stage + (256 if tail else 0) * nkv * 256 * 2):
                raise ValueError('invalid physical KV layout')
            q = nq * nh * dim * 4
            if (query != 2 * q + 4 * nq or kv != 2 * half or records + stage > half or
                    partial != (0 if direct else split * nq * nh * (dim + 2) * 4) or
                    output != q + nq * nh * 4):
                raise ValueError('invalid payload extents')
            total = query + kv + partial + output
            if total > LIMIT or HEADER + total + 32 != size:
                raise ValueError('payload length mismatch')
            view = memoryview(self.data)
            try:
                digest = hashlib.sha256(view[:-32]).digest()
            finally:
                view.release()
            if digest != self.data[-32:]:
                raise ValueError('SHA-256 mismatch')
            pos = HEADER
            self.spans = {'query': (pos, q), 'qrot': (pos + q, q),
                          'positions': (pos + 2 * q, nq * 4)}
            pos += query
            for plane in ('k', 'v'):
                self.spans[plane + '_records'] = (pos, records)
                self.spans[plane + '_stage'] = (pos + records, stage)
                self.spans[plane + '_tail'] = (pos + records + stage, half - records - stage)
                pos += half
            self.spans['partial'] = (pos, partial)
            pos += partial
            self.spans['output'] = (pos, q)
            self.spans['lse'] = (pos + q, nq * nh * 4)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.data is not None:
            self.data.close()
            self.data = None
        self.file.close()

    def span_hash(self, name):
        offset, size = self.spans[name]
        view = memoryview(self.data)
        try:
            return hashlib.sha256(view[offset:offset + size]).digest()
        finally:
            view.release()


def compare(a, b):
    compatible = (a.layer == b.layer and a.params == b.params and a.geometry == b.geometry and
                  a.descriptor == b.descriptor)
    spans = {}
    for name in a.spans:
        size_a, size_b = a.spans[name][1], b.spans[name][1]
        same = size_a == size_b and a.span_hash(name) == b.span_hash(name)
        spans[name] = {'equal': same, 'bytes_a': size_a, 'bytes_b': size_b}
    physical = all(spans[n]['equal'] for n in (
        'k_records', 'k_stage', 'k_tail', 'v_records', 'v_stage', 'v_tail'))
    return {'schema': 'rkva-mq-capture-comparison-v1',
            'layer_a': a.layer, 'layer_b': b.layer,
            'kernel_a': a.kernel, 'kernel_b': b.kernel,
            'remote_layer_constants_geometry_descriptor_equal': compatible,
            'all_physical_inputs_identical': bool(
                compatible and physical and all(spans[n]['equal'] for n in
                ('query', 'qrot', 'positions'))),
            'kv_consumed_status': ('identical_physical_planes' if physical and compatible
                                   else 'inconclusive_unproven_consumed_spans'),
            'spans': spans}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first')
    parser.add_argument('second')
    args = parser.parse_args()
    a = None
    b = None
    try:
        a = Capture(args.first)
        b = Capture(args.second)
        print(json.dumps(compare(a, b), indent=2))
    except (OSError, ValueError, struct.error) as error:
        parser.exit(1, 'invalid capture: ' + str(error) + '\n')
    finally:
        if a is not None:
            a.close()
        if b is not None:
            b.close()


if __name__ == '__main__':
    main()
