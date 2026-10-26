#!/usr/bin/env python3
"""Execute the compiled INT8 fast-record score path on CPU, with raw buffers.

Usage: python3 tools/xbox/tests/rkva_mq_i8_dxil_model.py shader.txt --require-uniform-reads
The input is DXC -Fc assembly of CSRangeAttnKvarn4Mq (MQ2, cs_6_4).

This deliberately interprets SSA instructions rather than transcribing the
HLSL algebra. It covers buffer addresses, packed signed dot4, F32 operations,
branches, phis, slice/row reductions and score broadcasts. It stops at the
first residual/tail F32 dot4; PV, final merge and GPU driver lowering are NOT
validated. Unsupported instructions and divergent branches fail explicitly.
WavePrefixSum uses a legal CPU scan order; comparisons allow F32 rounding.

Fixtures use the serial-prefix probe's canonical record encoder; the console
probe and its oracle remain unchanged. No sockets or GPU access.
"""
import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np

F = np.float32
I = np.int64
U = np.uint32


class VM:
    """Small, fail-closed interpreter for this DXC score path, not general DXIL."""

    def __init__(self, path, cb, buf, tid, group, require_uniform_reads=False):
        self.reg = {}
        self.buf = buf
        self.cb = np.asarray(cb, U)
        self.tid = tid
        self.group = group
        self.require_uniform_reads = require_uniform_reads
        self.rows = []
        self.prefix_ids = []
        self.prefix_scores = []
        src = Path(path).read_text().split('define void @CSRangeAttnKvarn4Mq() {')[1].split('\n}')[0]
        self.blocks = {0: []}
        b = 0
        for l in src.splitlines():
            m = re.match('; <label>:(\\d+)', l)
            if m:
                b = int(m[1])
                self.blocks[b] = []
            elif l.strip():
                self.blocks[b].append(l.strip().split('  ;')[0])

    def val(self, s):
        s = s.strip()
        if s.startswith('%'):
            return self.reg[int(s[1:])]
        if s == 'undef':
            return None
        if s in ('true', 'false'):
            return s == 'true'
        if s.startswith('0x'):
            return F(struct.unpack('>d', bytes.fromhex(s[2:]))[0])
        if any((c in s for c in '.eE')):
            return F(s)
        return I(s)

    def signed(self, x):
        return np.asarray(x, U).view(np.int32).astype(I)

    def run(self):
        b = 0
        prev = -1
        iters = 0
        while True:
            iters += 1
            if iters > 10000:
                raise RuntimeError('loop')
            regs = {}
            for l in self.blocks[b]:
                if ' = phi ' not in l:
                    continue
                dest, expr = l.split(' = ', 1)
                match = next((v for v, p in re.findall('\\[ ([^,]+), %(\\d+) \\]', expr) if int(p) == prev))
                regs[int(dest[1:])] = self.val(match)
            self.reg.update(regs)
            for l in self.blocks[b]:
                if ' = phi ' in l:
                    continue
                if l.startswith('br label'):
                    prev, b = (b, int(re.search('%(\\d+)', l)[1]))
                    break
                if l.startswith('br i1'):
                    cond, yes, no = re.match('br i1 ([^,]+), label %(\\d+), label %(\\d+)', l).groups()
                    c = self.val(cond)
                    if np.any(c) != np.all(c):
                        raise RuntimeError('divergence ' + l)
                    prev, b = (b, int(yes if np.all(c) else no))
                    break
                if l == 'ret void':
                    return
                if not ' = ' in l:
                    raise RuntimeError(l)
                dest, expr = l.split(' = ', 1)
                dest = int(dest[1:])
                v = None
                op = expr.split()[0]
                if op == 'call':
                    m = re.search('@dx.op.(\\w+)(?:\\.[\\w]+)?\\((.*)\\)', expr)
                    if not m:
                        raise RuntimeError(expr)
                    name, args = m.groups()
                    args = [self.val(x.split()[-1]) for x in args.split(',')]
                    opc = args[0]
                    if name == 'createHandle':
                        v = args[3]
                    elif name == 'flattenedThreadIdInGroup':
                        v = self.tid
                    elif name == 'groupId':
                        v = self.group[int(args[1])]
                    elif name == 'waveGetLaneCount':
                        v = 64
                    elif name == 'waveGetLaneIndex':
                        v = np.arange(64, dtype=I)
                    elif name == 'cbufferLoadLegacy':
                        v = self.cb[int(args[2]) * 4:int(args[2]) * 4 + 4]
                        if '.f32(' in expr:
                            v = v.view(F)
                    elif name == 'rawBufferLoad':
                        handle, offset = (args[1], np.asarray(args[2], I))
                        mem = self.buf[int(handle)]
                        mask = int(args[4])
                        assert np.all(offset >= 0) and np.all(offset % 4 == 0)
                        v = tuple((mem[offset // 4 + i].astype(I) if mask & 1 << i else 0 for i in range(4)))
                    elif name == 'legacyF16ToF32':
                        v = np.asarray(args[1], np.uint16).view(np.float16).astype(F)
                    elif name == 'waveReadLaneAt':
                        if self.require_uniform_reads:
                            assert np.ndim(args[2]) == 0, 'varying-index WaveReadLaneAt'
                        a = np.broadcast_to(args[1], (64,))
                        v = a[args[2]]
                        if np.ndim(args[2]) == 0 and int(args[2]) in (15, 31, 47, 63):
                            assert int(args[2]) == (15, 31, 47, 63)[len(self.prefix_ids)]
                            self.prefix_ids.append(dest)
                            if len(self.prefix_ids) == 1:
                                self.prefix_scores.append(float(v))
                    elif name == 'wavePrefixOp':
                        assert args[2] == 0, 'only prefix sum is supported'
                        a = np.broadcast_to(args[1], (64,))
                        # Legal scan tree; unlike a sequential cumsum this also
                        # exercises the cancellation from differently rounded
                        # neighboring prefixes.
                        v = a.copy()
                        for step in (1, 2, 4, 8, 16, 32):
                            prev_scan = v.copy()
                            v[step:] = (v[step:] + prev_scan[:-step]).astype(F)
                        v = np.r_[F(0), v[:-1]]
                    elif name == 'dot4':
                        return
                    elif name == 'waveActiveOp':
                        a = np.broadcast_to(args[1], (64,))
                        if args[2] == 3:
                            v = F(a.max())
                        else:
                            raise RuntimeError('unsupported wave reduction ' + expr)
                    elif name == 'binary':
                        fn = {35: np.maximum, 36: np.minimum, 39: np.maximum, 40: np.minimum}[int(opc)]
                        v = fn(np.asarray(args[1], I) & 4294967295, np.asarray(args[2], I) & 4294967295) if opc in (39, 40) else fn(args[1], args[2])
                    elif name == 'unary':
                        fn = {6: np.abs, 26: np.rint, 21: np.exp2}[int(opc)]
                        v = fn(args[1]).astype(F)
                    elif name == 'dot4AddPacked':
                        assert opc == 163, 'expected signed int8 dot4'
                        v = self.signed(args[1])
                        for sh in (0, 8, 16, 24):
                            a = np.asarray(args[2], I) >> sh & 255
                            b8 = np.asarray(args[3], I) >> sh & 255
                            v = v + np.where(a >= 128, a - 256, a) * np.where(b8 >= 128, b8 - 256, b8)
                        v = self.signed(v)
                    else:
                        raise RuntimeError(name + ' ' + expr)
                elif op == 'extractvalue':
                    m = re.match('extractvalue .* (%\\d+), (\\d+)', expr)
                    v = self.val(m[1])[int(m[2])]
                elif op == 'bitcast':
                    m = re.match('bitcast (\\w+) (.*?) to (\\w+)', expr)
                    a = self.val(m[2])
                    v = np.asarray(a, U if m[1] == 'i32' else F).view(F if m[3] == 'float' else U)
                elif op in ('fptosi', 'sitofp', 'uitofp', 'zext'):
                    m = re.match('\\w+ \\w+ (.*?) to (\\w+)', expr)
                    a = self.val(m[1])
                    v = np.asarray(a, I) if op in ('fptosi', 'zext') else np.asarray(self.signed(a) if op == 'sitofp' else np.asarray(a, U), F)
                elif op == 'select':
                    m = re.match('select i1 ([^,]+), \\w+ ([^,]+), \\w+ (.*)', expr)
                    v = np.where(self.val(m[1]), self.val(m[2]), self.val(m[3]))
                elif op in ('icmp', 'fcmp'):
                    m = re.match('(?:icmp|fcmp) (?:fast )?(\\w+) \\w+ ([^,]+), (.*)', expr)
                    pred, a, bb = m.groups()
                    a = self.val(a)
                    bb = self.val(bb)
                    if pred.startswith('s'):
                        a = self.signed(a)
                        bb = self.signed(bb)
                    elif pred.startswith('u'):
                        a = np.asarray(a, I) & 0xffffffff
                        bb = np.asarray(bb, I) & 0xffffffff
                    fn = {'eq': np.equal, 'ne': np.not_equal, 'gt': np.greater, 'ge': np.greater_equal, 'lt': np.less, 'le': np.less_equal}[pred[-2:]]
                    v = fn(a, bb)
                elif op in ('add', 'sub', 'mul', 'shl', 'lshr', 'and', 'or', 'xor', 'fadd', 'fsub', 'fmul', 'fdiv'):
                    m = re.match('\\w+ (?:(?:fast|nuw|nsw) )*(?:i1|i32|float) ([^,]+), (.*)', expr)
                    a, bb = map(self.val, m.groups())
                    fn = {'add': np.add, 'sub': np.subtract, 'mul': np.multiply, 'shl': np.left_shift, 'lshr': np.right_shift, 'and': np.bitwise_and, 'or': np.bitwise_or, 'xor': np.bitwise_xor, 'fadd': np.add, 'fsub': np.subtract, 'fmul': np.multiply, 'fdiv': np.divide}[op]
                    v = fn(a, bb)
                    v = np.asarray(v, F) if op.startswith('f') else np.asarray(v, I) & 4294967295
                    if op == 'fsub' and len(self.prefix_ids) == 4:
                        # Capture the actual DXIL subtractions, not a Python
                        # reconstruction that would hide a wrong score formula.
                        i = len(self.prefix_scores)
                        assert m.groups() == (f'%{self.prefix_ids[i]}', f'%{self.prefix_ids[i-1]}')
                        self.prefix_scores.append(float(v))
                        if len(self.prefix_scores) == 4:
                            self.rows.append(self.prefix_scores)
                            self.prefix_ids = []
                            self.prefix_scores = []
                else:
                    raise RuntimeError(expr)
                self.reg[dest] = v
            else:
                raise RuntimeError('fallthrough')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('assembly', type=Path)
    ap.add_argument('--require-uniform-reads', action='store_true')
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import rkva_kvarn_serial_prefix_model as sp
    import rkva_mq_i8_quality as Q

    rng = np.random.default_rng(42)
    worst = 0.0
    compared = 0
    for live in (256, 384):
        canon = sp.Canon(rng, live)
        canon.append(live)
        q = rng.normal(size=(2, 24, 256)).astype(F)
        qr = Q.butterfly_wht(q)
        n_groups = live // 128
        records = [b''.join(b''.join(canon.record(g, value)) for g in range(n_groups))
                   for value in (False, True)]
        # Only the fast path is executed. Zero padding lets residual loads run
        # up to the explicit stop at their first F32 dot4 instruction.
        padding = bytes((384 + 256) * 4 * 256 * 2)
        buffers = {
            0: np.concatenate([q.ravel(), qr.ravel()]).view(U),
            1: np.frombuffer(records[0] + padding, U),
            2: np.frombuffer(records[1] + padding, U),
            3: np.array([sp.ORIGIN + live - 2, sp.ORIGIN + live - 1], U),
        }
        cb = [2, live, 24, 4, 256, F(1 / 16).view(U), sp.ORIGIN,
              1, live, 1, len(records[0]), 128]
        fast_hi = min((live - 1) & ~127, live - 1 - 128) & ~7
        for kv in (0, 3):
            planes = [[Q.record_planes(b) for b in canon.record(g, False)[2*kv:2*kv+2]]
                      for g in range(n_groups)]
            class Store:
                groups = n_groups
                tail_begin = n_groups * 128
                kplanes = planes
            expected = {}
            for j in range(2):
                for g in range(6):
                    query = qr[j, 6 * kv + g]
                    expected[j, g] = Q.i8_scores(query, Store(), True)
            for wave in (0, 1):
                vm = VM(args.assembly, cb, buffers, np.arange(64, dtype=I) + wave * 64,
                        (kv, 0, 0), args.require_uniform_reads)
                vm.run()
                rows = np.array(vm.rows)
                tiles = list(range(wave * 8, fast_hi, 16))
                assert rows.shape == (len(tiles) * 24, 4), rows.shape
                assert not vm.prefix_ids, 'incomplete prefix score reduction'
                error = 0.0
                for tile, t0 in enumerate(tiles):
                    for half in (0, 1):
                        for j in (0, 1):
                            for g in range(6):
                                got = rows[tile * 24 + half * 12 + j * 6 + g]
                                begin = t0 + half * 4
                                want = expected[j, g][begin:begin + 4]
                                np.testing.assert_allclose(got, want, rtol=2e-5, atol=3e-5)
                                error = max(error, float(np.max(abs(want - got))))
                                compared += 4
                worst = max(worst, error)
                print(f'live={live} kv={kv} wave={wave} fast={fast_hi} max_score_error={error:.3g}')
    print(f'rkva_mq_i8_dxil_model: {compared} scores, max error {worst:.3g}, PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
