#!/usr/bin/env python3
"""Per-block DXIL instruction accounting for the RKVA KVarN4 MQ tile kernels.

Usage: rkva_dxil_block_mix.py file.txt [...]

Prints, for the entry function, the instruction mix of the fast shared-tile
loop body grouped by class, and a derived per-row budget (8 KV rows per tile).
DXIL is pre-ISA: `dot4` becomes 4 FMA, `waveActiveOp` becomes a 64-lane
butterfly (6 shuffle + 6 add on RDNA2), `legacyF16ToF32` and `unary` map to
conversion/SFU instructions.
"""
import collections
import pathlib
import re
import sys

OPS = set('''add fadd sub fsub mul fmul fma fdiv udiv sdiv urem srem shl lshr ashr and or xor
icmp fcmp select br ret phi call unreachable load store getelementptr zext sext trunc fptrunc
fpext bitcast extractvalue insertvalue alloca sitofp uitofp fptosi fptoui frem cmpxchg'''.split())


def opcode(line):
    text = line.split(';')[0].strip()
    if not text:
        return None
    if 'call ' in text:
        m = re.search(r'@(dx\.op\.[A-Za-z0-9.]+)', text)
        if m:
            return m.group(1)
        m = re.search(r'call [^@]*@(\w+)', text)
        return 'call:' + (m.group(1) if m else '?')
    tokens = text.split()
    for token in tokens:
        if token in OPS:
            return token
    return tokens[-1] if tokens else None


def blocks(path):
    lines = pathlib.Path(path).read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('define void @CSRangeAttnKvarn4Mq'))
    out, cur = [], None
    for l in lines[start:]:
        m = re.match(r'; <label>:(\S+)\s+; preds = (.*)', l)
        if m:
            cur = {'id': m.group(1), 'preds': m.group(2).strip(), 'lines': []}
            out.append(cur)
            continue
        if l.startswith('}'):
            break
        if cur is not None and l.strip():
            cur['lines'].append(l.strip())
    for b in out:
        b['ops'] = collections.Counter(x for x in map(opcode, b['lines']) if x)
        b['n'] = sum(b['ops'].values())
    return out


def classify(ops):
    """Map DXIL ops to RDNA2-class cost in wave64 issue slots."""
    c = collections.Counter()
    c['fma_class'] += ops['fmul'] + ops['fadd'] + ops['fsub'] + ops['fma'] + 4 * ops['dx.op.dot4.f32']
    c['reduce_add'] += 6 * ops['dx.op.waveActiveOp.f32']
    c['reduce_shfl'] += 6 * ops['dx.op.waveActiveOp.f32']
    c['sfu'] += ops['dx.op.unary.f32']            # exp2 / rcp
    c['cvt'] += ops['dx.op.legacyF16ToF32'] + ops['uitofp'] + ops['sitofp']
    c['lds'] += ops['barrier']
    c['load'] += ops['dx.op.bufferLoad.i32']
    c['store'] += ops['dx.op.bufferStore.i32']
    c['int'] = (ops['add'] + ops['mul'] + ops['and'] + ops['or'] + ops['xor'] + ops['shl'] +
                ops['lshr'] + ops['ashr'] + ops['icmp'] + ops['select'] + ops['getelementptr'] +
                ops['zext'] + ops['sext'] + ops['trunc'] + ops['extractvalue'] + ops['phi'])
    c['br'] = ops['br'] + ops['fcmp']
    c['cbuf'] = ops['dx.op.cbufferLoadLegacy.i32'] + ops['dx.op.cbufferLoadLegacy.f32']
    return c


def main():
    for path in sys.argv[1:]:
        bs = blocks(path)
        tile = [b for b in bs if b['ops'].get('dx.op.dot4.f32', 0) >= 4 and b['n'] > 20]
        small = [b for b in bs if b['ops'].get('dx.op.dot4.f32', 0) > 0 and b['n'] <= 20]
        total = collections.Counter()
        for b in tile:
            total += classify(b['ops'])
        print('==', path)
        print('  blocks(labels)=%d  tile-head-blocks=%d  residual/tail-head-blocks=%d'
              % (len(bs), len(tile), len(small)))
        print('  barriers(static, entry fn)=%d' % sum(b['ops'].get('dx.op.barrier', 0) for b in bs))
        print('  tile-loop instruction classes (one 8-row tile, both halves, all QG):')
        for k, v in sorted(total.items()):
            print('    %-11s %6d' % (k, v))
        # every tile-head block covers RKVA_HALF=4 rows of one QG; 4 dots per block
        dots = sum(b['ops'].get('dx.op.dot4.f32', 0) for b in tile)
        rows = dots / 12.0 if dots else 1.0
        print('  dots/tile=%d  implied rows/tile=%.1f' % (dots, rows))
        print('  per-row classes:')
        for k, v in sorted(total.items()):
            print('    %-11s %8.2f' % (k, v / rows))
        fma = total['fma_class'] / rows
        useful = 2.0 * (96.0 / 8.0)  # 12 QG x 8 rows x 4 dims / 8 rows = 48 QK + 48 PV lane-FMA
        print('    useful-fma  %8.2f (%.1f%% of fma_class)' % (useful, 100.0 * useful / fma))
        print()


if __name__ == '__main__':
    main()
