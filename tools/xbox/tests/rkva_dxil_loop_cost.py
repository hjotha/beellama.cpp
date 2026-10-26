#!/usr/bin/env python3
"""Loop-body instruction accounting for the RKVA KVarN4 MQ tile entry function.

Usage: rkva_dxil_loop_cost.py file.txt [file.txt ...]

Produce the input with the Windows DXC (cs_6_0) that builds the RKVA shaders:
  dxc -T cs_6_0 -E CSRangeAttnKvarn4Mq -D RKVA_MQ_QT=2 -Fc out.txt shader.hlsl -Fo out.dxil
The DXIL text is pre-ISA; see the class notes below for the wave64 cost model.

Parses the DXIL text dump, finds the natural loop that owns the 96 `dot4` calls
(the 8-row fast shared-tile loop of CSRangeAttnKvarn4Mq), and reports the
execution-weighted instruction mix of ONE loop iteration, i.e. one 8-row tile
covering RKVA_MQ_QT=2 queries x RKVA_GQA=6 heads. Blocks that are only reachable
through a runtime-constant branch arm must be measured with a forced build.

Class costs are RDNA2 wave64 issue slots: `dot4` is four FMA, `waveActiveOp` is
a 64-lane butterfly (six cross-lane moves + six adds).
"""
import collections
import pathlib
import re
import sys

DP4A = 'dx.op.dot4AddPacked.i32'  # dot4add_i8packed / dot4add_u8packed
OPS = set('''add fadd sub fsub mul fmul fma fdiv udiv sdiv urem srem shl lshr ashr and or xor
icmp fcmp select br ret phi call unreachable load store getelementptr zext sext trunc fptrunc
fpext bitcast extractvalue insertvalue alloca sitofp uitofp fptosi fptoui frem'''.split())


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
    for token in text.split():
        if token in OPS:
            return token
    tokens = text.split()
    return tokens[-1] if tokens else None


def parse(path):
    lines = pathlib.Path(path).read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('define void @CSRangeAttnKvarn4Mq'))
    order, blocks = [], {}
    cur = None
    for l in lines[start:]:
        m = re.match(r'; <label>:(\S+)\s+; preds = (.*)', l)
        if m:
            cur = m.group(1)
            order.append(cur)
            blocks[cur] = {'ops': [], 'term': None}
            continue
        if l.startswith('}'):
            break
        if cur is None or not l.strip():
            continue
        s = l.strip()
        blocks[cur]['ops'].append(opcode(s))
        if s.startswith('br ') or s.startswith('ret ') or s.startswith('unreachable'):
            blocks[cur]['term'] = s
    succ = {}
    for label, b in blocks.items():
        term = b['term'] or ''
        m = re.match(r'br i1 .*label %(\S+), label %(\S+)', term)
        if m:
            succ[label] = [m.group(1).rstrip(','), m.group(2)]
        else:
            m = re.match(r'br label %(\S+)', term)
            succ[label] = [m.group(1)] if m else []
    return order, blocks, succ


def dominators(order, blocks, succ, preds):
    dom = {label: set(order) for label in order}
    dom[order[0]] = {order[0]}
    changed = True
    while changed:
        changed = False
        for label in order:
            if label == order[0]:
                continue
            ps = [p for p in preds.get(label, []) if p in blocks]
            new = set(order) if not ps else set.intersection(*[dom[p] for p in ps])
            new = new | {label}
            if new != dom[label]:
                dom[label] = new
                changed = True
    return dom


def reverse_reachable(succ, blocks, header):
    """Nodes that can reach `header` (a natural loop only contains those)."""
    rev = {}
    for label, outs in succ.items():
        for o in outs:
            rev.setdefault(o, []).append(label)
    seen = {header}
    work = [header]
    while work:
        cur = work.pop()
        for prev in rev.get(cur, []):
            if prev not in seen and prev in blocks:
                seen.add(prev)
                work.append(prev)
    return seen


def natural_loop(order, blocks, succ, header):
    """Natural loop: header + every successor-reachable node that reaches header."""
    reaches = reverse_reachable(succ, blocks, header)
    body = {header}
    work = [header]
    while work:
        cur = work.pop()
        for nxt in succ.get(cur, []):
            if nxt not in body and nxt in blocks and order.index(nxt) > order.index(header) \
                    and nxt in reaches:
                body.add(nxt)
                work.append(nxt)
    return body


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith('--')]
    rows_override = None
    verbose = '--verbose' in sys.argv[1:]
    for a in sys.argv[1:]:
        if a.startswith('--rows='):
            rows_override = float(a.split('=', 1)[1])
    for path in paths:
        order, blocks, succ = parse(path)
        # back edges: successor appears earlier and points at a loop header
        headers = set()
        for label, outs in succ.items():
            for o in outs:
                if o in blocks and order.index(o) <= order.index(label):
                    headers.add(o)
        preds = {}
        for label, outs in succ.items():
            for o in outs:
                preds.setdefault(o, []).append(label)
        dom = dominators(order, blocks, succ, preds)
        best = None
        for h in headers:
            body = {b for b in natural_loop(order, blocks, succ, h) if h in dom.get(b, set())}
            # A score costs either one dot4.f32 (4 F32 MACs) or a chain of
            # dot4add_i8packed (4 int8 MACs each); both are counted as "dot units"
            # so the same 12-units-per-row baseline works for both kernels.
            dots = sum(blocks[b]['ops'].count('dx.op.dot4.f32') +
                       blocks[b]['ops'].count(DP4A) for b in body)
            # Steady state: among the loops that cover the whole 8-row tile, take
            # the smallest body (the innermost loop), not an enclosing one.
            key = (dots, -len(body))
            if best is None or key > best[0]:
                best = (key, h, dots, body)
        _, header, dots, body = best
        counters = collections.Counter()
        for b in body:
            for op in blocks[b]['ops']:
                if op:
                    counters[op] += 1
        rows = rows_override if rows_override else (dots / 12.0 if dots else 1.0)
        dp4a = counters[DP4A]
        # One dot unit is 4 MACs, hence 4 issue slots of FMA work in the F32
        # model; an i8 dot unit is one op, but it still represents 4 MACs.
        fma_class = (counters['fmul'] + counters['fadd'] + counters['fsub'] + counters['fma'] +
                     4 * counters['dx.op.dot4.f32'] + dp4a)
        # waveActiveOp is a full-wave butterfly: six cross-lane moves + six adds
        # in the wave64 model. waveReadLaneAt is a single permute (its add is a
        # separately counted `fadd`), so it costs one slot.
        reduce_add = 6 * counters['dx.op.waveActiveOp.f32']
        reduce_shfl = (6 * counters['dx.op.waveActiveOp.f32'] +
                       counters['dx.op.waveReadLaneAt.f32'])
        sfu = counters['dx.op.unary.f32']
        cvt = counters['dx.op.legacyF16ToF32'] + counters['uitofp'] + counters['sitofp'] + \
            counters['fptosi']
        loads = counters['dx.op.bufferLoad.i32'] + counters['dx.op.rawBufferLoad.i32']
        stores = counters['dx.op.bufferStore.i32'] + counters['dx.op.rawBufferStore.i32']
        barriers = counters['dx.op.barrier']
        ints = sum(counters[o] for o in ('add', 'mul', 'and', 'or', 'xor', 'shl', 'lshr', 'ashr',
                                        'icmp', 'select', 'getelementptr', 'zext', 'sext', 'trunc',
                                        'extractvalue', 'phi', 'bitcast', 'fptoui'))
        brs = counters['br'] + counters['fcmp']
        cbuf = counters['dx.op.cbufferLoadLegacy.i32'] + counters['dx.op.cbufferLoadLegacy.f32']
        total = (fma_class + reduce_add + reduce_shfl + sfu + cvt + loads + stores + barriers +
                 ints + brs + cbuf)
        print('==', pathlib.Path(path).name)
        print('   loop header %%%s  body blocks %d  dot-units/iter %d (dot4 %d, dp4a %d)  '
              'rows/iter %.0f' % (header, len(body), dots, counters['dx.op.dot4.f32'], dp4a, rows))
        print('   per-tile (8 rows, 2 queries x 6 heads):')
        print('     fma_class   %6d   reduce_add %6d   reduce_shfl %6d   sfu %5d'
              % (fma_class, reduce_add, reduce_shfl, sfu))
        print('     cvt         %6d   loads      %6d   stores      %6d   barrier %5d'
              % (cvt, loads, stores, barriers))
        print('     int/misc    %6d   branch     %6d   cbuf        %6d   TOTAL  %6d'
              % (ints, brs, cbuf, total))
        if verbose:
            for k, v in counters.most_common(14):
                print('       %6d %s' % (v, k))
        print('   per KV row:  TOTAL %.1f   fma_class %.1f   reduce %.1f   sfu %.1f   int %.1f'
              % (total / rows, fma_class / rows, (reduce_add + reduce_shfl) / rows, sfu / rows,
                 (ints + brs) / rows))
        useful = 96.0  # 12 QG x 4 dims QK + 12 QG x 4 dims PV per lane per row
        print('   useful FMA per row (QK+PV)  %.1f  => %.1f%% of fma_class, %.1f%% of TOTAL'
              % (useful, 100.0 * useful / (fma_class / rows), 100.0 * useful / (total / rows)))
        print()


if __name__ == '__main__':
    main()
