#!/usr/bin/env python3
"""Execution-weighted DXIL record-loop comparison, wave64 (not ISA timing).

Usage: rkva_decode_rowlane_cost.py baseline.txt rowlane.txt
Counts both sides of data-dependent branches, like rkva_dxil_loop_cost.py.
Rowlane: 2 dimension loops x32, 8-row PV loop x8, rest once per 64 rows/wave.
Legacy: one unrolled 8-row tile. Includes legacy group metadata each tile,
so baseline is an upper bound; also prints amortization-adjusted baseline.
All SSA ops count one except dot4=4 and wave reduction=12. Unlike the older
counter, LDS loads/stores and explicit FMad are included. PHI/GEP/extract
are DXIL bookkeeping, not necessarily issued ISA: do not infer latency.
"""
import collections
import json
import sys
import rkva_dxil_loop_cost as lc


def loops(path):
    order, blocks, succ = lc.parse(path, 'CSRangeAttnKvarn4Decode')
    preds = {}
    for a, outs in succ.items():
        for b in outs:
            preds.setdefault(b, []).append(a)
    dom = lc.dominators(order, blocks, succ, preds)
    bodies = {}
    for latch, outs in succ.items():
        for head in outs:
            if head not in dom.get(latch, set()):
                continue
            body = {head}
            work = [latch] if latch != head else []
            while work:
                b = work.pop()
                if b in body:
                    continue
                body.add(b)
                work.extend(p for p in preds.get(b, []) if head in dom.get(p, set()))
            bodies.setdefault(head, set()).update(body)
    counts = {b: collections.Counter(x for x in data['ops'] if x)
              for b, data in blocks.items()}
    return bodies, counts


def weighted(c):
    return sum(c.values()) + 3*c['dx.op.dot4.f32'] + 11*c['dx.op.waveActiveOp.f32']


def analyze(path, rowlane):
    bodies, counts = loops(path)
    if rowlane:
        dims = [h for h, body in bodies.items() if len(body) == 1 and
                counts[h]['dx.op.tertiary.f32'] == 24]
        assert len(dims) == 2, dims
        candidates = [h for h, body in bodies.items() if all(d in body for d in dims)]
        outer = min(candidates, key=lambda h: len(bodies[h]))
        nested = [h for h in bodies if h != outer and bodies[h] < bodies[outer]]
        pv = [h for h in nested if h not in dims]
        assert len(pv) == 1 and len(bodies[pv[0]]) == 1, pv
        factors = {h: 32 if h in dims else 8 for h in nested}
        rows = 64
    else:
        outer = max(bodies, key=lambda h: sum(counts[b]['dx.op.dot4.f32'] for b in bodies[h]))
        assert sum(counts[b]['dx.op.dot4.f32'] for b in bodies[outer]) == 48
        factors = {}
        rows = 8
    total = collections.Counter()
    for b in bodies[outer]:
        factor = factors.get(b, 1)
        total.update({k: v*factor for k, v in counts[b].items()})
    result = dict(file=path, loop=outer, rows_per_wave_iteration=rows,
                  inner_iterations=factors, weighted_dxil_per_row=weighted(total)/rows,
                  reductions_per_row=total['dx.op.waveActiveOp.f32']/rows,
                  lds_loads_per_row=total['load']/rows,
                  buffer_loads_per_row=total['dx.op.bufferLoad.i32']/rows,
                  barriers_per_iteration=total['dx.op.barrier'],
                  instruction_mix_per_iteration=dict(total))
    if not rowlane:
        # Metadata branch = loads + F16 conversions, no dot4. Happens once per
        # 8 tiles/wave (128 rows/group, two waves). The loop header also carries
        # state PHIs so only subtract the body block with conversion operations.
        meta = [b for b in bodies[outer] if counts[b]['dx.op.legacyF16ToF32'] and
                not counts[b]['dx.op.dot4.f32'] and not counts[b]['dx.op.waveActiveOp.f32'] and
                not counts[b]['fmul']]
        result['metadata_blocks'] = meta
        result['amortized_weighted_dxil_per_row'] = (
            weighted(total) - sum(weighted(counts[b])*7/8 for b in meta))/rows
    return result


if __name__ == '__main__':
    old, new = analyze(sys.argv[1], False), analyze(sys.argv[2], True)
    print(json.dumps(dict(baseline=old, rowlane=new,
                          ratio=old['weighted_dxil_per_row']/new['weighted_dxil_per_row'],
                          ratio_amortized=old['amortized_weighted_dxil_per_row']/
                          new['weighted_dxil_per_row']), indent=2, sort_keys=True))
