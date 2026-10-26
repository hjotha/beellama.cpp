#!/usr/bin/env python3
"""Fail closed if ANY broadcast index depends on a lane or memory value.

For this shader, all indices must be integer constants / arithmetic / induction
PHIs. A PHI's incoming values are checked recursively, including the backedge;
block labels are not values. This is a syntactic uniformity proof for these
broadcasts, not a general DXIL validator. DXC performs shader validation.
"""
import pathlib
import re
import sys


def audit(text):
    defs = dict(re.findall(r'^\s*(%\d+) = (.+)$', text, re.M))

    def uniform(value, seen):
        if re.fullmatch(r'\d+', value):
            return True
        if value in seen:
            return True  # induction recurrence; its other incoming values still checked
        expr = defs.get(value, '')
        if expr.startswith('phi i32 '):
            inputs = re.findall(r'\[\s*(%\d+|\d+),\s*%\d+\s*\]', expr)
        elif re.match(r'(add|sub|mul|shl|lshr|and|or)\b', expr):
            inputs = re.findall(r'%\d+', expr.split(';')[0])
        else:
            return False
        return bool(inputs) and all(uniform(v, seen | {value}) for v in inputs)

    indices = re.findall(r'call float @dx.op.waveReadLaneAt.f32\([^\n]+, i32 (%\d+|\d+)\)', text)
    assert indices, 'no broadcasts found'
    for index in indices:
        assert uniform(index, set()), f'nonuniform or unproved WaveReadLaneAt index: {index}'
    assert not re.search(r'\b(?:fadd|fmul|fsub|fdiv|phi) (?:fast )?half\b', text), 'native FP16'
    return len(indices), len(set(indices))


if __name__ == '__main__':
    text = pathlib.Path(sys.argv[1]).read_text()
    n, unique = audit(text)
    # Negative check: replacing an induction index by lane ID must be rejected.
    poisoned = text
    match = re.search(r'call float @dx.op.waveReadLaneAt.f32\([^\n]+, i32 (%\d+)\)', poisoned)
    assert match
    idx = match.group(1)
    poisoned = re.sub(r'^\s*' + re.escape(idx) + r' = .+$',
                      idx + ' = call i32 @dx.op.waveGetLaneIndex(i32 111)', poisoned, flags=re.M)
    try:
        audit(poisoned)
    except AssertionError:
        pass
    else:
        raise AssertionError('audit failed its lane-dependent negative control')
    print(f'PASS {n} broadcasts, {unique} uniform index expressions; lane-dependent negative rejected')
