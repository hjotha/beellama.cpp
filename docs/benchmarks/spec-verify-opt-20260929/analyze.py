#!/usr/bin/env python3
"""Paired tok/s analyzer for spec-verify-opt batteries.

Usage: analyze.py <result-dir> <baseline-mode> <candidate-mode> [<candidate-mode> ...]

For each case (curto192/codigo192/prosa192/long192) and each repeat index, pairs
the candidate's decode_tps against the baseline's at the same repeat, reports the
median paired percent delta and the dispersion, plus per-mode medians. Also reports
token-equality (from token-comparisons.jsonl) so a speed claim is never separated
from a correctness claim.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


def load_results(root):
    return json.loads((root / 'results.json').read_text())


def main():
    root = Path(sys.argv[1])
    baseline = sys.argv[2]
    candidates = sys.argv[3:]
    results = load_results(root)

    # index: (mode, case, repeat) -> decode_tps
    idx = {}
    for r in results:
        if r.get('warmup'):
            continue
        idx[(r['mode'], r['case'], r['repeat'])] = r

    cases = sorted({r['case'] for r in results if not r.get('warmup')})
    repeats = sorted({r['repeat'] for r in results if not r.get('warmup')})

    # token equality summary
    tcmp = root / 'token-comparisons.jsonl'
    divs = []
    if tcmp.exists():
        for line in tcmp.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                if not c.get('equal'):
                    divs.append(c)

    print(f'== {root.name}: baseline={baseline} candidates={candidates} ==')
    print(f'token divergences (across all comparisons): {len(divs)}')
    for d in divs[:10]:
        print('  DIV', d.get('kind'), d.get('case'), d.get('reference_mode'),
              'vs', d.get('actual_mode'), 'first_diff', d.get('first_difference'))

    for cand in candidates:
        print(f'\n--- {cand} vs {baseline} ---')
        all_deltas = []
        for case in cases:
            deltas = []
            base_vals = []
            cand_vals = []
            for rep in repeats:
                b = idx.get((baseline, case, rep))
                c = idx.get((cand, case, rep))
                if b and c:
                    bt, ct = b['decode_tps'], c['decode_tps']
                    base_vals.append(bt)
                    cand_vals.append(ct)
                    deltas.append(100.0 * (ct - bt) / bt)
            if deltas:
                all_deltas += deltas
                print(f'  {case:11s} n={len(deltas)} base_med={median(base_vals):7.3f} '
                      f'cand_med={median(cand_vals):7.3f} '
                      f'paired_delta_med={median(deltas):+6.2f}% '
                      f'min={min(deltas):+6.2f}% max={max(deltas):+6.2f}%')
        if all_deltas:
            print(f'  {"ALL":11s} n={len(all_deltas)} paired_delta_med={median(all_deltas):+6.2f}% '
                  f'min={min(all_deltas):+6.2f}% max={max(all_deltas):+6.2f}%')

    # per-mode overall medians
    print('\n-- per-mode median decode_tps by case --')
    modes = sorted({r['mode'] for r in results if not r.get('warmup')})
    header = 'mode          ' + ''.join(f'{c:>12s}' for c in cases)
    print(header)
    for m in modes:
        row = f'{m:13s}'
        for case in cases:
            vals = [r['decode_tps'] for r in results if r['mode'] == m and r['case'] == case and not r.get('warmup')]
            row += f'{median(vals):12.3f}' if vals else f'{"-":>12s}'
        print(row)


if __name__ == '__main__':
    main()
