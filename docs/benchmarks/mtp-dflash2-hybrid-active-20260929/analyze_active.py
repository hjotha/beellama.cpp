"""Audit active handoffs against authoritative target decisions and measure TPS."""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import re
import statistics


module = importlib.util.spec_from_file_location('dense_analysis', Path(__file__).resolve().parent.parent /
                                              'mtp-dflash2-suffix-dense-20260929/analyze.py')
dense = importlib.util.module_from_spec(module)
module.loader.exec_module(dense)


def active_events(text):
    result = []
    for m in re.finditer(r'SHADOWACTIVE (select|accept|cancel) ([^\n]*)', text):
        result.append(dict(event=m[1], **{k: int(v) for k, v in re.findall(r'(\w+)=(-?\d+)', m[2])}))
    return result


def audit_group(directory):
    plan = json.loads((directory / 'plan.json').read_text())
    text = (directory / 'server-output.txt').read_text(errors='replace')
    events, jobs = dense.parse_log(text)
    active = active_events(text)
    begins = [e for e in events if e['event'] == 'begin']
    records = sorted((json.loads(p.read_text()) for p in directory.glob('*.result.json')),
                     key=lambda r: r['request_index'])
    assert len(begins) == len(records) == len(plan['requests']), directory
    assert len({(e['request'], e['slot']) for e in begins}) == len(begins)
    result = []
    for begin, record in zip(begins, records):
        request = begin['request']
        assert begin['slot'] == 0
        trace = [e for e in events if e['request'] == request]
        ends = [e for e in trace if e['event'] == 'end']
        assert len(ends) == 1 and not ends[0]['truncated']
        singles = [e for e in trace if e['event'] == 'token']
        assert singles
        output = json.loads((directory / (record['scenario'] + '.tokens.json')).read_text())
        assert len(output) == ends[0]['generated'] == record['token_count']
        observed = {}
        for e in trace:
            if e['event'] == 'token':
                dense.put_token(observed, e['pos'], e['token'])
            elif e['event'] == 'commit':
                dense.put_token(observed, e['pos0'], e['anchor'])
                for i, token in enumerate(e['confirmed']):
                    dense.put_token(observed, e['pos0'] + 1 + i, token)
        assert all(observed.get(singles[0]['pos'] + i) == token for i, token in enumerate(output))
        drafts = {e['pos0']: e for e in trace if e['event'] == 'draft'}
        commits = {e['pos0']: e for e in trace if e['event'] == 'commit'}
        assert len(drafts) == sum(e['event'] == 'draft' for e in trace)
        assert len(commits) == sum(e['event'] == 'commit' for e in trace)
        selected = [e for e in active if e['request'] == request and e['event'] == 'select']
        assert len({e['job'] for e in selected}) == len(selected), 'An auxiliary job was consumed twice'
        row = dict(mode=plan['mode'], group=plan['group'], scenario=record['scenario'], case=record['case'],
                   repeat=record['repeat'], warmup=record['warmup'], request=request,
                   selected=len(selected), proposed=0, accepted=0, confirmed=0, cancelled=0,
                   zero=0, partial=0, full=0, resumed_primary=0, active_dispatch_us=0,
                   active_round_us=0, accepted_histogram=collections.Counter())
        selected_positions = {s['pos0'] for s in selected}
        for select in selected:
            source = [j for j in jobs if j['request'] == request and j['job'] == select['job']]
            assert len(source) == 1
            source = source[0]
            assert source['usable'] and source['ready'] and source['prefix_ok'] and source['position_ok']
            assert select['epoch'] == source['epoch']
            assert select['pos0'] == source['pos0'] + source['c']
            assert select['anchor'] == source['confirmed'][-1]
            draft = drafts[select['pos0']]
            assert draft['anchor'] == select['anchor']
            assert len(draft['proposed']) == select['n'] <= record['config']['verifier_max']
            assert draft['proposed'] == source['proposed'][source['c']:source['c'] + select['n']]
            assert len(draft['proposed']) >= record['config']['min_suffix']
            row['proposed'] += select['n']
            row['active_dispatch_us'] += draft['done_us'] - draft['start_us']
            finish = [e for e in active if e['request'] == request and e['job'] == select['job']
                      and e['event'] != 'select']
            assert len(finish) == 1, 'Every selected round must be verified or cancelled'
            finish = finish[0]
            if finish['event'] == 'cancel':
                row['cancelled'] += 1
                continue
            commit = commits[select['pos0']]
            matched, exact = dense.match_suffix(draft['proposed'], dict(enumerate(commit['confirmed'])), 0)
            assert exact and matched == finish['accepted']
            assert len(commit['confirmed']) == finish['confirmed']
            assert len(draft['proposed']) == finish['proposed']
            row['accepted'] += matched
            row['confirmed'] += finish['confirmed']
            row['accepted_histogram'][matched] += 1
            row['zero' if matched == 0 else 'full' if matched == select['n'] else 'partial'] += 1
            row['active_round_us'] += commit['done_us'] - draft['start_us']
            next_pos = select['pos0'] + len(commit['confirmed'])
            if next_pos in drafts and next_pos not in selected_positions:
                row['resumed_primary'] += 1
        for pos, draft in drafts.items():
            if pos not in selected_positions:
                assert len(draft['proposed']) <= record['config']['mtp_max']
        result.append(row)
    assert 'active shadow: primary MTP carry/KV' not in text
    assert 'active shadow: MTP carry failed' not in text
    accounting = re.findall(r'shadow active: selected=(\d+) proposed=(\d+) verified=(\d+) accepted=(\d+) cancelled=(\d+) pending=(\d+)', text)
    if accounting:
        selected, proposed, verified, accepted, cancelled, pending = map(int, accounting[-1])
        assert pending == 0 and selected == verified + cancelled
        assert selected == sum(r['selected'] for r in result)
        assert accepted == sum(r['accepted'] for r in result)
        assert proposed == sum(r['proposed'] for r in result)
    worker = re.findall(r'shadow auxiliary: [^\n]*errors=(\d+)', text)
    assert not worker or int(worker[-1]) == 0
    return result


def analyze(root):
    status = json.loads((root / 'status.json').read_text())
    assert status['state'] == 'complete' and not status['restoration_errors'], status
    rows = []
    for directory in sorted((root / 'groups').iterdir()):
        rows.extend(audit_group(directory))
    measured = [r for r in rows if not r['warmup']]
    results = json.loads((root / 'results.json').read_text())
    summary = []
    for mode in dict.fromkeys(r['mode'] for r in results):
        for case in ('curto192', 'codigo192', 'long192'):
            samples = [r for r in results if r['mode'] == mode and r['case'] == case]
            if not samples:
                continue
            audited = [r for r in measured if r['mode'] == mode and r['case'] == case]
            assert len(samples) == len(audited)
            rates = [r['decode_tps'] for r in samples]
            entry = dict(mode=mode, case=case, runs=len(samples), median_tps=statistics.median(rates),
                         min_tps=min(rates), max_tps=max(rates),
                         median_prefill_ms=statistics.median(r['prompt_ms'] for r in samples),
                         median_wall_s=statistics.median(r['wall_s'] for r in samples))
            for field in ('selected', 'proposed', 'accepted', 'confirmed', 'cancelled', 'zero', 'partial', 'full',
                          'resumed_primary', 'active_dispatch_us', 'active_round_us'):
                entry[field] = sum(r[field] for r in audited)
            entry['acceptance_pct'] = 100 * entry['accepted'] / entry['proposed'] if entry['proposed'] else None
            for baseline in ('mtp4-v4', 'mtp2-v4'):
                pairs = []
                for sample in samples:
                    control = [r for r in results if r['mode'] == baseline and r['case'] == case
                               and r['repeat'] == sample['repeat']]
                    if control:
                        assert len(control) == 1
                        pairs.append(100 * (sample['decode_tps'] / control[0]['decode_tps'] - 1))
                entry['paired_pct_vs_' + baseline] = pairs
                entry['median_pct_vs_' + baseline] = statistics.median(pairs) if pairs else None
            summary.append(entry)
    return dict(summary=summary, audits=rows,
                limitations='Tokens audited against actual target decisions. Cross-layout greedy equality is reported separately in token-comparisons.jsonl.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    (args.run / 'active-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))
