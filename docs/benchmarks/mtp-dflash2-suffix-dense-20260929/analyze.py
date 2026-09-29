"""Offline opportunity measurement; never supplies future tokens to the drafter."""
import argparse
import collections
import json
from pathlib import Path
import re
import statistics


def parse_log(text):
    events = [json.loads(m[1]) for m in re.finditer(r'SHADOWTRACE (\{[^\n]*\})', text)]
    jobs = []
    for match in re.finditer(r'SHADOWv2 job=[^\n]*', text):
        fields = dict(re.findall(r'(\w+)=(\[[^\]]*\]|-?\d+)', match[0]))
        jobs.append({k: json.loads(v) if v.startswith('[') else int(v) for k, v in fields.items()})
    return events, jobs


def put_token(truth, position, token):
    if position in truth and truth[position] != token:
        raise ValueError(f'Conflicting confirmed token at position {position}')
    truth[position] = token


def match_suffix(suffix, truth, start):
    """Return (known matching length, exact length). Missing is NOT rejection."""
    matched = 0
    for offset, token in enumerate(suffix):
        if start + offset not in truth:
            return matched, False
        if truth[start + offset] != token:
            return matched, True
        matched += 1
    return matched, True


def evaluate_job(job, truth, drafts, commits):
    c = len(job['confirmed'])
    assert c == job['c'] and len(job['proposed']) == job['d']
    for i, token in enumerate(job['confirmed']):
        if job['pos0'] + 1 + i in truth:
            assert truth[job['pos0'] + 1 + i] == token
    assert job['proposed'][:c] == job['confirmed']
    assert truth[job['pos0']] == job['anchor']
    next_pos = job['pos0'] + c
    suffix = job['proposed'][c:c + job['usable_tokens']]
    assert suffix and job['ready'] and job['position_ok'] and job['prefix_ok']
    length, exact = match_suffix(suffix, truth, next_pos + 1)
    draft = drafts.get(next_pos)
    if draft is None:
        raise ValueError(f'Missing next primary draft at {next_pos}')
    assert draft['anchor'] == job['confirmed'][-1]
    commit = commits.get(next_pos)
    if commit:
        assert commit['anchor'] == draft['anchor']
        next_c = len(commit['confirmed'])
        commit_us = commit['done_us']
    else:
        if draft['proposed']:
            raise ValueError(f'Missing commit for nonempty primary draft at {next_pos}')
        next_c = 1
        commit_us = None
    # Include the next round's bonus, not the round that produced this suffix.
    # Exclude request tails where either counterfactual extends past the output.
    complete = exact and next_pos + length + 1 in truth and next_pos + next_c in truth
    return dict(job=job['job'], epoch=job['epoch'], request=job['request'],
                pos0=job['pos0'], origin_c=c, next_pos=next_pos, suffix=suffix,
                length=length, length_exact=exact, comparison_complete=complete,
                first_match=None if next_pos + 1 not in truth else length > 0,
                next_mtp_c=next_c, delta_tokens=length + 1 - next_c if complete else None,
                next_draft_us=draft['done_us'] - draft['start_us'],
                next_round_us=commit_us - draft['start_us'] if commit_us else None)


def aggregate(rows):
    complete = [r for r in rows if r['comparison_complete']]
    first = [r for r in rows if r['first_match'] is not None]
    exact = [r for r in rows if r['length_exact']]
    return dict(usable=len(rows), comparisons_complete=len(complete),
                censored=len(rows) - len(complete), first_known=len(first),
                first_match=sum(r['first_match'] for r in first),
                length_histogram=dict(sorted(collections.Counter(r['length'] for r in exact).items())),
                mean_length=statistics.mean(r['length'] for r in exact) if exact else None,
                delta_histogram=dict(sorted(collections.Counter(r['delta_tokens'] for r in complete).items())),
                mean_delta=statistics.mean(r['delta_tokens'] for r in complete) if complete else None,
                mtp_advance=sum(r['next_mtp_c'] for r in complete),
                suffix_advance=sum(r['length'] + 1 for r in complete),
                next_draft_us_sum=sum(r['next_draft_us'] for r in complete),
                next_round_us_sum=sum(r['next_round_us'] or 0 for r in complete),
                # Diagnostic groups by available suffix length, not future success.
                by_suffix_length={str(n): dict(count=len(group),
                    mean_L=statistics.mean(r['length'] for r in group),
                    mean_next_c=statistics.mean(r['next_mtp_c'] for r in group),
                    mean_delta=statistics.mean(r['delta_tokens'] for r in group))
                    for n in sorted({len(r['suffix']) for r in complete})
                    if (group := [r for r in complete if len(r['suffix']) == n])})


def analyze_run(root):
    results = json.loads((root / 'results.json').read_text())
    report = {}
    all_rows = []
    all_tokens = {}
    for mode in dict.fromkeys(r['mode'] for r in results):
        events, jobs = parse_log((root / f'{mode}.server-output.txt').read_text(errors='replace'))
        begins = [e for e in events if e['event'] == 'begin']
        scenarios = ['warmup'] + [r['scenario'] for r in results if r['mode'] == mode]
        assert len(begins) == len(scenarios), (mode, len(begins), len(scenarios))
        assert len({(e['request'], e['slot']) for e in begins}) == len(begins)
        mode_rows = []
        for begin, scenario in zip(begins, scenarios):
            key = begin['request'], begin['slot']
            request_events = [e for e in events if (e['request'], e['slot']) == key]
            ends = [e for e in request_events if e['event'] == 'end']
            assert len(ends) == 1 and not ends[0]['truncated'], (mode, scenario, ends)
            assert begin['temperature'] == 0
            singles = [e for e in request_events if e['event'] == 'token']
            assert singles
            first_pos = singles[0]['pos']
            response = json.loads((root / f'{mode}-{scenario}.response.json').read_text())
            output = response['tokens']
            assert output and len(output) == ends[0]['generated']
            truth = {first_pos + i: token for i, token in enumerate(output)}
            all_tokens[mode, scenario] = output
            # Audit the full emitted trajectory against final server decisions.
            observed = {}
            for e in request_events:
                if e['event'] == 'token':
                    put_token(observed, e['pos'], e['token'])
                elif e['event'] == 'commit':
                    put_token(observed, e['pos0'], e['anchor'])
                    for i, token in enumerate(e['confirmed']):
                        put_token(observed, e['pos0'] + 1 + i, token)
            assert all(observed.get(pos) == token for pos, token in truth.items()), (mode, scenario)
            drafts = {e['pos0']: e for e in request_events if e['event'] == 'draft'}
            commits = {e['pos0']: e for e in request_events if e['event'] == 'commit'}
            assert len(drafts) == sum(e['event'] == 'draft' for e in request_events)
            assert len(commits) == sum(e['event'] == 'commit' for e in request_events)
            selected = [j for j in jobs if j['request'] == key[0] and j['usable']]
            assert key[1] == 0  # This experiment explicitly supports one slot.
            assert len({j['epoch'] for j in selected}) <= 1
            for job in selected:
                row = evaluate_job(job, truth, drafts, commits)
                # A separate verifier capacity of four is a future design, not
                # the configured capability of the n=2/3 experiment.
                wider = dict(job, usable_tokens=min(4, job['d'] - job['c']))
                at_four = evaluate_job(wider, truth, drafts, commits)
                row['capacity4'] = {k: at_four[k] for k in (
                    'suffix', 'length', 'length_exact', 'comparison_complete', 'delta_tokens')}
                row.update(mode=mode, scenario=scenario)
                if scenario != 'warmup':
                    mode_rows.append(row)
                all_rows.append(row)
        report[mode] = dict(jobs_including_warmup=len(jobs), measured=aggregate(mode_rows),
                            hypothetical_capacity4=aggregate([
                                dict(r, **r['capacity4']) for r in mode_rows]),
                            scenarios={s: aggregate([r for r in mode_rows if r['scenario'].split('-')[0] == s])
                                       for s in ('repeticao', 'codigo', 'longo')})
    # Depth may change the trajectory numerically; compare each observer with
    # its own-depth contemporaneous reference and also report n=4 equality.
    checks = []
    for (mode, scenario), tokens in all_tokens.items():
        if not mode.startswith('shadow-'):
            continue
        depth = mode.split('-')[1]
        baseline = 'mtp-n4-before' if depth == 'n4' else 'mtp-' + depth
        checks.append(dict(mode=mode, scenario=scenario,
                           matches_same_depth=tokens == all_tokens[baseline, scenario],
                           matches_n4=tokens == all_tokens['mtp-n4-before', scenario]))
    assert all(c['matches_same_depth'] for c in checks), 'Observer changed same-depth output'
    repeat_checks = {scenario: tokens == all_tokens['mtp-n4-before', scenario]
                     for (mode, scenario), tokens in all_tokens.items() if mode == 'mtp-n4-after'}
    assert repeat_checks and all(repeat_checks.values()), 'Repeated n4 control changed output'
    depth_differences = []
    for (mode, scenario), tokens in all_tokens.items():
        if mode not in ('mtp-n2', 'mtp-n3'):
            continue
        reference = all_tokens['mtp-n4-before', scenario]
        first = next((i for i, pair in enumerate(zip(reference, tokens)) if pair[0] != pair[1]), None)
        depth_differences.append(dict(mode=mode, scenario=scenario, first_difference=first,
            reference_token=reference[first] if first is not None else None,
            actual_token=tokens[first] if first is not None else None))
    projections = []
    for mode in report:
        if not mode.startswith('shadow-'):
            continue
        depth = mode.split('-')[1]
        baseline = 'mtp-n4-before' if depth == 'n4' else 'mtp-' + depth
        for scenario in ('all', 'repeticao', 'codigo', 'longo'):
            measured = [r for r in results if r['mode'] == mode and
                        (scenario == 'all' or r['scenario'].startswith(scenario))]
            control = [r for r in results if r['mode'] == baseline and
                       (scenario == 'all' or r['scenario'].startswith(scenario))]
            # API decode timing excludes the first sampled token.
            seconds = sum(r['decode_ms'] for r in measured) / 1000
            tokens = sum(r['predicted_n'] - 1 for r in measured)
            control_tps = sum(r['predicted_n'] - 1 for r in control) / (sum(r['decode_ms'] for r in control) / 1000)
            for minimum in (1, 2, 3, 4):
                rows = [r for r in all_rows if r['mode'] == mode and r['scenario'] != 'warmup'
                        and (scenario == 'all' or r['scenario'].startswith(scenario))
                        and r['comparison_complete'] and len(r['suffix']) >= minimum]
                avoided = sum(r['next_draft_us'] for r in rows) / 1e6
                delta = sum(r['delta_tokens'] for r in rows)
                # Local what-if only: leaves verification costs and subsequent
                # anchors fixed, so this is neither a replay nor a speed bound.
                estimate = (tokens + delta) / (seconds - avoided)
                projections.append(dict(mode=mode, scenario=scenario, minimum_suffix=minimum,
                    opportunities=len(rows), delta_tokens=delta, avoided_dispatch_s=avoided,
                    observer_tps=tokens / seconds, control_tps=control_tps,
                    local_estimate_tps=estimate, local_delta_pct=100 * (estimate / control_tps - 1)))
    return dict(modes=report, rows=all_rows, token_checks=checks, repeat_checks=repeat_checks,
                depth_differences=depth_differences, local_projections=projections)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = analyze_run(args.run)
    (args.run / 'suffix-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v['measured'] for k, v in result['modes'].items()}, indent=2))
