#!/usr/bin/env python3
"""Offline, conservative ABBA screening. Never deploys or collects console data."""
import argparse
import hashlib
import json
import math
import re
import statistics as st
from pathlib import Path

METRICS = ('kernel_ms', 'remote_ms', 'prefill_ms', 'decode_ms', 'total_ms')
GATES = ('serial', 'bulk_bitwise', 'matrix', 'long_context_quality', 'serving_mtp')
HASH = re.compile(r'^[0-9a-f]{64}$')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    require(isinstance(value, str) and HASH.fullmatch(value), 'expected lowercase SHA256')


def summary(values):
    med = st.median(values)
    return {'median': med, 'mad': st.median(abs(x-med) for x in values),
            'min': min(values), 'max': max(values), 'n': len(values)}


def analyze(data):
    require(data.get('schema') == 1, 'unsupported schema')
    require(data.get('data_kind') in ('synthetic', 'measured'), 'data_kind required')
    require(isinstance(data.get('workload'), dict) and data['workload'], 'workload required')
    workload = data['workload']
    for key in ('model_sha256', 'requests_sha256', 'tokens_sha256'):
        digest(workload.get(key))
    for key in ('hardware', 'configuration', 'command', 'sampling', 'thermal_clock_policy'):
        require(bool(workload.get(key)), 'missing workload ' + key)
    require(set(data.get('variants', {})) == {'A', 'B'}, 'variants A and B required')
    for variant in data['variants'].values():
        for key in ('source_sha256', 'binary_sha256', 'pso_sha256'):
            digest(variant.get(key))
        for key in ('commit', 'compiler', 'compiler_flags', 'feature_flags'):
            require(bool(variant.get(key)), 'missing variant ' + key)
    # Toolchain must match. Intentional experimental feature flags differ only here.
    for key in ('compiler', 'compiler_flags'):
        require(data['variants']['A'][key] == data['variants']['B'][key], 'toolchain mismatch: ' + key)
    gates = data.get('correctness', {})
    require(set(gates) == set(GATES), 'all correctness gates required')
    for gate in gates.values():
        require(gate.get('status') in ('pass', 'fail', 'not_run'), 'invalid gate status')
        if gate['status'] != 'not_run':
            digest(gate.get('evidence_sha256'))
    runs = data.get('runs', [])
    require(isinstance(runs, list) and runs, 'runs required')
    measured = []
    ids = set()
    sequence = 0
    for run in runs:
        require(isinstance(run.get('run_id'), str) and run['run_id'] and run['run_id'] not in ids, 'duplicate/missing run_id')
        ids.add(run['run_id'])
        require(type(run.get('sequence')) is int and run['sequence'] > sequence, 'non-monotonic sequence')
        sequence = run['sequence']
        require(type(run.get('warmup')) is bool, 'explicit warmup required')
        require(run.get('variant') in ('A', 'B'), 'invalid variant')
        require(run.get('workload') == workload, 'workload mismatch')
        require(run.get('artifact') == data['variants'][run['variant']], 'artifact mismatch')
        digest(run.get('raw_log_sha256'))
        require(set(run.get('timings', {})) == set(METRICS), 'all timing scopes required')
        for value in run['timings'].values():
            require(type(value) in (int, float) and math.isfinite(value) and value > 0, 'timings must be positive finite numbers')
        if run['warmup']:
            require(not measured, 'warmup must precede measured runs')
        else:
            measured.append(run)
    require(measured and len(measured) % 4 == 0, 'incomplete ABBA block')
    blocks = [measured[i:i+4] for i in range(0, len(measured), 4)]
    for block in blocks:
        require([r['variant'] for r in block] == list('ABBA'), 'expected A/B/B/A order')
    metrics = {}
    for metric in METRICS:
        gains, noise = [], []
        for block in blocks:
            a1, b1, b2, a2 = [r['timings'][metric] for r in block]
            a, b = (a1+a2)/2, (b1+b2)/2
            gains.append(100*(a-b)/a)  # latency reduction, NOT throughput gain
            noise.append(100*max(abs(a1-a2), abs(b1-b2))/a)
        stats = summary(gains)
        # Conservative screen, not a confidence interval or proof of independence.
        threshold = max(0.5, st.median(noise), 3*stats['mad'])
        wins = sum(g > threshold for g in gains)
        # Exact one-sided sign-test under independent blocks; diagnostic only.
        p = sum(math.comb(len(gains), k) for k in range(wins, len(gains)+1)) / 2**len(gains)
        state = 'inconclusive'
        if len(blocks) >= 6:
            if stats['median'] > threshold and p < 0.025:
                state = 'improvement_signal'
            elif stats['median'] < -threshold:
                state = 'regression_signal'
        metrics[metric] = {'A_ms': summary([r['timings'][metric] for r in measured if r['variant']=='A']),
                           'B_ms': summary([r['timings'][metric] for r in measured if r['variant']=='B']),
                           'block_latency_reduction_pct': stats, 'block_gains_pct': gains,
                           'within_block_spread_pct': summary(noise), 'noise_threshold_pct': threshold,
                           'sign_test_p_diagnostic': p, 'screen': state}
    reasons = []
    if data['data_kind'] != 'measured': reasons.append('synthetic_data_not_hardware_evidence')
    if len(blocks) < 6: reasons.append('insufficient_replicates_minimum_6_complete_blocks')
    for key, gate in gates.items():
        if gate['status'] != 'pass': reasons.append('correctness_' + key + '_' + gate['status'])
    for metric in ('decode_ms', 'total_ms'):
        if metrics[metric]['screen'] != 'improvement_signal': reasons.append(metric + '_gain_not_established')
    for metric in METRICS:
        if metrics[metric]['screen'] == 'regression_signal': reasons.append(metric + '_regression')
    return {'schema': 1, 'data_kind': data['data_kind'], 'complete_blocks': len(blocks),
            'decision': 'not_qualified' if reasons else 'eligible_for_human_review_only',
            'reasons': reasons, 'metrics': metrics, 'workload': workload,
            'variants': data['variants'], 'correctness': gates,
            'caution': 'Offline screening only. Hashes are declared, not attested. No automatic promotion.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    try:
        raw = args.input.read_bytes()
        result = analyze(json.loads(raw))
        result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        parser.exit(2, 'invalid evidence: ' + str(exc) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
