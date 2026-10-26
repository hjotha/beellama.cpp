#!/usr/bin/env python3
"""A/B the opt-in RKVA prefill INT8 QK kernel (`attn_kvarn4_mq2_i8`) on the Xbox.

PREVIEW BY DEFAULT. Nothing is executed without --execute, and even then this
script installs nothing: it only toggles the app-side LocalState flag file that
selects the INT8 PSO (through the operator-provided commands), then

  1. runs `rkva_kvarn_serial_prefix_probe.py` in each arm, so the attention
     OUT/LSE error against its NumPy oracle is measured with the unchanged
     probe (this is the numeric gate: the variant is NOT bit-exact),
  2. runs the frozen-prefix `bench-rkva-range.py` in each arm for `pass1_us`
     (from the engine buckets), prefill t/s and the 256 generated ids,
  3. restores the flag OFF in a `finally`, whatever happened.

Refuses to run when a benchmark/build already owns this host.

Example:

  python3 tools/xbox/scripts/run-rkva-prefill-int8-ab.py \
    --server-binary build/bin/llama-server \
    --model /path/Qwen3.8-27B-IQ3_XXS-mtp.gguf \
    --prompt-tokens /path/prompt-tokens.json \
    --xbox-host 192.168.1.200:50054 --probe-host 192.168.1.200 --probe-port 50054 \
    --prefix-snapshot /path/prefix-102400.slot \
    --enable-cmd  'python3 tools/xbox/scripts/xbox-localstate-flag.py set rkva-mq-i8.flag' \
    --disable-cmd 'python3 tools/xbox/scripts/xbox-localstate-flag.py clear rkva-mq-i8.flag' \
    --output .oc-artifacts/ab-int8 --execute --json .oc-artifacts/ab-int8.json

Exit codes: 0 pass, 2 refused (busy/unsafe), 3 failure, 4 preview.
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

FOREIGN_PATTERN = 'llama-server|bench-rkva|test-rkva|ninja|cmake --build|rkva_kvarn_serial'
BUCKET_PREFIX = 'attention/kvarn4-prefix/optimized-kvarn4_mq2'
I8_BUCKET = BUCKET_PREFIX + '_i8/nq2-16/r16k-32k/warm'
FINE_BUCKET = BUCKET_PREFIX + '_fine/nq2-16/r16k-32k/warm'
# Accept only if the variant stays inside the error the shipped 4-bit cache
# already introduces (the same 1.15x used by tools/xbox/tests/rkva_mq_i8_quality.py).
TOLERANCE_RATIO = 1.15


def active_foreign_processes():
    out = subprocess.run(['pgrep', '-af', FOREIGN_PATTERN], capture_output=True, text=True)
    rows = []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        pid, _, cmd = line.partition(' ')
        if str(os.getpid()) == pid:
            continue
        if 'run-rkva-prefill-int8-ab.py' in cmd or 'pgrep -af' in cmd:
            continue
        rows.append({'pid': int(pid), 'cmd': cmd[:400]})
    return rows


def run(cmd, log_path, timeout):
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + proc.stderr)
    return {
        'cmd': cmd,
        'returncode': proc.returncode,
        'elapsed_s': round(time.time() - started, 3),
        'log': str(log_path),
        'stdout_tail': proc.stdout[-2000:],
        'stderr_tail': proc.stderr[-2000:],
    }


def read_profile_buckets(output_dir):
    """Engine buckets from the benchmark's profile-lines.json (JSON array of log lines)."""
    path = pathlib.Path(output_dir) / 'profile-lines.json'
    if not path.exists():
        return {}
    lines = json.loads(path.read_text())
    for line in reversed(lines):
        marker = 'remote_attn gpu_profile '
        idx = line.find(marker)
        if idx < 0:
            continue
        try:
            doc = json.loads(line[idx + len(marker):])
        except json.JSONDecodeError:
            continue
        engine = doc.get('engine') or {}
        return {
            'counters': engine.get('counters', {}),
            'multi_query': (engine.get('kvarn4_multi_query') or {}),
            'buckets': engine.get('buckets', {}),
        }
    return {}


def bucket_summary(profile, key):
    bucket = (profile.get('buckets') or {}).get(key)
    if not bucket:
        return None
    pass1 = (bucket.get('gpu_us') or {}).get('pass1') or {}
    return {
        'samples': bucket.get('samples'),
        'nq_mean': (bucket.get('n_tokens') or {}).get('mean'),
        'active_rows_mean': (bucket.get('active_rows') or {}).get('mean'),
        'split_k_mean': (bucket.get('split_k') or {}).get('mean'),
        'pass1_us_mean': pass1.get('mean'),
        'pass1_us_p50': pass1.get('p50'),
        'pass1_us_p95': pass1.get('p95'),
        'pass1_us_sum': pass1.get('sum'),
    }


def generated_ids(result_json):
    path = pathlib.Path(result_json)
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    for key in ('generated_token_ids', 'generated_ids', 'output_ids', 'ids', 'tokens'):
        value = doc.get(key)
        if isinstance(value, list):
            return value
    runs = doc.get('runs')
    if isinstance(runs, list) and runs:
        for run_doc in runs:
            for key in ('generated_token_ids', 'ids'):
                if isinstance(run_doc, dict) and isinstance(run_doc.get(key), list):
                    return run_doc[key]
    return None


def probe_error_floor(probe_json):
    """Worst OUT/LSE absolute error the unchanged serial-prefix probe measured.

    `rkva_kvarn_serial_prefix_probe.py` compares every GPU reply against its
    independent NumPy oracle and records out_max_abs / lse_max_abs per check.
    The INT8 variant is not bit-exact, so this floor is the numeric gate.
    """
    path = pathlib.Path(probe_json)
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    out_max = 0.0
    lse_max = 0.0
    checks = 0
    for run_doc in doc.get('runs') or []:
        for result in run_doc.get('results') or []:
            for check in result.get('checks') or []:
                if 'out_max_abs' not in check:
                    continue
                checks += 1
                out_max = max(out_max, float(check['out_max_abs']))
                lse_max = max(lse_max, float(check.get('lse_max_abs') or 0.0))
    return {'checks': checks, 'out_max_abs': out_max, 'lse_max_abs': lse_max,
            'probe_ok': bool(doc.get('success'))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--execute', action='store_true', help='run the hardware A/B (default: preview)')
    ap.add_argument('--json', default='rkva-prefill-int8-ab.json')
    ap.add_argument('--output', default='rkva-prefill-int8-ab')
    ap.add_argument('--bench', default='tools/xbox/scripts/bench-rkva-range.py')
    ap.add_argument('--probe', default='tools/xbox/tests/rkva_kvarn_serial_prefix_probe.py')
    ap.add_argument('--server-binary', default='build/bin/llama-server')
    ap.add_argument('--model', default='')
    ap.add_argument('--prompt-tokens', default='')
    ap.add_argument('--xbox-host', default='')
    ap.add_argument('--probe-host', default='')
    ap.add_argument('--probe-port', type=int, default=50054)
    ap.add_argument('--probe-modes', default='2', help='engine modes to probe (2 = mq2 fine)')
    ap.add_argument('--prefix-snapshot', default='')
    ap.add_argument('--points', default='130560')
    ap.add_argument('--context', default='131072')
    ap.add_argument('--n-predict', default='256')
    ap.add_argument('--seed', default='1234')
    ap.add_argument('--threads', default='8')
    ap.add_argument('--enable-cmd', default='',
                    help='operator command that creates the LocalState flag rkva-mq-i8.flag')
    ap.add_argument('--disable-cmd', default='',
                    help='operator command that removes the same flag (also the restore step)')
    ap.add_argument('--timeout', type=int, default=7200)
    args = ap.parse_args()

    doc = {
        'schema': 'rkva-prefill-int8-ab-v1',
        'started_unix': int(time.time()),
        'cwd': str(pathlib.Path.cwd()),
        'mode': 'execute' if args.execute else 'preview',
        'flag': 'rkva-mq-i8.flag (LocalState)',
        'kernel_on': 'attn_kvarn4_mq2_i8 / bucket ' + I8_BUCKET,
        'kernel_off': 'attn_kvarn4_mq2 / bucket ' + FINE_BUCKET,
        'exact': False,
        'expectations': [
            'both arms run the same frozen prefix, same sampling settings, same seed',
            'ON arm profile reports kvarn4_multi_query.mq2_i8.calls > 0 and "exact":false',
            'ON arm bucket ' + I8_BUCKET + ' carries the pass1 samples',
            'ON pass1_us_mean < OFF pass1_us_mean (static count predicts about -35%%)',
            'ON probe out_max_abs <= %.2fx OFF probe out_max_abs' % TOLERANCE_RATIO,
            'generated ids are reported, not required to match: this variant is not exact',
        ],
    }

    # A preview starts no process and writes no artifact, so it is allowed while
    # a build owns the host; only --execute refuses.
    busy = active_foreign_processes()
    doc['foreign_processes'] = busy
    bench_base = [sys.executable, args.bench,
                  '--server-binary', args.server_binary,
                  '--model', args.model,
                  '--prompt-tokens', args.prompt_tokens,
                  '--xbox-host', args.xbox_host,
                  '--points', args.points,
                  '--context', args.context,
                  '--n-predict', args.n_predict,
                  '--threads', args.threads,
                  '--seed', args.seed,
                  '--range-kernel', 'fine',
                  '--cuda-fp8', 'mapped',
                  '--tcp-prefetch',
                  '--profile']
    if args.prefix_snapshot:
        bench_base += ['--prefix-snapshot', args.prefix_snapshot]
    probe_base = [sys.executable, args.probe, '--execute',
                  '--host', args.probe_host, '--port', str(args.probe_port),
                  '--precision-tails', '128',
                  '--modes'] + args.probe_modes.split()

    plan = []
    for arm, flag_cmd in (('off', args.disable_cmd), ('on', args.enable_cmd)):
        out_dir = str(pathlib.Path(args.output) / arm)
        plan.append({
            'arm': arm,
            'toggle': flag_cmd or None,
            'probe': probe_base + ['--output', str(pathlib.Path(out_dir) / 'probe.json')],
            'bench': bench_base + ['--output', out_dir],
        })
    plan.append({'arm': 'restore', 'toggle': args.disable_cmd or None})
    doc['plan'] = plan

    if not args.execute:
        doc['status'] = 'preview'
        doc['note'] = ('preview only; pass --execute, --enable-cmd and --disable-cmd to run. '
                       'This script installs nothing and needs a package that embeds '
                       'attn_kvarn4_mq2_i8 (SM 6.4 dot4add_i8packed).')
        pathlib.Path(args.json).write_text(json.dumps(doc, indent=1) + '\n')
        print(json.dumps(doc, indent=1))
        return 4

    if busy:
        doc['status'] = 'refused'
        doc['reason'] = 'another benchmark/build is active: ' + FOREIGN_PATTERN
        pathlib.Path(args.json).write_text(json.dumps(doc, indent=1) + '\n')
        print(json.dumps(doc, indent=1))
        return 2
    missing = [n for n, v in (('--model', args.model), ('--prompt-tokens', args.prompt_tokens),
                              ('--xbox-host', args.xbox_host), ('--probe-host', args.probe_host),
                              ('--enable-cmd', args.enable_cmd),
                              ('--disable-cmd', args.disable_cmd)) if not v]
    if missing:
        doc['status'] = 'refused'
        doc['reason'] = 'missing required arguments: ' + ', '.join(missing)
        pathlib.Path(args.json).write_text(json.dumps(doc, indent=1) + '\n')
        print(json.dumps(doc, indent=1))
        return 2
    if not shutil.which(args.server_binary) and not pathlib.Path(args.server_binary).exists():
        doc['status'] = 'refused'
        doc['reason'] = 'server binary not found: ' + args.server_binary
        pathlib.Path(args.json).write_text(json.dumps(doc, indent=1) + '\n')
        print(json.dumps(doc, indent=1))
        return 2

    logs = pathlib.Path(args.output) / 'logs'
    steps, profiles, results, errors = [], {}, {}, {}
    try:
        for arm, flag_cmd in (('off', args.disable_cmd), ('on', args.enable_cmd)):
            steps.append(run(flag_cmd, logs / f'{arm}-toggle.log', 300))
            if steps[-1]['returncode'] != 0:
                raise RuntimeError(f'{arm}: flag toggle failed: {flag_cmd}')
            out_dir = pathlib.Path(args.output) / arm
            out_dir.mkdir(parents=True, exist_ok=True)
            probe_step = run(probe_base + ['--output', str(out_dir / 'probe.json')],
                             logs / f'{arm}-probe.log', args.timeout)
            probe_step['arm'] = arm + '/probe'
            steps.append(probe_step)
            errors[arm] = probe_error_floor(out_dir / 'probe.json')
            bench_step = run(bench_base + ['--output', str(out_dir), '--execute'],
                             logs / f'{arm}-bench.log', args.timeout)
            bench_step['arm'] = arm + '/bench'
            steps.append(bench_step)
            if bench_step['returncode'] != 0:
                raise RuntimeError(f'{arm}: benchmark failed (see {bench_step["log"]})')
            profiles[arm] = read_profile_buckets(out_dir)
            results[arm] = {'ids': generated_ids(out_dir / 'results.json'),
                            'results_json': str(out_dir / 'results.json')}
    finally:
        restore = run(args.disable_cmd, logs / 'restore-toggle.log', 300)
        restore['arm'] = 'restore'
        steps.append(restore)

    doc['steps'] = steps
    off_summary = bucket_summary(profiles.get('off', {}), FINE_BUCKET)
    on_summary = bucket_summary(profiles.get('on', {}), I8_BUCKET)
    i8_calls = (((profiles.get('on', {}).get('multi_query') or {}).get('mq2_i8') or {})
                .get('calls'))
    off_ids, on_ids = results['off']['ids'], results['on']['ids']
    off_err, on_err = errors.get('off'), errors.get('on')
    doc['measurements'] = {
        'off_bucket': off_summary,
        'on_bucket': on_summary,
        'on_i8_calls': i8_calls,
        'on_i8_available': (((profiles.get('on', {}).get('multi_query') or {})
                             .get('mq2_i8') or {}).get('available')),
        'pass1_us_mean_delta': (
            None if not (off_summary and on_summary) else
            round(on_summary['pass1_us_mean'] - off_summary['pass1_us_mean'], 3)),
        'pass1_ratio': (
            None if not (off_summary and on_summary) or not off_summary['pass1_us_mean'] else
            round(on_summary['pass1_us_mean'] / off_summary['pass1_us_mean'], 6)),
        'off_probe_error': off_err,
        'on_probe_error': on_err,
        'out_error_ratio': (
            None if not (off_err and on_err) or not off_err['out_max_abs'] else
            round(on_err['out_max_abs'] / off_err['out_max_abs'], 6)),
        'lse_error_ratio': (
            None if not (off_err and on_err) or not off_err['lse_max_abs'] else
            round(on_err['lse_max_abs'] / off_err['lse_max_abs'], 6)),
        'generated_ids': {'off_count': None if off_ids is None else len(off_ids),
                          'on_count': None if on_ids is None else len(on_ids),
                          'equal': off_ids is not None and off_ids == on_ids,
                          'first_divergence': next(
                              (i for i, (a, b) in enumerate(zip(off_ids or [], on_ids or []))
                               if a != b), None)},
    }
    ratio = doc['measurements']['out_error_ratio']
    checks = {
        'on_arm_used_i8_kernel': bool(i8_calls),
        'on_arm_has_i8_bucket': on_summary is not None,
        'off_arm_has_fine_bucket': off_summary is not None,
        'probe_measured_in_both_arms': bool(off_err and on_err and off_err['checks']
                                           and on_err['checks']),
        'out_error_within_tolerance': None if ratio is None else ratio <= TOLERANCE_RATIO,
        'pass1_improved': (None if doc['measurements']['pass1_ratio'] is None
                           else doc['measurements']['pass1_ratio'] < 1.0),
    }
    doc['checks'] = checks
    gating = [v for v in checks.values() if isinstance(v, bool)]
    doc['status'] = 'pass' if gating and all(gating) else 'fail'
    doc['finished_unix'] = int(time.time())
    pathlib.Path(args.json).write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps({'status': doc['status'], 'checks': checks,
                      'measurements': doc['measurements']}, indent=1))
    return 0 if doc['status'] == 'pass' else 3


if __name__ == '__main__':
    sys.exit(main())
