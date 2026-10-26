#!/usr/bin/env python3
"""A/B the opt-in RKVA prefill rescale-elision kernel on the Xbox Series X.

PREVIEW BY DEFAULT. Nothing is executed without --execute, and even then this
script installs nothing: it only toggles the app-side LocalState flag file that
selects `attn_kvarn4_mq2_skip` (through the operator-provided commands), runs
the existing frozen-prefix benchmark twice, compares the generated tokens byte
for byte and reads the engine's `remote_attn gpu_profile` buckets.

Refuses to run when a Codex benchmark/build is already active on this host.

Example (fills in the paths of the frozen prefix campaign):

  python3 tools/xbox/scripts/run-rkva-prefill-kernel-ab.py \
    --server-binary build/bin/llama-server \
    --model /path/Qwen3.8-27B-IQ3_XXS-mtp.gguf \
    --prompt-tokens /path/prompt-tokens.json \
    --xbox-host 192.168.1.200:50054 \
    --prefix-snapshot /path/prefix-102400.slot \
    --points 130560 --context 131072 \
    --enable-cmd  'python3 tools/xbox/scripts/xbox-localstate-flag.py set rkva-mq-rescale-skip.flag' \
    --disable-cmd 'python3 tools/xbox/scripts/xbox-localstate-flag.py clear rkva-mq-rescale-skip.flag' \
    --output .oc-artifacts/ab-rescale-skip \
    --execute --json .oc-artifacts/ab-rescale-skip.json

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

CODEX_PROCESS_PATTERN = 'llama-server|bench-rkva|test-rkva|ninja|cmake --build'
BUCKET_PREFIX = 'attention/kvarn4-prefix/optimized-kvarn4_mq2'
SKIP_BUCKET = BUCKET_PREFIX + '_skip/nq2-16/r16k-32k/warm'
FINE_BUCKET = BUCKET_PREFIX + '_fine/nq2-16/r16k-32k/warm'


def active_foreign_processes():
    """Processes that mean a Codex benchmark/build owns this host right now."""
    out = subprocess.run(['pgrep', '-af', CODEX_PROCESS_PATTERN], capture_output=True, text=True)
    rows = []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        pid, _, cmd = line.partition(' ')
        if str(os.getpid()) == pid:
            continue
        if 'run-rkva-prefill-kernel-ab.py' in cmd or 'pgrep -af' in cmd:
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
            'gpu_timestamp_frequency_hz': engine.get('gpu_timestamp_frequency_hz'),
        }
    return {}


def bucket_summary(profile, key):
    bucket = (profile.get('buckets') or {}).get(key)
    if not bucket:
        return None
    pass1 = (bucket.get('gpu_us') or {}).get('pass1') or {}
    rows = bucket.get('active_rows') or {}
    split = bucket.get('split_k') or {}
    return {
        'samples': bucket.get('samples'),
        'nq_mean': (bucket.get('n_tokens') or {}).get('mean'),
        'active_rows_mean': rows.get('mean'),
        'split_k_mean': split.get('mean'),
        'pass1_us_mean': pass1.get('mean'),
        'pass1_us_p50': pass1.get('p50'),
        'pass1_us_p95': pass1.get('p95'),
        'pass1_us_sum': pass1.get('sum'),
    }


def generated_ids(result_json):
    doc = json.loads(pathlib.Path(result_json).read_text())
    for key in ('generated_token_ids', 'generated_ids', 'output_ids', 'ids', 'tokens'):
        value = doc.get(key)
        if isinstance(value, list):
            return value
    runs = doc.get('runs')
    if isinstance(runs, list) and runs:
        for key in ('generated_token_ids', 'ids'):
            for run in runs:
                if isinstance(run, dict) and isinstance(run.get(key), list):
                    return run[key]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--execute', action='store_true', help='run the hardware A/B (default: preview)')
    ap.add_argument('--json', default='rkva-prefill-kernel-ab.json')
    ap.add_argument('--output', default='rkva-prefill-kernel-ab')
    ap.add_argument('--bench', default='tools/xbox/scripts/bench-rkva-range.py')
    ap.add_argument('--server-binary', default='build/bin/llama-server')
    ap.add_argument('--model', default='')
    ap.add_argument('--prompt-tokens', default='')
    ap.add_argument('--xbox-host', default='')
    ap.add_argument('--prefix-snapshot', default='')
    ap.add_argument('--points', default='130560')
    ap.add_argument('--context', default='131072')
    ap.add_argument('--n-predict', default='256')
    ap.add_argument('--seed', default='1234')
    ap.add_argument('--threads', default='8')
    ap.add_argument('--enable-cmd', default='',
                    help='operator command that creates the LocalState flag rkva-mq-rescale-skip.flag')
    ap.add_argument('--disable-cmd', default='',
                    help='operator command that removes the same flag (also the restore step)')
    ap.add_argument('--timeout', type=int, default=7200)
    args = ap.parse_args()

    doc = {
        'schema': 'rkva-prefill-kernel-ab-v1',
        'started_unix': int(time.time()),
        'cwd': str(pathlib.Path.cwd()),
        'mode': 'execute' if args.execute else 'preview',
        'flag': 'rkva-mq-rescale-skip.flag (LocalState)',
        'kernel_on': 'attn_kvarn4_mq2_skip / bucket ' + SKIP_BUCKET,
        'kernel_off': 'attn_kvarn4_mq2 / bucket ' + FINE_BUCKET,
        'expectations': [
            'both arms run the same frozen prefix and identical sampling settings',
            'generated token ids identical between arms (bit-exact kernel)',
            'ON arm profile reports kvarn4_multi_query.mq2_rescale_skip.calls > 0',
            'ON arm bucket ' + SKIP_BUCKET + ' carries the pass1 samples',
        ],
    }

    busy = active_foreign_processes()
    doc['foreign_processes'] = busy
    if busy:
        doc['status'] = 'refused'
        doc['reason'] = 'another benchmark/build is active: ' + CODEX_PROCESS_PATTERN
        pathlib.Path(args.json).write_text(json.dumps(doc, indent=1) + '\n')
        print(json.dumps(doc, indent=1))
        return 2

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

    plan = []
    for arm, flag_cmd in (('off', args.disable_cmd), ('on', args.enable_cmd)):
        out_dir = str(pathlib.Path(args.output) / arm)
        plan.append({
            'arm': arm,
            'toggle': flag_cmd or None,
            'bench': bench_base + ['--output', out_dir],
        })
    plan.append({'arm': 'restore', 'toggle': args.disable_cmd or None})
    doc['plan'] = plan

    if not args.execute:
        doc['status'] = 'preview'
        doc['note'] = ('preview only; pass --execute, --enable-cmd and --disable-cmd to run. '
                       'This script installs nothing and needs the package with the skip PSO.')
        pathlib.Path(args.json).write_text(json.dumps(doc, indent=1) + '\n')
        print(json.dumps(doc, indent=1))
        return 4

    missing = [n for n, v in (('--model', args.model), ('--prompt-tokens', args.prompt_tokens),
                              ('--xbox-host', args.xbox_host), ('--enable-cmd', args.enable_cmd),
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
    steps = []
    profiles = {}
    results = {}
    try:
        for arm, flag_cmd in (('off', args.disable_cmd), ('on', args.enable_cmd)):
            steps.append(run(flag_cmd, logs / f'{arm}-toggle.log', 300))
            if steps[-1]['returncode'] != 0:
                raise RuntimeError(f'{arm}: flag toggle failed: {flag_cmd}')
            out_dir = str(pathlib.Path(args.output) / arm)
            step = run(bench_base + ['--output', out_dir, '--execute'],
                       logs / f'{arm}-bench.log', args.timeout)
            step['arm'] = arm
            steps.append(step)
            if step['returncode'] != 0:
                raise RuntimeError(f'{arm}: benchmark failed (see {step["log"]})')
            profiles[arm] = read_profile_buckets(out_dir)
            result_json = pathlib.Path(out_dir) / 'results.json'
            results[arm] = {
                'ids': generated_ids(result_json) if result_json.exists() else None,
                'results_json': str(result_json),
            }
    finally:
        restore = run(args.disable_cmd, logs / 'restore-toggle.log', 300)
        restore['arm'] = 'restore'
        steps.append(restore)

    doc['steps'] = steps
    off_summary = bucket_summary(profiles.get('off', {}), FINE_BUCKET)
    on_summary = bucket_summary(profiles.get('on', {}), SKIP_BUCKET)
    skip_calls = (((profiles.get('on', {}).get('multi_query') or {}).get('mq2_rescale_skip') or {})
                  .get('calls'))
    off_ids, on_ids = results['off']['ids'], results['on']['ids']
    doc['measurements'] = {
        'off_bucket': off_summary,
        'on_bucket': on_summary,
        'on_skip_calls': skip_calls,
        'pass1_us_mean_delta': (
            None if not (off_summary and on_summary) else
            round(on_summary['pass1_us_mean'] - off_summary['pass1_us_mean'], 3)),
        'pass1_ratio': (
            None if not (off_summary and on_summary) or not off_summary['pass1_us_mean'] else
            round(on_summary['pass1_us_mean'] / off_summary['pass1_us_mean'], 6)),
    }
    checks = {
        'on_arm_used_skip_kernel': bool(skip_calls),
        'on_arm_has_skip_bucket': on_summary is not None,
        'off_arm_has_fine_bucket': off_summary is not None,
        'generated_ids_equal': off_ids is not None and off_ids == on_ids,
        'generated_ids_count_off': None if off_ids is None else len(off_ids),
        'generated_ids_count_on': None if on_ids is None else len(on_ids),
    }
    doc['checks'] = checks
    doc['status'] = 'pass' if all(v for k, v in checks.items() if isinstance(v, bool)) else 'fail'
    doc['finished_unix'] = int(time.time())
    pathlib.Path(args.json).write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps({'status': doc['status'], 'checks': checks,
                      'measurements': doc['measurements']}, indent=1))
    return 0 if doc['status'] == 'pass' else 3


if __name__ == '__main__':
    sys.exit(main())
