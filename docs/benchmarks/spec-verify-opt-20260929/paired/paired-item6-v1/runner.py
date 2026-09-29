"""Speculative-verification optimization A/B evidence collector.

Adapted from mtp-dflash2-hybrid-active-20260929/runner.py (same methodology:
GPU lease, service stop/restore, privileged CPU controller with APERF/MPERF
readback, per-group server loads with warmup, token ID capture and comparison).

Differences:
- Modes come from --modes-json: {name: {"env": {...}, "libs": "live"|"snapshot",
  "cuda": "live"|"snapshot", "server_args": [...]}}. "env" entries are exported
  verbatim; SPEC_OPT_*/GGML_* toggles select code paths inside one binary.
  "libs"="snapshot" points LD_PRELOAD/LD_LIBRARY_PATH at the preserved baseline
  snapshot so baseline and candidate never share a modified library.
- Server config is fixed: MTP n4, p_min 0.70, no auxiliary DFlash, CUDA0.
- A fourth, less repetitive case (prosa192) joins the three reference prompts.
- Each group records the libraries actually mapped by the server process
  (/proc/<pid>/maps) with their sha256, so loaded-library identity is evidence.

Examples (run manually):
  python3 runner.py smoke --run-id smoke-v1 --modes-json modes-smoke.json
  python3 runner.py paired --run-id paired-v1 --modes-json modes-item1.json
"""

import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


SCRIPTS_ROOT = Path(__file__).resolve().parent
BASE = Path('/home/hjotha/beellama.cpp')
SNAPSHOT = Path('/home/hjotha/beellama-baseline-spec-verify-20260929')
MODEL = Path('/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf')
PORT = 59595
APU_DEVICE = Path('/sys/class/drm/card1/device')
SERVICES = (
    ('root', 'llama-server-root.service', 8090),
    ('user', 'qwen35-4b-mtp-8092.service', 8092),
)
SHORT = 'Escreva os numeros inteiros de 1 a 200 separados por virgula, sem comentarios.'
CODE = 'Escreva uma funcao Python merge_sort que recebe lista de inteiros. Inclua um exemplo e explique a complexidade em portugues.'
PROSA = ('Explique em portugues, com suas palavras, por que o ceu e azul durante o dia e '
         'avermelhado no por do sol. Inclua dois exemplos do cotidiano e uma analogia '
         'simples. Nao use listas nem repeticao de frases.')
LONG = 'Leia os registros e depois explique as tres principais tendencias, com exemplos concretos.\n' + ''.join(
    f'Registro {i:04d}: setor {i%23}, lote {i*17%997}, temperatura {20+i%17} C, demanda {i*31%509} unidades; revisao aprovada com checksum {i*131%10007:04d}.\n'
    for i in range(320)) + '\nResumo dos registros:'
CASES = {
    'warmup64': dict(prompt=CODE, n_predict=64, warmup=True),
    'curto192': dict(prompt=SHORT, n_predict=192, warmup=False),
    'codigo192': dict(prompt=CODE, n_predict=192, warmup=False),
    'prosa192': dict(prompt=PROSA, n_predict=192, warmup=False),
    'long192': dict(prompt=LONG, n_predict=192, warmup=False),
}
SHORT_CASES = ('curto192', 'codigo192', 'prosa192')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def append_json(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_modes(path):
    modes = json.loads(Path(path).read_text())
    if not isinstance(modes, dict) or not modes:
        raise ValueError('modes JSON must be a non-empty object')
    for name, config in modes.items():
        if not isinstance(config, dict):
            raise ValueError(f'mode {name}: config must be an object')
        for key in config:
            if key not in ('env', 'libs', 'cuda', 'server_args'):
                raise ValueError(f'mode {name}: unknown key {key}')
        if config.get('libs', 'live') not in ('live', 'snapshot'):
            raise ValueError(f'mode {name}: libs must be live|snapshot')
        if config.get('cuda', 'live') not in ('live', 'snapshot'):
            raise ValueError(f'mode {name}: cuda must be live|snapshot')
        config.setdefault('env', {})
        config.setdefault('libs', 'live')
        config.setdefault('cuda', 'live')
        config.setdefault('server_args', [])
    return modes


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('phase', choices=('smoke', 'paired'))
    parser.add_argument('--run-id', default=None, help='unique name inside the phase directory')
    parser.add_argument('--modes-json', required=True, help='JSON file mapping mode names to configs')
    parser.add_argument('--modes', nargs='+', help='subset/order of modes from the JSON (default: JSON order)')
    parser.add_argument('--repeats', type=int, help='per-mode short-case repetitions: smoke=1, paired=5 minimum')
    parser.add_argument('--long-repeats', type=int, help='per-mode long192 repetitions: smoke=1, paired=5 minimum')
    parser.add_argument('--group-size', type=int, default=2, help='paired repetitions per mode per load (default 2)')
    parser.add_argument('--scripts-root', type=Path, default=SCRIPTS_ROOT,
                        help='directory containing the privileged cpu-controller.py helper')
    args = parser.parse_args()
    all_modes = load_modes(args.modes_json)
    args.mode_configs = all_modes
    selected = [mode for entry in (args.modes or list(all_modes)) for mode in entry.split(',')]
    if not selected or len(set(selected)) != len(selected) or any(m not in all_modes for m in selected):
        parser.error('--modes must contain unique names from the JSON: ' + ', '.join(all_modes))
    args.modes = selected
    minimum = 1 if args.phase == 'smoke' else 5
    args.repeats = minimum if args.repeats is None else args.repeats
    args.long_repeats = minimum if args.long_repeats is None else args.long_repeats
    if min(args.repeats, args.long_repeats) < minimum:
        parser.error(f'{args.phase} requires at least {minimum} repetition(s) per case per mode')
    if args.group_size < 1:
        parser.error('--group-size must be positive')
    args.run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
    if args.run_id in ('.', '..') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.' for c in args.run_id):
        parser.error('--run-id must be a single filename using letters, numbers, dots, hyphens or underscores')
    args.modes_json = Path(args.modes_json).resolve()
    args.scripts_root = args.scripts_root.resolve()
    if not (args.scripts_root / 'cpu-controller.py').is_file():
        parser.error('cpu-controller.py is missing from --scripts-root')
    return args


def schedule(args):
    maximum = max(args.repeats, args.long_repeats)
    size = maximum if args.phase == 'smoke' else args.group_size
    groups = []
    for round_index, start in enumerate(range(0, maximum, size)):
        offset = (round_index // 2) % len(args.modes) if len(args.modes) > 2 else 0
        order = args.modes[offset:] + args.modes[:offset]
        if round_index % 2:
            order = list(reversed(order))
        for mode in order:
            requests = [dict(case='warmup64', repeat=None, scenario='warmup64')]
            for repeat in range(start, min(start + size, maximum)):
                for case in (*SHORT_CASES, 'long192'):
                    limit = args.long_repeats if case == 'long192' else args.repeats
                    if repeat < limit:
                        requests.append(dict(case=case, repeat=repeat, scenario=f'{case}-r{repeat:03d}'))
            groups.append(dict(group=f'g{len(groups):03d}-{mode}', round=round_index,
                               mode=mode, requests=requests))
    return groups


def library_roots(config):
    if config['libs'] == 'snapshot':
        dl = SNAPSHOT / 'dl-bin'
    else:
        dl = BASE / 'build-dflash-xbox-dl/bin'
    if config['cuda'] == 'snapshot':
        backends = SNAPSHOT / 'build-bin'
    else:
        backends = BASE / 'build/bin'
    return dl, backends


def server_command(config):
    dl, _ = library_roots(config)
    cmd = [str(dl / 'llama-server'), '--model', str(MODEL), '--device', 'CUDA0', '--n-gpu-layers', '99',
           '--fit', 'off', '--flash-attn', 'on', '--ctx-size', '16384', '--parallel', '1',
           '--batch-size', '64', '--ubatch-size', '64', '--cache-type-k', 'q4_0', '--cache-type-v', 'q4_0',
           '--host', '127.0.0.1', '--port', str(PORT), '--no-webui', '--spec-type', 'draft-mtp',
           '--spec-draft-n-max', '4', '--spec-draft-p-min', '0.70',
           '--spec-draft-type-k', 'q4_0', '--spec-draft-type-v', 'q4_0']
    cmd += [str(entry) for entry in config['server_args']]
    return cmd + ['--gpu-power-backend', 'amdgpu', '--apu-tdp', '20',
                  '--gpu-mem-clock-prefill', '2700', '--gpu-mem-clock-decode', '2700']


def server_environment(config):
    dl, backends = library_roots(config)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('DFLASH_XBOX_', 'GGML_MTP_PROF', 'GGML_DFLASH_LOCAL_PROF', 'GGML_DFLASH_SHADOW_',
                                  'SPEC_OPT_', 'GGML_GDN_'))
           and key != 'SUDO_PASS'}
    # Strict isolation: only the selected roots are searched, so a snapshot run
    # can never fall back to a modified live library (missing libs fail loudly).
    env['LD_LIBRARY_PATH'] = f'{backends}:{dl}'
    env['LD_PRELOAD'] = ':'.join(str(dl / name) for name in
                                 ('libllama.so.0.4.7', 'libllama-common.so.0.4.7', 'libllama-server-impl.so'))
    env['GGML_BACKEND_PATH'] = str(backends / 'libggml-vulkan.so')
    for name, value in config['env'].items():
        env[name] = str(value)
    return env


def capture(command, timeout=15, **kwargs):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout, **kwargs)
        return dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(command=command, error=str(exc))


def loaded_libraries(directory, pid):
    """Record every mapped library under BASE/SNAPSHOT with its sha256."""
    paths = set()
    with open(f'/proc/{pid}/maps') as maps:
        for line in maps:
            if '.so' not in line:
                continue
            fields = line.rstrip('\n').split(maxsplit=5)
            if len(fields) < 6:
                continue
            path = fields[5]
            if path.startswith((str(BASE) + '/', str(SNAPSHOT) + '/')):
                paths.add(path)
    out = {}
    for path in sorted(paths):
        resolved = str(Path(path).resolve())
        try:
            out[path] = dict(resolved=resolved, sha256=sha256(Path(resolved)))
        except OSError as exc:
            out[path] = dict(resolved=resolved, error=str(exc))
    write_json(directory / 'loaded-libraries.json', out)
    return out


def provenance(root, args):
    write_json(root / 'git.json', {label: capture(['git', *command], cwd=BASE) for label, command in (
        ('head', ['rev-parse', 'HEAD']), ('branch', ['branch', '--show-current']),
        ('status', ['status', '--short']), ('diff', ['diff', '--binary', 'HEAD']))})
    sources = [BASE / name for name in ('common/speculative.cpp', 'common/sampling.cpp',
                                      'tools/server/server-context.cpp', 'tools/server/server-task.h',
                                      'tools/server/server-adaptive-dm.h', 'src/llama-context.cpp',
                                      'src/llama-graph.cpp', 'common/common.h', 'common/arg.cpp')]
    for source in sources:
        dest = root / 'source' / source.relative_to(BASE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    (root / 'runner.py').write_bytes(Path(__file__).read_bytes())
    (root / 'cpu-controller.py').write_bytes((args.scripts_root / 'cpu-controller.py').read_bytes())
    (root / 'modes.json').write_bytes(args.modes_json.read_bytes())
    write_json(root / 'source-sha256.json', {str(p.relative_to(root)): sha256(p) for p in
               [root / 'runner.py', root / 'cpu-controller.py', root / 'modes.json',
                *sorted((root / 'source').rglob('*.*'))]})
    binaries = set()
    for directory in (BASE / 'build/bin', BASE / 'build-dflash-xbox-dl/bin',
                      SNAPSHOT / 'dl-bin', SNAPSHOT / 'build-bin'):
        if directory.is_dir():
            binaries.update(p for p in directory.glob('*.so*') if p.is_file())
            server = directory / 'llama-server'
            if server.is_file():
                binaries.add(server)
    write_json(root / 'binary-sha256.json', {str(p): dict(resolved=str(p.resolve()), sha256=sha256(p))
                                           for p in sorted(binaries)})
    write_json(root / 'model-sha256.json', {str(MODEL): dict(size=MODEL.stat().st_size, sha256=sha256(MODEL))})
    write_json(root / 'host.json', dict(hostname=socket.gethostname(), uname=list(os.uname()),
                                       python=sys.version, lscpu=capture(['lscpu']),
                                       apu_device=str(APU_DEVICE.resolve())))


class CPUController:
    def __init__(self, root, scripts_root):
        self.root = root
        self.helper = scripts_root / 'cpu-controller.py'
        self.process = None
        self.log = None
        self.ready = None
        self.lock = threading.Lock()

    def receive(self):
        if not select.select([self.process.stdout], [], [], 15)[0]:
            raise RuntimeError('CPU controller response timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('CPU controller stopped unexpectedly')
        response = json.loads(line)
        if response.get('error'):
            raise RuntimeError(response['error'])
        return response

    def start(self, password):
        self.log = (self.root / 'cpu-controller.stderr.log').open('w')
        self.process = subprocess.Popen(['sudo', '-k', '-S', '-p', '', 'python3', str(self.helper)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        text=True, bufsize=1, start_new_session=True)
        self.process.stdin.write(password + '\n')
        self.process.stdin.flush()
        self.ready = self.receive()
        write_json(self.root / 'cpu-original.json', self.ready)
        if not self.ready.get('ready'):
            raise RuntimeError(str(self.ready))
        self.nominal = {str(k): float(v) for k, v in self.ready['nominal_mhz'].items()}
        if not self.nominal or len(set(self.nominal.values())) != 1:
            raise RuntimeError('Mixed or absent nominal CPU frequencies are unsupported')
        self.reference = next(iter(self.nominal.values()))

    def call(self, action):
        with self.lock:
            self.process.stdin.write(json.dumps({'action': action}) + '\n')
            self.process.stdin.flush()
            return self.receive()

    def close(self):
        try:
            if self.process is not None:
                try:
                    if self.ready and self.ready.get('ready'):
                        state = self.call('close')
                        write_json(self.root / 'cpu-restored.json', state)
                        if state.get('restored') != self.ready['original']:
                            raise RuntimeError('CPU restore readback mismatch')
                finally:
                    try:
                        self.process.stdin.close()  # EOF also restores inside the helper.
                    finally:
                        self.process.wait(timeout=30)
                        self.process.stdout.close()
                if self.process.returncode:
                    raise RuntimeError(f'CPU helper exit {self.process.returncode}; inspect its stderr')
        finally:
            if self.log is not None:
                self.log.close()


def service_state(scope, name):
    cmd = ['systemctl'] + (['--user'] if scope == 'user' else []) + ['is-active', name]
    response = capture(cmd, timeout=5)
    state = response.get('stdout', '').strip()
    if state not in ('active', 'inactive', 'failed'):
        raise RuntimeError(f'Cannot establish stable service state: {response}')
    return state


def service_change(scope, name, action, password):
    cmd = (['sudo', '-S', '-p', '', 'systemctl'] if scope == 'root' else ['systemctl', '--user'])
    result = subprocess.run(cmd + [action, name], input=password + '\n' if scope == 'root' else None,
                            text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f'{name} {action}: {result.stderr.strip()}')


def check_port():
    with socket.socket() as probe:
        # Permit TIME_WAIT from the preceding group, while rejecting a listener.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', PORT))


def health(port):
    result = subprocess.run(['curl', '-fsS', '--retry', '120', '--retry-delay', '1',
                             '--retry-connrefused', '--retry-all-errors', '--max-time', '5',
                             f'http://127.0.0.1:{port}/health'], capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-1000:])


def check_policy(policy):
    for scope, name, _ in SERVICES:
        if service_state(scope, name) == 'active':
            raise RuntimeError('Production resumed during benchmark; discard run: ' + name)
    for directory, expected in policy['policy'].items():
        actual = {key: (Path(directory) / key).read_text().strip() for key in expected}
        if actual != expected:
            raise RuntimeError('CPU policy changed during benchmark: ' + directory)


def sensor_snapshot(hwmon, start):
    out = {'t_s': time.monotonic() - start}
    for key, path, scale in (
        ('gpu_clock_mhz', hwmon / 'freq1_input', 1e6), ('apu_ppt_w', hwmon / 'power1_average', 1e6),
        ('temp_c', hwmon / 'temp1_input', 1e3), ('gpu_busy_pct', APU_DEVICE / 'gpu_busy_percent', 1),
    ):
        try:
            out[key] = int(path.read_text()) / scale
        except (OSError, ValueError) as exc:
            out.setdefault('sensor_errors', {})[key] = str(exc)
    return out


def nvidia_snapshot():
    return capture(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,power.limit,power.draw,clocks.sm,clocks.max.sm,temperature.gpu,memory.used,memory.total', '--format=csv'])


def hardware_snapshot(path, hwmon, cpu):
    write_json(path, dict(utc=datetime.now(timezone.utc).isoformat(),
        nvidia=nvidia_snapshot(), apu=sensor_snapshot(hwmon, time.monotonic()),
        cpu=cpu.call('sample')))


def collect_request(root, directory, group, index, item, cpu, policy, hwmon, mode_config):
    check_policy(policy)
    case = CASES[item['case']]
    stem = directory / item['scenario']
    identity = dict(mode=group['mode'], group=group['group'], round=group['round'],
                    request_index=index, case=item['case'], scenario=item['scenario'], repeat=item['repeat'],
                    warmup=case['warmup'], n_predict=case['n_predict'])
    payload = dict(prompt=case['prompt'], n_predict=case['n_predict'], temperature=0, seed=42,
                   ignore_eos=True, cache_prompt=False, stream=False, return_tokens=True)
    write_json(stem.with_suffix('.request.json'), payload)
    append_json(root / 'requests.jsonl', dict(identity, event='begin', utc=datetime.now(timezone.utc).isoformat()))
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/completion', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    samples, monitor_errors = [], []
    stop = threading.Event()
    t0 = time.monotonic()

    def monitor():
        try:
            previous = cpu.call('sample')
            while not stop.wait(.1):
                current = cpu.call('sample')
                dt = current['time'] - previous['time']
                if dt <= 0:
                    raise RuntimeError('Nonpositive APERF/MPERF sample interval')
                sample = sensor_snapshot(hwmon, t0)
                total_a = total_m = 0
                per_cpu = {}
                for core, counts in current['counters'].items():
                    a = (counts[0] - previous['counters'][core][0]) & ((1 << 64) - 1)
                    m = (counts[1] - previous['counters'][core][1]) & ((1 << 64) - 1)
                    if m:
                        mhz = cpu.nominal[core] * a / m
                        if mhz > 10000:
                            raise RuntimeError('Invalid APERF/MPERF sample')
                        per_cpu[core] = dict(busy_mhz=mhz, busy_pct=100 * m / (dt * cpu.nominal[core] * 1e6))
                    total_a += a
                    total_m += m
                sample.update(cpu_aperf_delta=total_a, cpu_mperf_delta=total_m,
                              cpu_busy_mhz=cpu.reference * total_a / total_m if total_m else None,
                              cpu_util_pct=100 * total_m / (dt * cpu.reference * 1e6 * len(cpu.nominal)),
                              cpu_per_core=per_cpu)
                samples.append(sample)
                previous = current
        except Exception as exc:
            monitor_errors.append(str(exc))

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                raw = response.read()
            stem.with_suffix('.response.json').write_bytes(raw)
            data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            stem.with_suffix('.http-error.txt').write_bytes(exc.read())
            raise
    finally:
        elapsed = time.monotonic() - t0
        stop.set()
        thread.join(timeout=20)
        write_json(stem.with_suffix('.telemetry.json'), samples)
        if thread.is_alive():
            monitor_errors.append('CPU monitor did not stop')
        write_json(stem.with_suffix('.monitor.json'), dict(errors=monitor_errors, wall_s=elapsed))
    tokens = data.get('tokens')
    write_json(stem.with_suffix('.tokens.json'), tokens)
    if data.get('error'):
        raise RuntimeError(str(data['error']))
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
        raise RuntimeError('Missing or invalid output token IDs: ' + str(stem))
    if len(tokens) != case['n_predict']:
        raise RuntimeError(f'Incomplete response: {stem}: {len(tokens)} tokens, expected {case["n_predict"]}')
    if monitor_errors:
        raise RuntimeError('CPU monitor failed: ' + '; '.join(monitor_errors))
    check_policy(policy)
    timings = data.get('timings', {})
    decode = [s for s in samples if s['t_s'] > timings.get('prompt_ms', 0) / 1000]
    active = [s for s in decode if s.get('gpu_busy_pct', 0) >= 10]

    def aggregate(rows, key, fn=statistics.mean):
        values = [s[key] for s in rows if s.get(key) is not None]
        return round(fn(values), 3) if values else None

    sum_a = sum(s['cpu_aperf_delta'] for s in decode)
    sum_m = sum(s['cpu_mperf_delta'] for s in decode)
    if not sum_m:
        raise RuntimeError('No effective CPU frequency samples recorded: ' + str(stem))
    result = dict(identity, config={k: v for k, v in mode_config.items()}, tdp_w=20, context_size=16384,
                  prompt_n=timings.get('prompt_n'), prompt_ms=timings.get('prompt_ms'),
                  prompt_tps=timings.get('prompt_per_second'), predicted_n=timings.get('predicted_n'),
                  decode_tps=timings.get('predicted_per_second'), decode_ms=timings.get('predicted_ms'),
                  wall_s=round(elapsed, 3), token_count=len(tokens),
                  content_sha256=hashlib.sha256(data.get('content', '').encode()).hexdigest(),
                  tokens_sha256=hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest(),
                  response_file=str(stem.with_suffix('.response.json').relative_to(root)),
                  tokens_file=str(stem.with_suffix('.tokens.json').relative_to(root)),
                  server_output=str((directory / 'server-output.txt').relative_to(root)),
                  decode_ppt_w=aggregate(decode, 'apu_ppt_w'), active_gpu_clock_mhz=aggregate(active, 'gpu_clock_mhz'),
                  peak_temp_c=aggregate(samples, 'temp_c', max), cpu_busy_mhz=round(cpu.reference * sum_a / sum_m, 3),
                  cpu_util_pct=aggregate(decode, 'cpu_util_pct'), telemetry_samples=len(samples))
    write_json(stem.with_suffix('.result.json'), result)
    append_json(root / 'requests.jsonl', dict(identity, event='end', response_file=result['response_file']))
    return result, tokens


def token_comparison(reference, actual, reference_tokens, actual_tokens, kind):
    common = min(len(reference_tokens), len(actual_tokens))
    first = next((i for i in range(common) if reference_tokens[i] != actual_tokens[i]), None)
    if first is None and len(reference_tokens) != len(actual_tokens):
        first = common
    return dict(kind=kind, case=actual['case'], repeat=actual['repeat'],
                reference_mode=reference['mode'], actual_mode=actual['mode'],
                reference_file=reference['tokens_file'], actual_file=actual['tokens_file'],
                reference_count=len(reference_tokens), actual_count=len(actual_tokens), equal=first is None,
                first_difference=first, index_base=0,
                reference_token=reference_tokens[first] if first is not None and first < len(reference_tokens) else None,
                actual_token=actual_tokens[first] if first is not None and first < len(actual_tokens) else None)


def record_comparisons(root, history, result, tokens):
    comparisons = []
    same_case = [(row, previous) for row, previous in history if row['case'] == result['case']]
    repeat_reference = next(((row, previous) for row, previous in same_case if row['mode'] == result['mode']), None)
    if repeat_reference:
        comparisons.append(token_comparison(repeat_reference[0], result, repeat_reference[1], tokens, 'within-mode-repeat'))
    for row, previous in same_case:
        if (row['mode'] != result['mode'] and row['repeat'] == result['repeat']
                and (not result['warmup'] or row['round'] == result['round'])):
            comparisons.append(token_comparison(row, result, previous, tokens, 'across-mode-same-repeat'))
    for comparison in comparisons:
        append_json(root / 'token-comparisons.jsonl', comparison)
        if not comparison['equal']:
            print('TOKEN_DIVERGENCE ' + json.dumps(comparison), flush=True)
    history.append((result, tokens))
    return sum(not c['equal'] for c in comparisons)


def run_group(root, group, cpu, hwmon, results, history, status, mode_config):
    directory = root / 'groups' / group['group']
    directory.mkdir(parents=True, exist_ok=False)
    env = server_environment(mode_config)
    command = server_command(mode_config)
    write_json(directory / 'plan.json', group)
    write_json(directory / 'command.json', command)
    write_json(directory / 'environment.json', {k: v for k, v in env.items()
               if k.startswith(('GGML_', 'LD_', 'DFLASH_', 'CUDA_', 'VK_', 'OMP_', 'HIP_', 'ROCR_', 'SPEC_'))
               or k == 'PATH'})
    policy = cpu.call('benchmark')
    write_json(directory / 'cpu-policy.json', policy)
    check_policy(policy)
    hardware_snapshot(directory / 'hardware-before.json', hwmon, cpu)
    with (directory / 'server-output.txt').open('xb') as log:
        check_port()
        started = time.monotonic()
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, cwd=BASE)
        forced_kill = False
        try:
            health(PORT)
            if proc.poll() is not None:
                raise RuntimeError('server exit ' + str(proc.returncode))
            write_json(directory / 'load.json', dict(pid=proc.pid, load_s=time.monotonic() - started))
            loaded_libraries(directory, proc.pid)
            print(group['group'] + ' loaded', flush=True)
            for index, item in enumerate(group['requests']):
                result, tokens = collect_request(root, directory, group, index, item, cpu, policy, hwmon, mode_config)
                if not result['warmup']:
                    results.append(result)
                    write_json(root / 'results.json', results)
                    print(json.dumps(result), flush=True)
                status['token_divergences'] += record_comparisons(root, history, result, tokens)
                status['measured_requests'] = len(results)
                write_json(root / 'status.json', status)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                forced_kill = True
                proc.kill()
                proc.wait(timeout=5)
            log.flush()
            write_json(directory / 'server-exit.json', dict(returncode=proc.returncode, forced_kill=forced_kill))
            hardware_snapshot(directory / 'hardware-after.json', hwmon, cpu)
        if forced_kill or proc.returncode not in (0, -signal.SIGTERM):
            raise RuntimeError('Server did not shut down cleanly: ' + group['group'])


def interrupted(signum, _frame):
    raise KeyboardInterrupt(f'signal {signum}')


def main():
    args = arguments()
    groups = schedule(args)
    root = SCRIPTS_ROOT / args.phase / args.run_id
    with open('/tmp/beellama-gpu-benchmark.lock', 'a+') as lease:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another benchmark holds the GPU lease')
        check_port()
        root.mkdir(parents=True, exist_ok=False)
        print('RESULT_DIR=' + str(root), flush=True)
        write_json(root / 'manifest.json', dict(schema_version=2, phase=args.phase, run_id=args.run_id,
            modes={name: args.mode_configs[name] for name in args.modes}, repeats=args.repeats,
            long_repeats=args.long_repeats, group_size=args.group_size, port=PORT,
            scripts_root=str(args.scripts_root), modes_json=str(args.modes_json),
            snapshot_root=str(SNAPSHOT), groups=groups, argv=sys.argv,
            created_utc=datetime.now(timezone.utc).isoformat(),
            comparison_policy='Report every first token divergence; investigate before attributing to noise.'))
        write_json(root / 'cases.json', CASES)
        write_json(root / 'results.json', [])
        (root / 'token-comparisons.jsonl').touch(exist_ok=False)
        status = dict(state='preparing', token_divergences=0)
        write_json(root / 'status.json', status)
        cpu = CPUController(root, args.scripts_root)
        restore_services = []
        results, history = [], []
        password = None
        previous_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            provenance(root, args)
            hwmons = sorted((APU_DEVICE / 'hwmon').glob('hwmon*'))
            if len(hwmons) != 1:
                raise RuntimeError('Expected one APU hwmon at ' + str(APU_DEVICE))
            hwmon = hwmons[0]
            original_services = {name: service_state(scope, name) for scope, name, _ in SERVICES}
            write_json(root / 'services-original.json', original_services)
            password = getpass.getpass('sudo password: ')
            cpu.start(password)
            hardware_snapshot(root / 'hardware-original.json', hwmon, cpu)
            for scope, name, port in SERVICES:
                if original_services[name] == 'active':
                    restore_services.append((scope, name, port))  # Include partially failed stops.
                    service_change(scope, name, 'stop', password)
                    print(name + ' stopped', flush=True)
            status['state'] = 'running'
            write_json(root / 'status.json', status)
            for group in groups:
                run_group(root, group, cpu, hwmon, results, history, status, args.mode_configs[group['mode']])
            expected = len(args.modes) * (len(SHORT_CASES) * args.repeats + args.long_repeats)
            if len(results) != expected:
                raise RuntimeError(f'Incomplete collection: {len(results)}/{expected} measured requests')
            status.update(state='collected', measured_requests=len(results))
        except BaseException as exc:
            status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            failures = []
            try:
                cpu.close()
            except Exception as exc:
                failures.append('CPU restore: ' + str(exc))
            for scope, name, port in restore_services:
                try:
                    service_change(scope, name, 'start', password)
                    health(port)
                    if service_state(scope, name) != 'active':
                        raise RuntimeError('service did not remain active')
                    print(name + ' restored', flush=True)
                except Exception as exc:
                    failures.append(name + ': ' + str(exc))
            write_json(root / 'restoration.json', dict(errors=failures,
                       services_requiring_restore=[name for _, name, _ in restore_services]))
            status['restoration_errors'] = failures
            if failures:
                status['state'] = 'restore-failed'
            elif status['state'] == 'collected':
                status['state'] = 'complete'
            write_json(root / 'status.json', status)
            signal.signal(signal.SIGTERM, previous_handler)
            if failures:
                raise RuntimeError('RESTORE FAILED: ' + '; '.join(failures))
        print('RESULT_DIR=' + str(root), flush=True)
        print('TOKEN_DIVERGENCES=' + str(status['token_divergences']), flush=True)


if __name__ == '__main__':
    main()
