#!/usr/bin/env python3
"""Qualify native auto-restore: full-vocabulary first-token logits and 128/129 rewind.

Inert without --execute (prints the plan, touches nothing). With --execute it
reuses the already qualified 111999-token host snapshot and the six Xbox
objects, pauses production only after idle/identity rechecks, restarts it in
finally, and never writes to the Xbox store.

Two fresh private processes run the same request sequence:
  live     : cold 112000-token prefill, auto-restore disabled, empty store
  restored : auto-restore from the 111999-token snapshot (fresh session)
Q0 asks for n_probs=n_vocab at the sampled position and compares the full
distribution. Q2 rewinds 128 rows and re-appends 128 tokens, Q2b repeats it
(16-row rewind), and Q1 requires removing 129 rows.

HTTP has no rewind action: a request whose prompt shares only K resident tokens
drives server-context.cpp keep_first(K) + seq_rm_suffix, which reaches
rkva_native_rewind_locked and the revision-3 range-rewind RPC (<= 128 rows).
The script asserts the computed removal before each request so the probes can
never silently exercise the wrong boundary.
"""
import argparse
import base64
import fcntl
import getpass
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path('/home/hjotha/beellama-benchmark-results/xxxl-plan-execution-20261030')
RELEASE = Path('/home/hjotha/beellama-releases/xxxl-native-69e69dae5-20261031T212909Z')
REPO = Path('/home/hjotha/worktrees/beellama-native-deploy-20261031')
SERVICE = 'llama-server-root.service'
PRESET = Path('/home/hjotha/router-production.ini')
CORPUS = Path('/var/tmp/corpus_tokens_163840.json')
CODE = Path('/home/hjotha/beellama-benchmark-results/campaign-code-agentic-20261029/S_code.prompt.json')
MODEL = Path('/home/hjotha/models/Huihui-Qwen3.8-27B-abliterated-Swift-1.5-GSQ-RCO-IQ3_XXS-mtp.gguf')
MODEL_SHA = 'a7e8962f16a84945b2277663da41824b936d40ae90d55852718ac389431c9166'
XBOX_HOST = '192.168.1.200'
PFN = 'GianlucaMazza.xllama_1.6.0.467_x64__pj67f1fcj4n14'
EXPECTED_BUILD = 'b13306-69e69dae5'
EXPECTED_SHA = 'f701499cbd510731f4c931dc0e78c238eeedce1945fc0f8738e089ffa5a71c77'
REMOTE_DIR = 'rkva-snapshots-v2'
SNAPSHOT_NAME = 'native-code-prefix-111999.bin'
SOURCE = ROOT / 'native-auto-112k-r1/store/native-code-prefix-111999.bin'
PRIOR = ROOT / 'native-auto-restore-only-r1/result.json'
QUAL = ROOT / 'native-auto-112k-r1/result.json'
PREFIX = 111999          # tokens stored by the snapshot (positions 0..111998)
PROMPT_N = 112000        # code prompt; the snapshot is a strict prefix
SPLIT_P = 102400
VOCAB = 248320
REWIND = 128
CUT = PROMPT_N - REWIND - 1   # 111871: 129 cached rows removed when the request diverges here
PROTECTED = [
    Path('/home/hjotha/llama-slot-cache/auto-qwen-p102400/auto-4998744b417fa209-c570b6cc29dc6bd8-102399.bin'),
    ROOT / 'native-fifth-112k-r1/store/native-code-112k.snapshot',
    ROOT / 'native-fifth-restore-r1/store/native-code-112k.snapshot',
    SOURCE,
    Path('/home/hjotha/llama-slot-cache/auto-qwen-p102400/native-110k-20261030.snapshot'),
    Path('/home/hjotha/llama-slot-cache/auto-qwen-p102400/native-code-112k-plus1023-20261030.snapshot'),
]
# Why these numbers: live and restored run the same binary/model/prompt, so the
# only expected difference is float32 reduction order when restored KVarN stage
# planes/tail are re-materialised. A real state error (missing/wrong row) moves
# logits by O(0.1-1) and usually flips the argmax (KL >> 1e-2 nats), so 1e-4
# nats / 1e-3 TV / 0.05 max logprob delta sits ~100x above float noise and
# ~1000x below a real mismatch. Prior teacher-forced evidence was bit-identical.
TOL = {'kl': 1e-4, 'tv': 1e-3, 'dlogp': 5e-2, 'mass': 1e-3, 'top1_prob': 0.02}


class Unsupported(RuntimeError):
    """The server lacks the observability or capability this qualification needs."""


def require(cond, msg, exc=RuntimeError):
    if not cond:
        raise exc(msg)


def read_json(path):
    return json.loads(Path(path).read_text())


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def sha(path, deadline=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            if deadline is not None:
                require(time.monotonic() < deadline, 'time budget expired during hashing')
            digest.update(block)
    return digest.hexdigest()


def build_prompt(corpus, code, total=PROMPT_N, split=SPLIT_P, suffix=2048):
    require(isinstance(corpus, list) and isinstance(code, dict) and isinstance(code.get('tokens'), list),
            'invalid corpus or code token source', ValueError)
    tail = code['tokens'][-suffix:]
    require(len(tail) == suffix, 'code prompt is shorter than its suffix', ValueError)
    prompt = corpus[:total - suffix] + tail
    require(len(prompt) == total and prompt[:split] == corpus[:split] and
            all(type(t) is int and t >= 0 for t in prompt), 'code prompt differs', ValueError)
    return prompt


def lcp(left, right):
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def rewind_plan(cached_tokens, cached_rows, request):
    """Mirror server-context.cpp: n_past = LCP; if n_past == len(request), the
    server drops one more token to obtain logits ([TAG_PROMPT_LOGITS]); the
    suffix removal is rows - n_past. Native revision-3 accepts <= 128 rows."""
    common = lcp(cached_tokens, request)
    n_past = common - 1 if common == len(request) and common > 0 else common
    require(cached_rows >= n_past, 'request keeps more rows than the cached state', ValueError)
    return {'common_prefix': common, 'n_past': n_past, 'removed': cached_rows - n_past}


def divergent_probe(evaluated, keep, token):
    """Request over the evaluated prefix that keeps `keep` rows and replaces the
    next token. Length equals the prefix, so len(request) > keep and the server
    applies no extra TAG_PROMPT_LOGITS decrement: the removal is exactly
    len(prefix) - keep."""
    require(isinstance(evaluated, list) and 0 < keep < len(evaluated), 'invalid rewind keep', ValueError)
    require(type(token) is int and token >= 0 and token != evaluated[keep],
            'probe token does not diverge', ValueError)
    probe = evaluated[:keep] + [token] + evaluated[keep + 1:]
    require(len(probe) == len(evaluated), 'probe length changed', ValueError)
    return probe


def pick_token(sequence, forbidden):
    for token in sequence:
        if token not in forbidden:
            return token
    raise ValueError('no usable divergent token')


def distribution_from_top_logprobs(entries, n_vocab):
    require(isinstance(entries, list) and len(entries) == n_vocab,
            f'expected {n_vocab} logprobs, got {len(entries) if isinstance(entries, list) else entries!r}',
            ValueError)
    probs = [0.0] * n_vocab
    seen = bytearray(n_vocab)
    for entry in entries:
        token = entry.get('id')
        logprob = entry.get('logprob')
        require(type(token) is int and 0 <= token < n_vocab, 'invalid token id in logprobs', ValueError)
        require(not seen[token], 'duplicate token id in logprobs', ValueError)
        require(isinstance(logprob, (int, float)) and not isinstance(logprob, bool),
                'missing logprob', ValueError)
        seen[token] = 1
        probs[token] = 0.0 if logprob < -1e30 else math.exp(float(logprob))
    return probs


def distribution_metrics(left, right):
    require(len(left) == len(right) and left, 'distribution length differs', ValueError)
    mass_l, mass_r = math.fsum(left), math.fsum(right)
    support_loss = math.fsum(p for p, q in zip(left, right) if p > 0 and q <= 0)
    kl_lr = math.fsum(p * (math.log(p) - math.log(q)) for p, q in zip(left, right) if p > 0 and q > 0)
    kl_rl = math.fsum(q * (math.log(q) - math.log(p)) for p, q in zip(left, right) if p > 0 and q > 0)
    if support_loss > 0:
        kl_lr = kl_rl = math.inf
    shared = [(p, q) for p, q in zip(left, right) if p >= 1e-6 or q >= 1e-6]
    top1_l, top1_r = max(range(len(left)), key=left.__getitem__), max(range(len(right)), key=right.__getitem__)
    return {
        'mass_left': mass_l, 'mass_right': mass_r, 'support_loss': support_loss,
        'kl_left_right': kl_lr, 'kl_right_left': kl_rl, 'kl_symmetric': (kl_lr + kl_rl) * 0.5,
        'total_variation': 0.5 * math.fsum(abs(p - q) for p, q in zip(left, right)),
        'max_abs_logprob_delta': max((abs(math.log(p) - math.log(q)) for p, q in shared
                                      if p > 0 and q > 0), default=0.0),
        'top1_left': top1_l, 'top1_right': top1_r,
        'top1_prob_left': left[top1_l], 'top1_prob_right': right[top1_r],
        'n_nonzero_left': sum(1 for p in left if p > 0),
        'n_nonzero_right': sum(1 for q in right if q > 0),
    }


def gate_distributions(metrics):
    return [
        ('top1_agrees', metrics['top1_left'] == metrics['top1_right'],
         f"{metrics['top1_left']} vs {metrics['top1_right']}"),
        ('kl', metrics['kl_symmetric'] is not None and metrics['kl_symmetric'] <= TOL['kl'],
         f"{metrics['kl_symmetric']} <= {TOL['kl']}"),
        ('total_variation', metrics['total_variation'] <= TOL['tv'],
         f"{metrics['total_variation']:.3e} <= {TOL['tv']}"),
        ('max_abs_logprob_delta', metrics['max_abs_logprob_delta'] <= TOL['dlogp'],
         f"{metrics['max_abs_logprob_delta']:.3e} <= {TOL['dlogp']}"),
        ('mass_left', abs(metrics['mass_left'] - 1.0) <= TOL['mass'], f"{metrics['mass_left']:.9f}"),
        ('mass_right', abs(metrics['mass_right'] - 1.0) <= TOL['mass'], f"{metrics['mass_right']:.9f}"),
        ('support', metrics['support_loss'] == 0, f"lost={metrics['support_loss']:.3e}"),
        ('top1_prob', min(metrics['top1_prob_left'], metrics['top1_prob_right']) >= TOL['top1_prob'],
         f"{metrics['top1_prob_left']:.4f}/{metrics['top1_prob_right']:.4f}"),
    ]


def sampled_vram_min(lines):
    samples = []
    for line in lines:
        columns = line.split(',')
        require(len(columns) == 3, 'malformed VRAM sample')
        value = float(columns[1].strip())
        require(math.isfinite(value) and value >= 0, 'invalid VRAM sample')
        samples.append(value)
    require(samples, 'VRAM monitor produced no samples')
    return min(samples), len(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--label', required=True, help='new output directory under the benchmark root')
    parser.add_argument('--binary', default=str(RELEASE / 'bin/llama-server'))
    parser.add_argument('--expected-sha256', default=EXPECTED_SHA)
    parser.add_argument('--timeout', type=int, default=5400)
    parser.add_argument('--cleanup-timeout', type=int, default=900)
    args = parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.label), 'invalid label')
    require(re.fullmatch(r'[0-9a-f]{64}', args.expected_sha256), 'invalid expected SHA')
    require(1200 <= args.timeout <= 7200 and 120 <= args.cleanup_timeout <= 1800, 'invalid time budget')
    binary = Path(args.binary).resolve(strict=True)
    out = ROOT / args.label
    if not args.execute:
        print(json.dumps({
            'execute': False, 'out': str(out), 'binary': str(binary), 'expected_sha256': args.expected_sha256,
            'prompt_tokens': PROMPT_N, 'snapshot_prefix': PREFIX, 'n_probs': VOCAB,
            'sequence': {
                'q0': f'{PROMPT_N} tokens, n_predict=1, n_probs={VOCAB}',
                'q129': f'{PROMPT_N} tokens diverging at {CUT} (expected removal 129)',
                'q128': f'{PROMPT_N} tokens diverging at {CUT + 1} (expected removal 128), n_predict=16',
                'q128b': 'identical q128 repeat (expected removal 16), n_predict=1'},
            'tolerances': TOL, 'sessions': ['live cold prefill', 'fresh-process native auto-restore'],
            'status': 'PREVIEW'}, indent=2))
        return 0
    password = getpass.getpass('Senha sudo: ') if sys.stdin.isatty() else sys.stdin.readline().rstrip('\n')
    require(bool(password), 'sudo password required on stdin')
    deadline = time.monotonic() + args.timeout
    lock_fd = os.open(str(ROOT / '.native-restore-qualification.lock'),
                      os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        raise RuntimeError('another cooperating native qualification is running')
    out.mkdir(mode=0o700, exist_ok=False)
    store, store_live = out / 'store', out / 'store-live'
    store.mkdir(mode=0o700)
    store_live.mkdir(mode=0o700)
    result = {'status': 'RUNNING', 'scope': 'logits + rewind qualification of native auto-restore',
              'expected_pfn': PFN, 'expected_build': EXPECTED_BUILD, 'cleanup_errors': [], 'warnings': [],
              'checks': {}, 'timings_comparable': False, 'sessions': {}}
    state = {'process': None, 'monitor': None, 'monitor_output': None, 'paused': False,
             'pgid': None, 'flags': None, 'before': None, 'protected': {}}

    def write_result():
        (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    def remain(cap):
        left = deadline - time.monotonic()
        require(left > 0, 'time budget expired')
        return max(1, min(cap, left))

    def check(name, ok, detail):
        result['checks'][name] = {'ok': bool(ok), 'detail': detail}
        if not ok:
            raise RuntimeError(f'{name}: {detail}')

    def grade(name, ok, detail):
        result['checks'][name] = {'ok': bool(ok), 'detail': detail}

    def warn(name, detail):
        result['warnings'].append({'name': name, 'detail': detail})

    def sudo(*argv, binary_out=False):
        completed = subprocess.run(['sudo', '-S', '-p', '', *argv], input=(password + '\n').encode(),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=remain(180))
        require(completed.returncode == 0, 'privileged command failed: ' + argv[0])
        return completed.stdout if binary_out else completed.stdout.decode().strip()

    def post_raw(port, path, body, cap):
        request = urllib.request.Request(f'http://127.0.0.1:{port}{path}', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=remain(cap)) as response:
                return response.status, json.load(response), ''
        except urllib.error.HTTPError as exc:
            text = exc.read().decode('utf-8', 'replace')
            try:
                return exc.code, json.loads(text), text
            except ValueError:
                return exc.code, None, text

    def get(port, path, cap=30):
        with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=remain(cap)) as response:
            return json.load(response)

    def on_signal(signum, frame):
        raise KeyboardInterrupt('signal ' + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)

    gate = importlib.util.spec_from_file_location('rkva_gate', REPO / 'tools/xbox/scripts/rkva-quality-gate.py')
    quality = importlib.util.module_from_spec(gate)
    gate.loader.exec_module(quality)
    deploy = quality.DeployClient(REPO / 'tools/xbox/scripts/deploy.sh',
                                  Path('/home/hjotha/.config/xllama/xbox-env'), 60, xbox_ip=XBOX_HOST)
    credentials = deploy.environment()
    auth = base64.b64encode((credentials['XBOX_USER'] + ':' + credentials['XBOX_PASS']).encode()).decode()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                         urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))

    def portal(endpoint, params=None):
        url = f'https://{XBOX_HOST}:11443{endpoint}'
        if params is not None:
            url += '?' + urllib.parse.urlencode(params)
        return opener.open(urllib.request.Request(url, headers={'Authorization': 'Basic ' + auth}),
                           timeout=remain(120))

    def verify_pfn():
        with portal('/api/resourcemanager/processes') as response:
            document = json.load(response)
        entries = document.get('Processes') if isinstance(document, dict) else None
        require(isinstance(entries, list), 'Xbox process inventory malformed')
        running = [item for item in entries if isinstance(item, dict) and item.get('IsRunning') is True and
                   ('xllama' in str(item.get('ImageName', '')).lower() or
                    'xllama' in str(item.get('PackageFullName', '')).lower())]
        require(len(running) == 1 and running[0].get('PackageFullName') == PFN,
                f'PFN {PFN} is not the sole running xllama app')

    def current_flags():
        return sorted(n for n in deploy.list_names(PFN) if n.endswith('.flag'))

    def listing(subdir):
        response = deploy.run('list-localstate', PFN, subdir)
        require(response['returncode'] == 0, 'Xbox snapshot listing failed')
        document = json.loads(response['_stdout_full'])
        items = document.get('Items') if isinstance(document, dict) else None
        require(isinstance(items, list), 'Xbox snapshot listing malformed')
        return items

    def snapshot_names():
        items = listing(REMOTE_DIR)
        names = [item.get('Name') for item in items if isinstance(item, dict) and item.get('Type') == 16]
        require(len(names) == len(items) == len(set(names)) and
                all(isinstance(name, str) and re.fullmatch(r'[0-9a-f]{64}', name) for name in names),
                'Xbox snapshot store contains unexpected entries')
        return set(names)

    def remote_digest(directory, filename):
        params = {'knownfolderid': 'LocalAppData', 'packagefullname': PFN,
                  'path': '\\LocalState\\' + REMOTE_DIR + '\\' + directory, 'filename': filename}
        digest = hashlib.sha256()
        count = 0
        with portal('/api/filesystem/apps/file', params) as response:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                remain(120)
                digest.update(block)
                count += len(block)
        return {'bytes': count, 'sha256': digest.hexdigest()}

    def snapshot_inventory(names):
        inventory = {}
        for name in sorted(names):
            items = listing(REMOTE_DIR + '\\' + name)
            files = [item.get('Name') for item in items if isinstance(item, dict) and item.get('Type') != 16]
            require(len(files) == len(items) == len(set(files)) and
                    all(isinstance(f, str) and re.fullmatch(r'(?:manifest(?:-v2)?\.bin|(?:n-)?\d+\.kv)', f)
                        for f in files), 'unexpected snapshot file or directory')
            require('manifest.bin' in files or 'manifest-v2.bin' in files, 'snapshot manifest missing')
            inventory[name] = {file: remote_digest(name, file) for file in sorted(files)}
        return inventory

    def child(pid, privileged=False):
        raw = (sudo('cat', f'/proc/{pid}/task/{pid}/children') if privileged
               else Path(f'/proc/{pid}/task/{pid}/children').read_text())
        ports = []
        for item in raw.split():
            argv = (sudo('cat', f'/proc/{item}/cmdline', binary_out=True) if privileged
                    else Path(f'/proc/{item}/cmdline').read_bytes()).split(b'\0')
            if b'--port' in argv:
                ports.append((int(item), int(argv[argv.index(b'--port') + 1])))
        require(len(ports) == 1, 'expected exactly one private model child')
        return ports[0]

    def gpu_busy(cap=30):
        end = time.monotonic() + cap
        while True:
            rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                            '--format=csv,noheader,nounits'], text=True,
                                           timeout=remain(30)).strip()
            if not rows or time.monotonic() >= end:
                return rows
            time.sleep(1)

    def private_start(role, command, env, cwd, port):
        ready = threading.Event()
        listening = {'value': False}
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True, cwd=str(cwd))
        stat = Path(f'/proc/{process.pid}/stat').read_text()
        record = {'pid': process.pid, 'pgid': os.getpgid(process.pid),
                  'starttime': int(stat[stat.rfind(')') + 2:].split()[19])}
        require(record['pgid'] == process.pid and record['starttime'] > 1, 'private process identity invalid')
        state['process'], state['pgid'] = process, record
        (out / f'router-{role}-pid.json').write_text(json.dumps(record, indent=2))

        def drain():
            with (out / f'router-{role}.log').open('w', buffering=1) as log:
                for line in process.stdout:
                    log.write(line)
                    if f'listening on http://127.0.0.1:{port}' in line:
                        listening['value'] = True
                        ready.set()
            ready.set()
        threading.Thread(target=drain, daemon=True).start()
        require(ready.wait(remain(300)) and listening['value'], 'private router did not listen')

    def group_alive(pgid):
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False

    def stop_private():
        process, record = state['process'], state['pgid']
        if record is None:
            require(process is None, 'private process without identity')
            return {'state': 'not_started'}
        try:
            current = Path(f"/proc/{record['pid']}/stat").read_text()
            same = int(current[current.rfind(')') + 2:].split()[19]) == record['starttime']
        except FileNotFoundError:
            same = False
        if same:
            for sig, spins in ((signal.SIGTERM, 300), (signal.SIGKILL, 100)):
                try:
                    os.killpg(record['pgid'], sig)
                except ProcessLookupError:
                    break
                for _ in range(spins):
                    if not group_alive(record['pgid']):
                        break
                    time.sleep(0.1)
                if not group_alive(record['pgid']):
                    break
            require(not group_alive(record['pgid']), 'private process group did not exit')
        if process is not None:
            process.wait(timeout=10)
        state['process'], state['pgid'] = None, None
        return {'state': 'quiescent', 'recorded': record}

    def sampling(n_predict, prompt, n_probs=0):
        body = {'model': 'qwen-3.8-27b', 'prompt': prompt, 'n_predict': n_predict, 'id_slot': 0,
                'temperature': 0.0, 'seed': 42, 'top_k': 1, 'top_p': 1.0, 'min_p': 0.0,
                'repeat_penalty': 1.0, 'presence_penalty': 0.0, 'frequency_penalty': 0.0,
                'cache_prompt': True, 'ignore_eos': True, 'stream': False, 'return_tokens': True,
                'post_sampling_probs': False,
                'response_fields': ['completion_probabilities', 'timings', 'tokens', 'tokens_predicted',
                                    'error', 'truncated']}
        if n_probs:
            body['n_probs'] = n_probs
        return body

    def ask(port, body, cap):
        status, document, text = post_raw(port, '/completion', body, cap)
        if status != 200:
            return {'http_status': status, 'error': (document or {}).get('error', text[-800:])}
        require(isinstance(document, dict) and not document.get('error'),
                'completion failed: ' + str((document or {}).get('error')))
        require(not document.get('truncated'), 'completion truncated')
        require(isinstance(document.get('tokens'), list), 'completion returned no token ids')
        require(document.get('tokens_predicted') == len(document['tokens']), 'token count mismatch')
        return {'http_status': status, 'document': document, 'timings': document.get('timings', {})}

    def server_vocab(port):
        try:
            document = get(port, '/v1/models')
            return ((document.get('data') or [{}])[0].get('meta') or {}).get('n_vocab')
        except Exception:
            return None

    def verify_child(pid, port):
        props = get(port, '/props')
        require(props.get('build_info') == EXPECTED_BUILD,
                f"private server build {props.get('build_info')!r} != {EXPECTED_BUILD!r}")
        vocab = server_vocab(port)
        require(vocab is None or vocab == VOCAB, f'n_vocab {vocab} != {VOCAB}')
        if vocab is not None:
            result['server_vocab'] = vocab
        profile = props.get('adaptive_context') or {}
        require(profile.get('profile') == 'xxxlong' and profile.get('state') == 'ready' and
                profile.get('context_size') == 162816, 'private child is not ready in XXXL')
        maps = Path(f'/proc/{pid}/maps').read_text()
        for name in ('libllama.so', 'libllama-server-impl.so', 'libggml-cuda.so'):
            lines = [line for line in maps.splitlines() if name in line]
            require(lines and all(str(binary.parent) + '/' in line for line in lines),
                    f'{name} was not loaded from the release directory')
        return vocab

    def select_xxxl(router_port, corpus):
        prime = {'model': 'qwen-3.8-27b', 'prompt': corpus[:1], 'n_predict': 102401, 'temperature': 0.0,
                 'cache_prompt': False, 'ignore_eos': False, 'logit_bias': [[248046, 1000.0]], 'stream': False}
        status, document, _ = post_raw(router_port, '/completion', prime, 300)
        require(status == 200 and document.get('stop_type') == 'eos' and
                (document.get('timings') or {}).get('predicted_n') == 1,
                'unable to select XXXL with one forced-EOS token')
        pid, port = child(state['process'].pid)
        verify_child(pid, port)
        status, erased, _ = post_raw(port, '/slots/0?action=erase', {}, 60)
        require(status == 200 and (erased or {}).get('n_erased', -1) >= 1, 'tier-prime slot erase failed')
        return port

    def log_count(role, needle):
        path = out / f'router-{role}.log'
        require(path.is_file(), f'missing {role} log')
        return path.read_text(errors='replace').count(needle)

    def run_sequence(role, restore_expected):
        result['sessions'][role] = record = {'role': role}
        port = record['port'] = select_xxxl(router_port, corpus)
        q0 = ask(port, sampling(1, prompt, VOCAB), 2400)
        check(f'{role}_q0_ok', q0.get('http_status') == 200, str(q0.get('error')))
        try:
            probs = distribution_from_top_logprobs(
                (q0['document'].get('completion_probabilities') or [{}])[0].get('top_logprobs'), VOCAB)
        except ValueError as exc:
            raise Unsupported(f'{role} q0 did not return a full-vocabulary distribution: {exc}')
        order = sorted(range(VOCAB), key=probs.__getitem__, reverse=True)
        (out / f'dist-{role}.json').write_text(json.dumps({
            'n_vocab': VOCAB, 'sum': math.fsum(probs), 'argmax': order[0], 'top': order[:64],
            'top_probs': [probs[i] for i in order[:64]],
            'digest': hashlib.sha256(b''.join(struct.pack('<Id', i, p) for i, p in enumerate(probs))).hexdigest()
        }, indent=1) + '\n')
        timings = q0['timings']
        require(timings.get('cache_n') is not None and timings.get('prompt_n') is not None,
                'server does not expose cache_n/prompt_n')
        record['q0'] = {'timings': timings, 'tokens': q0['document']['tokens'],
                        'argmax': order[0], 'top1_prob': probs[order[0]]}
        check(f'{role}_q0_accounting', timings['cache_n'] + timings['prompt_n'] == PROMPT_N,
              f"cache_n={timings['cache_n']} prompt_n={timings['prompt_n']}")
        if restore_expected:
            check(f'{role}_q0_restored', timings['cache_n'] == PREFIX and timings['prompt_n'] == 1 and
                  timings.get('cache_source') == 'disk',
                  f"cache_n={timings['cache_n']} prompt_n={timings['prompt_n']} "
                  f"source={timings.get('cache_source')}")
            record['restore_lines_q0'] = log_count(role, 'auto-restore: native snapshot reused 111999 tokens')
            check(f'{role}_q0_restore_log', record['restore_lines_q0'] >= 1,
                  f"restore log lines={record['restore_lines_q0']}")
        else:
            check(f'{role}_q0_live', timings['cache_n'] == 0 and timings['prompt_n'] == PROMPT_N and
                  timings.get('cache_source') != 'disk',
                  f"cache_n={timings['cache_n']} prompt_n={timings['prompt_n']} "
                  f"source={timings.get('cache_source')}")

        # Rewind 128 + re-append, immediately on the restored owner.
        cached, rows = prompt + q0['document']['tokens'][-1:], PROMPT_N
        q128 = divergent_probe(cached[:rows], rows - REWIND, w)
        plan = rewind_plan(cached, rows, q128)
        check(f'{role}_q128_plan', plan['removed'] == REWIND, str(plan))
        q2 = ask(port, sampling(16, q128), 900)
        require(q2.get('http_status') == 200, f'{role} q128 failed: {q2.get("error")}')
        check(f'{role}_q128_rewind', q2['timings'].get('cache_n') == rows - REWIND and
              q2['timings'].get('prompt_n') == REWIND and len(q2['document']['tokens']) == 16,
              str(q2['timings']))
        record['q128'] = {'timings': q2['timings'], 'tokens': q2['document']['tokens'],
                          'sha256': token_hash(q2['document']['tokens']), 'expected': plan,
                          'probe_sha256': token_hash(q128)}
        if restore_expected:
            record['q128']['restore_lines'] = log_count(
                role, 'auto-restore: native snapshot reused 111999 tokens')
            check(f'{role}_q128_resident_restore', record['q128']['restore_lines'] == record['restore_lines_q0'],
                  f"restore log lines {record['q128']['restore_lines']} != {record['restore_lines_q0']}; "
                  f"the 128-row rewind must run on the restored owner, not a fresh restore")

        cached, rows = q128 + q2['document']['tokens'], PROMPT_N + 15
        plan = rewind_plan(cached, rows, q128)
        check(f'{role}_q128b_plan', plan['removed'] == 16, str(plan))
        q3 = ask(port, sampling(1, q128), 900)
        require(q3.get('http_status') == 200, f'{role} q128b failed: {q3.get("error")}')
        check(f'{role}_q128b_accounting', q3['timings'].get('cache_n') == PREFIX and
              q3['timings'].get('prompt_n') == 1, str(q3['timings']))
        record['q128b'] = {'timings': q3['timings'], 'tokens': q3['document']['tokens'],
                           'sha256': token_hash(q3['document']['tokens']), 'expected': plan}
        record['q128b']['first_token_matches'] = q3['document']['tokens'][:1] == q2['document']['tokens'][:1]
        check(f'{role}_q128b_deterministic', record['q128b']['first_token_matches'],
              f"repeat first token {q3['document']['tokens'][:1]} vs {q2['document']['tokens'][:1]}")

        # Rewind 129 from the same owner after the rewind/re-append round trip.
        cached, rows = q128 + q3['document']['tokens'], PROMPT_N
        require(cached[:rows] == q128, 'repeat request did not restore the q128 prefix')
        q129 = divergent_probe(cached[:rows], rows - REWIND - 1, v)
        plan = rewind_plan(cached, rows, q129)
        check(f'{role}_q129_plan', plan['removed'] == REWIND + 1, str(plan))
        q1 = ask(port, sampling(1, q129), 2400)
        check(f'{role}_q129_response', q1.get('http_status') == 200,
              f"HTTP {q1.get('http_status')} for the 129 probe: {q1.get('error')}; the server must "
              f"fail closed by full reprocess or reset+re-restore, not error out with unknown state")
        t = q1['timings']
        mode = ('full_reprocess' if t.get('cache_n') == 0 else
                'reset_rerestore' if t.get('cache_n') == rows - REWIND - 1 else 'other')
        check(f'{role}_q129_no_partial_trim', mode in ('full_reprocess', 'reset_rerestore') and
              t.get('cache_n', -1) + t.get('prompt_n', -1) == rows, str(t))
        record['q129'] = {'mode': mode, 'timings': t, 'http_status': q1['http_status'],
                          'expected': plan, 'probe_sha256': token_hash(q129)}
        if restore_expected and mode == 'reset_rerestore':
            record['q129']['restore_lines'] = log_count(role, 'auto-restore: native snapshot reused 111999 tokens')
            check(f'{role}_q129_rerestore_log', record['q129']['restore_lines'] >= 2,
                  f"restore log lines={record['q129']['restore_lines']}")
        return record, probs

    try:
        require(binary.is_file() and not binary.is_symlink(), 'candidate binary unavailable')
        result['release'] = {'binary': str(binary), 'sha256': sha(binary, deadline),
                             'libs': {str(p): sha(p, deadline) for p in
                                      sorted({q.resolve() for q in binary.parent.glob('lib*.so*')})}}
        check('release_sha', result['release']['sha256'] == args.expected_sha256, result['release']['sha256'])
        provenance = read_json(binary.parent.parent / 'provenance.json')
        check('release_build', provenance.get('binary_embedded_build_info') == EXPECTED_BUILD,
              str(provenance.get('binary_embedded_build_info')))
        prior, qualified = read_json(PRIOR), read_json(QUAL)
        require(prior.get('status') == 'PASS' and prior.get('production_service_after') == 'active' and
                prior.get('six_snapshots_preserved') is True and len(prior['six_snapshot_inventory_before']) == 6,
                'prior six-snapshot evidence invalid')
        expected_inventory = prior['six_snapshot_inventory_before']
        expected_names = set(expected_inventory)
        copy = prior['native_snapshot_copy']
        require(qualified.get('status') == 'PASS' and qualified.get('prompt_sha256') == prior.get('prompt_sha256'),
                'prior native snapshot qualification invalid')
        expected_snapshot = qualified['snapshot']
        require(expected_snapshot['path'] == str(SOURCE) and copy['source'] == str(SOURCE) and
                copy['sha256'] == expected_snapshot['sha256'], 'qualified snapshot identity differs')
        prior_tokens = prior['auto_restore']['tokens']
        corpus = read_json(CORPUS)
        prompt = build_prompt(corpus, read_json(CODE))
        result['prompt'] = {'tokens': len(prompt), 'sha256': token_hash(prompt),
                            'prefix_sha256': token_hash(prompt[:PREFIX])}
        check('prompt_sha', result['prompt']['sha256'] == qualified['prompt_sha256'], result['prompt']['sha256'])
        check('prompt_prefix_sha', result['prompt']['prefix_sha256'] == qualified['snapshot_prefix_sha256'],
              result['prompt']['prefix_sha256'])
        require(MODEL.is_file() and not MODEL.is_symlink() and sha(MODEL, deadline) == MODEL_SHA,
                'model differs from the previously qualified token stream')
        v = pick_token(corpus, {prompt[CUT]})
        w = pick_token(corpus, {prompt[CUT + 1], v})
        q128 = divergent_probe(prompt, PROMPT_N - REWIND, w)
        q129 = divergent_probe(q128, PROMPT_N - REWIND - 1, v)
        result['probes'] = {'v': v, 'w': w, 'cut': CUT,
                            'q128_sha256': token_hash(q128), 'q129_sha256': token_hash(q129)}
        check('probes', len(q129) == len(q128) == PROMPT_N and
              lcp(prompt, q128) == PROMPT_N - REWIND and lcp(q128, q129) == PROMPT_N - REWIND - 1 and
              q128[PROMPT_N - REWIND] != prompt[PROMPT_N - REWIND] and
              q129[PROMPT_N - REWIND - 1] != q128[PROMPT_N - REWIND - 1],
              'rewind probe construction failed')
        check('probe_plans',
              rewind_plan(prompt + [prompt[0]], PROMPT_N, q128)['removed'] == REWIND and
              rewind_plan(q128 + [q128[0]], PROMPT_N, q129)['removed'] == REWIND + 1,
              'rewind probe removal differs from 128/129')
        require(SOURCE.is_file() and not SOURCE.is_symlink() and
                SOURCE.stat().st_size == expected_snapshot['bytes'] and
                sha(SOURCE, deadline) == expected_snapshot['sha256'], 'canonical snapshot differs')
        for file in PROTECTED:
            require(file.is_file() and not file.is_symlink(), 'protected host file unavailable: ' + str(file))
            state['protected'][str(file)] = {'bytes': file.stat().st_size, 'sha256': sha(file, deadline)}
        result['protected_before'] = state['protected']
        snapshot = store / SNAPSHOT_NAME
        subprocess.run(['cp', '--reflink=auto', '--', str(SOURCE), str(snapshot)],
                       timeout=remain(600), check=True)
        require(not os.path.samefile(SOURCE, snapshot) and
                sha(snapshot, deadline) == expected_snapshot['sha256'], 'private native copy differs')
        result['native_snapshot_copy'] = {'path': str(snapshot), 'source': str(SOURCE),
                                          'sha256': expected_snapshot['sha256'],
                                          'bytes': expected_snapshot['bytes']}
        state['protected'][str(snapshot)] = {'bytes': snapshot.stat().st_size,
                                             'sha256': expected_snapshot['sha256']}
        result['protected_before'] = state['protected']
        check('store_free', shutil.disk_usage(store).free >= 3 * (1 << 30), 'less than 3 GiB free')
        require(subprocess.check_output(['systemctl', 'is-active', SERVICE], text=True,
                                        timeout=remain(30)).strip() == 'active', 'production service is not active')
        main_pid = int(subprocess.check_output(['systemctl', 'show', SERVICE, '-p', 'MainPID', '--value'],
                                               text=True, timeout=remain(30)))
        main_stat = sudo('cat', f'/proc/{main_pid}/stat')
        main_start = int(main_stat[main_stat.rfind(')') + 2:].split()[19])
        argv = [os.fsdecode(x) for x in sudo('cat', f'/proc/{main_pid}/cmdline', binary_out=True).split(b'\0') if x]
        old_binary = Path(sudo('readlink', '-f', f'/proc/{main_pid}/exe')).resolve(strict=True)
        cwd = Path(sudo('readlink', '-f', f'/proc/{main_pid}/cwd')).resolve(strict=True)
        check('production_release', old_binary == binary and sha(old_binary, deadline) == args.expected_sha256,
              str(old_binary))
        require('--models-preset' in argv and Path(argv[argv.index('--models-preset') + 1]).resolve() == PRESET,
                'unexpected production process')
        env = dict(os.fsdecode(x).split('=', 1) for x in
                   sudo('cat', f'/proc/{main_pid}/environ', binary_out=True).split(b'\0') if b'=' in x)
        preset_bytes = PRESET.read_bytes()
        preset_text = preset_bytes.decode('utf-8')
        result['preset_sha256'] = hashlib.sha256(preset_bytes).hexdigest()
        result['preset_overrides'] = {'ctx-checkpoints': '0 (isolate suffix rewind from checkpoint adoption)',
                                      'slot-save-block': '262144 (no background auto-save)',
                                      'slot-save-path': 'private store'}
        state['flags'] = current_flags()
        result['flags_before'] = state['flags']
        result['production_binary_sha256'] = args.expected_sha256
        verify_pfn()
        require(snapshot_names() == expected_names, 'Xbox inventory differs from the qualified six')
        production_port = child(main_pid, privileged=True)[1]
        production_props = get(production_port, '/props')
        check('production_build', production_props.get('build_info') == EXPECTED_BUILD,
              str(production_props.get('build_info')))
        production_vocab = server_vocab(production_port)
        check('production_vocab', production_vocab in (None, VOCAB),
              f'n_vocab {production_vocab} != {VOCAB}')
        if production_vocab is not None:
            result['production_vocab'] = production_vocab
        production_slots = get(production_port, '/slots')
        require(isinstance(production_slots, list) and production_slots, 'production slots unavailable')
        require(all(s.get('is_processing') is False for s in production_slots), 'production slot busy')
        require(all(s.get('is_processing') is False for s in get(production_port, '/slots')),
                'production became busy')
        current = int(subprocess.check_output(['systemctl', 'show', SERVICE, '-p', 'MainPID', '--value'],
                                              text=True, timeout=remain(30)))
        again = sudo('cat', f'/proc/{main_pid}/stat')
        require(current == main_pid and int(again[again.rfind(')') + 2:].split()[19]) == main_start and
                PRESET.read_bytes() == preset_bytes, 'production identity/config changed before stop')
        state['paused'] = True
        print('Pausing production after idle/identity checks.', flush=True)
        sudo('systemctl', 'stop', SERVICE)
        require(subprocess.check_output(['systemctl', 'show', SERVICE, '-p', 'ActiveState', '--value'],
                                        text=True, timeout=remain(30)).strip() == 'inactive', 'production did not stop')
        require(snapshot_names() == expected_names and current_flags() == state['flags'],
                'Xbox changed during isolation')
        require(not gpu_busy(), 'GPU already has a compute process; refusing overlap')
        peers = [line for line in subprocess.check_output(['ss', '-H', '-tn', 'state', 'established'],
                                                          text=True, timeout=remain(30)).splitlines()
                 if line.split() and line.split()[-1] == XBOX_HOST + ':50054']
        require(not peers, 'another local process has an established Xbox RKVA connection')
        before = snapshot_inventory(expected_names)
        check('xbox_inventory_before', before == expected_inventory, 'one of the six Xbox objects changed')
        result['six_snapshot_inventory_before'] = before
        state['before'] = before
        monitor_output = (out / 'vram.csv').open('w', buffering=1)
        state['monitor_output'] = monitor_output
        state['monitor'] = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.free,memory.used',
                                             '--format=csv,noheader,nounits', '-l', '1'],
                                            stdout=monitor_output, stderr=subprocess.DEVNULL)
        base = [str(binary), *argv[1:]]
        distributions = {}
        for role, auto_restore in (('live', False), ('restored', True)):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                router_port = sock.getsockname()[1]
            text, one = re.subn(r'(?m)^slot-save-path\s*=.*$',
                                'slot-save-path = ' + str(store if auto_restore else store_live) + '/',
                                preset_text)
            text, two = re.subn(r'(?m)^slot-save-block\s*=.*$', 'slot-save-block = 262144', text)
            text, three = re.subn(r'(?m)^ctx-checkpoints\s*=.*$', 'ctx-checkpoints = 0', text)
            require(one == two == three == 1, 'cannot apply private preset overrides')
            ini = out / f'candidate-{role}.ini'
            ini.write_text(text)
            current_env = dict(env)
            current_env.update({
                'LLAMA_SERVER_ROUTER_STATE_DIR': str(out / f'router-state-{role}'),
                'LLAMA_NATIVE_SNAPSHOT_AUTO_RESTORE': '1' if auto_restore else '0',
                'LLAMA_ADAPTIVE_DIRECT_PURE_DISK_RESTORE': '1',
                'GGML_RKVA_DISK_SNAPSHOTS': '1',
                'GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB': '18',
                'LD_LIBRARY_PATH': str(binary.parent) + ':/opt/cuda/lib64:/usr/lib'})
            command = list(base)
            for option, value in (('--models-preset', str(ini)), ('--host', '127.0.0.1'),
                                  ('--port', str(router_port))):
                require(option in command, 'production argument missing: ' + option)
                command[command.index(option) + 1] = value
            require(not auto_restore or snapshot.is_file(), 'snapshot copy missing')
            result.setdefault('private_commands', {})[role] = command
            print(f'Private session {role}: '
                  f'{"auto-restore" if auto_restore else "cold live prefill"}.', flush=True)
            private_start(role, command, current_env, cwd, router_port)
            record, probs = run_sequence(role, auto_restore)
            distributions[role] = probs
            result['sessions'][role] = record
            write_result()
            result[f'quiescence_{role}'] = stop_private()
            require(not gpu_busy(), 'GPU still busy after private session')
        metrics = distribution_metrics(distributions['live'], distributions['restored'])
        prior_rank = sum(1 for p in distributions['live'] if p > distributions['live'][prior_tokens[0]])
        if prior_rank:
            warn('prior_first_token_rank',
                 f'prior sampled token {prior_tokens[0]} is rank {prior_rank} in the raw live distribution')
        result['logits'] = {'metrics': {k: (v if isinstance(v, (int, str)) or (isinstance(v, float) and
                                                                            math.isfinite(v)) else None)
                                        for k, v in metrics.items()},
                            'tolerances': TOL,
                            'prior_first_token': prior_tokens[0],
                            'prior_first_token_rank': prior_rank}
        for name, ok, detail in gate_distributions(metrics):
            grade('logits_' + name, ok, detail)
        live, restored = result['sessions']['live'], result['sessions']['restored']
        grade('rewind_tokens_equal', live['q128']['tokens'] == restored['q128']['tokens'],
              f"{live['q128']['sha256']} vs {restored['q128']['sha256']}")
        grade('rewind_probe_equal', live['q128']['probe_sha256'] == restored['q128']['probe_sha256'],
              f"{live['q128']['probe_sha256']} vs {restored['q128']['probe_sha256']}")
        grade('rewind_accounting_equal',
              live['q128']['timings'].get('cache_n') == restored['q128']['timings'].get('cache_n') == CUT + 1 and
              live['q128']['timings'].get('prompt_n') == restored['q128']['timings'].get('prompt_n') == REWIND,
              f"{live['q128']['timings']} vs {restored['q128']['timings']}")
        result['rewind'] = {
            'q128': {'planned_removed': REWIND, 'live_timings': live['q128']['timings'],
                     'restored_timings': restored['q128']['timings'],
                     'tokens_sha256': live['q128']['sha256']},
            'q128b': {'live_timings': live['q128b']['timings'],
                      'restored_timings': restored['q128b']['timings'],
                      'first_token_matches': [live['q128b']['first_token_matches'],
                                              restored['q128b']['first_token_matches']]},
            'q129': {'planned_removed': REWIND + 1, 'live_mode': live['q129']['mode'],
                     'restored_mode': restored['q129']['mode'],
                     'live_timings': live['q129']['timings'],
                     'restored_timings': restored['q129']['timings']},
            'live_equals_restored': live['q128']['tokens'] == restored['q128']['tokens']}
        failed = sorted(name for name, item in result['checks'].items() if not item['ok'])
        result['status'] = 'PASS' if not failed else 'FAIL'
        if failed:
            result['error'] = 'checks failed: ' + ', '.join(failed)
    except Unsupported as exc:
        result['status'] = 'UNSUPPORTED'
        result['error'] = str(exc)
    except BaseException as exc:
        result['status'] = 'FAIL'
        result['error'] = type(exc).__name__ + ': ' + str(exc).replace(password, '[redacted]')
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        deadline = time.monotonic() + args.cleanup_timeout
        if state['monitor'] is not None:
            try:
                state['monitor'].terminate()
                state['monitor'].wait(timeout=10)
            except subprocess.TimeoutExpired:
                state['monitor'].kill()
                state['monitor'].wait(timeout=10)
            except Exception as exc:
                result['cleanup_errors'].append('VRAM monitor: ' + type(exc).__name__)
        if state['monitor_output'] is not None:
            state['monitor_output'].close()
        private_quiescent = False
        try:
            result['private_quiescence'] = stop_private()
            private_quiescent = True
        except Exception as exc:
            result['cleanup_errors'].append('private process: ' + type(exc).__name__ + ': ' + str(exc))
        if state['monitor_output'] is not None:
            try:
                minimum, count = sampled_vram_min((out / 'vram.csv').read_text().splitlines())
                result['vram_sampled_min_free_mib'] = minimum
                result['vram_sample_count'] = count
                if minimum < 18:
                    result['cleanup_errors'].append(f'sampled free VRAM {minimum} MiB below 18 MiB')
            except Exception as exc:
                result['cleanup_errors'].append('VRAM samples: ' + type(exc).__name__ + ': ' + str(exc))
        flags_ok = state['flags'] is None
        try:
            if state['flags'] is not None:
                require(current_flags() == state['flags'], 'Xbox flags changed')
                result['flags_after'] = state['flags']
                flags_ok = True
        except Exception as exc:
            result['cleanup_errors'].append('Xbox flags: ' + type(exc).__name__ + ': ' + str(exc))
        if state['paused']:
            try:
                require(private_quiescent, 'refusing production start: private process not quiescent')
                require(flags_ok, 'refusing production start: Xbox flags not verified')
                verify_pfn()
                print('Restoring production after private process quiescence.', flush=True)
                sudo('systemctl', 'start', SERVICE)
                active = subprocess.check_output(['systemctl', 'is-active', SERVICE], text=True,
                                                 timeout=remain(30)).strip()
                result['production_service_after'] = active
                require(active == 'active', 'production did not restart')
            except Exception as exc:
                result['cleanup_errors'].append('production service: ' + type(exc).__name__ + ': ' + str(exc))
        if state['before'] is not None and private_quiescent:
            try:
                verify_pfn()
                result['six_snapshots_preserved'] = snapshot_inventory(set(state['before'])) == state['before']
                result['snapshot_names_after'] = sorted(snapshot_names())
                require(result['six_snapshots_preserved'], 'Xbox snapshot bytes changed')
                require(set(state['before']) == set(result['snapshot_names_after']),
                        'Xbox snapshot names changed')
            except Exception as exc:
                result['cleanup_errors'].append('snapshot verification: ' + type(exc).__name__ + ': ' + str(exc))
        try:
            if 'native_snapshot_copy' in result:
                allowed = re.compile(r'^\.llama-kv-snapshot-(?:ref-\d+\.ref|store\.lock)$')
                names = sorted(p.name for p in store.iterdir())
                live_names = sorted(p.name for p in store_live.iterdir())
                require(all(allowed.fullmatch(name) or name == SNAPSHOT_NAME for name in names) and
                        all(allowed.fullmatch(name) for name in live_names),
                        f'private slot store gained files: {names} / {live_names}')
                require(SNAPSHOT_NAME in names, 'private snapshot copy disappeared')
                result['private_store_entries'] = {'restored': names, 'live': live_names}
        except Exception as exc:
            result['cleanup_errors'].append('private store: ' + type(exc).__name__ + ': ' + str(exc))
        try:
            for path, expected_file in state['protected'].items():
                file = Path(path)
                require(file.is_file() and not file.is_symlink() and
                        file.stat().st_size == expected_file['bytes'] and
                        sha(file, deadline) == expected_file['sha256'], 'protected host file changed: ' + path)
            if 'release' in result:
                require(sha(binary, deadline) == result['release']['sha256'], 'release binary changed during run')
                require(all(sha(Path(path), deadline) == digest
                            for path, digest in result['release']['libs'].items()),
                        'release library changed during run')
            result['protected_host_files_preserved'] = bool(state['protected'])
        except Exception as exc:
            result['cleanup_errors'].append('host verification: ' + type(exc).__name__ + ': ' + str(exc))
        if result['cleanup_errors'] and result['status'] in ('PASS', 'RUNNING'):
            result['status'] = 'FAIL'
            result['failure_phase'] = 'cleanup'
        write_result()
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    print(json.dumps({'status': result['status'], 'result': str(out / 'result.json'),
                      'logits': result.get('logits', {}).get('metrics'),
                      'rewind': result.get('rewind'), 'cleanup_errors': result['cleanup_errors']}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
