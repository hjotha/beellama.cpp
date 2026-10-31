#!/usr/bin/env python3
"""Qualify native auto-restore (PFN 468): full-vocabulary logits and rewind safety.

Inert without --execute (prints the plan, touches nothing). With --execute it
reuses the existing 111999-token host snapshot and the six Xbox objects, pauses
production only after idle/identity rechecks, restarts it in finally even if a
private survivor cannot be reaped, and never writes to the Xbox store.

Flow:
  1. live private process: cold 112000-token prefill, Q0 with n_probs=n_vocab.
  2. fresh private process: native auto-restore, Q0 with n_probs=n_vocab.
  3. full-vocabulary comparison is written to item-4-logits.json immediately,
     before any rewind probe.
  4. rewind probes run on the RESTORED session only:
       Q128  keep 111872 rows (128 removed) + re-append
       Q128b identical repeat (16 removed)
       Q129  keep rows-129 (129 removed)
     A hybrid/recurrent model cannot partially rewind: the safe and accepted
     outcome is a full reprocess (cache_n=0, cache_reprocessed_n=request length,
     prompt_n=request length, cache_lcp_n>=keep). A real row reuse
     (cache_n=keep, prompt_n=request-keep) is also accepted for Q128/Q128b.
     Q129 accepts only full reprocess (or an explicit HTTP refusal), never a
     partial trim. Q128b must reproduce Q128's continuation token for token.

HTTP has no rewind action: a request whose prompt shares only K resident tokens
drives server-context.cpp keep_first(K) + seq_rm_suffix, which reaches
rkva_native_rewind_locked and the revision-3 range-rewind RPC (<= 128 rows).

Mode --fresh-snapshot (same-build provenance, logits only):
  live cold Q0 at 112000 tokens with n_probs -> explicit native_stream slot
  save (host file + one new Xbox object) -> live reference Q0 at the same
  112000 tokens + the next token (112001) -> stop live -> fresh restored Q0 at
  112001 (cache_n=112000, prompt_n=1) -> item-4 comparison. No rewind probes.
  The six original Xbox keys are verified byte-unchanged; the seventh key is
  recorded, never deleted.
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
PFN = 'GianlucaMazza.xllama_1.6.0.468_x64__pj67f1fcj4n14'
EXPECTED_BUILD = 'b13306-69e69dae5'
EXPECTED_SHA = 'f701499cbd510731f4c931dc0e78c238eeedce1945fc0f8738e089ffa5a71c77'
REMOTE_DIR = 'rkva-snapshots-v2'
SNAPSHOT_NAME = 'native-code-prefix-111999.bin'
ITEM4 = 'item-4-logits.json'
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
    # The two production-cache native snapshots were already absent before this run
    # (evicted by production retention); the benchmark copies and Xbox objects are intact.
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


def finite(value):
    return value if isinstance(value, (int, str)) or (isinstance(value, float) and math.isfinite(value)) else None


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
    suffix removal is rows - n_past."""
    common = lcp(cached_tokens, request)
    n_past = common - 1 if common == len(request) and common > 0 else common
    require(cached_rows >= n_past, 'request keeps more rows than the cached state', ValueError)
    return {'common_prefix': common, 'n_past': n_past, 'removed': cached_rows - n_past}


def divergent_probe(evaluated, keep, token):
    """Request over the evaluated prefix that keeps `keep` rows and replaces the
    next token. Length equals the prefix, so the removal is len(prefix) - keep."""
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


def classify_rewind(timings, keep_rows, request_len):
    """Classify a rewind probe response without assuming the model can rewind.

    reuse         : the resident rows were kept (cache_n == keep_rows) and only
                    the suffix was re-evaluated (prompt_n == request-keep).
    full_reprocess: the server refused the partial removal and safely re-ran the
                    whole request (hybrid/recurrent contract).
    partial       : some other positive cache_n, i.e. an unexpected partial state.
    missing/unknown: no usable accounting (treated as a failure by the caller).
    """
    require(isinstance(timings, dict), 'timings unavailable', ValueError)
    cache_n, prompt_n = timings.get('cache_n'), timings.get('prompt_n')
    reprocessed, lcp_n = timings.get('cache_reprocessed_n'), timings.get('cache_lcp_n')
    base = {'cache_n': cache_n, 'prompt_n': prompt_n,
            'cache_reprocessed_n': reprocessed, 'cache_lcp_n': lcp_n,
            'cache_planned_n': timings.get('cache_planned_n'),
            'cache_reason': timings.get('cache_reason'),
            'cache_source': timings.get('cache_source')}
    if type(cache_n) is not int or type(prompt_n) is not int:
        return dict(base, mode='missing', rewind_reuse=None)
    if cache_n == keep_rows and prompt_n == request_len - keep_rows:
        return dict(base, mode='reuse', rewind_reuse=True)
    if (cache_n == 0 and prompt_n == request_len and reprocessed == request_len and
            type(lcp_n) is int and lcp_n >= keep_rows):
        return dict(base, mode='full_reprocess', rewind_reuse=False)
    return dict(base, mode='partial' if cache_n > 0 else 'unknown', rewind_reuse=False)


def split_inventory(names, originals):
    """Split a live Xbox key set into missing originals, present originals and extras."""
    names, originals = set(names), set(originals)
    return {'missing': sorted(originals - names), 'present': sorted(originals & names),
            'extras': sorted(names - originals)}


def new_object_keys(before, after):
    return sorted(set(after) - set(before))


def fresh_reference_prompt(prompt, next_token):
    """The fresh-snapshot reference/restore request: the saved 112000-token
    prefix plus the next token, so both sides score position 112000."""
    require(isinstance(prompt, list) and prompt, 'empty prompt', ValueError)
    require(type(next_token) is int and next_token >= 0, 'invalid next token', ValueError)
    return list(prompt) + [next_token]


def native_trailer(path):
    """Locate and validate the single RKVK trailer in a host snapshot.

    The Xbox object key is the SHA-256 of the host file through the end of this
    frame (header + payload + SHA-256 footer). Verified offline against the
    qualified 111999 pair (key 3c7218bd...).
    """
    path = Path(path)
    size = path.stat().st_size
    candidates = {}
    overlap, position, previous = 80, 0, b''
    with path.open('rb') as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            data = previous + block
            base = position - len(previous)
            start = 0
            while True:
                index = data.find(b'RKVK', start)
                if index < 0:
                    break
                start = index + 1
                if index + 80 > len(data):
                    continue
                _, version, header_bytes = struct.unpack_from('<IHH', data, index)
                if version != 1 or header_bytes != 80:
                    continue
                layers, _ = struct.unpack_from('<II', data, index + 8)
                origin, capacity = struct.unpack_from('<II', data, index + 16)
                heads, iterations = struct.unpack_from('<II', data, index + 24)
                tail_tokens, rows = struct.unpack_from('<II', data, index + 32)
                generation, payload = struct.unpack_from('<QQ', data, index + 48)
                offset = base + index
                frame_end = offset + 80 + payload + 32
                if layers and origin and capacity and payload and frame_end <= size:
                    candidates[offset] = {'offset': offset, 'frame_end': frame_end,
                                          'payload_bytes': payload, 'layers': layers,
                                          'origin': origin, 'capacity': capacity, 'heads': heads,
                                          'iterations': iterations, 'tail_tokens': tail_tokens,
                                          'rows': rows, 'generation': generation}
            previous = data[-overlap:]
            position += len(block)
    valid = []
    for candidate in candidates.values():
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            handle.seek(candidate['offset'])
            left = 80 + candidate['payload_bytes']
            while left:
                chunk = handle.read(min(8 << 20, left))
                if not chunk:
                    break
                digest.update(chunk)
                left -= len(chunk)
            footer = handle.read(32)
        if digest.digest() == footer:
            valid.append(candidate)
    require(len(valid) == 1, f'expected one valid RKVK trailer, found {len(valid)}', ValueError)
    return valid[0]


def file_hashes(path, prefix_bytes, deadline=None):
    """Full-file SHA-256 and the SHA-256 of the first `prefix_bytes` (Xbox key)."""
    full, prefix, seen = hashlib.sha256(), hashlib.sha256(), 0
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            if deadline is not None:
                require(time.monotonic() < deadline, 'time budget expired during hashing')
            full.update(block)
            if seen < prefix_bytes:
                prefix.update(block[:min(len(block), prefix_bytes - seen)])
            seen += len(block)
    require(seen == Path(path).stat().st_size, 'file size changed during hashing', ValueError)
    return full.hexdigest(), prefix.hexdigest()


def inspect_host_footer(path):
    with Path(path).open('rb') as handle:
        header = handle.read(12)
        require(len(header) == 12, 'host snapshot header truncated', ValueError)
        magic, version, count = struct.unpack('<III', header)
        handle.seek(-24, 2)
        length, checksum, footer = struct.unpack('<QQ8s', handle.read(24))
        require(footer == b'LLROUTE1' and 0 < length <= 65536 and
                length <= Path(path).stat().st_size - 24, 'host snapshot footer invalid', ValueError)
        handle.seek(-24 - length, 2)
        metadata = json.loads(handle.read(length))
    require(magic == 0x67677371 and version == 4, 'unexpected host snapshot envelope', ValueError)
    return {'n_tokens': count, 'metadata': metadata, 'manifest_checksum_recorded': checksum}


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
    parser.add_argument('--cleanup-timeout', type=int, default=1200)
    parser.add_argument('--fresh-snapshot', action='store_true',
                        help='save a new 112000-token snapshot with the current build/app and '
                             'compare live vs restored at the 112001st token (logits only)')
    args = parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.label), 'invalid label')
    require(re.fullmatch(r'[0-9a-f]{64}', args.expected_sha256), 'invalid expected SHA')
    require(1200 <= args.timeout <= 7200 and 120 <= args.cleanup_timeout <= 1800, 'invalid time budget')
    binary = Path(args.binary).resolve(strict=True)
    out = ROOT / args.label
    fresh = args.fresh_snapshot
    fresh_filename = f'fresh-112000-{args.label}.bin'
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.bin', fresh_filename), 'invalid fresh filename')
    if not args.execute:
        flow = (['live cold Q0 at 112000 (n_probs)',
                 'explicit native_stream slot save (112000 evaluated tokens)',
                 'live reference Q0 at 112001 (same 112000 + next token)',
                 'fresh restored Q0 at 112001 (cache_n=112000, prompt_n=1)',
                 'item-4 full-vocabulary comparison (no rewind probes)']
                if fresh else
                ['live cold Q0 (n_probs)',
                 'restored fresh-process Q0 (auto-restore 111999, n_probs)',
                 'item-4 full-vocabulary comparison',
                 'restored-only rewind probes: Q128 reuse|full, Q128b repeat, Q129 full|refusal'])
        print(json.dumps({
            'execute': False, 'mode': 'fresh_snapshot' if fresh else 'existing_snapshot',
            'pfn': PFN, 'out': str(out), 'binary': str(binary),
            'expected_sha256': args.expected_sha256, 'prompt_tokens': PROMPT_N,
            'snapshot_prefix': PREFIX, 'n_probs': VOCAB,
            'fresh_filename': fresh_filename if fresh else None,
            'flow': flow,
            'rewind_acceptance': {
                'reuse': 'cache_n == keep and prompt_n == request-keep',
                'full_reprocess': 'cache_n == 0 and prompt_n == cache_reprocessed_n == request and cache_lcp_n >= keep',
                'reject': '5xx, partial cache_n, or any Q129 partial trim'},
            'tolerances': TOL, 'status': 'PREVIEW'}, indent=2))
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
    result = {'status': 'RUNNING',
              'scope': ('PFN 468 fresh-snapshot logits qualification' if fresh
                        else 'PFN 468 logits + safe-rewind qualification'),
              'expected_pfn': PFN, 'expected_build': EXPECTED_BUILD, 'cleanup_errors': [],
              'warnings': [], 'checks': {}, 'timings_comparable': False, 'sessions': {}}
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

    def server_vocab(port):
        try:
            document = get(port, '/v1/models')
            return ((document.get('data') or [{}])[0].get('meta') or {}).get('n_vocab')
        except Exception:
            return None

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

    def proc_state(pid):
        raw = Path(f'/proc/{pid}/stat').read_text()
        tail = raw[raw.rfind(')') + 2:].split()
        return tail[0], int(tail[2]), int(tail[19])

    def private_pids():
        marker = os.fsencode(str(out))
        found = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == os.getpid():
                continue
            try:
                status, _, _ = proc_state(pid)
                if status in ('Z', 'X', 'x'):
                    continue
            except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
                continue
            matched = False
            try:
                argv = (entry / 'cmdline').read_bytes().split(b'\0')
                matched = b'--models-preset' in argv and any(marker in arg for arg in argv)
            except (FileNotFoundError, PermissionError):
                pass
            if not matched:
                try:
                    matched = marker in (entry / 'environ').read_bytes()
                except (FileNotFoundError, PermissionError):
                    matched = False
            if matched:
                found.append(pid)
        return sorted(found)

    def leader_alive(record):
        try:
            status, _, start = proc_state(record['pid'])
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
            return False
        return status not in ('Z', 'X', 'x') and start == record['starttime']

    def stop_private():
        process, record = state['process'], state['pgid']
        if process is None and record is None and not private_pids():
            return {'state': 'not_started', 'survivors': [], 'signaled': []}
        signaled = []
        for sig, seconds in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
            targets = set(private_pids())
            if record is not None and leader_alive(record):
                targets.add(record['pid'])
            for pid in sorted(targets):
                try:
                    os.kill(pid, sig)
                    signaled.append([pid, int(sig)])
                except ProcessLookupError:
                    pass
            if record is not None and leader_alive(record):
                try:
                    os.killpg(record['pgid'], sig)
                except ProcessLookupError:
                    pass
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                if process is not None:
                    process.poll()  # reap our direct child; a zombie keeps the group alive
                if not private_pids():
                    break
                time.sleep(0.2)
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        survivors = private_pids()
        state['process'], state['pgid'] = None, None
        return {'state': 'quiescent' if not survivors else 'survivors',
                'survivors': survivors, 'signaled': signaled}

    def wait_production(cap=240):
        status = {'active': None, 'props_ok': False}
        end = time.monotonic() + cap
        while time.monotonic() < end:
            try:
                active = subprocess.check_output(['systemctl', 'is-active', SERVICE], text=True,
                                                 timeout=remain(30)).strip()
            except Exception as exc:
                active = 'error:' + type(exc).__name__
            status['active'] = active
            if active == 'active':
                try:
                    pid = int(subprocess.check_output(['systemctl', 'show', SERVICE, '-p', 'MainPID', '--value'],
                                                      text=True, timeout=remain(30)))
                    _, port = child(pid, privileged=True)
                    props = get(port, '/props', 15)
                    status.update({'main_pid': pid, 'port': port, 'props_ok': True,
                                   'build_info': props.get('build_info')})
                    return status
                except Exception as exc:
                    status['last_error'] = type(exc).__name__ + ': ' + str(exc)[:200]
            time.sleep(1)
        return status

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

    def compare_logits(mode, reference, reference_probs, candidate, candidate_probs, snapshot):
        metrics = distribution_metrics(reference_probs, candidate_probs)
        gates = gate_distributions(metrics)
        payload = {
            'mode': mode,
            'snapshot': snapshot,
            'reference': {'timings': reference['timings'], 'argmax': reference['argmax'],
                          'top1_prob': reference['top1_prob'], 'tokens': reference['tokens'],
                          'prompt_len': reference['prompt_len']},
            'candidate': {'timings': candidate['timings'], 'argmax': candidate['argmax'],
                          'top1_prob': candidate['top1_prob'], 'tokens': candidate['tokens'],
                          'prompt_len': candidate['prompt_len'],
                          'cache_n': candidate['timings'].get('cache_n'),
                          'prompt_n': candidate['timings'].get('prompt_n'),
                          'cache_source': candidate['timings'].get('cache_source')},
            'n_vocab': VOCAB, 'prompt_sha256': result['prompt']['sha256'],
            'metrics': {k: finite(v) for k, v in metrics.items()},
            'tolerances': TOL,
            'gates': [{'name': name, 'ok': bool(ok), 'detail': detail} for name, ok, detail in gates],
            'verdict': 'PASS' if all(ok for _, ok, _ in gates) else 'FAIL',
            'top_left': [[i, reference_probs[i]] for i in
                         sorted(range(VOCAB), key=reference_probs.__getitem__, reverse=True)[:20]],
            'top_right': [[i, candidate_probs[i]] for i in
                          sorted(range(VOCAB), key=candidate_probs.__getitem__, reverse=True)[:20]],
            'tolerance_rationale': ('same binary/model/prompt; restored rows are bit-exact, so only float32 '
                                    'reduction-order noise is expected (prior teacher-forced evidence was '
                                    'bit-identical). KL 1e-4 nats is ~100x above noise and ~1000x below a '
                                    'real row mismatch.')}
        (out / ITEM4).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        result['logits'] = payload
        for name, ok, detail in gates:
            grade('logits_' + name, ok, detail)
        write_result()

    def capture_q0(session, label, port, tokens, expect, artifact):
        q0 = ask(port, sampling(1, tokens, VOCAB), 2400)
        check(f'{label}_http', q0.get('http_status') == 200, str(q0.get('error')))
        try:
            probs = distribution_from_top_logprobs(
                (q0['document'].get('completion_probabilities') or [{}])[0].get('top_logprobs'), VOCAB)
        except ValueError as exc:
            raise Unsupported(f'{label} did not return a full-vocabulary distribution: {exc}')
        order = sorted(range(VOCAB), key=probs.__getitem__, reverse=True)
        (out / artifact).write_text(json.dumps({
            'n_vocab': VOCAB, 'sum': math.fsum(probs), 'argmax': order[0], 'top': order[:64],
            'top_probs': [probs[i] for i in order[:64]],
            'digest': hashlib.sha256(b''.join(struct.pack('<Id', i, p) for i, p in enumerate(probs))).hexdigest()
        }, indent=1) + '\n')
        timings = q0['timings']
        require(timings.get('cache_n') is not None and timings.get('prompt_n') is not None,
                'server does not expose cache_n/prompt_n')
        restore_lines = None
        check(f'{label}_accounting', timings['cache_n'] + timings['prompt_n'] == len(tokens),
              f"cache_n={timings['cache_n']} prompt_n={timings['prompt_n']} tokens={len(tokens)}")
        if expect.get('disk'):
            check(f'{label}_restored', timings.get('cache_source') == 'disk' and
                  timings['cache_n'] == expect['cache_n'] and timings['prompt_n'] == expect['prompt_n'],
                  f"cache_n={timings['cache_n']} prompt_n={timings['prompt_n']} "
                  f"source={timings.get('cache_source')}")
            restore_lines = log_count(session, f'auto-restore: native snapshot reused {expect["cache_n"]} tokens')
            check(f'{label}_restore_log', restore_lines >= 1, f'restore log lines={restore_lines}')
        else:
            check(f'{label}_live', timings['cache_n'] == expect['cache_n'] and
                  timings['prompt_n'] == expect['prompt_n'] and timings.get('cache_source') != 'disk',
                  f"cache_n={timings['cache_n']} prompt_n={timings['prompt_n']} "
                  f"source={timings.get('cache_source')}")
        record = {'timings': timings, 'tokens': q0['document']['tokens'], 'argmax': order[0],
                  'top1_prob': probs[order[0]], 'top': order[:20], 'prompt_len': len(tokens)}
        if restore_lines is not None:
            record['restore_lines'] = restore_lines
        return record, probs

    def save_native(port, filename):
        status, document, text = post_raw(port, '/slots/0?action=save',
                                          {'filename': filename, 'native_stream': True}, 900)
        check('fresh_save_http', status == 200,
              f"HTTP {status}: {(document or {}).get('error', text[-500:])}")
        check('fresh_save_tokens', document.get('n_saved') == PROMPT_N, str(document.get('n_saved')))
        check('fresh_save_bytes', type(document.get('n_written')) is int and document['n_written'] > 0,
              str(document.get('n_written')))
        check('fresh_save_filename', document.get('filename') == filename, str(document.get('filename')))
        return document

    def start_session(role, auto_restore, slot_store):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            router_port = sock.getsockname()[1]
        text, one = re.subn(r'(?m)^slot-save-path\s*=.*$',
                            'slot-save-path = ' + str(slot_store) + '/', preset_text)
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
        result.setdefault('private_commands', {})[role] = command
        print(f'Private session {role}: '
              f'{"auto-restore" if auto_restore else "cold live"} on {slot_store}.', flush=True)
        private_start(role, command, current_env, cwd, router_port)
        return select_xxxl(router_port, corpus)

    def run_sequence(role, port, auto_restore, reference=None, probes=False):
        record = result['sessions'][role] = {'role': role, 'port': port}
        expect = ({'cache_n': PREFIX, 'prompt_n': 1, 'disk': True} if auto_restore else
                  {'cache_n': 0, 'prompt_n': PROMPT_N, 'disk': False})
        q0, probs = capture_q0(role, f'{role}_q0', port, prompt, expect, f'dist-{role}.json')
        record['q0'] = q0
        if reference is not None:
            compare_logits('existing_snapshot', reference[0], reference[1], q0, probs, result['snapshot'])
        if probes:
            record['rewinds'] = {}
            run_rewinds(role, port, q0, record['rewinds'])
        return record, probs

    def run_rewinds(role, port, q0, rewinds):
        cached, rows = prompt + q0['tokens'][-1:], PROMPT_N

        # 128-row rewind + re-append (accepted as reuse or safe full reprocess).
        q128 = divergent_probe(cached[:rows], rows - REWIND, w)
        plan = rewind_plan(cached, rows, q128)
        check(f'{role}_q128_plan', plan['removed'] == REWIND, str(plan))
        q2 = ask(port, sampling(16, q128), 900)
        check(f'{role}_q128_response', q2.get('http_status') == 200,
              f"HTTP {q2.get('http_status')}: {q2.get('error')}")
        signature = classify_rewind(q2['timings'], plan['n_past'], len(q128))
        check(f'{role}_q128_accounting', signature['mode'] in ('reuse', 'full_reprocess'),
              f"{signature} timings={q2['timings']}")
        rewinds['q128'] = {'mode': signature['mode'], 'rewind_reuse': signature['rewind_reuse'],
                           'planned_removed': plan['removed'], 'keep_rows': plan['n_past'],
                           'signature': signature, 'timings': q2['timings'],
                           'tokens': q2['document']['tokens'],
                           'sha256': token_hash(q2['document']['tokens']),
                           'probe_sha256': token_hash(q128)}

        cached, rows = q128 + q2['document']['tokens'], PROMPT_N + 15
        plan = rewind_plan(cached, rows, q128)
        check(f'{role}_q128b_plan', plan['removed'] == 16, str(plan))
        q3 = ask(port, sampling(16, q128), 900)
        check(f'{role}_q128b_response', q3.get('http_status') == 200,
              f"HTTP {q3.get('http_status')}: {q3.get('error')}")
        signature = classify_rewind(q3['timings'], plan['n_past'], len(q128))
        check(f'{role}_q128b_accounting', signature['mode'] in ('reuse', 'full_reprocess'),
              f"{signature} timings={q3['timings']}")
        deterministic = q3['document']['tokens'] == q2['document']['tokens']
        check(f'{role}_q128b_deterministic', deterministic,
              f"q128={q2['document']['tokens']} q128b={q3['document']['tokens']}")
        rewinds['q128b'] = {'mode': signature['mode'], 'rewind_reuse': signature['rewind_reuse'],
                            'planned_removed': plan['removed'], 'keep_rows': plan['n_past'],
                            'signature': signature, 'timings': q3['timings'],
                            'tokens': q3['document']['tokens'],
                            'sha256': token_hash(q3['document']['tokens']),
                            'deterministic': deterministic}

        cached, rows = q128 + q3['document']['tokens'], PROMPT_N + 15
        keep = rows - REWIND - 1
        q129 = divergent_probe(cached[:rows], keep, pick_token(corpus, {cached[keep]}))
        plan = rewind_plan(cached, rows, q129)
        check(f'{role}_q129_plan', plan['removed'] == REWIND + 1, str(plan))
        q1 = ask(port, sampling(1, q129), 2400)
        status = q1.get('http_status')
        if status == 200:
            signature = classify_rewind(q1['timings'], keep, len(q129))
            check(f'{role}_q129_no_partial_trim', signature['mode'] == 'full_reprocess',
                  f"{signature} timings={q1['timings']}; a 129-row partial trim must never be applied")
            mode, timings = 'full_reprocess', q1['timings']
        else:
            check(f'{role}_q129_refusal', 400 <= status < 600,
                  f"HTTP {status}: {q1.get('error')}")
            signature = {'mode': 'http_refusal', 'rewind_reuse': False, 'http_status': status}
            mode, timings = 'http_refusal', None
        rewinds['q129'] = {'mode': mode, 'rewind_reuse': False,
                           'planned_removed': plan['removed'], 'keep_rows': keep,
                           'signature': signature, 'http_status': status, 'timings': timings,
                           'probe_sha256': token_hash(q129)}
        return rewinds

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
        require(qualified.get('status') == 'PASS' and qualified.get('prompt_sha256') == prior.get('prompt_sha256'),
                'prior native snapshot qualification invalid')
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
        if not fresh:
            copy = prior['native_snapshot_copy']
            expected_snapshot = qualified['snapshot']
            require(expected_snapshot['path'] == str(SOURCE) and copy['source'] == str(SOURCE) and
                    copy['sha256'] == expected_snapshot['sha256'], 'qualified snapshot identity differs')
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
                  f'q128_lcp={lcp(prompt, q128)} q129_lcp={lcp(q128, q129)}')
            check('probe_plans',
                  rewind_plan(prompt + [prompt[0]], PROMPT_N, q128)['removed'] == REWIND and
                  rewind_plan(q128 + [q128[0]], PROMPT_N, q129)['removed'] == REWIND + 1,
                  'removals 128/129 confirmed')
            require(SOURCE.is_file() and not SOURCE.is_symlink() and
                    SOURCE.stat().st_size == expected_snapshot['bytes'] and
                    sha(SOURCE, deadline) == expected_snapshot['sha256'], 'canonical snapshot differs')
            result['snapshot'] = {'mode': 'existing_snapshot', 'n_tokens': PREFIX,
                                  'host_file': str(SOURCE), 'host_sha256': expected_snapshot['sha256'],
                                  'xbox_key': qualified.get('new_xbox_snapshot'),
                                  'prefix_sha256': result['prompt']['prefix_sha256']}
        for file in PROTECTED:
            require(file.is_file() and not file.is_symlink(), 'protected host file unavailable: ' + str(file))
            state['protected'][str(file)] = {'bytes': file.stat().st_size, 'sha256': sha(file, deadline)}
        result['protected_before'] = state['protected']
        if fresh:
            require(not any(store.iterdir()), 'fresh-snapshot store must start empty')
            result['fresh_intent'] = {'filename': fresh_filename, 'n_tokens': PROMPT_N,
                                      'save': 'explicit /slots/0?action=save native_stream'}
        else:
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
        free = shutil.disk_usage(store).free
        check('store_free', free >= 3 * (1 << 30), f'{free} bytes free')
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
        names_now = snapshot_names()
        if fresh:
            split = split_inventory(names_now, expected_names)
            check('xbox_originals_present', not split['missing'],
                  f"missing={split['missing']} extras={len(split['extras'])}")
            state['extras_before'] = split['extras']
            result['inventory'] = {'originals': sorted(expected_names), 'extras_before': split['extras']}
        else:
            require(names_now == expected_names, 'Xbox inventory differs from the qualified six')
        production_port = child(main_pid, privileged=True)[1]
        production_props = get(production_port, '/props')
        check('production_build', production_props.get('build_info') == EXPECTED_BUILD,
              str(production_props.get('build_info')))
        production_vocab = server_vocab(production_port)
        check('production_vocab', production_vocab in (None, VOCAB), f'n_vocab {production_vocab}')
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
        names_now = snapshot_names()
        if fresh:
            require(expected_names <= names_now and current_flags() == state['flags'],
                    'Xbox changed during isolation')
        else:
            require(names_now == expected_names and current_flags() == state['flags'],
                    'Xbox changed during isolation')
        require(not gpu_busy(), 'GPU already has a compute process; refusing overlap')
        peers = [line for line in subprocess.check_output(['ss', '-H', '-tn', 'state', 'established'],
                                                          text=True, timeout=remain(30)).splitlines()
                 if line.split() and line.split()[-1] == XBOX_HOST + ':50054']
        require(not peers, 'another local process has an established Xbox RKVA connection')
        before = snapshot_inventory(expected_names)
        check('xbox_inventory_before', before == expected_inventory, f'{len(before)} objects match prior evidence')
        result['six_snapshot_inventory_before'] = before
        state['before'] = before
        monitor_output = (out / 'vram.csv').open('w', buffering=1)
        state['monitor_output'] = monitor_output
        state['monitor'] = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.free,memory.used',
                                             '--format=csv,noheader,nounits', '-l', '1'],
                                            stdout=monitor_output, stderr=subprocess.DEVNULL)
        base = [str(binary), *argv[1:]]
        distributions = {}
        if fresh:
            port = start_session('live', False, store)
            live_record = result['sessions']['live'] = {'role': 'live', 'port': port}
            q0_save, _ = capture_q0('live', 'fresh_save_point', port, prompt,
                                    {'cache_n': 0, 'prompt_n': PROMPT_N, 'disk': False},
                                    'dist-live-save.json')
            live_record['q0_save'] = q0_save
            next_token = q0_save['tokens'][0]
            reference_tokens = fresh_reference_prompt(prompt, next_token)
            names_before = snapshot_names()
            save = save_native(port, fresh_filename)
            host_file = store / fresh_filename
            require(host_file.is_file() and not host_file.is_symlink(), 'saved host snapshot missing')
            check('fresh_host_size', host_file.stat().st_size == save['n_written'],
                  f"{host_file.stat().st_size} vs {save['n_written']}")
            trailer = native_trailer(host_file)
            check('fresh_trailer', trailer['origin'] == SPLIT_P and trailer['rows'] > 0,
                  json.dumps({k: v for k, v in trailer.items() if k != 'payload_bytes'})[:800])
            footer = inspect_host_footer(host_file)
            metadata = footer['metadata']
            check('fresh_metadata',
                  footer['n_tokens'] == PROMPT_N and metadata.get('n_tokens') == PROMPT_N and
                  metadata.get('position') == PROMPT_N - 1 and metadata.get('native_remote') is True and
                  metadata.get('native_split_p') == SPLIT_P and metadata.get('source_ctx') == 162816,
                  json.dumps(metadata)[:800])
            full_sha, key = file_hashes(host_file, trailer['frame_end'], deadline)
            check('fresh_key_shape', re.fullmatch(r'[0-9a-f]{64}', key) is not None, key)
            names_after = snapshot_names()
            new_keys = new_object_keys(names_before, names_after)
            reused = key in names_before
            result['fresh_inventory_probe'] = {'computed_key': key, 'key_reused': reused,
                                               'names_before': sorted(names_before),
                                               'names_after': sorted(names_after),
                                               'new_keys': new_keys}
            check('fresh_key_published', key in names_after, f'{key} not in Xbox store; new={new_keys}')
            check('fresh_key_new', new_keys == [key] or (not new_keys and reused),
                  f'new={new_keys} key={key} reused={reused}')
            if reused:
                warn('fresh_key_reused', f'{key} already existed; the save re-verified the object')
            result['fresh_snapshot'] = {
                'mode': 'fresh_snapshot', 'host_file': str(host_file), 'host_sha256': full_sha,
                'bytes': host_file.stat().st_size, 'n_saved': save['n_saved'],
                'xbox_key': key, 'namespace': f'{REMOTE_DIR}/{key}', 'key_reused': reused,
                'new_keys': new_keys, 'names_before': sorted(names_before),
                'names_after': sorted(names_after), 'trailer': trailer,
                'footer_metadata': metadata, 'reference_token': next_token,
                'reference_prompt_len': len(reference_tokens)}
            result['snapshot'] = {'mode': 'fresh_snapshot', 'n_tokens': PROMPT_N,
                                  'host_file': str(host_file), 'host_sha256': full_sha,
                                  'xbox_key': key, 'prefix_sha256': token_hash(prompt)}
            state['protected'][str(host_file)] = {'bytes': host_file.stat().st_size, 'sha256': full_sha}
            result['protected_before'] = state['protected']
            write_result()
            q0_ref, probs_ref = capture_q0('live', 'fresh_live_ref', port, reference_tokens,
                                           {'cache_n': PROMPT_N, 'prompt_n': 1, 'disk': False},
                                           'dist-live-ref.json')
            live_record['q0_ref'] = q0_ref
            result['quiescence_live'] = stop_private()
            require(not gpu_busy(), 'GPU still busy after live session')
            port = start_session('restored', True, store)
            restored_record = result['sessions']['restored'] = {'role': 'restored', 'port': port}
            q0_restored, probs_restored = capture_q0(
                'restored', 'fresh_restored', port, reference_tokens,
                {'cache_n': PROMPT_N, 'prompt_n': 1, 'disk': True}, 'dist-restored.json')
            restored_record['q0'] = q0_restored
            compare_logits('fresh_snapshot', q0_ref, probs_ref, q0_restored, probs_restored,
                           result['snapshot'])
            result['quiescence_restored'] = stop_private()
            require(not gpu_busy(), 'GPU still busy after restored session')
            result['rewind'] = {'mode': 'not_run',
                                'reason': 'fresh-snapshot mode qualifies full-vocabulary logits only'}
        else:
            for role, auto_restore in (('live', False), ('restored', True)):
                slot_store = store if auto_restore else store_live
                require(not auto_restore or (store / SNAPSHOT_NAME).is_file(), 'snapshot copy missing')
                port = start_session(role, auto_restore, slot_store)
                reference = distributions.get('live') if auto_restore else None
                record, probs = run_sequence(role, port, auto_restore, reference=reference,
                                             probes=auto_restore)
                distributions[role] = (record, probs)
                write_result()
                result[f'quiescence_{role}'] = stop_private()
                require(not gpu_busy(), 'GPU still busy after private session')
            rewinds = result['sessions']['restored'].get('rewinds', {})
            result['rewind'] = {
                'model_supports_row_reuse': bool(rewinds.get('q128', {}).get('rewind_reuse')),
                'q128': {key: rewinds.get('q128', {}).get(key) for key in
                         ('mode', 'rewind_reuse', 'planned_removed', 'keep_rows', 'timings', 'sha256')},
                'q128b': {key: rewinds.get('q128b', {}).get(key) for key in
                          ('mode', 'rewind_reuse', 'planned_removed', 'keep_rows', 'timings',
                           'deterministic', 'sha256')},
                'q129': {key: rewinds.get('q129', {}).get(key) for key in
                         ('mode', 'rewind_reuse', 'planned_removed', 'keep_rows', 'http_status', 'timings')},
                'q128_continuation_sha256': rewinds.get('q128', {}).get('sha256'),
                'q128b_continuation_sha256': rewinds.get('q128b', {}).get('sha256')}
            prior_rank = sum(1 for p in distributions['live'][1]
                             if p > distributions['live'][1][prior_tokens[0]])
            if prior_rank:
                warn('prior_first_token_rank',
                     f'prior sampled token {prior_tokens[0]} is rank {prior_rank} in the raw live distribution')
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
        try:
            quiescence = stop_private()
        except Exception as exc:
            quiescence = {'state': 'error', 'survivors': [], 'error': type(exc).__name__ + ': ' + str(exc)}
            result['cleanup_errors'].append('private process: ' + quiescence['error'])
        result['private_quiescence'] = {k: v for k, v in quiescence.items() if k != 'signaled'}
        result['private_survivors'] = quiescence.get('survivors', [])
        if quiescence.get('survivors'):
            result['cleanup_errors'].append(
                'private survivors after SIGKILL: ' + str(quiescence['survivors']))
            for pid in quiescence['survivors']:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if state['monitor_output'] is not None:
            try:
                minimum, count = sampled_vram_min((out / 'vram.csv').read_text().splitlines())
                result['vram_sampled_min_free_mib'] = minimum
                result['vram_sample_count'] = count
                if minimum < 18:
                    result['cleanup_errors'].append(f'sampled free VRAM {minimum} MiB below 18 MiB')
            except Exception as exc:
                result['cleanup_errors'].append('VRAM samples: ' + type(exc).__name__ + ': ' + str(exc))
        try:
            if state['flags'] is not None:
                current = current_flags()
                require(current == state['flags'], 'Xbox flags changed')
                result['flags_after'] = current
        except Exception as exc:
            result['cleanup_errors'].append('Xbox flags: ' + type(exc).__name__ + ': ' + str(exc))
        if state['paused']:
            result['production_restart_attempted'] = True
            try:
                try:
                    verify_pfn()
                    result['pfn_after'] = PFN
                except Exception as exc:
                    result['cleanup_errors'].append('Xbox PFN: ' + type(exc).__name__ + ': ' + str(exc))
                print('Restoring production (private survivors=%s).' % result['private_survivors'], flush=True)
                sudo('systemctl', 'start', SERVICE)
                status = wait_production()
                result['production_service_after'] = status.get('active')
                result['production_health_after'] = status
                if status.get('active') != 'active' or not status.get('props_ok'):
                    result['cleanup_errors'].append(
                        'production health: ' + json.dumps(status)[:500])
            except Exception as exc:
                result['cleanup_errors'].append('production service: ' + type(exc).__name__ + ': ' + str(exc))
        if state['before'] is not None:
            try:
                verify_pfn()
                result['six_snapshots_preserved'] = snapshot_inventory(set(state['before'])) == state['before']
                names_after = snapshot_names()
                result['snapshot_names_after'] = sorted(names_after)
                require(result['six_snapshots_preserved'], 'Xbox snapshot bytes changed')
                require(set(state['before']) <= names_after, 'one of the six Xbox objects disappeared')
                if fresh:
                    require(set(state.get('extras_before', [])) <= names_after,
                            'pre-existing extra Xbox object disappeared')
                    if 'fresh_snapshot' in result:
                        key = result['fresh_snapshot'].get('xbox_key')
                        require(key in names_after, 'new test object disappeared')
                        result['fresh_object_preserved'] = key in names_after
                        result['inventory_after'] = {'extras': sorted(names_after - set(state['before'])),
                                                     'new_key': key}
                else:
                    require(names_after == set(state['before']), 'Xbox snapshot names changed')
            except Exception as exc:
                result['cleanup_errors'].append('snapshot verification: ' + type(exc).__name__ + ': ' + str(exc))
        try:
            expected_file = fresh_filename if fresh else SNAPSHOT_NAME
            completed = ('fresh_snapshot' in result) if fresh else ('native_snapshot_copy' in result)
            if completed:
                allowed = re.compile(r'^\.llama-kv-snapshot-(?:ref-\d+\.ref|store\.lock)$')
                names = sorted(p.name for p in store.iterdir())
                live_names = sorted(p.name for p in store_live.iterdir())
                require(all(allowed.fullmatch(name) or name == expected_file for name in names) and
                        all(allowed.fullmatch(name) for name in live_names),
                        f'private slot store gained files: {names} / {live_names}')
                require(expected_file in names, 'private snapshot file disappeared')
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
    print(json.dumps({'status': result['status'], 'mode': 'fresh_snapshot' if fresh else 'existing_snapshot',
                      'result': str(out / 'result.json'),
                      'item4': str(out / ITEM4) if (out / ITEM4).is_file() else None,
                      'logits': result.get('logits', {}).get('metrics'),
                      'rewind': result.get('rewind'),
                      'fresh_snapshot': ({key: (result.get('fresh_snapshot') or {}).get(key) for key in
                                          ('xbox_key', 'namespace', 'host_file', 'host_sha256',
                                           'n_saved', 'key_reused', 'new_keys')}
                                         if fresh else None),
                      'inventory_after': result.get('inventory_after'),
                      'private_survivors': result.get('private_survivors'),
                      'production': result.get('production_service_after'),
                      'cleanup_errors': result['cleanup_errors']}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
