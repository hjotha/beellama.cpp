#!/usr/bin/env python3
"""Run one private small-context RKVA probe; never changes services or snapshots.

Requires an already isolated GPU/console. A completed run is only evidence,
not production qualification; see native_companion_acceptance.py.
"""
import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

from native_companion_acceptance import FORMATS, assess


def prompt_for_request(tokens, variant, index):
    if variant == 'original' and index == 0:
        return list(tokens)
    replacement = next((token for token in tokens if token != tokens[-1]), None)
    if replacement is None:
        raise ValueError('changed prompt needs at least two different token IDs')
    return tokens[:-1] + [replacement]


def prompt_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def self_test():
    source = [1, 2, 3, 3]
    warm_a = prompt_for_request(source, 'original', 0)
    warm_b = prompt_for_request(source, 'original', 1)
    cold_b = prompt_for_request(source, 'changed-last', 0)
    assert warm_a == source and warm_b == cold_b and warm_b[:-1] == source[:-1]
    assert warm_b[-1] != source[-1] and prompt_hash(warm_b) == prompt_hash(cold_b)
    assert prompt_hash(warm_a) != prompt_hash(cold_b) and source == [1, 2, 3, 3]
    try:
        prompt_for_request([1, 1], 'changed-last', 0)
    except ValueError:
        pass
    else:
        raise AssertionError('unchangeable prompt must fail')
    print('PASS: cold B reproduces modified-last-token request and hash')


def server_command(a, port):
    companion_format = getattr(a, 'standard_companion', None) or a.format
    companion = ['--remote-attn-cache-type-k', companion_format, '--remote-attn-cache-type-v', companion_format]
    tail_tokens = getattr(a, 'tail_tokens', None)
    tail = [] if tail_tokens is None else ['--kv-tail-tokens', str(tail_tokens)]
    wire = getattr(a, 'wire', None) or ('fp8_e5m2' if a.format in ('saw_int4', 'turboquant_nc') else 'f16')
    return [a.server_binary, '-m', a.model, '--host', '127.0.0.1', '--port', str(port),
            '--ctx-size', str(a.context), '--batch-size', '256', '--ubatch-size', '256',
            '--threads', str(a.threads), '--threads-batch', str(a.threads),
            '--parallel', '1', '--device', 'CUDA0', '--gpu-layers', '99',
            '--flash-attn', 'on', '--fit', 'off', '--cache-type-k', 'kvarn4', '--cache-type-v', 'kvarn4',
            '--remote-attn', a.xbox, '--kv-position-split-p', str(a.origin), '--remote-attn-layers', 'full',
            *companion, *tail, '--remote-attn-format', a.format,
            '--remote-attn-wire-format', wire, '--remote-attn-stats',
            '--spec-type', 'none', '--spec-draft-n-max', '0', '--cache-ram', '0',
            '--no-context-shift', '--no-webui', '--verbosity', '4']


def server_environment(fmt, library_dir, standard_companion=False):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('LLAMA_ARG_', 'GGML_RKVA_', 'LLAMA_OVF_'))}
    env.update({'GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL': '1',
                'GGML_RKVA_RANGE_STORAGE': fmt,
                'GGML_RKVA_NATIVE_PRODUCER': 'cuda' if fmt == 'kvarn4' else 'cpu',
                'GGML_RKVA_RANGE_KERNEL': 'fine',
                'GGML_RKVA_DISK_SNAPSHOTS': '0'})
    if fmt in ('kvarn4', 'saw_int4', 'turboquant_nc') and not standard_companion:
        env['GGML_RKVA_NATIVE_COMPANION'] = '1'
    env['LD_LIBRARY_PATH'] = str(library_dir) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--server-binary', required=True)
    p.add_argument('--library-dir', required=True, type=Path, help='candidate shared libraries, first in loader path')
    p.add_argument('--model', required=True)
    p.add_argument('--prompt-tokens', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path, help='new dedicated directory')
    p.add_argument('--xbox', required=True, help='reserved HOST:PORT, normally 192.168.1.200:50054')
    p.add_argument('--format', choices=FORMATS, required=True)
    p.add_argument('--wire', choices=('f16', 'fp8_e5m2', 'fp8_e4m3fn'))
    p.add_argument('--tail-tokens', help='nonnegative count or none')
    p.add_argument('--standard-companion', choices=('q4_0', 'q8_0'), help='legacy control for native formats')
    p.add_argument('--origin', type=int, default=1024)
    p.add_argument('--context', type=int, default=2048)
    p.add_argument('--prompt-length', type=int, default=1280)
    p.add_argument('--predict', type=int, default=16)
    p.add_argument('--n-probs', type=int, default=0)
    p.add_argument('--requests', type=int, default=1)
    p.add_argument('--reuse-prefix', action='store_true')
    p.add_argument('--prompt-variant', choices=('original', 'changed-last'), default='original',
                   help='changed-last reproduces cold the second prompt from an original two-request run')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--timeout', type=int, default=600)
    a = p.parse_args()
    if a.standard_companion and a.format not in ('kvarn4', 'saw_int4', 'turboquant_nc'):
        p.error('--standard-companion is a control for native formats')
    if a.wire == 'fp8_e4m3fn' and a.format != 'kvarn4':
        p.error('E4M3FN requires native KVarN4')
    if a.tail_tokens is not None and (a.format != 'kvarn4' or
            (a.tail_tokens != 'none' and not a.tail_tokens.isdecimal())):
        p.error('--tail-tokens requires native KVarN4 and a nonnegative value or none')
    if a.prompt_variant == 'changed-last' and a.requests != 1:
        p.error('changed-last is a single cold request')
    if (a.origin <= 0 or a.origin % 256 or a.context % 256 or
            not a.origin < a.prompt_length or a.prompt_length + a.predict >= a.context or
            a.predict <= 0 or a.n_probs < 0 or a.requests <= 0 or a.threads <= 0 or a.timeout <= 0):
        p.error('invalid P/context/prompt/predict/thread/timeout geometry')
    try:
        host, port = a.xbox.rsplit(':', 1)
        if not re.fullmatch(r'[A-Za-z0-9.-]+', host) or not 1 <= int(port) <= 65535:
            raise ValueError('invalid console')
    except ValueError:
        p.error('--xbox must be HOST:PORT')
    raw = json.loads(a.prompt_tokens.read_text())
    tokens = raw.get('prompt') if isinstance(raw, dict) else raw
    if not isinstance(tokens, list) or len(tokens) < a.prompt_length or any(type(x) is not int or x < 0 for x in tokens[:a.prompt_length]):
        p.error('prompt file must contain enough nonnegative integer token IDs')
    tokens = tokens[:a.prompt_length]
    try:
        planned_prompts = [prompt_for_request(tokens, a.prompt_variant, i) for i in range(a.requests)]
    except ValueError as error:
        p.error(str(error))
    command = server_command(a, 0)
    if not a.execute:
        print(json.dumps({'mode': 'preview', 'starts_server': False, 'command_with_port_0': command,
                          'prompt_variant': a.prompt_variant,
                          'prompt_hashes': [prompt_hash(prompt) for prompt in planned_prompts]}, indent=2))
        return 0
    if not a.library_dir.is_dir() or not Path(a.server_binary).is_file() or not Path(a.model).is_file():
        raise ValueError('candidate binary, library directory, and model must exist')
    # An existing connection or compute process is never stopped by this tool.
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                         text=True, capture_output=True, check=True).stdout.strip()
    peers = subprocess.run(['ss', '-Htn', 'state', 'established', 'dst', f'{host}:{port}'],
                           text=True, capture_output=True, check=True).stdout.strip()
    if gpu or peers:
        raise RuntimeError('GPU or local Xbox connection is in use; isolate explicitly before execution')
    out = a.output.resolve()
    if out in (Path('/'), Path.home(), Path(__file__).resolve().parents[3]):
        raise ValueError('output must be a new dedicated directory')
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        listen_port = sock.getsockname()[1]
    command = server_command(a, listen_port)
    env = server_environment(a.format, a.library_dir.resolve(), bool(a.standard_companion))
    ready = threading.Event()
    with (out / 'server.log').open('w', encoding='utf-8') as logfile:
        proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        def drain():
            for line in proc.stdout:
                logfile.write(line)
                logfile.flush()
                if f'listening on http://127.0.0.1:{listen_port}' in line:
                    ready.set()
            ready.set()  # process exited before readiness
        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        response = None
        requests_completed = 0
        prompt_hashes = []
        failure = None
        try:
            if not ready.wait(a.timeout) or proc.poll() is not None:
                raise RuntimeError('isolated server failed to become ready')
            (out / 'process-maps.txt').write_text(Path(f'/proc/{proc.pid}/maps').read_text())
            payload = json.dumps({'prompt': tokens, 'n_predict': a.predict, 'temperature': -1 if a.n_probs else 0,
                                  'n_probs': a.n_probs,
                                  'ignore_eos': True, 'cache_prompt': a.reuse_prefix,
                                  'stream': False, 'return_tokens': True}).encode()
            for request_index in range(a.requests):
                updated = dict(json.loads(payload))
                updated['prompt'] = planned_prompts[request_index]
                payload = json.dumps(updated).encode()
                sent_prompt = json.loads(payload)['prompt']
                prompt_hashes.append(prompt_hash(sent_prompt))
                req = urllib.request.Request(f'http://127.0.0.1:{listen_port}/completion', data=payload,
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=a.timeout) as reply:
                    response = json.load(reply)
                requests_completed += 1
                (out / f'response-{request_index + 1}.json').write_text(json.dumps(response, ensure_ascii=False) + '\n')
        except Exception as error:
            failure = repr(error)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            thread.join(timeout=10)
    log = (out / 'server.log').read_text(encoding='utf-8', errors='replace')
    if response is not None:
        (out / 'response.json').write_text(json.dumps(response, ensure_ascii=False) + '\n')
    result = assess(a.format, log, a.origin)
    result.update({'command': command, 'host': a.xbox, 'context': a.context,
                   'prompt_length': a.prompt_length,
                   'prompt_sha256': prompt_hash(planned_prompts[0]),
                   'prompt_input_sha256': prompt_hash(tokens), 'prompt_variant': a.prompt_variant,
                   'request_finished': requests_completed == a.requests, 'request_error': failure,
                   'requests_requested': a.requests, 'requests_completed': requests_completed,
                   'prompt_hashes': prompt_hashes,
                   'process_exit': proc.returncode})
    if failure:
        result['status'] = 'FAIL'
        result['errors'].append('request or startup failed: ' + failure)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'errors': result['errors'],
                      'unverified': result['unverified'], 'artifacts': str(out)}, indent=2))
    return 1 if result['status'] == 'FAIL' else 3


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        self_test()
    else:
        raise SystemExit(main())
