#!/usr/bin/env python3
"""Regression: LONG MTP q4 -> XL KVarN -> XXL, preserving complete prefixes.
Run with --server build-optimized/bin/llama-server --model MODEL.
Use a real Qwen MTP model with vocabulary. Default backend is CPU.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--server', required=True)
parser.add_argument('--model', required=True)
args = parser.parse_args()
with tempfile.TemporaryDirectory(prefix='adaptive-stream-', dir=os.environ.get('TMPDIR')) as temp:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    def request(path, body=None):
        req = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
            data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)
    cmd = [args.server, '--model', args.model, '--port', str(port), '--host', '127.0.0.1',
        '--device', 'none', '--gpu-layers', '0', '--no-host', '--no-repack',
        '--fit', 'off', '--parallel', '1',
        '--threads', '2', '--threads-batch', '2', '--flash-attn', 'on',
        '--ctx-size-s', '128', '--ctx-size-m', '256', '--ctx-size-l', '384',
        '--ctx-size-xl', '512', '--ctx-size-xxl', '768', '--spec-type', 'draft-mtp',
        '--spec-draft-n-max-s', '2', '--spec-draft-n-max-m', '2', '--spec-draft-n-max-l', '2',
        '--cache-type-k', 'q4_0', '--cache-type-v', 'q4_0',
        '--cache-type-k-l', 'q4_0', '--cache-type-v-l', 'q4_0',
        '--cache-type-k-xl', 'kvarn4', '--cache-type-v-xl', 'kvarn4',
        '--cache-type-k-xxl', 'kvarn4', '--cache-type-v-xxl', 'kvarn4',
        '--batch-size', '32', '--ubatch-size', '32', '--cache-ram', '2048',
        '--slot-save-path', temp+'/', '--slots', '--cache-prompt', '--no-context-shift']
    logpath = Path(temp)/'server.log'
    with logpath.open('w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=log)
        try:
            for _ in range(180):
                if proc.poll() is not None:
                    raise RuntimeError(logpath.read_text())
                try:
                    if request('/health')['status'] == 'ok':
                        break
                except Exception:
                    time.sleep(1)
            else:
                raise TimeoutError('server startup')
            chain = []
            for target, expected in [(300, 'long'), (400, 'xlong'), (600, 'xxlong')]:
                prompt = chain + [42] * (target - len(chain))
                body = request('/completion', {'prompt': prompt, 'n_predict': 16,
                    'temperature': 0, 'ignore_eos': True, 'cache_prompt': True, 'return_tokens': True})
                assert len(body['tokens']) == 16, body
                if chain:
                    assert body['timings']['cache_n'] >= len(chain)-1, body['timings']
                assert request('/props')['adaptive_context']['profile'] == expected
                chain = prompt + body['tokens']
                print(json.dumps({'profile': expected, 'timings': body['timings']}), flush=True)
            saved = request('/slots/0?action=save', {'filename': 'converted.bin'})
            assert saved['n_saved'] > 0, saved
            restored = request('/slots/0?action=restore', {'filename': 'converted.bin'})
            assert restored['n_restored'] > 0, restored
            assert 'adaptive streaming conversion: restored' in logpath.read_text()
        except Exception:
            print(logpath.read_text()[-16000:])
            raise
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
print('PASS: streaming conversion, prefix reuse, capacity transition and slot roundtrip')
