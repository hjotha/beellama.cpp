#!/usr/bin/env python3
"""Real isolated adaptive server -> LLAMASLT v2 -> standalone classifier."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
from urllib.request import Request, urlopen

p = argparse.ArgumentParser()
p.add_argument('server')
p.add_argument('model')
p.add_argument('classifier')
args = p.parse_args()
with tempfile.TemporaryDirectory(prefix='legacy-classifier-http-') as tmp:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [args.server, '-m', args.model, '--host', '127.0.0.1', '--port', str(port),
               '--gpu-layers', '0', '--fit', 'off', '-c', '128', '--ctx-size-mtp', '64',
               '--mtp-max-tokens', '64', '--spec-type', 'draft-mtp', '--spec-draft-n-max', '2',
               '--spec-draft-n-min', '0', '-np', '1', '-t', '2', '-b', '32', '-ub', '32',
               '--slot-save-path', tmp + '/', '--slot-save-max-count', '1', '--cache-ram', '2048']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', GGML_DISABLE_VULKAN='1')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, env=env)
    ready = threading.Event()
    output = []
    def drain():
        for line in process.stdout:
            output.append(line)
            if 'listening on http://' in line:
                ready.set()
    threading.Thread(target=drain, daemon=True).start()
    def request(path, body):
        req = Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=30) as response:
            return json.load(response)
    try:
        if not ready.wait(40):
            raise RuntimeError('server failed to start: ' + ''.join(output[-25:]))
        result = request('/completion', {'prompt': 'The quick brown fox', 'n_predict': 2,
                                         'cache_prompt': True, 'id_slot': 0})
        assert not result.get('error'), result
        result = request('/slots/0?action=save', {'filename': 'adaptive.slot'})
        assert result.get('n_saved', 0) > 0, result
        sample = Path(tmp) / 'adaptive.slot'
        with sample.open('rb') as stream:
            assert stream.read(12) == b'LLAMASLT\x02\x00\x00\x00'
        result = subprocess.run([args.classifier, str(sample)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (result.stdout, result.stderr)
        size = sample.stat().st_size
        # The temporary fixture needs no auxiliaries. Remove them to exercise
        # an eligible local-only victim using the server's actual leases.
        for suffix in ('.logits', '.meta', '.ckpt', '.ckpts'):
            Path(str(sample) + suffix).unlink(missing_ok=True)
        result = request('/slots/0?action=save', {'filename': 'next.slot'})
        assert result.get('n_saved', 0) > 0, result
        assert not sample.exists(), 'old eligible snapshot was not evicted'
        assert (Path(tmp) / 'next.slot').exists(), 'new snapshot was not published'
        assert not list(Path(tmp).glob('retention.tmp-*')), 'retention staging was not cleaned'
        print('PASS: real LLAMASLT accepted and reversible retention used actual server locks, bytes=' + str(size))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
