#!/usr/bin/env python3
"""Exercise reversible replacement of proven-local legacy slot files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('server')
    parser.add_argument('model')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='slot-replacement-http-') as tmp:
        root = Path(tmp)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        command = [args.server, '-m', args.model, '--host', '127.0.0.1', '--port', str(port),
                   '-c', '512', '-np', '1', '-t', '2', '--slot-save-path', tmp + '/',
                   '--slot-save-max-count', '8', '--slot-save-max-mb', '32']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', GGML_DISABLE_VULKAN='1')
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, env=env)
        ready = threading.Event()
        logs = []
        def drain():
            for line in child.stdout:
                logs.append(line)
                if 'listening on http://' in line:
                    ready.set()
        threading.Thread(target=drain, daemon=True).start()
        def request(path, data):
            req = Request(f'http://127.0.0.1:{port}{path}', json.dumps(data).encode(),
                          headers={'Content-Type': 'application/json'})
            try:
                with urlopen(req, timeout=60) as response:
                    return response.status, json.load(response)
            except HTTPError as error:
                return error.code, json.load(error)
        def complete(prompt):
            status, body = request('/completion', {'prompt': prompt, 'n_predict': 4,
                                                    'cache_prompt': True, 'id_slot': 0})
            assert status == 200, (status, body)
        def save(name):
            return request('/slots/0?action=save', {'filename': name})
        try:
            if not ready.wait(30):
                raise RuntimeError('server failed to start: ' + ''.join(logs[-15:]))

            complete('The quick brown fox jumps over the lazy dog.')
            status, body = save('replace.bin')
            assert status == 200, (status, body)
            first = root / 'replace.bin'
            first_bytes = first.read_bytes()
            first_hash = sha(first)

            # A second, different prefill and save through the same pathname must
            # publish the new state and must not leave the old bytes behind.
            complete('A completely different prompt for the replacement test.')
            status, body = save('replace.bin')
            assert status == 200, (status, body)
            assert sha(first) != first_hash
            assert first.read_bytes() != first_bytes

            # Unknown sidecars cannot be adopted as part of the old unit.
            status, body = save('blocked.bin')
            assert status == 200, (status, body)
            blocked = root / 'blocked.bin'
            blocked_bytes = blocked.read_bytes()
            (root / 'blocked.bin.meta').write_bytes(b'legacy-auto-cache-index')
            status, body = save('blocked.bin')
            assert status == 400, (status, body)
            assert blocked.read_bytes() == blocked_bytes
            (root / 'blocked.bin.meta').unlink()
            (root / 'blocked.bin.ckpt').write_bytes(b'unverified-checkpoint')
            status, body = save('blocked.bin')
            assert status == 400, (status, body)
            assert blocked.read_bytes() == blocked_bytes
            (root / 'blocked.bin.ckpt').unlink()

            # A corrupt .logits sidecar cannot be silently replaced.
            (root / 'blocked.bin.logits').write_bytes(b'not-a-logits-sidecar')
            status, body = save('blocked.bin')
            assert status == 400, (status, body)
            assert blocked.read_bytes() == blocked_bytes
            assert (root / 'blocked.bin.logits').read_bytes() == b'not-a-logits-sidecar'
            (root / 'blocked.bin.logits').unlink()

            # A raw sequence carrying RKVK is a native pair, not a local file.
            native = root / 'native-like.bin'
            native.write_bytes(b'qsgg' + (4).to_bytes(4, 'little') + (1).to_bytes(4, 'little') +
                               (1).to_bytes(4, 'little') + b'state' + b'RKVK')
            native_bytes = native.read_bytes()
            status, body = save('native-like.bin')
            assert status == 400, (status, body)
            assert native.read_bytes() == native_bytes

            # A force-failed new save must not leave its partial destination.
            status, body = save('failure.bin')
            assert status == 200, (status, body)
            failure = root / 'failure.bin'
            failure_bytes = failure.read_bytes()
            (root / 'failure.bin.ckpts').write_bytes(b'corrupt-checkpoints')
            status, body = save('failure.bin')
            assert status == 400, (status, body)
            assert failure.read_bytes() == failure_bytes
            assert (root / 'failure.bin.ckpts').read_bytes() == b'corrupt-checkpoints'
            (root / 'failure.bin.ckpts').unlink()

            print('PASS: reversible legacy replacement preserves unknown/native units')
        finally:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    main()
