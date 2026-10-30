#!/usr/bin/env python3
"""Exercise sidecar collision refusal on an isolated CPU server."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('server')
    parser.add_argument('model')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='slot-sidecar-http-') as tmp:
        root = Path(tmp)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        command = [args.server, '-m', args.model, '--host', '127.0.0.1', '--port', str(port),
                   '-c', '512', '-np', '1', '-t', '2', '--slot-save-path', tmp + '/',
                   '--slot-save-max-count', '1', '--slot-save-max-mb', '32']
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
                with urlopen(req, timeout=20) as response:
                    return response.status, json.load(response)
            except HTTPError as error:
                return error.code, json.load(error)
        try:
            if not ready.wait(30):
                raise RuntimeError('server failed to start: ' + ''.join(logs[-15:]))
            status, body = request('/completion', {'prompt': 'The quick brown fox', 'n_predict': 4,
                                                    'cache_prompt': True, 'id_slot': 0})
            assert status == 200, (status, body)
            for suffix in ('.logits', '.meta', '.ckpt', '.ckpts'):
                for existing in (False, True):
                    target = root / 'collision.bin'
                    side = root / ('collision.bin' + suffix)
                    if existing:
                        target.write_bytes(b'unknown target snapshot')
                    side.write_bytes(b'RKVK-independent-native-snapshot')
                    status, body = request('/slots/0?action=save', {'filename': target.name})
                    assert status == 400, (suffix, existing, status, body)
                    assert side.read_bytes() == b'RKVK-independent-native-snapshot'
                    assert target.read_bytes() == b'unknown target snapshot' if existing else not target.exists()
                    side.unlink()
                    if existing:
                        target.unlink()
                if os.name != 'nt':
                    target = root / 'dangling.bin'
                    side = root / ('dangling.bin' + suffix)
                    side.symlink_to(root / 'absent')
                    status, body = request('/slots/0?action=save', {'filename': target.name})
                    assert status == 400 and side.is_symlink() and not target.exists(), (status, body)
                    side.unlink()
                parent = root / 'parent.bin'
                parent.write_bytes(b'unknown parent state')
                status, body = request('/slots/0?action=save', {'filename': parent.name + suffix})
                assert status == 400 and parent.read_bytes() == b'unknown parent state', (status, body)
                parent.unlink()
            # Lock contention refuses a save before mutation, including cleanup.
            # This uses the same POSIX store lock file as the real handler.
            lock = (root / '.llama-kv-snapshot-store.lock').open('a+b')
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            pinned = root / 'locked.bin.logits'
            pinned.write_bytes(b'RKVK-independent-lock-test')
            status, body = request('/slots/0?action=save', {'filename': 'locked.bin'})
            assert status >= 400 and pinned.read_bytes() == b'RKVK-independent-lock-test' and not (root / 'locked.bin').exists(), (status, body)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
            pinned.unlink()
            status, body = request('/slots/0?action=save', {'filename': 'clean.bin'})
            assert status == 200 and (root / 'clean.bin').is_file(), (status, body)
            print('PASS: legacy HTTP save preserves colliding sidecars and accepts a clean destination')
        finally:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    main()
