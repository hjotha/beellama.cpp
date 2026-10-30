#!/usr/bin/env python3
"""Exercise real legacy checkpoint-append quota admission over HTTP on CPU."""
import argparse
import hashlib
import json
import math
import os
import resource
import signal
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def checkpoint_start(data):
    # SCKP/version/count then n_tokens, pos_min, pos_max and three blobs.
    marker = struct.pack('<II', 0x504b4353, 1)
    start = 0
    while True:
        start = data.find(marker, start)
        if start < 0:
            raise AssertionError('save has no complete checkpoint appendix')
        try:
            count, = struct.unpack_from('<I', data, start + 8)
            pos = start + 12
            if not 1 <= count <= 1024:
                raise ValueError('invalid checkpoint count')
            for _ in range(count):
                pos += 16  # int64 token count and two int32 positions
                for _ in range(3):
                    length, = struct.unpack_from('<Q', data, pos)
                    pos += 8 + length
                    if pos > len(data):
                        raise ValueError('checkpoint blob exceeds file')
            if pos == len(data):
                return start
        except (struct.error, ValueError):
            pass
        start += 1


class Server:
    def __init__(self, executable, model, directory, max_mb, max_file_size=None):
        self.logs = []
        self.ready = threading.Event()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        cmd = [executable, '-m', model, '--host', '127.0.0.1', '--port', str(self.port),
               '--gpu-layers', '0', '--fit', 'off', '-c', '512', '-np', '1', '-t', '2',
               '-b', '32', '-ub', '32', '--ctx-checkpoints', '8', '--checkpoint-min-step', '0',
               '--slot-save-path', str(directory) + '/', '--slot-save-max-count', '0',
               '--slot-save-max-mb', str(max_mb)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', GGML_DISABLE_VULKAN='1')
        def limit_file_size():
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_size, max_file_size))
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, env=env,
                                        preexec_fn=limit_file_size if max_file_size is not None else None)
        def drain():
            for line in self.process.stdout:
                self.logs.append(line)
                if 'listening on http://' in line:
                    self.ready.set()
        threading.Thread(target=drain, daemon=True).start()
        if not self.ready.wait(30):
            self.close()
            raise RuntimeError('server not ready: ' + ''.join(self.logs[-30:]))

    def request(self, path, body):
        req = Request(f'http://127.0.0.1:{self.port}{path}', json.dumps(body).encode(),
                      {'Content-Type': 'application/json'})
        try:
            with urlopen(req, timeout=30) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    def populate(self, prompt='The quick brown fox ' * 20):
        code, body = self.request('/completion', {'prompt': prompt,
                                                  'n_predict': 1, 'cache_prompt': True, 'id_slot': 0})
        assert code == 200, (code, body)

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('server')
    parser.add_argument('model')
    parser.add_argument('--quota-only', action='store_true', help='skip overwrite check for a negative-control build')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='legacy-append-quota-') as directory:
        root = Path(directory)
        baseline = root / 'baseline'
        baseline.mkdir()
        server = Server(args.server, args.model, baseline, 0)
        try:
            server.populate()
            code, body = server.request('/slots/0?action=save', {'filename': 'sample.slot'})
            assert code == 200, (code, body)
            saved = (baseline / 'sample.slot').read_bytes()
            base_size = checkpoint_start(saved)
            suffixes = ['.logits', '.meta', '.ckpt', '.ckpts']
            aux_size = sum((baseline / ('sample.slot' + suffix)).stat().st_size
                           for suffix in suffixes if (baseline / ('sample.slot' + suffix)).exists())
            assert len(saved) > base_size, (len(saved), base_size)
            assert any('appended ' in line and 'context checkpoint(s)' in line for line in server.logs)
        finally:
            server.close()
        # Keep a real, older snapshot and an unclassified fixture pinned. Set
        # the free logical quota strictly between the base and complete save.
        constrained = root / 'constrained'
        constrained.mkdir()
        existing = constrained / 'existing.slot'
        existing.write_bytes(saved)
        before = hashlib.sha256(saved).hexdigest()
        cap_mb = math.ceil((len(saved) * 2 + aux_size) / (1024 * 1024)) + 1
        cap = cap_mb * 1024 * 1024
        threshold = (base_size + len(saved)) // 2 + aux_size
        filler_size = cap - len(saved) - threshold
        assert filler_size > 0
        filler = constrained / 'pinned-filler'
        filler.write_bytes(b'P' * filler_size)
        assert base_size + aux_size < cap - len(saved) - filler_size < len(saved) + aux_size
        server = Server(args.server, args.model, constrained, cap_mb)
        try:
            server.populate('A quick brown fox jumps over a lazy dog. ' * 10)
            if not args.quota_only:
                # Replacing the old state removes it from the admission total,
                # so a same-sized replacement fits even though old+new would
                # not. The old bytes must be gone and the new bytes published.
                code, body = server.request('/slots/0?action=save', {'filename': 'existing.slot'})
                assert code == 200, (code, body)
                replaced = hashlib.sha256(existing.read_bytes()).hexdigest()
                assert replaced != before
                before = replaced
            code, body = server.request('/slots/0?action=save', {'filename': 'new.slot'})
            assert code == 400 and 'slot snapshot exceeds' in json.dumps(body), (code, body)
            assert hashlib.sha256(existing.read_bytes()).hexdigest() == before
            assert filler.stat().st_size == filler_size
            assert not (constrained / 'new.slot').exists()
            for suffix in suffixes:
                assert not (constrained / ('new.slot' + suffix)).exists()
            assert any('appended ' in line and 'context checkpoint(s)' in line for line in server.logs)
            print(f'PASS: base={base_size} appendix={len(saved)-base_size} sidecars={aux_size} '
                  f'available={threshold} cap={cap}; prior snapshot preserved')
        finally:
            server.close()
        # A real per-process file limit forces the checkpoint append to fail
        # after the base state was written. Existing data must survive and the
        # partial new output must disappear before any retention decision.
        failure = root / 'append-failure'
        failure.mkdir()
        existing = failure / 'existing.slot'
        existing.write_bytes(saved)
        before = hashlib.sha256(saved).hexdigest()
        file_limit = base_size + (len(saved) - base_size) // 2
        server = Server(args.server, args.model, failure, cap_mb, max_file_size=file_limit)
        try:
            server.populate()
            # Replacement must roll back the old state when the new checkpoint
            # append fails, not merely remove the partial new file.
            code, body = server.request('/slots/0?action=save', {'filename': 'existing.slot'})
            assert code >= 400 and 'incomplete context checkpoints' in json.dumps(body), (code, body)
            assert hashlib.sha256(existing.read_bytes()).hexdigest() == before
            code, body = server.request('/slots/0?action=save', {'filename': 'partial.slot'})
            assert code >= 400 and 'incomplete context checkpoints' in json.dumps(body), (code, body)
            assert hashlib.sha256(existing.read_bytes()).hexdigest() == before
            assert not (failure / 'partial.slot').exists()
            for suffix in suffixes:
                assert not (failure / ('partial.slot' + suffix)).exists()
            print(f'PASS: append I/O failure at file limit {file_limit}; prior snapshot preserved')
        finally:
            server.close()


if __name__ == '__main__':
    main()
