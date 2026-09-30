import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
root = Path('/home/hjotha/beellama.cpp/docs/benchmarks/mtp-dflash2-shadow-init-smoke-20260929')
root.mkdir(parents=True, exist_ok=True)
base = Path('/home/hjotha/beellama.cpp')
server = base / 'build-dflash-xbox-dl/bin/llama-server'
model = '/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf'
draft = '/home/hjotha/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf'
port = 59577
password = os.environ.get('SUDO_PASS')
if not password:
    raise SystemExit('SUDO_PASS not set')
print('SMOKE_DIR=' + str(root), flush=True)


def rootctl(verb):
    result = subprocess.run(['sudo', '-S', '-p', '', 'systemctl', verb, 'llama-server-root.service'], input=password + '\n', text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError('root service ' + verb + ': ' + result.stderr.strip())


def userctl(verb):
    subprocess.run(['systemctl', '--user', verb, 'qwen35-4b-mtp-8092.service'], check=True, timeout=90)


def health(url):
    result = subprocess.run(['curl', '-fsS', '--retry', '120', '--retry-delay', '1', '--retry-connrefused', '--retry-all-errors', '--max-time', '5', url], capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr.decode()[-500:])
    return result.stdout.decode()


cmd = [str(server), '--model', model, '--device', 'CUDA0', '--n-gpu-layers', '99', '--fit', 'off', '--flash-attn', 'on', '--ctx-size', '16384', '--parallel', '1', '--batch-size', '64', '--ubatch-size', '64', '--cache-type-k', 'q4_0', '--cache-type-v', 'q4_0', '--host', '127.0.0.1', '--port', str(port), '--no-webui', '--spec-type', 'draft-mtp', '--spec-draft-n-max', '4', '--spec-draft-p-min', '0.70', '--spec-draft-type-k', 'q4_0', '--spec-draft-type-v', 'q4_0', '--spec-draft-shadow-model', draft, '--spec-draft-shadow-device', 'Vulkan0', '--spec-draft-shadow-ngl', 'all', '--spec-draft-shadow-n-max', '7']
(root / 'command.json').write_text(json.dumps(cmd, indent=2))

env = os.environ.copy()
env.pop('SUDO_PASS', None)
env['LD_LIBRARY_PATH'] = f'{base}/build/bin:{base}/build-dflash-xbox-dl/bin'
env['LD_PRELOAD'] = ':'.join(str(base / ('build-dflash-xbox-dl/bin/' + name)) for name in ['libllama.so.0.4.7', 'libllama-common.so.0.4.7', 'libllama-server-impl.so'])
env['GGML_BACKEND_PATH'] = str(base / 'build/bin/libggml-vulkan.so')

binaries = [server, base / 'build-dflash-xbox-dl/bin/libllama.so.0.4.7', base / 'build-dflash-xbox-dl/bin/libllama-common.so.0.4.7', base / 'build/bin/libggml-vulkan.so']
(root / 'binary-sha256.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries if p.exists()}, indent=2))
(root / 'git.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base, text=True) + subprocess.check_output(['git', 'status', '--short'], cwd=base, text=True))

short = 'Escreva os numeros inteiros de 1 a 200 separados por virgula, sem comentarios.'
expected = '031bf5c616d5b0a473d97b785404aede9b8a3a6714ea6b55e9908f560c0e80b8'

was_user = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'qwen35-4b-mtp-8092.service']).returncode == 0
was_root = subprocess.run(['systemctl', 'is-active', '--quiet', 'llama-server-root.service']).returncode == 0
stopped_user = False
stopped_root = False
result = {}
try:
    if was_user:
        userctl('stop')
        stopped_user = True
        print('qwen35 stopped', flush=True)
    if was_root:
        rootctl('stop')
        stopped_root = True
        print('root server stopped', flush=True)

    log = (root / 'server.log').open('wb')
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    try:
        health(f'http://127.0.0.1:{port}/health')
        if proc.poll() is not None:
            raise RuntimeError('server exit ' + str(proc.returncode))
        print('server loaded', flush=True)
        payload = {'prompt': short, 'n_predict': 192, 'temperature': 0, 'seed': 42, 'ignore_eos': True, 'cache_prompt': False, 'stream': False}
        req = urllib.request.Request(f'http://127.0.0.1:{port}/completion', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=360) as response:
            data = json.load(response)
        elapsed = time.monotonic() - t0
        (root / 'response.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
        content_hash = hashlib.sha256(data.get('content', '').encode()).hexdigest()
        timings = data.get('timings', {})
        result = {
            'content_sha256': content_hash,
            'expected_mtp_n4_hash': expected,
            'hash_matches_baseline': content_hash == expected,
            'decode_tps': timings.get('predicted_per_second'),
            'wall_s': round(elapsed, 3),
            'error': data.get('error'),
        }
        (root / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log.close()
finally:
    failures = []
    if stopped_root:
        try:
            rootctl('start')
            health('http://127.0.0.1:8090/health')
            print('root server restored', flush=True)
        except Exception as exc:
            failures.append('root: ' + str(exc))
    if stopped_user:
        try:
            userctl('start')
            health('http://127.0.0.1:8092/health')
            print('qwen35 restored', flush=True)
        except Exception as exc:
            failures.append('qwen35: ' + str(exc))
    if failures:
        raise RuntimeError('RESTORE FAILED: ' + '; '.join(failures))

log_text = (root / 'server.log').read_text(errors='replace')
markers = [line for line in log_text.splitlines() if 'shadow' in line.lower()]
(root / 'shadow-log-lines.txt').write_text('\n'.join(markers) + '\n')
print('SHADOW_MARKERS=' + str(len(markers)), flush=True)
for line in markers:
    print('LOG ' + line, flush=True)
if not result.get('hash_matches_baseline'):
    raise SystemExit('SMOKE FAILED: output hash differs from the Etapa 0 MTP n=4 baseline')
if not any('shadow auxiliary context ready' in line for line in markers):
    raise SystemExit('SMOKE FAILED: auxiliary context ready marker not found')
print('SMOKE OK', flush=True)
