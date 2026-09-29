import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request

root = Path('/home/hjotha/beellama.cpp/docs/benchmarks/mtp-dflash2-shadow-async-20260929')
root.mkdir(parents=True, exist_ok=True)
base = Path('/home/hjotha/beellama.cpp')
server = base / 'build-dflash-xbox-dl/bin/llama-server'
model = '/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf'
draft = '/home/hjotha/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf'
port = 59577
hwmon = next(Path('/sys/class/drm/card1/device/hwmon').glob('hwmon*'))
password = os.environ.get('SUDO_PASS')
if not password:
    raise SystemExit('SUDO_PASS not set')
print('RESULT_DIR=' + str(root), flush=True)


class CPUController:
    def __init__(self):
        self.log = (root / 'cpu-controller.stderr.log').open('w')
        controller = str(Path(__file__).with_name('cpu-controller.py'))
        self.process = subprocess.Popen(['sudo', '-k', '-S', '-p', '', 'python3', controller], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        self.lock = threading.Lock()
        self.process.stdin.write(password + '\n')
        self.process.stdin.flush()
        self.ready = json.loads(self.process.stdout.readline())
        if not self.ready.get('ready'):
            raise RuntimeError(str(self.ready))
        self.nominal = {str(k): float(v) for k, v in self.ready['nominal_mhz'].items()}
        if len(set(self.nominal.values())) != 1:
            raise RuntimeError('Mixed nominal CPU frequencies are not supported by this collector')
        self.reference = next(iter(self.nominal.values()))
        (root / 'cpu-original.json').write_text(json.dumps(self.ready, indent=2))
        print('CPU APERF/MPERF telemetry ready; reference ' + str(self.reference) + ' MHz', flush=True)

    def call(self, action):
        with self.lock:
            self.process.stdin.write(json.dumps({'action': action}) + '\n')
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('CPU controller stopped unexpectedly')
            response = json.loads(line)
            if response.get('error'):
                raise RuntimeError(response['error'])
            return response

    def close(self):
        state = self.call('close')
        (root / 'cpu-restored.json').write_text(json.dumps(state, indent=2))
        self.process.stdin.close()
        self.process.wait(timeout=10)
        self.log.close()
        if state.get('restored') != self.ready['original']:
            raise RuntimeError('CPU restore readback mismatch')


cpu = CPUController()


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


short = 'Escreva os numeros inteiros de 1 a 200 separados por virgula, sem comentarios.'
code = 'Escreva uma funcao Python merge_sort que recebe lista de inteiros. Inclua um exemplo e explique a complexidade em portugues.'
long = 'Leia os registros e depois explique as tres principais tendencias, com exemplos concretos.\n' + ''.join(f'Registro {i:04d}: setor {i%23}, lote {i*17%997}, temperatura {20+i%17} C, demanda {i*31%509} unidades; revisao aprovada com checksum {i*131%10007:04d}.\n' for i in range(320)) + '\nResumo dos registros:'
(root / 'prompts.json').write_text(json.dumps(dict(repeticao=short, codigo=code, longo=long), ensure_ascii=False, indent=2))
results = []
summaries = []


def sensor_snapshot(start):
    out = {'t_s': time.monotonic() - start}
    for key, path, scale in [('gpu_clock_mhz', hwmon / 'freq1_input', 1e6), ('apu_ppt_w', hwmon / 'power1_average', 1e6), ('temp_c', hwmon / 'temp1_input', 1e3), ('gpu_busy_pct', Path('/sys/class/drm/card1/device/gpu_busy_percent'), 1)]:
        try:
            out[key] = int(path.read_text()) / scale
        except (OSError, ValueError):
            pass
    return out


def request(prompt, n_predict, scenario, mode, tdp):
    payload = {'prompt': prompt, 'n_predict': n_predict, 'temperature': 0, 'seed': 42, 'ignore_eos': True, 'cache_prompt': False, 'stream': False}
    req = urllib.request.Request(f'http://127.0.0.1:{port}/completion', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    samples = []
    stop = threading.Event()
    t0 = time.monotonic()
    monitor_errors = []

    def monitor():
        try:
            previous = cpu.call('sample')
            while not stop.wait(.1):
                current = cpu.call('sample')
                dt = current['time'] - previous['time']
                sample = sensor_snapshot(t0)
                total_a = total_m = 0
                per_cpu = {}
                for core, counts in current['counters'].items():
                    a = (counts[0] - previous['counters'][core][0]) & ((1 << 64) - 1)
                    m = (counts[1] - previous['counters'][core][1]) & ((1 << 64) - 1)
                    if m:
                        mhz = cpu.nominal[core] * a / m
                        if mhz > 10000:
                            raise RuntimeError('Invalid APERF/MPERF sample')
                        per_cpu[core] = {'busy_mhz': mhz, 'busy_pct': 100 * m / (dt * cpu.nominal[core] * 1e6)}
                    total_a += a
                    total_m += m
                sample.update(cpu_aperf_delta=total_a, cpu_mperf_delta=total_m, cpu_busy_mhz=cpu.reference * total_a / total_m if total_m else None, cpu_util_pct=100 * total_m / (dt * cpu.reference * 1e6 * len(cpu.nominal)), cpu_per_core=per_cpu)
                samples.append(sample)
                previous = current
        except Exception as exc:
            monitor_errors.append(str(exc))

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(req, timeout=360) as response:
            data = json.load(response)
    finally:
        stop.set()
        thread.join(timeout=2)
    if monitor_errors:
        raise RuntimeError('CPU monitor failed: ' + '; '.join(monitor_errors))
    elapsed = time.monotonic() - t0
    timings = data.get('timings', {})
    if data.get('error'):
        raise RuntimeError(data['error'])
    (root / f'{mode}-{scenario}.response.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
    (root / f'{mode}-{scenario}.telemetry.json').write_text(json.dumps(samples, indent=2))
    decode = [s for s in samples if s['t_s'] > timings.get('prompt_ms', 0) / 1000]
    active = [s for s in decode if s.get('gpu_busy_pct', 0) >= 10]

    def avg(rows, key):
        vals = [s[key] for s in rows if key in s]
        return round(statistics.mean(vals), 2) if vals else None

    def maximum(rows, key):
        vals = [s[key] for s in rows if key in s]
        return round(max(vals), 2) if vals else None

    result = dict(mode=mode, tdp_w=tdp, scenario=scenario, context_size=16384, prompt_n=timings.get('prompt_n'), prompt_ms=timings.get('prompt_ms'), prompt_tps=timings.get('prompt_per_second'), predicted_n=timings.get('predicted_n'), decode_tps=timings.get('predicted_per_second'), decode_ms=timings.get('predicted_ms'), wall_s=round(elapsed, 3), content_sha256=hashlib.sha256(data.get('content', '').encode()).hexdigest(), decode_ppt_w=avg(decode, 'apu_ppt_w'), active_gpu_clock_mhz=avg(active, 'gpu_clock_mhz'), peak_temp_c=maximum(samples, 'temp_c'), telemetry_samples=len(samples))
    sum_a = sum(s.get('cpu_aperf_delta', 0) for s in decode)
    sum_m = sum(s.get('cpu_mperf_delta', 0) for s in decode)
    result['cpu_busy_mhz'] = round(cpu.reference * sum_a / sum_m, 2) if sum_m else None
    result['cpu_util_pct'] = avg(decode, 'cpu_util_pct')
    if not sum_m:
        raise RuntimeError('No effective CPU frequency samples recorded')
    if scenario != 'warmup':
        results.append(result)
        (root / 'results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
    return result


def nvidia_brief(path):
    out = subprocess.run(['nvidia-smi', '--query-gpu=power.limit,power.draw,clocks.sm,clocks.max.sm,temperature.gpu,memory.used,memory.total', '--format=csv,noheader'], capture_output=True, text=True)
    Path(path).write_text(out.stdout)


def mtp_shadow_cmd(n):
    return [str(server), '--model', model, '--device', 'CUDA0', '--n-gpu-layers', '99', '--fit', 'off', '--flash-attn', 'on', '--ctx-size', '16384', '--parallel', '1', '--batch-size', '64', '--ubatch-size', '64', '--cache-type-k', 'q4_0', '--cache-type-v', 'q4_0', '--host', '127.0.0.1', '--port', str(port), '--no-webui', '--spec-type', 'draft-mtp', '--spec-draft-n-max', str(n), '--spec-draft-p-min', '0.70', '--spec-draft-type-k', 'q4_0', '--spec-draft-type-v', 'q4_0', '--spec-draft-shadow-model', draft, '--spec-draft-shadow-device', 'Vulkan0', '--spec-draft-shadow-ngl', 'all', '--spec-draft-shadow-n-max', '7', '--gpu-power-backend', 'amdgpu', '--apu-tdp', '20', '--gpu-mem-clock-prefill', '2700', '--gpu-mem-clock-decode', '2700']


modes = [
    ('mtp-n4-shadow-sync', mtp_shadow_cmd(4), 20),
]

env = os.environ.copy()
for key in list(env):
    if key.startswith(('DFLASH_XBOX_', 'GGML_MTP_PROF', 'GGML_DFLASH_LOCAL_PROF')):
        env.pop(key)
    if key == 'SUDO_PASS':
        env.pop(key)
env['LD_LIBRARY_PATH'] = f'{base}/build/bin:{base}/build-dflash-xbox-dl/bin'
env['LD_PRELOAD'] = ':'.join(str(base / ('build-dflash-xbox-dl/bin/' + name)) for name in ['libllama.so.0.4.7', 'libllama-common.so.0.4.7', 'libllama-server-impl.so'])
env['GGML_BACKEND_PATH'] = str(base / 'build/bin/libggml-vulkan.so')
env['GGML_DFLASH_SHADOW_PROF'] = '1'

(root / 'git.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base, text=True) + subprocess.check_output(['git', 'status', '--short'], cwd=base, text=True))
binaries = [server, base / 'build-dflash-xbox-dl/bin/libllama.so.0.4.7', base / 'build-dflash-xbox-dl/bin/libllama-common.so.0.4.7', base / 'build/bin/libggml-vulkan.so', base / 'build/bin/libggml-cuda.so']
(root / 'binary-sha256.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries if p.exists()}, indent=2))
(root / 'draft-model-sha256.json').write_text(json.dumps({draft: hashlib.sha256(Path(draft).read_bytes()).hexdigest()}, indent=2))

was_user = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'qwen35-4b-mtp-8092.service']).returncode == 0
was_root = subprocess.run(['systemctl', 'is-active', '--quiet', 'llama-server-root.service']).returncode == 0
stopped_user = False
stopped_root = False
try:
    if was_user:
        userctl('stop')
        stopped_user = True
        print('qwen35 stopped', flush=True)
    if was_root:
        rootctl('stop')
        stopped_root = True
        print('root server stopped', flush=True)
    for mode, cmd, tdp in modes:
        policy_state = cpu.call('original')
        (root / f'{mode}.cpu-policy.json').write_text(json.dumps(policy_state, indent=2))
        nvidia_brief(root / f'{mode}.nvidia-before.csv')
        (root / f'{mode}.command.json').write_text(json.dumps(cmd, indent=2))
        with (root / f'{mode}.server.log').open('wb') as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            try:
                health(f'http://127.0.0.1:{port}/health')
                if proc.poll() is not None:
                    raise RuntimeError('server exit ' + str(proc.returncode))
                print(mode + ' loaded', flush=True)
                request(code, 64, 'warmup', mode, tdp)
                mode_results = []
                for i in range(3):
                    mode_results.append(request(short, 192, f'repeticao-{i}', mode, tdp))
                    mode_results.append(request(code, 192, f'codigo-{i}', mode, tdp))
                for i in range(2):
                    mode_results.append(request(long, 192, f'longo-{i}', mode, tdp))
                summary = {'mode': mode, 'tdp_w': tdp}

                def scenario_stats(name):
                    rows = [r for r in mode_results if r['scenario'].startswith(name)]
                    tps = [r['decode_tps'] for r in rows]
                    hashes = {r['content_sha256'] for r in rows}
                    entry = {'runs': len(rows), 'tps_median': round(statistics.median(tps), 3), 'tps_min': round(min(tps), 3), 'tps_max': round(max(tps), 3), 'hashes_equal': len(hashes) == 1, 'content_sha256': sorted(hashes)[0] if len(hashes) == 1 else None, 'hashes': sorted(hashes)}
                    if name == 'longo':
                        entry['prefill_tps'] = round(statistics.median([r['prompt_tps'] for r in rows]), 2)
                    return entry

                summary['repeticao'] = scenario_stats('repeticao')
                summary['codigo'] = scenario_stats('codigo')
                summary['longo'] = scenario_stats('longo')
                summary['mean_decode_ppt_w'] = round(statistics.mean([r['decode_ppt_w'] for r in mode_results if r['decode_ppt_w']]), 2)
                active_clocks = [r['active_gpu_clock_mhz'] for r in mode_results if r['active_gpu_clock_mhz']]
                summary['mean_active_gpu_clock_mhz'] = round(statistics.mean(active_clocks), 2) if active_clocks else None
                summary['mean_cpu_busy_mhz'] = round(statistics.mean([r['cpu_busy_mhz'] for r in mode_results if r['cpu_busy_mhz']]), 2)
                summary['peak_temp_c'] = max(r['peak_temp_c'] for r in mode_results if r['peak_temp_c'])
                summaries.append(summary)
                (root / 'summary.json').write_text(json.dumps(summaries, indent=2))
                print('SUMMARY ' + json.dumps(summary), flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
finally:
    failures = []
    try:
        cpu.close()
        print('CPU policy restored and verified', flush=True)
    except Exception as exc:
        failures.append('CPU restore: ' + str(exc))
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
print('RESULT_DIR=' + str(root), flush=True)
