#!/usr/bin/env python3
"""GOKAYA graph headroom experiment. Run as root; production is restored on exit."""
import argparse, json, os, pathlib, subprocess, threading, time, urllib.request

ROOT = pathlib.Path('/home/hjotha/graph-headroom-20260908')
BIN = pathlib.Path(os.environ.get('LLAMA_GRAPH_TEST_BIN', '/home/hjotha/llama-releases/de57d0269/build/bin/llama-server'))
MODEL = '/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf'
URL = 'http://127.0.0.1:8095'
SERVICE = 'llama-server-root.service'

def http(path, body=None, headers=None):
    req = urllib.request.Request(URL + path, None if body is None else json.dumps(body).encode(),
                                 {'Content-Type': 'application/json', **(headers or {})})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.load(r)

def memory(pid=None):
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    result = {'gpu_used_free_mib': [int(x.strip()) for x in gpu.split(',')]}
    if pid:
        result['process_kib'] = {s.split(':')[0]: int(s.split()[1]) for s in
                                pathlib.Path(f'/proc/{pid}/status').read_text().splitlines()
                                if s.startswith(('VmRSS:', 'RssAnon:', 'RssFile:'))}
    return result

def run(mtp, ctx, tag, long=False, repeats=0, graphs=True):
    name = f'{tag}-{"mtp" if mtp else "nomtp"}-{ctx}'
    b = 64 if mtp else 512
    args = [str(BIN), '--model', MODEL, '--alias', 'graph-test', '--host', '127.0.0.1',
            '--port', '8095', '--ctx-size', str(ctx), '--batch-size', str(b), '--ubatch-size', str(b),
            '--parallel', '1', '--device', 'CUDA0', '--flash-attn', 'on', '--cache-type-k', 'q4_0',
            '--cache-type-v', 'q4_0', '--fit', 'off', '--cache-ram', '2048', '--ctx-checkpoints', '1',
            '--load-mode', 'none', '--gpu-power-prefill', '200', '--gpu-power-decode', '165',
            '--gpu-power-device', '0', '--metrics', '--slots',
            '--no-webui', '--no-context-shift']
    if os.environ.get('LLAMA_GRAPH_MEMORY_CLOCK'):
        args += ['--gpu-mem-clock-decode', os.environ['LLAMA_GRAPH_MEMORY_CLOCK']]
    if mtp:
        args += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '2', '--spec-draft-p-min', '0.80',
                 '--spec-draft-type-k', 'q4_0', '--spec-draft-type-v', 'q4_0']
    env = dict(os.environ, LD_LIBRARY_PATH=str(BIN.parent), GGML_CUDA_GRAPH_DEBUG='1')
    if os.environ.get('LLAMA_GRAPH_TRACE'):
        env['LD_PRELOAD'] = os.environ['LLAMA_GRAPH_TRACE']
    if not graphs:
        env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
    ready = threading.Event()
    listening = threading.Event()
    row = {'name': name, 'ctx': ctx, 'mtp': mtp, 'args': args, 'graphs_enabled': graphs,
           'unloaded_before': memory(), 'requests': []}
    with (ROOT / (name + '.log')).open('w') as f:
        proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        def consume():
            for line in proc.stdout:
                f.write(line)
                f.flush()
                if 'listening on http://' in line:
                    listening.set()
                    ready.set()
                if 'falling back' in line or 'CUDA error:' in line:
                    print(name, line.strip(), flush=True)
            ready.set()
        thread = threading.Thread(target=consume)
        thread.start()
        try:
            if not ready.wait(240) or not listening.is_set():
                raise RuntimeError('startup failed')
            row['loaded'] = memory(proc.pid)
            ids = http('/tokenize', {'content': 'The memory investigation tests different prompt lengths and complete inference. '})['tokens']
            sizes = [(9954, 265), (10952, 32), (15784, 32), (8439, 32), (1049, 32)]
            sizes += [(2048 + j, 16) for j in range(8, 129, 8)]
            if long:
                sizes += [(ctx - 4096, 4096), (10952, 32)]
            sizes += [(4096, 64)] * repeats
            for n, out in sizes:
                start = time.monotonic()
                response = http('/completion', {'prompt': (ids * (n // len(ids) + 1))[:n],
                                'n_predict': out, 'temperature': 0, 'ignore_eos': True,
                                'cache_prompt': False, 'seed': 42})
                r = {k: response.get(k) for k in ('tokens_evaluated', 'tokens_predicted', 'truncated', 'timings')}
                r.update(n=n, out=out, seconds=round(time.monotonic() - start, 2), memory=memory(proc.pid))
                row['requests'].append(r)
                print(name, json.dumps({k: r[k] for k in ('n', 'out', 'tokens_predicted', 'seconds', 'memory')}), flush=True)
                assert r['tokens_evaluated'] == n and out - 2 <= r['tokens_predicted'] <= out, r
                if not long and not repeats and 'falling back to direct execution' in (ROOT / (name + '.log')).read_text():
                    break
            row['completion_passed'] = True
        except Exception as e:
            row.update(completion_passed=False, error=repr(e))
            print(name, 'ERROR', repr(e), flush=True)
        finally:
            row['exit_before_stop'] = proc.poll()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            thread.join()
    log = (ROOT / (name + '.log')).read_text()
    row['fallbacks'] = log.count('falling back to direct execution')
    row['graph_instantiates'] = log.count('CUDA graph instantiate:')
    row['graph_updates'] = log.count('CUDA graph update-complete:')
    row['fatal_cuda'] = 'CUDA error:' in log
    row['unloaded_after'] = memory()
    row['passed'] = row.get('completion_passed', False) and not row['fallbacks'] and not row['fatal_cuda'] and (row['graph_instantiates'] > 0 or not graphs)
    (ROOT / (name + '.json')).write_text(json.dumps(row, indent=2))
    print('RESULT', json.dumps({k: row[k] for k in ('name', 'passed', 'fallbacks', 'graph_instantiates', 'unloaded_after')}), flush=True)
    return row

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--confirm', action='store_true')
    parser.add_argument('--mtp', action='store_true')
    parser.add_argument('--ctx', type=int)
    parser.add_argument('--tag', default='probe')
    parser.add_argument('--long', action='store_true')
    parser.add_argument('--repeats', type=int, default=0)
    parser.add_argument('--no-graphs', action='store_true')
    a = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    subprocess.run(['systemctl', 'stop', SERVICE], check=True)
    try:
        if a.sweep:
            results = []
            for mtp, base in [(True, 61184), (False, 98304)]:
                for delta in [0, 256, 512, 1024, 2048, 4096]:
                    r = run(mtp, base - delta, a.tag)
                    results.append(r)
                    if r['passed']:
                        break
            (ROOT / 'sweep.json').write_text(json.dumps(results, indent=2))
        elif a.confirm:
            selected = json.loads((ROOT / 'selected.json').read_text())
            rows = []
            for mtp in (True, False):
                key = 'mtp' if mtp else 'nomtp'
                for attempt in range(3):
                    r = run(mtp, selected[key], 'confirm', True, 20)
                    rows.append(r)
                    (ROOT / 'confirmation.json').write_text(json.dumps(rows, indent=2))
                    if r['passed']:
                        break
                    selected[key] -= 256
                else:
                    raise RuntimeError('confirmation failed: ' + key)
            (ROOT / 'selected.json').write_text(json.dumps(selected, indent=2))
        else:
            assert a.ctx
            run(a.mtp, a.ctx, a.tag, a.long, a.repeats, not a.no_graphs)
    finally:
        subprocess.run(['systemctl', 'start', SERVICE], check=True)

if __name__ == '__main__':
    main()
