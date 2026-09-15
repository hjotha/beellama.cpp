#!/usr/bin/env python3
"""Compare identical tier swaps with and without conversation slot transfer."""
import importlib.util, json, os, pathlib, subprocess, threading
spec = importlib.util.spec_from_file_location('bench', pathlib.Path(__file__).with_name('bench-cuda-graph-headroom.py'))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

def run(transfer):
    selected = json.loads((b.ROOT / 'selected.json').read_text())
    name = 'route-transfer-' + ('on' if transfer else 'off')
    text = (b.ROOT / 'preset.before.ini').read_text().replace('61184', str(selected['mtp'])).replace('98304', str(selected['nomtp']))
    text = text.replace('94,208', format(selected['nomtp'] - 4096, ','))
    text = text.replace('; MMQ variant fix + varied-shape', '; CUDA Graph headroom + varied-shape')
    clock = os.environ.get('LLAMA_GRAPH_MEMORY_CLOCK')
    text = '\n'.join(line if not line.startswith('gpu-mem-clock-decode') else
                     ('gpu-mem-clock-decode = ' + clock if clock else '; Memory clock uses driver defaults.')
                     for line in text.splitlines()) + '\n'
    preset = b.ROOT / 'candidate.ini'
    preset.write_text(text)
    logpath = b.ROOT / (name + '.log')
    args = [str(b.BIN), '--models-preset', str(preset), '--models-max', '1', '--host', '127.0.0.1',
            '--port', '8095', '--metrics', '--slot-save-path', '/dev/shm']
    env = dict(os.environ, LD_LIBRARY_PATH=str(b.BIN.parent), GGML_CUDA_GRAPH_DEBUG='1')
    out = {'name': name, 'args': args, 'cycles': [], 'unloaded_before': b.memory()}
    baseline_files = set(pathlib.Path('/dev/shm').glob('llama-router-state-*/*'))
    ready = threading.Event()
    listening = threading.Event()
    with logpath.open('w') as f:
        proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        def consume():
            for line in proc.stdout:
                f.write(line)
                f.flush()
                if 'listening on http://' in line:
                    listening.set()
                    ready.set()
                if any(x in line for x in ('falling back', 'saved route state', 'restored route state', 'CUDA error:')):
                    print(name, line.strip(), flush=True)
            ready.set()
        thread = threading.Thread(target=consume)
        thread.start()
        try:
            assert ready.wait(240) and listening.is_set(), 'router startup'
            # Warm tokenizer using an explicitly small request.
            b.http('/completion', {'model': 'qwen-3.8-27b', 'prompt': 'Write OK.', 'n_predict': 1})
            ids = b.http('/tokenize', {'model': 'qwen-3.8-27b', 'content': 'The memory investigation tests different prompt lengths and complete inference. '})['tokens']
            suffix = b.http('/tokenize', {'model': 'qwen-3.8-27b', 'content': ' Continue the same sequence.'})['tokens']
            answer = ' memory' * 64
            for i in range(4):
                headers = {'X-Conversation-Id': f'graph-headroom-{name}-{i}'} if transfer else {}
                prefix = (ids * (4096 // len(ids) + 1))[:4096]
                def request(prompt, budget):
                    r = b.http('/completion', {'model': 'qwen-3.8-27b', 'prompt': prompt,
                               'n_predict': budget, 'temperature': 0, 'grammar': 'root ::= ' + json.dumps(answer),
                               'cache_prompt': True, 'return_tokens': True}, headers)
                    assert r['content'] == answer, r
                    assert r['tokens_evaluated'] == len(prompt), r
                    return r
                first = request(prefix, 128)
                extended = prefix + first['tokens'] + suffix
                second = request(extended, selected['mtp'] + 1 - len(extended))
                assert (second['timings']['cache_n'] > 0) == transfer, second
                after = b.memory(proc.pid)
                files = set(pathlib.Path('/dev/shm').glob('llama-router-state-*/*')) - baseline_files
                row = {'cycle': i, 'first': first, 'second': second, 'router_memory': after,
                       'remaining_snapshot_bytes': sum(x.stat().st_size for x in files if x.is_file())}
                assert row['remaining_snapshot_bytes'] == 0, row
                out['cycles'].append(row)
                print(name, json.dumps({'cycle': i, 'cache_n': second['timings']['cache_n'], 'memory': after}), flush=True)
            out['completed'] = True
        except Exception as e:
            out.update(completed=False, error=repr(e))
            print(name, 'ERROR', repr(e), flush=True)
        finally:
            out['exit_before_stop'] = proc.poll()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(45)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            thread.join()
    log = logpath.read_text()
    out.update(fallbacks=log.count('falling back to direct execution'), graph_instantiates=log.count('CUDA graph instantiate:'),
               saves=log.count('saved route state for conversation'), restores=log.count('restored route state for conversation'),
               fatal_cuda='CUDA error:' in log, unloaded_after=b.memory())
    out['passed'] = out.get('completed', False) and not out['fallbacks'] and not out['fatal_cuda'] and out['graph_instantiates'] > 0 and out['saves'] == out['restores'] == (4 if transfer else 0)
    (b.ROOT / (name + '.json')).write_text(json.dumps(out, indent=2))
    print('RESULT', json.dumps({k: v for k, v in out.items() if k not in ('cycles', 'args')}), flush=True)
    return out

if __name__ == '__main__':
    subprocess.run(['systemctl', 'stop', b.SERVICE], check=True)
    try:
        rows = [run(False), run(True)]
        assert all(r['passed'] for r in rows), 'route memory gate failed'
    finally:
        subprocess.run(['systemctl', 'start', b.SERVICE], check=True)
