#!/usr/bin/env python3
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

base = sys.argv[1].rstrip('/')
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
results = []

def call(method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    elapsed = round((time.monotonic() - started) * 1000, 1)
    try:
        body = json.loads(raw)
    except Exception:
        body = raw.decode(errors='replace')
    result = {'method': method, 'path': path, 'status': status, 'elapsed_ms': elapsed, 'body': body}
    results.append(result)
    return result

def require(result, status):
    if result['status'] != status:
        raise AssertionError(f"{result['path']}: expected {status}, got {result['status']}: {result['body']}")

props = call('GET', '/props')
require(props, 200)
assert props['body']['adaptive_context']['profile'] == 'mtp', props

short_prompt = 'Explain adaptive cache in one sentence.'
short = call('POST', '/completion', {'prompt': short_prompt, 'id_slot': 0, 'cache_prompt': True, 'n_predict': 2, 'temperature': 0})
require(short, 200)
short_save = call('POST', '/slots/0?action=save', {'filename': 'short.bin'})
require(short_save, 200)
assert short_save['body']['n_saved'] > 0, short_save

long_prompt = ('adaptive context transition preserves the evaluated prompt state and reuses it safely. ' * 25).strip()
long = call('POST', '/completion', {'prompt': long_prompt, 'id_slot': 0, 'cache_prompt': True, 'n_predict': 1, 'temperature': 0})
require(long, 200)
props_long = call('GET', '/props')
require(props_long, 200)
assert props_long['body']['adaptive_context']['profile'] == 'long', props_long
assert props_long['body']['adaptive_context']['context_size'] == 512, props_long
long_save = call('POST', '/slots/0?action=save', {'filename': 'long.bin'})
require(long_save, 200)
assert long_save['body']['n_saved'] > 0, long_save

restore_short = call('POST', '/slots/0?action=restore', {'filename': 'short.bin'})
require(restore_short, 200)
props_short = call('GET', '/props')
require(props_short, 200)
assert props_short['body']['adaptive_context']['profile'] == 'mtp', props_short
assert props_short['body']['adaptive_context']['context_size'] == 256, props_short
short_reuse = call('POST', '/completion', {'prompt': short_prompt, 'id_slot': 0, 'cache_prompt': True, 'n_predict': 2, 'temperature': 0})
require(short_reuse, 200)

restore_long = call('POST', '/slots/0?action=restore', {'filename': 'long.bin'})
require(restore_long, 200)
props_long_again = call('GET', '/props')
require(props_long_again, 200)
assert props_long_again['body']['adaptive_context']['profile'] == 'long', props_long_again
assert props_long_again['body']['adaptive_context']['context_size'] == 512, props_long_again
long_reuse = call('POST', '/completion', {'prompt': long_prompt, 'id_slot': 0, 'cache_prompt': True, 'n_predict': 1, 'temperature': 0})
require(long_reuse, 200)

long_path = out / 'long.bin'
data = bytearray(long_path.read_bytes())
assert data
# Corruption must be rejected by the envelope checksum before any state is published.
data[len(data) // 2] ^= 0x01
(out / 'corrupt.bin').write_bytes(data)
corrupt = call('POST', '/slots/0?action=restore', {'filename': 'corrupt.bin'})
assert corrupt['status'] >= 400, corrupt
# A legacy/truncated target-only file has no adaptive identity and must fail explicitly.
(out / 'legacy.bin').write_bytes(b'\x01\x02\x03\x04')
legacy = call('POST', '/slots/0?action=restore', {'filename': 'legacy.bin'})
assert legacy['status'] >= 400, legacy

(out / 'result.json').write_text(json.dumps({'results': results}, indent=2) + '\n')
print(json.dumps({'steps': len(results), 'short_save': short_save['body'], 'long_save': long_save['body'], 'corrupt_status': corrupt['status'], 'legacy_status': legacy['status']}))
