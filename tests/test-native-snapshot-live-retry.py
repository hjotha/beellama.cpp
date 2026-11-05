#!/usr/bin/env python3
"""Exercise repeated native snapshot reuse on an explicitly prepared local server.

The server must have an empty slot, automatic native restore enabled, and a
matching host/Xbox snapshot. Token IDs must extend that snapshot by at least
one token. No server or Xbox lifecycle is managed by this test.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--url')
    p.add_argument('--tokens', type=Path)
    p.add_argument('--minimum-cache', type=int)
    p.add_argument('--timeout', type=int, default=180)
    a = p.parse_args()
    if not a.execute:
        print('Dry run: provide --execute, a private local server, token file and minimum cached tokens')
        return
    url = urlparse(a.url or '')
    if (url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1') or
            not url.port or url.path not in ('', '/') or not a.tokens or
            not a.minimum_cache or a.minimum_cache < 1 or a.timeout < 1):
        p.error('requires a local HTTP server, token file, positive cache minimum and timeout')
    tokens = json.loads(a.tokens.read_text())
    if not isinstance(tokens, list) or len(tokens) <= a.minimum_cache or not all(type(x) is int for x in tokens):
        p.error('token file must contain an integer array longer than the expected cache')
    data = json.dumps({'prompt': tokens, 'n_predict': 8, 'temperature': 0, 'seed': 1234,
                       'ignore_eos': True, 'cache_prompt': True, 'return_tokens': True}).encode()
    results = []
    for _ in range(2):
        req = Request(a.url.rstrip('/') + '/completion', data=data,
                      headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=a.timeout) as response:
            result = json.load(response)
        timing = result['timings']
        if timing.get('cache_n', 0) < a.minimum_cache or timing.get('predicted_n') != 8:
            raise AssertionError('native prefix was not reused: ' + repr(timing))
        results.append(result)
    if results[0].get('tokens') != results[1].get('tokens'):
        raise AssertionError('restored continuations differ')
    print(json.dumps({'status': 'pass', 'first': results[0]['timings'],
                      'second': results[1]['timings'], 'continuations_equal': True}, indent=2))


if __name__ == '__main__':
    main()
