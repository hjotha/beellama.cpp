#!/usr/bin/env python3
"""Record/compare exact native KVarN MQ2/MQ4 requests and attention replies.

Explicit console lease and --execute required. Reuses the original bulk/serial
oracle unchanged; ignores timing differences across packages. Upload requests,
query requests, OUT and LSE must all retain their exact byte hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--host')
    parser.add_argument('--port', type=int, default=50054)
    parser.add_argument('--output')
    parser.add_argument('--reference')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'execute': False, 'plan': 'Native F32 bulk/serial MQ4 mode1 and MQ2 mode2, both tail policies, exact request/OUT/LSE hashes across packages'}))
        return 0
    if not args.host or not args.output or not 0 < args.port < 65536:
        parser.error('--execute requires --host, --output and a valid port')

    import rkva_range_probe as proto
    import rkva_kvarn_bulk_probe as probe
    result = {'schema': 'rkva-native-bulk-replay-v1',
              'probe_sha256': hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest(),
              'modes': {}, 'records': []}
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    if reference is not None and (reference.get('schema') != result['schema'] or
                                  reference.get('probe_sha256') != result['probe_sha256'] or
                                  not reference.get('ok') or not reference.get('records')):
        parser.error('reference is not an approved replay of this exact probe')

    original = proto.Client.call
    for mode in (1, 2):
        def audited(client, op, payload, *positional, **kwargs):
            if op == 20 and len(payload) == 16 and struct.unpack_from('<I', payload)[0] in (0, 3):
                payload = struct.pack('<IIQ', struct.unpack_from('<I', payload)[0], mode, 0)
            reply = original(client, op, payload, *positional, **kwargs)
            if op in (14, 16, 22) and kwargs.get('success', True):
                record = {'mode': mode, 'op': op, 'session': kwargs.get('session', 0),
                          'nq': kwargs.get('nq', 0), 'request_bytes': len(payload),
                          'request_sha256': hashlib.sha256(payload).hexdigest()}
                # Upload generation acknowledgements are not attention data.
                if op != 16:
                    blob = bytes(reply)
                    record.update(reply_bytes=len(blob), reply_sha256=hashlib.sha256(blob).hexdigest())
                result['records'].append(record)
            return reply

        proto.Client.call = audited
        try:
            result['modes'][str(mode)] = probe.run(args.host, args.port)
        finally:
            proto.Client.call = original

    result['ok'] = all(run.get('ok') for run in result['modes'].values())
    if reference is not None:
        result['reference'] = str(Path(args.reference).resolve())
        result['requests_and_attention_replies_equal'] = result['records'] == reference['records']
        result['ok'] = result['ok'] and result['requests_and_attention_replies_equal']
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'ok': result['ok'], 'records': len(result['records']),
                      'reference': args.reference}, indent=2))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
