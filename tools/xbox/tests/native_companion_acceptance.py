#!/usr/bin/env python3
"""Audit an isolated RKVA run without touching the server or Xbox.

The run must use a private log and a prompt extending past P.  This checker
never upgrades externally supplied files to trusted process/allocation proof.
"""
import argparse
import json
import re
import sys
from pathlib import Path

FORMATS = ('kvarn4', 'saw_int4', 'turboquant_nc', 'q4_0', 'q8_0', 'f16')
NATIVE = frozenset(('kvarn4', 'saw_int4', 'turboquant_nc'))
MIXED = re.compile(r'mixed KV layer=(\d+)\s+device=\S+\s+type_k=(\S+)\s+type_v=(\S+)')
NATIVE_OWNER = re.compile(r'native companion negotiated format=(\S+) standard_cache=absent')
STATS = 'remote_attn stats '


def assess(fmt, text, origin):
    """Evaluate actual storage/calls, and classify companion evidence separately."""
    if fmt not in FORMATS or origin <= 0:
        raise ValueError('unknown format or invalid origin')
    errors, missing = [], []
    rows = [(int(a), b, c) for a, b, c in MIXED.findall(text)]
    stats = []
    for line in text.splitlines():
        if STATS in line:
            try:
                stats.append(json.loads(line.split(STATS, 1)[1]))
            except (ValueError, TypeError):
                errors.append('malformed remote statistics')
    if not stats:
        missing.append('remote statistics')
        latest = {}
    else:
        latest = stats[-1]
        expected = {'range_cache_type': fmt, 'range_origin': origin}
        for key, value in expected.items():
            if latest.get(key) != value:
                errors.append(f'{key}: expected {value!r}, observed {latest.get(key)!r}')
        for key in ('range_upload_calls', 'range_upload_rows', 'range_upload_bytes', 'range_attn_calls'):
            value = latest.get(key)
            if type(value) is not int or value <= 0:
                errors.append(f'{key} must be a positive integer')
        if type(latest.get('range_query_max')) is not int or latest['range_query_max'] < origin:
            errors.append('no confirmed query at or above P')
        if fmt in NATIVE and latest.get('range_source_cache_type') != 'projected_f32':
            errors.append('native records must come from original projected F32')
    native_owners = NATIVE_OWNER.findall(text)
    if fmt in NATIVE and native_owners and any(owner != fmt for owner in native_owners):
        errors.append(f'native companion format does not match {fmt}: {native_owners!r}')
    if not rows and not (fmt in NATIVE and native_owners):
        missing.append('actual companion allocation and format')
    else:
        bad = [(layer, k, v) for layer, k, v in rows if (k, v) != (fmt, fmt)]
        if bad:
            errors.append(f'companion allocation does not match {fmt}: {bad[:4]!r}')
    # The log may be historical or supplied by another process. Never infer a
    # trusted allocation or state/snapshot result from a user-provided file.
    missing.extend(('authenticated process/log provenance', 'companion state/restore/rewind qualification'))
    return {
        'format': fmt,
        'status': 'FAIL' if errors else 'UNVERIFIED',
        'errors': errors,
        'unverified': missing,
        'observed_companion_layers': rows,
        'observed_native_owners': native_owners,
        'remote_stats': latest,
        'warning': 'External log and manifest are unverified; this audit cannot qualify a release.',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', required=True, choices=FORMATS)
    parser.add_argument('--log', required=True, type=Path,
                        help='private server log from the isolated hardware run')
    parser.add_argument('--origin', required=True, type=int, help='P used in that run')
    parser.add_argument('--manifest', type=Path,
                        help='optional externally supplied manifest (reported as unverified)')
    args = parser.parse_args(argv)
    try:
        result = assess(args.format, args.log.read_text(encoding='utf-8', errors='replace'), args.origin)
        if args.manifest:
            result['external_manifest'] = json.loads(args.manifest.read_text(encoding='utf-8'))
            result['external_manifest_verified'] = False
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # No external evidence can satisfy provenance and state qualifications.
    return 1 if result['errors'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
