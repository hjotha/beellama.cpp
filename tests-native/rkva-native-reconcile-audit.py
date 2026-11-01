#!/usr/bin/env python3
"""Read-only native snapshot reconciliation audit.

Scans host native snapshot files, validates the RKVK trailer, derives the Xbox
object key (SHA-256 of the file through the RKVK frame end), and compares the
set with the Xbox object keys recorded by the latest qualification. It never
deletes, renames, saves or contacts the Xbox. Output is a JSON report.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path('/home/hjotha/beellama-benchmark-results/xxxl-plan-execution-20261030')
BASE = Path('/home/hjotha/worktrees/beellama-native-deploy-20261031/tests-native/native-restore-logits-rewind-qual-468.py')
FRESH = ROOT / 'native-restore-fresh-snapshot-468-r1/result.json'
OUT = ROOT / 'native-reconcile-audit-20261101.json'
ROOTS = [
    Path('/home/hjotha/llama-slot-cache'),
    Path('/home/hjotha/beellama-benchmark-results'),
]

spec = importlib.util.spec_from_file_location('qual468', BASE)
qual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qual)


def main():
    fresh = json.loads(FRESH.read_text())
    xbox_keys = set(fresh['inventory']['originals'])
    xbox_keys |= set(fresh['fresh_snapshot']['new_keys'])
    files = {}
    for root in ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob('*'):
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix not in ('.snapshot', '.bin'):
                continue
            if path.suffix == '.bin' and not (path.name.startswith('native-') or path.name.startswith('fresh-')):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size < (500 << 20):
                continue
            resolved = path.resolve()
            if resolved in files:
                continue
            try:
                trailer = qual.native_trailer(path)
                full, prefix = qual.file_hashes(path, trailer['frame_end'])
            except Exception as exc:
                files[resolved] = {'error': type(exc).__name__ + ': ' + str(exc)}
                continue
            files[resolved] = {'bytes': size, 'full_sha256': full, 'xbox_key': prefix,
                               'frame_end': trailer['frame_end'], 'path': str(path)}
    refs = {}
    errors = {}
    for path, info in files.items():
        if 'xbox_key' in info:
            refs.setdefault(info['xbox_key'], []).append(info['path'])
        else:
            errors[str(path)] = info['error']
    referenced = {k: refs.get(k, []) for k in sorted(xbox_keys)}
    orphan_host = {k: v for k, v in refs.items() if k not in xbox_keys}
    missing_host = sorted(k for k, v in referenced.items() if not v)
    report = {
        'scope': 'read-only host-file vs Xbox-key reconciliation; no delete',
        'xbox_keys': sorted(xbox_keys),
        'host_references': referenced,
        'missing_host_reference': missing_host,
        'host_keys_not_on_xbox': orphan_host,
        'host_files_scanned': len(files),
        'host_files_with_valid_trailer': sum(1 for i in files.values() if 'xbox_key' in i),
        'host_files_without_valid_trailer': len(errors),
        'errors': errors,
        'fresh_result': str(FRESH),
    }
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(OUT),
                      'xbox': len(xbox_keys),
                      'with_host_ref': len(xbox_keys) - len(missing_host),
                      'missing_host_ref': missing_host,
                      'host_keys_not_on_xbox': len(orphan_host),
                      'host_files_scanned': len(files)}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
