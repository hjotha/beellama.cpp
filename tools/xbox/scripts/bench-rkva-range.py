#!/usr/bin/env python3
"""Fresh RKVA range baseline; preview by default, --execute explicitly runs GPU/RPC.

Only an explicitly requested private-child prefix snapshot may touch disk slot state. Token input is a JSON array
(or a legacy request object's integer `prompt` array); repetition is opt-in.
The current server counts predicted_n-1 timed decode steps, since its first
output token comes from prefill. Preserve log and API timing independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
TIMING = re.compile(r"(?P<prompt>prompt\s+)?eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second")
BAD_LOG = re.compile(r"GGML_ABORT|GGML_ASSERT|CUDA error|out of memory|failed to (?:compute|decode)|decode failed|graph_compute.*failed|fallback to|falling back|async_overlap=0|wire=f16|CUDA graph[^\n]*failed|cannot query CUDA graph", re.I)
GRAPH_POLICY_WARNING = re.compile(
    r"disabl\w* CUDA graphs|CUDA graphs (?:remain disabled|re-enabled)|"
    r"released \d+ CUDA graph executables and \d+ captures under VRAM pressure|"
    r"CUDA graph instantiation ran out of memory; falling back to direct execution", re.I)
GRAPH_POLICY_FAILURE = re.compile(r"CUDA (?:error|graph[^\n]*failed)|cannot query CUDA graph", re.I)
GRAPH_GUARD_KEY = 'GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB'
GRAPH_GUARD_MIB = 128
PINNED_ENV = {
    GRAPH_GUARD_KEY: str(GRAPH_GUARD_MIB),
    # Match the configured production KVarN window; do not use memory-adaptive 64k windows.
    'GGML_KVARN_WINDOW_CHUNK': '4096',
    'GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL': '1',
    'GGML_RKVA_NATIVE_PRODUCER': 'cuda', 'GGML_RKVA_RANGE_STORAGE': 'kvarn4',
    'GGML_RKVA_RANGE_KERNEL': 'fine', 'GGML_RKVA_RANGE_FUSE': '0',
    'GGML_RKVA_NATIVE_TAIL_TOKENS': '128', 'GGML_RKVA_PREFILL_QUERY_CHUNK': '8',
    'GGML_RKVA_NATIVE_BULK': 'f16', 'GGML_RKVA_WIRE_FORMAT': 'fp8',
    'GGML_RKVA_WIRE_STRICT': '1', 'GGML_RKVA_ASYNC_LOCAL_REMOTE': '1',
    'GGML_RKVA_CUDA_FP8': 'prefill', 'GGML_RKVA_DISK_SNAPSHOTS': '0',
    'GGML_RKVA_WAVEFRONT_TILE': '0', 'GGML_RKVA_WAVEFRONT_RPC_BATCH': '1',
    'GGML_RKVA_TCP_PREFETCH': '0',
    'GGML_RKVA_LOCAL_INPUT_REUSE': '0',
    'GGML_CUDA_MASK_PACK': '0',
    'LLAMA_N256_LOOKAHEAD': '0', 'LLAMA_SERVER_F5_PER_TOKEN_TIMING': '1',
}


def memory_policy():
    # common.cuh reads this one variable with atoi(), defaulting to 18 MiB.
    # Freeze 128 for both arms; never silently weaken a stricter inherited guard.
    source = ROOT / 'ggml/src/ggml-cuda/common.cuh'
    source_bytes = source.read_bytes()
    match = re.search(rb'const int headroom_mb\s*=\s*env\s*!=\s*nullptr\s*\?\s*atoi\(env\)\s*:\s*(\d+)\s*;', source_bytes)
    if not match or GRAPH_GUARD_KEY.encode() not in source_bytes:
        raise ValueError('unrecognized CUDA graph guard source; review the memory policy before benchmarking')
    default = int(match[1])
    inherited = os.environ.get(GRAPH_GUARD_KEY)
    if inherited is not None and (not inherited.isdecimal() or not inherited.isascii()):
        raise ValueError('inherited CUDA graph guard must be a nonnegative integer MiB value')
    if max(default, int(inherited or 0)) > GRAPH_GUARD_MIB:
        raise ValueError('fixed 128 MiB graph guard would lower a stricter source/inherited policy; refusing')
    return {'variable': GRAPH_GUARD_KEY, 'effective_mib': GRAPH_GUARD_MIB,
            'source_default_mib': default, 'inherited_value': inherited,
            'source': str(source), 'source_symbol': 'ggml_backend_cuda_context::cuda_graph_memory_headroom_bytes',
            'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
            'rule': 'same explicit 128 MiB in both arms; stricter inherited/source policy is rejected, never lowered'}


def bootstrap_f16_confirmed_fp8(lines, index):
    # HELLO describes the legacy base transport before native-bulk negotiation.
    # Excuse only this exact bootstrap line, and only when THIS connection's
    # subsequent native negotiation confirms the required FP8/async profile.
    if not re.search(r'RemoteAttn: connected to \S+:\d+, \d+ remote layers, wire=f16, budget=\d+ MiB\s*$', lines[index]):
        return False
    for line in lines[index + 1:]:
        if 'RemoteAttn: connected to ' in line:
            return False
        if 'EXPERIMENTAL native KVarN4' in line:
            return bool(re.search(r'EXPERIMENTAL native KVarN4[^\n]*precision_tail=128[^\n]*async_overlap=1[^\n]*wire=fp8_e5m2\b', line))
    return False


def classify_logs(lines):
    lines = list(lines)
    errors, warnings = set(), set()
    for index, line in enumerate(lines):
        if bootstrap_f16_confirmed_fp8(lines, index):
            continue
        if GRAPH_POLICY_WARNING.search(line) and not GRAPH_POLICY_FAILURE.search(line):
            warnings.add(line.strip())
        elif BAD_LOG.search(line):
            errors.add(line.strip())
    return sorted(errors), sorted(warnings)


def validate_startup(args, lines):
    text = ''.join(lines)
    if not re.search(r'EXPERIMENTAL native KVarN4[^\n]*precision_tail=128[^\n]*async_overlap=1[^\n]*wire=fp8_e5m2', text):
        raise RuntimeError('required native KVarN4/tail128/async1/FP8 negotiation was not confirmed')
    if not re.search(r'position-split: P=' + str(args.p) + r' C=' + str(args.context) + r'\b', text):
        raise RuntimeError('effective P/C geometry was not confirmed in server logs')
    for field, expected in [('n_batch', args.batch), ('n_ubatch', args.ubatch)]:
        if not re.search(r'\b' + field + r'\s*=\s*' + str(expected) + r'\b', text):
            raise RuntimeError(f'effective {field}={expected} was not confirmed in server logs')
    errors, warnings = classify_logs(lines)
    if errors:
        raise RuntimeError('server startup logged an error/backend fallback')
    return warnings


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    size = Path(path).stat().st_size
    consumed = 0
    milestone = 1 << 30
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            digest.update(block)
            consumed += len(block)
            if consumed >= milestone:
                print(f'hashing {Path(path).name}: {consumed}/{size} bytes', flush=True)
                milestone += 1 << 30
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'inode': stat.st_ino, 'sha256': sha256(path)}


def token_hash(tokens):
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack('<I', token))
    return digest.hexdigest()


def effective_tokens(path, length, repeat=False):
    value = json.loads(Path(path).read_text())
    tokens = value.get('prompt') if isinstance(value, dict) else value
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int or not 0 <= t <= 0x7fffffff for t in tokens):
        raise ValueError('prompt-tokens must contain a nonempty array of nonnegative int32 token IDs')
    copies = math.ceil(length / len(tokens))
    if copies > 1 and not repeat:
        raise ValueError(f'{len(tokens)} IDs cannot cover {length}; --repeat-prompt must be explicit')
    stream = (tokens * copies)[:length]
    return stream, {'source_count': len(tokens), 'copies': copies, 'effective_count': len(stream),
                    'format': 'little-endian uint32 token IDs', 'stream_sha256': token_hash(stream)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('server-binary', 'model', 'prompt-tokens', 'output', 'xbox-host'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--points', default='102400,130560', help='strictly ascending prompt lengths')
    p.add_argument('--p', type=int, default=102400)
    p.add_argument('--context', type=int, default=131072)
    p.add_argument('--batch-size', '--batch', dest='batch', type=int, choices=(256, 512, 1024), default=256,
                   help='logical B512 baseline or common B1024 for controlled two/four N256 parents')
    p.add_argument('--ubatch', type=int, choices=(256, 512, 1024), default=256,
                   help='OFF uses256; WF256 uses parent512/two or1024/four N256 tiles')
    p.add_argument('--wavefront-tile', type=int, choices=(0, 128, 256), default=0,
                   help='CUDA/RKVA: OFF UB256, WF128 parentUB256, or WF256 parentUB512/1024')
    p.add_argument('--wavefront-tiles', type=int, choices=(2, 4), default=2,
                   help='four N256 tiles require B1024/UB1024; compare two tiles at B1024/UB512, same physical N256')
    p.add_argument('--n256-lookahead', action='store_true',
                   help='qualify the 2xN256 prototype (batch-size 512); timings require explicit CUDA 2x256 runtime activation')
    p.add_argument('--cuda-fp8', choices=('prefill', 'mapped'), default='prefill',
                   help='mapped also uses CUDA mapped-host FP8 helpers for nq<=9; requires independent decode quality/TPS checks')
    p.add_argument('--tcp-prefetch', action='store_true',
                   help='experimental two-frame FP8 prefill transport; requires runtime overlap counters')
    p.add_argument('--local-input-reuse', action='store_true',
                   help='reuse byte-identical local masks/indices on device; requires actual skipped-copy counters')
    p.add_argument('--mask-pack', choices=('off', 'on', 'default'), default='off',
                   help='lossless CUDA binary-F16 mask transport; controls pin OFF, default removes the core override')
    p.add_argument('--range-kernel', choices=('fine', 'balanced'), default='fine',
                   help='fine uses MQ2 and finer decode splits; balanced uses MQ4 and coarser decode splits')
    p.add_argument('--n-predict', type=int, default=256)
    p.add_argument('--threads', type=int, required=True)
    p.add_argument('--threads-batch', type=int)
    p.add_argument('--cache-ram', type=int, default=4096, help='MiB; only this fresh child\'s RAM cache')
    p.add_argument('--device', default='CUDA0')
    p.add_argument('--gpu-layers', type=int, default=99)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--repeat-prompt', action='store_true')
    p.add_argument('--library-dir', action='append', default=[])
    p.add_argument('--profile', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--trace', action='store_true')
    p.add_argument('--startup-timeout', type=float, default=600)
    p.add_argument('--request-timeout', type=float, default=3600)
    p.add_argument('--prefix-snapshot', help='frozen local-prefix slot file; reuse if present, otherwise create before overflow')
    p.add_argument('--frozen-prefix-batch', type=int, choices=(512,),
                   help='explicitly reuse an existing B512 local prefix as a frozen input for B1024 continuation; no cold-prefix claim')
    p.add_argument('--prefix-only', action='store_true', help='create/restore bounded local prefix only; no throughput qualification')
    p.add_argument('--execute', action='store_true', help='run the explicitly configured GPU/RPC benchmark')
    p.add_argument('--dry-run', action='store_true', help='preview only (the default); no child process')
    return p


def validate(args):
    try:
        args.points = [int(n) for n in args.points.split(',')]
    except ValueError as error:
        raise ValueError('points must be comma-separated integers') from error
    if not args.points or args.points != sorted(set(args.points)):
        raise ValueError('points must be strictly ascending and unique')
    if args.p <= 0 or args.p % 256 or args.context <= args.p or args.context % 256:
        raise ValueError('P and context must be positive multiples of 256 with P < context')
    if args.batch <= 0 or args.ubatch <= 0 or args.ubatch > args.batch or args.batch % args.ubatch:
        raise ValueError('batch must be a positive multiple of ubatch')
    if args.n_predict < 2 or min(args.points) < args.p or max(args.points) + args.n_predict >= args.context:
        raise ValueError('points must be >= P and leave n-predict >= 2 output slots within context')
    if args.threads <= 0 or (args.threads_batch is not None and args.threads_batch <= 0):
        raise ValueError('thread counts must be positive')
    if args.cache_ram < 0 or min(args.startup_timeout, args.request_timeout) <= 0:
        raise ValueError('cache RAM and timeouts must be bounded nonnegative/positive values')
    host, separator, port = args.xbox_host.rpartition(':')
    if not separator or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError('xbox-host must be explicit HOST:PORT')
    if args.wavefront_tile == 256 and (args.batch not in (512, 1024) or args.ubatch != 256 * args.wavefront_tiles):
        raise ValueError('--wavefront-tile 256 requires UB512/two tiles or B1024/UB1024/four tiles')
    if args.wavefront_tiles != 2 and args.wavefront_tile != 256:
        raise ValueError('four tiles are supported only with --wavefront-tile 256')
    if args.tcp_prefetch and (not args.wavefront_tile or not args.profile):
        raise ValueError('--tcp-prefetch requires wavefront and profiling counters to prove overlap')
    if args.wavefront_tile == 128 and (args.batch != 512 or args.ubatch != 256):
        raise ValueError('--wavefront-tile 128 requires --batch-size 512 --ubatch 256 (two compute tiles of 128 per physical parent)')
    if args.wavefront_tile == 0 and args.ubatch != 256:
        raise ValueError('OFF control requires physical --ubatch 256; parent 512 is reserved for --wavefront-tile 256')
    if args.n256_lookahead and (args.batch != 512 or args.ubatch != 256 or args.wavefront_tile):
        raise ValueError('--n256-lookahead is a separate prototype: requires batch-size 512, ubatch 256, wavefront-tile 0')
    if args.execute and args.dry_run:
        raise ValueError('choose --execute or --dry-run')
    for name in ('server_binary', 'model', 'prompt_tokens'):
        value = Path(getattr(args, name)).resolve(strict=True)
        if not value.is_file():
            raise ValueError(f'{name} is not a regular file')
        setattr(args, name, str(value))
    if args.prefix_only and not args.prefix_snapshot:
        raise ValueError('--prefix-only requires --prefix-snapshot')
    if args.prefix_snapshot:
        if Path(args.prefix_snapshot).is_symlink():
            raise ValueError('prefix snapshot main file must not be a symlink')
        args.prefix_snapshot = str(Path(args.prefix_snapshot).resolve())
        if args.frozen_prefix_batch and (args.batch != 1024 or not Path(args.prefix_snapshot).is_file()):
            raise ValueError('cross-batch frozen reuse requires B1024 and an existing B512 prefix snapshot')
        if Path(args.prefix_snapshot).name in ('', '.', '..'):
            raise ValueError('prefix snapshot must name a file')
    memory_policy()
    args.library_dir = [str(Path(p).resolve(strict=True)) for p in args.library_dir]
    if any(not Path(p).is_dir() for p in args.library_dir):
        raise ValueError('library-dir must be a directory')
    if args.frozen_prefix_batch and not args.prefix_snapshot:
        raise ValueError('--frozen-prefix-batch requires --prefix-snapshot')
    return args


def server_command(args, port):
    command = [args.server_binary, '-m', args.model, '--host', '127.0.0.1', '--port', str(port),
            '--ctx-size', str(args.context), '--batch-size', str(args.batch), '--ubatch-size', str(args.ubatch),
            '--threads', str(args.threads), '--threads-batch', str(args.threads_batch or args.threads),
            '--parallel', '1', '--device', args.device, '--gpu-layers', str(args.gpu_layers),
            '--flash-attn', 'on', '--fit', 'off', '--cache-type-k', 'kvarn4', '--cache-type-v', 'kvarn4',
            '--remote-attn', args.xbox_host, '--kv-position-split-p', str(args.p),
            '--remote-attn-layers', 'full', '--remote-attn-cache-type-k', 'q4_0', '--remote-attn-cache-type-v', 'q4_0',
            '--remote-attn-format', 'kvarn4', '--remote-attn-wire-format', 'fp8', '--remote-attn-stats',
            '--spec-type', 'none', '--spec-draft-n-max', '0', '--cache-ram', str(args.cache_ram),
            '--ctx-checkpoints', '1',
            '--no-context-shift', '--no-webui', '--metrics', '--verbosity', '4']
    if args.prefix_snapshot:
        command += ['--slot-save-path', str(Path(args.prefix_snapshot).parent) + os.sep]
    return command


def parse_timings(lines):
    records = []
    for line in lines:
        match = TIMING.search(line)
        if match:
            records.append({'kind': 'prompt' if match[1] else 'decode', 'ms': float(match[2]),
                            'tokens': int(match[3]), 'ms_per_token_reported': float(match[4]),
                            'tps_reported': float(match[5]), 'line': line.rstrip()})
    return records


def n256_activation(lines):
    # Core marker is emitted only after eligible preflight + both applies:
    # N256 lookahead: activated backend=CPU ubatches=2 ubatch_tokens=256 layers=32
    # The CPU prototype is real, but cannot qualify a CUDA/RKVA candidate.
    pattern = re.compile(r'N256 lookahead: activated backend=(\S+) ubatches=(\d+) ubatch_tokens=(\d+) layers=(\d+)')
    records = []
    fallbacks = []
    for line in lines:
        match = pattern.search(line)
        if match:
            records.append({'backend': match[1], 'ubatches': int(match[2]), 'ubatch_tokens': int(match[3]),
                            'layers': int(match[4]), 'line': line.rstrip()})
        if 'N256 lookahead: fallback backend=serial' in line:
            fallbacks.append(line.rstrip())
    cuda = [r for r in records if r['backend'] == 'CUDA' and r['ubatches'] == 2 and
            r['ubatch_tokens'] == 256 and r['layers'] > 0]
    return {'cuda_2x256_confirmed': bool(cuda), 'activations': records,
            'serial_fallback_count': len(fallbacks), 'serial_fallback_examples': fallbacks[:3]}


def wavefront_activation(lines, tile=256, tiles=2):
    records = []
    for line in lines:
        if 'RKVA wavefront graph:' not in line:
            continue
        fields = {key: int(value) for key, value in re.findall(r'\b(tile|tiles|parent_tokens|final_ffn_rows|max_outstanding)=(\d+)\b', line)}
        fields['line'] = line.rstrip()
        records.append(fields)
    def expected(record):
        return all(record.get(key) == value for key, value in
                   [('tile', tile), ('tiles', tiles), ('parent_tokens', tiles * tile), ('final_ffn_rows', 256)]) and (tiles != 4 or record.get('max_outstanding') == 2)
    valid = [record for record in records if expected(record)]
    invalid = [record for record in records if not expected(record)]
    return {'rkva_parent512_tile256_confirmed': tile == 256 and tiles == 2 and bool(valid) and not invalid,
            'rkva_parent1024_tile256_confirmed': tile == 256 and tiles == 4 and bool(valid) and not invalid,
            'requested_geometry_confirmed': bool(valid) and not invalid,
            'requested_tile': tile, 'requested_parent_tokens': tiles * tile,
            'activations': records, 'incompatible_activations': invalid}


def tcp_prefetch_activation(lines):
    records = []
    for line in lines:
        fields = {key: int(value) for key, value in re.findall(
            r'"tcp_prefetch_(admitted|sent|completed|ahead_sent|max_frames)"\s*:\s*(\d+)', line)}
        if fields:
            records.append(fields)
    final = records[-1] if records else {}
    confirmed = bool(records) and all(record.get('max_frames', 0) <= 2 for record in records) and (
        final.get('admitted', 0) == final.get('sent', 0) == final.get('completed', 0) > 0 and
        final.get('max_frames') == 2 and 0 < final.get('ahead_sent', 0) <= final['sent'])
    return {'two_frame_overlap_confirmed': confirmed, 'records': records}


def local_input_reuse_activation(lines):
    records = []
    malformed = False
    for line in lines:
        if 'ggml_input_cache: ' not in line:
            continue
        try:
            record = json.loads(line.split('ggml_input_cache: ', 1)[1])
        except (ValueError, TypeError):
            malformed = True
            continue
        if isinstance(record, dict) and all(type(record.get(k)) is int and record[k] >= 0
               for k in ('hits', 'misses', 'copied_bytes', 'skipped_bytes')):
            records.append(record)
        else:
            malformed = True
    final = records[-1] if records else {}
    return {'actual_reuse_confirmed': not malformed and all(final.get(k, 0) > 0
            for k in ('hits', 'misses', 'copied_bytes', 'skipped_bytes')), 'records': records,
            'malformed_records': malformed}


def wavefront_campaign_qualified(points):
    return any(point.get('valid') and point.get('wavefront_tile_requested') in (128, 256) and
               (point.get('uncached_overflow_tokens') or 0) >= point['wavefront_activation']['requested_parent_tokens'] and
               point['wavefront_activation']['requested_geometry_confirmed'] for point in points)


def prefix_geometry(lines, parent=512):
    records = []
    for line in lines:
        if 'RKVA wavefront prefix:' not in line:
            continue
        fields = {key: int(value) for key, value in re.findall(r'\b(requested_ubatch|effective_ubatch)=(\d+)\b', line)}
        fields['line'] = line.rstrip()
        records.append(fields)
    return {'prefix_compute_n256_confirmed': any(r.get('requested_ubatch') == parent and
            r.get('effective_ubatch') == 256 for r in records), 'records': records}


def assess(response, lines, expected, prompt_length=None, n256_lookahead=False, prior_n256=None,
           wavefront_tile=0, split_p=None, wavefront_tiles=2):
    errors = []
    timing = response.get('timings', {})
    predicted = timing.get('predicted_n')
    if type(predicted) is not int or predicted != expected:
        errors.append(f'predicted_n={predicted}, expected {expected}')
    if response.get('error') or response.get('truncated'):
        errors.append('response error or truncated prompt')
    if response.get('tokens_predicted', predicted) != predicted:
        errors.append('tokens_predicted disagrees with timings.predicted_n')
    if prompt_length is not None:
        cached, processed = timing.get('cache_n'), timing.get('prompt_n')
        if type(cached) is not int or type(processed) is not int or cached <= 0 or cached + processed != prompt_length:
            errors.append('warm prefix reuse or prompt/cache token accounting was not confirmed')
    if timing.get('draft_n', 0):
        errors.append('speculative tokens present despite MTP off')
    records = parse_timings(lines)
    decode = [r for r in records if r['kind'] == 'decode']
    prompt = [r for r in records if r['kind'] == 'prompt']
    log = decode[-1] if decode else None
    api_ms = timing.get('predicted_ms')
    api_tps = timing.get('predicted_per_second')
    if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (api_ms, api_tps)):
        errors.append('missing/nonpositive/nonfinite response decode duration or rate')
    else:
        # Current server: the first token is emitted from prefill; use actual count.
        if type(predicted) is int and not math.isclose(api_tps, max(0, predicted - 1) * 1000 / api_ms, rel_tol=.01, abs_tol=.02):
            errors.append('response TPS disagrees with current predicted_n-1 decode-step semantics')
        if log and (log['tokens'] != predicted or not math.isclose(log['ms'], api_ms, rel_tol=.01, abs_tol=1)
                    or not math.isclose(log['tps_reported'], api_tps, rel_tol=.01, abs_tol=.02)):
            errors.append('server eval-time and response timings disagree')
    if not log:
        errors.append('missing authoritative server eval-time record')
    flags, warnings = classify_logs(lines)
    if flags:
        errors.append('server logged error/fallback; inspect server.log')
    activation = n256_activation(lines)
    if n256_lookahead:
        # N1 decode and a cached prompt shorter than 512 cannot use a 2x256
        # prefill. They may inherit the warm activation proof; a fresh >=512
        # prefill must prove activation within its own request window.
        proof = activation if timing.get('prompt_n', 0) >= 512 else prior_n256 or activation
        if not proof.get('cuda_2x256_confirmed'):
            errors.append('N256 requested but CUDA 2x256 runtime activation was not confirmed (CPU/serial/ignored is not a candidate)')
        if any(r['backend'] != 'CUDA' for r in activation['activations']):
            errors.append('N256 request activated a non-CUDA backend')
    wf_activation = wavefront_activation(lines, wavefront_tile or 256, wavefront_tiles)
    overflow_processed = None
    if wavefront_tile:
        cached = timing.get('cache_n')
        if prompt_length is None or split_p is None or type(cached) is not int:
            errors.append(f'WF{wavefront_tile} overflow request geometry is missing')
        else:
            overflow_processed = max(0, prompt_length - max(split_p, cached))
            if overflow_processed >= wavefront_tiles * wavefront_tile and not wf_activation['requested_geometry_confirmed']:
                errors.append(f'overflow prefill did not prove RKVA parent{wavefront_tiles * wavefront_tile} / {wavefront_tiles} tiles{wavefront_tile} / final FFN256 activation')
            if wf_activation['incompatible_activations']:
                errors.append('wavefront activation used incompatible parent/tile/FFN geometry for the requested candidate')
    f5 = response.get('f5_per_token')
    f5_tps = None
    if f5 is not None:
        intervals = f5.get('token_ms', [])
        if (f5.get('truncated') or f5.get('speculative_seen') or f5.get('tokens_recorded') != predicted
                or type(predicted) is not int or len(intervals) != predicted - 1
                or not all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in intervals)):
            errors.append('incomplete/truncated/speculative F5 intervals')
        elif intervals:
            f5_tps = len(intervals) * 1000 / sum(intervals)
    return {'valid': not errors, 'errors': errors, 'server_flags': flags, 'memory_policy_warnings': warnings,
            'response_timings': timing, 'server_decode': log, 'server_prompt': prompt[-1] if prompt else None,
            'f5_interval_tps': f5_tps, 'f5_present': f5 is not None,
            'n256_requested': n256_lookahead, 'n256_activation': activation,
            'wavefront_tile_requested': wavefront_tile, 'wavefront_activation': wf_activation,
            'uncached_overflow_tokens': overflow_processed,
            'authoritative_decode_tps': log['tps_reported'] if log and not errors else None}


class LogDrain:
    def __init__(self, process, path, port):
        self.process = process
        self.path = path
        self.lines = []
        self.condition = threading.Condition()
        self.ready = threading.Event()
        self.listening = False
        self.eof = False
        self.url = re.compile(r'listening on http://127\.0\.0\.1:' + str(port) + r'(?:/|\s|$)')
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            with self.path.open('w', buffering=1) as log:
                for line in self.process.stdout:
                    log.write(line)
                    with self.condition:
                        self.lines.append(line)
                        if self.url.search(line):
                            self.listening = True
                            self.ready.set()
                            print(f'{now()} own server listening; starting health check', flush=True)
                        if 'prompt processing' in line or 'gpu_profile' in line:
                            # Progress is emitted by the server, never by a polling loop.
                            if 'prompt processing' in line:
                                print(line.strip(), flush=True)
                        self.condition.notify_all()
        finally:
            with self.condition:
                self.eof = True
                self.ready.set()
                self.condition.notify_all()

    def index(self):
        with self.condition:
            return len(self.lines)

    def request_lines(self, start, timeout=30):
        with self.condition:
            self.condition.wait_for(lambda: self.eof or any(r['kind'] == 'decode' for r in parse_timings(self.lines[start:])), timeout)
            return self.lines[start:].copy()


def http_json(port, endpoint, body=None, timeout=30):
    request = urllib.request.Request(f'http://127.0.0.1:{port}{endpoint}',
                                    data=json.dumps(body).encode() if body is not None else None,
                                    headers={'Content-Type': 'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


def git_identity():
    def git(*arguments):
        result = subprocess.run(['git', '-C', str(ROOT), *arguments], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        return result.stdout if result.returncode == 0 else None
    head = git('rev-parse', 'HEAD')
    if not head:
        raise RuntimeError('cannot freeze repository source identity')
    diff = git('diff', 'HEAD', '--binary')
    status = git('status', '--porcelain=v1')
    if diff is None or status is None:
        raise RuntimeError('cannot freeze repository diff/status identity')
    refs = {}
    for ref in ('origin/main', 'upstream/main', 'beellama-official/v0.4.9'):
        value = git('rev-parse', '--verify', ref)
        refs[ref] = value.decode().strip() if value else None
    return {'root': str(ROOT), 'head': head.decode().strip(),
            'branch': (git('branch', '--show-current') or b'').decode().strip(),
            'tracked_diff_sha256': hashlib.sha256(diff).hexdigest(), 'refs': refs,
            'status': status.decode().splitlines(),
            'binding': 'observed checkout identity; actual executable version is recorded separately',
            'timing_source': fingerprint(ROOT / 'tools/server/server-common.h')}


def local_libraries(args):
    paths = set()
    for folder in [Path(args.server_binary).parent, *map(Path, args.library_dir)]:
        for pattern in ('libllama*.so*', 'libggml*.so*', '*llama*.dll', '*ggml*.dll'):
            paths.update(p.resolve() for p in folder.glob(pattern) if p.is_file())
    return [fingerprint(p) for p in sorted(paths)]


def loaded_libraries(pid):
    maps = Path(f'/proc/{pid}/maps')
    if not maps.exists():
        raise RuntimeError('loaded-library provenance currently requires Linux /proc/PID/maps')
    paths = set()
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/') and '.so' in fields[5]:
            path = Path(fields[5])
            if not path.is_file():
                raise RuntimeError(f'loaded library no longer exists: {path}')
            paths.add(path.resolve())
    return [fingerprint(p) for p in sorted(paths)]


def pinned_environment(args):
    values = dict(PINNED_ENV)
    values['LLAMA_N256_LOOKAHEAD'] = '1' if args.n256_lookahead else '0'
    values['GGML_RKVA_WAVEFRONT_TILE'] = str(args.wavefront_tile)
    values['GGML_RKVA_WAVEFRONT_RPC_BATCH'] = '1'
    values['GGML_RKVA_CUDA_FP8'] = args.cuda_fp8
    values['GGML_RKVA_TCP_PREFETCH'] = '1' if args.tcp_prefetch else '0'
    values['GGML_RKVA_LOCAL_INPUT_REUSE'] = '1' if args.local_input_reuse else '0'
    values.pop('GGML_CUDA_MASK_PACK', None)
    if args.mask_pack != 'default':
        values['GGML_CUDA_MASK_PACK'] = '0' if args.mask_pack == 'off' else '1'
    values['GGML_RKVA_RANGE_KERNEL'] = args.range_kernel
    return values


def n256_capability(args):
    return {'requested': args.n256_lookahead, 'logical_batch': args.batch, 'physical_ubatch': args.ubatch,
            'harness_execution_enabled': True, 'runtime_activation_confirmed': False, 'production_qualified': False,
            'status': 'separate CPU prototype request; no import/support is inferred; qualification requires explicit CUDA 2x256 runtime activation',
            'required_pair': 'logical batch 512 OFF versus logical batch 512 ON; physical ubatch 256 in both'}


def wavefront_campaign(args):
    return {'requested_tile': args.wavefront_tile, 'requested_logical_batch': args.batch,
            'requested_tiles': args.wavefront_tiles,
            'requested_parent_ubatch': args.ubatch, 'expected_prefix_compute_ubatch': 256,
            'expected_overflow_parent_tokens': args.wavefront_tiles * args.wavefront_tile if args.wavefront_tile else None,
            'expected_compute_tile_tokens': args.wavefront_tile or 256, 'rpc_batch': 1,
            'cpu_n256_prototype': False if args.wavefront_tile else args.n256_lookahead,
            'runtime_activation_confirmed': False, 'production_qualified': False,
            'pair': 'OFF B512/UB256; WF128 B512/UB256; WF256 B512/UB512 or common B1024 UB512/two versus UB1024/four; final FFN N256, max outstanding2',
            'historical_wf128': 'prior holdout rejection remains unchanged; fresh WF128 measurements require their own quality checks'}


def environment(args, output):
    # Drop inherited model/backend overrides, not just the few historical flags.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('LLAMA_', 'GGML_', 'XLLAMA_', 'CUDA_')) or k == 'CUDA_VISIBLE_DEVICES'}
    memory_policy()  # refuse rather than erase a stricter inherited graph guard
    env.update(pinned_environment(args))
    env['OMP_NUM_THREADS'] = str(args.threads)
    env['OMP_DYNAMIC'] = 'FALSE'
    env['GGML_RKVA_PROFILE'] = '1' if args.profile else '0'
    env['LD_LIBRARY_PATH'] = os.pathsep.join([str(Path(args.server_binary).parent), *args.library_dir])
    for key, suffix in [('TMPDIR', 'tmp'), ('CUDA_CACHE_PATH', 'cuda-cache'), ('XDG_CACHE_HOME', 'cache')]:
        folder = output / suffix
        folder.mkdir()
        env[key] = str(folder)
    if args.trace:
        env['GGML_RKVA_TRACE_FILE'] = str(output / 'rkva-trace')
    frozen = {k: v for k, v in env.items() if k in PINNED_ENV or k in ('GGML_RKVA_PROFILE', 'GGML_RKVA_TRACE_FILE',
              'LD_LIBRARY_PATH', 'LD_PRELOAD', 'TMPDIR', 'CUDA_CACHE_PATH', 'XDG_CACHE_HOME', 'CUDA_VISIBLE_DEVICES',
              'OMP_NUM_THREADS', 'OMP_DYNAMIC')}
    return env, frozen


def request_body(tokens, count, seed, warm=False):
    return {'prompt': tokens, 'n_predict': count, 'temperature': 0 if warm else 1.0,
            'seed': seed, 'top_p': .95, 'top_k': 20, 'min_p': 0.0, 'repeat_penalty': 1.0,
            'presence_penalty': 0.0, 'ignore_eos': True, 'cache_prompt': True,
            'id_slot': 0, 'stream': False, 'return_tokens': True}



# Snapshot files are experimental frozen inputs, never an end-to-end prefix
# performance claim. Variant flags and build IDs are audit data, not codec keys.
def prefix_contract(args, tokens, model):
    prefix_env = pinned_environment(args)
    # A local prefix ends before P and contains no remote attention queries.
    # These two modes differ only for remote nq<=9, so the existing frozen
    # local-prefix codec contract is identical. Actual mode is audited in the
    # run environment; reuse never claims a new cold-prefix qualification.
    prefix_env['GGML_RKVA_CUDA_FP8'] = 'prefill'
    prefix_env['GGML_RKVA_RANGE_KERNEL'] = 'fine'  # no remote kernel executes inside the local prefix
    return {'schema': 1, 'codec': 'llama-state-seq-file', 'model_sha256': model['sha256'],
            'requested_prefix_sha256': token_hash(tokens[:args.p]), 'P': args.p, 'context': args.context,
            'slot': 0, 'batch': args.frozen_prefix_batch or args.batch, 'physical_prefix_ubatch': 256, 'flash_attn': 'on', 'n_rs_seq': 0,
            'native_kvarn_bits': 4, 'tail': 128, 'window': 4096,
            'environment': {k: v for k, v in prefix_env.items()
                            if k not in ('GGML_RKVA_WAVEFRONT_TILE', 'LLAMA_N256_LOOKAHEAD', 'GGML_RKVA_TCP_PREFETCH',
                                         'GGML_RKVA_LOCAL_INPUT_REUSE', 'GGML_CUDA_MASK_PACK')}}


def snapshot_files(path):
    path = Path(path)
    allowed = {path.name, path.name + '.ckpt', path.name + '.ckpts', path.name + '.logits', path.name + '.meta.json'}
    found = list(path.parent.glob(path.name + '.*'))
    if any(item.name not in allowed or not item.is_file() or item.is_symlink() for item in found):
        raise ValueError('unexpected prefix snapshot sidecar')
    files = [path] + [item for item in found if item.name != path.name + '.meta.json']
    if any(not item.is_file() or item.is_symlink() for item in files):
        raise ValueError('prefix snapshot main file missing or unsafe')
    return sorted(files)


def load_prefix_snapshot(args, tokens, model):
    path = Path(args.prefix_snapshot)
    meta_path = Path(str(path) + '.meta.json')
    if path.is_symlink():
        raise ValueError('prefix snapshot main file must not be a symlink')
    # A partial prior save must never silently become a fresh cache.
    if not path.exists():
        if list(path.parent.glob(path.name + '.*')):
            raise ValueError('orphan prefix snapshot metadata/sidecars')
        return None
    if not meta_path.is_file() or meta_path.is_symlink():
        raise ValueError('prefix snapshot metadata missing or unsafe')
    metadata = json.loads(meta_path.read_text())
    if metadata.get('contract') != prefix_contract(args, tokens, model):
        raise ValueError('prefix snapshot semantic/model/token contract mismatch')
    n = metadata.get('actual_prefix_tokens')
    if type(n) is not int or n not in (args.p, args.p - 128) or n <= 0:
        raise ValueError('prefix snapshot actual prefix length is invalid')
    if metadata.get('actual_prefix_sha256') != token_hash(tokens[:n]):
        raise ValueError('prefix snapshot saved token hash mismatch')
    actual = {item.name: {'size': item.stat().st_size, 'sha256': sha256(item)} for item in snapshot_files(path)}
    if actual != metadata.get('files') or metadata.get('save', {}).get('n_saved') != n:
        raise ValueError('prefix snapshot missing/corrupted/stale files or save metadata')
    if metadata['save'].get('n_written') != actual[path.name]['size']:
        raise ValueError('prefix snapshot byte count does not match saved main file')
    return metadata


def slot_operation(port, action, args):
    response = http_json(port, '/slots/0?action=' + action,
                         {'filename': Path(args.prefix_snapshot).name}, args.request_timeout)
    count_key, byte_key, timing_key = ('n_saved', 'n_written', 'save_ms') if action == 'save' else ('n_restored', 'n_read', 'restore_ms')
    if (response.get('error') or type(response.get(count_key)) is not int or
            response[count_key] not in (args.p, args.p - 128) or response[count_key] <= 0 or
            type(response.get(byte_key)) is not int or response[byte_key] <= 0 or
            not isinstance(response.get('timings', {}).get(timing_key), (float, int)) or
            not math.isfinite(response['timings'][timing_key]) or response['timings'][timing_key] < 0 or
            response.get('id_slot') != 0 or response.get('filename') != Path(args.prefix_snapshot).name):
        raise RuntimeError('invalid slot ' + action + ' response')
    return response


def save_prefix_snapshot(args, tokens, manifest, response):
    path = Path(args.prefix_snapshot)
    n = response['n_saved']
    files = {item.name: {'size': item.stat().st_size, 'sha256': sha256(item)} for item in snapshot_files(path)}
    if files[path.name]['size'] != response['n_written']:
        raise RuntimeError('saved prefix main file byte count differs from slot response')
    with path.open('rb') as state_file:
        header_hex = state_file.read(8).hex()
    metadata = {'contract': prefix_contract(args, tokens, manifest['model']), 'actual_prefix_tokens': n,
                'main_header_first8_hex_audit': header_hex,
                'actual_prefix_sha256': token_hash(tokens[:n]), 'files': files, 'save': response,
                'origin': {key: manifest[key] for key in ('server', 'source', 'local_libraries', 'loaded_libraries')},
                'purpose': 'frozen experimental prefix input; not new-prefix end-to-end qualification'}
    write_json(Path(str(path) + '.meta.json'), metadata)
    return metadata


def validate_bounded_warm(response, p):
    timings = response.get('timings', {})
    cache, prompt = timings.get('cache_n'), timings.get('prompt_n')
    if (response.get('error') or response.get('truncated') or timings.get('predicted_n') != 1 or
            type(cache) is not int or type(prompt) is not int or not p - 128 <= cache <= p or
            not 0 <= prompt <= 128 or cache + prompt != p):
        raise RuntimeError('restored prefix warm was not a bounded <=128 token replay to P')
    return {'cache_n': cache, 'prompt_n': prompt, 'timings': timings}


def run(args, tokens, stream_info):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)  # never reuse old cached state or results
    process = drain = None
    manifest = {'schema': 1, 'started': now(), 'configuration': vars(args), 'token_stream': stream_info,
                'host': {'hostname': socket.gethostname(), 'platform': platform.platform()},
                'decode_timing_semantics': 'predicted_n - 1 timed steps; first token from prefill',
                'comparison': 'fresh baseline only; no historical TPS equivalence is asserted',
                'memory_policy': memory_policy(), 'n256_capability': n256_capability(args),
                'wavefront_campaign': wavefront_campaign(args),
                'local_input_reuse_scope': {
                    'requested': args.local_input_reuse,
                    'indices': 'local I64 map; byte-exact final host contents',
                    'masks': 'serial/two-tile inputs; N1 at all geometries; four-tile prefill retains ordinary copies',
                    'four_tile_prefill_masks_cacheable': False}}
    result = {'status': 'failed', 'points': []}
    try:
        print(f'{now()} freezing model/binary/token/source hashes before starting any server', flush=True)
        manifest['model'] = fingerprint(args.model)
        manifest['server'] = fingerprint(args.server_binary)
        manifest['prompt_file'] = fingerprint(args.prompt_tokens)
        manifest['source'] = git_identity()
        manifest['runner'] = fingerprint(__file__)
        manifest['local_libraries'] = local_libraries(args)
        write_json(output / 'prompt-stream.json', tokens)
        snapshot = None
        if args.prefix_snapshot:
            snapshot = load_prefix_snapshot(args, tokens, manifest['model'])
            Path(args.prefix_snapshot).parent.mkdir(parents=True, exist_ok=True)
            manifest['prefix_cache'] = {'intent': 'restore' if snapshot else 'create', 'purpose': 'frozen prefix experimental input',
                                        'metadata': snapshot, 'source_batch': args.frozen_prefix_batch or args.batch,
                                        'continuation_batch': args.batch,
                                        'explicit_cross_batch_reuse': args.frozen_prefix_batch is not None}
        env, manifest['environment'] = environment(args, output)
        version = subprocess.run([args.server_binary, '--version'], env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', timeout=30)
        if version.returncode != 0:
            raise RuntimeError('server --version failed before benchmark launch')
        manifest['binary_version_output'] = version.stdout
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        manifest['port'] = port
        manifest['command'] = server_command(args, port)
        write_json(output / 'manifest.json', manifest)
        print(f'{now()} launching private child on 127.0.0.1:{port}', flush=True)
        process = subprocess.Popen(manifest['command'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1, start_new_session=True)
        manifest['pid'] = process.pid
        drain = LogDrain(process, output / 'server.log', port)
        if not drain.ready.wait(args.startup_timeout) or not drain.listening:
            raise RuntimeError('own server did not emit the matching listening URL before timeout/exit')
        manifest['health'] = http_json(port, '/health')
        manifest['props'] = http_json(port, '/props')
        manifest['loaded_libraries'] = loaded_libraries(process.pid)
        manifest['startup_memory_policy_warnings'] = validate_startup(args, drain.lines)
        manifest['negotiation'] = [line.strip() for line in drain.lines if re.search(r'connected to|native KVarN4|position-split|wire=|descriptor=', line)]
        write_json(output / 'manifest.json', manifest)
        if snapshot:
            print(f'{now()} restoring frozen prefix: actual_tokens={snapshot["actual_prefix_tokens"]}, bounded replay <=128', flush=True)
            restored = slot_operation(port, 'restore', args)
            if restored['n_restored'] != snapshot['actual_prefix_tokens'] or restored['n_read'] != snapshot['save']['n_written']:
                raise RuntimeError('slot restore token/byte count differs from frozen snapshot')
            manifest['prefix_cache']['restore'] = restored
        requests = [('warm', args.p, 1, True)]
        if not args.prefix_only:
            requests += [(f'point-{n}', n, args.n_predict, False) for n in args.points]
        for label, length, count, warm in requests:
            body = request_body(tokens[:length], count, args.seed, warm)
            write_json(output / f'{label}.request.json', body)
            start = drain.index()
            print(f'{now()} {label}: prompt={length}, requested outputs={count}', flush=True)
            began = time.monotonic()
            response = http_json(port, '/completion', body, args.request_timeout)
            write_json(output / f'{label}.response.json', response)
            lines = drain.request_lines(start)
            if warm:
                if response.get('error') or response.get('truncated') or response.get('timings', {}).get('predicted_n') != 1:
                    raise RuntimeError('warm prefix did not finish exactly one output token')
                if snapshot:
                    manifest['prefix_cache']['bounded_replay'] = validate_bounded_warm(response, args.p)
                    manifest['prefix_cache']['bounded_replay']['wall_s'] = time.monotonic() - began
                warm_errors, warm_warnings = classify_logs(lines)
                if warm_errors:
                    raise RuntimeError('warm prefix logged an error/fallback')
                if args.wavefront_tile == 256:
                    warm_geometry = prefix_geometry(lines, args.ubatch)
                    manifest['warm_prefix_geometry'] = warm_geometry
                    if not warm_geometry['prefix_compute_n256_confirmed']:
                        if snapshot and response.get('timings', {}).get('prompt_n') == 0:
                            warm_geometry['no_compute_frozen_prefix_cache_hit'] = True
                        else:
                            raise RuntimeError(f'WF256 warm local prefix did not confirm compute UB256 with requested parent UB{args.ubatch}')
                warm_activation = n256_activation(lines)
                manifest['n256_warm_activation'] = warm_activation
                if not args.prefix_only and args.n256_lookahead and length >= 512 and not warm_activation['cuda_2x256_confirmed']:
                    raise RuntimeError('N256 warm prefill did not prove CUDA 2x256 activation; CPU/serial/ignored does not qualify')
                manifest['n256_capability']['runtime_activation_confirmed'] = warm_activation['cuda_2x256_confirmed']
                manifest['warm'] = {'prompt_tokens': length, 'timings': response.get('timings'), 'wall_s': time.monotonic() - began,
                                    'memory_policy_warnings': warm_warnings}
                if args.prefix_snapshot and snapshot is None:
                    print(f'{now()} saving local prefix before any overflow', flush=True)
                    saved = slot_operation(port, 'save', args)
                    snapshot = save_prefix_snapshot(args, tokens, manifest, saved)
                    manifest['prefix_cache']['metadata'] = snapshot
                    manifest['prefix_cache']['save'] = saved
                    print(f'{now()} frozen prefix saved: actual_tokens={saved["n_saved"]}, bytes={saved["n_written"]}', flush=True)
                    if saved['n_saved'] != args.p and not args.prefix_only:
                        repair_start = drain.index()
                        repair_began = time.monotonic()
                        repair = http_json(port, '/completion', request_body(tokens[:args.p], 1, args.seed, True), args.request_timeout)
                        write_json(output / 'prefix-repair.response.json', repair)
                        manifest['prefix_cache']['bounded_replay'] = validate_bounded_warm(repair, args.p)
                        manifest['prefix_cache']['bounded_replay']['wall_s'] = time.monotonic() - repair_began
                        if classify_logs(drain.request_lines(repair_start))[0]:
                            raise RuntimeError('prefix repair logged an error/fallback')
            else:
                point = assess(response, lines, count, prompt_length=length, n256_lookahead=args.n256_lookahead,
                               prior_n256=manifest.get('n256_warm_activation'), wavefront_tile=args.wavefront_tile, split_p=args.p,
                               wavefront_tiles=args.wavefront_tiles)
                point.update({'prompt_tokens': length, 'wall_s': time.monotonic() - began,
                              'request_sha256': sha256(output / f'{label}.request.json')})
                result['points'].append(point)
                print(f"{now()} {label}: valid={point['valid']} server_decode_tps={point['authoritative_decode_tps']} cache_n={point['response_timings'].get('cache_n')}", flush=True)
                if not point['valid']:
                    raise RuntimeError(f'{label} failed qualification: {point["errors"]}')
            write_json(output / 'results.json', result)
        manifest['wavefront_run_activation'] = wavefront_activation(drain.lines, args.wavefront_tile or 256, args.wavefront_tiles)
        if not args.prefix_only and args.wavefront_tile and not wavefront_campaign_qualified(result['points']):
            raise RuntimeError(f'WF{args.wavefront_tile} campaign never proved its requested parent/tile activation in a measured request; no candidate throughput is qualified')
        manifest['wavefront_campaign']['runtime_activation_confirmed'] = wavefront_campaign_qualified(result['points'])
        manifest['n256_run_activation'] = n256_activation(drain.lines)
        if not args.prefix_only and args.n256_lookahead and not manifest['n256_run_activation']['cuda_2x256_confirmed']:
            raise RuntimeError('N256 ON run never activated CUDA 2x256; throughput is unqualified')
        # Freeze all actually mapped libraries, including any loaded lazily.
        manifest['loaded_libraries_final'] = loaded_libraries(process.pid)
        for item in [manifest['model'], manifest['server'], *manifest['local_libraries']]:
            stat = Path(item['path']).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (item['size'], item['mtime_ns'], item['inode']):
                raise RuntimeError('input changed while benchmark was running: ' + item['path'])
        final_source = git_identity()
        if any(final_source[k] != manifest['source'][k] for k in ('head', 'tracked_diff_sha256')):
            raise RuntimeError('source HEAD or tracked diff changed while benchmark was running')
        flags, result['memory_policy_warnings'] = classify_logs(drain.lines)
        if flags:
            raise RuntimeError('run-level error/fallback detected; throughput remains unqualified')
        result['status'] = 'passed'
        if args.prefix_only:
            result['purpose'] = 'prefix-cache-only; no TPS or wavefront activation qualification'
            manifest['purpose'] = result['purpose']
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        print(f'{now()} FAILED: {result["error"]}', file=sys.stderr, flush=True)
    finally:
        if process is not None:
            exited = process.poll()
            if exited is not None and exited != 0:
                result.update(status='failed', error=f'owned server exited unexpectedly ({exited})')
            if exited is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            result['child_exit_code'] = process.returncode
        if drain is not None:
            drain.thread.join(timeout=5)
            if drain.thread.is_alive():
                result.update(status='failed', error='stdout reader did not finish after owned child exit')
            result['server_errors'], result['memory_policy_warnings'] = classify_logs(drain.lines)
            if result['server_errors']:
                result['status'] = 'failed'
                result.setdefault('error', 'run-level error/fallback detected during teardown')
            result['server_log_sha256'] = sha256(output / 'server.log')
            write_json(output / 'profile-lines.json', [line.rstrip() for line in drain.lines
                       if re.search(r'gpu_profile|RemoteAttn|remote_attn|RKVA|ggml_input_cache', line)])
            manifest['tcp_prefetch_activation'] = tcp_prefetch_activation(drain.lines)
            if args.tcp_prefetch and not args.prefix_only and not manifest['tcp_prefetch_activation']['two_frame_overlap_confirmed']:
                result.update(status='failed', error='TCP prefetch requested but completed two-frame overlap was not proved')
            manifest['local_input_reuse_activation'] = local_input_reuse_activation(drain.lines)
            if args.local_input_reuse and not args.prefix_only and not manifest['local_input_reuse_activation']['actual_reuse_confirmed']:
                result.update(status='failed', error='local input reuse requested but actual skipped copies were not proved')
        if result['status'] != 'passed':
            for point in result['points']:
                point['authoritative_decode_tps'] = None
                point['run_qualified'] = False
        manifest['finished'] = now()
        write_json(output / 'manifest.json', manifest)
        write_json(output / 'results.json', result)
    return 0 if result['status'] == 'passed' else 1


def main(argv=None):
    try:
        args = validate(parser().parse_args(argv))
        tokens, info = effective_tokens(args.prompt_tokens, args.p if args.prefix_only else max([args.p] + args.points), args.repeat_prompt)
        if not args.execute:
            print(json.dumps({'mode': 'dry-run', 'starts_child': False, 'writes_output': False,
                              'configuration': vars(args), 'token_stream': info,
                              'prefix_cache_intent': ('reuse-or-create ' + args.prefix_snapshot) if args.prefix_snapshot else None,
                              'purpose': 'prefix-cache-only; no TPS' if args.prefix_only else 'range benchmark',
                              'command_preview': server_command(args, '<unique-loopback-port>'),
                              'pinned_environment': pinned_environment(args), 'memory_policy': memory_policy(),
                              'n256_capability': n256_capability(args), 'wavefront_campaign': wavefront_campaign(args),
                              'provenance': 'model/binary/libs/git hashes frozen on --execute before launch'}, indent=2))
            return 0
        return run(args, tokens, info)
    except (ValueError, OSError) as error:
        print(f'Configuration error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
