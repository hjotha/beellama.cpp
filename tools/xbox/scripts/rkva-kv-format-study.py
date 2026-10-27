#!/usr/bin/env python3
"""RKVA KV-format study: KVarN4, native Q8_0, and synchronous F16.

Preview is inert and the default.  ``--execute`` is required before any child
process, HTTP request, or output/prefix write.  This wrapper imports the
existing ``bench-rkva-range.py`` helper instead of forking its lifecycle,
snapshot hashing, or qualification checks.  Hardware work is serialized by the
caller: this tool never enables GPU work in preview, never installs/deploys the
Xbox app, and never edits the protected helper.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

SCRIPT_REPO = Path(__file__).resolve().parents[3]
DEFAULT_SERVER = Path('/home/hjotha/beellama.cpp/build-optimized/bin/llama-server')
MAX_CONTEXT = 160000
SCHEMA = 'rkva-kv-format-study-v1'
FORMAT_NAMES = ('kvar4', 'q8_0', 'f16')
ALIASES = {'kvarn4': 'kvar4', 'kvar': 'kvar4', 'q8': 'q8_0', 'fp16': 'f16'}

# These anchors make the in-memory overrides fail closed if the protected
# helper changes underneath this wrapper.
ANCHORS = {
    'global_kvarn4': "'--cache-type-k', 'kvarn4'",
    'remote_q4': "'--remote-attn-cache-type-k', 'q4_0'",
    'upload_kvarn4': "'--remote-attn-format', 'kvarn4'",
    'wire_fp8': "'--remote-attn-wire-format', 'fp8'",
    'producer_cuda': "'GGML_RKVA_NATIVE_PRODUCER': 'cuda'",
    'contract_tail': "'native_kvarn_bits': 4, 'tail': 128, 'window': 4096",
    'startup_kvar4': 'EXPERIMENTAL native KVarN4',
    'startup_tail': 'precision_tail=128',
    'measured_temp': "'temperature': 0 if warm else 1.0",
    'bounded_warm': 'p - 128 <= cache <= p',
}

H = None
ORIG = {}
H_META = {}
_FINGERPRINT_CACHE = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def load_h(repo):
    """Load the protected helper by path and verify the override anchors."""
    global H, H_META
    repo = Path(repo).resolve(strict=True)
    path = repo / 'tools/xbox/scripts/bench-rkva-range.py'
    if not path.is_file():
        raise ValueError(f'bench-rkva-range.py not found under --repo: {path}')
    source = path.read_text(encoding='utf-8')
    missing = [name for name, text in ANCHORS.items() if text not in source]
    if missing:
        raise ValueError('bench-rkva-range.py override anchors changed: ' + ', '.join(missing))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location('bench_rkva_range_study', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.ROOT.resolve() != repo:
        raise ValueError('loaded helper ROOT does not match --repo')
    H = module
    H_META = {'path': str(path), 'sha256': digest, 'root': str(repo),
              'anchors': ANCHORS, 'overrides': [
                  'server_command', 'pinned_environment', 'prefix_contract',
                  'classify_logs', 'validate_startup', 'request_body',
                  'fingerprint']}
    return module


def profiles(q8_profile='match-kvarn4'):
    kvar4 = {
        'name': 'kvar4', 'label': 'KVarN4 native (descriptor 3)',
        'remote_cache_k': 'q4_0', 'remote_cache_v': 'q4_0',
        'upload': 'kvarn4', 'wire': 'fp8', 'batch': 1024, 'ubatch': 512,
        'wavefront': 256, 'wavefront_tiles': 2, 'tcp_prefetch': True,
        'async_overlap': 1, 'native_bulk': '1', 'producer': 'cuda',
        'tail': 128, 'descriptor': 3, 'range_storage': 'kvarn4',
        'cuda_fp8': 'mapped', 'range_kernel': 'fine',
        'changed_vs_kvar4': [],
        'confounders': [
            'KVarN4 structured records (descriptor 3) with precision tail 128',
            'CUDA native KV producer; mapped FP8 Q/OUT activation wire',
            'async local/remote overlap; 2xN256 wavefront; TCP prefetch 2 frames'],
        'roles': {'kv_producer': 'CUDA native KVarN4', 'activation_wire': 'FP8 mapped',
                  'transport': 'native bulk + async + TCP prefetch', 'engine': 'KVarN4',
                  'kernel': 'fine/rowlane (actual audited)', 'descriptor': 3, 'tail': 128},
    }
    if q8_profile == 'serial':
        q8_transport = {'batch': 1024, 'ubatch': 256, 'wavefront': 0, 'wavefront_tiles': 2,
                        'tcp_prefetch': False, 'async_overlap': 0}
        q8_conf = ['explicit serial Q8_0 alternative: B1024/UB256, no wavefront, '
                   'no TCP prefetch, async overlap off']
    else:
        q8_transport = {'batch': 1024, 'ubatch': 512, 'wavefront': 256, 'wavefront_tiles': 2,
                        'tcp_prefetch': True, 'async_overlap': 1}
        q8_conf = ['transport matched to KVarN4: 2xN256 wavefront, async overlap, TCP prefetch']
    q8 = {
        'name': 'q8_0', 'label': 'native Q8_0 (descriptor 5)',
        'remote_cache_k': 'q8_0', 'remote_cache_v': 'q8_0',
        'upload': 'q8_0', 'wire': 'fp8', 'native_bulk': 'fp8', 'producer': 'cpu',
        'tail': 0, 'descriptor': 5, 'range_storage': 'q8_0',
        'cuda_fp8': 'prefill', 'range_kernel': 'fine', 'q8_profile': q8_profile,
        'changed_vs_kvar4': [
            'KV payload is plain Q8_0 rows, not KVarN4 records/tail',
            'KV producer is the host CPU; no CUDA KV producer claim',
            'Q/OUT activation wire stays FP8; no KVarN precision tail'] + q8_conf,
        'confounders': [
            'native Q8_0 rows (descriptor 5); host CPU KV producer',
            'FP8 Q/OUT activation wire; op16/op22 bulk envelope',
            'no KVarN records/stage/tail; KVarN4-only rowlane/INT8 not used'] + q8_conf,
        'roles': {'kv_producer': 'host CPU Q8_0', 'activation_wire': 'FP8',
                  'transport': 'native bulk' + (' + async + TCP prefetch' if q8_transport['tcp_prefetch'] else ' serial'),
                  'engine': 'Q8_0', 'kernel': 'q8_gqa fine (actual audited)',
                  'descriptor': 5, 'tail': 0},
    }
    q8.update(q8_transport)
    f16 = {
        'name': 'f16', 'label': 'synchronous F16 (real original K/V half bytes)',
        'remote_cache_k': 'f16', 'remote_cache_v': 'f16',
        'upload': 'f16', 'wire': 'f16', 'batch': 1024, 'ubatch': 256,
        'wavefront': 0, 'wavefront_tiles': 2, 'tcp_prefetch': False,
        'async_overlap': 0, 'native_bulk': '0', 'producer': 'serial',
        'tail': 0, 'descriptor': None, 'range_storage': 'f16',
        'cuda_fp8': '0', 'range_kernel': 'fine',
        'changed_vs_kvar4': [
            'KV payload is direct original F16 half bytes, not KVarN4 records',
            'no native bulk, no async overlap, no TCP prefetch',
            'F16 Q/OUT activation wire; CUDA FP8 helpers off; serial decode path',
            'logical B1024 with physical prefix UB256; 2xN256 wavefront disabled'],
        'confounders': [
            'F16 storage vs quantized KVarN4/Q8_0; no quantized intermediate',
            'synchronous serial transport; no native bulk/async/TCP prefetch',
            'FP32 math is the supported UWP path; FP16 PSO math is not claimed'] ,
        'roles': {'kv_producer': 'none (direct F16 K/V)', 'activation_wire': 'F16 (CUDA FP8 off)',
                  'transport': 'synchronous range; serial decode', 'engine': 'F16',
                  'kernel': 'f16_gqa fine (actual audited)', 'descriptor': None, 'tail': 0},
    }
    return {'kvar4': kvar4, 'q8_0': q8, 'f16': f16}


def parse_formats(value):
    if isinstance(value, list):
        return value
    result = []
    for item in value.split(','):
        item = ALIASES.get(item.strip(), item.strip())
        if item not in FORMAT_NAMES:
            raise ValueError(f'unknown format {item!r}; expected kvar4,q8_0,f16')
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError('--formats must select at least one format')
    return result


def sequence(formats, order):
    if order == 'ABBA':
        return formats + list(reversed(formats))
    if order == 'interleaved':
        return formats * 2
    return formats


class Patched:
    """Temporarily replace the protected helper's format-specific behaviour."""

    def __init__(self, profile):
        self.profile = profile
        self.saved = {}
        self.cache = _FINGERPRINT_CACHE

    def __enter__(self):
        for name in ('server_command', 'pinned_environment', 'prefix_contract',
                     'classify_logs', 'validate_startup', 'request_body', 'fingerprint'):
            if not hasattr(H, name):
                raise ValueError(f'bench-rkva-range.py is missing {name}')
            self.saved[name] = getattr(H, name)
        original = self.saved
        profile = self.profile
        h = H

        def pinned(args):
            env = original['pinned_environment'](args)
            if profile['name'] == 'kvar4':
                return env
            env = dict(env)
            env['GGML_RKVA_RANGE_STORAGE'] = profile['range_storage']
            env['GGML_RKVA_NATIVE_BULK'] = profile['native_bulk']
            env['GGML_RKVA_ASYNC_LOCAL_REMOTE'] = str(profile['async_overlap'])
            env['GGML_RKVA_TCP_PREFETCH'] = '1' if profile['tcp_prefetch'] else '0'
            env['GGML_RKVA_WIRE_FORMAT'] = profile['wire']
            env['GGML_RKVA_CUDA_FP8'] = profile['cuda_fp8']
            env['GGML_RKVA_NATIVE_TAIL_TOKENS'] = str(profile['tail'])
            env['GGML_RKVA_RANGE_KERNEL'] = profile['range_kernel']
            if profile['producer'] == 'cpu':
                env['GGML_RKVA_NATIVE_PRODUCER'] = 'cpu'
            elif profile['producer'] == 'serial':
                env.pop('GGML_RKVA_NATIVE_PRODUCER', None)
                env.pop('GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL', None)
            return env

        def command(args, port):
            cmd = original['server_command'](args, port)
            for flag, value in (
                    ('--cache-type-k', 'kvarn4'), ('--cache-type-v', 'kvarn4'),
                    ('--remote-attn-cache-type-k', profile['remote_cache_k']),
                    ('--remote-attn-cache-type-v', profile['remote_cache_v']),
                    ('--remote-attn-format', profile['upload']),
                    ('--remote-attn-wire-format', profile['wire'])):
                if flag in cmd:
                    cmd[cmd.index(flag) + 1] = value
                else:
                    cmd += [flag, value]
            if profile['name'] == 'f16' and '--remote-attn-prefill' not in cmd:
                cmd += ['--remote-attn-prefill', 'remote']
            if '--no-warmup' not in cmd:
                cmd += ['--no-warmup']
            return cmd

        def contract(args, tokens, model):
            value = original['prefix_contract'](args, tokens, model)
            if profile['name'] == 'kvar4':
                return value
            value = dict(value)
            # The local P prefix is still the global KVarN4 target cache.  The
            # remote overflow format is a separate compatibility dimension.
            value['local_prefix_format'] = 'kvarn4'
            value['format'] = profile['name']
            value['remote_kv'] = {
                'cache_type_k': profile['remote_cache_k'],
                'cache_type_v': profile['remote_cache_v'],
                'upload': profile['upload'], 'wire': profile['wire'],
                'native_bulk': profile['native_bulk'] != '0',
                'descriptor': profile['descriptor'], 'producer': profile['producer'],
                'tail': profile['tail']}
            env = dict(value.get('environment', {}))
            env['GGML_RKVA_RANGE_STORAGE'] = profile['range_storage']
            env['GGML_RKVA_NATIVE_BULK'] = profile['native_bulk']
            env['GGML_RKVA_ASYNC_LOCAL_REMOTE'] = str(profile['async_overlap'])
            env['GGML_RKVA_WIRE_FORMAT'] = profile['wire']
            env['GGML_RKVA_CUDA_FP8'] = profile['cuda_fp8']
            env['GGML_RKVA_NATIVE_TAIL_TOKENS'] = str(profile['tail'])
            if profile['producer'] == 'cpu':
                env['GGML_RKVA_NATIVE_PRODUCER'] = 'cpu'
            elif profile['producer'] == 'serial':
                env.pop('GGML_RKVA_NATIVE_PRODUCER', None)
            value['environment'] = env
            return value

        def classify(lines):
            return classify_for(profile, lines, original['classify_logs'])

        def startup(args, lines):
            return startup_for(profile, args, lines, original['validate_startup'],
                               lambda value: classify_for(profile, value, original['classify_logs']))

        def body(tokens, count, seed, warm=False):
            value = original['request_body'](tokens, count, seed, warm)
            value = dict(value)
            value.update(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
                         repeat_penalty=1.0, presence_penalty=0.0, seed=1234)
            return value

        def fingerprint(path):
            path = Path(path).resolve(strict=True)
            stat = path.stat()
            key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ino)
            if key not in self.cache:
                self.cache[key] = original['fingerprint'](path)
            return self.cache[key]

        h.server_command = command
        h.pinned_environment = pinned
        h.prefix_contract = contract
        h.classify_logs = classify
        h.validate_startup = startup
        h.request_body = body
        h.fingerprint = fingerprint
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            setattr(H, name, value)
        return False


def classify_for(profile, lines, original):
    if profile['name'] == 'kvar4':
        return original(lines)
    lines = list(lines)
    bad_text = H.BAD_LOG.pattern.replace('|wire=f16', '').replace('|async_overlap=0', '')
    if profile['name'] == 'q8_0' and profile['async_overlap']:
        bad_text += '|async_overlap=0'
    if profile['name'] == 'q8_0':
        bad_text += '|using the synchronous range path|native Q8_0 bulk is not advertised'
    if profile['name'] == 'f16':
        bad_text += '|native Q8_0 bulk|EXPERIMENTAL native KVarN4|quantized-source expansion'
    bad = re.compile(bad_text, re.I)
    errors, warnings = set(), set()

    def confirmed(index):
        for line in lines[index + 1:]:
            if 'RemoteAttn: connected to ' in line:
                return False
            if profile['name'] == 'q8_0' and re.search(
                    r'native Q8_0 bulk descriptor=5[^\n]*wire=fp8_e5m2', line):
                return True
            if profile['name'] == 'f16' and 'range source=f16 storage=f16' in line:
                return True
        return False

    for index, line in enumerate(lines):
        if re.search(r'RemoteAttn: connected to \S+:\d+, \d+ remote layers, wire=f16, budget=\d+ MiB\s*$', line):
            if confirmed(index):
                continue
        if H.GRAPH_POLICY_WARNING.search(line) and not H.GRAPH_POLICY_FAILURE.search(line):
            warnings.add(line.strip())
        elif bad.search(line):
            errors.add(line.strip())
    return sorted(errors), sorted(warnings)


def startup_for(profile, args, lines, original, classify):
    if profile['name'] == 'kvar4':
        return original(args, lines)
    text = ''.join(lines)
    if profile['name'] == 'q8_0':
        pattern = (r'native Q8_0 bulk descriptor=5[^\n]*async_overlap=' +
                   str(profile['async_overlap']) + r'[^\n]*wire=fp8_e5m2')
        if not re.search(pattern, text):
            raise RuntimeError('native Q8_0 descriptor=5/FP8 negotiation was not confirmed')
        if 'using the synchronous range path' in text:
            raise RuntimeError('Q8_0 fell back to the synchronous range path')
    else:
        direct = 'range source=f16 storage=f16; direct original K/V half bytes, no quantized intermediate'
        if direct not in text:
            raise RuntimeError('F16 direct original K/V negotiation was not confirmed')
        if 'EXPERIMENTAL native KVarN4' in text or 'native Q8_0 bulk' in text:
            raise RuntimeError('F16 arm negotiated an incompatible native bulk format')
    if not re.search(r'position-split: P=' + str(args.p) + r' C=' + str(args.context) + r'\b', text):
        raise RuntimeError('effective P/C geometry was not confirmed in server logs')
    for field, expected in [('n_batch', args.batch), ('n_ubatch', args.ubatch)]:
        if not re.search(r'\b' + field + r'\s*=\s*' + str(expected) + r'\b', text):
            raise RuntimeError(f'effective {field}={expected} was not confirmed in server logs')
    errors, warnings = classify(lines)
    if errors:
        raise RuntimeError('server startup logged an error/fallback')
    return warnings


def parser():
    p = H.parser()
    p.description = __doc__
    for action in p._actions:
        if '--server-binary' in action.option_strings:
            action.required = False
            action.default = str(DEFAULT_SERVER)
        if '--points' in action.option_strings:
            action.default = '130560'
    p.add_argument('--repo', default=str(SCRIPT_REPO))
    p.add_argument('--formats', default='kvar4,q8_0,f16')
    p.add_argument('--order', choices=('ABBA', 'interleaved', 'sequential'), default='ABBA')
    p.add_argument('--prefix-dir', required=True)
    p.add_argument('--allow-new-prefix', action='store_true')
    p.add_argument('--kvar4-prefix-snapshot')
    p.add_argument('--q8-profile', choices=('match-kvarn4', 'serial'), default='match-kvarn4')
    return p


def validate(args):
    args.formats = parse_formats(args.formats)
    if args.context > MAX_CONTEXT:
        raise ValueError(f'--context must not exceed the protected {MAX_CONTEXT}-token cap')
    if args.prefix_only:
        raise ValueError('--prefix-only is not part of this study')
    if args.prefix_snapshot:
        raise ValueError('use --kvar4-prefix-snapshot for the existing KVarN4 prefix')
    if args.order == 'sequential' and args.execute:
        raise ValueError('--execute requires ABBA or interleaved order for same-format determinism')
    frozen = args.frozen_prefix_batch
    if frozen:
        # The base namespace still has the helper's B256 default; the KVarN4 arm
        # is validated against B1024 after this generic pass.
        args.frozen_prefix_batch = None
    args = H.validate(args)
    args.frozen_prefix_batch = frozen
    args.formats = parse_formats(args.formats)
    root = Path(args.output).resolve()
    prefix = Path(args.prefix_dir).resolve()
    for candidate in (root, prefix):
        if candidate in (Path('/'), Path.home(), SCRIPT_REPO):
            raise ValueError('--output/--prefix-dir must be dedicated directories, not /, $HOME or the repo root')
    if root == prefix or root in prefix.parents or prefix in root.parents:
        raise ValueError('--output and --prefix-dir must be disjoint dedicated roots')
    return args


def prefix_path(args, profile):
    if profile['name'] == 'kvar4' and args.kvar4_prefix_snapshot:
        return Path(args.kvar4_prefix_snapshot)
    batch = (args.frozen_prefix_batch or profile['batch']) if profile['name'] == 'kvar4' else profile['batch']
    return Path(args.prefix_dir).resolve() / profile['name'] / f"C{args.context}" / f"P{args.p}-B{batch}.slot"


def arm_args(base, profile, prefix, output, preview=False):
    values = vars(base).copy()
    values.update(points=','.join(str(n) for n in base.points),
                  batch=profile['batch'], ubatch=profile['ubatch'],
                  wavefront_tile=profile['wavefront'], wavefront_tiles=profile['wavefront_tiles'],
                  tcp_prefetch=profile['tcp_prefetch'], local_input_reuse=False,
                  n256_lookahead=False, cuda_fp8=profile['cuda_fp8'], mask_pack='default',
                  range_kernel=profile['range_kernel'], prefix_only=False,
                  prefix_snapshot=str(prefix), output=str(output), execute=True, dry_run=False)
    values['frozen_prefix_batch'] = base.frozen_prefix_batch if profile['name'] == 'kvar4' else None
    if preview:
        values['frozen_prefix_batch'] = None
    return H.validate(argparse.Namespace(**values))


def preview_contract(args, profile, tokens):
    arm = arm_args(args, profile, prefix_path(args, profile), Path(args.output), preview=True)
    if profile['name'] == 'kvar4':
        arm.frozen_prefix_batch = args.frozen_prefix_batch
    with Patched(profile):
        return H.prefix_contract(arm, tokens, {'sha256': '<frozen-on-execute>'})


def plan(args):
    formats = args.formats
    seq = sequence(formats, args.order)
    all_profiles = profiles(args.q8_profile)
    tokens, stream = H.effective_tokens(args.prompt_tokens, max([args.p] + args.points), args.repeat_prompt)
    result = {
        'schema': SCHEMA, 'mode': 'preview', 'starts_child': False, 'writes_output': False,
        'generated': now(), 'repo': str(Path(args.repo).resolve()),
        'helper': H_META, 'formats': formats, 'order': args.order, 'sequence': seq,
        'campaign': {'p': args.p, 'context': args.context, 'max_context': MAX_CONTEXT,
                     'points': args.points, 'n_predict': args.n_predict, 'seed': args.seed,
                     'greedy': True, 'output': str(Path(args.output).resolve()),
                     'prefix_dir': str(Path(args.prefix_dir).resolve()),
                     'allow_new_prefix': args.allow_new_prefix,
                     'frozen_prefix_batch': args.frozen_prefix_batch,
                     'frozen_scope': 'KVarN4 arm only; Q8_0/F16 use their own prefixes',
                     'q8_profile': args.q8_profile, 'mask_pack': 'DEFAULT/unset',
                     'token_stream': stream},
        'protected': {'p_unchanged': True, 'context_cap': MAX_CONTEXT,
                      'tiers_or_ini_changed': False, 'guards_changed': False,
                      'app_or_service_touched': False, 'gpu_children': 1},
        'device_flags': {'management': 'none; this tool never calls deploy.sh or touches the app',
                         'expected': 'INT8 prefill off (default); rowlane default state preserved',
                         'proof': 'KVarN4 GPU profile must show '
                                  'engine.kvarn4_multi_query.mq2_i8.calls=0; '
                                  'actual rowlane state is recorded'},
        'comparison_policy': ('Do not attribute a delta to KV format alone: wire, async overlap, '
                              'producer, query cadence, engine and kernel differ. Same-format repeats '
                              'must be deterministic; cross-format IDs/quality are diagnostics only.'),
        'publication_scrub': ['F4', 'F5', 'per-token arrays', 'nprobs', 'full token dumps'],
        'maps': {name: all_profiles[name] for name in formats},
        'prefixes': {}, 'commands': {}, 'environments': {}, 'contracts': {},
    }
    for name in formats:
        profile = all_profiles[name]
        path = prefix_path(args, profile)
        exists = path.is_file()
        frozen_here = args.frozen_prefix_batch if name == 'kvar4' else None
        intent = 'reuse' if exists else (
            'BLOCKED: frozen B512 snapshot missing' if frozen_here else
            ('create-with---allow-new-prefix' if args.allow_new_prefix else 'BLOCKED'))
        result['prefixes'][name] = {
            'path': str(path), 'exists': exists, 'intent': intent,
            'format_context_contract': 'format-specific; never reuses a KVarN4 contract without conversion'}
        arm = arm_args(args, profile, path, Path(args.output) / f'00-{name}', preview=True)
        with Patched(profile):
            result['commands'][name] = H.server_command(arm, '<unique-loopback-port>')
            result['environments'][name] = H.pinned_environment(arm)
        result['contracts'][name] = preview_contract(args, profile, tokens)
    return result


def parse_last_json(text, marker):
    value = None
    for line in text.splitlines():
        if marker in line:
            try:
                value = json.loads(line.split(marker, 1)[1].strip())
            except (ValueError, TypeError):
                continue
    return value


def _dict_at(obj, key, path, errors):
    if not isinstance(obj, dict):
        errors.append(f'{path} parent is not an object')
        return None
    if key not in obj:
        errors.append(f'{path} missing')
        return None
    value = obj[key]
    if not isinstance(value, dict):
        errors.append(f'{path} is not an object')
        return None
    return value


def _int_at(obj, key, path, errors, minimum=0, exact=None):
    if not isinstance(obj, dict) or key not in obj:
        errors.append(f'{path}.{key} missing')
        return None
    value = obj[key]
    if type(value) is not int:
        errors.append(f'{path}.{key} is not an integer')
        return None
    if value < minimum:
        errors.append(f'{path}.{key}={value} is below {minimum}')
    if exact is not None and value != exact:
        errors.append(f'{path}.{key}={value}, expected exactly {exact}')
    return value


def _bool_at(obj, key, path, errors):
    if not isinstance(obj, dict) or key not in obj:
        errors.append(f'{path}.{key} missing')
        return None
    value = obj[key]
    if type(value) is not bool:
        errors.append(f'{path}.{key} is not a boolean')
        return None
    return value


def _str_at(obj, key, path, errors):
    if not isinstance(obj, dict) or key not in obj:
        errors.append(f'{path}.{key} missing')
        return None
    value = obj[key]
    if not isinstance(value, str) or not value:
        errors.append(f'{path}.{key} is not a non-empty string')
        return None
    return value


def _list_at(obj, key, path, errors):
    if not isinstance(obj, dict) or key not in obj:
        errors.append(f'{path}.{key} missing')
        return None
    value = obj[key]
    if not isinstance(value, list):
        errors.append(f'{path}.{key} is not a list')
        return None
    return value


def _cuda_calls(stats, errors):
    keys = ('cuda_pack_calls', 'cuda_unpack_calls',
            'cuda_small_pack_calls', 'cuda_small_unpack_calls')
    total = 0
    for key in keys:
        value = _int_at(stats, key, 'remote_attn stats', errors)
        if value is not None:
            total += value
    return total


def _command_value(cmd, flag, errors):
    if flag not in cmd:
        errors.append(f'manifest.command missing {flag}')
        return None
    index = cmd.index(flag)
    if index + 1 >= len(cmd):
        errors.append(f'manifest.command {flag} has no value')
        return None
    return cmd[index + 1]


def _validate_command(profile, manifest, errors):
    cmd = manifest.get('command') if isinstance(manifest, dict) else None
    if not isinstance(cmd, list) or not all(isinstance(item, str) for item in cmd):
        errors.append('manifest.command missing or not a string list')
        return
    expected = [
        ('--cache-type-k', 'kvarn4'), ('--cache-type-v', 'kvarn4'),
        ('--remote-attn-cache-type-k', profile['remote_cache_k']),
        ('--remote-attn-cache-type-v', profile['remote_cache_v']),
        ('--remote-attn-format', profile['upload']),
        ('--remote-attn-wire-format', profile['wire']),
        ('--batch-size', str(profile['batch'])),
        ('--ubatch-size', str(profile['ubatch'])),
        ('--remote-attn-layers', 'full'),
        ('--flash-attn', 'on'),
    ]
    if profile['name'] == 'f16':
        expected.append(('--remote-attn-prefill', 'remote'))
    for flag, value in expected:
        if _command_value(cmd, flag, errors) != value:
            errors.append(f'manifest.command {flag} is not {value!r}')
    if '--no-warmup' not in cmd:
        errors.append('manifest.command did not disable automatic warmup')
    if any('.ini' in item or 'preset' in item or item in
           ('--override-kv', '--models-preset', '--models-dir') for item in cmd):
        errors.append('manifest.command attempts an INI/tier/override change')


def _validate_environment(profile, manifest, errors):
    env = manifest.get('environment') if isinstance(manifest, dict) else None
    if not isinstance(env, dict):
        errors.append('manifest.environment missing or not an object')
        return
    if 'GGML_CUDA_MASK_PACK' in env:
        errors.append('GGML_CUDA_MASK_PACK was forced instead of left at DEFAULT/unset')
    expected = [
        ('GGML_RKVA_RANGE_STORAGE', profile['range_storage']),
        ('GGML_RKVA_ASYNC_LOCAL_REMOTE', str(profile['async_overlap'])),
        ('GGML_RKVA_TCP_PREFETCH', '1' if profile['tcp_prefetch'] else '0'),
        ('GGML_RKVA_WIRE_FORMAT', profile['wire']),
        ('GGML_RKVA_CUDA_FP8', profile['cuda_fp8']),
        ('GGML_RKVA_NATIVE_TAIL_TOKENS', str(profile['tail'])),
        ('GGML_RKVA_RANGE_KERNEL', profile['range_kernel']),
        ('GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB', '128'),
        ('GGML_KVARN_WINDOW_CHUNK', '4096'),
    ]
    for key, value in expected:
        if env.get(key) != value:
            errors.append(f'manifest.environment {key}={env.get(key)!r}, expected {value!r}')
    bulk_env = {'kvar4': 'f16', 'q8_0': profile['native_bulk'], 'f16': '0'}[profile['name']]
    if env.get('GGML_RKVA_NATIVE_BULK') != bulk_env:
        errors.append(f"manifest.environment GGML_RKVA_NATIVE_BULK="
                      f"{env.get('GGML_RKVA_NATIVE_BULK')!r}, expected {bulk_env!r}")
    if profile['producer'] == 'serial':
        if 'GGML_RKVA_NATIVE_PRODUCER' in env:
            errors.append('serial F16 profile unexpectedly set a native producer')
    elif env.get('GGML_RKVA_NATIVE_PRODUCER') != profile['producer']:
        errors.append(f"manifest.environment GGML_RKVA_NATIVE_PRODUCER="
                      f"{env.get('GGML_RKVA_NATIVE_PRODUCER')!r}, expected {profile['producer']!r}")


def _validate_engine(profile, engine, transport, errors, unsupported):
    name = profile['name']
    native = name in ('kvar4', 'q8_0')
    record = {'counters': {}, 'transport_ops': []}
    schema = _str_at(engine, 'schema', 'gpu_profile.engine', errors)
    if schema != 'rkva-range-profile-v1':
        errors.append(f'gpu_profile.engine.schema={schema!r}, expected rkva-range-profile-v1')
    if _bool_at(engine, 'profiling', 'gpu_profile.engine', errors) is not True:
        errors.append('gpu_profile.engine.profiling is not true')
    counters = _dict_at(engine, 'counters', 'gpu_profile.engine', errors)
    for key in ('upload_calls', 'attention_calls', 'bulk_calls', 'bulk_segments',
                'bulk_queries', 'failed_calls', 'submits', 'dispatches',
                'buffer_creates', 'buffer_create_bytes'):
        record['counters'][key] = _int_at(counters, key, 'gpu_profile.engine.counters', errors)
    if record['counters']['attention_calls'] is not None and record['counters']['attention_calls'] <= 0:
        errors.append('gpu_profile.engine.counters.attention_calls is not positive')
    if record['counters']['upload_calls'] is not None and record['counters']['upload_calls'] <= 0:
        errors.append('gpu_profile.engine.counters.upload_calls is not positive')
    if record['counters']['failed_calls'] not in (None, 0):
        errors.append('gpu_profile.engine.counters.failed_calls is nonzero')
    for key in ('bulk_calls', 'bulk_segments', 'bulk_queries'):
        value = record['counters'][key]
        if value is None:
            continue
        if native and value <= 0:
            errors.append(f'gpu_profile.engine.counters.{key} is not positive for a native bulk route')
        if not native and value != 0:
            errors.append(f'gpu_profile.engine.counters.{key} is nonzero for synchronous F16')

    optimized = _dict_at(engine, 'optimized_kernels', 'gpu_profile.engine', errors)
    entry = _dict_at(optimized, profile['range_storage'], 'gpu_profile.engine.optimized_kernels', errors)
    available = _bool_at(entry, 'available', 'gpu_profile.engine.optimized_kernels', errors)
    optimized_calls = _int_at(entry, 'optimized_calls', 'gpu_profile.engine.optimized_kernels', errors)
    reference_calls = _int_at(entry, 'reference_calls', 'gpu_profile.engine.optimized_kernels', errors)
    record['optimized'] = {'available': available, 'optimized_calls': optimized_calls,
                           'reference_calls': reference_calls}
    if available is False or (optimized_calls is not None and optimized_calls <= 0) or \
            (reference_calls is not None and reference_calls != 0):
        if name == 'kvar4':
            errors.append(f"gpu_profile.engine.optimized_kernels.{profile['range_storage']} "
                          'is unavailable or fell back to reference')
        else:
            unsupported.append(f"gpu_profile.engine.optimized_kernels.{profile['range_storage']} "
                               f'is unavailable or fell back to reference; {name} not qualified')

    buckets = _dict_at(engine, 'buckets', 'gpu_profile.engine', errors)
    record['kernels'] = []
    record['bucket'] = None
    if isinstance(buckets, dict):
        if not buckets:
            errors.append('gpu_profile.engine.buckets is empty')
        if not all(isinstance(key, str) for key in buckets):
            errors.append('gpu_profile.engine.buckets has a non-string key')
        prefix = {'kvar4': 'attention/kvarn4-prefix/optimized-',
                  'q8_0': 'attention/q8_0/optimized-',
                  'f16': 'attention/f16/optimized-'}[name]
        record['kernels'] = sorted(key for key in buckets if isinstance(key, str) and
                                   key.startswith('attention/'))
        matches = [key for key in buckets if isinstance(key, str) and key.startswith(prefix)]
        if not matches:
            if native and name == 'kvar4':
                errors.append(f'no {prefix} bucket; KVarN4 revision-3 prefix route not proven')
            else:
                unsupported.append(f'no {prefix} bucket; optimized {name} route not proven')
        else:
            record['bucket'] = matches[0]
            bucket = _dict_at(buckets, matches[0], 'gpu_profile.engine.buckets', errors)
            _int_at(bucket, 'samples', 'gpu_profile.engine.buckets.samples', errors, minimum=1)

    mq = _dict_at(engine, 'kvarn4_multi_query', 'gpu_profile.engine', errors)
    mq2 = _dict_at(mq, 'mq2', 'gpu_profile.engine.kvarn4_multi_query', errors)
    mq4 = _dict_at(mq, 'mq4', 'gpu_profile.engine.kvarn4_multi_query', errors)
    mq2_calls = _int_at(mq2, 'calls', 'gpu_profile.engine.kvarn4_multi_query', errors)
    mq4_calls = _int_at(mq4, 'calls', 'gpu_profile.engine.kvarn4_multi_query', errors)
    i8 = _dict_at(mq, 'mq2_i8', 'gpu_profile.engine.kvarn4_multi_query', errors)
    _bool_at(i8, 'available', 'gpu_profile.engine.kvarn4_multi_query.mq2_i8', errors)
    record['int8_calls'] = _int_at(i8, 'calls', 'gpu_profile.engine.kvarn4_multi_query.mq2_i8',
                                   errors, exact=0)
    if name == 'kvar4' and mq2_calls is not None and mq4_calls is not None and mq2_calls + mq4_calls <= 0:
        errors.append('gpu_profile.engine.kvarn4_multi_query has no actual KVarN4 multi-query calls')
    if name != 'kvar4' and ((mq2_calls is not None and mq2_calls > 0) or
                            (mq4_calls is not None and mq4_calls > 0)):
        errors.append('KVarN4 multi-query calls appeared in a Q8_0/F16 arm')

    rowlane = _dict_at(engine, 'kvarn4_decode_rowlane', 'gpu_profile.engine', errors)
    _bool_at(rowlane, 'available', 'gpu_profile.engine.kvarn4_decode_rowlane', errors)
    record['rowlane'] = {
        'enabled': _bool_at(rowlane, 'enabled', 'gpu_profile.engine.kvarn4_decode_rowlane', errors),
        'calls': _int_at(rowlane, 'calls', 'gpu_profile.engine.kvarn4_decode_rowlane', errors)}
    if name != 'kvar4' and record['rowlane']['calls'] not in (None, 0):
        errors.append('KVarN4 rowlane calls appeared in a Q8_0/F16 arm')

    format_mq = _dict_at(engine, 'format_multi_query', 'gpu_profile.engine', errors)
    if name != 'kvar4':
        fmt = _dict_at(format_mq, profile['range_storage'],
                       'gpu_profile.engine.format_multi_query', errors)
        for tile in ('mq2', 'mq4'):
            node = _dict_at(fmt, tile, 'gpu_profile.engine.format_multi_query', errors)
            _int_at(node, 'calls', f'gpu_profile.engine.format_multi_query.{tile}', errors)

    bulk = _dict_at(engine, 'bulk', 'gpu_profile.engine', errors)
    fp8_gpu_calls = _int_at(bulk, 'fp8_gpu_calls', 'gpu_profile.engine.bulk', errors)
    record['bulk'] = {
        'fp8_gpu_calls': fp8_gpu_calls,
        'fp8_cpu_fallback_calls': _int_at(bulk, 'fp8_cpu_fallback_calls',
                                          'gpu_profile.engine.bulk', errors, exact=0)}
    if native and fp8_gpu_calls is not None and fp8_gpu_calls <= 0:
        errors.append('gpu_profile.engine.bulk.fp8_gpu_calls is not positive')
    if not native and fp8_gpu_calls not in (None, 0):
        errors.append('gpu_profile.engine.bulk.fp8_gpu_calls is nonzero for F16')

    if _bool_at(transport, 'enabled', 'gpu_profile.transport', errors) is not True:
        errors.append('gpu_profile.transport.enabled is not true')
    _bool_at(transport, 'trace_requests', 'gpu_profile.transport', errors)
    fp8_gpu_enabled = _bool_at(transport, 'fp8_gpu_enabled', 'gpu_profile.transport', errors)
    if native and fp8_gpu_enabled is not True:
        errors.append('gpu_profile.transport.fp8_gpu_enabled is not true for a native FP8 route')
    ops = _list_at(transport, 'ops', 'gpu_profile.transport', errors)
    ops_by_number = {}
    if ops is not None:
        for index, op in enumerate(ops):
            if not isinstance(op, dict):
                errors.append(f'gpu_profile.transport.ops[{index}] is not an object')
                continue
            number = _int_at(op, 'op', f'gpu_profile.transport.ops[{index}]', errors)
            calls = _int_at(op, 'calls', f'gpu_profile.transport.ops[{index}]', errors)
            request = _int_at(op, 'request_bytes', f'gpu_profile.transport.ops[{index}]', errors)
            response = _int_at(op, 'response_bytes', f'gpu_profile.transport.ops[{index}]', errors)
            if number is not None:
                if number in ops_by_number:
                    errors.append(f'gpu_profile.transport.ops has duplicate op {number}')
                ops_by_number[number] = {'calls': calls, 'request_bytes': request,
                                         'response_bytes': response}
    bulk_op = ops_by_number.get(22)
    if native:
        if not bulk_op or bulk_op['calls'] is None or bulk_op['calls'] <= 0 or \
                bulk_op['request_bytes'] is None or bulk_op['request_bytes'] <= 0 or \
                bulk_op['response_bytes'] is None or bulk_op['response_bytes'] <= 0:
            errors.append('gpu_profile.transport op22 real bulk request/response counters missing')
    else:
        if bulk_op and bulk_op['calls'] not in (None, 0):
            errors.append('F16 transport unexpectedly used bulk op22')
        sync_ops = [number for number in (13, 14, 19)
                    if number in ops_by_number and
                    ops_by_number[number]['calls'] is not None and
                    ops_by_number[number]['calls'] > 0 and
                    ops_by_number[number]['request_bytes'] is not None and
                    ops_by_number[number]['request_bytes'] > 0 and
                    ops_by_number[number]['response_bytes'] is not None and
                    ops_by_number[number]['response_bytes'] > 0]
        if not sync_ops:
            errors.append('F16 transport has no real synchronous range upload/attention op')
    record['transport_ops'] = sorted(ops_by_number)
    return record


def prove_format(profile, result, manifest, log, stats, gpu):
    errors, unsupported = [], []
    manifest = manifest if isinstance(manifest, dict) else {}
    if not isinstance(result, dict) or result.get('status') != 'passed':
        errors.append('helper run did not pass: ' + str((result or {}).get('error', 'missing results')))
    if not all(point.get('valid') for point in (result or {}).get('points', [])):
        errors.append('one or more points failed helper qualification')
    if not stats:
        errors.append('no final remote_attn stats proof')
    if not gpu:
        errors.append('no remote_attn gpu_profile proof (profiling must stay on)')
    stats = stats if isinstance(stats, dict) else {}
    text = log or ''
    _validate_command(profile, manifest, errors)
    _validate_environment(profile, manifest, errors)
    description = {
        'format': profile['name'], 'roles': profile['roles'],
        'stats': {key: stats.get(key) for key in (
            'range_cache_type', 'range_source_cache_type', 'wire_format',
            'native_precision_tail_tokens', 'calls', 'range_upload_bytes',
            'range_upload_calls', 'range_upload_rows', 'range_attn_calls',
            'tcp_prefetch_completed')},
        'kernels': [], 'bucket': None, 'int8_calls': None, 'rowlane': None,
        'bulk': None, 'transport_ops': [], 'engine_schema': None,
    }
    if isinstance(gpu, dict):
        engine = gpu.get('engine')
        transport = gpu.get('transport')
        if not isinstance(engine, dict) or not engine:
            errors.append('gpu_profile.engine missing or empty')
        else:
            description['engine_schema'] = engine.get('schema')
            record = _validate_engine(profile, engine, transport, errors, unsupported)
            description.update(record)
        if not isinstance(transport, dict) or not transport:
            errors.append('gpu_profile.transport missing or empty')
        elif isinstance(engine, dict) and engine:
            if not description['transport_ops']:
                errors.append('gpu_profile.transport.ops has no authenticated op counters')

    for key in ('calls', 'range_upload_bytes', 'range_upload_calls',
                'range_upload_rows', 'range_attn_calls'):
        value = _int_at(stats, key, 'remote_attn stats', errors)
        if value is not None and value <= 0:
            errors.append(f'remote_attn stats {key} is not positive')
    _int_at(stats, 'native_precision_tail_tokens', 'remote_attn stats', errors)
    for key in ('range_cache_type', 'range_source_cache_type', 'wire_format'):
        _str_at(stats, key, 'remote_attn stats', errors)
    cuda_calls = _cuda_calls(stats, errors)

    if profile['name'] == 'kvar4':
        if not re.search(r'EXPERIMENTAL native KVarN4[^\n]*precision_tail=128[^\n]*descriptor=3'
                         r'[^\n]*async_overlap=1[^\n]*wire=fp8_e5m2', text):
            errors.append('KVarN4 descriptor=3/async=1/FP8 negotiation missing')
        expected = {'range_cache_type': 'kvarn4', 'range_source_cache_type': 'projected_f32',
                    'wire_format': 'fp8_e5m2', 'native_precision_tail_tokens': 128}
        for key, value in expected.items():
            if stats.get(key) != value:
                errors.append(f'KVarN4 stats {key}={stats.get(key)!r}, expected {value!r}')
        tcp = _int_at(stats, 'tcp_prefetch_completed', 'remote_attn stats', errors)
        if profile['tcp_prefetch'] and tcp is not None and tcp <= 0:
            errors.append('KVarN4 TCP prefetch completion counters missing')
        if cuda_calls <= 0:
            errors.append('no CUDA FP8 pack/unpack calls; mapped FP8 route not proven')
        if description['rowlane'] and description['rowlane']['enabled'] is False:
            description.setdefault('warnings', []).append(
                'rowlane was disabled on the device; state recorded, not changed')
    elif profile['name'] == 'q8_0':
        if not re.search(r'native Q8_0 bulk descriptor=5[^\n]*async_overlap=' +
                         str(profile['async_overlap']) + r'[^\n]*wire=fp8_e5m2', text):
            errors.append('native Q8_0 descriptor=5/FP8 negotiation missing')
        if 'using the synchronous range path' in text or 'native Q8_0 bulk is not advertised' in text:
            unsupported.append('Q8_0 did not negotiate native bulk; synchronous fallback is not the requested route')
        # The server reports projected_f32 for any native owner, including Q8_0;
        # range_cache_type is the format discriminator (layout 5 => q8_0).
        expected = {'range_cache_type': 'q8_0', 'range_source_cache_type': 'projected_f32',
                    'wire_format': 'fp8_e5m2', 'native_precision_tail_tokens': 0}
        for key, value in expected.items():
            if stats.get(key) != value:
                errors.append(f'Q8_0 stats {key}={stats.get(key)!r}, expected {value!r}')
        tcp = _int_at(stats, 'tcp_prefetch_completed', 'remote_attn stats', errors)
        if profile['tcp_prefetch'] and tcp is not None and tcp <= 0:
            errors.append('Q8_0 TCP prefetch completion counters missing')
    else:
        if 'range source=f16 storage=f16; direct original K/V half bytes, no quantized intermediate' not in text:
            errors.append('F16 direct original K/V negotiation missing')
        if 'quantized-source expansion' in text:
            errors.append('F16 arm tried to expand a quantized source')
        if 'EXPERIMENTAL native KVarN4' in text or 'native Q8_0 bulk' in text:
            errors.append('F16 arm negotiated an incompatible native bulk format')
        expected = {'range_cache_type': 'f16', 'range_source_cache_type': 'f16',
                    'native_precision_tail_tokens': 0}
        for key, value in expected.items():
            if stats.get(key) != value:
                errors.append(f'F16 stats {key}={stats.get(key)!r}, expected {value!r}')
        if stats.get('wire_format') not in ('f16', 'f32'):
            errors.append('F16 Q/OUT wire is not the explicit f16/f32 profile')
        tcp = _int_at(stats, 'tcp_prefetch_completed', 'remote_attn stats', errors,
                      exact=0)
        if tcp not in (None, 0):
            errors.append('F16 unexpectedly used TCP prefetch/async bulk')
        if cuda_calls != 0:
            errors.append('F16 CUDA FP8 helpers were not off')
    status = 'UNSUPPORTED' if unsupported else ('FAIL' if errors else 'PASS')
    description.update(status=status, errors=errors, unsupported=unsupported)
    return description


def inspect_arm(args, profile, output, rc, freeze):
    output = Path(output)
    result = manifest = None
    if (output / 'results.json').is_file():
        result = json.loads((output / 'results.json').read_text())
    if (output / 'manifest.json').is_file():
        manifest = json.loads((output / 'manifest.json').read_text())
    log = (output / 'server.log').read_text(errors='replace') if (output / 'server.log').is_file() else ''
    stats = parse_last_json(log, 'remote_attn stats ')
    gpu = parse_last_json(log, 'remote_attn gpu_profile ')
    proof = prove_format(profile, result, manifest, log, stats, gpu)
    if rc != 0:
        if proof['status'] == 'PASS':
            proof['status'] = 'FAIL'
        proof.setdefault('errors', []).append(f'bench-rkva-range.py returned {rc}')
        run_error = str((result or {}).get('error', '')) + '\n' + log
        if profile['name'] == 'q8_0' and re.search(
                r'wavefront|TCP prefetch|requested geometry|WF\d+|native Q8_0 bulk is not advertised|'
                r'not advertised by the negotiated server', run_error, re.I):
            proof['status'] = 'UNSUPPORTED'
            proof.setdefault('unsupported', []).append(
                'matched Q8_0 wavefront/TCP/native profile was not qualified; '
                'rerun the declared --q8-profile serial alternative')
        if profile['name'] == 'f16' and re.search(
                r'optimized f16 range kernel|native bulk RPC is not supported|'
                r'upload format f16 requires', run_error, re.I):
            proof['status'] = 'UNSUPPORTED'
            proof.setdefault('unsupported', []).append(
                'F16 synchronous profile is not advertised by this server')
    points = []
    for point in (result or {}).get('points', []):
        label = f"point-{point.get('prompt_tokens')}"
        response = output / f'{label}.response.json'
        tokens = None
        if response.is_file():
            tokens = json.loads(response.read_text()).get('tokens')
        points.append({'label': label, 'prompt_tokens': point.get('prompt_tokens'),
                       'cache_n': point.get('response_timings', {}).get('cache_n'),
                       'prompt_n': point.get('response_timings', {}).get('prompt_n'),
                       'predicted_n': point.get('response_timings', {}).get('predicted_n'),
                       'predicted_ms': point.get('response_timings', {}).get('predicted_ms'),
                       'valid': point.get('valid'), 'tokens': tokens,
                       'token_ids_sha256': hashlib.sha256(
                           b''.join(int(t).to_bytes(4, 'little') for t in tokens or [])).hexdigest()})
    warm = None
    for name in ('warm', 'prefix-repair'):
        path = output / f'{name}.response.json'
        if path.is_file():
            warm = json.loads(path.read_text()).get('timings')
            break
    for point in points:
        expected_prompt = (point.get('prompt_tokens') or 0) - args.p
        if point.get('cache_n') != args.p or point.get('prompt_n') != expected_prompt:
            proof['status'] = 'FAIL'
            proof.setdefault('errors', []).append(
                f"{point.get('label')}: cache_n/prompt_n did not prove cached P and "
                f'{expected_prompt} overflow tokens')
    prefix = (manifest or {}).get('prefix_cache', {})
    if prefix.get('intent') == 'create':
        if not warm or warm.get('cache_n') != 0 or warm.get('prompt_n') != args.p:
            proof['status'] = 'FAIL'
            proof.setdefault('errors', []).append('new snapshot was not created from a cold P prefix')
    elif prefix.get('intent') == 'restore' and not prefix.get('bounded_replay'):
        proof['status'] = 'FAIL'
        proof.setdefault('errors', []).append('restored prefix did not prove the bounded <=128 replay')
    freeze['snapshots'] = freeze.get('snapshots', {})
    path = Path(args.prefix_snapshot)
    if path.is_file():
        current = {item.name: {'size': item.stat().st_size, 'sha256': H.sha256(item)}
                   for item in H.snapshot_files(path)}
        previous = freeze['snapshots'].get(str(path))
        if previous is None:
            freeze['snapshots'][str(path)] = current
        elif previous != current:
            proof['status'] = 'FAIL'
            proof.setdefault('errors', []).append('prefix snapshot files changed during the arm')
    arm = {'format': profile['name'], 'label': profile['label'], 'output': str(output),
           'rc': rc, 'status': proof['status'], 'proof': proof, 'points': points,
           'warm': warm, 'prefix_intent': prefix.get('intent'),
           'manifest_status': (result or {}).get('status'),
           'server_error': (result or {}).get('error')}
    return arm, result, manifest


def _point_key(point):
    return point.get('label')


def campaign_summary(arms, n_predict):
    if not arms:
        return {'schema': SCHEMA, 'status': 'FAIL', 'formats': {},
                'comparison': {'accounting_congruent': False, 'quality': {}, 'deltas': {},
                               'interpretation': 'no completed arms'},
                'publication_scrub': ['F4', 'F5', 'per-token arrays', 'nprobs', 'full token dumps']}
    groups = {}
    for arm in arms:
        groups.setdefault(arm['format'], []).append(arm)
    summary = {}
    all_pass = True
    for name, runs in groups.items():
        statuses = [run['status'] for run in runs]
        point_map = {}
        for run in runs:
            for point in run['points']:
                point_map.setdefault(_point_key(point), []).append(point)
        deterministic, reasons = True, []
        pooled = {}
        for label, points in point_map.items():
            points = sorted(points, key=lambda p: (p.get('predicted_ms') or 0, p.get('token_ids_sha256') or ''))
            tokens = [p.get('tokens') for p in points]
            counts = [(p.get('cache_n'), p.get('prompt_n'), p.get('predicted_n')) for p in points]
            if len(points) < 2:
                deterministic = False
                reasons.append(f'{label}: fewer than two same-format repetitions')
            if any(value is None for value in tokens) or len({tuple(value) for value in tokens}) != 1:
                deterministic = False
                reasons.append(f'{label}: generated token IDs differ across same-format repetitions')
            if len(set(counts)) != 1:
                deterministic = False
                reasons.append(f'{label}: cache/prompt/predicted counts differ across repetitions')
            steps = sum(max(0, int(p.get('predicted_n') or 0) - 1) for p in points)
            ms = sum(float(p.get('predicted_ms') or 0) for p in points)
            if steps <= 0 or ms <= 0:
                deterministic = False
                reasons.append(f'{label}: no positive pooled decode interval')
            pooled[label] = {'steps': steps, 'ms': ms,
                             'pooled_decode_tps': steps * 1000 / ms if ms else None,
                             'expected_steps_per_run': n_predict - 1}
        upload = max((run['proof'].get('stats', {}).get('range_upload_bytes') or 0) for run in runs)
        summary[name] = {'statuses': statuses, 'deterministic': deterministic,
                         'determinism_errors': reasons, 'pooled_decode': pooled,
                         'upload_bytes': upload or None,
                         'gpu_mem': None, 'host_ram': None,
                         'memory_measurement': 'not instrumented by bench-rkva-range.py; attach external measurements',
                         'proof': runs[-1]['proof']}
        if any(status != 'PASS' for status in statuses) or not deterministic:
            all_pass = False
    passing = [name for name, value in summary.items() if value['statuses'] == ['PASS'] * len(value['statuses'])
               and value['deterministic']]
    labels = sorted({p['label'] for arm in arms for p in arm['points']})
    congruent, mismatch = True, []
    if len(passing) >= 2:
        for label in labels:
            values = {}
            for name in passing:
                run = next(run for run in groups[name] if any(p['label'] == label for p in run['points']))
                point = next(p for p in run['points'] if p['label'] == label)
                values[name] = (point.get('cache_n'), point.get('prompt_n'))
            if len(set(values.values())) != 1:
                congruent = False
                mismatch.append({'label': label, 'values': values})
    baseline = 'kvar4' if 'kvar4' in passing else (passing[0] if passing else None)
    quality = {}
    if baseline:
        for name in passing:
            if name == baseline:
                continue
            divergences = []
            for label in labels:
                base = next((p for p in groups[baseline][0]['points'] if p['label'] == label), None)
                other = next((p for p in groups[name][0]['points'] if p['label'] == label), None)
                if not base or not other:
                    continue
                a, b = base.get('tokens') or [], other.get('tokens') or []
                first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                divergences.append({'label': label, 'identical': a == b, 'first_divergence': first,
                                    'token_ids_sha256': other.get('token_ids_sha256')})
            quality[name] = {'baseline': baseline, 'diagnostic_only': True,
                             'note': 'quantization/cadence changes IDs; equality is not required across formats',
                             'points': divergences}
    deltas = {}
    if congruent and baseline:
        for name in passing:
            if name == baseline:
                continue
            deltas[name] = {}
            for label in labels:
                base = next((p for p in groups[baseline][0]['points'] if p['label'] == label), None)
                other = next((p for p in groups[name][0]['points'] if p['label'] == label), None)
                if base and other:
                    deltas[name][label] = {
                        'baseline': baseline,
                        'pooled_decode_tps_delta_vs_baseline': (
                            summary[name]['pooled_decode'][label]['pooled_decode_tps'] -
                            summary[baseline]['pooled_decode'][label]['pooled_decode_tps']),
                        'latency_ms_delta_vs_baseline': (
                            other.get('predicted_ms', 0) - base.get('predicted_ms', 0)),
                        'upload_bytes_delta_vs_baseline': (
                            (summary[name]['upload_bytes'] or 0) - (summary[baseline]['upload_bytes'] or 0))}
    if all_pass and congruent:
        status = 'PASS'
    elif any(status == 'FAIL' for value in summary.values() for status in value['statuses']):
        status = 'FAIL'
    elif any(status == 'UNSUPPORTED' for value in summary.values() for status in value['statuses']):
        status = 'UNSUPPORTED'
    else:
        status = 'FAIL'
    return {'schema': SCHEMA, 'status': status, 'formats': summary,
            'comparison': {'baseline': baseline, 'passing_formats': passing,
                           'accounting_congruent': congruent, 'accounting_mismatch': mismatch,
                           'quality': quality, 'deltas': deltas,
                           'interpretation': ('diagnostic only; transport/producer/wire/kernel '
                                              'differ and are not isolated by this study')},
            'publication_scrub': ['F4', 'F5', 'per-token arrays', 'nprobs', 'full token dumps']}


def freeze_inputs(args, tokens):
    return {
        'model': H.fingerprint(args.model), 'server': H.fingerprint(args.server_binary),
        'prompt': H.fingerprint(args.prompt_tokens), 'source': H.git_identity(),
        'libraries': H.local_libraries(args), 'memory_policy': H.memory_policy(),
        'token_stream': H.effective_tokens(args.prompt_tokens, max([args.p] + args.points),
                                           args.repeat_prompt)[1],
        'snapshots': {},
    }


def run_campaign(args):
    root = Path(args.output).resolve()
    if root.exists():
        raise ValueError(f'output already exists; refusing to overwrite: {root}')
    prefix_root = Path(args.prefix_dir).resolve()
    formats = args.formats
    all_profiles = profiles(args.q8_profile)
    tokens, _ = H.effective_tokens(args.prompt_tokens, max([args.p] + args.points),
                                   args.repeat_prompt)
    with Patched(all_profiles[formats[0]]):
        freeze = freeze_inputs(args, tokens)
    for name in formats:
        path = prefix_path(args, all_profiles[name])
        arm = arm_args(args, all_profiles[name], path, root / 'preflight', preview=False)
        path = Path(arm.prefix_snapshot)
        with Patched(all_profiles[name]):
            if path.is_file():
                try:
                    H.load_prefix_snapshot(arm, tokens, freeze['model'])
                except ValueError as error:
                    raise ValueError(f'{name} prefix contract mismatch at {path}: {error}') from error
                freeze['snapshots'][str(path)] = {
                    item.name: {'size': item.stat().st_size, 'sha256': H.sha256(item)}
                    for item in H.snapshot_files(path)}
            elif not args.allow_new_prefix:
                raise ValueError(f'{name} prefix snapshot missing at {path}; pass --allow-new-prefix '
                                 'to create a new format-owned prefix from a cold P prefix')
    root.mkdir(parents=True, exist_ok=False)
    prefix_root.mkdir(parents=True, exist_ok=True)
    seq = sequence(formats, args.order)
    study = {'schema': SCHEMA, 'started': now(), 'mode': 'execute',
             'helper': H_META, 'freeze': freeze, 'campaign': {
                 'p': args.p, 'context': args.context, 'max_context': MAX_CONTEXT,
                 'points': args.points, 'n_predict': args.n_predict, 'seed': args.seed,
                 'formats': formats, 'order': args.order, 'sequence': seq,
                 'prefix_dir': str(prefix_root), 'output': str(root),
                 'allow_new_prefix': args.allow_new_prefix,
                 'frozen_prefix_batch': args.frozen_prefix_batch,
                 'frozen_scope': 'KVarN4 arm only; Q8_0/F16 use their own prefixes',
                 'q8_profile': args.q8_profile, 'mask_pack': 'DEFAULT/unset'},
             'protected': {'p_unchanged': True, 'context_cap': MAX_CONTEXT,
                           'tiers_or_ini_changed': False, 'guards_changed': False,
                           'app_or_service_touched': False, 'gpu_children': 1},
             'device_flags': {'management': 'none; no deploy.sh/app/service access',
                              'expected_int8': 'off', 'actual_rowlane': None},
             'hardware_claim': 'none; only the caller may run hardware after QA INT8'}
    (root / 'study-manifest.json').write_text(json.dumps(study, indent=2, allow_nan=False) + '\n')
    arms = []
    try:
        for index, name in enumerate(seq):
            profile = all_profiles[name]
            output = root / f'{index:02d}-{name}'
            prefix = prefix_path(args, profile)
            arm = arm_args(args, profile, prefix, output, preview=False)
            print(f'{now()} arm {index + 1}/{len(seq)} format={name} output={output}', flush=True)
            try:
                with Patched(profile):
                    rc = H.run(arm, tokens, freeze['token_stream'])
            except Exception as error:
                print(f'{now()} arm {name} wrapper failure: {type(error).__name__}: {error}', flush=True)
                rc = 1
            checked, result, manifest = inspect_arm(arm, profile, output, rc, freeze)
            checked['occurrence'] = index
            checked['requested_p'] = args.p
            checked['requested_context'] = args.context
            arms.append(checked)
            (root / 'study-arms.json').write_text(json.dumps(
                [{k: v for k, v in arm.items() if k != 'points'} | {
                    'points': [{kk: vv for kk, vv in p.items() if kk != 'tokens'} for p in arm['points']]}
                 for arm in arms], indent=2, allow_nan=False) + '\n')
            if checked['status'] != 'PASS':
                print(f'{now()} arm {name} status={checked["status"]}: '
                      f'{checked["proof"].get("errors") or checked["proof"].get("unsupported")}', flush=True)
    finally:
        final_source = H.git_identity()
        final_bad = []
        for key in ('head', 'tracked_diff_sha256'):
            if final_source.get(key) != freeze['source'].get(key):
                final_bad.append(f'source {key} changed')
        for item in (freeze['model'], freeze['server'], freeze['prompt'], *freeze['libraries']):
            stat = Path(item['path']).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (item['size'], item['mtime_ns'], item['inode']):
                final_bad.append('input changed: ' + item['path'])
        for path, expected in freeze.get('snapshots', {}).items():
            if not Path(path).is_file():
                final_bad.append('prefix snapshot disappeared: ' + path)
                continue
            current = {item.name: {'size': item.stat().st_size, 'sha256': H.sha256(item)}
                       for item in H.snapshot_files(path)}
            if current != expected:
                final_bad.append('prefix snapshot changed: ' + path)
        study['device_flags']['actual'] = {
            arm['format']: {'rowlane': arm['proof'].get('rowlane'),
                            'int8_calls': arm['proof'].get('int8_calls')}
            for arm in arms}
        summary = campaign_summary(arms, args.n_predict)
        summary.update(started=study['started'], finished=now(), freeze=freeze,
                       helper=H_META, campaign=study['campaign'],
                       protected=study['protected'], device_flags=study['device_flags'],
                       hardware_claim=study['hardware_claim'],
                       freeze_errors=final_bad)
        if final_bad:
            summary['status'] = 'FAIL'
        (root / 'study-results.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        study['finished'] = now()
        study['arms'] = [{k: v for k, v in arm.items() if k != 'points'} | {
            'points': [{kk: vv for kk, vv in p.items() if kk != 'tokens'} for p in arm['points']]}
            for arm in arms]
        study['summary'] = summary
        (root / 'study-manifest.json').write_text(json.dumps(study, indent=2, allow_nan=False) + '\n')
    return 0 if summary['status'] == 'PASS' else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--repo', default=str(SCRIPT_REPO))
    known, _ = pre.parse_known_args(argv)
    try:
        load_h(known.repo)
        args = validate(parser().parse_args(argv))
    except (ValueError, OSError) as error:
        print(f'Configuration error: {error}', file=sys.stderr)
        return 2
    if not args.execute:
        try:
            print(json.dumps(plan(args), indent=2, ensure_ascii=False, allow_nan=False))
        except (ValueError, OSError) as error:
            print(f'Configuration error: {error}', file=sys.stderr)
            return 2
        return 0
    try:
        return run_campaign(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Campaign error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
