#!/usr/bin/env python3
"""Offline RAW logits campaign controller for the RKVA INT8 OFF/ON matrix.

This tool is inert by default.  It reads the frozen reference campaign, validates
the new case catalog and the prefix snapshot, prints a plan, and writes nothing.
Only ``--execute`` starts the CAPI capture child, touches Xbox LocalState flags,
or contacts the deploy endpoint.

The capture child is ``tools/xbox/tests/rkva_quality_capture`` (built from the
root checkout).  It writes all 248320 logits per teacher step as float32 LE.
This controller never parses the slow HTTP full-vocab JSON: it computes stable
float64 log-sum-exp, exact teacher NLL and full-vocabulary KL from the raw rows.

The result label is deliberately narrow: synthetic/default data numerical
stress only, never a broad semantic qualification and never INT8 default
approval.  No gate is relaxed automatically.
"""
from __future__ import annotations

import argparse
import array
import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace
import shutil
import signal
import struct
import subprocess
import sys
import threading

TOOL = 'rkva-raw-quality-matrix'
SCHEMA = 'rkva-raw-quality-matrix-v1'
GATE_REL = Path('tools/xbox/scripts/rkva-quality-gate.py')
HELPER_REL = Path('tools/xbox/scripts/bench-rkva-range.py')
DEPLOY_REL = Path('tools/xbox/scripts/deploy.sh')

P_DEFAULT = 102400
CONTEXT_DEFAULT = 131072
VOCAB_DEFAULT = 248320
TEACHER_STEPS_DEFAULT = 128
XBOX_HOST_DEFAULT = '192.168.1.200:50054'
XBOX_ENV_DEFAULT = Path.home() / '.config/xllama/xbox-env'
CHILD_TIMEOUT_DEFAULT = 7200
DEPLOY_TIMEOUT_DEFAULT = 120
DISK_HEADROOM_GB = 1.0

ENV_PREFIXES = ('GGML_', 'LLAMA_', 'CUDA_', 'XLLAMA_')
CASE_NAME_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*\Z')
MASK_PACK = 'GGML_CUDA_MASK_PACK'
STARTUP_MARKER = 'llama_context: backend_ptrs.size()'
STEP0_MARKER = 'CAPTURE step=0 '

_GATE = None
_HELPER = None


class MatrixError(RuntimeError):
    """Inconsistent capture/measurement data; never a PASS."""


class RefuseError(MatrixError):
    """Unsafe local configuration; refuse before any flags/network/hardware."""


class ChildCleanupError(MatrixError):
    """The owned capture child may still be alive or could not be reaped."""

    def __init__(self, message, pid, alive, details):
        super().__init__(message)
        self.pid = pid
        self.alive = bool(alive)
        self.details = details


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_text(value):
    return sha256_bytes(value.encode('utf-8'))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_bytes_new(path, value):
    path = Path(path)
    if path.exists():
        raise RefuseError(f'refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def write_json_new(path, value):
    path = Path(path)
    if path.exists():
        raise RefuseError(f'refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    if temporary.exists():
        raise RefuseError(f'refusing to overwrite temporary {temporary}')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def make_new_dir(path):
    path = Path(path)
    if path.exists():
        raise RefuseError(f'refusing to reuse existing directory {path}')
    path.mkdir(parents=True, exist_ok=False)
    return path


def token_hash(tokens):
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack('<I', token))
    return digest.hexdigest()


def stat_identity(path):
    path = Path(path)
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return {'path': str(path), 'exists': False}
    stat = resolved.stat()
    return {'path': str(resolved), 'exists': True, 'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns,
            'inode': stat.st_ino}


def require(condition, message):
    if not condition:
        raise MatrixError(message)


def load_runtime(repo):
    """Import the approved gate/helper modules; never copy their code."""
    global _GATE, _HELPER
    repo = Path(repo).resolve()
    if _GATE is None:
        gate_path = repo / GATE_REL
        helper_path = repo / HELPER_REL
        if not gate_path.is_file() or not helper_path.is_file():
            raise RefuseError(f'gate/helper not found under {repo}')
        spec = importlib.util.spec_from_file_location('rkva_raw_matrix_gate', gate_path)
        if spec is None or spec.loader is None:
            raise RefuseError(f'cannot import {gate_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _GATE = module
        _HELPER = module.load_helper(repo)
    return _GATE, _HELPER


# ---------------------------------------------------------------------------
# Case catalog
# ---------------------------------------------------------------------------

def _json_prompt_digest(tokens):
    return sha256_text(json.dumps(tokens, separators=(',', ':')))


def _case_rows(cases_dir):
    cases_dir = Path(cases_dir)
    if not cases_dir.is_dir():
        raise RefuseError(f'cases directory not found: {cases_dir}')
    catalog = cases_dir / 'catalog.json'
    rows = []
    if catalog.is_file():
        document = read_json(catalog)
        items = document.get('cases') if isinstance(document, dict) else document
        if not isinstance(items, list) or not items:
            raise ValueError('catalog.json needs a non-empty cases array')
        for item in items:
            if not isinstance(item, dict):
                raise ValueError('catalog.json cases must be objects')
            row = dict(item)
            row['_dir'] = cases_dir
            rows.append(row)
        return rows
    for entry in sorted(cases_dir.iterdir()):
        if entry.is_dir():
            case_json = entry / 'case.json'
            prompt_json = entry / 'prompt.json'
            tokens_json = entry / 'tokens.json'
            if case_json.is_file():
                row = read_json(case_json)
                row['_dir'] = entry
                rows.append(row)
            elif prompt_json.is_file():
                rows.append({'name': entry.name, 'prompt_json': str(prompt_json),
                             '_dir': entry})
            elif tokens_json.is_file():
                row = read_json(tokens_json)
                row['_dir'] = entry
                rows.append(row)
        elif entry.suffix == '.json' and entry.name != 'catalog.json':
            row = read_json(entry)
            row['_dir'] = entry.parent
            row.setdefault('name', entry.stem)
            rows.append(row)
    if not rows:
        raise ValueError(f'no case JSON found under {cases_dir}')
    return rows


def load_cases(cases_dir, steps, prefix_sha, base_tokens, p=P_DEFAULT,
               context=CONTEXT_DEFAULT):
    rows = _case_rows(cases_dir)
    cases, names = [], set()
    for index, row in enumerate(rows):
        name = row.get('name')
        if not isinstance(name, str) or not CASE_NAME_RE.match(name):
            raise ValueError(f'case[{index}] has an invalid name: {name!r}')
        if name in names:
            raise ValueError(f'duplicate case name: {name}')
        names.add(name)
        tokens = row.get('tokens')
        if row.get('prompt_json'):
            prompt_path = Path(row['prompt_json'])
            if not prompt_path.is_absolute():
                prompt_path = Path(row['_dir']) / prompt_path
            document = read_json(prompt_path)
            if isinstance(document, dict):
                document = document.get('prompt') or document.get('tokens')
            tokens = document
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f'{name}: needs a non-empty token array')
        if any(type(token) is not int or not 0 <= token < 2 ** 31 for token in tokens):
            raise ValueError(f'{name}: tokens must be nonnegative int32 ids')
        if row.get('vocab') not in (None, VOCAB_DEFAULT):
            raise ValueError(f'{name}: case vocab {row.get("vocab")} != {VOCAB_DEFAULT}')
        base_length = row.get('base_length')
        if base_length is not None:
            if base_length != p:
                raise ValueError(f'{name}: base_length {base_length} != {p}')
            if base_tokens is None:
                raise RefuseError(f'{name}: tail-only case needs the reference base token stream')
            if row.get('length') is not None and row['length'] != p + len(tokens):
                raise ValueError(f'{name}: length does not equal P + tail')
            if row.get('tokens_sha256') and row['tokens_sha256'] != token_hash(tokens):
                raise ValueError(f'{name}: tail token hash mismatch')
            prompt = list(base_tokens) + list(tokens)
        else:
            prompt = list(tokens)
            if row.get('length') is not None and row['length'] != len(prompt):
                raise ValueError(f'{name}: length does not equal the token array')
            if row.get('tokens_sha256') and row['tokens_sha256'] != token_hash(prompt):
                raise ValueError(f'{name}: full token hash mismatch')
        length = len(prompt)
        if length % 512:
            raise ValueError(f'{name}: prompt length {length} is not a multiple of 512')
        if length < p + 512:
            raise ValueError(f'{name}: prompt length {length} is shorter than P+512')
        if length + steps > context:
            raise ValueError(
                f'{name}: length {length} + teacher steps {steps} exceeds context {context}')
        actual_prefix = token_hash(prompt[:p])
        if actual_prefix != prefix_sha:
            raise RefuseError(
                f'{name}: first P tokens hash {actual_prefix} != reference prefix {prefix_sha}')
        if row.get('prompt_sha256') and row['prompt_sha256'] != _json_prompt_digest(prompt):
            raise ValueError(f'{name}: prompt_json digest mismatch')
        if row.get('prefix_sha256') and row['prefix_sha256'] != _json_prompt_digest(prompt[:p]):
            raise ValueError(f'{name}: prefix_json digest mismatch')
        answer = row.get('answer_ids')
        if answer is not None and (not isinstance(answer, list) or any(
                type(token) is not int or not 0 <= token < 2 ** 31 for token in answer)):
            raise ValueError(f'{name}: answer_ids must be nonnegative int ids')
        reference_name = row.get('reference') or name
        if not isinstance(reference_name, str) or not CASE_NAME_RE.match(reference_name):
            raise ValueError(f'{name}: invalid reference case name')
        source = row.get('source') or (
            'reference-tail-plus-base' if base_length is not None
            else 'explicit-catalog-token-array')
        cases.append({
            'name': name,
            'category': str(row.get('category') or 'synthetic'),
            'length': length,
            'tail_length': length - p,
            'prompt': prompt,
            'prompt_sha256': token_hash(prompt),
            'prefix_sha256': actual_prefix,
            'tokens_sha256': token_hash(prompt),
            'source': source,
            'reference': reference_name,
            'answer_tokens': answer,
            'row': {key: value for key, value in row.items() if key != '_dir'},
        })
    return cases


# ---------------------------------------------------------------------------
# Frozen reference campaign
# ---------------------------------------------------------------------------

def _pick_environment(reference_root):
    tokenizer = reference_root / 'tokenizer' / 'manifest.json'
    if tokenizer.is_file():
        document = read_json(tokenizer)
        environment = document.get('environment')
        if isinstance(environment, dict) and environment:
            return tokenizer, document, environment
    for manifest in sorted((reference_root / 'cases').glob('*/off/manifest.json')):
        document = read_json(manifest)
        environment = document.get('environment')
        if isinstance(environment, dict) and environment:
            return tokenizer if tokenizer.is_file() else manifest, document, environment
    raise RefuseError(f'{reference_root}: no frozen reference environment found')


def load_reference(reference_root, gate, helper):
    root = Path(reference_root).resolve()
    if not root.is_dir():
        raise RefuseError(f'reference campaign not found: {root}')
    binding_path = root / 'binding.json'
    preflight_path = root / 'preflight.json'
    binding = read_json(binding_path) if binding_path.is_file() else None
    preflight = read_json(preflight_path) if preflight_path.is_file() else None
    if binding is None:
        if not preflight or not isinstance(preflight.get('binding'), dict):
            raise RefuseError(f'{root}: reference binding/preflight is missing')
        binding = preflight['binding']
    for key in ('model', 'prefix'):
        if not isinstance(binding.get(key), dict):
            raise RefuseError(f'{root}: reference binding has no {key}')
    environment_path, environment_document, environment = _pick_environment(root)
    if any(not isinstance(value, str) for value in environment.values()):
        raise RefuseError(f'{root}: reference environment has non-string values')
    helper_args = {}
    if isinstance(environment_document, dict):
        helper_args = environment_document.get('helper_args') or {}
    base_tokens = None
    base_info = None
    token_path = helper_args.get('prompt_tokens')
    if token_path:
        try:
            stream, base_info = helper.effective_tokens(str(token_path), P_DEFAULT,
                                                        repeat=False)
        except (OSError, ValueError) as error:
            raise RefuseError(f'{root}: cannot read reference base tokens: {error}') from error
        base_tokens = list(stream[:P_DEFAULT])
    prefix_path = binding.get('prefix_path')
    if not prefix_path:
        prefix = (preflight or {}).get('prefix') or {}
        prefix_path = prefix.get('path')
    if not prefix_path:
        raise RefuseError(f'{root}: reference prefix path is missing')
    reference_gates = binding.get('gates') or (preflight or {}).get('gates')
    defaults = dict(gate.GATE_DEFAULTS)
    if reference_gates is not None and canonical(reference_gates) != canonical(defaults):
        raise RefuseError(
            f'{root}: reference gates differ from the frozen gate defaults; refusing to relax')
    return {
        'root': root,
        'binding': binding,
        'preflight': preflight or {},
        'environment': dict(environment),
        'environment_path': str(environment_path),
        'helper_args': helper_args,
        'base_tokens': base_tokens,
        'base_info': base_info,
        'model': binding['model'],
        'prefix_path': Path(prefix_path),
        'binding_sha256': binding.get('binding_sha256'),
        'source': binding.get('source'),
        'gates': defaults,
        'libraries': binding.get('libraries') or {},
    }


def verify_common_args(path, reference, xbox_host, output=None):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    try:
        argv = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f'common-args-json is not valid UTF-8 JSON: {error}') from error
    if not isinstance(argv, list) or len(argv) < 2 or any(
            not isinstance(item, str) for item in argv):
        raise ValueError('common-args-json must be a JSON array of strings')
    forbidden = {'--help', '-h', '--usage', '--version', '--list-devices', '--completion'}
    if any(item in forbidden for item in argv):
        raise ValueError('common args must not request help/version/device listing/completion')

    def value(flag):
        if flag in argv:
            index = argv.index(flag)
            if index + 1 >= len(argv):
                raise ValueError(f'{flag} has no value')
            return argv[index + 1]
        raise ValueError(f'common args must set {flag}')

    def require_value(flag, expected):
        actual = value(flag)
        if actual != expected:
            raise ValueError(f'common args require {flag} {expected}, got {actual}')

    model = value('-m') if '-m' in argv else value('--model')
    model_path = Path(model).resolve(strict=True)
    reference_model = Path(reference['model']['path']).resolve(strict=True)
    if model_path != reference_model:
        raise RefuseError(f'common args model {model_path} != reference model {reference_model}')
    for flag, expected in (
            ('--ctx-size', str(CONTEXT_DEFAULT)), ('--batch-size', '1024'),
            ('--ubatch-size', '512'), ('--parallel', '1'), ('--gpu-layers', '99'),
            ('--device', 'CUDA0'), ('--flash-attn', 'on'), ('--fit', 'off'),
            ('--cache-type-k', 'kvarn4'), ('--cache-type-v', 'kvarn4'),
            ('--kv-position-split-p', str(P_DEFAULT)),
            ('--remote-attn-layers', 'full'),
            ('--remote-attn-cache-type-k', 'q4_0'),
            ('--remote-attn-cache-type-v', 'q4_0'),
            ('--remote-attn-format', 'kvarn4'),
            ('--remote-attn-wire-format', 'fp8'),
            ('--remote-attn', xbox_host), ('--spec-type', 'none'),
            ('--spec-draft-n-max', '0')):
        require_value(flag, expected)
    if '--no-context-shift' not in argv:
        raise ValueError('common args must include --no-context-shift')
    if '--no-warmup' not in argv:
        raise ValueError('common args must include --no-warmup')
    frozen_path = (Path(output) / 'inputs' / 'common-args.json') if output else None
    return {'path': str(path), 'argv': argv, 'bytes': raw,
            'sha256': sha256_bytes(raw), 'stat': stat_identity(path),
            'model_path': str(model_path),
            'frozen_path': str(frozen_path) if frozen_path else None,
            'frozen_sha256': sha256_bytes(raw)}


# ---------------------------------------------------------------------------
# Prefix snapshot
# ---------------------------------------------------------------------------

def load_prefix(reference, base_tokens, gate, helper):
    """Reuse the approved FULL prefix validator, then freeze the HTTP identity.

    The helper validates the complete contract (schema/codec/model/P/context/
    slot/batch/flash/RoPE-format fields/environment) and re-hashes every file.
    The extra checks bind the resulting metadata SHA and file list to the
    original HTTP campaign, so a re-hashed forgery is still refused.
    """
    if base_tokens is None:
        raise RefuseError('prefix validation needs the exact reference base tokens')
    path = Path(reference['prefix_path'])
    if path.is_symlink():
        raise RefuseError('prefix snapshot main file must not be a symlink')
    meta_path = Path(str(path) + '.meta.json')
    if not path.is_file() or not meta_path.is_file():
        raise RefuseError(f'prefix snapshot or metadata missing: {path}')
    helper_args = SimpleNamespace(
        prefix_snapshot=str(path), p=P_DEFAULT, context=CONTEXT_DEFAULT,
        batch=1024, ubatch=512, frozen_prefix_batch=512,
        cuda_fp8='mapped', range_kernel='fine', wavefront_tile=256,
        n256_lookahead=False, tcp_prefetch=True, local_input_reuse=False,
        mask_pack='default')
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            metadata = helper.load_prefix_snapshot(
                helper_args, list(base_tokens)[:P_DEFAULT],
                {'sha256': reference['model'].get('sha256')})
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise RefuseError(f'prefix snapshot FAILED the frozen contract: {error}') from error
    if metadata is None:
        raise RefuseError('prefix snapshot is absent; context conversion is refused')
    meta_sha = sha256_file(meta_path)
    preflight = reference.get('preflight') or {}
    expected_meta = ((preflight.get('prefix_contract') or {}).get('meta_sha256')
                     or (reference['binding'].get('prefix') or {}).get('meta_sha256'))
    if expected_meta is not None and meta_sha != expected_meta:
        raise RefuseError('prefix metadata SHA differs from the original HTTP binding')
    reference_prefix = reference['binding'].get('prefix') or {}
    for field in ('actual_prefix_tokens', 'actual_prefix_sha256'):
        if field in reference_prefix and metadata.get(field) != reference_prefix[field]:
            raise RefuseError(f'prefix {field} differs from the original HTTP binding')
    if 'files' in reference_prefix and metadata.get('files') != reference_prefix['files']:
        raise RefuseError('prefix file list/details differ from the original HTTP binding')
    files = metadata.get('files') or {}
    records = {}
    for name, expected in files.items():
        item = path.parent / name
        if not item.is_file() or item.is_symlink():
            raise RefuseError(f'prefix snapshot file missing/unsafe: {item}')
        stat = item.stat()
        records[name] = {'size': stat.st_size, 'sha256': expected.get('sha256'),
                         'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino}
    return {
        'path': str(path), 'meta_path': str(meta_path),
        'meta_sha256': meta_sha, 'metadata': metadata,
        'contract': metadata.get('contract') or {}, 'files': records,
        'main_size': records[path.name]['size'], 'verify_sha': True,
    }

def verify_prefix_unchanged(prefix):
    for name, expected in prefix['files'].items():
        item = Path(prefix['path']).parent / name
        if not item.is_file():
            raise MatrixError(f'prefix snapshot file disappeared: {name}')
        stat = item.stat()
        if (stat.st_size != expected['size']
                or stat.st_mtime_ns != expected['mtime_ns']
                or stat.st_ino != expected['inode']):
            raise MatrixError(f'prefix snapshot changed during the run: {name}')


# ---------------------------------------------------------------------------
# Environment and library binding
# ---------------------------------------------------------------------------

def frozen_environment(reference_environment, arm_dir):
    arm_dir = Path(arm_dir)
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(ENV_PREFIXES + ('F4_', 'F5_'))
        or key == 'CUDA_VISIBLE_DEVICES'
    }
    environment.update({key: value for key, value in reference_environment.items()
                        if isinstance(value, str)})
    for key, suffix in (('TMPDIR', 'tmp'), ('CUDA_CACHE_PATH', 'cuda-cache'),
                        ('XDG_CACHE_HOME', 'cache')):
        folder = arm_dir / suffix
        folder.mkdir(parents=True, exist_ok=True)
        environment[key] = str(folder)
    if environment.get('GGML_RKVA_PROFILE') != '1':
        raise RefuseError('reference environment must set GGML_RKVA_PROFILE=1')
    if MASK_PACK in environment:
        raise RefuseError(f'reference environment must not set {MASK_PACK}')
    return environment


def active_libraries_for(binary, environment):
    try:
        result = subprocess.run(['ldd', str(binary)], capture_output=True, text=True,
                                env=environment, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RefuseError(f'cannot resolve capture libraries with ldd: {error}') from error
    paths = set()
    for line in (result.stdout or '').splitlines():
        if 'not found' in line and re.search(r'(llama|ggml)', line, re.I):
            raise RefuseError(f'missing capture shared library: {line.strip()}')
        match = re.search(r'=>\s+(\S+)', line)
        if match is None:
            match = re.match(r'\s*(\S+)\s+\(0x', line)
        if match is None:
            continue
        path = match.group(1)
        if not re.search(r'(llama|ggml)', Path(path).name, re.I):
            continue
        try:
            resolved = Path(path).resolve(strict=True)
        except OSError as error:
            raise RefuseError(f'cannot resolve capture library {path}') from error
        paths.add(resolved)
    if not paths:
        raise RefuseError('ldd found no llama/ggml shared libraries')
    return sorted(paths)


def build_binding(args, reference, common, prefix, cases, gate, helper, libraries):
    source = helper.git_identity()
    binary_stat = stat_identity(args.capture_binary)
    binding = {
        'schema': SCHEMA,
        'tool': {'path': str(Path(__file__).resolve()),
                 'sha256': sha256_file(Path(__file__).resolve())},
        'source': source,
        'capture_binary': {
            'path': binary_stat['path'], 'size': binary_stat['size'],
            'mtime_ns': binary_stat['mtime_ns'], 'ctime_ns': binary_stat['ctime_ns'],
            'inode': binary_stat['inode'], 'sha256': sha256_file(args.capture_binary),
        },
        'common_args': {
            'path': common['path'], 'sha256': common['sha256'],
            'stat': common.get('stat'),
            'frozen_path': common.get('frozen_path'),
            'frozen_sha256': common.get('frozen_sha256', common['sha256']),
        },
        'model': dict(reference['model'], stat=stat_identity(reference['model']['path'])),
        'reference': {
            'root': str(reference['root']),
            'binding_sha256': reference['binding_sha256'],
            'source': reference['source'],
            'environment_path': reference['environment_path'],
        },
        'prefix': {
            'path': prefix['path'], 'meta_sha256': prefix['meta_sha256'],
            'contract': prefix['contract'], 'actual_prefix_sha256':
                prefix['metadata'].get('actual_prefix_sha256'),
            'files': prefix['files'],
        },
        'environment': {
            'sha256': sha256_text(canonical(reference['environment'])),
            'keys': sorted(reference['environment']),
            'profile': 'GGML_RKVA_PROFILE=1', 'mask_pack': 'absent',
        },
        'libraries': {'files': libraries['files'], 'sha256': libraries['sha256']},
        'gates': dict(gate.GATE_DEFAULTS),
        'cases': [{'name': case['name'], 'length': case['length'],
                   'prompt_sha256': case['prompt_sha256'],
                   'prefix_sha256': case['prefix_sha256']} for case in cases],
        'teacher_steps': args.teacher_steps,
        'teacher_mode': args.teacher_mode,
        'determinism': args.determinism,
        'geometry': {'P': P_DEFAULT, 'context': CONTEXT_DEFAULT, 'batch': 1024,
                     'ubatch': 512, 'vocab': VOCAB_DEFAULT},
    }
    binding['binding_sha256'] = sha256_text(canonical(binding))
    return binding


class BindingGuard:
    def __init__(self, binding, binary, environment, libraries, helper,
                 path_provider=None):
        self.binding = binding
        self.binary = Path(binary)
        self.environment = environment
        self.helper = helper
        self.libraries_binding = libraries
        self.guard = _GATE.LibraryGuard(
            str(self.binary), libraries,
            path_provider=path_provider or (
                lambda path: active_libraries_for(path, environment)))
        self.source = helper.git_identity()

    def _verify_stat(self, phase, label, path, expected):
        stat = stat_identity(path)
        if not stat.get('exists'):
            raise MatrixError(f'{phase}: {label} disappeared')
        for key in ('size', 'mtime_ns', 'ctime_ns', 'inode'):
            if expected.get(key) is not None and stat.get(key) != expected.get(key):
                raise MatrixError(f'{phase}: {label} {key} changed')

    def verify(self, phase, loaded=None):
        expected = self.binding['capture_binary']
        stat = self.binary.stat()
        if stat.st_size != expected['size']:
            raise MatrixError(f'{phase}: capture binary size changed')
        if sha256_file(self.binary) != expected['sha256']:
            raise MatrixError(f'{phase}: capture binary content changed')
        source = self.helper.git_identity()
        for key in ('head', 'tracked_diff_sha256'):
            if source.get(key) != self.binding['source'].get(key):
                raise MatrixError(f'{phase}: repository {key} changed during the run')
        model = self.binding.get('model') or {}
        if model.get('path'):
            self._verify_stat(phase, 'model', model['path'], model.get('stat') or {})
            if model.get('sha256') is None:
                raise MatrixError(f'{phase}: model SHA was not frozen')
            if sha256_file(model['path']) != model['sha256']:
                raise MatrixError(f'{phase}: model content SHA changed')
        common = self.binding.get('common_args') or {}
        if common.get('path'):
            self._verify_stat(phase, 'common args', common['path'],
                              common.get('stat') or {})
            if sha256_file(common['path']) != common.get('sha256'):
                raise MatrixError(f'{phase}: common args content changed')
        if common.get('frozen_path'):
            frozen = Path(common['frozen_path'])
            if not frozen.is_file():
                raise MatrixError(f'{phase}: frozen common args copy is missing')
            if sha256_file(frozen) != common.get('frozen_sha256'):
                raise MatrixError(f'{phase}: frozen common args copy changed')
        try:
            self.guard.verify_local(phase)
            if loaded is not None:
                if not loaded:
                    raise MatrixError(f'{phase}: loaded library map is empty')
                self.guard.verify_loaded(loaded, phase)
                self.guard.require_loaded(loaded, phase)
                relevant = {str(Path(item).resolve()) for item in loaded
                            if re.search(r'(llama|ggml)', Path(item).name, re.I)}
                missing = set(self.guard.files) - relevant
                if missing:
                    raise MatrixError(
                        f'{phase}: expected loaded libraries missing: '
                        f'{sorted(missing)[:3]}')
        except MatrixError:
            raise
        except Exception as error:
            raise MatrixError(f'{phase}: {error}') from error
        if self.binding['prefix'].get('path'):
            verify_prefix_unchanged({'path': self.binding['prefix']['path'],
                                     'files': self.binding['prefix']['files']})

# ---------------------------------------------------------------------------
# Raw logits / metrics
# ---------------------------------------------------------------------------

def read_logits(path, vocab=VOCAB_DEFAULT):
    path = Path(path)
    if not path.is_file():
        raise MatrixError(f'raw logits file missing: {path}')
    size = path.stat().st_size
    if size != 4 * vocab:
        raise MatrixError(
            f'{path}: {size} bytes, expected exactly {4 * vocab} for {vocab} logits')
    values = array.array('f')
    try:
        with open(path, 'rb') as source:
            values.fromfile(source, vocab)
    except (EOFError, ValueError) as error:
        raise MatrixError(f'{path}: cannot read {vocab} float32 logits') from error
    if len(values) != vocab:
        raise MatrixError(f'{path}: short float32 array')
    if sys.byteorder != 'little':
        values.byteswap()
    return values


def raw_stats(path, vocab=VOCAB_DEFAULT, mass_tolerance=1e-3):
    values = read_logits(path, vocab)
    maximum = -math.inf
    second = -math.inf
    top1 = top2 = 0
    for token, value in enumerate(values):
        if not math.isfinite(value):
            raise MatrixError(f'{path}: non-finite logit at id {token}: {value!r}')
        if value > maximum:
            second, top2 = maximum, top1
            maximum, top1 = value, token
        elif value > second:
            second, top2 = value, token
    terms = array.array('d', (math.exp(value - maximum) for value in values))
    total = math.fsum(terms)
    if not math.isfinite(total) or total <= 0:
        raise MatrixError(f'{path}: invalid log-sum-exp total {total!r}')
    log_total = math.log(total)
    lse = maximum + log_total
    mass = math.fsum(term / total for term in terms)
    if not math.isfinite(mass) or abs(mass - 1.0) > mass_tolerance:
        raise MatrixError(f'{path}: probability mass {mass!r} is not close to one')
    return {
        'values': values, 'terms': terms, 'total': total, 'maximum': maximum,
        'log_total': log_total, 'lse': lse,
        'top1': top1, 'top2': top2, 'mass': mass,
        'prob_top1': 1.0 / total,
        'prob_top2': (terms[top2] / total) if second > -math.inf else 0.0,
        'margin': (1.0 / total) - ((terms[top2] / total) if second > -math.inf else 0.0),
    }


def step_metrics_from_stats(off, on, teacher, high_confidence_margin):
    vocabulary = len(off['values'])
    if type(teacher) is not int or not 0 <= teacher < vocabulary:
        raise MatrixError(f'teacher id {teacher!r} is outside vocabulary {vocabulary}')
    # Keep the shift and log(total) separate.  Folding them into
    # maximum+log(total) loses log(total) when the logits are ~1e20 (or more),
    # which would turn NLL/KL into a spurious zero.
    nll_off = -((off['values'][teacher] - off['maximum']) - off['log_total'])
    nll_on = -((on['values'][teacher] - on['maximum']) - on['log_total'])
    # Exact full-vocabulary KL in the shifted log domain.  No top-k truncation:
    # every id contributes off_probability * (log p_off - log p_on).
    kl = math.fsum(
        off['terms'][index] * (((off['values'][index] - off['maximum'])
                                - off['log_total'])
                               - ((on['values'][index] - on['maximum'])
                                  - on['log_total']))
        for index in range(vocabulary)) / off['total']
    if not all(math.isfinite(value) for value in (nll_off, nll_on, kl)):
        raise MatrixError('non-finite NLL/KL')
    agreed = off['top1'] == on['top1']
    high_confidence = (not agreed
                       and off['margin'] >= high_confidence_margin
                       and on['margin'] >= high_confidence_margin)
    return {
        'teacher_id': teacher,
        'nll_off': nll_off, 'nll_on': nll_on, 'nll_delta': nll_on - nll_off,
        'kl': kl,
        'top1_off': off['top1'], 'top1_on': on['top1'],
        'top1_prob_off': off['prob_top1'], 'top1_prob_on': on['prob_top1'],
        'margin_off': off['margin'], 'margin_on': on['margin'],
        'top1_agree': agreed, 'high_confidence_divergence': high_confidence,
        'mass_off': off['mass'], 'mass_on': on['mass'],
    }


def evaluate_window(name, steps, gates, gate):
    try:
        summary = gate.summarize_steps(steps)
    except (OverflowError, ValueError) as error:
        return {'window': name, 'steps': len(steps), 'metrics': None,
                'checks': {'finite': False}, 'pass': False,
                'failures': [f'metric error: {type(error).__name__}']}
    checks = {}
    finite = bool(steps) and summary.get('finite') is True
    checks['finite'] = finite
    if finite:
        checks['ppl_ratio'] = (summary.get('ppl_ratio') is not None
                               and summary['ppl_ratio'] <= gates['max_ppl_ratio'])
        checks['kl_mean'] = (summary.get('kl_mean') is not None
                             and summary['kl_mean'] <= gates['max_kl_mean'])
        checks['kl_p99'] = (summary.get('kl_p99') is not None
                            and summary['kl_p99'] <= gates['max_kl_p99'])
        checks['nll_delta'] = (summary.get('nll_delta') is not None
                               and summary['nll_delta'] <= gates['max_case_nll_delta'])
    else:
        for key in ('ppl_ratio', 'kl_mean', 'kl_p99', 'nll_delta'):
            checks[key] = False
    failures = [key for key, value in checks.items() if not value]
    return {'window': name, 'steps': len(steps), 'metrics': summary,
            'checks': checks, 'pass': not failures,
            'failures': failures}


# ---------------------------------------------------------------------------
# Capture child
# ---------------------------------------------------------------------------

def _terminate_child(process, term_timeout=60, kill_timeout=15):
    """Terminate, wait, kill if needed and report every cleanup failure.

    This function never claims success by swallowing a failed reap.  The
    caller decides whether the child is still alive, but it must treat any
    error as fatal for the arm.
    """
    details = {
        'pid': getattr(process, 'pid', None), 'alive': False, 'reaped': False,
        'terminated': False, 'killed': False, 'terminate_error': None,
        'kill_error': None, 'reap_error': None,
    }
    if process is None:
        details['reaped'] = True
        return details
    try:
        if process.poll() is not None:
            details['reaped'] = True
            return details
    except Exception as error:
        details['reap_error'] = f'initial poll failed: {error}'
    try:
        process.terminate()
        details['terminated'] = True
    except ProcessLookupError:
        pass
    except Exception as error:
        details['terminate_error'] = str(error)
    try:
        process.wait(timeout=term_timeout)
        details['reaped'] = True
    except subprocess.TimeoutExpired:
        details['reap_error'] = 'SIGTERM wait timed out'
        try:
            process.kill()
            details['killed'] = True
        except ProcessLookupError:
            pass
        except Exception as error:
            details['kill_error'] = str(error)
        try:
            process.wait(timeout=kill_timeout)
            details['reaped'] = True
        except subprocess.TimeoutExpired:
            details['reap_error'] = 'still alive after SIGKILL wait'
        except Exception as error:
            details['reap_error'] = f'SIGKILL wait failed: {error}'
    except Exception as error:
        details['reap_error'] = f'SIGTERM wait failed: {error}'
    try:
        details['alive'] = process.poll() is None
    except Exception as error:
        details['alive'] = True
        details['reap_error'] = (details['reap_error'] or '') + f'; final poll failed: {error}'
    return details


def _cleanup_failure(details, cause):
    errors = [value for key, value in details.items()
              if key in ('terminate_error', 'kill_error', 'reap_error') and value]
    if errors or details['alive'] or not details['reaped']:
        return ChildCleanupError(
            f'owned capture child cleanup failed (pid={details.get("pid")}, '
            f'alive={details.get("alive")}): {" | ".join(errors) or cause}',
            details.get('pid'), details.get('alive'), details)
    return None


def _record_cleanup_failure(log_path, details):
    try:
        write_json_new(Path(log_path).parent / 'cleanup-failure.json', {
            'owner': TOOL, 'at': now(), 'pid': details.get('pid'),
            'alive': details.get('alive'), 'terminated': details.get('terminated'),
            'killed': details.get('killed'), 'reaped': details.get('reaped'),
            'terminate_error': details.get('terminate_error'),
            'kill_error': details.get('kill_error'),
            'reap_error': details.get('reap_error'),
        })
    except BaseException:
        pass


def run_capture_child(command, environment, log_path, timeout, cwd,
                      popen_factory=None):
    """Tee the child log, sample libraries on the real startup marker.

    Reading is event-driven: a reader thread copies each line to the owned log
    and samples /proc/<pid>/maps exactly when the context/backend marker (or
    the step-0 fallback) arrives.  There is no RPC and no polling loop.
    """
    factory = popen_factory or subprocess.Popen
    log_path = Path(log_path)
    state = {'startup_event': None, 'startup_fallback': False, 'loaded': None,
             'loaded_error': None, 'reader_error': None}
    process = None
    reader = None
    cleaned = False

    def sample_maps():
        try:
            loaded = _GATE.loaded_library_paths(process.pid)
        except Exception as error:
            state['loaded_error'] = str(error)
            return
        if loaded:
            state['loaded'] = loaded
        else:
            state['loaded_error'] = 'empty library map'

    def pump():
        try:
            if process.stdout is None:
                raise RuntimeError('child stdout pipe is missing')
            for line in iter(process.stdout.readline, ''):
                log.write(line)
                log.flush()
                if STARTUP_MARKER in line and state['startup_event'] is None:
                    state['startup_event'] = line.rstrip()
                    sample_maps()
                elif STEP0_MARKER in line and state['loaded'] is None:
                    state['startup_fallback'] = True
                    if state['startup_event'] is None:
                        state['startup_event'] = line.rstrip()
                    sample_maps()
        except BaseException as error:
            state['reader_error'] = f'{type(error).__name__}: {error}'
        finally:
            try:
                process.stdout.close()
            except Exception:
                pass

    with open(log_path, 'w', encoding='utf-8', errors='replace') as log:
        process = factory(
            command, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            cwd=str(cwd), start_new_session=True)
        reader = threading.Thread(target=pump, name='rkva-capture-reader', daemon=True)
        reader.start()
        try:
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                cleanup = _terminate_child(process)
                cleaned = True
                _record_cleanup_failure(log_path, cleanup)
                fatal = _cleanup_failure(cleanup, 'capture child timed out')
                if fatal is not None:
                    raise fatal from error
                raise MatrixError(f'capture child timed out after {timeout}s') from error
            except BaseException as original:
                cleanup = _terminate_child(process)
                cleaned = True
                _record_cleanup_failure(log_path, cleanup)
                fatal = _cleanup_failure(cleanup, 'capture interrupted')
                if fatal is not None:
                    raise fatal from original
                raise
        finally:
            if not cleaned and process.poll() is None:
                cleanup = _terminate_child(process)
                _record_cleanup_failure(log_path, cleanup)
                fatal = _cleanup_failure(cleanup, 'capture cleanup at exit')
                if fatal is not None:
                    raise fatal
            if reader is not None:
                reader.join(timeout=30)
                if reader.is_alive():
                    state['reader_error'] = 'log reader did not finish after child exit'
    if state['reader_error']:
        raise MatrixError(f'capture log reader failed: {state["reader_error"]}')
    return {
        'returncode': returncode, 'loaded_startup': state['loaded'],
        'loaded_error': state['loaded_error'], 'startup_event': state['startup_event'],
        'startup_fallback': state['startup_fallback'],
        'log_path': str(log_path), 'command': list(map(str, command)),
    }

def read_log_lines(path):
    return Path(path).read_text(errors='replace').splitlines()


def validate_capture_manifest(raw_dir, spec, supplied_teacher, steps, prefix,
                              environment, common, xbox_host, gate,
                              vocab=VOCAB_DEFAULT, p=P_DEFAULT,
                              context=CONTEXT_DEFAULT):
    raw_dir = Path(raw_dir)
    manifest_path = raw_dir / 'manifest.json'
    if not manifest_path.is_file():
        raise MatrixError(f'{raw_dir}: capture manifest is missing')
    manifest = read_json(manifest_path)
    require(manifest.get('schema') == 'rkva-quality-raw-capture-v1',
            f'{raw_dir}: unexpected manifest schema')
    require(manifest.get('result') == 'PASS', f'{raw_dir}: capture result is not PASS')
    require(manifest.get('label') == 'RAW_CAPTURE_ONLY',
            f'{raw_dir}: capture label is not RAW_CAPTURE_ONLY')
    finalized = manifest.get('finalized') or {}
    for key in ('context_destroyed', 'profile_flushed', 'backend_freed',
                'manifest_after_context_destroy'):
        require(finalized.get(key) is True, f'{raw_dir}: manifest not finalized ({key})')
    capture = manifest.get('capture') or {}
    for key, expected in (('vocab', vocab), ('steps', steps), ('prefix_tokens', p),
                          ('prompt_tokens', spec['length']),
                          ('tail_tokens', spec['tail_length']),
                          ('step_logits_bytes', 4 * vocab), ('all_logits', True),
                          ('finite', True), ('source', 'capi_llama_decode')):
        require(capture.get(key) == expected,
                f'{raw_dir}: capture.{key}={capture.get(key)!r} != {expected!r}')
    contract = capture.get('logits_contract') or {}
    require(contract.get('sampled_logits_active') is False,
            f'{raw_dir}: sampled logits were active')
    require(contract.get('sampled_logits_count') == vocab,
            f'{raw_dir}: dense logits count mismatch')
    require(contract.get('sampled_candidates_count') == 0,
            f'{raw_dir}: sampled candidates were not empty')
    require(contract.get('outputs_requested') == 'last-row-only',
            f'{raw_dir}: unexpected output row request')
    require(manifest.get('prompt_sha256') == token_hash(spec['prompt']),
            f'{raw_dir}: prompt hash mismatch')
    step_meta = manifest.get('steps')
    require(isinstance(step_meta, list) and len(step_meta) == steps,
            f'{raw_dir}: step list is incomplete')
    observed = [item.get('teacher_id') for item in step_meta]
    require(all(type(token) is int and 0 <= token < vocab for token in observed),
            f'{raw_dir}: step teacher id is invalid')
    if supplied_teacher is None:
        teacher_ids = observed
        require(manifest.get('teacher', {}).get('source') == 'argmax',
                f'{raw_dir}: OFF arm did not use greedy argmax')
        require(manifest.get('teacher', {}).get('ids') == [],
                f'{raw_dir}: greedy arm unexpectedly supplied teacher ids')
    else:
        teacher_ids = list(supplied_teacher)
        require(len(teacher_ids) == steps, f'{raw_dir}: teacher sequence length mismatch')
        require(observed == teacher_ids, f'{raw_dir}: step teacher ids differ from supplied')
        require(manifest.get('teacher', {}).get('source') == 'fixed-file',
                f'{raw_dir}: teacher source is not fixed-file')
        require(manifest.get('teacher', {}).get('ids') == teacher_ids,
                f'{raw_dir}: manifest teacher ids differ from supplied')
    expected_hashes = [token_hash(spec['prompt'] + teacher_ids[:step])
                       for step in range(steps)]
    require(manifest.get('teacher', {}).get('prompt_hashes') == expected_hashes,
            f'{raw_dir}: teacher prompt hashes differ from the exact history')
    cadence = manifest.get('cadence') or {}
    require(cadence.get('p') == p and cadence.get('length') == spec['length'],
            f'{raw_dir}: cadence geometry mismatch')
    tail = spec['tail_length']
    covered = (cadence.get('ubatch_512', 0) * 512 + cadence.get('seam_256', 0) * 256
               + cadence.get('seam_128', 0) * 128)
    require(covered == tail, f'{raw_dir}: cadence covers {covered} != tail {tail}')
    plan = cadence.get('plan')
    require(isinstance(plan, list) and plan and sum(item.get('count', 0) for item in plan) == tail,
            f'{raw_dir}: cadence plan does not cover the tail')
    require(sum(1 for item in plan if item.get('want_logits') is True) == 1
            and plan[-1].get('want_logits') is True,
            f'{raw_dir}: cadence must request logits only on the final row')
    restored = manifest.get('prefix_restore') or {}
    require(restored.get('file') == prefix['path'], f'{raw_dir}: prefix path mismatch')
    require(restored.get('file_size') == prefix['main_size'],
            f'{raw_dir}: prefix snapshot size mismatch')
    require(restored.get('n_loaded') == p, f'{raw_dir}: prefix load count mismatch')
    require(restored.get('pos_max') == p - 1, f'{raw_dir}: prefix position mismatch')
    require(restored.get('token_ids_match') is True, f'{raw_dir}: prefix token ids not verified')
    declared = manifest.get('declared') or {}
    for key, expected in (('cache_k', 'kvarn4'), ('cache_v', 'kvarn4'),
                          ('device', 'CUDA0'), ('context', context),
                          ('p', p), ('parallel', 1), ('batch', 1024),
                          ('ubatch', 512), ('spec_types', 'none'),
                          ('load_mtp', False), ('warmup', False), ('n_rs_seq', 0)):
        require(declared.get(key) == expected,
                f'{raw_dir}: declared {key}={declared.get(key)!r} != {expected!r}')
    remote = declared.get('remote') or {}
    require(remote.get('host') == xbox_host.rpartition(':')[0],
            f'{raw_dir}: remote host differs from --xbox-host')
    require(remote.get('format') == 'kvarn4' and remote.get('wire_format') == 'fp8',
            f'{raw_dir}: remote format/wire mismatch')
    geometry = manifest.get('geometry') or {}
    for key, expected in (('n_ctx', context), ('n_batch', 1024),
                          ('n_ubatch', 512), ('n_ubatch_physical', 256),
                          ('n_outputs_max', 1), ('n_outputs_max_per_seq', 1),
                          ('n_rs_seq', 0), ('n_seq_max', 1)):
        require(geometry.get(key) == expected,
                f'{raw_dir}: geometry {key}={geometry.get(key)!r} != {expected!r}')
    config = manifest.get('config') or {}
    require(Path(config.get('path', '')).resolve() == Path(common['model_path']),
            f'{raw_dir}: capture model differs from common args')
    require(config.get('n_vocab') == vocab, f'{raw_dir}: capture vocab differs')
    expected_environment = {key: value for key, value in environment.items()
                            if key.startswith(ENV_PREFIXES)}
    # common/arg.cpp sets this from --remote-attn-format with overwrite=1.
    # The launch environment is frozen separately; the manifest is post-parse.
    expected_environment['GGML_RKVA_UPLOAD_FORMAT'] = remote['format']
    actual_environment = manifest.get('environment') or {}
    if actual_environment != expected_environment:
        missing = sorted(set(expected_environment) - set(actual_environment))
        extra = sorted(set(actual_environment) - set(expected_environment))
        changed = sorted(key for key in set(expected_environment) & set(actual_environment)
                         if expected_environment[key] != actual_environment[key])
        raise MatrixError(
            f'{raw_dir}: frozen environment differs (missing={missing}, extra={extra}, '
            f'changed={changed})')
    step_hashes = []
    for step, item in enumerate(step_meta):
        filename = f'{step:04d}.logits.f32.bin'
        require(item.get('filename') == filename, f'{raw_dir}: step {step} filename mismatch')
        require(item.get('bytes') == 4 * vocab, f'{raw_dir}: step {step} byte count mismatch')
        require(item.get('finite') is True, f'{raw_dir}: step {step} not finite')
        require(item.get('position') == spec['length'] - 1 + step,
                f'{raw_dir}: step {step} position mismatch')
        require(item.get('prompt_sha256') == expected_hashes[step],
                f'{raw_dir}: step {step} prompt hash mismatch')
        path = raw_dir / filename
        require(path.is_file(), f'{raw_dir}: step {step} file missing')
        digest = sha256_file(path)
        require(item.get('sha256') == digest, f'{raw_dir}: step {step} sha mismatch')
        step_hashes.append(digest)
        if step == 0:
            for key in ('ubatch_512', 'seam_256', 'seam_128'):
                require(item.get('cadence', {}).get(key) == cadence.get(key),
                        f'{raw_dir}: step 0 cadence {key} mismatch')
        else:
            require(item.get('cadence', {}).get('api_calls') == 1,
                    f'{raw_dir}: step {step} cadence is not one decode')
    history = manifest.get('token_history') or {}
    history_path = raw_dir / str(history.get('file'))
    require(history_path.is_file(), f'{raw_dir}: token history missing')
    expected_history = list(spec['prompt']) + list(teacher_ids[:steps - 1])
    require(history.get('count') == len(expected_history),
            f'{raw_dir}: token history count mismatch')
    require(history.get('sha256') == sha256_file(history_path),
            f'{raw_dir}: token history sha mismatch')
    values = array.array('I')
    with open(history_path, 'rb') as source:
        values.fromfile(source, len(expected_history))
    if sys.byteorder != 'little':
        values.byteswap()
    require(list(values) == expected_history, f'{raw_dir}: token history differs from teacher')
    failed = raw_dir / 'failed.json'
    require(not failed.exists(), f'{raw_dir}: failed.json exists')
    return {
        'manifest': manifest, 'teacher_ids': list(teacher_ids),
        'step_sha256': step_hashes, 'prompt_hashes': expected_hashes,
    }


# ---------------------------------------------------------------------------
# Case assembly, activation, determinism
# ---------------------------------------------------------------------------

def arm_capture(ctx, case, arm, teacher_ids, arm_dir, label):
    arm_dir = make_new_dir(arm_dir)
    write_json_new(arm_dir / 'owner.json', {'owner': TOOL, 'created': now()})
    environment = frozen_environment(ctx['reference']['environment'], arm_dir)
    frozen_common = ctx['common'].get('frozen_path')
    if not frozen_common or not Path(frozen_common).is_file():
        raise RefuseError(f'{case["name"]}/{label}: frozen common args copy is missing')
    command = [
        str(ctx['args'].capture_binary), '--execute',
        '--common-args-json', str(frozen_common),
        '--prompt-json', str(arm_dir / 'inputs' / 'prompt.json'),
        '--prefix-snapshot', str(ctx['prefix']['path']),
        '--p', str(P_DEFAULT), '--context', str(CONTEXT_DEFAULT),
        '--teacher-steps', str(ctx['args'].teacher_steps),
        '--output', str(arm_dir / 'raw'),
    ]
    (arm_dir / 'inputs').mkdir()
    write_json_new(arm_dir / 'inputs' / 'prompt.json', case['prompt'])
    if teacher_ids is not None:
        write_json_new(arm_dir / 'inputs' / 'teacher.json', {
            'teacher_ids': list(teacher_ids),
            'prompt_hashes': [token_hash(case['prompt'] + list(teacher_ids)[:step])
                              for step in range(ctx['args'].teacher_steps)],
            'steps': ctx['args'].teacher_steps,
            'source': 'raw-matrix-frozen-teacher',
        })
    if teacher_ids is not None:
        command += ['--teacher-json', str(arm_dir / 'inputs' / 'teacher.json')]
    ctx['binding_guard'].verify(f'{case["name"]}/{label}-pre')
    gpu_quiet(ctx)
    ctx['flags'].apply(arm == 'on')
    try:
        ctx['binding_guard'].verify(f'{case["name"]}/{label}-flags')
        result = run_capture_child(command, environment, arm_dir / 'capture.log',
                                   ctx['args'].child_timeout, arm_dir,
                                   popen_factory=ctx.get('popen_factory'))
    except ChildCleanupError as error:
        ctx['cleanup_failures'].append(error.details)
        if error.alive:
            ctx['model_alive_pid'] = error.pid
        raise
    finally:
        if ctx.get('model_alive_pid') is None:
            try:
                ctx['flags'].restore()
            except BaseException as error:
                ctx['restore_errors'].append(f'{type(error).__name__}: {error}')
                ctx['flags_restore_failed'] = str(error)
    if ctx.get('flags_restore_failed'):
        raise MatrixError(f'{case["name"]}/{label}: flag restore failed: '
                          f'{ctx["flags_restore_failed"]}')
    ctx['binding_guard'].verify(f'{case["name"]}/{label}-post',
                                loaded=result['loaded_startup'])
    if result['returncode'] != 0:
        failed = arm_dir / 'raw' / 'failed.json'
        detail = ''
        if failed.is_file():
            try:
                document = read_json(failed)
                detail = f": {document.get('phase')}: {document.get('error')}"
            except (OSError, ValueError):
                detail = ''
        raise MatrixError(f'{case["name"]}/{label}: capture exited {result["returncode"]}{detail}')
    if not result.get('startup_event'):
        raise MatrixError(f'{case["name"]}/{label}: startup marker was not observed before exit')
    if not result['loaded_startup']:
        raise MatrixError(f'{case["name"]}/{label}: loaded-library map is empty/missing: '
                          f'{result.get("loaded_error")}')
    manifest = validate_capture_manifest(
        arm_dir / 'raw', case, teacher_ids, ctx['args'].teacher_steps,
        ctx['prefix'], environment, ctx['common'], ctx['args'].xbox_host,
        ctx['gate'])
    lines = read_log_lines(arm_dir / 'capture.log')
    if not any('CAPTURE PASS' in line for line in lines):
        raise MatrixError(f'{case["name"]}/{label}: capture did not log CAPTURE PASS')
    errors, _warnings = ctx['helper'].classify_logs(lines)
    if errors:
        raise MatrixError(f'{case["name"]}/{label}: capture log errors: {errors[:3]}')
    write_json_new(arm_dir / 'libraries.json', {
        'owner': TOOL, 'binding_sha256': ctx['binding']['binding_sha256'],
        'startup_event': result['startup_event'],
        'startup_fallback': result['startup_fallback'],
        'loaded_startup': result['loaded_startup'],
    })
    return {'directory': str(arm_dir), 'raw': str(arm_dir / 'raw'),
            'manifest': manifest['manifest'], 'teacher_ids': manifest['teacher_ids'],
            'step_sha256': manifest['step_sha256'],
            'prompt_hashes': manifest['prompt_hashes'], 'log': str(arm_dir / 'capture.log'),
            'environment': environment, 'command': result['command'],
            'startup_event': result['startup_event'],
            'startup_fallback': result['startup_fallback'],
            'loaded_libraries': result['loaded_startup']}


def _profile(arm, ctx):
    lines = read_log_lines(arm['log'])
    filtered = [line.rstrip() for line in lines if ctx['gate'].PROFILE_FILTER.search(line)]
    return lines, filtered, ctx['gate'].profile_from_lines(lines)


def compute_case_result(ctx, case, off, on, teacher_mode):
    steps = ctx['args'].teacher_steps
    if off['teacher_ids'] != on['teacher_ids']:
        raise MatrixError(f'{case["name"]}: ON arm did not use the OFF teacher ids')
    if off['prompt_hashes'] != on['prompt_hashes']:
        raise MatrixError(f'{case["name"]}: candidate prompt history differs from the teacher')
    off_lines, off_filtered, off_engine = _profile(off, ctx)
    on_lines, on_filtered, on_engine = _profile(on, ctx)
    activation = ctx['gate'].activation_checks(
        off_engine, on_engine, off_filtered, on_filtered, ctx['helper'])
    metrics = []
    for step in range(steps):
        off_stats = raw_stats(Path(off['raw']) / f'{step:04d}.logits.f32.bin',
                              ctx['args'].vocab, ctx['gate'].GATE_DEFAULTS['mass_tolerance'])
        on_stats = raw_stats(Path(on['raw']) / f'{step:04d}.logits.f32.bin',
                             ctx['args'].vocab, ctx['gate'].GATE_DEFAULTS['mass_tolerance'])
        if teacher_mode == 'greedy' and off_stats['top1'] != off['teacher_ids'][step]:
            raise MatrixError(
                f'{case["name"]} step {step}: greedy teacher {off["teacher_ids"][step]} '
                f'is not the OFF argmax {off_stats["top1"]}')
        row = step_metrics_from_stats(
            off_stats, on_stats, off['teacher_ids'][step],
            ctx['gate'].GATE_DEFAULTS['high_confidence_margin'])
        row.update({'step': step, 'position': case['length'] - 1 + step,
                    'prompt_sha256': off['prompt_hashes'][step],
                    'logits_sha256': off['step_sha256'][step],
                    'candidate_logits_sha256': on['step_sha256'][step]})
        metrics.append(row)
    windows = {
        'first_1': evaluate_window('first_1', metrics[:1], ctx['gates'], ctx['gate']),
        'first_8': evaluate_window('first_8', metrics[:min(8, steps)], ctx['gates'],
                                   ctx['gate']),
        'all': evaluate_window('all', metrics, ctx['gates'], ctx['gate']),
    }
    failures = []
    if not activation.get('pass'):
        failures.append('activation: ' + str(activation.get('reason')))
    for name, window in windows.items():
        if not window['pass']:
            failures.append(f'{name}: ' + ', '.join(window['failures']))
    status = 'PASS' if not failures else 'FAIL'
    return {
        'case': case['name'], 'category': case['category'], 'length': case['length'],
        'tail_length': case['tail_length'], 'tokens_sha256': case['tokens_sha256'],
        'prompt_sha256': case['prompt_sha256'], 'prefix_sha256': case['prefix_sha256'],
        'teacher_mode': teacher_mode, 'teacher_ids': off['teacher_ids'],
        'activation': activation, 'steps': metrics, 'windows': windows,
        'status': status, 'failures': failures,
        'qualification': ('RAW_NUMERICAL_STRESS_ONLY' if status == 'PASS' else 'FAILED'),
        'off': {'directory': off['directory'], 'logits_sha256': off['step_sha256']},
        'on': {'directory': on['directory'], 'logits_sha256': on['step_sha256']},
    }


def _compare_control(case, main_arm, control_arm, ctx):
    teacher_equal = main_arm['teacher_ids'] == control_arm['teacher_ids']
    hashes_equal = main_arm['step_sha256'] == control_arm['step_sha256']
    maximum = 0.0
    if not hashes_equal:
        for step in range(ctx['args'].teacher_steps):
            left = raw_stats(Path(main_arm['raw']) / f'{step:04d}.logits.f32.bin',
                             ctx['args'].vocab, ctx['gate'].GATE_DEFAULTS['mass_tolerance'])
            right = raw_stats(Path(control_arm['raw']) / f'{step:04d}.logits.f32.bin',
                              ctx['args'].vocab, ctx['gate'].GATE_DEFAULTS['mass_tolerance'])
            maximum = max(maximum, max(
                abs(left['terms'][i] / left['total'] - right['terms'][i] / right['total'])
                for i in range(ctx['args'].vocab)))
    passed = teacher_equal and hashes_equal
    return {
        'case': case['name'], 'teacher_equal': teacher_equal,
        'identical_logits': hashes_equal,
        'max_abs_probability': maximum if not hashes_equal else 0.0,
        'pass': passed,
        'reason': ('fresh process reproduced the same logits and teacher sequence'
                   if passed else 'fresh process diverged from the baseline'),
    }


def run_control(ctx, case, arm, main_arm, main_teacher):
    control_dir = ctx['args'].output / 'control' / f'{case["name"]}-{arm}'
    teacher = None if (arm == 'off' and ctx['args'].teacher_mode == 'greedy') else main_teacher
    control = arm_capture(ctx, case, arm, teacher, control_dir, f'control-{arm}')
    return _compare_control(case, main_arm, control, ctx)


# ---------------------------------------------------------------------------
# Lifecycle / GPU quiet
# ---------------------------------------------------------------------------

def _compute_app_pids(text):
    pids = set()
    for line in (text or '').splitlines():
        line = line.strip()
        if not line:
            continue
        first = line.split(',', 1)[0].strip()
        if first.isdigit():
            pids.add(int(first))
    return pids


def gpu_quiet(ctx):
    ignored = ctx['gate'].self_pid_chain(ctx['gate'].foreign_process_rows())
    foreign = ctx['gate'].foreign_process_matches(
        ctx['gate'].foreign_process_rows(), ignored=ignored)
    if foreign:
        raise RefuseError('foreign benchmark/build process is active: '
                          + ', '.join(f'{row["comm"]}:{row["pid"]}' for row in foreign[:5]))
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RefuseError(f'cannot verify GPU quiet: {error}') from error
    if result.returncode != 0:
        raise RefuseError(f'nvidia-smi failed: {(result.stderr or "").strip()[:300]}')
    pids = _compute_app_pids(result.stdout) - ignored
    if pids:
        raise RefuseError('GPU compute process is active: ' + ', '.join(map(str, sorted(pids))))


def disk_preflight(output, cases, steps, vocab=VOCAB_DEFAULT, headroom_gb=DISK_HEADROOM_GB,
                   extra_arms=0):
    output = Path(output)
    if output.exists():
        raise RefuseError(f'output already exists; use a new --output directory: {output}')
    parent = output.parent
    if not parent.is_dir():
        raise RefuseError(f'output parent is not a directory: {parent}')
    arms = 2 + int(extra_arms)
    raw_bytes = len(cases) * arms * steps * 4 * vocab
    estimate = raw_bytes + (64 << 20)
    usage = shutil.disk_usage(parent)
    available = usage.free
    required = estimate + int(headroom_gb * (1 << 30))
    if available < required:
        raise RefuseError(
            f'insufficient disk for raw artifacts: free {available} bytes < '
            f'required {required} (estimated {estimate} + {headroom_gb} GiB)')
    return {'raw_bytes': raw_bytes, 'estimate_bytes': estimate,
            'headroom_bytes': required - estimate, 'required_bytes': required,
            'free_bytes': available, 'output': str(output)}


# ---------------------------------------------------------------------------
# Preview / execute
# ---------------------------------------------------------------------------

def parser():
    p = argparse.ArgumentParser(
        description='Offline RAW logits RKVA OFF/ON matrix controller (preview by default).',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--execute', action='store_true',
                   help='run the capture matrix (default: inert preview)')
    p.add_argument('--capture-binary', required=True,
                   help='raw CAPI capture executable')
    p.add_argument('--common-args-json', required=True,
                   help='frozen server argv JSON from the HTTP reference campaign')
    p.add_argument('--cases-dir', required=True,
                   help='case JSON directory (catalog.json or per-case directories)')
    p.add_argument('--reference-campaign', required=True,
                   help='existing HTTP campaign output with binding/preflight/environment')
    p.add_argument('--output', required=True, help='new output directory (never overwritten)')
    p.add_argument('--teacher-steps', type=int, default=TEACHER_STEPS_DEFAULT)
    p.add_argument('--xbox-host', default=XBOX_HOST_DEFAULT, help='HOST:PORT')
    p.add_argument('--repo', default=str(Path(__file__).resolve().parents[3]),
                   help='checkout containing tools/xbox/scripts gate/helper')
    p.add_argument('--xbox-env', default=str(XBOX_ENV_DEFAULT))
    p.add_argument('--deploy', default=None)
    p.add_argument('--deploy-timeout', type=float, default=DEPLOY_TIMEOUT_DEFAULT)
    p.add_argument('--child-timeout', type=float, default=CHILD_TIMEOUT_DEFAULT)
    p.add_argument('--teacher-mode', choices=('greedy', 'http'), default='greedy',
                   help='OFF teacher: raw greedy argmax or the frozen HTTP teacher ids')
    p.add_argument('--determinism', choices=('none', 'off', 'on', 'both'), default='none')
    p.add_argument('--determinism-case', default=None,
                   help='case name for determinism controls (default: first case)')
    p.add_argument('--disk-headroom-gb', type=float, default=DISK_HEADROOM_GB)
    p.add_argument('--vocab', type=int, default=VOCAB_DEFAULT)
    return p


def validate_args(args):
    args.repo = Path(args.repo).resolve()
    args.capture_binary = Path(args.capture_binary).resolve(strict=True)
    args.common_args_json = Path(args.common_args_json).resolve(strict=True)
    args.cases_dir = Path(args.cases_dir).resolve(strict=True)
    args.reference_campaign = Path(args.reference_campaign).resolve(strict=True)
    args.output = Path(args.output).resolve()
    args.xbox_env = Path(args.xbox_env).expanduser()
    args.deploy = Path(args.deploy).resolve() if args.deploy else args.repo / DEPLOY_REL
    if args.vocab != VOCAB_DEFAULT:
        raise ValueError(f'this controller only supports vocab {VOCAB_DEFAULT}')
    if not 1 <= args.teacher_steps <= 1024:
        raise ValueError('teacher-steps must be in [1, 1024]')
    if args.disk_headroom_gb < 0:
        raise ValueError('disk headroom must be nonnegative')
    if args.deploy_timeout <= 0 or args.child_timeout <= 0:
        raise ValueError('timeouts must be positive')
    host, separator, port = args.xbox_host.rpartition(':')
    if not separator or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError('xbox-host must be HOST:PORT')
    args.xbox_ip = host
    return args


def _plan_cases(args, reference, helper):
    base = reference['base_tokens']
    reference_prefix = reference['binding'].get('prefix')
    if not isinstance(reference_prefix, dict):
        raise RefuseError('reference binding has no prefix identity')
    prefix_sha = reference_prefix.get('actual_prefix_sha256')
    if not prefix_sha:
        raise RefuseError('reference binding has no actual prefix hash')
    cases = load_cases(args.cases_dir, args.teacher_steps, prefix_sha, base,
                       P_DEFAULT, CONTEXT_DEFAULT)
    if base is None:
        base = cases[0]['prompt'][:P_DEFAULT]
    return cases, base


def _extra_arms(args):
    if args.determinism == 'both':
        return 2
    if args.determinism in ('off', 'on'):
        return 1
    return 0


def preview(args):
    gate, helper = load_runtime(args.repo)
    reference = load_reference(args.reference_campaign, gate, helper)
    common = verify_common_args(args.common_args_json, reference, args.xbox_host, args.output)
    cases, base = _plan_cases(args, reference, helper)
    prefix = load_prefix(reference, base, gate, helper)
    preview_binding = build_binding(
        args, reference, common, prefix, cases, gate, helper,
        {'files': [], 'sha256': sha256_text(canonical([]))})
    disk = disk_preflight(args.output, cases, args.teacher_steps, args.vocab,
                          args.disk_headroom_gb, _extra_arms(args))
    teacher_plan = []
    for case in cases:
        teacher_plan.append({
            'case': case['name'], 'length': case['length'],
            'tail_length': case['tail_length'], 'teacher_mode': args.teacher_mode,
            'teacher_steps': args.teacher_steps,
            'reference_teacher': (str(reference['root'] / 'cases' / case['reference']
                                      / 'off' / 'teacher.json')),
        })
    return {
        'schema': SCHEMA, 'mode': 'preview', 'status': 'PREVIEW_ONLY',
        'label': 'RAW_NUMERICAL_STRESS_ONLY', 'starts_child': False,
        'writes_output': False, 'requires_execute': True, 'tool': TOOL,
        'repo': str(args.repo), 'capture_binary': stat_identity(args.capture_binary),
        'common_args': {key: value for key, value in common.items()
                        if key not in ('bytes', 'stat')},
        'reference': {
            'root': str(reference['root']),
            'binding_sha256': reference['binding_sha256'],
            'source': reference['source'], 'model': reference['model'],
            'environment_path': reference['environment_path'],
            'environment_keys': sorted(reference['environment']),
        }, 'prefix': {'path': prefix['path'], 'meta_sha256': prefix['meta_sha256'],
                      'contract': prefix['contract'],
                      'files': prefix['files'], 'verify_sha': prefix['verify_sha']},
        'cases': [{key: case[key] for key in
                   ('name', 'category', 'length', 'tail_length', 'prompt_sha256',
                    'prefix_sha256', 'tokens_sha256', 'source', 'reference')}
                  for case in cases],
        'teacher_plan': teacher_plan,
        'configuration': {
            'P': P_DEFAULT, 'context': CONTEXT_DEFAULT, 'vocab': args.vocab,
            'teacher_steps': args.teacher_steps, 'teacher_mode': args.teacher_mode,
            'determinism': args.determinism,
            'determinism_case': args.determinism_case or (cases[0]['name'] if cases else None),
            'xbox_host': args.xbox_host, 'output': str(args.output),
        },
        'thresholds': dict(gate.GATE_DEFAULTS),
        'semantic_reference': {
            'reference_answer_json_consumed': False,
            'fixed_reference_nll': False,
            'free_greedy_semantic_accuracy': False,
        },
        'disk': disk, 'binding': preview_binding,
        'foreign_processes': gate.foreign_process_matches(gate.foreign_process_rows()),
        'scope': ('synthetic/default token data numerical stress only; not broad semantic '
                  'quality and not INT8 default approval'),
        'next': ('re-run with --execute after the root owner confirms GPU/Xbox are free; '
                 'Astra reviews the artifacts before use'),
    }


class SignalGuard:
    def __init__(self):
        self.previous = {}

    def __enter__(self):
        def handler(signum, _frame):
            raise KeyboardInterrupt(f'signal {signum}')
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            self.previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)
        return self

    def __exit__(self, exc_type, exc, traceback):
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)
        return False


def _reference_teacher(reference, case, steps, gate):
    path = reference['root'] / 'cases' / case['reference'] / 'off' / 'teacher.json'
    if not path.is_file():
        raise RefuseError(f'{case["name"]}: frozen HTTP teacher is missing: {path}')
    document = read_json(path)
    ids = document.get('teacher_ids')
    hashes = document.get('prompt_hashes')
    if (not isinstance(ids, list) or len(ids) < steps
            or not isinstance(hashes, list) or len(hashes) < steps):
        raise RefuseError(f'{case["name"]}: frozen HTTP teacher is incomplete')
    ids = ids[:steps]
    if any(type(token) is not int or not 0 <= token < VOCAB_DEFAULT for token in ids):
        raise RefuseError(f'{case["name"]}: frozen HTTP teacher has an out-of-vocab id')
    expected = [token_hash(case['prompt'] + ids[:step]) for step in range(steps)]
    if hashes[:steps] != expected:
        raise RefuseError(f'{case["name"]}: frozen HTTP teacher prompt hashes do not match')
    return ids


def execute(args):
    gate, helper = load_runtime(args.repo)
    reference = load_reference(args.reference_campaign, gate, helper)
    common = verify_common_args(args.common_args_json, reference, args.xbox_host, args.output)
    cases, base = _plan_cases(args, reference, helper)
    prefix = load_prefix(reference, base, gate, helper)
    if args.teacher_mode == 'http':
        for case in cases:
            _reference_teacher(reference, case, args.teacher_steps, gate)
    model_stat = stat_identity(reference['model']['path'])
    if not model_stat.get('exists') or model_stat.get('size') != reference['model'].get('size'):
        raise RefuseError('reference model file is missing or has a different size')
    if sha256_file(reference['model']['path']) != reference['model'].get('sha256'):
        raise RefuseError('reference model content differs from the frozen binding')
    disk = disk_preflight(args.output, cases, args.teacher_steps, args.vocab,
                          args.disk_headroom_gb, _extra_arms(args))
    probe_dir = args.output.parent / f'.{TOOL}-env-probe-{os.getpid()}'
    environment = frozen_environment(reference['environment'], probe_dir)
    try:
        libraries = _GATE.build_library_binding(
            str(args.capture_binary), active_libraries_for(args.capture_binary, environment))
        _verify_libraries_against_reference(reference, libraries)
        binding = build_binding(args, reference, common, prefix, cases, gate, helper,
                                libraries)
        binding_guard = BindingGuard(binding, args.capture_binary, environment,
                                     libraries, helper)
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
    # All local validation is complete; no flags, network, or GPU yet.
    preflight_doc = preview(args)
    gpu_quiet({'gate': gate, 'args': args})
    ctx = {
        'gate': gate, 'helper': helper, 'args': args, 'reference': reference,
        'common': common, 'prefix': prefix, 'gates': dict(gate.GATE_DEFAULTS),
        'binding': binding, 'binding_guard': binding_guard, 'cases': cases,
        'libraries': libraries, 'flags': None, 'disk': disk,
        'popen_factory': None, 'model_alive_pid': None,
        'cleanup_failures': [], 'restore_errors': [], 'flags_restore_failed': None,
    }
    make_new_dir(args.output)
    if not common.get('frozen_path'):
        raise RefuseError('frozen common args path was not allocated')
    write_bytes_new(common['frozen_path'], common['bytes'])
    if sha256_file(common['frozen_path']) != common['frozen_sha256']:
        raise RefuseError('frozen common args copy changed after writing')
    write_json_new(args.output / 'binding.json', binding)
    preflight_doc['binding'] = binding
    preflight_doc['libraries'] = libraries
    preflight_doc['prefix'] = {
        'path': prefix['path'], 'meta_sha256': prefix['meta_sha256'],
        'contract': prefix['contract'], 'files': prefix['files'],
        'verify_sha': prefix.get('verify_sha', True)}
    write_json_new(args.output / 'preflight.json', preflight_doc)
    deploy = gate.DeployClient(
        args.deploy, args.xbox_env, args.deploy_timeout, xbox_ip=args.xbox_ip)
    pfn = deploy.pfn()
    flags = gate.LocalStateFlags(deploy, pfn, args.output)
    flags.capture_original()
    ctx['flags'] = flags
    results, controls, errors = [], [], []
    abort_stage = None
    try:
        for case in cases:
            case_dir = args.output / 'cases' / case['name']
            make_new_dir(case_dir)
            write_json_new(case_dir / 'case.json', {
                key: case[key] for key in
                ('name', 'category', 'length', 'tail_length', 'prompt_sha256',
                 'prefix_sha256', 'tokens_sha256', 'source', 'reference')
            } | {'teacher_mode': args.teacher_mode,
                 'answer_tokens_present': case['answer_tokens'] is not None})
            off_teacher = None
            if args.teacher_mode == 'http':
                off_teacher = _reference_teacher(reference, case, args.teacher_steps, gate)
            off = arm_capture(ctx, case, 'off', off_teacher,
                              case_dir / 'off', 'off')
            teacher = off['teacher_ids'] if args.teacher_mode == 'greedy' else off_teacher
            on = arm_capture(ctx, case, 'on', teacher, case_dir / 'on', 'on')
            result = compute_case_result(ctx, case, off, on, args.teacher_mode)
            write_json_new(case_dir / 'case-result.json', result)
            results.append(result)
            print(json.dumps({'case': result['case'], 'status': result['status'],
                              'windows': {key: value['pass']
                                          for key, value in result['windows'].items()},
                              'activation': result['activation'].get('pass')},
                             ensure_ascii=False), flush=True)
        if args.determinism != 'none' and results:
            target_name = args.determinism_case or cases[0]['name']
            target = next((item for item in results if item['case'] == target_name), None)
            if target is None:
                raise MatrixError(f'determinism case not selected: {target_name}')
            case = next(item for item in cases if item['name'] == target_name)
            case_dir = args.output / 'cases' / target_name
            if args.determinism in ('off', 'both'):
                control = run_control(ctx, case, 'off', {
                    'raw': str(case_dir / 'off' / 'raw'),
                    'teacher_ids': target['teacher_ids'],
                    'step_sha256': [step['logits_sha256'] for step in target['steps']],
                }, target['teacher_ids'])
                controls.append(control)
            if args.determinism in ('on', 'both'):
                control = run_control(ctx, case, 'on', {
                    'raw': str(case_dir / 'on' / 'raw'),
                    'teacher_ids': target['teacher_ids'],
                    'step_sha256': [step['candidate_logits_sha256'] for step in target['steps']],
                }, target['teacher_ids'])
                controls.append(control)
            for control in controls:
                if not control['pass']:
                    errors.append(f'control {control["case"]}/{control.get("teacher_equal")}: '
                                  + control['reason'])
    except BaseException as failure:
        abort_stage = type(failure).__name__
        if isinstance(failure, ChildCleanupError):
            if failure.details not in ctx['cleanup_failures']:
                ctx['cleanup_failures'].append(failure.details)
            if failure.alive:
                ctx['model_alive_pid'] = failure.pid
        errors.append(f'{type(failure).__name__}: {failure}')
        try:
            write_json_new(args.output / 'campaign-abort.json', {
                'owner': TOOL, 'stage': abort_stage, 'error': str(failure), 'at': now(),
            })
        except BaseException:
            pass
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if ctx.get('model_alive_pid') is not None:
            errors.append(
                f'flags were NOT restored because owned capture pid '
                f'{ctx["model_alive_pid"]} may still be alive')
            ctx['flags_restore_skipped'] = True
        else:
            try:
                flags.restore()
            except BaseException as first:
                try:
                    flags.restore()
                except BaseException as second:
                    ctx['flags_restore_failed'] = str(second)
                    errors.append(f'flag restore failed: {type(first).__name__}: {first}; '
                                  f'retry: {type(second).__name__}: {second}')
        for cleanup in ctx.get('cleanup_failures') or []:
            errors.append(f'owned child cleanup failure: {cleanup}')
        errors.extend(ctx.get('restore_errors') or [])
        try:
            verify_prefix_unchanged(prefix)
        except BaseException as error:
            errors.append(str(error))
        try:
            binding_guard.verify('final')
        except BaseException as error:
            errors.append(str(error))
    checks = _final_checks(results, cases, controls, errors)
    status = 'FAIL' if not all(check['pass'] for check in checks.values()) else (
        'NEEDS_REVIEW' if any(result['windows']['all']['metrics'].get(
            'high_confidence_divergences', 0) for result in results) else 'PASS')
    final = {
        'schema': SCHEMA, 'finished': now(), 'tool': TOOL, 'status': status,
        'qualification_label': ('RAW_NUMERICAL_STRESS_ONLY' if status == 'PASS'
                                else 'NEEDS_REVIEW' if status == 'NEEDS_REVIEW'
                                else 'FAILED'),
        'int8_default_authorized': False,
        'scope': ('synthetic/default token data numerical stress only; not broad semantic '
                  'quality and not INT8 default approval'),
        'hardware_run_by_this_tool': True,
        'binding_sha256': binding['binding_sha256'],
        'reference_binding_sha256': reference['binding_sha256'],
        'capture_binary': binding['capture_binary'],
        'libraries': libraries,
        'prefix': {'path': prefix['path'], 'meta_sha256': prefix['meta_sha256'],
                   'contract': prefix['contract']},
        'environment': {'sha256': binding['environment']['sha256'],
                        'keys': binding['environment']['keys']},
        'thresholds': dict(gate.GATE_DEFAULTS),
        'windows': {'first_1': 'independent one-step report',
                    'first_8': 'independent first-eight report',
                    'all': f'all {args.teacher_steps} steps'},
        'cases': results, 'controls': controls, 'checks': checks,
        'lifecycle_errors': errors, 'disk': disk,
        'teacher_mode': args.teacher_mode, 'determinism': args.determinism,
        'flags_restored': (ctx.get('model_alive_pid') is None
                           and not ctx.get('flags_restore_failed')),
        'model_alive_pid': ctx.get('model_alive_pid'),
        'cleanup_failures': ctx.get('cleanup_failures') or [],
        'common_args': binding.get('common_args'),
        'model_identity': binding.get('model'),
        'semantic_reference': {
            'reference_answer_json_consumed': False,
            'fixed_reference_nll': False,
            'free_greedy_semantic_accuracy': False,
        },
        'note': ('raw CAPI collector only; root must validate hardware output and Astra '
                 'must review before any integration'),
    }
    write_json_new(args.output / 'final.json', final)
    complete_files = ['binding.json', 'preflight.json', 'final.json'
                      ] + [f'cases/{case["case"]}/case-result.json' for case in results]
    try:
        write_json_new(args.output / 'complete.json', {
            'owner': TOOL, 'status': status, 'completed': now(),
            'files': {name: sha256_file(args.output / name) for name in complete_files
                      if (args.output / name).is_file()},
        })
    except BaseException as error:
        final['complete_marker_error'] = str(error)
    print(json.dumps({'status': status, 'qualification_label': final['qualification_label'],
                      'output': str(args.output),
                      'failed_checks': [name for name, check in checks.items()
                                        if not check['pass']]}, indent=2), flush=True)
    if status == 'FAIL':
        return 3
    if status == 'NEEDS_REVIEW':
        return 4
    return 0


def _verify_libraries_against_reference(reference, libraries):
    reference_files = {item['path']: item for item in
                       reference['libraries'].get('files') or []}
    for item in libraries['files']:
        expected = reference_files.get(item['path'])
        if expected is None:
            raise RefuseError(
                f'capture library {item["path"]} is not in the frozen reference binding')
        if expected.get('sha256') != item['sha256']:
            raise RefuseError(f'capture library content differs: {item["path"]}')


def _final_checks(results, cases, controls, errors):
    def add(name, passed, detail):
        return {'pass': bool(passed), 'detail': detail}

    checks = {}
    checks['no_lifecycle_or_protocol_errors'] = add(
        'no_lifecycle_or_protocol_errors', not errors,
        '; '.join(errors[:5]) if errors else 'none')
    checks['all_cases_complete'] = add(
        'all_cases_complete', len(results) == len(cases),
        f'{len(results)}/{len(cases)} cases completed')
    checks['all_activation_proved'] = add(
        'all_activation_proved', bool(results) and all(
            result['activation'].get('pass') is True for result in results),
        'both arms proved activation' if results else 'no cases')
    checks['all_numeric_finite'] = add(
        'all_numeric_finite', bool(results) and all(
            result['windows']['all']['metrics'].get('finite') is True
            for result in results), 'all NLL/KL finite' if results else 'no cases')
    checks['all_window_gates'] = add(
        'all_window_gates', bool(results) and all(
            result.get('status') == 'PASS'
            and all(window['pass'] for window in result['windows'].values())
            for result in results),
        'see per-case first_1, first_8 and all window results' if results else 'no cases')
    checks['determinism_controls'] = add(
        'determinism_controls', all(control.get('pass') is True for control in controls),
        'not selected' if not controls else
        ('all controls reproduced' if all(c.get('pass') for c in controls)
         else 'a control diverged'))
    return checks


def main(argv=None):
    try:
        args = validate_args(parser().parse_args(argv))
    except (SystemExit,):
        raise
    except (ValueError, OSError, RefuseError) as error:
        print(f'Configuration error: {error}', file=sys.stderr)
        return 2
    try:
        with SignalGuard():
            if not args.execute:
                print(json.dumps(preview(args), indent=2, ensure_ascii=False))
                return 0
            return execute(args)
    except RefuseError as error:
        print(f'Refused: {error}', file=sys.stderr)
        return 2
    except KeyboardInterrupt as error:
        print(f'Interrupted: {error}', file=sys.stderr)
        return 130
    except (MatrixError, ValueError, OSError, KeyError, TypeError) as error:
        print(f'FAILED: {type(error).__name__}: {error}', file=sys.stderr)
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
