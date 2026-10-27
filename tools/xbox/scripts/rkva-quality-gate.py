#!/usr/bin/env python3
"""Teacher-forced INT8 RKVA quality gate; preview by default, --execute explicitly runs.

This is a *quality* gate, not a throughput benchmark.  It compares the same
teacher-forced distributions produced by two arms of the installed Xbox RKVA
build:

  OFF : decode rowlane ON, opt-in INT8 multi-query kernel OFF
  ON  : decode rowlane ON, opt-in INT8 multi-query kernel ON

The comparison sequence is created once on the baseline (INT8 OFF) and is the
only sequence fed to both arms.  Candidate-produced tokens never become
teacher tokens.

Nothing is executed unless ``--execute`` is passed.  The preview starts no
server, touches no Xbox LocalState flag, writes no output and contacts no
network endpoint.  Even on ``--execute`` this script never starts/stops the
installed app, never compiles CUDA, never changes the tier/router production
files and never pushes/merges Git.

The lifecycle, binary command, pinned environment and frozen-prefix contract
are re-used from ``tools/xbox/scripts/bench-rkva-range.py`` by import; this
script does not fork a second copy of that 1200-line benchmark.

Exit codes: 0 preview/pass, 2 refused/configuration, 3 failure,
4 completed but NEEDS_REVIEW.
"""
from __future__ import annotations

import argparse
import array
import contextlib
import datetime
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.error

REPO_DEFAULT = Path(__file__).resolve().parents[3]
HELPER_REL = Path('tools/xbox/scripts/bench-rkva-range.py')
DEPLOY_REL = Path('tools/xbox/scripts/deploy.sh')
BINARY_DEFAULT = Path('/home/hjotha/beellama.cpp/build-optimized/bin/llama-server')
PREFIX_DEFAULT = Path(
    '/home/hjotha/llama-slot-cache/rkva-2xn256-dev/a7e8962f16a8/'
    'prod-prefix-p102400-c131072-n256-w4096.bin')
TOKENS_DEFAULT = Path('/home/hjotha/beellama-gate/20261017/tokens-prompt120k-t2.json')
XBOX_ENV_DEFAULT = Path.home() / '.config/xllama/xbox-env'

CONTEXT_DEFAULT = 131072
CONTEXT_MAX = 160000
P_DEFAULT = 102400
FROZEN_PREFIX_BATCH = 512
TEACHER_STEPS_DEFAULT = 128
SMOKE_STEPS = 2
LENGTHS_DEFAULT = (106496, 118784, 130560)
SMOKE_LENGTH_OFFSET = 4096
WAVEFRONT_TILE = 256
WAVEFRONT_TILES = 2

INT8_FLAG = 'rkva-mq-i8.flag'
GQA2_FLAG = 'rkva-mq-gqa2.flag'
ROWLAND_OPT_OUT_FLAG = 'rkva-decode-rowlane-off.flag'
# Every experimental engine flag this harness touches must be captured and
# restored, even when the current arm does not enable it: a stale GQA2 flag
# would otherwise contaminate INT8/rowlane controls.
MANAGED_FLAGS = (INT8_FLAG, ROWLAND_OPT_OUT_FLAG, GQA2_FLAG)
PFN_PREFIX = 'GianlucaMazza.xllama_'

# server-task.cpp: logarithm(0) returns std::numeric_limits<float>::lowest().
# Anything at/below this threshold is the "probability is exactly zero"
# sentinel, not a real log-probability.
SENTINEL_THRESHOLD = -3.0e38
DIST_MAGIC = b'RKQG1'
DIST_HEADER = struct.Struct('<II')

PROFILE_MARKER = 'remote_attn gpu_profile '
PROFILE_FILTER = re.compile(r'gpu_profile|RemoteAttn|remote_attn|RKVA|ggml_input_cache')
FOREIGN_COMMS = {'llama-server', 'llama-bench', 'llama-perplexity', 'ninja', 'cmake', 'ctest'}
FOREIGN_COMM_PREFIXES = ('test-rkva', 'bench-rkva')
FOREIGN_SCRIPT_HINTS = ('bench-rkva-range.py', 'run-rkva-', 'rkva-quality-gate.py')

CASE_CATEGORIES = ('portuguese', 'documents', 'code', 'numbers', 'retrieval', 'factual')
DEFAULT_CASE_NAMES = CASE_CATEGORIES

# Predeclared conservative gates.  They are configurable, but the effective
# configuration is frozen in preflight.json before --execute starts.  No code
# path relaxes a gate automatically.
GATE_DEFAULTS = {
    'max_ppl_ratio': 1.01,
    'max_kl_mean': 0.001,
    'max_kl_p99': 0.01,
    'max_case_nll_delta': 0.02,
    'mass_tolerance': 1e-3,
    'high_confidence_margin': 0.5,
}

INTEGRITY_SCOPE = {
    'full_overflow_save_supported': False,
    'native_kvarn4_full_save_refusal': 'src/llama-kv-cache-kvarn.cpp:5045-5050',
    'native_kvarn4_overflow_restore_refusal': 'src/llama-kv-cache-kvarn.cpp:5399-5405',
    'native_ops_17_18_enabled': False,
    'owner_ram_records_stage_tail_codec': False,
    'overflow_restore_qualified': False,
    'policy': ('Only the existing local prefix P snapshot is restored; the evaluated tail is '
               'never saved as FULL overflow and a save-with-overflow is never reported PASS.'),
}

LIMITATIONS = [
    'FULL slot save with occupied overflow is not supported by the native KVarN4 codec '
    '(src/llama-kv-cache-kvarn.cpp:5045-5050 refuses the save; :5399-5405 refuses the '
    'standard-cache overflow restore; native op 17/18 are disabled and owner RAM '
    'records/stage/tail have no codec). A save-with-overflow result must never be '
    'reported as PASS.',
    'Only the existing local prefix snapshot is reused. Its tail is always re-evaluated '
    'on each fresh server; a cold prefill is never silently accepted as a snapshot hit.',
    'A C131072 snapshot is not compatible with C160000. A larger context requires an '
    'explicitly compatible snapshot (and --allow-new-prefix for a brand-new prefix).',
    'A PASS over this limited matrix does not authorize INT8 as the global/default mode. '
    'The result label is QUALIFIED_SCOPE_ONLY.',
    'The installed app is never started, stopped or probed. Only two LocalState flag '
    'files are toggled and the original state is restored and verified.',
    'No tier/router production configuration is read or written by this tool.',
]


class GateError(RuntimeError):
    """A protocol/lifecycle error that must fail the current arm/case."""


class RefuseError(RuntimeError):
    """An unsafe configuration/ownership/binding condition; refuse before acting."""


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False)


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


def atomic_write_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + '.tmp')
    temporary.write_bytes(value)
    temporary.replace(path)


def json_safe(value):
    """Replace non-finite floats with null so artifacts stay valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path, value):
    atomic_write_bytes(path, (json.dumps(json_safe(value), indent=2,
                                        ensure_ascii=False, allow_nan=False)
                              + '\n').encode('utf-8'))


def read_json(path):
    return json.loads(Path(path).read_text())


def stat_identity(path):
    path = Path(path)
    if not path.exists():
        return {'path': str(path), 'exists': False}
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {'path': str(resolved), 'exists': True, 'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino}


def token_hash(tokens):
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack('<I', token))
    return digest.hexdigest()


def load_helper(repo):
    repo = Path(repo).resolve()
    helper_path = repo / HELPER_REL
    if not helper_path.is_file():
        raise RefuseError(f'helper not found: {helper_path}')
    spec = importlib.util.spec_from_file_location('rkva_bench_helper', helper_path)
    if spec is None or spec.loader is None:
        raise RefuseError(f'cannot import helper: {helper_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--execute', action='store_true',
                   help='run the hardware quality gate (default: inert preview)')
    p.add_argument('--repo', help='repository root (default: relative to this script)')
    p.add_argument('--server-binary', default=str(BINARY_DEFAULT),
                   help='llama-server used by both arms (default: %(default)s)')
    p.add_argument('--model', required=True, help='target GGUF model')
    p.add_argument('--prompt-tokens', default=str(TOKENS_DEFAULT),
                   help='JSON token array providing the shared local prefix P (default: %(default)s)')
    p.add_argument('--prefix-snapshot', default=str(PREFIX_DEFAULT),
                   help='existing frozen local-prefix slot file (default: %(default)s)')
    p.add_argument('--allow-new-prefix', action='store_true',
                   help='explicitly allow creating a brand-new P snapshot when none exists')
    p.add_argument('--output', required=True, help='new/resumable artifact directory')
    p.add_argument('--xbox-host', help='Xbox RKVA listener HOST:PORT (required for --execute)')
    p.add_argument('--xbox-port', type=int, default=50054,
                   help='default RKVA port when --xbox-host omits it (default: %(default)s)')
    p.add_argument('--xbox-env', default=str(XBOX_ENV_DEFAULT),
                   help='deploy.sh credential environment file (default: %(default)s)')
    p.add_argument('--deploy', default=None,
                   help='deploy.sh path (default: <repo>/tools/xbox/scripts/deploy.sh)')

    p.add_argument('--context', type=int, default=CONTEXT_DEFAULT,
                   help=f'test context, at most {CONTEXT_MAX} (default: %(default)s)')
    p.add_argument('--p', type=int, default=P_DEFAULT, help='local prefix P (default: %(default)s)')
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--ubatch', type=int, default=512)
    p.add_argument('--wavefront-tile', type=int, default=WAVEFRONT_TILE)
    p.add_argument('--wavefront-tiles', type=int, default=WAVEFRONT_TILES)
    p.add_argument('--cuda-fp8', choices=('prefill', 'mapped'), default='mapped')
    p.add_argument('--range-kernel', choices=('fine', 'balanced'), default='fine')
    p.add_argument('--tcp-prefetch', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--threads-batch', type=int)
    p.add_argument('--device', default='CUDA0')
    p.add_argument('--gpu-layers', type=int, default=99)
    p.add_argument('--cache-ram', type=int, default=4096)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--library-dir', action='append', default=[])

    p.add_argument('--cases', default='all',
                   help='comma-separated built-in case names or "all" (default: all)')
    p.add_argument('--cases-json',
                   help='external case corpus (replaces the built-in case names)')
    p.add_argument('--lengths', default=','.join(str(n) for n in LENGTHS_DEFAULT),
                   help='comma-separated total prompt lengths (default: %(default)s)')
    p.add_argument('--teacher-steps', type=int, default=TEACHER_STEPS_DEFAULT,
                   help='distributions per case (default: %(default)s)')
    p.add_argument('--smoke', action='store_true',
                   help=f'one case at P+{SMOKE_LENGTH_OFFSET}, {SMOKE_STEPS} teacher steps')
    p.add_argument('--retry-failed', action='store_true',
                   help='rerun a completed case whose stored status is FAIL')

    p.add_argument('--determinism-control', choices=('auto', 'on', 'off'), default='auto',
                   help='fresh-process baseline control (default: auto for long cases)')
    p.add_argument('--verify-overflow-save-refusal', action='store_true',
                   help='DISABLED: the native FULL overflow codec is unsupported; '
                        'no QA qualification is possible (root/Astra use a separate probe)')
    p.add_argument('--startup-timeout', type=float, default=600)
    p.add_argument('--request-timeout', type=float, default=1800)
    p.add_argument('--deploy-timeout', type=float, default=180)

    p.add_argument('--max-ppl-ratio', type=float, default=GATE_DEFAULTS['max_ppl_ratio'])
    p.add_argument('--max-kl-mean', type=float, default=GATE_DEFAULTS['max_kl_mean'])
    p.add_argument('--max-kl-p99', type=float, default=GATE_DEFAULTS['max_kl_p99'])
    p.add_argument('--max-case-nll-delta', type=float,
                   default=GATE_DEFAULTS['max_case_nll_delta'])
    p.add_argument('--mass-tolerance', type=float, default=GATE_DEFAULTS['mass_tolerance'])
    p.add_argument('--high-confidence-margin', type=float,
                   default=GATE_DEFAULTS['high_confidence_margin'])
    return p


def gate_config(args):
    gates = {
        'max_ppl_ratio': args.max_ppl_ratio,
        'max_kl_mean': args.max_kl_mean,
        'max_kl_p99': args.max_kl_p99,
        'max_case_nll_delta': args.max_case_nll_delta,
        'mass_tolerance': args.mass_tolerance,
        'high_confidence_margin': args.high_confidence_margin,
    }
    for name, value in gates.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
    if not 0 < gates['max_ppl_ratio'] <= 2:
        raise ValueError('max_ppl_ratio must be in (0, 2]')
    for name in ('max_kl_mean', 'max_kl_p99', 'max_case_nll_delta', 'mass_tolerance'):
        if not 0 < gates[name] <= 1:
            raise ValueError(f'{name} must be in (0, 1]')
    if not 0 <= gates['high_confidence_margin'] < 1:
        raise ValueError('high_confidence_margin must be in [0, 1)')
    if gates['max_kl_p99'] < gates['max_kl_mean']:
        raise ValueError('max_kl_p99 must be >= max_kl_mean')
    return gates


def validate(args):
    args.repo = Path(args.repo).resolve() if args.repo else REPO_DEFAULT
    args.server_binary = Path(args.server_binary)
    args.model = Path(args.model)
    args.prompt_tokens = Path(args.prompt_tokens)
    args.prefix_snapshot = Path(args.prefix_snapshot)
    args.output = Path(args.output).resolve()
    args.xbox_env = Path(args.xbox_env).expanduser()
    args.library_dir = [Path(p) for p in args.library_dir]

    if not 0 < args.context <= CONTEXT_MAX:
        raise ValueError(f'context must be in (0, {CONTEXT_MAX}]; never alter a tier file')
    if args.context % 256:
        raise ValueError('context must be a multiple of 256')
    if args.p <= 0 or args.p % 256 or args.p >= args.context:
        raise ValueError('P must be a positive multiple of 256 smaller than context')
    if args.batch <= 0 or args.ubatch <= 0 or args.ubatch > args.batch or args.batch % args.ubatch:
        raise ValueError('batch must be a positive multiple of ubatch')
    if args.device != 'CUDA0':
        raise ValueError('only the CUDA0 test device is supported by this gate')
    if args.gpu_layers != 99:
        raise ValueError('the frozen profile requires --gpu-layers 99')
    if args.wavefront_tile != WAVEFRONT_TILE or args.wavefront_tiles != WAVEFRONT_TILES:
        raise ValueError('the frozen profile requires wavefront 2xN256')
    if args.batch not in (512, 1024) or args.ubatch != 256 * args.wavefront_tiles:
        raise ValueError('WF2xN256 requires B512/UB512 or B1024/UB512')
    if args.cuda_fp8 != 'mapped':
        raise ValueError('the frozen profile requires mapped FP8')
    if args.range_kernel != 'fine':
        raise ValueError('the frozen profile requires the fine range kernel')
    if not args.tcp_prefetch:
        raise ValueError('the frozen profile requires TCP prefetch (proof is mandatory)')
    if args.threads <= 0 or (args.threads_batch is not None and args.threads_batch <= 0):
        raise ValueError('thread counts must be positive')
    if args.seed != 1234:
        raise ValueError('the frozen teacher protocol requires seed 1234')
    if args.cache_ram < 0 or args.startup_timeout <= 0 or args.request_timeout <= 0:
        raise ValueError('cache RAM must be nonnegative and timeouts positive')
    if args.teacher_steps < 1 or args.teacher_steps > 4096:
        raise ValueError('teacher-steps must be in [1, 4096]')
    args.frozen_prefix_batch = FROZEN_PREFIX_BATCH
    args.gates = gate_config(args)
    if args.verify_overflow_save_refusal:
        raise ValueError(
            '--verify-overflow-save-refusal is disabled: the native FULL overflow '
            'codec is unsupported and this QA matrix cannot qualify it; use the '
            'separate root/Astra manual probe')

    if args.smoke:
        if args.cases_json:
            raise ValueError('--smoke cannot be combined with --cases-json; select one case')
        args.cases = 'portuguese'
        args.lengths = str(args.p + SMOKE_LENGTH_OFFSET)
        args.teacher_steps = SMOKE_STEPS

    try:
        args.lengths_list = [int(n) for n in args.lengths.split(',')]
    except ValueError as error:
        raise ValueError('lengths must be comma-separated integers') from error
    if not args.lengths_list or args.lengths_list != sorted(set(args.lengths_list)):
        raise ValueError('lengths must be strictly ascending and unique')
    for length in args.lengths_list:
        if length <= args.p or length >= args.context:
            raise ValueError(f'length {length} must be > P and < context')
        if length + args.teacher_steps >= args.context:
            raise ValueError(
                f'length {length} + teacher-steps {args.teacher_steps} must stay strict inside context')
        if args.smoke and length != args.p + SMOKE_LENGTH_OFFSET:
            raise ValueError('smoke mode requires the single P+4096 length')

    if args.cases_json:
        args.case_source = ('external', str(Path(args.cases_json).resolve()))
    else:
        names = [name.strip() for name in args.cases.split(',') if name.strip()]
        if names == ['all']:
            names = list(CASE_CATEGORIES)
        if not names or len(names) != len(set(names)):
            raise ValueError('cases must be a non-empty unique list')
        unknown = [name for name in names if name not in CASE_CATEGORIES]
        if unknown:
            raise ValueError('unknown built-in cases: ' + ', '.join(unknown))
        args.case_source = ('builtin', names)

    if args.xbox_host and ':' not in args.xbox_host:
        args.xbox_host = f'{args.xbox_host}:{args.xbox_port}'
    if args.xbox_host:
        host, separator, port = args.xbox_host.rpartition(':')
        if not separator or not host or not port.isdigit() or not 0 < int(port) < 65536:
            raise ValueError('xbox-host must be explicit HOST:PORT')
        args.xbox_ip = host
        args.xbox_port = int(port)
    else:
        args.xbox_ip = None
        args.xbox_port = args.xbox_port
        if args.execute:
            raise ValueError('--xbox-host is required for --execute')

    for name in ('startup_timeout', 'request_timeout', 'deploy_timeout'):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f'{name} must be finite')
    return args


def helper_args(args):
    """Argument object consumed by the imported bench-rkva-range helpers."""
    return SimpleNamespace(
        server_binary=str(args.server_binary), model=str(args.model),
        prompt_tokens=str(args.prompt_tokens), output=str(args.output),
        xbox_host=args.xbox_host or 'HOST:PORT', host='127.0.0.1', port=0,
        context=args.context, batch=args.batch, ubatch=args.ubatch,
        threads=args.threads, threads_batch=args.threads_batch, device=args.device,
        gpu_layers=args.gpu_layers, p=args.p, cache_ram=args.cache_ram,
        points=list(args.lengths_list), n_predict=1, seed=args.seed,
        repeat_prompt=False, library_dir=[str(p) for p in args.library_dir],
        profile=True, trace=False, wavefront_tile=args.wavefront_tile,
        wavefront_tiles=args.wavefront_tiles, n256_lookahead=False,
        cuda_fp8=args.cuda_fp8, tcp_prefetch=args.tcp_prefetch,
        local_input_reuse=False, mask_pack='default', range_kernel=args.range_kernel,
        prefix_snapshot=str(args.prefix_snapshot),
        frozen_prefix_batch=args.frozen_prefix_batch, prefix_only=False,
        execute=True, dry_run=False, startup_timeout=args.startup_timeout,
        request_timeout=args.request_timeout,
    )


# ---------------------------------------------------------------------------
# Guard policy / environment
# ---------------------------------------------------------------------------

SAMPLE_RE = re.compile(
    r'getenv\("GGML_CUDA_GRAPH_GUARD_SAMPLE_MS"\);\s*'
    r'const int64_t ms = env != nullptr \? atoll\(env\) : (\d+);')
SLACK_RE = re.compile(
    r'getenv\("GGML_CUDA_GRAPH_GUARD_SLACK_MB"\);\s*'
    r'const long long mb = env != nullptr \? atoll\(env\) : (\d+);')
SAMPLE_DEFAULT = 10
SLACK_DEFAULT = 64


def parse_guard_policy(source_text):
    sample = SAMPLE_RE.search(source_text)
    slack = SLACK_RE.search(source_text)
    if not sample or not slack:
        raise GateError('unrecognized CUDA graph guard source; review the memory policy')
    return {'sample_ms_default': int(sample[1]), 'slack_mb_default': int(slack[1])}


def capture_guard_policy(repo):
    source = Path(repo) / 'ggml/src/ggml-cuda/common.cuh'
    try:
        text = source.read_text()
    except OSError as error:
        raise GateError(f'cannot read CUDA guard source: {source}') from error
    parsed = parse_guard_policy(text)
    if parsed['sample_ms_default'] != SAMPLE_DEFAULT or parsed['slack_mb_default'] != SLACK_DEFAULT:
        raise GateError(
            'CUDA graph guard defaults changed; refusing to pin a weaker policy '
            f'(sample_ms={parsed["sample_ms_default"]}, slack_mb={parsed["slack_mb_default"]})')
    inherited_sample = os.environ.get('GGML_CUDA_GRAPH_GUARD_SAMPLE_MS')
    inherited_slack = os.environ.get('GGML_CUDA_GRAPH_GUARD_SLACK_MB')
    if inherited_sample is not None:
        if not inherited_sample.isdecimal() or not inherited_sample.isascii():
            raise GateError('inherited GGML_CUDA_GRAPH_GUARD_SAMPLE_MS must be a nonnegative integer')
        if int(inherited_sample) < SAMPLE_DEFAULT:
            raise GateError(
                'inherited guard sampling is stricter than 10 ms; refusing to weaken it')
    if inherited_slack is not None:
        if not inherited_slack.isdecimal() or not inherited_slack.isascii():
            raise GateError('inherited GGML_CUDA_GRAPH_GUARD_SLACK_MB must be a nonnegative integer')
        if int(inherited_slack) > SLACK_DEFAULT:
            raise GateError('inherited guard slack is stricter than 64 MiB; refusing to weaken it')
    return {'variable_sample': 'GGML_CUDA_GRAPH_GUARD_SAMPLE_MS',
            'variable_slack': 'GGML_CUDA_GRAPH_GUARD_SLACK_MB',
            'sample_ms': SAMPLE_DEFAULT, 'slack_mb': SLACK_DEFAULT,
            'inherited_sample': inherited_sample, 'inherited_slack': inherited_slack,
            'source': str(source), 'source_sha256': sha256_bytes(text.encode('utf-8')),
            'rule': 'pin the current 10 ms / 64 MiB source defaults; never lower a stricter policy'}


def prepare_environment(helper, args, output, guard):
    """helper.environment() plus the quality-only in-memory overrides.

    The INT8 arm is a LocalState flag, not an environment variable, so it never
    enters the frozen-prefix contract.  F4/F5 callback variables are audit/test
    hooks; this gate disables them in the child while leaving the imported
    contract (which is about cache codec compatibility) untouched.
    """
    env, frozen = helper.environment(helper_args(args), output)
    for mapping in (env, frozen):
        for key in [k for k in mapping if k.startswith(('F4_', 'F5_'))]:
            mapping.pop(key, None)
    quality = {
        'GGML_CUDA_GRAPH_GUARD_SAMPLE_MS': str(guard['sample_ms']),
        'GGML_CUDA_GRAPH_GUARD_SLACK_MB': str(guard['slack_mb']),
        'LLAMA_SERVER_F5_PER_TOKEN_TIMING': '0',
    }
    env.update(quality)
    frozen.update(quality)
    return env, frozen


# ---------------------------------------------------------------------------
# Distribution parsing and metrics
# ---------------------------------------------------------------------------

def parse_distribution(response, vocab, mass_tolerance):
    """Return an ID-aligned probability vector from a full-vocab response."""
    if not isinstance(response, dict):
        raise GateError('completion response is not an object')
    if response.get('error'):
        raise GateError(f'server error: {str(response["error"])[:300]}')
    if response.get('truncated'):
        raise GateError('prompt was truncated')
    settings = response.get('generation_settings')
    if isinstance(settings, dict) and 'temperature' in settings and settings['temperature'] != 0:
        raise GateError('greedy temperature override was not effective')
    probabilities = response.get('completion_probabilities')
    if not isinstance(probabilities, list) or len(probabilities) != 1:
        raise GateError('completion_probabilities must contain exactly one token')
    entries = probabilities[0].get('top_logprobs')
    if not isinstance(entries, list) or len(entries) != vocab:
        raise GateError(
            f'full-vocab top_logprobs expected {vocab} entries, got '
            f'{len(entries) if isinstance(entries, list) else type(entries).__name__}')
    # One compact float32 vector per step; no Python float list of 250k items.
    values = array.array('f', [0.0]) * vocab
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise GateError('top_logprobs entry is not an object')
        token = entry.get('id')
        if type(token) is not int or not 0 <= token < vocab:
            raise GateError(f'invalid token id: {token!r}')
        if token in seen:
            raise GateError(f'duplicate token id: {token}')
        seen.add(token)
        logprob = entry.get('logprob')
        if logprob is None or isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
            raise GateError(f'token {token}: missing/non-numeric logprob')
        logprob = float(logprob)
        if math.isnan(logprob):
            raise GateError(f'token {token}: NaN logprob')
        if logprob == math.inf:
            raise GateError(f'token {token}: +inf logprob')
        if logprob <= SENTINEL_THRESHOLD:
            values[token] = 0.0
        elif logprob == -math.inf:
            values[token] = 0.0
        elif not math.isfinite(logprob):
            raise GateError(f'token {token}: non-finite logprob')
        elif logprob > 1e-6:
            raise GateError(f'token {token}: logprob {logprob} is greater than zero')
        else:
            values[token] = math.exp(logprob)
    if len(seen) != vocab:
        raise GateError(f'top_logprobs covers {len(seen)} of {vocab} ids')
    mass = math.fsum(values)
    if not math.isfinite(mass) or mass <= 0:
        raise GateError(f'invalid distribution mass: {mass}')
    if abs(mass - 1.0) > mass_tolerance:
        raise GateError(f'distribution mass {mass:.9f} is not close to 1')
    inverse = 1.0 / mass
    for token in range(vocab):
        values[token] = values[token] * inverse
    top1, top2, second = 0.0, 0.0, 0
    top_id = 0
    for token, value in enumerate(values):
        if value > top1:
            top2, second = top1, top_id
            top1, top_id = value, token
        elif value > top2:
            top2, second = value, token
    return {'probs': values, 'mass': mass, 'top1': top_id, 'top1_prob': top1,
            'top2': second, 'top2_prob': top2, 'margin': top1 - top2}



def distribution_view(probabilities):
    """Reconstruct the compact metrics view from a persisted probability list."""
    top1, top2, second, top_id = 0.0, 0.0, 0, 0
    for token, value in enumerate(probabilities):
        if value > top1:
            top2, second = top1, top_id
            top1, top_id = value, token
        elif value > top2:
            top2, second = value, token
    return {'probs': probabilities, 'mass': math.fsum(probabilities),
            'top1': top_id, 'top1_prob': top1, 'top2': second,
            'top2_prob': top2, 'margin': top1 - top2}


def teacher_nll(probabilities, teacher):
    probability = probabilities[teacher]
    if probability <= 0:
        return math.inf
    return -math.log(probability)


def kl_divergence(reference, candidate):
    total = 0.0
    for left, right in zip(reference, candidate):
        if left <= 0:
            continue
        if right <= 0:
            return math.inf
        total += left * (math.log(left) - math.log(right))
    return total


def step_metrics(off, on, teacher, high_confidence_margin):
    nll_off = teacher_nll(off['probs'], teacher)
    nll_on = teacher_nll(on['probs'], teacher)
    kl = kl_divergence(off['probs'], on['probs'])
    agreed = off['top1'] == on['top1']
    high_confidence = (not agreed and off['margin'] >= high_confidence_margin
                       and on['margin'] >= high_confidence_margin)
    return {
        'teacher_id': teacher,
        'nll_off': nll_off, 'nll_on': nll_on,
        'nll_delta': nll_on - nll_off,
        'kl': kl,
        'top1_off': off['top1'], 'top1_on': on['top1'],
        'top1_prob_off': off['top1_prob'], 'top1_prob_on': on['top1_prob'],
        'margin_off': off['margin'], 'margin_on': on['margin'],
        'top1_agree': agreed,
        'high_confidence_divergence': high_confidence,
    }


def nearest_rank(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(fraction * len(ordered))))
    return ordered[rank - 1]


def summarize_steps(steps):
    if not steps:
        return {'steps': 0, 'nll_off': None, 'nll_on': None, 'ppl_ratio': None,
                'nll_delta': None, 'kl_mean': None, 'kl_p99': None,
                'top1_agreement': None, 'high_confidence_divergences': 0}
    count = len(steps)
    triples = [(step.get('nll_off'), step.get('nll_on'), step.get('kl'))
               for step in steps]
    finite = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value)
        for triple in triples for value in triple)
    if finite:
        nll_off = math.fsum(triple[0] for triple in triples) / count
        nll_on = math.fsum(triple[1] for triple in triples) / count
        kls = [triple[2] for triple in triples]
        nll_delta = nll_on - nll_off
        ppl_ratio = math.exp(nll_delta)
        kl_mean = math.fsum(kls) / count
        kl_p99 = nearest_rank(kls, 0.99)
    else:
        nll_off = nll_on = nll_delta = ppl_ratio = kl_mean = kl_p99 = None
    return {
        'steps': count,
        'nll_off': nll_off, 'nll_on': nll_on,
        'nll_delta': nll_delta, 'ppl_ratio': ppl_ratio,
        'kl_mean': kl_mean, 'kl_p99': kl_p99,
        'top1_agreement': sum(1 for step in steps if step['top1_agree']) / count,
        'margin_off_mean': (math.fsum(step.get('margin_off', 0.0)
                                      for step in steps) / count),
        'margin_on_mean': (math.fsum(step.get('margin_on', 0.0)
                                     for step in steps) / count),
        'high_confidence_divergences': sum(
            1 for step in steps if step['high_confidence_divergence']),
        'finite': finite,
    }


# ---------------------------------------------------------------------------
# Deterministic case generator
# ---------------------------------------------------------------------------

WORD_POOLS = {
    'portuguese': [
        'análise', 'documento', 'processo', 'memória', 'sistema', 'resultado', 'amostra',
        'medida', 'contexto', 'servidor', 'resposta', 'pergunta', 'qualidade', 'variação',
        'controle', 'modelo', 'token', 'janela', 'execução', 'validação', 'preciso',
        'estável', 'local', 'remoto', 'rápido', 'lento', 'novo', 'antigo', 'grande', 'pequeno',
    ],
    'documents': [
        'SECAO', 'RESUMO', 'ITEM', 'TABELA', 'ANEXO', 'REFERENCIA', 'NOTA', 'REVISAO',
        'DOCUMENTO', 'PAGINA', 'TITULO', 'SUBTITULO', 'LISTA', 'CAMPO', 'VALOR', 'STATUS',
        'RESPONSAVEL', 'DATA', 'ORIGEM', 'DESTINO', 'OBSERVACAO', 'CRITERIO', 'ESCOPO',
    ],
    'code': [
        'def', 'return', 'value', 'index', 'result', 'cache', 'kernel', 'buffer', 'token',
        'layer', 'head', 'mask', 'offset', 'stride', 'count', 'limit', 'check', 'raise',
        'assert', 'range', 'enumerate', 'len', 'min', 'max', 'sum', 'append', 'extend',
    ],
    'numbers': [
        'registro', 'valor', 'total', 'media', 'delta', 'percentual', 'unidade', 'amostra',
        'limite', 'minimo', 'maximo', 'intervalo', 'contagem', 'soma', 'razao', 'indice',
    ],
    'retrieval': [
        'registro', 'chave', 'valor', 'consulta', 'indice', 'tabela', 'entrada', 'saida',
        'referencia', 'codigo', 'identificador', 'resposta', 'pergunta', 'contexto',
    ],
    'factual': [
        'fato', 'referencia', 'marcador', 'pergunta', 'resposta', 'contexto', 'nota',
        'registro', 'codigo', 'verificacao', 'exemplo', 'detalhe',
    ],
}
FILLER_CONNECTORS = ['de', 'para', 'com', 'sem', 'sobre', 'entre', 'após', 'antes', 'e', 'ou']
KEY_CANDIDATES = ['K7Q2', 'M4X9', 'R8T1', 'V2N6', 'B9F3', 'W5J8', 'C3P7', 'Z6H4']
VALUE_CANDIDATES = ['Q7F2', 'L3S8', 'N9D4', 'T5X1', 'G8R6', 'J2W7', 'P4C9', 'H6V3']
FACTUAL_PAIRS = [
    ('Qual é a capital da França?', 'Paris'),
    ('Quanto é 17 + 25?', '42'),
    ('Qual é o símbolo químico do ouro?', 'Au'),
    ('Qual é a capital do Japão?', 'Tóquio'),
]


def stable_index(seed, index, modulus):
    digest = hashlib.sha256(f'{seed}:{index}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little') % modulus


def sentence_words(category, seed, index, count=12):
    pool = WORD_POOLS[category]
    words = []
    for offset in range(count):
        words.append(pool[stable_index(seed, index * 131 + offset, len(pool))])
    return ' '.join(words)


def filler_text(category, seed, tokens_estimate):
    # Roughly 1.2 tokens per word in the target vocabularies; the tokenizer is
    # authoritative and the token stream is trimmed/extended to the exact
    # target length after /tokenize.
    sentences = []
    words = 0
    index = 0
    while words < max(64, tokens_estimate):
        if category == 'code':
            piece = (f'def fn_{index:05d}(value):\n'
                     f'    result = (value * {17 + index % 97} + {3 + index % 11}) % 1009\n'
                     f'    return result\n')
            words += 26
        elif category == 'documents':
            piece = (f'{sentence_words(category, seed, index, 4).upper()}\n'
                     f'- {sentence_words(category, seed, index + 1, 12)}\n')
            words += 18
        elif category == 'numbers':
            piece = (f'Registro {index:05d}: valor {1000 + index * 7},'
                     f'{index % 100:02d} unidades; delta {-17 + index % 35}%\n')
            words += 12
        else:
            piece = sentence_words(category, seed, index, 16) + '.\n'
            words += 17
        sentences.append(piece)
        index += 1
    return ''.join(sentences)


def _key_block(category, key, value):
    if category == 'factual':
        return f'Marcador único {key}. '
    return f'Registro único chave {key} valor {value}. '


def build_case_spec(name, category, length, seed, query=None, answer=None):
    return {
        'name': name,
        'category': category,
        'length': length,
        'seed': seed,
        'keys': KEY_CANDIDATES[:3],
        'values': VALUE_CANDIDATES[:3],
        'query': query,
        'answer': answer,
    }


def default_specs(args):
    if args.case_source[0] == 'external':
        return load_external_cases(args)
    names = args.case_source[1]
    specs = []
    for name in names:
        for length in args.lengths_list:
            if name == 'retrieval':
                query = (f'Pergunta: qual é o valor associado à chave {KEY_CANDIDATES[1]}? '
                         'Responda apenas com o valor. Resposta:')
                specs.append(build_case_spec(
                    f'{name}-{length}', 'retrieval', length, args.seed + length,
                    query=query, answer=VALUE_CANDIDATES[1]))
            elif name == 'factual':
                question, answer = FACTUAL_PAIRS[stable_index(
                    args.seed + length, 0, len(FACTUAL_PAIRS))]
                query = question + ' Responda apenas com a resposta. Resposta:'
                specs.append(build_case_spec(
                    f'{name}-{length}', 'factual', length, args.seed + length,
                    query=query, answer=answer))
            else:
                specs.append(build_case_spec(
                    f'{name}-{length}', name, length, args.seed + length))
    return specs


def load_external_cases(args):
    path = Path(args.cases_json).resolve(strict=True)
    document = read_json(path)
    cases = document.get('cases') if isinstance(document, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError('cases-json must contain a non-empty "cases" array')
    specs = []
    for index, item in enumerate(cases):
        if not isinstance(item, dict):
            raise ValueError(f'cases-json[{index}] is not an object')
        name = item.get('name')
        category = item.get('category')
        text = item.get('tail_text')
        if not isinstance(name, str) or not name:
            raise ValueError(f'cases-json[{index}] needs a non-empty name')
        if not isinstance(text, str) or not text:
            raise ValueError(f'cases-json[{index}] needs non-empty tail_text')
        if category not in CASE_CATEGORIES:
            raise ValueError(f'cases-json[{index}] category must be one of {CASE_CATEGORIES}')
        lengths = item.get('lengths')
        if lengths is None:
            lengths = [None]
        if not isinstance(lengths, list) or any(
                not (length is None or type(length) is int) for length in lengths):
            raise ValueError(f'cases-json[{index}] lengths must be integers or null')
        answer = item.get('answer')
        answer_ids = item.get('answer_ids')
        if answer_ids is not None and (not isinstance(answer_ids, list) or any(
                type(token) is not int or token < 0 for token in answer_ids)):
            raise ValueError(f'cases-json[{index}] answer_ids must be nonnegative integers')
        if category in ('retrieval', 'factual') and not (answer or answer_ids):
            raise ValueError(
                f'cases-json[{index}] {category} case needs answer or answer_ids')
        key = sha256_text(text)
        for length in lengths:
            specs.append({
                'name': name if length is None else f'{name}-{length}',
                'category': category,
                'length': length,
                'seed': args.seed,
                'text': text,
                'keys': [],
                'values': [],
                'query': None,
                'answer': answer,
                'answer_ids': answer_ids,
                'text_sha256': key,
            })
    return specs


def case_plan(args):
    specs = default_specs(args)
    plan = []
    for spec in specs:
        if spec.get('text') is not None:
            text_sha = spec.get('text_sha256') or sha256_text(spec['text'])
            plan.append({
                'name': spec['name'], 'category': spec['category'],
                'length': spec['length'], 'text_sha256': text_sha,
                'external': True, 'answer': spec.get('answer'),
                'answer_ids': spec.get('answer_ids'),
            })
        else:
            plan.append({
                'name': spec['name'], 'category': spec['category'],
                'length': spec['length'], 'seed': spec['seed'],
                'keys': spec['keys'], 'values': spec['values'],
                'query': spec.get('query'), 'answer': spec.get('answer'),
                'external': False,
                'text_sha256': sha256_text(canonical(spec)),
            })
    if not plan:
        raise ValueError('no cases selected')
    names = [item['name'] for item in plan]
    if len(names) != len(set(names)):
        raise ValueError('case names must be unique')
    return plan, specs


def render_category_text(spec, needed_tokens, tokenize):
    category = spec['category']
    seed = spec['seed']
    if category in ('retrieval', 'factual'):
        return render_fact_text(spec, needed_tokens, tokenize)
    text = filler_text(category, seed, needed_tokens)
    tokens = tokenize(text)
    if len(tokens) < needed_tokens:
        while len(tokens) < needed_tokens:
            text += filler_text(category, seed + len(text), needed_tokens)
            tokens = tokenize(text)
    return tokens[:needed_tokens], text, {}


def render_fact_text(spec, needed_tokens, tokenize):
    category = spec['category']
    seed = spec['seed']
    keys = spec['keys']
    values = spec['values']
    query = spec['query']
    query_tokens = tokenize(query)
    if len(query_tokens) >= 128:
        raise GateError('retrieval query is too long for the 128-token boundary')
    suffix_budget = 128 - len(query_tokens)
    end_filler = tokenize(filler_text(category, seed + 7, max(8, suffix_budget)))
    while len(end_filler) < suffix_budget:
        end_filler += tokenize(filler_text(category, seed + 11 + len(end_filler),
                                           max(8, suffix_budget)))
    end_filler = end_filler[:suffix_budget]
    if len(end_filler) != suffix_budget:
        raise GateError('cannot build the 128-token boundary suffix')
    start_key = tokenize(_key_block(category, keys[0], values[0]))
    mid_key = tokenize(_key_block(category, keys[1], values[1]))
    end_key = tokenize(_key_block(category, keys[2], values[2]))
    fixed = len(start_key) + len(mid_key) + len(end_key) + len(end_filler) + len(query_tokens)
    if fixed > needed_tokens:
        raise GateError(f'case {spec["name"]}: fixed fact blocks exceed tail length')
    filler_budget = needed_tokens - fixed
    filler = tokenize(filler_text(category, seed, max(32, filler_budget)))
    if len(filler) < filler_budget:
        while len(filler) < filler_budget:
            filler += tokenize(filler_text(category, seed + len(filler), max(32, filler_budget)))
    filler = filler[:filler_budget]
    middle = len(filler) // 2
    tokens = (start_key + filler[:middle] + mid_key + filler[middle:]
              + end_key + end_filler + query_tokens)
    if len(tokens) != needed_tokens:
        raise GateError(f'case {spec["name"]}: generated {len(tokens)} tokens, expected {needed_tokens}')
    text = (f'{_key_block(category, keys[0], values[0])}'
            f'[filler-{len(filler)}] '
            f'{_key_block(category, keys[1], values[1])}'
            f'[filler-{len(filler) - middle}] '
            f'{_key_block(category, keys[2], values[2])}'
            f'[boundary-{len(end_filler)}] {query}')
    metadata = {
        'start_key_tokens': len(start_key), 'mid_key_tokens': len(mid_key),
        'end_key_tokens': len(end_key), 'end_key_distance_tokens':
            len(end_filler) + len(query_tokens),
        'filler_tokens': len(filler), 'query_tokens': len(query_tokens),
    }
    if metadata['end_key_distance_tokens'] != 128:
        raise GateError('end key boundary is not exactly 128 tokens')
    return tokens, text, metadata


def fit_case_tokens(spec, base_length, tokenize, vocab):
    """Return suffix token IDs only; the shared base prefix is never touched."""
    length = spec['length']
    if length is None:
        if spec.get('text') is None:
            raise ValueError(f'case {spec["name"]} needs a length')
        tokens = tokenize(spec['text'])
        length = base_length + len(tokens)
    if length <= base_length:
        raise ValueError(f'case {spec["name"]} length must be > P')
    needed = length - base_length
    if spec.get('text') is not None:
        tokens = tokenize(spec['text'])
        if len(tokens) != needed:
            raise ValueError(
                f'external case {spec["name"]} tokenized to {len(tokens)}, expected {needed}; '
                'refusing to truncate or pad a real corpus silently')
        text = spec['text']
        metadata = {}
    else:
        tokens, text, metadata = render_category_text(spec, needed, tokenize)
    if len(tokens) != needed:
        raise ValueError(f'case {spec["name"]} has {len(tokens)} tokens, expected {needed}')
    bad = [token for token in tokens if type(token) is not int or not 0 <= token < vocab]
    if bad:
        raise ValueError(f'case {spec["name"]} produced out-of-vocab token {bad[0]!r}')
    answer_tokens = None
    answer_text = spec.get('answer')
    if spec.get('answer_ids') is not None:
        answer_tokens = list(spec['answer_ids'])
        if any(token < 0 or token >= vocab for token in answer_tokens):
            raise ValueError(f'case {spec["name"]} answer_ids outside vocab')
    elif answer_text:
        answer_tokens = tokenize(' ' + answer_text)
        if not answer_tokens:
            raise ValueError(f'case {spec["name"]} answer tokenized to nothing')
        if any(token < 0 or token >= vocab for token in answer_tokens):
            raise ValueError(f'case {spec["name"]} answer outside vocab')
    return {
        'name': spec['name'], 'category': spec['category'], 'length': length,
        'base_length': base_length, 'tail_length': needed,
        'tokens': tokens, 'tokens_sha256': token_hash(tokens),
        'text_sha256': sha256_text(text), 'metadata': metadata,
        'answer_text': answer_text, 'answer_tokens': answer_tokens,
        'query': spec.get('query'),
    }


class DistributionWriter:
    """Stream one float32 step at a time; never retain the full matrix in RAM."""

    def __init__(self, path, steps, vocab):
        if steps <= 0 or vocab <= 0:
            raise ValueError('distribution dimensions must be positive')
        self.path = Path(path)
        self.tmp = Path(str(self.path) + '.tmp')
        self.steps = steps
        self.vocab = vocab
        self.written = 0
        self.closed = False
        self.tmp.parent.mkdir(parents=True, exist_ok=True)
        self.raw = open(self.tmp, 'wb')
        self.gz = gzip.GzipFile(fileobj=self.raw, mode='wb', compresslevel=6, mtime=0)
        self.gz.write(DIST_MAGIC + DIST_HEADER.pack(steps, vocab))

    def append(self, values):
        if self.closed:
            raise GateError('distribution writer is closed')
        if len(values) != self.vocab:
            raise GateError('distribution width does not match vocab')
        if self.written >= self.steps:
            raise GateError('more distribution steps than declared')
        block = array.array('f', values)
        if sys.byteorder != 'little':
            block.byteswap()
        data = block.tobytes()
        self.gz.write(data)
        self.written += 1
        return sha256_bytes(data)

    def close(self):
        if self.closed:
            return
        if self.written != self.steps:
            self.abort()
            raise GateError(
                f'distribution stream has {self.written} steps, expected {self.steps}')
        self.gz.close()
        self.raw.close()
        self.tmp.replace(self.path)
        self.closed = True

    def abort(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.gz.close()
        except Exception:
            pass
        try:
            self.raw.close()
        except Exception:
            pass
        try:
            self.tmp.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.abort()
            return False
        self.close()
        return False


class DistributionReader:
    """Bounded streaming reader for a RKVA quality distribution file."""

    COMPRESSED_SLACK = 1 << 20

    def __init__(self, path, expected_steps, expected_vocab):
        self.path = Path(path)
        if expected_steps <= 0 or expected_vocab <= 0:
            raise GateError('distribution dimensions must be positive')
        self.steps = expected_steps
        self.vocab = expected_vocab
        self.index = 0
        self.reuses = 0
        self.max_buffers = 1
        self.buffer = array.array('f')
        expected_bytes = len(DIST_MAGIC) + DIST_HEADER.size + expected_steps * expected_vocab * 4
        try:
            size = self.path.stat().st_size
        except OSError as error:
            raise GateError(f'cannot stat distribution file: {self.path}') from error
        if size < len(DIST_MAGIC) + DIST_HEADER.size + 4:
            raise GateError(f'{self.path}: distribution file is too small')
        if size > expected_bytes + self.COMPRESSED_SLACK:
            raise GateError(f'{self.path}: distribution file is unexpectedly large')
        self.raw = open(self.path, 'rb')
        self.gz = gzip.GzipFile(fileobj=self.raw, mode='rb')
        try:
            header = self._read_exact(len(DIST_MAGIC) + DIST_HEADER.size)
        except BaseException:
            self._close_raw()
            raise
        if not header.startswith(DIST_MAGIC):
            self._close_raw()
            raise GateError(f'{self.path}: not a RKVA quality distribution file')
        steps, vocab = DIST_HEADER.unpack(
            header[len(DIST_MAGIC):len(DIST_MAGIC) + DIST_HEADER.size])
        if steps != expected_steps or vocab != expected_vocab:
            self._close_raw()
            raise GateError(
                f'{self.path}: header steps/vocab {steps}/{vocab} != '
                f'{expected_steps}/{expected_vocab}')

    def _read_exact(self, count):
        chunks = []
        remaining = count
        while remaining:
            try:
                chunk = self.gz.read(remaining)
            except (EOFError, OSError, gzip.BadGzipFile) as error:
                raise GateError(f'{self.path}: truncated/corrupt distribution stream') from error
            if not chunk:
                raise GateError(f'{self.path}: truncated distribution stream')
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def _close_raw(self):
        try:
            self.gz.close()
        except Exception:
            pass
        try:
            self.raw.close()
        except Exception:
            pass

    def read_step(self):
        if self.index >= self.steps:
            raise GateError('read past the declared distribution steps')
        data = self._read_exact(self.vocab * 4)
        # Replace the one reader-owned buffer; callers must consume it before
        # the next read_step(), which is exactly how the pairwise metrics loop
        # is written.
        self.buffer = array.array('f')
        self.buffer.frombytes(data)
        if sys.byteorder != 'little':
            self.buffer.byteswap()
        self.index += 1
        self.reuses += 1
        return self.buffer

    def close(self):
        if self.index != self.steps:
            self._close_raw()
            raise GateError(
                f'{self.path}: read {self.index} of {self.steps} distribution steps')
        try:
            extra = self.gz.read(1)
        except (EOFError, OSError, gzip.BadGzipFile) as error:
            self._close_raw()
            raise GateError(f'{self.path}: corrupt distribution stream') from error
        if extra:
            self._close_raw()
            raise GateError(f'{self.path}: trailing decompressed distribution data')
        try:
            trailing = self.raw.read(1)
        except OSError:
            trailing = b''
        if trailing:
            self._close_raw()
            raise GateError(f'{self.path}: trailing compressed distribution data')
        self._close_raw()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self._close_raw()
            return False
        self.close()
        return False

    def __iter__(self):
        while self.index < self.steps:
            yield self.read_step()


def save_distributions(path, distributions, vocab):
    """Small-test helper; production streams through DistributionWriter."""
    if not distributions:
        raise ValueError('no distributions to save')
    with DistributionWriter(path, len(distributions), vocab) as writer:
        for step in distributions:
            writer.append(step)


def load_distributions(path, expected_steps=None, expected_vocab=None):
    """Small-test helper; returns copies only for tests/audits."""
    if expected_steps is None or expected_vocab is None:
        # Header-only discovery is intentionally not exposed: production
        # callers always know the frozen step/vocab dimensions.
        with gzip.open(path, 'rb') as probe:
            header = probe.read(len(DIST_MAGIC) + DIST_HEADER.size)
        if not header.startswith(DIST_MAGIC):
            raise ValueError(f'{path}: not a RKVA quality distribution file')
        steps, vocab = DIST_HEADER.unpack(
            header[len(DIST_MAGIC):len(DIST_MAGIC) + DIST_HEADER.size])
        expected_steps = steps if expected_steps is None else expected_steps
        expected_vocab = vocab if expected_vocab is None else expected_vocab
    with DistributionReader(path, expected_steps, expected_vocab) as reader:
        return [array.array('f', step) for step in reader]


def verify_distribution_file(path, accounting, vocab, steps):
    if sha256_file(path) != accounting.get('distributions_sha256'):
        raise GateError(f'{path}: distribution file SHA does not match accounting')
    hashes = accounting.get('step_distribution_sha256')
    if not isinstance(hashes, list) or len(hashes) != steps:
        raise GateError(f'{path}: step hash metadata is incomplete')
    with DistributionReader(path, steps, vocab) as reader:
        for index, step in enumerate(reader):
            if step_digest(step) != hashes[index]:
                raise GateError(f'{path}: step {index} hash does not match accounting')


def step_digest(distribution):
    values = array.array('f', distribution)
    if sys.byteorder != 'little':
        values.byteswap()
    return sha256_bytes(values.tobytes())


def step_hashes(distributions):
    return [step_digest(step) for step in distributions]


def distribution_sha(path):
    return sha256_file(path)


# ---------------------------------------------------------------------------
# RPC profile parsing / activation proof
# ---------------------------------------------------------------------------

def deep_get(document, *keys):
    value = document
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def profile_from_lines(lines):
    """Return the last engine profile object from remote_attn gpu_profile lines."""
    for line in reversed(list(lines)):
        index = line.find(PROFILE_MARKER)
        if index < 0:
            continue
        try:
            document = json.loads(line[index + len(PROFILE_MARKER):])
        except (ValueError, TypeError):
            continue
        engine = document.get('engine') if isinstance(document, dict) else None
        if isinstance(engine, dict):
            return engine
    return None


def activation_checks(engine_off, engine_on, lines_off, lines_on, helper):
    """Fail-closed proof that both arms really exercised the intended paths."""
    def integer(engine, *keys):
        value = deep_get(engine, *keys)
        return value if type(value) is int else None

    checks = {}
    off_i8 = integer(engine_off, 'kvarn4_multi_query', 'mq2_i8', 'calls')
    on_i8 = integer(engine_on, 'kvarn4_multi_query', 'mq2_i8', 'calls')
    on_i8_available = deep_get(engine_on, 'kvarn4_multi_query', 'mq2_i8', 'available')
    on_i8_exact = deep_get(engine_on, 'kvarn4_multi_query', 'mq2_i8', 'exact')
    off_mq2 = integer(engine_off, 'kvarn4_multi_query', 'mq2', 'calls')
    checks['off_int8_calls'] = off_i8
    checks['on_int8_calls'] = on_i8
    checks['off_int8_zero'] = off_i8 == 0
    checks['on_int8_positive'] = type(on_i8) is int and on_i8 > 0
    checks['on_int8_available'] = on_i8_available is True
    checks['on_int8_non_exact'] = on_i8_exact is False
    checks['off_multiquery_calls'] = off_mq2
    checks['off_multiquery_positive'] = type(off_mq2) is int and off_mq2 > 0

    off_decode = integer(engine_off, 'kvarn4_decode', 'calls')
    on_decode = integer(engine_on, 'kvarn4_decode', 'calls')
    checks['off_decode_calls'] = off_decode
    checks['on_decode_calls'] = on_decode
    checks['off_decode_positive'] = type(off_decode) is int and off_decode > 0
    checks['on_decode_positive'] = type(on_decode) is int and on_decode > 0
    off_rowlane = integer(engine_off, 'kvarn4_decode_rowlane', 'calls')
    on_rowlane = integer(engine_on, 'kvarn4_decode_rowlane', 'calls')
    off_rowlane_enabled = deep_get(engine_off, 'kvarn4_decode_rowlane', 'enabled')
    on_rowlane_enabled = deep_get(engine_on, 'kvarn4_decode_rowlane', 'enabled')
    checks['off_rowlane_calls'] = off_rowlane
    checks['on_rowlane_calls'] = on_rowlane
    checks['off_rowlane_positive'] = type(off_rowlane) is int and off_rowlane > 0
    checks['on_rowlane_positive'] = type(on_rowlane) is int and on_rowlane > 0
    checks['off_rowlane_enabled'] = off_rowlane_enabled is True
    checks['on_rowlane_enabled'] = on_rowlane_enabled is True

    off_gpu = integer(engine_off, 'bulk', 'fp8_gpu_calls')
    on_gpu = integer(engine_on, 'bulk', 'fp8_gpu_calls')
    off_fallback = integer(engine_off, 'bulk', 'fp8_cpu_fallback_calls')
    on_fallback = integer(engine_on, 'bulk', 'fp8_cpu_fallback_calls')
    off_segments = integer(engine_off, 'bulk', 'bulk_segments')
    on_segments = integer(engine_on, 'bulk', 'bulk_segments')
    off_queries = integer(engine_off, 'bulk', 'bulk_queries')
    on_queries = integer(engine_on, 'bulk', 'bulk_queries')
    checks['off_fp8_gpu_calls'] = off_gpu
    checks['on_fp8_gpu_calls'] = on_gpu
    checks['off_fp8_gpu_positive'] = type(off_gpu) is int and off_gpu > 0
    checks['on_fp8_gpu_positive'] = type(on_gpu) is int and on_gpu > 0
    checks['off_fp8_cpu_fallback_zero'] = off_fallback == 0
    checks['on_fp8_cpu_fallback_zero'] = on_fallback == 0
    checks['off_bulk_segments_positive'] = type(off_segments) is int and off_segments > 0
    checks['on_bulk_segments_positive'] = type(on_segments) is int and on_segments > 0
    checks['off_bulk_queries_positive'] = type(off_queries) is int and off_queries > 0
    checks['on_bulk_queries_positive'] = type(on_queries) is int and on_queries > 0

    def bucket_samples(engine):
        buckets = engine.get('buckets') if isinstance(engine, dict) else None
        if not isinstance(buckets, dict):
            return 0
        total = 0
        for key, bucket in buckets.items():
            if 'nq2-16' not in key or not isinstance(bucket, dict):
                continue
            samples = bucket.get('samples')
            if type(samples) is int and samples > 0:
                total += samples
        return total

    off_nq = bucket_samples(engine_off)
    on_nq = bucket_samples(engine_on)
    checks['off_nq2_16_samples'] = off_nq
    checks['on_nq2_16_samples'] = on_nq
    checks['off_nq2_16_positive'] = off_nq > 0
    checks['on_nq2_16_positive'] = on_nq > 0

    off_wf = helper.wavefront_activation(lines_off, WAVEFRONT_TILE, WAVEFRONT_TILES)
    on_wf = helper.wavefront_activation(lines_on, WAVEFRONT_TILE, WAVEFRONT_TILES)
    checks['off_wavefront_confirmed'] = off_wf['requested_geometry_confirmed']
    checks['on_wavefront_confirmed'] = on_wf['requested_geometry_confirmed']
    checks['off_wavefront'] = off_wf
    checks['on_wavefront'] = on_wf
    off_tcp = helper.tcp_prefetch_activation(lines_off)
    on_tcp = helper.tcp_prefetch_activation(lines_on)
    checks['off_tcp_overlap_confirmed'] = off_tcp['two_frame_overlap_confirmed']
    checks['on_tcp_overlap_confirmed'] = on_tcp['two_frame_overlap_confirmed']
    checks['off_tcp'] = off_tcp
    checks['on_tcp'] = on_tcp

    required = [
        'off_int8_zero', 'on_int8_positive', 'on_int8_available', 'on_int8_non_exact',
        'off_multiquery_positive', 'off_decode_positive', 'on_decode_positive',
        'off_nq2_16_positive', 'on_nq2_16_positive',
        'off_rowlane_positive', 'on_rowlane_positive',
        'off_rowlane_enabled', 'on_rowlane_enabled', 'off_fp8_gpu_positive',
        'on_fp8_gpu_positive', 'off_fp8_cpu_fallback_zero', 'on_fp8_cpu_fallback_zero',
        'off_bulk_segments_positive', 'on_bulk_segments_positive',
        'off_bulk_queries_positive', 'on_bulk_queries_positive',
        'off_wavefront_confirmed', 'on_wavefront_confirmed',
        'off_tcp_overlap_confirmed', 'on_tcp_overlap_confirmed',
    ]
    checks['failed'] = [name for name in required if not checks[name]]
    checks['pass'] = not checks['failed'] and engine_off is not None and engine_on is not None
    checks['reason'] = ('all requested runtime paths proved' if checks['pass'] else
                        'missing/unproved activation: ' + ', '.join(checks['failed']))
    return checks


# ---------------------------------------------------------------------------
# Xbox LocalState flags through deploy.sh
# ---------------------------------------------------------------------------

def load_xbox_env(path):
    """Parse KEY=VALUE / export KEY=VALUE without ever logging the values."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise RefuseError(f'Xbox credential file not found: {path}')
    values = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].strip()
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    for key in ('XBOX_IP', 'XBOX_USER', 'XBOX_PASS'):
        if not values.get(key):
            raise RefuseError(f'Xbox credential file is missing {key}')
    return values


def redact(text, secrets):
    result = str(text)
    for secret in secrets:
        # A very short secret cannot be redacted without corrupting ordinary
        # words; credential values used here are longer than this bound and
        # are never printed in the first place.
        if secret and len(secret) >= 4:
            result = result.replace(secret, '***')
    return result


def validate_pfn(pfn, stderr=''):
    pfn = (pfn or '').strip()
    if not pfn or '\n' in pfn or not pfn.startswith(PFN_PREFIX):
        raise RefuseError(f'installed package is not the expected {PFN_PREFIX}* identity')
    if 'packages registered' in (stderr or ''):
        raise RefuseError('multiple xllama packages are registered; refusing an ambiguous PFN')
    return pfn


class DeployClient:
    def __init__(self, path, env_path, timeout, xbox_ip=None, runner=None):
        self.path = Path(path)
        self.env_path = Path(env_path)
        self.timeout = timeout
        self.xbox_ip = xbox_ip
        self.runner = runner or self._subprocess_runner
        self._secrets = []

    def environment(self):
        values = dict(os.environ)
        loaded = load_xbox_env(self.env_path)
        values.update(loaded)
        if self.xbox_ip:
            # The deploy target must be the same console as --xbox-host, even
            # when the credential file still names an older address.
            values['XBOX_IP'] = self.xbox_ip
        self._secrets = [loaded.get('XBOX_PASS', '')]
        return values

    def _subprocess_runner(self, argv, env, timeout):
        try:
            return subprocess.run(argv, env=env, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise GateError(f'deploy command timed out: {argv[1]}') from error

    def run(self, *argv, env=None):
        env = env or self.environment()
        result = self.runner([str(self.path), *map(str, argv)], env, self.timeout)
        # Keep the complete stdout/stderr only for internal parsing; every
        # audit-facing copy is redacted and bounded.  In particular, never
        # truncate before json.loads() in list_names().
        return {
            'argv': [str(self.path), *map(str, argv)],
            'returncode': result.returncode,
            '_stdout_full': result.stdout or '',
            '_stderr_full': result.stderr or '',
            'stdout': redact(result.stdout or '', self._secrets)[-4000:],
            'stderr': redact(result.stderr or '', self._secrets)[-4000:],
        }

    def pfn(self):
        result = self.run('pfn')
        if result['returncode'] != 0:
            raise RefuseError('deploy.sh pfn failed: ' + result['stderr'][-300:])
        lines = [line.strip() for line in result['_stdout_full'].splitlines()
                 if line.strip()]
        if len(lines) != 1:
            raise RefuseError('deploy.sh pfn did not return exactly one package')
        return validate_pfn(lines[0], result['stderr'])

    def list_names(self, pfn):
        result = self.run('list-localstate', pfn)
        if result['returncode'] != 0:
            raise GateError('deploy.sh list-localstate failed: ' + result['stderr'][-300:])
        try:
            document = json.loads(result['_stdout_full'])
        except ValueError as error:
            raise GateError('deploy.sh list-localstate returned non-JSON') from error
        items = document.get('Items') if isinstance(document, dict) else document
        if not isinstance(items, list):
            raise GateError('deploy.sh list-localstate returned an unexpected shape')
        names = set()
        for item in items:
            if isinstance(item, dict) and isinstance(item.get('Name'), str):
                names.add(item['Name'])
        return names


class LocalStateFlags:
    """Temporary experimental-flag owner with exact original-state restoration."""

    def __init__(self, deploy, pfn, output, names=MANAGED_FLAGS):
        self.deploy = deploy
        self.pfn = pfn
        self.names = tuple(names)
        self.output = Path(output)
        self.original = None
        self._empty = self.output / 'flags' / 'empty.flag'
        self._empty.parent.mkdir(parents=True, exist_ok=True)
        self._empty.write_bytes(b'')

    def read(self):
        return self.deploy.list_names(self.pfn)

    def capture_original(self):
        present = self.read()
        self.original = {name: name in present for name in self.names}
        write_json(self.output / 'flags-original.json', {
            'pfn': self.pfn, 'names': self.names, 'present': self.original,
            'captured': now(),
        })
        return dict(self.original)

    def adopt_original(self, present):
        """Adopt a persisted original-state map, migrating older two-flag runs.

        A run started before GQA2 became managed has no key for it. Read the
        current device state once and persist the complete map so the existing
        restore/verify path stays exact and old resumable runs remain valid.
        """
        if not isinstance(present, dict):
            raise GateError('stored original flag state is invalid')
        current = self.read()
        merged = {name: bool(present[name]) if name in present else name in current
                  for name in self.names}
        if merged != present:
            write_json(self.output / 'flags-original.json', {
                'pfn': self.pfn, 'names': self.names, 'present': merged,
                'captured': now(), 'migrated': now(),
            })
        self.original = merged
        return dict(merged)

    def set(self, name, present):
        if name not in self.names:
            raise ValueError(f'unmanaged flag: {name}')
        if present:
            result = self.deploy.run('upload-file', str(self._empty), self.pfn, '', name)
            if result['returncode'] != 0:
                raise GateError(f'deploy.sh upload-file {name} failed')
        else:
            result = self.deploy.run('delete-file', self.pfn, name)
            if result['returncode'] != 0:
                # A delete of an absent file is idempotent success only for an
                # explicit 404 confirmed by a fresh listing; any other failure
                # (403, transport, server error) is fatal.
                if '404' not in (result.get('stderr') or ''):
                    raise GateError(f'deploy.sh delete-file {name} failed')
                if name in self.read():
                    raise GateError(f'deploy.sh delete-file {name} failed and the flag remains')
        if (name in self.read()) != present:
            raise GateError(f'flag {name} did not reach the requested state')
        return present

    def apply(self, int8_on, gqa2_on=False):
        if int8_on and gqa2_on:
            # The engine treats this pair as a fail-closed conflict; never arm
            # an ambiguous control arm in the harness either.
            raise ValueError('INT8 and GQA2 flags are mutually exclusive')
        self.set(INT8_FLAG, int8_on)
        self.set(GQA2_FLAG, gqa2_on)
        # rowlane must be ON: the opt-out flag is absent.
        self.set(ROWLAND_OPT_OUT_FLAG, False)

    def restore(self):
        if self.original is None:
            raise GateError('original flag state was never captured')
        for name, present in self.original.items():
            self.set(name, present)
        self.verify()

    def verify(self):
        present = self.read()
        actual = {name: name in present for name in self.names}
        if self.original is None or actual != self.original:
            raise GateError(
                'LocalState flags were not restored to their original state: '
                f'{actual} != {self.original}')
        write_json(self.output / 'flags-restored.json', {
            'pfn': self.pfn, 'present': actual, 'verified': now(),
        })
        return actual

    @contextlib.contextmanager
    def arm(self, int8_on, gqa2_on=False):
        try:
            self.apply(int8_on, gqa2_on)
            yield
        finally:
            self.restore()


# ---------------------------------------------------------------------------
# Owned server lifecycle
# ---------------------------------------------------------------------------

def foreign_process_rows(proc_root='/proc'):
    rows = []
    root = Path(proc_root)
    if not root.is_dir():
        return rows
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            comm = (entry / 'comm').read_text().strip()
        except (OSError, UnicodeDecodeError):
            continue
        try:
            cmdline = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(
                'utf-8', 'replace').strip()
        except OSError:
            cmdline = ''
        rows.append({'pid': pid, 'comm': comm, 'cmdline': cmdline})
    return rows


def self_pid_chain(rows, pid=None):
    pid = os.getpid() if pid is None else pid
    parent = {row['pid']: 0 for row in rows}
    for row in rows:
        try:
            stat = Path(f"/proc/{row['pid']}/stat").read_text()
            fields = stat.rsplit(')', 1)[1].split()
            parent[row['pid']] = int(fields[1])
        except (OSError, ValueError, IndexError):
            parent[row['pid']] = 0
    chain = set()
    current = pid
    while current and current not in chain:
        chain.add(current)
        current = parent.get(current, 0)
    return chain


def foreign_process_matches(rows, ignored=()):
    ignored = set(ignored) | self_pid_chain(rows)
    matches = []
    for row in rows:
        if row['pid'] in ignored:
            continue
        comm = row['comm']
        cmdline = row['cmdline']
        if comm in FOREIGN_COMMS or comm.startswith(FOREIGN_COMM_PREFIXES):
            matches.append(row)
        elif any(hint in cmdline for hint in FOREIGN_SCRIPT_HINTS):
            matches.append(row)
    return matches


def loaded_library_paths(pid):
    """Return loaded .so paths from /proc without hashing their contents."""
    maps = Path(f'/proc/{pid}/maps')
    if not maps.exists():
        raise GateError('loaded-library provenance currently requires Linux /proc/PID/maps')
    paths = set()
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/') and '.so' in fields[5]:
            path = Path(fields[5])
            if not path.is_file():
                raise GateError(f'loaded library no longer exists: {path}')
            paths.add(str(path.resolve()))
    return sorted(paths)


class ServerSession:
    """One private llama-server child; no global service is touched."""

    def __init__(self, helper, helper_ns, args, output, guard, name,
                 popen=None, http=None, drain_factory=None, loaded_libraries=None):
        self.helper = helper
        self.helper_ns = helper_ns
        self.args = args
        self.output = Path(output)
        self.guard = guard
        self.name = name
        self.popen = popen or subprocess.Popen
        self.http = http or helper.http_json
        self.drain_factory = drain_factory or helper.LogDrain
        self.loaded_libraries = loaded_libraries or loaded_library_paths
        self.process = None
        self.drain = None
        self.port = None
        self.env = None
        self.frozen_env = None
        self.command = None
        self.manifest = {'name': name, 'started': now(), 'schema': 'rkva-quality-gate-arm-v1'}
        self.errors = []
        self.stop_error = None

    def __enter__(self):
        try:
            self._start()
        except BaseException:
            # __exit__ is not called when __enter__ fails: never leave the
            # child behind on a startup timeout/health failure.
            try:
                self._stop()
            finally:
                self._finalize()
            raise
        return self

    def _start(self):
        self.output.mkdir(parents=True, exist_ok=True)
        if any(self.output.iterdir()):
            owner = self.output / 'owner.json'
            if not owner.is_file() or read_json(owner).get('owner') != 'rkva-quality-gate':
                raise RefuseError(f'{self.output}: non-empty server dir is not owned by this tool')
        else:
            write_json(self.output / 'owner.json',
                       {'owner': 'rkva-quality-gate', 'created': now()})
        self.env, self.frozen_env = prepare_environment(
            self.helper, self.args, self.output, self.guard)
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            self.port = reservation.getsockname()[1]
        self.command = self.helper.server_command(self.helper_ns, self.port) + ['--no-warmup']
        self.manifest.update({
            'port': self.port, 'command': self.command, 'environment': self.frozen_env,
            'helper_args': vars(self.helper_ns), 'guard_policy': self.guard,
            'host': {'hostname': socket.gethostname(), 'platform': platform.platform()},
        })
        write_json(self.output / 'manifest.json', self.manifest)
        self.process = self.popen(self.command, env=self.env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                  errors='replace', bufsize=1, start_new_session=True)
        self.manifest['pid'] = self.process.pid
        self.drain = self.drain_factory(self.process, self.output / 'server.log', self.port)
        if not self.drain.ready.wait(self.args.startup_timeout) or not self.drain.listening:
            raise GateError('owned server did not emit its listening URL before timeout/exit')
        self.manifest['health'] = self.http(self.port, '/health', None, 60)
        self.manifest['props'] = self.http(self.port, '/props', None, 60)
        self.manifest['models'] = self.http(self.port, '/v1/models', None, 60)
        self.manifest['loaded_libraries'] = self.loaded_libraries(self.process.pid)
        self.manifest['startup_warnings'] = self.helper.validate_startup(
            self.helper_ns, self.drain.lines)
        self.manifest['negotiation'] = [
            line.strip() for line in self.drain.lines
            if re.search(r'connected to|native KVarN4|position-split|wire=|descriptor=', line)]
        write_json(self.output / 'manifest.json', self.manifest)

    def request(self, endpoint, body=None, timeout=None):
        if self.process.poll() is not None:
            raise GateError(f'owned server exited before {endpoint} ({self.process.returncode})')
        return self.http(self.port, endpoint, body, timeout or self.args.request_timeout)

    def n_vocab(self):
        models = self.manifest.get('models') or {}
        data = models.get('data') if isinstance(models, dict) else None
        value = None
        if isinstance(data, list) and data and isinstance(data[0], dict):
            value = deep_get(data[0], 'meta', 'n_vocab')
        if type(value) is not int or value <= 0:
            value = deep_get(self.manifest.get('props') or {}, 'model', 'meta', 'n_vocab')
        if type(value) is not int or value <= 0:
            raise GateError('server /v1/models did not expose a positive model n_vocab')
        return value

    def profile_lines(self):
        return [line.rstrip() for line in self.drain.lines if PROFILE_FILTER.search(line)]

    def __exit__(self, exc_type, exc, traceback):
        try:
            self._stop()
        finally:
            self._finalize()
        if exc_type is not None:
            return False
        if self.errors:
            raise GateError('; '.join(self.errors))
        return False

    def _finalize(self):
        try:
            if self.drain is not None:
                self.drain.thread.join(timeout=5)
                if self.drain.thread.is_alive():
                    self.errors.append('stdout reader did not finish after child exit')
                errors, warnings = self.helper.classify_logs(self.drain.lines)
                self.manifest['server_errors'] = errors
                self.manifest['memory_policy_warnings'] = warnings
                if errors:
                    self.errors.append('server logged an error/fallback: ' + '; '.join(errors[:3]))
                write_json(self.output / 'profile-lines.json', self.profile_lines())
                self.manifest['profile'] = profile_from_lines(self.drain.lines)
        finally:
            self.manifest['finished'] = now()
            self.manifest['errors'] = self.errors
            self.manifest['child_exit_code'] = (
                None if self.process is None else self.process.returncode)
            write_json(self.output / 'manifest.json', self.manifest)

    def _stop(self):
        process = self.process
        if process is None:
            return
        if process.poll() is not None:
            self.manifest['loaded_libraries_final_error'] = (
                'process exited before final library provenance')
        else:
            try:
                self.manifest['loaded_libraries_final'] = self.loaded_libraries(process.pid)
                self.manifest['loaded_libraries_final_error'] = None
            except Exception as error:  # provenance must not mask the real result
                self.manifest['loaded_libraries_final_error'] = str(error)
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired as error:
                    raise GateError('owned server could not be reaped') from error
        code = process.returncode
        if code != 0:
            self.errors.append(f'owned server did not exit 0 after SIGTERM (exit {code})')
        self.manifest['sigterm_exit'] = code


# ---------------------------------------------------------------------------
# Binding / resumable output
# ---------------------------------------------------------------------------

def source_identity(helper):
    try:
        return helper.git_identity()
    except Exception as error:  # helper raises RuntimeError with a useful message
        raise GateError(f'cannot freeze repository source identity: {error}') from error


LLAMA_GGML_NAME = re.compile(r'(llama|ggml)', re.I)


def active_library_paths(server_binary):
    """Resolve the actually linked LLAMA/GGML shared objects via ldd.

    H.local_libraries() hashes *every* versioned file matching libllama*/libggml*
    in the install directory, including stale multi-hundred-MB CUDA builds that
    the server never loads.  ldd is the robust equivalent for the active
    DT_NEEDED set; it does not start the model or touch the GPU.
    """
    try:
        result = subprocess.run(['ldd', str(server_binary)], capture_output=True,
                                text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GateError(f'cannot resolve shared libraries with ldd: {error}') from error
    paths = set()
    for line in (result.stdout or '').splitlines():
        if 'not found' in line and LLAMA_GGML_NAME.search(line):
            raise GateError(f'missing shared library: {line.strip()}')
        match = re.search(r'=>\s+(\S+)', line)
        if match is None:
            match = re.match(r'\s*(\S+)\s+\(0x', line)
        if match is None:
            continue
        path = match.group(1)
        if not LLAMA_GGML_NAME.search(Path(path).name):
            continue
        try:
            resolved = Path(path).resolve(strict=True)
        except OSError as error:
            raise GateError(f'cannot resolve shared library {path}') from error
        if not resolved.is_file():
            raise GateError(f'resolved shared library is not a file: {resolved}')
        paths.add(resolved)
    return sorted(paths)


def build_library_binding(server_binary, paths=None):
    paths = active_library_paths(server_binary) if paths is None else list(paths)
    files = []
    stats = {}
    for path in sorted({Path(item).resolve(strict=True) for item in paths}):
        stat = path.stat()
        files.append({'path': str(path), 'size': stat.st_size,
                      'sha256': sha256_file(path)})
        stats[str(path)] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                            'ctime_ns': stat.st_ctime_ns, 'inode': stat.st_ino}
    return {'files': files, 'stats': stats,
            'sha256': sha256_text(canonical(files))}


class LibraryGuard:
    """Fail-closed active/loaded library checks without re-hashing GB per step."""

    def __init__(self, server_binary, binding, path_provider=None):
        self.server_binary = server_binary
        self.binding = binding
        self.files = {item['path']: item for item in binding['files']}
        self.stats = {key: dict(value) for key, value in binding.get('stats', {}).items()}
        self.path_provider = path_provider or active_library_paths
        for path in self.files:
            if path not in self.stats:
                stat = Path(path).stat()
                self.stats[path] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                    'ctime_ns': stat.st_ctime_ns, 'inode': stat.st_ino}

    def _verify_path(self, path, phase):
        path = Path(path).resolve()
        key = str(path)
        expected = self.files.get(key)
        if expected is None:
            raise GateError(f'{phase}: relevant library loaded outside the frozen binding: {key}')
        try:
            stat = path.stat()
        except OSError as error:
            raise GateError(f'{phase}: bound library disappeared: {key}') from error
        if stat.st_size != expected['size']:
            raise GateError(f'{phase}: bound library size changed: {key}')
        recorded = self.stats.get(key)
        if (recorded is None or stat.st_mtime_ns != recorded.get('mtime_ns')
                or stat.st_ctime_ns != recorded.get('ctime_ns')
                or stat.st_ino != recorded.get('inode')):
            if sha256_file(path) != expected['sha256']:
                raise GateError(f'{phase}: bound library content changed: {key}')
            self.stats[key] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                               'ctime_ns': stat.st_ctime_ns, 'inode': stat.st_ino}

    def verify_local(self, phase):
        current = {str(Path(item).resolve()) for item in self.path_provider(self.server_binary)}
        if current != set(self.files):
            raise GateError(f'{phase}: active LLAMA/GGML library set changed')
        for path in sorted(self.files):
            self._verify_path(path, phase)

    def verify_loaded(self, loaded_paths, phase):
        for path in loaded_paths or []:
            name = Path(path).name
            if not LLAMA_GGML_NAME.search(name):
                continue
            self._verify_path(path, phase)

    def require_loaded(self, loaded_paths, phase):
        relevant = [path for path in (loaded_paths or [])
                    if LLAMA_GGML_NAME.search(Path(path).name)]
        if self.files and not relevant:
            raise GateError(f'{phase}: no relevant LLAMA/GGML library was loaded')


def compute_binding(args, plan, source, binary_stat, model_stat, prefix_stat,
                    snapshot_meta, guard, gates, libraries=None):
    """Tokenizer-independent binding; frozen before any child starts."""
    snapshot = None
    if snapshot_meta is not None and snapshot_meta.get('metadata') is not None:
        snapshot = {
            'meta_sha256': snapshot_meta['meta_sha256'],
            'actual_prefix_tokens': snapshot_meta['metadata']['actual_prefix_tokens'],
            'actual_prefix_sha256': snapshot_meta['metadata']['actual_prefix_sha256'],
            'requested_prefix_sha256': snapshot_meta['metadata']['contract']['requested_prefix_sha256'],
            'model_sha256': snapshot_meta['metadata']['contract']['model_sha256'],
            'P': snapshot_meta['metadata']['contract']['P'],
            'context': snapshot_meta['metadata']['contract']['context'],
            'batch': snapshot_meta['metadata']['contract']['batch'],
            'files': snapshot_meta['metadata']['files'],
        }
    binding = {
        'schema': 'rkva-quality-gate-binding-v1',
        'source': {'head': source['head'], 'tracked_diff_sha256': source['tracked_diff_sha256'],
                   'branch': source.get('branch')},
        'binary': {key: binary_stat[key] for key in ('path', 'size', 'sha256')},
        'model': {key: model_stat[key] for key in ('path', 'size', 'sha256')},
        'prefix': snapshot,
        'prefix_intent': (snapshot_meta or {}).get('intent'),
        'prefix_path': str(prefix_stat.get('path')),
        'geometry': {
            'P': args.p, 'context': args.context, 'batch': args.batch,
            'ubatch': args.ubatch, 'frozen_prefix_batch': args.frozen_prefix_batch,
            'wavefront_tile': args.wavefront_tile, 'wavefront_tiles': args.wavefront_tiles,
            'cuda_fp8': args.cuda_fp8, 'range_kernel': args.range_kernel,
            'tcp_prefetch': args.tcp_prefetch, 'local_input_reuse': False,
            'mask_pack': 'default',
        },
        'guard': {key: guard[key] for key in ('sample_ms', 'slack_mb', 'source_sha256')},
        'protocol': {
            'teacher_steps': args.teacher_steps, 'smoke': args.smoke,
            'determinism_control': args.determinism_control, 'seed': args.seed,
            'n_predict': 1, 'post_sampling_probs': False, 'cache_prompt': True,
            'temperature': 0.0, 'n_probs': 'model_n_vocab',
            'local_input_reuse': False, 'mask_pack': 'default',
        },
        'libraries': {
            'files': (libraries or {}).get('files', []),
            'sha256': (libraries or {}).get(
                'sha256', sha256_text(canonical([]))),
        },
        'gates': gates,
        'gate_sha256': sha256_text(canonical(gates)),
        'case_plan_sha256': sha256_text(canonical(plan)),
        'tool_sha256': sha256_file(Path(__file__).resolve()),
    }
    binding['binding_sha256'] = sha256_text(canonical(binding))
    return binding


def prepare_output(args, binding):
    out = args.output
    if not out.exists():
        out.mkdir(parents=True, exist_ok=False)
        atomic_write_bytes(out / 'binding.json', (canonical(binding) + '\n').encode('utf-8'))
        return {'mode': 'new'}
    if not out.is_dir():
        raise RefuseError(f'output exists and is not a directory: {out}')
    binding_path = out / 'binding.json'
    if not binding_path.is_file():
        raise RefuseError(
            f'output {out} exists without binding.json; refusing to overwrite unknown artifacts')
    existing = read_json(binding_path)
    comparison = binding
    if existing.get('prefix') is None and existing.get('prefix_intent') == 'create':
        # A run that deliberately created its prefix has no prefix identity in
        # its first binding.  Once the snapshot exists, resume must not look
        # like a different configuration.
        comparison = {key: value for key, value in binding.items()
                      if key != 'binding_sha256'}
        comparison['prefix'] = None
        comparison['prefix_intent'] = 'create'
        comparison['binding_sha256'] = sha256_text(canonical(comparison))
    if existing.get('binding_sha256') != comparison['binding_sha256']:
        raise RefuseError(
            'output binding differs (source/binary/model/prefix/geometry/gates/case plan); '
            'use a new --output instead of overwriting')
    return {'mode': 'resume'}


def complete_marker(path):
    marker = Path(path) / 'complete.json'
    if not marker.is_file():
        return None
    document = read_json(marker)
    if document.get('owner') != 'rkva-quality-gate':
        raise RefuseError(f'{path}: complete.json is not owned by this tool')
    for name, expected in (document.get('files') or {}).items():
        candidate = Path(path) / name
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise RefuseError(f'{path}: completed artifact {name} is missing or changed')
    return document


def write_complete(path, status, files):
    hashes = {name: sha256_file(Path(path) / name) for name in files}
    write_json(Path(path) / 'complete.json', {
        'owner': 'rkva-quality-gate', 'status': status,
        'completed': now(), 'files': hashes,
    })


def arm_directory(args, case_name, arm):
    return args.output / 'cases' / case_name / arm


def load_or_prepare_arm(args, case_name, arm):
    directory = arm_directory(args, case_name, arm)
    marker = complete_marker(directory) if directory.exists() else None
    if marker is not None:
        return {'mode': 'complete', 'directory': directory, 'marker': marker}
    if directory.exists():
        if not any(directory.iterdir()):
            directory.rmdir()
            return {'mode': 'new', 'directory': directory, 'marker': None}
        owner = directory / 'owner.json'
        if not owner.is_file() or read_json(owner).get('owner') != 'rkva-quality-gate':
            raise RefuseError(f'{directory}: non-empty artifact dir is not owned by this tool')
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    write_json(directory / 'owner.json', {'owner': 'rkva-quality-gate', 'created': now()})
    return {'mode': 'new', 'directory': directory, 'marker': None}


# ---------------------------------------------------------------------------
# Frozen prefix snapshot
# ---------------------------------------------------------------------------

def load_snapshot(helper, helper_ns, base_tokens, model_fingerprint, args):
    path = Path(args.prefix_snapshot)
    if path.is_symlink():
        raise RefuseError('prefix snapshot main file must not be a symlink')
    try:
        metadata = helper.load_prefix_snapshot(helper_ns, base_tokens, model_fingerprint)
    except ValueError as error:
        raise RefuseError(f'prefix snapshot contract mismatch: {error}') from error
    if metadata is None:
        if list(path.parent.glob(path.name + '.*')):
            raise RefuseError('orphan prefix snapshot sidecars exist; refusing a fresh save')
        if not args.allow_new_prefix:
            raise RefuseError(
                'no frozen prefix snapshot exists; pass --allow-new-prefix to create one explicitly')
        return {'metadata': None, 'meta_sha256': None, 'files': None, 'path': path,
                'intent': 'create'}
    meta_path = Path(str(path) + '.meta.json')
    # helper.load_prefix_snapshot already verified every size/SHA in the
    # metadata; do not re-hash the multi-GB payload a second time.
    files = metadata.get('files')
    stats = {name: {'size': item.stat().st_size, 'mtime_ns': item.stat().st_mtime_ns,
                    'inode': item.stat().st_ino}
             for name, item in ((p.name, p) for p in [path, *path.parent.glob(path.name + '.*')]
                                if p.is_file() and p.name != path.name + '.meta.json')}
    if set(files or {}) != set(stats):
        raise RefuseError('prefix snapshot file set changed between contract and fingerprint')
    return {'metadata': metadata, 'meta_sha256': sha256_file(meta_path),
            'files': files, 'stats': stats, 'path': path, 'intent': 'restore'}


def verify_snapshot_unchanged(snapshot):
    metadata = snapshot['metadata']
    if metadata is None:
        return
    for name, expected in metadata['files'].items():
        path = Path(snapshot['path']).parent / name
        if not path.is_file():
            raise GateError(f'prefix snapshot file disappeared during the run: {name}')
        stat = path.stat()
        recorded = (snapshot.get('stats') or {}).get(name)
        if (stat.st_size != expected['size']
                or recorded is not None and (
                    stat.st_mtime_ns != recorded.get('mtime_ns')
                    or stat.st_ino != recorded.get('inode'))):
            raise GateError(f'prefix snapshot file changed during the run: {name}')


def restore_snapshot(session, helper, helper_ns, snapshot):
    metadata = snapshot['metadata']
    if metadata is None:
        return None
    response = helper.slot_operation(session.port, 'restore', helper_ns)
    if (response.get('n_restored') != metadata['actual_prefix_tokens']
            or response.get('n_read') != metadata['save']['n_written']):
        raise GateError('slot restore token/byte count differs from the frozen snapshot')
    return response


def create_snapshot(session, helper, helper_ns, base_tokens, model_fingerprint, snapshot):
    """Only reachable with explicit --allow-new-prefix; never overwrites."""
    if snapshot['metadata'] is not None:
        return snapshot
    path = snapshot['path']
    if path.exists():
        raise RefuseError('refusing to overwrite an existing prefix snapshot')
    warm = session.request('/completion', teacher_body(
        base_tokens, 1, keep_full_logprobs=False))
    timings = warm.get('timings') or {}
    if warm.get('error') or warm.get('truncated') or timings.get('predicted_n') != 1:
        raise GateError('prefix warm request failed')
    saved = helper.slot_operation(session.port, 'save', helper_ns)
    manifest = {'model': model_fingerprint, 'server': helper.fingerprint(helper_ns.server_binary),
                'source': source_identity(helper), 'local_libraries': [],
                'loaded_libraries': []}
    metadata = helper.save_prefix_snapshot(helper_ns, base_tokens, manifest, saved)
    meta_path = Path(str(path) + '.meta.json')
    stats = {name: {'size': item.stat().st_size, 'mtime_ns': item.stat().st_mtime_ns,
                    'inode': item.stat().st_ino}
             for name, item in ((p.name, p) for p in [path, *path.parent.glob(path.name + '.*')]
                                if p.is_file() and p.name != path.name + '.meta.json')}
    return {'metadata': metadata, 'meta_sha256': sha256_file(meta_path),
            'files': metadata['files'], 'stats': stats, 'path': path, 'intent': 'create'}


# ---------------------------------------------------------------------------
# Teacher-forced requests
# ---------------------------------------------------------------------------

def teacher_body(prompt, vocab, keep_full_logprobs=True):
    body = {
        'prompt': prompt,
        'n_predict': 1,
        'temperature': 0.0,
        'seed': 1234,
        'top_p': 1.0,
        'min_p': 0.0,
        'repeat_penalty': 1.0,
        'presence_penalty': 0.0,
        'frequency_penalty': 0.0,
        'ignore_eos': True,
        'cache_prompt': True,
        'id_slot': 0,
        'stream': False,
        'return_tokens': True,
        'post_sampling_probs': False,
        'response_fields': [
            'completion_probabilities', 'timings', 'tokens', 'tokens_predicted',
            'generation_settings', 'error', 'truncated',
        ],
    }
    if keep_full_logprobs:
        body['n_probs'] = vocab
    return body


def validate_accounting(timings, prompt_length, p, step, previous_prompt_length=None):
    cached = timings.get('cache_n')
    processed = timings.get('prompt_n')
    if type(cached) is not int or type(processed) is not int:
        raise GateError(f'step {step}: cache_n/prompt_n are not integers')
    if cached < p - 128:
        raise GateError(f'step {step}: cache_n={cached} is below the frozen prefix')
    if step == 0:
        if cached > p:
            raise GateError(
                f'step 0: cache_n={cached} is beyond local P={p}; a tail was restored')
        if processed <= 0:
            raise GateError('step 0: the evaluated tail was not re-ingested from the snapshot')
    else:
        if previous_prompt_length is None or cached != previous_prompt_length:
            raise GateError(
                f'step {step}: cache_n={cached} != previous prompt {previous_prompt_length}')
        if processed != 1:
            raise GateError(
                f'step {step}: processed {processed} tokens, expected exactly one')
    if processed < 0 or cached + processed != prompt_length:
        raise GateError(
            f'step {step}: cache_n {cached} + prompt_n {processed} != prompt {prompt_length}')
    return {'cache_n': cached, 'prompt_n': processed}


def teacher_prompt_hashes(base_tokens, suffix, teacher_ids, steps):
    hashes = []
    prompt = list(base_tokens) + list(suffix)
    for step in range(steps):
        if step:
            prompt = prompt + [teacher_ids[step - 1]]
        hashes.append(token_hash(prompt))
    return hashes


def verify_prompt_alignment(off_hashes, on_hashes):
    if len(off_hashes) != len(on_hashes):
        raise GateError('teacher step count differs between arms')
    for step, (left, right) in enumerate(zip(off_hashes, on_hashes)):
        if left != right:
            raise GateError(f'step {step}: candidate prompt/history differs from the teacher')


def run_arm(session, helper, helper_ns, snapshot, base_tokens, case_tokens, arm,
            teacher_ids, vocab, args, arm_output, progress):
    """Run one fresh-server arm; stream each step and retain no matrix."""
    if snapshot['metadata'] is not None:
        progress['restore'] = restore_snapshot(session, helper, helper_ns, snapshot)
    else:
        created = create_snapshot(session, helper, helper_ns, base_tokens,
                                  progress['model'], snapshot)
        snapshot.update(created)
        progress['prefix_created'] = True
    prompt = list(base_tokens) + list(case_tokens['tokens'])
    step_hashes, accounting, teacher = [], [], []
    prompt_hashes = []
    previous_prompt_length = None
    if arm == 'on':
        if not isinstance(teacher_ids, dict):
            raise GateError('candidate arm needs a stored teacher document')
        teacher_sequence = teacher_ids.get('teacher_ids')
        teacher_hashes = teacher_ids.get('prompt_hashes')
        if not isinstance(teacher_sequence, list) or len(teacher_sequence) != args.teacher_steps:
            raise GateError('candidate teacher sequence is incomplete')
        if not isinstance(teacher_hashes, list) or len(teacher_hashes) != args.teacher_steps:
            raise GateError('candidate teacher prompt hashes are incomplete')
        if any(type(token) is not int or not 0 <= token < vocab for token in teacher_sequence):
            raise GateError('candidate teacher sequence contains an out-of-vocab token')
    else:
        teacher_sequence = None
        teacher_hashes = None
    writer = DistributionWriter(arm_output / 'distributions.bin.gz',
                                args.teacher_steps, vocab)
    try:
        for step in range(args.teacher_steps):
            if step:
                if arm == 'on':
                    prompt = prompt + [teacher_sequence[step - 1]]
                else:
                    prompt = prompt + [teacher[step - 1]]
            expected_hash = teacher_hashes[step] if teacher_hashes is not None else None
            current_hash = token_hash(prompt)
            if expected_hash is not None and current_hash != expected_hash:
                raise GateError(f'step {step}: candidate prompt/hash differs from the teacher')
            body = teacher_body(prompt, vocab)
            body_hash = sha256_text(canonical(body))
            began = time.monotonic()
            response = session.request('/completion', body)
            wall = time.monotonic() - began
            parsed = parse_distribution(response, vocab, args.gates['mass_tolerance'])
            timings = response.get('timings') or {}
            # The raw full-vocab JSON must not survive this iteration.
            del response
            if timings.get('predicted_n') != 1:
                raise GateError(f'step {step}: predicted_n is not 1')
            accounting.append(validate_accounting(
                timings, len(prompt), args.p, step, previous_prompt_length))
            block = array.array('f', parsed['probs'])
            step_hashes.append(writer.append(block))
            del block
            prompt_hashes.append(current_hash)
            if arm == 'off':
                teacher.append(parsed['top1'])
            progress['requests'].append({
                'step': step, 'prompt_tokens': len(prompt),
                'prompt_sha256': current_hash, 'body_sha256': body_hash,
                'wall_s': wall, 'cache_n': accounting[-1]['cache_n'],
                'prompt_n': accounting[-1]['prompt_n'], 'top1': parsed['top1'],
                'margin': parsed['margin'], 'mass': parsed['mass'],
            })
            if step == 0:
                # The first request ingests the mandatory multi-query tail in
                # this arm; n_predict=1 alone is not a multi-query proof.
                progress['tail'] = {'prompt_tokens': len(prompt),
                                    'tail_tokens': case_tokens['tail_length'],
                                    'cache_n': accounting[-1]['cache_n'],
                                    'prompt_n': accounting[-1]['prompt_n']}
            previous_prompt_length = len(prompt)
            del parsed, timings
        writer.close()
    except BaseException:
        writer.abort()
        raise
    distributions_sha = sha256_file(arm_output / 'distributions.bin.gz')
    write_json(arm_output / 'accounting.json', {
        'arm': arm, 'steps': args.teacher_steps, 'p': args.p, 'vocab': vocab,
        'case_tokens_sha256': case_tokens['tokens_sha256'],
        'base_tokens_sha256': token_hash(base_tokens),
        'model_sha256': progress['model']['sha256'],
        'binary_sha256': progress['binary']['sha256'],
        'teacher_ids': teacher if arm == 'off' else list(teacher_sequence),
        'prompt_hashes': prompt_hashes,
        'step_distribution_sha256': step_hashes,
        'distributions_sha256': distributions_sha,
        'accounting': accounting, 'tail': progress.get('tail'),
        'restore': progress.get('restore'),
        'prefix_created': progress.get('prefix_created', False),
    })
    write_json(arm_output / 'requests.json', progress['requests'])
    if arm == 'off':
        write_json(arm_output / 'teacher.json', {
            'teacher_ids': teacher, 'prompt_hashes': prompt_hashes,
            'step_distribution_sha256': step_hashes, 'steps': args.teacher_steps,
            'model_sha256': progress['model']['sha256'],
            'binary_sha256': progress['binary']['sha256'],
            'case_tokens_sha256': case_tokens['tokens_sha256'],
        })
    return teacher, prompt_hashes, accounting


# ---------------------------------------------------------------------------
# Case-level metrics / gates
# ---------------------------------------------------------------------------

def load_arm(path, vocab, steps):
    """Verify an arm file (streaming) and return only its small accounting."""
    accounting = read_json(Path(path) / 'accounting.json')
    if accounting.get('steps') != steps:
        raise GateError(f'{path}: step count changed')
    verify_distribution_file(Path(path) / 'distributions.bin.gz', accounting, vocab, steps)
    return accounting


def answer_report(case, teacher_ids, on_top1, answer_tokens):
    if not answer_tokens:
        return None
    expected = list(answer_tokens)
    baseline_top = teacher_ids[:len(expected)] if len(teacher_ids) >= len(expected) else []
    baseline_correct = baseline_top == expected
    candidate_top = list(on_top1[:len(expected)])
    candidate_correct = candidate_top == expected
    return {
        'expected': expected,
        'baseline_top': baseline_top,
        'candidate_top': candidate_top,
        'baseline_correct': baseline_correct,
        'candidate_correct': candidate_correct,
        'regression': baseline_correct and not candidate_correct,
    }


def compute_case_result(args, case_tokens, off_dir, on_dir, vocab):
    off_accounting = load_arm(off_dir, vocab, args.teacher_steps)
    on_accounting = load_arm(on_dir, vocab, args.teacher_steps)
    off_teacher = off_accounting.get('teacher_ids') or []
    on_teacher = on_accounting.get('teacher_ids') or []
    if len(off_teacher) != args.teacher_steps:
        raise GateError(f'{case_tokens["name"]}: baseline teacher sequence is incomplete')
    if off_teacher != on_teacher:
        raise GateError(f'{case_tokens["name"]}: candidate did not use the baseline teacher')
    verify_prompt_alignment(off_accounting.get('prompt_hashes') or [],
                            on_accounting.get('prompt_hashes') or [])
    off_account = off_accounting.get('accounting') or []
    on_account = on_accounting.get('accounting') or []
    if len(off_account) != len(on_account):
        raise GateError(f'{case_tokens["name"]}: accounting length differs')
    for step, (left, right) in enumerate(zip(off_account, on_account)):
        if left != right:
            raise GateError(
                f'{case_tokens["name"]} step {step}: cache accounting differs between arms '
                f'({left} != {right})')
    steps = []
    on_top1 = []
    off_hashes = off_accounting.get('step_distribution_sha256') or []
    on_hashes = on_accounting.get('step_distribution_sha256') or []
    off_path = Path(off_dir) / 'distributions.bin.gz'
    on_path = Path(on_dir) / 'distributions.bin.gz'
    with DistributionReader(off_path, args.teacher_steps, vocab) as off_reader, \
            DistributionReader(on_path, args.teacher_steps, vocab) as on_reader:
        for step in range(args.teacher_steps):
            off_step = off_reader.read_step()
            on_step = on_reader.read_step()
            if step_digest(off_step) != off_hashes[step]:
                raise GateError(f'{case_tokens["name"]} step {step}: OFF hash mismatch')
            if step_digest(on_step) != on_hashes[step]:
                raise GateError(f'{case_tokens["name"]} step {step}: ON hash mismatch')
            metrics = step_metrics(distribution_view(off_step),
                                   distribution_view(on_step),
                                   off_teacher[step],
                                   args.gates['high_confidence_margin'])
            metrics.update({
                'step': step,
                'cache_n': off_account[step]['cache_n'],
                'prompt_n': off_account[step]['prompt_n'],
                'prompt_sha256': off_accounting['prompt_hashes'][step],
            })
            steps.append(metrics)
            on_top1.append(metrics['top1_on'])
    summary = summarize_steps(steps)
    answer = None
    if case_tokens.get('answer_tokens'):
        answer = answer_report(
            case_tokens, off_teacher, on_top1, case_tokens['answer_tokens'])
        answer['baseline_absolute_accuracy'] = 1.0 if answer['baseline_correct'] else 0.0
        answer['candidate_absolute_accuracy'] = 1.0 if answer['candidate_correct'] else 0.0
    return {
        'case': case_tokens['name'], 'category': case_tokens['category'],
        'length': case_tokens['length'], 'tail_length': case_tokens['tail_length'],
        'tokens_sha256': case_tokens['tokens_sha256'],
        'answer': answer,
        'steps': steps, 'summary': summary,
        'off': {'directory': str(off_dir), 'restore': off_accounting.get('restore'),
                'tail': off_accounting.get('tail'),
                'distributions_sha256': off_accounting.get('distributions_sha256')},
        'on': {'directory': str(on_dir), 'restore': on_accounting.get('restore'),
               'tail': on_accounting.get('tail'),
               'distributions_sha256': on_accounting.get('distributions_sha256')},
    }


def stratify(case_results):
    strata = {}
    for result in case_results:
        for key, value in (('category', result['category']), ('length', str(result['length'])),
                           ('all', 'all')):
            strata.setdefault((key, value), []).extend(result['steps'])
    return strata


def evaluate_gates(case_results, gates, activation_ok, lifecycle_errors,
                   control, args):
    checks = {}

    def add(name, passed, detail):
        checks[name] = {'pass': bool(passed), 'detail': detail}

    add('no_lifecycle_or_protocol_errors', not lifecycle_errors,
        '; '.join(lifecycle_errors[:5]) if lifecycle_errors else 'none')
    add('activation_proof', activation_ok,
        'both arms proved the required runtime paths' if activation_ok
        else 'missing/unproved activation in at least one arm')
    completed = [result for result in case_results if result.get('status') != 'ABORTED']
    add('all_selected_cases_complete', len(completed) == len(case_results),
        f'{len(completed)}/{len(case_results)} cases completed')
    failed_cases = [result['case'] for result in case_results if result.get('status') == 'FAIL']
    add('no_case_failure', not failed_cases,
        ', '.join(failed_cases) if failed_cases else 'none')

    steps = [step for result in completed for step in result['steps']]
    aggregate = summarize_steps(steps)
    finite = bool(steps) and aggregate.get('finite')
    add('distributions_finite', finite,
        'all NLL/KL finite' if finite else 'missing or non-finite NLL/KL')
    if finite:
        add('aggregate_ppl_ratio', aggregate['ppl_ratio'] <= gates['max_ppl_ratio'],
            f"{aggregate['ppl_ratio']:.9f} <= {gates['max_ppl_ratio']}")
        add('aggregate_kl_mean', aggregate['kl_mean'] <= gates['max_kl_mean'],
            f"{aggregate['kl_mean']:.9f} <= {gates['max_kl_mean']}")
        add('aggregate_kl_p99', aggregate['kl_p99'] <= gates['max_kl_p99'],
            f"{aggregate['kl_p99']:.9f} <= {gates['max_kl_p99']}")
    else:
        for name in ('aggregate_ppl_ratio', 'aggregate_kl_mean', 'aggregate_kl_p99'):
            add(name, False, 'metric missing/not finite')

    for (kind, value), stratum_steps in sorted(stratify(completed).items()):
        summary = summarize_steps(stratum_steps)
        if not summary.get('finite'):
            add(f'stratum_{kind}_{value}_finite', False, 'missing/non-finite metrics')
            continue
        add(f'stratum_{kind}_{value}_ppl_ratio',
            summary['ppl_ratio'] <= gates['max_ppl_ratio'],
            f"{summary['ppl_ratio']:.9f} <= {gates['max_ppl_ratio']} "
            f"({summary['steps']} steps)")
        add(f'stratum_{kind}_{value}_kl_mean',
            summary['kl_mean'] <= gates['max_kl_mean'],
            f"{summary['kl_mean']:.9f} <= {gates['max_kl_mean']}")
        add(f'stratum_{kind}_{value}_kl_p99',
            summary['kl_p99'] <= gates['max_kl_p99'],
            f"{summary['kl_p99']:.9f} <= {gates['max_kl_p99']}")

    for result in completed:
        summary = result['summary']
        if not summary.get('finite'):
            add(f'case_{result["case"]}_nll_delta', False, 'missing/non-finite NLL')
        else:
            add(f'case_{result["case"]}_nll_delta',
                summary['nll_delta'] <= gates['max_case_nll_delta'],
                f"{summary['nll_delta']:.9f} <= {gates['max_case_nll_delta']}")

    informative = 0
    regressions = []
    for result in completed:
        answer = result.get('answer')
        if not answer:
            continue
        if answer.get('baseline_correct'):
            informative += 1
            if answer.get('regression'):
                regressions.append(result['case'])
    if args.smoke:
        add('retrieval_baseline_informative', True,
            'not required for a smoke run')
        add('no_retrieval_regression', True, 'not required for a smoke run')
    else:
        add('retrieval_baseline_informative', informative > 0,
            f'{informative} baseline-correct retrieval/factual case(s)')
        add('no_retrieval_regression', not regressions,
            ', '.join(regressions) if regressions else 'none')

    if control is not None:
        add('determinism_control', control.get('pass') is True,
            control.get('reason') or 'fresh-process baseline control')
    elif args.smoke:
        add('determinism_control', True,
            'not required for a smoke run; smoke never qualifies the matrix')
    elif args.determinism_control == 'auto' and all(
            (result.get('length') or 0) < 130560 for result in completed):
        add('determinism_control', True,
            'not applicable: no selected case reaches the 130560-token control stratum')
    else:
        add('determinism_control', False,
            'required fresh-process baseline control was not run')

    high_confidence = sum(result['summary'].get('high_confidence_divergences', 0)
                          for result in completed)
    needs_review = (high_confidence > 0) or any(
        not check['pass'] for name, check in checks.items()
        if name not in ('activation_proof', 'no_lifecycle_or_protocol_errors',
                        'all_selected_cases_complete', 'no_case_failure',
                        'distributions_finite', 'retrieval_baseline_informative',
                        'no_retrieval_regression', 'determinism_control'))
    # A failed gate is FAIL; a fully passing matrix with high-confidence
    # divergences is NEEDS_REVIEW, never PASS.
    passed = all(check['pass'] for check in checks.values())
    status = 'PASS' if passed and not needs_review else ('FAIL' if not passed else 'NEEDS_REVIEW')
    return {
        'status': status,
        'checks': checks,
        'metrics': {
            'steps': aggregate.get('steps', 0),
            'ppl_ratio': aggregate.get('ppl_ratio'),
            'kl_mean': aggregate.get('kl_mean'),
            'kl_p99': aggregate.get('kl_p99'),
            'nll_delta': aggregate.get('nll_delta'),
            'top1_agreement': aggregate.get('top1_agreement'),
            'top1_top2_margin_off': aggregate.get('margin_off_mean'),
            'top1_top2_margin_on': aggregate.get('margin_on_mean'),
            'high_confidence_divergences': high_confidence,
            'answer_accuracy': {
                'baseline': (sum(1 for result in completed
                                 if (result.get('answer') or {}).get('baseline_correct'))
                             / max(1, sum(1 for result in completed if result.get('answer')))),
                'candidate': (sum(1 for result in completed
                                  if (result.get('answer') or {}).get('candidate_correct'))
                              / max(1, sum(1 for result in completed if result.get('answer')))),
                'informative_cases': informative,
            },
        },
        'qualification_label': (
            'SMOKE_ONLY' if args.smoke else
            ('FAILED' if status == 'FAIL' else
             ('NEEDS_REVIEW' if status == 'NEEDS_REVIEW' else
              'QUALIFIED_SCOPE_ONLY'))),
        'int8_default_authorized': False,
        'scope': 'limited teacher-forced matrix only; root must validate hardware and Astra must review',
    }


# ---------------------------------------------------------------------------
# Execution orchestration
# ---------------------------------------------------------------------------

def assert_quiet():
    foreign = foreign_process_matches(foreign_process_rows())
    if foreign:
        raise RefuseError('another benchmark/build process became active: ' +
                          ', '.join(f'{row["comm"]}:{row["pid"]}' for row in foreign[:5]))


class CampaignAbort(RuntimeError):
    pass


def tokenize_text(session, text):
    response = session.request('/tokenize', {'content': text, 'add_special': False})
    tokens = response.get('tokens')
    if not isinstance(tokens, list) or not tokens:
        raise GateError('server /tokenize returned no tokens')
    for token in tokens:
        if type(token) is not int or token < 0:
            raise GateError('server /tokenize returned a non-integer token')
    return tokens


def ensure_case_tokens(session, specs, args):
    """Persist one token sequence per case using this binary's /tokenize.

    Only the suffix after P is tokenized; the shared base prefix is never
    touched.  Existing token files are verified, never silently retokenized.
    """
    vocab = session.n_vocab()
    prepared = {}
    for spec in specs:
        directory = args.output / 'cases' / spec['name']
        path = directory / 'tokens.json'
        if path.is_file():
            document = read_json(path)
            tokens = document.get('tokens')
            if not isinstance(tokens, list) or any(
                    type(token) is not int or not 0 <= token < vocab for token in tokens):
                raise RefuseError(f'{path}: persisted case tokens are invalid')
            if document.get('tokens_sha256') != token_hash(tokens):
                raise RefuseError(f'{path}: persisted case token hash is invalid')
            if document.get('length') != args.p + len(tokens):
                raise RefuseError(f'{path}: persisted case length is inconsistent')
            if document.get('vocab') != vocab:
                raise RefuseError(f'{path}: persisted case vocab changed')
            prepared[spec['name']] = document
            continue
        if directory.exists() and any(directory.iterdir()):
            raise RefuseError(f'{directory}: non-empty case dir without tokens.json')
        directory.mkdir(parents=True, exist_ok=True)
        document = fit_case_tokens(spec, args.p, lambda text: tokenize_text(session, text), vocab)
        document['vocab'] = vocab
        if document['length'] + args.teacher_steps >= args.context:
            raise ValueError(
                f'case {spec["name"]}: length {document["length"]} + teacher steps does not '
                'fit inside the context')
        write_json(path, document)
        prepared[spec['name']] = document
    return prepared


def finalize_arm_dir(arm_dir, session, files):
    profile = session.manifest.get('profile') if session is not None else None
    if not isinstance(profile, dict) or not profile:
        raise GateError('arm profile was not flushed after server teardown')
    if session.manifest.get('child_exit_code') != 0:
        raise GateError(
            f"owned server did not exit 0 after teardown "
            f"({session.manifest.get('child_exit_code')})")
    write_json(Path(arm_dir) / 'profile.json', {
        'engine': profile,
        'profile_lines': session.profile_lines(),
        'startup_warnings': session.manifest.get('startup_warnings') or [],
    })
    write_complete(Path(arm_dir), 'PASS', list(files) + ['profile.json', 'profile-lines.json'])


def prepare_all_tokens(args, helper, helper_ns, guard, specs, flags, hooks):
    """Return vocab and ensure every selected case has a persisted token file."""
    library_guard = hooks.get('library_guard')
    if library_guard is not None:
        library_guard.verify_local('tokenizer-pre')
    existing = []
    for spec in specs:
        path = args.output / 'cases' / spec['name'] / 'tokens.json'
        if path.is_file():
            existing.append(read_json(path))
    if len(existing) == len(specs):
        vocabs = {document.get('vocab') for document in existing}
        if len(vocabs) != 1 or not all(type(value) is int and value > 0 for value in vocabs):
            raise RefuseError('persisted case tokens disagree on vocab')
        vocab = vocabs.pop()
        for document in existing:
            tokens = document.get('tokens')
            if not isinstance(tokens, list) or any(
                    type(token) is not int or not 0 <= token < vocab for token in tokens):
                raise RefuseError('persisted case tokens are invalid')
            if document.get('tokens_sha256') != token_hash(tokens):
                raise RefuseError('persisted case token hash is invalid')
            if document.get('length') != args.p + len(tokens):
                raise RefuseError('persisted case length is inconsistent')
        return vocab
    token_dir = args.output / 'tokenizer'
    if token_dir.exists():
        if not any(token_dir.iterdir()):
            token_dir.rmdir()
        else:
            owner = token_dir / 'owner.json'
            if not owner.is_file() or read_json(owner).get('owner') != 'rkva-quality-gate':
                raise RefuseError(f'{token_dir}: non-empty tokenizer dir is not owned')
            shutil.rmtree(token_dir)
    token_dir.mkdir(parents=True, exist_ok=False)
    write_json(token_dir / 'owner.json', {'owner': 'rkva-quality-gate', 'created': now()})
    assert_quiet()
    with flags.arm(False):
        with ServerSession(helper, helper_ns, args, token_dir, guard,
                           'tokenizer', **hooks.get('server', {})) as session:
            if library_guard is not None:
                library_guard.verify_local('tokenizer-startup')
                library_guard.verify_loaded(
                    session.manifest.get('loaded_libraries'), 'tokenizer-startup')
            ensure_case_tokens(session, specs, args)
            vocab = session.n_vocab()
        if library_guard is not None:
            if session.manifest.get('loaded_libraries_final_error'):
                raise GateError('tokenizer-final: ' + str(
                    session.manifest.get('loaded_libraries_final_error')))
            library_guard.verify_local('tokenizer-final')
            library_guard.verify_loaded(
                session.manifest.get('loaded_libraries_final'), 'tokenizer-final')
            library_guard.require_loaded(
                session.manifest.get('loaded_libraries_final'), 'tokenizer-final')
    return vocab


def run_case_arm(args, helper, helper_ns, guard, snapshot, base_tokens, case_tokens,
                 arm, flags, teacher, vocab, hooks=None):
    hooks = hooks or {}
    arm_dir = arm_directory(args, case_tokens['name'], arm)
    state = load_or_prepare_arm(args, case_tokens['name'], arm)
    library_guard = hooks.get('library_guard')
    if state['mode'] == 'complete':
        if library_guard is not None:
            library_guard.verify_local('pre-arm-complete')
            libraries_path = arm_dir / 'libraries.json'
            if not libraries_path.is_file():
                raise RefuseError(f'{arm_dir}: completed arm has no libraries.json')
            libraries = read_json(libraries_path)
            library_guard.verify_loaded(libraries.get('loaded_startup'),
                                        'complete-startup')
            library_guard.verify_loaded(libraries.get('loaded_final'),
                                        'complete-final')
            library_guard.require_loaded(libraries.get('loaded_startup'),
                                         'complete-startup')
            library_guard.require_loaded(libraries.get('loaded_final'),
                                         'complete-final')
        accounting = load_arm(arm_dir, vocab, args.teacher_steps)
        if not (arm_dir / 'profile.json').is_file():
            raise RefuseError(f'{arm_dir}: completed arm has no profile.json')
        if accounting.get('case_tokens_sha256') != case_tokens['tokens_sha256']:
            raise RefuseError(f'{arm_dir}: completed arm belongs to different case tokens')
        if accounting.get('base_tokens_sha256') != token_hash(base_tokens):
            raise RefuseError(f'{arm_dir}: completed arm belongs to a different base prefix')
        if accounting.get('model_sha256') != hooks.get('model', {}).get('sha256'):
            raise RefuseError(f'{arm_dir}: completed arm belongs to a different model')
        if accounting.get('binary_sha256') != hooks.get('binary', {}).get('sha256'):
            raise RefuseError(f'{arm_dir}: completed arm belongs to a different binary')
        return {'mode': 'complete', 'directory': arm_dir, 'accounting': accounting}
    progress = {'requests': [], 'model': hooks['model'], 'binary': hooks['binary']}
    try:
        assert_quiet()
        if library_guard is not None:
            library_guard.verify_local('pre-arm')
        with flags.arm(arm == 'on'):
            with ServerSession(helper, helper_ns, args, arm_dir, guard,
                               f'{case_tokens["name"]}-{arm}', **hooks.get('server', {})) as session:
                if library_guard is not None:
                    library_guard.verify_local('startup')
                    library_guard.verify_loaded(
                        session.manifest.get('loaded_libraries'), 'startup')
                    library_guard.require_loaded(
                        session.manifest.get('loaded_libraries'), 'startup')
                run_arm(session, helper, helper_ns, snapshot, base_tokens,
                        case_tokens, arm, teacher, vocab, args, arm_dir, progress)
        # The flags context is closed (restore verified) and ServerSession has
        # flushed the final profile/log; only now may a PASS marker be written.
        if library_guard is not None:
            if session.manifest.get('loaded_libraries_final_error'):
                raise GateError('post-arm: ' + str(
                    session.manifest.get('loaded_libraries_final_error')))
            library_guard.verify_local('post-arm')
            library_guard.verify_loaded(
                session.manifest.get('loaded_libraries_final'), 'post-arm')
            library_guard.require_loaded(
                session.manifest.get('loaded_libraries_final'), 'post-arm')
            write_json(arm_dir / 'libraries.json', {
                'binding_sha256': library_guard.binding['sha256'],
                'local': library_guard.binding['files'],
                'loaded_startup': session.manifest.get('loaded_libraries'),
                'loaded_final': session.manifest.get('loaded_libraries_final'),
            })
        finalize_arm_dir(arm_dir, session,
                         ['distributions.bin.gz', 'accounting.json', 'requests.json']
                         + (['libraries.json'] if library_guard is not None else [])
                         + (['teacher.json'] if arm == 'off' else []))
        try:
            accounting = load_arm(arm_dir, vocab, args.teacher_steps)
        except BaseException:
            # Never leave a complete marker over unreadable artifacts.
            (arm_dir / 'complete.json').unlink(missing_ok=True)
            raise
        return {'mode': 'new', 'directory': arm_dir, 'accounting': accounting}
    except BaseException as failure:
        error = f'{type(failure).__name__}: {failure}'
        write_json(arm_dir / 'failure.json', {
            'owner': 'rkva-quality-gate', 'arm': arm, 'error': error, 'at': now(),
        })
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            raise
        raise CampaignAbort(f'{case_tokens["name"]}/{arm}: {error}') from failure


def load_teacher(args, case_name):
    path = arm_directory(args, case_name, 'off') / 'teacher.json'
    if not path.is_file():
        raise GateError(f'{case_name}: baseline teacher.json is missing')
    document = read_json(path)
    if document.get('steps') != args.teacher_steps:
        raise GateError(f'{case_name}: teacher step count changed')
    ids = document.get('teacher_ids')
    hashes = document.get('prompt_hashes')
    if not isinstance(ids, list) or len(ids) != args.teacher_steps:
        raise GateError(f'{case_name}: teacher ids are incomplete')
    if not isinstance(hashes, list) or len(hashes) != args.teacher_steps:
        raise GateError(f'{case_name}: teacher prompt hashes are incomplete')
    return document


def case_result_path(args, case_name):
    return args.output / 'cases' / case_name / 'case-result.json'


def load_case_result(args, case_name):
    directory = args.output / 'cases' / case_name
    marker = complete_marker(directory) if directory.exists() else None
    if marker is None:
        return None
    path = case_result_path(args, case_name)
    if not path.is_file():
        return None
    return read_json(path)


def case_activation(args, case_name):
    off_dir = arm_directory(args, case_name, 'off')
    on_dir = arm_directory(args, case_name, 'on')
    off_profile = read_json(off_dir / 'profile.json').get('engine')
    on_profile = read_json(on_dir / 'profile.json').get('engine')
    off_lines = read_json(off_dir / 'profile-lines.json')
    on_lines = read_json(on_dir / 'profile-lines.json')
    return activation_checks(off_profile, on_profile, off_lines, on_lines,
                             load_helper(args.repo))


def run_case(args, helper, helper_ns, guard, snapshot, base_tokens, spec, flags,
             vocab, hooks):
    case_name = spec['name']
    case_dir = args.output / 'cases' / case_name
    existing = load_case_result(args, case_name)
    if existing is not None and existing.get('status') in ('PASS', 'FAIL', 'NEEDS_REVIEW'):
        if existing.get('status') == 'PASS':
            # A stored PASS is reusable only while both arm markers, library
            # bindings and streamed distribution SHAs verify; otherwise fail
            # closed.
            library_guard = hooks.get('library_guard')
            if library_guard is not None:
                library_guard.verify_local('case-complete')
            for arm in ('off', 'on'):
                directory = arm_directory(args, case_name, arm)
                if complete_marker(directory) is None:
                    raise RefuseError(
                        f'{case_name}: PASS case result exists without complete {arm} arm')
                if library_guard is not None:
                    libraries_path = directory / 'libraries.json'
                    if not libraries_path.is_file():
                        raise RefuseError(
                            f'{directory}: completed arm has no libraries.json')
                    libraries = read_json(libraries_path)
                    library_guard.verify_loaded(libraries.get('loaded_startup'),
                                                'case-complete')
                    library_guard.verify_loaded(libraries.get('loaded_final'),
                                                'case-complete')
                    library_guard.require_loaded(libraries.get('loaded_startup'),
                                                 'case-complete')
                    library_guard.require_loaded(libraries.get('loaded_final'),
                                                 'case-complete')
                load_arm(directory, vocab, args.teacher_steps)
        if not (existing.get('status') == 'FAIL' and args.retry_failed):
            return existing
        # Keep completed PASS arms; remove only the stale case marker.
        (case_dir / 'complete.json').unlink(missing_ok=True)
    case_tokens_path = case_dir / 'tokens.json'
    if not case_tokens_path.is_file():
        raise RefuseError(f'{case_tokens_path}: case tokens were not prepared')
    case_tokens = read_json(case_tokens_path)
    if case_tokens.get('tokens_sha256') != token_hash(case_tokens.get('tokens') or []):
        raise RefuseError(f'{case_tokens_path}: token hash mismatch')
    try:
        off = run_case_arm(args, helper, helper_ns, guard, snapshot, base_tokens,
                           case_tokens, 'off', flags, None, vocab, hooks)
        teacher = load_teacher(args, case_name)
        on = run_case_arm(args, helper, helper_ns, guard, snapshot, base_tokens,
                          case_tokens, 'on', flags, teacher, vocab, hooks)
        result = compute_case_result(args, case_tokens, off['directory'], on['directory'], vocab)
        result['activation'] = case_activation(args, case_name)
        case_failures = []
        if not result['activation']['pass']:
            case_failures.append('activation: ' + result['activation']['reason'])
        summary = result['summary']
        if not summary.get('finite'):
            case_failures.append('non-finite NLL/KL')
        elif (summary.get('nll_delta') is None
              or summary['nll_delta'] > args.gates['max_case_nll_delta']):
            case_failures.append(
                f"case NLL delta {summary.get('nll_delta')} > "
                f"{args.gates['max_case_nll_delta']}")
        answer = result.get('answer')
        if answer and answer.get('regression'):
            case_failures.append('retrieval regression (baseline correct, candidate not)')
        result['case_gate'] = {'pass': not case_failures, 'failures': case_failures}
        result['status'] = 'PASS' if not case_failures else 'FAIL'
        result['qualification'] = ('QUALIFIED_SCOPE_ONLY' if result['status'] == 'PASS'
                                   else 'FAILED')
        result['status_detail'] = ('; '.join(case_failures) if case_failures
                                   else 'case activation and per-case NLL passed')
        write_json(case_result_path(args, case_name), result)
        write_complete(case_dir, result['status'], ['tokens.json', 'case-result.json'])
        return result
    except CampaignAbort as error:
        failure = {
            'case': case_name, 'status': 'ABORTED', 'error': str(error),
            'off_directory': str(arm_directory(args, case_name, 'off')),
            'on_directory': str(arm_directory(args, case_name, 'on')),
        }
        write_json(case_dir / 'campaign-abort.json', failure)
        raise


def run_determinism_control(args, helper, helper_ns, guard, snapshot, base_tokens,
                            case_tokens, flags, vocab, hooks, main_off_dir):
    control_dir = args.output / 'control' / f'{case_tokens["name"]}-off'
    library_guard = hooks.get('library_guard')
    marker = complete_marker(control_dir) if control_dir.exists() else None
    if marker is not None and library_guard is not None:
        library_guard.verify_local('control-complete')
        libraries_path = control_dir / 'libraries.json'
        if not libraries_path.is_file():
            raise RefuseError(f'{control_dir}: completed control has no libraries.json')
        libraries = read_json(libraries_path)
        library_guard.verify_loaded(libraries.get('loaded_startup'), 'control-complete')
        library_guard.verify_loaded(libraries.get('loaded_final'), 'control-complete')
        library_guard.require_loaded(libraries.get('loaded_startup'), 'control-complete')
        library_guard.require_loaded(libraries.get('loaded_final'), 'control-complete')
    if marker is None:
        control_dir.parent.mkdir(parents=True, exist_ok=True)
        if control_dir.exists():
            owner = control_dir / 'owner.json'
            if not owner.is_file() or read_json(owner).get('owner') != 'rkva-quality-gate':
                raise RefuseError(f'{control_dir}: non-empty control dir is not owned')
            shutil.rmtree(control_dir)
        control_dir.mkdir(parents=True, exist_ok=False)
        write_json(control_dir / 'owner.json',
                   {'owner': 'rkva-quality-gate', 'created': now()})
        progress = {'requests': [], 'model': hooks['model'], 'binary': hooks['binary']}
        assert_quiet()
        if library_guard is not None:
            library_guard.verify_local('control-pre')
        with flags.arm(False):
            with ServerSession(helper, helper_ns, args, control_dir, guard,
                               f'{case_tokens["name"]}-control-off',
                               **hooks.get('server', {})) as session:
                if library_guard is not None:
                    library_guard.verify_local('control-startup')
                    library_guard.verify_loaded(
                        session.manifest.get('loaded_libraries'), 'control-startup')
                    library_guard.require_loaded(
                        session.manifest.get('loaded_libraries'), 'control-startup')
                run_arm(session, helper, helper_ns, snapshot, base_tokens, case_tokens,
                        'off', None, vocab, args, control_dir, progress)
        if library_guard is not None:
            if session.manifest.get('loaded_libraries_final_error'):
                raise GateError('control-post: ' + str(
                    session.manifest.get('loaded_libraries_final_error')))
            library_guard.verify_local('control-post')
            library_guard.verify_loaded(
                session.manifest.get('loaded_libraries_final'), 'control-post')
            library_guard.require_loaded(
                session.manifest.get('loaded_libraries_final'), 'control-post')
            write_json(control_dir / 'libraries.json', {
                'binding_sha256': library_guard.binding['sha256'],
                'local': library_guard.binding['files'],
                'loaded_startup': session.manifest.get('loaded_libraries'),
                'loaded_final': session.manifest.get('loaded_libraries_final'),
            })
        finalize_arm_dir(control_dir, session,
                         ['distributions.bin.gz', 'accounting.json',
                          'requests.json', 'teacher.json']
                         + (['libraries.json'] if library_guard is not None else []))
        try:
            load_arm(control_dir, vocab, args.teacher_steps)
        except BaseException:
            (control_dir / 'complete.json').unlink(missing_ok=True)
            raise
    main_accounting = load_arm(main_off_dir, vocab, args.teacher_steps)
    control_accounting = load_arm(control_dir, vocab, args.teacher_steps)
    if control_accounting.get('case_tokens_sha256') != case_tokens['tokens_sha256']:
        raise RefuseError('control arm belongs to different case tokens')
    if control_accounting.get('base_tokens_sha256') != token_hash(base_tokens):
        raise RefuseError('control arm belongs to a different base prefix')
    teacher_equal = main_accounting.get('teacher_ids') == control_accounting.get('teacher_ids')
    agreement = 0
    maximum = 0.0
    main_path = Path(main_off_dir) / 'distributions.bin.gz'
    control_path = control_dir / 'distributions.bin.gz'
    with DistributionReader(main_path, args.teacher_steps, vocab) as main_reader, \
            DistributionReader(control_path, args.teacher_steps, vocab) as control_reader:
        for step in range(args.teacher_steps):
            left = main_reader.read_step()
            right = control_reader.read_step()
            left_top = distribution_view(left)['top1']
            right_top = distribution_view(right)['top1']
            agreement += left_top == right_top
            maximum = max(maximum, max(abs(a - b) for a, b in zip(left, right)))
    if library_guard is not None:
        library_guard.verify_local('control-final')
    passed = teacher_equal and agreement == args.teacher_steps
    return {
        'case': case_tokens['name'], 'mode': args.determinism_control,
        'directory': str(control_dir),
        'teacher_tokens_equal': teacher_equal,
        'top1_agreement': agreement / args.teacher_steps,
        'max_abs_probability_delta': maximum,
        'pass': passed,
        'reason': ('fresh baseline process reproduced the teacher and top-1 sequence'
                   if passed else 'fresh baseline process diverged from the stored baseline'),
    }


def run_preflight_smoke(args, helper, helper_ns, guard, snapshot, base_tokens,
                        flags, hooks):
    """Separate OFF/ON smoke before the matrix, with its own servers and flags."""
    length = min(args.p + SMOKE_LENGTH_OFFSET, args.context - SMOKE_STEPS - 1)
    if length <= args.p:
        raise ValueError('context is too small for the mandatory preflight smoke')
    preflight = SimpleNamespace(**vars(args))
    preflight.output = args.output / 'preflight-smoke'
    preflight.teacher_steps = SMOKE_STEPS
    preflight.smoke = False
    spec = build_case_spec(f'preflight-portuguese-{length}', 'portuguese',
                           length, args.seed + length)
    vocab = prepare_all_tokens(preflight, helper, helper_ns, guard, [spec], flags, hooks)
    result = run_case(preflight, helper, helper_ns, guard, snapshot, base_tokens,
                      spec, flags, vocab, hooks)
    for arm in ('off', 'on'):
        accounting_path = (preflight.output / 'cases' / spec['name'] / arm
                           / 'accounting.json')
        if accounting_path.is_file():
            accounting = read_json(accounting_path)
            if accounting.get('model_sha256') != hooks.get('model', {}).get('sha256'):
                raise GateError('preflight smoke model does not match the frozen binding')
            if accounting.get('binary_sha256') != hooks.get('binary', {}).get('sha256'):
                raise GateError('preflight smoke binary does not match the frozen binding')
    document = {
        'mode': 'preflight-smoke', 'case': result['case'],
        'status': result['status'], 'length': result['length'],
        'teacher_steps': SMOKE_STEPS,
        'model_sha256': hooks.get('model', {}).get('sha256'),
        'binary_sha256': hooks.get('binary', {}).get('sha256'),
        'activation': result.get('activation'),
        'summary': result.get('summary'),
        'off': result.get('off'), 'on': result.get('on'),
        'label': 'PREFLIGHT_SMOKE_ONLY',
    }
    write_json(args.output / 'preflight-smoke.json', document)
    if result['status'] != 'PASS' or not (result.get('activation') or {}).get('pass'):
        raise GateError('preflight smoke did not prove a passing OFF/ON arm')
    print(json.dumps({'preflight_smoke': {
        'status': result['status'], 'activation': True,
        'length': result['length'], 'teacher_steps': SMOKE_STEPS}}, ensure_ascii=False),
        flush=True)
    return document


def case_catalog(args):
    plan, specs = case_plan(args)
    lengths = [spec['length'] for spec in specs if spec['length'] is not None]
    if not lengths and any(spec['length'] is None for spec in specs):
        lengths = [args.p]
    return plan, specs, max(lengths or [args.p])


def preview(args):
    helper = load_helper(args.repo)
    plan, specs, maximum = case_catalog(args)
    guard = capture_guard_policy(args.repo)
    memory = helper.memory_policy()
    binary = stat_identity(args.server_binary)
    model = stat_identity(args.model)
    prefix = stat_identity(args.prefix_snapshot)
    tokens_stat = stat_identity(args.prompt_tokens)
    source = None
    source_error = None
    try:
        source = source_identity(helper)
    except GateError as error:
        source_error = str(error)
    prefix_contract = None
    if prefix.get('exists'):
        meta_path = Path(str(args.prefix_snapshot) + '.meta.json')
        if meta_path.is_file():
            try:
                metadata = read_json(meta_path)
                prefix_contract = {
                    'meta_sha256': sha256_file(meta_path),
                    'P': metadata.get('contract', {}).get('P'),
                    'context': metadata.get('contract', {}).get('context'),
                    'batch': metadata.get('contract', {}).get('batch'),
                    'actual_prefix_tokens': metadata.get('actual_prefix_tokens'),
                    'actual_prefix_sha256': metadata.get('actual_prefix_sha256'),
                    'model_sha256': metadata.get('contract', {}).get('model_sha256'),
                }
            except (OSError, ValueError) as error:
                prefix_contract = {'error': str(error)}
        else:
            prefix_contract = {'error': 'meta.json missing'}
    command = helper.server_command(helper_args(args), '<unique-loopback-port>') + ['--no-warmup']
    return {
        'schema': 'rkva-quality-gate-preview-v1',
        'mode': 'preview', 'status': 'PREVIEW_ONLY',
        'starts_child': False, 'writes_output': False, 'requires_execute': True,
        'tool': str(Path(__file__).resolve()),
        'repo': str(args.repo), 'branch': (source or {}).get('branch'),
        'source': source, 'source_error': source_error,
        'binary': binary, 'model': model, 'prefix': prefix, 'prefix_contract': prefix_contract,
        'prompt_tokens': tokens_stat,
        'configuration': {
            'context': args.context, 'context_cap': CONTEXT_MAX, 'p': args.p,
            'batch': args.batch, 'ubatch': args.ubatch,
            'wavefront_tile': args.wavefront_tile, 'wavefront_tiles': args.wavefront_tiles,
            'cuda_fp8': args.cuda_fp8, 'range_kernel': args.range_kernel,
            'tcp_prefetch': args.tcp_prefetch, 'local_input_reuse': False,
            'mask_pack': 'default', 'frozen_prefix_batch': args.frozen_prefix_batch,
            'teacher_steps': args.teacher_steps, 'smoke': args.smoke,
            'determinism_control': args.determinism_control,
            'cases': args.case_source, 'lengths': args.lengths_list,
            'max_case_length': maximum,
        },
        'guard_policy': guard, 'memory_policy': memory,
        'gates': args.gates, 'gate_sha256': sha256_text(canonical(args.gates)),
        'case_plan': plan, 'case_plan_sha256': sha256_text(canonical(plan)),
        'pinned_environment_preview': {
            key: value for key, value in helper.pinned_environment(
                helper_args(args)).items()
            if key in ('GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB', 'GGML_KVARN_WINDOW_CHUNK',
                       'GGML_RKVA_RANGE_KERNEL', 'GGML_RKVA_CUDA_FP8',
                       'GGML_RKVA_TCP_PREFETCH', 'GGML_RKVA_LOCAL_INPUT_REUSE',
                       'GGML_CUDA_MASK_PACK', 'LLAMA_N256_LOOKAHEAD')}
        | {'GGML_CUDA_GRAPH_GUARD_SAMPLE_MS': str(guard['sample_ms']),
           'GGML_CUDA_GRAPH_GUARD_SLACK_MB': str(guard['slack_mb'])},
        'command_preview': command,
        'foreign_processes': foreign_process_matches(foreign_process_rows()),
        'request_contract': {
            'endpoint': '/completion', 'prompt': 'shared P + case suffix + baseline teacher',
            'n_predict': 1, 'n_probs': 'model_n_vocab',
            'post_sampling_probs': False, 'cache_prompt': True, 'seed': 1234,
            'temperature': 0.0, 'id_slot': 0, 'stream': False,
        },
        'integrity_scope': INTEGRITY_SCOPE,
        'limitations': LIMITATIONS,
        'next': ('re-run with --execute and the same frozen configuration; root owns the '
                 'hardware run and Astra reviews before integration'),
    }


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

def execute(args):
    helper = load_helper(args.repo)
    helper_ns = helper_args(args)
    guard = capture_guard_policy(args.repo)
    memory = helper.memory_policy()
    source_start = source_identity(helper)
    plan, specs, maximum = case_catalog(args)

    for path, label in ((args.server_binary, 'server binary'), (args.model, 'model'),
                        (args.prompt_tokens, 'prompt tokens')):
        if not path.is_file():
            raise RefuseError(f'{label} not found: {path}')
    if not args.prefix_snapshot.parent.is_dir():
        raise RefuseError(f'prefix snapshot directory not found: {args.prefix_snapshot.parent}')
    if args.xbox_host is None:
        raise RefuseError('--xbox-host is required for --execute')
    foreign = foreign_process_matches(foreign_process_rows())
    if foreign:
        raise RefuseError('another benchmark/build process is active: ' +
                          ', '.join(f'{row["comm"]}:{row["pid"]}' for row in foreign[:5]))

    base_tokens, token_info = helper.effective_tokens(
        str(args.prompt_tokens), args.p, repeat=False)
    binary = helper.fingerprint(args.server_binary)
    model = helper.fingerprint(args.model)
    snapshot = load_snapshot(helper, helper_ns, base_tokens, model, args)
    libraries = build_library_binding(helper_ns.server_binary)
    binding = compute_binding(args, plan, source_start, binary, model,
                              stat_identity(args.prefix_snapshot), snapshot, guard,
                              args.gates, libraries)
    output_state = prepare_output(args, binding)
    preflight = preview(args)
    preflight.update({'mode': 'preflight', 'status': 'EXECUTING',
                      'output_state': output_state, 'binding': binding,
                      'libraries': libraries, 'token_stream': token_info,
                      'prefix_intent': snapshot['intent']})
    write_json(args.output / 'preflight.json', preflight)
    print(json.dumps(preflight, indent=2), flush=True)

    deploy = DeployClient(
        args.deploy or (args.repo / DEPLOY_REL), args.xbox_env, args.deploy_timeout,
        xbox_ip=args.xbox_ip or None)
    pfn = deploy.pfn()
    flags = LocalStateFlags(deploy, pfn, args.output)
    original_path = args.output / 'flags-original.json'
    if original_path.is_file():
        stored = read_json(original_path)
        if stored.get('pfn') != pfn:
            raise RefuseError(
                'installed package changed since the original flag state was captured')
        stored_present = stored.get('present')
        if not isinstance(stored_present, dict):
            raise RefuseError('stored original flag state is invalid')
        # Adopt the persisted map, adding flags introduced after that run (for
        # example GQA2) from the current device state, then reconcile a prior
        # crash before doing any new work.
        flags.adopt_original(stored_present)
        flags.restore()
    else:
        flags.capture_original()

    lifecycle_errors = []
    case_results = []
    aborted = None
    control = None
    library_guard = LibraryGuard(helper_ns.server_binary, libraries)
    hooks = {'model': model, 'binary': binary, 'server': {},
             'libraries': libraries, 'library_guard': library_guard}
    preflight_smoke = None
    try:
        if not args.smoke:
            try:
                preflight_smoke = run_preflight_smoke(
                    args, helper, helper_ns, guard, snapshot, base_tokens, flags, hooks)
                preflight_smoke['pfn'] = pfn
                preflight_smoke['flags_original'] = flags.original
                write_json(args.output / 'preflight-smoke.json', preflight_smoke)
            except Exception as error:
                aborted = f'{type(error).__name__}: {error}'
                lifecycle_errors.append('preflight smoke failed: ' + aborted)
                write_json(args.output / 'campaign-abort.json', {
                    'owner': 'rkva-quality-gate', 'stage': 'preflight_smoke',
                    'error': aborted, 'at': now(),
                })
                if (args.output / 'preflight-smoke.json').is_file():
                    try:
                        preflight_smoke = read_json(args.output / 'preflight-smoke.json')
                    except (OSError, ValueError):
                        preflight_smoke = None
        if aborted is None:
            try:
                vocab = prepare_all_tokens(args, helper, helper_ns, guard, specs, flags, hooks)
            except Exception as error:
                aborted = f'{type(error).__name__}: {error}'
                lifecycle_errors.append(aborted)
                write_json(args.output / 'campaign-abort.json', {
                    'owner': 'rkva-quality-gate', 'stage': 'prepare_all_tokens',
                    'error': aborted, 'at': now(),
                })
                vocab = None
        else:
            vocab = None
        token_manifest = {}
        for spec in (specs if vocab is not None else []):
            document = read_json(args.output / 'cases' / spec['name'] / 'tokens.json')
            token_manifest[spec['name']] = {
                'length': document['length'], 'tail_length': document['tail_length'],
                'tokens_sha256': document['tokens_sha256'],
                'answer_tokens': document.get('answer_tokens'),
                'end_key_distance_tokens': (document.get('metadata') or {}).get(
                    'end_key_distance_tokens'),
            }
        write_json(args.output / 'case-tokens.json', token_manifest)
        print(json.dumps({'case_tokens': token_manifest}, ensure_ascii=False), flush=True)
        for spec in (specs if vocab is not None and aborted is None else []):
            try:
                result = run_case(args, helper, helper_ns, guard, snapshot,
                                  base_tokens, spec, flags, vocab, hooks)
            except CampaignAbort as error:
                aborted = str(error)
                case_results.append({'case': spec['name'], 'status': 'ABORTED',
                                     'error': str(error), 'steps': [], 'summary': {}})
                break
            except Exception as error:  # protocol/binding corruption is never a PASS
                aborted = f'{type(error).__name__}: {error}'
                write_json(args.output / 'cases' / spec['name'] / 'failure.json',
                           {'owner': 'rkva-quality-gate', 'case': spec['name'],
                            'error': aborted, 'at': now()})
                case_results.append({'case': spec['name'], 'status': 'ABORTED',
                                     'error': aborted, 'steps': [], 'summary': {}})
                break
            case_results.append(result)
            print(json.dumps({'case': result['case'], 'status': result['status'],
                              'summary': result.get('summary'),
                              'activation': result.get('activation', {}).get('pass')},
                             ensure_ascii=False), flush=True)

        if aborted is None and args.determinism_control != 'off' and not args.smoke:
            long_results = [result for result in case_results
                            if result.get('status') != 'ABORTED'
                            and result.get('length') is not None
                            and result['length'] >= 130560]
            target = long_results[-1] if long_results else None
            if args.determinism_control == 'on' and case_results:
                target = case_results[-1]
            if target is not None:
                case_tokens = read_json(
                    args.output / 'cases' / target['case'] / 'tokens.json')
                main_off = arm_directory(args, target['case'], 'off')
                try:
                    control = run_determinism_control(
                        args, helper, helper_ns, guard, snapshot, base_tokens,
                        case_tokens, flags, vocab, hooks, main_off)
                except Exception as error:
                    control = {'pass': False, 'mode': args.determinism_control,
                               'reason': f'{type(error).__name__}: {error}'}
                    lifecycle_errors.append('determinism control failed: ' + control['reason'])
                print(json.dumps({'determinism_control': control}, indent=2), flush=True)
        elif args.determinism_control == 'off':
            control = {'pass': False, 'mode': 'off',
                       'reason': 'determinism control was explicitly skipped'}
    finally:
        # Always restore and verify the original LocalState state, even after
        # a protocol failure.  A restore failure is itself a FAIL gate.
        try:
            flags.restore()
        except BaseException as first_restore_error:
            try:
                flags.restore()
            except BaseException as restore_error:
                lifecycle_errors.append(
                    f'flag restore failed: {type(first_restore_error).__name__}: '
                    f'{first_restore_error}; retry: {type(restore_error).__name__}: '
                    f'{restore_error}')
        try:
            source_end = source_identity(helper)
            if (source_end['head'] != source_start['head']
                    or source_end['tracked_diff_sha256'] != source_start['tracked_diff_sha256']):
                lifecycle_errors.append('source HEAD/tracked diff changed during the run')
        except GateError as error:
            lifecycle_errors.append(f'source recheck failed: {error}')
        try:
            verify_snapshot_unchanged(snapshot)
        except GateError as error:
            lifecycle_errors.append(f'prefix snapshot changed during the run: {error}')

    try:
        library_guard.verify_local('pre-final')
    except GateError as error:
        lifecycle_errors.append(str(error))
    activation_ok = bool(case_results) and all(
        result.get('status') != 'ABORTED'
        and (result.get('activation') or {}).get('pass') is True
        for result in case_results)
    for result in case_results:
        if result.get('status') == 'ABORTED':
            lifecycle_errors.append(f'{result["case"]}: {result.get("error")}')
    gates = evaluate_gates(case_results, args.gates, activation_ok, lifecycle_errors,
                           control, args)
    final = {
        'schema': 'rkva-quality-gate-final-v1', 'finished': now(),
        'status': gates['status'], 'qualification_label': gates['qualification_label'],
        'int8_default_authorized': False,
        'scope': gates['scope'],
        'binding_sha256': binding['binding_sha256'],
        'libraries': binding['libraries'],
        'case_results': case_results, 'gates': gates,
        'preflight_smoke': preflight_smoke,
        'determinism_control': control,
        'guard_policy': guard, 'memory_policy': memory,
        'lifecycle_errors': lifecycle_errors,
        'hardware_run_by_this_tool': True,
        'note': ('tool-run evidence only; the root integrator validates hardware and '
                 'Astra reviews before any integration. Not a throughput benchmark.'),
        'request_contract': {
            'endpoint': '/completion', 'n_predict': 1, 'n_probs': 'model_n_vocab',
            'post_sampling_probs': False, 'cache_prompt': True, 'seed': 1234,
            'temperature': 0.0, 'id_slot': 0,
        },
        'integrity_scope': INTEGRITY_SCOPE,
        'limitations': LIMITATIONS,
    }
    write_json(args.output / 'final.json', final)
    print(json.dumps({'status': final['status'],
                      'qualification_label': final['qualification_label'],
                      'metrics': gates['metrics'],
                      'failed_checks': [name for name, check in gates['checks'].items()
                                        if not check['pass']],
                      'output': str(args.output)}, indent=2, ensure_ascii=False), flush=True)
    if final['status'] == 'FAIL':
        return 3
    if final['status'] == 'NEEDS_REVIEW':
        return 4
    return 0


def main(argv=None):
    try:
        args = validate(parser().parse_args(argv))
    except (ValueError, OSError) as error:
        print(f'Configuration error: {error}', file=sys.stderr)
        return 2
    if not args.execute:
        try:
            print(json.dumps(preview(args), indent=2, ensure_ascii=False))
        except (RefuseError, GateError, ValueError, OSError, RuntimeError) as error:
            print(f'Configuration error: {error}', file=sys.stderr)
            return 2
        return 0
    try:
        return execute(args)
    except RefuseError as error:
        print(f'Refused: {error}', file=sys.stderr)
        return 2
    except (GateError, ValueError, OSError, CampaignAbort, KeyError, TypeError,
            RuntimeError) as error:
        print(f'FAILED: {type(error).__name__}: {error}', file=sys.stderr)
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
