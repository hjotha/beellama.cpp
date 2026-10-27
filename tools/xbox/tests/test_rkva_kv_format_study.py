"""Hardware-free tests for the RKVA KV-format study wrapper.

Run: python3 -m unittest discover -s tools/xbox/tests -p test_rkva_kv_format_study.py
"""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'rkva-kv-format-study.py'
REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('rkva_kv_format_study', SCRIPT)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
study.load_h(REPO)


def connected():
    return 'RemoteAttn: connected to 192.0.2.1:50054, 16 remote layers, wire=f16, budget=4096 MiB'


# Compact subset of the real archived KVarN4 GPU profile
# (/home/hjotha/.codex/artifacts/rkva-priorities-20261026/overflow-refusal-r2/
# reference/server.log).  The schema is nested: {"engine": ..., "transport": ...};
# buckets and all engine counters live under "engine".
REAL_ENGINE = {
    'schema': 'rkva-range-profile-v1', 'profiling': True,
    'counters': {'upload_calls': 1024, 'attention_calls': 1024, 'bulk_calls': 48,
                 'bulk_segments': 1024, 'bulk_queries': 8192, 'failed_calls': 0,
                 'submits': 48, 'dispatches': 3584, 'buffer_creates': 165,
                 'buffer_create_bytes': 26026272},
    'optimized_kernels': {
        'kvarn4': {'available': True, 'status': 'ready', 'optimized_calls': 1024,
                   'reference_calls': 0, 'fast_nq_max': 9},
        'q8_0': {'available': True, 'status': 'ready', 'optimized_calls': 0,
                 'reference_calls': 0, 'fast_nq_max': 9},
        'f16': {'available': True, 'status': 'ready', 'optimized_calls': 0,
                'reference_calls': 0, 'fast_nq_max': 9}},
    'kvarn4_multi_query': {
        'mq2': {'available': True, 'status': 'ready', 'calls': 1024},
        'mq4': {'available': True, 'status': 'ready', 'calls': 0},
        'mq2_i8': {'available': True, 'status': 'ready', 'calls': 0, 'exact': False}},
    'kvarn4_decode_rowlane': {'available': True, 'enabled': True, 'status': 'ready',
                              'calls': 0},
    'format_multi_query': {
        'f16': {'mq2': {'available': True, 'status': 'ready', 'calls': 0},
                'mq4': {'available': True, 'status': 'ready', 'calls': 0}},
        'q8_0': {'mq2': {'available': True, 'status': 'ready', 'calls': 0},
                 'mq4': {'available': True, 'status': 'ready', 'calls': 0}}},
    'bulk': {'bulk_calls': 48, 'bulk_segments': 1024, 'bulk_queries': 8192,
             'fp8_gpu_calls': 48, 'fp8_cpu_fallback_calls': 0},
    'buckets': {'attention/kvarn4-prefix/optimized-kvarn4_mq2_fine/nq2-16/r1-256/warm':
                {'samples': 512}},
}
REAL_TRANSPORT = {
    'enabled': True, 'trace_requests': False, 'fp8_gpu_enabled': True,
    'ops': [{'op': 22, 'calls': 48, 'request_bytes': 93153536,
             'response_bytes': 51120768, 'payload_receive_us': 804287,
             'handle_us': 479027, 'send_enqueue_us': 17304}],
}
REAL_STATS = {
    'calls': 53, 'range_upload_bytes': 42729472, 'range_upload_calls': 1024,
    'range_upload_rows': 8192, 'range_attn_calls': 1024,
    'range_cache_type': 'kvarn4', 'range_source_cache_type': 'projected_f32',
    'wire_format': 'fp8_e5m2', 'native_precision_tail_tokens': 128,
    'tcp_prefetch_completed': 48, 'cuda_pack_calls': 48, 'cuda_unpack_calls': 48,
    'cuda_small_pack_calls': 0, 'cuda_small_unpack_calls': 0,
}
REAL_LOG = (connected() + '\nRemoteAttn: EXPERIMENTAL native KVarN4 precision_tail=128 '
            'descriptor=3 serial_prefix=1 async_overlap=1 wire=fp8_e5m2\n')


def gpu_fixture(name):
    engine = copy.deepcopy(REAL_ENGINE)
    transport = copy.deepcopy(REAL_TRANSPORT)
    if name == 'q8_0':
        engine['optimized_kernels'] = {
            'q8_0': {'available': True, 'status': 'ready', 'optimized_calls': 1024,
                     'reference_calls': 0, 'fast_nq_max': 9}}
        engine['buckets'] = {
            'attention/q8_0/optimized-q8_gqa_fine/nq1/r32k+/warm': {'samples': 512}}
        engine['kvarn4_multi_query']['mq2']['calls'] = 0
        engine['kvarn4_multi_query']['mq4']['calls'] = 0
        engine['kvarn4_decode_rowlane']['calls'] = 0
        engine['format_multi_query'] = {
            'q8_0': {'mq2': {'available': True, 'calls': 0},
                     'mq4': {'available': True, 'calls': 0}}}
    elif name == 'f16':
        engine['optimized_kernels'] = {
            'f16': {'available': True, 'status': 'ready', 'optimized_calls': 1024,
                    'reference_calls': 0, 'fast_nq_max': 9}}
        engine['buckets'] = {
            'attention/f16/optimized-f16_gqa_fine/nq1/r32k+/warm': {'samples': 512}}
        engine['counters'].update(bulk_calls=0, bulk_segments=0, bulk_queries=0)
        engine['kvarn4_multi_query']['mq2']['calls'] = 0
        engine['kvarn4_multi_query']['mq4']['calls'] = 0
        engine['kvarn4_decode_rowlane']['calls'] = 0
        engine['format_multi_query'] = {
            'f16': {'mq2': {'available': True, 'calls': 0},
                    'mq4': {'available': True, 'calls': 0}}}
        engine['bulk'] = {'fp8_gpu_calls': 0, 'fp8_cpu_fallback_calls': 0}
        transport['ops'] = [
            {'op': 13, 'calls': 48, 'request_bytes': 1000, 'response_bytes': 1000},
            {'op': 14, 'calls': 48, 'request_bytes': 1000, 'response_bytes': 1000}]
    return {'engine': engine, 'transport': transport}


def proof_inputs(name):
    stats = copy.deepcopy(REAL_STATS)
    if name == 'q8_0':
        stats.update(range_cache_type='q8_0', native_precision_tail_tokens=0,
                     cuda_pack_calls=0, cuda_unpack_calls=0)
    elif name == 'f16':
        stats.update(range_cache_type='f16', range_source_cache_type='f16',
                     wire_format='f16', native_precision_tail_tokens=0,
                     tcp_prefetch_completed=0, cuda_pack_calls=0,
                     cuda_unpack_calls=0)
    log = REAL_LOG
    if name == 'q8_0':
        log = (connected() + '\nRemoteAttn: native Q8_0 bulk descriptor=5 origin=1024 '
               'capacity=2048 async_overlap=1 wire=fp8_e5m2; host CPU q8_0 producer\n')
    elif name == 'f16':
        log = (connected() + '\nRemoteAttn: range source=f16 storage=f16; direct original '
               'K/V half bytes, no quantized intermediate\n')
    return stats, gpu_fixture(name), log


def manifest_for(name):
    profile = study.profiles()[name]
    command = ['fixture-server', '--cache-type-k', 'kvarn4', '--cache-type-v', 'kvarn4',
               '--remote-attn-cache-type-k', profile['remote_cache_k'],
               '--remote-attn-cache-type-v', profile['remote_cache_v'],
               '--remote-attn-format', profile['upload'],
               '--remote-attn-wire-format', profile['wire'],
               '--batch-size', str(profile['batch']), '--ubatch-size', str(profile['ubatch']),
               '--remote-attn-layers', 'full', '--flash-attn', 'on', '--no-warmup']
    if name == 'f16':
        command += ['--remote-attn-prefill', 'remote']
    environment = {
        'GGML_RKVA_RANGE_STORAGE': profile['range_storage'],
        'GGML_RKVA_ASYNC_LOCAL_REMOTE': str(profile['async_overlap']),
        'GGML_RKVA_TCP_PREFETCH': '1' if profile['tcp_prefetch'] else '0',
        'GGML_RKVA_WIRE_FORMAT': profile['wire'],
        'GGML_RKVA_CUDA_FP8': profile['cuda_fp8'],
        'GGML_RKVA_NATIVE_TAIL_TOKENS': str(profile['tail']),
        'GGML_RKVA_RANGE_KERNEL': profile['range_kernel'],
        'GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB': '128',
        'GGML_KVARN_WINDOW_CHUNK': '4096',
        'GGML_RKVA_NATIVE_BULK': 'f16' if name == 'kvar4' else profile['native_bulk'],
    }
    if name != 'f16':
        environment['GGML_RKVA_NATIVE_PRODUCER'] = profile['producer']
    return {'command': command, 'environment': environment,
            'configuration': {'p': 1024}}


def proof(profile_name, stats=None, gpu=None, log=None, manifest=None):
    stats, gpu, log = (proof_inputs(profile_name) if stats is None
                       else (stats, gpu, log))
    return study.prove_format(study.profiles()[profile_name], results(),
                              manifest if manifest is not None else manifest_for(profile_name),
                              log, stats, gpu)


def results():
    return {'status': 'passed', 'points': [{'valid': True}]}


def point(label='point-1536', cache=1024, prompt=512, n=256, tokens=None, ms=25500):
    return {'label': label, 'cache_n': cache, 'prompt_n': prompt, 'predicted_n': n,
            'predicted_ms': ms, 'valid': True, 'tokens': tokens}


def arm(name, tokens, occurrence=0, cache=1024, prompt=512, status='PASS', upload=1000):
    stats, gpu, log = proof_inputs(name)
    stats['range_upload_bytes'] = upload
    return {'format': name, 'status': status, 'occurrence': occurrence,
            'points': [point(tokens=tokens, cache=cache, prompt=prompt)],
            'proof': {'status': status, 'stats': stats, 'errors': [], 'unsupported': []},
            'warm': None, 'prefix_intent': 'reuse', 'server_error': None}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ('server', 'model'):
            (self.root / name).write_bytes(b'fixture-' + name.encode())
        (self.root / 'tokens.json').write_text(json.dumps(list(range(1536))))
        self.argv = ['--server-binary', str(self.root / 'server'),
                     '--model', str(self.root / 'model'),
                     '--prompt-tokens', str(self.root / 'tokens.json'),
                     '--output', str(self.root / 'out'),
                     '--prefix-dir', str(self.root / 'prefixes'),
                     '--xbox-host', '192.0.2.1:55432', '--threads', '4',
                     '--p', '1024', '--context', '2048', '--points', '1536', '--n-predict', '256']

    def tearDown(self):
        self.tmp.cleanup()

    def preview(self, extra=()):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(study.main(self.argv + list(extra)), 0)
        return json.loads(output.getvalue())

    def validated(self, extra=()):
        return study.validate(study.parser().parse_args(self.argv + list(extra)))


class PreviewTests(Base):
    def test_helper_hash_anchors_and_no_removed_integration_dependency(self):
        study.load_h(REPO)
        self.assertEqual(len(study.H_META['sha256']), 64)
        self.assertTrue(all(anchor in study.H_META['anchors'] for anchor in study.H_META['anchors']))
        self.assertEqual(Path(study.H_META['root']), REPO)
        self.assertNotIn('/integration', SCRIPT.read_text())

    def test_default_preview_is_inert(self):
        with mock.patch('subprocess.Popen', side_effect=AssertionError('child started')), \
             mock.patch('socket.socket', side_effect=AssertionError('socket opened')):
            plan = self.preview()
        self.assertEqual(plan['mode'], 'preview')
        self.assertFalse(plan['starts_child'])
        self.assertFalse(plan['writes_output'])
        self.assertFalse((self.root / 'out').exists())
        self.assertFalse((self.root / 'prefixes').exists())

    def test_commands_match_audited_formats_and_preserve_global_kvarn4(self):
        plan = self.preview()
        expected = {
            'kvar4': ('q4_0', 'q4_0', 'kvarn4', 'fp8', '1024', '512'),
            'q8_0': ('q8_0', 'q8_0', 'q8_0', 'fp8', '1024', '512'),
            'f16': ('f16', 'f16', 'f16', 'f16', '1024', '256'),
        }
        for name, (rk, rv, upload, wire, batch, ubatch) in expected.items():
            with self.subTest(name=name):
                cmd = plan['commands'][name]
                def value(flag):
                    return cmd[cmd.index(flag) + 1]
                self.assertEqual((value('--cache-type-k'), value('--cache-type-v')), ('kvarn4', 'kvarn4'))
                self.assertEqual((value('--remote-attn-cache-type-k'), value('--remote-attn-cache-type-v')), (rk, rv))
                self.assertEqual(value('--remote-attn-format'), upload)
                self.assertEqual(value('--remote-attn-wire-format'), wire)
                self.assertEqual((value('--batch-size'), value('--ubatch-size')), (batch, ubatch))
                self.assertIn('--no-warmup', cmd)
                self.assertFalse(any('.ini' in item or 'preset' in item for item in cmd))
        self.assertFalse(plan['protected']['tiers_or_ini_changed'])
        self.assertEqual(plan['protected']['context_cap'], 160000)

    def test_f16_is_serial_declared_with_fp8_off(self):
        plan = self.preview(['--formats', 'f16'])
        env = plan['environments']['f16']
        self.assertEqual(env['GGML_RKVA_ASYNC_LOCAL_REMOTE'], '0')
        self.assertEqual(env['GGML_RKVA_TCP_PREFETCH'], '0')
        self.assertEqual(env['GGML_RKVA_NATIVE_BULK'], '0')
        self.assertEqual(env['GGML_RKVA_CUDA_FP8'], '0')
        self.assertEqual(env['GGML_RKVA_RANGE_STORAGE'], 'f16')
        self.assertEqual(env['GGML_RKVA_NATIVE_TAIL_TOKENS'], '0')
        self.assertNotIn('GGML_RKVA_NATIVE_PRODUCER', env)
        self.assertIn('--remote-attn-prefill', plan['commands']['f16'])
        self.assertEqual(plan['maps']['f16']['changed_vs_kvar4'][0][:3], 'KV ')

    def test_q8_serial_alternative_declares_what_changed(self):
        plan = self.preview(['--formats', 'q8_0', '--q8-profile', 'serial', '--order', 'ABBA'])
        env = plan['environments']['q8_0']
        self.assertEqual(env['GGML_RKVA_ASYNC_LOCAL_REMOTE'], '0')
        self.assertEqual(env['GGML_RKVA_TCP_PREFETCH'], '0')
        self.assertEqual(env['GGML_RKVA_WAVEFRONT_TILE'], '0')
        self.assertEqual(plan['commands']['q8_0'][plan['commands']['q8_0'].index('--ubatch-size') + 1], '256')
        self.assertTrue(any('serial Q8_0 alternative' in item for item in plan['maps']['q8_0']['changed_vs_kvar4']))
        self.assertEqual(plan['sequence'], ['q8_0', 'q8_0'])

    def test_context_is_capped_and_p_is_not_silently_changed(self):
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(study.main(self.argv + ['--context', '160256', '--points', '130560']), 2)
        self.assertIn('160000', error.getvalue())
        plan = self.preview()
        self.assertEqual(plan['campaign']['p'], 1024)
        self.assertEqual(plan['campaign']['context'], 2048)
        self.assertEqual(plan['campaign']['points'], [1536])
        self.assertTrue(plan['campaign']['greedy'])

    def test_mask_pack_stays_default_unset_in_all_formats(self):
        plan = self.preview()
        for name in ('kvar4', 'q8_0', 'f16'):
            with self.subTest(name=name):
                env = plan['environments'][name]
                self.assertNotIn('GGML_CUDA_MASK_PACK', env)
                self.assertEqual(env['GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB'], '128')
                self.assertEqual(env['GGML_KVARN_WINDOW_CHUNK'], '4096')
        self.assertEqual(plan['campaign']['mask_pack'], 'DEFAULT/unset')
        self.assertTrue(any('serial' in item.lower() or 'synchronous' in item.lower()
                            for item in plan['maps']['f16']['changed_vs_kvar4']))

    def test_prefix_paths_are_format_and_context_owned(self):
        plan = self.preview()
        for name, item in plan['prefixes'].items():
            self.assertIn(f'/{name}/C2048/', item['path'])
            self.assertTrue(item['path'].endswith('P1024-B1024.slot'))
        self.assertEqual(len({item['path'] for item in plan['prefixes'].values()}), 3)

    def test_contracts_are_format_specific_and_never_kvar4_for_q8_f16(self):
        plan = self.preview()
        kvar = plan['contracts']['kvar4']
        self.assertEqual((kvar['native_kvarn_bits'], kvar['tail'], kvar['window']), (4, 128, 4096))
        self.assertNotIn('format', kvar)
        for name, descriptor in [('q8_0', 5), ('f16', None)]:
            with self.subTest(name=name):
                value = plan['contracts'][name]
                self.assertEqual(value['format'], name)
                self.assertEqual(value['local_prefix_format'], 'kvarn4')
                self.assertEqual((value['native_kvarn_bits'], value['tail'], value['window']), (4, 128, 4096))
                self.assertEqual(value['remote_kv']['descriptor'], descriptor)
                self.assertEqual(value['remote_kv']['tail'], 0)
                self.assertEqual(value['physical_prefix_ubatch'], 256)
        self.assertNotEqual(plan['contracts']['kvar4'], plan['contracts']['q8_0'])
        self.assertNotEqual(plan['contracts']['q8_0'], plan['contracts']['f16'])

    def test_measured_request_is_greedy_temperature_zero(self):
        args = self.validated(['--formats', 'f16'])
        for name in ('kvar4', 'q8_0', 'f16'):
            with self.subTest(name=name), study.Patched(study.profiles() [name]):
                body = study.H.request_body(list(range(8)), 256, 999, False)
                self.assertEqual(body['temperature'], 0.0)
                self.assertEqual(body['seed'], 1234)
                self.assertEqual(body['top_k'], 0)
                self.assertEqual(body['top_p'], 1.0)
                self.assertEqual(body['min_p'], 0.0)
                self.assertTrue(body['return_tokens'])


class SnapshotTests(Base):
    def make_kvar4_meta(self, frozen=True):
        path = self.root / 'kvar4.slot'
        path.write_bytes(b'ordinary-slot-SCKP')
        (self.root / 'kvar4.slot.logits').write_bytes(b'logits')
        extra = ['--formats', 'kvar4', '--kvar4-prefix-snapshot', str(path)]
        if frozen:
            extra += ['--frozen-prefix-batch', '512', '--batch-size', '1024']
        args = self.validated(extra)
        tokens = list(range(args.p))
        profile = study.profiles()['kvar4']
        arm = study.arm_args(args, profile, path, self.root / 'create')
        model = study.H.fingerprint(args.model)
        response = {'n_saved': args.p - 128, 'n_written': path.stat().st_size,
                    'timings': {'save_ms': 1.0}, 'id_slot': 0, 'filename': path.name}
        manifest = {'model': model, 'server': {'sha256': 's'}, 'source': {'head': 'h'},
                    'local_libraries': [], 'loaded_libraries': []}
        with study.Patched(profile):
            metadata = study.H.save_prefix_snapshot(arm, tokens, manifest, response)
        return args, tokens, model, metadata, path

    def test_frozen_b512_kvar4_contract_and_cross_format_rejection(self):
        args, tokens, model, metadata, path = self.make_kvar4_meta()
        self.assertEqual(metadata['contract']['batch'], 512)
        self.assertEqual(metadata['contract']['physical_prefix_ubatch'], 256)
        q8 = study.arm_args(args, study.profiles()['q8_0'], path, self.root / 'q8')
        with study.Patched(study.profiles()['q8_0']), self.assertRaises(ValueError):
            study.H.load_prefix_snapshot(q8, tokens, model)
        f16 = study.arm_args(args, study.profiles()['f16'], path, self.root / 'f16')
        with study.Patched(study.profiles()['f16']), self.assertRaises(ValueError):
            study.H.load_prefix_snapshot(f16, tokens, model)

    def test_missing_snapshot_requires_explicit_allow_new_prefix(self):
        args = self.validated(['--formats', 'kvar4'])
        with mock.patch.object(study.H, 'run', side_effect=AssertionError('child started')):
            with self.assertRaises(ValueError):
                study.run_campaign(args)
        self.assertFalse((self.root / 'out').exists())
        args = self.validated(['--formats', 'kvar4', '--allow-new-prefix'])
        with mock.patch.object(study.H, 'run', return_value=1) as run, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            study.run_campaign(args)
        self.assertTrue(run.called)
        self.assertTrue((self.root / 'out').exists())

    def test_execute_requires_a_repeated_order(self):
        with mock.patch.object(study.H, 'run', side_effect=AssertionError('child started')), \
             contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(study.main(self.argv + ['--order', 'sequential', '--execute']), 2)
        self.assertIn('ABBA or interleaved', error.getvalue())

    def test_documented_full_command_with_frozen_batch_previews_without_writes(self):
        snapshot = self.root / 'existing-kvar4-b512.slot'
        snapshot.write_bytes(b'fixture-slot')
        extra = ['--formats', 'kvar4,q8_0,f16', '--frozen-prefix-batch', '512',
                 '--kvar4-prefix-snapshot', str(snapshot), '--allow-new-prefix']
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(study.main(self.argv + extra), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['formats'], ['kvar4', 'q8_0', 'f16'])
        self.assertEqual(plan['prefixes']['kvar4']['path'], str(snapshot))
        self.assertTrue(plan['prefixes']['q8_0']['path'].endswith('P1024-B1024.slot'))
        self.assertTrue(plan['prefixes']['f16']['path'].endswith('P1024-B1024.slot'))
        self.assertEqual(plan['prefixes']['kvar4']['intent'], 'reuse')
        self.assertEqual(plan['prefixes']['q8_0']['intent'], 'create-with---allow-new-prefix')
        self.assertEqual(plan['prefixes']['f16']['intent'], 'create-with---allow-new-prefix')
        self.assertEqual(plan['campaign']['frozen_scope'],
                         'KVarN4 arm only; Q8_0/F16 use their own prefixes')
        self.assertFalse((self.root / 'out').exists())
        self.assertFalse((self.root / 'prefixes').exists())


class ProofTests(unittest.TestCase):
    def test_real_archived_engine_subset_passes(self):
        stats, gpu, log = proof_inputs('kvar4')
        value = proof('kvar4', copy.deepcopy(REAL_STATS), gpu, REAL_LOG)
        self.assertEqual(value['status'], 'PASS', value)
        self.assertEqual(value['engine_schema'], 'rkva-range-profile-v1')
        self.assertEqual(value['int8_calls'], 0)
        self.assertEqual(
            value['bucket'],
            'attention/kvarn4-prefix/optimized-kvarn4_mq2_fine/nq2-16/r1-256/warm')
        self.assertEqual(value['transport_ops'], [22])

    def test_all_three_schema_fixtures_pass_their_proofs(self):
        # Q8_0/F16 fixtures use the current C++ reporter schema; they are not
        # hardware evidence and must not be reported as a hardware PASS.
        for name in ('kvar4', 'q8_0', 'f16'):
            with self.subTest(name=name):
                self.assertEqual(proof(name)['status'], 'PASS')

    def test_q8_bulk_op22_with_zero_kvarn_calls_passes(self):
        stats, gpu, log = proof_inputs('q8_0')
        engine = gpu['engine']
        self.assertGreater(engine['bulk']['fp8_gpu_calls'], 0)
        self.assertEqual(gpu['transport']['ops'][0]['op'], 22)
        self.assertGreater(gpu['transport']['ops'][0]['calls'], 0)
        self.assertTrue(engine['kvarn4_multi_query']['mq2']['available'])
        self.assertTrue(engine['kvarn4_multi_query']['mq2_i8']['available'])
        self.assertEqual(engine['kvarn4_multi_query']['mq2']['calls'], 0)
        self.assertEqual(engine['kvarn4_multi_query']['mq4']['calls'], 0)
        self.assertEqual(engine['kvarn4_decode_rowlane']['calls'], 0)
        self.assertEqual(proof('q8_0', stats, gpu, log)['status'], 'PASS')

    def test_q8_rejects_kvarn_rows_and_multi_query_calls(self):
        for field in (('kvarn4_decode_rowlane', 'calls'),
                      ('kvarn4_multi_query', 'mq2', 'calls'),
                      ('kvarn4_multi_query', 'mq4', 'calls')):
            stats, gpu, log = proof_inputs('q8_0')
            node = gpu['engine']
            for key in field[:-1]:
                node = node[key]
            node[field[-1]] = 1
            with self.subTest(field=field):
                self.assertEqual(proof('q8_0', stats, gpu, log)['status'], 'FAIL')

    def test_root_level_old_profile_shape_is_rejected(self):
        stats, gpu, log = proof_inputs('kvar4')
        old = {'buckets': gpu['engine']['buckets'],
               'optimized_kernels': gpu['engine']['optimized_kernels'],
               'kvarn4_mq_i8': {'calls': 0},
               'kvarn4_decode_rowlane': gpu['engine']['kvarn4_decode_rowlane']}
        value = proof('kvar4', stats, old, log)
        self.assertNotEqual(value['status'], 'PASS')
        self.assertTrue(any('engine' in error for error in value['errors']))

    def test_missing_engine_or_transport_fails(self):
        stats, gpu, log = proof_inputs('kvar4')
        for broken in ({'transport': gpu['transport']}, {'engine': gpu['engine']}, {},
                       {'engine': {}, 'transport': {}}):
            with self.subTest(keys=list(broken)):
                value = proof('kvar4', stats, broken, log)
                self.assertNotEqual(value['status'], 'PASS')
        value = proof('kvar4', stats, {'engine': gpu['engine'],
                                       'transport': {'enabled': True, 'ops': []}}, log)
        self.assertNotEqual(value['status'], 'PASS')

    def test_int8_counter_is_nested_mq2_i8_and_must_be_exactly_zero(self):
        stats, gpu, log = proof_inputs('kvar4')
        del gpu['engine']['kvarn4_multi_query']['mq2_i8']
        self.assertEqual(proof('kvar4', stats, gpu, log)['status'], 'FAIL')
        stats, gpu, log = proof_inputs('kvar4')
        del gpu['engine']['kvarn4_multi_query']['mq2_i8']['calls']
        self.assertEqual(proof('kvar4', stats, gpu, log)['status'], 'FAIL')
        stats, gpu, log = proof_inputs('kvar4')
        gpu['engine']['kvarn4_multi_query']['mq2_i8']['calls'] = 1
        self.assertEqual(proof('kvar4', stats, gpu, log)['status'], 'FAIL')
        stats, gpu, log = proof_inputs('kvar4')
        del gpu['engine']['kvarn4_multi_query']['mq2_i8']
        gpu['kvarn4_mq_i8'] = {'calls': 0}  # old root shape must not rescue it
        self.assertEqual(proof('kvar4', stats, gpu, log)['status'], 'FAIL')

    def test_kvarn4_prefix_bucket_name_is_required(self):
        stats, gpu, log = proof_inputs('kvar4')
        gpu['engine']['buckets'] = {
            'attention/kvarn4/optimized-kvarn4_mq2_fine/nq2-16/r1-256/warm':
                {'samples': 512}}
        self.assertEqual(proof('kvar4', stats, gpu, log)['status'], 'FAIL')

    def test_missing_engine_counter_or_fp8_fallback_fails(self):
        stats, gpu, log = proof_inputs('kvar4')
        del gpu['engine']['bulk']['fp8_cpu_fallback_calls']
        self.assertEqual(proof('kvar4', stats, gpu, log)['status'], 'FAIL')
        stats, gpu, log = proof_inputs('kvar4')
        gpu['engine']['bulk']['fp8_cpu_fallback_calls'] = 1
        self.assertEqual(proof('kvar4', stats, gpu, log)['status'], 'FAIL')
        stats, gpu, log = proof_inputs('kvar4')
        del gpu['engine']['counters']['failed_calls']
        self.assertEqual(proof('kvar4', stats, gpu, log)['status'], 'FAIL')

    def test_missing_transport_bulk_counter_fails(self):
        stats, gpu, log = proof_inputs('kvar4')
        gpu['transport']['ops'] = []
        self.assertEqual(proof('kvar4', stats, gpu, log)['status'], 'FAIL')
        stats, gpu, log = proof_inputs('kvar4')
        del gpu['transport']['ops'][0]['request_bytes']
        self.assertEqual(proof('kvar4', stats, gpu, log)['status'], 'FAIL')
        stats, gpu, log = proof_inputs('kvar4')
        gpu['transport']['fp8_gpu_enabled'] = False
        self.assertEqual(proof('kvar4', stats, gpu, log)['status'], 'FAIL')

    def test_q8_route_command_source_is_required(self):
        stats, gpu, log = proof_inputs('q8_0')
        manifest = manifest_for('q8_0')
        index = manifest['command'].index('--remote-attn-cache-type-k')
        manifest['command'][index + 1] = 'q4_0'
        self.assertEqual(proof('q8_0', stats, gpu, log, manifest)['status'], 'FAIL')
        manifest = manifest_for('q8_0')
        index = manifest['command'].index('--remote-attn-format')
        manifest['command'][index + 1] = 'kvarn4'
        self.assertEqual(proof('q8_0', stats, gpu, log, manifest)['status'], 'FAIL')

    def test_mask_pack_and_guard_window_environment_are_audited(self):
        stats, gpu, log = proof_inputs('kvar4')
        self.assertEqual(proof('kvar4', stats, gpu, log)['status'], 'PASS')
        manifest = manifest_for('kvar4')
        manifest['environment']['GGML_CUDA_MASK_PACK'] = '0'
        self.assertEqual(proof('kvar4', stats, gpu, log, manifest)['status'], 'FAIL')
        manifest = manifest_for('kvar4')
        del manifest['environment']['GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB']
        self.assertEqual(proof('kvar4', stats, gpu, log, manifest)['status'], 'FAIL')
        manifest = manifest_for('kvar4')
        manifest['environment']['GGML_KVARN_WINDOW_CHUNK'] = '2048'
        self.assertEqual(proof('kvar4', stats, gpu, log, manifest)['status'], 'FAIL')

    def test_missing_command_or_environment_is_not_pass(self):
        stats, gpu, log = proof_inputs('kvar4')
        manifest = manifest_for('kvar4')
        del manifest['command']
        self.assertEqual(proof('kvar4', stats, gpu, log, manifest)['status'], 'FAIL')
        manifest = manifest_for('kvar4')
        del manifest['environment']
        self.assertEqual(proof('kvar4', stats, gpu, log, manifest)['status'], 'FAIL')

    def test_q8_sync_fallback_is_explicitly_unsupported(self):
        stats, gpu, log = proof_inputs('q8_0')
        log += ('RemoteAttn: native Q8_0 bulk is not advertised by the negotiated '
                'server; using the synchronous range path\n')
        value = proof('q8_0', stats, gpu, log)
        self.assertEqual(value['status'], 'UNSUPPORTED')
        self.assertTrue(value['unsupported'])

    def test_q8_reference_pso_is_unsupported_not_silent(self):
        stats, gpu, log = proof_inputs('q8_0')
        gpu['engine']['optimized_kernels']['q8_0']['available'] = False
        gpu['engine']['buckets'] = {
            'attention/q8_0/reference-f16_gqa_fine/nq1/r32k+/warm': {'samples': 512}}
        value = proof('q8_0', stats, gpu, log)
        self.assertEqual(value['status'], 'UNSUPPORTED')

    def test_f16_rejects_q8_expansion_and_requires_direct_line(self):
        stats, gpu, log = proof_inputs('f16')
        for bad in (log.replace('direct original K/V half bytes, no quantized intermediate', ''),
                    log + 'RemoteAttn: range source=q8_0 storage=f16; quantized-source '
                    'expansion does not recover precision\n'):
            with self.subTest(bad=bad[-40:]):
                self.assertEqual(proof('f16', stats, gpu, bad)['status'], 'FAIL')

    def test_same_format_native_sources_are_required(self):
        stats, gpu, log = proof_inputs('q8_0')
        stats['range_source_cache_type'] = 'kvarn4'
        self.assertEqual(proof('q8_0', stats, gpu, log)['status'], 'FAIL')
        stats, gpu, log = proof_inputs('f16')
        stats['range_source_cache_type'] = 'q8_0'
        self.assertEqual(proof('f16', stats, gpu, log)['status'], 'FAIL')
        stats, gpu, log = proof_inputs('kvar4')
        stats['range_source_cache_type'] = 'q8_0'
        self.assertEqual(proof('kvar4', stats, gpu, log)['status'], 'FAIL')

    def test_f16_requires_cuda_fp8_off_and_no_tcp_prefetch(self):
        for key, value in [('cuda_pack_calls', 1), ('tcp_prefetch_completed', 1)]:
            stats, gpu, log = proof_inputs('f16')
            stats[key] = value
            self.assertEqual(proof('f16', stats, gpu, log)['status'], 'FAIL')


class SummaryTests(Base):
    def test_pooled_decode_uses_255_steps_per_256_output_run(self):
        arms = [arm('kvar4', [1] * 256, 0), arm('kvar4', [1] * 256, 1)]
        value = study.campaign_summary(arms, 256)
        pooled = value['formats']['kvar4']['pooled_decode']['point-1536']
        self.assertEqual(pooled['steps'], 510)
        self.assertEqual(pooled['expected_steps_per_run'], 255)
        self.assertAlmostEqual(pooled['pooled_decode_tps'], 10.0)
        self.assertEqual(value['status'], 'PASS')

    def test_same_format_determinism_and_cross_format_quality_diagnostic(self):
        arms = []
        for name, token in [('kvar4', 1), ('q8_0', 2), ('f16', 3)]:
            arms += [arm(name, [token] * 256, 0), arm(name, [token] * 256, 1)]
        value = study.campaign_summary(arms, 256)
        self.assertEqual(value['status'], 'PASS')
        self.assertTrue(value['comparison']['accounting_congruent'])
        self.assertEqual(value['comparison']['quality']['q8_0']['points'][0]['first_divergence'], 0)
        self.assertFalse(value['comparison']['quality']['q8_0']['points'][0]['identical'])
        arms[3]['points'][0]['tokens'] = [9] * 256
        value = study.campaign_summary(arms, 256)
        self.assertEqual(value['status'], 'FAIL')
        self.assertFalse(value['formats']['q8_0']['deterministic'])

    def test_accounting_mismatch_blocks_comparison_before_tps_claims(self):
        arms = []
        for name in ('kvar4', 'q8_0', 'f16'):
            arms += [arm(name, [1] * 256, 0), arm(name, [1] * 256, 1)]
        arms[2]['points'][0]['cache_n'] = 1000
        arms[3]['points'][0]['cache_n'] = 1000
        value = study.campaign_summary(arms, 256)
        self.assertFalse(value['comparison']['accounting_congruent'])
        self.assertEqual(value['status'], 'FAIL')

    def test_unsupported_is_published_and_not_hidden_as_pass(self):
        arms = [arm('kvar4', [1] * 256, 0), arm('kvar4', [1] * 256, 1),
                arm('q8_0', [2] * 256, 2, status='UNSUPPORTED')]
        value = study.campaign_summary(arms, 256)
        self.assertEqual(value['status'], 'UNSUPPORTED')
        self.assertEqual(value['formats']['q8_0']['statuses'], ['UNSUPPORTED'])

    def test_fake_client_full_campaign_is_congruent(self):
        argv = self.argv + ['--formats', 'kvar4,q8_0,f16', '--order', 'ABBA',
                            '--allow-new-prefix', '--output', str(self.root / 'campaign'),
                            '--prefix-dir', str(self.root / 'campaign-prefixes')]
        args = study.validate(study.parser().parse_args(argv))

        def fake_run(arm, tokens, info):
            out = Path(arm.output)
            out.mkdir(parents=True)
            fmt = next(name for name in ('kvar4', 'q8_0', 'f16') if name in out.name)
            stats, gpu, log = proof_inputs(fmt)
            (out / 'server.log').write_text(
                log + 'remote_attn stats ' + json.dumps(stats) + '\n' +
                'remote_attn gpu_profile ' + json.dumps(gpu) + '\n')
            (out / 'warm.response.json').write_text(json.dumps(
                {'timings': {'cache_n': 0, 'prompt_n': args.p, 'predicted_n': 1}}))
            (out / 'point-1536.response.json').write_text(json.dumps(
                {'tokens': [1 if fmt == 'kvar4' else 2] * 256,
                 'timings': {'cache_n': args.p, 'prompt_n': 512, 'predicted_n': 256,
                             'predicted_ms': 25500.0}}))
            (out / 'results.json').write_text(json.dumps(
                {'status': 'passed', 'points': [{
                    'prompt_tokens': 1536, 'valid': True,
                    'response_timings': {'cache_n': args.p, 'prompt_n': 512,
                                         'predicted_n': 256, 'predicted_ms': 25500.0}}]}))
            manifest = manifest_for(fmt)
            manifest['prefix_cache'] = {'intent': 'create'}
            (out / 'manifest.json').write_text(json.dumps(manifest))
            return 0

        with mock.patch.object(study.H, 'run', side_effect=fake_run) as run, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(study.run_campaign(args), 0)
        self.assertEqual(run.call_count, 6)
        results = json.loads((self.root / 'campaign' / 'study-results.json').read_text())
        self.assertEqual(results['status'], 'PASS')
        self.assertTrue(results['comparison']['accounting_congruent'])
        self.assertEqual(results['formats']['kvar4']['pooled_decode']['point-1536']['steps'], 510)
        self.assertEqual(results['formats']['q8_0']['proof']['roles']['descriptor'], 5)


if __name__ == '__main__':
    unittest.main()
