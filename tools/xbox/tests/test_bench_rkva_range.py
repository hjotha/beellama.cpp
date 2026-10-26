"""Hardware-free benchmark qualification/provenance regressions.

Run: python3 -m unittest discover -s tools/xbox/tests -p test_bench_rkva_range.py
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'bench-rkva-range.py'
spec = importlib.util.spec_from_file_location('bench_rkva_range', SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def response(n=256):
    return {'timings': {'predicted_n': n, 'predicted_ms': (n - 1) * 100.,
                        'predicted_per_second': 10., 'prompt_n': 128, 'cache_n': 102272},
            'tokens_predicted': n, 'truncated': False}


def logs(n=256, rate=10):
    return [f'slot print_timing: id 0 | task 42 | prompt eval time = 1000.00 ms / 128 tokens (7.81 ms per token, 128.00 tokens per second)\n',
            f'slot print_timing: id 0 | task 42 |        eval time = {(n-1)*100.:.2f} ms / {n} tokens (100.00 ms per token, {rate:.2f} tokens per second)\n']


class TimingTests(unittest.TestCase):
    def test_hello_f16_requires_this_connections_native_fp8_confirmation(self):
        hello = 'RemoteAttn: connected to 192.168.1.27:50054, 16 remote layers, wire=f16, budget=4096 MiB'
        fp8 = 'RemoteAttn: EXPERIMENTAL native KVarN4 precision_tail=128 descriptor=3 serial_prefix=1 async_overlap=1 wire=fp8_e5m2'
        self.assertEqual(bench.classify_logs([hello, fp8])[0], [])
        for lines in ([hello], [hello, fp8.replace('wire=fp8_e5m2', 'wire=f16')],
                      [fp8, hello], [hello, fp8, hello], [hello, hello, fp8],
                      [hello, fp8, 'runtime wire=f16'], [hello, fp8, 'CUDA error: bad launch']):
            with self.subTest(lines=lines):
                self.assertTrue(bench.classify_logs(lines)[0])

    def test_no_f5_uses_log_and_api_not_constant_63000(self):
        result = bench.assess(response(), logs(), 256)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['authoritative_decode_tps'], 10)
        self.assertIsNone(result['f5_interval_tps'])
        self.assertFalse(result['f5_present'])
        self.assertEqual(result['server_prompt']['tokens'], 128)

    def test_270_outputs_derive_269_intervals(self):
        value = response(270)
        value['f5_per_token'] = {'tokens_recorded': 270, 'truncated': False,
                                'speculative_seen': False, 'token_ms': [100.] * 269}
        result = bench.assess(value, logs(270), 270)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['f5_interval_tps'], 10)
        self.assertEqual(result['authoritative_decode_tps'], 10)

    def test_extra_outputs_cannot_qualify_requested_256(self):
        result = bench.assess(response(270), logs(270), 256)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['authoritative_decode_tps'])

    def test_log_api_discrepancy_invalidates_rate(self):
        for lines in [logs(rate=4.33), logs(270), []]:
            with self.subTest(lines=lines):
                result = bench.assess(response(), lines, 256)
                self.assertFalse(result['valid'])
                self.assertIsNone(result['authoritative_decode_tps'])

    def test_old_formula_n_instead_of_n_minus_one_rejected(self):
        # A three-token sample exposes using n instead of n-1 despite rounding.
        value = response(3)
        value['timings']['predicted_per_second'] = 15
        self.assertFalse(bench.assess(value, logs(3, rate=15), 3)['valid'])

    def test_truncated_partial_or_speculative_f5_rejected(self):
        for f5 in [{'tokens_recorded': 256, 'token_ms': [100.] * 63},
                   {'tokens_recorded': 256, 'token_ms': [100.] * 255, 'truncated': True},
                   {'tokens_recorded': 256, 'token_ms': [100.] * 255, 'speculative_seen': True}]:
            with self.subTest(f5_keys=list(f5)):
                value = response()
                value['f5_per_token'] = f5
                self.assertFalse(bench.assess(value, logs(), 256)['valid'])

    def test_nonfinite_missing_and_abort_never_publish_tps(self):
        for change in [{'predicted_ms': float('nan')}, {'predicted_per_second': 0}]:
            value = response()
            value['timings'].update(change)
            self.assertFalse(bench.assess(value, logs(), 256)['valid'])
        for line in ['GGML_ABORT: bad state', 'CUDA error: out of memory',
                     'falling back to host', 'CUDA graph memory-pressure cleanup failed',
                     'cannot query CUDA graph headroom: device error']:
            result = bench.assess(response(), logs() + [line], 256)
            self.assertFalse(result['valid'])
            self.assertIsNone(result['authoritative_decode_tps'])

    def test_normal_graph_policy_is_recorded_without_disqualifying_timing(self):
        for warning in ['disabling CUDA graphs before capture (17.44 MiB free < 128.00 MiB headroom)',
                        'CUDA graphs remain disabled; VRAM recovery headroom is low',
                        'CUDA graphs re-enabled after VRAM headroom recovered',
                        'released 33 CUDA graph executables and 33 captures under VRAM pressure',
                        'CUDA graph instantiation ran out of memory; falling back to direct execution',
                        'disabling CUDA graphs due to unsupported node type']:
            with self.subTest(warning=warning):
                result = bench.assess(response(), logs() + [warning], 256)
                self.assertTrue(result['valid'], result)
                self.assertEqual(result['authoritative_decode_tps'], 10)
                self.assertEqual(result['memory_policy_warnings'], [warning])
                self.assertEqual(result['server_flags'], [])

    def test_prefix_cache_accounting_must_match_requested_length(self):
        self.assertTrue(bench.assess(response(), logs(), 256, prompt_length=102400)['valid'])
        self.assertFalse(bench.assess(response(), logs(), 256, prompt_length=130816)['valid'])
        value = response()
        value['timings']['cache_n'] = 0
        value['timings']['prompt_n'] = 102400
        self.assertFalse(bench.assess(value, logs(), 256, prompt_length=102400)['valid'])

    def test_n256_requires_explicit_cuda_2x256_not_cpu_or_ignored_flag(self):
        marker = 'N256 lookahead: activated backend=CUDA ubatches=2 ubatch_tokens=256 layers=64'
        value = response()
        value['timings']['prompt_n'] = 512
        good = bench.assess(value, logs() + [marker], 256, n256_lookahead=True)
        self.assertTrue(good['valid'], good)
        for candidate in ['', marker.replace('CUDA', 'CPU'), marker.replace('ubatches=2', 'ubatches=1'),
                          marker.replace('ubatch_tokens=256', 'ubatch_tokens=128'),
                          'N256 lookahead: fallback backend=serial (CPU prototype eligibility not met)']:
            with self.subTest(candidate=candidate):
                result = bench.assess(value, logs() + [candidate], 256, n256_lookahead=True)
                self.assertFalse(result['valid'])
                self.assertIsNone(result['authoritative_decode_tps'])
        proof = bench.n256_activation([marker])
        self.assertTrue(bench.assess(response(), logs(), 256, n256_lookahead=True, prior_n256=proof)['valid'])
        # A new full prefill cannot reuse a marker from an older request.
        self.assertFalse(bench.assess(value, logs(), 256, n256_lookahead=True, prior_n256=proof)['valid'])

    def test_wavefront_runtime_parent_and_ffn_shape_proof(self):
        marker = ('RKVA wavefront graph: tile=256 tiles=2 groups=16 logical_rpc=32 envelopes=32 '
                  'max_outstanding=2 parent_tokens=512 final_ffn_rows=256 masked=1')
        value = response()
        value['timings'].update(prompt_n=28416, cache_n=102400)
        def qualify(line):
            return bench.assess(value, logs() + ([line] if line else []), 256,
                                prompt_length=130816, wavefront_tile=256, split_p=102400)
        good = qualify(marker)
        self.assertTrue(good['valid'], good)
        self.assertEqual(good['uncached_overflow_tokens'], 28416)
        self.assertTrue(good['wavefront_activation']['rkva_parent512_tile256_confirmed'])
        self.assertTrue(bench.wavefront_campaign_qualified([good]))
        for invalid in ['', marker.replace('tile=256', 'tile=128'), marker.replace('tiles=2', 'tiles=4'),
                        marker.replace('parent_tokens=512', 'parent_tokens=256'),
                        marker.replace('final_ffn_rows=256', 'final_ffn_rows=512'),
                        'N256 lookahead: activated backend=CPU ubatches=2 ubatch_tokens=256 layers=32']:
            with self.subTest(invalid=invalid):
                bad = qualify(invalid)
                self.assertFalse(bad['valid'])
                self.assertIsNone(bad['authoritative_decode_tps'])
        mixed = bench.assess(value, logs() + [marker, marker.replace('final_ffn_rows=256', 'final_ffn_rows=512')],
                             256, prompt_length=130816, wavefront_tile=256, split_p=102400)
        self.assertFalse(mixed['valid'])
        # Attributes need not have a particular order in the core message.
        reordered = 'RKVA wavefront graph: final_ffn_rows=256 parent_tokens=512 tiles=2 tile=256 masked=0'
        self.assertTrue(qualify(reordered)['valid'])

    def test_wavefront_warm_and_short_cached_tail_are_not_fake_activation(self):
        prefix = 'RKVA wavefront prefix: requested_ubatch=512 effective_ubatch=256 reason=not_all_overflow'
        self.assertTrue(bench.prefix_geometry([prefix])['prefix_compute_n256_confirmed'])
        self.assertFalse(bench.prefix_geometry([])['prefix_compute_n256_confirmed'])
        self.assertFalse(bench.prefix_geometry([prefix.replace('effective_ubatch=256', 'effective_ubatch=512')])['prefix_compute_n256_confirmed'])
        self.assertFalse(bench.wavefront_activation([prefix])['rkva_parent512_tile256_confirmed'])
        # A cached point with no full overflow parent is a valid decode sample,
        # but cannot by itself qualify the campaign as an activated WF run.
        value = response()
        value['timings'].update(prompt_n=128, cache_n=130688)
        sample = bench.assess(value, logs(), 256, prompt_length=130816, wavefront_tile=256, split_p=102400)
        self.assertTrue(sample['valid'], sample)
        self.assertEqual(sample['uncached_overflow_tokens'], 128)
        self.assertFalse(sample['wavefront_activation']['rkva_parent512_tile256_confirmed'])
        self.assertFalse(bench.wavefront_campaign_qualified([sample]))
        sample['wavefront_activation']['rkva_parent512_tile256_confirmed'] = True
        self.assertFalse(bench.wavefront_campaign_qualified([sample]), 'a reserve marker cannot qualify a cached short tail')


    def test_wavefront128_geometry_cannot_qualify_wavefront256(self):
        marker = 'RKVA wavefront graph: tile=128 tiles=2 parent_tokens=256 final_ffn_rows=256 masked=0'
        value = response()
        value['timings'].update(prompt_n=28160, cache_n=102400)
        good = bench.assess(value, logs() + [marker], 256,
                            prompt_length=130560, wavefront_tile=128, split_p=102400)
        self.assertTrue(good['valid'], good)
        self.assertTrue(bench.wavefront_campaign_qualified([good]))
        self.assertFalse(good['wavefront_activation']['rkva_parent512_tile256_confirmed'])
        for bad_marker in ['', marker.replace('tile=128', 'tile=256'),
                           marker.replace('parent_tokens=256', 'parent_tokens=512'),
                           marker.replace('final_ffn_rows=256', 'final_ffn_rows=128')]:
            with self.subTest(marker=bad_marker):
                bad = bench.assess(value, logs() + [bad_marker], 256,
                                   prompt_length=130560, wavefront_tile=128, split_p=102400)
                self.assertFalse(bad['valid'])
        wrong = bench.assess(value, logs() + [marker], 256,
                             prompt_length=130560, wavefront_tile=256, split_p=102400)
        self.assertFalse(wrong['valid'])

    def test_four_tile_marker_requires_parent1024_and_two_outstanding(self):
        marker = 'RKVA wavefront graph: tile=256 tiles=4 parent_tokens=1024 final_ffn_rows=256 max_outstanding=2'
        value = response()
        value['timings'].update(prompt_n=28160, cache_n=102400)
        def assess(line):
            return bench.assess(value, logs() + ([line] if line else []), 256,
                                prompt_length=130560, wavefront_tile=256, split_p=102400, wavefront_tiles=4)
        good = assess(marker)
        self.assertTrue(good['valid'], good)
        self.assertTrue(good['wavefront_activation']['rkva_parent1024_tile256_confirmed'])
        self.assertFalse(good['wavefront_activation']['rkva_parent512_tile256_confirmed'])
        self.assertTrue(bench.wavefront_campaign_qualified([good]))
        for bad in ['', marker.replace('tiles=4', 'tiles=2'), marker.replace('parent_tokens=1024', 'parent_tokens=512'),
                    marker.replace('final_ffn_rows=256', 'final_ffn_rows=1024'),
                    marker.replace('max_outstanding=2', 'max_outstanding=4'), marker.replace(' max_outstanding=2', '')]:
            with self.subTest(marker=bad):
                self.assertFalse(assess(bad)['valid'])

    def test_four_tiles_cannot_qualify_a_short_tail(self):
        value = response()
        value['timings'].update(prompt_n=512, cache_n=130048)
        point = bench.assess(value, logs(), 256, prompt_length=130560,
                             wavefront_tile=256, split_p=102400, wavefront_tiles=4)
        self.assertTrue(point['valid'], point)
        self.assertFalse(bench.wavefront_campaign_qualified([point]))

    def test_prefetch_requires_completed_bounded_overlap(self):
        record = {'tcp_prefetch_admitted': 20, 'tcp_prefetch_sent': 20,
                  'tcp_prefetch_completed': 20, 'tcp_prefetch_ahead_sent': 9,
                  'tcp_prefetch_max_frames': 2}
        self.assertTrue(bench.tcp_prefetch_activation([json.dumps(record)])['two_frame_overlap_confirmed'])
        for key, value in [('tcp_prefetch_completed', 19), ('tcp_prefetch_ahead_sent', 0),
                           ('tcp_prefetch_max_frames', 3), ('tcp_prefetch_admitted', 0)]:
            with self.subTest(key=key):
                bad = dict(record, **{key: value})
                self.assertFalse(bench.tcp_prefetch_activation([json.dumps(bad)])['two_frame_overlap_confirmed'])
        self.assertFalse(bench.tcp_prefetch_activation([
            'RKVA TCP prefetch experimental: max_frames=2'])['two_frame_overlap_confirmed'])
        incomplete = dict(record, tcp_prefetch_admitted=22, tcp_prefetch_sent=22, tcp_prefetch_completed=21)
        self.assertFalse(bench.tcp_prefetch_activation([
            json.dumps(record), json.dumps(incomplete)])['two_frame_overlap_confirmed'])


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ['server', 'model']:
            (self.root / name).write_bytes(b'fixture-' + name.encode())
        (self.root / 'tokens.json').write_text(json.dumps(list(range(512))))
        self.arguments = ['--server-binary', str(self.root / 'server'), '--model', str(self.root / 'model'),
                          '--prompt-tokens', str(self.root / 'tokens.json'), '--output', str(self.root / 'out'),
                          '--xbox-host', '192.0.2.1:50054', '--threads', '4', '--p', '256',
                          '--context', '1024', '--points', '256,512', '--n-predict', '256']

    def tearDown(self):
        self.tmp.cleanup()

    def test_prompt_and_outputs_must_leave_one_context_slot(self):
        # The server marks context_limit/truncated before checking token_limit
        # when prompt + requested outputs reaches the context capacity.
        with self.assertRaises(ValueError):
            bench.validate(bench.parser().parse_args(self.arguments + ['--points', '768']))
        args = bench.validate(bench.parser().parse_args(self.arguments))
        self.assertLess(max(args.points) + args.n_predict, args.context)

    def test_dry_run_starts_no_child_or_network_and_writes_no_output(self):
        with mock.patch.object(bench.subprocess, 'Popen', side_effect=AssertionError('child started')), \
             mock.patch.object(bench.socket, 'socket', side_effect=AssertionError('socket opened')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(bench.main(self.arguments), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['mode'], 'dry-run')
        self.assertFalse(plan['starts_child'])
        self.assertFalse((self.root / 'out').exists())
        self.assertIn('--spec-type', plan['command_preview'])
        self.assertEqual(plan['pinned_environment']['GGML_RKVA_WAVEFRONT_TILE'], '0')

    def test_logical_batch512_off_and_prototype_preview_are_distinct(self):
        for enabled in [False, True]:
            extra = ['--batch-size', '512'] + (['--n256-lookahead'] if enabled else [])
            with self.subTest(enabled=enabled), contextlib.redirect_stdout(io.StringIO()) as output, \
                 mock.patch.object(bench.subprocess, 'Popen', side_effect=AssertionError('child started')):
                self.assertEqual(bench.main(self.arguments + extra), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['configuration']['batch'], 512)
            self.assertEqual(plan['configuration']['ubatch'], 256)
            self.assertEqual(plan['pinned_environment']['LLAMA_N256_LOOKAHEAD'], '1' if enabled else '0')
            self.assertFalse(plan['n256_capability']['production_qualified'])
            self.assertTrue(plan['n256_capability']['harness_execution_enabled'])
            self.assertFalse(plan['n256_capability']['runtime_activation_confirmed'])
            command = plan['command_preview']
            self.assertEqual(command[command.index('--batch-size') + 1], '512')
            self.assertEqual(command[command.index('--ubatch-size') + 1], '256')

    def test_n256_execute_dispatches_without_manual_approval_gate(self):
        with mock.patch.object(bench, 'run', return_value=1) as run, \
             mock.patch.object(bench.subprocess, 'Popen', side_effect=AssertionError('child started')), \
             mock.patch.object(bench.socket, 'socket', side_effect=AssertionError('socket opened')):
            self.assertEqual(bench.main(self.arguments + ['--batch-size', '512', '--n256-lookahead', '--execute']), 1)
        self.assertTrue(run.call_args.args[0].n256_lookahead)
        self.assertEqual(run.call_args.args[0].batch, 512)
        self.assertFalse((self.root / 'out').exists())

    def test_batch_choices_and_physical_n256_are_fixed(self):
        for extra in [['--batch-size', '128'], ['--batch-size', '2048'], ['--ubatch', '128']]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                bench.parser().parse_args(self.arguments + extra)
            self.assertEqual(error.exception.code, 2)
        args = bench.validate(bench.parser().parse_args(self.arguments + ['--batch-size', '512']))
        lines = ['n_batch = 512\n', 'n_ubatch = 256\n',
                 'KVarN position-split: P=256 C=1024 local=256 overflow=768\n',
                 'EXPERIMENTAL native KVarN4 precision_tail=128 descriptor=3 async_overlap=1 wire=fp8_e5m2\n']
        self.assertEqual(bench.validate_startup(args, lines), [])
        with self.assertRaises(RuntimeError):
            bench.validate_startup(args, [line.replace('n_batch = 512', 'n_batch = 256') for line in lines])

    def test_wavefront256_campaign_is_distinct_from_cpu_prototype(self):
        args = self.arguments + ['--batch-size', '512', '--ubatch', '512', '--wavefront-tile', '256']
        with contextlib.redirect_stdout(io.StringIO()) as output, \
             mock.patch.object(bench.subprocess, 'Popen', side_effect=AssertionError('child started')):
            self.assertEqual(bench.main(args), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['pinned_environment']['GGML_RKVA_WAVEFRONT_TILE'], '256')
        self.assertEqual(plan['pinned_environment']['GGML_RKVA_WAVEFRONT_RPC_BATCH'], '1')
        self.assertEqual(plan['pinned_environment']['LLAMA_N256_LOOKAHEAD'], '0')
        self.assertEqual(plan['wavefront_campaign']['requested_parent_ubatch'], 512)
        self.assertEqual(plan['wavefront_campaign']['expected_compute_tile_tokens'], 256)
        self.assertEqual(plan['wavefront_campaign']['expected_prefix_compute_ubatch'], 256)
        self.assertFalse(plan['wavefront_campaign']['cpu_n256_prototype'])
        self.assertFalse(plan['wavefront_campaign']['production_qualified'])
        for extra in [['--wavefront-tile', '256'], ['--batch-size', '512', '--wavefront-tile', '256'],
                      ['--batch-size', '512', '--ubatch', '512'],
                      ['--batch-size', '512', '--ubatch', '512', '--wavefront-tile', '256', '--n256-lookahead']]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(bench.main(self.arguments + extra), 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            bench.parser().parse_args(self.arguments + ['--wavefront-tile', '64'])
        self.assertEqual(error.exception.code, 2)

    def test_wavefront128_plan_uses_parent256_and_same_prefix_contract(self):
        off = bench.validate(bench.parser().parse_args(self.arguments + ['--batch-size', '512']))
        on = bench.validate(bench.parser().parse_args(self.arguments + ['--batch-size', '512', '--wavefront-tile', '128']))
        self.assertEqual(bench.pinned_environment(on)['GGML_RKVA_WAVEFRONT_TILE'], '128')
        campaign = bench.wavefront_campaign(on)
        self.assertEqual(campaign['expected_compute_tile_tokens'], 128)
        self.assertEqual(campaign['expected_overflow_parent_tokens'], 256)
        self.assertEqual(campaign['expected_prefix_compute_ubatch'], 256)
        self.assertEqual(bench.prefix_contract(off, [1] * off.p, {'sha256': 'model'}),
                         bench.prefix_contract(on, [1] * on.p, {'sha256': 'model'}))
        with self.assertRaises(ValueError):
            bench.validate(bench.parser().parse_args(self.arguments + ['--batch-size', '512', '--ubatch', '512', '--wavefront-tile', '128']))

    def test_two_four_comparison_holds_logical_batch_fixed(self):
        common = self.arguments + ['--batch-size', '1024', '--wavefront-tile', '256']
        control = bench.validate(bench.parser().parse_args(common + ['--ubatch', '512']))
        candidate = bench.validate(bench.parser().parse_args(common + ['--ubatch', '1024', '--wavefront-tiles', '4']))
        self.assertEqual(control.batch, candidate.batch)
        self.assertEqual(bench.wavefront_campaign(control)['expected_overflow_parent_tokens'], 512)
        self.assertEqual(bench.wavefront_campaign(candidate)['expected_overflow_parent_tokens'], 1024)
        self.assertEqual(bench.prefix_contract(control, [1] * control.p, {'sha256': 'model'}),
                         bench.prefix_contract(candidate, [1] * candidate.p, {'sha256': 'model'}))
        for extra in [['--ubatch', '1024'], ['--ubatch', '512', '--wavefront-tiles', '4']]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                bench.validate(bench.parser().parse_args(common + extra))

    def test_mapped_codec_mode_is_audited_but_local_prefix_is_reusable(self):
        off = bench.validate(bench.parser().parse_args(self.arguments))
        on = bench.validate(bench.parser().parse_args(self.arguments + ['--cuda-fp8', 'mapped']))
        self.assertEqual(bench.pinned_environment(off)['GGML_RKVA_CUDA_FP8'], 'prefill')
        self.assertEqual(bench.pinned_environment(on)['GGML_RKVA_CUDA_FP8'], 'mapped')
        self.assertEqual(bench.prefix_contract(off, [1] * off.p, {'sha256': 'model'}),
                         bench.prefix_contract(on, [1] * on.p, {'sha256': 'model'}))

    def test_prefetch_is_explicit_and_preserves_local_prefix_contract(self):
        common = self.arguments + ['--batch-size', '1024', '--ubatch', '512', '--wavefront-tile', '256']
        off = bench.validate(bench.parser().parse_args(common))
        on = bench.validate(bench.parser().parse_args(common + ['--tcp-prefetch']))
        self.assertEqual(bench.pinned_environment(off)['GGML_RKVA_TCP_PREFETCH'], '0')
        self.assertEqual(bench.pinned_environment(on)['GGML_RKVA_TCP_PREFETCH'], '1')
        self.assertEqual(bench.prefix_contract(off, [1] * off.p, {'sha256': 'model'}),
                         bench.prefix_contract(on, [1] * on.p, {'sha256': 'model'}))
        for extra in [self.arguments + ['--tcp-prefetch'], common + ['--tcp-prefetch', '--no-profile']]:
            with self.assertRaises(ValueError):
                bench.validate(bench.parser().parse_args(extra))

    def test_remote_kernel_variant_is_audited_and_does_not_change_local_prefix(self):
        off = bench.validate(bench.parser().parse_args(self.arguments))
        on = bench.validate(bench.parser().parse_args(self.arguments + ['--range-kernel', 'balanced']))
        self.assertEqual(bench.pinned_environment(off)['GGML_RKVA_RANGE_KERNEL'], 'fine')
        self.assertEqual(bench.pinned_environment(on)['GGML_RKVA_RANGE_KERNEL'], 'balanced')
        self.assertEqual(bench.prefix_contract(off, [1] * off.p, {'sha256': 'model'}),
                         bench.prefix_contract(on, [1] * on.p, {'sha256': 'model'}))

    def test_current_readiness_log_and_private_port_are_exact(self):
        from types import SimpleNamespace
        for address, expected in [('127.0.0.1:12345', True), ('127.0.0.1:123456', False), ('0.0.0.0:12345', False)]:
            with self.subTest(address=address), contextlib.redirect_stdout(io.StringIO()):
                process = SimpleNamespace(stdout=io.StringIO(f'srv main: listening on http://{address}\n'))
                drain = bench.LogDrain(process, self.root / 'log', 12345)
                self.assertTrue(drain.ready.wait(2))
                drain.thread.join(2)
                self.assertFalse(drain.thread.is_alive())
                self.assertEqual(drain.listening, expected)

    def test_repetition_explicit_and_hash_freezes_effective_stream(self):
        with self.assertRaises(ValueError):
            bench.effective_tokens(self.root / 'tokens.json', 768)
        tokens, info = bench.effective_tokens(self.root / 'tokens.json', 768, repeat=True)
        self.assertEqual(tokens, list(range(512)) + list(range(256)))
        self.assertEqual(info['copies'], 2)
        self.assertEqual(info['stream_sha256'], bench.token_hash(tokens))
        tokens[255] ^= 1
        self.assertNotEqual(info['stream_sha256'], bench.token_hash(tokens))

    def test_fingerprint_detects_binary_change(self):
        first = bench.fingerprint(self.root / 'server')
        (self.root / 'server').write_bytes(b'changed')
        second = bench.fingerprint(self.root / 'server')
        self.assertNotEqual(first['sha256'], second['sha256'])
        self.assertEqual(second['path'], str((self.root / 'server').resolve()))

    def test_invalid_geometry_and_host_are_refused_before_hardware(self):
        for extra in [['--p', '257'], ['--context', '256'], ['--points', '512,256'],
                      ['--points', '256,256'], ['--points', '1000'], ['--threads', '0'],
                      ['--n256-lookahead'], ['--xbox-host', 'localhost'],
                      ['--execute', '--dry-run']]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with mock.patch.object(bench.subprocess, 'Popen', side_effect=AssertionError('child started')):
                    self.assertEqual(bench.main(self.arguments + extra), 2)

    def test_token_types_reject_bool_negative_and_float(self):
        for tokens in [[True], [-1], [1.5], [], ['1']]:
            (self.root / 'tokens.json').write_text(json.dumps(tokens))
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                bench.effective_tokens(self.root / 'tokens.json', 1)

    def test_memory_guard_is_explicit_128_and_never_lowered(self):
        for inherited in [None, '18', '128']:
            with self.subTest(inherited=inherited), mock.patch.dict(os.environ, {}, clear=True):
                if inherited is not None:
                    os.environ[bench.GRAPH_GUARD_KEY] = inherited
                policy = bench.memory_policy()
                self.assertEqual(policy['effective_mib'], 128)
                self.assertEqual(policy['source_default_mib'], 18)
                self.assertEqual(policy['inherited_value'], inherited)
                self.assertIn('common.cuh', policy['source'])
                self.assertEqual(len(policy['source_sha256']), 64)
        for inherited in ['256', '-1', 'garbage']:
            with self.subTest(inherited=inherited), mock.patch.dict(os.environ, {bench.GRAPH_GUARD_KEY: inherited}):
                with self.assertRaises(ValueError):
                    bench.memory_policy()

    def test_startup_geometry_and_negotiation_match_current_source_format(self):
        args = bench.validate(bench.parser().parse_args(self.arguments))
        lines = ['llama_context: n_batch               = 256\n',
                 'llama_context: n_ubatch              = 256\n',
                 'KVarN position-split: P=256 C=1024 local=256 overflow=768 camadas full=16\n',
                 'RemoteAttn: EXPERIMENTAL native KVarN4 precision_tail=128 descriptor=3 serial_prefix=1 async_overlap=1 wire=fp8_e5m2; original F32 producer\n']
        warning = 'check_memory_pressure_before_graph: disabling CUDA graphs before capture (17.44 MiB free < 128.00 MiB headroom)\n'
        self.assertEqual(bench.validate_startup(args, lines + [warning]), [warning.strip()])
        for bad in [lines[:-1], [line.replace('C=1024', 'C=2048') for line in lines],
                    [line.replace('n_ubatch              = 256', 'n_ubatch = 128') for line in lines],
                    lines + ['CUDA graph memory-pressure cleanup failed']]:
            with self.subTest(lines=bad), self.assertRaises(RuntimeError):
                bench.validate_startup(args, bad)
        command = bench.server_command(args, 12345)
        arg_source = (bench.ROOT / 'common/arg.cpp').read_text()
        for flag in command:
            if flag.startswith('-'):
                self.assertIn('"' + flag + '"', arg_source)

    def test_production_window_is_preserved_for_both_arms_after_environment_cleanup(self):
        for tile, ubatch in ((0, 256), (256, 512)):
            args = bench.validate(bench.parser().parse_args(self.arguments + [
                '--batch-size', '512', '--ubatch', str(ubatch), '--wavefront-tile', str(tile)]))
            for inherited in (None, '65536'):
                with self.subTest(tile=tile, inherited=inherited), mock.patch.dict(os.environ, {}, clear=True):
                    if inherited is not None:
                        os.environ['GGML_KVARN_WINDOW_CHUNK'] = inherited
                    output = self.root / f'window-{tile}-{inherited}'
                    output.mkdir()
                    env, frozen = bench.environment(args, output)
                self.assertEqual(env['GGML_KVARN_WINDOW_CHUNK'], '4096')
                self.assertEqual(frozen['GGML_KVARN_WINDOW_CHUNK'], '4096')

    def test_inherited_backend_flags_are_cleared_and_not_mistaken_for_baseline(self):
        args = bench.validate(bench.parser().parse_args(self.arguments))
        with mock.patch.dict(os.environ, {'GGML_RKVA_WAVEFRONT_TILE': '128',
                                         'LLAMA_ARG_REMOTE_ATTN': 'wrong:1',
                                         'CUDA_LAUNCH_BLOCKING': '1', bench.GRAPH_GUARD_KEY: '18', 'API_SECRET': 'do-not-record'}):
            env, frozen = bench.environment(args, self.root)
        self.assertEqual(env['GGML_RKVA_WAVEFRONT_TILE'], '0')
        self.assertNotIn('LLAMA_ARG_REMOTE_ATTN', env)
        self.assertNotIn('CUDA_LAUNCH_BLOCKING', env)
        self.assertEqual(env[bench.GRAPH_GUARD_KEY], '128')
        self.assertEqual(frozen[bench.GRAPH_GUARD_KEY], '128')
        self.assertNotIn('API_SECRET', frozen)
        self.assertFalse(any('GUARD' in key or 'POWER' in key for key in frozen))



class PrefixCacheTests(unittest.TestCase):
    setUp = ProvenanceTests.setUp
    tearDown = ProvenanceTests.tearDown
    def cache_args(self, extra=()):
        return bench.validate(bench.parser().parse_args(self.arguments + [
            '--prefix-snapshot', str(self.root / 'frozen.slot')] + list(extra)))

    def make_cache(self, extra=()):
        args = self.cache_args(extra)
        path = Path(args.prefix_snapshot)
        path.write_bytes(b'ordinary-slot-with-SCKP')
        Path(str(path) + '.logits').write_bytes(b'logits')
        tokens = list(range(args.p))
        model = bench.fingerprint(args.model)
        manifest = {'model': model, 'server': {'sha256': 'old-binary'}, 'source': {'head': 'old'},
                    'local_libraries': [], 'loaded_libraries': []}
        response = {'n_saved': args.p - 128, 'n_written': path.stat().st_size,
                    'timings': {'save_ms': 1.25}, 'id_slot': 0, 'filename': path.name}
        metadata = bench.save_prefix_snapshot(args, tokens, manifest, response)
        return args, tokens, model, metadata

    def test_frozen_cache_binds_semantics_not_wavefront_build_variant(self):
        args, tokens, model, metadata = self.make_cache()
        self.assertEqual(metadata['actual_prefix_tokens'], args.p - 128)
        self.assertNotEqual(metadata['actual_prefix_sha256'], metadata['contract']['requested_prefix_sha256'])
        self.assertEqual(bench.load_prefix_snapshot(args, tokens, model), metadata)
        # A changed binary is not a cache compatibility key; origin remains auditable.
        (self.root / 'server').write_bytes(b'new-qualifying-build')
        self.assertEqual(bench.load_prefix_snapshot(args, tokens, model), metadata)
        args.batch = 512
        with self.assertRaises(ValueError):
            bench.load_prefix_snapshot(args, tokens, model)

        # Frozen B512 is shared between its OFF and WF256 physical-prefix256 arms.
        args.batch = 256
        args.ubatch = 512; args.wavefront_tile = 256
        self.assertEqual(bench.load_prefix_snapshot(args, tokens, model), metadata)
        for key, value in [('p', args.p + 256), ('context', args.context + 256)]:
            previous = getattr(args, key); setattr(args, key, value)
            with self.assertRaises(ValueError):
                bench.load_prefix_snapshot(args, tokens, model)
            setattr(args, key, previous)
        with self.assertRaises(ValueError):
            bench.load_prefix_snapshot(args, tokens, {'sha256': 'other-model'})
        tokens[0] ^= 1
        with self.assertRaises(ValueError):
            bench.load_prefix_snapshot(args, tokens, model)


    def test_cross_batch_reuse_requires_explicit_frozen_source_batch(self):
        args, tokens, model, metadata = self.make_cache(['--batch-size', '512'])
        control = self.cache_args(['--batch-size', '1024', '--ubatch', '512', '--wavefront-tile', '256'])
        with self.assertRaises(ValueError):
            bench.load_prefix_snapshot(control, tokens, model)
        control = self.cache_args(['--batch-size', '1024', '--ubatch', '512', '--wavefront-tile', '256', '--frozen-prefix-batch', '512'])
        candidate = self.cache_args(['--batch-size', '1024', '--ubatch', '1024', '--wavefront-tile', '256', '--wavefront-tiles', '4', '--frozen-prefix-batch', '512'])
        self.assertEqual(bench.load_prefix_snapshot(control, tokens, model), metadata)
        self.assertEqual(bench.load_prefix_snapshot(candidate, tokens, model), metadata)
        self.assertEqual(metadata['contract']['batch'], 512)
        self.assertEqual(candidate.batch, 1024)

    def test_dangling_snapshot_symlink_is_rejected_before_fresh_save(self):
        args = self.cache_args()
        path = Path(args.prefix_snapshot)
        target = self.root / 'outside-state-must-not-be-created'
        path.symlink_to(target)
        self.assertFalse(path.exists())
        self.assertTrue(path.is_symlink())
        with self.assertRaisesRegex(ValueError, 'symlink'):
            bench.load_prefix_snapshot(args, list(range(args.p)), bench.fingerprint(args.model))
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.cache_args()
        self.assertFalse(target.exists())

    def test_cache_corruption_missing_and_unexpected_sidecars_fail_closed(self):
        args, tokens, model, metadata = self.make_cache()
        path = Path(args.prefix_snapshot)
        companion = Path(str(path) + '.logits')
        companion.write_bytes(b'different')
        with self.assertRaises(ValueError):
            bench.load_prefix_snapshot(args, tokens, model)
        companion.unlink()
        with self.assertRaises(ValueError):
            bench.load_prefix_snapshot(args, tokens, model)
        companion.write_bytes(b'logits')
        unexpected = Path(str(path) + '.partial')
        unexpected.write_bytes(b'stale')
        with self.assertRaises(ValueError):
            bench.load_prefix_snapshot(args, tokens, model)
        unexpected.unlink(); path.unlink()
        with self.assertRaises(ValueError):
            bench.load_prefix_snapshot(args, tokens, model)

    def test_restore_replay_rejects_cold_prefill_and_bad_accounting(self):
        for cache, prompt in [(102400, 0), (102272, 128), (102399, 1)]:
            value = {'timings': {'cache_n': cache, 'prompt_n': prompt, 'predicted_n': 1}}
            self.assertEqual(bench.validate_bounded_warm(value, 102400)['prompt_n'], prompt)
        for cache, prompt, predicted in [(0, 102400, 1), (102271, 129, 1), (102272, 127, 1),
                                         (102400, 0, 2), (True, 102399, 1)]:
            with self.subTest(cache=cache, prompt=prompt), self.assertRaises(RuntimeError):
                bench.validate_bounded_warm({'timings': {'cache_n': cache, 'prompt_n': prompt,
                                                       'predicted_n': predicted}}, 102400)
        with self.assertRaises(RuntimeError):
            bench.validate_bounded_warm({'truncated': True, 'timings': {'cache_n': 102400, 'prompt_n': 0,
                                                                       'predicted_n': 1}}, 102400)

    def test_slot_api_requires_exact_counts_and_preserves_basename(self):
        args = self.cache_args()
        value = {'id_slot': 0, 'filename': 'frozen.slot', 'n_restored': args.p - 128, 'n_read': 20,
                 'timings': {'restore_ms': 3.5}}
        with mock.patch.object(bench, 'http_json', return_value=value) as request:
            self.assertEqual(bench.slot_operation(123, 'restore', args), value)
            self.assertEqual(request.call_args.args[:3], (123, '/slots/0?action=restore', {'filename': 'frozen.slot'}))
        for changes in [{'n_restored': args.p - 256}, {'n_read': 0}, {'id_slot': 1},
                        {'timings': {'restore_ms': float('nan')}}, {'filename': '/wrong/path'}]:
            with self.subTest(changes=changes), mock.patch.object(bench, 'http_json', return_value=value | changes):
                with self.assertRaises(RuntimeError):
                    bench.slot_operation(123, 'restore', args)

    def test_prefix_only_real_control_flow_saves_before_any_overflow_and_no_tps(self):
        from types import SimpleNamespace
        args = self.cache_args(['--prefix-only', '--execute', '--batch-size', '512', '--ubatch', '512',
                                '--wavefront-tile', '256'])
        process = mock.Mock(pid=12345, returncode=0)
        process.poll.return_value = None
        drain = mock.Mock(lines=[])
        drain.ready.wait.return_value = True
        drain.listening = True
        drain.request_lines.return_value = []
        drain.thread.is_alive.return_value = False
        operations = []
        def request(port, endpoint, body=None, timeout=30):
            operations.append(endpoint)
            if endpoint == '/completion':
                self.assertEqual(len(body['prompt']), args.p)
                self.assertEqual(body['n_predict'], 1)
                return {'timings': {'predicted_n': 1, 'prompt_n': args.p, 'cache_n': 0}}
            if endpoint == '/slots/0?action=save':
                path = Path(args.prefix_snapshot)
                path.write_bytes(b'state-SCKP')
                return {'id_slot': 0, 'filename': path.name, 'n_saved': args.p - 128,
                        'n_written': path.stat().st_size, 'timings': {'save_ms': 2.0}}
            return {}
        def make_drain(child, path, port):
            path.write_text('')
            return drain
        source = {'head': 'frozen', 'tracked_diff_sha256': 'stable'}
        with mock.patch.object(bench, 'http_json', side_effect=request), \
                mock.patch.object(bench.subprocess, 'Popen', return_value=process), \
                mock.patch.object(bench.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='version')), \
                mock.patch.object(bench, 'LogDrain', side_effect=make_drain), \
                mock.patch.object(bench, 'git_identity', return_value=source), \
                mock.patch.object(bench, 'local_libraries', return_value=[]), \
                mock.patch.object(bench, 'loaded_libraries', return_value=[]), \
                mock.patch.object(bench, 'validate_startup', return_value=[]), \
                mock.patch.object(bench, 'prefix_geometry', return_value={'prefix_compute_n256_confirmed': True}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bench.run(args, list(range(args.p)), {}), 0)
        self.assertEqual(operations, ['/health', '/props', '/completion', '/slots/0?action=save'])
        results = json.loads((Path(args.output) / 'results.json').read_text())
        self.assertEqual(results['status'], 'passed')
        self.assertEqual(results['points'], [])
        self.assertIn('no TPS', results['purpose'])
        manifest = json.loads((Path(args.output) / 'manifest.json').read_text())
        self.assertFalse(manifest['wavefront_campaign']['runtime_activation_confirmed'])
        self.assertEqual(manifest['prefix_cache']['metadata']['actual_prefix_tokens'], args.p - 128)
        process.terminate.assert_called_once()

    def test_prefix_only_preview_creates_nothing_and_keeps_autosave_off(self):
        args = self.cache_args(['--prefix-only'])
        command = bench.server_command(args, 123)
        self.assertEqual(command[command.index('--slot-save-path') + 1], str(self.root) + os.sep)
        self.assertEqual(bench.pinned_environment(args)['GGML_RKVA_DISK_SNAPSHOTS'], '0')
        before = set(self.root.iterdir())
        with mock.patch.object(bench.subprocess, 'Popen', side_effect=AssertionError('child started')), \
                mock.patch.object(bench, 'http_json', side_effect=AssertionError('network')), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(bench.main(self.arguments + ['--prefix-snapshot', args.prefix_snapshot, '--prefix-only']), 0)
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertIn('prefix-cache-only; no TPS', output.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(bench.main(self.arguments + ['--prefix-only']), 2)

if __name__ == '__main__':
    unittest.main()
