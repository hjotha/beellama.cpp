#!/usr/bin/env python3
"""Static control-path checks; CPU policy tests exercise state transitions."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
engine = (root / 'uwp/rkva/remote-attn-engine.cpp').read_text()
transport = (root / 'uwp/rkva/remote-attn.cpp').read_text()


def body(signature):
    start = engine.index(signature)
    opening = engine.index('{', start)
    depth = 1
    for index in range(opening + 1, len(engine)):
        if engine[index] == '{':
            depth += 1
        elif engine[index] == '}':
            depth -= 1
            if depth == 0:
                return engine[opening + 1:index]
    raise AssertionError(f'unclosed function: {signature}')


fmt = body('void RemoteAttnEngine::create_format_opt_pipelines()')
assert 'kvarn4_mq_i8_policy_.during_initialize(lazy_int8_init_)' in fmt
assert 'q8_mq_i8_policy_.during_initialize(lazy_int8_init_)' in fmt
assert fmt.index('create_kvarn4_mq_i8_pipeline()') < fmt.index('create_q8_mq_i8_pipeline()')
for name, policy, factory in (
        ('set_mq_i8', 'kvarn4_mq_i8_policy_', 'create_kvarn4_mq_i8_pipeline()'),
        ('set_q8_mq_i8', 'q8_mq_i8_policy_', 'create_q8_mq_i8_pipeline()')):
    setter = body(f'void RemoteAttnEngine::{name}(bool enabled)')
    assert f'{policy}.set_requested(enabled, ready_, lazy_int8_init_)' in setter
    assert factory in setter and '.Reset()' not in setter
    assert 'unconfirmed_submission_' in setter and 'terminal_failure_' in setter
for factory, macro in (('create_kvarn4_mq_i8_pipeline', 'RKVA_HAS_KVARN4_MQ2_I8_DXIL'),
                       ('create_q8_mq_i8_pipeline', 'RKVA_HAS_Q8_MQ2_I8_DXIL')):
    implementation = body(f'void RemoteAttnEngine::{factory}()')
    assert 'if (' in implementation and '.attempted) return;' in implementation
    assert '.mark_attempted()' in implementation and f'#if {macro}' in implementation
    assert 'CreateComputePipelineState' in implementation and 'init_timings_.record' in implementation
    assert '!range_attn_fmt_opt_root_' in implementation and '!range_reduce_opt_pso_' in implementation
assert 'wave_lane_min_ < rp::kKvarn4MqWaveLanes' in body('void RemoteAttnEngine::create_kvarn4_mq_i8_pipeline()')
attend = body('bool RemoteAttnEngine::attend_range(')
assert 'create_kvarn4_mq_i8_pipeline' not in attend and 'create_q8_mq_i8_pipeline' not in attend
assert 'mq_i8_ &&' in attend and 'q8_mq_i8_ &&' in attend
assert 'rkva-lazy-int8-init.flag' in transport
assert transport.index('set_lazy_int8_init(lazy_int8)') < transport.index('initialize(g, &engine_error)')
assert transport.index('const auto timings = state.engine->init_timings()') > transport.index('set_q8_mq_i8(state.q8_mq_i8_enabled)')
print('lazy INT8 control path and HELLO telemetry structure: PASS')
