#!/usr/bin/env python3
"""Source guards, not a substitute for Windows compilation/device testing."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
engine = (root / 'uwp/rkva/remote-attn-engine.cpp').read_text()
transport = (root / 'uwp/rkva/remote-attn.cpp').read_text()

def body(signature, source=engine):
    start = source.index('{', source.index(signature))
    depth = 1
    for i in range(start + 1, len(source)):
        depth += (source[i] == '{') - (source[i] == '}')
        if not depth:
            return source[start + 1:i]
    raise AssertionError(signature)

shared = body('void RemoteAttnEngine::create_shared_opt_pipelines()')
q4 = body('void RemoteAttnEngine::create_q4_opt_pipelines()')
assert 'CheckFeatureSupport' in shared and 'geo_.head_dim' in shared
assert 'range_attn_q4_opt_pso_' not in shared
assert 'range_reduce_opt_pso_.Reset()' in shared
assert 'range_reduce_opt_pso_.Reset()' not in q4
assert 'range_reduce_kvarn4_pso_.Reset()' not in q4
assert 'range_attn_q4_opt_pso_.Reset()' in q4
for signature in ('void RemoteAttnEngine::create_format_opt_pipelines()',
                  'void RemoteAttnEngine::create_kvarn4_mq_i8_pipeline()',
                  'void RemoteAttnEngine::create_q8_mq_i8_pipeline()',
                  'bool RemoteAttnEngine::set_range_profiling('):
    assert 'range_attn_q4_opt_pso_' not in body(signature)
    assert 'range_reduce_opt_pso_' in body(signature)
init = body('bool RemoteAttnEngine::initialize(')
assert init.index('create_shared_opt_pipelines()') < init.index('create_q4_opt_pipelines()') < init.index('create_format_opt_pipelines()')
row = body('void RemoteAttnEngine::create_decode_rowlane_pipeline()')
assert row.index('begin_attempt(') < row.index('CreateComputePipelineState')
for guard in ('unconfirmed_submission_', 'terminal_failure_', 'bulk_recording_',
              'wave_lane_min_ != 64', 'wave_lane_max_ != 64', 'RKVA_HAS_KVARN4_DECODE_ROWLANE_DXIL'):
    assert guard in row
assert 'init_timings_.record' in row and 'ready_' in row
setter = body('void RemoteAttnEngine::set_decode_rowlane(bool enabled)')
assert '.Reset()' not in setter
assert 'ready_ && enabled' in setter
assert transport.index('set_decode_rowlane(state.decode_rowlane_enabled)') < transport.index('initialize(g, &engine_error)')
assert transport.count('set_decode_rowlane(state.decode_rowlane_enabled)') == 1
fmt = body('void RemoteAttnEngine::create_format_opt_pipelines()')
assert 'if (rowlane_policy_.requested)' in fmt and 'deferred until enabled' in fmt
attend = body('bool RemoteAttnEngine::attend_range(')
assert 'create_decode_rowlane_pipeline' not in attend
assert 'decode_rowlane_ &&' in attend
assert 'range_attn_q4_opt_pso_ && range_reduce_opt_pso_' in attend
print('optional PSO control path, shared isolation and reference selection structure: PASS')
