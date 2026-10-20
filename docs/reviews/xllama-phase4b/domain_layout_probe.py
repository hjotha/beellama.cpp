import ctypes
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np

root = Path('/home/hjotha/xllama-qwen35-xbox')
sys.path.insert(0, str(root / 'tests'))
import rkva_range_kvarn4_opt_model as model
library = ctypes.CDLL('/tmp/xllama-phase4b-review/wht.so')
ptr = ctypes.POINTER(ctypes.c_float)
library.wht_cpu.argtypes = [ptr, ptr]
rng = np.random.default_rng(274)
source = np.concatenate([np.eye(256, dtype=np.float32), rng.normal(size=(256, 256)).astype(np.float32)])
output = np.empty_like(source)
for row in range(len(source)):
    library.wht_cpu(source[row].ctypes.data_as(ptr), output[row].ctypes.data_as(ptr))
expected = model.butterfly_wht(source)
assert np.array_equal(output.view(np.uint32), expected.view(np.uint32))
print('CPU butterfly vs shader float32 emulation: 512 vectors / 131072 components, bit-exact')

# Compare the newly introduced partial/reducer domains, independently of the
# repository oracle's old per-split WHT(body)+tail contract.
worst = 0.0
for splits in (2, 32, 64, 256):
    for kind in ('body', 'tail', 'mixed', 'cancellation'):
        body = rng.normal(size=(splits, 256)).astype(np.float32)
        tail = rng.normal(size=(splits, 256)).astype(np.float32)
        if kind == 'body': tail.fill(0)
        if kind == 'tail': body.fill(0)
        if kind == 'cancellation': tail = -model.butterfly_wht(body) + np.float32(1e-5) * tail
        weights = np.exp(rng.uniform(-12, 0, splits)).astype(np.float32)
        denoms = rng.uniform(1, 20, splits).astype(np.float32)
        den = np.sum(weights * denoms, dtype=np.float32)
        old_acc = np.zeros(256, dtype=np.float32)
        new_acc = np.zeros(256, dtype=np.float32)
        for i in range(splits):
            old_acc += (model.butterfly_wht(body[i]) + tail[i]) * weights[i]
            new_acc += (body[i] + model.butterfly_wht(tail[i])) * weights[i]
        old_out = old_acc / den
        new_out = model.butterfly_wht(new_acc / den)
        error = np.max(np.abs(old_out.astype(np.float64) - new_out))
        worst = max(worst, float(error))
        assert error < 2e-6
print(f'New rotated partial contract: 16 synthetic cases, max_abs_delta={worst:.9e}')

# A concrete counterexample to pairing original-domain partials with the new
# WHT reducer (legacy artifact mismatch).
e0 = np.eye(256, dtype=np.float32)[0]
wrong = model.butterfly_wht(e0)
assert not np.allclose(e0, wrong)
print(f'Legacy partial + new reducer: expected O[0]=1, got {wrong[0]:.9f}, max_abs_error={np.max(np.abs(e0-wrong)):.9f}')

engine = (root/'uwp/rkva/remote-attn-engine.cpp').read_text()
inc = (root/'uwp/rkva/rkva-kvarn-bulk-f16.inc').read_text()
unpack = (root/'shaders/rkva/rkva_range_bulk_f16.hlsl').read_text()
assert 'const uint64_t query_default_bytes = q_f32_bytes + pos_bytes;' in inc
assert 'const uint64_t query_bytes = q_f32_bytes + pos_bytes;' in inc
assert 'dst.Store((q_pairs * 4 + idx) * 4, pos_word);' in unpack
assert 'SetComputeRootShaderResourceView(4, q_addr + q_bytes);' in engine
for nq in (1, 2, 3, 4, 5, 8, 9):
    q_bytes = nq * 24 * 256 * 4
    declared = q_bytes + nq * 4
    # try_create_buffer uses aligned4(bytes).
    allocated = (declared + 3) // 4 * 4
    required = 2 * q_bytes + nq * 4
    assert 2 * q_bytes >= allocated
    print(f'nq={nq}: q_bytes={q_bytes}, bulk_buffer_width={allocated}, required={required}, first_new_position_store={2*q_bytes} (OOB)')

q = np.ones(256, dtype=np.float32)
q_rot = model.butterfly_wht(q)
wrong_position = q_rot.view(np.int32)[0]
assert wrong_position != 96256
print(f'CPU positions binding at old offset: Q_rot[0]={q_rot[0]:.9f} becomes int position={wrong_position}; actual position=96256')

paths = ['uwp/rkva/remote-attn-engine.cpp','uwp/rkva/rkva-kvarn-bulk-f16.inc','uwp/rkva/rkva-kvarn-range.h','shaders/rkva/rkva_range_attn_kvarn4_mq.hlsl','shaders/rkva/rkva_range_attn_kvarn4_opt.hlsl','shaders/rkva/rkva_range_bulk_f16.hlsl','shaders/rkva/rkva_range_reduce_opt.hlsl','tests/rkva_range_kvarn4_opt_model.py']
Path('/tmp/xllama-phase4b-review/sources.json').write_text(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}, indent=2)+'\n')
