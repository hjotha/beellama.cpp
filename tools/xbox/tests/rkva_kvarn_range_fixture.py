#!/usr/bin/env python3
"""CPU layout/domain oracle using fixtures emitted by current Bee llama-kvarn.cpp.

This does not run a D3D12 shader and is not Xbox GPU qualification.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def wht(x):
    x = np.asarray(x, dtype=np.float32).copy().reshape(-1, 2, 128)
    for stride in (1, 2, 4, 8, 16, 32, 64):
        z = x.reshape(-1, 2, 128 // (stride * 2), 2, stride)
        a, b = z[..., 0, :].copy(), z[..., 1, :].copy()
        z[..., 0, :], z[..., 1, :] = a + b, a - b
    x *= np.float32(0.08838834764831845)
    a, b = x[:, 0].copy(), x[:, 1].copy()
    x[:, 0] = (a + b) * np.float32(0.7071067811865475)
    x[:, 1] = (a - b) * np.float32(0.7071067811865475)
    return x.reshape(-1, 256)


def decode(blob):
    q = np.frombuffer(blob[:8192], dtype=np.uint8)
    quants = np.empty(16384, dtype=np.float32)
    quants[0::2], quants[1::2] = q & 15, q >> 4
    axes = np.frombuffer(blob[8192:], dtype='<f2').astype(np.float32).reshape(3, 128)
    return ((quants.reshape(128, 128) * axes[0, :, None]) + axes[1, :, None]) * axes[2, None, :]


def run(root):
    raw = (root / 'bee-kvarn4-records.bin').read_bytes()
    reference = np.fromfile(root / 'bee-kvarn4-dequant.bin', dtype='<f4').reshape(2, 128, 128)
    kd, vd = decode(raw[:8960]), decode(raw[8960:])
    layout_error = float(np.max(np.abs(np.stack([kd, vd]) - reference)))
    assert layout_error == 0, layout_error
    wf = np.fromfile(root / 'bee-kvarn4-wht.bin', dtype='<f4').reshape(2, 256)
    wht_error = float(np.max(np.abs(wht(wf[0])[0] - wf[1])))
    assert wht_error == 0, wht_error
    rng = np.random.default_rng(260)
    maximum_out, maximum_lse, checks = 0., 0., 0
    for global_origin_group in (0,800):
        for tail_tokens in (0,128):
            for live in (1,127,128,129,255,256,257,383,384,385,511,512,513,641,769,1025):
                # Independent original-domain materialization of exactly the same records/stage/tail.
                krot = rng.normal(0, .4, (live, 256)).astype(np.float16).astype(np.float32)
                vrot = rng.normal(0, .4, (live, 256)).astype(np.float16).astype(np.float32)
                live_group = (live - 1) // 128
                for token in range(live):
                    g, pos = divmod(token, 128)
                    if (global_origin_group + g) != 0 and (g < live_group or (g == live_group and live % 128 == 0)):
                        krot[token] = np.tile(kd[:, pos], 2)
                        vrot[token] = np.tile(vd[pos], 2)
                exact_begin = max(0, live-tail_tokens)
                korig, vorig = wht(krot), wht(vrot)
                exact_k = rng.normal(0,.4,(live-exact_begin,256)).astype(np.float16).astype(np.float32)
                exact_v = rng.normal(0,.4,(live-exact_begin,256)).astype(np.float16).astype(np.float32)
                if tail_tokens:
                    korig[exact_begin:], vorig[exact_begin:] = exact_k, exact_v
                for limit in sorted(set((0,1,min(2,live),max(0,live-129),max(0,live-1),live))):
                    q = rng.normal(0,.4,256).astype(np.float32)
                    qr = wht(q)[0]
                    if limit == 0:
                        checks += 1
                        continue
                    scores_original = korig[:limit].astype(np.float64) @ q.astype(np.float64) / 16
                    max_original = scores_original.max()
                    probs = np.exp(scores_original-max_original)
                    expected = probs @ vorig[:limit].astype(np.float64) / probs.sum()
                    expected_lse = max_original+np.log(probs.sum())
                    # Shader domain split: rotated body score/value, original precision-tail score/value.
                    scores = np.empty(limit,np.float32)
                    n_body = min(limit,exact_begin)
                    scores[:n_body] = krot[:n_body] @ qr / np.float32(16)
                    scores[n_body:] = korig[n_body:limit] @ q / np.float32(16)
                    m = float(scores.max())
                    weights = np.exp(scores.astype(np.float64)-m)
                    rotated_acc = (weights[:n_body] @ vrot[:n_body].astype(np.float64)).astype(np.float32)
                    body = wht(rotated_acc)[0].astype(np.float64)
                    tail = weights[n_body:] @ vorig[n_body:limit].astype(np.float64)
                    actual = (body+tail)/weights.sum()
                    actual_lse = m+np.log(weights.sum())
                    maximum_out=max(maximum_out,float(np.max(np.abs(actual-expected))))
                    maximum_lse=max(maximum_lse,abs(actual_lse-expected_lse))
                    checks+=1
    assert maximum_out < 1e-5 and maximum_lse < 1e-5, (maximum_out,maximum_lse)
    return {'success':True,'cpu_only':True,'checks':checks,'numpy_version':np.__version__,'record_decode_max_abs':layout_error,
            'wht_vs_bee_max_abs':wht_error,'attention_domain_max_abs':maximum_out,
            'lse_domain_max_abs':maximum_lse,'gpu_qualification':'not run; live Xbox reserved for model benchmarks'}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('fixture_dir',type=Path)
    print(json.dumps(run(p.parse_args().fixture_dir),indent=2))
