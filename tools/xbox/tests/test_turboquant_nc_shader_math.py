#!/usr/bin/env python3
"""CPU-only reference for the experimental RKVA TurboQuant NC shader.

This verifies layout, rotation and split softmax math; it does not execute HLSL.
Pass the CPU codec shared library as the only argument.
"""

import ctypes
import importlib.util
import pathlib
import sys

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    print("SKIP: NumPy is required for the TurboQuant NC shader math oracle", file=sys.stderr)
    sys.exit(77)


ROOT = pathlib.Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("tqnc_reference", ROOT / "tests/test-turboquant-nc.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def decode_bytes(row):
    data = np.frombuffer(row[:128], dtype=np.uint8)
    idx = np.empty(256, dtype=np.uint8)
    idx[0::2] = data & 15
    idx[1::2] = data >> 4
    return idx


def shader_style(q, krows, vrows, begin, end):
    q_rot = q.astype(np.float64) @ ref.make_hadamard()
    states = []
    for wave in range(2):
        maximum = np.full(6, -np.inf)
        denom = np.zeros(6)
        acc = np.zeros((6, 256))
        for row0 in range(begin + wave*64, end, 128):
            stop = min(row0 + 64, end)
            scores, decoded_values = [], []
            for row in range(row0, stop):
                key = krows[row]
                centers = ref.CENTROIDS[decode_bytes(key)].astype(np.float64)
                norm_bits = int.from_bytes(key[128:130], "little")
                norm = ref.half_value(norm_bits)
                factor = norm / np.sqrt(np.dot(centers, centers) + 1e-16)
                scores.append((q_rot @ centers) * factor / 16.0)
                value = vrows[row]
                scale = ref.half_value(int.from_bytes(value[128:130], "little"))
                minimum = ref.half_value(int.from_bytes(value[130:132], "little"))
                decoded_values.append(decode_bytes(value).astype(np.float64)*scale + minimum)
            scores = np.asarray(scores)
            decoded_values = np.asarray(decoded_values)
            next_max = np.maximum(maximum, np.max(scores, axis=0))
            old = np.where(denom > 0, np.exp(maximum - next_max), 0.0)
            p = np.exp(scores - next_max)
            acc = acc*old[:, None] + p.T @ decoded_values
            denom = denom*old + np.sum(p, axis=0)
            maximum = next_max
        states.append((maximum, denom, acc))
    maximum = np.maximum(states[0][0], states[1][0])
    if not np.any(np.isfinite(maximum)):
        return np.zeros((6, 256)), np.full(6, -np.inf)
    denom = np.zeros(6)
    acc = np.zeros((6, 256))
    for m, d, a in states:
        valid = d > 0
        w = np.zeros(6)
        w[valid] = np.exp(m[valid] - maximum[valid])
        denom += d*w
        acc += a*w[:, None]
    return acc/denom[:, None], maximum + np.log(denom)


def main(path):
    lib = ctypes.CDLL(path)
    ptr = ctypes.POINTER(ctypes.c_float)
    lib.ggml_tqnc_quantize_key.argtypes = [ptr, ctypes.POINTER(ref.Key)]
    lib.ggml_tqnc_quantize_value.argtypes = [ptr, ctypes.POINTER(ref.Value)]
    lib.ggml_tqnc_dequantize_key.argtypes = [ctypes.POINTER(ref.Key), ptr]
    lib.ggml_tqnc_dequantize_value.argtypes = [ctypes.POINTER(ref.Value), ptr]
    for name in ("ggml_tqnc_quantize_key", "ggml_tqnc_quantize_value",
                 "ggml_tqnc_dequantize_key", "ggml_tqnc_dequantize_value"):
        getattr(lib, name).restype = ctypes.c_bool

    rng = np.random.default_rng(684219)
    q = rng.normal(size=(6, 256)).astype(np.float32)
    krows, vrows, decoded_k, decoded_v = [], [], [], []
    for _ in range(257):
        k = rng.normal(size=256).astype(np.float32)
        v = rng.normal(size=256).astype(np.float32)
        kb, vb = ref.Key(), ref.Value()
        assert lib.ggml_tqnc_quantize_key(k.ctypes.data_as(ptr), ctypes.byref(kb))
        assert lib.ggml_tqnc_quantize_value(v.ctypes.data_as(ptr), ctypes.byref(vb))
        kd, vd = np.empty(256, np.float32), np.empty(256, np.float32)
        assert lib.ggml_tqnc_dequantize_key(ctypes.byref(kb), kd.ctypes.data_as(ptr))
        assert lib.ggml_tqnc_dequantize_value(ctypes.byref(vb), vd.ctypes.data_as(ptr))
        krows.append(bytes(kb) + b"\x00\x00")  # shader's 132-byte K stride
        vrows.append(bytes(vb))
        decoded_k.append(kd)
        decoded_v.append(vd)

    decoded_k = np.asarray(decoded_k, dtype=np.float64)
    decoded_v = np.asarray(decoded_v, dtype=np.float64)
    max_o = max_lse = 0.0
    for begin, end in ((0, 0), (0, 1), (0, 9), (0, 64), (0, 65),
                       (0, 129), (0, 257), (17, 80), (64, 192), (192, 257)):
        got_o, got_lse = shader_style(q, krows, vrows, begin, end)
        if begin == end:
            assert np.all(got_o == 0) and np.all(np.isneginf(got_lse))
            continue
        scores = q.astype(np.float64) @ decoded_k[begin:end].T / 16.0
        mx = np.max(scores, axis=1)
        p = np.exp(scores - mx[:, None])
        den = p.sum(axis=1)
        expected_o = p @ decoded_v[begin:end] / den[:, None]
        expected_lse = mx + np.log(den)
        max_o = max(max_o, np.max(np.abs(expected_o - got_o)))
        max_lse = max(max_lse, np.max(np.abs(expected_lse - got_lse)))
    assert max_o < 2e-6 and max_lse < 2e-6
    print(f"TurboQuant NC shader math (CPU only): 10 row ranges; max O={max_o:.3g}, max LSE={max_lse:.3g}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} path/to/libturboquant-nc.so")
    main(sys.argv[1])
