#!/usr/bin/env python3
"""CPU oracle for D=256 TurboQuant 4bit NC; pass the codec shared library path."""

import ctypes
import sys

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    print("SKIP: NumPy is required for the TurboQuant NC CPU oracle", file=sys.stderr)
    sys.exit(77)


D = 256
CENTROIDS = np.array([
    -1.7068263888e-01, -1.2927794456e-01, -1.0111761093e-01, -7.8516095877e-02,
    -5.8903016150e-02, -4.1054993868e-02, -2.4258611724e-02, -8.0267284065e-03,
     8.0267284065e-03,  2.4258611724e-02,  4.1054993868e-02,  5.8903016150e-02,
     7.8516095877e-02,  1.0111761093e-01,  1.2927794456e-01,  1.7068263888e-01,
], dtype=np.float32)


class Key(ctypes.Structure):
    _fields_ = [("qs", ctypes.c_uint8 * 128), ("norm", ctypes.c_uint16)]


class Value(ctypes.Structure):
    _fields_ = [("qs", ctypes.c_uint8 * 128), ("scale", ctypes.c_uint16), ("minimum", ctypes.c_uint16)]


def half_bits(value):
    return np.float16(value).view(np.uint16).item()


def half_value(bits):
    return np.array(bits, dtype=np.uint16).view(np.float16).astype(np.float32).item()


def packed(indices):
    return (indices[0::2] | (indices[1::2] << 4)).astype(np.uint8)


def make_hadamard():
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < D:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(D)


def key_reference(src, hadamard):
    # Independent dense FP64 oracle for vLLM's WHT + nearest-centroid rule.
    norm = np.linalg.norm(src.astype(np.float64))
    rotated = hadamard @ (src.astype(np.float64) / (norm + 1e-8))
    boundaries = (CENTROIDS[:-1].astype(np.float64) + CENTROIDS[1:].astype(np.float64)) / 2
    idx = np.searchsorted(boundaries, rotated, side="right").astype(np.uint8)
    c = CENTROIDS[idx].astype(np.float64)
    decoded = hadamard @ (c / np.sqrt(np.dot(c, c) + 1e-16) * np.float16(norm).astype(np.float64))
    return packed(idx), half_bits(norm), decoded, np.min(np.abs(rotated[:, None] - boundaries))


def value_reference(src):
    vmin = np.min(src)
    vmax = np.max(src)
    scale = max(np.float32((vmax - vmin) / np.float32(15)), np.float32(1e-8))
    idx = np.clip(np.floor((src - vmin) / scale + np.float32(0.5)), 0, 15).astype(np.uint8)
    decoded = idx.astype(np.float32) * np.float32(np.float16(scale)) + np.float32(np.float16(vmin))
    return packed(idx), half_bits(scale), half_bits(vmin), decoded


def main(path):
    assert ctypes.sizeof(Key) == 130 and ctypes.sizeof(Value) == 132
    lib = ctypes.CDLL(path)
    ptr = ctypes.POINTER(ctypes.c_float)
    lib.ggml_tqnc_quantize_key.argtypes = [ptr, ctypes.POINTER(Key)]
    lib.ggml_tqnc_quantize_value.argtypes = [ptr, ctypes.POINTER(Value)]
    lib.ggml_tqnc_dequantize_key.argtypes = [ctypes.POINTER(Key), ptr]
    lib.ggml_tqnc_dequantize_value.argtypes = [ctypes.POINTER(Value), ptr]
    for name in ("ggml_tqnc_quantize_key", "ggml_tqnc_quantize_value",
                 "ggml_tqnc_dequantize_key", "ggml_tqnc_dequantize_value"):
        getattr(lib, name).restype = ctypes.c_bool

    rng = np.random.default_rng(823716)
    hadamard = make_hadamard()
    checked = 0
    key_max = value_max = 0.0
    for _ in range(64):
        src = rng.normal(size=D).astype(np.float32)
        key_bytes, key_norm, key_decoded, margin = key_reference(src, hadamard)
        # CPU reduction and a dense reference may differ at a centroid midpoint.
        if margin < 2e-6:
            continue
        key, value = Key(), Value()
        assert lib.ggml_tqnc_quantize_key(src.ctypes.data_as(ptr), ctypes.byref(key))
        assert lib.ggml_tqnc_quantize_value(src.ctypes.data_as(ptr), ctypes.byref(value))
        assert bytes(key.qs) == key_bytes.tobytes()
        assert key.norm == key_norm
        value_bytes, scale, minimum, value_decoded = value_reference(src)
        assert bytes(value.qs) == value_bytes.tobytes()
        assert value.scale == scale and value.minimum == minimum
        actual_k, actual_v = np.empty(D, np.float32), np.empty(D, np.float32)
        assert lib.ggml_tqnc_dequantize_key(ctypes.byref(key), actual_k.ctypes.data_as(ptr))
        assert lib.ggml_tqnc_dequantize_value(ctypes.byref(value), actual_v.ctypes.data_as(ptr))
        key_max = max(key_max, np.max(np.abs(actual_k.astype(np.float64) - key_decoded)))
        value_max = max(value_max, np.max(np.abs(actual_v - value_decoded)))
        assert abs(np.linalg.norm(actual_k) - half_value(key.norm)) < 1e-4
        checked += 1
    assert checked >= 48
    assert key_max < 2e-5 and value_max < 2e-6

    # The generic ggml quantization test uses a structured cosine rather than
    # Gaussian rows. Check its codebook/BDR behavior against the independent
    # dense FP64 oracle as well; NumPy's float32 cos can differ by an ULP from
    # the C test's cosf, so each oracle is evaluated on its actual input.
    cosine = (np.float32(0.1) + np.float32(2.0) *
              np.cos(np.arange(4096, dtype=np.float32))).astype(np.float32)
    cosine_decoded = np.empty_like(cosine)
    cosine_expected = np.empty(cosine.size, dtype=np.float64)
    cosine_checked = cosine_midpoints = 0
    for start in range(0, cosine.size, D):
        src = cosine[start:start + D]
        key_bytes, key_norm, _, margin = key_reference(src, hadamard)
        key = Key()
        assert lib.ggml_tqnc_quantize_key(src.ctypes.data_as(ptr), ctypes.byref(key))
        assert key.norm == key_norm
        if margin >= 2e-6:
            assert bytes(key.qs) == key_bytes.tobytes()
            cosine_checked += 1
        else:
            # A centroid index can differ at a floating-point midpoint. The
            # decoder still has to agree with the independent dense inverse.
            cosine_midpoints += 1
        actual = cosine_decoded[start:start + D]
        assert lib.ggml_tqnc_dequantize_key(ctypes.byref(key), actual.ctypes.data_as(ptr))
        codes = np.frombuffer(bytes(key.qs), dtype=np.uint8)
        indices = np.empty(D, dtype=np.uint8)
        indices[0::2], indices[1::2] = codes & 15, codes >> 4
        centroids = CENTROIDS[indices].astype(np.float64)
        expected = hadamard @ (centroids / np.sqrt(np.dot(centroids, centroids) + 1e-16) * half_value(key.norm))
        cosine_expected[start:start + D] = expected
        assert np.max(np.abs(actual.astype(np.float64) - expected)) < 2e-5
    assert cosine_checked >= 12
    # Match array_rmse() in test-quantize-fns: sqrt(sum squared error)/n.
    cosine_metric = np.linalg.norm(cosine_decoded.astype(np.float64) - cosine.astype(np.float64)) / cosine.size
    oracle_metric = np.linalg.norm(cosine_expected - cosine.astype(np.float64)) / cosine.size
    assert cosine_metric < 0.009 and abs(cosine_metric - oracle_metric) < 1e-6

    boundaries = (CENTROIDS[:-1].astype(np.float64) + CENTROIDS[1:].astype(np.float64)) / 2
    near_k_changes = near_v_changes = 0
    for boundary in boundaries:
        for delta in (-1e-5, -1e-7, 0.0, 1e-7, 1e-5):
            rotated = np.zeros(D, dtype=np.float64)
            rotated[0] = boundary + delta
            while True:
                rest = rng.normal(size=D - 1)
                rest *= np.sqrt(1.0 - rotated[0]**2) / np.linalg.norm(rest)
                if np.min(np.abs(rest[:, None] - boundaries)) > 2e-6:
                    break
            rotated[1:] = rest
            src = (hadamard @ rotated).astype(np.float32)
            expected_bytes, _, _, margin = key_reference(src, hadamard)
            key = Key()
            assert lib.ggml_tqnc_quantize_key(src.ctypes.data_as(ptr), ctypes.byref(key))
            changed = bytes(key.qs) != expected_bytes.tobytes()
            if margin >= 2e-6:
                assert not changed
            else:
                near_k_changes += changed
    for boundary in range(15):
        for delta in (-1e-5, -1e-7, 0.0, 1e-7, 1e-5):
            src = np.full(D, -1.0, dtype=np.float32)
            src[0], src[1] = -1.0, 1.0
            src[2] = np.float32(-1.0 + (boundary + 0.5)*2.0/15.0 + delta)
            expected_bytes, scale, minimum, _ = value_reference(src)
            value = Value()
            assert lib.ggml_tqnc_quantize_value(src.ctypes.data_as(ptr), ctypes.byref(value))
            if bytes(value.qs) != expected_bytes.tobytes():
                assert abs(delta) < 2e-6
                near_v_changes += 1
            assert value.scale == scale and value.minimum == minimum

    zeros = np.zeros(D, dtype=np.float32)
    decoded = np.empty(D, dtype=np.float32)
    key = Key()
    assert lib.ggml_tqnc_quantize_key(zeros.ctypes.data_as(ptr), ctypes.byref(key))
    assert lib.ggml_tqnc_dequantize_key(ctypes.byref(key), decoded.ctypes.data_as(ptr))
    assert np.array_equal(zeros, decoded)
    impulse = zeros.copy()
    impulse[0] = 1.0
    assert lib.ggml_tqnc_quantize_key(impulse.ctypes.data_as(ptr), ctypes.byref(key))
    assert bytes(key.qs) == bytes([0xBB])*128 and key.norm == half_bits(1.0)
    assert lib.ggml_tqnc_dequantize_key(ctypes.byref(key), decoded.ctypes.data_as(ptr))
    assert np.max(np.abs(impulse - decoded)) < 2e-6
    const = np.full(D, 1.25, dtype=np.float32)
    value = Value()
    assert lib.ggml_tqnc_quantize_value(const.ctypes.data_as(ptr), ctypes.byref(value))
    assert lib.ggml_tqnc_dequantize_value(ctypes.byref(value), decoded.ctypes.data_as(ptr))
    assert np.array_equal(const, decoded)
    ramp = np.tile(np.arange(16, dtype=np.float32), 16)
    assert lib.ggml_tqnc_quantize_value(ramp.ctypes.data_as(ptr), ctypes.byref(value))
    assert bytes(value.qs) == bytes([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE])*16
    assert value.scale == half_bits(1.0) and value.minimum == half_bits(0.0)

    for bad in (np.nan, np.inf, -np.inf, 1e30):
        src = np.ones(D, dtype=np.float32)
        src[17] = bad
        key = Key()
        ctypes.memset(ctypes.byref(key), 0xA5, ctypes.sizeof(key))
        before = bytes(key)
        assert not lib.ggml_tqnc_quantize_key(src.ctypes.data_as(ptr), ctypes.byref(key))
        assert bytes(key) == before
        value = Value()
        ctypes.memset(ctypes.byref(value), 0xA5, ctypes.sizeof(value))
        before = bytes(value)
        assert not lib.ggml_tqnc_quantize_value(src.ctypes.data_as(ptr), ctypes.byref(value))
        assert bytes(value) == before
    key = Key()
    key.norm = half_bits(np.inf)
    decoded.fill(123.0)
    assert not lib.ggml_tqnc_dequantize_key(ctypes.byref(key), decoded.ctypes.data_as(ptr))
    assert np.all(decoded == 123.0)
    value = Value()
    value.scale = half_bits(-1.0)
    assert not lib.ggml_tqnc_dequantize_value(ctypes.byref(value), decoded.ctypes.data_as(ptr))
    assert np.all(decoded == 123.0)
    # FP16 scale rounding can make the reconstructed maximum overflow FP16.
    extreme = np.full(D, -65504.0, dtype=np.float32)
    extreme[0] = 65504.0
    ctypes.memset(ctypes.byref(value), 0xA5, ctypes.sizeof(value))
    before = bytes(value)
    assert not lib.ggml_tqnc_quantize_value(extreme.ctypes.data_as(ptr), ctypes.byref(value))
    assert bytes(value) == before
    print(f"TurboQuant NC: {checked} random rows, max key decode error={key_max:.3g}, "
          f"max value decode error={value_max:.3g}; centroid/V midpoint differences "
          f"near boundaries={near_k_changes}/{near_v_changes}; structured cosine "
          f"metric={cosine_metric:.6f}, oracle-checked rows={cosine_checked}, "
          f"midpoint rows={cosine_midpoints}; zero, constant and invalid cases passed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} path/to/libturboquant-nc.so")
    main(sys.argv[1])
