"""Model of exact Q decomposition and the INT8 group-of-four score path.

Hardware shader correctness and throughput are tested separately on Xbox.
"""
import math
import numpy as np


def decompose(q):
    q = np.asarray(q, dtype=np.float32)
    assert q.shape == (32,)
    m = float(np.max(np.abs(q)))
    if not np.isfinite(q).all() or (m and not 2**-16 <= m <= 57344):
        return None
    delta = np.float32(1 if m == 0 else 2 ** (math.floor(math.log2(m)) - 13))
    x = q / delta
    if not np.isfinite(x).all() or (np.abs(x) > 16383).any() or (np.trunc(x) != x).any():
        return None
    n = x.astype(np.int32)
    if not np.array_equal(n.astype(np.float32) * delta, q):
        return None
    hi = np.trunc(n / 128).astype(np.int32)
    lo = n - hi * 128
    assert ((-128 <= hi) & (hi <= 127) & (-128 <= lo) & (lo <= 127)).all()
    return delta, hi, lo


def test_all_finite_e5m2_codes_and_zeros():
    for v in range(256):
        if v & 0x7c == 0x7c:
            continue
        q = np.array([v << 8], dtype=np.uint16).view(np.float16)[0].astype(np.float32)
        result = decompose(np.full(32, q, dtype=np.float32))
        assert result is not None
        d, hi, lo = result
        assert np.array_equal((hi * 128 + lo).astype(np.float32) * d, np.full(32, q))


def test_unrepresentable_and_nonfinite_fallback():
    for pair in ((2**-16, 57344), (float('nan'), 1), (float('inf'), 0), (2**-20, 1)):
        assert decompose(np.array(list(pair) + [0]*30, dtype=np.float32)) is None


def test_signed_dot_four_exact_integer_and_reconstruction():
    rng = np.random.default_rng(2718)
    for _ in range(1000):
        q = rng.integers(-127, 128, 32).astype(np.float32) / 16
        result = decompose(q)
        assert result is not None
        delta, hi, lo = result
        k = rng.integers(-128, 128, 32).astype(np.int32)
        for j in range(0, 32, 4):
            h = int(np.dot(k[j:j+4], hi[j:j+4])); l = int(np.dot(k[j:j+4], lo[j:j+4]))
            n = h * 128 + l
            assert abs(n) <= 8388096
            assert np.float32(n) * delta == np.dot(k[j:j+4].astype(np.float32), q[j:j+4])


def test_cancel_sensitive_partial_tree():
    q = np.array([1.75]*12 + [2**-13] + [0]*3 + [-1.75]*12 + [0]*4, dtype=np.float32)
    k = np.array([127]*12 + [1] + [0]*3 + [127]*12 + [0]*4, dtype=np.int32)
    delta, hi, lo = decompose(q)
    z = np.float32(0)
    for j in range(0, 32, 4):
        n = int(np.dot(k[j:j+4], hi[j:j+4])) * 128 + int(np.dot(k[j:j+4], lo[j:j+4]))
        z = np.float32(z + np.float32(n) * delta)
    whole = np.float32(int(np.dot(k, hi)) * 128 + int(np.dot(k, lo))) * delta
    assert z == 0 and whole == np.float32(2**-13)
