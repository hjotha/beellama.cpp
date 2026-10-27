"""Raw Q8 FP8 query unpack layout oracle; actual shader needs Xbox tests."""
import numpy as np
import pytest


def raw_unpack(fp8, positions):
    x = np.asarray(fp8, dtype=np.uint8)
    if np.any((x & 0x7c) == 0x7c):
        raise ValueError('non-finite FP8')
    assert x.size % 4 == 0
    bits = x.astype(np.uint16) << 8
    return bits.view(np.float16).astype(np.float32), np.asarray(positions, dtype=np.int32)


def test_all_finite_fp8_bytes_preserved_without_rotation():
    b = np.array([x for x in range(256) if (x & 0x7c) != 0x7c], dtype=np.uint8)
    q, pos = raw_unpack(b, [102400, 102401, 102402])
    expected = (b.astype(np.uint16) << 8).view(np.float16).astype(np.float32)
    assert np.array_equal(q.view(np.uint32), expected.view(np.uint32))
    assert pos.tolist() == [102400, 102401, 102402]
    assert len(q.tobytes() + pos.tobytes()) == 4 * (len(b) + len(pos))


def test_nonfinite_fp8_fails_before_publication():
    for b in [0x7c, 0x7d, 0x7f, 0xfc, 0xff]:
        with pytest.raises(ValueError):
            raw_unpack([0, 1, 2, b], [102400])
