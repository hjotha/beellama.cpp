#!/usr/bin/env python3
"""CPU model of the gpupeak STREAM kernels (tools/xbox/shaders/gpupeak_stream.hlsl).

The model exists to pin the validation contract without a GPU:

  * CSWrite stores, in every 16-byte vector it owns, the word tuple
    ``(dtid.x + 1, gid.x + 1, n_vec, mode)``.
  * CSRead accumulates the XOR of all four words of every vector it owns and
    must publish that accumulator in a *per-thread* Sink slot.  The host folds
    the first ``groups_x * groups_y * 256`` slots and compares the result with
    ``gpupeak_pattern_xor()``.

The old shader stored the accumulator in ``Sink[flat_group]``, so 256 threads
raced on one plain (non-atomic) store.  The value left in that slot was the
accumulator of whichever thread won, not the XOR of the group.  For the shipped
256 MiB plan every thread owns 32 vectors, an even count, so every accumulator
happens to be zero and the race coincidentally looked harmless; any odd count
(or a different grid) exposes it.  This module models both mappings and proves
that only the per-thread fold reproduces the expected checksum.

Pure Python, no numpy, no socket.  Used by test_gpupeak_stream_model.py.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

MASK32 = 0xFFFFFFFF
THREADS_PER_GROUP = 256


def pattern_for(dtid_x: int, gid_x: int, n_vec: int, mode: int) -> Tuple[int, int, int, int]:
    """The uint4 CSWrite stores.  Every thread in a group shares gid_x."""
    return ((dtid_x + 1) & MASK32, (gid_x + 1) & MASK32, n_vec & MASK32, mode & MASK32)


def fold_pattern(pattern: Sequence[int]) -> int:
    """XOR of the four pattern words, i.e. the CSRead accumulator per vector."""
    out = 0
    for word in pattern:
        out ^= int(word) & MASK32
    return out


def stride_vectors(groups_x: int, groups_y: int) -> int:
    return int(groups_x) * int(groups_y) * THREADS_PER_GROUP


def vectors_per_thread(thread: int, n_vec: int, stride: int) -> int:
    if thread >= n_vec:
        return 0
    return (n_vec - 1 - thread) // stride + 1


def analytic_pattern_xor(n_vec: int, groups_x: int, groups_y: int, mode: int) -> int:
    """Mirror of the C++ gpupeak_pattern_xor() period folding.

    Kept independent of the shader model so the two implementations can be
    compared: the host's analytic helper must equal a direct walk over every
    vector.
    """
    if n_vec <= 0 or groups_x <= 0 or groups_y <= 0:
        return 0
    stride = stride_vectors(groups_x, groups_y)
    period = min(stride, n_vec)
    one_period = 0
    for flat in range(period):
        dtid_x = flat % (groups_x * THREADS_PER_GROUP)
        gid_x = dtid_x // THREADS_PER_GROUP
        one_period ^= (dtid_x + 1) & MASK32
        one_period ^= (gid_x + 1) & MASK32
        one_period ^= n_vec & MASK32
        one_period ^= mode & MASK32
    out = 0
    periods = n_vec // stride
    if periods & 1:
        out ^= one_period
    remainder = (n_vec // stride) * stride
    for flat in range(n_vec - remainder):
        dtid_x = flat % (groups_x * THREADS_PER_GROUP)
        gid_x = dtid_x // THREADS_PER_GROUP
        out ^= (dtid_x + 1) & MASK32
        out ^= (gid_x + 1) & MASK32
        out ^= n_vec & MASK32
        out ^= mode & MASK32
    return out


def naive_vector_xor(n_vec: int, groups_x: int, groups_y: int, mode: int) -> int:
    """Direct walk: XOR every pattern word of every vector, in dispatch order.

    Only for small shapes; the per-thread model covers the production size.
    """
    stride = stride_vectors(groups_x, groups_y)
    out = 0
    for vec in range(n_vec):
        thread = vec % stride
        dtid_x = thread % (groups_x * THREADS_PER_GROUP)
        gid_x = dtid_x // THREADS_PER_GROUP
        out ^= fold_pattern(pattern_for(dtid_x, gid_x, n_vec, mode))
    return out


def per_thread_sink(n_vec: int, groups_x: int, groups_y: int, mode: int) -> List[int]:
    """CSRead accumulator per flat thread (the corrected sink mapping)."""
    stride = stride_vectors(groups_x, groups_y)
    sink: List[int] = []
    for thread in range(stride):
        count = vectors_per_thread(thread, n_vec, stride)
        dtid_x = thread % (groups_x * THREADS_PER_GROUP)
        gid_x = dtid_x // THREADS_PER_GROUP
        acc = fold_pattern(pattern_for(dtid_x, gid_x, n_vec, mode))
        sink.append(acc if count & 1 else 0)
    return sink


def fixed_read_checksum(n_vec: int, groups_x: int, groups_y: int, mode: int = 0) -> int:
    """Host fold after the per-thread Sink fix: XOR of every slot."""
    out = 0
    for acc in per_thread_sink(n_vec, groups_x, groups_y, mode):
        out ^= acc
    return out


def group_slot_models(n_vec: int, groups_x: int, groups_y: int, mode: int = 0) -> Dict[str, int]:
    """Old Sink[group] race: one arbitrary thread wins each group's slot.

    The real result is nondeterministic.  ``first_lane`` and ``last_lane`` are
    two legal outcomes; if either equals the expected checksum it is a
    coincidence, not a proof of coverage.
    """
    sink = per_thread_sink(n_vec, groups_x, groups_y, mode)
    groups = int(groups_x) * int(groups_y)
    first = 0
    last = 0
    for group in range(groups):
        first ^= sink[group * THREADS_PER_GROUP]
        last ^= sink[group * THREADS_PER_GROUP + THREADS_PER_GROUP - 1]
    return {"first_lane": first, "last_lane": last}


def model_shape(n_vec: int, groups_x: int, groups_y: int, mode: int = 0) -> Dict[str, object]:
    expected = analytic_pattern_xor(n_vec, groups_x, groups_y, mode)
    fixed = fixed_read_checksum(n_vec, groups_x, groups_y, mode)
    groups = group_slot_models(n_vec, groups_x, groups_y, mode)
    stride = stride_vectors(groups_x, groups_y)
    return {
        "n_vec": n_vec,
        "groups": int(groups_x) * int(groups_y),
        "threads": stride,
        "vectors_per_thread": (
            vectors_per_thread(0, n_vec, stride) if n_vec else 0
        ),
        "expected": expected,
        "fixed_per_thread": fixed,
        "fixed_matches": fixed == expected,
        "old_group_first_lane": groups["first_lane"],
        "old_group_last_lane": groups["last_lane"],
        "old_group_race_can_match": (
            groups["first_lane"] == expected or groups["last_lane"] == expected
        ),
    }


def main() -> int:
    import json

    shapes = [
        (512, 2, 1),
        (1000, 2, 1),
        (4097, 2, 1),
        (16384, 4, 2),
        (65536, 2048, 1),
        (268435456 // 16, 2048, 1),
    ]
    report = [model_shape(*shape) for shape in shapes]
    print(json.dumps({"schema": "gpupeak-stream-model-v1", "shapes": report}, indent=1))
    return 0 if all(row["fixed_matches"] for row in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
